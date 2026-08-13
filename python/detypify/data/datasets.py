from __future__ import annotations

from functools import cache
from hashlib import blake2b
from json import dumps
from math import floor, isclose
from os import getpid
from typing import TYPE_CHECKING, TypedDict

from detypify.config import DETERMINISTIC_SPLIT_SEED, HF_RAW_DATASET_PATH, DataSetName
from detypify.data.paths import DEFAULT_DATA_PATHS, DataPaths
from detypify.data.rendering import rasterize_strokes
from detypify.data.symbols import get_tex_to_char, get_tex_typ_map_digest

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    import polars as pl


class RenderedSample(TypedDict):
    image: np.ndarray
    label: int


class RenderedDataset:
    """A map-style dataset that memory-maps Polars rows and rasterizes them on demand."""

    def __init__(self, ipc_path: str, sample_count: int, image_size: int) -> None:
        self.ipc_path = ipc_path
        self.sample_count = sample_count
        self.image_size = image_size
        self._frame: pl.DataFrame | None = None

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> RenderedSample:
        # Each DataLoader worker opens its own memory map on first access.
        if self._frame is None:
            import polars as pl

            self._frame = pl.read_ipc(self.ipc_path, memory_map=True, rechunk=False)
        strokes, label = self._frame.row(index)
        return {"image": rasterize_strokes(strokes, self.image_size), "label": label}

    def __getstate__(self) -> dict[str, object]:
        """Drop the process-local memory map when serializing a worker dataset."""
        state = self.__dict__.copy()
        state["_frame"] = None
        return state


def _dataset_name_values(dataset_names: tuple[DataSetName, ...]) -> list[str]:
    return [dataset_name.value for dataset_name in dataset_names]


@cache
def _load_raw_dataset_cached(dataset_names: tuple[DataSetName, ...], paths: DataPaths) -> pl.DataFrame:
    """Read the local raw Parquet file, downloading it only when absent."""
    import polars as pl

    if not paths.raw_converted_parquet.is_file():
        import fsspec

        # Publish the download atomically so interrupted or concurrent readers never see a partial Parquet file.
        paths.raw_converted_parquet.parent.mkdir(parents=True, exist_ok=True)
        temp_path = paths.raw_converted_parquet.with_name(f".{paths.raw_converted_parquet.name}.{getpid()}.tmp")
        try:
            fsspec.filesystem("hf").get_file(HF_RAW_DATASET_PATH, temp_path)
            temp_path.replace(paths.raw_converted_parquet)
        finally:
            temp_path.unlink(missing_ok=True)

    dataset = pl.read_parquet(paths.raw_converted_parquet)
    return dataset.filter(pl.col("source").is_in(_dataset_name_values(dataset_names)))


def load_raw_dataset(dataset_names: Sequence[DataSetName], paths: DataPaths) -> pl.DataFrame:
    """Load the raw Parquet dataset and filter by source."""
    return _load_raw_dataset_cached(tuple(dataset_names), paths)


@cache
def _map_raw_dataset_cached(
    dataset_names: tuple[DataSetName, ...],
    paths: DataPaths,
) -> tuple[pl.DataFrame, dict[str, set[str]]]:
    """Map raw LaTeX labels to Typst characters with Polars."""
    import polars as pl

    mapped = _load_raw_dataset_cached(dataset_names, paths).with_columns(
        pl.col("latex_label").replace_strict(get_tex_to_char(), default=None, return_dtype=pl.String).alias("label")
    )
    # Preserve missing labels for review before discarding samples that cannot be used for training.
    unmapped = {
        source: set(labels)
        for source, labels in mapped.filter(pl.col("label").is_null())
        .group_by("source")
        .agg(pl.col("latex_label").unique())
        .iter_rows()
    }
    mapped = mapped.filter(pl.col("label").is_not_null() & (pl.col("strokes").list.len() > 0))
    return mapped, unmapped


def map_raw_dataset(
    dataset_names: Sequence[DataSetName],
    *,
    paths: DataPaths = DEFAULT_DATA_PATHS,
) -> tuple[pl.DataFrame, dict[str, set[str]]]:
    """Map raw LaTeX labels to Typst characters."""
    return _map_raw_dataset_cached(tuple(dataset_names), paths)


def get_dataset_classes(
    dataset_names: Sequence[DataSetName],
    *,
    paths: DataPaths = DEFAULT_DATA_PATHS,
) -> list[str]:
    """Return the sorted classes in the mapped dataset."""
    mapped, _ = map_raw_dataset(dataset_names, paths=paths)
    return mapped.get_column("label").unique().sort().to_list()


def _allocate_split_counts(sample_count: int, split_ratio: tuple[float, float, float]) -> tuple[int, int, int]:
    """Round fractional split sizes while preserving the exact sample count."""
    exact_counts = [sample_count * ratio for ratio in split_ratio]
    counts = [floor(count) for count in exact_counts]
    remainder = sample_count - sum(counts)
    order = sorted(range(len(counts)), key=lambda index: exact_counts[index] - counts[index], reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _split_frame(
    dataset: pl.DataFrame,
    split_ratio: tuple[float, float, float],
) -> dict[str, pl.DataFrame]:
    """Shuffle and split each label partition independently."""
    import polars as pl

    split_frames: dict[str, list[pl.DataFrame]] = {"train": [], "test": [], "val": []}
    for partition in dataset.partition_by("label"):
        shuffled = partition.sample(fraction=1.0, shuffle=True, seed=DETERMINISTIC_SPLIT_SEED)
        train_count, test_count, val_count = _allocate_split_counts(len(shuffled), split_ratio)
        split_frames["train"].append(shuffled.head(train_count))
        split_frames["test"].append(shuffled.slice(train_count, test_count))
        split_frames["val"].append(shuffled.tail(val_count))

    empty = dataset.clear()
    return {name: pl.concat(frames) if frames else empty for name, frames in split_frames.items()}


def _validate_split_ratio(split_ratio: tuple[float, float, float]) -> None:
    if any(ratio < 0 for ratio in split_ratio) or not isclose(sum(split_ratio), 1.0):
        msg = "split_ratio must contain non-negative values that sum to 1"
        raise ValueError(msg)
    if split_ratio[1] == 0 or split_ratio[2] == 0:
        msg = "test and validation split ratios must be greater than 0"
        raise ValueError(msg)


def _split_cache_key(
    mapped: pl.DataFrame,
    dataset_names: Sequence[DataSetName],
    split_ratio: tuple[float, float, float],
    min_split_class_count: int,
) -> str:
    """Fingerprint both mapped content and every option that changes split membership."""
    row_hashes = mapped.hash_rows(seed=DETERMINISTIC_SPLIT_SEED)
    content_hash = blake2b(row_hashes.to_numpy().tobytes(), digest_size=16).hexdigest()
    payload = dumps(
        {
            "content_hash": content_hash,
            "dataset_names": _dataset_name_values(tuple(dataset_names)),
            "min_split_class_count": min_split_class_count,
            "sample_count": len(mapped),
            "split_ratio": split_ratio,
            "stage": "polars-vector-splits-v1",
            "tex_typ_map_digest": get_tex_typ_map_digest(),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return blake2b(payload.encode(), digest_size=16).hexdigest()


def _write_split_cache(frame: pl.DataFrame, path: Path) -> None:
    """Atomically materialize a split as an Arrow IPC file if it is not cached."""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{getpid()}.tmp")
    frame.write_ipc(temp_path)
    temp_path.replace(path)


def get_rendered_dataset_splits(
    dataset_names: Sequence[DataSetName],
    image_size: int,
    paths: DataPaths,
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    min_split_class_count: int | None = None,
) -> tuple[dict[str, RenderedDataset], list[str]]:
    """Build deterministic train/test/validation datasets from mapped Polars rows."""
    from math import ceil

    import polars as pl

    _validate_split_ratio(split_ratio)
    mapped, _ = map_raw_dataset(dataset_names, paths=paths)

    # Freeze the sorted character order into compact numeric targets shared by all splits.
    classes: list[str] = mapped.get_column("label").unique().sort().to_list()
    label_to_index = {label: index for index, label in enumerate(classes)}
    mapped = mapped.select(
        "strokes",
        pl.col("label").replace_strict(label_to_index, return_dtype=pl.UInt32),
    )

    _, test_ratio, val_ratio = split_ratio
    if min_split_class_count is None:
        min_split_class_count = max(2, ceil(1 / test_ratio), ceil(1 / val_ratio))
    cache_key = _split_cache_key(mapped, dataset_names, split_ratio, min_split_class_count)

    # Labels too small to reliably reach both evaluation splits remain training-only.
    label_counts = mapped.group_by("label").len()
    rare_labels = label_counts.filter(pl.col("len") < min_split_class_count).get_column("label")
    if 0 < len(rare_labels) < len(label_counts):
        rare = mapped.filter(pl.col("label").is_in(rare_labels))
        mapped = mapped.filter(~pl.col("label").is_in(rare_labels))
    else:
        rare = mapped.clear()

    splits = _split_frame(mapped, split_ratio)
    if not rare.is_empty():
        splits["train"] = pl.concat([splits["train"], rare])

    rendered = {}
    for name, frame in splits.items():
        # Rasterization stays lazy; only compact vector strokes and labels are cached here.
        ipc_path = paths.dataset_splits_dir / cache_key / f"{name}.arrow"
        _write_split_cache(frame, ipc_path)
        rendered[name] = RenderedDataset(str(ipc_path), len(frame), image_size)
    return rendered, classes
