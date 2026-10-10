from __future__ import annotations

from functools import cache
from hashlib import blake2b
from json import dumps
from math import floor, isclose
from os import getpid
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from detypify.config import DETERMINISTIC_SPLIT_SEED, DataSetName
from detypify.data.paths import DEFAULT_DATA_PATHS, DataPaths
from detypify.data.rendering import rasterize_strokes
from detypify.data.symbols import get_tex_to_char

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import polars as pl


class RenderedSample(TypedDict):
    image: np.ndarray
    label: int


SPLIT_ARRAYS = ("points", "stroke_offsets", "sample_offsets", "labels")
"""Files of a split cache directory. Strokes are stored CSR-style: ``points`` holds every (x, y) pair, stroke ``j``
is ``points[stroke_offsets[j]:stroke_offsets[j + 1]]`` and sample ``i`` owns strokes
``sample_offsets[i]:sample_offsets[i + 1]``."""


class RenderedDataset:
    """A map-style dataset over memory-mapped stroke arrays that rasterizes samples on demand.

    Every process maps the same ``.npy`` files, so DataLoader workers share the OS page cache instead of each holding
    a private copy of the split.
    """

    def __init__(self, directory: str, image_size: int) -> None:
        import numpy as np

        self.directory = directory
        self.image_size = image_size
        self.sample_count = len(np.load(Path(directory) / "labels.npy", mmap_mode="r"))
        self._arrays: dict[str, np.ndarray] | None = None

    def _open(self) -> dict[str, np.ndarray]:
        # Each process (main or DataLoader worker) maps the files on first access.
        if self._arrays is None:
            import numpy as np

            directory = Path(self.directory)
            self._arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in SPLIT_ARRAYS}
        return self._arrays

    def __len__(self) -> int:
        return self.sample_count

    def strokes(self, index: int) -> list[np.ndarray]:
        """Strokes of one sample as read-only ``(points, 2)`` float32 views into the memory map."""
        import numpy as np

        arrays = self._open()
        first, last = arrays["sample_offsets"][index], arrays["sample_offsets"][index + 1]
        offsets = arrays["stroke_offsets"][first : last + 1]
        points = arrays["points"][offsets[0] : offsets[-1]]
        return np.split(points, offsets[1:-1] - offsets[0])

    def __getitem__(self, index: int) -> RenderedSample:
        arrays = self._open()
        return {"image": rasterize_strokes(self.strokes(index), self.image_size), "label": int(arrays["labels"][index])}

    def __getstate__(self) -> dict[str, object]:
        """Never pickle the memory maps; each worker maps the files itself."""
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state


def _dataset_name_values(dataset_names: tuple[DataSetName, ...]) -> list[str]:
    return [dataset_name.value for dataset_name in dataset_names]


@cache
def _load_raw_dataset_cached(dataset_names: tuple[DataSetName, ...], paths: DataPaths) -> pl.DataFrame:
    """Read the locally converted raw Parquet file."""
    import polars as pl

    if not paths.data_parquet.is_file():
        msg = f"Raw dataset not found at {paths.data_parquet}; run python/data.py prepare first"
        raise FileNotFoundError(msg)

    dataset = pl.read_parquet(paths.data_parquet)
    return dataset.filter(pl.col("source").is_in(_dataset_name_values(dataset_names)))


def load_raw_dataset(dataset_names: Sequence[DataSetName], paths: DataPaths) -> pl.DataFrame:
    """Load the raw Parquet dataset and filter by source."""
    return _load_raw_dataset_cached(tuple(dataset_names), paths)


@cache
def _map_raw_dataset_cached(
    dataset_names: tuple[DataSetName, ...],
    paths: DataPaths,
) -> pl.DataFrame:
    """Map raw LaTeX labels to Typst characters with Polars."""
    import polars as pl

    mapped = _load_raw_dataset_cached(dataset_names, paths).with_columns(
        pl.col("latex_label").replace_strict(get_tex_to_char(), default=None, return_dtype=pl.String).alias("label")
    )
    return mapped.filter(pl.col("label").is_not_null() & (pl.col("strokes").list.len() > 0))


def map_raw_dataset(
    dataset_names: Sequence[DataSetName],
    *,
    paths: DataPaths = DEFAULT_DATA_PATHS,
) -> pl.DataFrame:
    """Map raw LaTeX labels to Typst characters."""
    return _map_raw_dataset_cached(tuple(dataset_names), paths)


def get_dataset_classes(
    dataset_names: Sequence[DataSetName],
    *,
    paths: DataPaths = DEFAULT_DATA_PATHS,
) -> list[str]:
    """Return the sorted classes in the mapped dataset."""
    mapped = map_raw_dataset(dataset_names, paths=paths)
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
            "stage": "numpy-mmap-splits-v1",
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return blake2b(payload.encode(), digest_size=16).hexdigest()


def _write_split_cache(frame: pl.DataFrame, directory: Path) -> None:
    """Atomically materialize a split as memory-mappable ``.npy`` arrays if it is not cached (see SPLIT_ARRAYS)."""
    import shutil

    import numpy as np
    import polars as pl

    if directory.is_dir():
        return
    strokes = frame.get_column("strokes")
    # Empty lists vanish instead of becoming null rows, so lengths and flattened values stay consistent.
    flat_strokes = strokes.explode(empty_as_null=False)  # one row per stroke
    flat_points = flat_strokes.explode(empty_as_null=False)  # one row per point
    coords = flat_points.cast(pl.List(pl.Float32)).explode(empty_as_null=False).to_numpy()
    if coords.size != 2 * len(flat_points):
        msg = "every stroke point must have exactly two coordinates"
        raise ValueError(msg)
    arrays = {
        "points": coords.reshape(-1, 2).astype(np.float32, copy=False),
        "stroke_offsets": np.concatenate([[0], np.cumsum(flat_strokes.list.len().to_numpy())]).astype(np.int64),
        "sample_offsets": np.concatenate([[0], np.cumsum(strokes.list.len().to_numpy())]).astype(np.int64),
        "labels": frame.get_column("label").to_numpy().astype(np.uint32),
    }

    temp_dir = directory.with_name(f".{directory.name}.{getpid()}.tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name, array in arrays.items():
            np.save(temp_dir / f"{name}.npy", np.ascontiguousarray(array))
        temp_dir.replace(directory)
    except OSError:
        # Another process may have published the same split first; its copy is identical.
        if not directory.is_dir():
            raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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
    mapped = map_raw_dataset(dataset_names, paths=paths)

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
        split_dir = paths.dataset_splits_dir / cache_key / name
        _write_split_cache(frame, split_dir)
        rendered[name] = RenderedDataset(str(split_dir), image_size)
    return rendered, classes
