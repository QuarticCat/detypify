from __future__ import annotations

from functools import cache
from hashlib import sha256
from json import dumps
from os import process_cpu_count
from typing import TYPE_CHECKING, Any, cast

from detypify.config import HF_DATASET_REPO, DataSetName
from detypify.data.paths import DEFAULT_DATA_PATHS, DataPaths
from detypify.data.raw_sources import collect_contrib_raw, collect_detexify_raw, collect_mathwriting_raw
from detypify.data.rendering import rasterize_strokes
from detypify.data.symbols import get_tex_to_char, get_tex_typ_map_digest

if TYPE_CHECKING:
    import polars as pl
    from datasets import Dataset, DatasetDict

DETERMINISTIC_SPLIT_SEED = 114514


def create_raw_dataset(dataset_names: list[DataSetName], paths: DataPaths = DEFAULT_DATA_PATHS) -> pl.DataFrame:
    """Create and upload the raw LaTeX-annotated dataset to Hugging Face."""
    import logging

    import polars as pl
    from datasets import Dataset, DatasetInfo, Features, List, Sequence, Value

    logger = logging.getLogger(__name__)
    logger.info("--- Creating Raw Dataset: %s ---", ",".join(dataset_names))

    lfs: list[pl.LazyFrame] = []
    for dataset_name in dataset_names:
        match dataset_name:
            case DataSetName.mathwriting:
                lfs.append(collect_mathwriting_raw(paths).with_columns(pl.lit(dataset_name.value).alias("source")))
            case DataSetName.detexify:
                lfs.append(collect_detexify_raw(paths).with_columns(pl.lit(dataset_name.value).alias("source")))
            case DataSetName.contrib:
                lfs.append(collect_contrib_raw(paths).with_columns(pl.lit(dataset_name.value).alias("source")))

    if not lfs:
        msg = "No valid datasets to process"
        raise ValueError(msg)

    df = (
        pl.concat(lfs)
        .collect()
        .rename({"symbol": "strokes"})
        .sample(fraction=1.0, shuffle=True, seed=DETERMINISTIC_SPLIT_SEED)
    )

    features: Features = Features(
        {
            "latex_label": Value("string"),
            "strokes": List(List(Sequence(Value("float32"), length=2))),
            "source": Value("string"),
        }
    )
    dataset_info = DatasetInfo(
        description="Raw detypify dataset with original LaTeX labels and vector strokes.",
        features=features,
    )
    dataset = Dataset.from_polars(df, info=dataset_info)

    logger.info("  -> Uploading raw dataset to %s...", HF_DATASET_REPO)
    dataset.push_to_hub(repo_id=HF_DATASET_REPO, config_name="raw", split="data", num_proc=process_cpu_count() or 1)

    paths.raw_dataset_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(paths.raw_dataset_parquet, compression="zstd")
    logger.info("--- Done. Raw dataset saved and uploaded. ---")
    return df


def _dataset_name_values(dataset_names: tuple[DataSetName, ...] | list[DataSetName]) -> list[str]:
    return [dataset_name.value for dataset_name in dataset_names]


def _normalize_dataset_names(dataset_names: tuple[DataSetName, ...] | list[DataSetName]) -> tuple[DataSetName, ...]:
    return tuple(dict.fromkeys(dataset_names))


@cache
def _load_raw_dataset_cached(
    dataset_names: tuple[DataSetName, ...],
    paths: DataPaths,
) -> Dataset:
    """Load and normalize raw data once per process."""
    from datasets import Dataset, load_dataset

    paths.datasets_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(HF_DATASET_REPO, name="raw", split="data", cache_dir=str(paths.datasets_cache_dir))
    if not isinstance(dataset, Dataset):
        msg = "Raw data is not a datasets.Dataset"
        raise TypeError(msg)

    if dataset_names:
        sources = set(_dataset_name_values(dataset_names))
        dataset = dataset.filter(lambda source: source in sources, input_columns="source")
    if "symbol" in dataset.column_names and "strokes" not in dataset.column_names:
        dataset = dataset.rename_column("symbol", "strokes")
    return cast("Dataset", dataset)


def load_raw_dataset(
    dataset_names: tuple[DataSetName, ...] | list[DataSetName],
    paths: DataPaths = DEFAULT_DATA_PATHS,
) -> Dataset:
    """Load the raw Hugging Face dataset and filter by source."""
    return _load_raw_dataset_cached(_normalize_dataset_names(dataset_names), paths)


@cache
def _map_raw_dataset_cached(
    dataset_names: tuple[DataSetName, ...],
    num_proc: int | None,
    paths: DataPaths,
) -> tuple[Dataset, dict[str, set[str]]]:
    """Map and filter raw data once per process; datasets persists Arrow cache on disk."""
    from datasets import Value

    tex_to_char = get_tex_to_char()
    tex_typ_map_digest = get_tex_typ_map_digest()
    raw_dataset = load_raw_dataset(dataset_names, paths)
    raw_dataset_fingerprint = getattr(raw_dataset, "_fingerprint", "")
    map_fingerprint = sha256(
        dumps(
            {
                "base": raw_dataset_fingerprint,
                "dataset_names": _dataset_name_values(dataset_names),
                "stage": "latex-to-typst-v1",
                "tex_typ_map_digest": tex_typ_map_digest,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()

    def map_labels(batch, mapping: dict[str, str]):
        return {"label": [mapping.get(label) for label in batch["latex_label"]]}

    mapped = cast(
        "Dataset",
        raw_dataset.map(
            map_labels,
            batched=True,
            num_proc=num_proc,
            fn_kwargs={"mapping": tex_to_char},
            new_fingerprint=map_fingerprint,
            writer_batch_size=1000,
            desc="Mapping LaTeX labels",
        ),
    )
    mapped = cast("Dataset", mapped.cast_column("label", Value("string")))

    unmapped: dict[str, set[str]] = {}
    unmapped_rows = cast(
        "Dataset",
        mapped.filter(lambda label: label is None, input_columns="label").select_columns(["latex_label", "source"]),
    )
    for row in cast("list[dict[str, Any]]", unmapped_rows.to_list()):
        unmapped.setdefault(row["source"], set()).add(row["latex_label"])

    def keep_mapped(label: str | None, strokes: list) -> bool:
        return label is not None and len(strokes) > 0

    mapped = cast(
        "Dataset",
        mapped.filter(
            keep_mapped,
            input_columns=["label", "strokes"],
            num_proc=num_proc,
            desc="Dropping unmapped or empty samples",
        ),
    )
    return mapped, unmapped


def map_raw_dataset(
    dataset_names: tuple[DataSetName, ...] | list[DataSetName],
    *,
    num_proc: int | None = None,
    paths: DataPaths = DEFAULT_DATA_PATHS,
) -> tuple[Dataset, dict[str, set[str]]]:
    """Map raw LaTeX labels to Typst chars using Hugging Face dataset caching."""
    normalized_num_proc = num_proc or None
    return _map_raw_dataset_cached(_normalize_dataset_names(dataset_names), normalized_num_proc, paths)


def get_dataset_classes(
    dataset_names: tuple[DataSetName, ...] | list[DataSetName] | None = None,
    paths: DataPaths = DEFAULT_DATA_PATHS,
    max_samples: int | None = None,
    num_proc: int | None = None,
) -> list[str]:
    """Return classes from locally mapped raw labels, without relying on HF ClassLabel metadata."""
    mapped, _ = map_raw_dataset(
        list(dataset_names or (DataSetName.detexify, DataSetName.mathwriting)),
        num_proc=num_proc,
        paths=paths,
    )
    if max_samples is not None:
        shuffled = cast("Dataset", mapped.shuffle(seed=DETERMINISTIC_SPLIT_SEED))
        mapped = cast(
            "Dataset",
            shuffled.select(range(min(max_samples, len(shuffled)))),
        )
    return sorted(cast("list[str]", mapped.unique("label")))


def get_rendered_dataset_splits(
    dataset_names: tuple[DataSetName, ...],
    image_size: int,
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    num_proc: int | None = None,
    paths: DataPaths = DEFAULT_DATA_PATHS,
    max_samples: int | None = None,
    min_split_class_count: int | None = None,
) -> tuple[DatasetDict, list[str]]:
    """Build rendered train/test/val splits using Hugging Face dataset caches."""
    from collections import Counter
    from math import ceil

    from datasets import Array2D, ClassLabel, DatasetDict, Features, concatenate_datasets

    mapped, _ = map_raw_dataset(dataset_names, num_proc=num_proc, paths=paths)
    if max_samples is not None:
        shuffled = cast("Dataset", mapped.shuffle(seed=DETERMINISTIC_SPLIT_SEED))
        mapped = cast(
            "Dataset",
            shuffled.select(range(min(max_samples, len(shuffled)))),
        )
    classes = sorted(cast("list[str]", mapped.unique("label")))
    label_to_idx = {label: idx for idx, label in enumerate(classes)}

    def rasterize_batch(batch, size: int, labels: dict[str, int]):
        return {
            "label": [labels[label] for label in batch["label"]],
            "image": [rasterize_strokes(strokes, size).tolist() for strokes in batch["strokes"]],
        }

    rendered = cast(
        "Dataset",
        mapped.map(
            rasterize_batch,
            batched=True,
            num_proc=num_proc,
            fn_kwargs={"size": image_size, "labels": label_to_idx},
            remove_columns=mapped.column_names,
            writer_batch_size=128,
            features=Features(
                {
                    "label": ClassLabel(names=classes),
                    "image": Array2D(shape=(image_size, image_size), dtype="uint8"),
                }
            ),
            desc=f"Rasterizing {image_size}px symbols",
        ),
    )

    _, test_r, val_r = split_ratio
    holdout_r = test_r + val_r
    if min_split_class_count is None:
        min_split_class_count = max(2, ceil(2 / holdout_r))

    label_counts = Counter(cast("list[int]", rendered["label"]))
    rare_labels = {label for label, count in label_counts.items() if count < min_split_class_count}
    if rare_labels and len(rare_labels) < len(label_counts):
        rare = cast("Dataset", rendered.filter(lambda label: label in rare_labels, input_columns="label"))
        rendered = cast("Dataset", rendered.filter(lambda label: label not in rare_labels, input_columns="label"))
    else:
        rare = None

    split = cast(
        "DatasetDict",
        rendered.train_test_split(
            test_size=holdout_r,
            seed=DETERMINISTIC_SPLIT_SEED,
            stratify_by_column=None if max_samples is not None else "label",
        ),
    )
    holdout = split["test"]
    test_val = cast(
        "DatasetDict",
        holdout.train_test_split(
            test_size=val_r / holdout_r,
            seed=DETERMINISTIC_SPLIT_SEED,
            stratify_by_column=None if max_samples is not None else "label",
        ),
    )
    train = split["train"]
    if rare is not None:
        train = concatenate_datasets([train, rare])

    splits = DatasetDict(
        {
            "train": train,
            "test": test_val["train"],
            "val": test_val["test"],
        }
    )
    return splits.with_format("torch"), classes
