from __future__ import annotations

from functools import cache
from hashlib import blake2b
from json import dumps
from typing import TYPE_CHECKING, Any, cast

from detypify.config import DETERMINISTIC_SPLIT_SEED, HF_DATASET_REPO, DataSetName
from detypify.data.paths import DEFAULT_DATA_PATHS, DataPaths
from detypify.data.rendering import rasterize_strokes
from detypify.data.symbols import get_tex_to_char, get_tex_typ_map_digest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from datasets import Dataset, DatasetDict


def _dataset_name_values(dataset_names: tuple[DataSetName, ...]) -> list[str]:
    return [dataset_name.value for dataset_name in dataset_names]


@cache
def _load_raw_dataset_cached(dataset_names: tuple[DataSetName, ...], paths: DataPaths) -> Dataset:
    """Load and normalize raw data once per process."""
    from datasets import Dataset, load_dataset

    paths.datasets_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(HF_DATASET_REPO, name="raw", split="data", cache_dir=str(paths.datasets_cache_dir))
    if not isinstance(dataset, Dataset):
        msg = "Raw data is not a datasets.Dataset"
        raise TypeError(msg)

    sources = set(_dataset_name_values(dataset_names))
    dataset = dataset.filter(lambda source: source in sources, input_columns="source")
    return cast("Dataset", dataset)


def load_raw_dataset(dataset_names: Sequence[DataSetName], paths: DataPaths) -> Dataset:
    """Load the raw Hugging Face dataset and filter by source."""
    return _load_raw_dataset_cached(tuple(dataset_names), paths)


@cache
def _map_raw_dataset_cached(
    dataset_names: tuple[DataSetName, ...],
    num_proc: int,
    paths: DataPaths,
) -> tuple[Dataset, dict[str, set[str]]]:
    """Map and filter raw data once per process; datasets persists Arrow cache on disk."""
    from datasets import Value

    hf_num_proc = num_proc or None
    tex_to_char = get_tex_to_char()
    tex_typ_map_digest = get_tex_typ_map_digest()
    raw_dataset = _load_raw_dataset_cached(dataset_names, paths)
    raw_dataset_fingerprint = getattr(raw_dataset, "_fingerprint", "")
    map_fingerprint = blake2b(
        data=dumps(
            {
                "base": raw_dataset_fingerprint,
                "dataset_names": _dataset_name_values(dataset_names),
                "stage": "latex-to-typst-v1",
                "tex_typ_map_digest": tex_typ_map_digest,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode(),
        # len(hexdigest) is 2 * digest_size
        # but the num_proc is also hashed implicitly by huggingface datasets, causing a hexdigest + num_proc size internal fingerprint
        # so use 16 here
        digest_size=16,
    ).hexdigest()

    def map_labels(batch, mapping: dict[str, str]):
        return {"label": [mapping.get(label) for label in batch["latex_label"]]}

    mapped = cast(
        "Dataset",
        raw_dataset.map(
            map_labels,
            batched=True,
            num_proc=hf_num_proc,
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
            num_proc=hf_num_proc,
            desc="Dropping unmapped or empty samples",
        ),
    )
    return mapped, unmapped


def map_raw_dataset(
    dataset_names: Sequence[DataSetName],
    *,
    num_proc: int = 0,
    paths: DataPaths = DEFAULT_DATA_PATHS,
) -> tuple[Dataset, dict[str, set[str]]]:
    """Map raw LaTeX labels to Typst chars using Hugging Face dataset caching."""
    return _map_raw_dataset_cached(tuple(dataset_names), num_proc, paths)


def get_dataset_classes(
    dataset_names: Sequence[DataSetName],
    *,
    max_samples: int | None,
    num_proc: int,
    paths: DataPaths = DEFAULT_DATA_PATHS,
) -> list[str]:
    """Return classes from locally mapped raw labels, without relying on HF ClassLabel metadata."""
    mapped, _ = map_raw_dataset(dataset_names, num_proc=num_proc, paths=paths)
    if max_samples is not None:
        shuffled = cast("Dataset", mapped.shuffle(seed=DETERMINISTIC_SPLIT_SEED))
        mapped = cast("Dataset", shuffled.select(range(min(max_samples, len(shuffled)))))
    return sorted(cast("list[str]", mapped.unique("label")))


def get_rendered_dataset_splits(
    dataset_names: Sequence[DataSetName],
    image_size: int,
    num_proc: int,
    paths: DataPaths,
    max_samples: int | None,
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    min_split_class_count: int | None = None,
) -> tuple[DatasetDict, list[str]]:
    """Build rendered train/test/val splits using Hugging Face dataset caches."""
    from collections import Counter
    from math import ceil

    from datasets import Array2D, ClassLabel, DatasetDict, Features, concatenate_datasets

    mapped, _ = map_raw_dataset(dataset_names, num_proc=num_proc, paths=paths)
    if max_samples is not None:
        shuffled = cast("Dataset", mapped.shuffle(seed=DETERMINISTIC_SPLIT_SEED))
        mapped = cast("Dataset", shuffled.select(range(min(max_samples, len(shuffled)))))
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

    splits = DatasetDict({"train": train, "test": test_val["train"], "val": test_val["test"]})
    return splits.with_format("torch"), classes
