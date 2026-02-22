"""Preprocess training datasets, helping functions and related constants/types."""

from __future__ import annotations

import logging
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import typer
from msgspec import Struct, json

if TYPE_CHECKING:
    import polars as pl

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]
type DataSetName = Literal["mathwriting", "detexify", "contrib"]
type SplitName = Literal["train", "test", "val"]


# Constants
DATASET_ROOT = Path("build/dataset")
DATA_ROOT = Path("data")
EXTERNAL_DATA_PATH = Path("build/raw_data")
MATH_WRITING_DATA_PATH = EXTERNAL_DATA_PATH / "mathwriting"
DETEXIFY_DATA_PATH = EXTERNAL_DATA_PATH / "detexify"
CONTRIB_DATA = Path("build/dataset.json")
DATASET_REPO = "Cloud0310/detypify-datasets"
TEX_TO_TYP_PATH = Path(__file__).parent / "tex_to_typ.yaml"
UPLOAD = True
RAW_POINT_LENGTH = 3


# Structs
class TypstSymInfo(Struct, kw_only=True, omit_defaults=True):
    char: str
    names: list[str]
    latex_name: str | None = None
    markup_shorthand: str | None = None
    math_shorthand: str | None = None
    accent: bool = False
    alternates: list[str] | None = None


class MathWritingDatasetInfo(Struct, kw_only=True, omit_defaults=True):
    name: str
    unmapped: set[str] | None = None


class DetexifySymInfo(Struct, kw_only=True, omit_defaults=True):
    command: str
    # package: str | None = None
    # mathmode: bool
    # textmode: bool
    id: str
    # css_class: str


class MathSymbolSample(Struct):
    label: str
    symbol: Strokes


type RAW_POINT = list[list[tuple[float, float, float]]]


# Helper functions
def is_invisible(c: str) -> bool:
    from unicodedata import category

    return category(c) in {"Zs", "Cc", "Cf"}


@cache
def get_typst_symbol_info() -> list[TypstSymInfo]:
    """Parses the Typst symbol page to extract symbol information.

    Retrieves the HTML content from the Typst documentation (downloading it if
    necessary) and parses it to find symbol names, characters, and their LaTeX
    equivalents.

    Returns:
        A list of `TypstSymInfo` objects containing details for each symbol.
    """

    import re
    from urllib.request import urlopen

    page_url = "https://typst.app/docs/reference/symbols/sym/"
    with urlopen(page_url) as resp:
        page_data = resp.read()

    from bs4 import BeautifulSoup

    sym_info = {}
    if page_data:
        soup = BeautifulSoup(page_data, "lxml")
        for li in soup.find_all("li", id=re.compile("^symbol-")):
            name = li["id"][len("symbol-") :]
            char = li["data-value"][0]
            if is_invisible(char) or li.get("data-deprecation"):
                # We don't care about invisible chars and deprecated names.
                continue
            if char in sym_info:
                # Repeated symbols. Merge names.
                sym_info[char].names.append(name)
            else:
                latex_name, markup_shorthand, math_shorthand, alternates = (
                    li.get("data-latex-name"),
                    li.get("data-markup-shorthand"),
                    li.get("data-math-shorthand"),
                    li.get("data-alternates", ""),
                )

                # New symbols. Add to map.
                sym_info[char] = TypstSymInfo(
                    char=char,
                    names=[cast("str", name)],
                    latex_name=cast("str | None", latex_name),
                    markup_shorthand=cast("str | None", markup_shorthand),
                    math_shorthand=cast("str | None", math_shorthand),
                    accent=li.get("accent") == "true",
                    alternates=cast("str", alternates).split(),
                )
    else:
        logging.warning("Unable to retrive page data.")

    return list(sym_info.values())


def rasterize_strokes(strokes: Strokes, output_size: int):
    """
    Normalizes vector strokes and rasterizes them into a binary NumPy array.

    Args:
        strokes: List of strokes, where each stroke is a list of (x, y) coordinates.
        output_size: The width and height of the output square grid.

    Returns:
        np.ndarray: A (size, size) uint8 array.
                    Background is 0 (black), Strokes are 255 (white).
    """

    import cv2
    import numpy as np

    if not strokes:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    stroke_arrays = [np.array(s, dtype=np.float32) for s in strokes if s]

    if not stroke_arrays:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    all_points = np.vstack(stroke_arrays)

    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    padding = 10
    target_size = output_size - (2 * padding)

    width = max(max_x - min_x, max_y - min_y)
    scale = target_size / width if width > 1e-6 else 1.0  # noqa: PLR2004
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # In-place transformation of the big array
    all_points = ((all_points - [center_x, center_y]) * scale) + [
        output_size / 2,
        output_size / 2,
    ]

    lengths = [len(a) for a in stroke_arrays]
    split_indices = np.cumsum(lengths)[:-1]
    normalized_strokes = np.split(all_points.astype(np.int32), split_indices)

    canvas = np.zeros((output_size, output_size), dtype=np.uint8)

    # decrease to increase the stroke thinkness
    # thinker stroke for better feature extraction
    thickness_factor = 25
    thickness = max(1, output_size // thickness_factor)
    cv2.polylines(canvas, normalized_strokes, isClosed=False, color=255, thickness=thickness)

    return canvas


@cache
def get_dataset_classes(dataset_name: str) -> list[str]:
    from datasets import load_dataset

    classes: set[str] = set()
    dataset = load_dataset(dataset_name)
    for split in dataset:
        classes.update(dataset[split].features["label"].names)
    return sorted(classes)


@cache
def map_tex_typ() -> dict[str, TypstSymInfo]:
    """Creates a mapping from TeX command names to Typst symbol information.

    Combines mappings from the Typst symbol page (via `get_typst_symbol_info`)
    and a manual fallback dictionary (`TEX_TO_TYP`).

    Returns:
        A dictionary where keys are TeX commands (e.g., "\\alpha") and values
        are `TypstSymInfo` objects.
    """
    typ_sym_info = get_typst_symbol_info()
    # mapping for symbol name to unicode char
    tex_to_typ = {s.latex_name: s for s in typ_sym_info if s.latex_name is not None}
    name_to_typ = {name: s for s in typ_sym_info for name in s.names}

    with TEX_TO_TYP_PATH.open("rb") as f:
        from msgspec.yaml import decode

        manual_mapping = decode(f.read(), type=dict[str, str])

    tex_to_typ |= {k: name_to_typ[v] for k, v in manual_mapping.items()}
    return tex_to_typ


@cache
def get_xml_parser():
    from lxml import etree

    return etree.XMLParser()


def parse_inkml_symbol(
    filepath: Path,
) -> MathSymbolSample | None:
    """Parses a single InkML file to extract the raw LaTeX label and stroke data.

    This version does NOT map to Typst symbols, keeping the original LaTeX label.

    Args:
        filepath: Path to the .inkml file.

    Returns:
        A `MathSymbolSample` with latex_label as label if successful.
        None if no label is found.
    """

    from lxml import etree

    # parsing
    root = etree.parse(filepath, parser=get_xml_parser()).getroot()
    namespace = {"ink": "http://www.w3.org/2003/InkML"}
    tex_label = root.findtext(".//ink:annotation[@type='label']", namespaces=namespace)

    # couldn't find data, return None
    if not tex_label:
        return None

    return MathSymbolSample(
        tex_label,
        [
            [
                (float(x), float(y))
                for x, y, _ in (
                    # keep only x,y, discard time
                    point_str.split()
                    for point_str in trace.text.split(",")
                    if len(point_str.split()) == RAW_POINT_LENGTH
                )
            ]
            for trace in root.iterfind(".//ink:trace", namespaces=namespace)
            if trace.text
        ],
    )


# Raw dataset functions
def collect_mathwriting_raw():
    """Collects raw MathWriting data with LaTeX labels (not mapped to Typst).

    Parses InkML files in parallel to extract strokes and original LaTeX labels.

    Returns:
        A Polars LazyFrame with columns:
            - latex_label: Original LaTeX command string
            - symbol: List of strokes as arrays of (x, y) coordinates
    """

    from concurrent.futures import ProcessPoolExecutor

    import polars as pl

    label_acc = []
    data_acc = []

    with ProcessPoolExecutor() as executor:
        results = executor.map(
            parse_inkml_symbol,
            MATH_WRITING_DATA_PATH.glob("*.inkml"),
            chunksize=500,
        )
        for result in results:
            if result is None:
                continue
            label_acc.append(result.label)
            data_acc.append(result.symbol)

    del results
    pl_schema = {
        "latex_label": pl.String,
        "symbol": pl.List(pl.List(pl.Array(pl.Float32, 2))),
    }

    return pl.DataFrame({"latex_label": label_acc, "symbol": data_acc}, schema=pl_schema).lazy()


def collect_detexify_raw():
    """Collects raw Detexify data with original command labels (not mapped to Typst).

    Reads the raw JSON data and formats strokes without Typst symbol mapping.

    Returns:
        A Polars LazyFrame with columns:
            - latex_label: Original LaTeX command string
            - symbol: List of strokes as arrays of (x, y) coordinates
    """

    import polars as pl

    pl.Config.set_engine_affinity("streaming")

    with (DETEXIFY_DATA_PATH / "symbols.json").open("rb") as f:
        tex_sym_info = json.decode(f.read(), type=list[DetexifySymInfo])

    # Create mapping from key to command (latex label)
    key_to_command = {x.id: x.command for x in tex_sym_info}

    with (DETEXIFY_DATA_PATH / "detexify.json").open("rb") as f:
        # Schema: list of (key, strokes)
        data = json.decode(f.read(), type=list[tuple[str, list[list[tuple[float, float, float]]]]])

    raw_data_schema = {
        "key": pl.String,
        "strokes": pl.List(pl.List(pl.Array(pl.Float32, 3))),
    }

    raw_lf = pl.DataFrame(data, schema=raw_data_schema, orient="row").lazy()
    del data

    # Prepare Mapping
    mapping_lf = pl.DataFrame(
        {
            "key": list(key_to_command.keys()),
            "latex_label": list(key_to_command.values()),
        }
    ).lazy()

    processed_lf = raw_lf.join(mapping_lf, on="key", how="left")

    return (
        processed_lf.filter(pl.col("latex_label").is_not_null())
        .select(
            [
                pl.col("latex_label"),
                pl.col("strokes")
                # Drop time (keep only x, y)
                .list.eval(pl.element().list.eval(pl.element().arr.head(2).list.to_array(2)))
                .alias("symbol"),
            ]
        )
        # Drop empty samples
        .filter(pl.col("symbol").list.len() > 0)
    )


def collect_contrib_raw():
    """Collects raw contributed data with symbol names (not mapped to Typst).

    Reads the contrib JSON and decodes strokes without Typst symbol mapping.

    Returns:
        A Polars LazyFrame with columns:
            - latex_label: Original symbol name string
            - symbol: List of strokes as arrays of (x, y) coordinates
    """

    import polars as pl

    # Load Data
    with CONTRIB_DATA.open("rb") as f:
        # Schema: list of dicts {sym: str, strokes: json_string}
        data = json.decode(f.read(), type=list[dict[str, str]])

    # Decode strokes and rename sym to latex_label
    processed_lf = (
        pl.DataFrame(data)
        .lazy()
        .rename({"sym": "latex_label"})
        # Decode strokes
        .with_columns(
            pl.col("strokes")
            .map_elements(
                json.decode,
                return_dtype=pl.List(pl.List(pl.Array(pl.Float32, 2))),
            )
            .alias("symbol")
        )
        .drop("strokes")
    )

    return (
        processed_lf.filter(pl.col("latex_label").is_not_null())
        .select(
            [
                pl.col("latex_label"),
                pl.col("symbol"),
            ]
        )
        # Drop empty samples
        .filter(pl.col("symbol").list.len() > 0)
    )


def create_raw_dataset(
    dataset_names: list[DataSetName],
    *,
    upload: bool = True,
) -> pl.DataFrame:
    """Creates and uploads raw dataset with original LaTeX/command labels.

    This dataset is intended for CI/CD processing pipelines. It contains
    the original labels before Typst symbol mapping.

    Args:
        dataset_names: List of dataset names to include ("mathwriting", "detexify", "contrib")
        upload: Whether to upload to HuggingFace (True) or save locally (False)
    """

    from os import process_cpu_count

    import polars as pl
    from datasets import (
        Dataset,
        DatasetInfo,
        Features,
        List,
        Sequence,
        Value,
    )

    logging.info(f"--- Creating Raw Dataset: {','.join(dataset_names)} ---")

    lfs = []
    for dataset_name in dataset_names:
        match dataset_name:
            case "mathwriting":
                math_writing_lf = collect_mathwriting_raw()
                # Add source column
                math_writing_lf = math_writing_lf.with_columns(pl.lit("mathwriting").alias("source"))
                lfs.append(math_writing_lf)
            case "detexify":
                detexify_lf = collect_detexify_raw()
                # Add source column
                detexify_lf = detexify_lf.with_columns(pl.lit("detexify").alias("source"))
                lfs.append(detexify_lf)
            case "contrib":
                contrib_lf = collect_contrib_raw()
                # Add source column
                contrib_lf = contrib_lf.with_columns(pl.lit("contrib").alias("source"))
                lfs.append(contrib_lf)

    # Concatenate all datasets
    lf = pl.concat(lfs)

    # Collect and shuffle
    df = lf.collect().sample(fraction=1.0, shuffle=True, seed=114514)

    if upload:
        features: Features = Features(
            {
                "latex_label": Value("string"),
                "symbol": List(List(Sequence(Value("float32"), length=2))),
                "source": Value("string"),
            }
        )  # type: ignore

        description = (
            "Raw detypify dataset with original LaTeX labels, "
            "composed by mathwriting, detexify and contributed datasets. "
            "Intended for CI/CD processing pipelines."
        )

        dataset_info = DatasetInfo(description=description, features=features)
        dataset = Dataset.from_polars(df, info=dataset_info)

        logging.info("  -> Uploading raw dataset to %s as 'raw_data'...", DATASET_REPO)
        dataset.push_to_hub(repo_id=DATASET_REPO, config_name="raw", split="data", num_proc=process_cpu_count() or 1)
        logging.info("--- Done. Raw dataset uploaded to %s (config: raw) ---", DATASET_REPO)
    else:
        # Save locally
        raw_dataset_path = DATASET_ROOT / "raw"
        raw_dataset_path.mkdir(parents=True, exist_ok=True)

        logging.info("  -> Saving raw dataset locally to %s...", raw_dataset_path)
        df.write_parquet(
            raw_dataset_path / "data.parquet",
            compression="zstd",
        )
        logging.info("--- Done. Raw dataset saved to %s ---", raw_dataset_path)
    return df


def load_raw_dataset(dataset_names: list[DataSetName]) -> pl.DataFrame:
    """Load raw dataset from HF config 'raw'.

    Raises:
        ValueError: If raw dataset not found on HF with helpful message.
    """
    import polars as pl
    from datasets import Dataset, load_dataset

    df = Dataset.to_polars(load_dataset(DATASET_REPO, name="raw", split="data"))
    if not isinstance(df, pl.DataFrame):
        err_msg = "Raw data is not pl.DataFrame"
        raise TypeError(err_msg)

    # Filter by source if dataset_names provided
    if dataset_names:
        df = df.filter(pl.col("source").is_in(dataset_names))

    return df


def apply_typst_mapping(
    df: pl.DataFrame,
) -> tuple[pl.LazyFrame, dict[DataSetName, set[str]]]:
    """Apply LaTeX→Typst mapping to raw data.

    Returns:
        - Mapped LazyFrame with columns: label, strokes, source
        - Dictionary of unmapped labels per source
    """
    import polars as pl

    tex_to_typ = map_tex_typ()

    # We need to flatten the TypstSymInfo to just the char for mapping
    tex_to_char = {k: v.char for k, v in tex_to_typ.items()}

    # Apply mapping
    mapped_df = df.with_columns(
        [
            pl.col("latex_label").replace(tex_to_char, default=None).alias("label"),
        ]
    )

    # Track unmapped per source
    unmapped_df = mapped_df.filter(pl.col("label").is_null()).group_by("source").agg(pl.col("latex_label").unique())
    unmapped = {row["source"]: set(row["latex_label"]) for row in unmapped_df.to_dicts()}

    # Filter out unmapped, select final columns, convert strokes format
    result_df = (
        mapped_df.filter(pl.col("label").is_not_null())
        .select(
            [
                pl.col("label"),
                pl.col("symbol").alias("strokes"),  # Rename to match processed format
                pl.col("source"),
            ]
        )
        .filter(pl.col("strokes").list.len() > 0)  # Drop empty
    )

    return result_df.lazy(), unmapped


def remap_from_raw(
    dataset_names: list[DataSetName],
    data: pl.DataFrame | None = None,
) -> tuple[pl.LazyFrame, dict[DataSetName, set[str]]]:
    """High-level function: Load raw data and apply fresh mapping.

    When Typst reference changes, reload raw data and remap.
    """

    if data is None:
        logging.info("  -> Loading raw dataset from HuggingFace...")
        data = load_raw_dataset(dataset_names)

    logging.info(f"  -> Applying LaTeX→Typst mapping to {len(data)} samples...")
    return apply_typst_mapping(data)


def generate_infer_data(classes: list[str] | None = None) -> None:
    """Generate the infer and contrib data

    Args:
        classes: Optional set of character classes to generate infer.json for.
                 If None, generates for all available symbols.
    """

    infer_path = DATA_ROOT / "infer.json"
    contrib_path = DATA_ROOT / "contrib.json"

    infer_path.parent.mkdir(parents=True, exist_ok=True)
    contrib_path.parent.mkdir(parents=True, exist_ok=True)

    typ_sym_info = get_typst_symbol_info()

    infer = []
    if classes:
        contrib = {n: s.char for s in typ_sym_info for n in s.names}
        chr_to_sym = {s.char: s for s in typ_sym_info}
        for c in classes:
            if c not in chr_to_sym:
                continue
            sym = chr_to_sym[c]
            info = {"char": sym.char, "names": sym.names}
            if sym.markup_shorthand and sym.math_shorthand:
                info["shorthand"] = sym.markup_shorthand
            elif sym.markup_shorthand:
                info["markupShorthand"] = sym.markup_shorthand
            elif sym.math_shorthand:
                info["mathShorthand"] = sym.math_shorthand
            infer.append(info)
    else:
        for sym in typ_sym_info:
            info = {"char": sym.char, "names": sym.names}
            if sym.markup_shorthand and sym.math_shorthand:
                info["shorthand"] = sym.markup_shorthand
            elif sym.markup_shorthand:
                info["markupShorthand"] = sym.markup_shorthand
            elif sym.math_shorthand:
                info["mathShorthand"] = sym.math_shorthand
            infer.append(info)

    for path, data in [(infer_path, infer), (contrib_path, contrib)]:
        with path.open("wb") as f:
            f.write(json.encode(data))


def create_dataset(
    dataset_names: list[DataSetName],
    raw_data: pl.DataFrame | None = None,
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    *,
    upload: bool = True,
    **kwargs,
) -> None:
    """Orchestrates the creation of a math symbol dataset.

    1. Loads symbol mappings using the specific project logic.
    2. Dispatches data loading to specific construct_* functions.
    3. Performs a stratified train/test/val split.
    4. Saves data as sharded files (Parquet/Vortex) and writes metadata.

    Args:
        dataset_names: The names of the dataset to use.
        raw_data: The dataframe. Optional, if nothing, load from huggingface
        split_ratio: A tuple defining the ratio for (train, test, val) splits.
        upload: whether or not uploading dataset to hugggingface.
        kwargs: valid only when not uploading.
            split_parts: Whether to split the dataset into multiple shards.
            batch_size: The number of rows per shard.
            file_format: The output file format ("parquet" or "vortex").
    """

    import polars as pl
    from datasets import (
        ClassLabel,
        Dataset,
        DatasetInfo,
        Features,
        LargeList,
        List,
        Value,
    )

    logging.info(f"--- Creating Datasets: {','.join(dataset_names)} ---")

    # load from raw data
    lf, unmapped = remap_from_raw(dataset_names, raw_data)

    dataset_path = DATASET_ROOT
    split_names: list[SplitName] = ["train", "test", "val"]

    from shutil import rmtree as rmdir

    if dataset_path.exists():
        rmdir(dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)

    train_r, test_r, _ = split_ratio
    t1 = train_r
    t2 = train_r + test_r

    logging.info("  -> Shuffling and splitting data...")

    # Add Stratified Indices
    # Casting label to Utf8 ensures consistency across datasets.
    base_lf = (
        lf.with_columns(pl.col("label").cast(pl.Utf8))
        .collect()
        .sample(fraction=1.0, shuffle=True, seed=114514)
        .lazy()
        .with_columns(
            [
                pl.len().over("label").alias("n"),
                pl.int_range(0, pl.len()).over("label").alias("idx"),
            ]
        )
    )
    logging.info("  -> Generating metadata...")
    # Use base_lf (materialized view logic) for fast stats
    stats_df = base_lf.select(pl.col("label")).collect().get_column("label").value_counts().sort("label")

    # Split and shuffle data
    train_lf = base_lf.filter(pl.col("idx") < (pl.col("n") * t1)).drop(["n", "idx"]).sort("label").collect()
    test_lf = (
        base_lf.filter((pl.col("idx") >= (pl.col("n") * t1)) & (pl.col("idx") < (pl.col("n") * t2)))
        .drop(["n", "idx"])
        .sort("label")
        .collect()
    )
    val_lf = base_lf.filter(pl.col("idx") >= (pl.col("n") * t2)).drop(["n", "idx"]).sort("label").collect()

    if upload:
        global_features: Features = Features(
            {
                "label": ClassLabel(names=stats_df["label"].to_list()),
                "strokes": LargeList(LargeList(List(Value("float32")))),
                "source": Value("string"),
            }
        )  # type: ignore

        def _upload_to_huggingface(
            split: SplitName,
            data_frame: pl.DataFrame,
        ):
            from os import process_cpu_count

            def encode_labels(batch):
                class_feature = global_features["label"]
                batch["label"] = [class_feature.str2int(label) for label in batch["label"]]
                return batch

            description = "Detypify dataset, composed by mathwriting, detexify and contributed datasets"
            dataset_info = DatasetInfo(description=description)
            dataset = cast(
                Dataset,
                Dataset.from_polars(data_frame, info=dataset_info)
                .map(encode_labels, batched=True)
                .cast(features=global_features),
            )
            dataset.push_to_hub(
                repo_id=DATASET_REPO, num_proc=process_cpu_count() or 1, split=split, set_default=True, create_pr=True
            )

        for df, split in zip([train_lf, test_lf, val_lf], split_names, strict=True):
            logging.info("  -> Uploading split: %s... to huggingface.", split)
            _upload_to_huggingface(split, df)
        logging.info("--- Done. Dataset uploaded to %s ---", DATASET_REPO)
    else:
        split_parts: bool = kwargs.get("split_parts", False)
        batch_size: int = kwargs.get("batch_size", 2000)
        file_format: Literal["vortex", "parquet"] = kwargs.get("file_format", "parquet")

        def _write_to_file(df: pl.DataFrame, path: Path, file_format="parquet"):
            if file_format == "vortex":
                import vortex as vx

                vx.io.write(vx.compress(vx.array(df.to_arrow())), str(path))
            else:
                df.write_parquet(path, compression="zstd", compression_level=19)

        def _write_shards(df: pl.DataFrame, output_dir: Path):
            total_rows = len(df)
            if total_rows == 0:
                return

            num_shards = total_rows // batch_size
            pad_width = len(str(num_shards))

            logging.info(f"  -> Writing {total_rows} rows to{output_dir.name} ({num_shards} shards).")

            for i, start_idx in enumerate(range(0, total_rows, batch_size)):
                chunk = df.slice(start_idx, batch_size)
                filename = f"part_{str(i).zfill(pad_width)}.{file_format}"
                _write_to_file(chunk, output_dir / filename)

        for df, split in zip([train_lf, test_lf, val_lf], split_names, strict=True):
            output_dir = dataset_path / split
            output_dir.mkdir(exist_ok=True)
            if split_parts:
                _write_shards(df, output_dir)
            else:
                logging.info(f"  -> Writing {len(df)} rows to{output_dir.name} as single file.")
                _write_to_file(df, output_dir / f"data.{file_format}")
        logging.info("--- Done. Dataset saved to %s ---", dataset_path)

    for dataset_name, symbols in unmapped.items():
        dataset_path = DATASET_ROOT / dataset_name
        dataset_path.mkdir(exist_ok=True)
        dataset_info = MathWritingDatasetInfo(
            name=dataset_name,
            unmapped=symbols or None,
        )
        with (dataset_path / "dataset_info.json").open("wb") as f:
            f.write(json.format(json.encode(dataset_info)))


class DataSetNameEnum(StrEnum):
    mathwriting = "mathwriting"
    detexify = "detexify"
    contrib = "contrib"


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    datasets: list[DataSetNameEnum] = typer.Option(
        [DataSetNameEnum.detexify, DataSetNameEnum.mathwriting],
        "--datasets",
        "-d",
        help="Datasets to process when converting data.",
    ),
    skip_convert_data: bool = typer.Option(False, help="Do not construct or upload local datasets."),
    skip_gen_info: bool = typer.Option(False, help="Skip writing symbol metadata and infer JSON files."),
    include_contrib: bool = typer.Option(
        False,
        help="Append the contrib dataset even if it was not listed explicitly.",
    ),
    upload: bool = typer.Option(
        False,
        help="Store processed datasets locally.",
    ),
    split_ratio: tuple[float, float, float] = typer.Option(
        (0.8, 0.1, 0.1),
        help="Train/test/val split ratios for the processed dataset.",
    ),
    split_parts: bool = typer.Option(
        False,
        help="Write each split as multiple shards instead of a single file.",
    ),
    batch_size: int = typer.Option(
        2000,
        help="Number of rows per shard when --split-parts is enabled.",
    ),
    file_format: str = typer.Option(
        "parquet",
        help="Output format(parquet/vortex) when writing dataset shards locally.",
    ),
    create_raw: bool = typer.Option(
        False,
        "--create-raw",
        help="Create and upload raw dataset with original LaTeX labels.",
    ),
    remap: bool = typer.Option(
        False,
        "--remap",
        help="Load raw dataset from HF and re-apply LaTeX →  Typst mapping.",
    ),
):
    """
    Preprocess datasets, generate metadata, and upload results.
    """
    logging.basicConfig(level=logging.INFO)

    convert_data: bool = not skip_convert_data
    gen_info: bool = not skip_gen_info

    dataset_names: list[DataSetName] = [cast("DataSetName", d.value) for d in dict.fromkeys(datasets)]
    if include_contrib and "contrib" not in dataset_names:
        dataset_names.append("contrib")

    raw_data = None
    if create_raw:
        raw_data = create_raw_dataset(
            dataset_names=dataset_names,
            upload=upload,
        )

    if convert_data:
        create_dataset(
            dataset_names=dataset_names,
            raw_data=raw_data,
            upload=upload,
            split_ratio=split_ratio,
            remap=remap,
            split_parts=split_parts,
            batch_size=batch_size,
            file_format=file_format,
        )

    if gen_info:
        generate_infer_data(classes=get_dataset_classes(DATASET_REPO))


if __name__ == "__main__":
    app()
