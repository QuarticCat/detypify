"""Preprocess training datasets, helping functions and related constants/types."""

import math
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from os import process_cpu_count
from pathlib import Path
from shutil import rmtree as rmdir
from typing import Literal, cast
from urllib.request import urlretrieve

import cv2
import msgspec
import numpy as np
import polars as pl
from bs4 import BeautifulSoup
from datasets import ClassLabel, Dataset, DatasetInfo, Features, LargeList, List, Value
from lxml import etree
from msgspec import json

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]
type DataSetName = Literal["mathwriting", "detexify", "contrib"]
type SplitName = Literal["train", "test", "val"]


# Constants
# Data
DATASET_ROOT = Path("build/dataset")
EXTERNAL_DATA_PATH = Path("build/raw_data")
MATH_WRITING_DATA_PATH = EXTERNAL_DATA_PATH / "mathwriting"
DETEXIFY_DATA_PATH = EXTERNAL_DATA_PATH / "detexify"
USE_CONTRIB = False
CONTRIB_DATA = Path("build/dataset.json")
DATASET_REPO = "Cloud0310/detypify-datasets"
UPLOAD = True
# Processing
IMG_SIZE = 224  # px
# Extra latex to typst mapping.
TEX_TO_TYP_PATH = Path(__file__).parent / "tex_to_typ.json"


# Structs
class TypstSymInfo(msgspec.Struct, kw_only=True, omit_defaults=True):
    char: str
    names: list[str]
    latex_name: str | None = None
    markup_shorthand: str | None = None
    math_shorthand: str | None = None
    accent: bool = False
    alternates: list[str] | None = None


class MathWritingDatasetInfo(msgspec.Struct, kw_only=True, omit_defaults=True):
    name: str
    unmapped: set[str] | None = None


class DetexifySymInfo(msgspec.Struct, kw_only=True, omit_defaults=True):
    command: str
    # package: str | None = None
    # mathmode: bool
    # textmode: bool
    id: str
    # css_class: str


class MathSymbolSample(msgspec.Struct):
    label: str
    symbol: Strokes


# Helper functions


def is_invisible(c: str) -> bool:
    return unicodedata.category(c) in ["Zs", "Cc", "Cf"]


@cache
def get_typst_symbol_info() -> list[TypstSymInfo]:
    """Parses the Typst symbol page to extract symbol information.

    Retrieves the HTML content from the Typst documentation (downloading it if
    necessary) and parses it to find symbol names, characters, and their LaTeX
    equivalents.

    Returns:
        A list of `TypstSymInfo` objects containing details for each symbol.
    """

    page_path = EXTERNAL_DATA_PATH / "typ_sym.html"
    if not page_path.exists():
        urlretrieve("https://typst.app/docs/reference/symbols/sym/", page_path)
    with page_path.open() as f:
        soup = BeautifulSoup(f.read(), "lxml")
    sym_info = {}

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

    return list(sym_info.values())


def rasterize_strokes(strokes: Strokes, output_size: int = IMG_SIZE) -> np.ndarray:
    """
    Normalizes vector strokes and rasterizes them into a binary NumPy array.

    Args:
        strokes: List of strokes, where each stroke is a list of (x, y) coordinates.
        output_size: The width and height of the output square grid.

    Returns:
        np.ndarray: A (size, size) uint8 array.
                    Background is 0 (black), Strokes are 255 (white).
    """
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
    scale = target_size / width if width > 1e-6 else 1.0
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

    thickness_factor = 30  # increase for better feature extraction
    thickness = max(1, output_size // thickness_factor)
    cv2.polylines(
        canvas, normalized_strokes, isClosed=False, color=255, thickness=thickness
    )

    return canvas


def get_dataset_info(dataset_name: str) -> MathWritingDatasetInfo:
    """Load dataset metadata from the info JSON file.

    Args:
        dataset_name: The name of the dataset (e.g., "detexify").

    Returns:
        A `DataSetInfo` object containing the metadata.

    Raises:
        FileNotFoundError: If the info file does not exist.
    """
    dataset_path = DATASET_ROOT / dataset_name
    info_path = dataset_path / "dataset_info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"Could not find dataset info at {info_path}")

    with info_path.open("rb") as f:
        return msgspec.json.decode(f.read(), type=MathWritingDatasetInfo)


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
        manual_mapping = msgspec.json.decode(f.read(), type=dict[str, str])

    tex_to_typ |= {k: name_to_typ[v] for k, v in manual_mapping.items()}
    return tex_to_typ


@cache
def get_xml_parser():
    return etree.XMLParser()


@cache
def map_sym() -> dict[str, TypstSymInfo]:
    """Get a mapping from Detexify keys to Typst symbol info.

    Reads the Detexify symbol list and maps them to Typst symbols using the
    TeX command mapping.

    Returns:
        A dictionary mapping Detexify IDs to Typst symbol info.
    """
    with (DETEXIFY_DATA_PATH / "symbols.json").open("rb") as f:
        tex_sym_info = msgspec.json.decode(f.read(), type=list[DetexifySymInfo])
    tex_to_typ = map_tex_typ()
    return {
        x.id: tex_to_typ[x.command] for x in tex_sym_info if x.command in tex_to_typ
    }


def parse_inkml_symbol(
    filepath: Path,
) -> MathSymbolSample | str | None:
    """Parses a single InkML file to extract the label and stroke data.

    Args:
        filepath: Path to the .inkml file.

    Returns:
        A `MathSymbolSample` if successful.
        A string (the TeX label) if the label could not be mapped to a Typst symbol.
        None if no label is found.
    """
    # parsing
    root = etree.parse(filepath, parser=get_xml_parser()).getroot()
    namespace = {"ink": "http://www.w3.org/2003/InkML"}
    tex_label = root.findtext(".//ink:annotation[@type='label']", namespaces=namespace)

    # couldn't find data, return None
    if not tex_label:
        return None
    tex_to_typ = map_tex_typ()
    typ = tex_to_typ.get(tex_label)
    # found mapped typst symbol
    if typ:
        label = typ.char
        return MathSymbolSample(
            label,
            [
                [
                    (float(x), float(y))
                    for x, y, _ in (
                        # point tuple[int, int, int]
                        point_str.split()
                        for point_str in trace.text.split(",")
                        if len(point_str.split()) == 3
                    )
                ]
                for trace in root.iterfind(".//ink:trace", namespaces=namespace)
                if trace.text
            ],
        )
    # missing mapping, return current label as unmap
    return tex_label


# Dataset creating functions
def construct_detexify_df() -> tuple[pl.LazyFrame, set[str] | None]:
    """Constructs a Polars LazyFrame for the Detexify dataset.

    Reads the raw JSON data, maps the keys to Typst symbols, filters out
    unmapped or empty samples, and formats the strokes.

    Returns:
        A tuple containing:
            - The processed LazyFrame with 'label' and 'strokes' columns.
            - A set of unmapped keys (commands that couldn't be mapped to Typst).
    """
    key_to_typ = map_sym()
    with (DETEXIFY_DATA_PATH / "detexify.json").open("rb") as f:
        # Schema: list of (key, strokes)
        raw_strokes_t = list[tuple[str, list[list[tuple[float, float, float]]]]]
        data = msgspec.json.decode(f.read(), type=raw_strokes_t)

    raw_data_schema = {
        "key": pl.String,
        # use float32 for mem saving
        "strokes": pl.List(pl.List(pl.Array(pl.Float32, 3))),
    }
    raw_lf = pl.DataFrame(data, schema=raw_data_schema, orient="row").lazy()
    del data

    # 2. Prepare Mapping

    mapping_lf = pl.DataFrame(
        {
            "key": list(key_to_typ.keys()),
            "label": [typ.char for typ in key_to_typ.values()],
        }
    ).lazy()

    processed_lf = raw_lf.join(mapping_lf, on="key", how="left")

    # Extract unmapped keys before filtering
    unmapped_keys = set(
        processed_lf.filter(pl.col("label").is_null())
        .select("key")
        .unique()
        .collect()
        .get_column("key")
        .to_list()
    )
    final_lf = (
        processed_lf.filter(pl.col("label").is_not_null())
        .select(
            [
                pl.col("label"),
                pl.col("strokes")
                # Drop time (keep only x, y)
                .list.eval(
                    pl.element().list.eval(pl.element().arr.head(2).list.to_array(2))
                ),
            ]
        )
        # 3. Drop empty samples
        .filter(pl.col("strokes").list.len() > 0)
    )

    return final_lf, unmapped_keys


def construct_mathwriting_df() -> tuple[pl.LazyFrame, set[str] | None]:
    """Constructs a Polars LazyFrame for the MathWriting dataset.

    Parses InkML files in parallel to extract strokes and labels.
    Returns:
        A tuple containing:
            - The processed LazyFrame with 'label' and 'strokes' columns.
            - A set of unmapped labels (TeX commands that couldn't be mapped).
    """
    label_acc = []
    data_acc = []
    unmapped: set[str] = set()

    with ProcessPoolExecutor() as executor:
        results = executor.map(
            parse_inkml_symbol,
            MATH_WRITING_DATA_PATH.glob("*.inkml"),
            chunksize=500,
        )
        for result in results:
            match result:
                case None:
                    continue
                case str():
                    unmapped.add(result)
                case MathSymbolSample():
                    label_acc.append(result.label)
                    data_acc.append(result.symbol)

    del results
    pl_schema = {
        "label": pl.String,
        "strokes": pl.List(pl.List(pl.Array(pl.Float32, 2))),
    }

    final_lf = pl.DataFrame(
        {"label": label_acc, "strokes": data_acc}, schema=pl_schema
    ).lazy()
    del label_acc, data_acc

    return final_lf, unmapped


# WIP
def construct_contribute_df() -> pl.LazyFrame:
    # 1. Load Data
    with CONTRIB_DATA.open("rb") as f:
        # Schema: list of dicts {sym: str, strokes: json_string}
        data = msgspec.json.decode(f.read(), type=list[dict[str, str]])

    typ_sym_info = get_typst_symbol_info()
    name_to_chr = {name: s.char for s in typ_sym_info for name in s.names}
    # 2. Decode JSON Strings & Join Labels Locally

    mapping_lf = pl.DataFrame(
        {
            "sym": list(name_to_chr.keys()),
            "label": list(name_to_chr.values()),
        }
    ).lazy()

    processed_lf = (
        pl.DataFrame(data)
        .lazy()
        # Decode strokes
        .with_columns(
            pl.col("strokes").map_elements(
                msgspec.json.decode,
                return_dtype=pl.List(pl.List(pl.Array(pl.Float32, 2))),
            )
        )
        # Join Labels
        .join(mapping_lf, on="sym", how="left")
    )
    return (
        processed_lf.filter(pl.col("label").is_not_null())
        .select(
            [
                pl.col("label"),
                pl.col("strokes")
                # Drop time (keep only x, y)
                .list.eval(pl.element().list.eval(pl.element().arr.head(2))),
            ]
        )
        # 3. Drop empty samples
        .filter(pl.col("strokes").list.len() > 0)
    )


def create_dataset(
    dataset_names: list[DataSetName],
    upload: bool = True,
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    **kwargs,
) -> None:
    """Orchestrates the creation of a math symbol dataset.

    1. Loads symbol mappings using the specific project logic.
    2. Dispatches data loading to specific construct_* functions.
    3. Performs a stratified train/test/val split.
    4. Saves data as sharded files (Parquet/Vortex) and writes metadata.

    Args:
        dataset_name: The name of the dataset to create.
        split_ratio: A tuple defining the ratio for (train, test, val) splits.
        upload: whether or not uploading dataset to hugggingface.
        kwargs: valid only when not uploading.
            split_parts: Whether to split the dataset into multiple shards.
            batch_size: The number of rows per shard.
            file_format: The output file format ("parquet" or "vortex").
    """

    print(f"--- Creating Datasets: {','.join(dataset_names)} ---")

    # ---------------------------------------------------------
    # 2. Dispatcher: Construct LazyFrame & Unmapped Set
    # ---------------------------------------------------------

    lfs = []
    unmapped: dict[DataSetName, set[str]] = {}
    for dataset_name in dataset_names:
        match dataset_name:
            case "mathwriting":
                math_writing_lf, unmapped_symbols = construct_mathwriting_df()
                lfs.append(math_writing_lf)
                if unmapped_symbols:
                    unmapped[dataset_name] = unmapped_symbols
            case "detexify":
                detexify_lf, unmapped_symbols = construct_detexify_df()
                lfs.append(detexify_lf)
                if unmapped_symbols:
                    unmapped[dataset_name] = unmapped_symbols
            case "contrib":
                # Requires: typ_sym_info list (to map by 'names')
                contrib_lf = construct_contribute_df()
                lfs.append(contrib_lf)
            case _:
                raise ValueError(f"Unknown dataset name: {dataset_name}")
    lf = pl.concat(lfs)

    # ---------------------------------------------------------
    # 3. Preparation & Splitting Logic
    # ---------------------------------------------------------
    dataset_path = DATASET_ROOT
    split_names: list[SplitName] = ["train", "test", "val"]

    # Clean/Create Directories
    if dataset_path.exists():
        rmdir(dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Calculate Split Thresholds
    train_r, test_r, _ = split_ratio
    t1 = train_r
    t2 = train_r + test_r

    print("  -> Shuffling and splitting data...")

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
    print("  -> Generating metadata...")
    # Use base_lf (materialized view logic) for fast stats
    stats_df = (
        base_lf.select(pl.col("label"))
        .collect()
        .get_column("label")
        .value_counts()
        .sort("label")
    )
    global_labels = stats_df["label"].to_list()

    # Split and shuffle data
    train_lf = base_lf.filter(pl.col("idx") < (pl.col("n") * t1))
    test_lf = base_lf.filter(
        (pl.col("idx") >= (pl.col("n") * t1)) & (pl.col("idx") < (pl.col("n") * t2))
    )
    val_lf = base_lf.filter(pl.col("idx") >= (pl.col("n") * t2))

    train_lf = train_lf.drop(["n", "idx"]).sort("label").collect()
    test_lf = test_lf.drop(["n", "idx"]).sort("label").collect()
    val_lf = val_lf.drop(["n", "idx"]).sort("label").collect()

    if upload:
        global_features: Features = Features(
            {
                "label": ClassLabel(names=global_labels),
                "strokes": LargeList(LargeList(List(Value("float32")))),
            }
        )  # type: ignore

        def _upload_to_huggingface(
            split: SplitName,
            data_frame: pl.DataFrame,
        ):
            def encode_labels(batch):
                class_feature = global_features["label"]
                batch["label"] = [
                    class_feature.str2int(label) for label in batch["label"]
                ]
                return batch

            description = (
                "Detypify dataset, "
                "composed by mathwriting, detexify and contributed datasets"
            )

            dataset_info = DatasetInfo(description=description)
            dataset = cast(
                "Dataset",
                Dataset.from_polars(data_frame, info=dataset_info)
                .map(encode_labels, batched=True)
                .cast(features=global_features),
            )
            dataset.push_to_hub(
                repo_id=DATASET_REPO,
                config_name=dataset_name,
                num_proc=process_cpu_count() if process_cpu_count() else 1,
                split=split,
            )

        for df, split in zip([train_lf, test_lf, val_lf], split_names):
            print(f"  -> Uploading split: {split}... to huggingface.")
            _upload_to_huggingface(split, df)
        print(f"--- Done. Dataset uploaded to {DATASET_REPO} ---")
    else:
        split_parts: bool = kwargs.get("split_parts", False)
        batch_size: int = kwargs.get("batch_size", 2000)
        file_format: Literal["vortex", "parquet"] = kwargs.get("file_format", "parquet")

        def _write_to_file(df: pl.DataFrame, path: Path):
            if file_format == "vortex":
                import vortex as vx

                vx.io.write(vx.compress(vx.array(df.to_arrow())), str(path))
            else:
                df.write_parquet(path, compression="zstd", compression_level=19)

        def _write_shards(df: pl.DataFrame, output_dir: Path):
            total_rows = len(df)
            if total_rows == 0:
                return

            num_shards = math.ceil(total_rows / batch_size)
            pad_width = len(str(num_shards))

            print(
                f"  -> Writing {total_rows} rows to"
                f"{output_dir.name} ({num_shards} shards)."
            )

            for i, start_idx in enumerate(range(0, total_rows, batch_size)):
                chunk = df.slice(start_idx, batch_size)
                filename = f"part_{str(i).zfill(pad_width)}.{file_format}"
                _write_to_file(chunk, output_dir / filename)

        for df, split in zip([train_lf, test_lf, val_lf], split_names):
            output_dir = dataset_path / split
            output_dir.mkdir(exist_ok=True)
            if split_parts:
                _write_shards(df, output_dir)
            else:
                print(
                    f"  -> Writing {len(df)} rows to{output_dir.name} as single file."
                )
                _write_to_file(df, output_dir / f"data.{file_format}")
        print(f"--- Done. Dataset saved to {dataset_path} ---")

    for dataset_name, symbols in unmapped.items():
        dataset_path = DATASET_ROOT / dataset_name
        dataset_path.mkdir(exist_ok=True)
        dataset_info = MathWritingDatasetInfo(
            name=dataset_name,
            unmapped=symbols if symbols else None,
        )
        with (dataset_path / "dataset_info.json").open("wb") as f:
            f.write(json.format(json.encode(dataset_info)))


if __name__ == "__main__":
    DATASET_ROOT.mkdir(exist_ok=True)

    # Get symbol info.
    typ_sym_info = get_typst_symbol_info()
    name_to_chr = {x.names[0]: x.char for x in typ_sym_info}

    symbols_info_path = DATASET_ROOT / "symbols.json"
    if not symbols_info_path.exists():
        with symbols_info_path.open("wb") as f:
            f.write(msgspec.json.encode(typ_sym_info))

    # Use streaming engine as default for less mem and speed
    pl.Config.set_engine_affinity("streaming")
    dataset_names: list[DataSetName] = ["detexify", "mathwriting"]
    # WIP
    if USE_CONTRIB:
        dataset_names.append("contrib")
    create_dataset(dataset_names=dataset_names)
