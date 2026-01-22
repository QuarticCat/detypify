"""Preprocess training datasets."""

import math
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path
from shutil import rmtree as rmdir
from typing import Literal, cast
from urllib.request import urlretrieve

import msgspec
import polars as pl
from bs4 import BeautifulSoup
from lxml import etree
from msgspec import json
from PIL import Image, ImageDraw

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]
type Datasets = Literal["mathwriting", "detexify", "contrib"]


# constants
DATASET_ROOT = Path("build/dataset")
EXTERNAL_DATA_PATH = Path("external/dataset")
MATH_WRITING_DATA_PATH = EXTERNAL_DATA_PATH / "mathwriting"
DETEXIFY_DATA_PATH = EXTERNAL_DATA_PATH / "detexify"
CONTRIB_DATA = Path("build/dataset.json")
USE_CONTRIB = False
IMG_SIZE = 224  # px

# Missing mappings in the Typst symbol page.
TEX_TO_TYP = {
    # Double Struck Capital Letters
    "\\mathds{A}": "AA",
    "\\mathds{B}": "BB",
    "\\mathds{C}": "CC",
    "\\mathds{D}": "DD",
    "\\mathds{E}": "EE",
    "\\mathds{F}": "FF",
    "\\mathds{G}": "GG",
    "\\mathds{H}": "HH",
    "\\mathds{I}": "II",
    "\\mathds{J}": "JJ",
    "\\mathds{K}": "KK",
    "\\mathds{L}": "LL",
    "\\mathds{M}": "MM",
    "\\mathds{N}": "NN",
    "\\mathds{O}": "OO",
    "\\mathds{P}": "PP",
    "\\mathds{Q}": "QQ",
    "\\mathds{R}": "RR",
    "\\mathds{S}": "SS",
    "\\mathds{T}": "TT",
    "\\mathds{U}": "UU",
    "\\mathds{V}": "VV",
    "\\mathds{W}": "WW",
    "\\mathds{X}": "XX",
    "\\mathds{Y}": "YY",
    "\\mathds{Z}": "ZZ",
    # Greek Capital Letters
    "\\Alpha": "Alpha",
    "\\Beta": "Beta",
    "\\Gamma": "Gamma",
    "\\Delta": "Delta",
    "\\Epsilon": "Epsilon",
    "\\Zeta": "Zeta",
    "\\Eta": "Eta",
    "\\Theta": "Theta",  # TODO: no Theta.alt
    "\\Iota": "Iota",
    "\\Kappa": "Kappa",
    "\\Lambda": "Lambda",
    "\\Mu": "Mu",
    "\\Nu": "Nu",
    "\\Xi": "Xi",
    "\\Omicron": "Omicron",
    "\\Pi": "Pi",
    "\\Rho": "Rho",
    "\\Sigma": "Sigma",
    "\\Tau": "Tau",
    "\\Upsilon": "Upsilon",
    "\\Phi": "Phi",
    "\\Chi": "Chi",
    "\\Psi": "Psi",
    "\\Omega": "Omega",
    # Greek Small Letters
    "\\alpha": "alpha",
    "\\beta": "beta",
    "\\gamma": "gamma",
    "\\delta": "delta",
    "\\varepsilon": "epsilon",
    "\\epsilon": "epsilon.alt",
    "\\zeta": "zeta",
    "\\eta": "eta",
    "\\theta": "theta",
    "\\vartheta": "theta.alt",
    "\\iota": "iota",
    "\\kappa": "kappa",
    "\\varkappa": "kappa.alt",
    "\\lambda": "lambda",
    "\\mu": "mu",
    "\\nu": "nu",
    "\\xi": "xi",
    "\\omicron": "omicron",
    "\\pi": "pi",
    "\\varpi": "pi.alt",
    "\\rho": "rho",
    "\\varrho": "rho.alt",
    "\\sigma": "sigma",
    "\\varsigma": "sigma.alt",
    "\\tau": "tau",
    "\\upsilon": "upsilon",
    "\\varphi": "phi",
    "\\phi": "phi.alt",
    "\\chi": "chi",
    "\\psi": "psi",
    "\\omega": "omega",
    # Hebrew Letters
    "\\aleph": "aleph",  # TODO: no beth, daleth, gimel
    # Others
    "\\&": "amp",
    "\\#": "hash",
    "\\%": "percent",
    "\\{": "brace.l",
    "\\}": "brace.r",
    "\\--": "dash.en",
    "\\---": "dash.em",
    "\\colon": "colon",
    "\\degree": "degree",
    "\\copyright": "copyright",
    "\\textcircledP": "copyright.sound",
    "\\textreferencemark": "refmark",
    "\\textperthousand": "permille",
    "\\simeq": "tilde.eq",
    "\\circlearrowleft": "arrow.ccw",
    "\\circlearrowright": "arrow.cw",
    "\\dashleftarrow": "arrow.l.dashed",
    "\\dashrightarrow": "arrow.r.dashed",
    "\\lightning": "arrow.zigzag",
    "\\circ": "compose",
    "\\bowtie": "join",
    "\\MVAt": "at",
    "\\EUR": "euro",
    "\\blacksquare": "qed",
    "\\emptyset": "emptyset",
    "\\|": "bar.v.double",
    "\\iff": "arrow.l.r.double.long",
    "\\bullet": "bullet.op",
}


# Structs
class TypstSymInfo(msgspec.Struct, kw_only=True, omit_defaults=True):
    char: str
    names: list[str]
    latex_name: str | None = None
    markup_shorthand: str | None = None
    math_shorthand: str | None = None
    accent: bool = False
    alternates: list[str] | None = None


class DataSetInfo(msgspec.Struct, kw_only=True, omit_defaults=True):
    name: str
    total_count: int
    class_count: dict[str, int]
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
def normalize(strokes: Strokes, target_size: int) -> Strokes:
    """Normalizes the strokes to fit within a target box while preserving aspect ratio.

    The normalization process involves:
    1. Finding the bounding box of the strokes.
    2. Centering the strokes.
    3. Scaling the strokes to fit within the `target_size` with a margin.

    Args:
        strokes: A list of strokes, where each stroke is a list of (x, y) points.
        target_size: The size of the output bounding box (square).

    Returns:
        The normalized strokes.
    """
    xs = [x for s in strokes for x, _ in s]
    min_x, max_x = min(xs), max(xs)
    ys = [y for s in strokes for _, y in s]
    min_y, max_y = min(ys), max(ys)

    width = max(max_x - min_x, max_y - min_y)
    if width == 0:
        return []
    width = width * 1.1 + 10  # leave margin to avoid edge cases
    zero_x = (max_x + min_x - width) / 2
    zero_y = (max_y + min_y - width) / 2
    scale = target_size / width

    return [
        [((x - zero_x) * scale, (y - zero_y) * scale) for x, y in s] for s in strokes
    ]


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

    page_path = Path("external/typ_sym.html")
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


def draw_to_img(strokes: Strokes, size: int, resize: bool = True) -> Image.Image:
    """Draws strokes onto a PIL Image.

    Args:
        strokes: A list of strokes to draw.
        size: The width and height of the output image.
        resize: Whether to normalize the strokes before drawing.

    Returns:
        A binary PIL Image with the drawn strokes, white strokes on black background
    """
    if resize:
        strokes = normalize(strokes, size)
    image = Image.new("1", (size, size), "black")
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        draw.line(stroke, fill="white")
    return image


def get_dataset_info(dataset_name: str) -> DataSetInfo:
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
        return msgspec.json.decode(f.read(), type=DataSetInfo)


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
    tex_to_typ |= {k: name_to_typ[v] for k, v in TEX_TO_TYP.items()}
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
    root = etree.parse(filepath, parser=get_xml_parser()).getroot()
    namespace = {"ink": "http://www.w3.org/2003/InkML"}
    tex_label = root.findtext(".//ink:annotation[@type='label']", namespaces=namespace)
    if not tex_label:
        return None
    if isinstance(tex_label, str):
        tex_to_typ = map_tex_typ()
        typ = tex_to_typ.get(tex_label)
        if typ is not None:
            label = typ.char
        # missing mapping, return current label
        else:
            return tex_label
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
                .list.eval(pl.element().list.eval(pl.element().arr.head(2))),
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

    name_to_chr = {x.names[0]: x.char for x in get_typst_symbol_info()}
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
    dataset_name: Literal["mathwriting", "detexify", "contrib"],
    file_format: Literal["vortex", "parquet"] = "parquet",
    split_parts: bool = True,
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    batch_size: int = 2000,
) -> None:
    """Orchestrates the creation of a math symbol dataset.

    1. Loads symbol mappings using the specific project logic.
    2. Dispatches data loading to specific construct_* functions.
    3. Performs a stratified train/test/val split.
    4. Saves data as sharded files (Parquet/Vortex) and writes metadata.

    Args:
        dataset_name: The name of the dataset to create.
        file_format: The output file format ("parquet" or "vortex").
        split_parts: Whether to split the dataset into multiple shards.
        split_ratio: A tuple defining the ratio for (train, test, val) splits.
        batch_size: The number of rows per shard.
    """
    print(f"--- Creating Dataset: {dataset_name} ---")

    # ---------------------------------------------------------
    # 2. Dispatcher: Construct LazyFrame & Unmapped Set
    # ---------------------------------------------------------
    lf: pl.LazyFrame
    unmapped_symbols: set[str] | None = None

    if dataset_name == "mathwriting":
        lf, unmapped_symbols = construct_mathwriting_df()

    elif dataset_name == "detexify":
        lf, unmapped_symbols = construct_detexify_df()

    elif dataset_name == "contrib":
        # Requires: typ_sym_info list (to map by 'names')
        lf = construct_contribute_df()

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # ---------------------------------------------------------
    # 3. Preparation & Splitting Logic
    # ---------------------------------------------------------
    dataset_path = DATASET_ROOT / dataset_name

    # Clean/Create Directories
    if dataset_path.exists():
        rmdir(dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)

    train_path = dataset_path / "train"
    test_path = dataset_path / "test"
    val_path = dataset_path / "val"
    for p in [train_path, test_path, val_path]:
        p.mkdir(exist_ok=True)

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
        .sample(fraction=1.0, shuffle=True, seed=42)
        .lazy()
        .with_columns(
            [
                pl.len().over("label").alias("n"),
                pl.int_range(0, pl.len()).over("label").alias("idx"),
            ]
        )
    )

    # Define Partitions (Lazy)
    train_lf = base_lf.filter(pl.col("idx") < (pl.col("n") * t1))
    test_lf = base_lf.filter(
        (pl.col("idx") >= (pl.col("n") * t1)) & (pl.col("idx") < (pl.col("n") * t2))
    )
    val_lf = base_lf.filter(pl.col("idx") >= (pl.col("n") * t2))

    # Cleanup temporary columns
    train_lf = train_lf.drop(["n", "idx"]).sort("label")
    test_lf = test_lf.drop(["n", "idx"]).sort("label")
    val_lf = val_lf.drop(["n", "idx"]).sort("label")

    def _write_to_file(df: pl.DataFrame, path: Path):
        if file_format == "vortex":
            import vortex as vx

            vx.io.write(vx.compress(vx.array(df.to_arrow())), str(path))
        else:
            df.write_parquet(path, compression="zstd", compression_level=19)

    def _write_shards(lazy_frame: pl.LazyFrame, output_dir: Path):
        df = lazy_frame.collect()
        total_rows = len(df)
        if total_rows == 0:
            return

        num_shards = math.ceil(total_rows / batch_size)
        pad_width = len(str(num_shards))

        print(
            f"  -> Writing {total_rows} rows to \
            {output_dir.name} ({num_shards} shards)..."
        )

        for i, start_idx in enumerate(range(0, total_rows, batch_size)):
            chunk = df.slice(start_idx, batch_size)
            filename = f"part_{str(i).zfill(pad_width)}.{file_format}"
            _write_to_file(chunk, output_dir / filename)

    if split_parts:
        _write_shards(train_lf, train_path)
        _write_shards(test_lf, test_path)
        _write_shards(val_lf, val_path)
    else:
        print("  -> Writing single files...")
        _write_to_file(train_lf.collect(), train_path / f"data.{file_format}")
        _write_to_file(test_lf.collect(), test_path / f"data.{file_format}")
        _write_to_file(val_lf.collect(), val_path / f"data.{file_format}")

    # ---------------------------------------------------------
    # 5. Metadata Generation
    # ---------------------------------------------------------
    print("  -> Generating metadata...")
    # Use base_lf (materialized view logic) for fast stats
    stats_df = (
        base_lf.select(pl.col("label")).collect().get_column("label").value_counts()
    )
    class_counters = dict(zip(stats_df["label"], stats_df["count"]))
    total_count = int(stats_df["count"].sum())

    dataset_info = DataSetInfo(
        name=dataset_name,
        total_count=total_count,
        class_count=class_counters,
        unmapped=unmapped_symbols if unmapped_symbols else None,
    )

    with (dataset_path / "dataset_info.json").open("wb") as f:
        f.write(json.format(json.encode(dataset_info)))

    print(f"--- Done. Dataset saved to {dataset_path} ---")


if __name__ == "__main__":
    DATASET_ROOT.mkdir(exist_ok=True)

    # Get symbol info.
    typ_sym_info = get_typst_symbol_info()
    name_to_chr = {x.names[0]: x.char for x in typ_sym_info}

    symbols_info_path = DATASET_ROOT / "symbols.json"
    if not symbols_info_path.exists():
        with symbols_info_path.open("wb") as f:
            f.write(msgspec.json.encode(typ_sym_info))

    create_dataset(dataset_name="detexify")

    # Parse MathWrting Dataset sole symboles
    create_dataset(dataset_name="mathwriting", split_parts=False)

    # Parse contributed data
    # if USE_CONTRIB:
    #     with CONTRIB_DATA.open("rb") as f:
    #         samples = msgspec.json.decode(f.read())[0]["results"]
    #         create_dataset(
    #             dataset_name="contrib",
    #         )
