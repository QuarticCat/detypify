"""Preprocess training datasets."""

import math
import re
import unicodedata
from dataclasses import dataclass
from functools import cache, partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable

import msgspec
import polars as pl
from bs4 import BeautifulSoup
from lxml import etree
from PIL import Image, ImageDraw
from tqdm import tqdm

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]


@dataclass
class MathSymbolSample:
    label: str
    symbol: Strokes


OUT_DIR = Path("build/data")
DATASET_PATH = Path("external/dataset")
MATH_WRITING_DATA_PATH = DATASET_PATH / "mathwriting-2024"
DETEXIFY_DATA_PATH = DATASET_PATH / "detexify"
CONTRIB_DATA_PATH = Path("build/dataset.json")
IMG_SIZE = 64  # px
USE_CONTRIB = False

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
}


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
    unmapped_latex_symbols: set[str] | None = None


class DetexifySymInfo(msgspec.Struct, kw_only=True, omit_defaults=True):
    command: str
    # package: str | None = None
    # mathmode: bool
    # textmode: bool
    id: str
    # css_class: str


class SymbolSegInfo(msgspec.Struct, kw_only=True, omit_defaults=True):
    sourceSampleId: str
    strokeIndices: list[int]
    label: str


class MathSymbol(msgspec.Struct, array_like=True):
    symbol: Strokes


def parse_detexify_symbol(
    key: str,
    raw_strokes: list[list[tuple[float, float, float]]],
) -> MathSymbolSample | None:
    key_to_typ = map_sym(typ_sym_info, tex_to_typ)
    typ = key_to_typ.get(key)
    if typ is None:
        return
    strokes = [
        [(float(x), float(y)) for x, y, _ in s] for s in raw_strokes if len(s) > 2
    ]
    if len(strokes) == 0:
        return
    return MathSymbolSample(typ.char, strokes)


@cache
def get_xml_parser():
    return etree.XMLParser()


def parse_inkml(filepath: Path, trace_ids: set[int] | None) -> Strokes:
    tree = etree.parse(filepath, get_xml_parser())
    root = tree.getroot()

    def _trace_to_stroke(trace: str) -> Stroke:
        """
        translate trace to stroke
        e.g. trace text:597.79 77.05 1841.0,596.41 80.84 1897.0 (a list of point(x,y,time) seperated by commas)
        """
        return [
            (float(parts[0]), float(parts[1]))
            for point_str in trace.split(",")
            for parts in point_str.strip().split()
            if len(parts) >= 2
        ]

    def _is_valid_trace(allowed_ids: set[int] | None, element: etree._Element) -> bool:
        """
        Predicate function to determine if a trace should be included.
        """
        if not allowed_ids:
            return True
            # Safely handle potential missing or non-integer IDs
        trace_id = element.attrib.get("id")
        if not trace_id:
            return False
        return int(trace_id) in allowed_ids

    return [
        _trace_to_stroke(trace.text)
        for trace in filter(
            lambda el: _is_valid_trace(trace_ids, el),
            root.iterfind("//trace"),
        )
        if trace.text
    ]


def parse_sole_symbol(
    filepath: Path,
    tex_to_typ: dict[str, TypstSymInfo],
) -> MathSymbolSample | str:
    tree = etree.parse(filepath, get_xml_parser())
    root = tree.getroot()
    tex_label = root.findtext('.//annotation[@type="label"]')
    if isinstance(tex_label, str):
        typ = tex_to_typ.get(tex_label)
        if typ is not None:
            label = typ.char
        else:
            return tex_label
    return MathSymbolSample(
        label,
        parse_inkml(filepath, None),
    )


def read_symbol_seg_anno() -> list[SymbolSegInfo]:
    """loads symbol extraction annotation of MathWrting Dataset"""
    decoder = msgspec.json.Decoder(SymbolSegInfo)
    with open(OUT_DIR / "mathwriting" / "symbols.jsonl", "rb") as f:
        symbol_annos = decoder.decode_lines(f.read())
    return symbol_annos


def extract_symbol(
    symbol_seg_anno: SymbolSegInfo,
    tex_to_typ: dict[str, TypstSymInfo],
) -> MathSymbolSample | str:
    """extract math symbol from math expression"""
    typ = tex_to_typ.get(symbol_seg_anno.label)
    if not typ:
        return symbol_seg_anno.label
    filepath = Path(
        MATH_WRITING_DATA_PATH,
        "train",
        f"{symbol_seg_anno.sourceSampleId}.inkml",
    )
    return MathSymbolSample(
        typ.char,
        parse_inkml(filepath, set(symbol_seg_anno.strokeIndices)),
    )


def parse_contrib(sample: dict[str, Any]) -> MathSymbolSample:
    sym, _strokes = (
        sample["sym"],
        sample["strokes"],
    )
    strokes: Strokes = msgspec.json.decode(_strokes)
    name_to_chr = {x.names[0]: x.char for x in get_typst_symbol_info()}
    label = name_to_chr[sym]
    return MathSymbolSample(label, strokes)


def is_invisible(c: str) -> bool:
    return unicodedata.category(c) in ["Zs", "Cc", "Cf"]


def get_typst_symbol_info() -> list[TypstSymInfo]:
    """Parse Typst symbol page to get information."""

    with open("external/typ_sym.html") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    sym_info = {}

    for li in soup.find_all("li", id=re.compile("^symbol-")):
        name = li["id"][len("symbol-") :]
        char = li["data-value"][0]
        if is_invisible(char) or li.get("data-deprecation"):
            # We don't care about invisible chars and deprecated names.
            continue
        elif char in sym_info:
            # Repeated symbols. Merge names.
            sym_info[char].names.append(name)
        else:
            latex_name, markup_shorthand, math_shorthand, alternates = (
                li.get("data-latex-name"),
                li.get("data-markup-shorthand"),
                li.get("data-math-shorthand"),
                li.get("data-alternates"),
            )

            # cursed type guards
            assert isinstance(name, str)
            assert isinstance(latex_name, str | None)
            assert isinstance(math_shorthand, str | None)
            assert isinstance(markup_shorthand, str | None)
            if isinstance(alternates, str):
                alternates = [alternates]

            # New symbols. Add to map.
            sym_info[char] = TypstSymInfo(
                char=char,
                names=[name],
                latex_name=latex_name,
                markup_shorthand=markup_shorthand,
                math_shorthand=math_shorthand,
                accent=li.get("accent") == "true",
                alternates=alternates,
            )

    return list(sym_info.values())


def map_sym(
    typ_sym_info: list[TypstSymInfo], tex_to_typ: dict[str, TypstSymInfo]
) -> dict[str, TypstSymInfo]:
    """Get a mapping from Detexify keys to Typst symbol info."""
    with open(DETEXIFY_DATA_PATH / "symbols.json", "rb") as f:
        tex_sym_info = msgspec.json.decode(f.read(), type=list[DetexifySymInfo])
    return {
        x.id: tex_to_typ[x.command] for x in tex_sym_info if x.command in tex_to_typ
    }


def normalize(strokes: Strokes) -> Strokes:
    xs = [x for s in strokes for x, _ in s]
    min_x, max_x = min(xs), max(xs)
    ys = [y for s in strokes for _, y in s]
    min_y, max_y = min(ys), max(ys)

    width = max(max_x - min_x, max_y - min_y)
    if width == 0:
        return []
    width = width * 1.2 + 20  # leave margin to avoid edge cases
    zero_x = (max_x + min_x - width) / 2
    zero_y = (max_y + min_y - width) / 2
    scale = IMG_SIZE / width

    return [
        [((x - zero_x) * scale, (y - zero_y) * scale) for x, y in s] for s in strokes
    ]


def draw_to_img(strokes: Strokes) -> Image.Image:
    image = Image.new("1", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        draw.line(stroke)
    return image


def get_dataset_info(dataset_name: str) -> DataSetInfo:
    """Load dataset metadata from the info JSON file."""
    dataset_path = DATASET_PATH / dataset_name
    info_path = dataset_path / "dataset_info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"Could not find dataset info at {info_path}")

    with open(info_path, "rb") as f:
        return msgspec.json.decode(f.read(), type=DataSetInfo)


def create_dataset(
    parse_func: Callable[[Any], MathSymbolSample | str | None],
    data: Any,
    dataset_name: str,
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> None:
    """
    Parses raw data, performs a stratified train/test/val split, and saves it as sharded Parquet files.

    This function processes the input `data` in parallel using `parse_func`.
    It also does:
        1. aggregates the results,
        2. performs a stratified shuffle split based on the labels
        3. writes the output to disk in chunks to optimize for large datasets.
        4. save dataset metadata

    Args:
        parse_func: A function that takes a raw data item and returns a `MathSymbolSample` (success),
            a `str` (indicating an unmapped LaTeX symbol error), or `None` (skip).
        data: An iterable of raw data to be processed.
        dataset_name: The name of the dataset. Output will be saved to `{OUT_DIR}/{dataset_name}`.
        split_ratio: A tuple representing the fraction of data for (Train, Test, Validation).
            Must sum to approximately 1.0. Default is (0.8, 0.1, 0.1).

    Returns:
        None

    Directory Structure:
        OUTPUT_DIR/
        └── dataset_name/
            ├── train/
            │   ├── part_000.parquet
            │   └── ...
            ├── test/
            │   └── ...
            ├── val/
            │   └── ...
            └── dataset_info.json
    """
    dataset_path = OUT_DIR / dataset_name
    train_path = dataset_path / "train"
    test_path = dataset_path / "test"
    val_path = dataset_path / "val"

    for p in [train_path, test_path, val_path]:
        p.mkdir(parents=True, exist_ok=True)

    # Accumulators
    labels_acc: list[str] = []
    data_acc: list[Strokes] = []
    unmapped_latex_symbols: set[str] = set()

    # 1. Parse Data
    with Pool() as exec:
        results = exec.map(parse_func, data)
        for result in tqdm(results, desc=f"Processing dataset {dataset_name}."):
            if not result:
                continue
            elif isinstance(result, str):
                unmapped_latex_symbols.add(result)
            else:
                labels_acc.append(result.label)
                data_acc.append(result.symbol)

    # 2. Prepare Base LazyFrame with Partitioning Info
    # We calculate the cumulative thresholds for splitting
    train_r, test_r, _ = split_ratio
    # Threshold 1: Boundary between Train and Test
    t1 = train_r
    # Threshold 2: Boundary between Test and Val
    t2 = train_r + test_r

    lf = pl.DataFrame({"label": labels_acc, "data": data_acc})

    #
    base_lf = (
        lf.sample(shuffle=True, seed=42)
        .with_columns(
            [
                pl.len().over("label").alias("n"),
                pl.int_range(0, pl.len()).over("label").alias("idx"),
            ]
        )
        .lazy()
    )

    # 3. Define Partitions (Lazy)
    # Train: [0, t1)
    train_lf = (
        base_lf.filter(pl.col("idx") < (pl.col("n") * t1))
        .drop(["n", "idx"])
        .sort("label")
    )

    # Test: [t1, t2)
    test_lf = (
        base_lf.filter(
            (pl.col("idx") >= (pl.col("n") * t1)) & (pl.col("idx") < (pl.col("n") * t2))
        )
        .drop(["n", "idx"])
        .sort("label")
    )

    # Validation: [t2, end]
    val_lf = (
        base_lf.filter(pl.col("idx") >= (pl.col("n") * t2))
        .drop(["n", "idx"])
        .sort("label")
    )

    # 4. Helper to Write Shards
    def _write_shards(lazy_frame: pl.LazyFrame, output_dir: Path):
        # We collect the specific split to memory to slice it efficiently.
        # Since input `labels_acc` fits in memory, a subset `df` will also fit.
        df = lazy_frame.collect()
        total_rows = len(df)
        batch_size = 1000

        if total_rows == 0:
            return

        # Calculate padding for filenames (e.g., part_001.parquet)
        num_shards = math.ceil(total_rows / batch_size)
        pad_width = len(str(num_shards))

        for i, start_idx in enumerate(range(0, total_rows, batch_size)):
            # Slice is zero-copy in Polars
            chunk = df.slice(start_idx, batch_size)
            filename = f"part_{str(i).zfill(pad_width)}.parquet"
            chunk.write_parquet(output_dir / filename, compression="zstd")

    # 5. Execute Writes
    _write_shards(train_lf, train_path)
    _write_shards(test_lf, test_path)
    _write_shards(val_lf, val_path)

    # 6. Metadata
    stats_df = lf["label"].value_counts()
    class_counters = dict(zip(stats_df["label"], stats_df["len"]))

    with open(dataset_path / "dataset_info.json", "wb") as f:
        dataset_info = DataSetInfo(
            name=dataset_name,
            total_count=len(labels_acc),
            class_count=class_counters,
            unmapped_latex_symbols=unmapped_latex_symbols
            if unmapped_latex_symbols
            else None,
        )
        f.write(msgspec.json.encode(dataset_info))


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)

    # Get symbol info.
    typ_sym_info = get_typst_symbol_info()
    tex_to_typ = {s.latex_name: s for s in typ_sym_info if s.latex_name is not None}
    key_to_typ = map_sym(typ_sym_info, tex_to_typ)
    with open(f"{OUT_DIR}/symbols.json", "wb") as f:
        f.write(msgspec.json.encode(typ_sym_info))

    # Parse math symbols from detexify dataset
    with open(DETEXIFY_DATA_PATH / "detexify.json", "rb") as f:
        detexify_raw_data = msgspec.json.decode(f.read())
        create_dataset(
            parse_func=partial(parse_detexify_symbol, key_to_typ),
            data=detexify_raw_data,
            dataset_name="detexify",
        )

    # Parse MathWrting Dataset sole symboles
    create_dataset(
        parse_func=partial(parse_sole_symbol, tex_to_typ),
        data=Path(MATH_WRITING_DATA_PATH, "symbols").glob("*.inkml"),
        dataset_name="mathwriting_symbols",
    )
    # Parse extracted symbols from MathWrting Dataset
    seg_annos = read_symbol_seg_anno()
    create_dataset(
        parse_func=partial(extract_symbol, tex_to_typ),
        data=seg_annos,
        dataset_name="mathwriting_extracted",
    )

    # Parse contributed data
    if USE_CONTRIB:
        with open(CONTRIB_DATA_PATH, "rb") as f:
            samples = msgspec.json.decode(f.read())[0]["results"]
            create_dataset(
                parse_func=parse_contrib, data=samples, dataset_name="contrib"
            )
