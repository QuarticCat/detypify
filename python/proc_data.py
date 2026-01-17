"""Preprocess training datasets."""

import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache, partial
from os import process_cpu_count
from pathlib import Path
from typing import Any, Callable

import msgspec
from bs4 import BeautifulSoup
from lxml import etree
from PIL import Image, ImageDraw
from rocksdict import (
    DBCompressionType,
    Options,
    Rdict,
    WriteBatch,
    WriteOptions,
)

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]

OUT_DIR = Path("build/data")
MATH_WRITING_DATASET_PATH = Path("external/mathwriting")
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
    alternates: list[str] = []


class DataSetInfo(msgspec.Struct, kw_only=True):
    name: str
    total_count: int
    class_count: dict[str, int]


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


@dataclass
class MathSymbolSample:
    label: str
    symbol: MathSymbol


def parse_detexify_symbol(
    key_to_typ: dict[str, TypstSymInfo],
    key: str,
    raw_strokes: list[list[tuple[float, float, float]]],
) -> MathSymbolSample | None:
    typ = key_to_typ.get(key)
    if typ is None:
        return
    strokes = [
        [(float(x), float(y)) for x, y, _ in s]
        for s in raw_strokes
        if len(s) > 1
    ]
    if len(strokes) == 0:
        return
    return MathSymbolSample(typ.char, MathSymbol(strokes))


@cache
def get_xml_parser():
    return etree.XMLParser()


def parse_inkml(filepath: Path, trace_ids: set[int] | None) -> MathSymbol:
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

    def _is_valid_trace(
        allowed_ids: set[int] | None, element: etree._Element
    ) -> bool:
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

    return MathSymbol(
        [
            _trace_to_stroke(trace.text)
            for trace in filter(
                lambda el: _is_valid_trace(trace_ids, el),
                root.iterfind("//trace"),
            )
            if trace.text
        ]
    )


def parse_sole_symbol(
    filepath: Path,
    tex_to_typ: dict[str, TypstSymInfo],
) -> MathSymbolSample | None:
    """parse math symbol inkml file directly"""
    tree = etree.parse(filepath)
    root = tree.getroot()
    tex_label = root.findtext('.//annotation[@type="label"]')
    if isinstance(tex_label, str):
        typ = tex_to_typ.get(tex_label)
        if typ is not None:
            label = typ.char
        else:
            return
    return MathSymbolSample(
        label,
        parse_inkml(filepath, None),
    )


def read_symbol_seg_anno() -> list[SymbolSegInfo]:
    """loads symbol extraction annotation of MathWrting Dataset"""
    decoder = msgspec.json.Decoder(SymbolSegInfo)
    with open(Path(OUT_DIR, "mathwriting", "symbols.jsonl"), "rb") as f:
        symbol_annos = decoder.decode_lines(f.read())
    return symbol_annos


def extract_symbol(
    symbol_seg_anno: SymbolSegInfo,
    tex_to_typ: dict[str, TypstSymInfo],
) -> MathSymbolSample | None:
    """extract math symbol from math expression"""
    typ = tex_to_typ.get(symbol_seg_anno.label)
    if not typ:
        return
    filepath = Path(
        MATH_WRITING_DATASET_PATH,
        "train",
        f"{symbol_seg_anno.sourceSampleId}.inkml",
    )
    return MathSymbolSample(
        typ.char,
        parse_inkml(filepath, set(symbol_seg_anno.strokeIndices)),
    )


def is_invisible(c: str) -> bool:
    return unicodedata.category(c) in ["Zs", "Cc", "Cf"]


def get_typst_symbol_info() -> list[TypstSymInfo]:
    """Parse Typst symbol page to get information."""

    with open("external/typ_sym.html") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
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
            # New symbols. Add to map.
            sym_info[char] = TypstSymInfo(
                char=char,
                names=[*name],
                latex_name=li.get("data-latex-name"),  # type: ignore
                markup_shorthand=li.get("data-markup-shorthand"),  # type: ignore
                math_shorthand=li.get("data-math-shorthand"),  # type: ignore
                accent=li.get("data-accent") == "true",  # type: ignore
                alternates=li.get("data-alternates", "").split(),  # type: ignore
            )

    return list(sym_info.values())


def map_sym(
    typ_sym_info: list[TypstSymInfo], tex_to_typ: dict[str, TypstSymInfo]
) -> dict[str, TypstSymInfo]:
    """Get a mapping from Detexify keys to Typst symbol info."""
    with open("external/symbols.json", "rb") as f:
        tex_sym_info = msgspec.json.decode(f.read(), type=list[DetexifySymInfo])
    return {
        x.id: tex_to_typ[x.command]
        for x in tex_sym_info
        if x.command in tex_to_typ
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
        [((x - zero_x) * scale, (y - zero_y) * scale) for x, y in s]
        for s in strokes
    ]


def draw_to_img(strokes: Strokes) -> Image.Image:
    image = Image.new("1", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        draw.line(stroke)
    return image


def parse_contrib(sample: dict[str, Any]) -> MathSymbolSample:
    sym, _strokes = (
        sample["sym"],
        sample["strokes"],
    )
    strokes: Strokes = msgspec.json.decode(_strokes)
    name_to_chr = {x.names[0]: x.char for x in get_typst_symbol_info()}
    label = name_to_chr[sym]
    return MathSymbolSample(label, MathSymbol(strokes))


@cache
def get_msgpack_encoder():
    return msgspec.msgpack.Encoder()


def create_dataset(
    parse_func: Callable[[Any], MathSymbolSample | None],
    data: Any,
    dataset_name: str,
) -> None:
    """Create an dataset database file from raw data using a parse function."""
    dataset_path = Path(OUT_DIR, dataset_name)
    db_path = Path(dataset_path, "db")
    db_path.mkdir(exist_ok=True, parents=True)
    class_counters: dict[str, int] = defaultdict(int)
    write_frequency = 1000
    total_length = 0

    def _get_opts() -> Options:
        opts = Options()
        opts.increase_parallelism(process_cpu_count())
        # zstd
        opts.set_compression_type(DBCompressionType.zstd())
        opts.set_compression_options(0, 0, 0, 32768)
        opts.set_zstd_max_train_bytes(32768 * 100)
        opts.set_bottommost_compression_type(DBCompressionType.zstd())
        opts.set_bottommost_compression_options(0, 0, 0, 32768, True)
        opts.set_bottommost_zstd_max_train_bytes(32768 * 100, True)
        # disable auto-compact, compact later
        opts.prepare_for_bulk_load()
        # increase buffer size and count
        opts.set_write_buffer_size(128 * 1024 * 1024)
        opts.set_max_write_buffer_number(process_cpu_count())
        opts.set_min_write_buffer_number_to_merge(2)
        return opts

    def _worker_func(
        parse_func: Callable[[Any], MathSymbolSample | None], data_item: Any
    ) -> tuple[str | None, bytes]:
        encoder = get_msgpack_encoder()

        parsed_data = parse_func(data_item)
        if parsed_data is None:
            return None, b""
        encoded = encoder.encode(parsed_data.symbol)

        return parsed_data.label, encoded

    db = Rdict(str(db_path), _get_opts())
    with ProcessPoolExecutor() as exec:
        results_iter = exec.map(
            partial(_worker_func, parse_func), data, chunksize=1000
        )
        w_opts = WriteOptions()
        w_opts.disable_wal(True)
        wb = WriteBatch()
        # write to lmdb
        for label, encoded_data in results_iter:
            if not label or len(encoded_data) == 0:
                continue
            current_idx = class_counters.get(label, 0)
            key = f"{ord(label):08d}_{current_idx:09d}"
            class_counters[label] += 1
            total_length += 1
            wb.put(key, encoded_data)
            if len(wb) > write_frequency:
                db.write(wb, w_opts)
                wb.clear()
        # write remaining items
        if len(wb) > 0:
            db.write(wb, w_opts)
        # compact database and close
    db.compact_range(None, None)
    db.close()

    # save dataset information
    with open(Path(dataset_path, "dataset_info.json"), "wb") as f:
        dataset_info = DataSetInfo(
            name=dataset_name,
            total_count=total_length,
            class_count=class_counters,
        )
        f.write(msgspec.json.encode(dataset_info))


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)

    # Get symbol info.
    typ_sym_info = get_typst_symbol_info()
    key_to_typ = map_sym(typ_sym_info)
    with open(f"{OUT_DIR}/symbols.json", "wb") as f:
        f.write(msgspec.json.encode(typ_sym_info))

    # Parse math symbols from detexify dataset
    with open("external/detexify.json", "rb") as f:
        detexify_raw_data = msgspec.json.decode(f.read())
        create_dataset(
            parse_func=partial(parse_detexify_symbol, key_to_typ),
            data=detexify_raw_data,
            dataset_name="detexify",
        )

    # Parse MathWrting Dataset sole symboles
    create_dataset(
        parse_func=partial(parse_sole_symbol, key_to_typ),
        data=Path(MATH_WRITING_DATASET_PATH, "symbols").glob("*.inkml"),
        dataset_name="math_writing_sole",
    )
    # Parse extracted symbols from MathWrting Dataset
    seg_annos = read_symbol_seg_anno()
    create_dataset(
        parse_func=partial(extract_symbol, tex_to_typ),
        data=seg_annos,
        dataset_name="math_writing_extracted",
    )

    # Parse contributed data
    if USE_CONTRIB:
        with open(CONTRIB_DATA_PATH, "rb") as f:
            samples = msgspec.json.decode(f.read())[0]["results"]
            create_dataset(
                parse_func=parse_contrib, data=samples, dataset_name="contrib"
            )
