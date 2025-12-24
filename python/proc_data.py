"""Preprocess training dataset."""

import os
import re
import unicodedata

import msgspec
import orjson
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]

OUT_DIR = "build/data"
IMG_SIZE = 32  # px

# Missing mappings in the Typst symbol page.
TEX_TO_TYP = {
    # Alphabet
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
    # Greek Alphabet (Uppercases)
    "\\Alpha": "Alpha",
    "\\Beta": "Beta",
    "\\Gamma": "Gamma",
    "\\Delta": "Delta",
    "\\Epsilon": "Epsilon",
    "\\Zeta": "Zeta",
    "\\Eta": "Eta",
    "\\Theta": "Theta",
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
    # Greek Alphabet (Lowercases)
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
    # Others
    "\\&": "amp",
    "\\#": "hash",
    "\\%": "percent",
    "\\{": "brace.l",
    "\\}": "brace.r",
    "\\--": "dash.en",
    "\\---": "dash.em",
    "\\colon": "colon",
    "\\aleph": "aleph",
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
    names: list[str]
    latex_name: str | None = None
    codepoint: int
    markup_shorthand: str | None = None
    math_shorthand: str | None = None
    accent: bool = False
    alternates: list[str] = []


class DetexifySymInfo(msgspec.Struct, kw_only=True, omit_defaults=True):
    command: str
    # package: str | None = None
    # mathmode: bool
    # textmode: bool
    id: str
    # css_class: str


def is_invisible(c: str) -> bool:
    return unicodedata.category(c) in ["Zs", "Cc", "Cf"]


def get_typst_symbol_info() -> list[TypstSymInfo]:
    """Parse Typst symbol page to get information."""

    soup = BeautifulSoup(open("external/typ_sym.html").read(), "html.parser")
    sym_info = {}

    for li in soup.find_all("li", id=re.compile("^symbol-")):
        codepoint = ord(li["data-value"][0])
        if is_invisible(chr(codepoint)) or li.get("data-deprecation"):
            # We don't care about invisible chars and deprecated names.
            continue
        elif codepoint in sym_info:
            # Repeated symbols. Merge names.
            sym_info[codepoint].names.append(li["id"][len("symbol-") :])
        else:
            # New symbols. Add to map.
            sym_info[codepoint] = TypstSymInfo(
                names=[li["id"][len("symbol-") :]],
                latex_name=li.get("data-latex-name"),
                codepoint=codepoint,
                markup_shorthand=li.get("data-markup-shorthand"),
                math_shorthand=li.get("data-math-shorthand"),
                accent=li.get("data-accent") == "true",
                alternates=li.get("data-alternates", "").split(),
            )

    return list(sym_info.values())


def map_sym(typ_sym_info: list[TypstSymInfo]) -> dict[str, TypstSymInfo]:
    """Get a mapping from Detexify keys to Typst symbol info."""

    tex_to_typ = {s.latex_name: s for s in typ_sym_info}
    name_to_typ = {name: s for s in typ_sym_info for name in s.names}
    tex_to_typ |= {k: name_to_typ[v] for k, v in TEX_TO_TYP.items()}

    content = open("external/symbols.json", "rb").read()
    tex_sym_info = msgspec.json.decode(content, type=list[DetexifySymInfo])
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


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    typ_sym_info = get_typst_symbol_info()
    key_to_typ = map_sym(typ_sym_info)
    typ_sym_names = sorted(set(n for x in key_to_typ.values() for n in x.names))

    open(f"{OUT_DIR}/symbols.json", "wb").write(msgspec.json.encode(typ_sym_info))
    open("assets/supported-symbols.txt", "w").write("\n".join(typ_sym_names) + "\n")

    # TODO: Use data from contrib.
    detexify_data = orjson.loads(open("external/detexify.json", "rb").read())
    for i, [key, strokes] in enumerate(detexify_data):
        typ = key_to_typ.get(key)
        if typ is None:
            continue
        strokes = [[(x, y) for x, y, _ in s] for s in strokes if len(s) > 1]
        if len(strokes) == 0:
            continue
        strokes = normalize(strokes)
        if len(strokes) == 0:
            continue
        os.makedirs(f"{OUT_DIR}/img/{typ.codepoint}", exist_ok=True)
        draw_to_img(strokes).save(f"{OUT_DIR}/img/{typ.codepoint}/{i}.png")
