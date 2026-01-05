"""Preprocess training dataset."""

import os
import re
import shutil
import unicodedata

import msgspec
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]

OUT_DIR = "build/data"
IMG_SIZE = 256  # px

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
                names=[name],
                latex_name=li.get("data-latex-name"),
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

    with open("external/symbols.json", "rb") as f:
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


if __name__ == "__main__":
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Get symbol info.
    typ_sym_info = get_typst_symbol_info()
    key_to_typ = map_sym(typ_sym_info)
    with open(f"{OUT_DIR}/symbols.json", "wb") as f:
        f.write(msgspec.json.encode(typ_sym_info))

    # TODO: Include data from contrib.

    # Convert strokes into images and label them.
    with open("external/detexify.json", "rb") as f:
        detexify_data = msgspec.json.decode(f.read())
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
        os.makedirs(f"{OUT_DIR}/img/{ord(typ.char)}", exist_ok=True)
        draw_to_img(strokes).save(f"{OUT_DIR}/img/{ord(typ.char)}/{i}.png")
