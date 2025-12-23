"""Preprocess training dataset."""

import os
import re

import msgspec
import orjson
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from unicodeit.data import REPLACEMENTS

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]

OUT_DIR = "build/data"
IMG_SIZE = 32  # px


class TypstSymInfo(msgspec.Struct, kw_only=True, omit_defaults=True):
    names: list[str]
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


def is_space(c: str) -> bool:
    return c.isspace() or c in "\u2060\u200b\u200c\u200d\u200e\u200f"


def get_typst_symbol_info() -> list[TypstSymInfo]:
    """Parse Typst symbol page to get information."""

    soup = BeautifulSoup(open("external/typ_sym.html").read(), "html.parser")
    sym_info = {}

    for li in soup.find_all("li", id=re.compile("^symbol-")):
        codepoint = int(li["data-codepoint"])
        if is_space(chr(codepoint)):
            # We don't care about whitespaces.
            continue
        if codepoint in sym_info:
            # Repeated symbols. Merge them.
            sym_info[codepoint].names.append(li["id"][len("symbol-") :])
        else:
            # New symbols. Add to map.
            sym_info[codepoint] = TypstSymInfo(
                names=[li["id"][len("symbol-") :]],
                codepoint=codepoint,
                markup_shorthand=li.get("data-markup-shorthand"),
                math_shorthand=li.get("data-math-shorthand"),
                accent=li.get("data-accent") == "true",
                alternates=li.get("data-alternates", "").split(),
            )

    return list(sym_info.values())


def map_sym(typ_sym_info: list[TypstSymInfo]) -> dict[str, TypstSymInfo]:
    """Get a mapping from Detexify keys to Typst symbol info."""

    norm = {n: x for x in typ_sym_info for n in x.names}
    norm |= {chr(x.codepoint): x for x in typ_sym_info}
    norm |= {x.markup_shorthand: x for x in typ_sym_info}
    norm |= {x.math_shorthand: x for x in typ_sym_info}

    tex_to_typ = {k[1:]: norm[v] for k, v in REPLACEMENTS if v in norm}

    mitex_map = orjson.loads(open("external/default.json", "rb").read())
    for k, v in mitex_map["commands"].items():
        if v["kind"] == "sym" and k in norm:
            tex_to_typ[k] = norm[k]
        elif v["kind"] == "alias-sym" and v["alias"] in norm:
            tex_to_typ[k] = norm[v["alias"]]

    tex_to_typ["bowtie"] = norm["join"]
    tex_to_typ["MVAt"] = norm["at"]

    content = open("external/symbols.json", "rb").read()
    tex_sym_info = msgspec.json.decode(content, type=list[DetexifySymInfo])
    return {
        x.id: tex_to_typ[x.command[1:]]
        for x in tex_sym_info
        if x.command[1:] in tex_to_typ
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
