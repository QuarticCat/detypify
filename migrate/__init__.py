import os
import re
import csv
from typing import Any, Optional

import orjson
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw

type Strokes = list[list[tuple[float, float]]]
type StrokesT = list[list[tuple[float, float, int]]]


def is_space(c) -> bool:
    return c.isspace() or c in "\u2060\u200b\u200c\u200d\u200e\u200f"


def parse_typ_sym_page() -> list[dict[str, Any]]:
    soup = BeautifulSoup(open("external/typ_sym.html").read(), "html.parser")
    return [
        {
            "name": li["id"][len("symbol-") :],
            "codepoint": int(li["data-codepoint"]),
            "markup-shorthand": li.get("data-markup-shorthand"),
            "math-shorthand": li.get("data-math-shorthand"),
            "accent": li.get("data-accent") == "true",
            "alternates": li.get_attribute_list("data-alternates", []),
        }
        for li in soup.find_all("li", id=re.compile("^symbol-"))
        if not is_space(chr(int(li["data-codepoint"])))
    ]


def map_sym(typ_sym_info) -> tuple[dict[str, str], set[str]]:
    label_list = orjson.loads(open("external/symbols.json").read())
    key_to_tex = {x["id"]: x["command"][1:] for x in label_list}

    unitex_map = open("external/unicode-math-table.tex").read()
    unitex_map = re.findall(r'"(.*?)}{\\(.*?) ', unitex_map)
    tex_to_typ = {t: chr(int(u, 16)) for u, t in unitex_map}

    mitex_map = orjson.loads(open("external/default.json").read())
    for k, v in mitex_map["commands"].items():
        if v["kind"] == "sym":
            tex_to_typ[k] = k
        elif v["kind"] == "alias-sym":
            tex_to_typ[k] = v["alias"]

    extra_map = csv.reader(open("assets/tex_to_typ_extra.csv"))
    tex_to_typ |= {k[1:]: v for k, v in extra_map}

    norm = {x["name"]: x["name"] for x in typ_sym_info}
    norm |= {chr(x["codepoint"]): x["name"] for x in typ_sym_info}
    norm |= {x["markup-shorthand"]: x["name"] for x in typ_sym_info}
    norm |= {x["math-shorthand"]: x["name"] for x in typ_sym_info}
    del norm[None]

    key_to_typ = {}
    for k, v in key_to_tex.items():
        x = norm.get(tex_to_typ.get(v))
        if x is not None:
            key_to_typ[k] = x
    return key_to_typ


def normalize(strokes: StrokesT, size: int) -> Optional[Strokes]:
    xs = [x for s in strokes for x, _, _ in s]
    min_x, max_x = min(xs), max(xs)
    ys = [y for s in strokes for _, y, _ in s]
    min_y, max_y = min(ys), max(ys)

    width = max(max_x - min_x, max_y - min_y)
    if width == 0:
        return None
    width *= 1.2  # leave margin to avoid edge cases
    zero_x = (max_x + min_x - width) / 2
    zero_y = (max_y + min_y - width) / 2
    scale = size / width

    return [
        [((x - zero_x) * scale, (y - zero_y) * scale) for x, y, _ in s] for s in strokes
    ]


def draw(strokes: Strokes, size: int) -> Image.Image:
    image = Image.new("1", (size, size), 1)
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        draw.line(stroke)
    return image


def main():
    os.makedirs("migrate-out", exist_ok=True)

    typ_sym_info = parse_typ_sym_page()
    key_to_typ = map_sym(typ_sym_info)
    typ_sym_names = sorted(set(key_to_typ.values()))

    open("migrate-out/symbols.json", "wb").write(orjson.dumps(typ_sym_info))
    open("assets/supported-symbols.txt", "w").write("\n".join(typ_sym_names) + "\n")

    detexify_data = orjson.loads(open("external/detexify.json").read())
    for i, [key, strokes] in enumerate(detexify_data):
        typ = key_to_typ.get(key)
        if typ is None:
            continue
        strokes = normalize(strokes, 32)
        if strokes is None:
            continue
        os.makedirs(f"migrate-out/data/{typ}", exist_ok=True)
        draw(strokes, 32).save(f"migrate-out/data/{typ}/{i}.png")
