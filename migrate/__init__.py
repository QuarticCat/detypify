import csv
import os
import re
from typing import Any, Optional

import orjson
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from unicodeit.data import REPLACEMENTS

type Strokes = list[list[tuple[float, float]]]

IMG_SIZE = 32  # px


def is_space(c: str) -> bool:
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
    label_list = orjson.loads(open("external/symbols.json", "rb").read())
    key_to_tex = {x["id"]: x["command"][1:] for x in label_list}

    uni_to_typ = {chr(x["codepoint"]): x["name"] for x in typ_sym_info}
    tex_to_typ = {k[1:]: uni_to_typ[v] for k, v in REPLACEMENTS if v in uni_to_typ}

    norm = {x["name"]: x["name"] for x in typ_sym_info} | uni_to_typ
    norm |= {x["markup-shorthand"]: x["name"] for x in typ_sym_info}
    norm |= {x["math-shorthand"]: x["name"] for x in typ_sym_info}

    mitex_map = orjson.loads(open("external/default.json", "rb").read())
    for k, v in mitex_map["commands"].items():
        if v["kind"] == "sym" and k in norm:
            tex_to_typ[k] = norm[k]
        elif v["kind"] == "alias-sym" and v["alias"] in norm:
            tex_to_typ[k] = norm[v["alias"]]

    extra_map = csv.reader(open("assets/tex_to_typ_extra.csv"))
    tex_to_typ |= {k[1:]: v for k, v in extra_map}

    return {k: tex_to_typ[v] for k, v in key_to_tex.items() if v in tex_to_typ}


def normalize(strokes: Strokes) -> Optional[Strokes]:
    xs = [x for s in strokes for x, _ in s]
    min_x, max_x = min(xs), max(xs)
    ys = [y for s in strokes for _, y in s]
    min_y, max_y = min(ys), max(ys)

    width = max(max_x - min_x, max_y - min_y)
    if width == 0:
        return None
    width *= 1.2  # leave margin to avoid edge cases
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


def main():
    os.makedirs("migrate-out", exist_ok=True)

    typ_sym_info = parse_typ_sym_page()
    key_to_typ = map_sym(typ_sym_info)
    typ_sym_names = sorted(set(key_to_typ.values()))

    open("migrate-out/symbols.json", "wb").write(orjson.dumps(typ_sym_info))
    open("assets/supported-symbols.txt", "w").write("\n".join(typ_sym_names) + "\n")

    detexify_data = orjson.loads(open("external/detexify.json", "rb").read())
    for i, [key, strokes] in enumerate(detexify_data):
        typ = key_to_typ.get(key)
        if typ is None:
            continue
        strokes = [[(x, y) for x, y, _ in s] for s in strokes]
        strokes = normalize(strokes)
        if strokes is None:
            continue
        os.makedirs(f"migrate-out/data/{typ}", exist_ok=True)
        draw_to_img(strokes).save(f"migrate-out/data/{typ}/{i}.png")
