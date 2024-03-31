import os
import re
from typing import Any, Optional, Iterable
from concurrent.futures import ProcessPoolExecutor

import orjson
from bs4 import BeautifulSoup
from fontTools import subset
from PIL import Image, ImageDraw

type Strokes = list[list[tuple[float, float]]]
type StrokesT = list[list[tuple[float, float, int]]]


def parse_typ_sym_page() -> list[dict[str, Any]]:
    soup = BeautifulSoup(open("data/typ_sym.html").read(), "html.parser")
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
        if not chr(int(li["data-codepoint"])).isspace()
    ]


def map_sym(typ_sym_info) -> tuple[dict[str, str], set[str]]:
    key_to_tex = orjson.loads(open("data/symbols.json").read())
    key_to_tex = {x["id"]: x["command"][1:] for x in key_to_tex}

    tex_to_typ = orjson.loads(open("data/default.json").read())["commands"]
    for k, v in tex_to_typ.items():
        if v["kind"] == "sym":
            tex_to_typ[k] = k
        elif v["kind"] == "alias-sym":
            tex_to_typ[k] = v["alias"]
        else:
            tex_to_typ[k] = None
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        tex_to_typ[f"mathds{{{c}}}"] = c + c

    norm = {x["name"]: x for x in typ_sym_info}
    norm |= {chr(x["codepoint"]): x for x in typ_sym_info}
    norm |= {x["markup-shorthand"]: x for x in typ_sym_info}
    norm |= {x["math-shorthand"]: x for x in typ_sym_info}
    del norm[None]

    key_to_typ = {}
    charset = set()
    for k, v in key_to_tex.items():
        x = norm.get(tex_to_typ.get(v))
        if x is not None:
            key_to_typ[k] = x["name"]
            charset.add(chr(x["codepoint"]))
    return key_to_typ, charset


def normalize(strokes: StrokesT, size: int) -> Optional[Strokes]:
    min_x = min([x for s in strokes for x, _, _ in s])
    max_x = max([x for s in strokes for x, _, _ in s])
    min_y = min([y for s in strokes for _, y, _ in s])
    max_y = max([y for s in strokes for _, y, _ in s])

    width = max(max_x - min_x, max_y - min_y)
    if width == 0:
        return None
    width *= 1.2  # leave margin to avoid edge cases
    zero_x = (max_x + min_x) / 2 - width / 2
    zero_y = (max_y + min_y) / 2 - width / 2
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


def strip_font(charset: Iterable[str]):
    subset.main(
        [
            "data/NewCMMath-Regular.otf",
            "--text=" + "".join(charset),
            "--flavor=woff2",
            "--output-file=migrate-out/NewCMMath-Detypify.woff2",
        ]
    )


def main():
    executor = ProcessPoolExecutor()

    typ_sym_info = parse_typ_sym_page()
    key_to_typ, charset = map_sym(typ_sym_info)
    typ_sym_names = sorted(set(key_to_typ.values()))

    executor.submit(strip_font, charset)

    os.makedirs("migrate-out", exist_ok=True)
    open("migrate-out/symbols.json", "wb").write(orjson.dumps(typ_sym_info))
    open("supported-symbols.txt", "w").write("\n".join(typ_sym_names) + "\n")

    def handle(i, row):
        typ = key_to_typ.get(row[0])
        if typ is None:
            return
        strokes = normalize(row[1], 32)
        if strokes is None:
            return
        os.makedirs(f"migrate-out/data/{typ}", exist_ok=True)
        draw(strokes, 32).save(f"migrate-out/data/{typ}/{i}.png")

    detexify_data = orjson.loads(open("data/detexify.json").read())
    executor.map(handle, enumerate(detexify_data), chunksize=1000)
