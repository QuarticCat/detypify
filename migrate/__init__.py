import json
import os
import re
from typing import Any, Iterable, Optional

import psycopg
from bs4 import BeautifulSoup
from fontTools import subset
from PIL import Image, ImageDraw

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]


def parse_typ_sym() -> list[dict[str, Any]]:
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


def map_sym(sym_list) -> tuple[dict[str, str], set[str]]:
    key_to_tex = json.load(open("data/symbols.json"))
    key_to_tex = {x["id"]: x["command"][1:] for x in key_to_tex}

    tex_to_typ = json.load(open("data/default.json"))["commands"]
    for k, v in tex_to_typ.items():
        if v["kind"] == "sym":
            tex_to_typ[k] = k
        elif v["kind"] == "alias-sym":
            tex_to_typ[k] = v["alias"]
        else:
            tex_to_typ[k] = None
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        tex_to_typ[f"mathds{{{c}}}"] = c + c

    norm = {x["name"]: x for x in sym_list}
    norm |= {chr(x["codepoint"]): x for x in sym_list}
    norm |= {x["markup-shorthand"]: x for x in sym_list}
    norm |= {x["math-shorthand"]: x for x in sym_list}
    del norm[None]

    key_to_typ = {}
    charset = set()
    for k, v in key_to_tex.items():
        x = norm.get(tex_to_typ.get(v))
        if x is not None:
            key_to_typ[k] = x["name"]
            charset.add(chr(x["codepoint"]))
    return key_to_typ, charset


def get_data(key_to_typ: dict[str, str]) -> Iterable[tuple[int, str, Strokes]]:
    conn = psycopg.connect("dbname=detypify")
    cur = conn.cursor()
    cur.execute("""
        CREATE TEMPORARY TABLE key_to_typ (
            key text,
            typ text)
    """)
    with cur.copy("COPY key_to_typ (key, typ) FROM STDIN") as copy:
        for item in key_to_typ.items():
            copy.write_row(item)
    return cur.execute("""
        SELECT id, typ, strokes
        FROM samples
        JOIN key_to_typ
        ON samples.key = key_to_typ.key
    """)


def normalize(strokes: Strokes, size: int) -> Optional[Strokes]:
    min_x = min(x for s in strokes for x, _ in s)
    max_x = max(x for s in strokes for x, _ in s)
    min_y = min(y for s in strokes for _, y in s)
    max_y = max(y for s in strokes for _, y in s)

    width = max(max_x - min_x, max_y - min_y)
    if width == 0:
        return None
    width *= 1.2  # leave margin to avoid edge cases
    zero_x = (max_x + min_x) / 2 - width / 2
    zero_y = (max_y + min_y) / 2 - width / 2
    scale = size / width

    return [
        [((x - zero_x) * scale, (y - zero_y) * scale) for x, y in s] for s in strokes
    ]


def draw(strokes: Strokes, size: int) -> Image.Image:
    image = Image.new("1", (size, size), 1)
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        draw.line(stroke)
    return image


def main():
    sym_list = parse_typ_sym()
    os.makedirs("migrate-out", exist_ok=True)
    json.dump(sym_list, open("migrate-out/symbols.json", "w"))

    key_to_typ, charset = map_sym(sym_list)
    subset.main(
        [
            "data/NewCMMath-Regular.otf",
            "--text=" + "".join(charset),
            "--flavor=woff2",
            "--output-file=migrate-out/NewCMMath-Detypify.woff2",
        ]
    )

    sym_list = sorted(set(key_to_typ.values()))
    open("supported-symbols.txt", "w").write("\n".join(sym_list) + "\n")

    for id, typ, strokes in get_data(key_to_typ):
        strokes = [[(x, y) for x, y, _ in s] for s in strokes]  # strip timestamps
        strokes = normalize(strokes, 32)
        if strokes is None:
            continue
        image = draw(strokes, 32)
        os.makedirs(f"migrate-out/data/{typ}", exist_ok=True)
        image.save(f"migrate-out/data/{typ}/{id}.png")
