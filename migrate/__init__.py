import json
import os
from typing import Iterable, Optional

import psycopg
from PIL import Image, ImageDraw

IMAGE_DIR = "migrate-out"

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]


def map_sym() -> dict[str, str]:
    key_to_tex = json.load(open("data/symbols.json"))
    key_to_tex = {x["id"]: x["command"][1:] for x in key_to_tex}

    tex_to_typ = json.load(open("data/default.json"))["commands"]
    tex_to_typ = {k: v["alias"] for k, v in tex_to_typ.items() if v.get("alias")}

    return {k: tex_to_typ[v] for k, v in key_to_tex.items() if v in tex_to_typ}


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
    min_x = min(x for stroke in strokes for x, _ in stroke)
    max_x = max(x for stroke in strokes for x, _ in stroke)
    min_y = min(y for stroke in strokes for _, y in stroke)
    max_y = max(y for stroke in strokes for _, y in stroke)
    width = max(max_x - min_x, max_y - min_y) + 40
    if width == 40:
        return None
    min_x = (max_x + min_x) / 2 - width / 2
    min_y = (max_y + min_y) / 2 - width / 2
    return [
        [((x - min_x) / width * size, (y - min_y) / width * size) for x, y in stroke]
        for stroke in strokes
    ]


def draw(strokes: Strokes, size: int) -> Image.Image:
    image = Image.new("1", (size, size), 1)
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        draw.line(stroke)
    return image


def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)

    for id, typ, strokes in get_data(map_sym()):
        strokes = [[(x, y) for x, y, _ in s] for s in strokes]  # strip timestamps
        strokes = normalize(strokes, 32)
        if strokes is None:
            continue
        image = draw(strokes, 32)
        # Prepend '-' to avoid names like '.'
        os.makedirs(os.path.join(IMAGE_DIR, f"-{typ}"), exist_ok=True)
        image.save(os.path.join(IMAGE_DIR, f"-{typ}", f"{id}.png"))
