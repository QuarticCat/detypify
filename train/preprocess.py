import os

import psycopg
from PIL import Image, ImageDraw

type Point = tuple[float, float]
type Stroke = list[Point]
type Strokes = list[Stroke]


def normalize(strokes: Strokes, size: int) -> Strokes:
    min_x = min(x for stroke in strokes for x, _ in stroke)
    max_x = max(x for stroke in strokes for x, _ in stroke)
    min_y = min(y for stroke in strokes for _, y in stroke)
    max_y = max(y for stroke in strokes for _, y in stroke)
    width = max(max_x - min_x, max_y - min_y)
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


def preprocess(image_dir="data/typ_images", conninfo="dbname=detypify"):
    os.makedirs(image_dir, exist_ok=True)

    conn = psycopg.connect(conninfo)
    for id, typ, strokes in conn.execute("SELECT id, typ, strokes FROM typ_samples"):
        strokes = [[(x, y) for x, y, _ in s] for s in strokes]  # strip timestamps
        strokes = normalize(strokes, 32)
        image = draw(strokes, 32)
        os.makedirs(os.path.join(image_dir, typ), exist_ok=True)
        image.save(os.path.join(image_dir, typ, f"{id}.png"))
        return
