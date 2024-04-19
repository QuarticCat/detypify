import os
import shutil

import orjson
from PIL import Image, ImageDraw, ImageFont

from migrate import draw_to_img, get_typ_sym_info, normalize

REF_SIZE = 100  # px


def main():
    # bunx wrangler d1 execute detypify --remote --command='SELECT * FROM samples' --json > raw.json

    samples = orjson.loads(open("raw.json", "rb").read())[0]["results"]

    print("\n### Generating images...")

    shutil.rmtree("bot-out", ignore_errors=True)
    os.mkdir("bot-out")

    sym_to_uni = {x["name"]: chr(x["codepoint"]) for x in get_typ_sym_info()}
    for s in samples:
        id_, token, sym, strokes = s["id"], s["token"], s["sym"], s["strokes"]
        img = draw_to_img(normalize(orjson.loads(strokes)))
        img.save(f"bot-out/{sym}-{id_}-{token}.png")

        if not os.path.exists(f"bot-out/{sym}-0-0.png"):
            text = sym_to_uni[sym]
            img = Image.new("1", (100, 100), "white")
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("external/NewCMMath-Regular.otf", size=80)
            _, _, w, h = draw.textbbox((0, 0), text, font=font)
            draw.text(((REF_SIZE - w) / 2, (REF_SIZE - h) / 2), text, font=font)
            img.save(f"bot-out/{sym}-0-0.png")

    print("\n### Go through bot-out folder and delete unwanted images")
    while input(">>> Input 'done' to proceed: ") != "done":
        pass

    print("\n### Collecting wanted samples...")
    id_to_strokes = {s["id"]: s["strokes"] for s in samples}
    for f in os.listdir("bot-out"):
        sym, id_, _ = f.rsplit(".", 1)[0].split("-")
        if id_ != "0":
            strokes = id_to_strokes[int(id_)]
            open(f"data/dataset/{sym}.txt", "a").write(strokes + "\n")

    # bunx wrangler d1 execute detypify --remote --command='DELETE FROM samples WHERE id <= n'
