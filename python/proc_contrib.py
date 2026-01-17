"""Preprocess contribution from the webpage."""

import os
import shutil

import msgspec
from PIL import Image, ImageDraw, ImageFont

from proc_data import draw_to_img, get_typst_symbol_info, normalize

OUT_DIR = "build/contrib"
REF_SIZE = 100  # px


def bold(s: str) -> str:
    return "\033[1m" + s + "\033[0m"


if __name__ == "__main__":
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    cmd = "bunx wrangler d1 execute detypify --remote --command='SELECT * FROM samples' --json > build/dataset.json"
    print("### Run this command to fetch data:")
    print(f"### $ {bold(cmd)}")
    while input(">>> Input 'done' to proceed: ") != "done":
        pass
    with open("build/dataset.json", "rb") as f:
        samples = msgspec.json.decode(f.read())[0]["results"]

    print("\n### Generating images...")
    name_to_chr = {x.names[0]: x.char for x in get_typst_symbol_info()}
    for s in samples:
        id_, token, sym, strokes = s["id"], s["token"], s["sym"], s["strokes"]
        img = draw_to_img(normalize(msgspec.json.decode(strokes)))
        img.save(f"{OUT_DIR}/{sym}-{id_}-{token}.png")

        if not os.path.exists(f"{OUT_DIR}/{sym}-0-0.png"):
            text = name_to_chr[sym]
            img = Image.new("1", (100, 100), "white")
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("external/NewCMMath-Regular.otf", size=80)
            _, _, w, h = draw.textbbox((0, 0), text, font=font)
            draw.text(((REF_SIZE - w) / 2, (REF_SIZE - h) / 2), text, font=font)
            img.save(f"{OUT_DIR}/{sym}-0-0.png")

    print(f"\n### Go through {bold(OUT_DIR)} and delete unwanted images")
    while input(">>> Input 'done' to proceed: ") != "done":
        pass

    print("\n### Collecting wanted samples...")
    id_to_strokes = {s["id"]: s["strokes"] for s in samples}
    for filename in os.listdir(OUT_DIR):
        sym, id_, _ = filename.rsplit(".", 1)[0].split("-")
        if id_ != "0":
            strokes = id_to_strokes[str(id_)]
            with open(f"data/dataset/{sym}.txt", "a") as f:
                f.write(strokes + "\n")

    cmd = "bunx wrangler d1 execute detypify --remote --command='DELETE FROM samples WHERE id <= n'"
    print("\n### Run this command to clean data:")
    print(f"### $ {bold(cmd)}")
