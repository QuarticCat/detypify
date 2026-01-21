"""Preprocess contribution from the webpage."""

import shutil
from pathlib import Path

import msgspec
from PIL import Image, ImageDraw, ImageFont
from proc_data import IMG_SIZE, draw_to_img, get_typst_symbol_info

OUT_DIR = Path("build/contrib")
REF_SIZE = 100  # px


def bold(s: str) -> str:
    return "\033[1m" + s + "\033[0m"


if __name__ == "__main__":
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.mkdir(exist_ok=True)

    cmd = "bunx wrangler d1 execute detypify --remote \
                --command='SELECT * FROM samples' --json > build/dataset.json"
    print("### Run this command to fetch data:")
    print(f"### $ {bold(cmd)}")
    while input(">>> Input 'done' to proceed: ") != "done":
        pass
    with Path("build/dataset.json").open("rb") as f:
        samples = msgspec.json.decode(f.read())[0]["results"]

    print("\n### Generating images...")
    name_to_chr = {x.names[0]: x.char for x in get_typst_symbol_info()}
    for s in samples:
        id_, token, sym, strokes = s["id"], s["token"], s["sym"], s["strokes"]
        img = draw_to_img(msgspec.json.decode(strokes), IMG_SIZE)
        img.save(f"{OUT_DIR}/{sym}-{id_}-{token}.png")

        if not Path(f"{OUT_DIR}/{sym}-0-0.png").exists():
            text = name_to_chr[sym]
            img = Image.new("1", (100, 100), "white")
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("external/NewCMMath-Regular.otf", size=80)
            _, _, w, h = draw.textbbox((0, 0), text, font=font)
            draw.text(((REF_SIZE - w) / 2, (REF_SIZE - h) / 2), text, font=font)
            img.save(f"{OUT_DIR}/{sym}-0-0.png")

    print(f"\n### Go through {bold(str(OUT_DIR))} and delete unwanted images")
    while input(">>> Input 'done' to proceed: ") != "done":
        pass

    print("\n### Collecting wanted samples...")
    id_to_strokes = {s["id"]: s["strokes"] for s in samples}
    for filename in OUT_DIR.iterdir():
        sym, id_, _ = str(filename).rsplit(".", 1)[0].split("-")
        if id_ != "0":
            strokes = id_to_strokes[str(id_)]
            with Path(f"data/dataset/{sym}.txt").open("a") as f:
                f.write(strokes + "\n")

    cmd = "bunx wrangler d1 execute detypify --remote \
                --command='DELETE FROM samples WHERE id <= n'"
    print("\n### Run this command to clean data:")
    print(f"### $ {bold(cmd)}")
