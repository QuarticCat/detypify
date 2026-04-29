"""Preprocess contribution from the webpage."""

import logging
import shutil

import cv2
from detypify.data.paths import DEFAULT_DATA_PATHS, DataPaths
from detypify.data.rendering import rasterize_strokes
from detypify.data.symbols import get_typst_symbol_info
from msgspec import json
from PIL import Image, ImageDraw, ImageFont

REF_SIZE = 100  # px
logger = logging.getLogger(__name__)


def bold(s: str) -> str:
    return "\033[1m" + s + "\033[0m"


def main(paths: DataPaths = DEFAULT_DATA_PATHS) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    shutil.rmtree(paths.contrib_review_dir, ignore_errors=True)
    paths.contrib_review_dir.mkdir(exist_ok=True, parents=True)
    img_size = 256

    cmd = "bunx wrangler d1 execute detypify --remote \
                --command='SELECT * FROM samples' --json > build/raw/contrib/dataset.json"
    print("### Run this command to fetch data:")  # noqa: T201
    print(f"### $ {bold(cmd)}")  # noqa: T201
    while input(">>> Input 'done' to proceed: ") != "done":
        pass
    with paths.contrib_raw_json.open("rb") as f:
        samples = json.decode(f.read())[0]["results"]

    logger.info("\n### Generating images...")
    name_to_chr = {x.names[0]: x.char for x in get_typst_symbol_info()}
    for s in samples:
        id_, token, sym, strokes = s["id"], s["token"], s["sym"], s["strokes"]
        img = rasterize_strokes(json.decode(strokes), img_size)
        cv2.imwrite(str(paths.contrib_review_dir / f"{sym}-{id_}-{token}.png"), img)

        ref_path = paths.contrib_review_dir / f"{sym}-0-0.png"
        if not ref_path.exists():
            text = name_to_chr[sym]
            img = Image.new("1", (100, 100), "white")
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(str(paths.math_font), size=80)
            _, _, w, h = draw.textbbox((0, 0), text, font=font)
            draw.text(((REF_SIZE - w) / 2, (REF_SIZE - h) / 2), text, font=font)
            img.save(ref_path)

    print(f"\n### Go through {bold(str(paths.contrib_review_dir))} and delete unwanted images")  # noqa: T201
    while input(">>> Input 'done' to proceed: ") != "done":
        pass

    logger.info("\n### Collecting wanted samples...")
    id_to_strokes = {s["id"]: s["strokes"] for s in samples}
    accepted_samples = []
    for filename in paths.contrib_review_dir.iterdir():
        sym, id_, _ = filename.stem.split("-")
        if id_ != "0":
            strokes = id_to_strokes[id_]
            accepted_samples.append({"sym": sym, "strokes": strokes})

    paths.contrib_accepted_json.parent.mkdir(parents=True, exist_ok=True)
    with paths.contrib_accepted_json.open("wb") as f:
        f.write(json.encode(accepted_samples))
    logger.info("Accepted %s samples into %s", len(accepted_samples), paths.contrib_accepted_json)

    cmd = "bunx wrangler d1 execute detypify --remote \
                --command='DELETE FROM samples WHERE id <= n'"
    print("\n### Run this command to clean data:")  # noqa: T201
    print(f"### $ {bold(cmd)}")  # noqa: T201
