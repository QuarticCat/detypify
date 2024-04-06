import csv
import os
import shutil
from getpass import getpass

import orjson
from github import Auth, Github
from PIL import Image, ImageDraw, ImageFont

from migrate import draw_to_img
from migrate import normalize, parse_typ_sym_page

REF_SIZE = 100  # px


def main():
    token = getpass(">>> Input GitHub token: ")
    gh = Github(auth=Auth.Token(token))
    repo = gh.get_repo("QuarticCat/detypify-data")

    print("\n### Downloading samples...")
    all_samples = {}
    for issue in repo.get_issues(state="open"):
        if issue.title != "Samples 0.2.0":
            continue
        body = issue.body
        json = body[body.find("[") : body.rfind("]") + 1]
        try:
            all_samples[issue.number] = orjson.loads(json)
        except Exception:
            print(f"- parse failed: #{issue.number}")

    print("\n### Generating images...")
    shutil.rmtree("bot-out", ignore_errors=True)
    os.mkdir("bot-out")
    for num, samples in all_samples.items():
        for idx, [name, strokes] in enumerate(samples):
            os.makedirs(f"bot-out/{name}", exist_ok=True)
            img = draw_to_img(normalize(strokes))
            img.save(f"bot-out/{name}/{num}.{idx}.png")

    print("\n### Generating references...")
    sym_map = {x["name"]: chr(x["codepoint"]) for x in parse_typ_sym_page()}
    for name in os.listdir("bot-out"):
        text = sym_map[name]
        img = Image.new("1", (100, 100), 1)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("external/NewCMMath-Regular.otf", size=80)
        _, _, w, h = draw.textbbox((0, 0), text, font=font)
        draw.text(((REF_SIZE - w) / 2, (REF_SIZE - h) / 2), text, font=font)
        img.save(f"bot-out/{name}/_ref.png")

    print("\n### Go through bot-out folder and delete unwanted images")
    while input(">>> Input 'done' to proceed: ") != "done":
        pass

    print("\n### Collecting wanted samples...")
    for name in os.listdir("bot-out"):
        for file in os.listdir(f"bot-out/{name}"):
            if file == "_ref.png":
                continue
            num, idx, _ = file.split(".")
            _, strokes = all_samples[int(num)][int(idx)]
            writer = csv.writer(open(f"data/dataset/{name}.csv", "a"))
            writer.writerow([num, idx, orjson.dumps(strokes).decode()])

    details = ", ".join([f"close #{n}" for n in all_samples.keys()])
    print("\n### Here's the generated commit message:")
    print(rf"<<< $'merge contributions\n\n{details}'")
