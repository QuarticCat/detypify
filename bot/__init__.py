import os
import shutil
import csv
from getpass import getpass

import requests
import orjson
from github import Auth, Github

from migrate import normalize, draw


def main():
    token = getpass(">>> Input GitHub token: ")
    gh = Github(auth=Auth.Token(token))
    repo = gh.get_repo("QuarticCat/detypify-data")

    print("Downloading samples...")
    all_samples = {}
    for issue in repo.get_issues(state="open"):
        if not issue.title.startswith("Samples"):
            continue
        pb_url = issue.body.splitlines()[0]
        all_samples[issue.number] = requests.get(pb_url).json()

    print("Generating images...")
    shutil.rmtree("bot-out", ignore_errors=True)
    os.mkdir("bot-out")
    for num, samples in all_samples.items():
        for idx, [name, strokes] in enumerate(samples):
            os.makedirs(f"bot-out/{name}", exist_ok=True)
            draw(normalize(strokes)).save(f"bot-out/{name}/{num}-{idx}.png")

    print("Go through bot-out folder and delete unwanted images")
    while input(">>> Input 'done' to proceed: ") != "done":
        pass

    print("Collecting wanted samples...")
    for name in os.listdir("bot-out"):
        for file in os.listdir(f"bot-out/{name}"):
            num, idx = file.split(".")[0].split("-")
            name, strokes = all_samples[int(num)][int(idx)]
            writer = csv.writer(open(f"data/dataset/{name}.csv", "a"))
            writer.writerow([num, idx, orjson.dumps(strokes).decode()])

    print("Here's the generated commit message:")
    print(", ".join([f"close #{n}" for n in all_samples.keys()]))
