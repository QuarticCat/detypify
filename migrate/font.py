import os

from fontTools import subset

from migrate import parse_typ_sym_page


def main():
    os.makedirs("migrate-out", exist_ok=True)

    typ_sym_info = parse_typ_sym_page()
    text = "".join([chr(x["codepoint"]) for x in typ_sym_info])

    subset.main(
        [
            "data/NewCMMath-Regular.otf",
            "--text=" + text,
            "--flavor=woff2",
            "--output-file=migrate-out/NewCMMath-Detypify.woff2",
        ]
    )
