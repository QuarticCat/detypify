import os

from fontTools import subset

from migrate import get_typ_sym_info


if __name__ == "__main__":
    os.makedirs("migrate-out", exist_ok=True)

    typ_sym_info = get_typ_sym_info()
    text = "".join([chr(x["codepoint"]) for x in typ_sym_info])

    subset.main(
        [
            "external/NewCMMath-Regular.otf",
            "--text=" + text,
            "--flavor=woff2",
            "--output-file=migrate-out/NewCMMath-Detypify.woff2",
        ]
    )
