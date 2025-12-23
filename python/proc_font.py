"""Preprocess font for the webpage."""

import os

from fontTools import subset

from proc_data import get_typ_sym_info

OUT_DIR = "build/font"

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    typ_sym_info = get_typ_sym_info()
    text = "".join([chr(x["codepoint"]) for x in typ_sym_info])

    subset.main(
        [
            "external/NewCMMath-Regular.otf",
            "--text=" + text,
            "--flavor=woff2",
            f"--output-file={OUT_DIR}/NewCMMath-Detypify.woff2",
        ]
    )
