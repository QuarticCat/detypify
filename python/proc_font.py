"""Preprocess font for the webpage."""

import os

from fontTools import subset

from proc_data import get_typst_symbol_info

OUT_DIR = "build/font"

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    sym_info = get_typst_symbol_info()
    text = "".join([chr(x.codepoint) for x in sym_info])

    subset.main(
        [
            "external/NewCMMath-Regular.otf",
            f"--text={text}",
            "--flavor=woff2",
            f"--output-file={OUT_DIR}/NewCMMath-Detypify.woff2",
        ]
    )
