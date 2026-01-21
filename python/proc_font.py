"""Preprocess font for the webpage."""

import shutil
from pathlib import Path

from fontTools import subset
from proc_data import get_typst_symbol_info

OUT_DIR = Path("build/font")

if __name__ == "__main__":
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.mkdir(exist_ok=True)

    sym_info = get_typst_symbol_info()
    args = [
        "external/NewCMMath-Regular.otf",
        f"--text={''.join([x.char for x in sym_info])}",
        "--flavor=woff2",
        f"--output-file={OUT_DIR}/NewCMMath-Detypify.woff2",
    ]
    subset.main(args)  # type: ignore
