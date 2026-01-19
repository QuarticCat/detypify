"""Train the model."""

from os import process_cpu_count
from pathlib import Path
from typing import Literal

import lightning as L
import msgspec
import torch
from dataset import MathSymbolDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from model import MobileNetV4, TypstSymbolClassifier
from proc_data import DATASET_PATH, IMG_SIZE, TypstSymInfo, get_dataset_info

OUT_DIR = Path("build/train")
DATA_DIR = Path("build/data")

FINE_TUNING = False
USE_TRANSFORMER = False
type model_size = Literal["small", "medium", "large"]


if __name__ == "__main__":
    datasets = ["detexify", "mathwriting_symbols", "mathwriting_extracted"]
    classes: set[str] = set()
    for dataset in datasets:
        data_info = get_dataset_info(dataset)
        classes.update(data_info.class_count.keys())

    models = [
        TypstSymbolClassifier(num_classes=len(classes)),
        MobileNetV4(num_classes=len(classes)),
    ]

    logger = TensorBoardLogger(OUT_DIR + "/tb_logs")  # type: ignore

    for model in models:
        compiled_model = torch.compile(model)
        num_workers = process_cpu_count() or 1

        # TODO: add code to freeze decoder weights on initial round for MobileNetV4
        trainer = L.Trainer(max_epochs=20, default_root_dir=OUT_DIR, logger=logger)
        for dataset in datasets:
            dm = MathSymbolDataModule(
                data_root=DATASET_PATH, dataset_name=dataset, fine_tuning=FINE_TUNING
            )
            # training
            trainer.fit(
                compiled_model,  # type: ignore
                datamodule=dm,  # type: ignore
            )
            # validate
            trainer.validate(datamodule=dm)  # type: ignore

        # Export ONNX.
        model.to_onnx(
            f"{OUT_DIR}/model.onnx",
            # 1 image , 3 color channels
            torch.randn(1, 3, IMG_SIZE, IMG_SIZE),
            dynamo=True,
            external_data=False,
        )

    # Generate JSON for the infer page.
    with open(f"{DATA_DIR}/symbols.json", "rb") as f:
        sym_info = msgspec.json.decode(f.read(), type=list[TypstSymInfo])
    chr_to_sym = {s.char: s for s in sym_info}
    infer = []
    for c in classes:
        sym = chr_to_sym[c]
        info = {"char": sym.char, "names": sym.names}
        if sym.markup_shorthand and sym.math_shorthand:
            info["shorthand"] = sym.markup_shorthand
        elif sym.markup_shorthand:
            info["markupShorthand"] = sym.markup_shorthand
        elif sym.math_shorthand:
            info["mathShorthand"] = sym.math_shorthand
        infer.append(info)
    with open(OUT_DIR / "infer.json", "wb") as f:
        f.write(msgspec.json.encode(infer))

    # Generate JSON for the contrib page.
    contrib = {n: s.char for s in sym_info for n in s.names}
    with open(OUT_DIR / "contrib.json", "wb") as f:
        f.write(msgspec.json.encode(contrib))
