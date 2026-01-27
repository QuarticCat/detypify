"""Train the model."""

from pathlib import Path

import lightning as L  # noqa
import lightning.pytorch
import msgspec
import torch
from dataset import MathSymbolDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from model import CNNModel, TimmModel
from proc_data import (
    DATASET_ROOT,
    DETEXIFY_DATA_PATH,
    IMG_SIZE,
    DataSetName,
    TypstSymInfo,
    get_dataset_info,
)

TRAIN_OUT_DIR = Path("build/train")


if __name__ == "__main__":
    datasets: list[DataSetName] = [
        "mathwriting",
        "detexify",
        # "contrib"
    ]
    classes: set[str] = set()
    for dataset in datasets:
        data_info = get_dataset_info(dataset)
        classes.update(data_info.count_by_class.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # hyper params
    batch_size = 48
    warmup_epochs = 5
    total_epochs = 20

    models = [
        TimmModel(
            num_classes=len(classes),
            warmup_rounds=warmup_epochs,
            total_rounds=total_epochs,
            batch_size=batch_size,
        ),
        CNNModel(len(classes)),
    ]

    logger = TensorBoardLogger(TRAIN_OUT_DIR / "tb_logs")  # type: ignore

    for model in models:
        torch.set_float32_matmul_precision("high")
        trainer = L.Trainer(
            max_epochs=total_epochs,
            default_root_dir=TRAIN_OUT_DIR,
            logger=logger,
            precision="bf16-mixed",
            gradient_clip_val=0.1,
            gradient_clip_algorithm="norm",
        )
        for dataset in datasets:
            dm = MathSymbolDataModule(
                data_root=DATASET_ROOT,
                dataset_name=dataset,
                batch_size=batch_size,
                class_to_idx=class_to_idx,
            )
            # training
            trainer.fit(model, datamodule=dm)
            trainer.test(model, datamodule=dm)

        model.eval()
        model.freeze()
        model.export = True
        # Export ONNX.
        model.to_onnx(
            "model.onnx",
            torch.randn(1, 1, IMG_SIZE, IMG_SIZE),
            dynamo=True,
        )

    # Generate JSON for the infer page.
    with (DETEXIFY_DATA_PATH / "symbols.json").open("rb") as f:
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
    with (TRAIN_OUT_DIR / "infer.json").open("wb") as f:
        f.write(msgspec.json.encode(infer))

    # Generate JSON for the contrib page.
    contrib = {n: s.char for s in sym_info for n in s.names}
    with (TRAIN_OUT_DIR / "contrib.json").open("wb") as f:
        f.write(msgspec.json.encode(contrib))
