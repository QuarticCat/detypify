"""Train the model."""

from pathlib import Path

import lightning as L  # noqa
import lightning.pytorch
import msgspec
import torch
from dataset import MathSymbolDataModule
from datasets import load_dataset
from lightning.pytorch.loggers import TensorBoardLogger
from model import CNNModel, TimmModel
from proc_data import (
    DATASET_REPO,
    DATASET_ROOT,
    IMG_SIZE,
    TypstSymInfo,
)

TRAIN_OUT_DIR = Path("build/train")


if __name__ == "__main__":
    dm = MathSymbolDataModule()
    classes: set[str] = set()
    dataset = load_dataset(DATASET_REPO)
    for split in dataset:
        classes.update(dataset[split].features["label"].names)

    # hyper params
    batch_size = 48
    warmup_epochs = 5
    total_epochs = 10

    models = [
        CNNModel(
            num_classes=len(classes),
        ),
        TimmModel(
            num_classes=len(classes),
            model_name="mobilenetv4_conv_small_035",
            warmup_rounds=warmup_epochs,
            total_rounds=total_epochs,
            batch_size=batch_size,
        ),
        TimmModel(
            num_classes=len(classes),
            model_name="mobilenetv4_conv_small_050",
            warmup_rounds=warmup_epochs,
            total_rounds=total_epochs,
            batch_size=batch_size,
        ),
        TimmModel(
            num_classes=len(classes),
            model_name="mobilenetv4_conv_small",
            warmup_rounds=warmup_epochs,
            total_rounds=total_epochs,
            batch_size=batch_size,
        ),
        TimmModel(
            num_classes=len(classes),
            model_name="efficientnet_b1",
            warmup_rounds=warmup_epochs,
            total_rounds=total_epochs,
            batch_size=batch_size,
        ),
        TimmModel(
            num_classes=len(classes),
            model_name="mobilenetv4_hybrid_medium",
            warmup_rounds=warmup_epochs,
            total_rounds=total_epochs,
            batch_size=batch_size,
        ),
    ]

    logger = TensorBoardLogger(TRAIN_OUT_DIR)  # type: ignore

    for model in models:
        torch.set_float32_matmul_precision("high")
        trainer = L.Trainer(
            max_epochs=total_epochs,
            default_root_dir=TRAIN_OUT_DIR,
            logger=logger,
            # fast_dev_run=True,
            precision="bf16-mixed",
            gradient_clip_val=0.1,
            gradient_clip_algorithm="norm",
        )
        dm = MathSymbolDataModule(
            batch_size=batch_size,
            image_size=IMG_SIZE,
        )
        # training
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)

        model.freeze()
        model.export = True
        # Export ONNX.
        name = (
            model.__class__.__name__
            if model.__class__.__name__ != "TimmModel"
            else model.model_name
        )
        model.to_onnx(
            TRAIN_OUT_DIR / f"{model.__class__.__name__}.onnx",
            torch.randn(1, 1, IMG_SIZE, IMG_SIZE),
            dynamo=True,
        )

    # Generate JSON for the infer page.
    with (DATASET_ROOT / "symbols.json").open("rb") as f:
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
