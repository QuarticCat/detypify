"""Train the model."""

from os import process_cpu_count
from pathlib import Path

import lightning as L  # noqa
import lightning.pytorch
import msgspec
import torch
from dataset import MathSymbolDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from model import MobileNetV4, TypstSymbolClassifier
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

    # hyper params
    batch_size = 64
    warmup_epochs = 20
    total_epochs = 200

    models = [
        MobileNetV4(
            num_classes=len(classes),
            warmup_rounds=warmup_epochs,
            total_rounds=total_epochs,
        ),
        TypstSymbolClassifier(num_classes=len(classes)),
    ]

    logger = TensorBoardLogger(TRAIN_OUT_DIR / "train" / "tb_logs")  # type: ignore

    for model in models:
        compiled_model = torch.compile(model)
        num_workers = process_cpu_count() or 1

        trainer = L.Trainer(
            max_epochs=total_epochs,
            default_root_dir=TRAIN_OUT_DIR,
            logger=logger,
            # See: https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
            # this should be the same as torch.set_float32_matmul_precision("high")
            precision="bf16-mixed",
        )
        for dataset in datasets:
            dm = MathSymbolDataModule(
                data_root=DATASET_ROOT,
                dataset_name=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            # training
            trainer.fit(
                model,
                datamodule=dm,  # type: ignore
            )
            # validate
            trainer.validate(datamodule=dm)  # type: ignore

        # Export ONNX.
        model.to_onnx(
            f"{TRAIN_OUT_DIR}/model.onnx",
            # 1 image , 1 color channels(grey scale)
            torch.randn(1, 1, IMG_SIZE, IMG_SIZE),
            dynamo=True,
            external_data=False,
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
