"""Train the model."""

from pathlib import Path
from typing import cast

import lightning as L  # noqa
import lightning.pytorch
import torch
from dataset import MathSymbolDataModule
from datasets import load_dataset
from lightning.pytorch.callbacks import WeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from model import CNNModel, ModelName, TimmModel
from proc_data import DATASET_REPO, IMG_SIZE
from torch.optim.swa_utils import get_ema_avg_fn

TRAIN_OUT_DIR = Path("build/train")
DEBUG = False


class EMAWeightAveraging(WeightAveraging):
    def __init__(self):
        super().__init__(avg_fn=get_ema_avg_fn())

    def should_update(self, step_idx=None, epoch_idx=None):
        return (step_idx is not None) and (step_idx >= 100) and (epoch_idx is not None)


if __name__ == "__main__":
    dm = MathSymbolDataModule()
    classes: set[str] = set()
    dataset = load_dataset(DATASET_REPO)
    for split in dataset:
        classes.update(dataset[split].features["label"].names)

    # hyper params
    batch_size = 48
    warmup_epochs = 5
    total_epochs = 40

    timm_model_list: list[ModelName] = [
        "mobilenetv4_conv_small_035",
        "mobilenetv4_conv_small_050",
        "mobilenetv4_conv_small",
    ]
    models = [
        CNNModel(num_classes=len(classes), image_size=IMG_SIZE),
    ]
    for model_name in timm_model_list:
        model = TimmModel(
            num_classes=len(classes),
            model_name=model_name,
            warmup_rounds=warmup_epochs,
            total_rounds=total_epochs,
            image_size=IMG_SIZE,
        )
        models.append

    dm = MathSymbolDataModule(
        batch_size=batch_size,
        image_size=IMG_SIZE,
    )

    for model in models:
        torch.set_float32_matmul_precision("medium")
        model_name = (
            model.__class__.__name__
            if model.__class__.__name__ != "TimmModel"
            else cast("str", model.model_name)
        )
        logger = TensorBoardLogger(save_dir=TRAIN_OUT_DIR, name=model_name)  # type: ignore
        trainer = L.Trainer(
            max_epochs=total_epochs,
            default_root_dir=TRAIN_OUT_DIR,
            logger=logger,
            fast_dev_run=DEBUG,
            precision="bf16-mixed",
            gradient_clip_val=0.1,
            gradient_clip_algorithm="norm",
            accumulate_grad_batches=10,
            callbacks=EMAWeightAveraging(),
        )

        # finetune learning rate and batch size
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, mode="power", datamodule=dm)
        lr_finder = tuner.lr_find(model)
        model.hparams.learning_rate = lr_finder.suggestion()  # type: ignore

        # training
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)

        if not DEBUG:
            model.freeze()
            model.export = True
            # Export ONNX.
            model.to_onnx(
                TRAIN_OUT_DIR / f"{model_name}.onnx",
                model.example_input_array,
                dynamo=True,
                external_data=False,
            )

    # TODO: migrate to `proc_data.py`

    # Generate JSON for the infer page.
    # infer_path = TRAIN_OUT_DIR / "infer.json"
    # contrib_path = TRAIN_OUT_DIR / "contrib.json"
    # with (DATASET_ROOT / "symbols.json").open("rb") as f:
    #     sym_info = msgspec.json.decode(f.read(), type=list[TypstSymInfo])
    # chr_to_sym = {s.char: s for s in sym_info}
    # if not infer_path.exists():
    #     infer = []
    #     for c in classes:
    #         sym = chr_to_sym[c]
    #         info = {"char": sym.char, "names": sym.names}
    #         if sym.markup_shorthand and sym.math_shorthand:
    #             info["shorthand"] = sym.markup_shorthand
    #         elif sym.markup_shorthand:
    #             info["markupShorthand"] = sym.markup_shorthand
    #         elif sym.math_shorthand:
    #             info["mathShorthand"] = sym.math_shorthand
    #             infer.append(info)
    #     with infer_path.open("wb") as f:
    #         f.write(msgspec.json.encode(infer))

    # # Generate JSON for the contrib page.
    # if not contrib_path.exists():
    #     contrib = {n: s.char for s in sym_info for n in s.names}
    #     with contrib_path.open("wb") as f:
    #         f.write(msgspec.json.encode(contrib))
