"""Train the model."""

from pathlib import Path
from typing import cast

from dataset import MathSymbolDataModule
from datasets import load_dataset
from lightning import Trainer
from lightning.pytorch.callbacks import EMAWeightAveraging, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from model import CNNModel, ModelName, TimmModel
from proc_data import DATASET_REPO
from torch import set_float32_matmul_precision

TRAIN_OUT_DIR = Path("build/train")
DEBUG = True
DEV_RUN = False


if __name__ == "__main__":
    # hyper params
    # use batch size scaler to adjust to your hardware
    init_batch_size = 256
    warmup_epochs = 7
    total_epochs = 40
    # scale up may increase accuracy, other params should be changed accordingly.
    image_size = 128

    classes: set[str] = set()
    dataset = load_dataset(DATASET_REPO)
    for split in dataset:
        classes.update(dataset[split].features["label"].names)

    # define models
    timm_model_list: list[ModelName] = [
        "mobilenetv4_conv_small_035",
        "mobilenetv4_conv_small_050",
        "mobilenetv4_conv_small",
    ]
    models: list[CNNModel | TimmModel] = [
        CNNModel(num_classes=len(classes), image_size=image_size),
    ]
    for model_name in timm_model_list:
        model = TimmModel(
            num_classes=len(classes),
            model_name=model_name,
            warmup_rounds=warmup_epochs,
            total_rounds=total_epochs,
            image_size=image_size,
        )
        models.append(model)

    # define data module
    dm = MathSymbolDataModule(
        batch_size=init_batch_size,
        image_size=image_size,
    )

    for model in models:
        set_float32_matmul_precision("medium")
        model_name = (
            model.__class__.__name__
            if model.__class__.__name__ != "TimmModel"
            else cast("str", model.model_name)
        )
        logging_path = TRAIN_OUT_DIR / "train_log"
        logging_path.mkdir(parents=True, exist_ok=True)
        logger = TensorBoardLogger(
            save_dir=logging_path, name=model_name, default_hp_metric=False
        )  # type: ignore
        trainer = Trainer(
            max_epochs=total_epochs,
            default_root_dir=TRAIN_OUT_DIR,
            logger=logger,
            fast_dev_run=DEV_RUN,
            precision="bf16-mixed",
            gradient_clip_val=0.1,
            gradient_clip_algorithm="norm",
            accumulate_grad_batches=16,
            callbacks=[
                EMAWeightAveraging(decay=0.9998, update_starting_at_step=100),
                LearningRateMonitor(
                    logging_interval="epoch", log_momentum=True, log_weight_decay=True
                ),
            ],
        )

        # finetune learning rate and batch size
        tuner = Tuner(trainer)
        # disable compiling as it required fixed batch size
        model.use_compile = False
        # NOTE: don't use fast_dev_run=True with scale batch and lr finder
        tuner.scale_batch_size(model, datamodule=dm, init_val=init_batch_size)
        lr_finder = tuner.lr_find(model, datamodule=dm, min_lr=1e-5)
        if DEBUG:
            fig = lr_finder.plot(suggest=True)  # type: ignore
            save_path = TRAIN_OUT_DIR / "lr_find" / f"{model_name}_{image_size}.svg"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)  # type: ignore
        model.hparams.learning_rate = lr_finder.suggestion()  # type: ignore

        # training
        model.use_compile = True
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)

        if not DEBUG:
            model.freeze()
            model.use_compile = False
            # Export ONNX.
            save_path = TRAIN_OUT_DIR / "onnx" / f"{model_name}.onnx"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.to_onnx(
                save_path,
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
