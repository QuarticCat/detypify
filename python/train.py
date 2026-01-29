"""Train the model."""

from pathlib import Path
from typing import cast

from dataset import MathSymbolDataModule
from lightning import Trainer
from lightning.pytorch.callbacks import EMAWeightAveraging, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from model import CNNModel, ModelName, TimmModel
from proc_data import DATASET_REPO, get_dataset_classes
from torch import set_float32_matmul_precision
from torch.cuda import device_count

TRAIN_OUT_DIR = Path("build/train")
DEBUG = False
DEV_RUN = False


if __name__ == "__main__":
    # hyper params
    # use batch size scaler to adjust to your hardware
    init_batch_size = 32
    warmup_epochs = 3
    total_epochs = 35
    # scale up may increase model accuracy
    # maybe more significant on mobilenet models
    # ref data:
    # | model | image size | top 1
    # mobilenetv4_conv_small_035 | 256 | 87%
    # mobilenetv4_conv_small_035 | 128 | 84%
    image_size = 128

    classes: set[str] = get_dataset_classes(DATASET_REPO)

    # define models
    timm_model_list: list[ModelName] = [
        "mobilenetv4_conv_small_035",
        "mobilenetv4_conv_small_050",
        "mobilenetv4_conv_small",
    ]
    models: list[CNNModel | TimmModel] = [
        CNNModel(
            num_classes=len(classes),
            image_size=image_size,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
        ),
    ]
    for model_name in timm_model_list:
        model = TimmModel(
            num_classes=len(classes),
            model_name=model_name,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            image_size=image_size,
        )
        models.append(model)

    # define data module
    dm = MathSymbolDataModule(
        batch_size=init_batch_size,
        num_workers=8,
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
            precision="16-mixed",
            callbacks=[
                EMAWeightAveraging(decay=0.99, update_starting_at_step=1200),
                LearningRateMonitor(logging_interval="epoch"),
            ],
        )

        # finetune learning rate and batch size
        tuner = Tuner(trainer)
        # disable compiling as it required fixed batch size
        model.use_compile = False
        # NOTE: don't use fast_dev_run=True with scale batch and lr finder
        if device_count() == 1:
            suggested_batch_size = tuner.scale_batch_size(
                model, datamodule=dm, init_val=init_batch_size
            )
            batch_size = (
                suggested_batch_size if suggested_batch_size else init_batch_size
            )
        print(f"The batch size is {batch_size}.")
        lr_finder = tuner.lr_find(model, datamodule=dm, min_lr=1e-6)
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
