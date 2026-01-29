"""Train the model."""

from os import process_cpu_count
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

if __name__ == "__main__":
    # misc config
    out_dir = Path("build/train")
    debug: bool = False
    dev_run: bool = False  # valid only when debug is True

    # hyper params
    # use batch size scaler (only single card is supported) to adjust to your hardware
    init_batch_size = 64
    warmup_epochs = 3
    total_epochs = 35

    # NOTE: scale up image size may increase model accuracy
    # maybe more significant on mobilenetv4 models
    # ref data:
    # | model | image size | top 1
    # mobilenetv4_conv_small_035 | 256 | 87%
    # mobilenetv4_conv_small_035 | 128 | 84%
    image_size = 128

    find_batch_size = True
    use_ema = True
    ema_decay = 0.99
    ema_start_step = 1200

    # precision:
    # Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
    # 16bit mixed precision (16, '16', '16-mixed')
    # or bfloat16 mixed precision ('bf16', 'bf16-mixed').
    # Can be used on CPU, GPU, TPUs, or HPUs.
    amp_precision = "16-mixed"

    # models to be trainned
    timm_model_list: list[ModelName] = [
        "mobilenetv4_conv_small_035",
        "mobilenetv4_conv_small_050",
        "mobilenetv4_conv_small",
    ]

    classes: set[str] = get_dataset_classes(DATASET_REPO)
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
        num_workers=min(12, process_cpu_count()),
        image_size=image_size,
    )

    for model in models:
        set_float32_matmul_precision("high")
        model_name = (
            model.__class__.__name__
            if model.__class__.__name__ != "TimmModel"
            else cast("str", model.model_name)
        )
        logging_path = out_dir / "train_log"
        logging_path.mkdir(parents=True, exist_ok=True)
        logger = TensorBoardLogger(
            save_dir=logging_path, name=model_name, default_hp_metric=False
        )  # type: ignore
        callbacks: list = [LearningRateMonitor(logging_interval="epoch")]
        if use_ema:
            callbacks.append(
                EMAWeightAveraging(
                    decay=ema_decay, update_starting_at_step=ema_start_step
                )
            )
        trainer = Trainer(
            max_epochs=total_epochs,
            default_root_dir=out_dir,
            logger=logger,
            fast_dev_run=debug and dev_run,
            precision=amp_precision,  # type: ignore
            callbacks=callbacks,
        )

        # finetune learning rate and batch size
        tuner = Tuner(trainer)
        # disable compiling as it required fixed batch size
        model.use_compile = False
        # NOTE: don't use fast_dev_run=True with scale batch and lr finder
        if not debug and device_count() == 1 and find_batch_size:
            suggested_batch_size = tuner.scale_batch_size(
                model, datamodule=dm, init_val=init_batch_size
            )
            batch_size = (
                suggested_batch_size if suggested_batch_size else init_batch_size
            )
        print(f"The final batch size is {batch_size}.")
        if not debug and not dev_run:
            lr_finder = tuner.lr_find(model, datamodule=dm, min_lr=1e-5)
            fig = lr_finder.plot(suggest=True)  # type: ignore
            save_path = out_dir / "lr_find" / f"{model_name}_{image_size}.svg"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)  # type: ignore
            model.hparams.learning_rate = lr_finder.suggestion()  # type: ignore

        # training
        model.use_compile = True
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)

        if not debug:
            model.freeze()
            model.use_compile = False
            # Export ONNX.
            save_path = out_dir / "onnx" / f"{model_name}.onnx"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.to_onnx(
                save_path,
                model.example_input_array,
                dynamo=True,
                external_data=False,
            )
