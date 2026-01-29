"""Train the model."""

import argparse
from pathlib import Path
from typing import cast

from dataset import MathSymbolDataModule
from lightning import Trainer
from lightning.pytorch.callbacks import EMAWeightAveraging, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner
from model import CNNModel, ModelName, TimmModel
from proc_data import DATASET_REPO, get_dataset_classes
from torch import set_float32_matmul_precision
from torch.cuda import device_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")

    # misc config
    parser.add_argument(
        "--out-dir", type=str, default="build/train", help="Output directory"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--dev-run",
        action="store_true",
        help="Fast dev run (valid only when debug is True)",
    )

    # hyper params
    parser.add_argument(
        "--init-batch-size", type=int, default=64, help="Initial batch size"
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=5, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--total-epochs", type=int, default=35, help="Total number of epochs"
    )
    parser.add_argument(
        "--image-size", type=int, default=128, help="Image size (e.g., 128, 256)"
    )

    # training options
    parser.add_argument(
        "--no-find-batch-size",
        action="store_true",
        help="Disable automatic batch size finding",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable EMA weight averaging \
            (default enabled, disabled when debug is True)",
    )
    parser.add_argument("--ema-decay", type=float, default=0.99, help="EMA decay rate")
    parser.add_argument(
        "--ema-start-step", type=int, default=1200, help="Step to start EMA"
    )

    # precision
    parser.add_argument(
        "--amp-precision",
        type=str,
        default="16-mixed",
        help="Precision: 64, 32, 16-mixed, bf16-mixed",
    )

    # models
    parser.add_argument(
        "--timm-models",
        type=str,
        nargs="+",
        default=[
            "mobilenetv4_conv_small_035",
            "mobilenetv4_conv_small_050",
            "mobilenetv4_conv_small",
        ],
        help="List of timm models to train",
        choices=[
            "mobilenetv4_conv_small_035",
            "mobilenetv4_conv_small_050",
            "mobilenetv4_conv_small",
            "mobilenetv4_conv_medium",
            "mobilenetv4_hybrid_medium_075",
        ],
    )

    args = parser.parse_args()

    # Convert args to variables
    out_dir = Path(args.out_dir)
    debug: bool = args.debug
    dev_run: bool = args.dev_run
    init_batch_size: int = args.init_batch_size
    warmup_epochs: int = args.warmup_epochs
    total_epochs: int = args.total_epochs
    image_size: int = args.image_size
    find_batch_size = not args.no_find_batch_size
    use_ema = debug and not args.no_ema
    ema_decay: int = args.ema_decay
    ema_start_step: int = args.ema_start_step
    amp_precision: int = args.amp_precision
    timm_model_list: list[ModelName] = args.timm_models

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
        batch_size = init_batch_size
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
