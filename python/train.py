"""Train the model."""

from pathlib import Path
from typing import cast

import typer
from callbacks import EMAWeightAveraging, LogWrongGuessesCallback
from dataset import MathSymbolDataModule
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner
from model import CNNModel, ModelName, TimmModel
from proc_data import DATASET_REPO, get_dataset_classes
from torch import set_float32_matmul_precision

if __name__ == "__main__":

    def main(
        out_dir: str = typer.Option("build/train", help="Output directory"),
        debug: bool = typer.Option(False, help="Enable debug mode"),
        dev_run: bool = typer.Option(
            False, help="Fast dev run (valid only when debug is True)"
        ),
        log_wrong_guesses: bool = typer.Option(
            False, help="Logging wrong guesses to logger for review."
        ),
        init_batch_size: int = typer.Option(128, help="Initial batch size"),
        warmup_epochs: int = typer.Option(3, help="Number of warmup epochs"),
        total_epochs: int = typer.Option(35, help="Total number of epochs"),
        image_size: int = typer.Option(224, help="Image size (e.g., 128, 224, 256)"),
        find_batch_size: bool = typer.Option(
            True, help="Enable/Disable automatic batch size finding"
        ),
        use_ema: bool = typer.Option(
            True, "--ema/--no-ema", help="Enable/Disable EMA weight averaging"
        ),
        ema_decay: float = typer.Option(0.995, help="EMA decay rate"),
        ema_start_epoch: int = typer.Option(7, help="Epoch to start EMA"),
        amp_precision: str = typer.Option(
            "bf16-mixed", help="Precision: 64, 32, 16-mixed, bf16-mixed"
        ),
        timm_models: list[str] = typer.Option(
            [
                "mobilenetv4_conv_small_035",
                "mobilenetv4_conv_small_050",
                "mobilenetv4_conv_small",
            ],
            "--timm-models",
            help="List of timm models to train",
        ),
    ):
        """Train the model."""
        # Logic adjustment:
        # Original code used `use_ema = debug and not args.no_ema`
        # where no_ema defaulted to False (meaning EMA enabled by default).
        # This implies EMA was only enabled when debug=True AND no_ema=False.
        # This seems contradictory to the help text "default enabled",
        # but we preserve behavior.

        final_use_ema = debug and use_ema

        out_dir_path = Path(out_dir)

        # timm_model_list logic
        timm_model_list: list[ModelName] = cast("list[ModelName]", timm_models)

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
            model_name_str = (
                model.__class__.__name__
                if model.__class__.__name__ != "TimmModel"
                else cast("str", model.model_name)
            )
            logger = TensorBoardLogger(
                save_dir=out_dir_path, name=model_name_str, default_hp_metric=False
            )  # type: ignore
            callbacks: list = [LearningRateMonitor(logging_interval="epoch")]
            if log_wrong_guesses:
                callbacks.append(LogWrongGuessesCallback(sorted(classes)))
            if final_use_ema:
                callbacks.append(
                    EMAWeightAveraging(
                        decay=ema_decay, update_starting_at_epoch=ema_start_epoch
                    )
                )
            trainer = Trainer(
                max_epochs=total_epochs,
                default_root_dir=out_dir_path,
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
            if not debug and trainer.num_devices == 1 and find_batch_size:
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
                save_path = (
                    out_dir_path / "lr_find" / f"{model_name_str}_{image_size}.svg"
                )
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
                save_path = out_dir_path / "onnx" / f"{model_name_str}.onnx"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                model.to_onnx(
                    save_path,
                    model.example_input_array,
                    dynamo=True,
                    external_data=False,
                )

    typer.run(main)
