"""Train the model."""

import logging

import typer

CUDA_AMPERE_VERSION = 8

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = typer.Typer(pretty_exceptions_show_locals=False)

    @app.command()
    def main(
        out_dir: str = typer.Option("build/train", help="Output directory"),
        debug: bool = typer.Option(False, help="Enable debug mode"),
        profiling: bool = typer.Option(False, help="Enable performance profiler."),
        dev_run: bool = typer.Option(False, help="Fast dev run (valid only when debug is True)"),
        log_pred: bool = typer.Option(True, help="Logging predictions to logger for review."),
        init_batch_size: int = typer.Option(128, help="Initial batch size"),
        warmup_epochs: int = typer.Option(3, help="Number of warmup epochs"),
        total_epochs: int = typer.Option(40, help="Total number of epochs"),
        image_size: int = typer.Option(224, help="Image size (e.g., 128, 224, 256)"),
        find_batch_size: bool = typer.Option(False, help="Enable/Disable automatic batch size finding"),
        use_ema: bool = typer.Option(True, "--ema/--no-ema", help="Enable/Disable EMA weight averaging"),
        ema_decay: float = typer.Option(0.995, help="EMA decay rate"),
        ema_start_epoch: int = typer.Option(5, help="Epoch to start EMA"),
        ema_warmup: bool = typer.Option(True, "--ema-warmup/--no-ema-warmup", help="Enable/Disable EMA warmup."),
        ema_warmup_gamma: float = typer.Option(25.0, help="EMA warmup gamma."),
        ema_warmup_power: float = typer.Option(0.7, help="EMA warmup power."),
        amp_precision: str = typer.Option("bf16-mixed", help="Precision: 64, 32, 16-mixed, bf16-mixed"),
        models: list[str] = typer.Option(
            [
                "mobilenetv4_conv_small_035",
                "mobilenetv4_conv_small_050",
            ],
            "--models",
            help="List of models to train (use 'CNNModel' for built-in CNN)",
        ),
    ):
        """Train the model."""
        # Collect all input arguments
        args_dict = {
            "out_dir": out_dir,
            "debug": debug,
            "profiling": profiling,
            "dev_run": dev_run,
            "log_pred": log_pred,
            "init_batch_size": init_batch_size,
            "warmup_epochs": warmup_epochs,
            "total_epochs": total_epochs,
            "image_size": image_size,
            "find_batch_size": find_batch_size,
            "use_ema": use_ema,
            "ema_decay": ema_decay,
            "ema_start_epoch": ema_start_epoch,
            "ema_warmup": ema_warmup,
            "ema_warmup_gamma": ema_warmup_gamma,
            "ema_warmup_power": ema_warmup_power,
            "amp_precision": amp_precision,
            "models": models,
        }

        # Lazy import
        from pathlib import Path

        from dataset import MathSymbolDataModule
        from lightning import Trainer
        from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
        from lightning.pytorch.loggers import TensorBoardLogger
        from lightning.pytorch.tuner.tuning import Tuner
        from model import CNNModel, TimmModel
        from msgspec import yaml
        from proc_data import DATASET_REPO, get_dataset_classes
        from torch import set_float32_matmul_precision
        from torch.cuda import get_device_properties, is_available

        out_dir_path = Path(out_dir)

        classes = get_dataset_classes(DATASET_REPO)
        model_instances: list[CNNModel | TimmModel] = []
        for model_name in models:
            if model_name == "CNN":
                model_instances.append(
                    CNNModel(
                        num_classes=len(classes),
                        image_size=image_size,
                        total_epochs=total_epochs,
                        warmup_epochs=warmup_epochs,
                    )
                )
            else:
                model_instances.append(
                    TimmModel(
                        num_classes=len(classes),
                        model_name=model_name,  # type: ignore[arg-type]
                        warmup_epochs=warmup_epochs,
                        total_epochs=total_epochs,
                        image_size=image_size,
                    )
                )

        # define data module
        dm = MathSymbolDataModule(
            batch_size=init_batch_size,
            image_size=image_size,
        )

        # for Ampere or later NVIDIA graphics only
        if is_available():
            if get_device_properties(0).major >= CUDA_AMPERE_VERSION:
                # set matmul precision for speed
                set_float32_matmul_precision("medium")
            # 20 series graphics don't support bf16 format precision
            elif amp_precision == "bf16-mixed":
                amp_precision = "16-mixed"
                args_dict["amp_precision"] = amp_precision
        for model in model_instances:
            model_name_str = (
                model.__class__.__name__ if model.__class__.__name__ != "TimmModel" else str(model.model_name)
            )
            logger = TensorBoardLogger(save_dir=out_dir_path, name=model_name_str, default_hp_metric=False)  # type: ignore

            final_output_dir = Path(logger.log_dir)
            checkpoints_dir = final_output_dir / "ckpts"
            train_args_path = final_output_dir / "training_args.yaml"
            train_args_path.parent.mkdir(parents=True, exist_ok=True)

            current_args = args_dict.copy()
            current_args.update(
                {
                    "model_name": model_name_str,
                    "num_classes": len(classes),
                }
            )

            if not debug:
                with train_args_path.open("wb") as f:
                    f.write(yaml.encode(current_args))

            callbacks: list = [LearningRateMonitor(logging_interval="epoch")]

            # Lazy import callbacks only when needed
            if log_pred:
                from callbacks import LogPredictCallback

                callbacks.append(LogPredictCallback(sorted(classes)))

            if use_ema:
                from callbacks import EMAWeightAveraging

                callbacks.append(
                    EMAWeightAveraging(
                        decay=ema_decay,
                        update_starting_at_epoch=ema_start_epoch,
                        use_warmup=ema_warmup,
                        warmup_gamma=ema_warmup_gamma,
                        warmup_power=ema_warmup_power,
                    )
                )

            # Add checkpoint callback to save best model
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoints_dir,
                save_weights_only=True,
                filename="best-{epoch:02d}-{val_acc:.4f}",
                monitor="val_acc",
                mode="max",
                save_top_k=1,
                save_last=True,
            )
            callbacks.append(checkpoint_callback)

            # Add ONNX export callback for best model
            if not debug:
                from callbacks import ExportBestModelToONNX

                callbacks.append(
                    ExportBestModelToONNX(
                        save_dir=checkpoints_dir,
                        model_name=model_name_str,
                        checkpoint_callback=checkpoint_callback,
                    )
                )

            trainer = Trainer(
                max_epochs=total_epochs,
                default_root_dir=out_dir_path,
                logger=logger,
                fast_dev_run=debug and dev_run,
                precision=amp_precision,  # type: ignore
                profiler="simple" if profiling else None,
                callbacks=callbacks,
            )

            # finetune learning rate and batch size
            tuner = Tuner(trainer)
            # disable compiling as it required fixed batch size
            model.use_compile = False
            # NOTE: don't use fast_dev_run=True with scale batch and lr finder
            batch_size = init_batch_size
            if not debug and trainer.num_devices == 1 and find_batch_size:
                suggested_batch_size = tuner.scale_batch_size(model, datamodule=dm, init_val=init_batch_size)
                batch_size = suggested_batch_size or init_batch_size
            logging.info("The final batch size is %s.", batch_size)
            if not debug and not dev_run:
                lr_finder = tuner.lr_find(model, datamodule=dm, min_lr=1e-4, max_lr=1e-3)
                fig = lr_finder.plot(suggest=True)  # type: ignore
                save_path = final_output_dir / f"lr_{batch_size}_{image_size}.svg"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path)  # type: ignore
                model.hparams.learning_rate = lr_finder.suggestion()  # type: ignore

            current_args["final_batch_size"] = batch_size
            lr = model.hparams.get("learning_rate")
            if lr is not None:
                current_args["suggested_learning_rate"] = lr

            with train_args_path.open("wb") as f:
                f.write(yaml.encode(current_args))

            # training
            model.use_compile = True
            trainer.fit(model, datamodule=dm)
            trainer.test(model, datamodule=dm)

    app()
