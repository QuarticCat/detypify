"""Train the model."""

import logging
from dataclasses import asdict, dataclass, field
from os import process_cpu_count
from pathlib import Path
from typing import Annotated

import cappa

from detypify.data.paths import DEFAULT_DATA_PATHS


@cappa.command(name="train", default_long=True)
@dataclass
class Args:
    """Train the model."""

    models: list[str] = field(default_factory=lambda: ["mobilenet_v4_035"])
    """Models to train, formatted as mobilenet_{v4|v5}_{size}. Size is divided by 100."""

    out_dir: Path = DEFAULT_DATA_PATHS.train_dir
    """Output directory."""

    profiling: bool = False
    """Enable performance profiler."""

    log_pred: Annotated[bool, cappa.Arg(long="--log-pred/--no-log-pred")] = True
    """Log predictions for review."""

    init_batch_size: int = 128
    """Initial batch size."""

    warmup_epochs: int = 3
    """Number of warmup epochs."""

    total_epochs: int = 40
    """Total number of epochs."""

    learning_rate: float = 0.002
    """Learning rate used when the LR finder is disabled."""

    label_smoothing: float = 0.0
    """Cross entropy label smoothing factor."""

    image_size: int = 224
    """Image size (e.g., 128, 224, 256)."""

    find_batch_size: bool = False
    """Enable automatic batch size finding."""

    find_lr: Annotated[bool, cappa.Arg(long="--find-lr/--no-find-lr")] = True
    """Enable learning rate finder."""

    num_workers: int = process_cpu_count() or 1
    """Number of DataLoader workers."""

    use_ema: Annotated[bool, cappa.Arg(long="--ema/--no-ema")] = True
    """Enable EMA weight averaging."""

    ema_decay: float = 0.995
    """EMA decay rate."""

    ema_warmup: Annotated[bool, cappa.Arg(long="--ema-warmup/--no-ema-warmup")] = True
    """Enable EMA warmup."""

    ema_warmup_gamma: float = 25.0
    """EMA warmup gamma."""

    ema_warmup_power: float = 0.7
    """EMA warmup power."""

    amp_precision: str = "bf16-mixed"
    """Precision: 64, 32, 16-mixed, bf16-mixed."""

    use_compile: Annotated[bool, cappa.Arg(long="--compile/--no-compile")] = False
    """Enable torch.compile."""


if __name__ == "__main__":
    args = cappa.parse(Args, completion=False)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Delay the heavy training stack until after CLI parsing so help output stays lightweight.
    from lightning import Trainer
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.tuner.tuning import Tuner
    from msgspec import yaml
    from torch import set_float32_matmul_precision
    from torch.cuda import is_bf16_supported

    from detypify.config import DataSetName
    from detypify.data.datasets import get_dataset_classes
    from detypify.training.datamodule import MathSymbolDataModule
    from detypify.training.model import MobileNetModel

    dataset_names = (DataSetName.detexify, DataSetName.mathwriting)
    classes = get_dataset_classes(dataset_names)

    # Emulated BF16 can be reported as supported but is not a useful training acceleration path.
    if not is_bf16_supported(including_emulation=False) and args.amp_precision == "bf16-mixed":
        logger.warning("Current device does not support native bfloat16 precision; using float16 instead.")
        args.amp_precision = "16-mixed"
    else:
        set_float32_matmul_precision("medium")
    # All model variants share the same deterministic dataset partitions and class order.
    dm = MathSymbolDataModule(
        batch_size=args.init_batch_size,
        image_size=args.image_size,
        dataset_names=dataset_names,
        num_workers=args.num_workers,
    )

    for model_name in args.models:
        model = MobileNetModel(
            num_classes=len(classes),
            model_name=model_name,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.total_epochs,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            label_smoothing=args.label_smoothing,
            use_compile=args.use_compile,
        )

        # Each model gets an isolated TensorBoard version, checkpoints, and effective-argument record.
        tb_logger = TensorBoardLogger(save_dir=args.out_dir, name=model_name, default_hp_metric=False)  # type: ignore

        final_output_dir = Path(tb_logger.log_dir)
        checkpoints_dir = final_output_dir / "ckpts"
        train_args_path = final_output_dir / "training_args.yaml"
        train_args_path.parent.mkdir(parents=True, exist_ok=True)

        current_args = {**asdict(args), "model_name": model_name, "num_classes": len(classes)}

        with train_args_path.open("wb") as f:
            f.write(yaml.encode(current_args, enc_hook=str))

        callbacks: list = [LearningRateMonitor(logging_interval="epoch")]

        # Optional callbacks pull in plotting or averaging dependencies only when enabled.
        if args.log_pred:
            from detypify.training.callbacks import LogPredictCallback, LogTestConfusionCallback

            callbacks.extend([LogPredictCallback(classes), LogTestConfusionCallback(classes)])

        if args.use_ema:
            from detypify.training.callbacks import EMAWeightAveraging

            callbacks.append(
                EMAWeightAveraging(
                    decay=args.ema_decay,
                    use_warmup=args.ema_warmup,
                    warmup_gamma=args.ema_warmup_gamma,
                    warmup_power=args.ema_warmup_power,
                )
            )

        # Keep the best validation checkpoint for evaluation and the last checkpoint for resuming.
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename="best-{epoch:02d}-{val_acc:.4f}",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        from detypify.training.callbacks import ExportBestModelToONNX

        callbacks.append(
            ExportBestModelToONNX(
                save_dir=checkpoints_dir,
                model_name=model_name,
                checkpoint_callback=checkpoint_callback,
            )
        )

        trainer = Trainer(
            max_epochs=args.total_epochs,
            default_root_dir=args.out_dir,
            logger=tb_logger,
            accelerator="auto",
            precision=args.amp_precision,  # type: ignore
            profiler="simple" if args.profiling else None,
            callbacks=callbacks,
        )

        # Tune with eager execution because probing uses variable batch sizes and short trial runs.
        tuner = Tuner(trainer)
        model.use_compile = False
        batch_size = args.init_batch_size
        if trainer.num_devices == 1 and args.find_batch_size:
            suggested_batch_size = tuner.scale_batch_size(model, datamodule=dm, init_val=args.init_batch_size)
            batch_size = suggested_batch_size or args.init_batch_size
        logger.info("The final batch size is %s.", batch_size)
        if args.find_lr:
            min_lr = min(1e-4, args.learning_rate / 20)
            max_lr = max(1e-3, args.learning_rate * 5)
            lr_finder = tuner.lr_find(model, datamodule=dm, min_lr=min_lr, max_lr=max_lr)
            if lr_finder is not None:
                fig = lr_finder.plot(suggest=True)  # type: ignore
                save_path = final_output_dir / f"lr_{batch_size}_{args.image_size}.svg"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path)  # type: ignore
                suggested_lr = lr_finder.suggestion()  # type: ignore
                if suggested_lr is not None:
                    model.learning_rate = suggested_lr
                    model.hparams["learning_rate"] = suggested_lr

        current_args["final_batch_size"] = batch_size
        current_args["effective_learning_rate"] = model.learning_rate
        logger.info("The effective learning rate is %s.", model.learning_rate)

        with train_args_path.open("wb") as f:
            f.write(yaml.encode(current_args, enc_hook=str))

        # Restore the requested compiled path only after tuning has fixed the effective settings.
        dm.batch_size = batch_size
        model.use_compile = args.use_compile
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm, ckpt_path="best")
