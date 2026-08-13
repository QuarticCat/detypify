"""Train the model."""

import logging
from dataclasses import asdict, dataclass, field
from os import process_cpu_count
from pathlib import Path
from typing import Annotated

import cappa
from detypify.data.paths import DEFAULT_DATA_PATHS

CUDA_AMPERE_VERSION = 8


@cappa.command(name="train", default_long=True)
@dataclass
class Args:
    """Train the model."""

    models: list[str] = field(default_factory=lambda: ["mobilenet_v4_035"])
    """Models to train, formatted as mobilenet_{v4|v5}_{size}. Size is divided by 100."""

    out_dir: Path = DEFAULT_DATA_PATHS.train_dir
    """Output directory."""

    debug: bool = False
    """Enable debug mode."""

    profiling: bool = False
    """Enable performance profiler."""

    dev_run: bool = False
    """Fast dev run (valid only when debug is true)."""

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

    # Lazy import
    from detypify.config import DataSetName
    from detypify.data.datasets import get_dataset_classes
    from detypify.training.datamodule import MathSymbolDataModule
    from detypify.training.model import MobileNetModel
    from lightning import Trainer
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.tuner.tuning import Tuner
    from msgspec import yaml
    from torch import set_float32_matmul_precision
    from torch.cuda import is_bf16_supported

    dataset_names = (DataSetName.detexify, DataSetName.mathwriting)
    is_debug_dev_run = args.debug and args.dev_run
    dev_max_samples = 2048 if is_debug_dev_run else None
    if is_debug_dev_run:
        args.num_workers = 0
    classes = get_dataset_classes(dataset_names, max_samples=dev_max_samples)

    # compatibility check for graphics
    if not is_bf16_supported() and args.amp_precision == "bf16-mixed":
        logger.warning("Current device don't support bfloat16 precision, use float16 instead.")
        args.amp_precision = "16-mixed"
    else:
        # use low precision acceleration
        set_float32_matmul_precision("medium")
    model_instances: list[MobileNetModel] = [
        MobileNetModel(
            num_classes=len(classes),
            model_name=model,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.total_epochs,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            label_smoothing=args.label_smoothing,
            use_compile=args.use_compile and not args.debug,
        )
        for model in args.models
    ]

    # define data module
    dm = MathSymbolDataModule(
        batch_size=args.init_batch_size,
        image_size=args.image_size,
        dataset_names=dataset_names,
        max_samples=dev_max_samples,
        num_workers=args.num_workers,
    )

    for model in model_instances:
        model_name_str = model.model_name
        tb_logger = TensorBoardLogger(save_dir=args.out_dir, name=model_name_str, default_hp_metric=False)  # type: ignore

        final_output_dir = Path(tb_logger.log_dir)
        checkpoints_dir = final_output_dir / "ckpts"
        train_args_path = final_output_dir / "training_args.yaml"
        train_args_path.parent.mkdir(parents=True, exist_ok=True)

        current_args = {**asdict(args), "model_name": model_name_str, "num_classes": len(classes)}

        if not args.debug:
            with train_args_path.open("wb") as f:
                f.write(yaml.encode(current_args, enc_hook=str))

        callbacks: list = [LearningRateMonitor(logging_interval="epoch")]

        # Lazy import callbacks only when needed
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

        # Add checkpoint callback to save best model
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename="best-{epoch:02d}-{val_acc:.4f}",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        # Add ONNX export callback for best model
        if not args.debug:
            from detypify.training.callbacks import ExportBestModelToONNX

            callbacks.append(
                ExportBestModelToONNX(
                    save_dir=checkpoints_dir,
                    model_name=model_name_str,
                    checkpoint_callback=checkpoint_callback,
                    use_compile=args.use_compile and not args.debug,
                )
            )

        trainer = Trainer(
            max_epochs=args.total_epochs,
            default_root_dir=args.out_dir,
            logger=tb_logger,
            fast_dev_run=args.debug and args.dev_run,
            accelerator="cpu" if is_debug_dev_run else "auto",
            precision="32-true" if is_debug_dev_run else args.amp_precision,  # type: ignore
            profiler="simple" if args.profiling else None,
            callbacks=callbacks,
        )

        # finetune learning rate and batch size
        tuner = Tuner(trainer)
        # disable compiling as it required fixed batch size
        model.use_compile = False
        # NOTE: don't use fast_dev_run=True with scale batch and lr finder
        batch_size = args.init_batch_size
        if not args.debug and trainer.num_devices == 1 and args.find_batch_size:
            suggested_batch_size = tuner.scale_batch_size(model, datamodule=dm, init_val=args.init_batch_size)
            batch_size = suggested_batch_size or args.init_batch_size
        logger.info("The final batch size is %s.", batch_size)
        if args.find_lr and not args.debug and not args.dev_run:
            min_lr = min(1e-4, args.learning_rate / 20)
            max_lr = max(1e-3, args.learning_rate * 5)
            lr_finder = tuner.lr_find(model, datamodule=dm, min_lr=min_lr, max_lr=max_lr)
            fig = lr_finder.plot(suggest=True)  # type: ignore
            save_path = final_output_dir / f"lr_{batch_size}_{args.image_size}.svg"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)  # type: ignore
            suggested_lr = lr_finder.suggestion()
            if suggested_lr is not None:
                model.learning_rate = suggested_lr
                model.hparams["learning_rate"] = suggested_lr

        current_args["final_batch_size"] = batch_size
        current_args["effective_learning_rate"] = model.learning_rate
        logger.info("The effective learning rate is %s.", model.learning_rate)

        with train_args_path.open("wb") as f:
            f.write(yaml.encode(current_args, enc_hook=str))

        # training
        dm.batch_size = batch_size
        model.use_compile = args.use_compile and not args.debug
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm, ckpt_path="best")
