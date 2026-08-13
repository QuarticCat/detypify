"""Evaluate an existing checkpoint and log test diagnostics."""

from dataclasses import dataclass, field
from os import process_cpu_count
from pathlib import Path
from typing import Annotated

import cappa
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from torch.cuda import is_bf16_supported

from detypify.config import DataSetName
from detypify.data.datasets import get_dataset_classes
from detypify.data.paths import DEFAULT_DATA_PATHS
from detypify.training.callbacks import LogPredictCallback, LogTestConfusionCallback
from detypify.training.datamodule import MathSymbolDataModule
from detypify.training.model import MobileNetModel


@cappa.command(name="test", default_long=True)
@dataclass
class Args:
    """Run only the test phase for an existing Lightning checkpoint."""

    ckpt_path: Annotated[Path, cappa.Arg(short="-c")]
    """Checkpoint path to evaluate."""

    out_dir: Path = DEFAULT_DATA_PATHS.train_dir / "_eval"
    """TensorBoard output directory."""

    run_name: str = "existing_model"
    """TensorBoard run name."""

    datasets: Annotated[list[DataSetName], cappa.Arg(short="-d")] = field(
        default_factory=lambda: [DataSetName.detexify, DataSetName.mathwriting]
    )
    """Datasets to use for the test split."""

    batch_size: int = 128
    """Evaluation batch size."""

    image_size: int | None = None
    """Override image size. Defaults to the checkpoint hparams image size."""

    num_workers: int = process_cpu_count() or 1
    """Number of DataLoader workers."""

    amp_precision: str = "32-true"
    """Precision: 64, 32, 16-mixed, bf16-mixed."""

    log_predictions: Annotated[bool, cappa.Arg(long="--log-predictions/--no-log-predictions")] = True
    """Log prediction image grids."""

    top_false_labels: int = 10
    """Number of most frequent false predicted labels to log."""

    examples_per_label: int = 4
    """Image examples to log for each top false predicted label."""

    max_confusion_labels: int = 40
    """Maximum labels shown in the confusion-matrix figure."""


if __name__ == "__main__":
    args = cappa.parse(Args, completion=False)
    if args.amp_precision == "bf16-mixed" and not is_bf16_supported(including_emulation=False):
        args.amp_precision = "16-mixed"

    model = MobileNetModel.load_from_checkpoint(args.ckpt_path)
    args.image_size = args.image_size or int(model.hparams["image_size"])
    # Recreate the checkpoint's sorted target space from the exact dataset selection under evaluation.
    dataset_names = tuple(dict.fromkeys(args.datasets))
    classes = get_dataset_classes(dataset_names)

    dm = MathSymbolDataModule(
        batch_size=args.batch_size,
        image_size=args.image_size,
        dataset_names=dataset_names,
        num_workers=args.num_workers,
    )

    # Confusion diagnostics are always emitted; image grids can be disabled for lighter evaluations.
    callbacks: list[Callback] = [
        LogTestConfusionCallback(
            classes,
            top_k_false_predicted_labels=args.top_false_labels,
            examples_per_label=args.examples_per_label,
            max_confusion_labels=args.max_confusion_labels,
        )
    ]
    if args.log_predictions:
        callbacks.append(LogPredictCallback(classes))

    logger = TensorBoardLogger(save_dir=args.out_dir, name=args.run_name, default_hp_metric=False)  # type: ignore[arg-type]
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        precision=args.amp_precision,  # type: ignore[arg-type]
    )
    trainer.test(model, datamodule=dm)

    cappa.Output().output(f"Evaluation logs written to: {logger.log_dir}")
