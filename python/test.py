"""Evaluate an existing checkpoint and log test diagnostics."""

from os import process_cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from detypify.config import DataSetName
from detypify.data.datasets import get_dataset_classes
from detypify.data.paths import DEFAULT_DATA_PATHS
from detypify.training.callbacks import LogPredictCallback, LogTestConfusionCallback
from detypify.training.datamodule import MathSymbolDataModule
from detypify.training.model import MobileNetModel
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.cuda import is_bf16_supported

if TYPE_CHECKING:
    from lightning.pytorch import Callback

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    *,
    ckpt_path: Annotated[Path, typer.Option("--ckpt-path", "-c", help="Checkpoint path to evaluate.")],
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="TensorBoard output directory.")
    ] = DEFAULT_DATA_PATHS.train_dir / "eval",
    run_name: Annotated[str, typer.Option("--run-name", help="TensorBoard run name.")] = "existing_model",
    datasets: Annotated[
        list[DataSetName] | None,
        typer.Option("--datasets", "-d", help="Datasets to use for the test split."),
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Evaluation batch size.")] = 128,
    image_size: Annotated[
        int | None,
        typer.Option("--image-size", help="Override image size. Defaults to the checkpoint hparams image_size."),
    ] = None,
    num_workers: Annotated[
        int | None, typer.Option("--num-workers", help="DataLoader and dataset mapping workers.")
    ] = None,
    max_samples: Annotated[
        int | None, typer.Option("--max-samples", help="Limit samples for a quick test run.")
    ] = None,
    amp_precision: Annotated[
        str, typer.Option("--amp-precision", help="Precision: 64, 32, 16-mixed, bf16-mixed.")
    ] = "32-true",
    log_predictions: Annotated[
        bool,
        typer.Option("--log-predictions/--no-log-predictions", help="Log prediction image grids."),
    ] = True,
    top_false_labels: Annotated[
        int,
        typer.Option("--top-false-labels", help="Number of most frequent false predicted labels to log."),
    ] = 10,
    examples_per_label: Annotated[
        int,
        typer.Option("--examples-per-label", help="Image examples to log for each top false predicted label."),
    ] = 4,
    max_confusion_labels: Annotated[
        int,
        typer.Option("--max-confusion-labels", help="Maximum labels shown in the confusion-matrix figure."),
    ] = 40,
):
    """Run only the test phase for an existing Lightning checkpoint."""
    if not ckpt_path.exists():
        msg = f"Checkpoint does not exist: {ckpt_path}"
        raise typer.BadParameter(msg, param_hint="--ckpt-path")

    if amp_precision == "bf16-mixed" and not is_bf16_supported():
        amp_precision = "16-mixed"

    model = MobileNetModel.load_from_checkpoint(ckpt_path)
    eval_image_size = image_size or int(model.hparams["image_size"])
    dataset_names = tuple(dict.fromkeys(datasets or [DataSetName.detexify, DataSetName.mathwriting]))
    data_num_workers = num_workers or process_cpu_count() or 1
    classes = sorted(get_dataset_classes(dataset_names, max_samples=max_samples, num_proc=data_num_workers))

    dm = MathSymbolDataModule(
        batch_size=batch_size,
        image_size=eval_image_size,
        dataset_names=dataset_names,
        max_samples=max_samples,
        num_workers=data_num_workers,
    )

    callbacks: list[Callback] = [
        LogTestConfusionCallback(
            classes,
            top_k_false_predicted_labels=top_false_labels,
            examples_per_label=examples_per_label,
            max_confusion_labels=max_confusion_labels,
        )
    ]
    if log_predictions:
        callbacks.append(LogPredictCallback(classes))

    logger = TensorBoardLogger(save_dir=out_dir, name=run_name, default_hp_metric=False)  # type: ignore[arg-type]
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        precision=amp_precision,  # type: ignore[arg-type]
    )
    trainer.test(model, datamodule=dm)

    typer.echo(f"Evaluation logs written to: {logger.log_dir}")


if __name__ == "__main__":
    app()
