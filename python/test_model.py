"""Script to test a trained model and log wrong guesses."""

from pathlib import Path

import torch
import typer
from callbacks import LogWrongGuessesCallback
from dataset import MathSymbolDataModule
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from model import CNNModel, TimmModel
from proc_data import DATASET_REPO, get_dataset_classes


def main(
    checkpoint: str = typer.Argument(..., help="Path to checkpoint file"),
    model_type: str = typer.Option(
        ..., "--model-type", help="Model type (timm or cnn)"
    ),
    model_name: str = typer.Option(
        None,
        "--model-name",
        help="Model name (required for timm if not in checkpoint hparams)",
    ),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size"),
    out_dir: str = typer.Option("build/test", "--out-dir", help="Output directory"),
):
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")

    # Determine class and load
    if model_type == "cnn":
        model_class: type[CNNModel | TimmModel] = CNNModel
        kwargs = {}
    else:
        model_class = TimmModel
        kwargs = {}
        if model_name:
            kwargs["model_name"] = model_name

    # Try loading
    try:
        model = model_class.load_from_checkpoint(ckpt_path, **kwargs)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        if model_type == "timm" and not model_name:
            print("Tip: Try providing --model-name if it was missing from hparams.")
        raise e

    model.eval()

    # Get image size from hparams or model
    image_size = model.hparams.image_size  # type: ignore
    print(f"Model loaded. Image size: {image_size}")

    # Data
    dm = MathSymbolDataModule(image_size=image_size, batch_size=batch_size)

    # Classes for callback
    classes = get_dataset_classes(DATASET_REPO)
    callback = LogWrongGuessesCallback(sorted(classes))

    # Logger
    logger = TensorBoardLogger(
        save_dir=out_dir_path, name="test_log", default_hp_metric=False
    )  # type: ignore

    trainer = Trainer(
        default_root_dir=out_dir_path,
        logger=logger,
        callbacks=[callback],
        precision="16-mixed" if torch.cuda.is_available() else "32",
    )

    print("Starting testing...")
    trainer.test(model, datamodule=dm)
    print(f"Testing finished. Logs saved to {out_dir_path}")


if __name__ == "__main__":
    typer.run(main)
