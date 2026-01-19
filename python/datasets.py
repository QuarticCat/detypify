from os import cpu_count
from pathlib import Path
from typing import Literal

import polars as pl
import pytorch_lightning as L
import torch
from proc_data import (
    DATASET_PATH,
    draw_to_img,
    get_dataset_info,
    normalize,
)
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose


class ParquetDataset(VisionDataset):
    def __init__(
        self,
        dataset_name: str,
        transform: Compose | None,
        split: Literal["train", "test", "val"] = "train",
    ):
        self.dataset_path = DATASET_PATH / dataset_name
        self.split_dir = self.dataset_path / split

        # 1. Load Metadata
        info_path = self.dataset_path / "dataset_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Could not find dataset info at {info_path}")

        self.info = get_dataset_info(dataset_name)

        self.classes = sorted(self.info.class_count.keys())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 3. Load Data from Shards
        # Use a glob pattern to match all parquet files in the split directory.
        # Polars handles the concatenation of shards automatically and efficiently.
        parquet_glob = str(self.split_dir / "*.parquet")

        try:
            # Check if any files exist before reading to avoid obscure Polars errors
            if not list(self.split_dir.glob("*.parquet")):
                raise FileNotFoundError(f"No .parquet files found in {self.split_dir}")

            df = pl.read_parquet(parquet_glob)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load parquet shards from {parquet_glob}"
            ) from e

        self.samples = df["data"].to_list()

        # Map string labels to integers immediately
        label_strs = df["label"].to_list()
        self.targets = [self.class_to_idx[label] for label in label_strs]

        # Clean up DataFrame to free memory
        del df

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        strokes = self.samples[idx]
        label_idx = self.targets[idx]

        # 2. Render Image
        image = draw_to_img(normalize(strokes))

        if self.transforms:
            image = self.transforms(image)

        return image, label_idx


class MathSymbolDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: Path,
        dataset_name: str,
        batch_size: int = 64,
        num_workers: int = cpu_count() if cpu_count() else 16,  # type: ignore
        fine_tuning: bool = True,  # Controls 1 vs 3 channels
    ):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fine_tuning = fine_tuning

        # --- 1. Define Base Normalization ---
        # ImageNet stats require 3 channels. If 1 channel, use standard 0.5.
        if self.fine_tuning:
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
            channels = 3
        else:
            norm_mean = [0.5]
            norm_std = [0.5]
            channels = 1

        # --- 2. Build Transforms ---
        # Common steps for all splits
        base_transforms = [
            v2.Grayscale(num_output_channels=channels),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]

        # Augmentation (Train only)
        augmentations = [
            v2.RandomRotation(15),  # type: ignore
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # type: ignore
        ]

        # Normalization (Last step)
        normalization = [
            v2.Normalize(mean=norm_mean, std=norm_std),
        ]

        # Compose them
        self.train_transform = v2.Compose(
            base_transforms + augmentations + normalization
        )

        # Eval: Same base + normalize, BUT NO ROTATION/SHIFT
        self.eval_transform = v2.Compose(base_transforms + normalization)

        self.train_dataset: ParquetDataset
        self.val_dataset: ParquetDataset
        self.test_dataset: ParquetDataset

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ParquetDataset(
                dataset_name=self.dataset_name,
                split="train",
                transform=self.train_transform,
            )
            self.val_dataset = ParquetDataset(
                dataset_name=self.dataset_name,
                split="val",
                transform=self.eval_transform,  # Note: No augmentation
            )

        if stage == "test" or stage is None:
            self.test_dataset = ParquetDataset(
                dataset_name=self.dataset_name,
                split="test",
                transform=self.eval_transform,  # Note: No augmentation
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
