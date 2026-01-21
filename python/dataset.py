from os import process_cpu_count
from pathlib import Path
from typing import Literal

import polars as pl
import torch
from lightning import LightningDataModule
from proc_data import DATASET_ROOT, IMG_SIZE, draw_to_img, get_dataset_info
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose


class SymbolDataset(VisionDataset):
    def __init__(
        self,
        dataset_name: str,
        transforms: Compose | None,
        split: Literal["train", "test", "val"],
    ):
        self.dataset_path = DATASET_ROOT / dataset_name
        split_dir = self.dataset_path / split
        self.transforms = transforms

        # 1. Load Metadata
        info_path = self.dataset_path / "dataset_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Could not find dataset info at {info_path}")

        self.info = get_dataset_info(dataset_name)
        self.classes = sorted(self.info.class_count.keys())

        # Create mapping dataframe for joining
        # more efficient than dict replace for large data
        class_map_df = pl.DataFrame(
            {"label": self.classes, "label_idx": range(len(self.classes))}
        ).lazy()

        # 2. Setup Lazy Loading Pattern
        parquet_glob = str(split_dir / "*.parquet")

        # Check existence cheaply before scanning
        if not list(split_dir.glob("*.parquet")):
            raise FileNotFoundError(f"No .parquet files found in {split_dir}")

        try:
            q = pl.scan_parquet(parquet_glob)

            # query plan:
            # 1. Join with class map to get integers
            # 2. Select only 'data' and 'label_idx'
            arrow_table = (
                q.join(class_map_df, on="label", how="left")
                .select(["data", "label_idx"])
                .collect()
                .to_arrow()
            )
            del class_map_df

        except Exception as e:
            raise RuntimeError(
                f"Failed to load parquet shards from {parquet_glob}"
            ) from e
        self.samples = arrow_table["data"]
        self.targets = arrow_table["label_idx"]

        del arrow_table

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        strokes = self.samples[idx].as_py()
        label_idx = self.targets[idx].as_py()

        # Draw image
        image = draw_to_img(strokes, IMG_SIZE)

        if self.transforms:
            image = self.transforms(image)

        return image, label_idx


class MathSymbolDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: Path,
        dataset_name: str,
        batch_size: int = 64,
        num_workers: int = process_cpu_count(),
    ):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        # --- 2. Build Transforms ---
        # Common steps for all splits
        base_transforms = [
            v2.Grayscale(num_output_channels=1),
            v2.ToImage(),
            v2.ToDtype(torch.float16, scale=True),
        ]

        # Augmentation (Train only)
        augmentations = [
            v2.RandomRotation(15),  # type: ignore
            v2.RandomAffine(
                degrees=0,  # type: ignore
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                shear=10,
            ),
        ]

        # Compose them
        self.train_transform = v2.Compose(base_transforms + augmentations)

        # Eval: Same base + normalize, BUT NO ROTATION/SHIFT
        self.eval_transform = v2.Compose(base_transforms)

        self.train_dataset: SymbolDataset
        self.val_dataset: SymbolDataset
        self.test_dataset: SymbolDataset

    def setup(self, stage: str | None = None):
        num_classes = len(get_dataset_info(self.dataset_name).class_count.keys())

        # data agumentaion with MixUp and CutMix
        mixup = v2.MixUp(num_classes=num_classes, alpha=0.8)
        cutmix = v2.CutMix(num_classes=num_classes, alpha=1.0)

        # 3. Create the Switch (Paper uses 0.5 switch probability )
        # This randomly picks either MixUp or CutMix for a given batch.
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        self.collate_fn = lambda batch: cutmix_or_mixup(*default_collate(batch))
        if stage == "fit":
            self.train_dataset = SymbolDataset(
                dataset_name=self.dataset_name,
                split="train",
                transforms=self.train_transform,
            )
            self.val_dataset = SymbolDataset(
                dataset_name=self.dataset_name,
                split="val",
                transforms=self.eval_transform,  # Note: No augmentation
            )

        if stage == "test" or stage is None:
            self.test_dataset = SymbolDataset(
                dataset_name=self.dataset_name,
                split="test",
                transforms=self.eval_transform,  # Note: No augmentation
            )
            self.num_classes = len(self.test_dataset.classes)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
