from os import process_cpu_count
from pathlib import Path
from typing import Literal

import msgspec
import polars as pl
import torch
from lightning import LightningDataModule
from PIL import Image, ImageDraw
from proc_data import IMG_SIZE, DataSetInfo, Strokes
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose

DATA_ROOT = Path("build/data")


def get_dataset_info(dataset_name: str) -> DataSetInfo:
    """Load dataset metadata from the info JSON file."""
    dataset_path = DATA_ROOT / dataset_name
    info_path = dataset_path / "dataset_info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"Could not find dataset info at {info_path}")

    with open(info_path, "rb") as f:
        return msgspec.json.decode(f.read(), type=DataSetInfo)


def normalize(strokes: Strokes, target_size: int) -> Strokes:
    xs = [x for s in strokes for x, _ in s]
    min_x, max_x = min(xs), max(xs)
    ys = [y for s in strokes for _, y in s]
    min_y, max_y = min(ys), max(ys)

    width = max(max_x - min_x, max_y - min_y)
    if width == 0:
        return []
    width = width * 1.2 + 20  # leave margin to avoid edge cases
    zero_x = (max_x + min_x - width) / 2
    zero_y = (max_y + min_y - width) / 2
    scale = target_size / width

    return [
        [((x - zero_x) * scale, (y - zero_y) * scale) for x, y in s] for s in strokes
    ]


def draw_to_img(strokes: Strokes, size: int, resize: bool = True) -> Image.Image:
    if resize:
        strokes = normalize(strokes, size)
    image = Image.new("1", (size, size), "black")
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        draw.line(stroke, fill="white")
    return image


class SymbolDataset(VisionDataset):
    def __init__(
        self,
        dataset_name: str,
        transforms: Compose | None,
        split: Literal["train", "test", "val"],
    ):
        self.dataset_path = DATA_ROOT / dataset_name
        self.split_dir = self.dataset_path / split
        self.transforms = transforms

        # 1. Load Metadata
        info_path = self.dataset_path / "dataset_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Could not find dataset info at {info_path}")

        self.info = get_dataset_info(dataset_name)
        self.classes = sorted(self.info.class_count.keys())

        # Create mapping dataframe for joining (more efficient than dict replace for large data)
        # We make a small LazyFrame for the class mapping
        class_map_df = pl.DataFrame(
            {"label": self.classes, "label_idx": range(len(self.classes))}
        ).lazy()

        # 2. Setup Lazy Loading Pattern
        parquet_glob = str(self.split_dir / "*.parquet")

        # Check existence cheaply before scanning
        if not list(self.split_dir.glob("*.parquet")):
            raise FileNotFoundError(f"No .parquet files found in {self.split_dir}")

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

        except Exception as e:
            raise RuntimeError(
                f"Failed to load parquet shards from {parquet_glob}"
            ) from e
        # These are zero-copy references to the table created above
        self.samples = arrow_table["data"]
        self.targets = arrow_table["label_idx"]

    def __len__(self):
        # Return length of the Arrow array
        return len(self.samples)

    def __getitem__(self, idx):
        # Access data by index
        # .as_py() deserializes the C++ binary format to Python objects on-the-fly
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
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self.collate_fn,
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
