from functools import partial
from os import process_cpu_count

import torch
from datasets import Array2D, Value, load_dataset
from lightning import LightningDataModule
from proc_data import DATASET_REPO, rasterize_strokes
from torch.utils.data import DataLoader
from torchvision.transforms import v2


def preprocess_images(batch, image_size):
    """Preprocess function for datasets.map - defined at module level for caching."""
    batch["image"] = [
        rasterize_strokes(strokes, image_size) for strokes in batch["strokes"]
    ]
    return batch


class MathSymbolDataModule(LightningDataModule):
    def __init__(
        self,
        image_size: int,
        batch_size: int = 64,
        num_workers: int = min(process_cpu_count(), 12),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.preprocess = partial(preprocess_images, image_size=self.image_size)

        base_transforms = [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
        ]

        augmentations = [
            v2.RandomRotation(10),  # type: ignore
            v2.RandomAffine(
                degrees=0,  # type: ignore
                translate=(0.1, 0.1),
                shear=10,
            ),
        ]

        # Compose them
        self.train_transform = v2.Compose(base_transforms + augmentations)

        # Eval: Same base + normalize, BUT NO ROTATION/SHIFT
        self.eval_transform = v2.Compose(base_transforms)

    def prepare_data(self):
        load_dataset(DATASET_REPO)

    def setup(self, stage: str | None = None):
        dataset = load_dataset(DATASET_REPO)

        def train_transform(batch):
            images = torch.tensor(batch["image"], dtype=torch.uint8)
            images = self.train_transform(images.unsqueeze(1))
            batch["image"] = images
            return batch

        def eval_transform(batch):
            images = torch.tensor(batch["image"], dtype=torch.uint8)
            images = self.eval_transform(images.unsqueeze(1))
            batch["image"] = images
            return batch

        if stage == "fit":
            self.train_dataset = (
                dataset["train"]
                .map(self.preprocess, batched=True, remove_columns="strokes")
                .cast_column(
                    "image",
                    Array2D(shape=(self.image_size, self.image_size), dtype="uint8"),
                )
                .with_format("torch")
                .with_transform(train_transform)
            )
            self.val_dataset = (
                dataset["val"]
                .map(self.preprocess, batched=True, remove_columns="strokes")
                .cast_column(
                    "image",
                    Array2D(shape=(self.image_size, self.image_size), dtype="uint8"),
                )
                .with_format("torch")
                .with_transform(eval_transform)
            )

        if stage == "test" or stage is None:
            self.test_dataset = (
                dataset["test"]
                .map(self.preprocess, batched=True, remove_columns="strokes")
                .cast_column("label", Value("uint32"))
                .cast_column(
                    "image",
                    Array2D(shape=(self.image_size, self.image_size), dtype="uint8"),
                )
                .with_format("torch")
                .with_transform(eval_transform)
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
