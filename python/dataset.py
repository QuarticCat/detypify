from os import process_cpu_count

import torch
from datasets import Array2D, Value, load_dataset
from lightning import LightningDataModule
from proc_data import DATASET_REPO, IMG_SIZE, rasterize_strokes
from torch.utils.data import DataLoader
from torchvision.transforms import v2


class MathSymbolDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = process_cpu_count(),
        image_size: int = IMG_SIZE,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        base_transforms = [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
        ]

        augmentations = [
            v2.RandomRotation(15),  # type: ignore
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

        self.image_size = image_size

    def prepare_data(self):
        load_dataset(DATASET_REPO)

    def setup(self, stage: str | None = None):
        dataset = load_dataset(DATASET_REPO)

        def preprocess(batch):
            batch["image"] = [
                rasterize_strokes(strokes, self.image_size)
                for strokes in batch["strokes"]
            ]
            return batch

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
                .map(preprocess, batched=True, remove_columns="strokes")
                .cast_column(
                    "image",
                    Array2D(shape=(self.image_size, self.image_size), dtype="uint8"),
                )
                .with_format("torch")
                .with_transform(train_transform)
            )
            self.val_dataset = (
                dataset["val"]
                .map(preprocess, batched=True, remove_columns="strokes")
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
                .map(preprocess, batched=True, remove_columns="strokes")
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
