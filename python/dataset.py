from os import process_cpu_count
from typing import Literal

import torch
from datasets import Array2D, Dataset, DatasetDict, fingerprint, load_dataset
from lightning import LightningDataModule
from proc_data import DATASET_REPO, rasterize_strokes
from torch.utils.data import DataLoader
from torchvision.transforms import v2


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

        base_transforms = [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
        ]

        augmentations = [
            v2.RandomRotation(10),  # type: ignore[arg-type]
            v2.RandomAffine(
                degrees=0,  # type: ignore[arg-type]
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

        def process_dataset(
            ds_split: Dataset, transform_type: Literal["eval", "train"]
        ) -> DatasetDict:
            def _rasterize_strokes_batched(batch, image_size):
                batch["image"] = [
                    rasterize_strokes(strokes, image_size)
                    for strokes in batch["strokes"]
                ]
                return batch

            return (
                ds_split.map(
                    _rasterize_strokes_batched,
                    batched=True,
                    remove_columns="strokes",
                    num_proc=self.num_workers,
                    new_fingerprint=fingerprint.Hasher.hash(self.image_size),
                    fn_kwargs={"image_size": self.image_size},
                )
                .cast_column(
                    "image",
                    Array2D(shape=(self.image_size, self.image_size), dtype="uint8"),
                )
                .with_format("torch")
                .with_transform(
                    train_transform if transform_type == "train" else eval_transform
                )
            )

        if stage == "fit":
            self.train_dataset = process_dataset(dataset["train"], "train")
            self.val_dataset = process_dataset(dataset["val"], "eval")

        if stage == "test" or stage is None:
            self.test_dataset = process_dataset(dataset["test"], "eval")

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
