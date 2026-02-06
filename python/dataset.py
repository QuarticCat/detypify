from os import process_cpu_count
from typing import override

from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader


class MathSymbolDataModule(LightningDataModule):
    def __init__(
        self,
        image_size: int,
        batch_size: int = 64,
        num_workers: int = process_cpu_count() or 1,
    ):
        from proc_data import DATASET_REPO
        from torch import float32 as t_float32
        from torchvision.transforms import v2

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.dataset_repo = DATASET_REPO

        self.eval_transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(dtype=t_float32, scale=True)]
        )
        self.train_transform = v2.Compose(
            [
                v2.ToImage(),
                # augmentations
                v2.RandomAffine(
                    degrees=10,  # type: ignore[arg-type]
                    translate=(0.1, 0.1),
                    shear=10,
                ),
                v2.ToDtype(dtype=t_float32, scale=True),
            ]
        )

    @override
    def prepare_data(self):
        load_dataset(self.dataset_repo)

    @override
    def setup(self, stage: str | None = None):
        from datasets import Array2D
        from proc_data import rasterize_strokes

        def _rasterize_strokes_batched(batch, image_size):
            batch["image"] = [
                rasterize_strokes(strokes, image_size) for strokes in batch["strokes"]
            ]
            return batch

        dataset = (
            load_dataset(self.dataset_repo)
            .map(
                _rasterize_strokes_batched,
                batched=True,
                remove_columns="strokes",
                num_proc=self.num_workers,
                fn_kwargs={"image_size": self.image_size},
            )
            .cast_column(
                "image",
                Array2D(shape=(self.image_size, self.image_size), dtype="uint8"),
            )
            .with_format("torch")
        )

        if stage == "fit":
            self.train_dataset = dataset["train"]
            self.val_dataset = dataset["val"]

        if stage == "test" or stage is None:
            self.test_dataset = dataset["test"]

    @override
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    @override
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    @override
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    @override
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # when batch is not a dict, means its not from dataloader, do nothing.
        if isinstance(batch, dict) and self.trainer:
            from lightning.pytorch.trainer.states import RunningStage
            from torch import uint8 as t_uint8

            original_images = batch["image"].to(dtype=t_uint8).unsqueeze(1)
            match self.trainer.state.stage:
                case RunningStage.TRAINING:
                    batch["image"] = self.train_transform(original_images)
                case _:
                    batch["image"] = self.eval_transform(original_images)

        return batch
