from typing import override

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from detypify.config import DataSetName
from detypify.data.datasets import get_rendered_dataset_splits, load_raw_dataset
from detypify.data.paths import DEFAULT_DATA_PATHS, DataPaths


class MathSymbolDataModule(LightningDataModule):
    """Build deterministic symbol splits and apply stage-specific image transforms."""

    def __init__(
        self,
        image_size: int,
        batch_size: int,
        num_workers: int,
        dataset_names: tuple[DataSetName, ...],
        max_samples: int | None,
        paths: DataPaths = DEFAULT_DATA_PATHS,
    ):
        from torch import float32 as t_float32
        from torchvision.transforms import v2

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.dataset_names = dataset_names
        self.paths = paths
        self.max_samples = max_samples
        self.classes: list[str] = []

        # Keep cached samples as uint8 and move augmentation to batched tensors after device transfer.
        self.eval_transform = v2.Compose([v2.ToImage(), v2.ToDtype(dtype=t_float32, scale=True)])
        self.train_transform = v2.Compose(
            [
                v2.ToImage(),
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
        load_raw_dataset(self.dataset_names, self.paths)

    @override
    def setup(self, stage: str | None = None):
        dataset, self.classes = get_rendered_dataset_splits(
            self.dataset_names,
            self.image_size,
            paths=self.paths,
            max_samples=self.max_samples,
        )

        if stage == "fit" or stage is None:
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
        """Convert uint8 images to model inputs, augmenting training batches only."""
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
