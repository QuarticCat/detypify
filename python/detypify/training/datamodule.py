from typing import cast, override

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from detypify.config import DataSetName
from detypify.data.datasets import RenderedDataset, get_rendered_dataset_splits, load_raw_dataset
from detypify.data.paths import DEFAULT_DATA_PATHS, DataPaths


class MathSymbolDataModule(LightningDataModule):
    """Build deterministic symbol splits and apply stage-specific image transforms."""

    train_dataset: RenderedDataset
    val_dataset: RenderedDataset
    test_dataset: RenderedDataset

    def __init__(
        self,
        image_size: int,
        batch_size: int,
        num_workers: int,
        dataset_names: tuple[DataSetName, ...],
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

        # Keep cached samples as uint8 and move augmentation to batched tensors after device transfer.
        self.eval_transform = v2.Compose([v2.ToImage(), v2.ToDtype(dtype=t_float32, scale=True)])
        self.train_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(dtype=t_float32, scale=True),
                v2.RandomAffine(
                    degrees=10,  # type: ignore[arg-type]
                    translate=(0.1, 0.1),
                    shear=10,
                ),
            ]
        )

    @override
    def prepare_data(self):
        load_raw_dataset(self.dataset_names, self.paths)

    @override
    def setup(self, stage: str | None = None):
        dataset, _ = get_rendered_dataset_splits(
            self.dataset_names,
            self.image_size,
            paths=self.paths,
        )

        if stage == "fit" or stage is None:
            self.train_dataset = dataset["train"]
            self.val_dataset = dataset["val"]

        if stage == "test" or stage is None:
            self.test_dataset = dataset["test"]

    @override
    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    @override
    def val_dataloader(self):
        return self._dataloader(self.val_dataset, shuffle=False)

    @override
    def test_dataloader(self):
        return self._dataloader(self.test_dataset, shuffle=False)

    def _dataloader(self, dataset: RenderedDataset, *, shuffle: bool):
        """Create a loader with the worker and memory settings shared by every split."""
        return DataLoader(
            cast("Dataset", dataset),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    @override
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Convert uint8 images to model inputs, augmenting training batches only."""
        if isinstance(batch, dict) and self.trainer:
            from lightning.pytorch.trainer.states import RunningStage

            original_images = batch["image"].unsqueeze(1)
            match self.trainer.state.stage:
                case RunningStage.TRAINING:
                    batch["image"] = self.train_transform(original_images)
                case _:
                    batch["image"] = self.eval_transform(original_images)

        return batch
