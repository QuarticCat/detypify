from os import process_cpu_count
from pathlib import Path
from typing import override

from detypify.config import DataSetName, TrainingDataMode
from detypify.data.datasets import DETERMINISTIC_SPLIT_SEED, get_rendered_dataset_splits, load_raw_dataset
from detypify.data.paths import DEFAULT_DATA_PATHS, DataPaths
from detypify.data.synthetic import (
    SyntheticSettings,
    get_synthetic_dataset_splits,
    prepare_synthetic_training_dataset,
    validate_synthetic_font,
)
from lightning import LightningDataModule
from torch.utils.data import DataLoader


class MathSymbolDataModule(LightningDataModule):
    def __init__(
        self,
        image_size: int,
        batch_size: int = 64,
        num_workers: int = process_cpu_count() or 1,
        dataset_names: tuple[DataSetName, ...] = (DataSetName.detexify, DataSetName.mathwriting),
        paths: DataPaths = DEFAULT_DATA_PATHS,
        max_samples: int | None = None,
        training_data: TrainingDataMode = TrainingDataMode.real,
        synthetic_samples_per_class: int = 500,
        synthetic_seed: int = DETERMINISTIC_SPLIT_SEED,
        synthetic_font: Path | None = None,
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
        self.training_data = training_data
        self.synthetic_settings = SyntheticSettings(
            image_size=image_size,
            samples_per_class=synthetic_samples_per_class,
            seed=synthetic_seed,
        )
        self.synthetic_font = synthetic_font or paths.math_font
        self.synthetic_cache_fingerprint: str | None = None
        self.classes: list[str] = []

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
        if self.training_data is TrainingDataMode.synthetic:
            validate_synthetic_font(self.synthetic_font)
        load_raw_dataset(self.dataset_names, self.paths)
        if self.training_data is TrainingDataMode.synthetic:
            _, _, self.synthetic_cache_fingerprint = prepare_synthetic_training_dataset(
                self.dataset_names,
                self.synthetic_settings,
                self.synthetic_font,
                num_proc=self.num_workers,
                paths=self.paths,
                max_samples=self.max_samples,
            )

    @override
    def setup(self, stage: str | None = None):
        if self.training_data is TrainingDataMode.synthetic:
            dataset, self.classes, self.synthetic_cache_fingerprint = get_synthetic_dataset_splits(
                self.dataset_names,
                self.synthetic_settings,
                self.synthetic_font,
                num_proc=self.num_workers,
                paths=self.paths,
                max_samples=self.max_samples,
            )
        else:
            dataset, self.classes = get_rendered_dataset_splits(
                self.dataset_names,
                self.image_size,
                num_proc=self.num_workers,
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
