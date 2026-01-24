from os import process_cpu_count
from pathlib import Path

from datasets import load_dataset
from lightning import LightningDataModule
from proc_data import DATASET_REPO, IMG_SIZE, DataSetName, rasterize_strokes
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import v2


class MathSymbolDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: Path,
        dataset_name: DataSetName,
        batch_size: int = 64,
        num_workers: int = process_cpu_count(),
        image_size: int = IMG_SIZE,
    ):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # --- 2. Build Transforms ---
        # Common steps for all splits
        base_transforms = [
            v2.ToImage(),
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

    def prepare_data(self):
        load_dataset(DATASET_REPO, name=self.dataset_name)

    def setup(self, stage: str | None = None):
        # We need classes for MixUp/CutMix and for SymbolDataset
        # Load the dataset (cached) to get info
        dataset = load_dataset(DATASET_REPO, name=self.dataset_name)
        num_classes = len(dataset.unique("label"))

        # Data agumentaion with MixUp and CutMix
        mixup = v2.MixUp(num_classes=num_classes, alpha=0.8)
        cutmix = v2.CutMix(num_classes=num_classes, alpha=1.0)

        # 3. Create the Switch (Paper uses 0.5 switch probability )
        # This randomly picks either MixUp or CutMix for a given batch.
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        self.collate_fn = lambda batch: cutmix_or_mixup(*default_collate(batch))
        if stage == "fit":
            self.train_dataset = dataset.map(
                lambda strokes: self.train_transform(
                    rasterize_strokes(strokes, self.image_size)
                )
            ).with_format("torch")
            self.val_dataset = dataset.map(
                lambda strokes: self.eval_transform(
                    rasterize_strokes(strokes, self.image_size)
                )
            ).with_format("torch")

        if stage == "test" or stage is None:
            self.test_dataset = dataset.map(
                lambda strokes: self.eval_transform(
                    rasterize_strokes(strokes, self.image_size)
                )
            ).with_format("torch")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collate_fn,
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
