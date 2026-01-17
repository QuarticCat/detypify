from pathlib import Path

import lightning as L
import msgspec
import numpy as np
from proc_data import DataSetInfo, MathSymbol, draw_to_img
from rocksdict import (
    AccessType,
    BlockBasedOptions,
    Cache,
    Options,
    Rdict,
    ReadOptions,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import VisionDataset


class RocksDBDataset(VisionDataset):
    def __init__(self, dataset_path: str | Path, transform=None):
        self.dataset_path = Path(dataset_path)
        self.db_path = self.dataset_path / "db"
        self.transform = transform
        self.db: Rdict | None = None
        self.decoder = msgspec.msgpack.Decoder(type=MathSymbol)

        # 1. Load Metadata
        info_path = self.dataset_path / "dataset_info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"Could not find dataset info at {info_path}"
            )

        with open(info_path, "rb") as f:
            self.info = msgspec.json.decode(f.read(), type=DataSetInfo)

        # 2. Setup Class Mappings (CRITICAL FOR PYTORCH)
        # Sort labels to ensure Class 0 is always the same class
        self.classes = sorted(self.info.class_count.keys())
        self.class_to_idx = {
            cls_name: i for i, cls_name in enumerate(self.classes)
        }

        # 3. Reconstruct Keys & Build Targets List
        self.keys = []
        self.targets = []  # This will hold the int label for every sample

        for label_name in self.classes:
            count = self.info.class_count[label_name]
            label_idx = self.class_to_idx[label_name]
            prefix = ord(label_name)

            # Generate Keys
            self.keys.extend([f"{prefix:08d}_{i:09d}" for i in range(count)])

            # Generate Targets (Much faster than parsing keys later)
            # This consumes very little RAM (e.g., 1MB for 1M images)
            self.targets.extend([label_idx] * count)

    def _open_db(self):
        """Lazy initialization to ensure fork-safety with multiprocessing."""
        # Read-Only mode allows lock-free access by multiple workers
        #
        if self.db:
            return
        access_type = AccessType.read_only()

        # Configure Caching and Block options
        opts = Options()
        block_opts = BlockBasedOptions()

        # 512MB Cache: Keeps index/filters in RAM to avoid disk seeks for metadata
        #
        cache = Cache.new_hyper_clock_cache(512 * 1024 * 1024, 0)
        block_opts.set_block_cache(cache)

        # Pinning blocks prevents cache thrashing
        block_opts.set_cache_index_and_filter_blocks(True)
        block_opts.set_pin_l0_filter_and_index_blocks_in_cache(True)
        opts.set_block_based_table_factory(block_opts)

        self.db = Rdict(str(self.db_path), opts, access_type=access_type)

        # Configure Read Options for high throughput
        self.read_opts = ReadOptions()
        # Readahead 4MB: Critical for sequential scanning speed
        #
        self.read_opts.set_readahead_size(4 * 1024 * 1024)

    def __len__(self):
        return len(self.keys)

    def __del__(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    def __getitem__(self, idx):
        if self.db is None:
            self._open_db()

        key = self.keys[idx]

        # Fetch Data
        raw_data: bytes | None = (
            self.db.get(key, read_opt=self.read_opts) if self.db else None
        )
        if raw_data is None:
            raise IndexError(f"Key {key} not found")

        # Decode
        symbol_bytes = self.decoder.decode(raw_data).symbol
        image = draw_to_img(symbol_bytes)

        # Get Label Integer (Fast lookup from RAM)
        label_idx = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        # Return IMAGE and INT (not string)
        return image, label_idx


# for dataset split reuse logic
class RocksDBDataModule(L.LightningDataModule):
    def __init__(self, dataset_path, batch_size=32):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.save_hyperparameters()

    def setup(self, stage=None):
        # 1. Instantiate Dataset
        # This loads the metadata and keys into memory (~seconds)
        full_dataset = RocksDBDataset(self.dataset_path)

        # 2. Stratified Split
        # We access full_dataset.targets (which we built in __init__)
        train_idx, val_idx = train_test_split(
            np.arange(len(full_dataset)),
            test_size=0.2,
            stratify=full_dataset.targets,
            shuffle=True,
        )

        self.train_ds = Subset(full_dataset, train_idx)
        self.val_ds = Subset(full_dataset, val_idx)

        # 3. Calculate Weights

        # Get counts for the classes

        train_targets = [full_dataset.targets[i] for i in train_idx]

        # We can use numpy unique for speed since we have ints
        unique, counts = np.unique(train_targets, return_counts=True)
        class_counts = dict(zip(unique, counts))

        # Weight = 1.0 / count
        weight_per_class = {
            cls: 1.0 / count for cls, count in class_counts.items()
        }

        # Assign weights to samples
        sample_weights = [weight_per_class[t] for t in train_targets]

        self.train_sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=self.train_sampler,  # Handles Imbalance
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,  # Validation is raw
            num_workers=4,
            pin_memory=True,
        )
