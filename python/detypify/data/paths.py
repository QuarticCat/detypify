from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    build_dir: Path = Path("build")

    @property
    def raw_mathwriting_dir(self) -> Path:
        return self.build_dir / "raw" / "mathwriting"

    @property
    def raw_detexify_dir(self) -> Path:
        return self.build_dir / "raw" / "detexify"

    @property
    def raw_converted_parquet(self) -> Path:
        return self.build_dir / "raw" / "_converted" / "data.parquet"

    @property
    def raw_metadata_dir(self) -> Path:
        return self.build_dir / "raw" / "_metadata"

    @property
    def train_dir(self) -> Path:
        return self.build_dir / "train"

    @property
    def dataset_splits_dir(self) -> Path:
        return self.train_dir / "_dataset_splits"


DEFAULT_DATA_PATHS = DataPaths()
