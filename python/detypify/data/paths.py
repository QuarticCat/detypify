from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    build_dir: Path = Path("build")

    @property
    def mathwriting_dir(self) -> Path:
        return self.build_dir / "data" / "mathwriting"

    @property
    def detexify_dir(self) -> Path:
        return self.build_dir / "data" / "detexify"

    @property
    def converted_dir(self) -> Path:
        return self.build_dir / "data" / "_converted"

    @property
    def data_parquet(self) -> Path:
        return self.converted_dir / "data.parquet"

    @property
    def typst_symbols(self) -> Path:
        return self.converted_dir / "typst_symbols.json"

    @property
    def metadata_dir(self) -> Path:
        return self.build_dir / "data" / "_metadata"

    @property
    def train_dir(self) -> Path:
        return self.build_dir / "train"

    @property
    def dataset_splits_dir(self) -> Path:
        return self.train_dir / "_dataset_splits"


DEFAULT_DATA_PATHS = DataPaths()
