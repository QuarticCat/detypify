from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    build_dir: Path = Path("build")

    @property
    def mathwriting_raw_archive(self) -> Path:
        return self.build_dir / "raw" / "mathwriting" / "mathwriting-2024.tgz"

    @property
    def detexify_raw_dir(self) -> Path:
        return self.build_dir / "raw" / "detexify"

    @property
    def generated_dir(self) -> Path:
        return self.build_dir / "generated"

    @property
    def infer_json(self) -> Path:
        return self.generated_dir / "infer.json"

    @property
    def contrib_json(self) -> Path:
        return self.generated_dir / "contrib.json"

    @property
    def unmapped_latex_symbols_json(self) -> Path:
        return self.generated_dir / "unmapped_latex_symbols.json"

    @property
    def datasets_cache_dir(self) -> Path:
        return self.build_dir / "datasets" / "cache"

    @property
    def raw_dataset_parquet(self) -> Path:
        return self.build_dir / "datasets" / "raw" / "data.parquet"

    @property
    def train_dir(self) -> Path:
        return self.build_dir / "train"


DEFAULT_DATA_PATHS = DataPaths()
