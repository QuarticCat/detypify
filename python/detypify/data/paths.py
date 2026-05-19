from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    build_dir: Path = Path("build")
    external_dir: Path = Path("external")

    @property
    def raw_dir(self) -> Path:
        return self.build_dir / "raw"

    @property
    def mathwriting_raw_dir(self) -> Path:
        return self.raw_dir / "mathwriting"

    @property
    def detexify_raw_dir(self) -> Path:
        return self.raw_dir / "detexify"

    @property
    def contrib_raw_json(self) -> Path:
        return self.raw_dir / "contrib" / "dataset.json"

    @property
    def contrib_accepted_json(self) -> Path:
        return self.raw_dir / "contrib" / "accepted.json"

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
    def dataset_artifacts_dir(self) -> Path:
        return self.build_dir / "datasets"

    @property
    def datasets_cache_dir(self) -> Path:
        return self.dataset_artifacts_dir / "cache"

    @property
    def raw_dataset_parquet(self) -> Path:
        return self.dataset_artifacts_dir / "raw" / "data.parquet"

    @property
    def review_dir(self) -> Path:
        return self.build_dir / "review"

    @property
    def contrib_review_dir(self) -> Path:
        return self.review_dir / "contrib"

    @property
    def train_dir(self) -> Path:
        return self.build_dir / "train"

    @property
    def math_font(self) -> Path:
        return self.external_dir / "fonts" / "NewCMMath-Regular.otf"


DEFAULT_DATA_PATHS = DataPaths()
