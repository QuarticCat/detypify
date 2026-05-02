import re
from dataclasses import dataclass
from enum import StrEnum

HF_DATASET_REPO = "Cloud0310/detypify-datasets"
MOBILENET_MODEL_NAME_RE = re.compile(r"^mobilenet_(v4|v5)_(\d{3})$")


class DataSetName(StrEnum):
    mathwriting = "mathwriting"
    detexify = "detexify"
    contrib = "contrib"


class ModelFamily(StrEnum):
    v4 = "v4"
    v5 = "v5"


@dataclass(frozen=True)
class MobileNetModelSpec:
    family: ModelFamily
    size: float


def parse_mobilenet_model_name(model_name: str) -> MobileNetModelSpec:
    match = MOBILENET_MODEL_NAME_RE.fullmatch(model_name)
    if match is None:
        msg = "Model name must match 'mobilenet_{v4|v5}_{size}', for example 'mobilenet_v4_035'."
        raise ValueError(msg)

    family, size_label = match.groups()
    return MobileNetModelSpec(
        family=ModelFamily(family),
        size=int(size_label) / 100,
    )
