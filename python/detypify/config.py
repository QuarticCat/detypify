from enum import StrEnum

HF_DATASET_REPO = "Cloud0310/detypify-datasets"


class DataSetName(StrEnum):
    mathwriting = "mathwriting"
    detexify = "detexify"
    contrib = "contrib"


class ModelName(StrEnum):
    conv_small_035 = "mobilenetv4_conv_small_035"
    conv_small_050 = "mobilenetv4_conv_small_050"
    conv_small_full = "mobilenetv4_conv_small"
    conv_medium = "mobilenetv4_conv_medium"
    hybrid_medium_075 = "mobilenetv4_hybrid_medium_075"
    hybrid_medium = "mobilenetv4_hybrid_medium"
