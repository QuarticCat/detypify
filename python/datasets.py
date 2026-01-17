import lightning as L
from torchvision.datasets import VisionDataset


class MathWritingSoleSymbol(VisionDataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class MathWritingExtracted(VisionDataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class HASYv2(VisionDataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


# for dataset split reuse logic
class DataModule(L.LightningDataModule):
    pass
