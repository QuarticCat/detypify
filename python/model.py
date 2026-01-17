from typing import Literal

import lightning as L
from timm import create_model
from torch import nn, optim
from torchmetrics import Accuracy

type model_size = Literal["small", "medium", "large"]


class MobileNetV4(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        use_transformer=False,
        fine_tune=False,
        model_size: model_size = "small",
    ):
        super().__init__()
        is_hybrid = "hybrid" if use_transformer else "conv"
        model_name = f"mobilenetv4_{is_hybrid}_{model_size}"
        if fine_tune:
            self.model = create_model(
                "mobilenetv4_conv_medium.e500_r256_in1k",
                pretrained=True,
                num_classes=num_classes,
            )
        else:
            self.model = create_model(
                model_name,
                num_classes=num_classes,
                in_chans=1,
            )
        self.criterion = nn.CrossEntropyLoss()

        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.val_acc_top5 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(pred, y), prog_bar=True)
        self.log("val_top5", self.val_acc_top5(pred, y))

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
        # see: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class TypstSymbolClassifier(L.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        acc = (pred.argmax(1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
