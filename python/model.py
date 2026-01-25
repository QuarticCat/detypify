from typing import Any, Literal

import lightning as L  # noqa
from timm import create_model
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import Accuracy

type model_size = Literal["small", "medium", "large"]
# result of timm.list_models("*mobilenetv4*")
type model_names = Literal[
    "mobilenetv4_conv_small",
    "mobilenetv4_conv_small_035",
    "mobilenetv4_conv_small_050",
    "mobilenetv4_hybrid_medium",
    "mobilenetv4_hybrid_medium_075",
    "efficientnet_b1",
]


class TimmModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_name: model_names = "mobilenetv4_hybrid_medium",
        warmup_rounds: int = 20,
        total_rounds: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Any = create_model(
            model_name,
            num_classes=num_classes,
            in_chans=1,
        )
        self.criterion = nn.CrossEntropyLoss()

        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.val_acc_top5 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )
        self.warm_up_epochs = warmup_rounds
        self.total_epochs = total_rounds

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch["image"], batch["label"]
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch["image"], batch["label"]
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(pred, y), prog_bar=True)
        self.log("val_top5", self.val_acc_top5(pred, y))

        return loss

    def test_step(self, batch):
        x, y = batch["image"], batch["label"]
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(pred, y), prog_bar=True)
        self.log("val_top5", self.val_acc_top5(pred, y))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
        # see: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        warmup_scheduler = LinearLR(
            optimizer,
            total_iters=self.warm_up_epochs,
        )
        decay_scheduler = CosineAnnealingLR(
            optimizer, T_max=(self.total_epochs - self.warm_up_epochs), eta_min=0.0
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[self.warm_up_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class CNNModel(L.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model: Any = nn.Sequential(
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
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.val_acc_top5 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch["image"], batch["label"]
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch["image"], batch["label"]
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(pred, y), prog_bar=True)
        self.log("val_top5", self.val_acc_top5(pred, y))
        return loss

    def test_step(self, batch):
        x, y = batch["image"], batch["label"]
        pred = self.model(x)
        self.log("val_acc", self.val_acc(pred, y), prog_bar=True)
        self.log("val_top5", self.val_acc_top5(pred, y))

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
