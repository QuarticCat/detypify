from typing import Any, Literal

import lightning as L  # noqa
import torch
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
        learning_rate: float = 1e-3,
        warmup_rounds: int = 20,
        total_rounds: int = 200,
        export: bool = False,
        batch_size: int = 48,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Any = create_model(
            model_name,
            num_classes=num_classes,
            in_chans=1,
            drop_path_rate=0.1,
            drop_rate=0.0,
        )
        self.criterion = nn.CrossEntropyLoss()

        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.val_acc_top3 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=3
        )
        self.warm_up_epochs = warmup_rounds
        self.total_epochs = total_rounds
        self.export = export
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def configure_model(self):
        self.model.to(memory_format=torch.channels_last)
        self.model_opt = torch.compile(
            self.model,
            options={"triton.cudagraphs": True, "shape_padding": True},
            dynamic=False,
        )  # type: ignore

    def forward(self, x):
        if self.export:
            return self.model_opt(x)
        return self.model(x)

    def training_step(self, batch):
        x, y = batch["image"], batch["label"]
        x = x.to(memory_format=torch.channels_last)
        pred = self.forward(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch["image"], batch["label"]
        x = x.to(memory_format=torch.channels_last)
        pred = self.forward(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc(pred, y), prog_bar=True)
        self.log("val_top3", self.val_acc_top3(pred, y))
        return loss

    def test_step(self, batch):
        x, y = batch["image"], batch["label"]
        x = x.to(memory_format=torch.channels_last)
        pred = self.forward(x)
        self.log("val_acc", self.val_acc(pred, y), prog_bar=True)
        self.log("val_top3", self.val_acc_top3(pred, y))

    def configure_optimizers(self):
        """
        AdamW optimization with strictly enforced parameter grouping.
        Paper specifies Weight Decay = 0.1 for weights, 0.0 for bias/norm.
        """
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Check for bias, norm, or batchnorm layers to exclude from decay
            if (
                param.ndim <= 1
                or name.endswith(".bias")
                or "norm" in name
                or "bn" in name
            ):
                no_decay.append(param)
            else:
                decay.append(param)

        optim_groups = [
            {"params": decay, "weight_decay": 0.1},  # High decay for weights
            {"params": no_decay, "weight_decay": 0.0},  # No decay for biases/norms
        ]

        # AdamW with eps=1e-7 is crucial for Hybrid models
        optimizer = optim.AdamW(
            optim_groups, lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-7
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=self.warm_up_epochs,
        )

        decay_scheduler = CosineAnnealingLR(
            optimizer, T_max=(self.total_epochs - self.warm_up_epochs), eta_min=1e-6
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
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }


class CNNModel(L.LightningModule):
    def __init__(self, num_classes: int, export: bool = False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.val_acc_top5 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )
        self.export = export

    def configure_model(self):
        self.features_opt = torch.compile(
            self.features,
            options={"triton.cudagraphs": True, "shape_padding": True},
            dynamic=False,
        )  # type: ignore
        self.classifier_opt = torch.compile(
            self.classifier,
            options={"triton.cudagraphs": True, "shape_padding": True},
            dynamic=False,
        )  # type: ignore

    def forward(self, x):
        if self.export:
            self.features(x)
            x = self.avgpool(x)
            return self.classifier(x)
        x = self.features_opt(x)
        x = self.avgpool(x)
        return self.classifier_opt(x)

    def training_step(self, batch):
        x, y = batch["image"], batch["label"]
        pred = self.forward(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch["image"], batch["label"]
        pred = self.forward(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(pred, y), prog_bar=True)
        self.log("val_top5", self.val_acc_top5(pred, y))
        return loss

    def test_step(self, batch):
        x, y = batch["image"], batch["label"]
        pred = self.forward(x)
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
