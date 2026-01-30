from typing import Literal

import torch
from lightning import LightningModule
from timm import create_model
from torch import nn, optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)
from torchmetrics import Accuracy

type ModelSize = Literal["small", "medium", "large"]
# Selected models for efficiency, ranked by model size, ascending
type ModelName = Literal[
    "mobilenetv4_conv_small_035",
    "mobilenetv4_conv_small_050",
    "mobilenetv4_conv_small",
    "mobilenetv4_conv_medium",
    "mobilenetv4_hybrid_medium_075",
    "mobilenetv4_hybrid_medium",
]


class TimmModel(LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_name: ModelName,
        total_epochs: int,
        image_size: int,
        warmup_epochs: int = 5,
        learning_rate: float = 0.002,
        use_compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(
            "num_classes",
            "warmup_epochs",
            "total_epochs",
            "image_size",
            "learning_rate",
        )
        model = create_model(
            model_name,
            num_classes=num_classes,
            in_chans=1,
            aa_layer="blurpc",
            drop_rate=0.25,
            exportable=True,
        )
        self.model = model.to(memory_format=torch.channels_last)  # type: ignore
        self.model_opt = torch.compile(
            self.model,
            options={"triton.cudagraphs": True, "shape_padding": True},
            dynamic=False,
        )  # type: ignore

        self.criterion = nn.CrossEntropyLoss()

        self.acc_top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.acc_top3 = Accuracy(task="multiclass", num_classes=num_classes, top_k=3)
        self.warm_up_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.use_compile = use_compile
        self.learning_rate = learning_rate
        self.model_name: str = model_name
        self.example_input_array: torch.Tensor = torch.randn(
            1, 1, image_size, image_size
        )

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        if self.use_compile:
            return self.model_opt(x)
        return self.model(x)

    def training_step(self, batch):
        image, image = batch["image"], batch["label"]
        pred = self.forward(image)
        loss = self.criterion(pred, image)
        self.log("train_loss", loss)
        self.log("train_acc", self.acc_top1(pred, image))
        return loss

    def validation_step(self, batch):
        image, label = batch["image"], batch["label"]
        pred = self.forward(image)
        loss = self.criterion(pred, label)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.acc_top1(pred, label), prog_bar=True)
        self.log("val_top3", self.acc_top3(pred, label))
        return loss

    def test_step(self, batch):
        image, label = batch["image"], batch["label"]
        pred = self.forward(image)
        self.log("val_acc", self.acc_top1(pred, label), prog_bar=True)
        self.log("val_top3", self.acc_top3(pred, label))

    def configure_optimizers(self):
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
            {"params": decay, "weight_decay": 0.06},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(
            optim_groups, lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-7
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-4,
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


class CNNModel(LightningModule):
    def __init__(
        self,
        num_classes: int,
        image_size: int,
        total_epochs: int,
        warmup_epochs: int = 5,
        use_compile: bool = False,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(
            "num_classes",
            "image_size",
            "learning_rate",
            "total_epochs",
            "warmup_epochs",
        )

        features = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        avgpool = nn.AdaptiveAvgPool2d((4, 4))
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )

        # Change model to channel last format for speed
        self.features = features.to(memory_format=torch.channels_last)  # type: ignore
        self.avgpool = avgpool.to(memory_format=torch.channels_last)  # type: ignore
        self.classifier = classifier.to(memory_format=torch.channels_last)  # type: ignore

        # Use torch.compile sometimes have problems with export, so add opt for this
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

        self.criterion = nn.CrossEntropyLoss()
        self.acc_top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.acc_top3 = Accuracy(task="multiclass", num_classes=num_classes, top_k=3)
        self.use_compile = use_compile
        self.example_input_array: torch.Tensor = torch.randn(
            1, 1, image_size, image_size
        )
        self.learning_rate = learning_rate
        self.total_epoches = total_epochs
        self.warm_up_epochs = warmup_epochs

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        if self.use_compile:
            x = self.features_opt(x)
            x = self.avgpool(x)
            return self.classifier_opt(x)
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)

    def training_step(self, batch):
        image, label = batch["image"], batch["label"]
        pred = self.forward(image)
        loss = self.criterion(pred, label)
        self.log("train_acc", self.acc_top1(pred, label))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        image, label = batch["image"], batch["label"]
        pred = self.forward(image)
        loss = self.criterion(pred, label)
        self.log("val_loss", loss)
        self.log("val_acc", self.acc_top1(pred, label), prog_bar=True)
        self.log("val_top3", self.acc_top3(pred, label))
        return loss

    def test_step(self, batch):
        image, label = batch["image"], batch["label"]
        pred = self.forward(image)
        self.log("val_acc", self.acc_top1(pred, label), prog_bar=True)
        self.log("val_top3", self.acc_top3(pred, label))

    def configure_optimizers(self):
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
            {"params": decay, "weight_decay": 0.06},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(
            optim_groups, lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-7
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=self.warm_up_epochs,
        )

        decay_scheduler = CosineAnnealingLR(
            optimizer, T_max=(self.total_epoches - self.warm_up_epochs), eta_min=1e-6
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
