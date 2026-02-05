from abc import abstractmethod
from typing import Literal, override

import torch
from lightning import LightningModule
from timm import create_model
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torchmetrics import Accuracy

# Selected models for efficiency, ranked by model size, ascending
type TimmModelName = Literal[
    # optimal ones
    "mobilenetv4_conv_small_035",
    "mobilenetv4_conv_small_050",
    # for exp
    "mobilenetv4_conv_small",
    "mobilenetv4_conv_medium",
    "mobilenetv4_hybrid_medium_075",
    "mobilenetv4_hybrid_medium",
]


class BaseModel(LightningModule):
    """Base class for math symbol recognition models."""

    def __init__(
        self,
        num_classes: int,
        image_size: int,
        total_epochs: int,
        warmup_epochs: int = 5,
        learning_rate: float = 1e-3,
        use_compile: bool = False,
    ):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.acc_top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.acc_top3 = Accuracy(task="multiclass", num_classes=num_classes, top_k=3)
        self.use_compile = use_compile
        self.learning_rate = learning_rate
        self.total_epochs = total_epochs
        self.warm_up_epochs = warmup_epochs
        self.example_input_array: Tensor = torch.randn(1, 1, image_size, image_size)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass - must be implemented by subclasses."""
        pass

    @override
    def training_step(self, batch, batch_idx=0):
        image, label = batch["image"], batch["label"]
        pred = self.forward(image)
        loss = self.criterion(pred, label)
        self.log("train_loss", loss)
        self.log("train_acc", self.acc_top1(pred, label))
        return loss

    @override
    def validation_step(self, batch, batch_idx=0):
        image, label = batch["image"], batch["label"]
        pred = self.forward(image)
        loss = self.criterion(pred, label)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.acc_top1(pred, label), prog_bar=True)
        self.log("val_top3", self.acc_top3(pred, label))
        return loss

    def test_step(self, batch, batch_idx=0):  # noqa: ARG002
        image, label = batch["image"], batch["label"]
        pred = self.forward(image)
        self.log("test_acc", self.acc_top1(pred, label), prog_bar=True)
        self.log("test_top3", self.acc_top3(pred, label))
        return pred

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


class TimmModel(BaseModel):
    def __init__(
        self,
        num_classes: int,
        model_name: TimmModelName,
        total_epochs: int,
        image_size: int,
        warmup_epochs: int = 5,
        learning_rate: float = 0.002,
        use_compile: bool = False,
    ):
        super().__init__(
            num_classes=num_classes,
            image_size=image_size,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            learning_rate=learning_rate,
            use_compile=use_compile,
        )
        self.save_hyperparameters(
            "num_classes",
            "model_name",
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
            drop_rate=0.15,
            exportable=True,
        )
        self.model = model.to(memory_format=torch.channels_last)  # type: ignore
        self.model_opt = torch.compile(
            self.model,
            options={"triton.cudagraphs": True, "shape_padding": True},
            dynamic=False,
        )

        self.model_name: str = model_name

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        if self.use_compile:
            return self.model_opt(x)
        return self.model(x)


class CNNModel(BaseModel):
    def __init__(
        self,
        num_classes: int,
        image_size: int,
        total_epochs: int,
        warmup_epochs: int = 5,
        use_compile: bool = False,
        learning_rate: float = 1e-3,
    ):
        super().__init__(
            num_classes=num_classes,
            image_size=image_size,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            learning_rate=learning_rate,
            use_compile=use_compile,
        )
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

        # Use `torch.compile` by default cannot be exported
        # so add it as seperate optimized module
        self.features_opt = torch.compile(
            self.features,
            options={"triton.cudagraphs": True, "shape_padding": True},
            dynamic=False,
        )
        self.classifier_opt = torch.compile(
            self.classifier,
            options={"triton.cudagraphs": True, "shape_padding": True},
            dynamic=False,
        )

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        if self.use_compile:
            x = self.features_opt(x)
            x = self.avgpool(x)
            return self.classifier_opt(x)
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)
