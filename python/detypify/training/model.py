from abc import abstractmethod
from typing import override

import torch
from detypify.config import ModelName
from lightning import LightningModule
from timm import create_model
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torchmetrics import Accuracy


class BaseModel(LightningModule):
    """Base class for math symbol recognition models."""

    def __init__(
        self,
        num_classes: int,
        image_size: int,
        total_epochs: int,
        warmup_epochs: int = 5,
        learning_rate: float = 4e-4,
        *,
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

    @override
    def test_step(self, batch, batch_idx=0):
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
            if param.ndim <= 1 or name.endswith(".bias") or "norm" in name or "bn" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        optim_groups = [
            {"params": decay, "weight_decay": 0.06},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-7)

        warmup_scheduler = LinearLR(
            optimizer,
            total_iters=self.warm_up_epochs,
        )

        decay_scheduler = CosineAnnealingLR(optimizer, T_max=(self.total_epochs - self.warm_up_epochs), eta_min=1e-6)

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
        model_name: ModelName | str,
        total_epochs: int,
        image_size: int,
        warmup_epochs: int = 5,
        learning_rate: float = 0.002,
        *,
        use_compile: bool = False,
    ):
        model_name = str(model_name)
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

        self.model_opt = torch.compile(self.model, mode="max-autotune", dynamic=False) if use_compile else None

        self.model_name: str = model_name

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        if self.use_compile and self.model_opt is not None:
            return self.model_opt(x)
        return self.model(x)
