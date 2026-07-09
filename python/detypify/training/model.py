from abc import abstractmethod
from functools import partial
from typing import override

import torch
from lightning import LightningModule
from timm.layers import set_layer_config
from timm.models._efficientnet_builder import decode_arch_def, round_channels
from timm.models.mobilenetv5 import MobileNetV5
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import Accuracy, F1Score

from detypify.config import ModelFamily, parse_mobilenet_model_name

_GELU = partial(nn.GELU, approximate="tanh")

# Keep the recognition-specific MobileNetV5 layout explicit so its channel multiplier remains configurable.
_MOBILENET_V5_ARCH_DEF = [
    ["er_r1_k3_s2_e4_c64", "er_r1_k3_s1_e4_c64"],
    ["uir_r1_a3_k5_s2_e6_c128", "uir_r1_a3_k0_s1_e4_c128"],
    ["uir_r1_a5_k5_s2_e6_c192", "uir_r1_a0_k0_s1_e2_c192", "uir_r1_a0_k0_s1_e2_c192", "uir_r1_a0_k0_s1_e2_c192"],
    ["uir_r1_a5_k5_s2_e6_c256", "uir_r1_a0_k0_s1_e2_c256", "uir_r1_a0_k0_s1_e2_c256"],
]


def create_project_model(model_name: str, **kwargs) -> nn.Module:
    """Create an exportable MobileNet variant from the project's compact model name."""
    model_spec = parse_mobilenet_model_name(model_name)
    with set_layer_config(exportable=True):
        if model_spec.family == ModelFamily.v4:
            from timm.models.mobilenetv3 import _gen_mobilenet_v4

            model = _gen_mobilenet_v4(
                "mobilenetv4_conv_small",
                channel_multiplier=model_spec.size,
                aa_layer="blurpc",
                **kwargs,
            )
        else:
            model = MobileNetV5(
                block_args=decode_arch_def(_MOBILENET_V5_ARCH_DEF),
                num_features=384,
                stem_size=24,
                use_msfa=True,
                norm_layer=nn.BatchNorm2d,
                act_layer=_GELU,
                round_chs_fn=partial(round_channels, multiplier=model_spec.size),
                layer_scale_init_value=1e-5,
                **kwargs,
            )
    return model


class BaseModel(LightningModule):
    """Base class for math symbol recognition models."""

    def __init__(
        self,
        num_classes: int,
        image_size: int,
        total_epochs: int,
        warmup_epochs: int,
        learning_rate: float,
        label_smoothing: float,
        *,
        use_compile: bool,
    ):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.train_acc_top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.val_acc_top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.val_acc_top3 = Accuracy(task="multiclass", num_classes=num_classes, top_k=3)
        self.val_f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.train_f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_acc_top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.test_acc_top3 = Accuracy(task="multiclass", num_classes=num_classes, top_k=3)
        self.test_f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.use_compile = use_compile
        self.learning_rate = learning_rate
        self.total_epochs = total_epochs
        self.warm_up_epochs = warmup_epochs
        self.example_input_array: Tensor = torch.randn(1, 1, image_size, image_size)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass - must be implemented by subclasses."""

    @override
    def training_step(self, batch, batch_idx):
        image, label = batch["image"], batch["label"]
        pred = self.forward(image)
        loss = self.criterion(pred, label)
        self.train_acc_top1.update(pred, label)
        self.train_f1_macro.update(pred, label)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc_top1, on_epoch=True)
        self.log("train_f1", self.train_f1_macro, on_epoch=True, prog_bar=True)
        return loss

    @override
    def validation_step(self, batch, batch_idx):
        image, label = batch["image"], batch["label"]
        pred = self.forward(image)
        loss = self.criterion(pred, label)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.val_acc_top1.update(pred, label)
        self.val_acc_top3.update(pred, label)
        self.val_f1_macro.update(pred, label)
        self.log("val_acc", self.val_acc_top1, on_epoch=True, prog_bar=True)
        self.log("val_top3", self.val_acc_top3, on_epoch=True)
        self.log("val_f1", self.val_f1_macro, on_epoch=True, prog_bar=True)
        return loss

    @override
    def test_step(self, batch, batch_idx):
        image, label = batch["image"], batch["label"]
        pred = self.forward(image)
        loss = self.criterion(pred, label)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.test_acc_top1.update(pred, label)
        self.test_acc_top3.update(pred, label)
        self.test_f1_macro.update(pred, label)
        self.log("test_acc", self.test_acc_top1, on_epoch=True, prog_bar=True)
        self.log("test_top3", self.test_acc_top3, on_epoch=True)
        self.log("test_f1", self.test_f1_macro, on_epoch=True, prog_bar=True)
        return pred

    def configure_optimizers(self):
        """Configure AdamW with selective decay and linear-warmup cosine scheduling."""
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Biases and scale/normalization parameters should not be regularized by weight decay.
            if param.ndim <= 1 or name.endswith(".bias") or "norm" in name or "bn" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        optim_groups = [
            {"params": decay, "weight_decay": 0.06},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(
            optim_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-7,
            fused=next(self.parameters()).is_cuda,
        )

        # SequentialLR switches once from warmup to cosine decay at the configured epoch boundary.
        warmup_scheduler = LinearLR(optimizer, total_iters=self.warm_up_epochs)
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
            },
        }


class MobileNetModel(BaseModel):
    """Lightning wrapper around the project's grayscale MobileNet classifiers."""

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        total_epochs: int,
        image_size: int,
        warmup_epochs: int,
        learning_rate: float,
        label_smoothing: float,
        *,
        use_compile: bool = False,
    ):
        super().__init__(
            num_classes=num_classes,
            image_size=image_size,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            learning_rate=learning_rate,
            label_smoothing=label_smoothing,
            use_compile=use_compile,
        )
        self.save_hyperparameters(
            "num_classes",
            "model_name",
            "warmup_epochs",
            "total_epochs",
            "image_size",
            "learning_rate",
            "label_smoothing",
        )
        model = create_project_model(model_name, num_classes=num_classes, in_chans=1, drop_rate=0.15)
        # Channels-last layout is shared by eager and compiled paths to avoid conversions inside the backbone.
        self.model = model.to(memory_format=torch.channels_last)  # type: ignore
        self.model_opt = torch.compile(self.model, mode="max-autotune", dynamic=False) if use_compile else None

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        if self.use_compile and self.model_opt is not None:
            return self.model_opt(x)
        return self.model(x)
