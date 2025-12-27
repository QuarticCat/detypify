"""Train the model."""

import os
import shutil

import lightning as L
import msgspec
import orjson
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from proc_data import OUT_DIR as DATA_DIR
from proc_data import TypstSymInfo

OUT_DIR = "build/train"


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
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        acc = (pred.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
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


if __name__ == "__main__":
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Prepare data.
    transforms = [
        v2.Grayscale(),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    orig_data = ImageFolder(f"{DATA_DIR}/img", v2.Compose(transforms))
    train_data, test_data = random_split(orig_data, [0.9, 0.1])

    # Train model.
    model = TypstSymbolClassifier(num_classes=len(orig_data.classes))
    num_workers = os.process_cpu_count() or 1
    trainer = L.Trainer(max_epochs=20, default_root_dir=OUT_DIR)
    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_data,
            batch_size=128,
            num_workers=num_workers,
            pin_memory=True,
        ),
        val_dataloaders=DataLoader(
            test_data,
            batch_size=128,
            num_workers=num_workers,
            pin_memory=True,
        ),
    )

    # Export ONNX.
    model.to_onnx(
        f"{OUT_DIR}/model.onnx",
        torch.randn(1, 1, 32, 32),
        dynamo=True,
        external_data=False,
    )

    # Generate JSON for the infer page.
    with open(f"{DATA_DIR}/symbols.json", "rb") as f:
        sym_info = msgspec.json.decode(f.read(), type=list[TypstSymInfo])
    chr_to_sym = {s.char: s for s in sym_info}
    infer = []
    for c in orig_data.classes:
        sym = chr_to_sym[chr(int(c))]
        info = {"char": sym.char, "names": sym.names}
        if sym.markup_shorthand and sym.math_shorthand:
            info["shorthand"] = sym.markup_shorthand
        elif sym.markup_shorthand:
            info["markupShorthand"] = sym.markup_shorthand
        elif sym.math_shorthand:
            info["mathShorthand"] = sym.math_shorthand
        infer.append(info)
    with open(f"{OUT_DIR}/infer.json", "wb") as f:
        f.write(orjson.dumps(infer))

    # Generate JSON for the contrib page.
    contrib = {n: s.char for s in sym_info for n in s.names}
    with open(f"{OUT_DIR}/contrib.json", "wb") as f:
        f.write(orjson.dumps(contrib))
