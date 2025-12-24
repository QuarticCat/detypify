"""Train the model."""

import os
from typing import Callable

import msgspec
import orjson
import torch
import torchinfo
from torch import Generator, Tensor, nn, onnx, optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from proc_data import OUT_DIR as DATA_DIR
from proc_data import TypstSymInfo

type LossFn = Callable[[Tensor, Tensor], Tensor]

OUT_DIR = "build/train"


def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: LossFn,
    optimizer: optim.Optimizer,
):
    size = len(dataloader.dataset)
    current = 0
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current += len(X)
        if batch % 100 == 0:
            print(f"Train> loss: {loss.item():>7f} [{current:>6d}/{size:>6d}]")


def test_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: LossFn,
):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).float().sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test > acc: {(100 * correct):>0.1f}%, avg loss: {test_loss:>8f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    transform = v2.Compose(
        [
            v2.Grayscale(),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    orig_data = ImageFolder(f"{DATA_DIR}/img", transform)
    train_data, test_data = random_split(orig_data, [0.9, 0.1], Generator(device))
    train_loader = DataLoader(train_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    model = nn.Sequential(
        nn.Conv2d(1, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 5 * 5, len(orig_data.classes)),
    )
    torchinfo.summary(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00025)

    for epoch in range(10):
        print(f"\n### Epoch {epoch}")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)
        print("-------------------------------------")

    os.makedirs(OUT_DIR, exist_ok=True)
    prog = onnx.export(model, (torch.randn(1, 1, 32, 32),), dynamo=True)
    prog.save(f"{OUT_DIR}/model.onnx")

    content = open(f"{DATA_DIR}/symbols.json", "rb").read()
    sym_info = msgspec.json.decode(content, type=list[TypstSymInfo])
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
    open(f"{OUT_DIR}/infer.json", "wb").write(orjson.dumps(infer))

    contrib = {n: s.char for s in sym_info for n in s.names}
    open(f"{OUT_DIR}/contrib.json", "wb").write(orjson.dumps(contrib))
