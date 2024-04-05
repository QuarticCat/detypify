import os

import orjson
import torch
import torchinfo
from torch import Generator, nn, onnx, optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


def train_loop(dataloader, model, loss_fn, optimizer):
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


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test>  acc: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    transform = v2.Compose(
        [
            v2.Grayscale(),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    orig_data = ImageFolder("migrate-out/data", transform)
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

    os.makedirs("train-out", exist_ok=True)
    onnx.export(model, torch.randn(1, 1, 32, 32), "train-out/model.onnx")

    symbols = orjson.loads(open("migrate-out/symbols.json", "rb").read())
    classes = [None] * len(orig_data.classes)
    for sym in symbols:
        if sym["name"] not in orig_data.class_to_idx:
            continue
        logo = chr(sym["codepoint"])
        info = [
            ("Name", sym["name"]),
            ("Escape", "\\u" + f"{{{sym['codepoint']:0>4X}}}"),
        ]
        if sym["markup-shorthand"] and sym["math-shorthand"]:
            info.append(("Shorthand", sym["markup-shorthand"]))
        elif sym["markup-shorthand"]:
            info.append(("Markup Shorthand", sym["markup-shorthand"]))
        elif sym["math-shorthand"]:
            info.append(("Math Shorthand", sym["math-shorthand"]))
        classes[orig_data.class_to_idx[sym["name"]]] = (logo, info)
    open("train-out/classes.json", "wb").write(orjson.dumps(classes))

    symbols = {s["name"]: chr(s["codepoint"]) for s in symbols}
    open("train-out/symbols.json", "wb").write(orjson.dumps(symbols))
