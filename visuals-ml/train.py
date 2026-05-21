import argparse
import random
from pathlib import Path

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from data.dataset import PositionDataset
from model.position_net import PositionNet

LOG_INTERVAL = 50


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def split_dataset(dataset, val_fraction: float, seed: int):
    n = len(dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    split = int(n * (1 - val_fraction))
    return Subset(dataset, indices[:split]), Subset(dataset, indices[split:])


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool, epoch: int, total_epochs: int):
    model.train(train)
    total_loss = 0.0
    phase = "train" if train else "val"
    n_batches = len(loader)

    with torch.set_grad_enabled(train):
        for batch_idx, (image, crop, coords, target) in enumerate(loader):
            image = image.to(device)
            crop = crop.to(device)
            coords = coords.to(device)
            target = target.to(device)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred = model(image, crop, coords)
                loss = criterion(pred, target)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * len(target)

            if (batch_idx + 1) % LOG_INTERVAL == 0 or (batch_idx + 1) == n_batches:
                print(f"  [{phase}] epoch {epoch}/{total_epochs}  "
                      f"batch {batch_idx+1}/{n_batches}  "
                      f"loss={loss.item():.4f}", flush=True)

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = PositionDataset(cfg["index_file"])
    print(f"Dataset size: {len(dataset)}")

    train_set, val_set = split_dataset(dataset, cfg["val_split"], cfg["seed"])
    print(f"Train: {len(train_set)}  Val: {len(val_set)}")

    pin = device.type == "cuda"
    workers = cfg["num_workers"]
    persistent = workers > 0
    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=workers, pin_memory=pin,
        persistent_workers=persistent, prefetch_factor=4 if persistent else None,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=workers, pin_memory=pin,
        persistent_workers=persistent, prefetch_factor=4 if persistent else None,
    )

    model = PositionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        print(f"Resumed from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True,  epoch=epoch, total_epochs=cfg["epochs"])
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, scaler, device, train=False, epoch=epoch, total_epochs=cfg["epochs"])
        print(f"Epoch {epoch:3d}/{cfg['epochs']}  train={train_loss:.4f}  val={val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss},
                       checkpoint_dir / "best.pt")
            print(f"  --> checkpoint saved (val_loss={val_loss:.4f})", flush=True)


if __name__ == "__main__":
    main()
