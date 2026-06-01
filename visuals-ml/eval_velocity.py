import argparse
import random

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import yaml
from torch.utils.data import DataLoader, Subset

from data.velocity_dataset import VelocityDataset
from model.velocity_net import VelocityNet


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/velocity_local.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/velocity_best.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VelocityDataset(cfg["index_file"])
    n = len(dataset)
    indices = list(range(n))
    random.Random(cfg["seed"]).shuffle(indices)
    split = int(n * (1 - cfg["val_split"]))
    val_set = Subset(dataset, indices[split:])

    pin = device.type == "cuda"
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=pin,
    )

    model = VelocityNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.6f})")

    abs_errors = torch.zeros(2)
    count = 0
    with torch.no_grad():
        for crop_t, crop_t1, coords, target in val_loader:
            crop_t  = crop_t.to(device)
            crop_t1 = crop_t1.to(device)
            coords  = coords.to(device)
            target  = target.to(device)
            pred = model(crop_t, crop_t1, coords)
            abs_errors += (pred - target).abs().sum(dim=0).cpu()
            count += len(target)

    mae = abs_errors / count
    print(f"\nMAE on val set ({count} samples):")
    print(f"  vx: {mae[0]:.4f} m/s")
    print(f"  vy: {mae[1]:.4f} m/s")
    print(f"  mean: {mae.mean():.4f} m/s")


if __name__ == "__main__":
    main()
