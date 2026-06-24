"""
PositionNet re-registered under the harness.

This is the trivial "regression" baseline that proves the abstraction generalizes
before MonoDETR (a per-image detector) is wired in. It reuses the existing
PositionDataset and PositionNet core verbatim; the only addition is per-sample
weather tagging so evaluate() can report the per-weather-variant breakdown that
is the whole point of the visuals dataset.
"""

from collections import defaultdict

import torch
import torch.nn as nn

from data.dataset import PositionDataset
from model.position_net import PositionNet as _PositionNetCore

from baselines.core.interface import BaselineModel
from baselines.core.registry import register_model
from baselines.core.utils import split_dataset

_AXES = ("x", "y", "z")


class _PositionWeatherDataset(PositionDataset):
    """PositionDataset that also returns the weather variant of each sample."""

    def __getitem__(self, idx):
        image, crop, coords, target = super().__getitem__(idx)
        weather = self._records[idx].get("weather", "unknown")
        return image, crop, coords, target, weather


@register_model("positionnet")
class PositionNetBaseline(BaselineModel):
    def __init__(self, cfg: dict):
        super().__init__()
        self.core = _PositionNetCore()
        self.criterion = nn.MSELoss()

    @staticmethod
    def build_datasets(cfg: dict):
        ds = _PositionWeatherDataset(cfg["index_file"])
        print(f"Dataset size: {len(ds)}")
        return split_dataset(ds, cfg["val_split"], cfg["seed"])

    def collate_fn(self, batch):
        images = torch.stack([b[0] for b in batch])
        crops = torch.stack([b[1] for b in batch])
        coords = torch.stack([b[2] for b in batch])
        targets = torch.stack([b[3] for b in batch])
        weather = [b[4] for b in batch]
        return images, crops, coords, targets, weather

    def training_step(self, batch, device):
        image, crop, coords, target, _weather = batch
        image, crop = image.to(device), crop.to(device)
        coords, target = coords.to(device), target.to(device)
        pred = self.core(image, crop, coords)
        loss = self.criterion(pred, target)
        return loss, {"mse": loss.item(), "batch_size": len(target)}

    @torch.no_grad()
    def evaluate(self, loader, device) -> dict:
        self.eval()
        # Sum of absolute error per axis, overall and per weather variant.
        overall = torch.zeros(3)
        overall_n = 0
        per_weather_err = defaultdict(lambda: torch.zeros(3))
        per_weather_n = defaultdict(int)

        for image, crop, coords, target, weather in loader:
            image, crop = image.to(device), crop.to(device)
            coords, target = coords.to(device), target.to(device)
            pred = self.core(image, crop, coords)
            abs_err = (pred - target).abs().cpu()   # (B, 3)

            overall += abs_err.sum(dim=0)
            overall_n += abs_err.shape[0]
            for i, w in enumerate(weather):
                per_weather_err[w] += abs_err[i]
                per_weather_n[w] += 1

        mae = overall / max(overall_n, 1)
        metrics = {
            "monitor": float(mae.mean()),     # lower is better
            "mae_mean": float(mae.mean()),
            "mae_x": float(mae[0]),
            "mae_y": float(mae[1]),
            "mae_z": float(mae[2]),
            "n": overall_n,
        }
        metrics["per_weather"] = {
            w: {
                "n": per_weather_n[w],
                **{f"mae_{ax}": float((per_weather_err[w] / per_weather_n[w])[i])
                   for i, ax in enumerate(_AXES)},
                "mae_mean": float((per_weather_err[w] / per_weather_n[w]).mean()),
            }
            for w in sorted(per_weather_err)
        }
        return metrics
