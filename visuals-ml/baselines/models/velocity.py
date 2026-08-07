"""
Baseline C — per-object velocity regression, ported into the harness.

Wraps the existing VelocityNet (siamese ResNet18 over crops at frame T and T+1,
plus ego-motion and delta_t) so it trains through the same runner as box3d and
monodetr, with the same segment-level split and per-weather reporting.

THE LABEL CAVEAT — read before trusting any number this produces.

The target is the Waymo `speed.x/speed.y` field, taken verbatim by
data/build_velocity_index.py. On the 2-segment local sample only 28% of LiDAR
boxes have |v_xy| > 0.01 m/s (mean speed 0.15 m/s) — i.e. the labels are
overwhelmingly "parked". A model that always outputs zero would score a very
good MAE while having learned nothing.

So evaluate() reports a PREDICT-ZERO CONTROL alongside the model, plus the
moving-object subset. The numbers that decide whether this task is real:

  * `skill_vs_zero` > 0    the model beats always-predicting-zero. If it is <= 0
                           the task as labelled is degenerate and the fix is to
                           derive velocity by finite-differencing track centres
                           with ego-motion compensation, not to train longer.
  * `moving_frac`          fraction of val objects actually in motion. If this
                           is ~0.28 at full scale the local finding holds.
  * `mae_moving`           MAE restricted to moving objects — the honest number.

This is the cheapest way to settle the question, because it answers it as a side
effect of the deploy smoke test rather than as a separate study.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from data.velocity_dataset import VelocityDataset

from baselines.core.interface import BaselineModel
from baselines.core.registry import register_model
from baselines.core.utils import split_by_group
from model.velocity_net import VelocityNet as _VelocityNetCore

# |v_xy| above which an object counts as moving (m/s). 0.5 m/s ~ 1.8 km/h, below
# which Waymo's speed estimate is not meaningfully distinguishable from zero.
MOVING_THRESH = 0.5


class _VelocityWeatherDataset(VelocityDataset):
    """VelocityDataset that also returns the weather variant of each sample."""

    def __getitem__(self, idx):
        crop_t, crop_t1, coords, target = super().__getitem__(idx)
        return crop_t, crop_t1, coords, target, self._records[idx].get("weather", "unknown")

    def groups_for(self, kind: str):
        if kind == "segment":
            return [r["segment"] for r in self._records]
        if kind == "frame":
            return [f"{r['segment']}|{r.get('camera_dir')}|{r.get('laser_id')}"
                    for r in self._records]
        raise ValueError(f"group_by must be 'segment' or 'frame', got {kind!r}")


@register_model("velocity")
class VelocityBaseline(BaselineModel):
    def __init__(self, cfg: dict):
        super().__init__()
        self.core = _VelocityNetCore()
        self.criterion = nn.SmoothL1Loss(beta=cfg.get("huber_beta", 0.5))

    @staticmethod
    def build_datasets(cfg: dict):
        ds = _VelocityWeatherDataset(cfg["index_file"])
        print(f"Dataset size: {len(ds)} object pairs")
        group_by = cfg.get("group_by", "segment")
        if group_by != "segment":
            print(f"WARNING: group_by={group_by!r} — validation is optimistic.")
        return split_by_group(ds, ds.groups_for(group_by),
                              cfg["val_split"], cfg["seed"])

    def collate_fn(self, batch):
        crop_t = torch.stack([b[0] for b in batch])
        crop_t1 = torch.stack([b[1] for b in batch])
        coords = torch.stack([b[2] for b in batch])
        target = torch.stack([b[3] for b in batch])
        weather = [b[4] for b in batch]
        return crop_t, crop_t1, coords, target, weather

    def training_step(self, batch, device):
        crop_t, crop_t1, coords, target, _w = batch
        crop_t, crop_t1 = crop_t.to(device), crop_t1.to(device)
        coords, target = coords.to(device), target.to(device)
        loss = self.criterion(self.core(crop_t, crop_t1, coords), target)
        return loss, {"loss": loss.item(), "batch_size": len(target)}

    @torch.no_grad()
    def evaluate(self, loader, device) -> dict:
        self.eval()
        preds, targets, weathers = [], [], []
        for crop_t, crop_t1, coords, target, weather in loader:
            crop_t, crop_t1 = crop_t.to(device), crop_t1.to(device)
            coords = coords.to(device)
            preds.append(self.core(crop_t, crop_t1, coords).float().cpu().numpy())
            targets.append(target.numpy())
            weathers.extend(weather)

        if not preds:
            return {"monitor": float("inf"), "n": 0}

        pred = np.concatenate(preds)
        gt = np.concatenate(targets)
        weathers = np.asarray(weathers)

        err = np.linalg.norm(pred - gt, axis=1)          # model error, m/s
        zero_err = np.linalg.norm(gt, axis=1)            # error of predicting 0
        speed = np.linalg.norm(gt, axis=1)
        moving = speed > MOVING_THRESH

        mae = float(err.mean())
        mae_zero = float(zero_err.mean())
        metrics = {
            "monitor": mae,                              # lower is better
            "mae": mae,
            "mae_zero_baseline": mae_zero,
            # >0 means the model is genuinely better than predicting nothing.
            "skill_vs_zero": float(mae_zero - mae),
            "moving_frac": float(moving.mean()),
            "mae_moving": float(err[moving].mean()) if moving.any() else float("nan"),
            "mae_zero_moving": float(zero_err[moving].mean()) if moving.any() else float("nan"),
            "mae_static": float(err[~moving].mean()) if (~moving).any() else float("nan"),
            "n": int(len(err)),
        }

        per_weather = defaultdict(dict)
        for w in sorted(set(weathers.tolist())):
            m = weathers == w
            per_weather[w] = {
                "n": int(m.sum()),
                "mae": float(err[m].mean()),
                "mae_zero_baseline": float(zero_err[m].mean()),
                "skill_vs_zero": float(zero_err[m].mean() - err[m].mean()),
            }
        metrics["per_weather"] = dict(per_weather)
        return metrics
