"""
Baseline B — amodal 3D box regression from a given 2D box.

The task: given the ground-truth 2D box of a vehicle and the image, recover its
full 3D box — centre (x, y, z), dimensions (h, w, l) and yaw ry — in the camera
optical frame. Detection is NOT part of the task; the 2D box is handed to the
model. That makes B the clean lower bound for the monocular detector (Baseline A),
which has to solve this *and* find the objects.

Architecture is deliberately ordinary: PositionNet's two-stream ResNet18 (full
image for context, crop for appearance) with the coord vector fused in, widened
from 3 outputs to 8 (see baselines/data/box3d_dataset.py for the encoding).
Nothing here is novel by design — it is the reference point A must beat.

Why it matters for introspection: evaluate() reports per-object 3D IoU, which is
a bounded, scale-free failure signal. The Paper-1 label pre-pass currently
thresholds centre distance in metres and needs --tau-percentile to avoid a
degenerate all-failure split; `iou_3d < 0.5` needs no such calibration.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from baselines.core.interface import BaselineModel
from baselines.core.iou3d import iou_3d
from baselines.core.registry import register_model
from baselines.core.utils import split_by_group
from baselines.data.box3d_dataset import (
    Box3DDataset, compute_dim_anchor, decode_targets, load_records,
)

N_OUTPUTS = 8
IOU_THRESHOLDS = (0.25, 0.5, 0.7)
DEPTH_BANDS = ((0, 20), (20, 40), (40, 1e9))


def _build_encoder() -> nn.Sequential:
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    return nn.Sequential(
        backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
    )


class Box3DNet(nn.Module):
    """Full image (3,640,960) + crop (3,224,224) + coords (8) -> 8 box params."""

    def __init__(self, depth_prior: float = 25.0):
        super().__init__()
        self.encoder_full = _build_encoder()
        self.encoder_crop = _build_encoder()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(512 + 512 + 8, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, N_OUTPUTS),
        )
        self._init_output_prior(depth_prior)

    def _init_output_prior(self, depth_prior: float):
        """Start the model at a sensible box instead of a random one.

        Without this, the depth output starts near 0, so exp(t2) ~ 1 m while the
        data sits at ~25 m. The resulting gradient dwarfs every other term and
        depth thrashes over hundreds of metres for the first epochs (observed:
        depth_mae swinging 4 m -> 82 m). Biasing the head to the dataset's mean
        depth, the mean dimensions (residual 0) and yaw 0 makes the first
        forward pass an average vehicle at average range. Same idea as
        RetinaNet's prior-probability bias init; it changes the starting point,
        not the hypothesis space.
        """
        final = self.head[-1]
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[2] = float(np.log(depth_prior))  # log-depth
            final.bias[3:6] = 0.0                       # dims == anchor
            final.bias[6] = 0.0                         # sin(ry)
            final.bias[7] = 1.0                         # cos(ry) -> ry = 0
            # Shrink the last layer so the prior dominates at step 0 and the
            # network has to earn its deviations from it.
            final.weight.mul_(0.01)

    def forward(self, image, crop, coords):
        feat_full = self.pool(self.encoder_full(image)).flatten(1)
        feat_crop = self.pool(self.encoder_crop(crop)).flatten(1)
        return self.head(torch.cat([feat_full, feat_crop, coords], dim=1))


@register_model("box3d")
class Box3DBaseline(BaselineModel):
    def __init__(self, cfg: dict):
        super().__init__()
        self.core = Box3DNet(depth_prior=cfg.get("depth_prior", 25.0))
        # SmoothL1 over the encoded target: robust to the long tail of far/tiny
        # objects that would otherwise dominate a plain MSE.
        self.criterion = nn.SmoothL1Loss(beta=cfg.get("huber_beta", 0.1))
        anchor = cfg.get("dim_anchor")
        self.register_buffer(
            "dim_anchor",
            torch.tensor(anchor if anchor else [2.03, 2.06, 4.56], dtype=torch.float32),
        )
        # Term weights: depth is the hard part of monocular 3D, so it is not
        # drowned out by the 5 easy parameters.
        self.w_center = cfg.get("w_center", 1.0)
        self.w_depth = cfg.get("w_depth", 2.0)
        self.w_dim = cfg.get("w_dim", 1.0)
        self.w_angle = cfg.get("w_angle", 1.0)

    @staticmethod
    def build_datasets(cfg: dict):
        records = load_records(cfg["index_file"])
        train_weathers = cfg.get("train_weathers")
        if train_weathers:
            keep = set(train_weathers)
            records = [r for r in records if r.get("weather") in keep]
            print(f"Filtered to weathers {sorted(keep)}: {len(records)} image records")

        anchor = compute_dim_anchor(records)
        print(f"Dimension anchor (h, w, l) = {np.round(anchor, 3).tolist()}")

        ds = Box3DDataset(
            records,
            dim_anchor=anchor,
            min_box_px=cfg.get("min_box_px", 0.0),
            min_pts=cfg.get("min_pts", 0),
            drop_truncated=cfg.get("drop_truncated", True),
        )
        print(f"Dataset size: {len(ds)} objects")
        # Split on segments, never on records: the same object recurs across all
        # 10 weather renderings and across neighbouring ~10 Hz frames.
        group_by = cfg.get("group_by", "segment")
        if group_by != "segment":
            print(f"WARNING: group_by={group_by!r} — validation is optimistic. "
                  "Only 'segment' gives a leak-free split; see box3d_dataset.py.")
        return split_by_group(ds, ds.groups_for(group_by),
                              cfg["val_split"], cfg["seed"])

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(cfg)

    def collate_fn(self, batch):
        full = torch.stack([b[0] for b in batch])
        crop = torch.stack([b[1] for b in batch])
        coords = torch.stack([b[2] for b in batch])
        target = torch.stack([b[3] for b in batch])
        gt = torch.stack([b[4] for b in batch])
        meta = [b[5] for b in batch]
        return full, crop, coords, target, gt, meta

    def _loss(self, pred, target):
        c = self.criterion(pred[:, 0:2], target[:, 0:2])
        d = self.criterion(pred[:, 2:3], target[:, 2:3])
        s = self.criterion(pred[:, 3:6], target[:, 3:6])
        a = self.criterion(pred[:, 6:8], target[:, 6:8])
        total = (self.w_center * c + self.w_depth * d
                 + self.w_dim * s + self.w_angle * a)
        return total, {"center": c.item(), "depth": d.item(),
                       "dim": s.item(), "angle": a.item()}

    def training_step(self, batch, device):
        full, crop, coords, target, _gt, _meta = batch
        full, crop = full.to(device), crop.to(device)
        coords, target = coords.to(device), target.to(device)
        pred = self.core(full, crop, coords)
        loss, parts = self._loss(pred, target)
        return loss, {**parts, "batch_size": len(target)}

    @torch.no_grad()
    def evaluate(self, loader, device) -> dict:
        self.eval()
        rows = []   # (weather, depth, iou, |dz|, z, centre_err, dim_err, yaw_err)

        for full, crop, coords, _target, gt, meta in loader:
            full, crop, coords = full.to(device), crop.to(device), coords.to(device)
            pred = decode_targets(self.core(full, crop, coords), self.dim_anchor)
            p = pred.float().cpu().numpy()
            g = gt.numpy()

            for i in range(len(g)):
                iou = iou_3d(p[i], g[i])
                center_err = float(np.linalg.norm(p[i, 0:3] - g[i, 0:3]))
                dim_err = float(np.abs(p[i, 3:6] - g[i, 3:6]).mean())
                dyaw = abs(float(p[i, 6] - g[i, 6])) % (2 * np.pi)
                dyaw = min(dyaw, 2 * np.pi - dyaw)
                # A box and its 180-degree flip are the same box; score the
                # ambiguity honestly rather than punishing a flipped yaw twice.
                dyaw = min(dyaw, abs(np.pi - dyaw))
                rows.append((
                    meta[i]["weather"], meta[i]["depth"], iou,
                    abs(float(p[i, 2] - g[i, 2])), float(g[i, 2]),
                    center_err, dim_err, dyaw,
                ))

        if not rows:
            return {"monitor": float("inf"), "n": 0}

        arr_iou = np.array([r[2] for r in rows])
        arr_dz = np.array([r[3] for r in rows])
        arr_z = np.array([r[4] for r in rows])
        arr_ce = np.array([r[5] for r in rows])
        arr_de = np.array([r[6] for r in rows])
        arr_yaw = np.array([r[7] for r in rows])

        def summarize(mask):
            if not mask.any():
                return None
            out = {
                "n": int(mask.sum()),
                "iou3d_mean": float(arr_iou[mask].mean()),
                "depth_mae": float(arr_dz[mask].mean()),
                "depth_absrel": float((arr_dz[mask] / np.maximum(arr_z[mask], 1e-6)).mean()),
                "center_mae": float(arr_ce[mask].mean()),
                "dim_mae": float(arr_de[mask].mean()),
                "yaw_mae_deg": float(np.degrees(arr_yaw[mask]).mean()),
            }
            for t in IOU_THRESHOLDS:
                out[f"iou@{t}"] = float((arr_iou[mask] >= t).mean())
            return out

        all_mask = np.ones(len(rows), dtype=bool)
        metrics = summarize(all_mask)
        # Lower is better; mean IoU is the headline, so monitor its complement.
        metrics["monitor"] = 1.0 - metrics["iou3d_mean"]

        weathers = np.array([r[0] for r in rows])
        metrics["per_weather"] = {
            w: summarize(weathers == w) for w in sorted(set(weathers.tolist()))
        }
        depths = np.array([r[1] for r in rows])
        metrics["per_depth"] = {
            f"{lo}-{hi if hi < 1e8 else 'inf'}m": summarize((depths >= lo) & (depths < hi))
            for lo, hi in DEPTH_BANDS
        }
        return metrics
