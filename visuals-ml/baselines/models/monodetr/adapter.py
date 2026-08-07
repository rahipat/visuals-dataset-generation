"""
MonoDETR wrapped behind the BaselineModel interface.

Builds the vendored MonoDETR model + SetCriterion, and bridges our per-image
detection dataset to its exact batch contract (NestedTensor images, calibs (B,3,4),
list-of-dict targets, img_sizes). Decodes predictions back to camera-frame
locations for the Option A center-error metric.

Vendored model defaults come from the upstream configs/monodetr.yaml; we override
for the visuals dataset: single vehicle class, and depth_max raised to cover
Waymo's longer range (objects out past 75 m vs KITTI's 60 m).
"""

from __future__ import annotations

import numpy as np
import torch

from baselines.core.interface import BaselineModel
from baselines.core.metrics import CenterErrorMetric, Sample
from baselines.core.registry import register_model
from baselines.core.utils import split_by_group
from baselines.data._vendor_path import ensure_vendor_on_path
from baselines.data.detection_dataset import (
    DetectionDataset, MAX_OBJS, compute_mean_size, load_records,
)

ensure_vendor_on_path()
from lib.models.monodetr.monodetr import build as build_monodetr  # noqa: E402

# Upstream defaults (configs/monodetr.yaml -> model:) with visuals overrides.
DEFAULT_MODEL_CFG = {
    "num_classes": 1, "device": "cuda", "return_intermediate_dec": True,
    "backbone": "resnet50", "train_backbone": True, "num_feature_levels": 4,
    "dilation": False, "position_embedding": "sine", "masks": False,
    "mode": "LID", "num_depth_bins": 80, "depth_min": 1e-3, "depth_max": 80.0,
    "with_box_refine": True, "two_stage": False, "use_dab": False, "use_dn": False,
    "two_stage_dino": False, "init_box": False, "enc_layers": 3, "dec_layers": 3,
    "hidden_dim": 256, "dim_feedforward": 256, "dropout": 0.1, "nheads": 8,
    "num_queries": 50, "enc_n_points": 4, "dec_n_points": 4, "group_num": 1,
    "scalar": 5, "label_noise_scale": 0.2, "box_noise_scale": 0.4, "num_patterns": 0,
    "aux_loss": True, "cls_loss_coef": 2, "focal_alpha": 0.25, "bbox_loss_coef": 5,
    "giou_loss_coef": 2, "3dcenter_loss_coef": 10, "dim_loss_coef": 1,
    "angle_loss_coef": 1, "depth_loss_coef": 1, "depth_map_loss_coef": 1,
    "set_cost_class": 2, "set_cost_bbox": 5, "set_cost_giou": 2, "set_cost_3dcenter": 10,
}

_TARGET_KEYS = ["labels", "boxes", "calibs", "depth", "size_3d",
                "heading_bin", "heading_res", "boxes_3d"]


@register_model("monodetr")
class MonoDETRBaseline(BaselineModel):
    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = {**DEFAULT_MODEL_CFG, **cfg.get("model_cfg", {})}
        if not torch.cuda.is_available():
            model_cfg["device"] = "cpu"
        self.model_cfg = model_cfg
        self.resolution = cfg.get("resolution", [960, 640])
        self.score_thresh = cfg.get("score_thresh", 0.2)
        self.model, self.criterion = build_monodetr(model_cfg)
        # mean_size for size-residual decode; filled in build_datasets.
        self.register_buffer("mean_size", torch.zeros(3))

    # ---- data ----------------------------------------------------------------
    def build_datasets(self, cfg: dict):
        records = load_records(cfg["index_file"])
        print(f"Detection records: {len(records)} images")
        train_weathers = cfg.get("train_weathers")
        if train_weathers:
            keep = set(train_weathers)
            records = [r for r in records if r.get("weather") in keep]
            print(f"Filtered to weathers {sorted(keep)}: {len(records)} records")

        mean = compute_mean_size(records)
        self.mean_size.copy_(torch.from_numpy(mean))

        # Split on segments, not records: every frame is re-rendered under 10
        # weathers and frames arrive at ~10 Hz, so a record split puts the same
        # scene on both sides and validation becomes fiction. Matches box3d.
        group_by = cfg.get("group_by", "segment")
        groups = [
            r["segment"] if group_by == "segment"
            else f"{r['segment']}|{r.get('camera')}|{r.get('stem')}"
            for r in records
        ]
        if group_by != "segment":
            print(f"WARNING: group_by={group_by!r} — validation is optimistic.")
        train_recs, val_recs = split_by_group(
            records, groups, cfg["val_split"], cfg["seed"]
        )
        train_recs = [records[i] for i in train_recs.indices]
        val_recs = [records[i] for i in val_recs.indices]
        make = lambda recs: DetectionDataset(recs, self.resolution, mean)
        return make(train_recs), make(val_recs)

    def collate_fn(self, batch):
        # Vendored backbone takes a plain (B,3,H,W) tensor and builds its own
        # (zero) mask; our images are uniform size so no padding is needed.
        images = torch.stack([b["image"] for b in batch])
        calibs = torch.stack([b["calib"] for b in batch])
        img_size = torch.stack([b["img_size"] for b in batch])
        targets = {"img_size": img_size}
        for k in ("labels", "boxes", "boxes_3d", "depth", "size_3d",
                  "heading_bin", "heading_res", "mask_2d"):
            targets[k] = torch.stack([b[k] for b in batch])
        # calibs broadcast per object so prepare_targets can index them
        targets["calibs"] = calibs[:, None, :, :].expand(-1, MAX_OBJS, -1, -1).clone()
        meta = {
            "weather": [b["weather"] for b in batch],
            "native_size": [b["native_size"] for b in batch],
            "gt_loc": [b["gt_loc"] for b in batch],
            "gt_box2d": [b["gt_box2d"] for b in batch],
        }
        return images, calibs, targets, meta

    @staticmethod
    def _prepare_targets(targets, device, batch_size):
        mask = targets["mask_2d"].to(device)
        out = []
        for bz in range(batch_size):
            m = mask[bz]
            out.append({k: targets[k][bz][m].to(device) for k in _TARGET_KEYS})
        return out

    # ---- train ---------------------------------------------------------------
    def training_step(self, batch, device):
        images, calibs, targets, meta = batch
        images = images.to(device)
        calibs = calibs.to(device)
        bs = calibs.shape[0]
        for k in targets:
            targets[k] = targets[k].to(device)
        img_sizes = targets["img_size"]
        targets_list = self._prepare_targets(targets, device, bs)

        outputs = self.model(images, calibs, targets_list, img_sizes)
        loss_dict = self.criterion(outputs, targets_list)
        wd = self.criterion.weight_dict
        loss = sum(loss_dict[k] * wd[k] for k in loss_dict if k in wd)

        logs = {"batch_size": bs}
        for k in ("loss_ce", "loss_bbox", "loss_center", "loss_depth", "loss_dim", "loss_angle"):
            if k in loss_dict:
                logs[k] = float(loss_dict[k])
        return loss, logs

    # ---- eval ----------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader, device) -> dict:
        self.eval()
        metric = CenterErrorMetric(iou_thr=0.5)
        res_w, res_h = self.resolution

        for images, calibs, targets, meta in loader:
            images = images.to(device)
            calibs_d = calibs.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            img_sizes = targets["img_size"]
            bs = calibs.shape[0]
            targets_list = self._prepare_targets(targets, device, bs)
            out = self.model(images, calibs_d, targets_list, img_sizes)

            scores = out["pred_logits"].sigmoid().max(-1).values  # (B,Q)
            boxes = out["pred_boxes"]                              # (B,Q,6) normalized
            depths = out["pred_depth"][..., 0]                    # (B,Q)

            for bi in range(bs):
                fu, fv = float(calibs[bi, 0, 0]), float(calibs[bi, 1, 1])
                cu, cv = float(calibs[bi, 0, 2]), float(calibs[bi, 1, 2])
                nw, nh = meta["native_size"][bi]
                keep = scores[bi] > self.score_thresh
                pb = boxes[bi][keep].cpu().numpy()
                pscore = scores[bi][keep].cpu().numpy()
                pz = depths[bi][keep].cpu().numpy()

                pred_box2d, pred_loc = [], []
                for (cx3d, cy3d, l, r, t, b), z in zip(pb, pz):
                    # 2D box (normalized -> native cxcywh)
                    x1, x2 = (cx3d - l) * nw, (cx3d + r) * nw
                    y1, y2 = (cy3d - t) * nh, (cy3d + b) * nh
                    pred_box2d.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
                    # 3D loc: backproject 3D-center pixel (resolution space) at depth z
                    u, v = cx3d * res_w, cy3d * res_h
                    X = (u - cu) / fu * z
                    Y = (v - cv) / fv * z
                    pred_loc.append([X, Y, z])

                metric.accumulate(Sample(
                    weather=meta["weather"][bi],
                    pred_box2d=np.asarray(pred_box2d, dtype=np.float32).reshape(-1, 4),
                    pred_loc=np.asarray(pred_loc, dtype=np.float32).reshape(-1, 3),
                    pred_score=pscore.astype(np.float32),
                    gt_box2d=meta["gt_box2d"][bi].reshape(-1, 4),
                    gt_loc=meta["gt_loc"][bi].reshape(-1, 3),
                ))

        result = metric.compute()
        return {
            "monitor": result["monitor"],
            "depth_mae": result["depth_mae"],
            "center_mae": result["center_mae"],
            "recall": result["recall"],
            "per_weather": result["per_weather"],
        }
