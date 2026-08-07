"""
Per-object dataset for the amodal 3D box baseline (Baseline B).

Reads the SAME per-image index as MonoDETR (build_detection_index.py) and flattens
it to one sample per (object, weather). Sharing the index means B and the detector
are trained and scored against byte-identical labels, so the comparison between
"given the 2D box, how well can you lift it to 3D" (B) and "find the boxes too"
(MonoDETR) is clean.

Sample: (full image, object crop, coord vector, target vector, meta dict).

Target encoding (8-d) — all standard mono-3D practice, nothing novel:
    t0, t1 : x, y            centre in the camera optical frame, metres
    t2     : log(z)          depth in log space: z spans ~2-80 m here, and a
                             plain L1/L2 on metres lets far objects dominate the
                             gradient while near ones (where error matters most)
                             are ignored. Log makes the loss relative.
    t3..t5 : log(dim / anchor)   dimension residual against the fleet-mean vehicle;
                             vehicle sizes are tightly clustered, so predicting a
                             residual is far easier than absolute metres.
    t6, t7 : sin(ry), cos(ry)    angles are circular; regressing the raw radian
                             puts a discontinuity at +/-pi. Decoded with atan2.

decode_targets() inverts this exactly, so eval works in real metres/radians and
feeds baselines.core.iou3d directly.

NOTE on the coord vector: intrinsics are normalised by image size (fu/W, cu/W,
...) rather than passed raw. The original PositionNet fed raw pixels (f_u ~ 2060)
into an MLP alongside inputs in [0, 1]; that scale mismatch dominates the first
layer's gradients for no reason.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.dataset import _crop_box_for

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Fallback anchor, overridden by anchors computed from the training index.
# (h, w, l) in the KITTI order the geometry adapter emits.
DEFAULT_DIM_ANCHOR = (2.03, 2.06, 4.56)

FULL_SIZE = (640, 960)   # (H, W) — matches PositionNet's full-image stream
CROP_SIZE = (224, 224)

_to_full = transforms.Compose([
    transforms.Resize(FULL_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])
_to_crop = transforms.Compose([
    transforms.Resize(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


def load_records(index_file: str):
    path = Path(index_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Detection index not found: {path}\n"
            "Run: python -m baselines.data.build_detection_index"
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_dim_anchor(records) -> np.ndarray:
    """Mean (h, w, l) over every object in `records`."""
    dims = [o["dim"] for r in records for o in r["objects"]]
    if not dims:
        return np.asarray(DEFAULT_DIM_ANCHOR, dtype=np.float32)
    return np.asarray(dims, dtype=np.float32).mean(axis=0)


def decode_targets(t: torch.Tensor, dim_anchor: torch.Tensor) -> torch.Tensor:
    """(B, 8) encoded -> (B, 7) as [x, y, z, h, w, l, ry] in metres/radians.

    Inverts the encoding in the module docstring. sin/cos are fed to atan2, which
    normalises implicitly, so the head is not required to emit a unit vector.
    """
    x, y = t[:, 0], t[:, 1]
    z = torch.exp(t[:, 2].clamp(max=6.0))          # clamp: exp overflow guard
    dims = torch.exp(t[:, 3:6].clamp(-3.0, 3.0)) * dim_anchor.to(t.device)
    ry = torch.atan2(t[:, 6], t[:, 7])
    return torch.stack([x, y, z, dims[:, 0], dims[:, 1], dims[:, 2], ry], dim=1)


class Box3DDataset(Dataset):
    """One sample per (object, weather variant)."""

    def __init__(self, records, dim_anchor=None, min_box_px=0.0, min_pts=0,
                 drop_truncated=True):
        self.records = records
        self.dim_anchor = np.asarray(
            DEFAULT_DIM_ANCHOR if dim_anchor is None else dim_anchor, dtype=np.float32
        )

        # Flatten to (record_idx, object_idx), applying quality filters once.
        # Two grouping keys are kept so the split granularity is selectable:
        #   segment — the correct one. Removes weather duplication AND the
        #             temporal correlation between neighbouring 10 Hz frames.
        #   frame   — weaker. Removes only the 10x weather duplication; adjacent
        #             frames of the same track still straddle the split, so val
        #             is optimistic. For smoke runs on a 1-2 segment sample.
        self.pairs, self.groups, self.groups_frame = [], [], []
        n_drop_geom = n_drop_size = n_drop_pts = n_drop_trunc = 0
        for ri, r in enumerate(records):
            for oi, o in enumerate(r["objects"]):
                _x, _y, z = o["loc"]
                h, w, l = o["dim"]
                if z <= 1e-3 or min(h, w, l) <= 1e-3:
                    n_drop_geom += 1
                    continue
                # A border-clipped 2D box no longer centres on the projected 3D
                # centre, so it teaches a corrupted box->3D mapping. See
                # build_detection_index._is_truncated for the measurements.
                if drop_truncated and o.get("truncated"):
                    n_drop_trunc += 1
                    continue
                _cx, _cy, sw, sh = o["box_2d"]
                if min(sw, sh) < min_box_px:
                    n_drop_size += 1
                    continue
                if min_pts and (o.get("n_pts") or 0) < min_pts:
                    n_drop_pts += 1
                    continue
                self.pairs.append((ri, oi))
                self.groups.append(r["segment"])
                self.groups_frame.append(
                    f"{r['segment']}|{r.get('camera')}|{r.get('stem')}"
                )

        dropped = n_drop_geom + n_drop_size + n_drop_pts + n_drop_trunc
        if dropped:
            print(f"Box3DDataset: dropped {dropped} objects "
                  f"(geometry {n_drop_geom}, truncated {n_drop_trunc}, "
                  f"box<{min_box_px}px {n_drop_size}, pts<{min_pts} {n_drop_pts})")

    def groups_for(self, kind: str):
        """Grouping keys for split_by_group. 'segment' (default) or 'frame'."""
        if kind == "segment":
            return self.groups
        if kind == "frame":
            return self.groups_frame
        raise ValueError(f"group_by must be 'segment' or 'frame', got {kind!r}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ri, oi = self.pairs[idx]
        r = self.records[ri]
        o = r["objects"][oi]
        native_w, native_h = r["image_size"]

        img = Image.open(r["image_path"]).convert("RGB")
        cx, cy, sw, sh = o["box_2d"]
        cx_n, cy_n = cx / native_w, cy / native_h
        sw_n, sh_n = sw / native_w, sh / native_h

        crop_box = _crop_box_for(native_w, native_h, cx_n, cy_n, sw_n, sh_n)
        crop = _to_crop(img.crop(crop_box))
        full = _to_full(img)

        intr = r["intrinsic"]
        coords = torch.tensor([
            cx_n, cy_n, sw_n, sh_n,
            intr["f_u"] / native_w, intr["f_v"] / native_h,
            intr["c_u"] / native_w, intr["c_v"] / native_h,
        ], dtype=torch.float32)

        x, y, z = o["loc"]
        h, w, l = o["dim"]
        ry = o["ry"]
        a = self.dim_anchor
        target = torch.tensor([
            x, y, math.log(z),
            math.log(h / a[0]), math.log(w / a[1]), math.log(l / a[2]),
            math.sin(ry), math.cos(ry),
        ], dtype=torch.float32)

        gt = torch.tensor([x, y, z, h, w, l, ry], dtype=torch.float32)
        meta = {
            "weather": r.get("weather", "unknown"),
            "depth": float(z),
            "n_pts": int(o.get("n_pts") or 0),
        }
        return full, crop, coords, target, gt, meta
