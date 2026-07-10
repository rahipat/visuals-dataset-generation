"""
Per-image detection dataset that emits MonoDETR's exact target encoding.

Reads the per-image index from build_detection_index.py and produces, per sample,
the padded target dict MonoDETR's SetCriterion expects, using our validated
geometry (camera optical frame). No augmentation — deterministic, so the
overfit-one-batch wiring test is clean.

Target encoding (mirrors vendored lib/datasets/kitti/kitti_dataset.py), all
normalized to the network input resolution [W, H]:
  labels       (max_objs,)        class id (vehicle -> 0)
  boxes        (max_objs, 4)       2D bbox center+size, cxcywh, normalized
  boxes_3d     (max_objs, 6)       [proj_3d_center_x, _y, l, r, t, b], normalized
  depth        (max_objs, 1)       metric depth z (m)
  size_3d      (max_objs, 3)       [h, w, l] - class_mean_size
  heading_bin  (max_objs, 1)       observation-angle (alpha) bin in [0, 12)
  heading_res  (max_objs, 1)       residual within bin
  mask_2d      (max_objs,)         1 for valid objects
Plus per-image: image tensor, calib P (3x4) scaled to resolution, img_size,
weather. calib and img_size are kept in resolution space so depth_geo (which uses
calib fx and box-height in pixels) is internally consistent.
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

# vendored angle encoder
from baselines.data._vendor_path import ensure_vendor_on_path
ensure_vendor_on_path()
from lib.datasets.utils import angle2class  # noqa: E402

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
MAX_OBJS = 50
VEHICLE_CLS_ID = 0  # single-class (vehicle) -> 0


def compute_mean_size(records) -> np.ndarray:
    """Mean [h, w, l] over all objects, for size residual encoding."""
    dims = [o["dim"] for r in records for o in r["objects"]]
    if not dims:
        return np.array([1.6, 2.0, 4.5], dtype=np.float32)  # rough vehicle prior
    return np.asarray(dims, dtype=np.float32).mean(axis=0)


class DetectionDataset(Dataset):
    def __init__(self, records, resolution, mean_size, clip_2d=True):
        self.records = records
        self.res_w, self.res_h = int(resolution[0]), int(resolution[1])
        self.mean_size = np.asarray(mean_size, dtype=np.float32)
        self.clip_2d = clip_2d
        self._to_tensor = transforms.Compose([
            transforms.Resize((self.res_h, self.res_w)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.records)

    def _calib(self, intr, native_w, native_h):
        sx, sy = self.res_w / native_w, self.res_h / native_h
        fu, fv = intr["f_u"] * sx, intr["f_v"] * sy
        cu, cv = intr["c_u"] * sx, intr["c_v"] * sy
        P = np.array([[fu, 0, cu, 0], [0, fv, cv, 0], [0, 0, 1, 0]], dtype=np.float32)
        return P, (fu, fv, cu, cv)

    def __getitem__(self, idx):
        r = self.records[idx]
        native_w, native_h = r["image_size"]
        img = Image.open(r["image_path"]).convert("RGB")
        image = self._to_tensor(img)

        P, (fu, fv, cu, cv) = self._calib(r["intrinsic"], native_w, native_h)
        sx, sy = self.res_w / native_w, self.res_h / native_h

        labels = np.zeros(MAX_OBJS, dtype=np.int64)
        boxes = np.zeros((MAX_OBJS, 4), dtype=np.float32)
        boxes_3d = np.zeros((MAX_OBJS, 6), dtype=np.float32)
        depth = np.zeros((MAX_OBJS, 1), dtype=np.float32)
        size_3d = np.zeros((MAX_OBJS, 3), dtype=np.float32)
        heading_bin = np.zeros((MAX_OBJS, 1), dtype=np.int64)
        heading_res = np.zeros((MAX_OBJS, 1), dtype=np.float32)
        mask_2d = np.zeros(MAX_OBJS, dtype=np.bool_)

        n = 0
        for o in r["objects"]:
            if n >= MAX_OBJS:
                break
            cx, cy, w, h = o["box_2d"]                 # native pixels
            x, y, z = o["loc"]                         # optical frame, metres
            if z <= 1e-3:
                continue
            # projected 3D center in resolution pixels (pinhole, scaled intrinsics)
            u = fu * (x / z) + cu
            v = fv * (y / z) + cv
            if not (0 <= u < self.res_w and 0 <= v < self.res_h):
                continue

            # normalized (resolution-space == native-space for [0,1] fractions)
            cx3d_n, cy3d_n = u / self.res_w, v / self.res_h
            x1_n = (cx - w / 2) / native_w
            x2_n = (cx + w / 2) / native_w
            y1_n = (cy - h / 2) / native_h
            y2_n = (cy + h / 2) / native_h
            l, rr = cx3d_n - x1_n, x2_n - cx3d_n
            t, bb = cy3d_n - y1_n, y2_n - cy3d_n
            if min(l, rr, t, bb) < 0:
                if self.clip_2d:
                    l, rr = max(l, 0.0), max(rr, 0.0)
                    t, bb = max(t, 0.0), max(bb, 0.0)
                else:
                    continue

            labels[n] = VEHICLE_CLS_ID
            boxes[n] = [cx / native_w, cy / native_h, w / native_w, h / native_h]
            boxes_3d[n] = [cx3d_n, cy3d_n, l, rr, t, bb]
            depth[n, 0] = z
            size_3d[n] = np.asarray(o["dim"], dtype=np.float32) - self.mean_size
            alpha = o["ry"] - math.atan2(x, z)
            alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
            cls_bin, res = angle2class(alpha)
            heading_bin[n, 0] = cls_bin
            heading_res[n, 0] = res
            mask_2d[n] = True
            n += 1

        return {
            "image": image,
            "calib": torch.from_numpy(P),
            "img_size": torch.tensor([self.res_w, self.res_h], dtype=torch.float32),
            "labels": torch.from_numpy(labels),
            "boxes": torch.from_numpy(boxes),
            "boxes_3d": torch.from_numpy(boxes_3d),
            "depth": torch.from_numpy(depth),
            "size_3d": torch.from_numpy(size_3d),
            "heading_bin": torch.from_numpy(heading_bin),
            "heading_res": torch.from_numpy(heading_res),
            "mask_2d": torch.from_numpy(mask_2d),
            "weather": r.get("weather", "unknown"),
            "native_size": (native_w, native_h),
            # raw GT for the metric (native pixels / metres), eval-only
            "gt_loc": np.asarray([o["loc"] for o in r["objects"]], dtype=np.float32),
            "gt_box2d": np.asarray([o["box_2d"] for o in r["objects"]], dtype=np.float32),
        }


def load_records(index_file: str):
    path = Path(index_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Detection index not found: {path}\n"
            "Run: python -m baselines.data.build_detection_index"
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
