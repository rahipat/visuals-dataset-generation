"""
PyTorch Dataset for velocity prediction, built from velocity_records.jsonl.

Each sample is:
    crop_t:  float32 (3, 224, 224) — padded crop of object at frame T
    crop_t1: float32 (3, 224, 224) — padded crop of same object at frame T+1
    coords:  float32 (15,) — [cx_n_t, cy_n_t, sw_n_t, sh_n_t,
                               cx_n_t1, cy_n_t1, sw_n_t1, sh_n_t1,
                               ego_linear_x/y/z, ego_angular_x/y/z, delta_t]
    target:  float32 (2,)  — [vx, vy] in vehicle frame (m/s)
"""

import json
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.paths import to_posix
from data.robust_load import load_skipping_corrupt

COORD_KEYS = [
    "cx_n_t", "cy_n_t", "sw_n_t", "sh_n_t",
    "cx_n_t1", "cy_n_t1", "sw_n_t1", "sh_n_t1",
    "ego_linear_x", "ego_linear_y", "ego_linear_z",
    "ego_angular_x", "ego_angular_y", "ego_angular_z",
    "delta_t",
]
TARGET_KEYS = ["vx", "vy"]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_to_crop_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

CROP_PAD_FRAC = 0.10
CROP_MIN_PX   = 32


def _crop_box(
    width: int, height: int,
    cx_n: float, cy_n: float, sw_n: float, sh_n: float,
) -> Tuple[float, float, float, float]:
    cx_px = cx_n * width
    cy_px = cy_n * height
    bw_px = max(sw_n * width,  CROP_MIN_PX)
    bh_px = max(sh_n * height, CROP_MIN_PX)
    bw_pad = bw_px * (1.0 + CROP_PAD_FRAC)
    bh_pad = bh_px * (1.0 + CROP_PAD_FRAC)
    left   = max(0.0,           cx_px - bw_pad / 2.0)
    top    = max(0.0,           cy_px - bh_pad / 2.0)
    right  = min(float(width),  cx_px + bw_pad / 2.0)
    bottom = min(float(height), cy_px + bh_pad / 2.0)
    if right - left < CROP_MIN_PX:
        left  = max(0.0, cx_px - CROP_MIN_PX / 2.0)
        right = min(float(width), left + CROP_MIN_PX)
    if bottom - top < CROP_MIN_PX:
        top    = max(0.0, cy_px - CROP_MIN_PX / 2.0)
        bottom = min(float(height), top + CROP_MIN_PX)
    return left, top, right, bottom


class VelocityDataset(Dataset):
    def __init__(self, index_file: str):
        index_path = Path(index_file)
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found: {index_path}\n"
                "Run: python data/build_velocity_index.py"
            )
        self._records = []
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))
        self._bad_indices = set()

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        def build(i):
            r = self._records[i]
            img_t = Image.open(to_posix(r["image_t"]))
            img_t.load()  # force decode now so truncated/empty files raise here
            img_t1 = Image.open(to_posix(r["image_t1"]))
            img_t1.load()
            return r, img_t.convert("RGB"), img_t1.convert("RGB")

        _, (r, img_t, img_t1) = load_skipping_corrupt(
            len(self._records), idx, build, context="VelocityDataset",
            bad_indices=self._bad_indices)
        w, h = img_t.size

        crop_t  = _to_crop_tensor(img_t.crop(_crop_box(w, h, r["cx_n_t"],  r["cy_n_t"],  r["sw_n_t"],  r["sh_n_t"])))
        crop_t1 = _to_crop_tensor(img_t1.crop(_crop_box(w, h, r["cx_n_t1"], r["cy_n_t1"], r["sw_n_t1"], r["sh_n_t1"])))

        coords = torch.tensor([r[k] for k in COORD_KEYS], dtype=torch.float32)
        target = torch.tensor([r[k] for k in TARGET_KEYS], dtype=torch.float32)

        return crop_t, crop_t1, coords, target
