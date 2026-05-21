"""
PyTorch Dataset that streams from a records.jsonl index built by build_index.py.

Each sample is:
    image:  float32 tensor of shape (3, 640, 960)  — full image (resized), normalised
    crop:   float32 tensor of shape (3, 224, 224)  — padded box crop from native-res image, normalised
    coords: float32 tensor of shape (8,)           — [cx_n, cy_n, sw_n, sh_n, fu, fv, cu, cv]
    target: float32 tensor of shape (3,)           — [tx, ty, tz] in ego-vehicle frame (metres)
"""

import json
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

INPUT_KEYS = ["cx_n", "cy_n", "sw_n", "sh_n", "fu", "fv", "cu", "cv"]
TARGET_KEYS = ["tx", "ty", "tz"]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_to_full_tensor = transforms.Compose([
    transforms.Resize((640, 960)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

_to_crop_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

CROP_PAD_FRAC = 0.10   # expand box by 10% on each side for context
CROP_MIN_PX = 32       # minimum crop edge length before resize


def _crop_box_for(
    width: int, height: int,
    cx_n: float, cy_n: float, sw_n: float, sh_n: float,
) -> Tuple[float, float, float, float]:
    """Compute padded crop bounds (l, t, r, b) clamped to the image."""
    cx_px = cx_n * width
    cy_px = cy_n * height
    bw_px = max(sw_n * width, CROP_MIN_PX)
    bh_px = max(sh_n * height, CROP_MIN_PX)
    bw_pad = bw_px * (1.0 + CROP_PAD_FRAC)
    bh_pad = bh_px * (1.0 + CROP_PAD_FRAC)
    left = max(0.0, cx_px - bw_pad / 2.0)
    top = max(0.0, cy_px - bh_pad / 2.0)
    right = min(float(width), cx_px + bw_pad / 2.0)
    bottom = min(float(height), cy_px + bh_pad / 2.0)
    # Guard against degenerate crops (box at edge + tiny size)
    if right - left < CROP_MIN_PX:
        left = max(0.0, cx_px - CROP_MIN_PX / 2.0)
        right = min(float(width), left + CROP_MIN_PX)
    if bottom - top < CROP_MIN_PX:
        top = max(0.0, cy_px - CROP_MIN_PX / 2.0)
        bottom = min(float(height), top + CROP_MIN_PX)
    return left, top, right, bottom


class PositionDataset(Dataset):
    def __init__(self, index_file: str):
        index_path = Path(index_file)
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found: {index_path}\n"
                "Run: python data/build_index.py"
            )

        self._records = []
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        r = self._records[idx]

        img = Image.open(r["image_path"]).convert("RGB")
        width, height = img.size

        crop_box = _crop_box_for(
            width, height,
            r["cx_n"], r["cy_n"], r["sw_n"], r["sh_n"],
        )
        crop_img = img.crop(crop_box)

        image = _to_full_tensor(img)
        crop = _to_crop_tensor(crop_img)

        coords = torch.tensor([r[k] for k in INPUT_KEYS], dtype=torch.float32)
        target = torch.tensor([r[k] for k in TARGET_KEYS], dtype=torch.float32)

        return image, crop, coords, target
