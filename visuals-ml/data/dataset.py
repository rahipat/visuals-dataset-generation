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

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.paths import to_posix
from data.robust_load import load_skipping_corrupt

INPUT_KEYS = ["cx_n", "cy_n", "sw_n", "sh_n", "fu", "fv", "cu", "cv"]
TARGET_KEYS = ["tx", "ty", "tz"]
# Column order of the packed numeric array: inputs first, then targets, so a
# single (N, 11) float32 buffer serves both slices without a copy.
_NUM_KEYS = INPUT_KEYS + TARGET_KEYS
_N_INPUTS = len(INPUT_KEYS)

# Records are parsed in chunks so the transient Python dicts from json.loads
# never all coexist -- otherwise peak RSS during __init__ alone would rival
# the steady-state footprint we're trying to eliminate.
_PARSE_CHUNK = 200_000

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

        # Records are stored as a few contiguous numpy buffers rather than a
        # list of per-record dicts. This is a memory fix, not a style choice:
        # DataLoader workers are forked, and CPython's refcounter writes to the
        # header of every object it touches, so copy-on-write pages backing
        # millions of dicts get privatised into each worker as it iterates --
        # RSS climbs steadily until the job is OOM-killed. Measured on this
        # dataset's records: ~1566 B/record as dicts (~5.3 GB per process at
        # 3.4M records, ~37 GB across a main process plus 6 workers) vs ~226
        # B/record packed (~0.77 GB, genuinely shared because numpy buffers
        # hold no per-element refcounts).
        nums_chunks, path_chunks, weather_chunks = [], [], []
        nums_buf, path_buf, weather_buf = [], [], []

        def _flush():
            if not nums_buf:
                return
            nums_chunks.append(np.asarray(nums_buf, dtype=np.float32))
            path_chunks.append(np.asarray(path_buf, dtype="S"))
            weather_chunks.append(np.asarray(weather_buf, dtype="S"))
            nums_buf.clear(); path_buf.clear(); weather_buf.clear()

        with open(index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                nums_buf.append([r[k] for k in _NUM_KEYS])
                # Normalise separators once here instead of per __getitem__.
                path_buf.append(to_posix(r["image_path"]))
                weather_buf.append(r.get("weather", "unknown"))
                if len(nums_buf) >= _PARSE_CHUNK:
                    _flush()
        _flush()

        if not nums_chunks:
            raise ValueError(f"Index is empty: {index_path}")

        # np.concatenate promotes the fixed-width byte dtypes to the widest
        # chunk's width, so per-chunk path lengths don't truncate.
        self._nums = np.concatenate(nums_chunks)
        self._paths = np.concatenate(path_chunks)
        self._weather = np.concatenate(weather_chunks)
        self._bad_indices = set()

    def __len__(self):
        return len(self._nums)

    def weather_at(self, i) -> str:
        """Weather variant for a resolved index (see _build_sample)."""
        return self._weather[i].decode("utf-8")

    def _build_sample(self, idx):
        """Returns (resolved_idx, image, crop, coords, target). The index is
        returned because a corrupt file makes the loader fall through to a
        later record, so callers needing per-sample metadata (e.g. weather)
        must key off the resolved index, not the requested one."""
        def build(i):
            img = Image.open(self._paths[i].decode("utf-8"))
            img.load()  # force decode now so truncated/empty files raise here
            return img.convert("RGB")

        i, img = load_skipping_corrupt(
            len(self._nums), idx, build, context="PositionDataset",
            bad_indices=self._bad_indices)
        width, height = img.size

        row = self._nums[i]
        crop_box = _crop_box_for(
            width, height,
            float(row[0]), float(row[1]), float(row[2]), float(row[3]),
        )
        crop_img = img.crop(crop_box)

        image = _to_full_tensor(img)
        crop = _to_crop_tensor(crop_img)

        # .copy() so the returned tensors don't keep a view alive on the
        # shared index buffer (and stay writable for downstream collation).
        coords = torch.from_numpy(row[:_N_INPUTS].copy())
        target = torch.from_numpy(row[_N_INPUTS:].copy())

        return i, image, crop, coords, target

    def __getitem__(self, idx):
        _, image, crop, coords, target = self._build_sample(idx)
        return image, crop, coords, target
