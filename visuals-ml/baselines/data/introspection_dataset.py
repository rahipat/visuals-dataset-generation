"""
Dataset for the Introspective Perception baseline (Paper 1).

Emits, per (frame, weather) record from introspection_label_prepass.py:

  spatial   (3, S, S)          RGB frame, ImageNet-normalized      -> spatial ConvNet
  flow      (2*L, S, S)        stacked optical-flow (u,v) fields   -> temporal ConvNet
  fail_frac float               continuous target (SVM regression)
  fail      long 0/1            binary target (CNN softmax loss)
  mean_err  float               underlying PositionNet 3D error (m; EFR curve)
  weather   str                 variant tag (per-weather metrics)

The temporal stream follows Simonyan & Zisserman's two-stream recipe used by the
paper: L optical-flow fields between the L+1 consecutive frames in `flow_frames`
are stacked channel-wise. Flow is TV-L1 when opencv-contrib is available, else
Farneback. Displacements are clipped to +/-FLOW_BOUND px and scaled to [-1, 1].

Per-frame flow stacks are cached to --flow-cache as .npy (keyed by the target
image path) so multi-epoch training does not recompute TV-L1 every epoch. On
HiPerGator, pre-warm the cache once (any single pass fills it).

NOTE (flagged for review): weather augmentations are applied per-frame
independently, so flow across augmented frames can mix real ego-motion with
augmentation flicker. This is a known caveat of running the temporal stream on
this dataset; see baselines/README (Introspection) and the project notes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.paths import to_posix, to_posix_all

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
FLOW_BOUND = 20.0   # px; clip displacements to +/-BOUND then scale to [-1, 1]


def _make_flow_engine():
    """TV-L1 (opencv-contrib) if available, else Farneback."""
    if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        return lambda a, b: tvl1.calc(a, b, None)
    if hasattr(cv2, "DualTVL1OpticalFlow_create"):
        tvl1 = cv2.DualTVL1OpticalFlow_create()
        return lambda a, b: tvl1.calc(a, b, None)
    return lambda a, b: cv2.calcOpticalFlowFarneback(
        a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)


class IntrospectionDataset(Dataset):
    def __init__(self, records, resolution=224, flow_cache=None):
        self.records = records
        self.res = int(resolution)
        self.flow_cache = Path(flow_cache) if flow_cache else None
        if self.flow_cache:
            self.flow_cache.mkdir(parents=True, exist_ok=True)
        self._flow = None  # lazily built per worker (cv2 objects aren't picklable)
        self._spatial_tf = transforms.Compose([
            transforms.Resize((self.res, self.res)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.records)

    # ---- optical flow --------------------------------------------------------
    def _gray(self, path):
        img = Image.open(path).convert("L").resize((self.res, self.res))
        return np.asarray(img, dtype=np.uint8)

    def _cache_path(self, target_path):
        key = hashlib.md5(target_path.encode("utf-8")).hexdigest()
        return self.flow_cache / f"{key}_{self.res}.npy"

    def _flow_stack(self, frames):
        """(2*L, S, S) float32 in [-1, 1] for the L consecutive pairs in frames."""
        target = frames[-1]
        if self.flow_cache:
            cpath = self._cache_path(target)
            if cpath.exists():
                return np.load(cpath)

        if self._flow is None:
            self._flow = _make_flow_engine()

        grays = [self._gray(p) for p in frames]
        chans = []
        for a, b in zip(grays[:-1], grays[1:]):
            f = self._flow(a, b)                       # (S, S, 2) px displacement
            f = np.clip(f, -FLOW_BOUND, FLOW_BOUND) / FLOW_BOUND
            chans.append(f[..., 0])
            chans.append(f[..., 1])
        stack = np.stack(chans, axis=0).astype(np.float32)   # (2L, S, S)

        if self.flow_cache:
            np.save(self._cache_path(target), stack)
        return stack

    # ---- item ----------------------------------------------------------------
    def __getitem__(self, idx):
        r = self.records[idx]
        spatial = self._spatial_tf(Image.open(to_posix(r["image_path"])).convert("RGB"))
        flow = torch.from_numpy(self._flow_stack(to_posix_all(r["flow_frames"])))
        fail_frac = torch.tensor(r["fail_frac"], dtype=torch.float32)
        fail = torch.tensor(int(r["fail"]), dtype=torch.long)
        mean_err = torch.tensor(r.get("mean_err", 0.0), dtype=torch.float32)
        return spatial, flow, fail_frac, fail, mean_err, r.get("weather", "unknown")


def load_labeled_records(index_file: str):
    path = Path(index_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Labeled introspection index not found: {path}\n"
            "Run build_introspection_index then introspection_label_prepass."
        )
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if recs and "fail_frac" not in recs[0]:
        raise KeyError(
            f"{path} has no 'fail_frac' — run introspection_label_prepass.py "
            "on the model-free index first."
        )
    return recs
