"""
Detection metrics — Option A: per-weather 3D center / depth error.

This is the porting instrument: interpretable distances in metres that tell you
whether the geometry and decode are right, stratified by weather variant. (Full
3D/BEV AP — Option B — is added later in metrics, once the model trains.)

A prediction is matched to a ground-truth object by 2D-box IoU (greedy, highest
score first). For matched pairs we accumulate absolute error on the 3D center
(camera optical frame) and on depth (z). We also report recall@IoU so a model
can't look good by detecting one easy object and missing the rest.

Usage: feed one Sample per image via accumulate(), then call compute().
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import numpy as np


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def match_by_iou(pred_xyxy, gt_xyxy, scores, iou_thr=0.5):
    """Greedy match predictions to GT by IoU, highest score first.
    Returns list of (pred_idx, gt_idx)."""
    order = sorted(range(len(pred_xyxy)), key=lambda i: -scores[i])
    used_gt = set()
    matches = []
    for pi in order:
        best_iou, best_gi = iou_thr, -1
        for gi in range(len(gt_xyxy)):
            if gi in used_gt:
                continue
            v = _iou(pred_xyxy[pi], gt_xyxy[gi])
            if v >= best_iou:
                best_iou, best_gi = v, gi
        if best_gi >= 0:
            used_gt.add(best_gi)
            matches.append((pi, best_gi))
    return matches


@dataclass
class Sample:
    weather: str
    pred_box2d: np.ndarray   # (N,4) cxcywh, pixels
    pred_loc: np.ndarray     # (N,3) camera optical frame
    pred_score: np.ndarray   # (N,)
    gt_box2d: np.ndarray     # (M,4) cxcywh
    gt_loc: np.ndarray       # (M,3)


@dataclass
class _Acc:
    center_abs: np.ndarray = field(default_factory=lambda: np.zeros(3))
    depth_abs: float = 0.0
    n_matched: int = 0
    n_gt: int = 0


class CenterErrorMetric:
    def __init__(self, iou_thr=0.5):
        self.iou_thr = iou_thr
        self._per_weather = defaultdict(_Acc)

    def accumulate(self, s: Sample):
        acc = self._per_weather[s.weather]
        acc.n_gt += len(s.gt_loc)
        if len(s.pred_box2d) == 0 or len(s.gt_box2d) == 0:
            return
        pred_xyxy = [cxcywh_to_xyxy(b) for b in s.pred_box2d]
        gt_xyxy = [cxcywh_to_xyxy(b) for b in s.gt_box2d]
        for pi, gi in match_by_iou(pred_xyxy, gt_xyxy, s.pred_score, self.iou_thr):
            err = np.abs(s.pred_loc[pi] - s.gt_loc[gi])
            acc.center_abs += err
            acc.depth_abs += err[2]
            acc.n_matched += 1

    @staticmethod
    def _summarize(acc: _Acc) -> dict:
        n = max(acc.n_matched, 1)
        center = acc.center_abs / n
        return {
            "n_gt": acc.n_gt,
            "n_matched": acc.n_matched,
            "recall": acc.n_matched / max(acc.n_gt, 1),
            "depth_mae": float(acc.depth_abs / n),
            "center_mae": float(center.mean()),
            "mae_x": float(center[0]),
            "mae_y": float(center[1]),
            "mae_z": float(center[2]),
        }

    def compute(self) -> dict:
        total = _Acc()
        per_weather = {}
        for w in sorted(self._per_weather):
            acc = self._per_weather[w]
            per_weather[w] = self._summarize(acc)
            total.center_abs += acc.center_abs
            total.depth_abs += acc.depth_abs
            total.n_matched += acc.n_matched
            total.n_gt += acc.n_gt
        overall = self._summarize(total)
        # monitor: depth error among matched, penalized by misses (lower is better).
        #
        # A detector that predicts NOTHING has n_matched == 0, hence depth_mae == 0
        # (an empty mean) and recall == 0. Dividing those gives monitor == 0.0 --
        # the best possible score -- so the runner would checkpoint the degenerate
        # model and never improve on it. An empty prediction set is the worst
        # outcome, not the best, so it must map to +inf.
        if overall["n_matched"] == 0:
            overall["monitor"] = float("inf")
        else:
            overall["monitor"] = overall["depth_mae"] / overall["recall"]
        overall["per_weather"] = per_weather
        return overall
