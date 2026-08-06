"""
Rotated 3D / BEV IoU in the camera optical frame (x right, y down, z forward).

Why this exists: the introspection pipeline needs a per-object scalar "was the
perception model wrong here". The existing label pre-pass uses 3D centre distance
in metres against a threshold `tau`, which has to be re-derived per depth range
(a 2 m error is nothing at 70 m and catastrophic at 5 m) — hence the
`--tau-percentile` workaround. 3D IoU is scale-free, bounded in [0, 1], and is
the standard detection currency, so it makes a much better-behaved failure label.

Conventions (matching baselines/core/boxes.Box3D and core/geometry):
  - BEV footprint lies in the (x, z) plane; `ry` rotates it about the y axis.
  - At ry = 0 the box's length `l` runs along x and its width `w` along z,
    per the KITTI convention the geometry adapter already targets.
  - `y` is the box CENTRE height, not the KITTI bottom-face convention:
    geometry.box_to_camera_frame maps y_opt = -z_cam directly from the LiDAR
    box centre. Height overlap is therefore computed on [y - h/2, y + h/2].

Both polygons are convex, so Sutherland-Hodgman clipping gives the exact
intersection area with no external dependency (shapely is not in the container).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def bev_corners(x: float, z: float, w: float, l: float, ry: float) -> np.ndarray:
    """The 4 BEV corners (x, z) of a box, counter-clockwise. See module docstring
    for the axis convention."""
    half_l, half_w = l / 2.0, w / 2.0
    local = np.array(
        [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]],
        dtype=np.float64,
    )
    c, s = np.cos(ry), np.sin(ry)
    # Rotation about the y (down) axis: x' = x c + z s, z' = -x s + z c
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + np.array([x, z], dtype=np.float64)


def _signed_area(poly: np.ndarray) -> float:
    xs, ys = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def polygon_area(poly: np.ndarray) -> float:
    return abs(_signed_area(poly))


def _as_ccw(poly: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman's inside test assumes a known winding; normalise to CCW."""
    return poly[::-1] if _signed_area(poly) < 0 else poly


def convex_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of one convex polygon by another. Returns the
    intersection vertices (possibly empty)."""
    subject, clip = _as_ccw(subject), _as_ccw(clip)
    output = list(subject)

    for i in range(len(clip)):
        if not output:
            return np.empty((0, 2))
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a

        def inside(p):
            # Left of the directed edge a->b (CCW winding => interior).
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -_EPS

        buf, prev = [], output[-1]
        prev_in = inside(prev)
        for cur in output:
            cur_in = inside(cur)
            if cur_in != prev_in:
                d = cur - prev
                denom = edge[0] * d[1] - edge[1] * d[0]
                if abs(denom) > _EPS:
                    t = (edge[0] * (prev[1] - a[1]) - edge[1] * (prev[0] - a[0])) / -denom
                    buf.append(prev + t * d)
            if cur_in:
                buf.append(cur)
            prev, prev_in = cur, cur_in
        output = buf

    return np.asarray(output) if output else np.empty((0, 2))


def bev_iou(pred, gt) -> float:
    """Bird's-eye-view IoU of two boxes, each (x, z, w, l, ry)."""
    px, pz, pw, pl, pry = pred
    gx, gz, gw, gl, gry = gt
    inter = polygon_area(
        convex_intersection(bev_corners(px, pz, pw, pl, pry),
                            bev_corners(gx, gz, gw, gl, gry))
    )
    union = pw * pl + gw * gl - inter
    return float(inter / union) if union > _EPS else 0.0


def iou_3d(pred, gt) -> float:
    """Rotated 3D IoU of two boxes, each (x, y, z, h, w, l, ry).

    `y` is the centre height (see module docstring)."""
    px, py, pz, ph, pw, pl, pry = pred
    gx, gy, gz, gh, gw, gl, gry = gt

    inter_bev = polygon_area(
        convex_intersection(bev_corners(px, pz, pw, pl, pry),
                            bev_corners(gx, gz, gw, gl, gry))
    )
    if inter_bev <= _EPS:
        return 0.0

    # Height overlap of the two centred intervals on y.
    top = max(py - ph / 2.0, gy - gh / 2.0)
    bot = min(py + ph / 2.0, gy + gh / 2.0)
    inter_h = max(0.0, bot - top)
    if inter_h <= _EPS:
        return 0.0

    inter_vol = inter_bev * inter_h
    union = ph * pw * pl + gh * gw * gl - inter_vol
    return float(inter_vol / union) if union > _EPS else 0.0


def iou_3d_batch(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Element-wise 3D IoU for paired arrays of shape (N, 7)."""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    return np.array([iou_3d(p, g) for p, g in zip(pred, gt)], dtype=np.float64)
