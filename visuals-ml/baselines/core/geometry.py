"""
Waymo -> camera-frame geometry for monocular 3D detection labels.

The conventions here mirror visuals_dataset/lidar_camera_association.py (the
pipeline code that produced the `projected_lidar_boxes` already in the metadata),
so our targets are consistent with what generated the dataset:

  - Waymo camera frame: x forward, y left, z up.  `extrinsic_transform` is
    vehicle_from_camera; camera_from_vehicle is its inverse.
  - A vehicle-frame point projects with depth = camera-frame x, and
        u_n = -y_cam / x_cam,  v_n = -z_cam / x_cam
    followed by radial+tangential distortion and the pinhole intrinsics.

MonoDETR works in the OPTICAL frame (x right, y down, z forward, depth = z), so
box_to_camera_frame() remaps Waymo-camera -> optical:
        x_opt = -y_cam,  y_opt = -z_cam,  z_opt = x_cam (depth)

Box3D dimensions follow KITTI/MonoDETR order (h, w, l); Waymo size is
(x=length, y=width, z=height), so h=size_z, w=size_y, l=size_x.
"""

from __future__ import annotations

import math

import numpy as np

from baselines.core.boxes import Box3D

# Waymo object type codes
TYPE_VEHICLE = 1


def projection_params(camera_calibration: dict) -> dict | None:
    """Build projection params from an image_metadata `camera_calibration` block
    (nested `intrinsic` dict + `extrinsic_transform` 16-list)."""
    intr = camera_calibration.get("intrinsic", {})
    fu, fv = intr.get("f_u"), intr.get("f_v")
    cu, cv = intr.get("c_u"), intr.get("c_v")
    transform = camera_calibration.get("extrinsic_transform")
    if None in (fu, fv, cu, cv) or not transform or len(transform) != 16:
        return None
    try:
        vehicle_from_camera = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        camera_from_vehicle = np.linalg.inv(vehicle_from_camera)
    except Exception:
        return None
    return {
        "fu": fu, "fv": fv, "cu": cu, "cv": cv,
        "k1": intr.get("k1", 0.0) or 0.0,
        "k2": intr.get("k2", 0.0) or 0.0,
        "p1": intr.get("p1", 0.0) or 0.0,
        "p2": intr.get("p2", 0.0) or 0.0,
        "k3": intr.get("k3", 0.0) or 0.0,
        "width": camera_calibration.get("width"),
        "height": camera_calibration.get("height"),
        "vehicle_from_camera": vehicle_from_camera,
        "camera_from_vehicle": camera_from_vehicle,
    }


def _box_corners_vehicle(lb: dict) -> np.ndarray | None:
    """8 corners of a LiDARBox (flat row form) in the vehicle frame."""
    cx = lb.get("[LiDARBoxComponent].box.center.x")
    cy = lb.get("[LiDARBoxComponent].box.center.y")
    cz = lb.get("[LiDARBoxComponent].box.center.z")
    sx = lb.get("[LiDARBoxComponent].box.size.x")
    sy = lb.get("[LiDARBoxComponent].box.size.y")
    sz = lb.get("[LiDARBoxComponent].box.size.z")
    heading = lb.get("[LiDARBoxComponent].box.heading")
    if None in (cx, cy, cz, sx, sy, sz, heading) or min(sx, sy, sz) <= 0:
        return None
    lx, ly, lz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    local = np.array([
        [lx, ly, -lz], [-lx, ly, -lz], [-lx, -ly, -lz], [lx, -ly, -lz],
        [lx, ly, lz], [-lx, ly, lz], [-lx, -ly, lz], [lx, -ly, lz],
    ], dtype=np.float64)
    c, s = math.cos(heading), math.sin(heading)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    center = np.array([cx, cy, cz], dtype=np.float64)
    return local @ rot.T + center


def vehicle_to_camera(points_vehicle: np.ndarray, params: dict) -> np.ndarray:
    """Transform vehicle-frame points to the Waymo camera frame (x fwd, y left, z up)."""
    n = points_vehicle.shape[0]
    ph = np.concatenate([points_vehicle, np.ones((n, 1))], axis=1)
    return (params["camera_from_vehicle"] @ ph.T).T[:, :3]


def project_camera_points(points_cam: np.ndarray, params: dict) -> np.ndarray:
    """Project Waymo-camera-frame points to pixels (with distortion). Returns
    (N,2) uv for points in front of the camera (x>0); others dropped."""
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    valid = x > 1e-6
    u_n = -y[valid] / x[valid]
    v_n = -z[valid] / x[valid]
    r2 = u_n * u_n + v_n * v_n
    radial = 1.0 + params["k1"] * r2 + params["k2"] * r2**2 + params["k3"] * r2**3
    u_nd = u_n * radial + 2.0 * params["p1"] * u_n * v_n + params["p2"] * (r2 + 2.0 * u_n**2)
    v_nd = v_n * radial + params["p1"] * (r2 + 2.0 * v_n**2) + 2.0 * params["p2"] * u_n * v_n
    u = params["fu"] * u_nd + params["cu"]
    v = params["fv"] * v_nd + params["cv"]
    return np.stack([u, v], axis=1)


def projected_2d_box(lb: dict, params: dict) -> tuple | None:
    """Reproduce the pipeline's projected 2D box (x1,y1,x2,y2) from the 8 corners.
    Used to self-check our projection against metadata `projected_lidar_boxes`."""
    corners = _box_corners_vehicle(lb)
    if corners is None:
        return None
    cam = vehicle_to_camera(corners, params)
    uv = project_camera_points(cam, params)
    uv = uv[np.all(np.isfinite(uv), axis=1)]
    if uv.size == 0:
        return None
    x1, y1 = float(uv[:, 0].min()), float(uv[:, 1].min())
    x2, y2 = float(uv[:, 0].max()), float(uv[:, 1].max())
    w, h = params.get("width"), params.get("height")
    if w and h:
        x1, x2 = max(0.0, min(x1, w - 1)), max(0.0, min(x2, w - 1))
        y1, y2 = max(0.0, min(y1, h - 1)), max(0.0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _camera_yaw(params: dict) -> float:
    """Yaw of the camera's forward (x) axis in the vehicle frame."""
    R = params["vehicle_from_camera"][:3, :3]
    return math.atan2(R[1, 0], R[0, 0])


def box_to_camera_frame(lb: dict, params: dict) -> Box3D | None:
    """Convert a LiDARBox (vehicle frame) to a Box3D in the camera OPTICAL frame.
    Returns None if the box is behind the camera or missing fields."""
    cx = lb.get("[LiDARBoxComponent].box.center.x")
    cy = lb.get("[LiDARBoxComponent].box.center.y")
    cz = lb.get("[LiDARBoxComponent].box.center.z")
    sx = lb.get("[LiDARBoxComponent].box.size.x")
    sy = lb.get("[LiDARBoxComponent].box.size.y")
    sz = lb.get("[LiDARBoxComponent].box.size.z")
    heading = lb.get("[LiDARBoxComponent].box.heading")
    if None in (cx, cy, cz, sx, sy, sz, heading) or min(sx, sy, sz) <= 0:
        return None

    center_cam = vehicle_to_camera(np.array([[cx, cy, cz]], dtype=np.float64), params)[0]
    x_cam, y_cam, z_cam = center_cam
    if x_cam <= 1e-6:  # behind camera
        return None

    # Waymo camera -> optical frame
    x_opt, y_opt, z_opt = -y_cam, -z_cam, x_cam
    # object yaw relative to camera forward axis, in optical (KITTI ry) convention
    ry = -(heading - _camera_yaw(params)) - math.pi / 2.0
    ry = math.atan2(math.sin(ry), math.cos(ry))  # wrap to [-pi, pi]

    return Box3D(
        x=float(x_opt), y=float(y_opt), z=float(z_opt),
        h=float(sz), w=float(sy), l=float(sx),
        ry=float(ry), cls=TYPE_VEHICLE, score=1.0,
    )


def projected_center_uv(lb: dict, params: dict) -> tuple | None:
    """Pixel location of the 3D box center (for sanity overlays)."""
    cx = lb.get("[LiDARBoxComponent].box.center.x")
    cy = lb.get("[LiDARBoxComponent].box.center.y")
    cz = lb.get("[LiDARBoxComponent].box.center.z")
    if None in (cx, cy, cz):
        return None
    cam = vehicle_to_camera(np.array([[cx, cy, cz]], dtype=np.float64), params)
    uv = project_camera_points(cam, params)
    if uv.size == 0:
        return None
    return float(uv[0, 0]), float(uv[0, 1])
