"""
Shared 3D box representation used by detection baselines and metrics.

Camera optical frame (the frame MonoDETR predicts in): x right, y down,
z forward. Depth is z. Dimensions follow the KITTI/MonoDETR convention (h, w, l).

This is the common currency between a model's decode() output and the metrics
module, so any baseline can be scored the same way. Fleshed out further in
milestone 2 (the Waymo -> camera-frame adapter).
"""

from dataclasses import dataclass


@dataclass
class Box3D:
    # 3D center in the camera optical frame (metres)
    x: float
    y: float
    z: float           # depth (forward)
    # dimensions (metres), KITTI/MonoDETR order
    h: float
    w: float
    l: float
    # observation/yaw angle in camera frame (radians)
    ry: float
    # class label (vehicle-only for now)
    cls: int = 1
    # detection confidence in [0, 1]; ground-truth boxes use 1.0
    score: float = 1.0

    @property
    def center(self):
        return (self.x, self.y, self.z)
