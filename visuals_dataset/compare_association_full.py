"""
Full-segment association comparison.

Reproduces our pipeline's LiDAR-camera association for EVERY frame of a segment by
driving the pipeline's own `enrich_camera_objects_with_lidar` (the exact function
generate_dataset.py uses), loading parquet via pyarrow to bypass this env's broken
pandas and avoid the Docker/augmentation path (not needed for associations).

Then compares the reproduced matches against Waymo's official
camera_to_lidar_box_association at the laser-id level per (frame, camera).

Usage (from visuals_dataset/):
    python compare_association_full.py 10023947602400723454_1120_000_1140_000
"""

import sys
from collections import defaultdict

import pyarrow.parquet as pq

from lidar_camera_association import enrich_camera_objects_with_lidar

BASE = "../dataset/waymo_open_dataset_v_2_0_1/training"
ASSOC = "../dataset/waymo_open_dataset_v_2_0_1/camera_to_lidar_box_association"


def read_rows(path):
    t = pq.ParquetFile(path).read()
    cols = {c: t.column(c).to_pylist() for c in t.column_names}
    return [{c: cols[c][i] for c in t.column_names} for i in range(t.num_rows)]


def group_by(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g


def main(segment):
    cam_rows = read_rows(f"{BASE}/camera_box/{segment}.parquet")
    lidar_rows = read_rows(f"{BASE}/lidar_box/{segment}.parquet")
    calib_rows = read_rows(f"{BASE}/camera_calibration/{segment}.parquet")

    cam_by_ts = group_by(cam_rows, "key.frame_timestamp_micros")
    lidar_by_ts = group_by(lidar_rows, "key.frame_timestamp_micros")

    # reproduce our association per frame
    ours = defaultdict(set)
    for ts, crows in cam_by_ts.items():
        cam_objs_by_camera = defaultdict(list)
        for r in crows:
            cam_objs_by_camera[str(r["key.camera_name"])].append(r)
        frame_components = {
            "lidar_box": lidar_by_ts.get(ts, []),
            "camera_calibration": calib_rows,
        }
        enriched = enrich_camera_objects_with_lidar(dict(cam_objs_by_camera), frame_components)
        for cam, objs in enriched.items():
            for o in objs:
                a = o.get("lidar_association", {})
                if a.get("status") == "matched":
                    for lid in a.get("laser_object_ids", []):
                        ours[(int(ts), int(cam))].add(lid)

    # official
    assoc_rows = read_rows(f"{ASSOC}/training_camera_to_lidar_box_association_{segment}.parquet")
    official = defaultdict(set)
    for r in assoc_rows:
        official[(int(r["key.frame_timestamp_micros"]), int(r["key.camera_name"]))].add(
            r["key.laser_object_id"])

    keys = set(official) | set(ours)
    inter = off_only = our_only = off_total = our_total = 0
    for k in keys:
        o, u = official.get(k, set()), ours.get(k, set())
        inter += len(o & u); off_only += len(o - u); our_only += len(u - o)
        off_total += len(o); our_total += len(u)

    print(f"=== {segment} (full segment, {len(cam_by_ts)} frames) ===")
    print(f"official associations : {off_total}")
    print(f"our matched           : {our_total}")
    print(f"agree (both)          : {inter}")
    print(f"official-only (missed): {off_only}")
    print(f"ours-only (extra)     : {our_only}")
    print(f"recall  (official recovered) : {inter / off_total:.3f}" if off_total else "recall: n/a")
    print(f"precision (ours confirmed)   : {inter / our_total:.3f}" if our_total else "precision: n/a")


if __name__ == "__main__":
    seg = sys.argv[1] if len(sys.argv) > 1 else "10023947602400723454_1120_000_1140_000"
    main(seg)
