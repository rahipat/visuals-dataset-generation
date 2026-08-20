"""
Build velocity_records.jsonl: paired consecutive-frame records for velocity prediction.

For each (segment, camera_dir, laser_object_id), find consecutive timestamps where
the same object is matched in both frames. For every weather variant where both
frame images exist, emit one record.

Usage:
    python data/build_velocity_index.py \
        --source-dir ../visuals_dataset/output \
        --index-file data/output/velocity_records.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

WEATHER_VARIANTS = [
    "clear", "rain", "fog", "snow", "frost",
    "sunglare", "brightness", "wildfire_smoke", "dust", "waterdrop",
]
IMG_W = 1920.0
IMG_H = 1280.0


def load_frame_objects(meta_path: Path, camera_dir: str):
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    timestamp = meta.get("timestamp_micros")
    segment = meta.get("segment_context_name")
    ego_vel = meta.get("velocity", {})
    stem = meta_path.stem

    results = []
    for obj in meta.get("Objects", []):
        assoc = obj.get("lidar_association", {})
        if assoc.get("status") != "matched":
            continue
        laser_ids = assoc.get("laser_object_ids", [])
        lidar_boxes = assoc.get("lidar_boxes", [])
        if not laser_ids or not lidar_boxes:
            continue

        lb = lidar_boxes[0]
        vx = lb.get("[LiDARBoxComponent].speed.x")
        vy = lb.get("[LiDARBoxComponent].speed.y")
        if vx is None or vy is None:
            continue

        box = obj.get("box_2d", {})
        cx = box.get("center_x")
        cy = box.get("center_y")
        sw = box.get("size_x")
        sh = box.get("size_y")
        if None in (cx, cy, sw, sh):
            continue

        results.append({
            "segment": segment,
            "camera_dir": camera_dir,
            "laser_id": laser_ids[0],
            "timestamp": timestamp,
            "stem": stem,
            "cx_n": cx / IMG_W,
            "cy_n": cy / IMG_H,
            "sw_n": sw / IMG_W,
            "sh_n": sh / IMG_H,
            "vx": vx,
            "vy": vy,
            "ego_linear_x":  ego_vel.get("linear_x",  0.0),
            "ego_linear_y":  ego_vel.get("linear_y",  0.0),
            "ego_linear_z":  ego_vel.get("linear_z",  0.0),
            "ego_angular_x": ego_vel.get("angular_x", 0.0),
            "ego_angular_y": ego_vel.get("angular_y", 0.0),
            "ego_angular_z": ego_vel.get("angular_z", 0.0),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="../visuals_dataset/output")
    parser.add_argument("--index-file", default="data/output/velocity_records.jsonl")
    args = parser.parse_args()

    output_dir = Path(args.source_dir)
    index_path = Path(args.index_file)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    meta_files = sorted(output_dir.glob("*/metadata/image_metadata/**/*.json"))
    print(f"Found {len(meta_files)} image metadata files")

    # Group: (segment, camera_dir, laser_id) -> list of frame dicts sorted by timestamp
    tracks = defaultdict(list)
    for meta_path in meta_files:
        camera_dir = meta_path.parent.name  # e.g. "camera_1"
        for obj in load_frame_objects(meta_path, camera_dir):
            key = (obj["segment"], obj["camera_dir"], obj["laser_id"])
            tracks[key].append(obj)

    for key in tracks:
        tracks[key].sort(key=lambda x: x["timestamp"])

    count = 0
    with open(index_path, "w", encoding="utf-8") as out:
        for (segment, camera_dir, laser_id), frames in tracks.items():
            seg_dirs = list(output_dir.glob(f"segment_{segment}*"))
            if not seg_dirs:
                continue
            seg_dir = seg_dirs[0]
            img_cam_dir = seg_dir / "images" / camera_dir

            for i in range(len(frames) - 1):
                f_t  = frames[i]
                f_t1 = frames[i + 1]
                delta_t = (f_t1["timestamp"] - f_t["timestamp"]) / 1e6  # seconds

                for variant in WEATHER_VARIANTS:
                    path_t  = img_cam_dir / variant / (f_t["stem"]  + ".jpeg")
                    path_t1 = img_cam_dir / variant / (f_t1["stem"] + ".jpeg")
                    if not path_t.exists() or not path_t1.exists():
                        continue

                    record = {
                        "segment":    segment,
                        "camera_dir": camera_dir,
                        "laser_id":   laser_id,
                        "weather":    variant,
                        "delta_t":    delta_t,
                        "image_t":    path_t.as_posix(),
                        "image_t1":   path_t1.as_posix(),
                        # box coords at T
                        "cx_n_t":  f_t["cx_n"],  "cy_n_t":  f_t["cy_n"],
                        "sw_n_t":  f_t["sw_n"],  "sh_n_t":  f_t["sh_n"],
                        # box coords at T+1
                        "cx_n_t1": f_t1["cx_n"], "cy_n_t1": f_t1["cy_n"],
                        "sw_n_t1": f_t1["sw_n"], "sh_n_t1": f_t1["sh_n"],
                        # ego velocity at T
                        "ego_linear_x":  f_t["ego_linear_x"],
                        "ego_linear_y":  f_t["ego_linear_y"],
                        "ego_linear_z":  f_t["ego_linear_z"],
                        "ego_angular_x": f_t["ego_angular_x"],
                        "ego_angular_y": f_t["ego_angular_y"],
                        "ego_angular_z": f_t["ego_angular_z"],
                        # targets: velocity at T
                        "vx": f_t["vx"],
                        "vy": f_t["vy"],
                    }
                    out.write(json.dumps(record) + "\n")
                    count += 1

    print(f"Wrote {count} records to {index_path}")


if __name__ == "__main__":
    main()
