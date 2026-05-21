"""
Walk visuals_dataset/output and write a flat records.jsonl index.

One line per matched object per weather variant. Each line contains the
image path, 8 input features, and 3 target values.

Usage:
    python data/build_index.py \
        --output-dir ../visuals_dataset/output \
        --index-file data/records.jsonl
"""

import argparse
import json
from pathlib import Path


IMG_W = 1920.0
IMG_H = 1280.0
WEATHER_VARIANTS = [
    "clear", "rain", "fog", "snow", "frost",
    "sunglare", "brightness", "wildfire_smoke", "dust", "waterdrop",
]


def extract_records(meta_path: Path, segment_dir: Path):
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    intr = meta.get("camera_calibration", {}).get("intrinsic", {})
    fu = intr.get("f_u")
    fv = intr.get("f_v")
    cu = intr.get("c_u")
    cv = intr.get("c_v")
    if None in (fu, fv, cu, cv):
        return

    camera = meta_path.parent.name  # e.g. camera_1
    stem = meta_path.stem

    image_paths = {}
    for variant in WEATHER_VARIANTS:
        candidate = segment_dir / "images" / camera / variant / (stem + ".jpeg")
        if candidate.exists():
            image_paths[variant] = str(candidate)

    if not image_paths:
        return

    for obj in meta.get("Objects", []):
        assoc = obj.get("lidar_association", {})
        if assoc.get("status") != "matched":
            continue

        lidar_boxes = assoc.get("lidar_boxes", [])
        if not lidar_boxes:
            continue
        lb = lidar_boxes[0]

        tx = lb.get("[LiDARBoxComponent].box.center.x")
        ty = lb.get("[LiDARBoxComponent].box.center.y")
        tz = lb.get("[LiDARBoxComponent].box.center.z")
        if None in (tx, ty, tz):
            continue

        box = obj.get("box_2d", {})
        cx = box.get("center_x")
        cy = box.get("center_y")
        sw = box.get("size_x")
        sh = box.get("size_y")
        if None in (cx, cy, sw, sh):
            continue

        base = {
            "cx_n": cx / IMG_W,
            "cy_n": cy / IMG_H,
            "sw_n": sw / IMG_W,
            "sh_n": sh / IMG_H,
            "fu": fu,
            "fv": fv,
            "cu": cu,
            "cv": cv,
            "tx": tx,
            "ty": ty,
            "tz": tz,
        }

        for variant, img_path in image_paths.items():
            yield {**base, "weather": variant, "image_path": img_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="../visuals_dataset/output")
    parser.add_argument("--index-file", default="data/output/records.jsonl")
    args = parser.parse_args()

    output_dir = Path(args.source_dir)
    index_path = Path(args.index_file)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    meta_files = sorted(output_dir.glob("*/metadata/image_metadata/**/*.json"))
    print(f"Found {len(meta_files)} image metadata files")

    count = 0
    with open(index_path, "w", encoding="utf-8") as out:
        for meta_path in meta_files:
            segment_dir = meta_path.parents[3]
            for record in extract_records(meta_path, segment_dir):
                out.write(json.dumps(record) + "\n")
                count += 1

    print(f"Wrote {count} records to {index_path}")


if __name__ == "__main__":
    main()
