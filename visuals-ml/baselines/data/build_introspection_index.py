"""
Build a per-FRAME index for the Introspective Perception baseline (Paper 1,
Daftry et al. 2016, arXiv 1607.08665).

Unlike data/build_index.py (one line per object, for PositionNet regression) and
build_detection_index.py (one line per image with a target list for MonoDETR),
this emits one line per (frame, weather variant) carrying:

  - the spatial-stream image (this frame, this weather),
  - an ordered list of the preceding consecutive frames of the SAME weather
    variant, used by the dataset to compute the temporal (optical-flow) stream,
  - the matched-object ground truth needed by the label pre-pass to score the
    underlying perception system (PositionNet) on this frame.

This stage is model-free: it only walks metadata + the image tree. The failure
label y_i itself is filled in later by introspection_label_prepass.py, which runs
a trained PositionNet over the objects here and writes an augmented index.

Record:
    {
      "frame_key": "<segment>|<timestamp>",
      "segment": "<segment ctx name>",
      "camera": "1",
      "weather": "fog",
      "image_path": ".../images/camera_1/fog/<stem>.jpeg",
      "flow_frames": [".../fog/<stem_{t-L}>.jpeg", ..., ".../fog/<stem_t>.jpeg"],
      "intrinsic": {"fu":.., "fv":.., "cu":.., "cv":..},
      "objects": [
        {"cx_n":.., "cy_n":.., "sw_n":.., "sh_n":.., "tx":.., "ty":.., "tz":..},
        ...
      ]
    }

`flow_frames` has up to FLOW_STACK+1 entries (this frame last); at a sequence
start it is left-padded by repeating the earliest available frame, so the dataset
always sees FLOW_STACK flow fields (zero flow across padded pairs).

Usage (from visuals-ml/):
    python -m baselines.data.build_introspection_index \
        --source-dir ../visuals_dataset/output \
        --index-file data/output/introspection_records.jsonl \
        --cameras 1 --flow-stack 5
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

# LiDAR box centre keys in the camera_to_lidar association (ego-vehicle frame),
# matching data/build_index.py so the label pre-pass scores PositionNet against
# the exact targets it was trained on.
_LB_X = "[LiDARBoxComponent].box.center.x"
_LB_Y = "[LiDARBoxComponent].box.center.y"
_LB_Z = "[LiDARBoxComponent].box.center.z"


def _matched_objects(meta: dict):
    """Matched-object GT (PositionNet inputs + targets). Mirrors build_index.py."""
    objs = []
    for obj in meta.get("Objects", []):
        assoc = obj.get("lidar_association", {})
        if assoc.get("status") != "matched":
            continue
        lbs = assoc.get("lidar_boxes", [])
        if not lbs:
            continue
        lb = lbs[0]
        tx, ty, tz = lb.get(_LB_X), lb.get(_LB_Y), lb.get(_LB_Z)
        if None in (tx, ty, tz):
            continue
        box = obj.get("box_2d", {})
        cx, cy = box.get("center_x"), box.get("center_y")
        sw, sh = box.get("size_x"), box.get("size_y")
        if None in (cx, cy, sw, sh):
            continue
        objs.append({
            "cx_n": cx / IMG_W, "cy_n": cy / IMG_H,
            "sw_n": sw / IMG_W, "sh_n": sh / IMG_H,
            "tx": tx, "ty": ty, "tz": tz,
        })
    return objs


def _flow_frames(ordered_paths, t, flow_stack):
    """The FLOW_STACK+1 consecutive image paths ending at index t, left-padded by
    repeating the earliest available frame at a sequence start."""
    start = t - flow_stack
    window = []
    for i in range(start, t + 1):
        window.append(ordered_paths[max(i, 0)])
    return window


def build(source_dir: Path, cameras, flow_stack: int):
    """Yield per-(frame, weather) records for the requested cameras."""
    camera_dirs = {f"camera_{c}" for c in cameras}

    for seg_dir in sorted(p for p in source_dir.glob("segment_*") if p.is_dir()):
        img_root = seg_dir / "images"
        meta_root = seg_dir / "metadata" / "image_metadata"
        if not img_root.is_dir() or not meta_root.is_dir():
            continue

        for cam_dir in sorted(p for p in meta_root.iterdir() if p.name in camera_dirs):
            camera = cam_dir.name  # camera_1
            # Ordered frame stems for this (segment, camera). Stems are shared
            # across weather variants (same underlying frame); zero-padded
            # timestamp+index means lexical order == temporal order.
            stems = sorted(p.stem for p in cam_dir.glob("*.json"))
            stem_pos = {s: i for i, s in enumerate(stems)}

            for meta_path in sorted(cam_dir.glob("*.json")):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)

                intr = meta.get("camera_calibration", {}).get("intrinsic", {})
                fu, fv = intr.get("f_u"), intr.get("f_v")
                cu, cv = intr.get("c_u"), intr.get("c_v")
                if None in (fu, fv, cu, cv):
                    continue

                objects = _matched_objects(meta)
                if not objects:
                    continue  # no GT -> no failure label possible

                stem = meta_path.stem
                t = stem_pos[stem]

                for variant in WEATHER_VARIANTS:
                    var_dir = img_root / camera / variant
                    image_path = var_dir / (stem + ".jpeg")
                    if not image_path.exists():
                        continue
                    ordered = [str(var_dir / (s + ".jpeg")) for s in stems]
                    yield {
                        "frame_key": f"{seg_dir.name}|{meta.get('timestamp_micros')}",
                        "segment": meta.get("segment_context_name", seg_dir.name),
                        "camera": meta.get("camera_name", camera.split("_")[-1]),
                        "weather": variant,
                        "image_path": str(image_path),
                        "flow_frames": _flow_frames(ordered, t, flow_stack),
                        "intrinsic": {"fu": fu, "fv": fv, "cu": cu, "cv": cv},
                        "objects": objects,
                    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="../visuals_dataset/output")
    parser.add_argument("--index-file", default="data/output/introspection_records.jsonl")
    parser.add_argument("--cameras", default="1",
                        help="Comma-separated camera ids (e.g. '1' or '1,2,3,4,5')")
    parser.add_argument("--flow-stack", type=int, default=5,
                        help="Number of optical-flow fields (=> flow_stack+1 frames)")
    args = parser.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    source = Path(args.source_dir)
    index_path = Path(args.index_file)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    n_records, n_objects = 0, 0
    with open(index_path, "w", encoding="utf-8") as out:
        for rec in build(source, cameras, args.flow_stack):
            out.write(json.dumps(rec) + "\n")
            n_records += 1
            n_objects += len(rec["objects"])

    print(f"Wrote {n_records} frame records ({n_objects} object instances) "
          f"to {index_path}")


if __name__ == "__main__":
    main()
