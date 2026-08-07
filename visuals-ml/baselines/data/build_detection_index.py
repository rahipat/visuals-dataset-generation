"""
Build a per-IMAGE detection index for monocular 3D baselines (MonoDETR).

Unlike data/build_index.py (one line per object, for PositionNet regression), this
emits one line per (image, weather variant), each carrying the full list of
ground-truth vehicle objects with camera-frame 3D labels precomputed by the
validated geometry adapter (baselines/core/geometry.py).

Each record:
    {
      "image_path": ".../camera_1/<weather>/<stem>.jpeg",
      "weather": "clear",
      "camera": "1",
      "image_size": [W, H],
      "intrinsic": {"fu":.., "fv":.., "cu":.., "cv":..},
      "objects": [
         {"box_2d": [cx,cy,w,h],        # pixels, native resolution
          "cls": 1,                      # vehicle
          "loc": [x, y, z],              # camera optical frame (z = depth), metres
          "dim": [h, w, l],              # KITTI/MonoDETR order, metres
          "ry": <radians>},
         ...
      ]
    }

Usage (from visuals-ml/):
    python -m baselines.data.build_detection_index \
        --source-dir ../visuals_dataset/output \
        --index-file data/output/det_records.jsonl \
        --cameras 1
"""

import argparse
import json
from pathlib import Path

from baselines.core import geometry as g

WEATHER_VARIANTS = [
    "clear", "rain", "fog", "snow", "frost",
    "sunglare", "brightness", "wildfire_smoke", "dust", "waterdrop",
]


BORDER_PX = 2.0  # tolerance for calling a 2D box "touching the frame edge"


def _is_truncated(cx, cy, sw, sh, width, height) -> bool:
    """True if the 2D box runs into the frame border.

    Matters because a clipped 2D box's centre no longer coincides with the
    projected 3D centre, so the box->3D correspondence the regressor learns is
    corrupted. Measured on the sample: the projected 3D centre falls inside its
    own 2D box for ~86% of whole objects but only ~48-61% of truncated ones,
    consistently across all three cameras. KITTI carries the same attribute.
    """
    if not width or not height:
        return False
    return (cx - sw / 2 <= BORDER_PX or cy - sh / 2 <= BORDER_PX
            or cx + sw / 2 >= width - BORDER_PX or cy + sh / 2 >= height - BORDER_PX)


def _objects_for_frame(meta: dict, params: dict):
    """Extract matched vehicle objects with camera-frame 3D labels."""
    size = meta.get("image_size", {})
    width, height = size.get("width"), size.get("height")
    objs = []
    for obj in meta.get("Objects", []):
        if obj.get("type") != g.TYPE_VEHICLE:
            continue
        assoc = obj.get("lidar_association", {})
        if assoc.get("status") != "matched":
            continue
        lbs = assoc.get("lidar_boxes", [])
        if not lbs:
            continue
        box3d = g.box_to_camera_frame(lbs[0], params)
        if box3d is None:
            continue
        b = obj.get("box_2d", {})
        cx, cy = b.get("center_x"), b.get("center_y")
        sw, sh = b.get("size_x"), b.get("size_y")
        if None in (cx, cy, sw, sh):
            continue
        objs.append({
            "box_2d": [cx, cy, sw, sh],
            "cls": g.TYPE_VEHICLE,
            "loc": [box3d.x, box3d.y, box3d.z],
            "dim": [box3d.h, box3d.w, box3d.l],
            "ry": box3d.ry,
            # Point count is the honest visibility/difficulty proxy in this
            # dataset (difficulty_level.detection is null on ~92% of boxes).
            "n_pts": lbs[0].get("[LiDARBoxComponent].num_lidar_points_in_box"),
            "track_id": lbs[0].get("key.laser_object_id"),
            "truncated": _is_truncated(cx, cy, sw, sh, width, height),
        })
    return objs


def extract_records(meta_path: Path, segment_dir: Path, weathers=WEATHER_VARIANTS):
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    params = g.projection_params(meta.get("camera_calibration", {}))
    if params is None:
        return

    objects = _objects_for_frame(meta, params)
    if not objects:
        return

    camera = meta_path.parent.name  # camera_1
    stem = meta_path.stem
    size = meta.get("image_size", {})
    intr = meta["camera_calibration"]["intrinsic"]
    base = {
        # `segment` is the grouping key for a leak-free train/val split: frames
        # inside a segment are ~10 Hz samples of one continuous drive, and every
        # frame is re-rendered under 10 weathers, so splitting on records would
        # put near-duplicates of the same object on both sides.
        "segment": segment_dir.name,
        "stem": stem,
        "timestamp_micros": meta.get("timestamp_micros"),
        "camera": meta.get("camera_name"),
        "image_size": [size.get("width"), size.get("height")],
        "intrinsic": {k: intr[k] for k in ("f_u", "f_v", "c_u", "c_v")},
        "objects": objects,
    }

    for variant in weathers:
        img = segment_dir / "images" / camera / variant / (stem + ".jpeg")
        if img.exists():
            yield {**base, "weather": variant, "image_path": str(img)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="../visuals_dataset/output")
    parser.add_argument("--index-file", default="data/output/det_records.jsonl")
    parser.add_argument("--cameras", default="1",
                        help="Comma-separated camera ids to include (e.g. '1' or '1,2,3,4,5')")
    parser.add_argument("--weathers", default=",".join(WEATHER_VARIANTS),
                        help="Comma-separated weather variants to emit. The full "
                             "800-segment tree is ~4.8M images; restrict this "
                             "(e.g. 'clear') to keep the index tractable.")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Keep every Nth frame per segment. Frames are ~10 Hz, "
                             "so consecutive ones are near-duplicates; stride 5 "
                             "cuts the index 5x at little information cost.")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Cap the number of segments (smoke runs).")
    args = parser.parse_args()

    cameras = {f"camera_{c.strip()}" for c in args.cameras.split(",")}
    weathers = [w.strip() for w in args.weathers.split(",") if w.strip()]
    unknown = set(weathers) - set(WEATHER_VARIANTS)
    if unknown:
        parser.error(f"Unknown weather variant(s): {sorted(unknown)}")

    source = Path(args.source_dir)
    index_path = Path(args.index_file)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    meta_files = [
        p for p in sorted(source.glob("*/metadata/image_metadata/**/*.json"))
        if p.parent.name in cameras
    ]

    # Stride within each (segment, camera) so the decimation is temporal rather
    # than an arbitrary slice of the global sorted order.
    if args.frame_stride > 1 or args.max_segments:
        by_group = {}
        for p in meta_files:
            by_group.setdefault((p.parents[3].name, p.parent.name), []).append(p)
        segments = sorted({seg for seg, _ in by_group})
        if args.max_segments:
            segments = segments[:args.max_segments]
        keep = set(segments)
        meta_files = [
            p for (seg, _cam), ps in sorted(by_group.items()) if seg in keep
            for p in ps[::args.frame_stride]
        ]

    print(f"Found {len(meta_files)} image metadata files for cameras "
          f"{sorted(cameras)} (stride {args.frame_stride}, {len(weathers)} weathers)")

    n_records, n_objects = 0, 0
    with open(index_path, "w", encoding="utf-8") as out:
        for meta_path in meta_files:
            segment_dir = meta_path.parents[3]
            for record in extract_records(meta_path, segment_dir, weathers):
                out.write(json.dumps(record) + "\n")
                n_records += 1
                n_objects += len(record["objects"])

    print(f"Wrote {n_records} image records ({n_objects} object instances) to {index_path}")


if __name__ == "__main__":
    main()
