"""
Compare Waymo's official camera_to_lidar_box_association against our pipeline's
algorithmic geometry_projection_iou matching.

Official associations are keyed by camera_object_id (Waymo's id), which differs
from the UUIDs our pipeline assigns, so we compare at the level of which
laser_object_ids are associated-to-a-camera per (frame, camera):

  official_lasers[(ts, cam)] = {laser ids Waymo links to any camera box}
  ours_lasers[(ts, cam)]     = {laser ids we matched (status == 'matched')}

and report recall / precision of our matching against the official labels.

Reads parquet via pyarrow.ParquetFile (avoids pyarrow.dataset, which trips this
env's broken pandas shim). Usage (from visuals_dataset/):
    python compare_association.py
"""

import glob
import json
import os
from collections import defaultdict

import pyarrow.parquet as pq

ASSOC_DIR = "../dataset/waymo_open_dataset_v_2_0_1/camera_to_lidar_box_association"
OUTPUT_DIR = "output"


def load_official(segment):
    hits = glob.glob(f"{ASSOC_DIR}/*{segment}*.parquet")
    if not hits:
        return None
    t = pq.ParquetFile(hits[0]).read()
    cols = {c: t.column(c).to_pylist() for c in t.column_names}
    ts = cols["key.frame_timestamp_micros"]
    cam = cols["key.camera_name"]
    laser = cols["key.laser_object_id"]
    by_key = defaultdict(set)
    for i in range(t.num_rows):
        by_key[(int(ts[i]), int(cam[i]))].add(laser[i])
    return by_key, t.num_rows


def load_ours(segment):
    seg_dir = f"{OUTPUT_DIR}/segment_{segment}"
    metas = glob.glob(f"{seg_dir}/metadata/image_metadata/camera_*/*.json")
    by_key = defaultdict(set)
    generated_keys = set()
    n_matched = 0
    for mp in metas:
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        ts = int(m["timestamp_micros"])
        cam = int(m["camera_name"])
        generated_keys.add((ts, cam))
        for obj in m.get("Objects", []):
            a = obj.get("lidar_association", {})
            if a.get("status") == "matched":
                for lid in a.get("laser_object_ids", []):
                    by_key[(ts, cam)].add(lid)
                    n_matched += 1
    return by_key, generated_keys, n_matched, len(metas)


def compare(segment):
    official = load_official(segment)
    if official is None:
        print(f"  [no official parquet for {segment}]")
        return
    off_by_key, off_rows = official
    our_by_key, generated_keys, our_matched, n_imgs = load_ours(segment)

    if n_imgs == 0:
        print(f"  official rows={off_rows}; no generated output for this segment (skip detailed compare)")
        return

    # Compare only over frame-cameras we actually generated (including those where
    # we matched nothing), so neither side is penalized for frames out of scope.
    keys = generated_keys
    off_in_scope = sum(len(off_by_key.get(k, set())) for k in keys)
    print(f"  official rows={off_rows} (assoc in our {len(keys)} generated frame-cams = {off_in_scope})")
    inter = off_only = our_only = off_total = our_total = 0
    for k in keys:
        o, u = off_by_key.get(k, set()), our_by_key.get(k, set())
        inter += len(o & u)
        off_only += len(o - u)
        our_only += len(u - o)
        off_total += len(o)
        our_total += len(u)

    recall = inter / off_total if off_total else float("nan")
    precision = inter / our_total if our_total else float("nan")
    print(f"  our matched laser ids in scope = {our_total}  (over {n_imgs} images)")
    print(f"  agree (in both)     = {inter}")
    print(f"  official-only (we missed)      = {off_only}")
    print(f"  ours-only (no official assoc)  = {our_only}")
    print(f"  recall (official recovered)    = {recall:.3f}")
    print(f"  precision (ours confirmed)     = {precision:.3f}")


def main():
    segments = sorted({
        os.path.basename(p).split("camera_to_lidar_box_association_")[-1].replace(".parquet", "")
        for p in glob.glob(f"{ASSOC_DIR}/*.parquet")
    })
    for seg in segments:
        print(f"\n=== {seg} ===")
        compare(seg)


if __name__ == "__main__":
    main()
