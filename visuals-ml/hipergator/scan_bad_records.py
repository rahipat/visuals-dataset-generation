#!/usr/bin/env python3
"""
Scan an index (.jsonl) for records whose GEOMETRY can poison a training batch.

Corrupt images were already handled (see scan_corrupt_images.py); this covers
the other class of bad record, where the file decodes fine but a numeric field
is non-finite or degenerate.

The dangerous ones are image-level, not per-object:

  intrinsic.f_u/f_v/c_u/c_v, image_size
      These build the calib matrix P. Inside MonoDETR,
          depth_geo = size3d / box2d_height * calibs[:, 0, 0]
      so a single NaN/Inf intrinsic makes pred_depth NaN for the entire image,
      and every loss component (ce, bbox, center, depth, dim, angle) goes NaN
      at once -- with perfectly healthy weights. Because it is one record in a
      shuffled epoch, it surfaces at an arbitrary batch index, which reads like
      a random "sudden divergence".

Per-object fields (box_2d, loc, dim) are also checked; the dataset already
skips those objects, so they are reported as INFO rather than as poison.

Usage:
    python hipergator/scan_bad_records.py data/output/det_records.jsonl
    python hipergator/scan_bad_records.py data/output/records.jsonl --kind position
"""

import argparse
import json
import math
import sys


def _finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def check_detection(r):
    """Returns (poison, notes) -- poison entries make a whole image NaN."""
    poison, notes = [], []

    size = r.get("image_size")
    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        poison.append(f"image_size={size!r}")
    else:
        for name, v in zip(("img_w", "img_h"), size):
            if not _finite(v):
                poison.append(f"{name}={v!r}")
            elif v <= 0:
                poison.append(f"{name}={v} (non-positive)")

    intr = r.get("intrinsic")
    if not isinstance(intr, dict):
        poison.append(f"intrinsic={intr!r}")
    else:
        for k in ("f_u", "f_v", "c_u", "c_v"):
            v = intr.get(k)
            if not _finite(v):
                poison.append(f"intrinsic.{k}={v!r}")

    for j, o in enumerate(r.get("objects", [])):
        for key, n in (("box_2d", 4), ("loc", 3), ("dim", 3)):
            vals = o.get(key)
            if not (isinstance(vals, (list, tuple)) and len(vals) == n
                    and all(_finite(v) for v in vals)):
                notes.append(f"obj[{j}].{key}={vals!r}")
        b = o.get("box_2d")
        if isinstance(b, (list, tuple)) and len(b) == 4 and all(_finite(v) for v in b):
            if b[2] <= 0 or b[3] <= 0:
                notes.append(f"obj[{j}].box_2d w/h non-positive: {b}")
    return poison, notes


def check_position(r):
    poison, notes = [], []
    for k in ("cx_n", "cy_n", "sw_n", "sh_n", "fu", "fv", "cu", "cv",
              "tx", "ty", "tz"):
        v = r.get(k)
        if not _finite(v):
            poison.append(f"{k}={v!r}")
    return poison, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("index", help="path to the .jsonl index")
    ap.add_argument("--kind", choices=("detection", "position"), default="detection")
    ap.add_argument("--max-report", type=int, default=20)
    args = ap.parse_args()

    check = check_detection if args.kind == "detection" else check_position

    n = n_poison = n_noted = 0
    shown = 0
    try:
        f = open(args.index, encoding="utf-8")
    except OSError as e:
        sys.exit(f"cannot open index: {e}")

    with f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                n_poison += 1
                if shown < args.max_report:
                    shown += 1
                    print(f"POISON line {i}: unparseable JSON ({e})")
                continue
            poison, notes = check(r)
            if poison:
                n_poison += 1
                if shown < args.max_report:
                    shown += 1
                    print(f"POISON line {i}: {'; '.join(poison)}")
                    print(f"    image_path={r.get('image_path')}")
            if notes:
                n_noted += 1

    print(f"\nscanned {n:,} records in {args.index}")
    print(f"  image-level POISON (whole-batch NaN risk): {n_poison:,}")
    print(f"  records with skippable per-object issues : {n_noted:,}")
    if n_poison:
        print("\nEven ONE poison record is enough: shuffled into an epoch it will\n"
              "make every loss component NaN at whatever batch it lands on.")
    sys.exit(1 if n_poison else 0)


if __name__ == "__main__":
    main()
