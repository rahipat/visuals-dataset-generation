#!/usr/bin/env python3
"""
Scan a Waymo output tree for 0-byte or otherwise corrupt/unreadable images.

Written after PositionDataset crashed training on a 0-byte JPEG left over
from a disk-quota failure during dataset generation (see
data/dataset.py's PositionDataset, which now skips such files at load time
instead of crashing). This script finds the full scope of the damage across
the dataset so it can be regenerated or otherwise accounted for.

Usage:
    python hipergator/scan_corrupt_images.py \
        --root /orange/iruchkin/patel.rahi/waymo/output \
        --out corrupt_images_report.json
"""

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def check_image(path_str: str):
    """Return an error string if the image at path_str is 0-byte or fails to
    decode, else None."""
    path = Path(path_str)
    try:
        size = path.stat().st_size
    except OSError as e:
        return str(path), None, f"stat failed: {e}"
    if size == 0:
        return str(path), size, "0-byte file"
    try:
        with Image.open(path) as img:
            img.load()
    except (OSError, UnidentifiedImageError) as e:
        return str(path), size, f"decode failed: {e}"
    return str(path), size, None


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/orange/iruchkin/patel.rahi/waymo/output")
    parser.add_argument("--out", default="corrupt_images_report.json")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"Root not found: {root}")

    print(f"Scanning {root} ...")
    paths = list(iter_images(root))
    n = len(paths)
    print(f"Found {n} image files, checking...")

    bad = []
    checked = 0
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for path_str, size, err in ex.map(check_image, paths, chunksize=64):
            checked += 1
            if err is not None:
                bad.append({"path": path_str, "size": size, "error": err})
                print(f"  BAD  {path_str}  ({err})")
            if checked % 5000 == 0:
                print(f"  ...{checked}/{n} checked, {len(bad)} bad so far", flush=True)

    bad.sort(key=lambda r: r["path"])
    report = {"root": str(root), "n_scanned": n, "n_bad": len(bad), "bad_files": bad}
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"\nScanned {n} images, found {len(bad)} corrupt/empty.")
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
