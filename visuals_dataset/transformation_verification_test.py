"""
Transformation verification test.

Reads the first segment in the output folder (errors if none present), finds all
camera images that contain at least one LiDAR-matched object, draws the 2D bounding
box used for IoU matching onto the corresponding clear image in green (solid border,
translucent fill), and writes results to output/verification/.

Usage:
    python transformation_verification_test.py [--output-dir output]
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def _draw_boxes(image: Image.Image, objects: list) -> Image.Image:
    """Overlay green bounding boxes for every LiDAR-matched object.

    Uses the projected LiDAR box coordinates — these are the boxes that were
    actually used in the IoU matching step, projected from 3D vehicle frame
    into pixel space.
    """
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        assoc = obj.get("lidar_association", {})
        if assoc.get("status") != "matched":
            continue

        for proj in assoc.get("projected_lidar_boxes", []):
            if not isinstance(proj, dict):
                continue
            cx = proj.get("[ProjectedLiDARBoxComponent].box.center.x")
            cy = proj.get("[ProjectedLiDARBoxComponent].box.center.y")
            sx = proj.get("[ProjectedLiDARBoxComponent].box.size.x")
            sy = proj.get("[ProjectedLiDARBoxComponent].box.size.y")
            if None in (cx, cy, sx, sy):
                continue

            x1 = cx - 0.5 * sx
            y1 = cy - 0.5 * sy
            x2 = cx + 0.5 * sx
            y2 = cy + 0.5 * sy

            draw.rectangle([x1, y1, x2, y2], fill=(0, 200, 0, 60))
            draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0, 255), width=2)

    return Image.alpha_composite(base, overlay).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw LiDAR-matched 2D bounding boxes on clear images for verification."
    )
    parser.add_argument(
        "--output-dir", default="output", help="Output root folder (default: output)"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        print("[error] Output directory not found: %s" % out_dir)
        sys.exit(1)

    segment_dirs = sorted(
        d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("segment_")
    )
    if not segment_dirs:
        print("[error] No segment_* folders found in %s" % out_dir)
        sys.exit(1)

    segment_dir = segment_dirs[0]
    print("[info] Using segment: %s" % segment_dir.name)

    image_meta_dir = segment_dir / "metadata" / "image_metadata"
    clear_dir = segment_dir / "images" / "clear"

    if not image_meta_dir.exists():
        print("[error] Image metadata folder not found: %s" % image_meta_dir)
        sys.exit(1)
    if not clear_dir.exists():
        print("[error] Clear image folder not found: %s" % clear_dir)
        sys.exit(1)

    meta_files = sorted(image_meta_dir.glob("*.json"))
    if not meta_files:
        print("[error] No image metadata JSON files found in %s" % image_meta_dir)
        sys.exit(1)

    verification_dir = out_dir / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped_no_match = 0
    skipped_no_image = 0

    for meta_path in meta_files:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        objects = metadata.get("Objects", [])
        has_match = any(
            isinstance(o, dict) and o.get("lidar_association", {}).get("status") == "matched"
            for o in objects
        )
        if not has_match:
            skipped_no_match += 1
            continue

        stem = meta_path.stem
        candidates = list(clear_dir.glob("%s.*" % stem))
        if not candidates:
            print("[warn] No clear image found for %s" % stem)
            skipped_no_image += 1
            continue

        try:
            img = Image.open(candidates[0])
        except Exception as exc:
            print("[warn] Failed to open %s: %s" % (candidates[0], exc))
            skipped_no_image += 1
            continue

        result = _draw_boxes(img, objects)
        out_path = verification_dir / ("%s.jpeg" % stem)
        result.save(str(out_path), format="JPEG", quality=95)
        processed += 1

    print("[success] Wrote %d verification images to %s" % (processed, verification_dir))
    if skipped_no_match:
        print("[info]   %d images skipped — no LiDAR matches" % skipped_no_match)
    if skipped_no_image:
        print("[info]   %d images skipped — no corresponding clear image found" % skipped_no_image)


if __name__ == "__main__":
    main()
