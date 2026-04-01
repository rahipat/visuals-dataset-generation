"""
Waymo Open Dataset v2 image fetcher with augmentation and metadata writing.

Default mode reads from local mirror:
  ../dataset/waymo_open_dataset_v_2_0_1

Use -cloud/--cloud to read directly from GCS.
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from augment import ImageAugmenter, decode_image, encode_image
from lidar_camera_association import enrich_camera_objects_with_lidar


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAYMO_UTILS_DIR = PROJECT_ROOT / "waymo-open-dataset-utils"
if WAYMO_UTILS_DIR.exists():
    sys.path.insert(0, str(WAYMO_UTILS_DIR))

from frame_utils import (  # type: ignore  # noqa: E402
    build_frame_key,
    build_scene_name,
    camera_image_row_to_record,
    extract_camera_calibration,
)
from parquet_component_loader import (  # type: ignore  # noqa: E402
    DATASET_BUCKET_NAME,
    DEFAULT_METADATA_COMPONENTS,
    SCENE_LEVEL_COMPONENTS,
    FrameMetadataRepository,
    build_frame_keys_dataframe_from_records,
    load_component_dataframe,
    load_filtered_component_tables,
    resolve_local_dataset_base,
)


CLOUD_DATASET_BASE = "gs://waymo_open_dataset_v_2_0_1"
DEFAULT_LOCAL_DATASET_ROOT = PROJECT_ROOT / "dataset"
METADATA_COMPONENTS = list(DEFAULT_METADATA_COMPONENTS)
DEFAULT_SKIPPED_CAMERA_NAMES = {"SIDE_LEFT", "SIDE_RIGHT"}


def _normalize_camera_name(camera_name: Any) -> str:
    return str(camera_name).strip().upper()


def fetch_camera_images(
    dataset_base: str,
    split: str = "training",
    max_files: Optional[int] = None,
    skip_camera_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Read camera image component rows and convert them to image records."""
    camera_df = load_component_dataframe(
        dataset_base=dataset_base,
        split=split,
        component="camera_image",
        max_files=max_files,
        verbose=verbose,
    )
    if camera_df.empty:
        print("[warn] No camera image rows found.")
        return []

    skipped = {_normalize_camera_name(name) for name in (skip_camera_names or []) if str(name).strip()}
    if skipped and "key.camera_name" in camera_df.columns:
        before_count = len(camera_df)
        keep_mask = ~camera_df["key.camera_name"].map(_normalize_camera_name).isin(skipped)
        camera_df = camera_df.loc[keep_mask]
        removed_count = before_count - len(camera_df)
        if verbose or removed_count:
            print(
                "[info] Skipped %d camera_image rows for cameras: %s"
                % (removed_count, ", ".join(sorted(skipped)))
            )

    if camera_df.empty:
        print("[warn] No camera image rows left after camera filtering.")
        return []

    records = []
    for idx, row in camera_df.iterrows():
        record = camera_image_row_to_record(row.to_dict())
        if record.get("image_bytes") is None:
            if verbose:
                print("[warn] Missing image bytes in camera row index=%s" % idx)
            continue
        records.append(record)

    if verbose:
        print("[debug] Converted %d camera rows to image records." % len(records))
    return records


def _group_camera_rows_by_name(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        camera_name = row.get("key.camera_name")
        if camera_name is None:
            continue
        grouped.setdefault(str(camera_name), []).append(row)
    return grouped


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _json_safe(to_list())
        except Exception:
            return str(value)
    return str(value)


def _compact_mapping(payload: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, dict) and not value:
            continue
        if isinstance(value, list) and not value:
            continue
        compact[key] = value
    return compact


def apply_degradations_and_save(
    images: List[Dict[str, Any]],
    output_dir: str,
    augmenter: ImageAugmenter,
    metadata_repo: FrameMetadataRepository,
    image_format: str = "JPEG",
    verbose: bool = False,
) -> List[Path]:
    """
    Apply augmentations, save clear + augmented images, and write metadata:
      - metadata/scene_metadata: one file per frame (scene/timestamp)
      - metadata/image_metadata: one file per camera image
    """
    out_dir = Path(output_dir)
    written: List[Path] = []
    scene_written_keys = set()

    current_frame_key: Optional[str] = None
    current_scene_components: Dict[str, List[Dict[str, Any]]] = {}
    current_camera_objects_by_camera: Dict[str, List[Dict[str, Any]]] = {}
    current_camera_objects_with_lidar_by_camera: Dict[str, List[Dict[str, Any]]] = {}
    current_camera_calibration_by_camera: Dict[str, Dict[str, Any]] = {}

    indexed_images = list(enumerate(images))
    indexed_images.sort(
        key=lambda item: (
            item[1].get("segment_context_name", ""),
            item[1].get("timestamp_micros", 0),
            item[0],
        )
    )

    meta_dir = out_dir / "metadata"
    scene_meta_dir = meta_dir / "scene_metadata"
    image_meta_dir = meta_dir / "image_metadata"
    scene_meta_dir.mkdir(parents=True, exist_ok=True)
    image_meta_dir.mkdir(parents=True, exist_ok=True)

    for idx, record in indexed_images:
        segment = str(record.get("segment_context_name", "unknown"))
        timestamp = int(record.get("timestamp_micros", 0))
        camera_name = str(record.get("camera_name", "unknown"))
        frame_key = build_frame_key(segment, timestamp)
        scene_name = build_scene_name(segment, timestamp)
        base_name = "%s_%03d" % (scene_name, idx)

        try:
            pil_img = decode_image(record["image_bytes"])
        except Exception as exc:
            print("[warn] Failed to decode image %s: %s" % (base_name, exc))
            continue

        if frame_key != current_frame_key:
            if verbose:
                print("[debug] Loading metadata components for frame %s" % frame_key)
            frame_components = metadata_repo.get_frame_components(segment, timestamp)
            current_scene_components = {
                name: frame_components.get(name, [])
                for name in SCENE_LEVEL_COMPONENTS
            }
            current_camera_objects_by_camera = _group_camera_rows_by_name(
                frame_components.get("camera_box", [])
            )
            current_camera_objects_with_lidar_by_camera = enrich_camera_objects_with_lidar(
                current_camera_objects_by_camera, frame_components
            )

            current_camera_calibration_by_camera = {}
            calibration_rows = frame_components.get("camera_calibration", [])
            for row in calibration_rows:
                if not isinstance(row, dict):
                    continue
                cam_key = row.get("key.camera_name")
                if cam_key is None:
                    continue
                current_camera_calibration_by_camera[str(cam_key)] = extract_camera_calibration(row)

            current_frame_key = frame_key
        elif verbose:
            print("[debug] Reusing metadata components for frame %s" % frame_key)

        try:
            clear_dir = out_dir / "clear"
            clear_dir.mkdir(parents=True, exist_ok=True)
            clear_path = clear_dir / ("%s.%s" % (base_name, image_format.lower()))
            clear_path.write_bytes(encode_image(pil_img, format=image_format))
            written.append(clear_path)

            scene_meta_path = scene_meta_dir / ("%s.json" % scene_name)
            if frame_key not in scene_written_keys:
                scene_metadata = {
                    "generated_at_utc": datetime.utcnow().isoformat() + "Z",
                    "dataset_version": DATASET_BUCKET_NAME,
                    "scene_name": scene_name,
                    "segment_context_name": segment,
                    "timestamp_micros": timestamp,
                    "components": _json_safe(current_scene_components),
                }
                with open(scene_meta_path, "w", encoding="utf-8") as f:
                    json.dump(scene_metadata, f, indent=2)
                written.append(scene_meta_path)
                scene_written_keys.add(frame_key)

            image_meta_path = image_meta_dir / ("%s.json" % base_name)
            image_metadata = {
                "generated_at_utc": datetime.utcnow().isoformat() + "Z",
                "dataset_version": DATASET_BUCKET_NAME,
                "frame_key": frame_key,
                "scene_name": scene_name,
                "scene_metadata_file": str(Path("scene_metadata") / ("%s.json" % scene_name)),
                "segment_context_name": segment,
                "timestamp_micros": timestamp,
                "camera_name": camera_name,
                "image_size": {
                    "width": pil_img.size[0],
                    "height": pil_img.size[1],
                },
                "velocity": _compact_mapping(_json_safe(record.get("velocity", {}))),
                "camera_timing": _compact_mapping(_json_safe(record.get("camera_timing", {}))),
                "camera_pose_transform": _json_safe(record.get("camera_pose_transform")),
                "camera_calibration": _compact_mapping(
                    _json_safe(current_camera_calibration_by_camera.get(camera_name, {}))
                ),
                "Objects": _json_safe(current_camera_objects_with_lidar_by_camera.get(camera_name, [])),
            }
            image_metadata = _compact_mapping(image_metadata)
            with open(image_meta_path, "w", encoding="utf-8") as f:
                json.dump(image_metadata, f, indent=2)
            written.append(image_meta_path)
        except Exception as exc:
            print("[warn] Failed to save clear image/metadata for %s: %s" % (base_name, exc))

        aug_map = augmenter.apply_all(pil_img)
        for aug_name, aug_img in aug_map.items():
            dest_dir = out_dir / aug_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            img_path = dest_dir / ("%s.%s" % (base_name, image_format.lower()))
            img_path.write_bytes(encode_image(aug_img, format=image_format))
            written.append(img_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Waymo images, apply augmentations, and write scene/image metadata."
    )
    parser.add_argument("--split", default="training", help="Dataset split (training/validation/test).")
    parser.add_argument("--count", type=int, default=10, help="Max images to process.")
    parser.add_argument("--output-dir", default="output", help="Output folder.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit camera_image parquet files scanned.")
    parser.add_argument(
        "--metadata-max-files",
        type=int,
        default=None,
        help="Optional cap per metadata component folder. Default: 10 in cloud mode, unlimited local.",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for augmentations.")
    parser.add_argument(
        "--random-start",
        type=int,
        default=None,
        help="If set, shuffles images using this seed before selecting --count.",
    )
    parser.add_argument(
        "-cloud",
        "--cloud",
        action="store_true",
        help="Read directly from GCS. Default mode reads local cached dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_LOCAL_DATASET_ROOT),
        help="Local dataset root. Can be .../dataset or .../dataset/waymo_open_dataset_v_2_0_1",
    )
    parser.add_argument(
        "--include-ultrawide",
        action="store_true",
        help="Include SIDE_LEFT and SIDE_RIGHT camera images.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    if args.cloud:
        dataset_base = CLOUD_DATASET_BASE
        print("[info] Data source mode: cloud (%s)" % dataset_base)
    else:
        dataset_base_path = resolve_local_dataset_base(Path(args.dataset_root).resolve())
        split_camera_path = dataset_base_path / args.split / "camera_image"
        if not split_camera_path.exists():
            print("[error] Local dataset not found at %s" % split_camera_path)
            print("[hint] Build a local subset first, for example:")
            print("       python cache_waymo_subset.py --split %s --max-camera-files 5" % args.split)
            return
        dataset_base = str(dataset_base_path)
        print("[info] Data source mode: local (%s)" % dataset_base)

    print("[info] Fetching images from split: %s" % args.split)
    skip_camera_names: List[str] = []
    if not args.include_ultrawide:
        skip_camera_names.extend(sorted(DEFAULT_SKIPPED_CAMERA_NAMES))
        print(
            "[info] Skipping ultrawide cameras by default: %s"
            % ", ".join(sorted(DEFAULT_SKIPPED_CAMERA_NAMES))
        )
    images = fetch_camera_images(
        dataset_base=dataset_base,
        split=args.split,
        max_files=args.max_files,
        skip_camera_names=skip_camera_names,
        verbose=args.verbose,
    )
    if not images:
        print("[error] No images found. Check split and credentials.")
        return
    print("[info] Fetched %d images before sampling." % len(images))

    if args.random_start is not None:
        rng = random.Random(args.random_start)
        rng.shuffle(images)
        if args.verbose:
            print("[info] Shuffled images using seed %d" % args.random_start)
    images = images[: args.count]
    if not images:
        print("[error] No images selected after applying --count.")
        return
    print("[info] Selected %d images for output." % len(images))

    frame_keys_df = build_frame_keys_dataframe_from_records(images)
    metadata_max_files = args.metadata_max_files
    if metadata_max_files is None and args.cloud:
        metadata_max_files = 10
    if args.verbose:
        print("[debug] Loading metadata components for %d unique frames." % len(frame_keys_df))

    component_tables = load_filtered_component_tables(
        dataset_base=dataset_base,
        split=args.split,
        components=METADATA_COMPONENTS,
        frame_keys_df=frame_keys_df,
        max_files_per_component=metadata_max_files,
        verbose=args.verbose,
    )
    metadata_repo = FrameMetadataRepository(component_tables=component_tables, verbose=args.verbose)

    augmenter = ImageAugmenter(seed=args.seed)
    paths = apply_degradations_and_save(
        images=images,
        output_dir=args.output_dir,
        augmenter=augmenter,
        metadata_repo=metadata_repo,
        verbose=args.verbose,
    )

    print("[success] Wrote %d files under %s" % (len(paths), args.output_dir))
    print(
        "  - Augmented images in: rain/, fog/, snow/, frost/, sunglare/, "
        "brightness/, wildfire_smoke/, dust/, waterdrop/"
    )
    print("  - Metadata JSON in: metadata/image_metadata/ and metadata/scene_metadata/")


if __name__ == "__main__":
    main()
