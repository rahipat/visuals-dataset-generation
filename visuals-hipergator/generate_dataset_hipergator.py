"""
HiPerGator entrypoint for Waymo image augmentation + metadata writing.

Differences from visuals_dataset/generate_dataset.py:
  - Cloud (GCS) mode removed; --dataset-root is required and must be local
  - Adds --shard-index / --num-shards for SLURM array parallelism
    (sharding is round-robin over camera_image parquet files)
  - --count defaults to unlimited (process every selected image)
  - Imports augment + lidar_camera_association from sibling visuals_dataset/
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAYMO_UTILS_DIR = PROJECT_ROOT / "waymo-open-dataset-utils"
VISUALS_DATASET_DIR = PROJECT_ROOT / "visuals_dataset"
for extra in (WAYMO_UTILS_DIR, VISUALS_DATASET_DIR):
    if extra.exists():
        sys.path.insert(0, str(extra))

from augment import ImageAugmenter, decode_image, encode_image  # type: ignore  # noqa: E402
from lidar_camera_association import enrich_camera_objects_with_lidar  # type: ignore  # noqa: E402
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
    list_parquet_files,
    load_filtered_component_tables,
    read_parquet_dataframe,
    resolve_local_dataset_base,
)


METADATA_COMPONENTS = list(DEFAULT_METADATA_COMPONENTS)


def fetch_camera_images_sharded(
    dataset_base: str,
    split: str,
    shard_index: int,
    num_shards: int,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    List camera_image parquet files, take this shard's slice (round-robin),
    read each, and convert rows to image records.

    Waymo v2 parquet files are typically one segment per file, so round-robin
    sharding keeps segment writes disjoint across array tasks.
    """
    camera_root = "%s/%s/camera_image" % (dataset_base, split)
    all_files = list_parquet_files(camera_root, verbose=verbose)
    if not all_files:
        print("[error] No camera parquet files found under %s" % camera_root)
        return []

    if num_shards < 1:
        num_shards = 1
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            "shard_index %d out of range for num_shards %d" % (shard_index, num_shards)
        )

    shard_files = all_files[shard_index::num_shards]
    print(
        "[info] Shard %d/%d: %d of %d camera parquet files"
        % (shard_index, num_shards, len(shard_files), len(all_files))
    )

    records: List[Dict[str, Any]] = []
    for file_path in shard_files:
        df = read_parquet_dataframe(file_path, verbose=verbose)
        if df is None or df.empty:
            continue
        for idx, row in df.iterrows():
            record = camera_image_row_to_record(row.to_dict())
            if record.get("image_bytes") is None:
                if verbose:
                    print("[warn] Missing image bytes at %s row=%s" % (file_path, idx))
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


def _write_track_instances(
    objects: List[Dict[str, Any]],
    camera_name: str,
    timestamp_micros: int,
    segment_tracks: Dict[str, List[Dict[str, Any]]],
    tracks_dir: Path,
    written: List[Path],
) -> None:
    for obj in objects:
        assoc = obj.get("lidar_association")
        if not isinstance(assoc, dict) or assoc.get("status") != "matched":
            continue
        laser_ids = assoc.get("laser_object_ids", [])
        lidar_boxes = assoc.get("lidar_boxes", [])
        projected_boxes = assoc.get("projected_lidar_boxes", [])
        for laser_object_id in laser_ids:
            if not laser_object_id:
                continue
            lidar_box = next(
                (b for b in lidar_boxes if b.get("key.laser_object_id") == laser_object_id),
                None,
            )
            projected_box = next(
                (b for b in projected_boxes if b.get("key.laser_object_id") == laser_object_id),
                None,
            )
            track_key = "%s-%s" % (laser_object_id, camera_name)
            instance = {
                "timestamp_micros": timestamp_micros,
                "camera_object_id": obj.get("camera_object_id"),
                "type": obj.get("type"),
                "box_2d": obj.get("box_2d"),
                "derived_match_iou": assoc.get("derived_match_iou"),
                "lidar_box": lidar_box,
                "projected_lidar_box": projected_box,
            }
            segment_tracks.setdefault(track_key, []).append(instance)
            tracks_dir.mkdir(parents=True, exist_ok=True)
            track_path = tracks_dir / ("%s.json" % track_key)
            with open(track_path, "w", encoding="utf-8") as f:
                json.dump(segment_tracks[track_key], f, indent=2)
            if track_path not in written:
                written.append(track_path)


def apply_degradations_and_save(
    images: List[Dict[str, Any]],
    output_dir: str,
    augmenter: ImageAugmenter,
    metadata_repo: FrameMetadataRepository,
    image_format: str = "JPEG",
    verbose: bool = False,
) -> List[Path]:
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

    current_segment: Optional[str] = None
    scene_meta_dir: Path = out_dir
    segment_tracks: Dict[str, List[Dict[str, Any]]] = {}
    camera_counters: Dict[str, int] = {}

    for _, record in indexed_images:
        segment = str(record.get("segment_context_name", "unknown"))
        timestamp = int(record.get("timestamp_micros", 0))

        if segment != current_segment:
            if current_segment is not None:
                segment_tracks = {}
            segment_dir = out_dir / ("segment_" + segment)
            scene_meta_dir = segment_dir / "metadata" / "scene_metadata"
            scene_meta_dir.mkdir(parents=True, exist_ok=True)
            current_segment = segment
            camera_counters = {}
        camera_name = str(record.get("camera_name", "unknown"))
        frame_key = build_frame_key(segment, timestamp)
        scene_name = build_scene_name(segment, timestamp)

        try:
            pil_img = decode_image(record["image_bytes"])
        except Exception as exc:
            print("[warn] Failed to decode image %s (%s): %s" % (scene_name, camera_name, exc))
            continue

        if pil_img.size != (1920, 1280):
            if verbose:
                print(
                    "[skip] Ignoring %s (%s) — size %dx%d is not 1920x1280"
                    % (scene_name, camera_name, pil_img.size[0], pil_img.size[1])
                )
            continue

        seq_idx = camera_counters.get(camera_name, 0)
        camera_counters[camera_name] = seq_idx + 1
        base_name = "%s_%03d" % (scene_name, seq_idx)

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

        camera_subdir = "camera_%s" % camera_name
        try:
            clear_dir = segment_dir / "images" / camera_subdir / "clear"
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

            image_meta_path = (
                segment_dir / "metadata" / "image_metadata" / camera_subdir / ("%s.json" % base_name)
            )
            image_meta_path.parent.mkdir(parents=True, exist_ok=True)
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
                "Objects": _json_safe(
                    current_camera_objects_with_lidar_by_camera.get(camera_name, [])
                ),
            }
            image_metadata = _compact_mapping(image_metadata)
            with open(image_meta_path, "w", encoding="utf-8") as f:
                json.dump(image_metadata, f, indent=2)
            written.append(image_meta_path)
        except Exception as exc:
            print("[warn] Failed to save clear image/metadata for %s: %s" % (base_name, exc))

        enriched_objects = _json_safe(
            current_camera_objects_with_lidar_by_camera.get(camera_name, [])
        )
        if isinstance(enriched_objects, list):
            tracks_dir = segment_dir / "metadata" / "tracks" / camera_subdir
            _write_track_instances(
                enriched_objects, camera_name, timestamp,
                segment_tracks, tracks_dir, written,
            )

        aug_map = augmenter.apply_all(pil_img)
        for aug_name, aug_img in aug_map.items():
            dest_dir = segment_dir / "images" / camera_subdir / aug_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            img_path = dest_dir / ("%s.%s" % (base_name, image_format.lower()))
            img_path.write_bytes(encode_image(aug_img, format=image_format))
            written.append(img_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HiPerGator: fetch Waymo images from a local mirror, augment, write metadata."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Local dataset root. Accepts .../dataset or .../dataset/waymo_open_dataset_v_2_0_1",
    )
    parser.add_argument("--output-dir", required=True, help="Output folder (created if missing).")
    parser.add_argument("--split", default="training", help="Dataset split (training/validation/test).")
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Optional cap on images processed (after shard slicing). Default: unlimited.",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for augmentations.")
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based index of this shard (SLURM_ARRAY_TASK_ID).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards (SLURM_ARRAY_TASK_COUNT).",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    dataset_base_path = resolve_local_dataset_base(Path(args.dataset_root).resolve())
    split_camera_path = dataset_base_path / args.split / "camera_image"
    if not split_camera_path.exists():
        print("[error] Local dataset not found at %s" % split_camera_path)
        sys.exit(2)
    dataset_base = str(dataset_base_path)
    print("[info] Data source: %s" % dataset_base)
    print("[info] Split: %s" % args.split)
    print("[info] Output dir: %s" % args.output_dir)
    print("[info] Shard: %d/%d" % (args.shard_index, args.num_shards))

    images = fetch_camera_images_sharded(
        dataset_base=dataset_base,
        split=args.split,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        verbose=args.verbose,
    )
    if not images:
        print("[warn] No images selected for this shard. Exiting cleanly.")
        return
    print("[info] Fetched %d images for shard." % len(images))

    if args.count is not None:
        images = images[: args.count]
        print("[info] Capped to %d images (--count)." % len(images))

    frame_keys_df = build_frame_keys_dataframe_from_records(images)
    if args.verbose:
        print("[debug] Unique frames in shard: %d" % len(frame_keys_df))

    component_tables = load_filtered_component_tables(
        dataset_base=dataset_base,
        split=args.split,
        components=METADATA_COMPONENTS,
        frame_keys_df=frame_keys_df,
        max_files_per_component=None,
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

    print("[success] Shard %d/%d wrote %d files under %s"
          % (args.shard_index, args.num_shards, len(paths), args.output_dir))


if __name__ == "__main__":
    main()
