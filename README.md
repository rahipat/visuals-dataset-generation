# visuals-dataset-generation
Outdated, TODO: update soon.


Dataset generation pipeline for Waymo Open Dataset camera images with HiKER-SGG image alterations and compact metadata export.

## Dataset Location

Place the local mirrored dataset here:

`visuals-dataset-generation/dataset/`


The generator defaults to this local path. Use `-cloud` to read directly from GCS instead (requires Google Cloud credentials read from default path).

## Docker Requirement

The augmentation stack depends on system libraries (ImageMagick/X11/OpenGL). Use Docker for consistent runtime:

1. Install Docker Desktop.
2. Build/run through scripts in `visuals_dataset/`.

## Windows Instructions

### 1) Cache a local subset

From repo root:

```powershell
cd .\visuals_dataset
python .\cache_waymo_subset.py --split training --max-camera-files 10 --output-root ..\dataset --verbose
```

### 2) Generate dataset from local cache (default mode)

```powershell
python .\visuals-dataset\generate_dataset.py --split training --count 50 --max-files 10 --output-dir output --verbose
```

### 3) Generate dataset with Docker (recommended)

```powershell
.\visuals-dataset\run_docker.ps1 --split training --count 50 --max-files 10 --output-dir output --verbose
```

### 4) Cloud mode (skip local cache)

```powershell
.\visuals-dataset\run_docker.ps1 -cloud --split training --count 50 --max-files 10 --output-dir output --verbose
```

### 5) Optional scene re-association pass

```powershell
python .\visuals-dataset\associate_lidar_scene.py --scene-metadata output\metadata\scene_metadata\<scene>.json --image-metadata-dir output\metadata\image_metadata --verbose
```

## Key Generator Arguments

`visuals_dataset/generate_dataset.py`:

- `--split`: dataset split (`training`, `validation`, `test`)
- `--count`: max images to process
- `--max-files`: max `camera_image` parquet files scanned
- `--output-dir`: output folder (contains image folders + metadata)
- `--dataset-root`: local root (`dataset` by default at repo root)
- `-cloud` / `--cloud`: read from GCS instead of local cache
- `--metadata-max-files`: optional cap per metadata component
- `--seed`: augmentation RNG seed
- `--random-start`: shuffle seed before selecting `--count`
- `--verbose`: verbose logs

## Output Layout

- Images:
  - `output/clear/`
  - `output/rain/`, `output/fog/`, `output/snow/`, `output/frost/`, `output/sunglare/`, `output/brightness/`, `output/wildfire_smoke/`, `output/dust/`, `output/waterdrop/`
- Metadata:
  - `output/metadata/scene_metadata/` (scene/frame-level, once per frame)
  - `output/metadata/image_metadata/` (camera-image-level, compact object list)

## macOS / Linux Instructions

TODO
