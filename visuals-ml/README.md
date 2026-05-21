# visuals-ml

3D position prediction model for Waymo Open Dataset camera images. Given a full camera image and its 2D bounding box + camera intrinsics, predicts the object's 3D center position in the ego-vehicle frame (metres).

## Architecture

Two-stream ResNet18: one branch sees the full scene (640×960), the other sees
a padded crop of the 2D bounding box at fixed resolution (224×224). Both 512-d
features are concatenated with the 8-d coordinate vector and fed to an MLP head.

**Input:**
- Full camera image (resized to 640×960)
- Padded box crop from the native-resolution image (10% padding, resized to 224×224)
- `[cx_n, cy_n, sw_n, sh_n, fu, fv, cu, cv]` — normalized 2D box center, normalized box size, camera focal lengths, camera principal point

**Output:** `[x, y, z]` in ego-vehicle frame (metres)

The crop stream gives the head direct access to object-level features (textures,
parts, sub-object structure) at high resolution, which the global-pooled full
image would otherwise lose. Apparent size of the object is recovered through
`sw_n, sh_n, fu, fv` in the coord vector.

## Setup

```bash
pip install torch torchvision pyyaml pillow
```

Requires a CUDA-capable GPU. Install the CUDA torch build if needed:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

## Workflow

### 1. Build the index

Walks `visuals_dataset/output` and writes a flat `records.jsonl` — one line per matched object per weather variant.

```bash
python data/build_index.py --source-dir ../visuals_dataset/output
```

Output: `data/output/records.jsonl`

### 2. Train

```bash
python train.py --config configs/local.yaml
```

Saves the best checkpoint (by val loss) to `checkpoints/best.pt`. To resume from a checkpoint:

```bash
python train.py --config configs/local.yaml --resume checkpoints/best.pt
```

### 3. Evaluate

```bash
python eval.py
```

Reports MAE per axis (x, y, z) on the validation split.

## Configuration

Edit `configs/local.yaml`:

| Key | Default | Description |
|---|---|---|
| `index_file` | `data/output/records.jsonl` | Path to built index |
| `batch_size` | `32` | Batch size |
| `num_workers` | `6` | DataLoader workers (set 0 on Windows if issues) |
| `val_split` | `0.2` | Fraction held out for validation |
| `lr` | `1e-4` | Adam learning rate |
| `epochs` | `30` | Training epochs |
| `checkpoint_dir` | `checkpoints/` | Where to save checkpoints |

## Notes

- Only objects with a matched LiDAR association are used for training (ground truth 3D position required)
- Each source image has 10 weather variants (clear, rain, fog, snow, frost, sunglare, brightness, wildfire_smoke, dust, waterdrop) — all 10 are used as separate training samples with the same metadata
- `x` is forward depth, `y` is lateral, `z` is vertical in the Waymo ego-vehicle frame
- `torch.compile` is not supported on Windows — use HiPerGator (Linux) for that optimization
