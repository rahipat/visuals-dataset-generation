# baselines

A model-agnostic harness for running paper baselines on the visuals (weather-altered
Waymo) dataset. Define a model once against a small interface, then train / evaluate it
locally and on HiPerGator through the same runner, with per-weather-variant metrics.

## Layout

```
baselines/
  core/
    interface.py   # BaselineModel — the contract every baseline implements
    registry.py    # @register_model("name") + build_model(cfg)
    runner.py      # model-independent train/eval loop, AMP, checkpointing
    boxes.py       # shared Box3D (camera optical frame) for detection baselines
    geometry.py    # Waymo vehicle frame -> camera optical frame labels
    iou3d.py       # rotated 3D / BEV IoU (no external deps)
    utils.py       # config loading, record split + leak-free group split
  models/
    position_net.py  # PositionNet (per-object 3D-center regressor) — reference baseline
    box3d_net.py     # Baseline B — amodal 3D box (centre + dims + yaw) regressor
    monodetr/        # MonoDETR (per-image detector) — added in milestone 2
    introspection.py # Introspective Perception (Paper 1) — failure predictor over PositionNet
  vendor/
    monodetr/        # vendored upstream MonoDETR (MIT) — see vendor/monodetr/VENDOR.md
  configs/
  train.py / eval.py # thin entrypoints; pick the model from config `model:` key
```

## Adding a baseline

Implement `BaselineModel` (see `core/interface.py`) and register it:

```python
@register_model("mymodel")
class MyBaseline(BaselineModel):
    @staticmethod
    def build_datasets(cfg): ...      # -> (train_ds, val_ds)
    def collate_fn(self, batch): ...  # override for variable-length detection batches
    def training_step(self, batch, device): ...  # -> (loss, log_dict)
    def evaluate(self, loader, device): ...       # -> metrics dict incl. float "monitor"
```

Then add `from baselines.models import mymodel` to `models/__init__.py` and a config with
`model: mymodel`.

**Contract notes**
- `evaluate()` must return a dict with a float `"monitor"` key where **lower is better**
  (for higher-is-better metrics like AP, return the negative). The runner checkpoints
  whenever `monitor` improves.
- `training_step` runs inside the runner's autocast context — return the loss, do **not**
  call `backward()`.

## Usage

Run from the `visuals-ml/` directory.

```bash
# train (model selected by the config's `model:` key)
python -m baselines.train --config configs/positionnet.yaml

# evaluate a checkpoint (defaults to <checkpoint_dir>/best.pt)
python -m baselines.eval --config configs/positionnet.yaml
```

Useful config keys: `max_samples` caps the dataset for fast local smoke runs;
`num_workers: 0` avoids Windows dataloader issues. See `configs/positionnet.yaml`.

## Baseline B — amodal 3D box regression (`box3d`)

Given the ground-truth 2D box of a vehicle, recover its full 3D box — centre
(x, y, z), dimensions (h, w, l) and yaw — in the camera optical frame. Detection
is *not* part of the task, which makes B the clean lower bound for MonoDETR: the
detector has to solve this **and** find the objects. Both read the same per-image
index, so they are scored against byte-identical labels.

```bash
# 1. Build the shared index (also used by MonoDETR).
#    The full 800-segment tree is ~4.8M images -- use the sampling flags.
python -m baselines.data.build_detection_index \
    --cameras 1 --weathers clear --frame-stride 5

# 2. Train / evaluate
python -m baselines.train --config configs/box3d.yaml
python -m baselines.eval  --config configs/box3d.yaml
```

Reports per-object **3D IoU** (mean, and the fraction above 0.25/0.5/0.7),
depth MAE and AbsRel, centre MAE, dimension MAE and yaw error — each broken down
per weather variant and per depth band (0-20 / 20-40 / 40+ m).

### Why 3D IoU is the metric that matters here

It is the failure signal introspection should consume. The Paper-1 label pre-pass
currently thresholds 3D **centre distance** in metres, which is not comparable
across depth (2 m of error is nothing at 70 m and catastrophic at 5 m) — hence
the `--tau-percentile` workaround for the saturating label. `iou_3d < 0.5` is
bounded, scale-free and standard, and needs no calibration.

### Three data properties this baseline is built around

1. **Split on segments, never on records.** Every frame is re-rendered under 10
   weathers and frames arrive at ~10 Hz, so a random record split puts the same
   physical object — sometimes the same *pixel content* — on both sides. Use
   `group_by: segment` (the default). `group_by: frame` removes only the 10x
   weather duplication and is for smoke runs on 1-2 segments; it warns.
2. **Truncated objects are dropped** (`drop_truncated: true`). When a 2D box is
   clipped by the frame border its centre stops matching the projected 3D
   centre. Measured on the sample: the projected 3D centre lands inside its own
   2D box for ~86% of whole objects but only 48-61% of truncated ones,
   consistently across cameras 1/2/3. KITTI carries the same attribute.
3. **Train on `clear`, evaluate on all 10.** The variants are pixel-wise
   different but geometrically identical, so training on all of them multiplies
   compute without adding geometric information — and weather is the *test*
   axis. `train_weathers: null` trains on everything in the index.

### Target encoding

`log(z)` for depth, `log(dim / anchor)` for dimensions, `sin/cos` for yaw — all
standard mono-3D practice, and all inverted exactly by `decode_targets()` so
metrics are in real metres and radians. The head's bias is initialised to the
mean depth / mean dimensions / yaw 0, which matters more than it sounds: without
it depth starts at `exp(0) ≈ 1 m` against data at ~25 m, and the first epochs
thrash over hundreds of metres (observed depth MAE swinging 4 m → 82 m).

Coordinate inputs are normalised by image size (`f_u/W`, `c_u/W`, ...). The
original PositionNet fed raw pixel intrinsics (`f_u ≈ 2060`) into an MLP
alongside inputs in [0, 1].

## Introspective Perception (Paper 1)

A faithful reimplementation of Daftry et al. 2016 (arXiv 1607.08665) — no official
code exists. Unlike the other baselines, this one is **not a perception model**: it
is a failure predictor that *wraps* a perception model. Here it wraps **PositionNet**
(the repo's 3D-position regressor) as the fixed substrate and learns, from the input
alone, whether PositionNet will fail on a frame.

Pipeline (the paper's multi-step recipe):

1. **Two-stream CNN** — an AlexNet-style spatial stream (RGB frame) and temporal
   stream (stacked optical flow) whose fc7 features fuse via fc8 into a deep feature.
2. Trained with a **softmax loss** on a binary failure label — this runs in the
   standard `baselines.train` runner.
3. A **linear SVM** regresses the continuous failure fraction from fc8 features →
   the final failure score, evaluated with the paper's **Error-vs-Failure-Rate**
   (Risk-Averse Metric) curve. This stage is `baselines.introspection_svm`.

### Workflow (from `visuals-ml/`)

```bash
# 1. Model-free per-frame index (+ temporal neighbours for the flow stream)
python -m baselines.data.build_introspection_index --cameras 1 --flow-stack 5

# 2. Label pre-pass: run a TRAINED PositionNet to derive per-frame failure labels.
#    --tau-percentile derives the miss threshold from the error distribution so the
#    binary label can't saturate (see below).
python -m baselines.data.introspection_label_prepass \
    --checkpoint baselines/checkpoints/positionnet/best.pt \
    --tau-percentile 50 --fail-thresh 0.5

# 3. Train the two-stream CNN (softmax)
python -m baselines.train --config configs/introspection.yaml

# 4. Fit the linear SVM stage → EFR/RAM + AUROC + per-weather report
python -m baselines.introspection_svm --config configs/introspection.yaml
```

On HiPerGator the whole chain runs as a Singularity job — see
`hipergator/train_introspection.sbatch` (+ `hipergator/train.def`, a PyTorch/CUDA
image distinct from the TensorFlow generation image in `visuals-hipergator/`).

### Label semantics

`fail_frac` = fraction of a frame's matched objects whose PositionNet 3D-centre
error exceeds `tau` metres (the SVM's continuous target). `fail` = `fail_frac >
fail_thresh` (the CNN's binary target). **Calibrate `tau` against the real trained
PositionNet**: with a weak/undertrained base model a fixed `tau` flags *every* frame
as a failure, leaving the softmax with no negative class. `--tau-percentile` avoids
this by construction.

### Documented deviations & caveats

- **No official code** → reimplemented from the paper; exact hyperparameters
  (SVM regularization, flow pretraining) are not recoverable.
- **Temporal stream** conv1 is channel-inflated from ImageNet weights instead of
  the paper's UCF-101/HMDB-51 flow pretraining (not available here). AlexNet input
  is 224 (torchvision pretrained) rather than the paper's 227.
- **Flow engine** is TV-L1 when `opencv-contrib-python` is present (the container),
  else Farneback.
- **Weather ⚠ (flagged for review):** the weather augmentations are synthetic and
  applied per-frame independently, so optical flow across augmented frames can mix
  real ego-motion with augmentation flicker. This is a known validity caveat of
  running the temporal stream on this dataset, raised for discussion, not resolved.
