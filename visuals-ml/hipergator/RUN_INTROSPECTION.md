# Running the Introspective Perception baseline on HiPerGator

This is a self-contained runbook for training and evaluating **Baseline 1
(Introspective Perception, Daftry et al. 2016)** on HiPerGator. It is written so
it can be **pasted whole into Claude** as context if you hit a problem — it
includes the system's shape, the exact commands, expected output, and a
troubleshooting guide.

---

## 0. Context (read this first, or paste it into Claude)

**What the baseline does.** It does *not* detect or locate objects. It is a
*failure predictor*: given a camera frame, it outputs a single score in `[0, 1]`
for how likely the underlying perception model is to be wrong on that frame. Higher
= less reliable. This is Paper 1's "introspection" idea, reimplemented from the
paper (no official code exists).

**The systems involved.**
- **Visuals dataset** — Waymo driving-camera frames, each re-rendered under 10
  weather variants (`clear, rain, fog, snow, frost, sunglare, brightness,
  wildfire_smoke, dust, waterdrop`). Frames are time-ordered within a segment;
  vehicles carry LiDAR-matched 3D ground truth. Layout on disk:
  `<output>/segment_*/images/camera_<N>/<weather>/<stem>.jpeg` and matching
  metadata under `<output>/segment_*/metadata/image_metadata/camera_<N>/<stem>.json`.
- **PositionNet** — the perception model *under test*. A two-stream ResNet18 that
  predicts an object's 3D position `(x, y, z)` in metres. It is imperfect,
  especially in bad weather. The introspection baseline learns to predict *its*
  failures. It is the fixed substrate; a trained PositionNet checkpoint is a
  prerequisite.
- **The harness** — `visuals-ml/baselines/`. A model-agnostic runner trains,
  evaluates, and checkpoints any model implementing the `BaselineModel` interface,
  reporting metrics per weather variant.

**The pipeline (5 stages).**
1. `build_introspection_index.py` — walk the dataset → one record per
   `(frame, weather)` with the image, its recent neighbour frames (for optical
   flow), and the frame's ground-truth objects. *Model-free.*
2. `introspection_label_prepass.py` — run trained PositionNet on each object →
   per-frame failure label `fail_frac` (fraction of objects with 3D error > `tau`
   m) and binary `fail`.
3. `introspection_dataset.py` — each sample = a spatial view (RGB frame) + a
   temporal view (stacked optical flow between recent frames; TV-L1, disk-cached).
4. `models/introspection.py` — two-stream AlexNet (spatial + temporal) → fused
   `fc8` feature → softmax head; trained with the standard runner.
5. `introspection_svm.py` — freeze the CNN, take `fc8` features, fit a linear SVM
   that regresses the failure score; report Error-vs-Failure-Rate (the paper's
   Risk-Averse Metric), AUROC, and a per-weather breakdown.

**Key design notes a troubleshooter must know (do not "fix" these — they're
intentional):**
- The baseline is **black-box** over PositionNet: it only uses PositionNet's
  *error* as a training label, never its internals.
- The failure threshold `tau` should be set with `--tau-percentile`, not a fixed
  metre value. With a weak PositionNet a fixed `tau` labels *every* frame a
  failure (no negative class). Percentile-derived `tau` guarantees a usable split.
- **Weather caveat (flagged, unresolved):** weather augmentations are synthetic and
  per-frame independent, so optical flow can mix real motion with augmentation
  flicker. The baseline runs; whether the temporal stream is meaningful on this
  data is an open question, not a bug.

**Faithfulness deviations from the paper:** temporal stream conv1 is
channel-inflated from ImageNet weights (paper uses UCF-101/HMDB-51 flow
pretraining, unavailable); AlexNet input is 224 not 227; TV-L1 flow requires
`opencv-contrib-python` (in the container) else Farneback fallback.

---

## 1. Prerequisites

1. **The visuals dataset output tree** on `/blue` (the generated per-segment
   images + metadata described above). Generate it with the pipeline in
   `visuals-hipergator/` if you don't have it yet.
2. **A trained PositionNet checkpoint.** Train it first with the existing
   `positionnet` baseline:
   ```bash
   # inside the container, from visuals-ml/
   python -m data.build_index --source-dir <DATASET_OUTPUT>
   python -m baselines.train --config configs/positionnet.yaml
   # -> baselines/checkpoints/positionnet/best.pt
   ```
   Note the module path: the PositionNet index builder is `data.build_index`
   (top-level `data/`), *not* `baselines.data.build_index` — the latter does not
   exist. On HiPerGator you can run both steps as a job with
   `sbatch train_positionnet.sbatch`.
   The label pre-pass accepts either a harness checkpoint (keys under `core.`) or
   a bare PositionNet state dict.
3. **The training Singularity image** (`train.sif`) — build once (step 2).

---

## 2. Build the Singularity image

The image is PyTorch + CUDA. It **bakes the AlexNet, ResNet18 and ResNet50
pretrained weights inside** (compute nodes have no internet) and **compiles
MonoDETR's MultiScaleDeformableAttention CUDA op**. Build it where you have
`singularity`/`apptainer` with fakeroot — usually a HiPerGator login node or a
dedicated build session.

The build needs the MonoDETR ops sources bind-mounted onto `/mnt`:

```bash
cd /blue/$GROUP/$HPG_USER/visuals-dataset-generation/visuals-ml/hipergator
OPS=$(cd ../baselines/vendor/monodetr/lib/models/monodetr/ops && pwd)
singularity build --fakeroot --bind "$OPS:/mnt:ro" train.sif train.def
# then move/keep train.sif wherever the sbatch's SIF= points
```

The bind source must be an **absolute** path, which is why `OPS` is computed with
`pwd` rather than written as a relative path. The mount is read-only: `%post`
copies the sources to `/tmp/ops` before patching, so the build never writes back
into your checkout.

**Why no `%files`.** `%files` is unreliable under `--fakeroot` on HiPerGator — it
can log a successful copy while leaving the destination empty, surfacing later as
a confusing `No such file or directory`. `train.def` therefore has no `%files`
section at all: the pip requirements are inlined as a heredoc in `%post`, and the
ops directory arrives via the bind mount above. Each step verifies its own inputs
and aborts with an actionable message instead of failing silently.

`requirements-ml.txt` is **no longer read by the build** — it stays in the repo for
bare (non-container) pip installs. Edit both it and the heredoc in `train.def` if
you change dependencies.

If your site disables user bind mounts at build time (`allow user bind = no` in
`singularity.conf`), use the `%setup` fallback documented in the `train.def`
header — `%setup` runs on the host and copies straight into the image rootfs,
needing neither `%files` nor `--bind`.

Expected build output includes `[info] python deps OK`, `[info] cv2.optflow OK`,
`[info] staged 14 ops files from /mnt`, and `[info] patched ops/setup.py`.

---

## 3. Configure and submit

The sbatch scripts read their paths from the environment. `GROUP` and `HPG_USER`
are required — leave either unset and the job aborts immediately with
`GROUP: set me - your HiPerGator group ...` rather than a confusing error:

```bash
export GROUP=<your-group>
export HPG_USER=$USER          # named HPG_USER because the shell always sets USER
```

Everything else has a default derived from those two, and can be overridden the
same way if your layout differs:

| Variable | Default |
|---|---|
| `REPO_ROOT` | `/blue/$GROUP/$HPG_USER/visuals-dataset-generation` |
| `DATASET_OUTPUT` | `/blue/$GROUP/$HPG_USER/waymo/output` (the visuals output tree) |
| `SIF` | `/blue/$GROUP/$HPG_USER/waymo/train.sif` |
| `POSITIONNET_CKPT` | `baselines/checkpoints/positionnet/best.pt` (relative to `visuals-ml/`) |
| `CONFIG` | `configs/introspection.yaml` |
| `CAMERAS` | `1` |

Submit (`sbatch` forwards your environment to the job by default):
```bash
cd "$REPO_ROOT"/visuals-ml/hipergator
sbatch --account=$GROUP --qos=$GROUP train_introspection.sbatch
```

The `--account` / `--qos` flags go on the command line because `#SBATCH` lines are
parsed by SLURM before the shell runs and get no variable expansion; the in-file
`<GROUP>` placeholders on those two directives are the alternative to edit by hand.
Set `--partition` too if your GPU partition differs.

The job runs the whole chain inside the container: build index + labels (skipped
if `data/output/introspection_labeled.jsonl` already exists) → train the CNN → fit
the SVM stage. Logs go to `hipergator/logs/introspection_<jobid>.out`.

### Running the stages manually (useful for debugging)

From `visuals-ml/` inside the container
(`singularity exec --nv --bind /blue/<GROUP> <SIF> bash`):

```bash
python -m baselines.data.build_introspection_index \
    --source-dir <DATASET_OUTPUT> \
    --index-file data/output/introspection_records.jsonl \
    --cameras 1 --flow-stack 5

python -m baselines.data.introspection_label_prepass \
    --index-file data/output/introspection_records.jsonl \
    --checkpoint baselines/checkpoints/positionnet/best.pt \
    --out-file data/output/introspection_labeled.jsonl \
    --tau-percentile 50 --fail-thresh 0.5

python -m baselines.train --config configs/introspection.yaml
python -m baselines.introspection_svm --config configs/introspection.yaml
```

---

## 4. What success looks like

| Stage | Expected output |
|---|---|
| index | `Wrote N frame records (M object instances) to data/output/introspection_records.jsonl` |
| pre-pass | `tau set to P50 of per-object error = X.XX m` then `Labeled N frames ... (k/N failures ...)` with **k strictly between 0 and N** (not all failures) |
| train | per-epoch lines `Epoch e/E train_loss=… monitor=… auroc=…` and `--> checkpoint saved` when AUROC improves; checkpoint at `baselines/checkpoints/introspection/best.pt` |
| SVM | `AUROC (failure prediction): …`, `EFR @FR=0.0/0.3/0.5: … m`, report at `baselines/checkpoints/introspection/introspection_svm_report.json` |

The final JSON report holds: `auroc`, the Error-vs-Failure-Rate curve
(`efr_introspection` vs the `efr_ideal` oracle and `efr_random` baseline), and
`per_weather` mean predicted-failure vs mean underlying error.

---

## 5. Config reference (`configs/introspection.yaml`)

| Key | Meaning |
|---|---|
| `index_file` | labeled index from the pre-pass |
| `resolution` | AlexNet input S×S (224) |
| `flow_stack` | number of optical-flow fields (temporal channels = 2×this) |
| `feat_dim` | fused `fc8` feature size |
| `flow_cache` | dir for cached `.npy` flow stacks (pre-warms on first pass) |
| `batch_size` | lower this on GPU OOM |
| `num_workers` | dataloader workers (lower on shared-memory errors) |
| `val_split`, `seed` | deterministic split |
| `lr`, `epochs` | CNN training |
| `checkpoint_dir` | where `best.pt` + the SVM report land |
| `max_samples` | (optional) cap dataset for a fast smoke run |

---

## 6. Troubleshooting

**Pre-pass reports every frame as a failure (`N/N failures`).**
The base PositionNet is weak relative to the fixed `tau`, so the binary label
saturates and the CNN has no negative class. Use `--tau-percentile 50` (already the
default in the sbatch). If it persists, the PositionNet checkpoint may be
mistrained — sanity-check `mean_err` values in the labeled jsonl (should span a
range, not all be huge).

**`No PositionNet weights found in <ckpt>`.**
The checkpoint isn't a PositionNet. The loader accepts a harness checkpoint (keys
under `core.`) or a bare PositionNet state dict (keys like `encoder_full.*`,
`head.*`). Point `POSITIONNET_CKPT` at the right file.

**Index is empty / `Wrote 0 frame records`.**
No matched objects found. Check: `--cameras` matches folders that exist
(`camera_1` etc.); the metadata has `Objects[].lidar_association.status == "matched"`;
`--source-dir` points at the segment-level output tree (the parent of
`segment_*/`).

**`alexnet ... download` / connection error at runtime.**
Compute node has no internet and the SIF wasn't built with baked weights. Rebuild
`train.sif` from the current `train.def` (its `%post` caches the weights at
`/opt/torch-cache` and `%environment` sets `TORCH_HOME` there).

**CUDA out of memory.**
Lower `batch_size` in the config (try 16 or 8). The two AlexNet streams at 224 are
modest, so OOM usually means a small GPU or a large `batch_size`.

**First epoch is very slow, later epochs fast.**
Expected: TV-L1 optical flow is computed and cached to `flow_cache` on first touch.
To pre-warm without training, run one pass (e.g. a `max_samples` smoke) first, or
just let epoch 1 populate the cache.

**`scikit-learn is required for the SVM stage`.**
Only happens outside the container. Inside `train.sif` it's installed. If running
bare, `pip install scikit-learn`.

**`DataLoader worker ... shared memory` / bus errors.**
Lower `num_workers` (e.g. to 2 or 0). HiPerGator containers sometimes have small
`/dev/shm`; you can also add `--bind /dev/shm` sizing per your site's guidance.

**GPU not used / runs on CPU.**
Ensure `--nv` is on the `singularity exec` line (it is in the sbatch) and the job
requested a GPU (`#SBATCH --gpus=1`, correct `--partition`).

**Flow uses Farneback, not TV-L1.**
`opencv-contrib-python` isn't present (the base `opencv-python` lacks `cv2.optflow`).
The container installs the contrib build; rebuild the SIF if you see the fallback.

---

## 7. File map

```
visuals-ml/
  baselines/
    data/
      build_introspection_index.py     # stage 1: per-frame index + neighbours + GT
      introspection_label_prepass.py   # stage 2: PositionNet -> failure labels
      introspection_dataset.py         # stage 3: spatial + optical-flow inputs
    models/
      introspection.py                 # stage 4: two-stream CNN (BaselineModel)
    introspection_svm.py               # stage 5: fc8 -> linear SVM + EFR/AUROC report
    README.md                          # "Introspective Perception" section
  configs/
    introspection.yaml                 # config (model: introspection)
  hipergator/
    train.def                          # PyTorch/CUDA Singularity image (bakes weights,
                                       #   builds MonoDETR's MSDeformAttn CUDA op)
    requirements-ml.txt                # bare-metal pip deps; NOT read by train.def
                                       #   (mirrored as a heredoc in its %post)
    train_positionnet.sbatch           # SLURM job: PositionNet (introspection's prereq)
    train_monodetr.sbatch              # SLURM job: MonoDETR
    train_introspection.sbatch         # SLURM job: full chain
    RUN_INTROSPECTION.md               # this file
```

Papers: Introspective Perception — arXiv 1607.08665. Underlying model: PositionNet
(`visuals-ml/model/position_net.py`). Harness contract: `baselines/core/interface.py`.
