# Running the perception baselines on HiPerGator

**Audience:** whoever has HiPerGator deploy access. You should not need to read
any Python to run this.

**The shape of it:** submit a cheap smoke test → read one JSON verdict → only
then submit the expensive full training. The smoke test exists so nobody burns
72 GPU-hours discovering that an image path was wrong.

---

## TL;DR

```bash
cd /blue/<GROUP>/<USER>/visuals-dataset-generation/visuals-ml/hipergator
# edit the <GROUP>/<USER> paths in the four .sbatch files first — see Setup
./submit_baselines.sh smoke
```

Wait for it to finish (~30–60 min including queue), then send back:

```
visuals-ml/reports/smoke_report.json
```

That file is the deliverable. It is machine-readable — hand it to an agent, or
to us, and the decision is "does `verdict` say PASS". If yes:

```bash
./submit_baselines.sh full
```

---

## What is being trained

Three monocular perception baselines on the weather-augmented Waymo dataset.
None is novel — that is deliberate. They are reference points.

| # | Model | Task | Given | Predicts |
|---|-------|------|-------|----------|
| **B** | `box3d` | Amodal 3D box regression | the GT 2D box | 3D centre, dimensions, yaw |
| **A** | `monodetr` | Monocular 3D detection | image only | finds objects **and** their 3D boxes |
| **C** | `velocity` | Per-object velocity | two consecutive crops + ego-motion | vx, vy |

B is the lower bound for A: same labels, same index, but B is handed the 2D box
while A has to find it. That comparison is the point.

Everything is evaluated **per weather variant** (clear, rain, fog, snow, frost,
sunglare, brightness, wildfire_smoke, dust, waterdrop). Weather is the test axis.

---

## Setup (once)

Each of the four `.sbatch` files has **two variables** at the top; every path
derives from them:

```bash
GROUP="${GROUP:-CHANGEME_GROUP}"        # your allocation group
USERDIR="${USERDIR:-CHANGEME_USER}"     # your directory under /blue/$GROUP
```

Either edit them in place, or leave them and export instead — they honour the
environment, so this works without touching the files:

```bash
export GROUP=yourgroup USERDIR=yourusername
./submit_baselines.sh smoke
```

Each script exits with a clear error if the placeholders are still set, rather
than failing deep inside a container with a confusing path error.

**Still needs manual editing:** the `#SBATCH --account=` and `--qos=` lines.
SLURM directives are comments to the shell and cannot use variables, so those
have to be set literally in all four files.

If the derived paths don't match your layout, override them directly:

```bash
REPO_ROOT=/blue/${GROUP}/${USERDIR}/visuals-dataset-generation
DATASET_OUTPUT=/blue/${GROUP}/${USERDIR}/waymo/output   # generated visuals tree
SIF=/blue/${GROUP}/${USERDIR}/waymo/train.sif
```

**Prerequisites**

1. The generated dataset tree on `/blue`, laid out as
   `<output>/segment_*/images/camera_<N>/<weather>/<stem>.jpeg` with matching
   `<output>/segment_*/metadata/image_metadata/camera_<N>/<stem>.json`.
2. `train.sif` — the **PyTorch/CUDA** image (`hipergator/train.def`). This is a
   different image from the TensorFlow one used for dataset generation
   (`visuals-hipergator/hiker.def`). Build it if it does not exist:
   ```bash
   singularity build train.sif train.def
   ```
3. A GPU partition you can actually submit to.

---

## The pipeline

`submit_baselines.sh` wires the SLURM dependencies. You do not submit the stages
by hand.

```
smoke:  prepare_indexes(MODE=smoke)  ──afterok──▶  smoke_baselines[0-2]  ──afterany──▶  aggregate_smoke
            CPU, ~10 min                             3 GPUs in parallel, ~20 min          CPU, ~1 min

full:   prepare_indexes(MODE=full)   ──afterok──▶  prepare_indexes(MODE=full-eval)  ──afterok──▶  train_baselines[0-2]
            CPU, ~1-3 h                                CPU, ~2-4 h                                  3 GPUs, 24-72 h
```

The model stages are **array jobs**, one task per model, so the three baselines
occupy three GPUs simultaneously instead of queueing behind one another.

The aggregation step depends on `afterany`, not `afterok`, on purpose: a model
that fails its gates must still be collected into the report. Otherwise a
legitimate FAIL would look like an infrastructure problem.

---

## What the smoke test actually checks

Per model, in order — later stages are skipped when an earlier one fails,
because their results would be meaningless:

| Stage | Catches |
|---|---|
| `build` | model won't instantiate; missing vendored deps |
| `data_batches` | index empty, or everything filtered out |
| `train_step` | non-finite loss; **zero gradient norm** (nothing wired to the loss) |
| `overfit` | loss must drop to ≤60% of its start over 12 refits of 3 fixed batches — this is the real test; a model that can't fit data it has seen 12 times is broken, not undertrained |
| `evaluate` | eval crashes, or returns no/NaN `monitor` (the runner checkpoints on it) |
| `throughput` | measured samples/s, extrapolated to a full-run wall-clock estimate |

Plus environment checks (GPU visible **and computing**, deps importable) and data
checks (index non-empty, images actually readable, enough distinct segments to
form a leak-free split).

Nothing here trains to convergence. **No number the smoke test prints belongs in
a paper.**

---

## Reading the verdict

```json
{
  "verdict": "PASS",
  "summary": { "passed": 34, "failed": 0, "skipped": 0, "warned": 1,
               "failures": [], "warnings": ["models.monodetr.evaluate"] },
  "models": { "box3d": {...}, "monodetr": {...}, "velocity": {...} }
}
```

- **`verdict: PASS`** → submit the full run.
- **`verdict: FAIL`** → read `summary.failures`; each failed stage carries a
  `reason` written to be actionable.
- **`warned`** does not block. The expected warning is
  `models.monodetr.evaluate`: an untrained detector detects nothing, so its
  `monitor` is `+inf`. That is the *correct* score for an empty detector and the
  `overfit` stage already proved the loss descends.
- **`missing_reports` non-empty** → an array task died before writing. Check
  `logs/smoke_<jobid>_<task>.err`.

### Two numbers worth looking at even on a PASS

**`models.velocity.evaluate.metrics.skill_vs_zero`** — velocity's labels are
Waymo's `speed` field, which was ~72% near-zero in our sample. If this value is
**≤ 0** the model has not beaten always-predicting-zero, and baseline C is
measuring nothing. Do not spend GPU-days on it; it needs its labels re-derived
from track differences first. This is the cheapest way to settle the question,
which is why it is wired into the smoke test.

**`models.*.full_run_estimate.hours`** — measured throughput extrapolated to the
full run. If it exceeds the `--time` budget in `train_baselines.sbatch`, raise
the walltime or cut `epochs` *before* submitting, not after a timeout kills it
at hour 71.

---

## About the train/val split — important

**The generated dataset is the Waymo `training` split only.** Both
`generate_dataset_hipergator.py` (`--split`, default `training`) and
`submit_hipergator.sbatch` (`SPLIT=training`) process one split per run, and
~800 segments is the whole Waymo training split. No validation or test data was
generated.

Two consequences:

1. **We make our own split**, at **segment** granularity, inside the harness
   (`group_by: segment`). This is not optional. Every frame is re-rendered under
   10 weathers and frames arrive at ~10 Hz, so splitting on individual records
   would put the same object — sometimes near-identical pixels — on both sides.
   Validation numbers would be fiction.
2. Waymo's `testing` split has **no public labels** (it is leaderboard-only), so
   it is useless to us regardless. If we ever want a genuinely held-out set, the
   thing to generate is the **`validation`** split — a separate generation run
   with `SPLIT=validation`. Worth doing eventually; not needed to start.

---

## Full run: what to expect

- `box3d` — cheapest. Trains on `clear` only; evaluated on all 10 variants.
- `monodetr` — the heavy one (37.4M params). Watch that **recall leaves 0**
  within the first few epochs. If it is still 0 after ~5 epochs, stop the job:
  the decode or `score_thresh` is wrong, and more epochs will not fix it.
- `velocity` — only worth running if `skill_vs_zero > 0` in the smoke report.

Training uses the `clear`-only index; evaluation uses the all-weather one. The
10 variants are pixel-wise different but geometrically identical, so training on
all of them multiplies compute without adding geometric information — and
weather is what we are trying to *measure*, not augment with.

Results land in `visuals-ml/reports/final_<model>.json`, with the per-weather and
per-depth-band breakdown.

---

## Troubleshooting

**`torch.cuda.is_available() is False`** — the job did not get a GPU, or
`singularity exec` ran without `--nv`. Check `--partition=gpu` and `--gpus=1`.

**`only N image(s) readable`** — image paths in the index are absolute. An index
built on a different filesystem will not resolve. Rebuild it in place
(`prepare_indexes.sbatch`).

**`only 1 segment(s); need >= 2`** — the index covers too little data for a
segment-level split. Raise `--max-segments` in `prepare_indexes.sbatch`.

**`loss fell to 0.95x of its start`** — the overfit gate. Something is wrong with
the targets or the decode; more epochs will not help.

**CUDA out of memory** — lower `batch_size` in `baselines/configs/smoke_deploy.yaml`
(smoke) or the model's config in `configs/` (full). `monodetr` at
`batch_size: 2` is already conservative.

**Shared-memory / dataloader worker crashes** — set `num_workers: 0`. The smoke
config already does, so that faults are attributable to the model and not the
loader.
