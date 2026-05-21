# HiPerGator run-book

Files in this folder:

- `generate_dataset_hipergator.py` — entrypoint. Local-only (no GCS). Sharded for SLURM arrays.
- `hiker.def` — Apptainer/Singularity definition (built from `tensorflow/tensorflow:2.11.0`).
- `requirements.txt` — Python deps (mirror of `visuals_dataset/requirements.txt`).
- `transfer_waymo.sbatch` — SLURM job that pulls the full Waymo v2.0.1 bucket via rclone.
- `submit_hipergator.sbatch` — SLURM array submission script (placeholders for group/user/paths).

## 0. Before you start: placeholders to fill in

Both sbatch files have `<GROUP>` and `<USER>` placeholders. Before submitting
anything, replace them with your actual values. Find every occurrence with:

```bash
grep -n '<GROUP>\|<USER>' transfer_waymo.sbatch submit_hipergator.sbatch
```

What goes where:

| Placeholder | Example value | Used by |
| --- | --- | --- |
| `<GROUP>` | your UFRC group / billing account (e.g. `smithlab`) | `--account`, `--qos`, all `/blue/<GROUP>` paths |
| `<USER>` | your GatorLink username | `/blue/<GROUP>/<USER>/...` paths |

The path layout the scripts expect (create with `mkdir -p` if missing):

```
/blue/<GROUP>/<USER>/
├── visuals-dataset-generation/    # this repo (git clone destination)
└── waymo/
    ├── hiker.sif                  # built by step 1a
    ├── dataset/                   # rclone destination (step 2)
    └── output/                    # full-run output (step 4)
```

## 1. One-time setup on HiPerGator

### 1a. Clone repo + build Singularity image

```bash
# Clone repo into your /blue space
cd /blue/<GROUP>/<USER>
git clone <repo-url> visuals-dataset-generation
cd visuals-dataset-generation/visuals-hipergator

# Build the Singularity image on the login node (no sudo needed)
module load singularity
singularity build /blue/<GROUP>/<USER>/waymo/hiker.sif hiker.def
```

Build takes ~15–25 min. The resulting SIF is ~2–3 GB.

### 1b. Configure rclone remote for Waymo GCS bucket

The Waymo dataset requires authenticating as a Google account that registered
at <https://waymo.com/open> and accepted the license. OAuth needs a browser,
so do this from an **Open OnDemand Desktop** session, not a plain SSH login.

1. Open <https://ood.rc.ufl.edu/> → Interactive Apps → Desktop → launch.
2. In the desktop terminal:

   ```bash
   module load rclone
   rclone config
   ```

3. Walk through the prompts:
   - `n` (new remote)
   - name: `waymo`
   - type: `google cloud storage`
   - `client_id` / `client_secret`: leave blank (use rclone defaults)
   - `project_number`: leave blank
   - `service_account_file`: leave blank
   - `anonymous`: `false`
   - `object_acl` / `bucket_acl` / `location` / `storage_class`: leave blank
   - Edit advanced config: `n`
   - Use auto config: `y` (opens a browser tab — sign in with the Waymo-registered account)
   - Confirm and quit (`y`, then `q`)

4. Sanity check (works from any login node now):

   ```bash
   module load rclone
   rclone ls waymo:waymo_open_dataset_v_2_0_1/training/camera_box | head
   ```

   If you see parquet filenames, auth is working.

## 2. Transfer the dataset

### 2a. Pre-check total size (run on login node, ~1 min)

Before committing a 48 h SLURM job, confirm the actual bucket size:

```bash
module load rclone
rclone size waymo:waymo_open_dataset_v_2_0_1
```

This prints something like `Total objects: 12345 / Total size: 987.6 GB`.
If the size is wildly different from ~1 TB (the rough estimate), revisit
whether you actually want the entire bucket vs. just the 7 components the
pipeline reads (`camera_image`, `camera_box`, `camera_calibration`,
`lidar_box`, `lidar_calibration`, `stats`, `vehicle_pose`).

### 2b. Submit the transfer job

```bash
cd /blue/<GROUP>/<USER>/visuals-dataset-generation/visuals-hipergator
sbatch transfer_waymo.sbatch
squeue -u $USER
tail -f logs/transfer_*.out
```

The job runs `rclone copy gs://waymo_open_dataset_v_2_0_1 → /blue/.../dataset/waymo_open_dataset_v_2_0_1`
with 32 parallel transfers and 48 h wall time. Re-running is safe (rclone skips
files already present with matching size/mtime), so if the job times out or
dies, just `sbatch` it again.

### 2c. Verify completion

```bash
rclone size waymo:waymo_open_dataset_v_2_0_1
du -sh /blue/<GROUP>/<USER>/waymo/dataset/waymo_open_dataset_v_2_0_1
```

Sizes should match (within a few MB). Directory layout should look like:

```
dataset/waymo_open_dataset_v_2_0_1/
├── training/
│   ├── camera_image/*.parquet
│   ├── camera_box/*.parquet
│   └── ...
├── validation/
└── testing/
```

## 3. Smoke test (one shard, capped count)

```bash
cd /blue/<GROUP>/<USER>/visuals-dataset-generation/visuals-hipergator

module load singularity
singularity exec \
    --bind /blue/<GROUP> \
    --bind "$PWD/..":/workspace \
    /blue/<GROUP>/<USER>/waymo/hiker.sif \
    python /workspace/visuals-hipergator/generate_dataset_hipergator.py \
        --dataset-root /blue/<GROUP>/<USER>/waymo/dataset \
        --output-dir   /blue/<GROUP>/<USER>/waymo/output-smoke \
        --split training \
        --shard-index 0 --num-shards 50 \
        --count 20 --verbose
```

Verify `output-smoke/segment_*/images/camera_*/clear/*.jpeg` and
`output-smoke/segment_*/metadata/...` look correct.

## 4. Full run

Before submitting, check how many camera parquet files actually exist so you
can right-size `--array`:

```bash
ls /blue/<GROUP>/<USER>/waymo/dataset/waymo_open_dataset_v_2_0_1/training/camera_image/*.parquet | wc -l
```

For Waymo v2 training this is ~798 files. With `--array=0-49` (50 shards) each
shard processes ~16 segments. Bump or trim the array size in
`submit_hipergator.sbatch` if you want more/less per-shard parallelism.

```bash
sbatch submit_hipergator.sbatch
squeue -u $USER
tail -f logs/shard_*_0.out  # first shard's stdout
```

Output lands in `/blue/<GROUP>/<USER>/waymo/output/segment_<id>/...` per the
layout described in step 3.

### Requeuing failed shards

`sacct -j <jobid>` shows which array tasks failed. Resubmit only those:

```bash
sbatch --array=7,12,33 submit_hipergator.sbatch
```

Re-runs are safe — image writes are deterministic per `(segment, frame, camera)`,
so a re-running shard just overwrites its own files.

## Notes

- **Sharding** is round-robin over camera parquet files. Waymo v2 is one segment per file,
  so shards write disjoint `segment_*` directories.
- **Waterdrop** augmentation requires OpenGL and will fail-and-skip on CPU compute nodes.
  The augmenter logs a warning and continues with the other 9 corruptions. Re-run only
  waterdrop on a GPU partition later if needed.

## Troubleshooting

**`module load rclone` says "not found".**
Install rclone into your home dir instead:
```bash
curl https://rclone.org/install.sh | bash -s -- --beta-latest=false
# or, no-sudo variant:
mkdir -p $HOME/bin && cd /tmp && curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip rclone-current-linux-amd64.zip && cp rclone-*-linux-amd64/rclone $HOME/bin/
export PATH=$HOME/bin:$PATH    # add to ~/.bashrc
```
Then edit `transfer_waymo.sbatch` and replace `module load rclone` with
`export PATH=$HOME/bin:$PATH`.

**OAuth browser doesn't work in Open OnDemand Desktop.**
Do the auth on your laptop instead:
```bash
# On your laptop (rclone installed locally):
rclone authorize "google cloud storage"
# Sign in via the browser tab it opens. It prints a JSON token.
```
Then on HiPerGator run `rclone config`, choose `n` (new), name `waymo`,
type `google cloud storage`, and when it asks "Use auto config?" answer `n`
and paste the JSON token from your laptop.

**SIF build fails with "no space left on device".**
`/tmp` on the login node is small. Redirect Singularity scratch:
```bash
export SINGULARITY_TMPDIR=/blue/<GROUP>/<USER>/singularity-tmp
export APPTAINER_TMPDIR=$SINGULARITY_TMPDIR
mkdir -p $SINGULARITY_TMPDIR
singularity build /blue/<GROUP>/<USER>/waymo/hiker.sif hiker.def
```

**Shard hangs / runs out of memory.**
The pipeline holds all of a shard's records in memory at once. If a shard
OOMs at 32 GB, bump `--mem=64G` in `submit_hipergator.sbatch` and resubmit
just that shard with `sbatch --array=<id>`.

**Dataset size on disk is much smaller than `rclone size` reported.**
Likely the transfer was killed mid-flight. Re-running `sbatch transfer_waymo.sbatch`
resumes — rclone skips matched files and pulls only the missing ones.
