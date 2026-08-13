#!/bin/bash
#
# Train the MonoDETR baseline (monocular 3D detector) on HiPerGator.
#
# Runs the full chain inside the PyTorch/CUDA Singularity image (train.sif):
#   1. baselines.data.build_detection_index  (per-image index with 3D boxes)
#   2. baselines.train                       (MonoDETR, ResNet50 + depth-aware tf)
#   3. baselines.eval                        (overall + per-weather-variant metrics)
#
# Step 1 is skipped if the index already exists.
#
# Note the walltime: configs/monodetr.yaml is 195 epochs (the upstream schedule).
# Lower `epochs` in the config if you only need a shorter run.
#
# Usage:
#   export GROUP=<your-group> HPG_USER=$USER
#   sbatch --account=$GROUP --qos=$GROUP train_monodetr.sbatch
#
#SBATCH --job-name=monodetr
# NOTE: #SBATCH lines are read by SLURM before the shell runs, so they get NO
# parameter expansion — a ${GROUP:?...} here would be sent to SLURM verbatim.
# Either edit the two placeholders below, or override them at submit time with
# `sbatch --account=... --qos=...` (command-line flags win over these lines).
#SBATCH --account=<GROUP>
#SBATCH --qos=<GROUP>
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=logs/monodetr_%j.out
#SBATCH --error=logs/monodetr_%j.err

set -euo pipefail

# ---- Configuration ----------------------------------------------------------
# Set these in your environment before submitting; sbatch forwards the submitting
# environment to the job by default (--export=ALL), so they reach the compute
# node. An unset value aborts with the message below instead of the cryptic
# "No such file or directory" a literal <GROUP> placeholder produces.
#
# HPG_USER rather than USER: every login shell already sets USER, so a
# ${USER:?...} guard could never fire and would silently bake in the login name.
: "${GROUP:?set me - your HiPerGator group, e.g. export GROUP=ufhpc}"
: "${HPG_USER:?set me - your HiPerGator username, e.g. export HPG_USER=\$USER}"

# Override any of these directly if your layout differs.
REPO_ROOT=${REPO_ROOT:-/blue/${GROUP}/${HPG_USER}/visuals-dataset-generation}
DATASET_OUTPUT=${DATASET_OUTPUT:-/blue/${GROUP}/${HPG_USER}/waymo/output}  # visuals_dataset output tree
SIF=${SIF:-/blue/${GROUP}/${HPG_USER}/waymo/train.sif}
CONFIG=${CONFIG:-configs/monodetr.yaml}
CAMERAS=${CAMERAS:-1}
# -----------------------------------------------------------------------------

mkdir -p logs

module load singularity

echo "[info] Host: $(hostname)"
echo "[info] Dataset output: ${DATASET_OUTPUT}"
echo "[info] SIF: ${SIF}"

singularity exec --nv \
    --bind /blue/"${GROUP}" \
    --bind "${REPO_ROOT}":/workspace \
    "${SIF}" bash -lc "
set -euo pipefail
cd /workspace/visuals-ml

# The compiled deformable-attention op is optional: the vendored code falls back
# to a pure-PyTorch implementation when it is missing. That fallback is correct
# but slow, so surface which path this job is on rather than letting a 195-epoch
# run silently take the slow one.
python -c \"
import MultiScaleDeformableAttention  # noqa: F401
print('[info] MSDeformAttn: compiled CUDA op (fast path)')
\" || echo '[warn] MSDeformAttn: pure-PyTorch fallback (SLOW) — rebuild train.sif from train.def'

INDEX=data/output/det_records.jsonl
if [ ! -f \"\$INDEX\" ]; then
    echo '[info] Building detection index'
    python -m baselines.data.build_detection_index \
        --source-dir '${DATASET_OUTPUT}' \
        --index-file \"\$INDEX\" \
        --cameras '${CAMERAS}'
else
    echo '[info] Reusing existing index'
fi

echo '[info] Training MonoDETR'
python -m baselines.train --config '${CONFIG}'

echo '[info] Evaluating best checkpoint'
python -m baselines.eval --config '${CONFIG}'
"

echo "[info] MonoDETR training done."
