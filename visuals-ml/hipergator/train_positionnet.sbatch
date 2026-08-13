#!/bin/bash
#
# Train the PositionNet baseline (per-object 3D-centre regressor) on HiPerGator.
#
# Runs the full chain inside the PyTorch/CUDA Singularity image (train.sif):
#   1. data.build_index      (per-object index over the visuals output tree)
#   2. baselines.train       (PositionNet, ResNet18 two-stream)
#   3. baselines.eval        (overall + per-weather-variant metrics)
#
# Step 1 is skipped if the index already exists.
#
# PositionNet is also the prerequisite for the Introspective Perception baseline:
# its best.pt is what train_introspection.sbatch's label pre-pass consumes.
#
# Usage:
#   export GROUP=<your-group> HPG_USER=$USER
#   sbatch --account=$GROUP --qos=$GROUP train_positionnet.sbatch
#
#SBATCH --job-name=positionnet
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
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/positionnet_%j.out
#SBATCH --error=logs/positionnet_%j.err

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
CONFIG=${CONFIG:-configs/positionnet.yaml}
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

INDEX=data/output/records.jsonl
if [ ! -f \"\$INDEX\" ]; then
    echo '[info] Building PositionNet index'
    # NOTE: this is 'data.build_index' (top-level data/), not
    # 'baselines.data.build_index' — the latter does not exist.
    python -m data.build_index \
        --source-dir '${DATASET_OUTPUT}' \
        --index-file \"\$INDEX\"
else
    echo '[info] Reusing existing index'
fi

echo '[info] Training PositionNet'
python -m baselines.train --config '${CONFIG}'

echo '[info] Evaluating best checkpoint'
python -m baselines.eval --config '${CONFIG}'
"

echo "[info] PositionNet training done."
