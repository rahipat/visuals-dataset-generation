#!/bin/bash
#
# Stage 4: FULL training of the three perception baselines, in PARALLEL.
#
#   task 0 -> box3d     (B) ~24 h budget
#   task 1 -> monodetr  (A) ~72 h budget, the heavy one
#   task 2 -> velocity  (C) ~24 h budget
#
# ONLY submit this after reading reports/smoke_report.json. Per-model walltime
# differs by ~3x, so the array requests the maximum and each task exits when it
# is done; that is cheaper than three separate scripts to maintain.
#
# Each task trains, then evaluates the best checkpoint against the ALL-WEATHER
# index and writes reports/final_<model>.json.
#
# Usage (normally via submit_baselines.sh):
#   sbatch --array=0-2 train_baselines.sbatch
#   sbatch --array=0 train_baselines.sbatch      # just box3d
#
#SBATCH --job-name=vis-train
#SBATCH --account=<GROUP>
#SBATCH --qos=<GROUP>
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

set -euo pipefail

# ---- Edit these two; every path below derives from them ---------------------
GROUP="${GROUP:-CHANGEME_GROUP}"        # your allocation group
USERDIR="${USERDIR:-CHANGEME_USER}"     # your directory under /blue/$GROUP
# -----------------------------------------------------------------------------
REPO_ROOT=/blue/${GROUP}/${USERDIR}/visuals-dataset-generation
DATASET_OUTPUT=/blue/${GROUP}/${USERDIR}/waymo/output
SIF=/blue/${GROUP}/${USERDIR}/waymo/train.sif
BIND_ROOT=/blue/${GROUP}

if [[ "${GROUP}" == CHANGEME* || "${USERDIR}" == CHANGEME* ]]; then
    echo "[error] Edit GROUP and USERDIR at the top of $(basename "$0"), and the" >&2
    echo "        #SBATCH --account/--qos lines (those cannot use variables)." >&2
    exit 2
fi

MODELS=(box3d monodetr velocity)
CONFIGS=(configs/box3d.yaml configs/monodetr.yaml configs/velocity.yaml)
IDX=${SLURM_ARRAY_TASK_ID:-0}
MODEL="${MODELS[$IDX]}"
CONFIG="${CONFIGS[$IDX]}"

# Train on the `full` index (clear only). Evaluate on `full-eval` (all 10
# weathers) — weather is the test axis, not a training augmentation.
TRAIN_SUFFIX="${TRAIN_SUFFIX:-full}"
EVAL_SUFFIX="${EVAL_SUFFIX:-full-eval}"

mkdir -p logs reports

module load singularity

echo "[info] Host:   $(hostname)"
echo "[info] Model:  ${MODEL}  (config ${CONFIG})"
echo "[info] Train index suffix: ${TRAIN_SUFFIX}   Eval: ${EVAL_SUFFIX}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run() {
    singularity exec --nv \
        --bind "${BIND_ROOT}" \
        --bind "${REPO_ROOT}":/workspace \
        --bind "${DATASET_OUTPUT}":/data \
        "${SIF}" \
        bash -lc "cd /workspace/visuals-ml && $*"
}

echo "[info] === train ${MODEL} ==="
run "python -m baselines.train \
        --config ${CONFIG} \
        --index-suffix ${TRAIN_SUFFIX}"

echo "[info] === evaluate ${MODEL} across all 10 weather variants ==="
run "python -m baselines.eval \
        --config ${CONFIG} \
        --index-suffix ${EVAL_SUFFIX} \
        --report reports/final_${MODEL}.json"

echo "[info] ${MODEL} done. Report: reports/final_${MODEL}.json"
