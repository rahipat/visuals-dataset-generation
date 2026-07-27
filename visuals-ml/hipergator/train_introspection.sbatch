#!/bin/bash
#
# Train the Introspective Perception baseline (Paper 1) on HiPerGator.
#
# Runs the full chain inside the PyTorch/CUDA Singularity image (train.sif):
#   1. build_introspection_index      (per-frame index + temporal neighbours)
#   2. introspection_label_prepass    (PositionNet -> per-frame failure labels)
#   3. baselines.train                (two-stream CNN, softmax)
#   4. baselines.introspection_svm    (fc8 -> linear SVM, EFR/RAM report)
#
# Steps 1-2 are skipped if the labeled index already exists. Requires a trained
# PositionNet checkpoint (POSITIONNET_CKPT).
#
# Usage:
#   sbatch train_introspection.sbatch
#
#SBATCH --job-name=introspection
#SBATCH --account=<GROUP>
#SBATCH --qos=<GROUP>
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/introspection_%j.out
#SBATCH --error=logs/introspection_%j.err

set -euo pipefail

# ---- Edit these paths for your HiPerGator setup -----------------------------
REPO_ROOT=/blue/<GROUP>/<USER>/visuals-dataset-generation
DATASET_OUTPUT=/blue/<GROUP>/<USER>/waymo/output          # visuals_dataset output tree
SIF=/blue/<GROUP>/<USER>/waymo/train.sif
POSITIONNET_CKPT=baselines/checkpoints/positionnet/best.pt  # relative to visuals-ml/
CONFIG=configs/introspection.yaml
CAMERAS=1
# -----------------------------------------------------------------------------

mkdir -p logs

module load singularity

echo "[info] Host: $(hostname)"
echo "[info] Dataset output: ${DATASET_OUTPUT}"
echo "[info] SIF: ${SIF}"

singularity exec --nv \
    --bind /blue/<GROUP> \
    --bind "${REPO_ROOT}":/workspace \
    "${SIF}" bash -lc "
set -euo pipefail
cd /workspace/visuals-ml

LABELED=data/output/introspection_labeled.jsonl
if [ ! -f \"\$LABELED\" ]; then
    echo '[info] Building introspection index + labels'
    python -m baselines.data.build_introspection_index \
        --source-dir '${DATASET_OUTPUT}' \
        --index-file data/output/introspection_records.jsonl \
        --cameras '${CAMERAS}' --flow-stack 5
    python -m baselines.data.introspection_label_prepass \
        --index-file data/output/introspection_records.jsonl \
        --checkpoint '${POSITIONNET_CKPT}' \
        --out-file \"\$LABELED\" \
        --tau-percentile 50 --fail-thresh 0.5
else
    echo '[info] Reusing existing labeled index'
fi

echo '[info] Training two-stream CNN'
python -m baselines.train --config '${CONFIG}'

echo '[info] Fitting linear SVM stage'
python -m baselines.introspection_svm --config '${CONFIG}'
"

echo "[info] Introspection training done."
