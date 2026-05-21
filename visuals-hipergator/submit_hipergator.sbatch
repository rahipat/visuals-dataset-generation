#!/bin/bash
#
# SLURM array submission for Waymo HiKER augmentation pipeline.
# Each array task processes a round-robin slice of camera_image parquet files.
#
# Usage:
#   sbatch submit_hipergator.sbatch
#
# Adjust --array=0-N below to control number of shards (N+1 total).
#
#SBATCH --job-name=waymo-hiker
#SBATCH --account=<GROUP>
#SBATCH --qos=<GROUP>
#SBATCH --partition=hpg-default
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/shard_%A_%a.out
#SBATCH --error=logs/shard_%A_%a.err

set -euo pipefail

# ---- Edit these paths for your HiPerGator setup -----------------------------
REPO_ROOT=/blue/<GROUP>/<USER>/visuals-dataset-generation
DATASET_ROOT=/blue/<GROUP>/<USER>/waymo/dataset
OUTPUT_DIR=/blue/<GROUP>/<USER>/waymo/output
SIF=/blue/<GROUP>/<USER>/waymo/hiker.sif
SPLIT=training
# -----------------------------------------------------------------------------

NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}

mkdir -p logs "$OUTPUT_DIR"

module load singularity

echo "[info] Host: $(hostname)"
echo "[info] Shard: ${SHARD_INDEX}/${NUM_SHARDS}"
echo "[info] Dataset: ${DATASET_ROOT}"
echo "[info] Output:  ${OUTPUT_DIR}"
echo "[info] SIF:     ${SIF}"

singularity exec \
    --bind /blue/<GROUP> \
    --bind "${REPO_ROOT}":/workspace \
    "${SIF}" \
    python /workspace/visuals-hipergator/generate_dataset_hipergator.py \
        --dataset-root "${DATASET_ROOT}" \
        --output-dir "${OUTPUT_DIR}" \
        --split "${SPLIT}" \
        --shard-index "${SHARD_INDEX}" \
        --num-shards "${NUM_SHARDS}" \
        --verbose

echo "[info] Shard ${SHARD_INDEX} done."
