#!/bin/bash
#
# Transfer the full Waymo Open Dataset v2.0.1 bucket to /blue using rclone.
#
# Prerequisite: rclone remote `waymo` of type "Google Cloud Storage" already
# configured for the Google account that registered at waymo.com/open.
# See README_HIPERGATOR.md for the one-time setup.
#
# Usage:
#   sbatch transfer_waymo.sbatch
#
# Re-running is safe: rclone skips files already present at the destination
# with matching size + modtime, so a killed job can be resubmitted as-is.
#
#SBATCH --job-name=waymo-transfer
#SBATCH --account=<GROUP>
#SBATCH --qos=<GROUP>
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/transfer_%j.out
#SBATCH --error=logs/transfer_%j.err

set -euo pipefail

# ---- Edit these paths for your HiPerGator setup -----------------------------
REMOTE_NAME=waymo
SOURCE_PATH="${REMOTE_NAME}:waymo_open_dataset_v_2_0_1"
DEST_PATH=/blue/<GROUP>/<USER>/waymo/dataset/waymo_open_dataset_v_2_0_1
# -----------------------------------------------------------------------------

mkdir -p logs "${DEST_PATH}"

module load rclone

echo "[info] Host:   $(hostname)"
echo "[info] Source: ${SOURCE_PATH}"
echo "[info] Dest:   ${DEST_PATH}"
echo "[info] rclone version: $(rclone version | head -1)"

rclone copy \
    "${SOURCE_PATH}" \
    "${DEST_PATH}" \
    --transfers 32 \
    --checkers 32 \
    --multi-thread-streams 4 \
    --fast-list \
    --progress \
    --stats 60s \
    --stats-one-line \
    --log-file "logs/rclone_${SLURM_JOB_ID:-manual}.log" \
    --log-level INFO

echo "[info] Transfer complete."
echo "[info] Verifying with rclone size..."
rclone size "${DEST_PATH}"
