#!/bin/bash
#
# Stage 2: deploy smoke test — the three perception baselines in PARALLEL.
#
# One array task per model, so a failure in one does not hide the others and the
# whole thing finishes in the time of the slowest single model.
#
#   task 0 -> box3d     (B, amodal 3D box regression)
#   task 1 -> monodetr  (A, monocular 3D detection)
#   task 2 -> velocity  (C, per-object velocity)
#
# Each task writes smoke_report_<model>.json. Stage 3 (aggregate_smoke.sbatch)
# merges them into one verdict file to hand to the deciding agent.
#
# This does NOT train anything to convergence. It proves the deploy works and the
# wiring is sound. Nothing it prints belongs in a paper.
#
# Usage (normally via submit_baselines.sh, which wires the dependencies):
#   sbatch --array=0-2 smoke_baselines.sbatch
#
#SBATCH --job-name=vis-smoke
#SBATCH --account=<GROUP>
#SBATCH --qos=<GROUP>
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_%A_%a.out
#SBATCH --error=logs/smoke_%A_%a.err

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

MODE="${MODE:-smoke}"
MODELS=(box3d monodetr velocity)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}"

mkdir -p logs reports

module load singularity

echo "[info] Host:   $(hostname)"
echo "[info] Model:  ${MODEL}  (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "[info] Mode:   ${MODE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# `set -e` is disabled around this call on purpose. smoke_deploy exits 1 on a
# FAIL verdict, and with -e that would abort the script before STATUS is read --
# so the "don't fail the array task" logic below would never run.
set +e
singularity exec --nv \
    --bind "${BIND_ROOT}" \
    --bind "${REPO_ROOT}":/workspace \
    --bind "${DATASET_OUTPUT}":/data \
    "${SIF}" \
    bash -lc "cd /workspace/visuals-ml && \
        python -m baselines.smoke_deploy \
            --config baselines/configs/smoke_deploy.yaml \
            --models ${MODEL} \
            --index-suffix ${MODE} \
            --report reports/smoke_report_${MODEL}.json"
STATUS=$?
set -e

echo "[info] ${MODEL} smoke exit status: ${STATUS} (0 = PASS, 1 = FAIL verdict)"

# Exit 0 regardless of the verdict: a FAIL is a legitimate result that must
# still reach the aggregator. A genuine crash shows up as a MISSING report,
# which the aggregator treats as a failure in its own right.
exit 0
