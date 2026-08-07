#!/bin/bash
#
# Stage 3: merge the per-model smoke reports into ONE verdict file.
#
# Runs with --dependency=afterany on the smoke array, so it still produces a
# report when a model crashed — a missing per-model report is itself recorded as
# a failure rather than silently read as "nothing wrong".
#
# The output, reports/smoke_report.json, is the artifact to hand to the deciding
# agent (or send back to us). CPU-only.
#
# Usage (normally via submit_baselines.sh):
#   sbatch --dependency=afterany:<smoke_array_jobid> aggregate_smoke.sbatch
#
#SBATCH --job-name=vis-smoke-agg
#SBATCH --account=<GROUP>
#SBATCH --qos=<GROUP>
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/smoke_agg_%j.out
#SBATCH --error=logs/smoke_agg_%j.err

set -uo pipefail

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

module load singularity

singularity exec \
    --bind "${BIND_ROOT}" \
    --bind "${REPO_ROOT}":/workspace \
    "${SIF}" \
    bash -lc "cd /workspace/visuals-ml && \
        python -m baselines.smoke_deploy \
            --merge reports/smoke_report_box3d.json \
                    reports/smoke_report_monodetr.json \
                    reports/smoke_report_velocity.json \
            --report reports/smoke_report.json"

VERDICT_STATUS=$?

echo
echo "============================================================"
echo "  Merged report: ${REPO_ROOT}/visuals-ml/reports/smoke_report.json"
echo "  Exit status ${VERDICT_STATUS} (0 = PASS, 1 = FAIL)"
echo "  Hand that JSON to the deciding agent before submitting the"
echo "  full training array (train_baselines.sbatch)."
echo "============================================================"

exit ${VERDICT_STATUS}
