#!/bin/bash
#
# Submit the baselines pipeline with the SLURM dependencies already wired.
#
#   ./submit_baselines.sh smoke     index (subset) -> 3 smoke jobs -> merge
#   ./submit_baselines.sh full      index (full)   -> 3 training jobs -> (eval)
#   ./submit_baselines.sh full-eval build the all-weather eval index only
#
# The smoke and training stages run as ARRAY jobs, one task per model, so the
# three baselines occupy three GPUs in parallel rather than queueing behind each
# other. Nothing is submitted twice: each stage waits on the previous job id.
#
# After `smoke` finishes, read visuals-ml/reports/smoke_report.json. Do not
# submit `full` until that verdict has been looked at — the full array is the
# expensive one.

set -euo pipefail

STAGE="${1:-smoke}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

case "$STAGE" in
  smoke)
    echo "[submit] stage 1: build smoke indexes (CPU)"
    IDX_JOB=$(sbatch --parsable --export=ALL,MODE=smoke prepare_indexes.sbatch)
    echo "         job ${IDX_JOB}"

    echo "[submit] stage 2: smoke test, 3 models in parallel (GPU array)"
    SMOKE_JOB=$(sbatch --parsable \
        --dependency=afterok:"${IDX_JOB}" \
        --export=ALL,MODE=smoke \
        --array=0-2 smoke_baselines.sbatch)
    echo "         job ${SMOKE_JOB} (array 0-2)"

    # afterany, not afterok: a model that FAILS its gates must still be
    # aggregated and reported, otherwise the verdict file never appears and the
    # failure looks like an infrastructure problem.
    echo "[submit] stage 3: merge reports (CPU)"
    AGG_JOB=$(sbatch --parsable \
        --dependency=afterany:"${SMOKE_JOB}" aggregate_smoke.sbatch)
    echo "         job ${AGG_JOB}"

    echo
    echo "Watch:   squeue -u \$USER"
    echo "Verdict: visuals-ml/reports/smoke_report.json  (after job ${AGG_JOB})"
    ;;

  full)
    echo "[submit] stage 1: build FULL training index (CPU, ~hours)"
    IDX_JOB=$(sbatch --parsable --export=ALL,MODE=full prepare_indexes.sbatch)
    echo "         job ${IDX_JOB}"

    echo "[submit] stage 1b: build ALL-WEATHER eval index (CPU)"
    EVAL_IDX_JOB=$(sbatch --parsable \
        --dependency=afterok:"${IDX_JOB}" \
        --export=ALL,MODE=full-eval prepare_indexes.sbatch)
    echo "         job ${EVAL_IDX_JOB}"

    echo "[submit] stage 4: full training, 3 models in parallel (GPU array)"
    TRAIN_JOB=$(sbatch --parsable \
        --dependency=afterok:"${EVAL_IDX_JOB}" \
        --array=0-2 train_baselines.sbatch)
    echo "         job ${TRAIN_JOB} (array 0-2)"

    echo
    echo "Watch:   squeue -u \$USER"
    echo "Results: visuals-ml/reports/final_<model>.json"
    ;;

  full-eval)
    JOB=$(sbatch --parsable --export=ALL,MODE=full-eval prepare_indexes.sbatch)
    echo "[submit] all-weather eval index: job ${JOB}"
    ;;

  *)
    echo "usage: $0 {smoke|full|full-eval}" >&2
    exit 2
    ;;
esac
