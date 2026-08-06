#!/bin/bash
#
# Stage 1 of the baselines pipeline: build the JSONL indexes the models train on.
#
# Runs ONCE, before the model array jobs, because they share the detection index
# and must not race to write it. CPU-only — no GPU is requested.
#
# Two indexes are produced:
#   det_records.jsonl       per-image, camera-optical-frame 3D labels.
#                           Used by BOTH box3d (baseline B) and monodetr (A).
#   velocity_records.jsonl  per-object frame pairs for velocity (C).
#
# MODE controls the size:
#   smoke  small subset, for the deploy test           (minutes)
#   full   the training set proper                     (up to a few hours)
#
# Usage:
#   sbatch --export=ALL,MODE=smoke prepare_indexes.sbatch
#   sbatch --export=ALL,MODE=full  prepare_indexes.sbatch
#
#SBATCH --job-name=vis-index
#SBATCH --account=<GROUP>
#SBATCH --qos=<GROUP>
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/index_%j.out
#SBATCH --error=logs/index_%j.err

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
CAMERAS="${CAMERAS:-1}"

case "$MODE" in
  smoke)
    # Enough segments to form a real segment-level split, small enough to be quick.
    MAX_SEGMENTS_ARG="--max-segments 40"
    FRAME_STRIDE=10
    WEATHERS="clear,fog,rain"
    ;;
  full)
    # All segments. Stride 5 because frames are ~10 Hz and neighbours are
    # near-duplicates; this cuts the index 5x at little information cost.
    # Train weathers only — the eval index is built separately (see below).
    MAX_SEGMENTS_ARG=""
    FRAME_STRIDE=5
    WEATHERS="clear"
    ;;
  full-eval)
    # All 10 weathers, for evaluation only. Much bigger; build it after training.
    MAX_SEGMENTS_ARG=""
    FRAME_STRIDE=5
    WEATHERS="clear,rain,fog,snow,frost,sunglare,brightness,wildfire_smoke,dust,waterdrop"
    ;;
  *)
    echo "[error] MODE must be smoke | full | full-eval (got '$MODE')" >&2
    exit 2
    ;;
esac

mkdir -p logs

module load singularity

echo "[info] Host:      $(hostname)"
echo "[info] Mode:      ${MODE}"
echo "[info] Cameras:   ${CAMERAS}"
echo "[info] Weathers:  ${WEATHERS}"
echo "[info] Stride:    ${FRAME_STRIDE}"
echo "[info] Dataset:   ${DATASET_OUTPUT}"

run() {
    singularity exec \
        --bind "${BIND_ROOT}" \
        --bind "${REPO_ROOT}":/workspace \
        --bind "${DATASET_OUTPUT}":/data \
        "${SIF}" \
        bash -lc "cd /workspace/visuals-ml && $*"
}

echo "[info] === detection index (box3d + monodetr) ==="
run "python -m baselines.data.build_detection_index \
        --source-dir /data \
        --index-file data/output/det_records_${MODE}.jsonl \
        --cameras ${CAMERAS} \
        --weathers ${WEATHERS} \
        --frame-stride ${FRAME_STRIDE} \
        ${MAX_SEGMENTS_ARG}"

echo "[info] === velocity index ==="
run "python data/build_velocity_index.py \
        --source-dir /data \
        --index-file data/output/velocity_records_${MODE}.jsonl"

echo "[info] === index summary ==="
run "python - <<'PY'
import json, glob, collections
for path in sorted(glob.glob('data/output/*_${MODE}.jsonl')):
    segs, n, weathers = set(), 0, collections.Counter()
    for line in open(path, encoding='utf-8'):
        if not line.strip():
            continue
        r = json.loads(line); n += 1
        segs.add(r.get('segment')); weathers[r.get('weather')] += 1
    print(f'{path}: {n} records, {len(segs)} segments, weathers={dict(weathers)}')
PY"

echo "[info] Indexes ready for MODE=${MODE}."
