#!/bin/bash
#SBATCH --job-name=precompute-lines
#SBATCH --partition=data
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Detects the lines of every TartanAir source frame once and caches them, so the training step
# only has to map them into the drawn camera. Pure CPU: NO --gres, partition data, no GPU to idle.
# See training/line_cache.py for why the split exists and scripts/precompute_lines.py for the file
# format.
#
# Usage:
#   sbatch --export=ALL,REPO_DIR=$PWD slurm/precompute_lines.sh
#   sbatch --export=ALL,REPO_DIR=$PWD,ENVS="HongKong OldTown" slurm/precompute_lines.sh
#
# DATA_PATH defaults to the personal TartanAir copy, CACHE_DIR to a sibling line-cache folder.
# Safe to re-run: archives that already have a cache file are skipped, so an interrupted job just
# needs resubmitting. Pass OVERWRITE=1 to rebuild them.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

DATA_PATH="${DATA_PATH:-/mnt/home/$USER/data/tartanair}"
CACHE_DIR="${CACHE_DIR:-/mnt/home/$USER/data/line-cache}"
CAMERA="${CAMERA:-lcam_fish}"
SOURCE_SIZE="${SOURCE_SIZE:-0}"
LINE_SAMPLES="${LINE_SAMPLES:-8}"
LINE_MAX_COUNT="${LINE_MAX_COUNT:-32}"
# Detector processes. The detection is one frame at a time and perfectly parallel, and a whole
# environment is hours of single threaded work, so this is what decides whether the job is minutes
# or a day. 0 lets the script take half the cores it can see.
WORKERS="${WORKERS:-0}"

mkdir -p "$CACHE_DIR"

ARGS=(
    --data-path "$DATA_PATH"
    --cache-dir "$CACHE_DIR"
    --camera "$CAMERA"
    --source-size "$SOURCE_SIZE"
    --num-samples "$LINE_SAMPLES"
    --max-count "$LINE_MAX_COUNT"
    --workers "$WORKERS"
)
if [[ -n "${ENVS:-}" ]]; then
    ARGS+=(--envs $ENVS)
fi
if [[ -n "${OVERWRITE:-}" ]]; then
    ARGS+=(--overwrite)
fi

in_container bash -c "cd '$REPO_DIR' && python scripts/precompute_lines.py $(printf '%q ' "${ARGS[@]}")"
