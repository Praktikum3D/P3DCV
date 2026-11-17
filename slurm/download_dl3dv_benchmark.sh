#!/bin/bash
#SBATCH --job-name=download-dl3dv
#SBATCH --partition=data
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Downloads the views of the DL3DV benchmark that the E-RayZer evaluation
# reads, converts them and stores them as one zip file in the team folder.
# See docs/datasets/dl3dv.md.
#
# Usage:
#   sbatch slurm/download_dl3dv_benchmark.sh              # all 140 scenes
#   LIMIT=4 sbatch slurm/download_dl3dv_benchmark.sh      # the first 4 scenes
#
# Needs a Hugging Face token with access to DL3DV/DL3DV-Benchmark in
# ~/.cache/huggingface/token (or set HF_TOKEN_FILE).
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

FRAME_DIST="${FRAME_DIST:-96}"
LIMIT="${LIMIT:-0}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/mnt/home/$USER/.cache/huggingface/token}"
if [ "$LIMIT" -gt 0 ]; then
    DEFAULT_ARCHIVE="$DATA_DIR/dl3dv/benchmark_fd${FRAME_DIST}_first${LIMIT}.zip"
else
    DEFAULT_ARCHIVE="$DATA_DIR/dl3dv/benchmark_fd${FRAME_DIST}.zip"
fi
ARCHIVE="${ARCHIVE:-$DEFAULT_ARCHIVE}"

if [ ! -s "$HF_TOKEN_FILE" ]; then
    echo "No Hugging Face token in $HF_TOKEN_FILE" >&2
    exit 1
fi

# Everything is unpacked on the local disk of the node; only the zip file
# goes to the team folder because of its file limit.
WORK_DIR="/tmp/$USER-dl3dv-$SLURM_JOB_ID"
trap 'rm -rf "$WORK_DIR"' EXIT

in_container python "$REPO_DIR/scripts/prepare_dl3dv_benchmark.py" \
    --scene-id-list "$REPO_DIR/data_splits/dl3dv/test_full.txt" \
    --work-dir "$WORK_DIR" \
    --archive "$ARCHIVE" \
    --frame-dist "$FRAME_DIST" \
    --limit "$LIMIT" \
    --token-file "$HF_TOKEN_FILE"

ls -lh "$(dirname "$ARCHIVE")"
