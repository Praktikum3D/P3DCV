#!/bin/bash
#SBATCH --job-name=eval-dl3dv
# E-RayZer runs in bf16, which xformers supports only on RTX 30xx and newer
# (ccc:86). Those GPUs are in the 24g partition.
#SBATCH --partition=24g
#SBATCH --gres=gpu:1,ccc:86
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Evaluates a checkpoint on the DL3DV benchmark with the protocol of the
# E-RayZer paper (10 views, FRAME_DIST=96) and writes summary.csv and
# metrics.txt to ~/runs/eval. See docs/evaluation.md.
#
# Usage:
#   sbatch slurm/evaluate_dl3dv.sh
#   CKPT=$PRETRAINED_DIR/erayzer_multi.pt sbatch slurm/evaluate_dl3dv.sh
#   ARCHIVE=/path/to/benchmark.zip sbatch slurm/evaluate_dl3dv.sh
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

FRAME_DIST="${FRAME_DIST:-96}"
CKPT="${CKPT:-$PRETRAINED_DIR/erayzer_dl3dv.pt}"
ARCHIVE="${ARCHIVE:-$DATA_DIR/dl3dv/benchmark_fd${FRAME_DIST}.zip}"
OUT_DIR="${OUT_DIR:-/mnt/home/$USER/runs/eval/dl3dv-$(basename "$CKPT" .pt)-$SLURM_JOB_ID}"

# The archive is unpacked on the local disk of the node.
WORK_DIR="/tmp/$USER-dl3dv-eval-$SLURM_JOB_ID"
trap 'rm -rf "$WORK_DIR"' EXIT
mkdir -p "$WORK_DIR"
in_container python -m zipfile -e "$ARCHIVE" "$WORK_DIR"

echo "checkpoint  $CKPT"
echo "archive     $ARCHIVE ($(wc -l < "$WORK_DIR/manifests/test_full.txt") scenes)"

# The archive holds only the frames of the protocol, so the upstream manifest
# check, which looks for the first two frames, is skipped.
in_container bash -c "cd '$REPO_DIR' && env \
    NUM_GPUS=1 \
    CHECK_MANIFEST=false \
    FRAME_DIST='$FRAME_DIST' \
    MANIFEST_PATH='$WORK_DIR/manifests/test_full.txt' \
    CKPT_PATH='$CKPT' \
    EVAL_OUT_DIR='$OUT_DIR' \
    SAVE_EXAMPLES='${SAVE_EXAMPLES:-8}' \
    bash scripts/evaluate_dl3dv.sh"

echo "results     $OUT_DIR"
cat "$OUT_DIR/metrics.txt"
