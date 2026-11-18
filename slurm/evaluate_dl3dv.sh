#!/bin/bash
#SBATCH --job-name=eval-dl3dv
#SBATCH --partition=24g
#SBATCH --gres=gpu:1,ccc:86
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Evaluates a checkpoint on the DL3DV benchmark with the protocol of the
# E-RayZer paper. See docs/evaluation.md. CKPT, ARCHIVE and OUT_DIR override
# the defaults.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

CKPT="${CKPT:-$PRETRAINED_DIR/erayzer_dl3dv.pt}"
ARCHIVE="${ARCHIVE:-$DATA_DIR/dl3dv/benchmark_fd96.zip}"
OUT_DIR="${OUT_DIR:-/mnt/home/$USER/runs/eval/dl3dv-$(basename "$CKPT" .pt)-$SLURM_JOB_ID}"

# The archive is unpacked on the local disk of the node.
WORK_DIR="/tmp/$USER-dl3dv-eval-$SLURM_JOB_ID"
trap 'rm -rf "$WORK_DIR"' EXIT
mkdir -p "$WORK_DIR"
in_container python -m zipfile -e "$ARCHIVE" "$WORK_DIR"

# The archive holds only the frames the protocol reads, so the manifest check,
# which looks for the first two frames, is skipped.
in_container bash -c "cd '$REPO_DIR' && env NUM_GPUS=1 CHECK_MANIFEST=false FRAME_DIST=96 \
    MANIFEST_PATH='$WORK_DIR/manifests/test_full.txt' CKPT_PATH='$CKPT' EVAL_OUT_DIR='$OUT_DIR' \
    bash scripts/evaluate_dl3dv.sh"

cat "$OUT_DIR/metrics.txt"
