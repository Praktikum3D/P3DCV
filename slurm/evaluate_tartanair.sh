#!/bin/bash
#SBATCH --job-name=eval-tartanair
#SBATCH --partition=24g
#SBATCH --gres=gpu:1,ccc:86
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Evaluates a checkpoint on held out TartanAir environments with the same
# metrics as the DL3DV benchmark. See docs/evaluation.md.
#
#   CAMERA      lcam_front (pinhole, default) or lcam_fish
#   ENVS        environments to evaluate on, default HongKong and Office
#   CKPT        checkpoint, default erayzer_multi.pt
#   FRAME_DIST  frames spanned by the 10 views, default 40 (4 s at 10 Hz)
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

CAMERA="${CAMERA:-lcam_front}"
ENVS="${ENVS:-HongKong/Data_hard Office/Data_hard}"
CKPT="${CKPT:-$PRETRAINED_DIR/erayzer_multi.pt}"
FRAME_DIST="${FRAME_DIST:-40}"
OUT_DIR="${OUT_DIR:-/mnt/home/$USER/runs/eval/tartanair-$CAMERA-$(basename "$CKPT" .pt)-$SLURM_JOB_ID}"

# One archive per environment, listed in a manifest the data loader reads.
WORK_DIR="/tmp/$USER-tartanair-eval-$SLURM_JOB_ID"
trap 'rm -rf "$WORK_DIR"' EXIT
mkdir -p "$WORK_DIR" "$OUT_DIR"
MANIFEST="$WORK_DIR/manifest.txt"
for env in $ENVS; do
    archive="$DATA_DIR/tartanair/$env/image_$CAMERA.zip"
    [ -f "$archive" ] || { echo "missing $archive" >&2; exit 1; }
    echo "$archive" >> "$MANIFEST"
done
echo "Evaluating $(basename "$CKPT") on $CAMERA of $ENVS"

in_container bash -c "cd '$REPO_DIR' && python evaluate.py \
    --config config/erayzer_eval_tartanair.yaml \
    -s dataset.eval_path '$MANIFEST' \
    -s dataset.camera_name '$CAMERA' \
    -s evaluation.checkpoint_path '$CKPT' \
    -s evaluation.out_dir '$OUT_DIR' \
    -s training.view_selector.min_frame_dist '$FRAME_DIST' \
    -s training.view_selector.max_frame_dist '$FRAME_DIST'"

cat "$OUT_DIR/metrics.txt"
