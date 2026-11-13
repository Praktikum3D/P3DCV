#!/bin/bash
#SBATCH --job-name=infer
#SBATCH --partition=12g
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Runs a pretrained E-RayZer checkpoint on a few views and writes the results
# to ~/runs/inference. Without arguments it uses one trajectory of the
# TartanAir sample. All arguments are passed on to scripts/infer.py, e.g.
#   sbatch slurm/infer.sh --images /mnt/home/$USER/my_scene
# CKPT selects the checkpoint.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

CKPT="${CKPT:-$PRETRAINED_DIR/erayzer_multi.pt}"
if [ "$#" -eq 0 ]; then
    set -- --archive "$DATA_DIR/tartanair/ArchVizTinyHouseNight/Data_hard/image_lcam_fish.zip" --trajectory P003
fi

in_container python "$REPO_DIR/scripts/infer.py" \
    --ckpt "$CKPT" --out "/mnt/home/$USER/runs/inference" "$@"
