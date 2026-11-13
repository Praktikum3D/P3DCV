#!/bin/bash
#SBATCH --job-name=tartanair-sample
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Sample run: reads the TartanAir sample archive straight from the zip, prints
# what it contains and writes a contact sheet of one trajectory.
# Pass a different archive as the first argument.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

ARCHIVE="${1:-$DATA_DIR/tartanair/ArchVizTinyHouseNight/Data_hard/image_lcam_fish.zip}"

in_container python "$REPO_DIR/scripts/tartanair_sample.py" \
    --archive "$ARCHIVE" \
    --out "/mnt/home/$USER/runs/tartanair_sample.png"
