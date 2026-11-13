#!/bin/bash
#SBATCH --job-name=check-env
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Smoke test: starts the container on a GPU node and checks the environment,
# the shared folders and the pretrained weights.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

in_container python "$REPO_DIR/scripts/check_env.py"
