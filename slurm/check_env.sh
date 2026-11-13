#!/bin/bash
#SBATCH --job-name=check-env
#SBATCH --partition=12g
# gsplat needs an RTX 20xx or newer. ccc is the CUDA compute capability of the
# node times ten; ccc:75 keeps the job off the TITAN X / TITAN Xp nodes.
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Smoke test: starts the container on a GPU node and checks the environment,
# the shared folders and the pretrained weights.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

in_container python "$REPO_DIR/scripts/check_env.py"
