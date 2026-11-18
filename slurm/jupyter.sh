#!/bin/bash
#SBATCH --job-name=jupyter
# E-RayZer runs in bf16, which xformers supports only on RTX 30xx and newer
# (ccc:86). Those GPUs are in the 24g partition.
#SBATCH --partition=24g
#SBATCH --gres=gpu:1,ccc:86
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Starts Jupyter Lab inside the container on a GPU node. The log prints the
# URL, which is reachable from inside the VPN only. 24g cancels jobs that
# leave the GPU idle for two hours, so cancel the job with scancel when you
# are done.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

PORT=$((20000 + RANDOM % 10000))
TOKEN=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')
echo "URL: http://$(hostname -i):$PORT/lab?token=$TOKEN"

in_container bash -c "cd /mnt/home/$USER && jupyter lab --no-browser --allow-root \
    --ip 0.0.0.0 --port $PORT --ServerApp.token=$TOKEN \
    --notebook-dir /mnt/home/$USER"
