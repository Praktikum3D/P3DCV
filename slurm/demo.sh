#!/bin/bash
#SBATCH --job-name=demo
# E-RayZer runs in bf16, which xformers supports only on RTX 30xx and newer
# (ccc:86). Those GPUs are in the 24g partition.
#SBATCH --partition=24g
#SBATCH --gres=gpu:1,ccc:86
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Starts the E-RayZer Gradio demo on a GPU node for two hours. The log prints
# the URL, which is reachable from inside the VPN only.
# Cancel the job with scancel when you are done.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

CKPT="${CKPT:-$PRETRAINED_DIR/erayzer_multi.pt}"
PORT=$((20000 + RANDOM % 10000))
echo "URL: http://$(hostname -i):$PORT"

in_container bash -c "cd '$REPO_DIR' && python -u gradio_app.py \
    --config config/erayzer_inference.yaml --ckpt '$CKPT' \
    --output-dir '/mnt/home/$USER/runs/demo' --server-port $PORT"
