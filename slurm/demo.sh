#!/bin/bash
#SBATCH --job-name=demo
#SBATCH --partition=12g
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Starts the E-RayZer Gradio demo on a GPU node for two hours. The log prints
# a local URL (reachable from the VPN) and a public gradio.live URL.
# Cancel the job with scancel when you are done.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

CKPT="${CKPT:-$PRETRAINED_DIR/erayzer_multi.pt}"
PORT=$((20000 + RANDOM % 10000))
echo "Local URL: http://$(hostname -i):$PORT"

in_container bash -c "cd '$REPO_DIR' && python -u gradio_app.py \
    --config config/erayzer_inference.yaml --ckpt '$CKPT' \
    --output-dir '/mnt/home/$USER/runs/demo' --server-port $PORT --share"
