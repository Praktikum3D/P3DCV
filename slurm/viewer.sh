#!/bin/bash
#SBATCH --job-name=viewer
# E-RayZer runs in bf16, which xformers supports only on RTX 30xx and newer
# (ccc:86). Those GPUs are in the 24g partition. Rendering alone needs no bf16,
# but reconstructing a scene in the viewer does.
#SBATCH --partition=24g
#SBATCH --gres=gpu:1,ccc:86
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Starts the interactive 3DGEER viewer on a GPU node for two hours. The log
# prints the address, which is reachable from inside the VPN only; open it with
# camp tunnel. 24g cancels jobs that leave the GPU idle for two hours, so cancel
# the job with scancel when you are done.
#
#   sbatch --export=ALL,REPO_DIR=$PWD slurm/viewer.sh
#
# IMAGE picks the container, SCENES a directory of precomputed scene files.
set -euo pipefail
# The viewer needs the 3DGEER build with the projection tile mode, so its default image is the
# beyond 90 one rather than the shared p3dcv.sqsh of common.sh. Set before sourcing, because
# common.sh only fills IMAGE in when it is still empty.
export IMAGE="${IMAGE:-/mnt/home/$USER/envs/p3dcv-beyond90-alex.sqsh}"
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

SCENES="${SCENES:-/mnt/home/$USER/runs/viewer/scenes}"
OUT_DIR="${OUT_DIR:-/mnt/home/$USER/runs/viewer}"
# Like the Gradio demo: bind every interface of the node, which is what the
# tunnel connects to. The node sits inside the VPN and the viewer has no share
# mode of its own, so nothing of this reaches the internet.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-$((20000 + RANDOM % 10000))}"
mkdir -p "$SCENES" "$OUT_DIR"

NODE_IP=$(hostname -i | awk '{print $1}')
echo "URL: http://$NODE_IP:$PORT"
echo "Tunnel: camp tunnel $NODE_IP:$PORT"

in_container bash -c "cd '$REPO_DIR' && PYTHONPATH='$REPO_DIR' python -u -m viewer.server \
    --host $HOST --port $PORT \
    --scenes '$SCENES' --out '$OUT_DIR' \
    --pretrained '$PRETRAINED_DIR/erayzer_multi.pt' \
    --runs-dir '$RUNS_DIR' --data-root '$DATA_DIR/tartanair'"
