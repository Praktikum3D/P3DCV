#!/bin/bash
#SBATCH --job-name=dev
# E-RayZer runs in bf16, which xformers supports only on RTX 30xx and newer
# (ccc:86). Those GPUs are in the 24g partition.
#SBATCH --partition=24g
#SBATCH --gres=gpu:1,ccc:86
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Interactive work on a GPU node. Starts an SSH server on the node for
# VS Code Remote SSH and Jupyter Lab inside the container. Both are reachable
# from inside the VPN only. The connection details go to the log and to
# ~/runs/dev-current.env, which is readable only by you because it holds the
# Jupyter token. 24g cancels jobs that leave the GPU idle for two hours, so
# cancel the job with scancel when you are done.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

IP=$(hostname -i)
SSH_PORT=$((20000 + RANDOM % 5000))
JUPYTER_PORT=$((25000 + RANDOM % 5000))
TOKEN=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')

# Cluster helper, starts sshd as your user with your ~/.ssh/authorized_keys.
start-ssh-server "$SSH_PORT"

INFO="/mnt/home/$USER/runs/dev-current.env"
umask 077
cat > "$INFO" <<EOF
JOB=$SLURM_JOB_ID
NODE=$(hostname)
IP=$IP
SSH_PORT=$SSH_PORT
JUPYTER_PORT=$JUPYTER_PORT
TOKEN=$TOKEN
EOF

echo "SSH: ssh -p $SSH_PORT $USER@$IP"
echo "Jupyter in a browser: http://$IP:$JUPYTER_PORT/lab?token=$TOKEN"
echo "Jupyter for VS Code on the node: http://127.0.0.1:$JUPYTER_PORT/?token=$TOKEN"
echo "Shell inside the container: bash $REPO_DIR/slurm/shell.sh"

in_container bash -c "cd /mnt/home/$USER && jupyter lab --no-browser --allow-root \
    --ip 0.0.0.0 --port $JUPYTER_PORT --ServerApp.token=$TOKEN \
    --notebook-dir /mnt/home/$USER"
