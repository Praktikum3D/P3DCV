#!/bin/bash
# Opens a shell inside the container. Use it in an SSH session on a node
# that runs slurm/dev.sh, for example in the VS Code terminal. Not a job
# script, do not submit it with sbatch.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"
# SSH sessions miss the variables Slurm sets for jobs, and enroot needs them.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-/tmp/.local/share}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"
in_container bash -c "cd '$REPO_DIR' && exec bash"
