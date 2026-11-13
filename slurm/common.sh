# Shared paths for all job scripts. Source this file, do not execute it.
# Every value can be overridden from the environment.

# Shared team folder: data, checkpoints and the container image.
export PROJECT_DIR="${PROJECT_DIR:-/mnt/projects/p3dcv/anycam}"
# Personal clone of this repository.
export REPO_DIR="${REPO_DIR:-/mnt/home/$USER/code/P3DCV}"

export DATA_DIR="$PROJECT_DIR/data"
export PRETRAINED_DIR="$PROJECT_DIR/checkpoints/pretrained"
export RUNS_DIR="$PROJECT_DIR/checkpoints/runs"
export IMAGE="${IMAGE:-$PROJECT_DIR/envs/p3dcv.sqsh}"

# Run a command inside the container with /mnt and /tmp mounted.
in_container() {
    enroot start --root --rw --mount /mnt:mnt --mount /tmp:tmp "$IMAGE" "$@"
}
