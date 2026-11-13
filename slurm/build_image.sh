#!/bin/bash
#SBATCH --job-name=build-image
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Builds the container image from docker/Dockerfile and stores it as a single
# squashfs file in the shared team folder. Only needed when docker/ changes.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

TAG=p3dcv
set -x

podman build -t "$TAG" -f "$REPO_DIR/docker/Dockerfile" "$REPO_DIR/docker"

# Write next to the final location and rename at the end, so running jobs
# never see a partial image. The image is too large for /tmp.
mkdir -p "$(dirname "$IMAGE")"
rm -f "$IMAGE.tmp"
enroot import -o "$IMAGE.tmp" "podman://$TAG"
mv "$IMAGE.tmp" "$IMAGE"

ls -lh "$IMAGE"
echo "Image written. Verify it with: sbatch $REPO_DIR/slurm/check_env.sh"
