#!/bin/bash
#SBATCH --job-name=build-image
#SBATCH --partition=12g
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Builds the container image from docker/Dockerfile and stores it as a single
# squashfs file in the shared team folder. Only needed when docker/ changes.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

TAG=p3dcv
TMP_IMAGE="/tmp/$USER-$TAG-$SLURM_JOB_ID.sqsh"

podman build -t "$TAG" -f "$REPO_DIR/docker/Dockerfile" "$REPO_DIR/docker"

rm -f "$TMP_IMAGE"
enroot import -o "$TMP_IMAGE" "podman://$TAG"

# Replace the shared image in one step so running jobs never see a partial file.
mkdir -p "$(dirname "$IMAGE")"
cp "$TMP_IMAGE" "$IMAGE.tmp"
mv "$IMAGE.tmp" "$IMAGE"
rm -f "$TMP_IMAGE"
podman rmi -f "$TAG" || true

ls -lh "$IMAGE"
in_container python "$REPO_DIR/scripts/check_env.py"
