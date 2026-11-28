#!/bin/bash
#SBATCH --job-name=build-image
#SBATCH --partition=12g
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Builds the container image from docker/Dockerfile and stores it as a single
# squashfs file in the shared team folder. Only needed when docker/ changes.
#
# This job is pure CPU and disk work, so it asks for no GPU. The cluster
# reports jobs that reserve a GPU without using one. If the scheduler should
# refuse jobs without --gres on the GPU partition 12g, run the build on the
# CPU partition instead: sbatch --partition=data slurm/build_image.sh
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

# IMAGE sets the output file, GSPLAT_REPO and GSPLAT_REF another gsplat (see docker/Dockerfile).
TAG="p3dcv-$(basename "$IMAGE" .sqsh)"

podman build -t "$TAG" ${GSPLAT_REPO:+--build-arg GSPLAT_REPO="$GSPLAT_REPO"} \
    ${GSPLAT_REF:+--build-arg GSPLAT_REF="$GSPLAT_REF"} \
    -f "$REPO_DIR/docker/Dockerfile" "$REPO_DIR/docker"

# Write next to the final location and rename at the end, so running jobs
# never see a partial image. The image is too large for /tmp.
mkdir -p "$(dirname "$IMAGE")"
rm -f "$IMAGE.tmp"
# On this cluster `enroot import` exits with status 127 after it has written
# the complete image, so its exit status is ignored and the result is
# verified instead.
enroot import -o "$IMAGE.tmp" "podman://$TAG" || echo "enroot import exited with status $?"
[ "$(head -c 4 "$IMAGE.tmp")" = hsqs ] || { echo "no squashfs image was written" >&2; exit 1; }
chmod 660 "$IMAGE.tmp"
mv "$IMAGE.tmp" "$IMAGE"

ls -lh "$IMAGE"
echo "Image written. Verify it with: sbatch $REPO_DIR/slurm/check_env.sh"
