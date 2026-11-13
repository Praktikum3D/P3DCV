#!/bin/bash
#SBATCH --job-name=download-pretrained
#SBATCH --partition=data
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Downloads the pretrained E-RayZer checkpoints and the VGG19 weights for the
# perceptual loss into the shared team folder. Safe to re-run: finished files
# are verified and skipped, partial files are resumed.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

mkdir -p "$PRETRAINED_DIR"
cd "$PRETRAINED_DIR"

# fetch <url> <file> <algorithm> <expected checksum>
fetch() {
    local url="$1" file="$2" algo="$3" expected="$4"
    if [ -f "$file" ] && [ "$("${algo}sum" "$file" | cut -d' ' -f1)" = "$expected" ]; then
        echo "OK       $file (already present)"
        return
    fi
    echo "FETCH    $file"
    wget --continue --no-verbose --tries=5 -O "$file" "$url"
    local actual
    actual="$("${algo}sum" "$file" | cut -d' ' -f1)"
    if [ "$actual" != "$expected" ]; then
        echo "CHECKSUM MISMATCH for $file: expected $expected, got $actual" >&2
        exit 1
    fi
    echo "OK       $file"
}

HF=https://huggingface.co/qitaoz/E-RayZer/resolve/main/checkpoints

# E-RayZer weights (Adobe Research License, see LICENSE-WEIGHTS upstream).
fetch "$HF/erayzer_multi.pt" erayzer_multi.pt sha256 \
    1994d8ca3b78757551f84863fa9d89fa712944e77225544a04487202d052bad4
fetch "$HF/erayzer_dl3dv.pt" erayzer_dl3dv.pt sha256 \
    56fd798d831e9c2a300932e5c41ead29a4ef3a084480c5bfc31d58ba3a834463

# MatConvNet VGG19, used by the E-RayZer perceptual loss.
fetch https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat \
    imagenet-vgg-verydeep-19.mat md5 106118b7cf60435e6d8e04f6a6dc3657

ls -lh "$PRETRAINED_DIR"
