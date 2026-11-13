#!/bin/bash
#SBATCH --job-name=download-tartanair
#SBATCH --partition=data
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Downloads TartanAir V2 archives into the shared team folder and keeps them
# zipped. See docs/datasets/tartanair.md.
#
# Usage:
#   sbatch slurm/download_tartanair.sh                      # the small sample
#   sbatch slurm/download_tartanair.sh RetroOffice/Data_hard OldTownFall/Data_easy
#
# MODALITY and CAMERA select the archive, e.g. MODALITY=depth CAMERA=rcam_fish.
# Safe to re-run: complete archives are skipped, partial ones are resumed.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

MODALITY="${MODALITY:-image}"
CAMERA="${CAMERA:-lcam_fish}"
SOURCE=https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanair_v2

if [ "$#" -eq 0 ]; then
    set -- ArchVizTinyHouseNight/Data_hard
fi

for sequence in "$@"; do
    archive="$sequence/${MODALITY}_${CAMERA}.zip"
    target="$DATA_DIR/tartanair/$archive"
    mkdir -p "$(dirname "$target")"

    # The server reports the MD5 of the object as its ETag.
    expected="$(curl -sI "$SOURCE/$archive" | tr -d '\r' | awk 'tolower($1) == "etag:" {print $2}')"
    if [ -z "$expected" ]; then
        echo "NOT FOUND $archive" >&2
        exit 1
    fi
    if [ -f "$target" ] && [ "$(md5sum "$target" | cut -d' ' -f1)" = "$expected" ]; then
        echo "OK       $archive (already present)"
        continue
    fi

    echo "FETCH    $archive"
    wget --continue --no-verbose --tries=10 -O "$target" "$SOURCE/$archive"
    if [ "$(md5sum "$target" | cut -d' ' -f1)" != "$expected" ]; then
        echo "CHECKSUM MISMATCH for $archive" >&2
        exit 1
    fi
    echo "OK       $archive"
done

du -sh "$DATA_DIR/tartanair"
