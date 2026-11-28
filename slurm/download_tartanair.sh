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
# TARGET_DIR redirects the download, e.g. TARGET_DIR=/mnt/home/$USER/data/tartanair
# for a private copy. It has to be set here because common.sh derives DATA_DIR
# from PROJECT_DIR and other scripts rely on that.
# Safe to re-run: complete archives are skipped, partial ones are resumed.
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

MODALITY="${MODALITY:-image}"
CAMERA="${CAMERA:-lcam_fish}"
TARGET_DIR="${TARGET_DIR:-$DATA_DIR/tartanair}"
SOURCE=https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanair_v2

if [ "$#" -eq 0 ]; then
    set -- ArchVizTinyHouseNight/Data_hard
fi

# Verifies the downloaded file against the headers of the remote object.
# The ETag of a Swift static large object is the MD5 of the concatenated
# segment MD5s, not the MD5 of the content, so an MD5 check can never pass for
# the large archives. Those are verified by size instead.
verify_archive() {
    local target="$1" is_slo="$2" expected_len="$3" expected_md5="$4"
    [ -f "$target" ] || return 1
    if [ "$is_slo" = yes ]; then
        [ "$(stat -c%s "$target")" = "$expected_len" ]
    else
        [ -n "$expected_md5" ] && [ "$(md5sum "$target" | cut -d' ' -f1)" = "$expected_md5" ]
    fi
}

for sequence in "$@"; do
    archive="$sequence/${MODALITY}_${CAMERA}.zip"
    target="$TARGET_DIR/$archive"
    mkdir -p "$(dirname "$target")"

    headers="$(curl -sI "$SOURCE/$archive" | tr -d '\r')"
    expected_md5="$(printf '%s\n' "$headers" | awk 'tolower($1) == "etag:" {gsub(/"/, "", $2); print $2}')"
    expected_len="$(printf '%s\n' "$headers" | awk 'tolower($1) == "content-length:" {print $2}')"
    if printf '%s\n' "$headers" | grep -qi '^x-static-large-object: *true'; then
        is_slo=yes
    else
        is_slo=no
    fi
    if [ -z "$expected_len" ]; then
        echo "NOT FOUND $archive" >&2
        exit 1
    fi

    if verify_archive "$target" "$is_slo" "$expected_len" "$expected_md5"; then
        echo "OK       $archive (already present)"
        continue
    fi

    echo "FETCH    $archive ($expected_len bytes, slo=$is_slo)"
    wget --continue --no-verbose --tries=10 -O "$target" "$SOURCE/$archive"
    if ! verify_archive "$target" "$is_slo" "$expected_len" "$expected_md5"; then
        echo "VERIFY FAILED for $archive" >&2
        exit 1
    fi
    echo "OK       $archive"
done

du -sh "$TARGET_DIR"
