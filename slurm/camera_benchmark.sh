#!/bin/bash
#SBATCH --job-name=camera-benchmark
#SBATCH --partition=24g
#SBATCH --gres=gpu:1,ccc:86
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Scores ONE checkpoint on ONE TartanAir environment against the fixed set of test cameras of
# scripts/camera_benchmark.py, whose true field of view and alpha are known by construction. Writes
# cameras.csv, summary.txt and one panel PNG per camera model family.
#
# Two kinds of checkpoint, exactly as in slurm/evaluate_model.sh:
#   * a run under $RUNS_DIR: its own config.yaml is used, so the camera settings of the run are kept,
#   * the released $PRETRAINED_DIR/erayzer_multi.pt: the repository config plus -s overrides.
#
# CAMERA_SET picks what is evaluated:
#   * remap (default): the 195 degree image_lcam_fish.zip of $ENV_NAME resampled to every test camera,
#   * front: the real image_lcam_front.zip of $ENV_NAME as it is, a 90 degree pinhole.
#
# IMAGE picks the container image and defaults to the shared one, see slurm/common.sh.
# Everything after the script name is passed on to camera_benchmark.py.
#
# Examples:
#   sbatch --export=ALL,RUN_NAME=b90-ucm-long,ENV_NAME=HongKong slurm/camera_benchmark.sh
#   sbatch --export=ALL,RUN_NAME=b90-ucm-long,ENV_NAME=Office,CAMERA_SET=front slurm/camera_benchmark.sh
#   sbatch --export=ALL,RUN_NAME=erayzer-multi,CKPT=/mnt/projects/p3dcv/anycam/checkpoints/pretrained/erayzer_multi.pt,\
#          CAMERA_MODEL=pinhole,ENV_NAME=HongKong slurm/camera_benchmark.sh
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

RUN_NAME="${RUN_NAME:-erayzer-multi}"
RUN_DIR="${RUN_DIR:-$RUNS_DIR/$RUN_NAME}"
ENV_NAME="${ENV_NAME:-HongKong}"
DIFFICULTY="${DIFFICULTY:-Data_hard}"
CAMERA_SET="${CAMERA_SET:-remap}"
DATA_PATH="${DATA_PATH:-$DATA_DIR/tartanair}"

# The remap set reads the fisheye archive, the front set the pinhole one of the same environment.
if [[ "$CAMERA_SET" == "front" ]]; then
    CAMERA="${CAMERA:-lcam_front}"
else
    CAMERA="${CAMERA:-lcam_fish}"
fi
# A directory name under $DATA_PATH, or a path to an image_*.zip archive of its own.
if [[ "$ENV_NAME" == *.zip || "$ENV_NAME" == /* ]]; then
    EVAL_PATH="$ENV_NAME"
else
    EVAL_PATH="$DATA_PATH/$ENV_NAME/$DIFFICULTY/image_$CAMERA.zip"
fi

# The run directory keeps the full config of the run, which carries its camera settings. Falling
# back to the repository config is what the released checkpoint needs, together with -s overrides.
if [[ -z "${CKPT:-}" && -f "$RUN_DIR/last.pt" ]]; then CKPT="$RUN_DIR/last.pt"; fi
CKPT="${CKPT:-$PRETRAINED_DIR/erayzer_multi.pt}"
if [[ -z "${CONFIG:-}" && -f "$RUN_DIR/config.yaml" ]]; then CONFIG="$RUN_DIR/config.yaml"; fi
CONFIG="${CONFIG:-config/erayzer_train_tartanair.yaml}"
OUT_DIR="${OUT_DIR:-/mnt/home/$USER/runs/bench/$RUN_NAME-$ENV_NAME-$CAMERA_SET-${SLURM_JOB_ID:-local}}"

if [[ ! -f "$CKPT" ]]; then echo "No checkpoint at $CKPT" >&2; exit 1; fi
if [[ ! -e "$EVAL_PATH" ]]; then echo "No data at $EVAL_PATH" >&2; exit 1; fi

SAMPLES="${SAMPLES:-10}"
SEED="${SEED:-20261128}"
SPAN="${SPAN:-45}"
MASK="${MASK:-circle}"
SUPERSAMPLE="${SUPERSAMPLE:-2}"

BENCH_ARGS=(
    --config "$CONFIG"
    --checkpoint "$CKPT"
    --data "$EVAL_PATH"
    --out-dir "$OUT_DIR"
    --camera-set "$CAMERA_SET"
    --samples "$SAMPLES"
    --seed "$SEED"
    --span "$SPAN"
    --mask "$MASK"
    --supersample "$SUPERSAMPLE"
)
if [[ -n "${FRONT_FOV:-}" ]]; then BENCH_ARGS+=(--front-fov "$FRONT_FOV"); fi
if [[ -n "${BASELINE_CSV:-}" ]]; then BENCH_ARGS+=(--baseline-csv "$BASELINE_CSV"); fi
if [[ -n "${CAMERAS:-}" ]]; then BENCH_ARGS+=(--cameras "$CAMERAS"); fi
# Camera settings of the released checkpoint, which has no run config to read them from.
if [[ -n "${CAMERA_MODEL:-}" ]]; then BENCH_ARGS+=(-s model.camera_model "$CAMERA_MODEL"); fi
if [[ -n "${F_SCALE:-}" ]]; then BENCH_ARGS+=(-s model.f_scale "$F_SCALE"); fi
if [[ -n "${NEAR:-}" ]]; then BENCH_ARGS+=(-s model.near_plane "$NEAR"); fi
if [[ -n "${ALPHA_INIT:-}" ]]; then BENCH_ARGS+=(-s model.alpha_init "$ALPHA_INIT"); fi
if [[ -n "${RENDERER:-}" ]]; then BENCH_ARGS+=(-s model.renderer "$RENDERER"); fi
if [[ -n "${GEER_TILE_MODE:-}" ]]; then BENCH_ARGS+=(-s model.geer_tile_mode "$GEER_TILE_MODE"); fi
BENCH_ARGS+=("$@")

mkdir -p "$OUT_DIR"
echo "Benchmarking $CKPT on $EVAL_PATH with $CONFIG, camera set $CAMERA_SET"
in_container bash -c "cd '$REPO_DIR' && python scripts/camera_benchmark.py \
    $(printf '%q ' "${BENCH_ARGS[@]}")"

cat "$OUT_DIR/summary.txt"
