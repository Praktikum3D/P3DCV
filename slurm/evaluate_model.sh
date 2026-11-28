#!/bin/bash
#SBATCH --job-name=eval-tartanair
#SBATCH --partition=24g
#SBATCH --gres=gpu:1,ccc:86
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Evaluates ONE checkpoint on ONE TartanAir environment, so several models can be compared on the
# same views. Writes metrics.txt, summary.csv, the example renders, the ground truth next to the
# rendering for SAMPLE_VIEWS and NOVEL_VIEWS renders from poses no input view had.
#
# Two kinds of checkpoint:
#   * a run under $RUNS_DIR: its own config.yaml is used, so the camera settings of the run are kept.
#   * the released $PRETRAINED_DIR/erayzer_multi.pt: the repository config plus -s overrides.
#
# IMAGE picks the container image and defaults to the shared one, see slurm/common.sh.
#
# Examples:
#   sbatch --export=ALL,RUN_NAME=b90-fisheye-long,ENV_NAME=HongKong slurm/evaluate_model.sh
#   sbatch --export=ALL,RUN_NAME=erayzer-multi,CKPT=/mnt/projects/p3dcv/anycam/checkpoints/pretrained/erayzer_multi.pt,\
#          CAMERA_MODEL=pinhole,ENV_NAME=HongKong slurm/evaluate_model.sh -s evaluation.mask_override circle
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

RUN_NAME="${RUN_NAME:-erayzer-multi}"
RUN_DIR="${RUN_DIR:-$RUNS_DIR/$RUN_NAME}"
ENV_NAME="${ENV_NAME:-HongKong}"
DIFFICULTY="${DIFFICULTY:-Data_hard}"
CAMERA="${CAMERA:-lcam_fish}"
DATA_PATH="${DATA_PATH:-$DATA_DIR/tartanair}"
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
OUT_DIR="${OUT_DIR:-/mnt/home/$USER/runs/eval/$RUN_NAME-$ENV_NAME-${SLURM_JOB_ID:-local}}"

if [[ ! -f "$CKPT" ]]; then echo "No checkpoint at $CKPT" >&2; exit 1; fi

# Scoring options. MASK_OVERRIDE=circle scores every model, pinhole included, on the image circle.
MASK_OVERRIDE="${MASK_OVERRIDE:-}"
SAMPLE_VIEWS="${SAMPLE_VIEWS:-0,2,4}"
NOVEL_VIEWS="${NOVEL_VIEWS:-3}"
NOVEL_OFFSET="${NOVEL_OFFSET:-0.3}"
MAX_BATCHES="${MAX_BATCHES:-0}"
SAVE_EXAMPLES="${SAVE_EXAMPLES:-8}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
# Pinned so every model sees the same views, whatever curriculum its own config was trained with.
MIN_FRAME_DIST="${MIN_FRAME_DIST:-15}"
MAX_FRAME_DIST="${MAX_FRAME_DIST:-45}"

EVAL_ARGS=(
    -s dataset.name tartanair
    -s dataset.eval_path "$EVAL_PATH"
    -s dataset.camera_name "$CAMERA"
    -s training.dataset tartanair
    -s data_repeat 1
    -s evaluation.checkpoint_path "$CKPT"
    -s evaluation.out_dir "$OUT_DIR"
    -s evaluation.max_batches "$MAX_BATCHES"
    -s evaluation.save_examples "$SAVE_EXAMPLES"
    -s evaluation.sample_views "$SAMPLE_VIEWS"
    -s evaluation.novel_views "$NOVEL_VIEWS"
    -s evaluation.novel_offset "$NOVEL_OFFSET"
    -s training.batch_size_per_gpu "$BATCH_SIZE_PER_GPU"
    -s training.num_workers "$NUM_WORKERS"
    -s training.view_selector.type two_frame
    -s training.view_selector.use_curriculum false
    -s training.view_selector.min_frame_dist "$MIN_FRAME_DIST"
    -s training.view_selector.max_frame_dist "$MAX_FRAME_DIST"
)
if [[ -n "$MASK_OVERRIDE" ]]; then
    EVAL_ARGS+=(-s evaluation.mask_override "$MASK_OVERRIDE")
fi
# Camera settings of the released checkpoint, which has no run config to read them from.
if [[ -n "${CAMERA_MODEL:-}" ]]; then EVAL_ARGS+=(-s model.camera_model "$CAMERA_MODEL"); fi
if [[ -n "${F_SCALE:-}" ]]; then EVAL_ARGS+=(-s model.f_scale "$F_SCALE"); fi
if [[ -n "${NEAR:-}" ]]; then EVAL_ARGS+=(-s model.near_plane "$NEAR"); fi
if [[ -n "${ALPHA_INIT:-}" ]]; then EVAL_ARGS+=(-s model.alpha_init "$ALPHA_INIT"); fi
if [[ -n "${RENDERER:-}" ]]; then EVAL_ARGS+=(-s model.renderer "$RENDERER"); fi
if [[ -n "${GEER_TILE_MODE:-}" ]]; then EVAL_ARGS+=(-s model.geer_tile_mode "$GEER_TILE_MODE"); fi
EVAL_ARGS+=("$@")

mkdir -p "$OUT_DIR"
echo "Evaluating $CKPT on $EVAL_PATH with $CONFIG"
in_container bash -c "cd '$REPO_DIR' && torchrun --standalone --nproc_per_node 1 evaluate.py \
    --config '$CONFIG' $(printf '%q ' "${EVAL_ARGS[@]}")"

cat "$OUT_DIR/metrics.txt"
