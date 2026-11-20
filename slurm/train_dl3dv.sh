#!/bin/bash
#SBATCH --job-name=train-dl3dv
#SBATCH --partition=24g
#SBATCH --gres=gpu:1,ccc:86
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Fine-tunes E-RayZer on the DL3DV subset in data/dl3dv/benchmark_fd96.zip on
# one GPU. See docs/training.md.
#
# The archive is unpacked to the local disk of the node, and the shared data
# folder is never written to. HOLDOUT scenes are kept out of training. Their
# ids are written to the run folder so that slurm/evaluate_dl3dv.sh can use
# them through SCENE_LIST.
#
# Checkpoints go to $RUNS_DIR/$RUN_NAME. If that folder already has a
# checkpoint, the job continues from the newest one. Otherwise it starts from
# INIT_CKPT. To train longer than 24 hours, queue several jobs that run one
# after another:
#   for i in 1 2 3; do sbatch --job-name=train-myrun --dependency=singleton \
#       --export=ALL,RUN_NAME=myrun slurm/train_dl3dv.sh; done
#
# Every argument is passed on to train.py as an extra override, e.g.
#   sbatch slurm/train_dl3dv.sh -s training.l1_loss_weight 0.1
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

RUN_NAME="${RUN_NAME:-dl3dv-finetune}"
RUN_DIR="${RUN_DIR:-$RUNS_DIR/$RUN_NAME}"
INIT_CKPT="${INIT_CKPT:-$PRETRAINED_DIR/erayzer_dl3dv.pt}"
ARCHIVE="${ARCHIVE:-$DATA_DIR/dl3dv/benchmark_fd96.zip}"
HOLDOUT="${HOLDOUT:-20}"
# Number of ckpt_*.pt files kept in RUN_DIR. Each one is several GB.
KEEP_CKPTS="${KEEP_CKPTS:-2}"

# Training settings for one 24 GB GPU. The effective batch size is
# BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS. Keep MAX_STEPS the same for every
# job of a run, the learning rate schedule depends on it.
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MAX_STEPS="${MAX_STEPS:-20000}"
LR="${LR:-0.0001}"
WARMUP="${WARMUP:-500}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-6}"

mkdir -p "$RUN_DIR"

prune_checkpoints() {
    # ckpt_<16-digit step>.pt sorts by step. Only older files are removed,
    # never the one being written.
    { ls -1 "$RUN_DIR"/ckpt_*.pt 2>/dev/null || true; } | sort | head -n "-$KEEP_CKPTS" | xargs -r rm -f --
}

# Continue from the newest checkpoint of this run, if there is one.
LATEST="$(ls -1t "$RUN_DIR"/ckpt_*.pt "$RUN_DIR"/last.pt 2>/dev/null | head -n 1 || true)"
if [[ -n "$LATEST" ]]; then
    RESUME_CKPT="$LATEST"
    RESET_TRAINING_STATE=false
    echo "Continuing run $RUN_NAME from $RESUME_CKPT"
else
    RESUME_CKPT="$INIT_CKPT"
    RESET_TRAINING_STATE=true
    echo "Starting run $RUN_NAME from $RESUME_CKPT"
fi
prune_checkpoints

WORK_DIR="/tmp/$USER-dl3dv-train-${SLURM_JOB_ID:-$$}"
PRUNER_PID=""
cleanup() {
    [[ -n "$PRUNER_PID" ]] && kill "$PRUNER_PID" 2>/dev/null || true
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT
mkdir -p "$WORK_DIR"

in_container python -m zipfile -e "$ARCHIVE" "$WORK_DIR"
in_container python "$REPO_DIR/scripts/prepare_dl3dv_train.py" "$WORK_DIR" \
    --holdout "$HOLDOUT" --holdout-list "$RUN_DIR/holdout_scenes.txt"

( while sleep 600; do prune_checkpoints; done ) &
PRUNER_PID=$!

# The scenes have exactly 10 frames with an image and training uses 10 views
# (5 input, 5 target), so the random view selector takes all of them and
# random_split chooses which are input. Covisibility labels are not needed.
EXTRA_ARGS=(
    -s data_repeat 100
    -s training.lr "$LR"
    -s training.warmup "$WARMUP"
    -s training.grad_accum_steps "$GRAD_ACCUM_STEPS"
    "$@"
)
in_container bash -c "cd '$REPO_DIR' && env NUM_GPUS=1 CHECK_MANIFEST=false \
    MANIFEST_PATH='$WORK_DIR/manifests/train.txt' \
    VIEW_SELECTOR_TYPE=random USE_CURRICULUM=false \
    CHECKPOINT_DIR='$RUN_DIR' RESUME_CKPT='$RESUME_CKPT' RESET_TRAINING_STATE=$RESET_TRAINING_STATE \
    PERCEPTUAL_LOSS_WEIGHT_PATH='$PRETRAINED_DIR/imagenet-vgg-verydeep-19.mat' \
    MAX_STEPS=$MAX_STEPS BATCH_SIZE_PER_GPU=$BATCH_SIZE_PER_GPU NUM_WORKERS=$NUM_WORKERS \
    CHECKPOINT_EVERY=$CHECKPOINT_EVERY \
    bash scripts/train_dl3dv.sh $(printf '%q ' "${EXTRA_ARGS[@]}")"

prune_checkpoints
