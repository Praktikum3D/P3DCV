#!/bin/bash
#SBATCH --job-name=train-tartanair
#SBATCH --partition=24g
#SBATCH --gres=gpu:1,ccc:86
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Fine-tunes E-RayZer on zipped TartanAir V2 archives on one 24 GB GPU.
# Archives are read directly from $DATA_PATH without unpacking.
#
# Checkpoints go to $RUNS_DIR/$RUN_NAME. If a checkpoint already exists there,
# the job resumes from the newest one; otherwise it starts from INIT_CKPT.
#
# Example:
#   sbatch --export=ALL,RUN_NAME=tartanair-fish slurm/train_tartanair.sh
#   sbatch slurm/train_tartanair.sh -s training.lr 0.00005
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

RUN_NAME="${RUN_NAME:-tartanair-finetune}"
RUN_DIR="${RUN_DIR:-$RUNS_DIR/$RUN_NAME}"
INIT_CKPT="${INIT_CKPT:-$PRETRAINED_DIR/erayzer_multi.pt}"
DATA_PATH="${DATA_PATH:-$DATA_DIR/tartanair}"
CAMERA="${CAMERA:-lcam_fish}"
KEEP_CKPTS="${KEEP_CKPTS:-2}"

# Defaults for one 24 GB GPU (effective batch size = BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS).
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MAX_STEPS="${MAX_STEPS:-20000}"
LR="${LR:-0.0001}"
WARMUP="${WARMUP:-500}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
# Weights & Biases logging. Set WANDB_PROJECT= (empty) to turn it off.
# The key file holds one line WANDB_API_KEY=... and should be readable only by you.
WANDB_PROJECT="${WANDB_PROJECT-p3dcv}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/mnt/home/$USER/.secrets/wandb.env}"
VIS_EVERY="${VIS_EVERY:-500}"

mkdir -p "$RUN_DIR"

prune_checkpoints() {
    { ls -1 "$RUN_DIR"/ckpt_*.pt 2>/dev/null || true; } | sort | head -n "-$KEEP_CKPTS" | xargs -r rm -f --
}

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

PRUNER_PID=""
cleanup() {
    [[ -n "$PRUNER_PID" ]] && kill "$PRUNER_PID" 2>/dev/null || true
}
trap cleanup EXIT

( while sleep 600; do prune_checkpoints; done ) &
PRUNER_PID=$!

EXTRA_ARGS=(
    -s dataset.train_path "$DATA_PATH"
    -s dataset.camera_name "$CAMERA"
    -s training.checkpoint_dir "$RUN_DIR"
    -s training.resume_ckpt "$RESUME_CKPT"
    -s training.reset_training_state "$RESET_TRAINING_STATE"
    -s training.batch_size_per_gpu "$BATCH_SIZE_PER_GPU"
    -s training.grad_accum_steps "$GRAD_ACCUM_STEPS"
    -s training.max_fwdbwd_passes "$MAX_STEPS"
    -s training.lr "$LR"
    -s training.warmup "$WARMUP"
    -s training.checkpoint_every "$CHECKPOINT_EVERY"
    -s training.num_workers "$NUM_WORKERS"
    -s training.perceptual_loss_weight_path "$PRETRAINED_DIR/imagenet-vgg-verydeep-19.mat"
    -s training.vis_every "$VIS_EVERY"
    "$@"
)
if [[ -n "$WANDB_PROJECT" ]]; then
    EXTRA_ARGS=(
        -s training.wandb_project "$WANDB_PROJECT"
        -s training.wandb_run_name "$RUN_NAME"
        -s training.wandb_key_file "$WANDB_KEY_FILE"
        "${EXTRA_ARGS[@]}"
    )
fi

in_container bash -c "cd '$REPO_DIR' && torchrun --standalone --nproc_per_node 1 train.py \
    --config config/erayzer_train_tartanair.yaml $(printf '%q ' "${EXTRA_ARGS[@]}")"

prune_checkpoints
