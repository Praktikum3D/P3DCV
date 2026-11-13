# Training

The goal is to fine-tune a pretrained E-RayZer model on TartanAir. This
document describes what is in place and how a run is meant to be set up.

> **Not runnable yet.** The E-RayZer model and training code and the
> TartanAir data loader are not in this repository yet. The sections "How a
> run is set up" and "Job script" describe the intended workflow and will be
> updated once the code is imported.

## Pretrained weights

In `/mnt/projects/p3dcv/anycam/checkpoints/pretrained` (`$PRETRAINED_DIR`):

| File | Size | Content |
| --- | --- | --- |
| `erayzer_multi.pt` | 3.0 GB | E-RayZer trained on multiple datasets |
| `erayzer_dl3dv.pt` | 3.0 GB | E-RayZer trained on DL3DV only |
| `imagenet-vgg-verydeep-19.mat` | 0.5 GB | VGG19 for the perceptual loss |

They are downloaded and checksum-verified by
`sbatch ~/code/P3DCV/slurm/download_pretrained.sh`. The E-RayZer weights come
from <https://huggingface.co/qitaoz/E-RayZer> under the Adobe Research
License and must not be committed to this repository.

## What upstream E-RayZer provides

From the upstream [training documentation](https://github.com/QitaoZhao/E-RayZer/blob/main/docs/train.md)
and `config/erayzer_train_dl3dv.yaml`:

- Entry point `train.py` with a YAML config, launched through
  `scripts/train_dl3dv.sh`; settings are overridden with environment
  variables.
- Resolution 256 x 256, patch size 16, 10 views per sample (5 input, 5
  target).
- Released setup: 8 GPUs, batch size 24 per GPU, 152 000 steps.
- Loss: MSE plus 0.2 times the VGG19 perceptual loss.
- Fine-tuning from a checkpoint with a fresh optimizer:
  `RESUME_CKPT=<ckpt> RESET_TRAINING_STATE=true`.
  Continuing an interrupted run: `RESET_TRAINING_STATE=false`.
- A checkpoint every 2000 steps in `CHECKPOINT_DIR`, the latest also as
  `last.pt`.
- Only a DL3DV data loader exists. Per sequence it expects covisibility and
  DINO feature-similarity files, which drive the view-sampling curriculum.
- Logging is `print` to stdout only.

## What changes for us

- **One GPU instead of eight.** A student job gets a single 24 GB GPU, so the
  batch size has to shrink and gradient accumulation has to make up for it.
  The batch size that fits is still to be measured.
- **24 hour limit.** Every run has to continue from `last.pt` when it is
  resubmitted.
- **TartanAir loader.** A new loader that reads from the zip archives, plus a
  replacement for the covisibility files (TartanAir has ground-truth poses).
- **Non-pinhole cameras.** The ray conditioning and the Gaussian renderer of
  E-RayZer assume a pinhole camera; both have to change for fisheye input.
- **Experiment tracking.** Weights & Biases is installed in the image.

## How a run is set up

Each run gets a name and its own folder:

```
/mnt/projects/p3dcv/anycam/checkpoints/runs/<run-name>/
  config.yaml      the exact config of the run
  last.pt          latest checkpoint, used to continue
  step_*.pt        periodic checkpoints
```

Use a descriptive name such as `2026-11-20_fisheye-baseline_alex`. Checkpoints
are 3 GB each, so delete intermediate ones you no longer need; the quota is
shared.

## Job script

A training job follows the same pattern as the other scripts in `slurm/`:

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=24g
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

RUN_NAME="$1"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
mkdir -p "$RUN_DIR"

# Continue from the last checkpoint if there is one, otherwise start from
# the pretrained model.
if [ -f "$RUN_DIR/last.pt" ]; then
    RESUME_CKPT="$RUN_DIR/last.pt" RESET_TRAINING_STATE=false
else
    RESUME_CKPT="$PRETRAINED_DIR/erayzer_multi.pt" RESET_TRAINING_STATE=true
fi

# Pass settings explicitly; do not rely on the container inheriting the
# environment of the job.
in_container env \
    RESUME_CKPT="$RESUME_CKPT" \
    RESET_TRAINING_STATE="$RESET_TRAINING_STATE" \
    CHECKPOINT_DIR="$RUN_DIR" \
    <training command>
```

Submitting the same command again continues the run:

```bash
sbatch slurm/train.sh 2026-11-20_fisheye-baseline_alex
```

Before a long run on `24g`, test the same configuration for a few steps on
`12g`. `24g` cancels jobs that use less than 12 GB of GPU memory or leave the
GPU idle.
