# Training

The goal is to continue training a pretrained E-RayZer model on TartanAir.
This is not set up yet. The training code runs in the container, but the only
data loader is the one for DL3DV.

## Pretrained weights

In `/mnt/projects/p3dcv/anycam/checkpoints/pretrained`, downloaded and
verified by `slurm/download_pretrained.sh`.

| File | Size | Content |
| --- | --- | --- |
| `erayzer_multi.pt` | 3.0 GB | E-RayZer trained on multiple datasets |
| `erayzer_dl3dv.pt` | 3.0 GB | E-RayZer trained on DL3DV only |
| `imagenet-vgg-verydeep-19.mat` | 0.5 GB | VGG19 for the perceptual loss |

The E-RayZer weights are under the Adobe Research License and must not be
committed.

## What the code provides

* `train.py` with `config/erayzer_train_dl3dv.yaml`, started through
  `scripts/train_dl3dv.sh` and configured with environment variables
* Resolution 256 x 256, patch size 16, 5 input and 5 target views per sample
* Released setup with 8 GPUs, batch size 24 per GPU and 152 000 steps
* Loss of MSE plus 0.2 times the VGG19 perceptual loss
* Start from a checkpoint with `RESUME_CKPT` and `RESET_TRAINING_STATE=true`,
  continue an interrupted run with `RESET_TRAINING_STATE=false`
* A checkpoint every 2000 steps in `CHECKPOINT_DIR`, the latest as `last.pt`
* Logging through `print` only

A test with two steps, batch size 2 and `VIEW_SELECTOR_TYPE=two_frame` ran on
one RTX 3090 in November 2026.

## Fine-tuning on the DL3DV subset

`slurm/train_dl3dv.sh` fine-tunes `erayzer_dl3dv.pt` on one 24 GB GPU with the
140 scenes of `data/dl3dv/benchmark_fd96.zip`. The job unpacks the archive to
`/tmp` and runs `scripts/prepare_dl3dv_train.py`. That script keeps the 10
frames per scene that have an image and holds 20 scenes out of training. The
shared data folder is only read.

```bash
sbatch --export=ALL,RUN_NAME=myrun slurm/train_dl3dv.sh
# longer than 24 hours: queue jobs that continue one after another
for i in 1 2 3; do sbatch --job-name=train-myrun --dependency=singleton \
    --export=ALL,RUN_NAME=myrun slurm/train_dl3dv.sh; done
# evaluate on the held-out scenes only
sbatch --export=ALL,CKPT=$RUNS/myrun/last.pt,SCENE_LIST=$RUNS/myrun/holdout_scenes.txt \
    slurm/evaluate_dl3dv.sh
```

with `RUNS=/mnt/projects/p3dcv/anycam/checkpoints/runs`. A job continues from
the newest checkpoint in the run folder and keeps the two newest `ckpt_*.pt`.
Defaults: batch size 2 with 4 accumulation steps, learning rate 1e-4 with 500
warmup steps, 20 000 steps. They can be changed with `BATCH_SIZE_PER_GPU`,
`GRAD_ACCUM_STEPS`, `LR`, `WARMUP` and `MAX_STEPS`. On an RTX 3090 one step
takes about 0.5 s.

The 140 scenes are the DL3DV test split of the paper. Numbers on all 140
scenes after fine-tuning are not comparable to the paper, so report the
held-out scenes.

## What is missing

* A data loader for TartanAir that reads from the zip archives
* Ray conditioning and rendering for cameras that are not pinhole
* Experiment tracking, for which `wandb` is already in the image

Checkpoints of our runs go to
`/mnt/projects/p3dcv/anycam/checkpoints/runs/<run name>/`. They are 3 GB
each, so delete the ones you no longer need.
