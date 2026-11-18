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

## What is missing

* A data loader for TartanAir that reads from the zip archives
* Ray conditioning and rendering for cameras that are not pinhole
* A job script that continues from `last.pt`, because jobs end after 24 hours
* A batch size and gradient accumulation that fit one 24 GB GPU
* Experiment tracking, for which `wandb` is already in the image

Checkpoints of our runs go to
`/mnt/projects/p3dcv/anycam/checkpoints/runs/<run name>/`. They are 3 GB
each, so delete the ones you no longer need.
