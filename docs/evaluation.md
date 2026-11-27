# Evaluation

`slurm/evaluate_dl3dv.sh` evaluates a checkpoint on the DL3DV benchmark with
the protocol of the E-RayZer paper. These numbers are the reference for our
own models.

## Running it

```bash
sbatch ~/code/P3DCV/slurm/evaluate_dl3dv.sh
CKPT=/mnt/projects/p3dcv/anycam/checkpoints/pretrained/erayzer_multi.pt \
    sbatch ~/code/P3DCV/slurm/evaluate_dl3dv.sh
cat ~/runs/eval-dl3dv-*.out
```

The job takes about 2 minutes on one GPU. The default checkpoint is
`erayzer_dl3dv.pt`. `ARCHIVE` selects another benchmark archive, see
[datasets/dl3dv.md](datasets/dl3dv.md), and `OUT_DIR` another output folder.

## Protocol

Per scene the model gets 10 views that span 96 frames.

```
selected  0 10 21 32 42 53 64 74 85 96
input     0 21 42 64 96
target    10 32 53 74 85
```

Images are cropped to the central square and resized to 256 x 256. The poses
of the dataset are used only to score the predicted poses.

| Metric | Meaning |
| --- | --- |
| PSNR | Rendered views against the real images, higher is better |
| RPA@5, RPA@15, RPA@30 | Share of view pairs whose relative rotation and translation direction are within 5, 15 and 30 degrees of the ground truth |

## Results

Measured on 17 November 2026 on all 140 scenes, against the values reported
for the [released code](https://github.com/QitaoZhao/E-RayZer/blob/4309c76/docs/eval.md).

| Checkpoint | | PSNR | RPA@5 | RPA@15 | RPA@30 |
| --- | --- | --- | --- | --- | --- |
| `erayzer_dl3dv.pt` | ours | 20.3168 | 0.7206 | 0.8848 | 0.9352 |
| | reported | 20.3273 | 0.7210 | 0.8844 | 0.9352 |
| `erayzer_multi.pt` | ours | 19.7205 | 0.6005 | 0.8287 | 0.9017 |
| | reported | 19.7037 | 0.5986 | 0.8286 | 0.9017 |

`erayzer_dl3dv.pt` is the checkpoint behind the DL3DV row of the paper.

## Output

`~/runs/eval/dl3dv-<checkpoint>-<job id>/` holds `metrics.txt` with the
averages, `summary.csv` with one row per scene and `sample_*` folders with
predicted and target renders. PSNR per scene ranges from about 14 to 30, so
compare models on the full split and not on a few scenes.

## Comparing models on TartanAir

`slurm/evaluate_model.sh` evaluates one checkpoint on one TartanAir
environment, so several models can be put side by side on the same views.

```bash
sbatch --export=ALL,RUN_NAME=b90-fisheye-long,ENV_NAME=HongKong,MASK_OVERRIDE=circle \
    slurm/evaluate_model.sh
```

A run under `checkpoints/runs/` brings its own `config.yaml`, which carries
the camera settings it was trained with. The released `erayzer_multi.pt` has
none, so it takes the repository config plus `CAMERA_MODEL`, `F_SCALE`,
`NEAR`, `ALPHA_INIT`, `RENDERER` and `GEER_TILE_MODE`. The view selection is
pinned in the script, so every model sees the same frames whatever curriculum
its own config was trained with.

### Scoring options

| Option | Meaning |
| --- | --- |
| `evaluation.mask_override` | Unset scores every model on the mask its own training loss used. `circle` scores every model, pinhole included, on the circle spanning the image width without the 90 degree cap, which is what makes a pinhole model and a 195 degree fisheye model comparable |
| `evaluation.sample_views` | Target view indices, for example `0,2,4`, written as `panel_<view>.png` with the real image next to the rendering |
| `evaluation.novel_views` | How many poses halfway between the input poses, moved sideways by `evaluation.novel_offset` scene radii, to render as `novel_*.png`. They have no ground truth and carry no score, they show what the model does away from its input views |

The circle mask is still intersected with the range the camera model itself
is defined on. For a UCM camera that range shrinks with a large alpha and a
short focal length, roughly below a focal length of half the image width times
the square root of `2 alpha - 1`, and such a view would be scored on fewer
pixels than the models it is compared against. At the values the b90 runs
learned the circle survives untouched; when it does not, the job log says so
and names the share of the circle that is missing.

`metrics.txt` and `summary.csv` report `psnr` and `ssim` on the active mask,
`psnr_unmasked` and `ssim_unmasked` on the whole frame, and `mask_fraction`,
the share of pixels the mask covers. SSIM is the usual one, an 11 by 11
Gaussian window with sigma 1.5 over a data range of 1, implemented in
`erayzer_core/utils/image_metrics.py` because the container image carries
neither torchmetrics nor skimage. LPIPS would need a package the image does
not have, so it is not reported.
