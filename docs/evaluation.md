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
