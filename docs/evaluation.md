# Evaluation

`slurm/evaluate_dl3dv.sh` evaluates a checkpoint on the DL3DV benchmark with
the protocol of the E-RayZer paper. It is the reference point for our own
models: a change should be compared against these numbers.

## Running it

```bash
sbatch ~/code/P3DCV/slurm/evaluate_dl3dv.sh                                     # erayzer_dl3dv.pt
CKPT=/mnt/projects/p3dcv/anycam/checkpoints/pretrained/erayzer_multi.pt \
    sbatch ~/code/P3DCV/slurm/evaluate_dl3dv.sh
cat ~/runs/eval-dl3dv-*.out
```

The job takes about 2 minutes on one GPU. It unpacks the benchmark archive
(see [datasets/dl3dv.md](datasets/dl3dv.md)) to the local disk of the node and
runs the upstream `scripts/evaluate_dl3dv.sh` in the container. Like all
E-RayZer jobs it needs an RTX 30xx or newer (see [inference.md](inference.md)).

| Variable | Default | Meaning |
| --- | --- | --- |
| `CKPT` | `$PRETRAINED_DIR/erayzer_dl3dv.pt` | checkpoint to evaluate |
| `ARCHIVE` | `$DATA_DIR/dl3dv/benchmark_fd96.zip` | benchmark archive |
| `OUT_DIR` | `~/runs/eval/dl3dv-<checkpoint>-<job id>` | where the results go |
| `SAVE_EXAMPLES` | `8` | number of scenes for which renders are saved |

## Protocol

Per scene the model gets 10 views spanning 96 frames:

```
selected: [0, 10, 21, 32, 42, 53, 64, 74, 85, 96]
input:    [0, 21, 42, 64, 96]
target:   [10, 32, 53, 74, 85]
```

Images are resized and centre-cropped to 256 x 256. The poses of the dataset
are used only to score the predicted poses, never as input.

- **PSNR**: rendered views against the real images.
- **RPA@5, @15, @30**: relative pose accuracy, the share of view pairs whose
  predicted relative rotation and translation direction are within 5, 15 and
  30 degrees of the ground truth.

## Results

Measured on 17 November 2026 on all 140 scenes (jobs 30132 and 30133),
against the values the E-RayZer authors report for the released code
([`docs/eval.md`](https://github.com/QitaoZhao/E-RayZer/blob/4309c76/docs/eval.md) of the original repository):

| Checkpoint | | PSNR | RPA@5 | RPA@15 | RPA@30 |
| --- | --- | ---: | ---: | ---: | ---: |
| `erayzer_dl3dv.pt` | ours | 20.3168 | 0.7206 | 0.8848 | 0.9352 |
| | reported | 20.3273 | 0.7210 | 0.8844 | 0.9352 |
| `erayzer_multi.pt` | ours | 19.7205 | 0.6005 | 0.8287 | 0.9017 |
| | reported | 19.7037 | 0.5986 | 0.8286 | 0.9017 |

The numbers match within 0.02 dB PSNR and 0.2 percentage points RPA. The
remaining difference is expected from different GPUs and library builds.
`erayzer_dl3dv.pt` is the checkpoint behind the DL3DV row of the paper.

## Output

```
~/runs/eval/dl3dv-<checkpoint>-<job id>/
  metrics.txt     the averages
  summary.csv     one row per scene: basename, mse, psnr, RPA@5, RPA@15, RPA@30
  sample_*/       predicted and target renders of the first scenes
```

`basename` is the line number of the scene in `manifests/test_full.txt`. The
spread between scenes is large (PSNR from about 14 to 30), so compare models
on the full split, not on a few scenes.

## Other datasets

The paper also reports ScanNet++ and BlendedMVS. The released code evaluates
DL3DV only; the scene lists for the other two are in `data_splits/`.
