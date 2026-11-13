# Inference with pretrained E-RayZer

Two jobs run the pretrained model without any training. Both need an RTX 30xx
or newer, see "GPU requirements" below.

## Headless run

```bash
# one trajectory of the TartanAir sample (fisheye input)
sbatch ~/code/P3DCV/slurm/infer.sh

# your own images: a folder with about 10 views of one scene
sbatch ~/code/P3DCV/slurm/infer.sh --images /mnt/home/$USER/my_scene

# another trajectory, every 3rd frame instead of spread over the whole clip
sbatch ~/code/P3DCV/slurm/infer.sh \
    --archive /mnt/projects/p3dcv/anycam/data/tartanair/ArchVizTinyHouseNight/Data_hard/image_lcam_fish.zip \
    --trajectory P005 --stride 3

# the DL3DV checkpoint instead of the multi-dataset one
sbatch --export=ALL,CKPT=/mnt/projects/p3dcv/anycam/checkpoints/pretrained/erayzer_dl3dv.pt \
    ~/code/P3DCV/slurm/infer.sh
```

The job takes about three minutes. Results are written to
`~/runs/inference/<timestamp>/`:

| File | Content |
| --- | --- |
| `pred_view_00.png` ... | rendered views |
| `point_cloud.glb` | the predicted Gaussians as a point cloud with the cameras |
| `render_video.mp4` | a camera path through the reconstruction |

A zip of the folder is written next to it. Copy it to your machine with `scp`
to look at the results.

[`scripts/infer.py`](../scripts/infer.py) is a thin command line around
`app_core.engine.ERayZerEngine`, the same class the demo uses.

### What the model does with the input

- It takes 10 views: 5 as input and 5 as targets to render. With fewer views
  the model pads, with more it prints a warning.
- Every image is cropped to its central square and resized to 256 x 256.
- Camera poses and intrinsics are not read from the data. The model estimates
  them, and it can only express a pinhole camera (`fx, fy, cx, cy`). Fisheye
  input therefore runs, but the model has no way to describe its camera
  correctly. This is the gap the project is about.

Loading the released checkpoints prints `Unexpected keys: 32`. These are
weights in the file that the released model code does not use.

## Interactive demo

```bash
sbatch ~/code/P3DCV/slurm/demo.sh
cat ~/runs/demo-*.out      # wait for the URL to appear
scancel <job-id>           # when you are done
```

The log shows a URL on the compute node. Open it in a browser while you are
connected to the VPN. There you upload images of one scene and get the
rendered views, the point cloud and the video. The demo accepts images only;
extract frames from a video first, for example with
`ffmpeg -i video.mp4 -vf fps=2 frame_%03d.png`.

The job stops after two hours. The demo is deliberately started without
Gradio's `--share` option, which would publish the service on the internet
from inside the cluster.

## GPU requirements

E-RayZer runs its transformer in bf16 through xformers, which supports that
only on GPUs with CUDA compute capability 8.0 or higher. On this cluster
these are the RTX 3090, RTX A5000 and RTX 4090 in the `24g` partition and the
H200 nodes. The job scripts request them with:

```bash
#SBATCH --partition=24g
#SBATCH --gres=gpu:1,ccc:86
```

On an RTX 2080 Ti the model fails with
`bf16 is only supported on A100+ GPUs`.
