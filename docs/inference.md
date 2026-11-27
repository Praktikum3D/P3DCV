# Inference

Two jobs run the pretrained model without any training. Both need an RTX 30xx
or newer.

## Headless run

```bash
# one trajectory of the TartanAir sample
sbatch ~/code/P3DCV/slurm/infer.sh

# your own images, a folder with about 10 views of one scene
sbatch ~/code/P3DCV/slurm/infer.sh --images /mnt/home/$USER/my_scene

# another trajectory with every third frame
sbatch ~/code/P3DCV/slurm/infer.sh \
    --archive /mnt/projects/p3dcv/anycam/data/tartanair/ArchVizTinyHouseNight/Data_hard/image_lcam_fish.zip \
    --trajectory P005 --stride 3
```

`CKPT` selects another checkpoint. The job takes about three minutes and
writes to `~/runs/inference/<timestamp>/`.

| File | Content |
| --- | --- |
| `pred_view_00.png` and following | Rendered views |
| `point_cloud.glb` | Predicted Gaussians as a point cloud with the cameras |
| `render_video.mp4` | A camera path through the reconstruction |

## What the model does with the input

* It takes 10 views, 5 as input and 5 as targets to render.
* Every image is cropped to its central square and resized to 256 x 256.
* Poses and intrinsics are not read from the data. The model estimates them
  and can only express a pinhole camera. Fisheye input runs, but the model
  cannot describe its camera correctly. This is the gap the project is about.

Loading a checkpoint prints `Unexpected keys: 32`. These are the convolutions
of the VGG 19 that the original training carried inside the model for its
perceptual loss, under `loss_computer.perceptual_loss_module`. No model in this
repository builds it, nothing is missing without it, and the viewer drops those
names before loading so that a real mismatch stays visible.

## Interactive demo

```bash
sbatch ~/code/P3DCV/slurm/demo.sh
cat ~/runs/demo-*.out      # wait for the URL
scancel <job-id>           # when you are done
```

Open the URL in a browser while connected to the VPN and upload images of one
scene. The job stops after two hours. Never start the demo with the `--share`
option of Gradio, because that would publish a service from inside the
cluster on the internet.

## GPU requirements

The model runs in bf16 through xformers, which needs CUDA compute capability
8.0 or higher. On this cluster these are the RTX 3090, RTX A5000 and RTX 4090
in `24g` and the H200 nodes. The job scripts request them with
`--gres=gpu:1,ccc:86`. On an RTX 2080 Ti the model fails with
`bf16 is only supported on A100+ GPUs`.
