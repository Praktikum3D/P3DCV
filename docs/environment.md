# Environment

All jobs run inside one container image,
`/mnt/projects/p3dcv/anycam/envs/p3dcv.sqsh`. It is the only supported way to
run the code. There is no conda or virtual environment and no git submodule.

## Content

Defined in [`docker/Dockerfile`](../docker/Dockerfile) and
[`docker/requirements.txt`](../docker/requirements.txt).

* Python 3.10 with torch 2.3.0, CUDA 12.1 and xformers 0.0.26.post1
* The gsplat fork of E-RayZer with its CUDA extension compiled
* The other Python packages of E-RayZer plus `wandb`

The image contains no project code, no data and no weights. Jobs mount `/mnt`
into the container, so your clone and the team folder appear under the same
paths as outside. Changing code never requires a rebuild.

## Using it

Job scripts source `slurm/common.sh` and call `in_container <command>`.

```bash
sbatch ~/code/P3DCV/slurm/check_env.sh
cat ~/runs/check-env-*.out
```

The check prints the library versions and the GPU, renders a few Gaussians and
looks for the pretrained weights. It ends with `environment OK`.

## Changing it

1. Edit `docker/requirements.txt` or `docker/Dockerfile` on a branch.
2. Tell the team, because a rebuild replaces the image for everyone.
3. Run `sbatch ~/code/P3DCV/slurm/build_image.sh` and then the check again.

The build takes about 30 minutes and the image is 18 GB. Set `IMAGE` to
another path to try an image without replacing the shared one.
