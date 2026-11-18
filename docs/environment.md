# Environment

## Content

Defined in [`docker/Dockerfile`](../docker/Dockerfile) and
[`docker/requirements.txt`](../docker/requirements.txt).

* Python 3.10 with torch 2.3.0, CUDA 12.1 and xformers 0.0.26.post1
* The gsplat fork of E-RayZer with its CUDA extension compiled
* The other Python packages of E-RayZer plus `wandb`

## Using it


```bash
sbatch ~/code/P3DCV/slurm/check_env.sh
cat ~/runs/check-env-*.out
```

The check prints the library versions and the GPU, renders a few Gaussians and
looks for the pretrained weights. It ends with `environment OK`.

