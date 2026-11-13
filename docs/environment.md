# Environment

All jobs run inside one container image,
`/mnt/projects/p3dcv/anycam/envs/p3dcv.sqsh`. It replaces a conda environment,
which would exceed the file limit of the cluster.

## Content

Defined in [`docker/Dockerfile`](../docker/Dockerfile) and
[`docker/requirements.txt`](../docker/requirements.txt):

- Base image `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel` (Python 3.10).
- The Python packages pinned by E-RayZer (torch 2.3.0, torchvision 0.18.0,
  xformers 0.0.26.post1, ...) plus `wandb`.
- The gsplat fork used by E-RayZer
  ([QitaoZhao/gsplat](https://github.com/QitaoZhao/gsplat), branch
  `v1.4.0-working`, pinned to a commit) with its CUDA extension compiled.

The image contains no project code, no data and no weights.

### Supported GPUs

gsplat is compiled for RTX 20xx and newer (CUDA architectures 7.5, 8.6, 8.9,
9.0). It cannot be compiled for older cards, so the image does not render on
the TITAN X and TITAN Xp nodes `dortmund`, `jena` and `ulm`.

## Using it

```bash
source ~/code/P3DCV/slurm/common.sh
in_container python "$REPO_DIR/scripts/check_env.py"
```

`in_container` runs a command in the image with `/mnt` and `/tmp` mounted:

```bash
enroot start --root --rw --mount /mnt:mnt --mount /tmp:tmp "$IMAGE" <command>
```

`--mount /mnt:mnt` makes the whole cluster storage visible in the container
under the same paths, which is how the code in your clone gets in.

## Smoke test

```bash
sbatch ~/code/P3DCV/slurm/check_env.sh
cat ~/runs/check-env-*.out
```

It prints the library versions and the GPU, renders a few Gaussians with
gsplat and checks that the pretrained weights are present.

## Changing the environment

1. Edit `docker/requirements.txt` or `docker/Dockerfile` on a branch.
2. Tell the team: the image is shared, and a rebuild replaces it for everyone.
3. Rebuild and verify:

```bash
sbatch ~/code/P3DCV/slurm/build_image.sh     # about 30 minutes, the image is 18 GB
sbatch ~/code/P3DCV/slurm/check_env.sh
```

The build runs `podman build` and converts the result with `enroot import`
into a single `.sqsh` file. To try an image without replacing the shared one,
set `IMAGE`:

```bash
sbatch --export=ALL,IMAGE=/mnt/home/$USER/p3dcv-test.sqsh ~/code/P3DCV/slurm/build_image.sh
```
