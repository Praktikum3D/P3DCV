# P3DCV

Self-supervised 3D reconstruction that accepts arbitrary cameras as input and
output. The project fine-tunes [E-RayZer](https://github.com/QitaoZhao/E-RayZer)
on [TartanAir V2](https://tartanair.org) so that it also handles non-pinhole
cameras such as fisheye.

Practical course "3D Computer Vision", CAMP, TUM, winter term 2026/27.

## Documentation

| Document | Content |
| --- | --- |
| [docs/cluster.md](docs/cluster.md) | The CAMP cluster: access, storage, partitions, rules, working interactively |
| [docs/environment.md](docs/environment.md) | The container image: what it contains, how to use and rebuild it |
| [docs/datasets/tartanair.md](docs/datasets/tartanair.md) | TartanAir V2: format, sizes, download, sample run |
| [docs/training.md](docs/training.md) | Pretrained weights and how a training run is set up |

## Status

| Part | State |
| --- | --- |
| Folder structure on the cluster | done |
| Container image (Python 3.10, torch 2.3.0, xformers, gsplat) | done |
| Pretrained E-RayZer weights and VGG19 | done |
| TartanAir sample (one small fisheye environment) | done |
| E-RayZer model and training code | not imported yet |
| TartanAir data loader | not started |
| Fine-tuning scripts and configs | not started |

## How it is organised

Three things are kept apart, each where it fits its size and how often it
changes:

| What | Where | Why |
| --- | --- | --- |
| Code | this repository, one clone per person in `/mnt/home/<user>/code/P3DCV` | small, changes constantly, shared through GitHub |
| Environment | one container image in the team folder | large, changes rarely, a single file |
| Data and weights | team folder `/mnt/projects/p3dcv/anycam` | large, identical for everyone, not allowed in git |

Jobs start the image with `/mnt` mounted, so your clone and the team folder are
visible inside the container under the same paths as outside. Changing code
never requires rebuilding the image.

```
docker/                     image definition
slurm/                      job scripts, one per task
  common.sh                 shared paths and the in_container helper
  build_image.sh            build the container image
  check_env.sh              smoke test of the image on a GPU node
  download_pretrained.sh    E-RayZer weights and VGG19
  download_tartanair.sh     TartanAir archives
  tartanair_sample.sh       sample run on the TartanAir sample
scripts/                    Python entry points called by the job scripts
docs/                       documentation
```

## Quick start

You need a cluster account and the CAMP VPN (see [docs/cluster.md](docs/cluster.md)).
The image, the weights and the TartanAir sample already exist in the team
folder, so a new team member only needs a clone:

```bash
ssh <user>@131.159.11.60

mkdir -p ~/code ~/runs
cd ~/code
git clone https://github.com/Praktikum3D/P3DCV.git

sbatch ~/code/P3DCV/slurm/check_env.sh        # is the environment usable?
sbatch ~/code/P3DCV/slurm/tartanair_sample.sh # can I read the data?

squeue -u $USER                               # wait until both have finished
cat ~/runs/check-env-*.out ~/runs/tartanair-sample-*.out
```

`check-env` ends with `environment OK`. `tartanair-sample` lists the
trajectories of the sample and writes `~/runs/tartanair_sample.png`.

All job logs go to `~/runs`. The scripts expect the clone at
`~/code/P3DCV`; set `REPO_DIR` if yours is elsewhere.

## Conventions

- Code, comments and documentation are written in English.
- Work on a branch and open a pull request into `main`.
- Never commit data, weights or images. They belong in the team folder.
- Use the variables from `slurm/common.sh` (`PROJECT_DIR`, `DATA_DIR`,
  `PRETRAINED_DIR`, `RUNS_DIR`, `IMAGE`) instead of hard-coded paths.
- Tell the team before you change anything in the team folder. It has one
  quota for all of us and no backup.
