## Quick start
You only have to do the setup once (create folders, etc...)
```bash
ssh <user>@131.159.11.60
mkdir -p ~/code ~/runs && cd ~/code
git clone https://github.com/Praktikum3D/P3DCV.git

sbatch ~/code/P3DCV/slurm/check_env.sh         # is the environment usable?
sbatch ~/code/P3DCV/slurm/tartanair_sample.sh  # can I read the data?
sbatch ~/code/P3DCV/slurm/infer.sh             # does the pretrained model run?
sbatch ~/code/P3DCV/slurm/evaluate_dl3dv.sh    # do I get the numbers of the paper?

squeue -u $USER
cat ~/runs/*.out
```

Jobs run one after the other. Add `--qos=students_opportunistic` to run up to
two more in parallel. All logs go to `~/runs`.

## Documentation

| Document | Content |
| --- | --- |
| [docs/cluster.md](docs/cluster.md) | Access, storage, partitions and rules |
| [docs/environment.md](docs/environment.md) | The container image |
| [docs/datasets/tartanair.md](docs/datasets/tartanair.md) | TartanAir V2 and the sample we store |
| [docs/datasets/dl3dv.md](docs/datasets/dl3dv.md) | The DL3DV benchmark subset |
| [docs/inference.md](docs/inference.md) | Running the pretrained model |
| [docs/evaluation.md](docs/evaluation.md) | Reproducing the numbers of the paper |
| [docs/training.md](docs/training.md) | Pretrained weights and what training needs |

## Layout

| Path | Content |
| --- | --- |
| `slurm/` | One job script per task, shared paths in `common.sh` |
| `scripts/` | Python entry points and the E-RayZer launch scripts |
| `docker/` | Definition of the container image |
| `erayzer_core/` | Model, losses, camera utilities and renderer |
| `training/` | Data loading, so far only for DL3DV |
| `app_core/` | Inference wrapper used by `scripts/infer.py` and the demo |
| `config/` | Model and run configurations |
| `train.py`, `evaluate.py`, `gradio_app.py` | Entry points |

