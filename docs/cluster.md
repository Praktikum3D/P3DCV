# CAMP cluster


## Storage

`/mnt` is the same on `head` and on every compute node.

| Path | Content |
| --- | --- |
| `/mnt/home/<user>` | Your home with `code/P3DCV` and the job logs in `runs` |
| `/mnt/projects/p3dcv/anycam` | Team folder, writable by all four of us |
| `/mnt/datasets` | Public datasets, read only |
| `/tmp` | Local disk of the node, deleted when the job ends |

Home and team folder are each limited to 250 GB and 100 000 files, without
backup. The file limit is the one that hurts, so keep datasets zipped.

```
/mnt/projects/p3dcv/anycam/
  data/tartanair/          TartanAir archives
  data/dl3dv/              DL3DV benchmark subset
  checkpoints/pretrained/  E-RayZer weights and VGG19
  checkpoints/runs/        one folder per training run
  envs/p3dcv.sqsh          container image
```

## Partitions

| Partition | GPUs | Time limit | Use for |
| --- | --- | --- | --- |
| `12g` | TITAN X, TITAN Xp, RTX 2080 Ti, RTX 5000 | 3 days | Small tests |
| `24g` | RTX 3090, RTX 4090, RTX A5000, TITAN RTX | 24 hours | Inference, evaluation, training |
| `h200` | H200 | 24 hours | Opportunistic only |
| `data` | None | 5 days | Downloads |



## Everyday commands

```bash
sbatch slurm/check_env.sh        # submit a job
squeue -u $USER                  # my queued and running jobs
scancel <job-id>                 # cancel a job
sacct -X -j <job-id>             # state and exit code of a finished job
tail -f ~/runs/<name>-<id>.out   # follow a log
```

For interactive work the admins provide example jobs in
`/mnt/general/examples`, for example `ssh.sh` for a shell or VS Code on a
compute node and `jupyter.sh` for Jupyter Lab.
