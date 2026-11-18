# CAMP cluster

Checked on the cluster in November 2026. Support is
`cluster-support@camp.cit.tum.de`.

## Access

The cluster is reachable only from the CAMP VPN (profile `student-campvpn`,
same credentials as the cluster login). Log in with `ssh <user>@131.159.11.60`.

That machine is `head`. Use it only to submit jobs and for light file
operations. Computing, building images, large copies and IDEs on `head` get
your account blocked. Everything else runs in a job.

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

The model needs an RTX 30xx or newer. Request it with `--gres=gpu:1,ccc:86`
on `24g`. The value of `ccc` is the CUDA compute capability times ten.
`--exclude` is not allowed.

## Rules for job scripts

* Do not set `--cpus-per-task` or `--mem`. The cluster rejects such jobs.
* The default QOS allows one job with one GPU. Add
  `--qos=students_opportunistic` for up to two more, which can be preempted.
* `24g` cancels jobs that leave the GPU idle for 2 hours, that average below
  10 percent GPU use over 6 hours or that use less than 12 GB of GPU memory.
* Compute nodes have internet access.

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
