# CAMP cluster

Everything here was checked on the cluster in November 2026. Support:
cluster-support@camp.cit.tum.de.

## Access

The cluster is reachable only from the CAMP VPN (profile `student-campvpn`,
same credentials as the cluster login). Then:

```bash
ssh <user>@131.159.11.60     # the head node
```

## The head node is not for work

`head` is for submitting jobs and light file operations (`sbatch`, `squeue`,
`ls`, `cat`, `git pull`). Computing, building images, large copies and IDEs
on `head` get your account blocked. `podman` and `enroot` are not even
installed there. Everything else runs in a job on a compute node.

## Storage

`/mnt` is a network file system that is identical on `head` and on every
compute node. A file written by a job is immediately visible on `head`.

| Path | Content |
| --- | --- |
| `/mnt/home/<user>` | your home: code and job logs |
| `/mnt/projects/p3dcv/anycam` | team folder, writable by all four of us |
| `/mnt/datasets` | public datasets, read-only, packed |
| `/mnt/general/examples` | example job scripts from the admins |
| `/tmp` | local disk of the node, deleted when the job ends |

Home and the team folder are each limited to 250 GB **and 100 000 files**, and
neither is backed up. The file limit is the one that hurts: keep datasets
zipped and never create a conda or virtual environment on `/mnt`.

### Team folder

```
/mnt/projects/p3dcv/anycam/
  data/tartanair/         TartanAir archives, always kept zipped
  checkpoints/pretrained/ erayzer_multi.pt, erayzer_dl3dv.pt,
                          imagenet-vgg-verydeep-19.mat
  checkpoints/runs/       one sub-folder per training run
  envs/p3dcv.sqsh         container image
```

New sub-folders inherit write access for the whole team.

### Your home

```
/mnt/home/<user>/
  code/P3DCV/             your clone
  runs/                   job logs (<job-name>-<job-id>.out / .err)
```

Everyone works in their own clone and shares through GitHub. A shared clone in
the team folder does not work, because a clone has exactly one checked-out
branch.

## Partitions

| Partition | GPUs | Time limit | Use for |
| --- | --- | --- | --- |
| `12g` | TITAN X, TITAN Xp, RTX 2080 Ti, RTX 5000 (11 to 16 GB) | 3 days | tests, interactive work |
| `24g` | RTX 3090, RTX 4090, RTX A5000, TITAN RTX | 24 hours | training |
| `h200` | H200, 140 GB | 24 hours | opportunistic only |
| `data` | nodes `jena`, `ulm` | 5 days | downloads |

`data` is a partition, not a folder: `sbatch --partition=data script.sh`.

gsplat does not run on the TITAN X and TITAN Xp nodes (`dortmund`, `jena`,
`ulm`). Jobs that render must exclude them:
`#SBATCH --exclude=dortmund,jena,ulm`.

## Rules for job scripts

- Do not set `--cpus-per-task` or `--mem`. Both are assigned per GPU and the
  cluster rejects jobs that request them.
- `students_normal` (the default) allows one job with one GPU. Add
  `--qos=students_opportunistic` for up to two more jobs; these can be
  preempted and requeued.
- `24g` cancels jobs that leave the GPU idle for 2 hours, average below 10 %
  GPU use over 6 hours, or use less than 12 GB of GPU memory. Run small tests
  on `12g`.
- Compute nodes have internet access (Hugging Face, Docker Hub, GitHub).

## Everyday commands

```bash
sbatch slurm/check_env.sh        # submit a job
squeue -u $USER                  # my queued and running jobs
scancel <job-id>                 # cancel a job
sacct -X -j <job-id>             # state and exit code of a finished job
tail -f ~/runs/<name>-<id>.out   # follow a log
```

## Working interactively

The admins provide example jobs in `/mnt/general/examples`
(see the `README.md` there):

| Script | Gives you |
| --- | --- |
| `ssh.sh` | an idle job with an SSH server; the connect command is in `ssh.out` |
| `jupyter.sh` | Jupyter Lab on a node; the URL is in `jupyter.out` |
| `remote.sh` | a remote desktop in the browser (noVNC) |

```bash
sbatch --partition=12g /mnt/general/examples/ssh.sh
cat ssh.out
```

To use VS Code, connect "Remote - SSH" to the compute node of such a job,
never to `head`.

### Git on the cluster

Either develop locally, push from your laptop and only `git pull` on the
cluster, or develop on a compute node and push from there. For the second
option use SSH agent forwarding so that no key is stored on the cluster:

```
Host campcluster
    HostName 131.159.11.60
    User <user>
    ForwardAgent yes
```
