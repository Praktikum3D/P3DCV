# Environment

## Content

Defined in [`docker/Dockerfile`](../docker/Dockerfile) and
[`docker/requirements.txt`](../docker/requirements.txt).

* Python 3.10 with torch 2.3.0, CUDA 12.1 and xformers 0.0.26.post1
* The gsplat fork of E-RayZer with its CUDA extension compiled
* The other Python packages of E-RayZer plus `wandb` and Jupyter Lab

## Using it


```bash
sbatch ~/code/P3DCV/slurm/check_env.sh
cat ~/runs/check-env-*.out
```

The check prints the library versions and the GPU, renders a few Gaussians and
looks for the pretrained weights. It ends with `environment OK`.

## Jupyter

```bash
sbatch ~/code/P3DCV/slurm/jupyter.sh
cat ~/runs/jupyter-*.out      # wait for the URL
scancel <job-id>              # when you are done
```

Open the URL in a browser while connected to the VPN. Notebooks run inside
the container on an RTX 30xx or newer and see `/mnt` as in every job.


## VS Code on a GPU node

`slurm/dev.sh` starts an SSH server on the node and Jupyter Lab inside the
container in one job. Your public key has to be in `~/.ssh/authorized_keys`
on the cluster.

```bash
sbatch ~/code/P3DCV/slurm/dev.sh
cat ~/runs/dev-current.env    # node IP, SSH port, Jupyter port and token
scancel <job-id>              # when you are done
```

* Connect with VS Code Remote SSH to `ssh -p <SSH_PORT> <user>@<IP>` and open `/mnt/home/<user>/code/P3DCV`
* The terminal runs on the node, not in the container. `bash slurm/shell.sh` opens a shell inside the container
* For notebooks pick the existing Jupyter server `http://127.0.0.1:<JUPYTER_PORT>/?token=<TOKEN>` as kernel, so they run in the container
* The VS Code server lives in `~/.vscode-server`, about 10 000 to 20 000 files of the home limit
