# DL3DV benchmark

[DL3DV](https://dl3dv-10k.github.io/DL3DV-10K/) is the dataset E-RayZer was
trained and evaluated on. We only keep the part needed to reproduce the
DL3DV numbers of the paper, see [evaluation.md](../evaluation.md). Training
on DL3DV is not planned.

## What is stored

The paper evaluates on 140 scenes
([`data_splits/dl3dv/test_full.txt`](../../data_splits/dl3dv/test_full.txt))
from the Hugging Face dataset `DL3DV/DL3DV-Benchmark`. The complete split has
48 991 images and 43.8 GB. The evaluation protocol is deterministic and reads
only 10 images per scene (frames 0, 10, 21, 32, 42, 53, 64, 74, 85 and 96),
so only those are downloaded:

| | |
| --- | --- |
| File | `/mnt/projects/p3dcv/anycam/data/dl3dv/benchmark_fd96.zip` |
| Content | 140 scenes, 1400 undistorted images (960 x 540 PNG), poses and intrinsics |
| Size | 1.1 GB, one file |

Layout inside the archive, identical to the output of the upstream converter
`scripts/convert_dl3dv_opencv_list.py`:

```
manifests/test_full.txt                     one scene.json path per line
scenes/<scene id>/scene.json                frames with file_path, w2c, fx, fy, cx, cy
scenes/<scene id>/images_undistort/*.png    the 10 images of the protocol
```

`scene.json` lists the first 97 frames with their poses, so the frame indices
are the same as in the full dataset. Only the 10 frames named in
`frames_with_image` have an image. A different `FRAME_DIST` therefore needs
its own archive.

## Downloading it again

The dataset is gated. Accept the terms on
<https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark> with your own account,
create a read token and store it on the cluster:

```bash
mkdir -p ~/.cache/huggingface
nano ~/.cache/huggingface/token      # paste the token, nothing else
chmod 600 ~/.cache/huggingface/token
```

Then:

```bash
sbatch ~/code/P3DCV/slurm/download_dl3dv_benchmark.sh            # all 140 scenes, about 10 minutes
LIMIT=4 sbatch ~/code/P3DCV/slurm/download_dl3dv_benchmark.sh    # only the first 4 scenes
FRAME_DIST=10 sbatch ~/code/P3DCV/slurm/download_dl3dv_benchmark.sh
```

The job runs [`scripts/prepare_dl3dv_benchmark.py`](../../scripts/prepare_dl3dv_benchmark.py)
in the container. It downloads and undistorts on the local disk of the node
and writes a single zip file to the team folder, because the team folder is
limited to 100 000 files.
