# TartanAir V2

Synthetic dataset from 65 simulated environments with ground truth poses. We
use its fisheye images. See <https://tartanair.org> and
[castacks/tartanairpy](https://github.com/castacks/tartanairpy).

## What is on the cluster

`/mnt/projects/p3dcv/anycam/data/tartanair/ArchVizTinyHouseNight/Data_hard/image_lcam_fish.zip`
with 0.5 GB. It is the smallest fisheye archive, meant for developing the
data loader and not for training.

## Format

There is one zip archive per environment, difficulty, modality and camera.
Single trajectories cannot be downloaded separately.

```
<Environment>/Data_hard/
  P000/
    image_lcam_fish/000000_lcam_fish_image.png
    pose_lcam_front.txt
  P001/
```

* `P000`, `P001` and so on are trajectories. The sample has 7 with 65 to 88
  frames each.
* Images are 1000 x 1000 RGB PNGs with a circular image area, about 1 MB each.
* `pose_lcam_front.txt` has one line per frame with `tx ty tz qx qy qz qw` in
  the NED frame.
* The fisheye images use the Linear Spherical camera model of TartanAir.

Still open are the exact field of view and the intrinsics of these images.

## Sizes

All 148 archives of `image_lcam_fish.zip` together have 1719 GB, the median
archive 8.7 GB and the largest 60 GB. The team folder holds 250 GB, so we
always work with a selection of environments. Sizes per archive are in the
[file list](https://github.com/castacks/tartanairpy/blob/main/tartanair/download_files.txt).

## Download

```bash
sbatch ~/code/P3DCV/slurm/download_tartanair.sh                        # the sample
sbatch ~/code/P3DCV/slurm/download_tartanair.sh RetroOffice/Data_hard  # another environment
```

`MODALITY` and `CAMERA` select another archive. The job verifies checksums,
skips complete archives and resumes partial ones. Keep the archives zipped
and read from the zip or unpack to `/tmp` inside a job.

## Sample run

```bash
sbatch ~/code/P3DCV/slurm/tartanair_sample.sh
cat ~/runs/tartanair-sample-*.out
```

It lists the trajectories and writes a contact sheet to
`~/runs/tartanair_sample.png`.
[`scripts/tartanair_sample.py`](../../scripts/tartanair_sample.py) is the
shortest example of reading images and poses from an archive.
