# TartanAir V2

Synthetic dataset rendered in 65 simulated environments, with ground-truth
poses. We use its fisheye images. Documentation: <https://tartanair.org>,
tools: [castacks/tartanairpy](https://github.com/castacks/tartanairpy).

## What is on the cluster

```
/mnt/projects/p3dcv/anycam/data/tartanair/
  ArchVizTinyHouseNight/Data_hard/image_lcam_fish.zip    0.5 GB, sample
```

This is the smallest fisheye archive of the dataset. It is there to develop
and test the data loader; it is not a training set.

## Format

The dataset is distributed as one zip archive per environment, difficulty
(`Data_easy`, `Data_hard`), modality and camera. Single trajectories cannot be
downloaded separately. Inside `image_lcam_fish.zip`:

```
<Environment>/Data_hard/
  P000/
    image_lcam_fish/000000_lcam_fish_image.png
    image_lcam_fish/000001_lcam_fish_image.png
    ...
    pose_lcam_front.txt
  P001/
  ...
```

- `P000`, `P001`, ... are trajectories. The sample has 7, with 65 to 88
  frames each (500 images in total).
- Images are 1000 x 1000 RGB PNGs with a circular image area, about 1 MB
  each. The archive stores them without compression.
- `pose_lcam_front.txt` has one line per frame: `tx ty tz qx qy qz qw`,
  translation in metres and orientation as a quaternion, in the NED frame.
- The ready-made fisheye images use TartanAir's "Linear Spherical" camera
  model. Other camera models can be resampled from the raw pinhole images
  with `tartanair.customize`.
- Camera names: `lcam_*` and `rcam_*` are the left and right camera of a
  stereo pair (baseline 0.25 m). Besides `fish` there are `equirect` and six
  pinhole views (`front`, `back`, `left`, `right`, `top`, `bottom`,
  640 x 640, 90 degrees).

Still open: the exact field of view and intrinsics of the fisheye images.

## Sizes

Fisheye images of the left camera (`image_lcam_fish.zip`), from the file list
of the download tool:

| | Size |
| --- | --- |
| All 148 archives | 1719 GB |
| Median archive | 8.7 GB |
| `ArchVizTinyHouseNight/Data_hard` | 0.50 GB |
| `ArchVizTinyHouseDay/Data_hard` | 0.60 GB |
| `RetroOffice/Data_hard` | 0.76 GB |
| `GreatMarsh/Data_easy` (largest) | 60 GB |

The team folder holds 250 GB, so we always work with a selection of
environments. Fisheye depth is about twice as large as the images (3088 GB in
total) and is only needed for evaluation.

## Download

```bash
# the sample
sbatch ~/code/P3DCV/slurm/download_tartanair.sh

# other environments
sbatch ~/code/P3DCV/slurm/download_tartanair.sh RetroOffice/Data_hard OldTownFall/Data_easy

# another modality or camera
sbatch --export=ALL,MODALITY=depth,CAMERA=rcam_fish \
    ~/code/P3DCV/slurm/download_tartanair.sh RetroOffice/Data_hard
```

The job runs on the `data` partition, verifies every archive against the
checksum of the server, skips archives that are already complete and resumes
partial ones. Check the size in the table of the
[file list](https://github.com/castacks/tartanairpy/blob/main/tartanair/download_files.txt)
before you download, and tell the team.

## Keep the archives zipped

Unpacking even a mid-sized environment would exhaust the limit of 100 000
files. Read images directly from the zip, or unpack to `/tmp` on the node at
the start of a job (it is deleted when the job ends).

## Sample run

```bash
sbatch ~/code/P3DCV/slurm/tartanair_sample.sh
cat ~/runs/tartanair-sample-*.out
```

[`scripts/tartanair_sample.py`](../../scripts/tartanair_sample.py) opens the
archive without unpacking it, lists the trajectories with frame count and
spatial extent, and writes a contact sheet of eight frames to
`~/runs/tartanair_sample.png`. It is also the shortest example of how to read
images and poses from an archive.

To look at the data on your own machine, copy the archive
(`scp`, 0.5 GB) and open it locally.
