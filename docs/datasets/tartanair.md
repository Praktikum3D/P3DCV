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

## Cameras

TartanAir records every environment with 12 pinhole cameras of 90 degrees,
two stereo sets of six pointing front, back, left, right, top and bottom. All
other cameras are resampled from these six raw images, so every camera of a
trajectory shows the same footage.

| Camera | Size | Model | fx = fy | cx = cy | Field of view |
| --- | --- | --- | --- | --- | --- |
| `lcam_front` | 640 x 640 | pinhole | 320 | 319.5 | 90 degrees |
| `lcam_fish` | 1000 x 1000 | equidistant, r = f θ | 293.8 | 499.5 | 195 degrees |

* For the fisheye fx is 1000 divided by 195 degrees in radians, so a pixel at
  distance r from the centre sees a ray at angle θ = r / 293.8 from the
  optical axis. The image circle has a radius of 500 pixels and touches the
  image border.
* This is the same model as `camera_model="fisheye"` in gsplat.
* `lcam_fish` looks in the same direction as `lcam_front`, so
  `pose_lcam_front.txt` is also the pose of the fisheye camera.
* Pixel centres follow OpenCV, so the image centre is at 499.5 and not 500.
* The data loader in `training/tartanair_dataset.py` still sets pinhole values
  of 90 degrees for every camera. The model does not read them yet.

### Assumptions

The TartanAir documentation names the fisheye model but gives no numbers. The
values above rest on these sources.

* [rerun dataforge](https://github.com/rerun-io/examples-monorepo/blob/main/packages/dataforge/dataforge/datasets/tartanair.py)
  states 1000 x 1000 equidistant over 195 degrees with cx = cy = 499.5 and the
  same orientation as `lcam_front`. Its authors checked this by matching
  patches between `lcam_front`, `lcam_fish` and `lcam_equirect`.
* The example `linear_fisheye_camera_model.py` in
  [castacks/mvs_utils](https://github.com/castacks/mvs_utils), the camera code
  of the TartanAir authors, builds `LinearSphere(fov_degree=195)` at
  1000 x 1000.
* On 20 November 2026 we measured the image circle of
  `ArchVizTinyHouseNight/Data_hard`. Non black pixels reach exactly 500 pixels
  from the centre, which matches the values above.
* Resampling `lcam_fish` to pinhole cameras with `scripts/tartanair_remap.py`
  gives straight edges at 60, 90 and 120 degrees. With a wrong field of view
  straight lines would come out bent.
* We have not yet compared a fisheye image with the matching `lcam_front`
  image ourselves. That needs an `image_lcam_front.zip` of the same
  environment.

## Same footage with other cameras

There are two ways to train on several cameras with the same footage.

* In the data loader. Any camera that looks forward and covers at most 195
  degrees can be resampled exactly from `lcam_fish`, for example pinhole with
  60 or 120 degrees or fisheye with 150 degrees. This needs no extra download.
  `scripts/tartanair_remap.py` does this, see below.
* With `ta.customize` from tartanairpy, which resamples the six raw cameras
  into any camera model, direction and field of view. It needs
  `image_lcam_front.zip`, `image_lcam_back.zip`, `image_lcam_left.zip`,
  `image_lcam_right.zip`, `image_lcam_top.zip` and `image_lcam_bottom.zip`
  unpacked, so run it on `/tmp` inside a job and zip the result again.

### Resampling the fisheye

`scripts/tartanair_remap.py` computes for every pixel of the new camera its
ray, looks up that ray in `lcam_fish` and returns the image, a mask of valid
pixels and the intrinsics. `remap_fisheye` in the same file is meant to be
called from the data loader.

```bash
sbatch --partition=12g --gres=gpu:1,ccc:75 --time=00:10:00 \
    --output=$HOME/runs/%x-%j.out --job-name=tartanair-remap --wrap \
    "source ~/code/P3DCV/slurm/common.sh && in_container python ~/code/P3DCV/scripts/tartanair_remap.py \
    --archive \$DATA_DIR/tartanair/ArchVizTinyHouseNight/Data_hard/image_lcam_fish.zip \
    --trajectory P003 --frame 40 --out ~/runs/tartanair-remap"
cat ~/runs/tartanair-remap-*.out
```

Output for 256 x 256 pixels.

| Camera | fx = fy | cx = cy | Valid pixels |
| --- | --- | --- | --- |
| pinhole 60 degrees | 221.7 | 127.5 | 100 % |
| pinhole 90 degrees | 128.0 | 127.5 | 100 % |
| pinhole 120 degrees | 73.9 | 127.5 | 100 % |
| fisheye 150 degrees | 97.8 | 127.5 | 78 % |
| fisheye 195 degrees | 75.2 | 127.5 | 78 % |

* The script first checks that resampling to the camera of the input gives the
  input back unchanged.
* It renders at four times the size and shrinks the result, which avoids
  aliasing.
* In the centre `lcam_fish` has 293.8 pixels per radian. At 256 x 256 a
  pinhole camera needs at most that down to about 50 degrees. Narrower
  cameras come out slightly blurred.

## Sizes

All 148 archives of `image_lcam_fish.zip` together have 1719 GB, the median
archive 8.7 GB and the largest 60 GB. The team folder holds 250 GB, so we
always work with a selection of environments. Sizes per archive are in the
[file list](https://github.com/castacks/tartanairpy/blob/main/tartanair/download_files.txt).

* The file list has 74 environments with fisheye images, each in `Data_easy`
  and `Data_hard`.
* `image_lcam_front.zip` is about half the size of `image_lcam_fish.zip` of
  the same environment and 833 GB in total.
* All six raw cameras of the left set together have 4796 GB.

There is no option to download a given number of gigabytes. A subset is a
list of environments, difficulties and cameras, and its size is the sum of
those archives. The 15 smallest environments in `Data_hard`, sizes in GB.

| Environment | `lcam_fish` | `lcam_front` | six raw cameras |
| --- | --- | --- | --- |
| ArchVizTinyHouseNight | 0.5 | 0.2 | 1.3 |
| ArchVizTinyHouseDay | 0.6 | 0.3 | 1.9 |
| RetroOffice | 0.8 | 0.4 | 2.3 |
| OldTownNight | 1.5 | 0.6 | 3.7 |
| CountryHouse | 1.6 | 0.7 | 4.4 |
| AmericanDiner | 1.5 | 0.8 | 4.6 |
| OldTownFall | 1.7 | 0.8 | 4.8 |
| HongKong | 2.0 | 0.8 | 4.7 |
| DesertGasStation | 2.1 | 1.0 | 5.6 |
| House | 2.5 | 1.2 | 7.3 |
| AbandonedFactory2 | 2.8 | 1.1 | 7.3 |
| OldTownWinter | 2.7 | 1.2 | 7.7 |
| CoalMine | 3.0 | 1.4 | 7.6 |
| Office | 3.5 | 1.7 | 9.9 |
| PolarSciFi | 4.0 | 1.9 | 10.8 |
| Total | 31.0 | 14.2 | 83.8 |

Keep a few environments out of training and use them only for evaluation.

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
