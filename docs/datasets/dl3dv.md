# DL3DV benchmark

[DL3DV](https://dl3dv-10k.github.io/DL3DV-10K/) is the dataset E-RayZer was
trained and evaluated on. We keep only what is needed to reproduce the DL3DV
numbers of the paper, see [evaluation.md](../evaluation.md).

## What is stored

The paper evaluates on 140 scenes of the Hugging Face dataset
`DL3DV/DL3DV-Benchmark`. The complete split has 48 991 images and 43.8 GB. The protocol reads only 10
images per scene, so only those are stored.

| | |
| --- | --- |
| File | `/mnt/projects/p3dcv/anycam/data/dl3dv/benchmark_fd96.zip` |
| Content | 140 scenes, 1400 undistorted images with 960 x 540 pixels, poses and intrinsics |
| Size | 1.1 GB |

```
manifests/test_full.txt                     one scene.json path per line
scenes/<scene id>/scene.json                frames with file_path, w2c, fx, fy, cx, cy
scenes/<scene id>/images_undistort/*.png    the 10 images of the protocol
```

`scene.json` lists the first 97 frames with their poses, so frame indices are
the same as in the full dataset. Only the frames named in `frames_with_image`
have an image. Another `FRAME_DIST` needs its own archive.

The images come from the Hugging Face dataset `DL3DV/DL3DV-Benchmark` and were
undistorted with the camera parameters of the dataset.
