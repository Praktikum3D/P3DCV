# Interactive viewer

A browser viewer for the Gaussians E-RayZer reconstructs, rendered on the GPU
by our 3DGEER with rays handed in from PyTorch. The same scene can be looked at
as a pinhole, as an equidistant fisheye past 195 degrees and under the unified
camera model, which is what the project is about.

The server is a small FastAPI app, the client a single page of vanilla
JavaScript. The browser posts the current camera and gets one JPEG back, then
asks for the next frame, so there is never more than one request in flight.

## Start it on the cluster

```bash
sbatch --export=ALL,REPO_DIR=$PWD ~/code/P3DCV/slurm/viewer.sh
cat ~/runs/viewer-*.out     # wait for the URL and the tunnel line
camp tunnel <ip>:<port>     # from the laptop, inside the VPN
scancel <job-id>            # when you are done
```

The job asks for `24g` with `--gres=gpu:1,ccc:86`, because reconstructing needs
bf16, and it stops after two hours. `24g` also cancels jobs whose GPU sits idle
for two hours, so cancel it yourself rather than leaving it.

| Variable | Meaning |
| --- | --- |
| `IMAGE` | Container, by default `~/envs/p3dcv-beyond90-alex.sqsh` |
| `SCENES` | Directory of precomputed scene files, by default `~/runs/viewer/scenes` |
| `OUT_DIR` | Uploads and saved images, by default `~/runs/viewer` |
| `PORT`, `HOST` | Port and bind address, by default a random port on every interface |

## Pick a model and a scene

The sidebar works top down.

1. **Model**: the dropdown lists `erayzer_multi.pt` and every run under
   `checkpoints/runs/`. A run is labelled with the camera settings of its own
   `config.yaml`, so a fisheye or UCM run loads as what it was trained as and
   not as a pinhole. Press `Load model`.
2. **Scene**: either a precomputed scene file, or your own images, or a
   TartanAir sample. Upload takes at most 20 images of at most 20 MB each and
   uses the first `num_views` of them, cropped to their central square like the
   Gradio demo. TartanAir lists the test environments first, the training ones
   with a suffix.
3. **Camera**: the camera model, the field of view, `alpha` for UCM. Every
   change renders a new frame. The field of view is clamped to what the camera
   model can represent and the slider follows the clamped value.

Reconstructing replaces the scene, loading a model replaces the model. One lock
serialises all GPU work, so a second browser tab cannot render into a scene that
is being replaced.

## Controls

| Input | Action |
| --- | --- |
| drag | orbit around the target |
| shift or right button and drag | pan |
| wheel | dolly towards the target |
| `W` `A` `S` `D` | fly forwards, backwards, sideways |
| `Q` `E` | fly up and down |
| shift while flying | five times the step |
| `View 1 ... View v` | jump to an input view |
| `Reset view` | back to the scene centre |
| `Save image` | render at full resolution and write a PNG |

## Heavy scenes without a reconstruction

A reconstruction needs bf16 and therefore the cluster, rendering does not. For
work on a smaller GPU, reconstruct once on the cluster and carry the result:

```bash
# on the cluster, inside the container
python notes/viewer/precompute_scene.py --ckpt <checkpoint> --env HongKong \
    --data-root /mnt/projects/p3dcv/anycam/data/tartanair --out scene.pt

# anywhere with a GPU
python -m viewer.server --scenes <directory with scene.pt> --port 7870
python -m viewer.shots --scene scene.pt --out renders/
```

`python -m viewer.server --scene synthetic` needs no file at all. It builds a
chequered floor, three blobs and a ring of coloured columns around the origin,
which makes rendering beyond 90 degrees visible: a 195 degree fisheye shows more
than half of the ring at once.

## Limits

* **Tile mode**: only `projection` draws anything beyond 90 degrees off axis.
  `pbf` bounds the Gaussians in tangent space and drops everything behind the
  camera plane. A run trained with `pbf` has never seen its wide Gaussians out
  there, so rendering it with `projection` can show haze at the rim. For clean
  wide images pick a run trained with `projection`.
* **Pretrained and wide fields of view**: the pretrained model is pinhole only
  and reconstructs only what the input images covered, so a 195 degree frame of
  it is empty at the outside.
* **One user**: model and scene are global state and the last click wins. There
  is no authentication, so keep the viewer inside the VPN and never expose it.
* **Resolution**: 256 to 768 pixels high, the aspect ratio between 0.1 and 10.
  A request outside that is answered with 422, because an unbounded frame size
  is an unbounded allocation.
* **Depth view**: offered only when the build renders expected depth through the
  3DGEER path. The viewer probes it once per scene and hides the option
  otherwise.
