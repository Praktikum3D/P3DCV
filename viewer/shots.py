"""Screenshots of one scene through every camera model, without starting the server.

The viewer proves "any camera" by showing the same Gaussians as a pinhole, as an equidistant fisheye
and under the unified camera model. This renders exactly that set as PNG files, which is what the
acceptance of the viewer asks for and what the render examples of a pull request need.

    python -m viewer.shots --scene synthetic --out ~/renders/viewer
    python -m viewer.shots --scene ~/runs/viewer/hongkong.pt --out ~/renders/viewer --size 768
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import torch

from .render import encode_png, render_view
from .scene import load_scene, make_synthetic_scene

# name, camera model, field of view across the image width, alpha
SHOTS: List[Tuple[str, str, float, float]] = [
    ("pinhole_90", "pinhole", 90.0, 0.5),
    ("fisheye_195", "fisheye", 195.0, 0.5),
    ("ucm_a07_160", "ucm", 160.0, 0.7),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="synthetic", help="a scene file written by the viewer, or 'synthetic'")
    parser.add_argument("--out", required=True, help="directory the PNG files go into")
    parser.add_argument("--size", type=int, default=512, help="square image side in pixels")
    parser.add_argument("--tile-mode", default="projection", choices=("projection", "pbf"))
    parser.add_argument("--view", default="rgb", choices=("rgb", "alpha", "depth"))
    parser.add_argument("--no-frusta", action="store_true", help="leave out the input camera cones")
    parser.add_argument("--prefix", default="", help="put in front of every file name")
    parser.add_argument("--view-index", type=int, default=None,
                        help="render from this input view instead of the scene centre")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    scene = make_synthetic_scene(device) if args.scene == "synthetic" else load_scene(args.scene, device)
    if args.view_index is None:
        c2w = torch.eye(4, dtype=torch.float32, device=device)
        c2w[:3, 3] = scene.center
    else:
        c2w = scene.c2w_input[args.view_index].to(device)

    os.makedirs(args.out, exist_ok=True)
    for name, camera_model, fov_deg, alpha in SHOTS:
        image, info = render_view(
            scene,
            c2w,
            camera_model,
            fov_deg,
            alpha,
            args.size,
            args.size,
            tile_mode=args.tile_mode,
            view=args.view,
            frusta=not args.no_frusta,
        )
        path = os.path.join(args.out, f"{args.prefix}{name}.png")
        with open(path, "wb") as handle:
            handle.write(encode_png(image))
        print(f"{path}, fov {info['fov_used']:.1f} degrees, {info['render_ms']:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
