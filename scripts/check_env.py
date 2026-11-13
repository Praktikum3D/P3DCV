"""Smoke test for the container environment.

Checks the library versions, renders a few Gaussians with gsplat on the GPU
and reports which shared files are present. Exits non-zero on failure.
"""

import os
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "/mnt/projects/p3dcv/anycam"))
PRETRAINED = PROJECT_DIR / "checkpoints" / "pretrained"
EXPECTED_FILES = [
    PRETRAINED / "erayzer_multi.pt",
    PRETRAINED / "erayzer_dl3dv.pt",
    PRETRAINED / "imagenet-vgg-verydeep-19.mat",
]


def check_versions() -> None:
    import torchvision
    import xformers

    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"torchvision {torchvision.__version__}")
    print(f"xformers    {xformers.__version__}")


def check_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the container")
    name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"gpu         {name} ({memory_gb:.0f} GB)")


def check_gsplat() -> None:
    import gsplat
    from gsplat import rasterization

    device = "cuda"
    n = 100
    means = torch.randn(n, 3, device=device) * 0.5 + torch.tensor([0.0, 0.0, 4.0], device=device)
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1)
    scales = torch.full((n, 3), 0.1, device=device)
    opacities = torch.full((n,), 0.8, device=device)
    colors = torch.rand(n, 3, device=device)
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor([[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]], device=device)[None]

    image, _, _ = rasterization(means, quats, scales, opacities, colors, viewmats, Ks, 256, 256)
    if not torch.isfinite(image).all() or image.sum() <= 0:
        raise RuntimeError("gsplat rendered an empty or invalid image")
    print(f"gsplat      {gsplat.__version__} (rendered {tuple(image.shape)})")


def check_files() -> bool:
    ok = True
    for path in EXPECTED_FILES:
        if path.is_file():
            print(f"found       {path} ({path.stat().st_size / 1024**3:.2f} GB)")
        else:
            print(f"MISSING     {path}")
            ok = False
    return ok


def main() -> int:
    check_versions()
    check_gpu()
    check_gsplat()
    files_ok = check_files()
    print("environment OK" if files_ok else "environment OK, but shared files are missing")
    return 0 if files_ok else 1


if __name__ == "__main__":
    sys.exit(main())
