"""Shared helpers for the Kannala-Brandt (KB) gsplat and Plücker tests.

* CPU tests load the PyTorch reference straight from the vendored file
  third_party/gsplat/gsplat/cuda/_torch_impl.py, so they run without building
  or installing gsplat.
* CUDA tests import the installed gsplat (the modified fork, built by
  slurm/test_gsplat_kb.sh) and skip when no GPU is available.
* erayzer.py imports the renderer (gsplat, cv2, plyfile, videoio). On machines
  without these packages, they are replaced by empty stub modules so that the
  pure PyTorch function cam_info_to_plucker() can still be tested.
"""

import importlib
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TORCH_IMPL_PATH = REPO_ROOT / "third_party/gsplat/gsplat/cuda/_torch_impl.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device and the built gsplat fork"
)


def load_torch_impl():
    """The vendored gsplat PyTorch reference, loaded by file path."""
    name = "p3dcv_gsplat_torch_impl"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TORCH_IMPL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stub_if_missing(name, attrs=()):
    try:
        importlib.import_module(name)
    except Exception:  # ImportError, or a broken binary package
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, None)
        sys.modules[name] = module


def import_erayzer():
    """erayzer_core.model.erayzer, with heavy optional imports stubbed if absent."""
    _stub_if_missing("gsplat", ["rasterization"])
    _stub_if_missing("cv2")
    _stub_if_missing("videoio")
    _stub_if_missing("plyfile", ["PlyData", "PlyElement"])
    _stub_if_missing("matplotlib")
    _stub_if_missing("xformers")
    _stub_if_missing("xformers.ops")
    if isinstance(sys.modules.get("xformers"), types.ModuleType) and not hasattr(sys.modules["xformers"], "ops"):
        sys.modules["xformers"].ops = sys.modules["xformers.ops"]
    import erayzer_core  # noqa: F401  registers the model.* / utils.* aliases

    return importlib.import_module("erayzer_core.model.erayzer")


def import_camera_utils():
    return importlib.import_module("erayzer_core.utils.camera_utils")


def report(name, data):
    """Append a result line to the file named by $GSPLAT_KB_REPORT (if set) and print it."""
    line = json.dumps({"test": name, **data}, default=float)
    print(line)
    path = os.environ.get("GSPLAT_KB_REPORT")
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def sphere_point(theta, phi=0.7, rho=2.0):
    """Camera-space point at incidence angle theta, azimuth phi, distance rho."""
    return [
        rho * math.sin(theta) * math.cos(phi),
        rho * math.sin(theta) * math.sin(phi),
        rho * math.cos(theta),
    ]


# Cameras used throughout. TartanAir lcam_fish: equidistant, 195 degrees.
TARTANAIR_FISH = dict(width=1000, height=1000, f=293.8, c=499.5, k=(0.0, 0.0, 0.0, 0.0))
# A moderately distorted KB camera (psi' stays positive up to 0.75 pi).
KB_MILD = dict(width=640, height=640, f=200.0, c=319.5, k=(0.03, -0.01, 0.002, -0.0001))
# A strongly distorted KB camera whose psi' has a root (theta_max ~ 1.61 rad).
KB_STRONG_K = (-0.3, 0.1, -0.02, 0.001)
