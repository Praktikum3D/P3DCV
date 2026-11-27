"""Tests 4 and 5a: the default pinhole path and the fisheye forward are unchanged.

Needs the file written by regression_reference.py with the unmodified fork,
named by $GSPLAT_KB_REGRESSION_REF (slurm/test_gsplat_kb.sh sets it up).

* Forward images and alphas must be bit-identical.
* Gradients go through atomicAdd, whose summation order is not deterministic
  between runs, so they are compared against the reference with a tight
  tolerance, next to the run-to-run spread of the modified build itself.
  For the pinhole path this includes v_Ks. For fisheye, v_Ks used to be
  silently zero and is now the real gradient, so it is excluded here (it is
  checked in test_kernel_cuda.py).
"""

import os

import pytest
import torch

from conftest import report, requires_cuda
import regression_reference as rr

pytestmark = requires_cuda

REF_PATH = os.environ.get("GSPLAT_KB_REGRESSION_REF")


@pytest.fixture(scope="module")
def reference():
    if not REF_PATH or not os.path.exists(REF_PATH):
        pytest.skip("set GSPLAT_KB_REGRESSION_REF to the output of regression_reference.py")
    return torch.load(REF_PATH)


@pytest.mark.parametrize("config", rr.CONFIGS, ids=[c[0] for c in rr.CONFIGS])
def test_unchanged(reference, config):
    name, camera_model = config[0], config[1]
    ref = reference[name]
    new = rr.compute(config)
    again = rr.compute(config)

    assert torch.equal(new["render"], ref["render"]), name
    assert torch.equal(new["alphas"], ref["alphas"]), name

    stats = {}
    for key in ref:
        if not key.startswith("grad_"):
            continue
        if camera_model == "fisheye" and key == "grad_Ks":
            # used to be zero; now the actual gradient (see module docstring)
            assert float(ref[key].abs().max()) == 0.0
            continue
        scale = float(ref[key].abs().max())
        diff = float((new[key] - ref[key]).abs().max())
        noise = float((new[key] - again[key]).abs().max())
        stats[key] = dict(rel_diff=diff / max(scale, 1e-12), rel_run_to_run=noise / max(scale, 1e-12))
        assert diff <= max(1e-5 * scale, 4 * noise), (name, key, diff, noise, scale)
    report(f"regression[{name}]", dict(forward_identical=True, **{k: v["rel_diff"] for k, v in stats.items()}))
