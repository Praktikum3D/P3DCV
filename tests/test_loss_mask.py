"""The loss mask of a wide angle camera, which has to be the same for both renderers.

3DGEER was taken to cover the whole image circle, so train.py asked render_mask for the exact mask.
It does not: Gaussian centres more than about 90 degrees off axis are culled and only splat tails
reach further, so the ring beyond 90 degrees is permanent background without a gradient. At the true
focal length of a 195 degree lcam_fish image that ring is 14.8 percent of the masked pixels and it
carried most of the masked error. These tests pin the mask to the 90 degree border both renderers share.

Run from the repository root:
    pytest tests/test_loss_mask.py
"""

import inspect
import math
import os
import sys

import pytest
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate
import train
from erayzer_core.model.utils_gaussian import pixel_angles, render_mask

H = W = 256
THETA_MAX = math.radians(195.0 / 2)  # lcam_fish sees 195 degrees across its image circle
F_TRUE = (W / 2) / THETA_MAX  # equidistant fisheye, 0.29383 * W


def intrinsics(camera_model, f=F_TRUE, alpha=0.6, v=2):
    """fxfycxcy [v, 4], for UCM [v, 5] with alpha last, principal point in the image centre."""
    k = torch.tensor([[f, f, W / 2, H / 2]]).repeat(v, 1)
    if camera_model == "ucm":
        k = torch.cat([k, torch.full((v, 1), alpha)], dim=1)
    return k


def loss_config(camera_model, renderer):
    return edict({"model": {"camera_model": camera_model, "renderer": renderer},
                  "dataset": {"camera_name": "lcam_fish"},
                  "training": {"l2_loss_weight": 1.0}})


def loss_inputs(fxfycxcy):
    """A render and a target for compute_loss, batch of 1 with as many views as fxfycxcy has."""
    v = fxfycxcy.shape[0]
    g = torch.Generator().manual_seed(0)
    render = torch.rand(1, v, 3, H, W, generator=g)
    target = torch.rand(1, v, 3, H, W, generator=g)
    result = edict(render=render, fxfycxcy_target=fxfycxcy[None], target_image=target)
    return result, {"image": target}


@pytest.mark.parametrize("camera_model", ["fisheye", "ucm"])
def test_training_mask_is_the_same_for_both_renderers(camera_model):
    """The mask the training loss applies no longer depends on which renderer produced the image."""
    fxfycxcy = intrinsics(camera_model)
    fractions = []
    for renderer in ["gsplat", "geer"]:
        result, batch = loss_inputs(fxfycxcy)
        out = train.compute_loss(result, batch, loss_config(camera_model, renderer), edict())
        fractions.append(out.mask_fraction.item())
    assert fractions[0] == fractions[1], f"{camera_model}: gsplat {fractions[0]} vs geer {fractions[1]}"
    expected = render_mask(fxfycxcy, H, W, camera_model, True).mean().item()
    assert fractions[1] == pytest.approx(expected), f"{camera_model}: {fractions[1]} vs {expected}"


@pytest.mark.parametrize("camera_model", ["fisheye", "ucm"])
def test_training_and_evaluation_use_the_same_mask(camera_model):
    """Training and evaluation have to score the very same pixels, whatever the renderer."""
    fxfycxcy = intrinsics(camera_model)
    # the call evaluate.main makes, with its own arguments
    circle = "fish" in str(loss_config(camera_model, "geer").dataset.get("camera_name", ""))
    evaluation = render_mask(fxfycxcy, H, W, camera_model, circle)
    for renderer in ["gsplat", "geer"]:
        result, batch = loss_inputs(fxfycxcy)
        out = train.compute_loss(result, batch, loss_config(camera_model, renderer), edict())
        assert out.mask_fraction.item() == pytest.approx(evaluation.mean().item())
    source = inspect.getsource(evaluate.main)
    assert "exact" not in source, "evaluation asks for a mask the renderers do not fill"


@pytest.mark.parametrize("camera_model", ["fisheye", "ucm"])
def test_the_two_masks_really_differ(camera_model):
    """Guard against the test above passing because render_mask ignores exact."""
    fxfycxcy = intrinsics(camera_model)
    new = render_mask(fxfycxcy, H, W, camera_model, True)
    old = render_mask(fxfycxcy, H, W, camera_model, True, exact=True)
    assert (old >= new).all(), "the 90 degree mask has to be a subset of the full circle"
    assert new.mean().item() < old.mean().item()


def test_the_dead_ring_beyond_90_degrees_is_gone():
    """At the true focal length of a 195 degree image the full circle reaches far past 90 degrees.

    The old mask was the whole inscribed disc of radius W / 2, the new one stops at the radius of the
    90 degree border, f * pi / 2, so the share of pixels drops by the square of the radius ratio. That
    ring is the 14.8 percent of masked pixels the renderer leaves at the background colour.
    """
    fxfycxcy = intrinsics("fisheye")
    new = render_mask(fxfycxcy, H, W, "fisheye", True)
    old = render_mask(fxfycxcy, H, W, "fisheye", True, exact=True)

    assert old.mean().item() == pytest.approx(math.pi / 4, abs=0.002), "the old mask is the whole disc"
    ratio = (F_TRUE * math.pi / 2) / (W / 2)
    assert ratio < 1.0, "a 195 degree image circle reaches beyond 90 degrees"
    assert new.mean().item() == pytest.approx(ratio**2 * old.mean().item(), rel=0.005)
    ring = (old - new).mean().item() / old.mean().item()
    assert ring == pytest.approx(1 - ratio**2, rel=0.01)
    assert 0.14 < ring < 0.16, f"dead ring {ring}"

    # the ring is exactly the pixels between 90 degrees and the edge of the image circle
    _, _, radius, theta, _ = pixel_angles(fxfycxcy, H, W, "fisheye")
    dead = (old - new)[:, 0] > 0
    assert (theta[dead] > math.pi / 2).all()
    assert (radius[dead] <= W / 2).all()
