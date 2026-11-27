"""Per sample camera augmentation: resample lcam_fish to a freshly drawn camera every sample.

The model is supposed to read focal length and distortion off the image. As long as every training
frame comes from the same 195 degree lcam_fish camera it can memorise that one camera instead, so
the data loader draws a camera per sample and remaps all views of that sample to it.

These tests pin the four things that can quietly go wrong:

* the remap still agrees with `scripts/tartanair_remap.py` on branch `tartanair-cameras`, whose
  pinhole and fisheye code is reproduced below as the reference,
* the new UCM branch is the exact inverse of `utils_gaussian.camera_theta`, which is what couples
  the augmentation to the renderer and the plucker embeddings,
* the loss and the corner masking really use the per sample valid region,
* with the flag off nothing moves at all.

Run from the repository root:
    pytest tests/test_camera_aug.py
"""

import importlib.util
import inspect
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cv2 = pytest.importorskip("cv2")

import train
from erayzer_core.model.erayzer import (
    CORNER_OPACITY_LOGIT,
    corner_gaussian_mask,
    mask_corner_opacity,
    mask_corner_opacity_from_data,
)
from erayzer_core.model.utils_gaussian import camera_theta
from training.camera_augmentation import (
    DEFAULT_MIX,
    FISH_C,
    FISH_F,
    FISH_FOV_DEG,
    FISH_SIZE,
    MODEL_CODES,
    AugCamera,
    camera_rays,
    fit_ucm_alpha,
    focal_for_fov,
    normalize_mix,
    remap_fisheye,
    remap_views,
    sample_camera,
    true_alpha,
    ucm_rho_norm,
    ucm_theta_max,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_FRAME = "/praktikum/data-sample/AmericanDiner/Data_hard/P000/image_lcam_fish/000000_lcam_fish_image.png"


# --------------------------------------------------------------------------------------------
# the template, reproduced verbatim from scripts/tartanair_remap.py on branch tartanair-cameras
# --------------------------------------------------------------------------------------------

def template_camera_rays(model, fov_deg, size):
    half = math.radians(fov_deg) / 2.0
    c = (size - 1) / 2.0
    u, v = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    x, y = u - c, v - c

    if model == "pinhole":
        if fov_deg >= 180.0:
            raise ValueError("a pinhole camera covers less than 180 degrees")
        f = (size / 2.0) / math.tan(half)
        rays = np.stack([x / f, y / f, np.ones_like(x)], axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        valid = np.ones(x.shape, dtype=bool)
    elif model == "fisheye":
        f = (size / 2.0) / half
        theta = np.hypot(x, y) / f
        phi = np.arctan2(y, x)
        rays = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
        valid = theta <= half
    else:
        raise ValueError(f"unknown camera model {model!r}")
    return rays, valid, f, c


def template_fisheye_pixels(rays):
    theta = np.arctan2(np.hypot(rays[..., 0], rays[..., 1]), rays[..., 2])
    phi = np.arctan2(rays[..., 1], rays[..., 0])
    r = FISH_F * theta
    inside = theta <= math.radians(FISH_FOV_DEG) / 2.0
    return FISH_C + r * np.cos(phi), FISH_C + r * np.sin(phi), inside


def template_remap_fisheye(image, model, fov_deg, size, supersample=4):
    if image.shape[:2] != (FISH_SIZE, FISH_SIZE):
        raise ValueError(f"expected a {FISH_SIZE} x {FISH_SIZE} lcam_fish image, got {image.shape[:2]}")
    big = size * supersample
    rays, valid, f_big, _ = template_camera_rays(model, fov_deg, big)
    map_x, map_y, inside = template_fisheye_pixels(rays)
    out = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = valid & inside
    out[~mask] = 0
    if supersample > 1:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 0.999
    f = f_big / supersample
    c = (size - 1) / 2.0
    return out, mask, (f, f, c, c)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------

def synthetic_fisheye(seed=0):
    """A 1000 x 1000 stand in for an lcam_fish frame, structured enough that remaps differ."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:FISH_SIZE, 0:FISH_SIZE].astype(np.float32)
    base = (
        127.0
        + 60.0 * np.sin(x / 37.0)
        + 50.0 * np.cos(y / 23.0)
        + 20.0 * np.sin((x + y) / 11.0)
    )
    image = np.stack([base, np.roll(base, 13, axis=0), np.roll(base, 29, axis=1)], axis=-1)
    image = image + rng.normal(0.0, 4.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def real_frame_or_skip():
    if not os.path.exists(SAMPLE_FRAME):
        pytest.skip(f"no lcam_fish sample frame at {SAMPLE_FRAME}")
    from PIL import Image

    return np.asarray(Image.open(SAMPLE_FRAME).convert("RGB"))


def loss_config(camera_model="ucm"):
    return edict(
        {
            "model": {"camera_model": camera_model, "renderer": "geer", "geer_tile_mode": "projection"},
            "dataset": {"camera_name": "lcam_fish"},
            "training": {"l2_loss_weight": 1.0},
        }
    )


# --------------------------------------------------------------------------------------------
# the remap still agrees with the template
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("model,fov", [("pinhole", 60.0), ("pinhole", 90.0), ("pinhole", 120.0),
                                       ("fisheye", 120.0), ("fisheye", 150.0), ("fisheye", 195.0)])
def test_remap_matches_the_template(model, fov):
    """The ported pinhole and fisheye paths must be bit for bit the offline script they came from."""
    image = synthetic_fisheye()
    got, got_mask, got_k = remap_fisheye(image, model, fov, 128, supersample=2)
    want, want_mask, want_k = template_remap_fisheye(image, model, fov, 128, supersample=2)
    assert np.array_equal(got, want), f"{model} {fov}: image differs from the template"
    assert np.array_equal(got_mask, want_mask), f"{model} {fov}: mask differs from the template"
    assert got_k == want_k


def test_remap_matches_the_template_on_a_real_frame():
    """The same check once on real lcam_fish data at the native size, not on synthetic texture."""
    image = real_frame_or_skip()
    got, got_mask, _ = remap_fisheye(image, "fisheye", 150.0, 256, supersample=2)
    want, want_mask, _ = template_remap_fisheye(image, "fisheye", 150.0, 256, supersample=2)
    assert np.array_equal(got, want)
    assert np.array_equal(got_mask, want_mask)


def test_identity_remap_returns_the_source():
    """Remapping lcam_fish to its own camera is the identity, the self check of the template."""
    image = real_frame_or_skip()
    same, _, _ = remap_fisheye(image, "fisheye", FISH_FOV_DEG, FISH_SIZE, supersample=1)
    err = np.abs(same.astype(np.int16) - image.astype(np.int16)).max()
    assert err == 0, f"identity remap changed the image by {err}"


# --------------------------------------------------------------------------------------------
# the UCM branch, and how it couples to the renderer
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.05, 0.3, 0.5, 0.62, 0.8, 0.9])
def test_ucm_rays_round_trip_through_camera_theta(alpha):
    """THE coupling: the angle the remap builds a ray from is the angle the renderer reads back.

    `camera_rays` turns a pixel radius into a ray, `utils_gaussian.camera_theta` turns the very same
    radius into the angle the plucker embeddings and 3DGEER use. If these two ever disagree the
    images would be resampled for one camera and rendered for another, which no other test catches.
    """
    size, fov = 96, 140.0
    if math.radians(fov) / 2.0 >= ucm_theta_max(alpha):
        pytest.skip(f"alpha {alpha} cannot reach {fov} degrees")
    rays, valid, f, c = camera_rays("ucm", fov, size, alpha)

    theta_rays = np.arctan2(np.hypot(rays[..., 0], rays[..., 1]), rays[..., 2])
    u, v = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    rho = np.hypot(u - c, v - c) / f

    k = torch.tensor([[f, f, size / 2.0, size / 2.0, alpha]], dtype=torch.float64)
    theta_repo = camera_theta(torch.from_numpy(rho), "ucm", k[:, 4]).numpy()

    err = np.abs(theta_rays - theta_repo)[valid].max()
    assert err < 1e-9, f"alpha {alpha}: remap and camera_theta disagree by {err} rad"


@pytest.mark.parametrize("alpha", [0.3, 0.62, 0.9])
def test_ucm_pixel_of_a_known_ray_matches_the_formula(alpha):
    """A ray at a known angle lands at the radius the UCM formula predicts."""
    size, fov = 128, 150.0
    if math.radians(fov) / 2.0 >= ucm_theta_max(alpha):
        pytest.skip(f"alpha {alpha} cannot reach {fov} degrees")
    f = focal_for_fov("ucm", fov, size, alpha)
    rays, valid, _, c = camera_rays("ucm", fov, size, alpha)

    for theta in np.radians([5.0, 20.0, 45.0, fov / 2.0]):
        want_radius = f * float(ucm_rho_norm(theta, alpha))
        # walk out along the +x row from the centre and find the pixel at that radius
        row = int(round(c))
        col = int(round(c + want_radius))
        if col >= size or not valid[row, col]:
            continue
        got = math.atan2(math.hypot(rays[row, col, 0], rays[row, col, 1]), rays[row, col, 2])
        # the pixel grid quantises the radius, so compare through the formula at the real radius
        real_radius = math.hypot(col - c, row - c)
        want = None
        lo, hi = 0.0, ucm_theta_max(alpha) * (1 - 1e-9)
        for _ in range(200):  # bisect the monotone rho_norm back to an angle
            mid = 0.5 * (lo + hi)
            if f * float(ucm_rho_norm(mid, alpha)) < real_radius:
                lo = mid
            else:
                hi = mid
        want = 0.5 * (lo + hi)
        assert abs(got - want) < 1e-9, f"alpha {alpha} theta {math.degrees(theta)}: {got} vs {want}"


def test_focal_for_fov_puts_the_field_of_view_on_the_image_edge():
    """f is defined by the half field of view landing on the image half width, for all three models."""
    size = 200
    for model, fov, alpha in [("pinhole", 90.0, None), ("fisheye", 195.0, None), ("ucm", 160.0, 0.5)]:
        f = focal_for_fov(model, fov, size, alpha)
        half = math.radians(fov) / 2.0
        if model == "pinhole":
            radius = f * math.tan(half)
        elif model == "fisheye":
            radius = f * half
        else:
            radius = f * float(ucm_rho_norm(half, alpha))
        assert radius == pytest.approx(size / 2.0), f"{model} {fov}"


def test_ucm_alpha_zero_is_a_pinhole():
    """UCM with alpha 0 is exactly the pinhole camera, a sanity check on the projection."""
    rays_u, _, f_u, _ = camera_rays("ucm", 90.0, 64, 0.0)
    rays_p, _, f_p, _ = camera_rays("pinhole", 90.0, 64)
    assert f_u == pytest.approx(f_p)
    assert np.abs(rays_u - rays_p).max() < 1e-12


def test_ucm_rejects_a_field_of_view_its_alpha_cannot_reach():
    assert ucm_theta_max(0.9) < math.radians(195.0) / 2.0
    with pytest.raises(ValueError):
        focal_for_fov("ucm", 195.0, 128, 0.9)


# --------------------------------------------------------------------------------------------
# valid masks
# --------------------------------------------------------------------------------------------

def test_valid_masks_per_model():
    """Pinhole fills the frame; every wide angle camera keeps exactly the inscribed circle.

    f is chosen so that the half field of view lands on the image half width, and the mask is
    "not further out than the half field of view", so the valid region of a fisheye or UCM camera is
    the circle of radius size / 2 whatever alpha is. The alpha constraint of `sample_camera` is what
    keeps the UCM invertibility test from ever cutting in front of that circle, and this test pins
    that too: if it ever did, the mask would be smaller than the circle.
    """
    size = 128
    _, pin, _, _ = camera_rays("pinhole", 90.0, size)
    assert pin.all(), "a pinhole camera has no invalid pixels"

    u, v = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    c = (size - 1) / 2.0
    circle = np.hypot(u - c, v - c) <= size / 2.0

    for model, fov, alpha in [("fisheye", 150.0, None), ("fisheye", 195.0, None),
                              ("ucm", 190.0, 0.52), ("ucm", 130.0, 0.4), ("ucm", 160.0, 0.8)]:
        _, valid, _, _ = camera_rays(model, fov, size, alpha)
        assert np.array_equal(valid, circle), f"{model} {fov} alpha {alpha} is not the inscribed circle"
        assert not valid[0, 0] and not valid[-1, -1], "the corners must be outside"
        assert valid[size // 2, size // 2], "the centre must be inside"
        assert 0.7 < valid.mean() < 0.82, f"circle in a square is about pi/4, got {valid.mean()}"


def test_remap_views_mask_is_the_same_for_every_view():
    """One camera per sample means one valid region for the whole sample."""
    frames = [synthetic_fisheye(seed=i) for i in range(3)]
    images, mask, intr, alpha = remap_views(frames, AugCamera("ucm", 160.0, 0.6), 64, 2)
    assert images.shape == (3, 3, 64, 64)
    assert mask.shape == (3, 1, 64, 64)
    assert torch.equal(mask[0], mask[1]) and torch.equal(mask[1], mask[2])
    assert images.dtype == torch.float32 and 0.0 <= float(images.min()) and float(images.max()) <= 1.0
    # the principal point follows this repository's convention, size / 2, not the template's
    assert intr[2].item() == pytest.approx(32.0) and intr[3].item() == pytest.approx(32.0)
    assert alpha == pytest.approx(0.6)
    # the different frames really produce different images
    assert not torch.equal(images[0], images[1])


def test_remap_views_rejects_a_wrong_source_size():
    with pytest.raises(ValueError):
        remap_views([np.zeros((256, 256, 3), np.uint8)], AugCamera("pinhole", 90.0), 64, 1)


# --------------------------------------------------------------------------------------------
# drawing cameras
# --------------------------------------------------------------------------------------------

def test_sample_camera_keeps_its_ranges_and_its_constraint():
    rng = np.random.default_rng(12345)
    drawn = [sample_camera(rng) for _ in range(4000)]
    seen = {m: 0 for m in MODEL_CODES}
    for cam in drawn:
        seen[cam.model] += 1
        if cam.model == "pinhole":
            assert 60.0 <= cam.fov_deg <= 120.0
            assert cam.alpha is None
        elif cam.model == "fisheye":
            assert 120.0 <= cam.fov_deg <= 195.0
            assert cam.alpha is None
        else:
            assert 120.0 <= cam.fov_deg <= 195.0
            assert 0.3 <= cam.alpha <= 0.9
            # the constraint that makes the camera representable at all
            assert math.radians(cam.fov_deg) / 2.0 < ucm_theta_max(cam.alpha)
            focal_for_fov("ucm", cam.fov_deg, 256, cam.alpha)  # must not raise
    total = len(drawn)
    for model, share in DEFAULT_MIX.items():
        assert abs(seen[model] / total - share) < 0.03, f"{model}: {seen[model] / total} vs {share}"


def test_sample_camera_follows_a_custom_mix():
    rng = np.random.default_rng(7)
    drawn = [sample_camera(rng, {"ucm": 1.0}) for _ in range(200)]
    assert {c.model for c in drawn} == {"ucm"}


def test_normalize_mix_checks_its_input():
    assert normalize_mix({"ucm": 2.0, "pinhole": 2.0}) == {"pinhole": 0.5, "fisheye": 0.0, "ucm": 0.5}
    with pytest.raises(ValueError):
        normalize_mix({"equirect": 1.0})
    with pytest.raises(ValueError):
        normalize_mix({"ucm": -1.0})
    with pytest.raises(ValueError):
        normalize_mix({"ucm": 0.0})


def test_sample_camera_is_deterministic_for_a_seed():
    a = [sample_camera(np.random.default_rng(3)) for _ in range(5)]
    b = [sample_camera(np.random.default_rng(3)) for _ in range(5)]
    assert a == b


# --------------------------------------------------------------------------------------------
# the true alpha of an equidistant camera
# --------------------------------------------------------------------------------------------

def test_true_alpha_of_an_equidistant_camera():
    """The 195 degree lcam_fish camera is best described by a UCM alpha near 0.62.

    0.62 is what `model.alpha_init` starts from, derived independently, so the fit reproducing it is
    a check on both. The tolerance is wide because the fit minimises a radius error, not alpha.
    """
    alpha = fit_ucm_alpha(195.0)
    assert alpha == pytest.approx(0.6207, abs=5e-3), f"got {alpha}"

    # the fit really is the optimum: neighbours on both sides are worse
    half = math.radians(195.0) / 2.0
    thetas = np.linspace(half / 64, half, 64)
    radii = thetas / half

    def cost(a):
        return float(np.sum((ucm_rho_norm(thetas, a) / float(ucm_rho_norm(half, a)) - radii) ** 2))

    assert cost(alpha) < cost(alpha - 0.01) and cost(alpha) < cost(alpha + 0.01)

    # and it describes the camera closely: under a pixel of 128 across the whole range
    scale = 128.0 / float(ucm_rho_norm(half, alpha))
    err = np.abs(scale * ucm_rho_norm(thetas, alpha) - 128.0 * radii).max()
    assert err < 1.0, f"radius error {err} px of 128"


def test_true_alpha_per_model():
    assert true_alpha(AugCamera("pinhole", 90.0)) == 0.0
    assert true_alpha(AugCamera("ucm", 150.0, 0.44)) == pytest.approx(0.44)
    assert 0.5 < true_alpha(AugCamera("fisheye", 150.0)) < 0.8


# --------------------------------------------------------------------------------------------
# the loss mask
# --------------------------------------------------------------------------------------------

def loss_inputs(v=3, h=32, w=32, camera_model="ucm", f=None, alpha=0.62):
    """A render, a target and a predicted camera. The default predicted camera is the wide one of a
    195 degree fisheye run; pass f and alpha to stand in for a narrower prediction."""
    g = torch.Generator().manual_seed(0)
    render = torch.rand(1, v, 3, h, w, generator=g)
    target = torch.rand(1, v, 3, h, w, generator=g)
    if f is None:
        f = (w / 2) / math.radians(195.0 / 2)
    k = torch.tensor([[f, f, w / 2, h / 2, alpha]]).repeat(v, 1)
    if camera_model != "ucm":
        k = k[:, :4]
    result = edict(render=render, fxfycxcy_target=k[None], target_image=target)
    return result, {"image": target}


def test_loss_uses_the_intersection_of_both_masks():
    """The loss ignores pixels the data does not hold, on top of the ones the camera cannot render."""
    from erayzer_core.model.utils_gaussian import loss_render_mask

    config = loss_config("ucm")
    result, batch = loss_inputs()
    b, v, _, h, w = result.render.shape
    # a data mask is about to be supplied, so the static image circle no longer applies
    camera_mask = loss_render_mask(config, result.fxfycxcy_target.reshape(b * v, -1), h, w,
                                   has_data_mask=True)
    camera_mask = camera_mask.reshape(b, v, 1, h, w)

    # a data mask that drops the left half of every view
    data_mask = torch.ones_like(camera_mask)
    data_mask[..., : w // 2] = 0.0

    without = train.compute_loss(result, batch, config, edict())
    result.target_data_mask = data_mask
    with_mask = train.compute_loss(result, batch, config, edict())

    expected = float((camera_mask * data_mask).mean())
    assert with_mask.mask_fraction.item() == pytest.approx(expected, rel=1e-6)
    assert with_mask.mask_fraction.item() < without.mask_fraction.item()


def test_loss_ignores_the_masked_pixels_exactly():
    """Changing a render pixel outside the intersection must not move the loss at all."""
    config = loss_config("ucm")
    result, batch = loss_inputs()
    data_mask = torch.ones(1, 3, 1, 32, 32)
    data_mask[..., :8] = 0.0
    result.target_data_mask = data_mask

    before = train.compute_loss(result, batch, config, edict()).loss.item()
    result.render = result.render.clone()
    result.render[:, :, :, :, :8] = 0.37  # a masked column
    after = train.compute_loss(result, batch, config, edict()).loss.item()
    assert after == pytest.approx(before, rel=1e-9), "a masked pixel changed the loss"

    result.render[:, :, :, 16, 16] += 0.5  # an unmasked pixel
    moved = train.compute_loss(result, batch, config, edict()).loss.item()
    assert moved != pytest.approx(before, rel=1e-9), "an unmasked pixel did not change the loss"


def test_loss_applies_the_data_mask_for_a_pinhole_model_too():
    """A pinhole run has no camera mask, but augmented data still has invalid pixels."""
    config = loss_config("pinhole")
    result, batch = loss_inputs(camera_model="pinhole")
    plain = train.compute_loss(result, batch, config, edict())
    assert plain.mask_fraction.item() == pytest.approx(1.0)

    data_mask = torch.ones(1, 3, 1, 32, 32)
    data_mask[..., :16] = 0.0
    result.target_data_mask = data_mask
    masked = train.compute_loss(result, batch, config, edict())
    assert masked.mask_fraction.item() == pytest.approx(0.5, rel=1e-6)


def effective_loss_mask(result, batch, config):
    """The mask compute_loss really applies, recovered by probing one pixel at a time is too slow,
    so rebuild it the way compute_loss does and check the fraction it reports agrees."""
    from erayzer_core.model.utils_gaussian import loss_render_mask

    b, v, _, h, w = result.render.shape
    data_mask = result.get("target_data_mask")
    mask = None
    if config.model.camera_model != "pinhole":
        k = result.fxfycxcy_target.detach().float().reshape(b * v, -1)
        mask = loss_render_mask(config, k, h, w, has_data_mask=data_mask is not None)
        mask = mask.reshape(b, v, 1, h, w)
    if data_mask is not None:
        mask = data_mask if mask is None else mask * data_mask
    reported = train.compute_loss(result, batch, config, edict()).mask_fraction.item()
    assert reported == pytest.approx(float(mask.mean()), rel=1e-6)
    return mask


def test_an_augmented_pinhole_sample_keeps_its_corners():
    """The corners of a pinhole sample carry the strongest focal length signal and must be scored.

    `loss_render_mask` guesses the image circle from `dataset.camera_name`, which says lcam_fish for
    an augmented run even when this particular sample was drawn as a 90 degree pinhole whose data
    mask is all ones. Left alone that guess throws away about 21 percent of the pinhole targets.
    """
    config = loss_config("ucm")  # a UCM run, but this sample was drawn as a pinhole
    # the predicted camera of a pinhole sample is a narrow one, not the wide default
    result, batch = loss_inputs(f=16.0, alpha=0.05)
    _, _, _, h, w = result.render.shape

    # a pinhole draw: the data mask covers the whole frame
    result.target_data_mask = torch.ones(1, 3, 1, h, w)
    mask = effective_loss_mask(result, batch, config)
    assert mask[0, 0, 0, 0, 0].item() == pytest.approx(1.0), "the corner is not scored"
    assert mask[0, 0, 0, -1, -1].item() == pytest.approx(1.0), "the corner is not scored"

    # and it is exactly the predicted camera's own validity, nothing else cut away
    from erayzer_core.model.utils_gaussian import render_mask

    b, v = 1, 3
    k = result.fxfycxcy_target.detach().float().reshape(b * v, -1)
    predicted = render_mask(k, h, w, "ucm", image_circle=False, exact=True).reshape(b, v, 1, h, w)
    assert torch.equal(mask, predicted)


def test_an_augmented_fisheye_sample_still_loses_its_corners():
    """The circle is not gone, it just comes from the sample's own data mask now."""
    config = loss_config("ucm")
    result, batch = loss_inputs()
    _, _, _, h, w = result.render.shape

    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    circle = ((x + 0.5 - w / 2) ** 2 + (y + 0.5 - h / 2) ** 2 <= (w / 2) ** 2).float()
    result.target_data_mask = circle[None, None, None].expand(1, 3, 1, h, w).contiguous()

    mask = effective_loss_mask(result, batch, config)
    assert mask[0, 0, 0, 0, 0].item() == 0.0, "the corner must stay out for a wide angle sample"
    assert mask[0, 0, 0, -1, -1].item() == 0.0
    assert mask[0, 0, 0, h // 2, w // 2].item() > 0.0, "the centre must stay in"


def test_the_image_circle_guess_is_unchanged_without_a_data_mask():
    """No data mask means no augmentation, and then the old guess has to apply exactly as before."""
    from erayzer_core.model.utils_gaussian import loss_render_mask

    config = loss_config("ucm")
    k = loss_inputs()[0].fxfycxcy_target.reshape(3, -1)
    old = loss_render_mask(config, k, 32, 32)
    assert torch.equal(loss_render_mask(config, k, 32, 32, has_data_mask=False), old)
    # the default argument is what evaluate.py relies on, so it must be the unchanged behaviour
    assert old[0, 0, 0, 0].item() == 0.0, "the fisheye guess still cuts the corner by default"
    assert not torch.equal(loss_render_mask(config, k, 32, 32, has_data_mask=True), old)


def test_loss_is_byte_identical_without_a_data_mask():
    """Default off: every number compute_loss returns is exactly what it returned before."""
    for camera_model in ["pinhole", "fisheye", "ucm"]:
        config = loss_config(camera_model)
        result, batch = loss_inputs(camera_model=camera_model)
        out = train.compute_loss(result, batch, config, edict())
        assert "target_data_mask" not in result
        # the reference values of the untouched code path, recomputed here the old way
        pred = result.render.clamp(0.0, 1.0)
        target = batch["image"]
        inside = 1.0
        if camera_model != "pinhole":
            from erayzer_core.model.utils_gaussian import loss_render_mask

            b, v, _, h, w = pred.shape
            mask = loss_render_mask(config, result.fxfycxcy_target.detach().float().reshape(b * v, -1), h, w)
            mask = mask.reshape(b, v, 1, h, w).to(pred.dtype)
            pred, inside = pred * mask + target * (1 - mask), max(mask.mean().item(), 1e-3)
        mse = torch.nn.functional.mse_loss(pred, target) / inside
        assert out.mse.item() == mse.item(), camera_model
        assert out.mask_fraction.item() == inside


# --------------------------------------------------------------------------------------------
# corner masking from the per sample data mask
# --------------------------------------------------------------------------------------------

def test_data_mask_corner_masking_matches_the_circle_when_fed_the_circle():
    """Feeding the old fixed circle in as a data mask must reproduce the old result exactly."""
    hh = ww = 4
    ph = pw = 2
    h, w = hh * ph, ww * pw
    n = hh * ww * ph * pw
    v_input = 3
    opacity = torch.randn(1, v_input * n, 1, generator=torch.Generator().manual_seed(1))

    circle = corner_gaussian_mask(hh, ww, ph, pw)
    old = mask_corner_opacity(opacity, circle, v_input)

    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    inside = ((x + 0.5 - w / 2) ** 2 + (y + 0.5 - h / 2) ** 2 <= (w / 2) ** 2).float()
    data_mask = inside[None, None, None].expand(1, v_input, 1, h, w).contiguous()
    new = mask_corner_opacity_from_data(opacity, data_mask, ph, pw)

    assert torch.equal(old, new), "the data mask path disagrees with the fixed circle"


def test_data_mask_corner_masking_follows_the_mask_per_view():
    """Different views may now have different invalid regions, and the order must still line up."""
    hh = ww = 2
    ph = pw = 2
    h, w = hh * ph, ww * pw
    n = hh * ww * ph * pw
    opacity = torch.zeros(1, 2 * n, 1)
    data_mask = torch.ones(1, 2, 1, h, w)
    data_mask[0, 0, 0, 1, 2] = 0.0  # view 0, pixel (y=1, x=2)
    data_mask[0, 1, 0, 3, 0] = 0.0  # view 1, pixel (y=3, x=0)
    out = mask_corner_opacity_from_data(opacity, data_mask, ph, pw)[0, :, 0]

    def gaussian_index(view, y, x):
        return view * n + ((y // ph * ww + x // pw) * ph + y % ph) * pw + x % pw

    hidden = {gaussian_index(0, 1, 2), gaussian_index(1, 3, 0)}
    for i in range(2 * n):
        if i in hidden:
            assert out[i].item() == CORNER_OPACITY_LOGIT, f"gaussian {i} should be hidden"
        else:
            assert out[i].item() == 0.0, f"gaussian {i} should be untouched"


# --------------------------------------------------------------------------------------------
# the dataset
# --------------------------------------------------------------------------------------------

def write_trajectory(root, frames, camera="lcam_fish"):
    from PIL import Image

    img_dir = os.path.join(root, "Scene", "Data_hard", "P000", f"image_{camera}")
    os.makedirs(img_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(img_dir, f"{i:06d}_{camera}_image.png"))
    poses = np.zeros((len(frames), 7), dtype=np.float64)
    poses[:, 0] = np.arange(len(frames))
    poses[:, 6] = 1.0
    np.savetxt(os.path.join(root, "Scene", "Data_hard", "P000", f"pose_{camera}.txt"), poses)
    return root


def build_dataset(root, **kwargs):
    from training.tartanair_dataset import TartanAirDataset

    options = dict(
        image_size=32,
        patch_size=16,
        num_views=2,
        num_input_views=1,
        view_selector={"type": "uniform"},
        camera_name="lcam_fish",
        normalize_intrinsics=True,
    )
    options.update(kwargs)
    return TartanAirDataset(root, **options)


@pytest.fixture(scope="module")
def tiny_root():
    with tempfile.TemporaryDirectory() as tmp:
        write_trajectory(tmp, [synthetic_fisheye(seed=i) for i in range(3)])
        yield tmp


def reference_load_image_and_intrinsics(path, image_size, patch_size, center_crop, normalize):
    """The image and intrinsic pipeline of the dataset before camera augmentation existed.

    Reproduced here rather than imported, so the "nothing changed with the flag off" guarantee is
    pinned by an independent implementation and does not need a git checkout to run.
    """
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    fx, fy, cx, cy = width * 0.5, height * 0.5, width * 0.5, height * 0.5

    if height < width:
        resize_y = image_size
        resize_x = int(width * (image_size / height))
    else:
        resize_x = image_size
        resize_y = int(height * (image_size / width))
    resize_x = max(patch_size, (resize_x // patch_size) * patch_size)
    resize_y = max(patch_size, (resize_y // patch_size) * patch_size)

    scale_x, scale_y = resize_x / width, resize_y / height
    image = image.resize((resize_x, resize_y), Image.Resampling.LANCZOS)
    fx, fy, cx, cy = fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y
    width, height = image.size

    if center_crop and width != height:
        crop = min(width, height)
        left, top = (width - crop) // 2, (height - crop) // 2
        image = image.crop((left, top, left + crop, top + crop))
        cx, cy = cx - left, cy - top

    if normalize:
        intr = torch.tensor([fx / image_size, fy / image_size, cx / image_size, cy / image_size],
                            dtype=torch.float32)
    else:
        intr = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous(), intr


def test_dataset_off_is_byte_identical_to_the_old_pipeline(tiny_root):
    """Default off: same keys, same tensors, bit for bit, as before camera augmentation existed."""
    import random as pyrandom

    data = build_dataset(tiny_root)
    pyrandom.seed(4242)
    got = data[0]

    assert list(got.keys()) == ["image", "fxfycxcy", "c2w", "index"], "the returned keys changed"
    assert "data_mask" not in got and "aug_fov_true" not in got

    # the uniform selector takes frames 0 and 2 of the three written by the fixture
    frames = sorted(
        os.path.join(root, name)
        for root, _, names in os.walk(tiny_root)
        for name in names
        if name.endswith(".png")
    )
    for view, frame_idx in enumerate([0, 2]):
        want_image, want_intr = reference_load_image_and_intrinsics(
            frames[frame_idx], image_size=32, patch_size=16, center_crop=True, normalize=True
        )
        assert torch.equal(got["image"][view], want_image), f"view {view}: image changed"
        assert torch.equal(got["fxfycxcy"][view], want_intr), f"view {view}: intrinsics changed"
    assert got["index"].tolist() == [[0, 0], [2, 0]]


def test_dataset_off_matches_the_previous_revision(tiny_root):
    """The same guarantee against the real previous revision, where a git checkout is reachable."""
    try:
        source = subprocess.run(
            ["git", "show", "98d58a6:training/tartanair_dataset.py"],
            cwd=REPO, capture_output=True, check=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        pytest.skip(f"cannot read the previous revision (git worktrees do not resolve here): {exc}")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "old_tartanair_dataset.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        spec = importlib.util.spec_from_file_location("old_tartanair_dataset", path)
        old_module = importlib.util.module_from_spec(spec)
        sys.modules["old_tartanair_dataset"] = old_module
        spec.loader.exec_module(old_module)

    options = dict(
        image_size=32, patch_size=16, num_views=2, num_input_views=1,
        view_selector={"type": "uniform"}, camera_name="lcam_fish", normalize_intrinsics=True,
    )
    old = old_module.TartanAirDataset(tiny_root, **options)
    new = build_dataset(tiny_root)

    import random as pyrandom

    pyrandom.seed(4242)
    want = old[0]
    pyrandom.seed(4242)
    got = new[0]

    assert list(want.keys()) == list(got.keys()), "the returned keys changed"
    for key in want:
        assert torch.equal(want[key], got[key]), f"{key} changed with the flag off"


def test_dataset_augmented_sample(tiny_root):
    """With the flag on every view of a sample shares one camera and the extra fields are there."""
    import random as pyrandom

    data = build_dataset(tiny_root, camera_augmentation=True)
    pyrandom.seed(11)
    sample = data[0]

    assert sample["image"].shape == (2, 3, 32, 32)
    assert sample["data_mask"].shape == (2, 1, 32, 32)
    assert torch.equal(sample["data_mask"][0], sample["data_mask"][1]), "one camera per sample"
    assert torch.equal(sample["fxfycxcy"][0], sample["fxfycxcy"][1]), "one camera per sample"
    assert sample["aug_model"].item() in set(MODEL_CODES.values())
    assert 60.0 <= sample["aug_fov_true"].item() <= 195.0
    assert 0.0 <= sample["aug_alpha_true"].item() <= 1.0
    # normalized intrinsics: the principal point sits at half the image
    assert sample["fxfycxcy"][0, 2].item() == pytest.approx(0.5)
    assert sample["c2w"].shape == (2, 4, 4)


def test_dataset_augmentation_is_deterministic_for_a_seed(tiny_root):
    import random as pyrandom

    data = build_dataset(tiny_root, camera_augmentation=True)
    pyrandom.seed(99)
    a = data[0]
    pyrandom.seed(99)
    b = data[0]
    assert torch.equal(a["image"], b["image"])
    assert a["aug_fov_true"].item() == b["aug_fov_true"].item()
    assert a["aug_model"].item() == b["aug_model"].item()


def test_dataset_augmentation_draws_different_cameras(tiny_root):
    import random as pyrandom

    data = build_dataset(tiny_root, camera_augmentation=True)
    pyrandom.seed(5)
    fovs = {round(data[0]["aug_fov_true"].item(), 6) for _ in range(12)}
    assert len(fovs) > 8, f"the camera barely changes between samples: {fovs}"


def test_dataset_augmentation_needs_a_fisheye_camera(tiny_root):
    with pytest.raises(ValueError, match="lcam_fish"):
        build_dataset(tiny_root, camera_augmentation=True, camera_name="lcam_front")


def test_dataset_collates_into_a_batch(tiny_root):
    from torch.utils.data import DataLoader

    data = build_dataset(tiny_root, camera_augmentation=True, data_repeat=4)
    batch = next(iter(DataLoader(data, batch_size=2)))
    assert batch["data_mask"].shape == (2, 2, 1, 32, 32)
    assert batch["aug_fov_true"].shape == (2,)
    assert batch["aug_alpha_true"].shape == (2,)
    assert batch["aug_model"].shape == (2,)


# --------------------------------------------------------------------------------------------
# the wandb metrics
# --------------------------------------------------------------------------------------------

def test_aug_stats_are_empty_without_augmentation():
    result = edict(fxfycxcy=torch.tensor([[0.3, 0.3, 0.5, 0.5]]))
    assert train.aug_stats(result, {"image": torch.zeros(1)}, "fisheye") == {}


def test_aug_stats_measure_the_error_against_the_drawn_camera():
    """A perfect prediction scores zero, and the per model split picks the right samples."""
    b, v = 2, 2
    fov_true = torch.tensor([150.0, 90.0])
    alpha_true = torch.tensor([0.5, 0.0])
    # the f that exactly produces those cameras, in the normalized units camera_stats uses
    fx = torch.tensor([0.5 / float(ucm_rho_norm(math.radians(150.0) / 2, 0.5)),
                       0.5 / float(ucm_rho_norm(math.radians(90.0) / 2, 0.0))])
    k = torch.stack([fx, fx, torch.full((b,), 0.5), torch.full((b,), 0.5), alpha_true], dim=1)
    result = edict(fxfycxcy=k[:, None].expand(b, v, 5).reshape(b * v, 5))
    batch = {
        "aug_fov_true": fov_true,
        "aug_alpha_true": alpha_true,
        "aug_model": torch.tensor([MODEL_CODES["ucm"], MODEL_CODES["pinhole"]]),
    }
    stats = train.aug_stats(result, batch, "ucm")
    assert stats["aug/fov_err_deg"] == pytest.approx(0.0, abs=1e-4)
    assert stats["aug/alpha_err"] == pytest.approx(0.0, abs=1e-6)
    assert stats["aug/fov_err_deg_ucm"] == pytest.approx(0.0, abs=1e-4)
    assert stats["aug/fov_err_deg_pinhole"] == pytest.approx(0.0, abs=1e-4)
    assert "aug/fov_err_deg_fisheye" not in stats

    # a wrong prediction shows up
    wrong = edict(fxfycxcy=result.fxfycxcy.clone())
    wrong.fxfycxcy[:, 0] *= 1.4
    assert train.aug_stats(wrong, batch, "ucm")["aug/fov_err_deg"] > 5.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_aug_stats_survive_a_batch_on_the_gpu():
    """train.py moves every batch tensor to the training device, the stats must cope with that."""
    b, v = 2, 2
    fx = torch.tensor([0.3, 0.4])
    k = torch.stack([fx, fx, torch.full((b,), 0.5), torch.full((b,), 0.5), torch.full((b,), 0.6)], dim=1)
    result = edict(fxfycxcy=k[:, None].expand(b, v, 5).reshape(b * v, 5).cuda())
    batch = {
        "aug_fov_true": torch.tensor([150.0, 120.0]).cuda(),
        "aug_alpha_true": torch.tensor([0.6, 0.6]).cuda(),
        "aug_model": torch.tensor([MODEL_CODES["ucm"], MODEL_CODES["ucm"]]).cuda(),
    }
    stats = train.aug_stats(result, batch, "ucm")
    assert stats["aug/fov_err_deg"] >= 0.0 and math.isfinite(stats["aug/fov_err_deg"])
    assert stats["aug/alpha_err"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_loss_data_mask_survives_a_batch_on_the_gpu():
    config = loss_config("ucm")
    result, batch = loss_inputs()
    result.render = result.render.cuda()
    result.target_image = result.target_image.cuda()
    result.fxfycxcy_target = result.fxfycxcy_target.cuda()
    batch = {"image": batch["image"].cuda()}
    result.target_data_mask = torch.ones(1, 3, 1, 32, 32).cuda()
    out = train.compute_loss(result, batch, config, edict())
    assert math.isfinite(out.loss.item())


# --------------------------------------------------------------------------------------------
# the model plumbing
# --------------------------------------------------------------------------------------------

def test_forward_pass_plumbs_the_data_mask_through():
    """Source level, because a full ERayZer forward needs a CUDA renderer.

    The four things that have to hold: the mask is padded with the images, it is split into input
    and target views like every other per view tensor, the input half drives the corner masking and
    the target half reaches the result, where compute_loss picks it up.
    """
    from erayzer_core.model.erayzer import ERayZer

    source = inspect.getsource(ERayZer.forward)
    assert 'data.get("data_mask")' in source
    # padded together with the images, or the view indices would point at the wrong views
    assert source.index("data_mask_all = torch.cat") < source.index("v_all = v")
    # split with the same batch_idx / input_idx gather the images use
    assert "data_mask_all[batch_idx, input_idx]" in source
    assert "data_mask_all[batch_idx, target_idx]" in source
    # the split happens before the masking and before the result
    split = source.index("data_mask_input = data_mask_all[batch_idx, input_idx]")
    assert split < source.index("mask_corner_opacity_from_data")
    assert split < source.index("result.target_data_mask")
    # and the old path is still there untouched for runs without augmentation
    assert "mask_corner_opacity(opacity, self.corner_mask, v_input)" in source


def test_data_mask_follows_the_same_views_as_the_images():
    """The gather that splits input from target views must treat mask and image identically."""
    b, v_all, h, w = 2, 6, 4, 4
    images = torch.arange(b * v_all * 3 * h * w, dtype=torch.float32).reshape(b, v_all, 3, h, w)
    masks = torch.arange(b * v_all * 1 * h * w, dtype=torch.float32).reshape(b, v_all, 1, h, w)
    batch_idx = torch.arange(b).unsqueeze(1)
    input_idx = torch.tensor([[0, 3, 5], [1, 2, 4]])

    picked_images = images[batch_idx, input_idx]
    picked_masks = masks[batch_idx, input_idx]
    for sample in range(b):
        for slot, view in enumerate(input_idx[sample].tolist()):
            assert torch.equal(picked_images[sample, slot], images[sample, view])
            assert torch.equal(picked_masks[sample, slot], masks[sample, view])


# --------------------------------------------------------------------------------------------
# cost, reported not asserted
# --------------------------------------------------------------------------------------------

def test_remap_speed(capsys):
    """Measure what one augmented image costs, so the supersample default stays justified."""
    image = real_frame_or_skip() if os.path.exists(SAMPLE_FRAME) else synthetic_fisheye()
    frames = [image] * 10
    lines = []
    for supersample in (2, 4):
        for camera in (AugCamera("pinhole", 90.0), AugCamera("fisheye", 180.0), AugCamera("ucm", 170.0, 0.6)):
            remap_views(frames[:1], camera, 256, supersample)  # warm the tables
            start = time.perf_counter()
            remap_views(frames, camera, 256, supersample)
            per_image = (time.perf_counter() - start) / len(frames) * 1000.0
            lines.append(f"supersample {supersample} {camera.model:8s} {per_image:6.2f} ms per image")
    with capsys.disabled():
        print("\n" + "\n".join(lines))
