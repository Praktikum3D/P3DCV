"""The Gaussians of the black fisheye corners, which have to be invisible from the start.

E-RayZer turns every pixel of every input image into a Gaussian. A TartanAir lcam_fish image only
fills the circle spanning its width, so about 21 percent of its pixels are black corners. Those
Gaussians get no gradient at all, because the training loss masks the very same circle away on the
target side, and nothing stops them from drifting into the view of another camera and painting
garbage there. Setting their opacity logit to a large negative constant takes them out of every
rendering without touching a single supervised Gaussian.

These tests pin the circle, the order in which the Gaussians are laid out, the fact that only the
corner entries change, and the default that leaves the pinhole training exactly as it was.

Run from the repository root:
    pytest tests/test_mask_corners.py
"""

import inspect
import math
import os
import sys

import pytest
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erayzer_core.model.erayzer import (
    CORNER_OPACITY_LOGIT,
    corner_gaussian_mask,
    corner_masking_default,
    mask_corner_opacity,
)


def pixel_is_corner(y, x, h, w):
    """The image circle of render_mask, written out by hand for one pixel."""
    return (x + 0.5 - w / 2) ** 2 + (y + 0.5 - h / 2) ** 2 > (w / 2) ** 2


def tiny_model_config(camera_model, camera_name=None, **model):
    """The smallest ERayZer the constructor accepts, a few thousand parameters on the CPU."""
    config = edict(
        {
            "inference": False,
            "model": {
                "image_tokenizer": {"image_size": 16, "patch_size": 8, "in_channels": 3},
                "transformer": {
                    "d": 8,
                    "d_head": 4,
                    "encoder_n_layer": 1,
                    "encoder_geom_n_layer": 1,
                    "decoder_n_layer": 1,
                    "neck_n_layer": 1,
                },
                "pose_latent": {"length": 1, "canonical": "middle", "mode": "pairwise"},
                "camera_model": camera_model,
                "alpha_init": 0.62,
                "target_image": {"height": 16, "width": 16},
                "gaussians": {
                    "n_gaussians": 1,
                    "sh_degree": 3,
                    "upsampler": {"upsample_factor": 1},
                    "range_setting": {"far": 8, "near": 0, "type": "linear_depth"},
                },
                "gs_per_pixel_sqrt": 1,
            },
            "training": {},
        }
    )
    config.model.update(model)
    if camera_name is not None:
        config.dataset = edict({"name": "tartanair", "camera_name": camera_name})
    return config


def build_model(config):
    pytest.importorskip("xformers", reason="erayzer.py imports xformers for its attention")
    from erayzer_core.model.erayzer import ERayZer

    return ERayZer(config)


# 1. the geometry of the mask and the order it is written in


def test_the_mask_holds_exactly_the_pixels_outside_the_image_circle():
    """Counted again pixel by pixel, without rearrange and without the circle of the helper."""
    hh = ww = 4
    ph = pw = 8
    h, w = hh * ph, ww * pw
    mask = corner_gaussian_mask(hh, ww, ph, pw)

    assert mask.shape == (h * w,) and mask.dtype == torch.bool
    expected = sum(pixel_is_corner(y, x, h, w) for y in range(h) for x in range(w))
    assert int(mask.sum()) == expected
    # a disc of radius w / 2 in a square of side w leaves 1 - pi / 4 of the pixels outside
    assert mask.float().mean().item() == pytest.approx(1 - math.pi / 4, abs=0.01)


def test_the_corners_are_masked_and_the_centre_is_not():
    """The four image corners are black, the middle of the image never is."""
    hh = ww = 4
    ph = pw = 8
    h, w = hh * ph, ww * pw
    mask = corner_gaussian_mask(hh, ww, ph, pw).reshape(hh, ww, ph, pw)

    for y, x in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        assert mask[y // ph, x // pw, y % ph, x % pw], f"pixel ({y}, {x}) should be a corner"
    for y, x in [(h // 2, w // 2), (h // 2 - 1, w // 2 - 1), (h // 2, 0), (0, w // 2)]:
        assert not mask[y // ph, x // pw, y % ph, x % pw], f"pixel ({y}, {x}) is inside the circle"


def test_the_mask_is_written_in_the_gaussian_order_of_the_forward_pass():
    """A 4 by 4 image in 2 by 2 patches, every index worked out by hand.

    The Gaussians are laid out as 'b (v hh ww ph pw) c', so for ph = pw = 2 and hh = ww = 2 the pixel
    (y, x) sits at index ((y // 2 * 2 + x // 2) * 2 + y % 2) * 2 + x % 2, which gives this table:

        row 0:  0  1  4  5
        row 1:  2  3  6  7
        row 2:  8  9 12 13
        row 3: 10 11 14 15

    With a radius of 2 around (2, 2) only the four image corners lie outside the circle, and by the
    table they are the Gaussians 0, 5, 10 and 15.
    """
    order = torch.arange(16).reshape(2, 2, 2, 2).permute(0, 2, 1, 3).reshape(4, 4)
    assert order.tolist() == [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]

    mask = corner_gaussian_mask(2, 2, 2, 2)
    assert mask.nonzero().flatten().tolist() == [0, 5, 10, 15]

    # the same statement once more, this time by marking a single pixel and following it through
    for y, x in [(0, 3), (2, 1), (3, 3)]:
        image = torch.zeros(4, 4, dtype=torch.bool)
        image[y, x] = True
        flat = image.reshape(2, 2, 2, 2).permute(0, 2, 1, 3).reshape(16)
        assert int(flat.nonzero()) == int(order[y, x])


# 2. what the masking does to the opacity logits


def test_only_the_corner_logits_change_and_only_to_the_constant():
    """Every corner Gaussian of every input view is set, everything else stays bit for bit."""
    hh = ww = ph = pw = 2
    v_input, b = 3, 2
    mask = corner_gaussian_mask(hh, ww, ph, pw)
    opacity = torch.randn(b, v_input * mask.numel(), 1)

    masked = mask_corner_opacity(opacity, mask, v_input)

    tiled = mask.repeat(v_input)[None, :, None].expand_as(opacity)
    assert (masked[tiled] == CORNER_OPACITY_LOGIT).all()
    assert torch.equal(masked[~tiled], opacity[~tiled])
    assert int(tiled.sum()) == v_input * b * int(mask.sum())
    assert CORNER_OPACITY_LOGIT == -20.0
    assert torch.sigmoid(torch.tensor(CORNER_OPACITY_LOGIT)).item() < 1e-8


def test_the_masked_gaussians_get_no_gradient_and_the_others_keep_theirs():
    """masked_fill with a constant mask is the identity on the gradient of the untouched entries."""
    hh = ww = ph = pw = 2
    v_input = 2
    mask = corner_gaussian_mask(hh, ww, ph, pw)
    opacity = torch.randn(1, v_input * mask.numel(), 1, requires_grad=True)
    weight = torch.randn_like(opacity)

    (mask_corner_opacity(opacity, mask, v_input) * weight).sum().backward()

    tiled = mask.repeat(v_input)[None, :, None].expand_as(opacity)
    assert torch.equal(opacity.grad, torch.where(tiled, torch.zeros_like(weight), weight))


def test_the_masking_follows_the_device_of_the_opacity():
    """The buffer could live anywhere, the result has to stay on the tensor it was given."""
    mask = corner_gaussian_mask(2, 2, 2, 2)
    opacity = torch.zeros(1, 16, 1)
    assert mask_corner_opacity(opacity, mask, 1).device == opacity.device


# 3. the default, and that an explicit setting always wins


@pytest.mark.parametrize(
    "camera_name, camera_model, expected",
    [
        ("lcam_fish", "fisheye", True),
        ("lcam_fish", "ucm", True),
        ("lcam_fish", "pinhole", False),
        ("lcam_front", "pinhole", False),
        ("lcam_front", "fisheye", False),
        (None, "fisheye", False),
    ],
)
def test_the_default_is_on_only_for_wide_angle_fisheye_data(camera_name, camera_model, expected):
    """Fisheye data has black corners, lcam_front does not, and pinhole supervises every pixel."""
    assert corner_masking_default(tiny_model_config(camera_model, camera_name)) is expected


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("camera_name, camera_model", [("lcam_fish", "ucm"), ("lcam_front", "pinhole")])
def test_an_explicit_setting_beats_the_default_in_both_directions(explicit, camera_name, camera_model):
    config = tiny_model_config(camera_model, camera_name, mask_input_corners=explicit)
    model = build_model(config)
    assert model.mask_input_corners is explicit


@pytest.mark.parametrize(
    "camera_name, camera_model, expected",
    [("lcam_fish", "ucm", True), ("lcam_fish", "fisheye", True), ("lcam_front", "pinhole", False)],
)
def test_the_model_picks_up_the_default_without_any_config_entry(camera_name, camera_model, expected):
    model = build_model(tiny_model_config(camera_model, camera_name))
    assert model.mask_input_corners is expected
    assert torch.equal(model.corner_mask, corner_gaussian_mask(model.hh, model.ww, model.ph, model.pw))
    assert "corner_mask" not in model.state_dict(), "a new buffer would break loading old checkpoints"


def test_the_forward_pass_masks_right_after_the_gaussians_are_built():
    """Applied before xyz, the renderer, the video renderer and the opacity statistics see it."""
    from erayzer_core.model.erayzer import ERayZer

    source = inspect.getsource(ERayZer.forward)
    assert "mask_corner_opacity(opacity, self.corner_mask, v_input)" in source
    assert source.index("to_gs(") < source.index("mask_corner_opacity") < source.index("gaussian_attrs")
    assert source.index("mask_corner_opacity") < source.index("self.renderer(")


# 4. the pinhole training stays exactly as it was


@pytest.mark.parametrize("camera_name", ["lcam_fish", "lcam_front", None])
def test_pinhole_is_never_masked_by_default(camera_name):
    """A pinhole loss has no mask, so its corners are supervised and must keep their Gaussians."""
    config = tiny_model_config("pinhole", camera_name)
    assert corner_masking_default(config) is False
    assert build_model(config).mask_input_corners is False
