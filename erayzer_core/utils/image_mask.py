"""Validity masks for images with black, empty borders (e.g. the corners of a
circular fisheye image such as TartanAir lcam_fish).

A pixel is invalid if it is black (every channel <= threshold) and connected to
the image border through black pixels. Black content inside the image (a dark
object, a shadow) therefore stays valid; only the empty region around the image
circle is masked. The valid region can additionally be shrunk by a margin to
drop the blurred rim that resizing leaves at the edge of the circle.

The masks are used by ERayZer (config.model.black_border_mask):
* patches without any valid pixel are not attended to (key mask),
* Gaussians of invalid pixels are made invisible,
* the photometric loss ignores invalid target pixels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

# Opacity logit for the Gaussians of invalid pixels. Equals the lower clamp in
# GaussiansUpsampler.to_gs: sigmoid(-10) = 4.5e-5 < 1/255, so the rasteriser
# skips them and they receive no gradient.
INVALID_OPACITY_LOGIT = -10.0


@torch.no_grad()
def black_border_mask(images: Tensor, threshold: float = 0.01, margin: int = 1) -> Tensor:
    """Valid-pixel mask of images whose border region is empty (black).

    Args:
        images: [..., C, H, W] in [0, 1].
        threshold: a pixel is black if all its channels are <= threshold.
        margin: number of pixels by which the valid region is shrunk next to
            the masked border region (0 disables).

    Returns:
        [..., 1, H, W] bool, True for valid pixels.
    """
    *batch, _, H, W = images.shape
    x = images.reshape(-1, images.shape[-3], H, W)
    black = (x <= threshold).all(dim=1, keepdim=True)  # [N, 1, H, W]

    border = torch.zeros_like(black)
    border[..., 0, :] = True
    border[..., -1, :] = True
    border[..., :, 0] = True
    border[..., :, -1] = True
    reach = black & border
    if not bool(reach.any()):
        return torch.ones(*batch, 1, H, W, dtype=torch.bool, device=images.device)

    # Flood fill (8-neighbourhood) from the black border pixels through black
    # pixels: grow by one pixel per step, check for convergence every few steps.
    blackf = black.to(torch.float32)
    reachf = reach.to(torch.float32)
    for it in range(H * W):
        reachf = F.max_pool2d(reachf, 3, stride=1, padding=1) * blackf
        if it % 8 == 7:
            grown = reachf.sum()
            if it > 7 and bool(grown == last):
                break
            last = grown
    invalid = reachf > 0

    if margin > 0 and bool(invalid.any()):
        k = 2 * margin + 1
        invalid = F.max_pool2d(invalid.to(torch.float32), k, stride=1, padding=margin) > 0
    return (~invalid).reshape(*batch, 1, H, W)


def patch_valid_mask(pixel_valid: Tensor, patch_size: int) -> Tensor:
    """Per-patch validity: a patch is valid if it contains at least one valid pixel.

    pixel_valid: [b, v, 1, H, W] bool. Returns [b, v, n] bool with
    n = (H / patch_size) * (W / patch_size), in the token order of the image
    tokenizer ("(hh ww)").
    """
    p = patch_size
    return rearrange(pixel_valid, "b v 1 (hh ph) (ww pw) -> b v (hh ww) (ph pw)", ph=p, pw=p).any(dim=-1)


def gaussian_valid_mask(pixel_valid: Tensor, patch_size: int) -> Tensor:
    """Per-Gaussian validity of pixel-aligned Gaussians (one per pixel).

    pixel_valid: [b, v, 1, H, W] bool. Returns [b, v * H * W, 1] bool in the
    order of ERayZer's Gaussians ("b (v hh ww ph pw) c").
    """
    p = patch_size
    return rearrange(pixel_valid, "b v c (hh ph) (ww pw) -> b (v hh ww ph pw) c", ph=p, pw=p)


def attention_key_mask(token_valid: Tensor) -> Tensor:
    """Make sure every attention row has at least one key: rows (sequences)
    without any valid token are fully unmasked instead (softmax over an empty
    set is undefined). token_valid: [B, L] bool."""
    return token_valid | ~token_valid.any(dim=-1, keepdim=True)
