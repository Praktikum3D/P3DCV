"""Masking of the empty black border of images (config.model.black_border_mask).

* black_border_mask: only black pixels connected to the image border are
  invalid (black content inside the image stays valid), margin, fast path;
* patch / Gaussian masks use the token and Gaussian order of ERayZer;
* attention with a key mask equals attention over the valid keys only;
* a small ERayZer on the CPU: the content of masked patches has no influence
  on anything else, Gaussians of invalid pixels are invisible and get no
  gradient, and the masking is off (outputs unchanged) without the config;
* train.compute_loss ignores invalid target pixels.

Runs on the CPU without xformers / gsplat: missing packages are stubbed, the
xformers attention by torch's scaled_dot_product_attention and the Gaussian
renderer by a differentiable stand-in.
"""

import importlib
import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stub(name, attrs=()):
    try:
        importlib.import_module(name)
    except Exception:
        module = types.ModuleType(name)
        for a in attrs:
            setattr(module, a, None)
        sys.modules[name] = module


def _sdpa_memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, op=None):
    assert attn_bias is None
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=p)
    return out.transpose(1, 2)


def _setup_imports():
    for name, attrs in [("gsplat", ["rasterization"]), ("cv2", []), ("videoio", []),
                        ("plyfile", ["PlyData", "PlyElement"]), ("matplotlib", [])]:
        _stub(name, attrs)
    _stub("xformers")
    _stub("xformers.ops")
    xops = sys.modules["xformers.ops"]
    if not hasattr(xops, "memory_efficient_attention") or xops.memory_efficient_attention is None:
        # CPU stand-in for the flash attention path (the masked path uses SDPA itself)
        xops.memory_efficient_attention = _sdpa_memory_efficient_attention
        flash = types.SimpleNamespace(FwOp=None, BwOp=None)
        xops.fmha = types.SimpleNamespace(flash=flash)
    if not hasattr(sys.modules["xformers"], "ops"):
        sys.modules["xformers"].ops = xops
    import erayzer_core  # noqa: F401

    return (
        importlib.import_module("erayzer_core.utils.image_mask"),
        importlib.import_module("erayzer_core.model.erayzer"),
        importlib.import_module("erayzer_core.model.utils_transformer"),
    )


image_mask, erayzer, utils_transformer = _setup_imports()


def fisheye_like(n=2, size=64, radius=None, seed=0):
    """Bright random images [n, 3, H, W] with a black outside of an inscribed circle."""
    g = torch.Generator().manual_seed(seed)
    img = 0.2 + 0.8 * torch.rand(n, 3, size, size, generator=g)
    yy, xx = torch.meshgrid(torch.arange(size) + 0.5, torch.arange(size) + 0.5, indexing="ij")
    r = radius if radius is not None else size / 2
    inside = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= r * r
    return img * inside, inside


# ----------------------------------------------------------------------------- masks


def test_black_border_is_masked_black_content_is_not():
    img, inside = fisheye_like()
    img[:, :, 28:36, 28:36] = 0.0  # a black object in the middle of the image circle
    valid = image_mask.black_border_mask(img, threshold=0.01, margin=0)
    assert valid.shape == (2, 1, 64, 64) and valid.dtype == torch.bool
    assert torch.equal(valid[:, 0], inside.expand(2, -1, -1))  # exactly the circle, blob included


def test_margin_shrinks_valid_region_next_to_border_only():
    img, inside = fisheye_like()
    img[:, :, 28:36, 28:36] = 0.0
    valid = image_mask.black_border_mask(img, threshold=0.01, margin=1)
    expected = ~(F.max_pool2d((~inside).float()[None, None], 3, stride=1, padding=1)[0, 0] > 0)
    assert torch.equal(valid[0, 0], expected)
    assert bool(valid[0, 0, 28:36, 28:36].all())  # the black object is untouched


def test_no_black_border_is_all_valid_and_batch_shapes():
    img = torch.rand(2, 3, 3, 32, 32) * 0.9 + 0.1
    img[..., 10:20, 10:20] = 0.0  # black, but not connected to the border
    valid = image_mask.black_border_mask(img)
    assert valid.shape == (2, 3, 1, 32, 32) and bool(valid.all())


def test_flood_fill_follows_long_thin_black_regions():
    # a black spiral-like corridor from the border deep into the image
    img = torch.ones(1, 3, 40, 40)
    img[..., 0, :] = 0  # top row black
    for c in range(2, 38, 4):
        img[..., 0:38, c] = 0  # columns hanging down from the top row
    valid = image_mask.black_border_mask(img, margin=0)
    assert not bool(valid[0, 0, 37, 2]) and not bool(valid[0, 0, 37, 34])
    assert bool(valid[0, 0, 39, 2])  # bright bottom row stays valid


def test_patch_and_gaussian_masks_follow_model_order():
    b, v, p, H, W = 2, 3, 4, 8, 12
    pix = torch.rand(b, v, 1, H, W) > 0.3
    pix[0, 0, :, :4, :4] = False  # one fully invalid patch
    patch = image_mask.patch_valid_mask(pix, p)
    # token order of ERayZer.image_tokenizer: "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)"
    tok = rearrange(pix.float(), "b v c (hh ph) (ww pw) -> b v (hh ww) (ph pw c)", ph=p, pw=p).amax(-1) > 0
    assert torch.equal(patch, tok) and not bool(patch[0, 0, 0])
    gs = image_mask.gaussian_valid_mask(pix, p)
    # inverse of the model's "b (v hh ww ph pw) c -> b v c (hh ph) (ww pw)"
    back = rearrange(gs, "b (v hh ww ph pw) c -> b v c (hh ph) (ww pw)", v=v, hh=H // p, ww=W // p, ph=p, pw=p)
    assert torch.equal(back, pix)


def test_attention_key_mask_unmasks_empty_rows():
    m = torch.tensor([[True, False, False], [False, False, False]])
    assert torch.equal(image_mask.attention_key_mask(m), torch.tensor([[True, False, False], [True, True, True]]))


# ----------------------------------------------------------------------------- attention


def test_masked_attention_equals_attention_over_valid_keys():
    torch.manual_seed(0)
    attn = utils_transformer.QK_Norm_SelfAttention(32, 8, use_qk_norm=True).eval()
    x = torch.randn(2, 10, 32)
    key_mask = torch.ones(2, 10, dtype=torch.bool)
    key_mask[0, [2, 5, 7]] = False
    out = attn(x, key_mask=key_mask)
    # reference: run each sequence with only its valid tokens as keys / values
    q, k, v = attn.to_qkv(x).split(32, dim=2)
    q, k, v = (rearrange(t, "b l (nh dh) -> b l nh dh", dh=8) for t in (q, k, v))
    q, k = attn.q_norm(q), attn.k_norm(k)
    for i in range(2):
        keep = key_mask[i]
        w = torch.einsum("qhd,khd->hqk", q[i], k[i][keep]) / math.sqrt(8)
        ref = torch.einsum("hqk,khd->qhd", w.softmax(-1), v[i][keep]).reshape(10, 32)
        torch.testing.assert_close(out[i], attn.fc(ref), rtol=1e-5, atol=1e-5)


# ----------------------------------------------------------------------------- model


def _tiny_config(enabled):
    return edict(
        inference=False,
        model=edict(
            image_tokenizer=edict(image_size=32, patch_size=8, in_channels=3),
            input_with_pe=True,
            target_image=edict(height=32, width=32),
            transformer=edict(dict(d=64, d_head=16, encoder_n_layer=4, encoder_geom_n_layer=4, special_init=True,
                                   depth_init=True, use_qk_norm=True)),
            pose_latent=edict(length=1, canonical="middle", mode="pairwise"),
            gaussians=edict(sh_degree=0, range_setting=edict(far=8, near=0, type="linear_depth"),
                            upsampler=edict(upsample_factor=1)),
            gs_per_pixel_sqrt=1,
            hard_pixelalign=True,
            black_border_mask=edict(enabled=enabled, threshold=0.01, margin=1),
        ),
        training=edict(num_views=10, num_input_views=5, num_target_views=5, random_split=False,
                       grad_checkpoint_every=1),
    )


class FakeRenderer(torch.nn.Module):
    """Differentiable stand-in for the gsplat renderer: records the Gaussians and
    returns an image that depends on every Gaussian's opacity and colour."""

    def __init__(self, real):
        super().__init__()
        self.gaussians_model = real.gaussians_model
        self.last = None

    def forward(self, xyz, features, scaling, rotation, opacity, height, width, C2W, fxfycxcy, deferred=True):
        self.last = edict(xyz=xyz, features=features, opacity=opacity)
        b, v = C2W.shape[:2]
        val = (torch.sigmoid(opacity) * features[:, :, 0, :]).mean(dim=1)  # [b, 3]
        return edict(render=val[:, None, :, None, None].expand(b, v, 3, height, width))


def _model(enabled, seed=0):
    torch.manual_seed(seed)
    model = erayzer.ERayZer(_tiny_config(enabled)).eval()
    model.renderer = FakeRenderer(model.renderer)
    return model


def _data(seed=0):
    img, _ = fisheye_like(n=10, size=32, radius=15.0, seed=seed)
    return {"image": img[None]}  # [1, 10, 3, 32, 32]


def test_model_masks_tokens_gaussians_and_reports_masks():
    model = _model(enabled=True)
    data = _data()
    res = model(data)
    valid = image_mask.black_border_mask(data["image"], 0.01, 1)
    assert res.input_mask is not None and res.target_mask is not None
    torch.testing.assert_close(res.target_mask, valid[:, res.target_idx[0]])
    gs_valid = image_mask.gaussian_valid_mask(valid[:, res.input_idx[0]], 8)
    op = model.renderer.last.opacity
    assert bool((op[~gs_valid.expand_as(op)] == image_mask.INVALID_OPACITY_LOGIT).all())
    assert torch.sigmoid(torch.tensor(image_mask.INVALID_OPACITY_LOGIT)) < 1 / 255  # skipped by the rasteriser
    assert bool((op[gs_valid.expand_as(op)] != image_mask.INVALID_OPACITY_LOGIT).all())


def test_content_of_masked_patches_has_no_influence():
    """With an explicit mask, changing the pixels of fully masked patches changes
    neither the cameras nor any valid Gaussian."""
    model = _model(enabled=False)  # the mask comes from data["mask"]
    data = _data()
    mask = torch.ones(1, 10, 1, 32, 32, dtype=torch.bool)
    mask[..., :8, :16] = False  # two whole patches in every view
    mask[:, 3, :, 16:, 16:] = False  # a quarter of view 3
    a = model({"image": data["image"], "mask": mask})
    ra = model.renderer.last
    img2 = data["image"].clone()
    img2[~mask.expand_as(img2)] = torch.rand(int((~mask).sum()) * 3)
    b = model({"image": img2, "mask": mask})
    rb = model.renderer.last
    torch.testing.assert_close(a.c2w, b.c2w, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a.fxfycxcy, b.fxfycxcy, rtol=1e-5, atol=1e-5)
    gs_valid = image_mask.gaussian_valid_mask(mask[:, a.input_idx[0]], 8)[0, :, 0]
    torch.testing.assert_close(ra.xyz[0, gs_valid], rb.xyz[0, gs_valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(ra.features[0, gs_valid], rb.features[0, gs_valid], rtol=1e-4, atol=1e-4)
    # and without masking the same change does matter (the test is sensitive)
    c = model({"image": img2})
    assert not torch.allclose(a.c2w, c.c2w, atol=1e-5)


def test_invalid_gaussians_get_no_gradient():
    """A loss on the Gaussians of invalid pixels does not reach the network."""
    model = _model(enabled=True).train()
    data = _data()

    def param_grads(select_valid):
        model.zero_grad(set_to_none=True)
        res = model(data)
        last = model.renderer.last
        valid = image_mask.black_border_mask(data["image"], 0.01, 1)
        gs_valid = image_mask.gaussian_valid_mask(valid[:, res.input_idx[0]], 8)[..., 0]  # [b, N]
        sel = gs_valid if select_valid else ~gs_valid
        assert int(sel.sum()) > 0
        loss = torch.sigmoid(last.opacity[..., 0][sel]).sum()
        loss.backward()
        return [p.grad for p in model.parameters() if p.grad is not None and bool((p.grad != 0).any())]

    assert param_grads(select_valid=False) == []
    assert len(param_grads(select_valid=True)) > 0  # sanity: valid Gaussians do train the network


def test_disabled_config_changes_nothing():
    """Without the config switch (and without data["mask"]) the forward pass is
    the unmasked one: the result equals a run with the mask helper disabled."""
    data = _data()
    m_off = _model(enabled=False)
    m_on = _model(enabled=True)
    m_on.load_state_dict(m_off.state_dict())
    r_off = m_off(data)
    assert r_off.input_mask is None and r_off.target_mask is None
    # images without a black border: enabling the mask changes nothing either
    clean = {"image": torch.rand(1, 10, 3, 32, 32) * 0.8 + 0.2}
    torch.testing.assert_close(m_off(clean).c2w, m_on(clean).c2w, rtol=0, atol=0)


# ----------------------------------------------------------------------------- loss


def _load_compute_loss():
    """train.compute_loss on its own (train.py imports the datasets, PIL, yaml, ...)."""
    import ast
    from typing import Dict

    src = (REPO_ROOT / "train.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "compute_loss")
    ns = {"torch": torch, "F": F, "edict": edict, "Dict": Dict}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train.py", "exec"), ns)
    return types.SimpleNamespace(compute_loss=ns["compute_loss"])


def test_compute_loss_ignores_invalid_target_pixels():
    train = _load_compute_loss()
    target = torch.rand(1, 2, 3, 16, 16)
    mask = torch.ones(1, 2, 1, 16, 16, dtype=torch.bool)
    mask[..., :4, :] = False
    pred = target.clone()
    pred[..., :4, :] = 1.0 - pred[..., :4, :]  # wrong only where invalid
    cfg = edict(training=edict(l2_loss_weight=1.0, l1_loss_weight=1.0, perceptual_loss_weight=1.0))

    class L1Perceptual(torch.nn.Module):
        def forward(self, a, b):
            return (a - b).abs().mean()

    mods = edict(perceptual=L1Perceptual())
    out = train.compute_loss(edict(render=pred, target_image=target, target_mask=mask), {"image": target}, cfg, mods)
    assert float(out.loss) == 0.0
    # without the mask the same prediction is penalised
    out = train.compute_loss(edict(render=pred, target_image=target), {"image": target}, cfg, mods)
    assert float(out.loss) > 0.1
    # masked averages are over the valid pixels: error 0.5 on all valid pixels -> mse 0.25
    t = torch.zeros_like(target)
    out = train.compute_loss(edict(render=t + 0.5, target_image=t, target_mask=mask), {"image": t}, cfg, edict())
    assert abs(float(out.mse) - 0.25) < 1e-6 and abs(float(out.l1) - 0.5) < 1e-6
