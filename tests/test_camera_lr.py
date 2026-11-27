"""A learning rate of its own for the camera heads, so the focal length stops drifting.

With rays built in PyTorch the focal length and the UCM alpha get a gradient from the rendering and
run to the right value within the first few thousand steps. Over a long run they then drift away
again: 20000 fisheye steps reached 176 degrees around the middle and ended at 99 degrees, because the
self supervised target rewards a camera that is too narrow once the Gaussians are free to follow it.
The cure is to let the camera heads learn early and freeze them afterwards, so they need their own
parameter group with its own schedule, without changing anything for the other parameters.

These tests pin that the groups hold exactly the camera head parameters, that the schedule follows
camera_lr(t) = lr(t) * camera_lr_scale * decay(t), and that the defaults are bit for bit the old run.

Run from the repository root:
    pytest tests/test_camera_lr.py
"""

import inspect
import math
import os
import sys

import pytest
import torch
import torch.nn as nn
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train

LR = 1e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.05


def tiny_model_config(camera_model):
    """The smallest ERayZer the constructor accepts, a few thousand parameters on the CPU."""
    return edict(
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


class DummyPosePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # the shape of the real PoseEstimator heads, the last linear without a bias
        self.rel_head = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 9))
        self.canonical_k_head = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 1, bias=False))


class DummyModel(nn.Module):
    """A stand in for ERayZer with both camera heads and parameters of every other kind."""

    def __init__(self):
        super().__init__()
        self.pose_predictor = DummyPosePredictor()
        self.alpha_head = nn.Linear(8, 1)
        self.backbone = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
        self.camera_token = nn.Parameter(torch.zeros(1, 1, 8))
        self.scale = nn.Parameter(torch.zeros(8))


class DDPLike(nn.Module):
    """DistributedDataParallel keeps the model under .module, so every name gains that prefix."""

    def __init__(self, module):
        super().__init__()
        self.module = module


def camera_parameter_names(model):
    return [name for name, _ in model.named_parameters() if train.is_camera_parameter(name)]


def group_names(optimizer):
    return [group["name"] for group in optimizer.param_groups]


def group_by_name(optimizer, name):
    return next(group for group in optimizer.param_groups if group["name"] == name)


def lr_sequence(scheduler, optimizer, steps):
    """The learning rate of every group over the given number of scheduler steps, before each step."""
    history = []
    for _ in range(steps):
        history.append({group["name"]: group["lr"] for group in optimizer.param_groups})
        scheduler.step()
    return history


def make_optimizer(model=None):
    return train.configure_optimizer(model or DummyModel(), WEIGHT_DECAY, LR, BETAS)


def old_lr_lambda(current_step, total_train_steps, warmup_steps):
    """The schedule as it was before the camera groups existed, for an independent comparison."""
    if warmup_steps > 0 and current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


# 1. the names the partition matches are the names the model really gives its camera heads


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye", "ucm"])
def test_the_camera_heads_carry_the_names_the_partition_looks_for(camera_model):
    """Both heads sit where erayzer.py puts them, so matching the names catches exactly them."""
    pytest.importorskip("xformers", reason="erayzer.py imports xformers for its attention")
    from erayzer_core.model.erayzer import ERayZer

    model = ERayZer(tiny_model_config(camera_model))
    expected = [
        "pose_predictor.canonical_k_head.0.weight",
        "pose_predictor.canonical_k_head.0.bias",
        "pose_predictor.canonical_k_head.2.weight",
    ]
    if camera_model == "ucm":
        # alpha_head only exists for UCM, it predicts the one alpha of the scene
        expected += ["alpha_head.weight", "alpha_head.bias"]
    assert camera_parameter_names(model) == expected
    # and the heads are a small part of the model, the partition must not swallow anything else
    assert len(expected) < len(list(model.named_parameters()))


def test_the_dummy_model_matches_the_real_parameter_names():
    """Guard that the dummy the other tests use really has the names of the real heads."""
    pytest.importorskip("xformers", reason="erayzer.py imports xformers for its attention")
    from erayzer_core.model.erayzer import ERayZer

    real = ERayZer(tiny_model_config("ucm"))
    assert camera_parameter_names(DummyModel()) == camera_parameter_names(real)


# 2. the partition itself


def test_the_camera_groups_hold_exactly_the_camera_head_parameters():
    model = DummyModel()
    optimizer = make_optimizer(model)
    assert group_names(optimizer) == ["decay", "nodecay", "camera_decay", "camera_nodecay"]
    # train/lr is logged from param_groups[0], which has to stay the main decay group
    assert optimizer.param_groups[0]["name"] == "decay"
    assert optimizer.param_groups[0]["weight_decay"] == WEIGHT_DECAY

    camera_ids = {id(param) for name, param in model.named_parameters() if train.is_camera_parameter(name)}
    in_camera_groups = {
        id(param)
        for group in optimizer.param_groups
        if group["name"].startswith("camera")
        for param in group["params"]
    }
    assert in_camera_groups == camera_ids
    assert len(camera_ids) == 5  # three for canonical_k_head, two for alpha_head


def test_the_groups_are_a_partition_of_the_trainable_parameters():
    model = DummyModel()
    optimizer = make_optimizer(model)
    all_ids = [id(param) for param in model.parameters() if param.requires_grad]
    in_groups = [id(param) for group in optimizer.param_groups for param in group["params"]]
    assert sorted(in_groups) == sorted(all_ids), "every parameter exactly once"
    assert len(set(in_groups)) == len(in_groups), "the groups are disjoint"


def test_weight_decay_still_follows_the_dimension_in_every_group():
    optimizer = make_optimizer()
    for group in optimizer.param_groups:
        expected = WEIGHT_DECAY if group["name"].endswith("decay") and "nodecay" not in group["name"] else 0.0
        assert group["weight_decay"] == expected
        for param in group["params"]:
            assert (param.dim() >= 2) == (group["weight_decay"] == WEIGHT_DECAY)


def test_a_model_without_the_ucm_head_gets_no_empty_group():
    """Fused AdamW rejects empty groups, and a pinhole run simply has no alpha_head."""
    model = DummyModel()
    del model.alpha_head
    optimizer = make_optimizer(model)
    assert group_names(optimizer) == ["decay", "nodecay", "camera_decay", "camera_nodecay"]
    assert all(len(group["params"]) > 0 for group in optimizer.param_groups)

    # a model without any camera head at all keeps the two groups it always had
    plain = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
    plain_optimizer = train.configure_optimizer(plain, WEIGHT_DECAY, LR, BETAS)
    assert group_names(plain_optimizer) == ["decay", "nodecay"]


# 3. the DDP prefix


def test_the_ddp_module_prefix_does_not_break_the_partition():
    """configure_optimizer is called with the wrapped model, so the names start with module."""
    inner = DummyModel()
    wrapped = DDPLike(inner)
    assert all(name.startswith("module.") for name, _ in wrapped.named_parameters())
    plain_optimizer = make_optimizer(inner)
    wrapped_optimizer = make_optimizer(wrapped)
    assert group_names(wrapped_optimizer) == group_names(plain_optimizer)
    for a, b in zip(plain_optimizer.param_groups, wrapped_optimizer.param_groups):
        assert [id(p) for p in a["params"]] == [id(p) for p in b["params"]]


# 4. the defaults are the old behaviour, bit for bit


@pytest.mark.parametrize("scheduler_type", ["cosine", "linear", "constant"])
def test_the_defaults_give_every_group_the_very_same_learning_rate(scheduler_type):
    optimizer = make_optimizer()
    scheduler = train.configure_lr_scheduler(optimizer, 200, 20, scheduler_type)
    for step, lrs in enumerate(lr_sequence(scheduler, optimizer, 200)):
        values = set(lrs.values())
        assert len(values) == 1, f"step {step}: {lrs}"


def test_the_defaults_reproduce_the_schedule_from_before_the_camera_groups():
    """Compared against a copy of the old lambda, not against the new code."""
    optimizer = make_optimizer()
    scheduler = train.configure_lr_scheduler(optimizer, 200, 20, "cosine")
    for step, lrs in enumerate(lr_sequence(scheduler, optimizer, 200)):
        expected = LR * old_lr_lambda(step, 200, 20)
        for name, value in lrs.items():
            assert value == expected, f"step {step} group {name}: {value} vs {expected}"


# 5. the schedule follows the plan


def test_the_scale_multiplies_the_camera_learning_rate_everywhere():
    optimizer = make_optimizer()
    scheduler = train.configure_lr_scheduler(optimizer, 200, 20, "cosine", camera_lr_scale=0.5)
    for step, lrs in enumerate(lr_sequence(scheduler, optimizer, 200)):
        main = lrs["decay"]
        assert main == LR * old_lr_lambda(step, 200, 20), f"step {step}: the main groups must not change"
        for name in ["camera_decay", "camera_nodecay"]:
            assert lrs[name] == pytest.approx(0.5 * main, rel=1e-12), f"step {step} {name}"


def test_the_camera_learning_rate_decays_to_exactly_zero_and_stays_there():
    decay_updates = 40
    optimizer = make_optimizer()
    scheduler = train.configure_lr_scheduler(
        optimizer, 200, 0, "constant", camera_lr_scale=2.0, camera_lr_decay_updates=decay_updates
    )
    history = lr_sequence(scheduler, optimizer, 200)
    factors = [lrs["camera_decay"] / lrs["decay"] for lrs in history]

    assert factors[0] == pytest.approx(2.0, rel=1e-12), "it starts at the scale"
    assert factors[decay_updates // 2] == pytest.approx(2.0 * 0.5, rel=1e-12), "cosine, half way down at the middle"
    for step in range(decay_updates, 200):
        assert history[step]["camera_decay"] == 0.0, f"step {step} is not exactly frozen"
        assert history[step]["camera_nodecay"] == 0.0
    for step in range(1, decay_updates):
        assert factors[step] < factors[step - 1], f"step {step} is not falling"
    # the main groups keep their constant schedule throughout
    assert all(lrs["decay"] == LR for lrs in history)


def test_the_decay_rides_on_top_of_the_warmup_of_the_main_schedule():
    """Multiplying keeps the warmup, so the camera heads do not start at the full rate either."""
    optimizer = make_optimizer()
    scheduler = train.configure_lr_scheduler(
        optimizer, 200, 20, "cosine", camera_lr_scale=1.0, camera_lr_decay_updates=100
    )
    history = lr_sequence(scheduler, optimizer, 30)
    assert history[0]["camera_decay"] == 0.0 and history[0]["decay"] == 0.0, "warmup starts at zero"
    for step in range(1, 20):
        base = LR * old_lr_lambda(step, 200, 20)
        decay = 0.5 * (1.0 + math.cos(math.pi * step / 100))
        assert history[step]["decay"] == base
        assert history[step]["camera_decay"] == pytest.approx(base * decay, rel=1e-12)
        assert history[step]["camera_decay"] < history[step]["decay"]


def test_a_decay_of_zero_updates_turns_the_decay_off():
    optimizer = make_optimizer()
    scheduler = train.configure_lr_scheduler(optimizer, 200, 0, "cosine", camera_lr_decay_updates=0)
    for lrs in lr_sequence(scheduler, optimizer, 50):
        assert lrs["camera_decay"] == lrs["decay"]


# 6. the unit of camera_lr_decay_steps


@pytest.mark.parametrize(
    "fwdbwd_passes,grad_accum,expected",
    [(3000, 4, 750), (6000, 4, 1500), (3001, 4, 751), (0, 4, 0), (20000, 1, 20000), (10, 3, 4)],
)
def test_forward_backward_passes_become_parameter_updates(fwdbwd_passes, grad_accum, expected):
    """camera_lr_decay_steps counts passes like max_fwdbwd_passes, the scheduler counts updates."""
    assert train.fwdbwd_passes_to_param_updates(fwdbwd_passes, grad_accum) == expected


def test_the_scheduler_also_ticks_for_a_skipped_optimizer_step():
    """train.py calls lr_scheduler.step() outside the skip branch, so the unit is the attempted update.

    allowed_gradnorm_factor skips a fair share of the steps of a fine tuning run. If the scheduler
    only moved on a real update, camera_lr_decay_steps would stretch over more passes than asked for.
    """
    source = inspect.getsource(train.main)
    body = source[source.index("if not skip_optimizer_step") :]
    tick = body.index("lr_scheduler.step()")
    update = body.index("param_update_step += 1")
    indent = body[:tick].rsplit("\n", 1)[-1]
    update_indent = body[:update].rsplit("\n", 1)[-1]
    assert len(indent) < len(update_indent), "the scheduler step has to stay outside the skip branch"


@pytest.mark.parametrize("scale,decay_updates,warmup", [(1.0, 0, 500), (1.0, -5, 0)])
def test_the_default_camera_schedule_prints_nothing(scale, decay_updates, warmup):
    assert train.camera_lr_notice(scale, decay_updates, warmup) is None


def test_a_decay_shorter_than_the_warmup_is_announced_as_a_warning():
    """Otherwise the camera heads crawl through the warmup, freeze, and the log looks healthy."""
    warning = train.camera_lr_notice(1.0, 100, 500)
    assert warning is not None and warning.startswith("WARNING")
    assert "100" in warning and "500" in warning
    # the boundary counts as broken too, a decay that ends exactly at the warmup learns nothing
    assert train.camera_lr_notice(1.0, 500, 500).startswith("WARNING")
    # a decay well past the warmup is the intended setting and only reports itself
    plain = train.camera_lr_notice(1.0, 1500, 500)
    assert plain is not None and not plain.startswith("WARNING")
    assert "1500" in plain
    # a scale without a decay still says what it is
    scaled = train.camera_lr_notice(0.5, 0, 500)
    assert scaled is not None and "0.5" in scaled and not scaled.startswith("WARNING")


def test_the_default_config_keeps_the_old_behaviour():
    """A config without the new keys, and the shipped defaults, both have to be the old schedule."""
    import yaml

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    with open(os.path.join(config_path, "erayzer_train_tartanair.yaml"), "r", encoding="utf-8") as handle:
        shipped = yaml.safe_load(handle)["training"]
    assert float(shipped["camera_lr_scale"]) == 1.0
    assert int(shipped["camera_lr_decay_steps"]) == 0

    for training in [edict({}), edict(shipped)]:
        optimizer = make_optimizer()
        scheduler = train.configure_lr_scheduler(
            optimizer,
            200,
            20,
            "cosine",
            camera_lr_scale=float(training.get("camera_lr_scale", 1.0)),
            camera_lr_decay_updates=train.fwdbwd_passes_to_param_updates(
                int(training.get("camera_lr_decay_steps", 0)), 4
            ),
        )
        for step, lrs in enumerate(lr_sequence(scheduler, optimizer, 200)):
            assert set(lrs.values()) == {LR * old_lr_lambda(step, 200, 20)}, f"step {step}"


# 7. resume


def test_a_resume_in_the_middle_continues_the_same_schedule():
    """Saving and loading optimizer and scheduler has to give the same learning rates as no break."""
    settings = dict(camera_lr_scale=0.5, camera_lr_decay_updates=60)
    optimizer = make_optimizer()
    scheduler = train.configure_lr_scheduler(optimizer, 200, 20, "cosine", **settings)
    uninterrupted = lr_sequence(scheduler, optimizer, 100)

    optimizer = make_optimizer()
    scheduler = train.configure_lr_scheduler(optimizer, 200, 20, "cosine", **settings)
    first_half = lr_sequence(scheduler, optimizer, 40)
    state = {"optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict()}

    resumed_optimizer = make_optimizer()
    resumed_scheduler = train.configure_lr_scheduler(resumed_optimizer, 200, 20, "cosine", **settings)
    resumed_optimizer.load_state_dict(state["optimizer"])
    resumed_scheduler.load_state_dict(state["lr_scheduler"])
    second_half = lr_sequence(resumed_scheduler, resumed_optimizer, 60)

    assert first_half + second_half == uninterrupted
    assert group_names(resumed_optimizer) == group_names(optimizer)


# the logging the slurm log and wandb show


def test_the_logged_camera_learning_rate_is_the_one_of_the_camera_group():
    """train.py picks the first group whose name starts with camera, and falls back to the main one."""
    optimizer = make_optimizer()
    train.configure_lr_scheduler(optimizer, 200, 0, "cosine", camera_lr_scale=0.25)

    def logged(opt):
        return next(
            (g["lr"] for g in opt.param_groups if str(g.get("name", "")).startswith("camera")),
            opt.param_groups[0]["lr"],
        )

    assert logged(optimizer) == group_by_name(optimizer, "camera_decay")["lr"]
    assert logged(optimizer) == pytest.approx(0.25 * optimizer.param_groups[0]["lr"], rel=1e-12)

    plain = train.configure_optimizer(nn.Linear(8, 8), WEIGHT_DECAY, LR, BETAS)
    assert logged(plain) == plain.param_groups[0]["lr"]
