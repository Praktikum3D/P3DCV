"""Holding the predicted focal length at a constant, to tell cause from symptom.

Every long run so far collapses between step 3000 and 5000, and the predicted focal length collapses
with it. A learning rate of zero for the camera heads does not decide which of the two leads, because
the head reads features from the shared transformer and that keeps learning. model.fix_focal_norm
writes the true value into the prediction instead, before it reaches any ray, so a run with a focal
length that cannot move answers the question.

These tests pin that the helper touches exactly fx and fy, that the constant carries no gradient and
leaves the other columns exactly as they were, that the UCM alpha survives, that the default changes
nothing, and that a -s option on the command line arrives as a float on the model.

Run from the repository root:
    pytest tests/test_fix_focal.py
"""

import inspect
import os
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train
from test_camera_lr import tiny_model_config

F_TRUE = 0.29383  # 1 / radians(195 / 2), the equidistant focal length of an lcam_fish image
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "erayzer_train_tartanair.yaml"
)


def erayzer_module():
    pytest.importorskip("xformers", reason="erayzer.py imports xformers for its attention")
    import erayzer_core.model.erayzer as module

    return module


def intrinsics(columns, requires_grad=False, dtype=torch.float32):
    """A [2, 3, columns] batch of predictions, four columns for fxfycxcy and five with the UCM alpha."""
    values = [0.41, 0.37, 0.52, 0.48, 0.62][:columns]
    tensor = torch.tensor(values, dtype=dtype).expand(2, 3, columns).clone()
    tensor = tensor + 0.01 * torch.arange(tensor.numel(), dtype=dtype).reshape(tensor.shape)
    return tensor.requires_grad_(requires_grad)


# 1. the helper replaces exactly fx and fy


@pytest.mark.parametrize("columns", [4, 5])
def test_only_the_two_focal_columns_change(columns):
    override_focal = erayzer_module().override_focal
    predicted = intrinsics(columns)
    fixed = override_focal(predicted, F_TRUE)

    assert fixed.shape == predicted.shape
    assert torch.equal(fixed[..., :2], torch.full_like(predicted[..., :2], F_TRUE))
    assert torch.equal(fixed[..., 2:], predicted[..., 2:]), "cx, cy and the UCM alpha stay as predicted"
    # and the prediction itself is not written over
    assert not torch.equal(predicted[..., :2], fixed[..., :2])


@pytest.mark.parametrize("value", [0.0, F_TRUE, 1.1, -0.5])
def test_the_constant_is_the_value_that_was_asked_for(value):
    override_focal = erayzer_module().override_focal
    fixed = override_focal(intrinsics(4), value)
    assert torch.equal(fixed[..., :2], torch.full((2, 3, 2), value, dtype=torch.float32))


def test_a_value_that_arrives_as_a_string_still_becomes_a_number():
    """load_config parses -s values, but a yaml entry or a caller could still hand over a string."""
    override_focal = erayzer_module().override_focal
    fixed = override_focal(intrinsics(4), str(F_TRUE))
    assert torch.equal(fixed[..., :2], torch.full((2, 3, 2), F_TRUE, dtype=torch.float32))


# 2. the constant has no gradient, the rest keeps the gradient it had


@pytest.mark.parametrize("columns", [4, 5])
def test_the_focal_columns_get_no_gradient_and_the_others_are_unchanged(columns):
    override_focal = erayzer_module().override_focal
    weights = torch.arange(1.0, columns + 1.0).expand(2, 3, columns).clone()

    predicted = intrinsics(columns, requires_grad=True)
    (override_focal(predicted, F_TRUE) * weights).sum().backward()
    with_override = predicted.grad

    plain = intrinsics(columns, requires_grad=True)
    (plain * weights).sum().backward()
    baseline = plain.grad

    assert torch.equal(with_override[..., :2], torch.zeros_like(with_override[..., :2])), "fx, fy frozen"
    assert (with_override[..., 2:] != 0).all(), "cx, cy and alpha still learn"
    assert torch.equal(with_override[..., 2:], baseline[..., 2:]), "and learn exactly as before"
    assert (baseline[..., :2] != 0).all(), "the comparison would be empty if they had no gradient anyway"


def test_the_ucm_alpha_keeps_its_own_gradient_path():
    """alpha comes from its own head and is concatenated later, so it must pass through untouched."""
    override_focal = erayzer_module().override_focal
    alpha = torch.tensor(0.62, requires_grad=True)
    predicted = torch.cat([intrinsics(4), alpha.expand(2, 3, 1)], dim=-1)
    override_focal(predicted, F_TRUE)[..., 4].sum().backward()
    assert alpha.grad is not None and alpha.grad.item() == pytest.approx(6.0)


def test_a_broken_prediction_is_replaced_instead_of_being_multiplied_away():
    """The run this switch is for is one where the focal length collapses, so it may arrive as nan.

    A constant built from the prediction (times zero, say) would carry nan and inf through, and the
    gradient of cx and cy with it. A fresh constant does not, which is the point of the switch.
    """
    override_focal = erayzer_module().override_focal
    predicted = torch.tensor([[float("nan"), float("inf"), 0.5, 0.5]], requires_grad=True)
    fixed = override_focal(predicted, F_TRUE)
    assert torch.isfinite(fixed).all()
    assert fixed[0, :2].tolist() == pytest.approx([F_TRUE, F_TRUE])
    fixed.sum().backward()
    assert torch.isfinite(predicted.grad).all()
    assert torch.equal(predicted.grad[:, :2], torch.zeros(1, 2))


def test_a_prediction_without_a_gradient_is_accepted():
    """At inference, and inside the freeze_focal_steps window, nothing requires a gradient."""
    override_focal = erayzer_module().override_focal
    with torch.no_grad():
        fixed = override_focal(intrinsics(4), F_TRUE)
    assert not fixed.requires_grad
    assert torch.equal(fixed[..., :2], torch.full((2, 3, 2), F_TRUE))


# 3. dtype, device and shape survive


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_dtype_and_device_are_the_ones_of_the_prediction(dtype):
    override_focal = erayzer_module().override_focal
    predicted = intrinsics(5, dtype=dtype)
    fixed = override_focal(predicted, F_TRUE)
    assert fixed.dtype == predicted.dtype
    assert fixed.device == predicted.device


def test_a_single_view_without_a_batch_dimension_works_too():
    override_focal = erayzer_module().override_focal
    predicted = torch.tensor([0.41, 0.37, 0.5, 0.5])
    fixed = override_focal(predicted, F_TRUE)
    assert fixed.shape == (4,)
    assert fixed.tolist() == pytest.approx([F_TRUE, F_TRUE, 0.5, 0.5])


# 4. the call site, so the constant reaches every ray


def forward_source():
    return inspect.getsource(erayzer_module().ERayZer.forward)


def test_the_override_sits_between_the_prediction_and_everything_that_uses_it():
    """Before the alpha concat, before the freeze detach, and before the input and target split."""
    source = forward_source()
    rearranged = source.index("pred_fxfycxcy = rearrange(")
    override = source.index("override_focal(pred_fxfycxcy")
    assert rearranged < override
    for later in [
        'if hasattr(self, "alpha_head")',  # the UCM alpha is concatenated after the override
        "pred_fxfycxcy.detach()",  # freeze_focal_steps
        "fxfycxcy_input = pred_fxfycxcy",  # the rays the scene representation is built from
        "fxfycxcy_target = pred_fxfycxcy",  # the rays that are rendered and compared to the image
        "fxfycxcy=rearrange(pred_fxfycxcy",  # and what camera_stats logs as fx_norm
    ]:
        assert override < source.index(later), f"the override has to run before {later}"


def test_the_override_runs_only_when_the_constant_is_set():
    source = forward_source()
    line = next(line for line in source.splitlines() if "override_focal(pred_fxfycxcy" in line)
    guard = next(line for line in source.splitlines() if "self.fix_focal_norm is not None" in line)
    assert source.index(guard) < source.index(line)
    assert len(guard) - len(guard.lstrip()) < len(line) - len(line.lstrip()), "the call is inside the guard"


# 5. the default is off, and -s switches it on


@pytest.mark.parametrize("camera_model", ["fisheye", "ucm"])
def test_the_default_leaves_the_model_without_a_constant(camera_model):
    ERayZer = erayzer_module().ERayZer
    config = tiny_model_config(camera_model)
    assert "fix_focal_norm" not in config.model
    assert ERayZer(config).fix_focal_norm is None


def test_the_shipped_config_does_not_set_the_constant():
    config = train.load_config(CONFIG_PATH, [])
    assert config.model.get("fix_focal_norm", None) is None


@pytest.mark.parametrize("text,expected", [("0.29383", F_TRUE), ("0.26", 0.26), ("1", 1.0)])
def test_a_set_option_arrives_on_the_model_as_a_number(text, expected):
    """-s model.fix_focal_norm 0.29383 has to survive the yaml loader and the model constructor."""
    config = train.load_config(CONFIG_PATH, [("model.fix_focal_norm", text)])
    assert float(config.model.fix_focal_norm) == pytest.approx(expected)

    ERayZer = erayzer_module().ERayZer
    tiny = tiny_model_config("fisheye")
    tiny.model.fix_focal_norm = config.model.fix_focal_norm
    model = ERayZer(tiny)
    assert isinstance(model.fix_focal_norm, float)
    assert model.fix_focal_norm == pytest.approx(expected)


def test_a_set_option_of_null_turns_it_off_again():
    config = train.load_config(CONFIG_PATH, [("model.fix_focal_norm", "null")])
    assert config.model.fix_focal_norm is None
    ERayZer = erayzer_module().ERayZer
    tiny = tiny_model_config("fisheye")
    tiny.model.fix_focal_norm = None
    assert ERayZer(tiny).fix_focal_norm is None


# 6. the slurm job script


SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slurm", "train_tartanair.sh"
)


def train_script():
    with open(SCRIPT_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


def test_the_job_script_passes_the_constant_on_only_when_it_is_set():
    lines = train_script().splitlines()
    assert 'FIX_FOCAL_NORM="${FIX_FOCAL_NORM:-}"' in lines, "empty is the default"
    # the option is added inside the guard, so an unset variable never names the key at all
    guard = lines.index('if [[ -n "$FIX_FOCAL_NORM" ]]; then')
    assert lines[guard + 1].strip() == 'EXTRA_ARGS+=(-s model.fix_focal_norm "$FIX_FOCAL_NORM")'
    assert lines[guard + 2].strip() == "fi"


def test_the_job_script_still_lets_a_hand_given_option_win():
    """The arguments of the sbatch call come last, so -s model.fix_focal_norm still overrides."""
    script = train_script()
    assert script.index('EXTRA_ARGS+=(-s model.fix_focal_norm') < script.index('EXTRA_ARGS+=("$@")')


def test_the_job_script_is_still_valid_bash():
    assert subprocess.run(["bash", "-n", SCRIPT_PATH]).returncode == 0
