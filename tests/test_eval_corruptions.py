"""The five corruptions: immutability, FK consistency, and their signatures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motionbricks_mlx.eval.corruptions import (  # noqa: E402
    EFFECT,
    add_jitter_noise,
    apply_corruption,
    freeze_frames,
    raise_feet,
    shuffle_frames,
    translate_roots,
)
from motionbricks_mlx.eval.metrics import jitter  # noqa: E402
from motionbricks_mlx.eval.synth import PARENTS, REST, synth_step  # noqa: E402
from motionbricks_mlx.eval.thresholds import (  # noqa: E402
    AIR_THRESHOLD_M,
    FREEZE_EPS_DEG,
)

STEP = synth_step(seed=0)


@pytest.mark.parametrize("name", sorted(EFFECT))
def test_corruption_leaves_the_original_untouched(name):
    before_roots = STEP.roots.copy()
    before_rot = STEP.rotations_xyzw.copy()
    apply_corruption(name, STEP, PARENTS, REST)
    np.testing.assert_array_equal(STEP.roots, before_roots)
    np.testing.assert_array_equal(STEP.rotations_xyzw, before_rot)


@pytest.mark.parametrize("name", sorted(EFFECT))
def test_corrupted_world_positions_are_the_real_fk(name):
    corrupted = apply_corruption(name, STEP, PARENTS, REST)
    from motionbricks_mlx.eval.document import recompute_world_positions
    np.testing.assert_allclose(
        corrupted.world_positions,
        recompute_world_positions(corrupted, PARENTS, REST),
        atol=1e-12,
    )
    # and the frames survive (corruptions reorder/hold, never drop)
    assert corrupted.frames == STEP.frames
    assert corrupted.joints == STEP.joints


def test_shuffle_frames_permutes_and_breaks_the_boundary():
    bad = shuffle_frames(STEP, PARENTS, REST)
    # same set of frames, different order
    np.testing.assert_array_equal(np.sort(bad.roots, axis=0),
                                  np.sort(STEP.roots, axis=0))
    assert not np.array_equal(bad.roots, STEP.roots)
    from motionbricks_mlx.eval.metrics import boundary_error
    values = boundary_error(bad)
    assert values["boundary_root_max_m"] > 0.1
    assert values["boundary_rotation_max_deg"] > 1.0


def test_freeze_frames_holds_and_drops_velocity():
    bad = freeze_frames(STEP, PARENTS, REST)
    # frames 12..31 all duplicate frame 12
    assert (bad.roots[12:32] == bad.roots[12]).all()
    assert (bad.rotations_xyzw[12:32] == bad.rotations_xyzw[12]).all()
    values = jitter(bad, freeze_eps_deg=FREEZE_EPS_DEG)
    assert values["freeze_frac"] >= 0.3
    # the timeline resumes where it left off (one catch-up jump, not drift)
    np.testing.assert_array_equal(bad.roots[32:], STEP.roots[32:])


def test_add_jitter_noise_raises_jerk_and_keeps_norms():
    bad = add_jitter_noise(STEP, PARENTS, REST)
    norms = np.linalg.norm(bad.rotations_xyzw, axis=-1)
    assert np.abs(norms - 1.0).max() <= 1e-12
    good, noisy = jitter(STEP), jitter(bad)
    assert noisy["jerk_p95_deg"] > good["jerk_p95_deg"] * 5
    assert noisy["jerk_max_deg"] > 25.0


def test_raise_feet_lifts_every_foot_point():
    bad = raise_feet(STEP, PARENTS, REST)
    heights = bad.foot_heights()
    assert (heights > AIR_THRESHOLD_M).all()
    from motionbricks_mlx.eval.metrics import grounding
    values = grounding(bad, air_threshold_m=AIR_THRESHOLD_M)
    assert values["ground_airborne_frac"] == 1.0
    # rigid lift: the pose itself is unchanged
    np.testing.assert_allclose(bad.rotations_xyzw, STEP.rotations_xyzw)


def test_translate_roots_shifts_exactly_and_breaks_boundary():
    dx, dz = 0.5, 0.5
    bad = translate_roots(STEP, PARENTS, REST, dx=dx, dz=dz)
    delta = bad.roots - STEP.roots
    np.testing.assert_allclose(delta, np.tile([dx, 0.0, dz], (STEP.frames, 1)))
    from motionbricks_mlx.eval.metrics import boundary_error
    values = boundary_error(bad)
    # targets stay put, tail moved diagonally: error >= the shift magnitude
    assert values["boundary_root_max_m"] >= np.hypot(dx, dz) - 1e-12


def test_apply_corruption_rejects_unknown_names():
    with pytest.raises(KeyError):
        apply_corruption("no_such_corruption", STEP, PARENTS, REST)


def test_effect_keys_exist_in_metric_output():
    from motionbricks_mlx.eval.gate import evaluate_step
    from motionbricks_mlx.eval.thresholds import THRESHOLDS
    values = evaluate_step(STEP, THRESHOLDS, expect_joints=6)
    for name, keys in EFFECT.items():
        assert keys, f"{name} maps to no metric keys"
        for key in keys:
            assert key in values, f"EFFECT[{name}] names unknown metric {key}"
