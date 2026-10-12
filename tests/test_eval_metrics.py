"""Known-answer tests for the metric algorithms (metrics 1-5).

Every expected value here is computed by hand in the comment; if one of
these breaks, either the test's arithmetic or the metric's algorithm
changed — and per metrics.py's contract, an algorithm change also means a
docs/status-glm.md recalibration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motionbricks_mlx.eval.document import PlanStep  # noqa: E402
from motionbricks_mlx.eval.metrics import (  # noqa: E402
    boundary_error,
    foot_sliding,
    grounding,
    jitter,
    motion_deviation,
    token_agreement,
)
from motionbricks_mlx.eval.synth import JOINT_NAMES, PARENTS, REST, synth_step  # noqa: E402
from motionbricks_mlx.export.plan_json import world_positions_xyz  # noqa: E402

J = len(JOINT_NAMES)


def quat_z_deg(angle_deg: float) -> np.ndarray:
    """Rotation about +z by the given angle, xyzw."""
    half = np.radians(angle_deg) / 2.0
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)])


def make_step(angles_deg, roots, target_roots=None, target_angles=None,
              world=None) -> PlanStep:
    """A PlanStep on the synthetic skeleton.

    ``angles_deg`` is (frames, joints) rotation angles about +z; targets
    default to the motion's last 4 frames (boundary error 0 by
    construction); ``world`` defaults to the writer's FK.
    """
    frames = roots.shape[0]
    angles_deg = np.asarray(angles_deg, dtype=np.float64)
    rotations = np.zeros((frames, J, 4), dtype=np.float64)
    for j in range(J):
        rotations[:, j] = [quat_z_deg(float(a)) for a in angles_deg[:, j]]
    target_roots = roots[-4:].copy() if target_roots is None else target_roots
    target_angles = (angles_deg[-4:].copy() if target_angles is None
                     else np.asarray(target_angles, dtype=np.float64))
    target_rotations = np.zeros((4, J, 4), dtype=np.float64)
    for j in range(J):
        target_rotations[:, j] = [quat_z_deg(float(a)) for a in target_angles[:, j]]
    if world is None:
        world = world_positions_xyz(roots, rotations, PARENTS, REST)
    return PlanStep(
        label="hand", style="test", seed=0, advance=0, fps=30,
        roots=roots, rotations_xyzw=rotations,
        target_roots=target_roots, target_rotations_xyzw=target_rotations,
        world_positions=world, joint_names=list(JOINT_NAMES),
    )


STATIC_ROOTS = np.tile([0.0, 0.72, 0.0], (8, 1))


def static_step() -> PlanStep:
    """No motion at all: every metric's degenerate case."""
    return make_step(np.zeros((8, J)), STATIC_ROOTS)


# --- metric 1: foot sliding -------------------------------------------------
def test_slide_zero_when_root_still():
    step = static_step()
    values = foot_sliding(step, contact_threshold_m=0.05)
    # feet never move in world coords -> slide exactly 0
    assert values["foot_slide_left_m"] == 0.0
    assert values["foot_slide_right_m"] == 0.0
    assert values["foot_slide_max_m"] == 0.0


def test_slide_equals_travel_for_dragging_feet():
    frames = 8
    roots = np.tile([0.0, 0.72, 0.0], (frames, 1))
    roots[:, 2] = 0.012 * np.arange(frames)          # 12 mm/frame forward
    step = make_step(np.zeros((frames, J)), roots)
    values = foot_sliding(step, contact_threshold_m=0.05)
    # feet glued to the translating body: slide = 0.012 * (frames-1)
    assert values["foot_slide_max_m"] == pytest.approx(0.012 * (frames - 1))


def test_slide_counts_only_contact_pairs():
    frames = 8
    roots = np.tile([0.0, 0.72, 0.0], (frames, 1))
    roots[:, 2] = 0.012 * np.arange(frames)
    step = make_step(np.zeros((frames, J)), roots)
    # threshold below every foot point -> never in contact -> no accumulation
    none_on = foot_sliding(step, contact_threshold_m=-1.0)
    assert none_on["foot_slide_max_m"] == 0.0
    assert none_on["contact_fraction_max"] == 0.0
    # threshold above everything -> every pair counts -> the full travel
    all_on = foot_sliding(step, contact_threshold_m=10.0)
    assert all_on["contact_fraction_max"] == 1.0
    assert all_on["foot_slide_max_m"] == pytest.approx(0.012 * (frames - 1))


# --- metric 2: jitter -------------------------------------------------------
def test_jitter_zero_for_frozen_motion():
    values = jitter(static_step())
    assert values["jerk_p50_deg"] == 0.0
    assert values["jerk_p95_deg"] == 0.0
    assert values["jerk_max_deg"] == 0.0
    # every increment is below the freeze epsilon -> fully stalled
    assert values["freeze_frac"] == 1.0


def test_jitter_hand_computed_second_difference():
    # angles [0, 3, 3, 1] deg -> increments theta = [3, 0, 2]
    # -> second differences 2 - 2*0 + 3 = 5  (single |d2| value)
    angles = np.tile([0.0, 3.0, 3.0, 1.0], (J, 1)).T
    roots = np.tile([0.0, 0.72, 0.0], (4, 1))
    values = jitter(make_step(angles, roots))
    # arccos-based angles carry ~1e-5 deg of float noise at these magnitudes
    assert values["jerk_max_deg"] == pytest.approx(5.0, abs=1e-4)
    assert values["jerk_p95_deg"] == pytest.approx(5.0, abs=1e-4)
    assert values["jerk_p50_deg"] == pytest.approx(5.0, abs=1e-4)


def test_freeze_frac_counts_zero_increments():
    # increments [3, 0, 2]: one of three is below the 0.01 deg epsilon
    angles = np.tile([0.0, 3.0, 3.0, 1.0], (J, 1)).T
    roots = np.tile([0.0, 0.72, 0.0], (4, 1))
    values = jitter(make_step(angles, roots))
    assert values["freeze_frac"] == 1.0 / 3.0


# --- metric 3: grounding ----------------------------------------------------
def hand_heights_step() -> PlanStep:
    """Left foot lowest point at hand-set heights; right foot held at 0.5."""
    frames = 8
    heights = np.array([0.02, 0.02, 0.02, 0.02, -0.01, -0.01, 0.2, 0.2])
    world = np.full((frames, J, 3), 0.5)
    world[:, :, 0] = 0.0
    world[:, :, 2] = 0.0
    for idx in (1, 2):                               # left ankle_roll, toe_base
        world[:, idx, 1] = heights
    roots = np.tile([0.0, 0.72, 0.0], (frames, 1))
    step = make_step(np.zeros((frames, J)), roots, world=world)
    # silence the pelvis-link mismatch this hand-set world would flag
    step.world_positions[:, 0, :] = step.roots
    return step


def test_grounding_hand_computed_fractions():
    values = grounding(hand_heights_step(), air_threshold_m=0.12)
    assert values["ground_penetration_frac"] == 0.25      # 2 of 8 below 0
    assert values["ground_airborne_frac"] == 0.25         # 2 of 8 above 0.12
    assert values["foot_height_min_m"] == -0.01
    assert values["foot_height_max_m"] == 0.2
    assert values["foot_height_p50_m"] == 0.02


# --- metric 4: boundary continuity -----------------------------------------
def test_boundary_error_zero_when_targets_are_the_tail():
    values = boundary_error(make_step(np.zeros((8, J)), STATIC_ROOTS))
    assert values["boundary_root_mean_m"] == 0.0
    assert values["boundary_root_max_m"] == 0.0
    assert values["boundary_rotation_mean_deg"] == 0.0
    assert values["boundary_rotation_max_deg"] == 0.0


def test_boundary_error_known_offset():
    frames = 8
    roots = np.tile([0.0, 0.72, 0.0], (frames, 1))
    target_roots = roots[-4:] + np.array([0.1, 0.0, 0.0])
    values = boundary_error(make_step(np.zeros((frames, J)), roots,
                                      target_roots=target_roots))
    assert values["boundary_root_mean_m"] == 0.1
    assert values["boundary_root_max_m"] == 0.1


def test_boundary_error_known_rotation():
    # targets rotated 2 deg away from the tail on every joint
    frames = 8
    angles = np.zeros((frames, J))
    values = boundary_error(make_step(angles, STATIC_ROOTS,
                                      target_angles=np.full((4, J), 2.0)))
    assert values["boundary_rotation_max_deg"] == pytest.approx(2.0, abs=1e-9)


# --- metric 5: cross-implementation -----------------------------------------
def test_token_agreement_exact_partial_mismatched():
    a = np.array([1, 2, 3, 4])
    assert token_agreement(a, a)["token_agreement_frac"] == 1.0
    one_off = token_agreement(a, np.array([1, 2, 3, 5]))
    assert one_off["token_agreement_frac"] == 0.75
    bad = token_agreement(a, np.array([1, 2]))
    assert bad["token_agreement_frac"] == 0.0            # length mismatch = 0
    assert bad["token_count_mine"] == 4
    assert bad["token_count_reference"] == 2


def test_motion_deviation_identical_is_zero():
    step = synth_step(seed=3)
    values = motion_deviation(step.roots, step.rotations_xyzw, step.roots,
                              step.rotations_xyzw, step.world_positions,
                              step.world_positions)
    assert values["deviation_roots_max_m"] == 0.0
    assert values["deviation_rotation_max_deg"] == 0.0
    assert values["deviation_world_max_m"] == 0.0


def test_motion_deviation_known_deltas():
    step = static_step()
    shifted_roots = step.roots + np.array([1.0e-3, 0.0, 0.0])
    values = motion_deviation(step.roots, step.rotations_xyzw, shifted_roots,
                              step.rotations_xyzw, step.world_positions,
                              step.world_positions)
    assert values["deviation_roots_max_m"] == 1.0e-3
    # rotating one joint by exactly 1 deg (same axis) reads as 1 deg of angle
    rotated = step.rotations_xyzw.copy()
    rotated[:, 0] = quat_z_deg(1.0)
    values = motion_deviation(step.roots, step.rotations_xyzw, step.roots,
                              rotated, step.world_positions,
                              step.world_positions)
    assert values["deviation_rotation_max_deg"] == pytest.approx(1.0, abs=1e-9)
    assert values["deviation_rotation_mean_deg"] == pytest.approx(1.0 / J,
                                                                  abs=1e-9)
