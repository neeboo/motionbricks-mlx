"""The A -> B switch: seam measurement, the joined clip, and the control arm.

Two layers, like ``tests/test_mlx_targets.py``:

* a **pure-math** layer on synthetic clips, which pins the seam definitions
  (what counts as a position gap, a velocity jump, an acceleration jump, and what
  the in-segment p95 of the same finite difference is) without a model in the
  loop -- these run everywhere;
* a **real model** layer that binds the actual bundle and checks the numbers this
  round reports: the conditioned arm's seam stays inside the p95 while the
  independent-reseed control does not, and the whole thing is deterministic.

The criterion under test is the task's own rule --
``seam |dq| <= in-segment p95(|dq|)`` -- not a threshold invented here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from motionbricks_mlx.data.agent import Motion                       # noqa: E402
from motionbricks_mlx.data.switch import (LEG_KINDS, P95,            # noqa: E402
                                          POSITION_CRITERION, SEAM_CRITERION,
                                          Leg, SwitchPlan, Switcher, _ratio,
                                          join_legs, place_at, seam_metrics)
from motionbricks_mlx.data.style import load_style                   # noqa: E402
from motionbricks_mlx.models.bundle import Bundle                    # noqa: E402
from motionbricks_mlx.models.transition import MlxPipeline           # noqa: E402

MLX_BUNDLE = ROOT / "checkpoints/g1-f32-mlx"
STYLES = Path.home() / "dev/motion-bricks-cpp/repo/generated/styles"
HAVE_MODEL = (MLX_BUNDLE / "manifest.json").is_file() and (STYLES / "walk.mbstyle").is_file()
NEEDS_MODEL = pytest.mark.skipif(not HAVE_MODEL, reason="bundle and reference styles required")

#: A two-joint chain is enough for every positional statement: joint 1 hangs off
#: joint 0 by half a metre, so a body rotation moves a world position.
PARENTS = np.array([-1, 0], dtype=np.int32)
REST = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
FPS = 30.0


def chain_motion(frames: int, speed: float = 1.0, start_z: float = 0.0) -> Motion:
    """A clip that slides along +z at a constant speed, joints held still."""
    roots = np.zeros((frames, 3), dtype=np.float64)
    roots[:, 2] = start_z + np.arange(frames) * speed / FPS
    rotations = np.tile(IDENTITY, (frames, PARENTS.shape[0], 1)).astype(np.float32)
    return Motion(roots=roots, rotations_xyzw=rotations,
                  target_roots=np.zeros((4, 3), np.float32),
                  target_local_rotations_xyzw=np.zeros((4, PARENTS.shape[0], 4), np.float32))


# ---------------------------------------------------------------------------
# the ratio policy
# ---------------------------------------------------------------------------
def test_zero_over_zero_is_zero_and_anything_over_zero_is_out_of_tolerance():
    # A perfectly uniform segment has p95 = 0; "no worse than p95" then means
    # "no jump at all".  The two cases must not be conflated, and neither may be
    # ``inf`` (the plan document is parsed by ``JSON.parse`` in the page).
    assert _ratio(0.0, 0.0) == 0.0
    assert _ratio(1.0e-3, 0.0) > 1.0
    assert np.isfinite(_ratio(1.0e-3, 0.0))
    assert _ratio(0.5, 2.0) == pytest.approx(0.25)
    assert _ratio(3.0, 2.0) == pytest.approx(1.5)


def test_the_criterion_is_the_task_rule_and_the_ratio_keys_are_the_compared_quantities():
    assert SEAM_CRITERION == ("seam |dq| <= max(in-segment p95 |dq|)  for q in "
                             "{position, velocity, acceleration}")
    assert "p95" in POSITION_CRITERION
    assert P95 == 95.0
    report = seam_metrics(chain_motion(44), 20, chain_motion(44, start_z=20 / FPS),
                          PARENTS, REST)
    assert set(report["ratio"]) == {"root_position", "joint_position", "root_velocity",
                                    "root_acceleration", "joint_velocity",
                                    "joint_acceleration"}
    assert set(report["within_p95"]) == set(report["ratio"])
    assert report["all_within_p95"] is True


def test_a_perfectly_continuous_handoff_is_inside_every_p95():
    incoming = chain_motion(44, speed=1.0)
    # The outgoing clip starts exactly where the handoff frame is, at the same speed.
    outgoing = chain_motion(44, speed=1.0, start_z=20 / FPS)
    report = seam_metrics(incoming, 20, outgoing, PARENTS, REST)
    assert report["root_position_m"] == pytest.approx(0.0, abs=1e-12)
    assert report["joint_position_m"] == pytest.approx(0.0, abs=1e-12)
    assert report["all_within_p95"] is True
    assert report["position_within_one_frame"] is True


def test_a_velocity_jump_is_measured_against_the_segment_p95():
    incoming = chain_motion(44, speed=1.0)
    outgoing = chain_motion(44, speed=4.0, start_z=20 / FPS)
    report = seam_metrics(incoming, 20, outgoing, PARENTS, REST)
    assert report["root_velocity_mps"] == pytest.approx(3.0, abs=1e-9)
    assert report["within_p95"]["root_velocity"] is False
    assert report["all_within_p95"] is False


def test_a_position_gap_is_measured_in_metres_and_against_one_frame_of_motion():
    incoming = chain_motion(44, speed=1.0)
    outgoing = chain_motion(44, speed=1.0, start_z=20 / FPS + 0.5)
    report = seam_metrics(incoming, 20, outgoing, PARENTS, REST)
    assert report["root_position_m"] == pytest.approx(0.5, abs=1e-9)
    assert report["position_within_one_frame"] is False
    # One frame of a 1 m/s walk is 1/30 m, so half a metre is 15 frames' worth.
    assert report["ratio"]["root_position"] == pytest.approx(0.5 * FPS, rel=1e-6)


def test_an_acceleration_jump_shows_up_only_in_the_acceleration_row():
    # Steady 1 m/s, then a clip that ramps: the velocity at the handoff matches,
    # the acceleration does not.
    incoming = chain_motion(44, speed=1.0)
    frames, roots = 44, np.zeros((44, 3), dtype=np.float64)
    # z(i) = i/30 + c*i*(i-1): the first finite-difference velocity is exactly
    # 1 m/s (so the handoff velocity matches) and the acceleration is 60c*30.
    for index in range(44):
        roots[index, 2] = 20 / FPS + index / FPS + 0.001 * index * (index - 1)
    ramp = Motion(roots=roots,
                  rotations_xyzw=np.tile(IDENTITY, (frames, 2, 1)).astype(np.float32),
                  target_roots=np.zeros((4, 3), np.float32),
                  target_local_rotations_xyzw=np.zeros((4, 2, 4), np.float32))
    report = seam_metrics(incoming, 20, ramp, PARENTS, REST)
    assert report["ratio"]["root_velocity"] <= 1.0
    assert report["within_p95"]["root_acceleration"] is False


def test_short_inputs_are_rejected_rather_than_guessed():
    with pytest.raises(ValueError):
        seam_metrics(chain_motion(44), 2, chain_motion(44), PARENTS, REST)
    with pytest.raises(ValueError):
        seam_metrics(chain_motion(44), 44, chain_motion(44), PARENTS, REST)
    with pytest.raises(ValueError):
        seam_metrics(chain_motion(44), 20, chain_motion(3), PARENTS, REST)


def test_every_reported_number_is_finite_and_json_representable():
    import json
    report = seam_metrics(chain_motion(44), 20, chain_motion(44, speed=3.0), PARENTS, REST)
    text = json.dumps(report, allow_nan=False)   # raises if anything is NaN/inf
    assert "Infinity" not in text


# ---------------------------------------------------------------------------
# the joined clip
# ---------------------------------------------------------------------------
def test_join_legs_numbers_the_legs_and_measures_each_seam():
    legs = [Leg(style="a", kind=LEG_KINDS[0], motion=chain_motion(44, start_z=0.0),
                advance=0, frames_shown=20),
            Leg(style="b", kind=LEG_KINDS[0], motion=chain_motion(44, speed=2.0, start_z=20 / FPS),
                advance=20, frames_shown=12),
            Leg(style="a", kind=LEG_KINDS[0], motion=chain_motion(44, speed=1.0,
                                                                 start_z=32 / FPS),
                advance=12, frames_shown=44)]
    plan = join_legs(legs, PARENTS, REST, ["pelvis", "torso"])
    assert isinstance(plan, SwitchPlan)
    assert plan.frames == 20 + 12 + 44
    assert [entry["joined_frame"] for entry in plan.seams] == [20, 32]
    assert [entry["to_style"] for entry in plan.seams] == ["b", "a"]
    assert [leg["start"] for leg in plan.legs_payload()] == [0, 20, 32]
    assert [leg["end"] for leg in plan.legs_payload()] == [19, 31, 75]
    document = plan.document()
    assert document["joined_frames"] == plan.frames
    assert len(document["seams"]) == 2
    assert document["criterion"] == SEAM_CRITERION
    # Leg 0 slides at 1 m/s and leg 1 at 2 m/s: the first seam must show it.
    first = document["seams"][0]["metrics"]
    assert first["root_velocity_mps"] == pytest.approx(1.0, abs=1e-9)
    assert first["within_p95"]["root_velocity"] is False


def test_join_legs_needs_an_advance_frame_per_handoff():
    legs = [Leg(style="a", kind=LEG_KINDS[0], motion=chain_motion(44), advance=0,
                frames_shown=20),
            Leg(style="b", kind=LEG_KINDS[0], motion=chain_motion(44), advance=0,
                frames_shown=44)]
    with pytest.raises(ValueError, match="advance"):
        join_legs(legs, PARENTS, REST)


def test_join_legs_rejects_an_empty_leg_list():
    with pytest.raises(ValueError, match="at least one leg"):
        join_legs([], PARENTS, REST)


# ---------------------------------------------------------------------------
# the real model: the method must work, and the control must fail
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rig():
    if not HAVE_MODEL:
        pytest.skip("bundle and reference styles required")
    bundle = Bundle(MLX_BUNDLE, device="cpu")
    pipeline = MlxPipeline(bundle)
    names = ("walk", "walk_zombie", "slow_walk", "stealth_walk")
    styles = {name: load_style(STYLES / f"{name}.mbstyle") for name in names}
    return Switcher(pipeline, bundle, styles, seed=1), bundle


@NEEDS_MODEL
def test_the_conditioned_switch_is_deterministic(rig) -> None:
    switcher, _ = rig
    first_in, first_out, _ = switcher.seamless("walk", "walk_zombie", 20)
    second_in, second_out, _ = switcher.seamless("walk", "walk_zombie", 20)
    assert np.array_equal(first_in.roots, second_in.roots)
    assert np.array_equal(first_out.roots, second_out.roots)
    assert np.array_equal(first_out.rotations_xyzw, second_out.rotations_xyzw)


@NEEDS_MODEL
def test_conditioning_on_the_current_state_is_what_changes_the_clip(rig) -> None:
    """If the two arms produced the same clip, the switch would do nothing."""
    switcher, _ = rig
    _, conditioned, _ = switcher.seamless("walk", "walk_zombie", 20)
    _, independent = switcher.naive("walk", "walk_zombie", 20, align=False)
    assert not np.array_equal(conditioned.roots, independent.roots)
    assert not np.array_equal(conditioned.rotations_xyzw, independent.rotations_xyzw)


@NEEDS_MODEL
def test_the_conditioned_seam_lands_on_the_handoff_pose_and_the_control_does_not(rig) -> None:
    switcher, _ = rig
    incoming, conditioned, _ = switcher.seamless("walk", "walk_zombie", 20)
    same_incoming, aligned = switcher.naive("walk", "walk_zombie", 20, align=True)
    _, raw = switcher.naive("walk", "walk_zombie", 20, align=False)
    assert np.array_equal(incoming.roots, same_incoming.roots)   # same leg A

    good = seam_metrics(incoming, 20, conditioned, switcher.parents, switcher.rest)
    charitable = seam_metrics(same_incoming, 20, aligned, switcher.parents, switcher.rest)
    worst_case = seam_metrics(same_incoming, 20, raw, switcher.parents, switcher.rest)

    # Position: the conditioned handoff is sub-millimetre, the control is not.
    assert good["root_position_m"] < 1.5e-3
    assert good["joint_position_m"] < 2.0e-2
    assert good["position_within_one_frame"] is True
    assert charitable["root_position_m"] > 3.0 * good["root_position_m"]
    assert charitable["joint_position_m"] > 10.0 * good["joint_position_m"]
    assert worst_case["root_position_m"] > 0.1          # the un-aligned cut teleports
    assert worst_case["position_within_one_frame"] is False

    # Velocity: the conditioned start carries leg A's velocity; the reseed does not.
    assert good["ratio"]["root_velocity"] <= 1.0
    assert charitable["ratio"]["root_velocity"] > 1.0
    assert good["ratio"]["root_velocity"] < charitable["ratio"]["root_velocity"] / 4.0

    # Joint angular velocity: the conditioned seam is several times smaller than
    # the control's -- but it is NOT reported as "inside p95", which is the honest
    # half of this test (see docs/VIEWER.md section 11).
    assert good["ratio"]["joint_velocity"] < charitable["ratio"]["joint_velocity"] / 2.0
    assert good["joints_over_p95"]["velocity"] < charitable["joints_over_p95"]["velocity"]


@NEEDS_MODEL
def test_the_same_style_replan_is_the_mechanisms_own_seam_floor(rig) -> None:
    """A -> A is what a 4-frame conditioning window can do with no style change.

    It bounds the claim: if the switch's joint-velocity ratio sits in the same
    place as the same-style replan's, the residual is the mechanism's floor rather
    than the style change.
    """
    switcher, _ = rig
    incoming, same, _ = switcher.seamless("walk", "walk", 20)
    floor = seam_metrics(incoming, 20, same, switcher.parents, switcher.rest)
    _, switched, _ = switcher.seamless("walk", "walk_zombie", 20)
    changed = seam_metrics(incoming, 20, switched, switcher.parents, switcher.rest)
    assert floor["root_position_m"] < 1.5e-3
    assert changed["root_position_m"] < 1.5e-3
    # The style change does not move the joint-velocity ratio far off that floor.
    assert changed["ratio"]["joint_velocity"] < 2.5 * floor["ratio"]["joint_velocity"] + 0.1


@NEEDS_MODEL
def test_the_sequence_is_one_clip_with_every_seam_measured(rig) -> None:
    switcher, _ = rig
    plan = switcher.sequence(["walk", "walk_zombie", "walk"], [20, 20])
    assert plan.styles == ["walk", "walk_zombie", "walk"]
    assert plan.frames == 20 + 20 + 44
    assert [entry["joined_frame"] for entry in plan.seams] == [20, 40]
    assert plan.legs[2].motion.frames == 44
    # Both handoffs are conditioned on the live state, so both are sub-millimetre
    # in the root and inside the velocity p95.
    for entry in plan.seam_payload():
        metrics = entry["metrics"]
        assert metrics["root_position_m"] < 1.5e-3
        assert metrics["ratio"]["root_velocity"] <= 1.0
    assert plan.document()["all_seams_within_p95"] is False      # acceleration rows


@NEEDS_MODEL
def test_place_at_puts_the_clip_on_the_handoff_pose_without_touching_the_joints(rig) -> None:
    switcher, _ = rig
    incoming, _, _ = switcher.seamless("walk", "walk_zombie", 20)
    _, independent = switcher.naive("walk_zombie", "walk_zombie", 20, align=False)
    placed = place_at(independent, float(incoming.roots[20, 0]), float(incoming.roots[20, 2]),
                      float(np.arctan2(1.0, 1.0)))
    assert placed.roots[0, 0] == pytest.approx(incoming.roots[20, 0], abs=1e-6)
    assert placed.roots[0, 2] == pytest.approx(incoming.roots[20, 2], abs=1e-6)
    # Joint angles are otherwise untouched -- a world restore cannot fix a foreign
    # pose.  Joint 0 is the pelvis, which ``restore_world`` deliberately rewrites by
    # the heading correction (src/agent.cpp:194-196: the stored rotations are
    # parent-relative, so correcting the root rotates the whole body once).
    assert np.array_equal(placed.rotations_xyzw[:, 1:], independent.rotations_xyzw[:, 1:])
    assert not np.array_equal(placed.rotations_xyzw[:, 0], independent.rotations_xyzw[:, 0])
