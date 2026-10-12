"""User-placed target keyframes: the boundary-override maths and its metrics.

Two layers, deliberately separated:

* the *pure* layer needs no model at all -- :func:`apply_placed_targets` runs on
  synthetic boundary arrays and :func:`measure_hit` runs on synthetic motions, so
  the window construction (frames 4..7, the local-root finite difference, the
  pose yaw rotation, the exported ghost) and the error metric are pinned down
  exactly;
* the *behavioural* layer loads the real bundle and asserts what the model
  actually does with a placed target: displacement follows the target
  monotonically (the negative control the task asks for), a reachable target is
  hit inside :data:`TARGET_TOLERANCES`, and the alignment rule (the window lands
  on the clip's **last four** frames) is re-derived rather than assumed.

The behavioural layer is skipped when the bundle or the reference styles are
absent, exactly like ``tests/test_mlx_agent_math.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motionbricks_mlx.data.agent import (JOINTS, Agent, Command,  # noqa: E402
                                         matrix_cont6d,
                                         build_constraints, canonicalize,
                                         cont6d_matrix, matrix_xyzw, quaternion_matrix,
                                         y_rotation)
from motionbricks_mlx.data.style import load_style                    # noqa: E402
from motionbricks_mlx.data.targets import (TARGET_TOLERANCES, WINDOW_FRAMES,  # noqa: E402
                                           PlacedTarget, aligned_slice, apply_placed_targets,
                                           measure_hit, parse_target, parse_targets,
                                           plan_with_targets)
from motionbricks_mlx.models.bundle import Bundle                     # noqa: E402
from motionbricks_mlx.models.schema import EXTERNAL_POSE_WIDTH, GLOBAL_ROOT_WIDTH  # noqa: E402
from motionbricks_mlx.models.transition import MlxPipeline, TransitionConstraints  # noqa: E402

MLX_BUNDLE = ROOT / "checkpoints/g1-f32-mlx"
STYLES = Path.home() / "dev/motion-bricks-cpp/repo/generated/styles"
HAVE_BUNDLE = (MLX_BUNDLE / "manifest.json").is_file()
HAVE_STYLES = (STYLES / "walk.mbstyle").is_file()
NEEDS_MODEL = pytest.mark.skipif(not (HAVE_BUNDLE and HAVE_STYLES),
                                 reason="bundle and reference styles required")


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_parse_target_accepts_two_and_three_fields():
    assert parse_target("0,3") == PlacedTarget(0.0, 3.0, 0.0)
    assert parse_target("1,-2,45") == PlacedTarget(1.0, -2.0, 45.0)
    with pytest.raises(ValueError):
        parse_target("1")
    with pytest.raises(ValueError):
        parse_target("1,2,3,4")


def test_parse_targets_accepts_query_json_and_dict_forms():
    assert parse_targets("0,3,0;1,4,90") == [PlacedTarget(0.0, 3.0, 0.0),
                                              PlacedTarget(1.0, 4.0, 90.0)]
    assert parse_targets([[0.0, 3.0, 0.0], [1.0, 4.0]]) == [PlacedTarget(0.0, 3.0, 0.0),
                                                             PlacedTarget(1.0, 4.0, 0.0)]
    assert parse_targets([{"x": 1.0, "z": 2.0, "heading_deg": 30.0}]) == \
        [PlacedTarget(1.0, 2.0, 30.0)]
    assert parse_targets(None) == [] and parse_targets("") == []
    with pytest.raises(ValueError):
        parse_targets([1.0, 2.0, 3.0])


def test_placed_target_rejects_non_finite_and_wrong_pose_width():
    with pytest.raises(ValueError):
        PlacedTarget(float("nan"), 0.0)
    with pytest.raises(ValueError):
        PlacedTarget(0.0, 0.0, 0.0, np.zeros(7, np.float32))
    with pytest.raises(ValueError):
        PlacedTarget(0.0, 0.0, 0.0, np.full(EXTERNAL_POSE_WIDTH, np.nan, np.float32))


# ---------------------------------------------------------------------------
# window construction (no model)
# ---------------------------------------------------------------------------
def identity_quaternions(shape: tuple) -> np.ndarray:
    xyzw = np.zeros((*shape, 4), np.float32)
    xyzw[..., 3] = 1.0
    return xyzw


def valid_pose(seed: int = 3) -> np.ndarray:
    """A 303-wide external pose whose six-D blocks are real rotations."""
    rng = np.random.default_rng(seed)
    pose = np.zeros(EXTERNAL_POSE_WIDTH, np.float32)
    pose[0:99] = rng.normal(size=99)
    for joint in range(JOINTS):
        quaternion = rng.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        matrix_cont6d(quaternion_matrix(quaternion.astype(np.float32)),
                      pose[99 + joint * 6:99 + joint * 6 + 6])
    return pose


def pose_six(pose: np.ndarray, joint: int) -> np.ndarray:
    return pose[99 + joint * 6:99 + joint * 6 + 6]


def synthetic_constraints():
    """A boundary that looks like ``build_constraints``' output, but invented.

    The six-D rotation blocks are built from *real* rotation matrices through the
    reference's own writer (``matrix_cont6d``).  Random six-vectors would not be a
    valid rotation parameterisation (``cont6d_matrix`` normalises the first
    column), so they would test the fixture rather than the code.
    """
    rng = np.random.default_rng(7)
    global_root = np.zeros((8, GLOBAL_ROOT_WIDTH), np.float32)
    for frame in range(8):
        heading = np.float32(0.1 * frame)
        global_root[frame] = [np.float32(0.25 * frame), 0.78, np.float32(0.5 * frame),
                              np.cos(heading), np.sin(heading)]
    local_root = np.zeros((8, 4), np.float32)
    poses = np.zeros((8, EXTERNAL_POSE_WIDTH), np.float32)
    for frame in range(8):
        poses[frame] = valid_pose(seed=100 + frame)
    return {
        "global_root": global_root,
        "local_root": local_root,
        "poses": poses,
        "has_local_root": np.array([1, 1, 1, 0, 1, 1, 1, 1], np.uint8),
        "allowed_tokens": np.ones(11, np.uint8),
        "target_root_translations": np.zeros((4, 3), np.float32),
        "target_local_rotations_xyzw": identity_quaternions((4, JOINTS)),
        "has_global_root": np.ones(8, np.uint8),
        "has_poses": np.ones(8, np.uint8),
    }


def test_window_fills_last_four_frames_and_holds_the_last_target():
    constraints = synthetic_constraints()
    origin = constraints["global_root"][0].copy()
    frame = type("F", (), {"x": 0.0, "z": 0.0, "heading": 0.0})()
    window = apply_placed_targets(constraints, [PlacedTarget(0.0, 3.0, 0.0)], frame)

    assert window.frames == WINDOW_FRAMES
    for slot in range(WINDOW_FRAMES):
        assert constraints["global_root"][4 + slot, 0] == pytest.approx(origin[0], abs=1e-6)
        assert constraints["global_root"][4 + slot, 2] == pytest.approx(origin[2] + 3.0,
                                                                       abs=1e-6)
    # frames 0..3 are untouched
    assert np.array_equal(constraints["global_root"][0:4],
                          synthetic_constraints()["global_root"][0:4])
    # the exported ghost is exactly the window the graph got
    assert np.array_equal(constraints["target_root_translations"],
                          constraints["global_root"][4:8, 0:3])


def test_window_holds_the_last_of_two_targets_and_rejects_five():
    constraints = synthetic_constraints()
    frame = type("F", (), {"x": 0.0, "z": 0.0, "heading": 0.0})()
    window = apply_placed_targets(constraints,
                                  [PlacedTarget(0.0, 1.0), PlacedTarget(0.0, 3.0)], frame)
    assert [row[2] for row in constraints["global_root"][4:8]] == pytest.approx([1.0, 3.0, 3.0, 3.0])
    assert [row[2] for row in window.world_roots] == pytest.approx([1.0, 3.0, 3.0, 3.0])
    with pytest.raises(ValueError):
        apply_placed_targets(synthetic_constraints(),
                             [PlacedTarget(0, 1)] * 5, frame)
    with pytest.raises(ValueError):
        apply_placed_targets(synthetic_constraints(), [], frame)
    with pytest.raises(ValueError):
        apply_placed_targets(synthetic_constraints(), [PlacedTarget(0, 1)], frame,
                             pose_mode="nonsense")


def test_local_root_is_the_window_finite_difference():
    constraints = synthetic_constraints()
    frame = type("F", (), {"x": 0.0, "z": 0.0, "heading": 0.0})()
    apply_placed_targets(constraints, [PlacedTarget(0.0, 0.0), PlacedTarget(0.0, 1.0)], frame)
    for slot in range(3):
        current = constraints["global_root"][4 + slot]
        following = constraints["global_root"][5 + slot]
        expected = np.array([0.0, (following[0] - current[0]) * 30.0,
                             (following[2] - current[2]) * 30.0, current[1]], np.float32)
        assert np.allclose(constraints["local_root"][4 + slot], expected, atol=1e-5)
    # slot 7 repeats slot 6, with y taken from the window (src/agent.cpp:331-334)
    assert np.array_equal(constraints["local_root"][7, 0:3], constraints["local_root"][6, 0:3])
    assert constraints["local_root"][7, 3] == constraints["global_root"][7, 1]


def test_heading_override_rotates_the_root_and_the_pose_together():
    frame = type("F", (), {"x": 0.0, "z": 0.0, "heading": 0.0})()
    base = synthetic_constraints()
    rotated = synthetic_constraints()
    apply_placed_targets(rotated, [PlacedTarget(0.0, 1.0, 90.0)], frame)

    heading = np.arctan2(rotated["global_root"][4, 4], rotated["global_root"][4, 3])
    assert np.rad2deg(heading) == pytest.approx(90.0, abs=1e-4)

    # Every joint's six-D rotation must have been yawed by the same angle the
    # root heading moved by, otherwise the body orientation disagrees with the
    # root.  Compared as decoded matrices, with a float32 six-D round-trip bound.
    delta = y_rotation(np.float32(np.deg2rad(90.0) - 0.4))
    for joint in range(JOINTS):
        expected = delta @ cont6d_matrix(pose_six(base["poses"][4], joint))
        got = cont6d_matrix(pose_six(rotated["poses"][4], joint))
        assert np.allclose(got, expected, atol=1e-5)

    # The exported ghost's root-joint local rotation carries the same yaw.
    assert np.array_equal(
        rotated["target_local_rotations_xyzw"][0, 0],
        matrix_xyzw(delta @ quaternion_matrix(base["target_local_rotations_xyzw"][0, 0])))


def test_pose_mode_hold_replaces_the_configuration():
    frame = type("F", (), {"x": 0.0, "z": 0.0, "heading": 0.0})()
    style_mode = synthetic_constraints()
    hold_mode = synthetic_constraints()
    apply_placed_targets(style_mode, [PlacedTarget(0.0, 1.0)], frame, "style")
    apply_placed_targets(hold_mode, [PlacedTarget(0.0, 1.0)], frame, "hold")
    # Both modes yaw the block so the body agrees with the requested heading
    # (the reference's ``correction``, src/agent.cpp:277-280) -- but from
    # different sources: "style" rotates the style's own target configuration,
    # "hold" rotates the configuration the clip starts from.
    original = synthetic_constraints()
    style_delta = y_rotation(np.float32(0.0 - 0.4))
    hold_delta = y_rotation(np.float32(0.0 - 0.0))
    for joint in range(JOINTS):
        assert np.allclose(cont6d_matrix(pose_six(style_mode["poses"][4], joint)),
                           style_delta @ cont6d_matrix(pose_six(original["poses"][4], joint)),
                           atol=1e-5)
        assert np.allclose(cont6d_matrix(pose_six(hold_mode["poses"][4], joint)),
                           hold_delta @ cont6d_matrix(pose_six(hold_mode["poses"][0], joint)),
                           atol=1e-5)
    assert not np.allclose(hold_mode["poses"][4], style_mode["poses"][4], atol=1e-4)


def test_explicit_pose_is_used_verbatim():
    frame = type("F", (), {"x": 0.0, "z": 0.0, "heading": 0.0})()
    mine = valid_pose(seed=11)
    constraints = synthetic_constraints()
    apply_placed_targets(constraints, [PlacedTarget(0.0, 2.0, 0.0, mine)], frame)
    # An explicit pose is defined in the canonical frame *at the clip's current
    # heading*, so placing it with that same heading applies no extra yaw -- the
    # block is the caller's configuration, verbatim.
    delta = y_rotation(np.float32(0.0))
    for slot in range(WINDOW_FRAMES):
        assert np.allclose(constraints["poses"][4 + slot][0:99],
                           (delta @ mine[0:99].reshape(JOINTS - 1, 3).T).T.reshape(-1),
                           atol=1e-5)
        for joint in range(JOINTS):
            assert np.allclose(cont6d_matrix(pose_six(constraints["poses"][4 + slot], joint)),
                               delta @ cont6d_matrix(pose_six(mine, joint)), atol=1e-5)


def test_world_to_canonical_round_trip_uses_the_canonicalize_frame():
    """A target placed at world (x, z) must enter the graph at that world point."""
    frame = type("F", (), {"x": 1.5, "z": -2.0, "heading": np.float32(np.deg2rad(37.0))})()
    constraints = synthetic_constraints()
    origin = constraints["global_root"][0].copy()
    window = apply_placed_targets(constraints, [PlacedTarget(4.0, 5.0, 100.0)], frame)
    # canonical offset, as canonicalize would have produced it
    offset = np.array([4.0 - 1.5, 0.0, 5.0 + 2.0], np.float32)
    canonical = y_rotation(np.float32(-frame.heading)) @ offset
    assert constraints["global_root"][4, 0] == pytest.approx(origin[0] + canonical[0], abs=1e-5)
    assert constraints["global_root"][4, 2] == pytest.approx(origin[2] + canonical[2], abs=1e-5)
    assert np.rad2deg(window.headings[0]) == pytest.approx(100.0 - 37.0, abs=1e-4)


# ---------------------------------------------------------------------------
# hit metric (no model)
# ---------------------------------------------------------------------------
def synthetic_motion(window, root_offset=0.0, heading_offset_deg=0.0):
    from motionbricks_mlx.data.agent import Motion
    frames = 12
    roots = np.zeros((frames, 3), np.float32)
    rotations = np.zeros((frames, JOINTS, 4), np.float32)
    rotations[:, :, 3] = 1.0                                    # identity quaternions
    roots[aligned_slice(frames)] = window.roots
    roots[aligned_slice(frames), 0] += root_offset
    if heading_offset_deg:
        delta = y_rotation(np.float32(np.deg2rad(heading_offset_deg)))
        for index in range(frames - WINDOW_FRAMES, frames):
            rotations[index, 0] = matrix_xyzw(delta)
    motion = Motion(roots, rotations, np.zeros((4, 3), np.float32),
                    np.zeros((4, JOINTS, 4), np.float32))
    motion.target_roots = np.asarray(window.roots, np.float32)
    motion.target_local_rotations_xyzw = np.zeros((4, JOINTS, 4), np.float32)
    motion.target_local_rotations_xyzw[:, :, 3] = 1.0
    return motion


def test_aligned_slice_is_the_last_four_frames():
    assert aligned_slice(44) == slice(40, 44)
    assert aligned_slice(24) == slice(20, 24)
    with pytest.raises(ValueError):
        aligned_slice(3)


def test_measure_hit_is_zero_for_a_perfect_match():
    frame = type("F", (), {"x": 0.0, "z": 0.0, "heading": 0.0})()
    constraints = synthetic_constraints()
    window = apply_placed_targets(constraints, [PlacedTarget(0.0, 3.0, 0.0)], frame)
    parents = np.array([-1] + [0] * (JOINTS - 1), np.int32)
    rest = np.zeros((JOINTS, 3), np.float64)
    hit = measure_hit(synthetic_motion(window), window, parents, rest)
    assert hit["errors"] == {"root_max_abs_m": 0.0, "root_planar_m": 0.0,
                             "heading_deg": 0.0, "joint_m": 0.0}
    assert hit["within_tolerance"] and hit["exceeded"] == []
    assert hit["window_frames"] == [8, 11]
    assert [slot["output_frame"] for slot in hit["per_slot"]] == [8, 9, 10, 11]


def test_measure_hit_reports_root_and_heading_errors_and_fails_closed():
    frame = type("F", (), {"x": 0.0, "z": 0.0, "heading": 0.0})()
    constraints = synthetic_constraints()
    window = apply_placed_targets(constraints, [PlacedTarget(0.0, 3.0, 0.0)], frame)
    parents = np.array([-1] + [0] * (JOINTS - 1), np.int32)
    rest = np.zeros((JOINTS, 3), np.float64)

    hit = measure_hit(synthetic_motion(window, root_offset=0.4, heading_offset_deg=9.0),
                      window, parents, rest)
    assert hit["errors"]["root_planar_m"] == pytest.approx(0.4, abs=1e-6)
    assert hit["errors"]["heading_deg"] == pytest.approx(9.0, abs=1e-3)
    assert not hit["within_tolerance"]
    # ``joint_m`` fails too: with every joint parented to the root and zero rest
    # offsets, the whole body inherits the root's 0.4 m error.
    assert set(hit["exceeded"]) == {"root_planar_m", "heading_deg", "joint_m"}
    assert hit["per_slot"][0]["root_planar_m"] == pytest.approx(0.4, abs=1e-6)


def test_target_tolerances_are_declared_with_the_metric_names():
    assert set(TARGET_TOLERANCES) == {"root_planar_m", "heading_deg", "joint_m"}
    assert all(value > 0 for value in TARGET_TOLERANCES.values())


# ---------------------------------------------------------------------------
# behavioural: the real model
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rig():
    bundle = Bundle(MLX_BUNDLE, device="cpu")
    return bundle, MlxPipeline(bundle)


def _plan(rig, style, targets, seed=1, pose_mode="style"):
    bundle, pipeline = rig
    agent = Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
    agent.reset(style)
    command = Command(style=style, movement=(0.0, 0.0, 1.0), facing=(0.0, 0.0, 1.0), seed=seed)
    return plan_with_targets(agent, command, style, targets, pose_mode)


def _world_ahead(agent, distance, heading_deg=0.0):
    """Place a target ``distance`` metres along the clip start's own facing."""
    root = np.asarray(agent.context_roots[0], np.float64)
    matrix = quaternion_matrix(np.asarray(agent.context_rotations[0, 0], np.float64))
    heading = float(np.arctan2(matrix[0, 2], matrix[2, 2]))
    offset = y_rotation(np.float32(heading)) @ np.array([0.0, 0.0, distance], np.float32)
    return PlacedTarget(float(root[0] + offset[0]), float(root[2] + offset[2]),
                        float(np.rad2deg(heading) + heading_deg))


@NEEDS_MODEL
def test_alignment_rule_is_the_argmin_on_the_accepted_default_path(rig):
    """Re-derive the "window lands on the last 4 frames" rule, do not assume it.

    On the *unmodified* default path the model's own target window is compared
    against its own output at every offset; the argmin must be ``frames-4``.
    """
    bundle, pipeline = rig
    style = load_style(STYLES / "walk.mbstyle")
    for seed in (1, 2, 7):
        agent = Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
        agent.reset(style)
        motion, _ = agent.plan(Command(style=style, movement=(0.0, 0.0, 1.0),
                                       facing=(0.0, 0.0, 1.0), seed=seed), style)
        frames = motion.frames
        errors = [np.abs(np.asarray(motion.roots)[k:k + WINDOW_FRAMES][:, [0, 2]]
                         - np.asarray(motion.target_roots)[:, [0, 2]]).max()
                  for k in range(frames - WINDOW_FRAMES + 1)]
        assert int(np.argmin(errors)) == frames - WINDOW_FRAMES


@NEEDS_MODEL
def test_displacement_is_monotone_in_the_placed_target_distance(rig):
    """The negative control: put the target further away, walk further.

    Nothing is clamped and nothing is refused; the *achieved* root displacement
    must increase strictly with the requested distance, including well past what
    a 44-frame clip can cover (where the report has to say so honestly).
    """
    style = load_style(STYLES / "walk.mbstyle")
    bundle, pipeline = rig
    displacements = []
    for distance in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
        probe = Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
        probe.reset(style)
        target = _world_ahead(probe, distance)
        motion, _, _ = _plan(rig, style, [target])
        assert np.isfinite(np.asarray(motion.roots)).all()
        displacements.append(float(np.asarray(motion.roots)[-1, 2]
                                   - np.asarray(motion.roots)[0, 2]))
    assert displacements == sorted(displacements)
    assert all(later > earlier for earlier, later in zip(displacements, displacements[1:]))


@NEEDS_MODEL
def test_reachable_target_is_hit_inside_the_declared_tolerance(rig):
    style = load_style(STYLES / "walk.mbstyle")
    bundle, _ = rig
    for distance in (0.0, 1.0, 3.0):
        agent = Agent(rig[1], bundle.neutral_joints, bundle.joint_parents)
        agent.reset(style)
        target = _world_ahead(agent, distance)
        motion, _, window = _plan(rig, style, [target])
        hit = measure_hit(motion, window, bundle.joint_parents, bundle.neutral_joints)
        assert hit["within_tolerance"], (distance, hit["errors"])
        assert hit["errors"]["root_planar_m"] <= TARGET_TOLERANCES["root_planar_m"]


@NEEDS_MODEL
def test_impossible_target_is_reported_not_clamped(rig):
    """A 50 m target in a 1.47 s clip: the report must show the real failure."""
    style = load_style(STYLES / "walk.mbstyle")
    bundle, pipeline = rig
    agent = Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
    agent.reset(style)
    target = _world_ahead(agent, 50.0)
    motion, _, window = _plan(rig, style, [target])
    roots = np.asarray(motion.roots)
    hit = measure_hit(motion, window, bundle.joint_parents, bundle.neutral_joints)

    assert np.isfinite(roots).all(), "no NaN/inf may escape an over-far target"
    assert hit["window_frames"] == [motion.frames - WINDOW_FRAMES, motion.frames - 1]
    # 50 m asked for, tens of metres produced, metres of error reported.
    assert 10.0 < float(roots[-1, 2] - roots[0, 2]) < 50.0
    assert hit["errors"]["root_planar_m"] > TARGET_TOLERANCES["root_planar_m"]
    assert not hit["within_tolerance"]
    assert "root_planar_m" in hit["exceeded"]


@NEEDS_MODEL
def test_placed_target_plan_is_deterministic(rig):
    style = load_style(STYLES / "walk.mbstyle")
    bundle, pipeline = rig
    agent = Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
    agent.reset(style)
    target = _world_ahead(agent, 2.0, heading_deg=30.0)
    first, _, _ = _plan(rig, style, [target])
    second, _, _ = _plan(rig, style, [target])
    assert np.array_equal(np.asarray(first.roots), np.asarray(second.roots))
    assert np.array_equal(np.asarray(first.rotations_xyzw),
                          np.asarray(second.rotations_xyzw))


@NEEDS_MODEL
def test_pose_mode_hold_and_style_differ_only_in_the_window(rig):
    """``pose_mode`` is a real degree of freedom: it changes the window, not the path."""
    style = load_style(STYLES / "walk.mbstyle")
    bundle, pipeline = rig
    agent_a = Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
    agent_a.reset(style)
    target = _world_ahead(agent_a, 1.0)
    style_motion, _, style_window = _plan(rig, style, [target], pose_mode="style")
    hold_motion, _, hold_window = _plan(rig, style, [target], pose_mode="hold")

    assert style_window.pose_mode == "style" and hold_window.pose_mode == "hold"
    # x/z are the placement and must be identical; y is the target pelvis height,
    # which "hold" takes from the clip's current pose and "style" from the style
    # window -- a deliberate difference, not drift.
    assert np.array_equal(style_window.roots[:, [0, 2]], hold_window.roots[:, [0, 2]]), \
        "pose_mode must not move the placed target"
    assert not np.array_equal(style_window.roots[:, 1], hold_window.roots[:, 1])
    assert not np.array_equal(np.asarray(style_motion.rotations_xyzw),
                             np.asarray(hold_motion.rotations_xyzw))
