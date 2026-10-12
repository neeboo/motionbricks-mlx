"""Document loading, skeleton tables and structural validation (metric 6)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motionbricks_mlx.eval.document import (  # noqa: E402
    load_step,
    skeleton_tables,
    validate_structure,
)
from motionbricks_mlx.eval.synth import (  # noqa: E402
    JOINT_NAMES,
    PARENTS,
    REST,
    synth_step,
    to_document,
)

STEP = synth_step(seed=0)
DOCUMENT = to_document(STEP)


def test_load_step_round_trips_the_arrays():
    loaded = load_step(DOCUMENT, 0)
    assert loaded.label == STEP.label
    assert loaded.joint_names == JOINT_NAMES
    assert loaded.fps == 30
    np.testing.assert_allclose(loaded.roots, STEP.roots)
    np.testing.assert_allclose(loaded.rotations_xyzw, STEP.rotations_xyzw)
    np.testing.assert_allclose(loaded.target_roots, STEP.target_roots)
    np.testing.assert_allclose(loaded.target_rotations_xyzw,
                               STEP.target_rotations_xyzw)
    np.testing.assert_allclose(loaded.world_positions, STEP.world_positions)


def test_skeleton_tables_match_the_synth_tables():
    parents, rest = skeleton_tables(DOCUMENT)
    np.testing.assert_array_equal(parents, PARENTS)
    np.testing.assert_allclose(rest, REST)


def test_second_step_without_skeleton_inherits_names():
    document = to_document(STEP)
    document["steps"].append(dict(document["steps"][0]))
    del document["steps"][1]["skeleton"]
    document["steps"][1]["label"] = "second"
    loaded = load_step(document, 1)
    assert loaded.label == "second"
    assert loaded.joint_names == JOINT_NAMES


def test_validate_structure_passes_on_the_synth_step():
    result = validate_structure(STEP, expect_joints=6)
    assert result["problems"] == []
    assert result["frames"] == STEP.frames
    assert result["joints"] == 6
    assert result["fps"] == 30
    assert result["finite"] is True


def test_validate_structure_flags_wrong_joint_count():
    result = validate_structure(STEP, expect_joints=34)
    assert any("joints" in problem for problem in result["problems"])


def test_validate_structure_flags_fps_drift():
    step = synth_step(seed=0)
    step.fps = 60
    result = validate_structure(step, expect_joints=6)
    assert any("fps" in problem for problem in result["problems"])


def test_validate_structure_flags_nan():
    document = to_document(STEP)
    document["steps"][0]["motion"]["rotations"][3] = float("nan")
    result = validate_structure(load_step(document, 0), expect_joints=6)
    assert result["finite"] is False
    assert any("NaN" in problem for problem in result["problems"])


def test_validate_structure_flags_unnormalised_quaternion():
    document = to_document(STEP)
    rot = document["steps"][0]["motion"]["rotations"]
    start = 4 * (4 * 0 + 1)                     # frame 0, joint 1
    for k in range(4):
        rot[start + k] *= 1.1                   # norm 1.1 -> error 0.1 > 1e-3
    result = validate_structure(load_step(document, 0), expect_joints=6)
    assert any("quaternion norm" in problem for problem in result["problems"])


def test_validate_structure_flags_broken_world_link():
    document = to_document(STEP)
    world = np.asarray(document["steps"][0]["world_positions_xyz"])
    world[:3] += 0.01                           # pelvis column no longer == roots
    document["steps"][0]["world_positions_xyz"] = world.tolist()
    result = validate_structure(load_step(document, 0), expect_joints=6)
    assert any("pelvis" in problem for problem in result["problems"])


def test_load_step_rejects_ragged_arrays():
    document = to_document(STEP)
    document["steps"][0]["motion"]["roots"] = \
        document["steps"][0]["motion"]["roots"][:-3]
    with pytest.raises(ValueError):
        load_step(document, 0)
