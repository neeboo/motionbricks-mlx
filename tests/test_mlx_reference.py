"""Numerical alignment against the reference runtime.

The golden values below were produced by the reference implementation itself
(built from ~/dev/motion-bricks-cpp/repo at the pinned revision) for the
deterministic synthetic constraint set in :func:`synthetic_constraints`, via
``tools/ref_harness.cpp``.  They are embedded so this test is a real numeric
gate even on a machine without the reference checkout.

Two optional tests additionally compare against live reference artifacts in
``out/ref/`` when they exist.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motionbricks_mlx.models.bundle import Bundle  # noqa: E402
from motionbricks_mlx.models.transition import MlxPipeline, TransitionConstraints  # noqa: E402

MLX_BUNDLE = ROOT / "checkpoints/g1-f32-mlx"
HAVE_BUNDLE = (MLX_BUNDLE / "manifest.json").is_file()
pytestmark = pytest.mark.skipif(not HAVE_BUNDLE,
                               reason="run scripts/mlx_convert_gguf.py first")

# --- reference golden (identical inputs, seed 1, Gumbel sampling) ----------
REFERENCE_FRAMES = 36
REFERENCE_TOKENS = [
    2, 7, 1, 7, 5, 3, 4, 7, 5, 4, 1, 7, 3, 4, 4, 9, 2, 3, 3, 5, 6, 2, 9, 8,
    2, 4, 3, 7, 7, 6, 4, 1, 2, 7, 3, 5, 6, 3, 6, 8, 5, 4, 1, 5, 3, 2, 4, 2,
    5, 4, 3, 7, 7, 3, 2, 8, 2, 4, 1, 8, 7, 3, 4, 2, 2, 3, 1, 7, 5, 2, 5, 8,
]
REFERENCE_DURATION_LOGITS = [
    -1.982918, -0.783087, -0.438872, -0.405664, -0.494324, -0.629432,
    -0.749406, -0.743562, -0.815375, -0.828947, -0.754676, -53.547089,
]
REFERENCE_ROOTS_HEAD = [0.0, 0.1826991, 0.0, 0.0069185, 0.1760559, 0.0035366]
REFERENCE_ROTATIONS_HEAD = [0.058454, -0.138546, -0.0244208, 0.9883278,
                            -0.4307471, -0.0063073, 0.0044868, 0.9024395]

TOLERANCE_ROOTS_M = 1.0e-3
TOLERANCE_ROTATIONS = 2.0e-3
TOLERANCE_LOGITS = 2.0e-3


def synthetic_constraints() -> TransitionConstraints:
    """Deterministic boundary constraints (not derived from any dataset)."""
    global_root = np.zeros((8, 5), np.float32)
    for frame in range(8):
        global_root[frame] = [0.01 * frame, 0.78 + 0.001 * frame, 0.02 * frame,
                              np.cos(0.01 * frame), np.sin(0.01 * frame)]
    local_root = np.tile(np.array([0.05, 0.1, 0.02, 0.78], np.float32), (8, 1))
    local_root[:, 0] *= np.arange(8)
    poses = np.zeros((8, 303), np.float32)
    poses[:, 99:] = np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (8, 34))
    has_local_root = np.ones(8, np.uint8)
    has_local_root[3] = 0
    return TransitionConstraints(
        global_root=global_root, local_root=local_root, poses=poses,
        has_global_root=np.ones(8, np.uint8), has_local_root=has_local_root,
        has_poses=np.ones(8, np.uint8), allowed_tokens=np.ones(11, np.uint8))


@lru_cache(maxsize=1)
def result():
    return MlxPipeline(Bundle(MLX_BUNDLE)).run_transition(synthetic_constraints(),
                                                           seed=1, capture_trace=True)


def test_selected_tokens_match_reference():
    assert result().selected_tokens == 9
    assert result().roots.shape[0] == REFERENCE_FRAMES


def test_pose_tokens_are_bit_identical_to_reference():
    """Gumbel sampling over the reference's own logits reproduces its draws."""
    mine = result().trace["pose.tokens"].astype(int).tolist()
    assert mine == REFERENCE_TOKENS


def test_duration_logits_within_tolerance():
    mine = result().trace["root.num_token_logits"]
    difference = np.abs(mine - np.array(REFERENCE_DURATION_LOGITS, np.float32))
    assert difference.max() < TOLERANCE_LOGITS


def test_decoded_roots_and_rotations_within_tolerance():
    graph = result()
    roots = graph.roots.reshape(-1)[:len(REFERENCE_ROOTS_HEAD)]
    rotations = graph.rotations_xyzw.reshape(-1)[:len(REFERENCE_ROTATIONS_HEAD)]
    assert np.abs(roots - np.array(REFERENCE_ROOTS_HEAD, np.float32)).max() \
        < TOLERANCE_ROOTS_M
    assert np.abs(rotations - np.array(REFERENCE_ROTATIONS_HEAD, np.float32)).max() \
        < TOLERANCE_ROTATIONS


def test_negative_control_golden_comparison_detects_a_shifted_array():
    """The tolerance gate must fail when the comparison input is perturbed."""
    roots = result().roots.reshape(-1)[:len(REFERENCE_ROOTS_HEAD)].copy()
    baseline = np.abs(roots - np.array(REFERENCE_ROOTS_HEAD, np.float32)).max()
    assert baseline < TOLERANCE_ROOTS_M
    perturbed = roots + 5.0e-3
    assert np.abs(perturbed - np.array(REFERENCE_ROOTS_HEAD, np.float32)).max() \
        > TOLERANCE_ROOTS_M


def test_same_seed_repeats_bitwise():
    """Determinism: two runs of the same plan agree exactly (no float drift)."""
    graph = MlxPipeline(Bundle(MLX_BUNDLE))
    first = graph.run_transition(synthetic_constraints(), seed=11)
    second = graph.run_transition(synthetic_constraints(), seed=11)
    assert np.array_equal(first.roots, second.roots)
    assert np.array_equal(first.rotations_xyzw, second.rotations_xyzw)
    assert first.selected_tokens == second.selected_tokens


def test_different_seed_changes_sampling_but_not_duration():
    graph = MlxPipeline(Bundle(MLX_BUNDLE))
    first = graph.run_transition(synthetic_constraints(), seed=1)
    second = graph.run_transition(synthetic_constraints(), seed=2)
    assert first.selected_tokens == second.selected_tokens
    assert not np.array_equal(first.rotations_xyzw, second.rotations_xyzw)


REFERENCE_PLAN = ROOT / "out/ref/plan-ref.json"
MLX_PLAN = ROOT / "out/plan-mlx.json"


@pytest.mark.skipif(not (REFERENCE_PLAN.is_file() and MLX_PLAN.is_file()),
                    reason="reference plan or MLX plan not generated")
def test_plan_documents_agree_within_tolerance():
    """End-to-end: the same style/seed inputs produce the same plan arrays."""
    reference = json.loads(REFERENCE_PLAN.read_text())
    mine = json.loads(MLX_PLAN.read_text())
    assert len(mine["steps"]) == len(reference["steps"])
    for ours, theirs in zip(mine["steps"], reference["steps"]):
        assert ours["motion"]["frames"] == theirs["motion"]["frames"]
        roots_a = np.array(ours["motion"]["roots"], np.float32).reshape(-1, 3)
        roots_b = np.array(theirs["motion"]["roots"], np.float32).reshape(-1, 3)
        rotations_a = np.array(ours["motion"]["rotations"], np.float32).reshape(-1, 34, 4)
        rotations_b = np.array(theirs["motion"]["rotations"], np.float32).reshape(-1, 34, 4)
        assert np.abs(roots_a - roots_b).max() < TOLERANCE_ROOTS_M
        assert np.abs(rotations_a - rotations_b).max() < TOLERANCE_ROTATIONS
        world_a = np.array(ours["world_positions_xyz"], np.float32).reshape(-1, 3)
        world_b = np.array(theirs["world_positions_xyz"], np.float32).reshape(-1, 3)
        assert np.abs(world_a - world_b).max() < 2.0e-3


REFERENCE_TRACE = ROOT / "out/ref/trace/transition-0.json"


@pytest.mark.skipif(not REFERENCE_TRACE.is_file(), reason="reference trace not captured")
def test_trace_stages_agree_beyond_presentation_tolerance():
    """Per-stage comparison: nothing may drift before the sampled tokens."""
    trace = json.loads(REFERENCE_TRACE.read_text())
    tensors = trace["tensors"]
    constraints = TransitionConstraints(
        global_root=np.array(tensors["input.global_root_values"], np.float32).reshape(8, 5),
        local_root=np.array(tensors["input.local_root_values"], np.float32).reshape(8, 4),
        poses=np.array(tensors["input.local_poses"], np.float32).reshape(8, 303),
        has_global_root=np.array(tensors["input.has_global_root_values"], np.uint8).reshape(8),
        has_local_root=np.array(tensors["input.has_local_root_values"], np.uint8).reshape(8),
        has_poses=np.array(tensors["input.has_local_poses"], np.uint8).reshape(8),
        allowed_tokens=np.array(tensors["input.allowed_pred_num_tokens"], np.uint8).reshape(11))
    mine = MlxPipeline(Bundle(MLX_BUNDLE)).run_transition(constraints, seed=1,
                                                          capture_trace=True)
    assert mine.selected_tokens == trace["selected_tokens"]
    assert np.array_equal(mine.trace["pose.tokens"],
                          np.array(tensors["pose.tokens"], np.float32))
    assert np.array_equal(mine.trace["pose.sampling_uniforms"],
                          np.array(tensors["pose.sampling_uniforms"], np.float32))
    assert np.abs(mine.trace["decoder.output"]
                  - np.array(tensors["decoder.output"], np.float32)).max() < 1.0e-3
    assert np.abs(mine.trace["decoded.roots"]
                  - np.array(tensors["decoded.roots"], np.float32)).max() < 1.0e-3
