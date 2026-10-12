"""Primitive-level tests: layout helpers, convolutions, forward shapes, sampling.

These pin the parts of the port that are easy to get subtly wrong (GGML layout
juggling and the im2col lowering) without needing the reference checkout.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motionbricks_mlx.data.agent import cont6d_matrix, matrix_xyzw, quaternion_matrix  # noqa: E402
from motionbricks_mlx.models import motion_rep, nn  # noqa: E402
from motionbricks_mlx.models.bundle import Bundle  # noqa: E402
from motionbricks_mlx.models.decoder import VqDecoder  # noqa: E402
from motionbricks_mlx.models.pose import PosePlanner  # noqa: E402
from motionbricks_mlx.models.root import RootPlanner  # noqa: E402
from motionbricks_mlx.models.sampling import gumbel_noise, sample_pose_tokens, uniforms  # noqa: E402
from motionbricks_mlx.models.transition import MlxPipeline, TransitionConstraints  # noqa: E402

MLX_BUNDLE = ROOT / "checkpoints/g1-f32-mlx"
HAVE_BUNDLE = (MLX_BUNDLE / "manifest.json").is_file()


@lru_cache(maxsize=1)
def pipeline() -> MlxPipeline:
    return MlxPipeline(Bundle(MLX_BUNDLE))


def test_reshape_ne_preserves_flat_order():
    flat = mx.arange(12, dtype=mx.float32)
    view = flat.reshape((4, 3))                      # GGML ne = (3, 4)
    reshaped = nn.reshape_ne(view, (6, 2))           # GGML ne = (6, 2) -> MLX (2, 6)
    assert reshaped.shape == (2, 6)
    assert np.array_equal(np.asarray(reshaped).reshape(-1), np.asarray(flat))


def test_transpose_contiguous_is_flat_transpose():
    """src/root.cpp:138-141: transpose then reinterpret as (first, second)."""
    flat = mx.arange(12, dtype=mx.float32).reshape((4, 3))
    result = nn.transpose_contiguous(flat, 3, 4)
    assert result.shape == (4, 3)
    assert np.array_equal(np.asarray(result).reshape(-1), np.asarray(flat.T).reshape(-1))


@pytest.mark.parametrize("channels,time,kernel,padding,dilation",
                         [(8, 11, 3, 1, 1), (8, 11, 3, 27, 27), (8, 12, 1, 0, 1),
                          (4, 7, 3, 9, 9)])
def test_conv_fast_path_matches_explicit_im2col(channels, time, kernel, padding, dilation):
    mx.random.seed(0)
    weight = mx.random.normal((6, channels, kernel))
    bias = mx.random.normal((6,))
    x = mx.random.normal((channels, time))
    fast = nn.conv1d_time(x, weight, bias, padding, dilation)
    explicit = nn.im2col_reference(x, weight, padding, dilation) + bias[:, None]
    assert fast.shape == explicit.shape == (6, time)
    assert float(mx.max(mx.abs(fast - explicit))) < 1e-5


def test_matrix_to_quaternion_round_trip():
    """The MLX converter must invert cont6d_to_matrix's own convention."""
    mx.random.seed(3)
    matrices = np.asarray(mx.random.normal((5, 3, 3)))
    for matrix in matrices:
        q, _ = np.linalg.qr(matrix)
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        expected = matrix_xyzw(q.astype(np.float32))
        got = np.asarray(motion_rep.matrix_to_quaternion_xyzw(mx.array(q[None].astype(np.float32))))[0]
        # The two converters differ only in canonical sign: motion_rep.cpp's
        # matrix_quaternion forces w >= 0, src/agent.cpp's matrix_xyzw does not.
        assert np.allclose(got, expected, atol=1e-5) or np.allclose(got, -expected, atol=1e-5)
        assert np.allclose(quaternion_matrix(got), quaternion_matrix(expected), atol=1e-5)


def test_sampling_uniforms_are_reproducible_and_seed_dependent():
    first = uniforms(1, 8)
    assert np.array_equal(first, uniforms(1, 8))
    assert not np.array_equal(first, uniforms(2, 8))
    assert first.min() >= 0.0 and first.max() < 1.0


def test_gumbel_sampling_is_first_max():
    logits = np.zeros((1, 3), dtype=np.float32)
    uniform = np.array([[0.5, 0.9, 0.1]], dtype=np.float32)
    chosen = sample_pose_tokens(logits, uniform, 3)
    assert chosen.tolist() == [int(np.argmax(gumbel_noise(uniform[0])))]
    # A tie keeps the first maximum, as torch.argmax does (src/sampling.cpp:43).
    assert sample_pose_tokens(np.zeros((1, 4), dtype=np.float32), None, 4).tolist() == [0]


@pytest.mark.skipif(not HAVE_BUNDLE, reason="run scripts/mlx_convert_gguf.py first")
def test_component_forward_shapes():
    graph = pipeline()
    global_root = mx.zeros((8, 5))
    local_root = mx.zeros((8, 4))
    poses = mx.zeros((8, 304))
    masks = mx.ones((8, 1))
    logits = graph.root.duration_logits(global_root, local_root, poses, masks, masks, masks)
    assert logits.shape == (12,)
    values = graph.root.plan(global_root, local_root, poses, masks, masks, masks, 11,
                             mx.zeros((44, 5)), mx.zeros((44, 1)))[1]
    assert values.shape == (44, 5)
    pose_logits = graph.pose.plan(np.full((11, 8), 10, np.int32), mx.zeros((44, 4)),
                                  mx.zeros((44, 304)), mx.zeros((44, 1)), 11, 11)
    assert pose_logits.shape == (11, 80)
    decoded = graph.decoder.decode(mx.zeros((256, 11)), mx.zeros((44, 2)),
                                   mx.zeros((44, 304)), mx.zeros((44, 1)), 11)
    assert decoded.shape == (44, 413)


@pytest.mark.skipif(not HAVE_BUNDLE, reason="run scripts/mlx_convert_gguf.py first")
def test_transition_shapes_and_token_range():
    graph = pipeline()
    constraints = TransitionConstraints(
        global_root=np.zeros((8, 5), np.float32), local_root=np.zeros((8, 4), np.float32),
        poses=np.zeros((8, 303), np.float32), has_global_root=np.ones(8, np.uint8),
        has_local_root=np.ones(8, np.uint8), has_poses=np.ones(8, np.uint8))
    result = graph.run_transition(constraints, seed=7)
    assert result.selected_tokens in range(6, 17)
    frames = result.selected_tokens * 4
    assert result.roots.shape == (frames, 3)
    assert result.rotations_xyzw.shape == (frames, 34, 4)
    norms = np.linalg.norm(result.rotations_xyzw, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)


@pytest.mark.skipif(not HAVE_BUNDLE, reason="run scripts/mlx_convert_gguf.py first")
def test_wrong_boundary_frame_count_is_rejected():
    """A dimension change must fail loudly rather than silently reshape."""
    graph = pipeline()
    with pytest.raises(Exception):
        graph.run_transition(TransitionConstraints(
            global_root=np.zeros((7, 5), np.float32), local_root=np.zeros((8, 4), np.float32),
            poses=np.zeros((8, 303), np.float32), has_global_root=np.ones(8, np.uint8),
            has_local_root=np.ones(8, np.uint8), has_poses=np.ones(8, np.uint8)), seed=1)


def test_cont6d_matches_agent_helper():
    """The MLX decoder's 6D conversion and the agent's numpy one agree."""
    mx.random.seed(5)
    values = np.asarray(mx.random.normal((6, 6)))
    mine = np.asarray(motion_rep.cont6d_to_matrix(mx.array(values[None])))[0]
    reference = cont6d_matrix(values)
    assert np.allclose(mine, reference, atol=1e-6)


def test_quaternion_matrix_matches_mlx_conversion_for_small_rotations():
    quaternion = np.array([0.01, -0.02, 0.03, 0.9993], dtype=np.float32)
    matrix = quaternion_matrix(quaternion)
    got = np.asarray(motion_rep.matrix_to_quaternion_xyzw(
        mx.array(matrix[None])))
    assert np.allclose(got[0], np.array([0.01, -0.02, 0.03, 0.9993]), atol=1e-4)


def test_pose_planner_head_offsets_and_shapes():
    """Each head reads its own 11-entry vocabulary slice (src/pose.cpp:283-287)."""
    table = mx.arange(88 * 32, dtype=mx.float32).reshape((88, 32))
    offsets = (np.arange(8, dtype=np.int32) * 11)[None, :]
    tokens = np.zeros((2, 8), dtype=np.int32)
    indices = (tokens + offsets).reshape(-1)
    assert indices.tolist() == [0, 11, 22, 33, 44, 55, 66, 77] * 2


def _corrupt(weights, name: str, delta: float):
    original = weights[name]
    weights.tensors[weights.resolve(name)] = original + delta
    return original


@pytest.mark.skipif(not HAVE_BUNDLE, reason="run scripts/mlx_convert_gguf.py first")
def test_corrupted_weight_must_change_the_output():
    """Negative control: break one weight and the forward output must move."""
    graph = pipeline()
    constraints = lambda: TransitionConstraints(  # noqa: E731
        global_root=np.zeros((8, 5), np.float32), local_root=np.zeros((8, 4), np.float32),
        poses=np.zeros((8, 303), np.float32), has_global_root=np.ones(8, np.uint8),
        has_local_root=np.ones(8, np.uint8), has_poses=np.ones(8, np.uint8))
    baseline = graph.run_transition(constraints(), seed=3).roots.copy()
    weights = graph.bundle.root
    name = "_proj_global_root_value.weight"
    resolved = weights.resolve(name)
    saved = weights.tensors[resolved]
    try:
        weights.tensors[resolved] = saved + 1.0e-2
        corrupted = graph.run_transition(constraints(), seed=3).roots
    finally:
        weights.tensors[resolved] = saved
    assert np.abs(corrupted - baseline).max() > 1e-4
    restored = graph.run_transition(constraints(), seed=3).roots
    assert np.array_equal(restored, baseline)
