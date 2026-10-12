"""Motion representation maths: normalization, 6D rotations, decode.

Port of the parts of ``src/motion_rep.cpp`` that the inference path needs, plus
the pieces the agent layer needs to build constraints.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .schema import (BOUNDARY_FRAME_COUNT, EXTERNAL_POSE_WIDTH, FPS, GLOBAL_ROOT_WIDTH,
                     JOINT_COUNT, LOCAL_MOTION_WIDTH, LOCAL_ROOT_WIDTH, STATS_EPSILON)


def stats_scale(std: mx.array) -> mx.array:
    """``sqrt(stddev^2 + 1e-5)`` (src/motion_rep.cpp:110-119)."""
    return mx.sqrt(std * std + STATS_EPSILON)


def normalize(value: mx.array, mean: mx.array, std: mx.array, index: int | None = None) -> mx.array:
    scale = stats_scale(std)
    if index is None:
        return (value - mean) / scale
    return (value - mean[index]) / scale[index]


def unnormalize(value: mx.array, mean: mx.array, std: mx.array, index: int | None = None) -> mx.array:
    scale = stats_scale(std)
    if index is None:
        return value * scale + mean
    return value * scale[index] + mean[index]


def local_motion_statistics_indices() -> np.ndarray:
    """``dual_index`` mapping used by ``decode_motion`` (src/motion_rep.cpp:267).

    Features 0..3 use statistics slot ``5 + feature``; features 4..412 use
    ``9 + feature - 4`` (slots 9..417).
    """
    indices = np.empty(LOCAL_MOTION_WIDTH, dtype=np.int32)
    indices[:4] = 5 + np.arange(4)
    indices[4:] = 9 + np.arange(LOCAL_MOTION_WIDTH - 4)
    return indices


def unnormalize_local_motion(normalized: mx.array, mean: mx.array, std: mx.array) -> mx.array:
    indices = mx.array(local_motion_statistics_indices())
    return normalized * stats_scale(std)[indices] + mean[indices]


def cont6d_to_matrix(values: mx.array) -> mx.array:
    """``cont6d_matrix`` (src/motion_rep.cpp:87-102).

    Two 3-vectors define the first two *columns* of a rotation matrix;
    the third column is their normalized cross product and the second column is
    re-orthogonalized: ``M = [x | y | z]``.
    """
    first = values[..., 0:3]
    second_raw = values[..., 3:6]
    length = mx.sqrt(mx.sum(first * first, axis=-1, keepdims=True))
    first = mx.where(length > 1.0e-12, first / mx.maximum(length, 1e-30), first)
    third = mx.stack([
        first[..., 1] * second_raw[..., 2] - first[..., 2] * second_raw[..., 1],
        first[..., 2] * second_raw[..., 0] - first[..., 0] * second_raw[..., 2],
        first[..., 0] * second_raw[..., 1] - first[..., 1] * second_raw[..., 0],
    ], axis=-1)
    length = mx.sqrt(mx.sum(third * third, axis=-1, keepdims=True))
    third = mx.where(length > 1.0e-12, third / mx.maximum(length, 1e-30), third)
    second = mx.stack([
        third[..., 1] * first[..., 2] - third[..., 2] * first[..., 1],
        third[..., 2] * first[..., 0] - third[..., 0] * first[..., 2],
        third[..., 0] * first[..., 1] - third[..., 1] * first[..., 0],
    ], axis=-1)
    return mx.stack([first, second, third], axis=-1)          # columns


def matrix_to_quaternion_xyzw(matrix: mx.array) -> mx.array:
    """``matrix_quaternion`` (src/motion_rep.cpp:60-85), returned as XYZW.

    All four branches are evaluated and selected elementwise; the reference
    picks one per matrix.  ``w`` is normalized and forced non-negative.
    """
    m = matrix
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

    scale0 = 2.0 * mx.sqrt(mx.maximum(trace + 1.0, 0.0))
    w0 = 0.25 * scale0
    x0 = (m[..., 2, 1] - m[..., 1, 2]) / scale0
    y0 = (m[..., 0, 2] - m[..., 2, 0]) / scale0
    z0 = (m[..., 1, 0] - m[..., 0, 1]) / scale0

    scale1 = 2.0 * mx.sqrt(mx.maximum(1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 0.0))
    w1 = (m[..., 2, 1] - m[..., 1, 2]) / scale1
    x1 = 0.25 * scale1
    y1 = (m[..., 0, 1] + m[..., 1, 0]) / scale1
    z1 = (m[..., 0, 2] + m[..., 2, 0]) / scale1

    scale2 = 2.0 * mx.sqrt(mx.maximum(1.0 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2], 0.0))
    w2 = (m[..., 0, 2] - m[..., 2, 0]) / scale2
    x2 = (m[..., 0, 1] + m[..., 1, 0]) / scale2
    y2 = 0.25 * scale2
    z2 = (m[..., 1, 2] + m[..., 2, 1]) / scale2

    scale3 = 2.0 * mx.sqrt(mx.maximum(1.0 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1], 0.0))
    w3 = (m[..., 1, 0] - m[..., 0, 1]) / scale3
    x3 = (m[..., 0, 2] + m[..., 2, 0]) / scale3
    y3 = (m[..., 1, 2] + m[..., 2, 1]) / scale3
    z3 = 0.25 * scale3

    first = trace > 0.0
    second = (~first) & (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])
    third = (~first) & (~second) & (m[..., 1, 1] > m[..., 2, 2])

    w = mx.where(first, w0, mx.where(second, w1, mx.where(third, w2, w3)))
    x = mx.where(first, x0, mx.where(second, x1, mx.where(third, x2, x3)))
    y = mx.where(first, y0, mx.where(second, y1, mx.where(third, y2, y3)))
    z = mx.where(first, z0, mx.where(second, z1, mx.where(third, z2, z3)))

    norm = mx.sqrt(w * w + x * x + y * y + z * z)
    safe = mx.maximum(norm, 1e-30)
    w, x, y, z = w / safe, x / safe, y / safe, z / safe
    sign = mx.where(w < 0.0, -1.0, 1.0)
    return mx.stack([x * sign, y * sign, z * sign, w * sign], axis=-1)


def decode_motion(normalized_local_motion: mx.array, mean: mx.array, std: mx.array,
                  parents: np.ndarray, initial_x: float,
                  initial_z: float) -> tuple[mx.array, mx.array]:
    """``decode_motion`` (src/motion_rep.cpp:252-303).

    Returns ``roots`` ``(frames, 3)`` and ``local_rotations_xyzw``
    ``(frames, 34, 4)``.
    """
    raw = unnormalize_local_motion(normalized_local_motion, mean, std)
    frames = raw.shape[0]

    velocity = raw[:, 1:3] / FPS
    travelled = mx.concatenate([mx.zeros((1, 2), dtype=raw.dtype),
                                mx.cumsum(velocity, axis=0)[:-1]], axis=0)
    roots = mx.stack([
        travelled[:, 0] + initial_x,
        raw[:, 3],
        travelled[:, 1] + initial_z,
    ], axis=-1)

    rotation_offset = 4 + (JOINT_COUNT - 1) * 3
    values = raw[:, rotation_offset:rotation_offset + JOINT_COUNT * 6]
    global_rotations = cont6d_to_matrix(values.reshape(frames, JOINT_COUNT, 6))

    blocks = []
    for joint in range(JOINT_COUNT):
        parent = int(parents[joint])
        if parent < 0:
            blocks.append(global_rotations[:, joint])
        else:
            blocks.append(mx.swapaxes(global_rotations[:, parent], -1, -2)
                          @ global_rotations[:, joint])
    local = mx.stack(blocks, axis=1)
    return roots, matrix_to_quaternion_xyzw(local)


def decode_motion_batch(normalized_local_motion: mx.array, mean: mx.array, std: mx.array,
                        parents: np.ndarray, initial_x: mx.array,
                        initial_z: mx.array) -> tuple[mx.array, mx.array]:
    """``decode_motion`` over a leading batch axis (round 6, bake path only).

    The same arithmetic as :func:`decode_motion` with one extra leading dimension:
    the *frame* axis becomes axis 1, so the cumulative travel still accumulates
    along frames **within** each scenario, and the per-joint rotation product gains
    a batch dimension.  ``initial_x`` / ``initial_z`` are ``(batch, 1)`` because
    every scenario of a bake has its own origin (``decode_motion`` takes a scalar
    because a single transition has one).

    This exists because the decode was, by measurement, the most expensive
    unbatched part of a bake: it is a *tiny* graph executed **once per scenario**,
    so it is dominated by launch/drain cost rather than by arithmetic (report
    section 45).  One batched graph is an order of magnitude cheaper than ``B``
    small ones, exactly as the four neural stages were (rounds 3-4).

    Bit-identity against ``B`` separate :func:`decode_motion` calls is asserted by
    the bake's own instrument check (``bake.compare_bitwise``) on the real rows of
    every campaign, and by ``tests/test_mlx_bake.py``; the switch that selects it
    is declared off (``nn.OPT_DEFAULTS``), so the accepted path is untouched.
    """
    raw = unnormalize_local_motion(normalized_local_motion, mean, std)
    batch, frames = raw.shape[0], raw.shape[1]

    velocity = raw[:, :, 1:3] / FPS
    travelled = mx.concatenate([mx.zeros((batch, 1, 2), dtype=raw.dtype),
                                mx.cumsum(velocity, axis=1)[:, :-1]], axis=1)
    roots = mx.stack([
        travelled[:, :, 0] + initial_x,
        raw[:, :, 3],
        travelled[:, :, 1] + initial_z,
    ], axis=-1)

    rotation_offset = 4 + (JOINT_COUNT - 1) * 3
    values = raw[:, :, rotation_offset:rotation_offset + JOINT_COUNT * 6]
    global_rotations = cont6d_to_matrix(values.reshape(batch, frames, JOINT_COUNT, 6))

    blocks = []
    for joint in range(JOINT_COUNT):
        parent = int(parents[joint])
        if parent < 0:
            blocks.append(global_rotations[:, :, joint])
        else:
            blocks.append(mx.swapaxes(global_rotations[:, :, parent], -1, -2)
                          @ global_rotations[:, :, joint])
    local = mx.stack(blocks, axis=2)
    return roots, matrix_to_quaternion_xyzw(local)
