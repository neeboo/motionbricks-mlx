"""The agent layer: context encoding, constraint building and playback seams.

Port of the parts of ``src/agent.cpp`` and ``src/motion_rep.cpp`` that sit
between a style asset + a command and the neural transition, plus the
post-processing the reference applies to the network's output.

All maths here is NumPy float32 on small arrays, mirroring the reference's
scalar C++ float loops one to one; the neural stages themselves stay in MLX.

Round 2 (``docs/optimization-report.md`` section 12) batches the per-joint and
per-frame loops that the reference writes as scalar loops.  The reference form
stays in this file verbatim as ``_*_ref`` so both arms of the A/B can run in one
process; ``MOTIONBRICKS_MLX_AGENT=0`` selects it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .style import Style

FPS = np.float32(30.0)
TWO_PI = np.float32(2.0 * np.pi)
JOINTS = 34
IDENTITY = np.eye(3, dtype=np.float32)

# ---------------------------------------------------------------------------
# Host-math switch
# ---------------------------------------------------------------------------
# Grammar identical to ``models.nn._option``: a bare token is the global switch,
# ``name=1`` / ``name=0`` overrides one name and ``*=1`` / ``*=0`` overrides all,
# applied in order.  The switch is read on every call so both arms can be
# interleaved inside one process.
#
# Decision (report section 12.2): the batched form is default-on only if it is
# token-identical, leaves ``mlx_align`` OK and improves the paired p50 by
# >= 1.05x.  The measured verdict is recorded in section 13 of the report.
_FALSY = frozenset(("", "0", "false", "no", "off"))
AGENT_MATH_DEFAULT = True


def _agent_math_option() -> bool:
    """True when the batched host maths is selected for this process."""
    raw = os.environ.get("MOTIONBRICKS_MLX_AGENT")
    if raw is None:
        return AGENT_MATH_DEFAULT
    enabled = AGENT_MATH_DEFAULT
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, separator, value = item.partition("=")
        key = key.strip().lower()
        if separator:
            if key in ("*", "all", "opt", "agent", "vector"):
                enabled = value.strip().lower() not in _FALSY
        else:
            enabled = key not in _FALSY
    return enabled


def y_rotation_batch(angles: np.ndarray) -> np.ndarray:
    """``y_rotation`` for an array of angles: ``(...,)`` -> ``(..., 3, 3)``.

    Same expression as :func:`y_rotation`, evaluated on the whole array.
    """
    cosine = np.cos(np.asarray(angles, dtype=np.float32))
    sine = np.sin(np.asarray(angles, dtype=np.float32))
    out = np.zeros(cosine.shape + (3, 3), dtype=np.float32)
    out[..., 0, 0] = cosine
    out[..., 0, 2] = sine
    out[..., 1, 1] = 1.0
    out[..., 2, 0] = -sine
    out[..., 2, 2] = cosine
    return out


def _tree_parent_gather(parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(parent_index, root_joints)`` for the batched parent-relative product.

    ``parent_index`` clamps the root's sentinel parent to joint 0 so the gather
    is well defined; ``root_joints`` names the entries whose local transform *is*
    the global one.
    """
    parents = np.asarray(parents, dtype=np.int32)
    return np.maximum(parents, 0), np.flatnonzero(parents < 0)


def rotate_rows(matrix: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """``matrix @ v`` for every row of ``vectors``: ``(3, 3) x (N, 3) -> (N, 3)``.

    An explicit contraction rather than ``matrix @ vectors.T`` because NumPy
    routes the batched matrix-matrix product and the per-row matrix-vector
    product to different BLAS kernels; on this machine that changes the last bit
    of a 3x3 result (measured 1.192e-07).  ``einsum`` reproduces the per-row gemv
    bit for bit, which is what keeps the batched path a drop-in for the loops.
    """
    return np.einsum("ij,kj->ki", matrix, vectors)


def dot_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """``np.dot(a, b)`` for every row pair: ``(..., D) x (..., D) -> (...,)``.

    ``np.dot`` on two 1-D float32 arrays goes to BLAS ``sdot``; neither
    ``np.sum(a * b, axis=-1)`` nor ``np.einsum`` reproduces its accumulation
    (measured 4.768e-07 on random 4-vectors).  The batch is therefore expressed
    as a batched matrix-vector product, which does match ``sdot`` bit for bit --
    the same class of trap as :func:`rotate_rows`, one level down.
    """
    return (left[..., None, :] @ right[..., :, None])[..., 0, 0]


def wrap(value: float) -> np.float32:
    """``std::remainder(value, 2*pi)`` (src/agent.cpp:126-129)."""
    return np.float32(np.remainder(np.float32(value) + np.float32(np.pi), TWO_PI)
                      - np.float32(np.pi))


def y_rotation(angle: float) -> np.ndarray:
    """``y_rotation`` (src/agent.cpp:33-36)."""
    cosine, sine = np.cos(np.float32(angle)), np.sin(np.float32(angle))
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
                    dtype=np.float32)


def quaternion_matrix(xyzw: np.ndarray) -> np.ndarray:
    """``quaternion_matrix`` (src/agent.cpp:44-50), XYZX input, row-major output."""
    x, y, z, w = xyzw[..., 0], xyzw[..., 1], xyzw[..., 2], xyzw[..., 3]
    scale = 2.0 / (x * x + y * y + z * z + w * w)
    matrix = np.empty(xyzw.shape[:-1] + (3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1.0 - scale * (y * y + z * z)
    matrix[..., 0, 1] = scale * (x * y - z * w)
    matrix[..., 0, 2] = scale * (x * z + y * w)
    matrix[..., 1, 0] = scale * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - scale * (x * x + z * z)
    matrix[..., 1, 2] = scale * (y * z - x * w)
    matrix[..., 2, 0] = scale * (x * z - y * w)
    matrix[..., 2, 1] = scale * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - scale * (x * x + y * y)
    return matrix


def matrix_xyzw(matrix: np.ndarray) -> np.ndarray:
    """``matrix_xyzw`` (src/agent.cpp:71-91): branchy, normalized, positive w."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return _matrix_xyzw(matrix)


def _matrix_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float32)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    shape = trace.shape
    wxyz = np.empty(shape + (4,), dtype=np.float32)

    first = trace > 0.0
    second = (~first) & (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])
    third = (~first) & (~second) & (m[..., 1, 1] > m[..., 2, 2])

    scale0 = 2.0 * np.sqrt(np.maximum(trace + 1.0, 0.0))
    scale1 = 2.0 * np.sqrt(np.maximum(1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 0.0))
    scale2 = 2.0 * np.sqrt(np.maximum(1.0 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2], 0.0))
    scale3 = 2.0 * np.sqrt(np.maximum(1.0 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1], 0.0))

    candidates = [
        (first, scale0,
         np.stack([0.25 * scale0, (m[..., 2, 1] - m[..., 1, 2]) / scale0,
                   (m[..., 0, 2] - m[..., 2, 0]) / scale0,
                   (m[..., 1, 0] - m[..., 0, 1]) / scale0], axis=-1)),
        (second, scale1,
         np.stack([(m[..., 2, 1] - m[..., 1, 2]) / scale1, 0.25 * scale1,
                   (m[..., 0, 1] + m[..., 1, 0]) / scale1,
                   (m[..., 0, 2] + m[..., 2, 0]) / scale1], axis=-1)),
        (third, scale2,
         np.stack([(m[..., 0, 2] - m[..., 2, 0]) / scale2,
                   (m[..., 0, 1] + m[..., 1, 0]) / scale2, 0.25 * scale2,
                   (m[..., 1, 2] + m[..., 2, 1]) / scale2], axis=-1)),
        (None, scale3,
         np.stack([(m[..., 1, 0] - m[..., 0, 1]) / scale3,
                   (m[..., 0, 2] + m[..., 2, 0]) / scale3,
                   (m[..., 1, 2] + m[..., 2, 1]) / scale3, 0.25 * scale3], axis=-1)),
    ]
    wxyz[:] = candidates[3][2]
    for mask, _, values in reversed(candidates[:3]):
        wxyz = np.where(mask[..., None], values, wxyz)
    norm = np.sqrt(np.sum(wxyz * wxyz, axis=-1, keepdims=True))
    wxyz = wxyz / norm
    # NOTE: unlike motion_rep.cpp's matrix_quaternion, agent.cpp's matrix_xyzw
    # does *not* canonically force w >= 0 (src/agent.cpp:88-90).  Adding that
    # flip here changes the stored context quaternions and the reference trace
    # diverges, so it is deliberately absent.
    return np.ascontiguousarray(np.stack(
        [wxyz[..., 1], wxyz[..., 2], wxyz[..., 3], wxyz[..., 0]], axis=-1))


def cont6d_matrix(values: np.ndarray) -> np.ndarray:
    """``cont6d_matrix`` (src/agent.cpp:52-64) - unguarded normalization."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return _cont6d_matrix(values)


def _cont6d_matrix(values: np.ndarray) -> np.ndarray:
    first = np.array(values[..., 0:3], dtype=np.float32)
    second_raw = np.array(values[..., 3:6], dtype=np.float32)
    first = first / np.sqrt(np.sum(first * first, axis=-1, keepdims=True))
    third = np.cross(first, second_raw)
    third = third / np.sqrt(np.sum(third * third, axis=-1, keepdims=True))
    second = np.cross(third, first)
    return np.stack([first, second, third], axis=-1)


def matrix_cont6d(matrix: np.ndarray, out: np.ndarray) -> None:
    """``matrix_cont6d`` (src/agent.cpp:66-69): first two columns, in place."""
    out[..., 0] = matrix[..., 0, 0]
    out[..., 1] = matrix[..., 1, 0]
    out[..., 2] = matrix[..., 2, 0]
    out[..., 3] = matrix[..., 0, 1]
    out[..., 4] = matrix[..., 1, 1]
    out[..., 5] = matrix[..., 2, 1]


def spring(current: float, velocity: float, target: float, half_life: float,
           time: float) -> np.float32:
    """``spring`` (src/agent.cpp:131-139)."""
    ln2 = np.float32(0.69314718056)
    y = np.float32((4.0 * ln2) / (np.float32(half_life) + np.float32(1.0e-5)) / 2.0)
    x = np.float32(y * np.float32(time))
    decay = np.float32(1.0 / (1.0 + x + np.float32(0.48) * x * x + np.float32(0.235) * x * x * x))
    j0 = np.float32(np.float32(current) - np.float32(target))
    j1 = np.float32(np.float32(velocity) + j0 * y)
    return np.float32((j0 + j1 * np.float32(time)) * decay + np.float32(target))


def slerp(source: np.ndarray, target: np.ndarray, amount: float) -> np.ndarray:
    """``slerp`` (src/agent.cpp:93-117)."""
    right = np.array(target, dtype=np.float32, copy=True)
    dot = float(np.dot(source, right))
    if dot < 0.0:
        dot = -dot
        right = -right
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        result = source + np.float32(amount) * (right - source)
    else:
        angle = np.arccos(np.float32(dot))
        denominator = np.sin(angle)
        left_weight = np.sin((1.0 - np.float32(amount)) * angle) / denominator
        right_weight = np.sin(np.float32(amount) * angle) / denominator
        result = left_weight * source + right_weight * right
    return (result / np.sqrt(np.sum(result * result))).astype(np.float32)


def physical_g1_joint(joint: int) -> bool:
    """``physical_g1_joint`` (src/agent.cpp:119-124)."""
    return joint not in (0, 7, 14, 25, 33)


# ---------------------------------------------------------------------------
# Context encoding (src/motion_rep.cpp:121-250)
# ---------------------------------------------------------------------------

@dataclass
class EncodedFrames:
    global_root: np.ndarray = field(default_factory=lambda: np.zeros((4, 5), np.float32))
    local_root: np.ndarray = field(default_factory=lambda: np.zeros((4, 4), np.float32))
    poses: np.ndarray = field(default_factory=lambda: np.zeros((4, 303), np.float32))


def _encode_global_frames_ref(joint_positions: np.ndarray, joint_rotations: np.ndarray,
                         repeat_last_velocity: bool) -> EncodedFrames:
    """``encode_global_frames`` (src/motion_rep.cpp:121-178)."""
    positions = np.ascontiguousarray(joint_positions, dtype=np.float32)
    rotations = np.ascontiguousarray(joint_rotations, dtype=np.float32).reshape(4, JOINTS, 3, 3)
    output = EncodedFrames()
    headings = np.arctan2(rotations[:, 0, 0, 2], rotations[:, 0, 2, 2]).astype(np.float32)
    root = positions[:, 0]
    output.global_root[:, 0:3] = root
    output.global_root[:, 3] = np.cos(headings)
    output.global_root[:, 4] = np.sin(headings)
    for frame in range(4):
        # x/z are root-relative, y stays absolute (src/motion_rep.cpp:151-153).
        offsets = positions[frame, 1:].copy()
        offsets[:, 0] -= root[frame, 0]
        offsets[:, 2] -= root[frame, 2]
        output.poses[frame, 0:99] = offsets.reshape(-1)
        six = output.poses[frame, 99:].reshape(JOINTS, 6)
        matrix_cont6d(rotations[frame], six)
    for frame in range(3):
        current = output.global_root[frame]
        following = output.global_root[frame + 1]
        output.local_root[frame] = [
            wrap(headings[frame + 1] - headings[frame]) * FPS,
            (following[0] - current[0]) * FPS,
            (following[2] - current[2]) * FPS,
            current[1],
        ]
    if repeat_last_velocity:
        output.local_root[3] = output.local_root[2]
        output.local_root[3, 3] = output.global_root[3, 1]
    return output


def _encode_context_ref(root_xyz: np.ndarray, local_rotation_xyzw: np.ndarray,
                   neutral_joints: np.ndarray, parents: np.ndarray) -> EncodedFrames:
    """``encode_context`` (src/motion_rep.cpp:180-250).

    Uses the last four frames of the supplied history; virtual hand/toe
    endpoints keep their FK positions but get global identity orientation.
    """
    frames = root_xyz.shape[0]
    first = frames - 4
    positions = np.zeros((4, JOINTS, 3), dtype=np.float32)
    rotations = np.zeros((4, JOINTS, 3, 3), dtype=np.float32)
    for frame in range(4):
        source = first + frame
        root = np.asarray(root_xyz[source], dtype=np.float32)
        for joint in range(JOINTS):
            local = quaternion_matrix(np.asarray(local_rotation_xyzw[source, joint],
                                                 dtype=np.float32))
            parent = int(parents[joint])
            if parent < 0:
                global_matrix = local
                positions[frame, joint] = root
            else:
                global_matrix = rotations[frame, parent] @ local
                rest = neutral_joints[joint] - neutral_joints[parent]
                positions[frame, joint] = positions[frame, parent] + \
                    rotations[frame, parent] @ rest
            rotations[frame, joint] = global_matrix
        for joint in (7, 14, 25, 33):
            rotations[frame, joint] = IDENTITY
    return _encode_global_frames_ref(positions, rotations, False)


# ---------------------------------------------------------------------------
# Commands, motion and the agent
# ---------------------------------------------------------------------------

@dataclass
class Command:
    """``mb_command`` (src/handles.hpp:44-58)."""

    style: Style | None = None
    movement: tuple[float, float, float] = (0.0, 0.0, 0.0)
    facing: tuple[float, float, float] = (0.0, 0.0, 1.0)
    target_speed: float = -1.0
    world_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    world_target_heading: float = 0.0
    has_world_target: bool = False
    skip_context_blend: bool = False
    seed: int = 0
    sampling_argmax: bool = False


@dataclass
class Motion:
    """``mb_motion`` (src/handles.hpp:70-78)."""

    roots: np.ndarray                     # (frames, 3)
    rotations_xyzw: np.ndarray            # (frames, 34, 4)
    target_roots: np.ndarray              # (4, 3)
    target_local_rotations_xyzw: np.ndarray  # (4, 34, 4)

    @property
    def frames(self) -> int:
        return int(self.roots.shape[0])


@dataclass
class WorldFrame:
    x: np.float32
    z: np.float32
    heading: np.float32


def _seed_agent_from_style_ref(style: Style, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``seed_agent_from_style`` (src/agent.cpp:381-417)."""
    roots = np.zeros((4, 3), dtype=np.float32)
    rotations = np.zeros((4, JOINTS, 4), dtype=np.float32)
    for frame in range(4):
        roots[frame] = style.global_root_positions[frame] + style.global_joint_positions[frame, 0]
        for joint in range(JOINTS):
            global_matrix = style.global_joint_rotations[frame, joint]
            parent = int(parents[joint])
            local = global_matrix if parent < 0 else \
                style.global_joint_rotations[frame, parent].T @ global_matrix
            rotations[frame, joint] = matrix_xyzw(local)
    return roots, rotations


def _canonicalize_ref(root_xyz: np.ndarray, local_rotations_xyzw: np.ndarray,
                 command: Command) -> tuple[WorldFrame, np.ndarray, np.ndarray, Command]:
    """``canonicalize`` (src/agent.cpp:147-181)."""
    first = root_xyz.shape[0] - 4
    first_root = root_xyz[first]
    root_matrix = quaternion_matrix(local_rotations_xyzw[first, 0])
    frame = WorldFrame(np.float32(first_root[0]), np.float32(first_root[2]),
                       np.float32(np.arctan2(root_matrix[0, 2], root_matrix[2, 2])))
    inverse_heading = y_rotation(-frame.heading)

    canonical_roots = np.empty_like(root_xyz)
    for index in range(root_xyz.shape[0]):
        shifted = np.array([root_xyz[index, 0] - frame.x, root_xyz[index, 1],
                            root_xyz[index, 2] - frame.z], dtype=np.float32)
        canonical_roots[index] = inverse_heading @ shifted
    canonical_rotations = np.array(local_rotations_xyzw, dtype=np.float32, copy=True)
    for index in range(local_rotations_xyzw.shape[0]):
        # Only the pelvis is rewritten: with parent-relative rotations a single
        # root rotation is exactly a whole-body rotation in FK, and rotating the
        # remaining joints as well would apply it twice (src/agent.cpp:165-168
        # has no inner joint loop, exactly like restore_world).
        canonical = inverse_heading @ quaternion_matrix(local_rotations_xyzw[index, 0])
        canonical_rotations[index, 0] = matrix_xyzw(canonical)

    canonical_command = Command(
        style=command.style,
        movement=tuple(float(v) for v in inverse_heading @ np.array(command.movement, np.float32)),
        facing=tuple(float(v) for v in inverse_heading @ np.array(command.facing, np.float32)),
        target_speed=command.target_speed,
        world_target=command.world_target,
        world_target_heading=command.world_target_heading,
        has_world_target=command.has_world_target,
        skip_context_blend=command.skip_context_blend,
        seed=command.seed,
        sampling_argmax=command.sampling_argmax,
    )
    if command.has_world_target:
        target = np.array([command.world_target[0] - frame.x, command.world_target[1],
                           command.world_target[2] - frame.z], dtype=np.float32)
        canonical_command.world_target = tuple(float(v) for v in inverse_heading @ target)
        canonical_command.world_target_heading = command.world_target_heading - frame.heading
    return frame, canonical_roots, canonical_rotations, canonical_command


def _build_constraints_ref(roots: np.ndarray, rotations: np.ndarray, command: Command,
                      style: Style, neutral_joints: np.ndarray,
                      parents: np.ndarray) -> dict:
    """``build_constraints`` (src/agent.cpp:209-350).

    Returns a dict of raw constraint arrays consumed by ``TransitionConstraints``.
    """
    context = _encode_context_ref(roots, rotations, neutral_joints, parents)
    current_x, current_z = context.global_root[0, 0], context.global_root[0, 2]
    velocity_x = np.float32((context.global_root[1, 0] - current_x) * FPS)
    velocity_z = np.float32((context.global_root[1, 2] - current_z) * FPS)

    direction_x, direction_z = np.float32(command.movement[0]), np.float32(command.movement[2])
    if np.hypot(direction_x, direction_z) <= np.float32(1.0e-5):
        direction_x = np.float32(command.facing[0] * 0.1)
        direction_z = np.float32(command.facing[2] * 0.1)
    travel = np.float32(command.target_speed * 2.0) if command.target_speed >= 0.0 \
        else np.float32(style.speed * 2.0)
    if command.target_speed < 0.0 and style.speed == 0.0:
        travel = np.float32(np.hypot(velocity_x, velocity_z) / 2.0)
    if travel <= np.float32(0.1):
        travel = np.float32(0.0)
    target_x = np.float32(command.world_target[0]) if command.has_world_target \
        else np.float32(current_x + travel * direction_x)
    target_z = np.float32(command.world_target[2]) if command.has_world_target \
        else np.float32(current_z + travel * direction_z)

    current_heading = np.arctan2(context.global_root[0, 4], context.global_root[0, 3])
    next_heading = np.arctan2(context.global_root[1, 4], context.global_root[1, 3])
    heading_velocity = np.float32(wrap(next_heading - current_heading) * FPS)
    desired_heading = np.float32(command.world_target_heading) if command.has_world_target \
        else np.arctan2(np.float32(command.facing[0]), np.float32(command.facing[2]))
    desired_heading = np.float32(current_heading + wrap(desired_heading - current_heading))

    start_root_x = np.empty(4, np.float32)
    start_root_z = np.empty(4, np.float32)
    start_heading = np.empty(4, np.float32)
    target_root_x = np.empty(4, np.float32)
    target_root_z = np.empty(4, np.float32)
    target_heading = np.empty(4, np.float32)
    for frame in range(4):
        start_time = np.float32(frame / 30.0)
        time = np.float32(1.0 + frame / 30.0)
        start_root_x[frame] = spring(current_x, velocity_x, target_x, 0.8, start_time)
        start_root_z[frame] = spring(current_z, velocity_z, target_z, 0.8, start_time)
        start_heading[frame] = spring(current_heading, heading_velocity, desired_heading,
                                      0.17, start_time)
        target_root_x[frame] = spring(current_x, velocity_x, target_x, 0.8, time)
        target_root_z[frame] = spring(current_z, velocity_z, target_z, 0.8, time)
        target_heading[frame] = spring(current_heading, heading_velocity, desired_heading,
                                       0.17, time)
    if style.speed == 0.0 and command.target_speed < 0.0 and not command.has_world_target:
        target_root_x[:] = target_root_x[0]
        target_root_z[:] = target_root_z[0]

    result = {
        "global_root": np.zeros((8, 5), np.float32),
        "local_root": np.zeros((8, 4), np.float32),
        "poses": np.zeros((8, 303), np.float32),
    }
    rotation_offset = (JOINTS - 1) * 3
    for frame in range(4):
        old_heading = np.arctan2(context.global_root[frame, 4], context.global_root[frame, 3])
        correction = y_rotation(start_heading[frame] - old_heading)
        pose = context.poses[frame]
        pose[0:99] = (correction @ pose[0:99].reshape(JOINTS - 1, 3).T).T.reshape(-1)
        for joint in range(JOINTS):
            six = pose[rotation_offset + joint * 6: rotation_offset + joint * 6 + 6]
            matrix_cont6d(correction @ cont6d_matrix(six), six)
        context.global_root[frame, 0] = start_root_x[frame]
        context.global_root[frame, 2] = start_root_z[frame]
        context.global_root[frame, 3] = np.cos(start_heading[frame])
        context.global_root[frame, 4] = np.sin(start_heading[frame])
    for frame in range(3):
        current = context.global_root[frame]
        following = context.global_root[frame + 1]
        context.local_root[frame] = [
            wrap(start_heading[frame + 1] - start_heading[frame]) * FPS,
            (following[0] - current[0]) * FPS,
            (following[2] - current[2]) * FPS,
            current[1],
        ]
    # std::copy writes only the first four frames; the rest stay zero until the
    # target block below fills frames 4..7 (src/agent.cpp:289-291).
    result["global_root"][0:4] = context.global_root
    result["local_root"][0:4] = context.local_root
    result["poses"][0:4] = context.poses

    first = 0 if style.frames == 4 else int(command.seed % (style.frames - 4))
    target_root_translations = np.zeros((4, 3), np.float32)
    target_local_rotations = np.zeros((4, JOINTS, 4), np.float32)
    for frame in range(4):
        source_frame = first + frame
        correction = wrap(target_heading[frame] - style.global_headings[source_frame])
        correction_matrix = y_rotation(correction)
        global_root = result["global_root"][frame + 4]
        global_root[0] = target_root_x[frame]
        global_root[1] = style.global_root_positions[source_frame, 1] + \
            style.global_joint_positions[source_frame, 0, 1]
        global_root[2] = target_root_z[frame]
        global_root[3] = np.cos(target_heading[frame])
        global_root[4] = np.sin(target_heading[frame])
        target_root_translations[frame] = global_root[0:3]
        pose = result["poses"][frame + 4]
        for joint in range(1, JOINTS):
            pose[(joint - 1) * 3:(joint - 1) * 3 + 3] = \
                correction_matrix @ style.global_joint_positions[source_frame, joint]
        for joint in range(JOINTS):
            matrix = correction_matrix @ style.global_joint_rotations[source_frame, joint]
            six = pose[rotation_offset + joint * 6: rotation_offset + joint * 6 + 6]
            matrix_cont6d(matrix, six)
        target_globals = np.stack(
            [correction_matrix @ style.global_joint_rotations[source_frame, joint]
             for joint in range(JOINTS)])
        for joint in range(JOINTS):
            parent = int(parents[joint])
            local = target_globals[joint] if parent < 0 else \
                target_globals[parent].T @ target_globals[joint]
            target_local_rotations[frame, joint] = matrix_xyzw(local)
    for frame in range(3):
        current = result["global_root"][frame + 4]
        following = result["global_root"][frame + 5]
        result["local_root"][frame + 4] = [
            wrap(target_heading[frame + 1] - target_heading[frame]) * FPS,
            (following[0] - current[0]) * FPS,
            (following[2] - current[2]) * FPS,
            current[1],
        ]
    result["local_root"][7] = result["local_root"][6]
    result["local_root"][7, 3] = result["global_root"][7, 1]

    result["has_local_root"] = np.array([1, 1, 1, 0, 1, 1, 1, 1], np.uint8)
    result["allowed_tokens"] = style.allowed_tokens.copy()
    result["target_root_translations"] = target_root_translations
    result["target_local_rotations_xyzw"] = target_local_rotations
    return result


def _restore_world_ref(motion: Motion, frame: WorldFrame) -> None:
    """``restore_world`` (src/agent.cpp:183-207), in place."""
    first_matrix = quaternion_matrix(motion.rotations_xyzw[0, 0])
    first_heading = np.arctan2(first_matrix[0, 2], first_matrix[2, 2])
    correction = y_rotation(frame.heading - first_heading)
    first_x, first_z = motion.roots[0, 0], motion.roots[0, 2]
    for index in range(motion.frames):
        shifted = np.array([motion.roots[index, 0] - first_x, motion.roots[index, 1],
                            motion.roots[index, 2] - first_z], dtype=np.float32)
        rotated = correction @ shifted
        motion.roots[index] = [rotated[0] + frame.x, rotated[1], rotated[2] + frame.z]
        # Only the pelvis is rewritten: the stored rotations are parent-relative,
        # so rotating joint 0 rotates the whole body in FK and rotating the other
        # joints as well would double-apply the correction.  This mirrors
        # src/agent.cpp:194-196, which has no inner joint loop.
        motion.rotations_xyzw[index, 0] = matrix_xyzw(
            correction @ quaternion_matrix(motion.rotations_xyzw[index, 0]))
    target_correction = y_rotation(frame.heading)
    for index in range(4):
        rotated = target_correction @ motion.target_roots[index]
        motion.target_roots[index] = [rotated[0] + frame.x, rotated[1], rotated[2] + frame.z]
        motion.target_local_rotations_xyzw[index, 0] = matrix_xyzw(
            target_correction @ quaternion_matrix(motion.target_local_rotations_xyzw[index, 0]))


def _apply_context_blend_ref(roots: np.ndarray, rotations: np.ndarray, motion: Motion) -> None:
    """``apply_context_blend`` (src/agent.cpp:354-379), in place."""
    weights = (np.float32(0.3), np.float32(0.433333337), np.float32(0.566666663),
               np.float32(0.7))
    begin = roots.shape[0] - 4
    for frame in range(4):
        weight = weights[frame]
        context_root = roots[begin + frame]
        for axis in range(3):
            motion.roots[frame, axis] = np.float32(
                context_root[axis] * (1.0 - weight) + motion.roots[frame, axis] * weight)
        for joint in range(1, JOINTS):
            if not physical_g1_joint(joint):
                continue
            motion.rotations_xyzw[frame, joint] = slerp(
                rotations[begin + frame, joint], motion.rotations_xyzw[frame, joint], weight)


# ---------------------------------------------------------------------------
# Batched host maths (round 2, item 1: "fewer calls")
# ---------------------------------------------------------------------------
# Every function below is the ``_ref`` implementation with the per-joint /
# per-frame Python loop replaced by one array expression over the same axis.
# Nothing is re-associated: the reduction axis of each dot product is still the
# same three (or four) elements in the same order, which is why these are
# expected to be bit-identical and are measured against the token gate anyway.
#
# Call counts per planning step drop from ~172 ``matrix_xyzw`` + ~135
# ``cont6d_matrix`` + ~175 ``quaternion_matrix`` calls to a handful, which is
# the whole point: at 34-46 us per call on 3x3 arrays the cost was the call, not
# the arithmetic (report section 3).


def _encode_global_frames_fast(joint_positions: np.ndarray,
                               joint_rotations: np.ndarray,
                               repeat_last_velocity: bool) -> EncodedFrames:
    """Batched :func:`_encode_global_frames_ref`."""
    positions = np.ascontiguousarray(joint_positions, dtype=np.float32)
    rotations = np.ascontiguousarray(joint_rotations, dtype=np.float32).reshape(4, JOINTS, 3, 3)
    output = EncodedFrames()
    headings = np.arctan2(rotations[:, 0, 0, 2], rotations[:, 0, 2, 2]).astype(np.float32)
    root = positions[:, 0]
    output.global_root[:, 0:3] = root
    output.global_root[:, 3] = np.cos(headings)
    output.global_root[:, 4] = np.sin(headings)
    # x/z are root-relative, y stays absolute (src/motion_rep.cpp:151-153).
    offsets = positions[:, 1:].copy()
    offsets[:, :, 0] -= root[:, 0:1]
    offsets[:, :, 2] -= root[:, 2:3]
    output.poses[:, 0:99] = offsets.reshape(4, -1)
    six = output.poses[:, 99:].reshape(4, JOINTS, 6)
    matrix_cont6d(rotations, six)
    output.poses[:, 99:] = six.reshape(4, -1)
    output.local_root[0:3, 0] = wrap(headings[1:4] - headings[0:3]) * FPS
    output.local_root[0:3, 1] = (output.global_root[1:4, 0]
                                 - output.global_root[0:3, 0]) * FPS
    output.local_root[0:3, 2] = (output.global_root[1:4, 2]
                                 - output.global_root[0:3, 2]) * FPS
    output.local_root[0:3, 3] = output.global_root[0:3, 1]
    if repeat_last_velocity:
        output.local_root[3] = output.local_root[2]
        output.local_root[3, 3] = output.global_root[3, 1]
    return output


def _forward_kinematics_fast(root_xyz: np.ndarray, local_rotation_xyzw: np.ndarray,
                             neutral_joints: np.ndarray,
                             parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The four-frame FK tree of :func:`_encode_context_ref`, batched.

    The parent chain is a genuine dependency, so the loop stays — but it runs 34
    times over all four frames at once instead of 4 x 34 times over scalars.
    """
    frames = root_xyz.shape[0]
    first = frames - 4
    positions = np.zeros((4, JOINTS, 3), dtype=np.float32)
    rotations = np.zeros((4, JOINTS, 3, 3), dtype=np.float32)
    roots4 = np.ascontiguousarray(root_xyz[first:first + 4], dtype=np.float32)
    locals_all = quaternion_matrix(
        np.asarray(local_rotation_xyzw[first:first + 4], dtype=np.float32))
    neutral = np.asarray(neutral_joints, dtype=np.float32)
    rest = np.zeros((JOINTS, 3), dtype=np.float32)
    for joint in range(JOINTS):
        parent = int(parents[joint])
        if parent >= 0:
            rest[joint] = neutral[joint] - neutral[parent]
    for joint in range(JOINTS):
        parent = int(parents[joint])
        if parent < 0:
            rotations[:, joint] = locals_all[:, joint]
            positions[:, joint] = roots4
        else:
            rotations[:, joint] = rotations[:, parent] @ locals_all[:, joint]
            positions[:, joint] = positions[:, parent] + (
                rotations[:, parent] @ rest[joint][None, :, None])[:, :, 0]
    for joint in (7, 14, 25, 33):
        rotations[:, joint] = IDENTITY
    return positions, rotations


def _encode_context_fast(root_xyz: np.ndarray, local_rotation_xyzw: np.ndarray,
                         neutral_joints: np.ndarray, parents: np.ndarray) -> EncodedFrames:
    """Batched :func:`_encode_context_ref`."""
    positions, rotations = _forward_kinematics_fast(
        root_xyz, local_rotation_xyzw, neutral_joints, parents)
    return _encode_global_frames_fast(positions, rotations, False)


def _local_rotations_fast(globals_all: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """``parent.T @ global`` per joint, batched on the leading axis.

    ``globals_all`` is ``(..., JOINTS, 3, 3)``; the result is
    ``matrix_xyzw`` of the parent-relative locals, so it is ``(..., JOINTS, 4)``.
    """
    parent_index, root_joints = _tree_parent_gather(parents)
    parent_matrices = np.swapaxes(globals_all, -1, -2)[..., parent_index, :, :]
    locals_all = parent_matrices @ globals_all
    if root_joints.size:
        locals_all[..., root_joints, :, :] = globals_all[..., root_joints, :, :]
    return matrix_xyzw(locals_all)


def _seed_agent_from_style_fast(style: Style,
                                parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched :func:`_seed_agent_from_style_ref`."""
    roots = np.ascontiguousarray(
        style.global_root_positions[0:4] + style.global_joint_positions[0:4, 0],
        dtype=np.float32)
    rotations = _local_rotations_fast(
        np.ascontiguousarray(style.global_joint_rotations[0:4], dtype=np.float32), parents)
    return roots, rotations


def _canonicalize_fast(root_xyz: np.ndarray, local_rotations_xyzw: np.ndarray,
                       command: Command,
                       ) -> tuple[WorldFrame, np.ndarray, np.ndarray, Command]:
    """Batched :func:`_canonicalize_ref` (same scalar tail for the command)."""
    first = root_xyz.shape[0] - 4
    first_root = root_xyz[first]
    root_matrix = quaternion_matrix(local_rotations_xyzw[first, 0])
    frame = WorldFrame(np.float32(first_root[0]), np.float32(first_root[2]),
                       np.float32(np.arctan2(root_matrix[0, 2], root_matrix[2, 2])))
    inverse_heading = y_rotation(-frame.heading)

    shifted = np.array(root_xyz, dtype=np.float32, copy=True)
    shifted[:, 0] -= frame.x
    shifted[:, 2] -= frame.z
    canonical_roots = np.ascontiguousarray(
        rotate_rows(inverse_heading, shifted), dtype=np.float32)
    canonical_rotations = np.array(local_rotations_xyzw, dtype=np.float32, copy=True)
    # Only the pelvis is rewritten: with parent-relative rotations a single root
    # rotation is exactly a whole-body rotation in FK, and rotating the remaining
    # joints as well would apply it twice (src/agent.cpp:165-168 has no inner
    # joint loop, exactly like restore_world).
    canonical_rotations[:, 0] = matrix_xyzw(inverse_heading @ quaternion_matrix(
        np.asarray(local_rotations_xyzw[:, 0], dtype=np.float32)))

    canonical_command = Command(
        style=command.style,
        movement=tuple(float(v) for v in inverse_heading @ np.array(command.movement, np.float32)),
        facing=tuple(float(v) for v in inverse_heading @ np.array(command.facing, np.float32)),
        target_speed=command.target_speed,
        world_target=command.world_target,
        world_target_heading=command.world_target_heading,
        has_world_target=command.has_world_target,
        skip_context_blend=command.skip_context_blend,
        seed=command.seed,
        sampling_argmax=command.sampling_argmax,
    )
    if command.has_world_target:
        target = np.array([command.world_target[0] - frame.x, command.world_target[1],
                           command.world_target[2] - frame.z], dtype=np.float32)
        canonical_command.world_target = tuple(float(v) for v in inverse_heading @ target)
        canonical_command.world_target_heading = command.world_target_heading - frame.heading
    return frame, canonical_roots, canonical_rotations, canonical_command


def _build_constraints_fast(roots: np.ndarray, rotations: np.ndarray, command: Command,
                            style: Style, neutral_joints: np.ndarray,
                            parents: np.ndarray) -> dict:
    """Batched :func:`_build_constraints_ref`.

    The scalar bookkeeping (velocity, travel, heading, the spring ramp) is
    unchanged and still runs one scalar iteration per frame; only the two
    per-joint blocks and the per-frame write-backs are batched.
    """
    context = _encode_context_fast(roots, rotations, neutral_joints, parents)
    current_x, current_z = context.global_root[0, 0], context.global_root[0, 2]
    velocity_x = np.float32((context.global_root[1, 0] - current_x) * FPS)
    velocity_z = np.float32((context.global_root[1, 2] - current_z) * FPS)

    direction_x, direction_z = np.float32(command.movement[0]), np.float32(command.movement[2])
    if np.hypot(direction_x, direction_z) <= np.float32(1.0e-5):
        direction_x = np.float32(command.facing[0] * 0.1)
        direction_z = np.float32(command.facing[2] * 0.1)
    travel = np.float32(command.target_speed * 2.0) if command.target_speed >= 0.0 \
        else np.float32(style.speed * 2.0)
    if command.target_speed < 0.0 and style.speed == 0.0:
        travel = np.float32(np.hypot(velocity_x, velocity_z) / 2.0)
    if travel <= np.float32(0.1):
        travel = np.float32(0.0)
    target_x = np.float32(command.world_target[0]) if command.has_world_target \
        else np.float32(current_x + travel * direction_x)
    target_z = np.float32(command.world_target[2]) if command.has_world_target \
        else np.float32(current_z + travel * direction_z)

    current_heading = np.arctan2(context.global_root[0, 4], context.global_root[0, 3])
    next_heading = np.arctan2(context.global_root[1, 4], context.global_root[1, 3])
    heading_velocity = np.float32(wrap(next_heading - current_heading) * FPS)
    desired_heading = np.float32(command.world_target_heading) if command.has_world_target \
        else np.arctan2(np.float32(command.facing[0]), np.float32(command.facing[2]))
    desired_heading = np.float32(current_heading + wrap(desired_heading - current_heading))

    start_root_x = np.empty(4, np.float32)
    start_root_z = np.empty(4, np.float32)
    start_heading = np.empty(4, np.float32)
    target_root_x = np.empty(4, np.float32)
    target_root_z = np.empty(4, np.float32)
    target_heading = np.empty(4, np.float32)
    for frame in range(4):
        start_time = np.float32(frame / 30.0)
        time = np.float32(1.0 + frame / 30.0)
        start_root_x[frame] = spring(current_x, velocity_x, target_x, 0.8, start_time)
        start_root_z[frame] = spring(current_z, velocity_z, target_z, 0.8, start_time)
        start_heading[frame] = spring(current_heading, heading_velocity, desired_heading,
                                      0.17, start_time)
        target_root_x[frame] = spring(current_x, velocity_x, target_x, 0.8, time)
        target_root_z[frame] = spring(current_z, velocity_z, target_z, 0.8, time)
        target_heading[frame] = spring(current_heading, heading_velocity, desired_heading,
                                       0.17, time)
    if style.speed == 0.0 and command.target_speed < 0.0 and not command.has_world_target:
        target_root_x[:] = target_root_x[0]
        target_root_z[:] = target_root_z[0]

    result = {
        "global_root": np.zeros((8, 5), np.float32),
        "local_root": np.zeros((8, 4), np.float32),
        "poses": np.zeros((8, 303), np.float32),
    }
    rotation_offset = (JOINTS - 1) * 3
    old_headings = np.arctan2(context.global_root[:, 4],
                              context.global_root[:, 3]).astype(np.float32)
    corrections = y_rotation_batch(start_heading - old_headings)
    pose_positions = context.poses[:, 0:99].reshape(4, JOINTS - 1, 3)
    context.poses[:, 0:99] = (corrections @ pose_positions.transpose(0, 2, 1)
                              ).transpose(0, 2, 1).reshape(4, 99)
    six = context.poses[:, rotation_offset:].reshape(4, JOINTS, 6)
    matrix_cont6d(corrections[:, None] @ cont6d_matrix(six), six)
    context.poses[:, rotation_offset:] = six.reshape(4, -1)
    context.global_root[:, 0] = start_root_x
    context.global_root[:, 2] = start_root_z
    context.global_root[:, 3] = np.cos(start_heading)
    context.global_root[:, 4] = np.sin(start_heading)
    context.local_root[0:3, 0] = wrap(start_heading[1:4] - start_heading[0:3]) * FPS
    context.local_root[0:3, 1] = (context.global_root[1:4, 0]
                                  - context.global_root[0:3, 0]) * FPS
    context.local_root[0:3, 2] = (context.global_root[1:4, 2]
                                  - context.global_root[0:3, 2]) * FPS
    context.local_root[0:3, 3] = context.global_root[0:3, 1]
    # std::copy writes only the first four frames; the rest stay zero until the
    # target block below fills frames 4..7 (src/agent.cpp:289-291).
    result["global_root"][0:4] = context.global_root
    result["local_root"][0:4] = context.local_root
    result["poses"][0:4] = context.poses

    first = 0 if style.frames == 4 else int(command.seed % (style.frames - 4))
    source_frames = first + np.arange(4)
    target_root_translations = np.zeros((4, 3), np.float32)
    target_local_rotations = np.zeros((4, JOINTS, 4), np.float32)
    corrections_t = y_rotation_batch(
        wrap(target_heading - style.global_headings[source_frames]))
    style_positions = np.ascontiguousarray(style.global_joint_positions[source_frames],
                                           dtype=np.float32)
    style_rotations = np.ascontiguousarray(style.global_joint_rotations[source_frames],
                                           dtype=np.float32)
    # The reference rotates each style joint position with a matrix-vector
    # product (src/agent.cpp:311-313).  A contraction reproduces that kernel bit
    # for bit; ``corrections_t @ style_positions[:, 1:].transpose(0, 2, 1)`` would
    # be a gemm and differs in the last bit on 3-element reductions (measured
    # 2.98e-08 on walk_stealth, which the pose planner then amplifies).
    target_positions = np.einsum("fij,fkj->fki", corrections_t, style_positions[:, 1:])
    global_root = result["global_root"][4:8]
    global_root[:, 0] = target_root_x
    global_root[:, 1] = (style.global_root_positions[source_frames, 1]
                         + style.global_joint_positions[source_frames, 0, 1])
    global_root[:, 2] = target_root_z
    global_root[:, 3] = np.cos(target_heading)
    global_root[:, 4] = np.sin(target_heading)
    target_root_translations[:] = global_root[:, 0:3]
    poses = result["poses"][4:8]
    poses[:, 0:(JOINTS - 1) * 3] = target_positions.reshape(4, (JOINTS - 1) * 3)
    target_globals = corrections_t[:, None] @ style_rotations
    six = poses[:, rotation_offset:].reshape(4, JOINTS, 6)
    matrix_cont6d(target_globals, six)
    poses[:, rotation_offset:] = six.reshape(4, -1)
    target_local_rotations[:] = _local_rotations_fast(target_globals, parents)
    local_root = result["local_root"][4:8]
    local_root[0:3, 0] = wrap(target_heading[1:4] - target_heading[0:3]) * FPS
    local_root[0:3, 1] = (result["global_root"][5:8, 0]
                          - result["global_root"][4:7, 0]) * FPS
    local_root[0:3, 2] = (result["global_root"][5:8, 2]
                          - result["global_root"][4:7, 2]) * FPS
    local_root[0:3, 3] = result["global_root"][4:7, 1]
    result["local_root"][7] = result["local_root"][6]
    result["local_root"][7, 3] = result["global_root"][7, 1]

    result["has_local_root"] = np.array([1, 1, 1, 0, 1, 1, 1, 1], np.uint8)
    result["allowed_tokens"] = style.allowed_tokens.copy()
    result["target_root_translations"] = target_root_translations
    result["target_local_rotations_xyzw"] = target_local_rotations
    return result


def _restore_world_fast(motion: Motion, frame: WorldFrame) -> None:
    """Batched :func:`_restore_world_ref`, in place."""
    first_matrix = quaternion_matrix(motion.rotations_xyzw[0, 0])
    first_heading = np.arctan2(first_matrix[0, 2], first_matrix[2, 2])
    correction = y_rotation(frame.heading - first_heading)
    shifted = np.array(motion.roots, dtype=np.float32, copy=True)
    shifted[:, 0] -= motion.roots[0, 0]
    shifted[:, 2] -= motion.roots[0, 2]
    rotated = rotate_rows(correction, shifted)
    motion.roots[:, 0] = rotated[:, 0] + frame.x
    motion.roots[:, 1] = rotated[:, 1]
    motion.roots[:, 2] = rotated[:, 2] + frame.z
    # Only the pelvis is rewritten: the stored rotations are parent-relative, so
    # rotating joint 0 rotates the whole body in FK and rotating the other joints
    # as well would double-apply the correction.  This mirrors
    # src/agent.cpp:194-196, which has no inner joint loop.
    motion.rotations_xyzw[:, 0] = matrix_xyzw(
        correction @ quaternion_matrix(motion.rotations_xyzw[:, 0]))
    target_correction = y_rotation(frame.heading)
    rotated_targets = rotate_rows(target_correction, motion.target_roots)
    motion.target_roots[:, 0] = rotated_targets[:, 0] + frame.x
    motion.target_roots[:, 1] = rotated_targets[:, 1]
    motion.target_roots[:, 2] = rotated_targets[:, 2] + frame.z
    motion.target_local_rotations_xyzw[:, 0] = matrix_xyzw(
        target_correction @ quaternion_matrix(motion.target_local_rotations_xyzw[:, 0]))


def _slerp_batch(source: np.ndarray, target: np.ndarray,
                 amount: np.float32) -> np.ndarray:
    """``slerp`` over ``(..., 4)`` quaternions, both branches evaluated.

    ``slerp`` chooses between a lerp and a spherical interpolation from the
    per-element ``dot``; the batch computes both and selects per element, so each
    element takes exactly the branch it would have taken one at a time.
    """
    right = np.array(target, dtype=np.float32, copy=True)
    dot = dot_rows(source, right)
    negative = dot < 0.0
    dot = np.where(negative, -dot, dot)
    right = np.where(negative[..., None], -right, right)
    dot = np.clip(dot.astype(np.float32), np.float32(-1.0), np.float32(1.0))
    lerp = source + amount * (right - source)
    # ``dot == 1`` makes the spherical branch 0/0; those elements take the lerp
    # branch, so the NaN is discarded by the selection below -- the same reason
    # ``matrix_xyzw`` runs under errstate.
    with np.errstate(divide="ignore", invalid="ignore"):
        angle = np.arccos(dot)
        denominator = np.sin(angle)
        left_weight = np.sin((1.0 - amount) * angle) / denominator
        right_weight = np.sin(amount * angle) / denominator
        spherical = left_weight[..., None] * source + right_weight[..., None] * right
    result = np.where((dot > 0.9995)[..., None], lerp, spherical).astype(np.float32)
    return (result / np.sqrt(np.sum(result * result, axis=-1, keepdims=True))).astype(np.float32)


def _apply_context_blend_fast(roots: np.ndarray, rotations: np.ndarray,
                              motion: Motion) -> None:
    """Batched :func:`_apply_context_blend_ref`, in place."""
    weights = np.array([0.3, 0.433333337, 0.566666663, 0.7], dtype=np.float32)
    begin = roots.shape[0] - 4
    context_root = np.asarray(roots[begin:begin + 4], dtype=np.float32)
    motion.roots[0:4] = (context_root * (np.float32(1.0) - weights)[:, None]
                         + motion.roots[0:4] * weights[:, None]).astype(np.float32)
    joints = np.array([joint for joint in range(1, JOINTS) if physical_g1_joint(joint)],
                      dtype=np.intp)
    source = np.ascontiguousarray(rotations[begin:begin + 4, joints], dtype=np.float32)
    target = np.ascontiguousarray(motion.rotations_xyzw[0:4, joints], dtype=np.float32)
    blended = np.empty_like(source)
    for frame in range(4):
        blended[frame] = _slerp_batch(source[frame], target[frame], weights[frame])
    motion.rotations_xyzw[0:4, joints] = blended


# --- public dispatchers ----------------------------------------------------
# The names below are what the rest of the port (and the tests) calls; each one
# selects the batched implementation or the verbatim reference one.

def encode_global_frames(joint_positions: np.ndarray, joint_rotations: np.ndarray,
                         repeat_last_velocity: bool) -> EncodedFrames:
    """``encode_global_frames`` (src/motion_rep.cpp:121-178)."""
    if _agent_math_option():
        return _encode_global_frames_fast(joint_positions, joint_rotations,
                                          repeat_last_velocity)
    return _encode_global_frames_ref(joint_positions, joint_rotations, repeat_last_velocity)


def encode_context(root_xyz: np.ndarray, local_rotation_xyzw: np.ndarray,
                   neutral_joints: np.ndarray, parents: np.ndarray) -> EncodedFrames:
    """``encode_context`` (src/motion_rep.cpp:180-250)."""
    if _agent_math_option():
        return _encode_context_fast(root_xyz, local_rotation_xyzw, neutral_joints, parents)
    return _encode_context_ref(root_xyz, local_rotation_xyzw, neutral_joints, parents)


def seed_agent_from_style(style: Style, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``seed_agent_from_style`` (src/agent.cpp:381-417)."""
    if _agent_math_option():
        return _seed_agent_from_style_fast(style, parents)
    return _seed_agent_from_style_ref(style, parents)


def canonicalize(root_xyz: np.ndarray, local_rotations_xyzw: np.ndarray, command: Command,
                 ) -> tuple[WorldFrame, np.ndarray, np.ndarray, Command]:
    """``canonicalize`` (src/agent.cpp:147-181)."""
    if _agent_math_option():
        return _canonicalize_fast(root_xyz, local_rotations_xyzw, command)
    return _canonicalize_ref(root_xyz, local_rotations_xyzw, command)


def build_constraints(roots: np.ndarray, rotations: np.ndarray, command: Command,
                      style: Style, neutral_joints: np.ndarray,
                      parents: np.ndarray) -> dict:
    """``build_constraints`` (src/agent.cpp:209-350)."""
    if _agent_math_option():
        return _build_constraints_fast(roots, rotations, command, style, neutral_joints,
                                       parents)
    return _build_constraints_ref(roots, rotations, command, style, neutral_joints, parents)


def restore_world(motion: Motion, frame: WorldFrame) -> None:
    """``restore_world`` (src/agent.cpp:183-207), in place."""
    if _agent_math_option():
        return _restore_world_fast(motion, frame)
    return _restore_world_ref(motion, frame)


def apply_context_blend(roots: np.ndarray, rotations: np.ndarray, motion: Motion) -> None:
    """``apply_context_blend`` (src/agent.cpp:354-379), in place."""
    if _agent_math_option():
        return _apply_context_blend_fast(roots, rotations, motion)
    return _apply_context_blend_ref(roots, rotations, motion)


class Agent:
    """``mb_agent`` (src/handles.hpp:60-68) driving one MLX pipeline."""

    def __init__(self, pipeline, neutral_joints: np.ndarray, parents: np.ndarray):
        self.pipeline = pipeline
        self.neutral_joints = np.ascontiguousarray(neutral_joints, dtype=np.float32)
        self.parents = np.ascontiguousarray(parents, dtype=np.int32)
        self.context_roots: np.ndarray | None = None
        self.context_rotations: np.ndarray | None = None
        self.current_motion: Motion | None = None
        self.current_frame = 0

    @property
    def context_frames(self) -> int:
        return 0 if self.context_roots is None else int(self.context_roots.shape[0])

    def reset(self, style: Style) -> None:
        self.context_roots, self.context_rotations = seed_agent_from_style(style, self.parents)
        self.current_motion = None
        self.current_frame = 0

    def advance(self, frames: int) -> None:
        """``advance_agent`` (src/agent.cpp:547-568)."""
        if self.current_motion is None:
            raise ValueError("agent has no planned motion")
        total = self.current_motion.frames
        self.current_frame = min(self.current_frame + frames, total - 1)
        roots = np.empty((4, 3), np.float32)
        rotations = np.empty((4, JOINTS, 4), np.float32)
        for frame in range(4):
            source = min(self.current_frame + frame, total - 1)
            roots[frame] = self.current_motion.roots[source]
            rotations[frame] = self.current_motion.rotations_xyzw[source]
        self.context_roots, self.context_rotations = roots, rotations

    def plan(self, command: Command, style: Style,
             capture_trace: bool = False,
             pose_logits=None) -> tuple[Motion, dict]:
        """``plan_agent_impl`` (src/agent.cpp:419-455).

        ``capture_trace`` forwards to :meth:`MlxPipeline.run_transition` so a
        caller can diff the discrete ``pose.tokens``; it is off by default
        because it copies every intermediate stage.  ``pose_logits`` is the
        round-3 bake seam (see ``MlxPipeline.run_transition``): a batch bake
        supplies the pose planner's output and the stage is skipped, while all
        the surrounding agent maths stays the accepted code.
        """
        if self.context_frames < 4:
            self.reset(style)
        assert self.context_roots is not None and self.context_rotations is not None
        frame, canonical_roots, canonical_rotations, canonical_command = canonicalize(
            self.context_roots, self.context_rotations, command)
        constraints = build_constraints(canonical_roots, canonical_rotations,
                                        canonical_command, style, self.neutral_joints,
                                        self.parents)
        from ..models.transition import TransitionConstraints
        result = self.pipeline.run_transition(
            TransitionConstraints(**{key: constraints[key] for key in (
                "global_root", "local_root", "poses", "has_local_root",
                "allowed_tokens", "target_root_translations",
                "target_local_rotations_xyzw")},
                has_global_root=np.ones(8, np.uint8),
                has_poses=np.ones(8, np.uint8)),
            seed=command.seed, sampling_argmax=command.sampling_argmax,
            capture_trace=capture_trace, pose_logits=pose_logits)
        motion = Motion(result.roots, result.rotations_xyzw,
                        constraints["target_root_translations"],
                        constraints["target_local_rotations_xyzw"])
        restore_world(motion, frame)
        if not command.skip_context_blend:
            apply_context_blend(self.context_roots, self.context_rotations, motion)
        self.current_motion = motion
        self.current_frame = 0
        return motion, {"selected_tokens": result.selected_tokens, "trace": result.trace}
