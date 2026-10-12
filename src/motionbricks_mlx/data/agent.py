"""The agent layer: context encoding, constraint building and playback seams.

Port of the parts of ``src/agent.cpp`` and ``src/motion_rep.cpp`` that sit
between a style asset + a command and the neural transition, plus the
post-processing the reference applies to the network's output.

All maths here is NumPy float32 on small arrays, mirroring the reference's
scalar C++ float loops one to one; the neural stages themselves stay in MLX.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .style import Style

FPS = np.float32(30.0)
TWO_PI = np.float32(2.0 * np.pi)
JOINTS = 34
IDENTITY = np.eye(3, dtype=np.float32)


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


def encode_global_frames(joint_positions: np.ndarray, joint_rotations: np.ndarray,
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


def encode_context(root_xyz: np.ndarray, local_rotation_xyzw: np.ndarray,
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
    return encode_global_frames(positions, rotations, False)


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


def seed_agent_from_style(style: Style, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def canonicalize(root_xyz: np.ndarray, local_rotations_xyzw: np.ndarray,
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


def build_constraints(roots: np.ndarray, rotations: np.ndarray, command: Command,
                      style: Style, neutral_joints: np.ndarray,
                      parents: np.ndarray) -> dict:
    """``build_constraints`` (src/agent.cpp:209-350).

    Returns a dict of raw constraint arrays consumed by ``TransitionConstraints``.
    """
    context = encode_context(roots, rotations, neutral_joints, parents)
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


def restore_world(motion: Motion, frame: WorldFrame) -> None:
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


def apply_context_blend(roots: np.ndarray, rotations: np.ndarray, motion: Motion) -> None:
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

    def plan(self, command: Command, style: Style) -> tuple[Motion, dict]:
        """``plan_agent_impl`` (src/agent.cpp:419-455)."""
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
            seed=command.seed, sampling_argmax=command.sampling_argmax)
        motion = Motion(result.roots, result.rotations_xyzw,
                        constraints["target_root_translations"],
                        constraints["target_local_rotations_xyzw"])
        restore_world(motion, frame)
        if not command.skip_context_blend:
            apply_context_blend(self.context_roots, self.context_rotations, motion)
        self.current_motion = motion
        self.current_frame = 0
        return motion, {"selected_tokens": result.selected_tokens, "trace": result.trace}
