"""User-placed target keyframes: the *external* half of the transition boundary.

Where this sits in the data path
--------------------------------
The reference transition boundary is a fixed **8-frame window**
(``src/root.cpp:366`` "4 start + 4 end", ``BOUNDARY_FRAMES = 8``): frames 0..3
are the current state and frames 4..7 are the goal.  ``build_constraints``
(``src/agent.cpp:209-350``, ported as
:func:`motionbricks_mlx.data.agent._build_constraints_ref`) *manufactures*
frames 4..7 from the style clip and a spring toward a command-derived goal, and
the root planner then copies exactly those 4 frames back into the ``dense_target``
tensor that conditions the root conv stack (``src/root.cpp:463-479``) -- so the
"target" the model conditions on **is** frames 4..7 of the boundary.

This module replaces *only* frames 4..7, after ``build_constraints`` has run, so
the accepted default path (``build_constraints`` unchanged, no override applied)
is untouched.  Three things are replaced per target frame:

* ``global_root`` -- ``[x, y, z, cos(heading), sin(heading)]``.  ``x`` / ``z`` are
  the placed point, ``heading`` is the placed facing, ``y`` stays the style's
  target pelvis height (or the current one under ``pose_mode="hold"``).
* ``poses`` -- the 303-wide external pose block (33 joint offsets + 34 six-D
  rotations, ``EXTERNAL_POSE_WIDTH``).  Two modes: keep the style's own
  configuration at that slot (``"style"``) or hold the configuration the clip
  starts from (``"hold"``); either way the block is yaw-rotated by the *same*
  angle the heading changed by, which is what the reference does with
  ``correction = wrap(target_heading[frame] - style.global_headings[source])``
  (``src/agent.cpp:277-280``).
* ``target_root_translations`` / ``target_local_rotations_xyzw`` -- the exported
  metadata that the page draws as the orange ghosts, so the ghost you see is
  exactly the window you asked for.

Alignment rule (measured, not assumed)
--------------------------------------
The 4 window samples are compared against the **last 4 frames of the generated
clip**.  That is a measurement, not a reading of the reference: on the accepted
default path the argmin of ``|generated root - window root|`` over every offset
is ``frames-4`` for 16/16 walk-family scenarios (styles ``walk``,
``walk_zombie``, ``slow_walk``, ``walk_left`` x seeds 1/2/7/77); the static
``idle`` style has no identifiable argmin because its whole root track moves
< 8 mm.  ``tests/test_mlx_targets.py`` re-derives that argmin so the rule cannot
silently rot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..export.plan_json import world_positions_xyz
from ..models.schema import EXTERNAL_POSE_WIDTH
from .agent import (JOINTS, Agent, Command, Motion, WorldFrame, apply_context_blend,
                    cont6d_matrix, matrix_cont6d, matrix_xyzw, quaternion_matrix,
                    restore_world, y_rotation)

#: How many samples the model's target window has (``BOUNDARY_FRAMES // 2``).
WINDOW_FRAMES = 4
#: ``build_constraints`` lays the target block at frames 4..7 (``src/root.cpp:366``).
WINDOW_OFFSET = 4
#: Joint-position block width inside the external pose (34 joints - 1 root) * 3.
POSITION_BLOCK = (JOINTS - 1) * 3
#: Where the 34 six-D joint rotations start inside the external pose.
ROTATION_OFFSET = (JOINTS - 1) * 3

#: Accepted errors for a placed target.  These are *behavioural* thresholds, not
#: the port-vs-reference numerical budget of ``scripts/mlx_align.py``
#: (1.0e-3 m / 2.0e-3 / 2.0e-1 deg), so they get their own table with its own
#: reason -- but the same measurement style: max absolute error, metres for
#: positions and degrees for angles, worst case over the compared block.
#:
#: Calibration: on the accepted default path (the model's *own* target window,
#: which is by construction reachable) the same metric reads 1.4e-3 m at a
#: standstill and <= 1.97e-2 m for a 44-frame walk clip.  A placed target is a
#: new request rather than an equivalence check, so the threshold is an order of
#: magnitude above that floor: 0.15 m root, 0.15 m joint, 5 deg heading.  It is
#: falsifiable on purpose -- see the report's distance sweep, where 3 m passes
#: (0.094 m) and 5 m does not (0.225 m).
TARGET_TOLERANCES = {
    "root_planar_m": 0.15,
    "heading_deg": 5.0,
    "joint_m": 0.15,
}


@dataclass(frozen=True)
class PlacedTarget:
    """One user-placed target keyframe, in **world** coordinates.

    ``x`` / ``z`` are world metres -- what a click on the ground gives -- and
    ``heading_deg`` is a world facing in degrees (0 = +z).  ``pose`` is an
    optional 303-wide external pose block *in the canonical frame at the current
    heading*; when omitted the configuration comes from ``pose_mode``.
    """

    x: float
    z: float
    heading_deg: float = 0.0
    pose: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not np.isfinite([self.x, self.z, self.heading_deg]).all():
            raise ValueError(f"placed target is not finite: {self!r}")
        if self.pose is not None:
            pose = np.ascontiguousarray(self.pose, dtype=np.float32)
            if pose.shape != (EXTERNAL_POSE_WIDTH,):
                raise ValueError(
                    f"pose must have {EXTERNAL_POSE_WIDTH} values, got {pose.shape}")
            if not np.isfinite(pose).all():
                raise ValueError("pose contains non-finite values")
            object.__setattr__(self, "pose", pose)


def parse_target(text: str) -> PlacedTarget:
    """``"x,z[,heading_deg]"`` -> :class:`PlacedTarget` (CLI / query-string form)."""
    parts = [piece for piece in str(text).replace(";", ",").split(",")]
    if len(parts) not in (2, 3):
        raise ValueError(f"expected 'x,z[,heading_deg]', got {text!r}")
    values = [float(piece) for piece in parts]
    return PlacedTarget(values[0], values[1], values[2] if len(values) == 3 else 0.0)


def parse_targets(raw) -> list[PlacedTarget]:
    """Parse a list of targets from JSON-ish or query-string input.

    Accepts a list of :class:`PlacedTarget` / dicts / ``[x, z, heading_deg]``
    triples / ``"x,z,deg"`` strings, or a single ``"x,z,deg;x,z,deg"`` string with
    ``;`` separators.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        pieces = [piece.strip() for piece in raw.split(";") if piece.strip()]
        return [parse_target(piece) for piece in pieces]
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"expected a target list, got {type(raw).__name__}")
    targets: list[PlacedTarget] = []
    for item in raw:
        if isinstance(item, PlacedTarget):
            targets.append(item)
        elif isinstance(item, str):
            targets.append(parse_target(item))
        elif isinstance(item, (list, tuple, np.ndarray)):
            # The page posts ``[[x, z, heading_deg], ...]`` -- a positional
            # triple, which is what a click produces.
            values = [float(value) for value in item]
            if len(values) not in (2, 3):
                raise ValueError(f"expected [x, z[, heading_deg]], got {item!r}")
            targets.append(PlacedTarget(values[0], values[1],
                                       values[2] if len(values) == 3 else 0.0))
        elif isinstance(item, dict):
            pose = item.get("pose")
            targets.append(PlacedTarget(
                float(item.get("x", item.get("z_offset", 0.0))),
                float(item.get("z", 0.0)),
                float(item.get("heading_deg", item.get("heading", 0.0))),
                None if pose is None else np.asarray(pose, dtype=np.float32)))
        else:
            raise ValueError(f"unsupported target {item!r}")
    return targets


@dataclass
class TargetWindow:
    """What was asked for: the replaced frames 4..7, in both frames of reference."""

    roots: np.ndarray                 # (4, 3) canonical, exactly what entered the graph
    headings: np.ndarray              # (4,) canonical radians
    world_roots: np.ndarray           # (4, 3) world (x, y kept from the canonical slot)
    world_headings_deg: np.ndarray    # (4,) world degrees
    pose_mode: str
    placed: list[PlacedTarget] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return int(self.roots.shape[0])


def _rotate_pose(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Yaw-rotate one 303-wide external pose block by ``delta``.

    Same two blocks the reference rotates (``src/agent.cpp:270-286``): the 33
    joint offsets, then each joint's six-D rotation.  ``delta`` is a yaw matrix
    from :func:`~motionbricks_mlx.data.agent.y_rotation`.
    """
    out = np.array(pose, dtype=np.float32, copy=True)
    out[0:POSITION_BLOCK] = (delta @ out[0:POSITION_BLOCK].reshape(JOINTS - 1, 3).T) \
        .T.reshape(-1)
    for joint in range(JOINTS):
        start = ROTATION_OFFSET + joint * 6
        six = out[start:start + 6].copy()
        matrix_cont6d(delta @ cont6d_matrix(six), six)
        out[start:start + 6] = six
    return out


def _block(constraints, name: str) -> np.ndarray:
    """One boundary block, from either a dict or a ``TransitionConstraints``."""
    if isinstance(constraints, dict):
        return constraints[name]
    return getattr(constraints, name)


def apply_placed_targets(constraints, targets: list[PlacedTarget], frame: WorldFrame,
                         pose_mode: str = "style") -> TargetWindow:
    """Replace boundary frames 4..7 with user-placed targets, in place.

    ``constraints`` is a
    :class:`~motionbricks_mlx.models.transition.TransitionConstraints` (or the
    plain dict :func:`motionbricks_mlx.data.agent.build_constraints` returns) in
    the canonical frame; ``frame`` is the :class:`WorldFrame` ``canonicalize``
    produced for the same call, which is what converts world coordinates into
    that canonical frame.

    Filling rule: the *n*-th placed target becomes window sample *n*; when fewer
    than 4 are placed the last one is **held** for the remaining samples.  A
    single placed target therefore asks for one constant goal across the window,
    which is both the best-hitting shape measured here and the shape the
    reference itself uses for a stationary style (``src/agent.cpp:246-249``).
    More than 4 targets is rejected rather than silently truncated.
    """
    if pose_mode not in ("style", "hold"):
        raise ValueError(f"pose_mode must be 'style' or 'hold', got {pose_mode!r}")
    if not targets:
        raise ValueError("at least one placed target is required")
    if len(targets) > WINDOW_FRAMES:
        raise ValueError(f"at most {WINDOW_FRAMES} placed targets, got {len(targets)}")

    inverse_heading = y_rotation(np.float32(-frame.heading))
    start = _block(constraints, "global_root")[0].copy()
    start_heading = float(np.arctan2(start[4], start[3]))
    start_pose = _block(constraints, "poses")[0].copy()
    # Read before the loop overwrites the block slot by slot.
    base_target_rotations = np.array(_block(constraints, "target_local_rotations_xyzw"),
                                     dtype=np.float32, copy=True)

    roots = np.zeros((WINDOW_FRAMES, 3), np.float32)
    headings = np.zeros(WINDOW_FRAMES, np.float32)
    world_roots = np.zeros((WINDOW_FRAMES, 3), np.float32)
    world_headings = np.zeros(WINDOW_FRAMES, np.float32)

    for slot in range(WINDOW_FRAMES):
        placed = targets[min(slot, len(targets) - 1)]
        index = WINDOW_OFFSET + slot
        global_root = _block(constraints, "global_root")[index]
        previous_heading = float(np.arctan2(global_root[4], global_root[3]))

        # world -> canonical, exactly the transform canonicalize applies to the
        # context and to command.world_target (src/agent.cpp:171-180).
        offset = np.array([placed.x - frame.x, 0.0, placed.z - frame.z], dtype=np.float32)
        canonical = inverse_heading @ offset
        heading = np.float32(np.deg2rad(placed.heading_deg) - frame.heading)

        global_root[0] = np.float32(start[0] + canonical[0])
        global_root[2] = np.float32(start[2] + canonical[2])
        global_root[3] = np.cos(heading)
        global_root[4] = np.sin(heading)

        if placed.pose is not None:
            global_root[1] = np.float32(start[1])
            delta = y_rotation(np.float32(heading - start_heading))
            _block(constraints, "poses")[index] = _rotate_pose(placed.pose, delta)
        elif pose_mode == "hold":
            global_root[1] = np.float32(start[1])
            delta = y_rotation(np.float32(heading - start_heading))
            _block(constraints, "poses")[index] = _rotate_pose(start_pose, delta)
        else:
            delta = y_rotation(np.float32(heading - previous_heading))
            _block(constraints, "poses")[index] = _rotate_pose(_block(constraints, "poses")[index], delta)

        _block(constraints, "target_root_translations")[slot] = global_root[0:3]
        # Only the root joint's local rotation is a world rotation
        # (src/agent.cpp:392-396: joint 0 has no parent), so a body yaw touches
        # exactly that one, the same way restore_world corrects it.
        _block(constraints, "target_local_rotations_xyzw")[slot, 0] = matrix_xyzw(
            delta @ quaternion_matrix(base_target_rotations[slot, 0]))

        roots[slot] = global_root[0:3]
        headings[slot] = heading
        world_roots[slot] = np.array([placed.x, global_root[1], placed.z], np.float32)
        world_headings[slot] = np.float32(np.rad2deg(heading + frame.heading))

    # local_root is a finite difference of the window, as in the reference
    # (src/agent.cpp:322-330): [heading_rate, dx*30, dz*30, y]; the last slot
    # repeats the previous one (src/agent.cpp:331-334).
    for slot in range(WINDOW_FRAMES - 1):
        current = _block(constraints, "global_root")[WINDOW_OFFSET + slot]
        following = _block(constraints, "global_root")[WINDOW_OFFSET + slot + 1]
        _block(constraints, "local_root")[WINDOW_OFFSET + slot] = [
            np.float32(0.0),
            np.float32((following[0] - current[0]) * 30.0),
            np.float32((following[2] - current[2]) * 30.0),
            current[1],
        ]
    _block(constraints, "local_root")[7] = _block(constraints, "local_root")[6]
    _block(constraints, "local_root")[7, 3] = _block(constraints, "global_root")[7, 1]

    return TargetWindow(roots=roots, headings=headings, world_roots=world_roots,
                        world_headings_deg=world_headings, pose_mode=pose_mode,
                        placed=list(targets))


def plan_with_targets(agent: Agent, command: Command, style, targets: list[PlacedTarget],
                      pose_mode: str = "style", capture_trace: bool = False
                      ) -> tuple[Motion, dict, TargetWindow]:
    """``Agent.plan`` with boundary frames 4..7 replaced by placed targets.

    Deliberately a free function calling the *same* three public agent steps
    ``Agent.plan`` calls (``constraints_for`` -> ``pipeline.run_transition`` ->
    ``finish_plan``), so it cannot drift from the accepted path and cannot change
    it: when no target is placed, callers keep using ``Agent.plan``, whose bytes
    this module never touches.
    """
    if agent.context_frames < 4:
        agent.reset(style)
    frame, constraints = agent.constraints_for(command, style)
    window = apply_placed_targets(constraints, targets, frame, pose_mode)
    result = agent.pipeline.run_transition(constraints, seed=command.seed,
                                           sampling_argmax=command.sampling_argmax,
                                           capture_trace=capture_trace)
    motion, info = agent.finish_plan(result, frame, constraints, command)
    return motion, info, window


# ---------------------------------------------------------------------------
# hit measurement
# ---------------------------------------------------------------------------
def aligned_slice(frames: int) -> slice:
    """The output frames a target window is compared against (the last 4).

    See the module docstring: this is the measured argmin of the root error on
    the accepted default path, not an assumption about the reference.
    """
    if frames < WINDOW_FRAMES:
        raise ValueError(f"a {frames}-frame clip cannot carry a {WINDOW_FRAMES}-sample window")
    return slice(frames - WINDOW_FRAMES, frames)


def clip_start_world(agent: Agent) -> tuple[float, float, float]:
    """World ``(x, z, heading_rad)`` the *next* plan will start from.

    ``canonicalize`` takes its world frame from ``context_roots[-4]``, i.e. the
    first frame of the current context, and ``Agent.advance`` rewrites that
    context to start at the frame the caller has already shown -- so this is the
    pose the page is displaying, which is what a click on the ground has to be
    measured against.
    """
    if agent.context_frames < 4:
        raise ValueError("agent has no 4-frame context yet")
    root = np.asarray(agent.context_roots[0], dtype=np.float64)
    matrix = quaternion_matrix(np.asarray(agent.context_rotations[0, 0], dtype=np.float64))
    return float(root[0]), float(root[2]), float(np.arctan2(matrix[0, 2], matrix[2, 2]))


def resolve_world_targets(agent: Agent, raw, space: str = "world") -> list[PlacedTarget]:
    """Parse ``raw`` and, for ``space == "offset"``, place it from the clip start.

    ``offset`` interprets each ``x, z`` as metres along the character's own right
    and forward axes, which is how the page expresses "3 m in front"; ``world``
    takes them as absolute world metres (what a raycast onto the floor gives).
    """
    targets = parse_targets(raw)
    if space not in ("world", "offset"):
        raise ValueError(f"target_space must be 'world' or 'offset', got {space!r}")
    if space == "world" or not targets:
        return targets
    origin_x, origin_z, heading = clip_start_world(agent)
    cosine, sine = float(np.cos(heading)), float(np.sin(heading))
    resolved: list[PlacedTarget] = []
    for target in targets:
        resolved.append(PlacedTarget(
            origin_x + target.x * cosine + target.z * sine,
            origin_z - target.x * sine + target.z * cosine,
            float(np.rad2deg(heading)) + target.heading_deg,
            target.pose))
    return resolved


def _heading_of(rotations_xyzw: np.ndarray, frames: slice | None = None) -> np.ndarray:
    """Body yaw in radians, from the root joint's world rotation (joint 0)."""
    block = rotations_xyzw[frames] if frames is not None else rotations_xyzw
    out = np.empty(block.shape[0], dtype=np.float64)
    for index in range(block.shape[0]):
        matrix = quaternion_matrix(block[index, 0])
        out[index] = np.arctan2(matrix[0, 2], matrix[2, 2])
    return out


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    """Smallest signed angle in degrees, in ``[-180, 180)``."""
    return (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0


def measure_hit(motion: Motion, window: TargetWindow, parents: np.ndarray,
                rest: np.ndarray) -> dict:
    """Errors between the generated clip and the placed target window.

    Reported in ``scripts/mlx_align.py``'s units -- max absolute error, metres
    for positions and degrees for angles, worst case over the compared block --
    against :data:`TARGET_TOLERANCES`.
    """
    frames = motion.frames
    window_slice = aligned_slice(frames)
    generated_roots = np.asarray(motion.roots, dtype=np.float64)[window_slice]
    target_roots = np.asarray(window.roots, dtype=np.float64)

    delta = generated_roots - target_roots
    root_max_abs_m = float(np.abs(delta[:, [0, 2]]).max())
    root_planar_m = float(np.hypot(delta[:, 0], delta[:, 2]).max())

    generated_headings = _heading_of(np.asarray(motion.rotations_xyzw, dtype=np.float64),
                                    window_slice)
    heading_err = wrap_degrees(np.rad2deg(generated_headings)) \
        - wrap_degrees(np.rad2deg(window.headings))
    heading_deg = float(np.abs(wrap_degrees(heading_err)).max())

    generated_world = world_positions_xyz(
        np.asarray(motion.roots, dtype=np.float64)[window_slice],
        np.asarray(motion.rotations_xyzw, dtype=np.float64)[window_slice], parents, rest)
    target_world = world_positions_xyz(
        np.asarray(motion.target_roots, dtype=np.float64),
        np.asarray(motion.target_local_rotations_xyzw, dtype=np.float64), parents, rest)
    joint_m = float(np.linalg.norm(generated_world - target_world, axis=-1).max())

    measured = {"root_max_abs_m": root_max_abs_m, "root_planar_m": root_planar_m,
                "heading_deg": heading_deg, "joint_m": joint_m}
    failures = [name for name, tolerance in TARGET_TOLERANCES.items()
                if measured[name] > tolerance]
    # Per-slot breakdown: a *moving* window is tracked with a lag (measured; see
    # the report), so one max would hide that the last slot usually lands while
    # the earlier ones trail.  Reported, not separately scored.
    per_slot = [
        {"slot": slot,
         "output_frame": int(frames - WINDOW_FRAMES + slot),
         "root_planar_m": float(np.hypot(delta[slot, 0], delta[slot, 2])),
         "heading_deg": float(abs(wrap_degrees(heading_err[slot]))),
         "joint_m": float(np.linalg.norm(generated_world[slot] - target_world[slot],
                                         axis=-1).max())}
        for slot in range(WINDOW_FRAMES)
    ]
    return {
        "frames": int(frames),
        "window_frames": [int(frames - WINDOW_FRAMES), int(frames - 1)],
        "pose_mode": window.pose_mode,
        "placed": [{"x": float(t.x), "z": float(t.z), "heading_deg": float(t.heading_deg)}
                   for t in window.placed],
        "window_world_roots": [[float(v) for v in row] for row in window.world_roots],
        "window_world_headings_deg": [float(v) for v in window.world_headings_deg],
        "errors": measured,
        "per_slot": per_slot,
        "tolerances": dict(TARGET_TOLERANCES),
        "within_tolerance": not failures,
        "exceeded": failures,
    }


__all__ = ["PlacedTarget", "TargetWindow", "WINDOW_FRAMES", "WINDOW_OFFSET",
           "TARGET_TOLERANCES", "aligned_slice", "apply_placed_targets",
           "clip_start_world", "measure_hit", "parse_target", "parse_targets",
           "plan_with_targets", "resolve_world_targets", "world_positions_xyz",
           "apply_context_blend", "restore_world"]
