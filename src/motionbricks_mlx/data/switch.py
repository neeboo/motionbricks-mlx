"""A -> B switching as a first-class operation: condition B on A's *current state*.

What "a switch" is, mechanically
--------------------------------
The model is a keyframe-conditioned generator over a **fixed 8-frame boundary**
(``models/schema.py``, ``BOUNDARY_FRAMES = 8``; reference ``src/root.cpp:366``
"4 start + 4 end"):

===========  ==========================================================
frames 0..3  the current state, from :func:`~motionbricks_mlx.data.agent.encode_context`
frames 4..7  the goal, manufactured by ``build_constraints`` from the style clip
===========  ==========================================================

``local_root`` (4 wide) is the **finite difference of that window**
(``src/agent.cpp:322-334``), so within the window the model can see position and
one difference -- **velocity**.  There is no acceleration channel: four samples
carry a first difference and nothing higher.

That is the whole basis of this module, and it is why a switch has a clean
definition:

* **seamless switch** -- plan B while the agent's context still *is* A's last four
  frames (``Agent.advance`` to the switch frame, then ``Agent.plan(command, B)``).
  Same ``plan`` code path, no reset.  B's target window (frames 4..7) comes from
  B's clip; its start window (frames 0..3), its ``local_root`` velocities and the
  inherited ``apply_context_blend`` all come from A.  Nothing new is computed --
  see :meth:`Switcher.seamless`.
* **naive hard cut** -- generate A and B *independently* (each from its own
  style reseed, which is exactly what the shipped viewer did on a style change),
  then butt the two clips together.  See :meth:`Switcher.naive`.

Both arms are the same code except for that one decision, so the comparison is a
real control rather than a strawman.

The seam criterion used here is the one stated by the task, not a new one:

    ``seam |dq| <= in-segment p95(|dq|)``   for q = position, velocity, acceleration

``|dq|`` at the seam is literally the same finite difference as ``|dq|`` inside a
segment -- both are "the change in q between consecutive frames" -- so the two
sides of the inequality have identical units and identical meaning.  Tolerance
*style* follows ``scripts/mlx_align.py`` where an absolute number is needed (max
absolute error, metres for positions / degrees for angles, worst case over the
compared block); the seam table reports that same worst-case form.

Nothing in this module changes the accepted default path: a switch is only ever a
*choice of which style object builds frames 4..7 while the existing context is
kept*.  With no switch requested the caller keeps calling ``Agent.plan`` and this
module is not on the path at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..export.plan_json import angle_between, world_positions_xyz
from .agent import (Agent, Command, Motion, WorldFrame, quaternion_matrix,
                    restore_world)

#: The stated seam rule.  Recorded verbatim so the report cannot drift from it.
SEAM_CRITERION = "seam |dq| <= max(in-segment p95 |dq|)  for q in {position, velocity, acceleration}"
#: A second, weaker position rule in the same "<= p95" shape: the handoff gap must
#: not exceed one normal frame of the same quantity.  This is what turns "1.4 mm"
#: into something a reader can judge without a ruler.
POSITION_CRITERION = "seam |dp| <= p95 per-frame |dp|"
#: Leg provenance labels.  ``kind`` is metadata for the report, not behaviour.
LEG_KINDS = ("seamless", "naive-aligned", "naive-raw")
P95 = 95.0


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------
@dataclass
class Signals:
    """Per-frame kinematic series of one clip, all at ``FPS``.

    ``root_*`` are 3-vectors (world metres), ``joint_*`` are per-joint scalars
    derived from the geodesic angle between consecutive rotations -- the same
    unsigned-increment convention ``motionbricks_mlx.eval.metrics.jitter`` uses.
    """

    frames: int
    root_position: np.ndarray          # (F, 3) m
    root_velocity: np.ndarray          # (F-1, 3) m/s
    root_acceleration: np.ndarray      # (F-2, 3) m/s^2
    joint_omega: np.ndarray            # (F-1, J) deg/s
    joint_alpha: np.ndarray            # (F-2, J) deg/s^2
    joint_position: np.ndarray         # (F, J, 3) m world


def signals(motion: Motion, parents: np.ndarray, rest: np.ndarray, fps: float,
            world=None) -> Signals:
    """Kinematic series of one clip (positions, velocities, accelerations).

    ``world`` is the forward-kinematics function; it defaults to
    ``plan_json.world_positions_xyz`` (the reference two-loop form).  The viewer
    passes its own bit-identical batched version, which is ~11x faster -- the
    seam report must not cost more than the plan it describes.
    """
    world = world or world_positions_xyz
    roots = np.asarray(motion.roots, dtype=np.float64)
    rotations = np.asarray(motion.rotations_xyzw, dtype=np.float64)
    if roots.shape[0] < 3:
        raise ValueError(f"a seam needs at least 3 frames, got {roots.shape[0]}")
    velocity = np.diff(roots, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    omega = np.degrees(angle_between(rotations[1:], rotations[:-1])) * fps
    alpha = np.diff(omega, axis=0) * fps
    return Signals(frames=int(roots.shape[0]), root_position=roots,
                   root_velocity=velocity, root_acceleration=acceleration,
                   joint_omega=omega, joint_alpha=alpha,
                   joint_position=world(roots, rotations, parents, rest))


def _p95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    return float(np.percentile(values, P95)) if values.size else 0.0


# ---------------------------------------------------------------------------
# the seam measurement
# ---------------------------------------------------------------------------
def seam_metrics(leg_in: Motion, index_in: int, leg_out: Motion,
                 parents: np.ndarray, rest: np.ndarray, fps: float = 30.0,
                 label: str = "", world=None) -> dict:
    """Measure the seam between ``leg_in[index_in]`` and ``leg_out[0]``.

    ``index_in`` is the frame of the incoming clip the outgoing clip was planned
    from (``Agent.advance(index_in)``): the outgoing clip's frame 0 is placed *at*
    that pose by ``restore_world``, so this is where the two clips meet.

    Every reported jump is the finite difference of the same quantity across the
    seam, and every baseline is the p95 of that quantity's finite difference over
    the interiors of both legs -- so ``ratio <= 1`` is exactly
    :data:`SEAM_CRITERION`.  ``p95`` is taken per channel for the joints and the
    worst channel is reported (the conservative direction).
    """
    if index_in < 3:
        raise ValueError(f"index_in must be >= 3 to carry an acceleration, got {index_in}")
    if index_in >= leg_in.frames:
        raise ValueError(f"index_in {index_in} is outside a {leg_in.frames}-frame clip")
    if leg_out.frames < 4:
        raise ValueError(f"the outgoing clip needs >= 4 frames, got {leg_out.frames}")

    inside = signals(leg_in, parents, rest, fps, world)
    outside = signals(leg_out, parents, rest, fps, world)

    # --- position: how far apart are the two clips at the handoff? -------------
    root_position = float(np.linalg.norm(outside.root_position[0] - inside.root_position[index_in]))
    joint_delta = np.linalg.norm(outside.joint_position[0] - inside.joint_position[index_in], axis=-1)
    joint_position = float(joint_delta.max())
    joint_position_worst = int(np.argmax(joint_delta))
    # Baseline in the same "<= p95" shape: a normal frame of the same clip.
    root_step_p95 = max(_p95(np.linalg.norm(np.diff(inside.root_position[:index_in + 1], axis=0), axis=1)),
                        _p95(np.linalg.norm(np.diff(outside.root_position, axis=0), axis=1)))
    joint_step_p95 = max(_p95(np.linalg.norm(np.diff(inside.joint_position[:index_in + 1], axis=0), axis=-1)),
                         _p95(np.linalg.norm(np.diff(outside.joint_position, axis=0), axis=-1)))

    # --- velocity / acceleration: root (3-vector) ------------------------------
    root_velocity = float(np.linalg.norm(outside.root_velocity[0] - inside.root_velocity[index_in - 1]))
    root_acceleration = float(np.linalg.norm(outside.root_acceleration[0]
                                             - inside.root_acceleration[index_in - 2]))
    root_velocity_p95 = max(_p95(np.linalg.norm(np.diff(inside.root_velocity[:index_in], axis=0), axis=1)),
                            _p95(np.linalg.norm(np.diff(outside.root_velocity, axis=0), axis=1)))
    root_acceleration_p95 = max(
        _p95(np.linalg.norm(np.diff(inside.root_acceleration[:index_in - 1], axis=0), axis=1)),
        _p95(np.linalg.norm(np.diff(outside.root_acceleration, axis=0), axis=1)))

    # --- velocity / acceleration: joints (per-joint scalar, worst channel) -----
    joint_velocity = np.abs(outside.joint_omega[0] - inside.joint_omega[index_in - 1])
    joint_acceleration = np.abs(outside.joint_alpha[0] - inside.joint_alpha[index_in - 2])
    joint_velocity_p95 = np.maximum(
        _p95_per_column(np.abs(np.diff(inside.joint_omega[:index_in], axis=0))),
        _p95_per_column(np.abs(np.diff(outside.joint_omega, axis=0))))
    joint_acceleration_p95 = np.maximum(
        _p95_per_column(np.abs(np.diff(inside.joint_alpha[:index_in - 1], axis=0))),
        _p95_per_column(np.abs(np.diff(outside.joint_alpha, axis=0))))
    velocity_ratio = _joint_ratios(joint_velocity, joint_velocity_p95)
    acceleration_ratio = _joint_ratios(joint_acceleration, joint_acceleration_p95)

    ratios = {
        "root_position": _ratio(root_position, root_step_p95),
        "joint_position": _ratio(joint_position, joint_step_p95),
        "root_velocity": _ratio(root_velocity, root_velocity_p95),
        "root_acceleration": _ratio(root_acceleration, root_acceleration_p95),
        "joint_velocity": float(velocity_ratio.max()),
        "joint_acceleration": float(acceleration_ratio.max()),
    }
    report = {
        "label": label,
        "frames_in": int(leg_in.frames),
        "frames_out": int(leg_out.frames),
        "index_in": int(index_in),
        "root_position_m": root_position,
        "joint_position_m": joint_position,
        "joint_position_worst_joint": joint_position_worst,
        "root_velocity_mps": root_velocity,
        "root_acceleration_mps2": root_acceleration,
        "joint_velocity_deg_s": float(joint_velocity.max()),
        "joint_acceleration_deg_s2": float(joint_acceleration.max()),
        "joint_velocity_worst_joint": int(np.argmax(velocity_ratio)),
        "joint_acceleration_worst_joint": int(np.argmax(acceleration_ratio)),
        "p95": {
            "root_position_m": root_step_p95,
            "joint_position_m": joint_step_p95,
            "root_velocity_mps": root_velocity_p95,
            "root_acceleration_mps2": root_acceleration_p95,
            "joint_velocity_deg_s": float(joint_velocity_p95.max()),
            "joint_acceleration_deg_s2": float(joint_acceleration_p95.max()),
        },
        "ratio": ratios,
        "joints_over_p95": {
            "velocity": int((joint_velocity > joint_velocity_p95).sum()),
            "acceleration": int((joint_acceleration > joint_acceleration_p95).sum()),
        },
        "criterion": SEAM_CRITERION,
        "position_criterion": POSITION_CRITERION,
        "within_p95": {key: ratios[key] <= 1.0 for key in ratios},
        "all_within_p95": all(value <= 1.0 for value in ratios.values()),
        "position_within_one_frame": bool(root_position <= root_step_p95
                                          and joint_position <= joint_step_p95),
    }
    return report


def _p95_per_column(values: np.ndarray) -> np.ndarray:
    """p95 down axis 0, keeping one value per joint."""
    if values.size == 0:
        return np.zeros(values.shape[1] if values.ndim > 1 else 0, dtype=np.float64)
    return np.percentile(values, P95, axis=0)


def _joint_ratios(values: np.ndarray, baselines: np.ndarray) -> np.ndarray:
    """Per-joint ratios with :func:`_ratio`'s zero policy, vectorised."""
    return np.array([_ratio(float(value), float(baseline))
                     for value, baseline in zip(np.ravel(values), np.ravel(baselines))],
                    dtype=np.float64)


#: Smallest baseline a ratio may divide by.  A *perfectly* uniform segment has a
#: p95 of exactly zero, where "no worse than p95" means "no jump at all": zero
#: stays inside it and anything else does not.  Reporting ``inf`` there would be
#: honest but not JSON-representable (``JSON.parse`` rejects ``Infinity``, so the
#: page would fail to decode the plan), so the division saturates at this floor
#: instead -- monotone, finite, and the same decision for every magnitude that can
#: occur in a real clip.
_RATIO_FLOOR = 1.0e-9


def _ratio(value: float, baseline: float) -> float:
    """``value / baseline`` with the zero-baseline policy above."""
    if value == 0.0:
        return 0.0
    if baseline > _RATIO_FLOOR:
        return float(value / baseline)
    return float(value / _RATIO_FLOOR)


# ---------------------------------------------------------------------------
# legs and the joined clip
# ---------------------------------------------------------------------------
@dataclass
class Leg:
    """One generated clip plus how much of it the joined clip shows."""

    style: str
    kind: str               # "seamless" | "naive" | "naive-aligned"
    motion: Motion
    advance: int = 0        # frame of the *previous* leg this leg was conditioned on
    frames_shown: int = 0   # frames of this clip that enter the joined motion


@dataclass
class SwitchPlan:
    """A joined multi-leg motion plus one seam report per handoff."""

    legs: list[Leg]
    seams: list[dict]
    roots: np.ndarray
    rotations_xyzw: np.ndarray
    target_roots: np.ndarray
    target_local_rotations_xyzw: np.ndarray
    joint_names: list[str] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return int(self.roots.shape[0])

    @property
    def styles(self) -> list[str]:
        return [leg.style for leg in self.legs]

    def legs_payload(self) -> list[dict]:
        """Where each leg sits inside the joined clip (page + HUD read this)."""
        out, cursor = [], 0
        for leg in self.legs:
            end = cursor + int(leg.frames_shown)
            out.append({"style": leg.style, "kind": leg.kind, "advance": int(leg.advance),
                        "start": cursor, "end": end - 1, "frames": int(leg.frames_shown),
                        "generated_frames": int(leg.motion.frames)})
            cursor = end
        return out

    def seam_payload(self) -> list[dict]:
        """Seam reports with their frame index inside the joined clip."""
        out = []
        for entry in self.seams:
            out.append({"joined_frame": int(entry["joined_frame"]),
                        "from_style": entry["from_style"], "to_style": entry["to_style"],
                        "metrics": entry["metrics"]})
        return out

    def document(self) -> dict:
        """The ``switch`` block the viewer and the CLI add to a plan document."""
        return {
            "criterion": SEAM_CRITERION,
            "position_criterion": POSITION_CRITERION,
            "legs": self.legs_payload(),
            "seams": self.seam_payload(),
            "joined_frames": self.frames,
            "worst_seam_ratio": max((max(entry["metrics"]["ratio"].values())
                                     for entry in self.seams), default=0.0),
            "all_seams_within_p95": all(entry["metrics"]["all_within_p95"]
                                        for entry in self.seams),
        }


def copy_motion(motion: Motion) -> Motion:
    """Deep enough copy that ``restore_world`` on it cannot touch the original."""
    return Motion(np.array(motion.roots, copy=True),
                  np.array(motion.rotations_xyzw, copy=True),
                  np.array(motion.target_roots, copy=True),
                  np.array(motion.target_local_rotations_xyzw, copy=True))


def place_at(motion: Motion, x: float, z: float, heading: float) -> Motion:
    """Put a clip's frame 0 at world ``(x, z, heading)`` (``restore_world``).

    This is the *charitable* half of the naive control: a hard cut that does not
    also teleport the character.  The joints are untouched -- restoring a world
    pose cannot fix a pose that was generated somewhere else.
    """
    placed = copy_motion(motion)
    restore_world(placed, WorldFrame(np.float32(x), np.float32(z), np.float32(heading)))
    return placed


def heading_at(motion: Motion, index: int) -> float:
    """Body yaw (radians) of frame ``index``, from the root joint's world rotation."""
    matrix = quaternion_matrix(np.asarray(motion.rotations_xyzw[index, 0], dtype=np.float64))
    return float(np.arctan2(matrix[0, 2], matrix[2, 2]))


def join_legs(legs: list[Leg], parents: np.ndarray, rest: np.ndarray,
              joint_names: list[str] | None = None, fps: float = 30.0,
              world=None) -> SwitchPlan:
    """Concatenate ``legs[i][:frames_shown]`` into one clip; measure every seam.

    The last leg contributes all of its frames.  The joined clip is what the page
    plays, so the numbers and the picture are the same object (no separate
    "analysis copy" that could disagree with the animation).
    """
    if not legs:
        raise ValueError("a switch needs at least one leg")
    roots, rotations, seams, cursor = [], [], [], 0
    for index, leg in enumerate(legs):
        shown = leg.motion.frames if index == len(legs) - 1 else int(leg.frames_shown)
        if shown < 1:
            raise ValueError(f"leg {index} ({leg.style}) shows {shown} frames")
        if index > 0:
            if leg.advance <= 0:
                raise ValueError(f"leg {index} ({leg.style}) has no advance frame to measure")
            seams.append({
                "joined_frame": cursor,
                "from_style": legs[index - 1].style,
                "to_style": leg.style,
                "metrics": seam_metrics(legs[index - 1].motion, leg.advance, leg.motion,
                                        parents, rest, fps=fps, world=world,
                                        label=f"{legs[index - 1].style}->{leg.style}"),
            })
        roots.append(np.asarray(leg.motion.roots)[:shown])
        rotations.append(np.asarray(leg.motion.rotations_xyzw)[:shown])
        cursor += shown
    return SwitchPlan(legs=legs, seams=seams,
                      roots=np.concatenate(roots, axis=0),
                      rotations_xyzw=np.concatenate(rotations, axis=0),
                      target_roots=np.asarray(legs[-1].motion.target_roots),
                      target_local_rotations_xyzw=np.asarray(
                          legs[-1].motion.target_local_rotations_xyzw),
                      joint_names=list(joint_names or []))


# ---------------------------------------------------------------------------
# the two arms
# ---------------------------------------------------------------------------
class Switcher:
    """Plan A -> B (-> A) two ways: conditioned on A's window, or independently."""

    def __init__(self, pipeline, bundle, styles: dict, seed: int = 1,
                 move=(0.0, 1.0), facing=(0.0, 1.0), world=None):
        self.pipeline = pipeline
        self.world = world
        self.bundle = bundle
        self.styles = styles
        self.seed = int(seed)
        self.move = tuple(move)
        self.facing = tuple(facing)
        self.parents = np.asarray(bundle.joint_parents)
        self.rest = np.asarray(bundle.neutral_joints)
        self.joint_names = list(bundle.joint_names)

    def command(self, style_name: str, seed: int | None = None) -> Command:
        seed = self.seed if seed is None else int(seed)
        style = self.styles[style_name]
        return Command(style=style, movement=(self.move[0], 0.0, self.move[1]),
                       facing=(self.facing[0], 0.0, self.facing[1]), seed=seed)

    def _new_agent(self, style_name: str) -> Agent:
        agent = Agent(self.pipeline, self.bundle.neutral_joints, self.parents)
        agent.reset(self.styles[style_name])
        return agent

    def first_leg(self, style_name: str, seed: int | None = None) -> tuple[Agent, Motion]:
        """Leg A: a fresh reseed of A (the same start for both arms)."""
        agent = self._new_agent(style_name)
        motion, _ = agent.plan(self.command(style_name, seed), self.styles[style_name])
        return agent, motion

    def seamless(self, style_a: str, style_b: str, seam_at: int,
                 seed: int | None = None) -> tuple[Motion, Motion, Agent]:
        """**The method.** Keep A's context, select B only for frames 4..7.

        The only difference from a plain A plan is which style object
        ``build_constraints`` reads; ``agent.plan`` is called unchanged, so the
        context (frames 0..3, with its velocities), the inherited
        ``apply_context_blend`` and the world restore are the accepted code.
        """
        agent, motion_a = self.first_leg(style_a, seed)
        agent.advance(int(seam_at))
        motion_b, _ = agent.plan(self.command(style_b, seed), self.styles[style_b])
        return motion_a, motion_b, agent

    def naive(self, style_a: str, style_b: str, seam_at: int, seed: int | None = None,
              align: bool = True) -> tuple[Motion, Motion]:
        """**The control.** Independent reseed for each style, then butt them.

        ``align=True`` additionally slides B's clip so its first frame sits at A's
        handoff pose -- the most charitable hard cut, and what the shipped viewer
        does *not* do.  Joint angles and their rates are still B's own, which is
        the whole point: a world restore cannot fix a pose generated elsewhere.
        """
        _, motion_a = self.first_leg(style_a, seed)
        _, motion_b = self.first_leg(style_b, seed)
        if align:
            motion_b = place_at(motion_b, float(motion_a.roots[seam_at, 0]),
                                float(motion_a.roots[seam_at, 2]),
                                heading_at(motion_a, seam_at))
        return motion_a, motion_b

    def sequence(self, styles: list[str], seam_frames: list[int], kind: str = "seamless",
                 seed: int | None = None) -> SwitchPlan:
        """A -> B -> A ... as one joined clip, measuring every seam.

        ``styles[0]`` is planned from a fresh reseed; every later style is reached
        with ``Agent.advance(seam_frames[i-1])`` on the *same* agent, so each leg
        is conditioned on the previous leg's live state.
        """
        if len(styles) < 1:
            raise ValueError("a sequence needs at least one style")
        return self.sequence_with(self._new_agent(styles[0]), styles, seam_frames, kind, seed)

    def sequence_with(self, agent: Agent, styles: list[str], seam_frames: list[int],
                      kind: str = "seamless", seed: int | None = None) -> SwitchPlan:
        """Same as :meth:`sequence` but on a caller-owned agent (the viewer's).

        ``agent`` must already be reset to ``styles[0]``.  The viewer uses this so
        the joined clip is produced by the *same* stateful agent that serves live
        plans: after the call the agent's context is the last leg's state, and a
        following live plan continues from there instead of from a private copy.
        """
        if len(seam_frames) != len(styles) - 1:
            raise ValueError(f"{len(styles)} styles need {len(styles) - 1} seam frames, "
                             f"got {len(seam_frames)}")
        legs: list[Leg] = []
        for index, style_name in enumerate(styles):
            if index == 0:
                motion, _ = agent.plan(self.command(style_name, seed), self.styles[style_name])
                advance = 0
            else:
                advance = int(seam_frames[index - 1])
                agent.advance(advance)
                motion, _ = agent.plan(self.command(style_name, seed), self.styles[style_name])
            shown = motion.frames if index == len(styles) - 1 else int(seam_frames[index])
            legs.append(Leg(style=style_name, kind=kind, motion=motion,
                            advance=advance, frames_shown=shown))
        return join_legs(legs, self.parents, self.rest, self.joint_names, world=self.world)


__all__ = ["LEG_KINDS", "Leg", "P95", "POSITION_CRITERION", "SEAM_CRITERION", "Signals", "SwitchPlan",
           "Switcher", "copy_motion", "heading_at", "join_legs", "place_at",
           "seam_metrics", "signals"]
