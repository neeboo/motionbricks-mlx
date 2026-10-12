"""A tiny synthetic smooth "walk" for tests and self-checks.

Builds a 6-joint PlanStep whose four foot joints carry the real G1 foot
names (metrics resolve feet by name), with every joint moving on a slow
sinusoid (so freeze_frac stays ~0 and jerk stays smooth), the root
advancing at constant speed, and targets equal to the motion's last 4
frames (boundary error 0 by construction).  Good enough to pass every
quality gate in thresholds.THRESHOLDS and to react visibly to each
corruption.
"""

from __future__ import annotations

import numpy as np

from ..export.plan_json import world_positions_xyz
from .document import PlanStep

JOINT_NAMES = [
    "pelvis_skel",
    "left_ankle_roll_skel",
    "left_toe_base",
    "right_ankle_roll_skel",
    "right_toe_base",
    "waist_yaw_skel",
]
PARENTS = np.array([-1, 0, 1, 0, 3, 0], dtype=np.int64)
# absolute rest positions; FK uses rest[j] - rest[parent] offsets.  The toes
# sit clearly BELOW the ankles (like the real G1 foot) so the lowest-point
# pick never flips between the two joints — a flip reads as a 12 cm slide.
REST = np.array([
    [0.0, 0.72, 0.0],     # pelvis
    [0.09, 0.01, 0.0],    # left ankle_roll
    [0.09, -0.02, 0.12],  # left toe_base
    [-0.09, 0.01, 0.0],   # right ankle_roll
    [-0.09, -0.02, 0.12], # right toe_base
    [0.0, 0.90, 0.0],     # waist
], dtype=np.float64)


def _quat_xyzw(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    return np.concatenate([axis * np.sin(half), [np.cos(half)]])


def synth_step(seed: int = 0, frames: int = 44, speed: float = 0.012) -> PlanStep:
    """A smooth forward walk: constant root advance + small sinusoidal joints."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames, dtype=np.float64)
    roots = np.stack([0.09 * np.sin(2 * np.pi * t / 88),
                      0.72 + 0.004 * np.sin(2 * np.pi * t / 22 + 0.3),
                      speed * t], axis=1)
    # every joint sways on a visible-but-smooth sinusoid: fast enough that the
    # second difference clears the jerk_p50 activity floor, small enough that
    # the pelvis-driven foot wobble stays within [-0.02, +0.03] m (no
    # penetration, no airborne) and slide stays under the gate.
    amplitudes = np.array([0.02, 0.05, 0.05, 0.05, 0.05, 0.20])
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(JOINT_NAMES))
    periods = np.array([12.0, 14.0, 16.0, 18.0, 20.0, 24.0])
    rotations = np.zeros((frames, len(JOINT_NAMES), 4), dtype=np.float64)
    for joint in range(len(JOINT_NAMES)):
        angle = amplitudes[joint] * np.sin(2 * np.pi * t / periods[joint]
                                           + phases[joint])
        axis = axes[joint] / np.linalg.norm(axes[joint])
        half = angle / 2.0
        quat = np.stack([axis[0] * np.sin(half), axis[1] * np.sin(half),
                         axis[2] * np.sin(half), np.cos(half)], axis=1)
        rotations[:, joint] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    world = world_positions_xyz(roots, rotations, PARENTS, REST)
    target_roots = roots[-4:].copy()
    target_rotations = rotations[-4:].copy()
    return PlanStep(
        label=f"synth-{seed}", style="synth", seed=seed, advance=0, fps=30,
        roots=roots, rotations_xyzw=rotations,
        target_roots=target_roots, target_rotations_xyzw=target_rotations,
        world_positions=world, joint_names=list(JOINT_NAMES),
    )


def to_document(step: PlanStep) -> dict:
    """Serialize a PlanStep back into the plan-document JSON shape.

    Carries the skeleton on step 0 so ``load_step``/``skeleton_tables`` can
    round-trip it exactly like a collected document.
    """
    frames = step.frames
    return {
        "library": "synthetic",
        "fps": step.fps,
        "joints": step.joints,
        "steps": [{
            "label": step.label, "style": step.style, "seed": int(step.seed),
            "advance": int(step.advance),
            "motion": {
                "frames": frames, "joints": step.joints,
                "roots": step.roots.reshape(-1).tolist(),
                "rotations": step.rotations_xyzw.reshape(-1).tolist(),
            },
            "targets": {
                "frames": int(step.target_roots.shape[0]),
                "roots": step.target_roots.reshape(-1).tolist(),
                "rotations": step.target_rotations_xyzw.reshape(-1).tolist(),
            },
            "world_positions_xyz": step.world_positions.reshape(-1).tolist(),
            "skeleton": [
                {"name": name, "parent": int(PARENTS[j]),
                 "position": REST[j].tolist()}
                for j, name in enumerate(step.joint_names)
            ],
        }],
    }


__all__ = ["JOINT_NAMES", "PARENTS", "REST", "synth_step", "to_document"]
