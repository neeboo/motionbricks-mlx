"""The five negative-control corruptions (brief-glm-eval.md).

Each function takes a :class:`PlanStep` plus the skeleton tables, returns a
corrupted copy whose ``world_positions`` has been recomputed by the document
writer's own forward kinematics, so the corrupted document stays internally
consistent (metrics judge the motion, not a broken file).  ``EFFECT`` names
the metric keys each corruption must drive past its threshold — that mapping
is asserted by the gate's negative-control section.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..export.plan_json import world_positions_xyz
from .document import PlanStep, recompute_world_positions  # noqa: F401 (re-export)

# corruption name -> metric keys that must go red when it is applied
EFFECT = {
    "shuffle_frames": ("jerk_p95_deg", "jerk_max_deg", "boundary_root_max_m",
                       "boundary_rotation_max_deg"),
    "freeze_frames": ("freeze_frac", "jerk_p50_deg", "foot_slide_max_m"),
    "add_jitter_noise": ("jerk_p95_deg", "jerk_max_deg"),
    "raise_feet": ("ground_airborne_frac",),
    "translate_roots": ("boundary_root_max_m", "boundary_rotation_max_deg"),
}


def _with_world(step: PlanStep, parents: np.ndarray,
                rest_positions: np.ndarray, **overrides) -> PlanStep:
    """Copy with replaced arrays and a freshly computed world_positions."""
    fields = {f.name for f in dataclasses.fields(PlanStep)}
    bad = set(overrides) - fields
    if bad:
        raise TypeError(f"unknown fields: {sorted(bad)}")
    corrupted = dataclasses.replace(step, **overrides)
    corrupted.world_positions = recompute_world_positions(corrupted, parents, rest_positions)
    return corrupted


def shuffle_frames(step: PlanStep, parents: np.ndarray, rest_positions: np.ndarray,
                   seed: int = 0) -> PlanStep:
    """① Permute the motion's frame order (roots + rotations together)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(step.frames)
    return _with_world(step, parents, rest_positions,
                       roots=step.roots[order], rotations_xyzw=step.rotations_xyzw[order])


def freeze_frames(step: PlanStep, parents: np.ndarray, rest_positions: np.ndarray,
                  start: int = 12, count: int = 20) -> PlanStep:
    """② Freeze whole frames (root + rotations) for ``count`` frames, then resume.

    A whole-frame animation stall: frames ``start .. start+count-1`` duplicate
    frame ``start`` and the timeline resumes where it left off, so the resume
    boundary carries one catch-up jump.

    Why whole frames and not pose-only: this model's gait drags its toes —
    the feet translate with the body essentially everywhere (baseline slide ~
    root travel).  Holding joint rotations while the root advances therefore
    leaves every world-space metric untouched (measured: slide 1.073 ->
    1.090 m on seed 1); the stall only becomes visible when the whole frame
    stops.  Its signatures are the velocity dropout (jerk p50 collapses
    below the activity floor) and the catch-up jump (a one-time ~0.4 m foot
    displacement lands in the slide accumulation).
    """
    stop = min(start + count, step.frames)
    roots = step.roots.copy()
    rotations = step.rotations_xyzw.copy()
    roots[start:stop] = roots[start]
    rotations[start:stop] = rotations[start]
    return _with_world(step, parents, rest_positions, roots=roots,
                       rotations_xyzw=rotations)


def add_jitter_noise(step: PlanStep, parents: np.ndarray, rest_positions: np.ndarray,
                     sigma: float = 0.04, seed: int = 0) -> PlanStep:
    """③ High-frequency rotation noise, renormalised so structure stays valid.

    ``sigma`` is the per-component quaternion noise in absolute units; it
    yields ~4.6 deg of per-frame rotation noise, whose second difference
    (sigma_d2 ~ sqrt(6)*4.6 ~ 11 deg) puts the jerk max around 35-40 deg —
    past the 30 deg gate with ~2x baseline headroom to spare.
    """
    rng = np.random.default_rng(seed)
    rotations = step.rotations_xyzw + rng.normal(0.0, sigma, step.rotations_xyzw.shape)
    rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
    return _with_world(step, parents, rest_positions, rotations_xyzw=rotations)


def raise_feet(step: PlanStep, parents: np.ndarray, rest_positions: np.ndarray,
               height: float = 0.25) -> PlanStep:
    """④ Translate the whole character up so no foot point is near the ground.

    Raising the root keeps the pose rigid and the FK consistent; every foot
    point ends up ``height`` metres higher, so the airborne fraction must hit
    1.0.
    """
    roots = step.roots.copy()
    roots[:, 1] += height
    return _with_world(step, parents, rest_positions, roots=roots)


def translate_roots(step: PlanStep, parents: np.ndarray, rest_positions: np.ndarray,
                    dx: float = 0.5, dz: float = 0.5) -> PlanStep:
    """⑤ Shift the whole root trajectory sideways; targets stay put.

    The placed future context no longer matches the motion's tail, so the
    boundary-continuity maxima must blow past their gates.
    """
    roots = step.roots.copy()
    roots[:, 0] += dx
    roots[:, 2] += dz
    return _with_world(step, parents, rest_positions, roots=roots)


def apply_corruption(name: str, step: PlanStep, parents: np.ndarray,
                     rest_positions: np.ndarray, **kwargs) -> PlanStep:
    functions = {
        "shuffle_frames": shuffle_frames,
        "freeze_frames": freeze_frames,
        "add_jitter_noise": add_jitter_noise,
        "raise_feet": raise_feet,
        "translate_roots": translate_roots,
    }
    if name not in functions:
        raise KeyError(f"unknown corruption {name!r}; known: {sorted(functions)}")
    return functions[name](step, parents, rest_positions, **kwargs)


__all__ = [
    "EFFECT",
    "add_jitter_noise",
    "apply_corruption",
    "freeze_frames",
    "raise_feet",
    "shuffle_frames",
    "translate_roots",
    "world_positions_xyz",
]
