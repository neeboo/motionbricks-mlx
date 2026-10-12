"""The five quality metric families (brief-glm-eval.md metrics 1-5).

Every function returns plain dicts of floats with units in the key suffix
(``_m`` metres, ``_deg`` degrees, ``_frac`` a fraction in [0, 1]).  Algorithms
are fixed here and referenced by name from ``docs/status-glm.md``; change one,
change the other.

Frame-rate dependence: metrics 1-3 are per-frame quantities at the document's
fps (30).  If the fps contract ever changes, thresholds must be recalibrated —
they are not scale-free.
"""

from __future__ import annotations

import numpy as np

from ..export.plan_json import angle_between
from .document import PlanStep

# Percentile cut points reported for the distributions (metric 2 and the
# height distribution of metric 3).
P50, P95 = 50.0, 95.0


def foot_sliding(step: PlanStep, contact_threshold_m: float) -> dict:
    """Metric 1 — foot sliding during contact, metres.

    A foot is *in contact* at frame t when its lowest point (min world height
    over ankle_roll + toe_base) is below ``contact_threshold_m``.  For each
    foot, the sliding distance is the sum over consecutive frame pairs
    (t, t+1) where the foot is in contact at BOTH frames of the horizontal
    (xz) displacement of the foot's lowest point.  Reported per foot and as
    the max of the two (the gate judges the max).
    """
    feet = step.foot_joint_indices()
    per_foot: dict[str, float] = {}
    contact_fraction: dict[str, float] = {}
    for name, indices in feet.items():
        heights = step.world_positions[:, indices, 1].min(axis=1)   # (frames,)
        lowest = _lowest_point(step.world_positions, indices)       # (frames, 3)
        contact = heights < contact_threshold_m
        slide = 0.0
        for t in range(step.frames - 1):
            if contact[t] and contact[t + 1]:
                dxz = lowest[t + 1][[0, 2]] - lowest[t][[0, 2]]
                slide += float(np.hypot(dxz[0], dxz[1]))
        per_foot[name] = slide
        contact_fraction[name] = float(contact.mean())
    return {
        "foot_slide_left_m": per_foot["left"],
        "foot_slide_right_m": per_foot["right"],
        "foot_slide_max_m": max(per_foot.values()),
        "contact_fraction_max": max(contact_fraction.values()),
        "contact_frames_left": int(round(contact_fraction["left"] * step.frames)),
    }


def _lowest_point(world: np.ndarray, indices: tuple[int, int]) -> np.ndarray:
    """Per-frame position of the lower of the foot's two joints, (frames, 3)."""
    heights = world[:, list(indices), 1]
    choice = np.argmin(heights, axis=1)
    pair = np.asarray(indices, dtype=np.int64)
    return world[np.arange(world.shape[0]), pair[choice], :]


def jitter(step: PlanStep, freeze_eps_deg: float = 0.01) -> dict:
    """Metric 2 — joint-angle second difference (jerk) distribution, degrees.

    For each joint j: the per-frame rotation increment is the geodesic angle
    theta_j(t) = angle(q_j(t), q_j(t-1)) in degrees (frame-rate dependent).
    The jerk is the second difference d2_j(t) = theta_j(t+1) - 2*theta_j(t)
    + theta_j(t-1).  The distribution over all (t, j) of |d2| is summarised
    as p50 / p95 / max.  Added noise drives p95 and max up.

    ``freeze_frac`` is the fraction of (frame-pair, joint) increments below
    ``freeze_eps_deg`` — the direct stall detector.  Displacement-style
    metrics are near-invariant to a stall+catch-up pair (the motionless
    segment and the resume jump cancel); the dropout fraction is not.
    """
    rotations = step.rotations_xyzw
    theta = np.degrees(angle_between(rotations[1:], rotations[:-1]))  # (F-1, J)
    d2 = np.diff(theta, n=2, axis=0)                                   # (F-3, J)
    absolute = np.abs(d2).ravel()
    return {
        "jerk_p50_deg": float(np.percentile(absolute, P50)),
        "jerk_p95_deg": float(np.percentile(absolute, P95)),
        "jerk_max_deg": float(absolute.max()),
        "freeze_frac": float((theta < freeze_eps_deg).mean()),
    }


def grounding(step: PlanStep, air_threshold_m: float) -> dict:
    """Metric 3 — ground interaction.

    Per frame the lowest foot point height is the min world y over both feet's
    (ankle_roll, toe_base) joints.  Penetrating frames: height < 0 (below the
    ground plane).  Airborne frames: height > ``air_threshold_m``.

    Calibration note (measured on the f32 baseline, docs/status-glm.md): the
    model's foot joint centres ride a few millimetres below the plane in
    normal stance, so ``ground_penetration_frac`` is 30-100% on GOOD motion
    and does not discriminate — it is reported, but the GATE is on the
    penetration DEPTH (``foot_height_min_m`` against a negative floor) and on
    the airborne fraction.
    """
    lowest = np.minimum(
        step.world_positions[:, step.foot_joint_indices()["left"], 1].min(axis=1),
        step.world_positions[:, step.foot_joint_indices()["right"], 1].min(axis=1),
    )
    return {
        "ground_penetration_frac": float((lowest < 0.0).mean()),
        "ground_airborne_frac": float((lowest > air_threshold_m).mean()),
        "foot_height_min_m": float(lowest.min()),
        "foot_height_p5_m": float(np.percentile(lowest, 5.0)),
        "foot_height_p50_m": float(np.percentile(lowest, P50)),
        "foot_height_p95_m": float(np.percentile(lowest, P95)),
        "foot_height_max_m": float(lowest.max()),
    }


def boundary_error(step: PlanStep) -> dict:
    """Metric 4 — continuity with the placed future context, m and degrees.

    The motion's last ``targets.frames`` (= 4) frames are compared against the
    ``targets`` block: per-frame root position error (Euclidean, metres) and
    per-frame-per-joint geodesic rotation angle (degrees).  Mean and max of
    both are reported; the gate judges the maxima.
    """
    n = step.target_roots.shape[0]
    tail_roots = step.roots[-n:]
    tail_rotations = step.rotations_xyzw[-n:]
    root_error = np.linalg.norm(tail_roots - step.target_roots, axis=1)
    angle = np.degrees(angle_between(tail_rotations, step.target_rotations_xyzw))
    return {
        "boundary_root_mean_m": float(root_error.mean()),
        "boundary_root_max_m": float(root_error.max()),
        "boundary_rotation_mean_deg": float(angle.mean()),
        "boundary_rotation_max_deg": float(angle.max()),
    }


def token_agreement(mine: np.ndarray, reference: np.ndarray) -> dict:
    """Metric 5a — discrete token agreement (the ``pose.tokens`` criteria).

    Both sides are the sampled pose-token sequences (``tokens x 8`` codebook
    indices, sampled with the same seed): MLX from
    ``Agent.plan(..., capture_trace=True)``'s ``trace["pose.tokens"]`` and the
    reference from ``out/ref/trace/transition-*.json``'s
    ``tensors["pose.tokens"]``.  Agreement is the fraction of equal positions;
    a length mismatch scores 0.0.  1.0 means the two implementations picked
    bit-identical discrete motion programs.
    """
    mine = np.asarray(mine).reshape(-1).astype(np.int64)
    reference = np.asarray(reference).reshape(-1).astype(np.int64)
    if mine.shape != reference.shape:
        return {"token_agreement_frac": 0.0, "token_count_mine": int(mine.size),
                "token_count_reference": int(reference.size)}
    equal = mine == reference
    return {
        "token_agreement_frac": float(equal.mean()) if equal.size else 0.0,
        "token_count_mine": int(mine.size),
        "token_count_reference": int(reference.size),
    }


def motion_deviation(roots_a: np.ndarray, rotations_a: np.ndarray,
                     roots_b: np.ndarray, rotations_b: np.ndarray,
                     world_a: np.ndarray, world_b: np.ndarray) -> dict:
    """Metric 5b — final motion deviation between two implementations.

    Same shape, same seed: max abs root difference (m), geodesic rotation
    angle between per-joint quaternions (mean/max, degrees), max abs
    world-position difference (m).  Gates sit at the shipped alignment
    tolerances (see thresholds.py for provenance).
    """
    if roots_a.shape != roots_b.shape:
        raise ValueError(f"root shape mismatch {roots_a.shape} vs {roots_b.shape}")
    root_delta = float(np.abs(roots_a - roots_b).max())
    angle = np.degrees(angle_between(rotations_a, rotations_b))
    world_delta = float(np.abs(world_a - world_b).max())
    return {
        "deviation_roots_max_m": root_delta,
        "deviation_rotation_mean_deg": float(angle.mean()),
        "deviation_rotation_max_deg": float(angle.max()),
        "deviation_world_max_m": world_delta,
    }


__all__ = [
    "boundary_error",
    "foot_sliding",
    "grounding",
    "jitter",
    "motion_deviation",
    "token_agreement",
]
