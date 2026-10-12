"""Reconstruction metrics, position and rotation reported separately.

Rotation convention: the contract's "6D rotation per joint" is read as two
orthonormal 3-vectors ``[u(3), v(3)]`` per joint (first two columns of a
rotation matrix, Zhou et al. 2019 continuity representation); the synthetic
writer (``synth.py``) uses the same layout.  Error is the geodesic distance
between predicted and target rotation matrices, in degrees.

All functions are numpy (metrics run offline after ``mx.eval``); the report
honours a per-frame mask so padded batch tails never enter the numbers.
"""

from __future__ import annotations

import numpy as np

from . import layout


def sixd_to_matrix(rot6: np.ndarray) -> np.ndarray:
    """``(..., 6*J)`` -> ``(..., J, 3, 3)`` rotation matrices (Gram-Schmidt)."""
    *lead, width = rot6.shape
    if width % 6 != 0:
        raise layout.LayoutError(f"6D rotation block width {width} is not a multiple of 6")
    r = rot6.reshape(*lead, width // 6, 6)
    u, v = r[..., :3], r[..., 3:6]
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v = v - (u * v).sum(-1, keepdims=True) * u
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    w = np.cross(u, v, axis=-1)
    return np.stack([u, v, w], axis=-1)  # columns


def rotation_geodesic_deg(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """Geodesic distance in degrees between ``(..., 3, 3)`` rotation matrices.

    ``trace(Ra^T Rb)`` expands to ``Σ_ij Ra[i,j]·Rb[i,j]`` — same-index
    elementwise sum, **no** transpose (verified against ``np.trace(Ra.T @ Rb)``
    after an earlier version computed ``trace(Ra Rb)`` here and inflated every
    angle; see ``docs/status-claude.md``).
    """
    trace = (r_a * r_b).sum(axis=(-1, -2))
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def position_error_m(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-frame Euclidean distance (metres) on a ``(..., 3)`` block."""
    return np.linalg.norm(pred - target, axis=-1)


def reconstruction_report(
    pred: np.ndarray,
    target: np.ndarray,
    offsets_table: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Masked reconstruction report for one clip, ``pred/target (T, D)``.

    ``mask (T,)`` marks real frames.  Position and rotation are reported
    separately, per the work ticket; velocity and an all-block frame MSE are
    included for completeness.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.shape[0] != pred.shape[0]:
        raise ValueError(f"mask length {m.shape[0]} != frames {pred.shape[0]}")
    if not m.any():
        raise ValueError("mask selects no frames")
    pos_sl = layout.block_slice(offsets_table, "root_position")
    rot_sl = layout.block_slice(offsets_table, "joint_rotations")
    vel_sl = layout.block_slice(offsets_table, "root_velocity")

    pos = position_error_m(pred[m][:, pos_sl], target[m][:, pos_sl])
    rot = rotation_geodesic_deg(
        sixd_to_matrix(pred[m][:, rot_sl]), sixd_to_matrix(target[m][:, rot_sl])
    )
    vel = position_error_m(pred[m][:, vel_sl], target[m][:, vel_sl])
    frame_mse = ((pred[m] - target[m]) ** 2).mean(axis=-1)
    return {
        "n_frames": int(m.sum()),
        "position_m": {"mean": float(pos.mean()), "p95": float(np.percentile(pos, 95))},
        "rotation_deg": {
            "mean": float(rot.mean()),
            "p95": float(np.percentile(rot, 95)),
            "max": float(rot.max()),
        },
        "velocity_mps": {"mean": float(vel.mean())},
        "frame_mse_all": float(frame_mse.mean()),
    }


def aggregate_reports(reports: list[dict]) -> dict:
    """Frame-weighted mean of per-clip reports (for dataset-level summaries)."""
    if not reports:
        raise ValueError("no reports to aggregate")

    def weighted(pick) -> float:
        return float(sum(pick(r) * r["n_frames"] for r in reports) / sum(r["n_frames"] for r in reports))

    return {
        "n_clips": len(reports),
        "n_frames": sum(r["n_frames"] for r in reports),
        "position_mean_m": weighted(lambda r: r["position_m"]["mean"]),
        "position_p95_m": weighted(lambda r: r["position_m"]["p95"]),
        "rotation_mean_deg": weighted(lambda r: r["rotation_deg"]["mean"]),
        "rotation_p95_deg": weighted(lambda r: r["rotation_deg"]["p95"]),
        "frame_mse_all": weighted(lambda r: r["frame_mse_all"]),
    }


def within_thresholds(aggregate: dict, position_m: float, rotation_deg: float) -> bool:
    """True iff the aggregated report meets both thresholds."""
    return (
        aggregate["position_mean_m"] <= position_m
        and aggregate["rotation_mean_deg"] <= rotation_deg
    )
