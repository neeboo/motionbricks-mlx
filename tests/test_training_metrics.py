"""Metrics: geodesic correctness spot-checks and red-on-corruption controls."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motionbricks_mlx.training import layout, metrics
from motionbricks_mlx.training import synth


def random_rotations(rng, shape):
    """Random rotation matrices via Gram-Schmidt on gaussian columns."""
    flat = rng.normal(size=(*shape, 3))
    u = flat / np.linalg.norm(flat, axis=-1, keepdims=True)
    raw = rng.normal(size=(*shape, 3))
    v = raw - (u * raw).sum(-1, keepdims=True) * u
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    w = np.cross(u, v, axis=-1)
    return np.stack([u, v, w], axis=-1)


def test_sixd_to_matrix_shape_and_orthonormality():
    rot6 = np.random.default_rng(0).normal(size=(5, 4 * 6))  # (T, 6*J)
    mats = metrics.sixd_to_matrix(rot6)
    assert mats.shape == (5, 4, 3, 3)
    gram = mats @ np.swapaxes(mats, -1, -2)
    assert np.allclose(gram, np.eye(3), atol=1e-5)


def test_sixd_to_matrix_rejects_bad_width():
    with pytest.raises(layout.LayoutError):
        metrics.sixd_to_matrix(np.zeros((2, 7)))  # 改错维度就红


def test_geodesic_known_angles():
    identity = np.eye(3)[None, None]
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    rz90 = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])[None, None]
    assert np.allclose(metrics.rotation_geodesic_deg(identity, identity), 0.0)
    assert np.allclose(metrics.rotation_geodesic_deg(identity, rz90), 90.0, atol=1e-6)


def test_geodesic_matches_closed_form_trace():
    """Regression: trace(Ra^T Rb) is the same-index elementwise sum, no transpose.

    An earlier version computed trace(Ra Rb) here and inflated every angle
    (123.7 deg reported where the truth was 2.07 deg).
    """
    rng = np.random.default_rng(42)
    ra = random_rotations(rng, (200,))
    rb = random_rotations(rng, (200,))
    got = metrics.rotation_geodesic_deg(ra, rb)
    ref_trace = np.trace(np.swapaxes(ra, -1, -2) @ rb, axis1=-2, axis2=-1)
    ref = np.degrees(np.arccos(np.clip((ref_trace - 1.0) / 2.0, -1.0, 1.0)))
    assert np.max(np.abs(got - ref)) < 1e-4
    assert np.all(got >= 0.0) and np.all(got <= 180.0)


def _report_fixture():
    table = layout.offsets(4, True)
    clip_frames, _, _ = synth.random_clip(np.random.default_rng(3), 4)
    target = clip_frames.astype(np.float64)
    pred = target.copy()
    return table, pred, target


def test_report_of_identical_arrays_is_zero():
    table, pred, target = _report_fixture()
    mask = np.ones(pred.shape[0], dtype=bool)
    report = metrics.reconstruction_report(pred, target, table, mask)
    assert report["position_m"]["mean"] < 1e-9
    # ~1e-7 deg residual: re-orthonormalizing already-orthonormal float32 data
    assert report["rotation_deg"]["mean"] < 1e-5
    assert report["frame_mse_all"] < 1e-12


def test_negative_corrupted_rotation_block_explodes_rotation_metric():
    """Shift each joint's [u v] pair by one vector slot — a plausible-but-wrong decode."""
    table, pred, target = _report_fixture()
    mask = np.ones(pred.shape[0], dtype=bool)
    sl = layout.block_slice(table, "joint_rotations")
    broken = pred.copy()
    block = broken[:, sl].reshape(pred.shape[0], 4, 6)
    broken[:, sl] = np.roll(block, 3, axis=-1).reshape(pred.shape[0], -1)
    report = metrics.reconstruction_report(broken, target, table, mask)
    assert report["rotation_deg"]["mean"] > 20.0
    assert report["position_m"]["mean"] < 1e-9  # position stays clean — reported separately


def test_negative_zeroed_position_block_explodes_position_metric():
    table, pred, target = _report_fixture()
    mask = np.ones(pred.shape[0], dtype=bool)
    sl = layout.block_slice(table, "root_position")
    broken = pred.copy()
    broken[:, sl] = 0.0
    report = metrics.reconstruction_report(broken, target, table, mask)
    assert report["position_m"]["mean"] > 0.1
    assert report["rotation_deg"]["mean"] < 1e-5


def test_report_honours_mask_padded_tail_excluded():
    table, pred, target = _report_fixture()
    mask = np.ones(pred.shape[0], dtype=bool)
    tail_mask = mask.copy()
    tail_mask[-2:] = False
    baseline = metrics.reconstruction_report(pred, target, table, tail_mask)
    poisoned = pred.copy()
    poisoned[-2:] += 10.0  # garbage lands only where the mask is False
    masked = metrics.reconstruction_report(poisoned, target, table, tail_mask)
    for key in ("position_m", "rotation_deg", "velocity_mps"):
        for stat, value in masked[key].items():
            assert abs(value - baseline[key][stat]) < 1e-9, key
    with pytest.raises(ValueError, match="no frames"):
        metrics.reconstruction_report(pred, target, table, np.zeros(pred.shape[0], bool))


def test_aggregate_and_thresholds():
    table, pred, target = _report_fixture()
    mask = np.ones(pred.shape[0], dtype=bool)
    reports = [metrics.reconstruction_report(pred, target, table, mask) for _ in range(3)]
    agg = metrics.aggregate_reports(reports)
    assert agg["n_clips"] == 3 and agg["n_frames"] == 3 * reports[0]["n_frames"]
    assert metrics.within_thresholds(agg, position_m=1e6, rotation_deg=1e6)
    # thresholds strictly below the measured values must fail the gate
    assert not metrics.within_thresholds(agg, position_m=agg["position_mean_m"] - 1e-6, rotation_deg=1e6)
    assert not metrics.within_thresholds(agg, position_m=1e6, rotation_deg=agg["rotation_mean_deg"] - 1e-6)
    with pytest.raises(ValueError):
        metrics.aggregate_reports([])
