"""End-to-end gate behaviour on a synthetic scenario set — no bundle needed.

The good set is ``eval.synth``'s smooth walk (6 joints, real foot names);
it must pass every quality gate, every corruption must trip its designated
metric somewhere, and the whole thing must FAIL when either side is
violated.  ``expect_joints=6`` threads through the gate exactly like the
production 34.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motionbricks_mlx.eval.corruptions import (  # noqa: E402
    EFFECT,
    apply_corruption,
)
from motionbricks_mlx.eval.document import load_step  # noqa: E402
from motionbricks_mlx.eval.gate import (  # noqa: E402
    evaluate_pair,
    evaluate_step,
    run_gate,
    run_negative_controls,
)
from motionbricks_mlx.eval.synth import (  # noqa: E402
    PARENTS,
    REST,
    synth_step,
    to_document,
)
from motionbricks_mlx.eval.thresholds import Thresholds, THRESHOLDS  # noqa: E402

GOOD = {f"seed-{s}": synth_step(seed=s) for s in (1, 7, 13)}
TOKENS = np.arange(88, dtype=np.int64)          # any fixed integer payload


def test_every_good_step_passes_all_verdicts():
    for label, step in GOOD.items():
        values = evaluate_step(step, THRESHOLDS, expect_joints=6)
        assert values["structure_problems"] == 0, values["structure_problem_text"]
        reds = [v for v in values["verdicts"] if not v.passed]
        assert reds == [], f"{label}: {[v.line() for v in reds]}"
        assert values["passed"] is True


def test_run_gate_green_on_the_good_set():
    report = run_gate(GOOD, THRESHOLDS, expect_joints=6)
    assert report.passed is True
    assert set(report.seed_groups) == set(GOOD)
    assert "GATE PASS" in report.lines()


def test_document_round_trip_through_the_gate():
    """The JSON shape a real collection produces feeds the same gate."""
    document = to_document(GOOD["seed-1"])
    step = load_step(document, 0)
    report = run_gate({"roundtrip": step}, THRESHOLDS, expect_joints=6)
    assert report.passed is True


@pytest.mark.parametrize("name", sorted(EFFECT))
def test_a_single_corrupted_group_turns_the_gate_red(name):
    steps = dict(GOOD)
    steps["corrupted"] = apply_corruption(name, GOOD["seed-1"], PARENTS, REST)
    report = run_gate(steps, THRESHOLDS, expect_joints=6)
    assert report.passed is False
    assert report.seed_groups["corrupted"]["passed"] is False
    assert "GATE FAIL" in report.lines()


def test_negative_controls_all_trip_with_evidence():
    controls = run_negative_controls(GOOD, PARENTS, REST, THRESHOLDS,
                                     expect_joints=6)
    assert set(controls) == set(EFFECT)
    for name, control in controls.items():
        assert control["tripped"] is True, f"{name} did not trip"
        assert control["required_keys"] == list(EFFECT[name])
        assert control["on_group"] in GOOD
        assert control["evidence"], f"{name} tripped without evidence"
        # evidence carries (metric, good value, bad value) with the bad one red
        tripped = {v.key for v in evaluate_step(
            apply_corruption(name, GOOD[control["on_group"]], PARENTS, REST),
            THRESHOLDS, expect_joints=6)["verdicts"] if not v.passed}
        assert tripped & set(EFFECT[name])


def test_loosened_thresholds_stop_the_controls_from_tripping():
    """A control only means something if it depends on the thresholds."""
    permissive = Thresholds(
        foot_slide_max_m=1e9, jerk_p50_deg_min=0.0, jerk_p95_deg_max=1e9,
        jerk_max_deg_max=1e9, freeze_frac_max=1.0, foot_height_min_m=-1e9,
        ground_airborne_frac_max=1.0, boundary_root_max_m=1e9,
        boundary_rotation_max_deg=1e9,
    )
    controls = run_negative_controls(GOOD, PARENTS, REST, permissive,
                                     expect_joints=6)
    assert not any(c["tripped"] for c in controls.values())
    report = run_gate(GOOD, permissive, expect_joints=6)
    report.negative_controls = controls
    assert report.passed is False        # silent gate = failed gate


def test_cross_impl_pair_passes_when_identical_and_fails_on_drift():
    step = GOOD["seed-1"]
    pair = evaluate_pair(step, step, TOKENS, TOKENS)
    assert pair["passed"] is True
    assert pair["values"]["token_agreement_frac"] == 1.0
    assert pair["values"]["deviation_roots_max_m"] == 0.0

    drifted = apply_corruption("translate_roots", step, PARENTS, REST, dx=0.01)
    bad = evaluate_pair(step, drifted, TOKENS, TOKENS)
    assert bad["passed"] is False
    assert any(not v.passed for v in bad["verdicts"])

    wrong_tokens = evaluate_pair(step, step, TOKENS, TOKENS[:-1])
    assert wrong_tokens["values"]["token_agreement_frac"] == 0.0
    assert wrong_tokens["passed"] is False


def test_run_gate_reports_cross_impl_summary():
    step = GOOD["seed-1"]
    report = run_gate(GOOD, THRESHOLDS,
                      pairs={"pair": (step, step, TOKENS, TOKENS)},
                      expect_joints=6)
    assert report.cross_impl["passed"] is True
    assert report.cross_impl["note"] == "1 same-seed pair(s) with reference data"
    assert report.passed is True
