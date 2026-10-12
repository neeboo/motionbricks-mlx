"""The regression gate: per-seed metric values, PASS/FAIL, negative controls.

``run_gate`` is pure (documents in, report out) so tests exercise it without
the MLX bundle; ``scripts/eval_gate.py`` is the CLI wrapper that collects
fresh f32 plans and prints the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .corruptions import EFFECT, apply_corruption
from .document import PlanStep, validate_structure
from .metrics import (
    boundary_error,
    foot_sliding,
    grounding,
    jitter,
    motion_deviation,
    token_agreement,
)
from .thresholds import FREEZE_EPS_DEG, Thresholds, Verdict, judge, judge_consistency


def evaluate_step(step: PlanStep, thresholds: Thresholds,
                  expect_joints: int = 34) -> dict:
    """Metrics 1-4 + 6 for one step, as a flat value dict with verdicts.

    ``expect_joints`` exists for synthetic test skeletons; production
    documents are the G1 34-joint skeleton (the default).
    """
    structure = validate_structure(step, expect_joints=expect_joints)
    values: dict = {
        "structure_problems": len(structure["problems"]),
        "structure_problem_text": "; ".join(structure["problems"]),
        **foot_sliding(step, thresholds_contact(step)),
        **jitter(step, freeze_eps_deg=FREEZE_EPS_DEG),
        **grounding(step, thresholds_air(step)),
        **boundary_error(step),
    }
    verdicts = judge(values, thresholds)
    if values["structure_problems"] > thresholds.structure_problems_max:
        verdicts.insert(0, Verdict("structure_problems",
                                   float(values["structure_problems"]),
                                   float(thresholds.structure_problems_max), "max", False))
    values["verdicts"] = verdicts
    values["passed"] = all(v.passed for v in verdicts)
    values["structure"] = structure
    return values


def thresholds_contact(step: PlanStep) -> float:
    from .thresholds import CONTACT_THRESHOLD_M

    return CONTACT_THRESHOLD_M


def thresholds_air(step: PlanStep) -> float:
    from .thresholds import AIR_THRESHOLD_M

    return AIR_THRESHOLD_M


@dataclass
class GateReport:
    seed_groups: dict[str, dict] = field(default_factory=dict)
    cross_impl: dict | None = None
    negative_controls: dict[str, dict] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        groups = all(group["passed"] for group in self.seed_groups.values())
        controls = all(c["tripped"] for c in self.negative_controls.values()) \
            if self.negative_controls else True
        cross = True if self.cross_impl is None else self.cross_impl["passed"]
        return bool(groups and controls and cross)

    def lines(self) -> list[str]:
        out: list[str] = []
        for label, group in self.seed_groups.items():
            out.append(f"[{'PASS' if group['passed'] else 'FAIL'}] {label}")
            for verdict in group["verdicts"]:
                out.append(verdict.line())
            if not group["passed"]:
                out.append(f"        structure: {group['structure_problem_text']}")
        if self.cross_impl is not None:
            cross = self.cross_impl
            out.append(f"[{'PASS' if cross['passed'] else 'FAIL'}] cross-impl "
                       f"({cross['note']})")
            for pair_name, pair in cross["pairs"].items():
                out.append(f"  pair {pair_name}")
                for verdict in pair["verdicts"]:
                    out.append("  " + verdict.line())
        if self.negative_controls:
            out.append("negative controls (each must trip its metric):")
            for name, control in self.negative_controls.items():
                status = "RED" if control["tripped"] else "NOT TRIPPED"
                out.append(f"  [{'PASS' if control['tripped'] else 'FAIL'}] {name:18s} "
                           f"{status} on {', '.join(control['tripped_keys']) or '(none)'}")
                for key, good, bad in control["evidence"]:
                    out.append(f"        {key:28s} good {good:.4g} -> bad {bad:.4g}")
        out.append(f"GATE {'PASS' if self.passed else 'FAIL'}")
        return out


def evaluate_pair(mine: PlanStep, reference: PlanStep,
                  mine_tokens: np.ndarray | None,
                  reference_tokens: np.ndarray | None) -> dict:
    """Metric 5 for one same-seed (mine, reference) pair."""
    values: dict = {}
    if mine_tokens is not None and reference_tokens is not None:
        values.update(token_agreement(mine_tokens, reference_tokens))
    if mine.frames == reference.frames:
        values.update(motion_deviation(
            mine.roots, mine.rotations_xyzw, reference.roots, reference.rotations_xyzw,
            mine.world_positions, reference.world_positions))
    else:
        values["frame_mismatch"] = f"{mine.frames} vs {reference.frames}"
    verdicts = judge_consistency(values)
    return {"values": values, "verdicts": verdicts,
            "passed": bool(verdicts) and all(v.passed for v in verdicts)}


def run_gate(seed_steps: dict[str, PlanStep], thresholds: Thresholds,
             pairs: dict[str, tuple[PlanStep, PlanStep, np.ndarray | None, np.ndarray | None]] | None = None,
             expect_joints: int = 34,
             ) -> GateReport:
    """Evaluate every seed group and the cross-impl pairs."""
    report = GateReport()
    for label, step in seed_steps.items():
        report.seed_groups[label] = evaluate_step(step, thresholds,
                                                  expect_joints=expect_joints)

    if pairs:
        pair_results = {}
        for name, (mine, reference, mine_tokens, ref_tokens) in pairs.items():
            pair_results[name] = evaluate_pair(mine, reference, mine_tokens, ref_tokens)
        report.cross_impl = {
            "pairs": pair_results,
            "passed": all(p["passed"] for p in pair_results.values()),
            "note": f"{len(pair_results)} same-seed pair(s) with reference data",
        }
    return report


def run_negative_controls(steps: dict[str, PlanStep], parents: np.ndarray,
                          rest_positions: np.ndarray, thresholds: Thresholds,
                          expect_joints: int = 34) -> dict[str, dict]:
    """Fire the five corruptions over every good step.

    A control counts as fired when its designated metric (``EFFECT[name]``)
    goes red on at least one step; evidence records good -> bad values for
    every metric the corruption moved on the step that fired.
    """
    good_values = {label: evaluate_step(step, thresholds, expect_joints=expect_joints)
                   for label, step in steps.items()}
    controls: dict[str, dict] = {}
    for name, required in EFFECT.items():
        fired = None
        for label, step in steps.items():
            corrupted = apply_corruption(name, step, parents, rest_positions)
            bad = evaluate_step(corrupted, thresholds, expect_joints=expect_joints)
            tripped = sorted({v.key for v in bad["verdicts"] if not v.passed})
            if not any(key in tripped for key in required):
                continue
            keys = sorted(set(required) | set(tripped))
            fired = {
                "tripped": True,
                "tripped_keys": tripped,
                "required_keys": list(required),
                "on_group": label,
                "evidence": [(key, float(good_values[label][key]), float(bad[key]))
                             for key in keys
                             if key in good_values[label] and key in bad],
            }
            break
        controls[name] = fired or {
            "tripped": False, "tripped_keys": [], "required_keys": list(required),
            "on_group": None, "evidence": [],
        }
    return controls


__all__ = [
    "GateReport",
    "evaluate_pair",
    "evaluate_step",
    "run_gate",
]
