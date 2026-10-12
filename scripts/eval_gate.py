#!/usr/bin/env python3
"""The motion quality regression gate (C-line).

Runs the fixed scenario set (6 sampling seeds + the reference scenario pair,
plus the reference JSON's own steps as quality groups), judges metrics 1-6
against the calibrated thresholds, verifies metric 5 against the reference
runtime's plan/trace artifacts, and fires the five negative-control
corruptions over every good motion — each control passes only when its
designated metric goes red on at least one of them:

    ./.venv/bin/python scripts/eval_gate.py            # collect + gate
    ./.venv/bin/python scripts/eval_gate.py --fresh    # re-collect first
    ./.venv/bin/python scripts/eval_gate.py --no-collect

Exit 0 only when every quality group passes, the cross-impl pairs pass, and
every negative control trips.  The JSON report lands in out/eval/gate-report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motionbricks_mlx.eval.document import load_step, skeleton_tables  # noqa: E402
from motionbricks_mlx.eval.gate import (  # noqa: E402
    GateReport,
    run_gate,
    run_negative_controls,
)
from motionbricks_mlx.eval.thresholds import THRESHOLDS, Thresholds  # noqa: E402

EVAL_DIR = REPO_ROOT / "out/eval"
SEEDS = (1, 7, 13, 42, 77, 123)
REFERENCE_PLAN = REPO_ROOT / "out/ref/plan-ref.json"
REFERENCE_PLAN_FALLBACK = Path.home() / "dev/motion-bricks-cpp/artifacts/plan-walk.json"
REFERENCE_TRACES = [REPO_ROOT / "out/ref/trace/transition-0.json",
                    REPO_ROOT / "out/ref/trace/transition-1.json"]


def ensure_inputs(fresh: bool, no_collect: bool) -> None:
    needed = [EVAL_DIR / f"plan-seed-{seed}.json" for seed in SEEDS]
    needed += [EVAL_DIR / "plan-refpair.json", EVAL_DIR / "tokens-refpair.json"]
    if not fresh and all(p.is_file() for p in needed):
        return
    if no_collect:
        missing = [str(p) for p in needed if not p.is_file()]
        raise SystemExit(f"missing inputs (--no-collect): {missing}")
    from eval_collect import collect_all

    meta = collect_all(out_dir=EVAL_DIR)
    print(f"collected scenario set ({meta['elapsed_seconds']}s)")


def load_reference_plan() -> dict:
    for path in (REFERENCE_PLAN, REFERENCE_PLAN_FALLBACK):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise SystemExit(f"no reference plan found (looked at {REFERENCE_PLAN} and "
                     f"{REFERENCE_PLAN_FALLBACK})")


def verdict_dicts(verdicts) -> list[dict]:
    return [{"key": v.key, "value": v.value, "limit": v.limit,
             "direction": v.direction, "passed": v.passed} for v in verdicts]


def build_report(thresholds: Thresholds, skip_cross_impl: bool) -> GateReport:
    # --- quality groups: 6 seed steps + the 2 MLX reference-pair steps + the
    # reference JSON's own 2 steps (metric-6 "green on the reference JSON").
    steps: dict[str, object] = {}
    documents: dict[str, dict] = {}
    for seed in SEEDS:
        documents[f"seed-{seed}"] = json.loads(
            (EVAL_DIR / f"plan-seed-{seed}.json").read_text(encoding="utf-8"))
        steps[f"seed-{seed}"] = load_step(documents[f"seed-{seed}"], 0)

    documents["refpair"] = json.loads(
        (EVAL_DIR / "plan-refpair.json").read_text(encoding="utf-8"))
    ref_names = [e["name"] for e in documents["refpair"]["steps"][0]["skeleton"]]
    pair_labels = [s["label"] for s in documents["refpair"]["steps"]]
    for index, label in enumerate(pair_labels):
        steps[f"mlx-{label}"] = load_step(documents["refpair"], index, ref_names)

    cross = None
    if not skip_cross_impl:
        reference = load_reference_plan()
        ref_labels = [s["label"] for s in reference["steps"]]
        tokens_mine = json.loads(
            (EVAL_DIR / "tokens-refpair.json").read_text(encoding="utf-8"))
        cross = {}
        for index, label in enumerate(pair_labels):
            if index >= len(reference["steps"]) or ref_labels[index] != label:
                raise SystemExit(f"reference step {index} is "
                                 f"{ref_labels[index:]} , expected {label}")
            mine = load_step(documents["refpair"], index, ref_names)
            theirs = load_step(reference, index, ref_names)
            mine_tokens = np.asarray(tokens_mine["steps"][index]["tokens"])
            ref_tokens = None
            if index < len(REFERENCE_TRACES) and REFERENCE_TRACES[index].is_file():
                trace = json.loads(
                    REFERENCE_TRACES[index].read_text(encoding="utf-8"))
                ref_tokens = np.asarray(trace["tensors"]["pose.tokens"])
            cross[f"{label} (seed {theirs.seed})"] = (mine, theirs, mine_tokens,
                                                      ref_tokens)
        for index, label in enumerate(ref_labels):
            steps[f"reference-{label}"] = load_step(reference, index, ref_names)
        documents["reference"] = reference

    report = run_gate(steps, thresholds, pairs=cross or None)

    # --- negative controls: every corruption over every good group; a control
    # counts as fired when its designated metric goes red somewhere.
    parents, rest = skeleton_tables(documents["refpair"], 0)
    report.negative_controls = run_negative_controls(steps, parents, rest, thresholds)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--fresh", action="store_true", help="re-collect the scenario set")
    parser.add_argument("--no-collect", action="store_true",
                        help="fail instead of collecting when inputs are missing")
    parser.add_argument("--skip-cross-impl", action="store_true",
                        help="skip metric 5 (when reference artifacts are absent)")
    args = parser.parse_args()

    ensure_inputs(args.fresh, args.no_collect)
    report = build_report(THRESHOLDS, args.skip_cross_impl)
    for line in report.lines():
        print(line)

    payload = {
        "passed": report.passed,
        "seed_groups": {label: {"passed": group["passed"],
                                "verdicts": verdict_dicts(group["verdicts"]),
                                "values": {k: v for k, v in group.items()
                                           if isinstance(v, (int, float, str))}}
                        for label, group in report.seed_groups.items()},
        "cross_impl": None if report.cross_impl is None else {
            "passed": report.cross_impl["passed"],
            "note": report.cross_impl["note"],
            "pairs": {name: {"passed": pair["passed"],
                             "verdicts": verdict_dicts(pair["verdicts"])}
                      for name, pair in report.cross_impl["pairs"].items()}},
        "negative_controls": report.negative_controls,
    }
    out = EVAL_DIR / "gate-report.json"
    out.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"report: {out}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
