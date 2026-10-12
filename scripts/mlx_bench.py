#!/usr/bin/env python3
"""Latency and memory measurement for the MLX port (CPU vs GPU backend).

Reports per-plan wall time (p50 / p90 / max) over a fixed number of planning
steps, plus cold start (bundle load + first plan) and peak memory.  Determinism
is checked alongside timing so a fast-but-wrong configuration cannot pass.

    ./.venv/bin/python scripts/mlx_bench.py --steps 25

``--paired`` additionally runs two configurations **interleaved inside one
process** on the identical step sequence, with the arm order rotated every step,
and reports the per-matched-step ratio (the only way to separate a small effect
from the drift of a shared desktop; see docs/optimization-report.md section 1):

    ./.venv/bin/python scripts/mlx_bench.py --paired \
        "MOTIONBRICKS_MLX_OPT=0" "MOTIONBRICKS_MLX_OPT=1" --steps 25
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motionbricks_mlx.data.agent import Agent, Command
from motionbricks_mlx.data.style import load_style
from motionbricks_mlx.models.bundle import Bundle
from motionbricks_mlx.models.transition import MlxPipeline

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "checkpoints/g1-f32-mlx"
DEFAULT_STYLES = Path.home() / "dev/motion-bricks-cpp/repo/generated/styles"
STYLES = ("walk", "walk_zombie", "slow_walk", "walk_stealth")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def measure(device: str, steps: int, styles_dir: Path, bundle_path: Path,
            seed: int, verify: bool) -> dict:
    mx.clear_cache() if hasattr(mx, "clear_cache") else None
    mx.reset_peak_memory()
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    cold_started = time.perf_counter()
    bundle = Bundle(bundle_path, device=device)
    pipeline = MlxPipeline(bundle)
    styles = {name: load_style(styles_dir / f"{name}.mbstyle") for name in STYLES}
    agent = Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
    agent.reset(styles["walk"])
    first, _ = agent.plan(Command(style=styles["walk"], seed=seed), styles["walk"])
    cold_seconds = time.perf_counter() - cold_started

    timings: list[float] = []
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(steps):
        name = STYLES[index % len(STYLES)]
        style = styles[name]
        if index and index % 4 == 0:
            agent.advance(3)
        command = Command(style=style, movement=(0.0, 0.0, 0.0),
                          facing=(0.0, 0.0, 1.0), seed=seed + index)
        started = time.perf_counter()
        motion, _ = agent.plan(command, style)
        mx.eval(mx.array(motion.roots.sum()))          # synchronize the backend
        timings.append((time.perf_counter() - started) * 1000.0)
        outputs.append((motion.roots.copy(), motion.rotations_xyzw.copy()))

    # Determinism: replan the identical sequence in a fresh agent and compare.
    deterministic = None
    if verify:
        agent2 = Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
        agent2.reset(styles["walk"])
        command = Command(style=styles["walk"], seed=seed)
        repeat_first, _ = agent2.plan(command, styles["walk"])
        deterministic = bool(
            np.array_equal(first.roots, repeat_first.roots)
            and np.array_equal(first.rotations_xyzw, repeat_first.rotations_xyzw))

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "device": device,
        "steps": steps,
        "cold_start_ms": cold_seconds * 1000.0,
        "p50_ms": statistics.median(timings),
        "p90_ms": percentile(timings, 0.9),
        "max_ms": max(timings),
        "min_ms": min(timings),
        "mean_ms": statistics.fmean(timings),
        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        "process_peak_rss_bytes": int(peak_rss),
        "process_peak_rss_before_load": int(baseline_rss),
        "deterministic": deterministic,
        "timings_ms": timings,
    }


def parse_arm(spec: str) -> dict[str, str]:
    """``"A=1;B=2"`` -> ``{"A": "1", "B": "2"}`` (``;`` separated, because the
    switch values themselves are comma separated)."""
    arm: dict[str, str] = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        key, separator, value = item.partition("=")
        if not separator:
            raise SystemExit(f"error: arm entry {item!r} is not KEY=VALUE")
        arm[key.strip()] = value
    return arm


def apply_arm(arm: dict[str, str]) -> None:
    """Set this arm's environment, clearing the variables it does not name."""
    for key in MANAGED_ENV:
        if key in arm:
            os.environ[key] = arm[key]
        else:
            os.environ.pop(key, None)


# Variables an arm may control; anything else is left alone so the harness
# cannot silently mutate unrelated process state.
MANAGED_ENV = ("MOTIONBRICKS_MLX_OPT", "MOTIONBRICKS_MLX_AGENT",
               "MOTIONBRICKS_MLX_QUANT")


def build_agent(device: str, styles_dir: Path, bundle_path: Path, seed: int):
    bundle = Bundle(bundle_path, device=device)
    pipeline = MlxPipeline(bundle)
    styles = {name: load_style(styles_dir / f"{name}.mbstyle") for name in STYLES}
    agent = Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
    agent.reset(styles["walk"])
    return bundle, styles, agent


def paired_measure(device: str, steps: int, styles_dir: Path, bundle_path: Path,
                   seed: int, arms: list[tuple[str, dict[str, str]]],
                   tokens_too: bool) -> dict:
    """Interleaved A/B(/C...) on the identical step sequence, order rotated.

    Every arm runs the same step *n* with the same command before any arm runs
    step *n+1*; the arm that goes first inside a step rotates, so no arm
    systematically pays the "first in the step" cost.
    """
    built = []
    for label, arm in arms:
        apply_arm(arm)
        bundle, styles, agent = build_agent(device, styles_dir, bundle_path, seed)
        # Warm up outside the timed region so the first-step constant work
        # (mask/embedding caches, compile variants) is not charged to one arm's
        # step 0 only.
        agent.plan(Command(style=styles["walk"], seed=seed), styles["walk"])
        mx.eval(mx.array(agent.current_motion.roots.sum()))
        built.append((label, arm, bundle, styles, agent))

    timings: dict[str, list[float]] = {label: [] for label, _, _, _, _ in built}
    tokens: dict[str, list[np.ndarray]] = {label: [] for label, _, _, _, _ in built}
    for index in range(steps):
        order = list(range(len(built)))
        if index % len(built):
            order = order[index % len(built):] + order[:index % len(built)]
        for position in order:
            label, arm, bundle, styles, agent = built[position]
            apply_arm(arm)
            name = STYLES[index % len(STYLES)]
            style = styles[name]
            if index and index % 4 == 0:
                # ``advance`` moves the arm's own context; identical schedule on
                # every arm because they started from the same state.
                agent.advance(3)
            command = Command(style=style, movement=(0.0, 0.0, 0.0),
                              facing=(0.0, 0.0, 1.0), seed=seed + index)
            started = time.perf_counter()
            motion, info = agent.plan(command, style, capture_trace=tokens_too)
            mx.eval(mx.array(motion.roots.sum()))
            elapsed = (time.perf_counter() - started) * 1000.0
            timings[label].append(elapsed)
            if tokens_too:
                tokens[label].append(np.asarray(info["trace"]["pose.tokens"]).copy())

    labels = [label for label, _, _, _, _ in built]
    baseline = labels[0]
    report: dict = {"device": device, "steps": steps,
                    "arms": {label: arm for label, arm, _, _, _ in built},
                    "order": labels}
    for label in labels:
        values = timings[label]
        report.setdefault("per_arm", {})[label] = {
            "p50_ms": statistics.median(values),
            "p90_ms": percentile(values, 0.9),
            "max_ms": max(values),
            "mean_ms": statistics.fmean(values),
            "timings_ms": values,
        }
    for label in labels[1:]:
        ratios = [b / a for a, b in zip(timings[label], timings[baseline])]
        ordered = sorted(ratios)
        report.setdefault("paired", {})[label] = {
            "vs": baseline,
            "p50_ratio": statistics.median(ratios),
            "q1_ratio": ordered[len(ordered) // 4],
            "q3_ratio": ordered[(3 * len(ordered)) // 4],
            "mean_ratio": statistics.fmean(ratios),
            "ratios": ratios,
        }
    if tokens_too:
        identity = {}
        for label in labels[1:]:
            same_steps = sum(
                1 for a, b in zip(tokens[baseline], tokens[label])
                if np.array_equal(a, b))
            total = sum(a.size for a in tokens[baseline])
            differing = sum(int(np.count_nonzero(a != b))
                            for a, b in zip(tokens[baseline], tokens[label]))
            identity[label] = {
                "identical_steps": same_steps, "steps": len(tokens[label]),
                "differing_tokens": differing, "total_tokens": total}
        report["token_identity"] = identity
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--styles", type=Path, default=DEFAULT_STYLES)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--devices", default="cpu,gpu")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the determinism re-plan (saves one plan)")
    parser.add_argument("--paired", nargs="+", default=None, metavar="ARM",
                        help="two or more arms as 'KEY=VALUE;KEY=VALUE'; the "
                             "first is the baseline and they are interleaved")
    parser.add_argument("--paired-tokens", action="store_true",
                        help="with --paired, also compare pose.tokens per step")
    args = parser.parse_args()

    if not (args.bundle / "manifest.json").is_file():
        print(f"error: {args.bundle} has no manifest.json", file=sys.stderr)
        return 2

    if args.paired:
        arms = [(f"arm{index}", parse_arm(spec))
                for index, spec in enumerate(args.paired)]
        for device in args.devices.split(","):
            device = device.strip()
            if not device:
                continue
            report = paired_measure(device, args.steps, args.styles, args.bundle,
                                    args.seed, arms, args.paired_tokens)
            print(f"device {report['device']}  steps {report['steps']}  "
                  f"order {report['order']}")
            for label, arm in report["arms"].items():
                stats = report["per_arm"][label]
                print(f"  {label}  {arm or '(clean env)'}")
                print(f"    per-plan ms   p50 {stats['p50_ms']:.2f}   "
                      f"p90 {stats['p90_ms']:.2f}   max {stats['max_ms']:.2f}")
            for label, ratio in report.get("paired", {}).items():
                print(f"  paired {label} / {ratio['vs']}   "
                      f"p50 {ratio['p50_ratio']:.4f}x "
                      f"[{ratio['q1_ratio']:.3f}-{ratio['q3_ratio']:.3f}]  "
                      f"mean {ratio['mean_ratio']:.4f}x")
            for label, identity in report.get("token_identity", {}).items():
                print(f"  tokens {label} vs {report['order'][0]}: "
                      f"{identity['identical_steps']}/{identity['steps']} steps identical, "
                      f"{identity['differing_tokens']}/{identity['total_tokens']} "
                      f"tokens differ")
        if args.json:
            args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {args.json}")
        return 0

    results = []
    for device in args.devices.split(","):
        device = device.strip()
        if not device:
            continue
        result = measure(device, args.steps, args.styles, args.bundle, args.seed,
                         not args.no_verify)
        results.append(result)
        print(f"device {result['device']}  steps {result['steps']}")
        print(f"  per-plan ms   p50 {result['p50_ms']:.1f}   p90 {result['p90_ms']:.1f}   "
              f"max {result['max_ms']:.1f}   min {result['min_ms']:.1f}   "
              f"mean {result['mean_ms']:.1f}")
        print(f"  cold start    {result['cold_start_ms']:.0f} ms (bundle load + first plan)")
        print(f"  peak memory   MLX {result['mlx_peak_memory_bytes'] / 1e6:.0f} MB, "
              f"process RSS {result['process_peak_rss_bytes'] / 1e6:.0f} MB")
        if result["deterministic"] is not None:
            print(f"  deterministic {result['deterministic']}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
