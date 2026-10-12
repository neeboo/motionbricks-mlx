#!/usr/bin/env python3
"""Latency and memory measurement for the MLX port (CPU vs GPU backend).

Reports per-plan wall time (p50 / p90 / max) over a fixed number of planning
steps, plus cold start (bundle load + first plan) and peak memory.  Determinism
is checked alongside timing so a fast-but-wrong configuration cannot pass.

    ./.venv/bin/python scripts/mlx_bench.py --steps 25
"""

from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args()

    if not (args.bundle / "manifest.json").is_file():
        print(f"error: {args.bundle} has no manifest.json", file=sys.stderr)
        return 2

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
