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


def bake_measure(device: str, styles_dir: Path, bundle_path: Path, batch_sizes,
                 seed: int, repeats: int, homogeneous: bool = False,
                 metrics: bool = False) -> dict:
    """Round-3 batch bake: ``B`` independent scenarios, serial vs batched pose.

    The reference arm plans every scenario **alone** through the ordinary
    pipeline (that *is* the ``B = 1`` result each batch row is compared with);
    the batched arm feeds one batched pose forward per token-count group back
    through the accepted pipeline via the ``pose_logits`` seam.  The two arms are
    timed inside one process, alternating which goes first, so drift affects
    both.

    ``homogeneous`` uses one style for every scenario, so the whole batch shares
    a duration and therefore a single batched forward: that is the *best case*
    for this design, and the difference against the default (styles cycling) is
    the cost of token-count fragmentation.

    ``metrics`` additionally reports the round-4 perturbation/margin numbers for
    this round-3 stage set, so the two rounds' safety factors are computed by the
    same code on the same scenarios.  It costs an extra snapshot pass and does
    not touch either timed arm.
    """
    from motionbricks_mlx.models import bake as bake_module

    bundle = Bundle(bundle_path, device=device)
    pipeline = MlxPipeline(bundle)
    style_names = [STYLES[0]] if homogeneous else list(STYLES)
    report: dict = {"device": device, "repeats": repeats,
                    "homogeneous": homogeneous, "batches": {}}
    for size in batch_sizes:
        scenarios = [
            bake_module.Scenario(
                label=f"B{size}-s{index}-{style_names[index % len(style_names)]}",
                style=load_style(styles_dir / f"{style_names[index % len(style_names)]}.mbstyle"),
                seed=seed + index,
                movement=(0.2, 0.0, 0.0))
            for index in range(size)]
        serial_times: list[float] = []
        batched_times: list[float] = []
        single_times: list[float] = []
        reference: list = []
        batched_runs: list = []
        # One untimed warm-up of each arm, so neither pays first-call compile
        # cost inside a timed region.
        records, single = bake_module.record_serial(pipeline, bundle, scenarios)
        mx.eval(*bake_module.batched_logits(pipeline, records))
        for repeat in range(repeats):
            order = (True, False) if repeat % 2 == 0 else (False, True)
            for arm_is_serial in order:
                if arm_is_serial:
                    started = time.perf_counter()
                    records, single = bake_module.record_serial(pipeline, bundle,
                                                                scenarios)
                    serial_times.append((time.perf_counter() - started) * 1000.0)
                    single_times = single
                    reference = [record.run for record in records]
                else:
                    started = time.perf_counter()
                    logits = bake_module.batched_logits(pipeline, records)
                    runs = bake_module.rerun_batched(pipeline, bundle, records, logits)
                    batched_times.append((time.perf_counter() - started) * 1000.0)
                    batched_runs = runs
        import statistics as _statistics
        comparison = bake_module.compare(reference, batched_runs)
        groups = bake_module.group_by_tokens(records)
        entry = {
            "batch_size": size,
            "serial_ms_p50": _statistics.median(serial_times),
            "serial_ms_min": min(serial_times),
            "batched_ms_p50": _statistics.median(batched_times),
            "batched_ms_min": min(batched_times),
            "single_scenario_ms_p50": _statistics.median(single_times),
            "single_scenario_ms_batched": _statistics.median(batched_times) / size,
            "throughput_serial_per_s": size / (_statistics.median(serial_times) / 1000.0),
            "throughput_batched_per_s": size / (_statistics.median(batched_times) / 1000.0),
            "speedup": (_statistics.median(serial_times)
                        / _statistics.median(batched_times)),
            "token_groups": {str(k): len(v) for k, v in groups.items()},
            "serial_times_ms": serial_times,
            "batched_times_ms": batched_times,
            "comparison": comparison,
        }
        if metrics:
            snapshots, _ = bake_module.record_serial(pipeline, bundle, scenarios,
                                                     capture_stages=True)
            perturbations = bake_module.stage_perturbations(snapshots, batched_runs)
            margins = bake_module.decision_margins(snapshots)
            entry["perturbations"] = perturbations
            entry["margins"] = margins
            entry["safety"] = bake_module.safety_factor(perturbations, margins)
        report["batches"][str(size)] = entry
        print(f"B={size:<3d} serial p50 {entry['serial_ms_p50']:8.2f} ms   "
              f"batched p50 {entry['batched_ms_p50']:8.2f} ms   "
              f"speedup {entry['speedup']:.3f}x   "
              f"per-scenario {entry['single_scenario_ms_batched']:6.2f} ms   "
              f"throughput {entry['throughput_batched_per_s']:6.2f}/s")
        print(f"       tokens {comparison['token_identical']}/{comparison['scenarios']} "
              f"identical, uniforms {comparison['uniform_identical']}/{comparison['scenarios']}, "
              f"{comparison['differing_tokens']}/{comparison['total_tokens']} tokens differ; "
              f"worst roots {comparison['worst_roots_m']:.3e} m, "
              f"rotations {comparison['worst_rotations']:.3e}, "
              f"angle {comparison['worst_angle_deg']:.4f} deg; "
              f"groups {entry['token_groups']}")
        if "safety" in entry:
            safety = entry["safety"]
            print(f"       safety pose {safety['pose_perturbation']:.3e}/"
                  f"{safety['pose_margin']:.3e} = {safety['pose_ratio']:.4f}x")
    bundle.release()
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
    parser.add_argument("--bake", default=None, metavar="B1,B2,...",
                        help="round-3 batch bake: plan B independent scenarios "
                             "serially and batched, and compare every row with "
                             "its own B=1 result (e.g. --bake 1,2,4,8,16)")
    parser.add_argument("--bake-repeats", type=int, default=5)
    parser.add_argument("--bake-homogeneous", action="store_true",
                        help="with --bake, use one style for every scenario so "
                             "the batch is duration-homogeneous (best case)")
    parser.add_argument("--bake-metrics", action="store_true",
                        help="with --bake-stages pose, also report the round-4 "
                             "perturbation/margin safety numbers")
    parser.add_argument("--bake-stages", choices=("pose", "root", "all"),
                        default="pose",
                        help="which stages the bake batches: 'pose' is the "
                             "round-3 bake, 'root' adds the root stage, 'all' "
                             "adds the root and decoder stages (the two new "
                             "switches are default-off; this turns them on)")
    parser.add_argument("--bake-single-pass", action="store_true",
                        help="round 5: with --bake, time three arms -- serial, "
                             "round-4 drive_batch and the round-5 single-pass "
                             "orchestrator -- and pin that the two batch arms "
                             "agree bit for bit")
    args = parser.parse_args()

    if not (args.bundle / "manifest.json").is_file():
        print(f"error: {args.bundle} has no manifest.json", file=sys.stderr)
        return 2

    if args.bake:
        sizes = [int(item) for item in args.bake.split(",") if item.strip()]
        enable_bake_stages(args.bake_stages)
        reports = []
        for device in args.devices.split(","):
            device = device.strip()
            if not device:
                continue
            if args.bake_single_pass:
                report = bake_single_pass_measure(device, args.styles, args.bundle,
                                                  sizes, args.seed, args.bake_repeats,
                                                  homogeneous=args.bake_homogeneous)
            elif args.bake_stages == "pose":
                report = bake_measure(device, args.styles, args.bundle, sizes,
                                      args.seed, args.bake_repeats,
                                      homogeneous=args.bake_homogeneous,
                                      metrics=args.bake_metrics)
            else:
                report = bake_full_measure(device, args.styles, args.bundle, sizes,
                                           args.seed, args.bake_repeats,
                                           homogeneous=args.bake_homogeneous)
            reports.append(report)
        if args.json:
            args.json.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {args.json}")
        return 0

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


# ---------------------------------------------------------------------------
# Round 4: the full-stage bake (root + decoder on top of round 3's pose batch)
# ---------------------------------------------------------------------------

BAKE_STAGE_SWITCHES = {
    "pose": "",
    "root": "bake_root_batch=1",
    "all": "bake_root_batch=1,bake_decoder_batch=1",
}


def enable_bake_stages(level: str) -> None:
    """Turn on the round-4 bake switches for this process (default: off)."""
    extra = BAKE_STAGE_SWITCHES[level]
    current = os.environ.get("MOTIONBRICKS_MLX_OPT")
    if not extra:
        if current is not None:
            os.environ["MOTIONBRICKS_MLX_OPT"] = current
        return
    os.environ["MOTIONBRICKS_MLX_OPT"] = f"{current},{extra}" if current else extra


def bake_full_measure(device: str, styles_dir: Path, bundle_path: Path, batch_sizes,
                      seed: int, repeats: int, homogeneous: bool = False) -> dict:
    """Round-4 bake: every batched stage, timed end to end as one region.

    "Serial" is ``B`` scenarios planned one after another through the accepted
    pipeline; "batched" is the stage-synchronised driver of
    :mod:`motionbricks_mlx.models.bake`, which substitutes all three batched
    forwards and reports ``B`` final motions.  Both arms warm up before timing and
    alternate which goes first, so drift is shared.
    """
    import statistics as _statistics

    from motionbricks_mlx.models import bake as bake_module

    stages = bake_module.batched_stages()
    bundle = Bundle(bundle_path, device=device)
    pipeline = MlxPipeline(bundle)
    style_names = [STYLES[0]] if homogeneous else list(STYLES)
    report: dict = {"device": device, "repeats": repeats, "homogeneous": homogeneous,
                    "stages": sorted(stages), "batches": {}}
    for size in batch_sizes:
        scenarios = [
            bake_module.Scenario(
                label=f"B{size}-s{index}-{style_names[index % len(style_names)]}",
                style=load_style(styles_dir / f"{style_names[index % len(style_names)]}.mbstyle"),
                seed=seed + index,
                movement=(0.2, 0.0, 0.0))
            for index in range(size)]
        # Reference arm with stage snapshots and the reference single-scenario
        # p50.  Untimed: the snapshot reads force a host drain per stage.
        reference, single = bake_module.record_serial(pipeline, bundle, scenarios,
                                                      capture_stages=True)
        # One untimed warm-up of each arm, so neither pays compile cost in a
        # timed region: the batched forwards compile one variant per shape.
        bake_module.drive_batch(pipeline, bundle, scenarios, stages, capture_trace=False)
        bake_module.record_serial(pipeline, bundle, scenarios, capture_trace=False)

        serial_times: list[float] = []
        batched_times: list[float] = []
        for repeat in range(repeats):
            order = (True, False) if repeat % 2 == 0 else (False, True)
            for arm_is_serial in order:
                if arm_is_serial:
                    started = time.perf_counter()
                    bake_module.record_serial(pipeline, bundle, scenarios,
                                              capture_trace=False)
                    serial_times.append((time.perf_counter() - started) * 1000.0)
                else:
                    started = time.perf_counter()
                    bake_module.drive_batch(pipeline, bundle, scenarios, stages,
                                            capture_trace=False)
                    batched_times.append((time.perf_counter() - started) * 1000.0)

        campaign = bake_module.drive_batch(pipeline, bundle, scenarios, stages,
                                           capture_trace=True)
        comparison = bake_module.compare([record.run for record in reference],
                                        campaign.runs)
        perturbations = bake_module.stage_perturbations(reference, campaign.runs)
        margins = bake_module.decision_margins(reference)
        groups = {str(record.tokens): 0 for record in reference}
        for record in reference:
            groups[str(record.tokens)] += 1
        entry = {
            "batch_size": size,
            "serial_ms_p50": _statistics.median(serial_times),
            "serial_ms_min": min(serial_times),
            "batched_ms_p50": _statistics.median(batched_times),
            "batched_ms_min": min(batched_times),
            "single_scenario_ms_p50": _statistics.median(single),
            "single_scenario_ms_batched": _statistics.median(batched_times) / size,
            "throughput_serial_per_s": size / (_statistics.median(serial_times) / 1000.0),
            "throughput_batched_per_s": size / (_statistics.median(batched_times) / 1000.0),
            "speedup": _statistics.median(serial_times) / _statistics.median(batched_times),
            "token_groups": groups,
            "suspensions": campaign.suspensions,
            "serial_times_ms": serial_times,
            "batched_times_ms": batched_times,
            "comparison": comparison,
            "perturbations": perturbations,
            "margins": margins,
            "safety": bake_module.safety_factor(perturbations, margins),
        }
        if homogeneous:
            stages_table = bake_module.stage_throughput(
                pipeline, campaign.drivers, repeats=max(repeats, 5))
            entry["stage_throughput"] = stages_table
            host = bake_module.host_pass_timing(pipeline, bundle, scenarios,
                                                campaign.drivers, repeats=repeats)
            stage_sum = sum(row["per_scenario_ms"] for row in stages_table.values()
                            if row["batch"] == size)
            host["batched_stage_sum_ms"] = stage_sum
            host["single_pass_ms_per_scenario"] = stage_sum + host["per_scenario_ms_p50"]
            host["measured_ms_per_scenario"] = (entry["batched_ms_p50"] / size)
            host["driver_overhead_ms_per_scenario"] = (
                entry["batched_ms_p50"] / size - host["single_pass_ms_per_scenario"])
            entry["residual"] = host
        report["batches"][str(size)] = entry
        safety = entry["safety"]
        print(f"B={size:<3d} serial p50 {entry['serial_ms_p50']:8.2f} ms   "
              f"batched p50 {entry['batched_ms_p50']:8.2f} ms   "
              f"speedup {entry['speedup']:.3f}x   "
              f"per-scenario {entry['single_scenario_ms_batched']:6.2f} ms   "
              f"throughput {entry['throughput_batched_per_s']:6.2f}/s")
        print(f"       tokens {comparison['token_identical']}/{comparison['scenarios']} "
              f"identical, uniforms {comparison['uniform_identical']}/{comparison['scenarios']}, "
              f"duration {comparison['selected_identical']}/{comparison['scenarios']}, "
              f"{comparison['differing_tokens']}/{comparison['total_tokens']} tokens differ; "
              f"worst roots {comparison['worst_roots_m']:.3e} m, "
              f"rotations {comparison['worst_rotations']:.3e}, "
              f"angle {comparison['worst_angle_deg']:.4f} deg")
        print(f"       safety pose {safety['pose_perturbation']:.3e}/"
              f"{safety['pose_margin']:.3e} = {safety['pose_ratio']:.4f}x   "
              f"duration {safety['duration_perturbation']:.3e}/"
              f"{safety['duration_margin']:.3e} = "
              f"{safety['duration_ratio'] if safety['duration_ratio'] is None else format(safety['duration_ratio'], '.4f')}x   "
              f"stages {sorted(stages)}")
        if "residual" in entry:
            r = entry["residual"]
            print(f"       per-scenario: batched stages {r['batched_stage_sum_ms']:5.2f} ms + "
                  f"host pass {r['per_scenario_ms_p50']:5.2f} ms = "
                  f"single-pass {r['single_pass_ms_per_scenario']:5.2f} ms;  "
                  f"measured {r['measured_ms_per_scenario']:5.2f} ms;  "
                  f"driver re-execution {r['driver_overhead_ms_per_scenario']:5.2f} ms")
        for stage, row in sorted(entry.get("stage_throughput", {}).items()):
            print(f"       stage {stage:<8s} serial {row['serial_p50_ms']:7.3f} ms/plan   "
                  f"batched {row['batched_p50_ms']:8.3f} ms   "
                  f"per-scenario {row['per_scenario_ms']:6.3f} ms   "
                  f"speedup {row['speedup']:5.2f}x")
    bundle.release()
    return report

# ---------------------------------------------------------------------------
# Round 5: the single-pass orchestrator, three arms (appended, round 4 untouched)
# ---------------------------------------------------------------------------

def bake_single_pass_measure(device: str, styles_dir: Path, bundle_path: Path,
                             batch_sizes, seed: int, repeats: int,
                             homogeneous: bool = False) -> dict:
    """Round-5 bake: **serial / round-4 batch / round-5 single pass** in one process.

    The three arms run the identical scenario set inside one process, in a
    **rotated** order per repeat, after one untimed warm-up each.  The round-4 arm
    is :func:`bake.drive_batch` (unchanged, suspend-and-restart) and is this
    round's control; the round-5 arm is :func:`bake.drive_batch_single_pass`,
    which drives one resumable transition generator per scenario so every host
    step runs once.  ``compare_bitwise`` pins that the two batch arms agree
    exactly before any speedup from the third arm is quoted.

    With ``homogeneous`` the batch also gets the stage table (measured on the
    round-4 arm's recorded inputs, which are the same rows the single pass
    consumes) and both host passes, which is what decomposes the single pass into
    batched forward / host glue / orchestration overhead.
    """
    import statistics as _statistics
    import time as _time

    from motionbricks_mlx.models import bake as bake_module

    stages = bake_module.batched_stages()
    bundle = Bundle(bundle_path, device=device)
    pipeline = MlxPipeline(bundle)
    style_names = [STYLES[0]] if homogeneous else list(STYLES)
    report: dict = {"device": device, "repeats": repeats, "homogeneous": homogeneous,
                    "stages": sorted(stages), "batches": {}}
    for size in batch_sizes:
        scenarios = [
            bake_module.Scenario(
                label=f"B{size}-s{index}-{style_names[index % len(style_names)]}",
                style=load_style(styles_dir / f"{style_names[index % len(style_names)]}.mbstyle"),
                seed=seed + index,
                movement=(0.2, 0.0, 0.0))
            for index in range(size)]
        # Reference arm with stage snapshots and the per-scenario single plan time.
        reference, single = bake_module.record_serial(pipeline, bundle, scenarios,
                                                      capture_stages=True)
        # One untimed warm-up of each arm: the batched forwards and the generator
        # machinery both compile on first use and must not be charged to a repeat.
        bake_module.record_serial(pipeline, bundle, scenarios, capture_trace=False)
        bake_module.drive_batch(pipeline, bundle, scenarios, stages, capture_trace=False)
        bake_module.drive_batch_single_pass(pipeline, bundle, scenarios, stages,
                                            capture_trace=False)

        arm_order = ("serial", "batch", "single")
        timings: dict[str, list[float]] = {name: [] for name in arm_order}
        for repeat in range(repeats):
            rotation = arm_order[repeat % len(arm_order):] + arm_order[:repeat % len(arm_order)]
            for name in rotation:
                started = _time.perf_counter()
                if name == "serial":
                    bake_module.record_serial(pipeline, bundle, scenarios,
                                              capture_trace=False)
                elif name == "batch":
                    bake_module.drive_batch(pipeline, bundle, scenarios, stages,
                                            capture_trace=False)
                else:
                    bake_module.drive_batch_single_pass(pipeline, bundle, scenarios,
                                                        stages, capture_trace=False)
                timings[name].append((_time.perf_counter() - started) * 1000.0)

        campaign_batch = bake_module.drive_batch(pipeline, bundle, scenarios, stages,
                                                 capture_trace=True)
        campaign_single = bake_module.drive_batch_single_pass(
            pipeline, bundle, scenarios, stages, capture_trace=True)
        reference_runs = [record.run for record in reference]
        comparison = bake_module.compare(reference_runs, campaign_single.runs)
        control = bake_module.compare(reference_runs, campaign_batch.runs)
        cross = bake_module.compare_bitwise(campaign_batch.runs, campaign_single.runs)
        perturbations = bake_module.stage_perturbations(reference, campaign_single.runs)
        margins = bake_module.decision_margins(reference)
        groups: dict[str, int] = {}
        for record in reference:
            groups[str(record.tokens)] = groups.get(str(record.tokens), 0) + 1

        def arm(name: str) -> dict:
            values = timings[name]
            median = _statistics.median(values)
            return {
                "p50_ms": median,
                "min_ms": min(values),
                "ms_per_scenario": median / size,
                "throughput_per_s": size / (median / 1000.0),
                "times_ms": values,
            }

        entry: dict = {
            "batch_size": size,
            "arms": {name: arm(name) for name in arm_order},
            "speedup_batch": arm("serial")["p50_ms"] / arm("batch")["p50_ms"],
            "speedup_single": arm("serial")["p50_ms"] / arm("single")["p50_ms"],
            "single_vs_batch": arm("batch")["p50_ms"] / arm("single")["p50_ms"],
            "single_scenario_ms_p50": _statistics.median(single),
            "token_groups": groups,
            "suspensions_round4": campaign_batch.suspensions,
            "comparison": comparison,
            "round4_control": control,
            "cross_arm_bitwise": cross,
            "perturbations": perturbations,
            "margins": margins,
            "safety": bake_module.safety_factor(perturbations, margins),
        }
        if homogeneous:
            stages_table = bake_module.stage_throughput(
                pipeline, campaign_batch.drivers, repeats=max(repeats, 5))
            # ``stage_throughput`` reports the four forwards *and* a derived
            # ``root`` row that is ``probe + plan``.  Round 4 summed every row, so
            # it counted the duration probe twice; this round sums the four
            # forwards only and reports the round-4 basis beside it.
            forwards = {name: row for name, row in stages_table.items() if name != "root"}
            stage_sum = sum(row["per_scenario_ms"] for row in forwards.values()
                            if row["batch"] == size)
            stage_sum_with_root = sum(row["per_scenario_ms"] for row in stages_table.values()
                                      if row["batch"] == size)
            host5 = bake_module.host_pass_timing_single_pass(
                pipeline, bundle, scenarios, stages, campaign_single.rows,
                repeats=repeats)
            host4 = bake_module.host_pass_timing(
                pipeline, bundle, scenarios, campaign_batch.drivers, repeats=repeats)
            measured5 = entry["arms"]["single"]["ms_per_scenario"]
            measured4 = entry["arms"]["batch"]["ms_per_scenario"]
            entry["stage_throughput"] = stages_table
            entry["residual"] = {
                "batched_stage_sum_ms": stage_sum,
                "batched_stage_sum_with_derived_root_ms": stage_sum_with_root,
                "host_pass_ms_per_scenario": host5["per_scenario_ms_p50"],
                "round4_host_pass_ms_per_scenario": host4["per_scenario_ms_p50"],
                "single_pass_ms_per_scenario": stage_sum + host5["per_scenario_ms_p50"],
                "measured_ms_per_scenario": measured5,
                "orchestration_overhead_ms_per_scenario": (
                    measured5 - stage_sum - host5["per_scenario_ms_p50"]),
                "round4_measured_ms_per_scenario": measured4,
                "round4_driver_overhead_ms_per_scenario": (
                    measured4 - stage_sum - host4["per_scenario_ms_p50"]),
                "host_pass_totals_ms": host5["totals_ms"],
            }
        entry["manifest"] = bake_module.manifest_fields(
            batch_size=size, stages=stages,
            grouping="duration-homogeneous" if homogeneous else "duration-mixed",
            token_groups=groups, comparison=comparison, safety=entry["safety"])
        report["batches"][str(size)] = entry

        serial, batch, single_arm = arm("serial"), arm("batch"), arm("single")
        print(f"B={size:<3d} serial p50 {serial['p50_ms']:8.2f} ms   "
              f"round4 p50 {batch['p50_ms']:8.2f} ms ({entry['speedup_batch']:.3f}x)   "
              f"single p50 {single_arm['p50_ms']:8.2f} ms "
              f"({entry['speedup_single']:.3f}x)   "
              f"single/round4 {entry['single_vs_batch']:.3f}x")
        print(f"       per scenario: serial {serial['ms_per_scenario']:6.2f} ms   "
              f"round4 {batch['ms_per_scenario']:6.2f} ms   "
              f"single {single_arm['ms_per_scenario']:6.2f} ms   "
              f"throughput {single_arm['throughput_per_s']:6.2f}/s   "
              f"groups {groups}")
        print(f"       single pass vs B=1: tokens {comparison['token_identical']}/"
              f"{comparison['scenarios']}, uniforms {comparison['uniform_identical']}, "
              f"duration {comparison['selected_identical']}, "
              f"{comparison['differing_tokens']}/{comparison['total_tokens']} differ; "
              f"worst roots {comparison['worst_roots_m']:.3e} m, "
              f"rotations {comparison['worst_rotations']:.3e}, "
              f"angle {comparison['worst_angle_deg']:.4f} deg")
        print(f"       single vs round4 bitwise: tokens {cross['tokens_identical']}/"
              f"{cross['scenarios']}, uniforms {cross['uniforms_identical']}, "
              f"duration {cross['selected_identical']}, roots {cross['roots_identical']}, "
              f"rotations {cross['rotations_identical']}, "
              f"bit_identical={cross['bit_identical']}")
        safety = entry["safety"]
        print(f"       safety pose {safety['pose_perturbation']:.3e}/"
              f"{safety['pose_margin']:.3e} = "
              f"{safety['pose_ratio'] if safety['pose_ratio'] is None else format(safety['pose_ratio'], '.4f')}x   "
              f"duration {safety['duration_perturbation']:.3e}/"
              f"{safety['duration_margin']:.3e} = "
              f"{safety['duration_ratio'] if safety['duration_ratio'] is None else format(safety['duration_ratio'], '.4f')}x")
        if "residual" in entry:
            r = entry["residual"]
            print(f"       per-scenario: batched stages {r['batched_stage_sum_ms']:5.2f} ms + "
                  f"host pass {r['host_pass_ms_per_scenario']:5.2f} ms -> single-pass "
                  f"{r['single_pass_ms_per_scenario']:5.2f} ms;  measured "
                  f"{r['measured_ms_per_scenario']:5.2f} ms;  orchestration overhead "
                  f"{r['orchestration_overhead_ms_per_scenario']:5.2f} ms")
            print(f"       round-4 control: host pass "
                  f"{r['round4_host_pass_ms_per_scenario']:5.2f} ms;  measured "
                  f"{r['round4_measured_ms_per_scenario']:5.2f} ms;  driver re-execution "
                  f"{r['round4_driver_overhead_ms_per_scenario']:5.2f} ms")
        for stage, row in sorted(entry.get("stage_throughput", {}).items()):
            print(f"       stage {stage:<8s} serial {row['serial_p50_ms']:7.3f} ms/plan   "
                  f"batched {row['batched_p50_ms']:8.3f} ms   "
                  f"per-scenario {row['per_scenario_ms']:6.3f} ms   "
                  f"speedup {row['speedup']:5.2f}x")
    bundle.release()
    return report


if __name__ == "__main__":
    raise SystemExit(main())
