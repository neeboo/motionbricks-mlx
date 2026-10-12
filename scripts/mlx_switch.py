#!/usr/bin/env python3
"""Switch from motion A to motion B and *measure the seam* -- with a control that
must show a jump if the method does nothing.

The method (``seamless``): plan B while the agent's context still is A's last four
frames, so B's boundary frames 0..3 -- and therefore its ``local_root`` velocities
and the inherited ``apply_context_blend`` -- come from A.  Only the style object
that builds the target window (frames 4..7) changes.  No reset, no new planning
code: ``src/motionbricks_mlx/data/switch.py`` says exactly where each piece comes
from.

The control (``naive``): generate A and B *independently* (each from its own style
reseed -- what a style change used to do) and butt the clips together, optionally
sliding B so its first frame sits at A's handoff pose so the control cannot be
dismissed as a teleport.

Single switch, writes a reference-shaped plan document::

    ./.venv/bin/python scripts/mlx_switch.py --style walk --to walk_zombie \\
        --seed 1 --seam-at 20 --device gpu --out /tmp/switch.json

A -> B -> A as one joined clip (the clip the page plays)::

    ./.venv/bin/python scripts/mlx_switch.py --style walk --to walk_zombie \\
        --back walk --seam-at 20 --seam-at-back 20 --out /tmp/switch-aba.json

The control matrix + the C-line quality gate::

    ./.venv/bin/python scripts/mlx_switch.py --style walk --to walk_zombie \\
        --control --quality --out /tmp/switch.json --report /tmp/control.json

``--out`` is required, exactly like ``scripts/mlx_plan.py`` (round-2 rule).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motionbricks_mlx.data.style import load_style                             # noqa: E402
from motionbricks_mlx.data.switch import (LEG_KINDS, SEAM_CRITERION, Leg,       # noqa: E402
                                          SwitchPlan, Switcher, heading_at, join_legs,
                                          place_at, seam_metrics)
from motionbricks_mlx.eval.document import PlanStep                             # noqa: E402
from motionbricks_mlx.eval.metrics import (boundary_error, foot_sliding,        # noqa: E402
                                          grounding, jitter)
from motionbricks_mlx.eval.thresholds import (AIR_THRESHOLD_M, CONTACT_THRESHOLD_M,  # noqa: E402
                                             FREEZE_EPS_DEG, THRESHOLDS, judge)
from motionbricks_mlx.export.plan_json import (FPS, Step, build_document,       # noqa: E402
                                               world_positions_xyz, write_document)
from motionbricks_mlx.models.bundle import Bundle                               # noqa: E402
from motionbricks_mlx.models.transition import MlxPipeline                      # noqa: E402

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "checkpoints/g1-f32-mlx"
DEFAULT_STYLES = Path.home() / "dev/motion-bricks-cpp/repo/generated/styles"
#: The matrix: A->B pairs, the same-style pair first (it is the mechanism's own
#: seam floor: A->A is a plain replan of one style, so whatever it measures is
#: the best any 4-frame-window conditioning can do).
DEFAULT_PAIRS = (
    "walk:walk",
    "walk:walk_zombie",
    "walk:slow_walk",
    "walk:stealth_walk",
    "walk:injured_walk",
    "walk:walk_happy_dance",
    "walk_zombie:walk",
    "slow_walk:injured_walk",
)
DEFAULT_SEEDS = (1, 7)
DEFAULT_SEAMS = (12, 20, 28)
#: Half-width (frames) of the window that straddles a seam in the quality gate.
QUALITY_WINDOW = 8


# ---------------------------------------------------------------------------
# steps and the C-line quality gate (imported read-only, nothing re-implemented)
# ---------------------------------------------------------------------------
def step_of(roots, rotations, target_roots, target_rotations, parents, rest,
            joint_names, *, label: str, style: str, seed: int, advance: int,
            world: np.ndarray | None = None) -> PlanStep:
    """One :class:`PlanStep` the C-line metrics can read."""
    roots = np.asarray(roots, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    return PlanStep(
        label=label, style=style, seed=int(seed), advance=int(advance), fps=int(FPS),
        roots=roots, rotations_xyzw=rotations,
        target_roots=np.asarray(target_roots, dtype=np.float64),
        target_rotations_xyzw=np.asarray(target_rotations, dtype=np.float64),
        world_positions=world if world is not None
        else world_positions_xyz(roots, rotations, parents, rest),
        joint_names=list(joint_names))


def step_of_plan(plan: SwitchPlan, bundle, *, label: str, seed: int) -> PlanStep:
    return step_of(plan.roots, plan.rotations_xyzw, plan.target_roots,
                   plan.target_local_rotations_xyzw, bundle.joint_parents,
                   bundle.neutral_joints, bundle.joint_names, label=label,
                   style="/".join(plan.styles), seed=seed, advance=0)


def slice_step(step: PlanStep, start: int, end: int) -> PlanStep:
    """A sub-window of a step; the target block is carried, not re-derived.

    ``PlanStep`` carries ``world_positions`` directly, so slicing it cannot
    disagree with the sliced roots/rotations (no second FK pass).
    """
    return PlanStep(label=step.label, style=step.style, seed=step.seed,
                    advance=step.advance, fps=step.fps,
                    roots=step.roots[start:end].copy(),
                    rotations_xyzw=step.rotations_xyzw[start:end].copy(),
                    target_roots=step.target_roots, target_rotations_xyzw=step.target_rotations_xyzw,
                    world_positions=step.world_positions[start:end].copy(),
                    joint_names=step.joint_names)


def quality_window(step: PlanStep, center: int, half: int = QUALITY_WINDOW) -> dict:
    """Metric 1/2/3 of one window, normalised per frame so windows compare."""
    start = max(0, center - half)
    end = min(step.frames, center + half)
    window = slice_step(step, start, end)
    slide = foot_sliding(window, CONTACT_THRESHOLD_M)
    jerk = jitter(window, FREEZE_EPS_DEG)
    ground = grounding(window, AIR_THRESHOLD_M)
    frames = max(1, window.frames - 1)
    return {
        "start": int(start), "end": int(end), "frames": int(window.frames),
        "foot_slide_max_m": slide["foot_slide_max_m"],
        "foot_slide_per_frame_mm": slide["foot_slide_max_m"] * 1000.0 / frames,
        "jerk_p95_deg": jerk["jerk_p95_deg"],
        "jerk_max_deg": jerk["jerk_max_deg"],
        "foot_height_min_m": ground["foot_height_min_m"],
        "ground_airborne_frac": ground["ground_airborne_frac"],
    }


def quality_gate(plan: SwitchPlan, bundle) -> dict:
    """Whole-clip C-line gate + seam windows against the clip's own interior.

    Two comparisons, because an absolute C-line limit and a local seam test answer
    different questions:

    * **per leg** -- every leg measured alone against the absolute C-line
      thresholds.  Those thresholds are calibrated on one 44-frame clip
      (``thresholds.py``), and ``foot_slide_max_m`` is an accumulation, so it can
      only be read on a single leg; a joined clip sums over both legs.
    * **seam window vs interior** -- "is the seam worse than the segment
      baseline", measured the only honest way: windows of the *same width* at
      every interior centre and at every seam, seam compared against the interior
      p95 of the same statistic.
    """
    step = step_of_plan(plan, bundle, label="switch", seed=0)
    values = {}
    values.update(foot_sliding(step, CONTACT_THRESHOLD_M))
    values.update(jitter(step, FREEZE_EPS_DEG))
    values.update(grounding(step, AIR_THRESHOLD_M))
    values.update(boundary_error(step))
    verdicts = judge(values, THRESHOLDS)
    legs = []
    for index, leg in enumerate(plan.legs):
        leg_step = step_of(leg.motion.roots, leg.motion.rotations_xyzw,
                           leg.motion.target_roots, leg.motion.target_local_rotations_xyzw,
                           bundle.joint_parents, bundle.neutral_joints, bundle.joint_names,
                           label=f"leg{index}", style=leg.style, seed=0, advance=leg.advance)
        leg_values = {}
        leg_values.update(foot_sliding(leg_step, CONTACT_THRESHOLD_M))
        leg_values.update(jitter(leg_step, FREEZE_EPS_DEG))
        leg_values.update(grounding(leg_step, AIR_THRESHOLD_M))
        leg_values.update(boundary_error(leg_step))
        leg_verdicts = judge(leg_values, THRESHOLDS)
        legs.append({"index": index, "style": leg.style, "frames": int(leg.motion.frames),
                     "metrics": leg_values,
                     "verdicts": [{"key": v.key, "value": v.value, "limit": v.limit,
                                   "direction": v.direction, "passed": v.passed}
                                  for v in leg_verdicts],
                     "pass": all(v.passed for v in leg_verdicts)})
    seams = [entry["joined_frame"] for entry in plan.seams]
    interior = [center for center in range(QUALITY_WINDOW, step.frames - QUALITY_WINDOW + 1)
                if all(abs(center - seam) > QUALITY_WINDOW for seam in seams)]
    interior_rows = [quality_window(step, center) for center in interior]
    seam_rows = [quality_window(step, seam) for seam in seams]
    interior_slide = [row["foot_slide_per_frame_mm"] for row in interior_rows]
    interior_jerk = [row["jerk_p95_deg"] for row in interior_rows]
    interior_height = [row["foot_height_min_m"] for row in interior_rows]
    comparisons = []
    for seam, row in zip(seams, seam_rows):
        entry = {
            "joined_frame": int(seam),
            "window": row,
            "interior_foot_slide_p95_mm": float(np.percentile(interior_slide, 95.0))
            if interior_slide else float("nan"),
            "interior_jerk_p95_p95_deg": float(np.percentile(interior_jerk, 95.0))
            if interior_jerk else float("nan"),
            "interior_foot_height_min_m": float(min(interior_height)) if interior_height
            else float("nan"),
        }
        entry["foot_slide_within_interior_p95"] = bool(
            entry["window"]["foot_slide_per_frame_mm"] <= entry["interior_foot_slide_p95_mm"])
        entry["jerk_within_interior_p95"] = bool(
            entry["window"]["jerk_p95_deg"] <= entry["interior_jerk_p95_p95_deg"])
        entry["penetration_within_interior"] = bool(
            entry["window"]["foot_height_min_m"] >= entry["interior_foot_height_min_m"])
        entry["jerk_over_interior_p95"] = (entry["window"]["jerk_p95_deg"]
                                           / entry["interior_jerk_p95_p95_deg"]
                                           if entry["interior_jerk_p95_p95_deg"] else float("nan"))
        comparisons.append(entry)
    return {
        "whole_clip": values,
        "whole_clip_frames": int(step.frames),
        "verdicts": [{"key": v.key, "value": v.value, "limit": v.limit,
                      "direction": v.direction, "passed": v.passed} for v in verdicts],
        "whole_clip_pass": all(v.passed for v in verdicts),
        "legs": legs,
        "legs_pass": all(entry["pass"] for entry in legs),
        "window_half_frames": QUALITY_WINDOW,
        "interior_windows": len(interior_rows),
        "seams": comparisons,
        "thresholds": {field: getattr(THRESHOLDS, field)
                       for field in THRESHOLDS.__dataclass_fields__},
    }


def leg_quality(step: PlanStep) -> dict:
    """The three C-line quality metrics of one leg plus its absolute verdicts."""
    values = {}
    values.update(foot_sliding(step, CONTACT_THRESHOLD_M))
    values.update(jitter(step, FREEZE_EPS_DEG))
    values.update(grounding(step, AIR_THRESHOLD_M))
    verdicts = judge(values, THRESHOLDS)
    return {"metrics": values,
            "verdicts": [{"key": v.key, "value": v.value, "limit": v.limit,
                          "direction": v.direction, "passed": v.passed} for v in verdicts],
            "pass": all(v.passed for v in verdicts)}


def quality_cell(switcher: Switcher, bundle, style_a: str, style_b: str, seed: int,
                 seam_at: int) -> dict:
    """Where does the switch's quality cost come from?

    Three legs, all style B, all 44 frames, all the same seed -- only the context
    they are planned from differs:

    ``fresh``        a fresh reseed of B (what the C-line gate was calibrated on);
    ``replan_same``  B planned from B's own mid-clip window (the round-8 continuity
                     path: no style change, just a replan while walking);
    ``switch``       B planned from A's mid-clip window (this round's switch).

    ``replan_same`` vs ``fresh`` is the cost of conditioning on *any* mid-clip
    window; ``switch`` vs ``replan_same`` is the cost of the style change itself.
    """
    _, plain_b = switcher.first_leg(style_b, seed)
    motion_a, motion_b, _ = switcher.seamless(style_a, style_b, seam_at, seed)
    agent_same, motion_same = switcher.first_leg(style_b, seed)
    agent_same.advance(int(seam_at))
    motion_same, _ = agent_same.plan(switcher.command(style_b, seed), switcher.styles[style_b])
    _, plain_a = switcher.first_leg(style_a, seed)

    def step(motion, label):
        return step_of(motion.roots, motion.rotations_xyzw, motion.target_roots,
                       motion.target_local_rotations_xyzw, bundle.joint_parents,
                       bundle.neutral_joints, bundle.joint_names, label=label,
                       style=style_b, seed=seed, advance=0)

    fresh = leg_quality(step(plain_b, "fresh_b"))
    replan_same = leg_quality(step(motion_same, "replan_same_b"))
    switched = leg_quality(step(motion_b, "switch_b"))
    leg_a = leg_quality(step(plain_a, "leg_a"))
    joined = switcher.sequence([style_a, style_b], [seam_at], kind=LEG_KINDS[0], seed=seed)
    gate = quality_gate(joined, bundle)
    seam = gate["seams"][0]
    naive_legs = []
    previous = None
    for index, style_name in enumerate((style_a, style_b)):
        if index == 0:
            _, motion = switcher.first_leg(style_name, seed)
        else:
            _, raw = switcher.first_leg(style_name, seed)
            motion = place_at(raw, float(previous.roots[seam_at, 0]),
                              float(previous.roots[seam_at, 2]),
                              heading_at(previous, seam_at))
        shown = seam_at if index == 0 else motion.frames
        naive_legs.append(Leg(style=style_name, kind=LEG_KINDS[1], motion=motion,
                              advance=seam_at if index else 0, frames_shown=shown))
        previous = motion
    naive_plan = join_legs(naive_legs, switcher.parents, switcher.rest, switcher.joint_names)
    naive_gate = quality_gate(naive_plan, bundle)
    return {
        "a": style_a, "b": style_b, "seed": int(seed), "seam_at": int(seam_at),
        "fresh": fresh, "replan_same": replan_same, "switch": switched, "leg_a": leg_a,
        "seam_window": seam,
        "naive_seam_window": naive_gate["seams"][0],
        "naive_whole_clip": naive_gate["whole_clip"],
        "replan_cost": {key: _safe_ratio(replan_same["metrics"][key], fresh["metrics"][key])
                        for key in ("foot_slide_max_m", "jerk_p95_deg", "jerk_max_deg")},
        "switch_cost": {key: _safe_ratio(switched["metrics"][key], replan_same["metrics"][key])
                        for key in ("foot_slide_max_m", "jerk_p95_deg", "jerk_max_deg")},
    }


def _safe_ratio(left: float, right: float) -> float:
    return float(left / right) if right else float("inf")


def summarize_quality(cells: list[dict]) -> dict:
    """Aggregate the quality grid: is the *switch* what costs quality?"""
    def ratios(key, source):
        return [cell[source][key] for cell in cells]

    seam_jerk = [cell["seam_window"]["jerk_over_interior_p95"] for cell in cells]
    naive_jerk = [cell["naive_seam_window"]["jerk_over_interior_p95"] for cell in cells]
    return {
        "cells": len(cells),
        "pass_counts": {
            "fresh_b": sum(cell["fresh"]["pass"] for cell in cells),
            "replan_same_b": sum(cell["replan_same"]["pass"] for cell in cells),
            "switch_b": sum(cell["switch"]["pass"] for cell in cells),
            "leg_a": sum(cell["leg_a"]["pass"] for cell in cells),
        },
        "replan_cost": {key: {"median": float(np.median(ratios(key, "replan_cost"))),
                              "max": float(np.max(ratios(key, "replan_cost")))}
                        for key in ("foot_slide_max_m", "jerk_p95_deg", "jerk_max_deg")},
        "switch_cost": {key: {"median": float(np.median(ratios(key, "switch_cost"))),
                              "max": float(np.max(ratios(key, "switch_cost")))}
                        for key in ("foot_slide_max_m", "jerk_p95_deg", "jerk_max_deg")},
        "seam_jerk_over_interior": {
            "median": float(np.median(seam_jerk)), "max": float(np.max(seam_jerk)),
            "cells_over_1": int(sum(value > 1.0 for value in seam_jerk))},
        "naive_seam_jerk_over_interior": {
            "median": float(np.median(naive_jerk)), "max": float(np.max(naive_jerk)),
            "cells_over_1": int(sum(value > 1.0 for value in naive_jerk))},
        "seam_foot_slide_over_interior": int(
            sum(not cell["seam_window"]["foot_slide_within_interior_p95"] for cell in cells)),
        "seam_penetration_worse": int(
            sum(not cell["seam_window"]["penetration_within_interior"] for cell in cells)),
        "switch_failures": [
            {"a": cell["a"], "b": cell["b"], "seed": cell["seed"], "seam_at": cell["seam_at"],
             "failed": [v["key"] for v in cell["switch"]["verdicts"] if not v["passed"]]}
            for cell in cells if not cell["switch"]["pass"]],
        "fresh_failures": [
            {"a": cell["a"], "b": cell["b"], "seed": cell["seed"], "seam_at": cell["seam_at"],
             "failed": [v["key"] for v in cell["fresh"]["verdicts"] if not v["passed"]]}
            for cell in cells if not cell["fresh"]["pass"]],
    }


# ---------------------------------------------------------------------------
# the control matrix
# ---------------------------------------------------------------------------
def control_row(switcher: Switcher, style_a: str, style_b: str, seed: int, seam_at: int) -> dict:
    """One (A, B, seed, seam) cell measured on both arms."""
    motion_a, motion_b, _ = switcher.seamless(style_a, style_b, seam_at, seed)
    naive_a, naive_aligned = switcher.naive(style_a, style_b, seam_at, seed, align=True)
    _, naive_raw = switcher.naive(style_a, style_b, seam_at, seed, align=False)
    row = {"a": style_a, "b": style_b, "seed": int(seed), "seam_at": int(seam_at)}
    for name, left, right in (("seamless", motion_a, motion_b),
                              ("naive_aligned", naive_a, naive_aligned),
                              ("naive_raw", naive_a, naive_raw)):
        row[name] = seam_metrics(left, seam_at, right, switcher.parents, switcher.rest,
                                 label=f"{style_a}->{style_b}")
    row["same_style"] = style_a == style_b
    return row


def summarize(rows: list[dict]) -> dict:
    """Per-arm summary over a set of cells (medians and maxima of the ratios)."""
    keys = ("root_position", "joint_position", "root_velocity", "root_acceleration",
            "joint_velocity", "joint_acceleration")
    summary = {}
    for arm in ("seamless", "naive_aligned", "naive_raw"):
        entry = {}
        for key in keys:
            values = [row[arm]["ratio"][key] for row in rows]
            entry[key] = {"median": float(np.median(values)), "max": float(np.max(values)),
                          "cells_over_p95": int(sum(value > 1.0 for value in values)),
                          "cells": len(values)}
        entry["root_position_m_max"] = float(max(row[arm]["root_position_m"] for row in rows))
        entry["joint_position_m_max"] = float(max(row[arm]["joint_position_m"] for row in rows))
        entry["joints_over_p95_velocity_median"] = float(
            np.median([row[arm]["joints_over_p95"]["velocity"] for row in rows]))
        entry["joints_over_p95_velocity_max"] = int(
            max(row[arm]["joints_over_p95"]["velocity"] for row in rows))
        entry["position_within_one_frame"] = int(
            sum(row[arm]["position_within_one_frame"] for row in rows))
        summary[arm] = entry
    summary["criteria"] = {"seam": SEAM_CRITERION, "p95": 95.0,
                           "n_joints": None, "cells": len(rows)}
    return summary


def print_control(rows: list[dict]) -> None:
    header = (f"{'A -> B':26s} {'sd':>2s} {'seam':>4s} | {'arm':13s} {'root dp mm':>10s} "
              f"{'joint dp mm':>11s} {'dv/p95':>7s} {'da/p95':>7s} {'jw/p95':>7s} "
              f"{'ja/p95':>7s} {'j>p95':>5s} {'ok':>3s}")
    print(header)
    print("-" * len(header))
    for row in rows:
        tag = f"{row['a']} -> {row['b']}"
        for arm in ("seamless", "naive_aligned", "naive_raw"):
            metrics = row[arm]
            print(f"{tag:26s} {row['seed']:2d} {row['seam_at']:4d} | {arm:13s} "
                  f"{metrics['root_position_m'] * 1000:10.3f} "
                  f"{metrics['joint_position_m'] * 1000:11.3f} "
                  f"{metrics['ratio']['root_velocity']:7.2f} "
                  f"{metrics['ratio']['root_acceleration']:7.2f} "
                  f"{metrics['ratio']['joint_velocity']:7.2f} "
                  f"{metrics['ratio']['joint_acceleration']:7.2f} "
                  f"{metrics['joints_over_p95']['velocity']:5d} "
                  f"{'yes' if metrics['all_within_p95'] else 'no':>3s}")


# ---------------------------------------------------------------------------
# plan documents
# ---------------------------------------------------------------------------
def document_of(plan: SwitchPlan, args, parameters: int, seed: int, bundle) -> dict:
    step = Step(label="switch", style="/".join(plan.styles), move=args.move,
                facing=args.facing, seed=int(seed), advance=0, roots=plan.roots,
                rotations_xyzw=plan.rotations_xyzw, target_roots=plan.target_roots,
                target_rotations_xyzw=plan.target_local_rotations_xyzw)
    document = build_document(
        library="motionbricks_mlx", model_bundle=str(args.bundle),
        device=f"mlx-{args.device}", sampling="gumbel", learned_parameters=parameters,
        steps=[step], parents=bundle.joint_parents, rest=bundle.neutral_joints,
        joint_names=bundle.joint_names)
    document["switch"] = plan.document()
    return document


def load_styles(names, path: Path):
    return {name: load_style(path / f"{name}.mbstyle") for name in names}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--styles", type=Path, default=DEFAULT_STYLES)
    parser.add_argument("--style", default="walk", help="motion A")
    parser.add_argument("--to", default=None, help="motion B (required unless --control only)")
    parser.add_argument("--back", default=None,
                        help="switch back to this style after B, making A->B->A one clip")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--move", default="0,1")
    parser.add_argument("--facing", default="0,1")
    parser.add_argument("--seam-at", type=int, default=20, help="frame of A the switch happens at")
    parser.add_argument("--seam-at-back", type=int, default=20,
                        help="frame of B the switch back happens at")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--control", action="store_true",
                        help="run the naive-vs-seamless matrix over --pairs/--seeds/--seam-frames")
    parser.add_argument("--quality", action="store_true",
                        help="run the C-line metrics gate on the joined clip and its seam windows")
    parser.add_argument("--quality-sweep", action="store_true",
                        help="run the C-line quality gate over --pairs/--seeds/--seam-frames and "
                             "compare switch-conditioned legs with independent ones")
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS),
                        help="'A:B,A:B' cells for --control")
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--seam-frames", default=",".join(str(v) for v in DEFAULT_SEAMS))
    parser.add_argument("--report", type=Path, default=None,
                        help="write the control matrix / quality gate as JSON here")
    parser.add_argument("--out", type=Path, required=True,
                        help="plan document to write (required, like mlx_plan.py)")
    args = parser.parse_args()

    if not (args.bundle / "manifest.json").is_file():
        print(f"error: {args.bundle} has no manifest.json", file=sys.stderr)
        return 2
    move = tuple(float(v) for v in args.move.split(","))
    facing = tuple(float(v) for v in args.facing.split(","))
    args.move, args.facing = move, facing
    if len(move) != 2 or len(facing) != 2:
        parser.error("--move and --facing take 'x,z'")
    if args.to is None:
        parser.error("--to is required (this script only exists for a switch; "
                     "use scripts/mlx_plan.py for a plain plan)")

    pairs = []
    wanted = {args.style, args.to, args.back}
    for cell in args.pairs.split(","):
        if not cell.strip():
            continue
        left, _, right = cell.partition(":")
        if not right:
            parser.error(f"expected 'A:B', got {cell!r}")
        pairs.append((left.strip(), right.strip()))
        wanted.update({left.strip(), right.strip()})
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    seam_frames = [int(v) for v in args.seam_frames.split(",") if v.strip()]

    missing = sorted(name for name in wanted if name
                     and not (args.styles / f"{name}.mbstyle").is_file())
    if missing:
        parser.error(f"unknown style(s): {', '.join(missing)}")

    bundle = Bundle(args.bundle, device=args.device)
    pipeline = MlxPipeline(bundle)
    styles = load_styles(sorted(name for name in wanted if name), args.styles)
    switcher = Switcher(pipeline, bundle, styles, seed=args.seed, move=move, facing=facing)

    styles_sequence = [args.style, args.to] + ([args.back] if args.back else [])
    seams = [args.seam_at] + ([args.seam_at_back] if args.back else [])

    started = time.perf_counter()
    plan = switcher.sequence(styles_sequence, seams, kind=LEG_KINDS[0], seed=args.seed)
    plan_again = switcher.sequence(styles_sequence, seams, kind=LEG_KINDS[0], seed=args.seed)
    for name, left, right in (("roots", plan.roots, plan_again.roots),
                              ("rotations", plan.rotations_xyzw, plan_again.rotations_xyzw)):
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            raise SystemExit(f"NOT DETERMINISTIC: {name} differs between two identical runs")
    elapsed = (time.perf_counter() - started) * 1000.0

    print(f"sequence {' -> '.join(styles_sequence)} (seam frames {seams}) seed={args.seed} "
          f"device={args.device}")
    for leg in plan.legs_payload():
        print(f"  leg {leg['style']:18s} joined frames {leg['start']:3d}..{leg['end']:<3d} "
              f"(of {leg['generated_frames']:3d} generated, conditioned on frame "
              f"{leg['advance']})")
    for entry in plan.seam_payload():
        metrics = entry["metrics"]
        print(f"  seam @frame {entry['joined_frame']:3d}  {entry['from_style']}->{entry['to_style']}: "
              f"root dp {metrics['root_position_m'] * 1000:.3f} mm   "
              f"joint dp {metrics['joint_position_m'] * 1000:.3f} mm   "
              f"dv {metrics['root_velocity_mps']:.3f}/{metrics['p95']['root_velocity_mps']:.3f} m/s "
              f"({metrics['ratio']['root_velocity']:.2f}x)   "
              f"da {metrics['root_acceleration_mps2']:.3f}/{metrics['p95']['root_acceleration_mps2']:.3f} "
              f"m/s2 ({metrics['ratio']['root_acceleration']:.2f}x)")
        print(f"                 joint dw {metrics['joint_velocity_deg_s']:.1f} deg/s "
              f"({metrics['ratio']['joint_velocity']:.2f}x of the worst joint's p95, "
              f"{metrics['joints_over_p95']['velocity']}/34 joints over)   "
              f"joint da {metrics['joint_acceleration_deg_s2']:.0f} deg/s2 "
              f"({metrics['ratio']['joint_acceleration']:.2f}x, "
              f"{metrics['joints_over_p95']['acceleration']}/34 over)")
    print(f"  joined {plan.frames} frames, two identical runs in {elapsed:.1f} ms "
          f"({elapsed / 2:.1f} ms/run)")
    verdict = "ALL SEAMS WITHIN p95" if plan.document()["all_seams_within_p95"] \
        else "SOME SEAM ABOVE SEGMENT p95 (see the table)"
    print(f"  verdict: {verdict}")

    parameters = sum(bundle.manifest["files"][name]["parameter_count"]
                     for name in ("pose", "root", "vq-decoder"))
    document = document_of(plan, args, parameters, args.seed, bundle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_document(args.out, document)
    print(f"wrote {args.out}")

    report: dict = {"sequence": styles_sequence, "seam_frames": seams, "seed": args.seed,
                    "device": args.device, "plan_ms_per_run": elapsed / 2.0,
                    "switch": plan.document()}

    if args.quality:
        for arm, arm_plan in _arms(switcher, styles_sequence, seams, args.seed):
            report.setdefault("quality", {})[arm] = quality_gate(arm_plan, bundle)
        print("\nquality gate (C-line metrics, imported read-only):")
        for arm, gate in report["quality"].items():
            print(f"  {arm:14s} per leg "
                  f"{'PASS' if gate['legs_pass'] else 'FAIL'}   "
                  f"joined {gate['whole_clip_frames']} frames whole-clip "
                  f"{'PASS' if gate['whole_clip_pass'] else 'FAIL'} "
                  f"(limits are calibrated on one ~44-frame clip; foot slide accumulates)")
            for leg in gate["legs"]:
                failed = [v["key"] for v in leg["verdicts"] if not v["passed"]]
                print(f"      leg {leg['index']} {leg['style']:18s} {leg['frames']:3d} frames  "
                      f"{'PASS' if leg['pass'] else 'FAIL ' + ','.join(failed)}   "
                      f"slide {leg['metrics']['foot_slide_max_m']:.3f} m   "
                      f"jerk p95 {leg['metrics']['jerk_p95_deg']:.2f} / max "
                      f"{leg['metrics']['jerk_max_deg']:.1f} deg   "
                      f"penetration {leg['metrics']['foot_height_min_m']:.4f} m")
            for seam in gate["seams"]:
                print(f"      seam @frame {seam['joined_frame']:3d}: "
                      f"slide/frame {seam['window']['foot_slide_per_frame_mm']:.2f} mm "
                      f"vs interior p95 {seam['interior_foot_slide_p95_mm']:.2f} mm "
                      f"({'ok' if seam['foot_slide_within_interior_p95'] else 'WORSE'})   "
                      f"jerk p95 {seam['window']['jerk_p95_deg']:.2f} vs "
                      f"{seam['interior_jerk_p95_p95_deg']:.2f} deg "
                      f"({seam['jerk_over_interior_p95']:.2f}x, "
                      f"{'ok' if seam['jerk_within_interior_p95'] else 'WORSE'})   "
                      f"penetration {seam['window']['foot_height_min_m']:.4f} vs "
                      f"{seam['interior_foot_height_min_m']:.4f} m "
                      f"({'ok' if seam['penetration_within_interior'] else 'WORSE'})")

    if args.control:
        rows = []
        started = time.perf_counter()
        for style_a, style_b in pairs:
            for seed in seeds:
                for seam_at in seam_frames:
                    rows.append(control_row(switcher, style_a, style_b, seed, seam_at))
        control_ms = (time.perf_counter() - started) * 1000.0
        report["control"] = {"rows": rows, "summary": summarize(rows),
                             "pairs": pairs, "seeds": seeds, "seam_frames": seam_frames,
                             "cells": len(rows), "wall_ms": control_ms}
        print(f"\ncontrol matrix: {len(rows)} cells in {control_ms:.0f} ms\n")
        print_control(rows)
        summary = report["control"]["summary"]
        print("\nsummary (ratio = seam jump / in-segment p95 of the same jump):")
        for arm in ("seamless", "naive_aligned", "naive_raw"):
            entry = summary[arm]
            print(f"  {arm:14s} root dp <= {entry['root_position_m_max'] * 1000:8.3f} mm   "
                  f"joint dp <= {entry['joint_position_m_max'] * 1000:8.3f} mm   "
                  f"dv median {entry['root_velocity']['median']:5.2f} max {entry['root_velocity']['max']:5.2f} "
                  f"({entry['root_velocity']['cells_over_p95']}/{entry['root_velocity']['cells']} over)   "
                  f"jw median {entry['joint_velocity']['median']:5.2f} max {entry['joint_velocity']['max']:5.2f} "
                  f"({entry['joint_velocity']['cells_over_p95']}/{entry['joint_velocity']['cells']} over)")
        print(f"  criterion: {SEAM_CRITERION}")

    if args.quality_sweep:
        cells = []
        started = time.perf_counter()
        for style_a, style_b in pairs:
            for seed in seeds:
                for seam_at in seam_frames:
                    cells.append(quality_cell(switcher, bundle, style_a, style_b, seed, seam_at))
        sweep_ms = (time.perf_counter() - started) * 1000.0
        quality_summary = summarize_quality(cells)
        report["quality_sweep"] = {"cells": cells, "summary": quality_summary,
                                   "wall_ms": sweep_ms}
        print(f"\nquality sweep: {len(cells)} cells in {sweep_ms:.0f} ms")
        counts = quality_summary["pass_counts"]
        print(f"  legs against the absolute C-line gate: fresh B "
              f"{counts['fresh_b']}/{len(cells)}, replan B (same style, mid-clip) "
              f"{counts['replan_same_b']}/{len(cells)}, switch B (from A's window) "
              f"{counts['switch_b']}/{len(cells)}")
        for label, key in (("replan cost  (replan_same / fresh)", "replan_cost"),
                           ("switch cost  (switch / replan_same)", "switch_cost")):
            entry = quality_summary[key]
            print(f"  {label}: foot slide median {entry['foot_slide_max_m']['median']:.3f} "
                  f"max {entry['foot_slide_max_m']['max']:.3f}   jerk p95 median "
                  f"{entry['jerk_p95_deg']['median']:.3f} max {entry['jerk_p95_deg']['max']:.3f}   "
                  f"jerk max median {entry['jerk_max_deg']['median']:.3f} "
                  f"max {entry['jerk_max_deg']['max']:.3f}")
        seam = quality_summary["seam_jerk_over_interior"]
        naive = quality_summary["naive_seam_jerk_over_interior"]
        print(f"  seam window vs interior, jerk p95: seamless median {seam['median']:.3f} "
              f"max {seam['max']:.3f} ({seam['cells_over_1']}/{len(cells)} above 1)   "
              f"naive median {naive['median']:.3f} max {naive['max']:.3f} "
              f"({naive['cells_over_1']}/{len(cells)} above 1)")
        print(f"  foot slide over interior p95 in "
              f"{quality_summary['seam_foot_slide_over_interior']}/{len(cells)}   "
              f"penetration worse in {quality_summary['seam_penetration_worse']}/{len(cells)}")
        for entry in quality_summary["fresh_failures"]:
            print(f"    FRESH leg FAILED {entry['b']} seed {entry['seed']}: "
                  f"{','.join(entry['failed'])}")
        for entry in quality_summary["switch_failures"]:
            print(f"    switch leg FAILED {entry['a']}->{entry['b']} seed {entry['seed']} "
                  f"seam {entry['seam_at']}: {','.join(entry['failed'])}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    print(json.dumps({"all_seams_within_p95": plan.document()["all_seams_within_p95"],
                      "worst_seam_ratio": plan.document()["worst_seam_ratio"]}))
    return 0


def _arms(switcher: Switcher, styles_sequence: list[str], seams: list[int], seed: int):
    """The joined clip of each arm, for the quality gate.

    The naive arm is the *charitable* hard cut at every handoff: each leg is
    generated independently from its own style reseed and then slid onto the
    previous leg's handoff pose, so the control is never a teleport.
    """
    yield "seamless", switcher.sequence(styles_sequence, seams, kind=LEG_KINDS[0], seed=seed)
    legs: list[Leg] = []
    previous = None
    for index, style_name in enumerate(styles_sequence):
        if index == 0:
            _, motion = switcher.first_leg(style_name, seed)
        else:
            _, raw = switcher.first_leg(style_name, seed)
            motion = place_at(raw, float(previous.roots[seams[index - 1], 0]),
                              float(previous.roots[seams[index - 1], 2]),
                              heading_at(previous, seams[index - 1]))
        shown = seams[index] if index < len(styles_sequence) - 1 else motion.frames
        legs.append(Leg(style=style_name, kind=LEG_KINDS[1], motion=motion,
                        advance=seams[index - 1] if index else 0, frames_shown=shown))
        previous = motion
    yield "naive_aligned", join_legs(legs, switcher.parents, switcher.rest,
                                     switcher.joint_names)


if __name__ == "__main__":
    raise SystemExit(main())
