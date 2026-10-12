#!/usr/bin/env python3
"""Bake a **library of real actions** and export it as viewable products.

Rounds 3-6 measured the bake: how fast, how much margin, how it decomposes.  They
never produced a *product*.  This script is the product side of the same code
path:

* ``batch``   runs the approved release configuration
  (``--bake-stages all+decode``: pose + root + decoder + motion decode through
  the round-5 single-pass orchestrator) over a campaign of real
  ``(style, movement, seed)`` scenarios, finishes every scenario as a full
  :mod:`motionbricks_mlx.export.plan_json` plan document, and writes the library
  manifest decision ``0007`` item 4 requires -- the twelve
  ``BAKE_MANIFEST_FIELDS``, taken from the campaign that was actually run.
* ``single``  re-runs the identical scenarios **one at a time in their own
  process** (``B = 1`` semantics, no bake switch set), which is the independent
  arm the equivalence check needs.
* ``compare`` pairs a batch arm against a single arm scenario by scenario and
  reports the identity rate and the worst deviation of the final action.
* ``check``   re-reads the written manifest and verifies the twelve fields.
* ``metrics`` reads the written documents back through the C-line metric code
  (``motionbricks_mlx.eval``, imported read-only) and reports the frozen
  thresholds' verdicts for the actual baked products.
* ``pairdoc`` builds a two-step document (batch step + single step of the same
  scenario) so one invocation of the reference renderer paints both arms.

Nothing here changes the accepted path: the bake switches are process-local
environment, and every arm prints its resolved switch values.

    ./.venv/bin/python scripts/mlx_bake_library.py batch --batch-size 8  --out artifacts/bake-.../
    ./.venv/bin/python scripts/mlx_bake_library.py single --batch-size 16 --out artifacts/bake-.../
    ./.venv/bin/python scripts/mlx_bake_library.py compare --batch ... --single ... --out ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from argparse import RawDescriptionHelpFormatter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motionbricks_mlx.data.agent import Agent, Command  # noqa: E402
from motionbricks_mlx.data.style import load_style  # noqa: E402
from motionbricks_mlx.export.plan_json import (  # noqa: E402
    Step,
    build_document,
    write_document,
)
from motionbricks_mlx.models.bundle import Bundle  # noqa: E402
from motionbricks_mlx.models.transition import MlxPipeline  # noqa: E402

DEFAULT_BUNDLE = REPO_ROOT / "checkpoints/g1-f32-mlx"
DEFAULT_STYLES = Path.home() / "dev/motion-bricks-cpp/repo/generated/styles"

#: The approved release bake configuration (decision 0009 item 2).  The flag
#: string is the one ``scripts/mlx_bench.py --bake-stages`` accepts; the three
#: switch names are the ones ``bake.batched_stages()`` / ``single_pass_stages()``
#: read.
RELEASE_STAGE_FLAG = "all+decode"
RELEASE_SWITCHES = ("bake_root_batch=1", "bake_decoder_batch=1", "bake_decode_batch=1")
BAKE_SWITCH_NAMES = ("bake_root_batch", "bake_decoder_batch", "bake_decode_batch")

FACING_XYZ = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Row:
    """One campaign row: the scenario, and the command that defines it."""

    label: str
    style: str
    movement: tuple[float, float, float]
    seed: int

    @property
    def filename(self) -> str:
        return f"{self.label}-{self.style}"


#: The **diverse** campaign: 8 distinct real styles and 14 distinct sampling
#: seeds.  Every row was chosen so that the **duration probe selects the same
#: token** -- the 24-frame outcome (duration token 6) for these style/seed pairs
#: was read off the pipeline itself before the campaign was fixed, which is what
#: makes the batch duration-homogeneous rather than merely declared to be.  The
#: batched arm re-checks it: ``token_groups`` in the manifest is measured, and
#: ``--strict`` fails the run when the declared and observed grouping disagree.
#: ``s00..s07`` is the ``B = 8`` sub-campaign (4 styles, 8 seeds); ``s00..s15``
#: the ``B = 16`` one (8 styles, 14 seeds).
CAMPAIGN_DIVERSE: tuple[Row, ...] = (
    Row("s00", "idle", (0.2, 0.0, 0.0), 5),
    Row("s01", "idle", (0.2, 0.0, 0.0), 9),
    Row("s02", "walk_zombie", (0.2, 0.0, 0.0), 6),
    Row("s03", "walk_zombie", (0.2, 0.0, 0.0), 8),
    Row("s04", "stealth_walk", (0.0, 0.0, 0.0), 1),
    Row("s05", "stealth_walk", (0.0, 0.0, 0.0), 3),
    Row("s06", "walk_stealth", (0.2, 0.0, 0.0), 7),
    Row("s07", "walk_stealth", (0.2, 0.0, 0.0), 13),
    Row("s08", "walk_happy_dance", (0.2, 0.0, 0.0), 10),
    Row("s09", "walk_happy_dance", (0.2, 0.0, 0.0), 14),
    Row("s10", "injured_walk", (0.2, 0.0, 0.0), 2),
    Row("s11", "injured_walk", (0.2, 0.0, 0.0), 6),
    Row("s12", "walk_gun", (0.2, 0.0, 0.0), 4),
    Row("s13", "walk_gun", (0.2, 0.0, 0.0), 12),
    Row("s14", "idle", (0.2, 0.0, 0.0), 17),
    Row("s15", "walk_boxing", (0.2, 0.0, 0.0), 2),
)

#: The **single-style** control: one style, sixteen seeds, so the batch is again
#: duration-homogeneous but carries no style diversity.  Rounds 4-6 measured
#: their safety ratios on a batch of this shape, so this is the arm that says
#: whether the diverse campaign's tier difference is the *styles* or the seeds.
#: ``walk`` selects the 44-frame duration (token 11) for every seed, which the
#: measured ``token_groups`` confirms at run time.
CAMPAIGN_SINGLE_STYLE: tuple[Row, ...] = tuple(
    Row(f"w{index:02d}", "walk", (0.2, 0.0, 0.0), 1 + index) for index in range(16))

#: The **idle-style** arm: the diverse campaign's worst perturbation contributor
#: (``idle``) on its own, one style and sixteen seeds.  It separates "this style's
#: batched forward is numerically the sensitive one" from "co-batching unlike
#: styles in one graph is what costs perturbation": the two readings predict
#: different numbers here, so the measurement decides.
CAMPAIGN_IDLE_STYLE: tuple[Row, ...] = tuple(
    Row(f"i{index:02d}", "idle", (0.2, 0.0, 0.0), 1 + index) for index in range(16))

CAMPAIGNS: dict[str, tuple[Row, ...]] = {
    "diverse": CAMPAIGN_DIVERSE,
    "single-style": CAMPAIGN_SINGLE_STYLE,
    "idle-style": CAMPAIGN_IDLE_STYLE,
}


def print_load(tag: str) -> list[float]:
    load = [round(value, 2) for value in os.getloadavg()]
    print(f"load[{tag}] {load}")
    return load


def enable_release_stages() -> None:
    """Turn on the release bake switches for this process (default: off)."""
    extra = ",".join(RELEASE_SWITCHES)
    current = os.environ.get("MOTIONBRICKS_MLX_OPT")
    os.environ["MOTIONBRICKS_MLX_OPT"] = f"{current},{extra}" if current else extra


def resolved_switches() -> dict[str, bool]:
    """Every bake switch's resolved value here (proof the arm is what it says)."""
    from motionbricks_mlx.models import nn

    return {name: bool(nn._opt(name)) for name in BAKE_SWITCH_NAMES}


class Library:
    """The bundle, the loaded styles and the skeleton tables the writer needs."""

    def __init__(self, bundle_dir: Path, styles_dir: Path, device: str,
                 campaign: tuple[Row, ...]) -> None:
        self.bundle_dir = bundle_dir
        self.device = device
        self.campaign = campaign
        self.bundle = Bundle(bundle_dir, device=device)
        self.pipeline = MlxPipeline(self.bundle)
        self.styles = {row.style: load_style(styles_dir / f"{row.style}.mbstyle")
                       for row in campaign}
        self.parameters = sum(self.bundle.manifest["files"][name]["parameter_count"]
                              for name in ("pose", "root", "vq-decoder"))

    def release(self) -> None:
        self.bundle.release()

    def scenarios(self, rows: tuple[Row, ...]):
        from motionbricks_mlx.models import bake as bake_module

        return [bake_module.Scenario(label=row.label, style=self.styles[row.style],
                                     seed=row.seed, movement=row.movement)
                for row in rows]

    def document(self, row: Row, motion) -> dict:
        step = Step(label=row.label, style=row.style,
                    move=(row.movement[0], row.movement[2]),
                    facing=(FACING_XYZ[0], FACING_XYZ[2]), seed=row.seed, advance=0,
                    roots=motion.roots, rotations_xyzw=motion.rotations_xyzw,
                    target_roots=motion.target_roots,
                    target_rotations_xyzw=motion.target_local_rotations_xyzw)
        return build_document(library="motionbricks_mlx", model_bundle=str(self.bundle_dir),
                              device=f"mlx-{self.device}", sampling="gumbel",
                              learned_parameters=self.parameters, steps=[step],
                              parents=self.bundle.joint_parents,
                              rest=self.bundle.neutral_joints,
                              joint_names=self.bundle.joint_names)

    def write(self, plans_dir: Path, row: Row, motion) -> dict:
        plans_dir.mkdir(parents=True, exist_ok=True)
        path = plans_dir / f"{plans_dir.name}-{row.filename}.json"
        write_document(path, self.document(row, motion))
        return {"file": str(path), "label": row.label, "style": row.style,
                "frames": int(motion.frames), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def rows_for(campaign: tuple[Row, ...], batch_size: int | None) -> tuple[Row, ...]:
    if batch_size is None:
        return campaign
    if not 1 <= batch_size <= len(campaign):
        raise SystemExit(f"error: --batch-size must be 1..{len(campaign)}")
    return campaign[:batch_size]


def safety_detail(bake_module, reference, runs) -> dict:
    """Per-scenario perturbation and margin -- **attribution only**.

    The tier the manifest records comes from the frozen global rule
    (``max`` perturbation over ``min`` margin, decision 0008).  This block exists
    so a reader can see *which* scenario supplies each of those two extremes;
    it is not a second criterion and never feeds ``bake_tier``.
    """
    rows = []
    for record, run in zip(reference, runs):
        perturbations = bake_module.stage_perturbations([record], [run])
        margins = bake_module.decision_margins([record])
        safety = bake_module.safety_factor(perturbations, margins)
        rows.append({
            "label": record.scenario.label, "style": record.scenario.style.name,
            "seed": int(record.scenario.seed),
            "pose_perturbation": safety["pose_perturbation"],
            "pose_margin": safety["pose_margin"],
            "pose_ratio": safety["pose_ratio"],
            "duration_perturbation": safety["duration_perturbation"],
            "duration_margin": safety["duration_margin"],
            "duration_ratio": safety["duration_ratio"],
        })
    worst = max(rows, key=lambda row: row["pose_perturbation"])
    tightest = min(rows, key=lambda row: row["pose_margin"])
    return {
        "note": "attribution only; the recorded tier is the global max/min rule of 0008",
        "scenarios": rows,
        "max_pose_perturbation": {"label": worst["label"], "style": worst["style"],
                                  "value": worst["pose_perturbation"]},
        "min_pose_margin": {"label": tightest["label"], "style": tightest["style"],
                            "value": tightest["pose_margin"]},
        "same_scenario": worst["label"] == tightest["label"],
    }


def array_digest(array) -> str:
    """SHA-256 of the float32 bytes of *array* -- a bit-exactness witness."""
    return hashlib.sha256(np.asarray(array, np.float32).tobytes()).hexdigest()


def run_record(label: str, row: Row, style_name: str, selected: int, tokens, uniforms,
               roots, rotations) -> dict:
    roots = np.asarray(roots)
    rotations = np.asarray(rotations)
    return {
        "label": label,
        "style": style_name,
        "seed": int(row.seed),
        "movement": [float(v) for v in row.movement],
        "facing": [float(v) for v in FACING_XYZ],
        "frames": int(roots.shape[0]),
        "selected_token": int(selected),
        "tokens": [int(v) for v in np.asarray(tokens).reshape(-1)],
        "uniforms": [float(v) for v in np.asarray(uniforms).reshape(-1)],
        "roots_sha256": array_digest(roots),
        "rotations_sha256": array_digest(rotations),
    }


# ---------------------------------------------------------------------------
# batch: the release-configuration bake
# ---------------------------------------------------------------------------


def time_arms(library: Library, scenarios, stages, repeats: int) -> dict:
    """Serial (B = 1) against the single-pass release bake, interleaved.

    Warm-up is untimed so neither arm pays a shape variant's compile cost inside
    a timed region, and the arm order rotates per repeat so neither is
    systematically first.  This is the *within-process* ratio; the report's
    cross-``B`` numbers come from separate processes (decision 0009 item 4).
    """
    from motionbricks_mlx.models import bake as bake_module

    pipeline, bundle = library.pipeline, library.bundle
    bake_module.record_serial(pipeline, bundle, scenarios, capture_trace=False)
    bake_module.drive_batch_single_pass(pipeline, bundle, scenarios, stages,
                                        capture_trace=False)
    timings: dict[str, list[float]] = {"serial": [], "single_pass": []}
    for repeat in range(repeats):
        order = ("serial", "single_pass") if repeat % 2 == 0 else \
            ("single_pass", "serial")
        for arm in order:
            started = time.perf_counter()
            if arm == "serial":
                bake_module.record_serial(pipeline, bundle, scenarios,
                                          capture_trace=False)
            else:
                bake_module.drive_batch_single_pass(pipeline, bundle, scenarios,
                                                    stages, capture_trace=False)
            timings[arm].append((time.perf_counter() - started) * 1000.0)
    serial = statistics.median(timings["serial"])
    single = statistics.median(timings["single_pass"])
    size = len(scenarios)
    return {
        "repeats": repeats,
        "serial_ms_p50": serial,
        "single_pass_ms_p50": single,
        "speedup": serial / single if single else None,
        "single_pass_ms_per_scenario": single / size,
        "throughput_per_s": size / (single / 1000.0) if single else None,
        "serial_times_ms": timings["serial"],
        "single_pass_times_ms": timings["single_pass"],
    }


def cmd_batch(args) -> int:
    from motionbricks_mlx.models import bake as bake_module

    enable_release_stages()
    switches = resolved_switches()
    print(f"resolved bake switches: {switches}")
    if not all(switches.values()):
        print("error: the release configuration needs all three bake switches on",
              file=sys.stderr)
        return 2
    load_before = print_load("before")
    library = Library(args.bundle, args.styles, args.device, args.campaign_rows)
    rows = rows_for(args.campaign_rows, args.batch_size)
    scenarios = library.scenarios(rows)
    stages = bake_module.batched_stages()
    owned = bake_module.single_pass_stages(stages)
    print(f"scenarios {len(scenarios)}  stages {sorted(owned)}")
    for index in range(0, len(rows), 8):
        chunk = rows[index:index + 8]
        print(f"  rows[{index}:{index + len(chunk)}] styles="
              f"{sorted({row.style for row in chunk})} seeds="
              f"{[row.seed for row in chunk]}")

    started = time.perf_counter()
    # Reference arm: every scenario planned alone (B = 1 semantics) with the
    # stage snapshots the safety numbers need.  Untimed: the snapshots force a
    # host read per stage.
    reference, single_times = bake_module.record_serial(
        library.pipeline, library.bundle, scenarios, capture_trace=True,
        capture_stages=True)
    campaign = bake_module.drive_batch_single_pass(
        library.pipeline, library.bundle, scenarios, stages, capture_trace=True)
    if len(campaign.motions) != len(scenarios):
        print(f"error: bake finished {len(campaign.motions)}/{len(scenarios)} scenarios",
              file=sys.stderr)
        return 2
    comparison = bake_module.compare([record.run for record in reference],
                                     campaign.runs)
    perturbations = bake_module.stage_perturbations(reference, campaign.runs)
    margins = bake_module.decision_margins(reference)
    safety = bake_module.safety_factor(perturbations, margins)
    elapsed = time.perf_counter() - started

    token_groups: dict[str, int] = {}
    for record in reference:
        token_groups[str(record.tokens)] = token_groups.get(str(record.tokens), 0) + 1
    declared = "duration-homogeneous" if args.grouping == "homogeneous" else "duration-mixed"
    observed = "duration-homogeneous" if len(token_groups) == 1 else "duration-mixed"
    if args.strict and declared != observed:
        print(f"error: declared {declared} but the campaign split into {token_groups}",
              file=sys.stderr)
        return 2

    manifest = bake_module.manifest_fields(
        batch_size=len(scenarios), stages=owned, grouping=observed,
        token_groups=token_groups, comparison=comparison, safety=safety)
    missing = [field for field in bake_module.BAKE_MANIFEST_FIELDS
               if field not in manifest]
    if missing:
        print(f"error: manifest is missing {missing}", file=sys.stderr)
        return 2

    out = args.out
    tag = f"B{len(scenarios)}"
    plans_dir = out / "plans" / f"batch-{tag}"
    products = [library.write(plans_dir, row, campaign.motions[index])
                for index, row in enumerate(rows)]

    (out / f"runs-batch-{tag}.json").write_text(json.dumps({
        "arm": "batch", "device": args.device, "batch_size": len(scenarios),
        "release_stage_flag": RELEASE_STAGE_FLAG, "switches": switches,
        "stages": sorted(owned), "grouping_declared": declared,
        "grouping_observed": observed, "token_groups": token_groups,
        "plans_dir": str(plans_dir),
        "scenarios": [run_record(run.label, row, row.style, run.selected, run.tokens,
                                 run.uniforms, run.roots, run.rotations)
                      for run, row in zip(campaign.runs, rows)],
        "reference_single_scenario_ms_p50": statistics.median(single_times),
    }, indent=1) + "\n", encoding="utf-8")

    entry = dict(manifest)
    entry.update({
        "manifest_check": {
            "required_fields": list(bake_module.BAKE_MANIFEST_FIELDS),
            "present_fields": [f for f in bake_module.BAKE_MANIFEST_FIELDS if f in manifest],
            "missing_fields": missing,
            "complete": not missing,
            "release_ratio_threshold": bake_module.RELEASE_RATIO,
            "grouped_above_b": bake_module.GROUPED_ABOVE_B,
        },
        "grouping_declared": declared,
        "grouping_observed": observed,
        "token_groups": token_groups,
        "timing": time_arms(library, scenarios, stages, args.repeats),
        "reference_single_scenario_ms_p50": statistics.median(single_times),
        "reference_single_scenario_ms": single_times,
        "perturbations": perturbations,
        "margins": margins,
        "safety": safety,
        "safety_detail": safety_detail(bake_module, reference, campaign.runs),
        "suspensions": campaign.suspensions,
        "campaign": {"name": args.campaign,
                     "scenarios": [{"label": row.label, "style": row.style,
                                    "movement": list(row.movement), "seed": row.seed}
                                   for row in rows]},
        "products": products,
        "device": args.device,
        "bundle": str(args.bundle),
        "release_stage_flag": RELEASE_STAGE_FLAG,
        "resolved_switches": switches,
        "elapsed_seconds": round(elapsed, 2),
        "load_before": load_before,
        "load_after": print_load("after"),
    })

    manifest_path = out / "manifest.json"
    payload = (json.loads(manifest_path.read_text(encoding="utf-8"))
               if manifest_path.is_file() else {
                   "schema": "motionbricks_mlx.bake.library/1",
                   "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "bundle": str(args.bundle),
                   "styles_dir": str(args.styles),
                   "device": args.device,
                   "python": platform.python_version(),
                   "release_stage_flag": RELEASE_STAGE_FLAG,
                   "release_switches": list(RELEASE_SWITCHES),
                   "campaign_name": args.campaign,
                   "campaign_definition": [
                       {"label": row.label, "style": row.style,
                        "movement": list(row.movement), "seed": row.seed}
                       for row in args.campaign_rows],
                   "groups": {}})
    payload["groups"][tag] = entry
    payload["load_after"] = entry["load_after"]
    manifest_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    print(f"manifest {manifest_path}  group {tag}: "
          f"{len(bake_module.BAKE_MANIFEST_FIELDS)}/{len(bake_module.BAKE_MANIFEST_FIELDS)}"
          f" fields present")
    print(f"  tier={manifest['bake_tier']}  ratio={manifest['bake_safety_ratio']}  "
          f"tokens differing={manifest['bake_token_decisions_differing']}/"
          f"{comparison['total_tokens']}  worst roots={manifest['bake_worst_roots_m']:.3e} m "
          f"rotations={manifest['bake_worst_rotations']:.3e}  "
          f"bit_identical={manifest['bake_action_bit_identical']}")
    print(f"  timing serial p50 {entry['timing']['serial_ms_p50']:.2f} ms  "
          f"single pass p50 {entry['timing']['single_pass_ms_p50']:.2f} ms  "
          f"speedup {entry['timing']['speedup']:.3f}x  "
          f"{entry['timing']['throughput_per_s']:.1f} scenarios/s")
    library.release()
    return 0


# ---------------------------------------------------------------------------
# single: the independent B = 1 arm, in its own process
# ---------------------------------------------------------------------------


def cmd_single(args) -> int:
    switches = resolved_switches()
    print(f"resolved bake switches: {switches}")
    if any(switches.values()):
        print("error: the B = 1 arm must run with every bake switch off",
              file=sys.stderr)
        return 2
    load_before = print_load("before")
    library = Library(args.bundle, args.styles, args.device, args.campaign_rows)
    rows = rows_for(args.campaign_rows, args.batch_size)
    out = args.out
    tag = f"B{len(rows)}"
    plans_dir = out / "plans" / f"single-{tag}"

    records: list[dict] = []
    products: list[dict] = []
    times: list[float] = []
    for row in rows:
        style = library.styles[row.style]
        agent = Agent(library.pipeline, library.bundle.neutral_joints,
                      library.bundle.joint_parents)
        agent.reset(style)
        command = Command(style=style, movement=row.movement, facing=FACING_XYZ,
                          seed=row.seed)
        started = time.perf_counter()
        motion, info = agent.plan(command, style, capture_trace=True)
        times.append((time.perf_counter() - started) * 1000.0)
        trace = info.get("trace", {})
        records.append(run_record(
            row.label, row, row.style, int(info["selected_tokens"]),
            trace.get("pose.tokens", np.zeros(0)),
            trace.get("pose.sampling_uniforms", np.zeros(0)),
            motion.roots, motion.rotations_xyzw))
        products.append(library.write(plans_dir, row, motion))

    (out / f"runs-single-{tag}.json").write_text(json.dumps({
        "arm": "single", "device": args.device, "batch_size": len(rows),
        "switches": switches, "plans_dir": str(plans_dir),
        "scenarios": records,
        "single_scenario_ms_p50": statistics.median(times),
        "single_scenario_ms": times,
        "products": products,
        "load_before": load_before, "load_after": print_load("after"),
    }, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} single-scenario products to {plans_dir}")
    library.release()
    return 0


# ---------------------------------------------------------------------------
# compare: is the batch telling the truth about the single scenario?
# ---------------------------------------------------------------------------


def _document(plans_dir: Path, row: dict) -> dict:
    path = plans_dir / f"{plans_dir.name}-{row['label']}-{row['style']}.json"
    if not path.is_file():
        raise SystemExit(f"missing plan document {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_compare(args) -> int:
    batch_runs = json.loads(Path(args.batch).read_text(encoding="utf-8"))
    single_runs = json.loads(Path(args.single).read_text(encoding="utf-8"))
    single_by_label = {item["label"]: item for item in single_runs["scenarios"]}
    rows = []
    for item in batch_runs["scenarios"]:
        other = single_by_label.get(item["label"])
        if other is None:
            raise SystemExit(f"single arm has no scenario {item['label']!r}")
        left = _document(Path(batch_runs["plans_dir"]), item)["steps"][0]
        right = _document(Path(single_runs["plans_dir"]), other)["steps"][0]
        batch_tokens = np.asarray(item["tokens"], np.int64)
        single_tokens = np.asarray(other["tokens"], np.int64)
        batch_uniforms = np.asarray(item["uniforms"], np.float32)
        single_uniforms = np.asarray(other["uniforms"], np.float32)
        batch_roots = np.asarray(left["motion"]["roots"], np.float64).reshape(-1, 3)
        single_roots = np.asarray(right["motion"]["roots"], np.float64).reshape(-1, 3)
        batch_rot = np.asarray(left["motion"]["rotations"], np.float64).reshape(-1, 34, 4)
        single_rot = np.asarray(right["motion"]["rotations"], np.float64).reshape(-1, 34, 4)
        batch_world = np.asarray(left["world_positions_xyz"], np.float64)
        single_world = np.asarray(right["world_positions_xyz"], np.float64)
        same_frames = batch_roots.shape == single_roots.shape
        same_tokens = batch_tokens.shape == single_tokens.shape
        dot = np.abs(np.sum(batch_rot * single_rot, axis=-1)) if same_frames else None
        rows.append({
            "label": item["label"], "style": item["style"], "seed": item["seed"],
            "frames": int(batch_roots.shape[0]),
            "frames_same": bool(same_frames),
            "tokens_identical": bool(same_tokens
                                     and np.array_equal(batch_tokens, single_tokens)),
            "tokens": int(batch_tokens.size),
            "differing_tokens": (int(np.count_nonzero(batch_tokens != single_tokens))
                                 if same_tokens else -1),
            "uniforms_identical": bool(batch_uniforms.shape == single_uniforms.shape
                                       and np.array_equal(batch_uniforms, single_uniforms)),
            "duration_identical": item["selected_token"] == other["selected_token"],
            "selected_token_batch": item["selected_token"],
            "selected_token_single": other["selected_token"],
            "roots_identical": bool(np.array_equal(batch_roots, single_roots)),
            "rotations_identical": bool(np.array_equal(batch_rot, single_rot)),
            "world_identical": bool(np.array_equal(batch_world, single_world)),
            "worst_roots_m": (float(np.abs(batch_roots - single_roots).max())
                              if same_frames else float("nan")),
            "worst_rotations": (float(np.abs(batch_rot - single_rot).max())
                                if same_frames else float("nan")),
            "worst_angle_deg": (float(np.degrees(2.0 * np.arccos(
                np.clip(dot, -1.0, 1.0))).max()) if same_frames else float("nan")),
            "worst_world_m": (float(np.abs(batch_world - single_world).max())
                              if batch_world.shape == single_world.shape
                              else float("nan")),
            "roots_sha256_batch": item["roots_sha256"],
            "roots_sha256_single": other["roots_sha256"],
            "rotations_sha256_batch": item["rotations_sha256"],
            "rotations_sha256_single": other["rotations_sha256"],
        })
    summary = {
        "scenarios": len(rows),
        "token_identical": sum(row["tokens_identical"] for row in rows),
        "uniform_identical": sum(row["uniforms_identical"] for row in rows),
        "duration_identical": sum(row["duration_identical"] for row in rows),
        "roots_bit_identical": sum(row["roots_identical"] for row in rows),
        "rotations_bit_identical": sum(row["rotations_identical"] for row in rows),
        "world_bit_identical": sum(row["world_identical"] for row in rows),
        "differing_tokens": sum(max(0, row["differing_tokens"]) for row in rows),
        "total_tokens": sum(row["tokens"] for row in rows),
        "worst_roots_m": max(row["worst_roots_m"] for row in rows),
        "worst_rotations": max(row["worst_rotations"] for row in rows),
        "worst_angle_deg": max(row["worst_angle_deg"] for row in rows),
        "worst_world_m": max(row["worst_world_m"] for row in rows),
    }
    payload = {
        "batch_arm": {"file": str(args.batch), "batch_size": batch_runs["batch_size"],
                      "stages": batch_runs["stages"],
                      "switches": batch_runs["switches"],
                      "token_groups": batch_runs["token_groups"]},
        "single_arm": {"file": str(args.single),
                       "batch_size": single_runs["batch_size"],
                       "switches": single_runs["switches"],
                       "single_scenario_ms_p50": single_runs["single_scenario_ms_p50"]},
        "summary": summary, "rows": rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"{'label':>5} {'style':<17} {'seed':>4} {'fr':>3} {'tok':>4} {'uni':>4} "
          f"{'dur':>4} {'roots':>6} {'rot':>6} {'worst_roots_m':>14} "
          f"{'worst_rot':>12} {'angle_deg':>10}")
    for row in rows:
        flag = lambda value: "yes" if value else "NO"  # noqa: E731
        print(f"{row['label']:>5} {row['style']:<17} {row['seed']:>4} "
              f"{row['frames']:>3} {flag(row['tokens_identical']):>4} "
              f"{flag(row['uniforms_identical']):>4} {flag(row['duration_identical']):>4} "
              f"{flag(row['roots_identical']):>6} {flag(row['rotations_identical']):>6} "
              f"{row['worst_roots_m']:>14.3e} {row['worst_rotations']:>12.3e} "
              f"{row['worst_angle_deg']:>10.4f}")
    print(f"identity: tokens {summary['token_identical']}/{summary['scenarios']}  "
          f"uniforms {summary['uniform_identical']}/{summary['scenarios']}  "
          f"duration {summary['duration_identical']}/{summary['scenarios']}  "
          f"roots {summary['roots_bit_identical']}/{summary['scenarios']}  "
          f"rotations {summary['rotations_bit_identical']}/{summary['scenarios']}")
    print(f"{summary['differing_tokens']}/{summary['total_tokens']} token decisions differ; "
          f"worst roots {summary['worst_roots_m']:.3e} m, "
          f"rotations {summary['worst_rotations']:.3e}, "
          f"angle {summary['worst_angle_deg']:.4f} deg, "
          f"world {summary['worst_world_m']:.3e} m")
    if args.out:
        print(f"wrote {args.out}")
    return 0


# ---------------------------------------------------------------------------
# check: the twelve fields, on the manifest as written
# ---------------------------------------------------------------------------


def cmd_check(args) -> int:
    from motionbricks_mlx.models import bake as bake_module

    path = Path(args.manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = list(bake_module.BAKE_MANIFEST_FIELDS)
    report = {"manifest": str(path), "required_fields": required, "groups": {},
              "passed": True}
    for tag, group in sorted(payload.get("groups", {}).items()):
        missing = [field for field in required if field not in group]
        report["groups"][tag] = {
            "present": [field for field in required if field in group],
            "missing": missing, "count": len(required) - len(missing),
            "expected": len(required),
            "values": {field: group.get(field) for field in required},
        }
        report["passed"] = report["passed"] and not missing
    for tag, group in report["groups"].items():
        print(f"group B={tag}: {group['count']}/{group['expected']} required fields present"
              + (f"  MISSING {group['missing']}" if group["missing"] else ""))
        for field in required:
            value = group["values"][field]
            rendered = json.dumps(value, sort_keys=True)
            if len(rendered) > 88:
                rendered = rendered[:85] + "..."
            print(f"    {field:<32} {rendered}")
    print(f"manifest check: {'PASS' if report['passed'] else 'FAIL'}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if report["passed"] else 1


# ---------------------------------------------------------------------------
# metrics: the C-line quality metrics, on the exported products
# ---------------------------------------------------------------------------


def cmd_metrics(args) -> int:
    from motionbricks_mlx.eval.document import load_step
    from motionbricks_mlx.eval.gate import evaluate_step
    from motionbricks_mlx.eval.thresholds import THRESHOLDS

    documents = sorted(Path(args.dir).glob("*.json"))
    if not documents:
        print(f"error: no plan documents under {args.dir}", file=sys.stderr)
        return 2
    payload: dict = {"thresholds_from": "motionbricks_mlx.eval.thresholds.THRESHOLDS",
                     "steps": {}, "passed": 0, "failed": 0}
    for path in documents:
        document = json.loads(path.read_text(encoding="utf-8"))
        step = load_step(document, 0)
        values = evaluate_step(step, THRESHOLDS)
        payload["steps"][path.name] = {
            "label": step.label, "style": step.style, "frames": step.frames,
            "passed": bool(values["passed"]),
            "values": {key: value for key, value in values.items()
                       if isinstance(value, (int, float, str, bool))},
            "verdicts": [{"key": v.key, "value": v.value, "limit": v.limit,
                          "direction": v.direction, "passed": v.passed}
                         for v in values["verdicts"]],
            "structure_problems": values["structure"]["problems"],
        }
        payload["passed" if values["passed"] else "failed"] += 1
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"{len(payload['steps'])} documents: {payload['passed']} pass, "
          f"{payload['failed']} fail (structure + metrics 1-4, 6)")
    for name, item in payload["steps"].items():
        red = [v["key"] for v in item["verdicts"] if not v["passed"]]
        print(f"  {'PASS' if item['passed'] else 'FAIL'}  {name:<44} "
              f"frames={item['frames']:<3} {item['style']:<17} "
              f"{'red=' + ','.join(red) if red else ''}")
    if args.out:
        print(f"wrote {args.out}")
    return 0


# ---------------------------------------------------------------------------
# pairdoc: one document carrying both arms of one scenario, for the renderer
# ---------------------------------------------------------------------------


def cmd_pairdoc(args) -> int:
    pairs = []
    for label, style in args.pairs:
        batch_dir = Path(args.batch_dir)
        single_dir = Path(args.single_dir)
        left = _document(batch_dir, {"label": label, "style": style})["steps"][0]
        right = _document(single_dir, {"label": label, "style": style})["steps"][0]
        skeleton = left.get("skeleton") or right.get("skeleton")
        if skeleton is None:
            print(f"error: no skeleton block for {label}", file=sys.stderr)
            return 2
        left = dict(left, label=f"{label}-batch", skeleton=skeleton)
        right = dict(right, label=f"{label}-single")
        right.pop("skeleton", None)
        document = {"library": "motionbricks_mlx.bake",
                    "model_bundle": args.bundle_name, "device": args.device_name,
                    "sampling": "gumbel", "fps": 30, "joints": len(skeleton),
                    "learned_f32_parameters": 0,
                    "steps": [left, right]}
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"render-{label}.json"
        path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        pairs.append({"label": label, "style": style, "document": str(path)})
        print(f"{path}  steps {left['label']}, {right['label']}")
    if args.manifest:
        Path(args.manifest).write_text(json.dumps({"pairs": pairs}, indent=1) + "\n",
                                       encoding="utf-8")
        print(f"wrote {args.manifest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--styles", type=Path, default=DEFAULT_STYLES)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--campaign", choices=tuple(CAMPAIGNS), default="diverse",
                        help="diverse: 8 real styles, 24 frames; single-style: one "
                             "style, 16 seeds, 44 frames (the rounds 4-6 shape)")
    sub = parser.add_subparsers(dest="command", required=True)

    batch = sub.add_parser("batch", help="release-configuration bake + manifest")
    batch.add_argument("--out", type=Path, required=True)
    batch.add_argument("--batch-size", type=int, default=16)
    batch.add_argument("--grouping", choices=("homogeneous", "mixed"),
                       default="homogeneous")
    batch.add_argument("--repeats", type=int, default=5)
    batch.add_argument("--strict", action="store_true",
                       help="fail when the observed grouping disagrees with --grouping")
    batch.set_defaults(func=cmd_batch)

    single = sub.add_parser("single", help="independent B = 1 arm, own process")
    single.add_argument("--out", type=Path, required=True)
    single.add_argument("--batch-size", type=int, default=16)
    single.set_defaults(func=cmd_single)

    compare = sub.add_parser("compare", help="batch vs single, scenario by scenario")
    compare.add_argument("--batch", type=Path, required=True)
    compare.add_argument("--single", type=Path, required=True)
    compare.add_argument("--out", type=Path, default=None)
    compare.set_defaults(func=cmd_compare)

    check = sub.add_parser("check", help="the twelve manifest fields, as written")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--out", type=Path, default=None)
    check.set_defaults(func=cmd_check)

    metrics = sub.add_parser("metrics", help="C-line metric verdicts on the products")
    metrics.add_argument("--dir", type=Path, required=True)
    metrics.add_argument("--out", type=Path, default=None)
    metrics.set_defaults(func=cmd_metrics)

    pair = sub.add_parser("pairdoc", help="one render document per scenario, both arms")
    pair.add_argument("--batch-dir", type=Path, required=True)
    pair.add_argument("--single-dir", type=Path, required=True)
    pair.add_argument("--pair", action="append", default=[], metavar="LABEL=STYLE",
                      required=True)
    pair.add_argument("--out", type=Path, required=True)
    pair.add_argument("--manifest", type=Path, default=None)
    pair.add_argument("--bundle-name", default="motionbricks_mlx")
    pair.add_argument("--device-name", default="mlx-gpu")
    pair.set_defaults(func=cmd_pairdoc)

    args = parser.parse_args()
    args.campaign_rows = CAMPAIGNS[args.campaign]
    if args.command == "pairdoc":
        pairs = []
        for item in args.pair:
            label, separator, style = item.partition("=")
            if not separator:
                print(f"error: --pair {item!r} is not LABEL=STYLE", file=sys.stderr)
                return 2
            pairs.append((label.strip(), style.strip()))
        args.pairs = pairs
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
