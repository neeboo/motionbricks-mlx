#!/usr/bin/env python3
"""Plan motion that must *hit user-placed target keyframes*.

This is the additive sibling of ``scripts/mlx_plan.py``.  It does not change
that script, ``mlx_align.py`` or any existing semantics; it exposes the
*external* half of the transition boundary that the port already carried but
never let a caller set.

The boundary is an 8-frame window (``src/root.cpp:366``): frames 0..3 are the
current state, frames 4..7 are the goal, and the root planner feeds exactly
frames 4..7 into the ``dense_target`` convo-lution condition
(``src/root.cpp:463-479``).  ``build_constraints`` normally manufactures those 4
frames from a style clip plus a spring toward a command-derived goal; this script
replaces them with targets you place, and then *measures* how close the generated
clip gets.

Single plan::

    ./.venv/bin/python scripts/mlx_plan_targets.py --style walk --seed 1 \\
        --from-current --target 0,3,0 --out /tmp/plan-targets.json

Negative control (does displacement actually follow the target?)::

    ./.venv/bin/python scripts/mlx_plan_targets.py --style walk --from-current \\
        --sweep 0,0.5,1,2,3,5,10,50

The plan document has the same shape as ``mlx_plan.py``'s (so ``mlx_align plan``
can still read it); two keys are appended to the step: ``placed_targets`` (the
window that was asked for) and ``target_hit`` (the measured errors and their
declared tolerances).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motionbricks_mlx.data.agent import Agent, Command, quaternion_matrix  # noqa: E402
from motionbricks_mlx.data.style import load_style                          # noqa: E402
from motionbricks_mlx.data.targets import (PlacedTarget, TARGET_TOLERANCES,  # noqa: E402
                                            measure_hit, parse_target,
                                            plan_with_targets)
from motionbricks_mlx.export.plan_json import Step, build_document, write_document  # noqa: E402
from motionbricks_mlx.models.bundle import Bundle                           # noqa: E402
from motionbricks_mlx.models.transition import MlxPipeline                  # noqa: E402

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "checkpoints/g1-f32-mlx"
DEFAULT_STYLES = Path.home() / "dev/motion-bricks-cpp/repo/generated/styles"


def three_vector(pair: tuple[float, float]) -> tuple[float, float, float]:
    return (pair[0], 0.0, pair[1])


def clip_start(agent: Agent) -> tuple[float, float, float]:
    """World ``(x, z, heading_rad)`` of the clip the next plan will start from."""
    root = np.asarray(agent.context_roots[0], dtype=np.float64)
    matrix = quaternion_matrix(np.asarray(agent.context_rotations[0, 0], dtype=np.float64))
    return float(root[0]), float(root[2]), float(np.arctan2(matrix[0, 2], matrix[2, 2]))


def resolve_targets(agent: Agent, raw: list[str], relative: bool) -> list[PlacedTarget]:
    """``--target`` strings -> world :class:`PlacedTarget`s.

    World by default; with ``--from-current`` each ``x,z`` is an offset from the
    clip start measured along the character's own facing, which is how a page
    click at "3 m in front" is expressed.
    """
    origin_x, origin_z, heading = clip_start(agent)
    targets: list[PlacedTarget] = []
    for text in raw:
        parsed = parse_target(text)
        if relative:
            cosine, sine = np.cos(heading), np.sin(heading)
            world_x = origin_x + parsed.x * cosine + parsed.z * sine
            world_z = origin_z - parsed.x * sine + parsed.z * cosine
            heading_deg = np.rad2deg(heading) + parsed.heading_deg
        else:
            world_x, world_z, heading_deg = parsed.x, parsed.z, parsed.heading_deg
        targets.append(PlacedTarget(float(world_x), float(world_z), float(heading_deg)))
    return targets


def one_plan(pipeline: MlxPipeline, bundle: Bundle, style, seed: int, move, facing,
             raw_targets: list[str], relative: bool, pose_mode: str,
             argmax: bool = False, advance: int = 0):
    agent = Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
    agent.reset(style)
    if advance:
        agent.advance(int(advance))
    command = Command(style=style, movement=three_vector(move), facing=three_vector(facing),
                      seed=int(seed), sampling_argmax=argmax)
    targets = resolve_targets(agent, raw_targets, relative)
    started = time.perf_counter()
    motion, info, window = plan_with_targets(agent, command, style, targets, pose_mode)
    elapsed = (time.perf_counter() - started) * 1000.0
    hit = measure_hit(motion, window, bundle.joint_parents, bundle.neutral_joints)
    return motion, info, window, hit, elapsed


def reproduce(pipeline: MlxPipeline, bundle: Bundle, style, seed: int, move, facing,
              raw_targets: list[str], relative: bool, pose_mode: str,
              argmax: bool) -> tuple[list[Step], dict, dict]:
    """Run the plan twice; refuse to write a document that is not deterministic."""
    first, info, window, hit, elapsed = one_plan(pipeline, bundle, style, seed, move, facing,
                                                 raw_targets, relative, pose_mode, argmax)
    second, _, _, _, elapsed2 = one_plan(pipeline, bundle, style, seed, move, facing,
                                         raw_targets, relative, pose_mode, argmax)
    for name, left, right in (("roots", first.roots, second.roots),
                              ("rotations", first.rotations_xyzw, second.rotations_xyzw),
                              ("target_roots", first.target_roots, second.target_roots)):
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            raise SystemExit(f"NOT DETERMINISTIC: {name} differs between two identical runs")
    step = Step(label="placed-target", style=style.name, move=move, facing=facing,
                seed=int(seed), advance=0, roots=first.roots,
                rotations_xyzw=first.rotations_xyzw, target_roots=first.target_roots,
                target_rotations_xyzw=first.target_local_rotations_xyzw)
    placed = {
        "count": len(window.placed),
        "pose_mode": window.pose_mode,
        "frames": window.frames,
        "input_frame": "world" if not relative else "offset-from-clip-start",
        "world_roots": [[float(v) for v in row] for row in window.world_roots],
        "world_headings_deg": [float(v) for v in window.world_headings_deg],
        "requested": [{"x": float(t.x), "z": float(t.z), "heading_deg": float(t.heading_deg)}
                      for t in window.placed],
        "selected_tokens": int(info["selected_tokens"]),
        "plan_ms": [round(elapsed, 4), round(elapsed2, 4)],
    }
    return [step], placed, hit


def print_hit(hit: dict) -> None:
    errors, tolerances = hit["errors"], hit["tolerances"]
    print(f"  window  : output frames {hit['window_frames'][0]}..{hit['window_frames'][1]} "
          f"of {hit['frames']} (the 4 samples land on the clip tail; see "
          f"motionbricks_mlx.data.targets)")
    print(f"  root    : {errors['root_planar_m']:.4f} m planar "
          f"({errors['root_max_abs_m']:.4f} m max |axis|)  tol "
          f"{tolerances['root_planar_m']:.2f} m")
    print(f"  heading : {errors['heading_deg']:.3f} deg"
          f"  tol {tolerances['heading_deg']:.2f} deg")
    print(f"  joints  : {errors['joint_m']:.4f} m (worst world joint)"
          f"  tol {tolerances['joint_m']:.2f} m")
    for slot in hit["per_slot"]:
        print(f"    slot {slot['slot']} @frame {slot['output_frame']:3d}: "
              f"root {slot['root_planar_m']:.4f} m  heading {slot['heading_deg']:6.3f} deg  "
              f"joint {slot['joint_m']:.4f} m")
    verdict = "WITHIN TOLERANCE" if hit["within_tolerance"] else \
        f"OUTSIDE TOLERANCE ({', '.join(hit['exceeded'])})"
    print(f"  verdict : {verdict}")


def run_sweep(pipeline: MlxPipeline, bundle: Bundle, style, seed: int, move, facing,
              distances: list[float], pose_mode: str, argmax: bool) -> int:
    """The negative control: does displacement follow the placed target?

    Prints one line per distance with the achieved root displacement, the peak
    implied root speed and the hit error, then states whether displacement was
    monotone.  A target the clip cannot reach is *not* silently clamped -- the
    row shows what actually happened.
    """
    print(f"sweep: style={style.name} seed={seed} pose_mode={pose_mode} "
          f"(target placed straight ahead of the clip start)")
    previous = None
    monotone = True
    rows = []
    for distance in distances:
        motion, info, _, hit, _ = one_plan(pipeline, bundle, style, seed, move, facing,
                                          [f"0,{distance:g},0"], True, pose_mode, argmax)
        displacement = float(motion.roots[-1, 2] - motion.roots[0, 2])
        peak = float(np.linalg.norm(np.diff(np.asarray(motion.roots), axis=0),
                                    axis=1).max() * 30.0)
        finite = bool(np.isfinite(np.asarray(motion.roots)).all())
        if previous is not None and displacement <= previous:
            monotone = False
        previous = displacement
        errors = hit["errors"]
        rows.append((distance, displacement, peak, errors, finite))
        print(f"  target {distance:6.2f} m -> displacement {displacement:8.3f} m  "
              f"peak root speed {peak:7.2f} m/s  root hit {errors['root_planar_m']:8.4f} m  "
              f"heading {errors['heading_deg']:6.2f} deg  tokens={info['selected_tokens']:2d}  "
              f"finite={finite}  {'ok' if hit['within_tolerance'] else 'OUT OF TOLERANCE'}")
    print(f"  monotone displacement in target distance: {monotone}")
    print("  NOTE: nothing is clamped or refused.  Beyond the distance the clip can cover "
          "the model still extrapolates; read the peak root speed column for whether the "
          "result is physically plausible (human walk is ~1-2 m/s).")
    return 0 if monotone else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--styles", type=Path, default=DEFAULT_STYLES)
    parser.add_argument("--style", default="walk")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--move", default="0,1", help="world travel direction 'x,z'")
    parser.add_argument("--facing", default="0,1", help="world facing direction 'x,z'")
    parser.add_argument("--target", action="append", default=[],
                        metavar="X,Z[,HEADING_DEG]",
                        help="a placed target keyframe; repeat up to 4 times")
    parser.add_argument("--from-current", action="store_true",
                        help="read every --target as an offset from the clip start")
    parser.add_argument("--pose-mode", choices=("style", "hold"), default="style",
                        help="joint configuration at the target: the style's own "
                             "(default) or the one the clip starts from")
    parser.add_argument("--sweep", default=None, metavar="D1,D2,...",
                        help="negative control: place one target straight ahead at each "
                             "distance and report displacement / hit / peak speed")
    parser.add_argument("--advance", type=int, default=0)
    parser.add_argument("--argmax", action="store_true")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--out", type=Path, required=True,
                        help="plan document to write (required, like mlx_plan.py)")
    args = parser.parse_args()

    if not (args.bundle / "manifest.json").is_file():
        print(f"error: {args.bundle} has no manifest.json", file=sys.stderr)
        return 2
    move = tuple(float(v) for v in args.move.split(","))
    facing = tuple(float(v) for v in args.facing.split(","))
    if len(move) != 2 or len(facing) != 2:
        parser.error("--move and --facing take 'x,z'")

    style = load_style(args.styles / f"{args.style}.mbstyle")
    bundle = Bundle(args.bundle, device=args.device)
    pipeline = MlxPipeline(bundle)

    if args.sweep:
        distances = [float(v) for v in args.sweep.split(",") if v.strip()]
        return run_sweep(pipeline, bundle, style, args.seed, move, facing, distances,
                         args.pose_mode, args.argmax)

    if not args.target:
        parser.error("at least one --target (or --sweep) is required; this script only "
                     "exists for placed targets -- use scripts/mlx_plan.py otherwise")
    if len(args.target) > 4:
        parser.error(f"at most 4 --target values, got {len(args.target)}")

    steps, placed, hit = reproduce(pipeline, bundle, style, args.seed, move, facing,
                                   args.target, args.from_current, args.pose_mode,
                                   args.argmax)
    parameters = sum(bundle.manifest["files"][name]["parameter_count"]
                     for name in ("pose", "root", "vq-decoder"))
    document = build_document(
        library="motionbricks_mlx", model_bundle=str(args.bundle),
        device=f"mlx-{args.device}", sampling="argmax" if args.argmax else "gumbel",
        learned_parameters=parameters, steps=steps, parents=bundle.joint_parents,
        rest=bundle.neutral_joints, joint_names=bundle.joint_names)
    # Additive only: the reference-shaped keys above are untouched, so
    # ``mlx_align plan`` still reads this document.
    document["steps"][0]["placed_targets"] = placed
    document["steps"][0]["target_hit"] = hit
    document["placed_targets"] = placed
    document["target_hit"] = hit
    document["target_tolerances"] = dict(TARGET_TOLERANCES)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_document(args.out, document)

    print(f"placed {len(placed['requested'])} target frame(s), pose_mode={placed['pose_mode']}, "
          f"seed={args.seed}, style={args.style}")
    for index, entry in enumerate(placed["requested"]):
        print(f"  placed[{index}] world x={entry['x']:.3f} z={entry['z']:.3f} "
              f"heading={entry['heading_deg']:.2f} deg")
    print(f"  window (world): "
          f"{[[round(v, 3) for v in row] for row in placed['world_roots']]}")
    print(f"  planned {hit['frames']} frames, {placed['selected_tokens']} tokens, "
          f"plan {placed['plan_ms'][0]:.1f} ms")
    print_hit(hit)
    print(f"wrote {args.out}")
    print(json.dumps({"within_tolerance": hit["within_tolerance"],
                      "errors": hit["errors"]}, indent=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
