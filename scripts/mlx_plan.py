#!/usr/bin/env python3
"""Plan motion with the MLX port and write the reference-shaped JSON.

Same inputs as the reference Go driver (style / movement / facing / seed /
advance) and the same output document, so a mechanical array-by-array
comparison against the reference's own JSON is possible:

    ./.venv/bin/python scripts/mlx_plan.py --out out/plan-mlx.json
    ./.venv/bin/python scripts/mlx_align.py out/ref/plan-ref.json out/plan-mlx.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motionbricks_mlx.data.agent import Agent, Command
from motionbricks_mlx.data.style import load_style
from motionbricks_mlx.export.plan_json import Step, build_document, write_document
from motionbricks_mlx.models.bundle import Bundle
from motionbricks_mlx.models.transition import MlxPipeline

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "checkpoints/g1-f32-mlx"
DEFAULT_STYLES = Path.home() / "dev/motion-bricks-cpp/repo/generated/styles"

# The reference Go driver's scenario (tools/mbplan/main.go:212-213).
REFERENCE_SCENARIO = (
    # label, style, move, facing, seed, advance
    ("session-initial", "walk", (0.0, 0.0), (0.0, 1.0), 1, 0),
    ("replan-turned", "walk_zombie", (1.0, 0.0), (1.0, 0.0), 77, 3),
)


def three_vector(pair: tuple[float, float]) -> tuple[float, float, float]:
    """The Go driver passes ``SetMovement(move[0], 0, move[1])``."""
    return (pair[0], 0.0, pair[1])


def run_scenario(pipeline: MlxPipeline, bundle: Bundle, styles: Path,
                 scenario=REFERENCE_SCENARIO, argmax: bool = False) -> list[Step]:
    first_style = load_style(styles / f"{scenario[0][1]}.mbstyle")
    agent = Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
    agent.reset(first_style)
    steps: list[Step] = []
    for label, style_name, move, facing, seed, advance in scenario:
        if advance:
            agent.advance(advance)
        style = load_style(styles / f"{style_name}.mbstyle")
        motion, _ = agent.plan(Command(style=style, movement=three_vector(move),
                                       facing=three_vector(facing), seed=seed,
                                       sampling_argmax=argmax), style)
        steps.append(Step(label=label, style=style_name, move=move, facing=facing,
                          seed=seed, advance=advance, roots=motion.roots,
                          rotations_xyzw=motion.rotations_xyzw,
                          target_roots=motion.target_roots,
                          target_rotations_xyzw=motion.target_local_rotations_xyzw))
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--styles", type=Path, default=DEFAULT_STYLES)
    parser.add_argument("--out", type=Path, default=Path("out/plan-mlx.json"))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                        help="MLX backend (mx.set_default_device)")
    parser.add_argument("--argmax", action="store_true",
                        help="diagnostic argmax sampling instead of Gumbel")
    parser.add_argument("--repeat", type=int, default=1,
                        help="plan the scenario N times (determinism check)")
    parser.add_argument("--time", action="store_true", help="print per-step wall time")
    args = parser.parse_args()

    if not (args.bundle / "manifest.json").is_file():
        print(f"error: {args.bundle} has no manifest.json; run "
              f"scripts/mlx_convert_gguf.py first", file=sys.stderr)
        return 2

    started = time.perf_counter()
    bundle = Bundle(args.bundle, device=args.device)
    pipeline = MlxPipeline(bundle)
    steps = run_scenario(pipeline, bundle, args.styles, argmax=args.argmax)
    load_seconds = time.perf_counter() - started

    if args.time:
        print(f"cold start (load + first plan): {load_seconds * 1000:.1f} ms")

    documents = []
    for _ in range(max(1, args.repeat)):
        started = time.perf_counter()
        repeated = run_scenario(pipeline, bundle, args.styles, argmax=args.argmax)
        documents.append(repeated)
        if args.time:
            print(f"scenario: {(time.perf_counter() - started) * 1000:.1f} ms")
        if args.repeat > 1:
            for earlier, later in zip(documents[0], repeated):
                if not (mx.array_equal(mx.array(earlier.roots), mx.array(later.roots))
                        and mx.array_equal(mx.array(earlier.rotations_xyzw),
                                           mx.array(later.rotations_xyzw))):
                    print(f"NOT DETERMINISTIC: step {earlier.label}", file=sys.stderr)
                    return 1

    parameters = sum(bundle.manifest["files"][name]["parameter_count"]
                     for name in ("pose", "root", "vq-decoder"))
    document = build_document(
        library="motionbricks_mlx", model_bundle=str(args.bundle),
        device=f"mlx-{args.device}", sampling="argmax" if args.argmax else "gumbel",
        learned_parameters=parameters, steps=documents[0], parents=bundle.joint_parents,
        rest=bundle.neutral_joints, joint_names=bundle.joint_names)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_document(args.out, document)
    print(f"wrote {args.out}")
    for step in documents[0]:
        travel = float(np.linalg.norm(np.diff(step.roots.astype(np.float64), axis=0),
                                      axis=1).sum())
        print(f"  {step.label:16s} style={step.style:13s} frames={step.roots.shape[0]:3d} "
              f"travelled={travel:.4f} m")
    if args.repeat > 1:
        print(f"deterministic over {args.repeat} runs with fixed seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
