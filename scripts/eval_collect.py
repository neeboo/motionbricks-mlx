#!/usr/bin/env python3
"""Collect the gate's fixed scenario set from the f32 MLX pipeline.

Writes, under out/eval/:

  plan-seed-<seed>.json     one-step walk plans, one per seed group (6 groups,
                            the >=5 the brief requires), full reference-shaped
                            documents including world_positions_xyz
  plan-refpair.json         the exact reference Go driver scenario (session
                            -initial seed 1 -> advance 3 -> replan-turned seed
                            77), used for metric 5 against the reference JSON
  tokens-refpair.json       the MLX pose.tokens (Agent capture_trace) for both
                            reference-pair steps
  collect-meta.json         bundle, device, seeds, library versions

The scenarios live here and nowhere else; scripts/eval_gate.py imports
collect_all() when its inputs are missing.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from argparse import RawDescriptionHelpFormatter
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
DEFAULT_OUT = REPO_ROOT / "out/eval"

# The gate's fixed seed groups: same command, different sampling seeds.  Seed 1
# and 77 overlap the reference driver's scenario so the cross-impl pairs and
# the seed groups share provenance.
SEED_GROUPS = (1, 7, 13, 42, 77, 123)
GROUP_COMMAND = {"style": "walk", "move": (1.0, 0.0), "facing": (0.0, 1.0)}

# The reference Go driver's scenario (tools/mbplan/main.go:212-213), replayed
# exactly for metric 5 — label, style, move, facing, seed, advance.
REFERENCE_SCENARIO = (
    ("session-initial", "walk", (0.0, 0.0), (0.0, 1.0), 1, 0),
    ("replan-turned", "walk_zombie", (1.0, 0.0), (1.0, 0.0), 77, 3),
)


def _three(pair: tuple[float, float]) -> tuple[float, float, float]:
    return (pair[0], 0.0, pair[1])


def collect_all(bundle_dir: Path = DEFAULT_BUNDLE, styles_dir: Path = DEFAULT_STYLES,
                out_dir: Path = DEFAULT_OUT, device: str = "cpu") -> dict:
    """Generate every gate input; returns the meta dict written alongside."""
    if not (bundle_dir / "manifest.json").is_file():
        raise SystemExit(f"error: {bundle_dir} has no manifest.json; run "
                         f"scripts/mlx_convert_gguf.py first")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = Bundle(bundle_dir, device=device)
    pipeline = MlxPipeline(bundle)
    document_kwargs = dict(parents=bundle.joint_parents, rest=bundle.neutral_joints,
                           joint_names=bundle.joint_names)
    parameters = sum(bundle.manifest["files"][name]["parameter_count"]
                     for name in ("pose", "root", "vq-decoder"))

    started = time.perf_counter()

    # --- seed groups -----------------------------------------------------
    style_walk = load_style(styles_dir / f"{GROUP_COMMAND['style']}.mbstyle")
    group_labels = []
    for seed in SEED_GROUPS:
        agent = Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
        agent.reset(style_walk)
        motion, _info = agent.plan(Command(style=style_walk,
                                           movement=_three(GROUP_COMMAND["move"]),
                                           facing=_three(GROUP_COMMAND["facing"]),
                                           seed=seed, sampling_argmax=False), style_walk)
        step = Step(label=f"seed-{seed}", style=GROUP_COMMAND["style"],
                    move=GROUP_COMMAND["move"], facing=GROUP_COMMAND["facing"],
                    seed=seed, advance=0, roots=motion.roots,
                    rotations_xyzw=motion.rotations_xyzw,
                    target_roots=motion.target_roots,
                    target_rotations_xyzw=motion.target_local_rotations_xyzw)
        document = build_document(library="motionbricks_mlx",
                                  model_bundle=str(bundle_dir), device=f"mlx-{device}",
                                  sampling="gumbel", learned_parameters=parameters,
                                  steps=[step], **document_kwargs)
        path = out_dir / f"plan-seed-{seed}.json"
        write_document(path, document)
        group_labels.append(f"seed-{seed}")

    # --- reference pair (with token traces) ------------------------------
    pair_steps: list[Step] = []
    pair_tokens: list[list[int]] = []
    agent = Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
    first_style = load_style(styles_dir / f"{REFERENCE_SCENARIO[0][1]}.mbstyle")
    agent.reset(first_style)
    for index, (label, style_name, move, facing, seed, advance) in enumerate(REFERENCE_SCENARIO):
        if advance:
            agent.advance(advance)
        style = load_style(styles_dir / f"{style_name}.mbstyle")
        motion, info = agent.plan(Command(style=style, movement=_three(move),
                                          facing=_three(facing), seed=seed,
                                          sampling_argmax=False), style,
                                  capture_trace=True)
        pair_steps.append(Step(label=label, style=style_name, move=move, facing=facing,
                               seed=seed, advance=advance, roots=motion.roots,
                               rotations_xyzw=motion.rotations_xyzw,
                               target_roots=motion.target_roots,
                               target_rotations_xyzw=motion.target_local_rotations_xyzw))
        pair_tokens.append([int(t) for t in
                            np.asarray(info["trace"]["pose.tokens"]).reshape(-1)])
    document = build_document(library="motionbricks_mlx", model_bundle=str(bundle_dir),
                              device=f"mlx-{device}", sampling="gumbel",
                              learned_parameters=parameters, steps=pair_steps,
                              **document_kwargs)
    write_document(out_dir / "plan-refpair.json", document)
    (out_dir / "tokens-refpair.json").write_text(
        json.dumps({"scenario": [list(s) for s in REFERENCE_SCENARIO],
                    "steps": [{"label": s[0], "seed": s[4], "tokens": tokens}
                              for s, tokens in zip(REFERENCE_SCENARIO, pair_tokens)]},
                   indent=1) + "\n", encoding="utf-8")

    meta = {
        "bundle": str(bundle_dir),
        "styles": str(styles_dir),
        "device": f"mlx-{device}",
        "sampling": "gumbel",
        "seed_groups": list(SEED_GROUPS),
        "group_command": {k: list(v) if isinstance(v, tuple) else v
                          for k, v in GROUP_COMMAND.items()},
        "library": "motionbricks_mlx",
        "python": platform.python_version(),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "collect-meta.json").write_text(json.dumps(meta, indent=1) + "\n",
                                               encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--styles", type=Path, default=DEFAULT_STYLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    meta = collect_all(args.bundle, args.styles, args.out, args.device)
    print(json.dumps(meta, indent=1))
    print(f"wrote {len(SEED_GROUPS)} seed groups + reference pair to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
