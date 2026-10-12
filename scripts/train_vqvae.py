#!/usr/bin/env python3
"""Train / evaluate the MLX motion VQ-VAE (B-side scaffold CLI).

Only the repo interpreter (docs/decisions/0001-python-environment.md):

    ./.venv/bin/python scripts/train_vqvae.py synth  --data-root datasets/synth-smoke --n-clips 6
    ./.venv/bin/python scripts/train_vqvae.py train  --data-root datasets/synth-smoke --out checkpoints/smoke-001
    ./.venv/bin/python scripts/train_vqvae.py eval   --checkpoint checkpoints/smoke-001 [--data-root ...]

``synth`` regenerates the synthetic smoke shards (seeded, so a manifest's
``data`` ids stay reproducible across machines).  ``eval`` rebuilds the model
from the run's manifest, scores reconstruction with position / rotation
reported separately, and exits 1 when either threshold below is breached.

Default thresholds are calibrated on the synthetic smoke run recorded in
``docs/status-claude.md`` — they are a floor for *that* pipeline, not a
product-quality bar.  Override with --position-threshold / --rotation-threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motionbricks_mlx.training import eval as eval_mod  # noqa: E402
from motionbricks_mlx.training import metrics, synth  # noqa: E402
from motionbricks_mlx.training.train import TrainConfig, train  # noqa: E402

POSITION_THRESHOLD_M = 0.35
ROTATION_THRESHOLD_DEG = 60.0
# Calibrated on smoke run ``smoke-claude-001`` (synthetic shards, seed 0,
# 40 steps, defaults).  Full 6-clip CLI eval measured position_mean 0.275 m /
# rotation_mean 53.0 deg; the 4-clip in-train report measured 0.279 m / 43.5 deg
# (rotation swings with clip composition).  Thresholds are measured + headroom
# (position ~27%, rotation ~13% on the full set).  They gate "the smoke
# pipeline still learns", NOT product quality (see docs/status-claude.md for
# the numbers and the eval-on-training-clips caveat).


def cmd_synth(args: argparse.Namespace) -> int:
    paths = synth.synth_capture(
        args.data_root,
        space=args.space,
        n_clips=args.n_clips,
        joints=args.joints,
        seed=args.seed,
        with_joint_velocities=not args.no_joint_velocities,
    )
    print(json.dumps({"data_root": str(args.data_root), "clips": [str(p) for p in paths]}, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        hidden=args.hidden,
        latent_dim=args.latent_dim,
        depth=args.depth,
        codebook_size=args.codebook_size,
        log_every=args.log_every,
        max_clips=args.max_clips,
        eval_clips=args.eval_clips,
    )
    summary = train(args.data_root, args.out, cfg)
    print(json.dumps({
        "checkpoint": summary["checkpoint"],
        "manifest": summary["manifest"],
        "steps": cfg.steps,
        "final_report": {k: v for k, v in summary["report"].items() if k != "per_clip"},
        "history_tail": summary["history"][-3:],
    }, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    report, _model, _manifest = eval_mod.evaluate_checkpoint(
        args.checkpoint, data_root=args.data_root, max_clips=args.max_clips,
    )
    ok = metrics.within_thresholds(report, args.position_threshold, args.rotation_threshold)
    print(json.dumps({
        "checkpoint": str(args.checkpoint),
        "thresholds": {
            "position_mean_m": args.position_threshold,
            "rotation_mean_deg": args.rotation_threshold,
        },
        "within_thresholds": ok,
        **{k: v for k, v in report.items() if k != "per_clip"},
    }, indent=2))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_synth = sub.add_parser("synth", help="write seeded synthetic shards")
    p_synth.add_argument("--data-root", type=Path, required=True)
    p_synth.add_argument("--space", default="synth-space")
    p_synth.add_argument("--n-clips", type=int, default=6)
    p_synth.add_argument("--joints", type=int, default=4)
    p_synth.add_argument("--seed", type=int, default=0)
    p_synth.add_argument("--no-joint-velocities", action="store_true")
    p_synth.set_defaults(fn=cmd_synth)

    p_train = sub.add_parser("train", help="smoke-scale VQ-VAE training")
    p_train.add_argument("--data-root", type=Path, required=True)
    p_train.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints" / "smoke")
    p_train.add_argument("--steps", type=int, default=TrainConfig.steps)
    p_train.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p_train.add_argument("--lr", type=float, default=TrainConfig.lr)
    p_train.add_argument("--seed", type=int, default=TrainConfig.seed)
    p_train.add_argument("--hidden", type=int, default=TrainConfig.hidden)
    p_train.add_argument("--latent-dim", type=int, default=TrainConfig.latent_dim)
    p_train.add_argument("--depth", type=int, default=TrainConfig.depth)
    p_train.add_argument("--codebook-size", type=int, default=TrainConfig.codebook_size)
    p_train.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    p_train.add_argument("--max-clips", type=int, default=None)
    p_train.add_argument("--eval-clips", type=int, default=TrainConfig.eval_clips)
    p_train.set_defaults(fn=cmd_train)

    p_eval = sub.add_parser("eval", help="score a checkpoint against thresholds")
    p_eval.add_argument("--checkpoint", type=Path, required=True)
    p_eval.add_argument("--data-root", type=Path, default=None,
                        help="override the manifest's data root")
    p_eval.add_argument("--max-clips", type=int, default=None)
    p_eval.add_argument("--position-threshold", type=float, default=POSITION_THRESHOLD_M)
    p_eval.add_argument("--rotation-threshold", type=float, default=ROTATION_THRESHOLD_DEG)
    p_eval.set_defaults(fn=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
