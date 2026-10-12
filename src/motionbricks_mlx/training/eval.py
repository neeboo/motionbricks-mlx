"""Reconstruction evaluation: per-position / per-rotation metrics + gating.

``evaluate_model`` scores a model on clips; ``evaluate_checkpoint`` rebuilds
the model from a run's ``manifest.json`` (hyperparams + weights) so a stored
run can be re-scored without the training code path.  Thresholds are **not**
hard-coded here: the CLI in ``scripts/train_vqvae.py`` owns the defaults (the
ticket puts them in the script) and exposes them as flags.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from . import checkpoint as ckpt
from . import metrics
from .dataset import Clip, ClipDataset, collate
from .vqvae import MotionVQVAE


def evaluate_model(model: MotionVQVAE, clips: list[Clip], downsample: int | None = None) -> dict:
    """Per-clip reconstruction reports + a frame-weighted aggregate."""
    if not clips:
        raise ValueError("no clips to evaluate")
    downsample = downsample or model.downsample
    reports = []
    for clip in clips:
        batch = collate([clip], downsample=downsample)
        out = model(batch.frames, batch.mask)
        mx.eval(out["recon"])
        t_used = int(batch.frames.shape[1])
        pred = np.array(out["recon"][0])
        target = np.array(batch.frames[0])
        mask = np.array(batch.mask[0], dtype=bool)
        reports.append(metrics.reconstruction_report(pred, target, clip.offsets, mask))
    aggregate = metrics.aggregate_reports(reports)
    return {**aggregate, "per_clip": reports}


def evaluate_checkpoint(
    run_dir: Path,
    data_root: Path | None = None,
    max_clips: int | None = None,
) -> tuple[dict, MotionVQVAE, dict]:
    """Rebuild a run from its manifest and score it; returns (report, model, manifest)."""
    run_dir = Path(run_dir)
    manifest = ckpt.load_manifest(run_dir)  # validates reproducibility keys
    hp = dict(manifest["hyperparams"])
    from .train import TrainConfig  # deferred: train.py imports this module

    cfg = TrainConfig(
        seed=manifest["seed"],
        hidden=hp["hidden"],
        latent_dim=hp["latent_dim"],
        depth=hp["depth"],
        codebook_size=hp["codebook_size"],
        commitment_beta=hp["commitment_beta"],
        ema_decay=hp["ema_decay"],
        quantizer_seed=hp.get("quantizer_seed"),
    )
    model = MotionVQVAE(frame_dim=hp["frame_dim"], **cfg.model_kwargs())
    ckpt.load_checkpoint(run_dir / ckpt.CHECKPOINT_NAME, model)

    root = Path(data_root) if data_root is not None else Path(manifest["data"]["root"])
    dataset = ClipDataset(root)
    if max_clips is not None:
        from .train import limit_dataset  # deferred: train.py imports this module

        dataset = limit_dataset(dataset, max_clips)
    report = evaluate_model(model, dataset.clips(), downsample=hp["downsample"])
    return report, model, manifest
