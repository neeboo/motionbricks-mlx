"""Training loop for the motion VQ-VAE (tiny-scale smoke by design).

The work ticket forbids long training; ``TrainConfig`` defaults are a
few-dozen-step smoke configuration.  Everything is seeded through one numpy
generator (data sampling, parameter init) plus the quantizer seed, so a run
is reproducible from ``manifest.json`` alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np

from . import checkpoint as ckpt
from . import eval as eval_mod
from .dataset import Batch, ClipDataset, collate
from .vqvae import MotionVQVAE, clip_gradients, seed_parameters, vqvae_loss


@dataclass
class TrainConfig:
    steps: int = 40
    batch_size: int = 4
    lr: float = 1e-3
    seed: int = 0
    hidden: int = 64
    latent_dim: int = 32
    depth: int = 2
    codebook_size: int = 64
    commitment_beta: float = 0.25
    ema_decay: float = 0.99
    quantizer_seed: int | None = None  # None -> reuse cfg.seed
    grad_clip: float = 1.0
    log_every: int = 10
    max_clips: int | None = None
    eval_clips: int = 4

    def model_kwargs(self) -> dict:
        return {
            "hidden": self.hidden,
            "latent_dim": self.latent_dim,
            "depth": self.depth,
            "codebook_size": self.codebook_size,
            "commitment": self.commitment_beta,
            "ema_decay": self.ema_decay,
            "quantizer_seed": self.seed if self.quantizer_seed is None else self.quantizer_seed,
        }


def limit_dataset(dataset: ClipDataset, max_clips: int | None) -> ClipDataset:
    if max_clips is not None and max_clips < 1:
        raise ValueError(f"max_clips must be >= 1, got {max_clips}")
    if max_clips is None or max_clips >= len(dataset):
        if not dataset.paths:
            raise ValueError(f"dataset at {dataset.root} has no clips")
        return dataset
    limited = ClipDataset(dataset.root, require_nonempty=False)
    limited.paths = sorted(dataset.paths)[:max_clips]
    if not limited.paths:
        raise ValueError(f"max_clips={max_clips} selected no clips")
    return limited


def _sample_batch(rng: np.random.Generator, dataset: ClipDataset, batch_size: int,
                  downsample: int) -> Batch:
    replace = len(dataset) < batch_size
    idx = rng.choice(len(dataset), size=min(batch_size, len(dataset)), replace=replace)
    return collate([dataset[int(i)] for i in idx], downsample=downsample)


def train(
    data_root: Path,
    out_dir: Path,
    cfg: TrainConfig,
    log=print,
    dataset: ClipDataset | None = None,
) -> dict:
    """Run a smoke-scale training; returns summary with paths + final metrics."""
    mx.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dataset = limit_dataset(dataset or ClipDataset(data_root), cfg.max_clips)
    first = dataset[0]
    offsets_joints, with_jv = _infer_layout(first)
    frame_dim = first.frame_dim
    log(f"dataset: {len(dataset)} clips, frame_dim={frame_dim}, joints={offsets_joints}, "
        f"joint_velocities={with_jv}")

    model = MotionVQVAE(frame_dim=frame_dim, **cfg.model_kwargs())
    seed_parameters(model.encoder, rng)
    seed_parameters(model.decoder, rng)

    # One warmup batch: seed the EMA codebook from real encoding statistics
    # (see EMAQuantizer.seed_from_encodings for why this is not optional).
    warmup = _sample_batch(rng, dataset, cfg.batch_size, model.downsample)
    enc0 = model.encoder(warmup.frames)
    warmup_mask = warmup.mask[:, :: model.downsample]
    model.quantizer.seed_from_encodings(
        enc0.reshape(-1, model.latent_dim), warmup_mask.reshape(-1)
    )
    mx.eval(enc0)
    log(f"codebook seeded from first batch ({cfg.batch_size} clips)")

    optimizer = opt.AdamW(learning_rate=cfg.lr)
    loss_and_grad = nn.value_and_grad(model, vqvae_loss)

    history: list[dict] = []
    for step in range(cfg.steps):
        batch = _sample_batch(rng, dataset, cfg.batch_size, model.downsample)
        (loss, aux), grads = loss_and_grad(model, batch, cfg.commitment_beta)
        grads, grad_norm = clip_gradients(grads, cfg.grad_clip)
        # MLX optimizer semantics (probe-verified on 0.32.3): update(dict,
        # grads) mutates the *dict* in place and returns None; the module only
        # sees the change after an explicit write-back.  Passing the trainable
        # subset also keeps AdamW weight decay off the frozen EMA state.
        trainable = model.trainable_parameters()
        optimizer.update(trainable, grads)
        model.update(trainable)
        model.quantizer.update_state(aux["state"])
        mx.eval(model.parameters(), loss, aux["recon"], aux["commit"], aux["perplexity"])
        entry = {
            "step": step,
            "loss": float(loss.item()),
            "recon": float(aux["recon"].item()),
            "commit": float(aux["commit"].item()),
            "perplexity": float(aux["perplexity"].item()),
            "grad_norm": float(grad_norm.item()),
        }
        history.append(entry)
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            log(f"step {entry['step']:4d}  loss {entry['loss']:.5f}  recon {entry['recon']:.5f}  "
                f"commit {entry['commit']:.5f}  ppl {entry['perplexity']:.1f}  "
                f"|g| {entry['grad_norm']:.3f}")

    eval_clips = limit_dataset(dataset, cfg.eval_clips)
    report = eval_mod.evaluate_model(model, eval_clips, downsample=model.downsample)
    log(f"final recon: position {report['position_mean_m']:.4f} m, "
        f"rotation {report['rotation_mean_deg']:.2f} deg over {report['n_clips']} clips")

    hyperparams = asdict(cfg)
    hyperparams.update({
        "frame_dim": frame_dim,
        "joints": offsets_joints,
        "with_joint_velocities": with_jv,
        "downsample": model.downsample,
        "optimizer": "AdamW",
    })
    paths = ckpt.save_checkpoint(
        out_dir, model, dataset, seed=cfg.seed, hyperparams=hyperparams, metrics=report,
    )
    log(f"checkpoint: {paths['checkpoint']}")
    return {
        "model": model,
        "history": history,
        "report": report,
        "checkpoint": str(paths["checkpoint"]),
        "manifest": str(paths["manifest"]),
    }


def _infer_layout(clip) -> tuple[int, bool]:
    from . import layout

    return layout.infer_joints(clip.offsets)
