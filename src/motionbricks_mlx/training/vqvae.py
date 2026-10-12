"""MLX motion VQ-VAE for the contract frame layout (B-side scaffold).

Architecture: 1-D conv encoder/decoder over time (MLX convs are channels-last,
``(B, T, C)``), each stride-2 block halving the temporal rate, so the latent
runs at ``T / 2**depth``.  A vector quantizer with an **EMA codebook** sits in
the middle.

Quantizer choice — EMA codebook + commitment loss, not gradient codebook loss
(the work ticket asks for the choice and its justification):

1. EMA updates codebook vectors from encoding statistics instead of gradients,
   so there is no extra loss term to balance; that balancing is the usual
   cause of codebook collapse, and our smoke budget is tens of steps at tiny
   batch sizes where it bites hardest.
2. It is the standard recipe (van den Oord et al. 2017; VQ-VAE-2/DALL-E keep
   it) and its state updates are plain array ops — trivially reproducible
   from (seed, hyperparams), which the manifest contract requires.
3. The commitment term (``beta * ||sg(e) - z||^2``) is kept because with an
   EMA codebook nothing else pulls encoder outputs toward their codes.

Known scaffold limitation (documented, not hidden): with stride-2 convs the
receptive field of a real frame near a padded tail includes zero-padding, so
those real frames reconstruct slightly worse.  Batch-level masking keeps
padded frames out of the loss and out of the EMA statistics; per-sequence
batching or overlap windows are the later fix and are tracked in
``docs/status-claude.md``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import layout

__all__ = [
    "ConvBlock",
    "Encoder",
    "Decoder",
    "EMAQuantizer",
    "MotionVQVAE",
    "seed_parameters",
    "vqvae_loss",
]


class ConvBlock(nn.Module):
    """stride-2 temporal conv + SiLU + LayerNorm (channels-last)."""

    def __init__(self, channels: int, kernel: int = 4, upsample: bool = False):
        super().__init__()
        if upsample:
            # k=4, s=2, p=1 doubles the length exactly (probe-verified).
            self.conv = nn.ConvTranspose1d(channels, channels, kernel, stride=2, padding=1)
        else:
            self.conv = nn.Conv1d(channels, channels, kernel, stride=2, padding=1)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(channels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(self.act(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, frame_dim: int, hidden: int, latent_dim: int, depth: int):
        super().__init__()
        self.in_proj = nn.Linear(frame_dim, hidden)
        self.blocks = [ConvBlock(hidden) for _ in range(depth)]
        self.out_proj = nn.Linear(hidden, latent_dim)
        self.act = nn.SiLU()

    def __call__(self, x: mx.array) -> mx.array:
        """``(B, T, D)`` -> ``(B, T / 2**depth, latent_dim)``."""
        h = self.act(self.in_proj(x))
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)


class Decoder(nn.Module):
    def __init__(self, frame_dim: int, hidden: int, latent_dim: int, depth: int):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, hidden)
        self.blocks = [ConvBlock(hidden, upsample=True) for _ in range(depth)]
        self.out_proj = nn.Linear(hidden, frame_dim)
        self.act = nn.SiLU()

    def __call__(self, z: mx.array) -> mx.array:
        """``(B, T', latent_dim)`` -> ``(B, T' * 2**depth, D)``."""
        h = self.act(self.in_proj(z))
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)


class EMAQuantizer(nn.Module):
    """Nearest-neighbour VQ with EMA codebook updates (pure-function update).

    The forward pass never mutates state: it returns the straight-through
    quantization, commitment loss, perplexity **and** the new EMA state; the
    caller evaluates it and calls :meth:`update_state` outside the gradient
    graph.  State arrays are frozen so neither the optimizer nor
    ``value_and_grad`` touches them (probe-verified on MLX 0.32.3).
    """

    def __init__(
        self,
        codebook_size: int,
        latent_dim: int,
        commitment: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        seed: int = 0,
    ):
        super().__init__()
        if codebook_size < 1 or latent_dim < 1:
            raise ValueError("codebook_size and latent_dim must be >= 1")
        rng = np.random.default_rng(seed)
        init = rng.uniform(-1.0 / codebook_size, 1.0 / codebook_size,
                           size=(codebook_size, latent_dim)).astype(np.float32)
        self.codebook = mx.array(init)
        self.embed_avg = mx.array(init)
        self.cluster_size = mx.zeros((codebook_size,), dtype=mx.float32)
        self.codebook_size = int(codebook_size)
        self.latent_dim = int(latent_dim)
        self.commitment = float(commitment)
        self.decay = float(decay)
        self.eps = float(eps)
        self.freeze()  # EMA state is updated manually, never by the optimizer

    def __call__(self, z: mx.array, weights: mx.array) -> dict:
        """``z (N, C)``, ``weights (N,)`` in [0, 1] (frame mask, flattened)."""
        wf = weights.astype(z.dtype).reshape(-1)
        cb = self.codebook  # (K, C)
        dist = (z * z).sum(-1, keepdims=True) - 2.0 * (z @ cb.T) + (cb * cb).sum(-1)
        indices = mx.argmin(dist, axis=-1)  # (N,)
        onehot = mx.eye(self.codebook_size, dtype=z.dtype)[indices]  # (N, K)

        # Pure EMA update; caller evals + assigns these via update_state().
        counts = onehot.T @ wf  # (K,)
        sums = onehot.T @ (z * wf[:, None])  # (K, C)
        cluster = self.cluster_size * self.decay + counts * (1.0 - self.decay)
        embed = self.embed_avg * self.decay + sums * (1.0 - self.decay)
        total = cluster.sum()
        laplace = (cluster + self.eps) / (total + self.codebook_size * self.eps) * total
        new_codebook = embed / laplace[:, None]

        q = onehot @ self.codebook  # (N, C) with the pre-update codebook
        commit = ((mx.stop_gradient(q) - z) ** 2).sum(-1) / self.latent_dim
        commit = (commit * wf).sum() / (wf.sum() + 1e-8)
        # Straight-through: gradients bypass the quantizer into the encoder.
        q_st = z + mx.stop_gradient(q - z)

        probs = (counts + 1e-8) / (counts.sum() + 1e-8)
        perplexity = mx.exp(-(probs * mx.log(probs)).sum())
        return {
            "q_st": q_st,
            "indices": indices,
            "commit": commit,
            "perplexity": perplexity,
            "state": (new_codebook, cluster, embed),
        }

    def update_state(self, state: tuple[mx.array, mx.array, mx.array]) -> None:
        new_codebook, cluster, embed = state
        self.codebook = new_codebook
        self.cluster_size = cluster
        self.embed_avg = embed

    def seed_from_encodings(self, z: mx.array, weights: mx.array) -> None:
        """First-batch codebook init (call once, before training).

        Without it the U(-1/K, 1/K) init sits orders of magnitude below the
        encoder output scale, and the commitment loss drags the encoder toward
        the near-zero codes — the latent collapses and reconstruction is worse
        than the zero baseline (observed on this very scaffold, see
        ``docs/status-claude.md``).  Seeding the codes from real encodings is
        the standard EMA-VQ remedy; it also removes the init-scale term from
        the first-step gradient explosion.
        """
        wf = weights.astype(z.dtype).reshape(-1)
        n_real = int(wf.sum().item())
        if n_real <= 0:
            raise ValueError("seed_from_encodings: batch has no real frames")
        real = z[mx.argsort(-wf)[:n_real]]  # drop padded encodings
        reps = math.ceil(self.codebook_size / n_real)
        tiled = mx.concatenate([real] * reps, axis=0)[: self.codebook_size]
        self.codebook = tiled
        self.embed_avg = tiled
        self.cluster_size = mx.ones((self.codebook_size,), dtype=z.dtype)
        mx.eval(self.codebook, self.embed_avg, self.cluster_size)


class MotionVQVAE(nn.Module):
    """Encoder -> EMA-VQ -> Decoder over the contract frame vector."""

    def __init__(
        self,
        frame_dim: int,
        hidden: int = 64,
        latent_dim: int = 32,
        depth: int = 2,
        codebook_size: int = 64,
        commitment: float = 0.25,
        ema_decay: float = 0.99,
        quantizer_seed: int = 0,
    ):
        super().__init__()
        if frame_dim < 1 or hidden < 1 or latent_dim < 1 or depth < 1:
            raise ValueError("frame_dim/hidden/latent_dim/depth must be >= 1")
        self.frame_dim = int(frame_dim)
        self.hidden = int(hidden)
        self.latent_dim = int(latent_dim)
        self.depth = int(depth)
        self.downsample = 2 ** int(depth)
        self.encoder = Encoder(frame_dim, hidden, latent_dim, depth)
        self.quantizer = EMAQuantizer(
            codebook_size, latent_dim, commitment=commitment, decay=ema_decay,
            seed=quantizer_seed,
        )
        self.decoder = Decoder(frame_dim, hidden, latent_dim, depth)

    def __call__(self, frames: mx.array, mask: mx.array) -> dict:
        """``frames (B, T, D)``, ``mask (B, T)`` bool; T must be divisible by
        ``downsample`` (see ``dataset.collate``)."""
        b, t, _ = frames.shape
        if t % self.downsample != 0:
            raise ValueError(
                f"T={t} not divisible by downsample={self.downsample}; use collate(downsample=...)"
            )
        m = mask[:, :: self.downsample]  # block-aligned: T = T' * downsample
        enc = self.encoder(frames)  # (B, T', C)
        q = self.quantizer(enc.reshape(-1, self.latent_dim), m.reshape(-1))
        latent = q["q_st"].reshape(b, -1, self.latent_dim)
        recon = self.decoder(latent)  # (B, T, D)
        return {
            "recon": recon,
            "indices": q["indices"],
            "commit": q["commit"],
            "perplexity": q["perplexity"],
            "mask_t": m,
            "state": q["state"],
        }


def seed_parameters(module: nn.Module, rng: np.random.Generator) -> None:
    """Deterministic N(0, 1/sqrt(fan_in)) init for weights, zeros for biases.

    Fan-in is the last axis: MLX Conv1d weights are ``(out, kernel, in)`` and
    Linear weights ``(out, in)``, both channels/in-features last.  LayerNorm
    rows (1-D weight/bias) keep their default ones/zeros init.
    """
    fresh = []
    for path, arr in nn.utils.tree_flatten(module.parameters()):
        if path.endswith(".weight") and arr.ndim >= 2:
            scale = 1.0 / math.sqrt(int(arr.shape[-1]))
            fresh.append((path, mx.array(rng.normal(0.0, scale, size=arr.shape).astype(np.float32))))
        elif path.endswith(".bias"):
            fresh.append((path, mx.zeros(arr.shape, dtype=arr.dtype)))
    module.update(nn.utils.tree_unflatten(fresh))


def vqvae_loss(model: MotionVQVAE, batch, commitment_beta: float = 0.25) -> tuple[mx.array, dict]:
    """Masked reconstruction MSE + ``beta`` * commitment; returns (loss, aux)."""
    out = model(batch.frames, batch.mask)
    m = batch.mask[..., None].astype(out["recon"].dtype)
    dims = int(batch.frames.shape[-1])
    se = (out["recon"] - batch.frames) ** 2 * m
    recon = se.sum() / (m.sum() * dims + 1e-8)
    loss = recon + commitment_beta * out["commit"]
    aux = {
        "recon": recon,
        "commit": out["commit"],
        "perplexity": out["perplexity"],
        "state": out["state"],
        "indices": out["indices"],
        "mask_t": out["mask_t"],
    }
    return loss, aux


def clip_gradients(grads: dict, max_norm: float) -> tuple[dict, mx.array]:
    """Global-norm gradient clipping (this MLX version has no util for it)."""
    leaves = nn.utils.tree_flatten(grads)
    sq = mx.zeros((), dtype=mx.float32)
    for _, g in leaves:
        sq = sq + (g.astype(mx.float32) ** 2).sum()
    total = mx.sqrt(sq)
    scale = mx.minimum(mx.array(1.0), mx.array(max_norm) / (total + 1e-12))
    clipped = [(k, g * scale) for k, g in leaves]
    return nn.utils.tree_unflatten(clipped), total
