"""Synthetic clip shards that conform to the data contract.

The B-side scaffold must be developed against ``datasets/<space>/<capture>/clip.npz``
shards before the A line produces real ones (``docs/brief-claude-training.md``).
This module writes shards whose *shapes, block order and metadata* follow
``docs/motion-representation.md`` exactly; the *values* are random but
**smooth** (integrated random walks, phase-driven joint wobble), so a tiny
model can actually reduce reconstruction error during smoke training.

Block semantics honoured where cheap (velocity = smoothed finite difference of
root position, phase = sin/cos of a per-clip cycle); the optional
``joint_velocities`` block is a small smooth walk — real angular velocity has
no cheap 6D analogue and the scaffold only needs the layout to be right.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import layout

STYLE_SET: tuple[str, ...] = ("idle", "walk", "run", "dance")

_SMOOTH_KERNEL = 5  # frames; matches the contract's "smoothed 3-frame" spirit


def _smooth(x: np.ndarray) -> np.ndarray:
    kernel = np.ones(_SMOOTH_KERNEL, dtype=np.float64) / _SMOOTH_KERNEL
    pad = _SMOOTH_KERNEL // 2
    padded = np.pad(x, ((pad, pad), *(((0, 0),) * (x.ndim - 1))), mode="edge")
    out = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="valid"), 0, padded
    )
    return out.astype(np.float64)


def _smooth_walk(rng: np.random.Generator, t: int, dim: int, step_scale: float) -> np.ndarray:
    steps = rng.normal(0.0, step_scale, size=(t, dim))
    return _smooth(np.cumsum(steps, axis=0))


def random_clip(
    rng: np.random.Generator,
    joints: int,
    fps: int = layout.DEFAULT_FPS,
    with_joint_velocities: bool = True,
    min_seconds: float = 1.0,
    max_seconds: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """One random clip: ``(frames (T, D) float32, offsets, meta)``."""
    seconds = rng.uniform(min_seconds, max_seconds)
    t = max(int(round(seconds * fps)), 2)
    time = np.arange(t, dtype=np.float64)

    blocks_out: dict[str, np.ndarray] = {}
    pos = _smooth_walk(rng, t, 3, step_scale=0.02)
    blocks_out["root_position"] = pos
    blocks_out["root_yaw"] = _smooth_walk(rng, t, 1, step_scale=0.03)
    # Contract: velocity is the smoothed finite difference of root position.
    blocks_out["root_velocity"] = _smooth(np.gradient(pos, axis=0))

    # Per joint: a random orthonormal pair (first two columns of R), perturbed
    # per frame and re-orthonormalised.  Layout within the 6 is [u(3), v(3)].
    def _gram_schmidt(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = u / np.linalg.norm(u, axis=-1, keepdims=True)
        v = v - (u * v).sum(-1, keepdims=True) * u
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        return u, v

    u0, v0 = _gram_schmidt(rng.normal(size=(joints, 3)), rng.normal(size=(joints, 3)))
    amp = rng.uniform(0.02, 0.08, size=(joints, 1, 1))
    freq = rng.uniform(0.2, 1.0, size=(joints, 1, 1))
    ph = rng.uniform(0.0, 2 * np.pi, size=(joints, 1, 1))
    sig = np.sin(2 * np.pi * freq * time[None, :, None] + ph)  # (joints, t, 1)
    ut = u0[:, None, :] + amp * sig  # (joints, t, 3)
    vt = v0[:, None, :] + amp * np.sin(2 * np.pi * freq * time[None, :, None] + ph + 0.7)
    ut, vt = _gram_schmidt(ut, vt)  # broadcasts over (joints, t)
    rot6 = np.concatenate(
        [np.transpose(ut, (1, 0, 2)), np.transpose(vt, (1, 0, 2))], axis=-1
    ).reshape(t, joints * 6)
    blocks_out["joint_rotations"] = rot6
    if with_joint_velocities:
        blocks_out["joint_velocities"] = _smooth_walk(rng, t, 3 * joints, step_scale=0.01)

    blocks_out["contacts"] = (_smooth_walk(rng, t, 2, step_scale=0.5) > 0.0).astype(np.float64)
    cycle = rng.uniform(0.5, 2.0) * fps  # frames per cycle
    ang = 2 * np.pi * time / cycle
    blocks_out["phase"] = np.stack([np.sin(ang), np.cos(ang)], axis=-1)

    order = layout.blocks(joints, with_joint_velocities)
    frames = np.concatenate([blocks_out[b.name] for b in order], axis=-1)
    frames = frames.astype(np.float32)
    table = layout.offsets(joints, with_joint_velocities)

    start = [float(pos[0, 0]), float(pos[0, 1]), float(pos[0, 2]), float(blocks_out["root_yaw"][0, 0])]
    end = [float(pos[-1, 0]), float(pos[-1, 1]), float(pos[-1, 2]), float(blocks_out["root_yaw"][-1, 0])]
    meta = {
        "style": str(rng.choice(STYLE_SET)),
        "target": [float(v) for v in rng.normal(0.0, 2.0, size=3)],
        "start_root": start,
        "end_root": end,
        "fps": int(fps),
        "skeleton_id": f"synth-skel-{joints}j",
        "provenance": {"space": "synthetic", "capture": "random", "git_sha": "synthetic"},
    }
    return frames, table, meta


def write_clip(directory: Path, frames: np.ndarray, offsets_table: np.ndarray, meta: dict) -> Path:
    """Write one shard: ``clip.npz`` (frames, offsets) plus a sibling ``meta.json``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "clip.npz", frames=frames.astype(np.float32),
             offsets=np.asarray(offsets_table, dtype=np.int64))
    (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return directory / "clip.npz"


def synth_capture(
    root: Path,
    space: str = "synth-space",
    capture_prefix: str = "cap",
    n_clips: int = 4,
    joints: int = 4,
    seed: int = 0,
    with_joint_velocities: bool = True,
) -> list[Path]:
    """Write ``n_clips`` shards under ``root/space/<capture_prefix>-NNN/``."""
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n_clips):
        frames, table, meta = random_clip(rng, joints, with_joint_velocities=with_joint_velocities)
        meta["provenance"] = {"space": space, "capture": f"{capture_prefix}-{i:03d}",
                              "git_sha": "synthetic", "seed": seed}
        paths.append(
            write_clip(Path(root) / space / f"{capture_prefix}-{i:03d}", frames, table, meta)
        )
    return paths
