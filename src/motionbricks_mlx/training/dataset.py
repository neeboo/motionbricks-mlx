"""Dataset reader for contract shards, with variable-length batching.

Reads ``<root>/<space>/<capture>/clip.npz`` (``frames (T, D)`` float32,
``offsets`` int table) plus the sibling ``meta.json``.  Layout and metadata are
validated against :mod:`.layout` (transcribed from the contract), so a broken
shard fails loudly instead of poisoning training.

``collate`` pads a batch to the longest clip and emits a boolean mask; the
padded tail is also truncated so its length is a multiple of the model's
temporal downsample factor (block-aligned convolutions require it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import layout


@dataclass(frozen=True)
class Clip:
    """One validated shard, fully materialised (smoke-scale datasets only)."""

    path: Path
    space: str
    capture: str
    frames: np.ndarray  # (T, D) float32
    offsets: np.ndarray  # int64 table, layout.offsets()-shaped
    meta: dict

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def frame_dim(self) -> int:
        return int(self.frames.shape[1])


def load_clip(path: Path) -> Clip:
    """Load and validate one shard; raises ``LayoutError`` on any violation."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as npz:
        missing = {"frames", "offsets"} - set(npz.files)
        if missing:
            raise layout.LayoutError(f"{path}: npz is missing arrays {sorted(missing)}")
        frames = np.asarray(npz["frames"], dtype=np.float32)
        offsets_table = np.asarray(npz["offsets"], dtype=np.int64)

    if frames.ndim != 2:
        raise layout.LayoutError(f"{path}: frames must be (T, D), got {frames.shape}")
    t, d = frames.shape
    if t < 1:
        raise layout.LayoutError(f"{path}: clip has no frames")
    layout.blocks_from_offsets(offsets_table)  # validates table shape/monotonicity/v0 layout
    if d != int(offsets_table[-1]):
        raise layout.LayoutError(
            f"{path}: frames has D={d} but offsets table ends at {int(offsets_table[-1])}"
        )
    if not np.isfinite(frames).all():
        raise layout.LayoutError(f"{path}: frames contain non-finite values")

    meta_path = path.parent / "meta.json"
    if not meta_path.is_file():
        raise layout.LayoutError(f"{path}: sibling meta.json not found at {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    layout.validate_meta(meta)

    parts = path.resolve().parts
    return Clip(
        path=path,
        space=parts[-3] if len(parts) >= 3 else "",
        capture=parts[-2] if len(parts) >= 2 else "",
        frames=frames,
        offsets=offsets_table,
        meta=meta,
    )


class ClipDataset:
    """All shards under ``root/<space>/<capture>/clip.npz``, discovered once."""

    def __init__(self, root: Path, require_nonempty: bool = True):
        self.root = Path(root)
        self.paths = sorted(self.root.glob("*/*/clip.npz"))
        if require_nonempty and not self.paths:
            raise FileNotFoundError(f"no clip.npz found under {self.root}/*/*/clip.npz")
        self._cache: dict[Path, Clip] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Clip:
        path = self.paths[index]
        if path not in self._cache:
            self._cache[path] = load_clip(path)
        return self._cache[path]

    def clips(self) -> list[Clip]:
        return [self[i] for i in range(len(self))]


@dataclass(frozen=True)
class Batch:
    frames: mx.array  # (B, T_max, D) float32, zero-padded tail
    mask: mx.array  # (B, T_max) bool, True on real frames
    lengths: tuple[int, ...] = field(default_factory=tuple)

    @property
    def batch_size(self) -> int:
        return int(self.frames.shape[0])


def collate(clips: list[Clip], downsample: int = 1, pad_value: float = 0.0) -> Batch:
    """Pad/truncate a list of clips into one fixed-shape batch.

    Truncates the *padded* length to a multiple of ``downsample`` (model input
    requirement for stride-2 conv stacks) and clips longer sequences to it,
    keeping the head of each clip (``start_root`` metadata refers to frame 0).
    """
    if not clips:
        raise ValueError("collate needs at least one clip")
    if downsample < 1:
        raise ValueError(f"downsample must be >= 1, got {downsample}")
    frame_dim = clips[0].frame_dim
    for clip in clips:
        if clip.frame_dim != frame_dim:
            raise layout.LayoutError(
                f"mixed frame dims in one batch: {clip.frame_dim} vs {frame_dim} ({clip.path})"
            )

    t_max = max(clip.n_frames for clip in clips)
    if downsample > 1:
        t_max = (t_max // downsample) * downsample
        if t_max == 0:
            raise ValueError(
                f"longest clip has {max(c.n_frames for c in clips)} frames, shorter than "
                f"one downsample block ({downsample})"
            )

    b = len(clips)
    frames = np.full((b, t_max, frame_dim), pad_value, dtype=np.float32)
    mask = np.zeros((b, t_max), dtype=bool)
    lengths: list[int] = []
    for i, clip in enumerate(clips):
        take = min(clip.n_frames, t_max)
        frames[i, :take] = clip.frames[:take]
        mask[i, :take] = True
        lengths.append(take)
    return Batch(
        frames=mx.array(frames),
        mask=mx.array(mask, dtype=mx.bool_),
        lengths=tuple(lengths),
    )
