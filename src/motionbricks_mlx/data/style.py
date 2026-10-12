"""``.mbstyle`` assets: the reference's own GGUF style bundles.

Port of ``load_style_file`` (``src/style.cpp:67-157``).  A style is a GGUF v3
file with ``motionbricks.component = "style"`` and exactly five tensors; the
frame loop the agent replays lives in ``global_joint_positions`` /
``global_joint_rotations`` / ``global_root_positions`` / ``global_headings``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gguf import GGML_TYPE_F32, GGML_TYPE_I32, GgufFile

UPSTREAM_REVISION = "a0732b642c0333077e127a2f56ab0014c196b4a"  # see schema notes


@dataclass
class Style:
    name: str
    speed: float
    frames: int
    global_joint_positions: np.ndarray   # (frames, 34, 3)
    global_joint_rotations: np.ndarray   # (frames, 34, 3, 3) row-major
    global_root_positions: np.ndarray    # (frames, 3)
    global_headings: np.ndarray          # (frames,)
    allowed_tokens: np.ndarray           # (11,) uint8


def load_style(path: str | Path) -> Style:
    path = Path(path)
    with GgufFile(path) as gguf:
        metadata = gguf.metadata
        if metadata.get("general.architecture") != "motionbricks":
            raise ValueError(f"{path}: not a MotionBricks asset")
        if metadata.get("motionbricks.component") != "style":
            raise ValueError(f"{path}: not a style component")
        if metadata.get("motionbricks.skeleton") != "g1skel34":
            raise ValueError(f"{path}: unexpected skeleton")
        name = metadata["motionbricks.style_name"]
        speed = float(metadata["motionbricks.style_speed"])
        frames = int(metadata["motionbricks.style_frames"])
        if frames < 4 or gguf.info("global_joint_positions").type_id != GGML_TYPE_F32:
            raise ValueError(f"{path}: invalid style payload")
        positions = gguf.tensor("global_joint_positions").reshape(frames, 34, 3)
        rotations = gguf.tensor("global_joint_rotations").reshape(frames, 34, 3, 3)
        roots = gguf.tensor("global_root_positions").reshape(frames, 3)
        headings = gguf.tensor("global_headings").reshape(frames)
        allowed = gguf.tensor("allowed_tokens")
        if gguf.info("allowed_tokens").type_id != GGML_TYPE_I32:
            raise ValueError(f"{path}: allowed_tokens must be I32")
    allowed = allowed.reshape(11).astype(np.uint8)
    if not np.isin(allowed, (0, 1)).all():
        raise ValueError(f"{path}: allowed_tokens is not binary")
    for array in (positions, rotations, roots, headings):
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: style contains non-finite values")
    return Style(name, speed, frames,
                 np.ascontiguousarray(positions, dtype=np.float32),
                 np.ascontiguousarray(rotations, dtype=np.float32),
                 np.ascontiguousarray(roots, dtype=np.float32),
                 np.ascontiguousarray(headings, dtype=np.float32),
                 allowed)
