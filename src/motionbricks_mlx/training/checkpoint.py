"""Checkpoint format + manifest (the README reproducibility contract).

A run directory contains:

* ``checkpoint.npz`` — every model parameter (including the quantizer's EMA
  state; a lost codebook is a lost model), keys are dotted paths with ``.``
  replaced by ``__`` for the npz container.
* ``manifest.json`` — ``git sha + seed + hyperparams + data ids`` per the
  README ("缺一项就不算可复现"), plus the final metrics.  ``validate_manifest``
  enforces exactly that so a test can go red when a field goes missing.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

CHECKPOINT_FORMAT = "motionbrick-mlx/ckpt-v0"
MANIFEST_NAME = "manifest.json"
CHECKPOINT_NAME = "checkpoint.npz"

#: Keys without which a run is not reproducible (README, 提交纪律).
REQUIRED_MANIFEST_KEYS: tuple[str, ...] = ("format", "git_sha", "seed", "hyperparams", "data")

_REPO_ROOT = Path(__file__).resolve().parents[3]


class ManifestError(ValueError):
    """A manifest does not carry the reproducibility contract."""


def git_state() -> tuple[str, bool]:
    """``(sha, dirty)`` of the repo this module lives in; 'unknown' if not a repo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=_REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip() != ""
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(
    out_dir: Path,
    model: nn.Module,
    dataset,
    seed: int,
    hyperparams: dict,
    metrics: dict,
) -> dict:
    """Write ``checkpoint.npz`` + ``manifest.json``; returns their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = nn.utils.tree_flatten(model.parameters())
    arrays = {path.replace(".", "__"): arr for path, arr in flat}
    ckpt_path = out_dir / CHECKPOINT_NAME
    mx.savez(str(ckpt_path), **arrays)

    sha, dirty = git_state()
    manifest = {
        "format": CHECKPOINT_FORMAT,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": sha,
        "git_dirty": dirty,
        "seed": int(seed),
        "hyperparams": hyperparams,
        "metrics": metrics,
        "data": {
            "root": str(dataset.root),
            "n_clips": len(dataset),
            "clips": [
                {
                    "path": str(clip.path.relative_to(dataset.root)),
                    "sha256": sha256_file(clip.path),
                    "n_frames": clip.n_frames,
                    "frame_dim": clip.frame_dim,
                    "offsets": [int(v) for v in clip.offsets],
                }
                for clip in dataset.clips()
            ],
        },
    }
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"checkpoint": ckpt_path, "manifest": manifest_path, "manifest_dict": manifest}


def load_checkpoint(ckpt_path: Path, model: nn.Module) -> None:
    """Load every parameter from ``checkpoint.npz`` into ``model`` in place."""
    loaded = mx.load(str(ckpt_path))
    weights = {key.replace("__", "."): value for key, value in loaded.items()}
    if not weights:
        raise ValueError(f"{ckpt_path}: no arrays found")
    model.update(nn.utils.tree_unflatten(sorted(weights.items())))


def validate_manifest(manifest: dict) -> None:
    """Raise :class:`ManifestError` unless the reproducibility contract holds."""
    if not isinstance(manifest, dict):
        raise ManifestError("manifest must be a JSON object")
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise ManifestError(f"manifest is missing required keys: {missing}")
    if manifest["format"] != CHECKPOINT_FORMAT:
        raise ManifestError(f"unsupported checkpoint format {manifest['format']!r}")
    if manifest["seed"] is None:
        raise ManifestError("manifest seed is null")
    if not isinstance(manifest["hyperparams"], dict) or not manifest["hyperparams"]:
        raise ManifestError("manifest hyperparams must be a non-empty object")
    data = manifest["data"]
    if not isinstance(data, dict) or not data.get("clips"):
        raise ManifestError("manifest data.clips must list at least one clip with an id")
    for clip in data["clips"]:
        if not clip.get("sha256"):
            raise ManifestError(f"clip {clip.get('path')!r} has no sha256 (data id)")


def load_manifest(run_dir: Path) -> dict:
    manifest = json.loads((Path(run_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest
