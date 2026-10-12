"""Tiny end-to-end training loop: checkpoint roundtrip, manifest contract, gating."""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motionbricks_mlx.training import checkpoint as ckpt
from motionbricks_mlx.training import eval as eval_mod
from motionbricks_mlx.training import metrics, synth
from motionbricks_mlx.training.dataset import ClipDataset, collate
from motionbricks_mlx.training.train import TrainConfig, train
from motionbricks_mlx.training.vqvae import MotionVQVAE

TINY_CFG = dict(
    steps=12, batch_size=2, hidden=16, latent_dim=8, depth=2,
    codebook_size=16, eval_clips=2, log_every=4, seed=0,
)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    synth.synth_capture(root, space="sp", n_clips=4, joints=4, seed=5)
    out_dir = tmp_path_factory.mktemp("ckpt") / "run-000"
    summary = train(root, out_dir, TrainConfig(**TINY_CFG), log=lambda *_: None)
    return root, out_dir, summary


def test_tiny_run_learns_and_saves(run):
    root, out_dir, summary = run
    history = summary["history"]
    assert len(history) == TINY_CFG["steps"]
    losses = [h["loss"] for h in history]
    assert all(np.isfinite(losses))
    # the optimizer actually applies (this scaffold once silently froze here)
    assert np.mean(losses[-3:]) < np.mean(losses[:3])
    assert (out_dir / "checkpoint.npz").is_file()
    assert (out_dir / "manifest.json").is_file()
    ckpt.load_manifest(out_dir)  # validates the saved manifest in place


def test_checkpoint_reload_reproduces_outputs(run):
    root, out_dir, summary = run
    manifest = ckpt.load_manifest(out_dir)
    hp = manifest["hyperparams"]
    ds = ClipDataset(root)

    batch = collate(ds.clips(), downsample=hp["downsample"])
    out_a = summary["model"](batch.frames, batch.mask)
    mx.eval(out_a["recon"])

    reloaded = MotionVQVAE(frame_dim=hp["frame_dim"], hidden=hp["hidden"],
                           latent_dim=hp["latent_dim"], depth=hp["depth"],
                           codebook_size=hp["codebook_size"],
                           commitment=hp["commitment_beta"], ema_decay=hp["ema_decay"],
                           quantizer_seed=hp["quantizer_seed"] or manifest["seed"])
    ckpt.load_checkpoint(out_dir / "checkpoint.npz", reloaded)
    out_b = reloaded(batch.frames, batch.mask)
    mx.eval(out_b["recon"])
    assert mx.array_equal(out_a["recon"], out_b["recon"])
    assert mx.array_equal(out_a["indices"], out_b["indices"])


def test_manifest_negative_controls(run):
    root, out_dir, _summary = run
    manifest = ckpt.load_manifest(out_dir)  # the saved manifest passes validation
    for key in ("seed", "git_sha", "hyperparams", "data", "format"):
        broken = {k: v for k, v in manifest.items() if k != key}
        with pytest.raises(ckpt.ManifestError):
            ckpt.validate_manifest(broken)
    bad_format = {**manifest, "format": "someone/else-v9"}
    with pytest.raises(ckpt.ManifestError, match="format"):
        ckpt.validate_manifest(bad_format)
    no_ids = {**manifest, "data": {"clips": [{"path": "x", "sha256": ""}]}}
    with pytest.raises(ckpt.ManifestError, match="sha256"):
        ckpt.validate_manifest(no_ids)


def test_eval_checkpoint_and_threshold_gating(run):
    root, out_dir, _summary = run
    report, model, manifest = eval_mod.evaluate_checkpoint(out_dir, data_root=root)
    assert manifest["seed"] == TINY_CFG["seed"]
    assert report["n_clips"] == 4
    for key in ("position_mean_m", "position_p95_m", "rotation_mean_deg",
                "rotation_p95_deg", "frame_mse_all"):
        assert np.isfinite(report[key])
    assert len(report["per_clip"]) == report["n_clips"]
    # generous thresholds pass, zero thresholds fail (the gate actually gates)
    assert metrics.within_thresholds(report, position_m=1e6, rotation_deg=1e6)
    assert not metrics.within_thresholds(report, position_m=0.0, rotation_deg=0.0)


def test_max_clips_limits_dataset(run, tmp_path):
    root, _out, _summary = run
    ds = ClipDataset(root)
    from motionbricks_mlx.training.train import limit_dataset

    assert len(limit_dataset(ds, None)) == 4
    assert len(limit_dataset(ds, 2)) == 2
    empty = ClipDataset(tmp_path, require_nonempty=False)  # selects no clips -> red
    with pytest.raises(ValueError):
        limit_dataset(empty, 3)
