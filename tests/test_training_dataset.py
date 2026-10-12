"""Dataset reader + collate: round-trip, variable lengths, and negative controls."""
import dataclasses
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motionbricks_mlx.training import dataset, layout, synth
from motionbricks_mlx.training.vqvae import MotionVQVAE, vqvae_loss


@pytest.fixture()
def synth_root(tmp_path):
    paths = synth.synth_capture(tmp_path, space="sp", n_clips=3, joints=4, seed=7)
    return tmp_path, paths


def test_write_read_roundtrip(synth_root):
    root, paths = synth_root
    ds = dataset.ClipDataset(root)
    assert len(ds) == 3
    clip = ds[0]
    assert clip.path == paths[0]
    assert clip.space == "sp" and clip.capture == "cap-000"
    with np.load(paths[0]) as npz:
        assert np.allclose(clip.frames, npz["frames"])
        assert np.array_equal(clip.offsets, npz["offsets"])
    assert clip.frames.dtype == np.float32
    assert clip.frames.shape[1] == int(clip.offsets[-1])
    layout.validate_meta(clip.meta)


def test_variable_length_collate_and_mask(synth_root):
    root, _ = synth_root
    ds = dataset.ClipDataset(root)
    lengths = [c.n_frames for c in ds.clips()]
    assert len(set(lengths)) > 1  # fixture really is variable-length
    batch = dataset.collate(ds.clips(), downsample=4)
    b, t_max, d = batch.frames.shape
    assert t_max % 4 == 0 and t_max == (max(lengths) // 4) * 4
    assert batch.mask.shape == (b, t_max) and batch.mask.dtype == mx.bool_
    for i, clip in enumerate(ds.clips()):
        kept = min(clip.n_frames, t_max)
        assert batch.lengths[i] == kept
        assert int(batch.mask[i].sum()) == kept
        # real frames survive the round-trip, padded tail is exactly zero
        assert np.allclose(np.array(batch.frames[i, :kept]), clip.frames[:kept])
        assert np.array_equal(np.array(batch.frames[i, kept:]), np.zeros((t_max - kept, d)))
        assert not np.array(batch.mask[i, kept:]).any()


def test_collate_rejects_bad_inputs(synth_root):
    root, _ = synth_root
    ds = dataset.ClipDataset(root)
    a, b = ds[0], ds[1]
    other = dataset.Clip(
        path=b.path, space=b.space, capture=b.capture,
        frames=b.frames[:, :-1], offsets=b.offsets, meta=b.meta,
    )
    with pytest.raises(layout.LayoutError, match="mixed frame dims"):
        dataset.collate([a, other])
    with pytest.raises(ValueError):
        dataset.collate([])
    with pytest.raises(ValueError):
        dataset.collate([a], downsample=0)


def test_shorter_than_downsample_block_rejected():
    frames = np.zeros((2, 47), dtype=np.float32)
    clip = dataset.Clip(Path("/tmp/x/clip.npz"), "", "", frames, layout.offsets(4, True), {})
    with pytest.raises(ValueError, match="downsample"):
        dataset.collate([clip], downsample=4)


def test_negative_wrong_frame_dim_rejected(synth_root):
    """改错维度就红：frames 与 offsets 表不一致的 shard 必须被拒绝。"""
    root, paths = synth_root
    bad_dir = paths[0].parent.parent / "cap-bad"
    bad_dir.mkdir()
    with np.load(paths[0]) as npz:
        frames = npz["frames"][:, :-1]  # drop one column, keep the old table
        offsets = npz["offsets"]
    np.savez(bad_dir / "clip.npz", frames=frames, offsets=offsets)
    (bad_dir / "meta.json").write_text((paths[0].parent / "meta.json").read_text())
    with pytest.raises(layout.LayoutError, match="D=46"):
        dataset.load_clip(bad_dir / "clip.npz")


def test_negative_missing_meta_key_rejected(synth_root):
    root, paths = synth_root
    import json

    meta = json.loads((paths[0].parent / "meta.json").read_text())
    del meta["provenance"]
    (paths[0].parent / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(layout.LayoutError, match="provenance"):
        dataset.load_clip(paths[0])


def test_negative_nonfinite_frames_rejected(synth_root):
    root, paths = synth_root
    with np.load(paths[0]) as npz:
        frames, offsets = npz["frames"].copy(), npz["offsets"]
    frames[0, 0] = np.nan
    np.savez(paths[0], frames=frames, offsets=offsets)
    with pytest.raises(layout.LayoutError, match="non-finite"):
        dataset.load_clip(paths[0])


def test_negative_padding_never_enters_loss(synth_root):
    """mask 正确性：污染 padding 区不改损失；打乱 mask 就红。"""
    root, _ = synth_root
    ds = dataset.ClipDataset(root)
    batch = dataset.collate(ds.clips(), downsample=4)
    t_max = int(batch.frames.shape[1])
    assert min(batch.lengths) < t_max  # ensure a real padded region exists

    model = MotionVQVAE(frame_dim=ds[0].frame_dim, hidden=16, latent_dim=8,
                        depth=2, codebook_size=16, quantizer_seed=0)
    loss_clean, _ = vqvae_loss(model, batch)

    corrupted = dataclasses.replace(batch, frames=mx.array(batch.frames))
    rng = np.random.default_rng(0)
    noise = mx.array(rng.normal(0, 10.0, size=corrupted.frames.shape).astype(np.float32))
    keep = batch.mask[:, :, None]
    corrupted = dataclasses.replace(
        corrupted, frames=mx.where(keep, batch.frames, batch.frames + noise),
    )
    loss_corrupted, _ = vqvae_loss(model, corrupted)
    mx.eval(loss_clean, loss_corrupted)
    # Masked loss cannot see padded frames directly; a tiny residual remains
    # because stride-2 convs let padded *inputs* bleed into nearby real frames'
    # reconstructions — the documented scaffold limitation (vqvae.py docstring).
    pad_diff = abs(float(loss_clean) - float(loss_corrupted))
    assert pad_diff < 0.01 * float(loss_clean)

    shuffled = dataclasses.replace(
        # contiguous copy: mlx cannot take DLPack arrays with negative strides
        corrupted, mask=mx.array(np.ascontiguousarray(np.array(batch.mask)[:, ::-1])),
    )
    loss_shuffled, _ = vqvae_loss(model, shuffled)
    mx.eval(loss_shuffled)
    assert abs(float(loss_shuffled) - float(loss_corrupted)) > 1e-3
