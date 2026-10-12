"""Model forward shapes, straight-through gradient flow, EMA state, negatives."""
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motionbricks_mlx.training import synth
from motionbricks_mlx.training.dataset import ClipDataset, collate
from motionbricks_mlx.training.vqvae import MotionVQVAE, seed_parameters, vqvae_loss


@pytest.fixture()
def synth_batch(tmp_path):
    paths = synth.synth_capture(tmp_path, space="sp", n_clips=3, joints=4, seed=11)
    ds = ClipDataset(tmp_path)
    return ds, collate(ds.clips(), downsample=4)


def tiny_model(frame_dim=47):
    model = MotionVQVAE(frame_dim=frame_dim, hidden=16, latent_dim=8,
                        depth=2, codebook_size=16, quantizer_seed=0)
    seed_parameters(model.encoder, np.random.default_rng(1))
    seed_parameters(model.decoder, np.random.default_rng(2))
    return model


def test_forward_shapes(synth_batch):
    ds, batch = synth_batch
    model = tiny_model(ds[0].frame_dim)
    out = model(batch.frames, batch.mask)
    b, t, d = batch.frames.shape
    assert out["recon"].shape == (b, t, d)
    t_latent = t // 4
    assert out["indices"].shape == (b * t_latent,)
    assert out["mask_t"].shape == (b, t_latent)
    # latent mask is the frame mask block-aligned by the downsample factor
    assert np.array_equal(np.array(out["mask_t"]), np.array(batch.mask)[:, ::4])
    mx.eval(out["recon"])


def test_forward_rejects_unaligned_time(synth_batch):
    _ds, batch = synth_batch
    model = tiny_model()
    frames = mx.zeros((2, 10, 47))  # 10 % 4 != 0
    mask = mx.ones((2, 10), dtype=mx.bool_)
    with pytest.raises(ValueError, match="downsample"):
        model(frames, mask)


def test_quantizer_frozen_from_optimizer(synth_batch):
    _ds, _batch = synth_batch
    model = tiny_model()
    leaves = [k for k, _ in nn.utils.tree_flatten(model.trainable_parameters())]
    assert not any(k.startswith("quantizer") for k in leaves)
    assert any(k.startswith("encoder") for k in leaves)
    # and the EMA state arrays are still present on the module itself
    assert model.quantizer.codebook.shape == (16, 8)


def test_commit_positive_before_seeding(synth_batch):
    _ds, batch = synth_batch
    model = tiny_model()
    _loss, aux = vqvae_loss(model, batch)
    mx.eval(aux["commit"])
    assert float(aux["commit"]) > 1e-6  # codes sit away from encoder outputs


def test_straight_through_reaches_encoder_and_decoder(synth_batch):
    ds, batch = synth_batch
    model = tiny_model(ds[0].frame_dim)
    (_, _aux), grads = nn.value_and_grad(model, vqvae_loss)(model, batch)
    flat = dict(nn.utils.tree_flatten(grads))  # keys are dotted: "encoder.in_proj.weight"
    enc_sum = sum(float(mx.abs(v).sum()) for k, v in flat.items() if k.startswith("encoder."))
    dec_sum = sum(float(mx.abs(v).sum()) for k, v in flat.items() if k.startswith("decoder."))
    assert enc_sum > 0.0, "no gradient reached the encoder (straight-through broken)"
    assert dec_sum > 0.0
    assert not any(k.startswith("quantizer.") for k in flat)


def test_ema_update_changes_state(synth_batch):
    _ds, batch = synth_batch
    model = tiny_model()
    out = model(batch.frames, batch.mask)
    mx.eval(out["state"][0])
    old_cb = np.array(model.quantizer.codebook)
    old_cs = np.array(model.quantizer.cluster_size)
    model.quantizer.update_state(out["state"])
    new_cb = np.array(model.quantizer.codebook)
    assert not np.array_equal(new_cb, old_cb)
    assert float(np.array(model.quantizer.cluster_size).sum()) > float(old_cs.sum())


def test_seed_from_encodings_rejects_empty_mask(synth_batch):
    _ds, batch = synth_batch
    model = tiny_model()
    z = mx.zeros((batch.frames.shape[0], 8))
    with pytest.raises(ValueError, match="no real frames"):
        model.quantizer.seed_from_encodings(z, mx.zeros((z.shape[0],)))


def test_seed_parameters_is_deterministic_and_seed_sensitive():
    def build(rng):
        m = MotionVQVAE(frame_dim=47, hidden=8, latent_dim=4, depth=1, codebook_size=4)
        seed_parameters(m.encoder, rng)
        return m.encoder

    a, b, c = build(np.random.default_rng(5)), build(np.random.default_rng(5)), build(np.random.default_rng(6))
    pa = dict(nn.utils.tree_flatten(a.parameters()))
    pb = dict(nn.utils.tree_flatten(b.parameters()))
    pc = dict(nn.utils.tree_flatten(c.parameters()))
    assert all(np.array_equal(np.array(pa[k]), np.array(pb[k])) for k in pa)
    assert any(not np.array_equal(np.array(pa[k]), np.array(pc[k])) for k in pa)


def test_negative_wrong_input_dim_raises(synth_batch):
    _ds, batch = synth_batch
    model = tiny_model(frame_dim=47)
    wrong = mx.zeros((2, 8, 35), dtype=mx.float32)  # D mismatch: 35 != 47
    mask = mx.ones((2, 8), dtype=mx.bool_)
    with pytest.raises(ValueError):
        model(wrong, mask)
