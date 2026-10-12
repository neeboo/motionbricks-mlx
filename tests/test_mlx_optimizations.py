"""Optimization switches: option grammar, bit-exactness and quantization plumbing.

The optimization brief fixes the acceptance rules in
``docs/optimization-report.md`` *before* the work; these tests pin the two
properties the decisions rest on:

* every float32 optimization is token-identical to the accepted path, and
* quantization is live and numerically different -- which is exactly why it was
  rejected by the token-identity gate rather than by a tolerance.

``MOTIONBRICKS_MLX_OPT`` is read per call, so a single test can compare the
accepted path with any single optimization without restarting the process.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motionbricks_mlx.models import nn, sampling  # noqa: E402
from motionbricks_mlx.models.bundle import Bundle, _parse_quantization, _quantizable  # noqa: E402
from motionbricks_mlx.models.pose import PosePlanner  # noqa: E402
from motionbricks_mlx.models.transition import MlxPipeline, TransitionConstraints  # noqa: E402

MLX_BUNDLE = ROOT / "checkpoints/g1-f32-mlx"
HAVE_BUNDLE = (MLX_BUNDLE / "manifest.json").is_file()

# The full switch set, in the order used by the report.
SWITCHES = ("pose_fast", "pose_compile", "root_fast", "root_compile", "root_dedup",
            "decoder_fast", "decoder_compile", "transition_fast", "sampling_vector")


def synthetic_constraints() -> TransitionConstraints:
    """Same deterministic boundary set as tests/test_mlx_reference.py."""
    global_root = np.zeros((8, 5), np.float32)
    for frame in range(8):
        global_root[frame] = [0.01 * frame, 0.78 + 0.001 * frame, 0.02 * frame,
                              np.cos(0.01 * frame), np.sin(0.01 * frame)]
    local_root = np.tile(np.array([0.05, 0.1, 0.02, 0.78], np.float32), (8, 1))
    local_root[:, 0] *= np.arange(8)
    poses = np.zeros((8, 303), np.float32)
    poses[:, 99:] = np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (8, 34))
    return TransitionConstraints(
        global_root=global_root, local_root=local_root, poses=poses,
        has_global_root=np.ones(8, np.uint8), has_local_root=np.ones(8, np.uint8),
        has_poses=np.ones(8, np.uint8), allowed_tokens=np.ones(11, np.uint8))


@pytest.fixture(autouse=True)
def restore_switches():
    import os
    saved = os.environ.get("MOTIONBRICKS_MLX_OPT")
    yield
    if saved is None:
        os.environ.pop("MOTIONBRICKS_MLX_OPT", None)
    else:
        os.environ["MOTIONBRICKS_MLX_OPT"] = saved


# --- option grammar -------------------------------------------------------

def test_option_grammar():
    import os
    T = [True] * 9
    cases = {
        # Decision 0004 item 1: the float32 group is the default path.
        None: T,
        "1": T,
        "0": [False] * 9,
        # A bare ``name=0`` overrides one switch of the default-on group.
        "pose_compile=0": [True, False, True, True, True, True, True, True, True],
        "1,pose_compile=0": [True, False, True, True, True, True, True, True, True],
        # The old "accepted path plus one switch" is still expressible: the
        # global token comes first and is applied in order.
        "0,pose_compile=1": [False, True, False, False, False, False, False, False, False],
        "*=0,sampling_vector=1": [False] * 8 + [True],
    }
    for value, expected in cases.items():
        if value is None:
            os.environ.pop("MOTIONBRICKS_MLX_OPT", None)
        else:
            os.environ["MOTIONBRICKS_MLX_OPT"] = value
        got = [nn._opt(name) for name in SWITCHES]
        assert got == expected, f"{value!r} -> {got}"


def test_default_is_the_adopted_optimized_path():
    """Decision 0004 item 1: the bit-identical f32 group is default-on.

    Section 8 of the report records every item as *rejected* under the section-0
    performance gate (>= 1.5x); the lead then adopted this group as a default-on
    row because it passes the correctness gate at its highest grade.  Both facts
    hold at once, and this test pins the second one so a later refactor cannot
    silently revert it.
    """
    import os
    os.environ.pop("MOTIONBRICKS_MLX_OPT", None)
    assert [nn._opt(name) for name in SWITCHES] == [True] * 9
    # Every switch the call sites can read has a declared default, and only the
    # bit-identical float32 group is on.
    assert set(nn.OPT_DEFAULTS) == set(SWITCHES)
    assert all(nn.OPT_DEFAULTS.values())


def test_off_restores_the_pre_optimization_path():
    """``MOTIONBRICKS_MLX_OPT=0`` is the accepted pre-optimization behaviour."""
    import os
    os.environ["MOTIONBRICKS_MLX_OPT"] = "0"
    assert [nn._opt(name) for name in SWITCHES] == [False] * 9


# --- vectorised sampling stream -------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 7, 12345, 2 ** 62])
@pytest.mark.parametrize("count", [1, 11, 80, 880, 1280])
def test_vectorised_uniforms_match_the_scalar_stream(seed, count):
    """The whole point of the vectorised SplitMix64: identical bits."""
    scalar = sampling._uniforms_scalar(seed, count)
    vectorised = sampling._uniforms_vectorised(seed, count)
    assert vectorised.dtype == np.float32
    assert np.array_equal(scalar, vectorised)


def test_uniforms_are_in_the_unit_interval():
    values = sampling.uniforms(3, 4096)
    assert values.min() >= 0.0
    assert values.max() < 1.0


# --- bit-exactness of the float32 optimizations ---------------------------

@pytest.mark.skipif(not HAVE_BUNDLE, reason="run scripts/mlx_convert_gguf.py first")
@pytest.mark.parametrize("switches", [
    "0",
    "1",
    "0,root_fast=1,root_dedup=1",
    "0,pose_fast=1,pose_compile=0",
    "0,pose_fast=1,pose_compile=1",
    "0,root_fast=1,root_compile=1,decoder_fast=1,decoder_compile=1",
    "0,transition_fast=1,sampling_vector=1",
])
def test_optimizations_preserve_pose_tokens(switches):
    """The gate from docs/optimization-report.md: identical discrete tokens.

    ``pose.tokens`` is the primary gate because the pose planner amplifies the
    root condition by ~1e3-1e4 and is then followed by an argmax/Gumbel choice.
    """
    import os
    constraints = synthetic_constraints()

    os.environ["MOTIONBRICKS_MLX_OPT"] = "0"
    reference = MlxPipeline(Bundle(MLX_BUNDLE, device="cpu")).run_transition(
        constraints, seed=1, capture_trace=True)

    os.environ["MOTIONBRICKS_MLX_OPT"] = switches
    candidate = MlxPipeline(Bundle(MLX_BUNDLE, device="cpu")).run_transition(
        constraints, seed=1, capture_trace=True)

    assert candidate.selected_tokens == reference.selected_tokens
    assert np.array_equal(candidate.trace["pose.tokens"], reference.trace["pose.tokens"])
    assert np.array_equal(candidate.trace["pose.sampling_uniforms"],
                          reference.trace["pose.sampling_uniforms"])
    assert np.array_equal(candidate.trace["decoder.quantized"],
                          reference.trace["decoder.quantized"])
    # The compile fusion is not bit-identical on the CPU backend (measured
    # 3.6e-7 m on roots, i.e. 3e-4 of the declared tolerance); everything stays
    # far inside the alignment budget.
    assert np.abs(candidate.roots - reference.roots).max() < 1.0e-5
    assert np.abs(candidate.rotations_xyzw - reference.rotations_xyzw).max() < 1.0e-4


@pytest.mark.skipif(not HAVE_BUNDLE, reason="run scripts/mlx_convert_gguf.py first")
def test_compiled_stage_is_reused_across_calls():
    """Repeated calls with one shape must not rebuild the compiled variant."""
    import os
    os.environ["MOTIONBRICKS_MLX_OPT"] = "1"
    pipeline = MlxPipeline(Bundle(MLX_BUNDLE, device="cpu"))
    first = pipeline.pose.plan(np.full((7, 8), 10, np.int32), mx.zeros((28, 4)),
                               mx.zeros((28, 304)), mx.zeros((28, 1)), 7, 7)
    mx.eval(first)
    variants_after_first = pipeline.pose._stage.variants()
    compiles_after_first = pipeline.pose._stage.compiles
    for _ in range(4):
        mx.eval(pipeline.pose.plan(np.full((7, 8), 10, np.int32), mx.zeros((28, 4)),
                                   mx.zeros((28, 304)), mx.zeros((28, 1)), 7, 7))
    assert pipeline.pose._stage.compiles == compiles_after_first == 1
    assert pipeline.pose._stage.variants() == variants_after_first == 1
    # a new token count is one more variant, not a rebuild of the first
    mx.eval(pipeline.pose.plan(np.full((9, 8), 10, np.int32), mx.zeros((36, 4)),
                               mx.zeros((36, 304)), mx.zeros((36, 1)), 9, 9))
    assert pipeline.pose._stage.compiles == 2


@pytest.mark.skipif(not HAVE_BUNDLE, reason="run scripts/mlx_convert_gguf.py first")
def test_all_zero_mask_is_equivalent_to_no_mask():
    """Dropping the additive zero mask is exact, not approximate."""
    import os
    os.environ["MOTIONBRICKS_MLX_OPT"] = "1"
    pipeline = MlxPipeline(Bundle(MLX_BUNDLE, device="cpu"))
    planner = pipeline.pose
    assert planner._mask(11, 11) is None
    masked = planner._mask(11, 7)
    assert masked is not None and masked.shape == (11, 11)
    assert np.isneginf(np.asarray(masked)[0, 7])


# --- quantization plumbing ------------------------------------------------

def test_quantization_specification_parsing():
    assert _parse_quantization(None) == (0, 0)
    assert _parse_quantization("off") == (0, 0)
    assert _parse_quantization("q8") == (64, 8)
    assert _parse_quantization("q4") == (64, 4)
    assert _parse_quantization(8) == (64, 8)
    assert _parse_quantization("4x32") == (32, 4)


def test_table_tensors_are_never_quantized():
    """Row-indexed tables are read without a dequantizing matmul."""
    assert not _quantizable("_pose_token_emb.weight", (88, 32))
    assert not _quantizable("_position_emb.embed", (16, 1024))
    assert not _quantizable("_middle_token_emb.weight", (11, 512))
    assert not _quantizable("_proj_num_valid_positions.weight", (11, 128))
    assert not _quantizable("decoder.model.0.weight", (512, 256, 3))
    assert _quantizable("_transformer_model.layers.0.linear1.weight", (2048, 1024))


@pytest.mark.skipif(not HAVE_BUNDLE, reason="run scripts/mlx_convert_gguf.py first")
def test_quantized_bundle_is_smaller_and_changes_the_model():
    """Quantization is live and numerically different.

    That is *why* the report rejects it: the frozen gate requires identical
    ``pose.tokens``, and a genuinely different model cannot satisfy it.
    """
    reference = Bundle(MLX_BUNDLE, device="cpu")
    quantized = Bundle(MLX_BUNDLE, device="cpu", quantize="q4")
    reference_weights = sum(value.nbytes for value in reference.pose.tensors.values())
    quantized_weights = sum(value.nbytes for value in quantized.pose.tensors.values())
    assert quantized.quantized_tensors > 0
    assert quantized_weights * 2 < reference_weights

    name = "_transformer_model.layers.0.linear1.weight"
    original = np.asarray(reference.pose[name])
    restored = np.asarray(nn.dequantize_weight(quantized.pose[name]))
    assert restored.shape == original.shape
    assert np.abs(restored - original).max() > 1.0e-3


@pytest.mark.skipif(not HAVE_BUNDLE, reason="run scripts/mlx_convert_gguf.py first")
def test_default_bundle_loading_is_unchanged():
    """No quantization unless it is asked for."""
    bundle = Bundle(MLX_BUNDLE, device="cpu")
    assert bundle.quantization == (0, 0)
    assert bundle.quantized_tensors == 0
    assert all(isinstance(value, mx.array) for value in bundle.pose.tensors.values())
