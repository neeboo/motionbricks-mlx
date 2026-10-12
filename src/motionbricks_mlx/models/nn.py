"""MLX primitives mirroring the reference implementation's GGML operations.

Every function here exists because the reference runtime (``motion-bricks.cpp``)
builds the same maths out of GGML ops; the docstrings name the exact C++ call
site so the port can be audited layer by layer.

Layout convention
-----------------
GGML stores a tensor of logical shape ``(a, b, c)`` with ``ne = (c, b, a)`` —
the *last* GGML axis varies fastest.  The NumPy/MLX view of the same bytes is
therefore ``(a, b, c)`` in C order, which is also PyTorch's order.  All tensors
in this module are in that MLX order:

* activations: ``(time, channels)``
* ``nn.Linear``-style weights: ``(out, in)``
* ``nn.Conv1d``-style weights: ``(out, in, kernel)``
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mlx.core as mx

LAYER_NORM_EPSILON = 1.0e-5  # ggml_norm(ctx, x, 1.0e-5F) in src/root.cpp:59

# ---------------------------------------------------------------------------
# Optimization switches
# ---------------------------------------------------------------------------
# Every optimization in this port sits behind one environment variable so that
# the accepted (pre-optimization) behaviour stays reproducible and each A/B
# measurement can be rerun from a single process:
#
#     MOTIONBRICKS_MLX_OPT=1                   -> every optimization
#     MOTIONBRICKS_MLX_OPT=transition_fast=1   -> just one of them
#
# **The defaults are OFF.**  `docs/optimization-report.md` fixed the adoption
# rule before the work: a candidate is adopted only at >= 1.5x per-step p50 or
# >= 2x cold start / resident memory.  The float32 optimizations reach 1.22-1.24x
# and are therefore recorded as *rejected*, and a rejected optimization must not
# silently become the default.  They are bit-exact and deterministic (see the
# report and tests/test_mlx_optimizations.py); enabling them is one environment
# variable, and the measured numbers are in the report.
#
# The value is read on every call (not at import time) so that both arms can be
# interleaved inside one process.  Unknown names keep their declared default, so
# adding a switch cannot silently change an existing run.
_FALSY = frozenset(("", "0", "false", "no", "off"))


def _option(name: str, default: bool = False) -> bool:
    """True when optimization *name* is enabled for this process.

    Grammar of ``MOTIONBRICKS_MLX_OPT`` (comma separated, applied in order):

    * a bare token is the global switch: ``MOTIONBRICKS_MLX_OPT=1`` enables
      every optimization, ``=0`` disables every one;
    * ``name=1`` / ``name=0`` overrides one optimization, and ``*=1`` / ``*=0``
      overrides all of them.

    So ``MOTIONBRICKS_MLX_OPT=pose_compile=1`` turns on only the pose planner's
    compiled graph, on top of the accepted baseline.
    """
    raw = os.environ.get("MOTIONBRICKS_MLX_OPT")
    if raw is None:
        return default
    enabled = default
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, separator, value = item.partition("=")
        key = key.strip().lower()
        if separator:
            if key in ("*", "all", "opt") or key == name:
                enabled = value.strip().lower() not in _FALSY
        else:
            enabled = key not in _FALSY
    return enabled


class StageCache:
    """``mx.compile`` wrapper keyed by an explicit shape signature.

    MLX compiles a function once per input signature; ``positions`` takes 11
    values (``tokens`` 6..16) in a single planning session, so the compiled
    variants are memoised here instead of being rebuilt.  ``compiles`` counts
    how many variants were actually built, which is what the "repeated graph
    compilation" question in the optimization brief needs to be answered with.
    """

    def __init__(self, name: str, body):
        self.name = name
        self._body = body
        self._variants: dict[tuple, object] = {}
        self.compiles = 0
        self.calls = 0

    def __call__(self, signature: tuple, *args):
        self.calls += 1
        compiled = self._variants.get(signature)
        if compiled is None:
            compiled = mx.compile(self._body)
            self._variants[signature] = compiled
            self.compiles += 1
        return compiled(*args)

    def variants(self) -> int:
        return len(self._variants)


# ---------------------------------------------------------------------------
# Weight quantization
# ---------------------------------------------------------------------------
# The inference stages are latency bound by the *dependent* kernel chain and by
# streaming every weight of the model on every plan (the pose planner alone
# re-reads ~537 MB of float32 matrices per forward).  MLX's affine group
# quantization cuts that traffic by 2x (q8) or 4x (q4); ``linear`` below is the
# only place weights are consumed as matrices, so a quantized weight only has to
# be understood there.
#
# Quantization changes numerics by construction, so it is off by default and
# measured against the token-identity gate rather than assumed safe.


@dataclass(frozen=True)
class QuantizedWeight:
    """An affine-quantized ``(out, in)`` matrix, ready for ``linear``."""

    packed: mx.array
    scales: mx.array
    biases: mx.array | None
    group_size: int
    bits: int
    shape: tuple[int, int]

    @property
    def nbytes(self) -> int:
        total = self.packed.nbytes + self.scales.nbytes
        if self.biases is not None:
            total += self.biases.nbytes
        return total


def quantize_weight(weight: mx.array, group_size: int, bits: int) -> mx.array | QuantizedWeight:
    """Quantize a 2-D linear weight, or return it unchanged when ineligible.

    MLX requires the last (input) dimension to be divisible by ``group_size``
    and only supports group sizes 32/64/128; a tensor that does not fit stays in
    float32 so the model remains complete.
    """
    if weight.ndim != 2:
        return weight
    if weight.shape[1] % group_size != 0:
        return weight
    packed, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
    return QuantizedWeight(packed, scales, biases, group_size, bits,
                           (int(weight.shape[0]), int(weight.shape[1])))


def dequantize_weight(weight: mx.array | QuantizedWeight) -> mx.array:
    """Float32 view of a weight, for tracing and for tests."""
    if not isinstance(weight, QuantizedWeight):
        return weight
    return mx.dequantize(weight.packed, weight.scales, weight.biases,
                         group_size=weight.group_size, bits=weight.bits).reshape(weight.shape)


def linear(x: mx.array, weight: mx.array | QuantizedWeight, bias: mx.array) -> mx.array:
    """``ggml_add(ctx, ggml_mul_mat(ctx, matrix, input), bias)``.

    ``weight`` is ``(out, in)`` and ``x`` is ``(..., in)``.  A quantized weight
    is multiplied through MLX's quantized kernel; the float32 path is unchanged.
    """
    if isinstance(weight, QuantizedWeight):
        result = mx.quantized_matmul(x, weight.packed, weight.scales, weight.biases,
                                     transpose=True, group_size=weight.group_size,
                                     bits=weight.bits)
        return result if bias is None else result + bias
    return x @ weight.T + bias


def layer_norm(x: mx.array, scale: mx.array, bias: mx.array) -> mx.array:
    """``ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5), scale), bias)``.

    ``ggml_norm`` normalizes over ``ne[0]`` — the fastest axis, i.e. the last
    MLX axis (the embedding dimension) — with a *biased* variance and no affine
    transform of its own (src/root.cpp:57-60).
    """
    return mx.fast.layer_norm(x, scale, bias, LAYER_NORM_EPSILON)


def leaky_relu(x: mx.array, slope: float = 0.01) -> mx.array:
    return mx.where(x > 0, x, x * slope)


def reshape_ne(x: mx.array, ne: tuple[int, ...]) -> mx.array:
    """``ggml_reshape_*``: reinterpret the flat buffer as GGML shape ``ne``.

    Because the MLX view of a contiguous GGML tensor is ``reversed(ne)``, a
    GGML reshape is an MLX reshape to ``reversed(new_ne)``; both preserve the
    flat (C-order) element sequence, so this is exact.
    """
    return x.reshape(tuple(reversed(ne)))


def transpose_contiguous(x: mx.array, first: int, second: int) -> mx.array:
    """``ggml_cont_2d(ggml_transpose(x), first, second)`` (src/root.cpp:138-141).

    ``ggml_transpose`` swaps ``ne[0]``/``ne[1]``; the following ``cont_2d``
    reinterprets the swapped buffer as ``ne = (first, second)``.
    """
    return x.T.reshape((second, first))


def conv1d_time(x: mx.array, weight: mx.array, bias: mx.array,
                padding: int, dilation: int) -> mx.array:
    """1-D convolution over time, exactly as ``src/root.cpp:109-118``.

    The reference lowers this to ``ggml_im2col`` (stride 1) + ``ggml_mul_mat``:

    * ``x`` has GGML ``ne = (T_in, C_in)`` -> MLX ``(C_in, T_in)``,
    * ``weight`` has GGML ``ne = (K, C_in, C_out)`` -> MLX ``(C_out, C_in, K)``,
    * output length is ``T_in + 2*padding - dilation*(K-1)``,
    * it is a plain cross-correlation, i.e. ``nn.Conv1d``.
    """
    time = x.shape[1]
    kernel = weight.shape[2]
    out_time = time + 2 * padding - dilation * (kernel - 1)
    if out_time <= 0:
        raise ValueError(
            f"conv1d_time: kernel {kernel} with padding {padding} / dilation "
            f"{dilation} does not fit {time} input frames"
        )
    # Reference call sites always keep the length (padding == dilation with K == 3,
    # or K == 1 with no padding); fall back to an explicit im2col if that changes.
    if out_time == time:
        # MLX wants (N, L, C_in) input and (C_out, K, C_in) weights.
        result = mx.conv1d(x.T[None, :, :], weight.transpose(0, 2, 1),
                           stride=1, padding=padding, dilation=dilation)
        return result[0].T + bias[:, None]
    padded = mx.pad(x, [(0, 0), (padding, padding)]) if padding else x
    gather = mx.arange(out_time)[:, None] + (mx.arange(kernel) * dilation)[None, :]
    patches = padded[:, gather]                               # (C_in, T_out, K)
    return mx.einsum("ctk,ock->ot", patches, weight) + bias[:, None]


def im2col_reference(x: mx.array, weight: mx.array, padding: int, dilation: int) -> mx.array:
    """Explicit ``ggml_im2col`` + ``ggml_mul_mat`` lowering, used by tests.

    Independent of :func:`conv1d_time`'s fast path so the two can be compared.
    """
    time = x.shape[1]
    kernel = weight.shape[2]
    out_time = time + 2 * padding - dilation * (kernel - 1)
    padded = mx.pad(x, [(0, 0), (padding, padding)]) if padding else x
    rows = [mx.stack([padded[:, position + index * dilation] for index in range(kernel)], axis=1)
            for position in range(out_time)]
    columns = mx.stack(rows, axis=1)                          # (C_in, T_out, K)
    return mx.einsum("ctk,ock->ot", columns, weight)


def attention(qkv: mx.array, heads: int, mask: mx.array | None = None) -> mx.array:
    """Multi-head self attention as built in ``src/root.cpp:62-107``.

    ``qkv`` is ``(T, 3 * embedding)`` (the three projections concatenated along
    the last axis, PyTorch ``nn.MultiheadAttention`` order).  ``mask`` is an
    additive ``(T, T)`` mask (``0`` / ``-inf``) as used by ``ggml_soft_max_ext``
    in the pose planner (src/pose.cpp:83-84); ``None`` means plain softmax.
    """
    time, width = qkv.shape
    embedding = width // 3
    head_width = embedding // heads
    query, key, value = qkv[:, :embedding], qkv[:, embedding:2 * embedding], qkv[:, 2 * embedding:]
    # -> (heads, T, head_width)
    split = lambda part: part.reshape(time, heads, head_width).transpose(1, 0, 2)
    query, key, value = split(query), split(key), split(value)
    scale = 1.0 / (head_width ** 0.5)
    scores = (query @ key.transpose(0, 2, 1)) * scale     # (heads, T_q, T_k)
    if mask is not None:
        scores = scores + mask[None, :, :]
    weights = mx.softmax(scores, axis=-1)
    attended = weights @ value                            # (heads, T, head_width)
    merged = attended.transpose(1, 0, 2).reshape(time, embedding)
    return merged


def transformer_layer(x: mx.array, weights, prefix: str, heads: int,
                      mask: mx.array | None = None) -> mx.array:
    """One pre-norm-free ``nn.TransformerEncoderLayer`` (post-norm), relu feed-forward.

    Mirrors ``transformer_layer`` (src/pose.cpp:50-100) and ``transformer``
    (src/root.cpp:62-107): attention and feed-forward are added to the residual
    *before* the layer norms.
    """
    qkv = linear(x, weights[f"{prefix}self_attn.in_proj_weight"],
                 weights[f"{prefix}self_attn.in_proj_bias"])
    attended = attention(qkv, heads, mask)
    attention_out = linear(attended, weights[f"{prefix}self_attn.out_proj.weight"],
                           weights[f"{prefix}self_attn.out_proj.bias"])
    hidden = layer_norm(x + attention_out, weights[f"{prefix}norm1.weight"],
                        weights[f"{prefix}norm1.bias"])
    feed = mx.maximum(linear(hidden, weights[f"{prefix}linear1.weight"],
                             weights[f"{prefix}linear1.bias"]), 0.0)
    feed = linear(feed, weights[f"{prefix}linear2.weight"],
                  weights[f"{prefix}linear2.bias"])
    return layer_norm(hidden + feed, weights[f"{prefix}norm2.weight"],
                      weights[f"{prefix}norm2.bias"])
