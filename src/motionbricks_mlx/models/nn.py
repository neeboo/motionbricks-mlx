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

import mlx.core as mx

LAYER_NORM_EPSILON = 1.0e-5  # ggml_norm(ctx, x, 1.0e-5F) in src/root.cpp:59


def linear(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    """``ggml_add(ctx, ggml_mul_mat(ctx, matrix, input), bias)``.

    ``weight`` is ``(out, in)`` and ``x`` is ``(..., in)``.
    """
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
