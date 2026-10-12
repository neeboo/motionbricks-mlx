"""Deterministic sampling: SplitMix64 uniforms, Gumbel noise, first-max argmax.

Exact port of ``src/sampling.cpp``.  The reference deliberately avoids any
library RNG so that a plan is reproducible from its seed alone; the same
property has to hold here.

Optimization note
-----------------
:func:`uniforms` used to advance SplitMix64 one element at a time in Python
(~1280 NumPy scalar operations per plan).  The stream is fully independent per
element -- element ``i`` is the mix of ``seed + (i+1) * GOLDEN`` -- so the same
arithmetic is evaluated as one vectorised uint64 array expression.  Integer
multiplication wraps modulo 2**64 in both forms, and ``z >> 40`` is exact in
float64, so the vectorised stream is bit-identical; ``test_mlx_optimizations``
checks that against the scalar loop element by element.
"""

from __future__ import annotations

import numpy as np

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
FLOAT_SCALE = np.float32(2.0 ** -24)


def sampling_uniform(state: np.uint64) -> tuple[np.float32, np.uint64]:
    """SplitMix64 step mapped to ``[0, 1)`` as ``(z >> 40) * 2**-24``.

    Returns the value and the advanced state (``src/sampling.cpp:7-13``).
    """
    with np.errstate(over="ignore"):
        state = (state + _GOLDEN) & _MASK64
        z = state
        z = ((z ^ (z >> np.uint64(30))) * _MIX1) & _MASK64
        z = ((z ^ (z >> np.uint64(27))) * _MIX2) & _MASK64
        z = z ^ (z >> np.uint64(31))
    return np.float32(np.float64(z >> np.uint64(40)) * np.float64(FLOAT_SCALE)), state


def _uniforms_scalar(seed: int, count: int) -> np.ndarray:
    state = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    values = np.empty(count, dtype=np.float32)
    for index in range(count):
        values[index], state = sampling_uniform(state)
    return values


def _uniforms_vectorised(seed: int, count: int) -> np.ndarray:
    """Same stream, evaluated as whole-array uint64 arithmetic."""
    base = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    steps = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        state = base + steps * _GOLDEN          # wraps modulo 2**64
        z = state
        z = (z ^ (z >> np.uint64(30))) * _MIX1
        z = (z ^ (z >> np.uint64(27))) * _MIX2
        z = z ^ (z >> np.uint64(31))
        magnitude = (z >> np.uint64(40)).astype(np.float64)
    return (magnitude * np.float64(FLOAT_SCALE)).astype(np.float32)


def uniforms(seed: int, count: int) -> np.ndarray:
    from .nn import _option      # local import: keeps this module MLX-free
    if _option("sampling_vector", False):
        return _uniforms_vectorised(seed, count)
    return _uniforms_scalar(seed, count)


def gumbel_noise(uniform: np.ndarray) -> np.ndarray:
    """``-log(max(-log(max(u, 1e-20)), 1e-20))`` in float32 (src/sampling.cpp:15-19)."""
    clipped = np.maximum(uniform, np.float32(1e-20))
    return -np.log(np.maximum(-np.log(clipped), np.float32(1e-20))).astype(np.float32)


def sample_pose_tokens(logits: np.ndarray, uniform: np.ndarray | None,
                       choices: int) -> np.ndarray:
    """``sample_pose_tokens`` (src/sampling.cpp:21-46).

    ``logits`` is ``(rows, choices)`` float32, ``uniform`` the same shape or
    ``None`` for argmax.  Ties resolve to the first maximum, as ``torch.argmax``.
    """
    rows = logits.shape[0]
    scores = logits.astype(np.float32, copy=True)
    if uniform is not None:
        scores = scores + gumbel_noise(uniform.astype(np.float32))
    best = np.argmax(scores, axis=1)
    # np.argmax already returns the first maximum; keep the assertion explicit.
    assert best.shape == (rows,)
    return best.astype(np.int32)
