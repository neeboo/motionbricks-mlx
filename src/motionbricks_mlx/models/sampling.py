"""Deterministic sampling: SplitMix64 uniforms, Gumbel noise, first-max argmax.

Exact port of ``src/sampling.cpp``.  The reference deliberately avoids any
library RNG so that a plan is reproducible from its seed alone; the same
property has to hold here.
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


def uniforms(seed: int, count: int) -> np.ndarray:
    state = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    values = np.empty(count, dtype=np.float32)
    for index in range(count):
        values[index], state = sampling_uniform(state)
    return values


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
