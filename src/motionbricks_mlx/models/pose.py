"""Pose planner: 16-layer transformer over pose tokens and boundary conditions.

Port of ``src/pose.cpp`` (``run_pose_planner``).  One forward returns
``positions x 80`` logits (8 heads x 10 codebook choices per position).

Optimization note
-----------------
``plan`` has two implementations.  ``_plan_reference`` is the accepted port,
kept verbatim so ``MOTIONBRICKS_MLX_OPT=pose_fast=0`` reproduces the baseline
byte for byte.  ``_core`` is the same maths with

* every per-call constant (head offsets, the position embedding slice, the
  duration embedding row, the additive mask) hoisted into a cache instead of
  being rebuilt -- and, for the mask, re-uploaded to the device -- on every
  call, and
* the all-zero mask dropped (``scores + 0`` is exact, and ``exp(-0.0) ==
  exp(+0.0)``, so plain softmax is bit-identical to softmax over an additive
  zero mask),

wrapped in ``mx.compile`` so MLX fuses the ~40 elementwise ops per transformer
layer into far fewer kernels.  MLX recompiles on every input-shape change, so
the 11 possible ``positions`` values are memoised in :class:`nn.StageCache`.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from . import nn
from .bundle import Weights
from .schema import (POSE_EMBEDDING, POSE_HEADS, POSE_LAYERS, POSE_TOKEN_CHOICES,
                     POSE_TRANSFORMER_HEADS)


class PosePlanner:
    """The ``pose.gguf`` component."""

    def __init__(self, weights: Weights):
        self.w = weights
        # Lazily-built, per-shape constants for the optimized path.
        self._head_offsets: mx.array | None = None
        self._masks: dict[tuple[int, int], mx.array | None] = {}
        self._position_tables: dict[int, mx.array] = {}
        self._duration_rows: dict[int, mx.array] = {}
        self._stage: nn.StageCache | None = None
        # Round-3 batch bake: one compiled vmapped variant per (positions,
        # num_tokens), because a batch can only hold scenarios that selected the
        # same duration.
        self._batch_variants: dict[tuple[int, int], object] = {}

    # -- reference implementation (unchanged) -----------------------------

    def token_embedding(self, pose_tokens: np.ndarray, positions: int) -> mx.array:
        """Token embedding + head offsets (src/pose.cpp:225-237).

        ``pose_tokens`` is ``(positions, 8)``; each head's vocabulary slice is
        offset by ``head * 11`` into the shared 88-row table, and the eight
        32-wide embeddings are concatenated into a 256-wide per-position vector.
        """
        offsets = (np.arange(POSE_HEADS, dtype=np.int32) * 11)[None, :]
        indices = (pose_tokens.astype(np.int32) + offsets).reshape(-1)
        table = self.w["_pose_token_emb.weight"]
        embedded = table[mx.array(indices)]                      # (positions * 8, 32)
        embedded = embedded.reshape(positions, 8 * 32)           # heads concatenated
        hidden = nn.leaky_relu(nn.linear(
            embedded, self.w["_proj_pose_token_emb.fc_layers.0.weight"],
            self.w["_proj_pose_token_emb.fc_layers.0.bias"]))
        hidden = nn.leaky_relu(nn.linear(
            hidden, self.w["_proj_pose_token_emb.fc_layers.1.weight"],
            self.w["_proj_pose_token_emb.fc_layers.1.bias"]))
        hidden = nn.linear(hidden, self.w["_proj_pose_token_emb.forward_projection.weight"],
                           self.w["_proj_pose_token_emb.forward_projection.bias"])
        frames = positions * 4
        return nn.reshape_ne(hidden, (160, frames))               # (frames, 160)

    def _plan_reference(self, pose_tokens: np.ndarray, local_root_values: mx.array,
                        pose_condition: mx.array, has_pose_condition: mx.array,
                        positions: int, num_tokens: int) -> mx.array:
        """Return per-position codebook logits ``(positions, 80)``."""
        frames = positions * 4
        token_embedding = self.token_embedding(pose_tokens, positions)

        condition_embedding = nn.linear(pose_condition, self.w["_proj_local_pose.weight"],
                                        self.w["_proj_local_pose.bias"])
        pose_embedding = token_embedding + \
            (condition_embedding - token_embedding) * has_pose_condition
        pose_embedding = nn.reshape_ne(pose_embedding, (640, positions))

        root_embedding = nn.linear(nn.reshape_ne(local_root_values, (16, positions)),
                                   self.w["_proj_local_root_values.weight"],
                                   self.w["_proj_local_root_values.bias"])
        duration_embedding = mx.broadcast_to(
            self.w["_proj_num_valid_positions.weight"][num_tokens - 6][None, :],
            (positions, 128))
        combined = mx.concatenate([pose_embedding, root_embedding, duration_embedding], axis=1)

        hidden = mx.maximum(nn.linear(combined, self.w["_proj_input.0.weight"],
                                      self.w["_proj_input.0.bias"]), 0.0)
        positions_table = self.w["_position_emb.embed"].reshape(16, POSE_EMBEDDING)
        hidden = hidden + positions_table[0:positions]

        mask = np.zeros((positions, positions), dtype=np.float32)
        if num_tokens < positions:
            mask[:, num_tokens:] = -np.inf
        mask_array = mx.array(mask)
        for layer in range(POSE_LAYERS):
            hidden = nn.transformer_layer(hidden, self.w,
                                          f"_transformer_model.layers.{layer}.",
                                          POSE_TRANSFORMER_HEADS, mask_array)
        return nn.linear(hidden, self.w["_proj_pose_output_logit.weight"],
                         self.w["_proj_pose_output_logit.bias"])

    # -- optimized implementation ----------------------------------------

    def _mask(self, positions: int, num_tokens: int) -> mx.array | None:
        """Additive attention mask, cached; ``None`` when it is all zeros."""
        key = (positions, num_tokens)
        if key not in self._masks:
            if num_tokens >= positions:
                # scores + 0.0 is exact and softmax(-0.0) == softmax(+0.0), so
                # the zero mask is dropped rather than materialised.
                self._masks[key] = None
            else:
                mask = np.zeros((positions, positions), dtype=np.float32)
                mask[:, num_tokens:] = -np.inf
                self._masks[key] = mx.array(mask)
        return self._masks[key]

    def _position_table(self, positions: int) -> mx.array:
        table = self._position_tables.get(positions)
        if table is None:
            table = self.w["_position_emb.embed"].reshape(16, POSE_EMBEDDING)[0:positions]
            self._position_tables[positions] = table
        return table

    def _duration_row(self, num_tokens: int) -> mx.array:
        row = self._duration_rows.get(num_tokens)
        if row is None:
            row = self.w["_proj_num_valid_positions.weight"][num_tokens - 6]
            self._duration_rows[num_tokens] = row
        return row

    def _core(self, pose_tokens: mx.array, local_root_values: mx.array,
              pose_condition: mx.array, has_pose_condition: mx.array,
              positions: int, num_tokens: int) -> mx.array:
        """All-MLX pose planner forward; ``positions``/``num_tokens`` are
        trace-time constants so each (positions, num_tokens) pair is one
        compiled variant."""
        frames = positions * 4
        if self._head_offsets is None:
            self._head_offsets = mx.array(
                (np.arange(POSE_HEADS, dtype=np.int32) * 11)[None, :])
        indices = (pose_tokens.astype(mx.int32) + self._head_offsets).reshape(-1)
        table = self.w["_pose_token_emb.weight"]
        embedded = table[indices].reshape(positions, 8 * 32)
        hidden = nn.leaky_relu(nn.linear(
            embedded, self.w["_proj_pose_token_emb.fc_layers.0.weight"],
            self.w["_proj_pose_token_emb.fc_layers.0.bias"]))
        hidden = nn.leaky_relu(nn.linear(
            hidden, self.w["_proj_pose_token_emb.fc_layers.1.weight"],
            self.w["_proj_pose_token_emb.fc_layers.1.bias"]))
        hidden = nn.linear(hidden, self.w["_proj_pose_token_emb.forward_projection.weight"],
                           self.w["_proj_pose_token_emb.forward_projection.bias"])
        token_embedding = nn.reshape_ne(hidden, (160, frames))     # (frames, 160)

        condition_embedding = nn.linear(pose_condition, self.w["_proj_local_pose.weight"],
                                        self.w["_proj_local_pose.bias"])
        pose_embedding = token_embedding + \
            (condition_embedding - token_embedding) * has_pose_condition
        pose_embedding = nn.reshape_ne(pose_embedding, (640, positions))

        root_embedding = nn.linear(nn.reshape_ne(local_root_values, (16, positions)),
                                   self.w["_proj_local_root_values.weight"],
                                   self.w["_proj_local_root_values.bias"])
        duration_embedding = mx.broadcast_to(self._duration_row(num_tokens)[None, :],
                                             (positions, 128))
        combined = mx.concatenate([pose_embedding, root_embedding, duration_embedding], axis=1)

        hidden = mx.maximum(nn.linear(combined, self.w["_proj_input.0.weight"],
                                      self.w["_proj_input.0.bias"]), 0.0)
        hidden = hidden + self._position_table(positions)

        mask = self._mask(positions, num_tokens)
        for layer in range(POSE_LAYERS):
            hidden = nn.transformer_layer(hidden, self.w,
                                          f"_transformer_model.layers.{layer}.",
                                          POSE_TRANSFORMER_HEADS, mask)
        return nn.linear(hidden, self.w["_proj_pose_output_logit.weight"],
                         self.w["_proj_pose_output_logit.bias"])

    def plan(self, pose_tokens: np.ndarray, local_root_values: mx.array,
             pose_condition: mx.array, has_pose_condition: mx.array,
             positions: int, num_tokens: int) -> mx.array:
        """Return per-position codebook logits ``(positions, 80)``."""
        if not nn._opt("pose_fast"):
            return self._plan_reference(pose_tokens, local_root_values, pose_condition,
                                        has_pose_condition, positions, num_tokens)
        tokens = pose_tokens if isinstance(pose_tokens, mx.array) else mx.array(pose_tokens)
        if nn._opt("pose_compile"):
            if self._stage is None:
                self._stage = nn.StageCache("pose", self._core)
            return self._stage((positions, num_tokens), tokens, local_root_values,
                               pose_condition, has_pose_condition, positions, num_tokens)
        return self._core(tokens, local_root_values, pose_condition, has_pose_condition,
                          positions, num_tokens)

    def plan_batch(self, pose_tokens: mx.array, local_root_values: mx.array,
                   pose_condition: mx.array, has_pose_condition: mx.array,
                   positions: int, num_tokens: int) -> mx.array:
        """``B`` independent scenarios through one batched forward, ``(B, positions, 80)``.

        Every row must share ``positions``/``num_tokens``: the compiled variants
        and the GGML layout reshapes are shape-keyed, and two scenarios that
        selected different durations cannot share a batch.  ``mx.vmap`` supplies
        the batch axis, so the body is the *single-scenario* code path.

        This is **not** a drop-in for :meth:`plan`: lowering the body to batched
        primitives changes the accumulation order of every ``matmul``, measured
        at 1.07e-04 on the logits **even at B = 1** (report section 23), so a
        batched row is never bitwise-equal to the serial plan.  That is exactly
        what the bake's token campaign measures.
        """
        key = (positions, num_tokens)
        variant = self._batch_variants.get(key)
        if variant is None:
            def body(tokens, local_root, condition, has_condition,
                     _positions=positions, _num_tokens=num_tokens):
                return self._core(tokens, local_root, condition, has_condition,
                                  _positions, _num_tokens)
            variant = mx.compile(mx.vmap(body, in_axes=(0, 0, 0, 0)))
            self._batch_variants[key] = variant
        return variant(pose_tokens, local_root_values, pose_condition, has_pose_condition)

    @staticmethod
    def reshape_logits(logits: mx.array) -> mx.array:
        """``(positions, 80)`` -> ``(positions * 8, 10)`` codebook scores.

        ``sample_pose_tokens`` (src/sampling.cpp:34-44) walks the flat logits in
        rows of ten, so row ``position * 8 + head`` holds that head's choices.
        """
        positions = logits.shape[0]
        return logits.reshape(positions * POSE_HEADS, POSE_TOKEN_CHOICES)
