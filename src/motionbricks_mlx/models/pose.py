"""Pose planner: 16-layer transformer over pose tokens and boundary conditions.

Port of ``src/pose.cpp`` (``run_pose_planner``).  One forward returns
``positions x 80`` logits (8 heads x 10 codebook choices per position).
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

    def plan(self, pose_tokens: np.ndarray, local_root_values: mx.array,
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

    @staticmethod
    def reshape_logits(logits: mx.array) -> mx.array:
        """``(positions, 80)`` -> ``(positions * 8, 10)`` codebook scores.

        ``sample_pose_tokens`` (src/sampling.cpp:34-44) walks the flat logits in
        rows of ten, so row ``position * 8 + head`` holds that head's choices.
        """
        positions = logits.shape[0]
        return logits.reshape(positions * POSE_HEADS, POSE_TOKEN_CHOICES)
