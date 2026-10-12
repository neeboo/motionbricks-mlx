"""Root planner: duration probe, root-token transformer and convolutional decoder.

Port of ``src/root.cpp`` (``run_root_planner_impl`` / ``run_root_duration_probe``)
and its conv-decoder helper.  Layout follows the reference: activations in the
convolutional decoder are ``(channels, time)`` (GGML ``ne = (time, channels)``),
everything in the transformers is ``(time, channels)``.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from . import nn
from .bundle import Weights
from .schema import (BOUNDARY_FRAMES, DILATIONS, ROOT_EMBEDDING, ROOT_HEADS, ROOT_LAYERS)


def nearest_upsample_time(x: mx.array) -> mx.array:
    """``ggml_interpolate(..., ne0 * 2, ..., GGML_SCALE_MODE_NEAREST)``.

    Doubles the time axis; ``out[t] = in[t // 2]``.
    """
    return mx.repeat(x, 2, axis=1)


class RootPlanner:
    """The ``root.gguf`` component: duration logits + dense global-root values."""

    def __init__(self, weights: Weights):
        self.w = weights

    # -- helpers ---------------------------------------------------------

    def _blend(self, projected: mx.array, absent_name: str, mask: mx.array) -> mx.array:
        """``blend`` in src/root.cpp:394-398: ``(projected - absent) * mask + absent``."""
        absent = self.w[absent_name]
        return (projected - absent) * mask + absent

    def _leaky_mlp(self, x: mx.array, prefix: str) -> mx.array:
        """``leaky_mlp`` in src/root.cpp:43-55 (two leaky-relu layers + projection)."""
        hidden = nn.leaky_relu(nn.linear(x, self.w[f"{prefix}.fc_layers.0.weight"],
                                         self.w[f"{prefix}.fc_layers.0.bias"]))
        hidden = nn.leaky_relu(nn.linear(hidden, self.w[f"{prefix}.fc_layers.1.weight"],
                                         self.w[f"{prefix}.fc_layers.1.bias"]))
        return nn.linear(hidden, self.w[f"{prefix}.forward_projection.weight"],
                         self.w[f"{prefix}.forward_projection.bias"])

    def _transformer(self, x: mx.array, model_name: str, layers: int) -> mx.array:
        hidden = x
        for layer in range(layers):
            hidden = nn.transformer_layer(hidden, self.w, f"{model_name}.layers.{layer}.",
                                          ROOT_HEADS)
        return hidden

    def _boundary(self, global_root: mx.array, local_root: mx.array, poses: mx.array,
                  global_mask: mx.array, local_mask: mx.array,
                  pose_mask: mx.array) -> tuple[mx.array, mx.array]:
        """Projection, start/end frame MLPs and position embedding (src/root.cpp:394-414)."""
        pose_emb = self._blend(nn.linear(poses, self.w["_proj_local_pose.weight"],
                                         self.w["_proj_local_pose.bias"]),
                               "_no_local_pose_emb", pose_mask)
        local_emb = self._blend(nn.linear(local_root, self.w["_proj_local_root_value.weight"],
                                          self.w["_proj_local_root_value.bias"]),
                                "_no_local_root_emb", local_mask)
        global_emb = self._blend(nn.linear(global_root, self.w["_proj_global_root_value.weight"],
                                           self.w["_proj_global_root_value.bias"]),
                                 "_no_global_root_emb", global_mask)
        boundary = mx.concatenate([pose_emb, local_emb, global_emb], axis=1)  # (8, 384)
        start_frame = self._leaky_mlp(boundary[0:4], "_proj_start_input")
        end_frame = self._leaky_mlp(boundary[4:8], "_proj_end_input")
        frame_emb = mx.concatenate([start_frame, end_frame], axis=0)          # (8, 512)
        positioned = frame_emb + self.w["_input_position_emb.weight"]
        return frame_emb, positioned

    # -- stages ----------------------------------------------------------

    def duration_logits(self, global_root: mx.array, local_root: mx.array,
                        poses: mx.array, global_mask: mx.array, local_mask: mx.array,
                        pose_mask: mx.array) -> mx.array:
        """``run_root_duration_probe``: the 12 duration logits (src/root.cpp:415-423).

        Uses the masked-duration embedding (upstream token index 12, i.e.
        ``duration_input_tokens = 18``), independent of the planned token count.
        """
        _, positioned = self._boundary(global_root, local_root, poses,
                                       global_mask, local_mask, pose_mask)
        duration_emb = self.w["_proj_input_num_tokens.weight"][12]
        first_input = mx.concatenate([duration_emb[None, :], positioned], axis=0)
        first_output = self._transformer(first_input, "_shared_transformer_model", ROOT_LAYERS)
        return nn.linear(first_output[0:1], self.w["_proj_num_token_output_logit.weight"],
                         self.w["_proj_num_token_output_logit.bias"])[0]

    def plan(self, global_root: mx.array, local_root: mx.array, poses: mx.array,
             global_mask: mx.array, local_mask: mx.array, pose_mask: mx.array,
             requested_tokens: int, dense_target: mx.array,
             dense_target_mask: mx.array) -> tuple[mx.array, mx.array]:
        """``run_root_planner_auto_probe``: ``(duration_logits, global_root_values)``.

        ``global_root_values`` is ``(requested_tokens * 4, 5)`` normalized.
        """
        frame_emb, positioned = self._boundary(global_root, local_root, poses,
                                               global_mask, local_mask, pose_mask)
        duration_emb = self.w["_proj_input_num_tokens.weight"][12]
        first_input = mx.concatenate([duration_emb[None, :], positioned], axis=0)
        first_output = self._transformer(first_input, "_shared_transformer_model", ROOT_LAYERS)
        duration_logits = nn.linear(first_output[0:1],
                                    self.w["_proj_num_token_output_logit.weight"],
                                    self.w["_proj_num_token_output_logit.bias"])[0]

        middle_emb = self.w["_middle_token_emb.weight"][requested_tokens - 6]
        positions = self.w["_position_emb.embed"].reshape(16, ROOT_EMBEDDING)
        second_input = mx.concatenate([middle_emb[None, :], positioned], axis=0)
        second_input = mx.concatenate([second_input, positions[0:requested_tokens]], axis=0)
        second_output = self._transformer(second_input, "_root_token_transformer_model",
                                          ROOT_LAYERS)
        root_tokens = second_output[9:9 + requested_tokens]
        decoder_tokens = nn.transpose_contiguous(root_tokens, requested_tokens, ROOT_EMBEDDING)

        has_frame = mx.clip(global_mask + local_mask + pose_mask, 0.0, 1.0)
        absent = self.w["_conv_no_frame_emb"]
        masked_frames = (frame_emb - absent) * has_frame + absent
        frames = requested_tokens * 4
        middle = mx.broadcast_to(absent[None, :], (frames - 8, ROOT_EMBEDDING))
        dense_frame = mx.concatenate([masked_frames[0:4], middle, masked_frames[4:8]], axis=0)
        values = self._decoder(decoder_tokens, dense_frame, dense_target, dense_target_mask,
                               requested_tokens)
        return duration_logits, values

    # -- convolutional decoder -------------------------------------------

    def _conv(self, x: mx.array, name: str, padding: int, dilation: int) -> mx.array:
        return nn.conv1d_time(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"],
                              padding, dilation)

    def _residuals(self, x: mx.array, stage: int) -> mx.array:
        """``residuals`` in src/root.cpp:120-136 (dilations 27/9/3/1)."""
        hidden = x
        for block, dilation in enumerate(DILATIONS):
            stem = f"_conv_output.model.{stage}.0.model.{block}."
            branch = self._conv(mx.maximum(hidden, 0.0), stem + "conv1", dilation, dilation)
            branch = self._conv(mx.maximum(branch, 0.0), stem + "conv2", 0, 1)
            hidden = hidden + branch
        return hidden

    def _decoder(self, tokens: mx.array, frames_in: mx.array, dense_target: mx.array,
                 dense_target_mask: mx.array, positions: int) -> mx.array:
        """``root_decoder`` in src/root.cpp:143-182."""
        frames = positions * 4
        hidden = mx.maximum(self._conv(tokens, "_conv_output.model.0", 1, 1), 0.0)
        for stage_index in range(2):
            stage = stage_index + 2
            group = 1 << (2 - stage_index)
            stage_positions = positions * (1 << stage_index)
            target_emb = mx.maximum(
                nn.linear(dense_target,
                          self.w[f"_conv_output.target_cond_blocks.{stage_index * 2}.weight"],
                          self.w[f"_conv_output.target_cond_blocks.{stage_index * 2}.bias"]),
                0.0)
            hidden_frames = nn.transpose_contiguous(hidden, 512, stage_positions)
            hidden_frames = nn.reshape_ne(hidden_frames, (512 // group, frames))
            hidden_frames = hidden_frames + (target_emb - hidden_frames) * dense_target_mask
            hidden_frames = nn.reshape_ne(hidden_frames, (512, stage_positions))
            external = nn.reshape_ne(frames_in, (512 * group, stage_positions))
            fused = mx.maximum(
                nn.linear(mx.concatenate([hidden_frames, external], axis=1),
                          self.w[f"_conv_output.external_cond_blocks.{stage_index * 2}.weight"],
                          self.w[f"_conv_output.external_cond_blocks.{stage_index * 2}.bias"]),
                0.0)
            hidden = nn.transpose_contiguous(fused, stage_positions, 512)
            hidden = self._residuals(hidden, stage)
            hidden = nearest_upsample_time(hidden)
            hidden = self._conv(hidden, f"_conv_output.model.{stage}.2", 1, 1)
        hidden = mx.maximum(self._conv(hidden, "_conv_output.model.4", 1, 1), 0.0)
        hidden = self._conv(hidden, "_conv_output.model.6", 1, 1)
        return nn.transpose_contiguous(hidden, 5, frames)
