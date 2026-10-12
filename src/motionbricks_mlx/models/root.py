"""Root planner: duration probe, root-token transformer and convolutional decoder.

Port of ``src/root.cpp`` (``run_root_planner_impl`` / ``run_root_duration_probe``)
and its conv-decoder helper.  Layout follows the reference: activations in the
convolutional decoder are ``(channels, time)`` (GGML ``ne = (time, channels)``),
everything in the transformers is ``(time, channels)``.

Optimization note
-----------------
``MlxPipeline.run_transition`` needs the duration logits *before* it can call
:meth:`RootPlanner.plan`, because the probing decides how many tokens are
planned.  The accepted port therefore ran the probe twice: once through
:meth:`duration_logits` and again inside :meth:`plan`, where its result was
discarded (``_, predicted_global = self.root.plan(...)``).  The probe is a
boundary embedding plus a 3-layer transformer, so that was a duplicate graph,
a duplicate dispatch chain and one extra device synchronisation per step.

:meth:`plan` now accepts the already-computed logits through
``duration_logits=``; when they are supplied the probe transformer is not built
at all.  ``MOTIONBRICKS_MLX_OPT=root_dedup=0`` restores the previous behaviour,
and ``root_fast=0`` restores the verbatim accepted implementation.
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
        self._middle_rows: dict[int, mx.array] = {}
        self._position_tables: dict[int, mx.array] = {}
        self._stages: dict[str, nn.StageCache] = {}

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

    def _middle_row(self, requested_tokens: int) -> mx.array:
        row = self._middle_rows.get(requested_tokens)
        if row is None:
            row = self.w["_middle_token_emb.weight"][requested_tokens - 6]
            self._middle_rows[requested_tokens] = row
        return row

    def _position_table(self, requested_tokens: int) -> mx.array:
        table = self._position_tables.get(requested_tokens)
        if table is None:
            table = self.w["_position_emb.embed"].reshape(16, ROOT_EMBEDDING)[0:requested_tokens]
            self._position_tables[requested_tokens] = table
        return table

    def _stage(self, name: str, body) -> nn.StageCache:
        cache = self._stages.get(name)
        if cache is None:
            cache = nn.StageCache(name, body)
            self._stages[name] = cache
        return cache

    # -- stages ----------------------------------------------------------

    def duration_logits(self, global_root: mx.array, local_root: mx.array,
                        poses: mx.array, global_mask: mx.array, local_mask: mx.array,
                        pose_mask: mx.array) -> mx.array:
        """``run_root_duration_probe``: the 12 duration logits (src/root.cpp:415-423).

        Uses the masked-duration embedding (upstream token index 12, i.e.
        ``duration_input_tokens = 18``), independent of the planned token count.
        """
        if not nn._opt("root_fast"):
            return self._duration_logits_reference(
                global_root, local_root, poses, global_mask, local_mask, pose_mask)
        _, positioned = self._boundary(global_root, local_root, poses,
                                       global_mask, local_mask, pose_mask)
        if nn._opt("root_compile"):
            return self._stage("duration", self._duration_core)((), positioned)
        return self._duration_core(positioned)

    def _duration_core(self, positioned: mx.array) -> mx.array:
        duration_emb = self.w["_proj_input_num_tokens.weight"][12]
        first_input = mx.concatenate([duration_emb[None, :], positioned], axis=0)
        first_output = self._transformer(first_input, "_shared_transformer_model", ROOT_LAYERS)
        return nn.linear(first_output[0:1], self.w["_proj_num_token_output_logit.weight"],
                         self.w["_proj_num_token_output_logit.bias"])[0]

    def _duration_logits_reference(self, global_root: mx.array, local_root: mx.array,
                                   poses: mx.array, global_mask: mx.array,
                                   local_mask: mx.array, pose_mask: mx.array) -> mx.array:
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
             dense_target_mask: mx.array,
             duration_logits: mx.array | None = None) -> tuple[mx.array, mx.array]:
        """``run_root_planner_auto_probe``: ``(duration_logits, global_root_values)``.

        ``global_root_values`` is ``(requested_tokens * 4, 5)`` normalized.  When
        ``duration_logits`` is supplied (and ``root_dedup`` is on) the probe
        transformer is not rebuilt: the caller already ran it, and the accepted
        port discarded this second copy.
        """
        if not nn._opt("root_fast"):
            return self._plan_reference(global_root, local_root, poses, global_mask,
                                        local_mask, pose_mask, requested_tokens,
                                        dense_target, dense_target_mask)
        frame_emb, positioned = self._boundary(global_root, local_root, poses,
                                               global_mask, local_mask, pose_mask)
        reuse = duration_logits is not None and nn._opt("root_dedup")
        if reuse:
            logits = duration_logits
        elif nn._opt("root_compile"):
            logits = self._stage("duration", self._duration_core)((), positioned)
        else:
            logits = self._duration_core(positioned)

        has_frame = mx.clip(global_mask + local_mask + pose_mask, 0.0, 1.0)
        arguments = (positioned, frame_emb, has_frame, dense_target, dense_target_mask,
                     requested_tokens, requested_tokens * 4)
        if nn._opt("root_compile"):
            values = self._stage("tokens_decoder", self._tokens_decoder_core)(
                (requested_tokens,), *arguments)
        else:
            values = self._tokens_decoder_core(*arguments)
        return logits, values

    def _tokens_decoder_core(self, positioned: mx.array, frame_emb: mx.array,
                             has_frame: mx.array, dense_target: mx.array,
                             dense_target_mask: mx.array, requested_tokens: int,
                             frames: int) -> mx.array:
        middle_emb = self._middle_row(requested_tokens)
        second_input = mx.concatenate([middle_emb[None, :], positioned], axis=0)
        second_input = mx.concatenate([second_input, self._position_table(requested_tokens)],
                                      axis=0)
        second_output = self._transformer(second_input, "_root_token_transformer_model",
                                          ROOT_LAYERS)
        root_tokens = second_output[9:9 + requested_tokens]
        decoder_tokens = nn.transpose_contiguous(root_tokens, requested_tokens, ROOT_EMBEDDING)

        absent = self.w["_conv_no_frame_emb"]
        masked_frames = (frame_emb - absent) * has_frame + absent
        middle = mx.broadcast_to(absent[None, :], (frames - 8, ROOT_EMBEDDING))
        dense_frame = mx.concatenate([masked_frames[0:4], middle, masked_frames[4:8]], axis=0)
        return self._decoder(decoder_tokens, dense_frame, dense_target, dense_target_mask,
                             requested_tokens)

    def _plan_reference(self, global_root: mx.array, local_root: mx.array, poses: mx.array,
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
