"""VQ decoder: convolutional decoder producing 413-wide normalized motion.

Port of ``src/decoder.cpp`` (``run_vq_decoder``).
"""

from __future__ import annotations

import mlx.core as mx

from . import nn
from .bundle import Weights
from .root import nearest_upsample_time
from .schema import DILATIONS


class VqDecoder:
    """The ``vq-decoder.gguf`` component."""

    def __init__(self, weights: Weights):
        self.w = weights

    def _conv(self, x: mx.array, name: str, padding: int, dilation: int) -> mx.array:
        return nn.conv1d_time(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"],
                              padding, dilation)

    def _residual_stack(self, x: mx.array, stage: int) -> mx.array:
        """``residual_stack`` in src/decoder.cpp:103-126."""
        hidden = x
        for block, dilation in enumerate(DILATIONS):
            stem = f"decoder.model.{stage}.0.model.{block}."
            branch = self._conv(mx.maximum(hidden, 0.0), stem + "conv1", dilation, dilation)
            branch = self._conv(mx.maximum(branch, 0.0), stem + "conv2", 0, 1)
            hidden = hidden + branch
        return hidden

    def decode(self, quantized: mx.array, external_condition: mx.array,
               target_condition: mx.array, has_target_condition: mx.array,
               positions: int) -> mx.array:
        """Return normalized local motion ``(positions * 4, 413)``."""
        frames = positions * 4
        hidden = mx.maximum(self._conv(quantized, "decoder.model.0", 1, 1), 0.0)  # (512, positions)
        for stage_index in range(2):
            stage = stage_index + 2
            frame_group = 1 << (2 - stage_index)
            stage_positions = positions * (1 << stage_index)
            stage_frames = stage_positions * frame_group

            target_embedding = nn.linear(
                target_condition,
                self.w[f"decoder.target_cond_blocks.{stage_index * 2}.weight"],
                self.w[f"decoder.target_cond_blocks.{stage_index * 2}.bias"])
            hidden_frames = nn.transpose_contiguous(hidden, 512, stage_positions)
            hidden_frames = nn.reshape_ne(hidden_frames, (512 // frame_group, stage_frames))
            hidden_frames = hidden_frames + \
                (mx.maximum(target_embedding, 0.0) - hidden_frames) * has_target_condition
            hidden_frames = nn.reshape_ne(hidden_frames, (512, stage_positions))

            external = nn.reshape_ne(external_condition, (2 * frame_group, stage_positions))
            fused = mx.concatenate([hidden_frames, external], axis=1)
            fused = mx.maximum(
                nn.linear(fused,
                          self.w[f"decoder.external_cond_blocks.{stage_index * 2}.weight"],
                          self.w[f"decoder.external_cond_blocks.{stage_index * 2}.bias"]),
                0.0)
            hidden = nn.transpose_contiguous(fused, stage_positions, 512)
            hidden = self._residual_stack(hidden, stage)
            hidden = nearest_upsample_time(hidden)
            hidden = self._conv(hidden, f"decoder.model.{stage}.2", 1, 1)
        hidden = mx.maximum(self._conv(hidden, "decoder.model.4", 1, 1), 0.0)
        hidden = self._conv(hidden, "decoder.model.6", 1, 1)
        return nn.transpose_contiguous(hidden, 413, frames)
