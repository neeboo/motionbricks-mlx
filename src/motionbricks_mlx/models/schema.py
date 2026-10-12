"""Machine-checkable transcription of the reference model definition.

Everything in this module is copied from the read-only reference checkout
``~/dev/motion-bricks-cpp/repo`` (upstream ``localai-org/motion-bricks.cpp``,
Apache-2.0).  ``source`` strings are ``file:line`` references into that
checkout so each number can be re-derived without trusting this port.

Nothing here is invented: if a shape or constant disagrees with the reference
the port must change, not this table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Scalar hyper-parameters
# --------------------------------------------------------------------------

FPS = 30.0                                  # src/motion_rep.cpp:14
STATS_EPSILON = 1.0e-5                      # src/motion_rep.cpp:15
JOINT_COUNT = 34                            # src/motion_rep.hpp:16
BOUNDARY_FRAME_COUNT = 4                    # src/motion_rep.hpp:17
GLOBAL_ROOT_WIDTH = 5                       # src/motion_rep.hpp:18
LOCAL_ROOT_WIDTH = 4                        # src/motion_rep.hpp:19
EXTERNAL_POSE_WIDTH = 303                   # src/motion_rep.hpp:20
INTERNAL_POSE_WIDTH = 304                   # src/motion_rep.hpp:21
LOCAL_MOTION_WIDTH = 413                    # src/motion_rep.hpp:22
BOUNDARY_FRAMES = 8                         # src/root.cpp:366 (4 start + 4 end)
DILATIONS = (27, 9, 3, 1)                   # src/root.cpp:122, src/decoder.cpp:106
FRAMES_PER_POSITION = 4                     # src/pose.cpp:188
ROLLOUT = 6                                 # root.cpp:367 requested_tokens >= 6
MAX_TOKENS = 16                             # root.cpp:367 requested_tokens <= 16
POSE_HEADS = 8                              # src/pose.cpp:187
POSE_TOKEN_CHOICES = 10                     # src/pose.cpp:187 (logits / 10)
POSE_TOKEN_WIDTH = 32                       # latent per head (src/pose.cpp:225-227)
VQ_CODEBOOK_SHAPE = (8, 10, 32)             # src/planner.cpp:187
MOTION_STATS_WIDTH = 418                    # src/model.cpp:201-202

# Root/pose transformer geometry.
ROOT_EMBEDDING = 512                        # src/root.cpp:65
ROOT_HEADS = 16                             # src/root.cpp:65
ROOT_HEAD_WIDTH = 32                        # src/root.cpp:65
ROOT_LAYERS = 3                             # src/root.cpp:419 / 433
ROOT_FIRST_SEQUENCE = 9                     # src/root.cpp:419 (1 duration + 8 frames)
POSE_EMBEDDING = 1024                       # src/pose.cpp:54
POSE_TRANSFORMER_HEADS = 16                 # src/pose.cpp:55
POSE_LAYERS = 16                            # src/pose.cpp:266
DECODER_LATENT_WIDTH = 256                  # src/decoder.cpp:167
DECODER_EXTERNAL_WIDTH = 2                  # src/decoder.cpp:168
DECODER_TARGET_WIDTH = 304                  # src/decoder.cpp:169
DECODER_OUTPUT_WIDTH = 413                  # src/decoder.cpp:170
ROOT_DECODER_WIDTH = 512                    # src/root.cpp:512-channel conv stack
DURATION_CLASSES = 11                       # planner.cpp:95 (indices 0..10 -> 6..16 tokens)
DURATION_LOGIT_WIDTH = 12                   # src/root.cpp:489
UNIFORM_MAPPING = "splitmix64>>40 * 2**-24"  # src/sampling.cpp:12


@dataclass(frozen=True)
class WeightSpec:
    component: str
    name: str
    shape: tuple[int, ...]
    source: str


def _transformer_specs(component: str, prefix: str, layers: int,
                       embedding: int, feed_forward: int, source: str) -> list[WeightSpec]:
    specs: list[WeightSpec] = []
    for layer in range(layers):
        stem = f"{prefix}layers.{layer}."
        specs += [
            WeightSpec(component, stem + "self_attn.in_proj_weight",
                       (3 * embedding, embedding), source),
            WeightSpec(component, stem + "self_attn.in_proj_bias", (3 * embedding,), source),
            # The on-disk name for the longest of these is compacted to
            # "attn.out_proj.weight" by scripts/convert_to_gguf.py; the
            # runtime's fallback in src/neural_runtime.cpp:272-276 handles it.
            WeightSpec(component, stem + "self_attn.out_proj.weight",
                       (embedding, embedding), source),
            WeightSpec(component, stem + "self_attn.out_proj.bias", (embedding,), source),
            WeightSpec(component, stem + "norm1.weight", (embedding,), source),
            WeightSpec(component, stem + "norm1.bias", (embedding,), source),
            WeightSpec(component, stem + "norm2.weight", (embedding,), source),
            WeightSpec(component, stem + "norm2.bias", (embedding,), source),
            WeightSpec(component, stem + "linear1.weight", (feed_forward, embedding), source),
            WeightSpec(component, stem + "linear1.bias", (feed_forward,), source),
            WeightSpec(component, stem + "linear2.weight", (embedding, feed_forward), source),
            WeightSpec(component, stem + "linear2.bias", (embedding,), source),
        ]
    return specs


def _residual_specs(component: str, prefix: str, source: str) -> list[WeightSpec]:
    specs: list[WeightSpec] = []
    for stage in (2, 3):
        for block in range(4):
            stem = f"{prefix}model.{stage}.0.model.{block}."
            specs += [
                WeightSpec(component, stem + "conv1.weight", (512, 512, 3), source),
                WeightSpec(component, stem + "conv1.bias", (512,), source),
                WeightSpec(component, stem + "conv2.weight", (512, 512, 1), source),
                WeightSpec(component, stem + "conv2.bias", (512,), source),
            ]
    return specs


def _root_specs() -> list[WeightSpec]:
    src = "src/root.cpp"
    specs = [
        WeightSpec("root", "_proj_local_pose.weight", (256, 304), src + ":400"),
        WeightSpec("root", "_proj_local_pose.bias", (256,), src + ":400"),
        WeightSpec("root", "_no_local_pose_emb", (256,), src + ":401"),
        WeightSpec("root", "_proj_local_root_value.weight", (64, 4), src + ":403"),
        WeightSpec("root", "_proj_local_root_value.bias", (64,), src + ":403"),
        WeightSpec("root", "_no_local_root_emb", (64,), src + ":404"),
        WeightSpec("root", "_proj_global_root_value.weight", (64, 5), src + ":406"),
        WeightSpec("root", "_proj_global_root_value.bias", (64,), src + ":406"),
        WeightSpec("root", "_no_global_root_emb", (64,), src + ":407"),
        WeightSpec("root", "_input_position_emb.weight", (8, 512), src + ":414"),
        WeightSpec("root", "_proj_input_num_tokens.weight", (13, 512), src + ":416"),
        WeightSpec("root", "_proj_num_token_output_logit.weight", (12, 512), src + ":422"),
        WeightSpec("root", "_proj_num_token_output_logit.bias", (12,), src + ":422"),
        WeightSpec("root", "_middle_token_emb.weight", (12, 512), src + ":425"),
        WeightSpec("root", "_position_emb.embed", (16, 1, 512), src + ":427"),
        WeightSpec("root", "_conv_no_frame_emb", (512,), src + ":440"),
        WeightSpec("root", "_conv_output.model.0.weight", (512, 512, 3), src + ":149"),
        WeightSpec("root", "_conv_output.model.0.bias", (512,), src + ":149"),
        WeightSpec("root", "_conv_output.target_cond_blocks.0.weight", (128, 5), src + ":156"),
        WeightSpec("root", "_conv_output.target_cond_blocks.0.bias", (128,), src + ":156"),
        WeightSpec("root", "_conv_output.target_cond_blocks.2.weight", (256, 5), src + ":156"),
        WeightSpec("root", "_conv_output.target_cond_blocks.2.bias", (256,), src + ":156"),
        WeightSpec("root", "_conv_output.external_cond_blocks.0.weight", (512, 2560), src + ":166"),
        WeightSpec("root", "_conv_output.external_cond_blocks.0.bias", (512,), src + ":166"),
        WeightSpec("root", "_conv_output.external_cond_blocks.2.weight", (512, 1536), src + ":166"),
        WeightSpec("root", "_conv_output.external_cond_blocks.2.bias", (512,), src + ":166"),
        WeightSpec("root", "_conv_output.model.2.2.weight", (512, 512, 3), src + ":173"),
        WeightSpec("root", "_conv_output.model.2.2.bias", (512,), src + ":173"),
        WeightSpec("root", "_conv_output.model.3.2.weight", (512, 512, 3), src + ":173"),
        WeightSpec("root", "_conv_output.model.3.2.bias", (512,), src + ":173"),
        WeightSpec("root", "_conv_output.model.4.weight", (512, 512, 3), src + ":177"),
        WeightSpec("root", "_conv_output.model.4.bias", (512,), src + ":177"),
        WeightSpec("root", "_conv_output.model.6.weight", (5, 512, 3), src + ":180"),
        WeightSpec("root", "_conv_output.model.6.bias", (5,), src + ":180"),
    ]
    for stem in ("_proj_start_input", "_proj_end_input"):
        specs += [
            WeightSpec("root", f"{stem}.fc_layers.0.weight", (512, 384), src + ":411"),
            WeightSpec("root", f"{stem}.fc_layers.0.bias", (512,), src + ":411"),
            WeightSpec("root", f"{stem}.fc_layers.1.weight", (512, 512), src + ":411"),
            WeightSpec("root", f"{stem}.fc_layers.1.bias", (512,), src + ":411"),
            WeightSpec("root", f"{stem}.forward_projection.weight", (512, 512), src + ":411"),
            WeightSpec("root", f"{stem}.forward_projection.bias", (512,), src + ":411"),
        ]
    specs += _transformer_specs("root", "_shared_transformer_model.", 3, 512, 2048, src + ":418")
    specs += _transformer_specs("root", "_root_token_transformer_model.", 3, 512, 2048, src + ":432")
    specs += _residual_specs("root", "_conv_output.", src)
    return specs


def _pose_specs() -> list[WeightSpec]:
    src = "src/pose.cpp"
    specs = [
        WeightSpec("pose", "_pose_token_emb.weight", (88, 32), src + ":225"),
        WeightSpec("pose", "_proj_pose_token_emb.fc_layers.0.weight", (640, 256), src + ":229"),
        WeightSpec("pose", "_proj_pose_token_emb.fc_layers.0.bias", (640,), src + ":229"),
        WeightSpec("pose", "_proj_pose_token_emb.fc_layers.1.weight", (640, 640), src + ":231"),
        WeightSpec("pose", "_proj_pose_token_emb.fc_layers.1.bias", (640,), src + ":231"),
        WeightSpec("pose", "_proj_pose_token_emb.forward_projection.weight", (640, 640), src + ":234"),
        WeightSpec("pose", "_proj_pose_token_emb.forward_projection.bias", (640,), src + ":234"),
        WeightSpec("pose", "_proj_local_pose.weight", (160, 304), src + ":240"),
        WeightSpec("pose", "_proj_local_pose.bias", (160,), src + ":240"),
        WeightSpec("pose", "_proj_local_root_values.weight", (256, 16), src + ":249"),
        WeightSpec("pose", "_proj_local_root_values.bias", (256,), src + ":249"),
        WeightSpec("pose", "_proj_num_valid_positions.weight", (11, 128), src + ":252"),
        WeightSpec("pose", "_proj_input.0.weight", (1024, 1024), src + ":258"),
        WeightSpec("pose", "_proj_input.0.bias", (1024,), src + ":258"),
        WeightSpec("pose", "_position_emb.embed", (16, 1, 1024), src + ":260"),
        WeightSpec("pose", "_proj_pose_output_logit.weight", (80, 1024), src + ":272"),
        WeightSpec("pose", "_proj_pose_output_logit.bias", (80,), src + ":272"),
    ]
    specs += _transformer_specs("pose", "_transformer_model.", 16, 1024, 2048, src + ":267")
    return specs


def _decoder_specs() -> list[WeightSpec]:
    src = "src/decoder.cpp"
    specs = [
        WeightSpec("vq-decoder", "decoder.model.0.weight", (512, 256, 3), src + ":197"),
        WeightSpec("vq-decoder", "decoder.model.0.bias", (512,), src + ":197"),
        WeightSpec("vq-decoder", "decoder.target_cond_blocks.0.weight", (128, 304), src + ":208"),
        WeightSpec("vq-decoder", "decoder.target_cond_blocks.0.bias", (128,), src + ":208"),
        WeightSpec("vq-decoder", "decoder.target_cond_blocks.2.weight", (256, 304), src + ":208"),
        WeightSpec("vq-decoder", "decoder.target_cond_blocks.2.bias", (256,), src + ":208"),
        WeightSpec("vq-decoder", "decoder.external_cond_blocks.0.weight", (512, 520), src + ":222"),
        WeightSpec("vq-decoder", "decoder.external_cond_blocks.0.bias", (512,), src + ":222"),
        WeightSpec("vq-decoder", "decoder.external_cond_blocks.2.weight", (512, 516), src + ":222"),
        WeightSpec("vq-decoder", "decoder.external_cond_blocks.2.bias", (512,), src + ":222"),
        WeightSpec("vq-decoder", "decoder.model.2.2.weight", (512, 512, 3), src + ":231"),
        WeightSpec("vq-decoder", "decoder.model.2.2.bias", (512,), src + ":231"),
        WeightSpec("vq-decoder", "decoder.model.3.2.weight", (512, 512, 3), src + ":231"),
        WeightSpec("vq-decoder", "decoder.model.3.2.bias", (512,), src + ":231"),
        WeightSpec("vq-decoder", "decoder.model.4.weight", (512, 512, 3), src + ":236"),
        WeightSpec("vq-decoder", "decoder.model.4.bias", (512,), src + ":236"),
        WeightSpec("vq-decoder", "decoder.model.6.weight", (413, 512, 3), src + ":241"),
        WeightSpec("vq-decoder", "decoder.model.6.bias", (413,), src + ":241"),
        WeightSpec("vq-decoder", "quantizer.vq._codebook.embed", VQ_CODEBOOK_SHAPE, "src/model.cpp:285"),
    ]
    specs += _residual_specs("vq-decoder", "decoder.", src)
    return specs


SUPPORT_SPECS = [
    WeightSpec("support", "joint_parents", (34,), "src/model.cpp:200"),
    WeightSpec("support", "motion_mean", (418,), "src/model.cpp:201"),
    WeightSpec("support", "motion_std", (418,), "src/model.cpp:202"),
    WeightSpec("support", "neutral_joints", (34, 3), "src/model.cpp:199"),
]

WEIGHT_SPECS: dict[str, list[WeightSpec]] = {
    "root": _root_specs(),
    "pose": _pose_specs(),
    "vq-decoder": _decoder_specs(),
    "support": SUPPORT_SPECS,
}


def check_bundle(bundle, *, check_values: bool = True) -> list[str]:
    """Return a list of human-readable schema problems (empty means aligned)."""
    problems: list[str] = []
    for component, specs in WEIGHT_SPECS.items():
        weights = bundle.weights(component)
        for spec in specs:
            try:
                resolved = weights.resolve(spec.name)
            except Exception as error:  # noqa: BLE001 - report, never raise here
                problems.append(f"{component}:{spec.name} unresolved ({error})")
                continue
            shape = tuple(weights.tensors[resolved].shape)
            if shape != spec.shape:
                problems.append(
                    f"{component}:{spec.name} -> {resolved}: shape {shape} != {spec.shape}"
                    f" (from {spec.source})"
                )
    if not check_values:
        return problems
    support = bundle.support
    parents = np.asarray(support["joint_parents"]).reshape(-1)
    if parents.shape != (JOINT_COUNT,):
        problems.append(f"support:joint_parents has {parents.shape} entries")
    else:
        for index, parent in enumerate(parents.tolist()):
            if index == 0 and parent != -1:
                problems.append("support:joint_parents[0] must be -1")
            if index != 0 and not (0 <= parent < index):
                problems.append(f"support:joint_parents[{index}] = {parent} is not an ancestor")
    deviations = np.asarray(support["motion_std"]).reshape(-1)
    if deviations.shape != (MOTION_STATS_WIDTH,):
        problems.append(f"support:motion_std has {deviations.shape}")
    elif (deviations < 0).any() or not np.isfinite(deviations).all():
        problems.append("support:motion_std has negative or non-finite entries")
    return problems
