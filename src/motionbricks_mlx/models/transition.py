"""End-to-end transition: the MLX port of ``run_transition`` (src/planner.cpp:27-225).

The neural stages run in MLX; the scalar bookkeeping around them (feature
normalization, velocity estimation, condition assembly, sampling) is done in
NumPy float32 exactly as the reference does in C++ float, because those are
scalar loops rather than tensor ops.

``run_transition`` returns a :class:`TransitionTrace`-compatible dict so the
result can be diffed stage by stage against the reference runtime's own
``MOTIONBRICKS_TRANSITION_TRACE_DIR`` dumps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from ..data.gguf import GgufFile  # noqa: F401  (re-exported convenience for callers)
from . import motion_rep, nn, sampling
from .bundle import Bundle
from .decoder import VqDecoder
from .pose import PosePlanner
from .root import RootPlanner
from .schema import (BOUNDARY_FRAMES, EXTERNAL_POSE_WIDTH, FPS, GLOBAL_ROOT_WIDTH,
                     INTERNAL_POSE_WIDTH, JOINT_COUNT, LOCAL_ROOT_WIDTH, STATS_EPSILON)

TWO_PI = np.float32(2.0 * np.pi)
PI = np.float32(np.pi)
FPS_F32 = np.float32(FPS)


def wrap_angle(value: float) -> np.float32:
    """``std::remainder`` (src/motion_rep.cpp:21-24): result in ``[-pi, pi]``."""
    return np.float32(np.remainder(np.float32(value) + np.float32(np.pi), TWO_PI) - np.float32(np.pi))


@dataclass
class TransitionConstraints:
    """``transition_constraints`` from src/planner.hpp:18-28."""

    global_root: np.ndarray                     # (8, 5)
    local_root: np.ndarray                      # (8, 4)
    poses: np.ndarray                           # (8, 303)
    has_global_root: np.ndarray                 # (8,) uint8
    has_local_root: np.ndarray                  # (8,) uint8
    has_poses: np.ndarray                       # (8,) uint8
    allowed_tokens: np.ndarray = field(
        default_factory=lambda: np.ones(11, dtype=np.uint8))
    target_root_translations: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 3), dtype=np.float32))
    target_local_rotations_xyzw: np.ndarray = field(
        default_factory=lambda: np.zeros((4, JOINT_COUNT, 4), dtype=np.float32))

    def __post_init__(self) -> None:
        self.global_root = np.ascontiguousarray(self.global_root, dtype=np.float32)
        self.local_root = np.ascontiguousarray(self.local_root, dtype=np.float32)
        self.poses = np.ascontiguousarray(self.poses, dtype=np.float32)
        self.has_global_root = np.ascontiguousarray(self.has_global_root, dtype=np.uint8)
        self.has_local_root = np.ascontiguousarray(self.has_local_root, dtype=np.uint8)
        self.has_poses = np.ascontiguousarray(self.has_poses, dtype=np.uint8)
        self.allowed_tokens = np.ascontiguousarray(self.allowed_tokens, dtype=np.uint8)


@dataclass
class TransitionResult:
    roots: np.ndarray                # (frames, 3)
    rotations_xyzw: np.ndarray       # (frames, 34, 4)
    selected_tokens: int
    trace: dict[str, np.ndarray] = field(default_factory=dict)


class MlxPipeline:
    """Loads a bundle and runs the four inference stages in MLX."""

    def __init__(self, bundle: Bundle):
        self.bundle = bundle
        self.root = RootPlanner(bundle.root)
        self.pose = PosePlanner(bundle.pose)
        self.decoder = VqDecoder(bundle.decoder)
        self.mean = mx.array(bundle.motion_mean.astype(np.float32))
        self.std = mx.array(bundle.motion_std.astype(np.float32))
        self.mean_np = bundle.motion_mean.astype(np.float32)
        self.std_np = bundle.motion_std.astype(np.float32)
        # Per-feature scale vector, built once with the very same scalar
        # expression as :meth:`_scale` so the vectorised host blocks below are
        # bit-identical to the per-element loops they replace.
        self._scale_all = np.array([self._scale(index) for index in range(self.std_np.size)],
                                   dtype=np.float32)
        # Device-side copies of the two statistics vectors, so a stage can stay
        # on the device without an upload per call (round-3 switch
        # ``pred_local_defer``).
        self._mean_device = mx.array(self.mean_np)
        self._scale_device = mx.array(self._scale_all)
        self.parents = bundle.joint_parents.astype(np.int32)
        self.codebook = bundle.vq_codebook
        self.codebook_np = np.asarray(bundle.vq_codebook, dtype=np.float32)

    # -- normalization helpers (src/motion_rep.cpp:110-119) --------------

    def _scale(self, index: int) -> np.float32:
        return np.float32(np.sqrt(self.std_np[index] * self.std_np[index] + STATS_EPSILON))

    def normalize(self, value: float, index: int) -> np.float32:
        return np.float32((np.float32(value) - self.mean_np[index]) / self._scale(index))

    def unnormalize(self, value: float, index: int) -> np.float32:
        return np.float32(np.float32(value) * self._scale(index) + self.mean_np[index])

    def _predicted_local_device(self, predicted_global: mx.array, frames: int,
                                last_row: mx.array | None) -> mx.array:
        """``src/planner.cpp:117-145`` as one device-side expression.

        Same arithmetic as the host block it replaces -- float32 elementwise ops,
        the same ``velocity_from``/``velocity_to`` index arrays, the same
        statistics vectors -- so the only op that can differ is MLX's
        ``arctan2``, which is **not** bit-identical to NumPy's (measured
        2.384e-07, one ulp, on both backends).  ``mx.remainder`` is exact against
        NumPy on this range.  The switch is therefore measured against the token
        gate rather than assumed safe.
        """
        global_scale = self._scale_device[0:GLOBAL_ROOT_WIDTH]
        global_mean = self._mean_device[0:GLOBAL_ROOT_WIDTH]
        unnormalized = predicted_global * global_scale + global_mean
        heading = mx.arctan2(unnormalized[:, 4], unnormalized[:, 3])
        # ``velocity_from[-1] = frames - 2`` / ``velocity_to[-1] = frames - 1``:
        # the last frame reuses the previous pair (src/planner.cpp:126-130).
        source = mx.array(np.concatenate([np.arange(frames - 1, dtype=np.int32),
                                          np.array([frames - 2], np.int32)]))
        target = mx.array(np.concatenate([np.arange(1, frames, dtype=np.int32),
                                          np.array([frames - 1], np.int32)]))
        delta_heading = heading[target] - heading[source]
        raw = mx.stack([
            (mx.remainder(delta_heading + PI, TWO_PI) - PI) * FPS_F32,
            (unnormalized[target, 0] - unnormalized[source, 0]) * FPS_F32,
            (unnormalized[target, 2] - unnormalized[source, 2]) * FPS_F32,
            unnormalized[:, 1],
        ], axis=1)
        predicted = (raw - self._mean_device[5:9]) / self._scale_device[5:9]
        if last_row is not None:
            predicted = mx.concatenate([predicted[0:frames - 1], last_row[None, :]], axis=0)
        return predicted

    # -- transition ------------------------------------------------------

    def run_transition(self, constraints: TransitionConstraints, seed: int,
                       sampling_argmax: bool = False,
                       capture_trace: bool = False,
                       pose_logits: mx.array | None = None,
                       bake=None) -> TransitionResult:
        """Run one transition.

        ``pose_logits`` is the round-3 *bake seam*: when a caller supplies the pose
        planner's output it is used as-is and the stage is skipped, so a batch
        bake can substitute a batched forward while every other stage -- sampling,
        the codebook gather, the decoder, the motion decode -- runs the accepted
        code.  ``None`` (the default) is the accepted path, byte for byte.

        ``bake`` is the round-4 *stage driver* (see
        :mod:`motionbricks_mlx.models.bake`): an object with ``duration``,
        ``plan``, ``pose`` and ``decoder`` methods that stands in for the four
        neural forwards and may raise ``BakeSuspend`` to end the run early so a
        batch can be synchronised at a stage boundary.  ``None`` (the default) is
        again the accepted path: neither seam is entered, so the two are inert
        unless a caller asks for them.
        """
        trace: dict[str, np.ndarray] = {}
        fast = nn._opt("transition_fast")
        initial_x = float(constraints.global_root[0, 0])
        initial_z = float(constraints.global_root[0, 2])

        raw_global = constraints.global_root
        normalized_global = np.empty((BOUNDARY_FRAMES, GLOBAL_ROOT_WIDTH), dtype=np.float32)
        normalized_local = np.empty((BOUNDARY_FRAMES, LOCAL_ROOT_WIDTH), dtype=np.float32)
        normalized_poses = np.empty((BOUNDARY_FRAMES, INTERNAL_POSE_WIDTH), dtype=np.float32)
        if fast:
            # Same elementwise arithmetic as normalize(), evaluated as arrays.
            adjusted = raw_global.copy()
            adjusted[:, 0] = adjusted[:, 0] - np.float32(initial_x)
            adjusted[:, 2] = adjusted[:, 2] - np.float32(initial_z)
            normalized_global[:] = (adjusted - self.mean_np[0:GLOBAL_ROOT_WIDTH]) \
                / self._scale_all[0:GLOBAL_ROOT_WIDTH]
            normalized_local[:] = (constraints.local_root - self.mean_np[5:9]) \
                / self._scale_all[5:9]
            normalized_poses[:, 0] = (raw_global[:, 1] - self.mean_np[8]) / self._scale_all[8]
            normalized_poses[:, 1:] = (constraints.poses - self.mean_np[9:9 + EXTERNAL_POSE_WIDTH]) \
                / self._scale_all[9:9 + EXTERNAL_POSE_WIDTH]
        else:
            for feature in range(GLOBAL_ROOT_WIDTH):
                values = raw_global[:, feature]
                if feature == 0:
                    values = values - np.float32(initial_x)
                elif feature == 2:
                    values = values - np.float32(initial_z)
                normalized_global[:, feature] = [
                    self.normalize(v, feature) for v in values]
            for feature in range(LOCAL_ROOT_WIDTH):
                normalized_local[:, feature] = [
                    self.normalize(v, 5 + feature) for v in constraints.local_root[:, feature]]
            for frame in range(BOUNDARY_FRAMES):
                normalized_poses[frame, 0] = self.normalize(raw_global[frame, 1], 8)
                normalized_poses[frame, 1:] = [
                    self.normalize(v, 9 + index)
                    for index, v in enumerate(constraints.poses[frame])]

        if capture_trace:
            trace["input.global_root_values"] = raw_global.copy()
            trace["input.local_root_values"] = constraints.local_root.copy()
            trace["input.local_poses"] = constraints.poses.copy()
            trace["input.has_global_root_values"] = constraints.has_global_root.astype(np.float32)
            trace["input.has_local_root_values"] = constraints.has_local_root.astype(np.float32)
            trace["input.has_local_poses"] = constraints.has_poses.astype(np.float32)
            trace["input.allowed_pred_num_tokens"] = constraints.allowed_tokens.astype(np.float32)
            trace["root.global_root_values"] = normalized_global.copy()
            trace["root.local_root_values"] = normalized_local.copy()
            trace["root.poses"] = normalized_poses.copy()

        global_mask = constraints.has_global_root.reshape(-1, 1).astype(np.float32)
        local_mask = constraints.has_local_root.reshape(-1, 1).astype(np.float32)
        pose_mask = constraints.has_poses.reshape(-1, 1).astype(np.float32)

        # --- duration selection (src/planner.cpp:79-105) -----------------
        allowed = constraints.allowed_tokens
        enabled = int(np.count_nonzero(allowed))
        duration_logits = np.zeros(1, dtype=np.float32)
        mx_duration_logits = None
        # Round-3 switch ``root_boundary_reuse``: keep the probe's boundary
        # embedding and hand it back to ``RootPlanner.plan`` so it is computed
        # once per step instead of twice (report section 24).
        reuse_boundary = fast and nn._opt("root_boundary_reuse")
        boundary = None
        if enabled == 1:
            tokens = 6 + int(np.flatnonzero(allowed)[0])
        elif enabled > 1:
            if bake is not None:
                mx_duration_logits = bake.duration(
                    mx.array(normalized_global), mx.array(normalized_local),
                    mx.array(normalized_poses), mx.array(global_mask),
                    mx.array(local_mask), mx.array(pose_mask))
            elif reuse_boundary:
                mx_duration_logits, boundary = self.root.probe(
                    mx.array(normalized_global), mx.array(normalized_local),
                    mx.array(normalized_poses), mx.array(global_mask),
                    mx.array(local_mask), mx.array(pose_mask))
            else:
                mx_duration_logits = self.root.duration_logits(
                    mx.array(normalized_global), mx.array(normalized_local),
                    mx.array(normalized_poses), mx.array(global_mask),
                    mx.array(local_mask), mx.array(pose_mask))
            duration_logits = np.asarray(mx_duration_logits)
            best = -np.inf
            tokens = 0
            for index in range(11):
                if allowed[index] != 0 and duration_logits[index] > best:
                    best = float(duration_logits[index])
                    tokens = 6 + index
        else:
            raise ValueError("style disallows every reachable duration")
        if tokens == 0:
            raise ValueError("style disallows every reachable duration")

        # --- root planner -------------------------------------------------
        frames = tokens * 4
        dense_target = np.zeros((frames, GLOBAL_ROOT_WIDTH), dtype=np.float32)
        dense_target_mask = np.zeros((frames, 1), dtype=np.float32)
        if fast:
            boundary_frame = np.array([0, 1, 2, 3, frames - 4, frames - 3,
                                       frames - 2, frames - 1])
            dense_target[boundary_frame] = normalized_global
            dense_target_mask[boundary_frame, 0] = \
                constraints.has_global_root.astype(np.float32)
        else:
            for boundary_index in range(BOUNDARY_FRAMES):
                frame = boundary_index if boundary_index < 4 else frames - 8 + boundary_index
                # run_transition passes the *normalized* boundary vector into the
                # planner (src/planner.cpp:106-107), so the target condition is
                # normalized too (src/root.cpp:464-469).
                dense_target[frame] = normalized_global[boundary_index]
                dense_target_mask[frame, 0] = 1.0 if constraints.has_global_root[boundary_index] else 0.0
        if bake is not None:
            # Round-4 stage driver: a batched root forward replaces this one, and
            # the accepted serial call is not made at all.
            predicted_global = bake.plan(
                mx.array(normalized_global), mx.array(normalized_local),
                mx.array(normalized_poses), mx.array(global_mask), mx.array(local_mask),
                mx.array(pose_mask), tokens, mx.array(dense_target),
                mx.array(dense_target_mask), duration_logits=mx_duration_logits)
        else:
            _, predicted_global = self.root.plan(
                mx.array(normalized_global), mx.array(normalized_local), mx.array(normalized_poses),
                mx.array(global_mask), mx.array(local_mask), mx.array(pose_mask),
                tokens, mx.array(dense_target), mx.array(dense_target_mask),
                duration_logits=mx_duration_logits, boundary=boundary)
        # Round-3 switch ``pred_local_defer``: keep the predicted-local-root block
        # on the device, so the root planner's output is not read back to the host
        # and the synchronisation that used to happen here disappears into the
        # pose planner's own drain.  ``predicted_global_np`` is therefore built
        # only when something actually needs host values (the reference path, or a
        # captured trace).
        defer_local = fast and nn._opt("pred_local_defer")
        need_global_np = capture_trace or not defer_local
        predicted_global_np = (np.asarray(predicted_global, dtype=np.float32).reshape(
            frames, GLOBAL_ROOT_WIDTH) if need_global_np else None)
        if capture_trace:
            trace["root.num_token_logits"] = np.asarray(duration_logits, dtype=np.float32).copy()
            trace["root.pred_global_root_values"] = predicted_global_np.reshape(-1).copy()

        # --- predicted local root (src/planner.cpp:117-145) --------------
        predicted_local_device: mx.array | None = None
        unnormalized_global = np.empty((frames, GLOBAL_ROOT_WIDTH), dtype=np.float32)
        predicted_local = np.empty((frames, LOCAL_ROOT_WIDTH), dtype=np.float32)
        if defer_local:
            last_row = (mx.array(normalized_local[7]) if constraints.has_local_root[7]
                        else None)
            predicted_local_device = self._predicted_local_device(
                predicted_global, frames, last_row)
        elif fast:
            unnormalized_global[:] = predicted_global_np * self._scale_all[0:GLOBAL_ROOT_WIDTH] \
                + self.mean_np[0:GLOBAL_ROOT_WIDTH]
            heading = np.arctan2(unnormalized_global[:, 4],
                                 unnormalized_global[:, 3]).astype(np.float32)
            velocity_from = np.arange(frames, dtype=np.intp)
            velocity_to = velocity_from + 1
            velocity_from[-1] = frames - 2
            velocity_to[-1] = frames - 1
            raw = np.empty((frames, LOCAL_ROOT_WIDTH), dtype=np.float32)
            delta_heading = heading[velocity_to] - heading[velocity_from]
            raw[:, 0] = (
                np.remainder(delta_heading + np.float32(np.pi), TWO_PI) - np.float32(np.pi)
            ) * np.float32(FPS)
            raw[:, 1] = (unnormalized_global[velocity_to, 0]
                         - unnormalized_global[velocity_from, 0]) * np.float32(FPS)
            raw[:, 2] = (unnormalized_global[velocity_to, 2]
                         - unnormalized_global[velocity_from, 2]) * np.float32(FPS)
            raw[:, 3] = unnormalized_global[:, 1]
            predicted_local[:] = (raw - self.mean_np[5:9]) / self._scale_all[5:9]
        else:
            for feature in range(GLOBAL_ROOT_WIDTH):
                unnormalized_global[:, feature] = [
                    self.unnormalize(v, feature) for v in predicted_global_np[:, feature]]
            heading = np.arctan2(unnormalized_global[:, 4], unnormalized_global[:, 3]).astype(np.float32)
            for frame in range(frames):
                if frame + 1 < frames:
                    velocity_from, velocity_to = frame, frame + 1
                else:
                    velocity_from, velocity_to = frame - 1, frame
                raw_values = np.array([
                    wrap_angle(heading[velocity_to] - heading[velocity_from]) * np.float32(FPS),
                    (unnormalized_global[velocity_to, 0] - unnormalized_global[velocity_from, 0])
                    * np.float32(FPS),
                    (unnormalized_global[velocity_to, 2] - unnormalized_global[velocity_from, 2])
                    * np.float32(FPS),
                    unnormalized_global[frame, 1],
                ], dtype=np.float32)
                for feature in range(LOCAL_ROOT_WIDTH):
                    predicted_local[frame, feature] = self.normalize(raw_values[feature], 5 + feature)
        if not defer_local and constraints.has_local_root[7]:
            predicted_local[frames - 1] = normalized_local[7]
        if capture_trace and not defer_local:
            trace["root.pred_local_root_values"] = predicted_local.reshape(-1).copy()

        # --- pose conditions (src/planner.cpp:147-166) -------------------
        pose_condition = np.zeros((frames, INTERNAL_POSE_WIDTH), dtype=np.float32)
        has_pose_condition = np.zeros(frames, dtype=np.uint8)
        if fast:
            condition_frames = np.array([0, 1, 2, 3, frames - 4, frames - 3,
                                         frames - 2, frames - 1])
            pose_condition[condition_frames] = normalized_poses
            has_pose_condition[condition_frames] = constraints.has_poses
        else:
            for boundary in range(BOUNDARY_FRAMES):
                frame = boundary if boundary < 4 else frames - 8 + boundary
                pose_condition[frame] = normalized_poses[boundary]
                has_pose_condition[frame] = constraints.has_poses[boundary]
        if defer_local:
            pose_root_argument = mx.stack([predicted_global[:, 0], predicted_global[:, 2],
                                           predicted_global[:, 3], predicted_global[:, 4]],
                                          axis=1)
            pose_root = pose_root_argument
        else:
            pose_root = np.stack([
                predicted_global_np[:, 0], predicted_global_np[:, 2],
                predicted_global_np[:, 3], predicted_global_np[:, 4]], axis=1).astype(np.float32)
            pose_root_argument = mx.array(pose_root)
        if capture_trace and not defer_local:
            trace["pose.root_condition"] = pose_root.reshape(-1).copy()
        if capture_trace:
            trace["pose.pose_condition"] = pose_condition.reshape(-1).copy()
            trace["pose.has_pose_condition"] = has_pose_condition.astype(np.float32)

        # --- pose planner + sampling (src/planner.cpp:167-184) -----------
        pose_tokens = np.full((tokens, 8), 10, dtype=np.int32)
        pose_condition_device = mx.array(pose_condition)
        has_pose_device = mx.array(has_pose_condition.astype(np.float32)[:, None])
        if bake is not None:
            logits = bake.pose(pose_tokens, pose_root_argument, pose_condition_device,
                               has_pose_device, tokens, tokens)
        else:
            logits = (self.pose.plan(pose_tokens, pose_root_argument, pose_condition_device,
                                     has_pose_device, tokens, tokens)
                      if pose_logits is None
                      else pose_logits)
        logits_np = np.asarray(PosePlanner.reshape_logits(logits), dtype=np.float32)
        uniform = None
        if not sampling_argmax:
            uniform = sampling.uniforms(seed, logits_np.size).reshape(logits_np.shape)
        selected = sampling.sample_pose_tokens(logits_np, uniform, 10)
        if capture_trace:
            trace["pose.logits"] = logits_np.reshape(-1).copy()
            trace["pose.tokens"] = selected.astype(np.float32).copy()
            trace["pose.sampling_uniforms"] = (
                np.zeros(logits_np.size, dtype=np.float32) if uniform is None
                else uniform.reshape(-1).copy())
        if capture_trace and defer_local:
            # One drain serves all three: the pose logits above already forced the
            # device graph that produced them, so these reads are free.
            local_np = np.asarray(predicted_local_device, dtype=np.float32)
            trace["root.pred_local_root_values"] = local_np.reshape(-1).copy()
            trace["pose.root_condition"] = np.asarray(
                pose_root_argument, dtype=np.float32).reshape(-1).copy()

        # --- VQ codebook lookup (src/planner.cpp:186-198) ----------------
        if fast:
            selected_2d = selected.reshape(tokens, 8)
            entries = self.codebook_np[np.arange(8)[None, :], selected_2d]   # (tokens, 8, 32)
            quantized = entries.transpose(1, 2, 0).reshape(256, tokens)
        else:
            quantized = np.empty((256, tokens), dtype=np.float32)
            for position in range(tokens):
                for head in range(8):
                    entry = int(selected[position * 8 + head])
                    quantized[head * 32:(head + 1) * 32, position] = self.codebook_np[head, entry]
        if capture_trace:
            trace["decoder.quantized"] = quantized.reshape(-1).copy()

        if defer_local:
            external_argument = mx.stack([predicted_local_device[:, 1],
                                          predicted_local_device[:, 2]], axis=1)
        else:
            external_condition = np.stack(
                [predicted_local[:, 1], predicted_local[:, 2]], axis=1).astype(np.float32)
            external_argument = mx.array(external_condition)
        if capture_trace:
            if defer_local:
                trace["decoder.external_condition"] = np.asarray(
                    external_argument, dtype=np.float32).reshape(-1).copy()
            else:
                trace["decoder.external_condition"] = external_condition.reshape(-1).copy()
            trace["decoder.has_target_condition"] = has_pose_condition.astype(np.float32)

        # --- VQ decoder ---------------------------------------------------
        # Round-3 switch ``decoder_defer``: when nothing needs the decoder output
        # on the host, hand it straight to ``decode_motion`` instead of reading it
        # back and re-uploading it.  Same values either way -- this removes a
        # host synchronisation, not a computation.
        if bake is not None:
            decoded_device = bake.decoder(
                mx.array(quantized), external_argument, pose_condition_device,
                has_pose_device, tokens)
        else:
            decoded_device = self.decoder.decode(
                mx.array(quantized), external_argument, pose_condition_device,
                has_pose_device, tokens)
        if capture_trace or not nn._opt("decoder_defer"):
            decoded = np.asarray(decoded_device, dtype=np.float32).reshape(frames, 413)
            decoded_argument = mx.array(decoded)
            if capture_trace:
                trace["decoder.output"] = decoded.reshape(-1).copy()
        else:
            decoded_argument = decoded_device

        # --- motion representation decode (src/motion_rep.cpp:252-303) ---
        roots, rotations = motion_rep.decode_motion(
            decoded_argument, self.mean, self.std, self.parents, initial_x, initial_z)
        roots_np = np.asarray(roots, dtype=np.float32)
        rotations_np = np.asarray(rotations, dtype=np.float32)
        if capture_trace:
            trace["decoded.roots"] = roots_np.reshape(-1).copy()
        return TransitionResult(roots_np, rotations_np, tokens, trace)
