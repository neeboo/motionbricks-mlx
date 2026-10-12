"""Batch bake: ``B`` independent scenarios through one batched pose forward.

Why only the pose stage is batched
----------------------------------
Round 1 (`docs/optimization-report.md` section 5) established that the pose
transformer is the one stage with independent work to batch, and measured
2.37x / 3.12x / 6.05x throughput at ``B`` = 2 / 4 / 8 on it.  The root planner's
two transformers are a strict dependency chain and the decoder's two stages are
dependent too, so a bake here batches **the pose stage only** and runs every
other stage per scenario, in the accepted code.  The end-to-end bake gain is
therefore bounded by the pose stage's share of a step, and is reported as such.

How the two arms stay comparable
--------------------------------
Each scenario is planned **alone** through the ordinary pipeline (the reference
arm, i.e. exactly ``B = 1``), and the pose planner's own inputs are recorded on
the way past.  The batched arm feeds those recorded inputs through
:meth:`PosePlanner.plan_batch` and hands each row back to the pipeline through
the ``pose_logits`` seam, so sampling, the codebook gather, the decoder and the
motion decode all run the accepted code on both arms.  The only difference
between the arms is which forward produced the pose logits.

Scenarios that selected different durations cannot share a batch (the compiled
variants and the GGML layout reshapes are shape-keyed), so the batched arm groups
them by token count and runs one batched forward per group.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from ..data import agent as A
from ..data.style import Style


@dataclass
class Scenario:
    """One independent bake input: a style, a seed and a command."""

    label: str
    style: Style
    seed: int
    movement: tuple[float, float, float] = (0.0, 0.0, 0.0)
    facing: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def command(self) -> A.Command:
        return A.Command(style=self.style, movement=self.movement,
                         facing=self.facing, seed=self.seed)


@dataclass
class Run:
    """One scenario's plan, as produced by one arm."""

    label: str
    selected: int
    tokens: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    uniforms: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    roots: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    rotations: np.ndarray = field(default_factory=lambda: np.zeros((0, 34, 4), np.float32))


@dataclass
class Recorded:
    """A scenario plus the pose planner inputs the reference arm consumed."""

    scenario: Scenario
    run: Run
    constraints: object
    tokens: int
    pose_root: np.ndarray          # (frames, 4)
    pose_condition: np.ndarray     # (frames, INTERNAL_POSE_WIDTH)
    has_pose_condition: np.ndarray  # (frames, 1)


def _new_agent(pipeline, bundle) -> A.Agent:
    return A.Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)


def record_serial(pipeline, bundle, scenarios: list[Scenario],
                  capture_trace: bool = True) -> tuple[list[Recorded], list[float]]:
    """The reference arm: every scenario planned alone, with its pose inputs kept.

    Returns the per-scenario records and each scenario's own wall time in
    milliseconds (the "single-scenario p50" the bake report quotes).
    """
    import time

    records: list[Recorded] = []
    times: list[float] = []
    for scenario in scenarios:
        recorded: dict[str, object] = {}
        original = pipeline.pose.plan

        def spy(pose_tokens, local_root_values, pose_condition, has_pose_condition,
                positions, num_tokens, _recorded=recorded, _original=original):
            _recorded["pose_root"] = np.asarray(local_root_values, dtype=np.float32)
            _recorded["pose_condition"] = np.asarray(pose_condition, dtype=np.float32)
            _recorded["has_pose_condition"] = np.asarray(has_pose_condition,
                                                         dtype=np.float32)
            _recorded["tokens"] = int(num_tokens)
            return _original(pose_tokens, local_root_values, pose_condition,
                             has_pose_condition, positions, num_tokens)

        pipeline.pose.plan = spy
        agent = _new_agent(pipeline, bundle)
        agent.reset(scenario.style)
        started = time.perf_counter()
        try:
            motion, info = agent.plan(scenario.command(), scenario.style,
                                      capture_trace=capture_trace)
            mx.eval(mx.array(motion.roots.sum()))
        finally:
            pipeline.pose.plan = original
        times.append((time.perf_counter() - started) * 1000.0)
        trace = info.get("trace", {})
        records.append(Recorded(
            scenario=scenario,
            run=Run(scenario.label, int(info["selected_tokens"]),
                    np.asarray(trace.get("pose.tokens", np.zeros(0)), np.float32),
                    np.asarray(trace.get("pose.sampling_uniforms", np.zeros(0)), np.float32),
                    motion.roots.copy(), motion.rotations_xyzw.copy()),
            constraints=None,
            tokens=int(recorded["tokens"]),
            pose_root=recorded["pose_root"],                 # type: ignore[arg-type]
            pose_condition=recorded["pose_condition"],       # type: ignore[arg-type]
            has_pose_condition=recorded["has_pose_condition"],  # type: ignore[arg-type]
        ))
    return records, times


def group_by_tokens(records: list[Recorded]) -> dict[int, list[int]]:
    """``{token count: [indices]}`` -- one batch per shape, never mixed."""
    groups: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.tokens, []).append(index)
    return groups


def batched_logits(pipeline, records: list[Recorded]) -> list[mx.array]:
    """One batched pose forward per token-count group; one row per scenario."""
    outputs: list[mx.array | None] = [None] * len(records)
    for tokens, indices in group_by_tokens(records).items():
        stacked_tokens = mx.array(np.stack([
            np.full((tokens, 8), 10, np.int32) for _ in indices]))
        stacked_root = mx.array(np.stack([records[i].pose_root for i in indices]))
        stacked_condition = mx.array(np.stack(
            [records[i].pose_condition for i in indices]))
        stacked_has = mx.array(np.stack(
            [records[i].has_pose_condition for i in indices]))
        logits = pipeline.pose.plan_batch(stacked_tokens, stacked_root,
                                          stacked_condition, stacked_has,
                                          tokens, tokens)
        mx.eval(logits)
        for row, index in enumerate(indices):
            outputs[index] = logits[row]
    return outputs  # type: ignore[return-value]


def rerun_batched(pipeline, bundle, records: list[Recorded],
                  logits: list[mx.array], capture_trace: bool = True) -> list[Run]:
    """Feed each batched row back through the accepted pipeline."""
    runs: list[Run] = []
    for record, row in zip(records, logits):
        agent = _new_agent(pipeline, bundle)
        agent.reset(record.scenario.style)
        motion, info = agent.plan(record.scenario.command(), record.scenario.style,
                                  capture_trace=capture_trace, pose_logits=row)
        mx.eval(mx.array(motion.roots.sum()))
        trace = info.get("trace", {})
        runs.append(Run(record.scenario.label, int(info["selected_tokens"]),
                        np.asarray(trace.get("pose.tokens", np.zeros(0)), np.float32),
                        np.asarray(trace.get("pose.sampling_uniforms", np.zeros(0)), np.float32),
                        motion.roots.copy(), motion.rotations_xyzw.copy()))
    return runs


def compare(reference: list[Run], candidate: list[Run]) -> dict:
    """Per-scenario identity and worst deviation of a batched bake vs B = 1."""
    rows = []
    for left, right in zip(reference, candidate):
        angle = float(np.degrees(2.0 * np.arccos(np.clip(
            np.abs(np.sum(np.asarray(left.rotations, np.float64)
                          * np.asarray(right.rotations, np.float64), axis=-1)),
            0.0, 1.0))).max())
        rows.append({
            "label": left.label,
            "tokens_identical": bool(left.tokens.size and
                                     np.array_equal(left.tokens, right.tokens)),
            "uniforms_identical": bool(left.uniforms.size and
                                       np.array_equal(left.uniforms, right.uniforms)),
            "selected_identical": left.selected == right.selected,
            "differing_tokens": int(np.count_nonzero(left.tokens != right.tokens)),
            "tokens": int(left.tokens.size),
            "worst_roots_m": float(np.abs(left.roots - right.roots).max()),
            "worst_rotations": float(np.abs(left.rotations - right.rotations).max()),
            "worst_angle_deg": angle,
        })
    return {
        "scenarios": len(rows),
        "token_identical": sum(row["tokens_identical"] for row in rows),
        "uniform_identical": sum(row["uniforms_identical"] for row in rows),
        "selected_identical": sum(row["selected_identical"] for row in rows),
        "differing_tokens": sum(row["differing_tokens"] for row in rows),
        "total_tokens": sum(row["tokens"] for row in rows),
        "worst_roots_m": max((row["worst_roots_m"] for row in rows), default=0.0),
        "worst_rotations": max((row["worst_rotations"] for row in rows), default=0.0),
        "worst_angle_deg": max((row["worst_angle_deg"] for row in rows), default=0.0),
        "rows": rows,
    }
