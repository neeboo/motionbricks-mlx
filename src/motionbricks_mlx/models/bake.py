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

import time
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from . import motion_rep
from . import nn
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
    # Round-4 stage snapshots (only filled when a caller asks for them), so the
    # campaign can quote the perturbation that reaches each discrete decision.
    duration_logits: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    root_values: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    pose_logits: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    decoded: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))


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
                  capture_trace: bool = True,
                  capture_stages: bool = False) -> tuple[list[Recorded], list[float]]:
    """The reference arm: every scenario planned alone, with its pose inputs kept.

    Returns the per-scenario records and each scenario's own wall time in
    milliseconds (the "single-scenario p50" the bake report quotes).

    ``capture_stages`` additionally snapshots the four neural stage outputs, which
    forces a host read per stage and is therefore off by default: a timed arm
    must not pay for instrumentation.  The round-4 campaign does its timing
    without it and its snapshots in a separate untimed pass.
    """
    import time

    records: list[Recorded] = []
    times: list[float] = []
    for scenario in scenarios:
        recorded: dict[str, object] = {}
        snapshots: dict[str, np.ndarray] = {}
        restore: list[tuple[object, str, object]] = []

        def capture(key: str, target, name: str, last: bool = False) -> None:
            """Wrap a stage method so its output is snapshotted, then restored."""
            method = getattr(target, name)

            def stage_spy(*args, **kwargs):
                result = method(*args, **kwargs)
                values = result if isinstance(result, tuple) else (result,)
                snapshots[key] = np.asarray(
                    values[-1] if last else values[0], dtype=np.float32).reshape(-1).copy()
                return result

            setattr(target, name, stage_spy)
            restore.append((target, name, method))

        if capture_stages:
            capture("duration_logits", pipeline.root, "duration_logits")
            capture("root_values", pipeline.root, "plan", last=True)
            capture("pose_logits", pipeline.pose, "plan")
            capture("decoded", pipeline.decoder, "decode")

        # ``base_pose`` is whatever ``pose.plan`` is now: the capture wrapper when
        # snapshots are on, the accepted method otherwise.  The input spy below
        # delegates to it, so both kinds of instrumentation compose.
        base_pose = pipeline.pose.plan

        def spy(pose_tokens, local_root_values, pose_condition, has_pose_condition,
                positions, num_tokens, _recorded=recorded, _original=base_pose):
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
        # The same constraints :meth:`Agent.plan` builds just below, kept so the
        # campaign can read the style's allowed durations for the duration margin.
        _, scenario_constraints = agent.constraints_for(scenario.command(), scenario.style)
        started = time.perf_counter()
        try:
            motion, info = agent.plan(scenario.command(), scenario.style,
                                      capture_trace=capture_trace)
            mx.eval(mx.array(motion.roots.sum()))
        finally:
            for target, name, method in restore:
                setattr(target, name, method)
        times.append((time.perf_counter() - started) * 1000.0)
        trace = info.get("trace", {})
        records.append(Recorded(
            scenario=scenario,
            run=Run(scenario.label, int(info["selected_tokens"]),
                    np.asarray(trace.get("pose.tokens", np.zeros(0)), np.float32),
                    np.asarray(trace.get("pose.sampling_uniforms", np.zeros(0)), np.float32),
                    motion.roots.copy(), motion.rotations_xyzw.copy(),
                    snapshots.get("duration_logits", np.zeros(0, np.float32)),
                    snapshots.get("root_values", np.zeros(0, np.float32)),
                    snapshots.get("pose_logits", np.zeros(0, np.float32)),
                    snapshots.get("decoded", np.zeros(0, np.float32))),
            constraints=scenario_constraints,
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
                        motion.roots.copy(), motion.rotations_xyzw.copy(),
                        pose_logits=np.asarray(row, dtype=np.float32).reshape(-1).copy()))
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


# ---------------------------------------------------------------------------
# Round 4: batching the root and decoder stages
# ---------------------------------------------------------------------------
# Round 3 batched the pose stage only and measured the consequence: the bake's
# ceiling is the *share of a step that is batched*, not the batch size.  Round 4
# batches the two stages that follow the pose planner through the same
# ``mx.compile`` + ``mx.vmap`` construction, and the three stages have a
# dependency chain through host-side glue, so they cannot be substituted by
# round 3's one-parameter seam: the pose planner's inputs are a function of the
# *batched* root planner's output, and the decoder's inputs are a function of the
# *batched* pose planner's output.
#
# The driver below therefore synchronises the batch at stage boundaries.  Each
# scenario is run through ``run_transition`` with the driver standing in for the
# batched forwards; when a forward is not available yet the driver records that
# stage's inputs and raises ``BakeSuspend``, which ends the run early.  Once
# every scenario in the batch has reached the same boundary, one batched forward
# produces every row and the next round of runs proceeds -- so the perturbation
# of each batched stage is propagated into the next one's inputs rather than
# being measured in isolation.
#
# The extra partial runs are the price of not rewriting the accepted
# orchestrator: they re-execute only the host glue (normalisation, the dense
# target, the predicted local root, sampling, the codebook gather), never a
# neural stage, because every neural forward the driver owns is substituted.

STAGE_ORDER = ("probe", "plan", "pose", "decoder")

#: Round 6 adds the motion decode as a fifth forward point of the step generator.
#: It is **not** part of ``STAGE_ORDER``: it has no ``StageDriver`` method and the
#: round-4 arm never owns it (``batched_stages()`` cannot contain it), so the
#: round-3/round-4 bake paths stay exactly as they were.  Only the single-pass
#: orchestrator can resolve it, and only when ``bake_decode_batch`` is on.
DECODE_STAGE = "decode"

#: Which of a driver's *stages* owns each forward point.  ``decode`` owns itself,
#: and no stage set ever contains it unless the round-6 switch asked for it.
REQUEST_STAGE = {"probe": "root", "plan": "root", "pose": "pose", "decoder": "decoder",
                 DECODE_STAGE: DECODE_STAGE}


def _owns_stage(stages, stage: str) -> bool:
    """``probe``/``plan`` are the two halves of the ``root`` stage."""
    return REQUEST_STAGE[stage] in stages


class BakeSuspend(Exception):
    """Ends a ``run_transition`` call at a stage boundary (see :class:`StageDriver`)."""

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


class StageDriver:
    """The ``bake=`` seam: a stage-synchronised stand-in for the neural forwards.

    ``stages`` names the stages this driver owns (``root`` covers both the
    duration probe and the root-token/conv decoder body).  ``run_transition``
    asks :meth:`batches` before each forward, so a stage the driver does not own
    is executed by the accepted code and never appears here.
    """

    def __init__(self, stages) -> None:
        self.stages = frozenset(stages)
        self.rows: dict[str, mx.array | None] = {name: None for name in STAGE_ORDER}
        self.inputs: dict[str, object] = {}
        self.suspensions = 0

    def batches(self, stage: str) -> bool:
        """True when this driver owns *stage*; ``run_transition`` asks first.

        The two halves of the root stage are asked for under their own names
        (``probe`` for the duration probe, ``plan`` for the token transformer and
        convolutional decoder), because they are separate suspension points.
        """
        if stage in ("probe", "plan"):
            return "root" in self.stages
        return stage in self.stages

    def _barrier(self, stage: str, inputs):
        row = self.rows[stage]
        if row is None:
            self.inputs[stage] = inputs
            self.suspensions += 1
            raise BakeSuspend(stage)
        return row

    # -- the four forwards run_transition can ask for --------------------

    def duration(self, global_root, local_root, poses, global_mask, local_mask,
                 pose_mask):
        return self._barrier("probe", (global_root, local_root, poses,
                                       global_mask, local_mask, pose_mask))

    def plan(self, global_root, local_root, poses, global_mask, local_mask, pose_mask,
             requested_tokens, dense_target, dense_target_mask, duration_logits=None):
        return self._barrier("plan", {
            "global": global_root, "local": local_root, "poses": poses,
            "global_mask": global_mask, "local_mask": local_mask, "pose_mask": pose_mask,
            "tokens": int(requested_tokens), "dense_target": dense_target,
            "dense_target_mask": dense_target_mask})

    def pose(self, pose_tokens, local_root_values, pose_condition, has_pose_condition,
             positions, num_tokens):
        return self._barrier("pose", {
            "pose_tokens": pose_tokens, "local_root": local_root_values,
            "condition": pose_condition, "has": has_pose_condition,
            "positions": int(positions), "num_tokens": int(num_tokens)})

    def decoder(self, quantized, external_condition, target_condition,
                has_target_condition, positions):
        return self._barrier("decoder", {
            "quantized": quantized, "external": external_condition,
            "target": target_condition, "has": has_target_condition,
            "positions": int(positions)})


def batched_stages() -> frozenset[str]:
    """The stages the round-4 bake substitutes, from the two new switches.

    ``pose`` is always included -- round 4 is an *addition* to the accepted pose
    batching, not a replacement.  The two new switches are declared off
    (``nn.OPT_DEFAULTS``), so without an explicit opt-in this is the round-3
    stage set and the round-3 code path is what runs.
    """
    stages = {"pose"}
    if nn._opt("bake_root_batch"):
        stages.add("root")
    if nn._opt("bake_decoder_batch"):
        stages.add("decoder")
    return frozenset(stages)


def single_pass_stages(stages) -> frozenset:
    """The stage set the single-pass orchestrator will own.

    Round 6's ``bake_decode_batch`` adds the motion decode as a fifth forward
    point.  It is declared off, so without an opt-in this is exactly the set the
    caller passed -- round 5's behaviour, byte for byte.  The decode is
    deliberately **not** part of :func:`batched_stages`: the round-4 arm has no
    ``StageDriver`` for it and must stay as it was.
    """
    stages = frozenset(stages)
    return stages | {DECODE_STAGE} if nn._opt("bake_decode_batch") else stages


def _owns(driver: StageDriver, stage: str) -> bool:
    """``probe``/``plan`` are the two halves of the driver's ``root`` stage."""
    return _owns_stage(driver.stages, stage)


def _stack(values) -> mx.array:
    return mx.stack([value if isinstance(value, mx.array) else mx.array(value)
                     for value in values])


def _groups(drivers: list[StageDriver], stage: str, key) -> dict[object, list[int]]:
    grouped: dict[object, list[int]] = {}
    for index, driver in enumerate(drivers):
        if driver.rows[stage] is None and driver.inputs.get(stage) is not None:
            grouped.setdefault(key(driver), []).append(index)
    return grouped


def _resolve(pipeline, drivers: list[StageDriver], stage: str) -> None:
    """One batched forward per shape group, for every driver waiting on *stage*."""
    if stage == "probe":
        groups = _groups(drivers, "probe", lambda _driver: ("probe",))
    elif stage == "plan":
        groups = _groups(drivers, "plan", lambda d: d.inputs["plan"]["tokens"])
    elif stage == "pose":
        groups = _groups(drivers, "pose",
                         lambda d: (d.inputs["pose"]["positions"],
                                    d.inputs["pose"]["num_tokens"]))
    elif stage == "decoder":
        groups = _groups(drivers, "decoder", lambda d: d.inputs["decoder"]["positions"])
    else:                                                  # pragma: no cover - defensive
        raise ValueError(f"unknown stage {stage!r}")

    for key, indices in groups.items():
        members = [drivers[index] for index in indices]
        if stage == "probe":
            stacked = [_stack([m.inputs["probe"][slot] for m in members])
                       for slot in range(6)]
            out = pipeline.root.duration_logits_batch(*stacked)
        elif stage == "plan":
            stacked = [_stack([m.inputs["plan"][name] for m in members]) for name in
                       ("global", "local", "poses", "global_mask", "local_mask",
                        "pose_mask", "dense_target", "dense_target_mask")]
            out = pipeline.root.plan_values_batch(*stacked, key)
        elif stage == "pose":
            stacked = [_stack([m.inputs["pose"][name] for m in members]) for name in
                       ("pose_tokens", "local_root", "condition", "has")]
            out = pipeline.pose.plan_batch(*stacked, key[0], key[1])
        else:
            stacked = [_stack([m.inputs["decoder"][name] for m in members]) for name in
                       ("quantized", "external", "target", "has")]
            out = pipeline.decoder.decode_batch(*stacked, key)
        mx.eval(out)
        for row, member in enumerate(members):
            member.rows[stage] = out[row]


@dataclass
class BatchResult:
    """One batched bake of ``B`` scenarios, plus the machinery that produced it."""

    labels: list[str]
    runs: list[Run]
    drivers: list[StageDriver]
    suspensions: int
    stage_sizes: dict[str, int]
    agents: list = field(default_factory=list)
    # Round 5: the mx row each stage's batched forward produced, per scenario, so
    # the single-pass arm's host pass can be re-timed with every row already
    # resolved (``inject=``).  Empty for the round-4 driver.
    rows: dict[str, list] = field(default_factory=dict)


def drive_batch(pipeline, bundle, scenarios: list[Scenario],
                stages: frozenset[str] | set[str],
                capture_trace: bool = True) -> BatchResult:
    """Plan every scenario of one batch through the stage-synchronised driver.

    The returned :class:`Run` per scenario is produced by the **accepted** code
    with the three batched forwards substituted, so the only difference against
    the same scenario run alone is which forward produced each stage's output.
    """
    drivers = [StageDriver(stages) for _ in scenarios]
    agents: list[A.Agent] = []
    constraints = []
    for scenario in scenarios:
        agent = _new_agent(pipeline, bundle)
        agent.reset(scenario.style)
        _, baked_constraints = agent.constraints_for(scenario.command(), scenario.style)
        agents.append(agent)
        constraints.append(baked_constraints)

    results: list[Run | None] = [None] * len(scenarios)
    pending = set(range(len(scenarios)))
    rounds = 0
    while pending:
        rounds += 1
        if rounds > 4 * len(scenarios) + 8:               # pragma: no cover - defensive
            raise RuntimeError("stage driver made no progress")
        resolved = False
        for stage in STAGE_ORDER:
            waiting = [drivers[i] for i in pending if _owns(drivers[i], stage)]
            if _groups(waiting, stage, lambda _d: 0):
                _resolve(pipeline, waiting, stage)
                resolved = True
        if resolved:
            continue
        for index in list(pending):
            driver = drivers[index]
            ready = all(driver.rows[name] is not None
                        for name in STAGE_ORDER if _owns(driver, name))
            scenario = scenarios[index]
            try:
                if ready:
                    motion, info = agents[index].plan(
                        scenario.command(), scenario.style,
                        capture_trace=capture_trace, bake=driver)
                    trace = info.get("trace", {})
                    results[index] = Run(
                        scenario.label, int(info["selected_tokens"]),
                        np.asarray(trace.get("pose.tokens", np.zeros(0)), np.float32),
                        np.asarray(trace.get("pose.sampling_uniforms",
                                             np.zeros(0)), np.float32),
                        motion.roots.copy(), motion.rotations_xyzw.copy(),
                        _row(driver, "probe"), _row(driver, "plan"),
                        _row(driver, "pose"), _row(driver, "decoder"))
                    pending.discard(index)
                else:
                    pipeline.run_transition(constraints[index], scenario.seed,
                                            capture_trace=False, bake=driver)
                    raise RuntimeError(          # pragma: no cover - defensive
                        f"driver skipped a boundary for {scenario.label}")
            except BakeSuspend:
                pass
    return BatchResult([s.label for s in scenarios],
                       [run for run in results if run is not None],  # type: ignore[misc]
                       drivers,
                       sum(d.suspensions for d in drivers),
                       {name: sum(1 for d in drivers if _owns(d, name))
                        for name in STAGE_ORDER},
                       agents)


def _row(driver: StageDriver, stage: str) -> np.ndarray:
    row = driver.rows[stage]
    return np.zeros(0, np.float32) if row is None else np.asarray(
        row, dtype=np.float32).reshape(-1).copy()


def host_pass_timing(pipeline, bundle, scenarios: list[Scenario],
                     drivers: list[StageDriver], repeats: int = 5) -> dict:
    """Time the pass in which every batched row is already available.

    That pass runs no neural forward at all, so what it measures is the *host*
    side of a bake -- normalisation, the dense target, the predicted local root,
    sampling, the codebook gather, the motion decode and the agent maths -- which
    is the part batching cannot remove.  Each repeat uses a freshly reset agent so
    the measurement is not affected by the context blend of the previous call.
    """
    import statistics
    import time

    per_scenario: list[list[float]] = [[] for _ in scenarios]
    for _ in range(repeats):
        for index, scenario in enumerate(scenarios):
            agent = _new_agent(pipeline, bundle)
            agent.reset(scenario.style)
            started = time.perf_counter()
            motion, _ = agent.plan(scenario.command(), scenario.style,
                                   capture_trace=False, bake=drivers[index])
            mx.eval(mx.array(motion.roots.sum()))
            per_scenario[index].append((time.perf_counter() - started) * 1000.0)
    medians = [statistics.median(values) for values in per_scenario]
    return {
        "repeats": repeats,
        "per_scenario_ms_p50": statistics.median(medians),
        "worst_scenario_ms": max(medians),
        "total_ms_p50": statistics.median(medians) * len(scenarios),
        "per_scenario_ms": medians,
    }


def stage_perturbations(reference: list[Recorded], candidate: list[Run]) -> dict[str, float]:
    """``max |batched - serial|`` per stage, over every scenario of the batch.

    The round-3 campaign quoted one such number (the pose logits).  Batching the
    root and decoder adds three more, and unlike round 3 the pose logits number is
    now the *compounded* one: the pose planner's inputs are the batched root
    planner's outputs, so its perturbation includes the root stage's.
    """
    out: dict[str, float] = {}
    for key in ("duration_logits", "root_values", "pose_logits", "decoded"):
        worst = 0.0
        for record, run in zip(reference, candidate):
            left = np.asarray(getattr(record.run, key), np.float64).reshape(-1)
            right = np.asarray(getattr(run, key), np.float64).reshape(-1)
            if left.size and left.size == right.size:
                worst = max(worst, float(np.abs(left - right).max()))
        out[key] = worst
    return out


def decision_margins(reference: list[Recorded]) -> dict[str, float]:
    """The tightest discrete-decision margin on the reference arm.

    Two discrete decisions exist: the duration argmax over the style's allowed
    durations, and the Gumbel-noised pose-token argmax (round 3's definition,
    computed on exactly the scores ``sample_pose_tokens`` ranks).  A token flips
    only when the perturbation exceeds that decision's own margin, so the
    smallest margin in the batch is the one the safety factor is quoted against.
    """
    from . import sampling

    pose = np.inf
    duration = np.inf
    for record in reference:
        run = record.run
        constraints = record.constraints
        if constraints is not None and run.duration_logits.size:
            allowed = np.flatnonzero(constraints.allowed_tokens)
            if allowed.size > 1:
                ordered = np.sort(run.duration_logits[allowed].astype(np.float64))
                duration = min(duration, float(ordered[-1] - ordered[-2]))
        if run.pose_logits.size:
            logits = run.pose_logits.reshape(-1, 10)
            uniforms = run.uniforms.reshape(-1, 10)
            scores = logits + sampling.gumbel_noise(uniforms.astype(np.float32))
            ordered = np.sort(scores.astype(np.float64), axis=1)
            pose = min(pose, float((ordered[:, -1] - ordered[:, -2]).min()))
    return {"pose_margin": pose, "duration_margin": duration}


def safety_factor(perturbations: dict[str, float], margins: dict[str, float]) -> dict:
    """``perturbation / margin`` for each discrete decision (lower is safer)."""
    pose_margin = margins.get("pose_margin", np.inf)
    duration_margin = margins.get("duration_margin", np.inf)
    return {
        "pose_perturbation": perturbations.get("pose_logits", 0.0),
        "pose_margin": pose_margin,
        "pose_ratio": (perturbations.get("pose_logits", 0.0) / pose_margin
                       if pose_margin not in (0.0, np.inf) else None),
        "duration_perturbation": perturbations.get("duration_logits", 0.0),
        "duration_margin": duration_margin,
        "duration_ratio": (perturbations.get("duration_logits", 0.0) / duration_margin
                           if duration_margin not in (0.0, np.inf) else None),
    }


def _serial_stage(pipeline, stage: str, driver: StageDriver) -> object:
    """Call the accepted (serial) forward for *stage* on one driver's inputs."""
    inputs = driver.inputs[stage]
    if stage == "probe":
        return pipeline.root.duration_logits(*inputs)
    if stage == "plan":
        return pipeline.root.plan(
            inputs["global"], inputs["local"], inputs["poses"], inputs["global_mask"],
            inputs["local_mask"], inputs["pose_mask"], inputs["tokens"],
            inputs["dense_target"], inputs["dense_target_mask"])
    if stage == "pose":
        return pipeline.pose.plan(
            inputs["pose_tokens"], inputs["local_root"], inputs["condition"],
            inputs["has"], inputs["positions"], inputs["num_tokens"])
    return pipeline.decoder.decode(
        inputs["quantized"], inputs["external"], inputs["target"], inputs["has"],
        inputs["positions"])


def _batched_stage(pipeline, stage: str, drivers: list[StageDriver],
                   key) -> object:
    """Call the batched forward for *stage* over a shape-homogeneous driver group."""
    if stage == "probe":
        return pipeline.root.duration_logits_batch(
            *[_stack([d.inputs["probe"][slot] for d in drivers]) for slot in range(6)])
    if stage == "plan":
        stacked = [_stack([d.inputs["plan"][name] for d in drivers]) for name in
                   ("global", "local", "poses", "global_mask", "local_mask",
                    "pose_mask", "dense_target", "dense_target_mask")]
        return pipeline.root.plan_values_batch(*stacked, key)
    if stage == "pose":
        stacked = [_stack([d.inputs["pose"][name] for d in drivers]) for name in
                   ("pose_tokens", "local_root", "condition", "has")]
        return pipeline.pose.plan_batch(*stacked, key[0], key[1])
    stacked = [_stack([d.inputs["decoder"][name] for d in drivers]) for name in
               ("quantized", "external", "target", "has")]
    return pipeline.decoder.decode_batch(*stacked, key)


def _stage_key(stage: str, driver: StageDriver):
    """The shape key of a batched forward: what makes two rows share one call."""
    if stage == "plan":
        return driver.inputs["plan"]["tokens"]
    if stage == "pose":
        return (driver.inputs["pose"]["positions"], driver.inputs["pose"]["num_tokens"])
    if stage == "decoder":
        return driver.inputs["decoder"]["positions"]
    return None


def stage_throughput(pipeline, drivers: list[StageDriver], repeats: int = 5) -> dict:
    """Batched stage p50 against ``B`` separate serial forwards, per stage.

    Reported as a **stage** number only: it says how the forward scales, not what
    the bake delivers (round 3, 23.4).  ``drivers`` must be one shape-homogeneous
    group, because that is the only case in which a single batched forward covers
    the whole batch.
    """
    import statistics
    import time

    size = len(drivers)
    table: dict[str, dict] = {}
    for stage in STAGE_ORDER:
        if any(d.inputs.get(stage) is None for d in drivers):
            continue                       # a stage this driver set did not own
        key = _stage_key(stage, drivers[0])
        serial: list[float] = []
        for driver in drivers:
            started = time.perf_counter()
            out = _serial_stage(pipeline, stage, driver)
            values = out if isinstance(out, tuple) else (out,)
            mx.eval(*[value for value in values if isinstance(value, mx.array)])
            serial.append((time.perf_counter() - started) * 1000.0)
        batched: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            out = _batched_stage(pipeline, stage, drivers, key)
            mx.eval(out)
            batched.append((time.perf_counter() - started) * 1000.0)
        serial_p50 = statistics.median(serial)
        batched_p50 = statistics.median(batched)
        table[stage] = {
            "batch": size,
            "serial_p50_ms": serial_p50,
            "serial_total_ms": serial_p50 * size,
            "batched_p50_ms": batched_p50,
            "per_scenario_ms": batched_p50 / size,
            "speedup": (serial_p50 * size) / batched_p50 if batched_p50 else None,
            "repeats": repeats,
        }
    if "probe" in table and "plan" in table:
        probe, plan = table["probe"], table["plan"]
        table["root"] = {
            "batch": size,
            "serial_p50_ms": probe["serial_p50_ms"] + plan["serial_p50_ms"],
            "serial_total_ms": probe["serial_total_ms"] + plan["serial_total_ms"],
            "batched_p50_ms": probe["batched_p50_ms"] + plan["batched_p50_ms"],
            "per_scenario_ms": probe["per_scenario_ms"] + plan["per_scenario_ms"],
            "speedup": ((probe["serial_p50_ms"] + plan["serial_p50_ms"]) * size
                        / (probe["batched_p50_ms"] + plan["batched_p50_ms"])),
            "repeats": repeats,
        }
    return table


# ---------------------------------------------------------------------------
# Round 5: the single-pass orchestrator
# ---------------------------------------------------------------------------
# Round 4's driver synchronises the batch at stage boundaries by *restarting*
# ``run_transition`` once per boundary per scenario, so every host step before a
# boundary is executed again and thrown away: measured 3.73 ms of a 9.92 ms
# ``B = 16`` bake (report 33.3).  Round 5 makes the transition resumable --
# ``MlxPipeline.transition_steps`` is a step generator whose locals survive the
# four forward points -- and drives one generator per scenario in lockstep here.
# Each host step then runs exactly once per scenario, each neural forward exactly
# once per shape group, and nothing is restarted.
#
# ``drive_batch`` above is deliberately untouched: it is round 5's *control arm*,
# and the two arms must agree bit for bit (``compare_bitwise``), otherwise the
# throughput comparison against round 4 would be measuring two different bakes.


# ---------------------------------------------------------------------------
# Round 6: the direct instrument (report section 43.1)
# ---------------------------------------------------------------------------
# Round 5 attributed the single pass by *subtracting* two independently timed
# instruments (``stage_throughput`` and an injected-row host pass), which left
# 2.29 ms/scenario of a 8.24 ms pass unexplained -- and section 38 shows what that
# kind of arithmetic costs when it is used to predict.  This instrument times the
# pass itself: one bracket per **leaf** segment, no bracket enclosing another, so
# the segments add up and the closure residual is only the orchestrator's own
# bookkeeping.  Everything stays off the hot path unless a caller passes a
# :class:`Profile`.

#: Leaf segments of the direct instrument.  ``glue_*`` are inside the transition
#: step generator (``models/transition.py``); the rest are the orchestrator's.
PROFILE_SEGMENTS = (
    # orchestrator
    "setup",           # per-scenario agent reset + constraints_for + generator prime
    "schedule",        # shape grouping and request keying for one stage
    "stack",           # gathering/concatenating one group's batched inputs
    "forward",         # the batched forward's Python call (graph build)
    "eval",            # mx.eval: the device drain
    "scatter",         # slicing the batched output back into per-scenario rows
    "collect",         # finish_plan (agent maths) + Run/trace construction
    # generator host steps
    "glue_normalize",  # input normalisation and the three masks
    "glue_duration",   # duration argmax over the allowed tokens
    "glue_dense",      # dense target + mask for the root planner
    "glue_pred_local", # predicted local root (host read-back + velocity block)
    "glue_condition",  # pose conditions and the pose root argument
    "glue_upload",     # materialising a stage's argument tuple for the device
    "glue_sample",     # logits reshape, uniforms, Gumbel sampling
    "glue_gather",     # codebook gather and the decoder's external condition
    "glue_decoder_io", # the decoder output's host read-back and re-upload
    "glue_decode",     # decode_motion and the host read-back of its outputs
)


class Profile:
    """Accumulator for the round-6 direct instrument.

    ``add`` takes the bracket's start timestamp so a segment costs one
    ``perf_counter`` pair and one dict update; :meth:`summary` turns the totals
    into per-scenario milliseconds and reports the **closure** against the
    measured region, so nothing is attributed by subtraction.
    """

    __slots__ = ("totals", "counts")

    def __init__(self) -> None:
        self.totals: dict[str, float] = dict.fromkeys(PROFILE_SEGMENTS, 0.0)
        self.counts: dict[str, int] = dict.fromkeys(PROFILE_SEGMENTS, 0)

    def add(self, name: str, started: float) -> None:
        self.totals[name] += time.perf_counter() - started
        self.counts[name] += 1

    def summary(self, measured_total_ms: float, scenarios: int) -> dict:
        segments = {
            name: {
                "total_ms": 1000.0 * seconds,
                "per_scenario_ms": 1000.0 * seconds / scenarios,
                "calls": self.counts[name],
            }
            for name, seconds in self.totals.items()
        }
        closed = sum(item["total_ms"] for item in segments.values())
        return {
            "segments": segments,
            "segments_total_ms": closed,
            "measured_total_ms": measured_total_ms,
            "closure_residual_ms": measured_total_ms - closed,
            "per_scenario_closure_residual_ms": (
                measured_total_ms - closed) / scenarios,
        }


def _stack_arguments(stage: str, arguments: list) -> list:
    """Gather one shape group's inputs into the stacked arrays a batched call takes.

    Byte for byte the stacking :func:`_batched_request` always did; split out so
    the direct instrument can time the gather separately from the call.
    """
    if stage == "probe":
        return [_stack([argument[slot] for argument in arguments]) for slot in range(6)]
    if stage == "plan":
        # The request is ``(global, local, poses, global_mask, local_mask,
        # pose_mask, tokens, dense_target, dense_target_mask)``; the token count is
        # a trace-time constant and is the *shape key*, never a stacked input --
        # the same eight arrays ``_resolve`` stacks, in the same order.
        return [_stack([argument[slot] for argument in arguments])
                for slot in (0, 1, 2, 3, 4, 5, 7, 8)]
    if stage == "pose":
        return [_stack([argument[slot] for argument in arguments]) for slot in range(4)]
    if stage == DECODE_STAGE:
        # ``(decoded, mean, std, parents, initial_x, initial_z)``: the statistics
        # and the parent table are one per bundle, the origins are one per
        # scenario and must become a ``(batch, 1)`` column so the cumulative
        # travel stays inside each scenario.
        origins = np.array([[float(argument[4]), float(argument[5])]
                            for argument in arguments], dtype=np.float32)
        return [_stack([argument[0] for argument in arguments]),
                arguments[0][1], arguments[0][2], arguments[0][3],
                mx.array(origins[:, 0:1]), mx.array(origins[:, 1:2])]
    return [_stack([argument[slot] for argument in arguments]) for slot in range(4)]


def _call_batched(pipeline, stage: str, stacked: list, key) -> object:
    """The batched forward for one stacked shape group (the accepted call)."""
    if stage == "probe":
        return pipeline.root.duration_logits_batch(*stacked)
    if stage == "plan":
        return pipeline.root.plan_values_batch(*stacked, key)
    if stage == "pose":
        return pipeline.pose.plan_batch(*stacked, key[0], key[1])
    if stage == DECODE_STAGE:
        return motion_rep.decode_motion_batch(*stacked)
    return pipeline.decoder.decode_batch(*stacked, key)


def _evaluate(out) -> None:
    """Drain a batched forward's output (a tuple when the decode returns two)."""
    if isinstance(out, tuple):
        mx.eval(*out)
    else:
        mx.eval(out)


def _rows_of(out, count: int) -> list:
    """Slice a batched forward's output back into one row per scenario."""
    if isinstance(out, tuple):
        return [tuple(part[row] for part in out) for row in range(count)]
    return [out[row] for row in range(count)]


def _request_key(stage: str, arguments) -> object:
    """The shape key of one forward request: what makes two scenarios share a call.

    The three keys are the ones :func:`_resolve` uses for the round-4 driver, read
    off the request's own arguments instead of off a ``StageDriver``.
    """
    if stage == "probe":
        return ("probe",)
    if stage == "plan":
        return int(arguments[6])                     # requested token count
    if stage == "pose":
        return (int(arguments[4]), int(arguments[5]))  # positions, num_tokens
    if stage == DECODE_STAGE:
        return int(arguments[0].shape[0])            # frame count of the decoded row
    return int(arguments[4])                         # decoder: positions


def _batched_request(pipeline, stage: str, arguments: list, key,
                     profile: Profile | None = None) -> list:
    """One batched forward for a shape-homogeneous group of step requests.

    Byte for byte the calls :func:`_resolve` makes for the round-4 driver: same
    stacking order, same shape key, same ``mx.eval`` and the same row slicing, so
    the two arms consume identical rows.  With ``profile`` the gather, the call,
    the drain and the scatter are timed as four separate leaf segments; without it
    this is the accepted code path.
    """
    if profile is None:
        out = _call_batched(pipeline, stage, _stack_arguments(stage, arguments), key)
        _evaluate(out)
        return _rows_of(out, len(arguments))

    started = time.perf_counter()
    stacked = _stack_arguments(stage, arguments)
    profile.add("stack", started)
    started = time.perf_counter()
    out = _call_batched(pipeline, stage, stacked, key)
    profile.add("forward", started)
    started = time.perf_counter()
    _evaluate(out)
    profile.add("eval", started)
    started = time.perf_counter()
    rows = _rows_of(out, len(arguments))
    profile.add("scatter", started)
    return rows


class _SinglePassSession:
    """One scenario's transition generator plus the agent state that finishes it."""

    def __init__(self, pipeline, bundle, scenario: Scenario, stages: frozenset,
                 capture_trace: bool, profile: Profile | None = None) -> None:
        self.scenario = scenario
        self.command = scenario.command()
        self.stages = stages
        self.agent = _new_agent(pipeline, bundle)
        self.agent.reset(scenario.style)
        # Exactly what ``Agent.plan`` builds, kept so the scenario can be finished
        # by ``Agent.finish_plan`` with the boundary set this pass actually used.
        self.frame, self.constraints = self.agent.constraints_for(
            self.command, scenario.style)
        self._pipeline = pipeline
        self._capture_trace = capture_trace
        self._profile = profile
        self.steps = None
        self.request = None
        self.rows: dict[str, object] = {}
        self.result = None

    def start(self) -> None:
        """Create the step generator and run it to its first forward request.

        Kept out of ``__init__`` so the direct instrument's ``setup`` bracket (agent
        reset + ``constraints_for``) does not enclose the generator's first host
        steps: those are bracketed inside the generator as ``glue_*``, and a
        bracket may not contain another one.
        """
        self.steps = self._pipeline.transition_steps(
            self.constraints, self.command.seed, self.command.sampling_argmax,
            self._capture_trace, self._profile)
        self.request = next(self.steps)

    def send(self, value) -> None:
        """Feed one forward's output back; ``None`` request means finished."""
        try:
            self.request = self.steps.send(value)
        except StopIteration as stop:
            self.result = stop.value
            self.request = None

    def advance(self, profile: Profile | None = None):
        """Run every host step and every forward this orchestrator does not own.

        Stops on the first request for a stage the orchestrator owns (or when the
        transition is finished), which is the whole point: the steps that ran are
        kept in the generator's frame, never re-executed.  ``profile`` only adds a
        bracket around the accepted forward of an *unowned* stage; the host steps
        themselves are bracketed inside the generator, which is why this does not
        time them a second time.
        """
        while self.request is not None and not _owns_stage(self.stages, self.request[0]):
            if profile is None:
                self.send(self.request[2]())               # the accepted forward
            else:
                started = time.perf_counter()
                value = self.request[2]()
                profile.add("forward", started)
                self.send(value)
        return self.request


def _session_row(session: _SinglePassSession, stage: str) -> np.ndarray:
    row = session.rows.get(stage)
    return (np.zeros(0, np.float32) if row is None
            else np.asarray(row, dtype=np.float32).reshape(-1).copy())


def drive_batch_single_pass(pipeline, bundle, scenarios: list[Scenario],
                            stages, capture_trace: bool = True,
                            inject: dict[str, list] | None = None,
                            profile: Profile | None = None) -> BatchResult:
    """Plan every scenario of one batch in **one pass** of the transition.

    Each scenario owns one step generator (:meth:`MlxPipeline.transition_steps`)
    and its own reset agent.  The orchestrator advances every generator to the
    first forward it owns, resolves all of them for that stage in shape groups,
    sends the rows back, and repeats until every generator returns -- so each host
    step runs once per scenario and each forward once per group.

    ``inject`` replaces the batched forwards with rows captured from an earlier
    run (``BatchResult.rows``), which is how the host side of the pass is timed
    without any neural work: the host steps still run, in the same order, and the
    rows they would have consumed are handed straight back.

    ``profile`` (round 6) turns on the direct instrument: every leaf segment is
    bracketed and accumulated into the passed :class:`Profile`, and the caller
    reports the closure against the region it timed.  ``None`` (the default) is
    the accepted code path.

    The round-6 switch ``bake_decode_batch`` adds the motion decode as a fifth
    forward point this orchestrator owns.  It is declared off, so the four-stage
    orchestrator of round 5 is what runs unless a caller opts in; when it is on,
    the decode of each shape group is issued as **one** batched graph, which is
    what turns the per-scenario decode drain into a single one.

    The returned :class:`BatchResult` carries the same :class:`Run` per scenario
    as :func:`drive_batch`, so the campaign comparison code is shared; ``drivers``
    is empty because this arm has no ``StageDriver`` and never suspends.
    """
    stages = single_pass_stages(stages)
    order = STAGE_ORDER + ((DECODE_STAGE,) if DECODE_STAGE in stages else ())
    if profile is not None:
        started = time.perf_counter()
    sessions = [_SinglePassSession(pipeline, bundle, scenario, stages, capture_trace,
                                   profile)
                for scenario in scenarios]
    if profile is not None:
        profile.add("setup", started)
    for session in sessions:
        session.start()
    for session in sessions:
        session.advance(profile)

    pending = [index for index, session in enumerate(sessions)
               if session.request is not None]
    rounds = 0
    while pending:
        rounds += 1
        if rounds > (len(order) + 1) * len(scenarios) + 8:  # pragma: no cover - defensive
            raise RuntimeError("single-pass orchestrator made no progress")
        for stage in order:
            if profile is not None:
                started = time.perf_counter()
            waiting = [index for index in pending
                       if sessions[index].request is not None
                       and sessions[index].request[0] == stage]
            if not waiting:
                if profile is not None:
                    profile.add("schedule", started)
                continue
            grouped: dict[object, list[int]] = {}
            for index in waiting:
                grouped.setdefault(
                    _request_key(stage, sessions[index].request[1]), []).append(index)
            if profile is not None:
                profile.add("schedule", started)
            for key, indices in grouped.items():
                arguments = [sessions[index].request[1] for index in indices]
                if inject is None:
                    rows = _batched_request(pipeline, stage, arguments, key, profile)
                else:
                    rows = [inject[stage][index] for index in indices]
                for row, index in zip(rows, indices):
                    sessions[index].rows[stage] = row
                    # The probe answers with ``(logits, boundary)`` on every path;
                    # a batched probe has no boundary pair, exactly as the round-4
                    # driver behaved (``MlxPipeline._stage_forward``).
                    sessions[index].send((row, None) if stage == "probe" else row)
                    sessions[index].advance(profile)
        if profile is not None:
            started = time.perf_counter()
        pending = [index for index in pending if sessions[index].request is not None]
        if profile is not None:
            profile.add("schedule", started)

    if profile is not None:
        started = time.perf_counter()
    runs: list[Run] = []
    for session in sessions:
        motion, info = session.agent.finish_plan(session.result, session.frame,
                                                 session.constraints, session.command)
        trace = info.get("trace", {})
        runs.append(Run(
            session.scenario.label, int(info["selected_tokens"]),
            np.asarray(trace.get("pose.tokens", np.zeros(0)), np.float32),
            np.asarray(trace.get("pose.sampling_uniforms", np.zeros(0)), np.float32),
            motion.roots.copy(), motion.rotations_xyzw.copy(),
            _session_row(session, "probe"), _session_row(session, "plan"),
            _session_row(session, "pose"), _session_row(session, "decoder")))
    if profile is not None:
        profile.add("collect", started)
    return BatchResult(
        [scenario.label for scenario in scenarios], runs, [], 0,
        {name: sum(1 for session in sessions if _owns_stage(session.stages, name))
         for name in STAGE_ORDER},
        [session.agent for session in sessions],
        {name: [session.rows.get(name) for session in sessions]
         for name in order})


def host_pass_timing_single_pass(pipeline, bundle, scenarios: list[Scenario],
                                 stages, rows: dict[str, list],
                                 repeats: int = 5) -> dict:
    """Time the single-pass orchestrator with every batched row already resolved.

    That pass runs no neural forward at all, so it measures the **host** side of a
    one-pass bake -- normalisation, the dense target, the predicted local root, the
    conditioning, sampling, the codebook gather, the motion decode, the agent maths
    and the orchestration itself -- which is the part batching cannot remove.  It
    is the round-5 counterpart of :func:`host_pass_timing`, measured on the same
    scenario set; the two differ only in that this one is interleaved by the
    orchestrator instead of run one scenario at a time.
    """
    import statistics
    import time

    totals: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        drive_batch_single_pass(pipeline, bundle, scenarios, stages,
                                capture_trace=False, inject=rows)
        totals.append((time.perf_counter() - started) * 1000.0)
    total = statistics.median(totals)
    return {
        "repeats": repeats,
        "total_ms_p50": total,
        "per_scenario_ms_p50": total / len(scenarios),
        "totals_ms": totals,
    }


def compare_bitwise(left: list[Run], right: list[Run]) -> dict:
    """Exact equality of two bake arms over the same scenarios (round-5 check).

    ``compare`` quantifies the deviation of a batched arm from the serial
    reference.  This one is the *instrument* check: the two batch arms must
    produce identical rows, so anything but exact equality -- ``np.array_equal``,
    no tolerance -- means the single-pass orchestrator changed an execution order
    that mattered and its throughput may not be compared with round 4's.
    """
    def equal(a, b) -> bool:
        first, second = np.asarray(a), np.asarray(b)
        return bool(first.shape == second.shape and np.array_equal(first, second))

    rows = []
    for a, b in zip(left, right):
        rows.append({
            "label": a.label,
            "tokens": equal(a.tokens, b.tokens),
            "uniforms": equal(a.uniforms, b.uniforms),
            "selected": a.selected == b.selected,
            "roots": equal(a.roots, b.roots),
            "rotations": equal(a.rotations, b.rotations),
            "differing_tokens": (int(np.count_nonzero(np.asarray(a.tokens)
                                                      != np.asarray(b.tokens)))
                                 if np.asarray(a.tokens).shape
                                 == np.asarray(b.tokens).shape else -1),
        })
    return {
        "scenarios": len(rows),
        "tokens_identical": sum(row["tokens"] for row in rows),
        "uniforms_identical": sum(row["uniforms"] for row in rows),
        "selected_identical": sum(row["selected"] for row in rows),
        "roots_identical": sum(row["roots"] for row in rows),
        "rotations_identical": sum(row["rotations"] for row in rows),
        "differing_tokens": sum(max(0, row["differing_tokens"]) for row in rows),
        "bit_identical": all(row["tokens"] and row["uniforms"] and row["selected"]
                             and row["roots"] and row["rotations"] for row in rows),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# What a baked product's manifest must carry (decision 0007)
# ---------------------------------------------------------------------------

#: Decision ``0007`` requires these on the manifest of anything a bake produced.
#: They are assembled here, from the campaign that was actually run, so the tier
#: cannot be written by hand from a previous release.
BAKE_MANIFEST_FIELDS = (
    "batch_size",
    "bake_tier",
    "bake_stages",
    "bake_stage_switches",
    "bake_grouping",
    "bake_campaign_scenarios",
    "bake_token_decisions_differing",
    "bake_worst_roots_m",
    "bake_worst_rotations",
    "bake_safety_ratio",
    "bake_min_decision_margin",
    "bake_action_bit_identical",
)

#: Decision ``0007``'s original headroom heuristic, kept only so the report and
#: the manifest can show what the stricter arithmetic would have said.  Decision
#: ``0008`` replaced it with the mechanically decidable ``RELEASE_RATIO`` below.
RELEASE_MARGIN = 1.5

#: Decision ``0008`` item 1: a release bake needs
#: ``perturbation / tightest decision margin <= RELEASE_RATIO``, i.e. at least
#: ``1 / 0.70 = 1.43x`` headroom.  0008 chose 0.70 over 0007's literal ``1/1.5``
#: (= 0.6667) because the measured safety boundary sits between mixed ``B = 8``
#: (0.6695, finite margin) and mixed ``B = 16`` (1.0426, none), and because 1.5
#: was an integer heuristic rather than a measured cliff.
RELEASE_RATIO = 0.70

#: Decision ``0008`` item 3: above this ``B`` the batch must be
#: duration-homogeneous, because the mixed group has no headroom at ``B = 16``.
GROUPED_ABOVE_B = 8


def bake_tier(pose_ratio: float | None, campaign_ok: bool,
              ratio: float = RELEASE_RATIO,
              grouping: str = "duration-mixed",
              batch_size: int = 0) -> str:
    """``reproducible`` or ``draft`` for one ``B`` (decisions 0007 and 0008).

    ``reproducible`` requires **all three** of:

    * the campaign passed -- every scenario still matches the same scenario run
      alone on the discrete outputs (0006/0007);
    * ``pose_ratio <= ratio`` -- 0008's mechanically decidable form of 0007's
      "margin" rule.  Passing the campaign alone is not enough: round 4's mixed
      ``B = 16`` passed it with no margin left (1.0426);
    * ``batch_size <= GROUPED_ABOVE_B`` **or** the grouping is
      duration-homogeneous (0008 item 3).

    ``ratio`` is exposed so the report can show the 0007 arithmetic beside 0008's
    without a second implementation.
    """
    if not campaign_ok or pose_ratio is None or pose_ratio <= 0.0:
        return "draft"
    if pose_ratio > ratio:
        return "draft"
    if batch_size > GROUPED_ABOVE_B and grouping != "duration-homogeneous":
        return "draft"
    return "reproducible"


def manifest_fields(*, batch_size: int, stages, grouping: str, token_groups: dict,
                    comparison: dict, safety: dict) -> dict:
    """The manifest block decision ``0007`` requires for one baked ``B``.

    ``comparison`` is :func:`compare`'s result for this ``B`` (the campaign) and
    ``safety`` is :func:`safety_factor`'s.  ``bake_action_bit_identical`` is
    derived, never asserted: full-stage batching does not reproduce the final
    action bit for bit, and the field is what lets a reviewer see that instead of
    having to trust a tolerance.  The tier is :func:`bake_tier` under decision
    ``0008`` (threshold **and** the grouping rule).
    """
    stages = frozenset(stages)
    campaign_ok = (comparison["token_identical"] == comparison["scenarios"]
                   and comparison["uniform_identical"] == comparison["scenarios"]
                   and comparison["selected_identical"] == comparison["scenarios"])
    return {
        "batch_size": int(batch_size),
        "bake_tier": bake_tier(safety.get("pose_ratio"), campaign_ok,
                               grouping=grouping, batch_size=int(batch_size)),
        "bake_stages": sorted(stages),
        "bake_stage_switches": [name for name, stage in
                                (("bake_root_batch", "root"),
                                 ("bake_decoder_batch", "decoder"),
                                 ("bake_decode_batch", DECODE_STAGE))
                                if stage in stages],
        "bake_grouping": {"mode": grouping, "token_groups": dict(token_groups)},
        "bake_campaign_scenarios": comparison["scenarios"],
        "bake_token_decisions_differing": comparison["differing_tokens"],
        "bake_worst_roots_m": comparison["worst_roots_m"],
        "bake_worst_rotations": comparison["worst_rotations"],
        "bake_safety_ratio": safety.get("pose_ratio"),
        "bake_min_decision_margin": safety.get("pose_margin"),
        "bake_action_bit_identical": bool(comparison["worst_roots_m"] == 0.0
                                          and comparison["worst_rotations"] == 0.0),
    }
