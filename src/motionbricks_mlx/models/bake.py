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


def _owns(driver: StageDriver, stage: str) -> bool:
    """``probe``/``plan`` are the two halves of the driver's ``root`` stage."""
    if stage in ("probe", "plan"):
        return "root" in driver.stages
    return stage in driver.stages


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
