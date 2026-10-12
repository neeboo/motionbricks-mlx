"""Round-3 batch bake: the seam, the batched forward, and the token campaign.

`docs/optimization-report.md` section 21 item 2 fixes the rules before the
measurement: the bake is judged on throughput **and** on per-scenario token
identity against the same scenario run alone, and the report's safe-B rule says
the largest all-identical B is the reproducible tier while anything larger is a
draft tier whose B must be recorded in the product manifest.

This file pins the machinery the campaign rests on:

* the ``pose_logits`` seam reproduces the accepted pipeline exactly when it is
  handed the pipeline's own logits -- without that, a bake comparison would be
  measuring the harness rather than the batching;
* ``plan_batch`` is a genuine batch with a **documented, non-zero** deviation
  from the serial forward (the vmap lowering changes matmul accumulation), so a
  later change that silently makes it exact, or silently makes it worse, fails
  here;
* scenarios that selected different durations are never mixed into one batch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motionbricks_mlx.data import agent as A  # noqa: E402
from motionbricks_mlx.data.style import load_style  # noqa: E402
from motionbricks_mlx.models import bake, nn  # noqa: E402
from motionbricks_mlx.models.bundle import Bundle  # noqa: E402
from motionbricks_mlx.models.pose import PosePlanner  # noqa: E402
from motionbricks_mlx.models.transition import MlxPipeline  # noqa: E402

MLX_BUNDLE = ROOT / "checkpoints/g1-f32-mlx"
HAVE_BUNDLE = (MLX_BUNDLE / "manifest.json").is_file()
STYLES_DIR = Path.home() / "dev/motion-bricks-cpp/repo/generated/styles"
HAVE_STYLES = STYLES_DIR.is_dir()
NEEDS = pytest.mark.skipif(not (HAVE_BUNDLE and HAVE_STYLES),
                           reason="bundle and reference styles required")


def scenarios(count: int) -> list[bake.Scenario]:
    names = ("walk", "walk_zombie", "slow_walk", "walk_stealth")
    return [bake.Scenario(label=f"s{index}-{names[index % 4]}",
                          style=load_style(STYLES_DIR / f"{names[index % 4]}.mbstyle"),
                          seed=1 + index, movement=(0.2, 0.0, 0.0))
            for index in range(count)]


@NEEDS
def test_pose_logits_seam_reproduces_the_accepted_pipeline():
    """Handing the pipeline its own logits must change nothing at all.

    This is what makes the batched arm comparable: every stage after the pose
    planner is the accepted code, and the only difference between the arms is
    which forward produced the logits.
    """
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    scenario = scenarios(1)[0]
    captured: dict[str, mx.array] = {}
    original = pipeline.pose.plan

    def spy(*args, **kwargs):
        logits = original(*args, **kwargs)
        captured["logits"] = logits
        return logits

    pipeline.pose.plan = spy
    agent = A.Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
    agent.reset(scenario.style)
    reference, ref_info = agent.plan(scenario.command(), scenario.style,
                                     capture_trace=True)
    pipeline.pose.plan = original

    agent2 = A.Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
    agent2.reset(scenario.style)
    candidate, cand_info = agent2.plan(scenario.command(), scenario.style,
                                       capture_trace=True,
                                       pose_logits=captured["logits"])

    assert np.array_equal(reference.roots, candidate.roots)
    assert np.array_equal(reference.rotations_xyzw, candidate.rotations_xyzw)
    assert np.array_equal(ref_info["trace"]["pose.tokens"], cand_info["trace"]["pose.tokens"])
    assert np.array_equal(ref_info["trace"]["pose.sampling_uniforms"],
                          cand_info["trace"]["pose.sampling_uniforms"])


@NEEDS
def test_plan_batch_shape_and_documented_deviation():
    """A batched forward is a real batch, and it is *not* bitwise the serial one.

    Measured deviation on the pose logits: ~1e-4 at B = 1 rising to ~4e-4 at
    B = 16.  The assertions below are a band, not an equality: they fail if the
    lowering becomes exact (which would invalidate the report's explanation of
    the campaign) or if it degrades by an order of magnitude.
    """
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    records, _ = bake.record_serial(pipeline, bundle, scenarios(2))
    serial = [np.asarray(PosePlanner.reshape_logits(
        pipeline.pose.plan(np.full((record.tokens, 8), 10, np.int32),
                           mx.array(record.pose_root), mx.array(record.pose_condition),
                           mx.array(record.has_pose_condition),
                           record.tokens, record.tokens)), np.float32)
        for record in records]
    # Both scenarios selected the same duration, so they must share one batch.
    groups = bake.group_by_tokens(records)
    assert len(groups) == 1 and list(groups.values())[0] == [0, 1]
    logits = bake.batched_logits(pipeline, records)
    assert len(logits) == 2
    assert logits[0].shape == (records[0].tokens, 80)
    for row, expected in zip(logits, serial):
        deviation = np.abs(np.asarray(row).reshape(-1) - expected.reshape(-1)).max()
        assert 0.0 < deviation < 1.0e-2, deviation


def test_scenarios_with_different_durations_are_never_mixed():
    """One compiled variant per shape: a group is one token count, never two."""
    records = [bake.Recorded(scenario=bake.Scenario(label=f"s{i}", style=None, seed=i),
                             run=bake.Run(label=f"s{i}", selected=1),
                             constraints=None, tokens=tokens,
                             pose_root=np.zeros((1, 4), np.float32),
                             pose_condition=np.zeros((1, 1), np.float32),
                             has_pose_condition=np.zeros((1, 1), np.float32))
               for i, tokens in enumerate((11, 7, 11, 6, 7, 11))]
    groups = bake.group_by_tokens(records)
    assert groups == {11: [0, 2, 5], 7: [1, 4], 6: [3]}
    for tokens, indices in groups.items():
        assert all(records[index].tokens == tokens for index in indices)


@NEEDS
def test_bake_rows_match_their_own_b1_run():
    """The campaign's claim, at the smallest size: tokens and action identical."""
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    records, single = bake.record_serial(pipeline, bundle, scenarios(2))
    logits = bake.batched_logits(pipeline, records)
    runs = bake.rerun_batched(pipeline, bundle, records, logits)
    comparison = bake.compare([record.run for record in records], runs)
    assert comparison["scenarios"] == 2
    assert comparison["token_identical"] == 2
    assert comparison["uniform_identical"] == 2
    assert comparison["selected_identical"] == 2
    assert comparison["differing_tokens"] == 0
    assert comparison["worst_roots_m"] == 0.0
    assert comparison["worst_rotations"] == 0.0
    assert len(single) == 2 and all(value > 0 for value in single)


def test_round3_sync_candidates_are_declared_off():
    """Section 24: none of the three cleared the frozen 1.05x bar.

    They stay available as switches -- the token gate and the final action were
    both bit-identical -- but the default path is unchanged.
    """
    assert nn.OPT_DEFAULTS["decoder_defer"] is False
    assert nn.OPT_DEFAULTS["pred_local_defer"] is False
    assert nn.OPT_DEFAULTS["root_boundary_reuse"] is False


# ---------------------------------------------------------------------------
# Round 4: the root and decoder stages
# ---------------------------------------------------------------------------

ROUND4_OPT = "bake_root_batch=1,bake_decoder_batch=1"


def test_bake_stage_switches_are_declared_off():
    """The two round-4 switches only mean something on the bake path.

    Round 3's bake must keep behaving exactly as it did, so the new stages are
    opt-in and the declared default is the round-3 stage set.
    """
    assert nn.OPT_DEFAULTS["bake_root_batch"] is False
    assert nn.OPT_DEFAULTS["bake_decoder_batch"] is False


def test_batched_stages_follows_the_switches(monkeypatch):
    """``batched_stages`` is the switch state, and pose is always included."""
    monkeypatch.delenv("MOTIONBRICKS_MLX_OPT", raising=False)
    assert bake.batched_stages() == frozenset({"pose"})
    monkeypatch.setenv("MOTIONBRICKS_MLX_OPT", "bake_root_batch=1")
    assert bake.batched_stages() == frozenset({"pose", "root"})
    monkeypatch.setenv("MOTIONBRICKS_MLX_OPT", ROUND4_OPT)
    assert bake.batched_stages() == frozenset({"pose", "root", "decoder"})


@NEEDS
def test_batched_root_and_decoder_are_real_batches_with_a_documented_band(
        monkeypatch):
    """Same band rule as the pose batch: genuine batches, never bitwise serial.

    Round 3 pinned the pose logits at ~1e-4.  The root planner's output and the
    decoder's output are smaller-magnitude quantities, so their bands are tighter
    but still strictly non-zero -- which is what makes the campaign necessary
    rather than optional, and what fails a change that silently makes the
    lowering exact.
    """
    monkeypatch.setenv("MOTIONBRICKS_MLX_OPT", ROUND4_OPT)
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    batch = scenarios(2)
    reference, _ = bake.record_serial(pipeline, bundle, batch, capture_stages=True)
    assert reference[0].tokens == reference[1].tokens, "a batch needs one duration"
    result = bake.drive_batch(pipeline, bundle, batch, bake.batched_stages())
    perturbations = bake.stage_perturbations(reference, result.runs)
    # duration logits: a 12-wide head, so the deviation is the smallest of all
    assert 0.0 < perturbations["duration_logits"] < 1.0e-3
    # root values: normalised (frames, 5) planner output
    assert 0.0 < perturbations["root_values"] < 1.0e-4
    # pose logits: the compounded one, amplified by the root condition
    assert 0.0 < perturbations["pose_logits"] < 1.0e-1
    # decoder output: normalised 413-wide motion
    assert 0.0 < perturbations["decoded"] < 1.0e-2


@NEEDS
def test_stage_driver_reproduces_the_reference_arm(monkeypatch):
    """The campaign's claim with every stage batched: tokens, uniforms, duration.

    The *action* is deliberately not asserted bitwise here -- batching the root
    and decoder stages perturbs the conditions that reach the final decode, so
    the action deviation is a measured quantity reported by the campaign, not a
    property of the seam.
    """
    monkeypatch.setenv("MOTIONBRICKS_MLX_OPT", ROUND4_OPT)
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    batch = scenarios(2)
    reference, _ = bake.record_serial(pipeline, bundle, batch, capture_stages=True)
    result = bake.drive_batch(pipeline, bundle, batch, bake.batched_stages())
    assert len(result.runs) == len(batch)
    # Two boundaries per neural stage, and one suspension per scenario per
    # boundary: the driver never skips a stage it owns.
    assert result.suspensions >= 2 * len(batch)
    assert set(result.stage_sizes) == {"probe", "plan", "pose", "decoder"}
    comparison = bake.compare([record.run for record in reference], result.runs)
    assert comparison["token_identical"] == len(batch)
    assert comparison["uniform_identical"] == len(batch)
    assert comparison["selected_identical"] == len(batch)
    assert comparison["differing_tokens"] == 0
    # The action moves, but only by float32 round-off: roots are in metres and
    # the alignment tolerance for them is 1e-3, so this is orders inside it.
    assert 0.0 < comparison["worst_roots_m"] < 1.0e-4
    assert 0.0 < comparison["worst_rotations"] < 1.0e-3


@NEEDS
def test_driver_safety_numbers_are_reported_and_below_one(monkeypatch):
    """The perturbation/margin ratio is the number the tier rule rests on."""
    monkeypatch.setenv("MOTIONBRICKS_MLX_OPT", ROUND4_OPT)
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    batch = scenarios(2)
    reference, _ = bake.record_serial(pipeline, bundle, batch, capture_stages=True)
    result = bake.drive_batch(pipeline, bundle, batch, bake.batched_stages())
    perturbations = bake.stage_perturbations(reference, result.runs)
    margins = bake.decision_margins(reference)
    safety = bake.safety_factor(perturbations, margins)
    assert margins["pose_margin"] > 0.0
    assert margins["duration_margin"] > 0.0
    assert safety["pose_perturbation"] > 0.0
    # Batching is never bitwise, and it must stay inside the decision margin on
    # this fixture: a ratio above 1 would mean the campaign had already flipped a
    # token somewhere.
    assert 0.0 < safety["pose_ratio"] < 1.0


@NEEDS
def test_constraints_for_is_what_plan_runs():
    """``Agent.plan`` and the bake must build identical constraints.

    The round-4 bake runs one scenario through ``run_transition`` several times,
    so a second, drifting construction of the boundary conditions would silently
    make the batched arm incomparable.  ``plan`` therefore calls
    ``constraints_for`` too, and this pins that the object it hands to the
    pipeline is the one the bake recorded.
    """
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    scenario = scenarios(1)[0]
    agent = A.Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
    agent.reset(scenario.style)
    _, before = agent.constraints_for(scenario.command(), scenario.style)
    # Two calls in a row agree, which is the invariant the bake rests on: it
    # runs `run_transition` several times per scenario and every pass must see
    # one boundary set.  (The context only moves once `plan` finishes, and the
    # bake calls `plan` exactly once per scenario.)
    _, again = agent.constraints_for(scenario.command(), scenario.style)
    assert np.array_equal(before.poses, again.poses)
    assert np.array_equal(before.target_root_translations, again.target_root_translations)
    assert np.array_equal(before.allowed_tokens, again.allowed_tokens)

    seen: dict[str, object] = {}
    original = pipeline.run_transition

    def spy(constraints, *args, **kwargs):
        # Snapshot *before* returning: ``Motion`` keeps this object's arrays and
        # ``restore_world`` / ``apply_context_blend`` rewrite them in place, so
        # reading it afterwards would compare a mutated object.
        seen["constraints"] = {field: np.array(getattr(constraints, field),
                                               copy=True) for field in (
            "global_root", "local_root", "poses", "has_local_root",
            "allowed_tokens", "target_root_translations",
            "target_local_rotations_xyzw")}
        return original(constraints, *args, **kwargs)

    pipeline.run_transition = spy
    try:
        agent.plan(scenario.command(), scenario.style)
    finally:
        pipeline.run_transition = original

    for field, value in seen["constraints"].items():
        assert np.array_equal(getattr(before, field), value), field


@NEEDS
def test_driver_only_intercepts_the_stages_it_owns(monkeypatch):
    """A stage the driver does not own must be run by the accepted code.

    Regression: the four hooks used to fire whenever a driver was present, so a
    ``{pose, root}`` driver was still asked for the decoder, suspended on a stage
    it did not own, and the scheduler could never resolve it.  A subset
    configuration is how the report attributes the gain between stages, so the
    stage set has to be honoured at the seam.
    """
    monkeypatch.setenv("MOTIONBRICKS_MLX_OPT", "bake_root_batch=1")
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    batch = scenarios(2)
    stages = bake.batched_stages()
    assert stages == frozenset({"pose", "root"})
    calls: list[str] = []
    original = pipeline.decoder.decode

    def spy(*args, **kwargs):
        calls.append("decoder")
        return original(*args, **kwargs)

    pipeline.decoder.decode = spy
    try:
        reference, _ = bake.record_serial(pipeline, bundle, batch, capture_stages=True)
        after_reference = len(calls)
        result = bake.drive_batch(pipeline, bundle, batch, stages)
        after_bake = len(calls)
        result_again = bake.drive_batch(pipeline, bundle, batch, stages)
        after_second = len(calls)
    finally:
        pipeline.decoder.decode = original
    assert len(result.runs) == len(batch) == len(result_again.runs)
    # Exactly one serial decoder call per scenario per *completed* bake: the
    # passes that suspend before the decoder never reach it.
    assert after_reference - 0 == len(batch)
    assert after_bake - after_reference == len(batch)
    assert after_second - after_bake == len(batch)
    comparison = bake.compare([record.run for record in reference], result.runs)
    assert comparison["token_identical"] == len(batch)
    assert comparison["differing_tokens"] == 0
    assert result.stage_sizes["decoder"] == 0


# ---------------------------------------------------------------------------
# Round 5: the single-pass orchestrator
# ---------------------------------------------------------------------------

def test_manifest_fields_follow_decision_0007():
    """The tier needs both halves of the rule: campaign **and** margin.

    Decision 0007 defines the release tier as `min`(campaign-passing `B`, margin
    `>= 1.5x`), so a campaign that passes with the worst perturbation already at
    the tightest decision margin is a *draft*, not a release -- which is exactly
    the `B = 16` mixed case round 4 measured at `1.0426x`.
    """
    assert bake.BAKE_MANIFEST_FIELDS == (
        "batch_size", "bake_tier", "bake_stages", "bake_stage_switches",
        "bake_grouping", "bake_campaign_scenarios", "bake_token_decisions_differing",
        "bake_worst_roots_m", "bake_worst_rotations", "bake_safety_ratio",
        "bake_min_decision_margin", "bake_action_bit_identical")
    # Campaign passed and 1/ratio = 1.89x: release.
    assert bake.bake_tier(0.5290, True) == "reproducible"
    # Campaign passed but 1/ratio = 0.96x: draft (round 4's mixed B=16).
    assert bake.bake_tier(1.0426, True) == "draft"
    # Campaign failed: draft whatever the ratio says.
    assert bake.bake_tier(0.1, False) == "draft"
    # The threshold itself belongs to the release side.
    assert bake.bake_tier(1.0 / 1.5, True) == "reproducible"

    comparison = {"scenarios": 16, "token_identical": 16, "uniform_identical": 16,
                  "selected_identical": 16, "differing_tokens": 0,
                  "worst_roots_m": 2.027e-06, "worst_rotations": 2.753e-05}
    safety = {"pose_ratio": 1.0426, "pose_margin": 2.985e-03}
    fields = bake.manifest_fields(batch_size=16, stages={"root", "pose", "decoder"},
                                  grouping="duration-mixed", token_groups={"11": 5},
                                  comparison=comparison, safety=safety)
    assert set(fields) == set(bake.BAKE_MANIFEST_FIELDS)
    assert fields["bake_tier"] == "draft"
    assert fields["bake_stages"] == ["decoder", "pose", "root"]
    assert fields["bake_stage_switches"] == ["bake_root_batch", "bake_decoder_batch"]
    assert fields["bake_action_bit_identical"] is False
    assert fields["bake_min_decision_margin"] == 2.985e-03


@NEEDS
def test_single_pass_matches_the_round4_driver_bit_for_bit(monkeypatch):
    """Round 5's instrument check: the two batch arms may not differ at all.

    The single-pass orchestrator resolves the same forwards on the same rows;
    anything but exact equality would mean the throughput comparison against round
    4 is measuring two different bakes, so it is asserted with no tolerance.
    """
    monkeypatch.setenv("MOTIONBRICKS_MLX_OPT", ROUND4_OPT)
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    batch = scenarios(4)
    reference, _ = bake.record_serial(pipeline, bundle, batch, capture_stages=True)
    round4 = bake.drive_batch(pipeline, bundle, batch, bake.batched_stages())
    single = bake.drive_batch_single_pass(pipeline, bundle, batch, bake.batched_stages())
    assert len(single.runs) == len(batch)
    assert single.suspensions == 0 and single.drivers == []
    cross = bake.compare_bitwise(round4.runs, single.runs)
    assert cross["bit_identical"], cross["rows"]
    assert cross["tokens_identical"] == len(batch)
    assert cross["roots_identical"] == len(batch)
    assert cross["rotations_identical"] == len(batch)
    # And the campaign claim itself still holds against the serial reference.
    comparison = bake.compare([record.run for record in reference], single.runs)
    assert comparison["token_identical"] == len(batch)
    assert comparison["uniform_identical"] == len(batch)
    assert comparison["selected_identical"] == len(batch)
    assert comparison["differing_tokens"] == 0


@NEEDS
def test_single_pass_runs_every_host_step_once(monkeypatch):
    """The whole point: no `run_transition` restart, and the accepted prefixes
    of an unowned stage run once per scenario instead of once per boundary.

    With the round-3 stage set (``pose`` only) the probe and the root planner are
    the *accepted* code, so they are the observable: round 4's driver runs them
    once per stage boundary per scenario (twice), the single pass runs them once.
    """
    monkeypatch.delenv("MOTIONBRICKS_MLX_OPT", raising=False)
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    batch = scenarios(3)
    stages = bake.batched_stages()
    assert stages == frozenset({"pose"})
    counts = {"probe": 0, "plan": 0, "restarts": 0}
    originals = (pipeline.root.duration_logits, pipeline.root.plan,
                 pipeline.run_transition)

    def probe_spy(*args, **kwargs):
        counts["probe"] += 1
        return originals[0](*args, **kwargs)

    def plan_spy(*args, **kwargs):
        counts["plan"] += 1
        return originals[1](*args, **kwargs)

    def restart_spy(*args, **kwargs):
        counts["restarts"] += 1
        return originals[2](*args, **kwargs)

    pipeline.root.duration_logits = probe_spy
    pipeline.root.plan = plan_spy
    pipeline.run_transition = restart_spy
    try:
        round4 = bake.drive_batch(pipeline, bundle, batch, stages)
        after_round4 = dict(counts)
        counts.update(probe=0, plan=0, restarts=0)
        single = bake.drive_batch_single_pass(pipeline, bundle, batch, stages)
        after_single = dict(counts)
    finally:
        (pipeline.root.duration_logits, pipeline.root.plan,
         pipeline.run_transition) = originals

    assert round4.suspensions >= len(batch)
    # Round 4 restarts the transition at every boundary it owns; round 5 never
    # restarts one, and the accepted prefix runs once per scenario.
    assert after_single["restarts"] == 0
    assert after_round4["restarts"] >= len(batch)
    assert after_single["probe"] == len(batch)
    assert after_single["plan"] == len(batch)
    assert after_round4["probe"] == 2 * len(batch)
    assert after_round4["plan"] == 2 * len(batch)
    assert bake.compare_bitwise(round4.runs, single.runs)["bit_identical"]


@NEEDS
def test_single_pass_host_pass_runs_no_forward(monkeypatch):
    """The decomposition instrument: injected rows must skip every batched forward.

    ``host_pass_timing_single_pass`` is what separates the host glue from the
    orchestration in the report, so it must not silently run a neural stage.
    """
    monkeypatch.setenv("MOTIONBRICKS_MLX_OPT", ROUND4_OPT)
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    batch = scenarios(2)
    stages = bake.batched_stages()
    campaign = bake.drive_batch_single_pass(pipeline, bundle, batch, stages)
    calls = []
    originals = (pipeline.root.duration_logits_batch,
                 pipeline.root.plan_values_batch,
                 pipeline.pose.plan_batch,
                 pipeline.decoder.decode_batch)

    def spy(name, original):
        def wrapped(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)
        return wrapped

    (pipeline.root.duration_logits_batch, pipeline.root.plan_values_batch,
     pipeline.pose.plan_batch, pipeline.decoder.decode_batch) = (
        spy("probe", originals[0]), spy("plan", originals[1]),
        spy("pose", originals[2]), spy("decoder", originals[3]))
    try:
        timing = bake.host_pass_timing_single_pass(pipeline, bundle, batch, stages,
                                                   campaign.rows, repeats=2)
    finally:
        (pipeline.root.duration_logits_batch, pipeline.root.plan_values_batch,
         pipeline.pose.plan_batch, pipeline.decoder.decode_batch) = originals
    assert calls == []
    assert timing["repeats"] == 2 and timing["per_scenario_ms_p50"] > 0.0


@NEEDS
def test_single_pass_subset_stage_set(monkeypatch):
    """A subset stage set still batches only what it owns."""
    monkeypatch.setenv("MOTIONBRICKS_MLX_OPT", "bake_root_batch=1")
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    batch = scenarios(2)
    stages = bake.batched_stages()
    assert stages == frozenset({"pose", "root"})
    calls = []
    original = pipeline.decoder.decode

    def spy(*args, **kwargs):
        calls.append("decoder")
        return original(*args, **kwargs)

    pipeline.decoder.decode = spy
    try:
        single = bake.drive_batch_single_pass(pipeline, bundle, batch, stages)
    finally:
        pipeline.decoder.decode = original
    # The decoder is not owned, so the accepted forward runs once per scenario.
    assert len(calls) == len(batch)
    assert single.stage_sizes == {"probe": len(batch), "plan": len(batch),
                                  "pose": len(batch), "decoder": 0}


# ---------------------------------------------------------------------------
# Round 6: the decode as a fifth forward point (report section 43.1)
# ---------------------------------------------------------------------------

ROUND6_OPT = "bake_root_batch=1,bake_decoder_batch=1,bake_decode_batch=1"


def test_decode_batch_switch_is_declared_off_and_never_a_round4_stage(monkeypatch):
    """The switch is off by default, and the round-4 driver can never own it.

    ``batched_stages()`` is what ``StageDriver`` is built from, so if the decode
    ever appeared there the round-4 arm would be asked for a forward it has no
    method for -- and, worse, the round-3/round-4 bake numbers would stop being
    reproducible from the same switches.
    """
    monkeypatch.delenv("MOTIONBRICKS_MLX_OPT", raising=False)
    assert nn.OPT_DEFAULTS["bake_decode_batch"] is False
    assert nn._opt("bake_decode_batch") is False
    assert bake.DECODE_STAGE not in bake.batched_stages()
    assert bake.single_pass_stages(frozenset({"pose", "root", "decoder"})) == \
        frozenset({"pose", "root", "decoder"})
    monkeypatch.setenv("MOTIONBRICKS_MLX_OPT", ROUND6_OPT)
    assert nn._opt("bake_decode_batch") is True
    # Even switched on, the round-4 stage set never gains the decode...
    assert bake.DECODE_STAGE not in bake.batched_stages()
    # ...and only the single-pass orchestrator's set does.
    assert bake.single_pass_stages(bake.batched_stages()) == \
        frozenset({"pose", "root", "decoder", "decode"})


@NEEDS
def test_batched_decode_is_bit_identical_to_the_serial_decode(monkeypatch):
    """The batched decode is a *batch*, not a different computation.

    The standalone instrument (report section 45) says one batched decode graph is
    an order of magnitude cheaper than ``B`` small ones; this pins that it is also
    exact, on real decoder rows, array for array and with no tolerance.
    """
    from motionbricks_mlx.models import motion_rep

    monkeypatch.setenv("MOTIONBRICKS_MLX_OPT", ROUND6_OPT)
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    batch = scenarios(4)
    campaign = bake.drive_batch_single_pass(pipeline, bundle, batch,
                                            bake.batched_stages(), capture_trace=True)
    rows = [mx.array(np.asarray(row, np.float32)) for row in campaign.rows["decoder"]]
    # The orchestrator groups a decode batch by frame count, so the comparison
    # must too: a duration-mixed batch has more than one group, and that
    # fragmentation is part of what the campaign measures.
    groups: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(int(row.shape[0]), []).append(index)
    assert sum(len(indices) for indices in groups.values()) == len(rows)
    for indices in groups.values():
        stacked = mx.stack([rows[index] for index in indices], axis=0)
        origins = mx.array(np.zeros((len(indices), 1), np.float32))
        roots_b, rotations_b = motion_rep.decode_motion_batch(
            stacked, pipeline.mean, pipeline.std, pipeline.parents, origins, origins)
        mx.eval(roots_b, rotations_b)
        for position, index in enumerate(indices):
            roots_s, rotations_s = motion_rep.decode_motion(
                rows[index], pipeline.mean, pipeline.std, pipeline.parents, 0.0, 0.0)
            mx.eval(roots_s, rotations_s)
            assert np.array_equal(np.asarray(roots_b)[position], np.asarray(roots_s))
            assert np.array_equal(np.asarray(rotations_b)[position],
                                  np.asarray(rotations_s))
    bundle.release()


@NEEDS
def test_single_pass_decode_batch_matches_the_round4_arm_bit_for_bit(monkeypatch):
    """The round-5 instrument check, now against a batched *decode*.

    The round-4 arm runs the accepted per-scenario decode; the single-pass arm
    runs one batched decode per shape group.  Everything the campaign compares --
    tokens, uniforms, duration, roots, rotations -- must be exactly equal, with no
    tolerance, or the throughput gain is not the same bake.
    """
    monkeypatch.setenv("MOTIONBRICKS_MLX_OPT", ROUND6_OPT)
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    batch = scenarios(4)
    stages = bake.batched_stages()
    reference, _ = bake.record_serial(pipeline, bundle, batch, capture_stages=True)
    round4 = bake.drive_batch(pipeline, bundle, batch, stages)
    single = bake.drive_batch_single_pass(pipeline, bundle, batch, stages)
    cross = bake.compare_bitwise(round4.runs, single.runs)
    assert cross["bit_identical"], cross["rows"]
    comparison = bake.compare([record.run for record in reference], single.runs)
    assert comparison["token_identical"] == len(batch)
    assert comparison["differing_tokens"] == 0
    bundle.release()


@NEEDS
def test_single_pass_decode_batch_issues_one_decode_graph_per_group(monkeypatch):
    """One batched decode per shape group, and no per-scenario decode at all.

    A duration-homogeneous batch is one group, so the whole batch is one decode
    graph; the accepted ``decode_motion`` must not be called by the orchestrator.
    """
    from motionbricks_mlx.models import motion_rep

    monkeypatch.setenv("MOTIONBRICKS_MLX_OPT", ROUND6_OPT)
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    walk = load_style(STYLES_DIR / "walk.mbstyle")
    batch = [bake.Scenario(label=f"h{index}-walk", style=walk, seed=5 + index,
                           movement=(0.2, 0.0, 0.0)) for index in range(3)]
    stages = bake.batched_stages()
    counts = {"batched": 0, "serial": 0}
    originals = (motion_rep.decode_motion_batch, motion_rep.decode_motion)

    def batched_spy(*args, **kwargs):
        counts["batched"] += 1
        return originals[0](*args, **kwargs)

    def serial_spy(*args, **kwargs):
        counts["serial"] += 1
        return originals[1](*args, **kwargs)

    motion_rep.decode_motion_batch = batched_spy
    motion_rep.decode_motion = serial_spy
    try:
        single = bake.drive_batch_single_pass(pipeline, bundle, batch, stages)
    finally:
        (motion_rep.decode_motion_batch, motion_rep.decode_motion) = originals
    assert len(single.runs) == len(batch)
    assert counts["batched"] == 1, counts
    assert counts["serial"] == 0, counts
    bundle.release()


def test_manifest_tier_follows_decision_0008_threshold_and_grouping():
    """The tier is 0008's mechanical rule, not 0007's literal 1/1.5 arithmetic.

    Mixed ``B = 8`` measures 0.6695, which **fails** 0007's literal
    ``1/ratio >= 1.5`` (1.494) but passes 0008's ``ratio <= 0.70``; 0008 rules that
    the measured boundary -- not the integer heuristic -- is the threshold.  The
    grouping rule is the third leg: above ``B = 8`` a mixed batch is a draft even
    when its ratio is comfortable, because that is the configuration whose margin
    collapsed at ``B = 16``.
    """
    comparison = {"scenarios": 8, "token_identical": 8, "uniform_identical": 8,
                  "selected_identical": 8, "differing_tokens": 0,
                  "total_tokens": 536, "worst_roots_m": 1.267e-06,
                  "worst_rotations": 8.076e-06, "worst_angle_deg": 0.0626}
    stages = {"root", "pose", "decoder", "decode"}

    def fields(batch_size, grouping, ratio):
        return bake.manifest_fields(
            batch_size=batch_size, stages=stages, grouping=grouping,
            token_groups={"11": batch_size}, comparison=comparison,
            safety={"pose_ratio": ratio, "pose_margin": 2.985e-03})

    # 0007's stricter arithmetic would have called this draft; 0008 calls it release.
    assert 1.0 / 0.6695 < bake.RELEASE_MARGIN                 # 1.494 < 1.5
    assert 0.6695 <= bake.RELEASE_RATIO                       # 0.6695 <= 0.70
    assert fields(8, "duration-mixed", 0.6695)["bake_tier"] == "reproducible"
    assert fields(8, "duration-mixed", 0.6695)["bake_stage_switches"] == [
        "bake_root_batch", "bake_decoder_batch", "bake_decode_batch"]
    # Zero headroom is a draft whatever the grouping.
    assert fields(16, "duration-homogeneous", 1.0426)["bake_tier"] == "draft"
    # Homogeneous B=16 has headroom and is release; the same ratio mixed is not.
    assert fields(16, "duration-homogeneous", 0.4089)["bake_tier"] == "reproducible"
    assert fields(16, "duration-mixed", 0.4089)["bake_tier"] == "draft"
    # A failed campaign is never release.
    failed = dict(comparison, token_identical=7)
    assert bake.manifest_fields(
        batch_size=4, stages=stages, grouping="duration-mixed",
        token_groups={"11": 4}, comparison=failed,
        safety={"pose_ratio": 0.2, "pose_margin": 1.0})["bake_tier"] == "draft"
