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
