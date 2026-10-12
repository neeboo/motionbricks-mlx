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
