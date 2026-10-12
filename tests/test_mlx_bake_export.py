"""Round-7 product-side bake: the export payload, the campaign, the twelve fields.

Rounds 3-6 pinned the *machinery* of the bake (``tests/test_mlx_bake.py``).  Round
7 adds the product side: a bake is finished as a full plan document and shipped
with the manifest decision ``0007`` item 4 requires.  This file pins the pieces of
that which can silently go wrong:

* ``BatchResult.motions`` must be **additive**: empty on the round-4 driver, one
  motion per scenario (matching that scenario's ``Run``) on the single pass --
  otherwise a product export would have to reconstruct the targets from
  somewhere else and could drift from what was actually planned;
* the campaign definition must really be what its docstring claims (labels
  unique, the ``B = 8`` slice covering >= 4 real styles and >= 3 seed groups, the
  single-style control being one style over sixteen seeds), because the
  duration-homogeneity that lets one batched forward cover the batch is a
  property of the rows;
* the manifest check must fail closed: twelve required fields, and a missing one
  is a failure rather than a note;
* the float32 digests the equivalence table quotes must be bit-exact, or
  "identical" would mean "close".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motionbricks_mlx.data.style import load_style  # noqa: E402
from motionbricks_mlx.models import bake  # noqa: E402
from motionbricks_mlx.models.bundle import Bundle  # noqa: E402
from motionbricks_mlx.models.transition import MlxPipeline  # noqa: E402

MLX_BUNDLE = ROOT / "checkpoints/g1-f32-mlx"
HAVE_BUNDLE = (MLX_BUNDLE / "manifest.json").is_file()
STYLES_DIR = Path.home() / "dev/motion-bricks-cpp/repo/generated/styles"
HAVE_STYLES = STYLES_DIR.is_dir()
NEEDS = pytest.mark.skipif(not (HAVE_BUNDLE and HAVE_STYLES),
                           reason="bundle and reference styles required")

#: The round-6 release bake stage set, as an environment string.
RELEASE_OPT = "bake_root_batch=1,bake_decoder_batch=1,bake_decode_batch=1"


def load_library_script():
    """Import ``scripts/mlx_bake_library.py`` by path (``scripts`` is not a package)."""
    path = ROOT / "scripts" / "mlx_bake_library.py"
    spec = importlib.util.spec_from_file_location("mlx_bake_library", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["mlx_bake_library"] = module
    spec.loader.exec_module(module)
    return module


def test_batch_result_motions_is_additive_and_defaults_empty():
    """The round-4 driver builds a ``BatchResult`` positionally; a new field must
    not disturb that, and must not pretend to carry motions it never produced."""
    result = bake.BatchResult(["a"], [], [], 0, {"pose": 1})
    assert result.motions == []
    assert result.rows == {}
    assert result.agents == []


def test_campaign_definition_is_what_the_docstring_claims():
    """The duration-homogeneous claim rests on the rows, so the rows are checked."""
    module = load_library_script()
    diverse = module.CAMPAIGNS["diverse"]
    control = module.CAMPAIGNS["single-style"]

    # Labels are the join key between the batch arm and the single arm.
    for campaign in (diverse, control):
        labels = [row.label for row in campaign]
        assert len(labels) == len(set(labels)) == 16
        assert all(row.movement for row in campaign)

    # The B = 8 slice and the whole B = 16 campaign both carry the required
    # diversity: >= 4 real styles and >= 3 distinct seed groups.
    for rows in (module.rows_for(diverse, 8), module.rows_for(diverse, 16)):
        assert len({row.style for row in rows}) >= 4
        assert len({row.seed for row in rows}) >= 3
    assert len({row.style for row in diverse}) == 8
    assert len({row.seed for row in diverse}) == 14

    # The control isolates style diversity: one style, sixteen seeds.
    assert {row.style for row in control} == {"walk"}
    assert len({row.seed for row in control}) == 16

    # Slicing is a prefix and bounds-checks.
    assert module.rows_for(diverse, 8) == diverse[:8]
    with pytest.raises(SystemExit):
        module.rows_for(diverse, 17)


def test_array_digest_is_bit_exact():
    """The equivalence table's digests must separate one float32 ULP."""
    module = load_library_script()
    base = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    step = np.nextafter(base, np.float32(np.inf))
    assert module.array_digest(base) == module.array_digest(base)
    assert module.array_digest(base) != module.array_digest(step)
    # float64 input is compared at float32 precision, which is the document's.
    assert module.array_digest(base.astype(np.float64)) == module.array_digest(base)


def test_manifest_check_fails_closed(tmp_path, capsys):
    """Twelve fields, or the check fails -- for the *written* manifest, not a dict."""
    module = load_library_script()
    complete = {field: None for field in bake.BAKE_MANIFEST_FIELDS}
    complete["bake_tier"] = "reproducible"
    payload = {"groups": {"B16": dict(complete)}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert module.cmd_check(argparse.Namespace(manifest=path, out=None)) == 0
    assert "manifest check: PASS" in capsys.readouterr().out

    broken = dict(complete)
    del broken["bake_safety_ratio"]
    payload = {"groups": {"B16": broken}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert module.cmd_check(argparse.Namespace(manifest=path, out=None)) == 1
    out = capsys.readouterr().out
    assert "MISSING ['bake_safety_ratio']" in out
    assert "manifest check: FAIL" in out


@NEEDS
def test_single_pass_exposes_every_scenario_motion_but_the_driver_does_not(monkeypatch):
    """One ``Motion`` per scenario on the single pass, none on the round-4 arm.

    The exported document needs the four placed target keyframes, which only the
    finished ``Motion`` carries; this pins that the orchestrator hands out exactly
    the motion it finished, and that the round-4 control arm is untouched.
    """
    monkeypatch.setenv("MOTIONBRICKS_MLX_OPT", RELEASE_OPT)
    bundle = Bundle(MLX_BUNDLE, device="gpu")
    pipeline = MlxPipeline(bundle)
    names = ("walk", "walk_zombie")
    batch = [bake.Scenario(label=f"m{index}-{names[index]}",
                           style=load_style(STYLES_DIR / f"{names[index]}.mbstyle"),
                           seed=1 + index, movement=(0.2, 0.0, 0.0))
             for index in range(2)]
    stages = bake.batched_stages()
    assert stages == frozenset({"pose", "root", "decoder"})
    assert bake.single_pass_stages(stages) == frozenset(
        {"pose", "root", "decoder", "decode"})

    single = bake.drive_batch_single_pass(pipeline, bundle, batch, stages)
    assert len(single.motions) == len(batch) == len(single.runs)
    for motion, run in zip(single.motions, single.runs):
        assert motion.frames == run.roots.shape[0]
        assert np.array_equal(motion.roots, run.roots)
        assert np.array_equal(motion.rotations_xyzw, run.rotations)
        assert motion.target_roots.shape == (4, 3)
        assert motion.target_local_rotations_xyzw.shape == (4, 34, 4)
        assert np.isfinite(motion.target_roots).all()

    round4 = bake.drive_batch(pipeline, bundle, batch, stages)
    assert round4.motions == []
    assert len(round4.runs) == len(batch)
