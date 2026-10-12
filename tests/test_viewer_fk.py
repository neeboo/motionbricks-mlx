"""The viewer's batched forward kinematics must equal the exporter's loop form.

``scripts/mlx_viewer.py`` computes ``world_positions`` for every clip it sends
to the browser.  That array is what the page renders, so it has to be the *same*
array the repository's own plan documents carry -- otherwise the viewer would be
showing something the accepted pipeline never produced.  The batched form keeps
the joint loop (tree order) and batches the frame axis; every output element is
the same fixed float64 expression, so the two agree bit for bit rather than
merely within a tolerance.  These tests are pure NumPy: no checkpoint, no MLX.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mlx_viewer import parse_pair, world_positions  # noqa: E402
from motionbricks_mlx.export.plan_json import world_positions_xyz  # noqa: E402

# A small branching skeleton in the project's own parent-before-child order.
PARENTS = np.array([-1, 0, 1, 1, 0, 4, 4], dtype=np.int32)
REST = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.5, 0.0],
    [0.2, 0.5, 0.0],
    [0.0, 0.5, 0.0],
    [0.3, 0.4, 0.0],
    [-0.3, 0.4, 0.0],
], dtype=np.float32)


def random_motion(frames: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    quaternions = generator.normal(size=(frames, PARENTS.shape[0], 4)).astype(np.float32)
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
    roots = generator.normal(size=(frames, 3)).astype(np.float32)
    return roots, quaternions


@pytest.mark.parametrize("frames,seed", [(1, 0), (2, 3), (44, 7), (24, 11)])
def test_batched_fk_is_bit_identical(frames: int, seed: int) -> None:
    roots, quaternions = random_motion(frames, seed)
    reference = world_positions_xyz(roots, quaternions, PARENTS, REST)
    batched = world_positions(roots, quaternions, PARENTS, REST)
    assert batched.shape == reference.shape
    # Bit equality, not closeness: the viewer must render the accepted array.
    assert np.array_equal(reference, batched)
    assert (reference == batched).all()


def test_batched_fk_matches_reference_on_float32_boundaries() -> None:
    """Extreme magnitudes and a zero quaternion: still exactly the same bits."""
    generator = np.random.default_rng(1234)
    quaternions = generator.normal(size=(6, PARENTS.shape[0], 4)).astype(np.float32) * np.float32(1e4)
    quaternions[0, 1] = np.float32(0.0)
    roots = (generator.normal(size=(6, 3)).astype(np.float32) * np.float32(1e3))
    reference = world_positions_xyz(roots, quaternions, PARENTS, REST)
    batched = world_positions(roots, quaternions, PARENTS, REST)
    assert np.array_equal(reference, batched)


def test_batched_fk_single_frame_is_the_root() -> None:
    roots, quaternions = random_motion(1, 5)
    batched = world_positions(roots, quaternions, PARENTS, REST)
    assert np.array_equal(batched[0, 0], roots[0].astype(np.float64))


def test_parse_pair_accepts_and_rejects() -> None:
    assert parse_pair(None, (0.0, 1.0)) == [0.0, 1.0]
    assert parse_pair("", (0.0, 1.0)) == [0.0, 1.0]
    assert parse_pair("-1,0.5", (0.0, 0.0)) == [-1.0, 0.5]
    # The page posts JSON, curl uses the query string; both must work.
    assert parse_pair([0, 1], (0.0, 0.0)) == [0.0, 1.0]
    assert parse_pair((0.25, -0.5), (0.0, 0.0)) == [0.25, -0.5]
    with pytest.raises(ValueError):
        parse_pair("1", (0.0, 1.0))
    with pytest.raises(ValueError):
        parse_pair("1,2,3", (0.0, 1.0))
    with pytest.raises(ValueError):
        parse_pair([1, 2, 3], (0.0, 1.0))
    with pytest.raises(ValueError):
        parse_pair("left,right", (0.0, 1.0))
