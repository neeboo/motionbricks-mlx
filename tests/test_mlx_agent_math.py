"""Round-2 host maths: batched `data/agent.py` vs the verbatim reference.

`docs/optimization-report.md` section 12 fixed the rules before the change: the
batched host maths may be a default only if it is **token-identical** and leaves
`mlx_align` OK, and it is adopted once the paired p50 improves by >= 1.05x.

This file pins the numeric half of that:

* every rewritten entry point is **bit-identical** to the reference
  implementation it replaces, on the real bundle's skeleton and on real styles
  when the reference checkout is present (the rewrite was designed to be
  bit-identical, so the test asserts equality, not a tolerance);
* the two NumPy facts the rewrite rests on -- that `einsum` reproduces a per-row
  gemv where a batched `matrix @ vectors.T` gemm does not -- stay pinned, so a
  later "simplification" back to the gemm form fails here instead of silently
  reintroducing a last-bit difference that the pose planner amplifies;
* the end-to-end gate: 5 seeds x 6 planning steps produce identical
  `pose.tokens` and `pose.sampling_uniforms` with the batched maths on and off.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motionbricks_mlx.data import agent as A  # noqa: E402
from motionbricks_mlx.data.style import Style, load_style  # noqa: E402
from motionbricks_mlx.models.bundle import Bundle  # noqa: E402
from motionbricks_mlx.models.transition import MlxPipeline  # noqa: E402

MLX_BUNDLE = ROOT / "checkpoints/g1-f32-mlx"
HAVE_BUNDLE = (MLX_BUNDLE / "manifest.json").is_file()
STYLES_DIR = Path.home() / "dev/motion-bricks-cpp/repo/generated/styles"
HAVE_STYLES = STYLES_DIR.is_dir()
STYLE_NAMES = ("walk", "walk_zombie", "slow_walk", "walk_stealth")
SEEDS = (1, 18, 35, 52, 69)

pytestmark = pytest.mark.skipif(not HAVE_BUNDLE,
                                reason="run scripts/mlx_convert_gguf.py first")


@pytest.fixture(autouse=True)
def restore_switch():
    saved = os.environ.get("MOTIONBRICKS_MLX_AGENT")
    yield
    if saved is None:
        os.environ.pop("MOTIONBRICKS_MLX_AGENT", None)
    else:
        os.environ["MOTIONBRICKS_MLX_AGENT"] = saved


def test_agent_switch_grammar_and_default():
    """The host-math switch mirrors `MOTIONBRICKS_MLX_OPT`'s grammar and is on.

    Section 12.2 adopts the batched maths only at >= 1.05x paired p50 with
    identical tokens; the measured numbers are in section 14 of the report.
    `MOTIONBRICKS_MLX_AGENT=0` must restore the reference loops, and the
    in-order grammar must still express "reference plus nothing" and
    "reference plus the batched maths only".
    """
    os.environ.pop("MOTIONBRICKS_MLX_AGENT", None)
    assert A._agent_math_option() is True
    assert A.AGENT_MATH_DEFAULT is True
    for value, expected in (("0", False), ("1", True), ("off", False), ("on", True)):
        os.environ["MOTIONBRICKS_MLX_AGENT"] = value
        assert A._agent_math_option() is expected, value


def test_y_rotation_batch_matches_the_scalar_y_rotation():
    angles = np.array([0.0, 0.3, -1.2, 3.0, -3.0], dtype=np.float32)
    loop = np.stack([A.y_rotation(angle) for angle in angles])
    assert np.array_equal(loop, A.y_rotation_batch(angles))


def synthetic_style(frames: int = 20, seed: int = 7) -> Style:
    """A deterministic in-memory style, so the maths test needs no checkout."""
    rng = np.random.default_rng(seed)
    quaternions = rng.standard_normal((frames, A.JOINTS, 4)).astype(np.float32)
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
    rotations = np.stack([A.quaternion_matrix(quaternions[:, joint])
                          for joint in range(A.JOINTS)], axis=1)
    rotations = np.ascontiguousarray(rotations, dtype=np.float32)
    positions = np.cumsum(rng.standard_normal((frames, A.JOINTS, 3)).astype(np.float32), axis=0)
    roots = np.ascontiguousarray(rng.standard_normal((frames, 3)).astype(np.float32))
    headings = np.ascontiguousarray(rng.uniform(-np.pi, np.pi, frames).astype(np.float32))
    return Style("synthetic", 1.4, frames, positions, rotations, roots, headings,
                 np.ones(11, np.uint8))


def synthetic_history(style: Style, parents: np.ndarray, seed: int = 3):
    """A 4-frame agent context, i.e. what ``Agent.plan`` feeds ``canonicalize``."""
    roots, rotations = A._seed_agent_from_style_ref(style, parents)
    rng = np.random.default_rng(seed)
    roots = np.ascontiguousarray(roots + rng.standard_normal((4, 3)).astype(np.float32) * 0.05)
    return roots, rotations


def assert_pair_identical(reference, candidate, label: str) -> None:
    """``assert_bit_identical`` for a tuple of differently-shaped arrays."""
    assert len(reference) == len(candidate)
    for index, (left, right) in enumerate(zip(reference, candidate)):
        assert_bit_identical(left, right, f"{label}[{index}]")


def assert_bit_identical(reference, candidate, label: str) -> None:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    assert reference.shape == candidate.shape, f"{label}: {reference.shape} vs {candidate.shape}"
    if not np.array_equal(reference, candidate):
        difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
        index = np.unravel_index(int(np.argmax(difference)), difference.shape)
        raise AssertionError(
            f"{label}: {int(np.count_nonzero(difference))}/{difference.size} elements differ, "
            f"max|d| {difference.max():.3e} at {index}")


# --- the NumPy facts the rewrite rests on ---------------------------------

def test_dot_rows_reproduces_blas_sdot():
    """A batched dot must match ``np.dot``, not ``np.sum(a * b)``.

    ``slerp`` accumulates its cosine with ``np.dot`` (BLAS ``sdot``); for
    non-unit quaternions the elementwise-then-sum form differs in the last bit
    (measured 4.768e-07), which the batched ``_slerp_batch`` must not introduce.
    """
    rng = np.random.default_rng(13)
    left = rng.standard_normal((112, 4)).astype(np.float32)
    right = rng.standard_normal((112, 4)).astype(np.float32)
    loop = np.array([float(np.dot(left[row], right[row])) for row in range(112)],
                    dtype=np.float32)
    assert np.array_equal(loop, A.dot_rows(left, right))
    assert not np.array_equal(loop, np.sum(left * right, axis=-1).astype(np.float32))


def test_einsum_reproduces_the_per_row_matrix_vector_product():
    """A per-row ``matrix @ v`` (gemv) is not a batched ``matrix @ vectors.T``.

    The reference writes the target block as a per-joint matrix-vector product.
    NumPy sends the batched matrix-matrix form to a different BLAS kernel, which
    changes the last bit of a 3-term reduction -- measured 2.384e-07 here and
    2.980e-08 in the real walk_stealth constraints.  ``einsum`` matches the gemv
    bit for bit; that is why :func:`A.rotate_rows` exists.
    """
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((3, 3)).astype(np.float32)
    vectors = rng.standard_normal((64, 3)).astype(np.float32)
    loop = np.stack([matrix @ vectors[row] for row in range(vectors.shape[0])])
    assert np.array_equal(loop, A.rotate_rows(matrix, vectors))
    assert not np.array_equal(loop, (matrix @ vectors.T).T)


def test_batched_frames_product_matches_the_per_frame_loop():
    """``einsum('fij,fkj->fki')`` == the per-frame, per-joint gemv of the reference."""
    rng = np.random.default_rng(12)
    matrices = rng.standard_normal((4, 3, 3)).astype(np.float32)
    vectors = rng.standard_normal((4, 33, 3)).astype(np.float32)
    loop = np.stack([np.stack([matrices[frame] @ vectors[frame, joint]
                               for joint in range(33)]) for frame in range(4)])
    batched = np.einsum("fij,fkj->fki", matrices, vectors)
    assert np.array_equal(loop, batched)
    assert not np.array_equal(loop, (matrices @ vectors.transpose(0, 2, 1)).transpose(0, 2, 1))


# --- bit-identity of every rewritten entry point --------------------------

def test_switched_helpers_are_bit_identical_on_synthetic_input():
    bundle = Bundle(MLX_BUNDLE, device="cpu")
    parents, neutral = bundle.joint_parents, bundle.neutral_joints
    style = synthetic_style()
    roots, rotations = synthetic_history(style, parents)

    assert_pair_identical(A._seed_agent_from_style_ref(style, parents),
                          A._seed_agent_from_style_fast(style, parents),
                          "seed_agent_from_style")

    command = A.Command(style=style, movement=(1.0, 0.0, 0.0), seed=5)
    frame_ref, roots_ref, rotations_ref, command_ref = A._canonicalize_ref(
        roots, rotations, command)
    frame_fast, roots_fast, rotations_fast, command_fast = A._canonicalize_fast(
        roots, rotations, command)
    assert_bit_identical([frame_ref.x, frame_ref.z, frame_ref.heading],
                         [frame_fast.x, frame_fast.z, frame_fast.heading],
                         "canonicalize frame")
    assert_bit_identical(roots_ref, roots_fast, "canonicalize roots")
    assert_bit_identical(rotations_ref, rotations_fast, "canonicalize rotations")
    assert command_ref.movement == command_fast.movement
    assert command_ref.facing == command_fast.facing

    encoded_ref = A._encode_context_ref(roots_ref, rotations_ref, neutral, parents)
    encoded_fast = A._encode_context_fast(roots_ref, rotations_ref, neutral, parents)
    for name in ("global_root", "local_root", "poses"):
        assert_bit_identical(getattr(encoded_ref, name), getattr(encoded_fast, name),
                             f"encode_context.{name}")

    reference = A._build_constraints_ref(roots_ref, rotations_ref, command_ref, style,
                                         neutral, parents)
    candidate = A._build_constraints_fast(roots_ref, rotations_ref, command_ref, style,
                                          neutral, parents)
    assert set(reference) == set(candidate)
    for name in reference:
        assert_bit_identical(reference[name], candidate[name], f"build_constraints[{name}]")


def test_restore_world_and_context_blend_are_bit_identical_on_synthetic_input():
    bundle = Bundle(MLX_BUNDLE, device="cpu")
    parents, neutral = bundle.joint_parents, bundle.neutral_joints
    style = synthetic_style()
    roots, rotations = synthetic_history(style, parents)
    command = A.Command(style=style, seed=2)
    frame, canonical_roots, canonical_rotations, _ = A._canonicalize_ref(
        roots, rotations, command)

    rng = np.random.default_rng(21)
    frames = 17
    motion_args = dict(
        roots=rng.standard_normal((frames, 3)).astype(np.float32),
        rotations_xyzw=(rng.standard_normal((frames, A.JOINTS, 4)).astype(np.float32)),
        target_roots=rng.standard_normal((4, 3)).astype(np.float32),
        target_local_rotations_xyzw=rng.standard_normal((4, A.JOINTS, 4)).astype(np.float32),
    )
    reference = A.Motion(**{key: value.copy() for key, value in motion_args.items()})
    candidate = A.Motion(**{key: value.copy() for key, value in motion_args.items()})
    A._restore_world_ref(reference, frame)
    A._restore_world_fast(candidate, frame)
    for name in ("roots", "rotations_xyzw", "target_roots", "target_local_rotations_xyzw"):
        assert_bit_identical(getattr(reference, name), getattr(candidate, name),
                             f"restore_world.{name}")

    A._apply_context_blend_ref(canonical_roots, canonical_rotations, reference)
    A._apply_context_blend_fast(canonical_roots, canonical_rotations, candidate)
    for name in ("roots", "rotations_xyzw"):
        assert_bit_identical(getattr(reference, name), getattr(candidate, name),
                             f"apply_context_blend.{name}")


@pytest.mark.skipif(not HAVE_STYLES, reason="reference style checkout not present")
def test_switched_helpers_are_bit_identical_on_real_styles():
    """Every style x seed x command shape the agent can produce, on real assets."""
    bundle = Bundle(MLX_BUNDLE, device="cpu")
    parents, neutral = bundle.joint_parents, bundle.neutral_joints
    for name in STYLE_NAMES:
        style = load_style(STYLES_DIR / f"{name}.mbstyle")
        roots, rotations = A._seed_agent_from_style_ref(style, parents)
        assert_pair_identical(A._seed_agent_from_style_ref(style, parents),
                              A._seed_agent_from_style_fast(style, parents),
                              f"{name}: seed_agent_from_style")
        for seed in SEEDS:
            for movement, facing in (((0.0, 0.0), (0.0, 1.0)),
                                     ((1.0, 0.0), (1.0, 0.0)),
                                     ((0.0, 0.0), (0.0, -1.0))):
                command = A.Command(style=style, movement=(movement[0], 0.0, movement[1]),
                                    facing=(facing[0], 0.0, facing[1]), seed=seed)
                ref = A._canonicalize_ref(roots, rotations, command)
                fast = A._canonicalize_fast(roots, rotations, command)
                assert_bit_identical(ref[1], fast[1], f"{name}/{seed}: canonicalize roots")
                assert_bit_identical(ref[2], fast[2], f"{name}/{seed}: canonicalize rotations")
                reference = A._build_constraints_ref(ref[1], ref[2], ref[3], style,
                                                     neutral, parents)
                candidate = A._build_constraints_fast(fast[1], fast[2], fast[3], style,
                                                      neutral, parents)
                for key in reference:
                    assert_bit_identical(reference[key], candidate[key],
                                         f"{name}/{seed}/{movement}: build_constraints[{key}]")


# --- end-to-end gate -------------------------------------------------------

def run_campaign(device: str, switch: str, steps: int = 6):
    """``SEEDS`` x ``steps`` plans on the real styles; returns tokens/uniforms."""
    os.environ["MOTIONBRICKS_MLX_AGENT"] = switch
    bundle = Bundle(MLX_BUNDLE, device=device)
    pipeline = MlxPipeline(bundle)
    styles = {name: load_style(STYLES_DIR / f"{name}.mbstyle") for name in STYLE_NAMES}
    collected = []
    for seed in SEEDS:
        agent = A.Agent(pipeline, bundle.neutral_joints, bundle.joint_parents)
        agent.reset(styles["walk"])
        for index in range(steps):
            name = STYLE_NAMES[index % len(STYLE_NAMES)]
            style = styles[name]
            if index and index % 3 == 0:
                agent.advance(2)
            command = A.Command(style=style, movement=(0.2, 0.0, 0.0),
                                facing=(0.0, 0.0, 1.0), seed=seed + index)
            _, info = agent.plan(command, style, capture_trace=True)
            collected.append((np.asarray(info["trace"]["pose.tokens"]).copy(),
                              np.asarray(info["trace"]["pose.sampling_uniforms"]).copy()))
    return collected


@pytest.mark.skipif(not HAVE_STYLES, reason="reference style checkout not present")
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_token_identity_end_to_end(device):
    """The gate: identical discrete tokens and sampling stream, switch on vs off."""
    reference = run_campaign(device, "0")
    candidate = run_campaign(device, "1")
    assert len(reference) == len(candidate) == len(SEEDS) * 6
    for index, ((tokens_ref, uniforms_ref), (tokens_new, uniforms_new)) in enumerate(
            zip(reference, candidate)):
        assert np.array_equal(tokens_ref, tokens_new), f"{device}: step {index} tokens differ"
        assert np.array_equal(uniforms_ref, uniforms_new), \
            f"{device}: step {index} sampling_uniforms differ"
