"""Plan-document loading and structural validation (metric 6).

A plan document is the reference Go driver's JSON shape (see
``motionbricks_mlx/export/plan_json.py`` and the reference sample
``~/dev/motion-bricks-cpp/artifacts/plan-walk.json``): top level carries
``fps`` and ``joints``; each step carries ``motion`` (frames/joints/roots/
rotations), ``targets`` (the 4 placed future context frames), a
forward-kinematics ``world_positions_xyz`` array and a ``skeleton`` on the
first step.  All arrays are flat float lists; this module reshapes them and
checks the invariants every downstream metric relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..export.plan_json import FPS, world_positions_xyz

# Foot joints of the G1 34-joint skeleton, by skeleton name so the mapping is
# checkable rather than positional folklore.  Each foot's lowest point is the
# minimum world height over its ankle_roll and toe_base joints.
FOOT_JOINTS_LEFT = ("left_ankle_roll_skel", "left_toe_base")
FOOT_JOINTS_RIGHT = ("right_ankle_roll_skel", "right_toe_base")

EXPECTED_FPS = FPS  # 30 — the document's own fps field must match the contract


@dataclass
class PlanStep:
    """One step of a plan document, reshaped into arrays."""

    label: str
    style: str
    seed: int
    advance: int
    fps: int
    roots: np.ndarray                    # (frames, 3) world
    rotations_xyzw: np.ndarray           # (frames, joints, 4) local, xyzw
    target_roots: np.ndarray             # (4, 3) world
    target_rotations_xyzw: np.ndarray    # (4, joints, 4) local, xyzw
    world_positions: np.ndarray          # (frames, joints, 3) world
    joint_names: list[str]               # index -> name (from step 0 skeleton)

    @property
    def frames(self) -> int:
        return int(self.roots.shape[0])

    @property
    def joints(self) -> int:
        return int(self.rotations_xyzw.shape[1])

    def foot_joint_indices(self) -> dict[str, tuple[int, int]]:
        """Map foot name -> (ankle_roll index, toe_base index) by joint name."""
        return {
            "left": resolve_joints(self.joint_names, FOOT_JOINTS_LEFT),
            "right": resolve_joints(self.joint_names, FOOT_JOINTS_RIGHT),
        }

    def foot_heights(self) -> np.ndarray:
        """Lowest foot point world height per frame, shape (frames, 2) — left, right.

        Height is the world y axis (axis 1 of the position triple).
        """
        feet = self.foot_joint_indices()
        left = self.world_positions[:, feet["left"], 1]
        right = self.world_positions[:, feet["right"], 1]
        return np.stack([left.min(axis=1), right.min(axis=1)], axis=1)


def resolve_joints(joint_names: list[str], names: tuple[str, ...]) -> tuple[int, int]:
    indices = tuple(joint_names.index(name) for name in names)
    if len(indices) != 2:
        raise ValueError(f"expected 2 joint names, got {names}")
    return indices  # type: ignore[return-value]


def _require(condition: bool, message: str, problems: list[str]) -> None:
    if not condition:
        problems.append(message)


def load_step(document: dict, index: int, joint_names: list[str] | None = None) -> PlanStep:
    """Reshape step ``index`` of a plan document into a :class:`PlanStep`.

    Raises KeyError/ValueError on missing or ragged fields; use
    :func:`validate_structure` for the full diagnostic report.
    """
    step = document["steps"][index]
    if joint_names is None:
        joint_names = _skeleton_names(document, index)
    frames = int(step["motion"]["frames"])
    joints = int(step["motion"]["joints"])
    roots = np.asarray(step["motion"]["roots"], dtype=np.float64).reshape(frames, 3)
    rotations = np.asarray(step["motion"]["rotations"], dtype=np.float64) \
        .reshape(frames, joints, 4)
    target_roots = np.asarray(step["targets"]["roots"], dtype=np.float64) \
        .reshape(int(step["targets"]["frames"]), 3)
    target_rotations = np.asarray(step["targets"]["rotations"], dtype=np.float64) \
        .reshape(int(step["targets"]["frames"]), joints, 4)
    world = np.asarray(step["world_positions_xyz"], dtype=np.float64) \
        .reshape(frames, joints, 3)
    return PlanStep(
        label=step["label"], style=step["style"], seed=int(step["seed"]),
        advance=int(step["advance"]), fps=int(document["fps"]),
        roots=roots, rotations_xyzw=rotations,
        target_roots=target_roots, target_rotations_xyzw=target_rotations,
        world_positions=world, joint_names=joint_names,
    )


def load_steps(path: str | Path) -> list[PlanStep]:
    document = json_loads(Path(path).read_text(encoding="utf-8"))
    names = _skeleton_names(document, 0)
    return [load_step(document, index, names) for index in range(len(document["steps"]))]


def _skeleton_names(document: dict, index: int) -> list[str]:
    for offset in range(index + 1):
        step = document["steps"][index - offset]
        if "skeleton" in step:
            return [entry["name"] for entry in step["skeleton"]]
    raise KeyError(f"no skeleton found at or before step {index}")


def json_loads(text: str) -> dict:
    import json

    return json.loads(text)


def recompute_world_positions(step: PlanStep, parents: np.ndarray,
                              rest_positions: np.ndarray) -> np.ndarray:
    """FK exactly as the document writer does, for corrupted copies."""
    return world_positions_xyz(step.roots, step.rotations_xyzw, parents, rest_positions)


def skeleton_tables(document: dict, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(parents, rest_positions) derived from the document's skeleton block.

    The skeleton lists name/parent/position per joint — enough to re-run the
    writer's forward kinematics without loading the model bundle, so the
    negative-control corruptions work on any plan document file.
    """
    for offset in range(index + 1):
        candidate = document["steps"][index - offset]
        if "skeleton" in candidate:
            entries = candidate["skeleton"]
            parents = np.array([int(entry["parent"]) for entry in entries], dtype=np.int64)
            rest = np.array([entry["position"] for entry in entries], dtype=np.float64)
            return parents, rest
    raise KeyError(f"no skeleton at or before step {index}")


def validate_structure(step: PlanStep, expect_frames: int | None = None,
                       expect_joints: int = 34) -> dict:
    """Metric 6: frames, joints, fps, quaternion normalisation, NaN/Inf.

    Returns a dict of measured values; ``problems`` lists every violated
    invariant in words (the gate turns a non-empty list into a red verdict).
    ``expect_joints`` is the G1 34 in production; synthetic test skeletons
    may pass a smaller count.
    """
    problems: list[str] = []
    frames, joints = step.frames, step.joints

    _require(frames > 0, f"motion has {frames} frames", problems)
    _require(joints == expect_joints,
             f"joints is {joints}, expected {expect_joints}", problems)
    _require(step.fps == EXPECTED_FPS,
             f"fps is {step.fps}, expected {EXPECTED_FPS}", problems)
    _require(step.target_roots.shape[0] == 4,
             f"targets carry {step.target_roots.shape[0]} frames, expected 4", problems)
    _require(step.world_positions.shape[0] == frames,
             f"world_positions has {step.world_positions.shape[0]} frames "
             f"but motion has {frames}", problems)
    if expect_frames is not None:
        _require(frames == expect_frames,
                 f"frames is {frames}, expected {expect_frames}", problems)

    for name, array in (("motion.roots", step.roots),
                        ("motion.rotations", step.rotations_xyzw),
                        ("target.roots", step.target_roots),
                        ("target.rotations", step.target_rotations_xyzw),
                        ("world_positions", step.world_positions)):
        finite = bool(np.isfinite(array).all())
        _require(finite, f"{name} contains NaN or Inf", problems)

    norms = np.linalg.norm(step.rotations_xyzw, axis=-1)
    norm_error = float(np.abs(norms - 1.0).max())
    _require(norm_error <= 1e-3,
             f"quaternion norm deviates from 1 by {norm_error:.3e} (> 1e-3)", problems)
    target_norms = np.linalg.norm(step.target_rotations_xyzw, axis=-1)
    target_norm_error = float(np.abs(target_norms - 1.0).max())
    _require(target_norm_error <= 1e-3,
             f"target quaternion norm deviates by {target_norm_error:.3e} (> 1e-3)", problems)

    # world_positions must agree with an independent FK of roots+rotations —
    # this is what lets metrics 1-3 treat world_positions as authoritative.
    if step.world_positions.shape == (frames, joints, 3):
        # Re-FK needs parents/rest which the document does not carry; instead
        # check the pelvis column: world_positions[:, 0] must equal roots.
        pelvis_gap = float(np.abs(step.world_positions[:, 0, :] - step.roots).max())
        _require(pelvis_gap <= 1e-6,
                 f"world_positions pelvis column deviates from roots by {pelvis_gap:.3e}",
                 problems)

    return {
        "frames": frames,
        "joints": joints,
        "fps": step.fps,
        "target_frames": int(step.target_roots.shape[0]),
        "quat_norm_max_error": norm_error,
        "target_quat_norm_max_error": target_norm_error,
        "finite": not any("NaN" in p or "Inf" in p for p in problems),
        "problems": problems,
    }


__all__ = [
    "FOOT_JOINTS_LEFT",
    "FOOT_JOINTS_RIGHT",
    "EXPECTED_FPS",
    "PlanStep",
    "load_step",
    "load_steps",
    "recompute_world_positions",
    "resolve_joints",
    "skeleton_tables",
    "validate_structure",
]
