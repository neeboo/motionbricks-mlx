"""Plan document writer: the same JSON shape as the reference Go driver.

``~/dev/motion-bricks-cpp/tools/mbplan/main.go`` emits, per step: the motion
(root translations ``[frames*3]`` + local rotations ``[frames*34*4]``), the four
placed target frames, forward-kinematics world joint positions, and summary
numbers.  Producing byte-comparable structure is what makes a mechanical
array-by-array comparison possible.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 30


def _quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Go driver's ``qmul`` (xyzw Hamilton product)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def _quaternion_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Go driver's ``qrot``: ``q * (v,0) * conj(q)`` vector part."""
    return _quaternion_multiply(_quaternion_multiply(q, np.array([*vector, 0.0])),
                                np.array([-q[0], -q[1], -q[2], q[3]]))[:3]


def world_positions_xyz(roots: np.ndarray, rotations_xyzw: np.ndarray,
                        parents: np.ndarray, rest_positions: np.ndarray) -> np.ndarray:
    """Forward kinematics exactly as the Go driver computes ``world_positions_xyz``."""
    frames, joints = roots.shape[0], rotations_xyzw.shape[1]
    positions = np.zeros((frames, joints, 3), dtype=np.float64)
    for frame in range(frames):
        world = np.zeros((joints, 4), dtype=np.float64)
        for joint in range(joints):
            rotation = rotations_xyzw[frame, joint].astype(np.float64)
            parent = int(parents[joint])
            if parent < 0:
                positions[frame, joint] = roots[frame].astype(np.float64)
                world[joint] = rotation
                continue
            rest = (rest_positions[joint] - rest_positions[parent]).astype(np.float64)
            positions[frame, joint] = positions[frame, parent] + \
                _quaternion_rotate(world[parent], rest)
            world[joint] = _quaternion_multiply(world[parent], rotation)
    return positions


@dataclass
class Step:
    label: str
    style: str
    move: tuple[float, float]
    facing: tuple[float, float]
    seed: int
    advance: int
    roots: np.ndarray                    # (frames, 3)
    rotations_xyzw: np.ndarray           # (frames, 34, 4)
    target_roots: np.ndarray             # (4, 3)
    target_rotations_xyzw: np.ndarray    # (4, 34, 4)


def step_document(step: Step, parents: np.ndarray, rest: np.ndarray,
                  skeleton: list[dict] | None = None) -> dict:
    frames, joints = step.roots.shape[0], step.rotations_xyzw.shape[1]
    world = world_positions_xyz(step.roots, step.rotations_xyzw, parents, rest)
    travel = float(np.sum(np.linalg.norm(np.diff(step.roots.astype(np.float64), axis=0), axis=1))) \
        if frames > 1 else 0.0
    document = {
        "label": step.label,
        "style": step.style,
        "move": [float(step.move[0]), float(step.move[1])],
        "facing": [float(step.facing[0]), float(step.facing[1])],
        "seed": int(step.seed),
        "advance": int(step.advance),
        "motion": {
            "frames": int(frames),
            "joints": int(joints),
            "roots": [float(v) for v in step.roots.reshape(-1)],
            "rotations": [float(v) for v in step.rotations_xyzw.reshape(-1)],
        },
        "targets": {
            "frames": int(step.target_roots.shape[0]),
            "joints": int(joints),
            "roots": [float(v) for v in step.target_roots.reshape(-1)],
            "rotations": [float(v) for v in step.target_rotations_xyzw.reshape(-1)],
        },
        "world_positions_xyz": [float(v) for v in world.reshape(-1)],
        "root_first_frame": [float(v) for v in step.roots[0]],
        "root_last_frame": [float(v) for v in step.roots[-1]],
        "root_travel_metres": travel,
        "duration_seconds": float(frames) / FPS,
    }
    if skeleton is not None:
        document["skeleton"] = skeleton
    return document


def build_document(library: str, model_bundle: str, device: str, sampling: str,
                   learned_parameters: int, steps: list[Step], parents: np.ndarray,
                   rest: np.ndarray, joint_names: list[str]) -> dict:
    skeleton = [{"name": joint_names[joint], "parent": int(parents[joint]),
                 "position": [float(v) for v in rest[joint]]}
                for joint in range(parents.shape[0])]
    documents = []
    for index, step in enumerate(steps):
        documents.append(step_document(step, parents, rest,
                                       skeleton if index == 0 else None))
    return {
        "library": library,
        "model_bundle": model_bundle,
        "device": device,
        "sampling": sampling,
        "fps": FPS,
        "joints": int(parents.shape[0]),
        "learned_f32_parameters": int(learned_parameters),
        "steps": documents,
    }


def write_document(path: str | Path, document: dict) -> None:
    Path(path).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation angle between two unit quaternions (xyzw), in radians."""
    dot = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


__all__ = ["FPS", "Step", "build_document", "step_document", "write_document",
           "world_positions_xyz", "angle_between", "math"]
