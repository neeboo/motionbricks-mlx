#!/usr/bin/env python3
"""Numerical alignment report: MLX port vs the reference runtime.

Two modes, both reporting absolute and relative error per array and naming the
first divergent tensor/layer if a tolerance is exceeded.

  plan   compare two plan documents (ours vs the reference Go driver's)
  trace  feed a reference transition trace's own constraints through the MLX
         pipeline and compare every intermediate stage (the only way to point
         at "the first layer that drifts")

Tolerances are declared in TOLERANCES with the reasoning behind them.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motionbricks_mlx.export.plan_json import angle_between
from motionbricks_mlx.models.bundle import Bundle
from motionbricks_mlx.models.transition import MlxPipeline, TransitionConstraints

# Tolerance policy.  The port is expected to match the reference to float32
# round-off, not bitwise: the reference accumulates in GGML kernels (its own
# summation order, F32 im2col) and this port accumulates in MLX.  A float32 dot
# product over K terms has a relative error bound of about K*2^-24; the deepest
# path here is a 16-layer transformer plus a conv stack, so a few times 1e-6
# relative is the expected floor.  Absolute tolerances below reflect the
# magnitudes actually involved (root translations ~1 m, quaternion components
# in [-1, 1]).
TOLERANCES = {
    "roots_m": 1.0e-3,
    "rotations": 2.0e-3,
    "rotation_angle_deg": 0.2,
    "target_roots_m": 1.0e-3,
    "target_rotations": 2.0e-3,
    "world_positions_m": 2.0e-3,
}
# Stage tolerances for the trace mode: intermediates see smaller magnitudes and
# fewer accumulated layers, but the pose logits are amplified by the pose
# planner's own sensitivity to the root condition (measured ~8e3 in the worst
# direction), so they get a wider budget.
STAGE_TOLERANCES = {
    "root.num_token_logits": 5.0e-3,
    "root.pred_global_root_values": 1.0e-4,
    "root.pred_local_root_values": 1.0e-3,
    "pose.logits": 5.0e-2,
    "pose.sampling_uniforms": 0.0,
    "pose.tokens": 0.0,
    "decoder.quantized": 0.0,
    "decoder.output": 1.0e-3,
    "decoded.roots": 1.0e-3,
}


@dataclass
class Comparison:
    name: str
    maximum: float
    rms: float
    scale: float
    relative: float
    tolerance: float
    index: int

    @property
    def passed(self) -> bool:
        return self.maximum <= self.tolerance

    def line(self) -> str:
        status = "ok  " if self.passed else "FAIL"
        return (f"  {status} {self.name:26s} max|Δ| {self.maximum:.3e}  rms {self.rms:.3e}  "
                f"rel {self.relative:.3e}  scale {self.scale:.4g}  tol {self.tolerance:.1e}"
                + ("" if self.passed else f"  first@{self.index}"))


def compare(name: str, mine: np.ndarray, reference: np.ndarray, tolerance: float) -> Comparison:
    mine = np.asarray(mine, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if mine.shape != reference.shape:
        raise SystemExit(f"{name}: shape mismatch {mine.shape} vs {reference.shape}")
    difference = np.abs(mine - reference)
    maximum = float(difference.max()) if difference.size else 0.0
    scale = float(np.abs(reference).max()) if reference.size else 0.0
    return Comparison(name, maximum, float(np.sqrt(np.mean(difference ** 2))), scale,
                      maximum / scale if scale else 0.0, tolerance,
                      int(np.argmax(difference)) if difference.size else 0)


def plan_mode(args: argparse.Namespace) -> int:
    mine = json.loads(Path(args.mine).read_text(encoding="utf-8"))
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    if len(mine["steps"]) != len(reference["steps"]):
        raise SystemExit("step count mismatch")
    results: list[Comparison] = []
    print(f"MLX {mine['device']} / {mine['sampling']}  vs  reference "
          f"{reference['device']} / {reference['sampling']}")
    for ours, theirs in zip(mine["steps"], reference["steps"]):
        print(f"step {theirs['label']} (style={theirs['style']} seed={theirs['seed']} "
              f"advance={theirs['advance']})")
        if ours["motion"]["frames"] != theirs["motion"]["frames"]:
            print(f"  FAIL frame count {ours['motion']['frames']} != "
                  f"{theirs['motion']['frames']}")
            return 1
        roots_a = np.array(ours["motion"]["roots"]).reshape(-1, 3)
        roots_b = np.array(theirs["motion"]["roots"]).reshape(-1, 3)
        rotations_a = np.array(ours["motion"]["rotations"]).reshape(-1, 34, 4)
        rotations_b = np.array(theirs["motion"]["rotations"]).reshape(-1, 34, 4)
        results.append(compare("roots (m)", roots_a, roots_b, TOLERANCES["roots_m"]))
        results.append(compare("rotations", rotations_a, rotations_b,
                               TOLERANCES["rotations"]))
        angles = np.degrees(angle_between(rotations_a, rotations_b))
        results.append(compare("rotation angle (deg)", angles,
                               np.zeros_like(angles), TOLERANCES["rotation_angle_deg"]))
        results.append(compare("target roots (m)",
                               np.array(ours["targets"]["roots"]).reshape(-1, 3),
                               np.array(theirs["targets"]["roots"]).reshape(-1, 3),
                               TOLERANCES["target_roots_m"]))
        results.append(compare("target rotations",
                               np.array(ours["targets"]["rotations"]).reshape(-1, 34, 4),
                               np.array(theirs["targets"]["rotations"]).reshape(-1, 34, 4),
                               TOLERANCES["target_rotations"]))
        results.append(compare("world positions (m)",
                               np.array(ours["world_positions_xyz"]).reshape(-1, 3),
                               np.array(theirs["world_positions_xyz"]).reshape(-1, 3),
                               TOLERANCES["world_positions_m"]))
    for result in results:
        print(result.line())
    worst = max(results, key=lambda item: item.maximum / item.tolerance if item.tolerance else 0)
    failures = [item for item in results if not item.passed]
    print(f"worst margin: {worst.name} at {worst.maximum / worst.tolerance:.2f}x tolerance"
          if worst.tolerance else "")
    if failures:
        print(f"ALIGNMENT FAILED: {len(failures)} array(s) outside tolerance")
        return 1
    print("ALIGNMENT OK: every compared array is inside its declared tolerance")
    return 0


def trace_mode(args: argparse.Namespace) -> int:
    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    tensors = trace["tensors"]
    bundle = Bundle(args.bundle, device=args.device)
    pipeline = MlxPipeline(bundle)
    constraints = TransitionConstraints(
        global_root=np.array(tensors["input.global_root_values"], np.float32).reshape(8, 5),
        local_root=np.array(tensors["input.local_root_values"], np.float32).reshape(8, 4),
        poses=np.array(tensors["input.local_poses"], np.float32).reshape(8, 303),
        has_global_root=np.array(tensors["input.has_global_root_values"], np.uint8).reshape(8),
        has_local_root=np.array(tensors["input.has_local_root_values"], np.uint8).reshape(8),
        has_poses=np.array(tensors["input.has_local_poses"], np.uint8).reshape(8),
        allowed_tokens=np.array(tensors["input.allowed_pred_num_tokens"], np.uint8).reshape(11))
    result = pipeline.run_transition(constraints, seed=args.seed,
                                     sampling_argmax=args.argmax, capture_trace=True)
    print(f"reference trace {args.trace} (seed {args.seed}, "
          f"tokens {trace['selected_tokens']}) vs MLX {args.device}")
    if result.selected_tokens != trace["selected_tokens"]:
        print(f"  FAIL selected tokens {result.selected_tokens} != {trace['selected_tokens']}")
        return 1
    print(f"  ok   selected tokens {result.selected_tokens}")
    failures = 0
    for name, tolerance in STAGE_TOLERANCES.items():
        if name not in result.trace or name not in tensors:
            print(f"  --   {name:26s} not captured on both sides")
            continue
        comparison = compare(name, result.trace[name],
                             np.array(tensors[name], np.float32), tolerance)
        print(comparison.line())
        failures += 0 if comparison.passed else 1
    print("TRACE OK: every stage inside tolerance" if not failures
          else f"TRACE FAILED: {failures} stage(s) outside tolerance")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("plan", help="compare two plan documents")
    plan.add_argument("reference")
    plan.add_argument("mine")
    plan.set_defaults(function=plan_mode)
    trace = sub.add_parser("trace", help="compare per-stage against a reference trace")
    trace.add_argument("trace")
    trace.add_argument("--bundle", type=Path,
                       default=Path(__file__).resolve().parents[1] / "checkpoints/g1-f32-mlx")
    trace.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    trace.add_argument("--seed", type=int, default=1)
    trace.add_argument("--argmax", action="store_true")
    trace.set_defaults(function=trace_mode)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
