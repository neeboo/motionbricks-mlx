"""Motion quality evaluation and regression gate (C-line).

Six metric families over plan documents (the reference Go driver's JSON
shape), a threshold table with provenance, five corruptions for negative
controls, and the gate that ties them together.  See ``docs/brief-glm-eval.md``
and ``docs/status-glm.md`` for the numbers.
"""

from .corruptions import (
    add_jitter_noise,
    apply_corruption,
    freeze_frames,
    raise_feet,
    shuffle_frames,
    translate_roots,
)
from .document import (
    FOOT_JOINTS_LEFT,
    FOOT_JOINTS_RIGHT,
    PlanStep,
    load_step,
    load_steps,
    recompute_world_positions,
    skeleton_tables,
    validate_structure,
)
from .gate import GateReport, evaluate_step, run_gate, run_negative_controls
from .metrics import (
    boundary_error,
    foot_sliding,
    grounding,
    jitter,
    motion_deviation,
    token_agreement,
)
from .thresholds import THRESHOLDS, Verdict, judge

__all__ = [
    "FOOT_JOINTS_LEFT",
    "FOOT_JOINTS_RIGHT",
    "GateReport",
    "PlanStep",
    "THRESHOLDS",
    "Verdict",
    "add_jitter_noise",
    "apply_corruption",
    "boundary_error",
    "evaluate_step",
    "foot_sliding",
    "freeze_frames",
    "grounding",
    "judge",
    "jitter",
    "load_step",
    "load_steps",
    "motion_deviation",
    "raise_feet",
    "recompute_world_positions",
    "run_gate",
    "run_negative_controls",
    "shuffle_frames",
    "skeleton_tables",
    "token_agreement",
    "translate_roots",
    "validate_structure",
]
