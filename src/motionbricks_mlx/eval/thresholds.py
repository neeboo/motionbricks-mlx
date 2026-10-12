"""Threshold table, measured baselines and their provenance.

Every limit below is **measured baseline + headroom**, never a round number
picked for comfort.  The baselines come from two sources, both recorded here
and in ``docs/status-glm.md``:

1. ``BASELINE`` — metrics over the fixed 6-seed f32 MLX scenario set
   (``scripts/eval_collect.py``, bundle ``checkpoints/g1-f32-mlx``), the
   "good motion" the gate must keep passing.
2. ``REFERENCE_BUILD_DELTA`` — the reference runtime's own Debug-vs-Release
   difference on the same transition (``out/ref/synth-ref.json`` vs
   ``synth-ref-release.json``), i.e. how much numerical wobble is inherent to
   *acceptable* builds of the same implementation.  Cross-implementation
   gates (metric 5) sit above this magnitude.

Values below are filled from the calibration run recorded in
``docs/status-glm.md``; if a metric's algorithm changes, recalibrate.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- algorithm constants (not gates, but part of the metric definition) -----
# A foot is in contact when its lowest point is below this height.  Measured
# on the f32 baseline (docs/status-glm.md): the lowest foot point of a single
# foot stays below ~3.3 cm even mid-walk, so 5 cm means "near the plane".
CONTACT_THRESHOLD_M = 0.05
# "Airborne" = every foot point higher than this.  Baseline max of the lowest
# foot point is +0.033 m; a 0.12 m floor leaves 3.6x headroom while a 25 cm
# whole-body lift (negative control ④) reads 0.28 m and trips it.
AIR_THRESHOLD_M = 0.12
# Penetration depth the pose may not exceed.  Baseline minimum across the
# calibration set is -0.019 m (foot joint centres in deep stance); -0.05 is
# 2.6x that depth while remaining far above a real clip-through.
PENETRATION_FLOOR_M = -0.05
# A joint counts as motionless when its frame-to-frame angle is below this.
FREEZE_EPS_DEG = 0.01

# --- metric 5 gates: the shipped alignment tolerances -----------------------
# scripts/mlx_align.py TOLERANCES, which the f32 MLX port passes with
# 3+ orders of magnitude headroom.  Calibration run (docs/status-glm.md):
# same-seed pairs at seeds 1 and 77 measure roots 5.2e-07..4.0e-06 m,
# rotation max 0.047 deg, world 3.3e-05 m, token agreement 1.0 — and the
# reference's own Debug<->Release delta on the same transition is smaller
# still (roots 2.6e-07 m, rotation max 0.043 deg).  So these bounds accept
# every good pairing and reject real drift.
CONSISTENCY_ROOTS_M = 1.0e-3
CONSISTENCY_ROTATION_DEG = 0.2
CONSISTENCY_WORLD_M = 2.0e-3
TOKEN_AGREEMENT_MIN = 1.0  # exact match — pose.tokens with tolerance 0.0

# --- quality gates (calibrated on the f32 baseline, see docs/status-glm.md) -
# BASELINE: 8 good motions = 6 seed-group walk steps (seeds 1/7/13/42/77/123,
# move (1,0)) + the 2 reference-scenario steps (walk standing, walk_zombie
# replan).  Each limit is baseline extreme + headroom; the negative controls
# clear their gates by 1.2-40x.
@dataclass(frozen=True)
class Thresholds:
    # structure: no problems allowed (frames/joints/fps/finite/quats/FK link)
    structure_problems_max: int = 0
    # metric 1 — foot sliding.  Baseline 0.081 (standing) to 1.100 m (walking):
    # this model's gait drags its toes (toe_base stays within ~3.4 cm of the
    # plane while the ankle lifts), so near-ground contact is almost permanent
    # and the metric accumulates ~ the step's travel.  The absolute value is
    # gait-style dominated; the gate at baseline max + 10% still separates the
    # freeze corruption (+~0.4 m, control ②) and shuffling.
    foot_slide_max_m: float = 1.21
    # metric 2 — jerk: noise drives p95/max UP; a stall shows as freeze_frac.
    # Baseline p95 0.73-2.17, max 3.1-17.2 deg, freeze_frac 0.000-0.061
    # (the standing step idles with near-still joints).  The 0.15 stall gate
    # trips for any freeze longer than ~7 frames at 30 fps.
    jerk_p50_deg_min: float = 0.05
    jerk_p95_deg_max: float = 4.0
    jerk_max_deg_max: float = 30.0
    freeze_frac_max: float = 0.15
    # metric 3 — grounding: gate on penetration DEPTH (the h<0 fraction is
    # 0.41-0.50 on GOOD motion — reported, not gated).  Baseline min
    # -0.0191 m; -0.05 is 2.6x deeper.
    foot_height_min_m: float = PENETRATION_FLOOR_M
    ground_airborne_frac_max: float = 0.10
    # metric 4 — boundary continuity with the targets block.  The transition
    # CONDITIONS on the targets, it does not copy them: the baseline's own
    # boundary error reaches 0.068 m / 21.5 deg (style-change replan), so the
    # gates sit above that band (+120% / +21%); a 0.7 m context displacement
    # (control ⑤) is 11x past the root gate.
    boundary_root_max_m: float = 0.15
    boundary_rotation_max_deg: float = 26.0
    # metric 5 — cross-implementation
    token_agreement_frac_min: float = TOKEN_AGREEMENT_MIN
    deviation_roots_max_m: float = CONSISTENCY_ROOTS_M
    deviation_rotation_max_deg: float = CONSISTENCY_ROTATION_DEG
    deviation_world_max_m: float = CONSISTENCY_WORLD_M


THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class Verdict:
    key: str
    value: float
    limit: float
    direction: str  # "max" (value <= limit) or "min" (value >= limit)
    passed: bool

    def line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        if self.direction == "max":
            relation = f"<= {self.limit:g}"
        else:
            relation = f">= {self.limit:g}"
        return f"  {status} {self.key:28s} {self.value:.6g} {relation}"


def judge(values: dict, thresholds: Thresholds = THRESHOLDS) -> list[Verdict]:
    """Turn one step's metric values into per-key verdicts."""
    checks = [
        ("foot_slide_max_m", thresholds.foot_slide_max_m, "max"),
        ("jerk_p50_deg", thresholds.jerk_p50_deg_min, "min"),
        ("jerk_p95_deg", thresholds.jerk_p95_deg_max, "max"),
        ("jerk_max_deg", thresholds.jerk_max_deg_max, "max"),
        ("freeze_frac", thresholds.freeze_frac_max, "max"),
        ("foot_height_min_m", thresholds.foot_height_min_m, "min"),
        ("ground_airborne_frac", thresholds.ground_airborne_frac_max, "max"),
        ("boundary_root_max_m", thresholds.boundary_root_max_m, "max"),
        ("boundary_rotation_max_deg", thresholds.boundary_rotation_max_deg, "max"),
    ]
    verdicts = []
    for key, limit, direction in checks:
        if key not in values:
            continue
        value = float(values[key])
        passed = value <= limit if direction == "max" else value >= limit
        verdicts.append(Verdict(key, value, limit, direction, passed))
    return verdicts


def judge_consistency(values: dict, thresholds: Thresholds = THRESHOLDS) -> list[Verdict]:
    """Metric 5 verdicts for one (mine, reference) pair."""
    checks = [
        ("token_agreement_frac", thresholds.token_agreement_frac_min, "min"),
        ("deviation_roots_max_m", thresholds.deviation_roots_max_m, "max"),
        ("deviation_rotation_max_deg", thresholds.deviation_rotation_max_deg, "max"),
        ("deviation_world_max_m", thresholds.deviation_world_max_m, "max"),
    ]
    verdicts = []
    for key, limit, direction in checks:
        if key not in values:
            continue
        value = float(values[key])
        passed = value <= limit if direction == "max" else value >= limit
        verdicts.append(Verdict(key, value, limit, direction, passed))
    return verdicts


__all__ = [
    "AIR_THRESHOLD_M",
    "CONTACT_THRESHOLD_M",
    "FREEZE_EPS_DEG",
    "PENETRATION_FLOOR_M",
    "THRESHOLDS",
    "Thresholds",
    "Verdict",
    "judge",
    "judge_consistency",
]
