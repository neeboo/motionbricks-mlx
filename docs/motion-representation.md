# Motion representation (v0 draft)

One clip = one fixed-fps sequence of frames. Default **30 fps**, length 1–10 s.

## Conventions (state them once, convert at the edges)
* Source capture: our Unity space — **Y-up, left-handed** (Unity).
* Training/interchange: **Z-up, right-handed** (robotics/ML norm), metres, radians.
* Conversion lives in `motion/retarget.py`; it is a hard requirement that a round
  trip of a captured clip is bit-stable within float32 tolerance.

## Per-frame features
| Block | Fields | Dim | Notes |
|---|---|---|---|
| root position | x, y, z | 3 | world; y/z up per convention |
| root rotation | yaw only | 1 | characters stay upright; full orientation is out of scope for v0 |
| root velocity | dx, dy, dz | 3 | finite difference, smoothed 3-frame |
| joint rotations | 6D rotation per joint | 6·J | Swivel-free, no gimbal wrap; J = skeleton joint count |
| joint velocities | 3D angular | 3·J | optional block |
| contacts | per-foot boolean | 2 | from ground-truth in the simulator, not inferred |
| phase | sin/cos of cycle | 2 | for locomotion style |

Facing direction is **yaw + forward vector**, never a quaternion, so the latent
space has no antipodal discontinuity.

## Clip-level features
`style` (one-hot over the style set), `target` (point or object id), `start_root`,
`end_root`, `fps`, `skeleton_id`, `provenance` (space id, capture id, git sha).

## What is deliberately out of v0
Upper-body object interaction beyond a single hand target; multi-character
scenes; cloth/foot IK; retarget to a different skeleton than the capture rig's.

## Storage on disk
`datasets/<space>/<capture>/clip.npz` with `frames (T, D)`, `offsets`, `meta.json`.
`D` is derived from the offsets table, so adding a block is a schema bump and the
loaders keep working on older shards.
