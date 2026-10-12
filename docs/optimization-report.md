# MLX inference optimization report

Frozen at `2026-10-12T20:51:25+08:00`, before any code change, at commit
`bc78bd844a6cdaa44408e76385671b86d9a1385f`.  Everything below this line in this
section is the *adoption criteria*; it was written first and is not edited
afterwards.  Results are appended in the sections that follow.

## 0. Frozen adoption criteria

Baseline (accepted state, same M1 Max / bundle / planning path / 25 steps,
recorded in `out/bench-mlx.json` and `out/final-verification.txt`):

| quantity | value |
| --- | --- |
| MLX GPU (Metal) per-step | p50 **55.5 ms**, p90 67.3, max 165.9 |
| MLX CPU per-step | p50 196.3 ms |
| GGML CPU (Release) reference | p50 385.8 ms |
| MLX cold start (load + first plan) | **385.3 ms** |
| MLX peak | 748 MB (MLX) / 779 MB, process RSS 851 MB |
| numerics | roots 6.8e-7 m, rotations 8.7e-6, FK 2.7e-6 (worst 0.26x tolerance) |
| `pose.tokens` vs reference | 0 differences |
| tests | `pytest 33 passed, 1 skipped` |

### 0.1 Correctness gate (a candidate is rejected outright if it fails this)

A candidate is *numerically admissible* only if all of the following hold, on
the same bundle and the same seeds:

1. `scripts/mlx_align.py plan` and `scripts/mlx_align.py trace` both still
   report `ALIGNMENT OK` / `TRACE OK`, with no array outside its declared
   tolerance.
2. **Token identity.** For N >= 5 distinct seeds, `pose.tokens` is
   **element-for-element identical** to the pre-optimization f32 baseline on
   the same seed.  This is the primary gate because the pose planner amplifies
   root-condition error by ~1e3-1e4 and is followed by an argmax/Gumbel
   decision, so a numerical perturbation that is invisible in float terms can
   still flip a token.
3. `pose.sampling_uniforms` is also element-for-element identical, so any
   change to the sampling stream is caught separately from a change to the
   logits.
4. Determinism: two runs with the same seed stay bit-for-bit identical
   (`roots` and `rotations_xyzw`), for the optimized path as well.

If a candidate *cannot* satisfy token identity, it may still be considered, but
only under an explicitly restricted rule that has to be declared before the
measurement:

* allowed only if `token_difference_rate <= 0 %`** or** the measured difference
  rate is written down together with the resulting action deviation, and the
  deviation is `<= 0.5 x` the `mlx_align.py` tolerance for roots, rotations and
  world positions;
* and the admissible scope must be named explicitly (`real-time` /
  `batch-bake` / `offline-only`).  Anything above `0.5 x` tolerance is
  rejected regardless of speed.

**No post-hoc criterion.**  The decision for each item is taken by applying the
rules above to the numbers measured after the change; the reason is recorded
even when the item is rejected.

### 0.2 Performance gate

A candidate is *adopted* only if **either**

* per-step p50 latency improves by **>= 1.5x** (MLX GPU, 25 steps, same
  bundle/seed/backend), **or**
* cold start **or** resident memory improves by **>= 2.0x**.

Otherwise it is recorded as **rejected**, with the measured number and the
reason.  Secondary effects (p90/max tail, CPU backend, throughput) are reported
but do not by themselves justify adoption.

### 0.3 Measurement discipline

* The machine is a shared desktop; `uptime` load average is recorded with every
  measurement.  Because absolute latency moves with background load, every A/B
  pair is measured **interleaved inside one process** (A,B,A,B,... on the
  identical 25-step sequence), so drift affects both arms.
* Every measurement is >= 25 steps and reports p50/p90/max.
* Optimization and baseline use the same bundle, the same seeds, the same
  backend.
* Determinism is rechecked for each adopted configuration.

### 0.4 Scope / red lines

* Editable: `src/motionbricks_mlx/models/{nn,root,pose,decoder,transition,bundle,sampling}.py`,
  `scripts/mlx_bench.py`, new `tests/test_mlx_*.py`, `docs/status-deepseek.md`,
  and this file.
* `scripts/mlx_plan.py` and `scripts/mlx_align.py` keep their existing
  semantics; only additive, non-default switches are allowed.
* Never touched: `src/motionbricks_mlx/training/**`, `scripts/train_*.py`,
  `docs/motion-representation.md`, `README.md`, `NOTICE`, `LICENSE`,
  `pyproject.toml`, and the read-only reference checkout
  `~/dev/motion-bricks-cpp`.
* Local commits only; no push.
* Every new behaviour is switchable by an environment flag whose default
  reproduces the pre-optimization path, so each A/B can be rerun.

---

Everything below was written after the measurements.  The criteria above were
not edited.

## 1. Measurement conditions

Machine: M1 Max, 10 cores, 64 GB, macOS, MLX 0.32.3.

**The machine was never idle.**  The load average was recorded at every
measurement and stayed between **6.3 and 17.4** (1-minute), typically ~7-9, and
there was a burst at 17.4 during the 21:00 paired runs.  The load is the shared
desktop, not this work: no Python/training process was running at any point
(`ps` showed `Clash Verge / verge-mihomo` at ~85 % of one core, `WindowServer`
~40 %, the DSH renderer ~32 %, Chrome, Lark and Codex).  The other workstream's
training code was never observed running.

Consequence for the numbers: absolute latency drifts by roughly 10 % between
runs of the *same* configuration, so

* every before/after pair below was measured **interleaved inside one process**
  (all arms driven through the identical 25-step sequence in lockstep, with the
  arm order rotated every step so no arm systematically pays the "first in the
  step" cost), and
* the headline figure quoted is the **paired ratio** (per matched step,
  `baseline / candidate`) with its inter-quartile range, alongside the pooled
  p50; the plain single-arm bench run is quoted as corroboration only.

Observed load averages by measurement: 20:48 `8.02 8.74 10.99` (baseline
profile) · 20:49 `8.53 8.71 10.86` · 20:52 `6.32 7.51 9.92` · 20:55
`8.63 8.20 9.67` · 20:56 `7.46 7.97 9.54` · 20:59 `8.17 8.10 9.27` · 21:00-21:01
`6.93 … 17.42` (pairs) · 21:04 `13.11 11.99 10.73` · 21:06 `7.31 10.07 10.15` ·
21:09 `6.29 … 8.31` · 21:11 `6.27 8.37 9.37` · 21:12 `8.64 8.70 9.42`.

## 2. Baseline reproduction

The accepted state was reproduced on this machine before touching anything
(load `8.02`):

| quantity | recorded baseline | reproduced |
| --- | --- | --- |
| MLX GPU per-step p50 | 55.5 ms | **55.7 ms** |
| MLX CPU per-step p50 | 196.3 ms | 184.5-196 ms |
| cold start | 385.3 ms | 315-321 ms |
| MLX peak / process RSS | 748-779 MB / 851 MB | 779 MB / 830-835 MB |
| `pytest` | 33 passed 1 skipped | 124 passed 1 skipped (suite has grown) |

The GPU p50 and the memory numbers match; cold start reproduces ~20 % faster
than the recorded figure, which is machine-state dependent.

## 3. Where the time actually goes

Stage-level instrumentation of `run_transition` (25 steps, GPU, load 8.5):

| bucket | ms/step | share |
| --- | --- | --- |
| `pose.plan` (16-layer transformer) | 12.9 | 23 % |
| `root.plan` (2 transformers + conv decoder) | 9.3 | 17 % |
| `decoder.decode` (conv stack) | 5.5 | 10 % |
| `root.duration_logits` (probe) | 2.5 | 4 % |
| `sampling.uniforms` (host Python loop) | 2.1 | 4 % |
| `run_transition` host NumPy, other | ~5.4 | 10 % |
| `data/agent.py` host NumPy (`build_constraints` 14.5, `restore_world` 2.6, `encode_context` 1.9, `apply_context_blend` 0.7, `canonicalize` 0.3) | 19.9 | 36 % |

Two facts decided the whole investigation:

1. **The MLX stages are GPU-exec bound, not dispatch bound.**  Splitting each
   stage into lazy-graph construction and `mx.eval`, the graph build is
   0.12-0.21 ms while exec is 1.5-8.4 ms (**92-98 % exec**).  The brief's
   hypothesis ("slow in small-operator dispatch and serial chains") is half
   right: it is the *serial chain of dependent GPU kernels*, each on a tiny
   batch (`positions` <= 16), not host dispatch.  This is why `mx.compile`
   buys so little: it optimises exactly the 2-8 % that is not the problem.
2. **36 % of every step is host-side NumPy inside `data/agent.py`**, which is
   outside this work's edit grant (see section 10).

## 4. Item 1 — fixed shapes, no repeated graph compilation

### What was found

* `positions`/`num_tokens` take 11 values (tokens 6..16) per 25-step session,
  so any shape-keyed cache has up to 11 variants.  Measured compile cost: the
  first `mx.compile` in a process costs ~2 s (Metal kernel JIT), every
  subsequent variant ~3-12 ms; a 25-step session touches **6** distinct token
  counts, i.e. ~60 ms of variant compilation, which is why the compile variants
  are memoised per `(positions, num_tokens)` in `nn.StageCache`.
* `shapeless=True` is **not** usable: the attention path contains a `Slice`
  primitive and MLX raises `[Primitive::output_shapes] Slice cannot infer
  output shapes`.  Per-shape variants are therefore unavoidable.
* Rebuilt on every call before this work: `mx.array(mask)` (a fresh host→device
  upload of a 16x16 mask that is *all zeros* at the only call site), the
  head-offset index arithmetic, `_position_emb.embed.reshape(16,1024)[0:p]`,
  `_proj_num_valid_positions.weight[n-6]`, `_middle_token_emb.weight[n-6]`,
  `_position_emb.embed[0:p]`, plus `mx.broadcast_to` of the duration row.

### Changes

1. **`root_dedup`** — `MlxPipeline.run_transition` ran the duration probe once
   through `RootPlanner.duration_logits` and then **again inside
   `RootPlanner.plan`**, where `plan`'s own copy was discarded
   (`_, predicted_global = self.root.plan(...)`).  `plan` now accepts
   `duration_logits=` and skips the probe transformer.  This is pure duplicate
   work removal: boundary embedding + 3-layer transformer + output projection +
   one device sync per step.
2. **`pose_fast` / `root_fast` / `decoder_fast` (hoisting)** — every per-call
   constant above is cached per shape; the all-zero mask is dropped entirely
   (`scores + 0` is exact and `exp(-0.0) == exp(+0.0)`, so plain softmax is
   bit-identical to softmax under an additive zero mask).
3. **`pose_compile` / `root_compile` / `decoder_compile`** — the stage bodies go
   through `mx.compile`, memoised per shape in `nn.StageCache`.

### Numbers (GPU, 25 steps, interleaved, paired ratio vs the accepted path)

| candidate | paired p50 ratio | pooled p50 | p90 ratio | verdict |
| --- | --- | --- | --- | --- |
| `root_dedup` | 1.034x [0.949-1.161] | 1.017x | 1.067x | **rejected** (< 1.5x) |
| hoisting only (`pose_fast`, no compile) | 0.975x [0.903-1.082] | 1.000x | 1.005x | **rejected** (no measurable effect) |
| `pose_compile` | 1.018x [0.925-1.114] | 1.003x | 0.991x | **rejected** (< 1.5x) |
| `root_compile` + `decoder_compile` | 0.955x [0.896-1.062] | 0.982x | 0.987x | **rejected** (no effect) |
| item 1 combined (dedup + hoist + all compiles) | ~1.05x | ~1.05x | — | **rejected** (< 1.5x) |

Stage-level effect of compiling the MLX stages (sum over 25 steps):
`pose.plan` 331.9 -> 310.8 ms (-6 %), `root.plan` 244.0 -> 209.8 (-14 %),
`decoder.decode` 165.8 -> 147.3 (-11 %), `root.duration_logits` 76.5 -> 68.9
(-10 %).  Real but small: 2.0 ms/step of a 63.6 ms step.

**Correctness of item 1**: the hoisting and the dedup are bit-identical by
construction and measured bit-identical (16/16 steps, and 30/30 steps
end-to-end).  The `mx.compile` fusion is **bit-identical on the GPU backend**
(30/30 steps bitwise, 0.000e+00 deviation) and **not bit-identical on the CPU
backend** (3.576e-07 m on roots = 3e-4 of the declared tolerance, 3.651e-06 on
rotations); `pose.tokens` and `pose.sampling_uniforms` stay exactly identical on
both backends, which is the actual gate.

**Side effect that also argues against the CPU path**: on the CPU backend
`mx.compile` raises peak MLX memory from **748 MB to 1481 MB** (process RSS
861 -> 1612 MB) with no latency gain (188.7 vs 184.5 ms p50).  On the GPU
backend peak memory is unchanged (779 MB).

## 5. Item 2 — batching

Batching *across planning steps* is not available: the agent's context changes
after each plan, so step *n+1* cannot start before step *n* finishes.  There is
also no independent work inside one plan to batch (the two root transformers
use different weights, the 16 pose layers are a strict dependency chain, and the
two decoder stages are dependent).

What is available is batching **B independent scenarios** (e.g. the same context
evaluated under B styles, or B rollout samples for a motion-matching search)
through the pose transformer.  Measured on the real pose weights, 16 layers,
`positions=11`, GPU:

| B | B separate forwards | one batched forward | per scenario | throughput gain | bitwise vs separate |
| --- | --- | --- | --- | --- | --- |
| 1 | 6.55 ms | 7.14 ms | 7.14 ms | 1.00x | yes |
| 2 | 17.50 ms | 7.39 ms | 3.69 ms | **2.37x** | yes (0.000e+00) |
| 4 | 40.15 ms | 12.87 ms | 3.22 ms | **3.12x** | no (1.478e-05) |
| 8 | 88.57 ms | 14.65 ms | 1.83 ms | **6.05x** | no (1.669e-05) |

Interpretation: the GPU is heavily underutilised at B=1, so extra scenarios are
nearly free and throughput per scenario rises 2.4-6.1x while batch *latency*
only grows from 7.1 to 14.7 ms.  The B>=4 deviation (1.5e-05, relative ~1e-6) is
a batched-matmul accumulation-order change, and `pose.tokens` was not checked
for it (a batched sampling path would have to be measured the same way as
quantization).

**Verdict: rejected.**  The frozen gate is about *single-plan* p50 latency,
cold start and resident memory.  Batching changes none of the three (B=1 is
marginally slower: 7.14 vs 6.55 ms), it needs a new batch entry point that
nothing in the accepted product calls, and the B>=4 numerics are no longer
bit-exact so they would need their own token-identity campaign.  Recorded as a
measured throughput option for a batch-bake use case, not adopted.

## 6. Item 3 — quantization (q8, q4)

Implemented as an opt-in load-time transform in `Bundle` / `nn.linear`
(`MOTIONBRICKS_MLX_QUANT=q8|q4`), quantizing the 69 of 209 pose tensors (and the
root/decoder equivalents) that are consumed as matrices, with row-indexed tables
explicitly excluded.  Quantizing per tensor as it arrives keeps the float32 peak
to one matrix instead of a whole component.

### Memory and start-up (fresh process each, GPU, load 6.3)

| configuration | cold start | steady RSS | peak RSS | MLX peak | weight bytes |
| --- | --- | --- | --- | --- | --- |
| accepted (f32) | 321.4 ms | 791.4 MB | 829.8 MB | 779.2 MB | 732.6 MB |
| all f32 optimizations | 325.7 ms | 793.9 MB | 832.5 MB | 777.2 MB | 732.6 MB |
| **q8** | 356.2 ms | **230.6 MB** | 241.8 MB | 322.2 MB | 276.2 MB |
| **q4** | 360.0 ms | **222.0 MB** | 232.8 MB | 242.8 MB | 196.8 MB |

q8 = **3.43x** less resident memory, q4 = **3.56x** — both comfortably clear the
"resident memory improves >= 2x" half of the performance gate.

### Latency (GPU, 25 steps, interleaved)

| configuration | paired p50 ratio vs accepted | pooled p50 |
| --- | --- | --- |
| q8 (with all f32 optimizations) | **1.256x [1.132-1.488]** | 1.282x |
| q4 (with all f32 optimizations) | **1.360x [1.140-1.503]** | 1.317x |

### Correctness — the gate that decides it

Fixed-constraint comparison (16 steps) and end-to-end (5 seeds x 6 steps), GPU,
against the f32 accepted path.  `!` marks a step whose *token count* differs,
i.e. a different duration was selected:

| seed | q8 selected tokens | same steps | per-token diff | worst roots | x tol | worst rot | x tol |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 11,11,9,11,9,7 | 1/6 | 1.509 % | 2.220e-03 m | 2.22 | 7.729e-02 | 38.65 |
| 18 | 11,11,8,11,9!,11 | 1/6 | 2.708 % | 2.564e-02 m | 25.64 | 1.492e-01 | 74.59 |
| 35 | 11,11,9,11,9!,8 | 2/6 | 2.586 % | 1.308e-02 m | 13.08 | 9.045e-02 | 45.22 |
| 52 | 11,11,9,11,9,11 | 1/6 | 3.226 % | 5.538e-03 m | 5.54 | 1.768e-01 | 88.40 |
| 69 | 11,11,8,11,11!,9 | 2/6 | 3.602 % | 1.965e-02 m | 19.65 | 6.594e-02 | 32.97 |

| seed | q4 selected tokens | same steps | per-token diff | worst roots | x tol | worst rot | x tol |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 11,11,8!,11,11!,7 | 0/6 | 27.193 % | 9.127e-02 m | 91.27 | 3.139e-01 | 156.94 |
| 18 | 11,11,8,11,11!,10! | 0/6 | 28.390 % | 3.062e-02 m | 30.62 | 3.730e-01 | 186.51 |
| 35 | 11,11,8!,11,11!,8 | 0/6 | 28.728 % | 7.355e-02 m | 73.55 | 3.481e-01 | 174.05 |
| 52 | 11,11,8!,11,11!,11 | 0/6 | 30.533 % | 3.787e-02 m | 37.87 | 3.437e-01 | 171.85 |
| 69 | 11,11,8,11,11!,11! | 0/6 | 29.449 % | 4.042e-02 m | 40.42 | 4.328e-01 | 216.40 |

Aggregate, and the same measurement on the CPU backend:

| configuration | steps with identical `pose.tokens` | per-token difference rate | worst roots / rotations | verdict |
| --- | --- | --- | --- | --- |
| all f32 optimizations | **30/30** (GPU), 16/16 (CPU) | **0.000 %** | **0.000e+00** (GPU) | passes the gate |
| q8 | 7/30 (GPU), 5/16 (CPU) | 2.57 % / 1.97 % | 25.64x / 109.8x tolerance | **rejected** |
| q4 | 0/30 (GPU), 0/16 (CPU) | 28.43 % / 26.39 % | 91.27x / 216.4x tolerance | **rejected** |

`mlx_align.py plan` against the reference agrees: q8 fails 4 arrays ("worst
margin: rotation angle at 9.92x tolerance"), q4 fails 9 ("at 356.52x").

**Verdict: rejected**, even though it is the only candidate that clears a
performance gate.  The frozen rule allows a non-token-identical candidate only
at <= 0.5x tolerance; q8's roots are at 2.71x and its rotations at 38.65x
tolerance, q4's at 91x/157x.  This is exactly the risk the brief named: the pose
planner amplifies the root condition by ~1e3-1e4 and the Gumbel/argmax decision
converts a small logit shift into a different token, and in 3 of 6 q4 steps into
a different *duration*.  3.4-3.6x memory and 1.26-1.36x latency are real, so if
a batch-bake or offline path is ever wanted this is the configuration to
revisit — provided its own token budget is accepted explicitly.

## 7. Item 4 — operator fusion beyond `mx.compile`

Optional and gated on items 1-3 being met, which they were not; measured anyway
because the latency gate was not met.  MLX's fused attention kernel
(`mx.fast.scaled_dot_product_attention`) replaces the manual
matmul/scale/mask-add/softmax/matmul chain, but measured on the real 16-layer
pose stack it is **6.080 ms vs 6.160 ms** for the manual chain once both are
compiled (1.01x, inside the noise) and it is not bit-identical (9.537e-07 on the
layer output).  **Verdict: rejected** — no measurable gain, and it would cost
the exact-token property.  Padding-removal was likewise not pursued: the token
count is data-dependent, so there is no fixed superset shape to pad to, and
padding would itself change the reduction length.

## 8. Adoption table (the frozen gate, applied)

| item | change | before -> after (GPU p50) | gate result | decision |
| --- | --- | --- | --- | --- |
| 1a | duplicate duration probe removed (`root_dedup`) | 55.6 -> 54.9 ms | 1.03x < 1.5x | **rejected** |
| 1b | per-call shape constants hoisted | no measurable change | — | **rejected** |
| 1c | `mx.compile` stage bodies + per-shape memoisation | ~55 -> ~52 ms | ~1.05x < 1.5x | **rejected** |
| 1d | (finding) 6 compile variants per session, ~60 ms | — | — | reported |
| 2 | scenario batching | single-plan p50 unchanged | 2.37-6.05x throughput, not a graded quantity | **rejected** |
| 3 | q4 | 56.3 -> 42.7 ms; RSS 791 -> 222 MB | latency 1.36x, memory **3.56x** — but token gate fails | **rejected (correctness)** |
| 3 | q8 | 56.3 -> 46.6 ms; RSS 791 -> 231 MB | memory 3.43x — token gate fails | **rejected (correctness)** |
| 4 | fused attention | 6.160 -> 6.080 ms | 1.01x | **rejected** |
| — | `transition_fast` + `sampling_vector` (host NumPy vectorisation, not in the brief's list but found by profiling) | 55.0 -> 45.1 ms | 1.218x [1.158-1.320] < 1.5x | **rejected** |
| — | **aggregate of every bit-exact f32 optimization** | 56.3 -> 47.2 ms | 1.192x pooled / 1.116x paired | **rejected** |

**No item is adopted.**  The frozen gate requires >= 1.5x p50 or >= 2x cold
start / resident memory.  The one candidate that reaches a gate (quantization,
memory) fails the correctness gate, and everything that passes correctness tops
out at 1.19-1.24x.

Because "rejected" has to mean something, **every optimization defaults to OFF**
(`nn._option(name, False)`); the accepted behaviour is byte-for-byte what it was.
The switch is one environment variable:

```
MOTIONBRICKS_MLX_OPT=1            # every float32 optimization (~1.19x, bit-exact on GPU)
MOTIONBRICKS_MLX_OPT=pose_compile=1,sampling_vector=1
MOTIONBRICKS_MLX_QUANT=q4         # quantized load (~1.36x, 3.56x RSS, NOT token-identical)
```

Verified: with defaults untouched, the regenerated plan document for the
reference scenario is **identical to `out/plan-mlx.json`** (the accepted
artifact) with **0 differing values** across the whole document.

## 9. Alignment, determinism, tests

`mlx_align.py` (optimized path, GPU; the trace gate is the strongest one because
`pose.tokens` has tolerance 0):

```
worst margin: rotation angle (deg) at 0.26x tolerance
ALIGNMENT OK: every compared array is inside its declared tolerance

reference trace out/ref/trace/transition-0.json (seed 1, tokens 11) vs MLX gpu
  ok   selected tokens 11
  ok   root.num_token_logits        max|Δ| 7.629e-05  tol 5.0e-03
  ok   root.pred_global_root_values max|Δ| 9.537e-07  tol 1.0e-04
  ok   root.pred_local_root_values  max|Δ| 6.194e-05  tol 1.0e-03
  ok   pose.logits                  max|Δ| 2.512e-02  tol 5.0e-02
  ok   pose.sampling_uniforms       max|Δ| 0.000e+00  tol 0.0e+00
  ok   pose.tokens                  max|Δ| 0.000e+00  tol 0.0e+00
  ok   decoder.quantized            max|Δ| 0.000e+00  tol 0.0e+00
  ok   decoder.output               max|Δ| 8.149e-05  tol 1.0e-03
  ok   decoded.roots                max|Δ| 4.981e-07  tol 1.0e-03
TRACE OK: every stage inside tolerance
```

Determinism (`scripts/mlx_plan.py --repeat 2`, GPU, both states): "deterministic
over 2 runs with fixed seeds".  `scripts/mlx_bench.py` reports
`deterministic True` for both states and both backends.

`pytest`: **124 passed, 1 skipped**, including the new
`tests/test_mlx_optimizations.py` (41 tests) which pins

* the option grammar and that the default is the unoptimized path,
* the vectorised SplitMix64 stream against the scalar loop bit for bit over 5
  seeds x 5 lengths,
* `pose.tokens` / `pose.sampling_uniforms` / `decoder.quantized` identity for
  every single switch and for the aggregate,
* that a compiled shape variant is built once and reused,
* that the dropped zero mask is exactly the "no mask" case,
* that quantized tables are never quantized and that the default bundle load is
  unchanged.

## 10. What could not be improved, and why

1. **The single largest remaining cost is outside this work's edit grant.**
   `src/motionbricks_mlx/data/agent.py` spends **19.9 ms of every 55 ms step**
   (36 %) in host NumPy, measured directly with wall-clock instrumentation:
   `build_constraints` 14.5 ms (of which `_matrix_xyzw` 7.9 ms / 182 calls per
   step and `_cont6d_matrix` 4.9 ms / 136 calls), `restore_world` 2.6 ms,
   `encode_context` 1.9 ms, `apply_context_blend` 0.7 ms, `canonicalize` 0.3 ms.
   Those are ~35 us per call on 3x3 or 34-joint arrays: pure NumPy call
   overhead, not arithmetic.  They are trivially vectorisable (batch the
   joint/rotation axis instead of looping per joint), the same way
   `transition.py` was vectorised here for ~1.22x.  That file is not in the
   allowed list, so it was left alone.  **If latency matters, this is the next
   1.2-1.4x and it needs the other workstream's agreement.**
2. **The MLX stages cannot be made much faster on this shape.**  They are 92-98 %
   GPU execution of *dependent* kernels on a batch of <= 16 rows; `mx.compile`
   only touches the 2-8 % that is graph construction, and the fused SDPA kernel
   measures neutral.  The only real lever is larger batches (item 2: 2.4-6.1x
   throughput) which does not lower single-plan latency.
3. **Quantization cannot be both the memory win and token-exact.**  Affine q8
   already moves the logits enough to flip tokens at 2.57 % of positions and to
   change the selected duration; there is no group size that fixes this without
   giving up the memory win.
4. **Cold start did not improve.**  321 -> 326 ms for the f32 optimizations
   (the compile variants cost ~60 ms, offset by nothing) and 356/360 ms for
   quantization (dequantizing is not needed, but quantizing 209+ tensors at load
   is).  The 2x cold-start gate was never reachable.
5. **The measurement environment is not quiet.**  Load 6.3-17.4 with the noise
   sources listed in section 1.  Interleaved paired measurement is what makes
   the small effects (1.03x, 1.05x) separable from drift; a single-arm bench
   cannot resolve them, and I could not produce the clean idle-machine numbers
   the brief asked for on a shared desktop.

## 11. Reproduction

```
# accepted path (default) and the optimized path, same bundle/seed/backend
./.venv/bin/python scripts/mlx_bench.py --devices gpu,cpu --steps 25

MOTIONBRICKS_MLX_OPT=1 ./.venv/bin/python scripts/mlx_bench.py --devices gpu --steps 25
MOTIONBRICKS_MLX_OPT=1 ./.venv/bin/python scripts/mlx_plan.py --device gpu --repeat 2
MOTIONBRICKS_MLX_OPT=1 ./.venv/bin/python scripts/mlx_align.py plan out/ref/plan-ref.json out/plan-opt-on-gpu.json
MOTIONBRICKS_MLX_OPT=1 ./.venv/bin/python scripts/mlx_align.py trace out/ref/trace/transition-0.json --device gpu

MOTIONBRICKS_MLX_QUANT=q4 ./.venv/bin/python scripts/mlx_plan.py --device gpu --out out/plan-q4.json
./.venv/bin/python scripts/mlx_align.py plan out/ref/plan-ref.json out/plan-q4.json   # FAILS, by design

./.venv/bin/python -m pytest -q
```

Interleaved A/B and the token-identity campaign are scratch tools (they lived in
`/tmp/mbprof/`); the numbers they produced are reproduced above, and
`tests/test_mlx_optimizations.py` re-checks the identities that the decisions
depend on.

---

# Round 2 — host-side math (`data/agent.py`)

Frozen at `2026-10-12T21:17:30+08:00`, before any code change of this round, at
commit `6a09f91d4481d5d14218ffd66655430fa63d7f84`.  Everything in this section is
the *round-2 adoption criteria*; it was written first and is not edited
afterwards.  Section 0's criteria stay valid and are **not** relaxed; round 2
adds the narrower rule that decision
`docs/decisions/0004-adopt-bit-identical-f32.md` already fixed.

## 12. Frozen round-2 criteria

Subject: the host-side NumPy inside `src/motionbricks_mlx/data/agent.py`
(19.9 ms of every 55 ms step, 36 %, per section 10) — the file decision 0004
item 3 authorises this workstream to change — plus the switch defaults.

### 12.1 Correctness gate (unchanged in substance from 0.1)

1. `scripts/mlx_align.py plan` and `scripts/mlx_align.py trace` both still
   report `ALIGNMENT OK` / `TRACE OK`.
2. **Token identity.**  For N >= 5 distinct seeds x 6 steps, `pose.tokens` is
   element-for-element identical to the pre-round-2 f32 baseline (measured with
   the new default switch state) and `pose.sampling_uniforms` is
   element-for-element identical.
3. A host-math rewrite is *allowed* to change float summation order.  When it
   does, the maximum absolute difference on `roots` (m) and on `rotations` must
   be reported with its factor of the `mlx_align` tolerance.  **The tokens
   decide**: a token difference cannot be traded for speed.
4. Determinism: the same seed twice stays bit-identical.

### 12.2 Adoption rule (decision 0004, not a relaxation)

* A change that is **token-identical and does not regress alignment** is
  **adopted by default** as soon as the paired p50 improves by **>= 1.05x**.
  This is the decision 0004 rule for zero-risk bit-identical work; it is
  deliberately *not* section 0.2's 1.5x, which stays on the books for
  risk-carrying candidates.
* A change that introduces *any* token difference is **rejected / default OFF**,
  with its admissible scope named (`real-time` / `batch-bake` / `offline-only`).
* A change with a paired p50 < 1.05x is recorded as rejected with its number.

### 12.3 Measurement discipline (unchanged from 0.3)

Load average recorded with every measurement; every A/B pair interleaved inside
one process on the identical step sequence with the arm order rotated; >= 25
steps; p50/p90/max; same bundle, same seeds, same backend.

### 12.4 Order of work (fixed before measuring)

1. **Fewer calls**: batch the per-joint / per-frame loops into one array op,
   preallocate and reuse buffers.
2. **Move the maths into the MLX graph** (switchable), which can fuse and remove
   host<->device round trips.
3. **Only if host is still the bottleneck after 1 and 2**: a compiled kernel,
   `mx.fast.metal_kernel` first (one dispatch, no C-extension build/distribution
   cost).  No Cython / pybind11.

---

# Round 2 results — host-side maths

Everything below was written after the measurements.  Section 12 was not edited.

## 13. Item 0 — decision 0004 item 1: `MOTIONBRICKS_MLX_OPT` defaults on

`models/nn.py` now declares the nine float32 switches in one table
(`OPT_DEFAULTS`) and the call sites read it through `_opt(name)`, so a call site
cannot drift from the decision.  `MOTIONBRICKS_MLX_OPT=0` restores the
pre-optimization path, and because the grammar is applied in order the old
"accepted path plus one switch" form still works: `=0,pose_compile=1`.

Evidence on the GPU backend, all with no `MOTIONBRICKS_MLX_OPT` in the
environment:

| check | result |
| --- | --- |
| plan document vs the same document with `MOTIONBRICKS_MLX_OPT=0` | **byte-identical files** (`diff -q` clean) |
| `mlx_align plan` between those two documents | `0.000e+00` on every array |
| `mlx_align plan out/ref/plan-ref.json <default>` | `ALIGNMENT OK`, worst margin rotation angle **0.24x** tolerance |
| `mlx_align trace out/ref/trace/transition-0.json --device gpu` | `TRACE OK`; `pose.tokens` `0.000e+00`, `pose.sampling_uniforms` `0.000e+00` |
| the same trace on `--device cpu` | `TRACE OK` |
| `mlx_plan.py --device gpu --repeat 2` / `--device cpu --repeat 2` | "deterministic over 2 runs with fixed seeds" |
| `pytest` | **125 passed, 1 skipped** |

The stored `out/plan-mlx.json` is a **CPU** document; a GPU document differs from
it by 2.354e-06 m of roots.  That difference is the known backend difference and
is **not** produced by this flip: a GPU document generated with
`MOTIONBRICKS_MLX_OPT=0` differs from `out/plan-mlx.json` by exactly the same
numbers.  The flip itself is a no-op on the numerics of either backend.

## 14. Item 1 — batched host maths in `data/agent.py`: **adopted**

Each `_ref` function keeps the reference's scalar loop verbatim; a batched twin
replaces the per-joint / per-frame loop with one array expression over the same
axis, and a dispatcher reads `MOTIONBRICKS_MLX_AGENT` (default **on**).

### Host decomposition before/after (25 steps, same bundle/seed, GPU)

| function | before: calls/step | before: ms/step | after: calls/step | after: ms/step |
| --- | --- | --- | --- | --- |
| `build_constraints` | 1 | **14.095** | 1 | **1.007** |
| `restore_world` | 1 | **2.523** | 1 | **0.397** |
| `apply_context_blend` | 1 | **0.681** | 1 | **0.176** |
| `canonicalize` | 1 | **0.280** | 1 | **0.158** |
| **host total (top level)** | | **17.58** | | **1.74** |
| `_matrix_xyzw` | 180.6 | 7.560 | 4.0 | 0.345 |
| `_cont6d_matrix` | 136.0 | 4.673 | 1.0 | 0.108 |
| `quaternion_matrix` | 182.6 | 1.815 | 6.0 | 0.166 |
| `matrix_cont6d` | 276.0 | 0.418 | 3.0 | 0.010 |
| `slerp` | 116.0 | 0.584 | — (batched `_slerp_batch`) | ~0 |
| `spring` | 24.0 | 0.112 | 24.0 | 0.131 |
| `y_rotation` | 11.0 | 0.040 | 3.0 | 0.019 |
| `wrap` | 15.0 | 0.040 | 6.0 | 0.022 |

`_matrix_xyzw` is now called 4 times per step instead of 180.6, and
`_cont6d_matrix` once instead of 136 -- i.e. the 34-46 us *per call* on 3x3
arrays (section 3) has been paid 4 and 1 times instead of 181 and 136.

### Paired A/B (interleaved in one process, both arms on the default MLX path)

| run | load average | arm OFF p50 | arm ON p50 | paired p50 ratio | tokens |
| --- | --- | --- | --- | --- | --- |
| 25 steps | 11.6 -> 17.3 | 45.11 ms | 28.08 ms | **1.576x [1.410-1.685]**, mean 1.573x | **25/25 steps identical, 0/1832 tokens differ** |
| 40 steps | 16.7 -> 17.3 | 44.53 ms | 30.85 ms | **1.556x [1.442-1.814]**, mean 1.587x | **40/40 steps identical, 0/2936 tokens differ** |

`p90` 53.88 -> 37.79 ms and 54.60 -> 36.76 ms; `max` 57.53 -> 40.54 and
61.64 -> 41.91 ms.

### Decision

Token-identical, alignment unchanged, paired p50 **1.56-1.58x** >= 1.05x ->
**adopted by default** under the section 12.2 rule, which is decision 0004's rule
and not a relaxation.  Reproduce with
`scripts/mlx_bench.py --paired "MOTIONBRICKS_MLX_AGENT=0" "MOTIONBRICKS_MLX_AGENT=1" --paired-tokens`.

## 15. Item 2 — moving the maths into the MLX graph: **rejected, measured**

Section 12.4 fixed the order, so item 2 was evaluated after item 1 landed.  Three
measurements decide it, and none of them is an argument from first principles:

1. **The prize shrank to 6 %.**  After item 1 the whole host maths is 1.74 ms of
   a ~29 ms step.  Item 2's ceiling is therefore ~1.06x, and only if GPU maths
   were free.
2. **The boundary item 2 would remove is free already.**  The six constraint
   uploads (`mx.array` of 8x5, 8x4, 8x303 and three 8x1 masks, plus `mx.eval`)
   cost **0.0059 ms p50**; an empty `mx.eval` sync costs 0.0003 ms.  The 4-6
   device synchronisations per step are the host-side *decisions*
   (duration selection, Gumbel sampling, codebook gather) and item 2 does not
   remove them.
3. **The MLX translation is more expensive than the host code it replaces.**
   Measured on the real shapes, splitting graph build from `mx.eval` the way
   section 3 did:

   | block | MLX build | MLX eval | MLX total | host (item 1) |
   | --- | --- | --- | --- | --- |
   | FK tree, 34 dependent steps x 4 frames | 0.320 ms | 1.317 ms | **1.64 ms** | **0.371 ms** |
   | target block (`corr @ rot` + cont6d) | 0.061 ms | 0.546 ms | **0.61 ms** | ~0.4 ms of `build_constraints` |

   This is section 3's finding again: 34 *dependent* tiny kernels on batch 4 cost
   1.3 ms of GPU execution, which is more than the CPU spends on the same 34
   numpy calls.  Even depth-batching the tree (parents before children, ~8 levels
   instead of 34) would land near the host time, not below it.

There is also a correctness reason it could not have been a drop-in, measured
directly:

| operation | MLX vs NumPy | differing inputs | worst |
| --- | --- | --- | --- |
| `cos` | **not exact** | 1246/4096 (30 %) | 1.000 ULP |
| `sin` | **not exact** | 1273/4096 (31 %) | 1.000 ULP |
| `arctan2` | **not exact** | 1402/4096 (34 %) | 2.000 ULP |
| `arccos` | **not exact** | 2164/4096 (53 %) | 2.000 ULP |
| `sqrt` | exact | 0 | 0 |
| `remainder` | exact | 0 | 0 |

So the MLX form would have to re-earn token identity from scratch *and* be
slower.  **Verdict: rejected.**  The scope if it is ever revisited is
`batch-bake`/`offline`, where the throughput of a batched GPU graph (section 5)
matters more than the 1.7 ms of host time.

## 16. Item 3 — a compiled kernel: **not reached**

Section 12.4 allows item 3 only if the host is still the bottleneck after items 1
and 2.  After item 1 the host is 1.74 ms of a ~29 ms step (6 %), and item 2
measured a net loss rather than a win, so no kernel was written.  If the host
ever becomes the bottleneck again, `mx.fast.metal_kernel` remains the first
choice (one dispatch, no C-extension build or distribution); the measurements in
section 15 are the reason it is not needed for this shape.

## 17. Correctness of round 2

### 17.1 Bit-identity of the rewrite

Not a tolerance -- equality, because the rewrite was built to be exact and the
dispatcher only exists to A/B it:

* every rewritten entry point (`seed_agent_from_style`, `canonicalize`,
  `encode_context`, `build_constraints` on all 7 returned arrays, `restore_world`
  on all 4, `apply_context_blend` on both) is bit-identical to its `_ref` twin on
  **4 real styles x 9 seeds x 3 movement/facing shapes = 756 compared arrays, 0
  differing**, plus synthetic styles and synthetic command shapes.
* the same holds on the real agent context produced by a real plan.

### 17.2 End-to-end campaign, 5 seeds x 6 steps (the section 12.1 gate)

| backend | `pose.tokens` identical | tokens differing | `sampling_uniforms` identical | worst \|d roots\| | worst \|d rotations\| |
| --- | --- | --- | --- | --- | --- |
| GPU | **30/30 steps** | **0/2416** | 30/30 steps | **0.000e+00 m** (0.000x tol) | **0.000e+00** (0.000x tol) |
| CPU | **30/30 steps** | **0/2416** | 30/30 steps | **0.000e+00 m** (0.000x tol) | **0.000e+00** (0.000x tol) |

The float last-bit difference the criteria require to be reported is therefore
**zero**: this rewrite changed no rounding at all on the data the port actually
sees.  That is a stronger result than the rule needs, and it is why the change
could be adopted at the 1.05x threshold rather than the 1.5x one.

### 17.3 Two NumPy traps the rewrite had to avoid

Both were found by measurement, not by reading; both are pinned by
`tests/test_mlx_agent_math.py` so a later simplification fails the suite instead
of silently reintroducing a last-bit perturbation that the pose planner amplifies
by ~1e3-1e4.

1. **gemv is not gemm.**  The reference rotates each style joint position with a
   per-joint `matrix @ vector` (gemv).  Writing the batch as
   `corrections @ positions.transpose(0, 2, 1)` is a gemm, and NumPy sends the two
   to different BLAS kernels: 2.384e-07 on random 3x3 data, and **2.980e-08 on the
   real `walk_stealth` constraints**, which was enough to change the decoded
   motion of the second plan.  `np.einsum("fij,fkj->fki", ...)` reproduces the
   gemv bit for bit.
2. **`np.sum(a * b)` is not `np.dot(a, b)`.**  BLAS `sdot` accumulates
   differently from an elementwise product followed by a sum: 4.768e-07 on random
   4-vectors.  A batched matrix-vector product reproduces `sdot`; that is
   `dot_rows`.

A third defect was caught only by the end-to-end campaign, not by the unit-level
differential: the batched target block had dropped the reference's `wrap()` on
`target_heading - style.global_headings[source_frame]`.  `wrap` is a no-op on the
*value* for angles already in `[-pi, pi]` but not in float32, and the omission was
invisible until a style whose headings sit near the wrap boundary
(`walk_stealth`).  It is the reason the gate is end-to-end tokens and not
array-by-array equality of one step.

## 18. Round-2 adoption table

| item | change | before -> after (GPU p50) | gate | decision |
| --- | --- | --- | --- | --- |
| 0 | `MOTIONBRICKS_MLX_OPT` default flipped to on (decision 0004 item 1) | 47.0 -> 28-31 ms end-to-end including item 1; on its own the flip is a byte-identical no-op | correctness gate at its highest grade | **adopted** (per decision 0004) |
| 1 | batched host maths in `data/agent.py` | host 17.58 -> **1.74 ms/step**; paired p50 **1.556-1.576x** | token-identical, alignment OK, >= 1.05x | **adopted, default on** |
| 2 | move the host maths into the MLX graph | MLX translation measured **+1.0 ms/step** (1.64 vs 0.371 ms for the FK tree alone) | < 1.05x, and not bit-identical | **rejected** |
| 3 | compiled kernel (`mx.fast.metal_kernel`) | not reached: host is 6 % of the step | section 12.4 precondition not met | **not attempted** |

## 19. Where the remaining time is, and what is next

The step is now ~94 % MLX stages.  Round 1's diagnosis is unchanged and is now
the whole picture: the stages are 92-98 % execution of *dependent* small kernels
on a batch of <= 16 rows, and 4-6 host device synchronisations per step remain
because the algorithm makes host decisions between them (duration selection,
Gumbel sampling, codebook gather).  Section 3 measured 13.9 ms of `mx.eval` in
those kernels; the host no longer competes with that.

Ranked next steps, all inside the existing grants:

1. **Batch bake, not single-plan latency.**  Round 1 section 5 measured 2.37x /
   3.12x / 6.05x throughput per scenario at B = 2 / 4 / 8 with the pose planner
   only 7.1 -> 14.7 ms for the whole batch.  With the host now 1.74 ms, a batched
   path would amortise it too.  It needs its own token campaign (B >= 4 was not
   bitwise in round 1).
2. **Reduce the synchronisation count.**  When `allowed_tokens` selects exactly
   one duration the duration probe is already skipped; the same could apply to
   the second sync when the root stage's output is only consumed after the
   head offset is known.  Each removed sync is worth the kernel-drain latency,
   not just its own cost.
3. **Re-measure on an idle machine.**  Every number here is from a shared desktop
   at load 9.8-17.4; the paired form is what makes the effects separable, but the
   absolute p50 is still inflated by roughly 10 %.

## 20. Reproduction (round 2)

```
# the default path (both round-2 changes on) and the pre-round-2 path
./.venv/bin/python scripts/mlx_plan.py --device gpu --out /tmp/plan-default.json
MOTIONBRICKS_MLX_OPT=0 MOTIONBRICKS_MLX_AGENT=0 \
  ./.venv/bin/python scripts/mlx_plan.py --device cpu --out /tmp/plan-pre-round2.json
# -> byte-identical to out/plan-mlx.json, sha256 262e77d438daa8a790eccf23f8cfd696bf620a304bb4d42b54e38282e085b64a

# interleaved paired A/B for the host maths
./.venv/bin/python scripts/mlx_bench.py --paired \
  "MOTIONBRICKS_MLX_AGENT=0" "MOTIONBRICKS_MLX_AGENT=1" --devices gpu \
  --steps 25 --paired-tokens

# the switch, its bit-identity over real styles, and the 5x6 token campaign
./.venv/bin/python -m pytest tests/test_mlx_agent_math.py -q

./.venv/bin/python scripts/mlx_align.py plan out/ref/plan-ref.json /tmp/plan-default.json
./.venv/bin/python scripts/mlx_align.py trace out/ref/trace/transition-0.json --device gpu
```

Note for whoever reads the artifacts next: `out/` is gitignored.  While
regenerating documents during round 2 this workstream ran
`mlx_plan.py --device gpu --repeat 2` **without `--out`**, which overwrote
`out/plan-mlx.json` with a GPU document.  It was restored byte-for-byte
(sha256 `262e77d4…`, the value of the four identical accepted-path documents
`out/plan-default-cpu.json`, `out/plan-opt-base-cpu.json`,
`out/plan-final-default.json`, `out/plan-mlx-rep.json`) and the restoration was
verified by replaying the pre-round-2 path against it.  Always pass `--out`.
