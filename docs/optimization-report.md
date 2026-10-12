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

### 13.1 The one measured cost of the flip: CPU-backend peak memory

The flip changes nothing numerically, but it does change what the **CPU**
(LLVM) backend pays, because round-1 section 4's CPU-only `mx.compile`
side-effect is now on by default.  Measured on the default path, `--devices cpu`,
25 steps, load ~13:

| configuration | p50 | MLX peak | process RSS |
| --- | --- | --- | --- |
| `MOTIONBRICKS_MLX_OPT=0` (pre-decision-0004) | 199.0 ms | 748 MB | 853 MB |
| default (everything on) | 178.8 ms | **1481 MB** | **1615 MB** |
| default except the three `*_compile` switches | 173.8 ms | **748 MB** | 854 MB |

So on the CPU backend the compile switches roughly **double peak memory and buy
nothing** -- the configuration without them measured *faster* (173.8 vs 178.8 ms),
though these are single-arm runs under drifting load and the honest statement is
only that the latency difference is inside that drift, while the 2x memory is
not.  Round 1 measured the same side-effect on the same backend (748 -> 1481 MB,
188.7 vs 184.5 ms).

This is **not** changed here: decision 0004 adopts the float32 group as a whole
and this workstream does not quietly carve a switch out of a ratified default.
It is recorded as the top candidate for the next refinement: give the three
`*_compile` switches a backend-dependent default (off on the non-Metal backend),
which recovers 748 MB there at no measured cost.  The exact environment that
reproduces the recommendation is

```
MOTIONBRICKS_MLX_OPT=0,pose_fast=1,root_fast=1,root_dedup=1,decoder_fast=1,transition_fast=1,sampling_vector=1
```

### 13.2 Headline: the default path against the original accepted baseline

| | accepted baseline (section 0) | default path now |
| --- | --- | --- |
| MLX GPU per-step p50 | 55.5 ms | **27.3 ms** |
| MLX GPU p90 | 67.3 ms | 38.7 ms |
| MLX CPU per-step p50 | 196.3 ms | 178.8 ms |
| cold start (GPU) | 385.3 ms | 336 ms |
| MLX peak (GPU) | 748-779 MB | 779 MB |
| `pose.tokens` vs reference | 0 differences | 0 differences |

i.e. **2.03x** on the single-plan Metal latency the product actually uses, with
the token identity and the alignment margin unchanged (worst margin 0.26x
tolerance, same as the recorded accepted state).

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
61.84 -> 41.91 ms.

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

1. **Give the `*_compile` switches a backend-dependent default.**  Section 13.1:
   on the CPU backend they cost 748 -> 1481 MB of MLX peak for no measured
   latency gain, while on Metal they are free (GPU peak unchanged at 779 MB).
   Defaulting them off on the non-Metal backend recovers the memory with no
   code change beyond the default table.  This is the only *regression* round 2
   leaves behind, so it ranks first.
2. **Batch bake, not single-plan latency.**  Round 1 section 5 measured 2.37x /
   3.12x / 6.05x throughput per scenario at B = 2 / 4 / 8 with the pose planner
   only 7.1 -> 14.7 ms for the whole batch.  With the host now 1.74 ms, a batched
   path would amortise it too.  It needs its own token campaign (B >= 4 was not
   bitwise in round 1).
3. **Reduce the synchronisation count.**  When `allowed_tokens` selects exactly
   one duration the duration probe is already skipped; the same could apply to
   the second sync when the root stage's output is only consumed after the
   head offset is known.  Each removed sync is worth the kernel-drain latency,
   not just its own cost.
4. **Re-measure on an idle machine.**  Every number here is from a shared desktop
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

---

# Round 3 — backend-conditional defaults, batch bake, fewer host syncs

Frozen at `2026-10-12T21:37:24+08:00`, **before any code change of this round**,
at commit `1a076e02c3020bdeb792305118acd046619707bf`.  Everything in this section
is the round-3 *adoption criteria*; it is written first and is not edited
afterwards.  Section 0 and section 12 stay valid and are **not** relaxed.

## 21. Frozen round-3 criteria

Three subjects, all inside the standing edit grant:

1. **Decision `docs/decisions/0005-backend-conditional-compile.md`** — give
   `pose_compile` / `root_compile` / `decoder_compile` a **backend-dependent
   default**: on for Metal, off for every non-Metal backend.  This is an
   *implementation of a ratified decision*, not a new candidate: its acceptance
   list is the one written in the decision, quoted here verbatim so the report
   is self-contained:
   1. the GPU default-path document is **byte-identical** to the same document
      produced before the change (`diff -q` clean); `mlx_align.py plan` and
      `mlx_align.py trace` still OK with `pose.tokens` and
      `pose.sampling_uniforms` at `0`;
   2. on the **CPU** default, peak MLX memory returns to ~748 MB / RSS ~854 MB,
      and the p50 is **not worse** than `MOTIONBRICKS_MLX_OPT=0` (inside drift is
      enough);
   3. 5 seeds x 6 steps stay token-identical; `mlx_plan.py --repeat 2` is
      deterministic on **both** backends; `pytest` stays green (135 passed /
      1 skipped at the start of this round — the count may only grow).

   The rule that comes with the decision is also binding: **`mlx_plan.py` is
   always run with an explicit `--out`.**  The conventional path
   `out/plan-mlx.json` is never the target of a measurement.

2. **Batch bake (`B` independent scenarios through one batched forward).**
   Round 1 measured the batched *pose planner* at 2.37x / 3.12x / 6.05x
   throughput for `B` = 2 / 4 / 8 and found `B >= 4` **not bitwise**
   (section 5).  This round asks whether that option is usable for *offline
   bake* with evidence:

   * for `B` = 1 / 2 / 4 / 8 / 16 report throughput (scenarios/s), whole-batch
     latency, and the single-scenario p50;
   * the **token campaign**: for every one of the `B` scenarios, compare
     `pose.tokens`, `pose.sampling_uniforms` and the final action against the
     **same scenario run at `B = 1`**, and report the per-group bit-identity rate
     and the worst deviation (max |Δ| on `roots` in m, on `rotations`, and the
     worst margin as a factor of the `mlx_align` tolerance);
   * the conclusion is the **safe-B rule**: *the largest `B` whose scenarios are
     all token-identical to `B = 1` is the **reproducible bake tier**; any larger
     `B` is a **fast-draft tier only**, and the batch size used must be recorded
     in the product manifest.*  The rule is written into this report and into a
     `docs/decisions/` **proposal** (the decision itself is the lead's to make).

   Batch bake is *outside* the single-plan latency gate of section 0.2 by
   construction: it does not lower single-plan p50, it raises throughput per
   scenario.  It is therefore judged on its own two numbers — throughput gain
   and token identity — and never reported as a latency win.

3. **Fewer host synchronisations per step.**  Section 19 item 3 records 4-6
   `np.asarray(mx...)` drains per step (duration selection, the root planner's
   output, the pose logits, the decoder output, the decoded motion).  Each one
   ends the current graph and waits for the device.  Each candidate is measured
   and judged on its own:

   * **correctness gate** (section 12.1, unchanged): `mlx_align.py plan` and
     `trace` still OK; for `N >= 5` seeds x 6 steps `pose.tokens` and
     `pose.sampling_uniforms` are element-for-element identical to the round-2
     default; the final action deviation is reported even when it is zero.
   * **adoption rule** (section 12.2, unchanged): *token-identical **and**
     alignment not regressed **and** paired p50 `>= 1.05x`* => default **on**.
     **Any** token difference => default **off**, with the admissible scope
     named (`real-time` / `batch-bake` / `offline-only`) and the measured
     deviation written down.
   * **measurement discipline** (section 0.3 / 12.3, unchanged): the `uptime`
     load average is recorded with every measurement; every A/B pair is
     **interleaved inside one process** on the identical step sequence with the
     arm order rotated; **>= 25 steps per arm**; p50 / p90 / max; same bundle,
     same seeds, same backend.

### 21.1 Reporting format fixed in advance

Every item reports, in this order: the paired ratio with its inter-quartile
range, the pooled p50 / p90 / max per arm, the token-identity count, and the
verdict against the rule above.  A rejected item keeps its number and its
reason.  Nothing in this section is edited after the measurements.

---

# Round 3 results

Everything below was written after the measurements.  Section 21 was not edited.

## 22. Item 1 — decision 0005: backend-conditional `*_compile` defaults

`models/nn.py` now derives each switch's *effective* default in
`declared_default()`: `OPT_DEFAULTS` remains the single description of the
adopted path (the float32 group on), and the three compile switches —
`pose_compile`, `root_compile`, `decoder_compile` — are the only members of
`_BACKEND_CONDITIONAL`, so they default on for Metal and off for every non-Metal
backend.  `metal_backend()` reads `mx.default_device()` per call (the backend is
chosen by `Bundle(device=...)` after import), and falls back to the declared
value if the probe ever fails.  An explicit `MOTIONBRICKS_MLX_OPT` value still
wins on either backend, so every earlier A/B stays reproducible.

### 22.1 Acceptance 1 — the GPU default path is unchanged

| check | result |
| --- | --- |
| GPU plan document before vs after the change | `diff -q` clean, sha256 `ccda1952c967c8d7e893a795908f7de278f2acda1ce5ba1e1889692a8904a7c5` |
| `mlx_align.py plan out/ref/plan-ref.json <gpu default>` | `ALIGNMENT OK`, worst margin rotation angle **0.26x** tolerance |
| `mlx_align.py trace out/ref/trace/transition-0.json --device gpu` | `TRACE OK`; `pose.tokens` `0.000e+00`, `pose.sampling_uniforms` `0.000e+00`, `decoder.quantized` `0.000e+00` |
| `mlx_align.py trace ... --device cpu` | `TRACE OK`; `pose.tokens` `0.000e+00`, `pose.sampling_uniforms` `0.000e+00` |
| `mlx_plan.py --device gpu --repeat 2 --out /tmp/...` | `deterministic over 2 runs with fixed seeds` |
| `mlx_plan.py --device cpu --repeat 2 --out /tmp/...` | `deterministic over 2 runs with fixed seeds` |

Both documents were written to `/tmp`, per decision 0005's rule that
`mlx_plan.py` is always given an explicit `--out`.

### 22.2 Acceptance 2 — the CPU default, and a correction to the premise

The decision's premise is that the compile switches cost the CPU backend a
**doubled** peak (`748 -> 1481 MB`).  That number does **not** reproduce in a
CPU-only process, and the ablations below locate what actually produced it.
Every row is a fresh process, `--steps 25`, recorded load in the table.

| # | configuration | protocol | CPU p50 | CPU MLX peak | CPU RSS |
| --- | --- | --- | --- | --- | --- |
| A | default, **before** this change | `--devices cpu` | 165.4 ms | **748 MB** | 865 MB |
| B | `MOTIONBRICKS_MLX_OPT=0` | `--devices cpu` | 178.2 ms | 748 MB | 853 MB |
| A' | default, **after** this change | `--devices cpu` | (paired, 22.3) | **748 MB** | 853 MB |
| C | default, before | `--devices gpu,cpu` | 189.3 ms | **1481 MB** | 1615 MB |
| D | `OPT=0` | `--devices gpu,cpu` | 175.4 ms | 748 MB | 1018 MB |
| E | default, after | `--devices gpu,cpu` | 178.9 ms | **1480 MB** | 1603 MB |
| F | default, after | `--devices cpu,cpu` | — | 748 / 748 MB | 853 / 854 MB |
| G | `OPT=0,pose_compile=1,root_compile=1,decoder_compile=1` | `--devices gpu,cpu` | — | 748 MB | 862 MB |
| H | `MOTIONBRICKS_MLX_OPT=1,pose_compile=0,root_compile=0,decoder_compile=0` | `--devices gpu,cpu` | — | 748 MB | 862 MB |
| I | default, before | `--devices cpu,gpu` | CPU 216.8 | CPU 748, **GPU 1512 MB** | 1593 MB |

What the table shows:

* **A CPU-only process never pays the 2x.**  Before the change the CPU default
  was 748 MB (A, three times), exactly like `OPT=0` (B) and exactly like the
  non-compiled CPU configuration the decision quotes (748 MB).  The change
  therefore recovers **no memory on this machine**; acceptance item 2's
  "back to ~748 MB" holds *before* the change as well.
* The 1481 MB is produced by the **two-backend protocol**: the second backend's
  `get_peak_memory()` and RSS include what the first one left behind.  It is not
  specific to the CPU side — run the devices the other way round and the **GPU**
  row is the inflated one (I: 1512 MB for a backend whose own peak is 779 MB).
* The trigger is `mx.compile` variants having actually been **built on the first
  backend**.  G (compiles enabled but the fast paths off, so no compile ever
  runs) is 748; H (every fast path on, compiles off) is 748; the full default,
  where Metal compiles, is 1480-1481.  Because the Metal backend keeps its
  compiles by design, a `gpu,cpu` process still shows ~1480 in the CPU row after
  this change (E) — the artifact is in the harness, not in the CPU default.

The honest reading: **the original claim "compiling costs the CPU backend 2x
memory" is not supported.**  The rule the decision ratifies is implemented as
written (and is still justified: the CPU compiles bought no latency and cost
~60 ms of cold start, section 4), but this report does not repeat the 2x
finding, and the harness's `--devices a,b` row for the *second* device should
not be read as that device's cost.

### 22.3 Acceptance 2 (latency) — CPU default vs `OPT=0` and vs the flip

Interleaved three-arm run, one process, CPU, 25 steps, load `9.17 -> 9.33`:

| arm | p50 | p90 | max | paired p50 vs clean env | IQR |
| --- | --- | --- | --- | --- | --- |
| clean env (**new** CPU default: compiles off) | 193.6 ms | 221.7 | 551.0 | 1.0000x | — |
| `MOTIONBRICKS_MLX_OPT=1` (the flip state, compiles on) | 186.1 ms | 241.5 | 294.5 | **1.0124x** | 0.969-1.060 |
| `MOTIONBRICKS_MLX_OPT=0` (pre-0004) | 190.8 ms | 215.6 | 229.3 | **0.9900x** | 0.949-1.059 |

Cold start, single fresh process per arm, load 10.7-11.0: default with the
compiles **on** 635 ms, with them **off** (the new default) 497 ms, `OPT=0`
451 ms.  These are one sample each and cold start is the noisiest number in this
report, so the honest statement is only that the CPU compile variants do not
*help* cold start (round 1 measured ~60 ms of variant compilation; see 26.5).

So on the CPU the new default is **1.24 % faster than the flip state** and
**1.00 % slower than `OPT=0`** — both inside their inter-quartile ranges, i.e.
"not worse, within drift", which is what the decision asks for.  Token identity
in the same process: **25/25 steps identical, 0/1832 tokens differ** against the
new default for both arms.

### 22.4 Acceptance 3 — tokens, determinism, tests

* `pytest`: **135 passed, 1 skipped** with the change in place (the round-3
  additions bring it to 140+; see section 25).
* 5 seeds x 6 steps on both backends: `tests/test_mlx_agent_math.py::test_token_identity_end_to_end`
  passes, i.e. `pose.tokens` and `pose.sampling_uniforms` are element-for-element
  identical between the pre-round-2 reference and the round-3 default.
* `--repeat 2` on both backends: deterministic (22.1).
* The CPU default document moves by `2.116e-06 m` on roots against the
  pre-change CPU document (the known CPU `mx.compile` fusion difference,
  section 4); alignment stays `0.27x` tolerance, which is the tool's own
  identical-file floor for the derived rotation-angle column (see 22.5).

### 22.5 A measurement artifact worth recording

`mlx_align.py`'s derived `rotation angle (deg)` column compares
`angle_between(a, b)` against zero, and `export/plan_json.angle_between`
amplifies float32 quaternion dot-product rounding through `arccos` near 1.  Two
**byte-identical** documents therefore report `5.070e-02 deg` (0.25x tolerance),
not zero: the column has a floor of about `5.4e-02 deg`.  Every "worst margin:
rotation angle at ~0.26x tolerance" line in this report is that floor, not a
measured drift.

## 23. Item 2 — batch bake: throughput, and the token campaign

### 23.1 What was built

`models/bake.py` batches **the pose stage only** — the one stage round 1 found to
have independent work (section 5) — through
`PosePlanner.plan_batch()`, which wraps the single-scenario `_core` in
`mx.vmap` + `mx.compile`.  Every other stage runs the accepted code per
scenario.  The two arms are:

* **reference**: each scenario planned **alone** through the ordinary pipeline,
  with each scenario's own pose inputs recorded on the way past;
* **batched**: those recorded inputs go through one batched forward per
  **token-count group**, and each row is handed back to the pipeline through a
  new one-parameter seam (`run_transition(..., pose_logits=...)`, threaded
  through `Agent.plan`).  Sampling, the codebook gather, the decoder and the
  motion decode are byte-for-byte the accepted code on both arms; the only
  difference is which forward produced the logits.

Scenarios that selected different durations cannot share a forward (the
compiled variants and the GGML layout reshapes are shape-keyed), so the batched
arm groups by token count.  `pose_logits=None` — the default — is the accepted
path byte for byte, verified by re-running the GPU default document (`diff -q`
clean).

### 23.2 Why the batched forward is not bitwise — and why that did not matter here

`mx.vmap` lowers every per-row `matmul` to a batched primitive with a different
accumulation order, so a batched row is **never** bitwise equal to the serial
forward, not even at `B = 1`.  Measured on the pose logits, same inputs:

| B | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| max abs deviation from the serial forward | 1.98e-04 | 1.98e-04 | 3.05e-04 | 3.05e-04 | 3.36e-04 |

(Round 1's hand-rolled batch reported `B = 1/2` bitwise and `B >= 4` at
`1.5e-05`; `mx.vmap` starts from a different primitive, which is why even
`B = 1` differs here.)  This is a numerics change and is treated as one: it is
why the switch's own correctness rests on the campaign below, not on equality.

### 23.3 End-to-end bake: throughput and latency

`--bake 1,2,4,8,16 --bake-repeats 5 --devices gpu`, load `13.7 -> 13.5`.
"Serial" is `B` scenarios planned one after another; "batched" is the batched
pose plus the per-scenario accepted tail, both inside one timed region.  Both
arms warm up once before timing, and the arm order alternates per repeat.

| B | serial p50 | batched p50 | speedup | per-scenario (batched) | throughput | single-scenario p50 | token groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 27.79 ms | 29.25 ms | **0.950x** | 29.25 ms | 34.19/s | 27.79 ms | {11:1} |
| 2 | 49.06 | 42.21 | **1.162x** | 21.10 ms | 47.38/s | 27.79 ms | {11:2} |
| 4 | 103.82 | 108.04 | **0.961x** | 27.01 ms | 37.02/s | 27.79 ms | {11:2, 7:1, 8:1} |
| 8 | 215.89 | 182.93 | **1.180x** | 22.87 ms | 43.73/s | 27.79 ms | {11:3, 7:2, 8:1, 6:2} |
| 16 | 426.56 | 345.99 | **1.233x** | 21.62 ms | 46.24/s | 27.79 ms | {11:5, 7:7, 8:1, 6:3} |

Styles cycling across four assets fragments the batch into 3-4 token-count
groups, and `B = 4` is *slower* than serial because two of its three groups hold
a single row.  With a **duration-homogeneous** batch (`--bake-homogeneous`, one
style, different seeds — one group of `B`) the same measurement gives:

| B | serial p50 | batched p50 | speedup | per-scenario (batched) | throughput | token groups |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 101.92 ms | 86.00 ms | **1.185x** | 21.50 ms | 46.51/s | {11:4} |
| 8 | 219.18 | 159.15 | **1.377x** | 19.89 ms | 50.27/s | {11:8} |
| 16 | 430.21 | 321.93 | **1.336x** | 20.12 ms | 49.70/s | {11:16} |

So **grouping by duration is worth more than raising B**: at `B = 16` the
homogeneous batch is 1.336x against 1.233x fragmented, and it uses one compiled
variant instead of four.

### 23.4 Stage-level view: where the ceiling comes from

Homogeneous batch, 16 real scenarios, pose stage timed alone (load 14.9):

| B | batched pose p50 | per scenario | vs `B` separate forwards |
| --- | --- | --- | --- |
| 1 | 7.789 ms | 7.789 ms | 0.83x |
| 2 | 10.225 | 5.112 | 1.26x |
| 4 | 10.338 | 2.585 | **2.50x** |
| 8 | 10.260 | 1.283 | **5.04x** |
| 16 | 14.398 | 0.900 | **7.18x** |

The stage itself scales as round 1 found (2.50x / 5.04x at `B` = 4 / 8 against
round 1's 3.12x / 6.05x; `B = 16`, not measured then, reaches 7.18x).  But a
single serial forward is 6.46 ms of a 27.79 ms step, so the pose stage is ~23 %
of the work and **batching it alone cannot exceed ~1.25x end-to-end** on this
composition; the measured 1.18-1.38x is that ceiling, reached.  The round-1
2.4-6.1x figure is the *stage's* throughput, not the bake's, and this section is
the number that should be quoted for a bake.

### 23.5 The token campaign

Per-scenario comparison against **the same scenario run alone** (`B = 1`
semantics), `pose.tokens`, `pose.sampling_uniforms`, the selected duration and
the final action:

| B | scenarios | tokens identical | uniforms identical | duration identical | differing tokens | worst `roots` | worst `rotations` | worst angle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | **1/1** | 1/1 | 1/1 | 0/88 | 0.000e+00 m | 0.000e+00 | 0.0548 deg |
| 2 | 2 | **2/2** | 2/2 | 2/2 | 0/176 | 0.000e+00 | 0.000e+00 | 0.0601 |
| 4 | 4 | **4/4** | 4/4 | 4/4 | 0/296 | 0.000e+00 | 0.000e+00 | 0.0626 |
| 8 | 8 | **8/8** | 8/8 | 8/8 | 0/536 | 0.000e+00 | 0.000e+00 | 0.0626 |
| 16 | 16 | **16/16** | 16/16 | 16/16 | 0/1040 | 0.000e+00 | 0.000e+00 | 0.0626 |
| 4 (homog.) | 4 | 4/4 | 4/4 | 4/4 | 0/352 | 0.000e+00 | 0.000e+00 | 0.0583 |
| 8 (homog.) | 8 | 8/8 | 8/8 | 8/8 | 0/704 | 0.000e+00 | 0.000e+00 | 0.0629 |
| 16 (homog.) | 16 | 16/16 | 16/16 | 16/16 | 0/1408 | 0.000e+00 | 0.000e+00 | 0.0711 |

**31/31 scenarios, 0 of 2136 token decisions differ, and because the tokens are
identical the final action is bit-identical** (the `roots`/`rotations` columns
are exactly zero; the "worst angle" column is the `angle_between` float32 floor
of 22.5, not drift).

That is not luck, and the campaign can say why.  A token changes only if the
logit perturbation exceeds that decision's own margin — the gap between the best
and second-best Gumbel-noised score.  Measured on the same 16 scenarios:

* largest logit perturbation, batched vs serial: **3.357e-04** (growing from
  1.984e-04 at `B = 1`);
* **smallest** decision margin over all 16 scenarios: **2.985e-03** (median
  3.683e-02);
* ratio: **0.112x** — the perturbation is ~9x smaller than the tightest
  decision it could flip.

### 23.6 Verdict and the safe-B rule

Verdict per section 21 item 2 (throughput **and** token identity; never quoted as
a single-plan latency win): **the bake is usable, and on this evidence every
measured `B` up to 16 is token-identical and action-identical.**

**Safe-B rule (proposed as a decision, section 26):**

1. A bake batch is **reproducible** only if every scenario in it matches the
   same scenario run alone on `pose.tokens`, `pose.sampling_uniforms` and the
   final action.  The largest `B` that passes is that release's *reproducible
   bake tier*.
2. On this evidence the reproducible tier is **`B <= 16`** (tested), with the
   measured safety factor above: perturbation 0.112x of the tightest margin.
3. Any larger `B`, or any `B` whose campaign was not re-run for the release, is
   a **fast-draft tier only**, and the **batch size must be recorded in the
   product manifest** together with the campaign result.
4. Batching changes numerics (it is not bit-identical to `B = 1`); the tier is
   about *discrete* identity and the final action, not about bitwise floats.
5. **Group by duration.**  Raising `B` inside one duration group is worth more
   than raising `B` across mixed ones (23.3), and mixed groups compile one
   variant per duration.

## 24. Item 3 — host synchronisations per step

### 24.1 The count, and the three candidates

`transition.py` materialises device arrays with explicit `np.asarray(...)` calls;
`/tmp`-based instrumentation that patches `numpy.asarray` and names its call site
counts them per step (GPU, 6 steps, `capture_trace=False`):

| arm | reads/step | sites |
| --- | --- | --- |
| round-3 default | **6.00** | duration probe, root output, pose logits, decoder output, `roots`, `rotations` |
| `decoder_defer=1` | **5.00** | decoder output gone |
| `pred_local_defer=1` | **5.00** | root output gone |
| `decoder_defer=1,pred_local_defer=1` | **4.00** | both gone |
| `+ root_boundary_reuse=1` | **4.00** | (removes device work, not a read) |

Two of the six reads are covered by an earlier read of the same graph and cost no
extra wait, so the honest statement is **five waits per step, three with both
deferrals on**.  The candidates:

1. **`decoder_defer`** — hand the decoder's output straight to
   `motion_rep.decode_motion` instead of reading it back and re-uploading it.
   Same values, one fewer wait.
2. **`pred_local_defer`** — compute the predicted local root on the device
   (`_predicted_local_device`), so the root planner's output is never read back;
   the read that remains happens when the pose logits are fetched.  MLX's
   `arctan2` is *not* bit-identical to NumPy's (measured 2.384e-07, one ulp);
   `mx.remainder` is exact.
3. **`root_boundary_reuse`** — `run_transition` ran the duration probe and then
   `RootPlanner.plan`, and **both** rebuilt the boundary embedding from the same
   tensors (only the probe transformer was deduplicated by round 1's
   `root_dedup`).  Measured in isolation: boundary embedding **0.683 ms p50**,
   i.e. 25 % of the 2.70 ms probe stage, duplicated on every step.  `probe()` now
   returns the boundary tensors and `plan(boundary=...)` reuses them.

With `capture_trace=True` the deferrals are *counterproductive* (6 -> 9 reads):
the trace materialises values the default path already had on the host
(`predicted_local`, `pose_root`, `external_condition`, the decoder output).  They
are throughput-path switches, not trace-path ones.

### 24.2 Correctness gate (all three, and together)

GPU, 5 seeds x 6 steps, interleaved arms in one process, arms verified by
printing their resolved switch values:

| arm | `pose.tokens` identical | `sampling_uniforms` identical | duration identical | worst `roots` | worst `rotations` |
| --- | --- | --- | --- | --- | --- |
| `decoder_defer=1` | **30/30** | 30/30 | 30/30 | 0.000e+00 | 0.000e+00 |
| `pred_local_defer=1` | **30/30** | 30/30 | 30/30 | 0.000e+00 | 0.000e+00 |
| `root_boundary_reuse=1` | **30/30** | 30/30 | 30/30 | 0.000e+00 | 0.000e+00 |
| all three | **30/30** | 30/30 | 30/30 | 0.000e+00 | 0.000e+00 |

`mlx_align plan`/`trace` were re-run with the switches on: both `OK`, tokens and
uniforms at `0.000e+00`.  The GPU default document with all three switches on is
**byte-identical** to the round-3 default document.  Alignment does not regress.

### 24.3 Speed: the paired numbers, and the verdict

Interleaved inside one process, arm order rotated, same step sequence, p50/p90/max
per arm.

| run | load | decoder_defer | pred_local_defer | root_boundary_reuse | all three |
| --- | --- | --- | --- | --- | --- |
| 25 steps (pre-fix*) | 10.9 | 1.0203x [0.960-1.057] | 0.9792x [0.825-1.029] | — | 1.0100x [0.945-1.086] (two of three) |
| 50 steps (pre-fix*) | 23.5 | 0.9814x [0.908-1.052] | 0.9325x [0.817-1.043] | 1.0137x [0.894-1.112] | 1.0199x [0.953-1.105] |
| 50 steps | 12.6 | 1.0035x [0.960-1.065] | 1.0207x [0.950-1.081] | 1.0443x [0.964-1.121] | 1.0579x [1.006-1.108] |
| 50 steps | 13.8 | — | — | — | 1.0554x [1.004-1.097] |
| 50 steps | 14.9 | — | — | — | 1.0339x [0.953-1.081] |
| 50 steps | 13.0 | — | — | — | 1.0389x [1.015-1.087] |
| 50 steps | 13.1 | — | — | — | 1.0411x [1.014-1.099] |
| 50 steps | 13.1 | — | — | — | 1.0333x [0.994-1.080] |

\* "pre-fix": the first `pred_local_defer` implementation still materialised the
root planner's output, so it removed no read at all (5.00 -> 6.00 reads/step
before the fix, 4.00 after).  Those two rows are kept because they are honest
measurements of "move the maths to the device without removing the read", and
they are superseded for the switch's verdict.

Applying the frozen rule:

* **No single item reaches 1.05x.**  Their point estimates are ~1.00-1.02x
  (`decoder_defer`), ~1.02x (`pred_local_defer`), ~1.03-1.04x
  (`root_boundary_reuse`) — with inter-quartile ranges that straddle 1.0.
* **The combination does not reach it either once sampled properly**: six
  50-step post-fix samples give 1.0579, 1.0554, 1.0411, 1.0389, 1.0339, 1.0333
  — **median 1.0400x**, below the bar.  (The first two samples alone would have
  read as a pass; this is exactly why the criteria fix a bar and why a single
  run is not evidence.)
* **Verdict: all three default OFF**, recorded as rejected with these numbers.
  They remain available as switches, with a verified correctness gate, for a
  caller that wants the ~4 % on a throughput path — and the safe statement is
  that the four-to-five per-step host synchronisations are worth **~1-4 % of a
  step between them**, not the kernel-drain cost section 19 hypothesised.  The
  dependent kernel chain, not the drain count, is the remaining cost.

## 25. Round-3 correctness and tests

* `mlx_align.py plan` (reference vs the GPU default) and `mlx_align.py trace`
  (both backends): `ALIGNMENT OK` / `TRACE OK`, with `pose.tokens`,
  `pose.sampling_uniforms` and `decoder.quantized` at `0.000e+00`.
* Token identity, 5 seeds x 6 steps, both backends:
  `tests/test_mlx_agent_math.py::test_token_identity_end_to_end` passes.
* The bake seam is exact: supplying the pipeline's own pose logits reproduces
  `roots`, `rotations_xyzw` and `pose.tokens` bit for bit
  (`tests/test_mlx_bake.py::test_pose_logits_seam_reproduces_the_accepted_pipeline`).
* The batched forward's deviation is pinned as a **band**, so a future change
  that makes the lowering exact — or ten times worse — fails the test.
* `tests/test_mlx_optimizations.py` now pins the backend-conditional rule in
  both directions, the fact that only the three compile switches are conditional,
  that an explicit environment value wins on either backend, and that the three
  round-3 candidates are declared off.
* `pytest`: **140 passed, 1 skipped** with round 3 in place, against the
  **135 passed, 1 skipped** this round started from (the count may only grow).

## 26. What is left, and the decisions this round proposes

1. **The bake's ceiling is the pose stage's share, not the batch size.**
   Batching the pose stage alone is capped at ~1.25x end-to-end (23.4) because
   the root planner and the decoder are ~76 % of a step and are not batched here.
   The next bake step is batching those two stages (they are per-time
   convolutional/transformer work with a natural leading axis), which is what
   would move the bake towards the stage-level 2.5-7.2x.
2. **Duration-homogeneous batches.**  Fragmentation into token-count groups cost
   15-25 % of the achievable speedup at `B = 16` (23.3).  A producer that can fix
   the duration (restrict `allowed_tokens`) should do so, and the manifest should
   record both the batch size and the grouping.
3. **A decision proposal for the bake rule** is written to
   `docs/decisions/0006-batch-bake-safe-b.md` (proposal only; the lead ratifies).
4. **The host synchronisations are not the lever.**  Removing two of five waits
   and a duplicated 0.68 ms embedding is worth ~4 % (24.3).  If single-plan
   latency matters again, the measured targets are the serial root/decoder stages
   and the host maths, not the drain count.
5. **Re-measure the `*_compile` default on an idle machine** before treating the
   CPU latency comparison of 22.3 as final: all three arms sit within 1.25 % of
   each other at load 9-14, which is at the resolution limit of the method.

## 27. Reproduction (round 3)

```
# decision 0005: the GPU default path is unchanged; always pass --out
./.venv/bin/python scripts/mlx_plan.py --device gpu --out /tmp/r3-plan-gpu.json
diff -q /tmp/r3-plan-gpu.json out/plan-gpu-before.json      # byte-identical

# the CPU default, per process and never across two backends in one run
./.venv/bin/python scripts/mlx_bench.py --devices cpu --steps 25
MOTIONBRICKS_MLX_OPT=0 ./.venv/bin/python scripts/mlx_bench.py --devices cpu --steps 25

# the interleaved paired arms (25+ steps, order rotated inside one process)
./.venv/bin/python scripts/mlx_bench.py --paired "" "MOTIONBRICKS_MLX_OPT=1" \
    "MOTIONBRICKS_MLX_OPT=0" --devices cpu --steps 25 --paired-tokens
./.venv/bin/python scripts/mlx_bench.py --paired "" \
    "MOTIONBRICKS_MLX_OPT=decoder_defer=1,pred_local_defer=1,root_boundary_reuse=1" \
    --devices gpu --steps 50

# the batch bake and its token campaign
./.venv/bin/python scripts/mlx_bench.py --bake 1,2,4,8,16 --bake-repeats 5 --devices gpu
./.venv/bin/python scripts/mlx_bench.py --bake 4,8,16 --bake-homogeneous \
    --bake-repeats 5 --devices gpu

# alignment, determinism, tests
./.venv/bin/python scripts/mlx_align.py plan out/ref/plan-ref.json /tmp/r3-plan-gpu.json
./.venv/bin/python scripts/mlx_align.py trace out/ref/trace/transition-0.json --device gpu
./.venv/bin/python -m pytest -q
```

The per-arm scratch tools used for the round-3 numbers (the 5x6 token gate, the
`np.asarray` drain counter) lived in `/tmp/r3/`; their measurements are quoted
above and the properties they rest on are pinned by `tests/test_mlx_bake.py` and
`tests/test_mlx_optimizations.py`.
