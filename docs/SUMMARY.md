# SUMMARY · A 侧（MLX 推理）收尾基线

**这份文档是 A 侧的成品**：任何人凭它可以复现全部数字、知道档位与边界，并用同一批命令验证。
所有数字都在 **HEAD 提交的这棵树**上量过；长版证据在
[optimization-report.md](optimization-report.md)（本轮是第 43–49 节）。

一句话结论：默认单场景规划 **GPU p50 26.6 ms / CPU 173.1 ms**、逐位一致；
烘焙 **release 档 = 时长混合 B ≤ 8 / 时长同构 B ≤ 16**，同构 B=16 端到端 **4.44×**
（153 场景/秒）；量化明确不做。

---

## 1. 已验收状态（默认路径）

命令（两条**分开的进程**，因为同进程跑两个后端会把第二个后端的峰值内存算进去，
见 [optimization-report.md](optimization-report.md) 第 22.2 节）：

```
./.venv/bin/python scripts/mlx_bench.py --steps 25 --devices gpu --json /tmp/gpu.json
./.venv/bin/python scripts/mlx_bench.py --steps 25 --devices cpu --json /tmp/cpu.json
```

实测（load average 6.1–7.7，25 步，4 个真实 style 轮转）：

| 后端 | p50 | p90 | max | 冷启动（载入+首步） | MLX 峰值 | 进程 RSS | 确定性 |
|---|---|---|---|---|---|---|---|
| GPU (Metal) | **26.6 ms** | 37.2 | 40.4 | 317 ms | 779 MB | 846 MB | True |
| CPU | **173.1 ms** | 198.0 | 213.3 | 497 ms | 748 MB | 856 MB | True |

初始验收基线是 GPU p50 55.5 ms（第 1 轮），即默认路径累计 **2.09×**。

**逐位一致性证据**（这三条是本项目真正的验收线）：

```
./.venv/bin/python scripts/mlx_plan.py --device gpu --out /tmp/step-gpu.json
./.venv/bin/python scripts/mlx_plan.py --device cpu --out /tmp/step-cpu.json
diff -q /tmp/step-gpu.json out/plan-mlx-gpu.json     # 无输出 = 相同
diff -q /tmp/step-cpu.json out/plan-mlx.json
shasum -a 256 /tmp/step-gpu.json /tmp/step-cpu.json
./.venv/bin/python scripts/mlx_align.py plan out/ref/plan-ref.json /tmp/step-gpu.json
./.venv/bin/python scripts/mlx_align.py trace out/ref/trace/transition-0.json --device gpu
./.venv/bin/python scripts/mlx_align.py trace out/ref/trace/transition-0.json --device cpu
./.venv/bin/python -m pytest -q
```

| 检查 | 实测 |
|---|---|
| GPU plan 文档 sha256 | `ccda1952c967c8d7e893a795908f7de278f2acda1ce5ba1e1889692a8904a7c5`（与 `out/plan-mlx-gpu.json` 逐字节相同） |
| CPU plan 文档 sha256 | `262e77d438daa8a790eccf23f8cfd696bf620a304bb4d42b54e38282e085b64a`（与 `out/plan-mlx.json` 逐字节相同） |
| `mlx_align plan` | `ALIGNMENT OK`；最差余量 **rotation angle 0.26×** 容差（roots 6.8e-07 m、rotations 8.7e-06） |
| `mlx_align trace --device gpu` / `cpu` | 双双 `TRACE OK`；`pose.tokens`、`pose.sampling_uniforms`、`decoder.quantized` 均 `0.000e+00` |
| 5 seed × 6 步跨版门（对比 `f699b84` 工作树） | tokens **30/30**、uniforms **30/30**、duration **30/30**，**0/2416** 决策不同，最终动作偏差 **0.000e+00** |
| `pytest -q` | **209 passed, 1 skipped**（本轮起点 204/1，只增不减） |

`mlx_align plan` 的容差是 1.0e-03 m / 2.0e-03（rotations）/ 2.0e-01 度（角）；
trace 的容差见 [scripts/mlx_align.py](scripts/mlx_align.py)。

---

## 2. 烘焙

**这一节是 A 侧的产品数字。** 三臂 = 串行（每场景单独规划，即 B=1 语义）／round-4 stage driver
（`drive_batch`，靠重启 `run_transition` 在阶段边界同步）／round-5 单趟编排器
（`drive_batch_single_pass`）+ round-6 的批量解码。**release 配置是 `--bake-stages all+decode`。**

```
# 同构（duration-homogeneous；B>8 的必选分组）
./.venv/bin/python scripts/mlx_bench.py --bake-stages all+decode \
    --bake 1,2,4,8,16 --bake-homogeneous --bake-single-pass --bake-repeats 9 \
    --devices gpu --json /tmp/release-homog.json
# 混合（duration-mixed）
./.venv/bin/python scripts/mlx_bench.py --bake-stages all+decode \
    --bake 1,2,4,8,16 --bake-single-pass --bake-repeats 9 \
    --devices gpu --json /tmp/release-mixed.json
```

GPU，9 次重复取中位数，load 6.7–12.7，同进程三臂轮换：

| 分组 | B | 串行 p50 | round-4 批量 | 单趟（release） | 单趟吞吐 | token 分组 |
|---|---|---|---|---|---|---|
| 混合 | 1 | 26.17 ms | 30.81（0.849×） | 27.24（0.961×） | 36.7/s | {11:1} |
| 混合 | 2 | 59.42 | 36.29（1.637×） | 36.16（1.643×） | 55.3/s | {11:2} |
| 混合 | 4 | 113.74 | 93.14（1.221×） | 94.82（1.200×） | 42.2/s | {11:2,7:1,8:1} |
| 混合 | 8 | 218.91 | 143.92（1.521×） | **131.07（1.670×）** | 61.0/s | {11:3,7:2,8:1,6:2} |
| 混合 | 16 | 470.72 | 194.88（2.415×） | 167.50（2.810×） | 95.5/s | {11:5,7:7,8:1,6:3} |
| 同构 | 1 | 28.37 | 30.00（0.946×） | 31.30（0.906×） | 32.0/s | {11:1} |
| 同构 | 2 | 57.04 | 38.96（1.464×） | 32.80（1.739×） | 61.0/s | {11:2} |
| 同构 | 4 | 112.72 | 52.54（2.145×） | **43.37（2.599×）** | 92.2/s | {11:4} |
| 同构 | 8 | 225.82 | 80.66（2.800×） | **71.36（3.165×）** | 112.1/s | {11:8} |
| 同构 | 16 | 463.25 | 144.59（3.204×） | **104.26（4.443×）** | **153.5/s** | {11:16} |

同构 B=16 另一次独立运行给 4.429×（144.5/s）；两次都 >4.4×。
round-4 与单趟两臂在每个 B、每种分组下 **逐位相同**（`cross_arm_bitwise.bit_identical=True`）。

**档位与依据（decision [0007](decisions/0007-bake-full-stage-tier.md)＋[0008](decisions/0008-margin-threshold-and-grouping.md)）**：
判据是 **扰动 / 最紧决策余量 ≤ 0.70**（0008 第 1 条）；`B > 8` 必须时长同构；`B = 1` 永不使用。

| 分组 | B | 扰动 / 余量 | 余量倍数 | 档位 |
|---|---|---|---|---|
| 混合 | 1 | 0.2019 | 4.95× | release |
| 混合 | 2 | 0.4843 | 2.06× | release |
| 混合 | 4 | 0.6849 | 1.46× | release |
| 混合 | 8 | **0.6695** | 1.49× | **release（上限）** |
| 混合 | 16 | **1.0426** | **0（零余量）** | draft |
| 同构 | 2 | 0.2441 | 4.10× | release |
| 同构 | 4 | 0.6031 | 1.66× | release |
| 同构 | 8 | 0.5290 | 1.89× | release |
| 同构 | 16 | 0.4089 | 2.45× | release |

最紧决策余量 = 2.985e-03（Gumbel 加噪后 top1−top2），扰动 = 批量与 B=1 的 logits 最大差。
**结论：release = 混合 B ≤ 8 或时长同构 B ≤ 16；draft = 混合 B ≥ 16；B = 1 永不使用。**
本轮（第 6 轮）重跑后裕度与第 5 轮**逐位相同**——批量解码不改变任何数值，这正是重点。

**每场景分解（B=16 同构，直接仪器，闭合 0.94%）**：见第 45 节。要点：
批量前向 3.55 ms（`stack`+`forward`+`eval`，其中 `mx.eval` 3.21）、宿主胶水 1.56、
每场景 setup 0.79、`finish_plan` 0.32、编排与 resume 0.06。

**产物 manifest 必记字段（0007 第 4 条 ＋ 0008）** —— `bake.manifest_fields` 逐 B 生成，
12 项一个不少：`batch_size`、`bake_tier`、`bake_stages`、`bake_stage_switches`、
`bake_grouping`（含 token 分组）、`bake_campaign_scenarios`、`bake_token_decisions_differing`、
`bake_worst_roots_m`、`bake_worst_rotations`、`bake_safety_ratio`、`bake_min_decision_margin`、
`bake_action_bit_identical`。**最后一项如实为 `false`**：全阶段批量后最终动作不再逐位
（roots ≤ 2.027e-06 m = 容差的 2.0e-03×、rotations ≤ 2.753e-05 = 1.4e-02×）。

**硬规则（不得绕过）**：

1. `B = 1` **永不**用于烘焙（比串行慢，混合 0.849–1.008×）；
2. `B > 8` **必须**按时长同构成组（混合 B=16 零余量）；
3. 换模型 / 换权重 / 换指令分布 → **必须重跑战役并重算裕度**，任一 B 比值 > 0.70 即机械降档；
4. 任何 `mlx_plan.py` 调用**必须显式 `--out`**（第 2 轮覆盖过一次已验收文档）。

---

## 3. 明确不做的

| 不做 | 为什么（证据） |
|---|---|
| **量化（q8/q4）** | 会改动作：q4 逐 token 差异 27–31%、3/6 步连时长都变、roots 30–91× 容差；q8 差异 1.5–3.6%、roots 2.2–25.6×；`mlx_align plan` 对参考 FAIL（q8 4 个数组、q4 9 个）。裁定见 [0004](decisions/0004-adopt-bit-identical-f32.md)、[0005](decisions/0005-backend-conditional-compile.md) 与报告第 7/9 节。 |
| **单场景实时路径再优化** | 宿主数学已从 17.58 → 1.74 ms/步（报告第 12–13 节）；把 FK/约束搬进 MLX 图在 B=1 是**净亏**（1.64 ms vs 宿主 0.371 ms，且 `cos/sin/arctan2` 与 NumPy 差 1–2 ULP）；`pred_local_defer` / `decoder_defer` 在 B=1 只有 1.0207× / 1.0035×，低于 1.05× 门槛，已按判据否决。**它们只在批量里才值钱**——这正是第 6 轮对解码做的事。 |
| **参考运行时侧的事** | `~/dev/motion-bricks-cpp` 全程只读、未改一个文件；GGUF/引擎侧不在 A 侧范围。 |
| **训练 / 评测** | `training/**`、`eval/**` 本轮未动；本轮的 bake 改动只影响离线烘焙。 |

---

## 4. 一键复现（每条都实际跑过）

```
# ── 默认路径：两个 sha256 + 冷启动/内存/确定性（分开进程） ──────────────
./.venv/bin/python scripts/mlx_plan.py --device gpu --out /tmp/step-gpu.json
./.venv/bin/python scripts/mlx_plan.py --device cpu --out /tmp/step-cpu.json
diff -q /tmp/step-gpu.json out/plan-mlx-gpu.json && diff -q /tmp/step-cpu.json out/plan-mlx.json
./.venv/bin/python scripts/mlx_bench.py --steps 25 --devices gpu --json /tmp/gpu.json
./.venv/bin/python scripts/mlx_bench.py --steps 25 --devices cpu --json /tmp/cpu.json

# ── 对齐：plan（对参考）与 trace（两后端） ────────────────────────────
./.venv/bin/python scripts/mlx_align.py plan out/ref/plan-ref.json /tmp/step-gpu.json
./.venv/bin/python scripts/mlx_align.py trace out/ref/trace/transition-0.json --device gpu
./.venv/bin/python scripts/mlx_align.py trace out/ref/trace/transition-0.json --device cpu

# ── 烘焙：release 档（B ≤ 8）与两个分组 ──────────────────────────────
./.venv/bin/python scripts/mlx_bench.py --bake-stages all+decode \
    --bake 1,2,4,8 --bake-single-pass --bake-repeats 9 --devices gpu \
    --json /tmp/release-mixed.json
./.venv/bin/python scripts/mlx_bench.py --bake-stages all+decode \
    --bake 1,2,4,8 --bake-homogeneous --bake-single-pass --bake-repeats 9 \
    --devices gpu --json /tmp/release-homog.json
# 同构 B=16（draft 档上限，同构才允许）
./.venv/bin/python scripts/mlx_bench.py --bake-stages all+decode \
    --bake 16 --bake-homogeneous --bake-single-pass --bake-repeats 9 \
    --devices gpu --json /tmp/release-b16.json

# ── 直接仪器：闭合分解（B=16 同构）与批量解码的配对前后 ────────────────
./.venv/bin/python scripts/mlx_bench.py --bake-stages all --bake 16 \
    --bake-homogeneous --bake-repeats 9 --bake-profile --devices gpu \
    --json /tmp/profile-b16-off.json
./.venv/bin/python scripts/mlx_bench.py --bake-stages all+decode --bake 16 \
    --bake-homogeneous --bake-repeats 15 --bake-profile --devices gpu \
    --json /tmp/profile-b16-on.json
for b in 4 8 16; do ./.venv/bin/python scripts/mlx_bench.py --bake $b \
    --bake-homogeneous --bake-repeats 25 --bake-decode-paired --devices gpu \
    --json /tmp/paired-b$b.json; done

# ── 正确性门 ─────────────────────────────────────────────────────────
./.venv/bin/python -m pytest -q
git worktree add /tmp/before f699b84          # 第 5 轮末提交
./.venv/bin/python /tmp/r5/campaign.py /tmp/before/src /tmp/campaign-before.json
./.venv/bin/python /tmp/r5/campaign.py "$PWD/src" /tmp/campaign-after.json
# 比较两份 JSON 的 tokens / uniforms / selected / roots / rotations
```

---

## 5. 诚实边界

1. **机器不空闲。** 本轮全部测量期间的 1 分钟 load average 是 **5.6–14.8**（桌面进程：
   mihomo、WindowServer、DSH renderer、Chrome、Lark；没有训练/评测进程）。
   绝对值被抬高约 **10 %**。因此每一项前后对比都是**同进程内交错配对**（顺序轮换）的比值，
   单臂绝对值只作佐证。**一个具体教训**：把 B=1/2/4/8/16 放在同一个进程里配对时，B=8 曾量出
   0.936×，而 B=16 是 1.175×；把每个 B 放进各自的进程后 B=8 是 **1.1726×**。
   长命进程会在 B 之间积累状态，所以本轮的配对数字**每个 B 一个进程**。
2. **`mlx_align` 的旋转角列有地板。** `plan` 比较里 rotation angle 列读到
   4.3e-02–5.1e-02 度（容差 2.0e-01 度，即 **0.26×**），这是该列的地板（浮点角度的量化），
   不是回归；其余列都在 1e-07–1e-05 量级。
3. **未归因项只剩 0.92–0.94 %。** 直接仪器把单趟烘焙的每个叶子段都括起来，各段之和与实测
   区间闭合，残差是 0.057–0.089 ms/场景，全部是编排循环本身、生成器 resume 机制与括号开销。
   **没有别的未归因项**。第 5 轮那个 2.29 ms 的来历已在第 45 节逐项拆开：
   前向仪器乐观 +1.33、宿主注入仪器悲观 −0.25、另有 setup +0.78 与 `finish_plan` +0.32
   两个**两件仪器都没数过**的项。
4. **批量解码只在 B ≥ 2 才有意义。** B=1 的配对比值是 **0.9925×**（中性偏负），
   所以它和 0007 的"B=1 永不使用"一致；B=4 是 1.056×（下四分位 0.935，边缘），
   B=8 1.173×、B=16 1.221×（下四分位分别 1.007 / 1.132）。
5. **跨后端内存那一行不能单独读。** 同进程先跑 GPU 再跑 CPU 会把第二个后端的峰值算高
   （报告第 22.2 节）；本文件的表来自分开的进程。
6. **动作一致性按容差判，不按逐位判**（0007 第 4 条）：全阶段批量后最终动作
   roots ≤ 2.027e-06 m、rotations ≤ 2.753e-05，分别是 `mlx_align` 容差的 2.0e-03× 与 1.4e-02×；
   manifest 里的 `bake_action_bit_identical=false` 就是这件事的如实记录。

---

## 6. 本轮（第 6 轮）改了什么

只加了一个**默认关闭**的烘焙开关，其余都是仪器与文档：

* `bake_decode_batch`（`nn.OPT_DEFAULTS`，默认 `False`）：把运动解码变成 step generator 的
  **第五个前向点**，由单趟编排器按形状分组、每组**一次**批量图（`motion_rep.decode_motion_batch`）。
  它只能被单趟编排器拥有（`batched_stages()` 里永远没有它），所以 round-3/round-4 路径一字未动，
  默认单场景路径的两个 sha256 也不动。
* 效果：B=16 同构 **3.49–3.72× → 4.43–4.44×**；配对实测 B=8 1.1726×、B=16 1.2207×，
  两臂 `compare_bitwise` **逐位相同**。
* 仪器：`scripts/mlx_bench.py --bake-profile`（闭合分解）与 `--bake-decode-paired`（配对前后），
  都是追加的开关，既有语义不变。
* 提案：[decisions/0009](decisions/0009-batch-the-motion-decode.md)（把批量解码列为 release 烘焙配置的
  一部分，阈值为 `bake_tier` 里落地的 0008 规则）。
