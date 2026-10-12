# 状态 · DeepSeek（A：表示与数据）

- 当前：等待人类两个决定（Python 环境怎么建；根朝向是否保留完整姿态）——在此之前不动表示代码。
- 正在动的文件：无。
- 下一步：把 `docs/motion-representation.md` 落成 `motion/representation.py`（常量与编解码），
  并写 `scripts/export_from_space.py`，产出第一批真实 `clip.npz`。
- 环境：已就绪——仓库内 `.venv`（Python 3.13.12 + MLX 0.32.3 + NumPy 2.5.4，Metal 设备），
  见 `docs/decisions/0001-python-environment.md`；`scripts/check_env.py` 打 READY。
- 已查既有实现：见 `docs/decisions/0002-prior-art.md`——MotionBricks 已有 GGML 版（295★）与 Core ML 版，
  MLX 侧只有 VQ-VAE 零件、**没有动作生成**。定位据此改为"MLX 做训练与数据，产物导出给既有运行时或烘焙进引擎"。
- 下游运行时已定：`motion-bricks.cpp`（Apache-2.0、GGML、GGUF 边界），见 `docs/decisions/0003-downstream-runtime.md`。
- 待办（第 1 步之后）：① 确认 macOS/Apple Silicon 的 CMake preset 与 Metal 路径能否跑通它的 bundle；
  ② 把它的 GGUF schema 约束写进动作模型设计——我们自训的权重必须导得出去。
- 仍等人类一个决定：动作表示里根朝向是否保留完整姿态（v0 只有偏航）。

---

## 2026-10-12 · MotionBricks 推理的 MLX 移植（人类指派，数值对齐为验收线）

**越界说明（人类明确授权）**：本轮按人类指示独占 `src/motionbricks_mlx/{models,data,export}/**`、
`scripts/mlx_*.py`、`tests/test_mlx_*.py`。`models/`、`export/` 按 README 属 B，本任务期间由我写；
B 若要在同一批文件上改动，先协调。`src/motionbricks_mlx/training/` 不是我建的，我没碰。

**参考实现只读**：`~/dev/motion-bricks-cpp/repo` 未改一个文件（`git -C ggml status --short` 为空；
父仓库只显示子模块指针差异，是clone 时就有的）。我另建了 `tools/ref_harness.cpp`（链接它的**静态库** +
内部头文件）用于取参考值，产物全落在本仓库 `out/`（gitignored）。

**做法与结论**

- 权重：`scripts/mlx_convert_gguf.py` 把 4 个 GGUF 转成 `checkpoints/g1-f32-mlx/*.npz`，
  逐张量比对：**F32/I32 载荷与参考读到的字节完全一致**（`byte_identical_to_gguf_payload` 全 true），
  张量数/参数量与 `src/model.cpp:40-53` 强制的一致，支持张量与参考 trace 的 `stats.*`、Go 驱动的
  skeleton **逐位一致**。参考有 3 个名字因 `GGML_MAX_NAME` 被压成 `attn.out_proj.weight`，
  运行时的回退规则（`neural_runtime.cpp:272-276`）已复刻并被测试锁住。
- 前向：`models/{nn,root,pose,decoder,motion_rep,sampling,transition}.py` 逐层照搬，布局按
  "MLX 视图 = reversed(ggml ne)" 处理，conv 用 im2col+matmul 等价式（`mx.conv1d` 快路径与显式
  im2col 在测试里对比）。
- agent 层：`data/{style,agent}.py`（.mbstyle 也是 GGUF；spring/canonicalize/restore_world/context blend）。

**数值对齐（同一输入、同一 seed，与参考自身输出逐数组比）**

| 对照 | roots | rotations | world 坐标 |
|---|---|---|---|
| 参考 Go 驱动 `plan-walk.json`（两步、Gumbel） | 2.1e-6 m | 2.9e-5 | 2.3e-5 m |
| 参考 trace 逐阶段（`transition-0`） | decoded.roots 4.4e-7 m | decoder.output 5.1e-5 | — |
| MLX CPU vs MLX GPU（Metal） | 2.4e-6 m | 4.5e-5 | — |

`pose.tokens` / `pose.sampling_uniforms` / `decoder.quantized` 与参考**完全相等**（0 差）。
两个 bug 是 trace 比对抓出来的，都不是"看着像"能发现的：
① `canonicalize` 与 `restore_world` **只重写 joint 0**（父相对旋转下等价于整体旋转，改全部关节会双倍施加）；
② agent 的 `matrix_xyzw` **不**强制 w≥0（`motion_rep.cpp` 的 `matrix_quaternion` 才强制）。

**已知敏感点（写进设计约束）**：pose planner 对 root condition 的扰动放大约 1e3–1e4 倍
（实测：输入 3.1e-6 的差 → logits 2.4e-2，且线性）。所以跨 BLAS 后端的 pose logits 不可能逐位一致，
验收只能落在 **采样出的离散 token 与解码后的动作**上——这两项确实一致。

**性能（同一台 M1 Max，25 步规划）**：MLX CPU p50 183 ms / p90 213 ms / max 223 ms；
MLX GPU p50 57 ms / p90 65 ms / max 212 ms；冷启动 457 ms(CPU)/310 ms(GPU)；
峰值 MLX 748/779 MB（进程 RSS 852 MB）。参考 GGML（debug 构建、CPU、10 线程）p50 **1858 ms**。
即 CPU 约 10×、Metal 约 33×。

---

## 2026-10-12 · 推理优化一轮（判据先写死，再动手）

判据在动手**之前**写进 `docs/optimization-report.md` 第 0 节并冻结：性能门 = p50 改善 ≥1.5× **或**
冷启动/常驻内存 ≥2×；正确性门 = `mlx_align` 仍 OK，且 N≥5 组 seed 的 `pose.tokens` 与 f32 基线
**逐 token 一致**（不一致时只允许"差异率 + 动作偏差 ≤0.5× 容差"并限定用途）。

**结论：一条都没"采用"。**（全部按上表判据判定，未事后改判）

- 唯一能过性能门的是量化 q4（常驻内存 791→222 MB = **3.56×**，p50 1.36×），但 token 门不过：
  5 组 seed 全 0/6 步 token 一致，逐 token 差异 27–31%，3/6 步连**时长**都变了，
  roots 偏差 30–91× 容差、旋转 157–216× 容差。q8 也一样不过（1.5–3.6% 差异、roots 2.2–25.6× 容差）。
  `mlx_align plan` 对参考同样是 FAIL（q8 4 个数组、q4 9 个）。→ **否决（正确性）**。
- 所有 f32 改动（重复 duration probe 去重、形状常量提升、`mx.compile`、host NumPy 向量化）
  **token 完全一致**（GPU 30/30 步逐位、CPU 16/16 步逐 token），但只有 **1.19×（pooled）/1.12×（paired）**
  p50，够不到 1.5×。→ 逐项 **否决**。
- 既然"否决"，这些开关一律**默认关闭**；默认路径重放已验收产物 `out/plan-mlx.json` **0 个值不同**。
  要开就 `MOTIONBRICKS_MLX_OPT=1`（或 `MOTIONBRICKS_MLX_QUANT=q4`）。这是留给人类/父 agent 的取舍，
  不是我事后放宽判据。

**诊断（比结论更重要）**：把每个 stage 拆成"建图 / `mx.eval`"两段后，建图只占 **2–8 %**，92–98 % 是
**GPU 上互相依赖的小 kernel**（batch ≤16），所以 `mx.compile` 只能优化那 2–8 %——这解释了它为何
几乎无用（pose −6 %、root −14 %、decoder −11 %，合计 2 ms/步）。真正的杠杆有两个：

1. **`data/agent.py` 每步 19.9 ms / 55 ms（36 %）是 host NumPy**（`build_constraints` 14.5 ms，其中
   `_matrix_xyzw` 7.9、`_cont6d_matrix` 4.9；每次调用 ~35 µs 而数组只有 3×3 / 34 关节——纯调用开销）。
   该文件不在本轮授权名单里，**没碰**，已写进报告第 10 节：这是下一段 1.2–1.4×，需要与 B 线协调。
2. **批量**：同一份 pose 权重跑 B 个场景，B=2/4/8 的单场景吞吐 2.37×/3.12×/6.05×，而整批延迟只从
   7.1 涨到 14.7 ms（GPU 在 B=1 时严重空转）。但单次规划 p50 不变、且 B≥4 不再逐位一致（1.5e-5），
   判据不覆盖它 → **记为否决**（batch-bake 可选，未采用）。

**红线遵守**：只改 `models/{nn,root,pose,decoder,transition,bundle,sampling}.py`、`scripts/mlx_bench.py`、
新增 `tests/test_mlx_optimizations.py`、`docs/optimization-report.md` 与本文件；`training/**`、
`train_*.py`、`motion-representation.md`、`README.md`、`NOTICE`、`LICENSE`、`pyproject.toml` 未动；
参考仓库只读；无 `mlx_plan.py`/`mlx_align.py` 语义改动（只加了不改变默认行为的开关，且默认关闭）。

**测量环境**：机器**不空闲**，load average 6.3–17.4（桌面进程：mihomo ~85 % 单核、WindowServer、
DSH renderer、Chrome、Lark、Codex；训练侧当时没在跑）。因此所有前后对比都在**同一进程内交替**
（A,B,A,B…，每步轮转先后顺序）取配对比值，单臂 bench 只作佐证。

**测试**：`pytest` **124 passed, 1 skipped**（新增 41 个，锁住开关语法、SplitMix64 向量化逐位一致、
每个开关与总开关的 token 一致率、编译变体只建一次、零 mask 与无 mask 等价、量化只作用于矩阵且默认不动）。
`mlx_plan.py --repeat 2` 两种状态都 "deterministic"。

## 2026-10-12 优化轮（已收尾）
- 判据先冻结后动手，逐项判定**全部否决**；详见 `docs/optimization-report.md`（481 行）。
- 人类（Lead）裁定见 `docs/decisions/0004-adopt-bit-identical-f32.md`：f32 那组**默认开启**（逐位一致、零回归风险），
  量化**继续关闭**（q8/q4 会把动作改掉，有 token 级证据），并**授权**我做 `data/agent.py` 的 host NumPy 向量化。
- 下一页工作：① 把 `MOTIONBRICKS_MLX_OPT` 默认值翻成开（改一行 + 重跑 `pytest`/`mlx_align` 验证）；
  ② `data/agent.py` 向量化（目标 1.2–1.4×，验收=逐位一致）；③ 等 GLM 的动作质量门建立后再讨论离线烘焙用不用 q8。

## 2026-10-12 · 第二轮：宿主数学（decision 0004 授权的那一页）

**① 默认值翻转（decision 0004 第 1 条）— 已完成。** `models/nn.py` 现在把九个 f32 开关写在一张表
`OPT_DEFAULTS` 里，调用点统一走 `_opt(name)`，所以调用点不会与裁定漂移。`MOTIONBRICKS_MLX_OPT=0`
回到优化前的路径；语法按顺序生效，所以旧的"已验收路径 + 单个开关"写法仍然可用（`=0,pose_compile=1`）。

| 检查（GPU，环境里没有 `MOTIONBRICKS_MLX_OPT`） | 结果 |
|---|---|
| 默认路径 plan 文档 vs `MOTIONBRICKS_MLX_OPT=0` 文档 | **文件逐字节相同** |
| 两者之间 `mlx_align plan` | 每个数组 `0.000e+00` |
| `mlx_align plan out/ref/plan-ref.json <默认>` | `ALIGNMENT OK`，最差余量 rotation angle **0.24×** 容差 |
| `mlx_align trace … --device gpu` / `--device cpu` | 双双 `TRACE OK`，`pose.tokens` 与 `sampling_uniforms` `0.000e+00` |
| `mlx_plan.py --repeat 2`（gpu/cpu） | 两种后端都 deterministic |
| `pytest` | **125 passed, 1 skipped** |

`out/plan-mlx.json` 是 **CPU** 文档，GPU 文档与它差 2.354e-06 m——那是已知的后端差，**不是**这次翻转
造成的：`MOTIONBRICKS_MLX_OPT=0` 的 GPU 文档与它的差**完全相同**。翻转本身在两种后端上都是数值空操作。

**② 宿主数学批量化（第 1 项）— 采用，默认开启。** 每个 `_ref` 原样保留标量循环，新增批量化孪生体，
调度器读 `MOTIONBRICKS_MLX_AGENT`（默认开）。

| 函数 | 前：调用/步 | 前：ms/步 | 后：调用/步 | 后：ms/步 |
|---|---|---|---|---|
| `build_constraints` | 1 | **14.095** | 1 | **1.007** |
| `restore_world` | 1 | **2.523** | 1 | **0.397** |
| `apply_context_blend` | 1 | **0.681** | 1 | **0.176** |
| `canonicalize` | 1 | **0.280** | 1 | **0.158** |
| **宿主合计** | | **17.58** | | **1.74** |
| `_matrix_xyzw` | 180.6 | 7.560 | 4.0 | 0.345 |
| `_cont6d_matrix` | 136.0 | 4.673 | 1.0 | 0.108 |
| `quaternion_matrix` | 182.6 | 1.815 | 6.0 | 0.166 |

配对测量（同进程内交替，默认路径）：25 步 → **1.576× [1.410-1.685]**（45.11→28.08 ms，load 11.6→17.3）；
40 步 → **1.556× [1.442-1.814]**（44.53→30.85 ms，load 16.7→17.3）。两次都 **0 个 token 不同**
（25/25、40/40 步）。

**③ 搬进 MLX 图（第 2 项）— 有据否决。** 三条测量，都不是"讲道理"：
① 第 1 项之后宿主只剩 1.74 ms/29 ms（6 %），天花板约 1.06×；② 第 2 项想省掉的约束上传实测
**0.0059 ms p50**，而每步 4–6 次的 device sync 是宿主**决策**（时长选择、Gumbel 采样、码本查表），
第 2 项消不掉；③ MLX 版本的代价更高：FK 链（34 级依赖 × 4 帧）build 0.320 + eval 1.317 = **1.64 ms**，
而宿主只要 **0.371 ms**；目标块 0.061 + 0.546 = 0.61 ms。另外逐位一致本来就不可能：MLX 的 `cos`/`sin`
在 30 %/31 % 的输入上与 NumPy 差 1 ULP，`arctan2` 34 % 差到 2 ULP，`arccos` 53 % 差到 2 ULP
（`sqrt`/`remainder` 精确）。**结论：净亏，默认关闭；若将来重做，适用范围是 batch-bake/离线。**

**④ 编译内核（第 3 项）— 未触发。** 冻结顺序规定"只有 1、2 之后宿主仍是瓶颈"才做；宿主现在占 6 %，
第 2 项又是净亏，所以没写内核。真到那步首选 `mx.fast.metal_kernel`（一次派发、无 C 扩展负担），
不引入 Cython/pybind11。

**⑤ 正确性报告（判据要的数字）。** 端到端 5 组 seed × 6 步：GPU 与 CPU 都是
**30/30 步 `pose.tokens` 一致、0/2416 token 不同**、`sampling_uniforms` 30/30 一致，
**roots/rotations 最大绝对差 0.000e+00（0.000× 容差）**——这次重写在实际数据上**没有任何舍入变化**，
比判据要求的更强。逐项比对：4 个真实 style × 9 seed × 3 种命令形状 = **756 个数组全逐位一致**。
plan 文档：默认路径与优化前默认路径**逐字节相同**；优化前路径重放 `out/plan-mlx.json`
（sha256 `262e77d4…`）**逐字节相同**。

**⑥ 两个 NumPy 陷阱（靠测量发现，已用测试钉住）**：
① **gemv ≠ gemm**——参考用逐关节 `matrix @ vector`，写成 `@ vectors.T` 会走不同 BLAS 内核，
随机 3×3 差 2.384e-07、**真实 `walk_stealth` 约束差 2.980e-08**，足以改变第二步的动作；
`einsum('fij,fkj->fki')` 才能逐位复现 gemv。
② **`np.sum(a*b)` ≠ `np.dot(a,b)`**——BLAS `sdot` 的累加不同，随机 4 维向量差 4.768e-07；
批量矩阵-向量积能逐位复现（`dot_rows`）。
第三个缺陷只有端到端 campaign 抓到、单步逐项比对抓不到：批量化时漏了参考对
`target_heading - style.global_headings[...]` 的 `wrap()`；`wrap` 对已在 `[-pi,pi]` 的**值**是恒等，
但在 float32 里不是，只有 `walk_stealth` 这种朝向靠近边界的 style 才暴露。这就是门禁必须是
"端到端 token"而不是"单步数组相等"的原因。

**⑦ 测量环境**：机器仍不空闲，load average **9.8–17.4**（桌面进程，本次运行期间没有训练/评测进程）。
所有前后对比都是同进程交替配对；绝对 p50 仍被抬高约 10 %。

**⑧ 一次事故与恢复（记在案）**：重新生成文档时我跑了一次
`mlx_plan.py --device gpu --repeat 2`（**没给 `--out`**），把 gitignored 的 `out/plan-mlx.json`
覆盖成了 GPU 文档。已从四份完全相同的已验收路径文档恢复，sha256 `262e77d4…`，
并用"优化前路径重放"验证过。以后一律显式传 `--out`。

**红线遵守**：只改 `data/**`、`models/{nn,root,pose,decoder,transition,bundle,sampling}.py`、
`scripts/mlx_bench.py`（新增 `--paired`，默认行为不变）、`tests/test_mlx_*.py`、
`docs/optimization-report.md`、本文件；`training/**`、`train_*.py`、`eval/**`、`eval_*.py`、
`motion-representation.md`、`README.md`、`NOTICE`、`LICENSE`、`pyproject.toml` 未动；
`mlx_plan.py`/`mlx_align.py` 语义未改；参考仓库只读；无 push。
