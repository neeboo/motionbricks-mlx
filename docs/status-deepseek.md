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

**⑨ 翻转的唯一实测代价（CPU 后端内存）**：decision 0004 的 f32 组默认开启后，round 1 记录的
"`mx.compile` 只在 CPU 后端抬内存"那条副作用就成了默认行为。CPU、25 步、load ~13：

| 配置 | p50 | MLX 峰值 | 进程 RSS |
|---|---|---|---|
| `MOTIONBRICKS_MLX_OPT=0`（decision 0004 之前） | 199.0 ms | 748 MB | 853 MB |
| 默认（全开） | 178.8 ms | **1481 MB** | **1615 MB** |
| 默认但关掉三个 `*_compile` | 173.8 ms | **748 MB** | 854 MB |

即 **CPU 后端上编译开关大约把峰值内存翻倍而没换来延迟**（不含它们的配置反而略快 173.8 vs 178.8 ms；
不过这是单臂、负载在漂，诚实的说法是"延迟差在漂移之内，而 2× 内存不是"）。**我没有擅自改**——
decision 0004 采用整组 f32 优化，我不从已裁定默认里偷偷摘开关；已写进报告 13.1 与"下一步"第 1 条：
建议把三个 `*_compile` 的默认值改成随后端而定（非 Metal 关闭），复现环境串是
`MOTIONBRICKS_MLX_OPT=0,pose_fast=1,root_fast=1,root_dedup=1,decoder_fast=1,transition_fast=1,sampling_vector=1`。

**⑩ 默认路径 vs 最初验收基线**：GPU 单步 p50 **55.5 → 27.3 ms（2.03×）**，p90 67.3 → 38.7 ms，
冷启动 385.3 → 336 ms，GPU 峰值内存不变（779 MB），token 与参考仍是 0 差，对齐最差余量仍是 0.26× 容差。

---

## 2026-10-12 · Round 3（A 侧：decision 0005 + 批量烘焙 + host 同步）

**判据先写、事后不改**：`docs/optimization-report.md` 第 21 节在动代码前写死并单独提交
（commit `cecf88a`）：正确性门＝`mlx_align` plan/trace OK ＋ N≥5 seed×6 步 `pose.tokens`/
`sampling_uniforms` 逐位一致＋最终动作偏差必报；采用规则＝**token 一致 ＋ 对齐不回归 ＋ 配对 p50 ≥1.05×**
才默认开；测量纪律＝记录 load、同进程交错配对、每项 ≥25 步。

**① decision 0005 已实现（`models/nn.py`）**：`declared_default()` 让三个 `*_compile`
按后端决定默认——Metal 开、非 Metal 关（`_BACKEND_CONDITIONAL`），环境变量仍可显式覆盖两端。
验收：GPU 默认文档与改前**逐字节一致**（sha256 `ccda1952…`）；`mlx_align plan/trace` 两后端 OK 且
`pose.tokens`/`sampling_uniforms` 均 `0.000e+00`；`--repeat 2` 两后端 deterministic；
5 seed×6 步 token 一致；`pytest` **135 → 140 passed / 1 skipped**。

**② 但 0005 的前提数字不成立，必须记在案**：CPU-only 进程里默认路径的 MLX 峰值是
**748 MB**（改前改后都是），不是决定里引用的 1481 MB；1481 MB 只在**同进程先跑 GPU 再跑 CPU**
时出现（改后仍是 1480 MB），控制实验：`OPT=0` 时 CPU 行 748 MB；只开编译开关（快路径关、编译根本
不执行）748 MB；全快路径关编译 748 MB；**反向 `--devices cpu,gpu` 时被抬高的是 GPU 行（1512 MB）**。
即"CPU 上编译翻倍内存"是**多后端 harness 的产物**（第二个后端的峰值计数含第一个后端留下的东西），
触发条件是**第一个后端真的建过 `mx.compile` 变体**。实现按裁定照做（且仍站得住：CPU 上编译无延迟
收益、冷启动无收益），但报告 22.2 不重复那个 2× 结论，并写清"跨后端那一行不能当作该后端的成本"。

**③ CPU 延迟（交错 3 臂，同进程，25 步，load 9.2–9.3）**：clean env（新默认）193.6 ms、
`OPT=1`（翻转态）186.1 ms、`OPT=0` 190.8 ms；配对 p50 = **1.0124×** 与 **0.9900×**，
都在 IQR 内 → "不劣于 `OPT=0`（漂移内）"成立；同进程 token 25/25 一致、0/1832 不同。

**④ 批量烘焙（`models/bake.py` + `mlx_bench.py --bake`）**：只批 pose 阶段（round 1 认定唯一可批的
阶段），其余阶段跑已验收代码；两臂唯一差别是用哪次前向产生 pose logits（新增 `pose_logits=` 缝，
默认 `None` ＝逐字节原路径）。B=1/2/4/8/16，GPU、5 次中位数：加速 **0.950× / 1.162× / 0.961× /
1.180× / 1.233×**，每场景 29.25→21.62 ms；**同 duration 批次**（同一 style 不同 seed）到
**1.185× / 1.377× / 1.336×**——**按 duration 分组比加大 B 更值钱**（B=16 时 1.336× vs 1.233×）。
阶段层面 B=4/8/16 的 pose 前向是 **2.50× / 5.04× / 7.18×**（round 1 是 3.12×/6.05×），
但单次串行 pose 只占一步 6.46/27.79 ms ≈ 23 %，**端到端上限 ~1.25×**；round 1 的 2.4–6.1×
是阶段吞吐，不是烘焙吞吐，报告 23.4 写清了。

**⑤ token 战役**：**31/31 场景 `pose.tokens` 逐位一致、`sampling_uniforms` 逐位一致、duration 一致、
最终动作逐位相同（roots/rotations 差恰为 0）**，0/2136 个 token 决策不同（含混合与同 duration 两种）。
可解释且量化：`mx.vmap` 把 matmul 降成批量原语，**B=1 就差 1.98e-04**，B=16 差 3.357e-04；
而战役中最紧的决策余量（Gumbel 加噪后 top1−top2）是 2.985e-03 → **扰动/最紧余量 = 0.112×**。
注意：**批量不逐位**（是数值改动，不是等价的），所以它的正确性只能靠战役，不能靠相等。
结论 + 规则写在 `docs/decisions/0006-batch-bake-safe-b.md`（**提案，等人批**）：可复现档＝跑过战役且
逐场景一致的最大 B（本版本证据 `B ≤ 16`）；更大的 B 或没重跑战役的 B 只能作草稿档，且
**manifest 必须记 `batch_size` 与档位**。

**⑥ host 同步削减（逐项试，全部默认关）**：默认每步 6 次 `np.asarray` 设备读（其中 5 次是真等待）。
① `decoder_defer`（解码器输出不回host）6→5；② `pred_local_defer`（预测 local root 留在设备，
`arctan2` 与 NumPy 差 1 ulp，是风险点）6→5；③ `root_boundary_reuse`——发现 `duration_logits` 与
`plan` **各重建一次 boundary embedding**，孤立实测 **0.683 ms p50**（探针阶段的 25 %），纯重复工作，
去掉后逐字节一致。正确性门：三臂各自与合起来都是 **30/30 token 一致、uniforms 一致、最终动作逐位相同**，
`mlx_align` 不回归，全开时 GPU 文档仍逐字节一致。速度（交错配对 p50）：单项 **1.0035× / 1.0207× /
1.0443×**，三项合起来 6 个 50 步样本 = 1.0579/1.0554/1.0411/1.0389/1.0339/1.0333 →
**中位数 1.0400×**，**都没过 1.05×** → **三项全部默认关闭**并写明（前两个样本单独看会误判为通过，
这正是"先钉死门槛"的意义）。即：那 4–5 次同步合计只值**约 1–4 %**，第 19 节的"kernel drain"假设
不成立，真正的成本是**互相依赖的 kernel 链**。

**⑦ 测量环境**：load 全程 **9.2–23.5**（1 分钟，桌面共享），每次测量都记录了 load；所有前后对比都是
同进程交错/轮换配对，每项 ≥25 步，报告 p50/p90/max。绝对值仍被抬高约 10 %。

**⑧ 一个必须记下的自身事故（方法论）**：我的 `/tmp` 门禁脚本一开始把 `--arm "label:MOTIONBRICKS_MLX_OPT=..."`
的整串当成了**选项值**，于是"单开关臂"其实跑的是默认路径——第一版 token 战役因此无效。
发现方式是给脚本加"打印解析后的开关值"，并写了一个统计 `np.asarray` 调用点的计数器（它同时暴露了
`pred_local_defer` 第一版仍在读回 root 输出、实际一次同步都没去掉）。修正后重跑了全部战役与计时。
教训：**每条臂都要有"这条臂真的生效了"的独立证据**，否则"一致"可能只是因为两边是同一个配置。

**红线遵守**：只改 `src/motionbricks_mlx/{data,models}/**`（`nn/root/pose/transition/agent` 与新增
`models/bake.py`）、`scripts/mlx_bench.py`（追加 `--bake*`，既有语义不变）、`tests/test_mlx_*.py`、
`docs/optimization-report.md`（追加）、本文件、新增 `docs/decisions/0006-...md`（**提案**）；
`training/**`、`train_*.py`、`eval/**`、`eval_*.py`、`motion-representation.md`、`README.md`、
`NOTICE`、`LICENSE`、`pyproject.toml` 未动；`eval/` 未跟踪也未提交；`mlx_plan.py`/`mlx_align.py`
语义未改；所有 `mlx_plan.py` 调用都显式带 `--out`；参考仓库只读；**无 push**。

---

## 2026-10-12 · Round 4（同一天）：批 root 与 decoder 阶段，端到端烘焙 1.25× → 2.8×

**任务**：round 3 证明只批 pose（占每步 ~23 %）时端到端上限约 1.25×；本轮的下一档是批 root
（512 维 transformer 规划 + 残差扩张卷积解码）与 decoder（两级 VQ 解码，latent 256）。

**先钉判据再动手**：`docs/optimization-report.md` 第 28 节（commit `94beb41`，本轮第一个 commit，
事后未改）。两个新开关 `bake_root_batch` / `bake_decoder_batch` 声明**默认关闭**。

**做法**：round 3 的单参数 seam 只够替换一个前向。三个阶段是一条经 host glue 串联的依赖链
（pose 的输入是批量 root 的输出，decoder 的输入是批量 pose 的输出），所以新增 **stage driver**：
每个场景用 `bake=driver` 跑 `run_transition`，某一阶段的行还没算好时 driver 记下该阶段输入并抛
`BakeSuspend` 提前结束；同一 stage 边界上所有场景都到齐后，**一次** vmap 前向算出所有行，再继续。
代价是部分 pass 会重跑 host glue——第 33 节量化了这笔代价（3.73 ms/场景，占批量时间 38 %）。

**结果**

- **端到端（B=16）**：混合 2.244×、同 duration **2.812×**（87.24 场景/秒）；串行 515.70 ms → 批量 183.41 ms。
  阶段拆分：pose-only 1.499× → +root 2.133× → +decoder 2.812×。阶段层 B=16 同 duration：
  root 10.82×、pose 7.61×、decoder 9.67×。**B=1 比串行慢**（0.895–1.008×），不该用于烘焙。
- **战役**：59/59 场景 token / uniform / duration **逐位一致，0/4600 个决策不同**；
  但**最终动作不再逐位**（roots ≤ 2.027e-06 m、rotations ≤ 2.753e-05，分别是 `mlx_align` 容差的
  2.0e-03× 与 1.4e-02×）。round 3 的"恰好 0"是只批 pose 的副产品。
- **安全裕度（本轮最重要的数字）**：对照臂用本轮代码复算 round 3，得到 `3.357e-04 / 2.985e-03
  = 0.1124×`，与 round 3 报告的 `3.357e-04 / 0.112×` **完全一致**，证明两轮同一把尺子。
  全阶段后：同 duration B=16 = **0.4089×**，混合 B=16 = **1.0426×（无余量）**。
  即批量 root 的输出经 predicted local root / pose condition 进入 pose 规划输入，把扰动放大
  **3.6×（同 duration）到 9.3×（混合）**；余量本身没变。**混合 B=16 通过了战役却没有裕度**——
  这是 0006 第 5 条警告的数据相关性，也是本轮提案 0007 收紧档位定义的理由（发布档 B ≤ 8）。
- **默认路径**：两个 seam 默认 `None`。GPU 文档与 `out/plan-mlx-gpu.json`（`ccda1952…`）、
  CPU 文档与 `out/plan-mlx.json`（`262e77d4…`）**逐字节一致**；`mlx_align plan/trace` OK
  （tokens / uniforms / quantized 全 0）；5 seed × 6 步 token 门通过；`pytest` **199 passed / 1 skipped**
  （本轮起点 192/1）；C 线的 `scripts/eval_gate.py --no-collect` 仍然 **GATE PASS**（只读调用，未改未提交）。
- **下一步该攻哪里（有测量支撑）**：B=16 同 duration 每场景 9.92 ms = 批量阶段 2.58 + host pass 3.62
  + driver 重跑 3.73。① 把 driver 换成单遍编排（把 glue 抽成可调用步骤）→ 约 28.37/6.19 = **4.6×**，
  不用碰批量前向；② 地板是 host 侧 3.62 ms/场景（`decode_motion` + agent 数学 + 归一化），
  无穷快的前向也只能到约 7.8×。**阶段已经不是瓶颈了**。

**红线遵守**：只改 `src/motionbricks_mlx/models/**`（`bake/root/decoder/nn/transition`）、
`src/motionbricks_mlx/data/agent.py`（抽出 `constraints_for`）、`scripts/mlx_bench.py`（追加
`--bake-stages` / `--bake-metrics`，既有语义不变）、`tests/test_mlx_{bake,optimizations}.py`、
`docs/optimization-report.md`（追加第 28–34 节）、本文件、新增 `docs/decisions/0007-...md`（**提案**）；
`training/**`、`train_*.py`、`eval/**`、`scripts/eval_*.py`、`tests/test_eval_*.py`、
`motion-representation.md`、`README.md`、`NOTICE`、`LICENSE`、`pyproject.toml` 未动；
C 线脚本只读调用；`mlx_plan.py`/`mlx_align.py` 语义未改；所有 `mlx_plan.py` 调用都显式带 `--out`；
参考仓库只读；**无 push**。

---

# Round 5（A 侧）：单趟编排器（single-pass orchestrator）

**先钉判据再动手**：`docs/optimization-report.md` 第 35 节（commit `7495db2`，本轮第一个 commit，
事后未改；`git diff --numstat 7495db2 HEAD -- docs/optimization-report.md` 只见 `+`）。

**做法**：round 4 的 stage driver 靠**重启** `run_transition` 在阶段边界同步批次，每次重启都重跑一遍
host glue（round 4 量化成 3.73 ms/场景，占批量时间 38 %）。本轮把 transition 变成**可恢复**的：

- `MlxPipeline.transition_steps` = 已验收函数体变成**步骤生成器**，四个前向位置改成
  `yield (stage, args, serial_call)`；归一化 / dense target / predicted local root / pose condition /
  sampling / codebook gather / motion decode 全部是生成器的局部变量，跨 yield 存活——**一行数学都没挪**。
- `run_transition` 退化成它的平凡驱动（14 行）+ `_stage_forward`，后者保持原有优先级
  （bake → `pose_logits` seam → 已验收前向）。
- `Agent.finish_plan` = `Agent.plan` 的后半段，编排器复用同一份 agent 数学。
- `bake.drive_batch_single_pass`：每场景一个 `_SinglePassSession`，全部推进到本编排器拥有的阶段后，
  **按 round 4 同样的 shape 分组**一次批量前向，回填，继续；每个 host 步骤每场景只跑一次，不重启。
  `inject=` 用已录好的行重放一整趟，从而把 host 侧单独计时。
- **`bake.drive_batch` / `StageDriver` 一行未动**——它是本轮对照臂；`compare_bitwise` 先钉住两臂
  逐位相同，才允许引用加速比。

**结果**

- **端到端三臂（同进程、轮换顺序、9 次取中位数，load 见报告 40.4）**：
  * 同 duration B=16：串行 504.66 ms → round 4 **3.140×** → 单趟 **3.720×**（单趟/round4 = 1.185×，
    吞吐 117.95 场景/秒）。另三次运行（5/9/15 次重复、三种 load）给 3.485× / 3.687× / 3.572×，
    round 4 臂 2.873× / — / 3.022×。**结论稳定：把 B=16 同构烘焙从 2.87–3.14× 抬到 3.49–3.72×。**
  * 混合 B=16：串行 475.82 → round 4 2.309× → 单趟 **2.608×**（1.129×）。
  * B=1 仍慢于串行（0.93–1.05×），0007 的"B=1 永不使用"不变。
- **两臂逐位相同**（新判据）：每个 B、两种分组下 `tokens / uniforms / duration / roots / rotations`
  全部 `16/16`，`bit_identical=true`。加速不是因为算了两件不同的事。
- **战役（本轮重跑）**：62 个场景，**0/4864 个 token 决策不同**，duration 全部重选一致；
  最终动作仍**不逐位**（roots ≤ 1.140e-06 m、rotations ≤ 2.753e-05 = 容差的 2.0e-03× / 1.4e-02×）。
  裕度与 round 4 **逐位复现**：混合 0.2019 / 0.4843 / 0.6849 / 0.6695 / 1.0426，同构 0.6031 / 0.5290 / 0.4089，
  最紧决策余量 2.985e-03。
- **本轮档位**：campaign 每个 B 都通过，所以档位完全由裕度决定。**结论与 0007 一致：release B ≤ 8、
  draft B ≥ 16、B=1 永不使用。** 但必须上报一个判据细节：**裕度对 B 非单调**——混合 B=4（0.6849×，1.46×）
  比混合 B=8（0.6695×，1.49×）更差，两者都略低于字面上的 1/ratio ≥ 1.5。按字面算术，混合分组最大合格 B 是 2；
  0007 批准文本把 0.6695 记作"1.5×"并据此给 B ≤ 8，本轮沿用该读法并把边界留给 lead 裁定。
  0007 的 12 项 manifest 字段已落地为代码（`bake.manifest_fields` / `bake.bake_tier`），逐 B 写进 bench JSON，
  含如实记录的 `bake_action_bit_identical=false`。
- **数字纠错**：round 4 的"批量阶段 2.58 ms"把 `stage_throughput` 的**派生 root 行**（= probe + plan）
  也加了进去，等于把 duration probe 数了两遍；只算四个前向是 2.02–2.80 ms（B=16）。本轮按修正后的口径算，
  并把 round 4 口径并列在 JSON 里。
- **4.6× 的预测没有兑现，原因在预测本身**：真实单趟是 8.24–8.48 ms/场景（= 前向 1.98 + host pass 3.97
  + 2.29 未归因交互项），而 round 4 用两个各自乐观的仪器相减得到 6.19 ms。**直接配对测量才是诚实值**：
  两臂同进程跑同样的行，round 4 臂减单趟臂 = **1.50–1.69 ms/场景**（B=16 同构），端到端 1.15–1.19×。
  编排器值得做（零语义变化的 1.2×），但"B=16 4.6×"从来不可达。
- **默认路径**：GPU 文档 `ccda1952…` == `out/plan-mlx-gpu.json`、CPU 文档 `262e77d4…` == `out/plan-mlx.json`，
  逐字节一致；`mlx_align plan/trace` OK（tokens / uniforms / quantized 全 0）；**5 seed × 6 步前后对照**
  （`0e247ea` 的 worktree vs 本轮工作树）：tokens 30/30、uniforms 30/30、duration 30/30 逐位一致，
  0/2416 决策不同，最终动作偏差 **恰好 0.000e+00**；`pytest` **204 passed / 1 skipped**（起点 199/1，只增不减）。
- **下一步（量化上限）**：B=16 同构单趟 8.24 ms/场景 = 前向 1.98（24 %）+ **host pass 3.97（48 %）**
  + 未归因 2.29（28 %）。① 先把 2.29 ms 的未归因项拆开（单趟里第二大成分，且它同时也是 round 4 臂
  3.95–4.11 残差的主要部分）——在归因完成前不该优化它；② 再动 host pass：减半 → 4.7×，归零 → **7.4× 上限**。
  本轮**没有**顺手改宿主数学。

**红线遵守**：只改 `src/motionbricks_mlx/models/**`（`transition.py` 抽生成器、`bake.py` 追加编排器与
manifest 助手）、`src/motionbricks_mlx/data/agent.py`（抽出 `finish_plan`）、`scripts/mlx_bench.py`
（追加 `--bake-single-pass`，既有语义不变）、`tests/test_mlx_bake.py`（+5 个测试）、
`docs/optimization-report.md`（追加第 35–42 节）、本文件；`training/**`、`train_*.py`、`eval/**`、
`scripts/eval_*.py`、`tests/test_eval_*.py`、`motion-representation.md`、`README.md`、`NOTICE`、
`LICENSE`、`pyproject.toml` 未动；参考仓库只读；`mlx_plan.py`/`mlx_align.py` 语义未改；
所有 `mlx_plan.py` 调用都显式带 `--out`；**无 push**。

---

# Round 6（A 侧）：把 2.29 ms 归因掉，然后批量化解码 —— 收尾基线

**先钉判据再动手**：`docs/optimization-report.md` 第 43 节（commit `428af08`，本轮第一个 commit，
事后未改；`git diff --numstat 428af08 -- docs/optimization-report.md` 只见 `+`）。

**① 任务一：直接仪器，先归因、不许相减。** 第 5 轮的 2.29 与 round-4 的 3.95 都是两个仪器相减来的，
本轮改成在每个**叶子段**外侧加 `perf_counter` 括号（`bake.Profile`）：编排器的 setup/schedule/stack/
forward/eval/scatter/collect ＋ 生成器里的 10 个 `glue_*` 宿主步骤，**括号不嵌套、不跨 `yield`**，
各段之和与实际区间闭合。第一版闭合到 3.83 %，两个错误都是闭合数抓出来的：
① `setup` 把 `next(self.steps)` 也括了进去，而生成器首段胶水自己也有括号 → 重复计数（改为 `start()`
单独 prime）；② 探针自身的 drain（`np.asarray(mx.duration_logits)`）落在 resume 之后、括号之外 →
把最大的一处漏账（0.38 ms/场景 × 16）摆在了括号外。修好后闭合 **0.92–0.94 %**。
仪器是**惰性**的：每次 profile 都跑一遍 profiled vs unprofiled 的 `compare_bitwise`（token/uniforms/
duration/roots/rotations，无容差），全部 `inert=True`；交错测量仪器自身代价 −1.8 %～+2 %。

**② 归因结果（B=16 同构，9–15 重复，load 11.4–14.8，7.948 ms/场景）**：`eval` 3.4858（43.9 %）、
**`glue_decode` 2.0282（25.5 %）**、`setup` 0.7823（9.8 %）、`collect` 0.3200（4.0 %）、
其余宿主胶水 1.331、编排+resume 0.073（0.9 %）。
**2.29 的来历**：前向仪器乐观 +1.33（`stage_throughput` 在紧循环里单测前向，而现场的 `mx.eval`
还折进了生成器为它建的设备图）、注入式宿主仪器悲观 −0.25、另有 **+0.78（`constraints_for`）
与 +0.32（`finish_plan`）是两件仪器都没数过的项** —— 合计 +2.18，对上报告的 2.29。

**③ 任务二：按判据改。** 归因显示第二大项不是宿主算术，而是**每场景 drain 一次的解码小图**：
建图 0.397 + drain 1.451 ms，而裸 `mx.eval(sum)` 只要 0.28–0.35 ms。于是新增**默认关闭**的
`bake_decode_batch`：把解码变成 step generator 的**第五个前向点**（`decode_motion_batch`，批量孪生体，
帧轴在轴 1、每场景各自的 `(batch,1)` 原点），由单趟编排器按形状分组、每组一次图。
**它不能被 round-4 臂拥有**（`batched_stages()` 不含它）→ round-3/4 路径与默认单场景路径一字未动。
配对前后（每 B 一个进程，25 重复，顺序轮换）：**B=4 1.0559× [0.935–1.207]、B=8 1.1726× [1.007–1.326]、
B=16 1.2207× [1.132–1.335]**，B=1 0.9925×（中性，与"B=1 永不使用"一致）；两臂 `compare_bitwise`
**逐位相同**。端到端：同构 B=16 **3.49–3.72× → 4.43–4.44×**（144–153 场景/秒），B=8 3.17–3.39×，
混合 B=16 2.81–2.95×。**方法学发现**：把 B=1/2/4/8/16 放同一个进程配对时 B=8 量出 0.936×，
独立进程重测是 1.1726× —— 长命进程跨 B 积累状态，配对必须每 B 一个进程。

**④ 档位（重跑战役后）**：裕度与第 5 轮**逐位相同**（批量解码不改数值）：混合 0.2019/0.4843/0.6849/
0.6695/1.0426，同构 0.2019/0.2441/0.6031/0.5290/0.4089，最紧余量 2.985e-03。
**结论不变：release = 混合 B ≤ 8 / 时长同构 B ≤ 16；draft = 混合 B ≥ 16；B=1 永不使用。**
同时把 `bake_tier` 从 0007 的字面 `1/ratio ≥ 1.5` 改成 **0008 已裁定的机械规则**
（`RELEASE_RATIO=0.70`、`GROUPED_ABOVE_B=8`、分组腿）——差别恰好落在边界上：混合 B=8 的 0.6695
（1.494×）在 0007 算术下是 draft、在 0008 下是 release。manifest 记录 `bake_decode_batch`。

**⑤ 默认路径与门禁**：GPU 文档 `ccda1952…` == `out/plan-mlx-gpu.json`、CPU 文档 `262e77d4…` ==
`out/plan-mlx.json`，逐字节一致；`mlx_align plan` `ALIGNMENT OK`（最差余量 rotation angle 0.26× 容差）、
`trace` 两后端 `TRACE OK`（tokens/uniforms/quantized 全 `0.000e+00`）；**5 seed × 6 步**跨版对照
（`f699b84` 工作树 vs 本树）：tokens/uniforms/duration **30/30**、**0/2416** 决策不同、
最终动作偏差**恰好 0.000e+00**；`pytest` **209 passed / 1 skipped**（起点 204/1，只增不减）；
默认路径 GPU p50 **26.6 ms**、CPU **173.1 ms**（冷启动 317/497 ms，MLX 峰值 779/748 MB，分开进程量）。
**成品是 `docs/SUMMARY.md`**（收尾基线全文），提案 `docs/decisions/0009-batch-the-motion-decode.md`。

**红线遵守**：只改 `src/motionbricks_mlx/models/{bake,transition,motion_rep,nn}.py`、
`scripts/mlx_bench.py`（追加 `--bake-profile` / `--bake-decode-paired` / `--bake-stages all+decode`，
既有语义不变）、`tests/test_mlx_{bake,optimizations}.py`、`docs/optimization-report.md`（追加第 43–49 节）、
本文件、新增 `docs/SUMMARY.md` 与 `docs/decisions/0009-...md`；`training/**`、`train_*.py`、`eval/**`、
`scripts/eval_*.py`、`tests/test_eval_*.py`、`motion-representation.md`、`README.md`、`NOTICE`、
`LICENSE`、`pyproject.toml` 未动；参考仓库只读；`mlx_plan.py`/`mlx_align.py` 语义未改；
所有 `mlx_plan.py` 调用都显式带 `--out`；**无 push**。

---

# Round 7（A 侧）：把"数字漂亮"变成"看得见的成品 + 有据可查的 manifest"

**这一轮不做新优化**：判据、开关、默认路径一行不改（唯一 src 改动是
`BatchResult.motions` 追加一个默认空字段）。本轮把前六轮的烘焙**做成成品**并第一次在
**风格多样的库**上重跑战役。

**① 三件事**：`scripts/mlx_bake_library.py`（新）＝ `batch`（release 配置 `--bake-stages all+decode`
＋ 12 项 manifest）／`single`（**另一个进程**、零 bake 开关的 B=1 语义）／`compare`（逐场景等价性）／
`check`（对**写盘的** manifest 逐字段核对）／`metrics`（用 C 线阈值评本轮产物）／`pairdoc`
（同场景两条臂拼成一份渲染文档）。`tools/side_by_side_gif.py`（新）只做拼接，骨架像素全部来自
参考侧 `~/dev/motion-bricks-cpp/tools/render_motion.py`（未改一行；本仓库 `.venv` 无 Pillow，
所以用系统 Homebrew python 3.14 + Pillow 12.2.0，没往 `.venv` 装包、没动 `pyproject.toml`）。

**② 实测结果（每个 B 单独进程，load 6.9–19.6，全部 `--strict` 通过）**：

| campaign | B | token 分组（实测） | tier | 比值 |
|---|---|---|---|---|
| diverse（8 真实风格 / 14 seed / 24 帧同构） | 8 / 16 | `{6:8}` / `{6:16}` | **draft** | **1.4402 / 1.2716** |
| single-style（`walk` / seed 1..16 / 44 帧同构） | 8 / 16 | `{11:8}` / `{11:16}` | reproducible | **0.5290 / 0.4089** |
| idle-style（`idle` / seed 1..16 / 24 帧同构） | 16 | `{6:16}` | reproducible | **0.2137** |

**新发现（本轮最重要的数）**：**同一个脚本、同一个 release 配置、同样时长同构，库里一旦有多风格，
冻结的 0008 全局规则就把档位从 release 打到 draft。** 归因：全局比值的分子与分母来自**不同风格**——
最大 pose 扰动由 `idle` 供（它单独烘时 7.7e-03，是 `walk` 的 6 倍），最紧余量由 `walk_zombie` 供
（2.99e-03，比 `idle` 的 3.6e-02 紧 12 倍）；**逐场景自己的比值最大只有 0.142**。
`single-style` 的 0.5290 / 0.4089 与最大扰动 1.2207e-03、`bake_worst_rotations` 2.753e-05
与第 5/6 轮**逐位复现**，说明两把尺子是同一把。**没有改判据、没有挑场景去把 draft 抬回 release。**

**③ 12 项 manifest**：三份 manifest、五个 group 全部 **12/12**（对写盘文件核对，`manifest check: PASS`），
`bake_action_bit_identical` 如实 `false`。

**④ 等价性**：5 个 group 的全部 **64 个场景** tokens / uniforms / duration **逐位一致**，
**4032 个 token 决策里 0 个不同**；
最终动作**不逐位**（0/84），最差 roots **1.140e-06 m**、rotations **2.753e-05**、角 0.0539°
（旋转角列的地板）；`world_positions_xyz` 最差 1.798e-05 m。

**⑤ 质量门**：`scripts/eval_gate.py --no-collect` 原样调用 → 10 组全 PASS + cross-impl 2 对 PASS +
5 个负控全触发 → **GATE PASS**（exit 0）。另外把本轮 64 份产物喂给同一套 C 线阈值：
diverse 14/16、single-style 16/16，**批量臂与单场景臂判定逐场景相同**，336 个数值最大差 5.6e-03
（红的两份是 `walk_happy_dance`，两条臂一样红——舞蹈动作 vs 行走阈值）。

**⑥ 看得见的成品**：4 个场景 × 两条臂的 GIF（idle / walk_zombie / stealth_walk / walk），
外加 4 个并排对照。**两条臂的 GIF 逐字节相同**（0/38,574,400 个像素不同），但两条臂的 roots
确实不同（最差 4.0e-07 m，亚像素）。

**⑦ 门禁**：GPU `ccda1952…` / CPU `262e77d4…` 改动前后各验一次，**四次都逐字节一致**；
`pytest`（HEAD 已跟踪的测试 + 本轮新文件）**214 passed / 1 skipped**（起点 209/1，只增）；
全量 `pytest` 是 240 passed / 1 skipped / **1 failed**，失败项来自**同工作区另一个 agent 正在做的
viewer**（`scripts/mlx_viewer.py`、`viewer/`、`tests/test_viewer_*.py`，HEAD 未跟踪），我没碰。
成品是 `docs/BATCH-BAKE-RUN.md`，产物在 `artifacts/bake-2026-10-12/`（含"哪些进 git"的说明）。

**红线遵守**：只新增 `scripts/mlx_bake_library.py`、`tools/side_by_side_gif.py`、
`tests/test_mlx_bake_export.py`、`docs/BATCH-BAKE-RUN.md`、`artifacts/bake-2026-10-12/**`，
只改 `src/motionbricks_mlx/models/bake.py`（追加）与本文件；`training/**`、`train_*.py`、`eval/**`、
`scripts/eval_*.py`、`tests/test_eval_*.py`、`motion-representation.md`、`README.md`、`NOTICE`、
`LICENSE`、`pyproject.toml` 未动；`mlx_plan.py` / `mlx_align.py` / `mlx_bench.py` 语义未改，
所有 `mlx_plan.py` 调用显式带 `--out`；参考仓库全程只读（`find -newermt` 证明零修改）；**无 push**。

# Round 8（A 侧）：给 MLX 版做一个"拉起来就能看"的页面

**任务**：用户要的是"像那个 cpp 版本一样，有个网页拉起来让我看效果"——不是跑分表。
所以本轮唯一的成品是**一个只绑 `127.0.0.1` 的本地页面**，上面每一段动作都由本机的
MLX 现场规划。长版说明在 [VIEWER.md](VIEWER.md)。

**① 页面**：`http://127.0.0.1:8848/`（PID `78447`，日志 `artifacts/viewer-run/server.log`，
PID 也写在 `artifacts/viewer-run/server.pid`；`kill "$(cat artifacts/viewer-run/server.pid)"` 停）。
受管后台任务起的，**没有 `nohup` 乱扔进程**。

**② 起法**：`./.venv/bin/python scripts/mlx_viewer.py --port 8848 --device gpu`。
只加 stdlib（`http.server`），**`pyproject.toml` 未动、没有新依赖**。

**③ `curl` 证据**（全部实测）：首页 `HTTP 200 / 4231 B / 2.9 ms`；
`style.css` `200 / 8520 B`、`app.js` `200 / 23985 B`、`rig.js` `200 / 9409 B`、
`vendor/three.module.min.js` `200 / 338914 B`、`vendor/three.core.min.js` `200 / 381130 B`、
`vendor/THREE-LICENSE.txt` `200 / 1102 B`；一次真实交互
`GET /api/plan?style=walk&seed=1&move=0,1&facing=0,1&reset=1`
→ `HTTP 200 / 96672 B / 59.3 ms`，返回 44 帧 × 34 关节的 `world_positions_xyz`
（4488 个数，长度自洽）+ 4 帧目标关键帧。换 4 个真实 style 各触发一次（都 `reset=1`，
即"新建 agent + 重播种 + 规划"）：

| style | frames | plan_ms | travel |
|---|---|---|---|
| `walk` | 44 | 27.8 | 1.097 m |
| `walk_zombie` | 44 | 26.1 | 0.710 m |
| `idle` | 24 | 39.3 | 0.007 m |
| `stealth_walk` | 32 | 101.5 | 1.028 m |

（`stealth_walk` 那 101.5 ms 是换 style 后第一拍的代价：新 agent + 新上下文 + 一次冷图，
不是稳态；稳态是下面那张表里的 28.3 ms。`idle` 几乎不位移是对的。）

**④ 交互清单**：15 个真实 `.mbstyle` 任选；Play/Pause；Reset camera；Ground trail；
Placed target keyframes；Replan while walking；W/A/S/D 盘（再点同键即停）+
`↺ Turn` / `Turn ↻` / `Stop`；seed 输入框 + `Next seed`；
`?style=&seed=&move=` 地址栏预置。换 style / 换 seed / 走停 / 转身**四种**都会立刻触发一次新规划。

**⑤ 性能实测**（load 8.1–18.0，含本轮的 Chrome）：

| 指标 | 实测 |
|---|---|
| 冷启动（载入 + 预热） | **370.7 ms** |
| 首帧（页面 start → 第一帧上屏） | **84.4 / 117.3 / 129.9 ms**（三次不同会话） |
| 页面可用（`ready_ms`） | **17.0 / 39.6 / 42.1 ms** |
| 连续步频（服务端规划 p50） | **28.3 ms → 27.9 clips/s**（30 clip 串行） |
| 其中 FK p50 | **2.5 ms** |
| HTTP 往返 p50（含 ~100 KB JSON） | **34.8 ms** |
| 等效帧吞吐 / 相对 30 fps 余量 | **1227 帧/秒 / 40.9×** |

参考版 GGML 的 **≈380 ms p50（M1 Max）** 是**任务书给的值**，我在只读 checkout 里
**没找到出处**（`docs/*.md` 无 p50 表），所以按引用处理；同口径可复现的对照是本仓库
自己的 **26.6 ms**（[SUMMARY.md](SUMMARY.md) 第 1 节），与本页服务端 p50 一致。

**⑥ 复用与归属**：参考仓库**全程只读、零修改**。`vendor/three.*` 与 `style.css` 前 63 行
**逐字节复制**（sha256 与参考版相同：three.module `852c06b9…d2af`、three.core
`4183cee0…ba82`、style.css 前缀 `bf95d13b…1e9a`）；`viewer/rig.js` 的
`app.js:88-95 / 97-113 / 115-252 / 254-272` 四段**逐字复制**（209 行里只有 32 行是新增的
来源注释 + `bindScene` + `export`），唯一机械改动是把模块级 `scene` 单例改成注入。
`index.html` / `app.js` / 服务器是**本仓库新写**。接缝之所以免费：参考版
`SkeletonRig.poseWorld()` 吃的就是本仓库导出器已经在产的展平 `world_positions_xyz`。

**⑦ 画面**：浏览器（无头 Chrome，REA `capture_browser_scenario`）**只做 goto + wait**，
没有合成输入、没点界面、没抢焦点、没音频。主图 `artifacts/viewer-shots/viewer-final.png`
（sha256 `f0f706960a1300b6…`，逐张重算校验）里 HUD 显示 `CLIPS 11 / WALKED 10.43 m /
ROOT Z −7.88 m`，W 键高亮，地面轨迹可见——是真的在走。另三张：
`viewer-hud-1-walk.png`、`viewer-walk-1440x900-1-walk.png`、`viewer-2-walk.png`。

**⑧ 服务器里唯一的实现**：`world_positions()` 把 `plan_json.world_positions_xyz` 的双层
Python 循环改成"保留关节序、批量化帧轴"，**逐位相同**（`np.array_equal`，见
`tests/test_viewer_fk.py`），**14.9 ms → 1.34 ms/clip（11×）**。

**⑨ 接缝连贯性**（`artifacts/viewer-run/continuity.txt`）：连续 5 个 clip，
`advance = frames − 11`，clip N+1 第 0 帧与 clip N 第 `advance` 帧的骨盆差
**4.4e-04 – 5.4e-04 m**，而世界 z 在累积 **0.775 → 1.760 → 2.706 → 3.666 m**。

**⑩ 门禁**：GPU `ccda1952…` / CPU `262e77d4…` 用 `mlx_plan.py --out /tmp/...` 各跑一次，
**两个都与 `out/` 逐字节相同**；`pytest -q` **242 passed / 1 skipped**
（起点 209/1；另有 agent 报过 240/1/1 failed，那个 failed 就是我这边**还在写**的
`tests/test_viewer_*.py`，现在是绿的）。load 8.07–18.02 已如实记录。

**踩到的两个坑（都修了）**：① 抄 rig 时漏了 `app.js:91-95` 的向量暂存变量 →
每个 rAF 一次 `ReferenceError: startPoint is not defined`；② `updateControl()` 在
`buildRigs()` 之前把初始 `state.move` 归零 → 角色**原地踏步**（HUD `WALKED 0.86 m`），
改成初始 `padKey:'w'` 后 `WALKED 10.43 m`。第 ② 个正好证明 HUD 必须画进画面里：
数字只在侧栏的话，一张截图看不出"在走但没位移"。

**红线遵守**：只新增 `viewer/**`、`scripts/mlx_viewer.py`、`tests/test_viewer_fk.py`、
`tests/test_viewer_server.py`、`docs/VIEWER.md`、`artifacts/viewer-*/**`，只追加本文件；
`training/**`、`train_*.py`、`eval/**`、`scripts/eval_*.py`、`tests/test_eval_*.py`、
`motion-representation.md`、`README.md`、`NOTICE`、`LICENSE`、`pyproject.toml` 未动；
没有新依赖；参考仓库只读；**无 push**。
