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
