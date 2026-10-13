# motionbrick-mlx

在 Apple Silicon 上用 **MLX** 跑 MotionBricks 推理：默认是**单场景规划**
（GPU p50 **26.6 ms**；GPU 与 CPU 两个后端产出**逐字节相同**的 plan 文档，对参考运行时的逐阶段对齐
`TRACE OK`、离散 token **零差异**），并在这条推理路径上落出一整套可用产物——
**离线批量烘焙**（一次烘一库真动作、导出可播的 plan 文档与动画，单风格时长同构库 release 档端到端 **4.44×**）、
**浏览器查看器**（页面上每一段动作都是这台机器上的 MLX 现场算的，不是预录）、
**目标关键帧**（在页面上摆 1–4 个目标帧，命中误差印在 HUD 上）
与 **A→B→A 动作切换**（条件化当前状态，并给出可证伪的硬切对照）。

**当前交付就到这里为止。** 训练线与"视频 / 第三方动作 → clip"**暂不在此范围**；
B/C 两条线现在是什么状态、为什么搁置、接口冻在哪，见 [§5「当前范围与不做的事」](#5-当前范围与不做的事)。

上游架构来自 NVIDIA MotionBricks（`NVlabs/GR00T-WholeBodyControl` 的 `motionbricks/`）。
本仓库**只走 Apache-2.0 源码路线**：不使用、不转换、不再分发其预训练权重，见 [`NOTICE`](NOTICE)。

> **数字出处约定**：下面每个数字都标注它来自哪份文档的哪一节；这些文档的数字都在**其 HEAD 提交的那棵树**上量过。
> 本机负载会抬高绝对值约 10 %，所以绝对耗时只作佐证，同进程配对比值才是结论（见"诚实边界"）。

---

## 1. 已验收的数字

### 1.1 默认路径（单场景规划）

两个后端**分开进程**跑，因为同进程先 GPU 后 CPU 会把第二个后端的峰值内存算高
（[SUMMARY](docs/SUMMARY.md) §1、[optimization-report](docs/optimization-report.md) §22.2）。
测量条件：load average 6.1–7.7，25 步，4 个真实 style 轮转。

| 后端 | p50 | p90 | max | 冷启动（载入+首步） | MLX 峰值 | 进程 RSS | 确定性 |
|---|---|---|---|---|---|---|---|
| GPU (Metal) | **26.6 ms** | 37.2 | 40.4 | 317 ms | 779 MB | 846 MB | True |
| CPU | **173.1 ms** | 198.0 | 213.3 | 497 ms | 748 MB | 856 MB | True |

出处：[SUMMARY](docs/SUMMARY.md) §1。相对第 1 轮的验收基线 GPU p50 55.5 ms 是 **2.09×**。

### 1.2 对齐、逐位一致与跨版门

| 检查 | 实测 | 出处 |
|---|---|---|
| GPU plan 文档 sha256 | `ccda1952c967c8d7e893a795908f7de278f2acda1ce5ba1e1889692a8904a7c5`，与 `out/plan-mlx-gpu.json` **逐字节相同** | [SUMMARY](docs/SUMMARY.md) §1 |
| CPU plan 文档 sha256 | `262e77d438daa8a790eccf23f8cfd696bf620a304bb4d42b54e38282e085b64a`，与 `out/plan-mlx.json` **逐字节相同** | [SUMMARY](docs/SUMMARY.md) §1 |
| `mlx_align.py plan`（对参考产物） | `ALIGNMENT OK`；最差余量 **rotation angle 0.26×** 容差（roots 6.8e-07 m、rotations 8.7e-06） | [SUMMARY](docs/SUMMARY.md) §1 |
| `mlx_align.py trace`（两后端） | 双双 `TRACE OK`；`pose.tokens` / `pose.sampling_uniforms` / `decoder.quantized` 都是 `0.000e+00` | [SUMMARY](docs/SUMMARY.md) §1 |
| 跨版门：5 seed × 6 步（对 `c81d8ee`） | tokens **30/30**、uniforms **30/30**、duration **30/30**，**0/26576** 决策不同，最终动作偏差 **0.000e+00** | [VIEWER](docs/VIEWER.md) §8 |
| `pytest -q` | **308 passed, 1 skipped** | [VIEWER](docs/VIEWER.md) §8 |

### 1.3 批量烘焙（release 配置 `--bake-stages all+decode`）

| 检查 | 实测 | 出处 |
|---|---|---|
| 同构（duration-homogeneous）B=16 端到端加速 | **4.443×**（**153.5 场景/秒**）；另一次独立运行 4.429×（144.5/s） | [SUMMARY](docs/SUMMARY.md) §2 |
| 混合分组 B=8 / B=16 单趟加速 | 1.670× / 2.810×（B=16 混合是 **draft**） | [SUMMARY](docs/SUMMARY.md) §2 |
| round-4 驱动臂 vs 单趟编排臂 | 每个 B、每种分组下**逐位相同**（`cross_arm_bitwise.bit_identical=True`） | [SUMMARY](docs/SUMMARY.md) §2 |
| 批量 vs 逐场景单跑（另一个进程、无任何 bake 开关） | 离散决策（tokens/uniforms/duration）**64/64 场景逐位一致**，**0/4032** 个 token 决策不同 | [BATCH-BAKE-RUN](docs/BATCH-BAKE-RUN.md) §3 |
| 12 项 manifest | 三份 manifest、五个分组全部 **12/12**，`manifest check: PASS` | [BATCH-BAKE-RUN](docs/BATCH-BAKE-RUN.md) §2 |
| 最终动作一致性（按容差判，不按逐位判） | roots ≤ **2.027e-06 m**（容差的 2.0e-03×）、rotations ≤ **2.753e-05**（1.4e-02×）；manifest 里 `bake_action_bit_identical` **如实为 `false`** | [SUMMARY](docs/SUMMARY.md) §2 |
| C 线质量门 `scripts/eval_gate.py --no-collect` | exit 0，**GATE PASS**：10 个质量组 + 2 组 cross-impl 全 PASS，5 项负对照全部按设计触发 | [BATCH-BAKE-RUN](docs/BATCH-BAKE-RUN.md) §4.1 |
| 渲染出的并排动画 | 批量臂与单场景臂 **0/38,574,400 个像素不同**（差异全部落在亚像素） | [BATCH-BAKE-RUN](docs/BATCH-BAKE-RUN.md) §5.3 |

![批量 B=16 与单场景 B=1 并排（walk，44 帧，左批量右单场景）](artifacts/bake-2026-10-12/gifs/pairs/w05-batch-vs-single.gif)

**这张图证明什么**：左（`BATCH B=16 (release cfg)`）与右（`SINGLE B=1`）是**同一个 walk 场景的两次规划**，
骨架像素全部来自参考渲染器，本仓库的 `tools/side_by_side_gif.py` 只做并排拼接并顺手数像素。
w05 这一对每帧的角上自己写着 `differing pixels 0/358400`；四个场景合计 **0/38,574,400 个像素不同**
（差异全部落在亚像素），离散决策则是 **64/64 场景、0/4032 个 token 逐位一致**（上表两行）。
**出处**：第 7 轮烘焙战役；渲染与拼接命令见 [BATCH-BAKE-RUN](docs/BATCH-BAKE-RUN.md) §5.1–§5.3，
文件取舍见 [`artifacts/bake-2026-10-12/README.md`](artifacts/bake-2026-10-12/README.md)。

### 1.4 切换（A→B 的条件化生成）

判据是任务书原文：接缝处的 `|dq|` ≤ 该片段内 `p95(|dq|)`，`q ∈ {位置, 速度, 加速度}`
（48 格 = 8 对 style × seed 1/7 × 接缝帧 12/20/28，同进程逐格交错测量，
[VIEWER](docs/VIEWER.md) §3）。

| 量 | 判据 | **本方法（条件化）** | 硬切对照（贴到接缝位姿） |
|---|---|---|---|
| root 位置差（48 格最大） | ≤ 一帧 p95 | **1.383 mm** | 116.654 mm |
| 最坏关节位置差 | 同上 | **15.352 mm** | 833.761 mm |
| root 速度比 中位 / 最大 / 超判据格数 | ≤ 1 | **0.302 / 0.472 / 0/48** | 3.084 / 6.455 / 46/48 |
| 关节角速度比 中位 / 最大 / 超 | ≤ 1 | 1.158 / 2.315 / **32/48（未达标）** | 3.788 / 4.982 / 48/48 |
| 关节角加速度比 中位 / 最大 / 超 | ≤ 1 | 1.398 / 3.301 / **42/48（未达标）** | 1.969 / 3.777 / 48/48 |

一句话读法：**位置与 root 速度达标且是数量级差**（位置差 230×、root 速度比 10×）；
**关节角速度大幅改善（3.3×）但没有达标**；**关节角加速度两种方法一样差**，是机制边界——
4 帧条件窗口里只有位置与一阶差分（速度），**没有加速度通道**（[VIEWER](docs/VIEWER.md) §3 末、§10）。

![A→B→A 整段（84 帧取 42 帧，底部字幕带标出当前在哪条腿，两处接缝有青色竖标）](artifacts/switch-shots/aba.gif)

**这张图证明什么**：`walk → walk_zombie → walk` 是**页面上一次连续播放**（第 20 帧、第 40 帧换 style），
不是两段视频拼起来的。**怎么读**：底部字幕带写当前在哪条腿（`walk` / `walk_zombie`）与页面的帧号；
画面里那两根**青色竖柱就是接缝标记**（图注里青色 = SWITCH SEAM），地面轨迹整条穿过了两处接缝
（GIF 末帧是页面第 82 帧，HUD `WALKED 2.06 m`）。**出处**：第 10 轮，42 帧都是
`?t=<帧>&paused=1` 的一次真实导航，源 PNG 的 sha256 记在 `artifacts/switch-run/gif-digests.txt`
（[VIEWER](docs/VIEWER.md) 第 10 轮 §6）。

| 条件化切换（本方法） | 硬切对照（同一时刻，只差 `?naive=1`） |
|---|---|
| ![条件化切换：SEAM GAP 0.17 mm、SEAM JUMP 1.66×、ROOT Z 0.88 m，画面里一根青色接缝柱](artifacts/switch-shots/live-switch-1440x900.png) | ![硬切对照：SEAM GAP —、SEAM JUMP —、ROOT Z 0.30 m，没有接缝柱](artifacts/switch-shots/hardcut-control-1440x900.png) |

**这两张图证明什么**：**同一时刻、同一相机、只差 `?naive=1` 一个参数**的一对。左边是条件化切换，
HUD 给出 `SEAM GAP 0.17 mm` / `SEAM JUMP 1.66×`（数字越小越好），画面里有一根**青色接缝柱**；
右边走的是硬切臂（`2:walk_zombie:reset/adv0`：重置到接缝位姿、前进 0 帧），HUD 变成
`SEAM GAP —` / `SEAM JUMP —`、**右侧无接缝柱**，同一时刻角色只在 z = 0.30 m（左边 0.88 m；
[VIEWER](docs/VIEWER.md) 第 10 轮 §6 记的差值是 0.57 m）。
**出处**：第 10 轮，无头 Chrome 只做导航与等待、不点界面、不合成输入；接缝那一帧的静帧
（`SEAM GAP 0.28 mm` + 两条接缝柱）在 [`artifacts/switch-shots/seam1-ab-1440x900.png`](artifacts/switch-shots/seam1-ab-1440x900.png)，
取舍见 [`artifacts/switch-shots/README.md`](artifacts/switch-shots/README.md)。

### 1.5 查看器（本机浏览器页面）

| 项 | 实测 | 出处 |
|---|---|---|
| 冷启动（载入权重 + 预热一次规划） | **370.7 ms**（三次启动 317.4 / 324.8 / 370.7） | [VIEWER](docs/VIEWER.md) §4 |
| 服务端规划 p50（连续 30 个串行 clip） | **28.3 ms → 27.9 clips/s**（其中 FK 2.5 ms，HTTP 往返 p50 34.8 ms） | [VIEWER](docs/VIEWER.md) §4 |
| 等效帧吞吐 / 相对 30 fps 的余量 | **1227 帧/秒** / **40.9×** | [VIEWER](docs/VIEWER.md) §4 |
| 换 style 后接缝连贯性（`advance`） | 接缝误差 **≤ 0.55 mm**，世界位移连续累积 | [VIEWER](docs/VIEWER.md) §2 |

![查看器实际画面：walk，CLIPS 11，WALKED 10.43 m，ROOT Z −7.88 m，W 键高亮、地面轨迹可见](artifacts/viewer-shots/viewer-final.png)

**这张图证明什么**：页面上这段 walk 是**浏览器里的真实 WebGL 画面**——`PLAN 63.0 ms` 与
`STEP RATE 26.0/s` 是页面自己报的实时数，`CLIPS 11` / `WALKED 10.43 m` 说明它是连续走出来的，
不是另写的渲染脚本替代品。**出处**：第 8 轮，无头 Chrome 只做 `goto` + `wait_for_timeout`、
无合成输入、未点界面（拍摄方法与逐张 sha256 见 [VIEWER](docs/VIEWER.md) §7；本图
`f0f706960a1300b6`）。

目标关键帧的命中画面（第 9 轮主图，`TARGET 0.111 m`；同一张的 DOM 属性
`data-mlx-target-hit=0.111320`、`data-mlx-target-hit-ok=true`、`data-mlx-target-window=40-43`）也在库里：
[`artifacts/viewer-targets-shots/final-900-settle.png`](artifacts/viewer-targets-shots/final-900-settle.png)
（出处 [VIEWER](docs/VIEWER.md) 第 9 轮 §5）。

画面证据**已入库**：`artifacts/viewer-shots/`（1 张）、`artifacts/viewer-targets-shots/`（1 张）、
`artifacts/switch-shots/`（4 个文件，2.64 MiB）。每个目录里都有一份极简 `README.md` 写明
哪张入库、哪些同批画面故意留在本机。

### 1.6 明确**未做到**的

| 未做到 | 数字 |
|---|---|
| **切换的关节角速度没有达标** | 32/48 个格子仍高于段内 p95；中位 1.158（对照 3.788）。地板控制（A→A 同 style 续规划）自己就是 1.131–2.174，所以是机制地板，不是切换引入的（[VIEWER](docs/VIEWER.md) §3、§10） |
| **切换的关节角加速度没有达标，而且两种方法一样差** | 本方法 42/48 超判据（中位 1.398），硬切对照 48/48（1.969）；`all_seams_within_p95` 在所有主配置里都是 `false`。4 帧窗口没有加速度通道，**要修得改模型条件通道（训练侧）**——训练线已按人类决定搁置（§5.2），所以这条残差是**已知、且不在本轮修复**的 |
| **全阶段批量后最终动作不是逐位的** | `bake_action_bit_identical=false`；roots ≤ 2.027e-06 m、rotations ≤ 2.753e-05，按容差判为合格 |
| **混风格同构库的档位掉到 draft** | `diverse`（8 种真实风格 · 同构 B=16）比值 **1.2716**（B=8 是 1.4402）→ draft；单风格同构批次才是 0.4089 → release（[BATCH-BAKE-RUN](docs/BATCH-BAKE-RUN.md) §0、§1.4） |
| **量化（q8/q4）不做** | 见 §5.3 |
| **查看器只验证过单客户端、且只实测过 GPU 档** | `--device cpu` 协议支持但页面没实测（[VIEWER](docs/VIEWER.md) §9 第 10 条） |
| **目标关键帧的命中包络有边界** | 正前方约 ≤4 m、侧向约 ≤2–3 m（按 style），超出就 `OUT OF TOLERANCE`；模型**不拒绝也不夹取**，会外推到 49 m/s@50 m（[VIEWER](docs/VIEWER.md) 第 9 轮 §3） |

---

## 2. 怎么用

环境是**仓库内的 `.venv`**（Python 3.13.12 + MLX 0.32.3 + NumPy 2.5.4，Metal 设备），
三条线（A/B/C）与人类只用 `./.venv/bin/python`，见 [decision 0001](docs/decisions/0001-python-environment.md)。

```sh
cd ~/dev/motionbrick-mlx
./.venv/bin/python scripts/check_env.py          # 打印平台/Python/MLX + 一次矩阵冒烟，期望 READY
```

**默认单场景规划**（`--out` 是必填，见 §4 硬规则）——两个后端分开进程跑：

```sh
./.venv/bin/python scripts/mlx_plan.py --device gpu --out /tmp/step-gpu.json
./.venv/bin/python scripts/mlx_plan.py --device cpu --out /tmp/step-cpu.json
diff -q /tmp/step-gpu.json out/plan-mlx-gpu.json     # 无输出 = 逐字节相同
diff -q /tmp/step-cpu.json out/plan-mlx.json
shasum -a 256 /tmp/step-gpu.json /tmp/step-cpu.json
```

**计时 / 冷启动 / 内存 / 确定性**（同样两个后端分开进程）：

```sh
./.venv/bin/python scripts/mlx_bench.py --steps 25 --devices gpu --json /tmp/gpu.json
./.venv/bin/python scripts/mlx_bench.py --steps 25 --devices cpu --json /tmp/cpu.json
```

**对齐**（对参考运行时的产物，以及逐阶段 trace）：

```sh
./.venv/bin/python scripts/mlx_align.py plan  out/ref/plan-ref.json /tmp/step-gpu.json
./.venv/bin/python scripts/mlx_align.py trace out/ref/trace/transition-0.json --device gpu
./.venv/bin/python scripts/mlx_align.py trace out/ref/trace/transition-0.json --device cpu
```

> `out/` 整个目录在 `.gitignore` 里、**没有入库**：`out/ref/**`（参考运行时的 plan 与逐阶段 trace）
> 由 [`tools/ref_harness.cpp`](tools/ref_harness.cpp) 链接参考 checkout 的静态库在本机生成，
> `out/plan-mlx*.json` 是上面那两条 `mlx_plan.py` 的产物。也就是说 `diff`/`align` 这两段命令
> 需要先在本机把参考产物造出来；仓库里入库的是**这些产物对应的 sha256 与对齐结论**（§1.2）。

**烘焙 release 档**（同构与混合两个分组；`B=1` 永不使用，`B>8` 必须时长同构）：

```sh
# 混合（duration-mixed）
./.venv/bin/python scripts/mlx_bench.py --bake-stages all+decode \
    --bake 1,2,4,8 --bake-single-pass --bake-repeats 9 --devices gpu --json /tmp/release-mixed.json
# 同构（duration-homogeneous），release 档上限 B=16
./.venv/bin/python scripts/mlx_bench.py --bake-stages all+decode \
    --bake 1,2,4,8,16 --bake-homogeneous --bake-single-pass --bake-repeats 9 \
    --devices gpu --json /tmp/release-homog.json
```

**烘一库真动作并导出**（12 项 manifest、等价性对照、C 线复核；完整流程与产物清单见
[BATCH-BAKE-RUN](docs/BATCH-BAKE-RUN.md) §5）：

```sh
# 批量臂（release 配置）与逐场景单跑臂（另一个进程、无任何 bake 开关），写到同一个 campaign 目录
./.venv/bin/python scripts/mlx_bake_library.py --campaign diverse \
    batch  --batch-size 16 --grouping homogeneous --strict --out artifacts/bake-2026-10-12/diverse
./.venv/bin/python scripts/mlx_bake_library.py --campaign diverse \
    single --batch-size 16 --out artifacts/bake-2026-10-12/diverse
# 逐场景配对比较（读两条臂写下的 runs-*.json，产物文档在各自的 plans/ 里）
./.venv/bin/python scripts/mlx_bake_library.py --campaign diverse compare \
    --batch  artifacts/bake-2026-10-12/diverse/runs-batch-B16.json \
    --single artifacts/bake-2026-10-12/diverse/runs-single-B16.json
# 对写盘的 manifest 逐字段核对 12 项
./.venv/bin/python scripts/mlx_bake_library.py --campaign single-style \
    check --manifest artifacts/bake-2026-10-12/single-style/manifest.json
# 用 C 线的同一套阈值评本轮的产物
./.venv/bin/python scripts/mlx_bake_library.py metrics \
    --dir artifacts/bake-2026-10-12/diverse/plans/batch-B16
```

> `*/plans/**` 与 `runs-*.json` **故意不入库**（可逐字节再生，128 份 plan 文档共 27.7 MiB）；
> 入库的是 `manifest.json`、`compare-B*.json`、`metrics-*.json`、`logs/` 与 GIF，取舍见
> [`artifacts/bake-2026-10-12/README.md`](artifacts/bake-2026-10-12/README.md)。
>
> 渲染成 GIF 用的是参考 checkout 里的 `tools/render_motion.py`（原文件未改），它要 Pillow；
> **本仓库 `.venv` 里没有 Pillow**，所以那一步用系统 Homebrew python，没有往 `.venv` 装东西。

**切换（A→B，以及 A→B→A）**：

```sh
# A->B->A 拼成一段，写 plan 文档并打印每个接缝的六行比值
./.venv/bin/python scripts/mlx_switch.py --style walk --to walk_zombie --back walk \
    --seed 1 --seam-at 20 --seam-at-back 20 --out /tmp/switch-aba.json
# 对照矩阵 + C 线质量门（硬切对照臂在同一个进程里逐格交错）
./.venv/bin/python scripts/mlx_switch.py --style walk --to walk_zombie --back walk \
    --control --quality --quality-sweep --seeds 1,7 --seam-frames 12,20,28 \
    --out artifacts/switch-run/switch-aba.json --report artifacts/switch-run/report.json
```

**浏览器查看器**（单线程，只绑 `127.0.0.1`；MLX GPU stream 是 thread-local 的，多线程会在 worker 里把进程 abort）：

```sh
./.venv/bin/python scripts/mlx_viewer.py --port 8848 --device gpu > /tmp/viewer.log 2>&1 &
curl -s http://127.0.0.1:8848/api/health          # 设备 / 参数量 / 冷启动 / 累计计数
curl -s "http://127.0.0.1:8848/api/plan?style=walk&seed=1&move=0,1&facing=0,1&reset=1" > /tmp/plan.json
curl -s http://127.0.0.1:8848/api/stats           # 服务器滚动计时 + 最近一次页面自报耗时
kill "$(lsof -nP -iTCP:8848 -sTCP:LISTEN -t)"     # 停
```

（连续步频的探针脚本 `artifacts/viewer-run/probe_viewer.py` 是本机产物、**没有入库**，
用法与输出见 [VIEWER](docs/VIEWER.md) §4、§10。）

页面在 `http://127.0.0.1:8848/`：15 个真实 `.mbstyle`、相机 orbit、地面轨迹、
**YOUR TARGET KEYFRAMES**（点地面摆 1–4 个目标帧，HUD 给 root 命中误差）、
**SWITCH STYLE MID-MOTION**（走着的时候切换，HUD 给 `SEAM GAP` 与 `SEAM JUMP`，
并可勾 `Hard cut (control)` 看对照臂）。URL 参数（`?style=`、`?seed=`、`?move=`、`?targets=`、
`?switch=`、`?t=&paused=1`）只是地址栏预置，不是合成输入。
**服务随会话结束而停**：它只有一份有状态 agent，没人连它就只是一个空转的本地进程。

**测试**：

```sh
./.venv/bin/python -m pytest -q
```

---

## 3. 目录结构

```
docs/decisions/0001..0010.md   每个关键裁定的正文（环境、既有实现、运行时、位一致、后端条件化、
                               烘焙安全档、档位规则、裕度阈值、批量解码、场景级安全度量）
docs/SUMMARY.md                A 侧收尾基线：默认路径数字、档位与依据、一键复现、诚实边界
docs/VIEWER.md                 查看器：URL/起停、协议、交互、性能、复用与归属、边界（含目标帧与切换）
docs/BATCH-BAKE-RUN.md         首次真烘焙实测：场景、12 项 manifest、等价性、质量门、产物
docs/optimization-report.md    长版证据（第 1–49 节，含被否决的项与理由）
docs/motion-representation.md  动作表示契约（数据块、单位、磁盘布局）
docs/status-{deepseek,claude,glm}.md   三条线各自的状态、在动文件、阻塞
docs/brief-{claude-training,glm-eval}.md   B / C 两线的工作单
src/motionbricks_mlx/motion/   表示、重定向、读写
src/motionbricks_mlx/models/   MLX 的 root/pose/decoder 与批量烘焙编排
src/motionbricks_mlx/data/     数据集加载、agent、上下文、目标帧、切换
src/motionbricks_mlx/export/   plan 文档 / VRMA 导出
src/motionbricks_mlx/training/ B：训练脚手架（VQ-VAE、数据集、checkpoint）——已提交，按 §5.2 **暂停**、未接真实数据
src/motionbricks_mlx/eval/     C：动作质量指标与回归门（已交付并在用，§5.2）
scripts/mlx_{plan,bench,align,convert_gguf,plan_targets,switch,viewer,bake_library}.py
viewer/                        浏览器页面（自有 HTML/CSS/JS + 参考 demo 的 three.js vendor 与骨架模块）
tools/side_by_side_gif.py      只做并排拼接（像素全部来自参考渲染器）
tests/                         308 passed, 1 skipped
artifacts/bake-2026-10-12/     烘焙证据（manifest、对照、日志、GIF；哪些入库见该目录 README.md）
artifacts/viewer-shots/        查看器主图（1 张）
artifacts/viewer-targets-shots/ 目标关键帧命中主图（1 张）
artifacts/switch-shots/        切换画面：aba.gif + live-switch / hardcut-control 对照对 + 接缝静帧
artifacts/switch-run/  artifacts/{bake-2026-10-12,viewer,viewer-targets}-run/
                               本机产物（report/plan/log/pid/探针/GIF 中间帧），**不入库**（逐目录取舍见各 README.md）
datasets/  checkpoints/  out/  本地产物，不入库（.gitignore）
```

---

## 4. 档位与硬规则

判据**先冻结再动手**，改一次要在 `docs/decisions/` 留一条：

* **release 档 = 扰动 / 最紧决策余量 ≤ 0.70**（[0008](docs/decisions/0008-margin-threshold-and-grouping.md)）；
  基于这条规则实测：**混合分组 release B ≤ 8**（B=8 比值 0.6695，B=16 是 1.0426 = 零余量 → draft），
  **时长同构 B ≤ 16**（0.4089–0.5290，[SUMMARY](docs/SUMMARY.md) §2）。
  档位是**数据相关**的：把 8 种风格放进同一个同构批次，实测比值 1.2716 → draft
  （机制是"分子来自 `idle` 的大扰动、分母来自 `walk_zombie` 的紧余量"，逐场景自己的比值最大只有 0.142；
  [BATCH-BAKE-RUN](docs/BATCH-BAKE-RUN.md) §1.4）。
* **`B = 1` 永不用于烘焙**——它比串行还慢（混合 0.849–1.008×，同构 0.906×）。
* **`B > 8` 必须按时长同构成组**；混合分组在 B=16 的余量为零。
* **换模型 / 换权重 / 换指令分布 → 必须重跑战役并重算裕度**，任一 B 比值 > 0.70 即机械降档。
* **任何 `mlx_plan.py` 调用必须显式 `--out`**（第 2 轮曾覆盖过一次已验收文档）。
* 产物 manifest 必须记 `0007` 第 4 条要求的 **12 项**，`bake.manifest_fields` 逐 B 生成，`check` 对写盘文件逐字段核对。

---

## 5. 当前范围与不做的事

### 5.1 当前交付范围（就到这里为止）

本仓库现在交付的是这条推理路径上的全部产物：

| 交付 | 状态 | 证据 |
|---|---|---|
| MLX 推理，对参考运行时逐位对齐 | **已完成** | §1.1、§1.2 |
| 优化定档（release / draft 的机械判据） | **已完成** | §1.3、§4 |
| release 档批量烘焙（`--bake-stages all+decode`） | **已完成** | §1.3、§1.6 |
| 浏览器查看器（单客户端、实测 GPU 档） | **已完成** | §1.5 |
| 目标关键帧（页面上摆、HUD 报命中） | **已完成**，命中包络有边界（§1.6） | §1.5 |
| 动作切换（A→B、A→B→A，条件化生成） | **已完成**：位置与 root 速度达标，关节角速度/角加速度**未**达标（§1.6） | §1.4 |

### 5.2 训练线与"视频 / 第三方动作 → clip"暂不在此范围

**人类已决定搁置这条线**，理由是数据源，不是工程量：空间内现有的动作是 **Unity 自带的一批动作
＋少量 BONES 动作**，直接拿它们训练只会把机械感一起学进去；"从视频 / 第三方动作转成 clip"
是另一半，也还没有解决数据从哪来——**另立一项，不进本仓库主干**。据此：

* **C 线（`src/motionbricks_mlx/eval/**`，动作质量评测与回归门）已交付并在用**：本仓库的烘焙产物
  就是用同一套阈值复核的（§1.3 的 `GATE PASS`、§1.4 的质量门）。
* **B 线（`src/motionbricks_mlx/training/**`，训练脚手架）已提交、但按此决定暂停**：它全链路能跑、
  测试是绿的，**只喂过合成 shard、没有接真实数据**——不要读成"训练已完成"。
* **两条线都是独立做法，通过已冻结的契约对接**：磁盘形态是 [`docs/motion-representation.md`](docs/motion-representation.md)
  定义的 `clip.npz`（外加 `motion/representation.py` 导出的常量）。将来若有人接手训练，
  **不进本仓库主干**，只按这份契约产出 `clip.npz` 交给推理侧。
* 这条决定**没有撤销任何已入库的数字**：§1.4 / §1.6 里"要修得改模型条件通道（训练侧）"那两行残差
  保持已知、保持不在本轮修复。

### 5.3 明确不做

| 不做 | 为什么（带证据） |
|---|---|
| **量化（q8 / q4）** | 会改动作：q4 逐 token 差异 27–31 %、3/6 步连时长都变、roots 达容差的 30–91×；q8 差异 1.5–3.6 %、roots 2.2–25.6×；`mlx_align plan` 对参考 FAIL（q8 4 个数组、q4 9 个）。裁定见 [0004](docs/decisions/0004-adopt-bit-identical-f32.md)、[0005](docs/decisions/0005-backend-conditional-compile.md) 与报告第 6/9 节。 |
| **单场景实时路径的进一步优化** | 宿主数学已从 17.58 → 1.74 ms/步（报告 §12–13）；把 FK/约束搬进 MLX 图在 B=1 是**净亏**（1.64 ms vs 宿主 0.371 ms，且 `cos/sin/arctan2` 与 NumPy 差 1–2 ULP）；`pred_local_defer` / `decoder_defer` 在 B=1 只有 1.0207× / 1.0035×，低于 1.05× 门槛，按判据否决。它们**只在批量里才值钱**。 |
| **参考运行时侧的事** | 参考 checkout `~/dev/motion-bricks-cpp` 全程只读、一个字节没改；GGUF / 引擎侧不在本仓库范围。 |
| **训练线与"视频 → clip"** | 见 §5.2：B 线提交后暂停、未接真实数据；训练与数据源另立一项，只按 `clip.npz` 契约交货。 |

---

## 6. 许可证与归属

* 本仓库代码 **Apache-2.0**（[`LICENSE`](LICENSE)）。
* **不使用、不转换、不再分发 NVIDIA 预训练权重**：上游源码是 Apache-2.0，权重是 NVIDIA Open Model License；
  本仓库产出的每个 checkpoint 都由我们自己生成的数据训练（[`NOTICE`](NOTICE)）——**当前没有发布任何
  checkpoint**，B 线训练脚手架只喂过合成 shard（§5.2）。
* 查看器复用：`viewer/vendor/three.module.min.js`、`three.core.min.js`、`THREE-LICENSE.txt` 是**逐字节复制**
  （three.js 本体是 MIT）；`viewer/rig.js` 的骨架 rig/label/画线四段与 `viewer/style.css` 前 63 行
  **逐字复制**自参考 demo（Apache-2.0），唯一机械改动与原始行号写在文件头注释里；
  `viewer/index.html`、`viewer/app.js`、`scripts/mlx_viewer.py` 是本仓库新写。
  逐块 sha256 与归属三段说明见 [VIEWER](docs/VIEWER.md) §5。

---

## 7. 决策与证据索引

| 决定 | 一句话 |
|---|---|
| [0001](docs/decisions/0001-python-environment.md) | 统一用仓库内 `.venv`（Homebrew python@3.13；MLX 0.32.3 / NumPy 2.5.4） |
| [0002](docs/decisions/0002-prior-art.md) | 先查既有实现：有 GGML 版与 Core ML 版，MLX 侧只有 VQ-VAE 零件 |
| [0003](docs/decisions/0003-downstream-runtime.md) | 下游运行时选 `motion-bricks.cpp`（Apache-2.0 / GGML / GGUF 边界） |
| [0004](docs/decisions/0004-adopt-bit-identical-f32.md) | 采用可证明逐位一致的 f32 优化；量化继续关闭 |
| [0005](docs/decisions/0005-backend-conditional-compile.md) | 三个 `*_compile` 开关按后端决定默认值（非 Metal 默认关，避免 2× 峰值内存） |
| [0006](docs/decisions/0006-batch-bake-safe-b.md) | 批量烘焙的"安全 B"规则 |
| [0007](docs/decisions/0007-bake-full-stage-tier.md) | 全阶段批量烘焙的档位规则与 manifest 必记 12 项 |
| [0008](docs/decisions/0008-margin-threshold-and-grouping.md) | 裕度阈值改成可机械判定（≤0.70）、`B>8` 必须时长同构、机械降档 |
| [0009](docs/decisions/0009-batch-the-motion-decode.md) | 把运动解码批量化，并列为 release 烘焙配置的一部分 |
| [0010](docs/decisions/0010-scenario-level-safety-metric.md) | 用场景级安全度量把扰动与"同一条决策的余量"对齐比较 |

证据文档：[SUMMARY](docs/SUMMARY.md)（收尾基线）、[optimization-report](docs/optimization-report.md)（第 1–49 节，含被否决项）、
[VIEWER](docs/VIEWER.md)（查看器 / 目标帧 / 切换）、[BATCH-BAKE-RUN](docs/BATCH-BAKE-RUN.md)（真烘焙实测）、
[motion-representation](docs/motion-representation.md)（数据契约）、
[status-deepseek](docs/status-deepseek.md) / [status-claude](docs/status-claude.md) / [status-glm](docs/status-glm.md)。

---

## 8. 诚实边界

1. **机器不空闲。** 测量期间 1 分钟 load average 常在 5.6–19.6（桌面进程 mihomo、WindowServer、
   Chrome、DSH renderer、Lark；没有训练/评测进程），绝对值被抬高约 **10 %**。
   所以前后对比一律用**同进程交错配对**的比值；跨 B 的配对每个 B 一个进程（长命进程会跨 B 积累状态，
   曾把 B=8 从 1.1726× 量成 0.936×）。
2. **`mlx_align` 的旋转角列有地板。** `plan` 比较里 rotation angle 列读到 4.3e-02–5.1e-02 度
   （容差 2.0e-01 度，即 **0.26×**），这是该列浮点角度的量化地板，**不是回归**；其余列在 1e-07–1e-05 量级。
3. **四帧条件窗口没有加速度通道。** 窗口携带的是位置与一阶差分；所以"位置 + 速度"能闭合，
   **加速度不能**——关节角速度/角加速度那两行的残差是机制的，要修得改模型条件通道（训练侧）。
4. **动作一致性按容差判，不按逐位判。** 全阶段批量后最终动作 roots ≤ 2.027e-06 m、rotations ≤ 2.753e-05，
   manifest 的 `bake_action_bit_identical=false` 就是如实记录。
5. **查看器一个进程 = 一个 agent。** 多客户端会互相打乱上下文；本轮所有测量都是单客户端串行。
   要多人用就得给 agent 加会话键。`--device cpu` 协议支持但页面没实测。
6. **服务随会话结束而停**，它不做鉴权、不绑远程地址（`--host` 只接受 `127.0.0.1`）。
7. **画面证据只入库了正文引用的那几张。** `artifacts/viewer-shots/`（1 张）、
   `artifacts/viewer-targets-shots/`（1 张）、`artifacts/switch-shots/`（4 个文件，2.64 MiB）**已提交**；
   同一批会话的其余帧、`artifacts/{switch,viewer,viewer-targets}-run/**` 与
   `artifacts/bake-2026-10-12/*/plans/**`、`runs-*.json`、`render-*.json`、`server.log`、`*.pid`、
   探针脚本、GIF 中间帧**故意不入库**（可再生产），逐目录取舍写在各目录的 `README.md` 里。
8. **参考版 GGML 的 380 ms 是引用值**，不是在只读 checkout 里复测或找到出处的；真正同口径的对照是本仓库
   自己的 GPU 单场景 p50 26.6 ms（[VIEWER](docs/VIEWER.md) §4）。
9. **`/api/client-metrics` 只留最后一次页面自报**，不聚合历史；HUD 的 step rate 是最近 8 次规划的滑动中位数。

---

## 9. 分工（本仓库由三个 agent 与一位人类共同推进）

**核心规则只有一条：一份文件同一时间只有一个写手。** 越界前必须先协调。

| 归属 | 谁 | 文件范围 | 现状 |
|---|---|---|---|
| **A. 推理移植 / 优化 / 烘焙 / 查看器 / 切换** | **DeepSeek** | `src/motionbricks_mlx/{models,data,export,motion}/**`、`scripts/mlx_*.py`、`tests/test_mlx_*.py`、`tests/test_viewer_*.py`、`viewer/**`、`docs/status-deepseek.md`、`docs/SUMMARY.md`、`docs/VIEWER.md`、`docs/BATCH-BAKE-RUN.md`、`docs/optimization-report.md`、`docs/decisions/**` 的提案 | **已完成**（第 1–10 轮）：MLX 推理与逐位对齐、优化定档、release 档批量烘焙、浏览器查看器、目标关键帧、动作切换 |
| **B. 模型与训练** | **Claude Code** | `src/motionbricks_mlx/training/**`、`scripts/train_vqvae.py`、`tests/test_training_*.py`、`docs/status-claude.md` | 训练脚手架**已提交**（`bc78bd8`）且全链路可跑、测试绿；但**只喂过合成 shard、没有接真实数据**，并按人类决定**暂停**（§5.2）：将来接手只按 `clip.npz` 契约交货，不进本仓库主干 |
| **C. 动作质量评测与回归门** | **GLM** | `src/motionbricks_mlx/eval/**`、`scripts/eval_{collect,gate}.py`、`tests/test_eval_*.py`、`docs/status-glm.md` | **已交付并在用**（`306a684`）：6 个指标族、带出处的阈值表、5 项负对照、`eval_gate.py` GATE PASS；A 侧用它复核烘焙产物（§1.3、§1.4） |
| **D. 冻结区** | **人类** | `LICENSE`、`NOTICE`、`pyproject.toml`、`.gitignore`、`README.md`、`docs/motion-representation.md` | `LICENSE` / `NOTICE` / `pyproject.toml` / `.gitignore` / `docs/motion-representation.md` **未动**；`README.md` 已由人类明确授权重写（上一轮写门面，本轮补画面证据并把范围收口到 §5） |

* **接口优先**：A 与 B/C 之间只通过 `docs/motion-representation.md`（数据契约）与
  `motion/representation.py` 导出的常量（`FRAME_DIM`、`OFFSETS`、`CLIP_META_KEYS`）互相依赖。
  A 改这些＝契约变更，必须同时给 schema 版本号与迁移说明；B/C 不得自行改契约。
  这份契约也是**训练线将来接手时的唯一交接面**（B 线已按 §5.2 暂停）。
* **状态不共享文件**：各自只写自己的 `docs/status-*.md`，避免互相覆盖（吃过一次教训）。
* **提交纪律**：只提交自己名下的文件；`datasets/`、`checkpoints/`、`out/` 不入库，
  数据集与 checkpoint 靠 **manifest**（`git sha + 随机种子 + 超参 + 数据来源 id`）复现；
  每次提交写清"本轮实际跑了什么验证"，**不许写"已验证全套"**；长时间训练同时只允许一个人跑。
* **人类决定的事**：范围与优先级、动作表示里的取舍、审美、许可证相关的一切、`docs/decisions/**` 的裁定
  （agent 只写提案）、以及推到 `main`。**"训练线与视频 → clip 暂不做"就是一条这样的范围决定**
  （§5.2），它搁置的是工作量、不是已入库的结论。
* **每个结论都要带证据**：阈值要能失败（"改坏就红"）、数字要来自真实运行、看不清的地方写"未验证"。
