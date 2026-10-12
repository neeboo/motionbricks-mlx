# 0005 · 三个 `*_compile` 开关改为按后端决定默认值（非 Metal 默认关闭）

**触发**：第二轮优化把 `MOTIONBRICKS_MLX_OPT` 默认翻成开后，实测发现**只在非 Metal 后端**出现回归：
CPU 25 步 `OPT=0` → p50 199.0 ms / MLX 峰值 748 MB / RSS 853 MB；默认 → 178.8 ms / **1481 MB / 1615 MB**；
默认再关掉那三个 `*_compile` → 173.8 ms / 748 MB / 854 MB。即**内存翻倍、速度无可测收益**（173.8 与 178.8 的差在单臂漂移内）。

**裁定**：给 `pose_compile` / `root_compile` / `decoder_compile` 三个开关**按后端决定默认值——Metal 上保持开，非 Metal 上默认关**。

**为什么这不是"从已批准的决定里挑肥拣瘦"，而是恢复该决定的前提**：`0004` 采纳整组的依据是
**逐位一致 ＋ 无回归**。在 CPU 上 2× 峰值内存且无速度收益，**本身就是回归**，因此按后端条件化是
**让默认值重新满足"无回归"**，而不是把某个收益私藏起来。GPU/Metal（我们的默认路径）行为不变。

**验收要求**（改完必须复跑）：
1. **GPU 默认路径产物与 flip 后逐字节一致**（`diff -q` 干净）；`mlx_align plan/trace` 仍 OK，
   `pose.tokens` 与 `sampling_uniforms` 仍 0 差；
2. **CPU 默认**下峰值 MLX 回到 ~748 MB / RSS ~854 MB，且 p50 不劣于 `OPT=0`（在漂移内即可）；
3. 5 组 seed × 6 步 token 一致；`mlx_plan --repeat 2` 两个后端都 deterministic；`pytest` 全绿。

**附带规矩**（来自本轮的 incident）：不要在没有 `--out` 的情况下跑 `mlx_plan.py`，它会写默认产物
`out/plan-mlx.json`；该文件已按四个同源文档逐字节还原（sha256 `262e77d4…`）并记录在
`docs/optimization-report.md` 第 20 节。凡是覆盖了"约定俗成的产物路径"的恢复，**必须给出 sha256 与重放证据**。

## 附注（2026-10-12，第三轮纠正）

**本决定引用的"CPU 峰值 748 → 1481 MB"前提不成立**：那是**多后端 harness 的产物**，不是 CPU 默认的成本。
控制实验：CPU-only 进程改前改后都是 748 MB；`OPT=0` 748；只开编译开关但不走快路径 748；全快路径但不编译 748；
反向 `--devices cpu,gpu` 时被抬高的是 **GPU** 那行（1512 MB，而其自身峰值 779 MB）。
触发条件是**同一进程里第一个真正建过 `mx.compile` 变体的后端**（Metal 按设计保留编译）。
逐条见 `docs/optimization-report.md` 22.2。

**规则本身不变，理由换成站得住的那一条**：在 CPU 上这三个编译开关**没有任何可测的延迟收益**
（三臂配对只差约 1.25%，在方法分辨率边缘），所以"非 Metal 默认关"仍然是合理的默认。
已按本决定落地实现（`nn.declared_default()` + `_BACKEND_CONDITIONAL`），无需改代码。
