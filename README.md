# motionbrick-mlx

在 Apple Silicon 上用 **MLX** 做**实时角色动作生成**：模块化潜在动作模型（动作 VQ-VAE，根位移与姿态解耦）
＋ 条件生成器，最终产出一条**离线烘焙**管线，输出我们引擎本来就能播的动作资产（VRMA / 动画剪辑）——
这样产品运行时的动作成本是零。

架构参考 MotionBricks（`NVlabs/GR00T-WholeBodyControl` 的 `motionbricks/`，见 `NOTICE`）。
**只走源码路线**：上游源码是 Apache-2.0，我们不使用、不转换、不再分发它的预训练权重，
本仓库产出的每个 checkpoint 都由我们自己生成的数据训练。

## 为什么是这个形态

- 参考实现是 PyTorch + CUDA + MuJoCo，且 demo 偏 Linux/X11；Apple 芯片没有 CUDA，MLX 是原生栈。
- 我们不缺语料：**我们自己的 3D 空间就是仿真器**，可以批量生成 (状态, 动作, 结果) 的动作数据。
- 我们要的是"能进产品的动作资产"，不是"跑得起来的论文 demo"。

## 五步计划（每步都有可验收的产出）

| 步 | 内容 | 产出 | 验收 |
|---|---|---|---|
| 1 | 动作表示 ＋ 从空间导数据 | `docs/motion-representation.md` 定稿、`clip.npz` 真实样本 | 往返转换 float32 容差内稳定；样本能被可视化脚本读出来 |
| 2 | MLX 版动作 VQ-VAE | `models/vqvae.py` ＋ 训练脚本 | 重建误差达标（阈值写进训练脚本）；改坏就红 |
| 3 | 条件生成器（风格 / 目标 / 根轨迹） | `models/generator.py` | 生成片段在物理与语义检查上通过（不是"看着像"） |
| 4 | 烘焙导出 | `export/` → VRMA / clip | 导出物能在引擎里播，帧数与元数据对得上 |
| 5 | （可选）端上实时 | Core ML 小解码器 | 只在第 4 步闭环后才启动 |

## 分工（本仓库由两个 agent 与一位人类共同推进）

**核心规则只有一条：一份文件同一时间只有一个写手。** 今天已经吃过一次教训——两个写手在同一批文件上
互相覆盖，构建坏掉、白干一轮。以下所有权是硬的，越界前必须先协调。

| 归属 | 谁 | 文件范围 |
|---|---|---|
| A. 表示与数据 | **DeepSeek** | `docs/motion-representation.md`、`src/motionbricks_mlx/motion/**`、`src/motionbricks_mlx/data/**`、`scripts/export_from_space.py`、`scripts/check_env.py`、`tests/test_motion_*.py`、`README.md` |
| B. 模型与训练 | **Claude** | `src/motionbricks_mlx/models/**`、`src/motionbricks_mlx/export/**`、`scripts/train_*.py`、`scripts/bake_*.py`、`tests/test_models_*.py`、`tests/test_export_*.py` |
| C. 冻结区 | 人类 | `LICENSE`、`NOTICE`、`pyproject.toml`、`.gitignore`、`docs/decisions/**` |

**接口优先。** 两边只通过两样东西互相依赖：`docs/motion-representation.md`（数据契约）与
`motion/representation.py` 里导出的常量（`FRAME_DIM`、`OFFSETS`、`CLIP_META_KEYS`）。
A 改这些＝契约变更：必须同时给出 schema 版本号与迁移说明；B 不得自行改契约，遇到不够用的字段就把需求写进
`docs/decisions/` 下的一条提案。

**状态不共享文件。** 各自只写自己的状态文件，避免又一次互相覆盖：

```
docs/status-deepseek.md    A 的当前进度、正在动的文件、下一步、阻塞
docs/status-claude.md      B 的当前进度、正在动的文件、下一步、阻塞
```

**提交纪律。**
* 只提交自己名下的文件；跨范围改动先问人类。
* `datasets/`、`checkpoints/`、`out/` 一律不进版本库（已在 `.gitignore`）。数据集与 checkpoint 靠
  **manifest** 复现：`git sha + 随机种子 + 超参 + 数据来源 id`，缺一项就不算可复现。
* 每次提交写清"本轮实际跑了什么验证"，**不许写"已验证全套"**。
* 长时间训练同一时间只允许一个人跑（算力与磁盘是共享的），开跑前在自己的状态文件里声明。

**人类决定的事**：范围与优先级、动作表示里"要不要完整根姿态"这类取舍、审美、许可证相关的一切、
以及推到 `main`（两个 agent 都不直接推主干，除非人类明确放行）。

**每个结论都要带证据。** 阈值要能失败（"改坏就红"）、数字要来自真实运行、看不清的地方写"未验证"。
这条不是流程洁癖——今天所有翻车都出在"测试绿了就当能用"。

## 目录

```
docs/motion-representation.md   动作表示：单位、手性、逐帧特征、片段元数据、磁盘布局
docs/status-*.md                各自状态
src/motionbricks_mlx/motion/    表示、重定向、读写
src/motionbricks_mlx/models/    vqvae.py、generator.py
src/motionbricks_mlx/data/      数据集与生成数据加载
src/motionbricks_mlx/export/    烘焙到引擎动作格式
scripts/check_env.py            这台机器是否就绪（Apple Silicon + MLX）
datasets/  checkpoints/         本地产物，不入库
```

## 环境

MLX 需要 **Python 3.11+** 与 Apple Silicon。先跑：

```sh
python3 scripts/check_env.py     # 打印平台、Python、MLX 与一次矩阵冒烟
```

## 许可证

本仓库代码 **Apache-2.0**；不使用也不分发任何 NVIDIA 权重，详见 `NOTICE`。
