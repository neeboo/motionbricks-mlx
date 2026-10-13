# 交给 Claude Code 的工单（B 侧：训练脚手架）

**工作目录**：`~/dev/motionbricks-mlx`。先读 `docs/status-claude.md`（分工与规则）、`NOTICE`（只走 Apache-2.0 源码路线）、
`docs/decisions/0001-python-environment.md`（统一用 `./.venv/bin/python`）、
`docs/decisions/0003-downstream-runtime.md`（下游是 motion-bricks.cpp，GGUF 是边界）。

## 你只写这些文件（与另一条并行线零重叠，不要碰别人的文件）
- `src/motionbricks_mlx/training/**`（新建目录）
- `scripts/train_*.py`
- `tests/test_training_*.py`
- `docs/status-claude.md`（只更新这个状态文件）

**不要碰**：`src/motionbricks_mlx/{models,data,export}/**`、`scripts/mlx_*.py`、`scripts/convert_*.py`、
`tests/test_mlx_*.py`、`docs/motion-representation.md`、`README.md`、`NOTICE`、`LICENSE`、`pyproject.toml`
——这些归另一条线或人类。

## 任务：在**尚未有真实数据**的前提下，按契约把训练脚手架写好
数据契约在 `docs/motion-representation.md`（逐帧特征、片段元数据、`npz` 布局、单位与手性）。
它由 A 侧负责实现；你按契约工作，并用**你自己生成的合成分片**（形状与元数据符合契约、数值随机）来开发与测试，
不要等真实数据、也不要自己造"另一种格式"。

要做：
1. `data/` 之外的**数据集读取器**：读 `datasets/<space>/<capture>/clip.npz`（`frames (T,D)`、`offsets`、`meta.json`），
   支持可变长度与 batch padding/mask。
2. **MLX 版动作 VQ-VAE**：编码器/解码器 + codebook；训练循环（含重建损失、codebook 损失、EMA 或 commitment 按你选择并说明理由）、
   检查点保存（`checkpoints/`，已被 gitignore）、以及一份 `manifest.json`（git sha + seed + 超参 + 数据 id）。
3. **评测脚本**：重建误差指标（按位置/旋转分别算），阈值写进脚本并可调；训练脚本要能跑通合成数据的极小规模（几十步）而不报错。
4. **测试**：形状、mask 正确性、一次前向、一次极小训练循环、以及"改坏就红"的负对照（例如故意打乱 mask 或改错维度）。

## 硬要求
- **接口优先**：契约文件的字段名/维度/顺序**不许改**；不够用就在 `docs/status-claude.md` 里写"需要新增字段 + 原因"，交人类决定。
- **每个结论带证据**：训练跑通的真实输出、指标数字、失败就贴原始报错。
- **不许**：push 任何远端（本地 commit 可以）、改冻结区文件、动产品代码、动 `~/dev/motion-bricks-cpp`（只读参考）、
  安装系统级依赖、长训练（只允许极小规模冒烟，几十步）。
- 完成后在 `docs/status-claude.md` 写：做了什么、跑了什么验证、下一步、阻塞。
