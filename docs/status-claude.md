# 状态 · Claude（B：模型与训练）

- 当前：工单 `docs/brief-claude-training.md` 四项交付全部完成——训练侧脚手架全链路可跑，全套测试绿
  （83 passed, 1 skipped；skip 是 A 线的 `test_representation_smoke`，与本线无关）。
- 正在动的文件：`src/motionbricks_mlx/training/**`、`scripts/train_vqvae.py`、`tests/test_training_*.py`、本文件。
- 等决定：对契约的两处解释需 A 确认；VQ-VAE 与 eval 入口的目录归属需人类定界（见下"需要协调"）。

---

## 2026-10-12 · B 线训练脚手架（工单 docs/brief-claude-training.md）

**越界说明**：只写了工单放行的文件——`src/motionbricks_mlx/training/**`（新目录）、
`scripts/train_vqvae.py`、`tests/test_training_{layout,dataset,vqvae,metrics,loop}.py`、本文件。
未碰 `models/**`、`data/**`、`export/**`、`scripts/mlx_*.py`、`scripts/convert_*.py`、
`tests/test_mlx_*.py`、README/NOTICE/LICENSE/pyproject。`~/dev/motion-bricks-cpp` 只读未动。

### 做了什么（对照工单四项交付）

1. **数据读取（在 `data/` 之外）**：`training/layout.py` 把契约块表转写成代码
   （块顺序 `root_position(3), root_yaw(1), root_velocity(3), joint_rotations(6J), joint_velocities(3J)?, contacts(2), phase(2)`；
   offsets 表校验；字段名/顺序未改动任何一字）。`training/dataset.py` 读
   `<root>/<space>/<capture>/clip.npz` + 兄弟 `meta.json`，变长 batch 走 padding + bool mask，
   padded 长度对齐模型 downsample 因子；坏 shard（D 不符、缺 meta 键、非有限值、混维度）一律 loud fail。
2. **合成数据**：`training/synth.py` 按契约布局生成 seeded 随机平滑行走片段（J=4 带 joint_velocities
   → D=47），6D 旋转用正交对 + Gram-Schmidt，未发明任何新格式。不依赖真实数据到齐。
3. **VQ-VAE + 训练**：`training/vqvae.py`——1D 时序卷积 encoder/decoder（channels-last）+
   EMA 码本量化器。**码本选 EMA 而非梯度码本损失**，理由：① EMA 从编码统计更新码字，没有需要平衡的
   额外损失项——码本塌缩在小 batch、几十步的冒烟预算里最容易由平衡失败引发；② 标准配方
   （van den Oord 2017），状态更新是纯数组运算，从 (seed, 超参) 可精确复现，契合 manifest 契约；
   ③ 仍保留 commitment β=0.25，因为 EMA 之外没有别的力把 encoder 输出拉向码字。
   训练循环 `training/train.py`：masked 重建 MSE + β·commit，AdamW + 手动全局范数裁剪
   （MLX 0.32.3 无 clip_grad_norm），首批用真实编码播种码本（`seed_from_encodings`，不做这步会塌缩，
   见教训 3）。checkpoint `training/checkpoint.py`：`checkpoint.npz` + `manifest.json`
   （git sha + dirty + seed + 全部超参 + 每条 clip 的 sha256 数据 id + 指标）。
4. **评估**：`training/metrics.py` + `training/eval.py`——**位置（m）与旋转（geodesic 角，deg）分开报告**
   （另附 p95 与全块 frame MSE）；阈值放在 `scripts/train_vqvae.py` 顶部、
   `--position-threshold` / `--rotation-threshold` 可调，越线 exit 1。
5. **CLI**：`scripts/train_vqvae.py` 三个子命令 `synth` / `train` / `eval`。
6. **测试 50 条**：形状、mask 正确性（污染 padding 区损失变化 <1% 相对差且已归因于已知的 conv 渗漏；
   打乱 mask 就红）、一次前向、12 步极小训练循环、checkpoint 往返逐位一致（`mx.array_equal`）、
   manifest 缺 `seed/git_sha/hyperparams/data/format` 任一就 ManifestError、
   维度改错（喂 D=35 给 D=47 模型、旋转块宽度非 6 倍数）就红、geodesic 与闭合式对拍 200 对。

### 跑了什么验证（真实输出）

- 全量：`./.venv/bin/python -m pytest tests/` → **83 passed, 1 skipped**（本线 50 条全绿）。
- smoke 训练：`scripts/train_vqvae.py synth`（seed 0，6 clip）→ `train`（40 步默认超参）：
  loss 1.13 → 0.121；**eval 全量 6 clip：position_mean 0.275 m / rotation_mean 53.0°**
  （训练内 4 clip 报告为 0.279 m / 43.5°，旋转随 clip 组成波动）。
  阈值定 0.35 m / 60°（全量实测 +27% / +13% 余量，依据记录在脚本顶部注释）。
  默认 eval exit 0；`--position-threshold 0.0` 实测 exit 1——门禁双向可用。
- 复现性：同 seed 重跑（/tmp 实验 vs 最终 `checkpoints/smoke-claude-001`）指标逐位一致
  （0.2790 / 43.51），checkpoint 覆盖重训后 eval 结果不变。
- checkpoint 往返：`load_checkpoint` 重建的模型输出与训练完模型逐位一致（测试钉死）。

### 开发中抓到并修掉的三个真 bug（均有复现证据）

1. **MLX 优化器语义**：`optimizer.update(dict, grads)` 原地改 dict 并返回 `None`，模块不会自动看到新值——
   最初实现传完 `trainable_parameters()` 就没了下文，损失 300 步纹丝不动（过拟合单 batch 诊断抓到）。
   修：`optimizer.update(trainable, grads)` 后显式 `model.update(trainable)` 写回。
   顺带收益：只传 trainable 子集，AdamW 的 weight decay 碰不到冻结的 EMA 状态。
2. **测地角公式**：`trace(Ra^T Rb)` 展开是同下标逐元素求和、**不转置**；早先写成 `trace(Ra·Rb)`，
   2.07° 真值报成 123.7°（MSE 正常但旋转指标爆炸，正是这个不一致暴露了它）。
   修后与 `np.trace(Ra.T @ Rb)` 闭合式对拍 200/200，回归测试钉死。
3. **码本塌缩**：U(-1/K, 1/K) 初始化与 encoder 输出尺度差数量级，commitment 把 encoder 拖向近零码本，
   重建比全零基线还差（position 1.99 m，初始 grad 范数 137）。
   修：首批 `seed_from_encodings` 播种，修后初始 grad 2.13、ppl 35.9。

### 已知限制（不藏）

- **conv 感受野渗漏**：padded 输入仍会轻微影响邻近真实帧的重建（测试容忍 <1% 相对差并注明原因）。
  逐序列 batching 或重叠窗是后续修法。
- **短程训练码本使用率下降**：ppl 35.9 → 3.8（40 步内正常）；死码唤醒策略留给真数据阶段，
  ppl 曲线已在 history 里可查。
- **eval 在训练用 clip 上做**：这是"流水线仍在学"的冒烟门，不是泛化度量；真实数据到位必须换 held-out。

### 对契约的两处解释（需 A 确认；若不符，只需改 `training/layout.py` 一处）

1. `offsets` 解读为**前导 0 的块偏移表**：长度 = 块数+1，`D = offsets[-1]`。
2. `meta.json` 解读为 clip.npz 的**兄弟文件**（同目录同名）。
   建议（走工单"需要新增字段 + 原因"通道）：meta.json 增加 `contract_version` 字段，
   否则 v0 定稿后新旧 shard 无法区分。

### 需要协调 / 人类定界

- **VQ-VAE 落在 `training/` 而非 `models/`**（models/** 本轮由 A 线占用）。是否并入 `models/vqvae.py`
  由人类定，纯文件移动 + 一行 import。
- **eval 入口**是 `scripts/train_vqvae.py` 的子命令（工单只放行 `train_*.py` 文件名）；
  如需独立 `scripts/eval_*.py` 请放行。
- **导出对齐**（decision 0003）：自训权重最终要过 GGUF 边界给 motion-bricks.cpp。本脚手架的
  npz + manifest 只是训练侧格式，导出器是独立后续工作，设计时应把 manifest 超参当作导出器的输入。
- 等 A：`motion/representation.py` 落地后，`training/layout.py` 的转写常量换成 import，消除两处真源。

### 下一步建议

1. A 侧契约定稿 + 第一批真实 `clip.npz` → 用真实数据重跑同一 smoke 门（阈值需重校）。
2. 码本使用率监控与死码唤醒（ppl 已逐步记录）。
3. 逐序列 batching 消掉 padding 渗漏，再评估旋转指标上限。
