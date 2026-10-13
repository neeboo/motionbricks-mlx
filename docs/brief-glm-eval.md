# 交给 GLM 的工单：动作质量评测与回归门

**工作目录**：`~/dev/motionbrick-mlx`。先读 `docs/status-glm.md`（分工规则）、`NOTICE`（只走 Apache-2.0 源码路线）、
`docs/decisions/0001-python-environment.md`（统一用 `./.venv/bin/python`）、
`docs/decisions/0003-downstream-runtime.md`（下游是 motion-bricks.cpp，GGUF 是边界）、
`docs/motion-representation.md`（动作表示契约）。

## 你只写这些文件（与另外两条线零重叠）
- `src/motionbricks_mlx/eval/**`（新建）
- `scripts/eval_*.py`
- `tests/test_eval_*.py`
- `docs/status-glm.md`（你自己的状态文件）

**绝对不要碰**：`src/motionbricks_mlx/{models,data,export,motion,training}/**`、
`scripts/mlx_*.py`、`scripts/train_*.py`、`docs/motion-representation.md`、`README.md`、
`NOTICE`、`LICENSE`、`pyproject.toml`；参考仓库 `~/dev/motion-bricks-cpp` 只读。

## 目标：把"动作好不好"变成数字，并做成门禁

现在我们有两条都"数值对齐"的实现（MLX 与官方），但**没有一个能回答"这段动作好不好"**。
本轮量化（q8/q4）与将来的批量烘焙都要靠这个门禁做决定——没有它，"量化值不值得用"只能靠感觉。

**输入形状**（来自 `export/plan_json.py` 与参考 JSON）：每步
`motion{frames, joints:34, roots[frames,3], rotations[frames,34,4], world_positions_xyz[...]}` 与
`targets{frames:4, ...}`。参考样例：`~/dev/motion-bricks-cpp/artifacts/plan-walk.json`（只读）。

## 指标（每条给精确算法与单位，不许含糊）
1. **脚滑**：接触帧内脚部世界坐标的累计位移（米）；接触判定用"最低脚点高度 < 阈值"。
2. **抖动**：关节角序列的二阶差分（加加速度）分布，报 p50/p95/max。
3. **接地**：全序列中"穿地"（低于地面）与"悬空"（高于阈值）帧的占比，以及最低脚点高度分布。
4. **首尾衔接**：边界帧与 `targets` 的误差（位置米 / 旋转分量）。
5. **跨实现一致性**：同 seed 下两条实现的**离散 token 一致率**（复用 `pose.tokens` 口径）与最终动作偏差。
6. **结构校验**：帧数、关节数、fps、四元数归一化、NaN/Inf 检查。

## 门禁与阈值
- 阈值与基线**一起提交**（基线取自 f32 MLX 与参考 JSON 的"好动作"），写在 `src/motionbricks_mlx/eval/thresholds.py`（或 `.yaml`）。
- `scripts/eval_gate.py` 跑一套固定场景（**≥5 组 seed**），输出每个指标的值与 **PASS/FAIL**。
- **必须做负对照**，并且要能证明门禁不是恒真：①帧序打乱 ②连续若干帧冻结 ③整体加抖动噪声 ④把脚抬高悬空
  ⑤把根位移整体平移 —— 每种都要让**对应指标越界变红**，并贴证据。
- 阈值来源要写清理由（例如同时报告 f32 基线与参考自身 Debug↔Release 的差异量级）。

## 交付与验收
1. `scripts/eval_gate.py` 在 f32 MLX 输出与参考 JSON 上**全绿**；在每种"坏动作"上**对应项变红**。
2. `docs/status-glm.md`：做了什么、跑了什么验证、数字、下一步、阻塞。
3. `tests/test_eval_*.py`：指标函数的单元测试 + 一次门禁的端到端测试；`./.venv/bin/python -m pytest -q` 不许让现有 **33 passed/1 skipped** 变差。
4. 报告里给一句**诚实结论**：这些指标在当前数据上**能不能区分好坏**，哪些指标其实没有分辨力（有就直说）。

## 红线
禁止合成输入（CGEvent/AX/osascript）；不点界面、不抢焦点、不播放音频、不改系统设置；
不做长训练（只允许极小规模冒烟）；不 push 任何远端（本地 commit 可以）；不改别人的文件。
