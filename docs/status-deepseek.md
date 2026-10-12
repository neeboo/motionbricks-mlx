# 状态 · DeepSeek（A：表示与数据）

- 当前：等待人类两个决定（Python 环境怎么建；根朝向是否保留完整姿态）——在此之前不动表示代码。
- 正在动的文件：无。
- 下一步：把 `docs/motion-representation.md` 落成 `motion/representation.py`（常量与编解码），
  并写 `scripts/export_from_space.py`，产出第一批真实 `clip.npz`。
- 环境：已就绪——仓库内 `.venv`（Python 3.13.12 + MLX 0.32.3 + NumPy 2.5.4，Metal 设备），
  见 `docs/decisions/0001-python-environment.md`；`scripts/check_env.py` 打 READY。
- 仍等人类一个决定：动作表示里根朝向是否保留完整姿态（v0 只有偏航）。
