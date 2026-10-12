# 状态 · DeepSeek（A：表示与数据）

- 当前：等待人类两个决定（Python 环境怎么建；根朝向是否保留完整姿态）——在此之前不动表示代码。
- 正在动的文件：无。
- 下一步：把 `docs/motion-representation.md` 落成 `motion/representation.py`（常量与编解码），
  并写 `scripts/export_from_space.py`，产出第一批真实 `clip.npz`。
- 环境：已就绪——仓库内 `.venv`（Python 3.13.12 + MLX 0.32.3 + NumPy 2.5.4，Metal 设备），
  见 `docs/decisions/0001-python-environment.md`；`scripts/check_env.py` 打 READY。
- 已查既有实现：见 `docs/decisions/0002-prior-art.md`——MotionBricks 已有 GGML 版（295★）与 Core ML 版，
  MLX 侧只有 VQ-VAE 零件、**没有动作生成**。定位据此改为"MLX 做训练与数据，产物导出给既有运行时或烘焙进引擎"。
- 仍等人类两个决定：① 下游运行时先接哪个（Core ML / GGML / 直接烘焙进 Unity）；② 根朝向是否保留完整姿态。
