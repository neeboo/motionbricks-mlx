# 状态 · Claude（B：模型与训练）

- 当前：未开工（第 2 步依赖 A 的数据契约落地）。
- 正在动的文件：无。
- 前置：`docs/motion-representation.md` 定稿、`motion/representation.py` 导出 `FRAME_DIM` / `OFFSETS`。
- 开工顺序建议：先写 VQ-VAE 的前向与形状测试（不依赖数据），等真实 `clip.npz` 到位再接训练循环。
