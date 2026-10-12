# 0002 · 先查有没有人做过（结论：做了，但不是 MLX）

**检索时间** 2026-10-12，方法：`gh search repos`（带 token 的仓库检索）。

## 找到的既有实现

| 仓库 | 星 | 形态 | 说明 |
|---|---|---|---|
| `localai-org/motion-bricks.cpp` | 295 | **GGML / C++** | 把 MotionBricks 移植到 GGML，从关键帧动态生成动画；Metal 后端可跑 |
| `cg25/motionbricks-coreml` | 0 | **Swift + Core ML** | Apple 平台原生运行时（推理） |
| `SarthakKalla/GR00T-Unreal-Plugin` | 3 | Unreal 插件 | MotionBricks + GEAR-SONIC 接进 UE 5.4–5.8 |
| `zeljkovranjes/sbox-motionbricks-weights` | 0 | 权重包 | 给 s&box 用的编译权重（再分发权重，注意模型许可） |
| `Aero-Ex/Nvidia_MotionBricks` | 13 | 镜像/复刻 | — |
| `deboradum/mlx-vqvae`, `mlx-vqvae2` | 0 | **MLX** | 只是 VQ-VAE 零件，不是动作生成 |

检索 `mlx motion` / `mlx human motion generation` / `mlx animation` / `mlx robotics` / `mlx text to motion` 全部**无结果**。

## 结论与调整

1. **"移植架构"已经被人做过两遍**（GGML 版、Core ML 版），我们**不再把它当主要工作**。
2. **MLX 那条确实是空的**，但空的是**推理运行时**吗？不是——Core ML 版已经占了 Apple 原生的推理位置。
   **真正空着的是训练栈**：GGML 与 Core ML 两个移植都只做推理，不训练；MLX 在 Apple 芯片上做训练/微调
   才是没有现成替代的那一格。
3. 所以本仓库的定位改为：**在 MLX 上训练我们自己的动作模型（数据由我们自己的空间生成），
   产物导出给既有推理运行时或直接烘焙进引擎**。既有实现从"竞争者"变成"下游目标"。
4. 五步计划不变，但第 5 步（端上实时）的默认答案改为**接既有 Core ML 运行时**，而不是自己蒸馏一套。

## 待人类决定

* 下游运行时选哪个：**Core ML 版**（Apple 原生、Swift）／**GGML 版**（跨平台、295★、活跃）／
  **直接烘焙进 Unity**（我们引擎已支持 VRMA，运行时零成本）。三者不冲突，但第一个要接的先定一个。
* 是否允许我们**只做训练与数据**、把移植工作让给上面这些仓库（我建议：是）。
