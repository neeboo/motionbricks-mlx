# 0003 · 下游运行时选 motion-bricks.cpp

**决定**（人类倾向 + 我核实后同意）：本仓库不做推理运行时，产物交给
[`localai-org/motion-bricks.cpp`](https://github.com/localai-org/motion-bricks.cpp)。

## 核实到的事实（2026-10-12，API + README）

| 项 | 值 |
|---|---|
| 许可证 | **Apache-2.0**（与上游源码一致，可自由使用/内嵌） |
| 语言 / 后端 | C++ / **GGML**（`ggml` 为子模块），CMake + presets，含 ASAN/UBSAN preset 与 tests |
| 星 / 分叉 | 295 / 33；创建 2026-09-02，最后推送 2026-09-21 |
| 权重格式 | **GGUF**；`scripts/convert_to_gguf.py` 是把参考实现权重转过来的入口，另有 `download_gguf_weights.py` 与 style bundles |
| 可用面 | `libmotionbricks`（C++ 库）＋ `motionbricks-cli`（含 `abi` / `inspect` 子命令）＋ Go demo ＋ bindings ＋ examples |
| 定位自述 | "ported to GGML/C++; dynamically generates animations from keyframes… can be used in games or with robots" |

## 它为什么合适

1. **许可证干净**：Apache-2.0，和上游源码同一条线，不牵涉权重许可（我们用自训权重）。
2. **GGUF 就是我们要的边界**：我们只负责"训练 → 导出成它认的 GGUF bundle"，接口可验证
   （它自带 `inspect`，能直接读回 bundle 结构）。
3. **为游戏而做**：C++ 库 + CLI + bindings，不是论文 demo；可以直接被产品进程链进来。
4. 与 0002 的定位自洽：**MLX 做训练与数据，它做推理**，两边职责不重叠。

## 必须先确认的两件事（第 1 步之后立刻做）

1. **macOS / Apple Silicon 预设**：README 的 CPU 预设与 AVX512 读数偏 x86；本机要走 Metal/Accelerate，
   需要确认存在可用 preset 并实测跑通它的 bundle（若无，就得自己加 CMake preset，这是我们要出的人力）。
2. **GGUF schema 对齐**：`convert_to_gguf.py` 面向**上游 PyTorch 参考实现**的模块布局。
   我们若训练自己的变体，要么保持同样的模块划分，要么扩展它的转换脚本——**这条约束要写进动作模型的
   设计里**，不能等训练完才发现导不出去。

## 明确不做的

* 不自己写 GGML/C++ 推理层（已有）；
* 不转换、不再分发 NVIDIA 预训练权重（只导出我们自训的权重，见 `NOTICE`）；
* 不把 Core ML 路线当主线（作为备选保留：若本机 GGML+Metal 不通，再评估 `cg25/motionbricks-coreml`）。
