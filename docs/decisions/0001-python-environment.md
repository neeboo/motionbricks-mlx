# 0001 · Python 环境（人类授权我决定）

**决定**：本仓库统一用**仓库内的 `.venv`**，由 Homebrew 的 **python@3.13** 创建；两个 agent 与人类
**只用** `./.venv/bin/python`，不各自另建环境。

**实测版本**（2026-10-12，Apple Silicon / Darwin 26.5.2）：

| 组件 | 版本 |
|---|---|
| Python | 3.13.12（`/opt/homebrew/bin/python3.13`） |
| MLX | 0.32.3 |
| NumPy | 2.5.4 |
| 设备 | `Device(gpu, 0)`（Metal） |

**重建命令**（换机器或删了 `.venv` 时）：

```sh
cd ~/dev/motionbricks-mlx
python3.13 -m venv .venv
./.venv/bin/python -m pip install -U pip mlx numpy pytest
./.venv/bin/python scripts/check_env.py      # 期望最后一行 READY
```

**理由**：本机系统自带的是 3.9（MLX 要 3.11+）；Homebrew 里 3.11–3.14 都在，选 3.13 是因为它在本机已验证
能装到 MLX 0.32.3 并跑通 Metal；`.venv` 放进仓库是为了让两个 agent 指向**同一个解释器**，
避免"你装了一套、我装了一套"这种和今天同类的分叉。

**约束**：`.venv/` 不进版本库（已在 `.gitignore`）；依赖清单的增删只改 `pyproject.toml`（冻结区）；
跑长训练前必须在自己的状态文件里声明。

**证据**：`./.venv/bin/python scripts/check_env.py` → `READY`；一次矩阵冒烟
`mx.sum(arange(6).reshape(2,3))` = **15.0**；`mx.default_device()` = `Device(gpu, 0)`。
