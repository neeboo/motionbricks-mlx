# `artifacts/viewer-shots` —— 第 8 轮（查看器）的入库画面

这一份只说明**哪些文件进了版本库、哪些故意没进**。拍摄方法（无头 Chrome，REA
`capture_browser_scenario`，**只有 `goto` + `wait_for_timeout`**，无合成输入、未点界面、
未抢焦点、无音频）与逐张 `sha256` 在 [`docs/VIEWER.md`](../../docs/VIEWER.md) §7；
起停、协议与性能数字在同文件 §1–§4。

**入库**（1 个文件，273 kB）：

| 文件 | 体积 | sha256（前 16） | 是什么 / 证明了什么 |
|---|---|---|---|
| `viewer-final.png` | 279,795 B | `f0f706960a1300b6` | **主图，浏览器里的真实 WebGL 画面**（1440×900）：`STYLE walk`、`CLIPS 11`、`WALKED 10.43 m`、`ROOT Z −7.88 m`、`W` 键高亮、地面轨迹可见。它证明页面上的动作**是这台机器上的 MLX 现场算的**，不是另写的渲染脚本替代品。 |

**没入库**（同一次会话的其余 9 张帧，为控制体量留在本机）：
`viewer-0-scenario_start.png`、`viewer-1-boot.png`、`viewer-2-walk.png`、
`viewer-hud-*-*.png`、`viewer-session-*-*.png`、`viewer-walk-1440x900-*-*.png` —— 同一条
`goto` + `wait_for_timeout` 命令换个时刻即可重拍（`docs/VIEWER.md` §7 有逐张说明）。
探针脚本与 plan 文档留在 `artifacts/viewer-run/`，那个目录**整目录不入库**。
