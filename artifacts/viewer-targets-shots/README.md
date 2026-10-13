# `artifacts/viewer-targets-shots` —— 第 9 轮（用户摆目标关键帧）的入库画面

这一份只说明**哪些文件进了版本库、哪些故意没进**。拍摄方法（无头 Chrome，REA
`capture_browser_scenario`，**只有 `goto` + `wait_for_timeout`**，无合成输入、未点界面、未抢焦点、
无音频；每张 PNG 用 base64 解码后重算 `sha256` 与证据比对）在
[`docs/VIEWER.md`](../../docs/VIEWER.md) 的「追加（第 9 轮）」§5；命中度实测与包络在同文件 §3。

**入库**（1 个文件，317 kB）：

| 文件 | 体积 | sha256（前 16） | 是什么 / 证明了什么 |
|---|---|---|---|
| `final-900-settle.png` | 324,300 B | `b9eb44fd29842dfa` | **主图，且是在最终代码那一版服务器上拍的**（1440×900）：橙色目标幽灵站在橙色圆环上（目标 `(0, 3)`），绿色角色走到半途，地面轨迹连起来；HUD `TARGET 0.111 m`、`CLIPS 1`。同一张的 DOM 属性 `data-mlx-target-hit=0.111320`、`data-mlx-target-hit-ok=true`、`data-mlx-target-window=40-43` 记在 `artifacts/viewer-targets-run/final-page-attributes.txt`。它证明**摆下去的目标关键帧真的被命中**，而且命中误差是页面自己报出来的数。 |

**没入库**（同轮其余 3 张，为控制体量留在本机）：

| 文件 | 为什么不留 |
|---|---|
| `tall-1180-settle.png`（1440×1180，侧栏目标面板整块可见） | 只是换个窗口高度的同一画面；正文没有引用。 |
| `live-off-settle.png` | 服务器还差最后一处**只影响"不摆目标"分支**的追加式改动时拍的上一版，已被主图取代。 |
| `targets-settle.png` | 第一次会话，页面当时回 400 `unsupported target [0, 3, 0]`（`docs/VIEWER.md` 第 9 轮 §6 坑 1），是失败态留档。 |

`artifacts/viewer-targets-run/`（包络表、`curl` 记录、DOM 属性、`server.log`）**整目录不入库**。
