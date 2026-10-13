# `artifacts/switch-shots` —— 第 10 轮（A→B→A 切换）的入库画面

这一份只说明**哪些文件进了版本库、哪些故意没进**。拍摄方法、逐张 `sha256`、
每张 HUD 上的数字，以及两张主表的原始出处都在
[`docs/VIEWER.md`](../../docs/VIEWER.md) 的「追加（第 10 轮）」§6（画面）与 §3（对照表）；
本目录与 `artifacts/switch-run/` 是同一场会话的产物，那里有脚本与 `report.json`。

**入库**（4 个文件，共 2.64 MiB）：

| 文件 | 体积 | sha256（前 16） | 是什么 / 证明了什么 |
|---|---|---|---|
| `aba.gif` | 1,711,070 B（1.63 MiB） | `b1c61ed376f6d1d5` | **主画面**：A→B→A 整段 84 帧的 GIF（每 2 帧取 1 = 42 帧，560×384，70 ms/帧，2.94 s）。底部字幕带写明当前在哪条腿（`walk` / `walk_zombie`），两处接缝有青色竖标。它证明**切换是页面上的真实连续动作**，不是两段视频拼接。 |
| `live-switch-1440x900.png` | 341,514 B | `0dce5b29b6b5a874` | **现场切换**（`?switch=walk_zombie&at=20&t=20`）：HUD `SEAM GAP 0.17 mm`、`SEAM JUMP 1.66×`、`ROOT Z 0.88 m`，画面里一根青色接缝柱。 |
| `hardcut-control-1440x900.png` | 330,213 B | `f6e07703b17c71b2` | **硬切对照臂**（同一 URL + `?naive=1`）：`SEAM GAP —`、`SEAM JUMP —`、**没有接缝柱**、`ROOT Z 0.30 m`。 |
| `seam1-ab-1440x900.png` | 380,514 B | `015c85ee797825da` | **接缝静帧**（`?t=20`）：HUD `STYLE walk → walk_zombie → walk`、`SEAM GAP 0.28 mm`、`SEAM JUMP 1.66×`，两条青色接缝柱与地面圆环都在画面里。 |

`live-switch` 与 `hardcut-control` 是**同一时刻、同一相机、只差一个 `naive` 参数**的一对
（`docs/VIEWER.md` 第 10 轮 §6 末）：切换后 20 帧，角色在 z = 0.88 m 还是 z = 0.30 m，差 0.57 m。

**没入库**（同一次会话的其余画面，为控制体量留在本机）：

| 文件 | 为什么不留 |
|---|---|
| `aba-walked-1440x900.png`、`seam2-ba-1440x900.png`、`tall-1440x1180.png` | 与上面同一条 `--screenshot` 命令、换 `?t=` 参数或换窗口高度即可重拍；正文只引用上表这几张。 |
| `../switch-run/**` | 整目录都是本机产物：`report.json` / `switch-aba.json` 是本轮两张表的原始出处，`gif-frames/`（约 9 MB 中间帧）、`shoot.sh`、`gif.py`、`server.log`、`server.pid` 可逐字节再生。按第 7 轮 `artifacts/bake-2026-10-12/README.md` 的同一套取舍，**整目录不入库**；本目录的 `aba.gif` 是它唯一入库存下的最终画面。 |
