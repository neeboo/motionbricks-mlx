# VIEWER · 在浏览器里看 MLX 版的动作

**一句话**：一条命令起一个只绑 `127.0.0.1` 的本地页面，页面上每一段动作都是**这台机器上的
MLX 规划**现场算出来的，不是预录。

```
http://127.0.0.1:8848/
```

页面长这样（1440×900，无头 Chrome 实拍，不是渲染脚本替代品）：

![MLX viewer](artifacts/viewer-shots/viewer-final.png)

同一批截图的其它帧：`artifacts/viewer-shots/viewer-hud-1-walk.png`（HUD 首次出现）、
`viewer-walk-1440x900-1-walk.png`。三张都带 `sha256` 校验（见 §7）。

---

## 1. 起与停

```bash
cd ~/dev/motionbrick-mlx
./.venv/bin/python scripts/mlx_viewer.py --port 8848 --device gpu
```

本轮实际是**受管后台任务**起的（不是 `nohup`）：

| 项 | 值 |
|---|---|
| URL | `http://127.0.0.1:8848/` |
| PID | `78447`（也写在 `artifacts/viewer-run/server.pid`） |
| 日志 | `artifacts/viewer-run/server.log` |
| 绑定 | `127.0.0.1:8848`（`--host` 只接受 `127.0.0.1`，其它值直接 `parser.error`） |

**停止方式**（任选一种）：

```bash
kill "$(cat artifacts/viewer-run/server.pid)"     # 干净退出，serve_forever 收 KeyboardInterrupt
# 或者若还在受管任务里：job_kill <job_id>
```

启动时它自己打的日志（`artifacts/viewer-run/server.log` 开头）：

```
loading /Users/ghostcorn/dev/motionbrick-mlx/checkpoints/g1-f32-mlx on mlx-gpu ...
ready: cold start 370.7 ms, 15 styles, warm clip 44 frames
viewer: http://127.0.0.1:8848/  (pid 78447)
```

参数：`--port`（默认 8848）、`--device {gpu,cpu}`、`--bundle`、`--styles`、`--quiet`。
`--styles` 默认指向 `~/dev/motion-bricks-cpp/repo/generated/styles`（只读使用，不写）。

**为什么是单线程**：MLX 的 GPU stream 是 thread-local 的，`ThreadingHTTPServer`
的 worker 里跑 `Agent.plan` 会直接把进程 abort 掉：

```
libc++abi: terminating due to uncaught exception of type std::runtime_error:
There is no Stream(gpu, 0) in current thread.
```

所以服务器故意用 `http.server.HTTPServer`（单线程）+ 每个响应 `Connection: close`。
本来也只有一个有状态的 agent，串行规划就是对的形状。

---

## 2. 协议（自定，不兼容参考版的 WebSocket）

参考版的运动指令只走 WebSocket（`GET /api/stream/socket`），HTTP 的 `/api/session`、
`/api/plan`、`/api/kimodo/start`、`/api/kimodo/finish` 一律 `410`
（`demo/stream.go:152-158`，本轮已在只读 checkout 里核对）。我们把传输换成了最笨的
**一次请求 = 一次规划** 的 JSON，这样 `curl` 和页面走的是同一条路。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | `viewer/index.html` |
| `GET` | `/style.css` `/app.js` `/rig.js` `/vendor/*` | 静态资源，只从 `viewer/` 里取（路径守卫见 `tests/test_viewer_server.py`） |
| `GET` | `/api/health` | 后端、参数量、冷启动、累计计数 |
| `GET` | `/api/styles` | 真实 `*.mbstyle` 清单 + 骨架（关节名/父节点/静止位置） |
| `GET` | `/api/plan?...` | 查询串形式的规划（`curl` 用） |
| `POST` | `/api/plan` | JSON 形式（页面用） |
| `POST` | `/api/client-metrics` | 页面把自己的加载/首帧耗时回报给服务器 |
| `GET` | `/api/stats` | 服务器滚动计时 + 最近一次页面自报的耗时 |

`/api/plan` 参数（除 `style` 外都可省）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `style` | 清单默认（`walk`） | 必须是 `/api/styles` 里的名字，否则 404 |
| `seed` | `1` | 采样种子 |
| `move` | `0,1` | 世界系移动方向，`"x,z"`（查询串）或 `[x,z]`（JSON） |
| `facing` | `0,1` | 世界系朝向 |
| `reset` | `0` | `1` 表示从 style 重新播种 agent |
| `advance` | `0` | **连贯性的接缝**，见下 |
| `argmax` | `0` | 诊断用的 argmax 采样 |

`/api/styles` 列的 15 个真实 style，各触发一次（都 `reset=1`，即"新建 agent + 重播种 +
规划"）的实测代价：

| style | frames | plan_ms | travel |
|---|---|---|---|
| `walk` | 44 | 27.8 | 1.097 m |
| `walk_zombie` | 44 | 26.1 | 0.710 m |
| `idle` | 24 | 39.3 | 0.007 m |
| `stealth_walk` | 32 | 101.5 | 1.028 m |

换成某个 style 后的**第一拍**可能明显慢（`stealth_walk` 101.5 ms：新 agent + 新上下文 +
一次冷图），稳态回到 ~28 ms。`idle` 几乎不位移是对的。

响应（节选真实字段）：

```json
{"ok":true,"style":"walk","frames":44,"joints":34,"fps":30,
 "world_positions_xyz":[…4488 个数…],
 "targets":{"frames":4,"joints":34,"world_positions_xyz":[…408 个数…]},
 "metrics":{"root_travel_metres":1.0915,"duration_seconds":1.4667,
            "root_first_frame":[…],"root_last_frame":[…]},
 "timing":{"plan_ms":29.0,"fk_ms":2.5,"server_total_ms":31.5}}
```

### `advance` 为什么是它

`Agent.advance(n)` 把规划上下文挪到当前动作的第 `n` 帧；`restore_world` 再把新动作
放回那个世界位姿，所以新片段的第 0 帧就是页面已经显示到的那一帧。客户端
`playhead -= advance` 就够了——这正是参考版 `requestPlan()` 里的
`replacementPlayhead = playhead - advance`。实测（`artifacts/viewer-run/continuity.txt`）：

```
clip 2: advance= 33  pelvis(clip1, f33)=[0.000, 0.769, 0.775]
          pelvis(clip2, f 0)=[0.000, 0.770, 0.775]   gap=4.7e-04 m
clip 3: … z 1.760   gap=5.4e-04 m
clip 4: … z 2.706   gap=5.2e-04 m
clip 5: … z 3.666   gap=4.4e-04 m
```

接缝误差 ≤ 0.55 mm（就是上下文混合本身），世界位移在连续累积——角色是真的在走，
不是原地踏步。

---

## 3. 页面上有什么

**风格**：`/api/styles` 直接列磁盘上的 15 个真实 `.mbstyle`
（`idle`、`walk`、`walk_zombie`、`stealth_walk`、`walk_scared`、`elbow_crawling`、
`hand_crawling`、`injured_walk`、`slow_walk`、`walk_boxing`、`walk_gun`、
`walk_happy_dance`、`walk_left`、`walk_right`、`walk_stealth`），换一个就重播种重规划。

**播放控制**：Pause / Play、Reset camera（双击画布同样有效）、Ground trail、
Placed target keyframes、Replan while walking 开关。相机是参考版那套
（拖拽 orbit、滚轮 zoom、yaw/pitch/distance 默认值一致）。

**"改一下再看"**：四种，都会立即触发一次新规划——

1. 换 style（下拉框）；
2. 换 seed（数字框或 `Next seed`）；
3. 走/停：W/A/S/D 方向盘（再点一次同一个键就停），方向是**相机相对**的，
   和参考版一致；`Stop` 按钮直接归零；
4. 转身：`↺ Turn` / `Turn ↻` 只改朝向不改移动，可以原地拐弯。

**键盘**：W/A/S/D 同方向盘，空格或 `J` 播放/暂停。页面不用鼠标也能玩。

**HUD**（画面左下角，所以数字和动画在同一帧、同一张截图里）：
`style / plan ms / step rate / walked m / root z / clips`。

**URL 参数**：`?style=walk_zombie&seed=7&move=0,1` 可以在地址栏里预置一次会话。
这是本轮唯一"从外部驱动页面"的方式，它只是地址栏参数，不是合成输入。

页面自己会回报耗时到 `/api/client-metrics`，所以"页面加载到可用"这种只存在于
浏览器里的数字也能用 `curl /api/stats` 读到。

---

## 4. 性能（实测）

机器不空闲：本轮测量期间 1 分钟 load average **8.1–18.0**（桌面进程 mihomo、
WindowServer、Chrome、ChatGPT/Codex、DSH renderer；没有训练/评测）。
按 [SUMMARY.md](SUMMARY.md) 第 5 节的同一修正，绝对值被抬高约 10%。

| 指标 | 实测 | 怎么量的 |
|---|---|---|
| **冷启动**（载入权重 + 预热一次规划） | **370.7 ms**（本轮三次启动 317.4 / 324.8 / 370.7；其中 324.8 那次随后在并发规划请求上崩了，见 §1 的单线程说明——崩的是请求路径，不是启动） | `artifacts/viewer-run/server.log`，也由 `/api/health.cold_start_ms` 返回 |
| **首帧**（页面 start → 第一帧动作上屏） | **84.4 / 117.3 / 129.9 ms** | 页面用 `performance.now()` 自量，POST 回来，`curl /api/stats` 可读 |
| 其中"页面可用"（脚本就绪 + 拿到 styles） | **17.0 / 39.6 / 42.1 ms** | 同上（`ready_ms`） |
| **一次会话的首个 clip**（`reset=1`：新建 agent + 重新播种 + 规划 + FK + JSON） | 服务端 **63.0 ms**，请求 83.0 ms（含建连）；换 style 的第一拍实测到 **101.5 ms**（`stealth_walk`） | `artifacts/viewer-run/probe-30.json` 与 §2 的换 style 表 |
| **连续步频**（服务端规划 p50） | **28.3 ms → 27.9 clips/s** | 同上，30 个串行 clip |
| 其中 FK（世界坐标） | **2.5 ms**（p50） | 同上 |
| HTTP 往返（p50，含 ~100 KB JSON） | **34.8 ms** | 同上，客户端侧墙钟 |
| p90 / max | 34.6 ms / 45.9 ms（服务端） | 同上 |
| 等效帧吞吐 | **1227 帧/秒** | 30 clips × 44 帧 / 1.075 s |
| 相对 30 fps 播放的余量 | **40.9×** | 一个 44 帧 clip 播 1.47 s，规划只要 34.8 ms |

服务端自己的滚动计数（`artifacts/viewer-run/evidence-snapshot.txt`，含所有浏览器会话）：

```
plans 98  plan_ms_p50 33.5  plan_ms_p90 63.0  plan_ms_max 101.5  steps_per_second_p50 29.8
client:  first_pose_ms 84.4   ready_ms 17.0   device mlx-gpu
```

（这一条比 probe 高，因为它把每次页面冷启动的 `reset=1`、每次换 style 的第一拍
（`max 101.5` 就是 `stealth_walk`）都算了进去。probe 那条才是稳态。）
`client` 只保留**最后一次**页面自报——上面那三个首帧数字是三次会话各读一次得到的。

### 和参考版对照

| | 参考版 GGML（M1 Max） | 本页（MLX，同机） |
|---|---|---|
| 单步 p50 | **≈380 ms**（**任务书给的值**） | **28.3 ms**（本轮 probe，load 12.9） |
| 比值 | 1× | **≈13.4×** |

**诚实边界**：那个 380 ms 我**没有**在只读的参考 checkout 里找到出处——`docs/*.md`
里 grep 不到任何 p50 表，`demo/` 里也没有延迟常量。所以它是**引用任务书的值**，
不是我复测的；本页这一列是我复测的。两边的口径也不完全一样：参考版那个是
GGML 引擎的推理步，本页这个是"一次 HTTP 规划调用（含 FK 与 JSON）"。
真正同口径、可复现的对照在本仓库自己的文档里：默认路径 GPU 单场景规划
**p50 26.6 ms**（[SUMMARY.md](SUMMARY.md) 第 1 节），本页的服务端 p50 28.3 ms
与它一致（差值就是机器负载）。

---

## 5. 复用了什么、归属怎么标

**参考仓库 `~/dev/motion-bricks-cpp/repo` 全程只读，一个字节没改。** 复用清单：

| 我们的文件 | 来源 | 改动 | 校验 |
|---|---|---|---|
| `viewer/vendor/three.module.min.js` | `demo/web/vendor/` | **逐字节复制** | sha256 `852c06b9…d2af` 两边相同 |
| `viewer/vendor/three.core.min.js` | `demo/web/vendor/` | **逐字节复制** | sha256 `4183cee0…ba82` 两边相同 |
| `viewer/vendor/THREE-LICENSE.txt` | `demo/web/vendor/` | 逐字节复制 | three.js 本体是 MIT |
| `viewer/rig.js`（骨架 rig、label、画线） | `demo/web/app.js` 第 88-95、97-113、115-252、254-272 行 | **逐字复制**；唯一机械改动是把参考版从 app.js 读的模块级 `scene` 单例改成 `bindScene()` 注入 | 四段都在文件里原样存在（逐块 sha256 见下） |
| `viewer/style.css` 前 63 行 | `demo/web/style.css` | **逐字复制** | 前 3073 字节 sha256 `bf95d13b…1e9a` 与参考版相同 |
| `viewer/index.html`、`viewer/app.js`、`style.css` 末尾追加块、`scripts/mlx_viewer.py` | 本仓库新写 | — | — |

`rig.js` 四段的逐块 sha256（前 16 位）：

```
app.js  88-95   d029679cc7c238de   present_verbatim=True
app.js  97-113  3c6be0ce8c7aa442   present_verbatim=True
app.js 115-252  f91330c29efa6cb1   present_verbatim=True
app.js 254-272  ff5e7c4a2d565a56   present_verbatim=True
```

`rig.js` 209 行里只有 32 行是新增的（版权/来源注释、`bindScene`、`export`）。

**归属怎么写**（三段，都在文件里）：

* `viewer/rig.js` 顶部：写明 Apache-2.0、来源 `demo/web/app.js`、原始行号、
  唯一改动点；指向本仓库的 `LICENSE` / `NOTICE`。
* `viewer/style.css` 追加块顶部：写明上半部分逐字来自参考版 Apache-2.0 的
  `style.css`，下半部分只覆盖新增控件、沿用同一套配色。
* `viewer/app.js` 顶部：写明 three.js / rig / css 三个来源，以及**本文件是本仓库
  新写的**（只是刻意沿用参考版的场景与相机约定），数据通路换成了 HTTP+JSON。

**唯一真正的接缝**：参考版 `SkeletonRig.poseWorld(positions, offset)` 吃的就是
`frames × joints × 3` 的展平 `world_positions_xyz`——那正好是本仓库导出器
`motionbricks_mlx.export.plan_json` 已经在产出的数组。所以渲染代码一行都不用适配，
只需要换一个生产者。这是这次复用便宜的全部原因。

---

## 6. 服务器里那个"自己的实现"

`scripts/mlx_viewer.py` 只加了一处算法性的东西：`world_positions()`，把
`plan_json.world_positions_xyz` 的双层 Python 循环改成"保留关节循环、批量化帧轴"。

* 逐位相同：`tests/test_viewer_fk.py` 用随机四元数、极端量级和零四元数断言
  `np.array_equal`（不是近似），真机上另测了 4 个真实 style 的动作与目标。
* 为什么值得：参考循环形态 **14.9 ms/clip**，批量化 **1.34 ms/clip**（11×）。
  在 28 ms 的规划旁边这不是零头。

页面侧没有任何复制粘贴的物理/数学：它只画服务器给的 `world_positions_xyz`。

---

## 7. 截图是怎么拿到的

用 Chrome（`/Applications/Google Chrome.app`，无头）通过 REA 的
`capture_browser_scenario` 打开 `http://127.0.0.1:8848/`，**只做 `goto` +
`wait_for_timeout`**——没有合成输入、没有点击界面、没抢焦点、没播放音频。
动作清单里唯一的两个 step 就是等待。

驱动的都是 HTTP：页面自己加载、自己 `/api/plan`、自己 `/api/client-metrics`。
本轮一共 4 次浏览器会话：第 1 次每个 `requestAnimationFrame` 报一次
`ReferenceError`（证据里记到 event sequence 438+，见 §8 的坑 1），修好之后
第 2–4 次的 `page-errors` 计数都是 **0**。

提取出来的 PNG 都逐张用 sha256 校验过（base64 解码后重算，必须等于证据里的
`sha256`）：

| 文件 | sha256（前 16） | 说明 |
|---|---|---|
| `artifacts/viewer-shots/viewer-final.png` | `f0f706960a1300b6` | **主图**：walk、CLIPS 11、WALKED 10.43 m、ROOT Z −7.88 m、W 键高亮、地面轨迹可见 |
| `viewer-hud-1-walk.png` | `4a6842a04d481a3a` | HUD 第一次出现；当时 move 被初始 `updateControl()` 清零，正在原地走（见 §8） |
| `viewer-walk-1440x900-1-walk.png` | `41a7a21eb065d4a4` | 1280×720 那次之后的中景 |
| `viewer-2-walk.png` | `018f5ff4585d5902` | 1280×720 首版 |

**为什么不是"替代渲染"**：上面这些是浏览器里真实的 WebGL 画面，不是我另外写的渲染
脚本。所以"页面到底能不能看"这件事没有靠替代品蒙过去。

---

## 8. 中间踩到的两个坑（都修了）

1. **`startPoint is not defined`**（第一次浏览器会话，每个 rAF 一次 `ReferenceError`）。
   抄 `app.js` 的 rig 时漏了第 91-95 行的向量暂存变量（`up`/`startPoint`/`endPoint`/
   `direction`/`midpoint`）。修法是把复制范围改成 88-95，并加了"四段逐字存在"的
   断言（§5 的表）。修完 `page-errors` 归零。
2. **角色原地踏步**。`buildRigs()` 之前那句 `updateControl()` 看到没按任何键，把
   `state.move` 归零了——所以页面一直在规划 `move=[0,0]`，HUD 上 `WALKED` 才
   0.86 m。修法是把初始状态设成"W 键按住"：`padKey: 'w'`，`updateControl()` 照常
   按相机相对算出 move/facing，W 按钮也自然高亮。修完主图 `WALKED 10.43 m`、
   `ROOT Z −7.88 m`。

第 2 个坑正好说明为什么 HUD 要放进画面里：如果数字只在侧栏，我不会在一张
截图上看出来"它在走但没有位移"。

---

## 9. 诚实边界

1. **一个进程 = 一个 agent。** 服务器只有一份有状态 agent，多个客户端会互相打乱
   上下文。本轮所有测量都是单客户端串行做的。当成本地单人查看器用没问题，
   要多人用就得给 agent 加会话键。
2. **`advance` 是近似的连贯性，不是无缝。** 接缝处有 ≤0.55 mm 的位姿差（上下文
   混合本身造成的），不是逐位连续。
3. **首帧那三个数不是同一次会话。** 84.4 / 117.3 / 129.9 ms 来自三次不同的浏览器
   会话，机器负载也不同。我给的是三次的分布，不是取最好那个。
4. **参考版 380 ms 是引用值**，不是我在只读 checkout 里复测或找到出处的
   （§4 已说明）。
5. **冷启动只算"载入权重 + 预热一次规划"**，不含 Python 解释器启动（那是
   `mlx_plan.py --time` 的量法）。用户真正感知的是"敲下命令到页面能看"，那要再加
   解释器启动与浏览器加载，本轮**没有**端到端量这个（浏览器加载那段由
   `ready_ms`/`first_pose_ms` 覆盖，但没和命令行接起来）。
6. **没有做**：GLB 上传、Kimodo 回放、物理/碰撞、WASD 之外的键盘映射、
   参考版的多会话/对比布局、任何形式的鉴权或远程绑定。参考版有、我们没有。
7. **相机只跟随骨架**，不做参考版那套 camera-follow 阻尼调参；`Reset camera`
   恢复的是参考版的默认 yaw/pitch/distance。
8. **`/api/client-metrics` 只留最后一次报告**，不聚合历史。
9. **HUD 的 step rate 是最近 8 次规划的滑动中位数**，不是全程统计；全程统计看
   `/api/stats`。
10. **`--device cpu` 没在页面里跑过。** 协议和服务器都支持，但本轮所有页面实测
    都是 GPU。CPU 档的规划 p50 是 173.1 ms（[SUMMARY.md](SUMMARY.md)），照这个
    余量页面仍不会卡，但我没有实测。

---

## 10. 一键复现

```bash
cd ~/dev/motionbrick-mlx

# 起服务（受管后台任务 / 或前台跑）
./.venv/bin/python scripts/mlx_viewer.py --port 8848 --device gpu \
    > artifacts/viewer-run/server.log 2>&1 &

# 首页 + 静态资源
curl -s -o /dev/null -w "HTTP %{http_code}  %{size_download} B  %{time_total}s\n" http://127.0.0.1:8848/
for a in style.css app.js rig.js vendor/three.module.min.js vendor/three.core.min.js; do
  curl -s -o /dev/null -w "HTTP %{http_code}  %{size_download} B  /$a\n" "http://127.0.0.1:8848/$a"
done

# 一次真实交互：现场规划一段 walk
curl -s -o /tmp/plan.json -w "HTTP %{http_code}  %{size_download} B  %{time_total}s\n" \
  "http://127.0.0.1:8848/api/plan?style=walk&seed=1&move=0,1&facing=0,1&reset=1"

# 连续步频 / 首帧 / 加载可用时间
./.venv/bin/python artifacts/viewer-run/probe_viewer.py --port 8848 --clips 30
curl -s http://127.0.0.1:8848/api/stats            # client.first_pose_ms / ready_ms

# 接缝连贯性
#   （见 artifacts/viewer-run/continuity.txt 的生成方式，脚本内联在报告里）

# 停
kill "$(cat artifacts/viewer-run/server.pid)"
```

> 页面静态资源是每次请求现读磁盘的，改 `viewer/*.js` 不用重启服务器；
> 改 `scripts/mlx_viewer.py` 要重启。

---

# 追加（第 9 轮）：用户自己摆目标关键帧

**一句话**：模型本来就是**关键帧条件生成器**；本轮把它的**目标窗口**从"由命令推出来"
改成"由你摆出来"，页面上能点、能填、能立刻重规划，并且把**命中度实测**印在 HUD 与侧栏里。

起法不变（本轮实际跑在 `8859`，`8848` 上仍是第 8 轮那个进程）：

```bash
./.venv/bin/python scripts/mlx_viewer.py --port 8859 --device gpu
```

## 1. 目标窗口到底是什么（发现，带行号）

参考实现的 transition boundary 是**固定 8 帧窗口**（`src/root.cpp:366` "4 start + 4 end"，
`BOUNDARY_FRAMES = 8`，移植见 `models/schema.py:24-26`）：

| 片段 | 帧 | 含义 | 来源 |
|---|---|---|---|
| start | 0..3 | 当前状态 | `encode_context`（`src/motion_rep.cpp:180-250`） |
| **target** | **4..7** | **目标窗口** | `build_constraints` 用 style 片段 + spring 造出来（`src/agent.cpp:289-330`） |

root planner 把**恰好这 4 帧**（按归一化后的 `global_root`）拷进 `dense_target` 张量，作为
root 卷积堆的条件（`src/root.cpp:463-479`）；pose planner 同样把 `poses` 的 4..7 帧作为
pose 条件（`src/pose.cpp`，移植在 `models/pose.py`）。**所以"目标"不是第 9 个输入，
它就是 frames 4..7 本身。**

每帧带的东西（可用的自由度）：

| 块 | 宽度 | 字段 | 本轮是否暴露 |
|---|---|---|---|
| `global_root` | 5 | `x, y(pelvis 高度), z, cos(heading), sin(heading)` | **x/z/heading 暴露**；y 由 style（或 `pose_mode=hold` 时取当前骨盆高）给 |
| `local_root` | 4 | `[heading_rate, dx·30, dz·30, y]` | 不单独暴露：它是 frames 4..7 的有限差分（`src/agent.cpp:322-334`），跟窗口一致 |
| `poses`（external pose） | 303 | 33 个关节偏移 (99) + 34 个 6D 旋转 (204) | **`pose_mode` 两档**（style / hold）+ 高阶 API 可直接给整块 |
| `target_root_translations` / `target_local_rotations_xyzw` | 4×3 / 4×34×4 | 导出元数据（页面画的橙色幽灵） | 由摆放结果重算 |

**代价**：本轮只在 `build_constraints` **之后**替换 frames 4..7（
`src/motionbricks_mlx/data/targets.py`），`build_constraints` 本身、`mlx_plan.py`、
`mlx_align.py` 一行未动。不摆目标时走的还是原来的 `Agent.plan`，两个 sha256 逐字节不变
（见第 4 节）。

**对齐规则（实测，不是读出来的）**：这 4 帧和生成动作的**最后 4 帧**对齐。在**未改动的
默认路径**上，对每个偏移算"生成 root − 窗口 root"，argmin 在 16/16 个 walk 系场景里
（`walk` / `walk_zombie` / `slow_walk` / `walk_left` × seed 1/2/7/77）都是 `frames-4`；
静态的 `idle` 无唯一 argmin（整条 root 轨迹位移 < 8 mm）。`tests/test_mlx_targets.py::
test_alignment_rule_is_the_argmin_on_the_accepted_default_path` 会重新推导这条规则。

**参考仓库自己那条路**（`has_world_target`，`src/agent.cpp:236-241`）移植里也在
（`data/agent.py:349-351`），但它造的窗口是 spring 在 t=1.1 s 的值，**只走到请求距离的
53.7%**（实测：请求 3 m → 窗口 1.611 m）；所以本轮的"摆帧"是**直接设窗口**，不是走那条路。

## 2. 怎么用

### HTTP（`curl`，验收就走这条）

```bash
# 目标 = 从当前朝向正前方 3 m（offset 空间），plan 后带 placed_targets / target_hit
curl -s --get "http://127.0.0.1:8859/api/plan" \
  --data-urlencode "style=walk" --data-urlencode "seed=1" \
  --data-urlencode "move=0,1" --data-urlencode "facing=0,1" \
  --data-urlencode "reset=1" \
  --data-urlencode "targets=0,3,0" --data-urlencode "target_space=offset" \
| ./.venv/bin/python -c 'import json,sys;d=json.load(sys.stdin);print(d["target_hit"]["errors"],d["target_hit"]["within_tolerance"])'
```

`POST /api/plan` 也收 JSON：`{"style":"walk","targets":[[0,3,0],[1,4,90]],"target_space":"world","pose_mode":"style"}`。
新增参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `targets` | 空 | 摆放的目标帧，最多 4 个；查询串 `"x,z,deg;x,z,deg"`，JSON `[[x,z,deg],...]`（`deg` 可省） |
| `target_space` | `world` | `world`＝绝对世界坐标（点击地面得到的就是这个）；`offset`＝相对 clip 起点、沿角色自身朝向前/右的米数 |
| `pose_mode` | `style` | 目标帧的关节构型：`style`＝沿用该 style 的目标窗口构型；`hold`＝保持本 clip 起步时的构型 |

**不摆目标时**：回包 key 与第 8 轮**完全相同**（`placed_targets` / `target_hit` 不出现），
发给 planner 的关键字也一个不多——`tests/test_viewer_targets.py` 把这条当契约测。

摆 1 个目标时，窗口 4 个采样点都取该目标（"到位并保持"）；摆 n<4 个时，第 n 个之后的采样点
**保持最后一个**；>4 个直接 400（不静默截断）。理由见第 3 节：保持型窗口是命中最好的形状。

### 页面

侧栏底部新增 **YOUR TARGET KEYFRAMES** 面板：

* **Place target** 按一下进入摆放态，再**点地面**落一个点（1–4 个）；点下去时朝向取
  "从角色指向该点"，地面上出现橙色圆环 + 朝向箭头；
* 面板里逐行可直接改 `x` / `z` / `facing`（不用点界面也能用），`×` 删除；
* **Replan to targets** 从**角色现在站的位置**（`advance = 当前播放帧`）plan 一段去命中这些点；
* **Clear targets** 清空；
* 橙色**幽灵骨架**就是返回的 `targets.world_positions_xyz`——也就是被写进 frames 4..7 的窗口，
  和地面上的圆环**必然重合**（`tests/test_mlx_targets.py` 断言本模块算出的
  `window.world_roots` 与 `restore_world` 之后的 `motion.target_roots` 逐位相同）；
* HUD 多一格 **TARGET**：本次目标规划的 root 命中误差（米）；侧栏状态行给全套数字与容差。

**URL 预置**（不用点界面，和 `?style=` 同一套机制）：
`?style=walk&seed=1&move=0,1&live=0&targets=0,3,0&to_targets=1`。
`live=0` 只做"取消勾选 Replan while walking"，让页面停在**那一次**目标规划上（HUD 的数字才不会
被后续自动重规划冲掉）。顺手修了一个既有小 bug：`?move=` 以前会被 `updateControl()` 按
"W 键按住"的相机相对方向覆盖掉，现在给 `?move=` 时不再持 W 键（`viewer/app.js`）。

### CLI（脚本化 / 负控）

```bash
# 一次摆放，写与 mlx_plan.py 同形状的 plan 文档（多两个 key）+ 命中报告
./.venv/bin/python scripts/mlx_plan_targets.py --style walk --seed 1 \
    --from-current --target 0,3,0 --out /tmp/plan-targets.json

# 负控：把目标一路放远，看位移是不是跟着单调走
./.venv/bin/python scripts/mlx_plan_targets.py --style walk --from-current \
    --sweep 0,0.5,1,2,3,5,10,50 --out /tmp/unused.json
```

`--out` 与 `mlx_plan.py` 一样是**必填**（第 2 轮的硬规则）。`--target` 默认是世界坐标，
`--from-current` 时按"相对 clip 起点、沿自身朝向"解释。

## 3. 命中度实测

容差表（`data/targets.py::TARGET_TOLERANCES`，口径与 `mlx_align` 同形：**最大绝对误差**，
位置用米、角度用度，取被比较块的最坏值）：

| 指标 | 容差 | 依据 |
|---|---|---|
| `root_planar_m` | **0.15 m** | 默认路径（模型自己的窗口）同一指标的实测地板是 1.4e-3 m（静止）– 1.97e-2 m（44 帧 walk）；摆帧是**新请求**不是等价性检查，所以取地板的一个数量级以上，且约等于一个步幅的 20% |
| `heading_deg` | **5.0°** | 同上；实测 0.13–3.0° 在容差内 |
| `joint_m` | **0.15 m** | 最坏世界关节距离，与 root 同尺度 |

**主实测（`walk`, seed 1, 正前方, `--sweep`，`artifacts/viewer-targets-run/cli-sweep.txt`）**：

| 目标距离 | 实际位移 | 峰值 root 速度 | root 命中 | 朝向命中 | 判定 |
|---|---|---|---|---|---|
| 0.00 m | 0.000 m | 0.49 m/s | 0.0014 m | 0.23° | ok |
| 0.50 m | 0.502 m | 0.84 m/s | 0.0113 m | 0.13° | ok |
| 1.00 m | 0.990 m | 2.08 m/s | 0.0340 m | 0.99° | ok |
| 2.00 m | 1.980 m | 2.86 m/s | 0.0543 m | 1.50° | ok |
| 3.00 m | 2.948 m | 3.55 m/s | 0.1113 m | 0.55° | ok |
| 5.00 m | 4.916 m | 5.24 m/s | 0.2519 m | 1.93° | **超容差** |
| 10.00 m | 9.529 m | 10.63 m/s | 0.8186 m | 5.89° | **超容差** |
| 50.00 m | 45.464 m | **49.10 m/s** | 5.8467 m | 14.18° | **超容差** |

* **负控 1（放远）**：位移随目标距离**严格单调**（`monotone: True`），50 m 那档也真的"走过去"
  了 45.5 m。这就是"模型确实在追目标"，不是"窗口被忽略"。
* **负控 2（不可能的位置）**：50 m 在 1.47 s 的 clip 里意味着 ~31 m/s，模型**既不拒绝也不夹取**：
  它把动作外推到峰值 **49.1 m/s**（约人类步行的 25 倍），如实给出 5.85 m 误差并判 `OUT OF TOLERANCE`。
  没有 NaN/inf（`finite=True`，`token` 仍在 style 的允许集里）。**这是本轮要如实报告的失败方式。**
* **可命中包络**（`artifacts/viewer-targets-run/envelope.txt`，3 个 style × 前后/侧向 × 5 个距离）：

| style | 正前方通过到 | 侧向通过到 |
|---|---|---|
| `walk` | 4 m（0.0645 m） | 2 m（0.0593 m） |
| `slow_walk` | 4 m（0.0828 m） | 3 m（0.0801 m） |
| `walk_zombie` | 2 m（0.0707 m） | 2 m（0.0993 m） |

  超出包络后误差**单调上升**（walk 侧向 6 m → 0.58 m；walk 正前 6 m → 0.18 m）。
* **多帧窗口（2–4 个点）是另一个形状**：把窗口设成一条*运动*轨迹时模型会**滞后**
  （4 帧 ramp 到 3 m → 0.745 m；2 点 1 m→3 m → 1.47 m），因为它更擅长收敛到一个**保持型**窗口。
  所以：**摆 1 个点是精确档，摆 2–4 个点是"带滞后的轨迹档"**，两者都用同一张容差表判定，
  超了就如实报 `OUT OF TOLERANCE`（侧栏那一行会变红）。
* `pose_mode` 是**真的自由度**：`hold` 与 `style` 的窗口 root 的 x/z **逐位相同**，
  但动作的旋转数组不同（`tests/test_mlx_targets.py` 断言）。

## 4. 一致性（不摆目标时）

```
./.venv/bin/python scripts/mlx_plan.py --device gpu --out /tmp/step-gpu.json
./.venv/bin/python scripts/mlx_plan.py --device cpu --out /tmp/step-cpu.json
shasum -a 256 /tmp/step-gpu.json /tmp/step-cpu.json out/plan-mlx-gpu.json out/plan-mlx.json
```

| 检查 | 实测 |
|---|---|
| GPU plan sha256 | `ccda1952c967c8d7e893a795908f7de278f2acda1ce5ba1e1889692a8904a7c5` = `out/plan-mlx-gpu.json` |
| CPU plan sha256 | `262e77d438daa8a790eccf23f8cfd696bf620a304bb4d42b54e38282e085b64a` = `out/plan-mlx.json` |
| 页面默认路径 vs 验收 CLI 路径 | `/api/plan?style=walk&seed=1&move=0,0&facing=0,1&reset=1` 的 `world_positions_xyz`（4488 个数）与 `mlx_plan.py` 的 `session-initial` 步**逐位相同**（`np.array_equal`），`root_travel_metres` 也相同（0.088814）；回包 15 个 key，无 `switch`/`placed_targets`/`target_hit`（`artifacts/switch-run/viewer-default-vs-cli.txt`） |
| 跨版 5 seed × 6 步（对 `c81d8ee`，`/tmp/r5/campaign.py` 原样复用） | `pose.tokens` **30/30**、`sampling_uniforms` **30/30**、时长 token **30/30**，**0/2416** 决策不同，动作偏差 **0.000e+00**（root 与 rotation 都是 0） |
| `pytest -q` | **278 passed, 1 skipped**（第 8 轮起点 242/1，只增） |

## 5. 画面

`artifacts/viewer-targets-shots/`（无头 Chrome，REA `capture_browser_scenario`，
**只有 `goto` + `wait_for_timeout`**，无合成输入、未点界面、未抢焦点、无音频；每张 PNG 用
base64 解码后重算 sha256 并与证据比对）：

| 文件 | sha256（前 16） | 说明 |
|---|---|---|
| `final-900-settle.png` | `b9eb44fd29842dfa` | **主图，且是在最终代码那一版服务器上拍的**：1440×900，橙色目标幽灵站在橙色圆环上（目标 `(0, 3)`），绿色角色走到半途，地面轨迹连起来；HUD `TARGET 0.111 m`、`CLIPS 1`、`Replan while walking` 未勾选。同一张的 DOM 属性：`data-mlx-status=ready`、`data-mlx-target-hit=0.111320`、`data-mlx-target-hit-ok=true`、`data-mlx-target-window=40-43`（落在 `artifacts/viewer-targets-run/final-page-attributes.txt`） |
| `tall-1180-settle.png` | `c41027072633ec35` | 1440×1180：侧栏 **YOUR TARGET KEYFRAMES** 面板整块可见（Place target / Replan to targets / Clear targets / +1 m ahead / `T1 0.00 3.00 0 ×`），状态行写着 `1 placed · hit frames 40–43 · root 0.1113 m (tol 0.15) · heading 0.55° · joints 0.1259 m · within tolerance` |
| `live-off-settle.png` | `fcce818b80394b74` | 1440×900 的上一版（服务器 python 还差最后一处**只影响"不摆目标"分支**的追加式改动）；命中数字与主图一致 |
| `targets-settle.png` | `9bc54e11cf46a065` | 第一次会话（页面当时回 400 `unsupported target [0, 3, 0]`，见第 6 节坑 1） |

四次会话的 `page-errors` 都是 **0**（最后两次的 `console` 也是 0 条）。

## 6. 本轮踩到的坑（都修了）

1. **JSON 数组没解析**：页面 POST `[[0,3,0]]`（点击产生的就是位置三元组），而
   `parse_targets` 只认字符串和 dict → 服务端 400 `unsupported target [0, 3, 0]`，
   截图里 `data-mlx-status="error"`、HUD 全是 `—`。修法是加 `(list, tuple, ndarray)` 分支，
   并把"服务端默认 `target_space=world` / `pose_mode=style`"也测进契约。
2. **`?move=` 名不副实**：文档说它预置一次会话，但初始状态持着 W 键，
   `updateControl()` 会把地址栏给的方向按相机相对覆盖掉。修法：给 `?move=` 时清掉 `padKey`。
   这直接决定了无头截图能不能"一次 `goto` 就落在预期朝向"。

## 7. 诚实边界（本轮新增）

1. **包络是有边界的**：正前方约 ≤4 m、侧向约 ≤2–3 m（按 style），超出就超容差；
   但这只说明"超出容差"，**不是说模型拒绝**——它会一路外推到物理上荒谬的速度（49 m/s@50 m）。
2. **多帧窗口滞后**：2–4 个点的*运动*窗口按实测有 ~0.7–1.5 m 的滞后；1 个点（保持型）才准。
3. **关节姿态只给了两档 + 一个高阶入口**：页面不提供 34 个关节的编辑器；`pose_mode`
   是 `style`/`hold`，API 层可以直接给整块 303 宽 external pose（`PlacedTarget.pose`），
   但那条路**没有页面交互**，也只在纯数学层被测过。
4. **y（骨盆高度）不是用户自由度**：它跟 style 走（`hold` 时跟当前姿态），摆帧只摆 x/z/朝向。
5. **`bend` 之类上层交互、碰撞、脚部 IK 都没有**——摆到墙里/悬空，模型只会照做。
6. **一次目标规划是一次性的**：`Replan to targets` 之后若开着 `Replan while walking`，
   后续自动重规划回到**命令驱动**（不是继续追那批点）。再按一次按钮才会从**新位置**重新去追。
7. **容差是自己定的**（`TARGET_TOLERANCES`），不是参考实现给的——参考实现里没有"命中度"
   这个概念，所以我用 `mlx_align` 的**口径**（最大绝对误差 / 米 / 度 / 取最坏）加自己的阈值，
   依据写在表里（地板的一个数量级以上 + 约 20% 步幅）。第 3 节那张包络表就是它的**可证伪面**。

---

# 追加（第 10 轮）：把"切换"做成一等公民 —— 条件化当前状态，并给出可证伪的对照

**一句话**：A→B 不是"生成 A、再生成 B、硬拼"，而是**用 A 此刻的那 4 帧（连它差分出来的速度）
去条件化 B 的生成**；接缝处的位置差从对照臂的 **40.8 mm / 317.3 mm** 降到 **1.4 mm 以内**，
根速度跳变从 **3.1–6.5× 段内 p95** 降到 **0.47× 以内**；**但加速度那一行两种方法都做不到
"≤ 段内 p95"**，如实报（第 3 节末与第 10 节）。

起法不变，仍然**只有一个端口**（第 7 节）：

```bash
cd ~/dev/motionbrick-mlx
./.venv/bin/python scripts/mlx_viewer.py --port 8848 --device gpu
```

## 1. 切换在机制上到底是什么（不是新模块，是"不 reset"）

第 9 轮已经把边界查清楚了：模型的 transition boundary 是**固定 8 帧**（`models/schema.py:24-26`，
参考 `src/root.cpp:366` "4 start + 4 end"）——

| 段 | 帧 | 谁造 | 切换时变不变 |
|---|---|---|---|
| start | 0..3 | `encode_context`（`src/motion_rep.cpp:180-250`） | **不变**：就是 A 现在的那 4 帧 |
| target | 4..7 | `build_constraints`（`src/agent.cpp:289-330`） | **变**：改由 B 的 style 片段造 |

`local_root`（4 宽）是这 8 帧窗口的有限差分（`src/agent.cpp:322-334`；`encode_context` 里是
`src/motion_rep.cpp:290-301`），所以**窗口里能表达的是位置和一阶差分（速度）**。
这句话决定了本轮能做到什么、做不到什么：**窗口里没有加速度通道**。

于是"切换"的实现只有一句话：

> **`Agent.advance(n)` 把上下文挪到 A 的第 n 帧，然后 `Agent.plan(command, B_style)` —— 不 reset。**

`build_constraints` 自己会读到上下文的速度（`velocity_x/z`、`heading_velocity`，
`src/agent.cpp:229-241`）并把它带进 frames 4..7 的 spring；`apply_context_blend`
（`src/agent.cpp:354-379`，权重 0.3/0.433/0.567/0.7）再把生成片段的前 4 帧往 A 的位姿上拉。
这两件事都不是本轮新写的——本轮做的是**让它们有机会发生**。

**朴素对照（证伪臂）**：A、B **各自独立生成**（各自从自己 style 的种子重新播种——这正是
第 8/9 轮换 style 时的行为），再拼起来。为免被说成"稻草人"，对照臂里最慈善的版本还会把 B 的
片段**平移旋转到 A 的接缝位姿**（`restore_world`），所以它在 root 位置上是"零误差"的起手；
它的关节构型和关节角速度仍然是 B 自己的 —— 那正是"独立生成"不可消除的痕迹。

两条臂**只差这一个决定**，所以这是对照，不是演示。

**新增文件**（都在允许范围内）：

| 文件 | 作用 |
|---|---|
| `src/motionbricks_mlx/data/switch.py`（新） | 腿/接缝的度量与拼装：`seam_metrics`、`Leg`、`SwitchPlan`、`join_legs`、`Switcher`（两条臂都在这里） |
| `scripts/mlx_switch.py`（新） | CLI：写 plan 文档、跑对照矩阵、跑 C 线质量门。`--out` 与 `mlx_plan.py` 一样**必填** |
| `scripts/mlx_viewer.py`（追加分支） | `/api/plan` 新增 `switch` / `sequence` / `seams`，回包多一个 `switch` 块 |
| `viewer/**`（追加） | 侧栏 SWITCH 面板、HUD 两格、画布上的青色接缝柱、`Hard cut (control)` 勾选框 |
| `tests/test_mlx_switch.py`、`tests/test_viewer_switch.py`（新） | 17 + 13 个测试（第 8 节） |

**判据是从任务书抄下来的，不是我造的**（`switch.py:SEAM_CRITERION` 原文存在代码里）：

```
seam |dq| <= max(in-segment p95 |dq|)   for q in {position, velocity, acceleration}
```

接缝处的 `|dq|` 和段内的 `|dq|` **是同一个有限差分**——都是"相邻两帧之间 q 的变化量"——
所以两边同量纲、同含义，"≤ p95" 是字面意义上的比较。位置另给一条同形状的判据
`seam |dp| <= p95 per-frame |dp|`（接缝的位置差不超过该片段自己一帧的正常位移）。
容差口径沿用 `scripts/mlx_align.py`：**最大绝对误差**、位置用米、角度用度、取被比较块的最坏值
（关节那两行是"每个关节各自算比值，取最坏的那个关节"）。

## 2. 怎么用

### HTTP（`curl`，验收走这条）

```bash
# 一次现场切换：从当前片段第 20 帧切到 walk_zombie
curl -s --get "http://127.0.0.1:8848/api/plan" \
  --data-urlencode "style=walk" --data-urlencode "seed=1" \
  --data-urlencode "move=0,1" --data-urlencode "facing=0,1" --data-urlencode "reset=1" >/dev/null
curl -s --get "http://127.0.0.1:8848/api/plan" \
  --data-urlencode "style=walk_zombie" --data-urlencode "advance=20" \
  --data-urlencode "switch=1" \
| ./.venv/bin/python -c 'import json,sys;d=json.load(sys.stdin);s=d["switch"];print(s["chain"],s["seams"][0]["metrics"]["root_position_m"])'

# A->B->A 拼成一段，一次回包，缝全测
curl -s -X POST -H 'content-type: application/json' \
  -d '{"style":"walk","move":[0,1],"facing":[0,1],"advance":20,
       "sequence":["walk_zombie","walk"],"seams":[20,20]}' \
  http://127.0.0.1:8848/api/plan
```

新增参数（都是追加的；不带它们时请求形状与回包 key 与第 9 轮**逐字相同**）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `switch` | `0` | 1 = **切换进** `style`：保留上下文，只换 target 窗口的 style。不带它时换 style 仍然是"reseed"（也就是对照臂） |
| `sequence` | 空 | `"B,A"`：一次回包返回整段 A→B→A（`style` 是 A） |
| `seams` | 空 | 每个 `sequence` 项的接缝帧（省则用 `advance`） |

回包多一个 `switch` 块（`mode`/`conditioned`/`from_style`/`chain`/`seam_at`/`legs`/`seams`/
`criterion`/`worst_seam_ratio`/`all_seams_within_p95`）。`legs` 给每段在拼接片段里的帧区间，
`seams[i].joined_frame` 给接缝在拼接片段里的帧号 —— 页面就是用这两个画接缝柱的。

### 页面

侧栏新增 **SWITCH STYLE MID-MOTION** 面板：

* **下拉 + `Switch now`**：**走着的时候点它即可**。它把两个下拉都切到目标 style，
  下一次 `/api/plan` 带 `switch=1` + `advance=当前播放帧`；
* **`Play A → B → A`**：从当前位置起，一次请求拿回整段 A→B→A（含两处缝），当普通片段播；
* **`Back to command`**：退回命令驱动（清掉切换状态）；
* **`Hard cut (control)`**：把切换退回第 8/9 轮的行为（reseed）——**证伪臂就在页面上**，
  勾上之后同一时刻的角色位置差 0.57 m（第 6 节）；
* **HUD 两格**：`SEAM GAP`（接缝位置差，mm）与 `SEAM JUMP`（六行比值里最坏的一行）。
  侧栏 `#switch-status` 给全部数字与判据原文。**每一条都是服务器自己量出来的**，
  所以截图里就带着连续性的证据。

**URL 预置**（与 `?style=`/`?targets=` 同一套机制，无合成输入）：
`?style=walk&move=0,1&ab=walk_zombie&seam=20` 开机就规划整段 A→B→A；
`?switch=walk_zombie&at=20` 开机走**现场切换**那条路（先规划 A，再切）；
`?t=<帧>&paused=1` 把播放头钉在某一帧上（第 6 节的画面就是这么拍的）；
`?naive=1` 勾上对照臂。

### CLI

```bash
# A->B->A：写 plan 文档（形状与 mlx_plan.py 同，多一个 switch 块）
./.venv/bin/python scripts/mlx_switch.py --style walk --to walk_zombie --back walk \
    --seed 1 --seam-at 20 --seam-at-back 20 --out /tmp/switch-aba.json

# 对照矩阵 + C 线质量门（本轮报告里两张表都是这条命令产的）
./.venv/bin/python scripts/mlx_switch.py --style walk --to walk_zombie --back walk \
    --control --quality --quality-sweep --seeds 1,7 --seam-frames 12,20,28 \
    --out artifacts/switch-run/switch-aba.json --report artifacts/switch-run/report.json
```

## 3. 朴素对照 vs 本方法：连续性对照表

**48 个格子**（8 对 style × seed 1/7 × 接缝帧 12/20/28），同一进程内**每个格子两条臂交错测量**。
比值 = 接缝处的 `|Δq|` ÷ 段内 `p95(|Δq|)`（同一差分的同一统计量）。

| 量 | 判据 | **本方法（条件化）** | 硬切（贴到接缝位姿） | 硬切（不贴） |
|---|---|---|---|---|
| root 位置差（48 格最大） | ≤ 一帧 p95 | **1.383 mm** | 116.654 mm | 622.346 mm |
| 最坏关节位置差（48 格最大） | 同上 | **15.352 mm** | 833.761 mm | 994.312 mm |
| root 速度比 中位 / 最大 / 超判据格数 | ≤ 1 | **0.302 / 0.472 / 0/48** | 3.084 / 6.455 / 46/48 | 3.032 / 6.454 / 46/48 |
| root 加速度比 中位 / 最大 / 超 | ≤ 1 | **0.256 / 0.640 / 0/48** | 0.508 / 1.240 / 4/48 | 0.505 / 1.242 / 4/48 |
| 关节角速度比 中位 / 最大 / 超 | ≤ 1 | 1.158 / 2.315 / **32/48** | 3.788 / 4.982 / 48/48 | 3.788 / 4.982 / 48/48 |
| 关节角加速度比 中位 / 最大 / 超 | ≤ 1 | 1.398 / 3.301 / **42/48** | 1.969 / 3.777 / 48/48 | 1.969 / 3.777 / 48/48 |

**逐项读法（这才是诚实的那一半）**：

* **位置：三条都达标，而且是数量级。** 本方法的 root 接缝差 48 格最大 **1.383 mm**，
  最坏关节位置差 **15.4 mm**（比该片段一帧的正常位移还小，`position_within_one_frame` 48/48 为真）；
  对照臂即使被贴到接缝位姿上，root 差仍有 **5.8–116.7 mm**（y/骨盆高度那一项它修不了），
  关节差 **0.44–0.83 m**；不贴的那条差 **0.13–0.62 m**，是货真价实的瞬移。
* **root 速度：达标 48/48。** 本方法 0.302（中位）、最坏 0.472；对照臂 46/48 超判据，最坏 6.46×。
* **root 加速度：达标 48/48，但这不是区分度**——对照臂也只有 4/48 超判据
  （它的段内 p95 本身就被自己的起步抖动抬高了）。
* **关节角速度：改善了 3.3×（中位 3.788 → 1.158），但"≤ 段内 p95"没有全部达标（32/48 超）。**
* **关节角加速度：不区分**（中位 1.398 vs 1.969，最大 3.301 vs 3.777），两边都几乎全超。

**这个残差是机制的，不是调参没调好。** 用 **A→A**（同 style 的**普通边走边重规划**，
没有换 style）当**地板控制**：那 6 个格子里，本方法的关节角速度比中位 **1.131**、最大 **2.174**，
关节角加速度中位 **1.265**、最大 **1.735**。也就是说——**这个 4 帧窗口条件化机制本身**
（第 8 轮就在跑的那条路）在关节角速度/角加速度上就做不到"≤ 段内 p95"，地板是 ~1.1–1.7。
A→B（42 个异 style 格子）的 1.158 / 1.440 落在这个地板上，**多出来的部分不是"切换"造成的**。

根因可以从前言直接推出来：**窗口 4 帧只携带位置与一阶差分（速度）；加速度在条件通道里不存在**。
所以"位置 + 速度"能闭合，"加速度"不能。这既是本轮的结论，也是它的边界。

**A→B→A 那一段的实际数字**（`walk → walk_zombie → walk`，seed 1，缝 20/20，`move=0,1`，
`artifacts/switch-run/switch-aba.json`，84 帧、163.2 ms/次）：

| 缝 | root 位置差 | 最坏关节位置差 | Δv / p95 | Δa / p95 | 关节 Δω / p95 | 关节 Δα / p95 |
|---|---|---|---|---|---|---|
| @20 `walk→walk_zombie` | **0.1701 mm** | 13.312 mm | 0.076 / 0.229 m/s（**0.333×**） | 1.776 / 7.645 m/s²（**0.232×**） | 118.71°/s（0.826×，**0/34** 关节超） | 8307°/s²（1.663×，2/34 超） |
| @40 `walk_zombie→walk` | **0.2753 mm** | 9.385 mm | 0.160 / 0.259 m/s（**0.616×**） | 3.673 / 10.873 m/s²（**0.338×**） | 100.39°/s（1.025×，1/34 超） | 3405°/s²（1.650×，4/34 超） |

**结论**：位置与速度的"无缝"是被数字证明的（毫米级、比值 <1、对照臂 40–600 mm）；
关节角速度是**大幅改善但未达标**；关节角加速度**两种方法一样差**，是机制边界。
两种方法**量出来不一样**（位置差 230×、root 速度比 10×），所以这个"无缝"确实起了作用 ——
不是任务书里那句"如果两种方法量出来一样就停下"的情形。

## 4. 质量门（切换前后有没有变差）

只读 import `motionbricks_mlx.eval`（`metrics.py` 的 1/2/3 号指标与 `thresholds.py` 的阈值表），
一行没改、也没新造阈值。48 个格子，三种腿各测一遍**同一 style、同一 seed、同一 44 帧长度**的片段，
只有"从哪个上下文规划"不同：

| 腿 | 绝对 C 线门通过 | 中位脚滑 |
|---|---|---|
| `fresh`（从 style 重新播种） | **42/48** | 0.962 m |
| `replan`（同 style，从自己片段中段续规划 = 第 8 轮那条路） | **28/48** | 1.036 m |
| `switch`（从 A 的中段续规划 = 本轮） | **25/48** | 1.036 m |

失败项分布：`fresh` 6 个全是 `jerk_p95_deg`；`replan` 18×`foot_slide_max_m` + 8×`jerk_max_deg`
+ 4×`jerk_p95_deg`；`switch` 21×`foot_slide_max_m` + 6×`jerk_max_deg`。

**把成本拆开**（同格配对，所以是比值而不是绝对值）：

* **中段续规划本身**（`replan / fresh`）：脚滑中位 **1.208**（最大 1.905）、jerk p95 **1.013**、
  jerk max **1.102**。这是**第 8 轮"边走边重规划"就已经有的**代价，不是本轮引入的。
* **换 style 本身**（`switch / replan`）：脚滑中位 **1.040**、jerk p95 中位 **1.000**（最大 1.802）、
  jerk max 中位 **1.000**（最大 5.646）。**分布上等于没变**；变的是单帧极值。
* 48 个格子里有 **9 个**，切换会让某条指标失败而同 style 续规划不失败。

**接缝局部 vs 片段内部**（同样用 ±8 帧的**同宽窗口**：接缝窗口 vs 该片段所有内部窗口的 p95）：

| | jerk p95 比值 中位 / 最大 / 超 1 的格数 | 脚滑超内部 p95 | 穿地超内部 |
|---|---|---|---|
| **本方法** | **0.954 / 1.584 / 19/48** | **3/48** | 5/48 |
| 硬切对照 | **10.917 / 26.077 / 48/48** | 18/48 | 21/48 |

**读法**：接缝处的抖动**中位比片段内部还低（0.954×）**，19/48 略高于内部 p95（最多 1.58×）；
硬切对照在**全部 48 个格子里**都是 2.8–26× 的爆点。**所以"切换没有让质量变差"这句只在
"中段续规划本来就有的代价"这个意义上成立**：相对同 style 续规划，切换的脚滑与 jerk p95 中位数
都 ≈1.00；但绝对 C 线门会失败更多（25/48 vs 42/48，其中约 2/3 是续规划路径就有的）。

## 5. 持续可用性：切换不是一次性事件

第 9 轮的缺口是"一次目标规划是一次性的，后续自动重规划退回命令式"。本轮把它和切换一起修了，
证据在 `artifacts/switch-run/stickiness.txt`（同一条 8848 上跑的）：

```
1. reset=1 walk                    -> root_first_frame [0.0,    0.7802, 0.0   ]
2. switch=1 walk_zombie advance=20 -> root_first_frame [0.0445, 0.7298, 0.3115]  chain [walk, walk_zombie]
3. 普通重规划（不带任何 flag）      -> root_first_frame [0.0731, 0.7441, 0.8752]   <- 继续从 B 出发
4. 再 switch=1 walk advance=20     -> root_first_frame [0.0853, 0.732,  1.2757]  chain [walk_zombie, walk]
5. 对照：换 style 不带 switch       -> root_first_frame [0.0,    0.7708, 0.0   ]   <- reseed，回到 B 自己的原点
```

第 3 行是关键：**切换之后，一次完全普通的 `/api/plan`（既没有 `switch` 也没有 `sequence`）
仍然从 B 的当前位置继续**，而不是回到 B 自己的原点。服务器里这件事靠一个字段
（`_agent_style` = "当前动作所处的 style"）——上一版只在 reseed 分支里更新它，
所以切换之后第一次"不带 flag 的同 style 请求"会**重新播种**（那正是"退回命令式"）。
本轮在两条分支上都更新它，并且重规划路径按 `switch=1` 发送，两处都钉在测试里。

页面上同一件事可以从 DOM 读出来（`--dump-dom`，无合成输入）：
`data-mlx-plan-log="1:walk:reset/adv0:44f | 2:walk_zombie:switch/adv20:44f/live"` ——
第 2 条请求确实是 `switch/adv20`，不是 `reset/adv0`。

第 9 轮"目标只生效一次"那个缺口也一起修了：目标面板多了 **Keep chasing while walking**
（默认勾上），勾着时后续每次自动重规划都继续带 `targets`；取消勾选即停止追（点还留着）。

## 6. A→B→A 的实际画面

`artifacts/switch-shots/`（无头 Chrome，`--screenshot`，**只做导航与等待，不点界面、不合成输入**；
每张 PNG 都记了 sha256）：

| 文件 | 内容 |
|---|---|
| **`aba.gif`** | **主画面**：A→B→A 整段 84 帧的 GIF（每 2 帧取 1，共 42 帧，560×384，70 ms/帧 = 14.3 fps，总长 2.94 s，**1.63 MiB**；真值是 2 帧 @30fps = 66.7 ms，GIF 的时间粒度只有 10 ms）。底部字幕带写明当前在哪条腿，两处接缝有青色竖标与 `SEAM A -> B` / `SEAM B -> A`；42 帧两两都不重复，源 PNG 的 sha256 在 `artifacts/switch-run/gif-digests.txt` |
| `seam1-ab-1440x900.png` | 接缝 1 那一帧（`?t=20`）：HUD `STYLE walk → walk_zombie → walk`、`SEAM GAP 0.28 mm`、`SEAM JUMP 1.66×`，两条青色接缝柱与地面圆环都在画面里 |
| `seam2-ba-1440x900.png` | 接缝 2（`?t=40`） |
| `aba-walked-1440x900.png` | 整段末尾（`?t=83`）：`WALKED 2.09 m`、`ROOT Z 2.04 m` —— 地面轨迹整条穿过了两处接缝 |
| `tall-1440x1180.png` | 侧栏 SWITCH 面板整块可见 |
| **`live-switch-1440x900.png`** | **现场切换**（`?switch=walk_zombie&at=20&t=20`）：`SEAM GAP 0.17 mm`、`SEAM JUMP 1.66×`、`WALKED 0.58 m`、`ROOT Z 0.88 m`，一根接缝柱 |
| **`hardcut-control-1440x900.png`** | **证伪臂**（同一 URL + `?naive=1`）：`SEAM GAP —`、`SEAM JUMP —`、**没有接缝柱**、`WALKED 0.31 m`、`ROOT Z 0.30 m` |

后两张是**同一时刻、同一相机、只差一个 `naive` 参数**的一对：切换后 20 帧，
角色在 z=0.88 m 还是 z=0.30 m，差 **0.57 m**；页面的 `plan-log` 分别是
`2:walk_zombie:switch/adv20` 与 `2:walk_zombie:reset/adv0`。

**GIF 是逐帧钉住的**：每一帧都是 `?t=<帧>&paused=1` 的一次真实导航，页面在 seek 时把位姿与相机
迭代收敛（200 次 lerp）而**不是**等下一个 rAF —— 虚拟时间预算会把 rAF 饿死，那样拍到的会是
"最后一个回调恰好画的那一帧"。同一 URL 连拍 4 次，**去掉 HUD 的那块视口区域逐像素完全相同（0/222000 px）**；整图有 113–202 px
不同，全部落在 HUD/侧栏里**故意带上的本次实时耗时数字**上：**位姿可复现，PNG 不逐字节可复现**。

## 7. 一个服务、一个端口

第 8 轮的 8848 与第 9 轮的 8859 **不再并存**：8859 的进程在本轮开始时已经不在（只剩 8848），
本轮把 8848 那个也停掉，用**最终这一版代码**只起**一个**进程，第 8/9/10 轮的功能都在这一个端口上。

| 项 | 值 |
|---|---|
| URL | `http://127.0.0.1:8848/` |
| PID | **`65252`**（也写在 `artifacts/switch-run/server.pid`；日志首行自带 `(pid 65252)`） |
| 日志 | `artifacts/switch-run/server.log` |
| 绑定 | `127.0.0.1:8848`（`--host` 只接受 `127.0.0.1`） |
| 停止 | `kill "$(lsof -nP -iTCP:8848 -sTCP:LISTEN -t)"`，或受管任务 `job_kill <job_id>` |

启动日志（`artifacts/switch-run/server.log`）：

```
loading /Users/ghostcorn/dev/motionbrick-mlx/checkpoints/g1-f32-mlx on mlx-gpu ...
ready: cold start 293.9 ms, 15 styles, warm clip 44 frames
viewer: http://127.0.0.1:8848/  (pid 65252)
```

**为什么仍然是单线程**：MLX 的 GPU stream 是 thread-local 的，`ThreadingHTTPServer` 的 worker
里跑 `Agent.plan` 会直接 abort 掉进程（第 8 轮 §1 记过）。测试里的 HTTP 层因此绑 **CPU** 档
（和 `tests/test_viewer_targets.py` 一样），协议层与服务层才有覆盖。

## 8. 一致性、测试、跨版门

```
./.venv/bin/python scripts/mlx_plan.py --device gpu --out /tmp/r10-step-gpu.json
./.venv/bin/python scripts/mlx_plan.py --device cpu --out /tmp/r10-step-cpu.json
shasum -a 256 /tmp/r10-step-gpu.json /tmp/r10-step-cpu.json out/plan-mlx-gpu.json out/plan-mlx.json
./.venv/bin/python -m pytest -q
```

| 检查 | 实测 |
|---|---|
| GPU plan sha256 | `ccda1952c967c8d7e893a795908f7de278f2acda1ce5ba1e1889692a8904a7c5` = `out/plan-mlx-gpu.json` **逐字节相同** |
| CPU plan sha256 | `262e77d438daa8a790eccf23f8cfd696bf620a304bb4d42b54e38282e085b64a` = `out/plan-mlx.json` **逐字节相同** |
| 跨版 5 seed × 6 步（对 `c81d8ee`，`/tmp/r5/campaign.py` 原样复用） | `pose.tokens` **30/30**、`sampling_uniforms` **30/30**、时长 token **30/30**、**0/26576** 决策不同、动作偏差 **0.000e+00**（root 与 rotation 都是 0） |
| `pytest -q` | **308 passed, 1 skipped**（第 9 轮起点 278/1，只增） |

新增测试：`tests/test_mlx_switch.py`（17：比值策略的两个零情形、六行比值键、
连续接缝/速度跳变/位置差/加速度跳变各自被正确归行、短输入拒绝、JSON 可表示、
拼接片段的帧区间与缝帧号、以及真机上的"条件化 vs reseed 必须不同""位置差 ≤1.5 mm 而对照 >3×"
"地板控制"）；`tests/test_viewer_switch.py`（13：无 flag 请求的调用形状与回包 key 集合不变、
`switch`/`sequence`/`seams` 的转发、空值不转发、未知 style 是 404、页面确实带上了这些控件、
真机上切换的缝被量出来且下一拍继续从 B 出发、不带 flag 的换 style 仍然瞬移）。

## 9. load

机器不空闲（桌面进程 mihomo / WindowServer / Chrome / DSH renderer / Lark）。本轮全程 1 分钟
load average 在 **8.1–16.5** 之间；对照矩阵与质量门是**同进程、同格子交错配对**的比值，
所以绝对值被抬高这件事不进入结论；跨版门与 sha256 是逐字节判据，与负载无关。

| 时刻 | load 1min | 出处 |
|---|---|---|
| 起单端口服务（最终那次） | **12.68** | `artifacts/switch-run/server-load.txt` |
| 默认路径 sha256 + 跨版门 | **12.40**（1/5/15 min: 12.40/13.58/13.27） | `artifacts/switch-run/verify.log`、`cross-version.txt` 首行 |
| 对照矩阵 + 质量门（48 格，两趟，约 37 s） | **15.65**（15.65/14.51/14.75） | `artifacts/switch-run/run-full.log` 首行 `uptime` |
| `pytest -q`（308 passed） | **7.33**（7.33/13.80/13.33） | 本节正文的那次运行 |
| 画面拍摄（42 帧 + 6 张静帧，约 2 min） | **7.45–12.7** | `artifacts/switch-run/shoot.sh` 运行期间 |

## 10. 诚实边界（本轮新增）

1. **加速度那一行没做到，而且两种方法一样差。** 关节角加速度的接缝比中位 1.398 vs 对照 1.969，
   都远在 1 以上；`all_seams_within_p95` 在本轮所有主配置里都是 **false**。
   页面上 `SEAM JUMP` 显示的就是这六行里最坏的一行，所以它常常 >1 并标红——**这是如实显示**，
   不是没调好。机制上的原因见第 3 节末：4 帧窗口里没有加速度通道。
2. **关节角速度"改善但未达标"**：32/48 个格子仍高于段内 p95。地板控制（A→A，同 style 续规划）
   自己就是 1.13–2.17，所以这是机制地板，不是切换引入的。
3. **绝对质量门上切换比"重新播种"差**：25/48 vs 42/48。约 2/3 的差距来自**第 8 轮就存在的**
   中段续规划路径（28/48）；换 style 本身在 9/48 个格子上多失败一条指标。
   中位脚滑在 `replan` 与 `switch` 之间**完全相同**（1.036 vs 1.036 m）。
4. **对照臂的"慈善版"是我选的**：把 B 贴到 A 的接缝位姿（`restore_world`）比真实硬切更宽容，
   因为真实硬切还会瞬移 0.13–0.62 m。两条都报了（`naive_aligned` / `naive_raw`），
   表里主列用的是更宽容的那条。
5. **接缝的"位置差"是 root + 最坏关节，不是逐关节。** 最坏关节 15.4 mm 已经小于该片段一帧的
   正常位移，但单个关节仍可能比它自己的 p95 大——关节那两行是逐关节取最坏比值的。
6. **PNG 不逐字节可复现**（HUD 带实时耗时）；GIF 的**帧**是钉住的（`?t=&paused=1`），
   pose 可复现（同 URL 4 次拍摄视口差 ≤11 px）。
7. **`?t=` / `?paused=1` / `?ab=` / `?switch=` / `?naive=` 是地址栏参数**，不是合成输入；
   它们只预置一次会话，和 `?style=` 同一套机制。页面的 `Hard cut (control)` 是**勾选框**，
   勾上之后行为退回第 8/9 轮（reseed），本轮所有"对照"数字都是代码里那条臂算的，不是勾出来的。
8. **没有做**：切换时的碰撞/脚步 IK、把 4 帧窗口换成更长的窗口（那要改模型条件通道，
   属于训练侧，本轮范围内不动）、多人会话（仍然一个进程一个 agent）。
