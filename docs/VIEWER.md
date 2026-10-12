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
