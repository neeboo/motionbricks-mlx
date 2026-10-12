# 第 7 轮 · 实测：烘一批真动作、导出可看的动画、12 项 manifest、两重校验

**这一份是"能不能搞"的实测回答**，不是跑分。前六轮量的是烘焙**有多快、有多少裕度、怎么分解**；
本轮第一次把它做成**成品**：每场景一份 plan 文档（reference driver 的文档形状）、一份库级
manifest（`0007` 第 4 条要求的 12 项一个不少）、一份独立进程的 B=1 复跑做等价性、一份 C 线质量门
原始输出、以及**能看的 GIF**。

工具与产物：

* `scripts/mlx_bake_library.py`（新，本轮）：`batch` / `single` / `compare` / `check` / `metrics` / `pairdoc`
* `tools/side_by_side_gif.py`（新，本轮）：只做**拼接**——两条臂的骨架像素全部来自参考侧渲染器
* `src/motionbricks_mlx/models/bake.py`：`BatchResult.motions` **追加**一个默认空字段（当前向派生的
  编排器把每条臂完整的 `Motion` 交出来，导出文档需要的 4 个 target 关键帧只在它里面）
* `tests/test_mlx_bake_export.py`（新，5 个测试）
* 产物：`artifacts/bake-2026-10-12/`（见该目录的 `README.md`，说明哪些进 git、哪些是可再生的）
* 全程 `--bake-stages all+decode`（已批准的 release 烘焙配置，`0009` 第 2 条）

---

## 0. 一句话结论

**"实测"能搞：release 配置下烘出真动作、导出成 GIF、12 项 manifest 齐全、批量与单场景的离散决策
逐位一致（0/1408 不同）、C 线质量门 PASS。**

但**档位（tier）必须如实报告**，而本轮第一次出现的实测结果是：

| 库（campaign） | B | 实测 tier | 扰动/最紧余量 |
|---|---|---|---|
| `single-style`（16 seed · 单风格 `walk`） | 8 / 16 | **reproducible** | **0.5290 / 0.4089**（与第 5、6 轮**逐位复现**） |
| `idle-style`（16 seed · 单风格 `idle`） | 16 | **reproducible** | **0.2137** |
| `diverse`（8 种真实风格 · 14 个 seed · 时长同构） | 8 / 16 | **draft** | **1.4402 / 1.2716** |

也就是说：**同一个脚本、同一个 release 配置、同样时长同构，只有"库里有没有多种风格"变了，
档位就从 release 掉到 draft。** 判据是 `0008` 冻结的机械规则（全局 max 扰动 / 全局 min 余量 ≤ 0.70），
本轮**不改判据**，如实记 draft。第 1.4 节给出机制归因：这不是"某条决策要翻了"，
而是**全局规则的分子与分母来自不同风格**——分子由 `idle` 供（它的 logits 绝对扰动大 3–6 倍），
分母由 `walk_zombie` 供（它的决策余量紧 12 倍）。**每场景自己的比值最大只有 0.142。**

---

## 1. 场景 / seed / style 清单与选档依据

### 1.1 时长同构是**测出来**的，不是声明的

`bake.manifest_fields` 里 `bake_grouping.token_groups` 是实测的。本轮的选场景方法：
先用管线把 `(style, movement, seed)` 的 **duration 选择**读出来（`selected_tokens` 6..11 →
`frames = 4 × token`，即 24/28/32/36/40/44 帧），再据此**只挑落在同一个 duration 上的行**。
三个 campaign 的实测 `token_groups` 都是**单组**：

* `diverse` → `{"6": 8}` / `{"6": 16}`（24 帧）
* `single-style` → `{"11": 8}` / `{"11": 16}`（44 帧，`walk` 对任何 seed 都选 11）
* `idle-style` → `{"6": 16}`（24 帧）

`--strict` 会在"实测分组 ≠ 声明分组"时直接退出；四次 bake 全部通过 `--strict`。

**一个结构性事实（值得记住）**：任务点名的四种风格 `idle` / `walk` / `walk_zombie` / `stealth_walk`
**不可能同时**落在一个时长同构批次里——`idle` 永远选 24 帧，`walk` 永远选 44 帧。
所以 `diverse` 用 8 种真实风格里能同时落在 24 帧的那些：`idle`、`walk_zombie`、`stealth_walk`、
`walk_stealth`、`walk_happy_dance`、`injured_walk`、`walk_gun`、`walk_boxing`（**任务点名的
`idle` / `walk_zombie` / `stealth_walk` 三种都在里面**）。

### 1.2 `diverse`（主 campaign，16 行 = B=8 是它的前 8 行）

| label | style | movement | seed | 实测 frames |
|---|---|---|---|---|
| s00 | idle | (0.2,0,0) | 5 | 24 |
| s01 | idle | (0.2,0,0) | 9 | 24 |
| s02 | walk_zombie | (0.2,0,0) | 6 | 24 |
| s03 | walk_zombie | (0.2,0,0) | 8 | 24 |
| s04 | stealth_walk | (0,0,0) | 1 | 24 |
| s05 | stealth_walk | (0,0,0) | 3 | 24 |
| s06 | walk_stealth | (0.2,0,0) | 7 | 24 |
| s07 | walk_stealth | (0.2,0,0) | 13 | 24 |
| s08 | walk_happy_dance | (0.2,0,0) | 10 | 24 |
| s09 | walk_happy_dance | (0.2,0,0) | 14 | 24 |
| s10 | injured_walk | (0.2,0,0) | 2 | 24 |
| s11 | injured_walk | (0.2,0,0) | 6 | 24 |
| s12 | walk_gun | (0.2,0,0) | 4 | 24 |
| s13 | walk_gun | (0.2,0,0) | 12 | 24 |
| s14 | idle | (0.2,0,0) | 17 | 24 |
| s15 | walk_boxing | (0.2,0,0) | 2 | 24 |

* **≥4 种真实 style**：B=8 切片就有 4 种（idle / walk_zombie / stealth_walk / walk_stealth），
  B=16 有 **8 种**；**≥3 组 seed**：B=8 有 8 个不同 seed，B=16 有 **14 个**。
* `stealth_walk` 的两行用 `movement=(0,0,0)`：探针实测该风格只有 movement 为 0 时才选 24 帧
  （0.2 时选 28/32）。逐场景命令写在 manifest 的 `campaign` 块里，不是隐藏状态。
* `(0.2,0,0)` 与 `mlx_bench.py` 的烘焙命令同形；`(0,0,0)` 是参考驱动 `session-initial` 用的命令。

### 1.3 两个对照臂（用来把"风格多样性"这一个变量单独拿出来）

| campaign | 行数 | style | seed | 用途 |
|---|---|---|---|---|
| `single-style` | 16 | 全是 `walk` | 1..16 | 第 4–6 轮量的就是这种形状 → 复现它们的裕度 |
| `idle-style` | 16 | 全是 `idle` | 1..16 | `diverse` 里扰动最大的那条风格，单独烘一次 |

### 1.4 选档依据（`0007` + `0008`，未新造判据）

判据原样：**扰动 / 最紧决策余量 ≤ 0.70（`RELEASE_RATIO`）**；`B > 8` 必须时长同构
（`GROUPED_ABOVE_B = 8`）；`B = 1` 永不用于烘焙；战役（逐场景 token/uniform/duration）
必须通过。四条腿全在 `bake.bake_tier` 里，本轮**没有改一个常数**。

实测（`load` 见 manifest 的 `load_before`/`load_after`，另见第 8 节）：

| campaign | B | 最紧余量 | 最大 pose 扰动 | 比值 | tier |
|---|---|---|---|---|---|
| diverse | 8 | 2.9879e-03 | 4.3030e-03 | **1.4402** | draft |
| diverse | 16 | 2.9879e-03 | 3.7994e-03 | **1.2716** | draft |
| single-style | 8 | 2.9855e-03 | 1.5793e-03 | **0.5290** | reproducible |
| single-style | 16 | 2.9855e-03 | 1.2207e-03 | **0.4089** | reproducible |
| idle-style | 16 | 3.6058e-02 | 7.7057e-03 | **0.2137** | reproducible |

**两处独立复现**（这是本轮最硬的自校验）：`single-style` 的 0.5290 / 0.4089 与参考文档
`SUMMARY.md` 第 2 节里第 5/6 轮的同构 0.5290 / 0.4089 一致；它的最大扰动 **1.2207e-03**
就是 `0007` 表里的"全阶段，同 duration **1.221e-03**"，`bake_worst_rotations` **2.753e-05**
也就是 `SUMMARY` 记的 **2.753e-05**。一个**新写的导出脚本**得到与旧战役相同的数，说明两把尺子是同一把。

**机制归因（只作诊断，不改档位）**——`manifest` 里另有 `safety_detail` 块，逐场景列
"自己的扰动 / 自己的余量 / 自己的比值"：

| campaign B=16 | 最大扰动来自 | 最紧余量来自 | 全局比值 | 逐场景自己比值的最大者 |
|---|---|---|---|---|
| diverse | `s01` **idle** 3.799e-03（它自己余量 3.606e-02） | `s02` **walk_zombie** 2.988e-03（它自己扰动 4.234e-04） | 1.2716 | **0.142** |
| single-style | `w13` walk 1.221e-03 | `w00` walk 2.985e-03 | 0.4089 | 0.408 |
| idle-style | `i06` idle 7.706e-03 | `i08` idle 3.606e-02 | 0.2137 | 0.105 |

读法：`idle` 的**绝对**扰动本来就是别的风格的 3–6 倍（它单独烘时最大 7.7e-03，比 `walk` 的
1.2e-03 大 6 倍），但它的**决策余量也大 12 倍**（3.6e-02 vs 3.0e-03）。冻结的全局规则把
"最敏感风格的扰动"除以"最紧风格 (walk_zombie) 的余量"，于是 cross-scenario 得到 1.27；
**没有任何一条决策自己的比值超过 0.142**。`idle` 单独烘反而是今天最安全的一档（0.2137）。
这不是判据的错，是判据保守性的来源——**本轮不动它，只把它记下来**。

---

## 2. manifest 全文（12 项逐一核对）

`bake.BAKE_MANIFEST_FIELDS` 十二项，`scripts/mlx_bake_library.py check` 对**写盘的 manifest**
逐字段核对（不是对内存里的 dict）：三份 manifest、五个 group 全部 **12/12，`manifest check: PASS`**。

### 2.1 `diverse`（主 campaign）

```
group B=B8: 12/12 required fields present
    batch_size                       8
    bake_tier                        "draft"
    bake_stages                      ["decode", "decoder", "pose", "root"]
    bake_stage_switches              ["bake_root_batch", "bake_decoder_batch", "bake_decode_batch"]
    bake_grouping                    {"mode": "duration-homogeneous", "token_groups": {"6": 8}}
    bake_campaign_scenarios          8
    bake_token_decisions_differing   0
    bake_worst_roots_m               5.960464477539062e-07
    bake_worst_rotations             6.541609764099121e-06
    bake_safety_ratio                1.4401532077880626
    bake_min_decision_margin         0.0029878616333007812
    bake_action_bit_identical        false
group B=B16: 12/12 required fields present
    batch_size                       16
    bake_tier                        "draft"
    bake_stages                      ["decode", "decoder", "pose", "root"]
    bake_stage_switches              ["bake_root_batch", "bake_decoder_batch", "bake_decode_batch"]
    bake_grouping                    {"mode": "duration-homogeneous", "token_groups": {"6": 16}}
    bake_campaign_scenarios          16
    bake_token_decisions_differing   0
    bake_worst_roots_m               1.0728836059570312e-06
    bake_worst_rotations             1.3649463653564453e-05
    bake_safety_ratio                1.2716246409192467
    bake_min_decision_margin         0.0029878616333007812
    bake_action_bit_identical        false
manifest check: PASS
```

### 2.2 `single-style`（对照臂，复现第 5/6 轮）

```
group B=B8: 12/12 required fields present
    batch_size                       8
    bake_tier                        "reproducible"
    bake_stages                      ["decode", "decoder", "pose", "root"]
    bake_stage_switches              ["bake_root_batch", "bake_decoder_batch", "bake_decode_batch"]
    bake_grouping                    {"mode": "duration-homogeneous", "token_groups": {"11": 8}}
    bake_campaign_scenarios          8
    bake_token_decisions_differing   0
    bake_worst_roots_m               6.258487701416016e-07
    bake_worst_rotations             8.434057235717773e-06
    bake_safety_ratio                0.5289889793962625
    bake_min_decision_margin         0.0029854774475097656
    bake_action_bit_identical        false
group B=B16: 12/12 required fields present
    batch_size                       16
    bake_tier                        "reproducible"
    bake_stages                      ["decode", "decoder", "pose", "root"]
    bake_stage_switches              ["bake_root_batch", "bake_decoder_batch", "bake_decode_batch"]
    bake_grouping                    {"mode": "duration-homogeneous", "token_groups": {"11": 16}}
    bake_campaign_scenarios          16
    bake_token_decisions_differing   0
    bake_worst_roots_m               1.1399388313293457e-06
    bake_worst_rotations             2.7533620595932007e-05
    bake_safety_ratio                0.4088803705478358
    bake_min_decision_margin         0.0029854774475097656
    bake_action_bit_identical        false
manifest check: PASS
```

### 2.3 `idle-style`（归因臂）

```
group B=B16: 12/12 required fields present
    batch_size                       16
    bake_tier                        "reproducible"
    bake_stages                      ["decode", "decoder", "pose", "root"]
    bake_stage_switches              ["bake_root_batch", "bake_decoder_batch", "bake_decode_batch"]
    bake_grouping                    {"mode": "duration-homogeneous", "token_groups": {"6": 16}}
    bake_campaign_scenarios          16
    bake_token_decisions_differing   0
    bake_worst_roots_m               2.662126541137695e-07
    bake_worst_rotations             3.874301910400391e-06
    bake_safety_ratio                0.21370007934408886
    bake_min_decision_margin         0.03605818748474121
    bake_action_bit_identical        false
manifest check: PASS
```

**逐项核对**：`batch_size`（8/16）✅、`bake_tier`（由 `bake_tier()` 现算，不是手写）✅、
`bake_stages`（`decode`+`decoder`+`pose`+`root` = `all+decode`）✅、`bake_stage_switches`
（三个开关名，由 stage 集合反推）✅、`bake_grouping`（**实测** mode + token 分组）✅、
`bake_campaign_scenarios`（8/16）✅、`bake_token_decisions_differing`（全 0）✅、
`bake_worst_roots_m` / `bake_worst_rotations`（本轮战役现算）✅、
`bake_safety_ratio` / `bake_min_decision_margin`（现算）✅、
**`bake_action_bit_identical` 如实为 `false`** ✅（全阶段批量后最终动作不再逐位，见第 3 节）。

---

## 3. 等价性：批量有没有说谎

**做法**：`compare` 子命令把"同一批场景、逐场景单独跑"的结果（**另一个进程**、`MOTIONBRICKS_MLX_OPT`
里**一个 bake 开关都没有**，脚本启动时自检 `resolved bake switches: {False, False, False}`）
与批量产物逐场景配对，比对 tokens / uniforms / duration / 最终动作 / `world_positions_xyz`。

| campaign | B | tokens 一致 | uniforms 一致 | duration 一致 | 决策不同 | 动作逐位一致 (roots / rotations) | 最差 roots | 最差 rotations | 最差角 | 最差 world |
|---|---|---|---|---|---|---|---|---|---|---|
| diverse | 8 | **8/8** | **8/8** | **8/8** | **0/384** | 0/8 · 0/8 | 5.960e-07 m | 6.542e-06 | 0.0527° | 2.469e-06 m |
| diverse | 16 | **16/16** | **16/16** | **16/16** | **0/768** | 0/16 · 0/16 | 1.073e-06 m | 1.365e-05 | 0.0539° | 7.475e-06 m |
| single-style | 8 | **8/8** | **8/8** | **8/8** | **0/704** | 0/8 · 0/8 | 6.258e-07 m | 8.434e-06 | 0.0582° | 3.374e-06 m |
| single-style | 16 | **16/16** | **16/16** | **16/16** | **0/1408** | 0/16 · 0/16 | 1.140e-06 m | 2.753e-05 | 0.0541° | 1.798e-05 m |
| idle-style | 16 | **16/16** | **16/16** | **16/16** | **0/768** | 0/16 · 0/16 | 2.662e-07 m | 3.874e-06 | 0.0526° | 1.521e-06 m |

* **离散决策（tokens / uniforms / duration）逐场景、逐位一致：64/64 个场景**（5 个 group 的全部行），
  合计 **0 个 token 决策不同**（4032 个 token 决策里 0 个不同）。
* **最终动作不是逐位的**（0/64 场景 roots 逐位相同）——这正是 manifest 里
  `bake_action_bit_identical=false` 的如实记录。偏差是 `mlx_align` 容差的
  1.0e-03 m / 2.0e-03 的 **1.1e-03× / 1.4e-02×**（最差 1.140e-06 m / 2.753e-05）。
* 最差角 0.0539° 落在 `mlx_align` 旋转角列的**地板**上（`SUMMARY` 第 5 节边界 2：那列读到
  4.3e-02–5.1e-02 度是浮点角度的量化地板，不是回归）。
* 逐场景表（含每个场景自己的 roots/rotations 的 sha256 对照）在
  `artifacts/bake-2026-10-12/*/compare-B*.json` 与 `logs/compare-*.log`。
* `targets` 两条臂**逐位相同**（targets 来自 command/style 的约束，不经过批量）——渲染文档里可以直接对照。

**同一批场景的增量证据**：`single-style` B=16 的 `bake_worst_rotations = 2.7533620595932007e-05`
与 `SUMMARY.md` 记的 `2.753e-05` 完全一致，而这份数字是**在另一个进程里、由另一条代码路径**产出的。

---

## 4. 质量门（C 线产物，只读调用）

### 4.1 `scripts/eval_gate.py --no-collect`（原样调用，未改一行）

```
./.venv/bin/python scripts/eval_gate.py --no-collect        # exit 0
```

原始输出 115 行全部在 `artifacts/bake-2026-10-12/logs/eval-gate.log`；结论：

* **10 个质量组（6 个 seed 组 + 2 个 MLX 参考对 + 2 个参考 JSON 自身步骤）全部 `[PASS]`**，
  每个组 9 条指标全绿；
* **cross-impl 2 对 `[PASS]`**：`token_agreement_frac 1 >= 1`，
  `deviation_roots_max_m 5.235e-07 / 3.994e-06 <= 1e-03`，
  `deviation_rotation_max_deg 0.0424 / 0.0466 <= 0.2`，
  `deviation_world_max_m 1.870e-06 / 3.270e-05 <= 2e-03`；
* **5 个负控全部按设计触发**（`shuffle_frames` / `freeze_frames` / `add_jitter_noise` /
  `raise_feet` / `translate_roots`）；
* **`GATE PASS`**，报告写在 `out/eval/gate-report.json`。

**边界（必须说清）**：这个门跑的是它自己的固定场景集（`out/eval/plan-seed-*.json` 等），
**不认识本轮的产物目录**——`--no-collect` 不会去读 `artifacts/bake-.../`。所以它证明的是
"这套管线当时的产物过门"，不是"我这批 GIF 过门"。下一小节补上后者。

### 4.2 把本轮的产物喂给同一套 C 线指标（只读 import `motionbricks_mlx.eval`）

`scripts/mlx_bake_library.py metrics` 用 `eval.document.load_step` + `eval.gate.evaluate_step`
+ `eval.thresholds.THRESHOLDS`（**同一套阈值、同一套代码，未改**）评本轮 4 个子目录 × 16 份 = **64 份**文档：

| campaign / 臂 | 通过 | 不通过 | 红了什么 |
|---|---|---|---|
| diverse / batch-B16 | 14/16 | 2 | `s08` `walk_happy_dance`：`boundary_rotation_max_deg 28.8 > 26`；`s09`：`jerk_p95_deg 4.774 > 4`、`boundary_rotation_max_deg 50.1 > 26` |
| diverse / single-B16 | 14/16 | 2 | **完全相同的两份文档、完全相同的红项** |
| single-style / batch-B16 | **16/16** | 0 | — |
| single-style / single-B16 | **16/16** | 0 | — |

而且**批量臂与单场景臂的判定逐场景相同**：diverse 的 336 个数值指标最大差
**5.579e-03（`jerk_p50_deg` @ s01）**，single-style 的 336 个最大差 **7.757e-03（`jerk_max_deg` @ w08）**
——也就是说：**红的两份不是被批量批红的**，
它们在单场景臂里一样红（`walk_happy_dance` 是舞蹈动作，拿行走标定的阈值去卡它）。
这是"批量没有降低质量"的独立证据。

---

## 5. 产物清单、渲染命令、体积、肉眼一致

### 5.1 渲染路径（参考侧脚本，**未改一行**）

`~/dev/motion-bricks-cpp/tools/render_motion.py` 的输入形状与 `export/plan_json.py` 的文档形状
**一致**，直接可用。它的 `main()` 按 `document["steps"]` 逐 step 输出 `<label>.gif`，
所以本轮的 `pairdoc` 子命令把**同一场景的两条臂**拼成一份两步文档
（`s00-batch` / `s00-single`），一次调用就画出并排对照的两幅。

```
# ① 组两份文档（batch step + single step，同一场景）
./.venv/bin/python scripts/mlx_bake_library.py --campaign diverse pairdoc \
    --batch-dir  artifacts/bake-2026-10-12/diverse/plans/batch-B16 \
    --single-dir artifacts/bake-2026-10-12/diverse/plans/single-B16 \
    --pair s00=idle --pair s02=walk_zombie --pair s04=stealth_walk \
    --out artifacts/bake-2026-10-12/render \
    --manifest artifacts/bake-2026-10-12/render/pairs.json
./.venv/bin/python scripts/mlx_bake_library.py --campaign single-style pairdoc \
    --batch-dir  artifacts/bake-2026-10-12/single-style/plans/batch-B16 \
    --single-dir artifacts/bake-2026-10-12/single-style/plans/single-B16 \
    --pair w05=walk --out artifacts/bake-2026-10-12/render \
    --manifest artifacts/bake-2026-10-12/render/pairs-single-style.json

# ② 参考渲染器（原文件，未改）。它要 Pillow；本仓库 .venv 里没有 Pillow，
#    所以用系统 Homebrew python（3.14 + Pillow 12.2.0）跑；没有往 .venv 装东西，
#    也没有动 pyproject.toml。
/opt/homebrew/bin/python3 ~/dev/motion-bricks-cpp/tools/render_motion.py \
    artifacts/bake-2026-10-12/render/render-s00.json artifacts/bake-2026-10-12/gifs
/opt/homebrew/bin/python3 ~/dev/motion-bricks-cpp/tools/render_motion.py \
    artifacts/bake-2026-10-12/render/render-s02.json artifacts/bake-2026-10-12/gifs
/opt/homebrew/bin/python3 ~/dev/motion-bricks-cpp/tools/render_motion.py \
    artifacts/bake-2026-10-12/render/render-s04.json artifacts/bake-2026-10-12/gifs
/opt/homebrew/bin/python3 ~/dev/motion-bricks-cpp/tools/render_motion.py \
    artifacts/bake-2026-10-12/render/render-w05.json artifacts/bake-2026-10-12/gifs

# ③ 并排拼接（本轮新工具；只贴像素，骨架像素全部来自 ②）
/opt/homebrew/bin/python3 tools/side_by_side_gif.py \
    --left  artifacts/bake-2026-10-12/gifs/s00-batch.gif \
    --right artifacts/bake-2026-10-12/gifs/s00-single.gif \
    --left-label "BATCH B=16 (release cfg)" --right-label "SINGLE B=1" \
    --out artifacts/bake-2026-10-12/gifs/pairs/s00-batch-vs-single.gif
# s02 / s04 / w05 同形
```

### 5.2 产物与体积（`ls -la`）

```
$ ls -la artifacts/bake-2026-10-12/gifs
  101745  s00-batch.gif            101745  s00-single.gif     <- idle, 24 帧
  147887  s02-batch.gif            147887  s02-single.gif     <- walk_zombie, 24 帧
  128980  s04-batch.gif            128980  s04-single.gif     <- stealth_walk, 24 帧
  275348  w05-batch.gif            275348  w05-single.gif     <- walk, 44 帧
$ ls -la artifacts/bake-2026-10-12/gifs/pairs
  173677  s00-batch-vs-single.gif   (1290x594, 24 帧)
  292528  s02-batch-vs-single.gif   (1290x594, 24 帧)
  279045  s04-batch-vs-single.gif   (1290x594, 24 帧)
  513339  w05-batch-vs-single.gif   (1290x594, 44 帧)
$ du -sh artifacts/bake-2026-10-12
 35M   (其中 27.7M 是 128 份 plan 文档，见该目录 README.md 的取舍说明)
```

**每份文档 169,901–297,862 B（≈170–298 kB）**（`diverse/plans/batch-B16/batch-B16-s00-idle.json`
= 172,280 B；44 帧的 `walk` 文档 ≈ 298 kB），因为文档形状是参考驱动的扁平 float 列表。
128 份合计 27.7 MiB。
每个文档的 sha256 记在 `manifest.json` 的 `groups[B].products` 里。

### 5.3 肉眼一致——而且比"看起来像"更强

`tools/side_by_side_gif.py` 在拼接时顺手把两幅图的像素**数**出来：

```
s00 (idle)        : pixels compared 8,601,600   differing 0  max channel delta 0
s02 (walk_zombie) : pixels compared 8,601,600   differing 0  max channel delta 0
s04 (stealth_walk): pixels compared 8,601,600   differing 0  max channel delta 0
w05 (walk)        : pixels compared 15,769,600  differing 0  max channel delta 0
```

* 四条臂的 GIF **逐字节相同**（sha256 相同，见 `logs/pair-*.log`）：
  `s00 3d5383b5…`、`s02 c2e9a720…`、`s04 4f63b776…`、`w05 524f67f9…`。
* 这不是"渲染了两次同一份数据"：`pairdoc` 的两步来自**不同进程**、不同文件，
  roots 数组确实不同（`s00 max|Δroot| = 1.582e-07 m`、`s02 = 4.023e-07`、
  `s04 = 3.213e-07`、`w05 = 2.123e-07`；sha256 不同）。
  差异**全部落在亚像素**：渲染器 640×560，本轮四个场景的实测投影是 **428–482 像素/米**
  （1 像素 ≈ 2.1–2.3 mm），所以本场最差的 1.073e-06 m roots 偏差 ≈ **4.6e-04 像素**，
  单帧 1e-7 m 量级的偏差 ≈ 4.3e-05 像素。
* 所以"批量与单场景看起来一致"在本轮是**量化成立**的：0/38,574,400 个像素不同。

（动图：`artifacts/bake-2026-10-12/gifs/pairs/*.gif`，左=B=16 批量、右=B=1 单场景；
即 [s00 idle](../artifacts/bake-2026-10-12/gifs/pairs/s00-batch-vs-single.gif)、
[s02 walk_zombie](../artifacts/bake-2026-10-12/gifs/pairs/s02-batch-vs-single.gif)、
[s04 stealth_walk](../artifacts/bake-2026-10-12/gifs/pairs/s04-batch-vs-single.gif)、
[w05 walk](../artifacts/bake-2026-10-12/gifs/pairs/w05-batch-vs-single.gif)。）
`walk_zombie` 那条还标出 4 个橙色 target 关键帧——两条臂的 targets 逐位相同。

---

## 6. 默认路径两个 sha256 复核

```
./.venv/bin/python scripts/mlx_plan.py --device gpu --out /tmp/r7/final-gpu.json
./.venv/bin/python scripts/mlx_plan.py --device cpu --out /tmp/r7/final-cpu.json
diff -q /tmp/r7/final-gpu.json out/plan-mlx-gpu.json
diff -q /tmp/r7/final-cpu.json out/plan-mlx.json
```

| 检查 | 实测 |
|---|---|
| GPU 文档 sha256 | `ccda1952c967c8d7e893a795908f7de278f2acda1ce5ba1e1889692a8904a7c5`（与 `out/plan-mlx-gpu.json` 逐字节相同）|
| CPU 文档 sha256 | `262e77d438daa8a790eccf23f8cfd696bf620a304bb4d42b54e38282e085b64a`（与 `out/plan-mlx.json` 逐字节相同）|

两个 sha256 在**改动前后各验一次**（23:17 与收尾），四次都一致。所有 `mlx_plan.py` 调用都显式带 `--out`。

---

## 7. `pytest`

```
# 本轮起点（HEAD 0f020f6 的本树）是 209 passed / 1 skipped
./.venv/bin/python -m pytest -q $(git ls-tree -r HEAD --name-only tests/ | grep '\.py$') \
    tests/test_mlx_bake_export.py
214 passed, 1 skipped in 32.06s          <- 209 + 本轮新增的 5 个，只增不减
```

`tests/test_mlx_bake_export.py` 的 5 个：

1. `BatchResult.motions` 默认空 / 追加式（位置构造 `BatchResult(labels, runs, drivers, suspensions, stage_sizes)` 仍可用）；
2. campaign 定义就是 docstring 说的那样（label 唯一；B=8 切片 ≥4 风格 ≥3 seed；B=16 ≥4 风格；
   `single-style` 单风格 16 seed；切片越界报错）；
3. `array_digest` **逐位**：差一个 float32 ULP 就不同，float64 输入按 float32 比；
4. `manifest check` **fail closed**：12 项齐 → PASS，少一项 → 退出码 1 且打印 `MISSING [...]`；
5. `drive_batch_single_pass` 每个场景交出**同一份** `Motion`（roots/rotations 与该场景的 `Run`
   逐位相同、targets 形状 `(4,3)`/`(4,34,4)`），而 round-4 的 `drive_batch` **`motions == []`**
   —— 钉住"round-4 路径未动"。

**全量 `pytest -q` 的如实报告**：`240 passed, 1 skipped, 1 failed`，失败的是
`tests/test_viewer_server.py::test_stats_endpoint_counts_plans`
（`scripts/mlx_viewer.py:368`，`'StubPlanner' object has no attribute 'client'`）。
那是**同工作区的另一个 agent 正在进行中的 viewer 改动**（`scripts/mlx_viewer.py`、`viewer/`、
`tests/test_viewer_fk.py`、`tests/test_viewer_server.py` 都是本轮期间新出现、HEAD 未跟踪的文件），
与本轮改动无关；把 HEAD 已跟踪的测试文件加上本轮新文件重跑就是上面的 **214 / 1**。
**我没有碰那些文件。**

---

## 8. 每次运行的 load

`load average` 由脚本自己读 `os.getloadavg()` 并在前后各记一次，写进对应 manifest 的
`load_before` / `load_after`；每次运行的完整 stdout 在 `artifacts/bake-2026-10-12/logs/`。

| 运行 | 进程 | load (before → after) |
|---|---|---|
| diverse batch B=8 | 独立进程 | 6.88 → 6.92 |
| diverse batch B=16 | 独立进程 | 6.92 → 6.93 |
| single-style batch B=8 | 独立进程 | 6.93 → 6.93 |
| single-style batch B=16 | 独立进程 | 6.93 → 8.07 |
| diverse single（B=1 语义） | 独立进程 | 8.33 → 8.33 |
| single-style single（B=1 语义） | 独立进程 | 8.33 → 8.33 |
| single-style batch B=16 `--repeats 9`（计时对照） | 独立进程 | 9.51 → 9.39 |
| idle-style batch B=16 | 独立进程 | 19.59 → 18.98 |
| idle-style single（B=1 语义） | 独立进程 | 18.98 → 18.98 |
| `mlx_plan.py --device gpu`（基线/收尾） | 独立进程 | 9.65 / 16.99 |
| `mlx_plan.py --device cpu`（基线/收尾） | 独立进程 | 9.65 / 16.99 |
| `eval_gate.py --no-collect` | 独立进程 | 11.21 → 11.21 |
| `pytest -q`（全量） | 独立进程 | 15.85 → 12.85 |

**跨 B 的配对每个 B 单独进程**（`0009` 第 4 条）：上表四个 batch 各是一个进程；
每个进程内的 serial/single-pass 配对是同进程顺序轮换。**机器不空闲**：整轮 load 6.9–19.6
（桌面进程），所以绝对值只作佐证，**同进程比值才是本轮引用的话**。

**一条计时方法学观察（新）**：`single-style` B=16 在 `--repeats 5` 下同进程比值是 **3.909×**
（serial 521.67 / single-pass 133.47 ms），改成 `--repeats 9` 是 **4.500×**
（505.47 / 112.32 ms），而参考文档第 5/6 轮的 9 次重复三臂轮换是 **4.443×**。
同一个 B、同一条臂、同样的场景，**只有重复次数/轮换表变了，单臂 p50 差了 16 %**——
`--repeats 5` 时单趟臂的中位数落在前几次较冷的调用上。这是 `0009` 第 4 条
"长命进程跨 B 积累状态"的镜像：**重复太少、轮换太短同样会给单臂带上状态。**
本轮 manifest 里记的是 5 次重复的值（那才是这次运行实际跑的），9 次重复的对照在
`artifacts/bake-2026-10-12/single-style-r9/`。

---

## 9. 诚实边界

1. **档位是数据相关的，而且本轮第一次出现"多样库 = draft"**。第 5/6 轮的 release 结论
   （同构 B ≤ 16）是在**单风格批次**上量的；本轮把 `idle`/`walk_zombie`/`stealth_walk`/
   `walk_stealth`/`walk_happy_dance`/`injured_walk`/`walk_gun`/`walk_boxing` 放进同一个同构批次，
   冻结的全局规则给出 **1.2716 / 1.4402 → draft**。我没有改判据、也没有挑场景去把它抬回 release。
2. **draft 的机制是 cross-scenario，不是"决策要翻"**。见 1.4 节：分子来自 `idle`（绝对扰动大），
   分母来自 `walk_zombie`（余量紧），**逐场景自己比值最大 0.142**。`safety_detail` 只是归因，
   **不参与 `bake_tier`**；我没有据此宣称 release。
3. **质量门的覆盖是部分的。** `eval_gate.py --no-collect` 不认识本轮的产物目录，它评的是它自己的
   固定场景集（10 组，全 PASS）。本轮产物另外用同一套阈值评了一遍（4.2 节），
   但那是**附加**证据、不是 C 线的正式门，而且 `walk_happy_dance` 两份在**两条臂里一样红**
   （舞蹈动作 vs 行走阈值）——阈值不是为本轮这批风格标定的。
4. **渲染是近似的、亚像素的。** 参考渲染器把 640×560 的骨架投影画出来，
   本轮实测投影 428–482 像素/米，最差的 1.140e-06 m 偏差约为 **4.9e-04 像素**，
   所以两条臂的 GIF **逐字节相同**。
   这证明"看不见差别"，**不证明**"动作完全相同"（动作确实不同，最差 1.140e-06 m）。
5. **`.venv` 里没有 Pillow。** 参考渲染器用系统 Homebrew python（3.14 + Pillow 12.2.0）跑；
   没有往 `.venv` 装包，也没有动 `pyproject.toml`。`tools/side_by_side_gif.py` 只用 Pillow + numpy。
6. **`--repeats 5` 的绝对值偏冷**（第 8 节）：本轮 manifest 的 timing 是按实际跑的 5 次重复记的，
   和参考文档 9 次三臂轮换的 4.443× 不可直接比。
7. **`bake_worst_rotations` 比 roots 高 1–2 个量级**（1.4e-05 vs 1.1e-06），最差角 0.0539° 落在
   `mlx_align` 旋转角那个 **4.3e-02–5.1e-02 度的地板**上（`SUMMARY` 第 5 节边界 2）。
8. **`BatchResult.motions` 是本轮唯一的 src 改动**，追加、默认空、只被单趟编排器填充；
   round-4 的 `drive_batch` 保持 `motions == []`（测试 5 钉住），两个默认 sha256 因此不变。
9. **128 份 plan 文档（27.7 MiB）、4 份 `render-*.json`（1.5 MiB）与 9 份 `runs-*.json`（2.1 MiB）没有进 git**
   （它们都在磁盘上；`compare-*.json` 是 `runs-*` 的逐场景蒸馏形式），原因与再生命令见
   `artifacts/bake-2026-10-12/README.md`；它们**在磁盘上**，`git status --short` 会显示为未跟踪。
10. **同工作区有另一个 agent 在改 viewer**（`scripts/mlx_viewer.py`、`viewer/`、
    `tests/test_viewer_*.py`）。全量 pytest 的那 1 个失败来自它，我没有碰这些文件。

---

## 10. `git log --oneline -3` / `git status --short`

本轮主体提交是 `b527271`；补上这份文档里那段 `git status` 正文的是它的后续提交。

```
$ git log --oneline -3
b527271 round 7: bake a real library, export the animations, and ship the twelve-field manifest
0f020f6 decision 0009 approved: batching the motion decode joins the release bake config
a49a7d1 round 6: attribute the single pass with a direct instrument, then batch the motion decode

$ git status --short
?? artifacts/bake-2026-10-12/diverse/plans/
?? artifacts/bake-2026-10-12/diverse/runs-batch-B16.json
?? artifacts/bake-2026-10-12/diverse/runs-batch-B8.json
?? artifacts/bake-2026-10-12/diverse/runs-single-B16.json
?? artifacts/bake-2026-10-12/idle-style/plans/
?? artifacts/bake-2026-10-12/idle-style/runs-batch-B16.json
?? artifacts/bake-2026-10-12/idle-style/runs-single-B16.json
?? artifacts/bake-2026-10-12/render/render-s00.json
?? artifacts/bake-2026-10-12/render/render-s02.json
?? artifacts/bake-2026-10-12/render/render-s04.json
?? artifacts/bake-2026-10-12/render/render-w05.json
?? artifacts/bake-2026-10-12/single-style-r9/plans/
?? artifacts/bake-2026-10-12/single-style-r9/runs-batch-B16.json
?? artifacts/bake-2026-10-12/single-style/plans/
?? artifacts/bake-2026-10-12/single-style/runs-batch-B16.json
?? artifacts/bake-2026-10-12/single-style/runs-batch-B8.json
?? artifacts/bake-2026-10-12/single-style/runs-single-B16.json
?? artifacts/viewer-run/
?? artifacts/viewer-shots/
?? scripts/mlx_viewer.py
?? tests/test_viewer_fk.py
?? tests/test_viewer_server.py
?? viewer/
```

（上面是**提交后**的状态：本轮新增/修改的文件已经在 `b527271` 里，所以不再出现在 status 中；剩下的是刻意不入库的产物与另一个 agent 的在制品。）

上面 `git status --short` 里：`plans/`、`runs-*.json`、`render/render-*.json` 是**本轮的产物，故意不进 git**（可逐字节再生，理由见 `artifacts/bake-2026-10-12/README.md`）；`artifacts/viewer-*/`、`scripts/mlx_viewer.py`、`tests/test_viewer_*.py`、`viewer/` 是**同工作区另一个 agent 的 viewer 工作**，与本轮无关，我没有碰。

本轮新增/修改（**只有这些**）：

* 新增 `scripts/mlx_bake_library.py`、`tools/side_by_side_gif.py`、
  `tests/test_mlx_bake_export.py`、`docs/BATCH-BAKE-RUN.md`、`artifacts/bake-2026-10-12/**`
* 修改 `src/motionbricks_mlx/models/bake.py`（`BatchResult.motions`，追加）、
  `docs/status-deepseek.md`（追加本轮小结）
* **未碰**：`training/**`、`scripts/train_*.py`、`eval/**`、`scripts/eval_*.py`、
  `tests/test_eval_*.py`、`docs/motion-representation.md`、`README.md`、`NOTICE`、`LICENSE`、
  `pyproject.toml`；`~/dev/motion-bricks-cpp` 全程只读（`find -newermt` 证明本轮期间零修改）；
  `mlx_plan.py` / `mlx_align.py` / `mlx_bench.py` 语义未改；**无 push**。
