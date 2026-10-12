# 状态 · GLM（C：动作质量评估与回归门禁）

- 当前：工单 `docs/brief-glm-eval.md` 全部交付——六个指标族、带出处的阈值表、
  5 种破坏的负对照、可复跑的门禁 CLI、52 条测试；门禁在 f32 MLX 输出 +
  参考 JSON 上 **GATE PASS**，5 项负对照全部变红并附前后数值。
- 正在动的文件：`src/motionbricks_mlx/eval/**`（新目录）、`scripts/eval_collect.py`、
  `scripts/eval_gate.py`、`tests/test_eval_*.py`、本文件。
- 等决定：无阻塞。一条诚实结论见下"判别力"——三个指标在当前数据上**不判别**，
  门禁已绕开它们（细节在内）。

---

## 2026-10-12 · C 线评估与门禁（工单 docs/brief-glm-eval.md）

**越界说明**：只写了工单放行的文件——`src/motionbricks_mlx/eval/**`、
`scripts/eval_{collect,gate}.py`、`tests/test_eval_{metrics,document,corruptions,gate}.py`、
本文件。未碰 `models/**`、`data/**`、`export/**`、`motion/**`、`training/**`、
`scripts/mlx_*.py`、`scripts/train_*.py`、`docs/motion-representation.md`、README/NOTICE/LICENSE/pyproject。
`~/dev/motion-bricks-cpp` 只读未动。评估侧不 import MLX、不加载 bundle——门禁的纯逻辑部分
（`run_gate`/`run_negative_controls`）吃的是 plan 文档 JSON，测试无需任何模型产物。

### 做了什么（对照工单六项指标 + 门禁）

1. **文档读取与结构校验（指标 6）**：`eval/document.py` 把 Go driver 形状的 plan 文档
   （顶层 `fps/joints`，步级 `motion/targets/world_positions_xyz/skeleton`，全平铺浮点表）
   reshape 成 `PlanStep`。校验项：帧数>0、关节数=34（可参数化，测试用 6）、fps=30、
   targets 恰 4 帧、motion+targets 四元数归一误差 ≤1e-3、全部数组有限（NaN/Inf）、
   `world_positions` 骨盆列 == `roots`（≤1e-6，世界坐标可信的链路保证）。
   脚关节按**名字**解析（`left/right_ankle_roll_skel` + `left/right_toe_base`），不靠位置口传。
2. **六个指标族**（`eval/metrics.py`，算法与单位定死在这里，改一处须同步 `docs/status-glm.md`）：
   - **① 脚滑** `foot_sliding`：脚最低点（ankle_roll/toe_base 取低者）在接触帧对
     （两侧帧最低点高 < 5 cm）间的 xz 累计位移，米；左右脚各报，门禁判 max。
   - **② 抖动** `jitter`：逐关节逐帧旋转增量 `θ=angle(q_t,q_{t-1})`（geodesic，度）的二阶差分
     `|d2|` 分布 p50/p95/max；外加 `freeze_frac` = 增量 <0.01° 的比例（直接停顿探测器）。
   - **③ 接地** `grounding`：双足最低点高度分布（min/p5/p50/p95/max，米）、穿地帧比例、
     悬空帧比例（>12 cm）。门禁判**穿地深度**（≥ -5 cm）与悬空比例（见"判别力"1）。
   - **④ 首尾衔接** `boundary_error`：motion 尾 4 帧 vs `targets` 块——根位移欧氏误差（米）
     与逐关节 geodesic 角（度），各报 mean/max，门禁判 max。
   - **⑤ 跨实现一致性**：`token_agreement`（复用 `pose.tokens` 口径——MLX 侧
     `Agent.plan(capture_trace=True)` 的 trace，参考侧 `out/ref/trace/transition-*.json`，
     逐位相等，1.0 为满分）+ `motion_deviation`（同种子双实现：根位移 max、旋转角 mean/max、
     世界坐标 max）。限值复用 A 线已发布的对齐容差（1e-3 m / 0.2° / 2e-3 m / tokens 精确）。
   - **⑥ 结构校验**：见第 1 条，任一问题即红。
3. **阈值与出处**（`eval/thresholds.py`，每条限值 = 实测基线极值 + 余量，出处写进注释与本文件）：
   - 质量门（8 条好动作标定 = 6 个种子 + 参考场景 2 步）：脚滑 ≤1.21 m（基线 max 1.100 +10%）；
     jerk p50 ≥0.05°、p95 ≤4.0°、max ≤30°（基线 0.086-0.376 / 0.73-2.17 / 3.1-17.2）；
     freeze_frac ≤0.15（基线 0-0.061，≈7 帧以上的停顿必触发）；穿地深度 ≥-0.05 m
     （基线 min -0.0191，2.6 倍余量）；悬空比例 ≤0.10（基线 0）；边界 ≤0.15 m / ≤26°
     （基线 0.020-0.068 m / 4.96-21.46°，见"判别力"3）。
   - 一致性门（指标 5）：如上复用容差。参考实现自身 Debug↔Release 差异
     （roots 2.589e-07 m，旋转 max 0.0429°）小于同种子 MLX↔参考实测偏差
     （roots 5.2e-07..4.0e-06 m，旋转 max 0.047°），门在两者之上、距真实漂移尚有 3 个数量级。
4. **场景收集**：`scripts/eval_collect.py` 固定 6 个种子组（1/7/13/42/77/123，walk，
   move (1,0)，facing (0,1)）+ 参考场景对（session-initial / replan-turned），
   写 `out/eval/plan-seed-*.json`、`plan-refpair.json`、`tokens-refpair.json`（~2.1 s 全量）。
5. **门禁**：`scripts/eval_gate.py`——10 个质量组（6 种子 + 2 MLX 参考对 + 参考 JSON 自带 2 步）
   全部指标 PASS、指标 5 两组同种子对 PASS、5 项负对照全部触发，三者齐备才 exit 0；
   报告落 `out/eval/gate-report.json`。`--fresh` 强制重采集，`--no-collect` 拒绝隐式采集，
   `--skip-cross-impl` 供参考产物缺席时降级（门禁会如实标注）。
6. **负对照**（`eval/corruptions.py` + `run_negative_controls`）：五种破坏各在全部好动作上执行，
   `EFFECT` 表声明各自必须变红的指标，触发即附 (指标, 好值 → 坏值) 证据。实测（seed 1）：
   - ① 帧序打乱 → slide 1.073→19.69 m，jerk_p95 2.17→31.4°，边界根误差 0.033→0.738 m
   - ② 冻结 20 帧（整帧冻结，见"判别力"2）→ freeze_frac 0.0014→0.326，jerk_p50 0.339→0.107°
   - ③ 加抖动噪声（σ=0.04，重归一）→ jerk_max 14.1→35.0°，p95 2.17→18.1°
   - ④ 整体抬高 0.25 m → 悬空比例 0→1.0
   - ⑤ 根位移平移 (0.5, 0.5) → 边界根误差 0.033→0.675 m（targets 不动，差值=平移量）
7. **测试 52 条**：已知答案单元测试（滑步=拖行距离、接触门开合、手算二阶差分 [0,3,3,1]°→5°、
   冻结比例 1/3、手工高度表 → 穿地/悬空比例 0.25/0.25、边界平移/旋转精确值、token 一致性
   三态、同实现偏差为零）；文档往返与逐项结构破坏检测；破坏的不变性（原件不可变、世界坐标
   必为真 FK、签名成立）；e2e：合成 6 关节骨架（真脚名）走完 run_gate +
   run_negative_controls，好集合绿、任一破坏入组即红、**放宽全部阈值后负对照必须失灵**
   （门禁依赖阈值本身，而非无条件报警）。全套 `pytest -q`：**192 passed, 1 skipped**
   （skip 为 A 线 `test_representation_smoke`，与上线前一致，未恶化）。

### 校准基线（f32 MLX，`out/eval/gate-report.json`）

| 组 | 脚滑 m | jerk p50/p95/max ° | freeze | 穿地 min m | 悬空 | 边界 m/° |
|---|---|---|---|---|---|---|
| seed-1 | 1.073 | 0.339 / 2.17 / 14.1 | 0.0014 | -0.0081 | 0 | 0.033 / 7.7 |
| seed-7 | 1.097 | 0.333 / 2.07 / 12.1 | 0.0041 | -0.0085 | 0 | 0.027 / 6.4 |
| seed-13 | 1.098 | 0.351 / 1.94 / 12.2 | 0.0014 | -0.0079 | 0 | 0.026 / 7.7 |
| seed-42 | 1.100 | 0.376 / 2.02 / 9.6 | 0.0014 | -0.0083 | 0 | 0.024 / 8.6 |
| seed-77 | 1.089 | 0.352 / 1.99 / 14.0 | 0.0021 | -0.0081 | 0 | 0.025 / 7.2 |
| seed-123 | 1.088 | 0.332 / 2.14 / 12.5 | 0.0034 | -0.0084 | 0 | 0.027 / 7.6 |
| session-initial（MLX） | 0.081 | 0.086 / 0.73 / 3.1 | 0.0609 | -0.0077 | 0 | 0.020 / 5.0 |
| replan-turned（MLX） | 0.893 | 0.321 / 1.84 / 17.2 | 0.0 | -0.0191 | 0 | 0.068 / 21.5 |
| 参考 JSON 两步 | 与 MLX 逐指标一致（差异 <2%） | | | | | |

跨实现（同种子）：token 一致性 1.0/1.0；roots max 5.2e-07 / 4.0e-06 m；旋转 max 0.042°/0.047°；
世界坐标 max 1.9e-06 / 3.3e-05 m。

### 判别力：诚实结论

在当前数据（单一 walk 命令、44 帧、toe-drag 步态）上，六个指标族**不是同等判别的**：

1. **穿地帧比例不判别**：好动作上 0.41-0.50（模型脚关节中心在深蹲态本来就骑在地面下几毫米），
   与"坏"无法区分。门禁改判**穿地深度**（-5 cm 地板），比例照报不判。
2. **位姿冻结对世界系指标不可见**：此步态拖行脚趾（脚几乎永远贴地、跟着身体走），
   只冻关节旋转、根继续走 → 脚滑、jerk p50 几乎不动（实测 slide 1.073→1.090）。
   负对照 ② 因此改为**整帧冻结**，其签名（速度骤降 + 一次性追赶跳）由 `freeze_frac`
   直接捕捉（0.0014→0.326）。若未来出现"位姿冻结但根运动"的真实缺陷形态，需要补根速度停顿探测器。
3. **边界误差非零是设计使然**：transition 以 targets 为条件而非复制它们，好动作边界误差
   0.020-0.068 m / 4.96-21.46°（换风格重排步最大）。门禁放在该带上方（0.15 m / 26°），
   能抓住 0.5 m 级的上下文错位，抓不住条件性偏差内部的质量退化——后者没有免费午餐，
   需要真值轨迹才能判。
4. **脚滑绝对值由步态风格主导**：toe-drag 使其 ≈ 步行里程（~1.1 m），"脚滑小 = 好"在此模型上
   不成立。1.21 m 的门只在**同一命令口径**内有意义（冻结/打乱能推过线）；换步态、换命令
   （跑、转身、横移）必须重标定，阈值表已注明这一点。
5. **判别良好的一侧**：jerk 分布（噪声/打乱敏感，分离 8-15 倍）、freeze_frac（停顿直接命中）、
   悬空比例（抬高即饱和到 1.0）、指标 5 全套（bit 级一致性 + 亚微米偏差，任何实现漂移
   都逃不过）——这四个是门禁的主力。

### 复跑

```bash
./.venv/bin/python scripts/eval_gate.py --fresh   # 采集 + 门禁 + 负对照，~3 s
./.venv/bin/python -m pytest -q                   # 192 passed, 1 skipped
```
