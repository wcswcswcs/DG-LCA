# DG-KAN v10.1 结果解读与 v10.2 完整实验计划

> 本文基于 v10.1 `Hypothesis Adjustment / FuturePath Generated Optimizer` 的真实执行结果制定。本文不把 discovery label 写成 official controller，不使用 dataset-specific tuning，不把 C3 20000 未执行解释成 fake success，也不把 FPO12 / generated discovery 的失败包装成小阈值问题。
>
> 本文所有公式使用 `$...$` 或 `$$...$$`，Typora 友好。

---

# 0. 一句话判断

v10.1 的核心结果是：

```text
route = R4-FPOAndGeneratedBothFail_UpdateRuleTheoryReset
primary_blocker = natural_density_insufficient
secondary_blocker = FPO12_and_generated_discovery_failed
system_legal_controller_pass = 0
generated_route_status = stopped_future_path_generated_optimizer_reset
```

这轮有进展，但不是能力成功。它真正说明：

$$
\boxed{
\text{自然动作池不够；训练当下识别器失败；初版生成器也失败；下一步必须重置 update rule 假设。}
}
$$

这不是“再调 FPO12 阈值”或“再把 generated 扩成 512”能解决的问题。v10.1 后，项目的主问题已经不是“从已有动作里挑一个好动作”，而是：

$$
\boxed{
\text{怎样构造一种小参数改动，使模型未来训练路径进入更好的函数空间区域？}
}
$$

---

# 1. v10.1 数据复盘

## 1.1 四线总览

v10.1 的关键结果：

```text
A 线 Fast / Slow / Risky / SafeLow / Bad = 8 / 26 / 521 / 34 / 19579
A discovery target pass = 0

B 线 FPO12 weak / discovery = 0 / 0
best = FPO12D-signal-reservoir-path-type-classifier
precision = 0.011494252873563218
V240 LCB = -0.18263554403155388

C 线 completed panels = 2
largest panel = 10000
natural density sufficient / insufficient / inconclusive = 0 / 1 / 0
C3 20000 = not_run because user cost cap

D 线 D0 / D1 / D2 = 1 / 1 / 0
D2 weak = 0
D3 256 opened = 0

P7 controller = not_run
P8 runtime = not_run
P9 paired replay = not_run
no fake / proxy / cpu = 0 / 0 / 0
```

## 1.2 对 C3 not_run 的判断

C3 20000 没执行是合理的成本控制，不是实验漏洞。v10.1 已经新跑了 5000 repeat 与 10000 confirmation，并且使用 strict-new-seed 模式，而不是复用 v10.0 source。C3 固化为 cost cap 后显式 `not_run`，且没有任何 20000 artifact。

因此 C 线结论应该这样写：

```text
自然动作密度在 5000 repeat + 10000 confirmation 下已经 insufficient；
20000 没跑，所以不能声称“无条件数学证明 natural density 永远不足”；
但足以把 natural harvesting 从主线降级为有限确认 / 辅助线。
```

更严谨地说：

$$
\boxed{
\text{C3 不执行不影响“natural harvesting 不能作为当前主线”的判断，}
}
$$

但它意味着：

$$
\boxed{
\text{若未来出现强反证，需要用更便宜的 sequential audit，而不是直接补 20000。}
}
$$

---

# 2. 四线独立判断

## 2.1 A 线：未来训练路径

A 线不是失败，而是给了一个更尖锐的事实：FastGood / SlowBurnGood 极少。

本轮 Fast + Slow 总数是：

$$
8 + 26 = 34.
$$

Risky + Bad 总数是：

$$
521 + 19579 = 20100.
$$

这说明在当前 action universe / generated discovery 中，真正符合未来路径好动作的区域非常稀疏。A 线的意义不是直接 controller，而是告诉我们目标形态：

```text
FastGood：当前和未来都好；
SlowBurnGood：当前不强甚至可能负，但后续 h20/h80/h240 变好；
RiskyHighAUV：未来值看起来大，但 longrisk / memory / offdiag 坏；
SafeLowValue：安全但没有足够收益；
BadPath：收益差或风险高。
```

最重要的是 `SlowBurnGood`。它代表我们真正想找的 functional update：

$$
\boxed{
\text{不是当前一步降 loss 最多，而是把后续训练带到更好轨迹。}
}
$$

但 A 线仍然不能 official，因为 path type 是事后标签，不是训练当下合法机制。

## 2.2 B 线：FuturePathOperator

B 线失败得很硬。`FPO12D-signal-reservoir-path-type-classifier` 是本轮最好，但 precision 只有 `0.01149`，V240 LCB 仍为负。

这说明：

```text
1. path-type classifier 当前没有学到好路径机制；
2. signal-reservoir / SNR / drift-diffusion 这些想法作为语言是对的，但当前实现没有捕捉到可用信号；
3. FPO 不能继续当作 scalar score 修补。
```

也就是说，B 线不应该继续问：

```text
FPO12D threshold 怎么调？
FPO12D feature 加哪个？
FPO12D topK 换成多少？
```

它应该问：

```text
为什么训练当下的 observable 完全无法预测 SlowBurnGood？
当前 features 是否根本缺少 future-path causal information？
是否需要直接构造 tiny virtual trajectory，而不是静态 feature classifier？
```

## 2.3 C 线：自然 AP0 动作扩流

C 线已经给出战略结论：自然 AP0 harvesting 不能继续作为主线。

v9.9.8 已经通过 G40 multi-marginal sampler 跑出真实 5000 panel 并显示 density insufficient；v10.0 又确认 largest 10000 insufficient；v10.1 使用 strict-new-seed 重新确认 C 线 insufficient。C3 20000 因成本没有跑，但两阶段证据已经足够支持路线降级。

现在 C 线应该从“主线”变成：

```text
有限确认线；
分布漂移监控线；
generated route 的自然参照线。
```

不应该继续把大量算力用于：

```text
20000 / 50000 natural panels；
重复估计 CoreLike density；
期待自然池子里自动出现足够 87 个好动作。
```

## 2.4 D 线：generated discovery

D 线按 family 执行 Stage8 -> Stage32 -> Stage64 discovery staircase，本轮 D0 / D1 过，D2 未过，D3 256 没打开。这是正确 gate。

但科学上，D 线目前非常弱：它没有产生足够 FastGood / SlowBurnGood，也没有形成 value-positive / risk-safe 的候选区域。结合 v9.9.8 的 64-action sandbox 结果，当前 generated objective 仍然没有对准 future-path-improving update。

这意味着不能做：

```text
把 Stage64 直接扩成 Stage256；
把 generated 家族数量扩大；
把 low-risk low-value 写成成功；
只看 D0/D1 工程 pass。
```

必须做：

```text
先解释 generated action 为什么落入 BadPath / RiskyHighAUV；
再重写目标；
每个家族只允许 8-action fail-fast；
没有 Fast/Slow-like seed 就停止该家族。
```

---

# 3. 这轮发现的本质问题

## 3.1 训练当下可识别机制比预想更难

v10.1 之后，必须承认：

$$
\boxed{
\text{训练当下可识别机制不是一个简单 feature learning 问题。}
}
$$

原因是好动作的价值不来自当前局部响应，而来自未来训练动力学。一个动作可能：

```text
h1 不好；
h20 开始变好；
h80/h240 继续变好；
longrisk 低；
memory/offdiag 不坏。
```

这类 `SlowBurnGood` 本质上是 trajectory-level effect。训练当下的静态特征很难预测它。

## 3.2 自然动作池不够，不能再期待 harvesting

自然动作池低密度说明：现有 AP0-style action distribution 不是为生成 future-path-improving update 设计的。它可以偶尔产生 Core77 / OldOnly 这种好动作，但不形成足够密度。

这就是：

$$
\boxed{
\text{existence 不等于 density；density 不够就不能 controller。}
}
$$

## 3.3 generated route 失败不是工程失败，而是目标失败

D 线不是没跑，也不是 payload 没 apply。它的问题是目标错。它仍然像是在生成合法扰动，而不是生成：

```text
future-path-improving；
risk-safe；
memory/offdiag-safe；
runtime-cheap；
train-time recognizable。
```

因此 generated route 需要 reset，不是扩容。

## 3.4 现在需要从“选择器问题”升级为“更新规则问题”

过去的问题是：

```text
有没有一个 selector 可以从动作池里挑好动作？
```

现在的问题是：

```text
怎样构造一个 update rule，使它自然产生 SlowBurnGood / FastGood？
```

这就是 v10.2 的核心。

---

# 4. 当前假设调整

## 4.1 需要降级的旧假设

### H_old_1：自然 AP0 harvesting 足够，只是 selector 不好

状态：基本降级。

理由：v9.9.8 / v10.0 / v10.1 均支持 natural density insufficient。C3 未跑不改变主线判断。

### H_old_2：FPO 只要调好特征就能识别好动作

状态：降级。

理由：FPO11 / FPO12 precision 都在 `0.01149` 附近，且 V/V240 为负。它不是 near miss。

### H_old_3：generated sandbox 多跑一些就能出现好动作

状态：降级。

理由：D 线 D2 weak fail，v9.9.8 64-action sandbox Fast/Slow 为 0，V/V240 负。

## 4.2 新主假设

### H_new_1：好 functional update 是 future-path operator，而不是 immediate response operator

一个动作 $a$ 的价值不应定义为：

$$
-g_t^\top \Delta_a.
$$

而应定义为：

$$
\Delta \mathcal{T}_{t:t+h}(a),
$$

即它对未来训练轨迹的影响。

### H_new_2：好动作要按 path type 生成，而不是按 scalar score 选择

目标不再是：

$$
score(a)>\tau.
$$

而是：

$$
Type(a)\in\{FastGood, SlowBurnGood\}
$$

并且：

$$
LongRisk(a)\le\tau_L,
\quad
MemoryOffdiagFail(a)\le\tau_M,
\quad
Cost(a)\le C_{max}.
$$

### H_new_3：generated optimizer 应该产生更新规则族，而不是 payload 变体族

生成器不应只是：

```text
APG family with payload templates。
```

它应该是：

```text
future-path-targeted update rule family。
```

换句话说，生成对象从 `payload` 升级为 `update rule`。

---

# 5. v10.2 总体目标

v10.2 的目标不是打开 official controller。v10.2 的目标是完成一次理论重置后的 discovery closure：

$$
\boxed{
\text{能否找到一种生成机制，使 8/32/64 小规模 discovery 中出现 FastGood 或 SlowBurnGood？}
}
$$

v10.2 成功不是 system pass，而是 discovery pass。

## v10.2 最低有效推进

满足以下任意一个，就算有效推进：

```text
1. D 线某个 generated family 在 8-action fail-fast 中产生 >= 1 个 Fast/Slow-like；
2. B 线某个 path-type predictor 能把 generated false positive 与 true SlowBurn 分开；
3. A 线证明 SlowBurnGood 的可生成机制，比如 delayed gain / memory-safe drift / low-curvature path；
4. 明确证明当前 generated objective 无法产生 Fast/Slow，并给出应停止的机制原因。
```

---

# 6. v10.2 实验计划

## P0：复现 v10.1 边界与 artifact lock

### 目标

复现 v10.1 的 terminal route，锁定所有输入 artifact，防止把旧 diagnostic 误当新结果。

### 假设

v10.1 的边界稳定：

```text
natural_density_insufficient = 1
FPO12 discovery pass = 0
generated discovery pass = 0
controller/runtime/paired replay = not_run
```

### 必须记录

```text
source_route_v1010
A Fast/Slow/Risky/SafeLow/Bad counts
B FPO12 best precision, V240 LCB
C panel sizes completed, C3 status, cost cap flag
D stage statuses
No-fake/proxy/cpu rows
artifact hash table
```

### 判断标准

P0 pass：

```text
source route exactly R4-FPOAndGeneratedBothFail_UpdateRuleTheoryReset
C3 status = not_run
C3_user_cost_cap = 1
fake/proxy/cpu = 0/0/0
```

### 不满足时 Codex 先尝试

```text
如果 route mismatch：检查是否读取了 v10.0 而不是 v10.1 artifact。
如果 C3 artifact 存在：检查是否误跑 20000，不能纳入 official。
如果 no-fake audit 不为 0：停止后续所有阶段。
```

### 可视化

```text
p0_v1010_boundary_table.md
p0_stage_gate_flow.svg
```

---

## P1：A 线路径类型机制固化

### 目标

把 FastGood / SlowBurnGood / RiskyHighAUV / SafeLowValue / BadPath 固定为 v10.2 的 discovery target，不再用 AUV 或 RAUV 单分数替代。

### 假设

`SlowBurnGood` 是 future-path update 的关键类型，它的形态是：

$$
V_1 \le \epsilon_1,
$$

$$
V_{20}^{LCB}>0,
$$

$$
V_{80}^{LCB}>0,
$$

$$
V_{240}^{LCB}>0,
$$

且：

$$
LongRisk_{240}^{UCB}\le0.05.
$$

### 必须记录

对每个 action 记录：

```text
action_id
source_family
path_type
V1, V5, V20, V80, V240
V1_LCB, V20_LCB, V80_LCB, V240_LCB
AUV, RAUV
longrisk_240
bad_event
null_event
memory_fail
offdiag_fail
hardtail_delta
old_family_delta
payload_norm
action_norm
cosine_to_adamw
dataset / seed / family / template only for diagnostics, not controller
```

### 判断标准

P1 discovery pass：

```text
FastGood + SlowBurnGood count >= 20
SlowBurnGood count >= 10
SlowBurnGood longrisk UCB <= 0.05
SlowBurnGood memory/offdiag UCB <= 0.05
SlowBurnGood not single dataset/family/template pocket
```

P1 strong pass：

```text
FastGood + SlowBurnGood count >= 87
V240 LCB > 0
longrisk/bad/null/memory/offdiag all within gate
LDO/LSO/LTO/LFO <= 0.10
```

### 不满足时 Codex 先尝试

```text
如果 SlowBurnGood 太少：
  放宽 V1 条件；不要放宽 V240 或 longrisk。
  检查 h80/h240 是否太严；尝试 SlowBurnRelaxed 作为 diagnostic。

如果 RiskyHighAUV 太多：
  强化 longrisk/memory/offdiag hard gate；不要提高 AUV threshold。

如果 SafeLowValue 多：
  检查是否是更长 horizon 才变好；可以加 h480 diagnostic，但不能 official。

如果 path type 被 dataset/family 垄断：
  做 group/drop autopsy；不按 dataset 调规则。
```

### 可视化

```text
p1_path_type_counts_bar.svg
p1_path_type_v_curve_h1_h5_h20_h80_h240.svg
p1_slowburn_vs_fastgood_risk_scatter.svg
p1_path_type_family_dataset_heatmap.svg
p1_v1_vs_v240_quadrant.svg
```

---

## P2：B 线 FPO v13，改成 path-type predictor

### 目标

停止 FPO12 scalar score 修补，改成训练当下预测 path type。

### 核心思想

训练当下不直接预测 GoodAction，而是预测：

$$
P(Type(a)=FastGood),
$$

$$
P(Type(a)=SlowBurnGood),
$$

$$
P(Type(a)=RiskyHighAUV),
$$

$$
P(Type(a)=BadPath).
$$

最终接受分数：

$$
S(a)=P(FastGood)+P(SlowBurnGood)-\lambda_R P(RiskyHighAUV)-\lambda_B P(BadPath).
$$

但 official controller 不能使用 future label；future label 只用于 discovery split 训练与 heldout 验证。

### 候选 FPO

```text
FPO13A: tiny-virtual-trajectory-1step
FPO13B: tiny-virtual-trajectory-3step
FPO13C: JVP/VJP future-gradient-alignment
FPO13D: memory-offdiag-hardgate + delayed-gain proxy
FPO13E: path-type classifier with hard risk veto
FPO13F: two-head path predictor, head1 value path, head2 risk path
FPO13G: calibrated SlowBurn detector
FPO13H: negative-control shuffled path predictor
```

### 必须记录

```text
fpo_id
feature_legality = green/yellow/red
cost_q50/q90/q99_ms
training_split / heldout_split
TopK8/32/64/87 path_type counts
Fast/Slow precision
Risky/Bad false positive rate
V1/V20/V80/V240 LCB
longrisk UCB
bad/null UCB
memory/offdiag UCB
LDO/LSO/LTO/LFO
calibration ECE only diagnostic
```

### 判断标准

P2 weak pass：

```text
TopK32 Fast+Slow precision >= 0.20
V240 LCB > 0
longrisk UCB <= 0.20
cost q90 <= 3 ms
```

P2 discovery pass：

```text
TopK64 Fast+Slow precision >= 0.30
V240 LCB > 0
longrisk UCB <= 0.10
RiskyHighAUV false positive <= 0.20
cost q90 <= 1.5 ms
```

P2 controller candidate pass：

```text
TopK87 Fast+Slow precision >= 0.75
V/V240 LCB > 0
longrisk/bad/null/memory/offdiag within gate
LDO/LSO/LTO/LFO <= 0.10
cost q90 <= 1.5 ms
```

### 不满足时 Codex 先尝试

```text
如果 precision 低但 risk 低：
  这个 FPO 是 veto，不是 selector；把它移到 hard gate，不继续调阈值。

如果 precision 高但 longrisk 高：
  加 memory/offdiag hard gate；不要调 value head。

如果 SlowBurn false negative 高：
  加 delayed-gain features；检查 V1-negative cases 是否被误过滤。

如果 RiskyHighAUV false positive 高：
  加 longrisk/memory/offdiag path contrast loss，但不改 task CE。

如果 cost q90 > 3 ms：
  减少 virtual steps；缓存 JVP/VJP；只用 stratified mini-buffer；禁止进入 controller。

如果 FPO13H negative control 也好：
  说明 leakage 或 label artifact；停止该 FPO family。
```

### 可视化

```text
p2_fpo_path_type_confusion_matrix.svg
p2_slowburn_recall_vs_cost.svg
p2_risky_false_positive_by_feature.svg
p2_topk_path_type_stack.svg
p2_cost_quality_pareto.svg
```

---

## P3：C 线自然密度有限确认，不再做无限扩流

### 目标

确认 natural harvesting 是否可以彻底降级。C3 20000 不跑，改用低成本 confirmation。

### 假设

v10.1 的 5000 repeat + 10000 confirmation 足以支持 natural density insufficient。v10.2 只做 sanity confirmation，不再做 20000。

### 必须记录

```text
panel_id
seed
panel_size
completed_rows
CoreLike count / LCB / UCB
PathGood count / LCB / UCB
FastGood count / LCB / UCB
SlowBurnGood count / LCB / UCB
CoreLike+SlowBurn UCB
generator profile
major/tail PSI/JS
missing tail
runtime / wallclock / peak gpu
```

### 判断标准

C 线关闭主线：

```text
Two independent panels with size >= 5000 have CoreLike+SlowBurn UCB < 0.03
and at least one panel size >= 10000 also has UCB < 0.03.
```

C 线保留主线：

```text
Any confirmed panel has CoreLike+SlowBurn LCB >= 0.03.
```

C 线不确定：

```text
LCB < 0.03 <= UCB，且 cost cap 不允许更大 panel。
```

### 不满足时 Codex 先尝试

```text
如果 panel distribution fidelity 失败：
  回到 G40/G-IPF sampler，不做 density 结论。

如果 5000 与 10000 contradiction：
  不跑 20000；先做 stratified contradiction audit，找 family/template/tail group 原因。

如果 runtime 超预算：
  降低 panel shard；启用 resume；不要 CPU offload；不要 fake rows。

如果 CoreLike 低但 SlowBurn 高：
  重新定义 density target 为 PathGood = Fast+Slow，而不是 CoreLike。
```

### 可视化

```text
p3_density_ci_by_panel.svg
p3_corelike_slowburn_density_curve.svg
p3_tail_group_good_density_heatmap.svg
p3_panel_runtime_cost_curve.svg
```

---

## P4：D 线 generated discovery，future-path-targeted 8 -> 32 -> 64 staircase

### 目标

D 线不再生成普通 payload 变体，而是生成 future-path-targeted update。每个 family 必须先过 8-action fail-fast。

### 生成家族

```text
GUP13A: SlowBurn target, small norm, memory/offdiag hard gate
GUP13B: delayed-gain residual, low immediate value allowed
GUP13C: future-gradient alignment, JVP/VJP constrained
GUP13D: risk-adjusted path update, longrisk hard veto
GUP13E: signal-channel drift update, reservoir suppression
GUP13F: cover-stability update, no collapse hard gate
GUP13G: negative control shuffled payload
GUP13H: low-risk low-value control
```

### 必须记录

```text
generated_action_id
family_id
stage = 8 / 32 / 64
source_path_type_target
payload_norm
action_norm
cosine_to_adamw
memory/offdiag proxy
future path rows h1/h5/h20/h80/h240
Fast/Slow/Risky/SafeLow/Bad label
V/V240 LCB
longrisk UCB
bad/null UCB
memory/offdiag UCB
new_positive_created
longrisk_created
Damage_V
Damage_path_type
```

### 判断标准

Stage8 open-to-32：

```text
Fast+Slow count >= 1
or SafeLowValue count >= 2 with V240 LCB not strongly negative
longrisk_created_rate <= 0.25
Damage V LCB >= -0.50
```

Stage32 open-to-64：

```text
Fast+Slow precision >= 0.10
V240 LCB > -0.10
longrisk UCB <= 0.30
memory/offdiag UCB <= 0.30
```

Stage64 discovery pass：

```text
Fast+Slow precision >= 0.20
V240 LCB > 0
longrisk UCB <= 0.20
new_positive_created >= 0.05
longrisk_created_rate <= 0.20
```

Generated route official candidate 仍需要更高门槛：

```text
accepted_count >= 87
Fast+Slow precision >= 0.75
V/V240 LCB > 0
longrisk/bad/null/memory/offdiag within official gate
runtime pass
```

### 不满足时 Codex 先尝试

```text
如果 Stage8 全 BadPath：
  停止该 family；检查 payload norm、action norm、memory/offdiag fail 是否异常。

如果 high AUV high risk：
  加 risk hard gate，不调 AUV。

如果 low risk low value：
  判断 SafeLowValue 是否延迟收益；必要时检查更长 horizon diagnostic，但不 official。

如果 all value negative：
  说明 generation target 错；回到 A/B 线，不扩大到 32。

如果 negative control 也出现 Fast/Slow：
  检查 label leakage 或 branch-horizon bug。
```

### 可视化

```text
p4_generated_stage_waterfall.svg
p4_generated_path_type_by_family.svg
p4_generated_v240_vs_longrisk.svg
p4_generated_payload_norm_vs_path_type.svg
p4_generated_damage_sankey.svg
```

---

## P5：失败机制归因

### 目标

如果 B/D 都失败，不能只写 failed，要归因为什么失败。

### 失败类别

```text
F1-natural-density-insufficient
F2-FPO-measures-current-response-not-future-path
F3-FPO-veto-only-no-value-selector
F4-generated-payload-OOD
F5-generated-memory-offdiag-damage
F6-generated-delayed-gain-target-missing
F7-generated-risk-gate-too-weak
F8-path-label-too-sparse
F9-action-space-wrong
F10-update-rule-theory-insufficient
```

### 必须记录

```text
failure_class
assigned_count
assigned_fraction
dominant_family
dominant_path_type
example_action_ids
recommended_next_action
```

### 判断标准

P5 pass：

```text
assigned_fraction >= 0.80
unknown_fraction <= 0.20
at least one dominant failure class identified
```

### 不满足时 Codex 先尝试

```text
如果 unknown high：
  增加 action-level trace fields，不增加新 generator。

如果 F4 dominant：
  限制 payload/action norm，检查 tail group distribution。

如果 F5 dominant：
  加 memory/offdiag hard gate。

如果 F10 dominant：
  停止当前 update family，回到理论设计。
```

### 可视化

```text
p5_failure_taxonomy_bar.svg
p5_family_failure_heatmap.svg
p5_path_type_failure_sankey.svg
```

---

## P6：controller boundary

### 目标

只有 C/B/D 至少一条线过硬门，才打开 controller。

### 开启条件

```text
C line density sufficient
or B line FPO controller candidate pass
or D line generated Stage64 discovery pass
```

### 必须记录

```text
controller_id
source = C / B / D
accepted_count
Fast/Slow precision
V/V240 LCB
longrisk/bad/null UCB
memory/offdiag UCB
LDO/LSO/LTO/LFO
feature cost q90
controller decision latency
```

### 判断标准

P6 pass：

```text
accepted_count >= 87
Fast+Slow precision >= 0.75
V LCB > 0
V240 LCB > 0
longrisk UCB <= 0.05
bad UCB <= 0.05
null UCB <= 0.15
memory/offdiag UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
```

### 不满足时 Codex 先尝试

```text
如果 accepted_count < 87：
  不放宽 risk；先查 source density 或 generated family。

如果 precision low：
  回到 B/D failure autopsy，不调 threshold。

如果 longrisk high：
  加 hard gate；不要把 V threshold 提高。

如果 LDO high：
  做 support/backfill decomposition；不按 dataset 调参。
```

---

## P7：selected runtime boundary

### 目标

controller 过线后，测真实 runtime。

### 必须记录

```text
step_count
active_step_count
accepted_per_step
feature_ms_q50/q90/q99
controller_ms_q50/q90/q99
payload_apply_ms_q50/q90/q99
kernel_count
sync_count
step_ratio_q50/q90/q99
```

### 判断标准

P7 pass：

```text
step_ratio_q90 <= 1.50
no CPU offload
no fake/proxy rows
no materializer in timed path
```

### 不满足时 Codex 先尝试

```text
如果 feature cost high：
  cache FPO features；reduce virtual samples；batch-major group active steps。

如果 payload apply high：
  fuse apply kernel；reduce payload sparsity fragmentation。

如果 sync high：
  remove per-step sync；persistent workspace。
```

---

## P8：paired replay / short-full boundary

### 目标

只有 controller + runtime 都过，才打开 paired replay。

### 必须比较

```text
RealFunctional
AdamWParallel
BestLR
NoOp
RandomPayload
ShuffledPayload
FPO-only-no-payload
Generated-negative-control
```

### 判断标准

P8 pass：

```text
RealFunctional beats all controls on V/V240
longrisk not higher than controls
bad/null within gate
memory/offdiag safe
leaveout stable
```

### 不满足时 Codex 先尝试

```text
如果 shuffled payload 也好：
  检查 payload identity bug。

如果 NoOp 差异不显著：
  functional update effect too small；回到 D family design。

如果 AdamWParallel 更好：
  update rule conflicts with AdamW；检查 cosine/alignment。
```

---

# 7. v10.2 成功 / 停止 / 转向规则

## 7.1 成功规则

v10.2 discovery success：

```text
B 或 D 至少一条线产生 Fast/Slow-like positive evidence；
且 V240 LCB > 0；
且 longrisk/memory/offdiag 不爆；
且 negative controls fail。
```

v10.2 official success：

```text
P6 controller pass
P7 runtime pass
P8 paired replay pass
```

## 7.2 停止规则

如果满足以下条件，停止当前 generated objective：

```text
D Stage8 连续 3 个 family 全 BadPath；
或 Stage32 Fast+Slow precision = 0；
或 V240 LCB < -0.5 且 longrisk UCB > 0.50；
或 negative control 与 real family 同样好。
```

如果满足以下条件，停止当前 FPO family：

```text
FPO precision < 0.05 且 V/V240 LCB < 0；
或 FPO 只会 low-risk low-value；
或 cost q90 > 3 ms 且质量不过。
```

## 7.3 转向规则

如果 B/D 都失败，且 C 已确认 natural insufficient，则进入理论重置：

```text
v10.3 = Update Rule Theory Rebuild
目标：从未来训练动力学推导可生成的 update rule，
不再从 AP0 payload family 里试错。
```

---

# 8. 本轮最终判断

v10.1 后，最准确的项目状态是：

$$
\boxed{
\text{future path 好动作现象可信；natural harvesting 已基本不足；FPO 识别失败；generated 初版失败。}
}
$$

所以 v10.2 不是小修。v10.2 是一次 update-rule discovery reset：

```text
A 线定义目标；
B 线尝试训练当下识别；
C 线只做有限确认；
D 线主动生成 future-path update；
P6-P8 只在 B/C/D 过硬门后打开。
```

如果 D 线仍然完全没有 Fast/Slow-like，那么我们应该诚实承认：当前 action/payload family 不足以构造目标更新，需要进入更高层的理论设计。
