# DG-KAN v9.8.4 未来轨迹算子 / 自然动作扩流 / 几何自适应优化器完整实验计划

> 基于 v9.8.3 `Four-Line Future Mechanism / Natural Stream / Geometry Adaptive` 的真实执行结果制定。  
> 本计划不继续小修 `OldRank_score`、`M_signal`、`memory_offdiag_safe`、P4 gate、或 APG/APGU/APGX 小变体。  
> 本计划的目标是把项目从“事后知道哪些动作好”推进到“训练当下能用合法、跨任务、跨数据分布的方式理解网络几何，并产生或选择能改善未来训练轨迹的小参数改动”。

---

## 0. 一句话判断

v9.8.3 的结果说明：

$$
\boxed{
\text{未来训练轨迹里确实存在好动作现象，但训练当下合法机制仍未找到，自然动作扩流也未落地。}
}
$$

所以 v9.8.4 的核心问题不是：

```text
OldRank_score 再调一下？
P2 legal mechanism 再加一个字段？
P4 official gate 能不能换？
APG/APGU 再造一个小变体？
```

而是：

```text
1. 好动作为什么会让未来训练路径变好？
2. 这种未来路径效应能不能用训练当下可计算的东西近似出来？
3. 自然 AP0 动作流扩大后，好动作密度是否够？
4. 如果已有动作流不够，能不能直接生成一种几何自适应更新？
```

本轮采用四线并行，不再串行一轮只清一个 blocker。

---

## 1. v9.8.3 的独立判断

### 1.1 v9.8.3 有进展，但不是能力成功

v9.8.3 的 route 是：

```text
route = RouteB-FuturePathExistsMechanismAbsent
primary_blocker = no_legal_training_time_mechanism
secondary_blocker = natural_stream_materializer_missing
route_secondary = RouteD-NaturalStreamMaterializerMissing
system_legal_controller_pass = 0
generated_route_status = stopped_no_legal_mechanism_or_natural_density_evidence
```

这不是 strict PureKAN functional success，也不是 controller success。P6-P10 全部 gate-blocked，没有 selected controller、selected runtime、official paired replay、short/full training。

但 v9.8.3 不是空转。它真正推进的是：未来路径现象更清楚了。

P1 的关键结果：

```text
Core77:
  actions = 77
  AUV LCB = 2.8296932871691163
  V1 LCB = 0.08815910625793319
  V20 LCB = 1.573549156031694
  V80 LCB = 3.1200217020467886
  V240 LCB = 4.750885696264312
  longrisk UCB = 0.03827572901862327
  memory UCB = 0.0
  offdiag UCB = 0.0

OldOnly:
  actions = 10
  AUV LCB = 1.0287178879851673
  V1 LCB = -0.174770564086397
  V20 LCB = 0.30014089601530236
  V80 LCB = 1.1093912874075194
  V240 LCB = 2.0678046281088522
  longrisk UCB = 0.0
  memory UCB = 0.0
  offdiag UCB = 0.0

RandomMatched:
  actions = 87
  AUV LCB = 1.8284728866377598
  V240 LCB = 3.5922551893613908
  longrisk UCB = 0.8485412092119363
  memory UCB = 0.9433993751492317
  offdiag UCB = 0.9605466806820027
```

这组数据说明：

```text
Core77 未来路径强，而且风险低。
OldOnly 不是 immediate descent：V1 为负，但 AUV 为正。
RandomMatched 也可能有大 AUV，但 longrisk / memory / offdiag 崩。
```

所以未来路径不是简单看 AUV 大小。好动作应该是：

$$
\boxed{
\text{未来路径增益为正，同时 longrisk / memory / offdiag 不崩。}
}
$$

### 1.2 v9.8.3 最大 blocker 不是未来路径，而是训练当下合法机制

P2 没有任何合法机制通过：

```text
P2 legal mechanism weak pass count = 0
P2 legal mechanism strong pass count = 0
```

关键机制结果：

```text
M5_oldrank_signal_diagnostic:
  legality = red:old_table_diagnostic
  precision = 0.8735632183908046
  V LCB = 0.14061570456586386
  AUV LCB = 2.7227479106902694
  longrisk UCB = 0.0
  LDO = 0.37931034482758624

M3_old_knowledge_memory_safety:
  legality = green
  precision = 0.41379310344827586
  V LCB = -0.11427822648412846
  AUV LCB = 2.716555586178205
  longrisk UCB = 0.0
  memory UCB = 0.0
  offdiag UCB = 0.0
  LDO = 0.09195402298850575
```

解释：

```text
OldRank 能选好动作，但它是 red diagnostic，不能上线。
Memory/offdiag 能压风险，但选不出 value-positive 动作。
```

所以 memory/offdiag 目前更像安全门，不是收益源。

### 1.3 自然 AP0 扩流仍然是硬 blocker

P3 仍然没有落地自然 AP0 扩流：

```text
Panel-target-2876:
  labeled = 2876

Panel-target-5000:
  not_run
  expected rows = 150000
  actual rows = 0
  missing labels = 2124

Panel-target-10000:
  not_run
  expected rows = 300000
  actual rows = 0
  missing labels = 7124

Panel-target-20000:
  not_run
  expected rows = 600000
  actual rows = 0
  missing labels = 17124
```

这意味着我们还不知道：

```text
自然动作流扩大后，Core-like 动作密度是否能稳定超过 0.03。
```

如果自然动作密度够，existing-action route 还有希望。  
如果自然动作密度不够，就必须转向生成新的 update rule。

### 1.4 P4 gate 不允许修改是正确的

P4 继续显示：

```text
raw_LDO = 0.4415584415584416
quality_LDO = 0.09260529551331587
support_LDO = 0.0
backfill_LDO = 0.34895314604512573
backfill_share = 0.7902762425139611
official_gate_modification_allowed = 0
```

backfill 解释了 raw LDO 的大部分，但 P3 自然扩流没落地，所以不能修改 official raw gate。否则会把“动作密度不足”误写成“gate 太保守”。

### 1.5 P5 说明诊断特征有用，但合法强机制仍没有

P5 的 OldRank_score 仍然强：

```text
OldRank_score:
  legality = red:old_table_diagnostic
  precision = 0.8735632183908046
  V LCB = 0.14061570456586386
  AUV LCB = 2.7227479106902694
  longrisk UCB = 0.0
```

但 green 特征仍不够：

```text
memory_offdiag_safe:
  legality = green
  precision = 0.41379310344827586
  V LCB = -0.11427822648412846
  AUV LCB = 2.716555586178205
  longrisk UCB = 0.0
```

结论：

$$
\boxed{
\text{有能解释风险的合法信号，但没有能解释收益的合法信号。}
}
$$

---

## 2. 当前最深层问题

现在不能把问题简化成：

```text
动作不够；
LDO 不过；
legal feature 不够；
APG 没跑；
```

这些都是表象。

更深的问题是：

$$
\boxed{
\text{我们已经能看到未来路径里的好动作，但还不知道优化器在训练当下该如何理解这种几何。}
}
$$

这也是为什么项目看起来慢：我们不是在提升一个已有 controller，而是在从经验筛选动作转向几何自适应优化器。

---

## 3. 数学视角：为什么不能只看当前一步 loss

一个动作是一个小参数改动：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{AdamW}+\Delta.
$$

过去 exact transfer 测的是当前一步：

$$
-g_t^\top \Delta.
$$

但真实好动作的价值是未来训练路径：

$$
V_h(\Delta)=L(\Phi_h(\theta_t))-L(\Phi_h(\theta_t+\Delta)),
$$

其中 $\Phi_h$ 是后续 $h$ 步训练动力学。

如果未来训练动力学会放大、旋转、抵消或修正当前动作，那么当前一步 loss response 就不是充分条件。

v9.8.3 的 OldOnly 结果正好支持这一点：

$$
V_1 < 0,
\quad
AUV > 0.
$$

所以：

$$
\boxed{
\text{好 functional update 是 future-path steering，不是 one-step descent。}
}
$$

Deep Manifold 的启发是，神经网络训练是在移动局部 cover、改变坐标、形成稳定 fixed-point region；优化器要理解的是这种演化几何，而不是某个固定基函数。Generalization paper 的启发是，优化方向应进入 signal channel，而不是只降低训练误差的 reservoir/noise 方向。

---

## 4. v9.8.4 总体目标

v9.8.4 的目标是：

$$
\boxed{
\text{把 future-path good action 从事后现象，推进成训练当下可估计、可选择、可生成的优化原则。}
}
$$

具体分成四线。

```text
A 线：完整未来路径与机制分解
B 线：训练当下合法几何机制发现
C 线：自然 AP0 动作扩流与密度判定
D 线：几何自适应生成器重开 gate
```

四线并行，不再串行等待。

---

# Part A：完整未来路径与机制分解

## A0. 目标

回答：

$$
\boxed{
\text{Core77 / OldOnly 为什么未来路径好？ExactOnly / Random 为什么不是好动作？}
}
$$

v9.8.3 已经说明 Core77 / OldOnly future path 好，但还没有解释机制。

## A1. 实验对象

至少包含四组：

```text
Core77：当前最干净的好动作。
OldOnly：OldRank 选中、ExactTransfer 没选中，但结果好的动作。
ExactOnly：ExactTransfer 选中、OldRank 没选中，但结果差的动作。
RandomMatched：按 dataset / seed / family / step / norm 匹配的随机动作。
```

新增两组：

```text
CoreExpansion10：补足 Core77 到 87 的扩展动作。
RiskCleanButLowValue：longrisk/memory/offdiag 安全但 V 低的动作。
```

## A2. 必须记录的路径指标

对每个 action 和每个 horizon：

```text
horizon = 1, 5, 20, 80, 240
```

记录：

```text
V_h
V_h_LCB
CE_delta_h
NLL_delta_h
margin_delta_h
CEp99_delta_h
hard_tail_loss_delta_h
old_family_loss_delta_h
old_stratum_loss_delta_h
memory_buffer_loss_delta_h
longrisk_h
bad_event_h
null_event_h
memory_fail_h
offdiag_fail_h
cover_entropy_delta_h
basis_effective_rank_delta_h
function_displacement_norm_h
AdamW_alignment_h
```

注意：cover / basis 指标可以继续作为 KAN-specific diagnostic，但不能作为唯一理论基础。必须同时记录 architecture-agnostic 的 function/output response。

## A3. 新增路径形状指标

定义：

$$
AUV(a)=\sum_{h\in\{1,5,20,80,240\}}w_h V_h(a),
$$

$$
Slope_{early}(a)=V_5(a)-V_1(a),
$$

$$
Slope_{mid}(a)=V_{80}(a)-V_{20}(a),
$$

$$
Slope_{late}(a)=V_{240}(a)-V_{80}(a),
$$

$$
RiskPath(a)=\max_h LongRisk_h(a).
$$

还要记录 immediate-not-dominant：

$$
IND(a)=1
$$

当且仅当：

$$
V_1(a)\le 0,
\quad
AUV(a)>0,
\quad
LongRisk_{240}(a)\le \tau_L.
$$

## A4. 判断标准

A 线 weak pass：

```text
Core77 AUV LCB > 0；
Core77 h240 longrisk UCB <= 0.05；
Core77 memory/offdiag UCB <= 0.05；
OldOnly 存在 immediate-not-dominant 证据；
ExactOnly 或 RandomMatched 至少一个表现出 value/risk mismatch。
```

A 线 strong pass：

```text
Core77 / OldOnly 的 AUV LCB 显著高于 ExactOnly / RandomMatched；
Core77 / OldOnly 的 RiskPath 显著更低；
Core77 / OldOnly 的 old-family / memory / hard-tail 路径更稳；
差异在 dataset-blind split 下仍成立。
```

## A5. 可视化

```text
fig_A1_group_future_path_V_curve.svg
fig_A2_group_future_path_risk_curve.svg
fig_A3_oldonly_immediate_vs_future.svg
fig_A4_exactonly_failure_path.svg
fig_A5_core77_vs_random_memory_offdiag_path.svg
fig_A6_path_shape_slope_scatter.svg
fig_A7_AUV_vs_longrisk_pareto.svg
```

---

# Part B：训练当下合法几何机制发现

## B0. 目标

回答：

$$
\boxed{
\text{训练当下是否存在合法信号，可以预测 A 线发现的未来路径好动作？}
}
$$

合法信号必须满足：

```text
不使用 future outcome；
不使用 old table diagnostic；
不使用 dataset name branch；
不使用 validation/test；
不使用 replay 后 label；
不使用 red feature。
```

## B1. 机制候选，不再做复杂手工加权

本轮不设计一个大 score，而是先做机制族对照。

### B1.1 Function-space local response

记录 action 对当前训练 batch / train-memory / hard-tail probe 的输出响应：

```text
output_delta_norm_current
output_delta_norm_memory
output_delta_norm_hard_tail
logit_margin_delta_current
logit_margin_delta_memory
CE_linear_delta_current
CE_linear_delta_memory
```

### B1.2 Signal consistency

记录多个样本是否支持同一个方向：

```text
per_example_response_mean
per_example_response_std
response_SNR = mean^2 / (var + eps)
positive_response_fraction
leave_one_out_response_stability
```

### B1.3 Memory / old knowledge safety

记录：

```text
old_family_probe_loss_delta
old_family_margin_delta
old_stratum_probe_loss_delta
memory_buffer_loss_delta
memory_offdiag_safe
```

### B1.4 Path proxy probe

允许一个小型 legal microprobe，但必须在训练当下完成，不能读取 future labels：

```text
virtual_steps = 1, 3, 5
probe_subset_size = 16 或 32
probe uses current train-memory buffer only
probe cannot use dataset name
probe cannot use heldout/test
```

记录：

```text
virtual_path_V_proxy
virtual_path_risk_proxy
virtual_path_memory_proxy
virtual_path_cost_ms
```

## B2. 机制评价方式

不以 AUC 为主，直接看 accepted region。

每个机制候选输出 TopK87，记录：

```text
accepted_count
GradeAB_precision
CoreLike_precision
AUV_LCB
V20_LCB
V80_LCB
V240_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO_drop
LSO_drop
LTO_drop
LFO_drop
feature_cost_q90_ms
```

## B3. 判断标准

B 线 weak pass：

```text
accepted_count >= 87；
GradeAB precision >= 0.75 或 CoreLike precision >= 0.75；
AUV LCB > 0；
longrisk UCB <= 0.05；
memory/offdiag UCB <= 0.05；
LDO/LSO/LTO/LFO drop <= 0.20；
feature cost q90 <= 0.25 ms。
```

B 线 strong pass：

```text
LDO/LSO/LTO/LFO drop <= 0.10；
feature cost q90 <= 0.10 ms；
not KAN-basis-specific；
pass on all four A-line groups without group-specific tuning。
```

## B4. 可视化

```text
fig_B1_mechanism_topK_quality_bar.svg
fig_B2_mechanism_auv_vs_risk.svg
fig_B3_signal_consistency_vs_future_AUV.svg
fig_B4_virtual_probe_cost_quality_pareto.svg
fig_B5_memory_safety_as_veto.svg
fig_B6_legal_vs_red_feature_comparison.svg
```

---

# Part C：自然 AP0 动作扩流与密度判定

## C0. 目标

回答：

$$
\boxed{
\text{自然 AP0 动作流扩大后，干净好动作的密度是否足够支撑 controller？}
}
$$

这是当前最拖慢项目的工程 blocker。v9.8.3 仍然没有 5000/10000/20000 panel。

## C1. 必须先做 materializer preflight

不能再 full runner 后才发现 materializer 不存在。

分四级：

```text
C1a: 1 action preflight；
C1b: 16 action preflight；
C1c: 128 action preflight；
C1d: 512 action preflight。
```

每一级都必须记录：

```text
action_count
branch_count
horizon_count
expected_rows
actual_rows
missing_labels
duplicate_rows
NaN/Inf
label_exclusivity_violation
runtime_rows_per_sec
OOM_count
retry_count
no_fake/proxy/cpu
```

只有 C1d 过，才能跑 panels。

## C2. Panels

```text
Panel A: 2876 existing AP0 actions
Panel B: 5000 natural AP0 actions
Panel C: 10000 natural AP0 actions
Panel D: 20000 natural AP0 actions
```

每个 panel 记录：

```text
CoreLike_count
CoreLike_rate
CoreLike_Wilson_LCB/UCB
GradeAB_count/rate
ValuePositiveNoLongRisk_count/rate
MemoryOffdiagCore_count/rate
T5-like_count/rate
per_dataset_rate
per_family_rate
per_template_rate
per_step_bucket_rate
```

## C3. 判断标准

C 线 weak pass：

```text
Panel B/C/D 至少一个完成；
CoreLike LCB >= 0.03 或 GradeAB clean LCB >= 0.03；
no fake/proxy/cpu；
quality audit pass。
```

C 线 strong pass：

```text
Panel C 或 D 完成；
CoreLike LCB >= 0.03；
每个 dataset 的 CoreLike rate LCB >= 0.02；
per-family/template 不塌缩；
runtime materializer throughput >= 15 rows/sec。
```

## C4. 可视化

```text
fig_C1_density_curve_panel_size.svg
fig_C2_corelike_rate_ci.svg
fig_C3_per_dataset_density.svg
fig_C4_per_family_density_heatmap.svg
fig_C5_materializer_throughput_curve.svg
fig_C6_missing_label_dashboard.svg
```

---

# Part D：几何自适应生成器重开 gate

## D0. 目标

回答：

$$
\boxed{
\text{如果已有动作不够，能否根据 A/B 线发现的机制直接生成小更新？}
}
$$

注意：D 线不允许盲目重开 APG/APGU/APGX。只有 A 或 B 给出机制后，才允许有限重开。

## D1. Generated route reopen 条件

满足任一条件才允许 D2：

```text
Condition 1:
  B 线找到 legal mechanism weak pass；

Condition 2:
  C 线证明自然动作密度 < 0.03，但 A 线证明明确 future trajectory mechanism；

Condition 3:
  virtual path probe 能在训练当下预测 AUV，且成本 <= 0.25 ms。
```

否则：

```text
generated_route_status = stopped_no_legal_mechanism_or_density_evidence
```

## D2. 新生成器原则

新生成器不再拘泥于 KAN 当前基函数。更新应定义在函数响应空间或低秩参数子空间中。

候选形式：

$$
\Delta^* = \arg\max_{\Delta \in \mathcal{S}} \widehat{AUV}(\Delta)
$$

约束：

$$
MemoryRisk(\Delta) \le \tau_M,
$$

$$
OffdiagRisk(\Delta) \le \tau_O,
$$

$$
LongRiskProxy(\Delta) \le \tau_L,
$$

$$
Cost(\Delta) \le C_{max},
$$

$$
\|\Delta\| \le \epsilon.
$$

其中 $\mathcal{S}$ 可以是：

```text
last-layer low-rank subspace；
edge-function coefficient subspace；
AdamW residual subspace；
function-response constrained subspace；
architecture-agnostic output-response subspace。
```

## D3. Generated action smoke

每个 generator 先跑：

```text
16 actions preflight；
64 actions smoke；
512 actions full smoke。
```

记录：

```text
payload_hash_missing
certificate_hash_missing
action_apply_linf_max
no_transform_equivalence
branch_horizon_expected/actual
V20/V80/V240
AUV
longrisk/bad/null
memory/offdiag
new_positive_created_rate
longrisk_created_rate
runtime_apply_ms
```

## D4. 判断标准

Generated weak pass：

```text
512 actions 完整 materialized；
GradeAB precision >= 0.25；
AUV LCB > 0；
longrisk UCB <= 0.10；
new_positive_created_rate >= 0.10；
longrisk_created_rate <= 0.20。
```

Generated strong pass：

```text
GradeAB precision >= 0.50；
AUV LCB > 0.10；
longrisk UCB <= 0.05；
memory/offdiag UCB <= 0.05；
passes at least 2 datasets without dataset-specific rule。
```

## D5. 可视化

```text
fig_D1_generated_quality_pareto.svg
fig_D2_generated_future_path_curves.svg
fig_D3_generated_memory_offdiag_dashboard.svg
fig_D4_generated_vs_existing_density.svg
fig_D5_runtime_apply_cost.svg
```

---

# Part E：Controller / Runtime / Downstream Gate

## E0. Controller 只在 A/B/C/D 至少一条强证据后打开

Controller 不允许由 red feature、future outcome feature、old table diagnostic、dataset name、validation/test 指标组成。

## E1. Minimal controller 候选

允许的 controller 形态：

```text
1. Legal mechanism topK with hard veto；
2. CoreLike density harvest controller；
3. Virtual path probe controller；
4. Generated action controller if D strong pass。
```

不允许：

```text
OldRank_score official；
AUV_future_label official；
future outcome diagnostic official；
dataset-specific threshold；
post-horizon label as commit feature。
```

## E2. Controller pass 标准

```text
accepted_count >= 87；
coverage >= 0.03；
GradeAB precision >= 0.75；
AUV LCB > 0；
V20/V80/V240 LCB not negative beyond epsilon；
longrisk UCB <= 0.05；
bad UCB <= 0.05；
null UCB <= 0.15；
memory/offdiag UCB <= 0.05；
LDO/LSO/LTO/LFO drop <= 0.10；
feature cost q90 <= 0.25 ms；
no fake/proxy/cpu。
```

## E3. Runtime

只有 controller pass 才跑 selected runtime。

记录：

```text
step_ratio_q90
step_ratio_p50
payload_apply_q90
feature_compute_q90
controller_decision_q90
memory_ratio
kernel_launch_count
active_step_count
selected_action_count
```

Runtime pass：

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
no offline materializer in timed path
```

## E4. Paired replay / short-full boundary

只有 controller + runtime pass 才打开：

```text
paired replay；
leave-dataset-out；
leave-stratum-out；
leave-template-out；
short run；
full run；
sample efficiency；
noise robustness；
continual / anti-forgetting。
```

---

# 5. 并行执行安排

为了防止继续“一轮一个 blocker”，v9.8.4 必须并行。

```text
Batch A:
  P0 + A-line future path mechanism decomposition。

Batch B:
  B-line legal mechanism / virtual path probe。

Batch C:
  C-line natural AP0 materializer preflight。

Batch D:
  C-line panel B/C/D density materialization，条件是 C1d pass。

Batch E:
  D-line generated route reopen gate，条件是 A/B/C 给出证据。

Batch F:
  Base-Acc Sentinel / no-fake / contract audit。
```

---

# 6. Stop / Pivot 规则

## Stop 1：如果 C 线扩流失败

```text
materializer_entrypoint_found = 0
或 C1d preflight fail
```

则 v9.8.4 不能继续讨论 density，下一轮优先工程化 natural AP0 stream materializer。

## Stop 2：如果 B 线没有 legal mechanism

```text
B weak pass = 0
```

则不能打开 existing-action controller。OldRank 仍是 diagnostic。

## Stop 3：如果 A 线 future path 不再支持 Core77/OldOnly

```text
Core77 AUV LCB <= 0
或 Core77 longrisk UCB > 0.05
```

则 existing-action route 应停止，回到 action source redesign。

## Stop 4：如果 D 线 generated smoke 继续 value-negative / high-risk

```text
best generated AUV LCB <= 0
或 longrisk UCB > 0.50
```

则 generated route 继续停止，不再添加小变体。

---

# 7. 最终 route 决策

## R1-FuturePathMechanismFoundLegalMechanismAbsent

```text
A strong pass；B fail；C uncertain。
下一步：继续 legal mechanism / virtual path probe。
```

## R2-NaturalStreamDensitySufficient

```text
C strong pass；CoreLike density LCB >= 0.03。
下一步：existing-action harvest controller。
```

## R3-NaturalStreamDensityInsufficient

```text
C strong pass；CoreLike density LCB < 0.03。
下一步：必须生成新 update，不再只筛 existing actions。
```

## R4-LegalMechanismFound

```text
B strong pass。
下一步：minimal controller + selected runtime。
```

## R5-GeneratedRouteReopened

```text
D weak/strong pass。
下一步：generated controller + runtime。
```

## R6-AllMechanismsFail

```text
A/B/C/D 都不能解释或生成好动作。
下一步：重新定义 functional update，不再以 AP0/APG action family 为主。
```

---

# 8. 本轮最低有效推进

即使没有 system pass，v9.8.4 至少必须完成：

```text
1. A 线：明确 Core77 / OldOnly / ExactOnly / RandomMatched 的完整路径机制差异；
2. B 线：确认是否存在 legal training-time mechanism；
3. C 线：落地至少 5000 panel 的 natural AP0 labeled extension，或明确 materializer blocker；
4. D 线：若机制成立，有限重开 generated route；若机制不成立，继续停止；
5. 不把 red diagnostic、future label、old table、dataset name 写成 official controller；
6. 不打开 runtime / paired replay / short-full，除非 controller gate 真的过。
```

---

# 9. 必须落盘的 artifact

```text
p0_boundary_reproduction_v9840.csv
p1_future_path_mechanism_decomposition_v3.csv
p2_legal_training_time_mechanism_v3.csv
p3_natural_ap0_stream_preflight_v9840.csv
p4_natural_ap0_density_panel_v9840.csv
p5_virtual_path_probe_v9840.csv
p6_geometry_adaptive_generated_reopen_gate_v9840.csv
p7_generated_update_smoke_v9840.csv
p8_minimal_controller_boundary_v9840.csv
p9_selected_runtime_boundary_v9840.csv
p10_paired_replay_boundary_v9840.csv
p11_short_full_boundary_v9840.csv
fig_A1_group_future_path_V_curve.svg
fig_A2_group_future_path_risk_curve.svg
fig_B1_mechanism_topK_quality_bar.svg
fig_B4_virtual_probe_cost_quality_pareto.svg
fig_C1_density_curve_panel_size.svg
fig_D1_generated_quality_pareto.svg
base_acc_sentinel_v9840.csv
contract_audit_v9840.csv
no_fake_audit_v9840.csv
field_legality_ledger_v9840.csv
route_decision_v9840.json
run_manifest_v9840.json
```

---

# 10. 最终判断

v9.8.3 之后，项目不能继续围绕旧 rank、P4 gate 或 APG 小变体打补丁。

当前最重要的科学结论是：

$$
\boxed{
\text{好动作的价值来自未来训练轨迹，而不是当前一步 loss；但我们还没有训练当下合法机制来读懂这种轨迹。}
}
$$

v9.8.4 的任务不是让结果看起来更好，而是回答一个更高层的问题：

$$
\boxed{
\text{优化器能否自适应理解网络的函数空间几何，并据此选择或生成会改善未来训练路径的小更新？}
}
$$
