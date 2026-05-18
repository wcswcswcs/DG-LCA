# DG-KAN v9.8.7 四线并行：自然动作扩流、未来路径机制、低成本训练当下代理、生成路线重开条件

> 本文件基于 v9.8.6 的真实结果制定。v9.8.7 不继续小修 `FO8`、不继续调 AUV 阈值、不在 2876 个旧动作里反复挤 10 个补丁动作，也不在没有合法机制和密度证据时重开 generated route。  
> 本轮的核心目标是把四条线的 blocker 拆成可以一次性并行验证的硬问题：  
> A 线：好动作的未来训练路径到底是什么样；  
> B 线：训练当下有没有低成本、合法、可部署的未来路径代理；  
> C 线：自然 AP0 动作池扩大以后，好动作密度到底够不够；  
> D 线：如果动作池不够，是否有条件重开几何自适应生成路线。

---

# 1. v9.8.6 的独立判断

v9.8.6 的 route 是：

```text
route = R5-LegalProxyFailNaturalStreamMissing
primary_blocker = legal_future_operator_proxy_failed
secondary_blocker = natural_stream_materializer_missing
system_legal_controller_pass = 0
generated_route_status = stopped_no_B_or_C_mechanism_evidence
```

这不是能力成功，也不是 controller 成功。controller、selected runtime、paired replay、short/full training 全部没有打开。

但是 v9.8.6 不是空转。它把 v9.8.5 的四线推进进一步压到了更明确的边界：

```text
A 线：未来路径组合有强诊断信号，但不是 official controller。
B 线：FO1-FO8 future-operator proxy 全部失败。
C 线：自然 AP0 扩流 materializer 仍未落地。
D 线：generated sandbox 仍不允许打开。
```

我对本轮的核心判断是：

$$
\boxed{
\text{未来路径现象仍然可信；失败的是训练当下低成本合法代理和自然动作密度裁决。}
}
$$

也就是说，现在不能说“未来轨迹思路失败”。A 线仍然显示 Core77 与一些组合动作有很强的未来路径收益。但我们还不能部署，因为 B 线找不到合法低成本代理，C 线还不知道自然动作池里好动作密度是否足够，D 线因此不能重开。

---

# 2. 四线分别的真实进展

## 2.1 A 线：未来训练路径组合，信号越来越实，但不能单独 controller

v9.8.6 的 P1 重新评估了多个未来路径组合。关键结果如下：

```text
Core77:
  actions = 77
  AUV LCB = 2.8296932871691163
  V80 LCB = 3.1200217020467886
  V240 LCB = 4.750885696264312
  longrisk UCB = 0.03827572901862327
  memory/offdiag UCB = 0.0
  GradeAB precision = 1.0
  raw/support LDO = 0.4415584415584416 / 0.09260529551331587

Core77 + CoreExpansion10:
  actions = 87
  AUV LCB = 2.722153601222041
  V80 LCB = 2.993949373729339
  V240 LCB = 4.588476154696239
  longrisk UCB = 0.03389313880385762
  memory/offdiag UCB = 0.0
  GradeAB precision = 0.8850574712643678
  raw/support LDO = 0.39080459770114945 / 0.13505747126436785

Core77 + RiskCleanButLowValueTop10:
  actions = 87
  AUV LCB = 3.2562463420439274
  V80 LCB = 3.632953825093594
  V240 LCB = 5.462997655266469
  longrisk UCB = 0.03389313880385762
  memory/offdiag UCB = 0.0
  GradeAB precision = 0.8850574712643678
  raw/support LDO = 0.39080459770114945 / 0.20084099015084253
```

这说明，Core77 以及 Core77 + expansion 组合并不是随机好。它们在未来路径上有持续收益，尤其 V80/V240 很强。Core77 + RiskCleanButLowValueTop10 的 AUV 甚至高于 Core77 本身。

但 A 线不能 controller，原因有三个。

第一，Core77 只有 77 个动作，低于约 87 个动作的最低 coverage 要求。

第二，Core77 + expansion 虽然数量够、未来路径强、风险低，但 raw LDO 仍然高。support-adjusted LDO 有改善，但 raw official gate 没过。

第三，AUV 本身不是充分条件。RandomMatched 的 AUV LCB 也有 `1.8284728866377598`，但 longrisk UCB 高达 `0.8485412092119363`，memory/offdiag UCB 高达 `1.9039460558312342`。所以未来路径收益必须和 longrisk、memory、offdiag、bad/null 一起判断。

A 线当前最重要的 insight 是：

$$
\boxed{
\text{好动作不是当前一步降 loss；好动作是让后续训练路径持续变好，同时不制造长期风险。}
}
$$

但 A 线还只是“事后看路径”。它还不是训练当下可用的优化器。

---

## 2.2 B 线：FO1-FO8 future-operator proxy 全部失败，不是阈值问题

v9.8.6 的 B 线测试了 FO1-FO8。最好的 `FO8_FO123HardVetoNoWeightedScore` 是：

```text
precision = 0.3563218390804598
V LCB = -0.27147985776773054
AUV LCB = 3.3271077422640407
RiskAdjustedAUV LCB = 3.3271077422640407
longrisk UCB = 0.0
memory/offdiag UCB = 0.0
cost q90 = 155.12761380523443 ms
weak/strong pass = 0 / 0
```

FO8 能把 longrisk、memory、offdiag 压住，但 precision 低，V LCB 为负，成本非常高。其他 FO proxy 也类似：有的 AUV 高，但 precision / V / longrisk / memory / cost 至少有一项严重失败。

所以 B 线的结论不是：

```text
FO8 阈值差一点。
```

而是：

$$
\boxed{
\text{当前 FO proxy 测到的是风险清理或函数响应强度，不是未来路径收益的低成本可部署代理。}
}
$$

更严重的是成本。当前 FO proxy 的 q90 大多在 `153-155 ms`，远远超过 selected runtime 需要的毫秒级预算。即便 precision 过了，这种成本也不能成为 MLP-comparable online optimizer。

因此，B 线下一步不能继续 FO9、FO10、FO11 小修。必须先回答：

```text
到底什么训练当下量能以低成本近似未来路径收益？
```

如果这个量不存在，existing-action route 就只能继续作为 diagnostic，不能变成 optimizer。

---

## 2.3 C 线：自然 AP0 扩流仍然没落地，这是目前最拖后腿的工程 blocker

v9.8.6 的自然动作密度仍停留在既有 2876 个 labeled actions：

```text
Panel-2876:
  labeled = 2876
  CoreLike rate = 0.026773296244784424
  Wilson LCB/UCB = 0.021475240123524635 / 0.033333885489495195

Panel-5000:
  expected rows = 150000
  actual rows = 0
  reason = P3_preflight_not_passed_materializer_missing

Panel-10000:
  expected rows = 300000
  actual rows = 0

Panel-20000:
  expected rows = 600000
  actual rows = 0
```

现在 CoreLike rate 的置信区间刚好跨过 $0.03$，所以我们无法判断自然动作源到底够不够：

```text
如果真实密度 >= 0.03：
  existing-action harvesting 可能可以继续。

如果真实密度 < 0.03：
  当前 AP0 动作源本身不够，需要新的生成原则。
```

这就是为什么 C 线必须优先落地。没有 C 线，我们会一直在 2876 个旧动作里重复争论“差 10 个动作怎么办”。

C 线的本质问题是：

$$
\boxed{
\text{我们还不知道好动作是稀有但可通过扩大动作池获得，还是当前动作源本身密度不足。}
}
$$

---

## 2.4 D 线：generated route 继续停止是正确的

v9.8.6 的 D 线结果是：

```text
generated_sandbox_allowed = 0
generated_route_status = stopped_no_B_or_C_mechanism_evidence
controller/runtime/paired replay/short-full = not_run
```

我认为这个判断正确。因为 B 线没有合法代理，C 线没有动作密度证据。此时如果重开 generated route，只会回到之前多轮 APG/APGU/APGX 的老问题：

```text
payload 合法；
branch-horizon 能跑；
action apply 数值闭合；
但 outcome value-negative / high-longrisk。
```

D 线下一步必须有严格 reopen gate。只有在下列至少一项成立时，才允许生成动作 sandbox：

```text
1. B 线发现低成本合法 proxy，且能解释 Core77 / OldOnly / CoreExpansion 的未来路径；
2. C 线证明自然动作密度不足，必须换动作源；
3. A 线发现清晰的未来路径机制，可以写成生成目标。
```

否则 generated route 继续停止。

---

# 3. 为什么还是感觉非常慢

你的感觉是对的。从系统能力角度看，v9.8.6 仍然没有进入：

```text
selected controller
-> selected runtime
-> official paired replay
-> short/full training
```

所以它没有带来“模型变强”的直观进展。

但从科学推进看，v9.8.6 又排除了一个假希望：

```text
FO1-FO8 这些看起来更接近未来轨迹的 proxy，仍然不能低成本、合法、稳定地选出好动作。
```

这意味着我们不能把“未来路径”思路简化成一个手工 future-operator score。现在真正需要的是两件事：

```text
1. 把自然动作扩流 materializer 工程化，回答密度问题；
2. 把未来路径机制从事后现象压缩成训练当下低成本可用的算子。
```

如果继续只修 FO8 或继续调 AUV threshold，项目会越来越像人工规则堆叠。

---

# 4. 现在最深的本质问题

当前最深的问题不是：

```text
有没有好动作；
有没有未来路径信号；
KAN base 是否能训练；
outcome table 是否可信；
branch-horizon replay 是否能跑。
```

这些大多已经有答案。

真正问题是：

$$
\boxed{
\text{如何让优化器在训练当下识别或生成能改善未来训练路径的更新，而不是事后 replay 才知道？}
}
$$

从数学上说，我们要找的不是当前一步的最大下降方向：

$$
-g_t^\top \Delta
$$

而是某种未来训练算子的正方向：

$$
\Delta \mapsto V_{1:240}(\Delta)
$$

其中 $V_{1:240}$ 不是一个单一 loss，而是未来训练路径上的多指标响应：

$$
V_{1:240}(\Delta)
=
\left(
V_1,V_5,V_{20},V_{80},V_{240},
LongRisk,MemoryRisk,OffdiagRisk,Bad,Null,Cost
\right).
$$

v9.8.6 告诉我们：

```text
A 线看得到 V_{1:240} 的好路径；
B 线还不能用训练当下字段近似它；
C 线不知道自然动作池够不够；
D 线没有生成目标。
```

这就是当前主矛盾。

---

# 5. v9.8.7 总体策略

v9.8.7 不应该继续小修。它应该做一次强制路线裁决：

```text
如果 C 线可以落地，优先用自然动作扩流判断 density；
如果 C 线仍不能落地，必须停止继续运行上层 route runner，先完成 materializer 工程；
如果 B 线找不到低成本合法 proxy，则不允许 controller；
如果 A 线的未来路径机制能被更清楚拆解，则生成路线才有重新打开的依据。
```

v9.8.7 的一句话目标是：

$$
\boxed{
\text{用自然动作扩流裁决密度，用未来路径机制裁决目标，用低成本合法代理裁决 controller，用严格 gate 裁决是否重开 generated route。}
}
$$

---

# 6. v9.8.7 详细实验计划

## P0：边界复现与 hard stop 检查

### 目标

复现 v9.8.6 边界，确认本轮没有把上轮 diagnostic 当成 official。

### 假设

v9.8.6 的真实边界是：

```text
future-path combo diagnostic 有信号；
FO1-FO8 proxy failed；
natural stream materializer missing；
generated route stopped；
controller not_run。
```

### 必须记录

```text
source_route_v9860
P1_A_combo_strong_pass
P2_weak_pass
P2_strong_pass
P3_materializer_entrypoint_found
P4_generated_sandbox_allowed
P5_controller_pass
system_legal_controller_pass
fake_data_used
proxy_row_used
cpu_offload_used
```

### 判断标准

必须全部满足：

```text
P0_boundary_pass = 1
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
controller_not_promoted = 1
generated_not_promoted = 1
```

如果 P0 不过，整轮停止，不进入 P1-P9。

### 可视化

```text
fig_p0_v9860_boundary_reproduction_bar.svg
fig_p0_gate_status_matrix.svg
```

---

## P1：自然 AP0 动作扩流 materializer 落地

### 目标

这是 v9.8.7 的最高优先级。不要再只写 `materializer missing`。本阶段必须真实实现 natural labeled AP0 stream extension materializer，并跑出 5000 panel 的完整 label。

### 核心假设

$$
H_C:
\text{如果自然 AP0 action stream 扩大，CoreLike / PathGood 动作密度可以被可靠估计。}
$$

这里不是要求立即 full success，而是要求 materializer 存在、可审计、可恢复、可扩展。

### 执行方式

P1 分四级执行，不允许直接跑 20000 panel：

```text
P1a: 1-action full branch-horizon labeled preflight
P1b: 16-action labeled preflight
P1c: 256-action labeled smoke
P1d: 5000-action panel
P1e: 条件性 10000 / 20000 panel
```

每个 action 必须完整跑：

```text
branches = RealFunctional, AdamWParallel, AdamWOnly, bestLR, NoOp, Random
horizons = 1, 5, 20, 80, 240
```

如果为了成本先裁剪，必须明确写成 diagnostic，不能 official density。

### 必须记录

```text
panel_id
target_action_count
new_action_count
existing_action_count
labeled_action_count
branch_count
horizon_count
expected_rows
actual_rows
completion_rate
rows_per_sec
wallclock_sec
unresolved_exception_count
exception_type_counts
payload_hash_missing_count
state_hash_missing_count
label_hash_missing_count
duplicate_row_count
label_exclusivity_violation_count
metric_nan_count
metric_inf_count
fake_row_count
proxy_row_count
cpu_offload_flag
```

每个 action 还要记录来源：

```text
action_id
event_id
step
seed
dataset_for_diagnostic_only
family
stratum
template_id
candidate_origin
action_norm
payload_norm
AdamW_cosine
memory_bucket
offdiag_bucket
```

注意：dataset 只用于诊断和 leaveout，不允许进入 selector/controller。

### 判断标准

P1c smoke pass：

```text
labeled_action_count >= 256
completion_rate = 1.0
label_exclusivity_violation_count = 0
unresolved_exception_count = 0
fake_row_count = 0
proxy_row_count = 0
```

P1d 5000 panel weak pass：

```text
labeled_action_count >= 5000
actual_rows = expected_rows
quality_audit_pass = 1
rows_per_sec >= 10
```

P1e 10000/20000 panel strong pass：

```text
labeled_action_count >= 10000
quality_audit_pass = 1
resumable_materializer_pass = 1
```

如果 P1a/P1b 不过，不能继续做上层 route；必须停在 materializer implementation。

### 可视化

```text
fig_p1_materializer_completion_by_stage.svg
fig_p1_rows_per_sec_by_panel.svg
fig_p1_exception_type_bar.svg
fig_p1_action_source_distribution.svg
fig_p1_quality_audit_heatmap.svg
```

---

## P2：自然动作密度曲线与 density 裁决

### 目标

判断当前 AP0 自然动作源是否足够产生可部署数量的好动作。

### 核心假设

$$
H_{density}:
P(\text{CoreLike or PathGood}) \ge 0.03
$$

如果这个假设成立，existing-action harvesting 仍有希望。  
如果不成立，继续从现有 AP0 源里筛动作没有意义，必须换动作生成方式。

### 标签定义

不能只用一个 GradeAB。至少并行计算：

```text
CoreLike:
  GradeAB = 1
  V_integrated LCB > 0
  LongRisk UCB <= 0.05
  Bad UCB <= 0.05
  Null UCB <= 0.15
  Memory/offdiag UCB <= 0.05

PathGood:
  AUV LCB > 0
  V80 LCB >= 0
  V240 LCB > 0
  LongRisk UCB <= 0.05
  Memory/offdiag UCB <= 0.05

RiskCleanValuePositive:
  V LCB > 0
  LongRisk/Bad/Null/Memory/Offdiag all pass

SlowBurnGood:
  V1 <= 0 allowed
  AUV LCB > 0
  V80/V240 positive
  risk clean
```

SlowBurnGood 是为了捕捉 OldOnly / CoreExpansion10 这类 immediate 不强但未来路径好的动作。

### 必须记录

```text
panel_size
CoreLike_count
CoreLike_rate
CoreLike_Wilson_LCB
CoreLike_Wilson_UCB
PathGood_count
PathGood_rate
PathGood_Wilson_LCB
PathGood_Wilson_UCB
SlowBurnGood_count
SlowBurnGood_rate
RiskCleanValuePositive_count
per_dataset_rate_diagnostic
per_family_rate
per_template_rate
per_step_bucket_rate
min_group_rate
max_group_rate
rate_PSI_between_panels
```

### 判断标准

Density sufficient：

$$
LCB(P(\text{PathGood})) \ge 0.03
$$

并且：

```text
min per-dataset diagnostic rate not catastrophically below global rate
positive family/template diversity pass
```

Density insufficient：

$$
UCB(P(\text{PathGood})) < 0.03
$$

在至少 10000 或 20000 labeled actions 上成立。

Density inconclusive：

```text
LCB < 0.03 < UCB
```

如果 P2 density sufficient，则进入 P5 existing-action controller。  
如果 P2 density insufficient，则 D 线 generated route 可以被重新考虑，但必须基于 A/B 的机制。  
如果 P2 inconclusive，则继续扩流，不允许改 official gate。

### 可视化

```text
fig_p2_density_curve_corelike.svg
fig_p2_density_curve_pathgood.svg
fig_p2_wilson_ci_by_panel.svg
fig_p2_rate_by_dataset_diagnostic.svg
fig_p2_rate_by_family_template.svg
fig_p2_density_pareto_corelike_pathgood.svg
```

---

## P3：未来路径机制分解，不再只看 AUV

### 目标

把 A 线从 “AUV 高” 变成可解释的未来路径机制。重点是判断好动作是否是 delayed improvement、risk-clean future operator、memory-safe trajectory correction，还是只是 label artifact。

### 核心假设

$$
H_A:
\text{Core77 / SlowBurnGood 的优势来自未来训练路径改善，而不是 immediate loss response。}
$$

### 对照组

必须至少包括：

```text
Core77
CoreExpansion10
Core77+CoreExpansion10
RiskCleanButLowValue
Core77+RiskCleanButLowValueTop10
OldOnly
ExactOnly
RandomMatched
PathGood from new natural panels
SlowBurnGood from new natural panels
```

### 必须记录

每组每个 horizon：

```text
V1_LCB
V5_LCB
V20_LCB
V80_LCB
V240_LCB
AUV_LCB
RiskAdjustedAUV_LCB
LongRisk_UCB
Bad_UCB
Null_UCB
Memory_UCB
Offdiag_UCB
CEp99_delta_UCB
margin_tail_delta_LCB
old_family_loss_delta_UCB
hard_tail_loss_delta_UCB
```

还要记录路径形态：

```text
immediate_gain = V1_LCB
mid_gain = V20/V80
long_gain = V240
slow_burn_score = AUV_LCB - max(0, V1_LCB)
risk_clean_path = longrisk/memory/offdiag all pass
monotone_value_path = V1 <= V5 <= V20 <= V80 <= V240 approximately
late_recovery_path = V1 < 0 and V80/V240 > 0
```

### 判断标准

A-line weak pass：

```text
Core77 or PathGood group has:
  AUV_LCB > 0
  V80_LCB > 0
  V240_LCB > 0
  LongRisk_UCB <= 0.05
  Memory/Offdiag_UCB <= 0.05
```

A-line strong mechanism pass：

```text
1. OldOnly or SlowBurnGood shows V1 <= 0 but AUV/V80/V240 > 0;
2. ExactOnly has worse V240 or worse risk than OldOnly;
3. RandomMatched has high risk even if AUV positive;
4. same pattern holds across dataset/family/template diagnostics;
5. at least one legal proxy family in P4 correlates with the path type.
```

如果 A-line strong fails but weak passes, A 线仍然只是 diagnostic。

### 可视化

```text
fig_p3_future_value_curve_by_group.svg
fig_p3_risk_curve_by_group.svg
fig_p3_slow_burn_scatter_V1_vs_AUV.svg
fig_p3_AUV_vs_longrisk_pareto.svg
fig_p3_core77_vs_random_path_grid.svg
fig_p3_path_type_sankey.svg
```

---

## P4：B 线重写，不再继续 FO1-FO8 小修

### 目标

找训练当下合法、低成本、能近似未来路径的 proxy。v9.8.6 已经证明 FO1-FO8 不行，所以 P4 不再新增同类复杂 weighted score。

### 核心假设

$$
H_B:
\text{未来路径可以由少量低成本训练当下算子近似，而不是由 155ms 级 virtual probe 近似。}
$$

### 新 proxy 原则

B 线只允许三类 proxy：

```text
1. Cheap memory / hard-tail response:
   用当前 train-memory / hard-tail 小样本，计算动作线性响应。

2. Low-rank future operator sketch:
   用最近若干 AdamW 更新方向构造低秩子空间，估计动作是否会被未来训练动力学放大或抵消。

3. Signal-channel agreement:
   检查多个 mini-subsets 是否同向支持该动作，不再只看当前 batch CE。
```

不能使用：

```text
future outcome
AUV label
GradeAB label
old table rank
V_integrated
risk_score
dataset name
validation/test metric
```

### 必须记录

```text
proxy_id
feature_legality = green/yellow/red
compute_cost_ms_p50/p90/p99
memory_overhead_mb
TopK87_precision
TopK87_V_LCB
TopK87_AUV_LCB
TopK87_LongRisk_UCB
TopK87_Bad_UCB
TopK87_Null_UCB
TopK87_Memory_UCB
TopK87_Offdiag_UCB
LDO_drop
LSO_drop
LTO_drop
LFO_drop
correlation_with_Core77
correlation_with_SlowBurnGood
correlation_with_RiskCleanButLowValue
```

### 判断标准

B weak pass：

```text
cost_q90 <= 1.5 ms
precision >= 0.70
V_LCB > 0
LongRisk_UCB <= 0.05
Memory/Offdiag_UCB <= 0.05
LDO/LSO/LTO <= 0.15
```

B strong pass：

```text
cost_q90 <= 1.0 ms
precision >= 0.75
V_LCB > 0
AUV_LCB > 0
risk/bad/null/memory/offdiag all pass
LDO/LSO/LTO/LFO <= 0.10
```

If B strong pass, proceed to P6 controller.  
If B weak but not strong, use only as diagnostic or as veto.  
If B fails again, stop B-score proliferation and require C density or D generator.

### 可视化

```text
fig_p4_proxy_precision_vs_cost.svg
fig_p4_proxy_V_vs_longrisk.svg
fig_p4_proxy_cost_breakdown.svg
fig_p4_proxy_score_vs_AUV_scatter.svg
fig_p4_proxy_score_vs_slow_burn.svg
fig_p4_proxy_leaveout_drop.svg
```

---

## P5：existing-action minimal controller，只允许在 A/B/C 条件满足后运行

### 目标

把已有 AP0 actions 形成一个可以冻结的 accepted region。不是为了继续调阈值，而是判断 existing-action route 是否真的可以成为 system controller。

### 运行条件

P5 只有在以下任一条件成立时运行：

```text
Condition 1:
  C line density sufficient.

Condition 2:
  B line weak/strong pass.

Condition 3:
  A line strong mechanism pass + C line panel >= 5000 complete.
```

否则 P5 必须 not_run。

### Controller 候选

只允许三类：

```text
C1: CoreLike density-harvesting controller
C2: PathGood / SlowBurnGood future-path controller
C3: B-line legal proxy + hard safety veto controller
```

禁止：

```text
OldRank direct controller
AUV label controller
future outcome controller
dataset-specific controller
FO1-FO8 high-cost controller
```

### 必须记录

```text
controller_id
accepted_count
coverage
precision_GradeAB
PathGood_precision
V_LCB
AUV_LCB
LongRisk_UCB
Bad_UCB
Null_UCB
Memory_UCB
Offdiag_UCB
LDO_drop
LSO_drop
LTO_drop
LFO_drop
candidate_template_count
max_template_share
feature_cost_q90_ms
payload_apply_cost_q90_ms
estimated_step_ratio_q90
```

### 判断标准

Controller weak pass：

```text
accepted_count >= 87
precision_GradeAB >= 0.70
V_LCB > 0
AUV_LCB > 0
LongRisk_UCB <= 0.05
Bad_UCB <= 0.05
Null_UCB <= 0.15
Memory/Offdiag_UCB <= 0.05
LDO/LSO/LTO <= 0.15
cost_q90 <= 1.5 ms
```

Controller strong pass：

```text
accepted_count >= 87
precision_GradeAB >= 0.75
PathGood_precision >= 0.75
V_LCB > 0
AUV_LCB > 0
all risk gates pass
LDO/LSO/LTO/LFO <= 0.10
cost_q90 <= 1.0 ms
```

Only if P5 strong pass can P7 runtime proceed.

### 可视化

```text
fig_p5_controller_pareto_precision_V_risk.svg
fig_p5_controller_leaveout_heatmap.svg
fig_p5_controller_cost_breakdown.svg
fig_p5_accepted_action_distribution.svg
```

---

## P6：Generated route reopen gate

### 目标

决定是否重开生成动作路线。v9.8.6 不允许生成，v9.8.7 也不能盲目生成。

### Reopen 条件

D 线只有在以下条件之一成立时允许 64-action sandbox：

```text
D1: C 线证明 natural AP0 CoreLike/PathGood density insufficient:
    UCB(PathGood rate) < 0.03 on >=10000 actions.

D2: B 线发现 legal low-cost proxy，且 proxy 可以写成生成目标。

D3: A 线发现清晰 future path mechanism，且该机制不是 outcome-only。
```

否则：

```text
generated_sandbox_allowed = 0
```

### 如果允许 sandbox

只跑 64 actions，不允许 512 起步。必须记录：

```text
generated_action_count
payload_hash_missing
certificate_hash_missing
action_apply_linf_max
branch_horizon_expected_rows
branch_horizon_actual_rows
PathGood_precision
V_LCB
AUV_LCB
LongRisk_UCB
Memory/Offdiag_UCB
new_positive_created_rate
longrisk_created_rate
runtime_cost_q90
```

### 判断标准

Generated sandbox weak pass：

```text
generated_action_count >= 64
PathGood_precision >= 0.25
V_LCB > 0
LongRisk_UCB <= 0.10
longrisk_created_rate <= 0.10
```

Generated sandbox strong pass：

```text
PathGood_precision >= 0.50
V_LCB > 0
AUV_LCB > 0
LongRisk_UCB <= 0.05
Memory/Offdiag_UCB <= 0.05
new_positive_created_rate >= 0.10
```

If generated weak fails, do not expand to 512.

### 可视化

```text
fig_p6_generated_gate_decision.svg
fig_p6_generated_outcome_pareto.svg
fig_p6_generated_damage_modes.svg
fig_p6_generated_vs_existing_path_curves.svg
```

---

## P7：selected runtime，只在 controller strong pass 后运行

### 目标

验证 selected controller 的 online runtime 是否可以与 MLP-comparable envelope 相容。

### 运行条件

```text
P5_controller_strong_pass = 1
```

或者：

```text
P6_generated_strong_pass = 1
```

否则 P7 not_run。

### 必须记录

```text
selected_controller_id
online_step_count
active_step_count
accepted_action_count
feature_compute_ms_q50/q90/q99
proxy_compute_ms_q50/q90/q99
payload_apply_ms_q50/q90/q99
controller_total_ms_q90
base_train_step_ms_q90
step_ratio_q90
peak_memory_ratio
kernel_launch_count
sync_count
materializer_in_timed_path
```

### 判断标准

```text
step_ratio_q90 <= 1.50
peak_memory_ratio <= 1.05
materializer_in_timed_path = 0
fake/proxy/cpu = 0
```

Strong runtime：

```text
step_ratio_q90 <= 1.20
peak_memory_ratio < 1.00 or <= 1.02
```

### 可视化

```text
fig_p7_runtime_step_ratio_distribution.svg
fig_p7_runtime_cost_stack.svg
fig_p7_runtime_active_step_timeline.svg
```

---

## P8：official paired replay boundary

### 目标

只有 selected controller + runtime 通过，才打开 official paired replay。

### 运行条件

```text
P5/P6 controller strong pass = 1
P7 runtime pass = 1
```

### 必须记录

```text
RealFunctional trajectory
AdamWParallel trajectory
AdamWOnly trajectory
bestLR trajectory
NoOp trajectory
RandomPayload trajectory
ShuffledPayload trajectory
horizons = 20, 80, 240
V_ctrl
AUV
accuracy_delta
CE_delta
NLL_delta
margin_delta
ECE_delta
CEp99_delta
longrisk
bad/null
memory/offdiag
```

### 判断标准

```text
RealFunctional beats AdamWParallel / bestLR / NoOp / Random on V and AUV
ShuffledPayload fails
longrisk/bad/null gates pass
leaveout gates pass
```

---

# 7. v9.8.7 route decision logic

```text
Case A: C density sufficient + B legal proxy weak/strong pass
  -> Build existing-action controller.

Case B: C density sufficient but B legal proxy fails
  -> Continue mechanism discovery; do not controller.

Case C: C density insufficient + A mechanism strong
  -> Allow small generated sandbox.

Case D: C density inconclusive
  -> Expand natural stream; do not alter official gate.

Case E: P1 materializer still missing
  -> Stop all high-level route work; prioritize materializer engineering only.

Case F: B proxy remains high-cost / low-quality
  -> Stop FO score proliferation; use B only for diagnostics.
```

---

# 8. 本轮必须产出的 artifact

```text
p0_boundary_reproduction_v9870.csv
p1_natural_materializer_preflight_v9870.csv
p1_natural_stream_panel_5000_v9870.csv
p1_natural_stream_panel_10000_v9870.csv
p1_natural_stream_panel_20000_v9870.csv
p2_density_curve_v9870.csv
p2_pathgood_label_audit_v9870.csv
p3_future_path_mechanism_v9870.csv
p4_low_cost_proxy_v9870.csv
p5_controller_candidates_v9870.csv
p6_generated_reopen_decision_v9870.csv
p7_runtime_boundary_v9870.csv
p8_paired_replay_boundary_v9870.csv
field_legality_ledger_v9870.csv
no_fake_audit_v9870.csv
failure_taxonomy_v9870.csv
route_decision_v9870.json
run_manifest_v9870.json
```

---

# 9. 最终判断

v9.8.6 之后，最不应该做的是继续围绕 FO8、AUV 阈值、CoreExpansion10 或 APG 变体小修。现在已经很清楚：

```text
未来路径现象是真的；
合法训练当下代理失败；
自然动作扩流未落地；
generated route 没有重开条件。
```

所以 v9.8.7 的优先级必须是：

```text
1. C 线 materializer 落地；
2. A 线未来路径机制细化；
3. B 线低成本 proxy 重写；
4. D 线严格 gate 后才重开。
```

最终一句话：

$$
\boxed{
\text{我们已经看见好训练路径，但还没有训练当下可部署的几何自适应优化器；下一步必须先裁决动作密度与低成本机制。}
}
$$
