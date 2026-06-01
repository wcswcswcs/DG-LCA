# DG-KAN v9.2.59 Oracle-Decomposed Risk/Support Reset 与 Deployable Exact-Signal Controller 完整实验计划

> 本计划基于 v9.2.58 `Risk-Support Sufficient Statistics 与 Exact-Signal Accept-Region Geometry` 的真实复盘制定。  
> v9.2.58 的 terminal route 是：
>
> ```text
> route = R10-ExactReferenceStillInfeasible
> base_candidate = LQ-t2-h256
> success_v9258_strict_purekan_functional = False
> success_v9258_full_functional = False
> success_v9258_external_ready = False
> ```
>
> v9.2.58 的关键事实是：
>
> ```text
> P1 exact-reference failure autopsy:
>   pass = 1
>   false positives = 732
>   false negatives = 905
>   accepted bad-events = 560
>   FP / FN / bad-event attribution fraction = 1.0
>   FP primary = FPA5-delta_gain_mismatch
>   FN primary = FNA4-family_scarcity
>   bad-event primary = FPA6-tail_instability
>
> P2 risk sufficient statistics:
>   best = RISK5-RiskLCB
>   bad-event AUC = 0.809103
>   corr = 0.563818
>   risk_stat_pass = 1
>   utility pass = 0
>   reason = clean slice is tiny / empty
>
> P3 support sufficient statistics:
>   best = SUP2-FamilyReliabilityV2
>   legal-oracle Jaccard = 0.506509
>   diagnostic pass = 1
>   precision = 0.572193
>   bad-event = 0.364973
>   support_stat_pass = 0
>
> P4 exact reference feasibility v2:
>   best = RISK6-HybridRiskSufficientStatistic + SUP0-ReferenceSupport
>   precision = 1.0
>   bad-event = 0.0
>   coverage = 0.000992 < 0.03
>   accepted signal strata = 1
>   accepted family = 3
>   exact_reference_feasible = 0
>
> P5 true-delta compute v3:
>   best true route = TBD0-V9256CBD0Reference
>   AUC = 0.888401
>   agreement = 1.0
>   step ratio = 2.863280 > 1.50
>   true_delta_compute_pass = 0
>
> P6 oracle support:
>   precision = 1.0
>   coverage = 0.120040
>   bad-event = 0.0
>   oracle_support_pass = 1
>   measured signal strata = 3
>   balanced diagnostic rows = 36
>   support_measurement_pass = 0
>
> current blocker:
>   exact_reference_still_infeasible
>
> next_required_implementation:
>   reset_risk_support_target_decomposition
> ```
>
> v9.2.58 的核心结论不是 “risk/support completely useless”。相反，它证明了：
>
> $$
> \boxed{
> \text{bad-event 有 legal risk signature，但当前 risk gate 太保守，只能得到 clean-too-tiny slice。}
> }
> $$
>
> $$
> \boxed{
> \text{support statistic 能部分贴近 oracle，但还不能同时保 precision / coverage / bad-event。}
> }
> $$
>
> $$
> \boxed{
> \text{oracle support 仍有足够 coverage，但 legal controller 只能找到极小 clean core。}
> }
> $$
>
> 因此 v9.2.59 不能继续只调一个 threshold，也不能继续堆一个新的混合 score。  
> 本轮必须把 target 本身拆开：从单一 `safe-good` 标签，转向 **oracle-decomposed safe-good factors + bad-event mode-specific risk/support sufficient statistics**。

---

## 0. 硬约束

本计划继续遵守：

```text
no teacher
no self-teacher
no distillation
no loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
official controller 不使用 dataset_name 分支
PureKANConv / PureKANFormer 继续 deferred
```

Functional update 仍然是 update rule，不是 loss：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\Delta\theta_{\text{functional}}.
$$

任务目标保持标准 CE：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

允许按 dataset 做诊断：

```text
allowed:
  report MNIST / Fashion-MNIST / KMNIST slice failure modes
  report bad-event by dataset
  report support density by dataset
  report leave-dataset-out matrices
  report whether certain datasets concentrate failure modes
```

禁止按 dataset 调参：

```text
forbidden:
  if dataset == Fashion: use threshold A
  if dataset == KMNIST: use risk statistic B
  if dataset == MNIST: skip branch-delta
  tune thresholds per dataset
  choose support family per dataset
  choose candidate generator per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

Official controller 只能使用 train-stream / event-level / role-level / risk/support/family-level / branch-delta features。Dataset name 只能出现在 report 中，不能出现在 commit rule 中。

---

# Part I. 对 v9.2.58 的独立判断

## 1. v9.2.58 没有达到目标

v9.2.58 没有达到 strict PureKAN functional success。它失败在两个前置 gate：

```text
exact_reference_feasible = 0
true_delta_compute_pass = 0
```

因此不能声明：

```text
strict PureKAN local causal evidence
full functional success
external-ready success
Beyond-MLP success
```

P7-P10 被 gate-blocked 是合理的，因为 P4 exact reference feasibility 没恢复。即使 P5 compute 后续过线，如果 P4 仍然 infeasible，也不能进入 official paired replay。

## 2. v9.2.58 的真实进展

v9.2.58 有三项重要进展。

第一，failure autopsy 已经闭合。FP、FN、bad-event 三类 attribution fraction 都是 `1.0`。这说明 controller failure 不是黑箱失败，而是集中在：

```text
delta_gain_mismatch
family_scarcity
tail_instability
```

第二，risk signal 明确存在。`RISK5-RiskLCB` 对 bad-event 的 AUC 达到 `0.809103`，corr 达到 `0.563818`。这说明 bad-event 不是完全后验不可预测；它有 legal pre-event / train-stream signature。

第三，support statistic 有局部价值。`SUP2-FamilyReliabilityV2` 的 legal-oracle Jaccard 达到 `0.506509`，说明 support/family 方向不是空的。但它还不能同时压 bad-event 和保 precision。

## 3. v9.2.58 的真实失败

v9.2.58 的失败不是 “risk/support 没信号”，而是：

$$
\boxed{
\text{risk/support signal 只找到极小 clean core，不能恢复 deployable coverage。}
}
$$

P4 best controller 的：

$$
Precision=1.0,
$$

$$
BadEvent=0.0,
$$

但：

$$
Coverage=0.000992.
$$

这说明当前 gate 极端保守。它能把坏事件筛掉，但也几乎把所有可用事件都筛掉。可部署 controller 需要：

$$
Coverage\in[0.03,0.15].
$$

而当前只有：

$$
0.000992.
$$

差距大约是：

$$
0.03-0.000992=0.029008.
$$

也就是至少需要约 $30\times$ 的 accepted coverage expansion，同时不能让 bad-event 回升。

## 4. 当前 blocker 的本质

当前 blocker 应写成：

$$
\boxed{
\text{safe-good target 太粗；需要分解 oracle success 与 bad-event modes。}
}
$$

现在单一 `safe-good` 标签混合了至少五个子条件：

```text
value positive
control resistant
tail safe
support stable
family reliable
```

只用一个 gap score 或一个 hybrid risk score 会出现两个极端：

```text
宽 gate:
  coverage 足够，但 bad-event 爆炸。

窄 gate:
  precision / bad-event 很好，但 coverage 只有 0.000992。
```

所以 v9.2.59 必须把 safe-good 拆成可学习、可诊断、可组合的 sufficient statistics，而不是继续把所有模式塞进一个混合分数。

---

# Part II. v9.2.59 总体目标

v9.2.59 的总体目标是：

$$
\boxed{
\text{通过 oracle-decomposed factor labels 与 mode-specific risk/support statistics，把 exact branch-delta signal 转化为 deployable accept region。}
}
$$

本轮必须同时验证两个问题：

$$
\boxed{
\text{Decision geometry: exact reference 是否能用新 risk/support factors 恢复 coverage？}
}
$$

$$
\boxed{
\text{Compute: true-delta v3/v4 是否能继续接近 system envelope？}
}
$$

但优先级上：

```text
如果 exact reference feasibility 仍失败:
  继续 kernelization 不是主线。

如果 exact reference feasibility 恢复但 compute fail:
  kernelization 是主 blocker。

如果二者都过:
  打开 LDO / LSO / official paired replay。
```

---

# Part III. 核心假设

## H1：当前 failure 来自 target decomposition 缺失，而不是 exact value signal 消失

证据：

```text
exact AUC = 0.888401
exact agreement = 1.0
oracle precision = 1.0
oracle coverage = 0.120040
oracle bad-event = 0.0
```

H1 成立标准：

用 factorized target 后，CBD0 exact reference controller 达到：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

H1 失败标准：

即使用 factorized labels 与 mode-specific risk/support stats，CBD0 exact reference 仍无 feasible region。

## H2：bad-event 是多模式混合，不能用单一 risk scalar 完整处理

v9.2.58 autopsy 已经显示 bad-event primary mode 是 `tail_instability`，FP primary 是 `delta_gain_mismatch`，FN primary 是 `family_scarcity`。H2 认为这些模式需要分别建模。

H2 成立标准：

至少三个 bad-event submode 中，每个都有一个 legal risk statistic 达到：

$$
AUC_{\text{submode}}\geq0.65
$$

or:

$$
Corr_{\text{submode}}\geq0.30.
$$

组合后：

$$
BadEventRate\leq0.05
$$

while:

$$
Coverage\geq0.03.
$$

H2 失败标准：

所有 submode-specific risk statistics 都低于 diagnostic threshold，或者组合 gate 仍只能得到 tiny clean core。

## H3：oracle/legal gap 来自 support family definition 不对

`SUP2` 的 Jaccard 有 diagnostic signal，但 official pass 失败。H3 认为 family definition 仍然粗或错，尤其缺少 delta-gain、tail-stability、control-dominance buckets。

H3 成立标准：

新 support family 使 legal-oracle overlap 提升到：

$$
Jaccard_{\text{legal,oracle}}\geq0.60
$$

并且 controller 满足：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

H3 失败标准：

family refinement 只降低 coverage，不提升 precision / bad-event。

## H4：support measurement 太窄仍是实质风险

v9.2.58 仍只有：

```text
measured signal strata = 3
balanced diagnostic rows = 36
```

H4 成立标准：

本轮必须达到：

```text
natural_real_event_count >= 16000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 6
measured_family_count >= 16
```

并且所有 controller 需要报告 bootstrap CI / variance。

H4 失败标准：

support 仍停留在 3 strata / tens of balanced rows；则任何 controller pass 都不能外推。

## H5：true-delta compute 仍需并行推进，但不能掩盖 decision blocker

H5 成立标准：

至少一个 true-delta compute candidate 达到：

$$
AUC\geq0.70,
$$

$$
Agreement\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

H5 失败标准：

true-delta compute 仍：

$$
StepRatio_{q90}>1.50
$$

or signal lost。

---

# Part IV. Oracle-decomposed target definition

## 1. 原始标签

v9.2.58 的单一标签：

$$
Y_{\text{safe-good}}
$$

太粗。v9.2.59 将它分解为以下 factors。

## 2. Value-positive factor

Functional branch 必须产生正价值：

$$
Y_{\text{value}}(e)
=
\mathbb{1}
[
Gain_{\text{RealFunctional}}(e)>0
].
$$

## 3. Control-resistant factor

Functional branch 必须强于 matched controls：

$$
Y_{\text{control}}(e)
=
\mathbb{1}
[
Gain_{\text{RealFunctional}}(e)
-
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})
>
\gamma_c
].
$$

## 4. Tail-safe factor

不能放大 tail risk：

$$
Y_{\text{tail-safe}}(e)
=
\mathbb{1}
[
\Delta CEp99(e)\leq \epsilon_{ce}
]
\cdot
\mathbb{1}
[
\Delta WrongConfP95(e)\leq \epsilon_{wc}
]
\cdot
\mathbb{1}
[
\Delta MarginP10(e)\geq-\epsilon_m
].
$$

## 5. Delta-gain consistency factor

避免 large movement / low realized gain 的 mismatch：

$$
Y_{\text{consistent}}(e)
=
\mathbb{1}
[
\frac{\|\Delta z_F\|}{|Gain_F|+\epsilon}\leq \rho_z
]
\cdot
\mathbb{1}
[
\frac{\|\Delta\theta_F\|}{|Gain_F|+\epsilon}\leq \rho_\theta
].
$$

## 6. Support-stable factor

事件必须在足够支持密度与可靠 family 中：

$$
Y_{\text{support}}(e)
=
\mathbb{1}
[
Density(e)\geq d_0
]
\cdot
\mathbb{1}
[
Rel_{\text{LFO}}(family(e))\geq r_0
].
$$

## 7. Factorized safe-good

最终不再直接学习一个粗标签，而是学习：

$$
Y_{\text{factor-safe}}
=
Y_{\text{value}}
\land
Y_{\text{control}}
\land
Y_{\text{tail-safe}}
\land
Y_{\text{consistent}}
\land
Y_{\text{support}}.
$$

Controller 不必显式使用这些 posthoc factor labels at commit time；这些 labels 用于训练/校准 legal sufficient statistics，official commit rule 只用 legal features。

---

# Part V. Candidate sufficient statistics

## 1. Risk statistics

### R1：TailInstabilityRiskV2

$$
S_{\text{tail}}
=
CEp99
+
\alpha_1WrongConfP95
-
\alpha_2MarginP10
+
\alpha_3Volatility_{\text{tail}}.
$$

### R2：DeltaGainMismatchRisk

$$
S_{\text{mismatch}}
=
\frac{\|\Delta z_F\|}{|Gain_F|+\epsilon}
+
\beta
\frac{\|\Delta\theta_F\|}{|Gain_F|+\epsilon}.
$$

### R3：ControlDominanceRisk

$$
S_{\text{control-dom}}
=
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})
-
Gain_{\text{RealFunctional}}.
$$

### R4：BranchDisagreementRisk

$$
S_{\text{branch-disagree}}
=
Var_b(Gain_b)
+
Var_b(CE_b)
+
Var_b(Margin_b).
$$

### R5：FamilyRiskLCBv2

$$
LCB_{\text{safe}}(f)
=
\mu_{\text{safe}}(f)
-
\kappa\sqrt{\frac{\sigma^2_{\text{safe}}(f)}{n_f+\epsilon}}.
$$

$$
S_{\text{family-risk}}=1-LCB_{\text{safe}}(f).
$$

### R6：ModeSpecificRiskUnion

Separate gates for failure modes:

$$
RiskSafe(e)
=
\mathbb{1}[S_{\text{tail}}\leq\rho_t]
\land
\mathbb{1}[S_{\text{mismatch}}\leq\rho_m]
\land
\mathbb{1}[S_{\text{control-dom}}\leq\rho_c]
\land
\mathbb{1}[S_{\text{family-risk}}\leq\rho_f].
$$

## 2. Support statistics

### S1：DeltaTailFamilyV2

Family includes no dataset name:

$$
family(e)
=
(
stratum,
horizon,
risk\_bucket,
role\_bucket,
attach\_type,
branch\_bucket,
delta\_gain\_bucket,
tail\_stability\_bucket,
control\_gap\_bucket
).
$$

### S2：LeaveFamilyOutReliabilityV2

$$
Rel_{\text{LFO}}(f)
=
\min_{f'\in\mathcal{N}(f)}
Rel(f').
$$

### S3：KNNDensityInRiskSupportSpace

Feature vector:

```text
gap
control_gap
tail_risk
mismatch_risk
branch_disagreement
support_density
role_bucket
horizon_bucket
stratum_bucket
```

Density:

$$
Density(e)=\frac{k}{N\cdot Volume(\mathcal{N}_k(e))}.
$$

### S4：OracleOverlapDiagnosticOnly

Posthoc diagnostic only; never official:

$$
Jaccard(A_{\text{legal}},A_{\text{oracle}})
=
\frac{|A_{\text{legal}}\cap A_{\text{oracle}}|}
{|A_{\text{legal}}\cup A_{\text{oracle}}|}.
$$

## 3. Controller candidates

### C1：FactorRiskExactGapController

$$
Accept(e)
=
Gap(e)\geq\tau_g
\land
RiskSafe(e)
\land
Density(e)\geq d_0.
$$

### C2：SupportPocketControllerV2

$$
Accept(e)
=
Gap(e)\geq\tau_g
\land
RiskSafe(e)
\land
Rel_{\text{LFO}}(family(e))\geq r_0.
$$

### C3：CoreExtensionController

High-precision core:

$$
Core(e)
=
Gap(e)\geq\tau_{high}
\land
Risk(e)\leq\rho_{low}
\land
Support(e)\geq d_{high}.
$$

Coverage extension:

$$
Extension(e)
=
Gap(e)\geq\tau_{mid}
\land
Risk(e)\leq\rho_{mid}
\land
Rel_{\text{LFO}}(family(e))\geq r_0
\land
TailSafe(e).
$$

Final:

$$
Accept(e)=Core(e)\lor Extension(e).
$$

### C4：ParetoModeSafeController

Accept events on Pareto frontier:

```text
positive exact gap
low tail risk
low delta-gain mismatch
low control dominance
high support density
family reliability
multi-family balance
```

### C5：Oracle

Posthoc diagnostic only. Never official.

---

# Part VI. 实验阶段

## P0：v9.2.58 boundary reproduction

### 目标

确认 v9.2.58 boundary 稳定。

### 必须记录

```text
route
source_route_v9257
reference_failure_autopsy_pass
risk_stat_pass
risk_gate_utility_pass
support_diagnostic_pass
support_stat_pass
exact_reference_feasible
reference_precision
reference_coverage
reference_bad_event
true_delta_compute_pass
true_delta_auc
true_delta_agreement
true_delta_step_ratio
oracle_support_pass
oracle_precision
oracle_coverage
oracle_bad_event
support_measurement_pass
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R10-ExactReferenceStillInfeasible
risk_stat_pass = 1
exact_reference_feasible = 0
oracle_support_pass = 1
fake/proxy/offload = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_clean_core_tiny_coverage_ladder.svg
p0_oracle_legal_gap.svg
```

---

## P1：oracle-decomposed label construction and autopsy

### 目标

把 exact reference failure 拆成 factor labels，并验证每个 factor 是否可解释。

### 必须记录

```text
row_id
split_id
dataset
seed
horizon
signal_stratum
event_family
safe_good
bad_event
oracle_accept
controller_accept
Y_value
Y_control
Y_tail_safe
Y_consistent
Y_support
Y_factor_safe
gap_score
control_gap
tail_risk
delta_gain_mismatch
family_reliability
support_density
fp_mode
fn_mode
bad_event_mode
```

### 判断标准

P1 pass：

```text
factor_label_validity_pass = 1
bad_event_mode_attribution_fraction >= 0.90
fp_mode_attribution_fraction >= 0.90
fn_mode_attribution_fraction >= 0.90
```

### 可视化

```text
p1_factor_label_venn.svg
p1_bad_event_mode_sankey.svg
p1_factor_safe_vs_oracle_heatmap.svg
p1_fp_fn_decomposition.svg
```

---

## P2：mode-specific risk sufficient statistics

### 目标

分别建模 tail instability、delta-gain mismatch、control dominance、family risk。

### 必须记录

```text
risk_stat_id
bad_event_mode
features_used
uses_dataset_name
uses_validation
uses_test
uses_posthoc
AUC_bad_event
AUC_submode
corr_submode
precision_after_gate
coverage_after_gate
bad_event_after_gate
bad_event_delta
coverage_delta
feature_overhead
memory_overhead
```

### 判断标准

P2 pass：

At least three submodes satisfy:

$$
AUC_{\text{submode}}\geq0.65
$$

or:

$$
Corr_{\text{submode}}\geq0.30.
$$

Combined risk utility:

$$
BadEventRate\leq0.05,
$$

$$
Coverage\geq0.03.
$$

Diagnostic pass：

$$
\Delta BadEventRate\leq-0.20
$$

with:

$$
Coverage\geq0.01.
$$

### 可视化

```text
p2_risk_submode_auc_matrix.svg
p2_risk_gate_frontier.svg
p2_bad_event_reduction_by_mode.svg
p2_risk_feature_ablation.svg
```

---

## P3：support sufficient statistics reset

### 目标

修复 family_scarcity 与 support measurement too narrow。

### 必须记录

```text
support_stat_id
family_definition
uses_dataset_name
support_density_method
family_reliability_method
leave_family_out_method
measured_family_count
accepted_family_count
accepted_signal_strata_count
max_family_share
precision_after_support_gate
coverage_after_support_gate
bad_event_after_support_gate
legal_oracle_jaccard
oracle_overlap
coverage_expansion_vs_v9258
feature_overhead
```

### 判断标准

P3 official support pass：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

and:

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

Diagnostic support pass：

$$
Jaccard_{\text{legal,oracle}}\geq0.60
$$

or:

$$
\Delta BadEventRate\leq-0.20
$$

with:

$$
Coverage\geq0.03.
$$

### 可视化

```text
p3_family_reliability_v2_heatmap.svg
p3_oracle_legal_overlap_v2.svg
p3_support_density_bad_event_surface.svg
p3_family_scarcity_repair.svg
```

---

## P4：exact reference feasibility v3

### 目标

用 CBD0 exact reference + factorized risk/support statistics 判断 deployable accept region 是否恢复。

### 必须记录

```text
controller_id
risk_stat_id
support_stat_id
thresholds
calibration_split_id
heldout_split_id
precision_cal
coverage_cal
bad_event_cal
precision_heldout
coverage_heldout
bad_event_heldout
AUC_heldout
corr_heldout
accepted_strata_count
accepted_family_count
max_family_share
legal_oracle_jaccard
oracle_overlap
dataset_name_used
posthoc_used_at_commit
reference_only
```

### 判断标准

Exact reference feasibility pass：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

Support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

If no controller passes, route must be:

```text
R8-ExactReferenceStillInfeasibleAfterTargetReset
```

### 可视化

```text
p4_reference_feasibility_v3_frontier.svg
p4_risk_support_threshold_surface_v3.svg
p4_clean_core_to_deployable_region.svg
p4_accepted_region_geometry.svg
```

---

## P5：true-delta compute v4 parallel lane

### 目标

继续推进 compute，但只有 P4 feasibility 恢复时才作为 primary route。

### 必须记录

```text
custom_delta_id
layout
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
AUC_safe_good
corr_safe_grounded
agreement_exact_accept
step_ratio_q90
memory_ratio
dominant_residual_subphase
risk_support_component_time
kernel_count
sync_count
read_MB
write_MB
```

### 判断标准

Compute pass：

$$
AUC_{\text{safe-good}}\geq0.70,
$$

$$
Agreement_{\text{accept}}\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Diagnostic compute pass：

$$
StepRatio_{q90}\leq2.00
$$

with signal retained.

### 可视化

```text
p5_true_delta_v4_cost_signal_pareto.svg
p5_residual_subphase_after_risk_support_reset.svg
p5_compute_vs_decision_gate_ladder.svg
```

---

## P6：support expansion

### 目标

真实扩大 signal strata 和 family coverage，不通过复制/补造 rows。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
horizons = 20,80,240,640
signal_strata = S1-S8
carriers = A0,A3,A6
row_sources = natural, balanced_diagnostic
```

Balanced diagnostic rows 的来源必须是真实 train-stream event rows；不能重复复制自然 rows 伪装 balanced support。

### 必须记录

```text
row_source
row_id
dataset
seed
horizon
signal_stratum
event_family
custom_delta_id
safe_good
bad_event
oracle_accept
controller_accept
risk_safe
value_positive
control_resistant
tail_safe
consistent
support_stable
risk_stat_values
support_stat_values
```

### 判断标准

Measurement pass：

```text
natural_real_event_count >= 16000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 6
measured_family_count >= 16
```

Official accepted support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

### 可视化

```text
p6_signal_strata_coverage_v3.svg
p6_family_support_heatmap_v3.svg
p6_natural_vs_balanced_distribution.svg
p6_bad_event_by_stratum_family.svg
```

---

## P7：system-legal exact-signal controller

### 目标

只有 P4 reference feasibility 与 P5 compute pass 同时成立时，建立 official controller。

### 必须记录

```text
controller_id
custom_delta_id
risk_stat_id
support_stat_id
thresholds
calibration_split_id
heldout_split_id
precision_cal
coverage_cal
bad_event_cal
precision_heldout
coverage_heldout
bad_event_heldout
AUC_heldout
corr_heldout
accepted_strata_count
accepted_family_count
max_family_share
step_q90
memory_ratio
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
official_eligible
```

### 判断标准

Controller pass：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p7_system_controller_precision_coverage_bad.svg
p7_system_controller_cost_vs_value.svg
p7_system_controller_family_coverage.svg
```

---

## P8：Leave-dataset-out / leave-stratum-out

### 目标

证明 controller 不是 dataset-specific 或 single-stratum overfit。

### 设置

Leave-dataset-out：

```text
calibrate on MNIST + Fashion, evaluate KMNIST
calibrate on MNIST + KMNIST, evaluate Fashion
calibrate on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
calibrate on all but one signal stratum
evaluate held-out stratum
```

### 必须记录

```text
split_type
heldout
controller_id
custom_delta_id
risk_stat_id
support_stat_id
threshold
precision
coverage
bad_event_rate
task_safe
CEp99_delta
margin_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
dataset_name_used
shuffle_control_pass
```

### 判断标准

LDO pass：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two held-out datasets：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

LSO pass：

At least $70\%$ held-out strata task-safe and：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

### 可视化

```text
p8_leave_dataset_out_matrix.svg
p8_leave_stratum_out_matrix.svg
p8_hidden_dataset_tuning_audit.svg
p8_leaveout_failure_modes.svg
```

---

## P9：Official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
controller_id = best P8 survivor
custom_delta_id = best P8 survivor
risk_stat_id = best P8 survivor
support_stat_id = best P8 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledTrueBranchDelta, ShuffledRiskStat, ShuffledSupportStat,
           ShuffledControlGain, ShuffledCandidateGate, ShuffledBranchRatio,
           ShuffledValueScore, ShuffledSignalChannel, FunctionalChannelShuffled,
           TailMaskShuffled, RoleScoreShuffled, DatasetRouteShuffled,
           EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
controller_id
custom_delta_id
risk_stat_id
support_stat_id
dataset
seed
horizon
signal_stratum
event_family
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_random
real_beats_noop
task_safe
event_count
coverage
bad_event_rate
step_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Paired replay pass：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{each slice,Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Shuffle controls fail：

```text
ShuffledTrueBranchDelta = fail
ShuffledRiskStat = fail
ShuffledSupportStat = fail
ShuffledControlGain = fail
ShuffledCandidateGate = fail
ShuffledBranchRatio = fail
ShuffledValueScore = fail
ShuffledSignalChannel = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p9_official_paired_replay_pareto.svg
p9_macro_beat_rate.svg
p9_signal_stratum_win_matrix.svg
p9_shuffle_control_matrix.svg
p9_system_gate_distribution.svg
```

---

## P10：Short-run scout

### 目标

如果 P9 pass，验证局部 paired replay 优势能否在连续训练中保持。

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledTrueBranchDelta
```

### 必须记录

```text
candidate
dataset
seed
steps
train_loss
holdout_loss
val_acc_proxy
CEp99
margin_p10
ECE_proxy
NLL_proxy
curvature
local_lipschitz
basis_usage_entropy
functional_channel_usage_entropy
event_count
coverage
bad_event_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Task safety：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

Mechanism pass, at least one：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

or:

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9259.csv
p0_v9258_boundary_reproduction.csv
p1_oracle_decomposed_label_construction.csv
p2_mode_specific_risk_statistics.csv
p3_support_sufficient_statistics_reset.csv
p4_exact_reference_feasibility_v3.csv
p5_true_delta_compute_v4_parallel_lane.csv
p6_online_support_stratum_expansion.csv
p7_system_legal_exact_signal_controller.csv
p8_leave_dataset_and_stratum_out.csv
p9_official_paired_replay.csv
p10_short_run_functional_validation.csv
factor_label_trace_v9259.csv
risk_submode_trace_v9259.csv
support_family_trace_v9259.csv
reference_feasibility_v3_trace.csv
true_delta_compute_v4_trace.csv
accepted_region_geometry_trace_v9259.csv
support_density_trace_v9259.csv
leaveout_trace_v9259.csv
paired_replay_branch_trace_v9259.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9258_boundary_unstable
F3_dataset_tuning_detected
F4_factor_label_invalid
F5_bad_event_mode_unattributed
F6_risk_submode_not_predictive
F7_risk_union_gate_tiny_coverage
F8_support_family_no_oracle_overlap
F9_support_gate_bad_event_fail
F10_exact_reference_still_infeasible_after_target_reset
F11_true_delta_compute_still_expensive
F12_true_delta_signal_lost
F13_system_controller_precision_fail
F14_system_controller_coverage_fail
F15_system_controller_bad_event_fail
F16_support_measurement_too_narrow
F17_oracle_support_collapse
F18_leave_dataset_out_fail
F19_leave_stratum_out_fail
F20_paired_replay_control_equivalent
F21_shuffle_control_pass
F22_functional_lr_equivalent
F23_short_run_task_drop
F24_full_run_no_macro_hard_stratum_gain
F25_strong_baseline_explains_gain
F26_robustness_fail
F27_external_not_ready
F28_fake_or_proxy_violation
F29_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-FactorLabelAutopsyPass:
  oracle-decomposed factor labels and failure modes are attributed.

R2-ModeSpecificRiskPass:
  bad-event submodes have legal predictive risk statistics.

R3-SupportSufficientStatisticPass:
  support family / density statistics improve oracle/legal overlap.

R4-ExactReferenceFeasibilityRestored:
  CBD0 exact reference + factorized risk/support passes heldout precision / coverage / bad-event gate.

R5-TrueDeltaComputeV4SystemPass:
  true branch-delta v4 passes system envelope while preserving signal.

R6-SystemLegalExactSignalControllerPass:
  system-legal true-delta controller passes heldout gate.

R7-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R8-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R9-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R10-ExactReferenceStillInfeasibleAfterTargetReset:
  even factorized risk/support cannot form legal accept region.

R11-RiskSupportStatsStillTinyCleanCore:
  clean slice remains too small; coverage below 0.03.

R12-ComputePassButControllerFail:
  true-delta compute is system-legal but controller geometry fails.

R13-ControllerFeasibleButComputeFail:
  exact reference controller feasible but true-delta compute remains too expensive.

R14-SupportMeasurementStillNarrow:
  measured signal strata / family support remain insufficient.

R15-OracleSupportCollapse:
  fresh natural replay no longer has enough safe-good support.

R16-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R17-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R18-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9258_boundary_pass
dataset_tuning_detected
factor_label_autopsy_pass
fp_mode_attribution_fraction
fn_mode_attribution_fraction
bad_event_mode_attribution_fraction
best_risk_stat_id
mode_specific_risk_pass
risk_union_gate_bad_event
risk_union_gate_coverage
best_support_stat_id
support_stat_pass
legal_oracle_jaccard
accepted_family_count
accepted_signal_strata_count
max_family_share
exact_reference_feasible
reference_controller_precision
reference_controller_coverage
reference_controller_bad_event
best_true_delta_id
true_delta_compute_pass
true_delta_auc
true_delta_agreement
true_delta_step_ratio_q90
true_delta_memory_ratio
best_system_controller_id
system_legal_controller_pass
controller_precision
controller_coverage
controller_bad_event
oracle_support_pass
oracle_precision
oracle_coverage
oracle_bad_event
support_measurement_pass
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9259_strict_purekan_functional
success_v9259_full_functional
success_v9259_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 factor label construction / failure autopsy
  P2 mode-specific risk statistics
  P3 support sufficient statistics reset
  P4 exact reference feasibility v3
  P5 true delta compute v4
  P6 support expansion

Batch 2:
  P7 system-legal controller calibration
  P8 leave-dataset-out / leave-stratum-out
  P9 paired replay scout

Batch 3:
  official P9 paired replay
  P10 short-run if paired replay passes

Batch 4:
  full 10-seed / robustness / strong baseline only if P10 passes
```

Gate rule：

```text
P4 reference feasibility can pass diagnostically before compute pass.
P7 official controller cannot pass unless P4 reference feasibility and P5 compute pass.
P8/P9 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  true branch-delta legality pass
  true branch-delta predictivity pass
  true branch-delta agreement pass
  true branch-delta system pass
  system-legal controller pass
  LDO/LSO pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.58 boundary reproduced
oracle-decomposed labels constructed
failure autopsy attributed
mode-specific risk statistics measured
support sufficient statistics reset measured
exact reference feasibility v3 measured
true delta compute v4 measured
support expansion measured
no fake/proxy/offload/loss/teacher violation
```

## Decision geometry success

```text
Minimum diagnostic success
+
exact reference feasibility restored
+
risk/support stats pass
+
multi-stratum / multi-family pass
```

## Compute success

```text
Minimum diagnostic success
+
at least one true branch-delta path predicts safe-good
+
accept agreement pass
+
system overhead gate pass
```

## Legal controller success

```text
Decision geometry success
+
Compute success
+
system-legal true-delta controller heldout pass
```

## Local functional success

```text
Legal controller success
+
LDO / LSO pass
+
official paired replay beats AdamWParallel / bestLR
```

## Full functional success

```text
Local functional success
+
short/full run task-safe mechanism gain
+
macro or hard-stratum improvement
```

## Failure stop

```text
1. v9.2.58 boundary cannot be reproduced；
2. factor labels cannot be constructed legally；
3. bad-event modes remain unattributed；
4. legal risk statistics cannot reduce bad-event without tiny coverage；
5. legal support statistics cannot recover oracle/legal overlap；
6. exact reference remains infeasible after target reset；
7. true branch-delta compute remains too expensive；
8. true branch-delta compute passes system but loses signal；
9. system-legal controller cannot meet precision / coverage / bad-event；
10. online support remains too narrow；
11. fresh natural oracle support collapses；
12. leave-dataset-out fails；
13. leave-stratum-out fails；
14. paired replay remains control-equivalent；
15. shuffle controls pass, indicating overfit；
16. short-run task drops；
17. full run gives no macro / hard-stratum / geometry gain；
18. functional breaks system gate；
19. gains are explained by QuadraticFeatureMLP；
20. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：reference feasibility + compute + controller + LDO/LSO + paired replay pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation and external robustness。

## Case B：reference feasibility restored but compute fail

必须声明：

```text
decision geometry is viable, but true branch-delta system implementation remains blocker.
```

下一步继续 kernel / layout / fused risk-support compute，不调 dataset。

## Case C：compute pass but reference/controller infeasible

必须声明：

```text
value is observable and cheap, but accept/abstain geometry is not deployable.
```

下一步继续修 risk/support sufficient statistics，不走 kernel-only。

## Case D：exact reference remains infeasible after target reset

必须声明：

```text
branch-delta exact signal has high AUC but lacks sufficient legal risk/support separation even after factorization.
```

下一步应重设 safe-good / bad-event target decomposition，不继续 threshold tuning。

## Case E：support expansion fixes controller variance

可以声明：

```text
previous infeasibility was partly narrow-support artifact.
```

但仍需 LDO/LSO and paired replay。

## Case F：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

## Case G：oracle support collapses

必须声明：

```text
fresh natural replay no longer has enough safe-good support.
```

下一步回到 carrier/support mechanism。

---

# Part XII. 最终建议

v9.2.59 的一句话策略是：

$$
\boxed{
\text{不要再把 safe-good 当单一黑箱标签；先分解 oracle success 与 bad-event modes，再恢复可部署 accept region。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
TBD0 有没有 AUC；
K7d local fusion 是否能跑；
formula proxy 是否够快；
C0 threshold 是否差一点；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. exact reference controller 的 FP / FN / bad-event 是哪些 factor 失败？
2. bad-event 是否可拆成 tail instability / delta-gain mismatch / control dominance / family scarcity？
3. 每个 bad-event mode 是否有 legal online risk signature？
4. support family 是否需要加入 delta-gain / tail-stability / control-gap buckets？
5. factorized risk/support 能否把 coverage 从 0.000992 拉回 >=0.03，同时保 bad-event <=0.05？
6. true delta compute v4 能否继续压到 step <=1.50？
7. support 是否能扩展到 >=6 measured strata？
8. system-legal controller 能否 LDO/LSO？
9. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.59 的结果将给出清晰分叉：

```text
if factorized risk/support restores reference feasibility and true-delta compute passes:
  open system-legal controller, LDO/LSO, paired replay.

if reference feasible but compute fail:
  kernelization remains blocker.

if compute pass but reference infeasible:
  risk/support/accept-region geometry remains blocker.

if both fail:
  exact branch-delta remains diagnostic; reset target decomposition and output-delta statistics.

if oracle support collapses:
  carrier/support stability is blocker.
```
