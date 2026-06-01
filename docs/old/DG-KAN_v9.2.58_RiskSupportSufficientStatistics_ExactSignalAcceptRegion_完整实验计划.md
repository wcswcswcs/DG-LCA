# DG-KAN v9.2.58 Risk-Support Sufficient Statistics 与 Exact-Signal Accept-Region Geometry 完整实验计划

> 本计划基于 v9.2.57 `True Branch-Delta Compute Closure 与 Exact-Signal Controller Feasibility` 的真实复盘制定。  
> v9.2.57 的 terminal route 是：
>
> ```text
> route = R11-ExactSignalControllerInfeasible
> base_candidate = LQ-t2-h256
> success_v9257_strict_purekan_functional = False
> success_v9257_full_functional = False
> success_v9257_external_ready = False
> ```
>
> v9.2.57 的关键事实是：
>
> ```text
> P0:
>   v9.2.56 boundary reproduced
>   source route = R12-TrueDeltaPredictiveButTooExpensive
>   true branch-delta interface pass = 1
>   K7d fusion pass = 1
>   fake/proxy/offload = 0
>
> P1:
>   true branch-delta residual attribution pass = 1
>   dominant residual subphase = R9-risk_support_component_compute
>   ratio = 0.568152
>   unknown_fraction = 0.0
>
> P2:
>   true branch-delta correctness / legality pass = 1
>   best legal correctness candidate = TBD1-LayoutRepairedFullLogitSmallC
>   uses_true_delta = 1
>   source_gap = 0
>   formula_proxy = 0
>
> P3:
>   true branch-delta exact signal still strong
>   best = TBD0-V9256CBD0Reference
>   AUC = 0.888401
>   agreement = 1.0
>   step ratio = 2.863280 > 1.50
>   system pass = 0
>
> P4:
>   formula / source-gap negative controls rejected
>   PX2 cheap and agreement = 1.0 but uses source-measured gap input
>   official pass = 0
>
> P5:
>   exact reference controller feasibility failed
>   best legal reference controller = C0-AllPassExactReferenceController
>   precision = 0.430279
>   coverage = 0.041501
>   bad-event = 0.418327
>
> P6:
>   system-legal controller not_run
>   reason = P3_true_delta_system_failed
>
> P7:
>   oracle support pass = 1
>   oracle precision = 1.0
>   oracle coverage = 0.120040
>   oracle bad-event = 0.0
>   measured signal strata = 3
>   balanced diagnostic rows = 36
>   support measurement pass = 0
>
> P8-P10:
>   not_run
>   reason = P5_reference_controller_infeasible
>
> current blocker:
>   exact_reference_controller_infeasible
> next_required_implementation:
>   redesign_risk_support_sufficient_statistics
> ```
>
> v9.2.58 的核心判断是：
>
> $$
> \boxed{
> \text{现在不能只继续 kernel 化，也不能只调 threshold。}
> }
> $$
>
> v9.2.57 已经证明了三件事：
>
> $$
> \boxed{
> \text{true branch-delta signal 仍强。}
> }
> $$
>
> $$
> \boxed{
> \text{oracle safe-good support 仍存在。}
> }
> $$
>
> $$
> \boxed{
> \text{但 exact reference signal 本身还不能形成 legal accept/abstain controller。}
> }
> $$
>
> 因此 v9.2.58 的主问题不是：
>
> ```text
> branch-delta 有没有 AUC；
> K7d fusion 能不能局部加速；
> source gap proxy 能不能跑快；
> C0/C6 threshold 差一点；
> 哪个 dataset 单独更好。
> ```
>
> 而是：
>
> $$
> \boxed{
> \text{能否设计 legal、online、dataset-agnostic 的 risk/support sufficient statistics，把 strong exact value signal 转成 safe deployable accept region？}
> }
> $$
>
> 如果 exact reference + redesigned risk/support statistics 仍不能得到 safe accept region，那么继续优化 true-delta kernel 只会得到一个更快但不可部署的 diagnostic signal。  
> 如果 exact reference feasibility 恢复，而 true-delta compute 仍太贵，那么 kernelization 才是唯一 blocker。  
> v9.2.58 必须把这两个分支同时验证清楚。

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

数据集只能作为诊断切片，不能作为 official controller 条件：

```text
allowed:
  report MNIST / Fashion-MNIST / KMNIST slice metrics
  report per-dataset bad-event mode
  report per-dataset support distribution
  report leave-dataset-out

forbidden:
  if dataset == Fashion: threshold = A
  if dataset == KMNIST: use risk statistic B
  if dataset == MNIST: skip branch-delta
  tune threshold separately per dataset
  choose candidate generator separately per dataset
  choose selected/full-logit mode separately per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

Official success 必须满足：

```text
base robust pass
attach equivalence pass
no-event preservation pass
carrier active
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
true branch-delta predictivity pass
true branch-delta agreement pass
true branch-delta system pass
risk/support controller heldout pass
LDO / LSO pass
official paired replay pass
```

但是 v9.2.58 允许在 diagnostic lane 中先使用 `TBD0 exact reference` 做 feasibility audit，因为这不是 official functional success，而是判断 controller geometry 是否值得继续 kernelization。

---

# Part I. 对 v9.2.57 的独立判断

## 1. v9.2.57 没有达到目标

v9.2.57 没有 strict PureKAN functional success。它失败在两个前置 gate：

```text
true_delta_system_pass = 0
reference_controller_feasible = 0
```

这意味着不能声明：

```text
strict PureKAN local causal evidence
full functional success
external-ready success
Beyond-MLP success
```

尤其要注意：v9.2.57 的 route 不是单纯 `TrueDeltaPredictiveButTooExpensive`，而是 `ExactSignalControllerInfeasible`。这说明当前最危险的误判是继续把全部注意力放在 kernel speed 上，而忽略 exact signal 本身无法形成 safe accept region 的事实。

## 2. v9.2.57 的真实进展

v9.2.57 至少推进了三件事。

第一，true delta residual attribution 闭合。P1 unknown fraction 为 `0.0`，dominant residual subphase 是：

```text
R9-risk_support_component_compute
ratio = 0.568152
```

这说明 residual cost 已经不是不可解释的系统噪声，而是风险/支持组件计算与若干 branch-forward residual 的组合。

第二，true branch-delta correctness / legality 继续成立。P2 best legal correctness candidate `TBD1-LayoutRepairedFullLogitSmallC` 明确满足：

```text
uses_true_delta = 1
source_gap = 0
formula_proxy = 0
```

因此我们不再停留在 v9.2.55 的 “true interface missing” 阶段。

第三，exact branch-delta signal 继续强：

$$
AUC_{\text{exact-delta}}=0.888401,
$$

$$
Agreement_{\text{accept}}=1.0.
$$

同时 oracle support 仍通过：

$$
OraclePrecision=1.0,
$$

$$
OracleCoverage=0.120040,
$$

$$
OracleBadEvent=0.0.
$$

这说明 safe-good events 存在，exact signal 也有强排序能力。

## 3. v9.2.57 的真实失败

v9.2.57 的核心失败是：

$$
\boxed{
\text{strong ranking signal does not imply deployable accept region。}
}
$$

P5 best legal reference controller 是：

```text
C0-AllPassExactReferenceController
precision = 0.430279
coverage = 0.041501
bad-event = 0.418327
```

它的 coverage 进入 `[0.03,0.15]`，但 precision 远低于 `0.75`，bad-event 远高于 `0.05`。这说明当前 exact gap / value score 在 accepted region 中混入了大量 high-risk / bad-event 样本。

这不是简单 threshold 问题。原因是如果只调高 threshold 来提升 precision，coverage 很可能低于 `0.03`；如果放宽 threshold 保住 coverage，bad-event 爆炸。过去多轮 controller 都反复出现这个结构性矛盾。

## 4. 当前问题的本质

当前 blocker 应改写为：

$$
\boxed{
\text{true branch-delta value signal 存在，但缺少 legal risk/support sufficient statistics 来定义可部署 accept region。}
}
$$

现在有四个同时成立的事实：

```text
1. exact delta signal has high AUC；
2. oracle can find safe-good support；
3. legal exact controller cannot avoid bad-events；
4. support measurement remains narrow。
```

这意味着我们缺的不是“更多 value signal”，而是：

```text
1. bad-event 的 legal pre-event / commit-time signature；
2. support density / family reliability 的正确统计；
3. risk-safe 与 control-gap-positive 的可分解 gate；
4. accepted-region geometry 的稳定性；
5. multi-stratum / multi-family coverage。
```

换句话说，v9.2.58 必须从 “value score closure” 转向 “risk-support sufficient statistics closure”。

## 5. 是否还在正确道路上

是，但必须改变下一步优先级。

正确路线：

```text
true delta interface exists
→ exact value signal exists
→ oracle support exists
→ redesign risk/support sufficient statistics
→ exact reference feasibility
→ system-legal true-delta controller
→ LDO / LSO
→ paired replay
```

错误路线：

```text
继续只优化 K7d / branch-delta kernel；
继续只调 exact gap threshold；
继续让 cheap candidate generator 主导；
继续按 dataset 调不同 gate；
把 oracle support 写成 success；
把 AUC 写成 success。
```

v9.2.57 的进展是把一个潜在误区暴露出来：即使 kernel 做成，controller 也可能失败。因此 v9.2.58 必须并行验证 compute 与 decision，但是主线优先级应从 kernel speed 稍微转向 risk/support statistics。

---

# Part II. v9.2.58 总体目标

v9.2.58 的总体目标是：

$$
\boxed{
\text{设计 legal online risk/support sufficient statistics，使 exact branch-delta signal 形成 safe coverage-preserving accept region。}
}
$$

本轮分成七个目标：

```text
1. 对 exact reference controller failure 做 FP/FN/bad-event autopsy；
2. 设计 legal risk/support sufficient statistics；
3. 用 exact reference signal 重做 feasibility frontier；
4. 同步继续 true-delta compute residual / kernel v3，但不让 kernel lane 掩盖 decision blocker；
5. 扩大 support measurement；
6. 如果 exact reference feasibility 与 system gate 同时恢复，打开 LDO/LSO；
7. 如果 LDO/LSO 通过，打开 official paired replay。
```

v9.2.58 的核心 stop-go 问题是：

$$
\boxed{
\text{exact reference + redesigned risk/support 是否可行？}
}
$$

如果不可行，必须声明：

```text
branch-delta value signal lacks deployable risk/support separation.
```

而不是继续把 v9.2.59 写成另一个 kernel-only 计划。

---

# Part III. 核心假设

## H1：P5 失败来自 risk/support insufficient，不是 exact delta value signal 不存在

证据：

```text
exact AUC = 0.888401
exact agreement = 1.0
oracle precision = 1.0
oracle coverage = 0.120040
oracle bad-event = 0.0
reference controller precision = 0.430279
reference controller bad-event = 0.418327
```

H1 成立标准：

新 risk/support statistics 加入后，CBD0 exact reference controller 达到：

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

即使使用 CBD0 exact reference signal 和所有 legal risk/support statistics，仍无 feasible region。

## H2：bad-event 有 legal pre-event / commit-time signature

H2 认为 bad-event 不是完全后验不可预测，它应该在以下统计中留下痕迹：

```text
CE tail instability
margin tail instability
wrong-confidence concentration
branch disagreement
control dominance
risk LCB
support density
family reliability
stratum/family sparsity
delta norm / gain mismatch
```

H2 成立标准：

至少一个 bad-event risk statistic 达到：

$$
AUC_{\text{bad-event}}\geq0.70
$$

or:

$$
Corr(S_{\text{risk}},Y_{\text{bad-event}})\geq0.35.
$$

更重要的是，加入 risk gate 后：

$$
\Delta BadEventRate\leq-0.20
$$

且 coverage 不低于 `0.03`。

H2 失败标准：

所有 legal risk statistics 对 bad-event 的 AUC 都低于 `0.60`，且不能降低 controller bad-event。

## H3：oracle/legal gap 来自 support geometry 缺失

oracle support 通过但 legal controller 不通过，说明 safe-good events 可能集中在某些 family / stratum / support pocket 中。H3 认为 legal support density and family reliability 可以逼近 oracle accept region。

H3 成立标准：

support statistics 使 accepted support 满足：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

同时保持：

$$
Precision\geq0.75,
$$

$$
BadEventRate\leq0.05.
$$

H3 失败标准：

support balancing 只降低 coverage，不提升 precision / bad-event。

## H4：support measurement 太窄会导致 controller illusion 或 controller pessimism

v9.2.57 measured signal strata 只有 `3`，balanced diagnostic rows 只有 `36`。H4 认为这会让 calibration 变得不稳定。

H4 成立标准：

support expansion 后：

```text
natural_real_event_count >= 12000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 6
measured_family_count >= 12
```

并且 precision / bad-event bootstrap variance 明显下降。

H4 失败标准：

support 扩大后 exact reference controller 仍 infeasible，且 failure mode 不再是 sample scarcity。

## H5：compute lane 不应被放弃，但必须降级为 parallel lane

H5 认为 true branch-delta compute 仍需推进，但只有当 exact reference feasibility 恢复时，kernelization 才能转为 primary route。

H5 成立标准：

至少一个 true-delta v3 candidate 达到：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05,
$$

while:

$$
AUC\geq0.70,
$$

$$
Agreement\geq0.90.
$$

H5 失败标准：

true-delta v3 仍太贵，或者系统过线但丢 signal。

## H6：如果 exact reference feasibility 失败，下一步是 sufficient statistics reset，不是 kernel-only

H6 是路线纪律假设。  
如果 CBD0 exact reference + legal risk/support still infeasible，则下一步必须重做：

```text
risk sufficient statistics
support family definition
bad-event label decomposition
control-dominance semantics
safe-good target decomposition
```

而不是再写一个 purely CUDA-focused plan。

---

# Part IV. 并行执行设计

v9.2.58 runner 必须并行执行：

```text
Lane A:
  v9.2.57 boundary reproduction

Lane B:
  exact reference controller failure autopsy:
    false positives
    false negatives
    bad-event accepted rows
    coverage loss
    threshold frontier

Lane C:
  risk sufficient statistics factory:
    CE tail / margin tail / wrong-confidence
    branch disagreement
    control dominance
    risk LCB
    delta-gain mismatch
    volatility / instability

Lane D:
  support sufficient statistics factory:
    support density
    family reliability
    family balance
    stratum coverage
    leave-family-out support

Lane E:
  exact reference feasibility v2:
    CBD0 exact signal + new risk/support stats
    heldout precision / coverage / bad-event frontier

Lane F:
  true delta compute v3:
    residual attribution
    layout repair
    risk/support component fusion
    system gate

Lane G:
  system-legal controller:
    only official if compute + reference feasibility pass

Lane H:
  support expansion:
    natural rows
    balanced diagnostic rows
    signal strata / family coverage

Lane I:
  LDO / LSO

Lane J:
  paired replay scout and official replay
```

Gate discipline：

```text
P5 reference feasibility can run before system pass.
P6 compute can run before controller pass.
P7 official controller cannot pass unless compute and reference feasibility both pass.
P9 paired replay cannot open unless controller + LDO/LSO pass.
```

---

# Part V. Candidate designs

## 1. Base and carrier

Official base remains：

```text
R2-LQ-fanin-output-scale-confirmed
LQ-t2-h256
```

Carrier candidates：

```text
A0-current-v9257
A3-RoleWiseFT7ResetCarrier
A6-HybridRoleControlRiskCarrier
```

Carrier reset is not primary unless fresh oracle support collapses.

---

## 2. Risk sufficient statistics candidates

All risk statistics must be legal at commit time.

### RISK0：Reference current risk

Existing risk score used in v9.2.57. Reference only.

### RISK1：TailInstabilityLCB

Use train-stream CE / margin tail instability:

$$
S_{\text{tail-risk}}
=
CEp99
+
\alpha_1 WrongConfP95
-
\alpha_2 MarginP10
+
\alpha_3 Volatility_{\text{tail}}.
$$

Volatility is computed over recent train-stream windows, not validation/test.

### RISK2：BranchDisagreementRisk

Bad events may occur when RealFunctional gain is high but branch outputs disagree.

$$
S_{\text{branch-disagree}}
=
\operatorname{Var}_b(Gain_b)
+
\operatorname{Var}_b(Margin_b)
+
\operatorname{Var}_b(CE_b).
$$

### RISK3：ControlDominanceRisk

If AdamWParallel / bestLR is nearly as good as RealFunctional or dominates on risk tail, functional update is unsafe.

$$
S_{\text{control-risk}}
=
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})
-
Gain_{\text{RealFunctional}}
+
\lambda Risk_{\text{control-tail}}.
$$

### RISK4：DeltaGainMismatch

Large delta norm with small realized gain often indicates unstable or spurious movement.

$$
S_{\text{mismatch}}
=
\frac{\|\Delta z_F\|}
{|Gain_F|+\epsilon}
+
\frac{\|\Delta\theta_F\|}
{|Gain_F|+\epsilon}.
$$

### RISK5：RiskLCB

Lower confidence bound over event family:

$$
LCB_{\text{safe}}(f)
=
\mu_{\text{safe}}(f)
-
\kappa
\sqrt{
\frac{\sigma^2_{\text{safe}}(f)}{n_f+\epsilon}
}.
$$

Risk score:

$$
S_{\text{risk-lcb}}=1-LCB_{\text{safe}}(f).
$$

### RISK6：HybridRiskSufficientStatistic

Monotone combination:

$$
S_{\text{risk}}
=
a_1S_{\text{tail-risk}}
+
a_2S_{\text{branch-disagree}}
+
a_3S_{\text{control-risk}}
+
a_4S_{\text{mismatch}}
+
a_5S_{\text{risk-lcb}}.
$$

Coefficients are selected on train-stream calibration split only.

---

## 3. Support sufficient statistics candidates

### SUP0：Reference support

Existing support metric. Reference only.

### SUP1：KNNFeatureDensity

Feature vector:

```text
gap
risk
margin_tail
CE_tail
control_gap
branch_disagreement
delta_norm
stratum id
role bucket
horizon bucket
```

Density:

$$
Density(e)
=
\frac{k}{N\cdot Volume(\mathcal{N}_k(e))}.
$$

### SUP2：FamilyReliabilityV2

Family must not contain dataset name:

$$
family(e)
=
(stratum,horizon,risk\_bucket,role\_bucket,attach\_type,branch\_bucket,delta\_bucket,control\_gap\_bucket).
$$

Reliability:

$$
Rel(f)
=
\mathbb{E}[Y_{\text{safe-good}}\mid f]
-
\kappa
\sqrt{
\frac{\operatorname{Var}(Y_{\text{safe-good}}\mid f)}{n_f+\epsilon}
}.
$$

### SUP3：LeaveFamilyOutReliability

A family is reliable only if its nearby families also support safe-good:

$$
Rel_{\text{LFO}}(f)
=
\min_{f'\in\mathcal{N}(f)}
Rel(f').
$$

### SUP4：BalancedSupportGate

Reject if a single family dominates accepted rows:

$$
max\_family\_share \leq 0.60.
$$

### SUP5：SupportRiskJointPocket

Safe accept pockets are defined by joint support:

$$
Pocket(e)
=
\mathbb{1}
[
Density(e)\geq d_0
]
\cdot
\mathbb{1}
[
Rel(f)\geq r_0
]
\cdot
\mathbb{1}
[
S_{\text{risk}}\leq\rho
].
$$

---

## 4. Controller candidates

### C0：v9.2.57 reference

Reference only.

### C1：ExactGapRiskLCBController

$$
Accept(e)
=
Gap(e)\geq\tau_g
\land
S_{\text{risk-lcb}}(e)\leq\rho
\land
Density(e)\geq d_0.
$$

### C2：ExactGapHybridRiskController

$$
Accept(e)
=
Gap(e)\geq\tau_g
\land
S_{\text{risk-hybrid}}(e)\leq\rho
\land
SupportStable(e).
$$

### C3：SupportPocketController

$$
Accept(e)
=
Gap(e)\geq\tau_g
\land
Pocket(e)=1.
$$

### C4：ParetoRiskSupportController

Accept events on Pareto frontier of:

```text
positive gap
low risk
high support density
high family reliability
low control dominance
low delta-gain mismatch
```

### C5：TwoStageSafeCoverageController

Stage 1 high-precision safe core:

$$
Core(e)
=
Gap(e)\geq\tau_{g,high}
\land
Risk(e)\leq\rho_{low}.
$$

Stage 2 coverage extension only from reliable support pockets:

$$
Extension(e)
=
Gap(e)\geq\tau_{g,mid}
\land
Risk(e)\leq\rho_{mid}
\land
Rel_{\text{LFO}}(f)\geq r_0.
$$

Final:

$$
Accept(e)=Core(e)\lor Extension(e).
$$

### C6：Oracle

Posthoc diagnostic only. Never official.

---

# Part VI. 实验阶段

## P0：v9.2.57 boundary reproduction

### 目标

确认 v9.2.57 boundary 稳定，尤其是 exact reference infeasibility。

### 必须记录

```text
route
source_route_v9256
true_delta_legality_pass
true_delta_predictivity_pass
true_delta_agreement_pass
true_delta_system_pass
true_delta_auc
true_delta_agreement
true_delta_step_ratio
reference_controller_precision
reference_controller_coverage
reference_controller_bad_event
oracle_precision
oracle_coverage
oracle_bad_event
support_measurement_pass
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R11-ExactSignalControllerInfeasible
true_delta_legality_pass = 1
true_delta_predictivity_pass = 1
reference_controller_feasible = 0
oracle_support_pass = 1
fake/proxy/offload = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_exact_signal_vs_controller_failure.svg
p0_oracle_legal_gap.svg
```

---

## P1：exact reference controller failure autopsy

### 目标

把 P5 failure 拆成 FP / FN / bad-event / coverage-loss，而不是继续只看 precision/coverage/bad-event aggregate。

### 必须记录

```text
row_id
split_id
controller_id
accepted
safe_good
bad_event
false_positive
false_negative
coverage_loss
gap_score
risk_score_current
support_score_current
oracle_accept
event_family
signal_stratum
dataset
seed
horizon
failure_mode
```

Failure modes：

```text
FPA1-high_gap_high_risk
FPA2-control_dominant_bad_event
FPA3-low_support_pocket
FPA4-family_reliability_mismatch
FPA5-delta_gain_mismatch
FPA6-tail_instability
FNA1-gap_threshold_too_high
FNA2-risk_gate_overreject
FNA3-support_gate_overreject
FNA4-family_scarcity
```

### 判断标准

Autopsy pass：

```text
false_positive_attribution_fraction >= 0.80
false_negative_attribution_fraction >= 0.80
bad_event_attribution_fraction >= 0.80
```

### 可视化

```text
p1_fp_fn_bad_event_sankey.svg
p1_gap_vs_bad_event_scatter.svg
p1_risk_support_failure_heatmap.svg
p1_threshold_frontier_old.svg
```

---

## P2：risk sufficient statistics factory

### 目标

寻找 legal online risk statistic，专门压低 bad-event。

### 必须记录

```text
risk_stat_id
features_used
uses_dataset_name
uses_validation
uses_test
uses_posthoc
AUC_bad_event
corr_bad_event
AUC_risk_safe
precision_after_risk_gate
coverage_after_risk_gate
bad_event_after_risk_gate
bad_event_delta_vs_reference
coverage_delta_vs_reference
feature_overhead
memory_overhead
```

### 判断标准

Risk statistic pass：

$$
AUC_{\text{bad-event}}\geq0.70
$$

or:

$$
Corr_{\text{bad-event}}\geq0.35.
$$

Controller utility pass：

$$
BadEventRate_{\text{after risk gate}}\leq0.05
$$

while:

$$
Coverage\geq0.03.
$$

Diagnostic pass：

$$
AUC_{\text{bad-event}}\geq0.60
$$

and:

$$
\Delta BadEventRate\leq-0.20.
$$

### 可视化

```text
p2_risk_stat_auc_matrix.svg
p2_risk_gate_precision_coverage_bad.svg
p2_bad_event_reduction_curve.svg
p2_risk_feature_ablation.svg
```

---

## P3：support sufficient statistics factory

### 目标

寻找 legal online support statistic，减少 oracle/legal gap。

### 必须记录

```text
support_stat_id
family_definition
uses_dataset_name
support_density_method
family_reliability_method
measured_family_count
accepted_family_count
accepted_signal_strata_count
max_family_share
precision_after_support_gate
coverage_after_support_gate
bad_event_after_support_gate
oracle_overlap
legal_oracle_jaccard
feature_overhead
```

### 判断标准

Support statistic pass：

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

```text
legal_oracle_jaccard >= 0.50
```

or:

```text
bad_event_delta <= -0.20 with coverage >= 0.03
```

### 可视化

```text
p3_support_family_heatmap.svg
p3_oracle_legal_overlap.svg
p3_support_density_vs_bad_event.svg
p3_family_reliability_curve.svg
```

---

## P4：exact reference feasibility v2

### 目标

用 CBD0 exact reference signal + new risk/support stats 判断 decision geometry 是否可行。

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
oracle_overlap
dataset_name_used
posthoc_used_at_commit
official_eligible_reference_only
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
R8-ExactReferenceStillInfeasible
```

### 可视化

```text
p4_reference_feasibility_frontier.svg
p4_risk_support_threshold_surface.svg
p4_reference_oracle_gap_after_repair.svg
p4_accepted_region_geometry.svg
```

---

## P5：true-delta compute v3 parallel lane

### 目标

继续推进 compute，但不让 compute lane 掩盖 controller infeasibility。

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

and signal retained.

### 可视化

```text
p5_true_delta_v3_cost_signal_pareto.svg
p5_residual_subphase_after_repair.svg
p5_compute_vs_controller_gate_ladder.svg
```

---

## P6：support expansion

### 目标

修复 measured strata 太窄的问题。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
horizons = 20,80,240,640
signal_strata = S1-S8
carriers = A0,A3,A6
row_sources = natural, balanced_diagnostic
```

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
risk_stat_values
support_stat_values
```

### 判断标准

Measurement pass：

```text
natural_real_event_count >= 12000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 6
measured_family_count >= 12
```

Official accepted support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

### 可视化

```text
p6_signal_strata_coverage.svg
p6_family_support_heatmap.svg
p6_natural_vs_balanced_distribution.svg
p6_bad_event_by_stratum_family.svg
```

---

## P7：system-legal exact-signal controller

### 目标

如果 P4 reference feasible 且 P5 compute pass，则建立 official controller。

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
contract_audit_v9258.csv
p0_v9257_boundary_reproduction.csv
p1_exact_reference_controller_failure_autopsy.csv
p2_risk_sufficient_statistics_factory.csv
p3_support_sufficient_statistics_factory.csv
p4_exact_reference_feasibility_v2.csv
p5_true_delta_compute_v3_parallel_lane.csv
p6_online_support_stratum_expansion.csv
p7_system_legal_exact_signal_controller.csv
p8_leave_dataset_and_stratum_out.csv
p9_official_paired_replay.csv
p10_short_run_functional_validation.csv
risk_stat_trace_v9258.csv
support_stat_trace_v9258.csv
reference_feasibility_trace_v9258.csv
true_delta_compute_trace_v9258.csv
accepted_region_geometry_trace_v9258.csv
support_density_trace_v9258.csv
leaveout_trace_v9258.csv
paired_replay_branch_trace_v9258.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9257_boundary_unstable
F3_dataset_tuning_detected
F4_reference_failure_autopsy_incomplete
F5_bad_event_unattributed
F6_risk_stat_not_predictive
F7_risk_gate_bad_event_fail
F8_support_stat_no_oracle_overlap
F9_support_gate_coverage_fail
F10_exact_reference_still_infeasible
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
R1-ReferenceFailureAutopsyPass:
  exact reference controller failure is attributed.

R2-RiskSufficientStatisticPass:
  legal risk statistic predicts bad-event / risk-safe and reduces bad-event.

R3-SupportSufficientStatisticPass:
  legal support statistic improves oracle/legal overlap and family balance.

R4-ExactReferenceFeasibilityRestored:
  CBD0 exact reference + new risk/support stats passes precision / coverage / bad-event gate.

R5-TrueDeltaComputeV3SystemPass:
  true branch-delta v3 passes system envelope while preserving signal.

R6-SystemLegalExactSignalControllerPass:
  system-legal true-delta controller passes heldout gate.

R7-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R8-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R9-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R10-ExactReferenceStillInfeasible:
  even CBD0 exact reference + redesigned risk/support cannot form legal accept region.

R11-RiskSupportStatsMissing:
  bad-event or support gap remains unattributed.

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
v9257_boundary_pass
dataset_tuning_detected
reference_failure_autopsy_pass
fp_attribution_fraction
fn_attribution_fraction
bad_event_attribution_fraction
best_risk_stat_id
risk_stat_pass
risk_stat_auc_bad_event
risk_gate_bad_event
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
success_v9258_strict_purekan_functional
success_v9258_full_functional
success_v9258_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 exact reference failure autopsy
  P2 risk sufficient statistics factory
  P3 support sufficient statistics factory
  P4 exact reference feasibility v2
  P5 true delta compute v3
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
P4 exact reference feasibility can pass diagnostically before compute pass.
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
v9.2.57 boundary reproduced
exact reference controller failure autopsied
risk sufficient statistics measured
support sufficient statistics measured
exact reference feasibility v2 measured
true delta compute v3 measured
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
1. v9.2.57 boundary cannot be reproduced；
2. exact reference failure cannot be attributed；
3. legal risk statistics cannot reduce bad-event；
4. legal support statistics cannot recover oracle/legal overlap；
5. exact reference remains infeasible；
6. true branch-delta compute remains too expensive；
7. true branch-delta compute passes system but loses signal；
8. system-legal controller cannot meet precision / coverage / bad-event；
9. online support remains too narrow；
10. fresh natural oracle support collapses；
11. leave-dataset-out fails；
12. leave-stratum-out fails；
13. paired replay remains control-equivalent；
14. shuffle controls pass, indicating overfit；
15. short-run task drops；
16. full run gives no macro / hard-stratum / geometry gain；
17. functional breaks system gate；
18. gains are explained by QuadraticFeatureMLP；
19. any teacher/loss/fake/proxy/offload violation occurs。
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

下一步修 risk/support sufficient statistics，不继续 kernel-only。

## Case D：exact reference remains infeasible

必须声明：

```text
branch-delta exact signal has high AUC but lacks sufficient legal risk/support separation.
```

下一步重设 risk/support target decomposition and bad-event sufficient statistics。

## Case E：support expansion fixes controller variance

可以声明：

```text
previous infeasibility was partly narrow-support artifact.
```

但仍需 LDO/LSO and paired replay.

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

v9.2.58 的一句话策略是：

$$
\boxed{
\text{先证明 exact signal 存在可部署 accept region，再把这个 region 用 system-legal true delta 实现。}
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
1. exact reference controller 的 FP / FN / bad-event 到底来自哪里？
2. bad-event 是否有 legal online risk signature？
3. oracle support 是否能被 support density / family reliability 逼近？
4. redesigned risk/support stats 能否让 CBD0 exact reference controller 过 precision / coverage / bad-event？
5. true delta compute v3 能否继续压到 step <= 1.50？
6. 如果 compute pass，system-legal controller 能否复现 reference feasible region？
7. support 是否能扩展到 >=6 measured strata？
8. controller 能否 LDO/LSO？
9. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.58 的结果将给出清晰分叉：

```text
if reference feasible + compute pass + controller + LDO/LSO + paired replay pass:
  strict PureKAN functional obtains local causal evidence.

if reference feasible but compute fail:
  kernelization is blocker.

if compute pass but reference infeasible:
  risk/support sufficient statistics are blocker.

if both fail:
  exact branch-delta remains diagnostic; redesign sufficient statistics and kernel representation.

if oracle support collapses:
  carrier/support stability is blocker.
```
