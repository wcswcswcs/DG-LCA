# DG-KAN v9.2.60 Clean-Core Expansion 与 Constraint-Calibrated Support Frontier 完整实验计划

> 本计划基于 v9.2.59 `Oracle-Decomposed Risk/Support Reset 与 Deployable Exact-Signal Controller` 的真实复盘制定。  
> v9.2.59 的 terminal route 是：
>
> ```text
> route = R11-RiskSupportStatsStillTinyCleanCore
> base_candidate = LQ-t2-h256
> success_v9259_strict_purekan_functional = False
> success_v9259_full_functional = False
> success_v9259_external_ready = False
> ```
>
> v9.2.59 的关键事实是：
>
> ```text
> P1 oracle-decomposed factor label/autopsy:
>   pass = 1
>   FP = 1108
>   FN = 1388
>   accepted bad-event = 1035
>   FP / FN / bad-event attribution fraction = 1.0
>
> P2 mode-specific risk:
>   four bad-event submodes are predictive
>   best route risk union bad-event = 0.0
>   risk union coverage = 0.001860
>   mode_specific_risk_pass = 0
>
> P3 support reset:
>   best = SUPR2-LeaveFamilyOutReliabilityV2
>   precision = 0.382818
>   coverage = 0.082279
>   bad-event = 0.542577
>   Jaccard = 0.328803
>   support_stat_pass = 0
>
> P4 exact reference feasibility v3:
>   best = RISKM3-ControlDominanceRisk + SUPR4-ModeAwareSupportPocket
>   precision = 0.886792
>   coverage = 0.003286
>   bad-event = 0.037736
>   exact_reference_feasible = 0
>
> P5 true-delta compute v4:
>   best = TBD0-V9256CBD0Reference
>   AUC = 0.883468
>   agreement = 1.0
>   step ratio = 2.863280 > 1.50
>   true_delta_compute_pass = 0
>
> P6 oracle support:
>   precision = 1.0
>   coverage = 0.119978
>   bad-event = 0.0
>   measured signal strata = 3
>   balanced diagnostic rows = 162
>   support_measurement_pass = 0
>
> current blocker:
>   risk_union_gate_tiny_coverage
> ```
>
> v9.2.60 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.59 已经找到了 clean core，但还没有找到 deployable region。}
> }
> $$
>
> v9.2.59 不是“没有进展”。相对 v9.2.58，exact reference best region 从 coverage `0.000992` 推进到 `0.003286`，同时仍保持 precision `0.886792` 与 bad-event `0.037736`。这是一个真实改善。但它仍远低于 official coverage 下限 `0.03`，至少还需要约：
>
> $$
> \frac{0.03}{0.003286}\approx 9.13
> $$
>
> 倍的 coverage expansion。
>
> 因此 v9.2.60 不应继续简单做：
>
> ```text
> 再加一个 hybrid risk score；
> 再调一个 gap threshold；
> 再把所有 risk modes 做更硬的 AND gate；
> 再做一个 kernel-only 版本；
> 按 dataset 单独调阈值。
> ```
>
> 本轮的核心任务是把 **tiny clean core** 变成 **constraint-calibrated deployable frontier**：
>
> $$
> \boxed{
> \text{从 clean core 出发，沿 legal support / family / risk frontier 扩张 coverage，同时严格约束 precision 与 bad-event。}
> }
> $$

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

数据集只能作为诊断切片，不能作为 official controller 条件。允许报告不同数据集上的 bad-event mode、support density、family scarcity、tail instability、delta-gain mismatch 分布；不允许使用 dataset name 选择 controller、risk statistic、support family、threshold 或 abstention rule。

---

# Part I. 对 v9.2.59 的独立判断

## 1. v9.2.59 没有达到目标

v9.2.59 没有 strict PureKAN functional success。失败不是发生在 full run，也不是 paired replay 输了，而是更前面的 decision geometry 与 compute gate 均未闭合：

```text
exact_reference_feasible = 0
mode_specific_risk_pass = 0
support_stat_pass = 0
true_delta_compute_pass = 0
support_measurement_pass = 0
LDO / LSO / paired replay = not_run
```

因此不能声明：

```text
strict PureKAN local causal evidence
full functional success
external-ready success
Beyond-MLP success
```

特别要注意：v9.2.59 的 best exact reference controller precision 与 bad-event 已经进入不错范围：

```text
precision = 0.886792
bad-event = 0.037736
```

但 coverage 只有：

```text
coverage = 0.003286
```

这不是 near-pass。它离 official lower bound `0.03` 仍差约 9 倍。

## 2. v9.2.59 的真实进展

v9.2.59 的进展很重要，主要体现在三个方面。

第一，oracle-decomposed factor autopsy 已经闭合。FP/FN/bad-event attribution fraction 均为 `1.0`，说明 failure 不是黑箱。当前可解释 failure modes 包括：

```text
delta_gain_mismatch
family_scarcity
tail_instability
control_dominance
support pocket mismatch
```

第二，mode-specific risk 有预测性。P2 显示四个 bad-event submodes 都有 predictive signal，并且 risk union 可以做到：

```text
bad-event = 0.0
```

这说明 bad-event 不是完全不可预测。问题是 risk union 太保守，coverage 只有 `0.001860`。

第三，exact reference feasibility 有改善。v9.2.58 的 clean slice coverage 是 `0.000992`；v9.2.59 的 best exact reference region coverage 到了 `0.003286`，且 precision `0.886792`、bad-event `0.037736`。这说明 factorized risk/support 方向不是空的，它确实把 clean region 扩张了一些。

## 3. v9.2.59 的真实失败

v9.2.59 的核心失败不是 risk 没信号，而是：

$$
\boxed{
\text{risk/support gate 仍然是 tiny clean core，不是 deployable frontier。}
}
$$

当前 controller 有两个极端。

宽 gate 的结果是：

```text
coverage 可以上来，例如 support reset coverage = 0.082279
但 precision = 0.382818
bad-event = 0.542577
```

窄 gate 的结果是：

```text
precision = 0.886792
bad-event = 0.037736
但 coverage = 0.003286
```

这表明当前缺的不是单个 risk score，而是一个 **coverage expansion mechanism under constraints**。换句话说，我们需要在 clean core 周围找到可扩张的 legal support neighborhoods，而不是继续把所有风险模式硬 AND 起来。

## 4. 当前 blocker 的本质

当前 blocker 应写成：

$$
\boxed{
\text{clean-core-to-frontier expansion failure。}
}
$$

更细地说：

```text
1. Exact branch-delta value signal 仍强；
2. Oracle support 仍强，coverage 约 0.12；
3. Legal risk/statistics 能找到一小片 clean core；
4. Legal support/statistics 还不能从 clean core 扩张到 oracle-size region；
5. true-delta compute 仍太贵，但 compute 不是当前唯一 blocker；
6. support measurement 仍太窄，signal strata 只有 3，balanced diagnostic rows 只有 162。
```

因此，v9.2.60 的重点必须从 “设计更严 risk gate” 改成 “在 constraints 下扩张 accept frontier”。

---

# Part II. v9.2.60 总体目标

v9.2.60 的总体目标是：

$$
\boxed{
\text{把 v9.2.59 的 tiny clean core 扩张为 deployable exact-signal accept region。}
}
$$

更具体地说，本轮必须同时验证：

```text
1. clean core 周围是否存在可安全扩张的 near-core / same-family / neighbor-family region；
2. bad-event risk 是否能从 hard veto 改为 calibrated risk budget；
3. support family 是否能从 sparse LCB 改为 multi-resolution reliability frontier；
4. exact reference feasibility 是否能达到 coverage >= 0.03；
5. true-delta compute 是否继续向 system envelope 收敛；
6. support measurement 是否真正扩展到 >=6 signal strata 和 >=16 families；
7. 如果 decision + compute 双过，是否能打开 LDO/LSO 与 paired replay。
```

本轮最重要的 stop-go 结论不是 “某个新 score AUC 更高”，而是：

$$
\boxed{
\text{exact reference clean core 是否可扩张？}
}
$$

如果 clean core 不可扩张，那么 functional update 的 current branch-delta score 仍只是 diagnostic signal。  
如果 clean core 可扩张但 compute 不过，那么 kernelization 仍是 blocker。  
如果两者都过，再进入 LDO/LSO 与 official paired replay。

---

# Part III. 核心假设

## H1：v9.2.59 的 clean core 是 deployable frontier 的保守下界

v9.2.59 已经找到：

$$
Precision=0.886792,
$$

$$
BadEventRate=0.037736,
$$

$$
Coverage=0.003286.
$$

H1 认为这不是孤立 tiny artifact，而是 deployable frontier 的 core seed。沿着相近的 risk/support/family neighborhoods 扩张，可以把 coverage 拉到 official range。

H1 成立标准：

在 CBD0 exact reference 上，core expansion 后：

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

所有 legal expansion strategies 都只能得到：

```text
coverage < 0.01
```

或 coverage 上来后：

```text
bad-event > 0.05
precision < 0.75
```

## H2：mode-specific risk 不应继续做 hard-AND，应做 calibrated risk budget

v9.2.59 的 risk union 可以做到 bad-event `0.0`，但 coverage 只有 `0.001860`。这说明 hard-AND 过保守。H2 认为不同 risk modes 应该作为 calibrated bad-event probability / budget，而不是所有 mode 一票否决。

定义每个 mode 的 legal risk probability：

$$
p_m(e)=\Pr(Y_{\text{bad},m}=1\mid x_e).
$$

组合风险：

$$
P_{\text{bad}}(e)
=
1-\prod_m(1-p_m(e)).
$$

或者使用风险预算：

$$
B(e)=\sum_m \lambda_m \cdot \mathbb{1}[p_m(e)>\rho_m].
$$

Accept 只要求：

$$
P_{\text{bad}}(e)\leq\rho_{\text{global}}
$$

or:

$$
B(e)\leq B_0.
$$

H2 成立标准：

risk-budget controller 比 hard-union controller coverage 至少提升：

$$
Coverage_{\text{budget}}-Coverage_{\text{union}}\geq0.02,
$$

且仍满足：

$$
BadEventRate\leq0.05.
$$

H2 失败标准：

risk-budget 一放宽就 bad-event 爆炸，说明 current bad-event modes 不是可组合预算问题，而是 missing risk variable。

## H3：support reset 失败来自 family resolution 错配

v9.2.59 的 SUPR2 结果：

```text
precision = 0.382818
coverage = 0.082279
bad-event = 0.542577
Jaccard = 0.328803
```

这说明 support statistic 找到了一些覆盖，但覆盖中混入大量 bad-event。H3 认为 family 粒度仍然错配：过粗 family 把安全 pocket 与高风险 pocket 混在一起，过细 family 又导致 family scarcity。

H3 成立标准：

multi-resolution family controller 达到：

$$
Jaccard_{\text{legal,oracle}}\geq0.60
$$

or:

$$
Precision\geq0.75,\quad
Coverage\in[0.03,0.15],\quad
BadEventRate\leq0.05.
$$

H3 失败标准：

multi-resolution family 只带来 sparsity，不能提升 legal-oracle overlap，也不能降低 bad-event。

## H4：support measurement 太窄仍是关键风险

v9.2.59 仍只有：

```text
measured signal strata = 3
balanced diagnostic rows = 162
```

这与计划要求相差很大。H4 认为当前 controller 的悲观或乐观都可能被 narrow support 放大。

H4 成立标准：

本轮必须达到：

```text
natural_real_event_count >= 20000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 6
measured_family_count >= 16
```

并且所有 key metrics 报告 bootstrap CI：

```text
precision CI
coverage CI
bad-event CI
Jaccard CI
family-share CI
```

H4 失败标准：

如果 support measurement 仍停在 3 strata 或 balanced diagnostic rows 远低于 6000，则任何 controller pass 都只能记为 diagnostic，不得 official promotion。

## H5：true-delta compute 是 parallel blocker，不是 v9.2.60 的唯一主 blocker

v9.2.59 true-delta compute：

```text
AUC = 0.883468
agreement = 1.0
step ratio = 2.863280
```

H5 成立标准：

至少一个 true-delta v5 candidate 达到：

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

但如果 H1-H4 失败，即使 H5 成立，也不能声明 functional success，因为 value signal 仍不可部署。

---

# Part IV. Clean-core expansion formulation

## 1. Clean core

用 v9.2.59 best exact reference controller 定义 clean core：

$$
Core(e)
=
\mathbb{1}
[
Gap(e)\geq\tau_{core}
]
\land
\mathbb{1}
[
Risk(e)\leq\rho_{core}
]
\land
\mathbb{1}
[
Support(e)\geq d_{core}
].
$$

v9.2.59 的 core 大约满足：

$$
Precision(Core)\approx0.886792,
$$

$$
BadEvent(Core)\approx0.037736,
$$

$$
Coverage(Core)\approx0.003286.
$$

## 2. Near-core candidates

定义 near-core candidates 为：

$$
Near(e)
=
\mathbb{1}
[
d(e,Core)\leq \epsilon_d
]
\lor
\mathbb{1}
[
family(e)\in\mathcal{N}(family(Core))
]
\lor
\mathbb{1}
[
RiskBudget(e)\leq B_0
].
$$

其中 $d(e,Core)$ 是 risk/support/gap space 中的 kNN distance：

$$
d(e,Core)
=
\min_{c\in Core}
\left\|
\Sigma^{-1/2}
(
\psi(e)-\psi(c)
)
\right\|_2.
$$

Feature vector：

```text
gap
control_gap
tail_risk
delta_gain_mismatch
control_dominance
branch_disagreement
family_reliability
support_density
horizon_bucket
stratum_bucket
role_bucket
```

## 3. Constraint-calibrated frontier

最终 accept frontier：

$$
Accept(e)
=
Core(e)
\lor
[
Near(e)
\land
RiskBudgetSafe(e)
\land
SupportReliable(e)
\land
CoverageBalanced(e)
].
$$

其中：

$$
RiskBudgetSafe(e)=\mathbb{1}[P_{\text{bad}}(e)\leq\rho_{\text{global}}],
$$

$$
SupportReliable(e)=
\mathbb{1}[Rel_{\text{multi}}(family(e))\geq r_0],
$$

$$
CoverageBalanced(e)
=
\mathbb{1}[max\_family\_share\leq0.60].
$$

## 4. Objective for calibration

Calibration 不直接最大化 AUC，而是最大化 constrained coverage：

$$
\max_{\tau,\rho,r,d}
Coverage(A_{\tau,\rho,r,d})
$$

subject to：

$$
Precision(A_{\tau,\rho,r,d})\geq0.75,
$$

$$
BadEventRate(A_{\tau,\rho,r,d})\leq0.05,
$$

$$
max\_family\_share(A_{\tau,\rho,r,d})\leq0.60.
$$

为了避免 tiny-slice overfit，加入 lower confidence constraints：

$$
Precision_{\text{LCB}} \geq 0.75,
$$

$$
BadEventRate_{\text{UCB}} \leq 0.05.
$$

---

# Part V. Candidate designs

## 1. Risk budget candidates

### RB0：v9.2.59 hard union reference

Reference only.

### RB1：CalibratedBadProbability

用每个 bad-event submode 的 calibration score 估计：

$$
P_{\text{bad}}(e)
=
1-\prod_m(1-p_m(e)).
$$

### RB2：ModeWeightedRiskBudget

$$
B(e)
=
\lambda_t I_{\text{tail}}(e)
+
\lambda_m I_{\text{mismatch}}(e)
+
\lambda_c I_{\text{control}}(e)
+
\lambda_f I_{\text{family}}(e).
$$

Accept if：

$$
B(e)\leq B_0.
$$

### RB3：TailFirstSoftBudget

Tail instability 是 accepted bad-event primary mode，所以 tail mode 仍是 hard veto，其他 modes budgeted：

$$
RiskSafe(e)
=
\mathbb{1}[p_{\text{tail}}(e)\leq\rho_t]
\land
\mathbb{1}
[
\lambda_m p_m+\lambda_c p_c+\lambda_f p_f\leq B_0
].
$$

### RB4：ControlGapConditionalRisk

对 high control-gap rows 放宽部分 risk，对 low control-gap rows 收紧：

$$
\rho(e)=
\rho_0
+
\alpha \cdot \sigma(Gap(e)-\tau_g).
$$

### RB5：CleanCoreExpansionRisk

只对 near-core rows 放宽，远离 core 的 rows 仍采用 hard veto：

$$
RiskSafe(e)
=
\begin{cases}
P_{\text{bad}}(e)\leq\rho_{\text{near}}, & e\in Near(Core) \\
P_{\text{bad}}(e)\leq\rho_{\text{hard}}, & otherwise
\end{cases}
$$

---

## 2. Support frontier candidates

### SF0：v9.2.59 support reference

Reference only.

### SF1：MultiResolutionFamilyReliability

定义三层 family：

```text
coarse:
  stratum, horizon, risk_bucket

mid:
  stratum, horizon, risk_bucket, role_bucket, control_gap_bucket

fine:
  stratum, horizon, risk_bucket, role_bucket, control_gap_bucket,
  delta_gain_bucket, tail_stability_bucket, family_risk_bucket
```

Reliability：

$$
Rel_{\text{multi}}(e)
=
\min(
Rel_{\text{coarse}},
Rel_{\text{mid}},
Rel_{\text{fine}}
)
$$

or empirical-Bayes smoothed：

$$
Rel_{\text{EB}}(f)
=
\frac{n_f}{n_f+\alpha}\hat{p}_f
+
\frac{\alpha}{n_f+\alpha}\hat{p}_{parent(f)}.
$$

### SF2：CoreNeighborFamilyExpansion

Accept nearby families if they share clean-core neighbors:

$$
Rel_{\text{core-neighbor}}(f)
=
\frac{|N(f)\cap Core|}{|N(f)|+\epsilon}.
$$

### SF3：LeaveFamilyOutStabilityV3

Family reliability must survive leave-neighbor-family-out:

$$
Rel_{\text{LFO3}}(f)
=
\min_{f'\in\mathcal{N}_3(f)}
Rel(f').
$$

### SF4：Density-Risk Joint Pocket

Support is defined in joint risk/value space:

$$
Pocket(e)
=
\mathbb{1}[Density_{\psi}(e)\geq d_0]
\land
\mathbb{1}[P_{\text{bad}}(e)\leq\rho]
\land
\mathbb{1}[Gap(e)\geq\tau].
$$

### SF5：BalancedExpansionCap

Expansion is allowed only if global accepted distribution stays balanced:

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

---

## 3. Controller candidates

### C0：v9.2.59 best reference

Reference only.

### C1：CleanCoreOnly

Validates the core:

$$
Accept(e)=Core(e).
$$

Expected to be safe but too low coverage.

### C2：CleanCorePlusRiskBudgetExpansion

$$
Accept(e)
=
Core(e)
\lor
[
Near(e)
\land
RB2(e)
\land
SF1(e)
].
$$

### C3：TailFirstExpansion

$$
Accept(e)
=
Core(e)
\lor
[
Near(e)
\land
RB3(e)
\land
SF2(e)
].
$$

### C4：CoreNeighborFamilyExpansion

$$
Accept(e)
=
Core(e)
\lor
[
Rel_{\text{core-neighbor}}(family(e))\geq r_0
\land
P_{\text{bad}}(e)\leq\rho
\land
Gap(e)\geq\tau
].
$$

### C5：TwoStageLCBController

Stage 1 high precision core：

$$
Core(e)=
Gap(e)\geq\tau_{high}
\land
P_{\text{bad}}(e)\leq\rho_{low}.
$$

Stage 2 expansion：

$$
Extension(e)=
Gap(e)\geq\tau_{mid}
\land
P_{\text{bad}}(e)\leq\rho_{mid}
\land
Rel_{\text{multi}}(e)\geq r_0
\land
Density(e)\geq d_0.
$$

Final：

$$
Accept(e)=Core(e)\lor Extension(e).
$$

### C6：ConstrainedFrontierOptimizer

Grid / monotone search over thresholds:

$$
\max Coverage
$$

subject to：

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEvent_{\text{UCB}}\leq0.05,
$$

$$
max\_family\_share\leq0.60.
$$

### C7：Oracle

Posthoc diagnostic only. Never official.

---

# Part VI. 实验阶段

## P0：v9.2.59 boundary reproduction

### 目标

确认 v9.2.59 的 terminal boundary 稳定。P0 不是重复报告，而是防止后续 expansion 基于不稳定 source。

### 必须记录

```text
route
source_route_v9258
mode_specific_risk_pass
risk_union_gate_bad_event
risk_union_gate_coverage
support_stat_pass
support_precision
support_coverage
support_bad_event
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
measured_signal_strata_count
balanced_diagnostic_rows
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R11-RiskSupportStatsStillTinyCleanCore
oracle_support_pass = 1
exact_reference_feasible = 0
risk_union_gate_tiny_coverage confirmed
fake/proxy/offload = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_v9258_to_v9259_coverage_ladder.svg
p0_clean_core_vs_oracle_gap.svg
```

---

## P1：clean-core / near-core autopsy

### 目标

把 v9.2.59 的 tiny clean core 拆开，判断它是不是可扩张的 core seed。

### 必须记录

```text
row_id
split_id
dataset
seed
horizon
signal_stratum
event_family
core_accept
oracle_accept
safe_good
bad_event
gap_score
risk_union_score
bad_probability
support_score
family_reliability
distance_to_core
near_core
blocked_by_tail_risk
blocked_by_mismatch_risk
blocked_by_control_risk
blocked_by_family_risk
blocked_by_support
blocked_by_gap
```

核心 row classes：

```text
A: core accepted safe-good
B: core accepted bad-event
C: oracle accepted but legal rejected
D: legal accepted but oracle rejected
E: near-core rejected safe-good
F: near-core rejected bad-event
```

### 判断标准

P1 pass：

```text
core_bad_event <= 0.05
near_core_oracle_overlap >= 0.30
near_core_safe_good_density >= global_safe_good_density + 0.10
near_core_bad_event <= 0.20 before final risk gate
```

### 可视化

```text
p1_core_nearcore_sankey.svg
p1_distance_to_core_histogram.svg
p1_oracle_miss_by_veto_reason.svg
p1_nearcore_bad_event_by_mode.svg
```

---

## P2：support measurement expansion

### 目标

修复连续多轮 support measurement 太窄的问题。本阶段不是可选诊断，而是 v9.2.60 的资格门之一。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
horizons = 20,80,240,640
signal_strata = S1-S8
carriers = A0,A3,A6
row_sources = natural, balanced_diagnostic
minimum natural real events = 20000
minimum balanced diagnostic events = 6000
```

Balanced diagnostic rows 必须是真实 train-stream event rows。禁止复制自然 rows 冒充 balanced rows。

### 必须记录

```text
row_source
row_id
dataset
seed
horizon
signal_stratum
event_family
carrier_id
safe_good
bad_event
oracle_accept
controller_accept
gap_score
risk_mode_scores
support_family_ids
support_density
family_reliability
duplicate_row_flag
source_hash
```

### 判断标准

Support measurement pass：

```text
natural_real_event_count >= 20000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 6
measured_family_count >= 16
duplicate_row_count = 0
```

Official accepted support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

### 可视化

```text
p2_signal_strata_coverage.svg
p2_family_support_heatmap.svg
p2_natural_vs_balanced_distribution.svg
p2_bad_event_by_stratum_family.svg
```

---

## P3：risk-budget frontier

### 目标

把 mode-specific risk 从 hard union 改为 calibrated risk budget，测试能否在不爆 bad-event 的情况下扩张 coverage。

### 必须记录

```text
risk_budget_id
risk_modes_used
calibration_method
thresholds
lambdas
bad_probability_calibration_error
precision_cal
coverage_cal
bad_event_cal
precision_heldout
coverage_heldout
bad_event_heldout
coverage_gain_vs_union
bad_event_delta_vs_union
dataset_name_used
validation_used
test_used
posthoc_used_at_commit
```

### 判断标准

Risk-budget pass：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\geq0.03,
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

Diagnostic pass：

$$
Coverage_{\text{heldout}}\geq0.01,
$$

$$
BadEventRate_{\text{heldout}}\leq0.08,
$$

and:

$$
Coverage_{\text{budget}} \geq 5 \cdot Coverage_{\text{hard-union}}.
$$

### 可视化

```text
p3_risk_budget_precision_coverage_bad_frontier.svg
p3_risk_mode_ablation.svg
p3_bad_probability_calibration.svg
p3_coverage_gain_vs_bad_event.svg
```

---

## P4：support-frontier expansion

### 目标

用 multi-resolution support / core-neighbor expansion / density-risk joint pocket 扩张 clean core。

### 必须记录

```text
support_frontier_id
family_resolution
family_definition
density_method
reliability_method
leave_family_out_method
core_neighbor_method
precision
coverage
bad_event
legal_oracle_jaccard
oracle_overlap
accepted_signal_strata_count
accepted_family_count
max_family_share
coverage_expansion_factor
bad_event_expansion_delta
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

Support-frontier pass：

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

Support balance：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

Diagnostic support pass：

$$
Jaccard_{\text{legal,oracle}}\geq0.60
$$

or:

$$
Coverage\geq0.01
$$

with:

$$
BadEventRate\leq0.08.
$$

### 可视化

```text
p4_support_frontier_surface.svg
p4_core_neighbor_expansion_graph.svg
p4_family_resolution_ablation.svg
p4_oracle_legal_overlap.svg
```

---

## P5：exact reference deployable frontier v4

### 目标

组合 P3/P4，使用 CBD0 exact reference signal 判断 deployable accept region 是否恢复。

### 必须记录

```text
controller_id
risk_budget_id
support_frontier_id
thresholds
calibration_split_id
heldout_split_id
precision_cal
coverage_cal
bad_event_cal
precision_heldout
coverage_heldout
bad_event_heldout
precision_lcb
bad_event_ucb
AUC_heldout
corr_heldout
accepted_strata_count
accepted_family_count
max_family_share
legal_oracle_jaccard
coverage_expansion_factor_vs_v9259
dataset_name_used
posthoc_used_at_commit
reference_only
```

### 判断标准

Exact reference deployable pass：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

Stricter confidence gate：

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05.
$$

Support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

If no controller passes, route must be:

```text
R8-CleanCoreNotExpandable
```

### 可视化

```text
p5_deployable_frontier_precision_coverage_bad.svg
p5_clean_core_to_frontier_ladder.svg
p5_threshold_surface.svg
p5_oracle_legal_gap_after_expansion.svg
```

---

## P6：true-delta compute v5 parallel lane

### 目标

继续推进 system path，但不让 compute lane 掩盖 decision feasibility。

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
p6_true_delta_v5_cost_signal_pareto.svg
p6_residual_subphase_after_support_frontier.svg
p6_compute_vs_decision_gate_ladder.svg
```

---

## P7：system-legal exact-signal controller

### 目标

只有 P5 reference deployable pass 与 P6 compute pass 同时成立时，建立 official controller。

### 必须记录

```text
controller_id
custom_delta_id
risk_budget_id
support_frontier_id
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

Official controller pass：

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
risk_budget_id
support_frontier_id
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
risk_budget_id = best P8 survivor
support_frontier_id = best P8 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledTrueBranchDelta, ShuffledRiskBudget, ShuffledSupportFrontier,
           ShuffledControlGain, ShuffledCandidateGate, ShuffledBranchRatio,
           ShuffledValueScore, ShuffledSignalChannel, FunctionalChannelShuffled,
           TailMaskShuffled, RoleScoreShuffled, DatasetRouteShuffled,
           EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
controller_id
custom_delta_id
risk_budget_id
support_frontier_id
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
ShuffledRiskBudget = fail
ShuffledSupportFrontier = fail
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
contract_audit_v9260.csv
p0_v9259_boundary_reproduction.csv
p1_clean_core_nearcore_autopsy.csv
p2_support_measurement_expansion.csv
p3_risk_budget_frontier.csv
p4_support_frontier_expansion.csv
p5_exact_reference_deployable_frontier_v4.csv
p6_true_delta_compute_v5_parallel_lane.csv
p7_system_legal_exact_signal_controller.csv
p8_leave_dataset_and_stratum_out.csv
p9_official_paired_replay.csv
p10_short_run_functional_validation.csv
clean_core_trace_v9260.csv
nearcore_expansion_trace_v9260.csv
risk_budget_trace_v9260.csv
support_frontier_trace_v9260.csv
reference_deployable_frontier_trace_v9260.csv
true_delta_compute_v5_trace.csv
support_density_expansion_trace_v9260.csv
leaveout_trace_v9260.csv
paired_replay_branch_trace_v9260.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9259_boundary_unstable
F3_dataset_tuning_detected
F4_clean_core_not_stable
F5_nearcore_not_oracle_dense
F6_risk_budget_bad_event_fail
F7_risk_budget_coverage_fail
F8_support_measurement_too_narrow
F9_support_frontier_oracle_overlap_fail
F10_support_frontier_bad_event_fail
F11_exact_reference_deployable_frontier_fail
F12_true_delta_compute_still_expensive
F13_true_delta_signal_lost
F14_system_controller_precision_fail
F15_system_controller_coverage_fail
F16_system_controller_bad_event_fail
F17_leave_dataset_out_fail
F18_leave_stratum_out_fail
F19_paired_replay_control_equivalent
F20_shuffle_control_pass
F21_functional_lr_equivalent
F22_short_run_task_drop
F23_full_run_no_macro_hard_stratum_gain
F24_strong_baseline_explains_gain
F25_robustness_fail
F26_external_not_ready
F27_fake_or_proxy_violation
F28_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-CleanCoreStable:
  v9.2.59 clean core is stable and safe.

R2-NearCoreExpandable:
  near-core rows have meaningful oracle overlap and manageable bad-event.

R3-SupportMeasurementExpanded:
  support measurement reaches >=6 strata and >=16 families.

R4-RiskBudgetFrontierPass:
  calibrated risk budget expands coverage while keeping bad-event <=0.05.

R5-SupportFrontierPass:
  multi-resolution support frontier improves oracle/legal overlap and passes balance.

R6-ExactReferenceDeployableFrontierPass:
  CBD0 exact reference + risk/support frontier passes precision / coverage / bad-event gate.

R7-TrueDeltaComputeV5SystemPass:
  true branch-delta v5 passes system envelope while preserving signal.

R8-SystemLegalExactSignalControllerPass:
  system-legal true-delta controller passes heldout gate.

R9-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R10-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R11-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R12-CleanCoreNotExpandable:
  clean core cannot be expanded to coverage >=0.03 without violating safety.

R13-RiskBudgetStillTiny:
  risk-budget relaxation still yields tiny clean core.

R14-SupportFrontierUnsafe:
  support expansion raises coverage but bad-event remains too high.

R15-ReferenceFeasibleButComputeFail:
  decision geometry is viable but true-delta compute remains too expensive.

R16-ComputePassButReferenceFail:
  compute is system-legal but accept-region geometry fails.

R17-SupportMeasurementStillNarrow:
  support expansion remains insufficient; no official promotion.

R18-OracleSupportCollapse:
  fresh natural replay no longer has enough safe-good support.

R19-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R20-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R21-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9259_boundary_pass
dataset_tuning_detected
clean_core_stable
core_precision
core_coverage
core_bad_event
nearcore_expandable
nearcore_oracle_overlap
support_measurement_pass
natural_real_event_count
balanced_diagnostic_real_event_count
measured_signal_strata_count
measured_family_count
best_risk_budget_id
risk_budget_pass
risk_budget_precision
risk_budget_coverage
risk_budget_bad_event
best_support_frontier_id
support_frontier_pass
legal_oracle_jaccard
accepted_family_count
accepted_signal_strata_count
max_family_share
exact_reference_deployable
reference_controller_precision
reference_controller_coverage
reference_controller_bad_event
reference_precision_lcb
reference_bad_event_ucb
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
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9260_strict_purekan_functional
success_v9260_full_functional
success_v9260_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 clean-core / near-core autopsy
  P2 support measurement expansion
  P3 risk-budget frontier
  P4 support-frontier expansion
  P5 exact reference deployable frontier
  P6 true-delta compute v5

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
P5 reference deployable frontier can pass diagnostically before compute pass.
P7 official controller cannot pass unless P5 reference deployable frontier and P6 compute pass.
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
v9.2.59 boundary reproduced
clean core autopsied
support measurement expanded or explicitly failed
risk-budget frontier measured
support-frontier expansion measured
exact reference deployable frontier measured
true-delta compute v5 measured
no fake/proxy/offload/loss/teacher violation
```

## Decision geometry success

```text
Minimum diagnostic success
+
exact reference deployable frontier pass
+
risk/support frontier pass
+
multi-stratum / multi-family support pass
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
1. v9.2.59 boundary cannot be reproduced；
2. clean core is not stable；
3. near-core region has no oracle overlap；
4. risk-budget expansion cannot raise coverage；
5. risk-budget expansion raises bad-event above gate；
6. support measurement remains too narrow；
7. support frontier cannot improve oracle/legal overlap；
8. exact reference deployable frontier still fails；
9. true branch-delta compute remains too expensive；
10. true branch-delta compute passes system but loses signal；
11. system-legal controller cannot meet precision / coverage / bad-event；
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

## Case A：clean core expands + compute + controller + LDO/LSO + paired replay pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation and external robustness。

## Case B：reference deployable frontier pass but compute fail

必须声明：

```text
decision geometry is viable, but true branch-delta system implementation remains blocker.
```

下一步继续 kernel / layout / fused risk-support compute，不调 dataset。

## Case C：compute pass but reference frontier fail

必须声明：

```text
value is observable and cheap, but accept/abstain geometry is not deployable.
```

下一步继续修 risk/support/frontier，不走 kernel-only。

## Case D：clean core not expandable

必须声明：

```text
current exact branch-delta score can produce a tiny clean core but not a deployable frontier.
```

下一步应重设 output-delta sufficient statistics or safe-good factor semantics。

## Case E：support expansion fixes controller variance

可以声明：

```text
previous tiny-core behavior was partly narrow-support artifact.
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

v9.2.60 的一句话策略是：

$$
\boxed{
\text{不要再追求更硬的 clean core；从 clean core 出发做 constraint-calibrated expansion。}
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
1. v9.2.59 的 clean core 是否是可扩张的 frontier seed？
2. near-core rows 是否有足够 oracle overlap？
3. mode-specific risk 能否从 hard veto 改成 calibrated budget？
4. multi-resolution family support 能否提高 legal-oracle overlap？
5. support measurement 是否能真正扩到 >=6 strata / >=16 families？
6. exact reference coverage 能否从 0.003286 拉到 >=0.03？
7. true delta compute v5 能否继续压到 step <=1.50？
8. system-legal controller 能否 LDO/LSO？
9. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.60 的结果将给出清晰分叉：

```text
if clean-core expansion restores reference feasibility and true-delta compute passes:
  open system-legal controller, LDO/LSO, paired replay.

if reference feasible but compute fail:
  kernelization remains blocker.

if compute pass but reference frontier fail:
  risk/support/frontier geometry remains blocker.

if clean core cannot expand:
  exact branch-delta remains diagnostic; reset output-delta sufficient statistics.

if oracle support collapses:
  carrier/support stability is blocker.
```
