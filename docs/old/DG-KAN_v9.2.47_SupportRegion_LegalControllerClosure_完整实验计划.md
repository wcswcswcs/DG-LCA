# DG-KAN v9.2.47 Support-Region Legal Controller Closure 与 Fresh Online MicroProbe Paired Replay 完整实验计划

> 本计划基于 v9.2.46 `Boundary-Probe Legal Sufficient Statistics` 的真实复盘制定。  
> v9.2.46 的 terminal route 是：
>
> ```text
> route = R3-LegalFeatureSafeGoodPass
> base_candidate = LQ-t2-h256
> success_v9246_strict_purekan_functional = False
> success_v9246_full_functional = False
> success_v9246_external_ready = False
> ```
>
> v9.2.46 的关键事实是：
>
> ```text
> P1 boundary-probe disagreement expansion pass = 1
>   FP = 570
>   FN = 495
>   FP/FN attribution pass = 1
>
> P2 target regrounding pass = 1
>   safe-good label reliability = 0.983073
>
> P3 legal sufficient statistics pass = 1
>   best legal safe-good feature = LF8-BoundaryProbeLegalStats
>   safe-good AUC = 0.736749
>
> P3 component gate pass = 1
>   risk/value best = LF2-RiskTailLCB
>   gap best = LF8-BoundaryProbeLegalStats
>
> P4 factorized controller pass = 0
>   best = C6-HybridLegalMonotoneV2
>   heldout precision = 0.500000
>   coverage = 0.001302
>   bad-event = 0.500000
>
> current blocker = factorized_controller_failed_heldout_gate
> ```
>
> 这说明 v9.2.46 已经把问题从 “没有 legal sufficient statistics” 推进到 “有 legal predictive signal，但 accept/abstain controller 不能部署”。  
> 当前不能再继续修 base、attach、carrier，也不能继续只调 S7/C1/C6 的阈值。下一步必须重构 controller：从 **score ranking** 转为 **support-region decision**，并用 fresh online microprobe 验证这些 feature 是否是真正 commit-time legal，而不是 source-measured event-time diagnostic。
>
> v9.2.47 的核心目标是：
>
> $$
> \boxed{
> \text{把 LF8/LF2 的 legal predictivity 转化为可部署的 support-region controller，并关闭 LDO/LSO 与 paired replay。}
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

数据集只能作为诊断切片，不能作为 official controller 条件：

```text
allowed:
  report MNIST / Fashion-MNIST / KMNIST slice metrics
  report FP/FN distribution by dataset
  report leave-dataset-out generalization
  report signal-stratum / family composition by dataset
  report per-dataset diagnostic plots

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use primitive B
  if dataset == MNIST: abstain
  tune threshold separately per dataset
  promote a controller because it rescues only one dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

---

# Part I. 对 v9.2.46 的独立判断

## 1. v9.2.46 没有达到目标

v9.2.46 不能声明 strict PureKAN functional success。虽然 P1/P2/P3 都有实质推进，但 P4 factorized controller 没过：

```text
best controller = C6-HybridLegalMonotoneV2
heldout precision = 0.500000
coverage = 0.001302
bad-event = 0.500000
```

这三个数同时失败，说明不是 “差一点 calibration”：

```text
precision 不够；
coverage 极低；
bad-event 极高。
```

因此 P5 leave-out、P6 paired replay、P7 short-run、P8 full-run、P9 robustness 都正确保持 not_run。不能把 P3 feature success 包装成 causal success。

## 2. v9.2.46 的真实进展

v9.2.46 的真实推进有三点。

第一，v9.2.45 的 P1 blocker 被推进。v9.2.45 中 false-positive rows 太少且 attribution 不够，导致 `legal_feature_gap_unattributed`。v9.2.46 用 boundary-probe disagreement expansion 把 FP/FN 扩到：

```text
FP = 570
FN = 495
FP/FN attribution pass = 1
```

这说明 boundary 诊断不再是小样本猜测。

第二，safe-good target 被重新落地，而且 label reliability 很高：

$$
Reliability(Y_{\text{safe-good}})=0.983073.
$$

这说明现在的 supervised diagnostic target 本身不是主要噪声源。

第三，legal sufficient statistics 不是空的。`LF8-BoundaryProbeLegalStats` 对 safe-good 的 AUC 达到：

$$
AUC_{\text{safe-good}}=0.736749.
$$

同时 component gate 也通过，risk/value 由 `LF2-RiskTailLCB` 最好，control-gap 由 `LF8` 最好。这说明我们已经不是 “没有可观测 signal”，而是 “signal 没有转化为可部署 decision region”。

## 3. 当前真正 blocker

当前 blocker 是：

$$
\boxed{
\text{ranking-level feature 存在，但 support-region controller 不成立。}
}
$$

也就是说，LF8 能区分 safe-good 的排序趋势，但 P4 的 accept/abstain 规则无法找到一块同时满足：

```text
precision high
coverage usable
bad-event low
multi-family / multi-stratum non-collapse
```

的区域。

这通常有四类原因：

```text
1. Score-to-policy cliff:
   AUC 高，但 top-k precision / coverage / risk 曲线没有稳定可部署 plateau。

2. Risk / value / gap conflict:
   安全事件、正收益事件、control-resistant 事件不在同一 feature region 内。

3. Boundary-probe diagnostic leakage risk:
   LF8 来自 source-measured event-time statistics，不是新生成的 train-stream microprobe。
   因此它可能是合法特征的候选，但还不是完全 official online controller。

4. Support sparsity:
   safe-good support 太薄，factorized intersection 后 coverage 被压到 0.001302。
```

因此 v9.2.47 不能继续 “微调一个 threshold”。必须重新定义 controller 为 support-region problem，并新增真实 online microprobe rows。

## 4. 为什么仍然感觉进展慢

进展慢是事实，但 v9.2.46 的慢不是绕圈。路线已经从：

```text
base / attach / carrier 是否可用
```

推进到：

```text
legal feature 能否形成 deployable controller。
```

这已经是 functional update 的核心难点。现在要加快，不是降低标准，而是把实验并行化：

```text
同轮完成：
  boundary autopsy
  fresh online microprobe
  controller family
  leave-out
  paired replay scout
```

不能再做：

```text
v9.2.47 只测一个新 controller
v9.2.48 再测 microprobe
v9.2.49 再测 paired replay
```

## 5. 是否在正确道路上

是，但路线必须继续收紧。

正确路线：

```text
legal feature signal exists
→ decision region reconstruction
→ fresh online commit-time verification
→ LDO/LSO
→ paired replay
```

错误路线：

```text
继续修 base；
继续修 snapshot attach；
继续问 carrier 是否 silent；
继续手调 S7/C6 threshold；
按 dataset 写 controller；
把 P3 AUC 写成 functional success。
```

v9.2.47 必须回答一个更尖锐的问题：

$$
\boxed{
\text{存在一个不使用 dataset name 的 train-stream legal decision region 吗？}
}
$$

如果存在，就进入 paired replay；如果不存在，就说明当前 legal sufficient statistics 仍不够，需要更强的 train-stream probe 或更深的 carrier mechanism。

---

# Part II. v9.2.47 总体目标

v9.2.47 的总体目标是：

$$
\boxed{
\text{用 fresh online microprobe 与 support-region controller，把 LF8/LF2 的 predictivity 转化为 paired replay causal evidence。}
}
$$

目标分为七层。

## 1. Boundary reproduction success

必须复现 v9.2.46 boundary：

```text
route = R3-LegalFeatureSafeGoodPass
P1 disagreement expansion pass = 1
P2 target regrounding pass = 1
P3 legal sufficient statistics pass = 1
P4 factorized controller pass = 0
```

## 2. Decision-region autopsy success

必须解释为什么 P3 feature AUC 过，但 P4 controller 崩：

```text
heldout precision = 0.5
coverage = 0.001302
bad-event = 0.5
```

至少归因到以下机制之一：

```text
D1-score_to_policy_cliff:
  AUC 高，但 precision/coverage/bad-event 曲线没有 deployable plateau。

D2-risk_value_gap_conflict:
  risk-safe、value-positive、control-resistant 三个区域相互冲突。

D3-support_sparsity:
  factorized intersection 后 coverage 被压到不可用。

D4-family_concentration:
  accepted events 被少数 family 主导，heldout 崩。

D5-boundaryprobe_not_online_legal:
  LF8 依赖 source-measured event-time statistic，无法作为 commit-time online controller。

D6-threshold_crossfit_miscalibration:
  calibration split 上可行，heldout split 上阈值错位。

D7-label_support_shift:
  boundary-probe rows 与 fresh replay rows 的 safe-good prevalence 不一致。
```

成功标准：

```text
>= 90% P4 false positives attributed
>= 90% P4 false negatives attributed
>= 90% coverage-loss rows attributed
dataset_name_used = 0
```

## 3. Fresh online microprobe success

必须新增真正 commit-time train-stream microprobe，而不是只从 source-measured event rows构造 feature。

Microprobe 必须只使用 train split：

```text
update batch:
  forms candidate functional update

probe batch:
  estimates risk/value/gap proxy on train-stream data
```

Probe score：

$$
S_{\text{probe}}
=
-\Delta CE_{\text{probe}}
+
\lambda_m\Delta Margin_{\text{probe}}
-
\lambda_r Risk_{\text{probe}}
+
\lambda_g GapProxy_{\text{probe}}.
$$

Legality：

```text
uses_dataset_name = 0
uses_validation = 0
uses_test = 0
uses_posthoc_replay = 0
```

System gate：

$$
Overhead_{\text{probe}}\leq0.20.
$$

Predictivity gate：

$$
AUC(S_{\text{probe}},Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(S_{\text{probe}},V_{\text{safe-grounded}})\geq0.35.
$$

If overhead exceeds gate, feature remains diagnostic only.

## 4. Support-region controller success

Controller 不再是单一 score threshold，而是 support-region decision：

$$
Accept(e)
=
RiskSafe(e)
\land
ValuePositive(e)
\land
ControlResistant(e)
\land
SupportStable(e).
$$

其中：

$$
RiskSafe(e)=\mathbb{1}[S_{\text{risk}}(e)\leq \rho],
$$

$$
ValuePositive(e)=\mathbb{1}[S_{\text{value}}(e)\geq \tau_v],
$$

$$
ControlResistant(e)=\mathbb{1}[S_{\text{gap}}(e)\geq \tau_g],
$$

$$
SupportStable(e)=\mathbb{1}[Density_{\text{support}}(e)\geq \tau_s].
$$

Heldout gate：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

Multi-support gate：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

System gate：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

## 5. Leave-out success

Leave-dataset-out：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two held-out datasets satisfy：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

Leave-stratum-out：

At least $70\%$ held-out strata task-safe and：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

## 6. Official paired replay success

Paired replay pass：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{Real,slice}}\geq Acc_{\text{AdamW,slice}}-0.005.
$$

Shuffle controls must fail：

```text
ShuffledRiskScore = fail
ShuffledBranchRatio = fail
ShuffledControlGap = fail
ShuffledValueScore = fail
ShuffledProbeScore = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

## 7. Short-run scout success

Only after paired replay pass：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

At least one mechanism improvement：

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

# Part III. 核心假设

## H1：v9.2.46 的失败是 decision-region failure，不是 legal feature absence

证据：

$$
AUC_{\text{LF8,safe-good}}=0.736749.
$$

但 controller：

$$
Precision=0.5,\quad Coverage=0.001302,\quad BadEvent=0.5.
$$

H1 认为 feature ranking 有信息，但现有 factorized accept rule找不到稳定 support region。

H1 成立标准：

如果 P1 autopsy 显示 P4 fail 主要来自 score-to-policy cliff、support sparsity 或 threshold miscalibration，而 LF8/LF2 feature ablation 仍有 AUC signal，则 H1 成立。

H1 失败标准：

如果 fresh online rows 上 LF8/LF2 AUC 下降到：

$$
AUC<0.60,
$$

则 v9.2.46 P3 只是 source-diagnostic artifact。

## H2：fresh online microprobe 能提供缺失的 population-risk signal

Boundary-probe LF8 是从 source-measured event-time statistics 构造的。H2 认为缺少的是 commit-time population-risk probe。

H2 成立标准：

$$
AUC(S_{\text{probe}},Y_{\text{safe-good}})\geq0.70
$$

or controller with probe satisfies heldout precision/coverage/bad-event gate.

H2 失败标准：

microprobe 无 predictivity或 overhead：

$$
Overhead_{\text{probe}}>0.20.
$$

## H3：support density / family reliability 是 coverage collapse 的关键

P4 coverage 只有 $0.001302$，说明 intersection 太窄。H3 认为需要 SupportStable / FamilyReliable gating，而不是进一步收紧 threshold。

H3 成立标准：

support-region controller achieves：

$$
Coverage\geq0.03
$$

without bad-event exceeding $0.05$.

## H4：risk/value/gap 三因子必须保留，但阈值要 cross-fit

单一 score 混合会导致 risk/value/gap 冲突；但是 naive factorized controller 过于稀疏。H4 认为 cross-fitted constrained thresholds 可以恢复 support。

H4 成立标准：

cross-fitted controller heldout pass, and shuffle controls fail.

## H5：如果 support-region controller 仍失败，但 oracle support 高，则 legal feature representation 仍不足

如果：

```text
oracle support pass = 1
all legal controllers fail
```

则不是 carrier 问题，下一步应设计 richer legal train-stream statistics。

## H6：如果 oracle support 也崩，则回到 carrier/support reset

如果 fresh online replay 中：

$$
OraclePrecision<0.75
$$

or：

$$
OracleBadEvent>0.05,
$$

则 v9.2.46 的 feature success不足以证明 current carrier可用，下一步回到 support/carrier design。

## H7：dataset tuning 禁止

即使某个 dataset 上 controller pass，也不能 official promote。必须 LDO/LSO。

---

# Part IV. 并行执行设计

v9.2.47 必须并行执行：

```text
Lane A:
  v9.2.46 boundary reproduction

Lane B:
  decision-region autopsy:
    score-to-policy curve
    risk/value/gap conflict
    support sparsity
    family concentration
    threshold crossfit drift

Lane C:
  fresh online microprobe:
    update/probe train split
    overhead measurement
    probe feature predictivity

Lane D:
  support-region controller family:
    constrained PR
    risk/value/gap factorized
    support density
    family reliability
    probe-gated controller
    Pareto-front accept region

Lane E:
  fresh natural replay rows:
    official evaluation distribution

Lane F:
  boundary-targeted diagnostic rows:
    FP/FN expansion only for autopsy, not official pass

Lane G:
  LDO/LSO

Lane H:
  paired replay scout and official replay

Lane I:
  short-run scout if paired replay passes
```

Gate discipline：

```text
diagnostic rows may be measured early
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  legal controller pass
  LDO/LSO pass
```

---

# Part V. Candidate features and controllers

## 1. Base and carrier

Official base remains：

```text
R2-LQ-fanin-output-scale-confirmed
```

Carrier candidates：

```text
A0-current-v9246
A3-RoleWiseFT7ResetCarrier
A6-HybridRoleControlRiskCarrier
```

A1-like risk-bounded carrier remains diagnostic unless carrier-active gate passes：

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

## 2. Legal feature groups

### LF0：Existing LF8 / LF2 reference

Reference only：

```text
LF8-BoundaryProbeLegalStats
LF2-RiskTailLCB
```

### LF1：Online branch-ratio decomposition

```text
branch_ratio_mean
branch_ratio_p10
branch_ratio_p90
branch_ratio_tail
effective_derivative
tail_activation_mass
functional_step_norm
task_step_norm
```

### LF2：Risk LCB v2

$$
S_{\text{risk}}
=
w_1CEp99
+
w_2WrongConfidenceP95
-
w_3MarginP10
+
w_4Uncertainty
+
w_5Curvature
+
w_6BranchRisk
+
w_7ProbeRisk.
$$

### LF3：Control-gap LCB v2

$$
S_{\text{gap}}
=
LCB(Gain_{\text{Real}})
-
UCB(\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})).
$$

### LF4：Value-positive probe

$$
S_{\text{value}}
=
-\Delta \widehat{CEp99}_{\text{probe}}
+
\lambda_m \Delta \widehat{MarginP10}_{\text{probe}}
-
\lambda_c \Delta \widehat{CurvatureRisk}_{\text{probe}}.
$$

### LF5：Support density / family reliability

Family must not include dataset name：

$$
family(e)
=
(stratum,horizon,risk\_bucket,role\_bucket,attach\_type,branch\_bucket).
$$

Reliability：

$$
Rel(f)
=
\mathbb{E}[Y_{\text{safe-good}}\mid f]
-
\kappa\sqrt{\operatorname{Var}(Y_{\text{safe-good}}\mid f)}.
$$

Density：

$$
Density(e)=\frac{k}{N\cdot Volume(\mathcal{N}_k(e))}.
$$

### LF6：Train-stream microprobe

Use train split only：

```text
update batch:
  compute candidate update

probe batch:
  estimate risk/value/gap proxy
```

Feature：

$$
S_{\text{probe}}
=
-\Delta CE_{\text{probe}}
+
\lambda_m\Delta Margin_{\text{probe}}
-
\lambda_r Risk_{\text{probe}}
+
\lambda_g GapProxy_{\text{probe}}.
$$

### LF7：Decision-margin stability

Measures how stable acceptance is under train-stream perturbations：

$$
Stability(e)=
1-
\operatorname{Var}_{a\in \mathcal{A}}
[
\mathbb{1}(Accept_a(e))
].
$$

---

## 3. Controller candidates

### C0：Current C6 reference

Reference only. Expected fail.

### C1：Constrained Precision-Coverage Controller

Threshold selected on calibration split:

$$
\max_\tau Coverage_{\text{cal}}(\tau)
$$

subject to：

$$
Precision_{\text{cal}}(\tau)\geq0.80,
$$

$$
BadEvent_{\text{cal}}(\tau)\leq0.03.
$$

Official evaluation only on heldout.

### C2：RiskValueGapTriStageV2

$$
Accept(e)
=
RiskSafe(e)
\land
ValuePositive(e)
\land
ControlResistant(e).
$$

### C3：ProbeGatedTriStage

$$
Accept(e)
=
RiskSafe(e)
\land
\mathbb{1}[S_{\text{probe}}\geq\tau_p]
\land
ControlResistant(e).
$$

### C4：SupportDensityFamilyBalanced

$$
Accept(e)
=
RiskSafe(e)
\land
ValuePositive(e)
\land
ControlResistant(e)
\land
Rel(family(e))\geq r_0
\land
Density(e)\geq d_0
\land
FamilyBalance(e).
$$

### C5：ParetoFrontController

Accepts only if event lies on legal Pareto frontier of:

```text
low risk
positive value
positive control gap
stable support
low overhead
```

No dataset name.

### C6：Abstain-First Conservative Controller

A deliberately conservative controller for paired replay scout:

$$
Accept(e)=
\mathbb{1}[
LCB(Value)>0
\land
LCB(Gap)>0
\land
UCB(Risk)<\rho
].
$$

### C7：Oracle

Posthoc diagnostic only. Never official.

---

# Part VI. 实验阶段

## P0：v9.2.46 boundary reproduction

### 目标

确认 v9.2.46 boundary 稳定。

### 必须记录

```text
route
source_route_v9245
fp_count
fn_count
fp_attribution_pass
fn_attribution_pass
safe_good_reliability
best_legal_feature
safe_good_auc
component_gate_pass
best_controller
controller_precision
controller_coverage
controller_bad_event
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R3-LegalFeatureSafeGoodPass
P3 legal feature pass = 1
P4 factorized controller pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_feature_pass_controller_fail_ladder.svg
```

---

## P1：Decision-region autopsy

### 目标

解释 P4 为什么崩，尤其是 precision/coverage/bad-event 三者同时失败。

### 必须记录

```text
row_id
score_LF8
score_LF2
risk_score
value_score
gap_score
probe_score_if_available
accepted_by_C6
safe_good
risk_safe
value_positive
control_resistant
bad_event
event_family
signal_stratum
support_density
family_reliability
false_positive_type
false_negative_type
coverage_loss_type
failure_mode
```

### 判断标准

P1 pass：

```text
FP attribution fraction >= 0.90
FN attribution fraction >= 0.90
coverage-loss attribution fraction >= 0.90
dataset_name_used = 0
```

### 可视化

```text
p1_precision_coverage_bad_curve.svg
p1_risk_value_gap_ternary.svg
p1_support_density_vs_accept.svg
p1_family_concentration.svg
p1_failure_mode_sankey.svg
```

---

## P2：Fresh online microprobe implementation

### 目标

实现真正 commit-time legal microprobe，而不是复用 source-measured event-time rows。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..7
horizons = 20,80,240,640
signal_strata = S1..S8
carriers = A0,A3,A6
probe_split = update_half / probe_half
```

### 必须记录

```text
row_id
carrier_id
dataset
seed
horizon
signal_stratum
update_batch_hash
probe_batch_hash
S_probe
delta_CE_probe
delta_margin_probe
risk_probe
gap_proxy_probe
probe_overhead
memory_overhead
uses_dataset_name
uses_validation
uses_test
uses_posthoc
AUC_to_safe_good
corr_to_safe_grounded_value
```

### 判断标准

Legal probe pass：

```text
uses_dataset_name = 0
uses_validation = 0
uses_test = 0
uses_posthoc = 0
```

Predictivity pass：

$$
AUC(S_{\text{probe}},Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(S_{\text{probe}},V_{\text{safe-grounded}})\geq0.35.
$$

System pass：

$$
Overhead_{\text{probe}}\leq0.20.
$$

### 可视化

```text
p2_probe_score_vs_safe_good.svg
p2_probe_overhead_pareto.svg
p2_probe_legality_matrix.svg
```

---

## P3：Fresh natural replay and boundary diagnostic expansion

### 目标

同时生成 official natural rows 与 boundary-targeted diagnostic rows。  
Natural rows 用于 official evaluation；boundary rows 只用于 autopsy。

### 必须记录

```text
row_source = natural | boundary_targeted
row_id
carrier_id
dataset
seed
horizon
signal_stratum
event_family
branch
real_gain
adamwparallel_gain
bestlr_gain
control_gap
bad_event
task_safe
safe_good
risk_safe
value_positive
control_resistant
r_z_tail
r_perp_tail
step_q90
memory_ratio
```

### 判断标准

Fresh natural pass：

```text
natural_real_event_count >= 2000
measured_signal_strata_count >= 6
carrier_active = 1
fake/proxy/offload = 0
```

Boundary diagnostic pass：

```text
FP/FN/coverage-loss rows sufficient for P1
```

### 可视化

```text
p3_natural_vs_boundary_distribution.svg
p3_safe_good_prevalence_by_stratum.svg
p3_control_gap_by_family.svg
```

---

## P4：Support-region controller calibration

### 目标

并行测试 C1-C7，并把 official pass 限制在 heldout natural rows 上。

### Split design

```text
calibration split:
  choose thresholds / monotone coefficients

heldout split:
  official controller gate

boundary diagnostic split:
  not official

leave-dataset-out:
  P5

leave-stratum-out:
  P5
```

### 必须记录

```text
controller_id
features_used
thresholds
coefficients
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
feature_overhead
step_q90
memory_ratio
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
official_eligible
```

### 判断标准

Heldout controller pass：

$$
AUC_{\text{heldout}}\geq0.70
$$

or:

$$
Corr_{\text{heldout}}\geq0.35.
$$

and：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

Multi-family：

```text
accepted_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
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
p4_controller_precision_coverage_bad.svg
p4_controller_auc_corr.svg
p4_controller_family_coverage.svg
p4_controller_factor_ablation.svg
p4_oracle_legal_gap.svg
```

---

## P5：Leave-dataset-out / leave-stratum-out

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
carrier_id
controller_id
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
p5_leave_dataset_out_matrix.svg
p5_leave_stratum_out_matrix.svg
p5_hidden_dataset_tuning_audit.svg
p5_leaveout_failure_modes.svg
```

---

## P6：Official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
carrier_id = best P5 survivor
controller_id = best P5 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRiskScore, ShuffledBranchRatio, ShuffledControlGap, ShuffledValueScore, ShuffledProbeScore, FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled, DatasetRouteShuffled, EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
carrier_id
controller_id
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
ShuffledRiskScore = fail
ShuffledBranchRatio = fail
ShuffledControlGap = fail
ShuffledValueScore = fail
ShuffledProbeScore = fail
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
p6_official_paired_replay_pareto.svg
p6_macro_beat_rate.svg
p6_signal_stratum_win_matrix.svg
p6_shuffle_control_matrix.svg
p6_system_gate_distribution.svg
```

---

## P7：Short-run scout

### 目标

如果 P6 pass，验证局部 paired replay 优势能否在连续训练中保持。

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRiskScore, ShuffledProbeScore
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

### 可视化

```text
p7_short_run_task_mechanism_pareto.svg
p7_short_run_controls.svg
p7_event_timeline.svg
p7_ce_tail_margin_panel.svg
```

---

## P8：Full 10-seed validation

### 目标

如果 short-run pass，验证 full functional route。

### 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRiskScore, ShuffledProbeScore
```

### 必须记录

```text
candidate
dataset
seed
val_acc
test_acc
delta_vs_mlp
delta_vs_adamw
delta_vs_adamwparallel
delta_vs_bestlr
delta_vs_quadratic_feature_mlp
near_pass
full_pass
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature_ratio
local_lipschitz_ratio
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
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Macro improvement：

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003.
$$

Hard-stratum repair：

$$
\Delta Acc_{\text{hard-stratum,functional}}
-
\Delta Acc_{\text{hard-stratum,AdamW}}
\geq0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

---

## P9：Robustness / strong baseline / external-ready

### 目标

确认 functional advantage 不是 clean MNIST-family artifact。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
strong baselines = MLP-match, hidden-bracket MLP, QuadraticFeatureMLP, repaired LQ AdamW-only
```

### 判断标准

Robustness pass：

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{AdamW}}.
$$

Noise-tail pass：

$$
CEp99_{\text{functional}}\leq CEp99_{\text{AdamW}}.
$$

Strong baseline pass：

$$
Acc_{\text{functional}}\geq Acc_{\text{QuadraticFeatureMLP}}-0.005
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{QuadraticFeatureMLP}}.
$$

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9247.csv
p0_v9246_boundary_reproduction.csv
p1_decision_region_autopsy.csv
p2_fresh_online_microprobe.csv
p3_fresh_natural_and_boundary_replay.csv
p4_support_region_controller_calibration.csv
p5_leave_dataset_and_stratum_out.csv
p6_official_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_robustness_external_ready.csv
decision_region_trace_v9247.csv
online_microprobe_trace_v9247.csv
support_density_trace_v9247.csv
controller_calibration_trace_v9247.csv
leaveout_trace_v9247.csv
paired_replay_branch_trace_v9247.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9246_boundary_unstable
F3_dataset_tuning_detected
F4_decision_region_autopsy_fail
F5_microprobe_not_implemented
F6_microprobe_not_predictive
F7_microprobe_system_too_expensive
F8_fresh_natural_rows_insufficient
F9_support_region_controller_fail
F10_oracle_support_collapse
F11_legal_feature_offline_only
F12_leave_dataset_out_fail
F13_leave_stratum_out_fail
F14_paired_replay_control_equivalent
F15_shuffle_control_pass
F16_functional_lr_equivalent
F17_short_run_task_drop
F18_full_run_no_macro_hard_stratum_gain
F19_strong_baseline_explains_gain
F20_robustness_fail
F21_external_not_ready
F22_fake_or_proxy_violation
F23_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-DecisionRegionFailureAttributed:
  P4 precision/coverage/bad-event collapse is explained.

R2-OnlineMicroProbePass:
  train-stream online microprobe is legal, predictive, and system-valid.

R3-SupportRegionControllerPass:
  support-region controller passes heldout precision / coverage / bad-event gate.

R4-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R5-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R6-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R7-OfflineFeatureOnly:
  LF8 works only as boundary-probe diagnostic, not as online commit-time feature.

R8-OracleSupportCollapse:
  fresh natural replay shows no sufficient safe-good oracle support.

R9-MicroProbeSystemTooExpensive:
  microprobe predicts value but violates system overhead.

R10-ControllerStillInvalid:
  legal features exist, but no deployable support-region controller passes.

R11-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R12-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R13-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9246_boundary_pass
dataset_tuning_detected
decision_region_autopsy_pass
failure_mode
online_microprobe_implemented
online_microprobe_pass
microprobe_auc
microprobe_corr
microprobe_overhead
fresh_natural_row_count
oracle_support_pass
best_controller_id
support_region_controller_pass
controller_auc
controller_corr
accepted_precision
accepted_coverage
accepted_bad_event_rate
accepted_strata_count
accepted_family_count
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9247_strict_purekan_functional
success_v9247_full_functional
success_v9247_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 decision-region autopsy
  P2 fresh online microprobe implementation
  P3 fresh natural / boundary replay generation

Batch 2:
  P4 support-region controller calibration
  P5 leave-dataset-out / leave-stratum-out
  P6 paired replay scout

Batch 3:
  official P6 paired replay
  P7 short-run if paired replay passes

Batch 4:
  P8 full 10-seed
  P9 robustness / strong baseline
```

Gate rule：

```text
P4/P6 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  legal online feature pass
  controller pass
  LDO/LSO pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.46 boundary reproduced
P4 decision-region failure attributed
fresh online microprobe implemented or explicitly ruled out
fresh natural replay measured
no fake/proxy/offload/loss/teacher violation
```

## Legal controller success

```text
Minimum diagnostic success
+
at least one online legal feature predicts safe-good
+
support-region controller passes heldout precision / coverage / bad-event gate
+
system overhead gate pass
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
1. v9.2.46 boundary cannot be reproduced；
2. P4 failure cannot be attributed；
3. fresh online microprobe cannot be implemented；
4. microprobe and all online legal features fail predictivity；
5. only predictive feature is too expensive；
6. fresh natural replay oracle support collapses；
7. all support-region controllers fail heldout gate；
8. leave-dataset-out fails；
9. leave-stratum-out fails；
10. paired replay remains control-equivalent；
11. shuffle controls pass, indicating overfit；
12. short-run task drops；
13. full run gives no macro / hard-stratum / geometry gain；
14. functional breaks system gate；
15. gains are explained by QuadraticFeatureMLP；
16. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：support-region controller passes

可以声明：

```text
v9.2.46 failed because ranking-level feature was not converted into a deployable support decision; v9.2.47 closes controller gate.
```

但不能声明 strict functional success unless LDO/LSO and paired replay pass.

## Case B：microprobe passes but controller fails

必须声明：

```text
safe-good is observable online, but accept/abstain policy remains undeployable.
```

下一步修 policy calibration / support geometry，不修 base/attach.

## Case C：microprobe works but overhead too high

必须声明：

```text
value is observable but not system-legal.
```

下一步做 probe amortization / kernelization。

## Case D：LF8 works only offline

必须声明：

```text
boundary-probe statistic was diagnostic, not deployable online.
```

下一步设计 true train-stream sufficient statistics。

## Case E：fresh natural oracle collapses

必须声明：

```text
carrier/event family does not provide robust safe-good support under natural fresh replay.
```

下一步回到 carrier/support reset。

## Case F：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

## Case G：paired replay passes

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation。

---

# Part XII. 最终建议

v9.2.47 的一句话策略是：

$$
\boxed{
\text{不要再只看 AUC；把 legal feature 变成 online support-region controller，并直接冲 LDO/LSO + paired replay。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
LF8 有没有 signal；
C6 threshold 是否差一点；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. 为什么 LF8 AUC 过而 C6 controller 崩？
2. risk/value/gap 三个方向是否在同一 support region？
3. boundary-probe LF8 是否能变成 true online microprobe feature？
4. support density / family reliability 是否能恢复 coverage 而不增加 bad-event？
5. controller 能否 LDO/LSO？
6. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.47 的结果会给出清晰分叉：

```text
if support-region controller + LDO/LSO + paired replay pass:
  strict PureKAN functional obtains local causal evidence.

if online feature works but controller fails:
  policy/support geometry is blocker.

if online feature is too expensive:
  feature amortization/kernelization is blocker.

if online feature fails but offline LF8 works:
  boundary-probe statistic is not deployable.

if fresh oracle collapses:
  carrier/support stability remains blocker.
```
