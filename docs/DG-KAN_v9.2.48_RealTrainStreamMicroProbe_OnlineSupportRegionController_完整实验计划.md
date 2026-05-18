# DG-KAN v9.2.48 Real Train-Stream MicroProbe 与 Online Support-Region Controller Closure 完整实验计划

> 本计划基于 v9.2.47 `Support-Region Legal Controller Closure` 的真实复盘制定。  
> v9.2.47 的 terminal route 是：
>
> ```text
> route = R7-OfflineFeatureOnly
> base_candidate = LQ-t2-h256
> success_v9247_strict_purekan_functional = False
> success_v9247_full_functional = False
> success_v9247_external_ready = False
> ```
>
> v9.2.47 的关键事实是：
>
> ```text
> P0:
>   v9.2.46 boundary reproduced
>   source route = R3-LegalFeatureSafeGoodPass
>   P1/P2/P3 pass
>   P4 factorized controller pass = 0
>
> P1:
>   decision-region autopsy pass = 1
>   heldout rows = 1536
>   accepted by C6 reference = 3
>   false positive = 3
>   false negative = 124
>   coverage-loss = 124
>   FP/FN/coverage-loss attribution fraction = 1.0/1.0/1.0
>   FP mode = D2-risk_value_gap_conflict
>   FN mode = D4-family_concentration
>   coverage-loss mode = D6-threshold_crossfit_miscalibration
>
> P2:
>   fresh online microprobe status = not_implemented
>   online_microprobe_implemented = 0
>   reason = no_fresh_train_stream_update_probe_rows_available
>
> P3:
>   source-measured natural/oracle support pass = 1
>   fresh natural rows = 3072
>   signal strata = 8
>   oracle precision = 1.0
>   oracle coverage = 0.12369791666666667
>
> P4:
>   support-region controller pass = 0
>   best legal controller = C1-ConstrainedPrecisionCoverage
>   controller AUC = 0.378213
>   controller corr = -0.044930
>   accepted precision = 0.157895
>   accepted coverage = 0.012370
>   accepted bad-event = 0.0
>
> Downstream:
>   P5 leave-out / P6 paired replay / P7 short-run / P8 full / P9 robustness = not_run
>
> current blocker:
>   fresh_online_microprobe_not_implemented
> ```
>
> 这轮最核心的结论不是 “controller 又失败了”，而是：
>
> $$
> \boxed{
> \text{v9.2.47 仍停留在 offline/source-measured diagnostic；真正的 online commit-time probe 没有实现。}
> }
> $$
>
> 因此 v9.2.48 不能继续在 v9.2.43-v9.2.46 的 source-measured event rows 上做更多后验 autopsy，也不能继续调 S7/C6/C1 threshold。  
> v9.2.48 的核心任务必须改为：
>
> $$
> \boxed{
> \text{实现真实 train-stream update/probe microprobe，并用它关闭 online legal controller、LDO/LSO 与 paired replay。}
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

# Part I. 对 v9.2.47 的独立判断

## 1. v9.2.47 没有达到目标

v9.2.47 不能声明 strict PureKAN functional success。原因不是 full-run 失败，而是更前面的 online controller gate 根本没有被真实实现：

```text
online_microprobe_implemented = 0
online_microprobe_pass = 0
microprobe_auc = 0
microprobe_corr = 0
```

由于 P2 没有真实 train-stream update/probe rows，P5-P9 正确保持 not_run。不能把 source-measured decision-region autopsy、oracle support 或 C7 oracle 写成 leave-out / paired replay / short-run / full-run success。

## 2. v9.2.47 的真实进展

v9.2.47 不是没有信息。它推进了两个重要边界。

第一，v9.2.46 的 controller failure 被解释得更清楚。P1 显示：

```text
FP mode = D2-risk_value_gap_conflict
FN mode = D4-family_concentration
coverage-loss mode = D6-threshold_crossfit_miscalibration
```

这意味着 v9.2.46 的失败不是一个单一 threshold 问题，而是三类结构问题叠加：

```text
risk/value/gap 不在同一区域；
safe-good family 分布过于集中；
cross-fit threshold 在 heldout 上失稳。
```

第二，P3 source-measured oracle support 仍然存在：

```text
oracle precision = 1.0
oracle coverage = 0.123698
```

这说明在 source-measured comparable rows 中，safe-good events 没有消失。也就是说，当前不是 “functional carrier 完全没有 good events”，而是 “offline 看到的 good events 还没有被 online legal feature 找到”。

## 3. 当前真正 blocker

当前真正 blocker 是：

$$
\boxed{
\text{online legal feature implementation gap}
}
$$

或者更具体：

$$
\boxed{
\text{我们缺少在 commit-time 用 train-stream 数据估计 RiskSafe / ValuePositive / ControlResistant / SignalChannel 的真实 microprobe。}
}
$$

v9.2.46 的 LF8 / boundary-probe feature 是 source-measured diagnostic；v9.2.47 的 P2 明确没有 fresh online microprobe rows。因此目前最关键的问题不是 “哪一个 offline score 更好”，而是：

```text
1. 能不能在真实训练 step 中形成 candidate functional update？
2. 能不能在同一 train-stream 的 probe half 上估计风险、收益、control gap？
3. 这些估计能不能在 commit 前形成 accept/abstain controller？
4. 这个 controller 能不能通过 heldout、LDO/LSO、paired replay？
```

## 4. 为什么进展仍显慢

进展慢是真的，而且 v9.2.47 暴露了一个流程问题：我们一直在 offline diagnostic rows 上推进解释能力，但真正计划要求的 online microprobe 还没有被实现。  
这不是科学路线错误，而是工程优先级偏了。v9.2.48 必须停止继续“解释已落盘 source rows”，转为实现真实 online runner。

v9.2.48 的实验节奏必须加快：同一 runner 中并行实现 microprobe、natural replay、support-region controller、LDO/LSO、paired replay scout，而不是本轮只实现 P2、下轮再做 P4、再下轮做 P5。

---

# Part II. v9.2.48 总体目标

v9.2.48 的总体目标是：

$$
\boxed{
\text{实现真实 online train-stream microprobe，并把 offline safe-good signal 转化为 official paired replay evidence。}
}
$$

目标分为八层。

## 1. Real online microprobe implementation success

必须在真实训练 step 中实现 update/probe split：

```text
train minibatch B_t
  split into:
    B_update
    B_probe

B_update:
  compute AdamW-equivalent task update
  compute candidate functional update
  compute branch features before commit

B_probe:
  evaluate candidate update effect on train-stream probe
  estimate risk/value/control-gap/signal-channel statistics
```

Microprobe 不能使用 validation / test / dataset name / posthoc replay result。

合法性：

```text
uses_dataset_name = 0
uses_validation = 0
uses_test = 0
uses_posthoc_replay = 0
uses_teacher = 0
uses_loss_modification = 0
```

实现成功标准：

```text
online_microprobe_implemented = 1
fresh_train_stream_update_probe_rows >= 2000
probe_batch_hash exists
update_batch_hash exists
candidate_update_hash exists
commit_decision_before_outcome = 1
```

## 2. Microprobe predictivity success

Microprobe 必须预测 safe-good，而不只是输出一个 diagnostic scalar。

定义：

$$
Y_{\text{risk-safe}}=\mathbb{1}[BadEvent=0],
$$

$$
Y_{\text{value-positive}}=\mathbb{1}[Gain_{\text{Real}}>0],
$$

$$
Y_{\text{control-resistant}}
=
\mathbb{1}
[
Gain_{\text{Real}}>
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})
],
$$

$$
Y_{\text{safe-good}}
=
Y_{\text{risk-safe}}
\cdot
Y_{\text{value-positive}}
\cdot
Y_{\text{control-resistant}}.
$$

Predictivity gate：

$$
AUC(S_{\text{probe}},Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(S_{\text{probe}},V_{\text{safe-grounded}})\geq0.35.
$$

Component gates：

$$
AUC(S_{\text{risk}},Y_{\text{risk-safe}})\geq0.70,
$$

$$
AUC(S_{\text{value}},Y_{\text{value-positive}})\geq0.65,
$$

$$
AUC(S_{\text{gap}},Y_{\text{control-resistant}})\geq0.65.
$$

## 3. Signal-channel / drift-diffusion success

Microprobe 不应只测 CE / margin。必须显式记录 signal-channel 与 drift-diffusion 统计：

$$
S_{\text{SNR}}
=
\frac{\|\mu_{\text{probe}}\|^2}
{\sigma^2_{\text{probe}}+\epsilon}.
$$

Functional update 的 signal-channel ratio：

$$
S_{\text{channel}}
=
\frac{
\|\Pi_{\text{signal}}\Delta z_F\|^2
}{
\|\Delta z_F\|^2+\epsilon
}.
$$

其中 $\Pi_{\text{signal}}$ 可以用 train-stream low-rank sketch 近似，不要求构造完整 eNTK。

Signal gate：

$$
AUC(S_{\text{SNR}},Y_{\text{safe-good}})\geq0.60
$$

or:

$$
AUC(S_{\text{channel}},Y_{\text{safe-good}})\geq0.60.
$$

Official controller 可以使用这些统计，但必须记录 system overhead。

## 4. System overhead success

Online microprobe 不能破坏 system gate。必须记录：

```text
probe_extra_read_MB
probe_extra_write_MB
probe_kernel_count
probe_sync_count
probe_update_shadow_copy_MB
probe_feature_factory_time_ms
probe_controller_time_ms
probe_total_overhead_ratio
memory_traffic_ratio
step_ratio_q90
memory_ratio
```

System gate：

$$
Overhead_{\text{probe}}\leq0.20,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Diagnostic stricter target：

$$
MemoryTraffic_{\text{probe}}\leq1.20MemoryTraffic_{\text{AdamW-step}}.
$$

If predictivity passes but overhead fails, route must be:

```text
R6-MicroProbePredictiveButTooExpensive
```

not success.

## 5. Support-region controller success

Controller 不能再是单一 ranking score。Official controller 必须是 support-region decision：

$$
Accept(e)
=
RiskSafe(e)
\land
ValuePositive(e)
\land
ControlResistant(e)
\land
SignalChannel(e)
\land
SupportStable(e).
$$

其中：

$$
RiskSafe(e)=\mathbb{1}[S_{\text{risk}}(e)\leq\rho],
$$

$$
ValuePositive(e)=\mathbb{1}[S_{\text{value}}(e)\geq\tau_v],
$$

$$
ControlResistant(e)=\mathbb{1}[S_{\text{gap}}(e)\geq\tau_g],
$$

$$
SignalChannel(e)=\mathbb{1}[S_{\text{channel}}(e)\geq\tau_c],
$$

$$
SupportStable(e)=\mathbb{1}[Density_{\text{support}}(e)\geq\tau_s].
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

## 6. Leave-out success

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

## 7. Official paired replay success

Only after P1-P6 pass，official paired replay opens。

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
ShuffledSignalChannel = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

## 8. Short-run scout success

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

## H1：v9.2.47 的失败是 online implementation gap，不是 offline feature absence

证据是 v9.2.47 P3 oracle support pass：

$$
Precision_{\text{oracle}}=1.0,
$$

$$
Coverage_{\text{oracle}}=0.123698.
$$

但 P2：

```text
online_microprobe_implemented = 0
```

H1 成立标准：

真实 online microprobe 实现后，至少一个 probe feature 对 safe-good 有：

$$
AUC\geq0.70
$$

or:

$$
Corr\geq0.35.
$$

H1 失败标准：

online rows 上全部 probe features：

$$
AUC<0.60,
$$

and oracle support remains high.  
此时说明 offline LF8 是不可部署 diagnostic，不能作为 online controller basis。

## H2：risk/value/gap conflict 必须由 multi-factor controller 解决

P1 归因显示 FP 来自 risk/value/gap conflict，FN 来自 family concentration，coverage-loss 来自 threshold crossfit miscalibration。  
H2 认为 single threshold controller 不可能稳定闭合，必须使用 multi-factor region：

```text
RiskSafe
ValuePositive
ControlResistant
SignalChannel
SupportStable
```

H2 成立标准：

support-region controller 相比 C1/C2/C6 同时提升：

```text
precision
coverage
bad-event
accepted family count
```

并通过 heldout gate。

## H3：family concentration 是 coverage-loss 的关键

P1 中 coverage-loss mode = `D6-threshold_crossfit_miscalibration`，FN mode = `D4-family_concentration`。  
H3 认为 threshold 不稳的根因是 accepted support 被少数 families 控制，heldout family shift 导致 coverage collapse。

H3 成立标准：

Family-balanced / density-stable controller achieves：

```text
accepted_family_count >= 4
max_family_share <= 0.60
coverage >= 0.03
bad-event <= 0.05
```

## H4：signal-channel statistics 比 offline boundary probe 更接近 functional update 的本质

H4 认为 safe-good event 不只是 probe CE 改善，而是 functional update 在 signal channel 中有高 drift/diffusion ratio，并且超过 controls。

H4 成立标准：

$$
AUC(S_{\text{SNR}},Y_{\text{safe-good}})\geq0.60
$$

or:

$$
AUC(S_{\text{channel}},Y_{\text{safe-good}})\geq0.60.
$$

If SNR/channel features improve controller precision or LDO stability by at least $0.05$，视为 strong diagnostic.

## H5：如果 online microprobe 有效但 too expensive，下一步是 feature kernelization，不是重做 controller

If:

```text
microprobe AUC pass
controller pass
probe overhead fail
```

then route:

```text
R6-MicroProbePredictiveButTooExpensive
```

下一步做 amortization / kernelization / two-step cached probe。

## H6：如果 online microprobe 失败但 fresh natural oracle support remains high，问题是 legal sufficient statistics 仍不足

If:

```text
oracle support pass = 1
online feature pass = 0
```

then blocker = legal feature representation, not carrier.

## H7：如果 fresh natural oracle support collapses，回到 carrier/support reset

If:

$$
OraclePrecision<0.75
$$

or:

$$
OracleBadEvent>0.05,
$$

then current carrier/event family lacks robust safe-good support.

---

# Part IV. 并行执行设计

v9.2.48 runner 必须并行执行：

```text
Lane A:
  v9.2.47 boundary reproduction

Lane B:
  real online microprobe implementation:
    update/probe train split
    candidate shadow update
    pre-commit feature computation
    commit decision logging

Lane C:
  fresh natural replay rows:
    official evaluation distribution

Lane D:
  boundary-targeted diagnostic rows:
    FP/FN/family-shift autopsy only
    not official pass

Lane E:
  feature factory:
    CE/margin/risk/gap probe
    branch-ratio online
    role-gap
    family reliability
    drift-diffusion SNR
    signal-channel sketch

Lane F:
  support-region controller family

Lane G:
  LDO/LSO

Lane H:
  paired replay scout and official replay

Lane I:
  short-run scout if paired replay passes

Lane J:
  system audit:
    probe overhead
    memory traffic
    kernel count
    sync count
```

Gate discipline：

```text
diagnostic rows may be measured early
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  online microprobe implemented
  online legal feature pass
  support-region controller pass
  LDO/LSO pass
```

---

# Part V. Microprobe implementation details

## 1. Online row generation

For each training step $t$:

```text
sample batch B_t
split B_t into B_update and B_probe

compute current logits z_t
compute AdamW-equivalent gradient/update on B_update
compute candidate functional update on B_update
shadow-apply functional update to copied parameter view or reversible delta view
evaluate probe metrics on B_probe
compute controller decision
commit or abstain
log row
rollback shadow if abstain or after probe
```

Important rule：

```text
B_probe is train-stream only.
B_probe is not validation.
B_probe is not test.
B_probe cannot be selected by dataset name.
```

## 2. Logged hashes

Each online row must record:

```text
row_id
dataset
seed
step
batch_hash
update_batch_hash
probe_batch_hash
task_param_hash_before
candidate_update_hash
probe_feature_hash
controller_decision_hash
commit_time_order_valid
```

`commit_time_order_valid = 1` only if all controller features are computed before any posthoc paired replay outcome.

## 3. Probe features

### PF1：Probe CE / margin

$$
S_{\text{value-probe}}
=
-\Delta CE_{\text{probe}}
+
\lambda_m\Delta MarginP10_{\text{probe}}.
$$

### PF2：Probe risk

$$
S_{\text{risk-probe}}
=
CEp99_{\text{probe}}
+
WrongConfidenceP95_{\text{probe}}
-
MarginP10_{\text{probe}}
+
Uncertainty_{\text{probe}}.
$$

### PF3：Probe control-gap proxy

$$
S_{\text{gap-probe}}
=
\widehat{Gain}_{\text{Real}}
-
\max(
\widehat{Gain}_{\text{AdamWParallel}},
\widehat{Gain}_{\text{bestLR}}
).
$$

This may use shadow control branches on train probe batch only. If too expensive, log diagnostic but do not official promote.

### PF4：Drift-diffusion SNR

$$
S_{\text{SNR}}
=
\frac{\|\mu_{\text{probe}}\|^2}
{\sigma^2_{\text{probe}}+\epsilon}.
$$

### PF5：Signal-channel sketch

Use low-rank sketch $U_k$ from train-stream output displacement covariance:

$$
S_{\text{channel}}
=
\frac{
\|U_k^T\Delta z_F\|^2
}{
\|\Delta z_F\|^2+\epsilon
}.
$$

### PF6：Support density

$$
Density(e)=\frac{k}{N\cdot Volume(\mathcal{N}_k(e))}.
$$

### PF7：Family reliability

Family must not contain dataset name：

$$
family(e)
=
(stratum,horizon,risk\_bucket,role\_bucket,attach\_type,branch\_bucket,probe\_bucket).
$$

Reliability：

$$
Rel(f)
=
\mathbb{E}[Y_{\text{safe-good}}\mid f]
-
\kappa\sqrt{\operatorname{Var}(Y_{\text{safe-good}}\mid f)}.
$$

---

# Part VI. Controller candidates

## C0：Offline C1 Reference

Reference only. Expected fail.

## C1：ProbeRiskValueGapTriStage

$$
Accept(e)
=
RiskSafe_{\text{probe}}(e)
\land
ValuePositive_{\text{probe}}(e)
\land
ControlResistant_{\text{probe}}(e).
$$

## C2：SignalChannelSupportController

$$
Accept(e)
=
RiskSafe(e)
\land
ValuePositive(e)
\land
ControlResistant(e)
\land
SignalChannel(e).
$$

## C3：FamilyBalancedProbeController

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
FamilyBalance(e).
$$

## C4：SupportDensityController

$$
Accept(e)
=
RiskSafe(e)
\land
ValuePositive(e)
\land
ControlResistant(e)
\land
Density(e)\geq d_0.
$$

## C5：ParetoFrontOnlineController

Accept if event lies on Pareto front of:

```text
low probe risk
positive probe value
positive control gap
high SNR
high signal-channel ratio
stable family support
low overhead
```

## C6：AbstainFirstLCBController

$$
Accept(e)
=
\mathbb{1}
[
LCB(Value)>0
\land
LCB(Gap)>0
\land
UCB(Risk)<\rho
\land
LCB(SignalChannel)>\tau_c
].
$$

## C7：Oracle

Posthoc diagnostic only. Never official.

---

# Part VII. 实验阶段

## P0：v9.2.47 boundary reproduction

### 目标

确认 v9.2.47 boundary 稳定。

### 必须记录

```text
route
source_route_v9246
decision_region_autopsy_pass
online_microprobe_implemented
online_microprobe_pass
fresh_natural_row_count
oracle_support_pass
oracle_precision
oracle_coverage
best_controller_id
controller_auc
controller_corr
accepted_precision
accepted_coverage
accepted_bad_event_rate
primary_blocker
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R7-OfflineFeatureOnly
online_microprobe_implemented = 0
oracle_support_pass = 1
support_region_controller_pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_offline_to_online_gap_ladder.svg
```

---

## P1：real online microprobe implementation

### 目标

实现真实 train-stream update/probe microprobe。

### 必须记录

```text
row_id
dataset
seed
step
signal_stratum
carrier_id
update_batch_hash
probe_batch_hash
candidate_update_hash
precommit_feature_hash
commit_decision_timestamp
posthoc_outcome_timestamp
commit_time_order_valid
uses_dataset_name
uses_validation
uses_test
uses_posthoc
online_microprobe_implemented
```

### 判断标准

P1 pass：

```text
online_microprobe_implemented = 1
commit_time_order_valid = 1
fresh_train_stream_update_probe_rows >= 2000
uses_dataset_name = 0
uses_validation = 0
uses_test = 0
uses_posthoc = 0
```

### 可视化

```text
p1_microprobe_row_flow.svg
p1_commit_time_order_audit.svg
p1_probe_coverage_by_stratum.svg
```

---

## P2：microprobe feature predictivity and signal-channel audit

### 目标

判断 online probe features 是否预测 safe-good。

### 必须记录

```text
feature_id
feature_group
AUC_to_risk_safe
AUC_to_value_positive
AUC_to_control_resistant
AUC_to_safe_good
corr_to_safe_grounded_value
precision_at_gate
coverage_at_gate
bad_event_at_gate
SNR_mean
SNR_std
signal_channel_ratio_mean
signal_channel_ratio_p90
feature_overhead
memory_overhead
```

### 判断标准

Feature pass：

$$
AUC(S,Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(S,V_{\text{safe-grounded}})\geq0.35.
$$

Component pass as defined in Part II.

System pass：

$$
FeatureOverhead\leq0.20.
$$

### 可视化

```text
p2_feature_predictivity_matrix.svg
p2_snr_vs_safe_good.svg
p2_signal_channel_ratio.svg
p2_feature_overhead_pareto.svg
```

---

## P3：fresh natural replay and oracle support

### 目标

确认在真实 online row distribution 下 safe-good support 是否仍存在。

### 必须记录

```text
row_id
row_source = natural
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

Fresh natural support pass：

```text
natural_real_event_count >= 2000
measured_signal_strata_count >= 6
carrier_active = 1
fake/proxy/offload = 0
```

Oracle support gate：

$$
OraclePrecision_{\text{safe-good}}\geq0.75,
$$

$$
OracleCoverage_{\text{safe-good}}\in[0.03,0.15],
$$

$$
OracleBadEventRate\leq0.05.
$$

### 可视化

```text
p3_fresh_oracle_support.svg
p3_safe_good_prevalence_by_stratum.svg
p3_control_gap_by_family.svg
```

---

## P4：support-region controller calibration

### 目标

并行校准 C1-C7，并把 official pass 限制在 heldout natural rows 上。

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
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRiskScore, ShuffledBranchRatio, ShuffledControlGap, ShuffledValueScore, ShuffledProbeScore, ShuffledSignalChannel, FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled, DatasetRouteShuffled, EventRouteShuffled, InvertedRoleMask
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

# Part VIII. Required artifacts

```text
run_manifest.json
contract_audit_v9248.csv
p0_v9247_boundary_reproduction.csv
p1_real_online_microprobe_implementation.csv
p2_microprobe_feature_predictivity_signal_channel.csv
p3_fresh_natural_replay_oracle_support.csv
p4_support_region_controller_calibration.csv
p5_leave_dataset_and_stratum_out.csv
p6_official_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_robustness_external_ready.csv
online_microprobe_trace_v9248.csv
commit_time_order_trace_v9248.csv
signal_channel_trace_v9248.csv
support_density_trace_v9248.csv
controller_calibration_trace_v9248.csv
leaveout_trace_v9248.csv
paired_replay_branch_trace_v9248.csv
system_probe_overhead_trace_v9248.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9247_boundary_unstable
F3_dataset_tuning_detected
F4_online_microprobe_not_implemented
F5_commit_time_order_violation
F6_microprobe_not_predictive
F7_signal_channel_not_predictive
F8_microprobe_system_too_expensive
F9_fresh_natural_rows_insufficient
F10_oracle_support_collapse
F11_support_region_controller_fail
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

# Part IX. Route decision

```text
R1-OnlineMicroProbeImplemented:
  real train-stream update/probe rows exist and pass legality.

R2-OnlineMicroProbePredictive:
  at least one online probe feature predicts safe-good.

R3-SignalChannelFeaturePass:
  SNR / signal-channel sketch adds predictive value.

R4-SupportRegionControllerPass:
  support-region controller passes heldout precision / coverage / bad-event gate.

R5-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R6-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R7-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R8-MicroProbePredictiveButTooExpensive:
  online probe predicts safe-good but violates system overhead.

R9-OfflineFeatureOnlyAgain:
  only source-measured / boundary-probe statistics work; online feature fails.

R10-OracleSupportCollapse:
  fresh natural replay shows no sufficient safe-good oracle support.

R11-ControllerStillInvalid:
  online features exist, but no deployable support-region controller passes.

R12-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R13-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R14-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9247_boundary_pass
dataset_tuning_detected
online_microprobe_implemented
online_microprobe_pass
microprobe_auc
microprobe_corr
microprobe_overhead
signal_channel_pass
signal_channel_auc
signal_channel_corr
fresh_natural_row_count
oracle_support_pass
oracle_precision
oracle_coverage
oracle_bad_event
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
success_v9248_strict_purekan_functional
success_v9248_full_functional
success_v9248_external_ready
```

---

# Part X. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 real online microprobe implementation
  P2 microprobe feature predictivity / signal-channel audit
  P3 fresh natural replay / oracle support

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
  online microprobe implemented
  online legal feature pass
  support-region controller pass
  LDO/LSO pass
```

---

# Part XI. 停止条件

## Minimum diagnostic success

```text
v9.2.47 boundary reproduced
real online microprobe implemented
fresh natural replay measured
microprobe legality audited
probe overhead measured
no fake/proxy/offload/loss/teacher violation
```

## Online feature success

```text
Minimum diagnostic success
+
at least one online legal feature predicts safe-good
+
feature/probe overhead gate pass
```

## Legal controller success

```text
Online feature success
+
support-region controller passes heldout precision / coverage / bad-event gate
+
multi-stratum / multi-family gate pass
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
1. v9.2.47 boundary cannot be reproduced；
2. online microprobe still not implemented；
3. commit-time order audit fails；
4. microprobe features fail safe-good predictivity；
5. only predictive feature is offline/source-measured；
6. microprobe is predictive but too expensive；
7. fresh natural replay oracle support collapses；
8. all support-region controllers fail heldout gate；
9. leave-dataset-out fails；
10. leave-stratum-out fails；
11. paired replay remains control-equivalent；
12. shuffle controls pass, indicating overfit；
13. short-run task drops；
14. full run gives no macro / hard-stratum / geometry gain；
15. functional breaks system gate；
16. gains are explained by QuadraticFeatureMLP；
17. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XII. 最终解释规则

## Case A：online microprobe implemented and predictive

可以声明：

```text
v9.2.47 failed because feature was offline-only; v9.2.48 implements true train-stream online feature.
```

但不能声明 strict functional success unless controller + LDO/LSO + paired replay pass.

## Case B：online microprobe implemented but not predictive

必须声明：

```text
offline boundary-probe features do not transfer to online commit-time statistics.
```

下一步设计 richer train-stream sufficient statistics，不调 dataset。

## Case C：online microprobe predictive but too expensive

必须声明：

```text
value is observable online but not system-legal.
```

下一步做 microprobe amortization / kernelization。

## Case D：support-region controller fails

必须声明：

```text
safe-good is observable, but accept/abstain policy remains undeployable.
```

下一步修 support geometry / family reliability / controller calibration。

## Case E：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

## Case F：paired replay passes

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation。

---

# Part XIII. 最终建议

v9.2.48 的一句话策略是：

$$
\boxed{
\text{停止 offline autopsy；实现真实 online train-stream microprobe，并用 support-region controller 直接冲 LDO/LSO + paired replay。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
LF8 有没有 AUC；
C1/C6 threshold 是否差一点；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. 能不能真实生成 update/probe train-stream rows？
2. online microprobe 是否能预测 safe-good？
3. SNR / signal-channel sketch 是否比 CE/margin probe 更稳定？
4. support-region controller 能否在 heldout 上同时满足 precision / coverage / bad-event？
5. controller 能否 LDO/LSO？
6. official paired replay 能否打过 AdamWParallel / bestLR？
7. probe overhead 是否在 system envelope 内？
```

v9.2.48 的结果将给出清晰分叉：

```text
if online microprobe + support-region controller + LDO/LSO + paired replay pass:
  strict PureKAN functional obtains local causal evidence.

if online microprobe predictive but too expensive:
  feature amortization / kernelization becomes blocker.

if offline LF8 works but online microprobe fails:
  offline boundary-probe statistic is not deployable.

if oracle support collapses:
  carrier/support stability remains blocker.

if controller passes local but LDO/LSO fails:
  no dataset tuning; controller needs broader signal-stratum/family support.
```
