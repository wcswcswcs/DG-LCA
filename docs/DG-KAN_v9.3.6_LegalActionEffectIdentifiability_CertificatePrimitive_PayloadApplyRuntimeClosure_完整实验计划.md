# DG-KAN v9.3.6 Legal Action-Effect Identifiability / Certificate Primitive / PayloadApply Runtime Closure 完整实验计划

> 本计划基于 v9.3.5 `ControlPositive Frontier Completion / Legal Probe / ActionDensity Runtime Closure` 的真实结果制定。  
> v9.3.6 不继续做 StableAccept patch，也不继续做单特征阈值搜索。  
> 本轮核心问题从 “control-positive frontier 是否存在” 转为：
>
> $$
> \boxed{
> \text{为什么 full control-positive oracle frontier 足够 dense，但 commit 前 legal observables 看不见它？}
> }
> $$

---

# 0. 执行摘要

v9.3.5 的最大贡献不是系统成功，而是把一个关键不确定性彻底消掉：

```text
v9.3.4:
  只完成 864 actions / 15552 branch-horizon rows panel；
  panel raw coverage = 0.0170855；
  因此被保守标成 ControlPositiveOracleAbsent。

v9.3.5:
  完成 full universe：
    actions = 2876 / 2876
    branch-horizon rows = 51768 / 51768
    missing branch/horizon/secondary = 0 / 0 / 0
  full weak control-positive oracle:
    accepted = 893
    coverage = 0.0984347
    coverage LCB = 0.0924734
    precision_control_positive = 1.0
    bad/null = 0.0 / 0.0
    V_ctrl_lcb = 0.2833108
    beats AdamWParallel / bestLR / NoOp / Random = 1.0 / 1.0 / 1.0 / 1.0
```

这说明 AP0 current action primitive 不是“完全不产生好 action”。相反，full universe 中大约有：

$$
\frac{893}{2876}=0.3105
$$

的 candidate/action 是 weak control-positive；换成 heldout denominator coverage 是：

$$
\frac{893}{9072}=0.0984.
$$

所以，v9.3.5 后不能再把主 blocker 写成 action primitive density absent。当前主 blocker 是：

$$
\boxed{
\text{control-positive action 足够多，但当前 legal commit-time signals 无法识别它们。}
}
$$

v9.3.5 同时暴露了三个更深层问题：

```text
1. Panel representativeness failure:
   v9.3.4 的 30% panel 明显低估 CP coverage。
   panel raw coverage = 0.0171
   full coverage = 0.0984
   weighted coverage error = 0.0503
   route flip = 1

2. Static / probe observability failure:
   best static feature = F4-StateNLL
   AUC_CP = 0.615171
   best probe = PR5-ProbeReliability
   AUC_CP = 0.503069
   probe cost q90 = 2.4969 ms

3. Payload-apply runtime failure:
   oracle-mask diagnostic accepted = 61
   payload_apply_time_ms_q90 = 0.762382
   base_train_step_time_ms_q90 = 0.803375
   total_step_time_ms_q90 = 1.554239
   step_ratio_q90 = 2.068737
```

v9.3.6 的方向因此必须是：

```text
A. 不再问 “有没有好 action”；
B. 改问 “好 action 的 legal action-effect 证据在哪里”；
C. 如果 AP0 的好 action 不可观测，则重构 action primitive，使 action 自带 legal certificate；
D. 同时把 payload apply 从独立重成本降到 fused / sparse / lazy apply，真实进入 step_ratio_q90 <= 1.50。
```

---

# 1. 独立数据判断

## 1.1 v9.3.5 是重大进展，但不是 controller 进展

v9.3.5 的 P1 full control outcome completion 是实质进展：

```text
action_count_expected/completed = 2876 / 2876
row_count_expected/actual = 51768 / 51768
row_count_reused_from_v9340 = 0
row_count_newly_materialized = 51768
row_count_failed/retried = 0 / 0
rows_per_sec_total = 24.3084
wallclock_sec = 2479.47
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1.0
quality_audit_pass = 1
```

这把 v9.3.3 的 `48 / 51768` dry-run 和 v9.3.4 的 `15552 / 51768` panel 推进到 full action-value universe。现在项目第一次真正拥有完整的 matched-control action-value label table。

但这不是 controller 进展。原因是：

```text
static_observability_pass = 0
probe_observability_pass = 0
decision_gate_pass = 0
payload_apply_runtime_pass = 0
system_legal_controller_pass = 0
```

也就是说，v9.3.5 完成的是 truth table，不是 deployable rule。

## 1.2 v9.3.4 的 ControlPositiveOracleAbsent 结论被纠正

v9.3.4 panel 上：

```text
panel_action_count = 864
control_positive_accepted_panel = 155
coverage_raw = 0.0170855
```

这个 coverage 低于 official `0.03`，所以 v9.3.4 route 停在 `R5-ControlPositiveOracleAbsent` 是当时合理的 gate。但 v9.3.5 full universe 显示：

```text
coverage_full = 0.0984347
accepted_full = 893
```

并且 v9.3.5 的 panel audit 给出：

```text
weighted_coverage_error = 0.0503334
panel_CP_rate_error_macro = 0.0931009
max_stratum_coverage_gap = 0.699583
route_flip_panel_vs_full = 1
```

这说明 v9.3.4 的 panel 不是可代表 full universe 的 official conclusion panel。以后不能再用 naive 30% panel 对 oracle density 下最终结论；只能作为 smoke 或成本估算。

## 1.3 当前 AP0 action primitive 有 value，但 value 不是 horizon-robust

v9.3.5 的 OR1 full weak oracle 很强：

```text
OR1-FullWeakControlPositiveOracle:
  accepted = 893
  coverage = 0.0984347
  precision_control_positive = 1.0
  V_ctrl_lcb = 0.283311
```

但 OR2 / OR3 提醒我们不要过度乐观：

```text
OR2-FullStrongControlPositiveOracle:
  accepted = 146
  coverage = 0.0160935

OR3-HorizonRobustControlPositiveOracle:
  accepted = 13
  coverage = 0.001433
```

这说明 AP0 的好 action 很多，但很多是 weak / horizon-conditional good，不是 strong / horizon-robust good。v9.3.6 必须把 value 分成：

```text
weak control-positive
strong control-positive
horizon-robust control-positive
short-horizon only positive
long-horizon negative
```

否则我们可能训练出一个能选到短期好 action、但连续训练中不稳定的 controller。

## 1.4 “static observability fail” 不能被误读成 “legal observability 不可能”

P4 只测了少数静态特征：

```text
PayloadNorm AUC_CP = 0.411375
PayloadLinf AUC_CP = 0.413821
StateTailCEp99 AUC_CP = 0.606758
StateMarginP10 AUC_CP = 0.610839
StateNLL AUC_CP = 0.615171
FamilySupportCount AUC_CP = 0.421735
```

P5 probe 也失败：

```text
best_probe = PR5-ProbeReliability
AUC_CP = 0.503069
PR lift = 1.08033
probe_cost_q90 = 2.4969 ms
```

但是，这只证明：

```text
payload magnitude；
simple state-tail features；
support count；
当前 probe design
```

不是 control-positive sufficient statistic。

它没有证明所有 legal observables 不存在。当前最合理的判断是：

$$
\boxed{
\text{CP label 可能依赖 action × state × gradient/control interaction，而不是任一静态边际特征。}
}
$$

因此 v9.3.6 必须从 “static row features” 转向 “legal action-effect fingerprints”。

## 1.5 Runtime 新 blocker 已经变成 payload apply，而不是 empty-step

v9.3.4 no-payload smoke 曾经：

```text
step_ratio_q90 = 1.026625
payload_apply_used = 0
```

v9.3.5 payload-apply diagnostic：

```text
payload_apply_used = 1
feature_compute_time_ms_q90 = 0.00647
score_accept_time_ms_q90 = 0.00323
payload_apply_time_ms_q90 = 0.76238
base_train_step_time_ms_q90 = 0.80337
total_step_time_ms_q90 = 1.55424
step_ratio_q90 = 2.06874
```

这说明 feature / score / accept 几乎不花时间；payload apply 是主成本。要达到：

$$
StepRatio_{q90}\le 1.50
$$

在 base q90 约 `0.803 ms` 的情况下，总 step q90 需要：

$$
TotalStepQ90 \le 1.50 \times 0.803 \approx 1.205 ms.
$$

当前 total 是 `1.554 ms`，需要至少降低：

$$
1.554 - 1.205 = 0.349 ms.
$$

由于 feature+score 只有约 `0.010 ms`，payload apply q90 必须从 `0.762 ms` 降到大约：

$$
0.35 \text{ ms}
$$

以内，或者与 base AdamW update 融合/重叠掉约一半成本。

---

# 2. 当前是否在正确道路上

## 2.1 高层方向仍然正确

从 v9.3.0 到 v9.3.5 的推进顺序是科学的：

```text
v9.3.0:
  证明 primary oracle frontier 存在；
  StableAccept patch exhausted；
  legal feature/controller 找不到 frontier。

v9.3.1:
  确认 action apply / secondary-control / runtime 缺失，阻止错误 officialization。

v9.3.2:
  action apply error 全量闭合。

v9.3.3:
  durable payload package / disk replay 闭合。

v9.3.4:
  control outcome materializer 扩到 30% panel。

v9.3.5:
  full control outcome universe 闭合；
  weak control-positive oracle pass；
  static/probe observability fail；
  payload-apply runtime fail。
```

这条路线没有按 dataset 调参，没有把 diagnostic 提升成 official，也没有把 full oracle pass 写成 system success。这是正确的。

## 2.2 但现在不能继续 “feature patching”

v9.3.5 之后，继续这样做是错误路线：

```text
再加 20 个静态 feature；
再调一个 payload_norm threshold；
再试一个 support_count / state_nll 组合；
再用同一 probe family 搜索阈值；
用 oracle mask runtime 当 official runtime。
```

因为现在的问题不是 “某个 feature 差一点”，而是 “当前特征族的物理对象可能错了”。v9.3.6 必须问：

```text
1. CP action 的 legal evidence 是不是局部一阶/二阶 action-effect？
2. 如果是，能不能从 manual gradient / logits / hard-tail Jacobian 中低成本提取？
3. 如果不是，AP0 是否需要变成 certificate-producing action primitive？
4. 如果 action 本身无法给出 legal certificate，那它即使 oracle 好，也可能无法成为 online controller。
```

---

# 3. v9.3.6 总体目标

v9.3.6 的总体目标是：

$$
\boxed{
\text{在 full control-positive label universe 上，判断 AP0 的 control-positive frontier 是否具有可部署的 legal action-effect identifiability。}
}
$$

同时，v9.3.6 必须让 payload apply runtime 从 diagnostic upperbound 进入 official-candidate measured path。

更具体地说，本轮要完成两条并行主线：

```text
Main Track A — Identifiability:
  从 full control-positive labels 出发，建立 legal action-effect fingerprint。
  如果 fingerprint 不能识别 CP frontier，则判定 AP0 是 oracle-good but non-identifiable。

Main Track B — Runtime:
  把 payload apply 从独立 0.762 ms q90 降到 fused/sparse/lazy apply，
  使 actual payload-apply online path 达到 step_ratio_q90 <= 1.50。
```

v9.3.6 最低有效推进：

```text
1. 完成 CP label decomposition：
   weak / strong / horizon-robust / short-only / long-risk 分清楚。

2. 完成 legal observability capacity audit：
   判断 static features 失败是 feature family 不够，还是 CP label 对 commit-time info 基本不可观测。

3. 至少实现一组 action-effect fingerprint：
   使用 manual gradient / AdamW delta / functional payload / logit-tail linearization；
   不用 future outcome，不用 dataset_name。

4. 完成 cheap legal micro-probe redesign：
   不再使用 expensive full probe；
   probe q90 必须进入可部署预算。

5. 完成 cross-fitted CP controller candidate：
   若 action-effect features 过 signal gate，必须跑 decision gate；
   若不过，不能继续阈值搜索，直接触发 primitive certificate reset。

6. 完成 payload-apply runtime optimization：
   oracle-mask diagnostic、selected-controller path、no-payload smoke 三者必须分开记录。

7. route 必须明确落在：
   LegalIdentifiabilityPass / LegalIdentifiabilityFail /
   CertificatePrimitiveNeeded / PayloadApplyRuntimeFail /
   SystemLegalControllerPass
   之一。
```

---

# 4. 本轮明确不做什么

v9.3.6 不做：

```text
1. 不继续修 StableAccept；
2. 不把 StableAccept、StateNLL、PayloadNorm 单独作为 final controller；
3. 不按 MNIST / Fashion-MNIST / KMNIST 调 threshold；
4. 不把 v9.3.5 oracle labels 当 commit-time feature；
5. 不用 test/validation metric at commit time；
6. 不用 teacher / distillation / auxiliary loss / loss modification；
7. 不把 expensive probe 当 legal online controller；
8. 不把 oracle-mask runtime 写成 official；
9. 不把 no-payload runtime smoke 写成 payload-apply runtime pass；
10. 不在 system controller 未过前打开 official LDO/LSO/paired replay/full-run。
```

允许：

```text
1. 按 dataset / family / horizon / stratum 诊断 failure，但不能按 dataset 制定规则；
2. 使用 full labels 做 offline observability analysis；
3. 使用 cross-fitted calibration labels 训练 risk/value estimator；
4. 使用 manual gradient、batch labels、logits、functional payload、AdamW delta；
5. 使用 legal local Taylor / Gauss-Newton / hard-tail linearization；
6. 并行运行 AP0 identifiability 和 AP1 certificate primitive smoke；
7. 对 runtime 做 oracle-mask diagnostic，但 route 中必须标成 diagnostic。
```

---

# 5. 核心假设

## H1：AP0 current action primitive 有足够 weak control-positive density

H1 已由 v9.3.5 支持，但 v9.3.6 要进一步分解它。

成立标准：

```text
weak CP coverage >= 0.03
weak CP precision oracle = 1.0
V_ctrl_lcb > 0
beats AdamWParallel / bestLR / NoOp / Random all >= 0.95
```

额外分解：

```text
strong CP coverage
horizon-robust CP coverage
short-only CP coverage
long-risk CP coverage
per-family CP density
per-horizon CP density
per-dataset diagnostic CP density
```

失败标准：

```text
重新计算 full labels 后 weak CP coverage < 0.03
或 label quality audit 出现 violation。
```

## H2：当前 observability 失败是因为缺少 action-effect features，而不是 CP label 不可观测

H2 是 v9.3.6 的核心科学假设。

如果 H2 成立，则 action-effect fingerprints 应显著优于 static features：

```text
AUC_CP >= 0.75
PR_lift_CP >= 2.0
top273 CP precision >= 0.75
LDO AUC drop <= 0.07
LSO AUC drop <= 0.10
feature_cost_q90 <= 0.20 ms
```

如果 H2 失败，则说明 AP0 的好 action 对当前 legal information 不可识别，需要 primitive reset。

## H3：CP frontier 的关键 evidence 是 local action-response，而非 state/payload marginal

H3 认为 `StateNLL` 只能达到 AUC `0.615`，因为它只描述 “当前状态难不难”，不描述 “这个 action 会不会改善当前状态”。

action-response feature 应类似：

$$
\widehat{\Delta CE}_{func}(e)=g_t^\top \Delta\theta_{func}(e)
$$

$$
\widehat{\Delta Margin}_{tail}(e)=J_{tail}(x_t)\Delta\theta_{func}(e)
$$

$$
Conflict(e)=1-\cos(\Delta\theta_{func}(e),\Delta\theta_{AdamW})
$$

若这些仍无法识别 CP，则 AP0 可能是 “oracle-good but legally opaque”。

## H4：legal pre-commit probe 失败是 probe design 失败，不一定是 probe 方向失败

v9.3.5 的 PR5 probe：

```text
AUC_CP = 0.5031
cost_q90 = 2.4969 ms
```

H4 认为它失败是因为 probe 太重、太粗、没有对准 action-effect。新的 micro-probe 必须是低成本局部响应，不是 mini replay。

成立标准：

```text
best_micro_probe_auc_CP >= 0.70
best_micro_probe_pr_lift >= 2.0
probe_cost_q90 <= 0.20 ms
probe_memory_ratio <= 1.02
```

若所有 micro-probe 仍接近随机，则停止 probe 路线。

## H5：payload apply runtime 可以通过 fused/sparse/lazy apply 进入 envelope

v9.3.5 的 payload apply q90 约 `0.762 ms`。H5 认为这是实现路径问题，不是不可避免成本。

成立标准：

```text
payload_apply_time_ms_q90 <= 0.35
total_step_ratio_q90 <= 1.50
payload_apply_error_linf_max <= 1e-8
payload_apply_cosine_min >= 0.999999
memory_ratio <= 1.05
```

失败标准：

```text
即使 fused/sparse/lazy apply 后，payload apply q90 仍 > 0.45
或 total step ratio q90 > 1.50。
```

## H6：如果 AP0 legal identifiability fail，则需要 certificate-producing action primitive

H6 认为，oracle-good but non-identifiable action 对 online controller 没用。AP1/AP2 应生成 action 时同时生成 legal certificate：

```text
predicted descent certificate
tail safety certificate
AdamW conflict certificate
support certificate
cost certificate
```

而不是先生成 opaque action，再让 controller 事后猜。

---

# 6. 数据合同

## 6.1 Full control-positive label contract

每个 action row 必须保留：

```text
candidate_id
action_id
event_id
dataset
seed
step
family_id
bucket_id
horizon
branch
control_branch
V_real
V_adamwparallel
V_bestlr
V_noop
V_random
V_ctrl
primary_safe_label
bad_event_label
null_event_label
weak_control_positive_label
strong_control_positive_label
horizon_robust_control_positive_label
short_only_positive_label
long_risk_label
support_stratum_id
```

定义：

$$
V_{ctrl}(e,h)=V_{RealFunctional}(e,h)-\max_{b\in Controls}V_b(e,h).
$$

weak CP：

$$
CP_{weak}(e)=1
\iff
PrimarySafe(e)=1
\land Bad(e)=0
\land Null(e)=0
\land V_{ctrl}(e)>0.
$$

strong CP：

$$
CP_{strong}(e)=1
\iff
CP_{weak}(e)=1
\land V_{ctrl}(e)\ge q_{strong}
\land BeatsAllControls(e)=1.
$$

horizon-robust CP：

$$
CP_{robust}(e)=1
\iff
\forall h\in\{20,80,240\},\quad V_{ctrl}(e,h)>0
\land Bad(e,h)=0.
$$

## 6.2 Legal feature contract

Legal features may use:

```text
current train batch inputs / labels
current logits
current CE / margin / entropy
manual gradients
manual AdamW delta
functional payload delta
candidate metadata except dataset_name
family/horizon/bucket as protocol variables
calibration-only support statistics
precomputed action payload hash/norm
preloaded payload tensors
```

Legal features may not use:

```text
future branch outcome
heldout label
control-positive label at commit time
test/validation metric
dataset_name branch
posthoc oracle mask
source measured gap
formula proxy promoted to official
```

Each feature row must record:

```text
feature_id
feature_group
feature_value
feature_cost_ms
feature_memory_delta_mb
uses_dataset_name
uses_future_outcome
uses_test_or_validation
uses_oracle_label
uses_posthoc_outcome
commit_time_available
```

## 6.3 Action-effect fingerprint contract

每个 candidate/action 必须记录：

```text
delta_func_norm
delta_adamw_norm
delta_func_over_adamw_norm
cos_func_adamw
cos_func_negative_grad
grad_dot_delta_func
grad_dot_delta_adamw
linearized_CE_delta_func
linearized_margin_delta_func
tail_logit_delta_p90
tail_logit_delta_p99
hard_tail_conflict_rate
role_energy_entropy
layerwise_delta_energy
functional_channel_entropy
estimated_curvature_penalty
gauss_newton_proxy_delta
control_conflict_score
action_effect_certificate_score
```

## 6.4 Runtime contract

必须区分三种 runtime：

```text
RT-A: no-payload smoke
RT-B: oracle-mask payload diagnostic
RT-C: selected-controller official candidate runtime
```

每 step 记录：

```text
runtime_mode
controller_id
oracle_mask_used
payload_apply_used
payload_apply_impl
step_count
active_step_count
candidate_count
accepted_count
feature_compute_time_ms_q90
score_accept_time_ms_q90
payload_lookup_time_ms_q90
payload_pack_time_ms_q90
payload_apply_time_ms_q90
fused_update_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
payload_apply_error_linf_max
payload_apply_cosine_min
```

---

# 7. 实验阶段

---

## P0：v9.3.5 boundary independent reanalysis

### 目标

重新计算 v9.3.5 的 gate gap，避免被 route 名称牵着走。

### 假设

H0/P0：v9.3.5 已经把 blocker 从 “oracle frontier absent” 推到 “legal observability gap + payload apply runtime gap”。

### 必须记录

```text
source_run_id
artifact_hash
route
candidate_count
action_count
full_control_rows_expected
full_control_rows_actual
rows_per_sec_total
full_control_outcome_ready
weak_CP_accepted_count
weak_CP_coverage
weak_CP_coverage_lcb
weak_CP_precision
weak_CP_bad_rate
weak_CP_null_rate
V_ctrl_mean
V_ctrl_lcb
strong_CP_accepted_count
strong_CP_coverage
horizon_robust_CP_accepted_count
horizon_robust_CP_coverage
best_static_feature
best_static_auc_CP
best_probe_id
best_probe_auc_CP
probe_cost_q90
payload_apply_time_ms_q90
step_ratio_q90
primary_blocker
```

### 判断标准

P0 pass：

```text
full control outcome ready = 1
weak CP oracle pass = 1
static observability pass = 0
probe observability pass = 0
payload apply runtime pass = 0
system legal controller pass = 0
```

### 可视化

```text
p0_v9350_route_ladder.svg
p0_oracle_vs_observability_gap.svg
p0_runtime_payload_apply_gap.svg
p0_v9340_panel_to_v9350_full_flip.svg
```

---

## P1：control-positive label decomposition and horizon structure

### 目标

把 CP label 从一个二元标签拆成 weak / strong / horizon-robust / short-only / long-risk，明确 AP0 的 value 是哪类 value。

### 假设

H1：AP0 weak CP density 足够，但 horizon-robust CP density 很低。controller 若直接追 weak CP，可能短期有效但 long-run 不稳。

### 实现

读取：

```text
full_control_outcome_table_v9350.csv
control_positive_oracle_trace_v9350.csv
```

生成：

```text
p1_cp_label_decomposition_v9360.csv
cp_horizon_structure_trace_v9360.csv
```

### 必须记录

```text
candidate_id
action_id
event_id
family_id
bucket_id
horizon
dataset
seed
V_ctrl_h20
V_ctrl_h80
V_ctrl_h240
bad_h20
bad_h80
bad_h240
null_h20
null_h80
null_h240
weak_CP
strong_CP
horizon_robust_CP
short_only_CP
long_risk
horizon_instability_score
V_ctrl_min_horizon
V_ctrl_mean_horizon
V_ctrl_std_horizon
```

### 判断标准

P1 pass：

```text
weak_CP_count >= 273
weak_CP_coverage >= 0.03
weak_CP_label_quality_pass = 1
horizon decomposition complete for all 2876 actions
missing horizon fields = 0
```

Route implications：

```text
if weak_CP pass and horizon_robust_CP fail:
  controller target must be horizon-aware or horizon-weighted；
  cannot claim robust functional advantage yet。

if weak_CP fail:
  route = R1-ActionDensityRegression
```

### 可视化

```text
p1_cp_density_by_horizon.svg
p1_weak_strong_robust_overlap_venn.svg
p1_V_ctrl_horizon_heatmap.svg
p1_short_only_vs_long_risk_distribution.svg
p1_cp_by_family_bucket_stratum.svg
```

---

## P2：legal observability capacity audit

### 目标

在不继续堆 controller 的情况下，先判断 “现有 legal feature sigma-algebra 是否有足够信息量”。这一步回答的是可识别性，不是最终部署。

### 假设

H2：静态特征失败是因为缺少 action-effect interaction；但在更完整的 legal feature groups 中，CP 应该可识别。

### Feature groups

```text
G0: v9.3.5 static baseline
  PayloadNorm, PayloadLinf, StateTailCEp99, StateMarginP10, StateNLL, FamilySupportCount

G1: action-state marginal
  state tail + payload norm + family/horizon/bucket support

G2: gradient-action alignment
  g dot delta_func, cos_func_adamw, cos_func_negative_grad, norm ratio

G3: output-space action response
  hard-tail logit delta, margin delta linearization, CE delta linearization

G4: control conflict estimate
  predicted RealFunctional improvement - predicted AdamW improvement

G5: support-calibrated action neighborhood
  local kNN support in action-effect feature space, EB bad UCB, EB value LCB

G6: minimal certificate features
  descent certificate, tail-safety certificate, conflict certificate, support certificate, cost certificate
```

### 模型只是 capacity diagnostic

允许使用以下 diagnostic models，但不得直接 official：

```text
single-feature ranking
monotone logistic regression
isotonic calibrated score
small monotone GBDT / GAM diagnostic
kNN oracle concentration diagnostic
```

禁止：

```text
dataset_name split
opaque 100-feature black-box official controller
using CP label at commit time
```

### 必须记录

```text
feature_group
model_class
feature_count
uses_dataset_name
uses_future_outcome
AUC_CP
PR_AUC_CP
PR_lift_CP
Brier_CP
ECE_CP
top273_CP_count
top273_CP_precision
top468_CP_precision
top893_CP_precision
leave_dataset_auc_drop
leave_stratum_auc_drop
leave_family_auc_drop
feature_cost_q90
feature_missing_rate
monotone_sign_consistency
label_shuffle_auc
```

### 判断标准

P2 strong pass：

```text
AUC_CP >= 0.75
PR_lift_CP >= 2.0
top273_CP_precision >= 0.75
ECE_CP <= 0.05
leave_dataset_auc_drop <= 0.07
feature_cost_q90 <= 0.20 ms
```

P2 weak pass：

```text
AUC_CP >= 0.68
PR_lift_CP >= 1.5
top273_CP_precision >= 0.60
```

P2 fail：

```text
best legal capacity model AUC_CP < 0.65
or top273_CP_precision < 0.55
or label_shuffle_auc is not clearly lower
```

若 P2 fail，不能继续 controller threshold 搜索，应进入 P6 primitive certificate reset。

### 可视化

```text
p2_feature_group_auc_prlift.svg
p2_topk_cp_precision_curve.svg
p2_calibration_curve_cp.svg
p2_leaveout_auc_drop_heatmap.svg
p2_knn_cp_concentration.svg
p2_capacity_vs_cost_pareto.svg
p2_feature_ablation_waterfall.svg
```

---

## P3：legal action-effect fingerprint factory

### 目标

实现第一性原理 action-effect features，而不是继续依赖 static state/payload marginal。

### 假设

H3：control-positive 的可观测 evidence 是 action 对当前 hard-tail / gradient / control update 的局部响应。

### Candidate fingerprints

#### AEF1：Gradient Descent Alignment

$$
AEF1(e)= -g_t^\top \Delta\theta_{func}(e)
$$

记录：

```text
grad_dot_delta_func
grad_dot_delta_adamw
func_descent_advantage = -g_dot_delta_func - (-g_dot_delta_adamw)
cos_func_negative_grad
cos_func_adamw
delta_func_over_adamw_norm
```

#### AEF2：Hard-tail logit response

对当前 batch 中 CE top-k 或 margin bottom-k 样本，估计：

$$
\Delta z_{tail}(e)=J_{tail}\Delta\theta_{func}(e)
$$

记录：

```text
tail_logit_true_class_delta_mean
tail_logit_true_class_delta_p10
tail_wrong_class_delta_p90
tail_margin_delta_mean
tail_margin_delta_p10
hard_tail_conflict_rate
```

#### AEF3：Linearized CE / margin certificate

$$
\widehat{\Delta CE}_{func}(e)=g_t^\top \Delta\theta_{func}(e)
$$

$$
\widehat{\Delta Margin}_{func}(e)=J_{margin}\Delta\theta_{func}(e)
$$

记录：

```text
linearized_CE_delta_func
linearized_CE_delta_adamw
linearized_CE_advantage
linearized_margin_delta_func
linearized_margin_advantage
linearization_confidence
```

#### AEF4：Curvature / trust certificate

用低成本 diagonal / block-diagonal / Gauss-Newton proxy：

$$
Penalty(e)=\frac{1}{2}\Delta\theta_{func}^\top \widehat{H}\Delta\theta_{func}
$$

记录：

```text
gn_proxy_penalty
diag_hessian_proxy_penalty
trust_ratio = predicted_gain / (penalty + eps)
curvature_risk_score
```

#### AEF5：Role / layer energy certificate

记录 functional delta 是否过度集中在 fragile roles：

```text
layer_energy_norms
role_energy_entropy
tail_role_energy_share
bridge_role_energy_share
functional_channel_entropy
payload_sparsity
```

#### AEF6：Control conflict certificate

估计 RealFunctional 与 controls 的冲突：

```text
predicted_func_gain
predicted_adamw_gain
predicted_bestlr_gain_proxy
predicted_noop_gain = 0
predicted_control_advantage
control_conflict_score
```

#### AEF7：Support-calibrated certificate

基于 calibration split 的 feature-space neighborhood：

```text
neighbor_count
effective_sample_size
EB_CP_mean
EB_CP_lcb
EB_bad_ucb
EB_long_risk_ucb
support_shift_score
```

### 必须记录

```text
action_id
candidate_id
feature_group
feature_id
feature_value
feature_cost_ms
feature_memory_ratio
commit_time_available
uses_outcome_at_commit
AUC_CP
PR_lift_CP
AUC_bad
AUC_null
AUC_long_risk
top273_CP_precision
top273_bad_rate
top273_null_rate
top273_long_risk_rate
leave_dataset_auc_drop
leave_horizon_auc_drop
leave_family_auc_drop
```

### 判断标准

P3 pass：

```text
at least one AEF group:
  AUC_CP >= 0.72
  PR_lift_CP >= 1.8
  top273_CP_precision >= 0.70
  top273_bad_rate <= 0.05
  feature_cost_q90 <= 0.20 ms
  uses_dataset_name = 0
```

P3 strong pass：

```text
AUC_CP >= 0.78
PR_lift_CP >= 2.5
top273_CP_precision >= 0.75
LDO/LSO stable
```

### 可视化

```text
p3_aef_auc_by_group.svg
p3_aef_topk_cp_precision.svg
p3_aef_bad_longrisk_tradeoff.svg
p3_gradient_alignment_vs_cp.svg
p3_tail_margin_response_vs_cp.svg
p3_trust_ratio_vs_cp.svg
p3_aef_cost_pareto.svg
```

---

## P4：cheap legal micro-probe redesign

### 目标

重新设计 probe。v9.3.5 的 probe 既接近随机又太贵，v9.3.6 只允许低成本局部响应 probe，不允许 full branch replay probe。

### Probe candidates

#### MP0：v9.3.5 PR5 reference

Expected fail：

```text
AUC_CP = 0.5031
cost_q90 = 2.4969 ms
```

#### MP1：LogitShadowProbe

只在当前 batch logits 上用 linearized output delta 估计 response，不更新权重。

```text
probe_tail_CE_delta
probe_margin_delta
probe_wrong_conf_delta
```

#### MP2：LastLayerShadowApply

只对最后映射层做 shadow delta，不完整 apply 全 payload。

```text
last_layer_CE_delta
last_layer_margin_delta
last_layer_tail_response
```

#### MP3：DiagonalGNProbe

使用 diagonal GN proxy 做 trust score。

```text
gn_predicted_gain
gn_penalty
gn_trust_ratio
```

#### MP4：HardTailTopKProbe

只在 hard-tail top-k examples 上计算 action response。

```text
topk_tail_gain
topk_tail_conflict
topk_margin_gain
```

#### MP5：ControlConflictProbe

只比较 functional vs AdamW predicted local response。

```text
func_vs_adamw_predicted_advantage
func_vs_noop_predicted_advantage
```

### 必须记录

```text
probe_id
probe_type
probe_action_count
AUC_CP
PR_lift_CP
AUC_bad
AUC_long_risk
top273_CP_precision
probe_cost_q50
probe_cost_q90
probe_cost_q99
probe_memory_ratio
extra_kernel_count
extra_sync_count
uses_future_outcome
uses_dataset_name
```

### 判断标准

P4 pass：

```text
AUC_CP >= 0.70
PR_lift_CP >= 2.0
top273_CP_precision >= 0.70
probe_cost_q90 <= 0.20 ms
extra_sync_count <= 1 per active step
memory_ratio <= 1.02
```

若 P4 fail：

```text
do not continue expensive probe route
route contribution = legal_probe_not_viable
```

### 可视化

```text
p4_probe_auc_cost_pareto.svg
p4_probe_topk_precision.svg
p4_probe_cost_distribution.svg
p4_probe_vs_aef_overlap.svg
```

---

## P5：cross-fitted control-positive controller

### 目标

如果 P2/P3/P4 找到足够 legal signal，则构造 dataset-agnostic controller。P5 不允许用 opaque feature pile；必须是 minimal, monotone, auditable controller。

### Controller candidates

#### C0：static v9.3.5 baseline

```text
score = StateNLL or PayloadLoadTime
```

Expected fail。

#### C1：single AEF controller

$$
Accept(e)=1 \iff AEF_i(e)\ge\tau.
$$

#### C2：certificate gate

$$
Accept(e)=1
\iff
LCB(V_{local}(e))>0
\land UCB(Bad(e))\le\tau_b
\land UCB(LongRisk(e))\le\tau_l
\land LCB(Support(e))\ge\tau_s
\land Cost(e)\le C_{max}.
$$

#### C3：monotone certificate score

$$
S(e)=
a_1 DescentCert(e)
+a_2 TailSafetyCert(e)
+a_3 ControlAdvCert(e)
+a_4 SupportLCB(e)
-a_5 CurvatureRisk(e)
-a_6 LongRiskUCB(e)
-a_7 Cost(e).
$$

Constraints：

```text
a_i >= 0
feature groups <= 7
no dataset_name
thresholds frozen on calibration split
```

#### C4：horizon-aware but dataset-agnostic gate

允许 horizon 作为 protocol variable：

$$
Accept(e,h)=1
\iff
S(e,h)\ge\tau_h
\land RiskUCB(e,h)\le\tau_b
$$

但禁止 dataset-specific thresholds。

#### C5：two-stage cheap + exact legal confirm

```text
Stage 1: cheap AEF / micro-probe prefilter
Stage 2: exact legal local-response confirm for borderline rows
```

Stage 2 不得使用 future outcome labels。

### Splits

```text
Seed folds:
  calibration seeds rotate
  heldout seeds rotate

Leave-dataset-out:
  calibrate on two datasets, evaluate third
  no dataset_name feature or branch

Leave-family-out:
  hold out high-volume family

Leave-horizon-out:
  train on h20/h80, evaluate h240；diagnostic only if official target is horizon-aware

Leave-stratum-out:
  hold out support / signal strata
```

### 必须记录

```text
controller_id
feature_set
feature_group_count
model_class
monotone_constraints
thresholds
calibration_split
heldout_split
accepted_count
coverage
CP_precision
primary_precision
bad_event_rate
null_rate
long_risk_rate
V_ctrl_mean
V_ctrl_lcb
beats_adamwparallel_rate
beats_bestlr_rate
beats_noop_rate
beats_random_rate
precision_lcb_CP
bad_event_ucb
coverage_lcb
support_balance_pass
accepted_family_count
accepted_stratum_count
max_family_share
max_stratum_share
feature_cost_q90
payload_apply_cost_estimate
uses_dataset_name
uses_future_outcome
```

### Decision pass

官方 v9.3.6 controller pass 必须同时满足：

$$
Coverage \in [0.03,0.15]
$$

$$
CPPrecision_{heldout}\ge0.75
$$

$$
PrimaryPrecision_{heldout}\ge0.90
$$

$$
BadEventRate_{heldout}\le0.05
$$

$$
NullRate_{heldout}\le0.15
$$

$$
V_{ctrl,LCB}>0
$$

$$
BeatRate_{AdamWParallel}\ge0.75
$$

$$
BeatRate_{bestLR}\ge0.75
$$

Support balance：

```text
accepted_family_count >= 32
accepted_stratum_count >= 5
max_family_share <= 0.50
max_stratum_share <= 0.60
coverage_heldout > 0 in all major folds
```

### 可视化

```text
p5_controller_cp_precision_coverage_frontier.svg
p5_value_risk_coverage_pareto.svg
p5_calibration_to_heldout_drift.svg
p5_leaveout_matrix.svg
p5_support_balance_sunburst.svg
p5_controller_ablation_waterfall.svg
```

---

## P6：observability failure autopsy and certificate primitive reset

### 目标

如果 P2/P3/P4/P5 未过，必须判断失败是 feature 不够、label horizon 不稳、还是 AP0 action 本身 legal opaque。不能只写 `legal_observability_gap`。

### Failure modes

```text
OF1-static_marginal_insufficient
OF2-action_effect_feature_weak
OF3-local_linearization_unreliable
OF4-horizon_instability_hidden
OF5-support_neighborhood_sparse
OF6-family_bucket_shift
OF7-control_positive_label_nonlocal
OF8-action_payload_opaque
OF9-probe_cost_infeasible
OF10-label_definition_misaligned_with_deploy_target
```

### 必须记录

```text
failure_mode
failure_fraction
missed_CP_count
false_positive_count
top273_CP_count
top273_bad_count
top273_long_risk_count
feature_values
oracle_CP_label
weak_CP
strong_CP
horizon_robust_CP
V_ctrl_by_horizon
family_id
bucket_id
horizon
dataset_diagnostic
seed
```

### Primitive reset branch

如果 AP0 被判定为 oracle-good but non-identifiable，则并行启动 AP1/AP2 certificate primitive smoke。

#### AP1：GradientCertifiedFunctionalAction

生成 action 时必须附带：

```text
descent_certificate
tail_safety_certificate
control_advantage_certificate
support_certificate
cost_certificate
```

#### AP2：TailSafeFunctionalAction

只生成对 hard-tail margin 有正向 certificate 的 actions：

$$
TailMarginCert(e)>0
\land CurvatureRisk(e)\le\tau_c.
$$

#### AP3：HorizonConsistentFunctionalAction

生成 action 时要求多个 legal horizon proxy 一致：

$$
S_{h20}(e)>0
\land S_{h80}(e)>0
\land S_{h240}(e)>0.
$$

#### AP4：AdamWConflictFreeFunctionalAction

只允许与 AdamW 方向不冲突或互补的 payload：

$$
\cos(\Delta\theta_{func},\Delta\theta_{AdamW})\ge\tau_{cos}
$$

或：

$$
ProjectedGain_{func} - ProjectedGain_{AdamW} > 0.
$$

### AP smoke 必须记录

```text
primitive_id
candidate_count
action_count
certificate_pass_rate
weak_CP_oracle_coverage
strong_CP_oracle_coverage
horizon_robust_CP_coverage
static_observability_auc
certificate_auc_CP
certificate_top273_CP_precision
payload_apply_cost_estimate
```

### P6 route

```text
if AP0 legal identifiability fail and AP1 certificate smoke pass:
  route = R6-CertificatePrimitivePromising

if AP0 legal identifiability fail and AP1/AP2 fail:
  route = R6-ActionPrimitiveRedesignRequired

if AP0 legal identifiability pass:
  continue P5/P8 system integration
```

### 可视化

```text
p6_observability_failure_sankey.svg
p6_missed_cp_feature_distribution.svg
p6_ap0_vs_ap1_certificate_auc.svg
p6_primitive_density_vs_observability.svg
p6_horizon_robust_density_by_primitive.svg
```

---

## P7：payload apply runtime closure

### 目标

把 v9.3.5 的 payload apply diagnostic 从 `step_ratio_q90 = 2.0687` 推进到 actual selected-controller path `step_ratio_q90 <= 1.50`。

### Runtime candidates

#### RT0：v9.3.5 oracle-mask payload diagnostic

Expected fail but reference：

```text
payload_apply_time_ms_q90 = 0.762382
step_ratio_q90 = 2.068737
```

#### RT1：SparsePayloadApply

只 apply nonzero / changed role slices：

```text
payload_sparsity_used = 1
coalesced_indices = 1
```

#### RT2：FusedAdamWFunctionalApply

把 AdamW update 和 functional payload apply 融合进同一 update kernel：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{AdamW}
+
\Delta\theta_{func}
$$

不再做两次 parameter write。

#### RT3：LazyFunctionalApplyBuffer

functional payload 先合入 optimizer delta buffer，在下一次 base update 中 flush。

```text
lazy_buffer_used = 1
flush_error_audit = 1
```

#### RT4：Per-layer Coalesced Apply

按 layer / role 聚合 payload，避免 per-action scatter。

```text
layer_coalescing_used = 1
per_layer_apply_kernel_count <= num_layers
```

#### RT5：Accepted Action Merge

同一 active step 多个 accepted actions 先合并 payload：

$$
\Delta\theta_{func,total}=\sum_{e\in AcceptedStep}\alpha_e\Delta\theta_{func}(e).
$$

#### RT6：Hybrid fused sparse lazy apply

组合 RT1-RT5，作为 P8 system candidate。

### 必须记录

```text
runtime_candidate_id
payload_apply_impl
controller_id
oracle_mask_used
payload_apply_used
accepted_count
candidate_count
step_count
active_step_count
feature_compute_time_ms_q90
score_accept_time_ms_q90
payload_lookup_time_ms_q90
payload_pack_time_ms_q90
payload_apply_time_ms_q90
fused_update_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
extra_kernel_count
extra_sync_count
parameter_write_count
payload_apply_error_linf_max
payload_apply_error_relative_max
payload_apply_cosine_min
```

### 判断标准

P7 runtime pass：

```text
oracle_mask_used = 0 for official candidate
payload_apply_used = 1
payload_apply_error_linf_max <= 1e-8
payload_apply_cosine_min >= 0.999999
payload_apply_time_ms_q90 <= 0.35
total_step_ratio_q90 <= 1.50
memory_ratio <= 1.05
extra_sync_count <= 1 per active step
```

Diagnostic pass：

```text
payload_apply_time_ms_q90 <= 0.45
step_ratio_q90 <= 1.70
```

### 可视化

```text
p7_payload_apply_runtime_waterfall.svg
p7_runtime_candidate_comparison.svg
p7_payload_apply_error_distribution.svg
p7_parameter_write_reduction.svg
p7_step_ratio_q90_by_impl.svg
```

---

## P8：system legal controller integration

### 目标

组合 P5 selected controller 与 P7 payload apply runtime，生成第一个 v9.3.x control-positive-aware system candidate。

### 必须记录

```text
system_candidate_id
controller_id
runtime_candidate_id
primitive_id
feature_set
thresholds
official_eligible
control_positive_controller_pass
payload_apply_runtime_pass
system_legal_controller_pass
candidate_count
accepted_count
coverage
CP_precision
primary_precision
bad_event_rate
null_rate
long_risk_rate
V_ctrl_mean
V_ctrl_lcb
beats_adamwparallel_rate
beats_bestlr_rate
step_ratio_q90
memory_ratio
feature_cost_q90
payload_apply_time_ms_q90
uses_dataset_name
uses_future_outcome
uses_test_or_validation
diagnostic_promoted_to_official
source_measured_gap_used
formula_proxy_used
```

### 判断标准

P8 pass：

```text
official_eligible = 1
control_positive_controller_pass = 1
payload_apply_runtime_pass = 1
system_legal_controller_pass = 1
uses_dataset_name = 0
uses_future_outcome = 0
diagnostic_promoted_to_official = 0
```

Decision gates 同 P5。

Runtime gates 同 P7。

### 可视化

```text
p8_system_gate_dashboard.svg
p8_quality_cost_pareto.svg
p8_controller_runtime_joint_frontier.svg
p8_official_vs_diagnostic_boundary.svg
```

---

## P9：parallel diagnostic causal scouts

### 目标

加快实验，但严格隔离 diagnostic。P9 只帮助判断后续方向，不允许影响 P5/P8 阈值。

### 前置条件

```text
P5 has at least weak controller candidate
or P6 has AP1/AP2 certificate primitive candidate
```

### 设置

```text
controllers:
  top2 AP0 AEF controllers
  top2 AP1/AP2 certificate controllers
  StableAccept negative control
  random accepted control

datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

horizons:
  20,80,240

branches:
  RealFunctional
  AdamWParallel
  bestLR
  NoOp
  Random
  ShuffledFunctionalPayload
  ShuffledCertificate
  ShuffledAEFScore
```

### 必须记录

```text
controller_id
primitive_id
dataset
seed
horizon
branch
accepted_count
coverage
CP_precision
V_ctrl_mean
V_ctrl_lcb
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
beats_adamwparallel
beats_bestlr
task_safe
status = diagnostic_not_official
diagnostic_used_for_threshold = 0
```

### 判断标准

P9 promising：

```text
RealFunctional beats AdamWParallel in >= 50% macro slices
RealFunctional beats bestLR in >= 50% macro slices
task_safe holds
shuffled controls do not match RealFunctional
```

P9 结果不得改变 P5/P8 thresholds。

### 可视化

```text
p9_diagnostic_macro_beat_matrix.svg
p9_branch_delta_pareto.svg
p9_shuffle_control_matrix.svg
p9_task_safety_slices.svg
```

---

## P10：leave-out and official paired replay boundary

### 目标

只有 P8 pass 后打开 official。v9.3.6 可以准备代码和 diagnostic，不允许提前 official。

### Official prerequisites

```text
P5 controller pass = 1
P7 runtime pass = 1
P8 system legal controller pass = 1
no fake/proxy/offload/teacher/loss violation
```

### Leave-out settings

```text
Leave-dataset-out:
  calibrate on two datasets, evaluate third
  no dataset_name route

Leave-family-out:
  hold out high-volume families

Leave-horizon-out:
  diagnostic unless horizon-aware target is official

Leave-stratum-out:
  hold out support/signal strata
```

### Official paired replay settings

```text
branches:
  RealFunctional
  AdamWParallel
  bestLR
  NoOp
  Random
  ShuffledFunctionalPayload
  ShuffledCertificate
  ShuffledAEFScore
  ShuffledRuntimePath
```

### 必须记录

```text
split_type
heldout_entity
controller_id
primitive_id
runtime_candidate_id
coverage
CP_precision
bad_event_rate
null_rate
V_ctrl_lcb
beats_adamwparallel_rate
beats_bestlr_rate
task_safe
step_ratio_q90
memory_ratio
dataset_name_used
```

### 判断标准

LDO pass：

```text
2/3 held-out datasets pass decision gates
no held-out dataset bad_event_rate > 0.10
coverage > 0 in all held-out datasets
dataset_name_used = 0
```

Paired replay pass：

$$
BeatRate_{Real \, vs \, AdamWParallel} \ge 0.60
$$

$$
BeatRate_{Real \, vs \, bestLR} \ge 0.60
$$

Task safety：

$$
Acc_{Real} \ge Acc_{AdamW} - 0.005
$$

### 可视化

```text
p10_leave_dataset_matrix.svg
p10_leave_family_matrix.svg
p10_official_paired_replay_pareto.svg
p10_shuffle_control_matrix.svg
```

---

# 8. 并行执行计划

## Batch A：立即并行

```text
A1: P0 v9.3.5 boundary reanalysis
A2: P1 CP label decomposition
A3: P7 payload apply RT1/RT2 microbench
A4: P2 static feature capacity baseline rerun
```

目标：一天内判断 CP label structure 和 runtime hard gap。

## Batch B：action-effect features 与 runtime 并行

```text
B1: P3 AEF1/AEF2/AEF3 implementation
B2: P4 MP1/MP2/MP4 cheap probe implementation
B3: P7 RT3/RT4/RT5 payload apply implementation
B4: P2 legal capacity diagnostic with AEF groups
```

目标：不等 controller 完成，先知道 legal signal 是否存在，payload apply 是否可降。

## Batch C：controller 与 primitive reset 并行

```text
C1: P5 cross-fitted AP0 AEF controller
C2: P6 AP0 observability failure autopsy
C3: P6 AP1/AP2 certificate primitive smoke
C4: P9 diagnostic causal scout for top candidates
```

目标：AP0 和 AP1 不串行等待，避免在 AP0 non-identifiable 上浪费多轮。

## Batch D：system integration

```text
D1: P8 AP0 selected-controller + RT survivor
D2: P8 AP1/AP2 certificate-controller + RT survivor
D3: P10 leave-out preparation
```

## Batch E：official downstream only after P8 pass

```text
E1: official LDO / LSO
E2: official paired replay
E3: short-run / full-run / robustness
```

---

# 9. Required artifacts

```text
run_manifest.json
contract_audit_v9360.csv
provenance_audit_v9360.csv

p0_v9350_boundary_reanalysis.csv
p1_cp_label_decomposition_v9360.csv
cp_horizon_structure_trace_v9360.csv

p2_legal_observability_capacity_audit.csv
legal_feature_capacity_trace_v9360.csv
feature_cost_trace_v9360.csv

p3_action_effect_fingerprint_factory.csv
aef_feature_trace_v9360.csv
aef_cost_trace_v9360.csv

p4_cheap_legal_microprobe_redesign.csv
microprobe_trace_v9360.csv
microprobe_cost_trace_v9360.csv

p5_crossfitted_control_positive_controller.csv
controller_frontier_trace_v9360.csv
controller_ablation_trace_v9360.csv
leaveout_diagnostic_trace_v9360.csv

p6_observability_failure_autopsy.csv
observability_failure_trace_v9360.csv
certificate_primitive_smoke_v9360.csv
certificate_feature_trace_v9360.csv

p7_payload_apply_runtime_closure.csv
payload_runtime_component_trace_v9360.csv
payload_apply_error_trace_v9360.csv

p8_system_legal_controller_v9360.csv
system_controller_trace_v9360.csv

p9_diagnostic_causal_scout.csv
diagnostic_causal_scout_trace_v9360.csv

p10_leaveout_and_official_paired_replay_boundary.csv
official_paired_replay_trace_v9360.csv

route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

---

# 10. 主要图表清单

v9.3.6 必须至少生成：

```text
figures/p0_oracle_vs_observability_gap.svg
figures/p0_payload_runtime_gap.svg
figures/p1_weak_strong_robust_overlap_venn.svg
figures/p1_V_ctrl_horizon_heatmap.svg
figures/p2_feature_group_auc_prlift.svg
figures/p2_topk_cp_precision_curve.svg
figures/p2_leaveout_auc_drop_heatmap.svg
figures/p3_aef_auc_by_group.svg
figures/p3_tail_margin_response_vs_cp.svg
figures/p3_aef_cost_pareto.svg
figures/p4_probe_auc_cost_pareto.svg
figures/p5_controller_cp_precision_coverage_frontier.svg
figures/p5_value_risk_coverage_pareto.svg
figures/p6_observability_failure_sankey.svg
figures/p6_ap0_vs_ap1_certificate_auc.svg
figures/p7_payload_apply_runtime_waterfall.svg
figures/p7_step_ratio_q90_by_impl.svg
figures/p8_system_gate_dashboard.svg
figures/p9_diagnostic_macro_beat_matrix.svg
```

---

# 11. Failure taxonomy

```text
F1_contract_violation
F2_fake_proxy_or_cpu_offload
F3_dataset_tuning_detected
F4_future_outcome_used_for_feature
F5_test_validation_used_at_commit
F6_full_control_outcome_regression
F7_CP_label_quality_fail
F8_panel_representativeness_fail
F9_weak_CP_density_regression
F10_horizon_robust_CP_too_sparse
F11_static_feature_unobservable
F12_action_effect_feature_unobservable
F13_microprobe_unobservable
F14_microprobe_cost_infeasible
F15_local_linearization_unreliable
F16_support_neighborhood_sparse
F17_horizon_instability_hidden
F18_AP0_oracle_good_but_nonidentifiable
F19_certificate_primitive_density_fail
F20_certificate_feature_unobservable
F21_no_dataset_agnostic_controller
F22_CP_precision_fail
F23_coverage_fail
F24_bad_event_fail
F25_null_rate_fail
F26_long_risk_fail
F27_Vctrl_lcb_fail
F28_leave_dataset_out_fail
F29_leave_family_out_fail
F30_leave_stratum_out_fail
F31_payload_apply_runtime_fail
F32_payload_apply_error_fail
F33_step_ratio_fail
F34_memory_ratio_fail
F35_oracle_mask_promoted_to_official
F36_no_payload_smoke_promoted_to_official
F37_system_integration_fail
F38_paired_replay_control_equivalent
F39_shuffle_control_pass
F40_short_run_task_drop
F41_full_run_no_macro_or_hard_stratum_gain
F42_strong_baseline_explains_gain
F43_external_not_ready
```

---

# 12. Route decision

```text
R1-BoundaryReanalyzed:
  v9.3.5 metrics reproduced and gate gaps computed.

R2-CPLabelDecomposed:
  weak / strong / horizon-robust CP labels complete.

R3-LegalObservabilityCapacityPass:
  legal feature sigma-algebra shows enough signal.

R4-LegalObservabilityCapacityFail:
  even capacity diagnostic cannot see CP frontier.

R5-ActionEffectFingerprintPass:
  AEF features identify CP frontier at deployable cost.

R6-ActionEffectFingerprintFail:
  AEF features fail; AP0 likely opaque.

R7-MicroProbePass:
  cheap legal micro-probe identifies CP frontier within cost.

R8-MicroProbeFail:
  probe route not viable.

R9-AP0ControllerPass:
  AP0 cross-fitted CP controller passes decision gates.

R10-AP0ControllerFail:
  AP0 features have signal but no stable controller.

R11-CertificatePrimitivePromising:
  AP1/AP2 certificate primitive has oracle density and observability.

R12-CertificatePrimitiveFail:
  primitive reset smoke fails.

R13-PayloadApplyRuntimePass:
  actual payload-apply runtime reaches step_ratio_q90 <= 1.50.

R14-PayloadApplyRuntimeFail:
  payload apply remains above envelope.

R15-SystemLegalControllerPass:
  decision + runtime + contracts pass.

R16-SystemLegalControllerFail:
  decision/runtime integration fails.

R17-LeaveOutPass:
  controller generalizes across held-out datasets/families/strata.

R18-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R19-PairedReplayFail:
  system legal but functional update control-equivalent.

R20-ExternalReadyCandidate:
  short/full/robustness gates ready to open.
```

`route_decision.json` 必须记录：

```text
route
base_candidate
primitive_id
candidate_count
action_count
full_control_outcome_ready
weak_CP_count
weak_CP_coverage
strong_CP_count
strong_CP_coverage
horizon_robust_CP_count
horizon_robust_CP_coverage
best_static_auc_CP
best_AEF_group
best_AEF_auc_CP
best_AEF_pr_lift
best_microprobe_id
best_microprobe_auc_CP
best_microprobe_cost_q90
legal_observability_capacity_pass
action_effect_fingerprint_pass
microprobe_pass
AP0_controller_pass
certificate_primitive_pass
controller_id
coverage_heldout
CP_precision_heldout
primary_precision_heldout
bad_event_heldout
null_rate_heldout
long_risk_rate_heldout
V_ctrl_lcb
beats_adamwparallel_rate
beats_bestlr_rate
support_balance_pass
payload_apply_impl
payload_apply_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
payload_apply_error_linf_max
payload_apply_cosine_min
payload_apply_runtime_pass
system_legal_controller_pass
leave_dataset_out_pass
leave_family_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
fake_data_used
proxy_row_used
cpu_offload_used
uses_dataset_name_for_controller
uses_future_outcome_for_features
uses_test_or_validation
diagnostic_promoted_to_official
oracle_mask_used_for_official
no_payload_smoke_promoted_to_official
```

---

# 13. 停止条件

## Minimum diagnostic success

```text
P0 boundary reanalysis pass
P1 CP label decomposition pass
P2 legal observability capacity measured
P3 AEF features measured
P4 microprobe measured
P5 controller measured if P2/P3/P4 pass
P6 failure autopsy measured if P5 fail
P7 payload apply runtime measured
no fake/proxy/offload/teacher/loss violation
```

## Pivot to AP1/AP2 certificate primitive

触发条件：

```text
weak CP oracle pass
but P2 legal observability capacity fail
and P3 AEF fail
and P4 cheap microprobe fail
```

解释：

```text
AP0 has oracle-good actions but no deployable legal observability.
Do not continue AP0 threshold search.
```

## Pivot to runtime redesign

触发条件：

```text
P5 controller pass
but P7 payload apply runtime fail
```

解释：

```text
Decision path is promising, but system envelope is blocked by payload apply.
Do not open official paired replay.
```

## Open official downstream

唯一条件：

```text
P8 system_legal_controller_pass = 1
and no diagnostic promoted to official
and no dataset tuning
```

---

# 14. 最终解释规则

## Case A：P5 pass + P7 pass + P8 pass

可以声明：

```text
v9.3.6 found a legal control-positive controller candidate and measured payload-apply system path.
```

但仍不能声明 full success，必须进入 LDO/LSO 和 official paired replay。

## Case B：P2/P3/P4 fail

必须声明：

```text
AP0 current action primitive is oracle-good but legally non-identifiable under tested commit-time observables.
```

下一步应推进 AP1/AP2 certificate-producing primitive，不继续 AP0 feature patching。

## Case C：P2/P3 pass but P5 fail

必须声明：

```text
Legal signal exists, but no stable dataset-agnostic controller frontier has been found.
```

下一步应做 controller minimality / support redesign，而不是扩 feature pile。

## Case D：P5 pass but P7 fail

必须声明：

```text
Decision route may be viable, but payload apply prevents system legality.
```

下一步只修 runtime，不能打开 official paired replay。

## Case E：P8 pass but P10 fail

必须声明：

```text
System-legal local controller exists, but generalization / causal advantage is not yet established.
```

不得通过 dataset-specific tuning 补救。

---

# 15. 最终建议

v9.3.6 的一句话策略是：

$$
\boxed{
\text{不要再问好 action 有没有；现在要问好 action 的 legal certificate 在哪里，以及 payload apply 能不能真实进 envelope。}
}
$$

如果 v9.3.6 成功，它应该给出两个结论之一：

```text
1. AP0 action 可以被 legal action-effect fingerprint 识别，并能进入 system controller；
2. AP0 action oracle-good but non-identifiable，必须转向 certificate-producing action primitive。
```

这两个结论都比继续 “再加一个 feature / 再调一个 threshold” 更接近项目本质。
