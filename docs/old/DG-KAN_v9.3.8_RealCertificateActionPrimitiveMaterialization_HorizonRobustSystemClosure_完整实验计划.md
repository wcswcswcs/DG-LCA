# DG-KAN v9.3.8 Real Certificate-Producing Action Primitive Materialization / Horizon-Robust System Closure 完整实验计划

> 本计划基于 v9.3.7 `Certificate-Producing Action Primitive / Horizon-Robust Runtime Closure` 的真实执行结果制定。  
> v9.3.7 的 terminal route 是：
>
> ```text
> route = R6-ActionPrimitiveRedesignRequired
> base_candidate = LQ-t2-h256
> success_v9370_strict_purekan_functional = False
> success_v9370_full_functional = False
> success_v9370_external_ready = False
> primary_blocker = certificate_producing_action_primitive_not_materialized
> next_required_implementation = implement_real_AP1_AP4_payload_generators_with_commit_time_certificate_tensors
> ```
>
> v9.3.8 的核心判断是：v9.3.7 不是证明 certificate primitive 失败，而是证明 **certificate primitive 还没有被实现**。本轮不能再做 AP0 threshold/probe patch，也不能继续生成 AP0 diagnostic certificate 后 gate-block。v9.3.8 必须第一次让 AP1/AP2/AP3/AP4 真实生成 payload、证书、action apply trace、matched-control outcome 与 selected-runtime trace。

---

# 0. 执行摘要

v9.3.7 的真实进展有两点：

```text
1. AP0 threshold/probe patch 已经按计划停止；
2. certificate schema / legality audit 可以落盘，说明证书字段、audit 结构和 no-fake 纪律可以建立。
```

但 v9.3.7 没有进入真正的 action primitive 实验。关键数据是：

```text
certificate_schema_rows_complete = 1
legality_audit_pass = 1
diagnostic_certificate_rows = 3456
primitive_count = 4
primitive_materialized_count = 0
certificate_pass_count = 0
generated_action_count_total = 0
primitive_generation_pass = 0
ap_smoke_outcome_pass = 0
full_ap_frontier_pass = 0
certificate_controller_pass = 0
selected_payload_runtime_pass = 0
official_eligible = 0
system_legal_controller_pass = 0
```

这说明 schema 可以诊断 AP0，但没有生成新的 AP action。AP1/AP2/AP3/AP4 的 `generated action` 全部为 `0`。P4 中所谓 primitive outcome summary 只来自 AP0 existing outcomes diagnostic，不是新 payload outcome；因此不能判断 AP1/AP2/AP3/AP4 是否有价值。

v9.3.8 的总体目标不是“把 certificate schema 再修一遍”，也不是“在 AP0 上换更多 certificate 条件”，而是：

$$
\boxed{
\text{实现真实 certificate-producing action primitives，并验证证书是否能在 commit 前预测 value/risk/horizon/cost。}
}
$$

更具体地说，v9.3.8 要让下面这个对象第一次完整存在：

$$
G_{APk}(s_t, b_t, g_t, \Delta\theta_{AdamW})
\rightarrow
(\Delta\theta_{APk}, Cert_{APk}, Hash_{APk}).
$$

其中 $G_{APk}$ 是 AP1/AP2/AP3/AP4 的真实生成器，$\Delta\theta_{APk}$ 是可 apply 的 functional payload，$Cert_{APk}$ 是 commit-time certificate tensor，$Hash_{APk}$ 是 durable payload / certificate hash。

只有当 action payload、certificate、apply replay、matched-control outcome 和 runtime 都落盘后，才能进入 controller、LDO/LSO、paired replay 和 short/full-run。

---

# 1. 独立实验结果判断

## 1.1 v9.3.7 有进展，但不是 functional 进展

v9.3.7 的进展属于 **boundary clarification / implementation audit**。它证明：

```text
AP0 patch route 已经应该停止；
certificate schema 可以写；
legality audit 可以写；
no-fake / no-proxy / no-offload 纪律没有回退；
但 repo 当前没有 AP1-AP4 真实 payload generator。
```

这不是坏结果。它避免了一个很危险的错觉：把 AP0 diagnostic certificate 当成 AP1/AP2/AP3/AP4 action primitive。v9.3.7 明确阻止了这个错误。

但是，这一轮没有产生新的 functional action。`generated_action_count_total = 0`，所以它没有测到：

```text
AP1 是否真的能产生 tail-safe action；
AP2 是否真的能产生 AdamW-residual advantage；
AP3 是否真的能产生 horizon-robust memory action；
AP4 是否真的能产生 low-rank edge action；
certificate 是否能预测 CP / strong CP / horizon-robust CP；
selected AP runtime 是否能进入 step_ratio_q90 <= 1.50。
```

因此不能把 v9.3.7 说成 “certificate primitive 失败”。更准确的说法是：

$$
\boxed{
\text{certificate primitive experiment 尚未开始；v9.3.7 只证明 primitive generator 缺失。}
}
$$

## 1.2 AP0 的结论已经足够清楚，不应再回头修 AP0

v9.3.6 已经证明 AP0 的 full weak CP frontier 存在：

```text
weak_CP_row_count = 893
weak_CP_coverage = 0.0984347442680776
weak_CP_coverage_lcb = 0.09247342078415305
weak_CP_V_ctrl_lcb = 0.2833107634852314
strong_CP_row_count = 353
strong_CP_coverage = 0.03891093474426808
```

但 AP0 horizon-robust density 很低：

```text
horizon_robust_CP_action_count = 13
horizon_robust_CP_coverage = 0.004298941798941799
short_only_CP_action_count = 201
long_risk_action_count = 294
```

同时，AP0 legal observability 很弱：

```text
best_legal_capacity_feature = F4-StateNLL
best_legal_capacity_auc_CP = 0.6151712728688206
best_top273_CP_precision = 0.12454212454212454
best_aef_feature = AEF6-ProbeReliability
best_aef_auc_CP = 0.5030687962043473
best_aef_top273_CP_precision = 0.28205128205128205
best_aef_cost_q90 = 2.4969042278826237 ms
```

v9.3.7 又检查了 24 个 AP0 feature，best AUC 仍只有 `0.6151712728688206`，best top273 CP precision 仍只有 `0.28205128205128205`，并且 `ap0_stop_condition = 1`。

这意味着 v9.3.8 不应再执行 AP0 threshold search。AP0 的问题不是没有 oracle-good rows，而是：

$$
\boxed{
\text{AP0 的好 action 是 posthoc 可见、commit-time 不可见，并且 horizon-fragile。}
}
$$

## 1.3 v9.3.7 的 diagnostic certificate rows 不能证明 AP1/AP2/AP3/AP4 弱

v9.3.7 中有：

```text
diagnostic_certificate_rows = 3456
primitive_count = 4
diagnostic_certificate_pass_ignoring_payload_count = 51
AP1 diagnostic pass ignoring payload = 4
AP2 diagnostic pass ignoring payload = 11
AP3 diagnostic pass ignoring payload = 36
AP4 diagnostic pass ignoring payload = 0
```

这些 rows 是把 AP1/AP2/AP3/AP4 的 certificate 条件套在 AP0 measured rows 上做诊断，不是新 AP payload。P4 的 weak CP precision：

```text
AP1 = 0.25
AP2 = 0.2727272727
AP3 = 0.25
AP4 = 0.0
```

只能说明：

```text
当前 certificate 条件直接套 AP0 旧 action 也不强；
不能说明 AP1/AP2/AP3/AP4 真实生成器生成的 action 不强；
也不能说明 certificate-producing primitive 方向失败。
```

如果下一轮继续用 AP0 old outcomes 来评估 AP1-AP4 证书，那就是在做 posthoc certificate filtering，而不是 certificate-producing action primitive。

## 1.4 当前真正卡在哪里

当前 blocker 只有一个主轴：

$$
\boxed{
\text{缺真实 AP payload generator 与 commit-time certificate tensor。}
}
$$

它包含五个子 blocker：

```text
1. AP1/AP2/AP3/AP4 生成器没有 materialize；
2. 新 AP payload 没有 durable tensor/hash；
3. 新 AP certificate tensors 没有和 payload 绑定；
4. 新 AP action 没有 action apply replay error；
5. 新 AP payload outcome 没有 branch/horizon materialize。
```

只要这五项没有完成，下面所有东西都不应打开：

```text
certificate calibration；
full AP frontier；
certificate controller；
selected primitive runtime；
system legal controller；
LDO/LSO；
official paired replay；
short/full run；
external-ready claim。
```

## 1.5 是否还在正确道路上

方向仍然正确，但执行必须从 audit 转向 implementation。

正确的路线是：

```text
StableAccept final-controller 失败
-> AP0 oracle-good but non-identifiable
-> certificate-producing action primitive
-> real AP payload generation
-> matched-control outcome
-> certificate controller
-> selected runtime
-> LDO/LSO
-> paired replay
-> short/full run
```

错误的路线是：

```text
继续 AP0 threshold/probe patch；
继续只写 certificate schema；
继续用 AP0 old outcome 代表 AP1/AP2/AP3/AP4；
继续让 generated_action_count_total = 0；
继续把 diagnostic pass ignoring payload 写成 signal；
继续打开 controller/runtime/downstream。
```

v9.3.7 已经把错误路线挡住了。v9.3.8 必须把正确路线的第一个硬实现补上。

---

# 2. v9.3.8 总体目标

v9.3.8 的总体目标是：

$$
\boxed{
\text{实现并验证真实 AP1/AP2/AP3/AP4 certificate-producing action primitive。}
}
$$

最低有效推进目标：

```text
1. AP1/AP2/AP3/AP4 至少一个 primitive 能生成非零 action payload；
2. 每个 generated action 都有 durable payload tensor、payload hash、certificate tensor、certificate hash；
3. action apply replay error 全量测量；
4. new AP smoke outcomes 覆盖 branch × horizon，不再使用 AP0 old outcome 代替；
5. 至少一个 primitive 的 certificate-pass rows 在 weak CP / strong CP / horizon-risk 上优于 AP0 diagnostic certificate baseline；
6. selected primitive runtime 被 measured，而不是 microbench-only；
7. route 能明确落在：generator fail、certificate fail、outcome fail、runtime fail、controller fail 或 system pass。
```

v9.3.8 的强目标：

```text
primitive_generation_pass = 1
ap_smoke_outcome_pass = 1
certificate_calibration_pass = 1
full_ap_frontier_pass = 1
certificate_controller_pass = 1
selected_payload_runtime_pass = 1
system_legal_controller_pass = 1
```

但强目标不是最低成功。v9.3.8 更重要的是不再停在 audit-only。

---

# 3. v9.3.8 明确不做什么

本轮不做：

```text
1. 不继续 AP0 threshold / probe / feature search；
2. 不把 AP0 diagnostic certificate rows 写成 AP1-AP4 primitive pass；
3. 不在 generated_action_count_total = 0 时继续 P4-P9；
4. 不把 schema legality pass 写成 primitive pass；
5. 不用 dataset_name 选择 primitive、threshold、runtime route；
6. 不用 validation/test metric at commit time；
7. 不用 teacher / distillation / auxiliary loss / loss modification；
8. 不用 old AP0 outcome 替代 new AP payload outcome；
9. 不用 formula proxy 或 source measured gap 进入 official；
10. 不在 selected controller 缺失时把 runtime microbench 写成 runtime pass。
```

允许做：

```text
1. dataset-level diagnostics，但不能 dataset-specific tuning；
2. AP primitive smoke panel、30% panel、full panel 的分层 materialization；
3. branch/horizon matched-control materializer；
4. diagnostic causal scout，但不得影响 controller threshold；
5. online runtime microbench + selected path runtime 双测量；
6. AP generator implementation race，并行实现 AP1/AP2/AP3/AP4。
```

---

# 4. 核心假设

## H0：AP0 patch route 应停止

H0 认为 AP0 已经 oracle-good but non-identifiable，继续 AP0 feature search 只会变成新的 patch treadmill。

H0 成立标准：

```text
ap0_stop_condition = 1
best_ap0_feature_auc_CP <= 0.65
best_ap0_top273_CP_precision <= 0.35
no AP0 threshold search executed
no AP0 diagnostic promoted to official
```

H0 失败标准：

```text
预注册 AP0 feature 在 LDO/LSO 下达到 official controller gate。
```

如果 H0 失败，可以恢复 AP0 official consideration；但 v9.3.7 已显示目前不支持这个方向。

## H1：真实 AP payload generator 是当前第一 blocker

H1 认为 v9.3.7 的 terminal blocker 不是 certificate schema，而是 generator 缺失。

H1 成立标准：

```text
v9.3.8 前置 audit 复现：
  primitive_materialized_count = 0
  generated_action_count_total = 0
  certificate_schema_contract_pass = 0
```

H1 completion 标准：

```text
至少一个 AP primitive:
  generated_action_count > 0
  durable_payload_written = 1
  payload_hash_missing_count = 0
  certificate_tensor_written = 1
  certificate_hash_missing_count = 0
  action_apply_error_measured = 1
  action_apply_error_linf_max <= 1e-6
```

## H2：certificate 必须在生成时产生，而不是事后贴标签

H2 认为 certificate-producing primitive 的核心不是“生成 action 后用 feature 过滤”，而是 action generator 本身必须根据 certificate 生成或约束 payload。

H2 成立标准：

```text
certificate_commit_time = before_outcome_materialization
certificate_hash_bound_to_payload_hash = 1
uses_outcome_at_commit = 0
uses_future_step = 0
uses_dataset_name = 0
```

H2 失败标准：

```text
certificate 需要 AP outcome label 才能计算；
certificate 只在 materialized outcome 后生成；
certificate 不绑定 payload hash；
certificate 与 action generator 无关。
```

## H3：真实 AP primitive 应提升 horizon-robust density，而不是只复制 AP0 weak CP

H3 认为 AP0 weak CP coverage 足够，但 horizon-robust density 极低。AP1/AP2/AP3/AP4 的价值不应只看 weak CP，而应看 strong CP、horizon-robust CP、long-risk。

H3 weak 成立标准：

```text
APk certificate-pass rows:
  weak_CP_precision >= 0.40
  long_risk_rate <= 0.15
  branch_horizon_completion_rate = 1.0
```

H3 strong 成立标准：

```text
APk certificate-pass rows:
  weak_CP_precision >= 0.75
  strong_CP_precision >= 0.40
  horizon_robust_CP_precision >= 0.10
  long_risk_rate <= 0.05
  support_balance_pass = 1
```

H3 failure 标准：

```text
所有 APk certificate-pass rows weak_CP_precision <= AP0 diagnostic baseline + 0.05；
或 long_risk_rate 明显高；
或 horizon completion 不完整。
```

## H4：certificate 应该是 sufficient statistic 的雏形，而不是 opaque score

H4 认为 certificate 不能再变成一个大杂烩 score。它必须分解成 value、bad risk、null risk、support、horizon、cost 六个可审计字段。

形式：

$$
Accept_{cert}(e)=1
\iff
LCB(V_{cert}(e))>0
\land UCB(B_{cert}(e))\le \tau_b
\land UCB(N_{cert}(e))\le \tau_n
\land LCB(S_{cert}(e))\ge \tau_s
\land H_{risk}(e)\le \tau_h
\land C(e)\le C_{max}.
$$

H4 成立标准：

```text
每个 AP action 都有：
  cert_value_lcb
  cert_bad_ucb
  cert_null_ucb
  cert_support_lcb
  cert_horizon_risk
  cert_cost_estimate
  cert_descent_margin
  cert_tail_safety_margin
  cert_norm_bound
```

H4 failure 标准：

```text
certificate 只有一个 opaque scalar；
certificate 无法拆解 failure；
certificate 无法和 matched-control outcome 对齐。
```

## H5：selected primitive runtime 必须和 selected controller 同路测量

H5 认为 v9.3.6 的 `RT3-LastLayerOnlyDiagnostic` 是有希望的工程线索，但不是 official runtime。v9.3.8 必须测 selected AP primitive 的完整 online path：

```text
feature/certificate compute
payload generation
payload apply
base step
accept decision
```

H5 pass：

```text
runtime_mode = online_sequential_selected_primitive
selected_controller_used = 1
selected_payload_apply_used = 1
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
payload_apply_error_linf_max <= 1e-6
```

H5 diagnostic pass：

```text
selected path not available, but per-primitive runtime measured；
step_ratio_q90 <= 2.00；
waterfall identifies exact dominant component。
```

---

# 5. AP primitive designs

## 5.1 AP1：LastEdgeLinearizedTailSafeCertificate

### 核心思想

AP1 只在 last edge / last layer equivalent 子空间生成 action。它牺牲表达范围，换取三件事：

```text
1. action apply 快；
2. linearized certificate 更可信；
3. tail safety 更容易约束。
```

AP1 生成器：

$$
\Delta\theta_{AP1}
=
-\eta_{ap}
P_{last}
M_{tail-safe}
\widehat g_{tail-corrected}.
$$

其中 $P_{last}$ 是 last-edge projection，$M_{tail-safe}$ 是 tail-safe mask，$\widehat g_{tail-corrected}$ 是当前 batch / hard-tail 子集上的手写梯度组合。

AP1 certificate：

$$
D_{CE}=g_{CE}^T\Delta\theta_{AP1},
$$

$$
D_{tail}=g_{tail}^T\Delta\theta_{AP1},
$$

$$
NormBound=\frac{\|\Delta\theta_{AP1}\|}{\|\Delta\theta_{AdamW}\|+\epsilon}.
$$

AP1 pass condition：

```text
D_CE < -gamma_value
D_tail <= gamma_tail
NormBound <= tau_norm
predicted_margin_tail_gain >= 0
payload_apply_time_ms_q90 <= 0.10
```

### 必须记录

```text
ap1_generated_action_count
ap1_certificate_pass_count
ap1_last_edge_param_count
ap1_delta_norm
ap1_norm_over_adamw
ap1_descent_margin
ap1_tail_safety_margin
ap1_predicted_CE_delta
ap1_predicted_margin_tail_delta
ap1_payload_apply_time_ms_q90
ap1_action_apply_error_linf_max
```

## 5.2 AP2：AdamWResidualOrthogonalBenefitCertificate

### 核心思想

AP2 不应该只是 AdamW 的 LR 改写。如果 functional update 只是沿 AdamW 方向缩放，就很难证明 Beyond-MLP。AP2 生成 AdamW-orthogonal residual action：

$$
\Delta\theta_{res}
=
\Delta\theta_{raw}
-
\frac{\langle \Delta\theta_{raw},\Delta\theta_{AdamW}\rangle}
{\|\Delta\theta_{AdamW}\|^2+\epsilon}
\Delta\theta_{AdamW}.
$$

然后对 residual 做安全缩放：

$$
\Delta\theta_{AP2}=\alpha_{safe}\Delta\theta_{res}.
$$

AP2 certificate：

```text
orthogonality certificate：|cos(delta_ap2, delta_adamw)| <= tau_cos
benefit certificate：linearized CE / margin still positive
conflict certificate：does not worsen AdamW hard-tail direction
```

形式：

$$
|\cos(\Delta\theta_{AP2},\Delta\theta_{AdamW})|\le \tau_{orth}.
$$

$$
g_{CE}^T\Delta\theta_{AP2}< -\gamma_{value}.
$$

### 必须记录

```text
ap2_generated_action_count
ap2_certificate_pass_count
ap2_cos_to_adamw
ap2_cos_to_negative_grad
ap2_residual_norm
ap2_residual_norm_over_adamw
ap2_projected_descent
ap2_tail_conflict_score
ap2_action_apply_error_linf_max
ap2_payload_apply_time_ms_q90
```

## 5.3 AP3：HorizonRobustTailMemoryCertificate

### 核心思想

AP0 的最大弱点是 horizon-fragile：short-only action 多，long-risk action 多。AP3 直接把 horizon robustness 写进 generator。它不追求单步最强收益，而追求跨 horizon 不翻车。

AP3 生成器：

```text
1. 从当前 batch hard-tail 和 recent-tail memory 中构造 tail prototype；
2. 生成只作用于 tail-sensitive edge/basis roles 的小幅 action；
3. 通过 multi-slice commit-time certificate 约束 short / medium / long proxy risk。
```

AP3 certificate：

$$
H_{risk}=\n\max(
RiskProxy_{short},
RiskProxy_{medium},
RiskProxy_{long}
).
$$

AP3 pass condition：

```text
cert_value_lcb > 0
cert_horizon_risk <= tau_h
predicted short gain > 0
predicted medium gain >= 0
predicted long risk <= tau_long
long_risk_rate in outcome <= 0.05 for certificate-pass rows
```

### 必须记录

```text
ap3_generated_action_count
ap3_certificate_pass_count
ap3_tail_memory_size
ap3_tail_memory_hash
ap3_short_proxy_value
ap3_medium_proxy_value
ap3_long_proxy_risk
ap3_horizon_risk_score
ap3_long_risk_outcome_rate
ap3_horizon_robust_CP_precision
ap3_action_apply_error_linf_max
ap3_payload_apply_time_ms_q90
```

## 5.4 AP4：LowRankEdgeCertificate

### 核心思想

AP4 生成低秩 edge-function payload，目标是可解释、可压缩、可快速 apply。它用于验证：是否存在一个更接近 PureKAN edge geometry 的低秩 functional update，既有 certificate 又有 runtime 可行性。

AP4 生成器：

$$
\Delta W_{edge}=U_r \Sigma_r V_r^T,
$$

其中 $r\le r_{max}$，并且 $U,V$ 来自 legal commit-time gradient / basis statistics，不来自 future outcome。

AP4 certificate：

```text
low-rank norm bound；
basis-role sparsity；
edge locality；
linearized descent；
tail safety；
runtime cost bound。
```

Pass：

```text
rank <= r_max
edge_update_density <= tau_density
cert_descent_margin > 0
cert_tail_safety_margin >= 0
payload_apply_time_ms_q90 <= 0.20
```

### 必须记录

```text
ap4_generated_action_count
ap4_certificate_pass_count
ap4_rank
ap4_update_density
ap4_basis_role_entropy
ap4_edge_locality_score
ap4_descent_margin
ap4_tail_safety_margin
ap4_action_apply_error_linf_max
ap4_payload_apply_time_ms_q90
```

---

# 6. 数据合同

## 6.1 action payload contract

每个 AP action 必须记录：

```text
action_id
event_id
candidate_id
primitive_id
primitive_version
payload_tensor_path
payload_hash
payload_dtype
payload_shape_signature
payload_role_signature
payload_norm
payload_linf
payload_sparsity
payload_rank
payload_generation_time_ms
payload_apply_time_ms
payload_apply_error_linf
payload_apply_error_relative
payload_apply_cosine_logged_applied
```

Pass：

```text
generated_action_count > 0
payload_tensor_path not empty
payload_hash_missing_count = 0
payload_apply_error_measured = 1
payload_apply_error_linf_max <= 1e-6
payload_apply_cosine_min >= 0.999999
```

## 6.2 certificate contract

每个 AP action 必须记录：

```text
certificate_id
certificate_schema_version
certificate_tensor_path
certificate_hash
certificate_commit_time
certificate_generated_before_outcome
certificate_bound_payload_hash
cert_value_lcb
cert_bad_ucb
cert_null_ucb
cert_support_lcb
cert_horizon_risk
cert_cost_estimate_ms
cert_descent_margin
cert_tail_safety_margin
cert_norm_bound
cert_orthogonality_score
cert_lowrank_score
cert_pass
cert_fail_reason
uses_dataset_name
uses_outcome_at_commit
uses_future_step
uses_validation_or_test
```

Pass：

```text
certificate_tensor_path not empty
certificate_hash_missing_count = 0
certificate_generated_before_outcome = 1
certificate_bound_payload_hash = 1
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
uses_validation_or_test = 0
```

## 6.3 matched-control outcome contract

每个 AP action 的 outcome 必须按 branch/horizon 记录：

```text
action_id
primitive_id
event_id
dataset
seed
step
branch
horizon
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
task_safe_label
bad_event_label
null_event_label
weak_CP_label
strong_CP_label
horizon_robust_CP_label
long_risk_label
V_ctrl
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
outcome_runtime_ms
outcome_row_hash
```

Branches：

```text
RealAP
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledAPPayload
ShuffledCertificateScore
CertificatePassNoPayload
```

Horizons：

```text
20
80
240
optional: 640 for selected survivors
```

Pass：

```text
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1.0
label_exclusivity_violation_count = 0
duplicate_outcome_row_id_count = 0
horizon_state_hash_mismatch_count = 0
metric_nan_count = 0
metric_inf_count = 0
```

## 6.4 runtime contract

每 selected AP primitive 必须记录：

```text
runtime_candidate_id
primitive_id
controller_id
runtime_mode
step_count
active_step_count
zero_candidate_step_count
generated_action_count
certificate_pass_action_count
accepted_action_count
feature_compute_time_ms_q90
certificate_compute_time_ms_q90
payload_generation_time_ms_q90
payload_lookup_time_ms_q90
payload_apply_time_ms_q90
score_accept_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
controller_launches_per_active_step_q90
controller_syncs_per_active_step_q90
allocation_count_per_active_step
payload_apply_error_linf_max
selected_controller_used
selected_payload_apply_used
materializer_in_timed_path
audit_in_timed_path
```

Official runtime pass：

```text
runtime_mode = online_sequential_selected_primitive
selected_controller_used = 1
selected_payload_apply_used = 1
materializer_in_timed_path = 0
audit_in_timed_path = 0
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
payload_apply_error_linf_max <= 1e-6
```

---

# 7. 实验阶段

## P0：v9.3.7 boundary reproduction and no-audit-only guard

### 目标

复现 v9.3.7 boundary，并加一个硬 guard：如果 v9.3.8 没有真实生成 AP action，runner 必须早停，不能继续跑 certificate calibration / controller / runtime。

### 假设

H0/H1：v9.3.7 的 blocker 是 generator missing，不是 schema missing。

### 必须记录

```text
source_route_v9370
candidate_count
action_count_v9370
diagnostic_certificate_rows_v9370
primitive_materialized_count_v9370
generated_action_count_total_v9370
certificate_schema_contract_pass_v9370
ap_smoke_outcome_pass_v9370
ap0_stop_condition_v9370
best_ap0_feature_auc_CP_v9370
best_ap0_top273_CP_precision_v9370
no_audit_only_guard_enabled
```

### Pass 标准

```text
P0 pass:
  source_route_v9370 = R6-ActionPrimitiveRedesignRequired
  ap0_stop_condition_v9370 = 1
  generated_action_count_total_v9370 = 0
  primary_blocker_v9370 = certificate_producing_action_primitive_not_materialized
  no_audit_only_guard_enabled = 1
```

### 可视化

```text
p0_v9370_boundary_ladder.svg
p0_ap0_patch_stop_dashboard.svg
p0_schema_vs_primitive_gap.svg
```

---

## P1：real AP1-AP4 generator implementation smoke

### 目标

实现 AP1/AP2/AP3/AP4 的真实 generator。P1 不看 outcome，只检查能否生成 payload、证书、hash，并能 replay apply。

### 假设

H1/H2：至少一个 AP primitive 可以生成合法 payload 与 commit-time certificate。

### 实现要求

新增或修改模块：

```text
dgkan/functional_primitives/ap_base.py
dgkan/functional_primitives/ap1_last_edge_tail_safe.py
dgkan/functional_primitives/ap2_adamw_residual.py
dgkan/functional_primitives/ap3_horizon_tail_memory.py
dgkan/functional_primitives/ap4_lowrank_edge.py
dgkan/functional_primitives/certificate.py
dgkan/functional_primitives/payload_packager.py
experiments/run_v9380_real_certificate_action_primitive_materialization.py
```

统一接口：

```text
generate_action(
  model_state,
  batch_state,
  manual_grad_state,
  adamw_delta_state,
  event_context,
  primitive_config
) -> ActionPayload, Certificate, GenerationAudit
```

### 必须记录

```text
primitive_id
generator_version
event_id
candidate_id
generated_action
generation_fail_reason
payload_tensor_path
payload_hash
certificate_tensor_path
certificate_hash
certificate_bound_payload_hash
certificate_generated_before_outcome
certificate_pass
cert_fail_reason
payload_generation_time_ms
certificate_compute_time_ms
action_apply_error_linf
action_apply_error_relative
action_apply_cosine
uses_dataset_name
uses_outcome_at_commit
uses_future_step
uses_validation_or_test
```

### Pass 标准

P1 smoke pass：

```text
at least one primitive:
  generated_action_count >= 64
  payload_hash_missing_count = 0
  certificate_hash_missing_count = 0
  certificate_bound_payload_hash_rate = 1.0
  certificate_generated_before_outcome_rate = 1.0
  action_apply_error_measured = 1
  action_apply_error_linf_max <= 1e-6
  uses_dataset_name = 0
  uses_outcome_at_commit = 0
```

P1 strong pass：

```text
at least two primitives:
  generated_action_count >= 512 each
  certificate_pass_count >= 64 each
```

P1 fail route：

```text
R1-APGeneratorStillMissing:
  generated_action_count_total = 0

R2-APPayloadContractFail:
  generated_action_count_total > 0 but payload/hash/apply contract fails

R3-APCertificateContractFail:
  payload ok but certificate not legal or not bound
```

### 可视化

```text
p1_generated_actions_by_primitive.svg
p1_certificate_pass_by_primitive.svg
p1_action_apply_error_hist.svg
p1_generation_time_by_primitive.svg
p1_payload_norm_distribution.svg
```

---

## P2：certificate legality and minimality audit

### 目标

证明 certificate 是 commit-time、legal、minimal、可拆解的 sufficient-statistic 雏形，而不是一个 posthoc opaque score。

### 假设

H2/H4：certificate 字段完整、合法、可解释，并且不使用 outcome/future/test/dataset branch。

### 必须记录

```text
certificate_id
primitive_id
cert_value_lcb
cert_bad_ucb
cert_null_ucb
cert_support_lcb
cert_horizon_risk
cert_cost_estimate_ms
cert_descent_margin
cert_tail_safety_margin
cert_norm_bound
cert_orthogonality_score
cert_lowrank_score
cert_field_missing_count
cert_monotone_sign_violation_count
certificate_generated_before_outcome
certificate_bound_payload_hash
uses_dataset_name
uses_outcome_at_commit
uses_future_step
uses_validation_or_test
opaque_score_only
```

### Pass 标准

```text
cert_field_missing_count = 0
certificate_generated_before_outcome = 1 for all rows
certificate_bound_payload_hash = 1 for all rows
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
uses_validation_or_test = 0
opaque_score_only = 0
cert_monotone_sign_violation_count = 0
```

Minimality gate：

```text
每个 primitive 的 controller-visible certificate groups <= 6：
  value
  bad risk
  null risk
  support
  horizon risk
  cost

禁止 20+ feature pile 直接进入 controller。
```

### 可视化

```text
p2_certificate_field_completeness.svg
p2_certificate_legality_dashboard.svg
p2_certificate_component_correlation.svg
p2_certificate_monotone_sign_audit.svg
```

---

## P3：AP smoke outcome materialization

### 目标

第一次 materialize 新 AP payload outcomes。P3 不允许使用 AP0 existing outcomes 替代。

### 假设

H3：至少一个 AP primitive 的 certificate-pass action 在 new AP outcomes 上优于 AP0 diagnostic certificate baseline。

### 样本设计

分三层并行：

```text
Smoke-64:
  每个 primitive 随机但 stratified 选 64 generated actions；
  必跑所有 branches × horizons；
  用于快速发现 payload/outcome bug。

Panel-864:
  每个 primitive 最多 864 generated actions；
  stratified by dataset/seed/family/horizon/certificate_pass；
  用于估计 weak/strong/horizon CP。

Full-generated:
  如果 Panel-864 达到 weak pass，扩展到全部 generated actions。
```

Branches：

```text
RealAP
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledAPPayload
ShuffledCertificateScore
CertificatePassNoPayload
```

Horizons：

```text
20
80
240
```

### 必须记录

```text
action_id
primitive_id
certificate_pass
branch
horizon
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
weak_CP_label
strong_CP_label
horizon_robust_CP_label
long_risk_label
bad_event_label
null_event_label
V_ctrl
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
branch_completion_rate
horizon_completion_rate
quality_audit_pass
```

### Pass 标准

P3 smoke pass：

```text
for at least one primitive:
  branch_completion_rate = 1.0
  horizon_completion_rate = 1.0
  secondary_delta_completion_rate = 1.0
  quality_audit_pass = 1
  certificate_pass_outcome_count >= 16
```

P3 weak action-value pass：

```text
for at least one primitive certificate-pass rows:
  weak_CP_precision >= 0.40
  V_ctrl_lcb > 0
  long_risk_rate <= 0.15
```

P3 strong action-value pass：

```text
for at least one primitive certificate-pass rows:
  weak_CP_precision >= 0.75
  strong_CP_precision >= 0.40
  horizon_robust_CP_precision >= 0.10
  long_risk_rate <= 0.05
  V_ctrl_lcb > 0
```

P3 fail route：

```text
R4-APSmokeOutcomeMaterializationFail:
  new AP outcomes cannot be materialized

R5-APGeneratedActionsValueFail:
  outcomes complete but all primitives weak_CP_precision < 0.40 or V_ctrl_lcb <= 0

R6-APHorizonRiskFail:
  weak CP exists but long-risk / horizon-fragility remains high
```

### 可视化

```text
p3_ap_outcome_completion_matrix.svg
p3_weak_cp_precision_by_primitive.svg
p3_strong_cp_precision_by_primitive.svg
p3_horizon_robust_precision_by_primitive.svg
p3_long_risk_by_primitive.svg
p3_vctrl_distribution_by_primitive.svg
p3_real_vs_controls_pareto.svg
```

---

## P4：full AP frontier completion

### 目标

如果 P3 smoke pass，扩展到足够密度的 AP action universe，判断 AP primitive 是否有足够 frontier 支撑 controller。

### 假设

H3：certificate-producing AP primitive 应产生比 AP0 更可见、更 horizon-robust 的 action frontier。

### Full universe 设计

```text
primitives = AP1, AP2, AP3, AP4
base events = 2876 frozen events
max action universe = 4 × 2876 = 11504 AP actions
branches = 6 official branches + shuffled controls
horizons = 20,80,240
expected official branch-horizon rows for 6 branches = 11504 × 6 × 3 = 207072
```

为了加快实验，采用 shard 并行：

```text
shard_by = primitive_id × seed × action_range
worker_count_target = 4 to 8
checkpoint_reuse = 1
payload_shard_cache = 1
retry_manifest = 1
```

### 必须记录

```text
primitive_id
generated_action_count
certificate_pass_count
full_outcome_row_count_expected
full_outcome_row_count_actual
branch_completion_rate
horizon_completion_rate
rows_per_sec_total
wallclock_sec
failed_row_count
retried_row_count
weak_CP_action_count
weak_CP_row_count
strong_CP_action_count
horizon_robust_CP_action_count
weak_CP_coverage
strong_CP_coverage
horizon_robust_CP_coverage
V_ctrl_mean
V_ctrl_lcb
long_risk_action_count
support_balance_pass
```

### Pass 标准

P4 full AP frontier weak pass：

```text
full_outcome_completion_rate >= 0.95
quality_audit_pass = 1
for at least one primitive:
  weak_CP_coverage >= 0.03
  weak_CP_precision_oracle >= 0.75
  V_ctrl_lcb > 0
  support_balance_pass = 1
```

P4 strong pass：

```text
for at least one primitive:
  strong_CP_coverage >= 0.03
  horizon_robust_CP_coverage >= 0.01
  long_risk_rate <= 0.05
```

注意：horizon-robust coverage 从 AP0 的 `0.0042989` 提升到 `>=0.01` 已经是重要进展；`>=0.03` 是下一阶段强目标。

### 可视化

```text
p4_full_ap_frontier_coverage.svg
p4_weak_strong_horizon_frontier.svg
p4_vctrl_lcb_by_primitive.svg
p4_long_risk_action_map.svg
p4_materializer_throughput_by_shard.svg
p4_support_balance_by_family_horizon.svg
```

---

## P5：certificate calibration and sufficiency audit

### 目标

评估 certificate 是否真能在 commit 前预测 AP action outcome。P5 不允许用 outcome label at commit time，只评估 certificate 与 outcome 的关系。

### 假设

H4：certificate 的 value/bad/null/support/horizon/cost 分解能形成可校准的 sufficient statistic。

### 必须记录

```text
primitive_id
certificate_component
AUC_weak_CP
AUC_strong_CP
AUC_horizon_robust_CP
AUC_bad_event
AUC_long_risk
PR_lift_weak_CP
PR_lift_long_risk
Brier_bad
ECE_bad
Brier_long_risk
ECE_long_risk
calibration_slope
calibration_intercept
top273_weak_CP_precision
top273_long_risk_rate
leave_dataset_auc_drop
leave_stratum_auc_drop
monotone_sign_pass
```

### Pass 标准

P5 weak pass：

```text
at least one primitive certificate:
  AUC_weak_CP >= 0.70
  AUC_long_risk >= 0.65
  top273_weak_CP_precision >= 0.45
  top273_long_risk_rate <= 0.15
  ECE_bad <= 0.10
```

P5 strong pass：

```text
AUC_weak_CP >= 0.78
AUC_bad_event >= 0.75
AUC_long_risk >= 0.72
top273_weak_CP_precision >= 0.75
top273_long_risk_rate <= 0.05
ECE_bad <= 0.05
leave_dataset_auc_drop <= 0.07
leave_stratum_auc_drop <= 0.10
```

P5 fail route：

```text
R7-CertificateNotPredictive:
  AP actions have frontier but certificate cannot identify it

R8-CertificateDatasetUnstable:
  certificate signal exists but LDO/LSO collapses
```

### 可视化

```text
p5_certificate_auc_bar.svg
p5_certificate_pr_lift.svg
p5_certificate_calibration_curve.svg
p5_certificate_topk_precision.svg
p5_leaveout_auc_drop_heatmap.svg
p5_certificate_component_ablation.svg
```

---

## P6：minimal certificate controller

### 目标

建立一个不按 dataset 调参、不使用 future outcome、不使用 opaque feature pile 的 minimal certificate controller。

### Controller form

$$
Accept_{AP}(e)=1
\iff
CertPass(e)=1
\land LCB(V_{cert}(e))>0
\land UCB(B_{cert}(e))\le \tau_b
\land UCB(N_{cert}(e))\le \tau_n
\land LCB(S_{cert}(e))\ge \tau_s
\land H_{risk}(e)\le \tau_h
\land C(e)\le C_{max}.
$$

Controller candidates：

```text
C0-AP0-negative-control
C1-CertificatePassOnly
C2-ValueBadNullSupportGate
C3-HorizonRobustCertificateGate
C4-CostAwareCertificateGate
C5-TwoStageCertificateThenExactLegalConfirm
```

### Splits

```text
seed-fold cross-fit：3 folds
leave-dataset-out：MNIST / Fashion-MNIST / KMNIST each held out
leave-stratum-out：signal stratum held out
leave-primitive-out diagnostic：train on AP1/AP2/AP3, evaluate AP4 等
```

### 必须记录

```text
controller_id
primitive_id
feature_groups_used
thresholds
split_type
calibration_fold
heldout_fold
heldout_entity
dataset_name_used
uses_outcome_at_commit
accepted_count_cal
accepted_count_heldout
coverage_cal
coverage_heldout
weak_CP_precision_cal
weak_CP_precision_heldout
strong_CP_precision_heldout
horizon_robust_CP_precision_heldout
bad_event_rate_heldout
null_rate_heldout
long_risk_rate_heldout
V_ctrl_mean_heldout
V_ctrl_lcb_heldout
precision_lcb
bad_event_ucb
support_balance_pass
accepted_family_count
accepted_signal_strata_count
max_family_share
max_stratum_share
```

### Pass 标准

P6 weak controller pass：

```text
coverage_heldout in [0.03, 0.15]
weak_CP_precision_heldout >= 0.75
bad_event_rate_heldout <= 0.05
null_rate_heldout <= 0.15
long_risk_rate_heldout <= 0.10
V_ctrl_lcb_heldout > 0
support_balance_pass = 1
coverage_heldout > 0 in every main fold
dataset_name_used = 0
```

P6 strong controller pass：

```text
strong_CP_precision_heldout >= 0.50
horizon_robust_CP_precision_heldout >= 0.20
long_risk_rate_heldout <= 0.05
LDO pass in at least 2/3 datasets
LSO macro weak_CP_precision >= 0.70
```

P6 fail route：

```text
R9-NoCertificateController:
  AP frontier exists but no legal controller can select it

R10-CertificateControllerCoverageCollapse:
  controller precision ok but coverage drops to zero

R11-CertificateControllerRiskFail:
  controller coverage ok but bad/long-risk too high
```

### 可视化

```text
p6_certificate_controller_frontier.svg
p6_precision_coverage_longrisk_pareto.svg
p6_calibration_to_heldout_drift.svg
p6_crossfit_fold_matrix.svg
p6_support_balance_sunburst.svg
p6_primitive_controller_overlap.svg
```

---

## P7：selected primitive online runtime

### 目标

测 selected controller + selected AP primitive 的完整 online runtime。不能只报 last-layer microbench。

### 假设

H5：至少一个 AP primitive 可以在 selected online path 下进入 MLP-comparable envelope。

### Runtime candidates

```text
RT0-AP0-reference-oracle-mask-diagnostic
RT1-AP1-last-edge-selected-online
RT2-AP2-residual-projection-selected-online
RT3-AP3-tail-memory-selected-online
RT4-AP4-lowrank-edge-selected-online
RT5-hybrid-selected-best
```

### 必须记录

```text
runtime_candidate_id
primitive_id
controller_id
runtime_mode
step_count
active_step_count
generated_action_count
certificate_pass_action_count
accepted_action_count
feature_compute_time_ms_q90
certificate_compute_time_ms_q90
payload_generation_time_ms_q90
payload_lookup_time_ms_q90
payload_apply_time_ms_q90
score_accept_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
payload_apply_error_linf_max
payload_apply_cosine_min
controller_launches_per_active_step_q90
controller_syncs_per_active_step_q90
allocation_count_per_active_step
materializer_in_timed_path
audit_in_timed_path
selected_controller_used
selected_payload_apply_used
```

### Pass 标准

P7 selected runtime pass：

```text
runtime_mode = online_sequential_selected_primitive
selected_controller_used = 1
selected_payload_apply_used = 1
materializer_in_timed_path = 0
audit_in_timed_path = 0
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
payload_apply_error_linf_max <= 1e-6
controller_launches_per_active_step_q90 <= 2
```

P7 diagnostic pass：

```text
step_ratio_q90 <= 2.00
exact dominant runtime component identified
```

### 可视化

```text
p7_selected_runtime_waterfall.svg
p7_step_ratio_by_primitive.svg
p7_payload_apply_q90_by_primitive.svg
p7_certificate_compute_q90.svg
p7_runtime_quality_pareto.svg
```

---

## P8：system integration gate

### 目标

组合 P6 controller 和 P7 runtime，判断 v9.3.8 是否产生第一个 legal AP system candidate。

### 必须记录

```text
system_candidate_id
primitive_id
controller_id
runtime_candidate_id
candidate_count
generated_action_count
certificate_pass_count
accepted_count
coverage_heldout
weak_CP_precision_heldout
strong_CP_precision_heldout
horizon_robust_CP_precision_heldout
bad_event_rate_heldout
null_rate_heldout
long_risk_rate_heldout
V_ctrl_lcb_heldout
step_ratio_q90
memory_ratio
action_lifecycle_pass
payload_binding_pass
certificate_binding_pass
matched_control_outcome_ready
materialized_system_path
diagnostic_derived_from_measured_components
uses_dataset_name
uses_outcome_at_commit
uses_future_step
uses_validation_or_test
official_eligible
system_legal_controller_pass
```

### Pass 标准

```text
official_eligible = 1
system_legal_controller_pass = 1
action_lifecycle_pass = 1
payload_binding_pass = 1
certificate_binding_pass = 1
matched_control_outcome_ready = 1
materialized_system_path = 1
diagnostic_derived_from_measured_components = 0
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
```

Decision gate：

$$
Coverage_{heldout}\in[0.03,0.15]
$$

$$
WeakCPPrecision_{heldout}\ge0.75
$$

$$
BadEventRate_{heldout}\le0.05
$$

$$
NullRate_{heldout}\le0.15
$$

$$
LongRiskRate_{heldout}\le0.10
$$

$$
LCB(V_{ctrl})>0
$$

System gate：

$$
StepRatio_{q90}\le1.50
$$

$$
MemoryRatio\le1.05
$$

### 可视化

```text
p8_system_gate_dashboard.svg
p8_quality_cost_frontier.svg
p8_failure_reason_matrix.svg
p8_official_candidate_lifecycle.svg
```

---

## P9：leave-dataset-out / leave-stratum-out

### 目标

只有 P8 pass 后 official 打开。证明 certificate primitive/controller 不是 pooled calibration artifact，也不是 dataset-specific tuning。

### 设置

```text
Leave-dataset-out:
  train/calibrate MNIST + Fashion-MNIST, evaluate KMNIST
  train/calibrate MNIST + KMNIST, evaluate Fashion-MNIST
  train/calibrate Fashion-MNIST + KMNIST, evaluate MNIST

Leave-stratum-out:
  train/calibrate all but one signal stratum
  evaluate held-out stratum

Leave-primitive-out diagnostic:
  train/calibrate on AP1/AP2/AP3, evaluate AP4 etc.
```

### 必须记录

```text
split_type
heldout_entity
primitive_id
controller_id
coverage
weak_CP_precision
strong_CP_precision
horizon_robust_CP_precision
bad_event_rate
null_rate
long_risk_rate
V_ctrl_lcb
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
step_ratio_q90
memory_ratio
dataset_name_used
support_balance_pass
```

### Pass 标准

LDO pass：

```text
at least 2/3 heldout datasets pass weak controller gates；
no heldout dataset has bad_event_rate > 0.10；
coverage > 0 in all heldout datasets；
dataset_name_used = 0。
```

LSO pass：

```text
>=70% heldout strata task-safe；
macro weak_CP_precision >= 0.70；
macro bad_event_rate <= 0.05；
macro long_risk_rate <= 0.10。
```

### 可视化

```text
p9_leave_dataset_out_matrix.svg
p9_leave_stratum_out_matrix.svg
p9_leave_primitive_diagnostic.svg
p9_dataset_diagnostic_no_tuning_audit.svg
```

---

## P10：official paired replay

### 目标

只有 P8/P9 pass 后打开。验证 selected AP primitive 是否真的打过 strong controls，而不是只满足 local labels。

### 设置

```text
primitives = selected AP survivor(s)
branches =
  RealAP
  AdamWOnly
  AdamWParallel
  bestLR
  NoOp
  Random
  ShuffledAPPayload
  ShuffledCertificateScore
  ShuffledCertificatePass
  ShuffledPrimitiveID
  ShuffledRuntimePath
  CertificatePassNoPayload
  APNoCertificateGate

horizons = 20, 80, 240, 640
seeds = 0,1,2,3,4
```

### 必须记录

```text
primitive_id
controller_id
dataset
seed
horizon
branch
accepted_count
coverage
weak_CP_precision
bad_event_rate
null_rate
long_risk_rate
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_noop
real_beats_random
task_safe
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### Pass 标准

Paired replay pass：

$$
BeatRate_{macro,RealAP\ vs\ AdamWParallel}\ge0.60.
$$

$$
BeatRate_{macro,RealAP\ vs\ bestLR}\ge0.60.
$$

Task safety：

$$
Acc_{RealAP}\ge Acc_{AdamW}-0.005
$$

for every official slice.

Shuffle controls：

```text
No shuffled control can match RealAP macro beat pattern.
CertificatePassNoPayload must fail.
APNoCertificateGate must not match selected certificate controller.
```

### 可视化

```text
p10_official_paired_replay_pareto.svg
p10_macro_beat_rate.svg
p10_shuffle_control_matrix.svg
p10_horizon_win_matrix.svg
p10_task_safety_by_slice.svg
```

---

## P11：short-run / full-run / sample efficiency / continual robustness

### 目标

只有 P10 pass 后打开。验证 local AP causal advantage 能否进入连续训练，并且是否在 sample efficiency、calibration、geometry 或 anti-forgetting 上形成 Beyond-MLP 证据。

### 设置

```text
short-run steps = 50, 240, 640
full-run seeds = 0..9
datasets = MNIST, Fashion-MNIST, KMNIST
controls =
  AdamWOnly
  AdamWParallel
  bestLR
  StrongLRGrid
  QuadraticFeatureMLP
  NoOp
  Random
  ShuffledAPPayload
  ShuffledCertificateScore
  ShuffledRuntimePath
  APNoCertificateGate
```

Continual block：

```text
Task order examples:
  MNIST -> Fashion-MNIST -> KMNIST
  Fashion-MNIST -> KMNIST -> MNIST
  KMNIST -> MNIST -> Fashion-MNIST
```

### 必须记录

```text
primitive_id
controller_id
dataset
seed
steps
final_acc
best_val_acc
test_acc
train_loss
val_loss
test_loss
ECE
NLL
CEp99
margin_p10
curvature
local_lipschitz
ValLossAUC_step
ValLossAUC_time
time_to_target
steps_to_target
functional_event_count
coverage
bad_event_rate
null_rate
long_risk_rate
step_ratio_q90
memory_ratio
strong_baseline_beaten
continual_retained_accuracy
forgetting
backward_transfer
forward_transfer
old_task_CEp99_drift
old_task_margin_drift
```

### Pass 标准

Short/full functional pass：

$$
Acc_{AP}\ge Acc_{AdamW}-0.005.
$$

并且至少一个成立：

$$
Acc_{AP}>Acc_{AdamWParallel},
$$

$$
ECE_{AP}<ECE_{AdamW},
$$

$$
NLL_{AP}<NLL_{AdamW},
$$

$$
Curvature_{AP}\le0.90Curvature_{AdamW},
$$

$$
ValLossAUC_{time,AP}<ValLossAUC_{time,AdamW}.
$$

Continual pass：

```text
retained_accuracy_AP >= retained_accuracy_AdamW - 0.005
forgetting_AP <= forgetting_AdamW
old_task_CEp99_drift_AP <= old_task_CEp99_drift_AdamW
```

### 可视化

```text
p11_learning_curve_by_time.svg
p11_val_loss_auc.svg
p11_time_to_target.svg
p11_full_run_accuracy_ece_nll.svg
p11_geometry_curvature_lipschitz.svg
p11_continual_forgetting_matrix.svg
p11_strong_baseline_comparison.svg
```

---

# 8. 并行执行计划

v9.3.8 必须并行推进，避免下一轮又变成 audit-only。

## Batch A：AP generator implementation race

并行实现：

```text
A1: AP1 LastEdgeLinearizedTailSafeCertificate
A2: AP2 AdamWResidualOrthogonalBenefitCertificate
A3: AP3 HorizonRobustTailMemoryCertificate
A4: AP4 LowRankEdgeCertificate
```

每个实现必须先通过本地 unit test：

```text
can_generate_payload = 1
can_write_payload_hash = 1
can_write_certificate_hash = 1
can_apply_payload = 1
apply_error_linf <= 1e-6
uses_dataset_name = 0
uses_outcome_at_commit = 0
```

任何 primitive 如果 `generated_action_count = 0`，不得进入 P3。

## Batch B：payload / certificate packager

并行构建：

```text
B1: durable payload shard writer
B2: certificate tensor writer
B3: payload-certificate hash binder
B4: action apply replay materializer
B5: no-fake / no-proxy / no-future audit
```

B 系列必须在 P1 前完成，否则 P1 不能 pass。

## Batch C：outcome materializer scaling

并行准备：

```text
C1: Smoke-64 branch/horizon materializer
C2: Panel-864 stratified materializer
C3: Full-generated AP universe materializer
C4: retry manifest and row quality audit
C5: AP-specific outcome tables
```

C1 和 C2 可在 A1-A4 出第一个 generator 后立即跑，不必等待全部 primitives 完成。

## Batch D：runtime path

并行测量：

```text
D1: AP1 selected last-edge runtime
D2: AP2 projection runtime
D3: AP3 tail-memory runtime
D4: AP4 low-rank runtime
D5: selected controller full online runtime
```

D 系列必须区分：

```text
microbench_runtime
selected_online_runtime
offline_materializer_runtime
```

只有 selected online runtime 能 official。

## Batch E：controller / leave-out / replay preparation

在 P3 smoke pass 后提前准备：

```text
E1: certificate calibration script
E2: cross-fit split builder
E3: LDO/LSO split builder
E4: paired replay branch runner
E5: visualization dashboard
```

E 系列不能使用 P10/P11 结果回调 P6 threshold。

---

# 9. Required artifacts

```text
run_manifest.json
contract_audit_v9380.csv
provenance_audit_v9380.csv
failure_table_v9380.csv
artifact_hashes_v9380.csv

p0_v9370_boundary_reproduction.csv
p1_real_ap_generator_smoke.csv
p2_certificate_legality_minimality_audit.csv
p3_ap_smoke_outcome_materialization.csv
p4_full_ap_frontier_completion.csv
p5_certificate_calibration_sufficiency.csv
p6_minimal_certificate_controller.csv
p7_selected_primitive_online_runtime.csv
p8_system_integration_gate.csv
p9_leave_dataset_stratum_out.csv
p10_official_paired_replay.csv
p11_short_full_sampleeff_continual_robustness.csv

ap_action_payload_trace_v9380.csv
ap_certificate_trace_v9380.csv
action_apply_replay_trace_v9380.csv
ap_smoke_outcome_trace_v9380.csv
full_ap_outcome_table_v9380.csv
certificate_calibration_trace_v9380.csv
certificate_controller_trace_v9380.csv
selected_runtime_component_trace_v9380.csv
leaveout_trace_v9380.csv
paired_replay_branch_trace_v9380.csv
short_full_trace_v9380.csv

figures/
route_decision.json
aggregate_decision.json
```

---

# 10. Failure taxonomy

```text
F0_v9370_boundary_unstable
F1_AP0_patch_route_reopened
F2_AP_generator_missing
F3_AP_payload_hash_missing
F4_AP_certificate_hash_missing
F5_certificate_not_bound_to_payload
F6_certificate_generated_after_outcome
F7_dataset_name_used
F8_future_outcome_used
F9_action_apply_error_fail
F10_AP_smoke_outcome_missing
F11_branch_horizon_completion_fail
F12_AP_generated_actions_value_fail
F13_AP_long_risk_fail
F14_full_AP_frontier_absent
F15_certificate_not_predictive
F16_certificate_calibration_drift
F17_certificate_controller_precision_fail
F18_certificate_controller_coverage_fail
F19_certificate_controller_bad_event_fail
F20_certificate_controller_null_fail
F21_certificate_controller_long_risk_fail
F22_selected_runtime_not_measured
F23_selected_runtime_step_ratio_fail
F24_memory_ratio_fail
F25_payload_apply_runtime_fail
F26_system_integration_fail
F27_leave_dataset_out_fail
F28_leave_stratum_out_fail
F29_paired_replay_control_equivalent
F30_shuffle_control_pass
F31_short_run_task_drop
F32_full_run_no_macro_or_hard_stratum_gain
F33_strong_baseline_explains_gain
F34_continual_forgetting_fail
F35_external_not_ready
F36_fake_or_proxy_violation
F37_artifact_missing
```

---

# 11. Route decision

```text
R0-BoundaryReproduced:
  v9.3.7 boundary reproduced; AP0 route remains stopped.

R1-APGeneratorStillMissing:
  AP1/AP2/AP3/AP4 generated_action_count_total remains 0.

R2-APPayloadContractFail:
  AP action generated but payload/hash/apply contract fails.

R3-APCertificateContractFail:
  payload ok but certificate illegal, incomplete, or not bound to payload.

R4-APSmokeOutcomeMaterializationFail:
  new AP payload outcomes cannot be materialized.

R5-APGeneratedActionsValueFail:
  outcomes complete but AP actions have no weak CP / V_ctrl signal.

R6-APHorizonRiskFail:
  weak CP exists but horizon fragility / long-risk remains high.

R7-FullAPFrontierPass:
  at least one AP primitive has sufficient weak CP frontier.

R8-CertificateNotPredictive:
  AP frontier exists but certificate cannot identify it.

R9-CertificateDatasetUnstable:
  certificate signal collapses in LDO/LSO.

R10-CertificateControllerPass:
  minimal certificate controller passes decision gates.

R11-CertificateControllerFail:
  no legal certificate controller passes decision gates.

R12-SelectedRuntimePass:
  selected primitive online runtime reaches step_ratio_q90 <= 1.50.

R13-SelectedRuntimeFail:
  selected primitive runtime too expensive.

R14-SystemLegalControllerPass:
  decision + runtime + contracts pass.

R15-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R16-LeaveStratumOutPass:
  controller generalizes across held-out strata.

R17-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R18-PairedReplayFail:
  system legal but AP functional update is control-equivalent.

R19-ShortRunFunctionalPass:
  short-run task-safe mechanism gain.

R20-FullFunctionalPass:
  full run task / geometry / system / control / robustness gates pass.

R21-ExternalReady:
  strict PureKAN functional route passes external-ready gates.
```

`route_decision.json` 必须记录：

```text
route
base_candidate
source_route_v9370
ap0_stop_condition
generated_action_count_total
primitive_generation_pass
best_generated_primitive_id
payload_binding_pass
certificate_binding_pass
action_apply_pass
ap_smoke_outcome_pass
full_ap_frontier_pass
certificate_calibration_pass
certificate_controller_pass
selected_payload_runtime_pass
system_legal_controller_pass
leave_dataset_out_pass
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
uses_outcome_at_commit
uses_future_step
success_v9380_strict_purekan_functional
success_v9380_full_functional
success_v9380_external_ready
```

---

# 12. 最终判定逻辑

v9.3.8 的判定必须非常硬：

```text
如果 generated_action_count_total = 0：
  直接 R1，禁止继续 certificate calibration / controller / runtime。

如果 payload / certificate / apply 不闭合：
  直接 R2/R3，禁止 outcome/controller promotion。

如果 new AP payload outcomes 不存在：
  直接 R4，禁止用 AP0 outcomes 代替。

如果 AP actions 无 weak CP / V_ctrl：
  判 primitive design fail，重设 generator，不调 controller。

如果 AP actions 有 frontier 但 certificate 不能识别：
  判 certificate design fail，不调 opaque threshold。

如果 certificate controller pass 但 runtime fail：
  优先优化 selected primitive runtime，不改 decision threshold。

如果 system pass 后 paired replay fail：
  才能判断 AP functional update control-equivalent。
```

最终一句话：

$$
\boxed{
\text{v9.3.8 的成功不是“再写一个 schema”，而是让 AP action payload、certificate、outcome、controller、runtime 第一次形成真实闭环。}
}
$$
