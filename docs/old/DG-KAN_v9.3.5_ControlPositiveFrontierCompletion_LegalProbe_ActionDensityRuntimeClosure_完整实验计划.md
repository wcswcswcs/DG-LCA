# DG-KAN v9.3.5 Control-Positive Frontier Completion / Legal Pre-Commit Probe / Action-Density Reset / Payload-Apply Runtime Closure 完整实验计划

> 本计划基于 v9.3.4 `Control Outcome Materializer Scale-up / Action-Value Identifiability / Online Runtime Remeasure` 的真实执行结果制定。  
> v9.3.5 不继续做 StableAccept patch，也不把 OGP / AVF feature factory 变成新的补丁堆。  
> 本轮核心目标是把 v9.3.4 暴露出的三个第一性问题一次性拆清楚：
>
> $$
> \boxed{
> \text{control-positive action 是否足够多、是否可在 commit 前识别、是否能以 payload-apply online runtime 进入系统 envelope？}
> }
> $$

---

# 0. v9.3.4 独立判断

## 0.1 这轮有真实进展，但不是 functional success

v9.3.4 的真实推进非常明确：

```text
v9.3.3:
  durable payload package 已闭合；
  control outcome 只有 8-action / 48-row dry-run。

v9.3.4:
  durable payload regression 继续闭合；
  control outcome materializer 扩展到 864-action / 15552-row official-minimum panel；
  panel quality audit 通过；
  online runtime smoke 独立重测且未混入 offline materializer；
  但 control-positive oracle raw coverage 只有 0.0171，action-value observability 也未形成可用 signal。
```

这不是失败倒退。它把 v9.3.3 的 “control outcome materializer 还没真正跑起来” 推进成了 “materializer 可以规模化、质量可审计、吞吐足够高”。但它仍不是 strict PureKAN functional success，因为：

```text
control_oracle_pass = 0
action_value_observability_pass = 0
online_runtime_pass = 0
system_legal_controller_pass = 0
```

因此 v9.3.4 的正确定位是：

$$
\boxed{
\text{label engine 进入可用阶段，但 control-positive frontier 与 legal observability 仍未闭合。}
}
$$

## 0.2 v9.3.4 最大进展：control outcome materializer 从不可用变成可扩展

v9.3.3 的 full control outcome 只有：

```text
actual_branch_horizon_rows = 48
expected_branch_horizon_rows = 51768
completion_rate = 0.000927
offline_rows_per_sec = 0.452111
```

v9.3.4 完成：

```text
panel_actions = 864
panel_rows = 15552
full_universe_rows = 51768
panel_completion_rate = 0.3004
rows_per_sec_total = 22.08765790513577
unresolved_failed_rows = 0
branch_missing = 0
horizon_missing = 0
secondary_delta_missing = 0
quality_audit_pass = 1
```

吞吐提升为：

$$
Speedup = \frac{22.0876579}{0.4521114} \approx 48.85.
$$

以 v9.3.4 吞吐估计，完整 full universe 所需时间约为：

$$
T_{full} = \frac{51768}{22.0876579} \approx 2343.75s \approx 39.1min.
$$

剩余未 materialize rows 为：

$$
51768-15552=36216.
$$

继续完成剩余 full universe 的估计时间约为：

$$
T_{remain} = \frac{36216}{22.0876579} \approx 1639.65s \approx 27.3min.
$$

这意味着 v9.3.5 不应该再停在 30% panel。现在 full materialization 已经从 “一天级 blocker” 变成 “小时内可以闭合的工程任务”。

## 0.3 v9.3.4 最大歧义：`control_positive_oracle_absent` 这个 route 名称可能过强

v9.3.4 报告的 P5 oracle 是：

```text
accepted_count = 155
coverage = 0.01708553791887125
precision_primary = 1.0
precision_control_positive = 1.0
bad_event_rate = 0.0
null_rate = 0.0
V_ctrl_mean = 0.40297435385084923
V_ctrl_median = 0.28818584233522415
beats_adamwparallel_rate = 1.0
beats_bestlr_rate = 1.0
beats_noop_rate = 1.0
beats_random_rate = 1.0
support_balance_pass = 1
control_oracle_pass = 0
```

报告因为 coverage $0.0171 < 0.03$ 判定 `R5-ControlPositiveOracleAbsent`。这个 gate 是保守的，但从数据本身看，不能直接推出 “control-positive frontier 不存在”。

原因是本轮只 materialize 了 864 / 2876 actions：

$$
PanelFraction = \frac{864}{2876} \approx 0.3004.
$$

如果 panel 近似代表 full action universe，则 accepted count 的粗略 full-universe 外推为：

$$
Accepted_{scaled} = \frac{155}{0.3004} \approx 516.
$$

对应 coverage：

$$
Coverage_{scaled} = \frac{516}{9072} \approx 0.0569.
$$

这反而高于 official coverage 下限 $0.03$。如果把 panel 中 control-positive action rate 当作简单二项估计：

$$
\hat p = \frac{155}{864} \approx 0.1794,
$$

$$
SE(\hat p) \approx 0.0131,
$$

则粗略 95% 区间外推到 full universe 后，对应 coverage 大约为：

$$
Coverage_{scaled,95\%CI} \approx [0.0488, 0.0650].
$$

这个区间仍高于 $0.03$。当然，这个估计只有在 panel 近似代表 full candidate/action universe 时才成立；如果 v9.3.4 的 panel 是非随机 shard / stratified smoke，那么不能直接外推。

因此，v9.3.4 的 route 更准确应写成：

```text
R5-ControlPositivePanelRawCoverageBelowGate
or
R5-ControlPositiveFullUniverseUnresolved
```

而不是已经科学证明：

```text
ControlPositiveOracleAbsent
```

v9.3.5 的第一优先级不是调 controller，而是把这个歧义消掉：

$$
\boxed{
\text{完成 full control outcome universe，并判断 control-positive frontier 到底是存在、稀疏、还是缺失。}
}
$$

## 0.4 v9.3.4 仍然暴露了一个真实 blocker：legal observability 很弱

虽然 coverage 结论有 panel 解释歧义，但 P6 action-value observability 的失败是真实的：

```text
F1-PayloadNorm AUC_control_positive = 0.4637702634620514
F2-PayloadLinf AUC_control_positive = 0.47698781969593695
F3-PayloadLoadTime AUC_control_positive = 0.527427910974128
best_auc_control_positive = 0.527427910974128
```

这些特征接近随机。它们不是 value sufficient statistic，也不是 risk sufficient statistic。更重要的是，best feature 是 `PayloadLoadTime`，这在机制上并不令人放心；它更像 artifact / IO / shape side-channel，而不是 functional action 的数学价值信号。

所以即便 full oracle 通过，v9.3.5 也不能继续只加 payload norm / linf / load time 这类弱特征。下一步必须引入更第一性原理的 legal pre-commit observables：

```text
gradient-action alignment；
linearized CE / margin effect；
tail-risk response；
AdamW conflict；
short legal action-response probe；
support density / calibration reliability；
action family × horizon stability。
```

## 0.5 runtime 方向明显改善，但还不是 official runtime pass

v9.3.4 online runtime smoke：

```text
runtime_candidate_id = RT4-online-feature-score-accept-no-payload-apply-smoke
control_outcome_materializer_in_timed_path = 0
offline_audit_in_timed_path = 0
disk_payload_lookup_in_timed_path = 0
payload_preloaded = 1
step_count = 96
active_step_count = 96
candidate_count = 199
zero_candidate_controller_kernel_count = 0
zero_candidate_controller_sync_count = 0
controller_launches_per_active_step_q90 = 0.0
controller_syncs_per_active_step_q90 = 0.0
feature_compute_time_ms_q90 = 0.007431
score_accept_time_ms_q90 = 0.006630
base_train_step_time_ms_q90 = 0.753258
total_step_time_ms_q90 = 0.774751
step_ratio_q90 = 1.026625
memory_ratio = 1.0
online_runtime_pass = 0
reason = runtime_smoke_measured_but_no_official_controller_or_payload_apply
```

这是好消息。它说明 v9.2.x 那个 empty-step fixed launch/sync 大坑基本绕开了，而且 online/offline 分账是正确的。但这还不是 official runtime，因为它没有：

```text
official controller；
actual accepted mask；
payload apply；
no-event preservation under actual training stream；
payload apply numerical audit；
full 8064-step stream timing；
controller threshold frozen from calibration；
system integration。
```

v9.3.5 的 runtime 不能继续只测 no-payload smoke；必须测：

$$
\text{feature + score + accept + payload apply + base step}
$$

的完整 online path。

---

# 1. 当前项目状态判断

## 1.1 已经闭合的资产

截至 v9.3.4，以下资产已经比较强：

```text
1. candidate/action count 已稳定在 2876；
2. durable payload package 已闭合；
3. disk replay hash / L∞ / cosine 仍稳定；
4. action apply physics 已在 v9.3.2/v9.3.3 闭合；
5. full control outcome materializer 已从 dry-run 推到 30% panel；
6. materializer throughput 已经足够支持 full universe；
7. panel quality audit 为 0 violation；
8. online/offline runtime 已分账；
9. runtime smoke step ratio 已接近 1.0；
10. no fake / no proxy / no teacher / no dataset-specific controller 纪律保持住了。
```

这些都是真进展，不应被 “strict functional 仍未成功” 掩盖。

## 1.2 仍未闭合的关键问题

当前真正卡在四个地方：

```text
1. control-positive full-universe density 未闭合；
2. legal pre-commit observability 未闭合；
3. official controller 未闭合；
4. official payload-apply online runtime 未闭合。
```

其中第一个问题是 v9.3.5 的起点。只有知道 full universe 中 control-positive action 到底有多少，才能决定下一步是：

```text
A. full oracle pass -> 重点做 legal observability / controller；
B. full oracle weak pass -> 做 observability + action-density improvement；
C. full oracle fail -> 回到 action primitive / candidate generator reset。
```

## 1.3 是否在正确道路上

高层路线仍然正确。v9.3.0-v9.3.4 的推进顺序是合理的：

```text
v9.3.0:
  从 StableAccept patching 转向 action / OGP framing。

v9.3.1:
  发现 action apply / secondary-control / runtime 不足，防止提前 official。

v9.3.2:
  全量 action apply error 闭合。

v9.3.3:
  durable payload package 闭合，control outcome 进入真实 dry-run。

v9.3.4:
  control outcome materializer 扩展到 30% panel，runtime smoke 分账成功。
```

但 v9.3.5 如果继续用 “找一个更好的 payload feature / threshold” 的方式推进，就会走偏。真正的第一性问题是：

$$
\boxed{
\text{当前 action primitive 产生的 control-positive frontier 是 dense enough 还是 sparse but real？}
}
$$

以及：

$$
\boxed{
\text{如果 frontier 存在，commit 前是否需要一个 legal action-response probe 才能识别它？}
}
$$

---

# 2. v9.3.5 总体目标

v9.3.5 的总体目标是：

$$
\boxed{
\text{完成 full control-positive frontier 判定，并建立第一个可 official 测量的 action-value controller + payload-apply runtime。}
}
$$

本轮不是为了在 MNIST / Fashion-MNIST / KMNIST 上调榜。数据集只用于诊断泛化、hardness、action density 和 failure mode，不允许出现 dataset-specific threshold / branch / feature route。

v9.3.5 必须回答以下六个问题：

```text
Q1. Full universe 中 control-positive oracle frontier 是否真实存在？
Q2. v9.3.4 raw coverage 低，是 panel coverage artifact，还是 action primitive density 不足？
Q3. 如果 full oracle pass，legal static features 是否能识别 control-positive action？
Q4. 如果 static features 不够，legal pre-commit action-response probe 是否能显著提高 observability？
Q5. 如果 full oracle fail，哪些 action primitive / candidate generator 方向能提高 control-positive density？
Q6. official controller + payload apply online runtime 是否能进入 step_ratio_q90 <= 1.50？
```

强目标：

```text
full_control_outcome_ready = 1
control_positive_oracle_pass = 1
legal_observability_pass = 1
decision_gate_pass = 1
payload_apply_online_runtime_pass = 1
system_legal_controller_pass = 1
```

最低有效推进：

```text
1. 完成剩余 36216 branch-horizon rows；
2. 给出 full-universe control-positive density 和 confidence interval；
3. 判断 v9.3.4 R5 是否是 panel artifact；
4. 至少测完一组 legal action-response probe；
5. 若 oracle fail，至少并行跑 3 类 action primitive density reset scout；
6. 测量 payload-apply online runtime upper/lower bounds；
7. 给出明确 route：oracle pass / observability fail / primitive density fail / runtime fail。
```

---

# 3. 核心定义

## 3.1 Branch value

对 action $e$、branch $b$、horizon $h$，定义 branch value：

$$
V_b(e,h)
=
-w_1\Delta CEp99_b(e,h)
+w_2\Delta MarginP10_b(e,h)
-w_3\Delta ECE_b(e,h)
-w_4\Delta NLL_b(e,h)
-w_5\Delta Curvature_b(e,h).
$$

权重 $w_i$ 必须在 protocol 中预注册，不得按 dataset 调参。建议默认：

```text
w1 = 1.0
w2 = 0.5
w3 = 0.5
w4 = 0.5
w5 = 0.25
```

v9.3.5 应同时记录 raw metrics，避免只依赖 composite value。

## 3.2 Control-positive value

Controls：

$$
\mathcal{C}=
\{
AdamWParallel,\ bestLR,\ AdamWOnly,\ NoOp,\ Random
\}.
$$

定义：

$$
V_{ctrl}(e,h)
=
V_{RealFunctional}(e,h)
-
\max_{b\in\mathcal{C}}V_b(e,h).
$$

## 3.3 Control-positive label

Weak control-positive：

$$
CP_{weak}(e)=1
$$

当且仅当：

```text
primary_safe_good = 1
bad_event = 0
null_event = 0
V_ctrl(e, h*) > 0 for official horizon h*
beats_adamwparallel = 1
beats_bestlr = 1
beats_noop = 1
beats_random = 1
```

Strong control-positive：

$$
CP_{strong}(e)=1
$$

当且仅当：

```text
CP_weak = 1
V_ctrl(e,h) > 0 for at least 2 of {20,80,240}
no horizon has bad_event = 1
horizon_instability <= tau_h
```

Official controller 默认以 $CP_{weak}$ 作为最低 gate，以 $CP_{strong}$ 作为强证据。

## 3.4 Oracle coverage

Raw coverage：

$$
Coverage_{raw}=\frac{Accepted_{CP}}{N_{heldout}}.
$$

Panel-adjusted coverage：

$$
Coverage_{panel-adjusted}
=
\frac{\sum_{e\in panel}w_e CP(e)}{N_{heldout}}.
$$

其中 $w_e$ 是 stratified inverse sampling weight。如果 panel 是 uniform action sample：

$$
w_e=\frac{N_{full}}{N_{panel}}.
$$

Full-universe coverage：

$$
Coverage_{full}
=
\frac{\sum_{e\in full}CP(e)}{N_{heldout}}.
$$

v9.3.5 的 official oracle 只能使用 $Coverage_{full}$ 或严格 stratified estimator 的 confidence lower bound，不能只用 raw 30% panel count 判死刑。

---

# 4. 硬约束

本轮继续遵守：

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
official threshold 不按 dataset 调
diagnostic paired replay 不得影响 official controller
```

Functional update 仍然是 update rule：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{AdamW}
+
\Delta\theta_{functional}.
$$

任务 loss 仍然是标准 CE：

$$
L_{task}=CE(y,p_\theta(x)).
$$

v9.3.5 新增硬约束：

```text
1. 不能把 30% panel raw coverage 当成 full oracle absence。
2. 不能把 PayloadLoadTime 当成 mechanistic value feature 转正，除非通过 artifact-side-channel audit。
3. 不能在 full control outcome 未完成时声明 oracle absent。
4. 不能把 runtime smoke 写成 official runtime pass。
5. official runtime 必须包含 payload apply 或明确证明 accepted_count=0 不是 controller success。
6. 如果 oracle density fail，必须 pivot action primitive / candidate generator，不得继续调 controller threshold。
7. 如果 oracle density pass 但 observability fail，必须设计 legal pre-commit probe 或新 observable，不得继续堆 payload norm 类弱 feature。
```

---

# 5. v9.3.5 核心假设

## H1：v9.3.4 的 low raw coverage 可能是 30% panel artifact，而非 full oracle absence

H1 成立标准：

```text
Full universe materialized 后：
  control_positive_accepted_count_full >= 273
  coverage_full >= 0.03
  precision_control_positive >= 0.90
  bad_event_rate <= 0.05
  null_rate <= 0.10
  V_ctrl_LCB > 0
```

H1 失败标准：

```text
Full universe materialized 后：
  coverage_full < 0.03
  and panel-adjusted estimate also < 0.03
  and support balance cannot rescue coverage
```

如果 H1 失败，不能继续把 controller 当主 blocker，应进入 action primitive density reset。

## H2：control-positive rows 的质量高，但密度可能不足

v9.3.4 已显示 panel control-positive rows：

```text
precision_control_positive = 1.0
bad_event = 0
null_rate = 0
beats all listed controls = 1.0
V_ctrl_mean = 0.403
```

H2 认为：当前 action primitive 可能不是完全错，而是产生的 high-value actions 太稀疏。

H2 成立标准：

```text
full coverage in [0.01,0.03)
precision_control_positive >= 0.90
V_ctrl_LCB > 0
support_balance_pass = 1
```

这时应做 action density improvement，而不是丢弃现有 primitive。

## H3：static payload features 不是 action-value sufficient statistic

H3 成立标准：

```text
PayloadNorm / PayloadLinf / PayloadLoadTime 等 static features:
  AUC_control_positive < 0.65
  PR-AUC lift weak
  calibration ECE high
  artifact-side-channel audit fail or inconclusive
```

如果 H3 成立，P4 不再扩展同类 static payload features，而转入 legal action-response probe。

## H4：legal pre-commit action-response probe 可以提供更强 observability

H4 认为：如果 action value 无法从 payload magnitude 看出，需要在 commit 前用低成本、训练批内、合法的 response measurement 估计 action 的局部效果。

Probe 不能使用 heldout/test，不得用未来 outcome label。它只能使用当前 training batch / calibration-authorized support statistics。

H4 成立标准：

```text
probe features:
  AUC_control_positive >= 0.75
  or PR-AUC lift >= 2.0
  bad/null separation improves
  feature_compute_time_ms_q90 within budget
  no dataset_name branch
```

H4 失败标准：

```text
all legal probes remain AUC <= 0.65
or runtime cost makes probe infeasible
```

若 H4 失败且 oracle pass，说明 current action-value is not legally observable enough under existing commit-time information。

## H5：official online runtime can pass only after payload apply is included

H5 成立标准：

```text
official controller selected
payload apply included in timed path
disk payload lookup excluded by preloading
zero-candidate controller kernel/sync = 0
controller_launches_per_active_step_q90 <= 2
payload_apply_error_max <= tolerance
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

H5 失败标准：

```text
no-payload smoke pass but payload-apply path pushes step_ratio_q90 > 1.50
```

## H6：dataset differences are diagnostics, not tuning targets

H6 成立标准：

```text
per-dataset density / value / AUC / failure mode recorded；
controller thresholds shared；
no dataset_name feature；
no dataset-specific route；
leave-dataset-out evaluated after system candidate exists。
```

H6 失败标准：

```text
any threshold / feature / action primitive selected because it helps one named dataset；
dataset_name_used_for_controller = 1。
```

---

# 6. Data contracts

## 6.1 Full control outcome table

每个 action、branch、horizon row 必须包含：

```text
run_id
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
payload_hash
checkpoint_hash_before
checkpoint_hash_after
branch_state_hash
CEp99_before
CEp99_after
CEp99_delta
margin_p10_before
margin_p10_after
margin_p10_delta
ECE_before
ECE_after
ECE_delta
NLL_before
NLL_after
NLL_delta
curvature_before
curvature_after
curvature_delta
acc_delta
task_safe_label
bad_event_label
null_event_label
safe_good_label
V_branch
V_ctrl
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
metric_nan_count
metric_inf_count
```

Pass：

```text
expected_rows = 51768
actual_rows = 51768
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1.0
duplicate_row_id_count = 0
branch_identity_violation_count = 0
horizon_state_hash_mismatch_count = 0
metric_nan_count = 0
metric_inf_count = 0
```

## 6.2 Panel representativeness contract

每个 action 记录：

```text
in_panel_v9340
panel_id
sampling_weight
sampling_stratum
family_id
horizon_bucket_id
dataset
seed
step_bin
stable_score_bin
candidate_origin_tag
payload_norm_bin
control_positive_label_if_materialized
```

必须输出：

```text
panel_fraction_by_dataset
panel_fraction_by_family
panel_fraction_by_bucket
panel_fraction_by_seed
panel_fraction_by_score_bin
panel_fraction_by_payload_norm_bin
panel_representativeness_chi2
panel_max_stratum_weight
coverage_raw
coverage_weighted
coverage_full
coverage_LCB
```

## 6.3 Legal feature contract

每个 candidate/action row 记录 feature groups：

```text
A_static_payload:
  payload_norm
  payload_linf
  payload_sparsity
  payload_role_entropy
  payload_group_norms
  payload_update_to_weight_norm_ratio

B_gradient_alignment:
  cos_func_delta_adamw_delta
  cos_func_delta_negative_grad
  projected_CE_descent
  projected_margin_gain
  gradient_conflict_rate
  AdamW_conflict_score

C_state_tail:
  batch_CE_mean
  batch_CE_p90
  batch_CE_p99
  batch_margin_p10
  wrong_conf_p90
  hard_tail_fraction
  entropy_mean
  confidence_gap

D_linearized_response:
  linearized_CE_delta
  linearized_margin_delta
  linearized_NLL_delta
  linearized_tail_CE_delta
  second_order_proxy
  linearization_reliability_lcb

E_support_calibration:
  family_support_count
  horizon_support_count
  feature_neighbor_count
  EB_value_lcb
  EB_bad_ucb
  EB_null_ucb
  calibration_bin_id
  support_lcb

F_legal_probe:
  probe_CE_delta
  probe_margin_delta
  probe_tail_CE_delta
  probe_wrong_conf_delta
  probe_cost_ms
  probe_branch_count
  probe_reliability_score

G_runtime_cost:
  candidate_count_in_step
  feature_compute_time_ms
  score_accept_time_ms
  payload_apply_time_ms_est
  active_step_flag
  runtime_bucket_id
```

Legality audit：

```text
uses_dataset_name = 0
uses_test_or_validation = 0
uses_future_outcome = 0
uses_posthoc_CP_label_at_commit = 0
uses_source_measured_gap = 0
formula_proxy_used_for_official = 0
feature_cost_recorded = 1
```

## 6.4 Runtime contract

每 measured online run 记录：

```text
runtime_candidate_id
controller_id
runtime_mode
step_count
active_step_count
zero_candidate_step_count
candidate_count
accepted_count
accepted_count_per_step_q90
zero_candidate_controller_kernel_count
zero_candidate_controller_sync_count
controller_launches_per_active_step_q90
controller_syncs_per_active_step_q90
feature_compute_time_ms_q90
probe_compute_time_ms_q90
score_accept_time_ms_q90
payload_prefetch_time_ms_q90
payload_apply_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
payload_apply_error_linf_max
payload_apply_cosine_min
accept_disagreement_count
no_event_preservation_pass
base_adamw_equivalence_zero_event_pass
materializer_in_timed_path
offline_audit_in_timed_path
disk_lookup_in_timed_path
```

Official pass：

```text
runtime_mode = online_sequential_official
official_controller_used = 1
payload_apply_in_timed_path = 1
materializer_in_timed_path = 0
offline_audit_in_timed_path = 0
disk_lookup_in_timed_path = 0
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
payload_apply_error_linf_max <= tolerance
```

---

# 7. 实验阶段

---

## P0：v9.3.4 boundary independent reanalysis

### 目标

重新计算 v9.3.4 的 gate gap，不直接接受 `ControlPositiveOracleAbsent` 这个 route 名称。P0 只做分析，不改变数据。

### 核心假设

H1：v9.3.4 raw coverage fail 可能是 panel raw-count artifact。

### 需要计算

```text
candidate_count_full = 2876
panel_action_count = 864
panel_fraction = 864 / 2876
heldout_denominator = 9072
control_positive_accepted_panel = 155
coverage_raw = 155 / 9072
accepted_scaled_uniform = 155 / panel_fraction
coverage_scaled_uniform = accepted_scaled_uniform / 9072
binomial_CI_for_panel_positive_rate
stratified_weighted_coverage_if_sampling_weights_available
coverage_gap_to_0p03
accepted_needed_for_raw_gate = ceil(0.03 * 9072)
raw_accepted_shortfall = accepted_needed_for_raw_gate - 155
full_oracle_uncertainty_class
```

### Pass / fail

P0 pass：

```text
coverage ambiguity classified；
raw panel coverage and panel-adjusted coverage both reported；
route renamed if necessary:
  R5a-PanelRawCoverageBelowGate
  R5b-FullOracleUnresolved
  R5c-FullOracleAbsent
```

### 可视化

```text
p0_panel_vs_scaled_coverage.svg
p0_coverage_ci_bar.svg
p0_panel_sampling_strata_heatmap.svg
p0_v9330_to_v9340_materializer_progress.svg
```

---

## P1：full control outcome universe completion

### 目标

完成剩余 control outcome rows。v9.3.4 已证明吞吐足够；v9.3.5 不应继续停在 30% panel。

### 实现

新增 runner：

```text
experiments/run_v9350_full_control_outcome_completion.py
```

运行策略：

```text
1. 读取 v9.3.4 full_control_outcome_table_v9340.csv；
2. 识别 remaining actions = full_action_set - panel_a_actions；
3. 复用 MAT2 checkpoint / payload shard / horizon cache；
4. 按 action shard 并行或顺序补齐；
5. 失败 rows 写 retry manifest；
6. 所有 rows 通过 P4-style quality audit；
7. 输出 full_control_outcome_table_v9350.csv。
```

推荐参数：

```bash
python experiments/run_v9350_full_control_outcome_completion.py \
  --source-v9340 results/real_rerun_20260506/v9340_control_outcome_materializer_scaleup_action_value_runtime_remeasure_first_20260514T010000Z \
  --out-dir results/real_rerun_20260506/v9350_full_control_outcome_completion_first \
  --fresh --device auto --data-root data --seed 1314 \
  --complete-all-actions 1 \
  --shard-size-actions 288 \
  --branches RealFunctional,AdamWParallel,bestLR,AdamWOnly,NoOp,Random \
  --horizons 20,80,240
```

### 必须记录

```text
action_count_expected
action_count_completed
row_count_expected
row_count_actual
row_count_reused_from_v9340
row_count_newly_materialized
row_count_failed
row_count_retried
rows_per_sec_total
wallclock_sec
branch_completion_rate
horizon_completion_rate
secondary_delta_completion_rate
missing_branch_count
missing_horizon_count
missing_secondary_delta_count
quality_audit_pass
```

### 判断标准

P1 pass：

```text
action_count_completed = 2876
row_count_actual = 51768
missing_branch_count = 0
missing_horizon_count = 0
missing_secondary_delta_count = 0
quality_audit_pass = 1
rows_per_sec_total >= 5.0
```

Strong pass：

```text
rows_per_sec_total >= 20.0
unresolved_failed_rows = 0
full_control_outcome_ready = 1
```

### 可视化

```text
p1_materializer_completion_curve.svg
p1_rows_per_sec_by_shard.svg
p1_failed_retry_rows.svg
p1_branch_horizon_completion_heatmap.svg
p1_quality_audit_dashboard.svg
```

---

## P2：full control-positive oracle and density classification

### 目标

在 full outcome universe 上重新计算 control-positive oracle，判断 v9.3.4 的 raw coverage fail 是 panel artifact 还是 action density 问题。

### 核心假设

H1 / H2：control-positive frontier 可能存在，但 panel raw count 不足以 official 判断。

### Oracle candidates

```text
OR0-v9340-panel-raw-reference
OR1-FullWeakControlPositiveOracle
OR2-FullStrongControlPositiveOracle
OR3-HorizonRobustControlPositiveOracle
OR4-ValueRiskParetoOracle
OR5-SupportBalancedControlPositiveOracle
OR6-DensityRelaxedHighValueOracle
```

### 必须记录

```text
oracle_id
candidate_universe
accepted_count
coverage
coverage_lcb
precision_primary
precision_control_positive
bad_event_rate
null_rate
V_ctrl_mean
V_ctrl_median
V_ctrl_lcb
V_ctrl_p10
beats_adamwparallel_rate
beats_bestlr_rate
beats_noop_rate
beats_random_rate
horizon_20_positive_rate
horizon_80_positive_rate
horizon_240_positive_rate
horizon_robust_positive_rate
accepted_dataset_count
accepted_family_count
accepted_signal_strata_count
max_dataset_share
max_family_share
max_stratum_share
support_balance_pass
```

### 判断标准

Full oracle strong pass：

```text
coverage >= 0.03
coverage_lcb >= 0.03
precision_control_positive >= 0.90
bad_event_rate <= 0.05
null_rate <= 0.10
V_ctrl_lcb > 0
beats_adamwparallel_rate >= 0.90
beats_bestlr_rate >= 0.90
support_balance_pass = 1
```

Full oracle weak pass：

```text
coverage >= 0.03
precision_control_positive >= 0.75
bad_event_rate <= 0.05
V_ctrl_mean > 0
support_balance_pass = 1
```

Sparse high-quality frontier：

```text
coverage in [0.01, 0.03)
precision_control_positive >= 0.90
V_ctrl_lcb > 0
bad_event_rate <= 0.05
```

Oracle absent：

```text
coverage < 0.01
or V_ctrl_lcb <= 0
or beats_adamwparallel_rate < 0.60
or beats_bestlr_rate < 0.60
```

### 路由

```text
if full_oracle_strong_pass:
  route = R6-ControlPositiveFullOraclePass
  proceed P4/P5/P6 controller.

if sparse_high_quality_frontier:
  route = R6b-ControlPositiveSparseHighValue
  run P7 action-density reset in parallel with P4/P5.

if oracle_absent:
  route = R30-ActionPrimitiveDensityInsufficient
  stop controller search; prioritize P7 primitive reset.
```

### 可视化

```text
p2_full_oracle_coverage_value_frontier.svg
p2_control_positive_density_by_dataset.svg
p2_control_positive_density_by_family_horizon.svg
p2_Vctrl_distribution.svg
p2_real_vs_controls_pareto.svg
p2_horizon_robustness_matrix.svg
p2_panel_vs_full_oracle_comparison.svg
```

---

## P3：panel-to-full representativeness audit

### 目标

判断 v9.3.4 的 30% panel 是否可以作为未来 official-minimum panel，还是只能作为 smoke panel。

### 核心假设

H1：panel 如果有系统性 bias，raw coverage 不能用于 full universe conclusion。

### 必须记录

```text
stratum_id
stratum_definition
full_action_count
panel_action_count
panel_fraction
full_CP_count
panel_CP_count
CP_rate_full
CP_rate_panel
absolute_rate_error
relative_rate_error
sampling_weight
weighted_CP_estimate
weighted_coverage_estimate
weighted_coverage_error
```

Strata：

```text
dataset
seed
family_id
bucket_id
horizon_group
stable_score_bin
payload_norm_bin
action_family
step_bin
candidate_origin_tag
```

### 判断标准

Panel representative pass：

```text
max_stratum_coverage_gap <= 0.20
weighted_coverage_error <= 0.005
panel_CP_rate_error_macro <= 0.05
```

Fail：

```text
panel raw coverage differs from full coverage enough to change route decision
```

### 可视化

```text
p3_panel_full_stratum_coverage_heatmap.svg
p3_panel_vs_full_CP_rate_scatter.svg
p3_sampling_weight_distribution.svg
p3_route_flip_panel_vs_full.svg
```

---

## P4：legal static action-value observability rebuild

### 目标

在 full labels 上重新评估 legal static features。P4 不是为了堆 feature，而是判断 static information 是否足以识别 $CP(e)$。

### 核心假设

H3：payload magnitude 类 static features不是 sufficient statistic；需要 gradient/action/state/legal-response features。

### Feature groups

```text
F0-v9340-PayloadNormLinfLoadTime reference
F1-PayloadStructure
F2-GradientActionAlignment
F3-StateTailRisk
F4-LinearizedResponse
F5-SupportEmpiricalBayes
F6-HorizonRobustnessPrior
F7-AdamWConflictAndComplementarity
F8-MinimalValueRiskSupportCost
```

Minimal score：

$$
S_{static}(e)
=
a_1 LCB(\widehat V(e))
-a_2 UCB(\widehat B(e))
-a_3 UCB(\widehat N(e))
+a_4 LCB(\widehat S(e))
-a_5 \widehat C(e).
$$

约束：

```text
a_i >= 0
feature_groups <= 5
dataset_name_used = 0
all thresholds frozen on calibration split
```

### 必须记录

```text
feature_group
feature_count
uses_dataset_name
uses_future_outcome
uses_validation_or_test
feature_missing_rate
feature_compute_time_ms_q90
AUC_CP_weak
AUC_CP_strong
PR_AUC_CP_lift
AUC_bad_event
AUC_null_event
AUC_Vctrl_positive
Spearman_Vctrl
Brier_CP
ECE_CP
calibration_slope
leave_dataset_auc_drop
leave_family_auc_drop
leave_stratum_auc_drop
artifact_side_channel_risk
minimality_pass
monotone_sign_pass
```

### 判断标准

Static observability pass：

```text
AUC_CP_weak >= 0.75
or PR_AUC_CP_lift >= 2.0
and ECE_CP <= 0.08
and feature_compute_time_ms_q90 <= 0.05
and leave_dataset_auc_drop <= 0.07
and feature_groups <= 5
and monotone_sign_pass = 1
```

Weak pass：

```text
AUC_CP_weak in [0.65,0.75)
or PR_AUC_CP_lift >= 1.5
```

Fail：

```text
all legal static feature groups AUC_CP < 0.65
```

### 可视化

```text
p4_static_feature_auc_bar.svg
p4_PR_AUC_lift.svg
p4_Vctrl_prediction_scatter.svg
p4_CP_calibration_curve.svg
p4_feature_cost_vs_auc_pareto.svg
p4_leaveout_auc_drop_heatmap.svg
p4_feature_ablation_waterfall.svg
```

---

## P5：legal pre-commit action-response probe

### 目标

如果 static observability 不足，增加一个原则性而非 patchy 的 legal probe：在 commit 前，用当前 train batch / candidate batch 对 action 做低成本 response measurement。

### 核心假设

H4：control-positive action 的价值需要 action-response observable，而不是 payload magnitude observable。

### Probe candidates

#### PR0：No-probe reference

```text
使用 P4 static features。
```

#### PR1：mini-batch logit response probe

对 current train batch 计算 action 前后 logits 的低成本差分：

$$
ProbeCE(e)=CE(f_{\theta+\epsilon\Delta\theta_e}(x_{batch}),y)-CE(f_{\theta}(x_{batch}),y).
$$

只允许使用 train batch，不允许 validation/test。

#### PR2：tail-slice response probe

只在当前 batch 的 hard-tail slice 上计算：

```text
CEp99 proxy
margin_p10 proxy
wrong_conf_p90 proxy
```

#### PR3：gradient-JVP response probe

用 manual gradient/JVP 估计：

$$
\widehat{\Delta CE}_{JVP}(e)=\nabla_\theta CE(\theta)^T\Delta\theta_e.
$$

#### PR4：AdamW conflict probe

测：

$$
Conflict(e)=1-\cos(\Delta\theta_e,\Delta\theta_{AdamW}).
$$

#### PR5：two-scale response probe

用两个 scale $\epsilon_1,\epsilon_2$ 检查 linearity：

$$
Reliability(e)=1-\left|\frac{Probe_{\epsilon_2}}{2Probe_{\epsilon_1}}-1\right|.
$$

#### PR6：batched native probe

把 PR1-PR5 合并成 batched kernel / manual forward path，测 runtime feasibility。

### 必须记录

```text
probe_id
candidate_id
action_id
probe_scale
probe_batch_size
probe_tail_slice_size
probe_CE_delta
probe_margin_delta
probe_CEp99_delta
probe_wrong_conf_delta
probe_entropy_delta
probe_JVP_delta
probe_adamw_conflict
probe_linearity_error
probe_reliability
probe_compute_time_ms
probe_memory_ratio
AUC_CP_weak
AUC_CP_strong
PR_AUC_CP_lift
AUC_bad_event
AUC_null_event
ECE_CP
```

### 判断标准

Probe observability pass：

```text
AUC_CP_weak >= 0.78
or PR_AUC_CP_lift >= 2.5
and ECE_CP <= 0.06
and probe_compute_time_ms_q90 <= 0.15
and memory_ratio <= 1.05
and leave_dataset_auc_drop <= 0.07
```

Runtime feasibility：

```text
probe + controller + payload apply step_ratio_q90 <= 1.50
```

Fail：

```text
probe improves AUC < 0.05 over static features
or cost makes system impossible
```

### 可视化

```text
p5_probe_feature_auc.svg
p5_probe_cost_auc_pareto.svg
p5_probe_linearity_reliability.svg
p5_probe_static_combined_ablation.svg
p5_probe_response_vs_Vctrl.svg
p5_tail_probe_failure_modes.svg
```

---

## P6：cross-fitted control-positive controller

### 目标

只有 P2 oracle pass 或 sparse high-quality frontier 时打开。建立不使用 dataset branch 的 controller，直接选择 control-positive actions。

### Controller candidates

```text
C0-StableAcceptNegativeControl
C1-PayloadStaticReference
C2-StaticMinimalVBNCS
C3-ProbeMinimalVBNCS
C4-StaticPlusProbeMonotoneScore
C5-TwoStageCheapStaticThenProbe
C6-HorizonRobustController
C7-CostAwareController
```

Controller 形式：

$$
Accept(e)=1
\iff
LCB(\widehat V_{ctrl}(e))>0
\land
UCB(\widehat B(e))\le\tau_b
\land
UCB(\widehat N(e))\le\tau_n
\land
LCB(\widehat S(e))\ge\tau_s
\land
\widehat C(e)\le C_{max}.
$$

### Splits

```text
Seed folds:
  calibration on seed groups；
  heldout seed groups；
  rotate if enough runs exist。

Leave-dataset-out:
  train/calibrate on two datasets；
  evaluate third。

Leave-family-out:
  hold out high-volume families。

Leave-stratum-out:
  hold out signal strata / feature strata。
```

### 必须记录

```text
controller_id
feature_set
probe_used
feature_group_count
thresholds
calibration_fold
heldout_fold
dataset_name_used
accepted_count_cal
accepted_count_heldout
coverage_cal
coverage_heldout
precision_primary_cal
precision_primary_heldout
precision_control_positive_cal
precision_control_positive_heldout
bad_event_cal
bad_event_heldout
null_rate_cal
null_rate_heldout
V_ctrl_mean_cal
V_ctrl_mean_heldout
V_ctrl_lcb_heldout
beats_adamwparallel_rate
beats_bestlr_rate
beats_noop_rate
beats_random_rate
precision_lcb
bad_event_ucb
coverage_lcb
accepted_dataset_count
accepted_family_count
accepted_signal_strata_count
max_dataset_share
max_family_share
max_stratum_share
```

### 判断标准

Decision pass：

```text
coverage_heldout in [0.03,0.15]
precision_control_positive_heldout >= 0.75
precision_primary_heldout >= 0.75
bad_event_heldout <= 0.05
null_rate_heldout <= 0.15
V_ctrl_lcb_heldout > 0
beats_adamwparallel_rate >= 0.60
beats_bestlr_rate >= 0.60
precision_lcb >= 0.75
bad_event_ucb <= 0.05
dataset_name_used = 0
```

Strong pass：

```text
precision_control_positive_heldout >= 0.90
coverage_lcb >= 0.03
V_ctrl_lcb_heldout > 0
LDO pass in at least 2/3 heldout datasets
```

### 可视化

```text
p6_controller_precision_coverage_value_frontier.svg
p6_controller_calibration_to_heldout_drift.svg
p6_Vctrl_lcb_by_controller.svg
p6_control_beat_rate_matrix.svg
p6_support_balance_sunburst.svg
p6_leaveout_controller_matrix.svg
```

---

## P7：action primitive density reset scouts

### 目标

如果 full oracle sparse 或 absent，不能继续修 controller。必须提高 action primitive 产生 control-positive candidates 的密度。

### 核心假设

H2：当前 action primitive 可能质量高但密度低。需要 candidate/action generator reset，而非 threshold patch。

### Action primitive candidates

#### AP0-current-reference

当前 v9.3.4 action primitive，作为 baseline。

#### AP1-gradient-aligned-functional-projection

将 functional payload 投影到不与 AdamW / negative gradient 冲突的子空间：

$$
\Delta\theta'_{func}
=
\Delta\theta_{func}
-
\lambda \operatorname{Proj}_{conflict}(\Delta\theta_{func}).
$$

Scale 不按 dataset 调，而按 norm ratio：

$$
r=\frac{\|\Delta\theta'_{func}\|}{\|\Delta\theta_{AdamW}\|}.
$$

#### AP2-tail-risk-constrained-generator

生成 action 前加入 hard-tail safety constraint：

```text
candidate action must not increase current-batch CEp99 proxy；
candidate action must not reduce margin_p10 proxy below threshold。
```

#### AP3-horizon-robust-generator

优先生成在 $20/80/240$ horizon proxy 上更稳定的 action，不只优化短 horizon。

#### AP4-value-diversity-generator

增加 action family diversity，避免只产生少量高-value action：

```text
role-channel diversity
basis-channel diversity
payload direction diversity
family quota without dataset branch
```

#### AP5-small-scale-multi-action-generator

对同一 event 生成多个 scale variants：

$$
\Delta\theta_{func}^{(s)}=s\Delta\theta_{func},
$$

其中 $s$ 由 global norm-ratio grid 给出，不按 dataset 调。

#### AP6-probe-guided-generator

用 P5 legal probe 在 commit 前筛掉明显 bad/null actions，但 action 生成本身仍不使用 outcome label。

### Scout design

对每个 AP，跑小规模但 stratified 的 control outcome materializer：

```text
actions_per_AP = 288 or 432
branches = RealFunctional, AdamWParallel, bestLR, AdamWOnly, NoOp, Random
horizons = 20,80,240
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = shared
```

### 必须记录

```text
action_primitive_id
candidate_count
action_count
candidate_rate
action_rate
control_positive_count
control_positive_coverage
control_positive_density
precision_control_positive
bad_event_rate
null_rate
V_ctrl_mean
V_ctrl_lcb
beats_adamwparallel_rate
beats_bestlr_rate
horizon_robust_positive_rate
feature_observability_auc
runtime_cost_estimate
payload_apply_cost_estimate
dataset_name_used
```

### 判断标准

Primitive scout pass：

```text
control_positive_density >= 1.5 * AP0_density
or coverage >= 0.03 with precision_control_positive >= 0.75
and bad_event_rate <= 0.05
and V_ctrl_lcb > 0
and dataset_name_used = 0
```

Strong primitive pass：

```text
coverage >= 0.05
precision_control_positive >= 0.90
V_ctrl_lcb > 0
horizon_robust_positive_rate >= 0.50
```

### 可视化

```text
p7_action_primitive_density_frontier.svg
p7_AP_value_risk_pareto.svg
p7_AP_control_beat_matrix.svg
p7_AP_horizon_robustness.svg
p7_AP_observability_vs_density.svg
p7_AP_runtime_cost_pareto.svg
```

---

## P8：online runtime with payload apply

### 目标

把 v9.3.4 的 no-payload smoke 升级为 official payload-apply online runtime measurement。

### Runtime candidates

```text
RT0-v9340-no-payload-smoke-reference
RT1-static-controller-payload-apply
RT2-probe-controller-payload-apply
RT3-two-stage-static-probe-payload-apply
RT4-synthetic-coverage-0p03-payload-upperbound
RT5-oracle-mask-diagnostic-payload-upperbound
RT6-official-controller-full-stream
```

RT4 / RT5 只做 runtime upper/lower bound，不得 official。RT6 只有 P6 controller selected 后才能 official。

### 必须记录

```text
runtime_candidate_id
controller_id
probe_used
payload_apply_used
accepted_count
accepted_count_per_active_step_q90
step_count
active_step_count
zero_candidate_step_count
candidate_count
zero_candidate_controller_kernel_count
zero_candidate_controller_sync_count
controller_launches_per_active_step_q90
controller_syncs_per_active_step_q90
feature_compute_time_ms_q90
probe_compute_time_ms_q90
score_accept_time_ms_q90
payload_apply_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
payload_apply_error_linf_max
payload_apply_cosine_min
accept_disagreement_count
no_event_preservation_pass
base_adamw_equivalence_zero_event_pass
```

### 判断标准

Payload runtime official pass：

```text
official_controller_used = 1
payload_apply_used = 1
materializer_in_timed_path = 0
offline_audit_in_timed_path = 0
disk_lookup_in_timed_path = 0
zero_candidate_controller_kernel_count = 0
zero_candidate_controller_sync_count = 0
controller_launches_per_active_step_q90 <= 2
controller_syncs_per_active_step_q90 <= 1
payload_apply_error_linf_max <= tolerance
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

Diagnostic pass：

```text
synthetic coverage 0.03 payload apply ratio <= 1.50
```

### 可视化

```text
p8_runtime_payload_apply_waterfall.svg
p8_no_payload_vs_payload_step_ratio.svg
p8_accepted_per_step_hist.svg
p8_probe_cost_runtime_pareto.svg
p8_payload_apply_error_hist.svg
p8_online_runtime_q50_q90.svg
```

---

## P9：system integration gate

### 目标

组合 P6 controller 与 P8 payload-apply runtime，生成 system-legal candidate。只要 decision、runtime、contract 任一失败，P9 不得 pass。

### 必须记录

```text
system_candidate_id
controller_id
runtime_candidate_id
action_primitive_id
full_control_outcome_ready
control_positive_oracle_pass
legal_observability_pass
decision_gate_pass
payload_apply_runtime_pass
payload_binding_pass
action_lifecycle_pass
candidate_lifecycle_pass
dataset_name_used
loss_modification_used
teacher_used
fake_data_used
proxy_row_used
diagnostic_promoted_to_official
precision_control_positive_heldout
precision_primary_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
V_ctrl_lcb_heldout
beats_adamwparallel_rate
beats_bestlr_rate
step_ratio_q90
memory_ratio
official_eligible
system_legal_controller_pass
```

### 判断标准

System pass：

```text
full_control_outcome_ready = 1
control_positive_oracle_pass = 1
legal_observability_pass = 1
decision_gate_pass = 1
payload_apply_runtime_pass = 1
payload_binding_pass = 1
action_lifecycle_pass = 1
dataset_name_used = 0
loss_modification_used = 0
teacher_used = 0
fake_data_used = 0
proxy_row_used = 0
official_eligible = 1
system_legal_controller_pass = 1
```

### 可视化

```text
p9_system_gate_dashboard.svg
p9_quality_cost_frontier.svg
p9_route_decision_matrix.svg
p9_failure_attribution_sankey.svg
```

---

## P10：dataset diagnostics without tuning

### 目标

诊断 MNIST / Fashion-MNIST / KMNIST 的差异，但不按 dataset 调参。

### 必须记录

```text
dataset
control_positive_density
coverage
precision_control_positive
bad_event_rate
null_rate
V_ctrl_mean
V_ctrl_lcb
feature_AUC_CP
probe_AUC_CP
controller_accept_rate
controller_precision
controller_bad_event
controller_Vctrl
runtime_step_ratio
failure_mode_distribution
```

### 判断标准

Dataset diagnostic pass：

```text
all datasets reported；
no dataset_name_used = 1；
thresholds identical；
differences attributed to density / hardness / feature observability / runtime only。
```

### 可视化

```text
p10_dataset_density_value_matrix.svg
p10_dataset_feature_observability_matrix.svg
p10_dataset_failure_mode_sankey.svg
p10_dataset_no_tuning_audit.svg
```

---

## P11：leave-dataset-out / leave-stratum-out

### 目标

只有 P9 system candidate 产生后打开 official leave-out。证明 controller 不是 pooled artifact。

### 设置

```text
Leave-dataset-out:
  train/calibrate MNIST + Fashion, evaluate KMNIST
  train/calibrate MNIST + KMNIST, evaluate Fashion
  train/calibrate Fashion + KMNIST, evaluate MNIST

Leave-stratum-out:
  hold out family / score-bin / payload-norm-bin / support-bin strata
```

### 必须记录

```text
split_type
heldout_entity
controller_id
feature_set
probe_used
thresholds
coverage
precision_control_positive
precision_primary
bad_event_rate
null_rate
V_ctrl_lcb
beats_adamwparallel_rate
beats_bestlr_rate
step_ratio_q90
memory_ratio
dataset_name_used
support_balance_pass
```

### 判断标准

LDO pass：

```text
at least 2/3 heldout datasets pass:
  coverage > 0
  precision_control_positive >= 0.75
  bad_event_rate <= 0.10
  V_ctrl_mean > 0
no heldout dataset has catastrophic task safety fail
dataset_name_used = 0
```

LSO pass：

```text
macro precision_control_positive >= 0.75
macro bad_event_rate <= 0.05
coverage > 0 in >=70% heldout strata
```

### 可视化

```text
p11_LDO_matrix.svg
p11_LSO_matrix.svg
p11_leaveout_Vctrl_distribution.svg
p11_support_balance_leaveout.svg
```

---

## P12：diagnostic paired replay scout

### 目标

加快并行验证，但不得影响 P6/P9 official controller。P12 结果只能用于未来优先级，不得调本轮 threshold。

### 隔离规则

```text
diagnostic_downstream_used_for_controller = 0
diagnostic_threshold_feedback_used = 0
diagnostic_feature_selection_feedback_used = 0
```

### 设置

```text
controllers = top2 P6 candidates + AP scouts + negative controls
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
horizons = 20,80,240
branches = RealFunctional, AdamWParallel, bestLR, AdamWOnly, NoOp, Random, shuffled payload, shuffled score
```

### 必须记录

```text
controller_id
action_primitive_id
dataset
seed
horizon
branch
accepted_count
coverage
precision_control_positive
bad_event_rate
null_rate
V_ctrl
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
```

### 判断标准

Scout promising：

```text
RealFunctional beats AdamWParallel in >=50% macro slices
RealFunctional beats bestLR in >=50% macro slices
task_safe holds
shuffled controls do not match RealFunctional
```

### 可视化

```text
p12_diagnostic_replay_beat_matrix.svg
p12_shuffle_control_matrix.svg
p12_replay_value_risk_pareto.svg
```

---

## P13：official paired replay

### 前置条件

```text
P9 system_legal_controller_pass = 1
P11 leave-out pass = 1
diagnostic isolation audit pass = 1
```

### 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0..4
horizons = 20,80,240,640
branches =
  RealFunctional
  AdamWParallel
  bestLR
  AdamWOnly
  NoOp
  Random
  ShuffledFunctionalPayload
  ShuffledActionScore
  ShuffledProbeScore
  ShuffledSupportScore
  ShuffledCandidatePayload
  ShuffledRuntimePath
```

### 必须记录

```text
controller_id
action_primitive_id
dataset
seed
horizon
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
V_ctrl
real_beats_adamwparallel
real_beats_bestlr
task_safe
event_count
candidate_rate
coverage
bad_event_rate
null_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Paired replay pass：

```text
macro BeatRate Real vs AdamWParallel >= 0.60
macro BeatRate Real vs bestLR >= 0.60
Acc_real >= Acc_AdamW - 0.005 for every official slice
shuffled controls fail to match RealFunctional pattern
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

### 可视化

```text
p13_official_paired_replay_pareto.svg
p13_macro_beat_rate.svg
p13_signal_stratum_win_matrix.svg
p13_shuffle_control_matrix.svg
p13_task_safety_by_slice.svg
```

---

## P14：short-run / full-run / sample efficiency / continual / robustness

### 前置条件

```text
P13 official paired replay pass = 1
```

### 设置

```text
short_run_steps = 50,240,640
full_run_seeds = 0..9
datasets = MNIST, Fashion-MNIST, KMNIST
controls =
  AdamWOnly
  AdamWParallel
  bestLR
  StrongLRGrid
  QuadraticFeatureMLP
  NoOp
  Random
  ShuffledFunctionalPayload
  ShuffledProbeScore
  ShuffledSupportScore
```

### 必须记录

```text
dataset
seed
candidate
steps
final_acc
best_val_acc
test_acc
train_loss
val_loss
test_loss
ValLossAUC_step
ValLossAUC_time
time_to_target
steps_to_target
ECE
NLL
CEp99
margin_p10
curvature
local_lipschitz
functional_event_count
coverage
bad_event_rate
null_rate
step_ratio_q90
memory_ratio
continual_task_id
retained_accuracy
forgetting
backward_transfer
forward_transfer
old_task_CEp99
old_task_margin_p10
strong_baseline_beaten
robustness_pass
```

### 判断标准

Full functional pass：

```text
task safe:
  Acc_functional >= Acc_AdamW - 0.005

and at least one advantage:
  Acc_functional > Acc_AdamWParallel
  or ECE_functional < ECE_AdamW
  or NLL_functional < NLL_AdamW
  or Curvature_functional <= 0.90 * Curvature_AdamW
  or ValLossAUC_time improves
  or time_to_target improves

strong baseline:
  not explained by LR grid
  not explained by QuadraticFeatureMLP
  not explained by shuffled controller/payload/probe
```

Continual pass：

```text
retained_accuracy drop <= AdamW retained_accuracy drop
forgetting <= AdamW forgetting
backward_transfer >= AdamW backward_transfer - tolerance
```

### 可视化

```text
p14_short_full_learning_curves.svg
p14_val_loss_auc_time.svg
p14_time_to_target.svg
p14_calibration_geometry_frontier.svg
p14_continual_forgetting_matrix.svg
p14_strong_baseline_comparison.svg
```

---

# 8. 并行执行计划

## Batch A：必须立即完成的闭环

```text
A1. P0 v9.3.4 independent reanalysis
A2. P1 full control outcome completion
A3. P2 full control-positive oracle
A4. P3 panel representativeness audit
```

这些可以直接并行或流水线执行。P1 完成后 P2/P3 立刻重算。

## Batch B：feature / probe / runtime 并行

在 P1 rows streaming 过程中即可开始：

```text
B1. P4 static legal features on v9.3.4 panel, then full table
B2. P5 legal probe implementation smoke
B3. P8 synthetic payload-apply runtime upperbound
B4. P10 dataset diagnostics without tuning
```

## Batch C：controller 与 primitive scout 分叉

P2 route 出来后：

```text
if oracle pass:
  C1. P6 cross-fitted control-positive controller
  C2. P8 official payload-apply runtime for controller

if sparse high-quality:
  C1. P6 controller attempt
  C2. P7 action primitive density reset scouts
  C3. P8 runtime upperbound

if oracle absent:
  C1. P7 action primitive density reset scouts
  C2. stop official controller search for AP0
```

## Batch D：system / leave-out / replay

```text
D1. P9 system integration
D2. P11 LDO/LSO
D3. P12 diagnostic paired replay isolated
D4. P13 official paired replay if gates pass
```

## Batch E：longer run

```text
E1. P14 short-run
E2. P14 full-run
E3. sample efficiency / continual / robustness
```

---

# 9. Required artifacts

```text
run_manifest.json
contract_audit_v9350.csv
provenance_audit_v9350.csv
artifact_hashes.csv
route_decision.json
aggregate_decision.json
failure_table.csv

p0_v9340_independent_reanalysis.csv
p1_full_control_outcome_completion.csv
full_control_outcome_table_v9350.csv
branch_horizon_completion_trace_v9350.csv
materializer_worker_trace_v9350.csv
materializer_retry_manifest_v9350.csv
p2_full_control_positive_oracle.csv
control_positive_oracle_trace_v9350.csv
p3_panel_representativeness_audit.csv
panel_full_comparison_trace_v9350.csv
p4_static_action_value_observability.csv
static_feature_trace_v9350.csv
p5_legal_precommit_probe.csv
probe_feature_trace_v9350.csv
probe_cost_trace_v9350.csv
p6_crossfitted_control_positive_controller.csv
controller_frontier_trace_v9350.csv
p7_action_primitive_density_reset_scout.csv
action_primitive_scout_trace_v9350.csv
p8_online_payload_apply_runtime.csv
online_payload_runtime_trace_v9350.csv
runtime_component_trace_v9350.csv
p9_system_integration.csv
system_controller_trace_v9350.csv
p10_dataset_diagnostics_no_tuning.csv
dataset_diagnostic_trace_v9350.csv
p11_leave_dataset_stratum_out.csv
leaveout_trace_v9350.csv
p12_diagnostic_paired_replay_scout.csv
diagnostic_replay_trace_v9350.csv
p13_official_paired_replay.csv
official_replay_trace_v9350.csv
p14_short_full_sampleeff_continual_robustness.csv
short_full_trace_v9350.csv

figures/
```

---

# 10. Failure taxonomy

```text
F1_contract_violation
F2_dataset_tuning_detected
F3_teacher_or_loss_modification_detected
F4_fake_or_proxy_violation
F5_payload_regression
F6_action_lifecycle_regression
F7_full_control_outcome_completion_fail
F8_materializer_throughput_regression
F9_outcome_quality_violation
F10_panel_representativeness_fail
F11_panel_raw_coverage_misclassified
F12_full_control_positive_oracle_absent
F13_control_positive_sparse_high_value
F14_Vctrl_not_positive
F15_beats_controls_fail
F16_horizon_robustness_fail
F17_static_feature_unobservable
F18_probe_feature_unobservable
F19_probe_runtime_too_expensive
F20_artifact_side_channel_feature
F21_no_dataset_agnostic_controller
F22_decision_precision_fail
F23_decision_coverage_fail
F24_decision_bad_event_fail
F25_decision_null_rate_fail
F26_Vctrl_lcb_fail
F27_support_balance_fail
F28_payload_apply_runtime_fail
F29_online_runtime_smoke_only
F30_payload_apply_error_fail
F31_no_event_preservation_fail
F32_system_integration_fail
F33_leave_dataset_out_fail
F34_leave_stratum_out_fail
F35_diagnostic_replay_leakage
F36_official_paired_replay_control_equivalent
F37_shuffle_control_pass
F38_action_primitive_density_insufficient
F39_action_primitive_value_low
F40_short_run_task_drop
F41_full_run_no_advantage
F42_sample_efficiency_fail
F43_continual_forgetting_fail
F44_strong_baseline_explains_gain
F45_external_not_ready
```

---

# 11. Route decision

```text
R1-BoundaryReanalyzed:
  v9.3.4 boundary and panel/full coverage ambiguity computed.

R2-FullControlOutcomeReady:
  full 51768 branch-horizon rows complete and quality-audited.

R3-PanelCoverageArtifact:
  v9.3.4 raw coverage fail was caused by 30% panel undercount.

R4-FullControlPositiveOraclePass:
  full universe has enough control-positive actions.

R5-ControlPositiveSparseHighValue:
  control-positive rows are high quality but coverage <0.03.

R6-ControlPositiveOracleAbsent:
  full universe lacks useful control-positive frontier.

R7-StaticObservabilityPass:
  legal static features identify control-positive actions.

R8-StaticObservabilityFail:
  static features insufficient.

R9-LegalProbeObservabilityPass:
  legal pre-commit action-response probe identifies control-positive actions.

R10-LegalProbeObservabilityFail:
  even legal probes cannot identify action value.

R11-ControlPositiveControllerPass:
  cross-fitted dataset-agnostic controller passes decision gates.

R12-ControlPositiveControllerFail:
  oracle exists but no legal controller selects it.

R13-ActionPrimitiveDensityScoutPass:
  new action primitive improves control-positive density.

R14-ActionPrimitiveDensityScoutFail:
  primitive reset did not improve density.

R15-PayloadApplyRuntimePass:
  official online runtime with payload apply passes.

R16-PayloadApplyRuntimeFail:
  runtime smoke pass but payload apply official path fails.

R17-SystemLegalControllerPass:
  decision + runtime + contracts pass.

R18-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R19-LeaveStratumOutPass:
  controller generalizes across held-out strata.

R20-PairedReplayPass:
  official paired replay beats controls.

R21-PairedReplayFail:
  system legal but functional is control-equivalent.

R22-ShortFullFunctionalPass:
  short/full runs show task-safe functional advantage.

R23-ExternalReady:
  full external-ready gates pass.
```

`route_decision.json` 必须记录：

```text
route
base_candidate
candidate_count
action_count
event_count
full_control_outcome_ready
full_control_rows_expected
full_control_rows_actual
rows_per_sec_total
panel_fraction
coverage_raw_v9340
coverage_scaled_uniform_v9340
coverage_full_v9350
control_positive_oracle_pass
control_positive_oracle_accepted_count
precision_control_positive
bad_event_rate
null_rate
V_ctrl_mean
V_ctrl_lcb
beats_adamwparallel_rate
beats_bestlr_rate
panel_representativeness_pass
static_observability_pass
best_static_feature_group
best_static_auc_CP
probe_observability_pass
best_probe_id
best_probe_auc_CP
probe_cost_q90
controller_id
decision_gate_pass
coverage_heldout
precision_primary_heldout
precision_control_positive_heldout
bad_event_heldout
null_rate_heldout
V_ctrl_lcb_heldout
payload_apply_runtime_pass
runtime_candidate_id
step_ratio_q90
memory_ratio
payload_apply_error_linf_max
system_legal_controller_pass
leave_dataset_out_pass
leave_stratum_out_pass
diagnostic_replay_used_for_controller
paired_replay_pass
short_run_pass
full_run_pass
sample_efficiency_pass
continual_pass
robustness_pass
strong_baseline_pass
external_ready
primary_blocker
next_required_implementation
success_v9350_strict_purekan_functional
success_v9350_full_functional
success_v9350_external_ready
```

---

# 12. 停止条件

## Minimum diagnostic success

```text
P0 reanalysis pass
P1 full control outcome completion measured
P2 full oracle measured
P3 panel representativeness measured
P4 static observability measured
P5 legal probe measured or explicitly skipped with reason
P8 payload-apply runtime measured
no fake/proxy/offload/teacher/loss violation
```

## Pivot to action primitive reset

如果：

```text
P2 full oracle absent
or full coverage < 0.01
or V_ctrl_lcb <= 0
```

则停止 AP0 controller search，进入 P7 action primitive density reset。

## Pivot to legal probe

如果：

```text
P2 oracle pass
but P4 static observability fail
```

则进入 P5 legal pre-commit probe，不继续扩展 payload norm / linf / load time 类 feature。

## Pivot to controller redesign

如果：

```text
P4/P5 observability pass
but P6 controller fail
```

则修 controller calibration / support / monotonicity，不修 action primitive。

## Pivot to runtime

如果：

```text
P6 controller pass
but P8 payload runtime fail
```

则停止 controller tuning，修 payload apply runtime。

## Open official paired replay

只有当：

```text
P9 system pass
and P11 leave-out pass
```

才打开 P13 official paired replay。

---

# 13. v9.3.5 最终判断标准

v9.3.5 成功不一定等于 full functional success。它至少要把项目推过以下一个清晰分叉：

## Case A：full oracle pass + legal probe/controller/runtime pass

可以声明：

```text
strict PureKAN functional route has a system-legal local control-positive controller candidate.
```

但仍需 official paired replay / short-full run。

## Case B：full oracle pass + observability fail

必须声明：

```text
control-positive actions exist, but current legal commit-time observables cannot identify them.
```

下一步应设计新的 legal action-response observables，而不是调 threshold。

## Case C：full oracle sparse high-value

必须声明：

```text
current action primitive can produce clean high-value actions, but density is insufficient for official coverage.
```

下一步应做 action primitive density reset。

## Case D：full oracle absent

必须声明：

```text
current action primitive does not produce enough control-positive actions under matched controls.
```

下一步停止 controller work，重建 action generator / functional update primitive。

## Case E：controller pass but payload runtime fail

必须声明：

```text
decision problem solved, but system envelope not solved.
```

下一步修 payload apply runtime，不继续调 controller。

---

# 14. 本轮一句话策略

$$
\boxed{
\text{先完成 full control-positive frontier 判定；若 frontier 存在，用 legal action-response probe 识别它；若 frontier 稀疏或缺失，重置 action primitive density；最后用 payload-apply runtime 做真正 system closure。}
}
$$

v9.3.5 不应该继续把 “coverage 0.0171” 当成定论，也不应该继续把 “best feature AUC 0.5274” 当成只需调参的问题。真正的下一步是把 full label universe 跑完，并把 controller 问题、action-density 问题、runtime 问题分开闭合。
