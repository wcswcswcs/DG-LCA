# DG-KAN v9.4.2 Source Outcome Materializer Closure / Value-Producing Source Triage / Accelerated Parallel Validation 完整实验计划

> 本计划基于 v9.4.1 `Value-Producing Source Generator / Legal Source Identifiability / Base-Acc Sentinel` 的真实执行结果制定。  
> v9.4.1 的 terminal route 是：
>
> ```text
> route = R4-GeneratedSourceImmediateDirectionFail
> base_candidate = LQ-t2-h256
> success_v9410_strict_purekan_functional = False
> success_v9410_full_functional = False
> success_v9410_external_ready = False
> ```
>
> 但本计划对这个 route 做一个更严格的重新解释：v9.4.1 的 P4/P5 branch-horizon row count 是 `expected = 7020, actual = 0`，所以当前不能说 generated source action 已经 value-negative，也不能说 h20 immediate direction 真实失败。更准确的 blocker 是：
>
> $$
> \boxed{\text{generated source outcome materializer did not produce evaluable branch-horizon rows.}}
> $$
>
> 因此 v9.4.2 不继续调 certificate threshold，不继续修 AP0 static selector，不继续按数据集调参。v9.4.2 的第一目标是让 generated source outcome rows 第一次完整存在；第二目标是在同一 panel 上分解 source AP0、AP0b-AP0f generator、certificate 三者到底谁在失败；第三目标是通过并行 preflight / smoke / official panel 设计加快实验。

---

# 0. 执行摘要

v9.4.1 有真实推进，但推进类型不是 functional success，也不是 controller success，而是 source-generation chain 的 implementation progress。

已经推进掉的 blocker：

```text
1. first-N biased source panel 已被 PANEL-S256 stratified panel 替代；
2. Base-Acc Sentinel 已真实训练并记录 LQ vs MatchedMLP；
3. AP0b-AP0f value-producing source generators 首次 materialize；
4. durable payload tensor 与 certificate tensor 已落盘；
5. payload/certificate hash missing = 0；
6. action apply replay 数值闭合，L∞ / relative error = 0.0；
7. commit-time legality audit 没有 dataset_name / outcome_at_commit / future_step。
```

没有推进掉的 blocker：

```text
1. generated source branch-horizon outcomes 没有落盘；
2. h20 immediate direction 没有真实评估；
3. h80/h240 horizon long-risk 没有真实评估；
4. certificate sufficiency 没有运行；
5. source controller 没有 selected；
6. selected runtime 没有运行；
7. paired replay、short/full、continual 全部 gate-blocked。
```

v9.4.2 的核心判断不是“继续慢慢审计”，而是把实验从单线 gate 改成并行闭环：

```text
Batch A: source outcome materializer preflight and row-write closure。
Batch B: same-panel source AP0 baseline and generated-source value comparison。
Batch C: certificate lift / damage matrix / horizon failure dissection。
Batch D: Base-Acc Sentinel 扩展和 runtime microbench，只做健康监控，不进入 controller。
Batch E: 若 P2/P3 出现 survivor，立刻启动 minimal controller 和 selected runtime。
```

v9.4.2 的最低有效推进目标：

```text
1. 让 P4/P5 actual branch-horizon rows 从 0 变成 expected count；
2. 对 AP0b-AP0f 生成的 260 actions 给出 h20/h80/h240 的 RealSource vs controls 结果；
3. 同 panel 评估 source AP0 baseline，分清 source bad、generator damage、certificate insufficient、horizon drift；
4. 明确下一轮是 source generator redesign、certificate redesign、controller selection，还是 runtime optimization；
5. 保持 Base-Acc Sentinel 只作为健康监控，不用于调 selector/controller。
```

---

# 1. 对 v9.4.1 的独立判断

## 1.1 有进展，但不是你真正想要的那种进展

v9.4.1 的进展是 infrastructure progress，不是 capability progress。它把 v9.4.0 的 `source_generator_materialized = 0` 推进成：

```text
source_generator_materialized = 1
primitive_materialized_count = 5
source_input_action_count = 52
generated_action_count_total = 260
payload_tensor_written = 1
certificate_tensor_written = 1
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
action_apply_error_relative_max = 0.0
action_apply_cosine_min = 0.9999999999999938
commit_time_available = 1
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
certificate_pass_action_count = 81
p3_materialization_pass = 1
```

这不是小事。AP0b-AP0f 终于不是 placeholder、schema、diagnostic，而是真实生成 payload/certificate 并能 apply。

但这轮没有给出 action value，因为最关键的 rows 没有出现：

```text
branch_horizon_row_count_expected = 7020
branch_horizon_row_count_actual = 0
source_outcome_materialized = 0
quality_audit_pass = 0
h20_immediate_direction_pass = 0
horizon_extension_pass = 0
```

因此 v9.4.1 不能被解释为：

```text
AP0b-AP0f generated source actions value-negative。
AP0b-AP0f immediate direction fail。
certificate primitive fail。
source controller fail。
```

它只能被解释为：

$$
\boxed{\text{AP0b-AP0f generated actions are real, but their outcomes are not materialized.}}
$$

## 1.2 “感觉慢”是合理的

你感觉慢，是因为从 v9.3.0 到 v9.4.1，很多轮都在把系统从“不能评估”推进到“可以评估”，但真正的 functional controller 仍然没出现。

这种慢有两类。

第一类是必要慢：

```text
不能把 primary oracle 写成 functional success；
不能把 diagnostic certificate 写成 official certificate；
不能把 payload generation 写成 value pass；
不能把 Base-Acc Sentinel 写成 full functional MLP comparison；
不能把 materializer actual rows = 0 写成 weak_CP = 0。
```

这些 gate 是保护科学性的。

第二类是不必要慢：

```text
P4/P5 actual rows = 0 这种问题应该通过 preflight 在几分钟内被截断；
不应该跑完整 v9.4.1 后才发现 row sink 为 0；
source generator、source outcome materializer、Base-Acc Sentinel、runtime microbench 可以并行；
P4 expected/actual mismatch 应该有 root-cause trace，而不是只写 route。
```

v9.4.2 必须修第二类慢。

## 1.3 有在数据集上训练吗？acc 如何？和 MLP 比如何？

v9.4.1 做了 Base-Acc Sentinel，是真实训练，不是 controller / functional 训练。它覆盖：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
sentinel_row_count = 27
same_seed_schedule = 1
same_budget = 1
hyperparams_fixed_before_run = 1
dataset_specific_tuning = 0
base_acc_used_for_controller = 0
```

结果：

```text
mean_test_acc_LQ = 0.6553819444444444
mean_test_acc_MLP = 0.5559895833333334
LQ_minus_MLP_mean_test_acc = 0.09939236111111105
LQ_catastrophic_fail = 0
```

我的判断：

```text
1. LQ-t2-h256 没有 catastrophic fail；
2. 在这个 fixed-config sentinel 下，LQ 比 MatchedMLP 高约 9.94 个百分点；
3. 但这不是 official full functional result；
4. 绝对 acc 只有 0.655 左右，说明这是健康哨兵 / 小预算固定配置，不是严肃打榜训练；
5. 不能拿它宣布 KAN 已经全面超过 MLP；
6. 也不能用它调 selector/controller，因为 base_acc_used_for_controller = 0。
```

换句话说，Base-Acc Sentinel 是“底座没有崩”的证据，不是“终极目标已经接近”的证据。

## 1.4 当前真正卡在哪里

当前主 blocker 是：

$$
\boxed{\text{generated source outcome materializer closure。}}
$$

不是：

```text
source panel 代表性；
source generator 是否存在；
payload/certificate hash；
action apply replay；
Base Acc catastrophic fail；
StableAccept score/rank；
数据集调参。
```

更具体地说，v9.4.1 已经有 260 个 generated source actions，但是没有任何可评估 branch-horizon outcome row。这个状态下，科学问题还不能进入 “value 是否正” 或 “certificate 是否有效”。必须先让以下对象存在：

$$
O(a,b,h)=\text{outcome of action } a \text{ under branch } b \text{ at horizon } h.
$$

如果 $O(a,b,h)$ 不存在，下面这些都没有定义：

$$
V_{ctrl}(a,h),\quad WeakCP(a,h),\quad LongRisk(a,h),\quad HorizonRobustCP(a).
$$

---

# 2. v9.4.2 总体目标

v9.4.2 的总体目标是：

$$
\boxed{\text{把 AP0b-AP0f generated source actions 的 branch-horizon outcomes materialize 出来，并判定 source generator 是否产生真实 value。}}
$$

这轮不做打榜，不按数据集调参，不把 acc sentinel 用于 controller。它围绕四个第一性问题设计实验：

```text
Q1. v9.4.1 actual rows = 0 是 materializer bug、join bug、gate bug、payload package bug，还是 checkpoint/horizon runner bug？
Q2. 若 outcome rows materialize，AP0b-AP0f 是否在 h20 产生 immediate positive direction？
Q3. 若 h20 有正向信号，h80/h240 是否保住，还是 long-risk 主导？
Q4. certificate 是否能在 commit time 区分 value-positive / long-risk source actions？
```

v9.4.2 的强目标：

```text
source_outcome_materialized = 1
branch_horizon_row_count_actual = branch_horizon_row_count_expected
h20_immediate_direction_pass = 1
horizon_extension_pass = 1
certificate_sufficiency_pass = 1
source_controller_pass = 1
selected_runtime_pass = 1
system_legal_controller_pass = 1
```

v9.4.2 的最低有效推进目标：

```text
1. P4/P5 actual rows 从 0 推到至少 smoke complete；
2. 对 260 generated source actions 产出完整 h20 branch outcomes；
3. 至少完成 h80/h240 horizon audit 的 stratified subset；
4. 用 same-panel source AP0 baseline 分清 source bad / generator damage / certificate fail；
5. 如果 generated source 仍失败，给出可执行的 AP0g-AP0k redesign direction；
6. 如果 generated source 成功，立刻进入 minimal controller + selected runtime。
```

---

# 3. 硬约束

v9.4.2 继续遵守：

```text
no teacher
no self-teacher
no distillation
no auxiliary loss
no loss modification
no label smoothing / focal loss / margin loss
no sampler / class weight tuning
no CPU offload
no fake rows
no proxy rows
no dataset-specific selector/controller branch
no validation/test leakage at commit time
no outcome_at_commit feature
no future_step feature
manual forward / manual backward / manual AdamW update preserved
functional update is update rule, not loss trick
```

Functional update 仍然是：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{AdamW}+\Delta\theta_{functional}.
$$

任务 loss 仍然是：

$$
L_{task}=CE(y,p_\theta(x)).
$$

允许做：

```text
per-dataset diagnostic;
leave-dataset-out diagnostic;
per-family / per-horizon failure diagnosis;
source outcome materializer debugging;
stratified panel comparison;
same-panel source AP0 baseline;
fixed-config Base-Acc Sentinel;
runtime microbench diagnostic;
parallel preflight and smoke tests.
```

但所有 diagnostic 都不能进入 controller threshold 或 dataset-specific tuning。

---

# 4. 核心假设

## H0：v9.4.1 的 P4/P5 失败首先是 materializer closure failure，不是 value failure

H0 认为 `branch_horizon_row_count_actual = 0` 不能解释为 weak_CP = 0，而是 materializer 没有成功写出可评估 rows。

H0 成立标准：

```text
preflight reproduces missing row path；
failure attribution identifies one or more root causes：
  payload join fail;
  action_id mismatch;
  primitive_id mismatch;
  checkpoint unavailable;
  branch runner exception;
  horizon runner exception;
  row sink/write fail;
  quality filter drops all rows;
  manifest points to wrong generated_action_table;
  seed/panel mismatch;
  branch list empty;
  horizon list empty.
```

H0 失败标准：

```text
materializer successfully runs and writes complete rows, but all generated source actions fail value gates.
```

只有 H0 失败后，才能讨论 AP0b-AP0f value failure。

## H1：stratified source panel 已足够代表 full AP0 universe，可以作为 source-generator input

H1 根据 v9.4.1 的 `PANEL-S256` 判断。v9.4.2 不再使用 first-N convenience slice。

H1 成立标准：

```text
panel_id = PANEL-S256 or stronger PANEL-S864
PSI_vs_full <= 0.05
KL_vs_full <= 0.03
max_family_gap <= 0.02
max_step_bucket_gap <= 0.02
max_score_bucket_gap <= 0.02
max_payload_bucket_gap <= 0.02
```

H1 失败标准：

```text
panel drift regresses，或者 source generator 实际输入不是 selected stratified panel。
```

## H2：AP0b-AP0f source generators 的 generation/apply chain 已闭合，但 value 仍未知

H2 成立标准：

```text
source_generator_materialized = 1
primitive_materialized_count = 5
generated_action_count_total > 0
payload_tensor_written = 1
certificate_tensor_written = 1
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max <= 1e-7
action_apply_cosine_min >= 0.999999
commit_time_available = 1
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
```

H2 失败标准：

```text
action apply replay or certificate/payload hash regresses。
```

## H3：AP0b-AP0f 中至少有一个 primitive 产生 h20 immediate positive source frontier

定义：

$$
V_{ctrl}(a,h)=V_{RealSource}(a,h)-\max_{b\in Controls}V_b(a,h).
$$

Weak control-positive row：

$$
WeakCP(a,h)=1 \iff V_{ctrl}(a,h)>0 \land BadEvent(a,h)=0 \land NullEvent(a,h)=0.
$$

H3 weak 成立标准：

```text
for at least one primitive on h20:
  branch_horizon_completion_rate = 1.0
  weak_CP_precision_h20 >= 0.30
  V_ctrl_lcb_h20 > 0
  bad_event_rate_h20 <= 0.10
  null_rate_h20 <= 0.20
  accepted_family_count >= 8 or support_balance_diagnostic_pass = 1
```

H3 strong 成立标准：

```text
for at least one primitive on h20:
  weak_CP_precision_h20 >= 0.50
  strong_CP_precision_h20 >= 0.20
  V_ctrl_lcb_h20 > 0.05
  bad_event_rate_h20 <= 0.05
  null_rate_h20 <= 0.15
```

H3 失败标准：

```text
source outcomes complete, but every primitive has V_ctrl_lcb_h20 <= 0 or weak_CP_precision_h20 < 0.20.
```

若 H3 失败，进入 immediate source generator redesign，不进入 horizon/controller。

## H4：如果 h20 有信号，h80/h240 必须证明不是 short-only / long-risk artifact

H4 成立标准：

```text
for selected primitive:
  weak_CP_precision_h80 >= 0.30
  weak_CP_precision_h240 >= 0.20
  long_risk_rate_h240 <= 0.15
  horizon_robust_action_coverage >= 0.01 diagnostic
```

H4 strong 成立标准：

```text
horizon_robust_action_coverage >= 0.03
long_risk_rate_h240 <= 0.10
V_ctrl_lcb_all_horizons > 0
```

H4 失败标准：

```text
h20 passes, but h80/h240 long-risk dominates，or h240 long_risk_rate > 0.30.
```

## H5：certificate 必须是 effect-valid，不只是 construction-valid

H5 认为 certificate 的价值不是字段完整，而是能提升 CP precision 并降低 long-risk。

Certificate lift：

$$
Lift_{weak}=\frac{P(WeakCP=1\mid CertPass=1)}{P(WeakCP=1\mid CertPass=0)+\epsilon}.
$$

Long-risk lift：

$$
Lift_{long}=\frac{P(LongRisk=1\mid CertPass=1)}{P(LongRisk=1\mid CertPass=0)+\epsilon}.
$$

H5 成立标准：

```text
Lift_weak >= 2.0
AUC_certificate_weak_CP >= 0.70
Lift_long <= 0.70
P(long_risk | cert_pass) <= 0.15
monotone_sign_pass = 1
calibration_to_heldout_drift <= 0.10
```

H5 失败标准：

```text
certificate only increases weak CP slightly but does not reduce long-risk，or certificate pass coverage collapses。
```

## H6：Base-Acc Sentinel 是健康哨兵，不是 official functional success

H6 成立标准：

```text
sentinel_complete = 1
same_seed_schedule = 1
same_budget = 1
base_acc_used_for_controller = 0
dataset_specific_tuning = 0
LQ_catastrophic_fail = 0
```

H6 失败标准：

```text
LQ catastrophic fail, or sentinel metrics used to choose source/controller threshold。
```

## H7：只有 source value + certificate + runtime 都有 survivor，才能打开 controller/system/downstream

H7 成立标准：

```text
h20_immediate_direction_pass = 1
horizon_extension_pass = 1
certificate_sufficiency_pass = 1
source_controller_pass = 1
selected_runtime_pass = 1
system_legal_controller_pass = 1
```

H7 失败标准：

```text
任意上游 gate 未过，却打开 paired replay / short/full / continual。
```

---

# 5. 数据合同

## 5.1 Generated source action contract

每个 generated source action 必须记录：

```text
generated_action_id
source_action_id
source_candidate_id
source_event_id
source_panel_id
primitive_id
primitive_version
payload_hash
certificate_hash
payload_tensor_path
certificate_tensor_path
action_apply_error_linf
action_apply_error_relative
action_apply_cosine
payload_norm
payload_linf
payload_role_entropy
certificate_pass
commit_time_available
uses_dataset_name
uses_outcome_at_commit
uses_future_step
seed
step
dataset
family_id
bucket_id
horizon_source
```

Pass：

```text
duplicate_generated_action_id_count = 0
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
payload_tensor_written = 1
certificate_tensor_written = 1
action_apply_error_linf_max <= 1e-7
action_apply_cosine_min >= 0.999999
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
```

## 5.2 Source outcome row contract

每个 outcome row 必须记录：

```text
outcome_row_id
generated_action_id
source_action_id
primitive_id
branch
horizon
dataset
seed
step
checkpoint_hash_before
checkpoint_hash_after
payload_hash
certificate_hash
branch_state_hash
horizon_state_hash
row_write_status
exception_type
exception_message
CE_delta
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
bad_event_label
null_event_label
safe_good_label
task_safe_label
useful_label
V_real
V_ctrl
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
beats_shuffled_payload
```

Required branches：

```text
RealSource
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledSourcePayload
CertificatePassNoPayload
SourceAP0Baseline
```

Required horizons：

```text
h20
h80
h240
```

Expected row count：

$$
N_{rows}=N_{actions}\times N_{branches}\times N_{horizons}.
$$

For v9.4.1 official smoke：

$$
N_{actions}=260,
$$

$$
N_{branches}\times N_{horizons}=\frac{7020}{260}=27.
$$

If $N_{horizons}=3$, then $N_{branches}=9$.

Pass：

```text
branch_horizon_row_count_actual = branch_horizon_row_count_expected
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1.0
missing_branch_count = 0
missing_horizon_count = 0
metric_nan_count = 0
metric_inf_count = 0
label_exclusivity_violation_count = 0
```

## 5.3 Same-panel source AP0 baseline contract

为了判断 generated source 是否破坏 source AP0，必须对同一批 source actions 做 AP0 baseline。

记录：

```text
source_action_id
source_panel_id
branch
horizon
AP0_V_ctrl
AP0_weak_CP
AP0_strong_CP
AP0_long_risk
AP0_bad_event
AP0_null_event
AP0_horizon_robust
```

Damage 定义：

$$
Damage(a,h)=V_{generated}(a,h)-V_{AP0}(a,h).
$$

Positive source lost rate：

$$
LostRate=\frac{\#\{a,h: WeakCP_{AP0}(a,h)=1 \land WeakCP_{generated}(a,h)=0\}}{\#\{a,h: WeakCP_{AP0}(a,h)=1\}+\epsilon}.
$$

## 5.4 Base-Acc Sentinel contract

Sentinel 继续记录，但不进入 controller。

```text
model_id
dataset
seed
train_acc
val_acc
test_acc
train_loss
val_loss
test_loss
ECE
NLL
CEp99
margin_p10
step_time_q90
memory_ratio
same_seed_schedule
same_budget
hyperparams_fixed_before_run
dataset_specific_tuning
base_acc_used_for_controller
```

Sentinel pass：

```text
sentinel_complete = 1
base_acc_used_for_controller = 0
dataset_specific_tuning = 0
LQ_catastrophic_fail = 0
```

---

# 6. 实验阶段

---

## P0：v9.4.1 boundary independent reanalysis

### 目标

重新计算 v9.4.1 的真实边界，不被 `R4-GeneratedSourceImmediateDirectionFail` 名称牵着走。P0 要明确区分：

```text
materializer rows absent
vs
materialized rows value-negative
```

### 输入

```text
route_decision.json
p1_stratified_source_panel_builder.csv
p2_base_acc_sentinel_lq_vs_mlp.csv
p3_real_value_producing_source_generator.csv
p4_h20_immediate_direction_smoke.csv
p5_horizon_extension_longrisk_audit.csv
source_payload_trace_v9410.csv
source_certificate_trace_v9410.csv
action_apply_replay_trace_v9410.csv
branch_horizon_completion_trace_v9410.csv
```

### 必须记录

```text
route_v9410
source_route_v9400
stratified_panel_pass
official_panel_id
official_panel_PSI_vs_full
official_panel_KL_vs_full
base_acc_sentinel_complete
mean_test_acc_LQ
mean_test_acc_MLP
LQ_minus_MLP_mean_test_acc
base_acc_used_for_controller
source_generator_materialized
primitive_materialized_count
source_input_action_count
generated_action_count_total
payload_tensor_written
certificate_tensor_written
action_apply_error_linf_max
certificate_pass_action_count
branch_horizon_row_count_expected
branch_horizon_row_count_actual
source_outcome_materialized
h20_immediate_direction_pass
horizon_extension_pass
certificate_sufficiency_pass
source_controller_pass
selected_runtime_pass
system_legal_controller_pass
```

### 判断标准

P0 pass：

```text
v9.4.1 boundary reproduced；
actual rows = 0 被标记为 materializer failure，而不是 value failure；
Base-Acc Sentinel 被标记为 health sentinel，不是 official full functional result；
source generator pass 和 source outcome pass 分离。
```

### 可视化

```text
p0_v9410_gate_ladder.svg
p0_generation_vs_outcome_boundary.svg
p0_base_acc_lq_vs_mlp_sentinel.svg
p0_expected_actual_outcome_rows.svg
```

---

## P1：source outcome materializer root-cause isolation

### 目标

在不跑大实验的情况下，定位为什么 expected `7020` rows 变成 actual `0`。P1 是 v9.4.2 最重要的工程阶段；如果 P1 不过，后续全部不跑。

### 设计

P1 分成三层 preflight。

#### P1a：single-action row-write preflight

选择：

```text
1 action
1 primitive
1 branch = RealSource
1 horizon = h20
```

预期：

```text
expected_rows = 1
actual_rows = 1
```

如果这一行都写不出来，说明不是 value 问题，是 materializer path 问题。

#### P1b：single-action branch fanout preflight

选择：

```text
1 action
1 primitive
all branches
1 horizon = h20
```

预期：

$$
expected\_rows=N_{branches}.
$$

#### P1c：mini matrix preflight

选择：

```text
2 actions per primitive
5 primitives
all branches
h20
```

若 $N_{branches}=9$：

$$
expected\_rows=2\times5\times9=90.
$$

### 必须记录

```text
preflight_id
generated_action_id
source_action_id
primitive_id
payload_tensor_path_exists
certificate_tensor_path_exists
payload_hash_match
certificate_hash_match
checkpoint_hash_exists
branch_runner_initialized
horizon_runner_initialized
row_sink_initialized
branch
horizon
expected_row_count
actual_row_count
rows_written_before_quality_filter
rows_dropped_by_quality_filter
exception_type
exception_message
join_failure_key
manifest_path_used
payload_shard_id
```

Root-cause categories：

```text
RC1_generated_action_table_not_loaded
RC2_payload_path_missing
RC3_certificate_path_missing
RC4_action_id_join_mismatch
RC5_source_action_id_join_mismatch
RC6_primitive_id_filter_drops_all
RC7_checkpoint_missing
RC8_branch_runner_exception
RC9_horizon_runner_exception
RC10_row_sink_write_exception
RC11_quality_filter_drops_all_rows
RC12_seed_panel_mismatch
RC13_branch_list_empty
RC14_horizon_list_empty
RC15_wrong_out_dir_or_manifest
```

### 判断标准

P1 pass：

```text
P1a actual_rows = expected_rows
P1b actual_rows = expected_rows
P1c actual_rows = expected_rows
rows_dropped_by_quality_filter = 0
unresolved_exception_count = 0
root_cause_table_complete = 1
```

P1 fail：

```text
任何 preflight actual_rows = 0 或 unresolved root cause 存在。
```

P1 fail 后 route：

```text
R1-SourceOutcomeMaterializerPathFail
```

不得继续跑 controller / certificate。

### 可视化

```text
p1_preflight_expected_actual_rows.svg
p1_materializer_failure_sankey.svg
p1_row_drop_by_stage.svg
p1_join_key_failure_heatmap.svg
```

---

## P2：source outcome materializer scale-up to v9410 official smoke

### 目标

把 v9.4.1 中已生成的 260 个 AP0b-AP0f source actions 的 branch-horizon outcomes 真正 materialize 出来。

### 输入

```text
PANEL-S256 source panel
AP0b-AP0f generated action table
source payload shards
source certificate tensor
checkpoint before event
branches = 9
horizons = h20,h80,h240
```

### 规模

v9.4.1 official smoke：

```text
generated_action_count_total = 260
expected_branch_horizon_rows = 7020
```

公式：

$$
7020=260\times9\times3.
$$

### 必须记录

```text
materializer_id
generated_action_count_input
branch_count
horizon_count
branch_horizon_row_count_expected
branch_horizon_row_count_actual
branch_completion_rate
horizon_completion_rate
secondary_delta_completion_rate
rows_per_sec_total
wallclock_sec
row_count_failed
row_count_retried
unresolved_failed_rows
quality_audit_pass
missing_branch_count
missing_horizon_count
missing_secondary_delta_count
metric_nan_count
metric_inf_count
label_exclusivity_violation_count
```

### 判断标准

P2 pass：

```text
branch_horizon_row_count_actual = 7020
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1.0
quality_audit_pass = 1
unresolved_failed_rows = 0
```

P2 weak pass：

```text
h20 rows complete, but h80/h240 incomplete。
```

P2 weak pass 可进入 P3 h20 immediate analysis，但不能进入 horizon/controller official。

### 可视化

```text
p2_branch_horizon_completion_heatmap.svg
p2_rows_per_sec_by_branch_horizon.svg
p2_missing_rows_by_primitive.svg
p2_quality_audit_dashboard.svg
```

---

## P3：same-panel source AP0 baseline

### 目标

避免把 generated source failure 全归咎于 AP0b-AP0f transform。必须在同一批 source actions 上同时评估 source AP0 baseline。

### 假设

如果 source AP0 同 panel 本身就 weak，问题在 source selection / source panel value；如果 source AP0 好而 generated source 坏，问题在 generator transform；如果 source AP0 和 generated 都好但 certificate 选不出来，问题在 certificate/controller。

### 必须记录

```text
source_action_id
source_panel_id
primitive_id = AP0-source-baseline
branch
horizon
branch_horizon_row_count_expected
branch_horizon_row_count_actual
AP0_weak_CP_precision_h20
AP0_strong_CP_precision_h20
AP0_weak_CP_precision_h80
AP0_weak_CP_precision_h240
AP0_long_risk_rate_h240
AP0_V_ctrl_lcb_h20
AP0_V_ctrl_lcb_all
AP0_horizon_robust_action_count
AP0_horizon_robust_coverage
```

### 判断标准

P3 source baseline useful：

```text
AP0_weak_CP_precision_h20 >= 0.30
AP0_V_ctrl_lcb_h20 > 0
AP0_long_risk_rate_h240 <= 0.20 diagnostic
```

P3 source baseline bad：

```text
AP0_weak_CP_precision_h20 < 0.20
or AP0_V_ctrl_lcb_h20 <= 0
```

若 P3 source baseline bad，route：

```text
R2-SourcePanelValuePoorDespiteStratification
```

此时不应继续 AP0b-AP0f transform tuning，而应回到 source selector / source generator input。

### 可视化

```text
p3_source_ap0_baseline_value_curve.svg
p3_source_ap0_cp_by_horizon.svg
p3_source_ap0_longrisk_distribution.svg
```

---

## P4：generated source immediate direction analysis

### 目标

在 P2 rows 完整后，判断 AP0b-AP0f 是否真的产生 h20 immediate positive direction。

### 必须记录

Per primitive：

```text
primitive_id
generated_action_count
certificate_pass_action_count
h20_row_count
weak_CP_precision_h20
strong_CP_precision_h20
bad_event_rate_h20
null_rate_h20
V_ctrl_mean_h20
V_ctrl_lcb_h20
CEp99_delta_mean_h20
margin_p10_delta_mean_h20
ECE_delta_mean_h20
NLL_delta_mean_h20
beats_adamwparallel_rate_h20
beats_bestlr_rate_h20
beats_noop_rate_h20
beats_random_rate_h20
support_balance_pass_h20
```

### 判断标准

P4 weak pass：

```text
at least one primitive:
  weak_CP_precision_h20 >= 0.30
  V_ctrl_lcb_h20 > 0
  bad_event_rate_h20 <= 0.10
  null_rate_h20 <= 0.20
```

P4 strong pass：

```text
at least one primitive:
  weak_CP_precision_h20 >= 0.50
  strong_CP_precision_h20 >= 0.20
  V_ctrl_lcb_h20 > 0.05
  bad_event_rate_h20 <= 0.05
  null_rate_h20 <= 0.15
```

P4 fail：

```text
all primitives weak_CP_precision_h20 < 0.20 or V_ctrl_lcb_h20 <= 0.
```

P4 fail route：

```text
R3-GeneratedSourceImmediateValueFail
```

### 可视化

```text
p4_h20_cp_precision_by_primitive.svg
p4_h20_vctrl_distribution.svg
p4_h20_real_vs_controls_delta.svg
p4_h20_bad_null_safe_phase.svg
```

---

## P5：source-to-generated damage matrix

### 目标

量化 AP0b-AP0f 是否保留 source AP0 的 value，还是破坏了 source direction。

### 必须记录

```text
source_action_id
generated_action_id
primitive_id
horizon
V_AP0
V_generated
Damage
WeakCP_AP0
WeakCP_generated
StrongCP_AP0
StrongCP_generated
LongRisk_AP0
LongRisk_generated
source_positive_lost_after_generation
source_negative_fixed_after_generation
payload_cosine_source_generated
payload_norm_ratio
certificate_pass
```

Damage：

$$
Damage(a,h)=V_{generated}(a,h)-V_{AP0}(a,h).
$$

### 判断标准

Generator value-preserving pass：

```text
Damage_median >= -0.02
source_positive_lost_after_generation_rate <= 0.30
source_negative_fixed_after_generation_rate >= 0.10 diagnostic
```

Generator damage fail：

```text
source_positive_lost_after_generation_rate >= 0.70
or Damage_median < -0.10
```

### 可视化

```text
p5_source_generated_damage_hist.svg
p5_source_vs_generated_vctrl_scatter.svg
p5_positive_lost_by_primitive.svg
p5_payload_geometry_vs_damage.svg
```

---

## P6：horizon extension and long-risk audit

### 目标

若 P4 h20 有 survivor，检查 h80/h240 是否仍安全。不能只追 immediate gain。

### 必须记录

```text
primitive_id
horizon
weak_CP_precision
strong_CP_precision
bad_event_rate
null_rate
long_risk_rate
V_ctrl_mean
V_ctrl_lcb
CEp99_delta_mean
margin_p10_delta_mean
ECE_delta_mean
NLL_delta_mean
horizon_robust_action_count
horizon_robust_action_coverage
short_only_action_count
long_risk_action_count
```

Long risk：

$$
LongRisk(a)=1 \iff BadEvent(a,h240)=1 \lor V_{ctrl}(a,h240)<-\tau_{long}.
$$

Horizon robust：

$$
HorizonRobustCP(a)=1 \iff WeakCP(a,h20)=1 \land WeakCP(a,h80)=1 \land WeakCP(a,h240)=1.
$$

### 判断标准

P6 weak pass：

```text
selected primitive:
  weak_CP_precision_h80 >= 0.30
  weak_CP_precision_h240 >= 0.20
  long_risk_rate_h240 <= 0.15
```

P6 strong pass：

```text
horizon_robust_action_coverage >= 0.03
V_ctrl_lcb_all_horizons > 0
long_risk_rate_h240 <= 0.10
```

P6 fail：

```text
h20 pass but h240 long_risk_rate > 0.30
or horizon_robust_action_count = 0 with large h20 short-only count.
```

### 可视化

```text
p6_horizon_value_curve_by_primitive.svg
p6_long_risk_by_primitive.svg
p6_horizon_robust_sankey.svg
p6_short_only_vs_long_risk.svg
```

---

## P7：effect-valid certificate sufficiency

### 目标

判断 commit-time certificate 是否真的能选择 value-positive、低 long-risk 的 generated source actions。

### 必须记录

```text
certificate_id
primitive_id
certificate_pass
certificate_score
certificate_component_values
weak_CP
strong_CP
horizon_robust_CP
long_risk
bad_event
null_event
V_ctrl
AUC_certificate_weak_CP
AUC_certificate_strong_CP
AUC_certificate_longrisk
P_weak_CP_given_cert_pass
P_weak_CP_given_cert_fail
P_longrisk_given_cert_pass
P_longrisk_given_cert_fail
Lift_weak
Lift_longrisk
calibration_to_heldout_drift
monotone_sign_pass
```

### 判断标准

P7 pass：

```text
Lift_weak >= 2.0
AUC_certificate_weak_CP >= 0.70
Lift_longrisk <= 0.70
P_longrisk_given_cert_pass <= 0.15
monotone_sign_pass = 1
certificate_pass_action_count >= 32
```

P7 weak pass：

```text
Lift_weak >= 1.5
AUC_certificate_weak_CP >= 0.65
but long-risk control not passed。
```

P7 fail：

```text
AUC_certificate_weak_CP < 0.60
or Lift_longrisk >= 0.90
or certificate pass coverage collapses。
```

### 可视化

```text
p7_certificate_calibration_curve.svg
p7_certificate_lift_bar.svg
p7_certificate_component_ablation.svg
p7_cert_pass_vs_fail_value_distribution.svg
```

---

## P8：primitive triage and redesign decision

### 目标

在 P3-P7 后，给出明确路线，不再进入盲目 threshold tuning。

### Triage matrix

```text
Case A: source AP0 bad, generated bad
  route = R4-SourceInputValuePoor
  next = source selector / source generation input redesign

Case B: source AP0 good, generated bad
  route = R5-GeneratorTransformDamage
  next = value-preserving transform redesign

Case C: generated h20 good, h240 bad
  route = R6-HorizonLongRiskDominates
  next = horizon-aware primitive redesign

Case D: generated good, certificate bad
  route = R7-CertificateNotEffectValid
  next = certificate component redesign

Case E: generated good, certificate good, runtime blocked
  route = R8-SourceRuntimeBlocked
  next = selected runtime optimization

Case F: generated good, certificate good, runtime pass
  route = R9-ReadyForSourceControllerSystemGate
```

### 必须记录

```text
triage_case
source_AP0_status
generated_source_status
generator_damage_status
horizon_status
certificate_status
runtime_status
next_required_implementation
stop_threshold_tuning_flag
```

### 判断标准

P8 pass：

```text
triage_case assigned exactly one primary route;
next_required_implementation is specific;
no threshold-only next step unless generated+certificate have survivor。
```

### 可视化

```text
p8_failure_attribution_ladder.svg
p8_triage_decision_tree.svg
```

---

## P9：Base-Acc Sentinel extension, not controller input

### 目标

回答“有训练吗、acc 如何、和 MLP 比如何”，但不让 acc 进入 source/controller 调参。v9.4.2 扩展 sentinel，是为了监控底座健康和防止 hidden catastrophic fail。

### 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2,3,4
models:
  LQ-t2-h256
  MatchedMLP
  StrongLRGridMLP diagnostic
  QuadraticFeatureMLP diagnostic
budgets:
  sentinel-short fixed
  optional sentinel-medium fixed
```

### 必须记录

```text
model_id
dataset
seed
budget_id
train_acc
val_acc
test_acc
train_loss
val_loss
test_loss
ECE
NLL
CEp99
margin_p10
ValLossAUC_step
ValLossAUC_time
time_to_60pct_acc
time_to_70pct_acc
step_time_q90
memory_ratio
same_seed_schedule
same_budget
hyperparams_fixed_before_run
dataset_specific_tuning
base_acc_used_for_controller
```

### 判断标准

P9 sentinel pass：

```text
sentinel_complete = 1
base_acc_used_for_controller = 0
dataset_specific_tuning = 0
LQ_catastrophic_fail = 0
```

Catastrophic fail：

```text
mean_test_acc_LQ < mean_test_acc_MLP - 0.05
or LQ test acc < 0.20 on any dataset/seed diagnostic
or step_time_q90 ratio > 1.50 in sentinel base path
```

### 可视化

```text
p9_base_acc_by_dataset_seed.svg
p9_lq_vs_mlp_acc_gap.svg
p9_val_loss_auc_step.svg
p9_time_to_target.svg
p9_base_step_time_memory.svg
```

---

## P10：minimal source certificate controller

### 目标

只有 P4/P6/P7 有 survivor 才运行。用 certificate 和 minimal legal features 选择 generated source actions，不能使用 dataset branch 或 outcome at commit。

### Controller form

$$
Accept(a)=1
\iff
LCB(V_{cert}(a))>0
\land UCB(Bad_{cert}(a))\le\tau_b
\land UCB(Null_{cert}(a))\le\tau_n
\land UCB(LongRisk_{cert}(a))\le\tau_l
\land LCB(Support(a))\ge\tau_s
\land C(a)\le C_{max}.
$$

### Candidate controllers

```text
SC0-CertPassOnly negative control
SC1-CertScoreTopK
SC2-CertValueRiskNullGate
SC3-HorizonGuardedCertGate
SC4-MinimalMonotoneCertController
SC5-TwoStageCertPlusExactCheapProbe
```

### 必须记录

```text
controller_id
primitive_id
feature_set
thresholds
calibration_split_id
heldout_split_id
dataset_name_used
uses_outcome_at_commit
accepted_count_cal
accepted_count_heldout
weak_CP_precision_cal
weak_CP_precision_heldout
strong_CP_precision_heldout
horizon_robust_precision_heldout
bad_event_rate_heldout
null_rate_heldout
long_risk_rate_heldout
V_ctrl_lcb_heldout
coverage_heldout
accepted_family_count
max_family_share
```

### 判断标准

P10 pass：

```text
dataset_name_used = 0
uses_outcome_at_commit = 0
coverage_heldout >= 0.03 diagnostic or accepted_count_heldout >= 64 for source-stage smoke
weak_CP_precision_heldout >= 0.50
long_risk_rate_heldout <= 0.10
V_ctrl_lcb_heldout > 0
support_balance_pass = 1
```

P10 fail：

```text
coverage collapse;
weak_CP_precision_heldout < 0.30;
long_risk_rate_heldout > 0.20;
V_ctrl_lcb_heldout <= 0.
```

### 可视化

```text
p10_source_controller_frontier.svg
p10_calibration_to_heldout_drift.svg
p10_controller_support_balance.svg
p10_cert_controller_overlap_by_primitive.svg
```

---

## P11：selected source online runtime

### 目标

只有 P10 有 selected controller 才 official。runtime 不能提前转正，也不能用 no-payload smoke。

### 必须记录

```text
runtime_candidate_id
controller_id
primitive_id
selected_controller_used
selected_payload_apply_used
step_count
active_step_count
accepted_action_count
feature_compute_time_ms_q90
certificate_score_time_ms_q90
accept_decision_time_ms_q90
payload_lookup_time_ms_q90
payload_apply_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
payload_apply_error_linf_max
payload_apply_cosine_min
audit_outside_timed_path
```

### 判断标准

P11 pass：

```text
selected_controller_used = 1
selected_payload_apply_used = 1
payload_apply_error_linf_max <= 1e-7
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
audit_outside_timed_path = 1
```

P11 diagnostic pass：

```text
step_ratio_q90 <= 2.00
but > 1.50, with component waterfall identifying dominant cost.
```

### 可视化

```text
p11_selected_runtime_waterfall.svg
p11_step_ratio_distribution.svg
p11_payload_apply_cost_by_primitive.svg
p11_runtime_vs_value_pareto.svg
```

---

## P12：system integration gate

### 目标

组合 source value、certificate、controller、runtime，决定是否能进入 LDO/paired replay/short/full。

### 必须记录

```text
system_candidate_id
primitive_id
controller_id
runtime_candidate_id
source_outcome_materialized
h20_immediate_direction_pass
horizon_extension_pass
certificate_sufficiency_pass
source_controller_pass
selected_runtime_pass
coverage_heldout
weak_CP_precision_heldout
strong_CP_precision_heldout
horizon_robust_precision_heldout
bad_event_rate_heldout
null_rate_heldout
long_risk_rate_heldout
V_ctrl_lcb_heldout
step_ratio_q90
memory_ratio
dataset_name_used
uses_validation_or_test
uses_outcome_at_commit
uses_future_step
diagnostic_promoted_to_official
fake_data_used
proxy_row_used
cpu_offload_used
official_eligible
system_legal_controller_pass
```

### 判断标准

P12 pass：

```text
official_eligible = 1
system_legal_controller_pass = 1
source_outcome_materialized = 1
h20_immediate_direction_pass = 1
horizon_extension_pass = 1
certificate_sufficiency_pass = 1
source_controller_pass = 1
selected_runtime_pass = 1
dataset_name_used = 0
uses_outcome_at_commit = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

### 可视化

```text
p12_system_gate_dashboard.svg
p12_quality_runtime_frontier.svg
p12_official_eligibility_ladder.svg
```

---

## P13：leave-out and diagnostic paired replay boundary

### 目标

只有 P12 pass 后打开 official LDO/LSO。若 P12 不过，只允许 diagnostic scout，并且不能影响 controller threshold。

### 设置

Leave-dataset-out：

```text
train/calibrate on two datasets, evaluate third。
```

Leave-stratum-out：

```text
train/calibrate all but one family/horizon/score stratum, evaluate held-out stratum。
```

Paired branches：

```text
RealSourceFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledSourcePayload
ShuffledCertificate
ShuffledControllerScore
ShuffledHorizonGuard
```

### 必须记录

```text
split_type
heldout_entity
controller_id
primitive_id
branch
horizon
weak_CP_precision
strong_CP_precision
horizon_robust_precision
V_ctrl_lcb
bad_event_rate
null_rate
long_risk_rate
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
shuffle_control_fail_rate
diagnostic_downstream_used_for_controller
```

### 判断标准

LDO/LSO pass：

```text
macro weak_CP_precision >= 0.50
long_risk_rate <= 0.10
V_ctrl_lcb > 0
beats_adamwparallel_rate >= 0.60
beats_bestlr_rate >= 0.60
shuffle controls fail
```

Isolation pass：

```text
diagnostic_downstream_used_for_controller = 0
```

### 可视化

```text
p13_leave_dataset_matrix.svg
p13_leave_stratum_matrix.svg
p13_paired_replay_branch_winrate.svg
p13_shuffle_control_matrix.svg
```

---

## P14：short/full training and MLP comparison boundary

### 目标

只有 P12/P13 pass 后打开真正的 short/full functional training。Base-Acc Sentinel 不能替代这里。

### 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0..4 for short, 0..9 for full if resources allow
models:
  LQ-t2-h256 base
  LQ-t2-h256 + selected source functional controller
  MatchedMLP
  StrongLRGridMLP
  QuadraticFeatureMLP
branches:
  AdamWOnly
  RealFunctional
  ShuffledFunctionalPayload
  NoOp
```

### 必须记录

```text
model_id
branch
dataset
seed
run_budget
train_acc
val_acc
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
time_to_target_acc
steps_to_target_acc
step_time_q90
memory_ratio
candidate_rate
accept_rate
bad_event_rate
null_rate
long_risk_rate
functional_event_count
```

### 判断标准

Short-run pass：

```text
RealFunctional test_acc >= AdamWOnly test_acc - 0.005
RealFunctional ValLossAUC_step improves over AdamWOnly
bad_event_rate remains <= 0.05
```

MLP comparison diagnostic：

```text
RealFunctional LQ beats MatchedMLP on macro test acc or sample efficiency;
StrongLRGridMLP recorded as strong baseline;
QuadraticFeatureMLP recorded as structure-control baseline。
```

Full success candidate：

```text
RealFunctional beats matched controls;
LDO/LSO pass;
paired replay pass;
step_ratio_q90 <= 1.50;
memory_ratio <= 1.05;
no dataset-specific tuning;
strong baseline not trivially better。
```

### 可视化

```text
p14_acc_by_dataset_seed_model.svg
p14_lq_functional_vs_mlp_gap.svg
p14_val_loss_auc_step_time.svg
p14_time_to_target.svg
p14_calibration_robustness_dashboard.svg
p14_system_cost_vs_accuracy_pareto.svg
```

---

# 7. 并行执行设计

为了解决“太慢”的问题，v9.4.2 必须并行，但并行不等于跳 gate。

## Batch A：materializer closure fast lane

```text
A1: P1a single-action row-write preflight
A2: P1b single-action all-branch preflight
A3: P1c mini matrix preflight
A4: P2 official 260-action source outcome materialization
```

Gate：

```text
A1/A2/A3 不过，A4 不跑。
```

## Batch B：Base-Acc Sentinel and runtime health lane

```text
B1: expand Base-Acc Sentinel to seeds 0..4
B2: record LQ vs MatchedMLP vs StrongLRGridMLP diagnostic
B3: selected-source payload apply microbench placeholder, not official
```

Gate：

```text
B results cannot affect source/controller thresholds。
```

## Batch C：source AP0 baseline lane

```text
C1: same-panel AP0 baseline h20
C2: same-panel AP0 baseline h80/h240
C3: AP0 vs generated damage matrix once generated rows exist
```

Gate：

```text
C1 can run once P1 preflight closes for AP0 baseline row path。
```

## Batch D：certificate and controller lane

```text
D1: certificate lift analysis
D2: certificate ablation
D3: minimal source certificate controller
```

Gate：

```text
D only runs after P2 rows complete。
```

## Batch E：system/downstream lane

```text
E1: selected runtime
E2: system gate
E3: leave-out / paired replay
E4: short/full training
```

Gate：

```text
E only runs after source value + certificate + controller have survivor。
```

---

# 8. 必须落盘的 artifacts

```text
p0_v9410_boundary_reanalysis.csv
p1_source_outcome_materializer_preflight.csv
source_outcome_materializer_failure_trace_v9420.csv
source_outcome_join_key_trace_v9420.csv
source_outcome_row_sink_trace_v9420.csv
p2_source_outcome_materializer_scaleup.csv
source_outcome_trace_v9420.csv
branch_horizon_completion_trace_v9420.csv
p3_same_panel_source_ap0_baseline.csv
source_ap0_outcome_trace_v9420.csv
p4_generated_source_immediate_direction.csv
p5_source_to_generated_damage_matrix.csv
p6_horizon_extension_longrisk_audit.csv
p7_certificate_sufficiency_lift_audit.csv
certificate_component_ablation_v9420.csv
p8_primitive_triage_route.csv
p9_base_acc_sentinel_extended.csv
base_acc_training_trace_v9420.csv
p10_minimal_source_certificate_controller.csv
source_controller_frontier_trace_v9420.csv
p11_selected_source_online_runtime.csv
runtime_component_trace_v9420.csv
p12_system_integration_gate_v9420.csv
p13_leaveout_paired_replay_boundary_v9420.csv
p14_short_full_training_mlp_comparison_boundary_v9420.csv
contract_audit_v9420.csv
provenance_audit_v9420.csv
failure_table_v9420.csv
route_decision.json
run_manifest.json
```

---

# 9. Route decision

v9.4.2 route 必须是下面之一，不允许模糊写法。

```text
R1-SourceOutcomeMaterializerPathFail
  P1 preflight 失败，actual rows 仍为 0 或 root cause 未闭合。

R2-SourcePanelValuePoorDespiteStratification
  AP0 same-panel baseline 本身 value-poor。

R3-GeneratedSourceImmediateValueFail
  AP0 baseline 可用，但 AP0b-AP0f h20 immediate value 失败。

R4-GeneratorTransformDamage
  source AP0 value-positive，但 generated transform 丢失 source-positive rows。

R5-HorizonLongRiskDominates
  h20 过，但 h80/h240 long-risk 失败。

R6-CertificateNotEffectValid
  generated source 有 value，但 certificate 不能选择低风险 good actions。

R7-SourceControllerSupportCollapse
  certificate 有信号，但 heldout/controller coverage collapse。

R8-SelectedRuntimeFail
  source/controller 过，但 selected payload runtime 超 envelope。

R9-SystemLegalSourceControllerPass
  source value、certificate、controller、runtime 全部过线。

R10-DownstreamCausalFail
  system pass 后，LDO/paired replay/short-run 失败。

R11-StrictPureKANFunctionalCandidate
  system + LDO/LSO + paired replay + short-run 均达到候选成功标准。
```

---

# 10. 停止条件

为了避免继续慢速绕圈，v9.4.2 必须有明确停止条件。

```text
Stop 1:
  P1a single-action row-write fail。
  立即停止所有 value/certificate/controller，修 materializer。

Stop 2:
  P2 official smoke rows incomplete。
  不允许把 weak_CP_precision = 0.0 当作 value failure。

Stop 3:
  P3 source AP0 baseline bad。
  停止 AP0b-AP0f transform tuning，回到 source selector/source input。

Stop 4:
  P4 h20 generated source fail。
  停止 horizon/controller，做 immediate source generator redesign。

Stop 5:
  P6 h240 long-risk fail。
  停止 controller，做 horizon-aware source primitive。

Stop 6:
  P7 certificate fail。
  不调 controller threshold，重做 effect-valid certificate。

Stop 7:
  P11 selected runtime fail。
  不打开 paired replay，先做 payload apply/runtime。
```

---

# 11. 最终成功判断

v9.4.2 不要求最终 full success，但如果要宣称 strict PureKAN functional candidate，至少必须满足：

```text
source_outcome_materialized = 1
h20_immediate_direction_pass = 1
horizon_extension_pass = 1
certificate_sufficiency_pass = 1
source_controller_pass = 1
selected_runtime_pass = 1
system_legal_controller_pass = 1
leave_dataset_out_pass = 1
leave_stratum_out_pass = 1
paired_replay_pass = 1
short_run_pass = 1
no_fake = 1
no_proxy = 1
no_dataset_tuning = 1
```

如果只完成 P1/P2，把 rows 从 0 推到 7020，也是有效进展，因为它会把当前不可判定状态推进为可判定状态。但 v9.4.2 不能再次只输出：

```text
branch_horizon_row_count_expected > 0
branch_horizon_row_count_actual = 0
```

这会说明实验流程本身需要重构，而不是继续设计新的 primitive。

---

# 12. 一句话结论

v9.4.2 的本质不是“继续找 feature”，而是：

$$
\boxed{\text{先让 generated source action 的 outcome universe 存在，然后再判断 value、horizon、certificate、controller。}}
$$

在 outcome rows 仍为 0 的情况下，任何关于 AP0b-AP0f value 的结论都是过早的。v9.4.2 必须把 `actual rows = 0` 变成完整 branch-horizon outcome table，并用同 panel AP0 baseline 做根因分解。只有这样，项目才能从“生成器实现了”推进到“functional source 是否真的有用”。
