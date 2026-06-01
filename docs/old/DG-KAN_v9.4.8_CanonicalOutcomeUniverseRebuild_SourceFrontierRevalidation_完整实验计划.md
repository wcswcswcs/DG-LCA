# DG-KAN v9.4.8 Canonical Outcome Universe Rebuild / Source Frontier Revalidation / Parallel Closure 完整实验计划

> 本计划基于 v9.4.7 `No-Transform Replay Semantics Closure 与 Robust Source Revalidation` 的真实结果制定。  
> v9.4.8 的重点不是继续微调 AP0r/AP0t/AP0u、CERT8 或 source selector threshold，而是先重建 canonical outcome universe。  
> v9.4.7 已经证明：no-transform source/clone replay semantics 可以闭合；但旧 v9.3.5 outcome table 的 ORC-D-K16 oracle survivor 不能在 canonical repaired runner 下复现。因此，所有依赖旧 v9.3.5 outcome table 的 generator / certificate / controller 结论都必须先进入 quarantine，再在 canonical runner 下重算。

---

# 0. 执行摘要

v9.4.7 的真实进展是实验物理层面的：

```text
source/clone replay key pass = 1
single-action deterministic preflight pass = 1
canonical runner semantics pass = 1
side-by-side no-transform replay pass = 1
no_transform_equivalence_pass = 1
negative_control_divergence_present = 1
```

这说明 v9.4.6 的 `I18-outcome-materializer-runner-drift` 已经被 canonical branch-name-invariant runner 闭合。过去“同 payload / 同 branch start state / no-transform clone 仍产生不同 horizon state”的问题不应再作为当前 blocker。

但 v9.4.7 的 P6 直接推翻了旧 v9.3.5 outcome table 上的 ORC-D-K16 survivor：

```text
old_h20_weak_CP = 0.8125
new_h20_weak_CP = 0.0
old_h20_V_ctrl_lcb = 0.13772944106165844
new_h20_V_ctrl_lcb = -1.2978123872692517
old_h240_long_risk = 0.0
new_h240_long_risk = 0.75
old_Y_robust_count = 13
new_Y_robust_count = 0
old_new_label_match_rate = 0.0
old_new_V_ctrl_abs_diff_max = 3.7012117721606046
old_table_quarantine_required = 1
source_oracle_revalidation_pass = 0
```

因此，本轮 v9.4.8 的第一性目标是：

$$
\text{在 canonical repaired runner 下重建 outcome universe，再判断 AP0 source frontier、generator、certificate、controller 是否真实存在。}
$$

v9.4.8 不应再问：

```text
AP0r trust-region 半径要不要调？
CERT8 threshold 要不要调？
state_NLL_proxy topK 要不要调？
AP0d/AP0t/AP0u 谁稍微好一点？
```

v9.4.8 应该先问：

```text
1. 旧 v9.3.5 outcome table 的漂移范围到底多大？
2. canonical repaired runner 下，AP0 full control-positive frontier 是否仍存在？
3. canonical repaired runner 下，是否仍存在 h20 value-positive 且 h240 safe 的 robust source frontier？
4. 如果 AP0 source frontier 不存在，之前的 generator/certificate 失败是否只是旧表幻觉？
5. 如果 AP0 source frontier 仍存在，legal selector / generator / certificate 是否能在同一个 canonical runner 下识别和保留它？
```

一句话：

$$
\boxed{
\text{v9.4.8 是 canonical outcome truth reset，不是 generator 小修补。}
}
$$

---

# 1. 对 v9.4.7 的独立判断

## 1.1 有进展，但不是能力进展

v9.4.7 有实质进展。它把 v9.4.6 的 no-transform replay inconsistency 修成了可判别的 deterministic replay contract：

```text
paired_action_count = 16
paired_branch_horizon_row_count_actual = 288
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
branch_state_hash_match_rate = 1.0
horizon_state_hash_match_rate = 1.0
optimizer_state_hash_match_rate = 1.0
batch_sequence_hash_match_rate = 1.0
metric_abs_diff_max = 0.0
metric_abs_diff_p99 = 0.0
label_match_rate = 1.0
weak_CP_match_rate = 1.0
strong_CP_match_rate = 1.0
long_risk_match_rate = 1.0
Y_robust_match_rate = 1.0
V_ctrl_abs_diff_max = 0.0
first_divergence_count = 0
```

同时，负控不是无脑相等，而是真的能分离：

```text
payload_shuffled_clone diverged = 1
rng_perturbed_clone diverged = 1
optimizer_state_perturbed_clone diverged = 1
```

这说明 canonical runner 的 no-transform gate 有判别力，不是把所有分支都强行对齐。

但这不是 functional 能力成功。P6 直接显示旧 ORC-D-K16 在 canonical repaired runner 下完全不复现。旧表中的 robust survivor 在新 runner 下变成：

```text
new_h20_weak_CP = 0.0
new_h20_V_ctrl_lcb = -1.2978123872692517
new_h240_long_risk = 0.75
new_Y_robust_count = 0
```

这意味着之前围绕 ORC-D-K16 做出的“source survivor exists / generator destructive / certificate fail”一部分结论必须重新验证。

## 1.2 为什么你会觉得慢

这个感觉是合理的。最近几轮的节奏是：

```text
v9.4.4: 旧表上找到 ORC-D-K16 horizon-robust survivor。
v9.4.5: no-transform 也不能复现 oracle survivor，因此 generator destructive 不能继续下结论。
v9.4.6: source identity、payload、old aggregation 都闭合，但 side-by-side outcome replay 不等价。
v9.4.7: canonical runner 修通 no-transform replay，但旧 v9.3.5 outcome table 的 survivor 被新 rollout 推翻。
```

科学上这是在收缩错误边界，但工程上确实“一轮只清一个 blocker”。v9.4.8 必须改成并行验证：

```text
canonical outcome full-table rebuild；
old-new drift taxonomy；
AP0 source frontier revalidation；
legal observability revalidation；
Base-Acc Sentinel continuation；
runtime microbench / materializer throughput monitoring。
```

不能再等一个大 runner 全跑完后才发现下一个 gate 是表版本不一致。

## 1.3 当前真正卡在哪里

当前不是卡在：

```text
source/clone identity；
no-transform payload equivalence；
branch name / clone id execution path；
single-action deterministic replay；
negative control 判别力；
Base LQ catastrophic failure。
```

当前真正卡在：

```text
old v9.3.5 outcome table 不再能作为 canonical truth；
canonical repaired runner 下 AP0 source frontier 是否仍存在未知；
generator / certificate / controller 都依赖 source frontier，因此必须 gate-block。
```

更严格地说，现在的 primary blocker 是：

$$
\boxed{
old\_v9350\_outcome\_table\_not\_reproduced\_by\_canonical\_runner
}
$$

这个 blocker 比“generator 破坏 source”更上游。只有 canonical table 重建后，才能重新判断 generator 是否真的 destructive。

## 1.4 数据集训练与 acc

v9.4.7 有数据集训练，但只是 Base-Acc Sentinel，不是 official functional training，也不是 system controller short/full run。

本轮 Base-Acc Sentinel：

```text
sentinel_row_count = 120
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
model_count = 4
mean_test_acc_LQ = 0.6537760416666667
mean_test_acc_MLP = 0.562890625
mean_test_acc_QuadraticFeatureMLP = 0.40234375
mean_test_acc_AdamWStrongLRGridMLP = 0.628515625
LQ_minus_MLP_mean_test_acc = 0.09088541666666672
LQ_minus_QuadraticFeatureMLP_mean_test_acc = 0.2514322916666667
LQ_minus_AdamWStrongLRGridMLP_mean_test_acc = 0.025260416666666674
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
base_acc_sentinel_pass = 1
```

解释：

```text
LQ-t2-h256 比 MatchedMLP 平均高约 9.09 个百分点；
比 AdamWStrongLRGridMLP 平均高约 2.53 个百分点；
没有 catastrophic fail；
但这个 sentinel 没有用于 selector / generator / controller；
因此不能把它写成 functional DG-KAN 超过 MLP。
```

它说明 base 仍然健康，但 functional update / controller 路线还没有被证明。

---

# 2. v9.4.8 总体目标

v9.4.8 的总体目标是：

$$
\boxed{
\text{用 canonical repaired runner 重建 AP0 outcome universe，并重新判定 source frontier / observability / generator / certificate 的真实状态。}
}
$$

v9.4.8 的强目标不是直接 full functional success，而是建立新的 truth base：

```text
1. old v9.3.5 outcome table 被明确隔离，不能再进入 official generator/certificate/controller；
2. canonical outcome table v9480 完整落盘；
3. AP0 full control-positive oracle 在 canonical runner 下重新计算；
4. AP0 source frontier 在 canonical runner 下重新计算；
5. legal selector / generator / certificate 只在 canonical labels 下重新评估；
6. Base-Acc Sentinel 继续隔离运行；
7. 若 canonical AP0 frontier 不存在，pivot 到 source primitive redesign；
8. 若 canonical AP0 frontier 存在但 legal selector 看不见，pivot 到 certificate-producing / objective-solving generator；
9. 若 canonical frontier + certificate controller 过线，再打开 selected runtime / LDO / paired replay / short-full。
```

最低有效推进：

```text
canonical_no_transform_equivalence_pass = 1
canonical_full_control_outcome_ready = 1 或 official-minimum canonical panel ready = 1
old_table_quarantine_enforced = 1
old_new_drift_taxonomy_complete = 1
canonical_AP0_frontier_route_decided = 1
```

强推进：

```text
canonical_full_control_outcome_ready = 1
canonical_control_positive_oracle_pass known
canonical_source_frontier_pass known
canonical_legal_observability_recomputed = 1
canonical_generator_preservation_revalidated = 1 if source frontier exists
```

---

# 3. 硬约束

v9.4.8 继续遵守：

```text
no teacher
no self-teacher
no distillation
no auxiliary loss
no loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
official selector/controller 不使用 dataset_name 分支
不使用 validation/test metric at commit time
不使用 future outcome feature
不使用 outcome-at-commit
不使用 source-measured gap / formula proxy officialization
不把 diagnostic oracle 写成 official controller
不把 Base-Acc Sentinel 写成 functional success
```

Functional update 仍然只能作为 update rule：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{AdamW-equivalent}
+
\Delta\theta_{functional}.
$$

任务 loss 保持标准 CE：

$$
L_{task}=CE(y,p_\theta(x)).
$$

v9.4.8 新增硬约束：

```text
1. 旧 v9.3.5 outcome table 只能 diagnostic，不能 official。
2. 所有 official generator/certificate/controller 必须读取 canonical_outcome_table_v9480。
3. 每个 outcome table 必须带 outcome_table_version、runner_semantics_version、branch_semantics_version、label_config_hash。
4. 若 canonical table 未闭合，不得打开 generator preservation / certificate / controller / selected runtime。
5. no-transform equivalence sentinel 必须在每个 canonical materializer shard 中插入。
6. old-new drift 不能用来调 dataset-specific controller，只能用于诊断 runner/table mismatch。
```

---

# 4. 核心假设

## H0：v9.4.7 已经修复 no-transform replay semantics

H0 认为 canonical branch-name-invariant runner 已经可以保证 source 与 no-transform clone 在相同 payload、相同 start state、相同 optimizer/RNG/batch sequence 下完全等价。

成立标准：

```text
source_clone_replay_key_pass = 1
single_action_preflight_pass = 1
canonical_runner_semantics_pass = 1
side_by_side_no_transform_replay_pass = 1
no_transform_equivalence_pass = 1
negative_control_divergence_present = 1
```

v9.4.8 需要把 H0 作为 preflight sentinel，而不是每次重新争论。

## H1：旧 v9.3.5 outcome table 不能继续 official 使用

H1 认为旧表不只是 ORC-D-K16 局部异常，而可能在更大范围内存在 runner semantics drift。因此所有依赖旧表的结论必须重新 canonicalize。

成立标准：

```text
old_new_label_match_rate on ORC-D-K16 = 0.0
old_new_V_ctrl_abs_diff_max > 0
old_table_quarantine_required = 1
```

v9.4.8 要进一步测量：

```text
old_new_label_match_rate_full_or_panel
old_new_V_ctrl_corr
old_new_CP_rank_overlap
old_new_longrisk_agreement
old_new_horizon_state_hash_consistency
```

## H2：canonical AP0 full universe 可能仍有 frontier，但必须重新测

H2 不假设旧 v9.3.5 的 full weak CP frontier 仍成立。v9.4.8 要重新回答：

$$
\exists A \subset \mathcal{A}_{AP0}^{canonical}
$$

使得：

$$
Coverage(A)\ge 0.03,
$$

$$
LCB(V_{ctrl}(A))>0,
$$

$$
Bad(A)\le \tau_b,
$$

$$
LongRisk_{h240}(A)\le \tau_h.
$$

若 H2 fail，则 AP0 oracle-good 结论必须撤回，下一步不是 legal selector，而是 source primitive redesign。

## H3：canonical source frontier 比 weak CP frontier 更重要

v9.4.x 已经多次显示 h20 weak CP 与 h240 long-risk 不等价。v9.4.8 不能只追 weak CP。

Robust source objective：

$$
Y_{robust}(a)=1
\iff
WeakCP_{h20}(a)=1
\land
LCB(V_{ctrl,h20}(a))>0
\land
LongRisk_{h240}(a)\le \tau_h
\land
Support(a)\ge s_{min}.
$$

成立标准：

```text
存在 K >= 16 的 source panel：
  h20 weak CP >= 0.60
  h20 V_ctrl LCB > 0
  h240 long-risk <= 0.10
  support_balance_pass = 1
```

## H4：如果 canonical AP0 source frontier 存在，generator preservation 必须在同一 runner 下重测

过去 “generator destructive” 结论混合了旧 table 和新 runner。v9.4.8 只允许在 canonical source outcomes 与 canonical generated outcomes 上计算 damage。

Generator preservation 定义：

$$
Damage(a,g,h)=V_{ctrl}(g(a),h)-V_{ctrl}(a,h).
$$

Pass：

```text
no-transform preservation pass = 1
source_positive_lost_after_generation_rate <= 0.30
Damage_median >= -0.05
h20 generated weak CP >= 0.50
h20 generated V_ctrl LCB > 0
h240 generated long-risk <= 0.10
```

## H5：certificate 必须是 effect-valid，不是 construction-valid

Certificate 不能只证明 payload/hash/schema 合法，还要能预测 canonical outcome。

Effect-valid pass：

```text
AUC_Yrobust >= 0.75
AUC_longrisk <= 0.25 if score is safety-positive, or inverse AUC >= 0.75
P(Yrobust | cert_pass) >= 0.50
P(longrisk | cert_pass) <= 0.10
monotone_sign_pass = 1
leave-family AUC drop <= 0.10
leave-horizon AUC drop <= 0.10
```

## H6：Base-Acc Sentinel 只是健康检查，不是 functional proof

Base-Acc Sentinel 继续运行，但必须隔离：

```text
base_acc_used_for_selector = 0
base_acc_used_for_generator = 0
base_acc_used_for_certificate = 0
base_acc_used_for_controller = 0
```

它只回答：

```text
LQ base 是否 catastrophic fail？
LQ base 是否在 fixed config 下弱于 MLP 到不可继续？
强 MLP baseline 是否让 base gap 消失？
```

---

# 5. 数据合同

## 5.1 outcome table version contract

每个 outcome row 必须有：

```text
outcome_row_id
outcome_table_version
runner_semantics_version
branch_semantics_version
materializer_id
label_config_hash
metric_config_hash
horizon_config_hash
branch_config_hash
batch_sequence_hash
optimizer_state_hash
rng_state_hash
state_before_hash
state_after_horizon_hash
payload_hash
action_id
source_action_id
clone_action_id_if_any
primitive_id
branch_id
horizon
seed
dataset
step
```

Official table 必须满足：

```text
outcome_table_version = canonical_v9480
runner_semantics_version = canonical_branch_name_invariant_v9470_or_later
old_table_quarantine_flag = 0
missing_hash_count = 0
duplicate_outcome_row_id_count = 0
```

## 5.2 old table quarantine contract

所有 official readers 必须执行：

```text
if outcome_table_version in {v9350, v9340, measured_v9350_reference_recompute}:
    official_eligible = 0
    diagnostic_only = 1
    require_canonical_revalidation = 1
```

必须落盘：

```text
old_table_read_attempt_count
official_old_table_read_blocked_count
diagnostic_old_table_read_count
canonical_table_read_count
quarantine_violation_count
```

Pass：

```text
quarantine_violation_count = 0
official_old_table_read_blocked_count = old_table_read_attempt_count in official path
```

## 5.3 canonical control outcome contract

Branches：

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
```

Horizons：

```text
h20
h80
h240
```

For each action:

```text
expected_rows_per_action = 6 * 3 = 18
```

For full AP0 universe:

```text
action_count = 2876
expected_rows = 2876 * 18 = 51768
```

Pass：

```text
row_count_actual = row_count_expected
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1.0
label_exclusivity_violation_count = 0
metric_nan_count = 0
metric_inf_count = 0
duplicate_row_id_count = 0
```

## 5.4 canonical label definitions

Control-positive value：

$$
V_{ctrl}(a,h)=V_{RealFunctional}(a,h)-\max_{b\in Controls}V_b(a,h).
$$

Weak CP：

$$
WeakCP(a,h)=1 \iff V_{ctrl}(a,h)>0 \land BadEvent(a,h)=0 \land NullEvent(a,h)=0.
$$

Strong CP：

$$
StrongCP(a,h)=1 \iff V_{ctrl}(a,h)>\delta_s \land BadEvent(a,h)=0 \land NullEvent(a,h)=0.
$$

Long risk：

$$
LongRisk(a)=1 \iff BadEvent(a,h240)=1 \lor V_{ctrl}(a,h240)<-\delta_r.
$$

Horizon robust source：

$$
Y_{robust}(a)=1
\iff
WeakCP(a,h20)=1
\land
V_{ctrl}(a,h20)>0
\land
LongRisk(a)=0.
$$

---

# 6. 并行执行总览

v9.4.8 必须并行，不再串行等待一个大 runner 暴露下一个问题。

## 并行批次 A：canonical replay preflight

目标：保证 v9.4.7 replay semantics 没回退。

运行：

```text
1-action preflight
3-action preflight
16-action ORC-D-K16 no-transform replay
64-action stratified no-transform replay
negative controls
```

## 并行批次 B：canonical AP0 outcome materializer

目标：重建 canonical AP0 outcome universe。

运行：

```text
B1: ORC-D-K16 direct rollout
B2: 864-action official-minimum panel
B3: full 2876-action universe
B4: worker/shard throughput and retry manifest
```

## 并行批次 C：old-new drift taxonomy

目标：量化旧表与新表差异，不用它调 controller。

运行：

```text
old vs new on ORC-D-K16
old vs new on stratified panel
old vs new on full if B3 complete
```

## 并行批次 D：Base-Acc Sentinel continuation

目标：继续监控 LQ base 与 MLP / StrongLRGridMLP / QuadraticFeatureMLP。

运行：

```text
MNIST / Fashion-MNIST / KMNIST
seeds 0..9
fixed config
no controller usage
```

## 并行批次 E：runtime/materializer throughput monitoring

目标：保证 canonical materializer 成本可控；不作为 official online runtime。

运行：

```text
rows/sec by shard
branch runtime q90
horizon checkpoint reuse hit rate
payload cache hit rate
retry count
```

---

# 7. 实验阶段

---

## P0：v9.4.7 boundary reproduction and quarantine activation

### 目标

复现 v9.4.7 boundary，并启用旧 table quarantine。P0 不做新模型能力判断，只确保下一步不会继续误用旧 v9.3.5 outcome table。

### 必须记录

```text
source_route_v9470
no_transform_equivalence_pass_v9470
source_oracle_revalidation_pass_v9470
old_table_quarantine_required_v9470
old_h20_weak_CP
new_h20_weak_CP
old_h20_V_ctrl_lcb
new_h20_V_ctrl_lcb
old_h240_long_risk
new_h240_long_risk
old_new_label_match_rate
system_legal_controller_pass_v9470
```

### Pass 标准

```text
v9470 boundary reproduced = 1
old_table_quarantine_required = 1
quarantine_enforced_in_all_official_readers = 1
```

### 可视化

```text
p0_v9470_boundary_ladder.svg
p0_old_new_orc_d_k16_metric_flip.svg
p0_quarantine_reader_audit.svg
```

---

## P1：canonical replay preflight and sentinel grid

### 目标

把 v9.4.7 的 canonical replay 语义扩展成 v9.4.8 的常驻 sentinel。每个 materializer shard 都必须插入 no-transform sentinel，防止未来再次出现 hidden drift。

### 假设

H0 成立：canonical branch-name-invariant runner 可以保证 no-transform equivalence。

### 设置

```text
preflight actions:
  1-action deterministic
  3-action smoke
  16-action ORC-D-K16
  64-action stratified panel
branches:
  RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random
horizons:
  20, 80, 240
negative controls:
  payload_shuffled
  rng_perturbed
  optimizer_state_perturbed
```

### 必须记录

```text
source_action_id
clone_action_id
payload_hash_match
state_before_hash_match
optimizer_state_hash_match
rng_state_hash_match
batch_sequence_hash_match
branch_config_hash_match
horizon_config_hash_match
label_config_hash_match
metric_abs_diff_max
metric_abs_diff_p99
label_match_rate
weak_CP_match_rate
strong_CP_match_rate
longrisk_match_rate
Y_robust_match_rate
V_ctrl_abs_diff_max
negative_control_diverged
first_divergence_field
first_divergence_step
```

### Pass 标准

```text
payload_hash_match_rate = 1.0
state_before_hash_match_rate = 1.0
optimizer_state_hash_match_rate = 1.0
rng_state_hash_match_rate = 1.0
batch_sequence_hash_match_rate = 1.0
branch_state_hash_match_rate = 1.0
horizon_state_hash_match_rate = 1.0
metric_abs_diff_max = 0.0
label_match_rate = 1.0
negative_control_divergence_present = 1
```

### 可视化

```text
p1_no_transform_equivalence_matrix.svg
p1_stepwise_hash_match_timeline.svg
p1_negative_control_divergence_bar.svg
p1_first_divergence_heatmap.svg
```

---

## P2：canonical AP0 full control outcome materializer v2

### 目标

用 canonical repaired runner 重新 materialize AP0 full control outcome universe。P2 是 v9.4.8 的核心工程阶段。

### 假设

旧 v9.3.5 table 不能 official；canonical table v9480 是新的 truth base。

### 输入

```text
frozen AP0 action universe: 2876 actions
payload package: durable AP0 payload shards
runner: canonical_branch_name_invariant_v9470_or_later
branches: 6
horizons: 3
expected rows: 51768
```

### 输出 artifact

```text
p2_canonical_ap0_full_control_outcome_materializer.csv
canonical_full_control_outcome_table_v9480.csv
canonical_branch_horizon_completion_trace_v9480.csv
canonical_materializer_worker_trace_v9480.csv
canonical_materializer_retry_manifest_v9480.csv
canonical_outcome_quality_audit_v9480.csv
```

### 必须记录

```text
materializer_id
action_count_expected
action_count_completed
row_count_expected
row_count_actual
row_count_failed
row_count_retried
row_count_unresolved
rows_per_sec_total
wallclock_sec
worker_count
shard_size_actions
branch_completion_rate
horizon_completion_rate
secondary_delta_completion_rate
missing_branch_count
missing_horizon_count
missing_secondary_delta_count
metric_nan_count
metric_inf_count
label_exclusivity_violation_count
duplicate_outcome_row_id_count
no_transform_sentinel_pass_by_shard
```

### Pass 标准

Official-minimum pass：

```text
action_count_completed >= 864
row_count_actual >= 864 * 18
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
quality_audit_pass = 1
no_transform_sentinel_pass_by_shard = 1
```

Full pass：

```text
action_count_completed = 2876
row_count_actual = 51768
row_count_unresolved = 0
missing_branch_count = 0
missing_horizon_count = 0
missing_secondary_delta_count = 0
quality_audit_pass = 1
```

Throughput target：

```text
rows_per_sec_total >= 20
```

### 可视化

```text
p2_canonical_materializer_completion_curve.svg
p2_rows_per_sec_by_shard.svg
p2_branch_horizon_completion_heatmap.svg
p2_quality_audit_zero_violation_table.svg
p2_no_transform_sentinel_by_shard.svg
```

---

## P3：old-vs-canonical outcome drift taxonomy

### 目标

量化旧 v9.3.5 table 与 canonical v9.4.8 table 的差异，找出旧表为什么产生 ORC-D-K16 survivor 幻觉或语义差异。P3 只做诊断，不允许进入 official controller。

### 比较范围

```text
ORC-D-K16
PANEL-ORC64
PANEL-ORC128
PANEL-S256
PANEL-LGL64-AdamWConflictLow
Full AP0 universe if P2 full pass
```

### 必须记录

```text
action_id
branch
horizon
old_metric_value
new_metric_value
metric_abs_diff
old_label_WeakCP
new_label_WeakCP
old_label_StrongCP
new_label_StrongCP
old_label_LongRisk
new_label_LongRisk
old_Y_robust
new_Y_robust
old_V_ctrl
new_V_ctrl
old_new_label_match
old_new_rank_delta
branch_config_hash_match
horizon_config_hash_match
label_config_hash_match
batch_sequence_hash_match
optimizer_state_hash_match
runner_semantics_version_old
runner_semantics_version_new
failure_class
```

### Drift classes

```text
D0-no-drift
D1-label-definition-drift
D2-branch-semantics-drift
D3-horizon-rollout-drift
D4-batch-sequence-drift
D5-optimizer-state-drift
D6-control-branch-winner-drift
D7-metric-aggregation-drift
D8-old-runner-materializer-bug
D9-unresolved
```

### Pass 标准

```text
old_new_drift_taxonomy_complete = 1
attribution_fraction >= 0.90
old_table_official_quarantine_remains = 1
```

### 可视化

```text
p3_old_new_label_match_by_panel.svg
p3_old_new_Vctrl_scatter.svg
p3_old_new_rank_overlap_curve.svg
p3_drift_class_pie.svg
p3_control_branch_winner_flip_matrix.svg
```

---

## P4：canonical AP0 control-positive oracle revalidation

### 目标

重新判断 AP0 full universe 在 canonical runner 下是否仍有 control-positive frontier。P4 不使用 legal selector；它是 upper-bound diagnosis。

### Oracle families

```text
OR0-CanonicalStableAcceptReference
OR1-CanonicalFullWeakControlPositiveOracle
OR2-CanonicalFullStrongControlPositiveOracle
OR3-CanonicalHorizonRobustOracle
OR4-CanonicalValueRiskParetoOracle
OR5-CanonicalSupportBalancedOracle
OR6-CanonicalLowLongRiskOracle
```

### 必须记录

```text
oracle_id
outcome_table_version
action_count
row_count
accepted_count
coverage
coverage_lcb
precision_control_positive
bad_event_rate
null_rate
V_ctrl_mean
V_ctrl_lcb
V_ctrl_p10
h20_weak_CP_rate
h80_weak_CP_rate
h240_weak_CP_rate
strong_CP_rate
horizon_robust_action_count
horizon_robust_coverage
long_risk_action_count
long_risk_rate
support_balance_pass
accepted_family_count
max_family_share
accepted_horizon_count
```

### Pass 标准

Weak control-positive oracle pass：

```text
coverage >= 0.03
V_ctrl_lcb > 0
bad_event_rate = 0.0 under oracle definition
null_rate = 0.0 under oracle definition
support_balance_pass = 1
```

Horizon-robust oracle pass：

```text
horizon_robust_coverage >= 0.01 diagnostic weak
horizon_robust_coverage >= 0.03 strong
h240_long_risk_rate <= 0.10
V_ctrl_lcb_h20 > 0
```

### Route after P4

```text
If weak CP oracle fails:
  route = R20-CanonicalAP0ControlPositiveFrontierAbsent
  next = source primitive redesign / direct objective-solved generator

If weak CP oracle passes but horizon-robust fails:
  route = R21-CanonicalAP0ShortHorizonOnly
  next = horizon-aware source generator / long-risk certificate

If horizon-robust oracle passes:
  route = R22-CanonicalAP0FrontierExists
  next = legal selector / generator preservation / certificate
```

### 可视化

```text
p4_canonical_oracle_coverage_frontier.svg
p4_canonical_CP_by_horizon.svg
p4_canonical_longrisk_vs_value.svg
p4_canonical_support_balance.svg
p4_old_vs_new_oracle_summary.svg
```

---

## P5：canonical robust source frontier search

### 目标

比 P4 更精确地判断是否存在可作为 generator source 的 robust source panel。Source frontier 必须同时满足 h20 value-positive 与 h240 safe。

### Candidate panels

```text
SRC-ORC-D-K16-recomputed
SRC-ORC-D-K32
SRC-ORC-D-K64
SRC-ORC-ValueRisk-K16/K32/K64
SRC-ORC-LowLongRisk-K16/K32/K64
SRC-REP-S256
SRC-LGL-topK-feature-panels
```

### 必须记录

```text
panel_id
selector_type
K
action_count
h20_weak_CP
h20_strong_CP
h20_V_ctrl_lcb
h80_weak_CP
h80_V_ctrl_lcb
h240_weak_CP
h240_V_ctrl_lcb
h240_long_risk
Y_robust_count
Y_robust_rate
support_balance_pass
family_count
max_family_share
old_panel_overlap
```

### Pass 标准

```text
K >= 16
h20_weak_CP >= 0.60
h20_V_ctrl_lcb > 0
h240_long_risk <= 0.10
Y_robust_rate >= 0.50 for K16 diagnostic, or >=0.30 for K64
support_balance_pass = 1
```

### 可视化

```text
p5_source_frontier_K_curve.svg
p5_h20_value_vs_h240_longrisk.svg
p5_panel_family_balance.svg
p5_source_survivor_overlap_old_new.svg
```

---

## P6：canonical legal observability revalidation

### 目标

在 canonical labels 下重新评估 legal selector 是否能看见 robust source。旧表下的 legal opacity 可能成立，也可能是旧表语义造成的错觉；必须重算。

### Feature groups

```text
G1-state-tail: StateNLL, StateCEp99, StateMarginP10
G2-payload: PayloadNorm, PayloadLinf, PayloadRoleEntropy
G3-gradient: GradientAlignment, AdamWConflictLow, FunctionalAdamWCosine
G4-support: FamilySupportCount, HorizonSupport, CalibrationSupportLCB
G5-cost: PayloadApplyEstimate, CandidateDensity
G6-minimal-effect: LinearizedCEDelta, LinearizedMarginDelta, TailResponseProbe
```

### 必须记录

```text
feature_id
feature_group
commit_time_available
uses_dataset_name
uses_outcome_at_commit
uses_future_step
feature_cost_ms_q90
AUC_WeakCP_h20
AUC_Yrobust
AUC_LongRisk_h240
PR_lift_Yrobust
TopK64_Yrobust_precision
TopK64_h20_V_ctrl_lcb
TopK64_h240_longrisk
leave_dataset_auc_drop
leave_family_auc_drop
leave_horizon_auc_drop
```

### Pass 标准

Weak observability pass：

```text
AUC_Yrobust >= 0.65
TopK64_Yrobust_precision >= 0.25
feature_cost_ms_q90 <= 0.20
no legality violation
```

Strong observability pass：

```text
AUC_Yrobust >= 0.75
AUC_LongRisk_h240 inverse >= 0.75
TopK64_Yrobust_precision >= 0.50
leave_dataset_auc_drop <= 0.07
leave_family_auc_drop <= 0.10
```

### 可视化

```text
p6_feature_auc_bar.svg
p6_feature_cost_vs_auc_pareto.svg
p6_topK_precision_curve.svg
p6_leaveout_auc_drop.svg
p6_legal_feature_vs_oracle_rank.svg
```

---

## P7：canonical generator preservation revalidation

### Gate

P7 只在 P5 source frontier pass 后运行。若 P5 fail，P7 not_run。

### 目标

在同一个 canonical runner 下，重新评估 AP0b-AP0f / AP0r-AP0v / AP0l-AP0q 是否真的破坏 source value。

### Required sentinel

每个 generator panel 必须包含：

```text
AP0w-NoTransformReplaySource
```

且 AP0w 必须：

```text
payload equivalence pass = 1
outcome equivalence pass = 1
```

否则整个 P7 invalid。

### Generator families

```text
AP0b-AP0f historical source generators
AP0r-AP0v robust objective generators
AP0l-AP0q direct source generators
AP0w no-transform replay sentinel
AP1-AP8 certificate-producing variants if applicable
```

### 必须记录

```text
source_panel_id
generator_id
generated_action_count
payload_hash_missing_count
certificate_hash_missing_count
action_apply_error_linf_max
no_transform_sentinel_pass
branch_horizon_rows_expected
branch_horizon_rows_actual
h20_weak_CP_generated
h20_V_ctrl_lcb_generated
h240_longrisk_generated
Y_robust_generated_count
Damage_mean
Damage_median
Damage_lcb
source_positive_lost_after_generation_rate
source_negative_fixed_after_generation_rate
```

### Pass 标准

```text
no_transform_sentinel_pass = 1
action_apply_error_linf_max = 0.0
branch_horizon_completion_rate = 1.0
source_positive_lost_after_generation_rate <= 0.30
Damage_median >= -0.05
h20_weak_CP_generated >= 0.50
h20_V_ctrl_lcb_generated > 0
h240_longrisk_generated <= 0.10
```

### 可视化

```text
p7_source_to_generated_damage_matrix.svg
p7_generator_value_retention_curve.svg
p7_generated_h20_vs_h240_scatter.svg
p7_no_transform_sentinel_status.svg
```

---

## P8：canonical effect-valid certificate revalidation

### Gate

P8 只在 P7 generator preservation 或 P5 source frontier 存在时运行。若 canonical frontier absent，则不做 certificate threshold search。

### 目标

重建 certificate，让它预测 canonical effect，而不是 construction validity。

### Candidate certificates

```text
CERT11-CanonicalValueLCBCert
CERT12-HorizonLongRiskUCBCert
CERT13-AdamWConflictValueCert
CERT14-TailMarginResponseCert
CERT15-MinimalHybridEffectCert
CERT16-SourcePreservationCert
```

### 必须记录

```text
certificate_id
certificate_fields
payload_hash_bound
commit_time_available
uses_outcome_at_commit
certificate_cost_ms_q90
certificate_pass_count
AUC_Yrobust
AUC_WeakCP_h20
AUC_LongRisk_h240
P_Yrobust_given_cert_pass
P_Yrobust_given_cert_fail
P_longrisk_given_cert_pass
P_longrisk_given_cert_fail
Lift_Yrobust
Lift_longrisk_inverse
monotone_sign_pass
leave_dataset_auc_drop
leave_family_auc_drop
```

### Pass 标准

Weak certificate pass：

```text
AUC_Yrobust >= 0.70
P_Yrobust_given_cert_pass >= 0.30
P_longrisk_given_cert_pass <= 0.15
monotone_sign_pass = 1
```

Strong certificate pass：

```text
AUC_Yrobust >= 0.75
inverse_AUC_LongRisk >= 0.75
P_Yrobust_given_cert_pass >= 0.50
P_longrisk_given_cert_pass <= 0.10
certificate_cost_ms_q90 <= 0.20
leaveout drops within bounds
```

### 可视化

```text
p8_certificate_roc_pr.svg
p8_certificate_calibration_curve.svg
p8_cert_pass_fail_outcome_bar.svg
p8_certificate_cost_pareto.svg
p8_certificate_ablation.svg
```

---

## P9：minimal canonical source controller

### Gate

P9 只在 P6/P8 至少一个 legal signal pass 后运行。

### 目标

构造最小 controller，不能堆 opaque feature factory。Controller 最多使用 5 个 primitive groups：

```text
value_lcb
bad_or_longrisk_ucb
support_lcb
certificate_score
cost
```

Controller 形式：

$$
Accept(a)=1
\iff
LCB(V(a))>0
\land
UCB(LongRisk(a))\le \tau_h
\land
LCB(Support(a))\ge \tau_s
\land
Cert(a)\ge \tau_c
\land
Cost(a)\le C_{max}.
$$

### 必须记录

```text
controller_id
feature_groups_used
feature_count
thresholds
calibration_split
heldout_split
leave_dataset_split
accepted_count_cal
accepted_count_heldout
coverage_heldout
precision_Yrobust_heldout
h20_V_ctrl_lcb_heldout
h240_longrisk_heldout
bad_event_heldout
null_rate_heldout
support_balance_pass
accepted_family_count
max_family_share
feature_cost_q90
controller_cost_q90
```

### Pass 标准

```text
feature_count <= 5
uses_dataset_name = 0
uses_validation_or_test = 0
uses_outcome_at_commit = 0
coverage_heldout in [0.03,0.15]
precision_Yrobust_heldout >= 0.50 weak, >=0.75 strong
h20_V_ctrl_lcb_heldout > 0
h240_longrisk_heldout <= 0.10
support_balance_pass = 1
controller_cost_q90 <= 0.20 ms
```

### 可视化

```text
p9_controller_frontier.svg
p9_coverage_precision_longrisk_tradeoff.svg
p9_controller_ablation.svg
p9_calibration_to_heldout_drift.svg
p9_family_support_balance.svg
```

---

## P10：selected controller online runtime boundary

### Gate

P10 只在 P9 selected controller pass 后运行。否则只运行 microbench diagnostic，不 official。

### 目标

验证 selected controller + selected payload apply 在 online training path 中满足 MLP-comparable cost。

### 必须记录

```text
runtime_candidate_id
controller_id
selected_payload_apply_used
control_outcome_materializer_in_timed_path
offline_audit_in_timed_path
disk_payload_lookup_in_timed_path
payload_preloaded
step_count
active_step_count
candidate_count
accepted_count
zero_candidate_controller_kernel_count
zero_candidate_controller_sync_count
feature_compute_time_ms_q90
score_accept_time_ms_q90
certificate_compute_time_ms_q90
payload_apply_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
payload_apply_error_linf_max
payload_apply_cosine_min
```

### Pass 标准

```text
selected_controller_used = 1
selected_payload_apply_used = 1
control_outcome_materializer_in_timed_path = 0
offline_audit_in_timed_path = 0
zero_candidate_controller_kernel_count = 0
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
payload_apply_error_linf_max <= 1e-6
```

### 可视化

```text
p10_runtime_waterfall.svg
p10_step_ratio_distribution.svg
p10_payload_apply_time_hist.svg
p10_selected_vs_base_step_time.svg
p10_memory_ratio_trace.svg
```

---

## P11：Base-Acc Sentinel continuation and stronger baseline monitor

### 目标

继续回答用户关心的 acc 问题，但保持隔离，不让 dataset accuracy 调 controller。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
models:
  LQ-t2-h256
  MatchedMLP
  AdamWStrongLRGridMLP
  QuadraticFeatureMLP
optional:
  MLP-wide-param-matched
  MLP-time-budget-matched
```

### 必须记录

```text
dataset
seed
model_id
param_count
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
memory_peak
hyperparams_fixed_before_run
dataset_specific_tuning
used_for_controller
```

### Pass 标准

```text
sentinel_complete = 1
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
same_seed_schedule = 1
same_budget = 1
dataset_specific_tuning = 0
```

Catastrophic fail definition：

```text
LQ mean test acc < MatchedMLP mean test acc - 0.05
or any dataset LQ collapse rate > 0.20 across seeds
```

### 可视化

```text
p11_acc_by_dataset_model.svg
p11_lq_minus_mlp_by_seed.svg
p11_loss_curve_sentinel.svg
p11_calibration_sentinel.svg
p11_step_time_memory_sentinel.svg
```

---

## P12：system integration gate

### 目标

统一判断 v9.4.8 是否可以打开 LDO/LSO / paired replay / short-full。

### 必须记录

```text
canonical_full_control_outcome_ready
old_table_quarantine_enforced
canonical_AP0_frontier_pass
canonical_source_frontier_pass
legal_observability_pass
generator_preservation_pass
certificate_effect_valid_pass
source_controller_pass
selected_runtime_pass
base_acc_sentinel_pass
official_eligible
system_legal_controller_pass
primary_blocker
next_required_implementation
```

### Pass 标准

```text
canonical_full_control_outcome_ready = 1
old_table_quarantine_enforced = 1
source_controller_pass = 1
selected_runtime_pass = 1
official_eligible = 1
system_legal_controller_pass = 1
```

### Route decisions

```text
R0-CanonicalOutcomeTableIncomplete
R1-OldTableQuarantineViolation
R2-CanonicalAP0ControlPositiveFrontierAbsent
R3-CanonicalAP0ShortHorizonOnlyLongRiskHigh
R4-CanonicalSourceFrontierExistsLegalOpaque
R5-CanonicalGeneratorDestructive
R6-CanonicalCertificateNotEffectValid
R7-CanonicalControllerSupportCollapse
R8-SelectedRuntimeFail
R9-SystemLegalControllerPass
```

### 可视化

```text
p12_system_gate_ladder.svg
p12_route_decision_tree.svg
p12_blocker_waterfall.svg
```

---

## P13：conditional LDO / LSO / paired replay boundary

### Gate

只有 P12 system pass 后打开。

### 目标

证明 selected controller 不是 dataset-specific / stratum-specific artifact。

### 设置

Leave-dataset-out：

```text
train/calibrate on two datasets, evaluate held-out third dataset
MNIST heldout
Fashion-MNIST heldout
KMNIST heldout
```

Leave-stratum-out：

```text
hold out family / horizon / score bucket / payload norm bucket strata
```

Official paired replay：

```text
RealFunctional
AdamWParallel
bestLR
NoOp
Random
ShuffledFunctionalPayload
```

### 必须记录

```text
split_type
heldout_dataset_or_stratum
controller_id
accepted_count
coverage
precision_Yrobust
h20_V_ctrl_lcb
h240_longrisk
beats_adamwparallel_rate
beats_bestlr_rate
beats_noop_rate
beats_random_rate
shuffle_control_pass
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
step_ratio_q90
memory_ratio
```

### Pass 标准

```text
LDO: at least 2/3 heldout datasets pass weak gate
LSO: at least 70% heldout strata pass weak gate
paired replay: RealFunctional beats AdamWParallel and bestLR at rate >= 0.50 weak, >=0.60 strong
shuffle controls fail to match RealFunctional
```

### 可视化

```text
p13_leave_dataset_out_matrix.svg
p13_leave_stratum_out_matrix.svg
p13_paired_replay_winrate.svg
p13_shuffle_control_ablation.svg
```

---

## P14：conditional short/full validation boundary

### Gate

只有 P12/P13 pass 后打开。

### 目标

回答最终用户关心的问题：functional DG-KAN 在真实 short/full training 中是否超越 matched MLP / strong MLP。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
models:
  LQ-t2-h256 base
  LQ-t2-h256 + selected functional controller
  MatchedMLP
  AdamWStrongLRGridMLP
  QuadraticFeatureMLP
  time-budget-matched MLP
runs:
  short-run
  full-run
  sample-efficiency run
  continual / anti-forgetting run
```

### 必须记录

```text
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
calibration_error
robustness_corruption_acc
ValLossAUC_step
ValLossAUC_time
steps_to_target
time_to_target
continual_retained_accuracy
forgetting
backward_transfer
forward_transfer
step_ratio_q90
memory_ratio
```

### Pass 标准

Short/full weak pass：

```text
FunctionalLQ mean test acc >= StrongLRGridMLP mean test acc - 0.005
and at least one of:
  ValLossAUC_time better
  ECE better
  CEp99 better
  sample efficiency better
  continual forgetting lower
```

Strong pass：

```text
FunctionalLQ mean test acc > StrongLRGridMLP by statistically supported margin
and no dataset catastrophic fail
and runtime envelope pass
```

### 可视化

```text
p14_acc_by_dataset_seed.svg
p14_functional_vs_base_vs_mlp.svg
p14_sample_efficiency_curve.svg
p14_continual_forgetting_curve.svg
p14_calibration_robustness_dashboard.svg
```

---

# 8. 判断标准总表

| Gate | Pass 条件 | Fail 后下一步 |
|---|---|---|
| P0 quarantine | old table official block = 1 | 修 reader/versioning |
| P1 replay preflight | no-transform eq = 1, neg ctrl diverges | 修 runner semantics |
| P2 canonical table | 51768 rows full or 864 panel official-minimum | 修 materializer / shard |
| P3 drift taxonomy | attribution >= 0.90 | 修 table / label config |
| P4 CP oracle | coverage >= 0.03, V LCB > 0 | source primitive redesign |
| P5 source frontier | h20 value positive, h240 long-risk <= 0.10 | direct robust source generator |
| P6 legal observability | AUC / TopK / cost pass | certificate-producing primitive |
| P7 generator | no-transform sentinel + preservation pass | objective-solving generator |
| P8 certificate | effect-valid pass | certificate redesign |
| P9 controller | heldout coverage / value / longrisk pass | minimal controller redesign |
| P10 runtime | step ratio <= 1.50 | payload/runtime optimization |
| P13 LDO/LSO | leave-out pass | support/causal redesign |
| P14 short/full | beats strong controls or wins efficiency/robustness | external-ready fail |

---

# 9. 需要落盘的关键 artifacts

```text
route_decision.json
run_manifest.json
aggregate_decision_v9480.json
p0_v9470_boundary_reproduction.csv
p0_old_table_quarantine_reader_audit.csv
p1_canonical_replay_preflight.csv
p1_no_transform_sentinel_grid.csv
p2_canonical_ap0_full_control_outcome_materializer.csv
canonical_full_control_outcome_table_v9480.csv
p2_quality_audit.csv
p3_old_new_drift_taxonomy.csv
p4_canonical_control_positive_oracle.csv
p5_canonical_source_frontier.csv
p6_canonical_legal_observability.csv
p7_canonical_generator_preservation.csv
p8_effect_valid_certificate.csv
p9_minimal_source_controller.csv
p10_selected_runtime.csv
p11_base_acc_sentinel_continuation.csv
p12_system_integration_gate.csv
p13_leaveout_paired_replay_boundary.csv
p14_short_full_validation_boundary.csv
no_fake_audit.csv
contract_audit.csv
provenance_audit.csv
failure_table.csv
```

---

# 10. 关键可视化清单

```text
v9480_dashboard_system_ladder.svg
p0_old_new_orc_d_k16_flip.svg
p1_no_transform_equivalence_matrix.svg
p2_materializer_completion_curve.svg
p2_branch_horizon_completion_heatmap.svg
p3_old_new_Vctrl_scatter.svg
p3_drift_taxonomy_pie.svg
p4_canonical_oracle_frontier.svg
p5_source_frontier_K_curve.svg
p6_legal_feature_auc_cost_pareto.svg
p7_generator_damage_matrix.svg
p8_certificate_calibration_curve.svg
p9_controller_frontier.svg
p10_runtime_waterfall.svg
p11_acc_by_dataset_model.svg
p13_paired_replay_winrate.svg
p14_sample_efficiency_curve.svg
```

---

# 11. v9.4.8 预期结果解释模板

## Case A：canonical AP0 frontier 消失

```text
canonical_control_positive_oracle_pass = 0
canonical_source_frontier_pass = 0
```

解释：旧 v9.3.5 outcome table 对 AP0 value 的判断不可信。下一步应停止 AP0 source selector/generator patch，转向 source primitive redesign，并重新定义 value-producing action。

## Case B：canonical AP0 weak frontier 存在，但 horizon-robust 不存在

```text
weak CP oracle pass = 1
horizon robust source pass = 0
```

解释：AP0 仍有短期正收益，但长期风险太高。下一步应做 horizon-aware generator / long-risk certificate，不应只追 h20 weak CP。

## Case C：canonical robust source frontier 存在，但 legal selector 不可见

```text
source_frontier_pass = 1
legal_observability_pass = 0
```

解释：good source exists but legally opaque。下一步应做 certificate-producing / objective-solving generator，而不是 static feature search。

## Case D：source frontier 存在，generator 破坏它

```text
source_frontier_pass = 1
generator_preservation_pass = 0
no_transform_sentinel_pass = 1
```

解释：这时才可以恢复“generator destructive”结论。下一步应改 generator objective，而不是修 table。

## Case E：certificate fail

```text
generator_preservation_pass = 1
certificate_effect_valid_pass = 0
```

解释：payload 生成有价值，但 certificate 不是 sufficient statistic。下一步应重构 effect certificate。

## Case F：controller/runtime pass

```text
system_legal_controller_pass = 1
```

解释：才允许打开 LDO/LSO、official paired replay 和 short/full validation。

---

# 12. 最终原则

v9.4.8 的核心纪律是：

$$
\boxed{
\text{先重建 canonical truth，再评价 generator/certificate/controller。}
}
$$

不要再做：

```text
AP0r/AP0t/AP0u 小参数微调；
CERT8 threshold 微调；
state_NLL_proxy topK 微调；
继续引用旧 v9.3.5 ORC-D-K16 survivor 作为 official；
继续用旧 full control table 训练或选择 controller；
继续把 Base-Acc Sentinel 当 functional proof。
```

要做：

```text
canonical outcome table v9480；
old-new drift taxonomy；
canonical AP0 source frontier revalidation；
canonical legal observability；
canonical generator preservation；
effect-valid certificate；
selected runtime；
then LDO / paired replay / short-full。
```

一句话总结：

$$
\boxed{
\text{v9.4.7 证明旧表不能信；v9.4.8 必须用新 runner 重新建立可被信任的 outcome universe。}
}
$$
