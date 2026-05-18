# DG-KAN v9.4.7 No-Transform Replay Semantics Closure / Robust Source Revalidation 完整实验计划

> 本计划基于 v9.4.6 的真实执行结果制定。v9.4.7 不新增 AP 小变体，不继续调 certificate threshold，不做 dataset-specific tuning。当前首要任务是修复并证明：同一个 source action 与 no-transform clone 在相同 branch start state、相同 payload、相同 branch/horizon semantics 下，必须产生相同 rollout state、metrics 与 labels。
>
> v9.4.7 的核心命题是：
>
> $$
> \boxed{\text{No-transform replay must be outcome-equivalent before generator, certificate, controller, runtime, paired replay can be trusted.}}
> $$

---

# 0. 执行摘要

v9.4.6 的结果不是 functional success，也不是 generator/certificate failure。它把 v9.4.5 的 `OracleSourceIdentityBug` 进一步拆开了：

```text
source identity ledger: pass
ORC-D-K16 source table recompute: pass
AP0w no-transform payload equivalence: pass
oracle aggregation recompute: pass
side-by-side no-transform outcome replay: fail
```

最关键的数据是：

```text
route = R1d-NoTransformOutcomeReplayBug
source_identity_ledger_pass = 1
ORC_D_K16_join_success_count = 16 / 16
source_recomputed_h20_weak_CP = 0.8125
source_recomputed_h20_V_ctrl_lcb = 0.13772944106165844
source_recomputed_h240_long_risk = 0.0
no_transform_payload_equivalence_pass = 1
clone_payload_hash_match_rate = 1.0
clone_payload_linf_max = 0.0
clone_payload_relative_error_max = 0.0
clone_payload_cosine_min = 0.9999999999999998
side_by_side_no_transform_replay_pass = 0
paired_branch_horizon_row_count_actual = 48 / 48
metric_abs_diff_max = 5.448057344648987
label_match_rate = 0.6666666666666666
branch_state_hash_match_rate = 1.0
horizon_state_hash_match_rate = 0.0
failure_class_primary = I18-outcome-materializer-runner-drift
primary_blocker = no_transform_outcome_replay_inconsistent
```

这说明当前不是 source ledger 缺失，也不是 payload 被 transform 改坏，而是 outcome materializer / runner semantics 不等价。更直白地说：

$$
\boxed{
\text{同 payload、同 branch start state 下，source 与 no-transform clone 走出了不同 horizon state。}
}
$$

在这个 blocker 没闭合前，不能继续判断：

```text
AP0r/AP0v generator 是否 destructive；
AP0b/AP0q direct generator 是否真的没有 value；
CERT8/CERT10 是否真的不 effect-valid；
ORC-D-K16 是否能作为新 rollout survivor；
source controller / selected runtime / paired replay 是否应该打开。
```

v9.4.7 的目标不是做新 primitive，而是把 outcome replay 变成可审计、可复现、branch-name-invariant 的 deterministic runner。只有 P5 no-transform outcome equivalence 过线，才允许重开 generator/certificate/controller。

---

# 1. 独立判断：v9.4.6 到底说明了什么

## 1.1 有进展，但不是能力进展

v9.4.6 的真实进展是 identity 层和 payload 层被排除为主因：

```text
1. ORC-D-K16 ledger rows = 16；
2. source join success = 16 / 16；
3. duplicate action id = 0；
4. payload/state/branch/label hash missing = 0；
5. source outcome oracle 可从 v9.3.5 measured source table 重算；
6. no-transform clone payload 与 source payload 完全等价。
```

这比 v9.4.5 更精确。v9.4.5 只能说 no-transform oracle-seeded failed；v9.4.6 进一步证明 no-transform 的 payload 没被改，source identity 也没断，真正断在 side-by-side rollout semantics。

## 1.2 为什么仍然感觉很慢

你的“慢、没进度”的感觉是合理的。最近几轮的进展顺序是：

```text
v9.4.1: source generator materialized，但 source outcome rows = 0；
v9.4.2: source outcome rows 修到 7020 / 7020，但 source panel / generated source value-poor；
v9.4.3: multi-panel oracle 接近但没 official survivor；
v9.4.4: exhaustive oracle 找到 ORC-D-K16 survivor，但 generator 看起来 destructive；
v9.4.5: no-transform replay 也失败，怀疑 source identity / replay path；
v9.4.6: identity 和 payload 排除，定位到 outcome runner drift。
```

科学上这不是原地踏步，因为每轮都把 blocker 往更基础的位置推进；但执行上确实太串行。v9.4.7 必须改成分层 preflight 与并行矩阵，不再等完整 runner 跑完才发现下一个一致性 bug。

## 1.3 当前结论不能被误读

不能把 v9.4.6 读成：

```text
generator 一定失败；
certificate primitive 一定失败；
ORC-D-K16 oracle survivor 是假阳性；
AP0 全体没有好 action；
LQ base 不如 MLP；
functional route 失败。
```

更准确的判断是：

$$
\boxed{
\text{ORC-D-K16 在旧 measured table 中是可重算的，但新 side-by-side no-transform rollout 还不能复现它。}
}
$$

这意味着当前科学对象还没稳定。只要 no-transform replay 不等价，所有 generator/certificate 的 outcome 判断都可能受到 runner drift 污染。

---

# 2. 当前项目状态

## 2.1 已经完成的资产

```text
LQ-t2-h256 base anchor；
manual forward / backward / AdamW update contract；
full AP0 control outcome universe；
durable AP0 payload package；
AP0 weak control-positive oracle existence；
AP0 horizon-robust oracle survivor diagnostic；
ORC-D-K16 source identity ledger；
AP0w no-transform payload equivalence；
Base-Acc Sentinel strong baseline；
no fake / no proxy / no teacher / no loss-modification audit。
```

## 2.2 当前未完成的核心门

```text
1. no-transform outcome replay equivalence；
2. deterministic branch/horizon runner semantics；
3. repaired-runner ORC-D-K16 direct rollout reproduction；
4. generator preservation revalidation；
5. effect-valid certificate revalidation；
6. minimal source certificate controller；
7. selected source online runtime；
8. LDO / LSO；
9. official paired replay；
10. short/full/sample-efficiency/continual validation。
```

## 2.3 有在数据集上训练吗？acc 如何？和 MLP 比如何？

v9.4.6 有训练，但只是 Base-Acc Sentinel，不是 official functional training，也不是 controller-selected short/full run。

Base-Acc Sentinel 配置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
model_count = 4
sentinel_row_count = 120
base_acc_used_for_controller = 0
```

结果：

```text
mean_test_acc_LQ = 0.6537760416666667
mean_test_acc_MLP = 0.562890625
mean_test_acc_QuadraticFeatureMLP = 0.40234375
mean_test_acc_AdamWStrongLRGridMLP = 0.628515625
LQ_minus_MLP_mean_test_acc = 0.09088541666666672
LQ_minus_AdamWStrongLRGridMLP_mean_test_acc = 0.025260416666666674
LQ_catastrophic_fail = 0
```

独立判断：

```text
1. LQ base 没有 catastrophic fail；
2. LQ 在固定 sentinel 下比 MatchedMLP 高约 9.09 个百分点；
3. LQ 在固定 sentinel 下比 AdamWStrongLRGridMLP 高约 2.53 个百分点；
4. 但这不是 functional DG-KAN 成功，因为 functional controller、selected runtime、paired replay、short/full training 都没打开；
5. 绝对 acc 仍不是打榜级别，只能作为 base health sentinel。
```

---

# 3. v9.4.7 总体目标

v9.4.7 的总体目标是：

$$
\boxed{
\text{修复并证明 AP0 source 与 AP0w no-transform clone 的 outcome replay 完全等价。}
}
$$

具体说，给定 source action $a$ 与 clone action $c$，如果满足：

$$
\Delta\theta_c = \Delta\theta_a,
$$

$$
Hash(\theta_{branch,0}^c)=Hash(\theta_{branch,0}^a),
$$

$$
Hash(Config_{branch}^c)=Hash(Config_{branch}^a),
$$

$$
Hash(Config_{horizon}^c)=Hash(Config_{horizon}^a),
$$

则必须满足：

$$
Hash(\theta_{branch,h}^c)=Hash(\theta_{branch,h}^a),
$$

$$
Metrics(c,b,h)=Metrics(a,b,h),
$$

$$
Labels(c,b,h)=Labels(a,b,h).
$$

v9.4.7 的强目标：

```text
side_by_side_no_transform_replay_pass = 1
branch_state_hash_match_rate = 1.0
horizon_state_hash_match_rate = 1.0
metric_abs_diff_max <= 1e-6
label_match_rate = 1.0
ORC_D_K16_direct_new_rollout_reproduces_source = 1
```

v9.4.7 的最低有效推进目标：

```text
1. 找到 first divergence step；
2. 将 failure class 从 I18-outcome-materializer-runner-drift 细分到具体语义 bug；
3. 证明修复前后 side-by-side metrics/labels 的变化；
4. 若修复后 ORC-D-K16 不再成立，明确 quarantine old v9350 outcome table；
5. 若修复后 ORC-D-K16 成立，重开 generator/certificate revalidation。
```

---

# 4. v9.4.7 不做什么

本轮不做：

```text
1. 不新增 AP0x/AP1x 小变体；
2. 不调 AP0r/AP0t/AP0u/AP0w 参数；
3. 不调 CERT8/CERT10 threshold；
4. 不使用 ORC-D-K16 作为 official selector；
5. 不用 old measured table 的 source outcome 直接做 controller；
6. 不打开 selected runtime；
7. 不打开 official paired replay；
8. 不打开 short/full functional validation；
9. 不按 dataset 调 branch semantics、seed、selector、threshold 或 runtime route；
10. 不把 Base-Acc Sentinel 当作 functional success。
```

允许做：

```text
1. dataset-level diagnostic；
2. branch-level diagnostic；
3. horizon-level diagnostic；
4. single-action deterministic replay preflight；
5. same payload source/clone side-by-side replay；
6. step-wise state hash bisection；
7. repaired-runner old-vs-new outcome drift analysis；
8. conditional generator/certificate revalidation。
```

---

# 5. 核心假设

## H1：当前 blocker 不在 source identity 或 payload identity，而在 outcome runner semantics

v9.4.6 已经支持 H1：source ledger pass、payload equivalence pass，但 side-by-side outcome replay fail。

H1 成立标准：

```text
source_identity_ledger_pass = 1
no_transform_payload_equivalence_pass = 1
side_by_side_no_transform_replay_pass = 0
branch_state_hash_match_rate = 1.0
horizon_state_hash_match_rate = 0.0
```

H1 失败标准：

```text
发现 source ledger 或 payload equivalence 仍有 hidden mismatch。
```

若 H1 失败，先回到 ledger/payload。若 H1 成立，进入 runner semantics repair。

## H2：branch name / branch dispatch path 不应改变 no-transform rollout

如果 payload、branch start state、optimizer state、RNG state、dataloader sequence、branch config 与 horizon config 相同，则 branch 名称不应影响 rollout。

H2 成立标准：

```text
rename branch_id / clone_action_id 后：
  horizon_state_hash_match_rate = 1.0
  metric_abs_diff_max <= 1e-6
  label_match_rate = 1.0
```

H2 失败标准：

```text
branch name 或 clone_action_id 改变 rollout path。
```

这类失败必须修复为 branch-name-invariant materializer。

## H3：horizon replay 必须锁定 batch sequence、RNG、optimizer state 与 update order

H3 认为 P4 的 `branch_state_hash_match = 1.0` 但 `horizon_state_hash_match = 0.0`，很可能来自 horizon 内部 rollout seed、batch cursor、optimizer state、branch dispatch 或 update order 不一致。

H3 成立标准：

```text
rng_state_hash_match_rate = 1.0
batch_sequence_hash_match_rate = 1.0
optimizer_state_hash_match_rate = 1.0
update_order_id_match_rate = 1.0
horizon_state_hash_match_rate = 1.0
```

H3 失败标准：

```text
任意一个 horizon-step 内部 state hash 出现 divergence。
```

## H4：如果 no-transform equivalence 过，才可以重新判断 generator/certificate

H4 成立标准：

```text
P5 side_by_side_no_transform_replay_pass = 1
P6 ORC-D-K16 direct rollout reproduced = 1
then run generator/certificate revalidation。
```

H4 失败标准：

```text
no-transform equivalence 未过，却继续评估 AP0r/AP0v generator。
```

这属于错误实验路线。

## H5：如果 repaired runner 下 ORC-D-K16 不复现，旧 outcome table 需要 quarantine

H5 认为旧 v9.3.5 measured table 虽然可重算，但如果新 deterministic runner 不能复现 ORC-D-K16，则旧 table 不能继续支撑 generator/certificate/controller 结论。

H5 成立标准：

```text
old_source_table_oracle_pass = 1
new_direct_rollout_oracle_pass = 0
old_new_outcome_drift_explained = 1
old_table_quarantined_for_official = 1
```

H5 失败标准：

```text
new direct rollout reproduces ORC-D-K16。
```

若 H5 成立，应进入 full AP0 outcome rerun under repaired runner。

---

# 6. 数据合同

## 6.1 source/clone identity contract

每个 source/clone pair 必须记录：

```text
source_action_id
clone_action_id
source_candidate_id
clone_candidate_id
source_payload_hash
clone_payload_hash
source_payload_linf
clone_payload_linf
payload_linf_diff
payload_relative_diff
payload_cosine
source_state_before_hash
clone_state_before_hash
source_optimizer_state_hash
clone_optimizer_state_hash
source_rng_state_hash
clone_rng_state_hash
source_dataloader_cursor_hash
clone_dataloader_cursor_hash
branch_config_hash
horizon_config_hash
label_config_hash
materializer_version
runner_version
update_order_id
branch_dispatch_id
```

Pass：

```text
payload_hash_match_rate = 1.0
payload_linf_max = 0.0
payload_relative_error_max = 0.0
payload_cosine_min >= 0.999999999999
state_before_hash_match_rate = 1.0
optimizer_state_hash_match_rate = 1.0
rng_state_hash_match_rate = 1.0
branch_config_hash_match_rate = 1.0
horizon_config_hash_match_rate = 1.0
label_config_hash_match_rate = 1.0
```

## 6.2 step-wise replay contract

对于每个 action pair、branch、horizon、inner step $k$，必须记录：

```text
source_action_id
clone_action_id
branch_id
horizon
inner_step_k
batch_id
batch_hash
rng_state_hash_before
rng_state_hash_after
theta_hash_before
theta_hash_after
optimizer_m_hash_before
optimizer_m_hash_after
optimizer_v_hash_before
optimizer_v_hash_after
logits_hash
loss_value
metric_snapshot_hash
payload_applied_flag
adamw_applied_flag
functional_applied_flag
update_order_id
first_divergence_flag
first_divergence_field
```

Pass：

```text
for all k:
  batch_hash_source = batch_hash_clone
  rng_state_hash_source = rng_state_hash_clone
  theta_hash_source = theta_hash_clone
  optimizer_state_hash_source = optimizer_state_hash_clone
  logits_hash_source = logits_hash_clone
```

允许浮点容差 metric pass：

$$
\max |Metric_{source}-Metric_{clone}| \le 10^{-6}.
$$

官方 hash pass 目标仍是 exact hash match。

## 6.3 outcome label contract

每个 source/clone pair 必须记录：

```text
safe_good_label_source
safe_good_label_clone
bad_event_label_source
bad_event_label_clone
null_event_label_source
null_event_label_clone
weak_CP_source
weak_CP_clone
strong_CP_source
strong_CP_clone
long_risk_source
long_risk_clone
Y_robust_source
Y_robust_clone
V_ctrl_source
V_ctrl_clone
V_ctrl_abs_diff
```

Pass：

```text
label_match_rate = 1.0
weak_CP_match_rate = 1.0
strong_CP_match_rate = 1.0
long_risk_match_rate = 1.0
Y_robust_match_rate = 1.0
max_V_ctrl_abs_diff <= 1e-6
```

---

# 7. 并行执行策略

v9.4.7 必须避免“一轮一个 blocker”。本轮使用四条并行 lane，但所有 lane 的 official promotion 都受 P5 no-transform equivalence gate 控制。

## Lane A：deterministic replay semantics

目标：最快定位 first divergence。

执行顺序：

```text
A1: 1 action × 1 branch × h=1
A2: 1 action × all branches × h=1
A3: 1 action × all branches × h=20
A4: 3 actions × all branches × h=20
A5: 16 ORC-D actions × all branches × h=20/80/240
```

每个 A 子阶段如果 fail，立即写 failure trace，不继续扩大规模。

## Lane B：old-vs-new outcome drift audit

目标：判断 v9350 measured table 是否仍可作为 official 依据。

执行：

```text
B1: ORC-D-K16 old table aggregation recompute；
B2: repaired runner direct rollout；
B3: old-vs-new metric/label drift；
B4: drift attribution by branch/horizon/inner-step。
```

## Lane C：generator/certificate sandbox revalidation

目标：准备好后续重开，但不抢跑 official。

执行：

```text
C1: AP0w no-transform only；
C2: AP0r/AP0v oracle-seeded transforms；
C3: AP0l/AP0q direct generators；
C4: CERT6-CERT10 diagnostics。
```

规则：

```text
若 P5 fail：C lane 只能落 diagnostic，不得进入 route decision。
若 P5 pass：C lane 可进入 P7/P8 official revalidation。
```

## Lane D：Base-Acc Sentinel continuation

目标：继续监控 base health，不参与 controller。

执行：

```text
MNIST / Fashion-MNIST / KMNIST
seeds 0..9
models: LQ-t2-h256, MatchedMLP, AdamWStrongLRGridMLP, QuadraticFeatureMLP
```

记录 acc、NLL、ECE、CEp99、time-to-target，但 `base_acc_used_for_controller = 0`。

---

# 8. 实验阶段

---

## P0：v9.4.6 boundary reproduction

### 目标

确认 v9.4.6 的 route、identity pass、payload pass、side-by-side fail 可以稳定复现。

### 假设

H1：当前 blocker 是 no-transform outcome replay inconsistent，不是 source ledger 或 payload equivalence。

### 必须记录

```text
route_v9460
source_route_v9450
source_identity_ledger_pass
ORC_D_K16_join_success_count
ORC_D_K16_join_missing_count
oracle_survivor_replay_pass
source_replay_mode
direct_new_rollout_performed
source_recomputed_h20_weak_CP
source_recomputed_h20_V_ctrl_lcb
source_recomputed_h240_long_risk
no_transform_payload_equivalence_pass
clone_payload_hash_match_rate
clone_payload_linf_max
clone_payload_relative_error_max
clone_payload_cosine_min
side_by_side_no_transform_replay_pass
paired_branch_horizon_row_count_expected
paired_branch_horizon_row_count_actual
metric_abs_diff_max
label_match_rate
branch_state_hash_match_rate
horizon_state_hash_match_rate
failure_class_primary
```

### 判断标准

P0 pass：

```text
source_identity_ledger_pass = 1
no_transform_payload_equivalence_pass = 1
side_by_side_no_transform_replay_pass = 0
branch_state_hash_match_rate = 1.0
horizon_state_hash_match_rate = 0.0
failure_class_primary = I18-outcome-materializer-runner-drift
```

### 可视化

```text
p0_v9460_boundary_ladder.svg
p0_identity_payload_outcome_gate_matrix.svg
p0_branch_vs_horizon_hash_match.svg
p0_metric_diff_distribution.svg
```

---

## P1：source/clone replay ledger hardening

### 目标

把 source 和 AP0w clone 的所有 replay keys 落成完整 ledger，避免“payload 一样但 optimizer/RNG/dataloader/branch semantics 不知道是否一样”。

### 假设

H1/H3：现有 P4 failure 可能来自未记录的 replay semantics，而不是 action payload。

### 必须记录

```text
source_action_id
clone_action_id
source_payload_hash
clone_payload_hash
source_state_before_hash
clone_state_before_hash
source_optimizer_state_hash
clone_optimizer_state_hash
source_rng_state_hash
clone_rng_state_hash
source_dataloader_cursor_hash
clone_dataloader_cursor_hash
branch_config_hash_source
branch_config_hash_clone
horizon_config_hash_source
horizon_config_hash_clone
label_config_hash_source
label_config_hash_clone
branch_dispatch_id_source
branch_dispatch_id_clone
runner_version_source
runner_version_clone
materializer_version_source
materializer_version_clone
update_order_id_source
update_order_id_clone
```

### 判断标准

P1 pass：

```text
all replay key match rates = 1.0
missing replay key count = 0
source_clone_ledger_complete = 1
```

P1 fail route：

```text
R1a-ReplayKeyLedgerIncomplete
```

### 可视化

```text
p1_replay_key_match_heatmap.svg
p1_source_clone_ledger_dag.svg
p1_missing_replay_key_table.svg
```

---

## P2：single-action deterministic no-transform preflight

### 目标

用最小 action/branch/horizon 组合快速定位 runner 是否 deterministic。先不跑 16 actions，避免浪费。

### 设计

选择：

```text
1 个 ORC-D survivor action；
1 个 branch；
horizon = 1, 2, 5, 20。
```

执行 source 与 clone side-by-side，逐 inner step 记录状态。

### 必须记录

```text
action_id
clone_action_id
branch_id
horizon
inner_step_k
batch_hash_source
batch_hash_clone
rng_hash_source
rng_hash_clone
theta_hash_source_before
theta_hash_clone_before
theta_hash_source_after
theta_hash_clone_after
optimizer_hash_source_before
optimizer_hash_clone_before
optimizer_hash_source_after
optimizer_hash_clone_after
logits_hash_source
logits_hash_clone
loss_source
loss_clone
payload_applied_flag_source
payload_applied_flag_clone
adamw_applied_flag_source
adamw_applied_flag_clone
update_order_id_source
update_order_id_clone
first_divergence_field
```

### 判断标准

P2 pass：

```text
first_divergence_field = none
all step hashes match
metric_abs_diff_max <= 1e-6
```

P2 fail route examples：

```text
R2a-BatchSequenceDrift
R2b-RNGStateDrift
R2c-OptimizerStateDrift
R2d-UpdateOrderDrift
R2e-BranchDispatchDrift
R2f-PayloadApplyOrderDrift
R2g-MetricComputationDrift
```

### 可视化

```text
p2_stepwise_hash_timeline.svg
p2_first_divergence_ladder.svg
p2_metric_diff_by_inner_step.svg
```

---

## P3：branch/horizon semantics canonical runner

### 目标

实现 branch-name-invariant 的 canonical runner。source 和 clone 必须调用同一个 rollout function，branch name 只能作为 logging metadata，不能改变 execution path。

### 设计原则

Canonical runner 函数签名：

```text
rollout_canonical(
  state_snapshot,
  optimizer_snapshot,
  rng_snapshot,
  dataloader_cursor,
  payload_tensor,
  branch_config,
  horizon_config,
  label_config,
  runner_config
) -> outcome_trace
```

禁止：

```text
if branch_name == "source" then ...
if clone_action_id then ...
if AP0w then use alternative runner path
```

允许：

```text
branch_id as metadata
clone_id as metadata
negative_control branch as explicitly different branch_config_hash
```

### 必须记录

```text
canonical_runner_version
branch_name_used_in_execution = 0
clone_id_used_in_execution = 0
runner_config_hash
branch_config_hash
horizon_config_hash
label_config_hash
same_function_pointer_hash
same_update_order_id
same_batch_sequence_hash
same_rng_sequence_hash
```

### 判断标准

P3 pass：

```text
branch_name_used_in_execution = 0
clone_id_used_in_execution = 0
same_function_pointer_hash = 1
same_update_order_id = 1
```

P3 fail route：

```text
R3-BranchNamedRunnerSemanticsLeak
```

### 可视化

```text
p3_runner_semantics_dag.svg
p3_branch_dispatch_invariance_audit.svg
p3_runner_config_hash_table.svg
```

---

## P4：side-by-side no-transform replay scale-up

### 目标

在 canonical runner 下，扩大到 ORC-D-K16 全部 source actions，并覆盖所有 official branches/horizons。

### 范围

```text
actions = 16 ORC-D-K16
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random, ShuffledFunctionalPayload
horizons = 20, 80, 240
expected paired rows = 16 * branches * horizons
```

其中 source 与 no-transform clone 的 RealFunctional branch 必须等价。Control branches 必须记录，但不要求与 RealFunctional 等价；要求同 branch_config 下 source/clone control 等价。

### 必须记录

```text
paired_action_count
paired_branch_horizon_row_count_expected
paired_branch_horizon_row_count_actual
branch_completion_rate
horizon_completion_rate
branch_state_hash_match_rate
horizon_state_hash_match_rate
optimizer_state_hash_match_rate
batch_sequence_hash_match_rate
metric_abs_diff_max
metric_abs_diff_p99
label_match_rate
weak_CP_match_rate
strong_CP_match_rate
long_risk_match_rate
Y_robust_match_rate
V_ctrl_abs_diff_max
first_divergence_count
```

### 判断标准

P4 pass：

```text
paired_branch_horizon_row_count_actual = expected
branch_state_hash_match_rate = 1.0
horizon_state_hash_match_rate = 1.0
optimizer_state_hash_match_rate = 1.0
batch_sequence_hash_match_rate = 1.0
metric_abs_diff_max <= 1e-6
label_match_rate = 1.0
```

P4 fail route：

```text
R4-NoTransformReplayStillInconsistent
```

### 可视化

```text
p4_hash_match_by_branch_horizon.svg
p4_metric_abs_diff_violin.svg
p4_label_confusion_source_clone.svg
p4_first_divergence_by_branch_horizon.svg
```

---

## P5：no-transform equivalence official gate

### 目标

把 P4 的 technical pass 汇总为 official no-transform replay gate。P5 是 v9.4.7 最重要的 gate。

### 必须记录

```text
no_transform_equivalence_candidate_id
source_action_count
clone_action_count
source_clone_payload_equivalence_pass
source_clone_replay_key_pass
side_by_side_no_transform_replay_pass
branch_state_hash_match_rate
horizon_state_hash_match_rate
metric_abs_diff_max
label_match_rate
negative_control_divergence_present
oracle_aggregation_recompute_pass
old_new_outcome_drift_audit_pass
```

Negative controls：

```text
payload_shuffled_clone should diverge；
rng_perturbed_clone should diverge；
optimizer_state_perturbed_clone should diverge。
```

这样可以确认 runner 不是“无论什么都 hash match”的假等价。

### 判断标准

P5 pass：

```text
source_clone_payload_equivalence_pass = 1
source_clone_replay_key_pass = 1
side_by_side_no_transform_replay_pass = 1
branch_state_hash_match_rate = 1.0
horizon_state_hash_match_rate = 1.0
metric_abs_diff_max <= 1e-6
label_match_rate = 1.0
negative_control_divergence_present = 1
```

若 P5 fail：

```text
Stop generator/certificate/controller/runtime.
Route = R5-NoTransformReplaySemanticsUnclosed.
```

若 P5 pass：

```text
Proceed to P6 repaired-runner source oracle direct rollout.
```

### 可视化

```text
p5_no_transform_gate_summary.svg
p5_positive_negative_control_equivalence.svg
p5_replay_contract_pass_fail_matrix.svg
```

---

## P6：repaired-runner ORC-D-K16 source oracle revalidation

### 目标

判断 ORC-D-K16 是否不仅能在 old v9350 measured table 中重算，也能在 repaired canonical runner 中 direct new rollout 复现。

### 必须记录

```text
source_replay_mode_old
source_replay_mode_new
direct_new_rollout_performed
old_h20_weak_CP
new_h20_weak_CP
old_h20_V_ctrl_lcb
new_h20_V_ctrl_lcb
old_h240_long_risk
new_h240_long_risk
old_Y_robust_count
new_Y_robust_count
old_new_label_match_rate
old_new_V_ctrl_abs_diff_max
old_new_metric_drift_mean
old_new_metric_drift_p99
old_table_quarantine_required
```

### 判断标准

P6 pass：

```text
direct_new_rollout_performed = 1
new_h20_weak_CP >= 0.60
new_h20_V_ctrl_lcb > 0
new_h240_long_risk <= 0.10
new_Y_robust_count >= 10 of 16
old_new_label_match_rate >= 0.95
```

P6 fail routes：

```text
R6a-OldOutcomeTableQuarantineRequired
R6b-ORCDK16NotRobustUnderCanonicalRunner
R6c-OracleObjectiveNeedsRerunUnderRepairedRunner
```

If P6 fail but P5 pass：

```text
Do not continue generator preservation based on old ORC-D-K16.
Run P6b full AP0 source frontier recompute under repaired runner.
```

### 可视化

```text
p6_old_vs_new_orc_metrics.svg
p6_orc_label_match_heatmap.svg
p6_orc_value_curve_h20_h80_h240.svg
p6_old_table_quarantine_decision.svg
```

---

## P6b：conditional full AP0 source frontier recompute under repaired runner

### 触发条件

```text
P5 pass and P6 fail.
```

### 目标

若 old ORC-D-K16 不复现，则不能继续依赖 v9350 outcome table。必须在 repaired runner 下重新计算 AP0 source frontier。

### 范围

优先级：

```text
Tier 1: ORC-D-K16 + nearby K64/K128 actions；
Tier 2: legal topK / random / representative panels；
Tier 3: full 2876 AP0 actions if Tier 1/2 route still ambiguous。
```

### 必须记录

```text
repaired_runner_action_count
branch_horizon_rows_expected
branch_horizon_rows_actual
rows_per_sec
quality_audit_pass
oracle_selector_id
K
h20_weak_CP
h20_V_ctrl_lcb
h240_long_risk
Y_robust_count
support_balance_pass
frontier_present
```

### 判断标准

P6b pass：

```text
frontier_present = 1
best selector satisfies:
  h20_weak_CP >= 0.60
  h20_V_ctrl_lcb > 0
  h240_long_risk <= 0.10
  support_balance_pass = 1
```

P6b fail：

```text
No repaired-runner AP0 source frontier.
Route = R6b-RepairedRunnerAP0SourceFrontierAbsent.
```

### 可视化

```text
p6b_repaired_runner_frontier_pareto.svg
p6b_old_v9350_vs_repaired_frontier_shift.svg
p6b_source_frontier_by_family_horizon.svg
```

---

## P7：generator preservation revalidation after no-transform closure

### 触发条件

```text
P5 pass and (P6 pass or P6b pass).
```

### 目标

重新判断 AP0r/AP0v 或 AP0b/AP0q 是否真的 destructive。v9.4.4/v9.4.5 的 destructive conclusion 在 P5 之前不能 official。

### 实验组

```text
G0: AP0w-NoTransformReplaySource
G1: AP0r-ObjectiveSolvedLastEdgeTrustRegionSource
G2: AP0t-TailProjectedHorizonGuardSource
G3: AP0u-AdamWCompatibleResidualSource
G4: AP0v-HorizonResidualBlendSource
G5: best AP0b/AP0f historical source generator
G6: best AP0l/AP0q direct generator
```

### 必须记录

```text
generator_id
source_panel_id
source_action_count
generated_action_count
branch_horizon_rows_expected
branch_horizon_rows_actual
source_h20_weak_CP
generated_h20_weak_CP
source_h20_V_ctrl_lcb
generated_h20_V_ctrl_lcb
source_h240_long_risk
generated_h240_long_risk
source_Y_robust_count
generated_Y_robust_count
Damage_mean
Damage_median
Damage_lcb
source_positive_lost_after_generation_rate
source_negative_fixed_after_generation_rate
payload_geometry_distortion
certificate_pass_count
```

### 判断标准

Generator preservation pass：

```text
AP0w pass by construction after P5
for nontrivial generator:
  generated_h20_weak_CP >= 0.60
  generated_h20_V_ctrl_lcb > 0
  generated_h240_long_risk <= 0.10
  source_positive_lost_after_generation_rate <= 0.25
  Damage_median >= -0.02
```

Weak diagnostic pass：

```text
generated_h20_weak_CP >= 0.40
source_positive_lost_after_generation_rate <= 0.50
```

### 可视化

```text
p7_source_to_generated_damage_matrix.svg
p7_generator_value_preservation_pareto.svg
p7_horizon_risk_by_generator.svg
p7_payload_geometry_vs_value_damage.svg
```

---

## P8：effect-valid certificate revalidation

### 触发条件

```text
P7 has at least one weak diagnostic survivor.
```

### 目标

判断 certificate 是否能在 repaired runner outcomes 上分离 robust source 与 long-risk。

### Certificate candidates

```text
CERT6-MinimalValueRiskCert
CERT7-HorizonGuardedTailCert
CERT8-AdamWConflictHorizonGuard
CERT9-SourcePreservationCert
CERT10-NoTransformEquivalenceAwareCert
CERT11-StepwiseDescentConsistencyCert
```

### 必须记录

```text
certificate_id
action_count
certificate_pass_action_count
AUC_Y_robust
AUC_weak_CP
AUC_strong_CP
AUC_longrisk_h240
P_Yrobust_given_cert_pass
P_Yrobust_given_cert_fail
P_longrisk_given_cert_pass
P_longrisk_given_cert_fail
Lift_Yrobust
Lift_longrisk_inverse
monotone_sign_pass
calibration_ECE
cost_q90_ms
```

### 判断标准

P8 pass：

```text
AUC_Y_robust >= 0.75
AUC_longrisk_h240 >= 0.75
P_Yrobust_given_cert_pass >= 0.60
P_longrisk_given_cert_pass <= 0.10
monotone_sign_pass = 1
cost_q90_ms <= 0.20
```

P8 weak pass：

```text
AUC_Y_robust >= 0.65
P_Yrobust_given_cert_pass >= 0.40
P_longrisk_given_cert_pass <= 0.20
```

### 可视化

```text
p8_certificate_roc_pr_curves.svg
p8_certificate_pass_fail_value_distribution.svg
p8_certificate_longrisk_lift.svg
p8_certificate_calibration_curve.svg
p8_certificate_cost_waterfall.svg
```

---

## P9：minimal source certificate controller

### 触发条件

```text
P7 generator pass or P8 certificate weak pass.
```

### 目标

形成最小 legal controller。controller 不得使用 outcome at commit，不得使用 dataset_name，不得使用 validation/test。

### Controller form

$$
Accept(a)=1
\iff
LCB(V_{cert}(a))>0
\land
UCB(LongRisk_{cert}(a))\le \tau_r
\land
LCB(Support(a))\ge \tau_s
\land
Cost(a)\le C_{max}.
$$

### 必须记录

```text
controller_id
generator_id
certificate_id
feature_count
thresholds
calibration_split_id
heldout_split_id
dataset_name_used
uses_outcome_at_commit
uses_future_step
accepted_count_cal
accepted_count_heldout
precision_Yrobust_cal
precision_Yrobust_heldout
h20_weak_CP_heldout
h20_V_ctrl_lcb_heldout
h240_long_risk_heldout
coverage_heldout
support_balance_pass
precision_lcb
longrisk_ucb
feature_cost_q90_ms
```

### 判断标准

P9 pass：

```text
dataset_name_used = 0
uses_outcome_at_commit = 0
uses_future_step = 0
coverage_heldout >= 0.03
h20_weak_CP_heldout >= 0.60
h20_V_ctrl_lcb_heldout > 0
h240_long_risk_heldout <= 0.10
support_balance_pass = 1
precision_lcb >= 0.50
longrisk_ucb <= 0.15
```

P9 fail：

```text
No legal controller can select robust source frontier.
Route = R9-LegalSourceCertificateControllerFail.
```

### 可视化

```text
p9_controller_frontier.svg
p9_calibration_to_heldout_drift.svg
p9_controller_support_balance.svg
p9_controller_value_risk_coverage_pareto.svg
```

---

## P10：selected source online runtime

### 触发条件

```text
P9 pass.
```

### 目标

测 selected controller + selected generator + selected payload apply 的真实 online runtime。不得用 no-payload smoke 或 oracle-mask diagnostic。

### 必须记录

```text
runtime_candidate_id
controller_id
generator_id
certificate_id
step_count
active_step_count
accepted_action_count
zero_candidate_controller_kernel_count
controller_launches_per_active_step_q90
feature_compute_time_ms_q90
certificate_compute_time_ms_q90
score_accept_time_ms_q90
payload_generate_time_ms_q90
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

P10 pass：

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
zero_candidate_controller_kernel_count = 0
controller_launches_per_active_step_q90 <= 2
payload_apply_error_linf_max <= 1e-6
audit_outside_timed_path = 1
```

### 可视化

```text
p10_selected_runtime_waterfall.svg
p10_step_ratio_distribution.svg
p10_payload_apply_time_hist.svg
p10_controller_launches_per_step.svg
```

---

## P11：Base-Acc Sentinel continuation

### 目标

继续监控 LQ base 是否健康，并与 MLP baselines 比较。但该结果不进入 controller/generator/certificate decision。

### 范围

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0..9
models = LQ-t2-h256, MatchedMLP, AdamWStrongLRGridMLP, QuadraticFeatureMLP
```

### 必须记录

```text
dataset
seed
model_id
train_acc
val_acc
test_acc
NLL
ECE
CEp99
margin_p10
train_time_ms
step_time_q90
memory_peak_mb
base_acc_used_for_controller
hyperparams_fixed_before_run
dataset_specific_tuning
```

### 判断标准

P11 pass：

```text
sentinel_complete = 1
base_acc_used_for_controller = 0
LQ_catastrophic_fail = 0
same_seed_schedule = 1
same_budget = 1
dataset_specific_tuning = 0
```

Catastrophic fail：

```text
mean_test_acc_LQ < mean_test_acc_MLP - 0.05
or any dataset LQ collapse unexplained.
```

### 可视化

```text
p11_base_acc_by_dataset_seed.svg
p11_lq_vs_mlp_acc_gap.svg
p11_lq_vs_strong_mlp_gap.svg
p11_ece_nll_comparison.svg
```

---

## P12：system integration gate

### 触发条件

```text
P5 pass and P9 pass and P10 pass.
```

### 必须记录

```text
system_candidate_id
no_transform_replay_pass
source_oracle_revalidation_pass
generator_preservation_pass
certificate_effect_valid_pass
source_controller_pass
selected_runtime_pass
base_acc_sentinel_pass
manual_forward
manual_backward
manual_adamw_update
uses_loss_backward
uses_teacher
uses_loss_modification
uses_dataset_name_for_controller
uses_validation_or_test
uses_future_outcome_for_features
uses_outcome_at_commit
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
no_transform_replay_pass = 1
source_controller_pass = 1
selected_runtime_pass = 1
manual_forward/manual_backward/manual_adamw_update = 1/1/1
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
uses_validation_or_test = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
official_eligible = 1
system_legal_controller_pass = 1
```

### 可视化

```text
p12_system_gate_ladder.svg
p12_contract_audit_matrix.svg
p12_success_failure_route.svg
```

---

## P13：conditional LDO / LSO and paired replay boundary

### 触发条件

```text
P12 system_legal_controller_pass = 1.
```

### 目标

系统过线后才开始判断 functional causal advantage。

### 必须记录

```text
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
RealFunctional_vs_AdamWParallel
RealFunctional_vs_bestLR
RealFunctional_vs_NoOp
RealFunctional_vs_Random
RealFunctional_vs_ShuffledPayload
CEp99_delta
margin_p10_delta
NLL_delta
ECE_delta
robust_value_lcb
long_risk_ucb
```

### 判断标准

P13 pass：

```text
LDO macro pass = 1
LSO macro pass = 1
RealFunctional beats AdamWParallel = 1
RealFunctional beats bestLR = 1
ShuffledPayload control fails = 1
NoOp control fails = 1
```

### 可视化

```text
p13_leave_dataset_out_metrics.svg
p13_leave_stratum_out_metrics.svg
p13_paired_replay_effect_size.svg
p13_real_vs_controls_pareto.svg
```

---

## P14：conditional short/full/sample-efficiency/continual validation

### 触发条件

```text
P13 paired replay pass = 1.
```

### 目标

只有在 system + causal replay 过线后，才进入 downstream task-level validation。

### 必须记录

```text
dataset
seed
model_id
controller_id
short_run_acc
full_run_acc
val_loss_auc_step
val_loss_auc_time
time_to_target
steps_to_target
ECE
NLL
CEp99
margin_p10
robustness_score
continual_task_order
retained_accuracy
forgetting
backward_transfer
forward_transfer
old_task_CEp99_drift
old_task_margin_drift
```

### 判断标准

P14 pass：

```text
LQ+functional beats matched MLP on macro test acc or sample efficiency；
beats AdamWStrongLRGridMLP on at least one non-accuracy axis；
no catastrophic regression on any dataset；
continual forgetting improvement if continual block is enabled；
no dataset-specific tuning。
```

### 可视化

```text
p14_short_full_acc_curves.svg
p14_val_loss_auc_step_time.svg
p14_time_to_target.svg
p14_continual_forgetting_matrix.svg
p14_calibration_robustness_comparison.svg
```

---

# 9. Route decision table

```text
R0-v9460BoundaryNotReproduced:
  P0 fail.

R1a-ReplayKeyLedgerIncomplete:
  P1 fail due missing optimizer/RNG/dataloader/config hash.

R2a-BatchSequenceDrift:
  first divergence from batch sequence mismatch.

R2b-RNGStateDrift:
  first divergence from RNG mismatch.

R2c-OptimizerStateDrift:
  first divergence from optimizer state mismatch.

R2d-UpdateOrderDrift:
  first divergence from AdamW/payload application order mismatch.

R2e-BranchDispatchDrift:
  branch name / clone id changes execution path.

R2f-PayloadApplyOrderDrift:
  payload same but applied at different point in step.

R2g-MetricComputationDrift:
  state hash same but metric/label differs.

R3-BranchNamedRunnerSemanticsLeak:
  canonical runner audit shows branch metadata affects execution.

R4-NoTransformReplayStillInconsistent:
  canonical runner implemented but side-by-side no-transform still fails.

R5-NoTransformReplaySemanticsUnclosed:
  P5 official no-transform equivalence gate fails.

R6a-OldOutcomeTableQuarantineRequired:
  P5 pass but old v9350 source outcome table not reproduced by repaired runner.

R6b-RepairedRunnerAP0SourceFrontierAbsent:
  repaired runner full/source panel recompute finds no robust source frontier.

R7-GeneratorPreservationFailAfterReplayClosure:
  no-transform pass, source frontier pass, but nontrivial generator destroys value.

R8-CertificateEffectFailAfterReplayClosure:
  generator has survivor but certificate cannot identify it.

R9-LegalSourceCertificateControllerFail:
  certificate feature exists but no heldout-valid legal controller.

R10-SelectedRuntimeFail:
  controller passes but online selected runtime exceeds envelope.

R11-SystemPassPairedReplayBlocked:
  system passes but paired replay not yet measured.

R12-PairedReplayCausalFail:
  system passes but RealFunctional fails vs controls.

R13-ShortFullFunctionalFail:
  paired replay passes but downstream short/full validation fails.

R14-v9470SystemReady:
  no-transform, source/generator/certificate/controller/runtime all pass.
```

---

# 10. 加速执行要求

## 10.1 分层 preflight kill-switch

每个大规模阶段前必须有小规模 preflight：

```text
1 action -> 3 actions -> 16 actions -> 64 actions -> full panel
```

如果小规模 fail，禁止扩大规模。

## 10.2 每阶段必须实时输出 gate summary

每个 P 阶段结束立即写：

```text
stage_status.json
stage_gate_summary.csv
failure_trace.csv
route_candidate.json
```

## 10.3 并行但不抢跑 official

允许并行跑 diagnostic lane，但 official route 只看已解锁阶段。

```text
Base-Acc Sentinel 可并行；
old-vs-new drift audit 可并行；
generator/certificate sandbox 可并行；
controller/runtime 不可在 P5 前 official。
```

## 10.4 重复使用 repaired runner

一旦 P5 pass，后续所有 outcome materializer 必须使用同一个：

```text
canonical_runner_version
branch_config_hash
horizon_config_hash
label_config_hash
update_order_id
```

任何版本变化都必须触发 no-transform re-audit。

---

# 11. 最终成功标准

v9.4.7 strict success 不是 full functional success，而是 replay semantics closure：

```text
source_identity_ledger_pass = 1
no_transform_payload_equivalence_pass = 1
source_clone_replay_key_pass = 1
side_by_side_no_transform_replay_pass = 1
branch_state_hash_match_rate = 1.0
horizon_state_hash_match_rate = 1.0
metric_abs_diff_max <= 1e-6
label_match_rate = 1.0
negative_control_divergence_present = 1
ORC_D_K16_direct_new_rollout_revalidated_or_old_table_quarantined = 1
```

v9.4.7 extended success：

```text
P5 no-transform pass；
P6 ORC-D-K16 revalidated；
P7 at least one generator preservation survivor；
P8 at least one effect-valid certificate weak pass；
P9 source controller weak pass；
P10 selected runtime measured or properly gate-blocked by P9。
```

如果只完成 P5，也算本轮有效推进，因为它会让后续 generator/certificate 结论终于有可信 replay foundation。

---

# 12. 最终判断

v9.4.7 的本质不是“再调一个 source generator”，而是修复实验物理学：

$$
\boxed{
\text{如果 no-transform clone 不能复现 source outcome，就没有资格判断 transform、certificate 或 controller。}
}
$$

当前最危险的错误路线是继续新增 AP primitive 或调 certificate threshold。正确路线是先把 outcome runner 变成 deterministic、branch-name-invariant、seed/batch/optimizer-state 完全受控的 replay engine。

只有这个闭合后，项目才能重新回到真正的科学问题：

$$
\boxed{
\text{是否存在 legal 方式生成、识别并部署 horizon-robust functional source action？}
}
$$
