# DG-KAN v9.4.6 Source Identity / No-Transform Replay Closure 与 Robust Source Revalidation 完整实验计划

> 本计划基于 v9.4.5 `Robust Source Generator / Objective-Aligned Certificate / Parallel Closure` 的真实执行结果制定。  
> v9.4.5 的 terminal route 是：
>
> ```text
> route = R1-OracleSourceIdentityBug
> base_candidate = LQ-t2-h256
> success_v9450_strict_purekan_functional = False
> success_v9450_full_functional = False
> success_v9450_external_ready = False
> ```
>
> v9.4.6 不再继续调 AP0r/AP0t/AP0u/CERT8 的 threshold，也不继续新增 source generator 小变体。  
> 本轮的第一性目标是先回答一个更底层的问题：
>
> $$
> \boxed{\text{同一个 source action，在 no-transform replay 下，是否能复现它自己的 AP0 oracle outcome？}}
> $$
>
> 只有这个问题闭合后，后续 generator preservation、certificate validity、controller、runtime、paired replay 才有解释意义。

---

# 0. 执行摘要

v9.4.5 的真正进展不是 source generator 成功，也不是 certificate 成功，而是暴露了一个比 “generator destructive” 更早的 blocker：

```text
AP0w-NoTransformReplaySource 作为 no-transform / identity diagnostic，
也没有复现 v9.4.4 的 ORC-D-HorizonRobust-K16 oracle survivor outcome。
```

v9.4.4 的结论是：full AP0 universe 中存在一个小而强的 K16 horizon-robust oracle source survivor：

```text
best_oracle_selector = ORC-D-HorizonRobust
best_K = 16
h20 weak CP = 0.8125
h20 V_ctrl LCB = 0.13772944106165844
h240 long-risk = 0.0
```

v9.4.5 复现了这个 boundary，但当这些 oracle source actions 被送入 oracle-seeded generator preservation test 时，即使是 AP0w no-transform replay，也出现：

```text
AP0w h20 weak CP = 0.0
AP0w h20 V_ctrl LCB = -1.714831942008832
AP0w h240 long-risk = 0.625
AP0w source-positive lost rate = 0.8444444444444444
AP0w Damage median = -1.1904930071905255
```

这不是普通 generator failure。因为 no-transform 本应满足：

$$
\Delta\theta_{AP0w}(a)=\Delta\theta_{AP0}(a)
$$

并且在相同 source state、相同 branch、相同 horizon、相同 outcome materializer 下：

$$
O_{AP0w}(a,b,h)=O_{AP0}(a,b,h).
$$

如果 no-transform 都不能复现，那么继续讨论 AP0r/AP0t/AP0u 是否破坏 value、CERT8 是否有效、runtime 是否值得测，都会把一个 identity/replay/join inconsistency 当成科学结论。

因此 v9.4.6 的路线是：

```text
Source Identity Ledger
+ No-Transform Payload Equivalence
+ Side-by-Side Outcome Replay Equivalence
+ Oracle Objective Recompute
+ Only-Then Generator / Certificate / Controller
```

v9.4.6 的最低有效推进不是 system pass，而是把以下二选一判清：

```text
Case A:
  v9.4.4 ORC-D-K16 oracle survivor 是真实可 replay 的；
  AP0w failure 来自 v9.4.5 source identity / payload / outcome join bug；
  修复后再判断 generator/certificate。

Case B:
  v9.4.4 ORC-D-K16 oracle survivor 不能在 side-by-side replay 中复现；
  则 v9.4.4 的 oracle survivor 是 oracle objective / aggregation / join artifact；
  必须重做 source oracle definition。
```

---

# 1. v9.4.5 独立判断

## 1.1 这轮不是能力成功

v9.4.5 没有 strict PureKAN functional success，也没有 full functional success，更没有 external-ready success。P8 source controller、P9 selected runtime、P11 system controller、P12 leave-out、P13 paired replay、P14 short/full training 全部没有打开。

关键 boundary 是：

```text
source_controller_pass = 0
selected_runtime_pass = 0
system_legal_controller_pass = 0
official_eligible = 0
primary_blocker = no_transform_oracle_seeded_failed
```

因此不能把以下内容写成 official success：

```text
oracle survivor diagnostic；
AP0r/AP0w generator smoke；
preflight pass；
CERT8 diagnostic；
Base-Acc Sentinel。
```

## 1.2 这轮有进展，但进展类型是“根因提前”

v9.4.4 的 route 是 `R3-GeneratorDestructive`，看起来像现有 AP0b-AP0q generator 会破坏 oracle survivor。v9.4.5 进一步加入 AP0w no-transform sanity diagnostic，结果 no-transform 也失败。

这把 blocker 从：

```text
GeneratorDestructive
```

提前到：

```text
SourceIdentity / Replay / OutcomeJoin inconsistency
```

这很重要。因为如果 no-transform 的等价性不成立，AP0r/AP0v 的失败不能被解释为 transform 破坏 value。也就是说，v9.4.5 不是证明新 generator 更差，而是证明当前 generator 评估 protocol 还不能解释。

## 1.3 robust source objective 仍然是必要方向

v9.4.5 的 P1 继续确认：weak CP 不能等同于 robust source objective。

记录值：

```text
action_count = 2876
WeakCP_h20_rate = 0.10674547983310154
V_ctrl_h20_LCB = -0.7504334275695862
LongRisk_h240_rate = 0.9022948539638387
P_Vctrl_positive_given_WeakCP_h20 = 1.0
P_LongRisk_h240_given_WeakCP_h20 = 0.8013029315960912
P_Yrobust_given_WeakCP_h20 = 0.1986970684039088
Y_robust_base_rate = 0.02121001390820584
```

这说明 h20 weak CP 与 h20 value 有关系，但它无法排除 h240 long-risk。以后不能再只追：

$$
WeakCP_{h20}(a)=1.
$$

robust source 目标必须显式写成：

$$
Y_{robust}(a)=1
\iff
WeakCP_{h20}(a)=1
\land
LCB(V_{ctrl,h20}(a))>0
\land
LongRisk_{h240}(a)\le \tau_{lr}
\land
Support(a)\ge \tau_s.
$$

## 1.4 legal selector 仍然看不见 oracle survivor

P2 oracle survivor anatomy 显示：

```text
oracle_action_count = 16
joined_with_legal_feature_count = 16
miss_reason_assigned_fraction = 1.0
dominant_miss_reason = M8-outcome-only-pattern
dominant_miss_reason_count = 6
```

这说明 oracle survivor 可以 join 到 legal feature table，但当前 legal feature / rank 找不到它。`M8-outcome-only-pattern` 的意思不是可以把 outcome 用作 selector，而是提醒我们：这些 survivor 的可见性主要来自 outcome oracle，commit-time observable 不足。

## 1.5 preflight 不是 blocker

P3 generator preflight 全过：

```text
preflight_row_count = 30
preflight_pass_count = 30
preflight_all_pass = 1
unresolved_exception_count = 0
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
branch_horizon_completion_rate = 1.0
```

这意味着 row sink、payload write、certificate write、action apply、branch/horizon row completion 这些基础链路不是 v9.4.5 的直接 blocker。问题更像是：写出来、跑出来了，但跑的不是同一个 source/action/outcome universe。

## 1.6 AP0r/AP0u/AP0t 的失败目前不能作为最终科学结论

v9.4.5 的 generator 结果都很弱：

```text
P4 oracle-seeded best AP0r:
  h20 weak CP = 0.125
  h20 V_ctrl LCB = -0.8833607173563967
  h240 long-risk = 0.75

P5 legal/representative best S1/AP0u:
  h20 weak CP = 0.0
  h20 V_ctrl LCB = -0.2707542846308277
  h240 long-risk = 0.6875

P6 direct best AP0t:
  h20 weak CP = 0.125
  h20 V_ctrl LCB = -0.7052174296088407
  h240 long-risk = 0.875
```

但这些失败必须在 no-transform equivalence 通过后才有完整解释。如果 no-transform 本身失败，AP0r/AP0u/AP0t 的 outcome 可能被同一个 identity/join/replay bug 污染。

## 1.7 certificate 仍没有 effect validity

P7 最好 certificate 是 `CERT8-AdamWConflictHorizonGuard`：

```text
AUC robust source = 0.549163179916318
AUC long-risk h240 = 0.4924701402111823
best_P_Yrobust_given_cert_pass = 0.0
best_P_longrisk_given_cert_pass = 0.0
monotone_sign_pass = 0
certificate_effect_valid_pass = 0
```

这说明 CERT6-CERT10 目前不是 robust source / long-risk 的 sufficient statistic。即使 no-transform identity 被修好，也不能直接拿 CERT8 进 controller；它最多是一个 diagnostic feature。

## 1.8 Base-Acc Sentinel 有训练，但不是 functional success

v9.4.5 的 Base-Acc Sentinel 已训练并完成：

```text
sentinel_row_count = 120
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
model_count = 4
sentinel_complete = 1
mean_test_acc_LQ = 0.6537760416666667
mean_test_acc_MLP = 0.562890625
mean_test_acc_QuadraticFeatureMLP = 0.40234375
mean_test_acc_AdamWStrongLRGridMLP = 0.628515625
LQ_minus_MLP_mean_test_acc = 0.09088541666666672
LQ_minus_AdamWStrongLRGridMLP_mean_test_acc = 0.025260416666666674
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
base_acc_sentinel_pass = 1
```

解释：LQ-t2-h256 base 在 fixed sentinel 设置下没有 catastrophic fail，并且平均 test acc 比 MatchedMLP 高约 9.09 个百分点，比 AdamWStrongLRGridMLP 高约 2.53 个百分点。但这个 sentinel 没有用于 selector、generator、certificate 或 controller。因此它只能说明 base 还活着，不能说明 functional DG-KAN 已经赢 MLP。

---

# 2. 当前真正卡在哪里

## 2.1 当前 primary blocker 不是 generator threshold，而是 identity equivalence

现在必须先闭合：

$$
\boxed{NoTransformReplay(a) \equiv AP0ReferenceReplay(a)}
$$

具体来说，对于每个 oracle source action $a$，必须有：

$$
source\_action\_id_{AP0w}=source\_action\_id_{AP0},
$$

$$
payload\_hash_{AP0w}=payload\_hash_{AP0},
$$

$$
state\_before\_hash_{AP0w}=state\_before\_hash_{AP0},
$$

$$
branch\_config\_hash_{AP0w}=branch\_config\_hash_{AP0},
$$

$$
label\_config\_hash_{AP0w}=label\_config\_hash_{AP0},
$$

并且：

$$
\max_{b,h,m}|O_{AP0w}(a,b,h,m)-O_{AP0}(a,b,h,m)| \le \epsilon_m.
$$

如果这些不成立，任何 generator / certificate 分析都不可靠。

## 2.2 full AP0 oracle survivor 仍不能被判死

v9.4.4 的 ORC-D-K16 survivor 仍然存在于 diagnostic table 中；v9.4.5 没有证明它不存在。v9.4.5 证明的是：把这些 survivor 送入当前 AP0w no-transform replay path 后，outcome 不一致。

因此不能说：

```text
AP0 full universe 没好 source。
```

更准确是：

```text
AP0 full universe 的 oracle survivor 目前无法被 no-transform replay path 复现。
```

这可能是 identity bug，也可能是 v9.4.4 oracle objective / row aggregation bug。v9.4.6 必须把这两种可能分开。

## 2.3 继续加 AP0r/AP0x 小变体是错误方向

在 no-transform equivalence 失败前，以下都属于小修小补：

```text
调 AP0r trust-region 半径；
调 AP0t tail projection；
调 AP0u AdamW residual blend；
调 CERT8 threshold；
继续筛 state_NLL_proxy / PayloadLinf；
把 AP0w failure 归因成 generator destructive。
```

这些动作会绕过最基本的 sanity contract。

## 2.4 进度慢的原因

你的“慢、没进度”的感觉是合理的。最近几轮每次都修一个 implementation blocker：

```text
v9.4.1: generator materialized，但 outcome rows = 0；
v9.4.2: outcome rows 修到 7020，但 panel/source value-poor；
v9.4.3: multi-panel oracle 接近但不 official；
v9.4.4: exhaustive oracle 找到 K16 survivor，但 generator destructive；
v9.4.5: no-transform replay 也失败，说明 identity/equivalence 未闭合。
```

这条链科学上有推进，但执行上太串行。v9.4.6 必须改成并行 preflight：先用 1-action、3-action、16-action 分层锁死 identity，不要再等完整大 runner 才发现 no-transform 失败。

---

# 3. v9.4.6 总体目标

v9.4.6 的总体目标是：

$$
\boxed{\text{闭合 source identity / no-transform replay / outcome join consistency，然后重新验证 robust source frontier。}}
$$

v9.4.6 不是为了调榜，也不是为了让某个 AP0r 直接过线。本轮要回答：

```text
Q1. v9.4.4 ORC-D-K16 oracle survivor 的 action identity 是否唯一、可追踪、可 replay？
Q2. AP0w no-transform clone 是否与 AP0 source payload 完全一致？
Q3. AP0w no-transform outcome 是否与 AP0 source outcome 在 branch/horizon/label 上一致？
Q4. 如果不一致，差异来自 source action ID、payload hash、state hash、branch/horizon config、label config、aggregation，还是 nondeterminism？
Q5. 如果一致，AP0r/AP0v generator 是否真实 destructive？
Q6. 是否存在 value-preserving robust source generator？
Q7. 是否存在 effect-valid certificate？
Q8. 是否可以打开 minimal source controller 与 selected runtime？
```

v9.4.6 的强 pass 不是 full functional success，而是：

```text
no_transform_equivalence_pass = 1
oracle_survivor_replay_pass = 1
robust_source_objective_recomputed = 1
generator_preservation_test_interpretable = 1
```

只有在这些 pass 后，才允许进入：

```text
generator_pass
certificate_effect_valid_pass
source_controller_pass
selected_runtime_pass
system_legal_controller_pass
```

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
official selector/controller 不使用 dataset_name 分支
不使用 validation/test metric at commit time
不使用 future outcome feature
不使用 outcome-at-commit feature
不把 diagnostic oracle 用作 official controller
不把 no-transform diagnostic 写成 official system pass
不把 Base-Acc Sentinel 用于 selector/generator/controller
```

新增硬约束：

```text
1. AP0w no-transform equivalence 未通过前，不得推进 AP0r-AP0v generator 结论。
2. AP0w no-transform equivalence 未通过前，不得评估 certificate/controller/runtime official pass。
3. ORC-D-K16 oracle survivor 必须能被 source ledger 逐行追踪到 source action/payload/outcome。
4. 每个 source action 必须记录 source_state_before_hash、optimizer_state_hash、payload_hash、branch_config_hash、horizon_config_hash、label_config_hash。
5. AP0 source replay 与 AP0w clone replay 必须 side-by-side 同 runner、同 seed、同 branch/horizon config 执行。
6. 任何 row aggregation 从 horizon row 到 action label 的过程必须单独落盘。
```

允许做：

```text
identity / payload / state / branch / label audit；
side-by-side no-transform replay；
oracle objective recompute；
source/action row ledger；
small-N preflight；
parallel Base-Acc Sentinel；
conditional generator/certificate test。
```

---

# 5. 核心假设

## H1：v9.4.5 AP0w failure 是 source identity / replay / outcome join inconsistency，而不是 science failure

H1 成立标准：

```text
在 v9.4.6 identity audit 中发现以下任一项：
source_action_id mismatch；
payload_hash mismatch；
state_before_hash mismatch；
optimizer_state_hash mismatch；
branch_config_hash mismatch；
horizon_config_hash mismatch；
label_config_hash mismatch；
row aggregation mismatch；
AP0w action_id collision；
source outcome table join key mismatch。
```

H1 失败标准：

```text
所有 identity/config/hash/aggregation 均匹配，但 AP0w side-by-side replay 仍不等价。
```

若 H1 失败，应进入 nondeterminism / replay semantics audit。

## H2：v9.4.4 ORC-D-K16 oracle survivor 是可 replay 的真实 AP0 source frontier

H2 成立标准：

```text
ORC-D-K16 source action IDs 全部可 join；
AP0 source replay reproduces v9.4.4 labels；
AP0w no-transform replay reproduces AP0 source replay；
recomputed K16 metrics equal v9.4.4 within tolerance：
  h20 weak CP = 0.8125 ± epsilon
  h20 V_ctrl LCB > 0
  h240 long-risk = 0.0 ± epsilon
```

H2 失败标准：

```text
AP0 source replay itself cannot reproduce v9.4.4 ORC-D-K16 metrics。
```

若 H2 失败，则 v9.4.4 oracle survivor 是 oracle objective / aggregation / join artifact，必须重建 oracle source definition。

## H3：如果 no-transform equivalence pass，而 AP0r/AP0v 仍 fail，则 generator destructive 才成立

H3 成立标准：

```text
AP0w no-transform equivalence pass = 1
AP0r/AP0v preservation pass = 0
source_positive_lost_rate >= 0.50
Damage_median <= -0.20
h240 long-risk rises above source baseline
```

H3 失败标准：

```text
AP0w pass 后至少一个 generator 保留 K16 value/horizon safety。
```

## H4：effect-valid certificate 必须预测 robust source，而不是 weak CP

H4 成立标准：

```text
AUC_Yrobust >= 0.75
AUC_longrisk >= 0.75
P(Yrobust | cert_pass) >= 0.60
P(LongRisk_h240 | cert_pass) <= 0.10
monotone_sign_pass = 1
leave-seed drop <= 0.10
```

H4 失败标准：

```text
certificate 只对 weak CP 有微弱 lift，或不能降低 long-risk。
```

## H5：Base-Acc Sentinel 只用于 base 健康监控，不用于 controller

H5 成立标准：

```text
base_acc_sentinel_complete = 1
base_acc_used_for_controller = 0
LQ_catastrophic_fail = 0
```

H5 失败标准：

```text
Base-Acc Sentinel 被用于 selector/generator/controller，或 LQ catastrophic fail。
```

---

# 6. 数据合同

## 6.1 Source action ledger contract

每个 source action 必须有唯一 ledger row：

```text
source_action_id
source_candidate_id
source_event_id
source_global_row_id
source_dataset
source_seed
source_step
source_batch_id
source_family_id
source_bucket_id
source_horizon
source_primitive_id
source_payload_hash
source_payload_shard_id
source_payload_offset
source_state_before_hash
source_optimizer_state_hash
source_model_param_hash
source_batch_hash
source_label_hash
source_branch_config_hash
source_horizon_config_hash
source_label_config_hash
source_outcome_table_hash
source_oracle_selector_id
source_oracle_rank
source_oracle_panel_id
source_oracle_objective_version
source_outcome_aggregation_version
```

Pass：

```text
source_action_ledger_complete = 1
source_action_id_duplicate_count = 0
source_payload_hash_missing_count = 0
source_state_before_hash_missing_count = 0
source_branch_config_hash_missing_count = 0
source_label_config_hash_missing_count = 0
ORC_D_K16_join_success_count = 16
ORC_D_K16_join_missing_count = 0
```

## 6.2 No-transform clone contract

对每个 source action $a$，AP0w clone row 必须记录：

```text
clone_action_id
clone_parent_source_action_id
clone_parent_source_payload_hash
clone_payload_hash
clone_payload_linf_to_source
clone_payload_relative_error_to_source
clone_payload_cosine_to_source
clone_state_before_hash
clone_optimizer_state_hash
clone_branch_config_hash
clone_horizon_config_hash
clone_label_config_hash
clone_no_transform_declared
clone_no_transform_verified
```

Pass：

```text
clone_parent_join_success = 1
clone_payload_hash == source_payload_hash
clone_state_before_hash == source_state_before_hash
clone_optimizer_state_hash == source_optimizer_state_hash
clone_branch_config_hash == source_branch_config_hash
clone_horizon_config_hash == source_horizon_config_hash
clone_label_config_hash == source_label_config_hash
clone_payload_linf_to_source <= 1e-8
clone_payload_relative_error_to_source <= 1e-7
clone_payload_cosine_to_source >= 0.99999999
clone_no_transform_verified = 1
```

## 6.3 Side-by-side outcome replay contract

每个 pair $(source, clone)$ 必须在同一 runner 中执行：

```text
pair_id
source_action_id
clone_action_id
branch
horizon
metric_name
source_metric_value
clone_metric_value
metric_abs_diff
metric_rel_diff
source_label
clone_label
label_match
source_branch_state_hash
clone_branch_state_hash
source_horizon_state_hash
clone_horizon_state_hash
```

Pass：

```text
paired_branch_horizon_row_count_expected = action_count * branch_count * horizon_count
paired_branch_horizon_row_count_actual = expected
metric_abs_diff_max <= metric_specific_tolerance
label_match_rate = 1.0
branch_state_hash_match_rate = 1.0
horizon_state_hash_match_rate = 1.0
```

## 6.4 Robust source objective contract

Robust source label：

$$
Y_{robust}(a)=1
\iff
WeakCP_{h20}(a)=1
\land
LCB(V_{ctrl,h20}(a))>0
\land
LongRisk_{h240}(a)\le 0.10
\land
SupportBalance(a)=1.
$$

必须记录：

```text
WeakCP_h20
StrongCP_h20
V_ctrl_h20_mean
V_ctrl_h20_LCB
V_ctrl_h80_LCB
V_ctrl_h240_LCB
LongRisk_h240
HorizonRobustCP
SupportBalance
Y_robust
source_objective_version
aggregation_version
```

Pass：

```text
Y_robust_recomputed = 1
source_objective_version_fixed = 1
aggregation_version_fixed = 1
oracle_recompute_matches_source_table = 1
```

---

# 7. Failure taxonomy

本轮不允许只写 `no_transform_failed`，必须归因到下列 failure classes。

```text
I1-source-action-id-mismatch
I2-source-candidate-id-mismatch
I3-payload-hash-mismatch
I4-payload-shard-offset-mismatch
I5-state-before-hash-mismatch
I6-optimizer-state-hash-mismatch
I7-branch-config-hash-mismatch
I8-horizon-config-hash-mismatch
I9-label-config-hash-mismatch
I10-control-branch-semantics-mismatch
I11-oracle-row-to-action-aggregation-mismatch
I12-source-outcome-table-join-key-mismatch
I13-clone-action-id-collision
I14-no-transform-generator-mutated-payload
I15-replay-nondeterminism
I16-metric-definition-drift
I17-robust-objective-version-drift
I18-outcome-materializer-runner-drift
```

每个 failure row 必须记录：

```text
failure_class
source_action_id
clone_action_id
branch
horizon
first_failed_contract
expected_value
observed_value
abs_diff
rel_diff
root_cause_candidate
repair_required
blocks_generator_analysis
```

---

# 8. 实验阶段

---

## P0：v9.4.5 boundary reproduction

### 目标

确认 v9.4.5 结果稳定复现，不在错误 artifact 上继续。

### 必须记录

```text
route_v9450
source_route_v9440
objective_mismatch_pass
Y_robust_base_rate
P_LongRisk_h240_given_WeakCP_h20
oracle_survivor_anatomy_pass
dominant_oracle_legal_miss_reason
preflight_all_pass
oracle_seeded_no_transform_pass
oracle_seeded_transform_preservation_pass
oracle_seeded_best_generator
oracle_seeded_best_h20_weak_CP
oracle_seeded_best_h20_V_ctrl_LCB
oracle_seeded_best_h240_longrisk
legal_representative_official_candidate_pass
direct_generator_pass
certificate_effect_valid_pass
base_acc_sentinel_pass
system_legal_controller_pass
primary_blocker
```

### 判断标准

P0 pass：

```text
route_v9450 = R1-OracleSourceIdentityBug
preflight_all_pass = 1
oracle_seeded_no_transform_pass = 0
system_legal_controller_pass = 0
fake/proxy/offload = 0/0/0
```

### 可视化

```text
p0_v9450_route_ladder.svg
p0_v9440_to_v9450_blocker_shift.svg
p0_no_transform_vs_transform_metrics.svg
```

---

## P1：ORC-D-K16 source identity ledger build

### 目标

把 v9.4.4 ORC-D-HorizonRobust-K16 的 16 个 oracle source actions 从 outcome oracle table 追踪到 AP0 payload、state、branch/horizon outcome rows。P1 不 replay，只建账。

### 假设

H2 的前半部分：ORC-D-K16 source action identity 可以唯一追踪。

### 实现

新增 runner：

```text
experiments/run_v9460_source_identity_no_transform_replay_closure.py
```

P1 从以下 artifacts 读取：

```text
v9440 p2_exhaustive_ap0_source_frontier.csv
v9350 full_control_outcome_table_v9350.csv
v9330 action_payload_manifest_v9330.json
v9330 action_payload_shards_v9330/
v9450 oracle_seeded_generator_trace_v9450.csv
```

构造：

```text
source_identity_ledger_v9460.csv
orc_d_k16_identity_trace_v9460.csv
source_payload_state_config_trace_v9460.csv
```

### 必须记录

```text
source_action_id
source_candidate_id
source_event_id
source_payload_hash
source_payload_shard_id
source_payload_offset
source_state_before_hash
source_optimizer_state_hash
source_model_param_hash
source_batch_hash
source_branch_config_hash
source_horizon_config_hash
source_label_config_hash
source_outcome_row_count
source_outcome_branch_count
source_outcome_horizon_count
source_oracle_selector_id
source_oracle_rank
source_Y_robust
source_WeakCP_h20
source_V_ctrl_h20_LCB
source_LongRisk_h240
join_status
missing_reason
```

### 判断标准

P1 pass：

```text
ORC_D_K16_join_success_count = 16
ORC_D_K16_join_missing_count = 0
source_payload_hash_missing_count = 0
source_state_hash_missing_count = 0
source_config_hash_missing_count = 0
duplicate_source_action_id_count = 0
source_outcome_branch_horizon_complete = 1
```

P1 fail route：

```text
R1a-OracleSourceLedgerIncomplete
```

### 可视化

```text
p1_oracle_source_identity_sankey.svg
p1_source_payload_hash_table.svg
p1_source_state_config_hash_matrix.svg
p1_oracle_action_join_missing_heatmap.svg
```

---

## P2：AP0 source outcome recompute

### 目标

不经过 AP0w clone，直接用 v9.4.6 runner 对 ORC-D-K16 source AP0 actions 重跑 AP0 source outcome，确认 v9.4.4 oracle survivor 本身可复现。

### 假设

H2：AP0 source replay 能复现 v9.4.4 ORC-D-K16 metrics。

### 必须记录

```text
source_action_id
branch
horizon
metric_name
v9440_metric_value
v9460_source_replay_metric_value
metric_abs_diff
metric_rel_diff
v9440_label
v9460_source_replay_label
label_match
source_replay_state_hash
source_replay_branch_config_hash
source_replay_label_config_hash
```

Aggregate：

```text
source_replay_action_count
source_replay_branch_horizon_expected
source_replay_branch_horizon_actual
source_replay_metric_abs_diff_max
source_replay_label_match_rate
recomputed_h20_weak_CP
recomputed_h20_V_ctrl_LCB
recomputed_h240_longrisk
recomputed_Y_robust_count
```

### 判断标准

P2 pass：

```text
source_replay_branch_horizon_actual = expected
source_replay_label_match_rate = 1.0
recomputed_h20_weak_CP >= 0.75
recomputed_h20_V_ctrl_LCB > 0
recomputed_h240_longrisk <= 0.10
recomputed_Y_robust_count >= 12 of 16
```

P2 fail route：

```text
R1b-OracleObjectiveOrSourceOutcomeReplayBug
```

### 可视化

```text
p2_v9440_vs_v9460_source_replay_metric_scatter.svg
p2_source_replay_label_confusion.svg
p2_recomputed_oracle_frontier_table.svg
```

---

## P3：AP0w no-transform payload equivalence preflight

### 目标

在不跑完整 outcome 前，先证明 AP0w clone payload 与 source AP0 payload 完全一致。P3 是 cheap preflight，必须在 1-action、3-action、16-action 三个规模上运行。

### 假设

H1：v9.4.5 no-transform failure 可能来自 payload clone / parent join / state config mismatch。

### 规模

```text
P3a: 1 action，选择 ORC-D rank 1。
P3b: 3 actions，选择 ORC-D rank 1/8/16。
P3c: 16 actions，完整 ORC-D-K16。
P3d: 16 negative controls，随机非 robust AP0 actions。
```

### 必须记录

```text
clone_action_id
clone_parent_source_action_id
source_payload_hash
clone_payload_hash
source_payload_norm
clone_payload_norm
payload_linf_diff
payload_l2_diff
payload_relative_error
payload_cosine
source_state_before_hash
clone_state_before_hash
source_optimizer_state_hash
clone_optimizer_state_hash
source_branch_config_hash
clone_branch_config_hash
source_horizon_config_hash
clone_horizon_config_hash
source_label_config_hash
clone_label_config_hash
no_transform_verified
```

### 判断标准

P3 pass：

```text
all clone_parent_source_action_id joined
payload_hash_match_rate = 1.0
payload_linf_diff_max <= 1e-8
payload_relative_error_max <= 1e-7
payload_cosine_min >= 0.99999999
state_before_hash_match_rate = 1.0
optimizer_state_hash_match_rate = 1.0
branch_config_hash_match_rate = 1.0
horizon_config_hash_match_rate = 1.0
label_config_hash_match_rate = 1.0
```

P3 fail route：

```text
R1c-NoTransformPayloadOrStateCloneBug
```

### 可视化

```text
p3_payload_equivalence_hist.svg
p3_hash_match_matrix.svg
p3_clone_parent_join_graph.svg
```

---

## P4：AP0 source vs AP0w no-transform side-by-side outcome replay

### 目标

在同一 runner 中，对 AP0 source 和 AP0w clone 进行 side-by-side branch/horizon replay，确认 no-transform outcome 等价。

### 假设

如果 P2 和 P3 pass，则 AP0w outcome 应复现 source outcome。若不复现，问题在 replay semantics、branch/horizon runner、label/materializer 或 nondeterminism。

### 必须记录

```text
pair_id
source_action_id
clone_action_id
branch
horizon
metric_name
source_value
clone_value
metric_abs_diff
metric_rel_diff
source_safe_good
clone_safe_good
source_bad_event
clone_bad_event
source_null_event
clone_null_event
source_WeakCP
clone_WeakCP
source_LongRisk
clone_LongRisk
label_match
branch_state_hash_match
horizon_state_hash_match
```

Aggregate：

```text
paired_action_count
paired_branch_horizon_expected
paired_branch_horizon_actual
metric_abs_diff_max_by_metric
metric_rel_diff_max_by_metric
label_match_rate
WeakCP_match_rate
LongRisk_match_rate
Yrobust_match_rate
source_positive_lost_rate_no_transform
Damage_median_no_transform
```

### 判断标准

P4 pass：

```text
paired_branch_horizon_actual = expected
label_match_rate = 1.0
WeakCP_match_rate = 1.0
LongRisk_match_rate = 1.0
Yrobust_match_rate = 1.0
source_positive_lost_rate_no_transform = 0
abs(Damage_median_no_transform) <= 1e-6
AP0w_h20_weak_CP == source_h20_weak_CP
AP0w_h20_V_ctrl_LCB == source_h20_V_ctrl_LCB within tolerance
AP0w_h240_longrisk == source_h240_longrisk
```

P4 fail route：

```text
R1d-NoTransformOutcomeReplayBug
```

### 可视化

```text
p4_source_vs_clone_metric_scatter.svg
p4_source_vs_clone_label_confusion.svg
p4_no_transform_damage_distribution.svg
p4_failure_taxonomy_by_branch_horizon.svg
```

---

## P5：oracle objective and aggregation recompute

### 目标

独立重算 ORC-D-HorizonRobust objective，确认 K16 是 action-level robust survivor，而不是 row-level / horizon-level aggregation artifact。

### 假设

H2：ORC-D-K16 的 robust source objective 在 v9.4.6 recompute 后仍成立。

### 必须记录

```text
source_action_id
row_level_WeakCP_h20
row_level_V_ctrl_h20
row_level_LongRisk_h240
action_level_WeakCP_h20
action_level_V_ctrl_h20_LCB
action_level_LongRisk_h240
action_level_Yrobust
aggregation_rule
aggregation_version
oracle_rank_v9440
oracle_rank_v9460
rank_diff
selected_in_K16_v9440
selected_in_K16_v9460
```

### 判断标准

P5 pass：

```text
ORC_D_K16_overlap_rate >= 0.95
rank_diff_max <= 2 或 equivalent frontier preserved
recomputed_K16_h20_weak_CP >= 0.75
recomputed_K16_h20_V_ctrl_LCB > 0
recomputed_K16_h240_longrisk <= 0.10
aggregation_version_fixed = 1
```

P5 fail route：

```text
R1e-OracleAggregationDefinitionBug
```

### 可视化

```text
p5_oracle_rank_v9440_vs_v9460.svg
p5_row_to_action_aggregation_waterfall.svg
p5_robust_source_objective_phase_diagram.svg
```

---

## P6：generator preservation matrix after identity closure

### 前置条件

必须满足：

```text
P1 pass
P2 pass
P3 pass
P4 pass
P5 pass
```

否则 P6 不运行，只落盘 not_run。

### 目标

在 no-transform equivalence 闭合后，重新评估 AP0r-AP0v 是否真的 destructive，或者是否有 value-preserving generator。

### Generator candidates

```text
AP0r-ObjectiveSolvedLastEdgeTrustRegionSource
AP0s-ValuePreservingResidualProjectionSource
AP0t-TailProjectedHorizonGuardSource
AP0u-AdamWCompatibleResidualSource
AP0v-SupportBalancedLowRankSource
AP0w-NoTransformReplaySource reference
```

### Seed panels

```text
S0-ORC-D-K16-oracle-diagnostic-only
S1-legal-topK-state_NLL
S2-legal-topK-AdamWConflictLow
S3-representative-stratified-K64
S4-random-negative-K64
```

### 必须记录

```text
panel_id
generator_id
source_action_count
generated_action_count
branch_horizon_rows
h20_weak_CP
h20_V_ctrl_LCB
h80_weak_CP
h240_weak_CP
h240_longrisk
Yrobust_rate
source_positive_lost_rate
source_negative_fixed_rate
Damage_mean
Damage_median
Damage_lcb
payload_distortion_linf
payload_distortion_l2
payload_cosine_to_source
support_balance_pass
generator_preservation_pass
```

### 判断标准

P6 weak preservation pass：

```text
AP0w no-transform pass = 1
generator h20 weak CP >= 0.60
generator h20 V_ctrl_LCB > 0
generator h240 long-risk <= 0.10
source_positive_lost_rate <= 0.20
Damage_median >= -0.05
support_balance_pass = 1
```

P6 strong pass：

```text
Yrobust_rate >= 0.50
source_positive_lost_rate <= 0.10
horizon_robust_action_coverage >= 0.30
```

P6 fail route：

```text
R2-GeneratorDestructiveConfirmedAfterIdentityClosure
```

### 可视化

```text
p6_generator_preservation_pareto.svg
p6_source_to_generated_damage_matrix.svg
p6_payload_distortion_vs_value.svg
p6_horizon_risk_by_generator.svg
```

---

## P7：direct objective-solved robust source generator reset

### 前置条件

P7 只在以下情况下运行：

```text
P1-P5 pass
P6 fail
```

### 目标

如果 existing generator 真 destructive，则不再小修 AP0r/AP0v，而是实现 direct objective-solved generator。它必须显式优化 h20 value 与 h240 safety，而不是 heuristic payload builder。

### Generator objective

对每个 action source state $s_t$，生成 $\Delta\theta$：

$$
\Delta\theta^*=
\arg\min_{\Delta\theta}
\left[
\widehat{\Delta CE}_{h20}(\Delta\theta)
+
\lambda_{tail}\widehat{LongRisk}_{h240}(\Delta\theta)
+
\lambda_{dist}\|\Delta\theta-\Delta\theta_{AP0}\|_2^2
+
\lambda_{adamw}\max(0,-\cos(\Delta\theta,\Delta\theta_{AdamW}))
\right]
$$

subject to：

$$
\|\Delta\theta\|_2 \le r,
$$

$$
\widehat{MarginGain}_{tail,h240}(\Delta\theta) \ge 0,
$$

$$
Cost(\Delta\theta) \le C_{max}.
$$

### Candidate generators

```text
AP1a-LinearizedCETrustRegionSolver
AP1b-TailMarginConstrainedSolver
AP1c-AdamWCompatibleProjectedSolver
AP1d-LowRankEdgeConstrainedSolver
AP1e-SourcePreservingMinimalCorrectionSolver
AP1f-HorizonRobustTwoObjectiveSolver
```

### 必须记录

```text
generator_id
optimization_status
solver_iterations
objective_before
objective_after
predicted_CE_h20_delta
predicted_margin_tail_h240_delta
predicted_longrisk_h240
payload_norm
payload_cost_estimate
constraint_violation_count
action_apply_error_linf
h20_weak_CP
h20_V_ctrl_LCB
h240_longrisk
Yrobust_rate
```

### 判断标准

P7 pass：

```text
constraint_violation_count = 0
action_apply_error_linf <= 1e-8
h20_weak_CP >= 0.60
h20_V_ctrl_LCB > 0
h240_longrisk <= 0.10
Yrobust_rate >= 0.30
```

P7 fail route：

```text
R3-DirectRobustSourceGeneratorObjectiveFail
```

### 可视化

```text
p7_solver_objective_before_after.svg
p7_constraint_violation_table.svg
p7_predicted_vs_real_value.svg
p7_direct_generator_frontier.svg
```

---

## P8：effect-valid certificate after identity closure

### 目标

重新设计 certificate，但只在 identity closure 后运行。certificate 目标不是预测 weak CP，而是预测 $Y_{robust}$ 与 long-risk。

### Certificate candidates

```text
CERT11-NoTransformEquivalenceCert
CERT12-LinearizedCEValueCert
CERT13-HorizonTailMarginGuardCert
CERT14-AdamWCompatibilityCert
CERT15-PayloadDistortionBoundCert
CERT16-SupportBalanceReliabilityCert
CERT17-MinimalHybridRobustSourceCert
```

### 必须记录

```text
certificate_id
certificate_field_count
commit_time_available
payload_hash_bound
uses_dataset_name
uses_outcome_at_commit
AUC_Yrobust
AUC_LongRisk_h240
AUC_h20_weak_CP
AUC_h20_Vctrl_positive
P_Yrobust_given_cert_pass
P_Yrobust_given_cert_fail
P_LongRisk_given_cert_pass
P_LongRisk_given_cert_fail
Lift_Yrobust
Lift_LongRiskReduction
monotone_sign_pass
leave_seed_drop
certificate_cost_ms_q90
```

### 判断标准

P8 pass：

```text
uses_dataset_name = 0
uses_outcome_at_commit = 0
payload_hash_bound = 1
AUC_Yrobust >= 0.75
AUC_LongRisk_h240 >= 0.75
P_Yrobust_given_cert_pass >= 0.60
P_LongRisk_given_cert_pass <= 0.10
monotone_sign_pass = 1
leave_seed_drop <= 0.10
certificate_cost_ms_q90 <= 0.20
```

P8 fail route：

```text
R4-CertificateEffectValidityFail
```

### 可视化

```text
p8_certificate_roc_yrobust.svg
p8_certificate_roc_longrisk.svg
p8_cert_pass_fail_lift_bar.svg
p8_certificate_calibration_curve.svg
p8_certificate_cost_distribution.svg
```

---

## P9：minimal source certificate controller

### 前置条件

必须满足：

```text
P1-P5 pass
(P6 pass or P7 pass)
P8 pass
```

### 目标

构建最小 controller，不做大 feature factory。controller 只能使用少量 certificate primitives。

### Controller form

$$
Accept(a)=1
\iff
LCB(V_{h20}(a))>0
\land
UCB(LongRisk_{h240}(a))\le \tau_{lr}
\land
LCB(Support(a))\ge \tau_s
\land
Cert_{robust}(a)=1
\land
Cost(a)\le C_{max}.
$$

### 必须记录

```text
controller_id
generator_id
certificate_id
thresholds
calibration_split
heldout_split
accepted_count_cal
accepted_count_heldout
coverage_cal
coverage_heldout
h20_weak_CP_cal
h20_weak_CP_heldout
h20_V_ctrl_LCB_heldout
h240_longrisk_heldout
Yrobust_precision_heldout
precision_lcb
longrisk_ucb
support_balance_pass
max_family_share
max_stratum_share
dataset_name_used
outcome_at_commit_used
```

### 判断标准

P9 pass：

```text
dataset_name_used = 0
outcome_at_commit_used = 0
coverage_heldout >= 0.03
coverage_heldout <= 0.15
Yrobust_precision_heldout >= 0.60
h20_V_ctrl_LCB_heldout > 0
h240_longrisk_heldout <= 0.10
support_balance_pass = 1
precision_lcb >= 0.50
longrisk_ucb <= 0.15
```

P9 fail route：

```text
R5-SourceControllerSupportOrHeldoutFail
```

### 可视化

```text
p9_controller_frontier.svg
p9_calibration_to_heldout_drift.svg
p9_support_balance.svg
p9_accepted_population_composition.svg
```

---

## P10：selected source online runtime

### 前置条件

P10 只在 P9 pass 后运行。

### 目标

测 selected controller + selected generator + selected certificate + payload apply 的 online runtime，不得用 no-payload smoke 或 oracle mask。

### 必须记录

```text
runtime_candidate_id
controller_id
generator_id
certificate_id
step_count
active_step_count
accepted_action_count
feature_compute_time_ms_q90
certificate_compute_time_ms_q90
generator_compute_time_ms_q90
payload_apply_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
zero_candidate_controller_kernel_count
zero_candidate_controller_sync_count
controller_launches_per_active_step_q90
selected_payload_apply_used
oracle_mask_used
materializer_in_timed_path
```

### 判断标准

P10 pass：

```text
selected_payload_apply_used = 1
oracle_mask_used = 0
materializer_in_timed_path = 0
zero_candidate_controller_kernel_count = 0
zero_candidate_controller_sync_count = 0
controller_launches_per_active_step_q90 <= 2
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

P10 fail route：

```text
R6-SelectedRuntimeFail
```

### 可视化

```text
p10_runtime_waterfall.svg
p10_step_ratio_distribution.svg
p10_payload_apply_cost.svg
p10_active_step_launches.svg
```

---

## P11：Base-Acc Sentinel continuation

### 目标

继续并行监控 LQ-t2-h256 base 是否 catastrophic fail，并继续记录与 MLP / StrongLRGridMLP / QuadraticFeatureMLP 的固定配置对照。P11 不参与 controller。

### 必须记录

```text
dataset
seed
model_id
train_acc
val_acc
test_acc
train_CE
val_CE
test_CE
ECE
NLL
CEp99
MarginP10
steps
wallclock
params
step_time_q90
hyperparams_fixed_before_run
dataset_specific_tuning
base_acc_used_for_controller
```

### 判断标准

P11 pass：

```text
sentinel_complete = 1
base_acc_used_for_controller = 0
LQ_catastrophic_fail = 0
same_seed_schedule = 1
same_budget = 1
```

Catastrophic fail 定义：

```text
mean_test_acc_LQ < mean_test_acc_MLP - 0.05
or LQ fails on >= 2 datasets by > 0.10
```

### 可视化

```text
p11_base_acc_by_dataset.svg
p11_lq_vs_mlp_gap_by_seed.svg
p11_train_val_test_curves.svg
p11_ece_nll_cepp99_margin_table.svg
```

---

## P12：system integration gate

### 前置条件

P12 只在 P9 和 P10 pass 后运行。

### 目标

决定是否可以打开 leave-out / paired replay / short-full。

### 必须记录

```text
system_candidate_id
controller_id
generator_id
certificate_id
runtime_candidate_id
source_identity_pass
no_transform_equivalence_pass
oracle_replay_pass
generator_pass
certificate_effect_valid_pass
source_controller_pass
selected_runtime_pass
base_acc_sentinel_pass
manual_forward
manual_backward
manual_adamw_update
uses_dataset_name_for_controller
uses_validation_or_test_for_controller
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
official_eligible = 1
system_legal_controller_pass = 1
source_identity_pass = 1
no_transform_equivalence_pass = 1
oracle_replay_pass = 1
generator_pass = 1
certificate_effect_valid_pass = 1
source_controller_pass = 1
selected_runtime_pass = 1
no fake/proxy/offload/leakage
```

---

## P13：leave-dataset-out / leave-stratum-out boundary

### 前置条件

P13 只在 P12 pass 后运行。

### 目标

验证 controller 不是某个 dataset / family / stratum 的偶然成功。

### 必须记录

```text
leaveout_type
heldout_dataset_or_stratum
accepted_count
coverage
Yrobust_precision
h20_V_ctrl_LCB
h240_longrisk
support_balance
family_count
max_family_share
thresholds_frozen
controller_retrained
```

### 判断标准

P13 pass：

```text
thresholds_frozen = 1
controller_retrained = 0
macro coverage >= 0.03
macro Yrobust_precision >= 0.55
macro h20_V_ctrl_LCB > 0
macro h240_longrisk <= 0.12
no dataset-specific branch
```

### 可视化

```text
p13_leave_dataset_metrics.svg
p13_leave_stratum_metrics.svg
p13_threshold_stability.svg
```

---

## P14：official paired replay boundary

### 前置条件

P14 只在 P13 pass 后运行。

### 目标

判断 RealFunctional source update 是否因果上打过 AdamWParallel / bestLR / NoOp / Random / shuffled payload。

### 必须记录

```text
action_id
branch
horizon
CEp99_delta
MarginP10_delta
ECE_delta
NLL_delta
acc_delta
curvature_delta
RealFunctional_value
AdamWParallel_value
bestLR_value
NoOp_value
Random_value
ShuffledPayload_value
Real_beats_AdamWParallel
Real_beats_bestLR
Real_beats_NoOp
Real_beats_Random
Real_beats_ShuffledPayload
```

### 判断标准

P14 pass：

```text
Real_beats_AdamWParallel_rate >= 0.60
Real_beats_bestLR_rate >= 0.55
Real_beats_NoOp_rate >= 0.70
Real_beats_Random_rate >= 0.70
ShuffledPayload_pass = 0
h240_longrisk <= 0.10
```

---

## P15：short/full training boundary

### 前置条件

P15 只在 P14 pass 后运行。

### 目标

打开真正的 short/full functional training 与 MLP comparison，但仍不进行 dataset-specific tuning。

### 必须记录

```text
dataset
seed
model_id
controller_id
generator_id
certificate_id
train_acc_curve
val_acc_curve
test_acc
train_loss_curve
val_loss_curve
ValLossAUC_step
ValLossAUC_time
time_to_target
steps_to_target
ECE
NLL
CEp99
MarginP10
robustness_metrics
continual_retained_accuracy
forgetting
backward_transfer
forward_transfer
step_time_q90
memory_ratio
```

### 判断标准

P15 short pass：

```text
functional_LQ beats matched LQ base or AdamW baseline on ValLossAUC_time or time_to_target
no degradation in test acc beyond tolerance
ECE/NLL not worse by > tolerance
```

P15 full pass：

```text
functional_LQ beats AdamWStrongLRGridMLP or matched MLP under same budget on at least one primary axis：
  test acc
  calibration
  sample efficiency
  time_to_target
  robustness
  continual / anti-forgetting
and no catastrophic failure on any dataset.
```

---

# 9. 并行执行计划

v9.4.6 必须避免“一轮只暴露一个 blocker”。建议并行跑以下批次：

## Batch A：Identity hard preflight

运行：

```text
P1 source ledger
P2 AP0 source replay
P3 no-transform payload equivalence
P4 source-vs-clone side-by-side replay
```

规模：

```text
1-action -> 3-action -> 16-action
```

停止条件：

```text
任一规模 fail，立即停止 generator/certificate official path，只输出 identity failure table。
```

## Batch B：Oracle objective recompute

运行：

```text
P5 oracle aggregation recompute
```

可以与 Batch A 并行，因为它主要读取 source outcome tables 与 aggregation logic。

## Batch C：Base-Acc Sentinel

运行：

```text
P11 Base-Acc Sentinel continuation
```

与所有 source identity 工作并行，不影响 controller。

## Batch D：Generator payload-only preflight

只生成 payload，不跑 outcome，不做 official：

```text
AP0r-AP0v payload equivalence / cost / shape / hash preflight
```

前置：P3 至少 payload equivalence pass。

## Batch E：Conditional generator outcome

只在 Batch A/B pass 后运行：

```text
P6 generator preservation matrix
P7 direct robust generator reset
P8 certificate effect validity
```

## Batch F：Conditional system path

只在 P6/P7 + P8 pass 后运行：

```text
P9 source controller
P10 selected runtime
P12 system gate
P13 leave-out
P14 paired replay
P15 short/full
```

---

# 10. Route decision

v9.4.6 route 按以下顺序判定：

```text
R1a-OracleSourceLedgerIncomplete:
  P1 fail。

R1b-OracleObjectiveOrSourceOutcomeReplayBug:
  P2 fail，AP0 source 自己不能复现 v9.4.4 oracle survivor。

R1c-NoTransformPayloadOrStateCloneBug:
  P3 fail，AP0w payload/state/config 与 source 不一致。

R1d-NoTransformOutcomeReplayBug:
  P4 fail，payload/state/config 一致但 source-vs-clone outcome 不一致。

R1e-OracleAggregationDefinitionBug:
  P5 fail，row/action aggregation 或 robust objective recompute 不一致。

R2-GeneratorDestructiveConfirmedAfterIdentityClosure:
  P1-P5 pass，P6 fail。

R3-DirectRobustSourceGeneratorObjectiveFail:
  P6 fail 后 P7 也 fail。

R4-CertificateEffectValidityFail:
  source generator pass，但 P8 certificate fail。

R5-SourceControllerSupportOrHeldoutFail:
  certificate pass，但 P9 controller fail。

R6-SelectedRuntimeFail:
  controller pass，但 P10 runtime fail。

R7-SystemPassDownstreamBlocked:
  system pass，但 leave-out / paired replay / short-full fail。

R8-v9460-StrictLocalSourceControllerPass:
  P12 pass，进入 official downstream。
```

---

# 11. 本轮成功与失败的解释边界

## 允许声明的成功

```text
source identity closure；
no-transform replay equivalence；
oracle survivor replay validity；
generator destruction confirmed after identity closure；
robust source generator pass；
effect-valid certificate pass；
minimal source controller pass；
selected runtime pass；
system legal controller pass。
```

每个成功必须对应具体 pass gate，不能跨级声明。

## 不允许声明的成功

```text
不能把 Base-Acc Sentinel 写成 functional success；
不能把 AP0w no-transform diagnostic 写成 official controller；
不能把 oracle survivor 写成 legal selector；
不能把 generator preflight pass 写成 value pass；
不能把 certificate field complete 写成 effect-valid；
不能把 no-payload runtime 写成 selected runtime；
不能在 P12 未过时打开 official paired replay / short-full success。
```

---

# 12. 预期结果与下一步分支

## 如果 v9.4.6 停在 R1a/R1b/R1c/R1d/R1e

说明当前最大问题是 identity / replay / aggregation。下一轮应继续修 source action ledger 与 outcome materializer，而不是做 generator。

## 如果 v9.4.6 停在 R2

说明 no-transform 等价已闭合，现有 generator 确实 destructive。下一轮应集中做 direct objective-solved robust generator，不再维护 AP0r/AP0u 的 heuristic 版本。

## 如果 v9.4.6 停在 R3

说明当前 source action frontier 真实存在，但我们还不会 legal 生成/保留它。下一轮应回到 source generation objective，更接近 constrained optimization / short inner solve。

## 如果 v9.4.6 停在 R4

说明 generator 已有 value，但 certificate 不是 sufficient statistic。下一轮做 certificate redesign，不做 generator。

## 如果 v9.4.6 停在 R5/R6

说明能力有了，但 controller/support/runtime 不过。下一轮做 system path，而不是 science reset。

## 如果 v9.4.6 达到 R8

才进入 official leave-out、paired replay、short/full functional evaluation。

---

# 13. 最终一句话

v9.4.6 的核心不是“再造一个更聪明的 source generator”，而是先建立一个不可绕过的等价性合同：

$$
\boxed{\text{如果 no-transform clone 不能复现 source oracle outcome，就不能讨论 transform、certificate 或 controller。}}
$$

这一步看起来不性感，但它会决定之后几轮是继续修一个数据生命周期 bug，还是正式进入 robust source generator 科学问题。
