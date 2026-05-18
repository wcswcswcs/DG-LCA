# DG-KAN v9.4.6 Source Identity / No-Transform Replay Closure 与 Robust Source Revalidation 实验复盘

> 本复盘记录 `DG-KAN_v9.4.6_SourceIdentityNoTransformReplayClosure_RobustSourceRevalidation_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 oracle survivor、AP0 source recompute、AP0w no-transform payload equivalence、Base-Acc Sentinel 或未打开的 generator/controller/runtime 写成 official system pass。

## 0. 最新结论

```text
route = R1d-NoTransformOutcomeReplayBug
base_candidate = LQ-t2-h256
success_v9460_strict_purekan_functional = False
success_v9460_full_functional = False
success_v9460_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9460_source_identity_no_transform_replay_closure_robust_source_revalidation_first_20260514T130000Z/
```

核心结论：

1. P0 复现 v9.4.5 boundary：source route = `R1-OracleSourceIdentityBug`，v9.4.5 no-transform oracle-seeded failed。
2. P1 ORC-D-K16 source identity ledger 过：ledger rows = `16`，join success = `16 / 16`，duplicate action id = `0`，payload/state/branch/label hash missing = `0`。
3. P2 AP0 source oracle survivor 可从 v9.3.5 measured full source outcome table 重算：h20 weak CP = `0.8125`，h20 V_ctrl LCB = `0.13772944106165844`，h240 long-risk = `0.0`，Y_robust count = `13`。
4. 注意：P2 的 `source_replay_mode = measured_v9350_reference_recompute`，`direct_new_rollout_performed = 0`；没有把它写成新的 official source rollout。
5. P3 AP0w no-transform payload equivalence 过：oracle clone rows = `16`，payload hash match rate = `1.0`，L∞ max = `0.0`，relative max = `0.0`，cosine min = `0.9999999999999998`。
6. P4 side-by-side outcome replay 未过：paired branch-horizon rows = `48 / 48`，metric abs diff max = `5.448057344648987`，label match rate = `0.6666666666666666`。
7. P4 关键异常：branch_state_hash match rate = `1.0`，但 horizon_state_hash match rate = `0.0`。
8. P4 failure class = `I18-outcome-materializer-runner-drift`；root cause candidate 是 same payload / same branch start state 下，branch-named rollout path 产生不同 horizon_state_hash。
9. P5 oracle objective aggregation recompute 过：overlap with v9.4.4 oracle = `1.0`，rank match rate = `1.0`。
10. P11 Base-Acc Sentinel continuation 完成：rows = `120`，LQ mean test acc = `0.6537760416666667`，MatchedMLP = `0.562890625`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
11. P6-P10/P12-P15 均 gate-blocked：no-transform outcome replay 未闭合前，不能重开 generator preservation、certificate、controller、runtime、paired replay 或 short/full validation。
12. 当前 primary blocker：`no_transform_outcome_replay_inconsistent`；下一步必须修复 no-transform outcome replay seed / branch semantics，再回到 generator 或 certificate。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9460_source_identity_no_transform_replay_closure.py` | v9.4.6 runner；读取 v9.4.5/v9.4.4/v9.3.5/v9.3.3 artifacts，执行 ORC-D-K16 source ledger、AP0 source outcome recompute、AP0w no-transform payload equivalence、side-by-side outcome replay、oracle aggregation recompute、Base-Acc Sentinel 与 system gate |

代码检查：

```text
python -m py_compile experiments/run_v9460_source_identity_no_transform_replay_closure.py
```

正式运行：

```bash
python experiments/run_v9460_source_identity_no_transform_replay_closure.py \
  --out-dir results/real_rerun_20260506/v9460_source_identity_no_transform_replay_closure_robust_source_revalidation_first_20260514T130000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --oracle-actions 16 --run-base-acc \
  --sentinel-seeds 0,1,2,3,4,5,6,7,8,9
```

运行结果：

```json
{
  "failure_class_primary": "I18-outcome-materializer-runner-drift",
  "no_transform_payload_equivalence_pass": 1,
  "out_dir": "results/real_rerun_20260506/v9460_source_identity_no_transform_replay_closure_robust_source_revalidation_first_20260514T130000Z",
  "route": "R1d-NoTransformOutcomeReplayBug",
  "side_by_side_no_transform_replay_pass": 0,
  "source_identity_ledger_pass": 1
}
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R1d-NoTransformOutcomeReplayBug",
  "source_route_v9450": "R1-OracleSourceIdentityBug",
  "source_identity_ledger_pass": 1,
  "ORC_D_K16_join_success_count": 16,
  "ORC_D_K16_join_missing_count": 0,
  "oracle_survivor_replay_pass": 1,
  "source_replay_mode": "measured_v9350_reference_recompute",
  "source_recomputed_h20_weak_CP": 0.8125,
  "source_recomputed_h20_V_ctrl_lcb": 0.13772944106165844,
  "source_recomputed_h240_long_risk": 0.0,
  "source_recomputed_Y_robust_count": 13,
  "no_transform_payload_equivalence_pass": 1,
  "clone_payload_hash_match_rate": 1.0,
  "clone_payload_linf_max": 0.0,
  "clone_payload_relative_error_max": 0.0,
  "clone_payload_cosine_min": 0.9999999999999998,
  "side_by_side_no_transform_replay_pass": 0,
  "paired_branch_horizon_row_count_actual": 48,
  "metric_abs_diff_max": 5.448057344648987,
  "label_match_rate": 0.6666666666666666,
  "branch_state_hash_match_rate": 1.0,
  "horizon_state_hash_match_rate": 0.0,
  "failure_class_primary": "I18-outcome-materializer-runner-drift",
  "oracle_aggregation_recompute_pass": 1,
  "base_acc_sentinel_pass": 1,
  "base_acc_used_for_controller": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "no_transform_outcome_replay_inconsistent"
}
```

判断：v9.4.6 不是 generator/certificate science failure；它把 blocker 精确定位到 AP0 source 与 AP0w no-transform 的 outcome replay 不等价。

## 3. P1 source identity ledger

Artifacts：

```text
p1_source_identity_ledger.csv
orc_d_k16_identity_trace_v9460.csv
source_payload_state_config_trace_v9460.csv
```

Summary：

```text
oracle_selector_id = ORC-D-HorizonRobust
oracle_panel_id = ORC-D-HorizonRobust-K16
source_action_ledger_rows = 16
source_action_id_duplicate_count = 0
source_payload_hash_missing_count = 0
source_state_before_hash_missing_count = 0
source_branch_config_hash_missing_count = 0
source_label_config_hash_missing_count = 0
ORC_D_K16_join_success_count = 16
ORC_D_K16_join_missing_count = 0
source_identity_ledger_pass = 1
```

判断：P1 闭合。ORC-D-K16 不是 source action ledger 缺失；每个 source action 都能 join 到 payload 与 full outcome table。`source_optimizer_state_hash` 在本轮以 replay-key hash 形式落盘，不伪造成 optimizer tensor snapshot。

## 4. P2 AP0 source outcome recompute

Artifacts：

```text
p2_ap0_source_outcome_recompute.csv
source_replay_trace_v9460.csv
```

Summary：

```text
source_replay_mode = measured_v9350_reference_recompute
direct_new_rollout_performed = 0
source_action_count = 16
h20_weak_CP = 0.8125
h20_V_ctrl_lcb = 0.13772944106165844
h240_long_risk = 0.0
Y_robust_count = 13
Y_robust_rate = 0.8125
source_replay_reproduces_v9440_oracle = 1
oracle_survivor_replay_pass = 1
```

判断：P2 证明 v9.4.4 的 ORC-D-K16 oracle aggregation 在 measured v9.3.5 source table 上可以重算复现；但它不是新的 same-run rollout，因此不能单独写成 official source replay pass。

## 5. P3 AP0w no-transform payload equivalence

Artifacts：

```text
p3_no_transform_payload_equivalence.csv
no_transform_payload_trace_v9460.csv
```

Summary：

```text
oracle_clone_row_count = 16
negative_control_row_count = 16
clone_payload_hash_match_rate = 1.0
clone_payload_linf_max = 0.0
clone_payload_relative_error_max = 0.0
clone_payload_cosine_min = 0.9999999999999998
clone_no_transform_verified_count = 16
no_transform_payload_equivalence_pass = 1
```

判断：P3 pass。AP0w 的 payload identity 本身不是本轮 blocker；没有证据显示 no-transform generator mutated payload。

## 6. P4 side-by-side outcome replay

Artifacts：

```text
p4_source_vs_ap0w_no_transform_side_by_side_replay.csv
no_transform_replay_failure_taxonomy_v9460.csv
```

Summary：

```text
paired_action_count = 16
paired_branch_horizon_row_count_expected = 48
paired_branch_horizon_row_count_actual = 48
pair_metric_row_count = 384
metric_abs_diff_max = 5.448057344648987
label_match_rate = 0.6666666666666666
branch_state_hash_match_rate = 1.0
horizon_state_hash_match_rate = 0.0
failure_class_primary = I18-outcome-materializer-runner-drift
failure_row_count = 48
side_by_side_no_transform_replay_pass = 0
```

示例 failure row：

```text
source_action_id = 2114d2469c2a4cdb87d7cb3869cc240e6b7fb21d42a51df70d97cc3e86517d92
clone_action_id = ffced33de1f358ac5ba4c69051266b3b1aed2ff9ff00536ee8df356d5753ed62
horizon = 20
first_failed_contract = horizon_state_hash_match
expected_value = 6bce4dc784b838f1c745719ec5af7eaf6acd3b8b3f90a96ec4f9e4ad9c2ab780
observed_value = 53da29c334d1431c1e723c5d75f2e9e0a1a3d3059418cef7254875afcee5068d
root_cause_candidate = same payload/start branch_state but branch-named rollout path produces different horizon_state_hash
repair_required = make AP0 source and AP0w clone use identical branch/horizon replay seed and materializer branch semantics
```

判断：P4 是本轮 terminal blocker。source 与 AP0w clone 的 branch start state hash 全部一致，但 horizon state hash 全部不一致；这说明 no-transform outcome replay path 不是等价 replay。

## 7. P5 oracle objective aggregation recompute

Artifacts：

```text
p5_oracle_objective_aggregation_recompute.csv
oracle_rank_recompute_trace_v9460.csv
```

Summary：

```text
oracle_K = 16
overlap_with_v9440_oracle = 1.0
rank_match_rate = 1.0
h20_weak_CP = 0.8125
h20_V_ctrl_lcb = 0.13772944106165844
h240_long_risk = 0.0
Y_robust_recomputed = 1
oracle_recompute_matches_source_table = 1
oracle_aggregation_recompute_pass = 1
```

判断：P5 pass。oracle aggregation definition 没有在本轮暴露 route-level mismatch；主 blocker 仍是 no-transform outcome replay semantics。

## 8. P11 Base-Acc Sentinel

Artifacts：

```text
p11_base_acc_sentinel_continuation.csv
base_acc_training_trace_v9460.csv
```

Summary：

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

判断：Sentinel 继续显示 base 没有 catastrophic fail，但它仍是隔离健康检查，没有用于 selector/generator/controller。

## 9. Gated boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p6_generator_revalidation_boundary.csv` | `P4_no_transform_side_by_side_replay_not_equivalent` |
| `p7_direct_robust_generator_boundary.csv` | same |
| `p8_effect_valid_certificate_boundary.csv` | same |
| `p9_source_certificate_controller_boundary.csv` | same |
| `p10_selected_source_runtime_boundary.csv` | same |
| `p13_leaveout_and_paired_replay_boundary_v9460.csv` | `P12_system_controller_not_official` |
| `p14_short_full_training_boundary_v9460.csv` | same |
| `p15_continual_robustness_boundary_v9460.csv` | same |

没有把 source ledger pass、oracle source recompute、payload equivalence、Base-Acc Sentinel 或 P5 aggregation pass 写成 official controller/runtime/downstream success。

## 10. No-fake audit

```text
rows_checked = 755
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = 1
no_proxy = 1
```

Contract audit：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
train_stream_probe = 1
source_identity_ledger_pass = 1
oracle_survivor_replay_pass = 1
no_transform_payload_equivalence_pass = 1
side_by_side_no_transform_replay_pass = 0
oracle_aggregation_recompute_pass = 1
diagnostic_oracle_used_for_official = 0
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
source_controller/selected_runtime/system = 0/0/0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R1d-NoTransformOutcomeReplayBug
F1a_source_ledger_incomplete = 0
F1b_oracle_source_replay_bug = 0
F1c_no_transform_payload_or_state_bug = 0
F1d_no_transform_outcome_replay_bug = 1
F1e_oracle_aggregation_bug = 0
F2_generator_revalidation_blocked = 1
F3_certificate_revalidation_blocked = 1
F4_controller_runtime_blocked = 1
F5_base_acc_catastrophic = 0
primary_blocker = no_transform_outcome_replay_inconsistent
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `c7364fa757ee2a9010c18f4b1bf90bf48c275c9d42a8bc1e38e76c699f450e84` |
| runner | `9e8013ea98901eafae3b3df4c6c702a2fd2fdb9feebc200dd3b42adfe0e5ad54` |
| run manifest | `f68288442cb208c59cd21b04fdf22588142882ba3421f0157cefff6afcc7e4c9` |
| route | `1fe3c64a5cf6fe8bf7c478de23a2ba0cd900e1e04176b57934ff9e0f11c6497b` |
| aggregate decision | `1fe3c64a5cf6fe8bf7c478de23a2ba0cd900e1e04176b57934ff9e0f11c6497b` |
| P0 boundary | `641294b87783da245a1c64b0428e4917b7ed7f562e7c3efbe55b60956124da3e` |
| P1 source identity ledger | `df6635765f60cce02996e1ca95e952d5441e2f7011f565c0a22d6e116e38a025` |
| ORC-D K16 identity trace | `20a715e136c24186b16d068f351c6766f7e3254c2205a82d2aeee73b8279e68d` |
| P2 source outcome recompute | `e60e7533f0d68d8e7f136239c3dda5866efa3e1937c598bb5e79388e19b73d44` |
| P3 no-transform payload | `f360af95360438c8fbfd94238b98b503fc5445c23d63194edfe8fa46165c86f7` |
| P4 side-by-side replay | `9332b6bcaad8c6eeaef591d218d51edc924b0148e8450e06bcfec46e485a481b` |
| P4 failure taxonomy | `0145c5b883944d07cadad371e194e80bf4498917b459d823256ae15421b29b80` |
| P5 oracle aggregation | `7cdc47467388294b11a2b0585e8b865bbf3d30e44acefa1e584655b621292e0b` |
| P11 Base-Acc Sentinel | `3336c566a52b0ebac86ad30f7402b96d5f4fb32ca8e596208d3eabba6192fdf1` |
| P12 system | `d3370bb069b7b3aaf844a0d45a066a604cdd4e9b108ae72f3ed90825ddf3d5a1` |
| contract audit | `2ee1d7204e850cbbe267110477779bece188b61e660d32990348c3c6b894a164` |
| provenance audit | `1ef6fd4f61b2e7ca80907b79a3595f08232e2066f727d32340ce414ab8628ab1` |
| failure table | `097958854d26dcc7fc849111baa846e4e0489de607b88b954c394c56bec589b6` |

## 12. 最终分析结论

v9.4.6 的真实推进是：

```text
v9.4.5:
  AP0w no-transform oracle-seeded diagnostic failed；
  因此 generator destructive 结论不能继续直接推进。

v9.4.6:
  ORC-D-K16 source identity ledger 闭合；
  measured v9350 source table 上 oracle survivor 可重算复现；
  AP0w no-transform payload 与 source payload 完全等价；
  但 side-by-side outcome replay 在 horizon_state_hash 和 metrics/labels 上不等价。
```

机制判断：

1. H1 部分成立：不是 source ledger / payload hash / state ledger 缺失，而是 outcome replay semantics 不等价。
2. H2 在 measured source table 层面成立：ORC-D-K16 oracle survivor 可重算，h20 weak CP = `0.8125`，h20 V_ctrl LCB > `0`，h240 long-risk = `0.0`。
3. H3 未打开：AP0w no-transform outcome replay 不过，因此 AP0r/AP0v generator destructive 不能在本轮转成最终 science conclusion。
4. H4 未打开：certificate revalidation 因 P4 gate-blocked。
5. H5 成立：Base-Acc Sentinel complete 且没有用于 controller；LQ catastrophic fail = `0`。
6. 当前下一步必须修复 no-transform source/clone 的 branch-horizon replay seed 与 materializer branch semantics，直到 branch_state_hash 和 horizon_state_hash 同时一致，再重跑 generator/certificate。

最终一句话：

> v9.4.6 真实执行后停在 `R1d-NoTransformOutcomeReplayBug`：ORC-D-K16 source 身份和 AP0w no-transform payload 都已闭合，oracle aggregation 也可重算；但同 payload、同 branch start state 的 source/clone side-by-side outcome replay 产生不同 horizon_state_hash 和不同 metrics/labels，因此 strict PureKAN functional 仍未成功，下一步应先修 no-transform replay semantics。
