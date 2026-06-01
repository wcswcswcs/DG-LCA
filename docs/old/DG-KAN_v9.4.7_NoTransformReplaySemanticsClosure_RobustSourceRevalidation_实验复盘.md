# DG-KAN v9.4.7 No-Transform Replay Semantics Closure 与 Robust Source Revalidation 实验复盘

> 本复盘记录 `DG-KAN_v9.4.7_NoTransformReplaySemanticsClosure_RobustSourceRevalidation_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 no-transform replay closure、old v9.3.5 oracle survivor、Base-Acc Sentinel 或未打开的 generator/controller/runtime 写成 official system pass。

## 0. 最新结论

```text
route = R6a-OldOutcomeTableQuarantineRequired
base_candidate = LQ-t2-h256
success_v9470_strict_purekan_functional = False
success_v9470_full_functional = False
success_v9470_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9470_no_transform_replay_semantics_closure_robust_source_revalidation_first_20260514T140000Z/
```

核心结论：

1. P0 复现 v9.4.6 boundary：source route = `R1d-NoTransformOutcomeReplayBug`，v9.4.6 no-transform side-by-side replay failed。
2. P1 source/clone replay ledger 过：ledger rows = `16`，payload/state/optimizer/RNG/branch/horizon/label config hash match rate 全部为 `1.0`。
3. P2 single-action deterministic preflight 过：inner step rows = `20`，first divergence = `none`，metric abs diff max = `0.0`，horizon state hash match rate = `1.0`。
4. P3 canonical runner semantics 过：branch name / clone id 没有进入 execution path，branch count = `6`，same update order = `1`。
5. P4 canonical no-transform scale-up 过：paired rows = `288 / 288`，branch/horizon/optimizer/batch sequence hash match rate 全部为 `1.0`，metric abs diff max = `0.0`，label match rate = `1.0`。
6. P5 no-transform equivalence gate 过：negative controls (`payload_shuffled` / `rng_perturbed` / `optimizer_perturbed`) 都产生 divergence，`no_transform_equivalence_pass = 1`。
7. 这说明 v9.4.6 的 `I18-outcome-materializer-runner-drift` 在 canonical branch-name-invariant runner 下被真实闭合。
8. P6 repaired-runner ORC-D-K16 direct new rollout 未复现旧 v9.3.5 oracle survivor：old h20 weak CP = `0.8125`，new h20 weak CP = `0.0`。
9. P6 新 rollout 指标很差：new h20 V_ctrl LCB = `-1.2978123872692517`，new h240 long-risk = `0.75`，new Y_robust count = `0`。
10. P6 old/new label match rate = `0.0`，old/new V_ctrl abs diff max = `3.7012117721606046`，因此旧 v9.3.5 outcome table 不能继续作为 official generator/certificate/controller 依据。
11. P11 Base-Acc Sentinel continuation 完成：rows = `120`，LQ mean test acc = `0.6537760416666667`，MatchedMLP = `0.562890625`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
12. P7-P10/P13-P14 均 gate-blocked：source oracle revalidation failed 前，不能重开 generator preservation、certificate、controller、selected runtime、paired replay 或 short/full validation。
13. 当前 primary blocker：`old_v9350_outcome_table_not_reproduced_by_canonical_runner`；下一步必须在 repaired canonical runner 下重跑或扩展 AP0 source frontier，再判断 generator/certificate。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9470_no_transform_replay_semantics_closure.py` | v9.4.7 runner；执行 source/clone replay ledger、single-action deterministic preflight、canonical branch-name-invariant runner、no-transform equivalence gate、repaired-runner ORC-D-K16 revalidation、Base-Acc Sentinel 与 system gate |

代码检查：

```text
python -m py_compile experiments/run_v9470_no_transform_replay_semantics_closure.py
```

正式运行：

```bash
python experiments/run_v9470_no_transform_replay_semantics_closure.py \
  --out-dir results/real_rerun_20260506/v9470_no_transform_replay_semantics_closure_robust_source_revalidation_first_20260514T140000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --oracle-actions 16 --run-base-acc \
  --sentinel-seeds 0,1,2,3,4,5,6,7,8,9
```

运行结果：

```json
{
  "new_h20_weak_CP": 0.0,
  "new_h240_long_risk": 0.75,
  "no_transform_equivalence_pass": 1,
  "out_dir": "results/real_rerun_20260506/v9470_no_transform_replay_semantics_closure_robust_source_revalidation_first_20260514T140000Z",
  "route": "R6a-OldOutcomeTableQuarantineRequired",
  "source_oracle_revalidation_pass": 0
}
```

说明：第一次执行在 P2 写 metric delta 时暴露 secondary metric 接口错误；修正为 `secondary_delta` 后清空该 out-dir 并 fresh 重跑。最终 artifact 只对应修正后的正式运行。

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R6a-OldOutcomeTableQuarantineRequired",
  "source_route_v9460": "R1d-NoTransformOutcomeReplayBug",
  "source_clone_replay_key_pass": 1,
  "single_action_preflight_pass": 1,
  "canonical_runner_semantics_pass": 1,
  "side_by_side_no_transform_replay_pass": 1,
  "no_transform_equivalence_pass": 1,
  "paired_branch_horizon_row_count_actual": 288,
  "branch_state_hash_match_rate": 1.0,
  "horizon_state_hash_match_rate": 1.0,
  "metric_abs_diff_max": 0.0,
  "label_match_rate": 1.0,
  "negative_control_divergence_present": 1,
  "direct_new_rollout_performed": 1,
  "old_h20_weak_CP": 0.8125,
  "new_h20_weak_CP": 0.0,
  "old_h20_V_ctrl_lcb": 0.13772944106165844,
  "new_h20_V_ctrl_lcb": -1.2978123872692517,
  "old_h240_long_risk": 0.0,
  "new_h240_long_risk": 0.75,
  "old_Y_robust_count": 13,
  "new_Y_robust_count": 0,
  "old_new_label_match_rate": 0.0,
  "old_table_quarantine_required": 1,
  "source_oracle_revalidation_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "old_v9350_outcome_table_not_reproduced_by_canonical_runner"
}
```

判断：v9.4.7 真实修掉了 no-transform replay semantics drift，但同时证明旧 v9.3.5 measured outcome table 的 ORC-D-K16 oracle survivor 不能被 canonical repaired runner direct rollout 复现。

## 3. P0 v9.4.6 boundary

Artifact：

```text
p0_v9460_boundary_reproduction.csv
```

Summary：

```text
route_v9460 = R1d-NoTransformOutcomeReplayBug
source_identity_ledger_pass = 1
ORC_D_K16_join_success_count = 16
oracle_survivor_replay_pass = 1
source_replay_mode = measured_v9350_reference_recompute
no_transform_payload_equivalence_pass = 1
side_by_side_no_transform_replay_pass = 0
paired_branch_horizon_row_count_actual = 48
metric_abs_diff_max = 5.448057344648987
label_match_rate = 0.6666666666666666
branch_state_hash_match_rate = 1.0
horizon_state_hash_match_rate = 0.0
failure_class_primary = I18-outcome-materializer-runner-drift
```

判断：P0 pass。v9.4.6 的 blocker 被复现，没有跳过旧失败。

## 4. P1/P2 replay key 与 deterministic preflight

P1 artifacts：

```text
p1_source_clone_replay_ledger.csv
```

P1 summary：

```text
source_clone_ledger_rows = 16
missing_replay_key_count = 0
payload_hash_match_rate = 1.0
payload_linf_max = 0.0
payload_relative_error_max = 0.0
payload_cosine_min = 0.9999999999999998
state_before_hash_match_rate = 1.0
optimizer_state_hash_match_rate = 1.0
rng_state_hash_match_rate = 1.0
branch_config_hash_match_rate = 1.0
horizon_config_hash_match_rate = 1.0
label_config_hash_match_rate = 1.0
source_clone_replay_key_pass = 1
```

P2 artifacts：

```text
p2_single_action_deterministic_preflight.csv
stepwise_replay_trace_v9470.csv
```

P2 summary：

```text
source_action_id = 2114d2469c2a4cdb87d7cb3869cc240e6b7fb21d42a51df70d97cc3e86517d92
clone_action_id = ffced33de1f358ac5ba4c69051266b3b1aed2ff9ff00536ee8df356d5753ed62
branch_id = RealFunctional
horizons = 1,2,5,20
inner_step_rows = 20
first_divergence_field = none
all_step_hashes_match = 1
metric_abs_diff_max = 0.0
horizon_state_hash_match_rate = 1.0
single_action_preflight_pass = 1
```

判断：source/clone identity、payload、RNG、optimizer、branch/horizon config 与单 action step-wise replay 都已闭合。

## 5. P3/P4 canonical no-transform runner

P3 artifacts：

```text
p3_canonical_runner_semantics.csv
```

P3 summary：

```text
canonical_runner_version = canonical_no_transform_v9470
branch_name_used_in_execution = 0
clone_id_used_in_execution = 0
same_function_pointer_hash = 1
same_update_order_id = 1
branch_count = 6
canonical_runner_semantics_pass = 1
```

P4 artifacts：

```text
p4_canonical_no_transform_scaleup.csv
canonical_no_transform_outcome_trace_v9470.csv
```

P4 summary：

```text
paired_action_count = 16
paired_branch_horizon_row_count_expected = 288
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
side_by_side_no_transform_replay_pass = 1
```

判断：v9.4.6 的 no-transform outcome replay inconsistency 已被 canonical runner 闭合。这里的 pass 只说明 source 与 no-transform clone 等价，不说明 old v9.3.5 outcome table 仍可 official。

## 6. P5 no-transform equivalence gate

Artifact：

```text
p5_no_transform_equivalence_gate.csv
```

Summary：

```text
no_transform_equivalence_candidate_id = NT-v9470-canonical-branch-name-invariant
source_action_count = 16
clone_action_count = 16
source_clone_payload_equivalence_pass = 1
source_clone_replay_key_pass = 1
side_by_side_no_transform_replay_pass = 1
branch_state_hash_match_rate = 1.0
horizon_state_hash_match_rate = 1.0
metric_abs_diff_max = 0.0
label_match_rate = 1.0
negative_control_divergence_present = 1
oracle_aggregation_recompute_pass = 1
old_new_outcome_drift_audit_pass = 1
no_transform_equivalence_pass = 1
```

Negative controls：

```text
payload_shuffled_clone: diverged = 1, V_branch_abs_diff = 0.028233455261215568
rng_perturbed_clone: diverged = 1, V_branch_abs_diff = 1.0965309366583824
optimizer_state_perturbed_clone: diverged = 1, V_branch_abs_diff = 0.0004553613252937794
```

判断：P5 pass。等价路径能相等，负控能分离，说明本轮 no-transform replay gate 是有判别力的。

## 7. P6 repaired-runner ORC-D-K16 revalidation

Artifact：

```text
p6_repaired_runner_orc_d_k16_revalidation.csv
```

Summary：

```text
source_replay_mode_old = measured_v9350_reference_recompute
source_replay_mode_new = canonical_no_transform_v9470_direct_rollout
direct_new_rollout_performed = 1
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
old_new_metric_drift_mean = 1.2722684671316529
old_new_metric_drift_p99 = 3.7012117721606046
old_table_quarantine_required = 1
source_oracle_revalidation_pass = 0
```

判断：P6 是本轮 route 的关键。canonical repaired runner 下，ORC-D-K16 不是 robust survivor；旧 v9.3.5 measured table 的 survivor 结论必须 quarantine，不能继续支撑 generator preservation、certificate 或 controller。

## 8. P7-P10/P12-P14 gated boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_generator_preservation_revalidation_boundary.csv` | `P6_source_oracle_revalidation_failed` |
| `p8_effect_valid_certificate_revalidation_boundary.csv` | same |
| `p9_source_certificate_controller_boundary.csv` | same |
| `p10_selected_source_online_runtime_boundary.csv` | same |
| `p13_ldo_lso_paired_replay_boundary.csv` | `P12_system_controller_not_official` |
| `p14_short_full_validation_boundary.csv` | `P13_paired_replay_not_open` |

P12：

```text
no_transform_replay_pass = 1
source_oracle_revalidation_pass = 0
generator_preservation_pass = 0
certificate_effect_valid_pass = 0
source_controller_pass = 0
selected_runtime_pass = 0
official_eligible = 0
system_legal_controller_pass = 0
reason = old_v9350_outcome_table_not_reproduced_by_canonical_runner
```

判断：没有把 no-transform replay closure 或 Base-Acc Sentinel 写成 system pass。P6 不过，generator/certificate/controller/runtime 必须关闭。

## 9. P11 Base-Acc Sentinel

Artifacts：

```text
p11_base_acc_sentinel_continuation.csv
base_acc_training_trace_v9470.csv
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
LQ_minus_QuadraticFeatureMLP_mean_test_acc = 0.2514322916666667
LQ_minus_AdamWStrongLRGridMLP_mean_test_acc = 0.025260416666666674
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
base_acc_sentinel_pass = 1
```

判断：Base-Acc Sentinel 继续健康，但没有用于 selector/generator/controller，也不是 functional success。

## 10. No-fake audit

```text
rows_checked = 2822
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
source_clone_replay_key_pass = 1
single_action_preflight_pass = 1
canonical_runner_semantics_pass = 1
no_transform_equivalence_pass = 1
source_oracle_revalidation_pass = 0
generator_preservation_pass = 0
certificate_effect_valid_pass = 0
source_controller_pass = 0
selected_runtime_pass = 0
system_legal_controller_pass = 0
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
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
route = R6a-OldOutcomeTableQuarantineRequired
F1_replay_key_ledger_incomplete = 0
F2_single_action_replay_divergence = 0
F3_branch_named_runner_leak = 0
F4_no_transform_still_inconsistent = 0
F5_no_transform_gate_unclosed = 0
F6_old_table_quarantine_required = 1
F7_generator_revalidation_blocked = 1
F8_certificate_revalidation_blocked = 1
F9_system_not_official = 1
F10_base_acc_catastrophic = 0
primary_blocker = old_v9350_outcome_table_not_reproduced_by_canonical_runner
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `ce11869cccf540c1063818caecd85dbfae98cbb4defb650e4c5b37086aca88ec` |
| runner | `2a406b5b4dc7af0853ce206fa934f385c7f2c8d8243df971074692ff628d3b0a` |
| run manifest | `41b8084a572b4d73b4fbce194b9002fade134c67ea1bc99366a2dc8dbfc40be5` |
| route | `ebb0611c7636876041cd0cface4b94d9f805c71cac74a08932927a562d3d9dc2` |
| aggregate decision | `ebb0611c7636876041cd0cface4b94d9f805c71cac74a08932927a562d3d9dc2` |
| P0 boundary | `9c38d27c64fd8209fb253fb78e433b0c70b7c130c8bfcf06dace548686903212` |
| P1 source/clone ledger | `657eea3cd64fb6e85fea09234e488b971a2b3c785424dfb8f8d48291399d5a3a` |
| P2 preflight | `1f71cfab462caaf8582d035643a7fab62cdfd980339869d323d1ae10265ef125` |
| stepwise replay trace | `cccbf85292f6e7550bbada6d343b4f4372503bb3d2e9abb55b8aff61655d4f21` |
| P3 canonical runner | `29a118518dcec3b05d89a9ba224b4895943e50b2b94421c62f44499c59be80d1` |
| P4 scaleup | `c0850324cf8c5819d221f0e9a8ab7c828a1cd7bd21b244290646f0cf764cb853` |
| canonical outcome trace | `6791a80f66a0327acc2de7240e4b2b7e0e8ca9c688dc7607b073ff0630504e66` |
| P5 equivalence gate | `00923d0cea05d4990770b126f6a708daf86513223cf25fc0f8bc284c40b2893d` |
| P6 revalidation | `6144c1562aa74d8089725d2120b5f795ceb2709cbf7f55a3c90f274b78cd6d44` |
| P11 Base-Acc Sentinel | `a958e4f9252f0d1c083a7212595a079adfc672ba18672d143f29a3b06b6c6150` |
| P12 system | `3c9f2bdcec97250c85967bb586915f0a1bc1a8a4178e7200a0d627317e244264` |
| contract audit | `5060db19611a45f237b4ef7d520b8764d162e14898c388ea76c5e1bb72326fdd` |
| provenance audit | `1a32e3000b0c9e5f268a1136670e075ea9a86133b633dff74c8f11f1a5c47d71` |
| failure table | `47c0e0a69dc1cbf955b5ac60fbeb778e0d4401cefd466b5d2448dd22b4777dac` |

## 12. 最终分析结论

v9.4.7 的真实推进是：

```text
v9.4.6:
  source identity、payload equivalence、old oracle aggregation 都过；
  但 no-transform source/clone side-by-side replay 不等价。

v9.4.7:
  canonical branch-name-invariant runner 修通 no-transform replay semantics；
  source 与 clone 在 16-action / 6-branch / 3-horizon 上完全等价；
  但 repaired runner direct new rollout 不复现旧 v9.3.5 ORC-D-K16 oracle survivor。
```

机制判断：

1. H1 成立并推进：v9.4.6 的 runner drift 确实在 outcome semantics 层，v9.4.7 用 canonical runner 闭合。
2. H2/H3 成立：branch name、clone id、batch sequence、RNG、optimizer state 和 update order 都不再导致 source/clone divergence。
3. H4 到 P5 成立：no-transform equivalence gate 已过，且负控能分离。
4. H5 成立：旧 v9.3.5 measured table 的 ORC-D-K16 survivor 不能在 repaired runner 下复现，必须 quarantine。
5. H6 未打开：P6 fail 后，generator/certificate/controller/runtime 都不能重开。
6. Base-Acc Sentinel 继续健康，但仍是 isolated diagnostic，没有进入 controller。
7. P12 未打开：不能进入 LDO/LSO、official paired replay 或 short/full validation。

最终一句话：

> v9.4.7 真实执行后停在 `R6a-OldOutcomeTableQuarantineRequired`：no-transform replay semantics 已经闭合，这是本轮真实推进；但 canonical repaired runner 下 ORC-D-K16 旧 oracle survivor 完全不复现，旧 v9.3.5 outcome table 必须隔离，strict PureKAN functional 仍未成功。
