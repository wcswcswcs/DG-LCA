# DG-KAN v9.7.0 双线验证 / ExistingAction 与 TransferPrinciple 实验复盘

> 本复盘记录 `DG-KAN_v9.7.0_双线验证_ExistingAction与TransferPrinciple_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 score-transport diagnostic、cross-sample transfer diagnostic、core-expansion diagnostic、direct-solved update boundary、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R4-TransferPrincipleFail
base_candidate = LQ-t2-h256
success_v9700_strict_purekan_functional = False
success_v9700_full_functional = False
success_v9700_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9700_dual_validation_existing_action_transfer_principle_first_20260515T190000Z/
```

核心结论：

1. P0 复现 v9.6.8 boundary：source route = `R6-GeneratedRouteStoppedNoNewObjective`，system pass = `0`，generated stop = `1`，APGU run = `0`。
2. P0 field legality 过：green/yellow/red field count = `14 / 4 / 0`，dataset 没有用于 commit feature / selector / controller。
3. P1 cross-sample transfer ledger 完整落盘：AP0 action rows = `2876`，transfer rows = `2876`，missing required field = `0`，nan/inf = `0`。
4. P1 确认 exact per-sample gradient 不在 landed artifacts 中：`exact_per_sample_gradient_available = 0`；本轮只使用 landed commit-time `loo_transfer_proxy/action_projection_signal/grad_mean_sq_group/grad_var_trace_group` 做 diagnostic。
5. P2 existing-action rank 比较未过：strong/weak pass = `0 / 0`。旧 raw R8A 仍是 best overall，TopK87 precision = `0.8735632183908046`，V LCB = `0.14111334880346277`，longrisk UCB = `0.0`，但 LDO drop = `0.37931034482758624`。
6. P2 transfer principle 没有优于旧 rank：best transfer = `R7-TransferLCB-global-conformal-threshold`，precision = `0.17525773195876287`，V LCB = `-0.21558700787971152`，LDO drop = `0.07216494845360824`。
7. P2 `R2-TransferLCB-only` TopK87 的 precision = `0.09195402298850575`，V LCB = `-0.24289252733948727`，longrisk UCB = `0.12221253967001162`；它降低 LDO，但质量崩。
8. P3 Core + Expansion with transfer 未过：best candidate = `B0-T3.1-CoreOnly`，accepted = `77`，coverage = `0.026773296244784424`，GradeAB = `1.0`，V LCB = `0.16402807605399972`，risk/bad/null/memory/offdiag UCB = `0`，但 coverage < `0.03` 且 LDO drop = `0.39080459770114945`。
9. P3 transfer expansion 补到 87 rows 后仍不闭合：`B3-CorePlus10TransferLCB` accepted = `87`，GradeAB = `0.8850574712643678`，V LCB = `0.11911876658700193`，risk/bad/null/memory/offdiag UCB = `0`，但 LDO drop = `0.39080459770114945`。
10. P4 dataset-shift no-tuning diagnostic 未解：raw_score PSI mean = `0.31313509863308747`；TransferLCB PSI mean = `0.4096172222597585`，precision = `0.09195402298850575`，V LCB = `-0.24289252733948727`。
11. P5 exact microprobe 按计划 not_run：reason = `exact_per_sample_apply_checkpoint_not_landed`，没有编造 exact transfer rows。
12. P6 direct transfer-solved update 按 gate not_run：generated actions = `0`，branch-horizon rows = `0`，direct solved weak/strong pass = `0 / 0`。
13. P8/P9 gate-blocked：existing-action controller 与 selected runtime 均 not_run。
14. P10/P11 gate-blocked：system legal controller pass = `0`，paired replay 与 short/full boundary 未打开。
15. Base-Acc Sentinel 继续健康：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
16. 当前 primary blocker：`cross_sample_transfer_not_better_than_old_rank`；secondary blocker：`generated_route_stopped_no_new_objective`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9700_dual_validation_existing_action_transfer_principle.py` | v9.7.0 runner；读取 v9.6.8/v9.6.7/v9.6.6/v9.6.5/v9.6.4/v9.6.3 artifacts 与 canonical AP0 ledger，执行 cross-sample transfer ledger、existing-action rank comparison、core-expansion transfer、dataset-shift no-tuning diagnostic、exact microprobe boundary、direct solved update boundary 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9700_dual_validation_existing_action_transfer_principle.py
```

正式运行：

```bash
python experiments/run_v9700_dual_validation_existing_action_transfer_principle.py \
  --out-dir results/real_rerun_20260506/v9700_dual_validation_existing_action_transfer_principle_first_20260515T190000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --direct-actions-per-subspace 64
```

运行结果：

```json
{
  "best_core_expansion_candidate": "B0-T3.1-CoreOnly",
  "best_rank_rule_id": "R0-raw-R8A-transported-value-rank",
  "best_transfer_rule_id": "R7-TransferLCB-global-conformal-threshold",
  "direct_solved_generated_actions": 0,
  "out_dir": "results/real_rerun_20260506/v9700_dual_validation_existing_action_transfer_principle_first_20260515T190000Z",
  "primary_blocker": "cross_sample_transfer_not_better_than_old_rank",
  "route": "R4-TransferPrincipleFail",
  "secondary_blocker": "generated_route_stopped_no_new_objective",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 proxy rows。P5 exact microprobe 与 P6 direct solved update 都因 landed exact objective/materializer 不满足 gate 而 not_run，没有生成 fake/proxy rows。

## 2. Route

`route_decision_v9700.json` 摘要：

```json
{
  "route": "R4-TransferPrincipleFail",
  "source_route_v9680": "R6-GeneratedRouteStoppedNoNewObjective",
  "p0_pass": 1,
  "p1_transfer_ledger_complete_pass": 1,
  "exact_per_sample_gradient_available": 0,
  "rank_strong_pass": 0,
  "rank_weak_pass": 0,
  "best_rank_rule_id": "R0-raw-R8A-transported-value-rank",
  "best_rank_precision": 0.8735632183908046,
  "best_rank_V_LCB": 0.14111334880346277,
  "best_rank_LDO_drop": 0.37931034482758624,
  "best_transfer_rule_id": "R7-TransferLCB-global-conformal-threshold",
  "best_transfer_precision": 0.17525773195876287,
  "best_transfer_V_LCB": -0.21558700787971152,
  "best_transfer_LDO_drop": 0.07216494845360824,
  "transfer_principle_improves_old_rank": 0,
  "p3_core_expansion_pass": 0,
  "dataset_shift_score_scale_solved": 0,
  "microprobe_pass": 0,
  "direct_solved_weak_pass": 0,
  "direct_solved_strong_pass": 0,
  "new_objective_evidence_present": 0,
  "controller_pass": 0,
  "selected_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "cross_sample_transfer_not_better_than_old_rank",
  "secondary_blocker": "generated_route_stopped_no_new_objective"
}
```

判断：v9.7.0 的关键不是 boundary regression，也不是缺少旧 rank signal。旧 R8A 仍有高 precision/value diagnostic，但 LDO 高；TransferLCB 系列降低 LDO 后 precision/value 崩，不能证明 transfer principle 是可部署替代。因此 route 停在 `R4-TransferPrincipleFail`。

## 3. P0 boundary

Artifacts：

```text
p0_boundary_reproduction_v9700.csv
p0_field_legality_audit_v9700.csv
```

Boundary summary：

```text
source_v9680_route = R6-GeneratedRouteStoppedNoNewObjective
system_legal_controller_pass_v9680 = 0
generated_route_stop_triggered_v9680 = 1
APGU_run_v9680 = 0
field_legality_pass = 1
green/yellow/red field count = 14 / 4 / 0
fake/proxy/cpu_offload = 0 / 0 / 0
base_acc_sentinel_pass = 1
p0_pass = 1
```

Field legality summary：

```text
dataset_allowed_for_leaveout = 1
dataset_name_commit_feature_count = 0
outcome_derived_field_used_count = 0
uses_loss_backward/teacher/loss_modification = 0 / 0 / 0
field_legality_pass = 1
```

判断：P0 pass。v9.7.0 没有跳过 v9.6.8 的 generated route stop / no system boundary，也没有把 dataset name、outcome-derived field 或 validation/test 信息带入 official path。

## 4. P1 Cross-Sample Transfer Ledger

Artifact：

```text
p1_cross_sample_transfer_ledger_v9700.csv
```

Summary：

```text
canonical_ap0_action_count = 2876
transfer_row_count = 2876
missing_required_field_count = 0
nan_inf_count = 0
feature_compute_ms_q90 = 0.047606011250485775
payload_apply_ms_q90_estimate = 0.022
exact_per_sample_gradient_available = 0
transfer_proxy_source = loo_transfer_proxy/action_projection_signal/grad_mean_sq_group/grad_var_trace_group
ledger_complete_pass = 1
fake/proxy/cpu_offload = 0 / 0 / 0
```

判断：P1 完整落盘，但 exact per-sample gradient 不存在于 landed artifacts。本轮 transfer 线只能验证 landed commit-time transfer/gradient proxy diagnostic；不能把它包装成 exact microprobe 或 official solved update objective。

## 5. P2 existing-action rank comparison

Artifact：

```text
p2_existing_action_rank_comparison_v9700.csv
```

Summary：

```text
rule_count = 9
evaluation_row_count = 45
strong_pass_count = 0
weak_pass_count = 0
best_rule_id = R0-raw-R8A-transported-value-rank
best_accepted_count = 87
best_GradeAB_precision = 0.8735632183908046
best_V_integrated_LCB = 0.14111334880346277
best_longrisk_UCB = 0.0
best_bad/null/memory/offdiag UCB = 0.0 / 0.0 / 0.0 / 0.0
best_LDO/LSO/LTO drop = 0.37931034482758624 / 0.02298850574712652 / 0.011494252873563315
best_transfer_rule_id = R7-TransferLCB-global-conformal-threshold
best_transfer_precision = 0.17525773195876287
best_transfer_V_LCB = -0.21558700787971152
best_transfer_LDO_drop = 0.07216494845360824
transfer_principle_improves_old_rank = 0
```

Representative rules：

| rule | TopK/region | precision | V LCB | longrisk UCB | LDO drop | pass |
|---|---:|---:|---:|---:|---:|---:|
| `R0-raw-R8A-transported-value-rank` | `87` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `0` |
| `R0-raw-R8A-transported-value-rank` | `64` | `1.0` | `0.19924925130425136` | `0.0` | `0.328125` | `0` |
| `R2-TransferLCB-only` | `87` | `0.09195402298850575` | `-0.24289252733948727` | `0.12221253967001162` | `0.04597701149425287` | `0` |
| `R3-TransferLCB-plus-SNR-gate` | `87` | `0.09195402298850575` | `-0.24403769497874778` | `0.12221253967001162` | `0.04597701149425287` | `0` |
| `R7-TransferLCB-global-conformal-threshold` | `97` | `0.17525773195876287` | `-0.21558700787971152` | `0.10979531263193916` | `0.07216494845360824` | `0` |
| `R8-TransferLCB-no-dataset-quantile-transport` | `87` | `0.09195402298850575` | `-0.24289252733948727` | `0.12221253967001162` | `0.04597701149425287` | `0` |

判断：旧 rank 的问题仍是 LDO；transfer rank 的问题是质量。TransferLCB 没有成为 “less dataset-shift and still useful” 的 selector。

## 6. P3 Core + Expansion with transfer

Artifact：

```text
p3_core_expansion_transfer_v9700.csv
```

Summary：

```text
candidate_count = 6
candidate_pass_count = 0
core_count = 77
best_candidate_id = B0-T3.1-CoreOnly
best_accepted_count = 77
best_GradeAB_precision = 1.0
best_V_integrated_LCB = 0.16402807605399972
best_longrisk_UCB = 0.0
best_LDO_drop = 0.39080459770114945
p3_core_expansion_pass = 0
```

Representative candidates：

| candidate | expansion | accepted | coverage | GradeAB | V LCB | risk/bad/null/memory/offdiag UCB | LDO | pass |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `B0-T3.1-CoreOnly` | `0` | `77` | `0.026773296244784424` | `1.0` | `0.16402807605399972` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `0` |
| `B1-T3.2-CorePlusNearest10TransportedScore` | `10` | `87` | `0.030250347705146036` | `0.8850574712643678` | `0.1597545290247637` | `0 / 0.03389313880385762 / 0.1674432324061406 / 0 / 0.18196532683597333` | `0.39080459770114945` | `0` |
| `B2-T3.4-CorePlusNearest10ConformalLowRisk` | `2` | `79` | `0.027468706536856746` | `1.0` | `0.16201929527024647` | `0 / 0 / 0 / 0.05995628911411127 / 0.05995628911411127` | `0.41379310344827586` | `0` |
| `B3-CorePlus10TransferLCB` | `10` | `87` | `0.030250347705146036` | `0.8850574712643678` | `0.11911876658700193` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `0` |
| `B4-CorePlus10TransferLCBMemoryGate` | `10` | `87` | `0.030250347705146036` | `0.8850574712643678` | `0.11911876658700193` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `0` |
| `B5-CorePlus10TransferSNROnly` | `10` | `87` | `0.030250347705146036` | `0.8850574712643678` | `0.12330513914825961` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `0` |

判断：transfer expansion 能找到 risk/bad/null/memory/offdiag 干净的 87-row diagnostic，但 LDO 不改善，且 V LCB 低于旧 expansion。Core-only 仍是 “干净但 coverage 不足”。

## 7. P4 dataset-shift no-tuning diagnostic

Artifact：

```text
p4_dataset_shift_no_tuning_diagnostic_v9700.csv
```

Summary：

```text
score_count = 5
raw_score_psi_mean = 0.31313509863308747
strong_diagnostic_count = 0
best_score_id = raw_score
best_psi_ratio_vs_raw = 1.0
best_TopK87_precision = 0.8735632183908046
best_TopK87_V_LCB = 0.14111334880346277
dataset_shift_score_scale_solved = 0
dataset_shift_target_density_still_present = 1
```

Representative scores：

| score | PSI mean | PSI ratio | TopK87 precision | V LCB | longrisk UCB | LDO drop |
|---|---:|---:|---:|---:|---:|---:|
| `raw_score` | `0.31313509863308747` | `1.0` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` |
| `transported_value_rank` | `0.31313509863308747` | `1.0` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` |
| `TransferLCB` | `0.4096172222597585` | `1.3081166054135722` | `0.09195402298850575` | `-0.24289252733948727` | `0.12221253967001162` | `0.04597701149425287` |
| `TransferLCB_plus_SNR` | `0.40864144826472265` | `1.3050004616171873` | `0.09195402298850575` | `-0.24403769497874778` | `0.12221253967001162` | `0.04597701149425287` |
| `TransferLCB_plus_memory_gate` | `0.36442826090225644` | `1.1638052153625587` | `0.09195402298850575` | `-0.24289252733948727` | `0.12221253967001162` | `0.04597701149425287` |

Per-dataset raw score diagnostic：

| dataset | PSI | TopK87 count / precision | V LCB | longrisk UCB | core action count |
|---|---:|---:|---:|---:|---:|
| `Fashion-MNIST` | `0.3477132255617457` | `34 / 1.0` | `0.2185522558920942` | `0.0` | `34` |
| `KMNIST` | `0.3472940184541398` | `36 / 0.7777777777777778` | `0.053201019604910284` | `0.0` | `28` |
| `MNIST` | `0.244398051883377` | `17 / 0.8235294117647058` | `0.05691790232751366` | `0.0` | `15` |

判断：dataset-shift 仍没有被 no-tuning transport 解决。TransferLCB 改变了排序，但不是向 target/controller 方向改善。

## 8. P5-P6 exact/direct transfer boundary

P5 artifact：

```text
p5_exact_microprobe_v9700.csv
```

P5 status：

```text
status = not_run
reason = exact_per_sample_apply_checkpoint_not_landed
diagnostic_proxy_available = 1
exact_per_sample_gradient_available = 0
microprobe_pass = 0
fake/proxy/cpu_offload = 0 / 0 / 0
```

P6 artifact：

```text
p6_direct_transfer_solved_update_v9700.csv
```

P6 summary：

```text
subspace_count = 5
generated_action_count = 0
branch_horizon_rows_actual = 0
direct_solved_weak_pass = 0
direct_solved_strong_pass = 0
new_objective_evidence_present = 0
generated_route_continue_stop = 1
reason = exact_transfer_solve_and_branch_horizon_materializer_not_opened_without_pass
```

P6 subspaces：

```text
D1-last-edge-coefficients-only = not_run
D2-final-KAN-basis-block-only = not_run
D3-low-rank-edge-residual-direction = not_run
D4-AdamW-orthogonal-residual-direction = not_run
D5-memory-gradient-orthogonal-residual-direction = not_run
reason = new_exact_transfer_objective_not_available_from_landed_artifacts
```

判断：P5/P6 是诚实 gate-block。没有 exact transfer objective，就不能运行 direct solved update，也不能生成 fake branch-horizon rows。

## 9. P8-P11 controller / runtime / system

P8：

```text
p8_minimal_controller_v9700.csv = not_run
reason = P2_P3_P6_no_weak_or_strong_pass
controller_selected = 0
controller_pass = 0
source_controller_pass = 0
```

P9：

```text
p9_selected_runtime_v9700.csv = not_run
reason = P8_controller_not_selected
selected_runtime_pass = 0
```

P10：

```text
p10_system_boundary_v9700.csv
controller_pass = 0
runtime_pass = 0
field_legality_pass = 1
system_legal_controller_pass = 0
paired_replay_opened = 0
```

P11：

```text
p11_paired_replay_boundary_v9700.csv = not_run
reason = P10_system_not_official
paired_replay_pass = 0
short_full_boundary_open = 0
```

Allowed next gates：

```json
{
  "selected_runtime_allowed": 0,
  "paired_replay_allowed": 0,
  "short_full_allowed": 0,
  "APGU_APGV_allowed": 0,
  "action_space_redesign_required": 1
}
```

Stop conditions：

```json
{
  "enter_system": 0,
  "stop_generated_blind_variants": 1,
  "stop_old_rank_patch": 1
}
```

判断：没有 P2/P3/P6 weak/strong pass，因此 controller/runtime/system/paired replay/short-full 全部关闭。Generated route 仍要求 objective/action-space redesign，而不是 blind APGU/APGV。

## 10. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9700.csv
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
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
base_acc_sentinel_pass = 1
```

判断：Base-Acc Sentinel 继续健康，但仍是 isolated diagnostic，没有用于 selector、generator、certificate 或 controller。

## 11. Figures

本轮额外落盘的诊断图：

```text
fig_p1_transfer_lcb_hist_by_dataset.svg
fig_p1_transfer_lcb_vs_V_integrated.svg
fig_p1_transfer_lcb_vs_longrisk.svg
fig_p1_snr_vs_gradeab.svg
fig_p1_memory_transfer_vs_longrisk.svg
fig_p2_rule_precision_value_risk_bar.svg
fig_p2_rule_ldo_lso_lto_heatmap.svg
fig_p2_topk_stability_waterfall.svg
fig_p2_transfer_vs_old_rank_scatter.svg
fig_p3_core_expansion_sankey.svg
fig_p3_expansion_candidates_transfer_scatter.svg
fig_p3_core_vs_expansion_quality_table.svg
fig_p3_core_expansion_dataset_support_heatmap.svg
fig_p4_score_distribution_by_dataset.svg
fig_p4_dataset_shift_waterfall.svg
fig_p4_dataset_target_density_map.svg
fig_p4_transfer_vs_raw_psi_bar.svg
fig_p5_linear_vs_exact_scatter.svg
fig_p5_sign_match_by_action_group.svg
fig_p5_microprobe_cost_hist.svg
fig_p6_direct_solved_frontier.svg
fig_p6_new_positive_vs_longrisk.svg
fig_p6_subspace_damage_matrix.svg
fig_p6_action_norm_vs_transfer_lcb.svg
```

判断：这些图只用于复核诊断，不构成 official pass。

## 12. No-fake audit

```text
rows_checked = 3070
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
v9680_boundary_pass = 1
transfer_ledger_pass = 1
rank_strong/weak pass = 0/0
core_expansion_pass = 0
dataset_shift_score_scale_solved = 0
microprobe_pass = 0
direct_solved_pass = 0
controller/runtime/system = 0/0/0
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
uses_old_table_for_official = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R4-TransferPrincipleFail
F0_boundary_or_legality_fail = 0
F1_existing_action_transfer_controller_pass = 0
F2_transfer_principle_weak_not_system = 0
F3_direct_solved_transfer_promising = 0
F4_transfer_principle_fail = 1
F5_core_expansion_density_absent = 0
F6_generated_route_stopped_no_new_objective = 1
F7_system_pass_ready = 0
F8_system_not_official = 1
F9_base_acc_catastrophic = 0
primary_blocker = cross_sample_transfer_not_better_than_old_rank
```

## 13. Hash

| artifact | SHA256 |
|---|---|
| plan | `25f63a3d0c71ad57e66875c9c29d7e54cadffe257d8f61bd55dc0134a65fec51` |
| runner | `a36b6707930a26a49e21cd8f46974077e684b248c2449a3b2590b8bd5703f4ff` |
| run manifest | `de67b898412dd0875b3aed6e0ea60e630c5cf38ed59072df0bfca8ae6068eac6` |
| route / P7 | `49ab7a85a59a9ec9ebf4080ba58e62c8a4fab2fe08a5a572163141a778df9e14` |
| P0 boundary | `4f376e156739e05e84a51368727e6f1c5f091d16e6d62a74ba113e4e0f57e6a7` |
| P0 field legality | `4220270a44ab3e0dd2cf2c77ef5ab64e0f33b3cfc118f0fda67b1ef135ed4687` |
| P1 transfer ledger | `8982b0b0fb11c3833f42e0b630f0e0dfb920498b8dab16320c63eeb76698b961` |
| P2 rank comparison | `795d1ca299b355de60c0e0383f242961a87a0be169bbdab33937b7c57597d722` |
| P3 core expansion | `2b841633f61827e470a1c51aee9c70def0ef0fc4addc216da6eb70814183ee1c` |
| P4 dataset shift | `a85df7820869f25fb43867f96679b3e83a1003bef506e3bd59da06003790ad1b` |
| P5 exact microprobe | `fed394a2a087c6b75003592e33300cae0b632e51923f7883f7b018625353f852` |
| P6 direct solved update | `3e696debba72a26591028b80476e8421e8453a62667a6b5e0c9506b7b9332b23` |
| P8 controller | `b1abe0db6005723904082d9aefec028b7c50b0989ea74b39a642353e1d4ae238` |
| P9 runtime | `6b1ad7b20aa8819196b7e19c3c4c0b7c7383b4cd50215fd6043f16aaad84c1f9` |
| P10 system | `4bc3b77280e3cdf599fd997c58df2b140f4690dcd2d44a54ba20759af47f6ddd` |
| P11 paired boundary | `945640c8514ea90bbb7e56fddb0042c5ac24d8a27f3fc1890222060e73c77f1a` |
| Base-Acc Sentinel | `ac070c3a80a6b8c65586062b0dd81247380c35c48404ea378a5d47efefda8139` |
| no-fake audit | `a46ccb7dfc796d1f034c664d38e1f5723949d3b20010e41e70fef21a64892d01` |
| contract audit | `c4f3747bdf789e853e03627c71200aa939947c8f837c0e435714249b766ecf7e` |
| failure taxonomy | `109d81bc4c90aeb250d7463fcc5a226407948b0caaf533e3216e33218690f59c` |
| allowed next gates | `6eddd348e9d92a605d8b51bffcbd9abc7a8a0e014a8de39ad5e3fd7c4224c843` |
| stop conditions | `3d8ca4c1345b2c25ea2b6c97b80f5a85bb353d133e6d63d30dfb244166ba8201` |

## 14. 最终分析结论

v9.7.0 的真实推进是：

```text
v9.6.8:
  score transport、core-expansion、veto-order、ranker、certificate 都没有形成 official controller；
  generated route stop-rule 继续触发，且没有 new objective evidence。

v9.7.0:
  同时验证 existing-action continuation 与 cross-sample transfer principle；
  landed AP0 artifacts 能构造完整 transfer diagnostic ledger；
  但 exact per-sample gradient / exact apply checkpoint 不在 landed artifacts 中；
  transfer ranking 能降低 LDO，却显著损伤 precision、value 与 risk；
  core-expansion 加 transfer 仍无法同时满足 coverage 与 LDO；
  direct transfer-solved update 因 exact objective/materializer 未打开而 not_run；
  controller/runtime/system/paired replay 全部 gate-blocked。
```

机制判断：

1. H0 成立：v9.6.8 boundary 被复现，没有跳过 generated stop / no system controller。
2. H1 部分成立：cross-sample transfer ledger 可完整落盘，但 exact per-sample gradient 不可用。
3. H2 未成立：TransferLCB 不是比旧 R8A 更好的 selector；best transfer precision 和 V LCB 明显不足。
4. H3 未成立：Core + Expansion with transfer 不能解决 coverage/LDO 冲突；补到 87 rows 后仍 LDO = `0.39080459770114945`。
5. H4 未成立：dataset shift no-tuning diagnostic 没有被 transfer score 解决；TransferLCB 的 PSI ratio 反而高于 raw。
6. H5 未打开：exact microprobe not_run，不能把 linear proxy 写成 exact transfer validation。
7. H6 未打开：direct solved update 没有 exact objective evidence，generated actions = `0`。
8. H7-H10 未打开：没有 weak/strong controller candidate，就不能打开 runtime、system、paired replay 或 short/full training。
9. H11 成立：Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.7.0 真实执行后停在 `R4-TransferPrincipleFail`：旧 existing-action rank 仍是高质量但 LDO 不稳的 diagnostic，cross-sample transfer principle 没能提供更好的 deployable selector；exact transfer microprobe 与 direct solved update 都没有 landed gate 支撑，generated route 仍处于 no-new-objective stop 状态，因此 strict PureKAN functional 仍未成功。
