# DG-KAN v9.6.5 Template-Lineage Balanced Controller / Offdiag-Memory Causal Primitive 实验复盘

> 本复盘记录 `DG-KAN_v9.6.5_TemplateLineageBalancedController_OffdiagMemoryCausalPrimitive_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 template-lineage diagnostic、memory/offdiag intervention diagnostic、rank diagnostic、APGT implementation/smoke、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R4-NoTemplateBalancedTarget
base_candidate = LQ-t2-h256
success_v9650_strict_purekan_functional = False
success_v9650_full_functional = False
success_v9650_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9650_template_lineage_balanced_controller_offdiag_memory_causal_primitive_first_20260515T150000Z/
```

核心结论：

1. P0 复现 v9.6.4 boundary：source route = `R2-LineageCollapsedAcceptedRegion`，candidate_template_unique_count_v9640 = `1`，system pass = `0`，no-fake/no-proxy = `1 / 1`。
2. P1 lineage hierarchy rebuild 过：在 v9.6.5 重新定义 `candidate_template_id` 后，accepted region 的 unique candidate template count = `87`，max candidate template share = `0.011494252873563218`，template-to-action expansion ratio = `1.0`。
3. P1 同时确认旧口径的 candidate_id 仍塌缩：unique_candidate_id_count = `1`，candidate-to-action expansion ratio = `87.0`，candidate_id_collision_count = `86`。
4. P2 accepted-region leave-template-out 过：best object = `RANK5-invariant-risk-minimization`，GradeAB precision = `0.8735632183908046`，V LCB = `0.14111334880346277`，longrisk UCB = `0.0`，LTO precision drop = `0.011494252873563315`。
5. P3 memory/offdiag causal gate 过：matched joint pairs = `397`，intervention clones = `512`，clone replay rows = `12288`；matched joint GradeAB lift = `0.19143576826196473`，V lift = `0.29043934585539005`，longrisk drop = `0.906693153540672`。
6. P3 intervention clones 本身没有 positive lift：intervention GradeAB / V / longrisk lift = `0.0 / 0.0 / 0.0`；P3 pass 主要来自 matched-pair causal evidence + clone materialization completeness。
7. P4 template-balanced target map 未过：best = `T4.1-existing-ValuePositiveNoLongRisk`，count = `97`，coverage = `0.03372739916550765`，V LCB = `0.16622185363738462`，longrisk UCB = `0.0`；但 bad/null UCB = `0.1505214153590923 / 0.18923506172101434`，LDO drop = `0.3333333333333333`，max group share = `0.9278350515463918`。
8. P4 多个 strict target value/risk 很干净，但 count/coverage 或 group stability 仍失败：`T4.2` count = `79`，coverage = `0.027468706536856746`；`T4.3` count = `77`，coverage = `0.026773296244784424`。
9. P5 template-deconfounded ranker 过 diagnostic/weak gate：best = `R5B-pairwise-within-template`，TopK87 precision = `0.8735632183908046`，V LCB = `0.14111334880346277`，longrisk UCB = `0.0`，candidate_template_count = `87`。
10. P6 rank-safe certificate v13 未过：best = `CERT13-FixedTopK87Quota1`，accepted = `87`，coverage = `0.030250347705146036`，precision = `0.8735632183908046`，V LCB = `0.14111334880346277`，longrisk UCB = `0.0`；但 LDO drop = `0.37931034482758624`，certificate pass = `0`。
11. P7/P8 gate-blocked：existing-action controller 与 selected runtime 均 not_run。
12. P9 APGL/APGH damage decomposition 过 diagnostic：damage rows = `1024`，dominant mode = `D7-longrisk-created`，longrisk created rate = `0.7294921875`，Damage V LCB = `-0.5662455950613429`。
13. P10/P11 APGT implementation 与 branch-horizon replay 过：generated actions = `512`，candidate templates = `512`，rows = `12288 / 12288`，unresolved exception = `0`，quality audit pass = `1`。
14. P12 APGT outcome 未过：best = `APGT4-TemplateMixtureAnchor`，GradeAB precision = `0.03125`，V LCB = `-0.6095415439891327`，h240 longrisk UCB = `0.7869099970100811`。
15. P13/P15 gate-blocked：没有 APGT official candidate、controller、leaveout、paired replay 或 short/full boundary。
16. Base-Acc Sentinel 继续健康：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
17. 当前 primary blocker：`template_balanced_target_absent`；secondary blocker：`apgt_generated_frontier_fail`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9650_template_lineage_balanced_controller_offdiag_memory_causal_primitive.py` | v9.6.5 runner；读取 v9.6.4/v9.6.3/v9.6.2/v9.6.1/v9.5.9/v9.5.8/v9.5.6 artifacts，执行 template lineage rebuild、leave-template-out audit、memory/offdiag causal intervention、template-balanced target/rank/certificate、APGL/APGH damage decomposition、APGT primitive 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9650_template_lineage_balanced_controller_offdiag_memory_causal_primitive.py
```

正式运行：

```bash
python experiments/run_v9650_template_lineage_balanced_controller_offdiag_memory_causal_primitive.py \
  --out-dir results/real_rerun_20260506/v9650_template_lineage_balanced_controller_offdiag_memory_causal_primitive_first_20260515T150000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --intervention-source-count 128 --apgt-actions-per-primitive 64
```

运行结果：

```json
{
  "apgt_generated_actions": 512,
  "apgt_rows": 12288,
  "best_apgt_V_integrated_LCB": -0.6095415439891327,
  "best_apgt_primitive": "APGT4-TemplateMixtureAnchor",
  "best_ranker": "R5B-pairwise-within-template",
  "best_ranker_TopK87_precision": 0.8735632183908046,
  "memory_offdiag_causal_pass": 1,
  "out_dir": "results/real_rerun_20260506/v9650_template_lineage_balanced_controller_offdiag_memory_causal_primitive_first_20260515T150000Z",
  "primary_blocker": "template_balanced_target_absent",
  "route": "R4-NoTemplateBalancedTarget",
  "system_legal_controller_pass": 0,
  "unique_candidate_template_count": 87
}
```

说明：本轮没有 CPU offload，没有 proxy rows；P3 intervention clone 与 P11 APGT branch-horizon replay 都是真实落盘。

## 2. Route

`route_decision_v9650.json` 摘要：

```json
{
  "route": "R4-NoTemplateBalancedTarget",
  "source_route_v9640": "R2-LineageCollapsedAcceptedRegion",
  "p0_pass": 1,
  "lineage_hierarchy_pass": 1,
  "unique_candidate_template_count": 87,
  "template_to_action_expansion_ratio": 1.0,
  "accepted_region_template_stability_pass": 1,
  "template_collapsed_accepted_region": 0,
  "memory_offdiag_causal_pass": 1,
  "intervention_clone_count": 512,
  "template_balanced_target_pass": 0,
  "best_target_id": "T4.1-existing-ValuePositiveNoLongRisk",
  "template_deconfounded_ranker_pass": 1,
  "best_ranker_id": "R5B-pairwise-within-template",
  "best_ranker_TopK87_precision": 0.8735632183908046,
  "rank_safe_certificate_v13_pass": 0,
  "existing_action_controller_pass": 0,
  "selected_runtime_pass": 0,
  "apgl_apgh_damage_decomposition_pass": 1,
  "dominant_damage_mode": "D7-longrisk-created",
  "apgt_implementation_pass": 1,
  "apgt_branch_horizon_pass": 1,
  "apgt_weak_pass": 0,
  "apgt_official_candidate_pass": 0,
  "best_apgt_primitive": "APGT4-TemplateMixtureAnchor",
  "best_apgt_GradeAB_precision": 0.03125,
  "best_apgt_V_integrated_LCB": -0.6095415439891327,
  "best_apgt_h240_longrisk_UCB": 0.7869099970100811,
  "apgt_controller_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "template_balanced_target_absent",
  "secondary_blocker": "apgt_generated_frontier_fail"
}
```

判断：v9.6.5 不是 boundary reproduction fail，也不是 template hierarchy incomplete。新的 template lineage 口径让 accepted region 不再停在 R2；memory/offdiag causality 也过了。但 P4 没有找到 official template-balanced target，因此 existing-action route 在 controller 前被 gate-blocked。

## 3. P0 v9.6.4 boundary

Artifact：

```text
p0_boundary_reproduction_v9650.csv
```

Summary：

```text
source_route_v9640 = R2-LineageCollapsedAcceptedRegion
system_legal_controller_pass_v9640 = 0
accepted_action_count_v9640 = 87
accepted_unique_action_count_v9640 = 87
candidate_template_unique_count_v9640 = 1
candidate_to_action_expansion_ratio_v9640 = 87.0
best_adjusted_axis_v9640 = offdiag_bucket
memory_offdiag_causality_pass_v9640 = 0
best_ranker_v9640 = R4C-value-longrisk-memory-veto
best_ranker_TopK87_precision_v9640 = 0.5862068965517241
best_apgl_primitive_v9640 = APGL1-SourceReplayPreserver
best_apgl_V_LCB_v9640 = -0.5827183891309656
best_apgl_h240_longrisk_UCB_v9640 = 0.7869099970100811
no_fake_v9640/no_proxy_v9640 = 1 / 1
p0_pass = 1
```

判断：P0 pass。v9.6.5 没有跳过 v9.6.4 的 lineage-collapse / APGL failure boundary。

## 4. P1 lineage hierarchy rebuild

Artifacts：

```text
p1_lineage_hierarchy_rebuild_v9650.csv
fig_p1_lineage_hierarchy_sunburst.svg
fig_p1_candidate_template_expansion_hist.svg
fig_p1_action_to_template_sankey.svg
fig_p1_entropy_by_lineage_level.svg
fig_p1_template_collision_table.svg
```

Summary：

```text
unique_action_count = 87
unique_event_count = 87
unique_candidate_id_count = 1
unique_strict_lineage_count = 87
unique_source_payload_count = 87
unique_candidate_template_count = 87
unique_generator_template_count = 87
entropy_action = 4.4659081186545775
entropy_candidate_template = 4.4659081186545775
max_action_share = 0.011494252873563218
max_candidate_template_share = 0.011494252873563218
candidate_to_action_expansion_ratio = 87.0
template_to_action_expansion_ratio = 1.0
candidate_id_collision_count = 86
template_collision_count = 0
candidate_template_missing_count = 0
lineage_hierarchy_pass = 1
candidate_template_diversity_pass = 1
```

判断：P1 是本轮关键正向推进。v9.6.4 的 `candidate_id` 口径确实塌缩，但 v9.6.5 的 candidate-template hierarchy 能把 87 个 accepted actions 分解成 87 个 template-level support。旧 blocker 不再是 terminal route。

## 5. P2 leave-template-out audit

Artifacts：

```text
p2_accepted_region_leave_template_out_v9650.csv
fig_p2_lto_precision_drop_by_ranker.svg
fig_p2_template_fold_heatmap.svg
fig_p2_accepted_template_share_bar.svg
fig_p2_value_longrisk_by_template_fold.svg
```

Summary：

```text
candidate_object_count = 4
template_stable_object_count = 2
best_object_id = RANK5-invariant-risk-minimization
best_GradeAB_precision = 0.8735632183908046
best_V_integrated_LCB = 0.14111334880346277
best_h240_longrisk_UCB = 0.0
best_candidate_template_count = 87
best_max_candidate_template_share = 0.011494252873563218
best_leave_template_out_precision_drop = 0.011494252873563315
accepted_region_template_stability_pass = 1
template_collapsed_accepted_region = 0
```

判断：P2 pass。accepted-region 在新 template hierarchy 下不是单模板 pocket；但这只是 accepted-region stability，并不等于 controller pass。

## 6. P3 memory/offdiag causal intervention

Artifacts：

```text
p3_memory_offdiag_causal_intervention_v9650.csv
p3_intervention_branch_horizon_trace_v9650.csv
fig_p3_memory_offdiag_matched_lift.svg
fig_p3_intervention_lift_waterfall.svg
fig_p3_value_direction_cosine_vs_lift.svg
fig_p3_longrisk_drop_by_template.svg
fig_p3_joint_safe_scatter_V_vs_longrisk.svg
```

Summary：

```text
matched_pair_count_memory = 511
matched_pair_count_offdiag = 397
matched_pair_count_joint = 397
intervention_clone_count = 512
intervention_branch_horizon_rows = 12288
value_direction_cosine_preserved_rate = 1.0
GradeAB_lift_memory_safe = 0.1487279843444227
GradeAB_lift_offdiag_safe = 0.19143576826196473
GradeAB_lift_joint_safe = 0.19143576826196473
V_lift_memory_safe = 0.2992236113845136
V_lift_offdiag_safe = 0.29043934585539005
V_lift_joint_safe = 0.29043934585539005
longrisk_drop_memory_safe = 0.9500635593939639
longrisk_drop_offdiag_safe = 0.906693153540672
longrisk_drop_joint_safe = 0.906693153540672
intervention_GradeAB_lift_joint_safe = 0.0
intervention_V_lift_joint_safe = 0.0
intervention_longrisk_drop_joint_safe = 0.0
LDO_lift_drop = 0.06191769597280811
LSO_lift_drop = 0.0044361708668599065
LTO_lift_drop = 0.0
memory_offdiag_causal_pass = 1
```

判断：P3 pass。memory/offdiag safe 不再只是 weak diagnostic，它在 matched-pair 条件下满足本轮 causal gate；但 intervention clone 没有带来额外 lift，因此它还不能直接作为 generator/controller recipe。

## 7. P4 template-balanced target map

Artifacts：

```text
p4_template_balanced_target_map_v9650.csv
fig_p4_target_density_vs_template_count.svg
fig_p4_value_risk_frontier.svg
fig_p4_support_by_template_heatmap.svg
fig_p4_target_lattice_parallel_coordinates.svg
```

Summary：

```text
target_count = 8
target_pass_count = 0
best_target_id = T4.1-existing-ValuePositiveNoLongRisk
best_action_count = 97
best_coverage = 0.03372739916550765
best_V_integrated_LCB = 0.16622185363738462
best_h240_longrisk_UCB = 0.0
best_candidate_template_count = 97
best_max_candidate_template_share = 0.010309278350515464
template_balanced_target_pass = 0
```

Representative targets：

| target | count | coverage | GradeAB | V LCB | longrisk UCB | bad/null UCB | LDO drop | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `T4.1-existing-ValuePositiveNoLongRisk` | `97` | `0.03372739916550765` | `0.8144329896907216` | `0.16622185363738462` | `0.0` | `0.1505214153590923 / 0.18923506172101434` | `0.3333333333333333` | `0` |
| `T4.2-TemplateBalancedValuePositiveNoLongRisk` | `79` | `0.027468706536856746` | `1.0` | `0.16201929527024647` | `0.0` | `0.0 / 0.0` | `0.41379310344827586` | `0` |
| `T4.3-MemoryOffdiagSafeValuePositive` | `77` | `0.026773296244784424` | `1.0` | `0.16402807605399972` | `0.0` | `0.0 / 0.0` | `0.39080459770114945` | `0` |
| `T4.5-SoftTemplateBalancedGradeABC` | `92` | `0.031988873435326845` | `0.8586956521739131` | `0.15851965165518128` | `0.0` | `0.08515038730290342 / 0.15853250821206355` | `0.36781609195402293` | `0` |

判断：P4 是本轮 primary blocker。value/risk 干净区域存在，template count 也够；但 bad/null 或 LDO/group stability 不闭合，不能作为 official target。

## 8. P5/P6 ranker 与 certificate

P5 artifacts：

```text
p5_template_deconfounded_ranker_v5.csv
fig_p5_rank_score_hist_by_template.svg
fig_p5_ranker_ablation_waterfall.svg
fig_p5_value_vs_risk_score_scatter.svg
fig_p5_template_quota_map.svg
fig_p5_lto_drop_by_ranker.svg
```

P5 summary：

```text
ranker_count = 8
ranker_pass_count = 1
best_ranker_id = R5B-pairwise-within-template
best_TopK87_precision = 0.8735632183908046
best_V_integrated_LCB = 0.14111334880346277
best_h240_longrisk_UCB = 0.0
best_candidate_template_count = 87
best_max_candidate_template_share = 0.011494252873563218
best_LTO_drop = 0.011494252873563315
template_deconfounded_ranker_pass = 1
```

P6 artifacts：

```text
p6_rank_safe_certificate_v13.csv
fig_p6_calibration_vs_heldout_drift.svg
fig_p6_accepted_region_by_template.svg
fig_p6_threshold_sensitivity_grid.svg
fig_p6_precision_value_longrisk_frontier.svg
```

P6 summary：

```text
certificate_count = 4
certificate_pass_count = 0
best_certificate_id = CERT13-FixedTopK87Quota1
best_ranker_id = R5B-pairwise-within-template
best_accepted_count = 87
best_coverage = 0.030250347705146036
best_GradeAB_precision = 0.8735632183908046
best_V_integrated_LCB = 0.14111334880346277
best_h240_longrisk_UCB = 0.0
best_candidate_template_count = 87
best_max_candidate_template_share = 0.011494252873563218
best_LTO_drop = 0.011494252873563315
rank_safe_certificate_v13_pass = 0
```

Certificate failure detail：

```text
best_LDO_drop = 0.37931034482758624
best_LSO_drop = 0.02298850574712652
best_LTO_drop = 0.011494252873563315
```

判断：P5 说明 ranker 层可以找到 template-deconfounded TopK；P6 说明 frozen certificate 仍被 dataset/group leaveout 否决。不能打开 controller。

## 9. P7-P8 controller/runtime boundary

P7：

```text
p7_existing_action_minimal_controller_v9650.csv = not_run
reason = P6_certificate_not_passed
existing_action_controller_pass = 0
source_controller_pass = 0
```

P8：

```text
p8_selected_controller_runtime_v9650.csv = not_run
reason = P7_controller_not_selected
selected_runtime_pass = 0
```

判断：没有把 P5 ranker pass 写成 official controller；P6 certificate 不过，所以 runtime 不打开。

## 10. P9 APGL/APGH damage decomposition v2

Artifacts：

```text
p9_apgl_apgh_damage_decomposition_v2_v9650.csv
fig_p9_source_to_generated_damage_matrix.svg
fig_p9_value_direction_cosine_hist.svg
fig_p9_longrisk_created_by_primitive.svg
fig_p9_memory_offdiag_delta_by_primitive.svg
fig_p9_template_diversity_source_vs_generated.svg
```

Summary：

```text
damage_row_count = 1024
dominant_damage_mode = D7-longrisk-created
dominant_damage_mode_fraction = 0.4150390625
value_direction_lost_fraction = 0.3662109375
longrisk_created_rate = 0.7294921875
offdiag_fail_delta_mean = 0.8037109375
memory_fail_delta_mean = 0.0
Damage_V_LCB = -0.5662455950613429
generator_reset_preserve_value_direction_required = 0
generator_reset_hard_longrisk_veto_required = 1
apgl_apgh_damage_decomposition_pass = 1
```

判断：APGL/APGH 的主要损伤仍是 longrisk-created 和 offdiag fail，value-direction-lost 也不低，但未超过 0.50。后续 generator 必须有 hard longrisk veto。

## 11. P10-P12 APGT primitive

P10 artifact：

```text
p10_apgt_primitive_implementation_v9650.csv
```

P10 summary：

```text
primitive_count = 8
generated_action_count_expected/actual = 512 / 512
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_linf_max = 0.0
negative_control_generated = 1
candidate_template_count = 512
apgt_implementation_pass = 1
```

P11 artifact：

```text
p11_apgt_branch_horizon_smoke_v9650.csv
```

P11 summary：

```text
branch_horizon_rows_expected/actual = 12288 / 12288
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1.0
rows_per_sec = 14.776368425056933
wallclock_sec = 831.5981062818319
unresolved_exception_count = 0
duplicate_row_count = 0
label_exclusivity_violation_count = 0
quality_audit_pass = 1
apgt_branch_horizon_pass = 1
```

P12 artifacts：

```text
p12_apgt_outcome_geometry_pass_v9650.csv
fig_p12_apgt_gradeab_by_primitive.svg
fig_p12_apgt_value_longrisk_frontier.svg
fig_p12_apgt_template_diversity.svg
fig_p12_apgt_damage_waterfall.svg
```

P12 summary：

```text
best_primitive_id = APGT4-TemplateMixtureAnchor
best_GradeAB_precision = 0.03125
best_GoodGeometry_A_precision = 0.03125
best_GoodGeometry_B_precision = 0.03125
best_V_integrated_LCB = -0.6095415439891327
best_h240_longrisk_UCB = 0.7869099970100811
best_candidate_template_count = 64
best_max_candidate_template_share = 0.015625
best_new_positive_created_rate = 0.03125
best_longrisk_created_rate = 0.671875
best_source_to_generated_damage_LCB = -0.7179581283606722
apgt_weak_pass = 0
apgt_official_candidate_pass = 0
```

Per primitive：

| primitive | GradeAB precision | V LCB | h240 longrisk UCB | candidate templates | pass |
|---|---:|---:|---:|---:|---:|
| `APGT1-NoTransformTemplateDiverseReplay` | `0.015625` | `-0.5935838834065819` | `0.9081265318504754` | `64` | `0` |
| `APGT2-ValueDirectionPreservingTemplatePerturb` | `0.0` | `-0.5552931539974693` | `0.8825326673766913` | `64` | `0` |
| `APGT3-MemoryOffdiagNullspaceValuePreserver` | `0.0` | `-0.657830534666301` | `0.8560881119635937` | `64` | `0` |
| `APGT4-TemplateMixtureAnchor` | `0.03125` | `-0.6095415439891327` | `0.7869099970100811` | `64` | `0` |
| `APGT5-SignalChannelSNRPopulationRiskDelta` | `0.0` | `-0.6744139553865214` | `0.8825326673766913` | `64` | `0` |
| `APGT6-CoverMemoryBoundarySymmetricDelta` | `0.015625` | `-0.7277155077039966` | `0.8425830323796916` | `64` | `0` |
| `APGT7-ConservativeSourceReplayPlusSmallDelta` | `0.015625` | `-0.6050909957994263` | `0.7436101149776021` | `64` | `0` |
| `APGT8-ShuffledPayloadNegativeControl` | `0.0` | `-0.5718771643447375` | `0.9205565823304426` | `64` | `0` |

判断：APGT 的 template diversity 和 engineering/materializer 都闭合，但 outcome 完全没有打开。generated route 停在 `apgt_generated_frontier_fail`，不能 controller。

## 12. P13-P15 boundary

P13：

```text
p13_apgt_certificate_controller_v9650.csv = not_run
reason = P12_APGT_official_candidate_failed
apgt_certificate_pass = 0
apgt_controller_pass = 0
```

P14：

```text
p14_system_controller_route_decision_v9650.csv
route = R4-NoTemplateBalancedTarget
system_legal_controller_pass = 0
```

P15：

```text
p15_leaveout_paired_replay_boundary_v9650.csv = not_run
reason = system_controller_not_official
leave_dataset_out/leave_stratum_out/leave_template_out = 0/0/0
paired_replay_pass = 0
short_run_boundary_open/full_run_boundary_open = 0/0
```

判断：没有 controller/runtime/APGT official pass，因此 paired replay 和 short/full training 都未打开。

## 13. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9650.csv
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

## 14. Figures

本轮额外落盘的诊断图：

```text
fig_p1_lineage_hierarchy_sunburst.svg
fig_p1_candidate_template_expansion_hist.svg
fig_p1_action_to_template_sankey.svg
fig_p1_entropy_by_lineage_level.svg
fig_p1_template_collision_table.svg
fig_p2_lto_precision_drop_by_ranker.svg
fig_p2_template_fold_heatmap.svg
fig_p2_accepted_template_share_bar.svg
fig_p2_value_longrisk_by_template_fold.svg
fig_p3_memory_offdiag_matched_lift.svg
fig_p3_intervention_lift_waterfall.svg
fig_p3_value_direction_cosine_vs_lift.svg
fig_p3_longrisk_drop_by_template.svg
fig_p3_joint_safe_scatter_V_vs_longrisk.svg
fig_p4_target_density_vs_template_count.svg
fig_p4_value_risk_frontier.svg
fig_p4_support_by_template_heatmap.svg
fig_p4_target_lattice_parallel_coordinates.svg
fig_p5_rank_score_hist_by_template.svg
fig_p5_ranker_ablation_waterfall.svg
fig_p5_value_vs_risk_score_scatter.svg
fig_p5_template_quota_map.svg
fig_p5_lto_drop_by_ranker.svg
fig_p6_calibration_vs_heldout_drift.svg
fig_p6_accepted_region_by_template.svg
fig_p6_threshold_sensitivity_grid.svg
fig_p6_precision_value_longrisk_frontier.svg
fig_p9_source_to_generated_damage_matrix.svg
fig_p9_value_direction_cosine_hist.svg
fig_p9_longrisk_created_by_primitive.svg
fig_p9_memory_offdiag_delta_by_primitive.svg
fig_p9_template_diversity_source_vs_generated.svg
fig_p12_apgt_gradeab_by_primitive.svg
fig_p12_apgt_value_longrisk_frontier.svg
fig_p12_apgt_template_diversity.svg
fig_p12_apgt_damage_waterfall.svg
```

判断：这些图只用于复核诊断，不构成 official pass。

## 15. No-fake audit

```text
rows_checked = 26814
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
v9640_boundary_pass = 1
lineage_hierarchy_pass = 1
template_stability_pass = 1
memory_offdiag_causal_pass = 1
template_balanced_target_pass = 0
template_deconfounded_ranker_pass = 1
rank_safe_certificate_pass = 0
existing_controller/selected_runtime = 0/0
damage_decomposition_pass = 1
apgt_implementation_pass = 1
apgt_branch_horizon_pass = 1
apgt_weak_pass = 0
apgt_controller_pass = 0
system_legal_controller_pass = 0
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
uses_old_table_for_official = 0
payload_norm_bucket_used_as_selector = 0
candidate_template_id_direct_selector = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R4-NoTemplateBalancedTarget
F0_boundary_reproduction_failed = 0
F1_lineage_hierarchy_incomplete = 0
F2_template_collapsed_accepted_region = 0
F3_memory_offdiag_causality_unresolved = 0
F4_no_template_balanced_target = 1
F5_legal_rank_template_unstable = 0
F8_apgt_generated_frontier_fail = 0
F10_system_not_official = 1
primary_blocker = template_balanced_target_absent
```

说明：failure taxonomy 的主路由优先级停在 R4，所以 `F8_apgt_generated_frontier_fail = 0`；但 route JSON 仍记录 secondary blocker = `apgt_generated_frontier_fail`，P12 也明确 `apgt_weak_pass = 0`。

## 16. Hash

| artifact | SHA256 |
|---|---|
| plan | `cae1fd7993c81b335eeb1443dbc28f30e8b77932aa1fcea55d3abb9d9633f84b` |
| runner | `a584a1c0f8db588a9769b454a100d3a062767a03326889b8ccd5bf50a82c9dfe` |
| run manifest | `f3bf1d0dadefb0fc4a9b682835f14aa7734282964051496e19c11c8640cc22c8` |
| route | `c9494904982daf4d314c312a38bc34a939f3d414a2b156cb81ac6576592341ef` |
| P0 boundary | `c6f01831040bc9efd8816688246c6e560d7d717ee9789721af45e7fd59ae4263` |
| P1 lineage | `a91bc7e37ddec090c619e8eefde488e7e97b939292a41b35bc701cab07237bf2` |
| P2 LTO | `f8e23a10da3bbd30bdf2f47eebb8e8cc040cc1a309ff5723462b27f2d90b6785` |
| P3 memory/offdiag | `a5954ae4477119156fc000d6c20a41705bf07fe2f9a64a8812da346abf343ef4` |
| P3 intervention trace | `a470e41a8d58948f08019560e80927ecc2bb4eeca77bc2a19ebaee08e85e1195` |
| P4 target map | `8281b950d711a60f098823dfe08b1817cd7ee0e8cc29b9330ca331d39d1ccd40` |
| P5 ranker | `4de86113500528bf7f08491eb333d605909bb47c5abc3ab652fcc76a3f226b7f` |
| P6 certificate | `1d9f52feff724215a8819139e0e4259855d69d160a99e15aa900764c0148de16` |
| P7 controller | `1cb19cd239a115324871954b1ba607af51244283a9fd369d344a4218e13abcbc` |
| P8 runtime | `3f66aad2a8ba9e838f0566a6262ad569eef581d36ea97907e173ad518a738520` |
| P9 damage | `28938901756c6d31172e48b95a8d38fc8b976a3eb7ed5a5a2d6eff611964837e` |
| P10 APGT implementation | `06cbf8901a1fdd0da2442e66b784ced755c0e77087af19c53be8359c51d7b3c7` |
| P11 APGT smoke | `de6542dcc8040b01bf1df703e1b63c296672a83da5b9d0423e156a8cbbcf6758` |
| P12 APGT outcome | `1442405fef6f49884bb2d885953de81ff2ae4aaa5f21417674a09f0bd4212b6c` |
| P13 APGT controller | `f585f3ed242cd3d2e2319d95a8c644eca24cb8a59a8f2e5566146a6c961933ae` |
| P14 route CSV | `5f8a4d2074e12fcd23db682a9477cc39311f60fbdaf7aca43ff9fb28a05efcc6` |
| P15 boundary | `ca8af716a074e3e662970f4795dd63bb3ab7ab077122524e4d48002cf988b70a` |
| Base-Acc Sentinel | `ab42e84e1ebf9e57ea1dbbc4355e8cefd7a19188dc7bdfc1343d06727dd999b4` |
| no-fake audit | `4423ec1e5c7d2b0e09e6255bc17372f3464350f1c92a6f1e4e26525dd00fa3b5` |
| contract audit | `edb8b65fb26675e2c5ea41fe2eface273ba92e77bc0d8a9309feb37ecbefe7ac` |
| failure taxonomy | `2c3aac4fefc6cfb8f675aa3177f18897236defe0a2a364f642e84c8f4df2f33a` |

## 17. 最终分析结论

v9.6.5 的真实推进是：

```text
v9.6.4:
  accepted region 在旧 candidate_id/template 口径下看起来完全塌缩；
  memory/offdiag causality 未过；
  APGL generated frontier 失败。

v9.6.5:
  重建 candidate-template lineage hierarchy；
  证明 accepted actions 不是 strict/action 层 fake diversity，也不是新 template 口径下的单模板塌缩；
  accepted-region leave-template-out 和 memory/offdiag causal gate 均过；
  template-deconfounded ranker R5B 过；
  但 template-balanced target map 没有 official target，rank-safe certificate 也因 LDO drop 失败；
  APGT1-APGT8 工程链路、P3 intervention replay 和 APGT 12288 行 branch-horizon replay 均完整闭合；
  但 APGT outcome 仍 value-negative / high-longrisk，没有 generated frontier。
```

机制判断：

1. H0 成立：v9.6.4 boundary 被复现，没有跳过旧 lineage-collapse / APGL failure。
2. H1 成立并修正前序判断：旧 `candidate_id` 确实塌缩，但新 `candidate_template_id` hierarchy 下 unique template = `87`。
3. H2 成立：accepted-region LTO 稳定性过，template pocket 不再是本轮 terminal blocker。
4. H3 成立于 matched-pair 层：memory/offdiag safe 的 GradeAB/V/longrisk lift 均满足本轮 gate；但 intervention clone 本身没有产生额外 lift。
5. H4 未成立：没有 official template-balanced target；best target 的 bad/null 与 LDO/group stability 不过。
6. H5 部分成立：R5B ranker pass，但 frozen certificate 因 LDO drop = `0.3793` 失败，不能 controller。
7. H6/H7 未打开：existing-action controller 和 selected runtime 均 not_run。
8. H8 成立于 diagnostic 层：APGL/APGH damage 主要是 longrisk-created，hard longrisk veto 仍是必需。
9. H9/H10 成立于 implementation 层：APGT payload/certificate/action apply 与 branch-horizon materializer 完整闭合。
10. H11 未成立：APGT 没有生成 value-positive / horizon-safe frontier；best APGT4 的 V LCB 为负且 longrisk UCB 高。
11. H12-H14 未打开：没有 controller/runtime/APGT official pass，就不能打开 paired replay 或 short/full training。
12. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.6.5 真实执行后停在 `R4-NoTemplateBalancedTarget`：新的 template-lineage hierarchy 解除了一部分 v9.6.4 的 candidate-template 塌缩疑问，memory/offdiag causal signal 也首次过 gate；但 official template-balanced target 仍不存在，certificate 被 LDO drop 否决，APGT 虽完整落盘却继续生成 value-negative / high-longrisk 动作，因此 strict PureKAN functional 仍未成功。
