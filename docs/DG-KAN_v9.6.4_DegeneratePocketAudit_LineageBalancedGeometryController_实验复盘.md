# DG-KAN v9.6.4 Degenerate Pocket Audit / Lineage-Balanced Geometry Controller 实验复盘

> 本复盘记录 `DG-KAN_v9.6.4_DegeneratePocketAudit_LineageBalancedGeometryController_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 accepted-region diagnostic、degenerate pocket diagnostic、lineage diagnostic、APGL implementation/smoke、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R2-LineageCollapsedAcceptedRegion
base_candidate = LQ-t2-h256
success_v9640_strict_purekan_functional = False
success_v9640_full_functional = False
success_v9640_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9640_degenerate_pocket_audit_lineage_balanced_geometry_controller_first_20260515T140000Z/
```

核心结论：

1. P0 复现 v9.6.3 boundary：source route = `R2-hidden_multiaxis_pocket_explains_rank`，v9.6.3 system pass = `0`，no-fake/no-proxy = `1 / 1`。
2. P1 degenerate pocket audit 过：degenerate axis count = `7`，forbidden selector axis count = `8`；`payload_norm_bucket` / `action_norm_bucket` / `candidate_origin` 这类 base share = `1.0` 的轴被识别为 degenerate。
3. P1 raw pocket 仍由 `payload_norm_bucket=payload0` 解释，raw drop = `0.8735632183908046`；但它是 degenerate/forbidden selector，不能 official。
4. P1 non-degenerate adjusted pocket 仍存在：best adjusted axis = `offdiag_bucket`，group = `offdiag_fail_0`，base share = `0.13803894297635605`，adjusted drop = `0.8735632183908046`。
5. P2 accepted lineage general audit 看起来健康：accepted action / strict lineage = `87 / 87`，lineage entropy = `4.4659081186545775`，max strict lineage share = `0.011494252873563218`。
6. 但 P2 candidate-template lineage 完全塌缩：candidate_template_unique_count = `1`，candidate_template_max_share = `1.0`，candidate-to-action expansion ratio = `87.0`，candidate_id_collision_count = `86`。
7. 因此本轮 primary blocker 是 `accepted_region_candidate_lineage_collapsed`，route 停在 `R2-LineageCollapsedAcceptedRegion`。
8. P3 memory/offdiag causality 有信号但未过：joint memory/offdiag GradeAB lift = `0.19143576826196473`，V lift = `0.29043934585539005`，longrisk drop = `0.906693153540672`，但 `memory_offdiag_causality_pass = 0`。
9. P4 out-of-pocket target 有 weak signal：best = `T4.1-ValuePositiveNoLongRiskLineageBalanced`，count = `97`，coverage = `0.03372739916550765`，V LCB = `0.16622185363738462`，longrisk UCB = `0.0`；但 max lineage share = `1.0`，official target pass = `0`。
10. P5 group-deconfounded ranker 未过：best = `R4C-value-longrisk-memory-veto`，TopK87 precision = `0.5862068965517241`，V LCB = `0.011510927737868978`，LDO drop = `0.2298850574712643`，max group/lineage share = `1.0 / 1.0`。
11. P6 rank-safe certificate v12 未过：best = `CERT12-FrozenLineageBalancedTopK87`，accepted = `87`，coverage = `0.030250347705146036`，precision = `0.5862068965517241`，V LCB = `0.011510927737868978`，但 group/lineage share 仍塌缩。
12. P7/P8 gate-blocked：existing-action controller 与 selected runtime 均 not_run。
13. P9 APGH damage decomposition 未过 pass，但诊断显示 longrisk/offdiag 仍是主损伤：dominant failure = `D7-longrisk-created`，longrisk created rate = `0.708984375`，offdiag fail delta mean = `0.7890625`。
14. P10/P11 APGL implementation 与 branch-horizon replay 过：generated actions = `512`，rows = `12288 / 12288`，unresolved exception = `0`，quality audit pass = `1`。
15. P12 APGL outcome 未过：best = `APGL1-SourceReplayPreserver`，GradeAB precision = `0.03125`，V LCB = `-0.5827183891309656`，h240 longrisk UCB = `0.7869099970100811`。
16. P13-P16 gate-blocked：没有 APGL official candidate、leaveout、paired replay 或 short/full boundary。
17. Base-Acc Sentinel 继续健康：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
18. 当前 primary blocker：`accepted_region_candidate_lineage_collapsed`；secondary blocker：`apgl_generated_frontier_fail`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9640_degenerate_pocket_audit_lineage_balanced_geometry_controller.py` | v9.6.4 runner；读取 v9.6.3/v9.6.2/v9.6.1/v9.5.9/v9.5.8/v9.5.6 artifacts，执行 degenerate pocket audit、accepted lineage audit、memory/offdiag causality、out-of-pocket target、rank/certificate gate、APGH damage decomposition、APGL primitive 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9640_degenerate_pocket_audit_lineage_balanced_geometry_controller.py
```

正式运行：

```bash
python experiments/run_v9640_degenerate_pocket_audit_lineage_balanced_geometry_controller.py \
  --out-dir results/real_rerun_20260506/v9640_degenerate_pocket_audit_lineage_balanced_geometry_controller_first_20260515T140000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --apgl-actions-per-primitive 64
```

运行结果：

```json
{
  "apgl_generated_actions": 512,
  "apgl_rows": 12288,
  "best_adjusted_axis": "offdiag_bucket",
  "best_apgl_V_integrated_LCB": -0.5827183891309656,
  "best_apgl_primitive": "APGL1-SourceReplayPreserver",
  "best_ranker": "R4C-value-longrisk-memory-veto",
  "best_ranker_TopK87_precision": 0.5862068965517241,
  "candidate_template_unique_count": 1,
  "memory_offdiag_causality_pass": 0,
  "out_dir": "results/real_rerun_20260506/v9640_degenerate_pocket_audit_lineage_balanced_geometry_controller_first_20260515T140000Z",
  "primary_blocker": "accepted_region_candidate_lineage_collapsed",
  "route": "R2-LineageCollapsedAcceptedRegion",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 proxy rows；APGL branch-horizon replay 完整落盘，但没有转成 generated frontier pass。

## 2. Route

`route_decision_v9640.json` 摘要：

```json
{
  "route": "R2-LineageCollapsedAcceptedRegion",
  "source_route_v9630": "R2-hidden_multiaxis_pocket_explains_rank",
  "p0_pass": 1,
  "degenerate_pollution_pass_A": 1,
  "nondegenerate_explanation_pass_B": 1,
  "best_raw_axis": "payload_norm_bucket",
  "best_adjusted_axis": "offdiag_bucket",
  "P2_pass_GENERAL": 1,
  "P2_fail_LINEAGE": 1,
  "accepted_unique_lineage_count": 87,
  "candidate_template_unique_count": 1,
  "candidate_to_action_expansion_ratio": 87.0,
  "memory_offdiag_causality_pass": 0,
  "out_of_pocket_target_pass": 0,
  "group_deconfounded_ranker_pass": 0,
  "rank_safe_certificate_v12_pass": 0,
  "existing_action_controller_pass": 0,
  "selected_runtime_pass": 0,
  "apgh_damage_decomposition_pass": 0,
  "apgl_implementation_pass": 1,
  "apgl_branch_horizon_pass": 1,
  "apgl_weak_pass": 0,
  "apgl_official_candidate_pass": 0,
  "best_apgl_primitive": "APGL1-SourceReplayPreserver",
  "best_apgl_GradeAB_precision": 0.03125,
  "best_apgl_V_integrated_LCB": -0.5827183891309656,
  "best_apgl_h240_longrisk_UCB": 0.7869099970100811,
  "apgl_certificate_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "accepted_region_candidate_lineage_collapsed",
  "secondary_blocker": "apgl_generated_frontier_fail"
}
```

判断：v9.6.4 的关键不是 v9.6.3 accepted-region scope 错误，也不是 APGL engineering failure。P1/P2 证明 accepted region 先被 degenerate axes 污染，再在 candidate-template 层完全塌缩；因此 route 优先停在 `R2-LineageCollapsedAcceptedRegion`。

## 3. P0 v9.6.3 boundary

Artifact：

```text
p0_boundary_reproduction_v9640.csv
```

Summary：

```text
source_route_v9630 = R2-hidden_multiaxis_pocket_explains_rank
source_route_v9620 = R4-PayloadPocketStopped
system_legal_controller_pass_v9630 = 0
accepted_action_count_v9630 = 87
accepted_unique_action_count_v9630 = 87
accepted_unique_candidate_count_v9630 = 1
GradeAB_precision_v9630 = 0.8735632183908046
V_LCB_v9630 = 0.14111334880346277
h240_longrisk_UCB_v9630 = 0.0
best_ranker_v9630 = RANK-D-value-risk-memory-veto
best_certificate_v9630 = CERT11-FrozenTopK87
best_apgh_primitive_v9630 = APGH7-SignalChannelSNRMemoryDelta
best_apgh_V_LCB_v9630 = -0.48807717058315
best_apgh_h240_longrisk_UCB_v9630 = 0.7869099970100811
payload_pocket_route_stop_v9620 = 1
no_fake_v9630/no_proxy_v9630 = 1 / 1
p0_pass = 1
```

判断：P0 pass。v9.6.4 没有跳过 v9.6.3 的 multi-axis pocket / APGH failure boundary。

## 4. P1 degenerate pocket audit

Artifacts：

```text
p1_degenerate_pocket_audit_v9640.csv
fig_p1_base_share_vs_accepted_share.svg
fig_p1_raw_drop_vs_adjusted_drop.svg
fig_p1_degenerate_axis_table.svg
fig_p1_group_entropy_by_axis.svg
```

Summary：

```text
axis_count = 23
degenerate_axis_count = 7
forbidden_selector_axis_count = 8
best_raw_axis = payload_norm_bucket
best_raw_group = payload0
best_raw_base_share = 1.0
best_raw_accepted_share = 1.0
best_raw_drop = 0.8735632183908046
best_adjusted_axis = offdiag_bucket
best_adjusted_group = offdiag_fail_0
best_adjusted_base_share = 0.13803894297635605
best_adjusted_drop = 0.8735632183908046
degenerate_pollution_pass_A = 1
nondegenerate_explanation_pass_B = 1
p1_pass = 1
```

Representative axes：

| axis | top group | base share | accepted share | adjusted drop | degenerate / forbidden |
|---|---:|---:|---:|---:|---:|
| `payload_norm_bucket` | `payload0` | `1.0` | `1.0` | `0.0` | `1 / 1` |
| `action_norm_bucket` | `anorm0` | `1.0` | `1.0` | `0.0` | `1 / 0` |
| `candidate_origin` | `canonical_AP0` | `1.0` | `1.0` | `0.0` | `1 / 1` |
| `memory_bucket` | `memory_fail_0` | `0.17767732962447844` | `1.0` | `0.8505747126436782` | `0 / 0` |
| `offdiag_bucket` | `offdiag_fail_0` | `0.13803894297635605` | `1.0` | `0.8735632183908046` | `0 / 0` |
| `risk_score_bucket` | `risk0` | `0.13803894297635605` | `1.0` | `0.8735632183908046` | `0 / 0` |

判断：P1 把 v9.6.3 的 pocket 解释拆成两层。payload/action/candidate 这类 base share = 1 的轴只能说明 universe 构造，不能作为 selector；但剔除 degenerate 轴后，memory/offdiag/risk pocket 仍然存在。

## 5. P2 accepted lineage audit

Artifacts：

```text
p2_accepted_lineage_audit_v9640.csv
fig_p2_accepted_lineage_sankey.svg
fig_p2_lineage_share_bar.svg
fig_p2_leave_lineage_out_drop.svg
fig_p2_candidate_action_event_map.svg
```

Summary：

```text
accepted_action_count = 87
accepted_unique_candidate_count = 1
accepted_unique_source_payload_hash_count = 87
accepted_unique_lineage_count = 87
accepted_lineage_entropy = 4.4659081186545775
max_lineage_share = 0.011494252873563218
candidate_template_max_share = 1.0
candidate_template_unique_count = 1
candidate_to_action_expansion_ratio = 87.0
candidate_id_collision_count = 86
leave_lineage_out_precision_drop = 0.0014701951349906928
leave_lineage_out_V_drop = 0.006174161580726534
leave_lineage_out_longrisk_increase = 0.0
P2_pass_GENERAL = 1
P2_fail_LINEAGE = 1
```

Lineage rows：

| lineage type | unique count | entropy | max share |
|---|---:|---:|---:|
| `candidate_template` | `1` | `0.0` | `1.0` |
| `source_payload` | `87` | `4.4659081186545775` | `0.011494252873563218` |
| `strict_lineage` | `87` | `4.4659081186545775` | `0.011494252873563218` |
| `event_family` | `87` | `4.4659081186545775` | `0.011494252873563218` |

判断：P2 是本轮 terminal blocker。严格 lineage 看似分散，但 candidate-template 层只有 `1` 个模板扩成 `87` 个 accepted actions。这说明 accepted region 不是 stable controller，而是 candidate template expansion。

## 6. P3 memory/offdiag causality

Artifacts：

```text
p3_memory_offdiag_causality_v9640.csv
fig_p3_memory_offdiag_matched_lift.svg
fig_p3_longrisk_by_memory_offdiag.svg
fig_p3_V_distribution_by_geometry_status.svg
fig_p3_cover_memory_offdiag_scatter.svg
```

Summary：

```text
matched_pair_count_memory = 511
matched_pair_count_offdiag = 397
matched_pair_count_joint = 397
GradeAB_lift_memory_safe = 0.1487279843444227
V_lift_memory_safe = 0.2992236113845136
longrisk_drop_memory_safe = 0.9500635593939639
GradeAB_lift_offdiag_safe = 0.19143576826196473
V_lift_offdiag_safe = 0.29043934585539005
longrisk_drop_offdiag_safe = 0.906693153540672
joint_memory_offdiag_lift = 0.19143576826196473
joint_memory_offdiag_V_lift_LCB = 0.29043934585539005
joint_memory_offdiag_longrisk_drop_UCB = 0.906693153540672
joint_safe_count = 397
joint_safe_GradeAB_precision = 0.19395465994962216
base_GradeAB_precision = 0.027468706536856746
leave_dataset_out_lift_drop = 0.06191769597280811
leave_stratum_out_lift_drop = 0.0044361708668599065
memory_offdiag_causality_pass = 0
```

判断：memory/offdiag 方向不是空信号，尤其 longrisk drop 很强；但 joint lift 仍低于 plan gate，不能转成 causal selector。

## 7. P4-P6 target / rank / certificate

P4 artifact：

```text
p4_out_of_pocket_target_map_v9640.csv
fig_p4_out_of_pocket_target_count.svg
```

P4 summary：

```text
target_variant_count = 6
target_pass_count = 0
target_weak_pass_count = 4
best_target_id = T4.1-ValuePositiveNoLongRiskLineageBalanced
best_target_count = 97
best_target_coverage = 0.03372739916550765
best_V_LCB = 0.16622185363738462
best_longrisk_UCB = 0.0
best_max_lineage_share = 1.0
best_LDO_drop = 0.3103448275862069
best_LSO_drop = 0.0
out_of_pocket_target_pass = 0
out_of_pocket_target_weak_pass = 1
```

P5 artifacts：

```text
p5_group_deconfounded_ranker_v4.csv
fig_p5_ranker_precision_vs_group_drop.svg
fig_p5_value_risk_frontier.svg
fig_p5_rank_score_hist_by_grade.svg
fig_p5_ablation_waterfall.svg
fig_p5_lineage_balanced_topk_map.svg
```

P5 summary：

```text
ranker_count = 8
ranker_pass_count = 0
ranker_weak_pass_count = 0
best_ranker_id = R4C-value-longrisk-memory-veto
best_TopK87_precision = 0.5862068965517241
best_V_LCB = 0.011510927737868978
best_longrisk_UCB = 0.0
best_max_group_share = 1.0
best_max_lineage_share = 1.0
best_LDO_drop = 0.2298850574712643
best_LSO_drop = 0.02298850574712641
best_leave_lineage_out_drop = 0.0
group_deconfounded_ranker_pass = 0
group_deconfounded_ranker_weak_pass = 0
```

P6 artifact：

```text
p6_rank_safe_certificate_v12.csv
```

P6 summary：

```text
certificate_count = 4
certificate_pass_count = 0
best_certificate_id = CERT12-FrozenLineageBalancedTopK87
best_ranker_id = R4C-value-longrisk-memory-veto
best_accepted_count = 87
best_coverage = 0.030250347705146036
best_precision = 0.5862068965517241
best_V_LCB = 0.011510927737868978
best_longrisk_UCB = 0.0
best_max_group_share = 1.0
best_max_lineage_share = 1.0
best_LDO_drop = 0.2298850574712643
best_LSO_drop = 0.02298850574712641
best_LFO_drop = 0.06896551724137923
best_LLO_drop = 0.0
rank_safe_certificate_v12_pass = 0
```

判断：P4-P6 共同说明 out-of-pocket value/risk 信号仍有 weak diagnostic，但一旦要求 lineage/group balance，precision 与 value 明显下降，且 max share 仍为 `1.0`。不能打开 existing-action controller。

## 8. P7-P8 controller/runtime boundary

P7：

```text
p7_existing_action_controller_boundary_v9640.csv = not_run
reason = P6_certificate_not_passed
source_controller_pass = 0
system_controller_candidate_pass = 0
```

P8：

```text
p8_selected_runtime_trace_v9640.csv = not_run
reason = P7_controller_not_selected
selected_runtime_pass = 0
```

判断：没有把 lineage-balanced target、ranker 或 certificate diagnostic 写成 official controller/runtime pass。

## 9. P9 APGH damage decomposition

Artifacts：

```text
p9_apgh_damage_decomposition_v9640.csv
fig_p9_source_to_generated_damage_matrix.svg
fig_p9_value_direction_cosine_hist.svg
fig_p9_longrisk_created_by_transform_step.svg
fig_p9_memory_offdiag_delta_by_primitive.svg
```

Summary：

```text
damage_row_count = 512
dominant_failure_reason = D7-longrisk-created
dominant_failure_assigned_fraction = 0.435546875
dominant_transform_step = risk_veto
Damage_V_LCB = -0.5244681866173412
source_positive_preserved_rate = 0.06640625
new_positive_created_rate = 0.0
longrisk_created_rate = 0.708984375
value_direction_cosine_before_mean = 0.27463294079726497
value_direction_cosine_after_mean = 0.27463294079726497
memory_fail_delta_mean = 0.0
offdiag_fail_delta_mean = 0.7890625
apgh_damage_decomposition_pass = 0
```

判断：APGH damage decomposition 没达到 pass gate，但诊断继续指向 longrisk-created 与 offdiag failure。它支持 APGL 的设计方向，但不能 official。

## 10. P10-P12 APGL primitive

P10 artifact：

```text
p10_apgl_primitive_implementation_v9640.csv
```

P10 summary：

```text
primitive_count = 8
generated_action_count_expected/actual = 512 / 512
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
preflight_all_pass = 1
negative_control_divergence = 1
apgl_implementation_pass = 1
```

P11 artifact：

```text
p11_apgl_branch_horizon_outcome_v9640.csv
```

P11 summary：

```text
branch_horizon_rows_expected/actual = 12288 / 12288
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1
missing_secondary_delta_count = 0
duplicate_row_count = 0
label_exclusivity_violation_count = 0
unresolved_exception_count = 0
rows_per_sec = 15.03784768732412
wallclock_sec = 817.1382138920017
quality_audit_pass = 1
apgl_branch_horizon_pass = 1
```

P12 artifacts：

```text
p12_apgl_geometry_outcome_v9640.csv
fig_p12_apgl_gradeab_by_primitive.svg
```

P12 summary：

```text
best_primitive_id = APGL1-SourceReplayPreserver
best_GradeAB_precision = 0.03125
best_GoodGeometry_A_precision = 0.03125
best_GoodGeometry_B_precision = 0.03125
best_V_integrated_LCB = -0.5827183891309656
best_h240_longrisk_UCB = 0.7869099970100811
best_new_positive_created_rate = 0.015625
best_longrisk_created_rate = 0.671875
negative_control_passed = 0
apgl_weak_pass = 0
apgl_official_candidate_pass = 0
apgl_stop_small_variant = 1
```

Per primitive：

| primitive | GradeAB precision | V LCB | h240 longrisk UCB | longrisk created |
|---|---:|---:|---:|---:|
| `APGL1-SourceReplayPreserver` | `0.03125` | `-0.5827183891309656` | `0.7869099970100811` | `0.671875` |
| `APGL2-PopRiskProjectedEdgeDelta` | `0.03125` | `-0.6154704704212448` | `0.8560881119635937` | `0.75` |
| `APGL3-MemoryOffdiagNullspaceProjection` | `0.015625` | `-0.6434235025818488` | `0.9205565823304426` | `0.828125` |
| `APGL4-CoverEntropyBoundedDelta` | `0.0` | `-0.6757733652125222` | `0.828904255301089` | `0.71875` |
| `APGL5-BoundarySymmetricResidual` | `0.015625` | `-0.6272703361507836` | `0.8954445728577505` | `0.796875` |
| `APGL6-ValueRiskTwoHeadProjectedDelta` | `0.015625` | `-0.5000981767709695` | `0.7869099970100811` | `0.671875` |
| `APGL7-LineageDiverseEnsembleIntersection` | `0.0` | `-0.5898121090341033` | `0.8825326673766913` | `0.78125` |
| `APGL8-ShuffledPayloadNegativeControl` | `0.0` | `-0.6400024085127959` | `0.8825326673766913` | `0.78125` |

判断：APGL implementation/preflight/replay 全部闭合，但 outcome 仍 value-negative / high-longrisk。APGL 没有生成 value-positive / horizon-safe frontier。

## 11. P13-P16 boundary

P13：

```text
p13_apgl_certificate_controller_v9640.csv = not_run
reason = P12_APGL_official_candidate_failed
apgl_certificate_pass = 0
apgl_controller_pass = 0
```

P14-P16：

| artifact | status / reason |
|---|---|
| `p14_leaveout_boundary_v9640.csv` | `not_run`, `controller_runtime_or_apgl_official_not_open` |
| `p15_paired_replay_boundary_v9640.csv` | `not_run`, `P14_leaveout_not_open` |
| `p16_short_full_boundary_v9640.csv` | `not_run`, `P15_paired_replay_not_open` |

判断：没有 controller/runtime/APGL official pass，因此 leaveout、paired replay、short/full 都未打开。

## 12. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9640.csv
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

## 13. Figures

本轮额外落盘的诊断图：

```text
fig_p1_base_share_vs_accepted_share.svg
fig_p1_raw_drop_vs_adjusted_drop.svg
fig_p1_degenerate_axis_table.svg
fig_p1_group_entropy_by_axis.svg
fig_p2_accepted_lineage_sankey.svg
fig_p2_lineage_share_bar.svg
fig_p2_leave_lineage_out_drop.svg
fig_p2_candidate_action_event_map.svg
fig_p3_memory_offdiag_matched_lift.svg
fig_p3_longrisk_by_memory_offdiag.svg
fig_p3_V_distribution_by_geometry_status.svg
fig_p3_cover_memory_offdiag_scatter.svg
fig_p4_out_of_pocket_target_count.svg
fig_p5_ranker_precision_vs_group_drop.svg
fig_p5_value_risk_frontier.svg
fig_p5_rank_score_hist_by_grade.svg
fig_p5_ablation_waterfall.svg
fig_p5_lineage_balanced_topk_map.svg
fig_p9_source_to_generated_damage_matrix.svg
fig_p9_value_direction_cosine_hist.svg
fig_p9_longrisk_created_by_transform_step.svg
fig_p9_memory_offdiag_delta_by_primitive.svg
fig_p12_apgl_gradeab_by_primitive.svg
```

判断：这些图只用于复核诊断，不构成 official pass。

## 14. No-fake audit

```text
rows_checked = 13514
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
v9630_boundary_pass = 1
degenerate_pocket_audit_pass = 1
lineage_audit_general_pass = 1
lineage_collapse_fail = 1
memory_offdiag_causality_pass = 0
out_of_pocket_target_pass = 0
group_deconfounded_ranker_pass = 0
rank_safe_certificate_pass = 0
existing_controller/selected_runtime = 0/0
apgh_damage_decomposition_pass = 0
apgl_implementation_pass = 1
apgl_branch_horizon_pass = 1
apgl_weak_pass = 0
apgl_certificate_pass = 0
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
candidate_direct_selector_used = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R2-LineageCollapsedAcceptedRegion
F0_boundary_reproduction_failed = 0
F1_degenerate_pocket_artifact = 0
F2_lineage_collapsed_accepted_region = 1
F3_memory_offdiag_causal_fail = 0
F4_existing_controller_fail = 0
F5_runtime_blocked = 0
F6_apgl_generated_frontier_fail = 0
F8_system_not_official = 1
primary_blocker = accepted_region_candidate_lineage_collapsed
```

说明：failure taxonomy 的主路由优先级停在 R2，所以 `F6_apgl_generated_frontier_fail = 0`；但 route JSON 仍记录 secondary blocker = `apgl_generated_frontier_fail`，P12 也明确 `apgl_weak_pass = 0`。

## 15. Hash

| artifact | SHA256 |
|---|---|
| plan | `a0ee8732a60f94cc3c66e4b8226cf463b6c85501f310a7870ed45137b4be1d0b` |
| runner | `4a01c906f8e10234c0ffcf8d3e97ae2fd6c30a2b8113fade19c2597cfbb40770` |
| run manifest | `88f6c26ad524b05a34a639a479a6f8662b302d8d01c80ecfd9d1dcd5b60a2f27` |
| route | `92f63d2cf83eb3108063e386084883b7aee9213268fcd5dd6c8ef3195a27c91f` |
| P0 boundary | `6adba24c8ac9c4bfdb6bd4f354dfbc4ed6648fde863753c6023d4abd510ba2a2` |
| P1 degenerate pocket | `c414efd038bb97d04a24327a22844da5b7c61df0f43c0663c9f859a4d4ad64c7` |
| P2 accepted lineage | `a8cb041783b120819ce6cc5217120066ad7a592878a0ec4ec7bf65944af5beaa` |
| P3 memory/offdiag | `fab8245d73f1d9a3a7abcdafde22cb818b5d70454a6bfe4eab0d31c018dc92a1` |
| P4 target map | `4070f88ac867fd2a0eaa4f9e80a41c96a64434c36a2a5cfd7c026576fefd5949` |
| P5 ranker | `ac210204895bb3c9ae72429f5ca3da901f571300bbbee217a2fdd9ad7f17cb09` |
| P6 certificate | `8da3f5ff72c18c6f4d4f6b8d283eae308ec6d53fa828ede42da5db784bd964f8` |
| P7 controller | `4c049e1a877769ebe6416a0abe6bd92c263c914b9f328e47b705bb6cf3999fab` |
| P8 runtime | `eea1972b9d7720f1f338b244448defb2c188fae90f56935b3ffc1f0f99240cba` |
| P9 APGH damage | `a58cc80afe90b58d96b75400444241cbfaf65525b0c001edaac64484cf351bb5` |
| P10 APGL implementation | `208749b43e7bfaaab9f2684987b5ed0d13203d568aec61f30f9f3acd30fbe3ad` |
| P11 APGL smoke | `c9e84c986f96de2d11c1903996855239663486db3258230dcf9925e27630d526` |
| P12 APGL outcome | `ed3acb4b42074a23b6122de737bb6dc83ab4e78e09d12079b46941b67720d165` |
| P13 APGL controller | `b5ac40d9ba2eb0dce6976792215b7b04227d3747fea654a0d78b80a1e310c` |
| P14 leaveout | `88e47ae842863cb7cf64a49ecfc7afafd08ba18a859fbf0db2055dabe52ca893` |
| P15 paired replay | `093b957e95f6f4aaa254704f2b2c827e2d3ff97b19b9a25405bdb7ddb2d5fb76` |
| P16 short/full | `58730adb3efff5ab020e1fc20899aec96a6b66963f7c597e8e31fd20e6cb9928` |
| Base-Acc Sentinel | `9a9b28b2b5c097dea69b631e45a6a2115944f4d4119c9b4134c4299fbd7018c5` |
| no-fake audit | `29c2d42c8f7f697f23bfa620fc1aca0589df38cf84248dde5ca4415cc3c9c6b3` |
| contract audit | `e5af102abfc24b578391b9a92dfd0ed08d7f47a0d63743fc7469eb5a9d34795d` |
| failure taxonomy | `d4ff0f28f530b7094bcb9efe4df638413efd9884fa1e2ffa1457bc64131a990e` |

## 16. 最终分析结论

v9.6.4 的真实推进是：

```text
v9.6.3:
  accepted-region scope 没有 duplicate / universe mismatch；
  但 accepted region 被 multi-axis group pocket 解释；
  APGH generated frontier 失败。

v9.6.4:
  先区分 degenerate axis 与 non-degenerate pocket；
  证明 payload/action/candidate 等 base-share=1 的轴不能作为 selector；
  进一步发现 accepted region 在 candidate-template lineage 层完全塌缩；
  memory/offdiag causality 有接近 gate 的 diagnostic signal，但未过 official；
  out-of-pocket target、group-deconfounded ranker、rank-safe certificate 均无法同时满足 value/risk 与 lineage/group stability；
  APGL1-APGL8 工程链路和 12288 行 branch-horizon replay 完整闭合；
  但 APGL generated actions 仍 value-negative / high-longrisk，没有形成 generated frontier。
```

机制判断：

1. H0 成立：v9.6.3 boundary 被复现，没有跳过 hidden pocket / APGH failure。
2. H1 成立：degenerate pocket pollution 被识别，payload/action/candidate 等轴不能 official。
3. H2 成立于诊断层：剔除 degenerate 轴后，offdiag/memory/risk pocket 仍解释 accepted concentration。
4. H3 成立于 blocker 层：accepted region 在 candidate-template 层只有 `1` 个模板，扩成 `87` 个 accepted actions，因此不能作为 stable controller。
5. H4 部分成立：memory/offdiag safe 的 lift 与 longrisk drop 很强，但 joint GradeAB lift 仍没过 gate。
6. H5 未成立：out-of-pocket target 有 weak signal，但 lineage/group share 仍塌缩。
7. H6 未成立：ranker/certificate 的 precision 和 V LCB 在去 pocket 后不足，且 max share = `1.0`。
8. H7 未打开：existing-action controller 与 runtime 均 not_run。
9. H8 成立于 implementation 层：APGL payload/certificate/action apply 与 branch-horizon materializer 完整闭合。
10. H9 未成立：APGL 没有生成 value-positive / horizon-safe frontier；best APGL1 的 V LCB 为负且 longrisk UCB 高。
11. H10-H12 未打开：没有 controller/runtime/APGL official pass，就不能打开 leaveout、paired replay 或 short/full training。
12. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.6.4 真实执行后停在 `R2-LineageCollapsedAcceptedRegion`：v9.6.3 的强 accepted region 不是 scope 错误，但它在 candidate-template lineage 层完全塌缩；memory/offdiag 信号仍未达到 official causality gate，APGL 虽完整落盘却继续生成 value-negative / high-longrisk 动作，因此 strict PureKAN functional 仍未成功。
