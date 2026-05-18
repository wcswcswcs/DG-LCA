# DG-KAN v9.6.6 Dataset-Invariant Template Target / LDO-Robust Controller / Memory-Offdiag Generator Stop 实验复盘

> 本复盘记录 `DG-KAN_v9.6.6_DatasetInvariantTemplateTarget_LDORobustController_MemoryOffdiagGeneratorStop_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 dataset-LDO diagnostic、memory/offdiag veto diagnostic、rank diagnostic、generated-route stop-rule、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R2-NoTemplateBalancedTargetStill
base_candidate = LQ-t2-h256
success_v9660_strict_purekan_functional = False
success_v9660_full_functional = False
success_v9660_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop_first_20260515T160000Z/
```

核心结论：

1. P0 复现 v9.6.5 boundary：source route = `R4-NoTemplateBalancedTarget`，lineage / memory-offdiag causal / ranker pass = `1 / 1 / 1`，system pass = `0`，no-fake/no-proxy = `1 / 1`。
2. P1 Dataset-LDO failure autopsy 过：LDO primary axis = `dataset_id`，dominant heldout dataset = `Fashion-MNIST`，dominant precision drop = `0.37931034482758624`，explained fraction = `1.0`。
3. P1 不是 dataset dispatch：`no_forbidden_dataset_specific_selector_produced = 1`；dataset 只用于 leaveout/autopsy。
4. P1 dataset support 显示 shift 明显：R5B TopK87 accepted count Fashion-MNIST / KMNIST / MNIST = `34 / 36 / 17`，对应 precision = `1.0 / 0.7777777777777778 / 0.8235294117647058`。
5. P2 Core + Expansion target lattice 未过：best = `T2.1-CoreExpansionValueNoLongRisk`，accepted = `97`，coverage = `0.03372739916550765`，GradeAB = `0.8144329896907216`，V LCB = `0.16622185363738462`，longrisk UCB = `0.0`；但 bad/null UCB = `0.1505214153590923 / 0.18923506172101434`，LDO drop = `0.41379310344827586`。
6. P2 clean targets 仍稀疏：`T2.2-CoreClean` accepted = `79`，`T2.3-MemoryOffdiagCore` accepted = `77`，虽然 bad/null/longrisk 都干净，但 coverage < `0.03` 且 LDO 仍高。
7. P3 memory/offdiag veto usable gate 过：matched joint GradeAB lift = `0.19143576826196473`，V lift = `0.29043934585539005`，longrisk drop = `0.906693153540672`，legal proxy AUC_longrisk = `0.8740889686131543`。
8. P3 risk-veto TopK removal 本身不强：TopK longrisk before/after veto = `21 / 19`，removal fraction = `0.09523809523809523`；P3 pass 主要来自 matched-pair evidence + legal proxy AUC。
9. P4 dataset-invariant ranker 未过：best = `R6E-GroupAdversarialScoreNormalization`，TopK87 precision = `0.45977011494252873`，V LCB = `0.0022189569478391763`，longrisk UCB = `0.15267457786429023`，LDO drop = `0.16091954022988503`。
10. P4 的 value/risk/memory/offdiag 分离降低了部分 LDO，但 TopK quality 不足；没有 ranker 同时满足 precision/value/risk/bad/null/LDO/LSO/LTO。
11. P5 rank-safe certificate v14 未过：best = `CERT14A-fixed-topK87`，heldout accepted = `87`，coverage = `0.030250347705146036`，precision = `0.45977011494252873`，V LCB = `0.0022189569478391763`，longrisk UCB = `0.15267457786429023`，ECE = `0.5601241178872637`。
12. P6/P7 gate-blocked：existing-action controller 与 selected runtime 均 not_run。
13. P8 generated route stop-rule 触发：APGH/APGL/APGT 连续满足 GradeAB < `0.10`、V LCB < `0`、longrisk UCB > `0.50`、new positive < `0.05`、longrisk created > `0.50`。
14. P8 damage consolidation：dominant damage = `D7-longrisk-created`，mean longrisk created rate = `0.7275390625`，mean Damage V LCB = `-0.6056825377331743`。
15. P9 APGU 按计划 not_run：reason = `P8_generated_route_blind_variant_stop_triggered`；没有生成 fake/proxy APGU rows。
16. P10-P12 gate-blocked：没有 system controller、leaveout paired replay 或 short/full boundary。
17. Base-Acc Sentinel 继续健康：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
18. 当前 primary blocker：`template_balanced_target_absent`；secondary blocker：`generated_route_blind_variant_stop`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop.py` | v9.6.6 runner；读取 v9.6.5/v9.6.4/v9.6.3/v9.6.2 artifacts 与 AP0 ledger，执行 Dataset-LDO autopsy、Core+Expansion target lattice、memory/offdiag veto audit、dataset-invariant ranker、certificate v14、generated stop-rule、APGU boundary 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop.py
```

正式运行：

```bash
python experiments/run_v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop.py \
  --out-dir results/real_rerun_20260506/v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop_first_20260515T160000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --apgu-actions-per-primitive 64
```

运行结果：

```json
{
  "best_ranker_id": "R6E-GroupAdversarialScoreNormalization",
  "best_target_id": "T2.1-CoreExpansionValueNoLongRisk",
  "dominant_heldout_dataset": "Fashion-MNIST",
  "generated_route_blind_variant_stop": 1,
  "out_dir": "results/real_rerun_20260506/v9660_dataset_invariant_template_target_ldo_robust_controller_memory_offdiag_generator_stop_first_20260515T160000Z",
  "primary_blocker": "template_balanced_target_absent",
  "rank_safe_certificate_v14_pass": 0,
  "route": "R2-NoTemplateBalancedTargetStill",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 proxy rows。P8 stop-rule 触发后，P9 APGU 按计划记录 `not_run`，没有生成 APGU fake/proxy rows。

## 2. Route

`route_decision_v9660.json` 摘要：

```json
{
  "route": "R2-NoTemplateBalancedTargetStill",
  "source_route_v9650": "R4-NoTemplateBalancedTarget",
  "p0_pass": 1,
  "p1_ldo_failure_autopsy_pass": 1,
  "LDO_primary_failure_axis": "dataset_id",
  "dominant_heldout_dataset": "Fashion-MNIST",
  "explained_LDO_drop_fraction": 1.0,
  "template_balanced_target_pass": 0,
  "best_target_id": "T2.1-CoreExpansionValueNoLongRisk",
  "best_target_accepted_count": 97,
  "best_target_LDO_drop": 0.41379310344827586,
  "memory_offdiag_veto_usable_pass": 1,
  "legal_proxy_AUC_longrisk": 0.8740889686131543,
  "dataset_invariant_ranker_pass": 0,
  "best_ranker_id": "R6E-GroupAdversarialScoreNormalization",
  "best_ranker_TopK87_precision": 0.45977011494252873,
  "best_ranker_LDO_drop": 0.16091954022988503,
  "rank_safe_certificate_v14_pass": 0,
  "best_certificate_id": "CERT14A-fixed-topK87",
  "best_certificate_precision": 0.45977011494252873,
  "best_certificate_LDO_drop": 0.16091954022988503,
  "existing_action_controller_pass": 0,
  "selected_runtime_pass": 0,
  "generated_route_blind_variant_stop": 1,
  "generated_stop_recent_GradeAB": "0.03125",
  "generated_stop_recent_V_LCB": "-0.6095415439891327",
  "generated_stop_recent_longrisk_UCB": "0.7869099970100811",
  "apgu_implementation_pass": 0,
  "apgu_weak_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "template_balanced_target_absent",
  "secondary_blocker": "generated_route_blind_variant_stop"
}
```

判断：v9.6.6 的 selection route 已把 LDO failure 归因到 dataset axis，但 P2 仍没有找到满足 coverage、bad/null、value、longrisk、LDO/LSO/LTO 的 target，因此主 route 停在 `R2-NoTemplateBalancedTargetStill`。同时 generated route 已触发 stop-rule，作为 secondary blocker 记录。

## 3. P0 v9.6.5 boundary

Artifacts：

```text
p0_boundary_reproduction_v9660.csv
p0_field_legality_audit_v9660.csv
```

Summary：

```text
source_route_v9650 = R4-NoTemplateBalancedTarget
lineage_hierarchy_pass_v9650 = 1
memory_offdiag_causal_pass_v9650 = 1
template_balanced_target_pass_v9650 = 0
template_deconfounded_ranker_pass_v9650 = 1
rank_safe_certificate_v13_pass_v9650 = 0
apgt_weak_pass_v9650 = 0
system_legal_controller_pass_v9650 = 0
no_fake/no_proxy = 1 / 1
forbidden_field_count = 0
outcome_field_used_count = 0
p0_pass = 1
```

判断：P0 pass。v9.6.6 没有跳过 v9.6.5 的 target/certificate/APGT boundary，也没有把 forbidden fields 带入 official path。

## 4. P1 Dataset-LDO failure autopsy

Artifacts：

```text
p1_ldo_failure_autopsy_v9660.csv
p1_dataset_score_shift_trace_v9660.csv
```

Summary：

```text
dataset_count = 3
CERT13_base_precision = 0.8735632183908046
CERT13_LDO_drop_v9650 = 0.37931034482758624
LDO_primary_failure_axis = dataset_id
dominant_heldout_dataset = Fashion-MNIST
dominant_precision_drop = 0.37931034482758624
explained_LDO_drop_fraction = 1.0
dominant_support_shortfall_reason = dataset_target_density_shift
no_forbidden_dataset_specific_selector_produced = 1
p1_ldo_failure_autopsy_pass = 1
```

Per dataset：

| dataset | action count | GradeAB count | T4.2 count | R5B TopK87 overlap | accepted precision | V LCB |
|---|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | `1226` | `36` | `36` | `34` | `1.0` | `0.2185522558920942` |
| KMNIST | `828` | `28` | `28` | `36` | `0.7777777777777778` | `0.053201019604910284` |
| MNIST | `822` | `15` | `15` | `17` | `0.8235294117647058` | `0.05691790232751366` |

判断：P1 解释了 v9.6.5 的 LDO drop。问题不是 template collapse，而是 dataset support / score distribution shift；Fashion-MNIST leaveout 是最大 drop。该诊断没有产生 dataset-specific selector。

## 5. P2 Core + Expansion target lattice

Artifacts：

```text
p2_core_expansion_target_lattice_v9660.csv
p2_target_leaveout_grid_v9660.csv
```

Summary：

```text
target_candidate_count = 7
target_pass_count = 0
best_target_id = T2.1-CoreExpansionValueNoLongRisk
best_accepted_count = 97
best_coverage = 0.03372739916550765
best_GradeAB_precision = 0.8144329896907216
best_V_integrated_LCB = 0.16622185363738462
best_h240_longrisk_UCB = 0.0
best_bad_UCB = 0.1505214153590923
best_null_UCB = 0.18923506172101434
best_LDO_drop = 0.41379310344827586
template_balanced_target_pass = 0
```

Representative targets：

| target | accepted | coverage | GradeAB | V LCB | longrisk | bad/null UCB | LDO drop | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `T2.1-CoreExpansionValueNoLongRisk` | `97` | `0.03372739916550765` | `0.8144329896907216` | `0.16622185363738462` | `0.0` | `0.1505214153590923 / 0.18923506172101434` | `0.41379310344827586` | `0` |
| `T2.2-CoreClean` | `79` | `0.027468706536856746` | `1.0` | `0.16201929527024647` | `0.0` | `0.0 / 0.0` | `0.41379310344827586` | `0` |
| `T2.3-MemoryOffdiagCore` | `77` | `0.026773296244784424` | `1.0` | `0.16402807605399972` | `0.0` | `0.0 / 0.0` | `0.39080459770114945` | `0` |

判断：P2 是本轮 primary blocker。Core targets 很干净但 coverage 不够；扩展到 count/coverage 后 bad/null 和 LDO 失败。没有 official template-balanced target。

## 6. P3 memory/offdiag veto audit

Artifact：

```text
p3_memory_offdiag_veto_audit_v9660.csv
```

Summary：

```text
matched_pair_joint_lift_GradeAB = 0.19143576826196473
matched_pair_joint_lift_V = 0.29043934585539005
matched_pair_longrisk_drop = 0.906693153540672
intervention_clone_positive_lift_required = 0
intervention_GradeAB_lift_joint_safe = 0.0
legal_proxy_AUC_longrisk = 0.8740889686131543
TopK_longrisk_before_veto = 21
TopK_longrisk_after_veto = 19
TopK_risk_veto_removes_longrisk_fraction = 0.09523809523809523
memory_offdiag_veto_usable_pass = 1
```

Veto rows：

| veto | accepted | GradeAB | V LCB | longrisk UCB | bad/null UCB |
|---|---:|---:|---:|---:|---:|
| `memory_safe_indicator` | `511` | `0.1506849315068493` | `-0.21136599256421748` | `0.0` | `0.06993445527354111 / 0.20494815148133042` |
| `offdiag_safe_indicator` | `397` | `0.19395465994962216` | `-0.21928308932569182` | `0.0` | `0.0 / 0.0` |
| `joint_memory_offdiag_safe` | `397` | `0.19395465994962216` | `-0.21928308932569182` | `0.0` | `0.0 / 0.0` |

判断：memory/offdiag 更适合作为 veto / risk guard，而不是单独 accepted target 或 generator recipe。matched-pair evidence 强，但 standalone safe set 的 V LCB 仍为负。

## 7. P4 dataset-invariant ranker

Artifacts：

```text
p4_dataset_invariant_ranker_v9660.csv
p4_ranker_ablation_v9660.csv
```

Summary：

```text
ranker_count = 8
ranker_pass_count = 0
best_ranker_id = R6E-GroupAdversarialScoreNormalization
best_TopK87_precision = 0.45977011494252873
best_TopK87_V_LCB = 0.0022189569478391763
best_TopK87_longrisk_UCB = 0.15267457786429023
best_LDO_drop = 0.16091954022988503
best_LSO_drop = 0.011494252873563204
best_LTO_drop = 0.0
dataset_invariant_ranker_pass = 0
```

Representative rankers：

| ranker | TopK87 precision | V LCB | longrisk UCB | LDO drop | pass |
|---|---:|---:|---:|---:|---:|
| `R6A-ValueOnlyBaseline` | `0.39080459770114945` | `-0.0285391768504236` | `0.33129982990159157` | `0.13793103448275862` | `0` |
| `R6B-ValueRankLongRiskVeto` | `0.3103448275862069` | `-0.05923287102722675` | `0.0` | `0.05747126436781608` | `0` |
| `R6C-ValueRankLongRiskBadNullVeto` | `0.25287356321839083` | `-0.08320247588604007` | `0.0` | `0.03448275862068967` | `0` |
| `R6E-GroupAdversarialScoreNormalization` | `0.45977011494252873` | `0.0022189569478391763` | `0.15267457786429023` | `0.16091954022988503` | `0` |

判断：risk/memory/offdiag veto 能降低 longrisk 或 LDO 的一部分，但会显著牺牲 precision/value。P4 没有形成 deployable dataset-invariant ranker。

## 8. P5 rank-safe certificate v14

Artifacts：

```text
p5_rank_safe_certificate_v14.csv
p5_certificate_threshold_sensitivity_v9660.csv
```

Summary：

```text
certificate_count = 6
certificate_pass_count = 0
best_certificate_id = CERT14A-fixed-topK87
best_ranker_id = R6E-GroupAdversarialScoreNormalization
best_accepted_count_heldout = 87
best_coverage_heldout = 0.030250347705146036
best_GradeAB_precision_heldout = 0.45977011494252873
best_V_integrated_LCB_heldout = 0.0022189569478391763
best_h240_longrisk_UCB_heldout = 0.15267457786429023
best_LDO_drop = 0.16091954022988503
best_LSO_drop = 0.011494252873563204
best_LTO_drop = 0.0
rank_safe_certificate_v14_pass = 0
```

Best certificate row：

```text
CERT14A-fixed-topK87:
  accepted_count_heldout = 87
  coverage_heldout = 0.030250347705146036
  GradeAB_precision_heldout = 0.45977011494252873
  V_integrated_LCB_heldout = 0.0022189569478391763
  h240_longrisk_UCB_heldout = 0.15267457786429023
  bad/null UCB = 0.10637805008576995 / 0.0899864912057361
  ECE = 0.5601241178872637
  rank_controller = 1
```

判断：P5 未过。即使把 P4 best ranker frozen 成 accepted region，precision/value/risk 与 LDO 都不够，不能打开 existing-action controller。

## 9. P6-P7 controller / runtime

P6：

```text
p6_existing_action_controller_v9660.csv = not_run
reason = P5_certificate_not_passed
controller_selected = 0
existing_action_controller_pass = 0
source_controller_pass = 0
```

P7：

```text
p7_selected_runtime_trace_v9660.csv = not_run
reason = P6_controller_not_selected
selected_runtime_pass = 0
```

判断：没有把 P4/P5 diagnostic 写成 controller 或 runtime pass。

## 10. P8 generated stop-rule

Artifacts：

```text
p8_generated_route_stop_rule_v9660.csv
p8_generated_damage_consolidation_v9660.csv
```

Stop-rule summary：

```text
family_count = 3
consecutive_failure_family_count = 3
generated_route_blind_variant_stop = 1
best_recent_family = APGT
best_recent_GradeAB_precision = 0.03125
best_recent_V_LCB = -0.6095415439891327
best_recent_longrisk_UCB = 0.7869099970100811
primary_generated_failure = longrisk_created_high_value_negative
```

Family rows：

| family | best primitive | GradeAB | V LCB | longrisk UCB | new positive | longrisk created | stop |
|---|---|---:|---:|---:|---:|---:|---:|
| APGH | `APGH7-SignalChannelSNRMemoryDelta` | `0.046875` | `-0.48807717058315` | `0.7869099970100811` | `0.015625` | `0.671875` | `1` |
| APGL | `APGL1-SourceReplayPreserver` | `0.03125` | `-0.5827183891309656` | `0.7869099970100811` | `0.015625` | `0.671875` | `1` |
| APGT | `APGT4-TemplateMixtureAnchor` | `0.03125` | `-0.6095415439891327` | `0.7869099970100811` | `0.03125` | `0.671875` | `1` |

Damage consolidation：

```text
damage_family_count = 3
dominant_damage_mode = D7-longrisk-created
mean_longrisk_created_rate = 0.7275390625
mean_Damage_V_LCB = -0.6056825377331743
generated_route_blind_variant_stop = 1
```

判断：P8 是本轮第二个硬结论。APGH/APGL/APGT 连续满足 stop-rule，盲目新增 APG* 小变体应停止；下一步需要重新定义 generator objective，而不是继续小修 primitive。

## 11. P9 APGU boundary

Artifacts：

```text
p9_apgu_primitive_implementation_v9660.csv
p9_apgu_branch_horizon_outcome_v9660.csv
```

Status：

```text
status = not_run
reason = P8_generated_route_blind_variant_stop_triggered
primitive_count = 0
generated_action_count = 0
branch_horizon_rows_actual = 0
apgu_implementation_pass = 0
apgu_weak_pass = 0
```

判断：P9 按计划 gate-blocked。没有在 stop-rule 已触发的情况下继续生成 APGU rows。

## 12. P10-P12 boundary

P10：

```text
system_legal_controller_pass = 0
official_eligible = 0
controller_selected = 0
selected_runtime_pass = 0
no_fake/no_proxy/no_dataset_dispatch = 1/1/1
```

P11-P12：

| artifact | status / reason |
|---|---|
| `p11_leaveout_paired_replay_boundary_v9660.csv` | `not_run`, `P10_system_not_official` |
| `p12_short_full_boundary_v9660.csv` | `not_run`, `P11_paired_replay_not_open` |

判断：没有 controller/runtime 或 generated primitive pass，因此 leaveout、paired replay、short/full 都未打开。

## 13. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9660.csv
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
fig_p1_ldo_drop_by_dataset.svg
fig_p1_score_shift_by_dataset.svg
fig_p1_target_density_by_dataset.svg
fig_p1_memory_offdiag_rate_by_dataset.svg
fig_p1_rank_score_shift_vs_precision_drop.svg
fig_p1_dataset_family_template_heatmap.svg
fig_p2_core_expansion_frontier.svg
fig_p2_target_frontier_count_vs_risk.svg
fig_p2_core_expansion_tradeoff.svg
fig_p2_count_vs_bad_null_longrisk.svg
fig_p2_target_lattice_parallel_coordinates.svg
fig_p2_ldo_drop_vs_coverage.svg
fig_p2_template_dataset_support_map.svg
fig_p2_reject_reason_stacked_bar.svg
fig_p3_memory_offdiag_lift_matrix.svg
fig_p3_memory_offdiag_veto_pr_curve.svg
fig_p3_memory_offdiag_proxy_vs_true.svg
fig_p3_old_family_fail_by_veto.svg
fig_p3_cover_entropy_by_veto.svg
fig_p4_value_score_vs_risk_veto.svg
fig_p4_veto_waterfall.svg
fig_p4_ranker_leaveout_drop_bars.svg
fig_p4_dataset_macro_micro_precision.svg
fig_p4_template_quota_map.svg
fig_p4_rejected_longrisk_examples.svg
fig_p4_value_rank_risk_veto_scatter.svg
fig_p5_calibration_vs_heldout_drift.svg
fig_p5_threshold_sensitivity_grid.svg
fig_p5_accepted_region_by_dataset_template.svg
fig_p5_precision_value_longrisk_frontier.svg
fig_p5_ece_vs_topk_quality.svg
fig_p8_generated_family_failure_timeline.svg
fig_p8_damage_mode_by_primitive.svg
fig_p8_value_direction_cosine_hist.svg
fig_p8_longrisk_created_waterfall.svg
fig_p8_generated_damage_timeline.svg
```

判断：这些图只用于复核诊断，不构成 official pass。

## 15. No-fake audit

```text
rows_checked = 226
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
v9650_boundary_pass = 1
ldo_failure_autopsy_pass = 1
template_balanced_target_pass = 0
memory_offdiag_veto_pass = 1
dataset_invariant_ranker_pass = 0
rank_safe_certificate_pass = 0
existing_controller/selected_runtime = 0/0
generated_route_stop = 1
apgu_implementation_pass = 0
system_legal_controller_pass = 0
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
route = R2-NoTemplateBalancedTargetStill
F0_boundary_reproduction_failed = 0
F1_dataset_LDO_failure_unexplained = 0
F2_no_template_balanced_target = 1
F3_memory_offdiag_veto_not_usable = 0
F4_legal_rank_LDO_unstable = 0
F5_certificate_accepted_region_fail = 0
F6_runtime_fail = 0
F7_existing_system_pass = 0
F8_generated_route_stop = 1
F9_generated_primitive_pass = 0
F10_system_not_official = 1
F11_base_acc_catastrophic = 0
primary_blocker = template_balanced_target_absent
```

说明：route 主优先级停在 R2，因此 generated stop-rule 作为 secondary blocker 记录；但 P8 明确 `generated_route_blind_variant_stop = 1`。

## 16. Hash

| artifact | SHA256 |
|---|---|
| plan | `5f308f86ba35e4391aad553f2fe49fa7cb83060466df8253e601b55813034415` |
| runner | `4bd712b2945fe9d1cd8883ee038726859c9a78d0d951bcb7dae6757f679efc2b` |
| run manifest | `e4d2a715778327d655702da0f3d57f8cd9852845d0a0306d6bcc10140ed062bc` |
| route | `70716c913d6a7d2761ffcdee9a5308956cbcd35dfe63101913c6f10688e29f9a` |
| P0 boundary | `ee4389e83e379299f95039bce0e914faf38c6a1070da3eec790dbb8bbec28ba7` |
| P0 field legality | `e60f6aae0ad8a8ab6a9a7efa60974bb492775bd4a24c1cb88c2683490d86c50d` |
| P1 LDO autopsy | `d360f915cccdcf651d8a5b06ae3b3f91afb5b98923f88f6833b4805ceceeb813` |
| P1 score shift | `c4ac57aced106b863c660d7e413f78757c6e8083bae21e7f1c4b43a757740587` |
| P2 target lattice | `734963fd4564a61c3bc996b47a86cae38dd5408ab6eb7c1ab7ce4c79b8053e06` |
| P2 leaveout grid | `62c16ca459e8d9d8614a453b3ff133b3f048c6e4e23a011d11e464dc52d98004` |
| P3 veto audit | `30b45e2b37f7da8e2f165ff8ebbee5fabe16c684a4c819b18f8f6ba675b56bb4` |
| P4 ranker | `2ebdb20a4ccbf0cfb2064daf114222b48738e2d70cc9956d4fc0204db1e17c20` |
| P4 ablation | `2864368500b45c56fa8442c7af3623b28702f8653b365945501773bb1de161e7` |
| P5 certificate | `e3255dabfc1f14b3e891f2a03e985b51f82f57317c1d5815772ab66c7c80f738` |
| P5 sensitivity | `c7850c8688a9230dbd841fcf352c4c2a77395ace4c1ca51139de5ecfecfc70ba` |
| P6 controller | `c83d9b920801af0ffb221906940726794e9e84536d95fd7eb6b5949a65a21b0c` |
| P7 runtime | `27e8d9a5762adeae16acd433b8c5818f129e45320b8e50b05fa88549c025c050` |
| P8 stop-rule | `45e865b2be13692a1fc9523c9d15d9a3e9d2c17154471b8a10ff6ac965abc3f6` |
| P8 damage | `a8b9faead8fdf307b1ac5e82723e1a6a86899010cb037e0a8cc03c8874518a93` |
| P9 APGU implementation | `cd133cb24bffeda27e0cfd060712becb803e6db4d3a4685765975cc079742155` |
| P9 APGU outcome | `a2f23ba41b8d02210b5cbad0e36eca8c4cc03b1d05ea215928bbdf97c8d78a16` |
| P10 system | `b1cad6acde60cb4b1a11f8adcc0dbb4810fdfffa866c0ce2c5d4ac334cbfa0d1` |
| P11 leaveout/paired | `2807f95d5b8d9ca8e529fcbc6ec38076537efe27338dc1fda545910ba4ffd095` |
| P12 short/full | `8025ddf06ceaa26ea813302a68fc3109a1d682f19cf7f9c25f5b745cd663ebfb` |
| Base-Acc Sentinel | `802c4c11c6e00b347d6ab40136131f023c3f8822166805c32627620095672f73` |
| no-fake audit | `b3e4a9fc1ece7cdfb0cb546c1e5cd786b293128023a9bcbbee90aa0a986a8019` |
| contract audit | `d69aaef98f5c20a16d667df22820a078abe5c54171f7f6e61adf28f748f08c50` |
| failure taxonomy | `e5d640bb36542029abe6fa2c67c300685d8fbffe8d06a3ec0eec3ca34bcfcc2a` |

## 17. 最终分析结论

v9.6.6 的真实推进是：

```text
v9.6.5:
  template-lineage hierarchy 修复了旧 candidate-template collapse；
  memory/offdiag causal gate 过；
  但 template-balanced target / certificate / APGT generated frontier 未闭合。

v9.6.6:
  将 v9.6.5 的 LDO failure 拆成 dataset-level support/score shift；
  证明 Fashion-MNIST leaveout 是最大 precision drop 来源；
  Core + Expansion target lattice 仍找不到同时满足 coverage、bad/null、value、longrisk、LDO/LSO/LTO 的 target；
  memory/offdiag signal 作为 veto 有 legal proxy AUC 支持，但单独 veto set 仍不是 high-value accepted region；
  dataset-invariant ranker 牺牲了 TopK quality，无法打开 certificate；
  APGH/APGL/APGT 连续满足 generated stop-rule，APGU 按计划不运行。
```

机制判断：

1. H0 成立：v9.6.5 boundary 被复现，没有跳过 target/certificate/APGT failure。
2. H1 成立：LDO drop 可归因，dominant heldout dataset = `Fashion-MNIST`，explained fraction = `1.0`。
3. H2 未成立：Core + Expansion 没有形成 official template-balanced target；coverage pass 的 target 坏在 bad/null/LDO，clean target 坏在 coverage/LDO。
4. H3 成立于 veto 层：memory/offdiag matched-pair signal 与 legal proxy AUC 支持其作为 risk veto；但它不是 standalone accepted region。
5. H4 未成立：dataset-invariant ranker 没有同时保住 precision/value/risk；best precision 只有 `0.4598`。
6. H5 未成立：rank-safe certificate v14 accepted-region quality 不够，controller 不打开。
7. H6/H7 未打开：existing-action controller 和 selected runtime 均 not_run。
8. H8 成立：generated route blind variant stop-rule 触发，APGH/APGL/APGT 连续 value-negative / high-longrisk / low-positive。
9. H9 未打开：APGU 因 stop-rule not_run，没有生成 fake/proxy rows。
10. H10-H12 未打开：没有 system controller，就不能打开 leaveout paired replay 或 short/full training。
11. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.6.6 真实执行后停在 `R2-NoTemplateBalancedTargetStill`：LDO failure 已被解释为 dataset-level support/score shift，memory/offdiag veto 有可用诊断信号；但仍没有 official template-balanced target 或 certificate，且 APGH/APGL/APGT 连续触发 generated stop-rule，因此 strict PureKAN functional 仍未成功。
