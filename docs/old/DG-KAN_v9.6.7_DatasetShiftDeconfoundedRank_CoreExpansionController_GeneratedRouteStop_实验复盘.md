# DG-KAN v9.6.7 Dataset-Shift Deconfounded Rank / Core-Expansion Controller / Generated Route Stop 实验复盘

> 本复盘记录 `DG-KAN_v9.6.7_DatasetShiftDeconfoundedRank_CoreExpansionController_GeneratedRouteStop_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 dataset-shift diagnostic、core-expansion target diagnostic、legal feature normalization diagnostic、rank/certificate diagnostic、generated-route stop-rule、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R2-DatasetShiftSolvedButTargetAbsent
base_candidate = LQ-t2-h256
success_v9670_strict_purekan_functional = False
success_v9670_full_functional = False
success_v9670_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9670_dataset_shift_deconfounded_rank_core_expansion_controller_generated_route_stop_first_20260515T170000Z/
```

核心结论：

1. P0 复现 v9.6.6 boundary：source route = `R2-NoTemplateBalancedTargetStill`，generated stop-rule status = `1`，system pass = `0`，no-fake/no-proxy = `1 / 1`。
2. P1 Dataset-LDO autopsy v2 完成：dominant LDO failure class = `score_scale_shift`，dominant heldout dataset = `Fashion-MNIST`，dominant precision drop = `0.37931034482758624`。
3. P1 全部 dataset 行完成归因：assigned failure fraction = `1.0`；dominant class fraction = `0.6666666666666666`；没有产生 dataset-specific selector。
4. P1 per-dataset shift 明显：Fashion-MNIST / KMNIST / MNIST 的 R5B score mean = `-6.021749447018721 / -3.732632624783545 / -4.067975926348259`。
5. P1 R5B TopK87 在各 dataset 的 accepted count = `34 / 36 / 17`，precision = `1.0 / 0.7777777777777778 / 0.8235294117647058`。
6. P2 Core + Expansion target lattice v2 未过：strong target pass count = `0`，weak target pass count = `0`。
7. P2 summary best target = `T2.3-MemoryOffdiagCore`，accepted = `77`，coverage = `0.026773296244784424`，GradeAB = `1.0`，V LCB = `0.16402807605399972`，longrisk/bad/null/memory/offdiag UCB = `0.0 / 0.0 / 0.0 / 0.0 / 0.0`，但 coverage < `0.03` 且 LDO drop = `0.39080459770114945`。
8. P2 count 够的 diagnostic target 仍被 LDO/风险否决：`T2.6-R5BHighScoreRiskClean` accepted = `87`，GradeAB = `0.8735632183908046`，V LCB = `0.14111334880346277`，risk/bad/null/memory/offdiag UCB = `0`，但 LDO drop = `0.37931034482758624`。
9. P2 `T2.5-CoreRiskCleanExpansion` accepted = `87`，GradeAB = `0.9080459770114943`，V LCB = `0.15673291999485897`，但 null UCB = `0.15267457786429023`、offdiag UCB = `0.18196532683597333`、LDO drop = `0.41379310344827586`。
10. P3 legal feature normalization / deconfounding 未过：best feature = `F3A-raw-value-proxy`，TopK87 precision = `0.39080459770114945`，V LCB = `-0.0285391768504236`，longrisk UCB = `0.33129982990159157`。
11. P3 风险归一化能压 longrisk 但牺牲 value/precision：`F3D-risk-normalized-value` TopK87 precision = `0.367816091954023`，V LCB = `-0.024880225746581084`，longrisk UCB = `0.0`，LDO drop = `0.09195402298850575`。
12. P4 value-rank + veto ranker 未过：best ranker = `R7A-ValueRankOnly`，accepted = `87`，precision = `0.39080459770114945`，V LCB = `-0.0285391768504236`，longrisk UCB = `0.33129982990159157`。
13. P4 所有 R7 ranker strong/weak pass = `0 / 0`；强 veto ranker 能降低 LDO/longrisk，但 precision/value 明显崩。
14. P5 rank-safe certificate v15 未过：best = `CERT15C-topK64-high-precision`，accepted = `64`，coverage = `0.022253129346314324`，precision = `0.484375`，V LCB = `-0.012902744078215206`，longrisk UCB = `0.37383303237969157`。
15. P6/P7 gate-blocked：existing-action controller 与 selected runtime 均 `not_run`。
16. P8 generated route stop-rule enforcement 过：APGH/APGL/APGT stop condition 均 = `1`，APGU_run = `0`，no fake APGU rows = `1`。
17. P9 generated damage notebook 完成：damage rows = `24`，dominant damage mode = `longrisk_created`，assigned fraction = `1.0`，recommendation = `stop_blind_generated_variants`。
18. P11 gate-blocked：没有 system controller，因此 paired replay / short-full boundary 未打开。
19. Base-Acc Sentinel 继续健康：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
20. 当前 primary blocker：`template_balanced_target_absent`；secondary blocker：`generated_route_stop_triggered`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9670_dataset_shift_deconfounded_rank_core_expansion_controller_generated_route_stop.py` | v9.6.7 runner；读取 v9.6.6/v9.6.5/v9.6.4/v9.6.3 artifacts 与 canonical AP0 ledger，执行 Dataset-LDO autopsy v2、Core+Expansion target lattice v2、legal feature normalization、value/risk/bad/null/memory/offdiag ranker、certificate v15、generated route stop-rule enforcement、damage notebook 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9670_dataset_shift_deconfounded_rank_core_expansion_controller_generated_route_stop.py
```

正式运行：

```bash
python experiments/run_v9670_dataset_shift_deconfounded_rank_core_expansion_controller_generated_route_stop.py \
  --out-dir results/real_rerun_20260506/v9670_dataset_shift_deconfounded_rank_core_expansion_controller_generated_route_stop_first_20260515T170000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --apgu-actions-per-primitive 64
```

运行结果：

```json
{
  "APGU_run": 0,
  "best_certificate_id": "CERT15C-topK64-high-precision",
  "best_ranker_id": "R7A-ValueRankOnly",
  "best_target_id": "T2.3-MemoryOffdiagCore",
  "dominant_LDO_failure_class": "score_scale_shift",
  "generated_route_stop_triggered": "1",
  "out_dir": "results/real_rerun_20260506/v9670_dataset_shift_deconfounded_rank_core_expansion_controller_generated_route_stop_first_20260515T170000Z",
  "primary_blocker": "template_balanced_target_absent",
  "route": "R2-DatasetShiftSolvedButTargetAbsent",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 proxy rows。APGU 因 generated route stop-rule 继续不运行，没有生成 APGU fake/proxy rows。

## 2. Route

`route_decision_v9670.json` 摘要：

```json
{
  "route": "R2-DatasetShiftSolvedButTargetAbsent",
  "source_route_v9660": "R2-NoTemplateBalancedTargetStill",
  "p0_pass": 1,
  "p1_pass": 1,
  "dominant_LDO_failure_class": "score_scale_shift",
  "dominant_heldout_dataset": "Fashion-MNIST",
  "template_balanced_target_pass": 0,
  "template_balanced_target_weak_pass": 0,
  "best_target_id": "T2.3-MemoryOffdiagCore",
  "best_target_accepted_count": 77,
  "best_target_LDO_drop": 0.39080459770114945,
  "p3_feature_pass": 0,
  "best_feature_name": "F3A-raw-value-proxy",
  "ranker_strong_pass": 0,
  "ranker_weak_pass": 0,
  "best_ranker_id": "R7A-ValueRankOnly",
  "best_ranker_precision": 0.39080459770114945,
  "best_ranker_LDO_drop": 0.13793103448275862,
  "rank_safe_certificate_v15_pass": 0,
  "certificate_weak_pass": 0,
  "best_certificate_id": "CERT15C-topK64-high-precision",
  "best_certificate_precision": 0.484375,
  "best_certificate_LDO_drop": 0.140625,
  "existing_action_controller_pass": 0,
  "selected_runtime_pass": 0,
  "generated_route_stop_triggered": "1",
  "APGU_run": 0,
  "p9_damage_notebook_pass": 1,
  "system_legal_controller_pass": 0,
  "primary_blocker": "template_balanced_target_absent",
  "secondary_blocker": "generated_route_stop_triggered"
}
```

判断：v9.6.7 把 dataset shift 诊断继续推进了，但没有得到 official target。目标层仍失败，因此 route 优先停在 `R2-DatasetShiftSolvedButTargetAbsent`；generated stop-rule 作为 secondary blocker 保留。

## 3. P0 boundary

Artifacts：

```text
p0_boundary_reproduction_v9670.csv
p0_field_legality_audit_v9670.csv
```

Summary：

```text
source_v9660_route = R2-NoTemplateBalancedTargetStill
generated_stop_rule_status = 1
no_fake/no_proxy/cpu_offload = 1 / 1 / 0
old_table_official_violation_count = 0
dataset_name_used_by_controller_count = 0
outcome_derived_field_used_count = 0
p0_pass = 1
```

Input hashes from v9.6.6 manifest：

```text
canonical_outcome_table_hash = ee4389e83e379299f95039bce0e914faf38c6a1070da3eec790dbb8bbec28ba7
grade_ledger_hash = 734963fd4564a61c3bc996b47a86cae38dd5408ab6eb7c1ab7ce4c79b8053e06
feature_ledger_hash = 30b45e2b37f7da8e2f165ff8ebbee5fabe16c684a4c819b18f8f6ba675b56bb4
ranker_input_hash = 2ebdb20a4ccbf0cfb2064daf114222b48738e2d70cc9956d4fc0204db1e17c20
certificate_input_hash = e3255dabfc1f14b3e891f2a03e985b51f82f57317c1d5815772ab66c7c80f738
base_acc_sentinel_hash = 802c4c11c6e00b347d6ab40136131f023c3f8822166805c32627620095672f73
```

判断：P0 pass。v9.6.7 没有跳过 v9.6.6 的 target/ranker/certificate/generated stop boundary，也没有把 dataset name 或 outcome-derived field 带入 official path。

## 4. P1 Dataset-LDO failure autopsy v2

Artifacts：

```text
p1_dataset_ldo_failure_autopsy_v2_v9670.csv
p1_dataset_score_shift_trace_v9670.csv
```

Summary：

```text
dataset_count = 3
dominant_LDO_failure_class = score_scale_shift
dominant_failure_class_fraction = 0.6666666666666666
assigned_failure_fraction = 1.0
dominant_heldout_dataset = Fashion-MNIST
dominant_precision_drop = 0.37931034482758624
no_dataset_specific_selector_produced = 1
p1_pass = 1
```

Per dataset：

| dataset | action | T2.1 | T2.2 | T2.3 | GradeAB | bad/null/longrisk | memory/offdiag fail | R5B mean | PSI | TopK87 count/precision | threshold precision | failure |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---|
| Fashion-MNIST | `1226` | `41` | `36` | `34` | `36` | `529 / 33 / 1001` | `0.9347471451876019 / 0.9469820554649266` | `-6.021749447018721` | `0.3477132255617457` | `34 / 1.0` | `1.0` | `score_scale_shift` |
| KMNIST | `828` | `32` | `28` | `28` | `28` | `84 / 99 / 527` | `0.7077294685990339 / 0.7801932367149759` | `-3.732632624783545` | `0.3472940184541398` | `36 / 0.7777777777777778` | `0.3888888888888889` | `score_scale_shift` |
| MNIST | `822` | `24` | `15` | `15` | `15` | `107 / 77 / 567` | `0.7700729927007299 / 0.8175182481751825` | `-4.067975926348259` | `0.244398051883377` | `17 / 0.8235294117647058` | `0.4166666666666667` | `target_density_shift` |

判断：P1 pass。Fashion-MNIST 是最大 LDO drop 的 heldout，但 per-dataset evidence 显示不是一个简单的“该 dataset 没有好动作”问题；score scale shift 和 target density shift 同时存在。dataset 仍只用于 autopsy / leaveout，没有进入 selector。

## 5. P2 Core + Expansion target lattice v2

Artifacts：

```text
p2_core_expansion_target_lattice_v2_v9670.csv
p2_target_leaveout_grid_v9670.csv
```

Summary：

```text
target_candidate_count = 7
strong_target_pass_count = 0
weak_target_pass_count = 0
best_target_id = T2.3-MemoryOffdiagCore
best_accepted_count = 77
best_coverage = 0.026773296244784424
best_GradeAB_precision = 1.0
best_V_integrated_LCB = 0.16402807605399972
best_h240_longrisk_UCB = 0.0
best_bad/null/memory/offdiag UCB = 0.0 / 0.0 / 0.0 / 0.0
best_LDO_drop = 0.39080459770114945
template_balanced_target_pass = 0
```

Representative targets：

| target | accepted | coverage | GradeAB | V LCB | longrisk | bad/null | memory/offdiag | LDO | pass |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| `T2.1-CoreExpansionV2` | `79` | `0.027468706536856746` | `1.0` | `0.16201929527024647` | `0.0` | `0.0 / 0.0` | `0.05995628911411127 / 0.05995628911411127` | `0.41379310344827586` | `0` |
| `T2.3-MemoryOffdiagCore` | `77` | `0.026773296244784424` | `1.0` | `0.16402807605399972` | `0.0` | `0.0 / 0.0` | `0.0 / 0.0` | `0.39080459770114945` | `0` |
| `T2.5-CoreRiskCleanExpansion` | `87` | `0.030250347705146036` | `0.9080459770114943` | `0.15673291999485897` | `0.0` | `0.0 / 0.15267457786429023` | `0.054480608015849245 / 0.18196532683597333` | `0.41379310344827586` | `0` |
| `T2.6-R5BHighScoreRiskClean` | `87` | `0.030250347705146036` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.0 / 0.0` | `0.0 / 0.0` | `0.37931034482758624` | `0` |
| `T2.7-ValuePositiveNoLongRisk` | `97` | `0.03372739916550765` | `0.8144329896907216` | `0.16622185363738462` | `0.0` | `0.1505214153590923 / 0.18923506172101434` | `0.12366038506016863 / 0.28669723614977566` | `0.41379310344827586` | `0` |

判断：P2 是本轮 primary blocker。干净 targets 仍低于 coverage 下界；补足到 87 后，要么 LDO 高，要么 null/offdiag/bad 超标。没有 official template-balanced target。

## 6. P3 legal feature normalization / deconfounding v3

Artifact：

```text
p3_legal_feature_normalization_deconfounding_v3.csv
```

Summary：

```text
feature_count = 7
feature_pass_count = 0
best_feature_name = F3A-raw-value-proxy
best_TopK87_precision = 0.39080459770114945
best_TopK87_V_LCB = -0.0285391768504236
best_TopK87_longrisk_UCB = 0.33129982990159157
best_LDO_drop = 0.13793103448275862
p3_pass = 0
```

Representative features：

| feature | AUC | precision | V LCB | longrisk | bad/null | LDO | pass |
|---|---:|---:|---:|---:|---|---:|---:|
| `F3A-raw-value-proxy` | `0.9267026606264397` | `0.39080459770114945` | `-0.0285391768504236` | `0.33129982990159157` | `0.10637805008576995 / 0.12221253967001162` | `0.13793103448275862` | `0` |
| `F3D-risk-normalized-value` | `0.9320791263695732` | `0.367816091954023` | `-0.024880225746581084` | `0.0` | `0.03389313880385762 / 0.12221253967001162` | `0.09195402298850575` | `0` |
| `F3E-value-minus-memory-offdiag` | `0.8920769540601821` | `0.21839080459770116` | `-0.088285883709398` | `0.0` | `0.0 / 0.12221253967001162` | `0.03448275862068967` | `0` |
| `F3G-event-template-diverse-score` | `0.8967972013414011` | `0.22988505747126436` | `-0.12062607046625373` | `0.0` | `0.0 / 0.0` | `0.09195402298850575` | `0` |

判断：P3 未过。归一化和 risk/memory/offdiag 分离能降低 longrisk/LDO，但无法保住 positive value 和 TopK precision。

## 7. P4 value-rank + bad/null/risk/memory veto ranker

Artifacts：

```text
p4_value_rank_bad_null_risk_memory_veto_ranker_v9670.csv
p4_ranker_ablation_v9670.csv
```

Summary：

```text
ranker_count = 7
ranker_strong_pass_count = 0
ranker_weak_pass_count = 0
best_ranker_id = R7A-ValueRankOnly
best_GradeAB_precision = 0.39080459770114945
best_V_integrated_LCB = -0.0285391768504236
best_longrisk_UCB = 0.33129982990159157
best_bad/null UCB = 0.10637805008576995 / 0.12221253967001162
best_memory/offdiag UCB = 0.3950403220387326 / 0.48127063018855076
best_LDO_drop = 0.13793103448275862
```

Rankers：

| ranker | precision | V LCB | longrisk | bad/null | memory/offdiag | LDO | pass |
|---|---:|---:|---:|---|---|---:|---:|
| `R7A-ValueRankOnly` | `0.39080459770114945` | `-0.0285391768504236` | `0.33129982990159157` | `0.10637805008576995 / 0.12221253967001162` | `0.3950403220387326 / 0.48127063018855076` | `0.13793103448275862` | `0` |
| `R7B-ValueRankLongRiskVeto` | `0.2988505747126437` | `-0.06243342216805068` | `0.0` | `0.0 / 0.10637805008576995` | `0.0 / 0.10637805008576995` | `0.057471264367816105` | `0` |
| `R7D-ValueRiskBadNullMemoryOffdiag` | `0.20689655172413793` | `-0.10493007407513574` | `0.0` | `0.0 / 0.0899864912057361` | `0.0 / 0.0899864912057361` | `0.06896551724137931` | `0` |
| `R7G-CoreExpansionRank` | `0.21839080459770116` | `-0.09711581968231431` | `0.0` | `0.0 / 0.0` | `0.0 / 0.0` | `0.06896551724137931` | `0` |

判断：P4 未过。多重 veto 把风险压下来了，但同时把 precision/value 压坏；没有 legal ranker 同时满足 accepted_count、precision、value、risk、bad/null、memory/offdiag 与 LDO/LSO/LTO。

## 8. P5 rank-safe certificate v15

Artifacts：

```text
p5_rank_safe_certificate_v15.csv
p5_certificate_threshold_sensitivity_v9670.csv
```

Summary：

```text
certificate_count = 5
certificate_strong_pass_count = 0
certificate_weak_pass_count = 0
best_certificate_id = CERT15C-topK64-high-precision
best_ranker_id = R7A-ValueRankOnly
best_accepted_heldout = 64
best_coverage_heldout = 0.022253129346314324
best_precision_heldout = 0.484375
best_V_LCB_heldout = -0.012902744078215206
best_longrisk_UCB_heldout = 0.37383303237969157
best_bad/null UCB = 0.12180505748880108 / 0.09866091513007542
best_LDO_drop = 0.140625
rank_safe_certificate_v15_pass = 0
```

判断：P5 未过。TopK64 提高了 precision，但 accepted_count/coverage 不够，V LCB 仍为负，longrisk 仍高。不能打开 existing-action controller。

## 9. P6-P7 controller / runtime

P6：

```text
p6_existing_action_minimal_controller_v9670.csv = not_run
reason = P5_certificate_strong_pass_failed
existing_action_controller_pass = 0
source_controller_pass = 0
```

P7：

```text
p7_selected_controller_runtime_v9670.csv = not_run
reason = P6_controller_not_selected
selected_runtime_pass = 0
```

判断：没有把 P3/P4/P5 diagnostic 写成 official controller 或 runtime pass。

## 10. P8 generated stop-rule enforcement

Artifact：

```text
p8_generated_route_stop_rule_enforcement_v9670.csv
```

Summary：

```text
generated_route_stop_triggered = 1
APGU_run = 0
APGU_not_run_reason = generated_route_stop_triggered
no_fake_APGU_rows = 1
new_positive_created_rate_mean = 0.020833333333333332
longrisk_created_rate_mean = 0.671875
p8_pass = 1
```

Family rows：

| family | best primitive | GradeAB | V LCB | longrisk UCB | new positive | longrisk created | stop |
|---|---|---:|---:|---:|---:|---:|---:|
| APGH | `APGH7-SignalChannelSNRMemoryDelta` | `0.046875` | `-0.48807717058315` | `0.7869099970100811` | `0.015625` | `0.671875` | `1` |
| APGL | `APGL1-SourceReplayPreserver` | `0.03125` | `-0.5827183891309656` | `0.7869099970100811` | `0.015625` | `0.671875` | `1` |
| APGT | `APGT4-TemplateMixtureAnchor` | `0.03125` | `-0.6095415439891327` | `0.7869099970100811` | `0.03125` | `0.671875` | `1` |

判断：P8 pass。generated route stop-rule 继续成立，APGU 没有被 blind run。

## 11. P9 generated damage mechanism notebook

Artifact：

```text
p9_generated_damage_mechanism_notebook_v9670.csv
```

Summary：

```text
damage_row_count = 24
dominant_damage_mode = longrisk_created
dominant_damage_assigned_fraction = 1.0
stop_go_recommendation = stop_blind_generated_variants
generated_route_stop_triggered = 1
p9_damage_notebook_pass = 1
```

判断：P9 pass。APGH/APGL/APGT 的 primitive-level damage 行都指向 longrisk-created 主导；继续小修 APGU 不是合法下一步。

## 12. P11 system / paired replay boundary

P11：

```text
p11_system_paired_replay_boundary_v9670.csv = not_run
reason = P6_or_P7_strong_pass_failed
system_legal_controller_pass = 0
paired_replay_pass = 0
short_full_boundary_open = 0
```

判断：没有 controller/runtime strong pass，因此 paired replay、short/full training 均未打开。

## 13. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9670.csv
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
fig_p1_per_dataset_target_density_bar.svg
fig_p1_score_distribution_by_dataset.svg
fig_p1_threshold_transfer_heatmap.svg
fig_p1_LDO_drop_waterfall.svg
fig_p1_dataset_bad_null_longrisk_stack.svg
fig_p2_core_expansion_venn.svg
fig_p2_precision_coverage_curve.svg
fig_p2_bad_null_vs_expansion_size.svg
fig_p2_LDO_drop_vs_expansion_size.svg
fig_p2_target_risk_table.svg
fig_p3_feature_shift_before_after.svg
fig_p3_feature_cost_vs_precision.svg
fig_p3_legal_feature_heatmap.svg
fig_p4_ranker_precision_coverage.svg
fig_p4_value_vs_longrisk_scatter.svg
fig_p4_veto_ablation_waterfall.svg
fig_p4_per_dataset_precision_heatmap.svg
fig_p4_ranker_drop_comparison.svg
fig_p5_calibration_vs_heldout.svg
fig_p5_accepted_region_value_risk.svg
fig_p5_leaveout_drop_bar.svg
fig_p5_certificate_threshold_sensitivity.svg
fig_p9_generated_damage_matrix.svg
fig_p9_value_direction_lost_by_primitive.svg
fig_p9_longrisk_created_by_primitive.svg
fig_p9_memory_offdiag_damage_heatmap.svg
```

判断：这些图只用于复核诊断，不构成 official pass。

## 15. No-fake audit

```text
rows_checked = 245
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
v9660_boundary_pass = 1
dataset_ldo_autopsy_pass = 1
template_balanced_target_pass = 0
legal_feature_normalization_pass = 0
ranker_strong/weak pass = 0/0
rank_safe_certificate_pass = 0
existing_controller/selected_runtime = 0/0
generated_route_stop = 1
APGU_run = 0
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
route = R2-DatasetShiftSolvedButTargetAbsent
F0_boundary_reproduction_failed = 0
F1_existing_controller_ready = 0
F2_dataset_shift_solved_but_target_absent = 1
F3_target_exists_but_legal_rank_fails = 0
F4_legal_rank_strong_but_LDO_still_fails = 0
F5_certificate_accepted_region_fails = 0
F6_runtime_fails = 0
F7_generated_route_stopped_existing_action_fails = 0
F8_no_usable_functional_update_route = 0
F9_system_not_official = 1
F10_base_acc_catastrophic = 0
primary_blocker = template_balanced_target_absent
```

说明：route 主优先级停在 R2；generated stop-rule 作为 secondary blocker 记录在 route JSON。

## 16. Hash

| artifact | SHA256 |
|---|---|
| plan | `0b8fe82fef88776ae6120090f18a22aaca23248355e42925565693ed2e5410d2` |
| runner | `0992e234aff6334bb1f2be848b4ca1b815c1eac0db3c41217bb99f71ed19aaab` |
| route | `da371c05bca7563cb435591e187f0896cbcc2d9951e09a5492545cc53c47ab39` |
| P0 boundary | `822f58b0c9ba9f9420f6395b018b676f8d991cb1c8294d45e42bcfa0245b3566` |
| P0 field legality | `ede14075dafe54159154eb3442c9e3f90d8d3727dad85fa4c553306195feb8b4` |
| P1 LDO autopsy | `cc04c45cd33fe7dac47b6191c5f1eae53e7fb4eed928736dbba2ca0373bc9911` |
| P1 score shift | `9e3d137a4fc21072d316aee18dfb47f1229589b254fa2ea2c99508a34ed665f5` |
| P2 target lattice | `424bc862a5a329d5bd69a724e1c9c6649f0826782864aa3f06d23553ba090ffb` |
| P2 leaveout grid | `d50abe1300ddccef8a3e992ee64f16f115299c0fea77d581fbeb907f6a1967a7` |
| P3 feature normalization | `8d4021f04c6a8471fba765b4bd2d08c6a32608fb778c84ebc7e0263736ac5756` |
| P4 ranker | `055c2fb0f2eb29a12c4b4f0acc054437cb2999e903d3b9409f47fe57ce0b77fd` |
| P4 ablation | `0d05c54106274fee4de1e2ecd0b1dd01547abefc68199f9677e894c115ce6119` |
| P5 certificate | `333695e0137bda2f3e1666d2d13e1673f83d49e7bea11e2cab64352f94739fc2` |
| P5 sensitivity | `b85d35b1205bf82e4f2adda8cee3ec28a180d53f0400734952683dcb337538e0` |
| P6 controller | `7d6dbd488e2aeda72ffd159ae688efb7e23b958278ccd7a64f9100a02acc98b1` |
| P7 runtime | `21a3b061e99d84c804fdf25c6ee0620be361a4dd0af9a1f9627fb79f93525e89` |
| P8 generated stop | `0da38e079a364b9e63942bbb9e3e67781491833eb9aab1c231c2325759eb0b65` |
| P9 damage notebook | `8dbbf56e68e818417b8e4531fb6bf249cba9b21f4daf41eacf17c93294f0414c` |
| P11 boundary | `6eb0e8660cbf78eb6c7b07e7fbb939b9568e0f839eaa7e3dbab71e88b177f9f8` |
| Base-Acc Sentinel | `5641ca25d7ece359317c6342fc473633b2aac03ffe131bff9bbd1d1e68824a27` |
| no-fake audit | `c0c9be3b9f814a5577a8404ae48425ad6d3aabcf9022df82fbd4749a2044b339` |
| contract audit | `da8d5f3c0c3a88419157798bff128649773937bca9ece443d4e482007d7c4e2b` |
| failure taxonomy | `fae17dfff01ef8a38941bfebfcbaa2bd3eafb181f385fdd63895fde48c82677d` |

## 17. 最终分析结论

v9.6.7 的真实推进是：

```text
v9.6.6:
  LDO failure 被拆到 dataset-level support/score shift；
  memory/offdiag veto 有诊断信号；
  但没有 template-balanced target 或 certificate；
  APGH/APGL/APGT 连续触发 generated stop-rule。

v9.6.7:
  进一步量化 dataset-level score shift、threshold transfer 和 target density；
  确认 R5B 的局部强信号不是可直接 official 的 dataset-invariant region；
  Core + Expansion lattice 仍无法同时满足 coverage、clean risk、memory/offdiag 与 LDO；
  legal feature normalization / risk veto 能降低 longrisk 或 LDO，但 precision/value 被破坏；
  rank-safe certificate v15 没有 weak/strong pass；
  generated route stop-rule 继续强制 APGU not_run。
```

机制判断：

1. H0 成立：v9.6.6 boundary 被复现，没有跳过 target/certificate/generated stop boundary。
2. H1 部分成立：LDO failure 主要表现为 score-scale shift，但 MNIST 仍有 target-density shift；score shift 的处理没有转成 deployable accepted region。
3. H2 未成立：Core + Expansion 不能解决“干净但少 / 数量够但不稳”的冲突；`T2.6` 质量强但 LDO = `0.3793`。
4. H3 成立于风险诊断层：memory/offdiag/risk veto 能压低 longrisk，但作为 ranker 组合后 precision/value 明显下降。
5. H4 未成立：existing-action route 没有出现 weak/strong controller candidate。
6. P6/P7 未打开：没有 rank-safe certificate，就没有 existing-action controller 或 selected runtime。
7. P8/P9 成立：generated blind variant route 应继续停止；APGU 不运行，damage notebook 指向 `longrisk_created`。
8. P11 未打开：没有 system controller，就不能打开 paired replay 或 short/full training。
9. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.6.7 真实执行后停在 `R2-DatasetShiftSolvedButTargetAbsent`：dataset shift 已进一步拆清，R5B/R7 的局部信号仍不能变成 template-balanced、dataset-invariant accepted region；generated route stop-rule 继续触发并阻止 APGU blind run，因此 strict PureKAN functional 仍未成功。
