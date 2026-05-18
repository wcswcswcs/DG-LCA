# DG-KAN v9.7.6 Core77 LDO / Density Mechanism / ExistingAction Controller 实验复盘

> 本复盘记录 `DG-KAN_v9.7.6_Core77_LDO_DensityMechanism_ExistingActionController_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 Core77 LDO definition diagnostic、natural stream density diagnostic、OldOnly matched contrast、minimal ranker diagnostic、support-aware diagnostic、generated-route boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R4-OldOnlyMechanismAbsent_OldRankDiagnosticOnly
base_candidate = LQ-t2-h256
success_v9760_strict_purekan_functional = False
success_v9760_full_functional = False
success_v9760_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9760_core77_ldo_density_mechanism_existing_action_controller_first_20260516T050000Z/
```

核心结论：

1. P0 复现 v9.7.5 boundary：source route = `R2-ExistingActionHighQualityButLDOBlocked`，system pass = `0`，generated route = `stopped_no_new_objective`，field red count = `0`。
2. P1 证明 Core77 raw LDO 被 leaveout 后外部低质动作回填/支持度惩罚显著放大：`LDO_raw = 0.4415584415584416`，但 `LDO_support_adjusted = 0.09260529551331587`，`LDO_equal_count = 0.0773703535816084`。
3. P1 precision-only LDO 为 `0.0`；value-only LDO 为 `0.09260529551331587`。Core77 三个 dataset 内 precision 都是 `1.0`，V LCB 都为正。
4. P1 `SupportStabilityPass = 1`，`Core77_LDO_support_artifact = 1`，`Core77_true_LDO_instability = 0`。但这只是 LDO 定义审计，不是 official controller pass。
5. P2 现有 2876 AP0 universe 中 Core-like count = `77`，rate = `0.026773296244784424`，precision = `1.0`，V LCB = `0.16402807605399972`，raw LDO = `0.4415584415584416`。
6. P2 natural AP0 stream extension 按 gate `not_run`：reason = `no_landed_natural_AP0_stream_extension_materializer_for_v9760`；没有伪造新增 action 或 outcome rows。
7. P3 OldOnly matched contrast 完成：matched pair count = `294`，matched success rate = `0.8521739130434782`，但 mechanism pass count = `0`。
8. P3 best matched feature = `WT80_LCB`，TopK87 precision = `0.47126436781609193`，V LCB = `-0.10386482729963209`，LDO = `0.13793103448275862`，未达机制 gate。
9. P3 若按 effect size 看，`memory_score_inverse / old_family_margin_delta / old_family_probe_loss_inverse` effect size 约 `1.022`，但 TopK87 precision 只有 `0.45977011494252873` 且 V LCB 为负。
10. P4 OldRank score 机制替代未过：best ranker = `R3-ValueProxy-HardRiskVeto`，precision = `0.8850574712643678`，V LCB = `0.14163282358223747`，risk/bad/null/memory/offdiag UCB = `0`，但 LDO drop = `0.39080459770114945`。
11. P4 原始 OldRank 仍是 high-quality / high-LDO diagnostic：precision = `0.8735632183908046`，V LCB = `0.14111334880346277`，LDO = `0.37931034482758624`。
12. P5 dataset-blind support-aware region 未过：best diagnostic = `S1-Core77Raw`，accepted = `77`，precision = `1.0`，V LCB = `0.16402807605399972`，raw LDO = `0.4415584415584416`，support-adjusted LDO = `0.09260529551331587`，但 accepted_count < `87` 且 LFO = `0.18181818181818177`。
13. P5 最接近 full87 的 `S4-Core77PlusOldOnlySupportBalanced` accepted = `87`，precision = `0.8850574712643678`，V LCB = `0.14120469391749882`，support-adjusted LDO = `0.08877827643026047`，但 raw LDO = `0.39080459770114945` 且 LFO = `0.16091954022988508`，因此 support-aware/raw official pass = `0 / 0`。
14. P6 expanded stream candidates `not_run`：P2 没有完成 natural AP0 stream extension。
15. P7/P8/P10/P11 gate-blocked：controller、selected runtime、paired replay、short/full boundary 均 `not_run`。
16. P9 generated route 继续停止：new objective evidence = `0`，APGU/APGV/APGW/APGX/APGY/APGZ run = `0`。
17. Base-Acc Sentinel 继续健康：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
18. 当前 primary blocker：`oldonly_mechanism_absent`；secondary blocker：`generated_route_stopped_no_new_objective`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9760_core77_ldo_density_mechanism_existing_action_controller.py` | v9.7.6 runner；读取 v9.7.5/v9.7.4/v9.7.2 artifacts 与 canonical AP0 ledger，执行 Core77 LDO definition audit、Core-like density diagnostic、OldOnly matched contrast、minimal ranker、support-aware accepted region 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9760_core77_ldo_density_mechanism_existing_action_controller.py
```

正式运行：

```bash
python experiments/run_v9760_core77_ldo_density_mechanism_existing_action_controller.py \
  --out-dir results/real_rerun_20260506/v9760_core77_ldo_density_mechanism_existing_action_controller_first_20260516T050000Z \
  --fresh --device auto --data-root data --seed 1314
```

运行结果：

```json
{
  "LDO_raw": 0.4415584415584416,
  "LDO_support_adjusted": 0.09260529551331587,
  "SupportStabilityPass": 1,
  "best_support_LDO_raw": 0.4415584415584416,
  "best_support_LDO_support_adjusted": 0.09260529551331587,
  "best_support_rule": "S1-Core77Raw",
  "out_dir": "results/real_rerun_20260506/v9760_core77_ldo_density_mechanism_existing_action_controller_first_20260516T050000Z",
  "primary_blocker": "oldonly_mechanism_absent",
  "route": "R4-OldOnlyMechanismAbsent_OldRankDiagnosticOnly",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 fake rows，没有 proxy rows。P2 natural stream extension 没有 landed materializer，因此按 gate `not_run`，没有生成假 AP0 扩展动作。

## 2. Route

`route_decision_v9760.json` 摘要：

```json
{
  "route": "R4-OldOnlyMechanismAbsent_OldRankDiagnosticOnly",
  "source_route_v9750": "R2-ExistingActionHighQualityButLDOBlocked",
  "p0_pass": 1,
  "SupportStabilityPass": 1,
  "LDO_raw": 0.4415584415584416,
  "LDO_precision_only": 0.0,
  "LDO_value_only": 0.09260529551331587,
  "LDO_support_adjusted": 0.09260529551331587,
  "LDO_equal_count": 0.0773703535816084,
  "P2_density_scaling_pass": 0,
  "natural_stream_extension_completed": 0,
  "P3_mechanism_pass": 0,
  "best_matched_feature": "WT80_LCB",
  "best_matched_feature_precision": 0.47126436781609193,
  "best_matched_feature_LDO": 0.13793103448275862,
  "P4_minimal_mechanism_pass": 0,
  "best_ranker_id": "R3-ValueProxy-HardRiskVeto",
  "best_ranker_precision": 0.8850574712643678,
  "best_ranker_LDO": 0.39080459770114945,
  "P5_support_aware_weak_pass": 0,
  "P5_raw_official_pass": 0,
  "best_support_rule": "S1-Core77Raw",
  "best_support_precision": 1.0,
  "best_support_LDO_raw": 0.4415584415584416,
  "best_support_LDO_support_adjusted": 0.09260529551331587,
  "controller_pass": 0,
  "selected_runtime_pass": 0,
  "generated_route_status": "stopped_no_new_objective",
  "system_legal_controller_pass": 0,
  "primary_blocker": "oldonly_mechanism_absent",
  "secondary_blocker": "generated_route_stopped_no_new_objective"
}
```

判断：v9.7.6 的最大新信息是 Core77 raw LDO 的主要惩罚来自 support/backfill 定义，而不是 dataset 内 precision failure。但由于 Core77 只有 77 rows，full87 support-aware 规则仍过不了 LFO/raw LDO，OldOnly 机制也没有被解释，route 停在 `R4-OldOnlyMechanismAbsent_OldRankDiagnosticOnly`。

## 3. P0 boundary

Artifacts：

```text
p0_boundary_reproduction_v9760.csv
p0_field_legality_audit_v9760.csv
```

Summary：

```text
source_route_v9750 = R2-ExistingActionHighQualityButLDOBlocked
system_legal_controller_pass_v9750 = 0
generated_route_status_v9750 = stopped_no_new_objective
Core77_precision = 1.0
Core77_LDO_drop = 0.4415584415584416
OldOnly_precision = 0.855072463768116
ExactOnly_precision = 0.014492753623188406
best_stability_precision = 0.8850574712643678
best_stability_LDO_drop = 0.39080459770114945
best_ablation_precision = 0.8735632183908046
best_ablation_LDO_drop = 0.37931034482758624
field green/yellow/red = 14 / 4 / 0
fake/proxy/cpu_offload = 0 / 0 / 0
P0_pass = 1
```

判断：P0 pass。v9.7.6 没有跳过 v9.7.5 的 existing-action LDO / OldOnly unexplained / generated stop boundary，也没有把 forbidden fields 带入 official path。

## 4. P1 Core77 LDO definition audit

Artifact：

```text
p1_core77_ldo_definition_audit_v9760.csv
```

Summary：

```text
Core77_count_total = 77
Core77_count_by_dataset = {"Fashion-MNIST": 34, "KMNIST": 28, "MNIST": 15}
Core77_precision_by_dataset = {"Fashion-MNIST": 1.0, "KMNIST": 1.0, "MNIST": 1.0}
Core77_V_LCB_by_dataset = {"Fashion-MNIST": 0.2185522558920942, "KMNIST": 0.07778373642602063, "MNIST": 0.07142278054068385}
Core77_longrisk/bad/null/memory/offdiag by dataset = all 0.0
LDO_raw = 0.4415584415584416
LDO_precision_only = 0.0
LDO_value_only = 0.09260529551331587
LDO_support_adjusted = 0.09260529551331587
LDO_equal_count = 0.0773703535816084
LDO_equal_count_p95 = 0.09652672025051506
bootstrap support-adjusted LDO p50/p95 = 0.07804030078779232 / 0.10214669940823856
bootstrap_prob_LDO_gt_0.10 / gt_0.20 = 0.0625 / 0.0
bootstrap raw LDO p50/p95 = 0.43999999999999995 / 0.5098039215686274
min/max dataset support = 15 / 34
support_imbalance_ratio = 2.2666666666666666
SupportStabilityPass = 1
Core77_LDO_support_artifact = 1
Core77_true_LDO_instability = 0
```

Per dataset：

| dataset | count | precision | V LCB | support-adjusted drop |
|---|---:|---:|---:|---:|
| `Fashion-MNIST` | `34` | `1.0` | `0.2185522558920942` | `0.0` |
| `KMNIST` | `28` | `1.0` | `0.07778373642602063` | `0.08624433962797909` |
| `MNIST` | `15` | `1.0` | `0.07142278054068385` | `0.09260529551331587` |

判断：P1 是本轮核心推进。旧 raw LDO 高主要因为 leave-dataset-out 后需要从 Core77 外部回填低质动作；如果只审计 selected region 内的 dataset support / value 下界，Core77 不像真实 precision failure。但 raw official gate 仍没有因此自动放宽。

## 5. P2 Core77 density / action universe

Artifact：

```text
p2_core77_density_action_universe_v9760.csv
```

Summary：

```text
canonical_ap0_action_count = 2876
natural_stream_extension_completed = 0
new_action_count_total = 0
best_existing_core_like_count = 77
best_existing_core_like_rate = 0.026773296244784424
best_existing_core_like_precision = 1.0
best_existing_core_like_V_LCB = 0.16402807605399972
best_existing_core_like_LDO = 0.4415584415584416
P2_density_scaling_pass = 0
P2_extension_gate_status = not_run_no_landed_materializer
```

Existing AP0 prefix diagnostic：

| prefix | seen | Core-like | rate | Core-like precision | V LCB | raw LDO |
|---|---:|---:|---:|---:|---:|---:|
| `25%` | `719` | `23` | `0.031988873435326845` | `1.0` | `0.22688741261238055` | `1.0` |
| `50%` | `1438` | `39` | `0.027121001390820583` | `1.0` | `0.20416982501314665` | `0.8717948717948718` |
| `75%` | `2157` | `65` | `0.03013444598980065` | `1.0` | `0.17321346270809634` | `0.523076923076923` |
| `100%` | `2876` | `77` | `0.026773296244784424` | `1.0` | `0.16402807605399972` | `0.4415584415584416` |

Natural stream extension：

```text
S1 +1 seed = not_run, no_landed_natural_AP0_stream_extension_materializer_for_v9760
S2 +3 seeds = not_run, no_landed_natural_AP0_stream_extension_materializer_for_v9760
S3 full extension = not_run, no_landed_natural_AP0_stream_extension_materializer_for_v9760
```

判断：P2 没有证明 density scaling works，也没有伪造新增 AP0 rows。当前只能说现有 universe 的 Core-like density 约 `2.68%`，自然扩流问题仍未被真实 materialize。

## 6. P3 OldOnly matched contrast

Artifact：

```text
p3_oldonly_matched_contrast_v9760.csv
```

Summary：

```text
matched_pair_count = 294
matched_success_rate = 0.8521739130434782
feature_count = 17
mechanism_pass_count = 0
best_feature_name = WT80_LCB
best_feature_precision = 0.47126436781609193
best_feature_V_LCB = -0.10386482729963209
best_feature_LDO = 0.13793103448275862
selected_minimal_features = ["memory_score_inverse", "old_family_probe_loss_inverse", "old_family_margin_delta"]
P3_mechanism_pass = 0
```

Representative feature rows：

| feature | effect size | AUC | TopK87 precision | V LCB | longrisk | LDO | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `WT80_LCB` | `-0.31668107799768724` | `0.4980046208779668` | `0.47126436781609193` | `-0.10386482729963209` | `0.0` | `0.13793103448275862` | `0` |
| `memory_score_inverse` | `1.022292288859202` | `0.8009871875656375` | `0.45977011494252873` | `-0.033718338838361464` | `0.0` | `0.04597701149425287` | `0` |
| `old_family_margin_delta` | `1.0222922206248868` | `0.8009871875656375` | `0.45977011494252873` | `-0.033718338838361464` | `0.0` | `0.04597701149425287` | `0` |
| `old_family_probe_loss_inverse` | `1.0222922337468694` | `0.8009871875656375` | `0.45977011494252873` | `-0.033718338838361464` | `0.0` | `0.04597701149425287` | `0` |

判断：P3 未找到可部署机制。部分 legal diagnostic features 能区分 OldOnly / ExactOnly，但用它们做 TopK87 会把 precision/value 打坏；WT80 比 v9.7.4 接近一些，但 V LCB 仍为负。

## 7. P4 OldRank score mechanism replacement

Artifact：

```text
p4_oldrank_score_mechanism_replacement_v9760.csv
```

Summary：

```text
ranker_count = 10
minimal_mechanism_pass_count = 0
best_ranker_id = R3-ValueProxy-HardRiskVeto
best_precision = 0.8850574712643678
best_V_LCB = 0.14163282358223747
best_LDO_drop = 0.39080459770114945
best_feature_count = 2
P4_minimal_mechanism_pass = 0
```

Representative rankers：

| ranker | features | precision | V LCB | longrisk | LDO | LFO | OldOnly overlap | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `R3-ValueProxy-HardRiskVeto` | `2` | `0.8850574712643678` | `0.14163282358223747` | `0.0` | `0.39080459770114945` | `0.16091954022988508` | `0.9565217391304348` | `0` |
| `R0-OldRank` | `6` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `0.14942528735632188` | `1.0` | `0` |
| `R1-ValueProxyOnly` | `1` | `0.4942528735632184` | `0.28517470943892703` | `0.5053405200342863` | `0.06896551724137934` | `0.022988505747126464` | `0.4927536231884058` | `0` |
| `R6-OldFamilyStableRank` | `2` | `0.45977011494252873` | `-0.033718338838361464` | `0.0` | `0.04597701149425287` | `0.011494252873563204` | `0.5217391304347826` | `0` |
| `R8-MinimalTwoFeatureRank` | `2` | `0.45977011494252873` | `-0.033718338838361464` | `0.0` | `0.04597701149425287` | `0.011494252873563204` | `0.5217391304347826` | `0` |

判断：P4 未过。最小 2-feature value+risk rule 可以重现高质量 clean-risk region，但 LDO/LFO 仍不过；低 LDO 的 simple features value 为负或 precision 不足。

## 8. P5 dataset-blind support-aware accepted region

Artifact：

```text
p5_dataset_blind_support_aware_region_v9760.csv
```

Summary：

```text
rule_count = 7
support_aware_weak_pass_count = 0
raw_official_pass_count = 0
best_rule_id = S1-Core77Raw
best_precision = 1.0
best_V_LCB = 0.16402807605399972
best_LDO_raw = 0.4415584415584416
best_LDO_support_adjusted = 0.09260529551331587
official_support_adjusted_LDO_allowed = 0
P5_support_aware_weak_pass = 0
P5_raw_official_pass = 0
```

Representative rules：

| rule | accepted | precision | V LCB | raw LDO | support-adjusted LDO | LFO | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `S1-Core77Raw` | `77` | `1.0` | `0.16402807605399972` | `0.4415584415584416` | `0.09260529551331587` | `0.18181818181818177` | `0` |
| `S3-Core77EqualCountBootstrap` | `45` | `1.0` | `0.14034000080454226` | `0.33333333333333337` | `0.07156707520635665` | `0.19999999999999996` | `0` |
| `S4-Core77PlusOldOnlySupportBalanced` | `87` | `0.8850574712643678` | `0.14120469391749882` | `0.39080459770114945` | `0.08877827643026047` | `0.16091954022988508` | `0` |
| `S5-OldRankWithSupportFloor` | `87` | `0.8045977011494253` | `0.12779778491856425` | `0.367816091954023` | `0.14745484400656816` | `0.09195402298850575` | `0` |
| `S6-MinimalFeatureRankWithSupportFloor` | `87` | `0.45977011494252873` | `-0.033718338838361464` | `0.1839080459770115` | `0.09310344827586209` | `0.04597701149425287` | `0` |
| `S7-DensityScaledCoreLikeNaturalStream` | `not_run` |  |  |  |  |  | `0` |

判断：P5 是“接近但不 official”的关键结果。Core77 与 Core77+OldOnly 的 support-adjusted LDO 都低于 0.10，但 Core77 不够 87 rows，full87 的 LFO/raw LDO 仍不过；本轮不允许把 support-adjusted diagnostic 写成 official controller。

## 9. P6-P11 boundary

P6：

```text
p6_core_density_expanded_controller_candidates_v9760.csv = not_run
reason = P2_natural_AP0_stream_extension_not_completed
expanded_action_count = 0
P6_expanded_stream_pass = 0
```

P7：

```text
p7_controller_boundary_v9760.csv = not_run
reason = P4_P5_P6_no_official_weak_pass
P7_controller_pass = 0
controller_selected = 0
```

P8：

```text
p8_selected_runtime_boundary_v9760.csv = not_run
reason = P7_controller_not_passed
selected_runtime_pass = 0
```

P9：

```text
generated_route_status = stopped_no_new_objective
new_objective_evidence_present = 0
APGU_APGV_APGW_APGX_APGY_APGZ_run = 0
generated_action_count = 0
branch_horizon_rows = 0
generated_route_stop_triggered = 1
```

P10/P11：

| artifact | status / reason |
|---|---|
| `p10_paired_replay_boundary_v9760.csv` | `not_run`, `P7_or_P8_not_passed` |
| `p11_short_full_boundary_v9760.csv` | `not_run`, `P10_paired_replay_not_open` |

Allowed next gates：

```json
{
  "selected_runtime_allowed": 0,
  "paired_replay_allowed": 0,
  "short_full_allowed": 0,
  "generated_route_allowed": 0,
  "direct_solved_sandbox_allowed": 0,
  "APGU_APGV_APGW_APGX_APGY_APGZ_run_allowed": 0,
  "support_adjusted_controller_candidate_requires_policy_decision": 0
}
```

Stop conditions：

```json
{
  "enter_system": 0,
  "stop_core77_direct_raw_ldo_controller": 1,
  "stop_generated_blind_variants": 1,
  "stop_natural_stream_density_claims": 1,
  "stop_oldrank_controller_promotion": 1,
  "support_adjusted_ldo_policy_decision_required": 0
}
```

判断：没有 official weak pass，因此 controller/runtime/paired replay/short-full 均关闭。Generated route 没有新 objective evidence，继续停止。

## 10. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9760.csv
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

本轮额外落盘诊断图：

```text
fig_p1_ldo_definition_decomposition.svg
fig_p1_core77_support_vs_ldo.svg
fig_p1_core77_bootstrap_ldo.svg
fig_p2_core_density_growth.svg
fig_p2_core_density_by_dataset.svg
fig_p3_oldonly_matched_contrast_forest.svg
fig_p3_response_vector_oldonly_exactonly.svg
fig_p4_ranker_ablation_frontier.svg
fig_p4_oldonly_overlap_heatmap.svg
fig_p5_support_aware_frontier.svg
```

判断：这些图只用于复核诊断，不构成 official pass。

## 12. No-fake audit

```text
rows_checked = 472
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
v9750_boundary_pass = 1
core77_ldo_definition_audit_pass = 1
support_stability_pass = 1
natural_stream_extension_completed = 0
oldonly_mechanism_pass = 0
minimal_ranker_mechanism_pass = 0
support_aware/raw_official_pass = 0/0
controller/runtime/system = 0/0/0
generated_route_stop = 1
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R4-OldOnlyMechanismAbsent_OldRankDiagnosticOnly
F0_boundary_fail = 0
F1_raw_LDO_high_support_adjusted_low = 1
F2_natural_stream_extension_missing = 1
F3_oldonly_mechanism_absent = 1
F4_minimal_ranker_fail = 1
F5_support_aware_not_official_raw_ldo_fail = 0
F6_controller_runtime_blocked = 1
F7_generated_route_stopped = 1
F8_system_not_official = 1
primary_blocker = oldonly_mechanism_absent
```

## 13. Hash

| artifact | SHA256 |
|---|---|
| plan | `4f75299b0e71176b95aab71c85f7f55382178b4fc19b9923df2ce514246922a9` |
| runner | `fbc3f1c55cc2e702b04b7e4f1dcd51ad583bb7231687bd9c9f70a96de8139f99` |
| route | `d3dbf9da5954963291de1d2580f85ed698e4384259c2778a754a5129ce02868c` |
| P0 boundary | `b27a2a7e6002d38831bc8e6f2a270bc25efdc9df71c100905d15e4ab2bf6ebdd` |
| P0 field legality | `8f84af577e914994a6e1dcc5563e0cd70090d0179f03aa8e4520d573c8e1400f` |
| P1 LDO audit | `719c455ad9292ff30d46d2de564a488d54f0efa446b70ad2fac0405d5ba15131` |
| P2 density | `6302db9aeb905980e0ab181620c2fa31028c8dc6b62b5119ec50fa0d11d1d4ac` |
| P3 matched contrast | `8265efbbc8a5ee8d3a1a5e961a4818e2f42ea5d5e690d1673d81d022c12ae758` |
| P4 ranker | `7988855af3d220c3fd5dd52e23402d80321cfdc127fd0d02fdfe9e99d403a754` |
| P5 support-aware | `4b9d9d0cf4c44c902ec84f9c8f3bc8c3cb9b0d5ef87a875f97657360ac2998b7` |
| P6 expanded stream | `5a36803553afe22df742cccf6a132adb1a48ae0a9e39ee76768548eda6f325ef` |
| P7 controller | `6b365d573955108ac2b49e05a23c0b58a3e19724213cf766d29c0badfcfe3510` |
| P8 runtime | `ccd318965f2eed2541c6e90f607296782aa577d0e0641453ff2d0d2754658765` |
| P9 generated route | `04e0c470b0a7b6eec184cb3ff584f076301bc6bcf1258989f0b0a93f92ce2e08` |
| P10 paired replay | `ad0662b064d2ce0fc91c839e330ae94b1a69ab15ee27b0883f90b641938e7b00` |
| P11 short/full | `9935bf0648595350c9d8e799c3e96c7aac8325d63027eca6236a21039ebfa8e9` |
| Base-Acc Sentinel | `542c165667337710fbd1915ed7e4ee3f8fbb77cda7b331f2b0e1f1b2fcc091e8` |
| no-fake audit | `6e5b88b011347af45b288cbb7eab8165bffed601d0898d9002aa731dc4990b47` |
| contract audit | `ffee5d37bf0dca95db183633f2a4d4fd103a1ace0049e86fe873011ad63d4dc0` |
| failure taxonomy | `a2c6ca0a0a8983e99e2c721cf74a295cf0ae119a6f55eecc90d9d800c38d9bdf` |
| allowed next gates | `4f342b547f89b106de0dd8f89551ce5243b07d489e450e8e4330c6153d5e46e9` |
| stop conditions | `ea87a6a9c7e8e928da972784c2c2633bfaae6cd98aa561d69f3e46866d739c83` |

## 14. 最终分析结论

v9.7.6 的真实推进是：

```text
v9.7.5:
  Core77/OldOnly high-quality existing-action signal 仍存在；
  但 raw LDO 高、OldRank mechanism 未解释、generated route 继续停止。

v9.7.6:
  首次把 Core77 raw LDO 拆成 precision/value/support/backfill 几个分量；
  证明 Core77 内部跨 dataset precision 全为 1.0，support-adjusted LDO 低于 0.10；
  但 Core77 只有 77 rows，不满足 accepted_count >= 87；
  Core77+OldOnly full87 虽 support-adjusted LDO 也低于 0.10，但 raw LDO 与 LFO 仍不过；
  OldOnly matched contrast 和 minimal ranker 没有找到简单合法机制；
  natural AP0 stream extension 没有 landed materializer，因此 density scaling 不能声称成功；
  controller/runtime/system/paired replay 全部 gate-blocked。
```

机制判断：

1. H0 成立：v9.7.5 boundary 被复现，没有跳过 existing-action LDO / generated stop / no system controller。
2. H1 支持 support-size/backfill artifact：Core77 dataset precision 全为 `1.0`，V LCB 都为正，`LDO_support_adjusted = 0.0926`，`LDO_equal_count = 0.0774`。
3. H2 不成立为“真实 precision instability”：precision-only LDO = `0.0`，support-adjusted bootstrap `P(LDO > 0.20) = 0.0`。
4. P2 未打开 natural density scaling：没有 landed natural AP0 stream extension materializer，不能伪造 S1/S2/S3 rows。
5. H3 未成立：OldOnly matched contrast 没有找到两个能同时满足 effect size、TopK quality、risk 与 LDO 的 legal features。
6. H4 未成立：Minimal ranker 没有 pass；best 2-feature ranker 质量高但 raw LDO/LFO 仍高。
7. H5 未成立：support-aware accepted region 只是诊断接近，未形成 official weak pass；Core77 count 不足，full87 raw LDO/LFO 不过。
8. H6 未打开：expanded stream controller candidates 因 P2 not_run 而 gate-blocked。
9. P7/P8 未打开：没有 official weak controller candidate，因此 controller 与 runtime 均 not_run。
10. P9 成立：generated route 继续 stopped_no_new_objective；APG blind variants 仍不允许。
11. P10/P11 未打开：没有 system controller，就不能打开 paired replay 或 short/full training。
12. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.7.6 真实执行后停在 `R4-OldOnlyMechanismAbsent_OldRankDiagnosticOnly`：Core77 的 raw LDO 很大程度是 support/backfill 定义放大，而不是 dataset 内 precision 崩；但 Core77 数量不足，full87 raw LDO/LFO 仍不过，OldOnly 机制仍未被简单合法特征解释，natural AP0 扩流也没有真实 materializer，因此 strict PureKAN functional 仍未成功。
