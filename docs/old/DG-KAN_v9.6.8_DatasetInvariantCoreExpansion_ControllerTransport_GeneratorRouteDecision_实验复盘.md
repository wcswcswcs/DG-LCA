# DG-KAN v9.6.8 Dataset-Invariant Core-Expansion / Controller Transport / Generator Route Decision 实验复盘

> 本复盘记录 `DG-KAN_v9.6.8_DatasetInvariantCoreExpansion_ControllerTransport_GeneratorRouteDecision_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 score-transport diagnostic、core-expansion target diagnostic、veto-order diagnostic、rank/certificate diagnostic、generated-route stop-rule、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R6-GeneratedRouteStoppedNoNewObjective
base_candidate = LQ-t2-h256
success_v9680_strict_purekan_functional = False
success_v9680_full_functional = False
success_v9680_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9680_dataset_invariant_core_expansion_controller_transport_generator_route_decision_first_20260515T180000Z/
```

核心结论：

1. P0 复现 v9.6.7 boundary：source route = `R2-DatasetShiftSolvedButTargetAbsent`，generated stop-rule status = `1`，no-fake/no-proxy = `1 / 1`。
2. P0 field legality 过：green/yellow/red field count = `14 / 4 / 0`；dataset 只允许 leaveout/autopsy，不用于 selector/controller。
3. P1 score transport 没有打开：transport strong/weak pass = `0 / 0`；best transport = `S0-raw-score`，macro precision = `0.8671023965141612`，macro V LCB = `0.10955705927483939`，longrisk UCB = `0.0`，但 LDO drop = `0.37931034482758624`。
4. P1 transport 能降低 LDO 的方案牺牲 value/precision：`S5-memory-offdiag-safe-subset-percentile` LDO drop = `0.06896551724137934`，但 pooled precision = `0.39080459770114945`，V LCB = `-0.01929919347292527`。
5. P2 Core + Expansion target lattice v3 未过：strong/weak target pass = `0 / 0`；best target = `T3.1-CoreOnly`，accepted = `77`，coverage = `0.026773296244784424`，GradeAB = `1.0`，V LCB = `0.16402807605399972`，risk/bad/null/memory/offdiag UCB = `0`，但 coverage < `0.03` 且 LDO drop = `0.39080459770114945`。
6. P2 扩展到 87 rows 仍不闭合：`T3.2` coverage = `0.030250347705146036`，GradeAB = `0.8850574712643678`，V LCB = `0.1597545290247637`，但 null UCB = `0.1674432324061406`，offdiag UCB = `0.18196532683597333`，LDO drop = `0.39080459770114945`。
7. P3 veto-order experiment 未过：best = `V7-two-stage-core-first-expansion-second`，accepted = `87`，GradeAB = `0.8850574712643678`，V LCB = `0.13618096164286234`，longrisk/bad/null/memory/offdiag UCB = `0`，但 LDO drop = `0.39080459770114945`。
8. P3 风险优先 veto 可降 LDO，但会破坏 precision/value：`V1-risk-veto-before-value-rank` precision = `0.27586206896551724`，V LCB = `-0.08151817674709344`，LDO drop = `0.06896551724137931`。
9. P4 dataset-invariant ranker v2 未过：strong/weak pass = `0 / 0`；best = `R8A-transported-value-rank`，TopK87 precision = `0.8735632183908046`，V LCB = `0.14111334880346277`，longrisk/bad/null/memory/offdiag UCB = `0`，但 LDO drop = `0.37931034482758624`。
10. P4 的 `R8I-core-expansion-ranker` 更接近 core-expansion，但仍 LDO fail：precision = `0.8390804597701149`，V LCB = `0.12532548621031714`，LDO drop = `0.3563218390804597`。
11. P5 rank-safe certificate v16 未过：best = `C16G-topK64-high-precision-diagnostic`，accepted = `64`，coverage = `0.022253129346314324`，precision = `1.0`，V LCB = `0.19924925130425136`，risk/bad/null/memory/offdiag UCB = `0`，但 coverage 不足且 LDO drop = `0.328125`。
12. P6/P7 gate-blocked：existing-action controller 与 selected runtime 均 `not_run`。
13. P8 generated route stop-rule 继续触发：APGH/APGL/APGT consecutive failure family count = `3`，new objective evidence = `0`，APGU_run = `0`，no fake APGU rows = `1`。
14. P9 generated damage notebook 完成：damage rows = `24`，dominant damage mode = `longrisk_created`，assigned fraction = `1.0`，recommendation = `stop_blind_generated_variants`。
15. P11/P12 gate-blocked：没有 system controller，因此 paired replay、short/full boundary 未打开。
16. Base-Acc Sentinel 继续健康：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
17. 当前 primary blocker：`generated_route_stopped_no_new_objective`；existing-action side 仍没有 deployable target/certificate。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9680_dataset_invariant_core_expansion_controller_transport_generator_route_decision.py` | v9.6.8 runner；读取 v9.6.7/v9.6.6/v9.6.5/v9.6.4/v9.6.3 artifacts，执行 score transport、core-expansion target lattice v3、veto-order experiment、dataset-invariant ranker v2、certificate v16、generated route stop-rule、damage notebook 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9680_dataset_invariant_core_expansion_controller_transport_generator_route_decision.py
```

正式运行：

```bash
python experiments/run_v9680_dataset_invariant_core_expansion_controller_transport_generator_route_decision.py \
  --out-dir results/real_rerun_20260506/v9680_dataset_invariant_core_expansion_controller_transport_generator_route_decision_first_20260515T180000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --apgu-actions-per-primitive 64
```

运行结果：

```json
{
  "APGU_run": 0,
  "best_certificate_id": "C16G-topK64-high-precision-diagnostic",
  "best_ranker_id": "R8A-transported-value-rank",
  "best_target_id": "T3.1-CoreOnly",
  "best_transport_id": "S0-raw-score",
  "generated_route_stop_triggered": "1",
  "out_dir": "results/real_rerun_20260506/v9680_dataset_invariant_core_expansion_controller_transport_generator_route_decision_first_20260515T180000Z",
  "primary_blocker": "generated_route_stopped_no_new_objective",
  "route": "R6-GeneratedRouteStoppedNoNewObjective",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 proxy rows。APGU 因 generated route stop-rule 继续不运行，没有生成 APGU fake/proxy rows。

## 2. Route

`route_decision_v9680.json` 摘要：

```json
{
  "route": "R6-GeneratedRouteStoppedNoNewObjective",
  "source_route_v9670": "R2-DatasetShiftSolvedButTargetAbsent",
  "p0_pass": 1,
  "score_transport_strong_pass": 0,
  "score_transport_weak_pass": 0,
  "best_transport_id": "S0-raw-score",
  "best_transport_LDO_drop": 0.37931034482758624,
  "template_balanced_target_pass": 0,
  "template_balanced_target_weak_pass": 0,
  "best_target_id": "T3.1-CoreOnly",
  "best_target_accepted_count": 77,
  "best_target_LDO_drop": 0.39080459770114945,
  "p3_veto_order_pass": 0,
  "best_veto_strategy_id": "V7-two-stage-core-first-expansion-second",
  "ranker_strong_pass": 0,
  "ranker_weak_pass": 0,
  "best_ranker_id": "R8A-transported-value-rank",
  "best_ranker_precision": 0.8735632183908046,
  "best_ranker_LDO_drop": 0.37931034482758624,
  "rank_safe_certificate_v16_pass": 0,
  "certificate_weak_pass": 0,
  "best_certificate_id": "C16G-topK64-high-precision-diagnostic",
  "best_certificate_precision": 1.0,
  "best_certificate_LDO_drop": 0.328125,
  "existing_action_controller_pass": 0,
  "selected_runtime_pass": 0,
  "generated_route_stop_triggered": "1",
  "new_objective_evidence_present": 0,
  "APGU_run": 0,
  "p9_damage_notebook_pass": 1,
  "system_legal_controller_pass": 0,
  "primary_blocker": "generated_route_stopped_no_new_objective",
  "secondary_blocker": "generated_route_stop_triggered"
}
```

判断：v9.6.8 没有把 score transport、core-expansion、veto-order、ranker 或 certificate diagnostic 写成 controller。generated route stop-rule 仍成立，且没有新 objective evidence，因此 route 停在 `R6-GeneratedRouteStoppedNoNewObjective`。

## 3. P0 boundary

Artifacts：

```text
p0_boundary_reproduction_v9680.csv
p0_field_legality_audit_v9680.csv
```

Boundary summary：

```text
source_v9670_route = R2-DatasetShiftSolvedButTargetAbsent
generated_stop_rule_status = 1
no_fake/no_proxy/cpu_offload = 1 / 1 / 0
old_table_official_violation_count = 0
dataset_name_used_by_controller_count = 0
outcome_derived_field_used_count = 0
validation_test_used_by_controller_count = 0
future_outcome_used_by_feature_count = 0
p0_pass = 1
```

Field legality summary：

```text
green/yellow/red field count = 14 / 4 / 0
dataset_allowed_for_leaveout = 1
dataset_name_commit_feature_count = 0
payload_norm_bucket_used_as_selector = 0
candidate_id/source_payload_hash used as selector = 0 / 0
outcome_derived_field_used_count = 0
field_legality_pass = 1
```

Input hashes from v9.6.7:

```text
canonical_table_hash = 822f58b0c9ba9f9420f6395b018b676f8d991cb1c8294d45e42bcfa0245b3566
feature_ledger_hash = 8d4021f04c6a8471fba765b4bd2d08c6a32608fb778c84ebc7e0263736ac5756
ranker_input_hash = 055c2fb0f2eb29a12c4b4f0acc054437cb2999e903d3b9409f47fe57ce0b77fd
certificate_input_hash = 333695e0137bda2f3e1666d2d13e1673f83d49e7bea11e2cab64352f94739fc2
base_acc_sentinel_hash = 5641ca25d7ece359317c6342fc473633b2aac03ffe131bff9bbd1d1e68824a27
```

判断：P0 pass。v9.6.8 没有跳过 v9.6.7 的 target/ranker/certificate/generated stop boundary，也没有把 dataset name 或 outcome-derived field 带入 official path。

## 4. P1 score transport audit

Artifacts：

```text
p1_score_transport_audit_v9680.csv
p1_score_distribution_by_dataset_v9680.csv
```

Summary：

```text
transport_count = 9
transport strong/weak pass count = 0 / 0
best_transport_id = S0-raw-score
best_TopK87_precision_macro = 0.8671023965141612
best_TopK87_V_LCB_macro = 0.10955705927483939
best_TopK87_longrisk_UCB_macro = 0.0
best_LDO_drop = 0.37931034482758624
uses_dataset_name/outcome_field = 0 / 0
```

Representative transports：

| transport | macro precision | pooled precision | V LCB | longrisk UCB | LDO drop | pass |
|---|---:|---:|---:|---:|---:|---:|
| `S0-raw-score` | `0.8671023965141612` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `0` |
| `S1-robust-z-score` | `0.8671023965141612` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `0` |
| `S5-memory-offdiag-safe-subset-percentile` | `0.5580808080808081` | `0.39080459770114945` | `-0.01929919347292527` | `0.0` | `0.06896551724137934` | `0` |
| `S7-shared-quantile-mapping-no-branch` | `0.5395445134575569` | `0.3563218390804598` | `-0.04311254254174981` | `0.0` | `0.04597701149425287` | `0` |
| `S8-conformal-rank-fixed-global-quota` | `0.12340425531914893` | `0.1839080459770115` | `-0.10816956908194428` | `0.0` | `0.04597701149425287` | `0` |

S0 per dataset：

| dataset | score mean | PSI | TopK87 count / precision | V LCB | longrisk | threshold precision |
|---|---:|---:|---:|---:|---:|---:|
| `Fashion-MNIST` | `-6.021749447018721` | `0.3477132255617457` | `34 / 1.0` | `0.2185522558920942` | `0.0` | `1.0` |
| `KMNIST` | `-3.732632624783545` | `0.3472940184541398` | `36 / 0.7777777777777778` | `0.053201019604910284` | `0.0` | `0.3888888888888889` |
| `MNIST` | `-4.067975926348259` | `0.244398051883377` | `17 / 0.8235294117647058` | `0.05691790232751366` | `0.0` | `0.4166666666666667` |

判断：score transport 没有把 dataset shift 转成 deployable accepted region。低 LDO 的 transport 会显著损伤 pooled precision 和 V。

## 5. P2 Core + Expansion target lattice v3

Artifacts：

```text
p2_core_expansion_target_lattice_v3_v9680.csv
p2_target_support_by_dataset_template_v9680.csv
```

Summary：

```text
target_candidate_count = 10
strong/weak target pass count = 0 / 0
best_target_id = T3.1-CoreOnly
best_accepted_count = 77
best_coverage = 0.026773296244784424
best_GradeAB_precision = 1.0
best_V_integrated_LCB = 0.16402807605399972
best_h240_longrisk/bad/null/memory/offdiag UCB = 0.0 / 0.0 / 0.0 / 0.0 / 0.0
best_LDO_drop = 0.39080459770114945
core_count = 77
needed_expansion_to_87 = 10
template_balanced_target_pass = 0
```

Representative targets：

| target | accepted | coverage | GradeAB | V LCB | bad/null | memory/offdiag | LDO | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `T3.1-CoreOnly` | `77` | `0.026773296244784424` | `1.0` | `0.16402807605399972` | `0.0 / 0.0` | `0.0 / 0.0` | `0.39080459770114945` | `0` |
| `T3.2-CorePlusNearest10TransportedScore` | `87` | `0.030250347705146036` | `0.8850574712643678` | `0.1597545290247637` | `0.03389313880385762 / 0.1674432324061406` | `0.0 / 0.18196532683597333` | `0.39080459770114945` | `0` |
| `T3.4-CorePlusNearest10ConformalLowRisk` | `79` | `0.027468706536856746` | `1.0` | `0.16201929527024647` | `0.0 / 0.0` | `0.05995628911411127 / 0.05995628911411127` | `0.41379310344827586` | `0` |
| `T3.9-CoreExpansionCapFamilyTemplateScoreBucket` | `87` | `0.030250347705146036` | `0.8850574712643678` | `0.15864282315952377` | `0.054480608015849245 / 0.15267457786429023` | `0.03389313880385762 / 0.18196532683597333` | `0.39080459770114945` | `0` |

判断：P2 继续卡在“干净但 coverage 不足”和“coverage 足但 LDO/null/offdiag 不稳”的冲突上，没有 official template-balanced target。

## 6. P3 veto-order experiment

Artifact：

```text
p3_veto_order_experiment_v9680.csv
```

Summary：

```text
strategy_count = 10
strategy_pass_count = 0
best_strategy_id = V7-two-stage-core-first-expansion-second
best_GradeAB_precision = 0.8850574712643678
best_V_LCB = 0.13618096164286234
best_longrisk/bad/null UCB = 0.0 / 0.0 / 0.0
best_LDO_drop = 0.39080459770114945
best_veto_false_positive_rate = 0.0
best_veto_false_negative_rate = 0.11494252873563218
p3_veto_order_pass = 0
```

Representative strategies：

| strategy | accepted | GradeAB | V LCB | longrisk | LDO | false positive / false negative | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `V0-value-rank-only` | `87` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `0.0 / 0.12643678160919541` | `0` |
| `V1-risk-veto-before-value-rank` | `87` | `0.27586206896551724` | `-0.08151817674709344` | `0.0` | `0.06896551724137931` | `0.9285714285714286 / 0.7241379310344828` | `0` |
| `V4-bad-null-veto-after-value-rank` | `87` | `0.42528735632183906` | `-0.005191784819410055` | `0.0` | `0.17241379310344823` | `0.9523809523809523 / 0.5747126436781609` | `0` |
| `V7-two-stage-core-first-expansion-second` | `87` | `0.8850574712643678` | `0.13618096164286234` | `0.0` | `0.39080459770114945` | `0.0 / 0.11494252873563218` | `0` |

判断：veto order 能在风险、LDO、precision/value 之间移动 tradeoff，但没有策略同时满足 official gate。

## 7. P4 ranker 与 P5 certificate

P4 artifacts：

```text
p4_dataset_invariant_ranker_v2_v9680.csv
p4_ranker_feature_ledger_v9680.csv
```

P4 summary：

```text
ranker_count = 10
ranker strong/weak pass count = 0 / 0
best_ranker_id = R8A-transported-value-rank
best_TopK87_precision = 0.8735632183908046
best_V_LCB = 0.14111334880346277
best_longrisk/bad/null/memory/offdiag UCB = 0.0 / 0.0 / 0.0 / 0.0 / 0.0
best_LDO_drop = 0.37931034482758624
uses_dataset_name_at_inference/outcome_field = 0 / 0
```

Representative rankers：

| ranker | TopK87 precision | V LCB | longrisk | LDO | worst dataset precision | pass |
|---|---:|---:|---:|---:|---:|---:|
| `R8A-transported-value-rank` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `0.7777777777777778` | `0` |
| `R8B-transported-value-risk-veto` | `0.3333333333333333` | `-0.05636589632572801` | `0.0` | `0.06896551724137928` | `0.29411764705882354` | `0` |
| `R8F-pairwise-score-stratum-rank` | `0.1724137931034483` | `-0.13090562904179798` | `0.0` | `0.04597701149425287` | `0.15` | `0` |
| `R8I-core-expansion-ranker` | `0.8390804597701149` | `0.12532548621031714` | `0.0` | `0.3563218390804597` | `0.7142857142857143` | `0` |

P5 artifacts：

```text
p5_rank_safe_certificate_v16_v9680.csv
p5_certificate_threshold_sensitivity_v9680.csv
```

P5 summary：

```text
certificate_count = 8
certificate strong/weak pass count = 0 / 0
best_certificate_id = C16G-topK64-high-precision-diagnostic
best_source_ranker_id = R8A-transported-value-rank
best_accepted_count_heldout = 64
best_coverage_heldout = 0.022253129346314324
best_GradeAB_precision_heldout = 1.0
best_V_LCB_heldout = 0.19924925130425136
best_longrisk/bad/null/memory/offdiag UCB = 0.0 / 0.0 / 0.0 / 0.0 / 0.0
best_LDO_drop = 0.328125
rank_safe_certificate_v16_pass = 0
```

Representative certificates：

| certificate | accepted | coverage | precision | V LCB | LDO | pass |
|---|---:|---:|---:|---:|---:|---:|
| `C16A-fixed-TopK87-global` | `87` | `0.030250347705146036` | `0.8735632183908046` | `0.14111334880346277` | `0.37931034482758624` | `0` |
| `C16C-conformal-risk-bound` | `87` | `0.030250347705146036` | `0.3793103448275862` | `-0.02184751223127553` | `0.06896551724137928` | `0` |
| `C16F-quota-free-transport-threshold` | `96` | `0.03337969401947149` | `0.3229166666666667` | `-0.05617237306235747` | `0.07291666666666669` | `0` |
| `C16G-topK64-high-precision-diagnostic` | `64` | `0.022253129346314324` | `1.0` | `0.19924925130425136` | `0.328125` | `0` |
| `C16H-hybrid-core-all-expansion-if-bound` | `79` | `0.027468706536856746` | `1.0` | `0.16201929527024647` | `0.45569620253164556` | `0` |

判断：P4/P5 均未打开。高 precision 的 certificate coverage 不够且 LDO 仍高；coverage 达标的 certificate precision/value 或 LDO 不够。

## 8. P6-P7 controller/runtime boundary

P6：

```text
p6_existing_action_minimal_controller_v9680.csv = not_run
reason = P5_certificate_strong_or_weak_pass_failed
controller_selected = 0
existing_action_controller_pass = 0
source_controller_pass = 0
```

P7：

```text
p7_selected_controller_runtime_v9680.csv = not_run
reason = P6_controller_not_selected
selected_runtime_pass = 0
```

判断：没有把 P1-P5 的 diagnostic 写成 official controller 或 runtime pass。

## 9. P8 generated route stop-rule

Artifact：

```text
p8_generated_route_stop_rule_v9680.csv
```

Summary：

```text
family_count = 3
consecutive_failure_family_count = 3
new_objective_evidence_present = 0
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

判断：P8 pass。APGU/APGV 不允许继续 blind variant；需要新 generator objective，而不是继续 APG* 小变体。

## 10. P9 generated damage notebook

Artifact：

```text
p9_generated_failure_mechanism_notebook_v9680.csv
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

判断：P9 支持 P8 的 stop-rule。APGH/APGL/APGT 的 primitive-level damage 行都指向 `longrisk_created` 主导。

## 11. P10-P12 route/system boundary

P10：

```text
p10_route_decision_v9680.json
route = R6-GeneratedRouteStoppedNoNewObjective
system_legal_controller_pass = 0
primary_blocker = generated_route_stopped_no_new_objective
secondary_blocker = generated_route_stop_triggered
```

P11：

```text
p11_paired_replay_boundary_v9680.csv = not_run
reason = P6_or_P7_strong_pass_failed
system_legal_controller_pass = 0
paired_replay_pass = 0
short_full_boundary_open = 0
```

P12：

```text
p12_short_full_boundary_v9680.csv = not_run
reason = P11_paired_replay_not_open
short_run_boundary_open = 0
full_run_boundary_open = 0
short_full_boundary_pass = 0
```

Allowed next gates：

```json
{
  "selected_runtime_allowed": 0,
  "paired_replay_allowed": 0,
  "short_full_allowed": 0,
  "APGU_APGV_allowed": 0,
  "primitive_redesign_required": 1
}
```

Stop conditions：

```json
{
  "stop_score_transport_patching": 1,
  "stop_core_expansion_target_patching": 1,
  "stop_existing_action_route": 0,
  "stop_generated_blind_variants": 1
}
```

判断：没有 controller/runtime strong pass，因此 paired replay、short/full training 均未打开。generated route 明确要求 primitive redesign。

## 12. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9680.csv
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
fig_p1_quantile_transport_curve.svg
fig_p1_score_distribution_by_dataset_before_after.svg
fig_p1_score_transport_pareto_precision_vs_LDO.svg
fig_p1_threshold_transfer_heatmap.svg
fig_p1_topk_count_precision_by_dataset.svg
fig_p2_core_expansion_frontier_count_vs_cleanliness.svg
fig_p2_core_to_expansion_nearest_neighbor_graph.svg
fig_p2_dataset_support_heatmap.svg
fig_p2_target_parallel_coordinates.svg
fig_p2_template_support_heatmap.svg
fig_p2_value_risk_null_3d_frontier.svg
fig_p3_bad_null_memory_offdiag_sankey.svg
fig_p3_value_vs_longrisk_after_veto.svg
fig_p3_veto_order_waterfall.svg
fig_p3_veto_removed_good_bad_bar.svg
fig_p4_accepted_region_risk_stack.svg
fig_p4_feature_importance_legal_only.svg
fig_p4_rank_score_by_dataset_after_transport.svg
fig_p4_ranker_pareto_precision_value_LDO.svg
fig_p4_ranker_worst_dataset_panel.svg
fig_p5_accepted_region_by_dataset.svg
fig_p5_accepted_region_by_template.svg
fig_p5_calibration_reliability.svg
fig_p5_certificate_threshold_sensitivity.svg
fig_p5_heldout_vs_calibration_drift.svg
fig_p9_generated_damage_matrix.svg
fig_p9_longrisk_created_by_primitive.svg
fig_p9_memory_offdiag_damage_heatmap.svg
fig_p9_value_direction_lost_by_primitive.svg
```

判断：这些图只用于复核诊断，不构成 official pass。

## 14. No-fake audit

```text
rows_checked = 315
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
v9670_boundary_pass = 1
score_transport_pass = 0
template_balanced_target_pass = 0
veto_order_pass = 0
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
route = R6-GeneratedRouteStoppedNoNewObjective
F0_boundary_regression = 0
F1_score_transport_helps_target_absent = 0
F2_core_expansion_target_ready_certificate_pending = 0
F3_existing_controller_candidate_ready = 0
F4_system_legal_controller_pass = 0
F5_dataset_invariant_rank_impossible = 0
F6_generated_route_stopped_no_new_objective = 1
F7_generated_route_restart_allowed = 0
F8_no_usable_functional_update_route = 0
F9_system_not_official = 1
F10_base_acc_catastrophic = 0
primary_blocker = generated_route_stopped_no_new_objective
```

## 15. Hash

| artifact | SHA256 |
|---|---|
| plan | `a9d8d465147360d5f0e2e8e9f27bd2ccbc2d67f7b044b22aeb40cc51fcf9a695` |
| runner | `4b2b2eb80cb35ac725e5f319a9a68b4dbb6c9ffa29b801eeefefe5cd7a2332b1` |
| run manifest | `de67b898412dd0875b3aed6e0ea60e630c5cf38ed59072df0bfca8ae6068eac6` |
| route / P10 | `5e785dc9f03de5078cea1c704d20fd041dbd38cf878cf41ce65a7883ee6620ea` |
| P0 boundary | `a6f282fbd6b8d1678a73dac7b255a317485dd737fd8385b0cc7b977cacc3d00f` |
| P0 field legality | `62e37802e62e9fbb753800c2d46527daba56b85c1d07ac6f4aeacaf4d95529ba` |
| P1 transport | `2adaa6de2f151d1c73efa17a8e329484782385968b5349a4a213d6db3e3a41e8` |
| P1 distribution | `cd7b7f4f4e634b1ab099f77c17634d689a324e8aa5a4e1ec6f71e81dd8955625` |
| P2 target lattice | `629f7acd6c7f67996d5d0762ccfe5f858fb59f2bd3aa8a01c883f5cc58edd4c6` |
| P2 support | `ea16a5e77dcbed1388281f4e45bc5cde6bec4d235b77ff012142ffdca723ef21` |
| P3 veto | `1bbe30ac8d73623bccf03fd80eb063abcf43e4256c82e65dc5b1ae4bdc91e53f` |
| P4 ranker | `eb1653ae2cb9f34902ebf833455d4a84073c6899412d1f1e770aa3bf2bd545c6` |
| P4 feature ledger | `c8a2e0515de82c24688e33ae8034263ff5cc7065ec7c6e281694a0eba8ca3931` |
| P5 certificate | `ee9c4eaa039a7e1ab1ebbf5197a1408532de460f59c6c66cc08e7721598b96a2` |
| P5 sensitivity | `5f6f93d8bf8208166f958a0488f4d724193a70cd94fe47fb038659f8a18e5ded` |
| P6 controller | `1f5985586e13426d9e35374f1887200a26fc09f41bff200a41ed9a7ec6d4d18b` |
| P7 runtime | `bdff6dc97d26ea49046d75aa0fa4782e02a7a67c40a3c9f153f2b991ff42cd06` |
| P8 generated stop | `f3d99c8fdefbbeb101da8a4023176c9cd8be2415ab6f449a7da41eb7448cd28f` |
| P9 damage notebook | `33fd9d3d17c140086784c1ceb143faa936a6f2452085edffac7251d27c25aa68` |
| P11 paired boundary | `2e70e646aefd078e625b07503d01e1e222c17796aa43faee7bbc9fca13de6f70` |
| P12 short/full | `7c6c4809a9a38970354b98ae54fb90a8f349300d5ea82ac65c330fed80568366` |
| Base-Acc Sentinel | `5df304ddd282f8ee42b58f171f0f3b055944187b822e22eee3cd427c1b372735` |
| no-fake audit | `9565fea6a5d0cfc447282b4f922a9df7143f1cfd139487c954e30660fd12c860` |
| contract audit | `45c1b969703227c05a26207a73ab58fc77fb5c0d21ef0646b5dd9c5502d128d6` |
| failure taxonomy | `9d89aa536206d269625f71440da425dc72dec1347ea4bb32891434f092a07d67` |
| stop conditions | `ce0054d23274f5feefde497cb2313edbe22ab82d17651a3fc2a11ee1be185ec2` |
| allowed next gates | `15f20a0d4a874ed040139b8d05694dfbb8ce0651c1eda62438883c251cafeb44` |

## 16. 最终分析结论

v9.6.8 的真实推进是：

```text
v9.6.7:
  dataset shift 被进一步拆清；
  Core + Expansion 仍无法形成 template-balanced target；
  legal feature normalization / risk veto 会牺牲 precision/value；
  generated route stop-rule 阻止 APGU blind run。

v9.6.8:
  系统扫描 dataset-agnostic score transport；
  证明 transport 不能同时保住 macro/pooled precision、positive value、low risk 与 LDO；
  Core + Expansion target lattice v3 仍停在干净但少、数量够但不稳的冲突；
  veto-order 和 dataset-invariant ranker 不能打开 weak/strong controller candidate；
  high-precision certificate 只在 TopK64 diagnostic 层成立，coverage 与 LDO 不过；
  generated route stop-rule 继续成立，且没有 new objective evidence。
```

机制判断：

1. H0 成立：v9.6.7 boundary 被复现，没有跳过 target/ranker/certificate/generated stop。
2. H1 未成立：score transport 没有解决 dataset shift；S0/S1/S2 保留 high precision 但 LDO 高，S5/S7 降低 LDO 但 precision/value 崩。
3. H2 未成立：Core + Expansion lattice v3 没有 official target；core 干净但 coverage 不够，expanded target 又触发 null/offdiag/LDO failure。
4. H3 未成立：veto-order 没有产生可部署 tradeoff；强 veto 会移除大量 good rows。
5. H4 未成立：dataset-invariant ranker v2 没有 strong/weak pass；R8A 质量好但 LDO 高，低 LDO ranker value/precision 低。
6. H5 未成立：certificate v16 没有打开 existing-action controller；C16G precision = `1.0` 但 accepted = `64` 且 LDO = `0.328125`。
7. H6/H7 未打开：existing-action controller 与 selected runtime 均 not_run。
8. H8 成立：generated route stop-rule 继续成立，APGU 不运行，无 fake/proxy APGU rows。
9. H9 成立于诊断层：generated damage notebook 指向 `longrisk_created`，recommendation = `stop_blind_generated_variants`。
10. H10-H12 未打开：没有 system controller，就不能打开 paired replay 或 short/full training。
11. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.6.8 真实执行后停在 `R6-GeneratedRouteStoppedNoNewObjective`：score transport、core-expansion、veto-order、ranker 与 certificate 都没有形成 official controller；APGH/APGL/APGT 的 generated stop-rule 继续触发且没有新 objective evidence，因此 APGU/APGV 不允许 blind run，strict PureKAN functional 仍未成功。
