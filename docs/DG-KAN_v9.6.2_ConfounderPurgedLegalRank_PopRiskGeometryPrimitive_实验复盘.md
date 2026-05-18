# DG-KAN v9.6.2 Confounder-Purged Legal Rank / PopRisk Geometry Primitive 实验复盘

> 本复盘记录 `DG-KAN_v9.6.2_ConfounderPurgedLegalRank_PopRiskGeometryPrimitive_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 confounder diagnostic、rank diagnostic、APGF implementation/smoke、certificate diagnostic、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R4-PayloadPocketStopped
base_candidate = LQ-t2-h256
success_v9620_strict_purekan_functional = False
success_v9620_full_functional = False
success_v9620_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9620_confounder_purged_legal_rank_poprisk_geometry_primitive_first_20260515T120000Z/
```

核心结论：

1. P0 复现 v9.6.1 boundary：source route = `R2-GroupPocketConfounded_StopPayloadPocketRoute`，payload0 confounded supported = `1`，v9.6.1 system pass = `0`。
2. P1 payload pocket stop audit 继续成立：matched-pair V lift = `-0.5198887260597593`，GradeAB lift = `-0.6134615384615385`，payload0 intervention precision = `0.009615384615384616`，memory projection longrisk = `0.8557692307692307`。
3. P1 明确 `payload_norm_bucket_allowed_as_selector = 0`，只允许作为 stratification；payload pocket route stop = `1`。
4. P2 hidden confounder search 未找到可 official 的 strong confounder：strong confounder count = `0`；best axis 仍是 `payload_norm_bucket`，drop if removed = `0.11494252873563218`。
5. P3 group-balanced target map 未过 official：best = `T2-ValuePositiveNoLongRisk`，count = `97`，coverage = `0.03372739916550765`，V LCB = `0.16622185363738462`，longrisk UCB = `0.0`，但 target_map_pass = `0`。
6. P4 legal feature deconfounding 未过：best = `COMBO-ValueRiskMemoryCover`，TopK87 precision = `0.6551724137931034`，V LCB = `0.049639752010919136`，longrisk UCB = `0.0`，但 LDO drop = `0.22988505747126436`。
7. P5 two-head ranker 仍是强 pooled diagnostic 但 group unstable：best = `RANK5-invariant-risk-minimization`，TopK87 GradeAB precision = `0.8735632183908046`，V LCB = `0.14111334880346277`，longrisk UCB = `0.0`，LDO drop = `0.37931034482758624`，max group share = `1.0`。
8. P6 rank-safe certificate v10 未过：best = `CERT10-FixedTopKGroupBalanced`，accepted = `87`，coverage = `0.030250347705146036`，GradeAB precision = `0.8735632183908046`，V LCB = `0.14111334880346277`，longrisk UCB = `0.0`，但 certificate pass = `0`，原因仍是 group drop / max group share。
9. P7 existing-action controller not_run：reason = `P6_certificate_not_passed`。
10. P8 generated damage autopsy 过 diagnostic：damage rows = `1536`，dominant damage mode = `D7-risk-veto-ineffective`，fraction = `0.7369791666666666`，longrisk created rate = `0.7369791666666666`，Damage V LCB = `-0.7420588191683418`。
11. P9 APGF1-APGF8 implementation 过：generated actions = `512`，payload/certificate hash missing = `0 / 0`，action apply L∞ max = `0.0`。
12. P9 APGF branch-horizon materializer 过：expected/actual rows = `12288 / 12288`，unresolved exception = `0`，duplicate rows = `0`，quality audit pass = `1`。
13. P9 APGF outcome 未过：best = `APGF5-CoverEntropyPreservingDelta`，GradeAB precision = `0.03125`，V LCB = `-0.33527855012819596`，h240 longrisk UCB = `0.828904255301089`。
14. P10 APGF certificate/controller 未过：best = `APGF-CERT-APGF5`，TopK64 GradeAB precision = `0.03125`，V LCB = `-0.33527855012819596`，longrisk UCB = `0.828904255301089`。
15. P12-P14 gate-blocked：没有 selected runtime、leaveout、paired replay 或 short/full boundary。
16. Base-Acc Sentinel 复用前序真实 sentinel：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
17. 当前 primary blocker：`payload_pocket_confounded_route_stopped`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9620_confounder_purged_legal_rank_poprisk_geometry_primitive.py` | v9.6.2 runner；读取 v9.6.1/v9.6.0/v9.5.9/v9.5.8/v9.5.5 artifacts，执行 confounder ledger、payload pocket stop audit、hidden confounder search、group-balanced target map、feature/rank/certificate gate、generated damage autopsy、APGF primitive 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9620_confounder_purged_legal_rank_poprisk_geometry_primitive.py
```

正式运行：

```bash
python experiments/run_v9620_confounder_purged_legal_rank_poprisk_geometry_primitive.py \
  --out-dir results/real_rerun_20260506/v9620_confounder_purged_legal_rank_poprisk_geometry_primitive_first_20260515T120000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --apgf-actions-per-primitive 64
```

运行结果：

```json
{
  "apgf_generated_actions": 512,
  "apgf_rows": 12288,
  "best_apgf_V_integrated_LCB": -0.33527855012819596,
  "best_apgf_primitive": "APGF5-CoverEntropyPreservingDelta",
  "best_ranker": "RANK5-invariant-risk-minimization",
  "best_ranker_TopK87_precision": 0.8735632183908046,
  "out_dir": "results/real_rerun_20260506/v9620_confounder_purged_legal_rank_poprisk_geometry_primitive_first_20260515T120000Z",
  "payload_pocket_route_stop": 1,
  "primary_blocker": "payload_pocket_confounded_route_stopped",
  "route": "R4-PayloadPocketStopped",
  "system_legal_controller_pass": 0
}
```

## 2. Route

`route_decision_v9620.json` 摘要：

```json
{
  "route": "R4-PayloadPocketStopped",
  "source_route_v9610": "R2-GroupPocketConfounded_StopPayloadPocketRoute",
  "p0_pass": 1,
  "payload0_confounded_supported": "1",
  "payload_pocket_route_stop": 1,
  "strong_confounder_count": 0,
  "best_confounder_axis": "payload_norm_bucket",
  "target_map_pass": 0,
  "best_target_id": "T2-ValuePositiveNoLongRisk",
  "feature_deconfounding_pass": 0,
  "two_head_ranker_pass": 0,
  "best_ranker_id": "RANK5-invariant-risk-minimization",
  "best_ranker_TopK87_GradeAB_precision": 0.8735632183908046,
  "best_ranker_LDO_drop": 0.37931034482758624,
  "legal_rank_signal_but_group_unstable": 1,
  "rank_safe_certificate_v10_pass": 0,
  "existing_controller_pass": 0,
  "damage_autopsy_pass": 1,
  "ap_generated_route": "source_OOD",
  "apgf_implementation_pass": 1,
  "apgf_quality_audit_pass": 1,
  "apgf_weak_pass": 0,
  "best_apgf_primitive": "APGF5-CoverEntropyPreservingDelta",
  "best_apgf_GradeAB_precision": 0.03125,
  "best_apgf_V_integrated_LCB": -0.33527855012819596,
  "best_apgf_h240_longrisk_UCB": 0.828904255301089,
  "apgf_certificate_pass": 0,
  "selected_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "payload_pocket_confounded_route_stopped"
}
```

判断：v9.6.2 的主结论不是 APGF engineering fail，而是 v9.6.1 已经反证的 payload pocket route 被正式停止；legal rank 仍有强 TopK diagnostic，但 group drop / max group share 让它不能成为 official controller。APGF 完整落盘，但 outcome 继续 value-negative / high-longrisk。

## 3. P0 v9.6.1 boundary

Artifact：

```text
p0_boundary_reproduction_v9620.csv
```

Summary：

```text
source_route_v9610 = R2-GroupPocketConfounded_StopPayloadPocketRoute
payload0_confounded_supported_v9610 = 1
H1_falsified_v9610 = 1
matched_pair_count_total_v9610 = 520
best_ranker_v9610 = GIR9-PairwiseWithinGroupRanker
best_ranker_TopK87_precision_v9610 = 0.8505747126436781
best_ranker_LDO_drop_v9610 = 0.3563218390804597
best_apge_primitive_v9610 = APGE6-SignalChannelSNRPreconditionedDelta
best_apge_V_lcb_v9610 = -0.8693640105318599
best_apge_longrisk_ucb_v9610 = 0.8825326673766913
system_legal_controller_pass_v9610 = 0
no_fake_v9610/no_proxy_v9610 = 1 / 1
p0_pass = 1
```

判断：P0 pass。v9.6.2 没有跳过 v9.6.1 的 payload pocket confounded / APGE failure boundary。

## 4. P1 confounder ledger 与 payload pocket stop

Artifacts：

```text
p1_confounder_ledger_v9620.csv
p1_payload_pocket_stop_audit.csv
field_legality_ledger_v9620.csv
```

Field legality summary：

```text
field_count = 16
green/yellow/red = 10 / 3 / 3
payload_norm_bucket_allowed_as_selector = 0
payload_norm_bucket_allowed_as_stratification = 1
outcome_derived_field_used_for_official = 0
field_legality_ledger_pass = 1
```

Payload stop summary：

```text
matched_pair_V_lift = -0.5198887260597593
matched_pair_GradeAB_lift = -0.6134615384615385
payload0_intervention_precision = 0.009615384615384616
payload0_normalized_longrisk = 0.8557692307692307
memory_projection_longrisk = 0.8557692307692307
payload0_causal_lift_supported = 0
payload0_confounded_supported = 1
payload_pocket_route_stop = 1
```

判断：P1 是本轮 route 的首要 gate。payload_norm_bucket 不允许作为 selector；payload pocket 已被停止为 confounded route，不能继续沿 payload0 pocket 写 official controller。

## 5. P2 hidden confounder search

Artifact：

```text
p2_hidden_confounder_search.csv
```

Summary：

```text
axis_count = 14
strong_confounder_count = 0
best_axis = payload_norm_bucket
best_drop_if_removed = 0.11494252873563218
best_matched_pair_count = 158
best_SMD_after_matching = 0.0
best_max_topK_group_share = 1.0
confounder_unresolved = 1
```

判断：P2 没有找到能解除 R4 的 strong confounder。payload_norm_bucket 仍解释 pocket，但只能作为 stratification/diagnostic。

## 6. P3 group-balanced target map

Artifact：

```text
p3_group_balanced_target_map_v2.csv
```

Summary：

```text
target_count = 10
official_density_target_count = 0
best_target_id = T2-ValuePositiveNoLongRisk
best_global_count = 97
best_global_coverage = 0.03372739916550765
best_V_integrated_LCB = 0.16622185363738462
best_longrisk_UCB = 0.0
target_map_pass = 0
```

Representative rows：

| target | count | coverage | V LCB | longrisk UCB |
|---|---:|---:|---:|---:|
| `T2-ValuePositiveNoLongRisk` | `97` | `0.03372739916550765` | `0.16622185363738462` | `0.0` |
| `T3-MemorySafeValuePositive` | `90` | `0.03129346314325452` | `0.1601306386208506` | `0.0` |
| `T1-GradeAB` | `79` | `0.027468706536856746` | `0.16201929527024647` | `0.0` |

判断：target map 有 value/risk-positive diagnostic，但没有 official target closure。

## 7. P4-P6 legal feature / rank / certificate

P4 artifact：

```text
p4_legal_feature_deconfounding_v2.csv
```

P4 summary：

```text
feature_count = 23
best_feature_id = COMBO-ValueRiskMemoryCover
best_TopK87_precision = 0.6551724137931034
best_TopK87_V_LCB = 0.049639752010919136
best_TopK87_longrisk_UCB = 0.0
best_LDO_drop = 0.22988505747126436
best_LSO_drop = 0.011494252873563204
legal_feature_deconfounding_pass = 0
```

P5 artifact：

```text
p5_two_head_ranker_value_risk_memory.csv
```

P5 summary：

```text
ranker_count = 9
official_ranker_count = 0
strong_pooled_ranker_count = 2
best_ranker_id = RANK5-invariant-risk-minimization
best_TopK87_GradeAB_precision = 0.8735632183908046
best_TopK87_V_LCB = 0.14111334880346277
best_TopK87_longrisk_UCB = 0.0
best_LDO_drop = 0.37931034482758624
best_LSO_drop = 0.02298850574712652
best_max_group_share = 1.0
legal_rank_signal_but_group_unstable = 1
two_head_ranker_pass = 0
```

P6 artifact：

```text
p6_rank_safe_certificate_v10.csv
```

P6 summary：

```text
certificate_count = 6
official_certificate_count = 0
best_certificate_id = CERT10-FixedTopKGroupBalanced
best_ranker_id = RANK5-invariant-risk-minimization
best_accepted_count_heldout = 87
best_coverage_heldout = 0.030250347705146036
best_GradeAB_precision_heldout = 0.8735632183908046
best_V_LCB_heldout = 0.14111334880346277
best_longrisk_UCB_heldout = 0.0
best_LDO_drop = 0.37931034482758624
best_LSO_drop = 0.02298850574712652
rank_safe_certificate_v10_pass = 0
```

判断：P5/P6 是本轮最强正向 diagnostic：TopK87 的 GradeAB/value/risk 都好于前几轮，但 max group share = `1.0` 且 LDO drop 仍高，不能打开 existing-action controller。

## 8. P7 existing-action controller

Artifact：

```text
p7_existing_action_minimal_controller.csv
```

Status：

```text
status = not_run
reason = P6_certificate_not_passed
source_controller_pass = 0
system_controller_candidate_pass = 0
```

判断：没有把 P5/P6 rank diagnostic 写成 official controller。

## 9. P8 generated damage autopsy

Artifact：

```text
p8_generated_damage_autopsy_v2.csv
```

Summary：

```text
damage_row_count = 1536
damage_autopsy_pass = 1
ap_generated_route = source_OOD
dominant_damage_mode = D7-risk-veto-ineffective
dominant_damage_mode_fraction = 0.7369791666666666
longrisk_created_rate = 0.7369791666666666
new_positive_created_rate = 0.0026041666666666665
offdiag_fail_rate = 0.8203125
cover_collapse_rate = 0.25
memory_fail_rate = 0.0
Damage_V_LCB = -0.7420588191683418
```

判断：P8 支持 generator OOD route：已有 generated families 的主要问题仍是 risk veto ineffective / longrisk created。

## 10. P9-P10 APGF primitive 与 certificate

P9 implementation artifact：

```text
p9_apgf_population_risk_geometry_primitive.csv
```

P9 implementation summary：

```text
primitive_count = 8
generated_action_count_expected/actual = 512 / 512
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
negative_control_present = 1
apgf_implementation_pass = 1
```

P9 branch-horizon artifact：

```text
p9_apgf_branch_horizon_outcome.csv
```

P9 materializer summary：

```text
branch_horizon_rows_expected/actual = 12288 / 12288
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1
rows_per_sec = 14.846036990575348
wallclock_sec = 827.6956340470351
unresolved_exception_count = 0
duplicate_row_count = 0
label_exclusivity_violation = 0
quality_audit_pass = 1
```

P9 outcome summary：

```text
best_primitive_id = APGF5-CoverEntropyPreservingDelta
best_GradeAB_precision = 0.03125
best_V_integrated_LCB = -0.33527855012819596
best_h240_longrisk_UCB = 0.828904255301089
best_new_positive_created_rate = 0.015625
best_Damage_V_LCB = -0.20510485204051287
apgf_weak_pass = 0
apgf_strong_pass = 0
```

Per primitive：

| primitive | GradeAB precision | V LCB | h240 longrisk UCB | new positive | pass |
|---|---:|---:|---:|---:|---:|
| `APGF1-PopRiskDiagonalSNRGate` | `0.03125` | `-0.37114044434687954` | `0.828904255301089` | `0.03125` | `0` |
| `APGF2-EdgeGroupSNRMask` | `0.0` | `-0.40801096700264794` | `0.8425830323796916` | `0.0` | `0` |
| `APGF3-MemoryAnchoredResidual` | `0.015625` | `-0.35971651779563524` | `0.8010605393336523` | `0.0` | `0` |
| `APGF4-OffdiagPopulationRiskGuard` | `0.03125` | `-0.4184294365785832` | `0.7869099970100811` | `0.015625` | `0` |
| `APGF5-CoverEntropyPreservingDelta` | `0.03125` | `-0.33527855012819596` | `0.828904255301089` | `0.015625` | `0` |
| `APGF6-SymmetricBoundaryCorrection` | `0.015625` | `-0.3319360087696793` | `0.7726149255507754` | `0.015625` | `0` |
| `APGF7-RankImitationWithRiskVeto` | `0.0` | `-0.37435360928803063` | `0.7581802303291282` | `0.0` | `0` |
| `APGF8-NegativeControlShuffled` | `0.015625` | `-0.3526443530868811` | `0.8010605393336523` | `0.015625` | `0` |

P10 artifact：

```text
p10_apgf_certificate_controller.csv
```

P10 summary：

```text
certificate_count = 8
best_certificate_id = APGF-CERT-APGF5
best_primitive_id = APGF5-CoverEntropyPreservingDelta
best_TopK64_GradeAB_precision = 0.03125
best_V_LCB = -0.33527855012819596
best_longrisk_UCB = 0.828904255301089
generated_action_controller_pass = 0
apgf_certificate_pass = 0
```

判断：APGF 工程链路和 replay materializer 完整闭合，但 generated frontier 完全没有打开。不能把 12288 行完整 replay 写成 APGF pass。

## 11. P12-P14 boundary

| artifact | status / reason |
|---|---|
| `p12_selected_runtime.csv` | `not_run`, `controller_not_selected` |
| `p13_leaveout_paired_replay_boundary.csv` | `not_run`, `P12_runtime_not_selected` |
| `p14_short_full_boundary.csv` | `not_run`, `P13_paired_replay_not_open` |

判断：没有 controller，因此 selected runtime、leaveout、paired replay、short/full 均未打开。

## 12. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9620.csv
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

## 13. No-fake audit

```text
rows_checked = 17452
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
v9610_boundary_pass = 1
payload_pocket_route_stop = 1
confounder_search_strong_count = 0
target_map_pass = 0
feature_deconfounding_pass = 0
two_head_ranker_pass = 0
rank_safe_certificate_pass = 0
existing_controller_pass = 0
damage_autopsy_pass = 1
apgf_implementation_pass = 1
apgf_branch_horizon_pass = 1
apgf_weak_pass = 0
apgf_certificate_pass = 0
selected_runtime/system = 0/0
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
route = R4-PayloadPocketStopped
F0_boundary_reproduction_failed = 0
F1_payload_pocket_confounded_stopped = 1
F2_existing_rank_group_unstable = 1
F3_existing_controller_fail = 1
F4_apgf_generated_primitive_fail = 1
F5_certificate_fail = 1
F6_runtime_blocked = 1
F7_system_not_official = 1
F8_base_acc_catastrophic = 0
primary_blocker = payload_pocket_confounded_route_stopped
```

## 14. Hash

| artifact | SHA256 |
|---|---|
| plan | `7eb5ddb08e4f78dfbf1aaf56966bb5050b1a3e9961633e85ecd09523b9499a5a` |
| runner | `1e44eb19f9e81783236b6cdcfe9eaa29920f4a0a4645e439b1b1c5d774ba50a0` |
| run manifest | `d3a7aa49332c674625531b63995edebb1d0253ce592962eca69ae3d88e4b208a` |
| route | `a21651d457e181004c48faf66070f7a4cbef623d434513b12d6d86d3ab54779f` |
| P0 boundary | `9a511eb39188077ac1215fa3f08d6f944e2b37b8639e8410037ca5d81ec12a38` |
| P1 confounder ledger | `5f11f62b72c23b6f0d792955fc23b998afd5910274fdae00c14cf16dc7f20453` |
| P1 payload stop | `b98a02f97cb74093071932081a49d5bc93aa67454883f6275a079831e01c9a7e` |
| field legality ledger | `1426173027f8e80bcf91e2e165c31e4e841ce02318e063e8fc0d2b3c12767d5d` |
| P2 hidden confounder | `ae24452f1ff0ebffca82763075e693d62759274ee1703fd2dbbc0313e956e80f` |
| P3 target map | `29ff1d7db239dbec41d18b464492b2861bdbec4c4f5c553ef516a3a8b65c26c8` |
| P4 feature | `5aa48acb0bf492434f89990ce7767c5d104912cdb5acf48073371bc521deadca` |
| P5 ranker | `34f9a07f5b9ffaa9047087b23d6e055ac09864b4ef55a5561c670e0bb7ec6a49` |
| P6 certificate | `31875fa57c652c43ef4f383d0a7df5119d2f3877277ae04de9c45cc601efbfcf` |
| P7 controller | `958a91d651662f22df48cf0a9290c4e8eef0e81ac69671418b6690b6a4fbd23e` |
| P8 damage | `2bea2737d3fe5fbcc98bc397e1b85667c5759170a18fd12a838dc5254650086e` |
| P9 APGF implementation | `c56d004203ab3cc35f346f43725d6c7f5ce3944d4344b1eb6c22b3ceef3d071c` |
| P9 APGF smoke/outcome | `9afc141f90b7d53a89d6ab2490b6631e76e7d624889a7f665dac3237a8bf1f47` |
| P10 APGF certificate | `3baca7b1a81b1cb7e639f3aee4355531b9ec00ea539876def35246d50607065a` |
| P11 decision route | `92a62e2ffc9cc166f528c67fd0a8c64dd1226a8210536bb47d785ca0124c2bd6` |
| P12 runtime | `6125221dd281afa76f672d6ae18e1c97b82ac96ac889c38dc7b02bcfd8db4898` |
| P13 leaveout/paired | `d9045b91467b9a5bbac7677cbea8c32a5e745bc43de01c45cbbabd001cedc4e3` |
| P14 short/full | `b49cd97508745107e192247231989dd801409e6121e551c754d6ba23633383b2` |
| Base-Acc Sentinel | `bcc25b4f61842f5031cdd0b53948c8eb5310833b4c84e0a64b2995bab6cbdca3` |
| no-fake audit | `e2b6f1cc331f033aaa2be6aa72b049d0c963243dd35202c9e837ec9af43307df` |
| contract audit | `b97e85bc8276c916e2ffd5d9c9c20bac5b7632a2203dfa86eb2e7b867bec88ba` |
| failure taxonomy | `ca0d54e5f1cb1f351efc48b1c4534b54e1590259e8ed80127d351e30a9763305` |

## 15. 最终分析结论

v9.6.2 的真实推进是：

```text
v9.6.1:
  payload0 pocket 经 intervention clone replay 与 matched-pair attribution 后不支持 causal mechanism；
  APGE generated frontier 失败；
  route 停在 payload pocket confounded。

v9.6.2:
  将 payload_norm_bucket 从 selector route 中正式停用，只保留为 stratification/diagnostic；
  搜索 hidden confounder，但没有找到可 official 的 strong confounder；
  group-balanced target / legal feature / two-head ranker 仍显示强 diagnostic；
  但 ranker/certificate 继续失败于 group instability / max group share；
  对 generated families 做 damage autopsy，确认主要是 risk-veto ineffective / source OOD；
  APGF1-APGF8 工程链路和 12288 行 branch-horizon replay 完整闭合；
  但 APGF outcome 仍 value-negative / high-longrisk，没有 generated frontier。
```

机制判断：

1. H0 成立：v9.6.1 boundary 被复现，没有跳过 payload pocket confounded / APGE failure。
2. H1 成立于 route 层：payload pocket stop audit 继续支持 confounded route，payload_norm_bucket 不允许作为 selector。
3. H2 未成立：hidden confounder search 没有找到 strong confounder，strong confounder count = `0`。
4. H3 未成立：group-balanced target map 有干净 target，但 official target_map_pass = `0`。
5. H4 部分成立：legal feature/rank signal 仍强；best ranker TopK87 GradeAB precision = `0.8736`，V LCB > 0，longrisk UCB = `0.0`。
6. H5 未成立：rank/certificate 仍 group-unstable，best LDO drop = `0.3793`，max group share = `1.0`。
7. H6 未打开：existing-action controller not_run，selected runtime 不打开。
8. H7 成立于 diagnostic 层：generated damage 主要是 `D7-risk-veto-ineffective`，longrisk_created_rate = `0.7370`。
9. H8/H9 成立于 implementation 层：APGF payload/certificate/action apply 与 branch-horizon materializer 完整闭合。
10. H10 未成立：APGF 没有生成 value-positive / horizon-safe frontier；best APGF5 的 V LCB 为负且 longrisk UCB 很高。
11. H11-H13 未打开：没有 controller/runtime/APGF official pass，就不能打开 leaveout、paired replay 或 short/full training。
12. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.6.2 真实执行后停在 `R4-PayloadPocketStopped`：payload pocket route 被正式停止为 confounded diagnostic，legal rank 仍有强 TopK 信号但 group instability 未解；APGF 虽完整落盘，却继续生成 value-negative / high-longrisk 动作，因此 strict PureKAN functional 仍未成功。
