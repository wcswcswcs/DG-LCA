# DG-KAN v9.6.1 Group Pocket Causal Intervention / Group-Invariant Controller / Memory-Safe Primitive 实验复盘

> 本复盘记录 `DG-KAN_v9.6.1_GroupPocketCausalIntervention_GroupInvariantController_MemorySafePrimitive_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 group-pocket diagnostic、intervention clone diagnostic、rank diagnostic、APGE implementation/smoke、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R2-GroupPocketConfounded_StopPayloadPocketRoute
base_candidate = LQ-t2-h256
success_v9610_strict_purekan_functional = False
success_v9610_full_functional = False
success_v9610_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9610_group_pocket_causal_intervention_group_invariant_controller_memory_safe_primitive_first_20260515T110000Z/
```

核心结论：

1. P0 复现 v9.6.0 boundary：source route = `R1-GroupPocketMechanismUnresolved`，system pass = `0`，v9.6.0 APGD weak pass = `0`。
2. P1 group-pocket intervention panel 完整落盘：source actions = `104`，intervention clones = `520`，branch-horizon rows = `12480 / 12480`。
3. P1 clone replay 没有支持 payload0 causal mechanism：`H1_strong_pass = 0`，`H1_falsified = 1`；payload0 normalized GradeAB precision = `0.009615384615384616`，memory projection longrisk = `0.8557692307692307`。
4. P2 matched-pair causal attribution 未过：matched pairs = `520`，mean V lift = `-0.5198887260597593`，mean GradeAB lift = `-0.6134615384615385`，payload0 causal lift supported = `0`，payload0 confounded supported = `1`。
5. P3 group-balanced target density 仍只有 weak diagnostic：best = `ValuePositiveNoLongRisk`，count = `97`，coverage = `0.03372739916550765`，official density pass = `0`。
6. P4 legal feature deconfounding 未过：best = `ControlTransferImprovement`，within-group AUC mean = `0.9842536058137868`，TopK64 group-balanced precision = `0.4603174603174603`，但 longrisk UCB = `0.5176365555683604`。
7. P5 value-rank/risk-memory veto 未过：best = `VR4-PairwiseValue-RiskVeto`，TopK87 GradeAB precision = `0.8390804597701149`，V LCB = `0.13479400136362346`，longrisk UCB = `0.0`，但 leaveout drop = `0.3448275862068965` 且 max group share = `1.0`。
8. P6 group-invariant pairwise ranker v3 未过：best = `GIR9-PairwiseWithinGroupRanker`，TopK87 precision = `0.8505747126436781`，V LCB = `0.15630165181090255`，LDO/LSO drop = `0.3563218390804597 / 0.3563218390804597`。
9. P7 rank-safe certificate v9 未过：best = `RC7-TopKFixedCountGroupBalanced`，heldout accepted = `87`，coverage = `0.10046189376443418`，GradeAB precision = `0.28735632183908044`，longrisk UCB = `0.369780988566077`。
10. P8/P9 gate-blocked：existing-action controller 未选中，selected runtime 未打开。
11. P10 APGA/APGD OOD causal autopsy 过 diagnostic：damage rows = `1024`，dominant damage mode = `memory_offdiag_longrisk_created`，fraction = `0.7294921875`。
12. P11/P12 APGE1-APGE8 implementation 与 branch-horizon materializer 过：generated actions = `512`，payload/certificate hash missing = `0 / 0`，rows = `12288 / 12288`，quality audit pass = `1`。
13. P13 APGE outcome 未过：best = `APGE6-SignalChannelSNRPreconditionedDelta`，GradeAB precision = `0.03125`，V_integrated LCB = `-0.8693640105318599`，h240 longrisk UCB = `0.8825326673766913`。
14. P14/P15 gate-blocked：没有 official system、leaveout、paired replay 或 short/full training。
15. Base-Acc Sentinel 复用前序真实 sentinel：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
16. 当前 primary blocker：`payload_pocket_confounded_or_not_causal`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9610_group_pocket_causal_intervention_group_invariant_controller_memory_safe_primitive.py` | v9.6.1 runner；读取 v9.6.0/v9.5.9/v9.5.8/v9.5.5 artifacts，执行 group-pocket intervention、matched-pair attribution、group-balanced target/rank/certificate、APGA/APGD OOD autopsy、APGE primitive 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9610_group_pocket_causal_intervention_group_invariant_controller_memory_safe_primitive.py
```

正式运行：

```bash
python experiments/run_v9610_group_pocket_causal_intervention_group_invariant_controller_memory_safe_primitive.py \
  --out-dir results/real_rerun_20260506/v9610_group_pocket_causal_intervention_group_invariant_controller_memory_safe_primitive_first_20260515T110000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --intervention-source-count 104 --apge-actions-per-primitive 64
```

运行结果：

```json
{
  "apge_generated_actions": 512,
  "apge_rows": 12288,
  "best_apge_V_integrated_LCB": -0.8693640105318599,
  "best_apge_primitive": "APGE6-SignalChannelSNRPreconditionedDelta",
  "intervention_clone_count": 520,
  "matched_pair_count_total": 520,
  "out_dir": "results/real_rerun_20260506/v9610_group_pocket_causal_intervention_group_invariant_controller_memory_safe_primitive_first_20260515T110000Z",
  "primary_blocker": "payload_pocket_confounded_or_not_causal",
  "route": "R2-GroupPocketConfounded_StopPayloadPocketRoute",
  "system_legal_controller_pass": 0
}
```

## 2. Route

`route_decision_v9610.json` 摘要：

```json
{
  "route": "R2-GroupPocketConfounded_StopPayloadPocketRoute",
  "source_route_v9600": "R1-GroupPocketMechanismUnresolved",
  "p0_pass": 1,
  "P1_intervention_pass": 1,
  "H1_strong_pass": 0,
  "H1_falsified": 1,
  "payload0_confounded_supported": 1,
  "matched_pair_count_total": 520,
  "group_balanced_target_density_pass": 0,
  "feature_invariant_pass": 0,
  "value_rank_risk_memory_veto_pass": 0,
  "group_invariant_pairwise_ranker_v3_pass": 0,
  "rank_safe_certificate_v9_pass": 0,
  "existing_action_controller_pass": 0,
  "selected_runtime_pass": 0,
  "P10_ood_causal_autopsy_pass": 1,
  "dominant_damage_mode": "memory_offdiag_longrisk_created",
  "apge_implementation_pass": 1,
  "apge_quality_audit_pass": 1,
  "apge_weak_pass": 0,
  "best_apge_primitive": "APGE6-SignalChannelSNRPreconditionedDelta",
  "best_apge_GradeAB_precision": 0.03125,
  "best_apge_V_integrated_LCB": -0.8693640105318599,
  "best_apge_h240_longrisk_UCB": 0.8825326673766913,
  "system_legal_controller_pass": 0,
  "primary_blocker": "payload_pocket_confounded_or_not_causal"
}
```

判断：v9.6.1 的关键结论是 payload0 pocket 没有被 intervention/matched-pair causal attribution 支持为可利用机制；它更像 confounded rank pocket。因此 route 优先停在 `R2-GroupPocketConfounded_StopPayloadPocketRoute`，后续 APGE 只作为真实落盘 diagnostic，不进入 official controller。

## 3. P0 v9.6.0 boundary

Artifact：

```text
p0_v9600_boundary_reproduction.csv
```

Summary：

```text
source_route_v9600 = R1-GroupPocketMechanismUnresolved
system_legal_controller_pass_v9600 = 0
p1_group_pocket_pass_v9600 = 0
payload_norm_bucket_drop_explained_fraction_v9600 = 0.18181818181818182
payload0_removal_precision_drop_v9600 = 0.8505747126436781
matched_pair_count_v9600 = 0
best_pairwise_ranker_v9600 = GIR9-PairwiseWithinGroupRanker
best_pairwise_TopK87_precision_v9600 = 0.8505747126436781
best_pairwise_LDO_drop_v9600 = 0.3563218390804597
rank_safe_certificate_pass_v9600 = 0
best_apgd_primitive_v9600 = APGD2-MemorySafeResidualBlend
best_apgd_gradeab_precision_v9600 = 0.0625
best_apgd_V_lcb_v9600 = -0.875224386072408
best_apgd_longrisk_ucb_v9600 = 0.8560881119635937
p0_pass = 1
```

判断：P0 pass。v9.6.1 没有跳过 v9.6.0 的 group-pocket unresolved / APGD failure boundary。

## 4. P1 group-pocket intervention panel

Artifacts：

```text
p1_group_pocket_intervention_panel.csv
p1_intervention_clone_trace.csv
```

Summary：

```text
source_action_count = 104
intervention_clone_count = 520
clone_type_count = 5
branch_horizon_rows_expected/actual = 12480 / 12480
branch_horizon_completion = 1
matched_source_clone_rows = 520
payload_norm_bucket_intervention_effect_measured = 1
negative_control_divergence_present = 1
no_transform_metric_abs_diff_max = 5.958666403079405
payload0_normalized_GradeAB_precision = 0.009615384615384616
nonpayload_normalized_GradeAB_precision = 0.0
memory_projection_GradeAB_precision = 0.009615384615384616
memory_projection_longrisk_rate = 0.8557692307692307
P1_intervention_pass = 1
H1_strong_pass = 0
H1_falsified = 1
```

Per clone diagnostic：

| clone | n | GradeAB | longrisk | V mean |
|---|---:|---:|---:|---:|
| `no_transform_clone` | `104` | `0.028846153846153848` | `0.7596153846153846` | `-0.9858645945751611` |
| `norm_normalized_clone` | `104` | `0.009615384615384616` | `0.8076923076923077` | `-0.9930388468711714` |
| `direction_preserved_small_scale_clone` | `104` | `0.0` | `0.8365384615384616` | `-0.976261706698614` |
| `direction_shuffled_negative_control` | `104` | `0.009615384615384616` | `0.8461538461538461` | `-1.1188992289788213` |
| `memory_preserving_projection_clone` | `104` | `0.009615384615384616` | `0.8557692307692307` | `-1.1554484388860424` |

判断：P1 materialization/replay 完整，但 intervention 没有把 payload0 pocket 转成可用 causal mechanism。normalized/memory-projected clones 的 GradeAB 很低且 longrisk 很高，因此 `H1_falsified = 1`。

## 5. P2 matched-pair causal attribution

Artifacts：

```text
p2_matched_pair_causal_attribution.csv
p2_pair_lift_trace.csv
```

Summary：

```text
matched_pair_count_total = 520
exact_or_coarsened_pair_count = 0
intervention_pair_count = 520
mean_V_lift = -0.5198887260597593
mean_gradeab_lift = -0.6134615384615385
longrisk_lift_CI_high = 0.5901441508280382
sign_test_p_value = 6.849716436251349e-32
bootstrap_ci_lift_low/high = -0.6375581843967891 / -0.40221926772272953
P2_matched_pair_pass = 0
payload0_causal_lift_supported = 0
payload0_confounded_supported = 1
```

判断：P2 是 route 的 terminal blocker。干预配对后 lift 方向为负，payload0 causal lift 不成立；本轮明确支持 `payload0_confounded_supported = 1`。

## 6. P3-P7 rank / target / certificate

P3 artifact：

```text
p3_group_balanced_target_density_v2.csv
```

P3 summary：

```text
target_candidate_count = 10
weak_density_target_count = 2
official_density_target_count = 0
best_target_id = ValuePositiveNoLongRisk
best_global_count = 97
best_global_coverage = 0.03372739916550765
best_group_axis = dataset
best_group_min_density = 0.029197080291970802
best_positive_group_coverage = 1.0
group_balanced_target_density_weak_pass = 1
group_balanced_target_density_pass = 0
```

P4 artifact：

```text
p4_legal_feature_deconfounding_v3.csv
```

P4 summary：

```text
best_feature_id = ControlTransferImprovement
best_AUC_within_group_mean = 0.9842536058137868
best_AUC_within_group_min = 0.9130434782608695
best_TopK64_group_balanced_precision = 0.4603174603174603
best_longrisk_UCB_group_balanced = 0.5176365555683604
feature_invariant_pass = 0
```

P5 artifact：

```text
p5_value_rank_risk_memory_veto_ranker.csv
```

P5 summary：

```text
best_ranker_id = VR4-PairwiseValue-RiskVeto
best_TopK87_GradeAB_precision = 0.8390804597701149
best_TopK87_V_integrated_LCB = 0.13479400136362346
best_TopK87_h240_longrisk_UCB = 0.0
best_leaveout_drop_max = 0.3448275862068965
best_max_group_share = 1.0
value_rank_risk_memory_veto_pass = 0
```

P6 artifact：

```text
p6_group_invariant_pairwise_ranker_v3.csv
```

P6 summary：

```text
best_ranker_id = GIR9-PairwiseWithinGroupRanker
best_TopK87_precision = 0.8505747126436781
best_TopK87_V_integrated_LCB = 0.15630165181090255
best_TopK87_h240_longrisk_UCB = 0.03389313880385762
best_LDO_drop_max = 0.3563218390804597
best_LSO_drop_max = 0.3563218390804597
best_max_group_share = 0.4367816091954023
group_invariant_pairwise_ranker_v3_pass = 0
```

P7 artifact：

```text
p7_rank_safe_certificate_v9.csv
```

P7 summary：

```text
best_certificate_id = RC7-TopKFixedCountGroupBalanced
best_ranker_id = GIR9-PairwiseWithinGroupRanker
best_heldout_accepted_count = 87
best_coverage_heldout = 0.10046189376443418
best_GradeAB_precision_heldout = 0.28735632183908044
best_V_integrated_LCB_heldout = 0.03544770490685384
best_h240_longrisk_UCB_heldout = 0.369780988566077
rank_safe_certificate_v9_pass = 0
```

判断：P3-P7 复现了 v9.5.9/v9.6.0 的 rank 结构：TopK signal 存在，但 group/heldout/certificate accepted-region 仍不稳，不能打开 existing-action controller。

## 7. P8-P10 controller boundary 与 OOD autopsy

P8：

```text
p8_existing_action_minimal_controller.csv = not_run
reason = P5_P6_or_P7_not_passed
existing_action_controller_pass = 0
source_controller_pass = 0
```

P9：

```text
p9_selected_runtime_preflight.csv = not_run
reason = P8_controller_not_selected
selected_runtime_pass = 0
```

P10 artifact：

```text
p10_apga_apgd_ood_causal_autopsy_v2.csv
```

P10 summary：

```text
damage_row_count = 1024
assigned_damage_fraction = 1.0
dominant_damage_mode = memory_offdiag_longrisk_created
dominant_damage_mode_fraction = 0.7294921875
longrisk_created_count = 747
longrisk_created_explained_fraction = 1.0
source_to_generated_damage_V_LCB = -0.6120673578682826
P10_ood_causal_autopsy_pass = 1
```

判断：controller/runtime 没有打开。P10 说明 APGA/APGD 的主要破坏仍是 memory/offdiag longrisk created，但这只是 generator diagnostic。

## 8. P11-P13 APGE primitive

P11 artifact：

```text
p11_apge_memory_offdiag_safe_primitive.csv
```

P11 summary：

```text
primitive_count = 8
generated_action_count = 512
generated_action_count_expected = 512
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
negative_control_present = 1
apge_implementation_pass = 1
```

P12 artifacts：

```text
p12_apge_branch_horizon_outcome.csv
```

P12 summary：

```text
expected_rows = 12288
actual_rows = 12288
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1
rows_per_sec = 15.676012494185711
wallclock_sec = 783.8728123339824
unresolved_exception_count = 0
duplicate_row_count = 0
label_exclusivity_violation = 0
quality_audit_pass = 1
```

P13 artifact：

```text
p13_apge_outcome_geometry_pass.csv
```

P13 summary：

```text
best_primitive_id = APGE6-SignalChannelSNRPreconditionedDelta
best_GradeAB_precision = 0.03125
best_V_integrated_LCB = -0.8693640105318599
best_h240_longrisk_UCB = 0.8825326673766913
best_longrisk_created_rate = 0.78125
best_source_to_generated_damage_LCB = -1.0920578806213115
apge_weak_pass = 0
apge_strong_pass = 0
apge_official_candidate_pass = 0
```

Per primitive：

| primitive | GradeAB precision | V LCB | h240 longrisk UCB | longrisk created |
|---|---:|---:|---:|---:|
| `APGE1-MemoryNullspaceProjectedResidual` | `0.015625` | `-0.8372801038046461` | `0.8150608413036654` | `0.703125` |
| `APGE2-OffdiagSafePopulationRiskGate` | `0.0` | `-0.9026237786033957` | `0.8954445728577505` | `0.796875` |
| `APGE3-CoverEntropyPreservingEdgeUpdate` | `0.0` | `-1.0629333093622668` | `0.8010605393336523` | `0.6875` |
| `APGE4-OldFamilyOrthogonalizedFunctionalDelta` | `0.0` | `-0.9265324670166863` | `0.7726149255507754` | `0.65625` |
| `APGE5-SymmetricBoundarySmallStepUpdate` | `0.015625` | `-0.8433142358591618` | `0.8694088506054019` | `0.765625` |
| `APGE6-SignalChannelSNRPreconditionedDelta` | `0.03125` | `-0.8693640105318599` | `0.8825326673766913` | `0.78125` |
| `APGE7-ValueRankSeededMemorySafeBlend` | `0.0` | `-1.0151123112974814` | `0.9081265318504754` | `0.8125` |
| `APGE8-NegativeControlShuffledMemoryProjection` | `0.0` | `-0.929366677326503` | `0.9081265318504754` | `0.8125` |

判断：APGE implementation/materializer 层闭合，但 generated outcome 全面失败。不能把完整 12288 行 replay 写成 generated frontier pass。

## 9. P14-P15 boundary 与 Base-Acc Sentinel

P14：

```text
p14_system_leaveout_boundary.csv = not_run
reason = controller_runtime_or_apge_official_not_open
official_eligible = 0
system_legal_controller_pass = 0
paired_replay_opened = 0
```

P15：

```text
p15_paired_replay_short_full_boundary.csv = not_run
reason = P14_system_not_official
paired_replay_pass = 0
short_full_boundary_pass = 0
```

Base-Acc Sentinel：

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

## 10. No-fake audit

```text
rows_checked = 28614
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
v9600_boundary_pass = 1
group_pocket_intervention_pass = 1
matched_pair_causal_pass = 0
group_balanced_target_density_pass = 0
feature_deconfounding_pass = 0
value_rank_veto_pass = 0
pairwise_ranker_pass = 0
rank_safe_certificate_pass = 0
existing_controller/selected_runtime/system = 0/0/0
ood_autopsy_pass = 1
apge_implementation_pass = 1
apge_branch_horizon_pass = 1
apge_weak_pass = 0
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
route = R2-GroupPocketConfounded_StopPayloadPocketRoute
F0_boundary_reproduction_failed = 0
F1_payload_pocket_causal_unresolved = 0
F2_payload_pocket_confounded = 1
F3_existing_controller_fail = 1
F4_apge_generated_frontier_fail = 1
F5_system_not_official = 1
F6_base_acc_catastrophic = 0
primary_blocker = payload_pocket_confounded_or_not_causal
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `17e53e620a4bfc62fb0e28748b1306adf88f8c45c1c8de8e60948222f8948866` |
| runner | `82d10defed2a284c61d961380ad03cfe046dd361f930aa461b57cf8b252f472c` |
| run manifest | `a7f54a0f6716f57a6cfadb89fd5bf72da92125a7df1b8e5bcc31ef2a823f248a` |
| route | `8280d481081597c995f83f9ee4db1d18d0a3a51a36e73f3b66b016eeafe0c317` |
| P0 boundary | `94273982f7f51b6f3ef38d8221c00003bb0770658ee058e0bdefea84a92e1908` |
| P1 intervention panel | `d08e392f61e71ef75078de9a7c37b66eaed2538d11289cb20be3de5b4ebefb65` |
| P1 clone trace | `b008a24b6809da6b7e2329e8e13ed1fee419f42f14b3b8bf32532e24d340cb52` |
| P2 matched pair | `6736657ca04c64675272835166480d5affee548b496ebeaee23be70464a5e7d4` |
| P2 pair trace | `55b1f4bf125cc2942131e2f9b3c91fee80880d21a529bfcd45ff50486bdc91a8` |
| P3 target density | `819098545294bca04b06a0ad7e65f373c892110adca08702eb74a823f874d1be` |
| P4 feature deconfounding | `a45c575c8c3748eb398c4d6c88e27596c6b53e3d02d4c7c8ff41f4bbab0c16bd` |
| P5 value rank veto | `fab2449220deb7582b47923d6773194d9cebfc88c57f17f14a3f2939bf49df3d` |
| P6 pairwise ranker | `ec7375a24ad5346865175128d4a868fa5a2f21ea59c8a28f3b1dd31d2f8bed95` |
| P7 certificate | `9f62d982dea55e0ff55fe5e5432e0087628109f4a6bd3371d9635dfcebdcd333` |
| P8 controller | `6754b31ca0450802494a458e6d03fb70411468a0071b605c099f05577ab37c7e` |
| P9 runtime | `6196d62dfc6e3d441ade211ef803590b31c555f129048c67ea2b190abf8bd584` |
| P10 OOD autopsy | `2434b67d65d53eb71dbe1d02a03ab9988be9803b83f31f74f929f534e6934076` |
| P11 APGE implementation | `c64c1eda2ef55ec0b1f0179cea223232e3b734f3b4ac641d3e542f9e187cac1b` |
| P12 APGE smoke | `6f511df8845bbe821c1f3962138e42a71e0d81a4aa0538816d0eea75e1a6ffe4` |
| P13 APGE outcome | `c625d442219e7ce505893d63533020eda9fd6d224b0b1a331392e8d8051ec798` |
| P14 system/leaveout | `edffeca76b0f38eb17955aaabfabec402f67400ba189477a18be0ce1c3593300` |
| P15 paired/short-full | `7b3c2ed4576484b926ad846034e31a2020d11a46e87219609a273633b9a82430` |
| Base-Acc Sentinel | `dc90790ad7bb853413900368aada11d9491d7bf7103696330ac2fa2c9b170557` |
| no-fake audit | `3721fe31422a79ccd8094e4dbee3dee852e0e69b273aeac740308383a5363a51` |
| contract audit | `150dad64375ba549ecd8d74e23e62a1e5652cc016faa6e3ce84cf0a4ad466f62` |
| failure taxonomy | `b5f78c333d59a2433192827d7fd8fa7ff282add2f05db91f800222546eec3e6b` |

## 12. 最终分析结论

v9.6.1 的真实推进是：

```text
v9.6.0:
  group pocket 仍未解释；
  rank signal 强但 group-stable/controller/certificate 未闭合；
  APGD generated frontier 失败。

v9.6.1:
  对 payload0 group pocket 做 intervention clone replay 与 matched-pair attribution；
  真实生成 520 个 intervention clones 和 12480 行 branch-horizon trace；
  结果不支持 payload0 是可利用 causal mechanism，反而支持 confounded pocket；
  group-balanced target/rank/certificate 仍未打开 existing-action controller；
  APGA/APGD OOD autopsy 指向 memory/offdiag longrisk created；
  APGE1-APGE8 工程链路和 12288 行 replay 完整闭合；
  但 APGE generated actions 仍 value-negative / high-longrisk。
```

机制判断：

1. H0 成立：v9.6.0 boundary 被复现，没有跳过 group-pocket unresolved / APGD failure。
2. H1 未成立：payload0 intervention 没有产生 causal positive lift；normalized/projection clones 的 GradeAB precision 极低，longrisk 高。
3. H2 成立于反证层：matched-pair attribution 支持 `payload0_confounded_supported = 1`，primary blocker 转为 `payload_pocket_confounded_or_not_causal`。
4. H3 未成立：group-balanced target 仍只有 weak diagnostic，official density 未闭合。
5. H4 未成立：ControlTransferImprovement 的 within-group AUC 很高，但 group-balanced TopK longrisk 太高。
6. H5/H6 未成立：VR4/GIR9 rank signal 仍强，但 leaveout/group/certificate gate 不过。
7. H7 未打开：existing-action controller 和 selected runtime 均 not_run。
8. H8 成立于诊断层：APGA/APGD OOD damage 主要是 `memory_offdiag_longrisk_created`。
9. H9/H10 成立于 implementation 层：APGE payload/certificate/action apply 与 branch-horizon materializer 完整闭合。
10. H11 未成立：APGE 没有生成 value-positive / horizon-safe frontier；best APGE6 的 V LCB 为负且 longrisk UCB 很高。
11. H12-H14 未打开：没有 controller/runtime/APGE official pass，就不能打开 leaveout、paired replay 或 short/full training。
12. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.6.1 真实执行后停在 `R2-GroupPocketConfounded_StopPayloadPocketRoute`：payload0 pocket 经 intervention clone replay 与 matched-pair attribution 后不再像可利用 causal mechanism，而更像 confounded rank pocket；APGE 虽完整落盘，但仍没有生成 value-positive / horizon-safe frontier，因此 strict PureKAN functional 仍未成功。
