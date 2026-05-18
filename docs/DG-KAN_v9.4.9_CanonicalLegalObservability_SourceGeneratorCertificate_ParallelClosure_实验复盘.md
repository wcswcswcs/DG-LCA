# DG-KAN v9.4.9 Canonical Legal Observability / Source Generator / Certificate Parallel Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.4.9_CanonicalLegalObservability_SourceGeneratorCertificate_ParallelClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 no-transform diagnostic、oracle source frontier、Base-Acc Sentinel 或未打开的 controller/runtime/downstream 写成 official system pass。

## 0. 最新结论

```text
route = R1-FrontierExistsButLegalOpaque
base_candidate = LQ-t2-h256
success_v9490_strict_purekan_functional = False
success_v9490_full_functional = False
success_v9490_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9490_canonical_legal_observability_source_generator_certificate_first_20260514T170000Z/
```

核心结论：

1. P0 复现 v9.4.8 recovery boundary：canonical full control outcome ready = `1`，rows = `51768 / 51768`，quality audit pass = `1`，old table quarantine enforced = `1`。
2. P1 YRobust definition audit 过：action count = `2876`，YRobust original count = `120`，manual recompute match rate = `1.0`，mismatch count = `0`。
3. P1 澄清 v9.4.8 的 YRobust 不要求 h80 weak：YRobust h20 weak rate = `1.0`，h80 weak rate = `0.175`，h240 longrisk rate = `0.0`。
4. P2 robust source anatomy 未找到足够强 legal separation：max AUC = `0.5375105829704886`，max effect size = `0.10892368693734876`，dominant miss reason = `M8-outcome-only-pattern`。
5. P3 legal action-effect feature factory v2 已真实 materialize：feature count = `41`，feature value rows = `117916`，action count = `2876`。
6. P3 best legal feature = `NegHardTailFraction`，AUC_YRobust = `0.5375105829704886`，TopK64 YRobust precision = `0.03125`，TopK64 h240 longrisk = `0.546875`，legal_feature_pass = `0`。
7. P4 existing generator revalidation 已运行：generated action = `192`，branch-horizon rows = `3456`，payload/certificate hash missing = `0`，action apply L∞ max = `0.0`。
8. P4 no-transform sanity 过：G0 no-transform metric abs diff max = `0.0`，label match rate = `1.0`，horizon state hash match rate = `1.0`；但它只是 diagnostic，不算 real transform generator pass。
9. P4 real transform generator 未过：transform_generator_preserve_pass = `0`，transform_generator_improve_pass = `0`，official existing_generator_preserve_pass = `0`。
10. P5 direct robust generator v2 未过：generated action = `96`，rows = `1728`，best = `VG3-TailMarginConservativeSource`，h20 weak CP = `0.375`，h20 V_ctrl LCB = `-0.16784700022835225`，h240 longrisk = `0.625`，YRobust precision = `0.125`。
11. P6 effect-valid certificate 未过：best = `CERT13-MinimalMonotoneCompositeCertificate`，AUC_YRobust = `0.536384252539913`，TopK64 YRobust precision = `0.03125`，TopK64 longrisk = `0.59375`，ECE = `0.6683514548369317`。
12. P8 runtime preflight diagnostic 过，但 controller 未选中：runtime_preflight_pass = `1`，official_runtime_pass = `0`。
13. P11 Base-Acc Sentinel 复用 v9.4.7/v9.4.8：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
14. P7/P9/P10/P13/P14 均 gate-blocked：没有 source controller、selected runtime、official paired replay 或 short/full validation。
15. 当前 primary blocker：`canonical_frontier_legal_generator_certificate_failed`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9490_canonical_legal_observability_source_generator_certificate.py` | v9.4.9 runner；读取 v9.4.8 canonical full table，执行 YRobust definition audit、legal action-effect feature factory v2、existing/direct generator revalidation、effect-valid certificate、runtime/system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9490_canonical_legal_observability_source_generator_certificate.py
```

正式运行：

```bash
python experiments/run_v9490_canonical_legal_observability_source_generator_certificate.py \
  --out-dir results/real_rerun_20260506/v9490_canonical_legal_observability_source_generator_certificate_first_20260514T170000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --feature-action-limit 2876 --generator-actions 16 --direct-actions-per-generator 16
```

运行结果：

```json
{
  "best_legal_auc": 0.5375105829704886,
  "best_legal_feature": "NegHardTailFraction",
  "certificate_pass": 0,
  "generator_pass": 0,
  "out_dir": "results/real_rerun_20260506/v9490_canonical_legal_observability_source_generator_certificate_first_20260514T170000Z",
  "route": "R1-FrontierExistsButLegalOpaque"
}
```

## 2. Route

`route_decision_v9490.json` 摘要：

```json
{
  "route": "R1-FrontierExistsButLegalOpaque",
  "source_route_v9480": "R4-CanonicalSourceFrontierExistsLegalOpaque",
  "canonical_full_control_outcome_ready": 1,
  "canonical_row_count_actual": 51768,
  "quality_audit_pass": 1,
  "YRobust_definition_consistency_pass": 1,
  "YRobust_original_count": 120,
  "YRobust_manual_recompute_match_rate": 1.0,
  "YRobust_h80_weak_rate": 0.175,
  "anatomy_pass": 0,
  "max_AUC_anatomy": 0.5375105829704886,
  "dominant_miss_reason": "M8-outcome-only-pattern",
  "legal_feature_pass": 0,
  "best_legal_feature_id": "NegHardTailFraction",
  "best_legal_AUC_YRobust": 0.5375105829704886,
  "best_legal_TopK64_YRobust_precision": 0.03125,
  "existing_no_transform_sanity_pass": 1,
  "existing_no_transform_preserve_pass": 1,
  "existing_generator_preserve_pass": 0,
  "direct_generator_pass": 0,
  "certificate_effect_valid_pass": 0,
  "runtime_preflight_pass": 1,
  "source_controller_pass": 0,
  "selected_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "canonical_frontier_legal_generator_certificate_failed"
}
```

判断：v9.4.9 没有回退 v9.4.8 的 canonical frontier 结论；本轮推进到了 legal observability、generator、certificate 的并行实测，但三者均未达到 official gate。

## 3. P0/P1 boundary 与 YRobust audit

Artifacts：

```text
p0_v9480_recovery_boundary_reproduction.csv
p1_yrobust_definition_consistency_audit.csv
```

P0 summary：

```text
route_v9480 = R4-CanonicalSourceFrontierExistsLegalOpaque
canonical_full_control_outcome_ready = 1
canonical_row_count_expected/actual = 51768 / 51768
quality_audit_pass = 1
old_table_quarantine_enforced = 1
canonical_source_frontier_pass = 1
legal_observability_pass = 0
system_legal_controller_pass = 0
p0_pass = 1
```

P1 summary：

```text
action_count = 2876
YRobust_original_count = 120
YRobust_v1_count = 16
YRobust_v2_count = 120
YRobust_v3_count = 120
YRobust_manual_recompute_match_rate = 1.0
YRobust_mismatch_count = 0
h80_required_by_original_definition = 0
YRobust_h80_weak_rate = 0.175
YRobust_definition_consistency_pass = 1
```

判断：YRobust label 本身可重算复现；v9.4.8 的 robust source definition 是 h20 weak + h20 value positive + h240 longrisk safe，不要求 h80 weak。

## 4. P2/P3 legal observability

Artifacts：

```text
p2_canonical_robust_source_anatomy.csv
p3_legal_action_effect_feature_factory_v2.csv
```

P2 summary：

```text
action_count = 2876
YRobust_action_count = 120
dimension_count = 41
max_abs_effect_size = 0.10892368693734876
max_AUC_YRobust = 0.5375105829704886
best_legal_anatomy_dimension = NegHardTailFraction
at_least_one_legal_feature_distribution_shift_pass = 0
dominant_miss_reason = M8-outcome-only-pattern
dominant_miss_reason_count = 118
oracle_outcome_only_pattern_likely = 1
p2_pass = 0
```

P3 summary：

```text
feature_count = 41
feature_value_row_count = 117916
action_count = 2876
best_feature_id = NegHardTailFraction
best_feature_group = G4-horizon-risk-proxy
best_AUC_YRobust = 0.5375105829704886
best_AUC_LongRisk = 0.41508897166902475
best_TopK16_YRobust_precision = 0.0
best_TopK64_YRobust_precision = 0.03125
best_TopK64_h240_longrisk = 0.546875
legal_feature_pass = 0
legal_feature_strong_pass = 0
```

判断：v2 legal/action-effect features 已经真实生成，但对 canonical YRobust 的可观测性仍接近随机；不能从 oracle frontier 直接转 official selector。

## 5. P4 existing generator revalidation

Artifacts：

```text
p4_canonical_existing_generator_preservation.csv
canonical_existing_generator_trace_v9490.csv
```

Summary：

```text
generator_panel_count = 12
generated_action_count_total = 192
branch_horizon_rows_actual = 3456
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
best_source_panel_id = S0-SRC-ORC-YRobust-K16-diagnostic
best_generator_id = G0-NoTransformCanonicalReplay
best_h20_weak_CP = 1.0
best_h20_V_ctrl_lcb = 0.8638057411703248
best_h240_longrisk = 0.0
best_YRobust_precision = 1.0
no_transform_sanity_pass = 1
no_transform_preserve_pass = 1
transform_generator_preserve_pass = 0
transform_generator_improve_pass = 0
generator_weak_pass = 0
```

Transform examples from oracle panel:

```text
AP0l h20 weak CP = 0.1875, h20 V_ctrl LCB = -1.6240692108021402, h240 longrisk = 0.75
AP0r h20 weak CP = 0.1875, h20 V_ctrl LCB = -1.9379268255199147, h240 longrisk = 0.75
AP0u h20 weak CP = 0.25,   h20 V_ctrl LCB = -1.523734753969325,  h240 longrisk = 0.9375
AP0v h20 weak CP = 0.1875, h20 V_ctrl LCB = -1.4569257776695466, h240 longrisk = 0.625
```

判断：no-transform replay 继续闭合，但所有真实 transform generator 均未保留 canonical robust source value；no-transform 不能作为 generator official success。

## 6. P5 direct robust generator v2

Artifacts：

```text
p5_direct_robust_source_generator_v2.csv
direct_robust_source_generator_trace_v9490.csv
```

Summary：

```text
generator_panel_count = 6
generated_action_count_total = 96
branch_horizon_rows_actual = 1728
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
best_source_panel_id = S2-TopLegalFeaturePanel
best_generator_id = VG3-TailMarginConservativeSource
best_h20_weak_CP = 0.375
best_h20_V_ctrl_lcb = -0.16784700022835225
best_h240_longrisk = 0.625
best_YRobust_precision = 0.125
best_source_positive_lost_rate = 0.8
best_Damage_median = 0.009064337587915361
direct_generator_pass = 0
```

判断：direct generator implementation 链路闭合，但 best candidate 的 value LCB 仍为负、h240 long-risk 过高，不能打开 controller。

## 7. P6 certificate v2

Artifact：

```text
p6_effect_valid_certificate_v2.csv
```

Summary：

```text
certificate_count = 5
best_certificate_id = CERT13-MinimalMonotoneCompositeCertificate
best_AUC_YRobust = 0.536384252539913
best_AUC_LongRisk = 0.41916336377998953
best_TopK16_YRobust_precision = 0.0
best_TopK64_YRobust_precision = 0.03125
best_TopK64_longrisk = 0.59375
best_P_YRobust_given_cert_pass = 0.03125
best_P_LongRisk_given_cert_pass = 0.59375
best_ECE_YRobust = 0.6683514548369317
certificate_effect_valid_pass = 0
certificate_effect_strong_pass = 0
```

判断：CERT9-CERT13 没有形成 YRobust / long-risk 的有效分离；certificate threshold patch 不能转成 source controller。

## 8. P7-P14 boundary

P7：

```text
p7_minimal_source_certificate_controller.csv = not_run
reason = upstream_legal_feature_or_certificate_failed
source_controller_pass = 0
```

P8：

```text
runtime_preflight_pass = 1
official_runtime_pass = 0
controller_id = not_selected
```

P9-P14：

| artifact | status / reason |
|---|---|
| `p9_leaveout_boundary.csv` | `not_run`, `P7_controller_not_selected` |
| `p10_official_paired_replay_boundary.csv` | `not_run`, `P7_controller_not_selected` |
| `p11_short_full_training_boundary.csv` | Base-Acc Sentinel complete, functional short run `not_run` |
| `p12_system_integration_gate_v9490.csv` | official eligible = `0` |
| `p13_ldo_lso_paired_replay_boundary.csv` | `not_run`, `P12_system_controller_not_official` |
| `p14_short_full_validation_boundary.csv` | `not_run`, `P13_paired_replay_not_open` |

没有把 runtime preflight、no-transform sanity、Base-Acc Sentinel 或 oracle frontier 写成 official system pass。

## 9. Base-Acc Sentinel

Artifact：

```text
p11_short_full_training_boundary.csv
base_acc_training_trace_v9490.csv
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
LQ_minus_AdamWStrongLRGridMLP_mean_test_acc = 0.025260416666666674
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
base_acc_reused_from_v9470 = 1
base_acc_reused_from_v9480 = 1
```

判断：Base-Acc Sentinel 继续健康，但仍是 isolated diagnostic，没有用于 selector/generator/certificate/controller。

## 10. No-fake audit

```text
rows_checked = 129220
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
canonical_full_control_outcome_ready = 1
YRobust_definition_consistency_pass = 1
legal_feature_pass = 0
existing_generator_measured = 1
existing_generator_preserve_pass = 0
direct_generator_measured = 1
direct_generator_pass = 0
certificate_effect_valid_pass = 0
runtime_preflight_pass = 1
source_controller/selected_runtime/system = 0/0/0
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
route = R1-FrontierExistsButLegalOpaque
F0_truth_base_failure = 0
F1_frontier_legal_opaque = 1
F2_legal_observability_pass_controller_pending = 0
F3_generator_pass_certificate_pending = 0
F4_runtime_pending = 0
F5_system_not_official = 1
F6_base_acc_catastrophic = 0
primary_blocker = canonical_frontier_legal_generator_certificate_failed
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `dedb16c3167cab59ece3b3ac5783d6cf988c9fb556f9a2ecc8c81db99fc46abd` |
| runner | `6289a69e542599f3761125b2dbb7e8549660a0170649c061e208aec51eb02881` |
| run manifest | `c1aa5a0346487dd2d4f50706fc692d251ef531df8f7b14debb297aa53b919997` |
| route | `a421d723acfee5d014d44d29e0e9ef6f3227fbc2fb43200134352ff362839dfe` |
| P0 boundary | `3201427c5b2819ca1d26cacc4a208aa7cf2f3d9e9927c7784496615dca17fc5d` |
| P1 YRobust | `40fde8c96cbd6b13b0eb61ce3281e0aa89e8126c5006840edf55419e0100345a` |
| P2 anatomy | `279a6a99baef1f28c93588ab0866cdee9928a0077c51660975c20355a81a5d3c` |
| P3 legal feature | `2f2a9bfc05b0aaa33816a6d02f5ea4ce2ccc8faca2d7a3684690750232af004c` |
| P4 existing generator | `e4927ba5063b3480480f207ce87c4373f5c460697cdc6bd37707e0bd6dfda21f` |
| P5 direct generator | `f194c2db4858e70f9565549f97146c88870e4bf4c872142d8f0202e3629e5b85` |
| P6 certificate | `6ec9c777943aa13c1593a74c70a4a19ef0d37067dca9182a8ad2b9af80cadd0a` |
| P7 controller | `444c354cb8f5c155db5f6da97fd4c1944eaaa33dd434ec8163247ba340842278` |
| P8 runtime | `c9673f4bf86dccc02a5335573b0b545de2fe45aa99f49661fd7cd43fbc7eab17` |
| P9 leaveout | `443003d6a5ef89ec8e0a38770c616fbe2d5ce3ccf8a403707f87d506b37d5abe` |
| P10 paired replay | `5f3f913b27de7e39f2f2e319c89ebe54480a0c52e3fc514acd2984121fc80089` |
| P11 short/full | `62b655382fd1149b094a5b1ebd656d6a36de7ca0480d729e7bac2db86cd5310d` |
| no-fake audit | `a6708edf37600a83b9655722f8a8bab068f5f1dcfc70325d796c6ef0eb726fd0` |
| contract audit | `3aeae2b0d31afdeb7267a5e2cdd5358cb1b13d6870725f0a660736f6d5fa58b7` |
| failure table | `f60985cfe8a6ada13c9ac408d038bc35fbdccf19b9f2ad75b494e9fcd73cd687` |

## 12. 最终分析结论

v9.4.9 的真实推进是：

```text
v9.4.8:
  canonical full AP0 outcome universe 已闭合；
  canonical CP/source frontier 存在；
  但 legal observability 看不见 robust source。

v9.4.9:
  明确 YRobust definition 并重算一致；
  扩展 legal action-effect features 到 41 个 commit-time features；
  并行重测 existing generator、direct generator 和 effect-valid certificates；
  但 legal features、真实 generator transform、direct generator、certificate 均未打开 official controller。
```

机制判断：

1. H0 成立：v9.4.8 recovery boundary 被复现，canonical truth base ready。
2. H1 成立：YRobust label 定义闭合；h80 weak 不属于 original gate。
3. H2 未成立：robust source anatomy 没有可用 legal separation，dominant miss reason 是 `M8-outcome-only-pattern`。
4. H3 未成立：legal action-effect feature factory v2 未过，best AUC 只有 `0.5375`。
5. H4 未成立：G0 no-transform sanity 过，但 AP0l/AP0r/AP0t/AP0u/AP0v 等真实 transform generator 仍破坏 robust value。
6. H5 未成立：direct generator v2 没产生可用 robust source frontier。
7. H6 未成立：CERT9-CERT13 对 YRobust / long-risk 没有有效分离。
8. H7-H8 未打开：controller/runtime/system/downstream 全部 gate-blocked；runtime preflight 只能说明 timed path 可测，不能 official。

最终一句话：

> v9.4.9 真实执行后停在 `R1-FrontierExistsButLegalOpaque`：canonical AP0 robust source frontier 确认存在，但当前 commit-time legal features 仍看不见它，真实 generator 和 certificate 也没有把它转成可用 controller；strict PureKAN functional 仍未成功。
