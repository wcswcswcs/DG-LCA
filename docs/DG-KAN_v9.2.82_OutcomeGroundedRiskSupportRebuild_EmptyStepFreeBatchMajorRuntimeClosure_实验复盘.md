# DG-KAN v9.2.82 Outcome-Grounded RiskSupport Rebuild 与 Empty-Step-Free BatchMajor Runtime Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.82_OutcomeGroundedRiskSupportRebuild_EmptyStepFreeBatchMajorRuntimeClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 risk/support diagnostic、active-step compaction estimate 或旧 full-system runtime 写成 official system pass。

## 0. 最新结论

```text
route = R16-StableAcceptDecisionRegionUnsafe
base_candidate = LQ-t2-h256
success_v9282_strict_purekan_functional = False
success_v9282_full_functional = False
success_v9282_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9282_outcome_grounded_risk_support_rebuild_empty_step_free_batch_major_runtime_closure_first_20260513T220000Z/
```

核心结论：

1. P0 复现 v9.2.81 boundary：source route = `R16-OutcomeGroundedStableAcceptRegionUnsafe`，P8 system controller pass = `0`。
2. P1 scoreq/rank contract 仍可由 QR5 fallback 闭合：score quantized disagreement = `0`，rank disagreement = `0`，accept disagreement = `0`。
3. P1 不是 native integer emit stronger closure：`native_scoreq_emit_bitexact = 0`，fallback row count = `20`，因此不能写成 native quantization 已彻底修复。
4. P2 candidate lifecycle 仍未闭合：old candidate count = `2493`，new candidate count = `2876`，Jaccard = `0.654036`，shared candidate 中 `candidate_id_mismatch_count = 2120`、`payload_hash_mismatch_count = 468`。
5. P3 primary outcome label 仍可靠：missing primary label = `0`，ambiguous label = `0`；secondary outcome delta fields 仍缺 `14380`，downstream 不 ready。
6. P4 bad accepted autopsy 过 gate：bad accepted count = `142`，attribution fraction = `1.0`，primary submode = `FMA3a-risk_mean_underestimated`。
7. P5 risk/support feature 有一个刚过计划 AUC 门槛：best bad-risk feature = `RSF6-RiskResidualScore`，bad-event AUC = `0.750961`；best safe-good AUC = `0.693120`。
8. P6 dataset-agnostic decision repair 失败：best selected candidate = `DR7-_risk_residual_score-thr-0.715503-n1.0-s0.05`，calibration precision = `0.763043`，但 calibration bad-event = `0.106522`，heldout accepted count = `0`，heldout coverage = `0.0`，decision gate pass = `0`。
9. v9.2.81 的 DR2 reference 仍不过：heldout precision = `0.630037`，bad-event = `0.183150`，null-rate = `0.194139`。
10. P7/P8 没有 materialize empty-step-free measured runtime：RT1/RT7 只是 active-step compaction diagnostic，`diagnostic_derived_from_measured_components = 1`，runtime pass = `0`。
11. P9 official controller 未打开：`official_eligible = 0`，`system_legal_controller_pass = 0`，reason = `risk_support_feature_unpredictive_and_runtime_not_materialized`。
12. 当前 primary blocker：`no_dataset_agnostic_decision_repair`；下一步需要重新设计 dataset-agnostic risk/support score，而不是继续把诊断估计转正。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9282_outcome_grounded_risk_support_rebuild_empty_step_free_batch_major_runtime_closure.py` | v9.2.82 runner；读取 v9.2.81/v9.2.80 artifacts，执行 scoreq/rank contract audit、candidate lifecycle audit、secondary outcome audit、risk/support feature factory、dataset-agnostic decision repair、empty-step runtime diagnostic 与 P9 gate |

代码检查：

```text
python -m py_compile experiments/run_v9282_outcome_grounded_risk_support_rebuild_empty_step_free_batch_major_runtime_closure.py
```

正式运行：

```bash
python experiments/run_v9282_outcome_grounded_risk_support_rebuild_empty_step_free_batch_major_runtime_closure.py \
  --out-dir results/real_rerun_20260506/v9282_outcome_grounded_risk_support_rebuild_empty_step_free_batch_major_runtime_closure_first_20260513T220000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际输入来自 `run_manifest.json`：

```text
source_v9281 = results/real_rerun_20260506/v9281_outcome_grounded_stable_accept_repair_batch_major_runtime_closure_first_20260513T210000Z
source_v9280 = results/real_rerun_20260506/v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_rerun_20260513T190000Z
source_v9272 = results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_async_basis_cuda_ext_20260513T110000Z
event_count = 24192
candidate_count = 2876
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R16-StableAcceptDecisionRegionUnsafe",
  "base_candidate": "LQ-t2-h256",
  "v9281_boundary_pass": 1,
  "reference_controller_id": "C3Q2-v13-OutcomeGroundedStableAcceptRepair",
  "stable_controller_id": "C3Q2-v14-OutcomeGroundedRiskSupportRebuild",
  "scoreq_rank_contract_pass": 1,
  "score_quantized_disagreement_count": 0,
  "rank_disagreement_count": 0,
  "accept_disagreement_count": 0,
  "native_scoreq_emit_bitexact": 0,
  "fallback_row_count": 20,
  "candidate_lifecycle_pass": 0,
  "candidate_jaccard": 0.6540357362908195,
  "candidate_lifecycle_primary_effect": "candidate_schema_and_payload_hash_drift_remains",
  "outcome_primary_pass": 1,
  "outcome_secondary_delta_pass": 0,
  "missing_secondary_delta_count": 14380,
  "bad_accepted_autopsy_pass": 1,
  "primary_bad_accepted_failure_mode": "FMA3a-risk_mean_underestimated",
  "risk_support_feature_pass": 1,
  "best_bad_risk_feature": "RSF6-RiskResidualScore",
  "best_bad_risk_auc": 0.7509612892076385,
  "best_safe_good_feature": "RSF6-RiskResidualScore",
  "best_safe_good_auc": 0.6931195175438596,
  "decision_repair_candidate_id": "DR7-_risk_residual_score-thr-0.715503-n1.0-s0.05",
  "decision_repair_pass": 0,
  "precision_cal": 0.7630434782608696,
  "coverage_cal": 0.03042328042328042,
  "bad_event_cal": 0.10652173913043478,
  "null_rate_cal": 0.13260869565217392,
  "precision_heldout": 0.0,
  "coverage_heldout": 0.0,
  "bad_event_heldout": 0.0,
  "null_rate_heldout": 0.0,
  "empty_step_runtime_pass": 0,
  "batch_major_runtime_pass": 0,
  "kernel_count_before": 72576,
  "kernel_count_after": 72576,
  "sync_count_before": 8064,
  "sync_count_after": 8064,
  "full_system_step_ratio_q90": 2.213009156635521,
  "new_full_system_step_ratio_measured": 0,
  "old_step_ratio_reused_as_measurement": 0,
  "official_eligible": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "no_dataset_agnostic_decision_repair",
  "next_required_implementation": "redesign_dataset_agnostic_risk_support_score"
}
```

判断：v9.2.82 有真实推进的是 risk/support feature factory 和 calibration-frozen decision repair 搜索，但没有找到能过 official gate 的 dataset-agnostic accepted region；runtime 也没有 measured batch-major materialization。

## 3. P1 scoreq/rank contract closure

Artifacts：

```text
p1_native_scoreq_rank_contract_closure.csv
scoreq_rank_contract_trace_v9282.csv
```

Summary：

```text
quantized_repair_id = QR0-v9281-QR5-fallback-reference
native_scoreq_emit_bitexact = 0
fallback_used = 1
fallback_row_count = 20
score_quantized_disagreement_count_before/after = 18 / 0
rank_disagreement_count_before/after = 2 / 0
accept_disagreement_count_before/after = 0 / 0
scoreq_rank_contract_pass = 1
native_stronger_pass = 0
```

判断：contract mismatch 在 full-row audit 上关闭，但仍依赖 borderline exact fallback；不能把它扩大成 native integer emit repair。

## 4. P2 candidate lifecycle

Artifacts：

```text
p2_candidate_lifecycle_canonicalization_drift_audit.csv
candidate_lifecycle_trace_v9282.csv
```

Summary：

```text
old_candidate_count = 2493
new_candidate_count = 2876
shared_candidate_count = 2123
old_only_candidate_count = 370
new_only_candidate_count = 753
candidate_jaccard = 0.6540357362908195
candidate_id_mismatch_count = 2120
payload_hash_mismatch_count = 468
new_only_precision = 0.398406374501992
shared_precision = 0.544983513895431
new_only_bad_event_rate = 0.2788844621513944
shared_bad_event_rate = 0.2779086198775318
candidate_count_change_explained = 0
candidate_lifecycle_pass = 0
```

判断：candidate drift 仍是事实，且 candidate id/hash schema 没有闭合；但 shared rows 自身 bad-event 也高，所以 failure 不能只归咎于 new-only rows。

## 5. P3 outcome secondary delta

Artifacts：

```text
p3_secondary_outcome_delta_materialization.csv
secondary_outcome_delta_trace_v9282.csv
```

Summary：

```text
missing_primary_label_count = 0
ambiguous_label_count = 0
label_exclusivity_violation_count = 0
label_source = same_run_train_stream
primary_outcome_pass = 1
missing_secondary_delta_count = 14380
downstream_ready_pass = 0
```

判断：primary decision gate 可以真实评估；paired replay / robustness 仍不能打开，因为 secondary outcome delta 仍未 materialize。

## 6. P4 bad accepted risk/support autopsy

Artifacts：

```text
p4_bad_accepted_risk_support_autopsy_v2.csv
bad_accepted_autopsy_trace_v9282.csv
```

Summary：

```text
bad_accepted_count = 142
bad_accepted_attribution_fraction = 1.0
top_failure_modes_explain_fraction = 1.0
primary_failure_submode = FMA3a-risk_mean_underestimated
failure_submode_counts = {
  "FMA3a-risk_mean_underestimated": 70,
  "FMA3e-candidate_origin_shift": 41,
  "FMA5a-null_bad_label_overlap": 20,
  "FMA7b-horizon_tail_context_missing": 11
}
bad_accepted_autopsy_pass = 1
```

判断：P4 继续确认主问题是 outcome-grounded decision region 本身不安全，尤其是风险均值/支持估计不足，而不是 scoreq/rank mismatch。

## 7. P5 risk/support feature factory

Artifacts：

```text
p5_outcome_grounded_risk_support_sufficient_statistics.csv
risk_support_feature_trace_v9282.csv
```

Feature summary：

| feature | AUC_bad_event | AUC_safe_good |
|---|---:|---:|
| `RSF1-BadLevelToken` | `0.710706` | `0.614437` |
| `RSF2-NullLevelToken` | `0.674112` | `0.553755` |
| `RSF3-HorizonTailRisk` | `0.691221` | `0.596820` |
| `RSF4-BadUCBMean` | `0.705187` | `0.642398` |
| `RSF5-BadUCBMax` | `0.500000` | `0.500000` |
| `RSF6-RiskResidualScore` | `0.750961` | `0.693120` |
| `RSF7-MonotoneRiskScore` | `0.749406` | `0.690049` |
| `RSF8-SupportLCBMin` | `0.500000` | `0.500000` |
| `RSF9-CandidateOriginRisk` | `0.526268` | `0.562061` |

判断：`RSF6-RiskResidualScore` 的 bad-risk AUC = `0.750961` 刚过计划 `>=0.75` 的 feature gate，但 safe-good AUC 只有 `0.693120`；feature diagnostic 过线不等于 official decision gate 过线。

## 8. P6 dataset-agnostic decision repair

Artifacts：

```text
p6_dataset_agnostic_decision_repair.csv
decision_repair_trace_v9282.csv
```

Reference candidates：

```text
DR0-v9281-DR2-reference:
  precision_heldout = 0.63003663003663
  coverage_heldout = 0.03009259259259259
  bad_event_heldout = 0.18315018315018314
  null_rate_heldout = 0.19413919413919414
  precision_lcb = 0.5713310098976009
  bad_event_ucb = 0.23332190828443805
  decision_gate_pass = 0

DR1-BitExactStableAcceptOnly:
  precision_heldout = 0.5128205128205128
  coverage_heldout = 0.051587301587301584
  bad_event_heldout = 0.3034188034188034
  null_rate_heldout = 0.12179487179487179
  precision_lcb = 0.46761511726824007
  bad_event_ucb = 0.3465326908369387
  decision_gate_pass = 0
```

Best selected repair：

```text
repair_candidate_id = DR7-_risk_residual_score-thr-0.715503-n1.0-s0.05
repair_type = risk_support_monotone_score
feature_set = _risk_residual_score
thresholds = {"null_ucb_max": 1.0, "risk_feature": "_risk_residual_score", "risk_max": 0.715503, "support_lcb_min": 0.05}
calibration_split_id = seed_0_4
heldout_split_id = seed_5_7
dataset_name_used = 0
accepted_count_cal = 230
precision_cal = 0.7630434782608696
coverage_cal = 0.03042328042328042
bad_event_cal = 0.10652173913043478
null_rate_cal = 0.13260869565217392
accepted_count_heldout = 0
precision_heldout = 0.0
coverage_heldout = 0.0
bad_event_heldout = 0.0
null_rate_heldout = 0.0
precision_lcb = 0.0
bad_event_ucb = 1.0
decision_gate_pass = 0
```

判断：本轮没有找到 official repair。最佳 selected repair 在 calibration 上 precision 达到 `0.763`，但 bad-event 仍 `0.1065 > 0.05`；冻结到 heldout 后没有接受任何 row，coverage = `0.0`，因此不能 official。

## 9. P7/P8 empty-step-free runtime

Artifacts：

```text
p7_runtime_empty_step_elimination.csv
empty_step_runtime_trace_v9282.csv
p8_measured_batch_major_native_runtime.csv
batch_major_native_runtime_trace_v9282.csv
```

Measured current path：

```text
runtime_candidate_id = RT0-v9281-full-system-native-reference
step_count_before = 8064
active_step_count = 1412
zero_candidate_step_count = 6652
empty_step_kernel_fraction_before/after = 0.8249007936507936 / 0.8249007936507936
kernel_count_before/after = 72576 / 72576
sync_count_before/after = 8064 / 8064
avg_candidates_per_kernel_after = 0.03962742504409171
step_ratio_q90 = 2.213009156635521
empty_step_runtime_materialized = 1
diagnostic_derived_from_measured_components = 0
empty_step_runtime_pass = 0
```

Diagnostic estimate only:

```text
runtime_candidate_id = RT1-ActiveStepCompactionDiagnosticOnly
empty_step_kernel_fraction_after = 0.0
kernel_count_after = 12708.0
sync_count_after = 1412
avg_candidates_per_kernel_after = 0.22631413282971358
empty_step_runtime_materialized = 0
diagnostic_derived_from_measured_components = 1
empty_step_runtime_pass = 0
```

P8 boundary：

```text
RT7-HybridEmptyStepFreeBatchMajorNativeDiagnosticOnly:
  batch_major_runtime_measured = 0
  diagnostic_derived_from_measured_components = 1
  batch_major_runtime_pass = 0
```

判断：runtime autopsy 已经说明 empty-step fixed launch 是主要浪费，但本轮没有新的 measured batch-major kernel/runtime。不能把 active-step compaction estimate 写成 runtime pass。

## 10. P9 system controller boundary

Artifact：

```text
p9_system_legal_exact_signal_controller_v14.csv
system_controller_trace_v9282.csv
```

Boundary：

```text
controller_id = C3Q2-v14-OutcomeGroundedRiskSupportRebuild
decision_repair_candidate_id = DR7-_risk_residual_score-thr-0.715503-n1.0-s0.05
runtime_candidate_id = RT0-v9281-full-system-native-reference
accept_contract_id = AC2Q2+QR5-borderline-exact-fallback
precision_cal = 0.7630434782608696
coverage_cal = 0.03042328042328042
bad_event_cal = 0.10652173913043478
null_rate_cal = 0.13260869565217392
precision_heldout = 0.0
coverage_heldout = 0.0
bad_event_heldout = 0.0
null_rate_heldout = 0.0
precision_lcb = 0.0
bad_event_ucb = 1.0
step_ratio_q90 = 2.213009156635521
secondary_outcome_delta_fields_present = 0
materialized_system_path = 1
native_bucket_kernel_used = 1
diagnostic_derived_from_measured_components = 0
dataset_name_used = 0
official_eligible = 0
system_legal_controller_pass = 0
reason = risk_support_feature_unpredictive_and_runtime_not_materialized
```

判断：P9 不能打开。decision repair failed，secondary delta 不 ready，batch-major runtime 未 materialize；没有复用旧 success，也没有把 diagnostic runtime 写成 official。

## 11. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p10_leave_dataset_and_stratum_out.csv` | `P9_system_controller_not_official` |
| `p11_official_paired_replay.csv` | same |
| `p12_short_full_robustness_strong_baseline.csv` | same |

没有把 scoreq/rank fallback、risk-support AUC diagnostic、P4 autopsy 或 runtime compaction estimate 写成 LDO/paired replay/short/full success。

## 12. No-fake audit

```text
rows_checked = 2876
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

Contract audit：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
train_stream_probe = 1
scoreq_rank_contract_pass = 1
outcome_primary_pass = 1
outcome_secondary_delta_pass = 0
bad_accepted_autopsy_pass = 1
risk_support_feature_pass = 1
decision_repair_pass = 0
empty_step_runtime_pass = 0
batch_major_runtime_pass = 0
system_legal_controller_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection_used_for_official = 0
source_measured_gap_used = 0
formula_proxy_used = 0
fake_data/proxy_row/cpu_offload = 0/0/0
```

Failure table：

```text
F12_no_dataset_agnostic_decision_repair = 1
F20_batch_major_runtime_not_materialized = 1
F9_secondary_outcome_delta_missing = 1
missing_secondary_delta_count = 14380
```

## 13. Hash

| artifact | SHA256 |
|---|---|
| plan | `c3b0c1d6dd5239295a34826004d6fb30f40fa60b8c9c3652f6d56761d9ad71c8` |
| runner | `49efe69e41d56a0bab181b27ac0b260b4194537a6ae9e9b30466f95e9118631a` |
| run manifest | `875739bb662989764a7dc549eaaca7e5f564d464215dc1e3dbd4f7a0871d16ec` |
| route | `1a1bb5b85e791ba967ffd9f8b228873ba2f143ebfb8815ea9fd9d88395bff05f` |
| aggregate decision | `1a1bb5b85e791ba967ffd9f8b228873ba2f143ebfb8815ea9fd9d88395bff05f` |
| P0 boundary | `18be6eccac14a3155c98c8f66912cf4b69a2f7a2821f554be87c979999ac7c9b` |
| P1 scoreq rank | `8ba77272cb9ee477f222cae532859b192cf421f9a0ba5a4a58e017b80edbc979` |
| P2 lifecycle | `51e6c1266956b21d4168946567ac701c27843782559248f87c76046ab9ee0f28` |
| P3 secondary | `30f7cd6bca79c98efe589ae533c3a8fdd65c719d26265ff1b88f782372ab5fb6` |
| P4 autopsy | `ecc31d115fc423555724604934d0f91f5794d2f36a6484ddd695540b89166faf` |
| P5 features | `6c04dc9ff7a2018b45a7133cd013e64b96bd3ca0ae2cb6f77b454ef0998d410e` |
| P6 decision repair | `9bc78ab10c137a9a64fb739d1f4f9f009829fcc6c97d8a649aa9e0d0e2735bf9` |
| P7 empty step | `431eaa296022e7905fe4d03a9852ef264420888791986302deca2944c9604ff6` |
| P8 batch major | `a532b379c834a2c921e0fa293ed10c1af6fdae2840f29f3a65058503e787f3f9` |
| P9 system | `5b7b5a98932a3d4344fc021fcb6684ae9602517937462a5cf0729c0877145459` |
| contract | `1309c9da80452bfca43003d6ae6e4ced9cc666decd4c439c1c4b77fe68e760c6` |
| provenance | `99f7e2b0578bf645346463fb5710b62b295fcfb4a62d856a3511e12ad073d8e0` |
| failure table | `c52894915e8cc697dc5bc742cb4d6712fd2695c2c1dd59b3ab3c13d67d63c15e` |

## 14. 最终分析结论

v9.2.82 的真实推进是：

```text
v9.2.81:
  QR5 清掉 quantized/rank mismatch；
  primary labels 和 decision/runtime failure autopsy 闭合；
  但 StableAccept region 不安全，batch-major runtime 未 materialize。

v9.2.82:
  继续用 full-row labels 重建 outcome-grounded risk/support features；
  找到一个 bad-risk AUC 刚过 0.75 的 risk residual score；
  但 calibration-frozen repair 无法在 heldout 上维持 coverage/precision/bad-event gate；
  runtime 仍停留在 measured old full-system native reference path。
```

机制判断：

1. H1 局部成立：scoreq/rank contract audit 为 `0` disagreement，但 native stronger pass 仍为 `0`，QR5 fallback 不能写成 native emit 修复。
2. H2 未闭合：candidate lifecycle 仍有明显 schema/hash drift，Jaccard = `0.654036`。
3. H3 未闭合：primary labels 可用，但 secondary delta 缺 `14380`，downstream 不 ready。
4. H4 部分成立：risk/support feature factory 找到 `RSF6`，bad-risk AUC = `0.750961`，但 safe-good AUC 不足，且不能直接产生 official accept region。
5. H5 未成立：dataset-agnostic repair 没有通过 official gate；best selected DR7 在 heldout 上 accepted count = `0`。
6. H6 未成立：empty-step-free / batch-major runtime 仍是 diagnostic estimate，没有 measured materialized runtime pass。
7. H7 未打开：P9 system controller pass = `0`，P10-P12 全部 gate-blocked。

最终一句话：

> v9.2.82 真实执行后停在 `R16-StableAcceptDecisionRegionUnsafe`：risk/support 重建找到了一个边缘可用的 bad-risk feature，但没有形成 heldout 可用的 dataset-agnostic decision repair；empty-step-free batch-major runtime 仍未 materialize，strict PureKAN functional 仍未成功。

## 15. 代码整理补充

按后续整理要求，本轮将 v9.2.82 runner 中可复用的表格/审计逻辑抽入框架模块：

```text
experiments/dgkan_outcome_controller.py
```

抽取内容：

```text
CSV/JSON/hash stable IO
fnum/inum/token_int
Wilson LCB/UCB
AUC
metric_summary
candidate feature attachment
calibrated risk/support stats
risk/support feature table
calibration-frozen decision repair search
```

重构后 runner：

```text
experiments/run_v9282_outcome_grounded_risk_support_rebuild_empty_step_free_batch_major_runtime_closure.py
```

重构验证命令：

```text
python -m py_compile experiments/dgkan_outcome_controller.py experiments/run_v9282_outcome_grounded_risk_support_rebuild_empty_step_free_batch_major_runtime_closure.py

python experiments/run_v9282_outcome_grounded_risk_support_rebuild_empty_step_free_batch_major_runtime_closure.py \
  --out-dir results/real_rerun_20260506/v9282_outcome_grounded_risk_support_rebuild_empty_step_free_batch_major_runtime_closure_refactor_check_20260513T230000Z \
  --fresh --device auto --data-root data --seed 1314
```

Refactor-check 结果保持主结论不变：

```text
route = R16-StableAcceptDecisionRegionUnsafe
decision_repair_pass = 0
batch_major_runtime_pass = 0
system_legal_controller_pass = 0
best_bad_risk_feature = RSF6-RiskResidualScore
best_bad_risk_auc = 0.7509612892076385
decision_repair_candidate_id = DR7-_risk_residual_score-thr-0.715503-n1.0-s0.05
precision_cal = 0.7630434782608696
bad_event_cal = 0.10652173913043478
precision_heldout = 0.0
coverage_heldout = 0.0
```

Refactor-check hash：

| artifact | SHA256 |
|---|---|
| framework module | `44f2d48e7012ff3cffce9ae9d231f33c1ca855b10330e75107d7e0f21f5fdea5` |
| refactored runner | `6723b790aba6ae1368f547ccc8e4b3152895be70b453547e8e807e0a5c90ce39` |
| refactor-check route | `1a1bb5b85e791ba967ffd9f8b228873ba2f143ebfb8815ea9fd9d88395bff05f` |
| refactor-check P5 features | `6c04dc9ff7a2018b45a7133cd013e64b96bd3ca0ae2cb6f77b454ef0998d410e` |
| refactor-check P6 decision repair | `9bc78ab10c137a9a64fb739d1f4f9f009829fcc6c97d8a649aa9e0d0e2735bf9` |
| refactor-check P9 system | `5b7b5a98932a3d4344fc021fcb6684ae9602517937462a5cf0729c0877145459` |

判断：代码整理只改变 runner 组织方式，没有改变 v9.2.82 的 route、decision gate 或 runtime gate。没有把 refactor-check 写成新的 official pass。
