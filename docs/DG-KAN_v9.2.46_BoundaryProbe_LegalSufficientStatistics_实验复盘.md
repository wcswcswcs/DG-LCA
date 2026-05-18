# DG-KAN v9.2.46 Boundary-Probe Legal Sufficient Statistics 实验复盘

> 本复盘记录 `DG-KAN_v9.2.46_BoundaryProbe_LegalSufficientStatistics_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R3-LegalFeatureSafeGoodPass
base_candidate = LQ-t2-h256
success_v9246_strict_purekan_functional = False
success_v9246_full_functional = False
success_v9246_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9246_boundary_probe_legal_sufficient_statistics_first_20260511T183000Z/
```

核心结论：

1. P0 复现 v9.2.45 boundary：source route = `R2-LegalFeatureGapUnattributed`，source tri-stage controller pass = `0`。
2. P1 boundary-probe disagreement expansion pass = `1`：FP `570`，FN `495`，FP/FN attribution pass 均为 `1`。
3. P2 target regrounding pass = `1`：safe-good label reliability = `0.983073`，四个 labels 均已从真实 comparable rows 测量。
4. P3 legal sufficient statistics pass = `1`：best legal safe-good feature = `LF8-BoundaryProbeLegalStats`，safe-good AUC = `0.736749`。
5. P3 component gate 也通过：risk/value best = `LF2-RiskTailLCB`，gap best = `LF8-BoundaryProbeLegalStats`。
6. P4 factorized controller pass = `0`，best = `C6-HybridLegalMonotoneV2`，heldout precision `0.5`，coverage `0.001302`，bad-event `0.5`。
7. 当前 blocker：`factorized_controller_failed_heldout_gate`。

说明：本轮 P1 的 “boundary-probe rows” 是对 v9.2.43/v9.2.44/v9.2.45 已落盘真实 event rows 的多 probe diagnostic view；没有生成新 replay outcome。`LF8` 是从 source-measured event-time statistics 构造的 boundary-probe legal statistic，不是补造新的 train-stream micro-probe。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9246_boundary_probe_legal_sufficient_statistics.py` | v9.2.46 runner；从 v9.2.43/v9.2.44/v9.2.45 source-measured fresh-v2 rows 执行 boundary disagreement expansion、target regrounding、legal feature v2、factorized controller calibration、route/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9246_boundary_probe_legal_sufficient_statistics.py
```

正式运行：

```bash
python experiments/run_v9246_boundary_probe_legal_sufficient_statistics.py \
  --out-dir results/real_rerun_20260506/v9246_boundary_probe_legal_sufficient_statistics_first_20260511T183000Z \
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

```json
{
  "route": "R3-LegalFeatureSafeGoodPass",
  "base_candidate": "LQ-t2-h256",
  "v9245_boundary_pass": 1,
  "f1_support_stability_pass": 1,
  "f1_oracle_precision": 0.967479674796748,
  "f1_oracle_coverage": 0.0400390625,
  "f1_oracle_bad_event": 0.032520325203252036,
  "f1_c1_jaccard": 0.29457364341085274,
  "fp_count": 570,
  "fn_count": 495,
  "fp_attribution_pass": 1,
  "fn_attribution_pass": 1,
  "target_grounding_pass": 1,
  "label_reliability_safe_good": 0.9830729166666666,
  "best_risk_feature": "LF2-RiskTailLCB",
  "best_value_feature": "LF2-RiskTailLCB",
  "best_gap_feature": "LF8-BoundaryProbeLegalStats",
  "best_microprobe_feature": "LF8-BoundaryProbeLegalStats",
  "legal_feature_predictivity_pass": 1,
  "component_all_pass": 1,
  "best_controller_id": "C6-HybridLegalMonotoneV2",
  "factorized_controller_pass": 0,
  "controller_auc": 0.48203189253404005,
  "controller_corr": -0.012860634240887493,
  "accepted_precision": 0.5,
  "accepted_coverage": 0.0013020833333333333,
  "accepted_bad_event_rate": 0.5,
  "primary_blocker": "factorized_controller_failed_heldout_gate"
}
```

## 3. P1 boundary disagreement expansion

Artifacts：

```text
p1_boundary_disagreement_expansion.csv
disagreement_trace_v9246.csv
```

Summary：

```text
boundary_probe_count = 6
false_positive_count = 570
false_negative_count = 495
agree_good_count = 243
false_positive_primary_mode = G2-threshold_miscalibration
false_positive_primary_fraction = 0.38070175438596493
false_positive_attribution_fraction = 1.0
false_negative_primary_mode = G2-threshold_miscalibration
false_negative_primary_fraction = 0.98989898989899
false_negative_attribution_fraction = 1.0
p1_pass = 1
```

判断：v9.2.45 的 “FP only 6 rows” 问题被 boundary-probe diagnostic 扩大。需要注意的是 FP primary mode 并不集中，primary fraction 只有 `0.380702`；本轮通过的是 attribution coverage，不是单一主因闭合。

## 4. P2 target regrounding

Artifact：

```text
p2_target_regrounding.csv
safe_good_label_trace_v9246.csv
```

| metric | value |
|---|---:|
| row count | `3072` |
| risk-safe rate | `0.923177` |
| value-positive rate | `0.889323` |
| control-resistant rate | `0.166667` |
| safe-good rate | `0.123698` |
| safe-good reliability | `0.983073` |
| target grounding pass | `1` |

判断：P2 只把 offline audit labels 重新落地；这些 labels 没有进入 commit-time controller。

## 5. P3 legal feature factory v2

Artifacts：

```text
p3_legal_feature_factory_v2.csv
legal_feature_trace_v9246.csv
train_stream_microprobe_trace_v9246.csv
```

| feature | official | predictive | risk AUC | value AUC | gap AUC | safe-good AUC | precision | coverage | bad-event |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LF1-BranchRatioDecomposition | `1` | `0` | `0.623853` | `0.594363` | `0.381728` | `0.409153` | `0.140921` | `0.120117` | `0.010840` |
| LF2-RiskTailLCB | `1` | `0` | `0.724460` | `0.723127` | `0.245498` | `0.265810` | `0.006508` | `0.150065` | `0.000000` |
| LF3-ControlTailGap | `1` | `1` | `0.279399` | `0.280848` | `0.738120` | `0.721080` | `0.265583` | `0.120117` | `0.157182` |
| LF4-ValuePositiveTailLCB | `1` | `0` | `0.723494` | `0.722205` | `0.249991` | `0.269392` | `0.006508` | `0.150065` | `0.000000` |
| LF8-BoundaryProbeLegalStats | `1` | `1` | `0.272416` | `0.272486` | `0.755219` | `0.736749` | `0.268293` | `0.040039` | `0.146341` |

Summary：

```text
best_legal_feature_id = LF8-BoundaryProbeLegalStats
best_legal_feature_auc_safe_good = 0.7367492375068428
best_risk_feature = LF2-RiskTailLCB
best_risk_feature_auc = 0.7244597308216395
best_value_feature = LF2-RiskTailLCB
best_value_feature_auc = 0.7231267763327879
best_gap_feature = LF8-BoundaryProbeLegalStats
best_gap_feature_auc = 0.755218505859375
component_all_pass = 1
legal_feature_predictivity_pass = 1
```

判断：P3 证明 legal sufficient statistics 不是全灭；但 risk/value 与 gap 的方向存在张力，尚未形成 usable factorized controller。

## 6. P4 factorized controller calibration

Artifacts：

```text
p4_factorized_controller_calibration.csv
controller_calibration_trace_v9246.csv
```

| controller | official | pass | AUC heldout | corr heldout | precision | coverage | bad-event | strata | families |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1-RiskValueGapTriStage | `1` | `0` | `0.395313` | `-0.038984` | `0.230769` | `0.008464` | `0.076923` | `6` | `8` |
| C2-BranchRiskControlGap | `1` | `0` | `0.368703` | `-0.059439` | `0.000000` | `0.005208` | `0.125000` | `5` | `7` |
| C3-ProbeGatedTriStage | `1` | `0` | `0.415508` | `-0.068631` | `0.187500` | `0.010417` | `0.062500` | `6` | `8` |
| C5-RoleGapRiskFirst | `1` | `0` | `0.355390` | `-0.025593` | `0.071429` | `0.045573` | `0.200000` | `8` | `35` |
| C6-HybridLegalMonotoneV2 | `1` | `0` | `0.482032` | `-0.012861` | `0.500000` | `0.001302` | `0.500000` | `2` | `2` |
| C7-Oracle | `0` | `0` | `1.000000` | `0.471071` | `1.000000` | `0.080729` | `0.000000` | `8` | `48` |

判断：P4 是 terminal blocker。即使 P3 features 有 safe-good / component predictivity，factorized acceptance 在 heldout 上没有同时满足 precision、coverage、bad-event、family/strata gate。

## 7. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p5_leave_dataset_and_stratum_out.csv` | `P4_factorized_controller_failed` |
| `p6_official_paired_replay.csv` | same |
| `p7_short_run_functional_validation.csv` | same |
| `p8_full_10seed_functional_validation.csv` | same |
| `p9_robustness_external_ready.csv` | same |

没有把 boundary-probe diagnostic、P3 feature predictivity 或 oracle rows 写成 paired replay / short-run / full-run success。

## 8. No-fake audit

```text
rows_checked = 4407
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 9. Hash

| artifact | SHA256 |
|---|---|
| plan | `ecd2eb128c2399b68fc6ab8c37736862e733c9869ac49ecce876839c6dd4e9db` |
| runner | `66c61fd2873ba5a860b606ad8bca5fd0adf0af2c273127a957d447c1a22f4bda` |
| route | `d1ba3f2e8e2dea5c6f8d86132f2fc37855ed921447f6f9cdcc9942c50a61618b` |
| P0 boundary | `d0d1aad850d18d134e4deb3c612e288186a043dfedba73a953c7a14a62a7503e` |
| P1 disagreement | `644f4e0686aa00fc54b9d9da0afe7219a72892c04274566ed0baf0f8223059e9` |
| P2 target grounding | `70a606a42830395ece27acfa9afa004821ebc03571df928f4ea09626c9d48951` |
| P3 legal feature v2 | `a086238f9a9bf4ff47db2c42f91890763f9ecbc37bf506f58a903653eb0ccce7` |
| P4 factorized controller | `1723b28c072ac2c04099fe74d993eaab7f2a45ae723e27cb9329a851c0f56ab6` |
| P5 boundary | `cc0c5c70011e7897b2f7d02c9d1739ffd0863ccb6cd57463fdc721454fea0b6d` |
| provenance audit | `95435971640c138589f5fd8dc49cd1c06cbfc82fe0af70b38d80b162c5991bef` |

## 10. 最终分析结论

v9.2.46 的真实推进是：

```text
v9.2.45: F1/C1 gap 未充分归因，legal controller 未过。
v9.2.46: 用 boundary-probe diagnostic 扩大 disagreement，
          重新落地 risk/value/control/safe-good labels，
          并证明 legal sufficient statistics 有可测信号。
```

机制判断：

1. v9.2.45 的小样本 FP attribution blocker 被推进；本轮 P1 有足够 FP/FN boundary-probe rows。
2. legal statistics 层面不是完全无信号：`LF8` 对 safe-good 的 AUC 达到 `0.736749`。
3. 但 risk/value 与 control-gap 的可观测方向冲突：低 tail 更安全/更 value-positive，高 tail 更 control-resistant。
4. 因此 P4 factorized controller 无法在 heldout 上同时满足 precision、coverage 与 bad-event gate。
5. 当前不应打开 leave-out 或 paired replay；下一步应重做 factorized threshold/support balancing，而不是把 P3 feature success 写成 functional success。

最终一句话：

> v9.2.46 真实执行后停在 `R3-LegalFeatureSafeGoodPass`：boundary attribution 和 legal sufficient statistics 已推进，但 factorized controller heldout gate 失败，strict PureKAN functional 仍未成功。
