# DG-KAN v9.2.45 Legal Feature Representation 与 Support-Matched Controller Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.45_LegalFeatureRepresentation_SupportMatchedControllerClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R2-LegalFeatureGapUnattributed
base_candidate = LQ-t2-h256
success_v9245_strict_purekan_functional = False
success_v9245_full_functional = False
success_v9245_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9245_legal_feature_representation_support_matched_controller_closure_first_20260511T173000Z/
```

核心结论：

1. P0 复现 v9.2.44 boundary：source route = `R8-OracleHighLegalFeatureGap`，F1 support oracle pass 已保留。
2. P1 F1/C1 gap autopsy 未过：`p1_gap_attribution_pass = 0`，`F1_C1_jaccard = 0.294574`。
3. C1 false-negative attribution 通过：`85` rows，primary = `G2-threshold_miscalibration`，fraction = `1.0`。
4. C1 false-positive attribution 未通过：`6` rows，primary = `G5-control_gap_mismatch`，fraction = `0.666667 < 0.90`。
5. P2 F1 support 在 source split 上保持 pass：precision `0.967480`，coverage `0.040039`，bad-event `0.032520`；但 heldout bad-event = `0.065574`。
6. P3 legal feature factory 有 predictivity survivor，best = `LF8-HybridLegalMonotone`；但由于 P1 gate 未过，这不能写成 controller success。
7. P4 tri-stage controller 未过，best = `C0-C1RiskFirstS7Reference`，coverage `0.012370`，bad-event `0.210526`。
8. 当前 blocker：`legal_feature_gap_unattributed`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9245_legal_feature_representation_support_matched_controller_closure.py` | v9.2.45 runner；从 v9.2.44/v9.2.43 source-measured fresh-v2 rows 重建 comparable event table，执行 F1/C1 gap、support split、legal feature factory、tri-stage controller gate、route/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9245_legal_feature_representation_support_matched_controller_closure.py
```

正式运行：

```bash
python experiments/run_v9245_legal_feature_representation_support_matched_controller_closure.py \
  --out-dir results/real_rerun_20260506/v9245_legal_feature_representation_support_matched_controller_closure_first_20260511T173000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

说明：本轮没有补造 train-stream micro-probe；`LF7-TrainStreamMicroProbe` 明确为 `not_measured_in_source` / `official_eligible = 0`。

## 2. Route

```json
{
  "route": "R2-LegalFeatureGapUnattributed",
  "base_candidate": "LQ-t2-h256",
  "v9244_boundary_pass": 1,
  "f1_support_stability_pass": 1,
  "f1_oracle_precision": 0.967479674796748,
  "f1_oracle_coverage": 0.0400390625,
  "f1_oracle_bad_event": 0.032520325203252036,
  "heldout_f1_oracle_precision": 0.9344262295081968,
  "heldout_f1_oracle_coverage": 0.039713541666666664,
  "heldout_f1_oracle_bad_event": 0.06557377049180328,
  "f1_c1_jaccard": 0.29457364341085274,
  "p1_gap_attribution_pass": 0,
  "c1_false_positive_count": 6,
  "c1_false_positive_primary_mode": "G5-control_gap_mismatch",
  "c1_false_positive_attribution_fraction": 0.6666666666666666,
  "c1_false_negative_count": 85,
  "c1_false_negative_primary_mode": "G2-threshold_miscalibration",
  "c1_false_negative_attribution_fraction": 1.0,
  "best_legal_feature_id": "LF8-HybridLegalMonotone",
  "legal_feature_predictivity_pass": 1,
  "best_controller_id": "C0-C1RiskFirstS7Reference",
  "tri_stage_controller_pass": 0,
  "controller_auc": 0.6552013703208556,
  "controller_corr": 0.3916916905783157,
  "accepted_precision": 0.7894736842105263,
  "accepted_coverage": 0.012369791666666666,
  "accepted_bad_event_rate": 0.21052631578947367,
  "primary_blocker": "legal_feature_gap_unattributed"
}
```

## 3. P1 F1/C1 gap autopsy

Artifact：

```text
p1_f1_c1_legal_feature_gap_autopsy.csv
f1_c1_overlap_trace_v9245.csv
```

Summary：

```text
p1_pass = 0
F1_accept_count = 123
C1_accept_count = 44
F1_C1_intersection_count = 38
F1_C1_jaccard = 0.29457364341085274
C1_false_positive_count = 6
C1_false_positive_primary_mode = G5-control_gap_mismatch
C1_false_positive_attribution_fraction = 0.6666666666666666
C1_false_negative_count = 85
C1_false_negative_primary_mode = G2-threshold_miscalibration
C1_false_negative_attribution_fraction = 1.0
```

判断：false negatives 可以解释为 threshold miscalibration，但 false positives 没有达到 `>=0.90` attribution gate，因此 P1 不允许写成 legal gap fully attributed。

## 4. P2 support stability

Artifact：

```text
p2_fresh_support_stability.csv
```

| split | rows | F1 accept | precision | coverage | bad-event | split pass |
|---|---:|---:|---:|---:|---:|---:|
| all_source_fresh_v2 | `3072` | `123` | `0.967480` | `0.040039` | `0.032520` | `1` |
| calibration_seed_0_3 | `1536` | `61` | `0.983607` | `0.039714` | `0.000000` | `1` |
| heldout_seed_4_7 | `1536` | `61` | `0.934426` | `0.039714` | `0.065574` | `0` |

判断：F1 support 在 all/calibration 上恢复 safe-good support，但 heldout bad-event 超过 `0.05`，说明 support 边界仍有泛化风险。

## 5. P3 legal feature factory

Artifact：

```text
p3_legal_feature_factory_audit.csv
legal_feature_trace_v9245.csv
feature_overhead_trace_v9245.csv
```

| feature | official | predictive | AUC to F1 | AUC to safe-good | corr | precision | coverage | bad-event |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LF0-CurrentC1S7 | `1` | `1` | `0.618848` | `0.688877` | `0.408795` | `0.717391` | `0.029948` | `0.239130` |
| LF1-BranchRatioRisk | `1` | `1` | `0.767685` | `0.381687` | `-0.317415` | `0.100977` | `0.099935` | `0.019544` |
| LF6-DriftDiffusionSNR | `1` | `1` | `0.806572` | `0.379672` | `-0.209281` | `0.138829` | `0.150065` | `0.008677` |
| LF8-HybridLegalMonotone | `1` | `1` | `0.823534` | `0.401034` | `-0.196182` | `0.146341` | `0.120117` | `0.008130` |
| LF7-TrainStreamMicroProbe | `0` | `0` | `0.500000` | `0.500000` | `0.000000` | `0.000000` | `0.150065` | `0.000000` |

判断：P3 显示 commit-time legal features 有 F1-support predictivity，但这一步在本轮只能作为诊断，因为 P1 gap attribution 未闭合。

## 6. P4 tri-stage controller

Artifact：

```text
p4_tristage_legal_controller_calibration.csv
controller_calibration_trace_v9245.csv
```

| controller | official | pass | AUC heldout | corr heldout | precision | coverage | bad-event |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0-C1RiskFirstS7Reference | `1` | `0` | `0.655201` | `0.391692` | `0.789474` | `0.012370` | `0.210526` |
| C1-BranchRiskTriStage | `1` | `0` | `0.512780` | `-0.046228` | `0.206349` | `0.041016` | `0.000000` |
| C5-SNR-GatedControlGap | `1` | `0` | `0.418395` | `-0.271356` | `0.128000` | `0.081380` | `0.032000` |
| C7-HybridLegalMonotone | `1` | `0` | `0.463845` | `-0.155025` | `0.141593` | `0.073568` | `0.026549` |
| C8-Oracle | `0` | `0` | `0.957219` | `1.000000` | `0.375000` | `0.083333` | `0.625000` |

判断：没有 legal tri-stage controller 通过 heldout gate。`C0` precision 够，但 coverage 太低且 bad-event 太高；`C1/C5/C7` bad-event 可控但 precision 不够。

## 7. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p5_leave_dataset_and_stratum_out.csv` | `P1_f1_c1_gap_attribution_failed` |
| `p6_official_paired_replay.csv` | same |
| `p7_short_run_functional_validation.csv` | same |
| `p8_full_10seed_functional_validation.csv` | same |
| `p9_robustness_external_ready.csv` | same |

没有把 P3 legal feature predictivity 或 P4 diagnostic controller rows 倒灌成 leave-out / paired replay / short-run / full-run success。

## 8. No-fake audit

```text
rows_checked = 3105
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
| plan | `45923a8e2452a0e6bc572348f3168954179d10db59a4b3c28a46e3b2e1987eae` |
| runner | `a941f8184dc6aa2e69d2476665da7d3e566f3dadcccbd061cb1fc9d262c6e8d1` |
| route | `00e24472d7cdc556003a25a57052a1050b5bfbc72605fae7be6407f9e591dc35` |
| P0 boundary | `86dab1d0e12523e235b18cdee54de2325413a035cd8c753f057dc6d38ea9c59d` |
| P1 gap autopsy | `a5cf2b7cffd3cb9ed07cd6f5644b0ce27fc389cfcf78fcb936cb7d7a70f703b8` |
| P2 support stability | `0738946d56edad001fcbae54fe70e48b9f349e88e56a47766cccbaba0a687ad6` |
| P3 legal feature factory | `47f2134fae0d08dbafe173efa191ecc84b5323f857577db82bdef4097d3c1c36` |
| P4 tri-stage controller | `b271b8b28e356fabea5c90477854c382ab5c77d7339976f95d9a2efbde6b3549` |
| P5 boundary | `b4dbd2199601b8eb942f6f2e65f0ed1e83c8e7615712a1dc6047ecc2aa3d5b52` |
| provenance audit | `52ba653a269cb3a3d869574bc97cc40bb8e2fc76f1cd7fb0b7f7c41c4b71efd5` |

## 10. 最终分析结论

v9.2.45 的真实推进是：

```text
v9.2.44: safe-good oracle support 已被 F1 risk-first filter 恢复；
v9.2.45: 尝试把 F1 support 转成 legal feature / support-matched controller。
```

机制判断：

1. F1 support 仍然不是空信号，all-source precision/coverage/bad-event 仍满足 oracle support gate。
2. 但 C1 与 F1 的 overlap 低，`F1_C1_jaccard = 0.294574`，说明 legal controller 没有稳定复现 F1 support。
3. false-negative 可以归因到 threshold miscalibration，但 false-positive attribution fraction 只有 `0.666667`，P1 attribution gate 没闭合。
4. P3 的 legal features 有可测 predictivity，但在 P1 未过时不能视作 official controller success。
5. P4 没有 tri-stage survivor，因此 leave-out、paired replay、short/full validation 与 external-ready 均保持关闭。

最终一句话：

> v9.2.45 真实执行后停在 `R2-LegalFeatureGapUnattributed`：F1 safe-good support 仍存在，但 C1/F1 gap 还没有被充分归因，legal controller 不能进入 leave-out 或 paired replay。
