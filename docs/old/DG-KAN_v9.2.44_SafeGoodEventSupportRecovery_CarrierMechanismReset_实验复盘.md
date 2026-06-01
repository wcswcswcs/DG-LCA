# DG-KAN v9.2.44 Safe Good-Event Support Recovery 与 Carrier Mechanism Reset 实验复盘

> 本复盘记录 `DG-KAN_v9.2.44_SafeGoodEventSupportRecovery_CarrierMechanismReset_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R8-OracleHighLegalFeatureGap
base_candidate = LQ-t2-h256
success_v9244_strict_purekan_functional = False
success_v9244_full_functional = False
success_v9244_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9244_safe_good_event_support_recovery_carrier_mechanism_reset_first_20260511T163000Z/
```

核心结论：

1. P0 复现 v9.2.43 boundary：source route = `R7-OracleLowCarrierMechanismReset`，fresh v2 bad-event rate = `0.07682291666666667`，source oracle pass = `0`。
2. P1 failure decomposition pass = `1`，primary mode = `A6-risk_score_missing`；bad-event 与 oracle-collapse attribution fraction 均为 `1.0`。
3. P2 risk-first support filter 找到 survivor：`F1-BranchRatioRiskSafe`，oracle precision = `0.967480`，coverage = `0.040039`，bad-event = `0.032520`。
4. P3 carrier reset matrix 没有 route-level carrier pass：best `A1-RiskBoundedTailCarrier` 有 oracle support，但 carrier active = `0`，`r_perp_tail = 0.084411 < 0.10`。
5. P4 legal controller 未通过：best `C1-RiskFirstS7` precision = `0.794118`，但 coverage = `0.022135 < 0.03` 且 bad-event = `0.117647 > 0.05`。
6. 当前 blocker：`safe_good_support_recovered_but_legal_controller_failed`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9244_safe_good_event_support_recovery_carrier_mechanism_reset.py` | v9.2.44 runner；从 v9.2.43 真实 fresh-v2 rows 重建 comparable events，执行 failure attribution、risk support、carrier reset、legal controller gate、route/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9244_safe_good_event_support_recovery_carrier_mechanism_reset.py
```

正式运行：

```bash
python experiments/run_v9244_safe_good_event_support_recovery_carrier_mechanism_reset.py \
  --out-dir results/real_rerun_20260506/v9244_safe_good_event_support_recovery_carrier_mechanism_reset_first_20260511T163000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

说明：本轮 P1-P4 使用 v9.2.43 已落盘 fresh-v2 measured rows 重建 RealFunctional / AdamWParallel / bestLR comparable event table；没有把未测的新 carrier 伪装成新训练结果。

## 2. Route

```json
{
  "route": "R8-OracleHighLegalFeatureGap",
  "base_candidate": "LQ-t2-h256",
  "v9243_boundary_pass": 1,
  "fresh_v2_failure_mode": "A6-risk_score_missing",
  "bad_event_attribution_pass": 1,
  "oracle_collapse_attribution_pass": 1,
  "best_risk_filter": "F1-BranchRatioRiskSafe",
  "risk_filter_support_pass": 1,
  "best_carrier_id": "A1-RiskBoundedTailCarrier",
  "carrier_oracle_support_pass": 1,
  "carrier_active": 0,
  "carrier_support_pass": 0,
  "oracle_precision": 0.967479674796748,
  "oracle_coverage": 0.0400390625,
  "oracle_bad_event": 0.032520325203252036,
  "best_controller_id": "C1-RiskFirstS7",
  "legal_controller_pass": 0,
  "controller_auc": 0.5025066844919787,
  "controller_corr": -0.08782524584578971,
  "accepted_precision": 0.7941176470588235,
  "accepted_coverage": 0.022135416666666668,
  "accepted_bad_event_rate": 0.11764705882352941,
  "paired_replay_pass": 0,
  "primary_blocker": "safe_good_support_recovered_but_legal_controller_failed",
  "next_required_implementation": "redesign_legal_features_on_branch_ratio_safe_support"
}
```

## 3. P1 fresh v2 failure decomposition

Artifact：

```text
p1_fresh_v2_failure_decomposition.csv
fresh_v2_failure_trace_v9244.csv
```

关键值：

| metric | value |
|---|---:|
| real event rows | `3072` |
| bad event count | `236` |
| oracle fail event count | `80` |
| primary failure mode | `A6-risk_score_missing` |
| bad-event attribution fraction | `1.000000` |
| oracle-collapse attribution fraction | `1.000000` |

判断：v9.2.43 fresh-v2 collapse 不是没有 event movement，而是高 value / high risk event 混在一起；继续只调 value threshold 会把 bad-event 带入 accepted set。

## 4. P2 risk-first support filters

Artifact：

```text
p2_risk_first_support_filters.csv
risk_filter_trace_v9244.csv
```

| filter | support pass | oracle precision | coverage | bad-event | strata | families | legal precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0-NoRiskFilter | `0` | `0.826464` | `0.150065` | `0.173536` | `8` | `101` | `0.717391` |
| F1-BranchRatioRiskSafe | `1` | `0.967480` | `0.040039` | `0.032520` | `3` | `26` | `0.608696` |
| F2-UncertaintyLCBStratumOnly | `0` | `0.695652` | `0.029948` | `0.000000` | `1` | `12` | `0.347826` |
| F3-ControlGapAttachSafe | `0` | `0.837398` | `0.040039` | `0.162602` | `8` | `18` | `0.674797` |
| F6-BranchOrLCBRiskSafe | `1` | `0.967480` | `0.040039` | `0.032520` | `3` | `26` | `0.608696` |

判断：safe-good support 可恢复；这把 v9.2.43 的 “oracle low” 推进为 “risk-first oracle high”。但 legal score 在同一支持集上的 precision 只有 `0.608696`，还不是 controller success。

## 5. P3 carrier reset matrix

Artifact：

```text
p3_carrier_reset_matrix.csv
carrier_reset_trace_v9244.csv
```

| carrier | active | oracle support | carrier pass | oracle precision | coverage | bad-event | r_z tail | r_perp tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0-CurrentCarrier-v9243 | `1` | `0` | `0` | `0.826464` | `0.150065` | `0.173536` | `0.171485` | `0.137168` |
| A1-RiskBoundedTailCarrier | `0` | `1` | `0` | `0.967480` | `0.040039` | `0.032520` | `0.107492` | `0.084411` |
| A3-RoleWiseFT7ResetCarrier | `1` | `0` | `0` | `0.804878` | `0.040039` | `0.195122` | `0.171485` | `0.137168` |
| A6-HybridRoleControlRiskCarrier | `0` | `1` | `0` | `0.943089` | `0.040039` | `0.032520` | `0.105529` | `0.084411` |

判断：risk-bounded support 能找出 safe-good oracle events，但 measured current rows 下对应 carrier active gate 不完整，主要卡在 `r_perp_tail < 0.10`。因此 P3 不能写成 carrier reset success。

## 6. P4 legal controller after support recovery

Artifact：

```text
p4_legal_controller_after_support_recovery.csv
legal_controller_trace_v9244.csv
```

| controller | official | pass | AUC | corr | heldout precision | coverage | bad-event |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1-RiskFirstS7 | `1` | `0` | `0.502507` | `-0.087825` | `0.794118` | `0.022135` | `0.117647` |
| C2-RiskFirstControlGap | `1` | `0` | `0.460248` | `-0.087825` | `0.037975` | `0.154297` | `0.113924` |
| C3-RiskFirstRoleGap | `1` | `0` | `0.496085` | `-0.087825` | `0.143541` | `0.136068` | `0.143541` |
| C4-TwoStageRiskFamily | `1` | `0` | `0.468737` | `-0.087825` | `0.084211` | `0.123698` | `0.063158` |
| C5-FamilyBalancedLCB | `1` | `0` | `0.469281` | `-0.087825` | `0.092784` | `0.126302` | `0.144330` |
| C6-Oracle | `0` | `0` | `0.565274` | `-0.087825` | `0.293103` | `0.151042` | `0.017241` |

判断：P4 是本轮 terminal blocker。`C1` 的 precision 看似过 `0.75`，但 coverage 低于 `0.03`，bad-event 高于 `0.05`，且 shape gate 也不过；不能打开 leave-out 或 paired replay。

## 7. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p5_leave_dataset_and_stratum_out.csv` | `P4_legal_controller_failed_after_support_recovery` |
| `p6_official_paired_replay.csv` | same |
| `p7_short_run_functional_validation.csv` | same |
| `p8_full_10seed_functional_validation.csv` | same |
| `p9_robustness_external_ready.csv` | same |

没有把 risk-filter oracle support、carrier diagnostic 或 posthoc oracle 写成 paired replay / short-run / full-run success。

## 8. No-fake audit

```text
rows_checked = 3104
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
| plan | `a9f8d74fffdbaabd0d1233af8918dacb62b670813e9b8948d09164809669700b` |
| runner | `94597c81d512dc0113c4f7a67364377458bec14c1b69cfee1cdf07b1e43ae0d4` |
| route | `fb9e2a9dee77b8bd00cc744acc9d415645d27ce910ae609f196954d1a3c86181` |
| P0 boundary | `9efaa37a75bb8527e424009649e5425a694febe59490be8cc532eebfe199b81a` |
| P1 decomposition | `f49ded9a09971aeb0cb37226010a579a8c171927b1397a7d06892be3fd233921` |
| P2 risk filters | `94c3f90d9148eeb802d4a860846c154e210359aaaa06f96814d29ff34c677ab9` |
| P3 carrier reset | `c0f3adab1200045757a8644851099a0252d2d94abfcafd005cf5336e0fa38ff6` |
| P4 legal controller | `7b3dcc9bb7efe35cad873232d937925acba186d092d7056add774e65cf7d19d1` |
| provenance audit | `72b0f1081423996514e0ec2851a2f47b9df313bc080524ca8ff461290de71a71` |

## 10. 最终分析结论

v9.2.44 的真实推进是：

```text
v9.2.43: expanded fresh-v2 support made bad-event and oracle both fail.
v9.2.44: failure attribution identifies missing risk gate;
          branch-ratio risk-safe support recovers oracle safe-good events;
          legal controller still cannot identify them robustly on heldout.
```

机制判断：

1. v9.2.43 不是 “S7 threshold 差一点” 的简单问题；bad-event / oracle collapse 可归因到 `A6-risk_score_missing`。
2. 本轮证明 safe-good support 并非完全不存在：`F1-BranchRatioRiskSafe` 的 oracle support 过 gate。
3. 但 support recovery 还没有转化成 legal controller success：best `C1` heldout coverage 与 bad-event 同时破 gate。
4. Carrier reset 层面也不能写成功：`A1/A6` 有 oracle support，但 measured carrier active 不过；`A3` active 但 bad-event 太高。
5. 因此 P5 leave-out、P6 official paired replay、short/full/external-ready 都正确保持关闭。

最终一句话：

> v9.2.44 真实执行后停在 `R8-OracleHighLegalFeatureGap`：safe-good oracle support 已通过 risk-first filter 恢复，但 legal controller 仍未能在 heldout 上稳定识别这些事件，strict PureKAN functional 仍未成功。
