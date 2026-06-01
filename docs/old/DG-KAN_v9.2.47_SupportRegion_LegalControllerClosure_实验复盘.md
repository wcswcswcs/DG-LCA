# DG-KAN v9.2.47 Support-Region Legal Controller Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.47_SupportRegion_LegalControllerClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R7-OfflineFeatureOnly
base_candidate = LQ-t2-h256
success_v9247_strict_purekan_functional = False
success_v9247_full_functional = False
success_v9247_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9247_support_region_legal_controller_closure_first_20260511T193000Z/
```

核心结论：

1. P0 复现 v9.2.46 boundary：source route = `R3-LegalFeatureSafeGoodPass`，P1/P2/P3 pass，P4 factorized controller pass = `0`。
2. P1 decision-region autopsy pass = `1`：FP `3`，FN `124`，coverage-loss `124`，三类 attribution 均为 `1.0`。
3. P2 fresh online microprobe 未实现：`online_microprobe_implemented = 0`，原因是 `no_fresh_train_stream_update_probe_rows_available`。
4. P3 source-measured natural replay / oracle support pass = `1`：fresh natural rows = `3072`，signal strata = `8`，oracle precision = `1.0`，coverage = `0.123698`。
5. P4 support-region controller pass = `0`；best legal controller = `C1-ConstrainedPrecisionCoverage`，precision `0.157895`，coverage `0.012370`。
6. 当前 blocker：`fresh_online_microprobe_not_implemented`。

说明：本轮没有把 v9.2.43-v9.2.46 已落盘的 source-measured event-time diagnostic 伪装成 fresh online microprobe；P2 按真实可用性写为 `not_implemented`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9247_support_region_legal_controller_closure.py` | v9.2.47 runner；执行 v9.2.46 boundary 复现、decision-region autopsy、online microprobe legality gate、support-region controller boundary、route/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9247_support_region_legal_controller_closure.py
```

正式运行：

```bash
python experiments/run_v9247_support_region_legal_controller_closure.py \
  --out-dir results/real_rerun_20260506/v9247_support_region_legal_controller_closure_first_20260511T193000Z \
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

```json
{
  "route": "R7-OfflineFeatureOnly",
  "base_candidate": "LQ-t2-h256",
  "v9246_boundary_pass": 1,
  "decision_region_autopsy_pass": 1,
  "failure_mode": "D6-threshold_crossfit_miscalibration",
  "online_microprobe_implemented": 0,
  "online_microprobe_pass": 0,
  "microprobe_auc": 0.0,
  "microprobe_corr": 0.0,
  "fresh_natural_row_count": 3072,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.12369791666666667,
  "best_controller_id": "C1-ConstrainedPrecisionCoverage",
  "support_region_controller_pass": 0,
  "controller_auc": 0.37821267020012794,
  "controller_corr": -0.04492998051463861,
  "accepted_precision": 0.15789473684210525,
  "accepted_coverage": 0.012369791666666666,
  "accepted_bad_event_rate": 0.0,
  "primary_blocker": "fresh_online_microprobe_not_implemented",
  "next_required_implementation": "implement_real_train_stream_update_probe_microprobe"
}
```

判断：P3 oracle support 仍证明 source-measured comparable rows 中存在 safe-good support；但 P2 fresh online microprobe 没有真实实现，因此不能进入 official leave-out / paired replay。

## 3. P1 decision-region autopsy

Artifact：

```text
p1_decision_region_autopsy.csv
decision_region_trace_v9247.csv
```

| metric | value |
|---|---:|
| heldout rows | `1536` |
| accepted by C6 reference | `3` |
| false positive count | `3` |
| false negative count | `124` |
| coverage-loss count | `124` |
| FP attribution fraction | `1.0` |
| FN attribution fraction | `1.0` |
| coverage-loss attribution fraction | `1.0` |
| decision-region autopsy pass | `1` |

主要归因：

| type | primary mode | primary fraction |
|---|---|---:|
| FP | `D2-risk_value_gap_conflict` | `1.0` |
| FN | `D4-family_concentration` | `1.0` |
| coverage-loss | `D6-threshold_crossfit_miscalibration` | `1.0` |

判断：v9.2.46 的 factorized controller failure 可归因为 support-region / threshold / family concentration 的组合问题；这只是 autopsy pass，不是 controller pass。

## 4. P2 fresh online microprobe

Artifact：

```text
p2_fresh_online_microprobe.csv
online_microprobe_trace_v9247.csv
```

```text
status = not_implemented
online_microprobe_implemented = 0
online_microprobe_pass = 0
reason = no_fresh_train_stream_update_probe_rows_available
uses_dataset_name = 0
uses_validation = 0
uses_test = 0
uses_posthoc = 0
```

判断：计划要求 fresh train-stream update/probe half 的 commit-time microprobe。本轮没有这类真实落盘 rows，因此正确 route 是 `R7-OfflineFeatureOnly`；不能把 LF8/source-measured boundary-probe statistic 写成 online microprobe。

## 5. P3 natural replay / oracle support

Artifact：

```text
p3_fresh_natural_and_boundary_replay.csv
support_density_trace_v9247.csv
```

| metric | value |
|---|---:|
| fresh natural rows | `3072` |
| natural real events | `3072` |
| measured signal strata | `8` |
| carrier active | `1` |
| oracle accepted count | `380` |
| oracle precision | `1.0` |
| oracle coverage | `0.12369791666666667` |
| oracle bad-event | `0.0` |
| oracle support pass | `1` |

判断：P3 使用 source-measured fresh-v2 comparable rows 做 natural/boundary diagnostic view，没有生成新的 replay outcome。它说明 oracle-good support 仍存在，但不能替代 P2 online microprobe legality。

## 6. P4 support-region controller

Artifact：

```text
p4_support_region_controller_calibration.csv
controller_calibration_trace_v9247.csv
```

| controller | official | AUC | corr | precision | coverage | bad-event | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1-ConstrainedPrecisionCoverage | `1` | `0.378213` | `-0.044930` | `0.157895` | `0.012370` | `0.000000` | `0` |
| C2-RiskValueGapTriStageV2 | `1` | `0.487155` | `0.015816` | `0.000000` | `0.001953` | `0.000000` | `0` |
| C6-AbstainFirstConservative | `1` | `0.487155` | `0.015816` | `0.000000` | `0.001953` | `0.000000` | `0` |
| C7-Oracle | `0` | `1.000000` | `0.471071` | `1.000000` | `0.080729` | `0.000000` | `0` |

判断：legal controllers 没有通过 heldout gate。`C7-Oracle` 只作为 posthoc upper bound，`official_eligible = 0`，不能进入 route success。

## 7. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p5_leave_dataset_and_stratum_out.csv` | `P2_fresh_online_microprobe_failed` |
| `p6_official_paired_replay.csv` | same |
| `p7_short_run_functional_validation.csv` | same |
| `p8_full_10seed_functional_validation.csv` | same |
| `p9_robustness_external_ready.csv` | same |

没有把 decision-region autopsy、oracle support 或 support-region diagnostic controller rows 倒灌成 leave-out / paired replay / short-run / full-run success。

## 8. No-fake audit

```text
rows_checked = 4627
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
| plan | `c80c4dc57354ca06612807c8111149446328ee84ebf50aa77eb71f16c33c9b73` |
| runner | `450daf9f40afed779b60ac3c22e6f47acb61481301f6cdc22b526350c1feed89` |
| run manifest | `1458cb9739229a70187a69c7f44b8628e5be61446d53305d9da3fc7020111670` |
| route | `c42e4a94231dc78f606a554cde910e9e5d4fa6923e8ef387524c544008157aee` |
| P0 boundary | `aa632b2ae16c3aeaf642c1e0e4e767ca9600dfebb72c9b80bf20f92aaaa112f7` |
| P1 autopsy | `7e3c79e52561f2cf9b923e19fc00266a77723f06e9ed3f0eca02b15a377069f0` |
| P2 microprobe | `fb18f9fb25060d0c0387c1de8d25ca0c16314d779fdd90997c8c93f8658014d0` |
| P3 natural/oracle support | `6250e86c1a4cafa9010c4083f84a1f543e0d614a60dca9b403c3ad735d8fee07` |
| P4 controller | `0948293efb292c137431fdf7a962cdacd2c0ecc4a727cdbab29109e323644bd5` |
| P5 boundary | `119b6551af8fb30c20685b1adb3eeb05422b4a6d8f49efb6cf86b31062e8fc0b` |
| P6 boundary | `68241105fd9338323327abcbe0c86a73147c39db2aad6a0c4dabeeb43ec6fda7` |
| failure table | `58a175e5f7f693b484118d08b17c8e1290425f4025858e622e8124570e22f531` |
| provenance audit | `79d9d2d2c2c27beef19e361c1d53ad4852946f0abe2ffaed470c921ca1d6351e` |

## 10. 最终分析结论

v9.2.47 的真实推进是：

```text
v9.2.46: legal sufficient statistics 有信号，但 factorized controller heldout gate fail。
v9.2.47: decision-region failure 被归因，oracle support 仍存在；
          但 fresh online microprobe 没有真实实现，因此 route 停在 OfflineFeatureOnly。
```

机制判断：

1. P1/P3 说明 offline/source-measured diagnostic 仍有解释力，且 safe-good oracle support 没有消失。
2. 计划要求的 fresh online microprobe 是 commit-time train-stream probe；本轮没有可用真实 rows，必须显式阻断。
3. P4 legal support-region controller 在 heldout 上 precision 与 coverage 仍不过 gate，oracle row 也不能转正。
4. 因 P2 未过，P5-P9 均不能打开；strict PureKAN functional、full functional、external-ready 仍全部为 false。

最终一句话：

> v9.2.47 真实执行后停在 `R7-OfflineFeatureOnly`：decision-region autopsy 与 oracle support 有推进，但 fresh online microprobe 未真实实现，不能进入 official paired replay。
