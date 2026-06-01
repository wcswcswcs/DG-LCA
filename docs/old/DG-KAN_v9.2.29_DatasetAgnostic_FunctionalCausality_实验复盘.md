# DG-KAN v9.2.29 Dataset-Agnostic Functional Causality 实验复盘

> 本复盘记录 `DG-KAN_v9.2.29_DatasetAgnostic_FunctionalCausality_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R1-SignalStrataExplainFailures
base_candidate = LQ-t2-h256
success_v9229_strict_purekan_functional = False
success_v9229_full_functional = False
success_v9229_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/
```

核心结论：

1. P0 复现 v9.2.28 boundary：source route = `R5-KMNISTSurvivorNotPreserved`，fake/proxy = `0`。
2. P1 signal strata 解释 failure 通过：association margin = `0.256620`。
3. P2 event-value predictor 未过：corr = `0.197001`，threshold 要求 `>=0.30`。
4. P2 abstention 局部可行：precision = `0.830189`，coverage = `0.020167`，bad-event = `0.000000`。
5. 当前 blocker：`event_value_predictor_correlation_below_gate`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9229_datasetagnostic_functional_causality.py` | v9.2.29 runner；从 v9.2.28 真实 replay rows 生成 P0-P8 artifacts、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9229_datasetagnostic_functional_causality.py
```

正式运行：

```bash
python experiments/run_v9229_datasetagnostic_functional_causality.py \
  --out-dir results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z \
  --fresh \
  --seed 1314
```

## 2. Route

```json
{
  "abstention_precision_pass": 1,
  "accepted_bad_event_rate": 0.0,
  "accepted_coverage": 0.020167427701674276,
  "accepted_event_count": 53,
  "accepted_precision": 0.8301886792452831,
  "accepted_recall": 0.06748466257668712,
  "adamw_fullpass": 0,
  "association_margin": 0.25662007790380964,
  "base_candidate": "LQ-t2-h256",
  "best_signal_router": "not_opened",
  "dataset_failure_association": 0.45750784979834186,
  "dataset_tuning_detected": 0,
  "event_value_corr": 0.1970014764196975,
  "event_value_predictor_pass": 0,
  "external_ready": 0,
  "full_run_pass": 0,
  "functional_control_pass": 0,
  "functional_system_pass": 0,
  "functional_task_safe": 0,
  "hard_stratum_repair_pass": 0,
  "leave_dataset_out_pass": 0,
  "next_required_implementation": "redesign_event_value_predictor_with_real_pre_event_movement_features",
  "p1_row_count": 2628,
  "p2_row_count": 2628,
  "paired_replay_pass": 0,
  "primary_blocker": "event_value_predictor_correlation_below_gate",
  "robustness_pass": 0,
  "route": "R1-SignalStrataExplainFailures",
  "short_run_pass": 0,
  "signal_strata_explain_failures": 1,
  "signal_stratum_failure_association": 0.7141279277021515,
  "strong_baseline_pass": 0,
  "success_v9229_external_ready": 0,
  "success_v9229_full_functional": 0,
  "success_v9229_strict_purekan_functional": 0,
  "v9228_boundary_pass": 1
}
```

## 3. P1 dataset-stratified diagnosis

Rows = `2628`。

| signal stratum | rows | top failure | top fraction | actual gap | beat AdamWParallel | beat best LR |
|---|---:|---|---:|---:|---:|---:|
| S6-DelayedTailSignal | `821` | M3-control_dominated_signal | `0.518879` | `0.043240` | `0.432400` | `0.654080` |
| S5-LowActualMovementSilent | `697` | M5-low_value_or_silent | `0.863702` | `-0.255380` | `0.100430` | `0.388809` |
| S4-HighActualMovementButControlDominated | `496` | M2-effect_magnitude_too_small | `1.000000` | `-0.380040` | `0.000000` | `0.239919` |
| S3-HighCurvatureLowConfidence | `219` | M3-control_dominated_signal | `0.570776` | `-0.009132` | `0.429224` | `0.552511` |
| S1-HighCEHighMarginRisk | `138` | M7-control_superior_event | `0.594203` | `0.126812` | `0.594203` | `0.659420` |
| S2-LowMarginHighWrongConfidence | `137` | M3-control_dominated_signal | `0.948905` | `-0.313869` | `0.051095` | `0.321168` |
| S8-AbstainCandidate | `120` | M6-insufficient_abstention | `0.633333` | `-0.070833` | `0.366667` | `0.491667` |

P1 判断：

- dataset-failure association = `0.457508`
- signal-stratum-failure association = `0.714128`
- margin = `0.256620`

这说明 Fashion / KMNIST / MNIST 的差异可以转成 signal strata 诊断；但这只是诊断成功，不是 controller 成功。

## 4. P2 event-value predictor

P2 使用 dataset-agnostic features：`ce_tail_rank`、`margin_tail_rank`、`wrong_confidence_rank`。没有把 dataset name、seed id 或 class name 作为 route key。

| metric | observed | threshold | pass |
|---|---:|---:|---:|
| prediction corr | `0.197001` | `>=0.30` | `0` |
| accepted precision | `0.830189` | `>=0.70` | `1` |
| accepted coverage | `0.020167` | `[0.02,0.15]` | `1` |
| bad-event rate | `0.000000` | `<=0.05` | `1` |

判断：abstention 的 top slice 有局部精度，但 score 与真实 control gap 的相关性不足，因此不能进入 P3/P4 official signal-routed replay。

## 5. Downstream boundary

P3-P8 均已落盘为 `not_run`，原因是 `P2_event_value_predictor_failed`。没有把 leave-dataset-out、official paired replay、short/full run 或 external-ready 写成通过。

## 6. No-fake audit

```text
rows_checked = 5274
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 7. Hash

| artifact | SHA256 |
|---|---|
| `experiments/run_v9229_datasetagnostic_functional_causality.py` | `47b74ba7ea155315d1d356209fce01103ffc6be5b51bfc0a3d054c1fcc159d65` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/contract_audit_v9229.csv` | `29dfe26ab96980dcb536b650401d640d2d8a818db8da43a75c74d877d3136305` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/failure_table.csv` | `2c12031c0123216c09735e16f3c9c7c63c274f4afa3d7aa3c4d1c5a17ce5b866` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/leave_dataset_out_trace_v9229.csv` | `03d7f9a80dad0e58c0994354a7dc931fb1e0b28c7a10b1aed3cfa1f74169b0a3` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/p0_v9228_boundary_reproduction.csv` | `c59e1e32db77d44c7725f808929fe898ae670e02458767fadeef8de3a0c99b85` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/p1_dataset_stratified_diagnosis.csv` | `bc90ef77c208281c8efc8e3f055bf07904eabb338540100d9f6a0863c0468de8` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/p1_signal_stratum_summary.csv` | `6190e6cfe5cd0a77916de5c01e2aa485d646e3808febf4c47ee4c9d3e93679e7` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/p2_event_value_predictor_abstention.csv` | `0f99f468c3d14d2f993629e19ecfe0ce339d59885adeccc37b77355e211f9ceb` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/p3_leave_dataset_out_controller_validation.csv` | `daaef92fac183d3e14a35eb493c7b3302dc7be2f12fb941361ffb39454295cb3` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/p4_official_signal_routed_paired_replay.csv` | `6f54815d2916fe22e2255e7149226db71dbf133c0c58b07e45423e879a3a9149` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/p5_short_run_functional_validation.csv` | `6c8f1fc3243e146e78cd9c6aeb1286a07047511f9cba288a2e8a850bed067051` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/p6_full_10seed_functional_validation.csv` | `7fe5da8a274c8124953044ce38694ec1bed02180cb3bfea25f9d85a98827718b` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/p7_adamw_only_fullpass_repair.csv` | `5c8165dd9c8043fe511b50bcb0628a633e368bd6e1e01683e28c2e03fc41c8ac` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/p8_robustness_external_ready.csv` | `e2bcbe46d52d2752f0dbe1ac5eb704d9c1f72ac40f9df9a22b914f9fc86bc02b` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/paired_replay_branch_trace_v9229.csv` | `0765fe9b3e3246b22ea27ddb3ae518af784e3d882e5430697c688503cf59e6ea` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/route_decision.json` | `2c9edd33cfbca1ba17e6968347091798813064150eb64d301c3fc7791a92bec4` |
| `results/real_rerun_20260506/v9229_datasetagnostic_functional_causality_first_20260510T200000Z/v9229_provenance_audit.csv` | `91578fc710898c4f6a7e679ffc1979c8dc5dd76a6d671448d6dc34a8dce3dc21` |

## 8. 最终分析结论

v9.2.29 的真实推进是：

```text
v9.2.28: Fashion / KMNIST / routed controller 都没有 strict functional success。
v9.2.29: 这些 failure 能被 dataset-agnostic signal strata 解释，
          但当前 event-value score 不能可靠预测 Real vs controls gap。
```

机制判断：

1. 这轮支持“不要按 dataset 调参”的方向：failure 可转成 delayed tail、actual-movement-control-dominated、silent/low-value 等 signal strata。
2. 但当前可用 features 对 control gap 的预测相关性只有 `0.197001`，低于 `0.30` gate。
3. 局部 top-slice precision 较高，说明 event-value 不是完全没信号；问题是 ranking 不够连续可靠，不能支撑 official controller。
4. 因 P2 未过，P3 leave-dataset-out 和 P4 official signal-routed replay 不应打开。
5. 下一步应重做 event-value predictor / primitive-interface features，而不是回到 Fashion/KMNIST dataset-specific patch。

最终一句话：

> v9.2.29 真实执行后停在 `R1-SignalStrataExplainFailures`：`event_value_predictor_correlation_below_gate`。
