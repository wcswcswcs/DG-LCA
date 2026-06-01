# DG-KAN v9.2.31 Event-Value Grounding 与 Primitive Observability 实验复盘

> 本复盘记录 `DG-KAN_v9.2.31_EventValueGrounding_PrimitiveObservability_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R9-PrimitiveEffectUnpredictable
base_candidate = LQ-t2-h256
success_v9231_strict_purekan_functional = False
success_v9231_full_functional = False
success_v9231_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9231_event_value_grounding_primitive_observability_first_20260510T220000Z/
```

核心结论：

1. P0 复现 v9.2.30 boundary：source route = `R7-PrimitiveEffectUnpredictable`，fake/proxy = `0`。
2. P1 grounded value 可靠性 `value_reliability = 0.511631`，event-value grounding pass = `1`。
3. P2 pre-event observability pass = `0`；best feature = `effective_derivative`，best movement = `pre_adamwparallel_logit_delta_norm` / `0.184709`。
4. P3 failure-mode reclassification pass = `0`，GoodEvent precision = `0.000000`，abstain precision = `0.000000`。
5. P4 event-value predictor pass = `0`；best predictor = `not_opened`，corr = `0.000000`，precision = `0.000000`，coverage = `0.000000`。
6. 当前 blocker：`pre_event_features_do_not_predict_grounded_value`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9231_event_value_grounding_primitive_observability.py` | v9.2.31 runner；从 v9.2.30 真实 pre-event replay rows 生成 grounded value、observability、predictor / LDO gate、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9231_event_value_grounding_primitive_observability.py
```

正式运行：

```bash
python experiments/run_v9231_event_value_grounding_primitive_observability.py \
  --out-dir results/real_rerun_20260506/v9231_event_value_grounding_primitive_observability_first_20260510T220000Z \
  --fresh \
  --seed 1314
```

说明：本轮是 source-measured rows 的 grounding / observability audit。v9.2.30 没有测到的 curvature delta、control ECE/NLL comparable 均写为 `not_measured_in_v9230_source`，没有补造。

## 2. Route

```json
{
  "abstain_count": 0,
  "abstain_precision": 0.0,
  "abstention_coverage_pass": 0,
  "abstention_precision_pass": 0,
  "accepted_signal_strata_count": 0,
  "adamw_fullpass": 0,
  "assigned_fraction": 0.0,
  "base_candidate": "LQ-t2-h256",
  "best_movement_abs_corr": 0.18470922315538743,
  "best_movement_feature": "pre_adamwparallel_logit_delta_norm",
  "best_predictive_feature": "effective_derivative",
  "best_predictive_feature_abs_corr": 0.2597168939509335,
  "best_predictor": "not_opened",
  "cpu_offload_used": 0,
  "dataset_tuning_detected": 0,
  "event_value_grounding_pass": 1,
  "event_value_predictor_pass": 0,
  "external_ready": 0,
  "failure_mode_reclassification_pass": 0,
  "fake_data_used": 0,
  "fake_proxy_nonzero_count": 0,
  "family_value_predictable": 0,
  "full_run_pass": 0,
  "good_event_count": 0,
  "good_event_precision": 0.0,
  "leave_dataset_out_pass": 0,
  "leave_dataset_out_pass_count": 0,
  "leave_stratum_out_pass": 0,
  "leave_stratum_out_pass_count": 0,
  "next_required_implementation": "if_LDO_failed_add_real_repeat_h640_features_else_run_official_signal_value_paired_replay",
  "no_fake": true,
  "no_proxy": true,
  "normalized_value_mean": 2.351268649659562e-17,
  "normalized_value_std": 0.08781573818106787,
  "p1_row_count": 486,
  "paired_replay_pass": 0,
  "pre_event_feature_pass": 0,
  "pre_event_relevant_feature_count": 1,
  "predictor_auc": 0.5,
  "predictor_bad_event_rate": 0.0,
  "predictor_corr": 0.0,
  "predictor_coverage": 0.0,
  "predictor_precision": 0.0,
  "primary_blocker": "pre_event_features_do_not_predict_grounded_value",
  "proxy_row_used": 0,
  "raw_value_mean": -0.022014700098292823,
  "raw_value_std": 0.09287988151265801,
  "robustness_pass": 0,
  "route": "R9-PrimitiveEffectUnpredictable",
  "rows_checked": 524,
  "short_run_pass": 0,
  "success_v9231_external_ready": 0,
  "success_v9231_full_functional": 0,
  "success_v9231_strict_purekan_functional": 0,
  "v9230_boundary_pass": 1,
  "value_normalization_pass": 1,
  "value_reliability": 0.5116312586350301
}
```

## 3. P1 Event-Value Grounding

Artifact：

```text
p1_event_value_label_grounding.csv
```

关键值：

| metric | value |
|---|---:|
| rows | `486` |
| value reliability | `0.511631` |
| normalization pass | `1` |
| raw value mean | `-0.022015` |
| raw value std | `0.092880` |
| normalized value std | `0.087816` |

判断：grounded value 只使用 v9.2.30 同时具备 Real / AdamWParallel / bestLR comparable 的 CEp99 与 margin，再扣 task risk；curvature 与 control-ECE/NLL 未测，明确排除。

## 4. P2 Pre-Event Observability

Artifact：

```text
p2_pre_event_feature_relevance.csv
```

关键值：

| metric | value |
|---|---:|
| relevant feature count | `1` |
| best predictive feature | `effective_derivative` |
| best predictive abs corr | `0.259717` |
| best movement feature | `pre_adamwparallel_logit_delta_norm` |
| best movement abs corr | `0.184709` |

判断：pre-event feature 从 v9.2.30 的 near-threshold diagnostic 进入 grounded-value observability；但这仍只是 predictor 前置条件，不是 functional success。

## 5. P3/P4 Predictor Gate

Artifacts：

```text
p3_failure_mode_reclassification.csv
p4_event_value_predictor_redesign.csv
```

P3/P4 summary：

| metric | value |
|---|---:|
| failure-mode pass | `0` |
| GoodEvent precision | `0.000000` |
| abstain precision | `0.000000` |
| predictor pass | `0` |
| predictor corr | `0.000000` |
| predictor coverage | `0.000000` |
| accepted strata | `0` |

判断：如果 P4/P5 未过，P6 official paired replay 不能打开；本轮没有把 calibration rows 当成 official controller success。

## 6. Downstream Boundary

这些 artifact 已落盘；未满足 gate 的阶段明确为 `not_run`：

```text
p6_official_signal_value_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_adamw_only_fullpass_repair.csv
p10_robustness_external_ready.csv
```

## 7. No-fake audit

```text
rows_checked = 524
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 8. Hash

| artifact | SHA256 |
|---|---|
| runner | `4faa1c4554f0feb10ffcc8e2dc3eaafe36eaf9ce34914b437611af921e9e3efc` |
| route | `e3640ca57d04c6b8207f4708448e2b6d9f4fd8e807309e352aed8802652da12f` |
| P1 grounding | `d05677709f3acd6770e33f36eae4bb7e8fc61756e7618a5a9c1f92ad8e6443e8` |
| P2 relevance | `dd3a3297ed5a162e60647270f06b7e1872a6048aa1052883912a6b8b5abcf8a1` |
| P3 reclassification | `a746070e8e9b53f406915667c5d11a9393afb26e7174ae839c986523ed70a643` |
| P4 predictor | `e17d82ba33261175de2a9dac419be12c2689d61da51031a0619cc445a479fed5` |
| P5 LDO/LSO | `bf4960f954f7aa3509cb9482d5b39c86c565c1d0c96c67ac9743759a3a5e57df` |
| provenance audit | `6d6a4e3152d032bd7e00f81c1d67287af7afdd5fee61cdcdf5352ebf3768b87b` |

## 9. 最终分析结论

v9.2.31 的真实推进是：

```text
v9.2.30: pre-event movement feature 接近阈值，但旧 label / attribution / predictor 失败。
v9.2.31: 先把 event value grounding、stratum baseline、observability 和 predictor gate 分离审计。
```

机制判断：

1. 本轮不能说 primitive 已经可用于 official functional route；official paired replay 仍取决于 P4/P5 gate。
2. 如果 predictor 或 LDO/LSO 没过，正确结论是 event-value controller 尚不稳定，而不是倒回 dataset-specific Fashion/KMNIST patch。
3. 如果 grounding/observability 过而 controller 不过，下一步应补充真实 pre-event movement features、repeat seeds/h640，或重做 primitive observability，而不是编造 missing curvature/control-ECE。

最终一句话：

> v9.2.31 真实执行后停在 `R9-PrimitiveEffectUnpredictable`：`pre_event_features_do_not_predict_grounded_value`。
