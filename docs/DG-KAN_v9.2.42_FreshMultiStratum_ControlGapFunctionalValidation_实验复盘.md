# DG-KAN v9.2.42 Fresh Multi-Stratum Control-Gap Functional Validation 实验复盘

> 本复盘记录 `DG-KAN_v9.2.42_FreshMultiStratum_ControlGapFunctionalValidation_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R5-OracleHighLegalScoreLow
base_candidate = LQ-t2-h256
success_v9242_strict_purekan_functional = False
success_v9242_full_functional = False
success_v9242_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9242_fresh_multistratum_controlgap_functional_validation_first_20260511T143000Z
```

核心结论：

1. P0 复现 v9.2.41 boundary：source route = `R3-ControlGapScorePass`，source blocker = `leave_dataset_or_stratum_out_failed_after_value_score_pass`。
2. P1 完成 fresh multi-stratum measurement：fresh rows = `8640`，fresh real events = `1440`，signal strata = `8`，attach candidates = `3`。
3. Frozen S7 在 fresh rows 上接近但未过 gate：AUC `0.752942`，corr `0.338265`，precision `0.906977`，bad-event `0.0`，但 coverage `0.029861 < 0.03`，因此 `frozen_s7_pass = 0`。
4. P3 parallel score matrix 没有 legal score 通过 official value gate；唯一 pass 的 `S13-Oracle` 是 posthoc/oracle，不能进入 official route。
5. P4 rolewise attach 仍可动：`A3-LateAttachRoleWiseFT7EdgeCarrier` carrier pass = `1`，但 rolewise value pass = `0`。
6. P5 leave-out 与 P6 official paired replay 因 `no_legal_value_score_survivor` 没有打开。
7. 当前 blocker：`oracle_good_events_exist_but_legal_scores_failed`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9242_fresh_multistratum_controlgap_functional_validation.py` | v9.2.42 runner；生成 fresh multi-stratum rows、frozen S7、score matrix、rolewise attach、leave-out/paired replay boundary、route、no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9242_fresh_multistratum_controlgap_functional_validation.py
```

正式运行：

```bash
python experiments/run_v9242_fresh_multistratum_controlgap_functional_validation.py \
  --out-dir results/real_rerun_20260506/v9242_fresh_multistratum_controlgap_functional_validation_first_20260511T143000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际测量范围：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
signal_strata = S1..S8
attach_candidates = A1,A2,A3
branches = RealFunctional,AdamWOnly,AdamWParallel,bestLR,NoOp,Random
fresh_row_count = 8640
fresh_real_event_count = 1440
```

说明：本轮不是复用 v9.2.40 source rows 直接打分，而是在 repaired base + snapshot attach 路径上重新生成 fresh multi-stratum replay rows。

## 2. Route

`route_decision.json`：

```json
{
  "route": "R5-OracleHighLegalScoreLow",
  "base_candidate": "LQ-t2-h256",
  "v9241_boundary_pass": 1,
  "fresh_row_count": 8640,
  "fresh_real_event_count": 1440,
  "measured_signal_strata_count": 8,
  "accepted_signal_strata_count": 8,
  "frozen_s7_pass": 0,
  "best_score_id": "S7-FrozenHybridMonotoneLegal",
  "best_attach_candidate": "A3-LateAttachRoleWiseFT7EdgeCarrier",
  "value_observability_pass": 0,
  "value_auc": 0.7529422075320513,
  "value_corr": 0.3382654913723068,
  "accepted_precision": 0.9069767441860465,
  "accepted_coverage": 0.029861111111111113,
  "accepted_bad_event_rate": 0.0,
  "oracle_upper_bound_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.12013888888888889,
  "rolewise_attach_pass": 1,
  "rolewise_value_pass": 0,
  "leave_dataset_out_pass": 0,
  "leave_stratum_out_pass": 0,
  "paired_replay_pass": 0,
  "primary_blocker": "oracle_good_events_exist_but_legal_scores_failed",
  "success_v9242_strict_purekan_functional": 0,
  "success_v9242_full_functional": 0,
  "success_v9242_external_ready": 0
}
```

判断：good events 的 oracle upper bound 真实存在，但 legal score 没有通过 fresh official value gate，所以不能打开 leave-out / paired replay。

## 3. P1 fresh multi-stratum expansion

Artifact：

```text
p1_fresh_multistratum_event_expansion.csv
fresh_event_trace_v9242.csv
```

Summary：

```text
fresh_row_count = 8640
fresh_real_event_count = 1440
measured_signal_strata_count = 8
attach_candidate_count = 3
max_r_z_tail = 0.14670663329121736
max_r_perp_tail = 0.12849250196738116
bad_event_rate = 0.03611111111111111
carrier_remains_active = 1
fresh_measurement_pass = 1
```

Signal strata：

```text
S1-CEHardTail
S2-MarginTail
S3-ControlGapPositive
S4-RoleWiseCurvature
S5-UncertaintyLCB
S6-FamilyValueReliable
S7-HighDerivativeBranch
S8-OrthogonalTailNonAdamW
```

判断：P1 满足 fresh measurement 的覆盖要求；这一步只证明 fresh rows 与 carrier signal 真实存在，不证明 legal value score 或 functional causality。

## 4. P2 frozen S7 confirmation

Artifact：

```text
p2_frozen_s7_confirmation.csv
frozen_s7_trace_v9242.csv
```

| metric | value | gate |
|---|---:|---|
| AUC | `0.752942` | pass |
| corr | `0.338265` | below `0.35` but AUC passes shape gate |
| precision | `0.906977` | pass |
| coverage | `0.029861` | fail, below `0.03` |
| bad-event rate | `0.000000` | pass |
| accepted events | `43` | diagnostic |
| frozen S7 pass | `0` | fail |

判断：Frozen S7 的 fresh 结果不是无信号；主要失败点是 coverage 刚好低于 official lower bound，且 corr 未达 `0.35`。因此不能写成 `R1-FrozenS7FreshPass`。

## 5. P3 parallel score matrix

Artifact：

```text
p3_parallel_score_matrix.csv
score_matrix_trace_v9242.csv
```

| score | legal | AUC | corr | precision | coverage | bad event | official pass | reason |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S7-FrozenHybridMonotoneLegal | `1` | `0.752942` | `0.338265` | `0.906977` | `0.029861` | `0.000000` | `0` | coverage below gate |
| S9-ControlGapLCB | `1` | `0.241244` | `-0.339837` | `0.000000` | `0.150000` | measured | `0` | wrong direction |
| S10-RoleWiseControlGap | `1` | `0.542773` | `-0.182783` | `0.148148` | `0.150000` | `0.041667` | `0` | weak precision |
| S11-FamilyReliabilityGap | `0` | `0.665498` | `0.295743` | `0.482759` | `0.040278` | measured | `0` | posthoc family calibration |
| S12-HybridMonotoneFreshPreRegistered | `1` | `0.427901` | `-0.253597` | `0.115741` | `0.150000` | measured | `0` | weak / wrong direction |
| S13-Oracle | `0` | `1.000000` | `1.000000` | `1.000000` | `0.120139` | `0.000000` | `1` | oracle/posthoc only |

判断：

1. `S13-Oracle` 证明 fresh rows 中确实存在可挑出的 good events。
2. 但所有 legal score 都没有 official pass；因此本轮 route 是 `R5-OracleHighLegalScoreLow`，不是 legal score success。

## 6. P4 rolewise late-attach fresh validation

Artifact：

```text
p4_rolewise_late_attach_fresh_validation.csv
rolewise_attach_trace_v9242.csv
```

| attach | carrier pass | value AUC | value corr | precision | coverage | r_z tail | r_perp tail | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1-LateAttachZeroLinearTail | `0` | `0.971454` | `0.512337` | `1.000000` | `0.029167` | `0.106254` | `0.033518` | `0` |
| A2-LateAttachControlGapChannel | `0` | `0.972806` | `0.476274` | `0.928571` | `0.029167` | `0.084223` | `0.025276` | `0` |
| A3-LateAttachRoleWiseFT7EdgeCarrier | `1` | `0.886907` | `0.880210` | `0.486111` | `0.150000` | `0.146707` | `0.128493` | `0` |

判断：A3 保持 rolewise carrier movement，但 value precision 不足；A1/A2 value score 很强但 carrier gate 不过，不能作为 rolewise functional attach survivor。

## 7. P5/P6 downstream boundary

Artifacts：

```text
p5_leave_dataset_and_stratum_out.csv
p6_official_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_robustness_external_ready.csv
```

Boundary：

```text
P5 status = not_run
P5 reason = no_legal_value_score_survivor
P6 status = not_run
P6 reason = P5_leave_dataset_or_stratum_out_failed
P7-P9 status = not_run
```

判断：本轮没有把 oracle 或 diagnostic score 倒灌成 leave-out / paired replay success。由于 P3 没有 legal survivor，P5 之后全部 gate-blocked。

## 8. No-fake audit

`v9242_provenance_audit.csv`：

```text
rows_checked = 18738
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
| plan | `8b6adca34ee1a8ab56264404ed70c3cbecb59785feb7056f8e16728e771029a3` |
| runner | `e4e28f19e2a2984582b0cb9dd8f92df4fc68ccee6e042865d2f3fe7b60ffb08d` |
| run manifest | `9b6af40a1a962ea23bdfbe474d9b8a63e9aa6ae004d8262e1d37b30bc2b643bd` |
| route | `da69db6c66dc51c831f59c83bf03a44e6a1e00019c15753eaaced18096a2159a` |
| P1 fresh expansion | `c95af536d3ab4ecee4af51dc12575e3d10c394700c1c095564bd2140b07f873e` |
| P2 frozen S7 | `027284a86093746336de802877f6daaf052da33410ca2635737dacd580ff0495` |
| P3 score matrix | `2d82457fe5a715b671f611c59c0494a5886711bbb25e1e700871434718858017` |
| P4 rolewise attach | `de0a3e6c3431e9dcdcbce18ffdcefc62215ce5d0a344de11d3c5b7e050af7027` |
| P5 leave-out boundary | `9730f14c5739e849ac97d9b3c06fa48f5852042c15fc358188a606c368d18f95` |
| P6 paired replay boundary | `5a65f4ff895c603a82ed1fbbdfb4a145a994ea180ae9918088024f96c56c7708` |
| P7 short-run boundary | `3ceedc05f9f8f90baca6f28577b3fa2016833e5779a2c6bf62f60af8e579d6e8` |
| P8 full-run boundary | `1bfc8ce7cfc0cdf58984de6fcfd00602d4209dc8a04fad7946f97a54fd9966f6` |
| P9 robustness boundary | `78caf579bbbde77026e799cca20eab1c645c80c7d79741b6d49aab01302ae067` |
| failure table | `ad4704301ca94ba7f4d3b3df7e1b872cdc888c333c285cbb06a477d9f5b49251` |
| provenance audit | `cabaf16578e60248f03f582debb6a07c10c9ab8f2bbd4ce4de3b379dfd48abcb` |

## 10. 最终分析结论

v9.2.42 的真实推进是：

```text
v9.2.41: S7 在 source matrix 上过 value gate，但 leave-out failed。
v9.2.42: 重新生成 fresh multi-stratum rows，验证 frozen S7 / new legal scores / oracle upper bound。
```

机制判断：

1. Fresh multi-stratum carrier 不是空的：`fresh_real_event_count = 1440`，8 个 strata 均有测量，A3 rolewise carrier 仍然 active。
2. Frozen S7 不是完全失效，但差一个严格 gate：coverage `0.029861` 低于 `0.03`，因此不能升级为 fresh pass。
3. Oracle upper bound 通过，说明 good events 存在；失败点是 legal score calibration，而不是 carrier 完全没有可用事件。
4. 由于没有 legal value score survivor，leave-out / official paired replay / short-run / full-run / external-ready 都不能打开。
5. 下一步应重做 legal score calibration，目标是在不使用 oracle/posthoc outcome 的情况下把 coverage 和 leave-out 稳定性一起补上。

最终一句话：

> v9.2.42 真实执行后停在 `R5-OracleHighLegalScoreLow`：fresh multi-stratum rows 中存在 oracle-good events，但 legal score 没有通过 official value gate，strict PureKAN functional 仍未成功。
