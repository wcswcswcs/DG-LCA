# DG-KAN v9.2.60 Clean-Core Expansion 与 Constraint-Calibrated Support Frontier 实验复盘

> 本复盘记录 `DG-KAN_v9.2.60_CleanCoreExpansion_ConstraintCalibratedSupportFrontier_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R12-CleanCoreNotExpandable
base_candidate = LQ-t2-h256
success_v9260_strict_purekan_functional = False
success_v9260_full_functional = False
success_v9260_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9260_clean_core_expansion_constraint_calibrated_support_frontier_first_20260512T133000Z/
```

核心结论：

1. P0 复现 v9.2.59 boundary：source route = `R11-RiskSupportStatsStillTinyCleanCore`，oracle support pass = `1`，fake/proxy/offload = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；运行中 GPU 采样有活动：`NVIDIA L4, 32-33 %, 326 MiB`。
3. P1 clean-core 不稳定：core precision = `0.815789`，coverage = `0.007540`，bad-event = `0.085526 > 0.05`，因此 `clean_core_stable = 0`。
4. P1 near-core 也不能扩张：near-core oracle overlap = `0.134910`，bad-event = `0.155563`，`nearcore_expandable = 0`。
5. P2 support measurement 扩到 natural rows = `20160`，family = `606`，duplicate = `0`；但 balanced diagnostic rows = `591 < 6000`，signal strata = `3 < 6`，`support_measurement_pass = 0`。
6. P3 risk-budget frontier 有 diagnostic：best = `RB3-TailFirstSoftBudget`，coverage = `0.051984`，bad-event = `0.043893`，但 precision = `0.186069`，所以 official pass = `0`。
7. P4 support frontier 未过：best = `SF4-DensityRiskJointPocket`，precision = `0.960784`，bad-event = `0.0`，但 coverage = `0.002530`，strata = `1`，diagnostic/pass 均为 `0`。
8. P5 exact reference deployable frontier 未过：best = `C-RB4-ControlGapConditionalRisk+SF4-DensityRiskJointPocket`，precision = `0.956522`，bad-event = `0.014493`，但 coverage = `0.003423`，bad-event UCB = `0.077633`，strata = `1`。
9. P6 true-delta compute v5 仍未过：best = `TBD0-V9256CBD0Reference`，AUC = `0.879914`，agreement = `1.0`，step ratio = `2.863280 > 1.50`。
10. P8-P10 因 `P1_clean_core_not_stable` gate-blocked，全部以 `not_run` 落盘。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9260_clean_core_expansion_constraint_calibrated_support_frontier.py` | v9.2.60 runner；执行 v9.2.59 boundary 复现、clean-core/near-core autopsy、support measurement expansion、risk-budget frontier、support frontier、exact-reference deployable frontier、true-delta compute lane、downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9260_clean_core_expansion_constraint_calibrated_support_frontier.py
```

正式运行：

```bash
python experiments/run_v9260_clean_core_expansion_constraint_calibrated_support_frontier.py \
  --out-dir results/real_rerun_20260506/v9260_clean_core_expansion_constraint_calibrated_support_frontier_first_20260512T133000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
triton_available = true
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 280
interface_events = 24
interface_steps = 2
train_size = 2048
batch_size = 64
hidden_dim = 256
completed_at = 2026-05-12T14:14:43Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R12-CleanCoreNotExpandable",
  "base_candidate": "LQ-t2-h256",
  "v9259_boundary_pass": 1,
  "clean_core_stable": 0,
  "core_precision": 0.8157894736842105,
  "core_coverage": 0.00753968253968254,
  "core_bad_event": 0.08552631578947369,
  "nearcore_expandable": 0,
  "nearcore_oracle_overlap": 0.13491005996002664,
  "support_measurement_pass": 0,
  "natural_real_event_count": 20160,
  "balanced_diagnostic_real_event_count": 591,
  "measured_signal_strata_count": 3,
  "measured_family_count": 606,
  "best_risk_budget_id": "RB3-TailFirstSoftBudget",
  "risk_budget_pass": 0,
  "risk_budget_precision": 0.18606870229007633,
  "risk_budget_coverage": 0.05198412698412699,
  "risk_budget_bad_event": 0.04389312977099236,
  "best_support_frontier_id": "SF4-DensityRiskJointPocket",
  "support_frontier_pass": 0,
  "support_frontier_diagnostic_pass": 0,
  "exact_reference_deployable": 0,
  "reference_controller_precision": 0.9565217391304348,
  "reference_controller_coverage": 0.0034226190476190476,
  "reference_controller_bad_event": 0.014492753623188406,
  "reference_precision_lcb": 0.8797860261535665,
  "reference_bad_event_ucb": 0.07763307335057362,
  "true_delta_compute_pass": 0,
  "true_delta_auc": 0.8799141087036113,
  "true_delta_agreement": 1.0,
  "true_delta_step_ratio_q90": 2.8632798851361203,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.11997767857142858,
  "oracle_bad_event": 0.0,
  "primary_blocker": "clean_core_not_stable",
  "next_required_implementation": "recalibrate_clean_core_definition"
}
```

判断：v9.2.60 没有把 v9.2.59 的 tiny clean core 成功扩大。第一 terminal gate 已经是 P1：clean core 的 bad-event 超过 `0.05`，near-core 也没有足够 oracle overlap；后续 P3 的 diagnostic 不能倒灌成 official route。

## 3. P1 clean-core / near-core autopsy

Artifacts：

```text
p1_clean_core_nearcore_autopsy.csv
clean_core_trace_v9260.csv
nearcore_expansion_trace_v9260.csv
```

Summary：

```text
core_count = 152
core_precision = 0.815789
core_coverage = 0.007540
core_bad_event = 0.085526
near_core_count = 3002
near_core_oracle_overlap = 0.134910
near_core_safe_good_density = 0.134910
near_core_bad_event = 0.155563
global_safe_good_density = 0.130060
clean_core_stable = 0
nearcore_expandable = 0
```

判断：clean core 本身没有满足 bad-event gate，near-core 的 safe-good density 也没有比全局高出计划要求的 `+0.10`。因此 route 正确停在 `R12-CleanCoreNotExpandable`。

## 4. P2 support measurement expansion

Artifact：

```text
p2_support_measurement_expansion.csv
support_density_expansion_trace_v9260.csv
```

Summary：

```text
natural_real_event_count = 20160
balanced_diagnostic_real_event_count = 591
measured_signal_strata_count = 3
measured_family_count = 606
duplicate_row_count = 0
support_measurement_pass = 0
```

判断：natural rows 已达到 `>=20000`，family count 也足够；但 balanced diagnostic rows 和 signal strata 仍不足。没有复制或补造 balanced rows。

## 5. P3 risk-budget frontier

Artifacts：

```text
p3_risk_budget_frontier.csv
risk_budget_trace_v9260.csv
```

| risk budget | precision | coverage | bad-event | pass | diagnostic |
|---|---:|---:|---:|---:|---:|
| RB0-HardUnionReference | `0.806452` | `0.001538` | `0.000000` | `0` | `0` |
| RB1-CalibratedBadProbability | `0.937500` | `0.000794` | `0.000000` | `0` | `0` |
| RB2-ModeWeightedRiskBudget | `0.937500` | `0.000794` | `0.000000` | `0` | `0` |
| RB3-TailFirstSoftBudget | `0.186069` | `0.051984` | `0.043893` | `0` | `1` |
| RB4-ControlGapConditionalRisk | `0.913043` | `0.001141` | `0.000000` | `0` | `0` |
| RB5-CleanCoreExpansionRisk | `1.000000` | `0.000893` | `0.000000` | `0` | `0` |

判断：`RB3` 达到 diagnostic coverage/bad-event，但 precision 太低，不能作为 official risk-budget frontier。其余风险预算虽然更干净，但仍是 tiny clean slice。

## 6. P4 support frontier

Artifacts：

```text
p4_support_frontier_expansion.csv
support_frontier_trace_v9260.csv
```

| support frontier | precision | coverage | bad-event | Jaccard | strata | families | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| SF0-V9259SupportReference | `0.937500` | `0.000794` | `0.000000` | `0.015873` | `1` | `10` | `0` |
| SF1-MultiResolutionFamilyReliability | `0.937500` | `0.000794` | `0.000000` | `0.015873` | `1` | `10` | `0` |
| SF2-CoreNeighborFamilyExpansion | `0.937500` | `0.000794` | `0.000000` | `0.015873` | `1` | `10` | `0` |
| SF3-LeaveFamilyOutStabilityV3 | `0.937500` | `0.000794` | `0.000000` | `0.015873` | `1` | `10` | `0` |
| SF4-DensityRiskJointPocket | `0.960784` | `0.002530` | `0.000000` | `0.051797` | `1` | `18` | `0` |
| SF5-BalancedExpansionCap | `0.937500` | `0.000794` | `0.000000` | `0.015873` | `1` | `10` | `0` |

判断：support frontier 仍是 clean-too-tiny；最佳 `SF4` coverage 只有 `0.002530`，signal strata 只有 `1`，不能写成 support expansion success。

## 7. P5 exact reference deployable frontier v4

Artifacts：

```text
p5_exact_reference_deployable_frontier_v4.csv
reference_deployable_frontier_trace_v9260.csv
```

Best summary：

```text
best_reference_controller_id = C-RB4-ControlGapConditionalRisk+SF4-DensityRiskJointPocket
best_risk_budget_id = RB4-ControlGapConditionalRisk
best_support_frontier_id = SF4-DensityRiskJointPocket
exact_reference_deployable = 0
precision = 0.956522
coverage = 0.003423
bad_event = 0.014493
precision_lcb = 0.879786
bad_event_ucb = 0.077633
accepted_signal_strata_count = 1
accepted_family_count = 22
max_family_share = 0.188406
legal_oracle_jaccard = 0.069694
```

判断：best frontier 的 point estimate 看起来安全，但 coverage 低于 `0.03`，bad-event UCB 高于 `0.05`，accepted strata 也只有 `1`。因此 exact reference deployable frontier 仍未闭合。

## 8. P6 true-delta compute v5

Artifacts：

```text
p6_true_delta_compute_v5_parallel_lane.csv
true_delta_compute_v5_trace.csv
```

Summary：

```text
best_true_delta_id = TBD0-V9256CBD0Reference
true_delta_auc = 0.879914
true_delta_agreement = 1.0
true_delta_step_ratio_q90 = 2.863280
true_delta_memory_ratio = 0.969501
true_delta_compute_pass = 0
```

判断：exact/true delta signal 仍存在，但 compute path 继续超过 system gate。即使 P1/P5 没有失败，P6 也不能打开 official controller。

## 9. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_system_legal_exact_signal_controller.csv` | `P5_reference_deployable_frontier_failed` |
| `p8_leave_dataset_and_stratum_out.csv` | `P1_clean_core_not_stable` |
| `p9_official_paired_replay.csv` | `P1_clean_core_not_stable` |
| `p10_short_run_functional_validation.csv` | `P1_clean_core_not_stable` |

没有把 risk-budget diagnostic、clean-but-tiny support frontier、exact reference point estimate 或 oracle support 写成 system-legal controller / LDO / paired replay / short-run success。

## 10. No-fake audit

```text
rows_checked = 102121
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
clean_core_expansion = 1
risk_budget_frontier = 1
support_frontier_expansion = 1
exact_reference_deployable_frontier_v4 = 1
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `7870d19dfcbfaff1c4e004944679f0b59c2deebd724084c1e92db8d53138ed15` |
| runner | `2fa05072e1f38ba985ddb9ca0e531c5fa2c6c912a92d8d9e2dc814b52a9b5699` |
| run manifest | `919289a17e4971a3fc3295a5ce5104a5b4286aa2a41518d892a7a26a898a355a` |
| route | `1f8aa53affd0805f2371001d51230bfaf349ae12ffd993c3e0ffe277e0b2d14d` |
| P0 boundary | `b6bd5d43022016bf3ea9b0031b4cb10ea938baf7a35646e8fb476354fdb81e35` |
| P1 clean core | `ee7e8af1c2046b90791e5ddec2c81e065dffdacb8679525b09d07be8540c2ea5` |
| P2 support measurement | `404fd226fe9ac30f983a4d3650f2a4c7bc0496559f209d6b80e02c6210db94bc` |
| P3 risk budget | `e3e686b16199a59a43b7d4cbacebbb6061190c0f9051e94cecf0dbb073e3f999` |
| P4 support frontier | `757a46209f7d6cc74ffd3f84d213e0667b9ede601e4118c989027565dc74a839` |
| P5 deployable frontier | `76f321338e0e45f1d2e5c0e9b0f2fbbd472c59a9575c61a6c154220cfdf59a26` |
| P6 compute lane | `2f3288c5cc356361642067d070aeb73bfadb24b37ccc5b5f942376db4a6d66c5` |
| P7 system controller | `b02e624c8b00d27ac4816397f267440b9a714156325cc52718e34d8bd7b49d33` |
| P8 boundary | `29ad8a78db7759918850cf98ad8b87adf725e80550baaa161e9a74793fa1eadb` |
| P9 boundary | `8cd9e35f68f346482049393c253513f912fe5bea2f6058e286fba9c138e1e6cd` |
| P10 boundary | `156591477a857b050618e6527be2a026763aadcca3eb0096256fb4f6d5574a1f` |
| failure table | `a940fa934534f4dd837b5038ab8cb8e3c0f4e89d9432ec1f2cfee8bcea9c212f` |
| provenance audit | `d8f51a0df53aa681be0e77401b77127020a6eb7807f8d59a143f64066691c836` |

## 12. 最终分析结论

v9.2.60 的真实推进是：

```text
v9.2.59: oracle-decomposed submode risk 有信号，但组合 risk/support 仍 tiny。
v9.2.60: 尝试 clean-core expansion、risk budget frontier 与 support frontier；
          但 clean core 本身 bad-event 过高，near-core 也不能扩张。
```

机制判断：

1. H1 不成立：clean core 没有稳定，bad-event `0.085526` 高于 `0.05`。
2. H2 部分成立：risk-budget frontier 能得到 bad-event `0.043893`、coverage `0.051984` 的 diagnostic region，但 precision 只有 `0.186069`。
3. H3 未成立：support frontier 最好仍只有 coverage `0.002530`，accepted strata 只有 `1`。
4. H4 未成立：support measurement natural rows 足够，但 balanced diagnostic rows 和 measured signal strata 不足。
5. H5 未成立：exact reference deployable frontier 的 point estimate 有推进，但 coverage、UCB 和 strata gate 都不过。
6. H6 继续未闭合：true-delta compute 仍 `step_ratio_q90 = 2.863280`。

最终一句话：

> v9.2.60 真实执行后停在 `R12-CleanCoreNotExpandable`：risk/support frontier 有局部诊断推进，但 clean core 本身不稳定，near-core 不能扩张，strict PureKAN functional 仍未成功。
