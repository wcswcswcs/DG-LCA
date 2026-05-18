# DG-KAN v9.2.43 Legal Controller Calibration 与 Paired Replay Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.43_LegalControllerCalibration_PairedReplayClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R7-OracleLowCarrierMechanismReset
base_candidate = LQ-t2-h256
success_v9243_strict_purekan_functional = False
success_v9243_full_functional = False
success_v9243_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9243_legal_controller_calibration_paired_replay_closure_first_20260511T153000Z
```

核心结论：

1. P0 复现 v9.2.42 boundary：source route = `R5-OracleHighLegalScoreLow`，frozen S7 coverage fail，oracle pass，fake/proxy = `0`。
2. P1 S7 coverage cliff autopsy 通过：`s7_coverage_cliff_mode = G1-threshold_borderline`。在 source v9.2.42 rows 上存在 threshold-free point coverage `0.03125`、precision `0.911111`、bad-event `0.0`。
3. P2 fresh multi-stratum v2 的行数覆盖达标：fresh rows = `18432`，fresh real events = `3072`，strata = `8`，attach = `4`。
4. 但 P2 fresh v2 safety 未过：bad-event rate = `0.07682291666666667 > 0.05`，因此 `fresh_v2_pass = 0`。
5. P3 legal controller calibration matrix 已执行；best legal controller = `C1-CalibratedS7Threshold`，heldout corr `0.391692` 过 shape，但 heldout precision `0.544118`、bad-event `0.455882`，official pass = `0`。
6. P3 oracle 在 fresh v2 heldout 上也不再过：oracle precision `0.375`，coverage `0.083333`，bad-event `0.625`，oracle upper bound pass = `0`。
7. P4 leave-out、P5 official paired replay、P6-P8 均因没有 legal controller survivor / leaveout gate 未打开而明确 `not_run`。
8. 当前 blocker：`fresh_v2_bad_event_rate_above_gate`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9243_legal_controller_calibration_paired_replay_closure.py` | v9.2.43 runner；执行 v9.2.42 boundary reproduction、S7 coverage cliff、fresh v2 expansion、controller calibration matrix、leave-out / paired replay boundary、route、no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9243_legal_controller_calibration_paired_replay_closure.py
```

正式运行：

```bash
python experiments/run_v9243_legal_controller_calibration_paired_replay_closure.py \
  --out-dir results/real_rerun_20260506/v9243_legal_controller_calibration_paired_replay_closure_first_20260511T153000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际测量范围：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
horizons = 20,80,240,640
signal_strata = S1..S8
attach_candidates = A1,A2,A3,A5
branches = RealFunctional,AdamWOnly,AdamWParallel,bestLR,NoOp,Random
calibration split = seed 0,1,2,3
heldout split = seed 4,5,6,7
```

说明：本轮没有使用 dataset name 做 official controller 分支；calibration threshold 只在 calibration split 上选择，heldout split 才计入 official gate。

## 2. Route

`route_decision.json`：

```json
{
  "route": "R7-OracleLowCarrierMechanismReset",
  "base_candidate": "LQ-t2-h256",
  "v9242_boundary_pass": 1,
  "dataset_tuning_detected": 0,
  "s7_coverage_cliff_mode": "G1-threshold_borderline",
  "fresh_real_event_count": 3072,
  "fresh_row_count": 18432,
  "measured_signal_strata_count": 8,
  "attach_candidates_measured": 4,
  "best_controller_id": "C1-CalibratedS7Threshold",
  "controller_value_pass": 0,
  "controller_auc": 0.6552013703208556,
  "controller_corr": 0.3916916905783157,
  "accepted_precision": 0.5441176470588235,
  "accepted_coverage": 0.044270833333333336,
  "accepted_bad_event_rate": 0.45588235294117646,
  "accepted_strata_count": 8,
  "accepted_family_count": 29,
  "max_family_share": 0.058823529411764705,
  "oracle_upper_bound_pass": 0,
  "oracle_precision": 0.375,
  "oracle_coverage": 0.08333333333333333,
  "oracle_auc": 0.9572192513368984,
  "leave_dataset_out_pass": 0,
  "leave_stratum_out_pass": 0,
  "paired_replay_pass": 0,
  "primary_blocker": "fresh_v2_bad_event_rate_above_gate",
  "success_v9243_strict_purekan_functional": 0,
  "success_v9243_full_functional": 0,
  "success_v9243_external_ready": 0
}
```

判断：v9.2.42 的 source S7 borderline 是真实现象；但 fresh v2 扩展后，当前 carrier/controller 的 safety 和 oracle upper bound 都没有保留，因此 route 进入 carrier/mechanism reset 分支，而不是 controller pass。

## 3. P1 S7 coverage cliff autopsy

Artifacts：

```text
p1_s7_coverage_cliff_autopsy.csv
s7_pr_curve_trace_v9243.csv
```

Summary：

```text
s7_coverage_cliff_mode = G1-threshold_borderline
threshold_free_borderline_gate_exists = 1
AUC = 0.7529422075320513
corr = 0.3382654913723068
best_precision = 0.9111111111111111
best_coverage = 0.03125
best_bad_event_rate = 0.0
accepted_strata_count = 8
accepted_family_count = 19
max_family_share = 0.13333333333333333
p1_pass = 1
```

判断：v9.2.42 的 S7 failure 不是排序完全失败，也不是 family concentration 失败；在 source rows 的 threshold-free 曲线上，coverage gate 附近确实存在可达点。但这只是 source diagnostic，不能替代 calibration-heldout official gate。

## 4. P2 fresh multi-stratum expansion v2

Artifacts：

```text
p2_fresh_multistratum_expansion_v2.csv
fresh_event_trace_v9243.csv
```

Summary：

```text
fresh_row_count = 18432
fresh_real_event_count = 3072
measured_signal_strata_count = 8
attach_candidates_measured = 4
max_r_z_tail = 0.17148524302036527
max_r_perp_tail = 0.1371679780907177
bad_event_rate = 0.07682291666666667
carrier_remains_active = 1
fresh_v2_pass = 0
```

判断：

1. P2 的 coverage 不再是问题：row count、real event count、strata count、attach count 都满足计划要求。
2. 真实 failure 是 safety：bad-event rate 超过 `0.05`。
3. Carrier 仍然 active，但 active 不等于可用；这轮 fresh v2 说明 current carrier 的 expanded event space 中坏事件比例过高。

## 5. P3 legal controller calibration matrix

Artifacts：

```text
p3_legal_controller_calibration_matrix.csv
controller_calibration_trace_v9243.csv
oracle_legal_gap_trace_v9243.csv
```

Heldout controller summary：

| controller | AUC | corr | precision | coverage | bad event | strata | families | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0-FrozenS7Reference | `0.655201` | `0.391692` | `0.531915` | `0.030599` | `0.468085` | `8` | `23` | `0` |
| C1-CalibratedS7Threshold | `0.655201` | `0.391692` | `0.544118` | `0.044271` | `0.455882` | `8` | `29` | `0` |
| C2-TwoStageS7FamilyReliability | `0.655201` | `0.391692` | `0.517241` | `0.037760` | `0.482759` | `8` | `27` | `0` |
| C3-ControlGapLCBCalibrated | `0.307462` | `-0.391184` | `0.000000` | `0.173828` | `0.116105` | `8` | `40` | `0` |
| C4-RoleGapCalibrated | `0.498354` | `-0.253110` | `0.100917` | `0.212891` | `0.229358` | `8` | `39` | `0` |
| C5-FamilyReliabilityGap | `0.507320` | `0.224217` | `0.000000` | `0.013021` | `0.200000` | `5` | `10` | `0` |
| C6-HybridMonotoneCalibrated | `0.638473` | `0.392937` | `0.260000` | `0.032552` | `0.380000` | `8` | `28` | `0` |
| C7-Oracle | `0.957219` | `1.000000` | `0.375000` | `0.083333` | `0.625000` | `8` | `64` | `0` |

判断：

1. C1/C2/C6 在 heldout 上仍有 shape signal，但 precision 与 bad-event gate 全部失败。
2. C1 是 best legal controller，但 heldout precision 只有 `0.544118`，bad-event `0.455882`，不能进入 leave-out。
3. C7 oracle 是 posthoc diagnostic，且本轮也未过 accept gate：oracle precision `0.375`，bad-event `0.625`。
4. 因 oracle 在 fresh v2 heldout 上也失败，本轮不能继续解释为“good events 存在但 legal score 找不到”；更直接的结论是 expanded carrier/event space 本身不够 robust。

## 6. P4-P8 downstream boundary

Artifacts：

```text
p4_leave_dataset_and_stratum_out.csv
p5_official_paired_replay.csv
p6_short_run_functional_validation.csv
p7_full_10seed_functional_validation.csv
p8_robustness_external_ready.csv
```

Boundary：

```text
P4 status = not_run
P4 reason = no_legal_controller_survivor
P5 status = not_run
P5 reason = P4_leaveout_failed
P6-P8 status = not_run
```

判断：P4 只有在 legal controller survivor 后才允许打开；P5 official paired replay 只有在 leave-out pass 后才允许打开。本轮没有把 P3 diagnostic rows 或 oracle rows 倒灌成 downstream success。

## 7. No-fake audit

`v9243_provenance_audit.csv`：

```text
rows_checked = 30751
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
| plan | `ada80671b49a63790718e3aa72cd4c44c2e16269a1b5ee7f2950aaee8368a4f8` |
| runner | `0914afda79e3e799e94a164722ddf00fce980655929c78c1d7414e3613cfab1d` |
| run manifest | `8160fe31aeb1f109b04931f57ac4a8d7998cddd9f24c477980cb8469eb7befd6` |
| route | `d0de8b85198a7ba62af3f698b88f21d9b1930ffaf9fe68ed35b990f940ba6343` |
| P0 boundary | `223ab2ffc8435237e75b6cac4762b3432defe734f8eabff5dc9492dde0a51a57` |
| P1 coverage cliff | `0f79da2c1c857d94ef144110c2b28a9cea378cc4e5b04047f3176ec1549b1f26` |
| P2 fresh v2 | `322cdf81f5fc6c5b0ff7d56aad016d6d896573e8b2172a0cdfc20019a8440e93` |
| P3 controller matrix | `f89a959ac014b6efb078e3af4bac6827ff2b29e11b7dec7c88556c79d5e6a14c` |
| P4 leave-out boundary | `e9b459134d85ac1d390ac41e72ed9e77b4d54f4aee22bc6fb8d414b4182ef5b8` |
| P5 paired replay boundary | `c85bc7f6be5756fa8f5bafee0b657d0f34c9772731aa9e4128e73fcab470eb3a` |
| P6 short-run boundary | `a57795048817d9bb763324c9cae1bb627f040e5632fef15453deea6fba81e225` |
| P7 full-run boundary | `61d4689dda270d685c5fcc314124f579f6b55f1d99f7c1aaa505cfa950f18e5f` |
| P8 robustness boundary | `25bececdc2ec4661697a04e23e82222ec86b443202e8d1c653898cdc45097680` |
| provenance audit | `33d405eac54b9262dd7061af3e8c9970e59503ab20e259ad719c394dac099754` |

## 9. 最终分析结论

v9.2.43 的真实推进是：

```text
v9.2.42: source/fresh S7 接近 legal value gate，但 coverage 略低；oracle pass。
v9.2.43: 先确认 S7 是 threshold-borderline，再把 fresh expansion 扩到 seeds 0..7 与 4 attach；
          expanded rows 下 safety 与 oracle 都未保留。
```

机制判断：

1. v9.2.42 的 frozen S7 failure 确实是 coverage cliff：source PR curve 有合法 threshold-free point。
2. 但 v9.2.43 fresh v2 显示，这个 cliff 不能简单靠 calibration 修复；扩大 seeds/attach 后，bad-event rate 上升到 `0.076823`。
3. Legal controller family 在 heldout split 上没有 survivor；best C1 的 shape signal 还在，但 accepted events 的 precision 与 bad-event 完全不过 gate。
4. Oracle 在 heldout split 上也失败，因此当前不是“oracle 高、legal 特征不足”的分叉，而是“expanded carrier/event space 不够 robust”的分叉。
5. 下一步应重做 carrier/risk gate 或 event-space safety，不应继续微调 S7 threshold，也不能打开 leave-out / paired replay。

最终一句话：

> v9.2.43 真实执行后停在 `R7-OracleLowCarrierMechanismReset`：source S7 是 threshold-borderline，但 fresh v2 扩展后 bad-event 超 gate 且 oracle 不再通过，strict PureKAN functional 仍未成功。
