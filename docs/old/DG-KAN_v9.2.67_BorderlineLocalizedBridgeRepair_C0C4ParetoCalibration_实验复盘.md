# DG-KAN v9.2.67 Borderline-Localized Bridge Repair 与 C0/C4 Pareto Calibration 实验复盘

> 本复盘记录 `DG-KAN_v9.2.67_BorderlineLocalizedBridgeRepair_C0C4ParetoCalibration_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R17-ReferenceFeasibleButComputeFail
base_candidate = LQ-t2-h256
success_v9267_strict_purekan_functional = False
success_v9267_full_functional = False
success_v9267_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9267_borderline_localized_bridge_repair_C0C4_pareto_calibration_first_20260512T223000Z/
```

核心结论：

1. P0 复现 v9.2.66 boundary：source route = `R12-C0BadTrimFail`，exact reference deployable = `0`，true-delta compute pass = `0`，fake/proxy/offload = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；运行中 GPU 有活动，后段进入 CPU-heavy frontier 搜索/汇总。
3. P1 C0/T2/C4/E2 autopsy pass = `1`：T2 removed / E2 added / C0 false-positive / oracle miss attribution fraction 全部为 `1.0`；best backfillable mode = `D3-overtrim-safe-useful`，coverage = `0.004340`。
4. P2 C0 fine trim 过 gate：best = `T2e-MinimalDeletionConstrainedTrimV2`，precision = `0.817241`，coverage = `0.031966`，bad-event = `0.020690`，null-rate = `0.144828`，precision LCB = `0.768711`，bad-event UCB = `0.044396`。
5. P3 T2 coverage backfill 过 gate：best = `B5-CoverageDeficitKnapsack`，precision = `0.838028`，coverage = `0.031305`，bad-event = `0.024648`，null-rate = `0.130282`，precision LCB = `0.790716`，bad-event UCB = `0.049995`。
6. P4 C4/E2 filtered expansion 未过 official：best = `X2-E2FilteredByLocalBadUCB`，coverage = `0.050265`，bad-event = `0.028509`，bad-event UCB = `0.048161`，Jaccard = `0.494135`，diagnostic pass = `1`；但 precision = `0.739035 < 0.75`，null-rate = `0.214912 > 0.15`，official pass = `0`。
7. P5 constrained bridge fill-up 过 gate：best = `C3-T2PlusBackfill`，precision = `0.838028`，coverage = `0.031305`，bad-event = `0.024648`，null-rate = `0.130282`，precision LCB = `0.790716`，bad-event UCB = `0.049995`。
8. P6 exact-reference deployable frontier v11 过 gate：best reference controller = `C3-T2PlusBackfill`，exact_reference_deployable = `1`。
9. P7 true-delta compute v12 仍未过：best = `TBD0-V9256CBD0Reference`，AUC = `0.879072`，bridge AUC = `0.811361`，agreement = `1.0`，step ratio q90 = `2.863280 > 1.50`。
10. P8-P11 因 `P6_reference_or_P7_compute_failed` / `P7_true_delta_compute_failed` gate-blocked，均以 `not_run` 落盘；当前 blocker = `true_delta_compute_still_expensive`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9267_borderline_localized_bridge_repair_C0C4_pareto_calibration.py` | v9.2.67 runner；执行 v9.2.66 boundary 复现、C0/T2/C4/E2 autopsy、C0 fine trim、T2 backfill、C4/E2 filtered expansion、constrained bridge fill-up、exact-reference frontier v11、true-delta compute lane、downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9267_borderline_localized_bridge_repair_C0C4_pareto_calibration.py
```

正式运行：

```bash
python experiments/run_v9267_borderline_localized_bridge_repair_C0C4_pareto_calibration.py \
  --out-dir results/real_rerun_20260506/v9267_borderline_localized_bridge_repair_C0C4_pareto_calibration_first_20260512T223000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
triton_available = true
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 336
interface_events = 24
interface_steps = 2
train_size = 2048
batch_size = 64
hidden_dim = 256
completed_at = 2026-05-12T21:27:14Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R17-ReferenceFeasibleButComputeFail",
  "base_candidate": "LQ-t2-h256",
  "v9266_boundary_pass": 1,
  "localized_autopsy_pass": 1,
  "T2_removed_attribution_fraction": 1.0,
  "E2_added_attribution_fraction": 1.0,
  "C0_false_positive_attribution_fraction": 1.0,
  "oracle_miss_attribution_fraction": 1.0,
  "best_C0_trim_id": "T2e-MinimalDeletionConstrainedTrimV2",
  "C0_fine_trim_pass": 1,
  "C0_trim_precision": 0.8172413793103448,
  "C0_trim_coverage": 0.03196649029982363,
  "C0_trim_bad_event": 0.020689655172413793,
  "C0_trim_null_rate": 0.14482758620689656,
  "C0_trim_precision_lcb": 0.7687107077210187,
  "C0_trim_bad_event_ucb": 0.044396247220322524,
  "best_T2_backfill_id": "B5-CoverageDeficitKnapsack",
  "T2_backfill_pass": 1,
  "T2_backfill_precision": 0.8380281690140845,
  "T2_backfill_coverage": 0.03130511463844797,
  "T2_backfill_bad_event": 0.02464788732394366,
  "T2_backfill_null_rate": 0.13028169014084506,
  "best_C4_expansion_id": "X2-E2FilteredByLocalBadUCB",
  "C4_filtered_expansion_pass": 0,
  "C4_expansion_precision": 0.7390350877192983,
  "C4_expansion_coverage": 0.05026455026455026,
  "C4_expansion_bad_event": 0.02850877192982456,
  "C4_expansion_null_rate": 0.2149122807017544,
  "best_bridge_id": "C3-T2PlusBackfill",
  "bridge_fillup_pass": 1,
  "bridge_precision": 0.8380281690140845,
  "bridge_coverage": 0.03130511463844797,
  "bridge_bad_event": 0.02464788732394366,
  "bridge_null_rate": 0.13028169014084506,
  "bridge_precision_lcb": 0.7907157243773478,
  "bridge_bad_event_ucb": 0.04999458813129621,
  "exact_reference_deployable": 1,
  "true_delta_compute_pass": 0,
  "true_delta_step_ratio_q90": 2.8632798851361203,
  "primary_blocker": "true_delta_compute_still_expensive"
}
```

判断：v9.2.67 首次把 localized bridge decision geometry 做到 exact-reference deployable；但 official controller 仍不能打开，因为 true-delta compute 没进 system envelope。

## 3. P1 C0/T2/C4/E2 autopsy

Artifacts：

```text
p1_C0_T2_C4_E2_autopsy.csv
C0_T2_C4_E2_membership_trace_v9267.csv
T2_removed_rows_trace_v9267.csv
E2_added_rows_trace_v9267.csv
```

Summary：

```text
T2_removed_count = 177
E2_added_count = 2388
C0_false_positive_count = 197
oracle_miss_count = 895
T2_removed_attribution_fraction = 1.0
E2_added_attribution_fraction = 1.0
C0_false_positive_attribution_fraction = 1.0
oracle_miss_attribution_fraction = 1.0
best_backfillable_mode = D3-overtrim-safe-useful
best_backfillable_mode_coverage = 0.004340277777777778
localized_autopsy_pass = 1
```

主要 mode distribution：

```text
T2 removed:
  D3-overtrim-safe-useful = 96
  D6-family-UCB-overconservative = 32
  D2-null-heavy-pocket = 22
  D1-true-bad-pocket = 18
  D5-horizon-overpenalized = 9

E2 added:
  E1-safe-useful-recovered = 1199
  E2-risky-useful = 692
  E3-harmless-null = 274
  E6-score-artifact = 208
  E4-bad-null = 13
```

判断：P1 支持 v9.2.67 的局部修复假设。T2 的 removed rows 中确实有 over-trim safe-useful coverage，可作为 backfill 来源；E2 也有大量 safe-useful recovered rows，但同时混入 risky/null，需要过滤。

## 4. P2 C0 fine trim

Artifacts：

```text
p2_C0_fine_trim_minimal_deletion.csv
C0_trim_finegrid_trace_v9267.csv
```

Best summary：

```text
best_C0_trim_id = T2e-MinimalDeletionConstrainedTrimV2
C0_fine_trim_pass = 1
minimal_deletion_diagnostic_pass = 1
rows_removed = 117
precision = 0.8172413793103448
coverage = 0.03196649029982363
bad_event = 0.020689655172413793
null_rate = 0.14482758620689656
precision_lcb = 0.7687107077210187
bad_event_ucb = 0.044396247220322524
accepted_signal_strata_count = 16
accepted_family_count = 75
max_family_share = 0.09310344827586207
max_stratum_share = 0.15172413793103448
```

判断：H1 成立。C0 broad seed 不是整体不可用；v9.2.66 的失败确实来自 coarse trim。更细的 minimal deletion trim 可以同时过 coverage、precision、bad-event、null、LCB/UCB gate。

## 5. P3 T2 coverage backfill

Artifacts：

```text
p3_T2_coverage_backfill.csv
T2_backfill_trace_v9267.csv
```

| backfill | precision | coverage | bad-event | null-rate | precision LCB | bad UCB | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0-T2Reference | `0.831224` | `0.026124` | `0.008439` | `0.151899` | `0.778341` | `0.030242` | `0` |
| B5-CoverageDeficitKnapsack | `0.838028` | `0.031305` | `0.024648` | `0.130282` | `0.790716` | `0.049995` | `1` |

Best summary：

```text
best_T2_backfill_id = B5-CoverageDeficitKnapsack
T2_backfill_pass = 1
rows_added = 198
accepted_signal_strata_count = 15
accepted_family_count = 65
max_family_share = 0.09507042253521127
max_stratum_share = 0.15140845070422534
```

判断：H2 成立。T2 的 coverage deficit 可以通过 constrained knapsack backfill 补回来，同时 bad-event UCB 仍压在 `0.05` 以下。

## 6. P4 C4/E2 filtered expansion

Artifacts：

```text
p4_C4_E2_filtered_expansion.csv
C4_filtered_expansion_trace_v9267.csv
```

| expansion | precision | coverage | bad-event | null-rate | Jaccard | diagnostic | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| X0-C4Reference | `0.839623` | `0.011684` | `0.066038` | `0.028302` | `0.153448` | `0` | `0` |
| X1-E2Reference | `0.534910` | `0.097884` | `0.268018` | `0.119369` | `0.486680` | `0` | `0` |
| X2-E2FilteredByLocalBadUCB | `0.739035` | `0.050265` | `0.028509` | `0.214912` | `0.494135` | `1` | `0` |
| X5-E2FilteredByC0Agreement | `0.650685` | `0.080467` | `0.117808` | `0.145205` | `0.580685` | `0` | `0` |

Best summary：

```text
best_C4_expansion_id = X2-E2FilteredByLocalBadUCB
C4_filtered_expansion_pass = 0
C4_filtered_expansion_diagnostic_pass = 1
rows_added = 1067
precision_lcb = 0.6968486722124184
bad_event_ucb = 0.04816051455315739
accepted_signal_strata_count = 21
accepted_family_count = 91
```

判断：H3 只达到 diagnostic。E2 filtering 能显著降低 bad-event 并保持 Jaccard/coverage，但 null-rate 和 precision 仍不过 official gate，因此不能把 C4/E2 expansion 写成 deployable controller。

## 7. P5/P6 bridge fill-up 与 exact-reference frontier

Artifacts：

```text
p5_constrained_bridge_fillup.csv
bridge_fillup_trace_v9267.csv
p6_exact_reference_deployable_frontier_v11.csv
reference_frontier_v11_trace.csv
```

| bridge | precision | coverage | bad-event | null-rate | precision LCB | bad UCB | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0-v9266BroadReference | `0.773292` | `0.035494` | `0.055901` | `0.133540` | `0.724493` | `0.086624` | `0` |
| C2-FineTrimOnly | `0.817241` | `0.031966` | `0.020690` | `0.144828` | `0.768711` | `0.044396` | `1` |
| C3-T2PlusBackfill | `0.838028` | `0.031305` | `0.024648` | `0.130282` | `0.790716` | `0.049995` | `1` |
| C4-C4FilteredExpansion | `0.739035` | `0.050265` | `0.028509` | `0.214912` | `0.696849` | `0.048161` | `0` |
| C6-ConstrainedParetoBridgeOptimizer | `0.819820` | `0.036706` | `0.033033` | `0.132132` | `0.774966` | `0.058174` | `0` |

Best summary：

```text
best_bridge_id = C3-T2PlusBackfill
bridge_fillup_pass = 1
exact_reference_deployable = 1
reference_precision = 0.8380281690140845
reference_coverage = 0.03130511463844797
reference_bad_event = 0.02464788732394366
reference_null_rate = 0.13028169014084506
reference_precision_lcb = 0.7907157243773478
reference_bad_event_ucb = 0.04999458813129621
accepted_signal_strata_count = 15
accepted_family_count = 65
max_family_share = 0.09507042253521127
max_stratum_share = 0.15140845070422534
legal_oracle_jaccard = 0.39080459770114945
oracle_gap_remaining = 0.030753968253968256
```

判断：H4 成立。v9.2.67 的 localized bridge repair 首次把 exact-reference decision geometry 做到 deployable gate。注意这是 exact-reference frontier success，不是 strict PureKAN functional success；P7 compute 仍是硬门。

## 8. P7 true-delta compute v12

Artifacts：

```text
p7_true_delta_compute_v12_parallel_lane.csv
true_delta_compute_v12_trace.csv
true_delta_correctness_trace_v9267.csv
true_delta_residual_trace_v9267.csv
```

Summary：

```text
best_true_delta_id = TBD0-V9256CBD0Reference
true_delta_auc = 0.8790720756550714
true_delta_safe_useful_auc = 0.7984599264206053
true_delta_bridge_auc = 0.8113610999018152
true_delta_agreement = 1.0
true_delta_step_ratio_q90 = 2.8632798851361203
true_delta_memory_ratio = 0.9695007261731864
true_delta_compute_pass = 0
```

判断：H5 未闭合。decision geometry 已 viable，但 true branch-delta system path 仍太贵，step ratio 远高于 `1.50`。因此 P8 official controller 不能打开。

## 9. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p8_system_legal_exact_signal_controller.csv` | `P6_reference_or_P7_compute_failed` |
| `p9_leave_dataset_and_stratum_out.csv` | `P7_true_delta_compute_failed` |
| `p10_official_paired_replay.csv` | `P7_true_delta_compute_failed` |
| `p11_short_run_functional_validation.csv` | `P7_true_delta_compute_failed` |

没有把 exact-reference deployable frontier、C0 fine trim、T2 backfill 或 bridge fill-up 写成 system-legal controller / LDO / paired replay / short-run success。

## 10. No-fake audit

```text
rows_checked = 51234
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
borderline_localized_bridge_repair = 1
C0_fine_trim = 1
T2_coverage_backfill = 1
C4_E2_filtered_expansion = 1
constrained_bridge_fillup = 1
exact_reference_deployable_frontier_v11 = 1
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `6eb6267a27c16aa59e46bb09938b29b16aa1586f04b2e86fddcdad0062957c91` |
| runner | `caabab1264284438a14de2d0abb2062a7770b1a1e6b1b0141798b7e9a1d6e4a8` |
| run manifest | `af9fe56824af146f664d291bba455584aba6449a4f3b3dd0e235c5e175d6f54b` |
| route | `f3edc5397d4ae8349e44620276074d09508b0b0b3f79a2c346957ae54f93c2d8` |
| P0 boundary | `fb84da8dd91dbf69488b21d5aee86ceeeae790f8488c1b991d8b99b98f6e4fa7` |
| P1 autopsy | `140c08672b912d51b6f9b9c0b568d3fa000d780cdeb1c4aaab8c14ac29c1f2e6` |
| P2 C0 fine trim | `493bfc1b25109e7d230c54aa55993f858a13189a2b341e0c5199d182447386a0` |
| P3 T2 backfill | `62c763695d6942420436525223b00397eab7b5877e69dbfcdf4f71635835ab37` |
| P4 C4/E2 expansion | `3ac790caa71c70bc538dbcd987a69fd604e8a0f1b2a2142380a61d4a007728a2` |
| P5 bridge fill-up | `6bcce8e189b6bc373577a1d0f3b07b22fae078fa673b7739efdff18fc1692b04` |
| P6 reference frontier | `1e6fd0a6dabe5ce3a6d79a2b086205bc8289ea228ff44b09d2120884ed853243` |
| P7 compute lane | `5d1d4f4fea29c97c150bfb44d9a84ac144aa717d6a73268163ae866838aff007` |
| P8 system controller | `8ffe834df22a40be09587b027c5115d2400620c03f3148fe338f538836ce808a` |
| P9 boundary | `b792cfbf21abbfc88da967899ffe88965d37b7c34ebfce9145d0372225a6e140` |
| P10 boundary | `687454251a79c867157ef3b1ca697805670e0952445370a98054693b0c9f3a99` |
| P11 boundary | `28ede86de922a11f458e19fd5e982d0369fde236879ad0ea4b9de29ca799dc52` |
| failure table | `df93ef895f8c7ad0a02e886a8d9d2796848d8c3037437a71ba2f98925bc1332e` |
| provenance audit | `b741853410a2e0dd1fb2062c8f035044bcaf8d657fdedddf409ef90e2993481c` |

## 12. 最终分析结论

v9.2.67 的真实推进是：

```text
v9.2.66: oracle/legal gap 已定位，但 C0 trim 与 C4 expansion 没能 bridge 成 deployable frontier。
v9.2.67: C0 fine trim 与 T2 backfill 成功；
          constrained bridge fill-up 与 exact-reference frontier v11 首次过 deployable gate；
          但 true-delta compute 仍未过 system envelope。
```

机制判断：

1. H1 成立：C0 broad seed 可以通过 finer minimal trim 过 official decision gate。
2. H2 成立：T2 safe core 可以通过 small constrained backfill 恢复 coverage，同时保持 bad/null/confidence。
3. H3 只达到 diagnostic：E2 filtered expansion 仍受 null/precision 限制，不能作为 official expansion path。
4. H4 成立：localized bridge fill-up 过 exact-reference deployable frontier。
5. H5 未成立：true-delta compute 仍 `step_ratio_q90 = 2.863280`，因此 system-legal controller、LDO/LSO、paired replay 全部不能打开。

最终一句话：

> v9.2.67 真实执行后进入 `R17-ReferenceFeasibleButComputeFail`：localized bridge decision geometry 已经可部署，但 true branch-delta compute 仍太贵，strict PureKAN functional 仍未成功。
