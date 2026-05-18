# DG-KAN v9.2.51 Branch-Logit Delta Kernel 与 Coverage-Preserving Online Controller 实验复盘

> 本复盘记录 `DG-KAN_v9.2.51_BranchLogitDeltaKernel_CoveragePreservingOnlineController_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R9-BranchDeltaPredictiveButTooExpensive
base_candidate = LQ-t2-h256
success_v9251_strict_purekan_functional = False
success_v9251_full_functional = False
success_v9251_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9251_branch_logit_delta_kernel_coverage_preserving_online_controller_first_20260512T030000Z/
```

核心结论：

1. P0 复现 v9.2.50 boundary：source route = `R10-PredictiveButMetricKernelTooExpensive`，oracle support pass = `1`，support-region controller pass = `0`，fake/proxy = `0`。
2. 本轮 manifest 记录 `device = cuda`；运行中 GPU 确认有活动，未使用 CPU offload。
3. P1 M0 logit-prep attribution pass = `1`，dominant subphase = `L1-RealFunctional delta preparation`，ratio = `0.508055`；branch-logit path ratio = `0.889772`。
4. P2 branch-logit delta 有预测性：best measured/audited candidate = `BLD1-DeltaViewExactBranchLogits`，AUC = `0.886903`，accept agreement = `1.0`。
5. P2 system gate 未过：`BLD1` step ratio = `4.700044 > 1.50`；因此 branch delta system pass = `0`。
6. `BLD6-FusedMultiBranchDeltaKernel` 只作为 `not_real_fused_kernel_exact_score_diagnostic` 留在 matrix 中，没有被写成 route-level best 或 system success。
7. P3 cheap recall generator 未过：best = `CR3-FamilyReliability`，recall = `0.346837`，candidate rate = `0.25`，candidate bad-event = `0.225347`。
8. P4 cascade controller 未过：best = `C4-LowRankDeltaController`，precision = `0.322581`，coverage = `0.008073`，bad-event = `0.215054`，official eligible = `0`。
9. P5 oracle support 仍通过：precision = `1.0`，coverage = `0.122135`，bad-event = `0.0`；但 measured signal strata = `3`，balanced diagnostic rows = `18`，support measurement pass = `0`。
10. 当前 blocker：`branch_logit_delta_predictive_but_not_system_legal`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9251_branch_logit_delta_kernel_coverage_preserving_online_controller.py` | v9.2.51 runner；重新生成 real train-stream rows，执行 M0 logit-prep attribution、branch-logit delta matrix、cheap recall、cascade controller、support expansion、route/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9251_branch_logit_delta_kernel_coverage_preserving_online_controller.py
```

正式运行：

```bash
python experiments/run_v9251_branch_logit_delta_kernel_coverage_preserving_online_controller.py \
  --out-dir results/real_rerun_20260506/v9251_branch_logit_delta_kernel_coverage_preserving_online_controller_first_20260512T030000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 160
train_size = 2048
batch_size = 64
hidden_dim = 256
```

## 2. Route

```json
{
  "route": "R9-BranchDeltaPredictiveButTooExpensive",
  "base_candidate": "LQ-t2-h256",
  "v9250_boundary_pass": 1,
  "m0_subphase_attribution_pass": 1,
  "dominant_m0_subphase": "L1-RealFunctional delta preparation",
  "branch_logit_path_ratio": 0.8897715421548243,
  "best_branch_delta_id": "BLD1-DeltaViewExactBranchLogits",
  "branch_delta_predictivity_pass": 1,
  "branch_delta_system_pass": 0,
  "branch_delta_auc": 0.8869032948295491,
  "branch_delta_corr": -0.012347004615124783,
  "branch_delta_accept_agreement": 1.0,
  "branch_delta_step_ratio_q90": 4.700043754266211,
  "branch_delta_memory_ratio": 0.9695007261731864,
  "best_cheap_recall_id": "CR3-FamilyReliability",
  "cheap_recall_pass": 0,
  "safe_good_recall": 0.3468372423596304,
  "candidate_rate": 0.25,
  "candidate_bad_event": 0.22534722222222223,
  "best_controller_id": "C4-LowRankDeltaController",
  "cascade_controller_pass": 0,
  "controller_auc": 0.5459881417353576,
  "controller_corr": 0.18102644407712914,
  "accepted_precision": 0.3225806451612903,
  "accepted_coverage": 0.008072916666666667,
  "accepted_bad_event_rate": 0.21505376344086022,
  "accepted_signal_strata_count": 2,
  "accepted_family_count": 15,
  "max_family_share": 0.17204301075268819,
  "support_measurement_pass": 0,
  "natural_real_event_count": 11520,
  "balanced_diagnostic_real_event_count": 18,
  "measured_signal_strata_count": 3,
  "measured_family_count": 42,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.12213541666666666,
  "oracle_bad_event": 0.0,
  "primary_blocker": "branch_logit_delta_predictive_but_not_system_legal",
  "next_required_implementation": "implement_real_fused_branch_delta_kernel",
  "success_v9251_strict_purekan_functional": 0,
  "success_v9251_full_functional": 0,
  "success_v9251_external_ready": 0
}
```

判断：branch-output displacement 能保留 safe-good 信号，但当前可测 exact/delta-view 路径仍太贵；近似低成本路径没有达到 agreement / predictivity / accept gate。

## 3. P1 M0 logit-preparation subphase attribution

Artifacts：

```text
p1_m0_logit_preparation_subphase_attribution.csv
m0_logit_prep_trace_v9251.csv
```

| subphase | time ms | ratio | branch count |
|---|---:|---:|---:|
| L1-RealFunctional delta preparation | `11.797958` | `0.508055` | `1` |
| L5-branch forward preparation | `5.750522` | `0.247635` | `4` |
| L0-base logits reuse | `1.741129` | `0.074978` | `1` |
| L6-logit buffer allocation | `1.635920` | `0.070448` | `5` |
| L3-bestLR delta preparation | `0.774380` | `0.033347` | `1` |

Summary：

```text
profile_event_count = 9
unknown_fraction = 0.0
m0_phase_time_sum_close_to_F7 = 1
branch_logit_path_ratio = 0.8897715421548243
m0_subphase_attribution_pass = 1
```

判断：v9.2.50 的 M0 成本继续被拆开；本轮确认主要成本确实在 functional delta preparation 与 branch forward/logit path，不是 controller decision 或 logging。

## 4. P2 branch-logit delta kernel matrix

Artifacts：

```text
p2_branch_logit_delta_kernel_matrix.csv
branch_logit_delta_trace_v9251.csv
system_branch_delta_overhead_trace_v9251.csv
```

| branch delta | status | full branch forward | AUC | agreement | step ratio | precision@0.03 | bad-event@0.03 | kernel pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BLD0-ExactGapProbeReference | measured_full_branch_forward_reference | `1` | `0.886903` | `1.000000` | `6.138950` | `0.439306` | `0.540462` | `0` |
| BLD1-DeltaViewExactBranchLogits | delta_view_exact_branch_forward | `1` | `0.886903` | `1.000000` | `4.700044` | `0.439306` | `0.540462` | `0` |
| BLD2-LinearizedLogitDeltaJVP | metadata_linearized_delta_formula | `0` | `0.534446` | `0.401389` | `1.411116` | `0.205202` | `0.089595` | `0` |
| BLD3-LowRankOutputDeltaSketch | low_rank_delta_sketch_formula | `0` | `0.541952` | `0.316753` | `1.513895` | `0.283237` | `0.407514` | `0` |
| BLD4-TopClassTailClassDelta | selected_class_delta_formula | `0` | `0.720016` | `0.317708` | `1.616674` | `0.231214` | `0.332370` | `0` |
| BLD5-CachedControlBaselineDelta | cached_control_baseline_formula | `0` | `0.596259` | `0.455816` | `1.308337` | `0.213873` | `0.231214` | `0` |
| BLD6-FusedMultiBranchDeltaKernel | not_real_fused_kernel_exact_score_diagnostic | `1` | `0.886903` | `1.000000` | `3.312527` | `0.439306` | `0.540462` | `0` |
| BLD7-HybridDeltaCascade | hybrid_borderline_exact_diagnostic | `1` | `0.700584` | `0.316753` | `2.130569` | `0.242775` | `0.294798` | `0` |

判断：

1. `BLD1` 保留 exact gap-probe 的 predictivity 和 agreement，但仍依赖 branch forward，step ratio `4.700044` 远高于 `1.50`。
2. `BLD4` 达到 AUC `0.720016`，但 agreement 只有 `0.317708`，bad-event 也过高，不能成为 controller。
3. 低成本 `BLD2/BLD3/BLD5` 没有保留 safe-good signal 或 accept agreement。
4. `BLD6` 只是 exact-score diagnostic upper-bound；本轮明确没有把它写成 real fused kernel pass。

## 5. P3 cheap recall generator

Artifacts：

```text
p3_cheap_recall_generator_matrix.csv
cheap_recall_trace_v9251.csv
```

Summary：

| metric | value |
|---|---:|
| best cheap recall | `CR3-FamilyReliability` |
| cheap recall AUC | `0.648250` |
| safe-good recall | `0.346837` |
| candidate rate | `0.250000` |
| candidate precision | `0.169444` |
| candidate bad-event | `0.225347` |
| cheap recall pass | `0` |

判断：cheap recall 没有达到 `recall >= 0.80` / `bad-event <= 0.20`。它仍只能作为诊断特征，不能支撑 coverage-preserving cascade。

## 6. P4 coverage-preserving cascade controller

Artifacts：

```text
p4_coverage_preserving_cascade_controller.csv
cascade_controller_trace_v9251.csv
```

| controller | official eligible | AUC heldout | corr heldout | precision | coverage | bad-event | step q90 | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1-CheapRecallThenBranchDelta | `0` | `0.499617` | `0.176922` | `0.174359` | `0.016927` | `0.153846` | `4.700044` | `0` |
| C2-FamilyBalancedBranchDelta | `0` | `0.500486` | `0.176582` | `0.124098` | `0.060156` | `0.320346` | `4.700044` | `0` |
| C3-TopClassDeltaController | `0` | `0.507804` | `0.194319` | `0.187500` | `0.016667` | `0.140625` | `4.700044` | `0` |
| C4-LowRankDeltaController | `0` | `0.545988` | `0.181026` | `0.322581` | `0.008073` | `0.215054` | `4.700044` | `0` |
| C5-HybridBorderlineExactController | `0` | `0.512189` | `0.202208` | `0.172043` | `0.016146` | `0.145161` | `4.700044` | `0` |
| C6-ParetoCostAwareController | `0` | `0.495961` | `0.169646` | `0.180412` | `0.016840` | `0.170103` | `4.700044` | `0` |
| C7-Oracle | `0` | `1.000000` | `0.368272` | `1.000000` | `0.042622` | `0.000000` | `4.700044` | `0` |

判断：因为 P2 没有 system-legal branch-delta survivor，所有 legal controllers 的 `official_eligible = 0`。诊断层面也没有 controller 同时满足 precision / coverage / bad-event。

## 7. P5 online support / stratum expansion

Artifacts：

```text
p5_online_support_stratum_expansion.csv
support_density_trace_v9251.csv
```

Summary：

```text
natural_real_event_count = 11520
balanced_diagnostic_real_event_count = 18
measured_signal_strata_count = 3
measured_family_count = 42
support_measurement_pass = 0
oracle_support_pass = 1
oracle_precision = 1.0
oracle_coverage = 0.12213541666666666
oracle_bad_event = 0.0
```

Natural signal strata：

| stratum | rows |
|---|---:|
| S4-RoleWiseCurvature | `11074` |
| S1-CEHardTail | `440` |
| S5-UncertaintyLCB | `6` |

Dataset distribution：

| dataset | rows |
|---|---:|
| MNIST | `3840` |
| Fashion-MNIST | `3840` |
| KMNIST | `3840` |

判断：natural rows 和 family count 足够，但 signal strata 仍过窄；balanced diagnostic 只能从真实 measured rows 中抽取，只有 `18` rows，没有补造或重复扩写成 support pass。

## 8. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p6_leave_dataset_and_stratum_out.csv` | `P2_branch_logit_delta_system_failed` |
| `p7_official_paired_replay.csv` | same |
| `p8_short_run_functional_validation.csv` | same |
| `p9_full_10seed_functional_validation.csv` | same |
| `p10_robustness_external_ready.csv` | same |

没有把 exact branch probe、fused diagnostic、oracle row 或 posthoc label 写成 LDO/LSO、paired replay、short-run、full-run 或 external-ready success。

## 9. No-fake audit

```text
rows_checked = 23223
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 10. Hash

| artifact | SHA256 |
|---|---|
| plan | `af1b3ed46abd3de6da7fc5623e1248e3f4c4e7422b20ad9c40c998fed865ae7e` |
| runner | `78d8dbe0c52583e031072973596df61b720624f312902ef160571740a55ee3da` |
| run manifest | `9df520de85abdfcfb7e1b9b0e919039f8716fde7daf670ed9563cc7d4ed450ef` |
| route | `54ed05e80156dc5ae1caa1ed3ee8040bf74cba04960f46fd233f2c48d5da3e69` |
| P0 boundary | `436f50eb20b49d332fb5e3a818ffec73e4baafe46f3a3c8f66aa53ceab56b3e2` |
| P1 M0 attribution | `00f3127bad138be3e819a882e9f3fc47458a2130b33e7d13aea91179b2e0e2af` |
| P2 branch delta | `f7c253f9c8c21cb6675cab809f5c59c5ca28ae9e1dabfbd099c48350930c8e62` |
| P3 cheap recall | `538814915a29c2da58837ae8586fc1457a4fa16bfd30992fab01aed07110e86f` |
| P4 cascade controller | `aea8cbff5d51e343c48023b1c4a873d16a2b81bf45ab9270b372325896e57b38` |
| P5 support expansion | `1d7c6cbead2dac6f34720072b7af5db9f4f68b11c2225f7ba8b4d01c12c67fbb` |
| P6 boundary | `de784a519e143b377f0ac906b6a902965ab1b6b1fe267d8680a1e17f9d18e245` |
| P7 boundary | `2a742b884af67ea5b8fa526e190cbed856e4b57445ab2f652a6387210e0593f2` |
| P8 boundary | `9c8c5ecfdc9e80335c3d7f265e6580867b530e81a5f390d1feafee4df8afdc63` |
| P9 boundary | `16f8c6cf8543d2fa7dd5b5702c0f130b0ff4b8662738462a4e311d3e4367368d` |
| P10 boundary | `0a5001995fd09c7e7549fbed46f26cb8db4657c449184f3541f78113423c0f72` |
| failure table | `bfcc81f7f27739e2af6bd7bfcc6e7e4a016b7831c9d3b2ea6ccfac18ba1e69c9` |
| provenance audit | `bd7226910b60f20fc975bb02c590f83ec540eb804d60daf890a9fe0c42437561` |

## 11. 最终分析结论

v9.2.51 的真实推进是：

```text
v9.2.50: metric/logit preparation expensive，oracle support high。
v9.2.51: M0 logit-prep 被拆开；branch-delta exact/delta-view 保留 signal，
          但仍不 system-legal。
```

机制判断：

1. H1 基本成立：M0 的主要成本确实在 branch-logit path，branch path ratio 达到 `0.889772`。
2. H2 成立：fresh natural rows 中 oracle support 没有消失，precision `1.0`，coverage `0.122135`，bad-event `0.0`。
3. H4 只部分成立：exact/delta-view branch delta 能保留 safe-good signal，但 linearized/low-rank/cached cheap approximation 没有保留 agreement。
4. H3 不成立：cheap recall generator recall 远低于 `0.80`，candidate bad-event 也高于 `0.20`。
5. Controller 与 support 不允许打开 downstream：P2 system fail 是 terminal blocker，P5 support 也仍然 narrow。

最终一句话：

> v9.2.51 真实执行后停在 `R9-BranchDeltaPredictiveButTooExpensive`：branch-logit delta 确实能观察到 safe-good，但当前可测 exact/delta-view 路径太贵，低成本近似没有保留 agreement，strict PureKAN functional 仍未成功。
