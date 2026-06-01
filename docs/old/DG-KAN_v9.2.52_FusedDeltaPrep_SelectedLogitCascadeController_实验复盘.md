# DG-KAN v9.2.52 Fused Delta-Prep 与 Selected-Logit Cascade Controller 实验复盘

> 本复盘记录 `DG-KAN_v9.2.52_FusedDeltaPrep_SelectedLogitCascadeController_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R10-FusedDeltaPredictiveButTooExpensive
base_candidate = LQ-t2-h256
success_v9252_strict_purekan_functional = False
success_v9252_full_functional = False
success_v9252_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9252_fused_delta_prep_selected_logit_cascade_controller_first_20260512T040000Z/
```

核心结论：

1. P0 复现 v9.2.51 boundary：source route = `R9-BranchDeltaPredictiveButTooExpensive`，fake/proxy = `0`。
2. 本轮 manifest 记录 `device = cuda`；运行中确认 GPU 有活动：`NVIDIA L4, 33 %, 326 MiB`。
3. P1 L1 delta-prep attribution pass = `1`，dominant subphase = `D0-role score / controller scalar preparation`，ratio = `0.770145`。
4. P2 fused delta-prep matrix 未过 system gate：best = `FDP6-HybridFusedDeltaPrep`，correctness pass = `1`，但 step ratio = `3.898403 > 1.50`。
5. `FDP5-FusedRealFunctionalDeltaPrep` 明确为 `not_implemented_real_fused_kernel`，没有写成 pass。
6. P3 selected-logit branch delta 保留 signal：best route-level reference = `SLD0-FullLogitExactReference`，AUC = `0.888401`，agreement = `1.0`，但 step ratio = `6.096370`。
7. P3 中 `SLD5-CachedControlSelectedDelta` step ratio = `1.356746`，但 agreement = `0.448661`，不能 official。
8. P4 candidate generator 未过：best = `CG1-SelectedLogitRecall`，recall = `0.576766`，candidate bad-event = `0.251535`。
9. P5 cascade controller 未过：best = `C4-HybridExactBorderlineController`，precision = `0.152439`，coverage = `0.040675`，bad-event = `0.256098`，official eligible = `0`。
10. P6 oracle support 仍通过：precision = `1.0`，coverage = `0.121693`，bad-event = `0.0`；但 measured signal strata = `3`，balanced diagnostic rows = `36`，support measurement pass = `0`。
11. 当前 blocker：`selected_or_fused_delta_predictive_but_not_system_legal`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9252_fused_delta_prep_selected_logit_cascade_controller.py` | v9.2.52 runner；重新生成 real train-stream rows，执行 L1 delta-prep attribution、fused delta-prep matrix、selected-logit delta matrix、candidate generator、cascade controller、support expansion、route/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9252_fused_delta_prep_selected_logit_cascade_controller.py
```

正式运行：

```bash
python experiments/run_v9252_fused_delta_prep_selected_logit_cascade_controller.py \
  --out-dir results/real_rerun_20260506/v9252_fused_delta_prep_selected_logit_cascade_controller_first_20260512T040000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 168
train_size = 2048
batch_size = 64
hidden_dim = 256
```

## 2. Route

```json
{
  "route": "R10-FusedDeltaPredictiveButTooExpensive",
  "base_candidate": "LQ-t2-h256",
  "v9251_boundary_pass": 1,
  "l1_subphase_attribution_pass": 1,
  "dominant_l1_subphase": "D0-role score / controller scalar preparation",
  "best_fused_delta_id": "FDP6-HybridFusedDeltaPrep",
  "fused_delta_correctness_pass": 1,
  "fused_delta_system_pass": 0,
  "fused_delta_step_ratio_q90": 3.898402841026808,
  "fused_delta_memory_ratio": 0.6107854574891074,
  "best_selected_delta_id": "SLD0-FullLogitExactReference",
  "selected_delta_predictivity_pass": 1,
  "selected_delta_system_pass": 0,
  "selected_delta_auc": 0.8884008136827201,
  "selected_delta_corr": -0.010621839689848079,
  "selected_delta_accept_agreement": 1.0,
  "best_candidate_generator_id": "CG1-SelectedLogitRecall",
  "candidate_generator_pass": 0,
  "safe_good_recall": 0.576766304347826,
  "candidate_rate": 0.3500330687830688,
  "candidate_bad_event": 0.25153519130845536,
  "best_controller_id": "C4-HybridExactBorderlineController",
  "cascade_controller_pass": 0,
  "controller_auc": 0.5135211501703648,
  "controller_corr": 0.21575694691235842,
  "accepted_precision": 0.1524390243902439,
  "accepted_coverage": 0.040674603174603176,
  "accepted_bad_event_rate": 0.25609756097560976,
  "support_measurement_pass": 0,
  "natural_real_event_count": 12096,
  "balanced_diagnostic_real_event_count": 36,
  "measured_signal_strata_count": 3,
  "measured_family_count": 45,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.12169312169312169,
  "oracle_bad_event": 0.0,
  "primary_blocker": "selected_or_fused_delta_predictive_but_not_system_legal"
}
```

判断：oracle-good events 仍存在，selected/full exact delta 仍有强信号，但 fused/selected delta 还没有合法系统路径；因此 controller、LDO/LSO、paired replay 全部不能打开。

## 3. P1 L1 delta-prep attribution

Artifacts：

```text
p1_l1_delta_prep_subphase_attribution.csv
l1_delta_prep_trace_v9252.csv
```

| subphase | time ms | ratio | delta tensors | delta buffer MB |
|---|---:|---:|---:|---:|
| D0-role score / controller scalar preparation | `25.658071` | `0.770145` | `0` | `0.000000` |
| D2-delta tensor assembly | `3.544766` | `0.106399` | `27` | `7.066406` |
| D1-functional delta coefficient compute | `2.760031` | `0.082844` | `0` | `0.000000` |
| D7-host sync / logging | `0.399406` | `0.011988` | `0` | `0.000000` |
| D3-role-wise delta packing | `0.283266` | `0.008502` | `9` | `7.066406` |

Summary：

```text
profile_event_count = 9
unknown_fraction = 0.0
l1_subphase_attribution_pass = 1
dominant_l1_subphase = D0-role score / controller scalar preparation
```

判断：L1 的 dominant 这轮落到 role score / scalar preparation，而不是写成未归因开销。

## 4. P2 fused delta-prep kernel matrix

Artifacts：

```text
p2_fused_delta_prep_kernel_matrix.csv
fused_delta_prep_trace_v9252.csv
system_fused_delta_overhead_trace_v9252.csv
```

| fused delta | status | relerr | cos | time ratio | step ratio | memory | pass |
|---|---|---:|---:|---:|---:|---:|---:|
| FDP0-V9251-BLD1-reference | measured_reference_full_delta_prep | `0.000000` | `1.000000` | `1.000000` | `6.096370` | `0.969501` | `0` |
| FDP1-CachedFunctionalDeltaState | measured_cache_reuse_smoke | `0.000000` | `1.000000` | `0.740000` | `5.075885` | `0.911331` | `0` |
| FDP2-PreallocatedDeltaBuffers | measured_preallocated_workspace_smoke | `0.000000` | `1.000000` | `0.810000` | `5.350631` | `0.853161` | `0` |
| FDP3-PackedRoleWiseDeltaPrep | metadata_packed_layout_audit | `0.000030` | `0.999800` | `0.620000` | `4.604892` | `0.698041` | `0` |
| FDP4-InPlaceDeltaViewNoCopy | measured_delta_view_no_copy_smoke | `0.000000` | `1.000000` | `0.580000` | `4.447895` | `0.678651` | `0` |
| FDP5-FusedRealFunctionalDeltaPrep | not_implemented_real_fused_kernel | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0` |
| FDP6-HybridFusedDeltaPrep | derived_cache_prealloc_pack_view_not_single_kernel | `0.000030` | `0.999800` | `0.440000` | `3.898403` | `0.610785` | `0` |

判断：`FDP6` 是最好的 measured/derived audit，但不是 single fused kernel，且 step ratio 仍远高于 `1.50`。`FDP5` 没有真实实现，明确不写成 pass。

## 5. P3 selected-logit branch delta matrix

Artifacts：

```text
p3_selected_logit_branch_delta_matrix.csv
selected_logit_delta_trace_v9252.csv
```

| selected delta | status | AUC | corr | agreement | precision | coverage | bad-event | step | system |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SLD0-FullLogitExactReference | full_logit_exact_reference | `0.888401` | `-0.010622` | `1.000000` | `0.432507` | `0.030010` | `0.548209` | `6.096370` | `0` |
| SLD1-TrueTopHardNegativeDelta | selected_logit_formula | `0.535613` | `0.327863` | `0.364914` | `0.195592` | `0.030010` | `0.079890` | `1.407710` | `1` |
| SLD2-CEMarginSelectedDelta | selected_ce_margin_formula | `0.721109` | `0.271644` | `0.433780` | `0.242424` | `0.030010` | `0.327824` | `1.509637` | `0` |
| SLD3-TopKClassDelta | topk_selected_formula | `0.719907` | `0.269796` | `0.314732` | `0.239669` | `0.030010` | `0.330579` | `1.611564` | `0` |
| SLD4-SelectedLogitLowRankDelta | lowrank_selected_formula | `0.539936` | `-0.046540` | `0.313823` | `0.272727` | `0.030010` | `0.424242` | `1.509637` | `0` |
| SLD5-CachedControlSelectedDelta | cached_control_selected_formula | `0.589056` | `0.390732` | `0.448661` | `0.203857` | `0.030010` | `0.225895` | `1.356746` | `1` |
| SLD6-FusedMultiBranchSelectedLogitKernel | not_implemented_real_fused_selected_kernel | `0.888401` | `-0.010622` | `1.000000` | `0.432507` | `0.030010` | `0.548209` | `1.458673` | `0` |
| SLD7-HybridSelectedExactBorderline | hybrid_borderline_exact_diagnostic | `0.711423` | `0.238913` | `0.313823` | `0.256198` | `0.030010` | `0.289256` | `2.274093` | `0` |

判断：

1. Full exact reference 仍强，但太贵且 bad-event gate 不可用。
2. `SLD5` 进入 step envelope 且 corr 过形状 gate，但 accept agreement 只有 `0.448661`，不能 official。
3. `SLD6` 是 not implemented upper-bound row，不允许写成 fused selected kernel success。

## 6. P4 high-recall candidate generator

Artifacts：

```text
p4_high_recall_candidate_generator.csv
candidate_generator_trace_v9252.csv
```

| generator | AUC | recall | candidate rate | precision | bad-event | pass |
|---|---:|---:|---:|---:|---:|---:|
| CG1-SelectedLogitRecall | `0.719907` | `0.576766` | `0.350033` | `0.200520` | `0.251535` | `0` |
| CG0-V9251-CR3-Reference | `0.638037` | `0.476223` | `0.350033` | `0.165564` | `0.206424` | `0` |
| CG4-FamilyReliability | `0.602849` | `0.474185` | `0.350033` | `0.164856` | `0.201701` | `0` |

判断：candidate recall 从 v9.2.51 有推进，但仍低于 `0.80`，且 bad-event 高于 `0.20`。不能作为 official candidate stage。

## 7. P5 cascade controller

Artifacts：

```text
p5_coverage_preserving_cascade_controller.csv
cascade_controller_trace_v9252.csv
```

| controller | official | AUC | corr | precision | coverage | bad-event | strata | families | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1-CheapRecallThenFusedDelta | `0` | `0.501782` | `0.192240` | `0.098485` | `0.010913` | `0.136364` | `2` | `12` | `0` |
| C2-SelectedLogitCascade | `0` | `0.512339` | `0.214592` | `0.127854` | `0.018105` | `0.164384` | `2` | `12` | `0` |
| C3-FamilyBalancedSelectedDelta | `0` | `0.498512` | `0.178194` | `0.114809` | `0.099372` | `0.361897` | `2` | `18` | `0` |
| C4-HybridExactBorderlineController | `0` | `0.513521` | `0.215757` | `0.152439` | `0.040675` | `0.256098` | `2` | `12` | `0` |
| C5-FusedMultiBranchController | `0` | `0.496563` | `0.176870` | `0.152130` | `0.040757` | `0.261663` | `2` | `12` | `0` |
| C6-ParetoCostAwareController | `0` | `0.498328` | `0.184974` | `0.149701` | `0.041419` | `0.263473` | `2` | `12` | `0` |
| C7-Oracle | `0` | `1.000000` | `0.367692` | `1.000000` | `0.043403` | `0.000000` | `2` | `18` | `0` |

判断：P2/P3 没有 system-legal survivor，所以 P5 `official_eligible = 0`。diagnostic best 也无法同时满足 precision / bad-event gate。

## 8. P6 support expansion

Artifacts：

```text
p6_online_support_stratum_expansion.csv
support_density_trace_v9252.csv
```

Summary：

```text
natural_real_event_count = 12096
balanced_diagnostic_real_event_count = 36
measured_signal_strata_count = 3
measured_family_count = 45
support_measurement_pass = 0
oracle_support_pass = 1
oracle_precision = 1.0
oracle_coverage = 0.12169312169312169
oracle_bad_event = 0.0
```

判断：natural rows 达到 `>=12000`，但 signal strata 仍只有 `3`，balanced diagnostic rows 只有 `36`，离 `>=6000` 很远。没有补造 balanced rows。

## 9. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P2_P3_branch_delta_system_failed` |
| `p8_official_paired_replay.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_10seed_functional_validation.csv` | same |
| `p11_robustness_external_ready.csv` | same |

没有把 selected/full exact delta、not-implemented fused kernel、oracle row 或 diagnostic controller 写成 LDO/LSO、paired replay、short-run、full-run 或 external-ready success。

## 10. No-fake audit

```text
rows_checked = 24436
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `5d46c6f241adbab50764d8c66517851bb76cd3407d44ac6478d23b09285b9479` |
| runner | `3d5734feedec197bbcad1809f8c86f94f8bcc1eebcc9b5795afd7ae067e339fb` |
| run manifest | `221572b770afdb8752c842b895f9b28bfbba7f9b39eebc42d2a7f925dda95b52` |
| route | `e22d943366a668939fa47778479aa84705f4166b0ec0753d324a7a119df96a58` |
| P0 boundary | `7324ddc0b20e8c0d83f77c7b6815e516cd093c94189e1cdfe97c15d5a0114c97` |
| P1 L1 attribution | `a0dbf9bdb080f2ef48436544c1f4c2fe060060be9802db47eef69a3f3c737537` |
| P2 fused delta prep | `70bdb703edbbcc314358f7578794f7b976f7dde02cc02a26a3821b8153fad3f5` |
| P3 selected delta | `1169f7ce7d5cd617becba385021ed7dfc5865969c5bf38cde4694a35173ab3c1` |
| P4 candidate generator | `aa5b41b7d30068e324f473a724f77e122f6afa9e63b3bd3442b17e1b46c18285` |
| P5 cascade controller | `aa20a843bf7980e5a497953874c1dfe2b0794edf02ee47661e90d28d5c3c5ae6` |
| P6 support expansion | `8e838afb932aefc877b13cf4b6cbe9e9731db6616bf8f817dc3830887eca6d18` |
| P7 boundary | `02e46396ff2bfcbce90604b02aec62582d425a638c311014f99f4368e7ac17dd` |
| P8 boundary | `9b0b5c67da07a95caa315bac36050d76729649d32c24a5c31bd479b7d2947e96` |
| P9 boundary | `74f9a7bbab9c975ed42eb40fa4cd81a7bbece62ce67a00207ef1fe2a96f33530` |
| P10 boundary | `4166ff8a43b30d7b43797a4e827faac3d25bd6bd2260fd761a4fe9b6fee16ba2` |
| P11 boundary | `9e3307fd0f28e27151637509d9b8799d11e7702d06fa4f547454e27f6961f8b4` |
| failure table | `814ee768c92e578b2ba9703325e4eb8c2f51e3d6de5408633083c68fdb473089` |
| provenance audit | `a39faaf6b8317bdbcf334b7e17baeeeed69003b2b215b3a90b74f12178a31770` |

## 12. 最终分析结论

v9.2.52 的真实推进是：

```text
v9.2.51: exact branch-logit delta 可预测 safe-good，但太贵。
v9.2.52: L1 delta-prep 被拆到 D0-D7；
          fused/selected branch delta matrix 已测；
          selected/full exact signal 仍在，但 system/legal 仍未闭合。
```

机制判断：

1. H1 得到推进但没有闭合：L1 cost 已归因，dominant 是 D0 role/scalar preparation；但 fused delta-prep best step ratio 仍 `3.898403`。
2. H2 只部分成立：selected/full exact 保留强 signal；低成本 selected formula 中 `SLD5` 进入 step envelope，但 agreement 只有 `0.448661`。
3. H3 未成立：candidate recall 最高只有 `0.576766`，bad-event `0.251535`。
4. H4 成立：fresh oracle support 没有消失，precision `1.0`，coverage `0.121693`，bad-event `0.0`。
5. H5 指向下一步：真正需要 real fused delta-prep / selected-logit kernel，而不是继续用 formula diagnostic 或 threshold patch。

最终一句话：

> v9.2.52 真实执行后停在 `R10-FusedDeltaPredictiveButTooExpensive`：selected/full exact branch-delta signal 仍强，但 fused/selected system-legal kernel 没有闭合，strict PureKAN functional 仍未成功。
