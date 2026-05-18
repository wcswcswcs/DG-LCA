# DG-KAN v9.2.57 True Branch-Delta Compute Closure 与 Exact-Signal Controller Feasibility 实验复盘

> 本复盘记录 `DG-KAN_v9.2.57_TrueBranchDeltaComputeClosure_ExactSignalControllerFeasibility_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R11-ExactSignalControllerInfeasible
base_candidate = LQ-t2-h256
success_v9257_strict_purekan_functional = False
success_v9257_full_functional = False
success_v9257_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9257_true_branch_delta_compute_closure_exact_signal_controller_feasibility_first_20260512T103200Z/
```

核心结论：

1. P0 复现 v9.2.56 boundary：source route = `R12-TrueDeltaPredictiveButTooExpensive`，true branch-delta interface pass = `1`，K7d fusion pass = `1`，fake/proxy/offload = `0`。
2. P1 true branch-delta residual attribution pass = `1`，dominant residual subphase = `R9-risk_support_component_compute`，ratio = `0.568152`，unknown fraction = `0.0`。
3. P2 true branch-delta correctness / legality pass = `1`：best legal correctness candidate = `TBD1-LayoutRepairedFullLogitSmallC`，uses true delta = `1`，source gap / formula proxy = `0`。
4. P3 true branch-delta exact signal 仍强：best = `TBD0-V9256CBD0Reference`，AUC = `0.888401`，agreement = `1.0`；但 step ratio = `2.863280 > 1.50`，system pass = `0`。
5. P4 formula/source-gap negative controls 正确拒绝；`PX2` 虽然便宜且 agreement = `1.0`，但使用 source-measured gap input，不能 official。
6. P5 exact reference controller feasibility 未过：best legal reference controller = `C0-AllPassExactReferenceController`，precision = `0.430279`，coverage = `0.041501`，bad-event = `0.418327`。
7. P6 system-legal controller 未运行：原因 `P3_true_delta_system_failed`。
8. P7 oracle support 仍通过：precision = `1.0`，coverage = `0.120040`，bad-event = `0.0`；但 support measurement 仍失败：measured signal strata = `3`，balanced diagnostic rows = `36`。
9. P8-P10 因 `P5_reference_controller_infeasible` 全部 gate-blocked，均以 `not_run` 落盘。
10. 当前 blocker：`exact_reference_controller_infeasible`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9257_true_branch_delta_compute_closure_exact_signal_controller_feasibility.py` | v9.2.57 runner；复现 v9.2.56 boundary，执行 true-delta residual attribution、correctness legality、kernel v2 matrix、proxy audit、exact-reference controller feasibility、support/downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9257_true_branch_delta_compute_closure_exact_signal_controller_feasibility.py
```

正式运行：

```bash
python experiments/run_v9257_true_branch_delta_compute_closure_exact_signal_controller_feasibility.py \
  --out-dir results/real_rerun_20260506/v9257_true_branch_delta_compute_closure_exact_signal_controller_feasibility_first_20260512T103200Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
triton_available = true
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 168
interface_events = 24
interface_steps = 2
train_size = 2048
batch_size = 64
hidden_dim = 256
completed_at = 2026-05-12T10:35:38Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R11-ExactSignalControllerInfeasible",
  "base_candidate": "LQ-t2-h256",
  "v9256_boundary_pass": 1,
  "true_delta_residual_attribution_pass": 1,
  "dominant_true_delta_residual_subphase": "R9-risk_support_component_compute",
  "k7d_local_fusion_deployed_end_to_end": 1,
  "true_delta_legality_pass": 1,
  "uses_true_branch_delta": 1,
  "uses_source_measured_gap": 0,
  "uses_formula_proxy": 0,
  "best_true_delta_id": "TBD0-V9256CBD0Reference",
  "true_delta_predictivity_pass": 1,
  "true_delta_agreement_pass": 1,
  "true_delta_system_pass": 0,
  "true_delta_auc": 0.8884008136827201,
  "true_delta_accept_agreement": 1.0,
  "true_delta_step_ratio_q90": 2.8632798851361203,
  "formula_proxy_negative_control_pass": 1,
  "reference_controller_feasible": 0,
  "reference_controller_precision": 0.4302788844621514,
  "reference_controller_coverage": 0.041501322751322754,
  "reference_controller_bad_event": 0.41832669322709165,
  "system_legal_controller_pass": 0,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.12003968253968254,
  "oracle_bad_event": 0.0,
  "support_measurement_pass": 0,
  "primary_blocker": "exact_reference_controller_infeasible",
  "next_required_implementation": "redesign_risk_support_sufficient_statistics"
}
```

判断：v9.2.57 不再只是 “true delta 太贵”。即使使用 exact reference signal，当前 legal reference controller 也无法同时满足 precision / coverage / bad-event gate。

## 3. P1 true branch-delta residual attribution

Artifacts：

```text
p1_true_branch_delta_residual_attribution.csv
true_delta_residual_trace_v9257.csv
```

| subphase | time ms | ratio | note |
|---|---:|---:|---|
| R9-risk_support_component_compute | `1.795203` | `0.568152` | dominant |
| R8-K7d_fused_gain_compute | `0.228825` | `0.072419` | K7d local fusion deployed |
| R3-branch_logits_compute_real_forward_residual | `0.225868` | `0.071483` | train-stream timing residual |
| R4-branch_logits_compute_adamwparallel_forward_residual | `0.225868` | `0.071483` | train-stream timing residual |
| R5-branch_logits_compute_bestlr_forward_residual | `0.225868` | `0.071483` | train-stream timing residual |
| R7-layout_transform_or_copy | `0.073622` | `0.023300` | layout/copy |
| R6-selected_or_full_logit_gather | `0.063816` | `0.020197` | gather |

Summary：

```text
measured_subphase_sum_ms = 3.159720
total_residual_ms_est = 3.159720
unknown_fraction = 0.0
true_delta_residual_attribution_pass = 1
```

判断：P1 residual 已闭合，主要成本不是未解释 overhead；dominant 转为 risk/support component compute，branch-forward residual 也来自真实 train-stream timing residual。

## 4. P2 true branch-delta correctness / legality

Artifacts：

```text
p2_true_branch_delta_correctness_legality_gate.csv
true_delta_correctness_trace_v9257.csv
```

| candidate | legality | agreement | gap error mean | source gap | formula proxy | pass |
|---|---:|---:|---:|---:|---:|---:|
| TBD1-LayoutRepairedFullLogitSmallC | `1` | `1.000000` | `0.000000` | `0` | `0` | `1` |
| TBD2-SelectedLogitExactV2 | `1` | `0.882812` | `0.016775` | `0` | `0` | `0` |
| TBD3-BranchDeltaLogitsPlusK7dFused | `1` | `1.000000` | `0.000000` | `0` | `0` | `1` |
| PX1-FormulaProxyNegativeControl | `0` | `0.882812` | `0.016775` | `0` | `1` | `0` |
| PX2-SourceMeasuredGapInput | `0` | `1.000000` | `0.000000` | `1` | `0` | `0` |

判断：true branch-delta legality gate 通过，但 source-measured gap 与 formula proxy 均被排除，不能作为 official feature。

## 5. P3 true branch-delta kernel v2 matrix

Artifacts：

```text
p3_true_branch_delta_kernel_v2_matrix.csv
true_delta_kernel_v2_trace_v9257.csv
system_true_delta_kernel_overhead_trace_v9257.csv
```

| candidate | true | AUC | agreement | step q90 | system |
|---|---:|---:|---:|---:|---:|
| TBD0-V9256CBD0Reference | `1` | `0.888401` | `1.000000` | `2.863280` | `0` |
| TBD1-LayoutRepairedFullLogitSmallC-event-major | `1` | `0.888401` | `1.000000` | `3.575813` | `0` |
| TBD2-SelectedLogitExactV2 | `1` | `0.589056` | `0.448661` | `1.480000` | `0` |
| TBD3-BranchDeltaLogitsPlusK7dFused | `1` | `0.888401` | `1.000000` | `3.260300` | `0` |
| TBD4-TwoStageExactBorderline | `1` | `0.597459` | `0.455357` | `1.820000` | `0` |
| TBD5-DeltaStatePrepacked | `1` | `0.888401` | `1.000000` | `3.365471` | `0` |
| TBD6-HybridBestLayoutTrueDelta | `1` | `0.888401` | `1.000000` | `3.049958` | `0` |
| PX1-FormulaProxyNegativeControl | `0` | `0.589056` | `0.448661` | `1.350000` | `0` |
| PX2-SourceMeasuredGapInput | `0` | `0.888401` | `1.000000` | `1.020000` | `0` |

Summary：

```text
best_true_delta_id = TBD0-V9256CBD0Reference
true_delta_predictivity_pass = 1
true_delta_agreement_pass = 1
true_delta_system_pass = 0
true_delta_auc = 0.8884008136827201
true_delta_step_ratio_q90 = 2.8632798851361203
```

判断：exact true-delta signal 仍强，但所有 official true route 都没有进入 `step_ratio_q90 <= 1.50` envelope。便宜的 proxy row 被 P4 排除。

## 6. P5 exact reference controller feasibility

Artifacts：

```text
p5_exact_reference_controller_feasibility.csv
exact_reference_controller_trace_v9257.csv
```

| controller | official | AUC heldout | corr | precision | coverage | bad-event | feasible |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0-AllPassExactReferenceController | `1` | `0.894914` | `-0.016454` | `0.430279` | `0.041501` | `0.418327` | `0` |
| C1-RiskFirstExactReferenceController | `1` | `0.539712` | `0.171419` | `0.176944` | `0.030837` | `0.235925` | `0` |
| C2-SupportBalancedExactReferenceController | `1` | `0.866576` | `-0.019727` | `0.401028` | `0.032159` | `0.388175` | `0` |
| C3-ParetoExactReferenceController | `1` | `0.598485` | `0.368486` | `0.195736` | `0.042659` | `0.127907` | `0` |
| C4-Oracle | `0` | `1.000000` | `1.000000` | `1.000000` | `0.120040` | `0.000000` | `0` |

Summary：

```text
reference_controller_feasible = 0
best_reference_controller_id = C0-AllPassExactReferenceController
reference_controller_precision = 0.4302788844621514
reference_controller_coverage = 0.041501322751322754
reference_controller_bad_event = 0.41832669322709165
oracle_gap_to_legal = 0.5697211155378485
```

判断：这是本轮 terminal blocker。即便 exact reference signal 有高 AUC，legal reference controller 仍无法在 heldout 上同时满足 precision、coverage、bad-event gate；不能把 oracle row 写成 feasible controller。

## 7. P6/P7 support 与 downstream boundary

P6 system legal controller：

```text
status = not_run
reason = P3_true_delta_system_failed
system_legal_controller_pass = 0
```

P7 support：

```text
natural_real_event_count = 12096
balanced_diagnostic_real_event_count = 36
measured_signal_strata_count = 3
measured_family_count = 45
support_measurement_pass = 0
oracle_support_pass = 1
oracle_precision = 1.0
oracle_coverage = 0.12003968253968254
oracle_bad_event = 0.0
```

P8-P10 均落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p8_leave_dataset_and_stratum_out.csv` | `P5_reference_controller_infeasible` |
| `p9_official_paired_replay.csv` | same |
| `p10_short_run_functional_validation.csv` | same |

判断：oracle support 仍存在，但 support strata 仍过窄，且 exact reference controller infeasible；因此不能打开 LDO/LSO、paired replay 或 short/full validation。

## 8. No-fake audit

```text
rows_checked = 24374
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
| plan | `6bc681909a73f2da4310817e536ea8fd2b69621beccf6b743bd009146e23a71d` |
| runner | `43c0b811dc7a287724bbe6f0ec4b7afc46ed84b4a45736f4ae18261fcfe252e1` |
| run manifest | `263e5ea38c30d65607438cbcf870a1b09f5baca1e8437ced4e966bb345419b9c` |
| route | `bea58b40aa390a80465ce8b0bf88ef2a791c2a05b093b399c70d489ff89a20b7` |
| P0 boundary | `4ad30db6c052714c5fd82d0387d54eed990ef50c939342765947caebfdf42f58` |
| P1 residual attribution | `080d742d8525febe63df9cd5c40223fd62e13957ed25eb69b33e1290f712847a` |
| P2 correctness legality | `a59ca6a6727e353b038e4cbefd05161866d8d1807f3151b3644c683db50c6f1e` |
| P3 true delta kernel | `8713560c9f74d433e8494b18e88b28b08a9015b85f351828eb0e1a63c800c620` |
| P4 proxy audit | `406e54567862cd9d338e73b1f310ab6ab4e28cc95702b8b720945d6fe7473acb` |
| P5 reference controller | `8217dd9f043c3404b0ed5e3a3dc09e57f2d0174533ce8250f45b1e62c75c39c2` |
| P6 system controller | `b75ebfce1f5875dad3b69d87f2a4d12f91375ac41e60ec7851e7e17aecc3754d` |
| P7 support | `95e03826521c7e7790086b86d0b963e49d5bdfb8020ce05974250cad57e4d147` |
| P8 boundary | `335fcc6c535954e3f7af7baf2926bebad8abea7446bd939f615cb21cc8a82f30` |
| P9 boundary | `3993b6fb2b1209ac89408b4fd52dc7226d8a66bc635e0471ace101783235e8ce` |
| P10 boundary | `60f3d55e139736f6fa38811f215db66c0add3ecfa3afc56880f056ce9b7923fb` |
| failure table | `2b24b577c36330abb7d5e01b0e0aa2b69129bbce9fbd580eb4fe4062224b280d` |
| provenance audit | `43179e3c4d2d17b535ae00b022930d77026df54cabc6d76a86b532dfc0b9e045` |

## 10. 最终分析结论

v9.2.57 的真实推进是：

```text
v9.2.56: true branch-delta tensor interface 与 K7d local fusion 已推进，
          但 exact true-delta path 仍太贵。
v9.2.57: true-delta residual attribution 闭合；
          correctness / legality gate 继续过；
          但 exact reference controller feasibility 明确失败。
```

机制判断：

1. H1 得到推进：residual attribution 没有留 unknown overhead，dominant 是 `R9-risk_support_component_compute`。
2. H2 仍成立但不足：true branch-delta exact signal 继续保持 AUC `0.888401` 和 agreement `1.0`，但 system gate 未过。
3. H3 是本轮关键否定结果：即使把 exact reference signal 当作上界，best legal reference controller 的 precision 只有 `0.430279`，bad-event 达到 `0.418327`。
4. H4 继续成立：oracle support 没有消失，precision `1.0`，coverage `0.120040`，bad-event `0.0`。
5. 当前不应继续只做 kernel 化或 selected-logit threshold patch；下一步需要重做 risk/support sufficient statistics 与 accepted-region geometry，否则 exact signal 也无法转成 legal controller。

最终一句话：

> v9.2.57 真实执行后停在 `R11-ExactSignalControllerInfeasible`：true branch-delta signal 仍强且 legality 过，但 exact reference controller 本身无法满足 heldout safety gate，strict PureKAN functional 仍未成功。
