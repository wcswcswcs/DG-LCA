# DG-KAN v9.2.53 D0 Vectorized Controller-Prep 与 Real Fused Branch-Delta Kernel 实验复盘

> 本复盘记录 `DG-KAN_v9.2.53_D0VectorizedControllerPrep_RealFusedBranchDeltaKernel_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R12-FusedBranchDeltaPredictiveButNeedsCustomKernel
base_candidate = LQ-t2-h256
success_v9253_strict_purekan_functional = False
success_v9253_full_functional = False
success_v9253_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9253_d0_vectorized_controller_prep_real_fused_branch_delta_kernel_first_20260512T050000Z/
```

核心结论：

1. P0 复现 v9.2.52 boundary：source route = `R10-FusedDeltaPredictiveButTooExpensive`，fake/proxy = `0`。
2. 本轮 manifest 记录 `device = cuda`；运行中 GPU 采样有活动：`NVIDIA L4, 31-33 %, 326 MiB`。
3. P1 D0 micro-attribution pass = `1`，dominant subphase = `D0a-role_scalar_compute`，ratio = `0.323653`。
4. P2 D0 vectorization pass = `1`：best = `D0V6-D0VectorizedAll`，decision agreement = `0.999917`，D0 time ratio = `0.022203`；但 system step ratio = `3.375423`，未过 `<=1.50`。
5. P3 real fused branch-delta audit 有 implemented candidate：best = `FBD6-D0VectorizedFusedDelta`，AUC = `0.888401`，agreement = `1.0`，correctness pass = `1`；但 step ratio = `2.863280 > 1.50`，system pass = `0`。
6. P4 selected-logit exactness 未过：best reference = `SLR0-V9252-SLD0-reference`，AUC/agreement 保留但 step ratio = `6.139253`；低成本 selected repair 仍没有 official agreement/safety。
7. P5 candidate generator 未过：best = `CG0-V9252-CG1-Reference`，recall = `0.576766`，candidate bad-event = `0.251535`。
8. P6 cascade controller 未过：best = `C4-D0VectorizedParetoController`，precision = `0.152439`，coverage = `0.040675`，bad-event = `0.256098`。
9. P7 oracle support 仍通过：precision = `1.0`，coverage = `0.121693`，bad-event = `0.0`；但 measured signal strata = `3`，balanced diagnostic rows = `36`，support measurement pass = `0`。
10. 当前 blocker：`fused_branch_delta_predictive_but_not_system_legal`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9253_d0_vectorized_controller_prep_real_fused_branch_delta_kernel.py` | v9.2.53 runner；重新生成 real train-stream rows，执行 D0 micro-attribution、D0 vectorized prep、real fused branch-delta audit、selected-logit repair、candidate/controller/support gates、route/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9253_d0_vectorized_controller_prep_real_fused_branch_delta_kernel.py
```

正式运行：

```bash
python experiments/run_v9253_d0_vectorized_controller_prep_real_fused_branch_delta_kernel.py \
  --out-dir results/real_rerun_20260506/v9253_d0_vectorized_controller_prep_real_fused_branch_delta_kernel_first_20260512T050000Z \
  --fresh --device auto --data-root data --seed 1314
```

运行中 GPU 采样：

```text
NVIDIA L4, 32 %, 326 MiB
NVIDIA L4, 31 %, 326 MiB
NVIDIA L4, 33 %, 326 MiB
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R12-FusedBranchDeltaPredictiveButNeedsCustomKernel",
  "base_candidate": "LQ-t2-h256",
  "v9252_boundary_pass": 1,
  "d0_subphase_attribution_pass": 1,
  "dominant_d0_subphase": "D0a-role_scalar_compute",
  "d0_vectorization_pass": 1,
  "d0_time_ratio_vs_ref": 0.022203195016300085,
  "d0_decision_agreement": 0.999917328042328,
  "real_fused_branch_delta_implemented": 1,
  "best_fused_branch_delta_id": "FBD6-D0VectorizedFusedDelta",
  "fused_branch_delta_predictivity_pass": 1,
  "fused_branch_delta_correctness_pass": 1,
  "fused_branch_delta_system_pass": 0,
  "fused_branch_delta_auc": 0.8884008136827201,
  "fused_branch_delta_accept_agreement": 1.0,
  "fused_branch_delta_step_ratio_q90": 2.8632798851361203,
  "candidate_generator_pass": 0,
  "cascade_controller_pass": 0,
  "oracle_support_pass": 1,
  "primary_blocker": "fused_branch_delta_predictive_but_not_system_legal"
}
```

判断：D0 vectorization 与 exact-like fused signal 都有真实推进，但 fused branch-delta 仍未进入 system envelope，所以 route 必须停在 `R12`，不能打开 LDO/LSO 或 paired replay。

## 3. P1/P2 D0 attribution 与 vectorization

Artifacts：

```text
p1_d0_controller_scalar_prep_micro_attribution.csv
d0_subphase_trace_v9253.csv
p2_d0_vectorized_controller_prep.csv
d0_vectorization_trace_v9253.csv
```

P1 summary：

```text
profile_event_count = 9
dominant_d0_subphase = D0a-role_scalar_compute
dominant_d0_subphase_ratio = 0.3236526626119091
d0_subphase_attribution_pass = 1
unknown_fraction = 0.0
```

P2 summary：

```text
best_d0_vector_id = D0V6-D0VectorizedAll
d0_time_ratio_vs_ref = 0.022203195016300085
d0_decision_agreement = 0.999917328042328
d0_vectorization_pass = 1
d0_step_ratio_q90 = 3.3754232392687697
d0_vectorization_system_pass = 0
```

判断：D0 vectorization 成功降低 D0 local prep cost，并保持 decision agreement；但单靠 D0 vectorization 不足以让整条 online branch-delta path 过 `step_ratio_q90 <= 1.50`。

## 4. P3 real fused branch-delta

Artifact：

```text
p3_real_fused_branch_delta_implementation.csv
real_fused_branch_delta_trace_v9253.csv
system_fused_delta_overhead_trace_v9253.csv
```

| candidate | implemented/status | AUC | agreement | step ratio | system |
|---|---|---:|---:|---:|---:|
| FBD0-SLD0FullLogitExactReference | measured full branch forward | `0.888401` | `1.000000` | `6.139253` | `0` |
| FBD1-FDP5RealFusedFunctionalDeltaPrep | implemented torch vectorized, not custom CUDA | `0.888401` | `1.000000` | `4.000767` | `0` |
| FBD6-D0VectorizedFusedDelta | implemented D0-vectorized delta-view, not single kernel | `0.888401` | `1.000000` | `2.863280` | `0` |

判断：`FBD6` 是本轮 best，signal / agreement / correctness 都过，但仍太贵。它不是 fake row，也没有被写成 custom fused kernel success；route 正确进入 “predictive but needs custom kernel”。

## 5. P4-P6 controller boundary

P4 selected-logit exactness：

```text
best_selected_repair_id = SLR0-V9252-SLD0-reference
selected_repair_auc = 0.8884008136827201
selected_repair_agreement = 1.0
selected_repair_step_ratio_q90 = 6.1392531712210054
selected_logit_exactness_pass = 0
```

P5 candidate generator：

```text
best_candidate_generator_id = CG0-V9252-CG1-Reference
safe_good_recall = 0.576766304347826
candidate_rate = 0.3500330687830688
candidate_bad_event = 0.25153519130845536
candidate_generator_pass = 0
```

P6 cascade controller：

```text
best_controller_id = C4-D0VectorizedParetoController
controller_auc = 0.5135211501703648
controller_corr = 0.21575694691235842
accepted_precision = 0.1524390243902439
accepted_coverage = 0.040674603174603176
accepted_bad_event_rate = 0.25609756097560976
cascade_controller_pass = 0
```

判断：controller 层没有因为 D0 vectorization 自动闭合。当前 legal controller precision 和 bad-event 明显不过 gate；同时 P3 system gate 已经失败，因此 P6 不允许转正。

## 6. P7 support 与 downstream boundary

P7 support：

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

P8-P12 均落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p8_leave_dataset_and_stratum_out.csv` | `P3_fused_branch_delta_system_failed` |
| `p9_official_paired_replay.csv` | same |
| `p10_short_run_functional_validation.csv` | same |
| `p11_full_10seed_functional_validation.csv` | same |
| `p12_robustness_external_ready.csv` | same |

没有把 oracle support、exact branch signal、D0 vectorization 或 diagnostic controller 倒灌成 LDO/LSO、paired replay、short-run、full-run 或 external-ready success。

## 7. No-fake audit

```text
rows_checked = 24424
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
| plan | `2868ba5d903d0f2fcb3398771ce253dfa6330182d1d762c855ed7874969895d9` |
| runner | `f2796b61a2af14db83ed340a58e9ccec2a85cb950962e7c53566661c1a491f00` |
| run manifest | `dfbf1635cfab5a7995624fa1fac013526afa3f8b5adce75cb8f81818f2ea2220` |
| route | `4bcf00341cdcf61abd5a8a6b7c392dd40391d8cf0545e477a641aed40eca3dc6` |
| P0 boundary | `799cd9cbe45e39d615e4766a2105634ab8fec08e67be37c2adcafd47a1811a26` |
| P1 D0 attribution | `5a2c951f722867bb521854fea15069c1e1ed46d67d0f9079e391d3245bc257cb` |
| P2 D0 vectorization | `8933c59e6e703d9887194a49c92f71c9a1518a3367168b7f116f8bdea4debe3b` |
| P3 real fused branch delta | `6261dbc475a80a27e19ebba162317d5af978d316b3105330c1add159542a8d39` |
| P4 selected exactness | `dbeee96582b15b3ba545dceb22eb863a944e865a3e5f2ac03e80e8b976e8260d` |
| P5 candidate generator | `ec9ae814386142491420fccc21e9be5d70dbe1d64ff6e4ce0dcbd6b70818c71b` |
| P6 cascade controller | `99a85c8059a804331a9a962a2b06c13325cb303fe6e81cd2dfb7784d714d52c7` |
| P7 support expansion | `8bc11ab99ebb8c1e3d2ca2c9ef7691135731fe83119f58f9626854aa892d102b` |
| P8 boundary | `b2db1821eac6d9a093efb37542d9bbc42c3c09bec0b4e81c6deb747a931130ef` |
| P9 boundary | `1a9faa9d699b8111b92f4c0cb1d6afe404e8a3204ab90fcc82efa168ae0bb64b` |
| failure table | `e49a557209dfb7408908287f05be92d4cdc9f5a6d7b75eee84a2976eba8e4ca0` |
| provenance audit | `6b4ff90a0102c497b8462c9a3f47a071ba043c9365e2fff076e5c56a071010dc` |

## 9. 最终分析结论

v9.2.53 的真实推进是：

```text
v9.2.52: exact/full branch-delta signal strong, but D0/fused path too expensive.
v9.2.53: D0 controller-prep vectorization succeeds;
          exact-like fused branch-delta signal and agreement remain strong;
          but branch-delta system path still exceeds step-ratio gate.
```

机制判断：

1. H1 得到推进：D0 subphase 已归因，且 D0 vectorized prep 的 decision agreement 达到 `0.999917`。
2. H2 仍未闭合：`FBD6` 保留 exact signal，但 step ratio `2.863280` 仍大于 `1.50`。
3. H3 未成立：candidate recall `0.576766`，bad-event `0.251535`，不能支撑 coverage-preserving cascade。
4. H4 成立：fresh oracle support 仍高，good events 没有消失。
5. 下一步应实现真正 CUDA/Triton fused branch-delta kernel，或把 exact confirmation 的 branch-forward cost 进一步压进 system envelope；继续调 cheap selected-logit threshold 不足以解决当前 blocker。

最终一句话：

> v9.2.53 真实执行后停在 `R12-FusedBranchDeltaPredictiveButNeedsCustomKernel`：D0 vectorization 过了，exact-like fused branch-delta 仍强，但系统路径仍太贵，strict PureKAN functional 仍未成功。
