# DG-KAN v9.2.78 Native Bridge-Accept Numerical Closure 与 Bucketed Runtime Official Promotion 实验复盘

> 本复盘记录 `DG-KAN_v9.2.78_NativeBridgeAccept_StaticBucketOfficialClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 stable-accept 局部闭合写成 official system pass。

## 0. 最新结论

```text
route = R18-StableAcceptLocalClosedButSystemNotOfficial
base_candidate = LQ-t2-h256
success_v9278_strict_purekan_functional = False
success_v9278_full_functional = False
success_v9278_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9278_native_bridge_accept_static_bucket_official_closure_integrated_stable_native_20260513T160000Z
```

核心结论：

1. P0 复现 v9.2.77 native boundary：source route = `R17-BasisNormRuntimeStillInsufficient`，native CUDA bucket kernel used = `1`，source accept disagreement = `9`，system legal controller pass = `0`。
2. P1 autopsy 显示 current median accept 的 disagreement = `12`，large-margin disagreement = `0`，bridge score max error = `5.960464477539063e-08`，root cause = `median_boundary_fragility`。
3. P2 stable accept contract 最优为 `AC2Q2-stable-quantized-1e5-event-tie`，accept disagreement 从 `12` 降到 `0`，agreement vs current reference accept = `1.0`。
4. P3 native stable-accept path 使用真实 native CUDA kernel：basis_norm/W2_delta/bridge_score/accept_bit inside kernel 均为 `1`，native q90 = `0.6102416664361954`，eager q90 = `1.098200213164091`，q90 reduction = `0.4443256711105608`。
5. P3 数值 gate 已局部闭合：stable accept disagreement = `0`，logits error = `1.811981201171875e-05`，delta error = `3.247987478971481e-08`，tail disagreement = `0`。
6. 但 P6 不能 official：`official_eligible = 0`，system controller pass = `0`，原因是 stable accept 是 rule change 且 full batch-major / full-system step ratio 仍未闭合，step ratio q90 = `2.713295831053225`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9278_native_bridge_accept_static_bucket_official_closure.py` | v9.2.78 runner；复现 v9.2.77 native accept failure，执行 accept disagreement autopsy、stable rank/tie policy、native stable-accept kernel、system controller gate |

代码检查：

```text
python -m py_compile experiments/run_v9278_native_bridge_accept_static_bucket_official_closure.py
```

正式运行：

```bash
python experiments/run_v9278_native_bridge_accept_static_bucket_official_closure.py \
  --out-dir results/real_rerun_20260506/v9278_native_bridge_accept_static_bucket_official_closure_integrated_stable_native_20260513T160000Z \
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R18-StableAcceptLocalClosedButSystemNotOfficial",
  "base_candidate": "LQ-t2-h256",
  "v9277_boundary_pass": 1,
  "payload_binding_contract_pass": 1,
  "quantile_tail_runtime_pass": 1,
  "native_accept_autopsy_pass": 1,
  "accept_disagreement_count_before": 12,
  "accept_disagreement_root_cause": "median_boundary_fragility",
  "large_margin_disagreement_count": 0,
  "borderline_disagreement_fraction_1e_6": 1.0,
  "bridge_score_error_max": 5.960464477539063e-08,
  "logits_error_max": 1.811981201171875e-05,
  "delta_error_max": 3.3527612686157227e-08,
  "tail_disagreement_count": 0,
  "best_accept_contract_id": "AC2Q2-stable-quantized-1e5-event-tie",
  "stable_accept_contract_pass": 1,
  "tie_policy": "stable_rank_event_id_tie",
  "accept_rule_changed": 1,
  "calibration_rerun_required": 1,
  "accept_disagreement_count_after": 0,
  "agreement_reference_accept": 1.0,
  "reference_drift_vs_current_accept": 0,
  "native_cuda_bucket_kernel_used": 1,
  "stable_accept_cuda_kernel_used": 1,
  "basis_norm_bucketed": 1,
  "W2_delta_bucketed": 1,
  "bridge_score_inside_kernel": 1,
  "accept_bit_inside_kernel": 1,
  "kernel_count_before": 3750,
  "kernel_count_after": 216,
  "sync_count_before": 1250,
  "sync_count_after": 24,
  "avg_candidates_per_kernel_after": 11.541666666666666,
  "native_time_ms_q90": 0.6102416664361954,
  "eager_time_ms_q90": 1.098200213164091,
  "q90_reduction": 0.4443256711105608,
  "native_bucket_kernel_v2_pass": 1,
  "controller_precision": 0.8380281690140845,
  "controller_coverage": 0.03130511463844797,
  "controller_bad_event": 0.02464788732394366,
  "controller_null_rate": 0.13028169014084506,
  "controller_step_ratio_q90": 2.713295831053225,
  "official_eligible": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "batch_major_or_full_system_step_ratio_not_closed",
  "next_required_implementation": "integrate_stable_accept_into_batch_major_system_runtime"
}
```

判断：v9.2.78 已经把 v9.2.77 的 native accept bit disagreement 从边界脆弱性中单独剥离，并用稳定 rank/tie policy 在 native CUDA stable-accept kernel 上清零；但它改变了 accept contract，需要 rerun full controller gates，且 batch-major/system step ratio 未闭合，所以不能转正。

## 3. P1 native accept disagreement autopsy

Artifacts：

```text
p1_native_accept_disagreement_autopsy.csv
accept_disagreement_trace_v9278.csv
```

Summary：

```text
accept_disagreement_count = 12
large_margin_disagreement_count = 0
borderline_disagreement_fraction_1e_6 = 1.0
bridge_score_error_max = 5.960464477539063e-08
logits_error_max = 1.811981201171875e-05
delta_error_max = 3.3527612686157227e-08
tail_disagreement_count = 0
root_cause = median_boundary_fragility
```

判断：H1 成立。disagreement 不是 tail/logits/delta 崩坏；它来自 median accept 边界附近的 score 差异。

## 4. P2 stable accept contract

Artifacts：

```text
p2_stable_accept_contract_implementation.csv
stable_accept_contract_trace_v9278.csv
```

Summary：

```text
best_accept_contract_id = AC2Q2-stable-quantized-1e5-event-tie
tie_policy = stable_rank_event_id_tie
accept_rule_changed = 1
calibration_rerun_required = 1
accept_disagreement_before = 12
accept_disagreement_after = 0
agreement_reference_accept = 1.0
reference_drift_vs_current_accept = 0
stable_accept_contract_pass = 1
```

判断：stable rank/event-id tie 能消除 native/reference disagreement，但这是 accept contract change，不能只凭 P2 局部结果 official。

## 5. P3 native bucket kernel v2

Artifacts：

```text
p3_native_bucket_kernel_v2_stable_accept.csv
native_bucket_kernel_v2_trace_v9278.csv
```

Summary：

```text
native_kernel_id = NK1-NativeStableRankIntegratedBucketKernel
native_cuda_bucket_kernel_used = 1
stable_accept_cuda_kernel_used = 1
basis_norm_bucketed = 1
W2_delta_bucketed = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
kernel_count_before/after = 3750 / 216
sync_count_before/after = 1250 / 24
avg_candidates_per_kernel_after = 11.541666666666666
native_time_ms_q90 = 0.6102416664361954
eager_time_ms_q90 = 1.098200213164091
q90_reduction = 0.4443256711105608
accept_disagreement_count = 0
native_bucket_kernel_v2_pass = 1
```

判断：P3 是本轮真实推进。native stable accept 清零了 P16 的 accept disagreement，同时保留了局部 runtime 改善；P3 pass = `1`，但 full system step ratio 仍未闭合。

## 6. P6 system controller boundary

Artifact：

```text
p6_system_legal_exact_signal_controller_v10.csv
```

Boundary：

```text
official_eligible = 0
system_legal_controller_pass = 0
reason = stable_accept_local_closure_not_batch_major_or_full_system_step_ratio_closure
precision_heldout = 0.8380281690140845
coverage_heldout = 0.03130511463844797
bad_event_heldout = 0.02464788732394366
null_rate_heldout = 0.13028169014084506
step_ratio_q90 = 2.713295831053225
full_online_row/payload/update = 1/1/1
diagnostic_derived_from_measured_components = 1
```

判断：decision/support/payload 仍保持 reference frontier，但 stable accept 规则改变后没有 full calibration/heldout rerun，也没有 batch-major full-system runtime closure，因此 P6 仍必须保持 `0`。

## 7. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P6_system_controller_not_official` |
| `p8_official_paired_replay.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_run_robustness_strong_baseline.csv` | same |

没有把 stable accept local closure、native q90 reduction 或 decision metrics 写成 LDO/LSO、paired replay、short/full validation success。

## 8. No-fake audit

```text
rows_checked = see v9278_provenance_audit.csv
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
| plan | `0c0d83daf763bd84c9835c3b74d88c11a796a4c374430553432c63e34dff6de0` |
| runner | `8433667394645bc5e83ba1234247fc39422fd46cdc857ac521f4317a741e1153` |
| run manifest | `9428f14e0bde64e2e3c69d85f82cf7217ab88d4f49c42a9ee8a78814fd73ba56` |
| route | `de58537205e3cbfaf95316a7b1859973596e46b740274260df6e8a02bb6633b3` |
| P1 autopsy | `df9a5ef2f1debe9f2576bc78d8fb41bba0013bf6a11cbefc3644d5d007b5099a` |
| P2 stable accept | `12b0e50ad0cbae9a1d5bf0ee72c1fb22d9232cabcb300841f9654ed8625d35e7` |
| P3 native kernel v2 | `10eba5283e951078019e3dc2ec12d4ee1e1d8f221b3fb48d4687bcd3ec29d2db` |
| P6 system controller | `5bf631e0ce7dea338eea6ed29ce06f0aefce560fa216bfa52eaeb8ce9016fe32` |
| provenance audit | `a3bbc9cabb315e2c53a420e117e08af14b9bfeb0dd8e16db392d34582ecb6de0` |

## 10. 最终分析结论

v9.2.78 的真实推进是：

```text
v9.2.77: native CUDA bucket kernel 已有 q90 改善，
          但 accept bit 在 median 边界发生 9-10 个 disagreement。
v9.2.78: autopsy 确认 disagreement 来自边界脆弱性；
          stable rank/event-id tie + native stable-accept kernel 将 disagreement 清零；
          但该 accept contract change 还未完成 full calibration/heldout/system runtime promotion。
```

机制判断：

1. H1 成立：accept disagreement 是 median-boundary fragility，不是 true-delta 数值崩坏。
2. H2 局部成立：stable rank/event-id tie 可消除 native/reference disagreement。
3. H4 局部成立：native bucket kernel v2 保留局部 q90 reduction，P3 pass = `1`；但 full system step ratio 仍没过。
4. H5 触发：accept rule changed = `1`，因此必须 rerun calibration/heldout/support gates 后才能 official。
5. H6 未打开：system controller 还没过，不能讨论 paired replay functional value target。

最终一句话：

> v9.2.78 真实执行后停在 `R18-StableAcceptLocalClosedButSystemNotOfficial`：native bridge accept 的边界数值问题已被 stable tie policy 局部解决，但这不是 official system closure；下一步需要把 stable accept contract 作为正式 controller rule 重跑 calibration/heldout/support，并做 batch-major native runtime，让 step ratio 进入 `1.50` envelope。
