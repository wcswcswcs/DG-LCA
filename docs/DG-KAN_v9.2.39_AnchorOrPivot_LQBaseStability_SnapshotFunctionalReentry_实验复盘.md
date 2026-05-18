# DG-KAN v9.2.39 Anchor-or-Pivot LQ Base Stability 与 Snapshot Functional Re-entry 实验复盘

> 本复盘记录 `DG-KAN_v9.2.39_AnchorOrPivot_LQBaseStability_SnapshotFunctionalReentry_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R5-BaseRepairPass
base_candidate = LQ-t2-h256
success_v9239_strict_purekan_functional = False
success_v9239_full_functional = False
success_v9239_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z
```

核心结论：

1. P0 复现 v9.2.38 boundary：source route = `R1-LQBaseAnchorBroken`，fake/proxy = `0`。
2. P1 miss-row autopsy 已归因：miss rows = `3`，primary = `M1-KAN_acc_dropped`，borderline rate = `0.3333333333333333`。
3. P2 protocol audit：protocol mismatch detected = `1`，unresolved = `0`，MLP drift = `0`。
4. P3 Current LQ repeated re-anchor：near-rate mean/min/max = `0.666667/0.666667/0.666667`，macro mean = `-0.004500`。
5. P5 global base repair pass = `1`，best = `R2-LQ-fanin-output-scale-confirmed`。
6. 当前 blocker：`snapshot_late_attach_not_implemented_after_base_repair_pass`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry.py` | v9.2.39 runner；生成 P0-P14 artifacts、anchor/protocol/repeat/base-repair audit、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry.py
```

正式运行：

```bash
python experiments/run_v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry.py \
  --out-dir results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z \
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

```json
{
  "route": "R5-BaseRepairPass",
  "base_candidate": "LQ-t2-h256",
  "v9238_boundary_pass": 1,
  "source_route": "R1-LQBaseAnchorBroken",
  "dataset_tuning_detected": 0,
  "lq_anchor_pass": 0,
  "lq_anchor_failure_mode": "M1-KAN_acc_dropped",
  "current_lq_near_rate": 0.6666666666666666,
  "current_lq_macro_delta": -0.004499991734822591,
  "historical_current_lq_delta": -0.0004999770058525931,
  "miss_row_count": 3,
  "miss_row_borderline_rate": 0.3333333333333333,
  "protocol_mismatch_detected": 1,
  "protocol_mismatch_unresolved": 0,
  "mlp_match_drift_detected": 0,
  "seed_protocol_drift_detected": 1,
  "current_lq_rerun_count": 3,
  "current_lq_repeated_near_rate_mean": 0.6666666666666666,
  "current_lq_repeated_near_rate_min": 0.6666666666666666,
  "current_lq_repeated_near_rate_max": 0.6666666666666666,
  "current_lq_repeated_macro_delta_mean": -0.004499991734822591,
  "miss_row_stability": 1.0,
  "base_repair_required": 1,
  "base_repair_pass": 1,
  "base_repair_survivor_count": 3,
  "best_base_candidate": "R2-LQ-fanin-output-scale-confirmed",
  "best_base_macro_delta": -0.0030555526415506997,
  "best_base_near_rate": 1.0,
  "best_base_step_q90": 1.0129602004984652,
  "snapshot_attach_pass": 0,
  "checkpoint_inactive_equivalence_pass": 0,
  "no_event_replay_preservation_pass": 0,
  "functional_carrier_pass": 0,
  "value_observability_pass": 0,
  "leave_dataset_out_pass": 0,
  "leave_stratum_out_pass": 0,
  "paired_replay_pass": 0,
  "short_run_pass": 0,
  "full_run_pass": 0,
  "functional_task_safe": 0,
  "functional_control_pass": 0,
  "functional_system_pass": 0,
  "hard_stratum_repair_pass": 0,
  "adamw_fullpass": 0,
  "strong_baseline_pass": 0,
  "robustness_pass": 0,
  "external_ready": 0,
  "primary_blocker": "snapshot_late_attach_not_implemented_after_base_repair_pass",
  "next_required_implementation": "implement_snapshot_late_attach_on_best_base_repair",
  "success_v9239_strict_purekan_functional": 0,
  "success_v9239_full_functional": 0,
  "success_v9239_external_ready": 0,
  "rows_checked": 196,
  "fake_proxy_nonzero_count": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true,
  "completed_at": "2026-05-11T11:11:13Z"
}
```

## 3. P1 LQ anchor miss-row autopsy

| dataset | seed | current delta | historical delta | near margin | KAN drift | MLP drift | mode |
|---|---:|---:|---:|---:|---:|---:|---|
| MNIST | `2` | `-0.016000` | `-0.004500` | `-0.006000` | `-0.008500` | `0.003000` | `M1-KAN_acc_dropped` |
| KMNIST | `0` | `-0.011500` | `-0.012500` | `-0.001500` | `0.006500` | `0.005500` | `M4-threshold_borderline` |
| KMNIST | `1` | `-0.012500` | `-0.000500` | `-0.002500` | `-0.005000` | `0.007000` | `M3-both_changed` |

判断：miss rows 都有明确 attribution；不是把 `6/9` 直接写成 LQ 崩溃。只有 near-threshold rows 计入 brittleness，未测字段保留为 `not_measured`。

## 4. P3 repeated re-anchor

```text
reruns = 3
near_rate_mean = 0.6666666666666666
near_rate_min = 0.6666666666666666
macro_delta_mean = -0.004499991734822591
miss_row_stability = 1.0
```

判断：Current LQ 在当前协议下 repeated anchor 没有达到 `min near-pass >= 0.80`，因此不能直接打开 snapshot late attach。

## 5. P5 global base repair

| candidate | P4 system | near rows | macro delta | step q90 | repair pass |
|---|---:|---:|---:|---:|---:|
| R0-LQ-current | `1` | `8/9` | `-0.003389` | `1.103946` | `1` |
| R2-LQ-fanin-output-scale-confirmed | `1` | `9/9` | `-0.003056` | `1.012960` | `1` |
| R3-LQ-orthogonal-lift-init | `1` | `7/9` | `-0.005500` | `1.014484` | `0` |
| R4-LQ-hidden224 | `0` | `5/9` | `-0.008389` | `1.907358` | `0` |
| R5-LQ-hidden256 | `1` | `8/9` | `-0.003389` | `1.112380` | `1` |
| R6-LQ-hidden288 | `1` | `7/9` | `-0.004556` | `1.167020` | `0` |
| R8-LQ-legendre2-loworder | `1` | `3/9` | `-0.014889` | `1.204660` | `0` |

判断：P5 是全局 base repair，不使用 dataset-specific route。P6-P14 只有在 base/attach/value gates 可用时打开；本轮未打开阶段均以 `not_run` row 落盘。

## 6. No-fake audit

```text
rows_checked = 196
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 7. Hash

| artifact | SHA256 |
|---|---|
| runner | `85663bb46f596410b21a6a4467fa5d722be79563b9699e327dbba9ae4d4f7ec3` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/run_manifest.json` | `6c836da70a196a38f015c56a1461139b94188113d5c364b8814036343eda5469` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/contract_audit_v9239.csv` | `58b36cb22d6c75c9bb70bf6b595bfa6e804b8cb271f924700d06a212b659be08` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p0_v9238_boundary_reproduction.csv` | `8cc99089d40a2edb4cec081063dd625674739f29c7a9f835e9bab05139f8b8d6` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p1_lq_anchor_miss_row_autopsy.csv` | `05dcdb7cd64df8a3e872a589bf783a99663b62c00e4199b45c393009d3644eaf` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p2_protocol_hash_and_mlp_match_audit.csv` | `da9738237ed25a7749a5055f8ad5965d36370441ad1ff3d6a7a4e5d212d1efb8` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p3_current_lq_repeated_reanchor.csv` | `b5625e03d7c12ef4a4c600581ac95899a46f15e7b77b0c29471a243772273351` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p4_anchor_decision_and_base_repair_gate.csv` | `d73966d43ffbe3e0d4cddb512a6cc72b0cb5065a6cbfe191af369c7d40b38217` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p5_global_lq_base_repair.csv` | `2eb2bccd87c43332f9554da4f49355e0b1969d2867262957056d39955ea6d793` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p6_snapshot_late_attach_implementation.csv` | `032b5f5f64fe72347406559d8ae7f84bc263d52a4ec890d5d5dc977f629e8c22` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p7_checkpoint_inactive_noevent_equivalence.csv` | `6e869438c1ea4d5ee6e8d3ba2e825137936b0efa6267f4ac938bcaa2ee55714a` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p8_functional_carrier_actuatability.csv` | `7aba6d5bf787d7c20264abf243f5c2db36696aae6f215aebbab592188abdfc84` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p9_value_observability_audit.csv` | `aad986c6d954fa244af1a642599331f9e92cefff43de81203c673d588e8d4b0a` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p10_leave_dataset_and_stratum_out_validation.csv` | `a7a89bf6dcc65748d8c5ecf5af6c622f75ba7250ee17c372a8352740a6e6e45f` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p11_official_snapshot_late_attach_paired_replay.csv` | `85fc4f4fbc22aa1de03351715d33f541c4c0c2b4d61e7e3811ed9c4ed246ff4e` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p12_short_run_functional_validation.csv` | `2d698d58978c48da9718ef3f872fb608a3f19e9747a80e0485f7de7cea17624c` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p13_full_10seed_functional_validation.csv` | `d4ae20c7c98b8c25858db9b02f302597e8fbc7ed8a80f4d9afc2117ef9f9e679` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/p14_robustness_external_ready.csv` | `d92e207a79aaa863b3d28ea26f35c83364a348c61642339eb9de664051724103` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/lq_anchor_trace_v9239.csv` | `bdefc0190bca9bc946f1db80ebb2ffaafe2e467c2e06aa6f3188cc1f52668fb9` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/miss_row_trace_v9239.csv` | `bae6e41099ed20c74ffab4110d453f5c58f8c742c1cef4b554aa1dc28c7b6838` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/protocol_hash_diff_v9239.csv` | `da9738237ed25a7749a5055f8ad5965d36370441ad1ff3d6a7a4e5d212d1efb8` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/snapshot_attach_trace_v9239.csv` | `82adc73bc62b667924713304bd2ad63f150c7a3f8142367a75f83c0c3bbf1d8b` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/functional_carrier_trace_v9239.csv` | `e9edb1699b95d0e9fa84339694a69ca44a9e6e75bec32d0ce36aeb71a2b12eff` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/value_score_trace_v9239.csv` | `057b6a0fb46e6e691917d26eb03900ea1fd290ebe11b2467564bf0e4afd52a4a` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/leave_dataset_out_trace_v9239.csv` | `68ff22b6a21b69af26dcf5053ea29f4892803ae2968c419ce3482a56c8e1a352` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/paired_replay_branch_trace_v9239.csv` | `2f39e698536d54e7f73665481400e5354f40b854f215aea8191172d8fe76a02e` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/route_decision.json` | `da6a8af151674629f7548a0039f523442cb0e1f9e08981dbbc2eda923a0e8115` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/aggregate_decision.json` | `da6a8af151674629f7548a0039f523442cb0e1f9e08981dbbc2eda923a0e8115` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/failure_table.csv` | `9cd6f65d08f4cfa473f27cd02a791c9460ddd57b59622da908fea98e848a4852` |
| `results/real_rerun_20260506/v9239_anchor_or_pivot_lq_base_stability_snapshot_functional_reentry_first_20260511T120000Z/v9239_provenance_audit.csv` | `b8db9c7acf457315d2c46f564785a061fba91f0f352377bec63caa2de4d6fdad` |

## 8. 最终分析结论

v9.2.39 的真实推进是：

```text
v9.2.38: current LQ base anchor failed at 6/9 near-pass.
v9.2.39: miss-row attribution, protocol audit, repeated current anchor, and global base repair are separated and audited.
```

机制判断：

1. 本轮没有在未锚定 base 上推进 functional；snapshot late attach、carrier、value、paired replay 都由 gate 控制。
2. Current LQ 的 macro 与 historical 接近，但 repeated current protocol 仍不满足 row-level anchor gate。
3. 如果 P5 找到 global base repair survivor，它只是恢复 base 地基；strict PureKAN functional 仍要从 snapshot attach 重新进入。
4. 如果 P5 未找到 survivor，则 LQ-based functional route 必须 pivot 到 base repair / deeper primitive reset。

最终一句话：

> v9.2.39 真实执行后停在 `R5-BaseRepairPass`：`snapshot_late_attach_not_implemented_after_base_repair_pass`。
