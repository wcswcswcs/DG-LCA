# DG-KAN v9.2.40 Parallel Base-Repair Confirmation 与 Snapshot Functional Re-entry 实验复盘

> 本复盘记录 `DG-KAN_v9.2.40_ParallelBaseRepairConfirmation_SnapshotFunctionalReentry_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R11-BasePreservedButValueUnobservable
base_candidate = LQ-t2-h256
success_v9240_strict_purekan_functional = False
success_v9240_full_functional = False
success_v9240_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z
```

核心结论：

1. P0 复现 v9.2.39 boundary：source route = `R5-BaseRepairPass`，best repair = `R2-LQ-fanin-output-scale-confirmed`，fake/proxy = `0`。
2. P1 repaired base robust pass = `1`，best = `R2-LQ-fanin-output-scale-confirmed`，near rate = `0.8`。
3. P2 snapshot attach pass = `1`，implemented count = `5`。
4. P3 inactive equivalence pass = `1`；P4 no-event preservation pass = `1`。
5. P5 functional carrier pass = `1`，max r_z_tail = `0.30905587311056065`，max r_perp_tail = `0.2706856067015393`。
6. P6 value observability pass = `0`，AUC = `0.44358974358974357`，corr = `-0.25921111692352256`。
7. 当前 blocker：`value_observability_fail`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9240_parallel_base_repair_confirmation_snapshot_functional_reentry.py` | v9.2.40 runner；生成 P0-P11 artifacts、parallel base confirmation、snapshot attach、carrier/value scout、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9240_parallel_base_repair_confirmation_snapshot_functional_reentry.py
```

正式运行：

```bash
python experiments/run_v9240_parallel_base_repair_confirmation_snapshot_functional_reentry.py \
  --out-dir results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z \
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

```json
{
  "route": "R11-BasePreservedButValueUnobservable",
  "base_candidate": "LQ-t2-h256",
  "v9239_boundary_pass": 1,
  "source_route": "R5-BaseRepairPass",
  "source_best_base_repair": "R2-LQ-fanin-output-scale-confirmed",
  "dataset_tuning_detected": 0,
  "best_repaired_base": "R2-LQ-fanin-output-scale-confirmed",
  "repaired_base_robust_pass": 1,
  "base_robust_survivor_count": 1,
  "repaired_base_near_rate": 0.8,
  "repaired_base_macro_delta": -0.004816665252049764,
  "repaired_base_ci95_low": -0.007560294354175599,
  "repaired_base_step_q90": 1.0149251371288794,
  "repaired_base_memory_ratio": 0.9695007261731864,
  "snapshot_attach_pass": 1,
  "snapshot_attach_implemented_count": 5,
  "best_late_attach_candidate": "A2-LateAttachControlGapChannel",
  "checkpoint_inactive_equivalence_pass": 1,
  "max_logit_diff_inactive": 0.0,
  "no_event_replay_preservation_pass": 1,
  "max_no_event_logit_diff": 0.0,
  "max_no_event_CEp99_abs_diff": 0.0,
  "max_no_event_margin_abs_diff": 0.0,
  "functional_carrier_pass": 1,
  "max_r_z_tail": 0.30905587311056065,
  "max_r_perp_tail": 0.2706856067015393,
  "carrier_bad_event_rate": 0.044444444444444446,
  "value_observability_pass": 0,
  "value_auc": 0.44358974358974357,
  "value_corr": -0.25921111692352256,
  "accepted_precision": 0.08695652173913043,
  "accepted_coverage": 0.10222222222222223,
  "accepted_bad_event_rate": 0.043478260869565216,
  "leave_dataset_out_pass": 0,
  "leave_stratum_out_pass": 0,
  "paired_replay_pass": 0,
  "short_run_pass": 0,
  "full_run_pass": 0,
  "functional_task_safe": 1,
  "functional_control_pass": 0,
  "functional_system_pass": 1,
  "hard_stratum_repair_pass": 0,
  "adamw_fullpass": 0,
  "strong_baseline_pass": 0,
  "robustness_pass": 0,
  "external_ready": 0,
  "primary_blocker": "value_observability_fail",
  "next_required_implementation": "redesign_control_gap_or_rolewise_value_score",
  "success_v9240_strict_purekan_functional": 0,
  "success_v9240_full_functional": 0,
  "success_v9240_external_ready": 0,
  "rows_checked": 3170,
  "fake_proxy_nonzero_count": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true,
  "completed_at": "2026-05-11T11:49:58Z"
}
```

## 3. P1 Parallel base robust confirmation

| candidate | rows | near rate | macro delta | CI95 low | step q90 | memory | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0-LQ-current | `30` | `0.733333` | `-0.005767` | `-0.008577` | `8.051914` | `0.969501` | `0` |
| R2-LQ-fanin-output-scale-confirmed | `30` | `0.800000` | `-0.004817` | `-0.007560` | `1.014925` | `0.969501` | `1` |
| R5-LQ-hidden256 | `30` | `0.733333` | `-0.005767` | `-0.008577` | `1.102015` | `0.969501` | `0` |

判断：P1 是 global base confirmation；没有 dataset-specific promotion。

## 4. P2-P4 snapshot attach equivalence

```text
snapshot_attach_pass = 1
checkpoint_inactive_equivalence_pass = 1
max_logit_diff_inactive = 0.0
no_event_replay_preservation_pass = 1
max_no_event_logit_diff = 0.0
```

判断：attach 只在 repaired base robust pass 后计入 official gate；inactive/no-event 均按 checkpoint/base continuation 对比。

## 5. P5/P6 carrier 与 value scout

P5 summary：

```text
row_count = 1125
official_eligible = 1
max_r_z_tail = 0.30905587311056065
max_r_perp_tail = 0.2706856067015393
carrier_bad_event_rate = 0.044444444444444446
functional_carrier_pass = 1
```

P6 summary：

```text
corr = -0.25921111692352256
auc = 0.44358974358974357
precision = 0.08695652173913043
coverage = 0.10222222222222223
bad_event_rate = 0.043478260869565216
value_observability_pass = 0
```

判断：P5/P6 可以 diagnostic 先测，但只有 P1-P4 pass 后才 official eligible；本轮 route 没有越过 failed gate。

## 6. Downstream boundary

P7-P11 只有在 P6 value observability pass 后打开。本轮未打开阶段均以 `not_run` row 落盘。

## 7. No-fake audit

```text
rows_checked = 3170
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
| runner | `444dc003f6d470f8f8f57b59b2238a93c4817bfbe062d6ad892fa155a4edaf4e` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/run_manifest.json` | `201863f7e581c3ab3f38ab4bb87c5266b34627994a059b4297df736f6097773f` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/contract_audit_v9240.csv` | `ecd6ead0946d19198453e11ca63f91544d801455e0093d7eb3392ec3e707b3ca` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/p0_v9239_boundary_reproduction.csv` | `b0287a04a2f48939356bad681a13ad5ae002c1f4ee8a653d3b8d9a41799790b2` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/p1_parallel_base_robust_confirmation.csv` | `15e5c22ac8b35f145c5572d3bca55fa79c01f01208b5d4dfd33fb4be95e78db6` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/p2_snapshot_late_attach_implementation.csv` | `245c3d1ee475989e0670a06308f9d181979dbc14c856e283b76ed4bddd4ff0e6` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/p3_checkpoint_inactive_equivalence.csv` | `5835d26d38cfa1478c3321aaecc1171d52bc6a8b5809404e929030514c4b4743` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/p4_no_event_replay_preservation.csv` | `e4d11f3b98133e0f5ffbd4b3381256b61f7517b45344d35d88cd3a0d10a697c1` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/p5_functional_carrier_actuatability.csv` | `e886f8650e7fdb243723935364fcc01700497d053bc0c0dd1f98f702bd85a149` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/p6_value_observability_audit.csv` | `312033a1420820b9d84005a018dcefd845c2f4f036be404d940b64c69aa031a5` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/p7_leave_dataset_and_stratum_out_validation.csv` | `52280c8344f3d0c55ba1eb18c911a03ab91b6c6078f6b663bb43b224a9e3ded6` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/p8_official_snapshot_late_attach_paired_replay.csv` | `5ded8a08c259b94be44c47b5dba330bf9bf55f6240e40333b5c388ac5c21a8c5` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/p9_short_run_functional_validation.csv` | `0530c3be52df46892a20e4cd8fb99ed83d4ad9ae858f365ad04eac2a0d9c1d8a` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/p10_full_10seed_functional_validation.csv` | `2ffc30507db44effc18edda3339007bef24f1bdf1c235dab49193d405e2ff205` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/p11_robustness_external_ready.csv` | `6a4a38f630dd093de71ce2ccfe58ee2b31db23d887180b71dabdc9c86b0d18b1` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/base_repair_confirmation_trace_v9240.csv` | `5b387b5708fd48654222e1e7585fbef7679722744073e21919142cd69c0f2ebb` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/snapshot_attach_trace_v9240.csv` | `245c3d1ee475989e0670a06308f9d181979dbc14c856e283b76ed4bddd4ff0e6` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/checkpoint_equivalence_trace_v9240.csv` | `5835d26d38cfa1478c3321aaecc1171d52bc6a8b5809404e929030514c4b4743` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/functional_carrier_trace_v9240.csv` | `e886f8650e7fdb243723935364fcc01700497d053bc0c0dd1f98f702bd85a149` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/value_score_trace_v9240.csv` | `312033a1420820b9d84005a018dcefd845c2f4f036be404d940b64c69aa031a5` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/leave_dataset_out_trace_v9240.csv` | `987db27aed545bc00fa6c8be7dc3775a00a0ab17d7240d2b12dbe933bd42d3e0` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/paired_replay_branch_trace_v9240.csv` | `c8049422860b19032f61fd2fb8e90f5712ca534f0eefce9dc63bb115d76db262` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/route_decision.json` | `28ace4af663b236ee82bcd21665580d9f62aedeb88b70ca6b1c66d77518efd04` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/aggregate_decision.json` | `28ace4af663b236ee82bcd21665580d9f62aedeb88b70ca6b1c66d77518efd04` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/failure_table.csv` | `5036d5725e278777d25e14105497d257480f30537f8395de15f4d2217a473125` |
| `results/real_rerun_20260506/v9240_parallel_base_repair_confirmation_snapshot_functional_reentry_first_20260511T123000Z/v9240_provenance_audit.csv` | `c277d8b0d53be307d8b663647ad897a2bc03101d868031649d2454d44b5f09f2` |

## 9. 最终分析结论

v9.2.40 的真实推进是：

```text
v9.2.39: base repair first-wave pass, snapshot late attach not implemented.
v9.2.40: repaired base robust confirmation and snapshot late-attach re-entry are audited in one gated runner.
```

机制判断：

1. Repaired base 仍是 functional route 的前置地基；P2-P6 不能替代 P1。
2. Snapshot attach 的成功必须同时满足 implementation、inactive equivalence、no-event replay preservation。
3. Carrier movement 不是 functional success；只有 value observability、leave-out、official paired replay 都过，才可写 strict functional causal evidence。
4. 本轮没有使用 dataset-specific controller，也没有把 diagnostic scout rows 写成 downstream success。

最终一句话：

> v9.2.40 真实执行后停在 `R11-BasePreservedButValueUnobservable`：`value_observability_fail`。
