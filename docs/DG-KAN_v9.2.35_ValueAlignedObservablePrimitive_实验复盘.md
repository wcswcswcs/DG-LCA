# DG-KAN v9.2.35 Value-Aligned Observable Primitive 实验复盘

> 本复盘记录 `DG-KAN_v9.2.35_ValueAlignedObservablePrimitive_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R9-VAOPAllFailPrimitiveReset
base_candidate = LQ-t2-h256
success_v9235_strict_purekan_functional = False
success_v9235_full_functional = False
success_v9235_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z
```

核心结论：

1. P0 复现 v9.2.34 boundary：source route = `R6-ObservablePrimitiveEffectUnpredictable`，OP implemented count = `6`，observability pass = `0`。
2. P1 OP failure attribution pass = `1`，primary mode = `F4-control_gap_unmodeled`。
3. P2 VAOP implemented count = `6`；P3 contract/grad pass = `1/1`。
4. P4 best measured VAOP = `VAOP6-FamilyValueChannel`，P4 pass = `1`，P5 near-pass = `0`，base-qualified VAOP = ``。
5. P5 value observability pass = `0`，best controller = ``，corr = `0.000000`，AUC = `0.500000`。
6. 当前 blocker：`all_vaop_failed_P5_nearpass`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `dgkan/models/fc_purekan_actuator.py` | 新增 VAOP1-VAOP6 value-aligned observable specs |
| `experiments/run_v9235_value_aligned_observable_primitive.py` | v9.2.35 runner；生成 P0-P11 artifacts、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile dgkan/models/fc_purekan_actuator.py experiments/run_v9235_value_aligned_observable_primitive.py
```

正式运行：

```bash
python experiments/run_v9235_value_aligned_observable_primitive.py \
  --out-dir results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z \
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

```json
{
  "route": "R9-VAOPAllFailPrimitiveReset",
  "base_candidate": "LQ-t2-h256",
  "v9234_boundary_pass": 1,
  "source_route": "R6-ObservablePrimitiveEffectUnpredictable",
  "source_implemented_op_count": 6,
  "source_observability_pass": 0,
  "dataset_tuning_detected": 0,
  "op_failure_mode": "F4-control_gap_unmodeled",
  "op_failure_attribution_pass": 1,
  "corr_r_perp_tail": -0.031948448199853374,
  "corr_obs_score": 0.0008482374944324951,
  "corr_control_gap": 0.288809800503711,
  "vaop_implemented_count": 6,
  "best_vaop_candidate": "",
  "best_p4_vaop_candidate": "VAOP6-FamilyValueChannel",
  "vaop_contract_pass": 1,
  "vaop_grad_pass": 1,
  "vaop_p4_pass": 1,
  "vaop_p5_nearpass": 0,
  "best_base_macro_delta": 0.0,
  "best_base_step_q90": 0.0,
  "best_p4_macro_delta": -0.00433333052529229,
  "best_p4_step_q90": 1.1494791358400787,
  "vaop_observability_pass": 0,
  "best_value_controller": "",
  "best_value_corr": 0.0,
  "best_value_auc": 0.5,
  "accepted_precision": 0.0,
  "accepted_coverage": 0.0,
  "accepted_bad_event_rate": 0.0,
  "vaop_system_pass": 1,
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
  "primary_blocker": "all_vaop_failed_P5_nearpass",
  "next_required_implementation": "if_vaop_observability_failed_return_to_rolewise_edge_function_design_else_run_official_paired_replay",
  "success_v9235_strict_purekan_functional": 0,
  "success_v9235_full_functional": 0,
  "success_v9235_external_ready": 0,
  "rows_checked": 890,
  "fake_proxy_nonzero_count": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true,
  "completed_at": "2026-05-11T07:33:33Z"
}
```

## 3. P4 VAOP base qualification

| primitive | P4 | P5 near | near rows | macro delta | step q90 | memory |
|---|---:|---:|---:|---:|---:|---:|
| VAOP4-RoleWiseFT7EdgeChannel | `1` | `0` | `6/9` | `-0.005667` | `1.212533` | `0.969731` |
| VAOP5-ConservativeControlGapLowerBoundChannel | `1` | `0` | `7/9` | `-0.006667` | `1.153620` | `0.969731` |
| VAOP6-FamilyValueChannel | `1` | `0` | `7/9` | `-0.004333` | `1.149479` | `0.969731` |

## 4. P5 value observability

| controller | corr | AUC | precision | coverage | bad event | pass |
|---|---:|---:|---:|---:|---:|---:|
| not_run |  |  |  |  |  | `no_P4_P5_base_qualified_VAOP` |

判断：value score 只使用 commit-time train/probe statistics；grounded value / Y_beat 只作为离线 audit label，没有进入 commit rule。

## 5. Downstream boundary

P7-P11 只有在 P6 leave-out gate 后打开。本轮未打开阶段均以 `not_run` row 落盘。

## 6. No-fake audit

```text
rows_checked = 890
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
| runner | `7f99cf4888a8691e709e2cbd7ffcbfe3020074c60f37effb0d48de3dc5164fe9` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/contract_audit_v9235.csv` | `c8bf78b780b17b514c00f4d37e6cc010c66b4f5a8c07c0bf77242888d578a557` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/p0_v9234_boundary_reproduction.csv` | `4061a574bad789c4a90cce70d24ec819b7b67a50157d51e2d79fea4afe283696` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/p1_op_failure_attribution.csv` | `afaf3a3e69aa7c1c1a0c9490c8d7982c5779f9b3b512bb9898bb8d516dc473b2` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/p2_value_aligned_primitive_implementation.csv` | `bc30abdfac0703a21f77f3e8cdedc5339a1a6f7bc2adcfc86a09b724009ada19` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/p3_contract_grad_interaction_audit.csv` | `b1ab4ff224d04b9415790d9c54b96f740341f6d648e2e028019cb57089560d7f` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/p4_vaop_p4_p5_base_qualification.csv` | `4b642ef002fab2a05670a36f03cdc31fffca7c9e53980e7e16aa01bdae90deab` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/p5_value_observability_audit.csv` | `11a8d9a94a1d0c99131a2409bf96ddd52f118e0db5409c34d1eed9b7c3a0ded1` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/p6_leave_dataset_and_stratum_out_validation.csv` | `2442097f959c583896641746669277015fa6c5ae931d9c824a66f3ac70890b65` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/p7_official_value_aligned_paired_replay.csv` | `21a97cf885351f723b62fac55d27865f72671e0f3f4cc649addd7602ec167900` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/paired_replay_branch_trace_v9235.csv` | `9820674ea993b0aca73991568b2a84802c12bda8acf54731110b3cfe58c91389` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/p8_short_run_functional_validation.csv` | `25b851fc2a53ba9ffa56f2a4bcf2ba577566b77065ef83adc990d6267f3d8a11` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/p9_full_10seed_functional_validation.csv` | `4d465b233dd9db9fcece90f68fe9129fd9fbbfc9b1a3b018ca2a5937e663f849` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/p10_adamw_only_fullpass_repair.csv` | `a6b93ffc221ce698c7a569690a358825f6e15460b5d3e4ceec8c0e343a6f1b18` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/p11_robustness_external_ready.csv` | `3197bbfc492f803376f10340f4ab90a5620b53f114e47ce3db9e4e5541719d8c` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/vaop_primitive_trace_v9235.csv` | `bc30abdfac0703a21f77f3e8cdedc5339a1a6f7bc2adcfc86a09b724009ada19` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/value_score_trace_v9235.csv` | `11a8d9a94a1d0c99131a2409bf96ddd52f118e0db5409c34d1eed9b7c3a0ded1` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/op_failure_trace_v9235.csv` | `afaf3a3e69aa7c1c1a0c9490c8d7982c5779f9b3b512bb9898bb8d516dc473b2` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/leave_dataset_out_trace_v9235.csv` | `2442097f959c583896641746669277015fa6c5ae931d9c824a66f3ac70890b65` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/run_manifest.json` | `e8b9d4d093b6a5c07da786ec94b2ddd2a7cf6630649ef35e85f0613777c0ff87` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/route_decision.json` | `b3e78dc7dec3218947710a2bff2dc42fbdc54b74f9d05d74b467a87c1942935d` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/aggregate_decision.json` | `b3e78dc7dec3218947710a2bff2dc42fbdc54b74f9d05d74b467a87c1942935d` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/failure_table.csv` | `f209116127d3d02fcfd1d1310c31b05281cd8691ce56535fe3025a946d6bab69` |
| `results/real_rerun_20260506/v9235_value_aligned_observable_primitive_first_20260511T080000Z/v9235_provenance_audit.csv` | `66dd90360ed2de84993b47d4a68241d755bc5e073171ba521c4330a5a1b4629b` |

## 8. 最终分析结论

v9.2.35 的真实推进是：

```text
v9.2.34: OP4 可实现、P4/P5 过，但 movement score 不预测 value。
v9.2.35: VAOP family 进入 value-aligned primitive audit；downstream 仍由 gate 决定。
```

机制判断：

1. 本轮没有继续按 Fashion/KMNIST dataset patch，而是把失败归因和 value-aligned primitive 分开审计。
2. VAOP 的成功条件不是 movement 大，而是 commit-time value score 对 grounded control-relative value 可预测。
3. 如果 P4/P5 或 observability 未过，不能打开 official paired replay。

最终一句话：

> v9.2.35 真实执行后停在 `R9-VAOPAllFailPrimitiveReset`：`all_vaop_failed_P5_nearpass`。
