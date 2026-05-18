# DG-KAN v9.2.34 Observable Primitive First 实验复盘

> 本复盘记录 `DG-KAN_v9.2.34_ObservablePrimitiveFirst_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R6-ObservablePrimitiveEffectUnpredictable
base_candidate = LQ-t2-h256
success_v9234_observable_primitive = False
success_v9234_strict_purekan_functional = False
success_v9234_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z
```

核心结论：

1. P0 复现 v9.2.33 boundary：source route = `R14-ReturnToInterfacePrimitiveDesign`，legal probe predictive/system pass 均为 `0`，OP not implemented count = `6`。
2. P2 已将 OP1-OP6 纳入真实 implementation smoke：implemented count = `6`。
3. P3 contract/grad/P4 eligible count = `5`；P4 pass count = `2`；P5 near-pass count = `1`。
4. best OP = `OP4-ObservableOrthogonalTailChannel`，observability pass = `0`，corr = `0.099935`，AUC = `0.415519`。
5. 当前 blocker：`observable_primitive_did_not_predict_grounded_value`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `dgkan/models/fc_purekan_actuator.py` | 新增 OP1-OP6 observable edge-owned actuator basis |
| `experiments/run_v9234_observable_primitive_first.py` | v9.2.34 runner；生成 P0-P11 artifacts、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile dgkan/models/fc_purekan_actuator.py experiments/run_v9234_observable_primitive_first.py
```

正式运行：

```bash
python experiments/run_v9234_observable_primitive_first.py \
  --out-dir results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

## 2. Route

```json
{
  "accepted_bad_event_rate": 0.0,
  "accepted_coverage": 0.1111111111111111,
  "accepted_precision": 0.6666666666666666,
  "adamw_fullpass": 0,
  "base_candidate": "LQ-t2-h256",
  "best_base_macro_delta": -0.003388894928826226,
  "best_base_qualified_op": "OP4-ObservableOrthogonalTailChannel",
  "best_base_step_q90": 1.2129731716452363,
  "best_observability_auc": 0.41551939924906134,
  "best_observability_corr": 0.09993491139435033,
  "best_observable_controller": "C5-OPValueScore+OrthogonalTail",
  "best_observable_primitive": "OP4-ObservableOrthogonalTailChannel",
  "completed_at": "2026-05-11T06:35:43Z",
  "cpu_offload_used": 0,
  "external_ready": 0,
  "fake_data_used": 0,
  "fake_proxy_nonzero_count": 0,
  "full_run_pass": 0,
  "legal_probe_predictive_pass": 0,
  "legal_probe_system_pass": 0,
  "next_required_implementation": "continue_OP_interface_design_or_run_official_replay_if_observability_passed",
  "no_fake": true,
  "no_proxy": true,
  "observable_primitive_pass": 0,
  "op_contract_pass_count": 6,
  "op_grad_pass_count": 6,
  "op_implementation_pass": 1,
  "op_implemented_count": 6,
  "op_p4_eligible_count": 5,
  "op_p4_pass_count": 2,
  "op_p5_nearpass_count": 1,
  "p1_major_failure_reason": "L6-current_primitive_unobservable",
  "paired_replay_pass": 0,
  "primary_blocker": "observable_primitive_did_not_predict_grounded_value",
  "proxy_row_used": 0,
  "route": "R6-ObservablePrimitiveEffectUnpredictable",
  "rows_checked": 466,
  "short_run_pass": 0,
  "source_op_not_implemented_count": 6,
  "source_route": "R14-ReturnToInterfacePrimitiveDesign",
  "success_v9234_external_ready": 0,
  "success_v9234_full_functional": 0,
  "success_v9234_observable_primitive": 0,
  "success_v9234_strict_purekan_functional": 0
}
```

## 3. P2/P3 OP implementation 与 contract

P2 implementation pass 只表示 forward/backward/update smoke 可执行；P3 才检查 strict contract、grad 和 interaction/local-bump。

关键值：

| metric | value |
|---|---:|
| implemented OP count | `6` |
| contract pass count | `6` |
| grad pass count | `6` |
| P4 eligible count | `5` |

## 4. P4/P5 base qualification

| primitive | P4 | P5 near | near rows | macro delta | step q90 | memory |
|---|---:|---:|---:|---:|---:|---:|
| OP4-ObservableOrthogonalTailChannel | `1` | `1` | `8/9` | `-0.003389` | `1.212973` | `0.969731` |
| OP5-ObservableControlGapChannel | `1` | `0` | `6/9` | `-0.006333` | `1.151444` | `0.969731` |

## 5. P5 Primitive observability

| controller | corr | AUC | precision | coverage | bad event | pass |
|---|---:|---:|---:|---:|---:|---:|
| C1-OPValueScore | `0.288810` | `0.472778` | `0.222222` | `0.111111` | `0.000000` | `0` |
| C2-OPValueScore+UncertaintyLCB | `0.288811` | `0.473717` | `0.222222` | `0.111111` | `0.000000` | `0` |
| C3-OPValueScore+ControlContrastive | `0.288810` | `0.472778` | `0.222222` | `0.111111` | `0.000000` | `0` |
| C4-OPValueScore+HorizonConsistency | `0.059058` | `0.407071` | `0.111111` | `0.111111` | `0.000000` | `0` |
| C5-OPValueScore+OrthogonalTail | `0.099935` | `0.415519` | `0.666667` | `0.111111` | `0.000000` | `0` |

判断：P5 只有在 OP 同时通过 P4/P5 base gate 后才打开；没有把 source CP5 或 legal probe rows 倒灌成 OP observability success。

## 6. Downstream boundary

P6-P11 只有在 primitive observability pass 后打开。本轮未打开阶段均以 `not_run` row 落盘。

## 7. No-fake audit

```text
rows_checked = 466
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
| runner | `2ec0e21b3babe947b7be723b5d8bacb4820b7c5aca3a67302868ae92e06244f8` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/p0_v9233_boundary_reproduction.csv` | `d304e01aaa99574f4ffbef1501b14dac364a74bdb0f17f5d8732cbbf56a64090` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/p1_legal_probe_failure_autopsy.csv` | `65e3490b2bb6a61e4ec002b562d54c55d07b5b4d1e97f46bd0e1fb8cb6d84109` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/p2_observable_primitive_implementation.csv` | `db07f8d1f289a356b880fc238309882a04d3415280330ab8a669ce678037b4d9` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/p3_contract_gradcheck_interaction.csv` | `db7f26121002ea59dcb77e584f7bf6819a9bc114f36eb6154574eccdce0018f9` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/p4_p4_p5_base_qualification.csv` | `fc3780a926d5101821e4faf168c2e9bd59f225c9f25f660cb85892c40db81283` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/p5_primitive_observability_audit.csv` | `236ff4d488707e1c6f4b0cbba59e4f2bfeebb831d2c31022fe9820089c11fb50` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/p6_leave_dataset_out_validation.csv` | `13fe0cccfcccfd2a0422f902f941fd8fca8424831fc80e44947971b9a89b449d` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/p7_official_signal_routed_paired_replay.csv` | `27660463349046036343e71efb4348dd4d2ea3ef7d85c8e1f588682dd5870a5c` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/p8_short_run_functional_validation.csv` | `209f90c48a30a2fd3df9066618fe2935f50b9d6854f037170fd967d6ea605fec` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/p9_full_10seed_functional_validation.csv` | `abf80d7428c88b424c9831f92c8249056eb7929bc5165530c755c17c0e671b2d` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/p10_adamw_only_fullpass_repair.csv` | `f6d1694e8b7b59dce4cf6c69087b532cf0f4b45520a713a2fb064c3e13eaa313` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/p11_robustness_external_ready.csv` | `746968a55d0bed953b5d2f8295af7f415d7b45a97e5828f922e5de723e75b0a7` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/route_decision.json` | `7e984cead65d6f54005ca0cbd24a94dc76eb54402d11c6c0d6d6ca1b0579a910` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/failure_table.csv` | `21d1233623a9d8d93ca4be746c5ff8b7e773972a8c560f9244dac8529e905af2` |
| `results/real_rerun_20260506/v9234_observable_primitive_first_20260511T000000Z/v9234_provenance_audit.csv` | `3f21110d33b318e3197b9441bed37febdfd5cc6ed4b26eeef47b724abad0e4db` |

## 9. 最终分析结论

v9.2.34 的真实推进是：

```text
v9.2.33: legal train-stream probe 不预测且太贵；OP1-OP6 未实现。
v9.2.34: OP1-OP6 进入真实 strict primitive audit；downstream 只按 gate 打开。
```

机制判断：

1. Observable primitive 的第一关是 base/system/contract，而不是直接 paired replay。
2. 如果没有 P4/P5 survivor，结论是 observable primitive interface 仍未闭合。
3. 如果有 survivor 但 observability 不过，结论是 primitive effect 仍不可预测。
4. 本轮没有继续做 Fashion/KMNIST dataset patch，也没有把 source-logged CP5 当作 legal controller。

最终一句话：

> v9.2.34 真实执行后停在 `R6-ObservablePrimitiveEffectUnpredictable`：`observable_primitive_did_not_predict_grounded_value`。
