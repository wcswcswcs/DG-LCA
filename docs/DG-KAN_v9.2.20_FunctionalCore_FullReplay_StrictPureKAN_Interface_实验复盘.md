# DG-KAN v9.2.20 FunctionalCore FullReplay StrictPureKAN Interface 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.20_FunctionalCore_FullReplay_StrictPureKAN_Interface_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 strict PureKAN interface / external-ready 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.20 执行到一个可审计 terminal route：

```text
route = R2-v8FunctionalFullReplayPass
base_candidate = LQ-t2-h256
success_v9220_functional_core_retained = true
success_v9220_strict_purekan_functional = false
success_v9220_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9220_functional_core_full_replay_strictpurekan_interface_first_20260510T090000Z/
```

核心结论：

1. P0 复现 v9.2.19 boundary：one-step 有 curvature signal，但不是 full replay。
2. P1 已真实执行 v8 50/240-step strong-control replay，覆盖 15 个 branch/control、3 个任务、指定 seeds。
3. P1 route gate 结果：`v8_short_run_pass = 1`，row survivor count = `50`。
4. P2 full replay 只有在 P1 pass 时打开；本轮 P2 状态为 `measured`，并且 `v8_full_replay_pass = 1`。
5. P2 full replay survivor count = `2`，row-level functional survivor count = `22/30`，主要由 `V8-FT7-RoleWiseFunctional` 的 full replay tail/margin/control 优势支撑。
6. 这轮把 v9.2.18/v9.2.19 的暂停状态推进为 `functional_core_retained = true`，但 strict PureKAN functional interface 仍未实现，external-ready 仍不能打开。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9220_functional_core_full_replay.py` | v9.2.20 runner；生成 P0-P8 artifacts、P1 short-run replay、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9220_functional_core_full_replay.py
```

正式运行：

```bash
python experiments/run_v9220_functional_core_full_replay.py \
  --out-dir results/real_rerun_20260506/v9220_functional_core_full_replay_strictpurekan_interface_first_20260510T090000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{
  "adamw_fullpass": 0,
  "base_candidate": "LQ-t2-h256",
  "basis_factory_functional_pass": 0,
  "external_ready": 0,
  "functional_core_retained": 1,
  "functional_short_full_pass": 0,
  "next_required_implementation": "if_short_run_failed_stop_target_patching_else_complete_full_replay_and_strict_purekan_interface",
  "p2_status": "measured",
  "primary_blocker": "strict_purekan_interface_not_yet_implemented_after_v8_full_replay_pass",
  "route": "R2-v8FunctionalFullReplayPass",
  "success_v9220_external_ready": 0,
  "success_v9220_functional_core_retained": 1,
  "success_v9220_strict_purekan_functional": 0,
  "v8_full_replay_pass": 1,
  "v8_full_replay_survivor_count": 2,
  "v8_lr_equivalent": 0,
  "v8_mechanism_identified": 1,
  "v8_short_run_pass": 1,
  "v8_short_run_row_pass_count": 50,
  "v8_short_run_survivor_count": 4,
  "v9_interface_pass": 0
}
```

## 3. P1 v8 50/240-step strong-control replay

Artifacts：

```text
p1_v8_short_run_strong_control_replay.csv
p1_v8_short_run_summary.csv
paired_replay_branch_trace_v9220.csv
```

Top rows（按 CEp99 delta / margin delta 排序）：

| candidate | steps | CEp99 delta | margin delta | curvature delta | step q90 | beats AdamWParallel | beats best LR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `V8-ShuffledRoleMask` | `50` | `-0.294923` | `0.262317` | `-8.83683` | `1.24869` | `0.000` | `0.000` |
| `V8-FT7-RoleWiseFunctional` | `50` | `-0.294795` | `0.262297` | `-8.82902` | `1.2914` | `1.000` | `1.000` |
| `V8-RandomMatchedNorm` | `50` | `-0.288604` | `0.255188` | `1.28671` | `1.16555` | `0.000` | `0.000` |
| `V8-Adaptive-FT-P` | `50` | `-0.287716` | `0.256283` | `0.582374` | `1.20173` | `1.000` | `1.000` |
| `V8-B0-AdamWOnly` | `50` | `-0.286517` | `0.255393` | `1.95139` | `1` | `0.000` | `0.000` |
| `V8-NoOpMatchedOverhead` | `50` | `-0.286517` | `0.255393` | `1.95139` | `1.2617` | `0.000` | `0.000` |
| `V8-AdamWParallelTrustRatio-0.003` | `50` | `-0.285655` | `0.255067` | `1.96468` | `1.17313` | `0.000` | `0.000` |
| `V8-LRScale-1.003` | `50` | `-0.285655` | `0.255067` | `1.96468` | `1.12589` | `0.000` | `0.000` |
| `V8-AdamWParallelTrustRatio-0.01` | `50` | `-0.283613` | `0.25312` | `1.9958` | `1.07527` | `0.000` | `0.000` |
| `V8-LRScale-1.01` | `50` | `-0.283613` | `0.25312` | `1.9958` | `1.09092` | `0.000` | `0.000` |

说明：本轮 P1 使用 `manual_relative_update_replay_no_torch_loss_backward`，同一初始化、同一 batch sequence；`test_acc_proxy` 是 test prefix evaluation，不伪装成完整 external final test。

## 4. P2 v8 full-epoch strong-control replay

Artifacts：

```text
p2_v8_full_epoch_strong_control_replay.csv
p2_v8_full_epoch_summary.csv
```

Protocol：

```text
total replay steps = 640
branches / controls = 15
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
optimizer_protocol = manual_relative_update_replay_no_torch_loss_backward
eval_scope = test_prefix_512
```

Full replay summary：

| candidate | CEp99 delta | margin delta | curvature delta | delta vs AdamW | step q90 | beats AdamWParallel | beats best LR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `V8-FT7-RoleWiseFunctional` | `0.482419` | `0.052268` | `-91.747243` | `-0.002995` | `1.068363` | `1.000000` | `1.000000` |
| `V8-Adaptive-FT-P` | `0.565869` | `0.030103` | `25.561534` | `+0.000521` | `1.147234` | `0.866667` | `0.866667` |
| `V8-B0-AdamWOnly` | `0.561367` | `0.030046` | `38.706858` | `0.000000` | `1.000000` | `0.000000` | `0.000000` |
| `V8-AdamWParallelSameNorm` | `0.546778` | `0.063907` | `40.329388` | `+0.005859` | `1.077066` | `0.000000` | `0.000000` |
| `V8-LRScale-1.10` | `0.546779` | `0.063907` | `40.329387` | `+0.005859` | `1.145172` | `0.000000` | `0.000000` |
| `V8-ShuffledRoleMask` | `0.485271` | `0.058482` | `-91.836780` | `-0.002083` | `1.147926` | `0.000000` | `0.000000` |

Gate summary：

```text
v8_full_replay_pass = 1
v8_full_replay_survivor_count = 2
functional row survivor count = 22/30
V8-FT7-RoleWiseFunctional row survivor count = 9/15
V8-Adaptive-FT-P row survivor count = 13/15
```

判断：

1. P2 不是 one-step，也不是 source recap；它是真实 640-step replay。
2. `V8-FT7-RoleWiseFunctional` 在 full replay 下同时保持 `beats_adamwparallel_rate = 1.0` 和 `beats_best_lr_rate = 1.0`，并且 curvature delta 明显区别于 AdamWParallel / LR controls。
3. `V8-Adaptive-FT-P` 也有 row-level survivor，但 aggregate control beat rate 为 `0.866667`，机制上不如 FT7 干净。
4. `V8-ShuffledRoleMask` 的 CEp99 / curvature 很接近 FT7，但作为 control branch 没有通过 beats AdamWParallel / best LR gate；这说明 full replay 仍需要后续机制抽取，而不能直接等价为 strict PureKAN success。

## 5. Downstream boundary

| artifact | status / reason |
|---|---|
| `p4_strict_purekan_functional_interface_design.csv` | `not_run`: `not_implemented_after_v8_full_replay_pass_in_this_runner` |
| `p5_basis_factory_functional_actuatability.csv` | `not_run`: strict PureKAN interface not implemented |
| `p6_adamw_only_fullpass_repair.csv` | `not_run`: strict PureKAN interface not implemented |
| `p7_functional_survivor_full_validation.csv` | `not_run`: strict PureKAN interface not implemented |
| `p8_robustness_external_ready.csv` | `not_run`: strict PureKAN interface not implemented |

判断：P2 已经打开并通过；当前 blocker 不再是 v8 full replay，而是把保留下来的 functional core 迁移成 strict PureKAN interface。

## 6. No-fake audit

`v9220_provenance_audit.csv`：

```text
rows_checked = 3413
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

## 7. Hash

| artifact | SHA256 |
|---|---|
| v9.2.20 plan | `0ea661748f6942e73e7af05650454653ededa4d72c2510d4fae90fcc3ee60e88` |
| `experiments/run_v9220_functional_core_full_replay.py` | `c2713a4af72f8078a883eb0911f4a057b1849878cbee5122918ede21f8f3d6db` |
| route | `cb3418f0eebd3468b242f2da75a7d9723754055f5ee3e5c3f511f271c1cd0174` |
| `p1_v8_short_run_summary.csv` | `2402d4fc50aab4fda49729df47c54ad909ae79c1e4becf6ee9d7f282a4ee2a11` |
| `p2_v8_full_epoch_summary.csv` | `88de5f1c65764bb9a4c785ea636f1414bdd9aae51e26a8b7916daf0b60809626` |
| provenance audit | `dd834bcc5e71304a1e1e99cb0c39aec7e10cb9fd8ff5d5a9e1091a97de6bae44` |

## 8. 最终分析结论

v9.2.20 的真实推进是：

```text
v9.2.19: v8 one-step strong controls completed.
v9.2.20: v8 short-run strong controls completed and P2 full replay completed;
          v8 functional core retained, but strict PureKAN interface not yet implemented.
```

机制判断：

1. 这轮证明 v8 functional signal 不只是 one-step artifact：P1 short-run 和 P2 640-step full replay 都留下 strong-control survivor。
2. `V8-FT7-RoleWiseFunctional` 是当前最重要的 functional core 证据：它在 P2 中 beat AdamWParallel / best LR 的 aggregate rate 均为 `1.0`，且 curvature delta 与 task-gradient controls 明显不同。
3. 但这仍不是 strict PureKAN success。当前 replay 是 v8 functional core protocol，不是已经接入 LQ/A7c strict FC-PureKAN interface 的训练路线。
4. 因此下一步不应继续修 output target / SNR threshold，而应做 strict PureKAN functional interface：把 FT7-like role functional signal 变成 edge-owned、P4-qualified、AdamW-only base 不破坏、且能过 paired/full validation 的接口。

最终一句话：

> v9.2.20 真实执行后停在 `R2-v8FunctionalFullReplayPass`：v8 functional core 已通过 short-run 与 640-step full strong-control replay，`success_v9220_functional_core_retained = true`；但 strict PureKAN functional interface 仍未实现，external-ready 不能打开。
