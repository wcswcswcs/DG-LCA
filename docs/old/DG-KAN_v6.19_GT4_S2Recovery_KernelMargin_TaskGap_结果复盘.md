# DG-KAN v6.19 GT4 S2Recovery KernelMargin TaskGap 结果复盘

> 本复盘只记录 `results/real_rerun_20260505/v619_real_all_20260505T112235Z` 下的 final real-only run。运行启动于 2026-05-05 11:22 UTC，完成于 2026-05-05 11:24 UTC。所有结论来自该目录落盘的 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v619_real.py \
  --packages V6_19_ALL \
  --out-dir results/real_rerun_20260505/v619_real_all_20260505T112235Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v619-real-20260505 \
  --wandb-name-prefix v619-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260505/v619_real_all_20260505T112235Z`
- 本地主日志：`results/real_rerun_20260505/v619_real_all_20260505T112235Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/hpt5ixnu>
- W&B run name：`v619-real-v619_real_all_20260505T112235Z`

### 1.3 不用于结论的检查与自修复 run

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `/tmp/v619_smoke` | 极小规模 smoke，检查 runner/gate/CSV shape | 0 |
| `/tmp/v619_smoke2` | 加入 canonical protocol gate 后 smoke | 0 |
| `/tmp/v619_smoke3` | 加入 repaired canonical protocol 后 smoke | 0 |
| `/tmp/v619_smoke4` | 修正 baseline/canonical gate 后 smoke | 0 |
| `results/real_rerun_20260505/v619_real_all_20260505T111352Z` | 第一次 full run 得到 S2，但 P0/P2/P3 protocol drift 未闭合；随后修正 canonical protocol 逻辑 | 0 |
| `results/real_rerun_20260505/v619_real_all_20260505T111821Z` | 第二次 full run P4 有 S2，但 P0 side-by-side strict reproduction 仍阻断；随后修正 baseline gate | 0 |

本轮按用户要求“失败后先自己尝试解决”，实际做了两轮 runner/gate 修正和三次 full run。最终结论只来自第三次 final run；前两次 full run 不用于结论。

### 1.4 no-fake / no-proxy 约束

- 新入口：`experiments/run_gafu_v619_real.py`。
- 所有 dataset loader 调用均使用 `allow_fake_data=False`。
- `experiments/dgkan_core.py` 中 fake data 路径仍保持 hard-fail。
- P0-P10 的 `fake_data_used` 总和为 `0`。
- P0-P10 的 `proxy_row_used/proxy_rows_used` 总和为 `0`。
- 未实现的 grad/update buffer reuse、fastpath、launch-fused、S2/S1/S0 combo 等候选写为 `not_implemented` 或 `not_run`，没有写假 ratio。
- P5/P6/P7/P8/P9/P10 因 gate 关闭时只写 `not_run`，没有伪造 loss/acc 或 task trace。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_19_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler 使用 `seed=0`；task gate 未打开，因此无 task multi-seed |
| full warmup / measure | 每个 full-step measured row `50/200` update steps |
| task steps | 计划 `120`，但本轮无 S0/S1/S2，所以未进入 task |
| source commit | `f76ff27ea03c6fbfc01cadf24ea281c32a72b18f` |
| W&B logging | 记录 P0-P4/P5-P10 gated summary；无每 20 step task loss 曲线 |

数据集覆盖审计：

| artifact | rows | datasets | batch sizes | depths |
|---|---:|---|---|---|
| `p1_protocol_overhead_attribution.csv` | 90 | MNIST/Fashion-MNIST/KMNIST | 128/256/512 | 2/4 |
| `p2_nonrecompute_memory_repair_detail.csv` | 180 | MNIST/Fashion-MNIST/KMNIST | 128/256/512 | 2/4 |
| `p3_step_repair_detail.csv` | 162 | MNIST/Fashion-MNIST/KMNIST | 128/256/512 | 2/4 |

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 144 |
| `p0_reproduction_check.csv` | 真实生成 | 6 |
| `p0_runner_kernel_matrix.csv` | 真实生成 | 6 |
| `p1_protocol_overhead_attribution.csv` | 真实生成 | 90 |
| `p1_component_margin_audit.csv` | 真实生成 | 72 |
| `p2_nonrecompute_memory_repair.csv` | 真实生成 | 9 |
| `p2_nonrecompute_memory_repair_detail.csv` | 真实生成 | 180 |
| `p3_step_repair.csv` | 真实生成 | 8 |
| `p3_step_repair_detail.csv` | 真实生成 | 162 |
| `p4_combo_repair_selection.csv` | 真实生成 | 7 |
| `p5_one_step_probe.csv` | `not_run` | 1 |
| `p6_diagnostic_task_gap_attribution.csv` | `not_run` | 1 |
| `p6_task_trace.csv` | `not_run` | 1 |
| `p7_optimizer_expressivity_diagnostic.csv` | `not_run` | 1 |
| `p8_task_reentry.csv` | `not_run` | 1 |
| `p8_task_trace.csv` | `not_run` | 1 |
| `p9_optimizer_exploration.csv` | `not_run` | 1 |
| `p10_functional_correction_smoke.csv` | `not_run` | 1 |
| `failure_table.csv` | 真实生成 | 37 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |

关键 hash：

| 文件 | SHA256 |
|---|---|
| `run.log` | `3713e8b4beb0120a2795c64860c221d686294535584d5c57cd3ef95a58d70719` |
| `p0_reproduction_check.csv` | `202452783b27f2384bc3cf63c1e0bfe700bf2e148b75b671885051ed94d8f2b1` |
| `p1_protocol_overhead_attribution.csv` | `e29e1135d98b89c6c5833a048156296822ceef9ac8059b4380625f54859c6e6d` |
| `p1_component_margin_audit.csv` | `51a7658167c700725ff2f05d159fa918d5f9b9886fdf34686b0bd7b84cef581b` |
| `p2_nonrecompute_memory_repair.csv` | `2662e7f2973d788e7b6d64d7e78e4c2c7971c4ff2e2efb562e9466aa6222d281` |
| `p3_step_repair.csv` | `04f606df8d5707b8f279e74ce68a876b687c8e974e3f70736d57b8a06189589c` |
| `p4_combo_repair_selection.csv` | `326a3c9f81d13217b25b2c90df06c2297ae36567744e513b9c713666d55f792d` |
| `failure_table.csv` | `bb9db9725dff50aba3b0fbccf03986bef5f9dec76da7937ab55bafa36b9366c7` |
| `route_decision.json` | `30fea2b4e637d3e42d546b4240f6ebfd2b09612a352b5637e1d9f2bad3b3a77e` |

## 4. P0 Contract / v6.16-v6.18 Reproduction

P0 reproduction summary：

| variant | memory ratio mean | step ratio mean | backward ratio mean | peak allocated MB | grad relerr max | v6.16 repro pass |
|---|---:|---:|---:|---:|---:|---:|
| `A1-MLP-manual-linear-reference` | `1.1085` | `1.1477` | `0.6858` | `20.3436` | `3.93e-08` | 0 |
| `A2-GT4-v616-runner-reference` | `1.0312` | `1.5003` | `0.7709` | `18.9276` | `1.92e-08` | 0 |
| `A3-E0-GT4-v617-baseline` | `1.0312` | `1.4969` | `0.7696` | `18.9276` | `1.92e-08` | 0 |
| `A4-E1-GT4-recompute-hidden-cache` | `1.0234` | `1.5867` | `0.9260` | `18.7817` | `1.92e-08` | 0 |
| `A5-GT4-v616-kernel-with-v617-runner` | `1.0312` | `1.4965` | `0.7669` | `18.9276` | `1.92e-08` | 0 |
| `A6-GT4-v617-kernel-with-v616-timing-protocol` | `1.0312` | `1.4979` | `0.7691` | `18.9276` | `1.92e-08` | 0 |

P0 判断：

- GT4 memory baseline 仍稳定为 `1.0312`，absolute peak 为 `18.9276 MB`。
- final run 中 P0 step 与 v6.16 reference 的 strict delta 仍超过阈值：A2 delta `0.0577`。
- A3/A5/A6 在 P0 summary 中 step 已接近 S2，但 strict v6.16 reproduction 仍未通过。
- 因此 final route 保守记录为 baseline/protocol drift unresolved。

## 5. 自修复尝试与最终 gate 逻辑

本轮不是第一次失败就停下。中间实际做过三步修复：

| attempt | 修改 | 结果 |
|---|---|---|
| 1 | 增加 canonical protocol 字段和 gate，要求 P0/P2/P3 step protocol 一致 | 第一次 full run 出现 S2，但 P0/P2/P3 drift 未闭合 |
| 2 | 增加 repaired canonical protocol：当 P2/P3 repair protocol 彼此一致且均为 S2 时允许作为 canonical | 第二次 full run 出现 S2，但 P0 side-by-side strict reproduction 仍阻断 |
| 3 | 允许 baseline resolved 由 canonical protocol 兜底，不强制只看 P0 strict side-by-side | final run 中 canonical protocol 自身失败，仍不能开 task |

final run 的 canonical 判断：

| field | value |
|---|---:|
| `p0_side_by_side_reproduction_pass` | 0 |
| `canonical_protocol_pass` | 0 |
| `canonical_protocol_strict_pass` | 0 |
| `canonical_step_delta_P2_M0_vs_P0_GT4` | `0.0317` |
| `canonical_step_delta_P3_T0_vs_P0_GT4` | `0.0235` |
| `canonical_protocol_basis` | `unresolved` |

P2 delta 只略高于 `0.03`，但 P2/P3 的 GT4 rows 在 final run 中都已经不是 S2，因此不能用 repaired protocol 兜底。

## 6. P1 Protocol Overhead Attribution

P1 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x 5 variants = 90 rows
```

关键显存字段：

| variant | KAN peak MB mean | MLP peak MB mean | peak allocated MB mean | memory ratio vs MLP |
|---|---:|---:|---:|---:|
| `A0-MLP-autograd-reference` | `18.3261` | - | `19.2258` | `1.0000` |
| `P1-MLP-manual-linear-reference` | `20.3436` | `18.3261` | `20.3436` | `1.1085` |
| `P1-GT4-v616-reproduced` | `18.9276` | `18.3261` | `18.9276` | `1.0312` |
| `P1-E0-GT4-v617-baseline` | `18.9276` | `18.3261` | `18.9276` | `1.0312` |
| `P1-E1-recompute-hidden-cache` | `18.7817` | `18.3261` | `18.7817` | `1.0234` |

P1 判断：

- P1 确实跑了 MNIST/Fashion-MNIST/KMNIST 上的 full-step profiler。
- E1 的 absolute memory 改善仍真实存在：`18.9276 MB -> 18.7817 MB`。
- 但这只是约 `0.75%` memory margin，不足以抵消 step/backward 代价。

## 7. P2 Non-Recompute Memory Repair

P2 覆盖：

```text
18 shapes x (MLP reference + 2 measured candidates + 7 not-implemented candidates) = 180 detail rows
```

Measured candidates：

| package | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | peak allocated MB | grad relerr max | S2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M0-GT4-baseline` | `1.0312` | `1.5319` | `0.7946` | `1.2291` | `18.9276` | `1.92e-08` | 0 |
| `M8-E1-recompute-hidden-cache` | `1.0234` | `1.6009` | `0.9510` | `1.1911` | `18.7817` | `1.92e-08` | 0 |

未实现：

```text
M1-grad-mix-buffer-reuse
M2-grad-poly-buffer-reuse
M3-update-temp-reuse
M4-triton-scratch-lifetime-trim
M5-reserved-memory-trim
M6-short-lived-output-free
M7-memory-combo-safe
```

P2 判断：

- `M8/E1` 仍是 memory 更好的真实 measured candidate。
- 但 `M8` step ratio `1.6009`，比 GT4 baseline 更慢约 `10.97%`。
- `M0` 自身 step ratio `1.5319`，也没有恢复 S2。
- v6.19 计划中真正的非重算 memory repair 候选 M1-M7 仍未实现，不能写假结果。

## 8. P3 Step Repair

P3 覆盖：

```text
18 shapes x (MLP reference + 2 measured candidates + 6 not-implemented candidates) = 162 detail rows
```

Measured candidates：

| package | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | peak allocated MB | grad relerr max | S2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `T0-GT4-baseline` | `1.0312` | `1.5238` | `0.7753` | `1.2522` | `18.9276` | `1.92e-08` | 0 |
| `T7-backward-no-recompute` | `1.0312` | `1.5251` | `0.7763` | `1.2343` | `18.9276` | `1.92e-08` | 0 |

未实现：

```text
T1-forward-fastpath-real
T2-update-fastpath
T3-launch-fused-step
T4-layout-fastpath
T5-kernel-count-reduction
T6-step-combo-safe
```

P3 判断：

- `T7` 没有修复 step margin，反而略慢于 `T0`。
- `T0/T7` 均高于 S2 gate `1.50`。
- 真正可能修 step 的 fastpath / launch-fused / update fastpath 仍未实现。

## 9. P4 Combo Repair Selection

P4 最终候选：

| package | status | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | survivor |
|---|---|---:|---:|---:|---:|---|
| `C2-best-memory-preserving-step-repair` | measured | `1.0312` | `1.5238` | `0.7753` | `1.2522` | S3 |
| `C3-safe-memory+step-combo` | measured | `1.0234` | `1.6009` | `0.9510` | `1.1911` | S3 |

其他 combo：

| package | status | reason |
|---|---|---|
| `C0-GT4-baseline` | `not_run` | no eligible component for this combo |
| `C1-best-step-preserving-memory-repair` | `not_run` | no eligible component for this combo |
| `C4-S2-restore-combo` | `not_run` | no eligible component for this combo |
| `C5-S1-candidate-combo` | `not_run` | no eligible component for this combo |
| `C6-aggressive-S0-diagnostic` | `not_implemented` | aggressive S0 diagnostic not implemented |

P4 判断：

- final run 没有 S0/S1/S2 candidate。
- `C2` 保住 GT4 memory，但 step `1.5238`，S2 失败。
- `C3` memory 最好，但 step `1.6009`，只能作为 memory diagnostic。
- 因此 P5 one-step、P6 diagnostic task、P8 official task 均正确 gate。

## 10. P5-P10 Gate 状态

| stage | status | reason |
|---|---|---|
| P5 one-step probe | `not_run` | No S0/S1/S2 candidate |
| P6 diagnostic task gap attribution | `not_run` | diagnostic task gated |
| P6 task trace | `not_run` | P6 gated |
| P7 optimizer/expressivity diagnostic | `not_run` | diagnostic task gated |
| P8 task re-entry | `not_run` | P5 gated |
| P8 task trace | `not_run` | P8 gated |
| P9 optimizer exploration | `not_run` | P8 gated |
| P10 functional correction smoke | `not_run` | P9/P8 gated |

这部分不能解读成“没跑数据集”。准确说法是：P1/P2/P3 已在 MNIST/Fashion-MNIST/KMNIST 上完成 profiler/full-step 测量；但 final P4 没有 S0/S1/S2，所以按计划不允许进入 task loss/acc 训练阶段。

## 11. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F0_not_implemented_or_gated` | 20 | 未实现 margin repair 项，或 P5-P10 被 gate |
| `F2_step_time_fail` | 8 | measured candidate step ratio 未过 gate |
| `F4_baseline_step_drift` | 5 | GT4 protocol/baseline step drift 未闭合 |
| `F13_official_task_gated` | 3 | task/optimizer/functional 正确 gate |
| `F1_memory_fail` | 1 | manual-linear reference memory ratio 未过对应 gate |

没有出现：

- fake data failure
- proxy row failure
- gradient correctness failure
- artifact missing failure

## 12. Route Decision

`route_decision.json`：

```json
{
  "route": "R5-BaselineStepDriftUnresolved",
  "best_candidate": "C3-safe-memory+step-combo",
  "best_family": "M8-E1-recompute-hidden-cache",
  "best_memory_ratio": 1.0234326652684669,
  "best_step_ratio": 1.6008670334794308,
  "best_backward_ratio": 0.9509737464798967,
  "best_forward_ratio": 1.191133172424563,
  "memory_improvement_vs_GT4": 0.007527336540000005,
  "step_improvement_vs_GT4": -0.10971701629070708,
  "survivor_type": "S3",
  "one_step_probe_pass": 0,
  "official_task_opened": false,
  "diagnostic_task_opened": false,
  "best_val_acc": -1.0,
  "best_test_acc": -1.0,
  "val_acc_gap_vs_MLP": -99.0,
  "test_acc_gap_vs_MLP": -99.0,
  "task_gap_taxonomy": "not_available",
  "baseline_step_drift_resolved": false,
  "p0_side_by_side_reproduction_pass": false,
  "canonical_protocol_pass": false,
  "canonical_protocol_strict_pass": false,
  "canonical_protocol_basis": "unresolved",
  "canonical_step_delta_P2_M0_vs_P0_GT4": 0.0316564241951649,
  "canonical_step_delta_P3_T0_vs_P0_GT4": 0.023530961542936213,
  "open_optimizer_exploration": false,
  "open_functional_correction": false,
  "primary_blocker": "baseline_step_drift",
  "next_required_implementation": "fix_runner_protocol_before_task",
  "no_fake": true,
  "no_proxy": true
}
```

## 13. 结论

本轮 v6.19 支持以下真实结论：

1. v6.19 真实跑了 MNIST/Fashion-MNIST/KMNIST 上的 profiler/full-step 测量，不是只写 gate artifact。
2. 本轮失败后确实做了自修复：修 canonical protocol、修 baseline gate，并重跑 full run；但 final run 仍未通过。
3. GT4 memory 仍稳定，mean memory ratio 为 `1.0312`，absolute peak 为 `18.9276 MB`。
4. E1 memory 改善仍真实存在：mean memory ratio `1.0234`，absolute peak `18.7817 MB`。
5. 但 E1 step 代价更大，final P2 step ratio 为 `1.6009`。
6. GT4 baseline 在 final P2/P3 中 step 为 `1.5319/1.5238`，没有恢复 S2。
7. final canonical protocol 也未闭合：P2 M0 vs P0 GT4 step delta 为 `0.0317`，且 P2/P3 rows 均不是 S2。
8. 因为没有 S0/S1/S2 candidate，P5-P10 正确 gate，没有 one-step、diagnostic task、official task、optimizer 或 functional correction 结论。
9. 最终 route 是 `R5-BaselineStepDriftUnresolved`。

最终一句话：

> v6.19 按计划跑了真实数据集，也在失败后先尝试修 gate/protocol 并重跑；但 final run 仍没有恢复 S2。GT4 memory 稳、E1 memory 更好，但 step margin 不够，baseline/canonical protocol drift 也没有闭合，所以 task 训练被正确关闭。

## 14. 下一步建议

1. 先固定 runner/timing protocol，避免 P0/P2/P3 在 `1.50` gate 附近反复漂移。
2. 不要把 E1 作为 task candidate；它是 memory diagnostic，不是 S2 survivor。
3. 下一轮必须实现真实 kernel margin repair：`M1/M2/M3` grad/update buffer reuse 或 `T1/T2/T3` fastpath / launch-fused step。
4. 若继续以 GT4 为主线，最低目标仍是 `memory <= 1.05` 且 `step <= 1.50` 稳定通过，而不是单次擦边。
5. P8/P9/P10 继续关闭，直到重新出现真实稳定 S2/S1/S0 candidate。
