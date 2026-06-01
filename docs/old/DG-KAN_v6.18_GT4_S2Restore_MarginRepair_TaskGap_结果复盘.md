# DG-KAN v6.18 GT4 S2Restore MarginRepair TaskGap 结果复盘

> 本复盘只记录 `results/real_rerun_20260505/v618_real_all_20260505T105435Z` 下的 final real-only run。运行启动于 2026-05-05 10:54 UTC，完成于 2026-05-05 10:57 UTC。所有结论来自该目录落盘的 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v618_real.py \
  --packages V6_18_ALL \
  --out-dir results/real_rerun_20260505/v618_real_all_20260505T105435Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v618-real-20260505 \
  --wandb-name-prefix v618-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260505/v618_real_all_20260505T105435Z`
- 本地主日志：`results/real_rerun_20260505/v618_real_all_20260505T105435Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/w1qsxk5o>
- W&B run name：`v618-real-v618_real_all_20260505T105435Z`

### 1.3 不用于结论的检查

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `/tmp/v618_smoke` | 极小规模 smoke，暴露 baseline gate 顺序问题 | 0 |
| `/tmp/v618_smoke2` | 修正 baseline 未复现时不得开 task 后的 smoke | 0 |

### 1.4 no-fake / no-proxy 约束

- 新入口：`experiments/run_gafu_v618_real.py`。
- 所有 dataset loader 调用均使用 `allow_fake_data=False`。
- `experiments/dgkan_core.py` 中 fake data 路径仍保持 hard-fail。
- P0-P10 的 `fake_data_used` 总和为 `0`。
- P0-P10 的 `proxy_row_used/proxy_rows_used` 总和为 `0`。
- 未实现的 grad/update buffer reuse、fastpath、launch-fused、aggressive S0/S1 combo 等候选写为 `not_implemented` 或 `not_run`，没有写假 ratio。
- P5/P6/P7/P8/P9/P10 因 gate 关闭时只写 `not_run`，没有伪造 loss/acc 或 task trace。

### 1.5 W&B 显存记录修正

针对 v6.16/v6.17 中“看不到显存占用”的问题，v6.18 runner 显式向 W&B 记录了绝对显存字段：

```text
memory/KAN_peak_MB
memory/MLP_peak_MB
memory/peak_allocated_MB
memory/peak_reserved_MB
memory/ratio_vs_MLP
```

这些字段来自本地 profiler CSV 的真实落盘值，不是只记录 ratio。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_18_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler 使用 `seed=0`；task gate 未打开，因此无 task multi-seed |
| full warmup / measure | 每个 full-step measured row `50/200` update steps |
| task steps | 计划 `120`，但本轮无 S0/S1/S2，所以未进入 task |
| run duration | `173.09` sec |
| source commit | `f76ff27ea03c6fbfc01cadf24ea281c32a72b18f` |
| Triton | available |
| memory history | available |
| W&B logging | 记录 P0-P4/P5-P10 gated summary；无每 20 step task loss 曲线 |

数据集覆盖审计：

| artifact | rows | datasets | batch sizes | depths |
|---|---:|---|---|---|
| `p1_efficiency_margin_attribution.csv` | 90 | MNIST/Fashion-MNIST/KMNIST | 128/256/512 | 2/4 |
| `p2_step_preserving_memory_repair_detail.csv` | 180 | MNIST/Fashion-MNIST/KMNIST | 128/256/512 | 2/4 |
| `p3_memory_preserving_step_repair_detail.csv` | 162 | MNIST/Fashion-MNIST/KMNIST | 128/256/512 | 2/4 |

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 144 |
| `p0_reproduction_check.csv` | 真实生成 | 6 |
| `p0_runner_kernel_matrix.csv` | 真实生成 | 6 |
| `p1_efficiency_margin_attribution.csv` | 真实生成 | 90 |
| `p1_component_kernel_audit.csv` | 真实生成 | 72 |
| `p2_step_preserving_memory_repair.csv` | 真实生成 | 9 |
| `p2_step_preserving_memory_repair_detail.csv` | 真实生成 | 180 |
| `p3_memory_preserving_step_repair.csv` | 真实生成 | 8 |
| `p3_memory_preserving_step_repair_detail.csv` | 真实生成 | 162 |
| `p4_combo_repair_selection.csv` | 真实生成 | 7 |
| `p5_one_step_probe.csv` | `not_run` | 1 |
| `p6_diagnostic_task_gap_attribution.csv` | `not_run` | 1 |
| `p6_task_trace.csv` | `not_run` | 1 |
| `p7_optimizer_expressivity_diagnostic.csv` | `not_run` | 1 |
| `p8_task_reentry.csv` | `not_run` | 1 |
| `p8_task_trace.csv` | `not_run` | 1 |
| `p9_optimizer_exploration.csv` | `not_run` | 1 |
| `p10_functional_correction_smoke.csv` | `not_run` | 1 |
| `failure_table.csv` | 真实生成 | 32 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |

关键 hash：

| 文件 | SHA256 |
|---|---|
| `run.log` | `c27ce666ff9d0443af937d252171e2d4f5ca00d53321f6c175f2ca31b9cc4329` |
| `p1_efficiency_margin_attribution.csv` | `1b4d479e2d86fedc2a3ea0da1fa04a2cebad465fba9c26f20db9dd4fcc1b5391` |
| `p2_step_preserving_memory_repair.csv` | `df6b71030cbb24836d9873ffff49155e29db99088d55ff141bb719707c227006` |
| `p3_memory_preserving_step_repair.csv` | `0a13c2aafcedaef5d1a5925cea3948b62999db8e34233c9ae812a192f7d6fcc6` |
| `p4_combo_repair_selection.csv` | `fbf82f2fc65e87d8ed8631d2e3b6534a45c0c17ecfba913163969c74d45a8523` |
| `failure_table.csv` | `0a3ac24180843971caa0cdb411d5588f25138c027df9448d5f1dd12394069b19` |
| `route_decision.json` | `6f18a0c1c970bd6bd9bf78dc66e90be225f48916e05ced258ccc7048f1d71c6b` |

## 4. P0 Contract / v6.16-v6.17 Reproduction

P0 reproduction summary：

| variant | memory ratio mean | step ratio mean | backward ratio mean | peak allocated MB | grad relerr max | v6.16 repro pass |
|---|---:|---:|---:|---:|---:|---:|
| `A1-MLP-manual-linear-reference` | `1.1085` | `1.1367` | `0.6786` | `20.3436` | `3.93e-08` | 0 |
| `A2-GT4-v616-runner-reference` | `1.0312` | `1.4774` | `0.7571` | `18.9276` | `1.92e-08` | 1 |
| `A3-E0-GT4-v617-baseline` | `1.0312` | `1.4845` | `0.7616` | `18.9276` | `1.92e-08` | 1 |
| `A4-E1-GT4-recompute-hidden-cache` | `1.0234` | `1.5722` | `0.9145` | `18.7817` | `1.92e-08` | 0 |
| `A5-GT4-v616-kernel-with-v617-runner` | `1.0312` | `1.4785` | `0.7559` | `18.9276` | `1.92e-08` | 1 |
| `A6-GT4-v617-kernel-with-v616-timing-protocol` | `1.0312` | `1.4727` | `0.7548` | `18.9276` | `1.92e-08` | 1 |

P0 判断：

- v6.16 GT4 baseline 在 v6.18 中复现通过，`baseline_step_drift_resolved=true`。
- v6.17 E1 的 memory 改善真实存在，但 step drift 也复现：`1.5722` 高于 GT4 baseline。
- DWM2 patch 主线继续冻结；本轮只围绕 v6.16 fused grouped GT4 path 做 margin repair。

## 5. P1 Efficiency Margin Attribution

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

- E1 确实降低了 absolute peak allocated：`18.9276 MB -> 18.7817 MB`。
- 该改善幅度只有约 `0.75%` vs GT4，属于 memory margin diagnostic。
- E1 的主要代价仍在 backward/step；不是显存记录缺失，也不是梯度错误。

## 6. P2 Step-Preserving Memory Repair

P2 覆盖：

```text
18 shapes x (MLP reference + 2 measured candidates + 7 not-implemented candidates) = 180 detail rows
```

Measured candidates：

| package | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | peak allocated MB | grad relerr max | S2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M0-GT4-baseline` | `1.0312` | `1.5228` | `0.7849` | `1.2331` | `18.9276` | `1.92e-08` | 0 |
| `M8-E1-recompute-hidden-cache` | `1.0234` | `1.5907` | `0.9419` | `1.1889` | `18.7817` | `1.92e-08` | 0 |

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

- `M8/E1` 是真实 measured memory repair，memory 更低。
- 但 `M8` step ratio mean 为 `1.5907`，比 GT4 baseline 更差，未恢复 S2。
- `M0` 在 P2 协议下 step ratio `1.5228`，也略高于 S2 gate `1.50`。

## 7. P3 Memory-Preserving Step Repair

P3 覆盖：

```text
18 shapes x (MLP reference + 2 measured candidates + 6 not-implemented candidates) = 162 detail rows
```

Measured candidates：

| package | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | peak allocated MB | grad relerr max | S2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `T0-GT4-baseline` | `1.0312` | `1.5375` | `0.8365` | `1.1980` | `18.9276` | `1.92e-08` | 0 |
| `T7-backward-no-recompute` | `1.0312` | `1.5332` | `0.8357` | `1.1816` | `18.9276` | `1.92e-08` | 0 |

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

- `T7` 只带来很小 step 改善：`1.5375 -> 1.5332`。
- `T7` 没有达到 S2 的 `step <= 1.50`。
- 真正可能修 step margin 的 fastpath / launch-fused / update fastpath 均未实现，因此没有写 measured ratio。

## 8. P4 Combo Repair Selection

P4 最终候选：

| package | status | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | survivor |
|---|---|---:|---:|---:|---:|---|
| `C2-best-memory-preserving-step-repair` | measured | `1.0312` | `1.5375` | `0.8365` | `1.1980` | S3 |
| `C3-safe-memory+step-combo` | measured | `1.0234` | `1.5907` | `0.9419` | `1.1889` | S3 |

其他 combo：

| package | status | reason |
|---|---|---|
| `C0-GT4-baseline` | `not_run` | no eligible component for this combo |
| `C1-best-step-preserving-memory-repair` | `not_run` | no eligible component for this combo |
| `C4-S2-restore-combo` | `not_run` | no eligible component for this combo |
| `C5-S1-candidate-combo` | `not_run` | no eligible component for this combo |
| `C6-aggressive-S0-diagnostic` | `not_implemented` | aggressive S0 diagnostic not implemented |

P4 判断：

- v6.18 没有恢复 v6.16 的 S2。
- 最佳 memory candidate 是 `C3/M8-E1`，但 step ratio `1.5907`，只能是 S3。
- `C2/T0` 保留 GT4 memory，但 step ratio `1.5375`，仍不达 S2。
- 因此 P5 one-step、P6 diagnostic task、P8 official task 全部正确 gate。

## 9. P5-P10 Gate 状态

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

这部分不能解读成没有跑数据集。准确说法是：P1/P2/P3 已在 MNIST/Fashion-MNIST/KMNIST 上完成 profiler/full-step 测量；但 P4 没有 S0/S1/S2，所以按计划不允许进入 task loss/acc 训练阶段。

## 10. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F0_not_implemented_or_gated` | 20 | 未实现 margin repair 项，或 P5-P10 被 gate |
| `F2_step_time_fail` | 7 | measured candidate step ratio 未过 gate |
| `F13_official_task_gated` | 3 | task/optimizer/functional 正确 gate |
| `F1_memory_fail` | 1 | measured summary row memory ratio 未过对应 gate |
| `F4_baseline_step_drift` | 1 | 非 GT4/manual 对照中的 step drift 记录 |

没有出现：

- fake data failure
- proxy row failure
- gradient correctness failure
- artifact missing failure

## 11. Route Decision

`route_decision.json`：

```json
{
  "route": "R6-S2NotRestored",
  "best_candidate": "C3-safe-memory+step-combo",
  "best_family": "M8-E1-recompute-hidden-cache",
  "best_memory_ratio": 1.0234326652684669,
  "best_step_ratio": 1.5907257933383978,
  "best_backward_ratio": 0.9418785208611228,
  "best_forward_ratio": 1.1888704015983154,
  "memory_improvement_vs_GT4": 0.007527336540000005,
  "step_improvement_vs_GT4": -0.10268713403600495,
  "survivor_type": "S3",
  "one_step_probe_pass": 0,
  "official_task_opened": false,
  "diagnostic_task_opened": false,
  "best_val_acc": -1.0,
  "best_test_acc": -1.0,
  "val_acc_gap_vs_MLP": -99.0,
  "test_acc_gap_vs_MLP": -99.0,
  "task_gap_taxonomy": "not_available",
  "baseline_step_drift_resolved": true,
  "open_optimizer_exploration": false,
  "open_functional_correction": false,
  "primary_blocker": "efficiency_margin",
  "next_required_implementation": "kernel_margin_repair",
  "no_fake": true,
  "no_proxy": true
}
```

## 12. 结论

本轮 v6.18 支持以下真实结论：

1. v6.18 真实执行了 MNIST/Fashion-MNIST/KMNIST 上的 full profiler 测量；不是只写 gate artifact。
2. W&B 中已显式记录 `memory/KAN_peak_MB`、`memory/MLP_peak_MB`、`memory/peak_allocated_MB`、`memory/peak_reserved_MB`，不是只有 ratio。
3. v6.16 GT4 baseline 复现通过，baseline step drift 在 P0 层面已解决。
4. E1 hidden-cache recompute 的 memory 改善真实存在：mean memory ratio `1.0312 -> 1.0234`，absolute peak `18.9276 MB -> 18.7817 MB`。
5. 但 E1 的 step 明显变差：P2 中 step ratio mean `1.5907`，超过 S2 gate。
6. P3 的 step repair 只把 step 修到 `1.5332`，仍高于 `1.50`。
7. 本轮没有 S0/S1/S2 candidate，因此 P5-P10 正确 gate，没有 one-step、diagnostic task、official task、optimizer 或 functional correction 结论。
8. 最终 route 是 `R6-S2NotRestored`。

最终一句话：

> v6.18 跑了真实数据集和真实 fused GT4 margin profiler，也把绝对显存记录到了 W&B；但它没有恢复 S2。memory repair 继续有效但太慢，step repair 不够强，因此 task 训练按 gate 关闭是正确结果，不是漏跑。

## 13. 下一步建议

1. 不要用 E1 作为 task candidate；它只是 memory diagnostic。
2. 下一轮必须实现真实 kernel margin repair，例如 grad/update buffer reuse、launch-fused step 或 update-temp reuse，而不是继续 backward recompute。
3. 恢复 S2 的最低目标仍是 `memory <= 1.05` 且 `step <= 1.50`；要开 official task 还必须进一步达到 S1/S0。
4. W&B 后续继续保留 absolute memory metrics，避免只看 ratio。
5. P8/P9/P10 继续关闭，直到重新出现真实 S2/S1/S0 candidate。
