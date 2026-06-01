# DG-KAN v6.20 GT4 CanonicalProtocol NonRecomputeRepair 结果复盘

> 本复盘只记录 `results/real_rerun_20260505/v620_real_all_20260505T144340Z` 下的 final real-only run。运行启动于 2026-05-05 14:43:42 UTC，完成于 2026-05-05 14:47:41 UTC。所有结论来自该目录落盘的 CSV/JSON/log/manifest；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v620_real.py \
  --packages V6_20_ALL \
  --out-dir results/real_rerun_20260505/v620_real_all_20260505T144340Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v620-real-20260505 \
  --wandb-name-prefix v620-real
```

本轮 stdout/stderr 同时写入：

```text
results/real_rerun_20260505/v620_real_all_20260505T144340Z/run.log
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260505/v620_real_all_20260505T144340Z`
- 本地主日志：`results/real_rerun_20260505/v620_real_all_20260505T144340Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/jyxe3eev>
- W&B run name：`v620-real-v620_real_all_20260505T144340Z`

### 1.3 不用于结论的检查与自修复 run

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `/tmp/v620_smoke` | 初版 runner/gate/CSV shape smoke | 0 |
| `/tmp/v620_smoke2` | 修正 E1 不得进入主 combo 后的 smoke | 0 |
| `/tmp/v620_smoke3` | 调整 phase order / sync 后的 smoke | 0 |
| `/tmp/v620_smoke4` | P3-before-P2 与 cooldown 后的 smoke | 0 |
| `/tmp/v620_smoke5` | canonical P0 MLP denominator rebase 后的 smoke | 0 |
| `results/real_rerun_20260505/v620_real_all_20260505T132834Z` | 第一次 full run：P0 稳定 S2，但 P2/P3 drift 未闭合 | 0 |
| `results/real_rerun_20260505/v620_real_all_20260505T133252Z` | 第二次 full run：P2 恢复 S2，P3 drift 仍阻断 | 0 |
| `results/real_rerun_20260505/v620_real_all_20260505T133856Z` | 第三次 full run：P2 S2，P3 仍慢；随后定位为 MLP denominator 漂移并修复 | 0 |
| `results/real_rerun_20260505/v620_real_all_20260505T134556Z` | 旧 v6.20 final：global gate 未打开；用户要求继续修后被 supersede | 0 |
| `/tmp/v620_fast_smoke` | 实现 M3/T2/T3 fastpath 后的极小规模 smoke | 0 |
| `/tmp/v620_fast_gate_smoke` | 修正 global canonical gate 后的极小规模 smoke；受小样本 timing 噪声影响不用于结论 | 0 |
| `results/real_rerun_20260505/v620_faststep_all_20260505T141522Z` | no-W&B 完整验证：fastpath S2，但旧 gate 仍关 task；随后修 gate | 0 |
| `results/real_rerun_20260505/v620_faststep_all_20260505T142044Z` | no-W&B 完整验证：S2 恢复且 diagnostic task 打开；正式 W&B 前验证 | 0 |

本轮按用户要求“继续修，别停，直到完成目标”，实际做了额外实现和重跑：

| attempt | 修改 | 结果 |
|---|---|---|
| 1 | 实现 policy-gated fast optimizer：缓存 params/grads、去掉 CPU update-norm 诊断开销 | `M3/T2` 变成真实 measured，step 显著下降 |
| 2 | 实现 `T3-launch-fused-step`：使用 `torch._foreach_*` 进行 AdamW update | `T3` step 降到 S2 内 |
| 3 | 修正 global canonical gate：protocol gate 只判协议一致性，是否开 task 由最终 S0/S1/S2 candidate 决定 | P5 one-step 和 P8 diagnostic task 正确打开 |

### 1.4 no-fake / no-proxy 约束

- 新入口：`experiments/run_gafu_v620_real.py`。
- 所有 dataset loader 调用均使用 `allow_fake_data=False`。
- `experiments/dgkan_core.py` 中 fake data 路径仍保持 hard-fail。
- P0-P10 的 `fake_data_used` 总和为 `0`。
- P0-P10 的 `proxy_row_used/proxy_rows_used` 总和为 `0`。
- P2/P3 的 canonical denominator rebase 使用 P0 中同一 shape 的真实 MLP autograd measured row；detail CSV 保留 `local_*_ratio_vs_MLP` 与 `canonical_ratio_basis`。
- `M3-update-temp-reuse`、`T2-update-fastpath`、`T3-launch-fused-step` 已经是真实 measured code path，不再写 `not_implemented`。
- 仍未实现的 `M1/M2/M4-M8`、`T1/T4-T6`、aggressive S0 diagnostic、部分 optimizer/expressivity repairs 写为 `not_implemented` 或 `not_run`，没有写假 ratio。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_20_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler 使用 `seed=0`；diagnostic task 使用 seeds `0,1,2` |
| full warmup / measure | 每个 full-step measured row `50/200` update steps |
| task steps | `120`，每 `20` steps 记录 trace |
| run duration | `238.64` sec |
| source commit | `f76ff27ea03c6fbfc01cadf24ea281c32a72b18f` |
| Triton | available |
| memory history | available |
| W&B logging | 记录 P0-P10/failure/route summary；包含 absolute memory metrics 和 P8 diagnostic task trace |

数据集覆盖审计：

| artifact | rows | datasets | batch sizes | depths |
|---|---:|---|---|---|
| `p0_contract.csv` | 162 | MNIST/Fashion-MNIST/KMNIST | 128/256/512 | 2/4 |
| `p1_protocol_overhead_attribution.csv` | 108 | MNIST/Fashion-MNIST/KMNIST | 128/256/512 | 2/4 |
| `p2_nonrecompute_memory_repair_detail.csv` | 198 | MNIST/Fashion-MNIST/KMNIST | 128/256/512 | 2/4 |
| `p3_step_repair_detail.csv` | 162 | MNIST/Fashion-MNIST/KMNIST | 128/256/512 | 2/4 |
| `p8_task_reentry.csv` | 36 | MNIST/Fashion-MNIST/KMNIST | task batch 128 | task depth 2 |
| `p8_task_trace.csv` | 216 | MNIST/Fashion-MNIST/KMNIST | task batch 128 | task depth 2 |

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 162 |
| `p0_reproduction_check.csv` | 真实生成 | 7 |
| `p0_runner_kernel_matrix.csv` | 真实生成 | 7 |
| `p1_protocol_overhead_attribution.csv` | 真实生成 | 108 |
| `p1_component_margin_audit.csv` | 真实生成 | 90 |
| `p2_nonrecompute_memory_repair.csv` | 真实生成 | 10 |
| `p2_nonrecompute_memory_repair_detail.csv` | 真实生成 | 198 |
| `p3_step_repair.csv` | 真实生成 | 8 |
| `p3_step_repair_detail.csv` | 真实生成 | 162 |
| `p4_combo_repair_selection.csv` | 真实生成 | 7 |
| `p5_one_step_probe.csv` | 真实生成 | 1 |
| `p6_diagnostic_task_gap_attribution.csv` | 真实生成 | 2 |
| `p6_task_trace.csv` | 真实生成 | 216 |
| `p7_one_step_probe.csv` | 真实生成 | 1 |
| `p7_optimizer_expressivity_diagnostic.csv` | 真实生成 | 10 |
| `p8_task_reentry.csv` | diagnostic measured | 36 |
| `p8_task_trace.csv` | diagnostic measured | 216 |
| `p9_optimizer_exploration.csv` | `not_run` | 1 |
| `p10_functional_correction_smoke.csv` | `not_run` | 1 |
| `failure_table.csv` | 真实生成 | 25 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |
| `run_manifest.json` | 真实生成 | - |

关键 hash：

| 文件 | SHA256 |
|---|---|
| `run.log` | `e6e52c413bfbee73b169d1037049a3b21fb03fedd0742602dab9c51967569135` |
| `run_manifest.json` | `cfe94e5d89297c631ca8f25a3cf2d431ad15b7dce0081ad871d0ec3e5a443fe7` |
| `p0_reproduction_check.csv` | `4cd59670ddc773da3f9e91befc42af33746a2fdfd3edeec2ece65ba6830f4b4a` |
| `p2_nonrecompute_memory_repair.csv` | `be1e92b66e841b0e84c898e99bc613258cb71dc906ea4f2e0d426f42c1fc751e` |
| `p3_step_repair.csv` | `20ecce907982c2b60c4a9c5c69a4fd9bed0f635c684b8f010fabb9f45a70b2c7` |
| `p4_combo_repair_selection.csv` | `b1abab3a30c97976e90dc29b986f7a90a46752699dd2fd85696cc037b09af79d` |
| `p5_one_step_probe.csv` | `99847c169b80b930b83f06605b57d8ff09b0f1f6a8e896ea12b58166a91fe77a` |
| `p6_diagnostic_task_gap_attribution.csv` | `3157fde4f16bbdcc82fbb1e32b2f697d6e20e30b319eafa36da44195f35c41ea` |
| `p7_optimizer_expressivity_diagnostic.csv` | `1931f38e82ff75a9a59bf81d3ff37fbab24cdbb0d2b3df75905b5d0a14a9f676` |
| `p8_task_reentry.csv` | `67a91162075f1664ea5f01435015323c275983c3d4f89c90d96a7399b843783f` |
| `p8_task_trace.csv` | `5d099650ac2712f4c529f4fa98dfb48335d3bf6f9f33f18a2ef98c8bb698e4d6` |
| `failure_table.csv` | `1e97beba84d889902bf4e0f2172b4ed81f5e7788543d599ebcb5b1cf69c63bdd` |
| `route_decision.json` | `01c58919044986e86205f59e8c4fe995716c2b8d1e73614ed076dab24dc888a2` |

## 4. P0 Canonical Protocol Lock

P0 reproduction summary：

| variant | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | peak allocated MB | survivor |
|---|---:|---:|---:|---:|---:|---|
| `A1-MLP-manual-linear-reference` | `1.1085` | `1.0839` | `0.6156` | `0.7855` | `20.3436` | S4 |
| `A2-GT4-protocol_A_v616_timing` | `1.0312` | `1.4160` | `0.6989` | `1.1893` | `18.9276` | S2 |
| `A3-GT4-protocol_B_v618_p0_side_by_side` | `1.0312` | `1.4157` | `0.6944` | `1.2050` | `18.9276` | S2 |
| `A4-GT4-protocol_C_v619_repair_protocol` | `1.0312` | `1.4114` | `0.6949` | `1.1932` | `18.9276` | S2 |
| `A5-GT4-protocol_D_new_canonical_minimal_logging` | `1.0312` | `1.4081` | `0.6902` | `1.1974` | `18.9276` | S2 |
| `A6-GT4-protocol_E_new_canonical_with_component_logging` | `1.0312` | `1.4104` | `0.6947` | `1.1880` | `18.9276` | S2 |
| `A7-E1-recompute-hidden-cache-diagnostic` | `1.0234` | `1.4799` | `0.8284` | `1.2021` | `18.7817` | S2 |

P0 判断：

- GT4 memory baseline 稳定：所有 GT4 protocol 的 memory ratio mean 都是 `1.0312`。
- P0 内所有 GT4 protocol 均为 S2。
- E1 recompute 本轮在 P0 中擦过 S2，但它仍不是主路线，因为后续 P2 中 step 不如 `M3/T3`。

## 5. P1 Protocol Overhead Attribution

P1 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x (MLP baseline + 5 measured variants) = 108 rows
component margin audit = 90 rows
```

P1 measured summary：

| variant | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | peak allocated MB |
|---|---:|---:|---:|---:|---:|
| `P1-MLP-manual-linear-reference` | `1.1085` | `1.1410` | `0.6684` | `0.8231` | `20.3436` |
| `P1-GT4-canonical-minimal` | `1.0312` | `1.4933` | `0.7593` | `1.2548` | `18.9276` |
| `P1-GT4-canonical-with-component-logging` | `1.0312` | `1.4947` | `0.7593` | `1.2547` | `18.9276` |
| `P1-GT4-repair-dispatch-wrapper` | `1.0312` | `1.5136` | `0.7534` | `1.2885` | `18.9276` |
| `P1-E1-recompute-hidden-cache` | `1.0234` | `1.5623` | `0.9090` | `1.1776` | `18.7817` |

P1 判断：

- P1 支持 GT4 baseline 的 memory 稳定性。
- E1 memory 改善真实存在，但 step/backward 代价也稳定存在。
- P1 没有 fake/proxy，也没有 gradient correctness failure。

## 6. P2 Non-Recompute Memory Repair

P2 覆盖：

```text
18 shapes x (MLP reference + 3 measured candidates + 7 not-implemented candidates) = 198 detail rows
```

Measured candidates：

| package | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | peak allocated MB | update ms mean | grad relerr max | survivor |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `M0-GT4-canonical-baseline` | `1.0312` | `1.4177` | `0.6970` | `1.1968` | `18.9276` | - | `1.92e-08` | S2 |
| `M3-update-temp-reuse` | `1.0312` | `1.0069` | `0.6813` | `1.1093` | `18.9276` | `0.5232` | `1.92e-08` | S2 |
| `M9-E1-recompute-hidden-cache` | `1.0234` | `1.4841` | `0.8366` | `1.1844` | `18.7817` | - | `1.92e-08` | S2 |

未实现：

```text
M1-grad-mix-buffer-reuse
M2-grad-poly-buffer-reuse
M4-triton-scratch-lifetime-trim
M5-reserved-memory-trim
M6-short-lived-output-free
M7-grad-update-buffer-combo
M8-safe-memory-combo
```

P2 判断：

- `M3-update-temp-reuse` 已经是真实 measured candidate。
- `M3` 保持 GT4 memory `1.0312`，同时把 step ratio 从 M0 的 `1.4177` 降到 `1.0069`。
- 这不是 fake/proxy；detail rows 的 `workspace_policy=reset_v10_update_temp_reuse`，梯度 relerr max `1.92e-08`。

## 7. P3 Step Repair

P3 覆盖：

```text
18 shapes x (MLP reference + 4 measured candidates + 4 not-implemented candidates) = 162 detail rows
```

Measured candidates：

| package | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | peak allocated MB | update ms mean | grad relerr max | survivor |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `T0-GT4-canonical-baseline` | `1.0312` | `1.4246` | `0.7091` | `1.1845` | `18.9276` | - | `1.92e-08` | S2 |
| `T2-update-fastpath` | `1.0312` | `1.0105` | `0.6850` | `1.1024` | `18.9276` | `0.5251` | `1.92e-08` | S2 |
| `T3-launch-fused-step` | `1.0312` | `0.8436` | `0.6822` | `1.0499` | `18.9276` | `0.3021` | `1.92e-08` | S2 |
| `T7-backward-no-recompute` | `1.0312` | `1.4159` | `0.7026` | `1.1701` | `18.9276` | - | `1.92e-08` | S2 |

未实现：

```text
T1-forward-fastpath-real
T4-layout-fastpath
T5-kernel-count-reduction
T6-step-combo-safe
```

P3 判断：

- `T2-update-fastpath` 已经恢复 S2。
- `T3-launch-fused-step` 是本轮最佳 step repair：step ratio mean `0.8436`，update mean `0.3021 ms`。
- `T3` 使用真实 `torch._foreach_*` AdamW update fastpath；模型 forward/backward 数学不变，梯度正确。

## 8. P4 Combo Selection / P5 One-Step

P4 selection：

| package | components | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | survivor |
|---|---|---:|---:|---:|---:|---|
| `C1-best-step-preserving-memory-repair` | `M0-GT4-canonical-baseline` | `1.0312` | `1.4177` | `0.6970` | `1.1968` | S2 |
| `C2-best-memory-preserving-step-repair` | `T0-GT4-canonical-baseline` | `1.0312` | `1.4246` | `0.7091` | `1.1845` | S2 |
| `C3-safe-memory+step-combo` | `T3-launch-fused-step` | `1.0312` | `0.8436` | `0.6822` | `1.0499` | S2 |
| `C4-S2-restore-combo` | `T3-launch-fused-step` | `1.0312` | `0.8436` | `0.6822` | `1.0499` | S2 |

P5 one-step probe：

| metric | value |
|---|---:|
| candidate | `T3-launch-fused-step` |
| loss before | `2.5434` |
| loss after | `2.4604` |
| loss delta | `-0.0831` |
| grad norm | `0.7044` |
| one-step pass | 1 |

P4/P5 判断：

- Global canonical gate 已通过：`canonical_protocol_pass=true`、`canonical_protocol_strict_pass=true`。
- `T3` 作为 S2 candidate 正确打开 P5 one-step。
- `T3` 未达到 official task-open 的 S0/S1，因为 memory ratio 仍为 `1.0312 > 1.00`；因此只打开 diagnostic task。

## 9. P8 Diagnostic Task / P6-P7 Gap

P8 diagnostic task：

| variant | rows | val acc mean | test acc mean | train loss delta mean |
|---|---:|---:|---:|---:|
| `MLP-autograd-reference` | 9 | `0.6981` | `0.6523` | `-2.3124` |
| `MLP-manual-linear-reference` | 9 | `0.6808` | `0.6337` | `-2.4837` |
| `T3-launch-fused-step+ManualAdamW` | 9 | `0.6387` | `0.6018` | `-2.5256` |
| `T3-launch-fused-step+ManualAdanLite` | 9 | `0.6280` | `0.5927` | `-2.5004` |

P8 trace：

```text
3 datasets x 3 seeds x 4 variants x 6 logged steps = 216 rows
logged steps = 20,40,60,80,100,120
task_mode = diagnostic
```

P6 task gap attribution：

| variant | val acc mean | test acc mean | val gap vs MLP | taxonomy |
|---|---:|---:|---:|---|
| `T3-launch-fused-step+ManualAdamW` | `0.6387` | `0.6018` | `-0.0595` | `T3-generalization_gap` |
| `T3-launch-fused-step+ManualAdanLite` | `0.6280` | `0.5927` | `-0.0701` | `T3-generalization_gap` |

P7 optimizer/expressivity diagnostic：

| candidate | status | val acc | test acc | useful |
|---|---|---:|---:|---:|
| `D0-best-S2-baseline-ManualAdanLite` | measured_from_P6_task | `0.6280` | `0.5927` | 0 |
| `D1-best-S2-ManualAdamW` | measured_from_P6_task | `0.6387` | `0.6018` | 0 |

P8/P6/P7 判断：

- 本轮已经真实进入 diagnostic task，不再是 gate-only。
- `T3` 训练 loss 下降真实存在，但 validation/test 明显低于 MLP reference。
- 当前 blocker 从 protocol/step margin 转为 task gap，taxonomy 为 `T3-generalization_gap`。

## 10. P9-P10 Gate 状态

| stage | status | reason |
|---|---|---|
| P9 optimizer exploration | `not_run` | P8 diagnostic task only or task did not pass |
| P10 functional correction smoke | `not_run` | gated behind P8/P9 |

这部分不能解读成 optimizer/functional 失败。准确说法是：v6.20 已得到 S2 并打开 diagnostic task，但 diagnostic task 没有显示 task improvement，因此 P9/P10 继续 gate。

## 11. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F0_not_implemented_or_gated` | 22 | 未实现的 M/T/C/P7 repair，或 P9/P10 gate |
| `F13_official_task_gated` | 2 | official optimizer/functional 正确 gate |
| `F1_memory_fail` | 1 | ManualLinear reference memory ratio 未过对应 gate |

没有出现：

- fake data failure
- proxy row failure
- gradient correctness failure
- artifact missing failure

## 12. Route Decision

`route_decision.json`：

```json
{
  "route": "R8-S2RestoredTaskGapPersists",
  "best_candidate": "C3-safe-memory+step-combo",
  "best_family": "T3-launch-fused-step",
  "best_memory_ratio": 1.0311948156844268,
  "best_step_ratio": 0.8435986563176388,
  "best_backward_ratio": 0.6821847831960708,
  "best_forward_ratio": 1.0499291143794571,
  "memory_improvement_vs_GT4": 0.0,
  "step_improvement_vs_GT4": 0.41521952525878647,
  "survivor_type": "S2",
  "one_step_probe_pass": 1,
  "official_task_opened": false,
  "diagnostic_task_opened": true,
  "best_val_acc": 0.638671875,
  "best_test_acc": 0.6017795138888888,
  "val_acc_gap_vs_MLP": -0.05946180555555558,
  "test_acc_gap_vs_MLP": -0.05056423611111116,
  "task_gap_taxonomy": "T3-generalization_gap",
  "baseline_step_drift_resolved": true,
  "p0_side_by_side_reproduction_pass": true,
  "canonical_protocol_pass": true,
  "canonical_protocol_strict_pass": true,
  "canonical_protocol_basis": "v620_protocol_lock_P0_P2_P3",
  "canonical_step_delta_P2_M0_vs_P0_GT4": 0.005343337875980181,
  "canonical_step_delta_P3_T0_vs_P0_GT4": 0.01226101637121535,
  "open_optimizer_exploration": false,
  "open_functional_correction": false,
  "primary_blocker": "task_gap",
  "next_required_implementation": "task_gap_repair",
  "no_fake": true,
  "no_proxy": true
}
```

## 13. 结论

本轮 v6.20 支持以下真实结论：

1. v6.20 真实跑了 MNIST/Fashion-MNIST/KMNIST 上的 profiler/full-step 测量和 diagnostic task。
2. 用户要求继续修后，`M3-update-temp-reuse`、`T2-update-fastpath`、`T3-launch-fused-step` 已经真实实现并测量，不再是 `not_implemented`。
3. `T3-launch-fused-step` 成为最佳 candidate：memory ratio mean `1.0312`，step ratio mean `0.8436`，backward ratio mean `0.6822`，gradient relerr max `1.92e-08`。
4. Global canonical gate 已通过：`canonical_protocol_pass=true`，`canonical_step_delta_P2=0.0053`，`canonical_step_delta_P3=0.0123`。
5. P5 one-step probe 真实通过：loss delta `-0.0831`。
6. P8 diagnostic task 真实运行并写入 36 行 summary、216 行 trace。
7. `T3+ManualAdamW` 与 `T3+ManualAdanLite` 的 loss 都下降，但 validation/test 仍低于 MLP reference。
8. 本轮最终 blocker 已从 protocol/step margin 转为 task gap，taxonomy 是 `T3-generalization_gap`。
9. 因为 `T3` 是 S2 而不是 S0/S1，official task re-entry 仍关闭；P9/P10 继续 gate 是正确结果。
10. 最终 route 是 `R8-S2RestoredTaskGapPersists`。

最终一句话：

> v6.20 通过真实 update fastpath / launch-fused step 修复了 P3 step margin，并恢复 S2：`T3-launch-fused-step` 把 step ratio 压到 `0.8436`，打开 one-step 和 diagnostic task。新的主要问题不再是 kernel/protocol gate，而是 task gap：诊断训练能降 loss，但 val/test 仍落后 MLP。

## 14. 下一步建议

1. 不要再回到 E1 recompute 或单纯 timing protocol 修补；v6.20 的 step/protocol 目标已经达成。
2. 下一轮主线应进入 task gap repair：优先分析 `T3` 的 generalization gap、optimizer behavior 和 expressivity。
3. 可先做低风险 optimizer diagnostics：学习率、weight decay、AdanLite variants、label smoothing，但必须继续 no-fake/no-proxy。
4. 若要打开 official task，仍需把 memory 从 `1.0312` 推到 `<1.00` 或达到 S1/S0；否则保持 diagnostic task 模式。
5. W&B 后续继续保留 absolute memory metrics 和 P8 trace，避免只看 ratio 或 summary。
