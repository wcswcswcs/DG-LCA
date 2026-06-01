# DG-KAN v6.11 DWM2Stop BoundedPrimitiveReset 结果复盘

> 本复盘只记录 `results/real_rerun_20260505/v611_real_all_20260505T054051Z` 下的 final real-only run。运行启动于 2026-05-05 05:40 UTC，完成于 2026-05-05 05:43 UTC。所有结论来自该目录落盘的 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v611_real.py \
  --packages V6_11_ALL \
  --out-dir results/real_rerun_20260505/v611_real_all_20260505T054051Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v611-real-20260505 \
  --wandb-name-prefix v611-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260505/v611_real_all_20260505T054051Z`
- 本地主日志：`results/real_rerun_20260505/v611_real_all_20260505T054051Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/xc7knjjx>
- W&B run name：`v611-real-v611_real_all_20260505T054051Z`

### 1.3 不用于结论的检查

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `/tmp/v611_smoke` | 极小规模 smoke，检查 runner/no-fake/CSV shape | 0 |
| `/tmp/v611_smoke2` | 修正 reset gate 后的极小规模 smoke | 0 |

### 1.4 no-fake / no-proxy 约束

- 新入口：`experiments/run_gafu_v611_real.py`。
- 所有 dataset loader 调用均使用 `allow_fake_data=False`。
- `experiments/dgkan_core.py` 中 fake data 路径仍保持 hard-fail。
- P0/P1/P2/P3/P4/P5/P6/P7/P8 的 `fake_data_used` 总和均为 `0`。
- P0/P1/P2/P3/P4/P5/P6/P7/P8 的 `proxy_row_used/proxy_rows_used` 总和均为 `0`。
- DWM2 terminal true single-call full-step 候选没有真实实现，因此全部写为 `not_implemented`，没有写入假 memory/time ratio。
- P5/P6/P7/P8 因 gate 写为 `not_run`，没有包装成 task 失败或成功数值。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_11_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler/kernel 使用 `seed=0`；无 task multi-seed confirm |
| full warmup / measure | 每个 full-step measured row `50/200` update steps |
| micro warmup / measure | `50/200` calls |
| run duration | `143.36` sec |
| Triton | available |
| memory history | available |
| Nsight | `ncu_available=1`，`nsys_available=0` |
| W&B logging | 记录 P0-P4/P11/failure/route summary；P6 task 被 gate，未产生每 20 step loss trace |

本轮真实新增实现：

| item | 实现说明 |
|---|---|
| P1 stable taxonomy attribution | 在 v6.10 memory history snapshot 基础上，把 allocation stack/source 映射到 `v611_stack_taxonomy_v1` |
| `ResetV4-lowrank-r4-poly1` | 低秩 bounded reset primitive，manual forward/backward/update |
| `ResetV4-lowrank-r8-poly1` | 低秩 bounded reset primitive，rank=8 |
| `ResetV4-chunked-poly1-c16` | chunked mixing bounded reset primitive，chunk=16 |
| `ResetV4-chunked-poly1-c32` | chunked mixing bounded reset primitive，chunk=32 |

仍未实现但按计划显式记录：

```text
DWM2 terminal single-call depth2
DWM2 terminal single-call layer
DWM2 terminal single-call full-step
DWM2 terminal one-buffer full-step
DWM2 terminal chunked-mix full-step
ResetV4 piecewise2 / grouped / streamingGrad variants
```

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 11 |
| `p0_reproduction_check.csv` | 真实生成 | 1 |
| `p1_taxonomy_attribution.csv` | 真实生成 | 36 |
| `p2_dwm2_terminal_full_step.csv` | 真实生成 | 6 |
| `p2_dwm2_terminal_full_step_detail.csv` | 真实生成 | 126 |
| `p3_reset_v4_family.csv` | 真实生成 | 12 |
| `p3_reset_v4_family_detail.csv` | 真实生成 | 234 |
| `p4_reset_component_audit.csv` | 真实生成 | 7 |
| `p5_one_step_probe.csv` | `not_run` | 1 |
| `p6_task_reentry.csv` | `not_run` | 1 |
| `p6_task_trace.csv` | `not_run` | 1 |
| `p7_optimizer_exploration.csv` | `not_run` | 1 |
| `p8_functional_correction_smoke.csv` | `not_run` | 1 |
| `failure_table.csv` | 真实生成 | 42 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |

关键 hash（复盘时重新计算）：

| 文件 | SHA256 |
|---|---|
| `run.log` | `1017df071cf88583f46c517ca44b83180c37ddc79dddc732aae8d602bf7591f1` |
| `p1_taxonomy_attribution.csv` | `9c2872779d7f6dcd4985e6c0ca3717fb8099445644d3255fbbc29fc9184d07e6` |
| `p2_dwm2_terminal_full_step.csv` | `6c77209b56a7f517b56f9a74b62ec6f4386a9d7ff40594c3981cf6c327a033e1` |
| `p3_reset_v4_family.csv` | `72fab34ce44df4a89e5fdbcb6095a3e9cf190ce24bdec3603b5eb6a55129f172` |
| `p4_reset_component_audit.csv` | `bf55c4cf1c8b93becf88106c7afe071d8c9ec13a9ee3a0dfd4da6ae03531a1df` |
| `failure_table.csv` | `f691de2dc8a9bb31844abef285c3167125866d04b9ba154712f3655d245518ef` |
| `route_decision.json` | `271a48ce958b4a08b5e631e6ee59756dc7a8bc56b0f97544436cf8eb6bd0381c` |

## 4. P0 Contract / v6.10 Reproduction

P0 contract：

| variant | status | implementation | nonKAN | manual backward | grad pass | grad relerr |
|---|---|---|---:|---:|---:|---:|
| `MLP-autograd-reference` | measured | reference | 55050 | 0 | - | - |
| `MLP-manual-linear-reference` | measured | manual-linear | 0 | 1 | 1 | `5.98e-08` |
| `DWM2-current` | measured | current | 0 | 1 | 1 | `1.52e-07` |
| `DWM2-K3-triton-fused-dx-coeffgrad-reference` | measured | disallowed local reference | 0 | 1 | 1 | `2.79e-07` |
| `DWM2-terminal-single-call-full-step` | `not_implemented` | terminal-dwm2 | - | - | - | - |
| `ResidualBoundedReset-v3-scale002-current` | measured | reset-v3 baseline | 0 | 1 | 1 | `9.88e-08` |
| `ResidualBoundedReset-v3-scale005-current` | measured | reset-v3 baseline | 0 | 1 | 1 | `1.00e-07` |
| `ResetV4-lowrank-r4-poly1` | measured | reset-v4 lowrank | 0 | 1 | 1 | `5.91e-08` |
| `ResetV4-lowrank-r8-poly1` | measured | reset-v4 lowrank | 0 | 1 | 1 | `7.60e-08` |
| `ResetV4-chunked-poly1-c16` | measured | reset-v4 chunked | 0 | 1 | 1 | `2.74e-07` |
| `ResetV4-chunked-poly1-c32` | measured | reset-v4 chunked | 0 | 1 | 1 | `2.84e-07` |

v6.10 reproduction：

| metric | v6.10 ref | v6.11 current | delta | pass |
|---|---:|---:|---:|---:|
| current memory ratio mean | `1.2917923089` | `1.2917923089` | `0.0` | 1 |
| current step ratio mean | `1.8969413769` | `1.9293781755` | `0.0324367986` | 1 |

P0 结论：v6.11 与 v6.10 memory baseline 可比；真实实现项梯度正确；未实现 terminal DWM2 没有被伪造成测量结果。

## 5. P1 Stable Taxonomy Attribution

P1 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x (MLP + current DWM2) = 36 rows
```

Current DWM2 summary：

| metric | min / mean / max |
|---|---:|
| memory ratio vs MLP | `1.1441 / 1.2918 / 1.4866` |
| step ratio vs MLP | `1.6035 / 1.9294 / 2.2681` |
| backward ratio vs MLP | `1.1799 / 1.4812 / 1.7762` |
| exact live tensor total MB | `3.7908 / 7.2872 / 11.7195` |
| peak gap MB | `2.5239 / 5.4430 / 9.2358` |
| top1 source fraction | `0.6902 / 0.7570 / 0.8980` |
| top3 taxonomy gap fraction | `1.0000 / 1.0000 / 1.0000` |
| unknown gap fraction | `0.0 / 0.0 / 0.0` |
| allocation trace events | `452 / 660.17 / 868` |
| attribution pass | `18/18` |

P1 判断：

- 本轮没有把 phase diagnostic 直接当成 attribution pass，而是使用 v6.10 的 memory history snapshot 和 v6.11 taxonomy mapper。
- `unknown_gap_fraction=0`，`top3_taxonomy_gap_fraction=1.0`，因此 H2/stable taxonomy gate 通过。
- 这只说明 attribution taxonomy 关闭；不代表 DWM2 memory/time gate 通过。

## 6. P2 Terminal DWM2 Full-Step

P2 summary：

| package | status | memory ratio mean | step ratio mean | backward ratio mean | memory improvement | step improvement | near pass |
|---|---|---:|---:|---:|---:|---:|---:|
| `D0-current` | measured | `1.2918` | `2.0249` | `1.6365` | `0.0` | `0.0` | 0 |
| `D1-terminal-single-call-depth2` | `not_implemented` | - | - | - | - | - | - |
| `D2-terminal-single-call-layer` | `not_implemented` | - | - | - | - | - | - |
| `D3-terminal-single-call-full-step` | `not_implemented` | - | - | - | - | - | - |
| `D4-terminal-onebuffer-full-step` | `not_implemented` | - | - | - | - | - | - |
| `D5-terminal-chunked-mix-full-step` | `not_implemented` | - | - | - | - | - | - |

P2 判断：

- v6.11 没有新增真实 DWM2 terminal single-call/full-step implementation。
- `dwm2_terminal_measured=0`，`dwm2_terminal_pass=0`。
- 按 v6.10 `STOP_DWM2_PATCHING` 和 v6.11 计划，本轮不能继续用局部 DWM2 patch 打开 task。

## 7. P3 Bounded Reset v4 Family

P3 覆盖：

```text
18 shapes x (DWM2 baseline + reset-v3 baseline + 4 measured reset-v4 + 6 not-implemented reset-v4) = 234 detail rows
```

Measured reset：

| primitive | family | memory ratio mean | step ratio mean | backward ratio mean | memory improvement vs current | step improvement vs current | residual/base | residual pass | near pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `DWM2-current-baseline` | DWM2 | `1.2918` | `1.9282` | `1.5074` | `0.0` | `0.0` | `0.0010` | 0/18 | 0 |
| `R0-reset-v3-scale002-current` | reset-v3 | `1.2981` | `1.7915` | `1.2523` | `-0.52%` | `+7.00%` | `0.0200` | 18/18 | 0 |
| `A0-inplace-poly1-chunked-mix-c16` | FAM-A | `1.1954` | `2.2574` | `1.9263` | `+7.16%` | `-16.75%` | `0.0200` | 18/18 | 0 |
| `A1-inplace-poly1-chunked-mix-c32` | FAM-A | `1.1954` | `1.9766` | `1.5136` | `+7.16%` | `-2.43%` | `0.0200` | 18/18 | 0 |
| `B0-lowrank-r4-poly1` | FAM-B | `1.1285` | `2.0684` | `1.2993` | `+12.42%` | `-6.99%` | `0.0200` | 18/18 | 0 |
| `B1-lowrank-r8-poly1` | FAM-B | `1.1344` | `2.0531` | `1.3030` | `+11.96%` | `-6.36%` | `0.0200` | 18/18 | 0 |

未实现 reset-v4：

```text
A2-inplace-piecewise2-chunked-mix
B2-lowrank-r4-piecewise2
B3-lowrank-r8-piecewise2
C0-grouped-g4-poly1
C1-grouped-g8-poly1
D1-piecewise2-streamingGrad
```

P3 判断：

- `B0-lowrank-r4-poly1` 是本轮最佳 reset candidate，memory ratio mean 从 current `1.2918` 降到 `1.1285`，改善 `12.42%`。
- 但 `B0` 仍没有 memory near-pass：`1.1285 > 1.05`。
- `B0` step ratio mean `2.0684`，比 current 更慢约 `6.99%`，远高于 task re-entry gate。
- 所有 reset-v4 measured candidates 的 residual effect 均真实达标，但没有任何 near-pass。

## 8. P4 Reset Component Audit

P4 对最佳 candidate `B0-lowrank-r4-poly1` 做 component audit。该表来自 full-step phase fields，标记为 `measured_from_full_step_phase_fields`，不是低层 kernel counter。

| component | component peak MB | component time ms | repair target |
|---|---:|---:|---:|
| `residual_transform` | `2.1055` | `1.5949` | 1 |
| `mixing_forward` | `2.1055` | `1.5949` | 1 |
| `backward_mixing_delta` | `1.0527` | `0.6496` | 1 |
| `backward_residual_dx` | `1.0527` | `0.6496` | 1 |
| `loss_delta` | `0.0` | `0.0` | 0 |
| `update_prep` | `0.0` | `0.0` | 0 |
| `optimizer_update` | `0.0` | `0.0` | 0 |

P4 判断：

- 主要 repair target 仍集中在 residual transform / mixing forward / backward mixing-delta / backward residual-dx。
- 由于 component allocation/kernel count 为 `metric_unavailable`，P4 不能被解读为低层 Nsight kernel 分解。
- P4 支持的实际结论是：低秩 reset 已经降低了 peak，但 runtime 仍偏重，且 memory 仍未到 near-pass。

## 9. P5-P8 Gate 状态

| stage | status | reason |
|---|---|---|
| P5 one-step probe | `not_run` | No DWM2 terminal near-pass and no reset-v4 near-pass |
| P6 task re-entry | `not_run` | P5 did not pass or no official memory/time survivor |
| P6 task trace | `not_run` | P6 is gated |
| P7 optimizer exploration | `not_run` | P6 official task did not pass |
| P8 functional correction smoke | `not_run` | P8 is gated behind P6/P7 |

这部分不能解读成 task/optimizer/functional 失败；准确说法是 v6.11 memory/time gate 没有打开，所以按计划没有进入训练。W&B 中因此没有每 20 step task loss 曲线。

## 10. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F0_not_implemented_or_gated` | 8 | reset-v4 未实现项或 P5/P6 后续阶段 gate |
| `F1_memory_fail` | 7 | measured DWM2/reset rows memory ratio 未过 gate |
| `F2_step_time_fail` | 7 | measured DWM2/reset rows step ratio 未过 gate |
| `F8_reset_memory_fail` | 6 | reset-v4/residual reset memory 未 near-pass |
| `F9_reset_step_fail` | 6 | reset-v4/residual reset step 未 near-pass |
| `F6_dwm2_terminal_not_implemented` | 5 | DWM2 terminal true full-step 候选未实现 |
| `F7_reset_residual_effect_fail` | 1 | DWM2-current baseline residual effect 不达标 |
| `F11_optimizer_gated` | 1 | optimizer exploration 正确 gate |
| `F12_functional_gated` | 1 | functional correction 正确 gate |

没有出现：

- fake data failure
- proxy row failure
- gradient correctness failure
- artifact missing failure
- attribution incomplete failure

## 11. Route Decision

`route_decision.json`：

```json
{
  "route": "R6-ResetV4TooHeavy",
  "best_candidate": "B0-lowrank-r4-poly1",
  "best_family": "FAM-B",
  "best_memory_ratio": 1.1284832216701082,
  "best_step_ratio": 2.068419612198206,
  "best_backward_ratio": 1.2992681947113272,
  "memory_improvement_vs_current": 0.12421622783586556,
  "step_improvement_vs_current": -0.06992669576317546,
  "survivor_type": "S6",
  "dwm2_terminal_measured": 0,
  "dwm2_terminal_pass": 0,
  "reset_v4_measured": 1,
  "reset_v4_pass": 0,
  "attribution_pass": 1,
  "open_task_reentry": false,
  "open_optimizer_exploration": false,
  "open_functional_correction": false,
  "stop_dwm2_patching": 1,
  "next_required_implementation": "new_bounded_workspace_primitive_family",
  "no_fake": true,
  "no_proxy": true
}
```

## 12. 结论

本轮 v6.11 支持以下真实结论：

1. v6.11 与 v6.10 current memory baseline 完全一致，结果可比较。
2. P1 stable taxonomy attribution 已通过：current DWM2 `attribution_pass=18/18`，`unknown_gap_fraction=0`。
3. DWM2 terminal true single-call/full-step 候选仍未实现，因此 DWM2 主线没有新的可测 terminal survivor。
4. Reset-v4 确实测到了更好的 memory signal：最佳 `B0-lowrank-r4-poly1` memory ratio mean 为 `1.1285`，比 current 改善 `12.42%`。
5. 但 `B0` 仍然没有 near-pass，且 step ratio mean `2.0684`，比 current 更慢。
6. 所有 reset-v4 measured candidates residual effect 都真实达标，但 memory/time 仍不达标。
7. P5-P8 正确 gate，没有 task、optimizer、functional 结论。
8. 最终 route 是 `R6-ResetV4TooHeavy`，并保持 `stop_dwm2_patching=1`。

最终一句话：

> v6.11 成功把 attribution blocker 变成可解释 taxonomy，并证明 low-rank reset-v4 有真实 memory 改善；但它仍太重、太慢，没有达到 near-pass。DWM2 terminal kernel 未实现，DWM2 patch 主线继续停止，下一步应设计新的 bounded-workspace primitive family，而不是打开 task 或继续 DWM2 小修。

## 13. 下一步建议

1. 不再做 DWM2 局部 patch；除非真正实现 terminal single-call full-step，否则 DWM2 保持冻结。
2. 以 `B0-lowrank-r4-poly1` 为 reset-v4 的正向线索，但必须重新设计 runtime：目标是把 step ratio 从 `2.07` 降到 `<=1.35`，并把 memory ratio 从 `1.1285` 降到 `<=1.05`。
3. 新 primitive family 应优先控制 mixing/workspace lifetime，而不是继续只调 residual scale。
4. P6 task、P7 optimizer、P8 functional correction 继续关闭，直到出现真实 memory/time survivor。
5. 下一轮如果继续 reset，应把 component audit 升级为真实 allocation/kernel counter，而不是只使用 full-step phase fields。
