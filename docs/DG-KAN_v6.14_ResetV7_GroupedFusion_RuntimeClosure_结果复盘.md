# DG-KAN v6.14 ResetV7 GroupedFusion RuntimeClosure 结果复盘

> 本复盘只记录 `results/real_rerun_20260505/v614_real_all_20260505T081002Z` 下的 final real-only run。运行启动于 2026-05-05 08:10 UTC，完成于 2026-05-05 08:20 UTC。所有结论来自该目录落盘的 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v614_real.py \
  --packages V6_14_ALL \
  --out-dir results/real_rerun_20260505/v614_real_all_20260505T081002Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v614-real-20260505 \
  --wandb-name-prefix v614-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260505/v614_real_all_20260505T081002Z`
- 本地主日志：`results/real_rerun_20260505/v614_real_all_20260505T081002Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/p4p2m36t>
- W&B run name：`v614-real-v614_real_all_20260505T081002Z`

### 1.3 不用于结论的检查

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `/tmp/v614_smoke` | 极小规模 smoke，检查 runner/no-fake/new grouped candidates/CSV shape | 0 |
| `mpd5fuej` / `v614_real_all_20260505T075747Z` | 第一次 full run 发现 P1 grouped code-loop count 未回填，route 解释有诊断缺陷；修正后重跑 | 0 |

### 1.4 no-fake / no-proxy 约束

- 新入口：`experiments/run_gafu_v614_real.py`。
- 所有 dataset loader 调用均使用 `allow_fake_data=False`。
- `experiments/dgkan_core.py` 中 fake data 路径仍保持 hard-fail。
- P0-P10 的 `fake_data_used` 总和为 `0`。
- P0-P10 的 `proxy_row_used/proxy_rows_used` 总和为 `0`。
- DWM2 继续作为 frozen baseline，没有新增 DWM2 patch。
- 未实现的 fused grouped/Triton/CUDA/low-rank one-buffer/piecewise residual-only 候选写为 `not_implemented` 或 `not_implemented_cuda_extension_absent`；P7-P10 因 gate 写为 `not_run`。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_14_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler/kernel 使用 `seed=0`；无 task multi-seed confirm |
| full warmup / measure | 每个 full-step measured row `50/200` update steps |
| run duration | `637.53` sec |
| source commit | `f76ff27ea03c6fbfc01cadf24ea281c32a72b18f` |
| Triton | available |
| memory history | available |
| Nsight | `ncu_available=1`，`nsys_available=0` |
| W&B logging | 记录 P0-P6/P11/failure/route summary；P8 task 被 gate，未产生每 20 step task loss trace |

本轮真实新增实现：

| item | 实现说明 |
|---|---|
| `VG2-memory-preserving-vectorized-g16` | grouped einsum 中避免显式 materialize `xg = x * factor` |
| `VG3-tiled-vectorized-g16-tile4` | 按 group tile 做 bounded materialization |
| `VG7-batched-group-gemm-no-materialize` | 复用 no-xg grouped einsum 做 batched group GEMM diagnostic |
| `GK7/GK8 no-xg custom diagnostic` | g8/g16 no-xg grouped einsum backend diagnostic |

仍未实现但按计划显式记录：

```text
fused grouped forward/backward/full
single-kernel grouped mix
Triton grouped forward/backward/full
CUDA grouped forward-backward extension
lowrank one-buffer / Triton / full-fused
piecewise residual-only / noMix / custom-knot
```

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 9 |
| `p0_reproduction_check.csv` | 真实生成 | 1 |
| `p1_reset_runtime_attribution.csv` | 真实生成 | 198 |
| `p1_component_kernel_audit.csv` | 真实生成 | 1980 |
| `p2_memory_preserving_grouped_fusion.csv` | 真实生成 | 10 |
| `p2_memory_preserving_grouped_fusion_detail.csv` | 真实生成 | 198 |
| `p3_custom_grouped_kernel_diagnostic.csv` | 真实生成 | 9 |
| `p3_custom_grouped_kernel_diagnostic_detail.csv` | 真实生成 | 180 |
| `p4_lowrank_r2_runtime_closure.csv` | 真实生成 | 11 |
| `p4_lowrank_r2_runtime_closure_detail.csv` | 真实生成 | 216 |
| `p5_piecewise_residual_vs_mixing.csv` | 真实生成 | 8 |
| `p5_piecewise_residual_vs_mixing_detail.csv` | 真实生成 | 162 |
| `p6_reset_v7_selection.csv` | 真实生成 | 1 |
| `p7_one_step_probe.csv` | `not_run` | 1 |
| `p8_task_reentry.csv` | `not_run` | 1 |
| `p8_task_trace.csv` | `not_run` | 1 |
| `p9_optimizer_exploration.csv` | `not_run` | 1 |
| `p10_functional_correction_smoke.csv` | `not_run` | 1 |
| `failure_table.csv` | 真实生成 | 2037 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |

关键 hash（复盘时重新计算）：

| 文件 | SHA256 |
|---|---|
| `run.log` | `b26cd0bab663b34718ad1d5c39574b1c800147b55c8e7b1caabe18402eb8bfe3` |
| `p1_component_kernel_audit.csv` | `923bb68b5f38471d8ba6e160f9842cc6d77e2c19676bb7a5b445d6904c10213b` |
| `p2_memory_preserving_grouped_fusion.csv` | `51137f8746bc6b0dfe726a64531e6b61e908e641994ec65e8d3b391ef21293cb` |
| `p3_custom_grouped_kernel_diagnostic.csv` | `2b1b808e3428c083f4ab876ec0b3ac1afb471d669e25c31b7c1a3c3ead7941ba` |
| `p4_lowrank_r2_runtime_closure.csv` | `5c2d153f6b4f2142bd0271c3d1d8c30edd56ea5661a4814c5418908a56b1337f` |
| `p5_piecewise_residual_vs_mixing.csv` | `43d102327b77ee3d1b30b01905219f03b2d17b0c6cd6e5ba59353480fd81eb8e` |
| `p8_task_reentry.csv` | `4c671bfe0d2e560730d7f0f5cb690a5692c355e9feeca1539700f4278883673d` |
| `p8_task_trace.csv` | `0ef7e781bd48fe2b0635f7dad3c928476b5638effd5bd7aba49b9fcd369926b8` |
| `failure_table.csv` | `a4af90fb379c11ccb34c34fd7ff8c9c76ff8a5b024fec92055dfe58c05cd3542` |
| `route_decision.json` | `d956d0cfe875bed09ce65d986b03f995d4ba3693d94b79e3b0976a7e527e9dd0` |

## 4. P0 Contract / v6.13 Reproduction

P0 contract：

| variant | status | nonKAN | manual backward | fake/proxy |
|---|---|---:|---:|---:|
| `MLP-autograd-reference` | measured | 55050 | 0 | 0/0 |
| `MLP-manual-linear-reference` | measured | 0 | 1 | 0/0 |
| `DWM2-current-baseline` | measured frozen baseline | 0 | 1 | 0/0 |
| `L6-lowrank-r2-fast-v613` | measured lineage baseline | 0 | 1 | 0/0 |
| `G2-grouped-g16-poly1-v613` | measured lineage baseline | 0 | 1 | 0/0 |
| `GR3-vectorized-g16-v613` | measured lineage baseline | 0 | 1 | 0/0 |
| `PW1-piecewise2-streamingGrad-v613` | measured lineage baseline | 0 | 1 | 0/0 |
| `ResetV7-memory-preserving-vectorized-g16` | measured | 0 | 1 | 0/0 |
| `ResetV7-tiled-vectorized-g16-tile4` | measured | 0 | 1 | 0/0 |

v6.13 reproduction：

| metric | v6.13 ref | v6.14 reproduction | delta | pass |
|---|---:|---:|---:|---:|
| DWM2 current memory ratio mean | `1.2917923089` | `1.2917923089` | `0.0` | 1 |
| L6 memory ratio mean | `1.0785729221` | `1.0785729221` | `0.0` | 1 |
| L6 step ratio mean | `2.0141154785` | `1.9991722991` | `-0.0149431794` | 1 |
| G2 memory ratio mean | `1.0459591815` | `1.0459591815` | `0.0` | 1 |
| GR3 memory ratio mean | `1.1748654001` | `1.1748654001` | `0.0` | 1 |

P0 结论：v6.14 与 v6.13 memory baseline 可比；DWM2 patch 主线继续冻结；新增 ResetV7 grouped candidates 是真实 code path。

## 5. P1 Runtime Attribution

P1 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x (MLP/manual/DWM2/lowrank/grouped/vectorized/tiled/piecewise) = 198 rows
```

### 5.1 full-step summary

| variant | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | grad relerr max |
|---|---:|---:|---:|---:|
| `DWM2-current-baseline` | `1.2918` | `1.9264` | `1.4971` | `9.00e-08` |
| `L6-lowrank-r2-fast` | `1.0786` | `2.0481` | `1.2693` | `6.36e-08` |
| `G2-grouped-g16-poly1` | `1.0460` | `9.8758` | `7.9475` | `2.22e-08` |
| `GR3-g16-vectorized-forward-backward` | `1.1749` | `1.9944` | `1.5208` | `1.89e-08` |
| `VG2-memory-preserving-vectorized-g16` | `1.1270` | `2.0149` | `1.5526` | `9.67e-08` |
| `VG3-tiled-vectorized-g16-tile4` | `1.0673` | `3.7634` | `4.2625` | `9.67e-08` |

### 5.2 runtime blocker taxonomy

`p1_component_kernel_audit.csv` 有 `1980` 行。code-loop / phase component diagnostics 可用；真实低层 kernel/allocation counter 仍不可用，因此 failure table 中保留：

```text
F9_runtime_counter_unavailable = 1980
```

P1 blocker 统计：

| primary_runtime_blocker | rows |
|---|---:|
| `vectorized_group_materialization` | 594 |
| `unknown` | 396 |
| `lowrank_factor_materialization` | 396 |
| `group_loop_fragmentation` | 396 |
| `chunk_loop_fragmentation` | 198 |

P1 结论：

- Grouped loop 的 runtime blocker 已能通过 code path 诊断为 `group_loop_fragmentation`。
- Vectorized/no-xg/tiled grouped 的 blocker 主要归为 `vectorized_group_materialization`。
- 低层 kernel/allocation counter 仍不可用，不能解读成 Nsight 级闭环；只能说 code-loop/phase-field attribution 已能支持 route 判断。

## 6. P2 Memory-Preserving Grouped Fusion

P2 是本轮主实验，目标是把 G2 的 memory 优势和 GR3 的 runtime 优势合并。

Measured packages：

| package | memory ratio mean | step ratio mean | backward ratio mean | residual pass | near pass |
|---|---:|---:|---:|---:|---:|
| `VG0-G2-grouped-g16-loop-baseline` | `1.0460` | `9.9129` | `7.9535` | 18/18 | 0 |
| `VG1-GR3-vectorized-g16-baseline` | `1.1749` | `2.0074` | `1.5265` | 18/18 | 0 |
| `VG2-memory-preserving-vectorized-g16` | `1.1270` | `2.0312` | `1.5615` | 18/18 | 0 |
| `VG3-tiled-vectorized-g16-tile4` | `1.0673` | `3.7881` | `4.2786` | 18/18 | 0 |
| `VG7-batched-group-gemm-no-materialize` | `1.1270` | `2.0307` | `1.5579` | 18/18 | 0 |

未实现：

```text
VG4-fused-grouped-forward-g16
VG5-fused-grouped-backward-g16
VG6-fused-grouped-forward-backward-g16
VG8-single-kernel-grouped-mix-g16
VG9-grouped-g16-lowrank-crossgroup-light
```

P2 判断：

- `VG2` 真实减少了 vectorized grouped 的 memory 回退：从 GR3 的 `1.1749` 降到 `1.1270`。
- 但 `VG2` 仍高于 memory near-pass `1.05`，step 仍约 `2.03`。
- `VG3` 更保 memory，达到 `1.0673`，但 step 变差到 `3.79`。
- P2 没有 candidate 同时满足 `memory <= 1.05` 且 `step <= 1.50`。

## 7. P3 Custom / Batched Grouped Kernel Diagnostic

Measured packages：

| package | backend | memory ratio mean | step ratio mean | backward ratio mean | residual pass | near pass |
|---|---|---:|---:|---:|---:|---:|
| `GK0-grouped-loop-baseline` | torch loop | `1.0460` | `9.2461` | `7.3350` | 18/18 | 0 |
| `GK1-batched-group-gemm` | torch batched | `1.1749` | `1.8879` | `1.4223` | 18/18 | 0 |
| `GK7-grouped-g8-custom` | no-xg einsum | `1.1315` | `1.9071` | `1.4498` | 18/18 | 0 |
| `GK8-grouped-g16-custom` | no-xg einsum | `1.1270` | `1.9057` | `1.4522` | 18/18 | 0 |

未实现：

```text
GK2-triton-grouped-forward
GK3-triton-grouped-backward
GK4-triton-grouped-forward-backward
GK5-cuda-grouped-forward-backward
GK6-blockdiag-grouped-mix
```

P3 判断：

- Batched/grouped torch paths 真实修复了 runtime，大约从 `9.25` 降到 `1.89-1.91`。
- 但 memory 仍回退到 `1.127-1.175`，没有保住 G2 的 `1.046`。
- 因为 Triton/CUDA grouped kernel 未实现，本轮不能声称 custom kernel 路线失败，只能说 torch/no-xg diagnostic 仍是 memory/time tradeoff。

## 8. P4 Low-Rank r2 Runtime Closure

Measured packages：

| package | memory ratio mean | step ratio mean | backward ratio mean | residual pass | near pass |
|---|---:|---:|---:|---:|---:|
| `LR0-L6-lowrank-r2-current` | `1.0786` | `1.9992` | `1.2324` | 18/18 | 0 |
| `LR3-r2-fused-forward-backward-v2` | `1.0786` | `1.9992` | `1.2333` | 18/18 | 0 |
| `LR4-r2-no-intermediate-v2` | `1.0786` | `1.9963` | `1.2315` | 18/18 | 0 |
| `LR6-r2-vectorized-lowrank-v2` | `1.0786` | `1.9981` | `1.2331` | 18/18 | 0 |

未实现：

```text
LR1-r2-fused-forward-v2
LR2-r2-fused-backward-v2
LR5-r2-onebuffer-v2
LR7-r2-triton-lowrank-forward
LR8-r2-triton-lowrank-backward
LR9-r2-full-fused-lowrank
LR10-r2-lowrank+tiny-grouped-correction
```

P4 判断：

- Low-rank r2 仍是最平衡的 fallback，但 memory ratio `1.0786` 没有到 `1.05`。
- Step ratio 约 `2.0`，没有达到 `1.50`。
- 现有 v2 variants 没有带来真实 runtime closure。

## 9. P5 Piecewise Residual vs Mixing

Measured packages：

| package | memory ratio mean | step ratio mean | backward ratio mean | structural/stability | residual pass | near pass |
|---|---:|---:|---:|---:|---:|---:|
| `PW0-piecewise2-forwardOnly-full-current` | `1.3277` | `3.7257` | `3.7602` | pass | 18/18 | 0 |
| `PW4-piecewise2-streamingGrad-lowrankMix-r2` | `1.4862` | `4.1024` | `3.9951` | pass | 18/18 | 0 |
| `PW5-piecewise2-streamingGrad-groupedMix-vectorized` | `1.5400` | `4.0090` | `4.2044` | pass | 18/18 | 0 |
| `PW7-piecewise2-no-knot-materialization-v2` | `1.4862` | `4.1006` | `3.9918` | pass | 18/18 | 0 |

未实现：

```text
PW1-piecewise2-residualOnly-forward
PW2-piecewise2-residualOnly-forward-backward
PW3-piecewise2-streamingGrad-noMix
PW6-piecewise2-streamingGrad-custom-knot
```

P5 判断：

- Piecewise full variants 继续通过 structural/stability/gradient/residual gates。
- 但 memory/time 离 near-pass 很远，最佳 memory 仍为 `1.3277`，最佳 step 仍约 `3.73`。
- residual-only/noMix 未实现，因此不能判断 piecewise residual 本体是否可独立过关。

## 10. P6 Selection / P7-P10 Gate

P6 最终选择：

| field | value |
|---|---:|
| best candidate | `GK0-grouped-loop-baseline` |
| best family | `grouped` |
| best memory ratio | `1.0459591815` |
| best step ratio | `9.2461018289` |
| best backward ratio | `7.3350349230` |
| best forward ratio | `9.7101870825` |
| memory improvement vs current | `19.03%` |
| step improvement vs current | `-395.56%` |
| survivor type | `S3` |
| open one-step probe | 0 |
| open task re-entry | 0 |

Gate 状态：

| stage | status | reason |
|---|---|---|
| P7 one-step probe | `not_run` | No S0/S1/S2 reset-v7 survivor |
| P8 task re-entry | `not_run` | P6 did not produce official S0/S1 task-open survivor or P7 did not pass |
| P8 task trace | `not_run` | P8 task is gated |
| P9 optimizer exploration | `not_run` | P6/P7 did not open official task |
| P10 functional correction smoke | `not_run` | P10 is gated behind P8/P9 |

这部分不能解读成 task/optimizer/functional 失败。准确说法是 v6.14 没有 S0/S1/S2 survivor，所以按计划没有进入训练。W&B 中因此没有每 20 step task loss 曲线，只有真实 profiler/summary/route 指标。

## 11. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F9_runtime_counter_unavailable` | 1980 | P1 仍缺真实低层 kernel/allocation counter |
| `F0_not_implemented_or_gated` | 21 | fused/Triton/CUDA/onebuffer/residual-only 候选未实现，或 P7-P10 被 gate |
| `F1_memory_fail` | 15 | measured reset-v7 candidates memory ratio 未过 near-pass |
| `F6_grouped_runtime_fail` | 9 | grouped branch step/time 未过 near-pass |
| `F7_lowrank_runtime_fail` | 4 | low-rank branch step/time 未过 near-pass |
| `F2_step_time_fail` | 4 | piecewise/full measured rows step ratio 超过 near-pass |
| `F10_task_gated` | 4 | P7-P10 正确 gate |

没有出现：

- fake data failure
- proxy row failure
- gradient correctness failure
- artifact missing failure

## 12. Route Decision

`route_decision.json`：

```json
{
  "route": "R4-GroupedMemoryRuntimeTradeoff",
  "best_candidate": "GK0-grouped-loop-baseline",
  "best_family": "grouped",
  "best_memory_ratio": 1.0459591815484715,
  "best_step_ratio": 9.246101828857643,
  "best_backward_ratio": 7.335034923020024,
  "best_forward_ratio": 9.710187082483243,
  "memory_improvement_vs_current": 0.19030391001713978,
  "step_improvement_vs_current": -3.955569637076666,
  "survivor_type": "S3",
  "residual_effect_pass": 1,
  "grad_pass": 1,
  "runtime_attribution_pass": 1,
  "open_one_step_probe": false,
  "open_task_reentry": false,
  "open_diagnostic_task": false,
  "open_optimizer_exploration": false,
  "open_functional_correction": false,
  "primary_blocker": "runtime_too_slow",
  "stop_dwm2_patching": 1,
  "reset_v7_measured": 1,
  "reset_v7_pass": 0,
  "next_required_implementation": "fused_grouped_kernel_or_stop_grouped_branch",
  "no_fake": true,
  "no_proxy": true
}
```

## 13. 结论

本轮 v6.14 支持以下真实结论：

1. v6.14 与 v6.13 baseline 可比，DWM2 current memory ratio mean 仍为 `1.2918`，DWM2 patch 继续冻结。
2. Grouped loop 继续给出最强 memory signal：`1.0460`，但 step ratio 仍在 `9.25-9.91`，只能是 S3 memory-only diagnostic。
3. `VG2-memory-preserving-vectorized-g16` 真实减少了 vectorized grouped 的 memory 回退，但仍为 `1.1270`，没有接近 `1.05`。
4. `VG3-tiled-vectorized-g16-tile4` 进一步保 memory 到 `1.0673`，但 runtime 变差到 `3.79`。
5. Batched/no-xg grouped paths 能把 runtime 修到约 `1.89-2.03`，但 memory 回退到 `1.127-1.175`。
6. Low-rank r2 仍较平衡，但 `memory=1.0786`、`step≈2.0`，没有 S2。
7. Piecewise local 继续 structural/stability/gradient/residual pass，但 memory/time 明显失败。
8. 低层 kernel/allocation counter 仍不可用，P1 只能作为 code-loop/phase-field attribution，不是 Nsight 级闭环。
9. P7-P10 正确 gate，没有 task、optimizer、functional 结论，也没有每 20 step task loss trace。
10. 最终 route 是 `R4-GroupedMemoryRuntimeTradeoff`。

最终一句话：

> v6.14 没有得到 ResetV7 S0/S1/S2 survivor。Grouped branch 的 memory 和 runtime 优势仍然不能合并：loop path 保 memory 但太慢，vectorized/no-xg path 修 runtime 但 memory 回退。下一步只有两条合理路：实现真正 fused grouped kernel，或停止 grouped branch 作为主线。

## 14. 下一步建议

1. 若继续 grouped，必须实现 fused grouped forward/backward kernel，目标同时保持 memory `<=1.05` 和 step `<=1.50`。
2. 不要继续 Python group loop；`GK0/VG0` 只能作为 memory diagnostic。
3. no-xg/tiled grouped 已证明 torch-level折中不足，除非进入 Triton/CUDA，否则很难合并 G2/GR3 优势。
4. Low-rank r2 可保留为 fallback，但必须把 memory 从 `1.0786` 降到 `<=1.05`，并把 step 从 `~2.0` 降到 `<=1.50`。
5. Piecewise local 需要先实现 residual-only/noMix 分解，否则不能判断是否值得继续。
6. P8 task、P9 optimizer、P10 functional correction 继续关闭，直到出现真实 S0/S1/S2 survivor。
