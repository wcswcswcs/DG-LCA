# DG-KAN v6.13 ResetV6 RuntimeRepair PiecewiseLocal 结果复盘

> 本复盘只记录 `results/real_rerun_20260505/v613_real_all_20260505T070847Z` 下的 final real-only run。运行启动于 2026-05-05 07:08 UTC，完成于 2026-05-05 07:19 UTC。所有结论来自该目录落盘的 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v613_real.py \
  --packages V6_13_ALL \
  --out-dir results/real_rerun_20260505/v613_real_all_20260505T070847Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v613-real-20260505 \
  --wandb-name-prefix v613-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260505/v613_real_all_20260505T070847Z`
- 本地主日志：`results/real_rerun_20260505/v613_real_all_20260505T070847Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/g9vfllc3>
- W&B run name：`v613-real-v613_real_all_20260505T070847Z`

### 1.3 不用于结论的检查

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `/tmp/v613_smoke` | 极小规模 smoke，检查 runner/no-fake/vectorized grouped/piecewise/P6 gate/CSV shape | 0 |

### 1.4 no-fake / no-proxy 约束

- 新入口：`experiments/run_gafu_v613_real.py`。
- 所有 dataset loader 调用均使用 `allow_fake_data=False`。
- `experiments/dgkan_core.py` 中 fake data 路径仍保持 hard-fail。
- P0-P10 的 `fake_data_used` 总和均为 `0`。
- P0-P10 的 `proxy_row_used/proxy_rows_used` 总和均为 `0`。
- DWM2 继续作为 frozen baseline，没有新增 DWM2 patch。
- 未实现的 fused-forward/backward-only、one-buffer、Triton/custom kernel、adaptive chunk 等候选写为 `not_implemented`；P7-P10 因 gate 写为 `not_run`。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_13_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler/kernel 使用 `seed=0`；无 task multi-seed confirm |
| full warmup / measure | 每个 full-step measured row `50/200` update steps |
| micro warmup / measure | `50/200` calls |
| run duration | `650.31` sec |
| source commit | `f76ff27ea03c6fbfc01cadf24ea281c32a72b18f` |
| Triton | available |
| memory history | available |
| Nsight | `ncu_available=1`，`nsys_available=0` |
| W&B logging | 记录 P0-P6/P11/failure/route summary；P8 task 被 gate，未产生每 20 step task loss trace |

本轮真实新增实现：

| item | 实现说明 |
|---|---|
| `ResetV6-vectorized-grouped-g16` | 使用 batched grouped matmul/einsum 替代 Python group loop |
| `ResetV6-vectorized-grouped-g8/g4` | vectorized grouped scaling variants |
| `ResetV6-piecewise2-forwardOnly-diagnostic` | active-bin piecewise residual，固定 slope diagnostic，无 dense `[B,d,K]` basis |
| `ResetV6-piecewise2-streamingGrad` | active-bin piecewise residual，streaming slope grad |
| `ResetV6-piecewise4-streamingGrad` | 4-bin active-bin piecewise residual |
| `ResetV6-piecewise2-groupedMix` | piecewise residual + vectorized grouped mixing |
| `ResetV6-piecewise2-lowrankMix-r2` | piecewise residual + low-rank r2 mixing |

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 10 |
| `p0_reproduction_check.csv` | 真实生成 | 1 |
| `p1_reset_runtime_attribution.csv` | 真实生成 | 162 |
| `p1_component_kernel_audit.csv` | 真实生成 | 1584 |
| `p2_fused_lowrank_repair_v2.csv` | 真实生成 | 12 |
| `p2_fused_lowrank_repair_v2_detail.csv` | 真实生成 | 234 |
| `p3_vectorized_grouped_repair.csv` | 真实生成 | 10 |
| `p3_vectorized_grouped_repair_detail.csv` | 真实生成 | 198 |
| `p4_chunked_lowrank_pareto.csv` | 真实生成 | 8 |
| `p4_chunked_lowrank_pareto_detail.csv` | 真实生成 | 162 |
| `p5_piecewise_local_reset.csv` | 真实生成 | 6 |
| `p5_piecewise_local_reset_detail.csv` | 真实生成 | 126 |
| `p6_reset_v6_selection.csv` | 真实生成 | 1 |
| `p7_one_step_probe.csv` | `not_run` | 1 |
| `p8_task_reentry.csv` | `not_run` | 1 |
| `p8_task_trace.csv` | `not_run` | 1 |
| `p9_optimizer_exploration.csv` | `not_run` | 1 |
| `p10_functional_correction_smoke.csv` | `not_run` | 1 |
| `failure_table.csv` | 真实生成 | 1646 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |

关键 hash（复盘时重新计算）：

| 文件 | SHA256 |
|---|---|
| `run.log` | `82e6ac36ee1a035361f548ebb30a2471512b0d450ecea1c4719d7c5c4b2b65f3` |
| `p1_component_kernel_audit.csv` | `252333ce9d27f26d63806e4fa36e7ea69451c71e918d604307bcd8f0b0eb4932` |
| `p2_fused_lowrank_repair_v2.csv` | `ec6f21390e67b37e0e654d0c58cff2f24dfa29b62b030f7f529f4af743ce92cc` |
| `p3_vectorized_grouped_repair.csv` | `2dcbcb94488e910ae9c77cd415829731dbf37e1ee1b70f7e21701115792e3f38` |
| `p4_chunked_lowrank_pareto.csv` | `7007e5b68fb95dcb1b54c05cf84201ead53dffba9bdd60511d78b8a072720be4` |
| `p5_piecewise_local_reset.csv` | `7e7509f98388a365392689d48c39f7732e4e481e34b98d32f1129bc681ca64e9` |
| `p8_task_reentry.csv` | `5a9e3d67d3c5559ebcc0b6a461394267a874b613d314070f46f118057a3a177e` |
| `p8_task_trace.csv` | `239ea497a86ae16db1f677ff5259b0faf6fe971f9b0fc22f9e7d1831db3f0417` |
| `failure_table.csv` | `df4cc8924e83ff7eacc7fdba6a430d1bf361bdb45a99bb98c4d1074125f35598` |
| `route_decision.json` | `0ab97f0aca93b233c3603aeee6f6ef3f6dbf8f1cfeff11b5e1e4d08aedd3eed9` |

## 4. P0 Contract / v6.12 Reproduction

P0 contract：

| variant | status | nonKAN | grad pass | fake/proxy |
|---|---|---:|---:|---:|
| `MLP-autograd-reference` | measured | 55050 | - | 0/0 |
| `MLP-manual-linear-reference` | measured | 0 | 1 | 0/0 |
| `DWM2-current-baseline` | measured frozen baseline | 0 | 1 | 0/0 |
| `L6-lowrank-r2-fast-v612` | measured lineage baseline | 0 | 1 | 0/0 |
| `L4-lowrank-no-intermediate-v612` | measured lineage baseline | 0 | 1 | 0/0 |
| `G2-grouped-g16-poly1-v612` | measured lineage baseline | 0 | 1 | 0/0 |
| `G1-grouped-g8-poly1-v612` | measured lineage baseline | 0 | 1 | 0/0 |
| `ResetV6-fused-lowrank-r2` | measured | 0 | 1 | 0/0 |
| `ResetV6-vectorized-grouped-g16` | measured | 0 | 1 | 0/0 |
| `ResetV6-piecewise2-streamingGrad` | measured | 0 | 1 | 0/0 |

v6.12 reproduction：

| metric | v6.12 ref | v6.13 reproduction | delta | pass |
|---|---:|---:|---:|---:|
| DWM2 current memory ratio mean | `1.2917923089` | `1.2917923089` | `0.0` | 1 |
| L6 memory ratio mean | `1.0785729221` | `1.0785729221` | `0.0` | 1 |
| L6 step ratio mean | `2.0998166478` | `2.0141154785` | `-0.0857011694` | 1 |

P0 结论：v6.13 与 v6.12 memory baseline 可比；真实实现项梯度正确；DWM2 patch 主线继续冻结。

## 5. P1 Runtime Attribution

P1 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x (MLP + manual-linear + DWM2 + L6/L4/G2/G1/C5/GR3) = 162 rows
```

### 5.1 full-step summary

| variant | memory ratio vs MLP | step ratio vs MLP | grad relerr max |
|---|---:|---:|---:|
| `DWM2-current-baseline` | `1.1441 / 1.2918 / 1.4866` | `1.5618 / 1.8658 / 2.2250` | `9.00e-08` |
| `L6-lowrank-r2-fast` | `1.0077 / 1.0786 / 1.1821` | `1.6638 / 1.9968 / 2.5431` | `6.36e-08` |
| `L4-lowrank-no-intermediate` | `1.0116 / 1.0822 / 1.1855` | `1.6680 / 1.9830 / 2.3893` | `7.49e-08` |
| `G2-grouped-g16-poly1` | `0.9971 / 1.0460 / 1.1203` | `7.0290 / 9.5095 / 12.4888` | `2.22e-08` |
| `G1-grouped-g8-poly1` | `1.0031 / 1.0562 / 1.1379` | `4.1049 / 5.4029 / 6.9857` | `2.08e-08` |
| `GR3-g16-vectorized-forward-backward` | `1.0518 / 1.1749 / 1.3439` | `1.6172 / 1.9291 / 2.3098` | `1.89e-08` |
| `C5-lowrank-r4-chunk64` | `1.0116 / 1.0822 / 1.1855` | `1.7685 / 2.1393 / 2.5967` | `7.49e-08` |

### 5.2 low-level counter status

`p1_component_kernel_audit.csv` 有 `1584` 行。component phase fields 和 code-loop diagnostics 可用，但真实低层 kernel/allocation counters 仍不可用，因此 failure table 中保留：

```text
F9_runtime_counter_unavailable = 1584
```

P1 结论：

- Low-rank 的 memory 已接近，但 step 仍约 `2.0`。
- Grouped loop 的 memory 最好，但 runtime 极慢。
- Vectorized grouped 把 step 从 grouped loop 的 `9.51` 降到 `1.93` 左右，但 memory 回退到 `1.1749`。
- 低层 kernel/allocation counter 仍没有真正闭合，不能声称 Nsight 级 runtime attribution 完成。

## 6. P2 Fused Low-Rank Runtime Repair v2

Measured packages：

| package | memory ratio mean | step ratio mean | backward ratio mean | memory improvement vs DWM2 | step improvement vs DWM2 | residual pass | near pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LR0-L6-lowrank-r2-fast-current` | `1.0786` | `2.0141` | `1.2396` | `16.51%` | `-0.96%` | 18/18 | 0 |
| `LR3-r2-fused-forward-backward` | `1.0786` | `2.0196` | `1.2438` | `16.51%` | `-1.24%` | 18/18 | 0 |
| `LR4-r2-no-materialized-lowrank-intermediate` | `1.0786` | `2.0170` | `1.2416` | `16.51%` | `-1.11%` | 18/18 | 0 |
| `LR6-r2-vectorized-lowrank` | `1.0786` | `2.0092` | `1.2363` | `16.51%` | `-0.72%` | 18/18 | 0 |

未实现：

```text
LR1-r2-fused-forward
LR2-r2-fused-backward
LR5-r2-onebuffer-lowrank
LR7-r2-triton-lowrank-forward
LR8-r2-triton-lowrank-backward
LR9-r2-full-fused-lowrank
LR10-r4-full-fused-lowrank
```

P2 结论：

- `LR6-r2-vectorized-lowrank` 是 low-rank 中 step 最好的 measured variant，step ratio mean `2.0092`。
- 但它没有比 L6 达到 `>=25%` runtime repair，也没有 memory near-pass。
- Low-rank branch 仍是 balanced but too slow / memory not near-pass。

## 7. P3 Vectorized Grouped Mixing Repair

Measured packages：

| package | memory ratio mean | step ratio mean | backward ratio mean | memory improvement vs DWM2 | step improvement vs G2 | residual pass | near pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `GR0-G2-grouped-g16-current` | `1.0460` | `9.7583` | `7.7978` | `19.03%` | `3.62%` | 18/18 | 0 |
| `GR3-g16-vectorized-forward-backward` | `1.1749` | `1.9817` | `1.4968` | `9.05%` | `80.43%` | 18/18 | 0 |
| `GR4-g16-batched-group-gemm` | `1.1749` | `1.9806` | `1.4953` | `9.05%` | `80.44%` | 18/18 | 0 |
| `GR6-g8-vectorized-forward-backward` | `1.1793` | `1.9810` | `1.4955` | `8.70%` | `80.43%` | 18/18 | 0 |
| `GR7-g4-vectorized-forward-backward` | `1.1885` | `1.9869` | `1.4993` | `8.00%` | `80.38%` | 18/18 | 0 |

未实现：

```text
GR1-g16-vectorized-forward
GR2-g16-vectorized-backward
GR5-g16-single-kernel-grouped-mix
GR8-g16-triton-grouped-mix
GR9-g16-triton-grouped-backward
```

P3 结论：

- H2 部分成立：vectorized grouped 真实消除了大部分 runtime overhead，step 从 `9.7583` 降到约 `1.98`。
- 但 H2 没有转化为 survivor：memory 从 `1.0460` 回退到 `1.1749`，高于 near-pass gate。
- `GR0` 仍是 memory 最好但 runtime fail；`GR3/GR4` 是 runtime repair 成功但 memory fail。

## 8. P4 Chunked Low-Rank Pareto

Measured packages：

| package | memory ratio mean | step ratio mean | backward ratio mean | memory improvement vs DWM2 | residual pass | near pass |
|---|---:|---:|---:|---:|---:|---:|
| `CH0-L6-r2-current` | `1.0786` | `2.0623` | `1.3076` | `16.51%` | 18/18 | 0 |
| `CH1-r2-chunk32` | `1.0786` | `2.3570` | `1.7264` | `16.51%` | 18/18 | 0 |
| `CH2-r2-chunk64` | `1.0786` | `2.2303` | `1.5237` | `16.51%` | 18/18 | 0 |
| `CH3-r2-chunk128` | `1.0786` | `2.2289` | `1.5235` | `16.51%` | 18/18 | 0 |
| `CH4-r4-chunk64` | `1.0822` | `2.2245` | `1.5184` | `16.23%` | 18/18 | 0 |
| `CH5-r4-chunk128` | `1.0822` | `2.2195` | `1.5148` | `16.23%` | 18/18 | 0 |

未实现：

```text
CH6-r2-adaptive-chunk
CH7-r4-adaptive-chunk
```

P4 结论：

- chunked low-rank 没有找到 Pareto survivor。
- chunking 没有降低 memory 到 `<=1.05`，还使 step time 变差。
- H3 不成立：本轮没有 `memory <= 1.05` 且 `step <= 1.50` 的 chunked point。

## 9. P5 Piecewise Local Bounded Reset

Measured packages：

| package | memory ratio mean | step ratio mean | residual/base | structural pass | stability pass | residual pass | near pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PW0-piecewise2-forwardOnly-diagnostic` | `1.3277` | `3.7215` | `0.0200` | 1 | 1 | 18/18 | 0 |
| `PW1-piecewise2-streamingGrad` | `1.4862` | `4.0981` | `0.0200` | 1 | 1 | 18/18 | 0 |
| `PW2-piecewise4-streamingGrad` | `1.4888` | `4.0869` | `0.0200` | 1 | 1 | 18/18 | 0 |
| `PW3-piecewise2-groupedMix` | `1.5400` | `4.0157` | `0.0200` | 1 | 1 | 18/18 | 0 |
| `PW4-piecewise2-lowrankMix-r2` | `1.4862` | `4.0863` | `0.0200` | 1 | 1 | 18/18 | 0 |
| `PW5-piecewise2-no-knot-materialization` | `1.4862` | `4.0912` | `0.0200` | 1 | 1 | 18/18 | 0 |

Stability details：

| package | active bins/sample | dead bin fraction | out-of-grid fraction |
|---|---:|---:|---:|
| `PW0` | `1.0` | `0.0` | `0.0023` |
| `PW1` | `1.0` | `0.0` | `0.0032` |
| `PW2` | `1.0` | `0.2604` | `0.0045` |
| `PW3` | `1.0` | `0.0` | `0.0009` |
| `PW4` | `1.0` | `0.0` | `0.0023` |
| `PW5` | `1.0` | `0.0` | `0.0031` |

P5 结论：

- v6.13 真实补上了 piecewise local measured branch；不再是 v6.12 的空缺。
- Piecewise structural/stability/gradient/residual effect 都通过。
- 但 piecewise memory/time 明显失败：最佳 memory 仍为 `1.3277`，最佳 step 仍约 `3.72`，没有 near-pass。
- H4 的“可实现且稳定”成立，但“成为 memory/time survivor”不成立。

## 10. P6 Selection / P7-P10 Gate

P6 最终选择：

| field | value |
|---|---:|
| best candidate | `GR0-G2-grouped-g16-current` |
| best family | `grouped` |
| best memory ratio | `1.0459591815` |
| best step ratio | `9.7582509943` |
| best backward ratio | `7.7978052738` |
| best forward ratio | `10.1395220715` |
| memory improvement vs current | `19.03%` |
| step improvement vs current | `-389.17%` |
| survivor type | `S3` |
| primary blocker | `runtime_too_slow` |
| open one-step probe | 0 |
| open task re-entry | 0 |

Gate 状态：

| stage | status | reason |
|---|---|---|
| P7 one-step probe | `not_run` | No S0/S1/S2 reset-v6 survivor |
| P8 task re-entry | `not_run` | P6 did not produce official S0/S1 task-open survivor or P7 did not pass |
| P8 task trace | `not_run` | P8 task is gated |
| P9 optimizer exploration | `not_run` | P6/P7 did not open official task |
| P10 functional correction smoke | `not_run` | P10 is gated behind P8/P9 |

这部分不能解读成 task/optimizer/functional 失败。准确说法是 v6.13 没有 S0/S1/S2 survivor，所以按计划没有进入训练。W&B 中因此没有每 20 step task loss 曲线，只有真实 profiler/summary/route 指标。

## 11. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F9_runtime_counter_unavailable` | 1584 | P1 仍缺真实低层 kernel/allocation counter |
| `F1_memory_fail` | 21 | measured candidates memory ratio 未过 near-pass |
| `F0_not_implemented_or_gated` | 14 | fused/one-buffer/Triton/adaptive candidates 未实现 |
| `F2_step_time_fail` | 6 | measured rows step ratio 超过 near-pass |
| `F7_chunked_time_fail` | 6 | chunked low-rank step 未过 |
| `F5_lowrank_runtime_fail` | 5 | low-rank runtime repair 未达标 |
| `F6_grouped_runtime_fail` | 5 | grouped branch runtime 仍不达标 |
| `F10_task_gated` | 4 | P7-P10 正确 gate |
| `F4_residual_effect_fail` | 1 | DWM2 baseline residual effect 不达标 |

没有出现：

- fake data failure
- proxy row failure
- gradient correctness failure
- artifact missing failure

## 12. Route Decision

`route_decision.json`：

```json
{
  "route": "R4-GroupedRuntimeFail",
  "best_candidate": "GR0-G2-grouped-g16-current",
  "best_family": "grouped",
  "best_memory_ratio": 1.0459591815484715,
  "best_step_ratio": 9.758250994271396,
  "best_backward_ratio": 7.79780527379197,
  "best_forward_ratio": 10.139522071513348,
  "memory_improvement_vs_current": 0.19030391001713978,
  "step_improvement_vs_current": -3.8916596078064885,
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
  "reset_v6_measured": 1,
  "reset_v6_pass": 0,
  "next_required_implementation": "fused_grouped_kernel_or_stop_grouped_branch",
  "no_fake": true,
  "no_proxy": true
}
```

## 13. 结论

本轮 v6.13 支持以下真实结论：

1. v6.13 与 v6.12 memory baseline 可比，DWM2 current memory ratio mean 仍为 `1.2918`。
2. DWM2 patch 主线继续冻结。
3. Low-rank r2 branch 仍是较平衡方向，但 memory ratio mean `1.0786`、step ratio mean 约 `2.01`，没有 S2。
4. Vectorized grouped branch 真实修复了 runtime 大头：`GR3/GR4` step 从 `9.76` 降到约 `1.98`，提速约 `80%`。
5. 但 vectorized grouped memory 回退到 `1.1749`，因此没有 survivor。
6. Grouped loop branch 仍给出最强 memory signal：`GR0/G2` memory ratio mean `1.0460`，但 step ratio mean `9.7583`，属于 memory-only diagnostic。
7. Piecewise local branch 已真实实现并通过 structural/stability/gradient/residual gates，但 memory/time 明显失败。
8. P7-P10 正确 gate，没有 task、optimizer、functional 结论，也没有每 20 step task loss trace。
9. 最终 route 是 `R4-GroupedRuntimeFail`。

最终一句话：

> v6.13 证明了 ResetV6 的 grouped runtime 确实可以通过 vectorization 大幅修复，但当前 vectorized path 牺牲了 memory；piecewise local 也已真实实现但太重。没有任何 S0/S1/S2 survivor，不能打开 task。下一步应只在“保持 G2 memory 约 1.046 的同时 fused/vectorized runtime”的方向继续，否则 grouped branch 应停止作为主线。

## 14. 下一步建议

1. 若继续 grouped，必须实现 fused grouped kernel，让 memory 保持 `~1.046`，同时把 step 从 `9.76` 降到 `<=1.50`。
2. 不要继续 Python group loop；`GR0` 只能作为 memory diagnostic。
3. Low-rank branch 需要同时把 memory 从 `1.0786` 降到 `<=1.05`，并把 step 从 `~2.01` 降到 `<=1.50`，否则不应开 task。
4. Piecewise local 下一步要先重做 workspace model；当前 structural/stability 通过，但 memory/time 太差。
5. P7/P8/P9/P10 继续关闭，直到出现真实 S0/S1/S2 survivor。
