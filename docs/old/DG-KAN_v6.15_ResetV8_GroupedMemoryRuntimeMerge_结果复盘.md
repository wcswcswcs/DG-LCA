# DG-KAN v6.15 ResetV8 GroupedMemoryRuntimeMerge 结果复盘

> 本复盘只记录 `results/real_rerun_20260505/v615_real_all_20260505T084644Z` 下的 final real-only run。运行启动于 2026-05-05 08:46 UTC，完成于 2026-05-05 09:01 UTC。所有结论来自该目录落盘的 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v615_real.py \
  --packages V6_15_ALL \
  --out-dir results/real_rerun_20260505/v615_real_all_20260505T084644Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v615-real-20260505 \
  --wandb-name-prefix v615-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260505/v615_real_all_20260505T084644Z`
- 本地主日志：`results/real_rerun_20260505/v615_real_all_20260505T084644Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/zteprb2f>
- W&B run name：`v615-real-v615_real_all_20260505T084644Z`

### 1.3 不用于结论的检查

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `/tmp/v615_smoke` | 极小规模 smoke，检查 runner/no-fake/P2-P5 新分支/CSV shape | 0 |

smoke 中曾暴露 `PW1-piecewise2-residualOnly-forward` 没有 trainable gradient path，旧 gradient check 无法对空 autograd grads 做合法比较。因此 final runner 中将该 forward-only 诊断明确写为 `not_implemented_forward_only_has_no_trainable_gradient_path`，没有伪造 measured ratio；真实 residual-only 由 `PW2/PW3/PW8` 测量。

### 1.4 no-fake / no-proxy 约束

- 新入口：`experiments/run_gafu_v615_real.py`。
- 所有 dataset loader 调用均使用 `allow_fake_data=False`。
- `experiments/dgkan_core.py` 中 fake data 路径仍保持 hard-fail。
- P0-P10 的 `fake_data_used` 总和均为 `0`。
- P0-P10 的 `proxy_row_used/proxy_rows_used` 总和均为 `0`。
- DWM2 继续作为 frozen baseline，没有新增 DWM2 patch。
- 未实现的 two-level/adaptive grouped、fused/Triton/CUDA grouped、low-rank one-buffer/full-fused、piecewise custom-knot 等候选写为 `not_implemented` 或明确的 not-implemented 状态。
- P7-P10 因 gate 写为 `not_run`，没有包装成 task/optimizer/functional 失败或成功。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_15_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler/kernel 使用 `seed=0`；无 task multi-seed confirm |
| full warmup / measure | 每个 full-step measured row `50/200` update steps |
| run duration | `912.83` sec |
| source commit | `f76ff27ea03c6fbfc01cadf24ea281c32a72b18f` |
| Triton | available |
| memory history | available |
| Nsight | `ncu_available=1`，`nsys_available=0` |
| W&B logging | 记录 P0-P6/P11/failure/route summary；P8 task 被 gate，未产生每 20 step task loss trace |

本轮真实新增/修正实现：

| item | 实现说明 |
|---|---|
| `GS4/GS5/GS6/GS7/GS8 streamed grouped tile sweep` | 用真实 tiled grouped path 测 tile1/2/4/8/16 的 memory/runtime tradeoff |
| `GS11/GS12/GS13 no-xg/tiled no-xg diagnostics` | 复用 no-materialize grouped einsum 和 tiled path 做 memory preserving 诊断 |
| `LR9-r1-fast-diagnostic` | rank=1 low-rank diagnostic，真实测量但名字含 diagnostic，不参与 P6 gate selection |
| `PW2/PW3/PW8 residual-only/noMix` | active-bin piecewise residual-only/noMix 路径，真实 forward/backward/update |
| P1 runtime attribution proxies | 记录 code-path proxy counts：Python loop、torch op、kernel launch proxy、allocation proxy、cuda event proxy |

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 11 |
| `p0_reproduction_check.csv` | 真实生成 | 1 |
| `p1_runtime_attribution.csv` | 真实生成 | 198 |
| `p1_component_kernel_audit.csv` | 真实生成 | 1980 |
| `p2_grouped_tile_stream_sweep.csv` | 真实生成 | 14 |
| `p2_grouped_tile_stream_sweep_detail.csv` | 真实生成 | 270 |
| `p3_fused_grouped_custom_kernel.csv` | 真实生成 | 9 |
| `p3_fused_grouped_custom_kernel_detail.csv` | 真实生成 | 180 |
| `p4_lowrank_r2_s2_closure.csv` | 真实生成 | 12 |
| `p4_lowrank_r2_s2_closure_detail.csv` | 真实生成 | 234 |
| `p5_piecewise_residual_vs_mixing.csv` | 真实生成 | 9 |
| `p5_piecewise_residual_vs_mixing_detail.csv` | 真实生成 | 180 |
| `p6_reset_v8_selection.csv` | 真实生成 | 1 |
| `p7_one_step_probe.csv` | `not_run` | 1 |
| `p8_task_reentry.csv` | `not_run` | 1 |
| `p8_task_trace.csv` | `not_run` | 1 |
| `p9_optimizer_exploration.csv` | `not_run` | 1 |
| `p10_functional_correction_smoke.csv` | `not_run` | 1 |
| `failure_table.csv` | 真实生成 | 73 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |

关键 hash（复盘时重新计算）：

| 文件 | SHA256 |
|---|---|
| `run.log` | `7621960d91287f770f1833ba59f00bd4d8558bd3572b74c55dc22f214b6be30c` |
| `p1_component_kernel_audit.csv` | `58b931adc807bd8c4e91abedc483048d99e25486e52c6bc4b78f18743620746e` |
| `p2_grouped_tile_stream_sweep.csv` | `54516230a056d3d24e2b92c841681a1cddfcf8573cf4876c25c2ee8c3bebf357` |
| `p3_fused_grouped_custom_kernel.csv` | `170aabf95694145cef90b18a3a465f2512c757abfd36da1909ffe98454cd3a3c` |
| `p4_lowrank_r2_s2_closure.csv` | `66f848227200bf12abdfe44af7e28d48757c0ad5147931fb2349f627be1247f4` |
| `p5_piecewise_residual_vs_mixing.csv` | `7ddd1582def83b05bc7d1449aef6a5f3e24a4ffa7bb62365d0761c3b97b9b447` |
| `failure_table.csv` | `ac479f1cfe8ee8137d73f69e32ddb83aa72a9d1e0a234691bd500939dda46ada` |
| `route_decision.json` | `3c889bb880d7541f298ccf2b375217c27006c7a9e39a874967d9121df25de733` |

## 4. P0 Contract / v6.14 Reproduction

P0 contract 覆盖：

| variant | status | lineage | fake/proxy |
|---|---|---|---:|
| `MLP-autograd-reference` | measured | none | 0/0 |
| `MLP-manual-linear-reference` | measured | manual | 0/0 |
| `DWM2-current-baseline` | measured frozen baseline | DWM2 | 0/0 |
| `L6-lowrank-r2-fast-v614` | measured lineage baseline | v614 | 0/0 |
| `G2-grouped-g16-poly1-v614` | measured lineage baseline | v614 | 0/0 |
| `GR3-vectorized-g16-v614` | measured lineage baseline | v614 | 0/0 |
| `GK8-grouped-g16-custom-v614` | measured lineage baseline | v614 | 0/0 |
| `PW1-piecewise2-streamingGrad-v614` | measured lineage baseline | v614 | 0/0 |
| `ResetV8-memory-preserving-vectorized-g16` | measured | reset-v8 | 0/0 |
| `ResetV8-tiled-vectorized-g16-tile4` | measured | reset-v8 | 0/0 |
| `ResetV8-piecewise2-residualOnly-streamingGrad` | measured | reset-v8 | 0/0 |

v6.14 reproduction：

| metric | v6.14 ref | v6.15 reproduction | delta | pass |
|---|---:|---:|---:|---:|
| DWM2 current memory ratio mean | `1.2917923089` | `1.2917923089` | `0.0` | 1 |
| L6 memory ratio mean | `1.0785729221` | `1.0785729221` | `0.0` | 1 |
| L6 step ratio mean | `1.9991722991` | `2.0250928296` | `+0.0259205306` | 1 |
| G2 memory ratio mean | `1.0459591815` | `1.0459591815` | `0.0` | 1 |
| GR3 memory ratio mean | `1.1748654001` | `1.1748654001` | `0.0` | 1 |

P0 结论：v6.15 与 v6.14 memory baseline 可比；DWM2 patch 主线继续冻结；新增 ResetV8 grouped/piecewise code paths 真实可执行。

## 5. P1 Runtime Attribution Closure

P1 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x 11 variants = 198 rows
component audit = 1980 rows
```

### 5.1 full-step summary

| variant | memory ratio mean | step ratio mean | backward ratio mean |
|---|---:|---:|---:|
| `DWM2-current-baseline` | `1.2918` | `1.8908` | `1.4378` |
| `L6-lowrank-r2-fast` | `1.0786` | `2.0000` | `1.2131` |
| `G2-grouped-g16-poly1` | `1.0460` | `9.5803` | `7.5451` |
| `GR3-g16-vectorized-forward-backward` | `1.1749` | `1.9528` | `1.4568` |
| `VG2-memory-preserving-vectorized-g16` | `1.1270` | `1.9684` | `1.4813` |
| `VG3-tiled-vectorized-g16-tile4` | `1.0673` | `3.6758` | `4.0758` |

### 5.2 runtime proxy status

| item | value |
|---|---:|
| component rows | `1980` |
| component_status | `measured_with_code_path_proxy_counts` for all rows |
| metric availability | `phase_fields_code_path_proxy_counts_low_level_counter_unavailable` |
| runtime attribution pass count | `1584/1980` |

Blocker taxonomy：

| primary_runtime_blocker | rows |
|---|---:|
| `vectorized_group_materialization` | 594 |
| `unknown` | 396 |
| `lowrank_factor_materialization` | 396 |
| `group_loop_fragmentation` | 396 |
| `chunk_loop_fragmentation` | 198 |

P1 判断：

- v6.15 没有继续只写 `metric_unavailable`：Python loop、chunk loop、torch op proxy、kernel launch proxy、allocation proxy 和 CUDA event proxy 都真实落盘。
- 低层 Nsight/allocation counter 仍不可用，因此 P1 不是 Nsight 级闭环，但足以支持 code-path 级 route 判定。
- Grouped loop 的 blocker 仍是 loop fragmentation；vectorized/no-xg/tiled 的 blocker 仍是 materialization/workspace tradeoff。

## 6. P2 Grouped Tile/Stream Sweep

P2 覆盖：

```text
18 shapes x (MLP reference + 12 measured grouped variants + 2 not-implemented variants) = 270 detail rows
```

Measured grouped packages：

| package | memory ratio mean | step ratio mean | backward ratio mean | residual pass | near pass |
|---|---:|---:|---:|---:|---:|
| `GS4-streamed-grouped-tile1` | `1.0374` | `9.0878` | `12.7687` | 18/18 | 0 |
| `GS0-G2-grouped-g16-loop-baseline` | `1.0460` | `9.6785` | `7.7331` | 18/18 | 0 |
| `GS5-streamed-grouped-tile2` | `1.0493` | `5.7495` | `7.3270` | 18/18 | 0 |
| `GS3/GS6 tile4` | `1.0673` | `3.70-3.72` | `4.14-4.17` | 18/18 | 0 |
| `GS7/GS12 tile8` | `1.1031` | `2.68-2.69` | `2.56` | 18/18 | 0 |
| `GS2/GS11 no-xg` | `1.1270` | `1.98-1.99` | `1.50-1.51` | 18/18 | 0 |
| `GS1 vectorized` | `1.1749` | `1.9652` | `1.4796` | 18/18 | 0 |
| `GS8/GS13 tile16` | `1.1749` | `2.15` | `1.72` | 18/18 | 0 |

未实现：

```text
GS9-two-level-streamed-g16
GS10-adaptive-tile-g16
```

P2 判断：

- Tile1 给出本轮最佳 memory：`1.0374`，比 DWM2 current 改善约 `19.69%`。
- Tile2 仍在 memory near threshold 内：`1.0493`，但 step ratio 为 `5.7495`。
- tile 越大 runtime 越好，memory 越差；这是 v6.15 想验证的 tradeoff，结果真实成立。
- P2 没有任何 candidate 同时满足 `memory <= 1.05` 和 `step <= 1.50`。

## 7. P3 Fused / Custom Grouped Go-No-Go

Measured packages：

| package | backend | memory ratio mean | step ratio mean | backward ratio mean | residual pass | near pass |
|---|---|---:|---:|---:|---:|---:|
| `GK0-grouped-loop-baseline` | torch loop | `1.0460` | `9.6280` | `7.6413` | 18/18 | 0 |
| `GK1-batched-group-gemm` | torch batched | `1.1749` | `1.9679` | `1.4797` | 18/18 | 0 |
| `GK7-grouped-g8-custom` | no-xg einsum | `1.1315` | `1.9885` | `1.5100` | 18/18 | 0 |
| `GK8-grouped-g16-custom` | no-xg einsum | `1.1270` | `1.9856` | `1.5097` | 18/18 | 0 |

未实现：

```text
GK2-triton-grouped-forward
GK3-triton-grouped-backward
GK4-triton-grouped-forward-backward
GK5-cuda-grouped-forward-backward
GK6-blockdiag-grouped-mix
```

P3 判断：

- torch batched/no-xg paths 的 runtime 真实降到约 `1.97-1.99`，但 memory 回退到 `1.127-1.175`。
- loop path 保 memory，但 step 仍约 `9.63`。
- 因 Triton/CUDA grouped kernel 未实现，本轮不能声称 custom kernel 路线失败；只能说 torch/no-xg diagnostic 仍无法合并 memory/runtime 优势。

## 8. P4 Low-Rank r2 S2 Closure

Measured packages：

| package | memory ratio mean | step ratio mean | backward ratio mean | residual pass | near pass |
|---|---:|---:|---:|---:|---:|
| `LR0-L6-lowrank-r2-current` | `1.0786` | `2.0251` | `1.2489` | 18/18 | 0 |
| `LR3-r2-fused-forward-backward-v2` | `1.0786` | `2.0277` | `1.2491` | 18/18 | 0 |
| `LR4-r2-no-intermediate-v2` | `1.0786` | `2.0258` | `1.2480` | 18/18 | 0 |
| `LR6-r2-vectorized-lowrank-v2` | `1.0786` | `2.0246` | `1.2477` | 18/18 | 0 |
| `LR9-r1-fast-diagnostic` | `1.0772` | `1.9897` | `1.1758` | 18/18 | 0 |

未实现：

```text
LR1-r2-fused-forward-v2
LR2-r2-fused-backward-v2
LR5-r2-onebuffer-v2
LR7-r2-triton-lowrank-forward
LR8-r2-triton-lowrank-backward
LR10-r2-lowrank+tiny-grouped-correction
LR11-r2-full-fused-lowrank
```

P4 判断：

- Low-rank r2 仍是相对平衡 fallback，但没有进 S2：memory `1.0786 > 1.05`，step `~2.025 > 1.50`。
- `LR9-r1-fast-diagnostic` 真实可测，但仍没有 near-pass，也不参与 P6 official gate selection。
- H3 不成立：本轮没有把 low-rank r2 的 step 降 25%，也没有把 memory 降到 `<=1.05`。

## 9. P5 Piecewise Residual-Only / Mixing Decomposition

Measured packages：

| package | memory ratio mean | step ratio mean | backward ratio mean | structural/stability | residual pass | near pass |
|---|---:|---:|---:|---|---:|---:|
| `PW0-piecewise2-forwardOnly-full-current` | `1.3277` | `3.9142` | `4.0933` | pass/pass | 18/18 | 0 |
| `PW2-piecewise2-residualOnly-forward-backward` | `1.4832` | `3.5067` | `3.9670` | pass/pass | 18/18 | 0 |
| `PW3-piecewise2-streamingGrad-noMix` | `1.4832` | `3.5157` | `3.9543` | pass/pass | 18/18 | 0 |
| `PW8-piecewise4-residualOnly-streamingGrad` | `1.4858` | `3.5040` | `3.9411` | pass/fail | 18/18 | 0 |
| `PW4/PW7 lowrankMix` | `1.4862` | `4.28-4.29` | `4.36` | pass/pass | 18/18 | 0 |
| `PW5 groupedMix-vectorized` | `1.5400` | `4.2096` | `4.5826` | pass/pass | 18/18 | 0 |

Piecewise stability:

| package | active bins/sample | dead bin fraction | out-of-grid fraction |
|---|---:|---:|---:|
| `PW2` | `1.0` | `0.0694` | `0.0` |
| `PW3` | `1.0` | `0.0694` | `0.0` |
| `PW8` | `1.0` | `0.3299` | `0.0` |

P5 判断：

- v6.15 真实补上了 residual-only / noMix 分支，不再把 full piecewise 失败直接外推到 residual 本体。
- `PW2/PW3` structural/stability/residual/gradient 通过，但 memory/time 明显失败。
- `PW8` residual/gradient 通过，但 dead bin fraction `0.3299` 超过 `0.30`，稳定性失败。
- H4 不成立：piecewise residual-only 本体也不轻，不能作为当前 survivor。

## 10. P6 Selection / P7-P10 Gate

P6 最终选择：

| field | value |
|---|---:|
| best candidate | `GS4-streamed-grouped-tile1` |
| best family | `grouped` |
| best memory ratio | `1.0373867960` |
| best step ratio | `9.0878284009` |
| best backward ratio | `12.7686793416` |
| best forward ratio | `16.7699445345` |
| memory improvement vs current | `19.69%` |
| step improvement vs current | `-371.75%` |
| survivor type | `S3` |
| primary blocker | `runtime_too_slow` |
| open one-step probe | 0 |
| open task re-entry | 0 |

Gate 状态：

| stage | status | reason |
|---|---|---|
| P7 one-step probe | `not_run` | No S0/S1/S2 reset-v8 survivor |
| P8 task re-entry | `not_run` | P6 did not produce official S0/S1 task-open survivor or P7 did not pass |
| P8 task trace | `not_run` | P8 task is gated |
| P9 optimizer exploration | `not_run` | P6/P7 did not open official task |
| P10 functional correction smoke | `not_run` | P10 is gated behind P8/P9 |

这部分不能解读成 task/optimizer/functional 失败。准确说法是 v6.15 没有 S0/S1/S2 survivor，所以按计划没有进入训练。W&B 中因此没有每 20 step task loss 曲线，只有真实 profiler/summary/route 指标。

## 11. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F1_memory_fail` | 24 | measured candidates memory ratio 未过 near-pass |
| `F6_grouped_runtime_fail` | 16 | grouped branch step/time 未过 near-pass |
| `F0_not_implemented_or_gated` | 16 | fused/Triton/CUDA/adaptive/onebuffer/custom-knot 等未实现 |
| `F2_step_time_fail` | 7 | measured rows step ratio 超过 near-pass |
| `F7_lowrank_runtime_fail` | 5 | low-rank branch step/time 未过 near-pass |
| `F10_task_gated` | 4 | P7-P10 正确 gate |
| `F8_piecewise_stability_fail` | 1 | `PW8` dead-bin fraction 超过阈值 |

没有出现：

- fake data failure
- proxy row failure
- gradient correctness failure
- artifact missing failure
- runtime counter all-unavailable failure

## 12. Route Decision

`route_decision.json`：

```json
{
  "route": "R4-GroupedTradeoffUnresolved",
  "best_candidate": "GS4-streamed-grouped-tile1",
  "best_family": "grouped",
  "best_memory_ratio": 1.037386796002879,
  "best_step_ratio": 9.087828400873642,
  "best_backward_ratio": 12.76867934160884,
  "best_forward_ratio": 16.769944534522434,
  "memory_improvement_vs_current": 0.19693995010411144,
  "step_improvement_vs_current": -3.717461621822591,
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
  "reset_v8_measured": 1,
  "reset_v8_pass": 0,
  "next_required_implementation": "fused_grouped_kernel_or_stop_grouped_branch",
  "no_fake": true,
  "no_proxy": true
}
```

## 13. 结论

本轮 v6.15 支持以下真实结论：

1. v6.15 与 v6.14 baseline 可比，DWM2 current memory ratio mean 仍为 `1.2918`，DWM2 patch 继续冻结。
2. P1 不再只有 unavailable counter；code-path proxy counts 已落盘，runtime attribution 足以支持 grouped/lowrank/piecewise route 判定。
3. Grouped tile sweep 找到更强 memory-only signal：`GS4 tile1` memory ratio mean `1.0374`，优于 v6.14 G2 的 `1.0460`。
4. 但 `GS4 tile1` step ratio mean `9.0878`，backward ratio mean `12.7687`，runtime 严重失败。
5. Tile2 也接近 memory gate：`1.0493`，但 step 仍为 `5.7495`。
6. no-xg/vectorized paths 继续修 runtime到约 `1.97-1.99`，但 memory 回退到 `1.127-1.175`。
7. Low-rank r2 仍没有进入 S2：memory `1.0786`，step `~2.025`。
8. Piecewise residual-only/noMix 已真实实现并测量，但 residual 本体 memory/time 也失败，不能保留为 survivor。
9. P7-P10 正确 gate，没有 task、optimizer、functional 结论，也没有每 20 step task loss trace。
10. 最终 route 是 `R4-GroupedTradeoffUnresolved`。

最终一句话：

> v6.15 真实证明 grouped tile/stream 可以把 memory 压到 `1.037`，但 runtime 仍约 `9.09x`；vectorized/no-xg 可以把 runtime 降到约 `2x`，但 memory 回退。ResetV8 没有 S0/S1/S2 survivor，不能打开 task。下一步只有真正 fused grouped kernel 才可能合并这两个优势，否则 grouped branch 应停止作为主线。

## 14. 下一步建议

1. 若继续 grouped，必须进入 Triton/CUDA fused grouped forward/backward，不要继续 Python loop 或 torch-level tile sweep。
2. `GS4/GS5` 只能作为 memory diagnostic，不应开 task。
3. no-xg/vectorized 只能作为 runtime diagnostic，除非 memory 回到 `<=1.05`。
4. Low-rank r2 若保留为 fallback，目标仍是 memory 从 `1.0786` 到 `<=1.05`，step 从 `~2.03` 到 `<=1.50`。
5. Piecewise residual-only 本轮已失败，除非出现全新 workspace model，否则不应作为下一轮主线。
6. P8 task、P9 optimizer、P10 functional correction 继续关闭，直到出现真实 S0/S1/S2 survivor。
