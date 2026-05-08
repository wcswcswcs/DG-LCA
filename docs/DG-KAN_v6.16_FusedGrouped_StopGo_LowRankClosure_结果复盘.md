# DG-KAN v6.16 FusedGrouped StopGo LowRankClosure 结果复盘

> 本复盘只记录 `results/real_rerun_20260505/v616_real_all_20260505T095145Z` 下的 final real-only run。运行启动于 2026-05-05 09:51 UTC，完成于 2026-05-05 09:59 UTC。所有结论来自该目录落盘的 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v616_real.py \
  --packages V6_16_ALL \
  --out-dir results/real_rerun_20260505/v616_real_all_20260505T095145Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v616-real-20260505 \
  --wandb-name-prefix v616-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260505/v616_real_all_20260505T095145Z`
- 本地主日志：`results/real_rerun_20260505/v616_real_all_20260505T095145Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/xekdh96f>
- W&B run name：`v616-real-v616_real_all_20260505T095145Z`

### 1.3 不用于结论的检查

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `results/real_rerun_20260505/v616_real_all_20260505T094433Z` | 最初 runner 仍只把 fused grouped 写成 absent；用户指出后停止 | 0 |
| `/tmp/v616_smoke` | 极小规模 smoke，检查旧 runner shape | 0 |
| `/tmp/v616_fused_smoke` | 首版 Triton fused kernel smoke；因默认 TF32 导致 grad relerr 约 `2.8e-3`，修正前不用于结论 | 0 |
| `/tmp/v616_fused_smoke2` | 加入 `input_precision="ieee"` 后 smoke，通过梯度检查 | 0 |
| `/tmp/v616_fused_smoke3` | 修正 P3 gate 后 smoke，确认 task trace shape | 0 |

### 1.4 本轮 fused kernel 实现

用户问“有没有实现 fused kernel”后，本轮确实新增了真实 Triton fused grouped kernel，不再只是 `not_implemented`：

| 文件 | 位置 | 实现 |
|---|---:|---|
| `experiments/run_gafu_v616_real.py` | line 59 | `_v616_grouped_fwd_kernel`，fuse residual scaling + grouped mixing |
| `experiments/run_gafu_v616_real.py` | line 99 | `_v616_grouped_bwd_kernel`，fuse `dx + grad_mix + grad_poly` |
| `experiments/run_gafu_v616_real.py` | line 627 | `V616TritonFusedGroupedPoly1Layer` |
| `experiments/run_gafu_v616_real.py` | line 865 | policy `reset_v9_triton_grouped_g16_poly1` |

关键修正：Triton `tl.dot` 使用 `input_precision="ieee"`。不用该设置时首版 smoke 出现 `1e-3` 量级梯度误差，本轮 final run 的 fused rows 梯度 relerr 降到 `1.92e-08`。

### 1.5 no-fake / no-proxy 约束

- 所有 dataset loader 均使用 `allow_fake_data=False`。
- `experiments/dgkan_core.py` 中 fake data 路径仍保持 hard-fail。
- P0-P10 的 `fake_data_used` 总和为 `0`。
- P0-P10 的 `proxy_row_used/proxy_rows_used` 总和为 `0`。
- 未实现的 tile-specific fused grouped、blockdiag grouped、low-rank Triton/full-fused、piecewise Triton/custom-knot 等候选写为 `not_implemented`，没有写假 ratio。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_16_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler 使用 `seed=0`；diagnostic task 使用 seeds `0,1,2` |
| full warmup / measure | 每个 full-step measured row `50/200` update steps |
| task steps | `120`，每 `20` steps 记录 loss/acc |
| run duration | `461.72` sec |
| source commit | `f76ff27ea03c6fbfc01cadf24ea281c32a72b18f` |
| Triton | available |
| memory history | available |
| Nsight | `ncu_available=1`，`nsys_available=0` |
| W&B logging | 记录 P0-P10/failure/route summary；P8 diagnostic task 有每 20 step loss trace |

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 13 |
| `p0_reproduction_check.csv` | 真实生成 | 1 |
| `p1_grouped_attribution.csv` | 真实生成 | 162 |
| `p1_component_kernel_audit.csv` | 真实生成 | 1584 |
| `p2_fused_grouped_kernel.csv` | 真实生成 | 11 |
| `p2_fused_grouped_kernel_detail.csv` | 真实生成 | 216 |
| `p3_fused_grouped_fullstep_tuning.csv` | 真实生成 | 7 |
| `p3_fused_grouped_fullstep_tuning_detail.csv` | 真实生成 | 144 |
| `p4_lowrank_final_closure.csv` | 真实生成 | 11 |
| `p4_lowrank_final_closure_detail.csv` | 真实生成 | 216 |
| `p5_piecewise_lightweight_recheck.csv` | 真实生成 | 5 |
| `p5_piecewise_lightweight_recheck_detail.csv` | 真实生成 | 108 |
| `p6_reset_v9_selection.csv` | 真实生成 | 1 |
| `p7_one_step_probe.csv` | 真实生成 | 1 |
| `p8_task_reentry.csv` | diagnostic measured | 36 |
| `p8_task_trace.csv` | diagnostic measured | 216 |
| `p9_optimizer_exploration.csv` | `not_run` | 1 |
| `p10_functional_correction_smoke.csv` | `not_run` | 1 |
| `failure_table.csv` | 真实生成 | 41 |

关键 hash：

| 文件 | SHA256 |
|---|---|
| `run.log` | `c2ef16c3e4562f4b03f7f854d4f4a7d5d3c0a4bc4fa64ab7e1f220e0fb164877` |
| `p2_fused_grouped_kernel.csv` | `5690c7823f2873b6ef1ed8483d10b8456e9f658ab59e391668655ca35d260d50` |
| `p3_fused_grouped_fullstep_tuning.csv` | `bf005958388703ce9b60ebbd94a190c8ed40ef73c8129610bcdfb565fe6cfd1b` |
| `p6_reset_v9_selection.csv` | `0800a58381e34bb688829d589b1132c09dd7ad13b0c04f4f3b51d1157f9f5c62` |
| `p7_one_step_probe.csv` | `0d17ac73386c91fd8fba6fcde8a2ad118f3827343cdc80ff5f043984304931f8` |
| `p8_task_reentry.csv` | `c06aef451a71b0525e506869354682e890d3e8f6ec8ba5898b47ee7082c5fd0e` |
| `p8_task_trace.csv` | `d5bc1bc11af491e54b46c637732addf8acdb9b65f9162f9a522f9002f68f039d` |
| `failure_table.csv` | `950f8e51b222b8058d30916fbdc9eeb913b8d6bf32c93a27cecd17ab820cd6ed` |
| `route_decision.json` | `85cf05fd6a844013d35dc3e59670c66813082d33f025eaf9a8fd21ee327394eb` |

## 4. P0 Contract / v6.15 Reproduction

P0 结论：

- v6.16 与 v6.15 memory baseline 可比，DWM2 current memory ratio mean 仍为 `1.2917923089`。
- DWM2 patch 主线继续冻结。
- 新增 ResetV9 fused grouped path 是真实 code path，梯度检查通过。
- 单独 fused-forward、单独 fused-backward、tiled-fused、blockdiag grouped、low-rank full-fused 等未实现项没有被伪造成结果。

## 5. P1 Grouped Attribution

P1 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x 9 variants = 162 rows
component audit = 1584 rows
```

关键 lineage summary：

| variant | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | grad relerr max |
|---|---:|---:|---:|---:|
| `DWM2-current-baseline` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.6034` / mean `1.9357` / max `2.5801` | min `1.1911` / mean `1.5464` / max `2.4944` | `9.00e-08` |
| `GS4-streamed-grouped-tile1` | min `0.9895` / mean `1.0374` / max `1.1120` | min `6.6798` / mean `9.3690` / max `13.5864` | min `9.3764` / mean `13.6864` / max `23.4756` | `1.13e-07` |
| `GS5-streamed-grouped-tile2` | min `0.9949` / mean `1.0493` / max `1.1321` | min `4.2640` / mean `5.8311` / max `8.3704` | min `5.3537` / mean `7.7462` / max `13.2707` | `9.67e-08` |
| `L6-lowrank-r2-fast` | min `1.0077` / mean `1.0786` / max `1.1821` | min `1.6790` / mean `2.0583` / max `2.7509` | min `1.0230` / mean `1.3111` / max `2.0852` | `6.36e-08` |

P1 判断：

- v6.15 的 memory-only grouped signal 复现。
- low-rank 仍较平衡但不达 S2。
- fused grouped 的正式判断在 P2/P3，而不是 P1 lineage table。

## 6. P2 Fused Grouped Kernel

P2 覆盖：

```text
18 shapes x (MLP reference + measured grouped/fused variants + not-implemented variants) = 216 detail rows
```

真实 measured fused rows：

| package | backend | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | grad relerr max | near pass |
|---|---|---:|---:|---:|---:|---:|---:|
| `FG5-fused-grouped-forward-backward-g16` | `triton-fused-full` | `1.0312` | `1.5139` | `0.7815` | `1.2272` | `1.92e-08` | 0 |
| `FG6-fused-grouped-forward-backward-no-materialize` | `triton-fused-no-materialize` | `1.0312` | `1.5092` | `0.7791` | `1.1850` | `1.92e-08` | 0 |
| `FG9-single-kernel-grouped-mix` | `triton-single-kernel` | `1.0312` | `1.5099` | `0.7780` | `1.1827` | `1.92e-08` | 0 |

对照：

| package | memory ratio mean | step ratio mean | 判断 |
|---|---:|---:|---|
| `FG0-GS4-tile1-memory-reference` | `1.0374` | `9.4098` | memory 好但 runtime 失败 |
| `FG1-GS2-no-xg-runtime-reference` | `1.1270` | `2.0405` | runtime 较好但 memory 回退 |
| `FG2-GK8-no-xg-runtime-reference` | `1.1270` | `2.0414` | runtime 较好但 memory 回退 |

P2 判断：

- 这是真实 Triton fused grouped kernel，不是 fake/proxy。
- Fused kernel 成功合并了 memory 和 runtime 两个方向的大部分优势：memory 比 tile1 更低，step 从 `9.x` 降到约 `1.51`。
- 但 P2 standalone fused rows 的 step ratio 仍略高于 S2 gate `1.50`，因此 P2 自身 `near_pass=0`。

## 7. P3 Fused Grouped Full-Step Tuning

P3 覆盖：

```text
18 shapes x (MLP reference + measured tuning rows + not-implemented rows) = 144 detail rows
```

Measured tuning rows：

| package | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | grad relerr max | near pass |
|---|---:|---:|---:|---:|---:|---:|
| `GT0-best-P2-fused-grouped` | `1.0312` | `1.4472` | `0.7293` | `1.1621` | `1.92e-08` | 1 |
| `GT4-best-fused-no-materialize` | `1.0312` | `1.4426` | `0.7281` | `1.1425` | `1.92e-08` | 1 |

未实现：

```text
GT1-best-fused-tile1
GT2-best-fused-tile2
GT3-best-fused-tile4
GT5-best-fused-blockdiag
GT6-best-fused-with-update-prep
```

P3 判断：

- `GT4-best-fused-no-materialize` 是本轮最佳 candidate。
- 它达到 S2 near-pass：`memory <= 1.05` 且 `step <= 1.50`，梯度正确。
- 它没有达到 official task-open 的 S0/S1：memory 仍为 `1.0312 > 1.00`，step 也高于 `1.35`。

## 8. P4 Low-Rank Final Closure

Measured low-rank rows：

| package | memory ratio mean | step ratio mean | backward ratio mean | grad relerr max | near pass |
|---|---:|---:|---:|---:|---:|
| `LR0-L6-lowrank-r2-current` | `1.0786` | `1.9801` | `1.2447` | `6.36e-08` | 0 |
| `LR5-r2-no-materialize-v4` | `1.0786` | `1.9759` | `1.2425` | `6.36e-08` | 0 |
| `LR9-r1-fast-official-if-residual-pass` | `1.0772` | `1.9250` | `1.1683` | `4.15e-08` | 0 |

P4 判断：

- Low-rank 仍没有 S2。
- Fused grouped 已明显优于 low-rank fallback。

## 9. P5 Piecewise Lightweight Recheck

Measured piecewise rows：

| package | memory ratio mean | step ratio mean | backward ratio mean | grad relerr max |
|---|---:|---:|---:|---:|
| `PW0-piecewise2-residualOnly-current-reference` | `1.4832` | `3.5542` | `4.1769` | `8.23e-08` |
| `PW2-piecewise2-residualOnly-no-knot-materialization-v3` | `1.4832` | `3.5518` | `4.1738` | `8.25e-08` |

P5 判断：piecewise residual-only 继续 memory/time 失败，不作为下一步主线。

## 10. P6 Selection / P7-P10 Gate

P6 selection：

| field | value |
|---|---:|
| best candidate | `GT4-best-fused-no-materialize` |
| best memory ratio | `1.0311948157` |
| best step ratio | `1.4425903271` |
| best backward ratio | `0.7280782853` |
| best forward ratio | `1.1425000613` |
| memory improvement vs current | `20.17%` |
| step improvement vs current | `23.70%` |
| survivor type | `S2` |
| open one-step probe | 1 |
| open official task reentry | 0 |
| open diagnostic task | 1 |

P7 one-step probe：

| metric | value |
|---|---:|
| loss before | `2.4913` |
| loss after | `2.4002` |
| loss delta | `-0.0911` |
| grad norm | `0.9294` |
| one-step pass | 1 |

P8 diagnostic task：

| variant | rows | val acc mean | test acc mean | train loss delta mean |
|---|---:|---:|---:|---:|
| `MLP-autograd-reference` | 9 | `0.6981` | `0.6523` | `-2.3124` |
| `MLP-manual-linear-reference` | 9 | `0.6762` | `0.6402` | `-2.4051` |
| `GT4-best-fused-no-materialize+ManualAdamW` | 9 | `0.6224` | `0.5933` | `-2.5027` |
| `GT4-best-fused-no-materialize+ManualAdanLite` | 9 | `0.6302` | `0.5961` | `-2.5734` |

P8 trace：

```text
3 datasets x 3 seeds x 4 variants x 6 logged steps = 216 rows
logged steps = 20,40,60,80,100,120
task_mode = diagnostic
```

P9/P10 状态：

| stage | status | reason |
|---|---|---|
| P9 optimizer exploration | `not_run` | P8 diagnostic task only or task did not pass |
| P10 functional correction smoke | `not_run` | gated behind P8/P9 |

这部分不能解读成 official task 成功。准确说法是：v6.16 得到 S2 near-pass candidate，因此允许 diagnostic task trace；但 official task re-entry 仍然关闭。

## 11. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F0_not_implemented_or_gated` | 17 | tiled/blockdiag/lowrank/piecewise 部分候选未实现，或 P9/P10 gate |
| `F1_memory_fail` | 7 | measured rows memory ratio 未达对应 gate |
| `F8_grouped_runtime_fail` | 6 | grouped non-fused branch runtime 未过 |
| `F7_grouped_fused_not_implemented` | 4 | tile-specific/blockdiag fused 仍未实现 |
| `F10_lowrank_runtime_fail` | 3 | low-rank runtime 未过 |
| `F2_step_time_fail` | 2 | measured rows step time 未过 |
| `F14_task_gated` | 2 | official task/optimizer/functional 正确 gate |

没有出现：

- fake data failure
- proxy row failure
- gradient correctness failure
- artifact missing failure

## 12. Route Decision

`route_decision.json`：

```json
{
  "route": "R2-ResetV9NearPass",
  "best_candidate": "GT4-best-fused-no-materialize",
  "best_memory_ratio": 1.0311948156844268,
  "best_step_ratio": 1.4425903270641203,
  "best_backward_ratio": 0.7280782853241525,
  "best_forward_ratio": 1.1425000612767886,
  "memory_improvement_vs_current": 0.2017332750651097,
  "step_improvement_vs_current": 0.23703816789456314,
  "survivor_type": "S2",
  "open_one_step_probe": true,
  "open_task_reentry": false,
  "open_diagnostic_task": true,
  "reset_v9_pass": 1,
  "no_fake": true,
  "no_proxy": true
}
```

## 13. 结论

本轮 v6.16 支持以下真实结论：

1. 用户指出 fused kernel 未实现后，本轮确实新增并运行了真实 Triton fused grouped kernel。
2. 首版 fused smoke 因 TF32 导致梯度失败，已修正为 `input_precision="ieee"`，final run fused rows 梯度 relerr 为 `1.92e-08`。
3. P2 fused grouped measured rows 已经把 memory ratio 降到 `1.0312`，step ratio 降到约 `1.51`，但 P2 standalone 仍略高于 `1.50` gate。
4. P3 tuning 后 `GT4-best-fused-no-materialize` 达到 S2：memory `1.0312`、step `1.4426`、backward `0.7281`。
5. S2 不是 official task-open。由于没有达到 S0/S1，P8 只开启 diagnostic task trace，official task re-entry、optimizer exploration、functional correction 仍关闭。
6. P8 diagnostic task 真实运行了 3 datasets x 3 seeds x 4 variants，并在 W&B/CSV 中每 20 steps 记录 loss/acc。
7. Low-rank 和 piecewise 分支没有超过 fused grouped，继续不是主线。
8. 最终 route 是 `R2-ResetV9NearPass`。

最终一句话：

> v6.16 真实实现了 Triton fused grouped kernel，并首次得到 Reset 系列的 S2 near-pass candidate：`GT4-best-fused-no-materialize`。这不是 task 成功，但它结束了 v6.15 的 grouped memory/runtime tradeoff 僵局，下一步应围绕该 fused path 做 runtime/memory margin repair，目标从 S2 推到 S1/S0 后再打开 official task。

## 14. 下一步建议

1. 继续 fused grouped 路线，但不要回到 Python group loop / torch tile sweep。
2. 目标一：把 memory 从 `1.0312` 压到 `<1.00`，打开 S1/S0。
3. 目标二：把 step 从 `1.4426` 降到 `<=1.35`，避免 S2 只停留在 diagnostic task。
4. 优先优化 forward ratio `1.1425` 和 residual/mix update lifetime，而不是再调 low-rank/piecewise。
5. P8 official task、P9 optimizer、P10 functional correction 继续 gate，直到 fused grouped 达到真实 S0/S1。
