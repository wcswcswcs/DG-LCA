# DG-KAN v6.12 ResetV5 BoundedWorkspace 结果复盘

> 本复盘只记录 `results/real_rerun_20260505/v612_real_all_20260505T063024Z` 下的 final real-only run。运行启动于 2026-05-05 06:30 UTC，完成于 2026-05-05 06:37 UTC。所有结论来自该目录落盘的 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v612_real.py \
  --packages V6_12_ALL \
  --out-dir results/real_rerun_20260505/v612_real_all_20260505T063024Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v612-real-20260505 \
  --wandb-name-prefix v612-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260505/v612_real_all_20260505T063024Z`
- 本地主日志：`results/real_rerun_20260505/v612_real_all_20260505T063024Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/f7sht2pm>
- W&B run name：`v612-real-v612_real_all_20260505T063024Z`

### 1.3 不用于结论的检查

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `/tmp/v612_smoke` | 极小规模 smoke，检查 runner/no-fake/CSV shape | 0 |
| `/tmp/v612_smoke2` | 修正 grouped policy 解析后的 smoke | 0 |
| `/tmp/v612_smoke3` | 修正 P7/P8 gate metadata 后的 smoke | 0 |

### 1.4 no-fake / no-proxy 约束

- 新入口：`experiments/run_gafu_v612_real.py`。
- 所有 dataset loader 调用均使用 `allow_fake_data=False`。
- `experiments/dgkan_core.py` 中 fake data 路径仍保持 hard-fail。
- P0/P1/P2/P3/P4/P5/P6/P7/P8/P9/P10 的 `fake_data_used` 总和均为 `0`。
- P0/P1/P2/P3/P4/P5/P6/P7/P8/P9/P10 的 `proxy_row_used/proxy_rows_used` 总和均为 `0`。
- DWM2 仍按 v6.10/v6.11 决策冻结，没有把局部 DWM2 patch 伪造成新 terminal full-step 结果。
- 未实现的 fused/one-buffer/piecewise/blockdiag/crossgroup 变体写为 `not_implemented`；P7-P10 因 gate 写为 `not_run`。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_12_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler/kernel 使用 `seed=0`；无 task multi-seed confirm |
| full warmup / measure | 每个 full-step measured row `50/200` update steps |
| micro warmup / measure | `50/200` calls |
| run duration | `440.79` sec |
| source commit | `f76ff27ea03c6fbfc01cadf24ea281c32a72b18f` |
| Triton | available |
| memory history | available |
| Nsight | `ncu_available=1`，`nsys_available=0` |
| W&B logging | 记录 P0-P6/P11/failure/route summary；P8 task 被 gate，未产生每 20 step loss trace |

本轮真实新增实现：

| item | 实现说明 |
|---|---|
| `ResetV5-lowrank-r4-nointermediate` | low-rank reset-v5，去掉部分中间 materialization |
| `ResetV5-lowrank-r2-fast` | rank=2 low-rank fast reset-v5 |
| `ResetV5-lowrank-r4-chunk{4,8,16,32,64}` | low-rank chunked mixing sweep |
| `ResetV5-lowrank-r8-chunk16` | rank=8 chunked low-rank diagnostic |
| `ResetV5-lowrank-r2-chunk16` | rank=2 chunked low-rank diagnostic |
| `ResetV5-grouped-g{4,8,16}-poly1` | grouped bounded reset-v5 |

仍未实现但按计划显式记录：

```text
fused lowrank forward
fused lowrank backward
lowrank one-buffer
compiled/custom-kernel lowrank
piecewise local / streamingGrad / chunkedMix / groupedMix / lowrankMix
grouped piecewise2 / blockdiag / crossgroup variants
```

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 12 |
| `p0_reproduction_check.csv` | 真实生成 | 1 |
| `p1_lowrank_attribution_detail.csv` | 真实生成 | 108 |
| `p1_component_kernel_audit.csv` | 真实生成 | 720 |
| `p2_fused_lowrank_runtime_repair.csv` | 真实生成 | 10 |
| `p2_fused_lowrank_runtime_repair_detail.csv` | 真实生成 | 198 |
| `p3_chunked_lowrank_mixing_sweep.csv` | 真实生成 | 8 |
| `p3_chunked_lowrank_mixing_detail.csv` | 真实生成 | 162 |
| `p4_grouped_block_bounded_reset.csv` | 真实生成 | 8 |
| `p4_grouped_block_reset_detail.csv` | 真实生成 | 162 |
| `p5_piecewise_local_bounded_reset.csv` | `not_implemented` rows | 6 |
| `p6_reset_v5_package_selection.csv` | 真实生成 | 1 |
| `p7_one_step_probe.csv` | `not_run` | 1 |
| `p8_task_reentry.csv` | `not_run` | 1 |
| `p8_task_trace.csv` | `not_run` | 1 |
| `p9_optimizer_exploration.csv` | `not_run` | 1 |
| `p10_functional_correction_smoke.csv` | `not_run` | 1 |
| `failure_table.csv` | 真实生成 | 772 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |

关键 hash（复盘时重新计算）：

| 文件 | SHA256 |
|---|---|
| `run.log` | `fea3342c20e7f7071ab0e4f154aea9e85208f25fd0690e5dfa5cb77222b62999` |
| `p1_component_kernel_audit.csv` | `a3419d52596f58299e7660587bf037b0e6e783e6ed0d574a7c3402b42f311f24` |
| `p2_fused_lowrank_runtime_repair.csv` | `99eb596bf401afa5455bc45e81b39dd7a68dd2d270e8d34c92776ecebcc3d8a9` |
| `p3_chunked_lowrank_mixing_sweep.csv` | `29ec08977854bc41a5780e7047cfcd2a249a7545b18e36c47573af8b770cbd1d` |
| `p4_grouped_block_bounded_reset.csv` | `e039a736f565ac7bfbe54527306da3dceab25300e78cc2f2ccf7fe01ccf03fd2` |
| `p8_task_reentry.csv` | `b5b8040f5d0f36532b56ba0243829b397e74789d1075f4e2330c74deffd0f760` |
| `failure_table.csv` | `3efd6d42a68dcf2dce6b596228feef0140ad4d5333bbbc2a070b4b11f817b5a5` |
| `route_decision.json` | `47a377eafea1124367103a450beb22afdf2d8649b57e91287b727d6d254d9371` |

## 4. P0 Contract / v6.11 Reproduction

P0 contract：

| variant | status | nonKAN | grad pass | fake/proxy |
|---|---|---:|---:|---:|
| `MLP-autograd-reference` | measured | 55050 | - | 0/0 |
| `MLP-manual-linear-reference` | measured | 0 | 1 | 0/0 |
| `DWM2-poly2-compiled` | measured frozen baseline | 0 | 1 | 0/0 |
| `ResetV4-lowrank-r4-poly1` | measured lineage baseline | 0 | 1 | 0/0 |
| `ResetV4-lowrank-r8-poly1` | measured lineage baseline | 0 | 1 | 0/0 |
| `ResetV4-chunked-poly1-c16` | measured lineage baseline | 0 | 1 | 0/0 |
| `ResetV4-chunked-poly1-c32` | measured lineage baseline | 0 | 1 | 0/0 |
| `ResetV5-lowrank-r4-nointermediate` | measured | 0 | 1 | 0/0 |
| `ResetV5-lowrank-r2-fast` | measured | 0 | 1 | 0/0 |
| `ResetV5-lowrank-r4-chunk16` | measured | 0 | 1 | 0/0 |
| `ResetV5-grouped-g4-poly1` | measured | 0 | 1 | 0/0 |
| `ResetV5-grouped-g8-poly1` | measured | 0 | 1 | 0/0 |

v6.11 reproduction：

| metric | v6.11 ref | v6.12 reproduction | delta | pass |
|---|---:|---:|---:|---:|
| DWM2 current memory ratio mean | `1.2917923089` | `1.2917923089` | `0.0` | 1 |
| B0 memory ratio mean | `1.1284832217` | `1.1284832217` | `0.0` | 1 |
| B0 step ratio mean | `2.0684196122` | `2.1419213260` | `+0.0735017138` | 1 |

P0 结论：v6.12 与 v6.11 memory baseline 可比；真实实现项梯度正确；DWM2 patch 主线继续冻结。

## 5. P1 Low-Rank Attribution / Component Audit

P1 attribution 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x (MLP + DWM2 + 4 reset-v4 lineage variants) = 108 rows
```

### 5.1 full-step summary

| method | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | grad relerr max | attribution pass |
|---|---:|---:|---:|---:|---:|
| `DWM2-poly2-compiled` | `1.1441 / 1.2918 / 1.4866` | `1.5946 / 1.9201 / 2.3265` | `1.1834 / 1.4965 / 1.9279` | `9.00e-08` | 0/18 |
| `ResetV4-lowrank-r4-poly1` | `1.0317 / 1.1285 / 1.2647` | `1.6785 / 2.0578 / 2.5227` | `1.0337 / 1.3053 / 1.6712` | `4.94e-08` | 18/18 |
| `ResetV4-lowrank-r8-poly1` | `1.0382 / 1.1344 / 1.2701` | `1.6831 / 2.0663 / 2.5566` | `1.0376 / 1.3100 / 1.7007` | `5.88e-08` | 18/18 |
| `ResetV4-chunked-poly1-c16` | `1.0996 / 1.1954 / 1.3245` | `1.8128 / 2.2504 / 2.7777` | `1.4530 / 1.9178 / 2.5364` | `2.84e-07` | 6/18 |
| `ResetV4-chunked-poly1-c32` | `1.0996 / 1.1954 / 1.3245` | `1.6227 / 1.9724 / 2.4052` | `1.1767 / 1.5067 / 1.9647` | `2.84e-07` | 6/18 |

### 5.2 component audit

`p1_component_kernel_audit.csv` 有 `720` 行，均来自 full-step phase fields。低层 kernel/allocation counter 仍不可用，因此 failure table 中保留 `F5_low_level_counter_unavailable=720`。

P1 结论：

- low-rank reset lineage 的 attribution 明显优于 DWM2 current，但 runtime 仍偏慢。
- P1 component audit 不能被解读成 Nsight 级 kernel counter；它只支持 phase-level repair target 分析。

## 6. P2 Fused Low-Rank Runtime Repair

P2 覆盖：

```text
18 shapes x (MLP reference + DWM2 baseline + B0 baseline + measured reset-v5 lowrank + not-implemented lowrank packages) = 198 detail rows
```

Measured packages：

| package | memory ratio mean | step ratio mean | backward ratio mean | memory improvement vs DWM2 | step improvement vs DWM2 | residual pass | near pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DWM2-current-baseline` | `1.2918` | `1.9949` | `1.6131` | `0.00%` | `-3.46%` | 0/18 | 0 |
| `L0-B0-lowrank-r4-current` | `1.1285` | `2.1419` | `1.3957` | `12.64%` | `-11.09%` | 18/18 | 0 |
| `L3-fused-lowrank-forward-backward` | `1.0822` | `2.1119` | `1.3669` | `16.23%` | `-9.53%` | 18/18 | 0 |
| `L4-lowrank-no-intermediate` | `1.0822` | `2.1064` | `1.3631` | `16.23%` | `-9.24%` | 18/18 | 0 |
| `L6-lowrank-r2-fast` | `1.0786` | `2.0998` | `1.3601` | `16.51%` | `-8.90%` | 18/18 | 0 |

未实现：

```text
L1-fused-lowrank-forward
L2-fused-lowrank-backward
L5-lowrank-onebuffer
L7-lowrank-r4-compiled
L8-lowrank-r4-custom-kernel
```

P2 结论：

- `L6-lowrank-r2-fast` 是 P2 中 memory 最好的 measured low-rank variant，memory ratio mean `1.0786`。
- 该 memory ratio 仍高于 near-pass `<=1.05`，step ratio mean `2.0998` 也远高于 gate。
- P2 没有 S0/S1/S2 survivor。

## 7. P3 Chunked Low-Rank Mixing Sweep

P3 覆盖：

```text
18 shapes x (MLP reference + B0 baseline + 7 measured chunk/rank variants) = 162 detail rows
```

Measured chunked packages：

| package | memory ratio mean | step ratio mean | backward ratio mean | memory improvement vs DWM2 | step improvement vs DWM2 | residual pass | near pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `C0-B0-lowrank-r4-current` | `1.1285` | `1.9693` | `1.2150` | `12.64%` | `-2.13%` | 18/18 | 0 |
| `C1-lowrank-r4-chunk4` | `1.0822` | `4.0445` | `4.1975` | `16.23%` | `-109.76%` | 18/18 | 0 |
| `C2-lowrank-r4-chunk8` | `1.0822` | `3.0245` | `2.7119` | `16.23%` | `-56.86%` | 18/18 | 0 |
| `C3-lowrank-r4-chunk16` | `1.0822` | `2.5073` | `1.9621` | `16.23%` | `-30.04%` | 18/18 | 0 |
| `C4-lowrank-r4-chunk32` | `1.0822` | `2.2474` | `1.5838` | `16.23%` | `-16.56%` | 18/18 | 0 |
| `C5-lowrank-r4-chunk64` | `1.0822` | `2.1178` | `1.3919` | `16.23%` | `-9.83%` | 18/18 | 0 |
| `C6-lowrank-r8-chunk16` | `1.0894` | `2.5048` | `1.9600` | `15.67%` | `-29.91%` | 18/18 | 0 |
| `C7-lowrank-r2-chunk16` | `1.0786` | `2.4929` | `1.9518` | `16.51%` | `-29.29%` | 18/18 | 0 |

P3 结论：

- chunking 没有改善 speed；chunk 越小，step time 越差。
- `C5` 是 r4 chunk sweep 中相对最不慢的 chunked variant，但 step ratio mean 仍为 `2.1178`。
- P3 仍没有 memory/time 同时 near-pass 的 candidate。

## 8. P4 Grouped / Block Bounded Reset

P4 覆盖：

```text
18 shapes x (MLP reference + 3 measured grouped variants + 5 not-implemented grouped/block/crossgroup variants) = 162 detail rows
```

Measured grouped packages：

| package | memory ratio mean | step ratio mean | backward ratio mean | memory improvement vs DWM2 | step improvement vs DWM2 | residual pass | near pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `G0-grouped-g4-poly1` | `1.0812` | `3.6111` | `2.9156` | `16.30%` | `-87.28%` | 18/18 | 0 |
| `G1-grouped-g8-poly1` | `1.0562` | `5.7707` | `4.7890` | `18.24%` | `-199.29%` | 18/18 | 0 |
| `G2-grouped-g16-poly1` | `1.0460` | `10.1243` | `8.5911` | `19.03%` | `-425.08%` | 18/18 | 0 |

未实现：

```text
G3-grouped-g4-piecewise2
G4-grouped-g8-piecewise2
G5-blockdiag-b4-poly1
G6-blockdiag-b8-poly1
G7-grouped-g8-crossgroup-light
```

P4 结论：

- `G2-grouped-g16-poly1` 是本轮 memory 最好的 candidate，memory ratio mean `1.0460`，已经低于 reset near-memory threshold `1.05`。
- 但 `G2` step ratio mean 是 `10.1243`，backward ratio mean 是 `8.5911`，速度严重失败。
- 因为 step/time gate 没过，`G2` 不能作为 S2，更不能打开 task。

## 9. P5 Piecewise Local Bounded Reset

P5 全部是显式 `not_implemented`：

```text
P0-piecewise2-local-forwardOnly-diagnostic
P1-piecewise2-streamingGrad
P2-piecewise4-streamingGrad
P3-piecewise2-chunkedMix
P4-piecewise2-groupedMix
P5-piecewise2-lowrankMix-r4
```

P5 没有 measured result，因此不能声称 piecewise 路线失败或成功。

## 10. P6 Package Selection

`p6_reset_v5_package_selection.csv` 与 `route_decision.json` 的最终选择：

| field | value |
|---|---:|
| best candidate | `G2-grouped-g16-poly1` |
| best family | `grouped` |
| best memory ratio | `1.0459591815` |
| best step ratio | `10.1243381061` |
| best backward ratio | `8.5910524987` |
| memory improvement vs current | `19.03%` |
| step improvement vs current | `-425.08%` |
| survivor type | `S3` |
| reset-v5 pass | 0 |
| open one-step probe | 0 |
| open task re-entry | 0 |

P6 判断：

- v6.12 出现了真实 memory-near signal：`G2` memory ratio mean `1.0460`。
- 但 speed 失败太大，因此路线不是 `S2 diagnostic task`，而是 `S3 memory-only / too slow`。
- P7/P8/P9/P10 正确 gate。

## 11. P7-P10 Gate 状态

| stage | status | reason |
|---|---|---|
| P7 one-step probe | `not_run` | No S0/S1/S2 reset-v5 survivor |
| P8 task re-entry | `not_run` | P6 did not produce official S0/S1 task-open survivor or P7 did not pass |
| P8 task trace | `not_run` | P8 task is gated |
| P9 optimizer exploration | `not_run` | P6/P7 did not open official task |
| P10 functional correction smoke | `not_run` | P10 is gated behind P8/P9 |

这部分不能解读成 task/optimizer/functional 失败。准确说法是 v6.12 的 memory/time gate 没有打开，所以按计划没有进入训练。W&B 中因此没有每 20 step task loss 曲线。

## 12. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F5_low_level_counter_unavailable` | 720 | P1 component audit 未取得真实低层 kernel/allocation counter |
| `F0_not_implemented_or_gated` | 20 | fused/one-buffer/piecewise/blockdiag/crossgroup 未实现，或 P7-P10 被 gate |
| `F2_step_time_nearpass_fail` | 16 | measured reset-v5 candidates step/time 未过 near-pass |
| `F1_memory_nearpass_fail` | 15 | measured reset-v5 candidates memory 未过 near-pass |
| `F7_reset_residual_effect_fail` | 1 | DWM2 baseline residual effect 不达标 |

没有出现：

- fake data failure
- proxy row failure
- gradient correctness failure
- artifact missing failure

## 13. Route Decision

`route_decision.json`：

```json
{
  "route": "R6-ResetV5TooSlow",
  "best_candidate": "G2-grouped-g16-poly1",
  "best_family": "grouped",
  "best_memory_ratio": 1.0459591815484715,
  "best_step_ratio": 10.124338106124204,
  "best_backward_ratio": 8.591052498692243,
  "memory_improvement_vs_current": 0.19030391001713978,
  "step_improvement_vs_current": -4.2507936830728585,
  "survivor_type": "S3",
  "open_one_step_probe": false,
  "open_task_reentry": false,
  "open_diagnostic_task": false,
  "open_optimizer_exploration": false,
  "open_functional_correction": false,
  "stop_dwm2_patching": 1,
  "reset_v5_measured": 1,
  "reset_v5_pass": 0,
  "next_required_implementation": "runtime_repair_or_new_primitive_family",
  "no_fake": true,
  "no_proxy": true
}
```

## 14. 结论

本轮 v6.12 支持以下真实结论：

1. v6.12 与 v6.11 memory baseline 可比，DWM2 current memory ratio mean 仍为 `1.2918`。
2. DWM2 patch 主线继续冻结，没有新增 terminal DWM2 implementation。
3. reset-v5 low-rank no-intermediate 方向有真实 memory 改善：`L6-lowrank-r2-fast` memory ratio mean `1.0786`，比 DWM2 current 改善 `16.51%`。
4. grouped reset-v5 给出最强 memory signal：`G2-grouped-g16-poly1` memory ratio mean `1.0460`，比 DWM2 current 改善 `19.03%`，已经接近/达到 memory near threshold。
5. 但 grouped reset-v5 runtime 严重失败：`G2` step ratio mean `10.1243`，backward ratio mean `8.5911`。
6. 所有 measured reset-v5 candidates 的 residual effect 与 gradient correctness 均真实通过，但没有 memory/time 同时过关的 survivor。
7. Piecewise/blockdiag/crossgroup 变体未实现，不能解读成失败。
8. P7-P10 正确 gate，没有 task、optimizer、functional 结论，也没有每 20 step loss trace。
9. 最终 route 是 `R6-ResetV5TooSlow`。

最终一句话：

> v6.12 真实证明 reset-v5 grouped/low-rank family 可以进一步降低 memory，其中 grouped-g16 已到 `1.046` memory ratio；但 runtime 代价过大，尤其 grouped-g16 step ratio 到 `10.12`，所以不能打开 task。下一步必须做 runtime repair 或重新设计 primitive，而不是把 memory-only signal 当成训练成功。

## 15. 下一步建议

1. 优先修 `G2-grouped-g16-poly1` 的 runtime，而不是继续只压 memory；目标是把 step ratio 从 `10.12` 降到 `<=1.35`。
2. 若保留 grouped 路线，必须实现 fused grouped mixing / grouped backward kernel，避免 Python/Torch group loop。
3. 若保留 low-rank 路线，`L6-lowrank-r2-fast` 是更平衡的起点，但仍需把 memory ratio 从 `1.0786` 降到 `<=1.05`。
4. P1 component audit 下一轮要升级为真实 kernel/allocation counter，否则 `F5_low_level_counter_unavailable` 会继续阻断 runtime diagnosis。
5. P8 task、P9 optimizer、P10 functional correction 继续关闭，直到出现真实 S0/S1/S2 reset survivor。
