# DG-KAN v6.5 MemoryKernel PhaseLocal Redesign 结果复盘

> 本复盘只记录 2026-05-04 完成的 v6.5 final real-only run。所有关键结论来自 `results/real_rerun_20260504/v65_real_all_20260504T213223Z` 下的真实 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v65_real.py \
  --packages V6_5_ALL \
  --out-dir results/real_rerun_20260504/v65_real_all_20260504T213223Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v65-real-20260504 \
  --wandb-name-prefix v65-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260504/v65_real_all_20260504T213223Z`
- 本地主日志：`results/real_rerun_20260504/v65_real_all_20260504T213223Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/626rdt85>
- W&B run name：`v65-real-v65_real_all_20260504T213223Z`

### 1.3 废弃的中间 run

本轮有两个中间 run 未用于结论：

| run | 原因 | 是否用于结论 |
|---|---|---:|
| `5391okca` / `v65_real_all_20260504T212616Z` | runner 在 P1 前因 tuple 字段错误崩溃 | 0 |
| `m3pv5bqv` / `v65_real_all_20260504T212654Z` | 完整跑完，但 `failure_table.csv` 漏统计 P3 detail measured rows；修正后重跑 | 0 |

### 1.4 no-fake / no-proxy 约束

- `experiments/dgkan_core.py` 已封禁 `allow_fake_data=True`。
- v6.5 runner 调用 `load_vision_bundle(..., allow_fake_data=False)`。
- P1/P2/P3/P9 的 CSV 中 `fake_data_used` 总和均为 `0`。
- P1/P2/P3/P9 的 CSV 中 `proxy_row_used/proxy_rows_used` 总和均为 `0`。
- 未实现策略写为 `not_implemented` 或 `not_applicable...`，未写入假 ratio。
- P4/P5/P7/P8 因 gate 阻断写为 `not_run`，没有包装成失败数值或成功数值。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_5_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler 使用 `seed=0`；本轮没有 task multi-seed confirm，因为无 memory survivor |
| warmup / measure | 每个 measured row `50/200` update steps |
| measured update loops | P1 `54` rows，P2 `54` rows，P3 `54` rows，P9 `108` rows；共约 `67,500` 个 warmup+measure update steps |
| W&B logging | 记录各 measured row 的 memory/time/gradient/failure summary；本轮 task trace 被 gate，未产生每 20 step task loss 曲线 |

注意：v6.5 计划的核心是 memory kernel / phase-local profiler，不是 task recipe。由于 P6 没有产生 S0/S1 survivor，P7 task re-entry 与 P8 functional correction 没有资格执行。

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 11 |
| `p1_phase_local_memory.csv` | 真实生成 | 54 |
| `p1_memory_attribution.csv` | 真实生成 | 54 |
| `p2_single_factor_memory_repair.csv` | 真实生成 | 144 |
| `p3_combined_memory_packages.csv` | 真实生成 | 10 |
| `p3_combined_memory_packages_detail.csv` | 真实生成 | 198 |
| `p4_runtime_kernel_audit.csv` | `not_run` | 1 |
| `p5_one_step_probe.csv` | `not_run` | 1 |
| `p6_survivor_selection.csv` | 真实生成 | 1 |
| `p7_task_reentry.csv` | `not_run` | 1 |
| `p7_task_trace.csv` | `not_run` | 1 |
| `p8_functional_correction_smoke.csv` | `not_run` | 1 |
| `p9_fallback_primitive_decision.csv` | 真实生成 | 108 |
| `failure_table.csv` | 真实生成 | 462 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |

关键 hash（复盘时重新计算自 final run 文件）：

| 文件 | SHA256 |
|---|---|
| `run.log` | `de58663e37d1a9a28c71c71c672579ccc0a386e8a280a5fb4827464f333f72e0` |
| `p1_phase_local_memory.csv` | `280ab9df3a5139f743d9c4637344de2760858ef35d441774c05a4ffc0403436e` |
| `p2_single_factor_memory_repair.csv` | `e470f8b9b54e237a204122a8537993e36a4f8c01ded1657294ce0bfd97610dd5` |
| `p3_combined_memory_packages_detail.csv` | `0fa814d85cc5588265fe6ff7ee8801375cff755e93d8637099b491939e0b6d91` |
| `p9_fallback_primitive_decision.csv` | `d37ab3955e43f91789128cec4d8ccfeadcf7457cd13955d21cf5733e94e2ad9f` |
| `failure_table.csv` | `71edf939075e2f00e03efc15b1f0a26457a88248bca4b971da7f7aabe292a98e` |
| `route_decision.json` | `43036bbd77426c05c0668675245e9e462160b2a9380fdbdd442e40bf2638c971` |

## 4. P0 Implementation Contract

P0 contract 结果：

| variant | status | manual backward | nonKAN | fake/proxy |
|---|---|---:|---:|---:|
| `MLP-autograd-reference` | measured reference | 0 | 55050 | 0/0 |
| `MLP-manual-linear-reference` | measured | 1 | 0 | 0/0 |
| `DWM2-poly2-compiled-current` | measured | 1 | 0 | 0/0 |
| `DWM2-poly2-compiled-noHiddenCache` | measured | 1 | 0 | 0/0 |
| `DWM2-poly2-compiled-bf16Cache` | measured | 1 | 0 | 0/0 |
| `DWM2-poly2-compiled-noHidden+bf16Cache` | measured diagnostic | 1 | 0 | 0/0 |
| `bufferReuse` / `deltaStreaming` / combined memory optimized | `not_implemented` | - | - | 0/0 |
| `noTempPoly2` | `not_applicable_current_does_not_store_poly_temp` | - | - | 0/0 |

P0 结论：当前可执行候选仍满足 graph-free/manual backward/no fake contract；但 v6.5 计划中最重要的 `bufferReuse` 与 `deltaStreaming` 还没有真实实现，因此不能声称它们失败或成功，只能记录 `not_implemented`。

## 5. P1 Phase-Local Memory

P1 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x (MLP + current + noHiddenCache) = 54 measured rows
```

### 5.1 memory/time/gradient 汇总

| variant | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | grad relerr max | grad cos min | survivor |
|---|---:|---:|---:|---:|---:|---:|
| `P1-current` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.6477` / mean `1.9295` / max `2.1930` | min `1.2618` / mean `1.4867` / max `1.7192` | `8.38e-08` | `1.0` | 0 |
| `P1-noHiddenCache` | min `1.1423` / mean `1.2840` / max `1.4800` | min `1.7702` / mean `2.1442` / max `2.5311` | min `1.5111` / mean `1.9236` / max `2.3530` | `8.12e-08` | `0.99999988` | 0 |

### 5.2 attribution

| variant | manual cache MB | hidden cache MB | workspace/temp MB | unexplained gap MB | phase explain ratio |
|---|---:|---:|---:|---:|---:|
| `P1-current` | min `0.414` / mean `1.039` / max `1.906` | min `0.031` / mean `0.146` / max `0.375` | min `19.629` / mean `22.730` / max `26.753` | min `18.550` / mean `21.570` / max `25.513` | `1.0` |
| `P1-noHiddenCache` | min `0.383` / mean `0.893` / max `1.531` | `0.0` | min `19.629` / mean `22.730` / max `26.753` | min `18.550` / mean `21.570` / max `25.513` | `1.0` |

P1 判断：

- `noHiddenCache` 真实消除了 hidden cache，但 memory ratio 只从 mean `1.2918` 到 `1.2840`，下降约 `0.58%`。
- `noHiddenCache` 的 step time 变差，mean 从 `1.9295` 到 `2.1442`。
- phase peak explain ratio 为 `1.0`，说明本轮 profiler 捕获到 peak phase；但 manual cache 远小于 workspace/temp/unexplained gap，说明 hidden cache 不是主因。
- P1 没有 memory survivor。

## 6. P2 Single-Factor Memory Repair

P2 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x (MLP + current + bf16Cache + not-implemented/not-applicable factors)
```

真实 measured factor：

| factor | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | grad relerr max | grad cos min | survivor |
|---|---:|---:|---:|---:|---:|---:|
| `M0-current` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.5850` / mean `1.8946` / max `2.2119` | min `1.0508` / mean `1.4646` / max `1.8729` | `8.38e-08` | `1.0` | 0 |
| `M1-bf16Cache` | min `1.1777` / mean `1.3674` / max `1.6109` | min `1.6170` / mean `1.9393` / max `2.2685` | min `1.1211` / mean `1.5005` / max `1.9091` | `1.61e-03` | `0.99999869` | 0 |

未实现/不适用：

| factor | status |
|---|---|
| `M2-deltaStreaming` | `not_implemented` |
| `M3-bufferReuse` | `not_implemented` |
| `M4-noTempPoly2` | `not_applicable_current_does_not_store_poly_temp` |
| `M5-updateInPlaceSafe` | `not_applicable_current_update_already_in_place` |
| `M6-optimizerStateSplit` | `not_implemented` |

P2 判断：

- `bf16Cache` 没有降低 total CUDA peak，反而让 memory ratio mean 从 `1.2918` 上升到 `1.3674`。
- `bf16Cache` 梯度正确性失败：`grad_relerr max = 1.61e-03`，超过 `1e-4` gate。
- 真实可测 factor 没有任何 memory/time survivor。

## 7. P3 Combined Memory Packages

P3 detail 覆盖：

```text
198 rows = 18 shapes x (MLP + C0-current + Cdiag-noHidden+bf16Cache + 8 not-implemented combos)
```

真实 measured combination：

| combination | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | grad relerr max | grad cos min | both-pass count |
|---|---:|---:|---:|---:|---:|---:|
| `C0-current` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.6271` / mean `2.0263` / max `2.5616` | min `1.2974` / mean `1.6559` / max `2.4743` | `8.38e-08` | `1.0` | 0 |
| `Cdiag-noHidden+bf16Cache` | min `1.1751` / mean `1.3558` / max `1.6010` | min `1.7422` / mean `2.2802` / max `2.9698` | min `1.5487` / mean `2.1756` / max `3.4586` | `1.62e-03` | `0.99999881` | 0 |

未实现 combinations：

```text
C1-bufferReuse+deltaStreaming
C2-bufferReuse+bf16Cache
C3-deltaStreaming+bf16Cache
C4-bufferReuse+noTempPoly2
C5-bufferReuse+deltaStreaming+bf16Cache
C6-bufferReuse+deltaStreaming+updateInPlaceSafe
C7-allMemoryOptimized-light
C8-allMemoryOptimized-full
```

P3 判断：

- `noHidden+bf16Cache` 是 diagnostic 组合，不是有效修复；memory/time 都比 current 差。
- P3 没有 memory-pass、time-pass、gradient-pass 三者同时成立的 candidate。
- 未实现组合没有被伪造成失败数值。

## 8. P4-P8 Gate 状态

| stage | status | reason |
|---|---|---|
| P4 runtime kernel audit | `not_run` | No memory near-pass candidate from P2/P3 |
| P5 one-step probe | `not_run` | No memory/time survivor |
| P6 survivor selection | measured decision | survivor type `S3` |
| P7 task re-entry | `not_run` | P6 produced no S0/S1 survivor |
| P7 task trace | `not_run` | P7 is gated |
| P8 functional correction smoke | `not_run` | P8 is gated behind P7 |

这部分不能解读成“task 训练失败”。准确说法是：v6.5 memory gate 没有过，后续 task/functional 按计划被阻断。

## 9. P9 Fallback Primitive Decision

由于 P6 是 `S3`，P9 fallback 被触发。

| fallback | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | grad relerr max | grad cos min | near pass |
|---|---:|---:|---:|---:|---:|---:|
| `F0-DWM2-poly3` | min `1.1673` / mean `1.3411` / max `1.5685` | min `1.7216` / mean `2.0955` / max `2.4441` | min `1.3993` / mean `1.8180` / max `2.2550` | `8.37e-08` | `1.0` | 0 |
| `F1-DWM2-poly2+silu-base` | min `1.2120` / mean `1.4377` / max `1.7308` | min `2.0098` / mean `2.4941` / max `2.9594` | min `1.5882` / mean `2.0790` / max `2.6077` | `8.46e-08` | `0.99999970` | 0 |
| `F2-RationalKAT-lite-fastpoly` | min `1.3593` / mean `1.7309` / max `2.2187` | min `1.8467` / mean `2.2670` / max `2.7383` | min `1.6223` / mean `2.1216` / max `2.6665` | `7.30e-08` | `1.0` | 0 |
| `F3-SparseInterp fused rewrite` | min `1.3794` / mean `1.8080` / max `2.3290` | min `3.1540` / mean `4.1324` / max `5.1527` | min `3.4375` / mean `4.7063` / max `6.1711` | `3.62e-08` | `1.0` | 0 |
| `F4-DWM2-RBFK2 optimized` | min `1.2971` / mean `1.6393` / max `2.0514` | min `1.6767` / mean `2.0319` / max `2.3947` | min `1.3003` / mean `1.6716` / max `2.0825` | `4.01e-08` | `0.99999976` | 0 |

P9 判断：

- fallback primitive 中没有 near-pass candidate。
- 最低 fallback memory ratio 是 `F0-DWM2-poly3` 的 `1.1673`，仍高于 `1.10` near-pass 阈值，也高于 `1.0` memory pass。
- route 从 P6 的 `R3` 进一步落到 `R6`。

## 10. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F1_memory_fail` | 162 | P2/P3/P9 measured candidates 的 memory ratio 均未低于 1 |
| `F2_step_time_fail` | 162 | P2/P3/P9 measured candidates 的 step ratio 均超过 1.35 |
| `F3_gradient_correctness_fail` | 36 | `bf16Cache` 与 `noHidden+bf16Cache` 的 relerr 超过 `1e-4` |
| `F11_gated_not_run` | 102 | 未实现、不可用或 gate 阻断阶段 |

没有出现：

- fake data failure
- proxy row failure
- artifact integrity failure

## 11. Route Decision

`route_decision.json`：

```json
{
  "route": "R6",
  "best_memory_ratio": 1.1750557413600893,
  "best_step_ratio": 1.742215198774254,
  "survivor_type": "S3",
  "open_task_reentry": false,
  "open_functional_correction": false,
  "no_proxy": true,
  "fallback_near_pass_count": 0,
  "fallback_triggered": true
}
```

`aggregate_decision.json`：

```json
{
  "status": "gated",
  "fake_data_used": 0,
  "proxy_rows_used_as_results": 0,
  "best_memory_ratio": 1.1750557413600893,
  "best_step_ratio": 1.742215198774254,
  "survivor_type": "S3",
  "fallback_triggered": true,
  "fallback_near_pass_count": 0,
  "route": "R6"
}
```

## 12. 结论

本轮 v6.5 支持以下真实结论：

1. `DWM2-poly2-compiled-current` 的 manual gradient correctness 仍然可靠，但 memory/time gate 仍失败。
2. hidden cache 不是主要 memory peak 来源；去掉 hidden cache 只带来约 `0.58%` mean memory ratio 下降，同时 step time 变差。
3. `bf16Cache` 不是可用修复：total peak 没降，mean memory ratio 反而升至 `1.3674`，并且梯度 relerr 达到约 `1.6e-03`。
4. `noHidden+bf16Cache` 组合也不是可用修复：memory/time 均变差，梯度 relerr 同样超过 gate。
5. v6.5 当前没有 S0/S1/S2 memory survivor，因此 P7/P8 正确 gate，没有 task/functional 结论。
6. P9 fallback 全部没有 near-pass，最终 route 是 `R6`。

最终一句话：

> v6.5 没有得到 memory-pass graph-free PureKAN candidate，也没有得到 fallback near-pass primitive；正确下一步不是继续跑 task seeds，而是先实现真实的 `bufferReuse` / `deltaStreaming` / workspace temp 降峰 kernel，再重新跑 P1-P3。

## 13. 下一步建议

优先级建议：

1. 实现真实 `M3-bufferReuse`，目标是降低 actual CUDA peak，而不是只改 manual cache estimate。
2. 实现真实 `M2-deltaStreaming`，逐层 consume upstream delta 并复用 buffer。
3. 加强 phase-local attribution，把 workspace/temp/unexplained gap 分解到具体 tensor / op / allocator phase。
4. 暂停 `bf16Cache` 作为主线，除非先解决 `1e-3` 量级 gradient relerr。
5. 只有 P1/P2/P3 出现 `memory_ratio < 1.0` 且 `step_ratio <= 1.35` 的真实 candidate，才重新打开 P7 task re-entry。
