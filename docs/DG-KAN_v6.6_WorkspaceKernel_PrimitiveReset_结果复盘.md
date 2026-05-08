# DG-KAN v6.6 WorkspaceKernel PrimitiveReset 结果复盘

> 本复盘只记录 2026-05-04 完成的 v6.6 final real-only run。所有关键结论来自 `results/real_rerun_20260504/v66_real_all_20260504T221940Z` 下的真实 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v66_real.py \
  --packages V6_6_ALL \
  --out-dir results/real_rerun_20260504/v66_real_all_20260504T221940Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v66-real-20260504 \
  --wandb-name-prefix v66-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260504/v66_real_all_20260504T221940Z`
- 本地主日志：`results/real_rerun_20260504/v66_real_all_20260504T221940Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/djoyo1wl>
- W&B run name：`v66-real-v66_real_all_20260504T221940Z`

### 1.3 废弃的中间 run

| run | 原因 | 是否用于结论 |
|---|---|---:|
| `/tmp/v66_smoke` | 极小规模 smoke test，无 W&B，只用于检查 runner shape/buffer 错误 | 0 |
| `7f9fj2wz` / `v66_real_all_20260504T221459Z` | P6 gate 误把 manual-linear reference 的 attribution pass 算作 DWM2 attribution pass；修正后重跑 | 0 |

### 1.4 no-fake / no-proxy 约束

- 新入口为 `experiments/run_gafu_v66_real.py`。
- 所有 dataset loader 调用均使用 `allow_fake_data=False`。
- P1/P2/P3/P9 的 CSV 中 `fake_data_used` 总和均为 `0`。
- P1/P2/P3/P9 的 CSV 中 `proxy_row_used/proxy_rows_used` 总和均为 `0`。
- 未实现策略写为 `not_implemented` 或 `not_implemented_bf16_frozen_after_v65_grad_fail`，没有写假 ratio。
- P5/P7/P8 因 gate 阻断写为 `not_run`，没有包装成失败数值或成功数值。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_6_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler 使用 `seed=0`；无 task multi-seed confirm |
| warmup / measure | 每个 measured row `50/200` update steps |
| run duration | `228.47` sec |
| W&B logging | 记录 memory/time/gradient/route/failure summary；task 被 gate，未产生每 20 step task loss 曲线 |

本轮真实新增实现：

| variant | 实现说明 |
|---|---|
| `bufferReuse-v1` | 新增预分配 poly transform / dz / deriv / output / grad tmp workspace buffer 的 poly2 manual layer |
| `deltaStreaming-v1` | backward 中对 activation delta 做原地 consume，避免额外保留新 delta tensor |
| `bufferReuse+deltaStreaming` | 组合上述两者 |
| `poly1-minimal` | P9 reset diagnostic：把 poly2 降为 poly1 的 workspace stack |

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 8 |
| `p0_reproduction_check.csv` | 真实生成 | 1 |
| `p1_phase_local_attribution.csv` | 真实生成 | 126 |
| `p1_tensor_lifetime_topk.csv` | 真实生成 | 1080 |
| `p1_allocation_phase_summary.csv` | 真实生成 | 108 |
| `p2_single_factor_workspace_repair.csv` | 真实生成 | 144 |
| `p2_repair_gradient_correctness.csv` | 真实生成 | 72 |
| `p3_combined_workspace_packages.csv` | 真实生成 | 9 |
| `p3_combined_workspace_packages_detail.csv` | 真实生成 | 180 |
| `p4_runtime_kernel_audit.csv` | 真实生成 | 90 |
| `p5_one_step_probe.csv` | `not_run` | 1 |
| `p6_survivor_selection.csv` | 真实生成 | 1 |
| `p7_task_reentry.csv` | `not_run` | 1 |
| `p7_task_trace.csv` | `not_run` | 1 |
| `p8_functional_correction_smoke.csv` | `not_run` | 1 |
| `p9_primitive_reset_kernel_benchmark.csv` | 真实生成 | 126 |
| `failure_table.csv` | 真实生成 | 1099 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |

关键 hash（复盘时重新计算自 final run 文件）：

| 文件 | SHA256 |
|---|---|
| `run.log` | `6b5d094997be67d9aa48294abff806a7c74b9f533bc6130b50a19a0ccb8dd3de` |
| `p1_phase_local_attribution.csv` | `b78f8efaef29da9a35891cf1f0a3131c3437cbceb93fedf8cb938ef9e774a2b6` |
| `p2_single_factor_workspace_repair.csv` | `f7cbdc765786e11f57691e02ae8427f8e7fc65c7258c498fb39c57c87b1434fb` |
| `p3_combined_workspace_packages_detail.csv` | `a78be757c0d1e56fb0c7cf1d9545060f044515163c8f0e547db5596f3079a7c4` |
| `p9_primitive_reset_kernel_benchmark.csv` | `aa1cd39794af455634ae231fdc7948fe403f0361123a363bd665c42087efd66c` |
| `failure_table.csv` | `22a6c1c0495a4d512dc30c4aa2a1fec8ed6b1e0df992a8400bffb7436170b012` |
| `route_decision.json` | `910c1939164eb8fe4b68e26397aa6a7a1998665edeb33297c2958b406dccf27f` |

## 4. P0 Contract 与 v6.5 Reproduction

P0 contract：

| variant | status | manual backward | nonKAN | rollback | fake/proxy | workspace pool MB |
|---|---|---:|---:|---:|---:|---:|
| `MLP-autograd-reference` | measured reference | 0 | 55050 | 0 | 0/0 | - |
| `MLP-manual-linear-reference` | measured | 1 | 0 | 0 | 0/0 | 0 |
| `DWM2-poly2-compiled-current` | measured | 1 | 0 | 0 | 0/0 | 0 |
| `DWM2-poly2-compiled-noHiddenCache` | measured | 1 | 0 | 0 | 0/0 | 0 |
| `DWM2-poly2-compiled-bf16Cache-diagnostic` | measured | 1 | 0 | 0 | 0/0 | 0 |
| `DWM2-poly2-bufferReuse-v1` | measured | 1 | 0 | 0 | 0/0 | 1.085 |
| `DWM2-poly2-deltaStreaming-v1` | measured | 1 | 0 | 0 | 0/0 | 0 |
| `DWM2-poly2-bufferReuse+deltaStreaming` | measured | 1 | 0 | 0 | 0/0 | 1.085 |

v6.5 reproduction check：

| metric | v6.5 final ref | v6.6 reproduction | delta | pass |
|---|---:|---:|---:|---:|
| current memory ratio min | `1.1440635452` | `1.1440635452` | `0.0` | 1 |
| current step ratio min | `1.5850136586` | `1.5289121645` | `-0.0561014941` | 1 |

P0 结论：v6.6 与 v6.5 profiler baseline 可比较；新增 workspace variants 是真实代码路径，不是 proxy row。

## 5. P1 Phase-Local Attribution v2

P1 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x (MLP + manual-linear + 5 DWM2 variants) = 126 measured rows
```

### 5.1 汇总

| variant | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | grad relerr max | attr pass | survivor |
|---|---:|---:|---:|---:|---:|---:|
| `A0-MLP-manual-linear-reference` | min `1.0652` / mean `1.1085` / max `1.1718` | min `0.9904` / mean `1.1340` / max `1.2532` | min `0.5536` / mean `0.6640` / max `0.7336` | `3.49e-08` | 3/18 | 0 |
| `A1-DWM2-current` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.5289` / mean `1.9079` / max `2.2236` | min `1.1100` / mean `1.4752` / max `1.7471` | `8.38e-08` | 0/18 | 0 |
| `A2-DWM2-noHiddenCache` | min `1.1423` / mean `1.2840` / max `1.4800` | min `1.5645` / mean `2.1142` / max `2.5691` | min `1.2627` / mean `1.9024` / max `2.4303` | `8.12e-08` | 0/18 | 0 |
| `A3-DWM2-bf16Cache-diagnostic` | min `1.1777` / mean `1.3674` / max `1.6109` | min `1.4972` / mean `1.9568` / max `2.2959` | min `1.0917` / mean `1.5189` / max `1.8020` | `1.61e-03` | 0/18 | 0 |
| `A4-DWM2-bufferReuse-v1` | min `1.1652` / mean `1.3470` / max `1.5851` | min `1.5436` / mean `2.0101` / max `2.3857` | min `1.1417` / mean `1.5929` / max `1.9270` | `2.75e-07` | 0/18 | 0 |
| `A5-DWM2-deltaStreaming-v1` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.4863` / mean `1.9184` / max `2.2293` | min `1.0817` / mean `1.4874` / max `1.7588` | `9.29e-08` | 0/18 | 0 |

### 5.2 Attribution 判断

| variant | unexplained gap MB | phase explain ratio | workspace pool MB |
|---|---:|---:|---:|
| `A1-DWM2-current` | min `1.770` / mean `3.880` / max `6.989` | `1.0` | 0 |
| `A2-DWM2-noHiddenCache` | min `1.770` / mean `3.880` / max `6.989` | `1.0` | 0 |
| `A3-DWM2-bf16Cache-diagnostic` | min `2.597` / mean `5.812` / max `10.176` | `1.0` | 0 |
| `A4-DWM2-bufferReuse-v1` | min `2.171` / mean `4.913` / max `8.675` | `1.0` | min `1.960` / mean `4.747` / max `8.742` |
| `A5-DWM2-deltaStreaming-v1` | min `1.770` / mean `3.880` / max `6.989` | `1.0` | 0 |

P1 结论：

- DWM2 variants 的 `attribution_pass` 全部为 `0/18`。
- phase peak 本身捕获到了 peak phase，但 v6.6 的 tensor attribution 仍不能把 DWM2-over-MLP gap 分解到计划要求的可行动 top-3 source。
- 因此 P6 route 必须以 `R3-workspaceAttributionIncomplete` 为主，不能把后续 repair 结果解释成完整 workspace diagnosis。

## 6. P2 Single-Factor Workspace Repair

P2 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x (MLP + current + bufferReuse + deltaStreaming + bufferReuse+deltaStreaming + not-implemented factors)
```

真实 measured factor：

| factor | memory ratio vs MLP | step ratio vs MLP | memory reduction vs current | grad relerr max | useful pass |
|---|---:|---:|---:|---:|---:|
| `R0-current` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.5793` / mean `2.0118` / max `2.5554` | - | `8.38e-08` | 0 |
| `R1-bufferReuse-v1` | min `1.1652` / mean `1.3470` / max `1.5851` | min `1.6506` / mean `2.1215` / max `2.7491` | mean `-4.17%` | `2.75e-07` | 0 |
| `R2-deltaStreaming-v1` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.5838` / mean `2.0161` / max `2.5984` | mean `0.00%` | `9.29e-08` | 0 |
| `R6-bufferReuse+deltaStreaming` | min `1.1616` / mean `1.3392` / max `1.5723` | min `1.6520` / mean `2.1122` / max `2.7211` | mean `-3.59%` | `2.75e-07` | 0 |

未实现/冻结：

| factor | status |
|---|---|
| `R3-updateWorkspaceReuse-v1` | `not_implemented` |
| `R4-polyTransformNoAlloc-v1` | `not_implemented` |
| `R5-safeMixedCache-diagnostic` | `not_implemented_bf16_frozen_after_v65_grad_fail` |

P2 结论：

- `bufferReuse-v1` 是真实实现，但没有降低 actual CUDA peak；memory ratio mean 从 `1.2918` 变差到 `1.3470`。
- `deltaStreaming-v1` 梯度正确，但 memory ratio 与 current 完全相同，未降低 peak。
- `bufferReuse+deltaStreaming` 仍比 current 差。
- P2 没有 diagnostic+memory useful+near-pass 同时成立的 factor。

## 7. P3 Combined Workspace Packages

真实 measured package：

| package | memory ratio vs MLP | step ratio vs MLP | memory reduction vs current | grad pass | near pass |
|---|---:|---:|---:|---:|---:|
| `C0-current` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.6152` / mean `1.9184` / max `2.3125` | `0.00%` | 1 | 0 |
| `C1-bufferReuse` | min `1.1652` / mean `1.3470` / max `1.5851` | min `1.6695` / mean `2.0172` / max `2.4541` | `-4.17%` | 1 | 0 |
| `C2-deltaStreaming` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.6087` / mean `1.9203` / max `2.3309` | `0.00%` | 1 | 0 |
| `C3-bufferReuse+deltaStreaming` | min `1.1616` / mean `1.3392` / max `1.5723` | min `1.6651` / mean `2.0114` / max `2.4592` | `-3.59%` | 1 | 0 |
| `C7-allWorkspaceOptimized-light` | min `1.1616` / mean `1.3392` / max `1.5723` | min `1.6685` / mean `2.0100` / max `2.4583` | `-3.59%` | 1 | 0 |

未实现 packages：

```text
C4-bufferReuse+updateWorkspaceReuse
C5-deltaStreaming+updateWorkspaceReuse
C6-bufferReuse+deltaStreaming+polyNoAlloc
C8-allWorkspaceOptimized-full
```

P3 结论：

- 没有 S0/S1/S2 survivor。
- `best_package` 仍是 `C0-current`，因为新增 workspace packages 没有带来实际 memory improvement。
- `survivor_type = S3`。

## 8. P4-P8 Gate 状态

| stage | status | reason |
|---|---|---|
| P4 runtime kernel audit | measured diagnostic | 对 P3 measured rows 做 runtime/kernel summary |
| P5 one-step probe | `not_run` | No S0/S1/S2 package survivor |
| P6 survivor selection | measured decision | `S3` + `R3-workspaceAttributionIncomplete` |
| P7 task re-entry | `not_run` | P6 did not open task re-entry |
| P7 task trace | `not_run` | P7 is gated |
| P8 functional correction smoke | `not_run` | P8 is gated behind P7 |

这部分不能解读成 task 失败；准确说法是 memory/workspace gate 没有过，所以 task/functional 正确关闭。

## 9. P9 Primitive Reset Kernel Benchmark

P9 被触发，但由于 route 主因是 P1 attribution incomplete，P9 只作为 reset diagnostic，不打开 task。

真实 measured reset：

| reset | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | grad relerr max | near pass |
|---|---:|---:|---:|---:|---:|
| `R0-ManualLinear+TinyChannelResidual` | min `1.0652` / mean `1.1085` / max `1.1718` | min `1.0791` / mean `1.1853` / max `1.5328` | min `0.6381` / mean `0.6925` / max `0.8157` | `3.49e-08` | 11/18 |
| `R1-DWM2-poly1-minimal` | min `1.1422` / mean `1.2981` / max `1.5058` | min `1.5653` / mean `1.8302` / max `2.1771` | min `1.0835` / mean `1.2823` / max `1.6208` | `7.58e-08` | 0/18 |
| `R2-DWM2-poly2-singleBuffer` | min `1.1616` / mean `1.3392` / max `1.5723` | min `1.7337` / mean `2.0698` / max `2.4868` | min `1.3538` / mean `1.6563` / max `2.1327` | `2.75e-07` | 0/18 |

未实现 reset：

```text
R3-SparseInterp-gather2-forwardOnly
R4-SparseInterp-gather2-streamingGrad
R5-RationalKAT-oneBuffer-fastpoly
```

P9 判断：

- `fallback_near_pass_count = 11` 只来自 `R0-ManualLinear+TinyChannelResidual` diagnostic。
- DWM2-family reset (`poly1-minimal` / `poly2-singleBuffer`) 没有 near-pass。
- 因为 DWM2 P1 attribution 未通过，最终 route 保持 `R3-workspaceAttributionIncomplete`，不能用 manual-linear reset near-pass 直接重开 task。

## 10. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F1_memory_fail` | 414 | measured rows memory ratio 仍未低于 1 |
| `F2_step_time_fail` | 379 | measured rows step ratio 超过 1.35 |
| `F12_attribution_incomplete` | 105 | DWM2/manual measured rows 中多数未达到 attribution gate；DWM2 全部未过 |
| `F3_gradient_correctness_fail` | 18 | bf16 diagnostic 仍然梯度 relerr 超 gate |
| `F11_gated_not_run` | 183 | 未实现、冻结或 gate 阻断 |

没有出现：

- fake data failure
- proxy row failure
- artifact integrity failure

## 11. Route Decision

`route_decision.json`：

```json
{
  "route": "R3-workspaceAttributionIncomplete",
  "best_package": "C0-current",
  "best_memory_ratio": 1.2917923088533285,
  "best_step_ratio": 1.9183739604097723,
  "best_backward_ratio": 1.497198764394076,
  "memory_improvement_vs_current": 0.0,
  "step_improvement_vs_current": 0.0,
  "survivor_type": "S3",
  "fallback_triggered": true,
  "fallback_near_pass_count": 11,
  "open_task_reentry": false,
  "open_functional_correction": false,
  "no_fake": true,
  "no_proxy": true,
  "primary_blocker": "P1 attribution did not reduce unexplained peak gap to plan threshold",
  "next_required_implementation": "cuda_profiler_trace_and_lower_level_workspace_kernel"
}
```

## 12. 结论

本轮 v6.6 支持以下真实结论：

1. v6.6 与 v6.5 baseline 复现通过，当前结果可与 v6.5 对比。
2. 新增 `bufferReuse-v1` 和 `deltaStreaming-v1` 是真实实现，但没有降低 DWM2 actual CUDA peak。
3. `bufferReuse-v1` 反而增加 memory ratio 和 step time，说明 Python/Torch 层预分配 workspace pool 没有解决当前 allocator/peak 问题。
4. `deltaStreaming-v1` 梯度正确，但 memory ratio 与 current 完全一致，说明当前 peak 不是这个 delta lifetime 改法能处理的部分。
5. P1 attribution 对 DWM2 全部未通过，因此最终 route 是 `R3-workspaceAttributionIncomplete`。
6. P9 manual-linear reset diagnostic 出现 near-pass，但 DWM2 reset 没有 near-pass；不能据此打开 task 或 functional correction。

最终一句话：

> v6.6 真实实现了 buffer reuse / delta streaming，但 DWM2 memory peak 没有下降；当前 blocker 已经不是“再跑 task”的问题，而是需要 CUDA profiler trace 或更低层 Triton/CUDA workspace kernel 才能继续定位和修复。

## 13. 下一步建议

优先级建议：

1. 加入真实 CUDA allocation trace：记录 allocation stack、tensor lifetime、allocator block reuse，而不是只做 phase peak。
2. 把 poly2 transform/backward 写到 Triton/CUDA fused kernel，避免 Python/Torch 层 workspace pool 反而增加 peak。
3. 继续冻结 bf16 cache 主线，除非先把 `1e-3` relerr 降到 `<1e-4`。
4. 若要走 primitive reset，优先研究 manual-linear-like one-buffer residual primitive；但它不是当前 DWM2-poly2 的成功结论。
5. P7/P8 继续关闭，直到 DWM2 或明确 reset primitive 同时通过 real memory/time/gradient gate。
