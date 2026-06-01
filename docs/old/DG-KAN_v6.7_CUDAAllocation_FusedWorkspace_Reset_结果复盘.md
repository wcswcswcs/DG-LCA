# DG-KAN v6.7 CUDAAllocation FusedWorkspace Reset 结果复盘

> 本复盘只记录 2026-05-04 完成的 v6.7 final real-only run。所有关键结论来自 `results/real_rerun_20260504/v67_real_all_20260504T232309Z` 下的真实 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v67_real.py \
  --packages V6_7_ALL \
  --out-dir results/real_rerun_20260504/v67_real_all_20260504T232309Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v67-real-20260504 \
  --wandb-name-prefix v67-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260504/v67_real_all_20260504T232309Z`
- 本地主日志：`results/real_rerun_20260504/v67_real_all_20260504T232309Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/5cthipet>
- W&B run name：`v67-real-v67_real_all_20260504T232309Z`

### 1.3 不用于结论的检查

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `/tmp/v67_smoke` | 极小规模 smoke，检查 runner/FlashKAT import/CSV shape | 0 |

本轮 final run 结束后做了一次 gate 审计：原始测量 CSV 未改动，只把 route 逻辑修正为必须看 `A2-DWM2-current` 的 attribution pass。因为 current DWM2 的 attribution pass 是 `0/18`，最终 route 保守写为 `R3-AttributionIncomplete`。

### 1.4 no-fake / no-proxy 约束

- 新入口：`experiments/run_gafu_v67_real.py`。
- 所有 dataset loader 调用均使用 `allow_fake_data=False`。
- P1/P2/P2.5/P3/P4 的 `fake_data_used` 与 `proxy_row_used/proxy_rows_used` 均为 `0`。
- Nsight / lower-level counters 不可用时写为 `metric_unavailable`，没有填假数。
- 未实现 CUDA/Triton package 写为 `not_implemented`；P5/P6/P7 因 gate 写为 `not_run`。
- FlashKAT 只做真实 import/smoke audit，不被当作 DWM2-poly2 成功结果。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_7_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| micro datasets | `Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler/kernel 使用 `seed=0`；无 task multi-seed confirm |
| full warmup / measure | 每个 full-step measured row `50/200` update steps |
| micro warmup / measure | 每个 coeffgrad micro-kernel row `50/200` calls |
| trace | `trace_batch_size=128`，`trace_steps=20` |
| run duration | manifest 含 route 审计后处理 `403.36` sec |
| W&B logging | 记录 P0/P1/P2/P2.5/P3/P4/P8/failure summary；P6 task 被 gate，未产生 loss trace |

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 12 |
| `flashkat_source_audit.csv` | 真实生成 | 1 |
| `p0_reproduction_check.csv` | 真实生成 | 1 |
| `p1_cuda_allocation_trace.csv` | 真实生成 | 900 |
| `p1_tensor_lifetime_topk.csv` | 真实生成 | 30 |
| `p1_allocator_block_summary.csv` | 真实生成 | 1 |
| `p1_phase_peak_summary.csv` | 真实生成 | 180 |
| `p1_coeffgrad_attribution.csv` | 真实生成 | 126 |
| `p2_component_microkernel.csv` | 真实生成 | 15 |
| `p2_component_repair_summary.csv` | 真实生成 | 15 |
| `p25_flash_coeffgrad_audit.csv` | 真实生成 | 64 |
| `p25_flash_coeffgrad_correctness.csv` | 真实生成 | 40 |
| `p25_flash_coeffgrad_memory_stall.csv` | 真实生成 | 64 |
| `p3_fused_dwm2_packages.csv` | 真实生成 | 12 |
| `p3_fused_dwm2_package_detail.csv` | 真实生成 | 234 |
| `p4_bounded_workspace_reset.csv` | 真实生成 | 198 |
| `p5_one_step_probe.csv` | `not_run` | 1 |
| `p6_task_reentry.csv` | `not_run` | 1 |
| `p6_task_trace.csv` | `not_run` | 1 |
| `p7_functional_correction_smoke.csv` | `not_run` | 1 |
| `failure_table.csv` | 真实生成 | 964 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |

关键 hash：

| 文件 | SHA256 |
|---|---|
| `run.log` | `ea737195af8efccd5833464fb99a1b0982363ad0f02c2ccaf9c6d3d9d56a1a5c` |
| `p1_phase_peak_summary.csv` | `7880612ef169aed501786d025bdeea6bdc534755d173ed9e430a05c70669379a` |
| `p25_flash_coeffgrad_audit.csv` | `da439f5a29441e7de812c16cddb9493c8299bddd050ee2f9905f4d46c75ccf69` |
| `p3_fused_dwm2_packages.csv` | `f5d0cb913c861ef7834840deb1bf2943433ed61b1f0ef28c0137f49d091a88b4` |
| `p4_bounded_workspace_reset.csv` | `961b39355dc0dfe1b843627cb9ad99abb74ff3f9422dfd4733b0c8e5841d9364` |
| `failure_table.csv` | `0bcfef80d0e59ffc60c564409f4c32e40b31af9a5cda64bbacb15d1c0777d435` |
| `route_decision.json` | `4b6f40820759e686695e48b25af498928e954d41b7ec28acb68e1c2eb830ccf2` |

## 4. P0 Contract / FlashKAT Audit

P0 真实检查了 v6.6 lineage、FlashDWM2 torch-level coeffgrad variants、reset diagnostic。

FlashKAT audit：

| 项目 | 结果 |
|---|---|
| import | `ok` |
| class | `FlashKAT_Group` |
| Triton smoke | measured |
| smoke forward+backward time | `285.35 ms` |
| smoke peak allocated / reserved | `19.94 / 28.0 MB` |
| 是否作为 DWM2 结果 | 0 |

解释：`third_party/FlashKAT` 的 rational Triton 代码真实 import 并跑了 smoke backward；但它的数学形式和接口是 rational KAT autograd Function，不是 DWM2-poly2 manual backward，因此本轮只作为 source audit / diagnostic，不作为 DWM2 成功证据。

v6.6 reproduction：

| metric | v6.6 ref | v6.7 current | delta | pass |
|---|---:|---:|---:|---:|
| current memory ratio mean | `1.2917923089` | `1.2917923089` | `0.0` | 1 |
| current step ratio mean | `1.9079` | `1.9002321983` | `-0.0076678017` | 1 |

P0 结论：v6.7 baseline 与 v6.6 可比，新增 FlashDWM2 路径是真实 torch/manual code path；未实现的 Triton/CUDA fused package 没有被伪造成结果。

## 5. P1 CUDA Allocation / Attribution

P1 full grid 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x (MLP + 9 measured candidates) = 180 rows
```

### 5.1 current / repair / FlashDWM2 full-step 汇总

| variant | memory ratio vs MLP | step ratio vs MLP | backward ratio mean | grad relerr max | attribution pass |
|---|---:|---:|---:|---:|---:|
| `A2-DWM2-current` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.5965` / mean `1.9002` / max `2.2020` | `1.4751` | `7.30e-08` | `0/18` |
| `A3-DWM2-bufferReuse-v1` | min `1.1652` / mean `1.3470` / max `1.5851` | min `1.6792` / mean `1.9861` / max `2.3699` | `1.5755` | `2.30e-07` | `6/18` |
| `A4-DWM2-deltaStreaming-v1` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.5967` / mean `1.8906` / max `2.2457` | `1.4687` | `8.09e-08` | `0/18` |
| `A5-DWM2-bufferReuse+deltaStreaming` | min `1.1616` / mean `1.3392` / max `1.5723` | min `1.6610` / mean `1.9826` / max `2.3929` | `1.5702` | `2.30e-07` | `9/18` |
| `A6-FlashDWM2-local-reduce` | min `1.1898` / mean `1.3734` / max `1.6127` | min `2.1229` / mean `2.8169` / max `3.7581` | `3.3623` | `2.30e-07` | `18/18` |
| `A7-FlashDWM2-two-stage` | min `1.1898` / mean `1.3734` / max `1.6127` | min `2.1746` / mean `2.9473` / max `4.0893` | `3.6391` | `2.30e-07` | `18/18` |
| `A8-FlashDWM2-fused-dx` | min `1.1863` / mean `1.3655` / max `1.5999` | min `2.1411` / mean `2.8088` / max `3.7795` | `3.3390` | `2.30e-07` | `18/18` |

P1 判断：

- `A2-DWM2-current` attribution pass 是 `0/18`。这意味着 current bottleneck 仍未被 allocation stack / tensor lifetime 解释到计划阈值。
- FlashDWM2 variants 能记录 coeffgrad phase，但 memory/time 全部更差，不能反推 current 已被解释。
- Nsight counters 仍不可用，相关字段均写 `metric_unavailable`。

## 6. P2.5 FlashDWM2 Coeffgrad Audit

P2.5 覆盖：

```text
2 datasets x 2 batch sizes x 2 depths x 8 Flash audit variants = 64 rows
```

真实 measured variants：

| flash variant | time ratio vs current mean | memory ratio vs current mean | grad relerr max | pass count |
|---|---:|---:|---:|---:|
| `F0-current-coeffgrad` | `1.0000` | `1.0000` | `0.0` | 0 |
| `F1-local-reduce` | `2.8876` | `1.0000` | `1.12e-07` | 0 |
| `F2-two-stage-reduce` | `3.3559` | `0.9957` | `1.19e-07` | 0 |
| `F5-fused-dx-coeffgrad` | `5.0135` | `0.9929` | `1.12e-07` | 0 |
| `F7-chunked-batch-reduce` | `4.9621` | `1.0000` | `9.89e-08` | 0 |

未实现：

```text
F3-shared-memory-reduce
F4-warp-register-reduce
F6-fused-dx-coeffgrad-update
```

P2.5 判断：

- torch-level local/two-stage/chunked reduction 梯度正确，但没有达到 FlashDWM2 pass。
- memory ratio 只从 `1.0` 到最低约 `0.9929`，远不到 `<=0.70`。
- time 全部显著变慢，说明 Python/Torch chunking 不是 FlashKAT 论文级别的 kernel 重构。
- 本轮没有 `F-pass`，P3 不应把 Flash coeffgrad 作为 task path。

## 7. P3 Lower-Level Fused DWM2 Package

P3 full-step package 汇总：

| package | memory ratio vs MLP | step ratio vs MLP | memory improvement vs current | grad relerr max | near pass |
|---|---:|---:|---:|---:|---:|
| `D0-current` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.6770` / mean `2.1233` / max `3.0591` | `0.0` | `7.30e-08` | 0 |
| `D5-flash-coeffgrad-local-reduce` | min `1.1898` / mean `1.3734` / max `1.6127` | min `2.0953` / mean `3.0937` / max `4.1409` | `-6.23%` | `2.30e-07` | 0 |
| `D6-flash-coeffgrad-two-stage-reduce` | min `1.1898` / mean `1.3734` / max `1.6127` | min `2.1693` / mean `3.2716` / max `4.4145` | `-6.23%` | `2.30e-07` | 0 |
| `D7-fused-dx-flash-coeffgrad` | min `1.1863` / mean `1.3655` / max `1.5999` | min `2.1131` / mean `3.0973` / max `4.1395` | `-5.65%` | `2.30e-07` | 0 |

未实现 package：

```text
D1-torch-compile-current
D2-fused-transform
D3-fused-transform-derivative
D4-fused-delta-inputgrad
D8-fused-update-workspace
D9-full-fused-light
D10-full-fused-flash
D11-full-fused-onebuffer
```

P3 判断：

- 没有 S0/S1/S2 survivor。
- Flash coeffgrad full-step package 均比 current memory/time 更差。
- `survivor_type = S4`，因为 current attribution 未过且没有 near-pass package。

## 8. P4 Bounded-Workspace Reset

P4 reset measured variants：

| reset | memory ratio mean | step ratio mean | near pass count | residual pass count | residual_over_base mean |
|---|---:|---:|---:|---:|---:|
| `B0-ManualLinear-reference` | `1.1085` | `1.1578` | 0 | 0 | `0.0` |
| `B1-ManualLinear+TinyChannelResidual-edgeOwned` | `1.2981` | `1.8838` | 0 | 0 | `0.0010` |
| `B2-OneBufferPoly1Residual` | `1.2981` | `1.8808` | 0 | 0 | `0.0010` |
| `B9-FlashTinyResidual` | `1.3734` | `3.0384` | 0 | 0 | `0.0010` |

P4 判断：

- 本轮没有 reset near-pass。
- tiny residual 的 `residual_over_base mean = 0.0010`，低于计划要求 `>=0.02`。
- 因此 reset 不能打开 P5/P6，也不能作为 KAN success。

## 9. P5-P7 Gate 状态

| stage | status | reason |
|---|---|---|
| P5 one-step probe | `not_run` | No S0/S1/S2 DWM2 survivor and no reset near-pass with residual effect |
| P6 task re-entry | `not_run` | P5 did not pass or no S0/S1 candidate |
| P6 task trace | `not_run` | P6 is gated |
| P7 functional correction smoke | `not_run` | P7 is gated behind P6 |

这部分不能解读成 task 失败；准确说法是 v6.7 kernel/reset gate 没有过，所以 task 和 functional correction 正确关闭。

## 10. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F1_memory_fail` | 306 | measured full-step candidates memory ratio 仍高于 1 |
| `F11_gated_not_run` | 290 | 未实现、冻结或 gate 阻断 |
| `F2_step_time_fail` | 271 | measured candidates step ratio 超过 1.35 |
| `F4_attribution_incomplete` | 57 | current / 部分 DWM2 attribution 未过 |
| `F14_coeffgrad_reduction_no_effect` | 40 | Flash coeffgrad micro-kernel 未带来有效 memory/time 改善 |

没有出现：

- fake data failure
- proxy row failure
- gradient correctness failure

## 11. Route Decision

`route_decision.json`：

```json
{
  "route": "R3-AttributionIncomplete",
  "best_candidate": "D0-current",
  "best_family": "DWM2-poly2",
  "best_memory_ratio": 1.2917923088533285,
  "best_step_ratio": 2.1233236247008684,
  "best_backward_ratio": 1.6098761492659395,
  "memory_improvement_vs_current": 0.0,
  "step_improvement_vs_current": 0.0,
  "survivor_type": "S4",
  "attribution_pass": 0,
  "coeffgrad_bottleneck_pass": 0,
  "flash_coeffgrad_pass": 0,
  "flash_coeffgrad_variant": "",
  "fallback_triggered": true,
  "fallback_near_pass_count": 0,
  "residual_effect_pass": 0,
  "open_task_reentry": false,
  "open_functional_correction": false,
  "no_fake": true,
  "no_proxy": true,
  "primary_blocker": "CUDA allocation attribution did not explain peak gap to plan threshold",
  "next_required_implementation": "nsight_or_triton_full_backward_kernel"
}
```

## 12. 结论

本轮 v6.7 支持以下真实结论：

1. v6.7 与 v6.6 current baseline 复现通过，memory ratio mean 完全一致。
2. FlashKAT 代码真实 import 并执行了 Triton smoke backward，但没有被混入 DWM2 结果。
3. torch-level FlashDWM2 local/two-stage/fused-dx coeffgrad 梯度正确，但 memory 基本不降、time 明显变慢。
4. full-step FlashDWM2 packages 全部比 current 更差，没有 S0/S1/S2 survivor。
5. bounded-workspace reset 没有 near-pass，tiny residual effect 也未达非平凡 gate。
6. `A2-DWM2-current` attribution pass 仍是 `0/18`，最终 route 是 `R3-AttributionIncomplete`。

最终一句话：

> v6.7 没有证明 FlashDWM2 的 torch-level coeffgrad 重构能修复 DWM2-poly2；当前正确下一步不是开 task，而是获取 Nsight/allocator stack 级 trace，并实现真正的 Triton/CUDA full backward kernel，而不是 Python/Torch chunked reduction。

## 13. 下一步建议

1. 给 `A2-DWM2-current` 增加真实 Nsight Systems / Nsight Compute trace，优先拿到 allocation stack、global memory traffic、stall reason。
2. 如果继续 FlashDWM2，必须实现 Triton/CUDA coeffgrad kernel，而不是 Python chunk loop。
3. full backward 应优先融合 `dz -> dx + coeffgrad + update/lifetime`，单独 coeffgrad torch 重写没有改变 peak。
4. reset primitive 需要先把 `residual_over_base` 从 `0.001` 提到 `>=0.02`，同时保持 memory/time gate，否则只是弱 residual diagnostic。
5. P5/P6/P7 继续关闭，直到出现真实 S0/S1/S2 或 reset near-pass + residual-effect pass。
