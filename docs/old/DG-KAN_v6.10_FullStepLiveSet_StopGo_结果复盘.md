# DG-KAN v6.10 FullStepLiveSet StopGo 结果复盘

> 本复盘只记录 `results/real_rerun_20260505/v610_real_all_20260505T045309Z` 下的 final real-only run。运行启动于 2026-05-05 04:53 UTC，完成于 2026-05-05 04:55 UTC。所有结论来自该目录落盘的 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v610_real.py \
  --packages V6_10_ALL \
  --out-dir results/real_rerun_20260505/v610_real_all_20260505T045309Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v610-real-20260505 \
  --wandb-name-prefix v610-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260505/v610_real_all_20260505T045309Z`
- 本地主日志：`results/real_rerun_20260505/v610_real_all_20260505T045309Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/yx9bfl2y>
- W&B run name：`v610-real-v610_real_all_20260505T045309Z`

### 1.3 不用于结论的检查

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `/tmp/v610_smoke` | 极小规模 smoke，检查 runner/no-fake/snapshot/CSV shape | 0 |

### 1.4 no-fake / no-proxy 约束

- 新入口：`experiments/run_gafu_v610_real.py`。
- 所有 dataset loader 调用均使用 `allow_fake_data=False`。
- `experiments/dgkan_core.py` 中 fake data 路径仍保持 hard-fail。
- P1/P2/P3/P4/P5/P6 的 `fake_data_used` 与 `proxy_row_used/proxy_rows_used` 总和均为 `0`。
- P1 使用 `torch.cuda.memory._record_memory_history/_snapshot` 真实采集 allocator trace；未把 phase diagnostic 伪装成 exact attribution pass。
- 未实现 single-call full-layer、multi-layer chunked、true one-buffer full-step、reset fused/chunked/lowrank/piecewise variants 均写为 `not_implemented`。
- P7/P8/P9/P10 因 gate 写为 `not_run`，没有包装成失败数值或成功数值。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_10_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| micro datasets | `Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler/kernel 使用 `seed=0`；无 task multi-seed confirm |
| full warmup / measure | 每个 full-step measured row `50/200` update steps |
| boundary warmup / measure | 每个 boundary row `50/200` calls |
| run duration | `154.36` sec |
| Triton | available |
| memory history | available |
| Nsight | `ncu_available=1`，`nsys_available=0` |
| W&B logging | 记录 P0-P6/P11/failure summary；P8 task 被 gate，未产生每 20 step loss trace |

本轮真实新增实现：

| item | 实现说明 |
|---|---|
| P1 exact live-set snapshot | 使用 PyTorch CUDA memory history trace 重建 peak active allocation set |
| top-20 live allocation stack | 记录 allocation stack hash、source file/op、source class、lifetime start/peak time |
| Stop/Go route | 明确输出 `STOP_DWM2_PATCHING` 或 gated go decision |

未实现但按计划显式记录：

```text
single-call no-dx-materialize full-layer
single-call update-prep full-layer
multi-layer chunked backward
true one-buffer full-step
bounded reset fused/chunked/lowrank/piecewise/streaming variants
```

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 11 |
| `p0_reproduction_check.csv` | 真实生成 | 1 |
| `p1_exact_live_set_attribution.csv` | 真实生成 | 36 |
| `p1_boundary_materialization.csv` | 真实生成 | 18 |
| `p2_boundary_audit.csv` | 真实生成 | 72 |
| `p3_full_layer_backward.csv` | 真实生成 | 7 |
| `p3_full_layer_backward_detail.csv` | 真实生成 | 144 |
| `p4_multilayer_chunked_backward.csv` | 真实生成 | 6 |
| `p4_multilayer_chunked_backward_detail.csv` | 真实生成 | 126 |
| `p5_onebuffer_full_step.csv` | 真实生成 | 5 |
| `p5_onebuffer_full_step_detail.csv` | 真实生成 | 108 |
| `p6_bounded_reset_v3.csv` | 真实生成 | 180 |
| `p7_one_step_probe.csv` | `not_run` | 1 |
| `p8_task_reentry.csv` | `not_run` | 1 |
| `p8_task_trace.csv` | `not_run` | 1 |
| `p9_optimizer_exploration.csv` | `not_run` | 1 |
| `p10_functional_correction_smoke.csv` | `not_run` | 1 |
| `failure_table.csv` | 真实生成 | 422 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |

关键 hash：

| 文件 | SHA256 |
|---|---|
| `run.log` | `656c9ae01854bcac664e0a6c7704b7130a8cfc4d5ad9842e8c4d645b622e7930` |
| `p1_exact_live_set_attribution.csv` | `0defeec04dfab248e40c3d41c95eb46d89cfff78cc9a0e381146e691d5dab27c` |
| `p2_boundary_audit.csv` | `9036df1752c82bf3513c9ff8b4a02bb90599600182294ffdabf7f34d86d16afd` |
| `p3_full_layer_backward.csv` | `3fd87354ed238dd947717eb211f8f201500949bb83309cc69394f3dffb4e5dd6` |
| `p4_multilayer_chunked_backward.csv` | `20904f813e6e705556240820756cfe14c370c0984db0aeb2ef40a2d2d5951748` |
| `p5_onebuffer_full_step.csv` | `5a8051cb8aae11639ef6e9ac5ee7e49185d4e19d11af699a83ae27b0f937701f` |
| `p6_bounded_reset_v3.csv` | `30e213d2bfa4052e9d25af69aff4722c4062503375130c516ca86796ec9df990` |
| `failure_table.csv` | `0b5e4a65c69e0604c91f233f01176ead5cf7a2a36902bbe00be77b3d3fac6d38` |
| `route_decision.json` | `3b5faa6ad50c48383e81fd8de53324baec90487b9f05fd55d1ecd629a54704c4` |

## 4. P0 Contract / v6.9 Reproduction

P0 contract：

| variant | status | Triton | nonKAN | manual backward | grad relerr |
|---|---|---:|---:|---:|---:|
| `MLP-autograd-reference` | measured | 0 | 55050 | 0 | - |
| `MLP-manual-linear-reference` | measured | 0 | 0 | 1 | `5.98e-08` |
| `DWM2-current` | measured | 0 | 0 | 1 | `1.52e-07` |
| `DWM2-Triton-fused-dx-coeffgrad-v1` | measured | 1 | 0 | 1 | `2.79e-07` |
| `DWM2-Triton-full-backward-light-v1` | measured | 1 | 0 | 1 | `2.79e-07` |
| `DWM2-FullLayerBackward-v3-no-dx-materialize` | `not_implemented` | 0 | - | - | - |
| `DWM2-FullLayerBackward-v3-update-prep` | `not_implemented` | 0 | - | - | - |
| `DWM2-MultiLayerChunkedBackward-v1` | `not_implemented` | 0 | - | - | - |
| `DWM2-OneBufferFullStep-v2` | `not_implemented` | 0 | - | - | - |
| `ResidualBoundedReset-v3-scale002` | measured | 0 | 0 | 1 | `9.88e-08` |
| `ResidualBoundedReset-v3-scale005` | measured | 0 | 0 | 1 | `1.00e-07` |

v6.9 reproduction：

| metric | v6.9 ref | v6.10 current | delta | pass |
|---|---:|---:|---:|---:|
| current memory ratio mean | `1.2917923089` | `1.2917923089` | `0.0` | 1 |
| current step ratio mean | `1.9041161522` | `1.8969413769` | `-0.0071747753` | 1 |

P0 结论：v6.10 与 v6.9 memory baseline 可比；真实实现项梯度正确，未实现项没有被写成假结果。

## 5. P1 Exact Peak Live-Set Attribution

P1 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x (MLP + current DWM2) = 36 rows
```

Current DWM2 summary：

| metric | min / mean / max |
|---|---:|
| memory ratio vs MLP | `1.1441 / 1.2918 / 1.4866` |
| step ratio vs MLP | `1.5718 / 1.8969 / 2.2610` |
| backward ratio vs MLP | `1.1248 / 1.4429 / 1.7389` |
| exact live tensor total MB | `3.7908 / 7.2872 / 11.7195` |
| peak gap MB | `2.5239 / 5.4430 / 9.2358` |
| allocation trace events | `452 / 660.17 / 868` |
| top3 gap fraction | `0.4332 / 0.4823 / 0.5058` |
| unknown gap fraction | `0.0 / 0.0 / 0.0` |
| attribution pass | `0/18` |

P1 判断：

- 本轮确实采集了 PyTorch CUDA memory history snapshot：current 18/18 行均为 `measured_memory_history`。
- allocation stack 可用：current 18/18 行均有 stack hash/source file/source op/source class。
- 但按 v6.10 gate，top-3 source 必须解释至少 `70%` peak gap；实际 mean 只有 `0.4823`，max 只有 `0.5058`。
- 因此 P1 仍不能算 attribution closed。准确结论是：比 v6.9 更真实地拿到了 allocator trace，但仍没有达到计划定义的 attribution pass。

## 6. P2 Boundary / Materialization Audit

P2 覆盖：

```text
2 micro datasets x 2 batch sizes x 2 depths x 9 boundary entries = 72 rows
```

Measured boundary rows：

| kernel | time ratio vs current | peak MB mean | boundary bottleneck pass | grad pass |
|---|---:|---:|---:|---:|
| `K0-current-torch-only-coeffgrad` | `1.0000 / 1.0000 / 1.0000` | `16.7988` | 0/8 | 8/8 |
| `K3-triton-fused-dx-coeffgrad-microkernel` | `0.0680 / 0.3110 / 0.3758` | `16.6426` | 0/8 | 8/8 |
| `K3-wrapper-only-no-op` | mean `0.0417` | `16.4854` | 0/8 | 8/8 |
| `K3-input-contiguous-only` | mean `0.0409` | `16.4854` | 0/8 | 8/8 |
| `K3-output-materialize-only` | mean `0.4433` | `16.6426` | 1/8 | 8/8 |
| `K3-layout-conversion-only` | mean `0.1305` | `16.6416` | 8/8 | 8/8 |
| `K3-call-overhead-only` | mean `0.1880` | `16.6416` | 8/8 | 8/8 |
| `K3-batched-layer-call` | mean `0.6575` | `16.7212` | 0/8 | 8/8 |
| `K3-single-call-multi-layer` | `not_implemented` | - | - | - |

P2 判断：

- `K3-triton-fused-dx-coeffgrad` 仍是有效 microkernel：梯度正确，mean time ratio `0.3110`。
- boundary/layout/call overhead diagnostic 仍有信号：`layout-conversion` 与 `call-overhead` 均为 8/8 bottleneck pass，`output-materialize` 为 1/8。
- 但 P2 仍只是 micro/boundary audit；不能替代 full-step pass。

## 7. P3 Single-Call Full-Layer Backward

P3 detail 覆盖：

```text
18 shapes x (MLP + current + 6 not-implemented single-call packages) = 144 rows
```

Package summary：

| package | status | memory ratio mean | step ratio mean | backward ratio mean | near pass |
|---|---|---:|---:|---:|---:|
| `L0-current` | measured | `1.2918` | `1.9693` | `1.5433` | 0 |
| `L3-full-layer-backward-no-dx-materialize` | `not_implemented` | - | - | - | - |
| `L4-full-layer-backward-update-prep` | `not_implemented` | - | - | - | - |
| `L4b-full-layer-backward-no-partial-integration` | `not_implemented` | - | - | - | - |
| `L4c-full-layer-backward-inplace-buffer` | `not_implemented` | - | - | - | - |
| `L5-multi-layer-chunked-backward` | `not_implemented` | - | - | - | - |
| `L6-single-call-depth2-backward` | `not_implemented` | - | - | - | - |

P3 判断：v6.10 没有新增真实 single-call full-layer implementation。不能声称 single-call 路线失败，只能说当前代码尚未实现计划要求的关键候选。

## 8. P4 Multi-Layer Chunked Backward

P4 detail 覆盖：

```text
18 shapes x (MLP + current + 5 not-implemented chunked packages) = 126 rows
```

| package | status | memory ratio mean | step ratio mean | backward ratio mean | near pass |
|---|---|---:|---:|---:|---:|
| `M0-current` | measured | `1.2918` | `2.0609` | `1.7780` | 0 |
| `M1-depth2-single-call-backward` | `not_implemented` | - | - | - | - |
| `M2-depth4-two-chunk-backward` | `not_implemented` | - | - | - | - |
| `M3-depth4-single-call-backward` | `not_implemented` | - | - | - | - |
| `M4-depth4-single-call-no-update` | `not_implemented` | - | - | - | - |
| `M5-depth4-single-call-update-prep` | `not_implemented` | - | - | - | - |

P4 判断：multi-layer chunked backward 没有真实候选，不能打开 task。

## 9. P5 True One-Buffer Full-Step

P5 detail 覆盖：

```text
18 shapes x (MLP + current + 4 not-implemented one-buffer packages) = 108 rows
```

| package | status | memory ratio mean | step ratio mean | backward ratio mean | onebuffer pass |
|---|---|---:|---:|---:|---:|
| `O0-current` | measured | `1.2918` | `2.0182` | `1.6497` | 0 |
| `O5-onebuffer-full-step` | `not_implemented` | - | - | - | - |
| `O6-onebuffer-full-step-chunked-mix` | `not_implemented` | - | - | - | - |
| `O7-onebuffer-full-step-update-prep` | `not_implemented` | - | - | - | - |
| `O8-onebuffer-full-step-lowrank-mix` | `not_implemented` | - | - | - | - |

P5 判断：true one-buffer full-step 仍未实现。当前只有 `O0-current` baseline，不能形成 S0/S1/S2。

## 10. P6 Bounded Reset v3

P6 覆盖：

```text
18 shapes x 10 reset entries = 180 rows
```

Measured reset：

| primitive | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | residual/base | residual pass | reset near pass |
|---|---:|---:|---:|---:|---:|---:|
| `R0-ManualLinear-reference` | `1.0652 / 1.1085 / 1.1718` | `1.0545 / 1.1544 / 1.2414` | mean `0.6773` | `0.0` | 0/18 | 0/18 |
| `R1-ResidualEffectiveTinyKAN-scale002-current` | `1.1422 / 1.2981 / 1.5058` | `1.5515 / 1.8162 / 2.0891` | mean `1.2576` | mean `0.0200` | 18/18 | 0/18 |
| `R2-ResidualEffectiveTinyKAN-scale005-current` | `1.1422 / 1.2981 / 1.5058` | `1.5479 / 1.8105 / 2.0832` | mean `1.2546` | mean `0.0500` | 18/18 | 0/18 |
| `R3-BoundedResidual-inplace-transform-v2` | `1.1422 / 1.2981 / 1.5058` | `1.5445 / 1.8086 / 2.1006` | mean `1.2519` | mean `0.0200` | 18/18 | 0/18 |

未实现 reset：

```text
R4-BoundedResidual-fused-mix
R5-BoundedResidual-chunked-mix
R6-BoundedResidual-lowrank-mix
R7-BoundedResidual-piecewise2
R8-BoundedResidual-streaming-backward
```

P6 判断：

- residual effect 继续真实达标：scale002/scale005 分别约 `0.02/0.05`。
- memory/time 仍没有 near-pass：reset mean memory 约 `1.2981`，mean step 约 `1.81`。
- ManualLinear reference 不是 KAN success，因为 residual effect 为 0。

## 11. P7-P10 Gate 状态

| stage | status | reason |
|---|---|---|
| P7 one-step probe | `not_run` | No S0/S1/S2 DWM2 survivor and no reset near-pass |
| P8 task re-entry | `not_run` | P7 did not pass or no memory/time survivor |
| P8 task trace | `not_run` | P8 is gated |
| P9 optimizer exploration | `not_run` | P8 task re-entry did not pass |
| P10 functional correction smoke | `not_run` | P10 is gated behind P8/P9 |

这部分不能解读成 task/optimizer/functional 失败；准确说法是 v6.10 Stop/Go gate 没有打开，所以按计划没有进入训练。W&B 中因此没有每 20 step task loss 曲线。

## 12. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F0_not_implemented_or_gated` | 115 | single-call/one-buffer/reset variants 未实现，或 P7/P8 被 gate |
| `F1_memory_fail` | 90 | measured DWM2/reset rows memory ratio 仍高于 gate |
| `F2_step_time_fail` | 72 | measured rows step ratio 超过 gate |
| `F10_reset_memory_fail` | 72 | residual reset memory/time 未 near-pass |
| `F4_attribution_incomplete` | 18 | current DWM2 exact attribution 未过 |
| `F6_live_set_overlap` | 18 | exact live-set 总量显示 peak live overlap 仍存在 |
| `F9_reset_residual_effect_fail` | 18 | ManualLinear reference residual effect 为 0 |
| `F5_boundary_overhead` | 17 | boundary/layout/call diagnostic 达到 bottleneck threshold |
| `F12_optimizer_gated` | 1 | optimizer exploration 正确 gate |
| `F13_functional_gated` | 1 | functional correction 正确 gate |

没有出现：

- fake data failure
- proxy row failure
- gradient correctness failure
- artifact missing failure

## 13. Route Decision

`route_decision.json`：

```json
{
  "route": "R3-AttributionIncomplete",
  "best_candidate": "L0-current",
  "best_family": "DWM2-poly2",
  "best_memory_ratio": 1.2917923088533285,
  "best_step_ratio": 1.9693036439136276,
  "best_backward_ratio": 1.5432731452376705,
  "memory_improvement_vs_current": 0.0,
  "step_improvement_vs_current": 0.0,
  "survivor_type": "S6",
  "attribution_pass": 0,
  "boundary_bottleneck_pass": 1,
  "onebuffer_pass": 0,
  "reset_residual_effect_pass": 1,
  "fallback_near_pass_count": 0,
  "open_one_step_probe": false,
  "open_task_reentry": false,
  "open_optimizer_exploration": false,
  "open_functional_correction": false,
  "primary_blocker": "exact peak live-set attribution did not close the DWM2 peak gap",
  "next_required_implementation": "real_single_call_full_step_or_new_primitive_family",
  "stop_go_decision": "STOP_DWM2_PATCHING",
  "no_fake": true,
  "no_proxy": true
}
```

## 14. 结论

本轮 v6.10 支持以下真实结论：

1. v6.10 current DWM2 memory baseline 与 v6.9 完全一致，结果可比较。
2. P1 已经真实采集 CUDA memory history snapshot，不再只是 phase diagnostic。
3. P1 仍未通过 attribution gate：top-3 source mean 只解释 `48.23%` peak gap，低于计划要求的 `70%`。
4. `K3-triton-fused-dx-coeffgrad` microkernel 继续有真实速度信号，mean time ratio `0.3110`，但 boundary/layout/call overhead 仍是 blocker 信号。
5. v6.10 计划中的 single-call full-layer、multi-layer chunked、true one-buffer full-step 关键候选仍未实现，因此不能开 task。
6. bounded reset v3 的 residual effect 达标，但 memory/time 没有 near-pass。
7. 最终 route 是 `R3-AttributionIncomplete`，Stop/Go 决策是 `STOP_DWM2_PATCHING`。

最终一句话：

> v6.10 真实推进了 peak live-set instrumentation，但仍没有让 DWM2-poly2 达到 attribution/full-step/one-buffer/reset 任一通路的 go 条件；当前不应该继续跑 task，而应停止 DWM2 小修，转向真实 single-call full-step kernel 或新的 bounded-workspace primitive family。

## 15. 下一步建议

1. 若继续 DWM2，优先实现真正 single-call full-step kernel；只替换 micro coeffgrad 已经不足。
2. exact live-set attribution 下一步要把 allocation stack 进一步映射到稳定 tensor/op taxonomy，让 top-3 source 能闭合到计划阈值。
3. true one-buffer 的验收必须落到 `O5/O6/O7/O8`，而不是继续用 current/buffer diagnostic。
4. reset 分支必须改变 workspace model；当前 residual scale 已不是主要问题。
5. P8 task、P9 optimizer、P10 functional correction 继续关闭，直到出现真实 S0/S1/S2 或 reset near-pass。
