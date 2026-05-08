# DG-KAN v6.9 FullStepLiveSetFusion BoundedReset 结果复盘

> 本复盘只记录 `results/real_rerun_20260505/v69_real_all_20260505T042909Z` 下的 final real-only run。运行启动于 2026-05-05 04:29 UTC，完成于 2026-05-05 04:32 UTC。所有结论来自该目录落盘的 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v69_real.py \
  --packages V6_9_ALL \
  --out-dir results/real_rerun_20260505/v69_real_all_20260505T042909Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v69-real-20260505 \
  --wandb-name-prefix v69-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260505/v69_real_all_20260505T042909Z`
- 本地主日志：`results/real_rerun_20260505/v69_real_all_20260505T042909Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/so7hk7sc>
- W&B run name：`v69-real-v69_real_all_20260505T042909Z`

### 1.3 不用于结论的检查

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `/tmp/v69_smoke` | 极小规模 smoke，检查 runner/Triton/no-fake/CSV shape | 0 |

### 1.4 no-fake / no-proxy 约束

- 新入口：`experiments/run_gafu_v69_real.py`。
- 所有 dataset loader 调用均使用 `allow_fake_data=False`。
- `experiments/dgkan_core.py` 中 fake data 路径仍保持 hard-fail。
- P1/P2/P3/P4/P5 的 `fake_data_used`、`proxy_row_used`、`proxy_rows_used` 总和均为 `0`。
- P1 没有把 phase-level 估计伪装成 peak-time allocation stack；由于 exact live tensor stack 不可用，`attribution_pass` 保守写为 `0`。
- 未实现的 single-call full-layer / true one-buffer / CUDA package 写为 `not_implemented`，没有写假 ratio。
- P6/P7/P8/P9 因 gate 写为 `not_run`，没有包装成失败数值或成功数值。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_9_ALL` |
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
| run duration | `208.81` sec |
| Triton | available |
| Nsight | `ncu_available=1`，`nsys_available=0`；exact live-set allocation stack 未取得 |
| W&B logging | 记录 P0-P5/P10/failure summary；P7 task 被 gate，未产生每 20 step loss trace |

本轮真实新增/复用实现：

| variant | 实现说明 |
|---|---|
| `DWM2-Triton-fused-dx-coeffgrad-v1` | 复用 v6.8 真实 Triton fused `dx+coeffgrad` kernel |
| `DWM2-Triton-full-backward-light-v1` | full-step path 使用 Triton fused coeffgrad/dx，但不是 true single-call layer kernel |
| `O1/O2/O4 one-buffer diagnostics` | 真实 buffer-reuse / delta-streaming code path，用于诊断，不等于 true one-buffer full-step |
| `ResidualBoundedOneBuffer-v2` | 真实 poly1 bounded reset diagnostic，residual scale 达标但 workspace model 仍未 near-pass |

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 10 |
| `p0_reproduction_check.csv` | 真实生成 | 1 |
| `p1_live_set_attribution.csv` | 真实生成 | 36 |
| `p1_boundary_overhead.csv` | 真实生成 | 18 |
| `p2_boundary_audit.csv` | 真实生成 | 72 |
| `p3_full_layer_fusion.csv` | 真实生成 | 7 |
| `p3_full_layer_fusion_detail.csv` | 真实生成 | 144 |
| `p4_onebuffer_full_step.csv` | 真实生成 | 7 |
| `p4_onebuffer_full_step_detail.csv` | 真实生成 | 144 |
| `p5_residual_bounded_reset.csv` | 真实生成 | 180 |
| `p6_one_step_probe.csv` | `not_run` | 1 |
| `p7_task_reentry.csv` | `not_run` | 1 |
| `p7_task_trace.csv` | `not_run` | 1 |
| `p8_optimizer_exploration.csv` | `not_run` | 1 |
| `p9_functional_correction_smoke.csv` | `not_run` | 1 |
| `failure_table.csv` | 真实生成 | 468 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |

注：final run 后只对 `failure_table.csv` taxonomy 做了标签修正，把 `not_implemented/not_run` 从误导性的 `F14_artifact_missing` 改为 `F0_not_implemented_or_gated`；所有 measured CSV 未改动。

关键 hash：

| 文件 | SHA256 |
|---|---|
| `run.log` | `88edef0baa3f78a3e0b7918411df650b819ea3ed5db31c6a76d74731335fb402` |
| `p1_live_set_attribution.csv` | `592cfb00588d0f2fee1ae59991ae8febf508efe617d9ab57e132d5c21329f0b4` |
| `p2_boundary_audit.csv` | `0081f2355b5a041bad08e1362cea38b33cc0580dbc6d800f0317f5101aec9732` |
| `p3_full_layer_fusion.csv` | `b71106c090afd95f3a1cb59ad5a8cc6bc524b0fe50628cd41fef65af430a9b03` |
| `p4_onebuffer_full_step.csv` | `ee5bcdc316bcfcb63137905c1acb4b0e4779b84e6871ad7ad95e7f9ed55fe75f` |
| `p5_residual_bounded_reset.csv` | `1e36314aa6d6868e806cf1d7091ae256b500fc4db9a7da1315d2fe0e9ad84484` |
| `failure_table.csv` | `8954cada9729c423b8c33a8407f809f0314fad3bdd7e8cd552c8d2c2e3bf848f` |
| `route_decision.json` | `c4921a1704958ebf4bfd2570de84ac69027e6b3a1b003259453e1761960d9c21` |

## 4. P0 Contract / v6.8 Reproduction

P0 contract：

| variant | status | Triton | nonKAN | grad pass | coeff relerr |
|---|---|---:|---:|---:|---:|
| `MLP-autograd-reference` | measured | 0 | 55050 | - | - |
| `MLP-manual-linear-reference` | measured | 0 | 0 | 1 | `3.93e-08` |
| `DWM2-current` | measured | 0 | 0 | 1 | `9.00e-08` |
| `DWM2-Triton-fused-dx-coeffgrad-v1` | measured | 1 | 0 | 1 | `2.68e-07` |
| `DWM2-Triton-full-backward-light-v1` | measured | 1 | 0 | 1 | `2.68e-07` |
| `DWM2-FullLayerBackward-v2` | `not_implemented` | 0 | - | - | - |
| `DWM2-OneBufferFullStep-v1` | `not_implemented` | 0 | - | - | - |
| `ResidualEffectiveTinyKAN-scale002` | measured | 0 | 0 | 1 | `7.06e-08` |
| `ResidualEffectiveTinyKAN-scale005` | measured | 0 | 0 | 1 | `7.36e-08` |
| `ResidualBoundedOneBuffer-v2` | measured | 0 | 0 | 1 | `7.06e-08` |

v6.8 reproduction check：

| metric | v6.8 ref | v6.9 current | delta | pass |
|---|---:|---:|---:|---:|
| current memory ratio mean | `1.2917923089` | `1.2917923089` | `0.0` | 1 |
| current step ratio mean | `1.9697941070` | `1.9041161522` | `-0.0656779548` | 1 |

P0 结论：v6.9 与 v6.8 memory baseline 可比；新增 runner 没有 fake/proxy，真实实现项梯度正确。

## 5. P1 Full-Step Live-Set Attribution

P1 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x (MLP + current DWM2) = 36 rows
```

### 5.1 summary

| variant | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | explain ratio | attribution pass | live-set overlap |
|---|---:|---:|---:|---:|---:|---:|
| `A0-MLP-autograd-reference` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | `1.0` diagnostic | 0/18 | 18/18 |
| `A1-DWM2-current` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.5833` / mean `1.9041` / max `2.2635` | min `1.1514` / mean `1.4447` / max `1.7511` | `1.0` diagnostic | 0/18 | 18/18 |

### 5.2 attribution 判断

- `explain_ratio` 来自 phase component diagnostic，不是 exact peak-time live tensor allocation stack。
- 本轮没有取得真实 peak timestamp / allocation stack / exact top-20 live tensor set。
- 因此即使 phase component sum 可以覆盖 gap，也不能按计划算 attribution pass。
- `attribution_pass` 保守写为 `0/18`。

P1 结论：v6.9 仍没有关闭 v6.8 的 attribution blocker。正确 route 必须保守，不允许把 phase-level diagnostic 说成 live-set attribution 成功。

## 6. P2 Triton/PyTorch Boundary Audit

P2 覆盖：

```text
2 micro datasets x 2 batch sizes x 2 depths x 9 boundary entries = 72 rows
```

Measured boundary rows：

| kernel | total call time ratio vs current | peak allocated MB | boundary bottleneck pass | grad pass |
|---|---:|---:|---:|---:|
| `K0-current-torch-only-coeffgrad` | `1.0 / 1.0 / 1.0` | mean `16.7988` | 0/8 | 8/8 |
| `K3-triton-fused-dx-coeffgrad-microkernel` | min `0.0797` / mean `0.5871` / max `2.5801` | mean `16.6426` | 0/8 | 8/8 |
| `K3-wrapper-only-no-op` | mean `0.0361` | mean `16.4854` | 0/8 | 8/8 |
| `K3-input-contiguous-only` | mean `0.0416` | mean `16.4854` | 0/8 | 8/8 |
| `K3-output-materialize-only` | mean `0.0720` | mean `16.6426` | 0/8 | 8/8 |
| `K3-layout-conversion-only` | mean `0.1328` | mean `16.6416` | 7/8 | 8/8 |
| `K3-call-overhead-only` | mean `0.1924` | mean `16.6416` | 7/8 | 8/8 |
| `K3-batched-layer-call` | mean `0.7082` | mean `16.7212` | 0/8 | 8/8 |
| `K3-single-call-multi-layer` | `not_implemented` | - | - | - |

P2 判断：

- K3 microkernel 仍是真实可执行且梯度正确，但本轮 boundary audit 的 time ratio 波动很大，max 达 `2.58`。
- launch-only 和 layout-conversion diagnostic 多数超过 boundary threshold，提示 boundary / layout 仍可能是 blocker。
- 但这不能单独证明 full-step 会变好；必须看 P3/P4。

## 7. P3 Full-Layer Fused Backward

P3 detail 覆盖：

```text
18 shapes x (MLP + 3 measured packages + 4 not-implemented packages) = 144 rows
```

Package 汇总：

| package | status | memory ratio mean | step ratio mean | backward ratio mean | memory improvement vs current | step improvement vs current | near pass | grad relerr max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `L0-current` | measured | `1.2918` | `1.8776` | `1.4261` | `0.0` | `0.0` | 0 | `9.00e-08` |
| `L1-fused-dx-coeffgrad-only` | measured | `1.3472` | `2.3699` | `2.3695` | `-4.18%` | `-25.85%` | 0 | `2.68e-07` |
| `L2-full-layer-backward` | measured | `1.3472` | `2.3577` | `2.3552` | `-4.18%` | `-25.20%` | 0 | `2.68e-07` |
| `L3-full-layer-backward-no-dx-materialize` | `not_implemented` | - | - | - | - | - | - | - |
| `L4-full-layer-backward-update-prep` | `not_implemented` | - | - | - | - | - | - | - |
| `L5-multi-layer-chunked-backward` | `not_implemented` | - | - | - | - | - | - | - |
| `L6-single-call-depth2-backward` | `not_implemented` | - | - | - | - | - | - | - |

P3 判断：

- measured full-layer diagnostics 没有把 K3 microkernel 收益转成 full-step 收益。
- `L1/L2` memory/time 都比 current 更差。
- 没有 S0/S1/S2 full-layer survivor。

## 8. P4 One-Buffer Full-Step

P4 detail 覆盖：

```text
18 shapes x (MLP + 4 measured packages + 3 not-implemented packages) = 144 rows
```

Package 汇总：

| package | status | memory ratio mean | step ratio mean | backward ratio mean | memory improvement vs current | step improvement vs current | near pass | live buffer count peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `O0-current` | measured | `1.2918` | `1.9551` | `1.5397` | `0.0` | `0.0` | 0 | `0` |
| `O1-onebuffer-forward-transform` | measured | `1.3470` | `2.0479` | `1.6525` | `-4.17%` | `-4.65%` | 0 | `31` |
| `O2-onebuffer-backward-delta` | measured | `1.2918` | `1.9489` | `1.5360` | `0.0` | `+0.29%` | 0 | `0` |
| `O4-onebuffer-forward-backward` | measured | `1.3392` | `2.0435` | `1.6438` | `-3.59%` | `-4.47%` | 0 | `31` |
| `O3-onebuffer-update-prep` | `not_implemented` | - | - | - | - | - | - | - |
| `O5-onebuffer-full-step` | `not_implemented` | - | - | - | - | - | - | - |
| `O6-onebuffer-full-step-chunked-mix` | `not_implemented` | - | - | - | - | - | - | - |

P4 判断：

- `O2` 只带来很小 step improvement，memory 没变。
- buffer-reuse variants 仍增加 memory peak；`live_buffer_count_peak=31`，没有接近 one-buffer gate。
- true one-buffer full-step 仍未实现，不能声称失败或成功。

## 9. P5 Residual-Effective Bounded Reset v2

P5 覆盖：

```text
18 shapes x 10 reset entries = 180 rows
```

Measured reset：

| primitive | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | residual/base | residual pass | reset near pass |
|---|---:|---:|---:|---:|---:|---:|
| `R0-ManualLinear-reference` | min `1.0652` / mean `1.1085` / max `1.1718` | min `1.0671` / mean `1.1511` / max `1.4198` | mean `0.6804` | `0.0` | 0/18 | 0/18 |
| `R1-ResidualEffectiveTinyKAN-scale002-current` | min `1.1422` / mean `1.2981` / max `1.5058` | min `1.5433` / mean `1.8014` / max `2.3600` | mean `1.2555` | mean `0.0200` | 18/18 | 0/18 |
| `R2-ResidualEffectiveTinyKAN-scale005-current` | min `1.1422` / mean `1.2981` / max `1.5058` | min `1.5472` / mean `1.7994` / max `2.3652` | mean `1.2541` | mean `0.0500` | 18/18 | 0/18 |
| `R3-BoundedResidual-inplace-transform` | min `1.1422` / mean `1.2981` / max `1.5058` | min `1.5472` / mean `1.7961` / max `2.3594` | mean `1.2499` | mean `0.0200` | 18/18 | 0/18 |
| `R6-BoundedResidual-single-buffer-backward` | min `1.1422` / mean `1.2981` / max `1.5058` | min `1.5473` / mean `1.7971` / max `2.4025` | mean `1.2526` | mean `0.0500` | 18/18 | 0/18 |

未实现：

```text
R4-BoundedResidual-fused-mix
R5-BoundedResidual-chunked-mix
R7-BoundedResidual-lowrank-mix
R8-BoundedResidual-piecewise2
```

P5 判断：

- residual effect 继续真实通过：`0.02/0.05` scale 均达标。
- reset memory/time 没有 near-pass，mean memory 仍约 `1.2981`，mean step 约 `1.80`。
- ManualLinear reference time/memory 较好但 residual effect 为 0，不能作为 KAN success。

## 10. P6-P9 Gate 状态

| stage | status | reason |
|---|---|---|
| P6 one-step probe | `not_run` | No S0/S1/S2 full-step DWM2 survivor and no reset near-pass |
| P7 task re-entry | `not_run` | P6 did not pass or no memory/time survivor |
| P7 task trace | `not_run` | P7 is gated |
| P8 optimizer exploration | `not_run` | P7 task re-entry did not pass |
| P9 functional correction smoke | `not_run` | P9 is gated behind P7/P8 |

这部分不能解读成 task/optimizer/functional 失败；准确说法是 kernel/reset gate 没有过，所以按计划没有进入这些阶段。W&B 中因此没有每 20 step loss 曲线。

## 11. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F1_memory_fail` | 108 | measured DWM2/reset rows memory ratio 仍高于 gate |
| `F2_step_time_fail` | 91 | measured rows step ratio 超过 gate |
| `F9_reset_memory_fail` | 90 | reset residual effect 有效但 memory/time 未 near-pass |
| `F0_not_implemented_or_gated` | 89 | true full-layer/one-buffer/reset variants 未实现或后续阶段被 gate |
| `F6_live_set_overlap` | 36 | phase diagnostics 指示 live-set overlap 可能存在 |
| `F4_attribution_incomplete` | 18 | current DWM2 exact live-set attribution 未过 |
| `F8_reset_residual_effect_fail` | 18 | ManualLinear reference residual effect 为 0 |
| `F5_boundary_overhead` | 14 | boundary/layout/launch diagnostic 达到 bottleneck threshold |
| `F7_full_fusion_no_effect` | 2 | measured full-layer fusion 没有带来有效 improvement |
| `F11_optimizer_gated` | 1 | optimizer exploration 正确 gate |
| `F12_functional_gated` | 1 | functional correction 正确 gate |

没有出现：

- fake data failure
- proxy row failure
- gradient correctness failure

## 12. Route Decision

`route_decision.json`：

```json
{
  "route": "R3-AttributionIncomplete",
  "best_candidate": "L0-current",
  "best_family": "DWM2-poly2",
  "best_memory_ratio": 1.2917923088533285,
  "best_step_ratio": 1.877616986741421,
  "best_backward_ratio": 1.426089351321692,
  "memory_improvement_vs_current": 0.0,
  "step_improvement_vs_current": 0.0,
  "survivor_type": "S6",
  "attribution_pass": 0,
  "boundary_bottleneck_pass": 1,
  "live_set_overlap_pass": 1,
  "reset_residual_effect_pass": 1,
  "open_task_reentry": false,
  "open_optimizer_exploration": false,
  "open_functional_correction": false,
  "primary_blocker": "live-set attribution still cannot explain current DWM2 peak to threshold",
  "next_required_implementation": "real_live_set_profiler_or_new_primitive_family",
  "no_fake": true,
  "no_proxy": true
}
```

## 13. 结论

本轮 v6.9 支持以下真实结论：

1. v6.9 current DWM2 memory baseline 与 v6.8 完全一致，结果可比较。
2. exact peak-time live tensor attribution 仍未取得，因此 route 保守保持 `R3-AttributionIncomplete`。
3. boundary audit 显示 launch/layout diagnostic 可能是 blocker，但不能单独证明 full-step 会变好。
4. measured full-layer fusion (`L1/L2`) 没有收益，memory/time 均比 current 更差。
5. measured one-buffer diagnostics 没有形成 true one-buffer；`O2` 只有轻微 step improvement，memory 没变。
6. residual reset 的 residual effect 真实达标，但 memory/time 仍没有 near-pass。
7. P6-P9 正确 gate，没有 task、optimizer、functional 结论。

最终一句话：

> v6.9 没有得到 DWM2 full-step live-set survivor，也没有得到 bounded reset near-pass；当前不能开 task，下一步必须拿到真实 peak-time live-set/allocator stack，或停止 DWM2 小修、进入新 primitive family 设计。

## 14. 下一步建议

1. 实现真正的 peak-time live tensor stack capture，而不是 phase component diagnostic。
2. 如果继续 DWM2，必须实现 `L3/L4/L5/L6` 级别的 true single-call full-layer/full-step kernel；当前 `L1/L2` 只是 partial integration。
3. true one-buffer 必须让 `live_buffer_count_peak <= 3`，当前 buffer-reuse diagnostic 的 peak count 仍为 `31`。
4. reset 分支要优先改 workspace model，而不是继续调 residual scale。
5. P7/P8/P9 继续关闭，直到出现真实 S0/S1/S2 或 reset near-pass。
