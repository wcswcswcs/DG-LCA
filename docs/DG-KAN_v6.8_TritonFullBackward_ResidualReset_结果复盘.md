# DG-KAN v6.8 TritonFullBackward ResidualReset 结果复盘

> 本复盘只记录 `results/real_rerun_20260504/v68_real_all_20260504T235743Z` 下的 final real-only run。运行启动于 2026-05-04 23:57 UTC，完成于 2026-05-05 00:01 UTC。所有结论来自该目录落盘的 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v68_real.py \
  --packages V6_8_ALL \
  --out-dir results/real_rerun_20260504/v68_real_all_20260504T235743Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v68-real-20260504 \
  --wandb-name-prefix v68-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260504/v68_real_all_20260504T235743Z`
- 本地主日志：`results/real_rerun_20260504/v68_real_all_20260504T235743Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/7vr1f1b3>
- W&B run name：`v68-real-v68_real_all_20260504T235743Z`

### 1.3 不用于结论的检查

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `/tmp/v68_smoke` | 极小规模 smoke，检查 runner/Triton kernel/CSV shape | 0 |

### 1.4 no-fake / no-proxy 约束

- 新入口：`experiments/run_gafu_v68_real.py`。
- 所有 dataset loader 调用均使用 `allow_fake_data=False`。
- `experiments/dgkan_core.py` 中 fake data 路径仍保持 hard-fail。
- P1/P2/P3/P4 的 `fake_data_used`、`proxy_row_used`、`proxy_rows_used` 总和均为 `0`。
- Nsight / lower-level counters 不可用时写为 `metric_unavailable`，没有填假数。
- 未实现的 Triton/CUDA package 写为 `not_implemented` 或 `not_implemented_cuda_extension_absent`。
- P5/P6/P7/P8 因 gate 写为 `not_run`，没有包装成失败数值或成功数值。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_8_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| micro datasets | `Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler/kernel 使用 `seed=0`；无 task multi-seed confirm |
| full warmup / measure | 每个 full-step measured row `50/200` update steps |
| micro warmup / measure | 每个 Triton micro-kernel measured row `50/200` calls |
| trace batch/steps | `128 / 20` |
| run duration | `216.96` sec |
| Triton | available |
| Nsight | `ncu_available=1`，`nsys_available=0`；counter 字段仍为 `metric_unavailable` |
| W&B logging | 记录 P0/P1/P2/P3/P4/P9/failure summary；P6 task 被 gate，未产生每 20 step loss trace |

本轮真实新增实现：

| variant | 实现说明 |
|---|---|
| `DWM2-Triton-coeffgrad-v1` | Triton coeffgrad local reduce microkernel |
| `DWM2-Triton-fused-dx-coeffgrad-v1` | Triton fused `dx + coeffgrad` microkernel |
| `DWM2-Triton-full-backward-v1` | full-step path 使用 Triton fused coeffgrad/dx，但 forward/其他阶段仍为 Torch/manual 组合 |
| `ResidualEffectiveTinyKAN-scale002` | poly1 residual reset，目标 residual/base 约 `0.02` |
| `ResidualEffectiveTinyKAN-scale005` | poly1 residual reset，目标 residual/base 约 `0.05` |

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 11 |
| `p0_reproduction_check.csv` | 真实生成 | 1 |
| `p1_nsight_allocation_trace.csv` | 真实生成 | 180 |
| `p1_tensor_lifetime_topk.csv` | 真实生成 | 30 |
| `p1_phase_peak_summary.csv` | 真实生成 | 36 |
| `p1_nsight_stall_summary.csv` | 真实生成 | 1 |
| `p2_triton_microkernel_audit.csv` | 真实生成 | 72 |
| `p2_triton_microkernel_correctness.csv` | 真实生成 | 24 |
| `p3_full_backward_packages.csv` | 真实生成 | 7 |
| `p3_full_backward_package_detail.csv` | 真实生成 | 144 |
| `p4_residual_effective_reset.csv` | 真实生成 | 180 |
| `p5_one_step_probe.csv` | `not_run` | 1 |
| `p6_task_reentry.csv` | `not_run` | 1 |
| `p6_task_trace.csv` | `not_run` | 1 |
| `p7_optimizer_exploration.csv` | `not_run` | 1 |
| `p8_functional_correction_smoke.csv` | `not_run` | 1 |
| `failure_table.csv` | 真实生成 | 709 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |

关键 hash：

| 文件 | SHA256 |
|---|---|
| `run.log` | `891133a8a2562c3998b033266b47894fd996da204d015237608035fb54058796` |
| `p1_phase_peak_summary.csv` | `cf0d01f17d972b384440b401d681475562fdfd3c872f4f938962a7a5df0826aa` |
| `p2_triton_microkernel_audit.csv` | `19cec5d258bf5916cd72190c1dfc3f15134d05681e4244666983a3f466e2ac8b` |
| `p3_full_backward_packages.csv` | `3a7f59378dbce718d41a39f6034fcb3f0ca7e87c79e0e381706d0c992907ba3e` |
| `p4_residual_effective_reset.csv` | `609bcc68ed1c134726b953afaac1dd2d69f54cdbe2d829bd2b1b52601396c651` |
| `failure_table.csv` | `f48cc5e0c2ffa4fbbc25f06f2e6a46c678d861d35848b2e68a0805b4f055ab84` |
| `route_decision.json` | `5e5f15753cd6aafd39dc0dae7d1793b449fc6f87369cb79f3e8a2cca52dee157` |

## 4. P0 Contract / v6.7 Reproduction

P0 contract：

| variant | status | manual backward | Triton | nonKAN | grad pass | coeff relerr |
|---|---|---:|---:|---:|---:|---:|
| `MLP-autograd-reference` | measured | 0 | 0 | 55050 | - | - |
| `MLP-manual-linear-reference` | measured | 1 | 0 | 0 | 1 | `3.93e-08` |
| `DWM2-current` | measured | 1 | 0 | 0 | 1 | `9.00e-08` |
| `DWM2-FlashTorch-local-reduce` | measured | 1 | 0 | 0 | 1 | `2.68e-07` |
| `DWM2-FlashTorch-two-stage` | measured | 1 | 0 | 0 | 1 | `2.68e-07` |
| `DWM2-FlashTorch-fused-dx` | measured | 1 | 0 | 0 | 1 | `2.73e-07` |
| `DWM2-Triton-coeffgrad-v1` | measured | 1 | 1 | 0 | 1 | `2.68e-07` |
| `DWM2-Triton-fused-dx-coeffgrad-v1` | measured | 1 | 1 | 0 | 1 | `2.68e-07` |
| `DWM2-Triton-full-backward-v1` | measured | 1 | 1 | 0 | 1 | `2.68e-07` |
| `ResidualEffectiveTinyKAN-scale002` | measured | 1 | 0 | 0 | 1 | `7.06e-08` |
| `ResidualEffectiveTinyKAN-scale005` | measured | 1 | 0 | 0 | 1 | `7.36e-08` |

v6.7 reproduction check：

| metric | v6.7 ref | v6.8 current | delta | pass |
|---|---:|---:|---:|---:|
| current memory ratio mean | `1.2917923089` | `1.2917923089` | `0.0` | 1 |
| current step ratio mean | `1.9002321983` | `1.9697941070` | `0.0695619087` | 1 |

P0 结论：v6.8 baseline 与 v6.7 memory baseline 可比；Triton paths 是真实 kernel/code path，不是 proxy row。

## 5. P1 Nsight / Allocation Attribution

P1 phase summary 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x (MLP + current DWM2) = 36 rows
```

P1 allocation trace 覆盖：

```text
18 shapes x 2 variants x 5 phases = 180 rows
```

### 5.1 phase peak summary

| variant | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | grad relerr max | attribution pass |
|---|---:|---:|---:|---:|---:|
| `A0-MLP-autograd-reference` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | `1.0 / 1.0 / 1.0` | - | 0/18 |
| `A1-DWM2-current` | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.5986` / mean `1.9698` / max `2.3642` | min `1.2221` / mean `1.5417` / max `1.8528` | `9.00e-08` | 0/18 |

### 5.2 Nsight/counter 状态

`p1_nsight_stall_summary.csv` 记录：

| item | value |
|---|---|
| `ncu_available` | 1 |
| `nsys_available` | 0 |
| `sm_occupancy` / DRAM / L2 / stall counters | `metric_unavailable` |
| fake/proxy | 0/0 |

P1 判断：

- current DWM2 的 memory ratio mean 仍为 `1.2918`，没有比 v6.7 改善。
- `A1-DWM2-current` attribution pass 是 `0/18`。
- 低层 Nsight counter 仍未形成可用数值，不能把 current peak gap 解释到计划阈值。
- 因此最终 route 必须保守保持 `R3-AttributionIncomplete`，即使 P2 有局部 Triton microkernel pass。

## 6. P2 Triton Microkernel Audit

P2 覆盖：

```text
2 micro datasets x 2 batch sizes x 2 depths x 9 microkernel entries = 72 rows
```

真实 measured microkernel：

| microkernel | time ratio vs current | memory ratio vs current | grad relerr vs torch max | grad cos min | triton pass |
|---|---:|---:|---:|---:|---:|
| `K0-current-coeffgrad-torch` | `1.0000 / 1.0000 / 1.0000` | `1.0000 / 1.0000 / 1.0000` | `0.0` | `0.99999988` | 0/8 |
| `K1-triton-coeffgrad-local-reduce` | min `0.9184` / mean `0.9470` / max `0.9603` | `1.0000 / 1.0000 / 1.0000` | `9.45e-08` | `0.99999988` | 0/8 |
| `K3-triton-fused-dx-coeffgrad` | min `0.3018` / mean `0.3147` / max `0.3281` | min `0.9858` / mean `0.9910` / max `0.9962` | `9.45e-08` | `0.99999988` | 8/8 |

未实现/可选：

```text
K2-triton-coeffgrad-two-stage-reduce
K4-triton-fused-dx-coeffgrad-update-prep
K5-triton-transform-derivative
K6-triton-delta-dx-only
K7-cuda-extension-coeffgrad
K8-cuda-extension-full-backward
```

P2 判断：

- `K3-triton-fused-dx-coeffgrad` 是本轮唯一真实 pass：micro time 约为 current 的 `0.3147`，梯度正确。
- `K1` 也真实使用 Triton，但只带来约 `5.3%` coeffgrad time 改善，未达到 pass gate。
- P2 pass 只说明 microkernel 局部有效，不能自动推导 full-step memory/time 过关。

## 7. P3 Full Backward Packages

P3 full-step detail 覆盖：

```text
18 shapes x (MLP + 4 measured packages + 3 not-implemented packages) = 144 rows
```

Package 汇总：

| package | status | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | memory improvement vs current | grad relerr max | near pass |
|---|---|---:|---:|---:|---:|---:|---:|
| `D0-current` | measured | min `1.1441` / mean `1.2918` / max `1.4866` | min `1.7222` / mean `2.1140` / max `2.7430` | min `1.3251` / mean `1.7092` / max `2.6572` | `0.0` | `9.00e-08` | 0/18 |
| `D1-triton-coeffgrad-only` | measured | min `1.2091` / mean `1.4429` / max `1.7417` | min `2.1061` / mean `2.6530` / max `3.5264` | min `2.0853` / mean `2.7913` / max `4.4724` | `-11.28%` | `2.68e-07` | 0/18 |
| `D2-triton-fused-dx-coeffgrad` | measured | min `1.1654` / mean `1.3472` / max `1.5852` | min `2.0086` / mean `2.5495` / max `3.3379` | min `1.9299` / mean `2.5661` / max `4.0260` | `-4.18%` | `2.68e-07` | 0/18 |
| `D4-triton-full-backward-light` | measured | min `1.1654` / mean `1.3472` / max `1.5852` | min `2.0418` / mean `2.5535` / max `3.3309` | min `1.9389` / mean `2.5666` / max `4.0131` | `-4.18%` | `2.68e-07` | 0/18 |

未实现：

```text
D3-triton-transform-derivative+coeffgrad
D5-triton-full-backward-onebuffer
D6-cuda-full-backward
```

P3 判断：

- Triton microkernel 的速度收益没有转化成 full-step 收益。
- `D1/D2/D4` 的 memory 和 time 都比 `D0-current` 更差。
- 没有 S0/S1/S2 full-step survivor；`survivor_type = S6`。

## 8. P4 Residual-Effective Reset

P4 覆盖：

```text
18 shapes x 10 reset entries = 180 rows
```

真实 measured reset：

| reset | memory ratio vs MLP | step ratio vs MLP | residual/base | residual effect pass | reset near pass | grad relerr max |
|---|---:|---:|---:|---:|---:|---:|
| `B0-ManualLinear-reference` | min `1.0652` / mean `1.1085` / max `1.1718` | min `1.1045` / mean `1.1849` / max `1.3568` | `0.0` | 0/18 | 0/18 | `3.93e-08` |
| `B1-TinyResidual-scale002` | min `1.1422` / mean `1.2981` / max `1.5058` | min `1.6512` / mean `1.9014` / max `2.2590` | mean `0.0200` | 18/18 | 0/18 | `7.06e-08` |
| `B2-TinyResidual-scale005` | min `1.1422` / mean `1.2981` / max `1.5058` | min `1.6480` / mean `1.9021` / max `2.2818` | mean `0.0500` | 18/18 | 0/18 | `7.36e-08` |
| `B3-OneBufferPoly1Residual-scale002` | min `1.1422` / mean `1.2981` / max `1.5058` | min `1.6299` / mean `1.9462` / max `2.8480` | mean `0.0200` | 18/18 | 0/18 | `7.06e-08` |
| `B4-OneBufferPoly1Residual-scale005` | min `1.1422` / mean `1.2981` / max `1.5058` | min `1.6454` / mean `1.9001` / max `2.2609` | mean `0.0500` | 18/18 | 0/18 | `7.36e-08` |
| `B8-ResidualOnlyAblationProbe` | min `1.1422` / mean `1.2981` / max `1.5058` | min `1.6457` / mean `1.9019` / max `2.2583` | mean `0.0500` | 18/18 | 0/18 | `7.36e-08` |

未实现：

```text
B5-OneBufferPiecewiseLinear2-scale002
B6-OneBufferFastRational-scale002
B7-ChunkedMixingResidualKAN
```

P4 判断：

- v6.8 解决了 v6.7 中 residual 太弱的问题：`scale002/scale005` 的 residual/base 分别真实达到约 `0.02/0.05`。
- 但所有 residual-effective reset 的 memory mean 约 `1.2981`、step mean 约 `1.90`，没有 near-pass。
- ManualLinear reference 仍不是 KAN success；它没有 residual effect，不能打开 task。

## 9. P5-P8 Gate 状态

| stage | status | reason |
|---|---|---|
| P5 one-step probe | `not_run` | No S0/S1/S2 full-step package and no reset near-pass |
| P6 task re-entry | `not_run` | P5 gated |
| P6 task trace | `not_run` | P6 gated |
| P7 optimizer exploration | `not_run` | P6 gated |
| P8 functional correction smoke | `not_run` | P7 gated |

这部分不能解读成 task 训练失败；准确说法是 v6.8 kernel/reset gate 没有过，所以按计划没有进入任务训练。W&B 中因此没有每 20 step loss 曲线，只有真实 kernel/profiler/route/failure 指标。

## 10. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F1_memory_fail` | 198 | measured full-step/reset candidates memory ratio 未低于 gate |
| `F2_step_time_fail` | 181 | measured candidates step ratio 超过 gate |
| `F14_gated_not_run` | 160 | 未实现、CUDA extension 缺失或后续阶段被 gate |
| `F9_reset_no_near_pass` | 108 | residual/reset candidates 没有 near-pass |
| `F4_attribution_incomplete` | 36 | P1 current attribution 未过 |
| `F8_residual_effect_fail` | 18 | ManualLinear reference 无 residual effect |
| `F6_triton_kernel_no_effect` | 8 | Triton local reduce 未达到 microkernel pass |

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
  "best_step_ratio": 2.1139623910972194,
  "best_backward_ratio": 1.7092446582928233,
  "memory_improvement_vs_current": 0.0,
  "step_improvement_vs_current": 0.0,
  "survivor_type": "S6",
  "attribution_pass": 0,
  "triton_kernel_pass": 1,
  "reset_residual_effect_pass": 0,
  "fallback_triggered": true,
  "fallback_near_pass_count": 0,
  "open_task_reentry": false,
  "open_optimizer_exploration": false,
  "open_functional_correction": false,
  "no_fake": true,
  "no_proxy": true,
  "primary_blocker": "Nsight/allocator trace still did not explain current DWM2 peak gap to threshold",
  "next_required_implementation": "nsight_trace_or_new_primitive_family"
}
```

## 12. 结论

本轮 v6.8 支持以下真实结论：

1. v6.8 与 v6.7 current memory baseline 复现通过，current DWM2 memory ratio mean 仍为 `1.2918`。
2. `K3-triton-fused-dx-coeffgrad` 是真实 Triton microkernel，梯度正确，并在 micro benchmark 中达到 time pass。
3. Triton microkernel 的局部收益没有转化成 full-step package 收益；`D1/D2/D4` full-step memory/time 均比 current 更差。
4. P1 current attribution 仍为 `0/18`，Nsight/counter 字段仍未提供可解释 peak gap 的可用数值。
5. residual-effective reset 的 residual/base 已真实达到 `0.02/0.05`，但 memory/time 没有 near-pass。
6. P5-P8 正确 gate，没有任务训练、optimizer exploration 或 functional correction 结论。

最终一句话：

> v6.8 证明了一个 Triton fused `dx+coeffgrad` microkernel 可以在局部加速且保持梯度正确，但它没有解决 DWM2 full-step memory/time，也没有形成 reset near-pass；当前路线仍卡在 attribution incomplete + full backward package ineffective，不能开启 task。

## 13. 下一步建议

1. 真正拿到 Nsight Systems allocation timeline 或 PyTorch allocator stack trace，把 `A1-DWM2-current` 的 peak gap 定位到具体 op/tensor lifetime。
2. 不要只替换 coeffgrad；full backward 必须融合 forward transform、derivative、delta、dx、coeffgrad 和 update lifetime，目标是减少 full-step peak，而不是只加速 microkernel。
3. 如果继续 Triton，优先实现 `D5-triton-full-backward-onebuffer`，并用 `p3_full_backward_package_detail.csv` 的 full-step gate 判断。
4. residual reset 方向需要同时满足 residual effect 和 near-pass；当前 scale 达标但 memory/time 不达标。
5. P6/P7/P8 继续关闭，直到出现真实 S0/S1/S2 full-step survivor 或 reset near-pass。
