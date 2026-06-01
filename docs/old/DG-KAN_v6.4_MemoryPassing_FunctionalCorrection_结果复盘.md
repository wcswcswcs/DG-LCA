# DG-KAN v6.4 Memory-Passing Functional Correction 结果复盘

> 本复盘只记录 2026-05-04 执行的 v6.4 real-only run。v6.4 旧主脚本 `experiments/run_gafu_v64.py` 已封禁，因为它历史上混入 proxy/derived rows；本轮使用新入口 `experiments/run_gafu_v64_real.py`，未通过真实 gate 的阶段全部保持 `not_run`。

## 1. 实验溯源

### 1.1 运行命令

```bash
python experiments/run_gafu_v64_real.py \
  --packages V6_4_ALL \
  --out-dir results/real_rerun_20260504/v64_real_all_20260504T205213Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v64-real-20260504 \
  --wandb-name-prefix v64-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260504/v64_real_all_20260504T205213Z`
- 本地主日志：`results/real_rerun_20260504/v64_real_all_20260504T205213Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/3aouu69o>
- W&B run name：`v64-real-v64_real_all_20260504T205213Z`

### 1.3 本轮 no-fake 约束

- `experiments/dgkan_core.py` 已禁用 fake vision bundle。
- v6.4 real runner 没有 `--allow-fake-data` 参数。
- `experiments/run_gafu_v64.py` 保持 hard-fail stub，防止旧 proxy generator 被误用。
- `experiments/run_gafu_v64_real_p2.py` 已修正为 diagnostic-only：不再写 `step_ratio=1.0` / `bmem_ratio=1.0` 这类占位数。
- 本轮 P2/P3/P4/P5/P6/P7/P8 没有通过 gate，因此写为 `not_run`，没有把未实现内容包装成失败数值或通过数值。

### 1.4 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_4_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| seeds | P0/P1 使用 `seed=0` profiler；P2 计划 seeds 为 `0,1,2`，但被 P1 gate 阻断 |
| device | `auto`，本机解析为 CUDA |
| train/val/test size | `1536/512/512` |
| batch size | task default `128` |
| P1 batch sizes | `128,256,512` |
| P1 depths | `2,4` |
| hidden dim | `64` |
| P1 warmup/measure | `50/200` steps |
| P2 budget | `240` steps，约等价 20 epochs，但本轮未进入 |

## 2. 产物清单

| artifact | 状态 |
|---|---|
| `p0_contract.csv` | 真实生成 |
| `p0_contract_heatmap.svg` | 真实生成 |
| `p1_memory_finalization.csv` | 真实生成 |
| `p1_memory_decomposition.csv` | 真实生成 |
| `p1_phase_local_memory.csv` | 真实生成 |
| `p1_runtime_decomposition.csv` | 真实生成 |
| `p1_gradient_correctness.csv` | 真实生成 |
| `p2_task_recipe.csv` | `not_run` |
| `p2_task_trace.csv` | `not_run` |
| `p3_functional_correction_direction.csv` | `not_run` |
| `p3_one_step_probe.csv` | `not_run` |
| `p3_direction_gate_summary.csv` | `not_run` |
| `p4_acceleration_stability.csv` | `not_run` |
| `p4_seedwise_trace.csv` | `not_run` |
| `p5_geometry_integration.csv` | `not_run` |
| `p5_event_trace.csv` | `not_run` |
| `p6_confirm3.csv` | `not_run` |
| `p7_confirm5.csv` | `not_run` |
| `p8_confirm10.csv` | `not_run` |
| `failure_table.csv` | 真实生成 |
| `route_decision.json` | 真实生成 |
| `aggregate_decision.json` | 真实生成 |

关键 hash：

| 文件 | SHA256 |
|---|---|
| `run.log` | `7e5aed7f3d1004920b58a71b97d7d765264ee2d3b148ba27d5f9451f7ec21ec6` |
| `p1_memory_finalization.csv` | `44b256169ae8abd82a9bbf97844aca5f3b43f5ae6bf0f1bee32d476bbb3006b2` |
| `failure_table.csv` | `76eba5edc312f28b0f16c4eaebf3c5b03cf56202e977dcb5a2b3b03b4f29b6e9` |
| `route_decision.json` | `93dc9713f1c2e6c34fe043f4bda77ab79cd6872b95c4c6f34a53d40907f2ca5c` |

## 3. P0 Implementation Contract

P0 真实检查了可执行候选和未实现 hook：

| method | status | nonKAN | manual backward | fake data |
|---|---|---:|---:|---:|
| `MLP-autograd-reference` | measured reference | 55050 | 0 | 0 |
| `MLP-manual-linear-reference` | measured | 0 | 1 | 0 |
| `DWM2-poly2-compiled-current` | measured | 0 | 1 | 0 |
| `DWM2-poly2-compiled-memoryOptimized` | not_implemented | - | - | 0 |
| `DWM2-poly2-compiled+LightSmooth-hook` | not_implemented | - | - | 0 |
| `DWM2-poly2-compiled+FunctionalCorrection-hook` | not_implemented | - | - | 0 |
| `DWM2-poly3-reference` | measured | 0 | 1 | 0 |
| `DWM2-poly2-silu-base-reference` | measured | 0 | 1 | 0 |

结论：

- 当前 `DWM2-poly2-compiled-current` 仍满足 graph-free/manual backward/manual update 基本 contract。
- v6.4 计划中的 memory optimized、LightSmooth hook、FunctionalCorrection hook 尚未实现，不能进入结果结论。

## 4. P1 Memory Finalization

P1 是本轮硬门禁。按计划，只有 memory 和 step time 同时过线，才允许进入 P2 task recipe。

本轮真实测量覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x 2 measured variants = 36 DWM2 rows
```

测量对象：

- `P1-M0-current`：当前 `DWM2-poly2-compiled`
- `P1-M2-noHiddenCache`：本轮新增真实实现，不保存 hidden cache，backward 从 input 重算 hidden

### 4.1 汇总结果

| variant | backward_memory_ratio_vs_MLP | step_time_ratio_vs_MLP | backward_time_ratio_vs_MLP | grad_relerr | grad_cos | survivor |
|---|---:|---:|---:|---:|---:|---:|
| `P1-M0-current` | min 1.144 / mean 1.292 / max 1.487 | min 1.613 / mean 1.936 / max 2.284 | min 1.200 / mean 1.493 / max 1.845 | 8.38e-08 | 1.0 | 0 |
| `P1-M2-noHiddenCache` | min 1.142 / mean 1.284 / max 1.480 | min 1.724 / mean 2.142 / max 2.599 | min 1.423 / mean 1.927 / max 2.513 | 8.12e-08 | 0.99999988 | 0 |

### 4.2 Cache 分解

| variant | manual_cache_MB | cache_hidden_MB | 观察 |
|---|---:|---:|---|
| `P1-M0-current` | min 0.414 / mean 1.039 / max 1.906 | min 0.031 / mean 0.146 / max 0.375 | hidden cache 存在，但不是全部 memory 缺口 |
| `P1-M2-noHiddenCache` | min 0.383 / mean 0.893 / max 1.531 | 0.000 | hidden cache 被真实消除，但整体 memory ratio 仍 > 1 |

### 4.3 P1 判断

P1 hard pass 要求：

```text
backward_memory_ratio_vs_MLP < 1.0
step_time_ratio_vs_MLP <= 1.10
grad_relerr < 1e-4
grad_cos > 0.999
```

实际情况：

- gradient correctness 通过。
- memory 没有过：最佳 memory ratio 仍为 `1.1422798216`。
- step time 没有过：最佳 step ratio 仍为 `1.6129775842`。
- `noHiddenCache` 是真实实现，不是 proxy；它降低了 hidden cache，但因为 backward 重算增加开销，step time 明显变差。

P1 结论：

> v6.4 当前没有 P1 memory survivor，也没有 exploratory pass。`DWM2-poly2-compiled` 仍然 memory-blocked + time-blocked，不能进入 P2/P4/P6。

## 5. P2-P8 Gate 状态

| stage | status | reason |
|---|---|---|
| P2 task recipe | `not_run` | P1 produced no memory survivor |
| P3 functional correction direction | `not_run` | P2/P4 survivor not available |
| P4 acceleration stability | `not_run` | P2 did not produce official memory-passing survivor |
| P5 geometry integration | `not_run` | requires P1 memory pass and P4 task-stable survivor |
| P6 confirm3 | `not_run` | gated behind P5 |
| P7 confirm5 | `not_run` | gated behind P6 |
| P8 confirm10 | `not_run` | gated behind P7 |

这部分不能解读成“模型训练失败”；更准确地说，是 P1 efficiency gate 真实失败，所以后续训练/confirm 按计划被阻断。

## 6. Failure Table

`failure_table.csv` 的失败类型统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F1_memory_fail` | 36 | 两个真实 P1 variant 在所有 dataset/batch/depth 条件下 memory ratio 均未低于 1 |
| `F2_step_time_fail` | 36 | 两个真实 P1 variant 在所有条件下 step time ratio 均高于 gate |
| `F11_gated_not_run` | 13 | 未实现 variant 或 P2-P8 被正确 gate |

没有出现：

- `F3_gradient_correctness_fail`
- fake data failure
- artifact integrity failure

## 7. Route Decision

`route_decision.json`：

```json
{
  "route": "R3",
  "decision": "DWM2-poly2 task line remains memory-blocked; focus on memory kernel before confirm seeds.",
  "no_proxy": true
}
```

`aggregate_decision.json`：

```json
{
  "status": "running_or_gated",
  "p1_memory_survivor": 0,
  "p2_officially_gated": 1,
  "fake_data_used": 0,
  "proxy_rows_used_as_results": 0
}
```

## 8. 结论

本轮 v6.4 真实执行支持以下结论：

1. `DWM2-poly2-compiled` 的 manual gradient correctness 仍然可靠。
2. v6.4 的 P1 memory target 没有达成，当前最低 memory ratio 仍为 `1.1423`，没有低于 `1.0`。
3. `noHiddenCache` 真实消除了 hidden cache，但 memory ratio 只小幅下降，同时 step time 明显变差。
4. P2 及后续 task/correction/confirm 阶段没有运行，是按计划 gate，不是漏跑。
5. 本轮 route 是 `R3`：先继续 memory kernel，不应该消耗更多 task/confirm seed。

最终一句话：

> v6.4 目前没有得到 memory-pass graph-free PureKAN candidate。正确下一步不是继续跑 P2/P6，而是实现真正降低 total backward peak 的 memory kernel，例如 buffer reuse、delta streaming、bf16 cache 或 fused forward/backward cache policy，并重新跑 P1。

## 9. 下一步

优先级建议：

1. 实现 `P1-M4-bufferReuse`，目标是降低 workspace/phase-local peak，而不是只去掉 hidden cache。
2. 实现 `P1-M3-deltaStreaming`，避免 backward adjoint 阶段保留不必要 delta。
3. 实现 `P1-M1-cacheCompressed-bf16`，但必须重新做 gradient correctness check。
4. 重新定义 P1 phase-local memory instrumentation，区分 actual CUDA peak、manual cache estimate、workspace temp。
5. 只有当 P1 出现 `p1_memory_survivor=1`，才重新打开 P2/P4/P6。

