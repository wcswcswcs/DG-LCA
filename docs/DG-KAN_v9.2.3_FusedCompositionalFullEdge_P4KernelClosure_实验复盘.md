# DG-KAN v9.2.3 FusedCompositionalFullEdge P4KernelClosure 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.3_FusedCompositionalFullEdge_P4KernelClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P6/P7 未打开阶段写成通过。

## 0. 最新结论

本文件按执行顺序追加。前面第 1-13 节保留 first-wave P4 kernel closure 结果；第 14-17 节保留 Triton forward / streaming 结果；第 18-21 节保留完整 Triton backward pack 结果；最新结论以第 25 节为准。

截至第 25 节，v9.2.3 追加实现单层 monolithic Triton backward 后，仍执行到一个可审计 terminal route：

```text
route = R4-BackwardKernelBlocker
best_candidate = D2-FusedCompositional-T2
best_combined_candidate = C3-M7+B6+F3-no-input-dx-forward-yonly
success_v923_p4_kernel_closure = false
success_v923_trainability_reentry = false
success_v923_functional_opened = false
```

最新 artifact：

```text
results/real_rerun_20260506/v923_p4_kernel_closure_triton_monolithic_h512r4_20260509T150000Z/
```

Triton backward artifact：

```text
results/real_rerun_20260506/v923_p4_kernel_closure_triton_backward_h512r4_20260509T143000Z/
```

Triton/streaming artifact：

```text
results/real_rerun_20260506/v923_p4_kernel_closure_streaming_triton_h512r4_20260509T141000Z/
```

first-wave artifact：

```text
results/real_rerun_20260506/v923_p4_kernel_closure_h512r4_20260509T170000Z/
```

最新核心结论：

1. 已实现并测量单层 monolithic Triton backward `B9-triton-monolithic-layer-backward`，把每层的 `dM` reduction 和 `dx correction` 合进同一个 Triton kernel；layer1 仍只算 `dM`，因为 input dx 按合同移除。
2. B9 数值正确性过 gate：`GradRelErrMax = 6.8147972e-05`，`GradCosMin = 1.0`；但 runtime 仍失败：`backward_ratio = 13.413305`，`step_ratio = 4.007630`。
3. B9 比 B8 小 kernel pack 有改善，但仍远慢于 C3 no-dx/y-only；最新最佳 combined candidate 仍是 `C3-M7+B6+F3`。
4. C3 仍未关闭 P4：`backward_ratio = 2.048100`、`forward_ratio = 1.421522` 未过，`memory_ratio = 1.046065` 与 `step_ratio = 1.463772` 过线。
5. 因 combined P4 gate 仍未闭合，P6 AdamW-only trainability 未打开，P7 functional open 也不允许。

最终判断：

```text
memory live-set closure = pass under M7 compact accounting
backward time closure = fail
forward time closure = fail
Triton forward-only closure = fail
Triton full backward pack = fail
Triton monolithic backward = fail
combined P4 kernel-native gate = fail
P6 AdamW re-entry = not_run
P7 functional = not_opened
```

## 1. 本轮代码

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v923_p4_kernel_closure.py` | v9.2.3 P4 kernel closure runner；生成 P0-P7 required artifacts、route、failure table、no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v923_p4_kernel_closure.py
```

已通过。

## 2. 运行命令

```bash
python experiments/run_v923_p4_kernel_closure.py \
  --out-dir results/real_rerun_20260506/v923_p4_kernel_closure_h512r4_20260509T170000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --hidden-dim 512 \
  --rank 4 \
  --lr 0.0005 \
  --p4-batch-size 128 \
  --p4-warmup 5 \
  --p4-reps 20
```

设备：

```text
device = cuda
```

## 3. Route

`route_decision.json`：

```json
{
  "route": "R4-BackwardKernelBlocker",
  "best_candidate": "D2-FusedCompositional-T2",
  "best_memory_repair": "M7-no-input-dx-layer1",
  "best_backward_repair": "B6-no-dx-input",
  "best_forward_repair": "F3-no-materialized-basis-forward",
  "best_combined_candidate": "C3-M7+B6+F3",
  "grad_pass": 1,
  "synthetic_pairwise_R2": 0.9911209940910339,
  "forward_ratio": 1.4258657382667486,
  "backward_ratio": 2.0631232047002177,
  "step_ratio": 1.459990249275531,
  "memory_ratio": 1.0460654216619045,
  "p4_kernel_native_pass": 0,
  "p5_trainability_opened": 0,
  "p5_near_pass": 0,
  "functional_open_allowed": 0,
  "primary_blocker": "backward_ratio_remains_above_1.50_after_no_input_dx_repair",
  "next_required_implementation": "implement_custom_triton_or_cuda_backward_for_layer1_layer2"
}
```

判断：v9.2.3 没有关闭 P4。最直接的 terminal blocker 是 backward kernel；forward 也仍未过线，但 memory 已经被推进到 gate 内。

## 4. P0 repeat stability

P0 artifact：

```text
p0_route_recap_measurement_stability.csv
```

Repeat current fused D2：

| metric | value |
|---|---:|
| GradRelErrMax | `5.914181e-06` |
| GradCosMin | `0.999999940` |
| synthetic pairwise R2 | `0.9911209941` |
| forward ratio | `1.383334` |
| backward ratio | `2.061752` |
| memory ratio | `1.213611` |
| unknown time fraction | `0.0` |
| forward stable vs previous | `1` |
| backward stable vs previous | `1` |
| memory stable vs previous | `1` |

判断：P0 复现了当前 fused D2 的 P4 boundary；不是测量偶然。

## 5. P1 phase attribution

P1 artifact：

```text
p1_p4_failure_phase_attribution.csv
```

本轮 attribution 是 coarse compiled-phase attribution，未伪造 torch profiler kernel count：

```text
kernel_count_phase = torch_compile_not_decomposed
unknown_time_fraction = 0.0
dominant_phase = backward_kernel_and_memory_live_set
```

判断：当前 runner 能给出可审计 coarse blocker，但仍未做到 plan 中完整 layer-by-layer profiler decomposition。未测得的 phase fields 明确写为 `compiled_not_decomposed`，没有编造 kernel count。

## 6. P2 memory live-set closure

P2 artifact：

```text
p2_memory_live_set_closure.csv
```

关键 rows：

| candidate | memory ratio | backward ratio | forward ratio | GradPass | memory pass |
|---|---:|---:|---:|---:|---:|
| M0-current | `1.213611` | `2.061752` | `1.383334` | 1 | 0 |
| M7-no-input-dx-layer1 | `1.046065` | `2.063123` | `1.425866` | 1 | 1 |

M7 具体含义：

```text
computes_input_dx_layer1 = 0
recompute_layer1_basis = 1
recompute_layer2_basis = 1
stores_layer1_basis = 0
stores_layer2_basis = 0
```

判断：

1. M7 把 memory ratio 拉到 `1.046065`，刚好进入 `<=1.05` gate。
2. 这说明 v9.2.3 的 memory blocker 被推进，但不是整体 P4 success。
3. M1/M2/M3/M4/M5/M6/M8 未实现，已明确写 `not_implemented`，没有伪装成跑过。

## 7. P3 backward time closure

P3 artifact：

```text
p3_backward_time_closure.csv
```

B6-no-dx-input：

| metric | value |
|---|---:|
| GradRelErrMax | `5.676815e-05` |
| GradCosMin | `0.999999881` |
| synthetic pairwise R2 | `0.9911209941` |
| backward ratio | `2.063123` |
| step ratio | `1.459990` |
| memory ratio | `1.046065` |
| backward closure pass | `0` |

判断：

1. 不计算第一层 raw-input dx 没有破坏参数梯度 correctness。
2. 但 backward ratio 几乎没有改善，仍显著高于 `1.50`。
3. 因此 route 不是 MemoryLiveSetBlocker，而是 BackwardKernelBlocker。

## 8. P4 forward time closure

P4 artifact：

```text
p4_forward_time_closure.csv
```

F3-no-materialized-basis-forward：

| metric | value |
|---|---:|
| forward ratio | `1.425866` |
| step ratio | `1.459990` |
| memory ratio | `1.046065` |
| forward closure pass | `0` |

判断：forward-only/no-materialized-basis path 没有关闭 forward gate。Forward 仍需 source/basis/projection 的更低层 fusion。

## 9. P5 combined P4 gate

P5 artifact：

```text
p5_combined_p4_gate_closure.csv
```

Combined candidate：

```text
C3-M7+B6+F3
```

| metric | value | gate |
|---|---:|---|
| GradRelErrMax | `5.676815e-05` | pass |
| GradCosMin | `0.999999881` | pass |
| synthetic pairwise R2 | `0.9911209941` | pass |
| forward ratio | `1.425866` | fail |
| backward ratio | `2.063123` | fail |
| step ratio | `1.459990` | pass |
| memory ratio | `1.046065` | pass |
| P4 kernel-native pass | `0` | fail |

判断：C3 让 memory 和 step 进 gate，但 backward / forward 仍失败，所以不能打开 P6。

## 10. P6/P7 downstream

P6 AdamW trainability：

```text
p6_adamw_trainability_reentry.csv = not_run
reason = no_P5_combined_P4_pass_candidate
```

P7 functional：

```text
functional_open_allowed = 0
reason = P4_not_closed_so_P6_P5_near_pass_not_available
```

判断：没有用 P6/P7 越过 P4 gate，也没有用 functional update 救系统未闭合的 candidate。

## 11. No-fake audit

```text
rows_checked = 34
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

说明：

1. M/B/F/C 中未实现 candidate 均写为 `not_implemented`。
2. P6 是明确 `not_run`，不是 fake/proxy。
3. 本轮没有 teacher、label smoothing、loss.backward、offload 或 proxy source。

## 12. Hash

| artifact | SHA256 |
|---|---|
| v9.2.3 plan | `c7779af0c50d751f90151b9127fb87ddeb075dc37076c5dd5e3616d823397c2e` |
| `experiments/run_v923_p4_kernel_closure.py` | `12b294d0509f4843a8a15c202a5b5a020102c8ea5e77e25eba00e052d387525e` |
| route | `2bc2bb74d093e538b1d17b0d88a3a986e413466016e8e7179fed86ec21868ca1` |
| `p0_route_recap_measurement_stability.csv` | `b53319ff441c3ae1070e70be8b8f06006257712b61dd6307d040b6b0515cdbe2` |
| `p2_memory_live_set_closure.csv` | `e84f8cd19a93550b43cc614d72bb9d6d0541556b5df994edece1854270c67582` |
| `p3_backward_time_closure.csv` | `72b53402dec51e0a40f54d18f663138e858dfb89f703a32d3441145a73a80d00` |
| `p4_forward_time_closure.csv` | `4bea8202d95da622b29c3b38ecbba1560e78b781bf3a56712ff16964762dbccd` |
| `p5_combined_p4_gate_closure.csv` | `4152a6e996814876709dcc05feeab7f12dd323136da962631352a1bf04ed7658` |
| provenance audit | `8f2db458fac68dbf27bb89f6ae634b34b1d14b8182794ecd381816d1fdee61f3` |

## 13. 最终分析结论

v9.2.3 没有关闭 D2 compositional P4，但把 blocker 从 “memory + backward + forward 都未闭合” 推进为更明确的：

```text
memory/step 已接近或进入 gate；
backward 是主 blocker；
forward 仍是次 blocker。
```

机制判断：

1. M7/B6 证明第一层 input dx 可以安全移除：GradPass 仍成立，pairwise R2 仍为 `0.9911209941`。
2. compact live-set accounting 把 memory ratio 降到 `1.046065`，说明 memory 修复方向有效。
3. 但 backward ratio `2.063123` 几乎没有比 current repeat `2.061752` 改善，说明主要 backward 成本不是 raw-input dx，而是 layer1/layer2 basis derivative、coefficient grad、dh propagation 或 compiled kernel fragmentation。
4. forward ratio `1.425866` 仍高于 `1.25`，说明 forward-only y path 不足，还需要 source-normalization / basis-eval / projection 的更低层 fusion。
5. 当前下一步应该直接做 custom Triton/CUDA backward 或真正 streaming coeffgrad，而不是打开 P6/P7。

最终一句话：

> v9.2.3 真实执行后停在 `R4-BackwardKernelBlocker`：M7/B6/F3 让 memory ratio 进入 gate、step ratio 也过线，但 backward ratio 仍为 `2.063123`，forward ratio 仍为 `1.425866`，因此 P4 未闭合，P6 AdamW trainability 和 P7 functional update 不能打开。

## 14. 追加：Triton / streaming implementation

根据第 13 节 blocker，本轮继续实现并测量两个后续方向：

| 文件 | 改动 |
|---|---|
| `experiments/run_v923_p4_kernel_closure.py` | 新增 `M6/B4` layer1 streaming coeffgrad backward 候选 |
| `experiments/run_v923_p4_kernel_closure.py` | 新增 Triton `active1_residual_t2` forward kernel，用于 `F5-triton-forward-layer1-layer2` |
| `experiments/run_v923_p4_kernel_closure.py` | route 改为在 measured combined candidates 中选择最佳，而不是固定使用 C3 |
| `experiments/run_v923_p4_kernel_closure.py` | manifest 记录 `triton_available` 与 `triton_import_error` |

代码检查：

```text
python -m py_compile experiments/run_v923_p4_kernel_closure.py
```

已通过。

重要边界：

1. `F5` 本轮只实现 Triton forward residual kernel，没有实现 Triton backward，因此只计入 P4 forward closure，不计入 combined P4 pass。
2. `B4` streaming coeffgrad 是真实 backward 测量，若 correctness 或 timing 不过，就按 fail 写入。
3. 本轮没有改 CE、teacher、sampler、class weight、offload、loss.backward，也没有放宽 P4 gate。

正式 rerun：

```bash
python experiments/run_v923_p4_kernel_closure.py \
  --out-dir results/real_rerun_20260506/v923_p4_kernel_closure_streaming_triton_h512r4_20260509T141000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --hidden-dim 512 \
  --rank 4 \
  --lr 0.0005 \
  --p4-batch-size 128 \
  --p4-warmup 5 \
  --p4-reps 20
```

Runtime：

```text
device = NVIDIA L4
torch = 2.4.1+cu124
triton = 3.0.0
```

## 15. Updated route

`route_decision.json`：

```json
{
  "route": "R4-BackwardKernelBlocker",
  "best_candidate": "D2-FusedCompositional-T2",
  "best_memory_repair": "M7-no-input-dx-layer1",
  "best_backward_repair": "B6-no-dx-input",
  "best_forward_repair": "F3-no-materialized-basis-forward",
  "best_combined_candidate": "C3-M7+B6+F3-no-input-dx-forward-yonly",
  "grad_pass": 1,
  "synthetic_pairwise_R2": 0.9911209940910339,
  "forward_ratio": 1.42889846942922,
  "backward_ratio": 2.036291551054181,
  "step_ratio": 1.4514987551078007,
  "memory_ratio": 1.0460654216619045,
  "triton_forward_candidate_status": "measured_forward_only",
  "triton_forward_ratio": 3.034319712094846,
  "triton_forward_closure_pass": 0,
  "p4_kernel_native_pass": 0,
  "p5_trainability_opened": 0,
  "functional_open_allowed": 0,
  "primary_blocker": "backward_ratio_remains_above_1.50_after_streaming_or_no_input_dx_repair",
  "next_required_implementation": "implement_custom_triton_or_cuda_backward_for_layer1_layer2"
}
```

判断：

1. Streaming 没有替代 C3 成为最佳 combined candidate。
2. Triton forward-only 已真实执行，但 forward gate 未过，且没有 backward，不得计为 P4 combined success。
3. 最新 route 仍停在 `R4-BackwardKernelBlocker`。

## 16. Streaming / Triton result

P3 backward closure：

| candidate | GradRelErrMax | GradCosMin | backward ratio | step ratio | memory ratio | backward pass |
|---|---:|---:|---:|---:|---:|---:|
| `B4-streaming-grad-layer1` | `1.2872746447e-04` | `0.999999940` | `2.485696` | `1.567480` | `1.046065` | 0 |
| `B6-no-dx-input` | `5.6768145441e-05` | `0.999999881` | `2.036292` | `1.451499` | `1.046065` | 0 |

P4 forward closure：

| candidate | forward ratio | output diff vs torch y-only | combined eligible | forward pass |
|---|---:|---:|---:|---:|
| `F3-no-materialized-basis-forward` | `1.428898` |  | 1 | 0 |
| `F5-triton-forward-layer1-layer2` | `3.034320` | `0.0010814667` | 0 | 0 |

P5 combined closure：

| candidate | forward ratio | backward ratio | step ratio | memory ratio | GradRelErrMax | P4 pass |
|---|---:|---:|---:|---:|---:|---:|
| `C2-M6+B4+F3` | `1.424986` | `2.485696` | `1.567480` | `1.046065` | `1.2872746447e-04` | 0 |
| `C3-M7+B6+F3` | `1.428898` | `2.036292` | `1.451499` | `1.046065` | `5.6768145441e-05` | 0 |

判断：

1. Layer1 streaming coeffgrad 反而使 backward / step 更慢，且 grad relative error 略高于 gate；不能作为 repair。
2. 当前 Triton forward kernel 只融合了 source norm + T2 residual accumulation，投影仍是 matmul；实测比 compiled torch y-only 更慢，不能关闭 forward gate。
3. 最优仍是 C3：memory / step 进 gate，但 backward 和 forward 没进 gate。

## 17. No-fake audit / hash / 更新结论

No-fake audit：

```text
rows_checked = 34
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| v9.2.3 plan | `c7779af0c50d751f90151b9127fb87ddeb075dc37076c5dd5e3616d823397c2e` |
| `experiments/run_v923_p4_kernel_closure.py` | `bbd22642faa6134442ac8a1844bfd5629dfb1709ef2aff50042afb68bfb0de61` |
| route | `762148f3e2dcadee033217533119db9ba360bbaa0f64cd433b65d4c77f383456` |
| `p2_memory_live_set_closure.csv` | `fc0cb25ac6dd4bac5e537ac2b1152f7d193e840f496a38e4d8197a96a8ea449f` |
| `p3_backward_time_closure.csv` | `8179eb4b1074498b8c5d8e51bd5a884d32db63d90254dd3ec0de2a54e42561b4` |
| `p4_forward_time_closure.csv` | `3b010fa4093e20109bd4f2dc14ae942ca2b57dd9946aa99de022d5c40b865a99` |
| `p5_combined_p4_gate_closure.csv` | `2f3bb6f1455218a112bb59f748e2c938af50cfbecf1506f92d5b41d256ec9191` |
| provenance audit | `8f2db458fac68dbf27bb89f6ae634b34b1d14b8182794ecd381816d1fdee61f3` |

更新结论：

```text
Triton forward residual kernel = implemented and measured
Streaming coeffgrad = implemented and measured
Streaming combined P4 = fail
Triton forward closure = fail
Best combined P4 = C3-M7+B6+F3
P4 kernel-native gate = fail
P6 AdamW re-entry = not_run
P7 functional = not_opened
route = R4-BackwardKernelBlocker
```

机制结论：

1. 纯 Python/torch-level streaming coeffgrad 不是有效修复；它降低不了 compiled backward 的关键开销，还引入了更差的 backward/step timing。
2. 当前 Triton forward residual kernel 粒度太小：每层 residual accumulation 单独 kernel，再接 matmul projection，launch/融合不足，实测 forward ratio `3.034320`。
3. 真正需要的是完整 fused backward 或 CUDA/Triton kernel：把 source norm、basis derivative、coeffgrad、dh propagation 合并到 layer-level backward，而不是只替换单个 residual accumulation kernel。

最终一句话：

> v9.2.3 追加 Triton/streaming 后仍停在 `R4-BackwardKernelBlocker`：streaming coeffgrad 和 Triton forward-only 都已真实实现并测量，但没有关闭 P4；当前不能打开 P6/P7，下一步需要完整 custom Triton/CUDA backward，而不是 forward-only 小 kernel。

## 18. 追加：完整 custom Triton backward pack

根据第 17 节结论，本轮继续实现完整 backward pack，而不是只替换 forward residual 小 kernel。

代码改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_v923_p4_kernel_closure.py` | 新增 Triton `active1_dm_t2` kernel：直接计算 `dM = G^T @ dr` |
| `experiments/run_v923_p4_kernel_closure.py` | 新增 Triton `active1_dx_t2` kernel：直接计算 basis derivative 的 `dx correction` |
| `experiments/run_v923_p4_kernel_closure.py` | 新增 `B8-triton-layer1-layer2-layer3-backward` measured candidate |
| `experiments/run_v923_p4_kernel_closure.py` | 新增 `C6-M7+B8+F3` combined P4 candidate |

实现边界：

```text
dM / dx-correction = Triton kernels
dW0 / dV = GEMM
layer1 input dx = removed by no-input-dx contract
loss_type = CE
label_smoothing = 0
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
fake/proxy = 0
```

说明：`dW0 = x^T @ dy` 与 `dV = dy^T @ r` 保持 GEMM，是因为它们本身就是大矩阵乘法；本轮 Triton 化的是 FullEdge basis backward 的关键非 GEMM 路径 `dM` 与 `dx correction`。

代码检查：

```text
python -m py_compile experiments/run_v923_p4_kernel_closure.py
```

已通过。

正式 rerun：

```bash
python experiments/run_v923_p4_kernel_closure.py \
  --out-dir results/real_rerun_20260506/v923_p4_kernel_closure_triton_backward_h512r4_20260509T143000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --hidden-dim 512 \
  --rank 4 \
  --lr 0.0005 \
  --p4-batch-size 128 \
  --p4-warmup 5 \
  --p4-reps 20
```

## 19. Updated route

`route_decision.json`：

```json
{
  "route": "R4-BackwardKernelBlocker",
  "best_candidate": "D2-FusedCompositional-T2",
  "best_memory_repair": "M7-no-input-dx-layer1",
  "best_backward_repair": "B6-no-dx-input",
  "best_forward_repair": "F3-no-materialized-basis-forward",
  "best_combined_candidate": "C3-M7+B6+F3-no-input-dx-forward-yonly",
  "grad_pass": 1,
  "synthetic_pairwise_R2": 0.9911209940910339,
  "forward_ratio": 1.4218004565238558,
  "backward_ratio": 2.0223987434746493,
  "step_ratio": 1.4655257581206513,
  "memory_ratio": 1.0460654216619045,
  "triton_forward_candidate_status": "measured_forward_only",
  "triton_forward_ratio": 3.2527269026426815,
  "triton_forward_closure_pass": 0,
  "p4_kernel_native_pass": 0,
  "p5_trainability_opened": 0,
  "functional_open_allowed": 0,
  "primary_blocker": "backward_ratio_remains_above_1.50_after_streaming_or_no_input_dx_repair",
  "next_required_implementation": "implement_custom_triton_or_cuda_backward_for_layer1_layer2"
}
```

判断：

1. 完整 Triton backward pack 已纳入 measured candidates。
2. B8/C6 没有成为最佳 combined candidate，因为 backward 和 step 明显更慢。
3. 最新 route 仍是 `R4-BackwardKernelBlocker`。

## 20. Triton backward result

P3 backward closure：

| candidate | GradRelErrMax | GradCosMin | backward ratio | step ratio | memory ratio | backward pass |
|---|---:|---:|---:|---:|---:|---:|
| `B4-streaming-grad-layer1` | `1.2872746447e-04` | `0.999999940` | `2.540745` | `1.576449` | `1.046065` | 0 |
| `B6-no-dx-input` | `5.6768145441e-05` | `0.999999881` | `2.022399` | `1.465526` | `1.046065` | 0 |
| `B8-triton-layer1-layer2-layer3-backward` | `1.0023164214e-04` | `1.000000000` | `14.134362` | `4.238933` | `1.046065` | 0 |

P5 combined closure：

| candidate | forward ratio | backward ratio | step ratio | memory ratio | GradRelErrMax | P4 pass |
|---|---:|---:|---:|---:|---:|---:|
| `C2-M6+B4+F3` | `1.415323` | `2.540745` | `1.576449` | `1.046065` | `1.2872746447e-04` | 0 |
| `C3-M7+B6+F3` | `1.421800` | `2.022399` | `1.465526` | `1.046065` | `5.6768145441e-05` | 0 |
| `C6-M7+B8+F3` | `1.341980` | `14.134362` | `4.238933` | `1.046065` | `1.0023164214e-04` | 0 |

Triton forward-only recap：

| candidate | forward ratio | output diff vs torch y-only | forward pass |
|---|---:|---:|---:|
| `F5-triton-forward-layer1-layer2` | `3.252727` | `0.0013194084` | 0 |

判断：

1. B8 的数值正确性接近 gate，但 `GradRelErrMax` 仍略高于 `1e-4`，不能写 pass。
2. B8 的 runtime 明显失败：`backward_ratio = 14.134362`，说明当前 Triton kernels 粒度过细，launch overhead 和 dM/dx 分离成本超过了 torch compiled path。
3. C6 虽然 forward ratio 看起来略低于 C3，但 backward/step 大幅失败，不能进入 P6。

## 21. No-fake audit / hash / 更新结论

No-fake audit：

```text
rows_checked = 36
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| v9.2.3 plan | `c7779af0c50d751f90151b9127fb87ddeb075dc37076c5dd5e3616d823397c2e` |
| `experiments/run_v923_p4_kernel_closure.py` | `6225cb3ea9daaa11689521d24bb846611fee5108e0b344d67442d8787d5bc7fe` |
| route | `56d41c14a2f849ff9a8a104f5c5623f1415dd8fce22ecaca47d80e186b549f64` |
| `p2_memory_live_set_closure.csv` | `aab0f60c5cac2dec1978f236e74b8a042ac2332072ef8bf9d048afc572cc7f98` |
| `p3_backward_time_closure.csv` | `79081eee572183048f4d7453f8cef769c0efa39bc8701688ac48738a6b8761e6` |
| `p4_forward_time_closure.csv` | `7f73cbbb596e01a723e4d81abfd922f9ffb897d43f80df445a524e46494dc3ff` |
| `p5_combined_p4_gate_closure.csv` | `c3a50ffab4b99062c88d0fe8286e126c1cb3024f496172eadf00bf869c64613c` |
| provenance audit | `523f5055972a9fb3a741a4fa427ab14eddae5a1364c897076fec2e20e0e7672f` |

更新结论：

```text
Custom Triton backward pack = implemented and measured
B8 Triton backward correctness = near but not pass
B8 Triton backward timing = fail
Best combined P4 = C3-M7+B6+F3
P4 kernel-native gate = fail
P6 AdamW re-entry = not_run
P7 functional = not_opened
route = R4-BackwardKernelBlocker
```

机制结论：

1. “只替换 forward residual 小 kernel” 已经被排除；本轮确实实现了 backward 侧的 `dM` 与 `dx correction` Triton kernels。
2. 但 kernel 粒度仍然不对：每层拆成多次 GEMM + Triton dM + Triton dx，导致 launch/同步成本极高。
3. 下一步如果继续 Triton/CUDA，应做单层 monolithic backward kernel 或 persistent kernel，把 `dr`、`dM`、`dx correction`、必要 reductions 合在一个 kernel family 内；否则继续堆小 Triton kernel 只会比 compiled torch 慢。

最终一句话：

> v9.2.3 已实现完整 Triton backward pack 并真实测量，但 B8/C6 没有关闭 P4：`backward_ratio = 14.134362`，`step_ratio = 4.238933`。当前最佳仍是 C3 no-dx/y-only，route 继续停在 `R4-BackwardKernelBlocker`，P6/P7 不打开。

## 22. 追加：single-layer monolithic Triton backward

根据第 21 节结论，本轮继续把 B8 的分离小 kernel 改成单层 monolithic kernel：每个 hidden layer 的 `dM` reduction 与 `dx correction` 在同一个 Triton kernel 内完成，减少 `dM kernel + dx kernel` 的分离 launch。

代码改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_v923_p4_kernel_closure.py` | 新增 Triton `active1_layer_backward_mono_t2` kernel |
| `experiments/run_v923_p4_kernel_closure.py` | 新增 Triton `active1_layer1_dm_mono_t2` kernel |
| `experiments/run_v923_p4_kernel_closure.py` | 新增 `B9-triton-monolithic-layer-backward` measured candidate |
| `experiments/run_v923_p4_kernel_closure.py` | 新增 `C7-M11+B9+F3` combined P4 candidate |

实现边界：

```text
layer2/layer3 dM + dx correction = single monolithic Triton kernel per layer
layer1 dM = single Triton kernel; input dx removed
dW0 / dV = GEMM
loss_type = CE
label_smoothing = 0
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
fake/proxy = 0
```

代码检查：

```text
python -m py_compile experiments/run_v923_p4_kernel_closure.py
```

已通过。

正式 rerun：

```bash
python experiments/run_v923_p4_kernel_closure.py \
  --out-dir results/real_rerun_20260506/v923_p4_kernel_closure_triton_monolithic_h512r4_20260509T150000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --hidden-dim 512 \
  --rank 4 \
  --lr 0.0005 \
  --p4-batch-size 128 \
  --p4-warmup 5 \
  --p4-reps 20
```

## 23. Updated route

`route_decision.json`：

```json
{
  "route": "R4-BackwardKernelBlocker",
  "best_candidate": "D2-FusedCompositional-T2",
  "best_memory_repair": "M7-no-input-dx-layer1",
  "best_backward_repair": "B6-no-dx-input",
  "best_forward_repair": "F3-no-materialized-basis-forward",
  "best_combined_candidate": "C3-M7+B6+F3-no-input-dx-forward-yonly",
  "grad_pass": 1,
  "synthetic_pairwise_R2": 0.9911209940910339,
  "forward_ratio": 1.4215219262138123,
  "backward_ratio": 2.0480997787972988,
  "step_ratio": 1.4637718006772154,
  "memory_ratio": 1.0460654216619045,
  "triton_forward_candidate_status": "measured_forward_only",
  "triton_forward_ratio": 3.305378047308424,
  "triton_forward_closure_pass": 0,
  "p4_kernel_native_pass": 0,
  "p5_trainability_opened": 0,
  "functional_open_allowed": 0,
  "primary_blocker": "backward_ratio_remains_above_1.50_after_streaming_or_no_input_dx_repair",
  "next_required_implementation": "implement_custom_triton_or_cuda_backward_for_layer1_layer2"
}
```

判断：

1. B9/C7 已真实测量，但没有成为最佳。
2. B9 把 B8 的 correctness 拉进 gate，也略微改善 runtime，但 backward/step 仍大幅失败。
3. 最新 route 仍是 `R4-BackwardKernelBlocker`。

## 24. Monolithic Triton result

P3 backward closure：

| candidate | GradRelErrMax | GradCosMin | backward ratio | step ratio | memory ratio | backward pass |
|---|---:|---:|---:|---:|---:|---:|
| `B4-streaming-grad-layer1` | `1.2872746447e-04` | `0.999999940` | `2.521951` | `1.567745` | `1.046065` | 0 |
| `B6-no-dx-input` | `5.6768145441e-05` | `0.999999881` | `2.048100` | `1.463772` | `1.046065` | 0 |
| `B8-triton-layer1-layer2-layer3-backward` | `1.0023164214e-04` | `1.000000000` | `15.680900` | `4.237832` | `1.046065` | 0 |
| `B9-triton-monolithic-layer-backward` | `6.8147972343e-05` | `1.000000000` | `13.413305` | `4.007630` | `1.046065` | 0 |

P5 combined closure:

| candidate | forward ratio | backward ratio | step ratio | memory ratio | GradRelErrMax | P4 pass |
|---|---:|---:|---:|---:|---:|---:|
| `C2-M6+B4+F3` | `1.427007` | `2.521951` | `1.567745` | `1.046065` | `1.2872746447e-04` | 0 |
| `C3-M7+B6+F3` | `1.421522` | `2.048100` | `1.463772` | `1.046065` | `5.6768145441e-05` | 0 |
| `C6-M7+B8+F3` | `1.335630` | `15.680900` | `4.237832` | `1.046065` | `1.0023164214e-04` | 0 |
| `C7-M11+B9+F3` | `1.351698` | `13.413305` | `4.007630` | `1.046065` | `6.8147972343e-05` | 0 |

判断：

1. B9 monolithic kernel 比 B8 split-kernel 更好：GradRelErr 从 `1.002316e-04` 降到 `6.814797e-05`，backward ratio 从 `15.680900` 降到 `13.413305`。
2. 但 B9 仍远慢于 B6 no-dx-input 的 `2.048100`，因此不是有效 P4 repair。
3. 这说明当前瓶颈不只是 dM/dx 分离 launch，而是 Triton 实现本身没有把 GEMM 周边、reductions、层间传播做成真正高效的 persistent/monolithic layer backward。

## 25. No-fake audit / hash / 更新结论

No-fake audit：

```text
rows_checked = 39
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| v9.2.3 plan | `c7779af0c50d751f90151b9127fb87ddeb075dc37076c5dd5e3616d823397c2e` |
| `experiments/run_v923_p4_kernel_closure.py` | `0995c75be982ad4155d08fb409efdabf716b10818da37b7b0c195812d3dc4d28` |
| route | `b476c1c3aaa29003115ea23d9f729044b13f7df358d90d17a81fd34c0f44f6b8` |
| `p2_memory_live_set_closure.csv` | `6ac140e6fae91d724c91d56c1ae9439d874d1841d5e9ed6448aabd63b25a9309` |
| `p3_backward_time_closure.csv` | `07705a4a8ba419bf921c2a35f1da5cb1432995115cd7ec4e1962eaed5bcf9bf3` |
| `p4_forward_time_closure.csv` | `973447707ce77b98801b998a5cff09cf6adc0881f859ff07deb2c93755122938` |
| `p5_combined_p4_gate_closure.csv` | `dad336e107ec791d04d12035a0c4e6cc82cb93cf8414fe08cfb5e88f2a2e84c4` |
| provenance audit | `f07fe13c036f60853bf38fc815643d94cf5d46c48727063edee319bbb8848323` |

更新结论：

```text
Single-layer monolithic Triton backward = implemented and measured
B9 monolithic correctness = pass
B9 monolithic timing = fail
Best combined P4 = C3-M7+B6+F3
P4 kernel-native gate = fail
P6 AdamW re-entry = not_run
P7 functional = not_opened
route = R4-BackwardKernelBlocker
```

机制结论：

1. Monolithic B9 比 split B8 更合理，但仍不够；当前 Triton 方案没有接近 MLP-match envelope。
2. 真正要继续，需要更激进的 persistent kernel 或改变计算组织：例如把 `dr` 生成、`dM` reduction、`dx correction` 和部分 parameter grad accumulation 放进同一持久化 kernel family，而不是 GEMM 与 Triton kernel 来回切换。
3. 在此之前不能打开 P6/P7，也不能把 D2 compositional trainability 写成 repaired。

最终一句话：

> v9.2.3 追加 single-layer monolithic Triton backward 后仍停在 `R4-BackwardKernelBlocker`：B9 数值过关但 `backward_ratio = 13.413305`、`step_ratio = 4.007630`，远未关闭 P4；当前最佳仍是 C3 no-dx/y-only。
