# DG-KAN v9.2.4 GEMMNative PersistentCompositionalFullEdge 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.4_GEMMNative_PersistentCompositionalFullEdge_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P7/P8 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.4 执行到一个可审计 terminal route：

```text
route = R4-BackwardLoweringLimit
best_candidate = G2-GEMMNative-FusedPointwise-compiled
success_v924_p4_kernel_closure = false
success_v924_trainability_reentry = false
success_v924_functional_opened = false
```

最终 artifact：

```text
results/real_rerun_20260506/v924_gemm_native_persistent_h512r4_20260509T162000Z/
```

核心结论：

1. v9.2.3 的 C3/B9 boundary 被真实复现：C3 repeat backward 仍约 `2.013340x`，B9 monolithic backward 仍远慢，约 `14.325169x`。
2. Algebraic lowering audit 已完成，12 个 primitive 均给出 GEMM / fused pointwise / reduction / persistent lowering decision。
3. GEMM-native 当前最佳是 `G2-GEMMNative-FusedPointwise-compiled`，数值正确且 memory/step 过线，但 forward `1.439289x`、backward `2.042472x` 均未过 P4 gate。
4. CUDA Graph candidate `G6` 真实尝试捕获，但 runtime 报错，按 `not_run` 记录，没有计入 pass。
5. Persistent/custom kernel 本轮只复测 B9 reference；B9 数值过关，但 backward `12.440080x`、step `3.821693x`，仍明显失败。更强 persistent / CUDA extension 分支未实现，明确写 `not_implemented`。
6. P6 smaller width/rank fallback 全部 system gate fail，且没有测 synthetic R2，因此没有用 R2 proxy 把任何 fallback 写成 official pass。
7. 因 P4 未闭合，P7 AdamW-only trainability 与 P8 functional open 均未打开。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v924_gemm_native_persistent.py` | v9.2.4 runner；生成 P0-P8 required artifacts、route、failure table、no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v924_gemm_native_persistent.py
```

已通过。

正式运行：

```bash
python experiments/run_v924_gemm_native_persistent.py \
  --out-dir results/real_rerun_20260506/v924_gemm_native_persistent_h512r4_20260509T162000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --hidden-dim 512 \
  --rank 4 \
  --lr 0.0005 \
  --p4-batch-size 128 \
  --p4-warmup 5 \
  --p4-reps 20 \
  --p6-hidden-dims 384,256 \
  --p6-ranks 4,2
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R4-BackwardLoweringLimit",
  "best_candidate": "G2-GEMMNative-FusedPointwise-compiled",
  "best_lowering_path": "GEMMNative",
  "best_backward_path": "G2-GEMMNative-FusedPointwise-compiled",
  "best_forward_path": "F1-GEMMNativeForward",
  "best_hidden_dim": 512,
  "best_rank": 4,
  "full_edge_equivalence_pass": 1,
  "no_external_residual_pass": 1,
  "grad_pass": 1,
  "synthetic_pairwise_R2": 0.9911209940910339,
  "forward_ratio": 1.4392885670662556,
  "backward_ratio": 2.042472474440813,
  "step_ratio": 1.3739124628600536,
  "memory_ratio": 1.0460654216619045,
  "p4_kernel_native_pass": 0,
  "p5_trainability_opened": 0,
  "p5_near_pass": 0,
  "functional_open_allowed": 0,
  "primary_blocker": "gemm_native_cuda_graph_and_persistent_reference_do_not_reduce_backward_below_1.50",
  "next_required_implementation": "pivot_to_CUDA_extension_persistent_layer_backward_or_patch_local_KANConv_primitive",
  "success_v924_p4_kernel_closure": 0,
  "success_v924_trainability_reentry": 0,
  "success_v924_functional_opened": 0
}
```

判断：本轮没有关闭 P4。当前最佳 candidate 的 memory 与 step 已过线，但 backward 和 forward 均未进入 MLP-match envelope。

## 3. P0 latest route reproduction

P0 artifact：

```text
p0_latest_route_reproduction.csv
```

关键复现：

| candidate | GradRelErrMax | forward ratio | backward ratio | step ratio | memory ratio |
|---|---:|---:|---:|---:|---:|
| `D2-C3-current-repeat` | `5.914181e-06` | `1.406714` | `2.013340` | `7.541445` | `1.046065` |
| `D2-B9-monolithic-repeat` | `8.479931e-05` | `1.248832` | `14.325169` | `4.086781` | `1.046065` |

判断：

1. C3 的 backward boundary 仍稳定在 `~2.0x`，没有变成偶然 pass。
2. B9 monolithic Triton reference 仍远慢于 C3，说明 v9.2.3 的“当前 Triton 组织不够 GEMM/persistent-grade”结论继续成立。
3. P0 repeat 的 C3 step timing 出现较大波动，因此最终 route 不使用 P0 step 作成功依据，而使用 P2/P5 的正式 candidate timing。

## 4. P1 algebraic lowering audit

P1 artifact：

```text
p1_algebraic_lowering_audit.csv
```

审计结果：

```text
primitive rows = 12
lowering_decision_present = 12/12
unknown lowering = 0
```

Lowering 分类：

| primitive | recommended lowering |
|---|---|
| source_norm | fused_pointwise |
| T2_eval | fused_pointwise |
| R_generation | GEMM |
| identity_projection | GEMM |
| correction_projection | GEMM |
| dV | GEMM |
| dW0 | GEMM |
| dR | GEMM |
| dU | GEMM_or_streaming_reduction |
| da | streaming_reduction |
| dx_correction | persistent_kernel |
| dh_propagation | fused_pointwise |

判断：P1 没有跳过。当前未闭合点集中在 `dU/da/dx_correction` 的 reduction / derivative lowering，而不是 FullEdge 数学图不清楚。

## 5. P2 GEMM-native backward

P2 artifact：

```text
p2_gemm_native_backward.csv
```

实测 rows：

| candidate | GradPass | forward ratio | backward ratio | step ratio | memory ratio | P4 pass |
|---|---:|---:|---:|---:|---:|---:|
| `G1-GEMMNative-VJP-eager` | 1 | `6.429508` | `2.084071` | `2.266089` | `1.046065` | 0 |
| `G2-GEMMNative-FusedPointwise-compiled` | 1 | `1.439289` | `2.042472` | `1.373912` | `1.046065` | 0 |
| `G6-GEMMNative-CUDAGraph` | 0 | `not_run` | `not_run` | `not_run` | `not_run` | 0 |

G6 not_run reason：

```text
cuda_graph_capture_failed:RuntimeError:Cannot prepare for replay during capturing stage...
```

未实现 rows：

```text
G3-GEMMNative-TwoPassReduction = not_implemented
G4-GEMMNative-BatchedLayerBackward = not_implemented
G5-GEMMNative-NoMaterializedDr = not_implemented
```

判断：

1. G1 eager GEMM-native 正确但明显慢，forward `6.43x`，step `2.27x`。
2. G2 compiled 是本轮最佳，但 backward `2.042472x` 没有比 C3 降到 `0.75x` 以内，也没有过 `1.50x` gate。
3. CUDA Graph 没有成功捕获，不能把它写成 measured pass。

## 6. P3 persistent / custom backward

P3 artifact：

```text
p3_persistent_custom_backward.csv
```

实测 reference：

| candidate | GradRelErrMax | forward ratio | backward ratio | step ratio | memory ratio | P4 pass |
|---|---:|---:|---:|---:|---:|---:|
| `P1-PersistentLayerBackward-Minimal-B9-reference` | `8.479931e-05` | `1.350444` | `12.440080` | `3.821693` | `1.046065` | 0 |

未实现 rows：

```text
P2-PersistentLayerBackward-WithDVec = not_implemented
P3-PersistentLayerBackward-OneCTAperR = not_implemented
P4-PersistentLayerBackward-OneCTAperOutputTile = not_implemented
P5-PersistentLayerBackward-WarpReduction = not_implemented
P6-CUDAExtensionLayerBackward = not_implemented
```

判断：B9 reference 数值过关，但 runtime 仍远离 gate；更强 persistent / CUDA extension 没有在本轮实现，因此没有被伪造成结果。

## 7. P4 forward closure

P4 artifact：

```text
p4_forward_closure.csv
```

实测 rows：

| candidate | output diff | forward ratio | memory ratio | forward pass |
|---|---:|---:|---:|---:|
| `F1-GEMMNativeForward` | `3.099442e-06` | `1.439289` | `1.046065` | 0 |
| `F4-CUDAGraphForward` | `not_run` | `not_run` | `not_run` | 0 |
| `F5-PersistentForward-triton-forward-reference` | `0.002103806` | `3.204830` | `1.046065` | 0 |

判断：

1. F1 数值正确，但 forward ratio 仍超过 `1.25`。
2. F5 Triton forward reference 仍慢且 output diff 明显大于 `1e-5`，不能算 forward closure。
3. Forward 仍是 secondary blocker；primary blocker 仍是 backward。

## 8. P5 combined P4 closure

P5 artifact：

```text
p5_combined_p4_closure.csv
```

Combined rows：

| candidate | lowering path | forward ratio | backward ratio | step ratio | memory ratio | P4 pass |
|---|---|---:|---:|---:|---:|---:|
| `C1-G2+F2` | GEMMNative | `1.439289` | `2.042472` | `1.373912` | `1.046065` | 0 |
| `C4-G6+F4` | CUDA graph | `not_run` | `not_run` | `not_run` | `not_run` | 0 |
| `C5-P1+F2` | PersistentReference | `1.350444` | `12.440080` | `3.821693` | `1.046065` | 0 |

未实现 combined rows：

```text
C2-G3+F2 = not_implemented
C3-G5+F2 = not_implemented
C6-P3+F5 = not_implemented
C7-P6+F2 = not_implemented
```

判断：没有任何 combined candidate 关闭 P4；因此 P7/P8 不允许打开。

## 9. P6 system Pareto fallback

P6 artifact：

```text
p6_system_pareto_fallback.csv
```

本轮 fallback 只测系统 gate；没有测这些 width/rank 变体的 synthetic pairwise R2，因此不允许用 proxy R2 写成 official pass。

| candidate | forward ratio | backward ratio | step ratio | memory ratio | system gate |
|---|---:|---:|---:|---:|---:|
| `S-depth2-hidden384-rank4` | `1.523745` | `2.449736` | `1.521784` | `1.066302` | 0 |
| `S-depth2-hidden384-rank2` | `1.588623` | `2.636023` | `1.443259` | `1.067604` | 0 |
| `S-depth2-hidden256-rank4` | `1.520637` | `2.835486` | `1.574893` | `1.107991` | 0 |
| `S-depth2-hidden256-rank2` | `1.632686` | `3.023604` | `1.504245` | `1.111442` | 0 |

判断：smaller width/rank fallback 没有形成系统 Pareto 修复，且 memory 往往离开 gate。

## 10. P7/P8 downstream

P7 AdamW-only trainability：

```text
p7_adamw_trainability_reentry.csv = not_run
reason = no_P5_or_P6_P4_pass_candidate
```

P8 functional open：

```text
functional_open_allowed = 0
reason = P4_not_closed
```

判断：没有用 trainability 或 functional update 越过 P4 kernel gate。

## 11. No-fake audit

No-fake audit：

```text
rows_checked = 48
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

说明：

1. P0/P2/P3/P4/P5/P6 均来自真实测量或明确的 `not_implemented` / `not_run` row。
2. P6 fallback 没有使用 proxy synthetic R2。
3. P7/P8 是明确 gate-blocked，不是 fake success。

## 12. Hash

| artifact | SHA256 |
|---|---|
| v9.2.4 plan | `2bf3afae1b5cea4881c9e24f798d8fe524e75a65ff822219fce793b4796b12ff` |
| `experiments/run_v924_gemm_native_persistent.py` | `169869567bb58d1f80407dc695254d3273f3173b6631c2812c8f7ad1ab6eebee` |
| route | `ee3d966b1922f9ec1c8bfff1f8bd6a1f1a7464686b5dbae427b6729782b8dbba` |
| `p0_latest_route_reproduction.csv` | `1bd1f1fc904e6eb020cc37caec95dc8bcace889b195d9d2f682be7e22858073e` |
| `p1_algebraic_lowering_audit.csv` | `9fdc706f42c3113265f34e04568754fa2754964a2c71df8ce60ad0ee87455d92` |
| `p2_gemm_native_backward.csv` | `788ba9630b293bc6c3a392641ca710e5cecd12d28dc585a5944833b6dd1f1c2d` |
| `p3_persistent_custom_backward.csv` | `24af4a1350ac24bea731b43d62a56714a7388a835542db80615d0b706d532993` |
| `p4_forward_closure.csv` | `ee5dd2114d0fa515a7dbb083affc1a2611cab4a3e5f2b20cbed8a707ebdb49ca` |
| `p5_combined_p4_closure.csv` | `c058bc122582a014f5cec5c61fff579e33af0a34ca0ad6bf75a572a946fe8dfb` |
| `p6_system_pareto_fallback.csv` | `d4df659084c84e69f83d6cd16863e1fbda6f7cc8fb4f4384bc2a647500068427` |
| provenance audit | `744046cdd48768b5ac77d909781d7f7d117642854912c0474f818c2df363aa12` |

## 13. 最终分析结论

v9.2.4 没有关闭 P4，但把 v9.2.3 后的系统判断推进到一个更明确边界：

```text
GEMM-native compiled path 是当前最佳，但不能把 backward 从 ~2.04x 降到 <=1.50x；
B9-style Triton / monolithic reference 仍远慢；
CUDA Graph 捕获未成功；
smaller width/rank fallback 也没有系统 gate pass。
```

机制判断：

1. 当前 D2 compositional FullEdge 的数学与梯度仍成立：best candidate 的 `GradPass=1`，`synthetic_pairwise_R2=0.9911209941`。
2. Memory 与 step 已不是主要 blocker：best candidate memory `1.046065`、step `1.373912` 均在 gate 内。
3. GEMM-native lowering 没有实质改善 backward：`G2` backward `2.042472`，相对 C3 约等价，不满足 `<=1.50`。
4. 当前 Triton reference 也没有可用：B9 backward `12.440080`，说明继续堆小 Triton / 半 persistent 方向不够。
5. Forward 也未闭合：best forward `1.439289`，但 backward gap 更大，因此 route 写为 `R4-BackwardLoweringLimit`。
6. 下一步若继续 FC D2，应实现真正 CUDA extension persistent layer backward，或按计划转向 patch-local / KANConv primitive；不能打开 P7/P8，也不能声明 trainability repaired。

最终一句话：

> v9.2.4 真实执行后停在 `R4-BackwardLoweringLimit`：GEMM-native compiled path 是当前最佳，但 `backward_ratio=2.042472`、`forward_ratio=1.439289`，P4 仍未闭合；P7 AdamW-only 和 functional update 均不能打开。
