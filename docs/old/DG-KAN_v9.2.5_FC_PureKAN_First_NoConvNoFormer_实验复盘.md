# DG-KAN v9.2.5 FC-PureKAN First NoConvNoFormer 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.5_FC_PureKAN_First_NoConvNoFormer_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 PureKANConv / PureKANFormer 或 P6/P7 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.5 执行到一个可审计 terminal route：

```text
route = R8-FCPureKANSystemNotClosed
best_candidate = R7-InteractionRetainingLowRankD2
kanconv_deferred = true
purekanformer_deferred = true
success_v925_fc_d2_closure = false
success_v925_fc_purekan_redesign = false
success_v925_trainability_reentry = false
success_v925_functional_opened = false
```

最终 artifact：

```text
results/real_rerun_20260506/v925_fc_purekan_first_h512r4_20260509T170000Z/
```

核心结论：

1. 本轮按计划收紧路线：PureKANConv、PureKANFormer、KAN-FFN 全部 `deferred`，`measured=0`，`eligible_for_route=0`。
2. v9.2.4 的 G2 boundary 被复现：`backward_ratio=2.047977`，仍稳定高于 `1.50`；`forward_ratio=1.400478` 也高于 `1.25`。
3. FC-D2 CUDA / persistent final attempt 没有产生有效 candidate：FC1-FC5 因 CUDA extension 未实现写为 `not_implemented`，FC6 CUDA Graph retry 真实尝试后仍 capture fail。
4. FC-D2 viability decision 明确进入 FC-PureKAN redesign，不允许 pivot Conv / Former。
5. FC-only redesign 中，一层 dense edge-basis R1/R2/R4/R5 的 system timing 很快，但 memory 明显超 gate，且 pairwise synthetic R2 为负，说明它们退化为 additive edge model，不能替代 D2 interaction。
6. R7 当前 D2 lowrank 保留 pairwise R2 `0.9911209941` 和 GradPass，但 P4 仍 fail。
7. 因没有 FC-D2 或 FC-redesign P4-pass candidate，P6 AdamW-only 和 P7 functional 均未打开。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v925_fc_purekan_first.py` | v9.2.5 runner；生成 P0-P7 required artifacts、route、failure table、no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v925_fc_purekan_first.py
```

已通过。

正式运行：

```bash
python experiments/run_v925_fc_purekan_first.py \
  --out-dir results/real_rerun_20260506/v925_fc_purekan_first_h512r4_20260509T170000Z \
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

## 2. Route

`route_decision.json`：

```json
{
  "route": "R8-FCPureKANSystemNotClosed",
  "best_candidate": "R7-InteractionRetainingLowRankD2",
  "route_family": "FC-PureKAN",
  "fc_d2_continue_allowed": 0,
  "fc_purekan_redesign_required": 1,
  "kanconv_deferred": 1,
  "purekanformer_deferred": 1,
  "best_fc_candidate": "D2-G2-current-repeat",
  "best_redesign_candidate": "R7-InteractionRetainingLowRankD2",
  "full_edge_equivalence_pass": 1,
  "no_external_residual_pass": 1,
  "grad_pass": 1,
  "synthetic_pairwise_R2": 0.9911209940910339,
  "forward_ratio": 1.4004781860234998,
  "backward_ratio": 2.0479766422406374,
  "step_ratio": 7.639726748587679,
  "memory_ratio": 1.0460654216619045,
  "p4_pass": 0,
  "p5_trainability_opened": 0,
  "p5_near_pass": 0,
  "functional_open_allowed": 0,
  "primary_blocker": "FC_D2_final_attempt_not_available_or_failed_and_FC_redesign_candidates_all_failed_P4",
  "next_required_implementation": "FC_PureKAN_primitive_redesign_with_real_kernel_or_trainability_preserving_factorization",
  "success_v925_fc_d2_closure": 0,
  "success_v925_fc_purekan_redesign": 0,
  "success_v925_trainability_reentry": 0,
  "success_v925_functional_opened": 0
}
```

判断：本轮没有进入 Conv / Former，也没有关闭 FC-PureKAN P4。`step_ratio` 在本次 repeat 中波动很大，但由于 forward/backward 已经 fail，本轮没有用 step timing 推导任何成功结论。

## 3. P0 route tightening / deferred registry

P0 artifacts：

```text
contract_audit_v925_fc_only.csv
deferred_architecture_registry_v925.csv
```

Deferred registry：

| candidate | status | measured | eligible for route | reason |
|---|---|---:|---:|---|
| Deferred-PureKANConv | deferred | 0 | 0 | FC_PureKAN_not_yet_P5_near_pass |
| Deferred-PureKANFormer | deferred | 0 | 0 | FC_PureKAN_not_yet_P5_near_pass |
| Deferred-KANFFN | deferred | 0 | 0 | FC_PureKAN_not_yet_P5_near_pass |

Contract audit：

```text
loss_type = CE
label_smoothing = 0
external_teacher_used = 0
self_teacher_used = 0
uses_loss_backward = 0
fake_data_used = 0
proxy_row_used = 0
full_edge_equivalence_pass = 1
no_external_residual_pass = 1
ordinary_mlp_hidden_path_used = 0
trainable_preprocessor_used = 0
```

判断：v9.2.5 的路线收紧已落实；没有任何 PureKANConv / PureKANFormer active measured row。

## 4. P1 v9.2.4 boundary repeat

P1 artifact：

```text
p1_v924_boundary_repeat.csv
```

结果：

| candidate | GradRelErrMax | GradCosMin | forward ratio | backward ratio | step ratio | memory ratio | stable |
|---|---:|---:|---:|---:|---:|---:|---:|
| D2-G2-current-repeat | `5.914181e-06` | `0.999999940` | `1.400478` | `2.047977` | `7.639727` | `1.046065` | 1 |

判断：

1. backward boundary 与 v9.2.4 的 `2.042472` 一致，repeat stable。
2. forward 仍高于 `1.25`。
3. step timing 本轮异常偏高，但不影响 terminal route：P4 已经因 forward/backward fail 而不能通过。

## 5. P2 FC-D2 final lowering audit

P2 artifact：

```text
p2_fc_d2_final_lowering_audit.csv
```

Rows：

```text
primitive rows = 12
recommended lowering = GEMM / fused_pointwise / persistent_kernel
estimated_backward_lower_bound_ratio = 1.48
estimated_forward_lower_bound_ratio = 1.22
estimated_memory_lower_bound_ratio = 1.046
```

说明：P2 是 lower-bound audit，不是实测成功。它只说明理论上 FC-D2 仍值得一次 CUDA / persistent final attempt；真实 P3 仍必须有实现和测量。

关键 persistent primitives：

| primitive | recommended lowering | requires reduction | persistent needed |
|---|---|---:|---:|
| dU | persistent_kernel | 1 | 1 |
| dA | persistent_kernel | 1 | 1 |
| dx_correction | persistent_kernel | 0 | 1 |

判断：P2 没有宣称 P4 成功；它只是把 final attempt 的必要性定位到 dU/dA/dx correction 的 persistent lowering。

## 6. P3 FC-D2 CUDA / persistent final attempt

P3 artifact：

```text
p3_fc_d2_cuda_persistent_attempt.csv
```

Rows：

| candidate | status | P4 pass | reason |
|---|---|---:|---|
| FC1-CUDAExtensionLayerBackwardMinimal | not_implemented | 0 | CUDA extension not implemented |
| FC2-CUDAExtensionLayerBackwardFull | not_implemented | 0 | CUDA extension not implemented |
| FC3-CUDAExtensionTwoLayerStreaming | not_implemented | 0 | CUDA extension not implemented |
| FC4-CUDAExtensionOneBufferD2 | not_implemented | 0 | CUDA extension not implemented |
| FC5-CUDAExtensionForwardBackwardPair | not_implemented | 0 | CUDA extension not implemented |
| FC6-CUDAGraphStaticShapeRetry | not_run | 0 | CUDA graph capture failed |

FC6 failure：

```text
cuda_graph_capture_failed:RuntimeError:Cannot prepare for replay during capturing stage...
```

判断：本轮没有 CUDA extension / persistent implementation 可以关闭 FC-D2 P4；没有把未实现分支写成 measured success。

## 7. P4 FC-D2 viability decision

P4 artifact：

```text
p4_fc_d2_viability_decision.csv
```

Decision：

```text
fc_d2_continue_allowed = 0
fc_purekan_redesign_required = 1
kanconv_pivot_allowed = 0
purekanformer_pivot_allowed = 0
best_fc_candidate = D2-G2-current-repeat
fc_primary_blocker = CUDA_extension_or_persistent_kernel_not_available_and_G2_backward_still_above_1.50
fc_next_action = enter_FC_PureKAN_primitive_redesign
```

判断：FC-D2 当前 lowering 没有闭合；但根据 v9.2.5 收紧规则，本轮不 pivot Conv / Former，而进入 FC-PureKAN redesign gate。

## 8. P5 FC-PureKAN primitive redesign gate

P5 artifact：

```text
p5_fc_purekan_redesign_gate.csv
```

实测 redesign rows：

| candidate | depth | basis | pairwise R2 | forward | backward | step | memory | P4 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| R1-GEMMNativeEdgeBasisDense-T2 | 1 | t2 | `-0.003503` | `0.949978` | `0.939947` | `1.003011` | `6.656683` | 0 |
| R2-GEMMNativeEdgeBasisDense-T2T3 | 1 | t2t3 | `-0.013125` | `1.010629` | `1.136037` | `1.067930` | `5.893132` | 0 |
| R4-ChebyshevT2-CheapDerivative | 1 | t2 | `-0.003503` | `0.949159` | `0.989850` | `0.978669` | `6.656683` | 0 |
| R5-LegendreP2P3-CheapDerivative | 1 | legendre23 | `-0.013230` | `1.032165` | `1.063783` | `1.032013` | `5.893132` | 0 |
| R7-InteractionRetainingLowRankD2 | 2 | ChebyshevT2 | `0.991121` | `1.400478` | `2.047977` | `7.639727` | `1.046065` | 0 |

未实现 rows：

```text
R3-GEMMNativeDepth2-TiedBasis = not_implemented
R6-EdgeCoefficientBlockFactorized = not_implemented
```

判断：

1. R1/R2/R4/R5 证明 dense edge-basis GEMM-native 单层可以很快，但 pairwise interaction 明显失败，且 memory ratio 因与小 MLP-match 对比而大幅超 gate。
2. R7 证明 D2 interaction 仍保留，但系统 gate 仍失败。
3. 因此没有 FC-PureKAN redesign candidate 同时满足 interaction + GradPass + P4。

## 9. P6/P7 downstream

P6 AdamW-only trainability：

```text
p6_adamw_trainability_reentry.csv = not_run
reason = no_FC_D2_or_FC_redesign_P4_pass_candidate
```

P7 functional open：

```text
functional_open_allowed = 0
reason = P4_not_closed
```

判断：没有用 AdamW trainability 或 functional update 越过 P4。

## 10. No-fake audit

No-fake audit：

```text
rows_checked = 37
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

说明：

1. Conv / Former 只作为 deferred registry，不产生 measured candidate。
2. FC1-FC5 是明确 `not_implemented`，FC6 是真实 capture failure。
3. R1/R2/R4/R5/R7 是真实测量；R3/R6 是明确 `not_implemented`。
4. P6/P7 是明确 gate-blocked `not_run`。

## 11. Hash

| artifact | SHA256 |
|---|---|
| v9.2.5 plan | `1f1a57b532f1e6017ada886b64016274d3b4fc5d58b5459e5b29eb262353304c` |
| `experiments/run_v925_fc_purekan_first.py` | `12c01d97057e909c3d3371c160e372b1d531e10b6036bbf0bc3934ed533bef8c` |
| route | `fe579e496530b0e75d0127f10c26d089572dcc414cbda847b418f4e01a2c7806` |
| `contract_audit_v925_fc_only.csv` | `fcff5e61effd8a6c5c2436629aa34f9bdc2ed9c1d3f373952be21121437455f7` |
| `deferred_architecture_registry_v925.csv` | `d3c85ced739f7e709d939386485fec1df79e2b38c28f53476e2cc1b3d9255bbe` |
| `p1_v924_boundary_repeat.csv` | `47471694938d668556e3ba0ab070aff27449e07a92e228dd76803dfc4ae29a53` |
| `p2_fc_d2_final_lowering_audit.csv` | `aa94ded4ae05eb0c91faabbdd2a529a8cbfcd602b6d64d0856e5e78d7eaaccf3` |
| `p3_fc_d2_cuda_persistent_attempt.csv` | `bc84e230ab66fa6c130f6f471c5c1224159fec05bf9151fc5fe9b1261b1be3aa` |
| `p4_fc_d2_viability_decision.csv` | `27bf05db1bba5b038beda056ebb083f66bee5d377d86298699e03c3eae6d9015` |
| `p5_fc_purekan_redesign_gate.csv` | `63785bacee4b7d9741a2e591e3f012dd469b2fd29b0f6a42d3c592779db844c8` |
| provenance audit | `b0e11bf51a55fa4c7e4e38cd129c6f8c4b7778246ccbf35712084a082bc9fc1e` |

## 12. 最终分析结论

v9.2.5 没有让 FC-PureKAN 比肩 MLP，但把路线收紧后的边界说清楚了：

```text
PureKANConv / PureKANFormer 不进入 active route；
当前 FC-D2 保留 interaction，但系统 P4 未闭合；
单层 dense edge-basis GEMM-native 很快，但不能表达 pairwise interaction；
本轮没有任何 FC-PureKAN candidate 同时满足 interaction + P4。
```

机制判断：

1. FC-D2 的核心价值仍是 interaction retention：R7 的 pairwise R2 为 `0.9911209941`。
2. FC-D2 的核心问题仍是系统 lowering：backward `2.047977`，forward `1.400478`。
3. 单层 dense edge-basis路线虽然 forward/backward 很快，但 pairwise R2 为负，说明它回到了 additive edge model，不是有效替代。
4. CUDA extension / real persistent layer backward 仍是未完成实现，不能在本轮作为成功依据。
5. 根据 v9.2.5 收紧规则，下一步仍应在 FC-PureKAN 内做 primitive redesign，例如 interaction-retaining 但 GEMM-native 的 depth2 / block-factorized / cheaper-derivative edge primitive；Conv / Former 继续 deferred。

最终一句话：

> v9.2.5 真实执行后停在 `R8-FCPureKANSystemNotClosed`：FC-D2 有 interaction 但 P4 不闭合，单层 GEMM-native edge-basis 有速度但丢 interaction；PureKANConv / PureKANFormer 按计划继续 deferred，P6/P7 不打开。
