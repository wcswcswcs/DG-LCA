# DG-KAN v9.2.2 FusedCompositionalFullEdge KernelClosure 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.2_FusedCompositionalFullEdgeKernelClosure_补充实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P3/P7 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.2 fused compositional kernel closure 执行到一个可审计 terminal route：

```text
route = R2-CompositionalKernelRequired
best_candidate = D2-FusedCompositional-T2
best_kernel_path = FC1-FC2-FC3-compiled-active-basis-projection
success_v922_kernel_closure = false
success_v922_trainability_repair = false
success_v922_functional_opened = false
success_v922_external_fair_opened = false
```

最终 artifact：

```text
results/real_rerun_20260506/v922_fused_compositional_kernel_closure_h512r4_20260509T161500Z/
```

核心结论：

1. P0 复用了 v9.2.2 已真实测过的 current route、D2 synthetic interaction success 与 generic D2/D3 P4 fail rows，并保留 source tracking。
2. P1 对 generic D2 做了 coarse failure attribution：step time `8.298989 ms`，coarse dominant phase 记为 `kernel_launch_overhead`，`unknown_time_fraction = 0.0`。
3. P2 新增 D2 三层 active Chebyshev T2 compiled path，FC1/FC2/FC3 的 Grad correctness 通过：`GradRelErrMax = 9.247307e-06`，`GradCosMin = 1.0`。
4. P2 synthetic interaction retention 保留：pairwise-product `R2 = 0.9911209941`，仍高于 `0.95`。
5. 但 P2 P4 kernel-native gate 未过：forward ratio `1.412113`、backward ratio `2.072152`、memory ratio `1.213611` 均超阈值；step ratio `1.387424` 和 FLOPs ratio 过。
6. 因没有 `GradPass + P4 pass + synthetic retention` 的 P2 survivor，P3 AdamW-only trainability、P4 Pareto、P5 patch source、P6 scale confirmation、P7 functional open 均未打开。

最终判断：

```text
fused D2 compiled correctness = pass
synthetic interaction retained = pass
P4 kernel-native gate = fail
P3 AdamW trainability = not_run
P7 functional = not_opened
```

## 1. 本轮代码

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v922_fused_compositional_kernel_closure.py` | v9.2.2 fused compositional kernel closure runner；生成 P0-P7 required artifacts、route、failure table、no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v922_fused_compositional_kernel_closure.py
```

已通过。

## 2. 运行命令

```bash
python experiments/run_v922_fused_compositional_kernel_closure.py \
  --out-dir results/real_rerun_20260506/v922_fused_compositional_kernel_closure_h512r4_20260509T161500Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --batch-size 128 \
  --eval-batch-size 512 \
  --hidden-dim 512 \
  --rank 4 \
  --lr 0.0005 \
  --p1-profile-reps 5 \
  --p4-batch-size 128 \
  --p4-warmup 5 \
  --p4-reps 20 \
  --p3-datasets MNIST,Fashion-MNIST,KMNIST \
  --p3-seeds 0,1,2 \
  --p3-train-size 9984 \
  --p3-test-size 2000 \
  --p3-epochs 20
```

## 3. Route

`route_decision.json`：

```json
{
  "route": "R2-CompositionalKernelRequired",
  "best_candidate": "D2-FusedCompositional-T2",
  "best_kernel_path": "FC1-FC2-FC3-compiled-active-basis-projection",
  "best_depth": "D2",
  "best_hidden_dim": 512,
  "best_rank": 4,
  "best_basis_channels": "T2",
  "full_edge_equivalence_pass": 1,
  "no_external_residual_pass": 1,
  "p4_kernel_native_pass": 0,
  "synthetic_interaction_retained": 1,
  "synthetic_pairwise_R2": 0.9911209940910339,
  "p5_near_pass": 0,
  "p5_pass": 0,
  "functional_open_allowed": 0,
  "primary_blocker": "compiled_fused_D2_grad_and_synthetic_retention_available_but_P4_kernel_native_gate_failed",
  "next_required_implementation": "implement_lower_level_triton_or_streaming_grad_compositional_kernel"
}
```

判断：本轮没有达到 minimum success。虽然 fused D2 的 correctness 和 synthetic retention 成立，但 P4 不过，因此不能打开 P3/P7。

## 4. P0/P1 baseline and attribution

P0 复用 source artifacts：

```text
results/real_rerun_20260506/v922_compositional_trainability_first_20260509T153000Z/
```

关键 baseline：

| item | value |
|---|---:|
| previous route | `R2-CompositionalFullEdgeRequired` |
| D2 pairwise synthetic R2 | `0.9911209941` |
| generic D2 P4 pass | `0` |
| generic D3 P4 pass | `0` |
| K0 previous P4 pass | `1` |

P1 coarse attribution for generic D2:

| metric | value |
|---|---:|
| step time | `8.298989 ms` |
| forward time | `0.483501 ms` |
| backward/update residual time | `7.815488 ms` |
| activation/cache estimate | `10.058807 MB` |
| unknown time fraction | `0.0` |
| dominant phase | `kernel_launch_overhead` |

说明：P1 只做 coarse phase attribution；没有伪造 torch profiler kernel count，`kernel_count_total` 明确写为 `not_measured_torch_profiler_unavailable`。

## 5. P2 fused kernel correctness

P2 candidates：

| candidate | status |
|---|---|
| FC0-GenericReference | reference only，复用 previous generic D2 P4 fail |
| FC1-LayerwiseTorchCompile | measured |
| FC2-RecomputeBackward | measured as same compiled implementation family |
| FC3-FusedBasisProjection | measured as same compiled implementation family |
| FC4-TritonLayer1 | not implemented |
| FC5-TritonLayer1Layer2 | not implemented |
| FC6-CheckpointedComposition | not implemented |
| FC7-StreamingGradComposition | not implemented |

FC1/FC2/FC3 correctness：

| metric | value |
|---|---:|
| hidden / rank | `512 / 4` |
| basis channel | `T2` |
| GradRelErrMax | `9.247307e-06` |
| GradCosMin | `1.000000` |
| OutputAbsDiffMax | `0.000128508` |
| ParamGradAbsDiffMax | `0.000001673` |
| GradPass | `1` |

判断：

1. 这排除了 “compiled D2 backward 明显错” 这个解释。
2. FC4-FC7 未实现，没有伪装成已测；下一步若继续 kernel closure，应优先实现 Triton / streaming grad，而不是把 FC1-3 的 P4 fail 写成成功。

## 6. P2 synthetic retention

Synthetic retention 复用 v9.2.2 真实 measured synthetic rows，并由本轮 compiled correctness 连接到 fused path：

| target | D2 R2 |
|---|---:|
| additive | `0.9990290403` |
| pairwise-product | `0.9911209941` |
| local-xor | `0.6195937395` |
| composition | `0.9816548824` |

判断：H3 成立。P4 修复尝试没有把 D2 退化成 additive-only；pairwise-product R2 仍超过 `0.95`。

## 7. P2 P4 kernel-native gate

FC1/FC2/FC3 P4 metrics：

| metric | value | gate |
|---|---:|---|
| params ratio vs MLP-match | `1.000624` | pass |
| forward ratio vs MLP-match | `1.412113` | fail |
| backward ratio vs MLP-match | `2.072152` | fail |
| step ratio vs MLP-match | `1.387424` | pass |
| memory ratio vs MLP-match | `1.213611` | fail |
| forward FLOPs ratio | `1.007275` | pass |
| backward FLOPs ratio | `1.003949` | pass |
| materializes dense edge tensor layer1/2 | `0 / 0` | pass |
| P4 kernel-native pass | `0` | fail |

Raw timing:

| metric | KAN | MLP-match |
|---|---:|---:|
| forward ms | `0.066326` | `0.046970` |
| backward ms | `0.103482` | `0.049939` |
| step ms | `0.292904` | `0.211114` |
| peak memory MB | `10.058807` | `8.288330` |

判断：

1. Compared to previous generic D2 forward ratio `28.230764`, compiled active path 大幅降低了 forward overhead。
2. 但 P4 仍未闭合：forward、backward、memory 三项超线。
3. FLOPs 接近 1，说明当前失败主要仍是 kernel/runtime/memory live-set，而不是理论 compute 超额。

## 8. Downstream stages

由于 P2 没有 survivor：

| artifact | status | reason |
|---|---|---|
| `p3_compositional_adamw_trainability.csv` | `not_run` | `no_P2_survivor_with_GradPass_P4_pass_and_synthetic_pairwise_retention` |
| `p3_compositional_trainability_trace.csv` | `not_run` | same |
| `p4_depth_width_rank_pareto.csv` | `not_run` | `no_fused_compositional_P4_survivor` |
| `p5_fixed_patch_source_diagnostic.csv` | `not_run` | `no_fused_compositional_P4_survivor` |
| `p6_margin_scale_confirmation.csv` | `not_run` | `no_fused_compositional_P4_survivor` |
| `p7_functional_open_decision.csv` | `functional_open_allowed = 0` | `P5_near_pass_required_before_functional` |

判断：没有把 P3/P7 未打开写成通过，也没有用 functional update 越过 P4/P5 gate。

## 9. No-fake audit

```text
rows_checked = 36
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

说明：

1. P0/P2 synthetic rows 中的复用都记录了 `source_artifact`。
2. P3-P6 是明确 `not_run` artifact，不是 fake/proxy。
3. FC4-FC7 未实现，被记录为 `not_implemented`，没有写成 measured pass。

## 10. Hash

| artifact | SHA256 |
|---|---|
| v9.2.2 fused plan | `210caa89f9452ad49c3b9e1480fedaed1c808235d8642734bd592273b8d65fa2` |
| `experiments/run_v922_fused_compositional_kernel_closure.py` | `d286576d1778f43335c3ce514b229d9babce97308cddb109da9140c27fc3035a` |
| route | `a4abf6414c49553af0b19bd94ee43069226d744556d38222eddab9b6fb64d8c7` |
| `p1_compositional_p4_failure_attribution.csv` | `94d7b0c33dd995a8648ec1706047e5d0498f7026e9933ece9f2de622b3599822` |
| `p2_fused_compositional_kernel_correctness.csv` | `69992189d8cc60b9e3c6f0d3259eed569369bd56204c14919d99e24edf95b941` |
| `p2_fused_compositional_kernel_p4.csv` | `7b96f39fcc01421497804f3ecf355cd5abc9500376cec53a79d0f7c023f3b340` |
| `p2_synthetic_interaction_retention.csv` | `26f7170e5a6af29ae445364a9372be39db1efb87596fc1018ff9a86cbddc052b` |
| provenance audit | `523f5055972a9fb3a741a4fa427ab14eddae5a1364c897076fec2e20e0e7672f` |

## 11. 最终分析结论

本轮 v9.2.2 fused closure 没有修复 compositional P4，但把边界推进了一步：

```text
generic D2 P4 fail -> compiled/fused D2 correctness pass + synthetic retention pass -> P4 still fail
```

机制判断：

1. D2 compositionality 仍然是必要方向：pairwise-product R2 保持 `0.9911209941`。
2. compiled active T2 path 显著降低了 generic D2 的 forward overhead，但还不足以进入 P4 envelope。
3. P4 fail 的主要数字是 forward `1.412x`、backward `2.072x`、memory `1.214x`；step `1.387x` 已经进入阈值。
4. 由于 FLOPs ratio 约 `1.00x`，下一步不是换数学 basis，而是更低层 kernel/memory repair：Triton layer fusion、streaming grad、activation live-set reduction。
5. 当前仍不能声明 trainability repair、functional opened 或 external fair opened。

最终一句话：

> v9.2.2 fused closure 真实执行后停在 `R2-CompositionalKernelRequired`：D2 compiled path 的 grad correctness 与 synthetic interaction retention 成立，但 P4 kernel-native gate 仍失败，因此 P3 AdamW trainability 和 P7 functional update 不能打开。
