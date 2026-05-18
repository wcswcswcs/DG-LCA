# DG-KAN v9.2.2 PureFullEdge CompositionalTrainability 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.2_PureFullEdge_CompositionalTrainability_实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P4/P5/P7 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.2 执行到一个可审计 terminal route：

```text
route = R2-CompositionalFullEdgeRequired
best_candidate = S3-B2-P4-F0
best_depth = D2
success_v922_trainability_repair = false
success_v922_functional_opened = false
success_v922_external_fair_opened = false
```

最终 artifact：

```text
results/real_rerun_20260506/v922_compositional_trainability_first_20260509T153000Z/
```

核心结论：

1. v9.2.1 的 P5 failure 与 margin/logit pathology 被纳入 v9.2.2 审计：K0 lr0.002 与 lr0.0005 rows 复用真实 source artifact，并保留 source tracking。
2. Synthetic interaction 诊断给出强信号：depth1 在 pairwise-product target 上 `R2=-0.187756`，depth2 达到 `R2=0.991121`，差值 `+1.178877`。
3. 这说明 depth/compositionality 是真实方向，不是继续单层 active-k/rank 修补就能解决。
4. 但本轮 D1/D2/D3 generic compositional FullEdge vision candidates 全部 P4 kernel-native gate fail，`compositional_p4_kernel_native_pass_count = 0`，因此 P5 trainability 不打开。
5. K0 本身仍保留 v9.2/v9.2.1 已测 P4 pass，`k0_p4_kernel_native_pass = 1`；当前 blocker 是 compositional FullEdge kernel-native path 尚未闭合。
6. P7 functional 没有打开；没有用 functional update 去救 P5 failed base。

## 1. 本轮新增代码

| 文件 | 作用 |
|---|---|
| `experiments/run_v922_compositional_trainability.py` | v9.2.2 runner；生成 P0-P7 required artifacts、route、failure table、no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v922_compositional_trainability.py
```

已通过。

## 2. 运行命令

```bash
python experiments/run_v922_compositional_trainability.py \
  --out-dir results/real_rerun_20260506/v922_compositional_trainability_first_20260509T153000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --synthetic-input-dim 8 \
  --synthetic-hidden-dim 32 \
  --synthetic-rank 4 \
  --synthetic-train-size 2048 \
  --synthetic-test-size 1024 \
  --synthetic-steps 800 \
  --synthetic-batch-size 256 \
  --synthetic-lr 0.001 \
  --p3-hidden-dim 512 \
  --p3-rank 4 \
  --p4-batch-size 128 \
  --p4-warmup 5 \
  --p4-reps 20
```

## 3. Route

`route_decision.json`：

```json
{
  "route": "R2-CompositionalFullEdgeRequired",
  "best_candidate": "S3-B2-P4-F0",
  "best_depth": "D2",
  "best_source": "S3",
  "best_basis": "B2",
  "best_correction_channel": "T2",
  "full_edge_equivalence_pass": 1,
  "no_external_residual_pass": 1,
  "k0_p4_kernel_native_pass": 1,
  "compositional_p4_kernel_native_pass_count": 0,
  "p4_kernel_native_pass": 1,
  "p5_near_pass": 0,
  "p5_pass": 0,
  "interaction_deficit_confirmed": 1,
  "margin_pathology_confirmed": 1,
  "patch_source_evidence": 0,
  "functional_open_allowed": 0,
  "primary_blocker": "synthetic_interaction_depth2_improves_but_compositional_vision_candidates_break_P4",
  "next_required_implementation": "implement_fused_compositional_full_edge_kernel_then_rerun_P3_P7",
  "success_v922_trainability_repair": 0,
  "success_v922_functional_opened": 0,
  "success_v922_external_fair_opened": 0
}
```

判断：

1. v9.2.2 没有达到 minimum success，因为没有 P5 near-pass candidate。
2. 机制判断从 v9.2.1 的 `R8-OptimizerSanityNotEnough` 推进到：compositionality 确实有必要，但 compositional candidate 的 P4 kernel-native implementation 不足。
3. 当前不能打开 functional 或 external fair。

## 4. P0 contract / equivalence

P0 artifact：

```text
contract_equivalence_audit_v922.csv
```

所有候选均按 PureKAN 边界记录：

```text
loss_type = CE
label_smoothing = 0
external_teacher_used = 0
self_teacher_used = 0
uses_loss_backward = 0
full_edge_equivalence_pass = 1
each_layer_full_edge_equivalence_pass = 1
ordinary_mlp_hidden_activation_used = 0
external_residual_shortcut_used = 0
ordinary_linear_skip_used = 0
trainable_preprocessor_used = 0
hidden_activation_introduced = 0
```

判断：本轮没有把普通 MLP hidden path、external residual shortcut 或 trainable Conv/Linear preprocessor 作为 official route。

## 5. P1 failure / margin diagnostics

P1 artifact：

```text
p1_p5_failure_interaction_margin_diagnostics.csv
p1_loss_margin_logit_trace.csv
```

说明：P1 复用 v9.2.1 的真实 rerun rows：

```text
K0-lr002 source = results/real_rerun_20260506/v921_p1_reproduce_T2_h4096r4_e20_diag_20260509T131500Z/
K0-lr0005 source = results/real_rerun_20260506/v921_p5_lr0005_T2_h4096r4_e20_diag_20260509T141500Z/
```

关键事实仍成立：

| variant | macro KAN | macro MLP | macro delta | near-pass rows |
|---|---:|---:|---:|---:|
| K0 lr0.002 | `0.786056` | `0.886444` | `-0.100389` | `0/9` |
| K0 lr0.0005 | `0.860333` | `0.888556` | `-0.028222` | `0/9` |

判断：

1. P5 failure 已经复现，不是 metric artifact。
2. lr0.0005 明显改善 CE tail 与 macro gap，但仍没到 near-pass。
3. 因此 margin/scale repair 是必要但不足。

## 6. P2 synthetic interaction diagnostics

P2 artifact：

```text
p2_synthetic_interaction_diagnostics.csv
```

关键结果：

| target | depth1 R2 | depth2 R2 | depth2 - depth1 |
|---|---:|---:|---:|
| additive | `0.964272` | `0.999029` | `+0.034757` |
| pairwise-product | `-0.187756` | `0.991121` | `+1.178877` |
| local-xor | `-0.006188` | `0.619594` | `+0.625782` |
| composition | `0.824231` | `0.981655` | `+0.157424` |

Interaction score：

| target | depth1 score | depth2 score |
|---|---:|---:|
| additive | `0.000206` | `0.501929` |
| pairwise-product | `0.000109` | `1.595321` |
| local-xor | `0.000052` | `3.041067` |
| composition | `0.000293` | `1.284460` |

判断：

1. H1 成立：depth1 对 additive target 可拟合，但对 pairwise/product interaction 明显不行。
2. depth2 能显著恢复 pairwise/product 与 composition fit，说明 compositional FullEdge 是真实机制方向。
3. 这支持 `interaction_deficit_confirmed = 1`。

## 7. P3 compositional FullEdge P4 gate

P3 artifact：

```text
p3_compositional_full_edge_trainability.csv
p3_compositional_full_edge_trace.csv
```

本轮只打开 P4 gate；P4 不过则 P5 trainability 不打开。

| candidate | depth | P4 pass | forward ratio | backward ratio | step ratio | memory ratio | FLOPs ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| D1-Depth1-generic-diagnostic | D1 | 0 | `17.801375` | `2.118455` | `3.425794` | `1.192878` | `1.008137 / 1.004221` |
| D2-Depth2-IdentityCorrection | D2 | 0 | `28.215177` | `2.603769` | `4.528721` | `1.167056` | `1.007204 / 1.003878` |
| D3-Depth3-LightCorrection | D3 | 0 | `36.002862` | `3.371440` | `5.997697` | `1.154785` | `1.005951 / 1.002887` |

判断：

1. P3 没有跳过：D1/D2/D3 都真实跑了 P4 gate。
2. FLOPs ratio 接近 1，但 wall-clock forward/backward/step 大幅超线，说明问题是 compositional materialization-free path 的 kernelization / launch / graph overhead，而不是理论 FLOPs 超额。
3. D2 synthetic interaction 有用，但 D2 vision candidate 不能进入 P5，因为 P4 fail。
4. 这正是 route `R2-CompositionalFullEdgeRequired` 的含义：需要 compositional FullEdge，但当前 fused/kernel-native implementation 没闭合。

## 8. P4 correction organization / P6 margin-scale

P4 correction organization artifact 复用 v9.2.1 真实 rows：

```text
p4_edge_correction_organization.csv
p4_correction_channel_usage.csv
```

| variant | macro delta vs MLP |
|---|---:|
| C1-T2-only | `-0.100389` |
| C2-T3-only | `-0.113222` |
| C1-T2-rank8 | `-0.107222` |

P6 margin-scale artifact 复用 v9.2.1 真实 rows：

```text
p6_margin_scale_repair.csv
```

| variant | macro delta vs MLP |
|---|---:|
| K0_T2_r4_lr0.002 | `-0.100389` |
| Z_lr0.001_T2_r4 | `-0.047389` |
| Z_lr0.0005_T2_r4 | `-0.028222` |

判断：

1. 单层 correction organization 仍无 gain。
2. scale/margin repair 有 gain，但不够 near-pass。
3. 这支持“需要 architecture/compositional repair，而不是继续单层 active-k/rank/lr 小修”。

## 9. P5 fixed source / P7 functional

P5 fixed source artifact：

```text
p5_fixed_source_patch_diagnostic.csv = not_run
reason = patch/local derivative path for compositional stack not implemented; no proxy source used
```

P7 survivor confirmation：

```text
p7_survivor_confirmation.csv = not_run
reason = no P5 near-pass candidate; functional re-entry not allowed
```

判断：

1. 本轮没有用 proxy patch/local source 伪造 source repair。
2. 没有 P5 near-pass，所以 functional update 不允许打开。

## 10. No-fake audit

No-fake audit：

```text
rows_checked = 821
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

说明：

1. P2 synthetic 是真实训练/拟合 rows，不作为 external success。
2. P3 compositional 是真实 P4 gate rows。
3. P5/P7 是明确 `not_run`，没有伪装为通过。

## 11. Hash

| artifact | SHA256 |
|---|---|
| v9.2.2 plan | `86a38f0146453175752a4ae7aac684b53dd0e8fa8e4f868763f34766d5fa1ebd` |
| `experiments/run_v922_compositional_trainability.py` | `02c19e7b0977c9fb36f88837db7ff9e9f3294240aae031dca74f435f539d6126` |
| route | `d1f3ecf66eb9ad727877d8857f124628e0695e103c3c9ec4d6984d959b7e57ba` |
| P1 diagnostics | `d83ba7fe53baaf5b452f23108f5944281586e92501660c20a09673607d1304a6` |
| P2 synthetic diagnostics | `63fe3a40fd4b5c4acc288816019c803f8912639da8fa7376af0843004a39f467` |
| P3 compositional trainability | `3d640836699e83f14137335e9dc7537ad08816937af706e9885db38b1ff5c3b7` |
| P4 correction organization | `4b007ae9f258187657ba7f1e1f8058f9c6e4cd79b3a2d52ed5f31b9bfa8cf861` |
| P6 margin-scale repair | `c4cc4f91b0e123699bfaf86a54ab2e5da11a3823ade5aac53bde6254160bfa7a` |
| provenance audit | `22071b98407a8dad15f3564830b9f65150732c16e9c00d83ebd369cf2e4b5b5e` |

## 12. 最终分析结论

v9.2.2 没有修复 trainability，但回答了一个关键机制问题：

```text
当前 P5 failure 不是继续调 lr / rank / active channel 就能闭合；
synthetic interaction 明确显示 depth2 compositional FullEdge 能恢复 interaction；
但当前 compositional FullEdge 的 kernel-native path 还没过 P4。
```

机制判断：

1. depth1 FullEdge 对 additive target 能过 `R2=0.964272`，但 pairwise-product 为 `R2=-0.187756`，说明单层/浅层 additive edge path 对 interaction 明显不足。
2. depth2 在 pairwise-product 上达到 `R2=0.991121`，说明 PureKAN composition 不是形式主义，它确实能表达 interaction。
3. D2/D3 vision candidates 的 FLOPs ratio 接近 1，却 forward/step 超线，说明下一步不是再改数学 basis，而是做 fused compositional FullEdge kernel。
4. K0 P4 仍然成立，但 K0 P5 不成立；D2 机制上更对，但 P4 不成立，所以不能进入 P5/P7。
5. 当前正确下一步是实现 fused compositional FullEdge forward/backward/update，再重跑 P3/P7；不能退回 transitional route，也不能用 functional update 越过 P5。

最终一句话：

> v9.2.2 真实执行后停在 `R2-CompositionalFullEdgeRequired`：depth2 在 synthetic interaction 上把 pairwise R2 从 `-0.187756` 提到 `0.991121`，证明 compositionality 是必要方向；但 D1/D2/D3 compositional vision candidates 均未通过 P4 kernel-native gate，因此 P5/P7 不能打开。
