# DG-KAN v9.2.1 PureFullEdge EdgeBasisChannel TrainabilityRepair 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.1_PureFullEdge_EdgeBasisChannel_TrainabilityRepair_更新计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P6/P7 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.1 继续执行到一个可审计 terminal route：

```text
route = R8-OptimizerSanityNotEnough
best_candidate = S3-B2-P4-F0
best_correction_channel_variant = Z_lr0005_T2_r4
success_v921_trainability_repair = false
success_v921_functional_opened = false
success_v921_external_fair_opened = false
```

最终 summary artifact：

```text
results/real_rerun_20260506/v921_trainability_repair_summary_20260509T150000Z/
```

核心结论：

1. v9.2 的 P5 failure 被真实复现：`S3-B2-P4-F0` 在 T2/rank4/lr0.002 下三任务三 seed 仍是 `0/9` near-pass，macro delta vs MLP-match 为 `-0.100389`。
2. 新增 CE tail / margin / logit 诊断后，确认 failure 不是记录错误：K0 repeat 的 mean CEp99 为 `82.916358`，mean train loss head2048 为 `1.875384`，wrong confidence p95 多数为 `1.0`，margin p10 为负。
3. 容量修复没有成功：T3/rank4 与 T2/rank8 都保持 P4 pass，但 P5 仍 `0/9` near-pass，macro delta 分别为 `-0.113222` 与 `-0.107222`。
4. 小型 lr sanity 有显著改善但未达到 near-pass：T2/rank4/lr0.0005 保持 P4 pass，macro delta 改善到 `-0.028222`，mean CEp99 降到 `9.676426`，但仍是 `0/9` near-pass。
5. Identity-only diagnostic 真实训练后远低于 MLP-match，macro delta `-0.107778`，并且 P4 kernel-native gate fail；说明当前 flattened-source identity edge-basis channel 不能单独解释为 MLP-equivalent。
6. 因 P6 near-pass 未达到，P7 functional update 没有打开；没有用 functional update 去救 AdamW-only 不成立的 base。

## 1. 本轮代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_v92_basis_source_kernel_gate.py` | P5 诊断新增 CE quantiles、logit norm、correct margin、wrong confidence、train-head metrics |
| `experiments/run_v92_basis_source_kernel_gate.py` | 新增 forced source-basis diagnostic gate，用于 identity-only H2 诊断，不计为自然 P1 survivor |
| `experiments/run_v92_basis_source_kernel_gate.py` | 新增 diagnostic-only P5-after-P4-fail 开关，保留 P4 fail 事实，不计 official survivor |
| `experiments/summarize_v921_trainability_repair.py` | 汇总真实 v9.2.1 rerun rows，生成 required artifacts、route、failure table、hash、no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v92_basis_source_kernel_gate.py
python -m py_compile experiments/summarize_v921_trainability_repair.py experiments/run_v92_basis_source_kernel_gate.py
```

均已通过。

## 2. 本轮真实运行

P1 failure reproduction：

```text
results/real_rerun_20260506/v921_p1_reproduce_T2_h4096r4_e20_diag_20260509T131500Z/
```

P3 capacity repair probes：

```text
results/real_rerun_20260506/v921_p3_capacity_T3_h4096r4_e20_diag_20260509T133000Z/
results/real_rerun_20260506/v921_p3_capacity_T2_h4096r8_e20_diag_20260509T134500Z/
```

P5 lr sanity：

```text
results/real_rerun_20260506/v921_p5_lr001_T2_h4096r4_e20_diag_20260509T140000Z/
results/real_rerun_20260506/v921_p5_lr0005_T2_h4096r4_e20_diag_20260509T141500Z/
```

P2 identity-only diagnostic：

```text
results/real_rerun_20260506/v921_p2_identity_only_S3B0_h4096r4_e20_lr0005_diag_20260509T144500Z/
```

Summary:

```text
results/real_rerun_20260506/v921_trainability_repair_summary_20260509T150000Z/
```

## 3. Route

`route_decision.json`：

```json
{
  "route": "R8-OptimizerSanityNotEnough",
  "best_candidate": "S3-B2-P4-F0",
  "best_source": "S3",
  "best_basis": "B2",
  "best_parameterization": "P4",
  "best_correction_channel_variant": "Z_lr0005_T2_r4",
  "best_init_variant": "lr=0.0005,current_init,current_output_scale",
  "full_edge_equivalence_pass": 1,
  "no_external_residual_pass": 1,
  "lowrank_edge_factorization_pass": 1,
  "p4_kernel_native_pass": 1,
  "p5_failure_reproduced": 1,
  "p5_near_pass": 0,
  "p5_pass": 0,
  "functional_opened": 0,
  "functional_pass": 0,
  "primary_blocker": "lr_scale_reduced_CE_tail_and_macro_gap_but_no_candidate_reached_P5_near_pass",
  "success_v921_trainability_repair": 0,
  "success_v921_functional_opened": 0,
  "success_v921_external_fair_opened": 0
}
```

判断：v9.2.1 没有完成 trainability repair。最佳真实候选仍低于 near-pass 要求：near-pass rows `0/9`，macro delta `-0.028222`，未达到 `>= -0.01`。

## 4. P1 failure reproduction and diagnostics

K0 repeat candidate：

```text
candidate = S3-B2-P4-F0
active channel = Chebyshev T2 / index1
hidden / rank = 4096 / 4
lr = 0.002
epochs = 20
```

P1 result：

| metric | value |
|---|---:|
| P5 near-pass rows | `0/9` |
| macro KAN acc | `0.786056` |
| macro MLP-match acc | `0.886444` |
| macro delta | `-0.100389` |
| mean train loss head2048 | `1.875384` |
| mean CEp99 | `82.916358` |

Per task：

| task | KAN acc | MLP-match acc | delta | CEp99 |
|---|---:|---:|---:|---:|
| MNIST | `0.874000` | `0.952667` | `-0.078667` | `88.900586` |
| Fashion-MNIST | `0.784000` | `0.866500` | `-0.082500` | `36.973736` |
| KMNIST | `0.700167` | `0.840167` | `-0.140000` | `122.874751` |

判断：

1. H0 成立：P5 failure 可复现，不是旧 artifact 偶然失败。
2. H1 部分成立：lr0.002 下 train loss 与 CE tail 明显病态，margin p10 为负，wrong-confidence tail 很高。
3. 但这不是单纯日志错误；test acc gap 和 CE/NLL 方向一致。

## 5. P2 identity channel diagnostic

Identity-only diagnostic：

```text
candidate = S3-B0-P4-F0
basis = B0 identity only
hidden / rank = 4096 / 4
lr = 0.0005
diagnostic override = true
```

结果：

| metric | value |
|---|---:|
| P4 kernel-native pass | `0` |
| forward ratio | `10.365180` |
| backward ratio | `2.721662` |
| step ratio | `3.672943` |
| macro KAN acc | `0.780778` |
| macro delta vs MLP-match | `-0.107778` |
| mean CEp99 | `9.560137` |

Per task：

| task | KAN acc | delta vs MLP-match |
|---|---:|---:|
| MNIST | `0.868833` | `-0.088333` |
| Fashion-MNIST | `0.833167` | `-0.036333` |
| KMNIST | `0.640333` | `-0.198667` |

判断：

1. Identity-only 可以 512-sample overfit，但 full-task trainability 远低于 MLP-match。
2. Identity-only 同时 P4 fail，因此不能作为 official candidate。
3. H2 的负面证据成立：当前 flattened-source identity edge-basis channel 不是 MLP-equivalent；P5 failure 不能只归咎于 nonlinear correction。

## 6. P3 capacity repair

| variant | P4 pass | macro KAN | macro MLP | macro delta | near-pass rows | mean CEp99 |
|---|---:|---:|---:|---:|---:|---:|
| T2/rank4/lr0.002 | 1 | `0.786056` | `0.886444` | `-0.100389` | 0 | `82.916358` |
| T3/rank4/lr0.002 | 1 | `0.773222` | `0.886444` | `-0.113222` | 0 | `77.884044` |
| T2/rank8/lr0.002 | 1 | `0.777944` | `0.885167` | `-0.107222` | 0 | `91.583834` |

判断：

1. T3 channel 和 rank8 都没有破坏 P4，但也没有带来 trainability gain。
2. H4 在本轮未被支持：单纯换 active channel 或增加 rank 不是有效 repair。
3. 这些结果支持 `F10_capacity_repair_no_task_gain`，不是 P4 gate 过窄导致的单点误判。

## 7. P5 lr / update-scale sanity

| variant | P4 pass | macro KAN | macro MLP | macro delta | near-pass rows | mean train loss | mean CEp99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| lr0.002 | 1 | `0.786056` | `0.886444` | `-0.100389` | 0 | `1.875384` | `82.916358` |
| lr0.001 | 1 | `0.842722` | `0.890111` | `-0.047389` | 0 | `0.248888` | `20.656453` |
| lr0.0005 | 1 | `0.860333` | `0.888556` | `-0.028222` | 0 | `0.048667` | `9.676426` |

Best lr0.0005 per row：

| task | seed | KAN acc | MLP-match acc | delta |
|---|---:|---:|---:|---:|
| MNIST | 0 | `0.938000` | `0.957000` | `-0.019000` |
| MNIST | 1 | `0.935000` | `0.957500` | `-0.022500` |
| MNIST | 2 | `0.938500` | `0.957000` | `-0.018500` |
| Fashion-MNIST | 0 | `0.844000` | `0.874000` | `-0.030000` |
| Fashion-MNIST | 1 | `0.847000` | `0.864000` | `-0.017000` |
| Fashion-MNIST | 2 | `0.839500` | `0.870500` | `-0.031000` |
| KMNIST | 0 | `0.789500` | `0.840000` | `-0.050500` |
| KMNIST | 1 | `0.810000` | `0.838000` | `-0.028000` |
| KMNIST | 2 | `0.801500` | `0.839000` | `-0.037500` |

判断：

1. H7 的 lr sanity 有真实机制信号：降低 lr 明显修复 CE tail 与 train loss。
2. 但它没有达到 v9.2.1 near-pass：`0/9` rows 满足 `Acc_KAN >= Acc_MLP - 0.01`，macro delta 仍是 `-0.028222`。
3. 因此 route 不是 success，也不是 functional 可打开状态，而是 `R8-OptimizerSanityNotEnough`。

## 8. P4 source repair / P7 functional

P4 source repair artifact：

```text
p4_source_repair_conditioning.csv = not_run
p4_source_repair_trainability.csv = not_run
reason = patch/local FullEdge derivative and architecture not implemented in current FC runner; no proxy source used
```

P7 functional artifact：

```text
p7_functional_reentry_if_opened.csv = not_run
p7_functional_event_trace_if_opened.csv = not_run
reason = P6_near_pass_failed_so_functional_update_not_opened
```

判断：本轮没有把 fixed patch/local source 用 proxy 方式接进 FC FullEdge，也没有用 functional update 救 P5 failed base。

## 9. No-fake audit

Summary no-fake audit：

```text
rows_checked = 473
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

说明：

1. P1/P2/P3/P5/P6 summary rows 都来自真实 rerun artifacts。
2. P4 source repair 与 P7 functional 是明确 `not_run`，不是 fake/proxy。
3. Identity-only 使用 forced diagnostic override，但 CSV 中保留了 override 与 P4 fail，不计 official survivor。

## 10. Hash

| artifact | SHA256 |
|---|---|
| v9.2.1 plan | `2c53bd45c00b95ef868d9bbc45ebac7a1ad17a636ef855a0cf3aec170111e5e9` |
| `experiments/run_v92_basis_source_kernel_gate.py` | `4ce88dde070795f369487eb16e7e1f0fb5ae7e3480291617136f43c7e9944d76` |
| `experiments/summarize_v921_trainability_repair.py` | `a42dd210aadc4893046a77f20bd5ffce10d96bb9e2c4054b6186be94661e16b9` |
| route | `39c8dfa64a72067ff08bd94e69dec8629702c4acd615e2cc989e7d0a86341ba9` |
| P1 reproduction diagnostics | `dac78024f5ebfa27054dbed13e34ab19cbef93c95b1e774130dbe7a5d8fb8cd9` |
| P2 identity/correction ablation | `37377f33385afd6a5d7f9aac93fa07df15ca63ffa5aa6973a0fec08bcc9cb8c3` |
| P3 capacity kernel gate | `158188b52ce68836d73554d8b43ece37a32e2034a44652a8914a25348d654fe3` |
| P3 capacity trainability | `6fe37b18481da574d22b57053629b08232f7ddb95e5d6c20a7f6b7a27f3eb3fe` |
| P5 init/update-scale sanity | `32722dfe75111e4e337212be329a95be069cc2b92106e1023d78513440c6b331` |
| P6 survivor confirmation | `0743f7e9f0d8c352177a12020e2779e66861b2a30cf38fe89c1be35ccf47c794` |
| provenance audit | `297573e43303b9228c74cbf07db7227ba967ae2ba9e7a817fd2cfd18cb7795d3` |

## 11. 最终分析结论

本轮 v9.2.1 没有修复 AdamW-only trainability，但把失败机制推进了一步：

```text
P4 kernel-native gate 已不是当前 blocker；
P5 failure 可复现；
lr/update scale 明显影响 CE tail 和 macro gap；
但合法 PureKAN candidate 仍未达到 P5 near-pass。
```

机制判断：

1. `S3-B2-P4-F0` 的 lr0.002 failure 包含明显 margin/logit pathology；lr0.0005 能把 CEp99 从 `82.916358` 降到 `9.676426`，说明 update scale 是真实因素。
2. 但 lr0.0005 仍低于 MLP-match，尤其 KMNIST gap 仍达到 `-0.038667` macro task gap的一部分；因此不能写 trainability repaired。
3. Identity-only diagnostic 说明当前 flattened FullEdge source/depth 不是简单 MLP-equivalent，不能指望 identity edge-basis 单独闭合 P5。
4. T3/rank4 与 T2/rank8 的 capacity probes 没有 task gain，说明只增加 rank或换单一 active channel 不够。
5. P7 functional update 不允许打开；下一步必须先在 PureKAN contract 下继续修 P5，比如 correction-channel scale/init、真正 fused multi-channel T2T3 path、或 patch-local/KANConv-like source primitive。

最终一句话：

> v9.2.1 已按计划执行到 terminal boundary：`S3-B2-P4-F0` 保持 PureKAN/P4 gate，但 P5 near-pass 未达到。最佳 lr0.0005 把 macro delta 从 `-0.100389` 改到 `-0.028222`，仍是 `0/9` near-pass，因此本轮诚实停在 `R8-OptimizerSanityNotEnough`，不打开 functional/external fair。
