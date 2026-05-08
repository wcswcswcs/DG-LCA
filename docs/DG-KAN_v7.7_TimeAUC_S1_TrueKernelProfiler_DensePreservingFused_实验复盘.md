# DG-KAN v7.7 TimeAUC S1 TrueKernelProfiler DensePreservingFused 实验复盘

> 本复盘记录 `docs/DG-KAN_v7.7_TimeAUC_S1_TrueKernelProfiler_DensePreservingFused_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。未真实测到或未完成 phase mapping 的字段写为 `metric_unavailable` 或不记为 pass。本轮没有做 loss、sampler、classwise、focal、margin 或 class weight 调参。

## 1. 目标是否达成

最终 clean run：

```bash
python experiments/run_gafu_v77_real.py \
  --out-dir results/real_rerun_20260506/v77_timeauc_s1_profiler_m12_clean_10seed_20260506T143000Z \
  --fresh \
  --device auto \
  --candidates B0,B2,M12 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

| 目标 | 结论 | 证据 |
|---|---|---|
| real-only / no-fake / no-proxy | 达成 | `v77_provenance_audit.csv`: `rows_checked=2526`, fake/proxy nonzero `0` |
| StrictPass | 达成 | `M12` strict KAN head，non-KAN trainable params `0` |
| GradPass | 达成 | `M12` gradient `6/6` pass，max relerr `8.07e-05` |
| 10-seed MacroSignificantPass | 达成 | val gap `+0.02057`，CI95 low `+0.01491`，Holm p `1.35e-06`，test gap `+0.01966` |
| Fairness vs MLP-distill | 达成 | `M12` vs `B2` val gap `+0.01517`，NLL delta `-0.02413` |
| FullGridS2Pass | 达成 in clean run | memory max `1.0381`，step max `1.4098`，S2 shapes `9/9` |
| FullGridS1Pass | 未达成 | S1 shapes `0/9`; bs256/512 memory `>1.00`，bs128 step 也 `>1.35` |
| TimeAUCPass | 未达成 | `M12` ValLossAUC_time ratio vs B0 `1.8365` |
| TimeAccountingPass | 未达成 | CUDA sync 未能与 block timing 真实分离，严格 pass rows `0/54` |
| ProfilerPass | 未达成 | torch profiler 测到 CUDA kernel events，但 phase-to-kernel mapping `0`，top3 kernel time only `4.2%-6.7%` |
| v7.7 minimum success | 未达成 | 缺 `ProfilerPass` / `TimeAccountingPass`，且 P0 efficiency reproduction step 超出 v7.6 tolerance |
| v7.7 formal success | 未达成 | 缺 S1、TimeAUC、TimeAccounting、ProfilerPass |

最终判断：

> v7.7 没有达成 minimum 或 formal success。`M12` 的 task/fairness/Grad 信号仍真实稳定，clean run 也恢复了 FullGridS2；但 v7.7 的核心新增目标没有闭合：TimeAUC 仍失败，S1 仍失败，严格时间归因和真实 kernel phase mapping 都没有达到 pass 标准。

## 2. Route Decision

`v77_route_decision.json`：

```json
{
  "route": "R1-TrueKernelProfilerNotClosed",
  "best_candidate_id": "M12",
  "p0_reproduction_pass": 0,
  "strict_pass": 1,
  "grad_pass": 1,
  "macro_significant_pass_10seed": 1,
  "fairness_pass": 1,
  "fullgrid_s2_pass": 1,
  "fullgrid_s1_pass": 0,
  "time_accounting_pass": 0,
  "profiler_pass": 0,
  "time_auc_pass": 0,
  "success_v77_minimum": 0,
  "success_v77_formal": 0,
  "macro_gap": 0.020572916666666666,
  "ci95_low": 0.014908854166666667,
  "holm_p": 1.346826955306228e-06,
  "test_gap": 0.019661458333333333,
  "memory_ratio_mean": 1.0143359939345469,
  "memory_ratio_max": 1.0381213653964307,
  "step_ratio_mean": 1.3701071204598974,
  "step_ratio_max": 1.409765334915835,
  "s2_shape_count": 9,
  "s1_shape_count": 0,
  "val_loss_auc_step_ratio_vs_B0": 0.9319501639698512,
  "val_loss_auc_time_ratio_vs_B0": 1.83648242324758,
  "primary_blocker": "true_cuda_kernel_profiler_unavailable_or_incomplete",
  "no_fake": true,
  "no_proxy": true
}
```

说明：

- `p0_reproduction_pass=0` 不是 task 失败，而是 efficiency reproduction 的 step mean 从 v7.6 clean `1.1683` 漂到 `1.3701`，超过计划容差 `0.10`。
- clean run 仍满足 FullGridS2，但 S1 与 TimeAUC 均未闭合。
- Profiler 有真实 CUDA kernel events，但没有真实 phase-to-kernel mapping，因此不允许记为 `ProfilerPass`。

## 3. Task / Fairness 结果

| candidate | role | val acc | test acc | val gap vs MLP | ECE | NLL |
|---|---|---:|---:|---:|---:|---:|
| `B0` | MLP-AdamW | `0.8449` | `0.7958` | `0.0000` | `0.0619` | `0.5745` |
| `B2` | MLP + C3 logit distill | `0.8503` | `0.8001` | `+0.0054` | `0.0450` | `0.4981` |
| `M12` | PureKAN-NG + C3 logit distill | `0.8655` | `0.8154` | `+0.0206` | `0.0578` | `0.4739` |

Macro significance:

| candidate | mean val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta |
|---|---:|---:|---:|---:|---:|---:|
| `M12` | `+0.02057` | `+0.01491` | `1.35e-06` | `+0.01966` | `-0.00415` | `-0.10059` |
| `B2` | `+0.00540` | `+0.00098` | `0.1216` | `+0.00436` | `-0.01695` | `-0.07646` |

判断：

- `M12` 的 v7.6 task/fairness 结论在 v7.7 clean run 中复现。
- `B2` 仍不能解释 M12 的 advantage；M12 相对 B2 的 macro val gap 仍约 `+0.01517`。

## 4. Gradient Correctness

| candidate | pass rows | max relerr | grad cos min |
|---|---:|---:|---:|
| `M12` | `6/6` | `8.07e-05` | `0.999998` |

判断：GradPass 成立。没有手动改 pass。

## 5. Efficiency / S1

Clean run `M12` full-grid：

| dataset | batch | memory ratio | step ratio | S2 | S1 |
|---|---:|---:|---:|---:|---:|
| MNIST | 128 | `0.9962` | `1.3716` | 1 | 0 |
| MNIST | 256 | `1.0086` | `1.4033` | 1 | 0 |
| MNIST | 512 | `1.0381` | `1.3656` | 1 | 0 |
| Fashion-MNIST | 128 | `0.9962` | `1.3628` | 1 | 0 |
| Fashion-MNIST | 256 | `1.0086` | `1.3123` | 1 | 0 |
| Fashion-MNIST | 512 | `1.0381` | `1.3939` | 1 | 0 |
| KMNIST | 128 | `0.9962` | `1.3760` | 1 | 0 |
| KMNIST | 256 | `1.0086` | `1.4098` | 1 | 0 |
| KMNIST | 512 | `1.0381` | `1.3356` | 1 | 0 |

Summary:

| candidate | memory mean | memory max | step mean | step max | S2 shapes | S1 shapes |
|---|---:|---:|---:|---:|---:|---:|
| `M12` | `1.0143` | `1.0381` | `1.3701` | `1.4098` | `9/9` | `0/9` |

S1 attribution:

| batch | s1 memory gap | root input cache | hidden-y cache | manual cache | optimizer state | cache fraction peak | optimizer fraction peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | `+0.00865` | `0.7656 MB` | `0.1250 MB` | `0.8906 MB` | `0.4521 MB` | `2.66%` | `1.35%` |
| 512 | `+0.03812` | `1.5313 MB` | `0.2500 MB` | `1.7813 MB` | `0.4521 MB` | `5.02%` | `1.27%` |

判断：

- clean run 恢复 FullGridS2，但没有恢复 S1。
- v7.6 clean 的 S1 fail 主要是 memory；v7.7 clean 中 step 均值上漂后，bs128 也因 step `>1.35` 失去 S1。
- bs512 memory gap 可以由 manual cache source 解释，root input cache + hidden-y cache 是 actionable source；optimizer state 约 `1.27%-1.35%`，单独不足以闭合 bs512 S1。

## 6. AUC_step vs AUC_time

| candidate | ValLossAUC_step | step ratio vs B0 | ValLossAUC_time | time ratio vs B0 | 判断 |
|---|---:|---:|---:|---:|---|
| `B0` | `133.6539` | `1.0000` | `0.1615` | `1.0000` | reference |
| `B2` | `130.6872` | `0.9778` | `0.2682` | `1.6609` | runtime fail |
| `M12` | `124.5588` | `0.9320` | `0.2966` | `1.8365` | runtime fail |

判断：

- `M12` 的 step-domain learning curve 明显优于 MLP：ValLossAUC_step ratio `0.9320`。
- Time-domain AUC 明显失败：ValLossAUC_time ratio `1.8365`。
- 因此 v7.7 的 TimeAUC blocker 不是 optimizer dynamics，而是 runtime / loop / teacher / validation / sync accounting 方向。

## 7. Time Accounting

严格口径下，CUDA sync 没有被真实单独分离，因此 `TimeAccountingPass=0`。不把 block timing 中包含的 sync 开销伪装成已分离 sync。

Measured mean timing:

| candidate | teacher mode | train step sec | teacher sec | teacher+validation+logging fraction | unknown fraction | pass rows |
|---|---|---:|---:|---:|---:|---:|
| `B0` | none | `0.001740` | `0.000000` | `0.2192` | `0.0446` | `0/9` |
| `B2` | online | `0.002129` | `0.000496` | `0.3615` | `0.0310` | `0/9` |
| `B2` | offline logits | `0.002156` | `0.000000` | `0.1555` | `0.0326` | `0/9` |
| `M12` | online | `0.002194` | `0.000894` | `0.4616` | `0.0314` | `0/9` |
| `M12` | offline logits | `0.002017` | `0.000000` | `0.2377` | `0.1164` | `0/9` |
| `M12` | no teacher | `0.002481` | `0.000000` | `0.2220` | `0.0405` | `0/9` |

判断：

- `M12` online teacher path 的 teacher + validation/logging fraction 约 `46.2%`，支持 H1：TimeAUC fail 不只是 primitive step fail。
- `M12` offline logits 的 train step 比 online teacher 低，但没有完整 10-seed offline-logit task trace，因此不能 claim official TimeAUCPass。
- CUDA sync 未能严格拆分，按计划不能记 TimeAccountingPass。

## 8. True Kernel Profiler

torch profiler 确实测到 CUDA kernel events，但没有完成 phase-to-kernel mapping，因此 ProfilerPass 不成立。

| dataset | batch | kernel count | unique kernels | top3 kernel time explained | phase mapping | profiler pass |
|---|---:|---:|---:|---:|---:|---:|
| MNIST | 128 | `235` | `45` | `0.0419` | 0 | 0 |
| MNIST | 512 | `229` | `46` | `0.0674` | 0 | 0 |
| KMNIST | 128 | `235` | `45` | `0.0420` | 0 | 0 |
| KMNIST | 512 | `229` | `46` | `0.0674` | 0 | 0 |

Top kernels include GEMM/reduction kernels such as:

```text
ampere_sgemm_64x32_sliced1x4_tn
ampere_sgemm_32x32_sliced1x4_nt
void at::native::reduce_kernel<...>
```

判断：

- 可以说“kernel events 真实测到了”。
- 不能说“ProfilerPass 成立”，因为没有真实 phase-to-kernel mapping。
- top3 kernel time explained 只有 `4.2%-6.7%`，说明时间分散在大量小 kernel 中；但在 phase mapping 完成前，不把它写成完整 kernel bottleneck 结论。

## 9. Offline Teacher 诊断

新增 `p7_offline_teacher_fair_time_v77.csv` 是 derived diagnostic，不是 official full task trace。它只使用已测 `ValLossAUC_step` 与已测 train-step time ratio 做诊断，不作为 TimeAUCPass。

| candidate | mode | train step ratio vs B0 | derived train-only AUC ratio | derived total AUC ratio | official claim |
|---|---|---:|---:|---:|---|
| `B2` | online | `1.2237` | `1.1965` | `1.1518` | no |
| `B2` | offline logits | `1.2390` | `1.2115` | `1.1645` | no |
| `M12` | online | `1.2607` | `1.1749` | `1.2075` | no |
| `M12` | offline logits | `1.1593` | `1.0804` | `1.1280` | no |

判断：

- M12 offline logits 可能让 train-only AUC ratio 进入 `<=1.10` 附近：diagnostic ratio `1.0804`。
- 但 total ratio 仍约 `1.1280`，且没有完整离线 teacher task trace，所以不能记成功。
- 下一步如果继续做系统修复，应实现 official offline-logit training recipe，并同时计入 teacher precompute/storage。

## 10. 失败后自修复

| attempt | run | 结果 | 判断 |
|---|---|---|---|
| first v7.7 full run | `v77_timeauc_s1_profiler_m12_10seed_20260506T140000Z` | task/grad/fairness 复现，但 step max `1.7909`，S2 `7/9` | efficiency profiler 不稳定 |
| same-dir efficiency-only rerun | 同目录复用真实 task rows | memory max `1.0613`，step max `1.7240`，S2 `5/9` | 没有修复，反而暴露 allocator/measurement instability |
| clean fresh rerun | `v77_timeauc_s1_profiler_m12_clean_10seed_20260506T143000Z` | memory max `1.0381`，step max `1.4098`，S2 `9/9` | 恢复 S2，但 S1/TimeAUC/Profiler 仍 fail |
| offline teacher diagnostic | `p7_offline_teacher_fair_time_v77.csv` | M12 offline train-only derived ratio `1.0804` | 有信息量，但不是 official success |
| P5/P6 fused package | `p5_p6_fused_package_status_v77.csv` | `not_implemented` | 没有 P3 phase mapping，不允许盲目写 fused package pass |

这些尝试均为 runtime/profiler/cache 方向，不是 loss 或 classwise 调参。

## 11. No-Fake / No-Proxy 审计

| artifact | value |
|---|---|
| final run | `results/real_rerun_20260506/v77_timeauc_s1_profiler_m12_clean_10seed_20260506T143000Z` |
| rows checked | `2526` |
| fake_proxy_nonzero_count | `0` |
| no_fake | `true` |
| no_proxy | `true` |
| plan path | `docs/DG-KAN_v7.7_TimeAUC_S1_TrueKernelProfiler_DensePreservingFused_完整实验计划.md` |
| script path | `experiments/run_gafu_v77_real.py` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v77_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

关键 hash：

| file | SHA256 |
|---|---|
| `run_manifest.json` | `3a7c520525123d3519de23f2467a9bbe40de468c9ede2e262196491fb324794a` |
| `v77_manifest.json` | `093e8669e3e7b61747cfa9fe2b09cbbd1be59a0283d0926e226b3c5cdca0654f` |
| `v77_route_decision.json` | `06f0d54a16d6d2b090dd6416bd7433d644f9891cdd1edc80e50d073550e63c14` |
| `p9_task_summary.csv` | `bb27b02f1170145254846e173ee6d166971a1fd27ad180247b62f6c69f020e1c` |
| `p1_significance_audit.csv` | `ee53590095cfd62b05ec5ff552e46c13b4f68b7656c1c626960cc2794d243a4c` |
| `p2_full_gradient_correctness.csv` | `a909838d5e66f1ee6a20e83bc0e629e5c6c3cf030758829c131cef8ac3169df0` |
| `p10_efficiency_summary.csv` | `5ad4a4e1c49d4ddb29b4d0f5441b47eeaa00ba9458e27ee3af6bdfb8a1b12417` |
| `p1_time_accounting_v77.csv` | `0509f665d542c106e085551c652263085b8bdd5c4dcbeaa8510516d33c12adee` |
| `p2_auc_dynamics_v77.csv` | `78d3a813fb0a6b82f8b45bec397b4906b31ed905679db83d9426bbb0885b9d78` |
| `p3_true_kernel_profiler_v77.csv` | `ee9c6fd31c03084e98a5904f18242135fdde3b096429f96682d9c5b09d5b13be` |
| `p4_s1_memory_attribution_v77.csv` | `9f942d9e59bb16b0ca6849931e4fc0061e877e685044f0e38e2eaf47ff866829` |
| `p7_offline_teacher_fair_time_v77.csv` | `ef3bd61b5899d5692af5879e5893c01d854f4b5c74b4071c9bd2b897711d3fb0` |
| `v77_provenance_audit.csv` | `a3c3928b13ccb90c45e022b608f30c2ef45da3781918b3684e77d83ff04340a6` |
| `experiments/run_gafu_v77_real.py` | `e48a07762c71d88adf14d494bfb796492228096c4f2fc7e9e2c1d834de56c1f8` |

## 12. 最终结论

v7.7 没有达成目标：

```text
StrictPass = true
GradPass = true
MacroSignificantPass_10seed = true
FairnessPass_vs_MLP_distill = true
FullGridS2Pass = true in clean run
FullGridS1Pass = false
TimeAUCPass = false
TimeAccountingPass = false
ProfilerPass = false
success_v77_minimum = false
success_v77_formal = false
```

机制结论：

1. `M12` 的 task/fairness/Grad 结论没有退化；v7.7 再次确认它不是靠 fake/proxy 或 distillation-only 得到的 macro advantage。
2. `M12` 的 step-domain learning curve 比 MLP 更好：ValLossAUC_step ratio `0.9320`。
3. `M12` 的 time-domain AUC 明显失败：ValLossAUC_time ratio `1.8365`，所以 blocker 是 runtime / loop / teacher / validation / sync，而不是 optimizer dynamics。
4. offline teacher diagnostic 有价值：M12 offline-logit train-only derived ratio `1.0804`，提示下一步值得做 official offline-logit full recipe；但当前不能 claim success。
5. S1 仍未闭合：bs256/512 memory `>1.00`，且本轮 clean step 上漂后 bs128 也未过 S1 step。
6. torch profiler 测到了真实 CUDA kernel events，但没有 phase-to-kernel mapping；按 v7.7 计划，不能把 profiler 记为 pass，也不能基于不完整 mapping 写 fused-kernel 成功。
7. 当前最硬 blocker 是：`true_phase_mapped_kernel_profiler + official offline-teacher/time-accounted recipe + S1 memory/cache trim`。

最终一句话：

> v7.7 失败得更具体了：M12 的学习曲线按 step 看优于 MLP，但系统 wall-clock 输；offline logits 可能能修 train-only time，但还没有 official full trace；S1 仍差 cache/memory margin；真实 profiler 已看到很多 CUDA kernel events，却还缺 phase mapping。因此下一步不应调 loss，而应先实现 phase-mapped profiler/NVTX timeline，再做 official offline-logit recipe 和有根据的 fused/streaming package。
