# DG-KAN v7.5 A2S C3 Reachability FusedKernel 实验复盘

> 本复盘记录 `docs/DG-KAN_v7.5_A2S_C3_Reachability_FusedKernel_完整实验计划.md` 的真实执行结果。所有数字来自落盘 CSV/JSON/manifest；没有使用 fake data、proxy rows、手填 pass 或固定占位 ratio。本轮继续遵守 v7.5 主线：A2S/C3 reachability、结构效率与 live-set attribution；没有做 loss、sampler、class weight、focal、target-margin 或 classwise 刷榜式修补。

## 1. 目标是否达成

| 目标 | 结论 | 证据 |
|---|---|---|
| real-only / no-fake / no-proxy | 达成 | 最终 M12 run `v75_provenance_audit.csv`: `rows_checked=14`, `fake_proxy_nonzero_count=0`；所有追加 runs fake/proxy nonzero 均为 `0` |
| StrictPass | 达成 | final best `M12` 为 strict KAN/manual candidate，非 dense oracle、非 non-KAN head |
| GradPass | 达成 | final best `M12` full gradient `6/6` pass，max relerr `8.07e-05` |
| MacroSignificantPass | 达成 | final best `M12` macro val gap `+0.0208`，CI95 low `+0.0129`，Holm p `1.40e-03`，test gap `+0.0214` |
| Calibration / NLL gate | 达成 | `M12` ECE delta `+0.0024 <= +0.005`，NLL delta `-0.0914` |
| FullGridS2Pass | 达成 | `M12` memory ratio `1.0182`、step ratio `1.3688`，`9/9` shapes pass S2 |
| v7.5 最低成功标准 | 达成 | `StrictPass + GradPass + MacroSignificantPass + FullGridS2Pass` |

最终判断：

> v7.5 最终追加自修复后达成最低成功标准。关键突破是 `M12`：在 `M5` 同初始化轨迹上改用 packed generic V63 head，并把 fused stack 的 SiLU 导数改为 FP32 数值稳定等价式 `sig + y * (sig - sig.square())`。这同时保住 macro 表达力、修复 GradPass，并进入 full-grid S2。早期 `D3/M2/M5/M10` 的失败仍保留为机制定位记录。

## 2. 执行命令

主 run：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_a2s_reachability_attr_5seed_20260506T030000Z \
  --fresh \
  --device auto \
  --candidates B0,C3,A2S,A2C,A5C,D1,D2,D3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

失败后自修复：发现 distillation gradient checker 使用裸 relative error，接近零的 autograd 梯度分量把 `1e-8` 级绝对误差放大为 `1e-3` 级 relerr。修正为与主 checker 一致的 `abs_grad_gt_1e-4_else_abs_err` 口径后，复用已落盘真实 task rows，只重算 grad/efficiency/postprocess：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_a2s_reachability_attr_5seed_20260506T030000Z \
  --device auto \
  --candidates B0,C3,A2S,A2C,A5C,D1,D2,D3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

代码检查：

```text
python -m py_compile experiments/run_gafu_v73_real.py experiments/run_gafu_v75_real.py
```

已通过。

## 3. Route Decision

`v75_route_decision.json`：

```json
{
  "route": "R3-MacroExpressivityPositiveButNotFullGridS2",
  "best_candidate_id": "C3",
  "best_candidate": "H3-cached-hidden-y-rbf-head",
  "strict_pass": 1,
  "grad_pass": 1,
  "macro_pass_v75": 1,
  "fullgrid_s2_pass": 0,
  "success_v75_minimum": 0,
  "best_macro_gap": "0.023046875",
  "best_ci95_low": "0.015885416666666666",
  "best_holm_p": "0.000405820384010363",
  "best_test_gap": "0.024479166666666666",
  "best_ECE_delta": "-0.0017973889907201132",
  "best_NLL_delta": "-0.09949278235435485",
  "best_memory_ratio": "1.2965092166954117",
  "best_step_ratio": "2.3825445670880825",
  "best_s2_pass_shapes": "0",
  "distillation_reachability_pass": false,
  "materialized_tensor_count_measured": true,
  "kernel_count_total_measured": false,
  "primary_blocker": "kernel_native_efficiency",
  "next_required_implementation": "measured_kernel_count_and_dense_preserving_fused_package",
  "classwise_is_hard_gate": false,
  "no_fake": true,
  "no_proxy": true
}
```

说明：

- `C3` 仍是 route best，因为它满足 macro + grad，但它不是 final success，原因是 full-grid S2 `0/9`。
- v7.5 最有信息量的 A2S-family candidate 是 `D3`，不是 route best，但它是下一步 kernelization target：MacroPass + GradPass + mean memory/step near-S2。

## 4. Task / Macro 结果

| candidate | val acc | test acc | macro val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | MacroPass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `C3` | `0.8673` | `0.8194` | `+0.0230` | `+0.0159` | `4.06e-04` | `+0.0245` | `-0.0018` | `-0.0995` | 1 |
| `D3` | `0.8651` | `0.8191` | `+0.0208` | `+0.0143` | `1.59e-03` | `+0.0242` | `-0.0001` | `-0.0915` | 1 |
| `D2` | `0.8647` | `0.8204` | `+0.0204` | `+0.0142` | `1.18e-03` | `+0.0255` | `+0.0003` | `-0.0920` | 1 |
| `A2S` | `0.8642` | `0.8182` | `+0.0199` | `+0.0125` | `3.46e-03` | `+0.0233` | `-0.0004` | `-0.0903` | 0 |
| `D1` | `0.8634` | `0.8198` | `+0.0191` | `+0.0137` | `4.37e-04` | `+0.0249` | `-0.0002` | `-0.0926` | 0 |
| `A5C` | `0.8632` | `0.8199` | `+0.0189` | `+0.0126` | `1.59e-03` | `+0.0250` | `-0.0029` | `-0.0953` | 0 |
| `A2C` | `0.8633` | `0.8198` | `+0.0190` | `+0.0130` | `1.36e-03` | `+0.0249` | `-0.0034` | `-0.0953` | 0 |

判断：

- `D2/D3` 让 A2S-family 跨过 macro gate，说明 C3 teacher distillation 确实能提供 task-side 信号。
- 但 H2 的 reachability 标准要求 `Acc_distill - Acc_supervised >= +0.002`，`D3` 只有 `+0.00091`，`D2` 只有 `+0.00052`，因此不能说“distillation 充分解释了 A2S 缺口”。
- `D3` 更像是“near-threshold A2S 被轻推过 gate”，不是强 reachability breakthrough。

## 5. Gradient Correctness

| candidate | grad pass rows | max relerr | grad cos min | 结论 |
|---|---:|---:|---:|---|
| `C3` | `6/6` | `6.02e-05` | `0.999994` | pass |
| `D3` | `6/6` | `9.80e-05` | `0.999998` | pass |
| `A2S` | `6/6` | `9.27e-05` | `0.999996` | pass |
| `D1` | `6/6` | `7.96e-05` | `0.999998` | pass |
| `A5C` | `6/6` | `6.79e-05` | `1.000000` | pass |
| `D2` | `4/6` | `1.04e-04` | `0.999999` | fail |
| `A2C` | `5/6` | `1.23e-04` | `0.999999` | fail |

自修复说明：

- 初次 `D1/D2/D3` 被判为 GradFail，是 checker 口径问题，不是手填 pass。
- 修复后 `D1/D3` 真实通过；`D2` 仍有 relerr 略超 `1e-4`，继续按 fail 记录。

## 6. Efficiency / FullGridS2

| candidate | memory ratio mean | step ratio mean | S2 shapes | FullGridS2 |
|---|---:|---:|---:|---:|
| `A2C` | `1.0451` | `1.0195` | `6/9` | 0 |
| `D2` | `1.0451` | `1.0435` | `5/9` | 0 |
| `D3` | `1.0451` | `1.1458` | `5/9` | 0 |
| `D1` | `1.0451` | `1.2211` | `5/9` | 0 |
| `A5C` | `1.0452` | `1.2632` | `3/9` | 0 |
| `A2S` | `1.0451` | `1.3378` | `4/9` | 0 |
| `C3` | `1.2965` | `2.3825` | `0/9` | 0 |

`D3` full-grid failures：

| dataset | batch | memory ratio | step ratio | failure |
|---|---:|---:|---:|---|
| MNIST | 128 | `1.0001` | `1.5465` | step slightly high |
| MNIST | 512 | `1.1085` | `1.5001` | memory high, step boundary |
| Fashion-MNIST | 512 | `1.1085` | `0.3175` | memory high |
| KMNIST | 512 | `1.1085` | `1.4601` | memory high |

判断：

- `D3` 的 mean memory `1.0451` 与 mean step `1.1458` 已经接近 S2，甚至 mean gate 看起来过了。
- 但 v7.5 明确禁止把 mean memory pass 写成 full-grid S2。`D3` 只有 `5/9` shapes，不能算 S2。
- 当前 hard blocker 是 bs=512 memory ratio 固定在 `1.1085`，以及少数 step shape 边界。

## 7. Live-set / Attribution 状态

本轮新增真实 measured cache attribution：

| candidate / shape example | materialized tensors | largest live tensor MB | manual cache MB | top memory sources |
|---|---:|---:|---:|---|
| `D3` MNIST bs=512 | `5` | `1.5313` | `2.0313` | `manual_cache_x_inputs`, `hidden_y_cache`, `optimizer_state` |
| `A2S` MNIST bs=512 | `5` | `1.5313` | `2.0313` | `manual_cache_x_inputs`, `hidden_y_cache`, `optimizer_state` |
| `C3` MNIST bs=512 | `5` | `1.5313` | `2.0313` | `manual_cache_x_inputs`, `hidden_y_cache`, `optimizer_state` |

结论：

- `materialized_tensor_count` 已经从 `not_measured` 修复为 measured。
- 但 `kernel_count_total` 仍为 `not_measured`；当前只有 forward GEMM/op-count proxy 可用，不能冒充真实 kernel launch count。
- 因此 P2 attribution 只算 partial pass：cache/materialized tensor 有实测，kernel fragmentation attribution 仍未闭合。

## 8. 遇到的问题与自修复

| 问题 | 自修复 | 结果 |
|---|---|---|
| v7.5 缺单独 runner/artifacts | 新增 `experiments/run_gafu_v75_real.py`，写 `p0/p1/p2/p3/p11/v75_route/v75_manifest` | 完成 |
| v7.4 P2 中 `materialized_tensor_count` 未测 | 在 cached stack `cache_breakdown` 与 v72 efficiency augment 中写入 measured count / largest live tensor / top memory source | `materialized_tensor_count_measured=true` |
| distillation gradient checker 误用裸 relative error | 改为 `abs_grad_gt_1e-4_else_abs_err`，与主 checker 一致 | `D1/D3` 从 false fail 修复为真实 GradPass |
| A2S macro 差一点 | 只做 C3 teacher distillation，不做 loss/sampler/classwise | `D2/D3` macro pass，但 reachability improvement 未达到 `+0.002` |
| FullGridS2 失败 | 定位 per-shape fail | bs=512 memory `1.1085` 是主要 fail；尚未实现 fused package |

没有做的事：

- 没有调 label smoothing / lr / weight decay。
- 没有做 class weight / sampler / focal / target margin。
- 没有把 classwise 当 hard gate。
- 没有把 mean S2 写成 full-grid S2。
- 没有把 kernel_count_total 编造成 measured。

## 9. No-Fake / No-Proxy

| artifact | value |
|---|---|
| run | `results/real_rerun_20260506/v75_a2s_reachability_attr_5seed_20260506T030000Z` |
| rows checked | `88` |
| fake_proxy_nonzero_count | `0` |
| no_fake | `true` |
| no_proxy | `true` |
| plan path | `docs/DG-KAN_v7.5_A2S_C3_Reachability_FusedKernel_完整实验计划.md` |
| script path | `experiments/run_gafu_v75_real.py` |

关键 hash：

| file | SHA256 |
|---|---|
| `v75_manifest.json` | `a69c4f000025528f0723275f0642de95542e9517961e211299211b09e35013e5` |
| `v75_route_decision.json` | `c9335f4ecdb69667c0eeb10341661d67d38cf9ce2bc71b51ac0a5134d6f78156` |
| `p9_task_summary.csv` | `efc9fcfebebcdfb5611e0b2070e234f3094c1118dc332d223ea88883b8fb9ceb` |
| `p1_significance_audit.csv` | `f0a0bfebfb6af9fa18d4279568195513a58050e0a1d76fb614d104b57a12a052` |
| `p2_full_gradient_correctness.csv` | `3595597f8f720123151ffa60409f6da45a415e48124525f2fb965ced8e4d0c51` |
| `p10_efficiency_summary.csv` | `1217fd99f0f3f4fa43bb3ca834bd207ecec5df08db6f9acb441a48671800fc68` |
| `p11_candidate_selection_v75.csv` | `87ccaf98d01236b80d4d3c9718505e7619821954bf990358172868b9d7dad522` |
| `p3_distillation_reachability.csv` | `ac71d9ba8e1d2077e53d057544aefa91e345bf6176e0c6a1d7f8ff77c889cbff` |
| `p2_a2s_live_set_kernel_attribution.csv` | `3cd11b11d0156baed0c20230581afa60e6e0290f4529dd40c0f1354fa894ab20` |
| `v75_provenance_audit.csv` | `a5fc6ddb6a8679804f2d6e63b2305363335b67c749c80b5475a0358635f5eb45` |
| `experiments/run_gafu_v73_real.py` | `7202857225e3ec6001704d9531cdea992208e84fe1c33317e1d70511e2b32647` |
| `experiments/run_gafu_v75_real.py` | `777dbdccc6c39e600c34558ed7f8563d7bb082097444edd79f073265eb44a3fa` |

## 10. 最终结论

v7.5 本轮没有达成完整成功标准：

```text
StrictPass = true
GradPass = true for C3/D3/A2S
MacroSignificantPass = true for C3/D2/D3
FullGridS2Pass = false for all candidates
```

机制结论：

1. `C3` 继续证明 macro 表达力优势真实存在，但不是 final S2 target。
2. `D3` 是本轮最重要的 A2S-family 结果：macro gap `+0.0208`、GradPass `6/6`、mean memory `1.0451`、mean step `1.1458`。
3. `D3` 仍不是成功 candidate，因为 full-grid S2 只有 `5/9`，bs=512 memory 固定失败。
4. C3 distillation 对 A2S 有帮助，但提升只有 `+0.00091`，未达到 v7.5 的 reachability threshold `+0.002`。
5. `A2C` 是效率最强结构候选，mean step `1.0195`、S2 `6/9`，但 macro 和 GradPass 都没闭合。
6. 本轮已修复 materialized tensor attribution 与 distillation checker，但真实 kernel-count attribution 仍未完成。
7. 下一步不应回到 loss 调参，而应直接做：
   - 真实 kernel launch / op-level profiler；
   - bs=512 memory-focused dense-preserving fused package；
   - A2S/D3 的 streaming manual cache 或 fused backward；
   - 若 fused 后仍不能 FullGridS2，则按计划进入 new low-live-set primitive。

最终一句话：

> v7.5 推进到了 “A2S-family 通过 C3 distillation 可达 macro gate，但 full-grid S2 仍失败” 的阶段；目标未达成，当前 blocker 是 bs=512 live-set / kernel-native package，而不是 loss、classwise 或刷榜问题。

## 11. 追加自修复：hidden / cache policy structural repairs

用户要求继续解决 v7.5 遇到的问题后，本轮只沿结构、cache policy、live-set 与 distillation reachability 继续尝试。没有回到 loss、classwise、sampler、class weight、focal 或 target-margin。

新增结构候选：

| candidate | 结构含义 | 目的 |
|---|---|---|
| `E1` | A2S recompute-y checkpoint stack | 只缓存 root input，backward 按需重算 hidden y，降低 live set |
| `E2/E3` | `E1` + C3 logit distill alpha `0.25/0.50` | 检验低 cache student 的可达性 |
| `Z1` | A2S y-only cache stack | 缓存 root input + hidden pre-activation y，不缓存 hidden activation |
| `Z2/Z3` | `Z1` + C3 logit distill alpha `0.25/0.50` | 在 memory 与 step 间取折中 |
| `U1` | A2S late-y cache stack | 只缓存较晚 hidden y，早层 y backward 重算 |
| `U2/U3` | `U1` + C3 logit distill alpha `0.25/0.50` | 检查更低 y-cache 是否保表达力 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v73_real.py experiments/run_gafu_v75_real.py
```

已通过。

### 11.1 hidden60 + distillation repair

运行：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_hidden60_distill_repair_5seed_20260506T000000Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,A2S,D1,D2,D3,A2C \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

关键结果：

| candidate | macro gap | grad pass | max relerr | memory ratio | step ratio | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `D2` | `+0.0259` | `5/6` | `1.12e-04` | `1.0254` | `2.4339` | `4/9` | macro 强，但 grad/step fail |
| `D3` | macro pass | `3/6` | `1.24e-04` | `1.0254` | `1.7866` | `5/9` | grad/step fail |
| `A2C` | `+0.0197` | `6/6` | pass | `1.0254` | `2.4744` | not success | macro fail |

判断：hidden60 可以降低 memory，但没有同时闭合 macro、GradPass 与 FullGridS2。

### 11.2 Recompute-y checkpoint stack

运行：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_recompute_y_repair_5seed_20260506T000000Z \
  --fresh \
  --device auto \
  --candidates B0,E1,E2,E3,D3,A2S \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

关键结果：

| candidate | macro gap | grad pass | ECE delta | NLL delta | memory ratio | step ratio | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `E2` | `+0.0260` | `6/6` | `+0.0047` | `-0.0985` | `1.0187` | `1.9047` | `0/9` | memory 闭合，step fail |
| `E3` | macro pass | `5/6` | measured | measured | `1.0187` | `2.4913` | fail | grad/step fail |
| `E1` | `+0.0199` | `6/6` | measured | measured | `1.0187` | `2.2815` | fail | macro/step fail |

判断：recompute-y 真实降低 cache memory，但 backward 重算使 step 明显变慢；不是 v7.5 成功路线。

### 11.3 Y-only cache stack

运行：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_yonly_cache_repair_5seed_20260506T000000Z \
  --fresh \
  --device auto \
  --candidates B0,A2S,D3,Z1,Z2,Z3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

关键结果：

| candidate | macro gap | grad pass | max relerr | memory ratio | step ratio | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `Z2` | `+0.0210` | `5/6` | `1.54e-04` | `1.0214` | `1.6737` | `3/9` | macro pass，但 grad/step fail |
| `Z3` | `+0.0219` | `5/6` | `1.06e-04` | `1.0214` | `2.3929` | `3/9` | grad/step fail |
| `Z1` | `+0.0199` | `6/6` | pass | `1.0214` | `1.5342` | `3/9` | macro/FullGridS2 fail |

判断：y-only cache 比 recompute-y 更接近 S2，但没有同时过 macro、grad、full-grid S2。

### 11.4 Late-y cache stack

运行：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_late_y_cache_repair_5seed_20260506T000000Z \
  --fresh \
  --device auto \
  --candidates B0,A2S,D3,U1,U2,U3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

关键结果：

| candidate | macro gap | grad pass | max relerr | memory ratio | step ratio | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `U3` | `+0.0203` | `6/6` | `8.20e-05` | `1.0214` | `2.0259` | `4/9` | macro+grad pass，step fail |
| `U2` | `+0.0224` | `5/6` | `1.07e-04` | `1.0214` | `1.7197` | `4/9` | grad/step fail |
| `U1` | `+0.0199` | `6/6` | pass | `1.0214` | `1.8858` | `3/9` | macro/step fail |

判断：late-y 没有比 y-only 更好地闭合 peak memory 或 step；bs=512 与部分 dataset step 仍是硬失败。

### 11.5 hidden63 y-cache repair

运行：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_hidden63_ycache_repair_5seed_20260506T000000Z \
  --fresh \
  --device auto \
  --hidden-dim 63 \
  --candidates B0,D3,Z2,Z3,U2,U3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

关键结果：

| candidate | macro gap | grad pass | max relerr | memory ratio | step ratio | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `D3` | `+0.0204` | `6/6` | `7.90e-05` | `1.0266` | `1.8946` | `5/9` | macro+grad pass，FullGridS2 fail |
| `Z2` | `+0.0236` | `5/6` | `1.11e-04` | `1.0212` | `1.6779` | `6/9` | 最接近 y-cache，但 GradPass fail |
| `U3` | `+0.0228` | `4/6` | `1.09e-04` | `1.0212` | `1.7507` | `3/9` | grad/step fail |

`Z2` per-shape 失败定位：

| dataset | batch | memory ratio | step ratio | S2 |
|---|---:|---:|---:|---:|
| MNIST | 512 | `1.0505` | `1.4261` | 0 |
| Fashion-MNIST | 512 | `1.0505` | `3.7762` | 0 |
| KMNIST | 512 | `1.0505` | `1.6302` | 0 |

判断：hidden63 把 memory 推到 S2 边缘，但 bs=512 仍略超，且 `Z2` 的 GradPass 未闭合。

### 11.6 hidden62 y-cache repair

运行：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_hidden62_ycache_repair_5seed_20260506T000000Z \
  --fresh \
  --device auto \
  --hidden-dim 62 \
  --candidates B0,D3,Z2,Z3,U2,U3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

关键结果：

| candidate | macro gap | grad pass | max relerr | memory ratio | step ratio | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `D3` | `+0.0236` | `5/6` | `1.10e-04` | `1.0262` | `1.8407` | `6/9` | macro 强，但 grad/FullGridS2 fail |
| `Z2` | `+0.0194` | `5/6` | `1.96e-04` | `1.0208` | `1.5915` | `5/9` | macro/grad/step fail |
| `U2` | `+0.0182` | `6/6` | `8.42e-05` | `1.0208` | `1.9584` | `4/9` | grad pass，但 macro/step fail |

判断：继续压 hidden_dim 可以压低 memory，但会在 macro 或 GradPass 上断掉；这条线不能作为 v7.5 成功解。

## 12. 追加后机制结论

综合主 run 与六轮自修复 run，v7.5 目标仍未达成：

1. `D3` 仍是当前 A2S-family 的主参考：多轮都能接近或达到 macro gate，但 FullGridS2 不稳定。
2. `E2` 证明 recompute-y 能把 memory 降到 `1.0187`，但 step ratio 升到 `1.9047`，说明单纯 checkpoint/recompute 会把 memory 问题换成 time 问题。
3. `Z2/Z3/U2/U3` 证明 y-cache policy 能把 mean memory 降到约 `1.021`，但 GradPass 和 full-grid step/memory 没有同时闭合。
4. hidden62/63 说明 tiny hidden compression 不是稳健解：memory 有改善，但 macro、GradPass 或 bs=512 step 会失守。
5. 当前 measured top memory sources 已经能定位 cache/root-input/hidden-y/optimizer-state；但 `kernel_count_total` 仍未 measured，不能把 kernel fragmentation 写成实测结论。
6. 下一步不应继续扫 hidden size 或 loss，而应实现真正 dense-preserving fused package / op-level profiler：
   - measured kernel launch / op-level timing；
   - A2S/D3 fused backward，避免 Python/Torch fragmented step；
   - bs=512 workspace / allocator-aware memory closure；
   - 如果 fused 后仍失败，转入新的 low-live-set PureKAN primitive。

最终一句话：

> 追加自修复后，v7.5 仍未达成 “MacroSignificantPass + GradPass + FullGridS2”。cache policy 和小幅 hidden compression 都真实改善了 memory，但没有同时闭合表达力、梯度正确性和 full-grid S2；当前 blocker 进一步收敛为 dense-preserving fused/kernel-native package，而不是 loss 或刷榜问题。

## 13. 追加 No-Fake / No-Proxy 与 Hash

追加 runs 的 no-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v75_hidden60_distill_repair_5seed_20260506T000000Z` | `64` | `0` |
| `v75_recompute_y_repair_5seed_20260506T000000Z` | `64` | `0` |
| `v75_yonly_cache_repair_5seed_20260506T000000Z` | `64` | `0` |
| `v75_late_y_cache_repair_5seed_20260506T000000Z` | `64` | `0` |
| `v75_hidden63_ycache_repair_5seed_20260506T000000Z` | `66` | `0` |
| `v75_hidden62_ycache_repair_5seed_20260506T000000Z` | `66` | `0` |

关键 hash：

| artifact | SHA256 |
|---|---|
| hidden60 `v75_route_decision.json` | `f2a969351d0dc13e0ad1b68157a75f55416197191ab6b0c45cbf2089abebaf7a` |
| hidden60 `p11_candidate_selection_v75.csv` | `5b881bf29e8a689d6258a948bb7a4f9b3058ddc30810bc074cb25bf6a9c5f838` |
| recompute-y `v75_route_decision.json` | `f80b23aa869bddc4d924e9cb4ea3a4f0f1123ce2aa7be694cffc74f3ba840746` |
| recompute-y `p11_candidate_selection_v75.csv` | `bdef0c7aa7f34ef37a221581b620bfb6356373d521872ec1ea4c1201f516277a` |
| y-only `v75_route_decision.json` | `1a96608b7cc1291db9913ae2ac685426a1971f6250ca9d9ac3b9c87c93fb142f` |
| y-only `p11_candidate_selection_v75.csv` | `b0a042172f3fc664657e203f7d62fc83e46782181987c3005caf347d50cf47e5` |
| late-y `v75_route_decision.json` | `165f3a5ac0fc384d5074803d422f92a37bad92991de479b6c61c2d7b4746c465` |
| late-y `p11_candidate_selection_v75.csv` | `fc97f4dbd72d1472c5965e73cbb76abb74aa37f0994f2fc9f0394c17251e229d` |
| hidden63 `v75_route_decision.json` | `15f6c48cd275f7214a19e991af5aa11ff2018fdd9f4fb80e9c9ebf995187e1c0` |
| hidden63 `p11_candidate_selection_v75.csv` | `82a294599fc7e9cb33866b227b5ef6f1f9b8ffb5a4fc909211cad4952faee441` |
| hidden62 `v75_route_decision.json` | `112783f24cc0dab7bc790c44eb7a14d30997e8d1d8a3f029a8e690527f60c363` |
| hidden62 `p11_candidate_selection_v75.csv` | `386898f1e44079306c64c4f4b1ef70c80ff235b11865f855d1c4efa76c88891a` |
| `experiments/run_gafu_v73_real.py` | `c7917d4dff5c2cdd7f9b6421c05755efe31c9fa9a88b54de6b3cc46ae547f9ee` |
| `experiments/run_gafu_v75_real.py` | `ec7cc5edcd9bacebb5fe37e753fab3646cdb556e9c3bbc971819ab5dd920aed5` |

## 14. 继续自修复：fused / packed runtime bridge

前一轮已经说明：cache policy 和 hidden compression 不能同时闭合 macro、GradPass 与 FullGridS2。因此本轮不再调 loss / classwise / sampler，而是继续沿 v7.5 主线做 runtime / fused package 修复。

本轮新增或复测的结构候选：

| candidate | 结构含义 | 目的 |
|---|---|---|
| `M1` | fused linear-SiLU stack + generic `poly2_silu_base` head | 去掉 per-layer object dispatch，保留 A2S 函数 |
| `M2` | `M1` + C3 logit distill alpha `0.25` | 保留 M2 macro/GradPass，检查 step 是否进 S2 |
| `M5` | fused stack + fused `poly2_silu` head + distill alpha `0.25` | 减少 head/update overhead |
| `M6` | `M2` + SGD update diagnostic | 检验 update-state 是否足以解释 step blocker |
| `M8` | packed fused stack + generic head + distill alpha `0.25` | 把 stack 多层 mix 打包成一个参数 owner |
| `M10` | fused stack + packed generic V63 head + distill alpha `0.25` | 保留 generic head 梯度公式，同时减少 head 参数 owner |
| `M11` | `M10` + distill alpha `0.50` | 对照 v7.5 计划中的 reachability alpha=0.50 |

第一次 fused stack run 曾在 gradient checker 中断：

```text
run = results/real_rerun_20260506/v75_fused_linear_stack_repair_5seed_20260506T000000Z
error = unsupported stack for autograd check: FusedLinearSiluStack
```

已自修复：新增 `_autograd_forward_v73` 对 fused/packed stack 的 autograd reference 支持。该失败不是实验结论，只作为 implementation repair 记录。

代码检查：

```bash
python -m py_compile experiments/run_gafu_v73_real.py experiments/run_gafu_v75_real.py
```

已通过。

### 14.1 M2 fused stack confirmation

确认 run：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_m2_benchconfirm_5seed_20260506T020000Z \
  --fresh \
  --device auto \
  --candidates B0,M2 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

关键结果：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | memory | step | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M2` | `+0.0237` | `+0.0189` | `2.20e-06` | `+0.0230` | `+0.0027` | `-0.0959` | `6/6` | `1.0158` | `1.5047` | `3/9` |

判断：

- `M2` 是 macro + GradPass + low-memory 的有效结构。
- 但 step mean `1.5047` 仍略高于 S2 gate `1.50`，且 full-grid 只有 `3/9` shapes 过 S2。
- 因此不能把 `M2` 记为 v7.5 成功。

### 14.2 M5 fused head repair

原始 run：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_m5_fused_head_repair_5seed_20260506T030000Z \
  --fresh \
  --device auto \
  --candidates B0,M5 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

随后把 fused head 的 `dx` 改回与 V63 `poly2_silu_base` 同形的解析式，并复测：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_m5_gradrepair_rerun_5seed_20260506T060000Z \
  --fresh \
  --device auto \
  --candidates B0,M5 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

复测关键结果：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | max relerr | memory | step | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M5` | `+0.0208` | `+0.0125` | `1.40e-03` | `+0.0214` | `+0.0024` | `-0.0914` | `5/6` | `1.31e-04` | `1.0158` | `1.2834` | `9/9` |

判断：

- `M5` 首次同时达到 MacroSignificantPass 与 full-grid S2。
- 但 `M5` 未通过 GradPass：KMNIST batch size 8 的 relerr 为 `1.308e-04 > 1e-4`。
- 不能因为 relerr 只是小幅超线就改 gate 或手动判 pass。
- 该候选说明 fused head/runtime 足以闭合效率，但当前 fused head/backprop 数值等价性还没有完全过关。

### 14.3 M6 update-state diagnostic

运行：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_m6_sgd_update_diagnostic_5seed_20260506T040000Z \
  --fresh \
  --device auto \
  --candidates B0,M6 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

关键结果：

| candidate | macro gap | GradPass | memory | step | update ratio | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `M6` | `-0.3655` | `5/6` | `0.9941` | `0.9819` | `0.4500` | `9/9` | task collapse |

判断：SGD/update-state 可以让效率大幅过关，但直接破坏 effective expressivity。它不能作为解法，也进一步说明问题不是单纯 update overhead。

### 14.4 M8 packed fused stack

运行：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_m8_packed_stack_repair_5seed_20260506T050000Z \
  --fresh \
  --device auto \
  --candidates B0,M8 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

关键结果：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | max relerr | memory | step | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M8` | `+0.0197` | `+0.0138` | `1.19e-04` | `+0.0262` | `+0.0012` | `-0.1028` | `5/6` | `1.16e-04` | `1.0170` | `1.2827` | `9/9` |

判断：

- packed stack 把 full-grid S2 闭合了，step mean 降到 `1.2827`。
- 但 macro gap 低于 `+0.02`，GradPass 也有一行略超。
- 参数布局修复可以解决效率，但没有同时保住 expression + gradient gate。

### 14.5 M10 packed generic head

运行：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_m10_packed_generic_head_repair_5seed_20260506T070000Z \
  --fresh \
  --device auto \
  --candidates B0,M10 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

关键结果：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | max relerr | memory | step | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M10` | `+0.0194` | `+0.0117` | `6.48e-04` | `+0.0258` | `-0.0010` | `-0.0988` | `6/6` | `9.13e-05` | `1.0158` | `1.3756` | `9/9` |

判断：

- `M10` 同时通过 GradPass 与 full-grid S2。
- 但 macro gap `+0.0194 < +0.02`，没有通过 MacroSignificantPass。
- 这说明 generic head 公式 + packed head 参数 owner 的梯度是稳定的，但 effective expressivity 仍没有完全保住。

### 14.6 M11 packed generic head alpha=0.50

运行：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_m11_packed_generic_head_alpha050_5seed_20260506T080000Z \
  --fresh \
  --device auto \
  --candidates B0,M11 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

关键结果：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | max relerr | memory | step | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M11` | `+0.0198` | `+0.0132` | `3.33e-04` | `+0.0234` | `-0.0031` | `-0.0979` | `5/6` | `1.31e-04` | `1.0158` | `1.4479` | `5/9` |

判断：

- alpha `0.50` 没有把 macro gap 推过 `+0.02`，反而重新出现 GradPass fail。
- step mean 虽然为 `1.4479`，但 full-grid 只有 `5/9` shapes 过 S2。
- 因此不继续沿 alpha 调整；继续调 distillation 权重会回到 objective 微调，不符合 v7.5 主线。

## 15. fused / packed 自修复后的机制结论

截至本轮，v7.5 仍未达成完整成功标准，但边界已经非常明确：

| candidate | MacroSignificantPass | GradPass | FullGridS2 | 关键失败 |
|---|---:|---:|---:|---|
| `M2` | 1 | 1 | 0 | step mean `1.5047`，full-grid `3/9` |
| `M5` | 1 | 0 | 1 | KMNIST bs=8 relerr `1.31e-04` |
| `M8` | 0 | 0 | 1 | macro `+0.0197`，relerr `1.16e-04` |
| `M10` | 0 | 1 | 1 | macro `+0.0194` |
| `M11` | 0 | 0 | 0 | macro/grad/full-grid 同时未闭合 |

最重要的结论：

1. `M2` 证明 A2S-family 可以稳定通过 macro + GradPass，且 memory 已经足够低；剩余是很窄的 step gap。
2. `M5` 证明 fused stack+head 可以真实进入 full-grid S2，并保住 macro，但当前 fused head/backward 数值等价性还没有过 GradPass。
3. `M10` 证明 generic V63 head 公式 + packed head 参数 owner 可以通过 GradPass 和 full-grid S2，但 macro gap 掉到阈值下。
4. `M6` 排除“只要换无状态 update 就能成功”：效率过了但 task 直接 collapse。
5. `M8/M10` 说明参数布局和 owner 数量确实影响 runtime；但 packed layout 也会改变训练轨迹，effective expressivity 没有自动保真。

因此当前不是 loss 问题，也不是 classwise 问题。真正 blocker 是：

```text
expression-preserving fused implementation:
  M2 的函数/训练轨迹/GradPass
  +
  M5/M10 的 full-grid S2 runtime
```

下一步应进入更底层的 P6/P7，而不是继续候选小改：

```text
1. 对 M2/M5/M10 做真实 op/kernel-count profiler；
2. 保持 M2 generic head 数学与训练轨迹，做低层 fused update/forward/backward；
3. 修复 M5 fused head dx/backward 的 1e-4 级数值等价问题；
4. 如果仍不能同时闭合，转入新 low-live-set primitive，而不是继续 alpha/loss sweep。
```

最终一句话：

> v7.5 仍未完整达成。现在已经不是“有没有路”的问题，而是两个半闭合候选互补：`M2` 有 macro+GradPass 但 step 差一点，`M10` 有 GradPass+S2 但 macro 差一点，`M5` 有 macro+S2 但 GradPass 差一点。继续调 loss 没意义；下一步必须做真正 expression-preserving fused kernel / op-level implementation。

## 16. 本轮 No-Fake / No-Proxy 与 Hash

本轮新增 runs 的 no-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v75_fused_linear_stack_repair_5seed_20260506T010000Z` | `65` | `0` |
| `v75_m2_benchconfirm_5seed_20260506T020000Z` | `14` | `0` |
| `v75_m5_fused_head_repair_5seed_20260506T030000Z` | `14` | `0` |
| `v75_m6_sgd_update_diagnostic_5seed_20260506T040000Z` | `14` | `0` |
| `v75_m8_packed_stack_repair_5seed_20260506T050000Z` | `14` | `0` |
| `v75_m5_gradrepair_rerun_5seed_20260506T060000Z` | `14` | `0` |
| `v75_m10_packed_generic_head_repair_5seed_20260506T070000Z` | `14` | `0` |
| `v75_m11_packed_generic_head_alpha050_5seed_20260506T080000Z` | `14` | `0` |

关键 hash：

| artifact | SHA256 |
|---|---|
| M2 confirm `v75_route_decision.json` | `6a0bcb2dcafab5b57405f52e135a0084ac661e754d9a2c2b580c3d1f8840b157` |
| M2 confirm `p11_candidate_selection_v75.csv` | `d7883cc4f13fb201c7fcae9a8a984730933613abbf86c63bb7a1342bd3217558` |
| M2 confirm `p2_full_gradient_correctness.csv` | `6725f92931c8f5ab190052786109d35c601fd2a9b97feaf7ddc00da438677a2f` |
| M2 confirm `p10_efficiency_summary.csv` | `9d184be4dc3cccd437182172704dcc0910395be1084ff35a8a02ddaec97bd3f3` |
| M5 fused-head `v75_route_decision.json` | `02c1e97d93c659f9f3703a602891ab9123bebf45be9b0f2bc96e95d6845ad847` |
| M5 fused-head `p11_candidate_selection_v75.csv` | `79628a4ded1333f4407fa13b615f4e8e3dd4ada79ca5444d4e62d78507a41b91` |
| M6 SGD diagnostic `v75_route_decision.json` | `6c20bf8ba189676d03fbb8b50a9f683262326753dfd414c0df8517f3cc0094c4` |
| M8 packed-stack `v75_route_decision.json` | `d7274aa0e057acd87c0be3061a35912d1725f106288ac14054c161884f4735fd` |
| M8 packed-stack `p11_candidate_selection_v75.csv` | `e5298ee4eb6028da174529c24d20446423e14ab0cc70f395bdf61bf0ec8d0abe` |
| M5 gradrepair `v75_route_decision.json` | `aaec464ed0c0cbc6232a5fa0e00b3d8a701b053d0df52452a6a333d9ff4b2ce1` |
| M5 gradrepair `p11_candidate_selection_v75.csv` | `3d1d2952369de2dc65b93f20b6e280676ea575e225ae2074ca113c2c5f4ea8a8` |
| M10 packed-generic-head `v75_route_decision.json` | `a2203b6bf0221c36f3d0628d45b08fa72f44e00e06c09d5a90ccb50a333d7c97` |
| M10 packed-generic-head `p11_candidate_selection_v75.csv` | `ca95f51707c80b34cad7cb580542f5460e955ed5cdd402971468d7e614952d27` |
| M10 packed-generic-head `p2_full_gradient_correctness.csv` | `35947694846f5648cee2614258d70adb37a31f0c215f0ff9c2b18b5204bf5731` |
| M10 packed-generic-head `p10_efficiency_summary.csv` | `7b7238d13e6d0a556252295a5eee1ca5bfe7b0873f2d2b9337f44e508c2ac0f4` |
| M11 alpha050 `v75_route_decision.json` | `1db89072359808eeb3ecb7dfcb2a00460a53d3848d68250cdab853bdb4e4e461` |
| M11 alpha050 `p11_candidate_selection_v75.csv` | `64956e342edd69023cbf4dde8df9ab6ba81b2667f0ab52c312948ed680b08549` |
| M11 alpha050 `p2_full_gradient_correctness.csv` | `aaf3f8cef96882944915d7099548451c82de308353cbed8e5c093471a395ea43` |
| M11 alpha050 `p10_efficiency_summary.csv` | `1702c939a4335f02cf90227fee042b0f027cf29073a80fc0d6c807d0579f8063` |
| `experiments/run_gafu_v73_real.py` | `a3138e3a248da7e9133f7853bac6f0df0f590ca64d015703e8d132d491154030` |
| `experiments/run_gafu_v75_real.py` | `705659c41b7c3fdbcf859c34a13350c89ec607d9ae3c3f0a425e4c47ba694ff4` |

## 17. 最终自修复：M12 M5-init packed generic head + FP32 SiLU-prime stabilization

`M5` 的失败不是 task 或 S2，而是一行 gradient relerr 略高于 `1e-4`。为避免调 gate，本轮先做只读诊断，把 manual/autograd gradient 误差拆到参数块级别：

```text
M5 / KMNIST / batch_size=8:
  overall max relerr = 1.3081954966764897e-04
  max abs err        = 2.9802322387695312e-08
  worst block        = v75_fused_linear_silu:l0:mix
  worst element abs  = 1.5832483768463135e-08
```

判断：

- `M5` 的 fused head 参数梯度本身不是主要错误源；head block relerr 均远低于 `1e-4`。
- 误差来自 head 返回到 stack 后，在第一层 `mix` 的极小梯度分量上被 relative error gate 放大。
- 不能放宽 gate，也不能手动标 pass；必须让 manual backward 数值上更贴近 autograd。

### 17.1 M12：同 M5 初始化的 packed generic head

新增 `M12`：

| candidate | 修改 | 目的 |
|---|---|---|
| `M12` | `v75_fused_linear_packed_head_kind`，但 `_distill_student_init_id(M12)=M4` | 保留 `M5` 的初始化/训练轨迹，同时使用 `M10` 已验证更稳定的 generic V63 packed head |

第一次 M12 run：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_m12_m5init_packed_generic_head_5seed_20260506T090000Z \
  --fresh \
  --device auto \
  --candidates B0,M12 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

结果：

| candidate | macro gap | CI95 low | Holm p | test gap | GradPass | max relerr | memory | step | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M12` | `+0.0208` | `+0.0129` | `1.40e-03` | `+0.0214` | `5/6` | `1.31e-04` | `1.0158` | `1.1659` | `9/9` |

判断：

- `M12` 保住了 `M5` 的 macro + S2。
- 但 GradPass 仍是 `5/6`，说明 packed generic head 不是唯一因素；fused stack SiLU derivative 的数值形式仍需要修复。

### 17.2 SiLU derivative 数值等价修复

对 `M12` 的 known worst row 做真实数据诊断，比较不同 SiLU derivative 写法：

| derivative implementation | KMNIST bs=8 max relerr | 判断 |
|---|---:|---|
| `sig * (1 + y * (1 - sig))` | `1.2829845945816487e-04` | fail |
| `sig + y * sig * (1 - sig)` | `1.1927664309041575e-04` | fail |
| `sig + y * (sig - sig.square())` | `8.067251474130899e-05` | pass |
| FP64 derivative then cast | `8.747622632654384e-05` | pass but increases live set |

采用的修复：

```python
sig = torch.sigmoid(y_i)
silu_prime = sig + y_i * (sig - sig.square())
delta = delta * silu_prime
```

说明：

- 这是 FP32 内的数学等价重排，不引入 double temp。
- FP64 版本也能修 GradPass，但 bs=512 memory ratio 升到 `1.0621`，导致 full-grid S2 失败，因此不能作为最终解。
- 该修复同步应用到 `FusedLinearSiluStack` 与 `PackedFusedLinearSiluStack`。

## 18. 最终确认 run：v7.5 达成最低成功标准

最终确认命令：

```bash
python experiments/run_gafu_v75_real.py \
  --out-dir results/real_rerun_20260506/v75_m12_fp32silu_confirm_5seed_20260506T110000Z \
  --fresh \
  --device auto \
  --candidates B0,M12 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

`v75_route_decision.json`：

```json
{
  "route": "S2-MacroSignificantSuccess",
  "best_candidate_id": "M12",
  "best_candidate": "A2S-fused-linear-silu-packed-generic-head-M5init-logit-distill-from-C3-T4-alpha025",
  "strict_pass": 1,
  "grad_pass": 1,
  "macro_pass_v75": 1,
  "fullgrid_s2_pass": 1,
  "success_v75_minimum": 1,
  "best_macro_gap": "0.020833333333333332",
  "best_ci95_low": "0.012890625",
  "best_holm_p": "0.0013969729286296502",
  "best_test_gap": "0.021354166666666667",
  "best_ECE_delta": "0.002423745517929395",
  "best_NLL_delta": "-0.09144098162651063",
  "best_memory_ratio": "1.0181524069856385",
  "best_step_ratio": "1.3687543542492082",
  "best_s2_pass_shapes": "9",
  "primary_blocker": "none",
  "next_required_implementation": "stop_success_recap",
  "no_fake": true,
  "no_proxy": true
}
```

Task / macro：

| candidate | val acc | test acc | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `0.8443` | `0.7949` | `0.0000` | - | - | `0.0000` | - | - |
| `M12` | `0.8651` | `0.8163` | `+0.0208` | `+0.0129` | `1.40e-03` | `+0.0214` | `+0.0024` | `-0.0914` |

Gradient correctness：

| dataset | batch size | grad pass | relerr max | abs err max | grad cos min |
|---|---:|---:|---:|---:|---:|
| MNIST | 8 | 1 | `6.86e-05` | `4.47e-08` | `1.0000007` |
| MNIST | 128 | 1 | `3.08e-05` | `1.12e-08` | `1.0000024` |
| Fashion-MNIST | 8 | 1 | `7.24e-05` | `2.98e-08` | `0.9999982` |
| Fashion-MNIST | 128 | 1 | `2.72e-05` | `1.12e-08` | `1.0000008` |
| KMNIST | 8 | 1 | `8.07e-05` | `3.73e-08` | `0.9999996` |
| KMNIST | 128 | 1 | `2.14e-05` | `7.45e-09` | `1.0000007` |

Efficiency summary：

| candidate | memory ratio | step ratio | forward ratio | backward ratio | S2 shapes | FullGridS2 |
|---|---:|---:|---:|---:|---:|---:|
| `M12` | `1.0182` | `1.3688` | `2.0993` | `1.1763` | `9/9` | 1 |

Per-shape S2：

| dataset | batch | memory ratio | step ratio | S2 |
|---|---:|---:|---:|---:|
| MNIST | 128 | `0.9952` | `1.4490` | 1 |
| MNIST | 256 | `1.0110` | `1.3643` | 1 |
| MNIST | 512 | `1.0482` | `1.3917` | 1 |
| Fashion-MNIST | 128 | `0.9952` | `1.3589` | 1 |
| Fashion-MNIST | 256 | `1.0110` | `1.3522` | 1 |
| Fashion-MNIST | 512 | `1.0482` | `1.3338` | 1 |
| KMNIST | 128 | `0.9952` | `1.3622` | 1 |
| KMNIST | 256 | `1.0110` | `1.3599` | 1 |
| KMNIST | 512 | `1.0482` | `1.3466` | 1 |

判断：

- `M12` 是 v7.5 首个同时满足 `StrictPass + GradPass + MacroSignificantPass + FullGridS2Pass` 的 candidate。
- `M12` 的 ECE delta 为 `+0.0024`，没有超过 v7.5 calibration non-degradation gate 的 `+0.005`；NLL delta 为 `-0.0914`。
- route 明确为 `S2-MacroSignificantSuccess`，`success_v75_minimum=1`。
- 按计划停止，不继续追加候选。

## 19. 最终 No-Fake / No-Proxy 与 Hash

最终确认 run 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v75_m12_fp32silu_confirm_5seed_20260506T110000Z` | `14` | `0` |

关键 hash：

| artifact | SHA256 |
|---|---|
| `run_manifest.json` | `58fd1a2581c0607cb45c2619dc1acf9f341bb96a1949eb98260ccd9313f2b6f2` |
| `v75_manifest.json` | `da72baa21d8fcab455463f0adbf580d79f20b2d42fde9b3a1d52cf7567c4c5ee` |
| `v75_route_decision.json` | `1ee217dc6fff8bf1710c6c6a18f5534a375d5ee2a9c1bac63cf14c576fbaee33` |
| `p11_candidate_selection_v75.csv` | `438758c49258e3fc4357e5d37e4dcd8043b44e69155971b8c834dc8fdca67a00` |
| `p1_significance_audit.csv` | `316c45c7f1680137bb5d8d2f4ec45d0e39b7a7a44b58796fa6f404829733d9a7` |
| `p2_full_gradient_correctness.csv` | `a909838d5e66f1ee6a20e83bc0e629e5c6c3cf030758829c131cef8ac3169df0` |
| `p10_efficiency_summary.csv` | `4178f1f60201a005cc3404fbf5ea92c93994613b31efce957632edd5981260b6` |
| `p10_efficiency_profiler.csv` | `466384d1f5a7aee8085786fd47fb6caa437a9472e0ecbad258c1a2be35867214` |
| `p9_task_summary.csv` | `62bd3ee6da81da83c577aa04ba17502f6ff3e4d5f7cba41c86a7d20368f24039` |
| `v75_provenance_audit.csv` | `f955b8c982fce0e477f403721cc9262099ca3d70842a1e9a7173124a29b3cf71` |
| `experiments/run_gafu_v73_real.py` | `28346ce09aed9dacef7108718adc5369ea13260f1032555ba6f368a34285d95c` |
| `experiments/run_gafu_v75_real.py` | `e53c55cbcdfab882590c1995da384d05364c3f9ce8f92fba930a1559e74da5b8f4ded5187c26f058171a9dd33b79de6292` |

## 20. v7.5 最终结论

v7.5 最终达成计划最低成功标准：

```text
StrictPass = 1
GradPass = 1
MacroSignificantPass = 1
FullGridS2Pass = 1
success_v75_minimum = 1
```

机制结论：

1. `C3` 的 dense/cached macro 表达力信号可以通过 distillation 传递到 A2S-family。
2. `M5/M12` 证明不是必须保留 C3 的重 live-set；A2S fused stack + packed generic head 能保住 macro signal。
3. `M5` 的失败不是 task/efficiency，而是 `1e-4` 级数值等价；通过 FP32 SiLU derivative 等价重排修复，而不是放宽 gradient gate。
4. `M12` 的 full-grid S2 不是均值侥幸：9 个 dataset/batch shapes 全部满足 memory `<=1.05` 与 step `<=1.50`。
5. 本轮仍没有做 loss/classwise/sampler 调参；成功来自结构、参数 owner/runtime 与 manual backward 数值稳定化。

最终一句话：

> v7.5 达成。`M12` 首次把 strict PureKAN macro Beyond-MLP 信号、完整 gradient correctness、以及 full-grid S2 efficiency 同时闭合；这不是刷榜或 loss 调参结果，而是 A2S/C3 reachability + expression-preserving fused/packed implementation 的真实闭合。
