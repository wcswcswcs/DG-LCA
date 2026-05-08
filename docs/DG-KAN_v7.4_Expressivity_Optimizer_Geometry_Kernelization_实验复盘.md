# DG-KAN v7.4 Expressivity Optimizer Geometry Kernelization 实验复盘

> 本复盘记录 `docs/DG-KAN_v7.4_Expressivity_Optimizer_Geometry_Kernelization_完整实验计划.md` 的执行结果。所有结论只来自落盘 CSV/JSON/manifest/log；未实现或未测字段不补数。v7.4 明确不把 classwise-safe、loss 调参、sampler、class weight、focal、target margin 作为主线，因此本轮只做结构/表达力/效率方向的机制性 probe。

## 1. 目标是否达成

| 目标 | 结论 | 证据 |
|---|---|---|
| real-only / no-fake / no-proxy | 达成 | `v74_provenance_audit.csv`: `rows_checked=55`, `fake_proxy_nonzero_count=0` |
| StrictPass | 达成 | route best `C3` 为 strict KAN head candidate |
| GradPass | 达成 | `C3` full gradient `6/6` pass，max relerr `6.02e-05` |
| MacroSignificantPass | 达成 | `C3` macro val gap `+0.0230`，CI95 low `+0.0159`，Holm p `2.90e-04`，test gap `+0.0245` |
| Calibration / NLL 不劣化 | 达成 | `C3` ECE delta `-0.0018`，NLL delta `-0.0995` |
| S2 efficiency | 未达成 | `C3` memory ratio `1.3069`，step ratio `3.9090`；所有 structural candidates `survivor=FAIL` |
| v7.4 完整成功标准 | 未达成 | 缺 `S2Pass` / `S1Pass` |

最终判断：

> v7.4 本轮没有达成完整目标。`C3` 继续证明 dense/cached strict KAN 的 macro 表达力优势真实存在，但它不是 kernel-native efficient。结构性自修复候选 `A2S/A2C/A4C/A5C` 将 memory live-set 明显压低，其中 `A2S` 已接近 macro 与 S2，但没有同时闭合 macro gate、gradient gate、全 shape S2 gate。

## 2. 本轮执行

主 run：

```bash
python experiments/run_gafu_v74_real.py \
  --out-dir results/real_rerun_20260506/v74_mechanism_kernel_bridge_5seed_20260506T021447Z \
  --fresh \
  --device auto \
  --candidates B0,C3,A2C,A2S,A4C,A5C \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

代码检查：

```text
python -m py_compile experiments/run_gafu_v73_real.py experiments/run_gafu_v74_real.py
```

已通过。

实现说明：

| 文件 | 作用 |
|---|---|
| `experiments/run_gafu_v73_real.py` | 承载 v7.3/v7.4 structural probe：cached kind stack、matched-init、expressivity/efficiency postprocess |
| `experiments/run_gafu_v74_real.py` | v7.4 wrapper，写入 v7.4 plan/script provenance，并镜像 `v74_*` artifacts |

新增/使用的结构候选：

| candidate | 结构含义 | 目的 |
|---|---|---|
| `C3` | cached hidden-y strict rbf head | dense/cached strict reference，保表达力 |
| `A2C` | cached linear stack + poly2-gate KAN head | 去掉 stack poly2，检查低 live-set 是否保表达力 |
| `A2S` | cached linear stack + poly2-silu KAN head | 比 `A2C` 更轻的 head 变体，不改 loss |
| `A4C` | first layer poly2，其余 linear + poly2-gate head | 保留少量 early poly2 交互 |
| `A5C` | last layer poly2，其余 linear + poly2-gate head | 保留 late poly2 交互 |

没有执行的方向：

- 没有做 label smoothing / lr / weight decay sweep。
- 没有做 class weight / sampler / focal / target margin。
- 没有把 classwise 当 hard gate。
- 没有写 fake ratio、proxy row 或手填 pass。

## 3. Route Decision

`v74_route_decision.json`：

```json
{
  "route": "R3-MacroExpressivityPositiveButNotKernelNative",
  "best_candidate_id": "C3",
  "best_candidate": "H3-cached-hidden-y-rbf-head",
  "macro_pass_v73": 1,
  "grad_pass": 1,
  "s2_pass": 0,
  "best_macro_gap": 0.023046875,
  "best_ci95_low": 0.015885416666666666,
  "best_holm_p": 0.00028987170286454497,
  "best_test_gap": 0.024479166666666666,
  "best_ECE_delta": -0.0017973889907201132,
  "best_NLL_delta": -0.09949278235435485,
  "best_memory_ratio": "1.3069108164142476",
  "best_step_ratio": "3.9089837583560585",
  "optimization_or_expressivity_explained": true,
  "primary_blocker": "kernel_native_efficiency",
  "classwise_is_hard_gate": false,
  "no_fake": true,
  "no_proxy": true
}
```

说明：route JSON 中字段名 `macro_pass_v73` 来自复用的 v7.3 postprocess，但本 run 的 `v74_manifest.json` 已指向 v7.4 plan；v7.4 口径下 `C3` 满足 MacroSignificantPass，但不满足 S2。

## 4. Task / Macro 结果

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8443` | `0.7949` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `C3` | `0.8673` | `0.8194` | `+0.0230` | F-MNIST `+0.0164`, KMNIST `+0.0230`, MNIST `+0.0297` | 1 |
| `A2S` | `0.8642` | `0.8182` | `+0.0199` | F-MNIST `+0.0141`, KMNIST `+0.0184`, MNIST `+0.0273` | 1 |
| `A2C` | `0.8633` | `0.8198` | `+0.0190` | F-MNIST `+0.0172`, KMNIST `+0.0145`, MNIST `+0.0254` | 1 |
| `A4C` | `0.8633` | `0.8201` | `+0.0190` | F-MNIST `+0.0172`, KMNIST `+0.0148`, MNIST `+0.0250` | 1 |
| `A5C` | `0.8632` | `0.8199` | `+0.0189` | F-MNIST `+0.0172`, KMNIST `+0.0145`, MNIST `+0.0250` | 1 |

Macro significance / calibration：

| candidate | mean val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | v7.4 macro pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `C3` | `+0.0230` | `+0.0159` | `2.90e-04` | `+0.0245` | `-0.0018` | `-0.0995` | 1 |
| `A2S` | `+0.0199` | `+0.0125` | `2.51e-03` | `+0.0233` | `-0.0004` | `-0.0903` | 0 |
| `A2C` | `+0.0190` | `+0.0130` | `9.76e-04` | `+0.0249` | `-0.0034` | `-0.0953` | 0 |
| `A4C` | `+0.0190` | `+0.0128` | `9.01e-04` | `+0.0251` | `-0.0036` | `-0.0952` | 0 |
| `A5C` | `+0.0189` | `+0.0126` | `1.13e-03` | `+0.0250` | `-0.0029` | `-0.0953` | 0 |

判断：

- `C3` 不是刷榜式偶然提升：macro val/test、CI、Holm、ECE、NLL 同时支持正向。
- `A2S` 是最接近的低 live-set bridge：只差 `0.000078125` 到 `+0.02` macro gate，且 ECE/NLL 仍优于 MLP。
- 但 `A2S` 不能被记为达成，因为 gate 是预先定义的，不能四舍五入为成功。

## 5. Gradient Correctness

| candidate | pass rows | max relerr | grad cos min | worst row |
|---|---:|---:|---:|---|
| `C3` | `6/6` | `6.02e-05` | `0.9999939` | KMNIST bs=8 |
| `A2S` | `6/6` | `9.27e-05` | `0.9999961` | KMNIST bs=8 |
| `A5C` | `6/6` | `6.79e-05` | `1.0000000` | Fashion-MNIST bs=8 |
| `A2C` | `5/6` | `1.23e-04` | `0.9999991` | Fashion-MNIST bs=8 |
| `A4C` | `5/6` | `1.05e-04` | `0.9999903` | KMNIST bs=8 |

判断：

- `C3/A2S/A5C` 通过 GradPass。
- `A2C/A4C` 的 relerr 略高于 `1e-4`，不能进入 route best。
- 本轮没有手动改 pass；所有 relerr 直接来自 `p2_full_gradient_correctness.csv`。

## 6. Efficiency / Kernelization

| candidate | memory ratio | step ratio | forward ratio | backward ratio | S2 pass shapes | survivor |
|---|---:|---:|---:|---:|---:|---|
| `A2C` | `1.0368` | `1.6446` | `3.2895` | `1.1959` | `4/9` | FAIL |
| `A2S` | `1.0368` | `1.6691` | `2.7607` | `1.3023` | `4/9` | FAIL |
| `A4C` | `1.2129` | `2.0602` | `2.1634` | `2.3496` | `0/9` | FAIL |
| `A5C` | `1.0369` | `2.2790` | `2.2271` | `2.1251` | `0/9` | FAIL |
| `C3` | `1.3069` | `3.9090` | `3.7245` | `4.7303` | `0/9` | FAIL |

`A2S` per-shape S2 details：

| dataset | batch | memory ratio | step ratio | S2 |
|---|---:|---:|---:|---:|
| MNIST | 128 | `1.0001` | `1.4787` | 1 |
| MNIST | 256 | `1.0287` | `1.4645` | 1 |
| MNIST | 512 | `1.0816` | `1.6492` | 0 |
| Fashion-MNIST | 128 | `1.0001` | `1.5129` | 0 |
| Fashion-MNIST | 256 | `1.0287` | `1.0320` | 1 |
| Fashion-MNIST | 512 | `1.0816` | `2.8830` | 0 |
| KMNIST | 128 | `1.0001` | `1.4383` | 1 |
| KMNIST | 256 | `1.0287` | `2.0855` | 0 |
| KMNIST | 512 | `1.0816` | `1.4777` | 0 |

判断：

- `A2S/A2C` 证明低 live-set structural bridge 是真实方向：memory 均值已经低于 S2 的 `1.05`。
- 但 full-grid S2 失败来自两个硬点：bs=512 memory 固定到约 `1.0816`，部分 shape step 仍明显高于 `1.50`。
- `C3` 保住表达力，但 memory/step 都远离 S2，不能作为 kernel-native success。
- `A4C/A5C` 说明把 poly2 只放在一层不能自然闭合：表达力没有超过 `A2S`，效率也更差或不稳定。

## 7. 遇到的问题与自修复

| 问题 | 自修复 | 结果 |
|---|---|---|
| v7.2/v7.3 复盘中曾把 classwise repair 写成主 blocker | v7.4 中降级为 diagnostic，不再作为 hard gate | 本轮 route 不看 classwise pass/fail |
| 早期 `A2C` cached probe 存在 unmatched-init protocol risk | 修复 `_uses_v72_train_path` 与 `_init_seed_candidate_id_v73`，让 `A2C/A2R/A2S/A4C/A5C` 与 `A2` matched init | apparent near-success 被重新测量；protocol-fixed 后 `A2C` macro `+0.0190`，GradPass `5/6` |
| `C3` 保表达力但效率失败 | 新增低 live-set structural bridge：`A2C/A2S/A4C/A5C` | `A2S` 达到 memory mean `1.0368`、GradPass `6/6`，但 macro `+0.0199`、step mean `1.6691`，仍未成功 |
| 低 live-set bridge 仍有 step/memory shape fail | 做 hidden60 structural Pareto follow-up（real run，非 loss 调参） | `A2C` GradPass 修复到 `6/6`，memory `1.0341`，但 step `1.7867`；`A2S` step `1.5759` 仍未过 S2，GradPass `5/6` |
| live-set attribution 未完整到 top memory/kernel source | 当前 postprocess 只写 partial measured cache/timing，materialized tensor count 标记 `not_measured` | 没有编造 P5 attribution；下一步必须补真实 live-set/kernel-count instrumentation |

支持性自修复 runs：

| run | 关键结果 | 判断 |
|---|---|---|
| `v73_a2_cached_matchedinit_5seed_20260506T014748Z` | `A2C` memory `1.0368`，step `1.3581`，但 macro `+0.0190`、GradPass `5/6` | protocol-fixed 后不是成功 |
| `v73_lowcost_structural_followup_5seed_20260506T015139Z` | `A2S` macro `+0.0199`，GradPass `6/6`，memory `1.0368`，step `1.3759`，S2 shapes `5/9` | 最接近但 full-grid S2 未闭合 |
| `v73_a2s_hidden60_pareto_5seed_20260506T015403Z` | `A2C` GradPass `6/6`，memory `1.0341`，step `1.7867`; `A2S` memory `1.0341`，step `1.5759`，GradPass `5/6` | hidden compression 不能同时闭合 task/grad/S2 |

这些 self-repair 都是结构/效率方向，不是 loss、sampler 或 classwise 方向。

## 8. 机制分析

### 8.1 Dense/cached KAN 的 macro 表达力真实存在

`C3` 在 5 seeds x 3 datasets 下：

```text
mean val gap = +0.0230
CI95 low = +0.0159
Holm p = 2.90e-04
test gap = +0.0245
ECE delta = -0.0018
NLL delta = -0.0995
GradPass = 6/6
```

这支持 v7.4 的核心前提：问题不是“KAN 函数空间弱于 MLP”。

### 8.2 压低 live-set 会损失有效表达力

`A2S/A2C/A4C/A5C` 都把 `C3` 的 memory ratio 从 `1.3069` 降到 `1.0368-1.2129`，但 macro gap 全部低于 `C3`：

```text
C3  macro gap = +0.0230
A2S macro gap = +0.0199
A2C macro gap = +0.0190
A4C macro gap = +0.0190
A5C macro gap = +0.0189
```

这说明低 live-set bridge 在表达力上已经接近，但仍没有完整保住 dense/cached KAN 的优势。

### 8.3 当前失败不是 loss 问题

本轮没有做 loss/sampler/classwise 修补。失败的直接形式是：

```text
C3:
  MacroSignificantPass + GradPass
  S2 fail

A2S:
  GradPass + near macro + near memory
  S2 shape fail + macro gap just below threshold

A2C/A4C:
  near memory/step or near macro
  GradPass incomplete
```

因此下一步不应回到 focal/margin/class weighting，而应做真正的 P5/P6/P7：

```text
1. 真实 live-set / kernel-count attribution；
2. materialization-free fused transform+mix backward；
3. streaming head / gate / grad temps；
4. tile-stream dense-preserving package；
5. 若仍不能 S2，则进入 new low-live-set primitive。
```

### 8.4 目前不能宣称 H5 完整成立

v7.4 计划要求 P5 定位 materialized live set、kernel fragmentation、top memory/time source。本轮仅有 partial cache/timing attribution，`materialized_tensor_count` 写为 `not_measured`。因此：

```text
H5 direction is supported by efficiency symptoms,
but H5 attribution gate is not fully completed.
```

不能写“materialization 占 peak >=35%”这类没有实测的比例。

## 9. No-Fake / No-Proxy 审计

| artifact | value |
|---|---|
| run | `results/real_rerun_20260506/v74_mechanism_kernel_bridge_5seed_20260506T021447Z` |
| rows checked | `55` |
| fake_proxy_nonzero_count | `0` |
| no_fake | `true` |
| no_proxy | `true` |
| plan path | `docs/DG-KAN_v7.4_Expressivity_Optimizer_Geometry_Kernelization_完整实验计划.md` |
| script path | `experiments/run_gafu_v74_real.py` |

关键 hash：

| file | SHA256 |
|---|---|
| `run_manifest.json` | `aa0773f40aa0997be2eb22a5d2adc79c50642f29f06899a824b952e2984a620f` |
| `v74_manifest.json` | `c4f818ce67340c189074f55f61d758f73f7e16e4f1ab057d4a4d051d6e21003e` |
| `v74_route_decision.json` | `a385546429223eb65ac5e40a8fe536352739b9ae5ada18cc95918f826c2f1d6d` |
| `p9_task_summary.csv` | `e27fa1e6c01763083ce26e8d754af887777eebb4ba4469d835b04313cd8e299a` |
| `p1_significance_audit.csv` | `f434d554f998b13cecc336d258f451abb062b95520e1766a121dd442ce15c2cf` |
| `p2_full_gradient_correctness.csv` | `3438ed5a23e84b526bb242f2b74be2f58c158fd429c26b1af791e06c115e2a3b` |
| `p10_efficiency_summary.csv` | `d2be95e29a1a057a663fdd1ef8304702021a2674b05809235f00bca386d71217` |
| `v74_provenance_audit.csv` | `43fac06522873cd674aecdf8c3710ac116cfa68aedfd3dcc7ce29557e051c9e4` |
| `experiments/run_gafu_v73_real.py` | `cb3e12bee3c218698e3068a7ab4bbc0c915f218278195aea6f986813544d2b9d` |
| `experiments/run_gafu_v74_real.py` | `9c275ae13a1d54a9bed2c7c840b5b8f4ded5187c26f058171a9dd33b79de6292` |

## 10. 最终结论

v7.4 本轮目标没有达成完整成功标准：

```text
StrictPass = true
GradPass = true for C3
MacroSignificantPass = true for C3
S2Pass = false
S1Pass = false
```

但这不是“刷榜失败”或“loss 没调好”。本轮机制结论是：

1. `C3` 继续验证 dense/cached strict KAN 的 macro 表达力优势，且 ECE/NLL 没有变差。
2. `A2S` 是当前最接近 low-live-set bridge 的候选：macro `+0.0199`、GradPass `6/6`、memory `1.0368`，但 step 和 full-grid S2 未闭合。
3. `A2C/A4C` 接近低 memory/near macro，但 gradient gate 不完整。
4. `A4C/A5C` 说明单层 poly2 bridge 不能自然保住 `C3` 表达力并进入 S2。
5. 继续做 loss/classwise/sampler 不符合 v7.4 计划，也不会触及当前 blocker。
6. 下一步必须先补 P5 的真实 live-set/kernel-count attribution，然后做 P6/P7 的 dense-preserving materialization-free kernelization；如果 fused 后仍不能达到 S2，则按 v7.4 计划转入 new low-live-set function-space primitive。

最终一句话：

> v7.4 目前推进到了“macro 表达力真实、低 live-set bridge 接近但未闭合”的阶段；完整目标未达成，硬 blocker 是 kernel-native efficiency 与表达力保真无法同时满足。下一步不应再调 loss，而应做真实 live-set attribution 与 dense-preserving fused/kernel-native forward-backward。
