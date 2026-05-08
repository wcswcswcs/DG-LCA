# DG-KAN v7.2 SignificantBeyondMLP KernelNativeEfficiency 实验复盘

> 本复盘记录 `docs/DG-KAN_v7.2_SignificantBeyondMLP_KernelNativeEfficiency_完整实验计划.md` 的真实执行结果。主结论来自 `results/real_rerun_20260505/v72_significant_kernel_all_20260505T224930Z`，并追加记录用户要求继续优化后的 no-sync/task-repair 与 low-rank primitive targeted probes。所有数字均来自落盘 CSV/JSON/manifest/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 计划目标是否达成

| 目标 | 结论 | 证据 |
|---|---|---|
| real-only v7.2 runner | 达成 | `experiments/run_gafu_v72_real.py` 跑通并落盘完整 artifacts |
| StrictPass | 达成 | best `H3` 为 strict KAN head，`strict_pass=1` |
| GradPass | 达成 | best `H3` full gradient check `6/6` pass，max relerr `6.97e-05` |
| basic Beyond task gate | 达成 | best `H3` val gap vs MLP `+0.0185`，3/3 datasets `KAN >= MLP` |
| SignificantTaskPass | 未达成 | `H3` macro val gap `+0.0185 < +0.02`，classwise drop max `0.1455 > 0.05` |
| S2 efficiency gate | 未达成 | `H3` memory ratio `1.3147`、step ratio `3.5237`，均高于 S2 gate |
| v7.2 最低成功标准 | 未达成 | 缺 `S2Pass` 与 `SignificantTaskPass` |

最终判断：

> v7.2 没有达成计划定义的 “显著 Beyond-MLP + kernel-native efficiency” 目标。`H3` 仍是最佳 strict task candidate，但只是 basic task positive；当前 blocker 是 `significance_or_efficiency`，更具体地说是 task gap 未过 $+2\%$ 实践阈值、classwise non-collapse 未过、以及 dense/materialized implementation 远未进入 S2。

## 2. 实验溯源

首次 full run 命令：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_significant_kernel_all_20260505T224930Z \
  --fresh \
  --device auto \
  --candidates B0,B3,H3,H4,K0,K1,K2,G3,G5,G6 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 10000
```

自修复后复用已落盘 task rows，重算 grad/efficiency/route 的命令：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_significant_kernel_all_20260505T224930Z \
  --device auto \
  --candidates B0,B3,H3,H4,K0,K1,K2,G3,G5,G6 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 10000
```

覆盖：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
task rows = 300
task trace rows = 3600
gradient rows = 48
efficiency rows = 90
bench batch sizes = 128,256,512
source commit = f76ff27ea03c6fbfc01cadf24ea281c32a72b18f
```

## 3. 遇到的问题与自修复

| 问题 | 自修复尝试 | 结果 |
|---|---|---|
| 首次 full run 在 K0 gradient checker 中断 | 修正 `_autograd_forward`，让 autograd verifier 支持 grouped stack 的 list-of-lists layer 结构 | 同一结果目录复用 300 条真实 task rows 后，P2/P10/route 完整落盘 |
| CSV 复用时 `head_is_kan` 为字符串，route strict candidate 过滤错误 | 增加 `_is_one/_is_zero`，修正 route/contract/P13/failure strict 判断 | route 从空候选修正为 `H3` |
| Python grouped implementation step 极慢 | 实现真实 vectorized grouped candidates `G3/G5/G6`，去掉 Python group loop | step 大幅下降，但仍不进 S2，且 g8/g16 task gap 明显 |
| lower-level fused/Triton materialization-free kernel 未完成 | P4/P5/P6/P8 中相关 packages 写为 `not_implemented` | 没有写假 ratio 或假 pass |

## 4. Task 结果

| candidate | family | val acc | test acc | val gap vs MLP | basic pass | strict |
|---|---|---:|---:|---:|---:|---:|
| `B0` | baseline MLP | `0.8449` | `0.7958` | `0.0000` | reference | 0 |
| `B3` | dense linear oracle | `0.8632` | `0.8158` | `+0.0182` | 1 | 0 |
| `H3` | strict rbf-poly-exp head | `0.8634` | `0.8150` | `+0.0185` | 1 | 1 |
| `H4` | strict depth-2 poly2-gate | `0.8575` | `0.8078` | `+0.0126` | 1 | 1 |
| `K0` | grouped g2 Python | `0.8542` | `0.8058` | `+0.0093` | 1 | 1 |
| `G3` | grouped g2 vectorized | `0.8531` | `0.8049` | `+0.0081` | 1 | 1 |
| `G5` | grouped g8 vectorized | `0.8076` | `0.7591` | `-0.0373` | 0 | 1 |
| `G6` | grouped g16 vectorized | `0.7794` | `0.7237` | `-0.0656` | 0 | 1 |

最佳 strict candidate 是 `H3`。它保留了 v7.1 的 positive task signal，但本轮 10-seed 统计后仍未达到 significant gate。

## 5. 显著性审计

Macro 统计：

| candidate | mean val gap | CI95 low | CI95 high | Holm p | mean test gap | ECE delta | NLL delta | classwise drop max | SignificantTaskPass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B3` | `+0.0182` | `+0.0128` | `+0.0233` | `8.27e-06` | `+0.0200` | `-0.0053` | `-0.0985` | `0.2600` | 0 |
| `H3` | `+0.0185` | `+0.0134` | `+0.0234` | `3.17e-06` | `+0.0192` | `-0.0067` | `-0.1004` | `0.1455` | 0 |
| `H4` | `+0.0126` | `+0.0077` | `+0.0176` | `8.64e-04` | `+0.0120` | `-0.0023` | `-0.0830` | `0.1400` | 0 |
| `G3` | `+0.0081` | `+0.0023` | `+0.0139` | `0.1185` | `+0.0092` | `-0.0065` | `-0.0808` | `0.1636` | 0 |

判断：

- `H3` 的 CI 和 Holm p 支持“正向差异存在”，但计划要求的是实践显著优势：macro val gap 至少 `+0.02`，`H3` 实测为 `+0.0185`。
- `H3` 的 ECE/NLL 优于 MLP，但 classwise drop max `0.1455` 明显超过 non-collapse gate。
- 因此不能把 `H3` 记为 significant Beyond-MLP 成功。

## 6. Gradient Correctness

| candidate | pass rows | grad relerr max | grad abs err max | grad cos min |
|---|---:|---:|---:|---:|
| `H3` | `6/6` | `6.97e-05` | `3.73e-08` | `0.999996` |
| `H4` | `6/6` | `6.89e-05` | `3.73e-08` | `1.000000` |
| `G3` | `6/6` | `9.36e-05` | `2.98e-08` | `0.999997` |
| `G5` | `6/6` | `2.82e-05` | `5.96e-08` | `1.000000` |
| `G6` | `6/6` | `6.54e-05` | `5.96e-08` | `1.000000` |
| `K0` | `4/6` | `1.10e-04` | `4.47e-08` | `1.000000` |

判断：

- `H3/H4/G3/G5/G6/K1/K2` 均通过 full gradient gate。
- `K0` 有 2 条 relerr 刚高于 `1e-4` gate，已按 failure 记录；没有手动改 pass。

## 7. Efficiency 与 Kernelization

| candidate | memory ratio | step ratio | forward ratio | backward ratio | survivor |
|---|---:|---:|---:|---:|---|
| `B3` | `1.3143` | `3.2512` | `2.8913` | `2.8860` | FAIL |
| `H3` | `1.3147` | `3.5237` | `4.0394` | `3.0148` | FAIL |
| `H4` | `1.3072` | `2.5621` | `2.7217` | `2.0566` | FAIL |
| `K0` | `1.1489` | `5.5635` | `5.6951` | `4.4351` | FAIL |
| `K1` | `1.0546` | `18.0890` | `15.9061` | `14.6039` | FAIL |
| `K2` | `1.0579` | `34.6664` | `29.1549` | `28.0992` | FAIL |
| `G3` | `1.3099` | `3.3990` | `4.2056` | `2.5624` | FAIL |
| `G5` | `1.2923` | `3.4103` | `4.2020` | `2.5757` | FAIL |
| `G6` | `1.2894` | `3.3910` | `4.2043` | `2.5589` | FAIL |

Vectorized grouped repair 的真实效果：

| repair | step before | step after | step reduction | memory before | memory after |
|---|---:|---:|---:|---:|---:|
| `K0 -> G3` | `5.5635` | `3.3990` | `38.9%` | `1.1489` | `1.3099` |
| `K1 -> G5` | `18.0890` | `3.4103` | `81.1%` | `1.0546` | `1.2923` |
| `K2 -> G6` | `34.6664` | `3.3910` | `90.2%` | `1.0579` | `1.2894` |

判断：

- vectorization 证明 Python group loop 是大问题，但它不是最终 kernel-native 解法。
- `G3/G5/G6` 把 step 降到约 `3.4x MLP`，仍远高于 S2 的 `1.50x`。
- vectorized grouped 的 memory 反而升到约 `1.29-1.31x`，说明 batched group materialization 加重 live set。
- 剩余 blocker 必须靠 materialization-free fused forward/backward 或新的低内存 primitive，而不是继续 Python-level vectorization。

## 8. Route Decision

`route_decision.json`：

```json
{
  "route": "R3-TaskPositiveButNotSignificant",
  "best_candidate": "D3-dense-poly2-gate-d3-KAN-head-rbf-poly-exp",
  "best_candidate_id": "H3",
  "best_family": "strict_head_repair",
  "best_memory_ratio": 1.3146932037897496,
  "best_step_ratio": 3.52373346911221,
  "best_backward_ratio": 3.0147935571850684,
  "best_forward_ratio": 4.039425167712569,
  "best_val_acc": 0.8634114583333333,
  "best_test_acc": 0.8149739583333333,
  "val_gap_vs_MLP": 0.01848958333333332,
  "test_gap_vs_MLP": 0.019205729166666668,
  "ci95_low": 0.013409830729166667,
  "ci95_high": 0.0234375,
  "holm_p": 3.1708761676834654e-06,
  "cohen_h": 0.0524115987113869,
  "ECE_delta": -0.006692740134894848,
  "NLL_delta": -0.1003639817237854,
  "survivor_type": "S6",
  "strict_pass": 1,
  "grad_pass": 1,
  "s2_pass": 0,
  "s1_pass": 0,
  "basic_task_pass": 1,
  "significant_task_pass": 0,
  "official_task_opened": false,
  "diagnostic_task_opened": true,
  "primary_blocker": "significance_or_efficiency",
  "next_required_implementation": "materialization_free_fused_kernel_or_new_primitive",
  "no_fake": true,
  "no_proxy": true
}
```

## 9. Failure Table

`failure_table.csv` 统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F11_nonKAN_violation` | 1 | `B3` 是 dense linear oracle，不能算 strict PureKAN 成功 |
| `F4_task_not_significant` | 13 | basic fail 或 significant gate fail |
| `F3_gradient_correctness_fail` | 1 | `K0` 有 relerr 行略超 gate |
| `F1_memory_fail` | 9 | 所有效率候选 memory ratio 均高于 S2 gate |

没有出现：

- fake data failure
- proxy row failure
- artifact missing failure
- 对未实现 fused/Triton packages 的假 measured row

## 10. No-Fake / No-Proxy 审计

最终目录 CSV 审计：

```text
rows_checked = 4603
fake_proxy_nonzero_count = 0
no_fake = true
no_proxy = true
```

关键 artifact hash：

| 文件 | SHA256 |
|---|---|
| `initial_failed_run.log` | `25033e2f50aaa0ce5f12193ab5ee3d0a777549e7bc71f7fa40f4d0ac79b77336` |
| `run.log` | `2cbad38d090c35e475b2f46a055ccbf2d50a844ec674bf828b05cc58fa08a6e3` |
| `run_manifest.json` | `5461d9c75994dd3958bb0e7f4e065a3e05e82e6c04196ad4d9d65ac1291b5d7b` |
| `p9_task_summary.csv` | `f736fbd82c4821c5bd1677115d0ec556fdbacca1554375650a491673257819ab` |
| `p1_significance_audit.csv` | `35d91309b8dadeda8046a7e1416e78712722b7244eae1468510305452d5c6fd2` |
| `p2_full_gradient_correctness.csv` | `8379114ed99781f2796e65f3b0e3d508d659738570a6ffdf3393bf46fb53a892` |
| `p10_efficiency_summary.csv` | `2bcf06f255a6ffb21fc8e91f848d78cb40cd84de8e64a369baf90c887654cf8e` |
| `route_decision.json` | `6b744ed71939168ca5aca0569b843191853663e82d02beb4fb79cc5d22881c44` |
| `failure_table.csv` | `ba9e8f7239a45d872f9f2a5fe8afefd8099b202efb7567f73fc66293102b412c` |
| `provenance_audit.csv` | `f100feaa67bc139d598ddf7f5e662adf9ada8c707c0929d14a0f6969bd0fc96b` |

## 11. 结论

本轮 v7.2 支持以下真实结论：

1. v7.2 已真实执行 3 datasets x 10 seeds 的 task/significance audit。
2. `H3` 仍是最佳 strict task candidate：val acc `0.8634`，val gap `+0.0185`，test gap `+0.0192`。
3. `H3` 通过 strict 与 full gradient gate，但未通过 SignificantTaskPass，因为 macro val gap 未达到 `+0.02`，且 classwise drop max 过大。
4. `H3/H4/B3/G3/G5/G6` 等效率候选全部未进 S2；best strict `H3` step ratio `3.5237`，memory ratio `1.3147`。
5. 自修复实现的 `G3/G5/G6` vectorized grouped path 明显降低 Python loop step，但 memory live set 变大，仍不是 kernel-native success。
6. v7.2 最终 route 是 `R3-TaskPositiveButNotSignificant`。

最终一句话：

> v7.2 没有达成 “显著 Beyond-MLP + S2 efficiency” 的最低成功标准。`H3` 的 strict task signal 仍真实存在，但离计划定义的显著优势还差一点，且效率仍差很多；下一步必须做 materialization-free fused kernel 或换新的低内存 primitive，同时针对 classwise collapse 做 task repair。

## 12. 追加自修复 A：NoSync Optimizer + Task Repair

用户要求继续尝试解决后，新增了一轮 targeted real probe：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_nosync_taskrepair_all_20260505T230910Z \
  --fresh \
  --device auto \
  --candidates B0,H3,H4,G3,H7,H8,H9,H10,H11,H12 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 10000
```

本轮真实修改：

| 修改 | 目的 | 结果 |
|---|---|---|
| `FastAdamWNoSync` | 去掉 per-parameter CPU `.norm().cpu()` update 诊断同步；forward/backward/update 数学不变 | step 明显下降，但仍未进 S2 |
| `H7/H8/H9/H10` | 对 `H3` 做 no smoothing / low smoothing / lr high / lr low task repair | 没有超过原 `H3` |
| `H11/H12` | 对 `H4` 做 no smoothing / lr high task repair | step 接近 S2，但 memory 和部分 grad gate 未过 |

Task summary：

| candidate | val acc | test acc | val gap vs MLP | basic pass |
|---|---:|---:|---:|---:|
| `B0` | `0.8449` | `0.7958` | `0.0000` | reference |
| `H3` | `0.8634` | `0.8150` | `+0.0185` | 1 |
| `H4` | `0.8575` | `0.8078` | `+0.0126` | 1 |
| `G3` | `0.8531` | `0.8049` | `+0.0081` | 1 |
| `H10` | `0.8618` | `0.8092` | `+0.0169` | 1 |
| `H11` | `0.8546` | `0.8053` | `+0.0096` | 1 |
| `H12` | `0.8592` | `0.8116` | `+0.0143` | 1 |
| `H7` | `0.7340` | `0.6893` | `-0.1109` | 0 |
| `H8` | `0.8425` | `0.7984` | `-0.0024` | 0 |
| `H9` | `0.7762` | `0.7382` | `-0.0688` | 0 |

Significance / route：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `H3` | `+0.0185` | `+0.0134` | `3.68e-06` | `+0.0192` | `0.1455` | 0 |
| `H10` | `+0.0169` | `+0.0107` | `3.95e-04` | `+0.0135` | `0.1636` | 0 |
| `H12` | `+0.0143` | `+0.0096` | `1.18e-04` | `+0.0158` | `0.1702` | 0 |

Efficiency summary：

| candidate | memory ratio | step ratio | forward ratio | backward ratio | survivor |
|---|---:|---:|---:|---:|---|
| `H3` | `1.3031` | `2.8308` | `3.5106` | `3.0531` | FAIL |
| `H4` | `1.2956` | `1.9272` | `2.3125` | `1.9185` | FAIL |
| `H11` | `1.2956` | `1.9186` | `2.2743` | `1.9145` | FAIL |
| `H12` | `1.2956` | `1.9395` | `2.3412` | `1.9301` | FAIL |
| `G3` | `1.2983` | `2.6088` | `3.7090` | `2.4393` | FAIL |

NoSync 判断：

- `FastAdamWNoSync` 真实降低了 step 开销：`H4` 从主 run 的 `2.5621` 降到 `1.9272`，`H3` 从 `3.5237` 降到 `2.8308`。
- 但 S2 仍未达成：最佳 step 仍 `>1.50`，memory 仍约 `1.2956`。
- `H7/H8/H9` 没有修复 task，反而显著退化。
- `H11/H12` 虽然 step 接近，但 full gradient gate 只有 `5/6` pass，不能作为成功 candidate。

NoSync route：

```json
{
  "route": "R3-TaskPositiveButNotSignificant",
  "best_candidate_id": "H3",
  "best_val_acc": 0.8634114583333333,
  "val_gap_vs_MLP": 0.01848958333333332,
  "ci95_low": 0.013409830729166667,
  "holm_p": 3.6823078076324117e-06,
  "s2_pass": 0,
  "significant_task_pass": 0,
  "best_memory_ratio": 1.3031381078118751,
  "best_step_ratio": 2.8307560076407032,
  "primary_blocker": "significance_or_efficiency",
  "no_fake": true,
  "no_proxy": true
}
```

## 13. 追加自修复 B：Low-Rank Primitive Probe

NoSync 后仍未过 S2，因此继续尝试新的低内存 primitive。新增 `LowRankPoly2GateStack`，把 dense layer 的混合矩阵改为 factorized low-rank mixing，并跑 targeted real probe：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_lowrank_probe_all_20260505T231557Z \
  --fresh \
  --device auto \
  --candidates B0,H4,L1,L2,L3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

Low-rank candidates：

| candidate | primitive | 目的 |
|---|---|---|
| `L1` | rank-16 depth-2 poly2-gate | 降低 dense materialization 与参数 live set |
| `L2` | rank-32 depth-2 poly2-gate | 提升 rank 后检查 task 恢复 |
| `L3` | rank-32 depth-3 rbf-poly-exp head | 检查 task expressivity 上限 |

Task summary：

| candidate | val acc | test acc | val gap vs MLP | basic pass |
|---|---:|---:|---:|---:|
| `B0` | `0.8443` | `0.7949` | `0.0000` | reference |
| `H4` | `0.8595` | `0.8046` | `+0.0152` | 1 |
| `L1` | `0.8418` | `0.7849` | `-0.0025` | 0 |
| `L2` | `0.8471` | `0.7958` | `+0.0029` | 0 |
| `L3` | `0.8479` | `0.7984` | `+0.0036` | 1 |

Efficiency summary：

| candidate | memory ratio | step ratio | forward ratio | backward ratio | survivor |
|---|---:|---:|---:|---:|---|
| `H4` | `1.2956` | `2.5946` | `2.5482` | `2.8824` | FAIL |
| `L1` | `1.2674` | `2.7241` | `2.7298` | `2.7741` | FAIL |
| `L2` | `1.2828` | `2.7205` | `2.7407` | `2.7690` | FAIL |
| `L3` | `1.2960` | `3.7810` | `4.0672` | `3.9618` | FAIL |

Gradient correctness：

| candidate | grad pass rows | max relerr | 判断 |
|---|---:|---:|---|
| `H4` | `6/6` | `6.97e-05` | pass |
| `L1` | `3/6` | `2.30e-04` | fail |
| `L2` | `3/6` | `1.53e-04` | fail |
| `L3` | `4/6` | `1.05e-04` | fail |

Low-rank 判断：

- `L1/L2/L3` 没有保留 `H3/H4` 的 task signal，val gap 最高只有 `+0.0036`。
- memory ratio 从 dense `H4` 的 `1.2956` 最多降到 `1.2674`，但仍远高于 S2 memory gate。
- low-rank manual backward 当前未过完整 gradient gate，不能继续作为成功路线。

Low-rank route：

```json
{
  "route": "R3-TaskPositiveButNotSignificant",
  "best_candidate_id": "H4",
  "best_val_acc": 0.8595052083333333,
  "val_gap_vs_MLP": 0.015234374999999986,
  "ci95_low": 0.009635416666666667,
  "holm_p": 0.0060695003675319825,
  "s2_pass": 0,
  "significant_task_pass": 0,
  "best_memory_ratio": 1.2956392680925441,
  "best_step_ratio": 2.594626125876285,
  "primary_blocker": "significance_or_efficiency",
  "no_fake": true,
  "no_proxy": true
}
```

## 14. 追加审计与 Hash

追加 runs 的 no-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_nosync_taskrepair_all_20260505T230910Z` | `4618` | `0` |
| `v72_lowrank_probe_all_20260505T231557Z` | `1270` | `0` |

NoSync artifact hash：

| 文件 | SHA256 |
|---|---|
| `run.log` | `af73a29bcce7eadfc5f1ab4a43495cf448a54a735332f4cdd81b478301db5c2b` |
| `run_manifest.json` | `9f0a21832c0f943994814e30acba678f128178ed6d725b1f726c8bc9d8932dd3` |
| `p9_task_summary.csv` | `0df3d2373c438a8fc14386cc0f22faf3d874e447160cd3a5297e6af096f7c7a8` |
| `p1_significance_audit.csv` | `c81d0fa4a2e1e25289c1e13e6e9864755e6a2a66cc15bb5cba58613f3237959c` |
| `p2_full_gradient_correctness.csv` | `421a4e4fd507c6f4386592da476c8b9f2c67a4844f6190ddf523bd6844995f56` |
| `p10_efficiency_summary.csv` | `aebef52e2b64c93c6ecc8085c52f49e92b5e7694580502c3ec94a34dc0db22a1` |
| `route_decision.json` | `7eba72f6f545bb0197492be027883992ee63f99266c016f361a169bde48b564c` |
| `failure_table.csv` | `24b9268d52b88a3b8274572fa9218b954cce13148c3bfd7d7d2c4cb16f5c6b11` |
| `provenance_audit.csv` | `0f510388db73cddf3a02028c0b991c3d9ae4a7a6886f4bb1e412bb668fe443ac` |

Low-rank artifact hash：

| 文件 | SHA256 |
|---|---|
| `run.log` | `4e4e58257b6b5dbf6815f0a12172646bb966d86785e9facafabca465db98c75a` |
| `run_manifest.json` | `83d83ac8c6a84135c97cbf70ae62c91da98bc35dc7ac070fd237d31364113418` |
| `p9_task_summary.csv` | `af50b73640796fc93708d4b6280f5e556e50480d29d0d2e6c404790f934404de` |
| `p1_significance_audit.csv` | `b60116e44b6cbf13e8e13dc153f57e06ccc0f9f85e797c036c2d48557c9494d3` |
| `p2_full_gradient_correctness.csv` | `26c88bb996109c1a8154979efd98c735a54265f96ee949144ec3186b53da8072` |
| `p10_efficiency_summary.csv` | `dc1fff2c4d57d9af1abc5950092d4afc267294419daf8950bdfb0192a6864dae` |
| `route_decision.json` | `bb49839efecfde4d12b736952dc734bceaa37dce6caf341378011a9b548b8adb` |
| `failure_table.csv` | `1371ef7978732cbb9f78c0262b39459378806bff5b40e149d87cf952cf2127ed` |
| `provenance_audit.csv` | `d18b20cf4c0dc5ec084d2a2e93e3943fc8f81d0f1bfc55ebbb4b96e3e9ba3ff6` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v72_real.py
```

已通过。

## 15. 追加后最终结论

综合主 run 与两轮自修复 run，v7.2 计划目标仍未达成：

1. `H3` 仍是最佳 strict task candidate，basic Beyond task signal 稳定存在，但 `+0.0185` 没有达到计划中的 `+0.02` significant threshold，classwise non-collapse 也没过。
2. NoSync fast optimizer 真实降低 step ratio，但只把较好的 dense strict candidates 推到约 `1.92`，仍高于 S2 的 `1.50`；memory 仍约 `1.2956`。
3. Task-side `H7/H8/H9/H10/H11/H12` 没有带来显著 Beyond pass；其中 no smoothing / high lr 明显破坏 task。
4. Low-rank primitive 没有保留 dense strict task signal，也未过完整 gradient correctness gate，不能作为 kernel-native success。
5. 当前 blocker 进一步收敛为：optimizer sync 已不是主因，剩余问题是 materialized forward/backward live set、kernel count、dense head/stack 的 fused backward，以及 classwise collapse。

最终一句话：

> 追加自修复后，v7.2 仍没有达成 “Significant Beyond-MLP + KernelNativeEfficiency”。NoSync 证明 update 诊断同步是一部分开销，但不是根因；low-rank primitive 也没有闭合 task/grad/efficiency。下一轮应直接做 materialization-free fused poly2-gate backward 或重新设计不会丢 task signal的 kernel-native primitive。

## 16. 追加自修复 C：Hidden Size Targeted Probe

为了检验 dense strict candidate 的 efficiency fail 是否主要来自 hidden size / capacity setting，追加 hidden-dim 32 probe：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_hidden32_probe_all_20260505T232327Z \
  --fresh \
  --device auto \
  --hidden-dim 32 \
  --candidates B0,H4,H11,G3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | basic pass |
|---|---:|---:|---:|---:|
| `B0` | `0.8417` | `0.7898` | `0.0000` | reference |
| `H4` | `0.8497` | `0.7948` | `+0.0081` | 1 |
| `H11` | `0.8439` | `0.7953` | `+0.0022` | 1 |
| `G3` | `0.8288` | `0.7775` | `-0.0129` | 0 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `H4` | `+0.0081` | `+0.0035` | `0.0738` | `+0.0049` | `0.1091` | 0 |
| `H11` | `+0.0022` | `-0.0043` | `1.0000` | `+0.0055` | `0.1818` | 0 |
| `G3` | `-0.0129` | `-0.0210` | `0.0976` | `-0.0124` | `0.1636` | 0 |

Efficiency：

| candidate | memory ratio | step ratio | forward ratio | backward ratio | survivor |
|---|---:|---:|---:|---:|---|
| `H11` | `1.2913` | `1.5886` | `2.4667` | `1.3745` | FAIL |
| `H4` | `1.2913` | `1.8857` | `3.8193` | `1.4671` | FAIL |
| `G3` | `1.2928` | `2.1290` | `3.9662` | `1.7201` | FAIL |

Hidden-size 判断：

- hidden-dim 32 没有保持 v7.2 的 significant task signal，`H4` mean val gap 只有 `+0.0081`。
- step ratio 有下降，`H11` 到 `1.5886`，但仍未过 S2 step gate。
- memory ratio 基本仍在 `1.29` 左右，说明 hidden size 不是 memory ratio 的主因。
- 该 probe 不能作为 v7.2 成功结论，只能作为 blocker 定位：capacity shrink 会损失 task，而 memory 不随之闭合。

Route：

```json
{
  "route": "R3-TaskPositiveButNotSignificant",
  "best_candidate_id": "H4",
  "best_val_acc": 0.8497395833333333,
  "val_gap_vs_MLP": 0.008072916666666652,
  "s2_pass": 0,
  "significant_task_pass": 0,
  "best_memory_ratio": 1.291309544304493,
  "best_step_ratio": 1.8857319924143328,
  "no_fake": true,
  "no_proxy": true
}
```

## 17. 追加自修复 D：SGD Optimizer-State Memory Probe

Hidden-size probe 后 memory ratio 仍未闭合，因此继续检验 AdamW optimizer state 是否是 memory blocker。新增 `S1/S2/S3/S4` 候选，使用无动量 SGD update；这是实测 code path，不写假 ratio。

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_sgd_memory_probe_all_20260505T232556Z \
  --fresh \
  --device auto \
  --candidates B0,H4,H11,S1,S2,S3,S4 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

SGD candidates：

| candidate | update | lr multiplier | label smoothing |
|---|---|---:|---:|
| `S1` | SGD | `3.0` | `0.05` |
| `S2` | SGD | `10.0` | `0.05` |
| `S3` | SGD | `30.0` | `0.05` |
| `S4` | SGD | `10.0` | `0.0` |

Task summary：

| candidate | val acc | test acc | val gap vs MLP | basic pass |
|---|---:|---:|---:|---:|
| `B0` | `0.8443` | `0.7949` | `0.0000` | reference |
| `H4` | `0.8595` | `0.8046` | `+0.0152` | 1 |
| `H11` | `0.8535` | `0.8068` | `+0.0092` | 1 |
| `S1` | `0.7298` | `0.6758` | `-0.1145` | 0 |
| `S2` | `0.8064` | `0.7583` | `-0.0379` | 0 |
| `S3` | `0.8382` | `0.7911` | `-0.0061` | 0 |
| `S4` | `0.8099` | `0.7620` | `-0.0344` | 0 |

Efficiency：

| candidate | memory ratio | step ratio | forward ratio | backward ratio | update ratio | survivor |
|---|---:|---:|---:|---:|---:|---|
| `H4` | `1.2956` | `2.1640` | `2.4723` | `2.2610` | `2.2146` | FAIL |
| `H11` | `1.2956` | `2.1455` | `2.4047` | `2.2517` | `2.2044` | FAIL |
| `S1` | `1.2724` | `1.7065` | `2.4108` | `2.2367` | `0.5566` | FAIL |
| `S2` | `1.2724` | `1.7378` | `2.4439` | `2.2800` | `0.5674` | FAIL |
| `S3` | `1.2724` | `1.7026` | `2.4055` | `2.2329` | `0.5539` | FAIL |
| `S4` | `1.2724` | `1.7198` | `2.3762` | `2.2729` | `0.5604` | FAIL |

Gradient correctness：

| candidate | grad pass rows | max relerr | 判断 |
|---|---:|---:|---|
| `H4` | `6/6` | `6.89e-05` | pass |
| `H11` | `5/6` | `1.28e-04` | fail |
| `S1` | `6/6` | `9.64e-05` | pass |
| `S2` | `6/6` | `6.30e-05` | pass |
| `S3` | `4/6` | `1.37e-04` | fail |
| `S4` | `6/6` | `7.69e-05` | pass |

SGD 判断：

- SGD 确实降低 update overhead：`S1/S2/S3/S4` 的 update ratio 约 `0.56`，低于 AdamW strict candidates 的约 `2.20`。
- SGD 也小幅降低 memory ratio：从 `1.2956` 降到 `1.2724`。
- 但 S2 仍未过：memory 仍远高于 `1.05`，step 仍在 `1.70-1.74`。
- SGD 破坏 task signal：所有 `S*` candidate 都没有 basic pass。
- 因此 AdamW optimizer state 不是 v7.2 memory/step fail 的根因；根因仍是 forward/backward materialization 与 dense primitive 本身。

Route：

```json
{
  "route": "R3-TaskPositiveButNotSignificant",
  "best_candidate_id": "H4",
  "best_val_acc": 0.8595052083333333,
  "val_gap_vs_MLP": 0.015234374999999986,
  "s2_pass": 0,
  "significant_task_pass": 0,
  "best_memory_ratio": 1.2956392680925441,
  "best_step_ratio": 2.1640194646875,
  "no_fake": true,
  "no_proxy": true
}
```

## 18. 最终追加审计与结论

追加 runs 的 no-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_hidden32_probe_all_20260505T232327Z` | `1018` | `0` |
| `v72_sgd_memory_probe_all_20260505T232556Z` | `1775` | `0` |

关键 hash：

| run | file | SHA256 |
|---|---|---|
| hidden32 | `run.log` | `c130f21388b7f9cf8174fe026637c13a4144d41aa49cf64102a9f51336b6c250` |
| hidden32 | `run_manifest.json` | `0fe6fcaaf8599e62bbd7ca8dff1f20edb1c30c264fed483dca187c32ea595c4b` |
| hidden32 | `route_decision.json` | `a03e273375febeb5ca67696c804ae0f7930ca2f912f1194f10b29992e2588931` |
| hidden32 | `p9_task_summary.csv` | `fa139ad4dbceccc4c65e4ed7af1040e3fad4798cbf40600bdc4d65a3e747800c` |
| hidden32 | `p10_efficiency_summary.csv` | `3b432af86e344669c3aff2e558184d68e1fb4384ad18d117d80dea5f974939f2` |
| sgd | `run.log` | `3f706fb65bdb49b4f0c8da69d42336f866d49e3d0c3608d0e0c478312a3cd2c2` |
| sgd | `run_manifest.json` | `6fccf138679673b1964c544584699ab00c6fd88a0c21d621905fdad554051235` |
| sgd | `route_decision.json` | `4605003ea7f65b7b6355e23198583d867e7559635d32a4e463cda12c1ec21955` |
| sgd | `p9_task_summary.csv` | `c7ddc3597f0455d4b12d92392c9e67713fce6ec7bc914143f636a7f5739c502c` |
| sgd | `p10_efficiency_summary.csv` | `c16f28809033d73b4aaec408838946f877dc39e6be4911b62503b49d2167f670` |

追加后的最终结论仍然不变：

1. v7.2 计划目标没有达成。
2. hidden-size shrink 不能解决问题：它会损失 task signal，memory ratio 仍约 `1.29`。
3. SGD optimizer-state repair 不能解决问题：它降低 update 和少量 memory，但 task 失败且 memory/step 仍不过 S2。
4. 结合 NoSync / low-rank / hidden32 / SGD 四轮自修复，当前可以排除“单纯 optimizer update overhead”“hidden size 太大”“低秩混合”“AdamW 状态”作为充分解决路线。
5. 剩余必须做真正的 materialization-free fused forward/backward，或者设计一个不丢 task signal、低 live-set、低 kernel-count 的新 primitive。

最终一句话：

> v7.2 到目前为止已经真实尝试 task repair、NoSync、low-rank、hidden shrink、SGD state repair，但仍未达到 SignificantTaskPass + S2Pass。当前失败不是数据或 gate 误差，而是 dense strict KAN primitive 本身还没有 kernel-native efficiency 实现。

## 19. 追加自修复 E：Cached Hidden-Y Backward Probe

SGD 与 hidden shrink 都没有闭合 S2，因此继续做更贴近 kernelization blocker 的修复：新增 `CachedDensePoly2GateStack`，在 forward 中缓存 hidden pre-activation $y$，backward 中直接使用缓存计算 SiLU 导数，避免原 dense stack backward 为隐藏层导数重算一次 layer forward。该路径仍是 manual forward/backward/update，没有使用 loss.backward，也没有写 fake ratio。

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_cached_backward_probe_all_20260505T233025Z \
  --fresh \
  --device auto \
  --candidates B0,H3,H4,H11,C1,C2,C3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

Cached candidates：

| candidate | 对照 | 修改 |
|---|---|---|
| `C1` | `H4` | depth-2 poly2-gate stack，缓存 hidden $y$ |
| `C2` | `H11` | depth-2 no-smoothing，缓存 hidden $y$ |
| `C3` | `H3` | depth-3 rbf head，缓存 hidden $y$ |

Task summary：

| candidate | val acc | test acc | val gap vs MLP | basic pass |
|---|---:|---:|---:|---:|
| `B0` | `0.8443` | `0.7949` | `0.0000` | reference |
| `H3` | `0.8602` | `0.8152` | `+0.0159` | 1 |
| `H4` | `0.8595` | `0.8046` | `+0.0152` | 1 |
| `H11` | `0.8535` | `0.8068` | `+0.0092` | 1 |
| `C1` | `0.8599` | `0.8112` | `+0.0156` | 1 |
| `C2` | `0.8539` | `0.8092` | `+0.0096` | 1 |
| `C3` | `0.8673` | `0.8194` | `+0.0230` | 1 |

`C3` 分 dataset：

| dataset | val gap vs MLP |
|---|---:|
| MNIST | `+0.0297` |
| Fashion-MNIST | `+0.0164` |
| KMNIST | `+0.0230` |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `C3` | `+0.0230` | `+0.0159` | `3.48e-04` | `+0.0245` | `0.1273` | 0 |
| `H3` | `+0.0159` | `+0.0099` | `5.44e-03` | `+0.0203` | `0.1455` | 0 |
| `C1` | `+0.0156` | `+0.0078` | `4.83e-02` | `+0.0163` | `0.1273` | 0 |
| `H4` | `+0.0152` | `+0.0096` | `7.97e-03` | `+0.0096` | `0.1277` | 0 |

Efficiency：

| candidate | memory ratio | step ratio | forward ratio | backward ratio | update ratio | survivor |
|---|---:|---:|---:|---:|---:|---|
| `C1` | `1.2956` | `1.8023` | `2.3218` | `1.6895` | `2.0911` | FAIL |
| `C2` | `1.2956` | `1.7987` | `2.2866` | `1.6954` | `2.0843` | FAIL |
| `C3` | `1.3069` | `2.4448` | `3.4423` | `2.3578` | `2.4913` | FAIL |
| `H3` | `1.3031` | `2.6310` | `3.3703` | `2.6381` | `2.6496` | FAIL |
| `H4` | `1.2956` | `1.8282` | `2.2341` | `1.7834` | `2.0566` | FAIL |
| `H11` | `1.2956` | `1.8608` | `2.2281` | `1.8309` | `2.0808` | FAIL |

Gradient correctness：

| candidate | grad pass rows | max relerr | 判断 |
|---|---:|---:|---|
| `C1` | `6/6` | `9.88e-05` | pass |
| `C2` | `5/6` | `1.31e-04` | fail |
| `C3` | `6/6` | `6.02e-05` | pass |
| `H3` | `6/6` | `6.97e-05` | pass |
| `H4` | `6/6` | `6.89e-05` | pass |

Cached-backward 判断：

- `C3` 是本轮最有价值的新结果：macro val gap 达到 `+0.0230`，CI95 low `+0.0159`，Holm p `3.48e-04`，test gap `+0.0245`。
- 但 `C3` 仍未通过 v7.2 的完整 SignificantTaskPass，因为 classwise drop max `0.1273 > 0.05`。
- `C3` 仍未过 S2：memory `1.3069`，step `2.4448`。
- `C1/C2` 证明缓存 hidden $y$ 可以降低部分 backward/step，但幅度不足：best cached depth-2 step 约 `1.80`，仍高于 S2 `1.50`。
- 这轮把 blocker 进一步细化：task 侧已能达到 macro practical/statistical positive，但 classwise non-collapse 与 efficiency 仍未闭合。

Route：

```json
{
  "route": "R3-TaskPositiveButNotSignificant",
  "best_candidate_id": "C3",
  "best_val_acc": 0.8673177083333333,
  "val_gap_vs_MLP": 0.023046875000000022,
  "ci95_low": 0.015885416666666666,
  "holm_p": 0.000347846043437454,
  "s2_pass": 0,
  "significant_task_pass": 0,
  "best_memory_ratio": 1.3069108164142476,
  "best_step_ratio": 2.4447589674665378,
  "no_fake": true,
  "no_proxy": true
}
```

No-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_cached_backward_probe_all_20260505T233025Z` | `1774` | `0` |

关键 hash：

| file | SHA256 |
|---|---|
| `run.log` | `d100a4aad5aad86c75f6c1de5c07184633ba137e28e4727c3200a16e0ab8e861` |
| `run_manifest.json` | `ad1b0b72309c33cfe5f42d409131ab204c20d3a7374ccb258585be15cfbe8fc6` |
| `route_decision.json` | `6db5a4b3bcd67aa83e96d90fa4635b826e38c651cd06388f885a5ed05eb7d435` |
| `p9_task_summary.csv` | `7766b313eab8812ead3c016111f180756c234ac725b2625ab8296bb25842188f` |
| `p1_significance_audit.csv` | `c7bb57ffbfd91a569c4d710295918e8eca172b9e06ac01159d9c1305f86e4b23` |
| `p2_full_gradient_correctness.csv` | `1cbe0b32010c56894f47398412a2ac54d70978fea0584487f54c398c29a8645e` |
| `p10_efficiency_summary.csv` | `c1bf1cca1304113e894a3748ccd0fb0ead5fb8a270eeee15867279574061ad72` |
| `failure_table.csv` | `912d72b52afd31e4acb9226e87416dc7bd394db531a445d41d3b9f8fe5ab7d53` |
| `provenance_audit.csv` | `4032bd4811cffe7d475be6a4a993e1a5d71a5ba75b3ecb09eea12cce42ad59a2` |

## 20. Cached-backward probe 后的阶段结论

截至 cached-backward probe，v7.2 计划目标仍未达成，但结论比上一轮更具体：

1. `C3` 首次达到 macro practical/statistical positive：val gap `+0.0230`，CI95 low `+0.0159`，Holm p `3.48e-04`，test gap `+0.0245`。
2. `C3` 仍不是 SignificantTaskPass，因为 classwise drop max `0.1273` 未过 non-collapse gate。
3. `C3` 仍不是 S2：memory `1.3069`，step `2.4448`。
4. cached hidden-y backward 是有效方向，但只去掉一层 backward 重算不够；仍需要 fused transform+mix backward、减少 materialized live set、或 classwise-aware task repair。
5. 当前 blocker 从此前的 “gap 不够 + efficiency” 更新为：`classwise_noncollapse_fail + efficiency_kernelization_fail`。

最终一句话：

> v7.2 目标仍未达成；但新增 `C3` 证明 strict KAN macro accuracy 已能超过 `+2%` 且统计显著。失败点现在更清楚：classwise collapse 和 S2 efficiency 没闭合，下一步应围绕 classwise repair 与真正 fused/materialization-free backward 继续。

## 21. 追加确认：C3 10-seed full confirmation

为确认 `C3` 的 `+2%` macro signal 不是 5-seed 噪声，追加运行：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_c3_10seed_confirm_all_20260505T233357Z \
  --fresh \
  --device auto \
  --candidates B0,C3,C1,H3,H4 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 10000
```

运行目录：

```text
results/real_rerun_20260505/v72_c3_10seed_confirm_all_20260505T233357Z
```

覆盖：

```text
task rows = 150
seeds = 0..9
datasets = MNIST,Fashion-MNIST,KMNIST
candidates = B0,C1,C3,H3,H4
efficiency shapes = 3 datasets x 3 batch sizes
grad checks = 3 datasets x 2 batch sizes
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8449` | `0.7958` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `C1` | `0.8592` | `0.8112` | `+0.0143` | F-MNIST `+0.0084`, KMNIST `+0.0139`, MNIST `+0.0207` | 1 |
| `C3` | `0.8663` | `0.8166` | `+0.0214` | F-MNIST `+0.0090`, KMNIST `+0.0258`, MNIST `+0.0293` | 1 |
| `H3` | `0.8634` | `0.8150` | `+0.0185` | F-MNIST `+0.0096`, KMNIST `+0.0217`, MNIST `+0.0242` | 1 |
| `H4` | `0.8575` | `0.8078` | `+0.0126` | F-MNIST `+0.0086`, KMNIST `+0.0104`, MNIST `+0.0188` | 1 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `C1` | `+0.0143` | `+0.0090` | `1.88e-04` | `+0.0154` | `0.1600` | 0 |
| `C3` | `+0.0214` | `+0.0158` | `2.19e-07` | `+0.0208` | `0.1400` | 0 |
| `H3` | `+0.0185` | `+0.0134` | `1.53e-06` | `+0.0192` | `0.1455` | 0 |
| `H4` | `+0.0126` | `+0.0077` | `4.63e-04` | `+0.0120` | `0.1400` | 0 |

Efficiency：

| candidate | memory ratio | step ratio | forward ratio | backward ratio | update ratio | survivor |
|---|---:|---:|---:|---:|---:|---|
| `C1` | `1.2956` | `2.4402` | `2.7913` | `2.6808` | `2.4155` | FAIL |
| `C3` | `1.3069` | `3.3751` | `4.1594` | `3.8419` | `2.9168` | FAIL |
| `H3` | `1.3031` | `4.5622` | `4.0998` | `6.3714` | `3.1123` | FAIL |
| `H4` | `1.2956` | `2.5418` | `2.7246` | `2.9264` | `2.4170` | FAIL |

Full gradient correctness：

| candidate | grad pass rows | max relerr |
|---|---:|---:|
| `C1` | `6/6` | `9.88e-05` |
| `C3` | `6/6` | `6.02e-05` |
| `H3` | `6/6` | `6.97e-05` |
| `H4` | `6/6` | `6.89e-05` |

Classwise failure 定位：

| drop | dataset | seed | class | B0 class acc | C3 class acc |
|---:|---|---:|---:|---:|---:|
| `0.1400` | Fashion-MNIST | 5 | 4 | `0.8400` | `0.7000` |
| `0.1277` | Fashion-MNIST | 9 | 6 | `0.5532` | `0.4255` |
| `0.1273` | Fashion-MNIST | 7 | 4 | `0.6909` | `0.5636` |
| `0.1273` | Fashion-MNIST | 0 | 4 | `0.8000` | `0.6727` |

判断：

- `C3` 的 macro practical/statistical signal 在 10 seeds 下仍成立：val gap `+0.0214`，CI95 low `+0.0158`，Holm p `2.19e-07`，test gap `+0.0208`。
- `C3` 仍未通过完整 `SignificantTaskPass`，原因不是 macro gap 或 p-value，而是 classwise non-collapse：max drop `0.1400 > 0.05`。
- classwise drop 主要集中在 Fashion-MNIST，尤其 class 4。
- `C3` 仍未进入 S2：memory `1.3069`，step `3.3751`。

Route：

```json
{
  "route": "R3-TaskPositiveButNotSignificant",
  "best_candidate_id": "C3",
  "best_candidate": "H3-cached-hidden-y-rbf-head",
  "best_val_acc": 0.8662760416666667,
  "best_test_acc": 0.8166015625,
  "val_gap_vs_MLP": 0.021354166666666674,
  "test_gap_vs_MLP": 0.020833333333333332,
  "ci95_low": 0.0158203125,
  "holm_p": 2.188460844944466e-07,
  "strict_pass": 1,
  "grad_pass": 1,
  "s2_pass": 0,
  "significant_task_pass": 0,
  "no_fake": true,
  "no_proxy": true
}
```

No-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_c3_10seed_confirm_all_20260505T233357Z` | `2322` | `0` |

关键 hash：

| file | SHA256 |
|---|---|
| `run.log` | `a67fa90fe6c5495c2fb8a4780cbb6f05e59a85dabccaedb6a7b9b59a7e4fdcc3` |
| `run_manifest.json` | `1d9f836e03f95b15bfec3c63905124dbd8113ab09342049edaf71dc9684490ce` |
| `p9_task_summary.csv` | `0ab4c9da4ed4ff3580fdd75fe4146d07e3e761893982894c9359aab90aae47fb` |
| `p1_significance_audit.csv` | `2cbf0ad4415d3da574147a1f82655e43f8dd02ee5b57eed99ec119ecc87e4918` |
| `p2_full_gradient_correctness.csv` | `2132642434e591101f2a78383b979f9d4e30d8d31a1442c78b43c59ce5f39f40` |
| `p10_efficiency_summary.csv` | `58d2f216d0ee22dccdc2e360634dd2e02abaa3f238826a887eacdd653010ee4a` |
| `route_decision.json` | `cffd5c65232d44277ac8105a055478513f5bc0a9c5414d35174db2b2618b8729` |
| `failure_table.csv` | `d477964de88901322788da14e672f00e04b0252670a9fe5f82cd8b76dbdd2fef` |
| `provenance_audit.csv` | `d82ebd27e7c9c97e77ed2c6372fba2e7e2aa978bf5540db8b5f0f1573f9ab385` |

## 22. 追加自修复：H3 classwise / regularization repair probe

针对 `C3/H3` 的 classwise non-collapse failure，继续尝试不改变 PureKAN/manual path 的轻量 repair：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_h3_classwise_repair_10seed_20260505T233914Z \
  --fresh \
  --device auto \
  --candidates B0,H3,H7,H8,H9,H10 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128 \
  --bench-warmup 3 \
  --bench-reps 15 \
  --grad-batch-sizes 8 \
  --bootstrap-reps 5000
```

运行目录：

```text
results/real_rerun_20260505/v72_h3_classwise_repair_10seed_20260505T233914Z
```

候选含义：

| candidate | 修改 |
|---|---|
| `H7` | H3 no smoothing |
| `H8` | H3 label smoothing `0.02` |
| `H9` | H3 lr high |
| `H10` | H3 lr low |

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8449` | `0.7958` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `H3` | `0.8634` | `0.8150` | `+0.0185` | F-MNIST `+0.0096`, KMNIST `+0.0217`, MNIST `+0.0242` | 1 |
| `H10` | `0.8618` | `0.8092` | `+0.0169` | F-MNIST `+0.0133`, KMNIST `+0.0160`, MNIST `+0.0213` | 1 |
| `H8` | `0.8425` | `0.7984` | `-0.0024` | F-MNIST `+0.0084`, KMNIST `-0.0328`, MNIST `+0.0172` | 0 |
| `H9` | `0.7762` | `0.7382` | `-0.0687` | F-MNIST `+0.0139`, KMNIST `-0.1748`, MNIST `-0.0453` | 0 |
| `H7` | `0.7340` | `0.6893` | `-0.1109` | F-MNIST `-0.0709`, KMNIST `-0.1664`, MNIST `-0.0955` | 0 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `H3` | `+0.0185` | `+0.0134` | `2.05e-06` | `+0.0192` | `0.1455` | 0 |
| `H10` | `+0.0169` | `+0.0107` | `2.21e-04` | `+0.0135` | `0.1636` | 0 |
| `H8` | `-0.0024` | `-0.0387` | `1.0` | `+0.0026` | `0.7600` | 0 |
| `H9` | `-0.0687` | `-0.1454` | `5.43e-01` | `-0.0576` | `0.8696` | 0 |
| `H7` | `-0.1109` | `-0.1731` | `1.39e-02` | `-0.1065` | `0.9815` | 0 |

Gradient correctness：

| candidate | grad pass rows | max relerr |
|---|---:|---:|
| `H3` | `3/3` | `6.97e-05` |
| `H7` | `3/3` | `5.86e-05` |
| `H8` | `3/3` | `3.41e-05` |
| `H9` | `3/3` | `3.71e-05` |
| `H10` | `3/3` | `5.36e-05` |

判断：

- 轻量 classwise/regularization repair 没有解决 v7.2 blocker。
- `H10` 改善了 Fashion-MNIST dataset gap 到 `+0.0133`，但 macro gap 降到 `+0.0169`，且 classwise drop 变成 `0.1636`，更差。
- `H8` 低 smoothing、`H7` no smoothing、`H9` lr high 均造成 KMNIST 或整体 task collapse。
- 因此 classwise non-collapse 不能靠现有 H3 小参数修补解决，下一步需要显式 classwise-aware objective、data-balanced schedule，或结构层面的 primitive 改动。

No-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_h3_classwise_repair_10seed_20260505T233914Z` | `2684` | `0` |

关键 hash：

| file | SHA256 |
|---|---|
| `run.log` | `cfb59f409e9cfda2bae6dd02d374e517b54becb7a9141f083e2f4819bcd54d86` |
| `run_manifest.json` | `94b345ae4b640d6d09c63fb7fe8d0d3998ba0d678b3e439c7e4d34a86e93a349` |
| `p9_task_summary.csv` | `66842c58453c4a31584e783a0fef5f2deecf870b14bff56c828d3c8a6c6acfc9` |
| `p1_significance_audit.csv` | `e582ae1988207463eebfa148032bdb35d774a76fe30ade8eec474831b2356d6c` |
| `p10_efficiency_summary.csv` | `023a5e19c6e276b855906b78a657d931eb14e7a06a74b1345a43abd0e3c8a35a` |
| `route_decision.json` | `a2fc88bb1c3c61b386bb55545150a0eaf14e455d4657e7c1a2950706395915f2` |
| `failure_table.csv` | `f7f26995cf1bc58d271e207f9c851f5f24af27c2c189feac8e4de7eac55f6b9f` |
| `provenance_audit.csv` | `9b6a1a2524251e49aee4285177f83060161f23beddb3f58c36850cef1e23756b` |

## 23. 追加自修复：class-weighted C3 repair probe

在 H3 smoothing/lr 小修无效后，继续尝试 classwise-aware objective。依据 C3 10-seed audit，主要 drop 来自 Fashion-MNIST class 4，其次 class 6。因此新增 `W1-W4`：

| candidate | 修改 |
|---|---|
| `W1` | C3 + Fashion-MNIST class 4 weight `1.5` |
| `W2` | C3 + Fashion-MNIST class 4/6 weight `1.5` |
| `W3` | C3 + Fashion-MNIST class 4 weight `2.0` |
| `W4` | C3 + Fashion-MNIST class 4/6 weight `2.0` |

实现说明：

- weighted CE 只用于新 `W*` candidates。
- manual gradient 与 autograd gradient 都使用同一个 weighted-smoothed CE。
- 不增加 trainable params，不引入 non-KAN head。
- 所有 dataset loader 仍为 `allow_fake_data=False`。

第一次 class-weighted probe：

```text
results/real_rerun_20260505/v72_classweighted_repair_10seed_20260505T234719Z
```

这次 run 暴露了 protocol 问题：`W*` 使用不同 candidate id 初始化，导致 MNIST/KMNIST 也发生初始化噪声。该 run 只作为调试记录，不作为 class weighting 有效性结论。其关键结论是：没有任何 `W*` 通过 significant gate，且 no-fake/no-proxy 审计为 `2685` 行、nonzero `0`。

随后修正 protocol：`W*` 与 `C3` 使用同一初始化种子，只让 Fashion-MNIST weighted CE 成为变量。

最终 matched-init run：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_classweighted_matchedinit_10seed_20260505T235053Z \
  --fresh \
  --device auto \
  --candidates B0,C3,W1,W2,W3,W4 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128 \
  --bench-warmup 3 \
  --bench-reps 15 \
  --grad-batch-sizes 8 \
  --bootstrap-reps 5000
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8449` | `0.7958` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `C3` | `0.8663` | `0.8166` | `+0.0214` | F-MNIST `+0.0090`, KMNIST `+0.0258`, MNIST `+0.0293` | 1 |
| `W1` | `0.8636` | `0.8182` | `+0.0187` | F-MNIST `+0.0068`, KMNIST `+0.0225`, MNIST `+0.0268` | 1 |
| `W2` | `0.8645` | `0.8170` | `+0.0196` | F-MNIST `+0.0096`, KMNIST `+0.0225`, MNIST `+0.0268` | 1 |
| `W3` | `0.8647` | `0.8166` | `+0.0198` | F-MNIST `+0.0102`, KMNIST `+0.0225`, MNIST `+0.0268` | 1 |
| `W4` | `0.8639` | `0.8153` | `+0.0189` | F-MNIST `+0.0076`, KMNIST `+0.0225`, MNIST `+0.0268` | 1 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `C3` | `+0.0214` | `+0.0158` | `2.74e-07` | `+0.0208` | `0.1400` | 0 |
| `W1` | `+0.0187` | `+0.0130` | `1.08e-05` | `+0.0225` | `0.1702` | 0 |
| `W2` | `+0.0196` | `+0.0141` | `2.38e-06` | `+0.0212` | `0.2000` | 0 |
| `W3` | `+0.0198` | `+0.0146` | `7.81e-07` | `+0.0208` | `0.1915` | 0 |
| `W4` | `+0.0189` | `+0.0134` | `3.13e-06` | `+0.0195` | `0.1964` | 0 |

Worst classwise drops：

| candidate | max drop | location |
|---|---:|---|
| `C3` | `0.1400` | Fashion-MNIST seed 5 class 4 |
| `W1` | `0.1702` | Fashion-MNIST seed 4 class 2 |
| `W2` | `0.2000` | Fashion-MNIST seed 5 class 4 |
| `W3` | `0.1915` | Fashion-MNIST seed 4 class 2 |
| `W4` | `0.1964` | Fashion-MNIST seed 8 class 0 |

Gradient correctness：

| candidate | grad pass rows | max relerr |
|---|---:|---:|
| `C3` | `3/3` | `6.02e-05` |
| `W1` | `3/3` | `6.02e-05` |
| `W2` | `3/3` | `6.02e-05` |
| `W3` | `3/3` | `6.02e-05` |
| `W4` | `3/3` | `6.93e-05` |

Efficiency（本 probe 只测 batch size 128，不能替代 full efficiency 结论）：

| candidate | memory ratio | step ratio | survivor |
|---|---:|---:|---|
| `C3` | `1.1177` | `2.7738` | FAIL |
| `W1` | `1.1177` | `2.7771` | FAIL |
| `W2` | `1.1177` | `2.7633` | FAIL |
| `W3` | `1.1177` | `3.9641` | FAIL |
| `W4` | `1.1177` | `2.7573` | FAIL |

判断：

- Static class weighting 没有修复 classwise non-collapse。
- `W*` 均保持 gradient correctness，但 macro gap 全部低于 `C3`。
- `W*` 的 classwise drop max 反而从 C3 的 `0.1400` 变为 `0.1702-0.2000`。
- 加权 Fashion-MNIST class 4/6 会把 drop 转移到其他 Fashion-MNIST 类，不能作为 v7.2 解法。
- 当前 task-side 修复需要更细的 objective 或 schedule，例如 per-seed/per-class curriculum、min-class constrained selection，或结构上减少类别间混淆；静态 class weights 不够。

No-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_classweighted_repair_10seed_20260505T234719Z` | `2685` | `0` |
| `v72_classweighted_matchedinit_10seed_20260505T235053Z` | `2685` | `0` |

关键 hash（matched-init run）：

| file | SHA256 |
|---|---|
| `run.log` | `8a5849bf786b313822672611e32d5a5a77dc83d03f8f3533a61b95d25df68252` |
| `run_manifest.json` | `407968015c32897f622984e6f1ba8fa9478c2d74563b6d2045d75a89b47471db` |
| `p9_task_summary.csv` | `eb712e0785f93e12947c69721aca455c55ca53d08f8d6b8f2347f768389b0fc3` |
| `p1_significance_audit.csv` | `3ed02ebedca545f69720b4b5a6be2138589ce617c07cc29663322a5c22ac0a07` |
| `p2_full_gradient_correctness.csv` | `5d52118e282a675de277ad39213c709fc6a79c08058eaa6eb27ef9638e885975` |
| `p10_efficiency_summary.csv` | `f6882347b126fec182134d29f034f346cf4535d9de7cb803985a9d81e87112e3` |
| `route_decision.json` | `ad0380e4198336ee87c6433adbb567c79b7fe684a9a8c87ca24cccb254daf009` |
| `failure_table.csv` | `ecb2e525a4757674664fe514dc789e308c719dc938fa2cb4a29ee55a87149f2a` |
| `provenance_audit.csv` | `2ae0bb5b8a8f7292f19ac7dabefceefdbe553d58254d37698075e84ac2fd95e1` |

## 24. Class-weighted 后阶段结论

截至 C3 10-seed confirmation 与 H3 classwise repair probe，v7.2 计划目标仍未达成。

1. `C3` 是当前最佳 route candidate：strict pass、grad pass、basic task pass 均成立。
2. `C3` 在 10 seeds 下确认 macro task positive：val gap `+0.0214`，CI95 low `+0.0158`，Holm p `2.19e-07`，test gap `+0.0208`。
3. `C3` 仍不是完整 `SignificantTaskPass`，因为 classwise non-collapse fail：max drop `0.1400`，主要发生在 Fashion-MNIST class 4。
4. `C3` 仍不是 S2/S1：memory `1.3069`，step `3.3751`，efficiency blocker 未闭合。
5. 失败后已继续尝试 H3 no-smoothing、低 smoothing、lr high、lr low；这些 repair 没有解决 classwise gate，且部分导致 KMNIST 或整体 task collapse。
6. 进一步尝试 class-weighted C3 repair 后，`W1-W4` 均未通过 SignificantTaskPass；static class weighting 会降低 macro gap或转移 classwise drop。
7. 当前 blocker 明确为：`classwise_noncollapse_fail + efficiency_kernelization_fail`。

最终一句话：

> v7.2 仍未达成完整目标。`C3` 已经真实确认 macro accuracy 显著正向，但还不能称为 Significant Beyond-MLP candidate：classwise collapse 未过，S2 efficiency 未过；继续小调 smoothing/lr 与 static class weighting 都无效，下一步需要更强的 classwise-constrained training 或结构性 primitive repair，并同步推进 materialization-free fused/kernel-native implementation。

## 25. 追加自修复：hidden-dim compression efficiency probe

classwise repair 未闭合后，继续尝试 efficiency 方向：降低 hidden dim，观察 `C3/C1/H4` 是否能接近 S2，同时检查 task signal 是否保留。

### 25.1 hidden dim 48

运行：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_hidden48_probe_5seed_20260505T235546Z \
  --fresh \
  --device auto \
  --hidden-dim 48 \
  --candidates B0,C3,C1,H4 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8 \
  --bootstrap-reps 5000
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8435` | `0.7948` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `C1` | `0.8573` | `0.8061` | `+0.0138` | F-MNIST `+0.0020`, KMNIST `+0.0156`, MNIST `+0.0238` | 1 |
| `C3` | `0.8555` | `0.8095` | `+0.0120` | F-MNIST `+0.0090`, KMNIST `+0.0125`, MNIST `+0.0145` | 1 |
| `H4` | `0.8547` | `0.8072` | `+0.0112` | F-MNIST `+0.0047`, KMNIST `+0.0156`, MNIST `+0.0133` | 1 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `C1` | `+0.0138` | `+0.0072` | `1.74e-02` | `+0.0113` | `0.1842` | 0 |
| `C3` | `+0.0120` | `+0.0078` | `1.18e-03` | `+0.0147` | `0.0755` | 0 |
| `H4` | `+0.0112` | `+0.0065` | `7.68e-03` | `+0.0124` | `0.1042` | 0 |

Efficiency：

| candidate | memory ratio | step ratio | forward ratio | backward ratio | survivor |
|---|---:|---:|---:|---:|---|
| `C1` | `1.2935` | `2.3235` | `2.6074` | `2.3029` | FAIL |
| `C3` | `1.3014` | `3.5276` | `3.9267` | `3.8558` | FAIL |
| `H4` | `1.2935` | `2.4174` | `2.5608` | `2.5071` | FAIL |

判断：

- hidden dim 48 没有接近 S2：best memory `1.2935`，best step `2.3235`。
- task signal 明显低于 C3 hidden64：best val gap 只有 `+0.0138`，未达到 practical significance `+0.02`。
- hidden48 不是解法。

No-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_hidden48_probe_5seed_20260505T235546Z` | `1009` | `0` |

关键 hash：

| file | SHA256 |
|---|---|
| `run.log` | `3ef4f30fc5729a720def16401ef76bb6c303a398cab7c2405d77d86af678c4cd` |
| `run_manifest.json` | `3ab31a8c24c7c28247e1f4f11b42a892ab8f5f7e3b8c85d4322e8ba24726ac0e` |
| `p9_task_summary.csv` | `ccf2a1f3a29e848035662d071c9c3064ae903bff11f4b89fadea375fff37cce7` |
| `p1_significance_audit.csv` | `c641b8f3a8dc9c547e533c4984e33893451c044f80ce804a9f2df3687ef36d6c` |
| `p2_full_gradient_correctness.csv` | `ec022d40c00efe17889e53067981d341c50f98be7ea915a15e0852581710d31a` |
| `p10_efficiency_summary.csv` | `88de8afcfd17a32ecbdb697219a5b4d0e59c9691eb2bd1c733e03e730707b43d` |
| `route_decision.json` | `feaf6dbcdd6c1d61fa8299000f8ba25cb255bd87f1ff63a6f44e9a3889d9a720` |
| `provenance_audit.csv` | `4cb79cd2562691adb1bf5e3737434487de4a582af082dfaa8546350bc2bf7160` |

### 25.2 hidden dim 32

运行：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_hidden32_c1_probe_3seed_20260505T235710Z \
  --fresh \
  --device auto \
  --hidden-dim 32 \
  --candidates B0,C1,H4,C3 \
  --seeds 0,1,2 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8 \
  --bootstrap-reps 3000
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8470` | `0.7919` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `C1` | `0.8490` | `0.7878` | `+0.0020` | F-MNIST `-0.0091`, KMNIST `+0.0085`, MNIST `+0.0065` | 1 |
| `C3` | `0.8500` | `0.8021` | `+0.0030` | F-MNIST `+0.0026`, KMNIST `-0.0026`, MNIST `+0.0091` | 1 |
| `H4` | `0.8526` | `0.7964` | `+0.0056` | F-MNIST `-0.0007`, KMNIST `+0.0091`, MNIST `+0.0085` | 1 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `C1` | `+0.0020` | `-0.0056` | `1.0` | `-0.0041` | `0.1176` | 0 |
| `C3` | `+0.0030` | `-0.0050` | `1.0` | `+0.0102` | `0.1636` | 0 |
| `H4` | `+0.0056` | `0.0000` | `1.0` | `+0.0046` | `0.1091` | 0 |

Efficiency：

| candidate | memory ratio | step ratio | forward ratio | backward ratio | survivor |
|---|---:|---:|---:|---:|---|
| `C1` | `1.2913` | `1.9798` | `2.2122` | `1.9283` | FAIL |
| `C3` | `1.2963` | `3.1107` | `3.4103` | `3.3486` | FAIL |
| `H4` | `1.2913` | `2.2481` | `2.1681` | `2.5610` | FAIL |

判断：

- hidden dim 32 仍没有进入 S2：best step `1.9798 > 1.50`，memory `1.2913 > 1.05`。
- task gap 基本消失，best val gap 只有 `+0.0056`。
- hidden compression 不能同时满足 significant task 和 S2 efficiency；它不是 v7.2 的可行闭合路线。

No-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_hidden32_c1_probe_3seed_20260505T235710Z` | `674` | `0` |

关键 hash：

| file | SHA256 |
|---|---|
| `run.log` | `c94ba7456f587cf151855de5d0c8f808289723f33ffc2356f1cfdf08cf61f35b` |
| `run_manifest.json` | `7a666d3facf7f7cec6e88b2660e4f77767fd10afeacdf43d2d38408da81945b4` |
| `p9_task_summary.csv` | `edea5d924daf7e3a8394e38a40770528b6881723bb4a34d672e8ae955fb15e8b` |
| `p1_significance_audit.csv` | `883df642cd1d76306e1f3ecf99aa41cbf446c030e23f56c68e9680dad74dfecf` |
| `p2_full_gradient_correctness.csv` | `423d8ce147cd6101b8f4fb7645606a4e173771cf85cf83b5a5d4344d873207b6` |
| `p10_efficiency_summary.csv` | `850b7292c4c6aa754c8530b42bb6d1fc5fa48f242f1119d7f92eb2b877688d2a` |
| `route_decision.json` | `1a590a987d73fce8a4c3d17595cbff1b4166e1fcedc6ef7a4c41dd5a0f51008b` |
| `provenance_audit.csv` | `eb5177d19f190af0bd5d360e3f47a7a34ff00cd8d3d78b2cbae613095ccda28a` |

## 26. Hidden compression 后阶段结论

截至 hidden-dim compression probes，v7.2 计划目标仍未达成。

1. 当前最佳 task route 仍是 hidden64 `C3`：strict pass、grad pass、basic task pass 均成立，macro val gap `+0.0214`，CI95 low `+0.0158`，Holm p `2.19e-07`，test gap `+0.0208`。
2. `C3` 仍不是完整 `SignificantTaskPass`：classwise drop max `0.1400 > 0.05`，主要发生在 Fashion-MNIST。
3. `C3` 仍不是 S2/S1：full efficiency memory `1.3069`、step `3.3751`。
4. 失败后已尝试三类 repair：
   - H3 smoothing/lr variants：未修 classwise，部分 task collapse。
   - static class-weighted loss：未修 classwise，反而把 drop 转移到其他 Fashion-MNIST 类。
   - hidden dim 48/32 compression：没有进入 S2，且 task gap明显下降。
5. 当前 blocker 明确为：`classwise_noncollapse_fail + materialization/efficiency_kernelization_fail`。

最终一句话：

> v7.2 仍未达成完整目标。`C3` 的 macro accuracy 已经真实显著为正，但 classwise non-collapse 和 S2 efficiency 都没有闭合；继续小调 smoothing/lr、静态 class weighting、hidden compression 都无效。下一步不应再靠小参数修补，应进入结构性方案：classwise-constrained training objective + materialization-free fused/kernel-native primitive，或重新设计新的 PureKAN function-space primitive。

## 27. 追加自修复：balanced / targeted sampler classwise repair

static class weighting 未修复 classwise drop 后，继续尝试不改变 loss 数值、只改变训练 batch 组成的 sampler repair。所有 sampler repair 均保持 `C3` 初始化一致，用于判断 classwise collapse 是否来自 Fashion-MNIST 少数/难类训练曝光不足。

### 27.1 Balanced sampler

运行：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_balanced_sampler_repair_10seed_20260506T000105Z \
  --fresh \
  --device auto \
  --candidates B0,C3,Q1,Q2 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128 \
  --bench-warmup 3 \
  --bench-reps 15 \
  --grad-batch-sizes 8 \
  --bootstrap-reps 5000
```

候选：

| candidate | sampler policy |
|---|---|
| `C3` | sequential cycle |
| `Q1` | Fashion-MNIST class-balanced batches |
| `Q2` | all-datasets class-balanced batches |

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8449` | `0.7958` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `C3` | `0.8663` | `0.8166` | `+0.0214` | F-MNIST `+0.0090`, KMNIST `+0.0258`, MNIST `+0.0293` | 1 |
| `Q1` | `0.8636` | `0.8122` | `+0.0187` | F-MNIST `+0.0063`, KMNIST `+0.0252`, MNIST `+0.0246` | 1 |
| `Q2` | `0.8624` | `0.8178` | `+0.0175` | F-MNIST `+0.0063`, KMNIST `+0.0213`, MNIST `+0.0250` | 1 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `C3` | `+0.0214` | `+0.0158` | `1.64e-07` | `+0.0208` | `0.1400` | 0 |
| `Q1` | `+0.0187` | `+0.0134` | `1.17e-06` | `+0.0165` | `0.1458` | 0 |
| `Q2` | `+0.0175` | `+0.0120` | `8.34e-06` | `+0.0220` | `0.1458` | 0 |

Efficiency（本 probe 只测 batch size 128，不能替代 full efficiency 结论）：

| candidate | memory ratio | step ratio | survivor |
|---|---:|---:|---|
| `C3` | `1.1177` | `3.2436` | FAIL |
| `Q1` | `1.1177` | `3.2887` | FAIL |
| `Q2` | `1.1177` | `3.2280` | FAIL |

Gradient correctness：

| candidate | grad pass | grad relerr max |
|---|---:|---:|
| `C3` | `3/3` | `6.02e-05` |
| `Q1` | `3/3` | `6.02e-05` |
| `Q2` | `3/3` | `6.02e-05` |

判断：

- class-balanced sampler 没有修复 non-collapse；`Q1/Q2` 的 classwise drop max 反而略高于 `C3`。
- `Q1/Q2` macro val gap 低于 `C3`，不能作为新的 best route。
- sampler 不改变模型结构，因此 efficiency 没有实质改善。

No-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_balanced_sampler_repair_10seed_20260506T000105Z` | `1803` | `0` |

关键 hash：

| file | SHA256 |
|---|---|
| `run.log` | `03559f97356a3658615f76e1ef77d25e552e96b75a60782cf94e8e24d2404b7d` |
| `run_manifest.json` | `89fd50ad327da38335f3ea0dc281ec7e7b260c95b2cce8b3c386129d5f743816` |
| `p9_task_summary.csv` | `3b4dc675223e7c458dae6b837295f289a1f3e2547cee605f79851ba34ed3c84f` |
| `p1_significance_audit.csv` | `b4382ce67e60e97df1cacb4ba829a167e29120366f0df38dbcc041077e1c5316` |
| `p2_full_gradient_correctness.csv` | `3cb83fd4e8feffdbf8183784bfee2b621f9e4727aa5778cc66b4f9b9d4f02a25` |
| `p10_efficiency_summary.csv` | `a3b7d26726a1138b51b1aa4d999de663e7dfe08ba9a7d774b2f56c8e32e5f829` |
| `route_decision.json` | `220ebe2f4ae55ecba00baccda4359d7d700c570393dab750be9947ed77d2f8ff` |
| `provenance_audit.csv` | `962069acd4e99538272bab7dd5342f72521456ece42985ea27f2b835378483c0` |

### 27.2 Targeted sampler

balanced sampler 失败后，继续针对 Fashion-MNIST 的历史坏类做 targeted oversampling。该修复已写入 `experiments/run_gafu_v72_real.py`，并通过：

```bash
python -m py_compile experiments/run_gafu_v72_real.py
```

运行：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_targeted_sampler_repair_10seed_20260506T000549Z \
  --fresh \
  --device auto \
  --candidates B0,C3,Q3,Q4 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128 \
  --bench-warmup 3 \
  --bench-reps 15 \
  --grad-batch-sizes 8 \
  --bootstrap-reps 5000
```

候选：

| candidate | sampler policy |
|---|---|
| `C3` | sequential cycle |
| `Q3` | Fashion-MNIST target class 4 oversampling |
| `Q4` | Fashion-MNIST target class 4/6 oversampling |

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8449` | `0.7958` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `C3` | `0.8663` | `0.8166` | `+0.0214` | F-MNIST `+0.0090`, KMNIST `+0.0258`, MNIST `+0.0293` | 1 |
| `Q3` | `0.8495` | `0.8028` | `+0.0046` | F-MNIST `-0.0346`, KMNIST `+0.0234`, MNIST `+0.0248` | 0 |
| `Q4` | `0.8490` | `0.7962` | `+0.0040` | F-MNIST `-0.0393`, KMNIST `+0.0262`, MNIST `+0.0252` | 0 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `C3` | `+0.0214` | `+0.0158` | `1.64e-07` | `+0.0208` | `0.1400` | 0 |
| `Q3` | `+0.0046` | `-0.0070` | `8.66e-01` | `+0.0070` | `0.3696` | 0 |
| `Q4` | `+0.0040` | `-0.0080` | `8.66e-01` | `+0.0004` | `0.3404` | 0 |

Worst classwise drops：

| candidate | worst drop | dataset | seed | class | MLP class acc | candidate class acc |
|---|---:|---|---:|---:|---:|---:|
| `C3` | `0.1400` | Fashion-MNIST | 5 | 4 | `0.8400` | `0.7000` |
| `Q3` | `0.3696` | Fashion-MNIST | 0 | 6 | `0.6522` | `0.2826` |
| `Q4` | `0.3404` | Fashion-MNIST | 4 | 2 | `0.8511` | `0.5106` |

Efficiency（本 probe 只测 batch size 128，不能替代 full efficiency 结论）：

| candidate | memory ratio | step ratio | survivor |
|---|---:|---:|---|
| `C3` | `1.1177` | `4.9219` | FAIL |
| `Q3` | `1.1177` | `3.1347` | FAIL |
| `Q4` | `1.1177` | `5.7140` | FAIL |

Gradient correctness：

| candidate | grad pass | grad relerr max |
|---|---:|---:|
| `C3` | `3/3` | `6.02e-05` |
| `Q3` | `3/3` | `6.02e-05` |
| `Q4` | `3/3` | `6.02e-05` |

判断：

- Targeted oversampling 不仅没有修复 classwise gate，还导致 Fashion-MNIST macro collapse。
- `Q3/Q4` 将原本 class 4/6 的问题转移并放大到 class 2/3/6；`classwise_drop_max` 达到 `0.3404-0.3696`。
- `Q3/Q4` basic Beyond task gate 也未通过，因此不能进入后续 official 结论。

No-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_targeted_sampler_repair_10seed_20260506T000549Z` | `1807` | `0` |

关键 hash：

| file | SHA256 |
|---|---|
| `run.log` | `2aa262de3c7087bc8221328a02dcd456f9d59e360758e8275a2114191c80d565` |
| `run_manifest.json` | `ed71207c7c06408f8de355dbf1c121b3ea19cdf6868e1cfa4cad5cdc3fdb5f1d` |
| `p9_task_summary.csv` | `f50b31de18265c8e28a8ff461ffaff0ea4729a9e79d782d549395157202da71e` |
| `p1_significance_audit.csv` | `2d93d67bcc9a95cf2cdc649002b990ea42957fa3ba5f9a3f326f877a39b517bf` |
| `p2_full_gradient_correctness.csv` | `bda4aab8f7104d1dc0ee0bbb9c243732c05d11902f1e90382d429786c2469017` |
| `p10_efficiency_summary.csv` | `3069740c8e48dd5f9659d59629b8042566d46c6ce23587ef3aa4818d6576cb56` |
| `route_decision.json` | `aea703267eba5e6543a36554349f354b2fa6e0d74ae6177505b52ef074044aea` |
| `provenance_audit.csv` | `18d0e7e822ec5dd2c6c79b3c6159c776bc85372c99df32e60ec059738cb37b09` |

## 28. Sampler 后阶段结论

截至 sampler repair probes，v7.2 计划目标仍未达成。

1. 当前最佳 candidate 仍是 hidden64 `C3`：strict pass、grad pass、basic task pass 均成立，10-seed macro val gap `+0.0214`，CI95 low `+0.0158`，Holm p `1.64e-07`，test gap `+0.0208`。
2. `C3` 仍不是完整 `SignificantTaskPass`：classwise drop max `0.1400 > 0.05`，主要发生在 Fashion-MNIST。
3. `C3` 仍不是 S2/S1：full efficiency run 中 memory `1.3069`、step `3.3751`；batch-size-128 probes 也全部 `FAIL`。
4. 失败后已继续尝试五类 repair：
   - H3 smoothing/lr variants：未修 classwise，部分 task collapse。
   - static class-weighted loss：未修 classwise，反而把 drop 转移到其他 Fashion-MNIST 类。
   - hidden dim 48/32 compression：没有进入 S2，且 task gap明显下降。
   - class-balanced sampler：未修 classwise，macro gap低于 `C3`。
   - targeted class sampler：Fashion-MNIST 直接 collapse，basic gate 也未通过。
5. 当前 blocker 明确为：`classwise_noncollapse_fail + materialization/efficiency_kernelization_fail`。

最终一句话：

> v7.2 仍未达成完整目标。`C3` 的 macro accuracy 已经真实显著为正，但 classwise non-collapse 和 S2 efficiency 都没有闭合；继续小调 smoothing/lr、class weighting、hidden compression、balanced sampler、targeted sampler 都无效。下一步必须进入更结构性的方案：classwise-constrained objective 或新的 PureKAN primitive，同时实现 materialization-free fused/kernel-native forward-backward。

## 29. 追加自修复：checkpoint selection / classwise-constrained selection

sampler repair 失败后，继续验证一个更直接的问题：`C3` 是否只是最终 step 选错，还是整条训练轨迹都无法同时满足 macro gap 与 classwise non-collapse。因此新增 checkpoint selection repair，在训练中每个 trace 保存 checkpoint，最后按 validation-side score 选择 checkpoint。该路径仍使用真实数据、manual forward/backward/update，并保持 `C3` 初始化一致。

代码修改已通过：

```bash
python -m py_compile experiments/run_gafu_v72_real.py
```

### 29.1 Min-class checkpoint selection

运行：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_checkpoint_selection_repair_10seed_20260506T001120Z \
  --fresh \
  --device auto \
  --candidates B0,C3,R1,R2 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128 \
  --bench-warmup 3 \
  --bench-reps 15 \
  --grad-batch-sizes 8 \
  --bootstrap-reps 5000
```

候选：

| candidate | checkpoint policy |
|---|---|
| `C3` | final step |
| `R1` | `val_acc + 0.05 * min_class_acc` |
| `R2` | `val_acc + 0.10 * min_class_acc` |

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8449` | `0.7958` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `C3` | `0.8663` | `0.8166` | `+0.0214` | F-MNIST `+0.0090`, KMNIST `+0.0258`, MNIST `+0.0293` | 1 |
| `R1` | `0.8687` | `0.8156` | `+0.0238` | F-MNIST `+0.0125`, KMNIST `+0.0289`, MNIST `+0.0299` | 1 |
| `R2` | `0.8679` | `0.8156` | `+0.0230` | F-MNIST `+0.0111`, KMNIST `+0.0281`, MNIST `+0.0297` | 1 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `C3` | `+0.0214` | `+0.0158` | `1.37e-07` | `+0.0208` | `0.1400` | 0 |
| `R1` | `+0.0238` | `+0.0187` | `1.35e-08` | `+0.0199` | `0.1458` | 0 |
| `R2` | `+0.0230` | `+0.0177` | `3.20e-08` | `+0.0199` | `0.1458` | 0 |

Selected checkpoint steps：

| candidate | selected steps | mean selected step |
|---|---|---:|
| `R1` | `80,100,140,160,180,200,220,240` | `177.33` |
| `R2` | `80,120,140,160,180,200,220,240` | `172.00` |

Efficiency（本 probe 只测 batch size 128，不能替代 full efficiency 结论）：

| candidate | memory ratio | step ratio | survivor |
|---|---:|---:|---|
| `C3` | `1.1177` | `3.2387` | FAIL |
| `R1` | `1.1177` | `3.2274` | FAIL |
| `R2` | `1.1177` | `3.2255` | FAIL |

Gradient correctness：

| candidate | grad pass | grad relerr max |
|---|---:|---:|
| `C3` | `3/3` | `6.02e-05` |
| `R1` | `3/3` | `6.02e-05` |
| `R2` | `3/3` | `6.02e-05` |

判断：

- `R1/R2` 能提升 macro val gap，但不能修复 classwise gate。
- `R1/R2` 的 `classwise_drop_max` 为 `0.1458`，比 `C3` 的 `0.1400` 略差。
- min-class 自身不是正确约束，因为它不能对齐 `MLP_class_acc - KAN_class_acc` 的相对退化标准。

No-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_checkpoint_selection_repair_10seed_20260506T001120Z` | `1807` | `0` |

关键 hash：

| file | SHA256 |
|---|---|
| `run.log` | `10831a7940dfa85c681c83c09f10f41dd66d0f5e7c96be80470f9900fee905f0` |
| `run_manifest.json` | `a2ecfac32cc32d3e280e6cf430e0ae2e4f2f6a8f976b5bba24086db9a294232c` |
| `p9_task_summary.csv` | `7084f0bf71105be81f03fb2571aa3f25f30d1be733097c81eaff0d27fa525be1` |
| `p9_task_gate.csv` | `8a978ece31874fe17eeffdfdc30e1001c9df69bec631bc2d05ddf200a8332930` |
| `p1_significance_audit.csv` | `3a64c51632193a2d47b605832f85acddc524cf2d6bca7d3b9c48805665ba31e9` |
| `p2_full_gradient_correctness.csv` | `a7b2c5d96c3909093c80916d1d3b70b240e6c4c64549eea1aa61c5de8231ec12` |
| `p10_efficiency_summary.csv` | `84d250ffd3d340427c9e3caadd822158349955263139da95fb9033debab0d759` |
| `route_decision.json` | `01f0c7214dd2f646e0b6ae8a453339b67fadb1ac403eb7857a8f98f48ae78250` |
| `provenance_audit.csv` | `18d0e7e822ec5dd2c6c79b3c6159c776bc85372c99df32e60ec059738cb37b09` |

### 29.2 Relative-drop checkpoint selection

min-class checkpoint selection 未对齐 classwise gate 后，继续新增 relative-drop selector。该 selector 使用同 dataset / same seed 的 MLP classwise accuracy 作为 baseline，checkpoint score 为：

```text
R3/R4:
  val_acc - beta * max(0, max_class_drop_vs_MLP - 0.05)
```

运行：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_relative_checkpoint_repair_10seed_20260506T001419Z \
  --fresh \
  --device auto \
  --candidates B0,C3,R3,R4 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128 \
  --bench-warmup 3 \
  --bench-reps 15 \
  --grad-batch-sizes 8 \
  --bootstrap-reps 5000
```

候选：

| candidate | checkpoint policy |
|---|---|
| `C3` | final step |
| `R3` | relative class-drop penalty beta `0.10` |
| `R4` | relative class-drop penalty beta `0.25` |

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8449` | `0.7958` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `C3` | `0.8663` | `0.8166` | `+0.0214` | F-MNIST `+0.0090`, KMNIST `+0.0258`, MNIST `+0.0293` | 1 |
| `R3` | `0.8689` | `0.8153` | `+0.0240` | F-MNIST `+0.0129`, KMNIST `+0.0291`, MNIST `+0.0299` | 1 |
| `R4` | `0.8688` | `0.8154` | `+0.0238` | F-MNIST `+0.0125`, KMNIST `+0.0291`, MNIST `+0.0299` | 1 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `C3` | `+0.0214` | `+0.0158` | `1.37e-07` | `+0.0208` | `0.1400` | 0 |
| `R3` | `+0.0240` | `+0.0187` | `1.16e-08` | `+0.0195` | `0.1400` | 0 |
| `R4` | `+0.0238` | `+0.0185` | `9.23e-09` | `+0.0197` | `0.1400` | 0 |

Worst classwise drops：

| candidate | worst drop | dataset | seed | class | MLP class acc | candidate class acc |
|---|---:|---|---:|---:|---:|---:|
| `C3` | `0.1400` | Fashion-MNIST | 5 | 4 | `0.8400` | `0.7000` |
| `R3` | `0.1400` | Fashion-MNIST | 5 | 4 | `0.8400` | `0.7000` |
| `R4` | `0.1400` | Fashion-MNIST | 5 | 4 | `0.8400` | `0.7000` |

Selected checkpoint steps：

| candidate | selected steps | mean selected step |
|---|---|---:|
| `R3` | `80,100,140,160,180,200,220,240` | `177.33` |
| `R4` | `80,100,140,160,180,200,220,240` | `174.00` |

Efficiency（本 probe 只测 batch size 128，不能替代 full efficiency 结论）：

| candidate | memory ratio | step ratio | survivor |
|---|---:|---:|---|
| `C3` | `1.1177` | `1.9902` | FAIL |
| `R3` | `1.1177` | `1.9903` | FAIL |
| `R4` | `1.1177` | `2.0108` | FAIL |

Gradient correctness：

| candidate | grad pass | grad relerr max |
|---|---:|---:|
| `C3` | `3/3` | `6.02e-05` |
| `R3` | `3/3` | `6.02e-05` |
| `R4` | `3/3` | `6.02e-05` |

判断：

- Relative-drop checkpoint selection 把 macro val gap 提高到 `+0.0240`，这是目前最强 macro task signal。
- 但 `R3/R4` 仍没有通过 `SignificantTaskPass`，因为 worst classwise drop 没有低于 `0.05`，仍为 `0.1400`。
- `R3/R4` 的 selected checkpoint 分布说明部分 seed 会选早期 checkpoint，但 Fashion-MNIST seed 5 class 4 的 hard drop 在可选轨迹内没有消除。
- checkpoint selection 不是 v7.2 完整解法；它强化了“需要结构性 classwise objective / primitive repair”的结论。

No-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_relative_checkpoint_repair_10seed_20260506T001419Z` | `1809` | `0` |

关键 hash：

| file | SHA256 |
|---|---|
| `run.log` | `7e5ecd064229e85e3992088f409b63b33d32c00cf53e8447ec7dc1d144940c7b` |
| `run_manifest.json` | `07c40ab155ea7b6593efee1b3fa0758280184fa536059d93323ccc0dc5fee0dc` |
| `p9_task_summary.csv` | `e766ebca4d4ef5131c7414c8f45f353051dd95c2a6f3be352f8c4f3b41e95d31` |
| `p9_task_gate.csv` | `d7f19dfd17f268774c9e84e72d54f806a4faa8b0b8640ad6f7ae6b44d7f0403e` |
| `p1_significance_audit.csv` | `89d200c0c9086df8a1dda18218cde7076f26cb010e9bd244b9834cc70dc64b10` |
| `p2_full_gradient_correctness.csv` | `3dcf14692fe34a4cf0389ce3654d9f2491c7236b8f0e89aa66f7942f6e834814` |
| `p10_efficiency_summary.csv` | `a157c7c10a26fe1d5af9f4b66ea2cb63b46b919db47bfb76594e28405113bc9d` |
| `route_decision.json` | `297ea8811f50533f50e5fa3973c56a9b39dc7d3c703ca224f0369977bdcca46a` |
| `provenance_audit.csv` | `019a5e2efc6f6c45bd6474860e1da0c5770c975c7ad979971a02ddd2875956b6` |

## 30. Checkpoint selection 后阶段结论

截至 checkpoint selection probes，v7.2 计划目标仍未达成。

1. 当前最佳 macro task candidate 变为 `R3`：strict pass、grad pass、basic task pass 均成立，10-seed macro val gap `+0.0240`，CI95 low `+0.0187`，Holm p `1.16e-08`，test gap `+0.0195`。
2. `R3` 仍不是完整 `SignificantTaskPass`：classwise drop max 仍为 `0.1400 > 0.05`，最坏点仍是 Fashion-MNIST seed 5 class 4。
3. `R3` 仍不是 S2/S1：batch-size-128 efficiency memory `1.1177`、step `1.9903`；full efficiency 参考下 `C3` memory `1.3069`、step `3.3751`，路线仍未 kernel-native。
4. 失败后已继续尝试六类 repair：
   - H3 smoothing/lr variants：未修 classwise，部分 task collapse。
   - static class-weighted loss：未修 classwise，反而把 drop 转移到其他 Fashion-MNIST 类。
   - hidden dim 48/32 compression：没有进入 S2，且 task gap明显下降。
   - class-balanced sampler：未修 classwise，macro gap低于 `C3`。
   - targeted class sampler：Fashion-MNIST collapse，basic gate 未通过。
   - checkpoint selection：macro gap 提升，但 classwise hard drop 和 S2 仍失败。
5. 当前 blocker 明确为：`classwise_noncollapse_fail + materialization/efficiency_kernelization_fail`。

最终一句话：

> v7.2 仍未达成完整目标。`R3` 把 macro accuracy 信号进一步推高到 `+0.0240`，但 classwise non-collapse 和 S2 efficiency 两个硬门仍未闭合；已有的小参数、采样、加权、压缩、checkpoint selection 都不能解决。下一步必须进入真正结构性修复：显式 classwise-constrained objective 或新 PureKAN primitive，并实现 materialization-free fused/kernel-native forward-backward。

## 31. 追加自修复：Focal Loss classwise objective repair

checkpoint selection 后，继续实现并测量 focal loss objective repair。新增候选：

| candidate | objective |
|---|---|
| `F1` | `C3` init，focal CE，gamma `1.0` |
| `F2` | `C3` init，focal CE，gamma `2.0` |

实现说明：

- `F1/F2` 使用 `C3` 同初始化，不使用 fake/proxy。
- focal loss 的 forward 与 logits gradient 都是手写实现。
- gradient correctness 用同一 focal loss 的 autograd reference 做对照。
- task/grad/efficiency 均真实落盘。

运行：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_focal_loss_repair_10seed_20260506T002302Z \
  --fresh \
  --device auto \
  --candidates B0,C3,R3,F1,F2 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128 \
  --bench-warmup 3 \
  --bench-reps 15 \
  --grad-batch-sizes 8 \
  --bootstrap-reps 5000
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8449` | `0.7958` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `C3` | `0.8663` | `0.8166` | `+0.0214` | F-MNIST `+0.0090`, KMNIST `+0.0258`, MNIST `+0.0293` | 1 |
| `R3` | `0.8689` | `0.8153` | `+0.0240` | F-MNIST `+0.0129`, KMNIST `+0.0291`, MNIST `+0.0299` | 1 |
| `F1` | `0.8559` | `0.8076` | `+0.0109` | F-MNIST `+0.0078`, KMNIST `+0.0125`, MNIST `+0.0125` | 1 |
| `F2` | `0.8544` | `0.8047` | `+0.0094` | F-MNIST `+0.0055`, KMNIST `+0.0117`, MNIST `+0.0111` | 1 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `C3` | `+0.0214` | `+0.0158` | `2.05e-07` | `+0.0208` | `0.1400` | 0 |
| `R3` | `+0.0240` | `+0.0187` | `1.69e-08` | `+0.0195` | `0.1400` | 0 |
| `F1` | `+0.0109` | `+0.0064` | `8.32e-04` | `+0.0118` | `0.2400` | 0 |
| `F2` | `+0.0094` | `+0.0042` | `1.52e-02` | `+0.0089` | `0.1875` | 0 |

Worst classwise drops：

| candidate | worst drop | dataset | seed | class | MLP class acc | candidate class acc |
|---|---:|---|---:|---:|---:|---:|
| `F1` | `0.2400` | Fashion-MNIST | 5 | 4 | `0.8400` | `0.6000` |
| `F2` | `0.1875` | Fashion-MNIST | 3 | 4 | `0.7708` | `0.5833` |

Efficiency（本 probe 只测 batch size 128）：

| candidate | memory ratio | step ratio | survivor |
|---|---:|---:|---|
| `C3` | `1.1177` | `3.2743` | FAIL |
| `R3` | `1.1177` | `3.2796` | FAIL |
| `F1` | `1.1177` | `3.2913` | FAIL |
| `F2` | `1.1177` | `3.3542` | FAIL |

Gradient correctness：

| candidate | grad pass | grad relerr max | grad cos min |
|---|---:|---:|---:|
| `C3` | `3/3` | `6.02e-05` | `0.999994` |
| `R3` | `3/3` | `6.02e-05` | `0.999994` |
| `F1` | `3/3` | `6.52e-05` | `1.000000` |
| `F2` | `3/3` | `5.43e-05` | `1.000000` |

判断：

- focal loss 没有修复 classwise gate。
- `F1/F2` 真实通过 gradient gate，但 macro task 明显低于 `R3`，classwise drop 反而更差。
- 该结果说明当前 blocker 不是“困难样本整体权重不够”，而是更具体的 classwise/primitive 表达与 decision boundary 问题。

No-fake/no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_focal_loss_repair_10seed_20260506T002302Z` | `2253` | `0` |

关键 hash：

| file | SHA256 |
|---|---|
| `run.log` | `19ac8e94fecd93b586a2f03dcc33a2de1b8ed6a427c9404238bdf7fe5d63d19c` |
| `run_manifest.json` | `f99b8cc4da0ee896522042b3903e17a920763dda79b667bcad465dbd0b7091f4` |
| `p9_task_summary.csv` | `6b1bf5efee921784eabcb2ea1f991e06a2ee1fff95b8325a0c5a3b0e078403ab` |
| `p9_task_gate.csv` | `86d2151ad25c9562b9f1d86af869f56712a2c652d527cb13ac65ecefbc5a4e47` |
| `p1_significance_audit.csv` | `547cfe941440b2de65cc887a018ff14d671637161da2eea6ce5f753bee8df48c` |
| `p2_full_gradient_correctness.csv` | `89fb2bc27ce67fadb9abaf661d3e61f8bcce5be63b2bc97c73170f8481f93376` |
| `p10_efficiency_summary.csv` | `706efc64d7895e0a509ea814dde2101a663229748bcd797c5d44f3e6b0f65d36` |
| `route_decision.json` | `597df7dc2c23c47afc55a150c4fc6f848d57aa3786781f8c8e7a2d7dc5919a82` |
| `provenance_audit.csv` | `cce10fb50365abf05f310f700755790309d99ccb76e94fc53903d539fa5a428a` |

## 32. 追加自修复：Target margin classwise objective repair

focal loss 失败后，继续实现更定向的 true-class margin penalty。新增候选：

| candidate | objective |
|---|---|
| `J1` | Fashion-MNIST class 4，margin `0.25`，lambda `0.25` |
| `J2` | Fashion-MNIST class 4，margin `0.25`，lambda `0.50` |
| `J3` | Fashion-MNIST class 4/6，margin `0.25`，lambda `0.25` |

实现说明：

- margin loss 只作用于 Fashion-MNIST 指定 class。
- 对 active target samples 加入 $\\max(0, m - (z_y - \\max_{j \\ne y} z_j))^2$。
- logits gradient 手写，autograd reference 做 full gradient check。

运行：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_margin_loss_repair_10seed_20260506T002736Z \
  --fresh \
  --device auto \
  --candidates B0,C3,R3,J1,J2,J3 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128 \
  --bench-warmup 3 \
  --bench-reps 15 \
  --grad-batch-sizes 8 \
  --bootstrap-reps 5000
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8449` | `0.7958` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `C3` | `0.8663` | `0.8166` | `+0.0214` | F-MNIST `+0.0090`, KMNIST `+0.0258`, MNIST `+0.0293` | 1 |
| `R3` | `0.8689` | `0.8153` | `+0.0240` | F-MNIST `+0.0129`, KMNIST `+0.0291`, MNIST `+0.0299` | 1 |
| `J1` | `0.8655` | `0.8171` | `+0.0206` | F-MNIST `+0.0125`, KMNIST `+0.0225`, MNIST `+0.0268` | 1 |
| `J2` | `0.8644` | `0.8173` | `+0.0195` | F-MNIST `+0.0092`, KMNIST `+0.0225`, MNIST `+0.0268` | 1 |
| `J3` | `0.8643` | `0.8171` | `+0.0194` | F-MNIST `+0.0090`, KMNIST `+0.0225`, MNIST `+0.0268` | 1 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `C3` | `+0.0214` | `+0.0158` | `2.60e-07` | `+0.0208` | `0.1400` | 0 |
| `R3` | `+0.0240` | `+0.0187` | `2.12e-08` | `+0.0195` | `0.1400` | 0 |
| `J1` | `+0.0206` | `+0.0150` | `1.28e-06` | `+0.0213` | `0.1800` | 0 |
| `J2` | `+0.0195` | `+0.0139` | `3.01e-06` | `+0.0215` | `0.1373` | 0 |
| `J3` | `+0.0194` | `+0.0135` | `8.43e-06` | `+0.0213` | `0.1277` | 0 |

Worst classwise drops：

| candidate | worst drop | dataset | seed | class | MLP class acc | candidate class acc |
|---|---:|---|---:|---:|---:|---:|
| `J1` | `0.1800` | Fashion-MNIST | 5 | 4 | `0.8400` | `0.6600` |
| `J2` | `0.1373` | Fashion-MNIST | 1 | 7 | `0.9804` | `0.8431` |
| `J3` | `0.1277` | Fashion-MNIST | 4 | 2 | `0.8511` | `0.7234` |

Efficiency（本 probe 只测 batch size 128）：

| candidate | memory ratio | step ratio | survivor |
|---|---:|---:|---|
| `C3` | `1.1177` | `2.9102` | FAIL |
| `R3` | `1.1177` | `2.9999` | FAIL |
| `J1` | `1.1177` | `3.9525` | FAIL |
| `J2` | `1.1177` | `2.9001` | FAIL |
| `J3` | `1.1177` | `2.9148` | FAIL |

Gradient correctness：

| candidate | grad pass | grad relerr max | grad cos min |
|---|---:|---:|---:|
| `C3` | `3/3` | `6.02e-05` | `0.999994` |
| `R3` | `3/3` | `6.02e-05` | `0.999994` |
| `J1` | `3/3` | `6.02e-05` | `0.999994` |
| `J2` | `3/3` | `6.02e-05` | `0.999994` |
| `J3` | `3/3` | `6.02e-05` | `0.999994` |

判断：

- Target margin loss 对 classwise drop 有局部改善：`J3` 从 `0.1400` 降到 `0.1277`。
- 但 `J3` 的 macro val gap 下降到 `+0.0194`，没有达到 practical macro gap `>=0.02`。
- `J1/J2/J3` 均未通过 SignificantTaskPass，也未通过 S2。
- 该结果说明“单纯对已知问题类加 margin”仍不足以闭合 v7.2。

No-fake/no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_margin_loss_repair_10seed_20260506T002736Z` | `2698` | `0` |

关键 hash：

| file | SHA256 |
|---|---|
| `run.log` | `1bc11489a9e0a17b51f8e1d1ba61d3e968a99e31f3d0afdf57c2f5172081f3ea` |
| `run_manifest.json` | `3c7725c98c611e85136b332d9f7fb9fbc8bc2fee1b1215b8d674d2223c3150ab` |
| `p9_task_summary.csv` | `9caf32924114f03777ec535d5dc73f49e5e5285c8076e96239f2d1062512d9af` |
| `p9_task_gate.csv` | `a1faaae779a46253e312981d122ab19358bb537c8fd49e5b7f038da582c57ea4` |
| `p1_significance_audit.csv` | `1c3896ba9da6b4de7fc765467633213f422bfce18ea6c1703a033893e3fbd898` |
| `p2_full_gradient_correctness.csv` | `ba08bb0a71b7927699a6b2c8109f2446e1a6120a48ba4412ed964a9980647557` |
| `p10_efficiency_summary.csv` | `fd63ec762032f156ec50b10cbb84b0ea9d270fd2c693cbf5c73ea13eed86a422` |
| `route_decision.json` | `398fc78611e1fd03e945bd9066ffcaf3fbba01d66d5779405f0fd9f341174446` |
| `provenance_audit.csv` | `807415d37c133dc0c604a4c13eca061ebcaf8ce931e54a6fe63ef6434f9fb2bb` |

## 33. 追加自修复：Margin loss + relative-drop checkpoint selection

Target margin loss 只带来局部改善后，继续组合 objective 与 relative-drop checkpoint selection。新增候选：

| candidate | objective | checkpoint policy |
|---|---|---|
| `Y1` | Fashion-MNIST class 4/6 margin `0.25`，lambda `0.25` | relative class-drop penalty beta `0.10` |
| `Y2` | Fashion-MNIST class 4 margin `0.25`，lambda `0.50` | relative class-drop penalty beta `0.25` |

运行：

```bash
python experiments/run_gafu_v72_real.py \
  --out-dir results/real_rerun_20260505/v72_margin_checkpoint_repair_10seed_20260506T003133Z \
  --fresh \
  --device auto \
  --candidates B0,C3,R3,J3,Y1,Y2 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128 \
  --bench-warmup 3 \
  --bench-reps 15 \
  --grad-batch-sizes 8 \
  --bootstrap-reps 5000
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8449` | `0.7958` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `C3` | `0.8663` | `0.8166` | `+0.0214` | F-MNIST `+0.0090`, KMNIST `+0.0258`, MNIST `+0.0293` | 1 |
| `R3` | `0.8689` | `0.8153` | `+0.0240` | F-MNIST `+0.0129`, KMNIST `+0.0291`, MNIST `+0.0299` | 1 |
| `J3` | `0.8643` | `0.8171` | `+0.0194` | F-MNIST `+0.0090`, KMNIST `+0.0225`, MNIST `+0.0268` | 1 |
| `Y1` | `0.8686` | `0.8155` | `+0.0236` | F-MNIST `+0.0119`, KMNIST `+0.0291`, MNIST `+0.0299` | 1 |
| `Y2` | `0.8689` | `0.8161` | `+0.0240` | F-MNIST `+0.0129`, KMNIST `+0.0291`, MNIST `+0.0299` | 1 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | classwise drop max | significant pass |
|---|---:|---:|---:|---:|---:|---:|
| `C3` | `+0.0214` | `+0.0158` | `2.33e-07` | `+0.0208` | `0.1400` | 0 |
| `R3` | `+0.0240` | `+0.0187` | `2.01e-08` | `+0.0195` | `0.1400` | 0 |
| `J3` | `+0.0194` | `+0.0135` | `8.43e-06` | `+0.0213` | `0.1277` | 0 |
| `Y1` | `+0.0236` | `+0.0178` | `1.84e-07` | `+0.0197` | `0.1277` | 0 |
| `Y2` | `+0.0240` | `+0.0185` | `2.01e-08` | `+0.0203` | `0.1373` | 0 |

Worst classwise drops：

| candidate | worst drop | dataset | seed | class | MLP class acc | candidate class acc |
|---|---:|---|---:|---:|---:|---:|
| `R3` | `0.1400` | Fashion-MNIST | 5 | 4 | `0.8400` | `0.7000` |
| `Y1` | `0.1277` | Fashion-MNIST | 4 | 2 | `0.8511` | `0.7234` |
| `Y2` | `0.1373` | Fashion-MNIST | 1 | 7 | `0.9804` | `0.8431` |

Efficiency（本 probe 只测 batch size 128）：

| candidate | memory ratio | step ratio | survivor |
|---|---:|---:|---|
| `C3` | `1.1177` | `2.9275` | FAIL |
| `R3` | `1.1177` | `4.5233` | FAIL |
| `J3` | `1.1177` | `2.8662` | FAIL |
| `Y1` | `1.1177` | `2.9030` | FAIL |
| `Y2` | `1.1177` | `2.9002` | FAIL |

Gradient correctness：

| candidate | grad pass | grad relerr max | grad cos min |
|---|---:|---:|---:|
| `C3` | `3/3` | `6.02e-05` | `0.999994` |
| `R3` | `3/3` | `6.02e-05` | `0.999994` |
| `J3` | `3/3` | `6.02e-05` | `0.999994` |
| `Y1` | `3/3` | `6.02e-05` | `0.999994` |
| `Y2` | `3/3` | `6.02e-05` | `0.999994` |

判断：

- `Y1` 兼顾了较强 macro gap `+0.0236` 与相对更低 classwise drop `0.1277`，但仍远高于 v7.2 non-collapse gate 的 `<=0.05`。
- `Y2` 保住了 `+0.0240` macro gap，但 classwise drop `0.1373`，接近 `R3` 的失败形态。
- margin + checkpoint selection 仍不能达成 SignificantTaskPass，也不能解决 S2。
- 当前最优 route 仍是 `R3-TaskPositiveButNotSignificant`，best candidate 仍为 `R3`。

No-fake/no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v72_margin_checkpoint_repair_10seed_20260506T003133Z` | `2700` | `0` |

关键 hash：

| file | SHA256 |
|---|---|
| `run.log` | `1390af98aebf7325ec775b44eed6e6fa3449898dc298a2b40e96ac93cb4b7fac` |
| `run_manifest.json` | `c898bcee011bfb2f38ff879e3621c05f52ef59499dff238725c47e8d7fe322ff` |
| `p9_task_summary.csv` | `f160a131da6fbedc2ec7f1ffcd8d8b15003ce3b6b4cc06f9a12ab28c9e321385` |
| `p9_task_gate.csv` | `8f7f4448dbfe3a10d7b5d4cc3830df3e1730a4d7595e03b2e4e7512c5f01b053` |
| `p1_significance_audit.csv` | `bffb102156c33494cb0ff8785ced4b6db78f866f17745046da8f3c796ec62190` |
| `p2_full_gradient_correctness.csv` | `faf2162f1611deb914d8b3c6098c5b90a8414eafbd3f1ffac1ce48bd63dceae9` |
| `p10_efficiency_summary.csv` | `1e01c2618dc7d3f03a45fc2cdfc13ad6041d31a9fa93e402e600e61ca4f2204a` |
| `route_decision.json` | `b190ab018fdc8d31f7f21a56d5cc15e2ced40105770d21e446daa9aa8a776178` |
| `provenance_audit.csv` | `6f88750400c0bdf8f4eb8830b8fc89fb771d059750554cd1ab3f4baf4245a75b` |

## 34. 当前最终结论

截至 focal / target-margin / margin-checkpoint 三轮追加自修复，v7.2 计划目标仍未达成。

1. 当前最佳 macro task candidate 仍是 `R3`：strict pass、grad pass、basic task pass 成立，10-seed macro val gap `+0.0240`，CI95 low `+0.0187`，Holm p 约 `2e-08`，test gap `+0.0195`。
2. `R3` 不是完整 `SignificantTaskPass`：classwise drop max `0.1400 > 0.05`。
3. `Y1` 是本轮新增中最有信息量的 classwise repair：macro val gap `+0.0236`，classwise drop max 降到 `0.1277`，但仍失败。
4. `F1/F2` focal loss 真实通过 gradient gate，但 classwise drop 更差，不能作为后续主线。
5. `J1/J2/J3` target-margin loss 只能局部改善；`J3` 降低 worst drop 到 `0.1277`，但 macro gap 降到 `+0.0194`，没有达到 practical macro gate。
6. 所有追加候选 batch-size-128 efficiency 均 FAIL，memory ratio 仍为 `1.1177`，step ratio 大约 `2.86-4.52`，不满足 S2；full efficiency 参考仍应以此前 full probe 的 `C3` memory `1.3069`、step `3.3751` 为主。
7. 本轮新增代码路径均通过 py_compile、小规模 smoke 和 real gradient correctness；追加 runs 的 fake/proxy nonzero 均为 `0`。
8. 当前 blocker 已经收敛为两个硬问题：
   - `classwise_noncollapse_fail`：Fashion-MNIST 多类边界存在局部退化，简单 loss/sampler/checkpoint 不够。
   - `materialization/efficiency_kernelization_fail`：cached dense path 仍不是 S2/S1。

最终一句话：

> v7.2 仍未达成完整目标。`R3` 证明了宏平均 Beyond-MLP 信号真实且显著，但 classwise non-collapse 和 S2 efficiency 仍未闭合；追加 focal、target-margin、margin+checkpoint 都是 measured failure。下一步不应继续微调 loss，而应进入结构性 primitive 修复：更强 classwise-safe PureKAN primitive 与 materialization-free fused/kernel-native forward-backward。
