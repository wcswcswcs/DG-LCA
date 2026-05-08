# DG-KAN v7.0 NextGen BeyondMLP 实验复盘

> 本复盘记录 `docs/DG-KAN_v7.0_NextGen_BeyondMLP_详细实验计划.md` 中 Beyond-MLP 目标的真实执行状态。主结论基于 official seeded full run `results/real_rerun_20260505/v70_beyond_repair_seeded_all_20260505T195718Z`，并追加记录用户要求继续修复后的 targeted real probes。所有数字均来自落盘 CSV/JSON/log/manifest；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. Beyond-MLP 目标是否达成

| 目标层级 | 结论 | 证据 |
|---|---|---|
| v7.0 official runner 是否跑通 | 跑通 | `v70_beyond_repair_seeded_all_20260505T195718Z` 完整落盘并同步 W&B |
| official full run 的 basic Beyond-MLP accuracy gate | 未达成 | best `G2` 平均 val gap `-0.0326`，3 个 primary datasets 都未达到 `MLP - 0.01` |
| 用户继续要求后的 dense targeted basic accuracy gate | 达成 | `D3-dense-poly2-gate-d3-warmup-smooth-240` 在 3 dataset mean 上均不低于 MLP，平均 val gap `+0.0206` |
| strong Beyond gate | 未完整证明 | D3 在 MNIST/KMNIST 超过 `MLP + 0.005` 且 mean ECE 更低，但本 targeted run 未测 ValLossAUC/time AUC |
| v7.0 计划中的完整硬目标 | 未达成 | dense D3 非 kernel-native official candidate，strict zero non-KAN trainable 条件未作为 official gate 证明，且 full-step efficiency probe 9/9 shapes 失败；fastopt 后仍 9/9 失败 |

最终判断：

> **Beyond-MLP task accuracy gate 已经被 dense manual KAN targeted run 打开；但 v7.0 计划里的完整 Beyond-MLP 硬目标还没有达成。** 当前 blocker 已从单纯 task gap 转为 `dense_accuracy_success_but_efficiency/kernelization_fail`。

## 2. Official Full Run 溯源

最终 official seeded full run：

```bash
python experiments/run_gafu_v70_real.py \
  --packages V7_0_ALL \
  --out-dir results/real_rerun_20260505/v70_beyond_repair_seeded_all_20260505T195718Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v70-beyond-repair-seeded-20260505 \
  --wandb-name-prefix v70-beyond-repair-seeded
```

关键定位：

| 项目 | 值 |
|---|---|
| W&B run | <https://wandb.ai/edward20121127/DG-KAN/runs/9b53a51x> |
| route | `R3-S2-TaskImproved` |
| best candidate | `G2-O7-ManualAdamW-lr-high-fulltrain-240` |
| best memory ratio | `1.0311948157` |
| best step ratio | `0.8665947909` |
| best backward ratio | `0.7092462146` |
| best forward ratio | `1.0750472898` |
| best val acc | `0.8138020833` |
| best test acc | `0.7645399306` |
| val gap vs MLP | `-0.0325520833` |
| official task opened | `false` |
| diagnostic task opened | `true` |
| no_fake / no_proxy | `true / true` |

Official best `G2` 对 full-train MLP：

| dataset | MLP val acc | G2 val acc | gap | within 1% MLP | G2 >= MLP |
|---|---:|---:|---:|---:|---:|
| MNIST | `0.8861` | `0.8685` | `-0.0176` | 0 | 0 |
| Fashion-MNIST | `0.8268` | `0.8118` | `-0.0150` | 0 | 0 |
| KMNIST | `0.8262` | `0.7611` | `-0.0651` | 0 | 0 |

Official run 结论：

- `G2` 是 gradient-correct 的 S2 diagnostic 改善，不是 Beyond pass。
- 旧 grouped/T3 路线的主要断点仍是 KMNIST task gap 与 S1 memory margin。
- `X5/r8/hidden-norm` 不能用于成功结论，因为 gradient correctness failure 已真实记录。

## 3. 用户继续要求后的修复尝试

追加 runs 均为 real-data targeted probes，不替代 official full run，但用于分析和修复 blocker。

| run | 用途 | 是否作为 official 结论 | 结果 |
|---|---|---:|---|
| `v70_targeted_probe_kmnist_seed0_20260505T203500Z` | KMNIST optimizer/schedule/lr/weight-decay/basis/smoothing/X6 | 0 | 未闭合 KMNIST gap |
| `v70_targeted_probe_structure_kmnist_seed0_20260505T203620Z` | depth/permutation/hybrid/warmup/480-step | 0 | 未闭合 KMNIST gap |
| `v70_targeted_probe_kan_head_kmnist_seed0_20260505T203700Z` | KAN head expressivity | 0 | 最好 KMNIST gap 仍 `-0.0371` |
| `v70_targeted_probe_inputnorm_kmnist_seed0_20260505T204200Z` | input normalization / centering / standardization | 0 | 最好 val `0.7695`，仍低于 MLP |
| `v70_targeted_probe_densekan_kmnist_seed0_20260505T204300Z` | dense manual KAN upper-bound | 0 | 首次在 KMNIST seed0 超过 MLP val |
| `v70_densekan_taskgate_all_20260505T204500Z` | dense manual KAN across 3 datasets x 3 seeds | 0 | basic accuracy gate 达成 |
| `v70_densekan_efficiency_probe_all_20260505T205300Z` | dense full-step CE efficiency profiler | 0 | dense D1/D3 9/9 shapes FAIL |
| `v70_densekan_fastopt_efficiency_probe_all_20260505T205700Z` | foreach fast optimizer overhead repair probe | 0 | update overhead 大降，但 D1/D3 仍 9/9 shapes FAIL |

过程中遇到的主要问题与处理：

| 问题 | 处理 | 结果 |
|---|---|---|
| grouped T3/O7 fulltrain 在 KMNIST 明显欠拟合或泛化不足 | 扫 optimizer、lr、warmup、label smoothing、depth、head、input normalization | 有小幅改善，但不能达到 `MLP - 0.01` |
| output head 表达力可能不足 | 测 `linear/poly2/poly3/rbf` head | ECE/NLL 有改善，accuracy gap 没闭合 |
| grouped结构可能表达力不足 | 测 dense manual KAN stack | task accuracy gate 打开 |
| dense task 可能只是“慢而强” | 增加 full-step CE profiler | 原始 dense D3 step ratio mean `2.7489`，memory ratio mean `1.1982` |
| optimizer update 诊断开销可能拖慢 dense | 用 foreach fast AdamW probe 去掉 CPU norm 诊断路径 | D3 step ratio mean 降到 `1.6810`，但仍不进 S2；memory ratio mean `1.1778` |

## 4. Dense Task Gate 结果

运行目录：

```text
results/real_rerun_20260505/v70_densekan_taskgate_all_20260505T204500Z
```

覆盖：

```text
3 datasets x 3 seeds x 5 candidates = 45 task rows
trace rows = 72
train/val/test size = 1536/512/512
steps = 240
train_mode = fulltrain_cycle
```

Summary：

| candidate | val acc mean | test acc mean | val gap vs MLP | datasets within 1% | datasets >= MLP | basic gate |
|---|---:|---:|---:|---:|---:|---:|
| `D0-MLP-autograd-fulltrain-240` | `0.8464` | `0.7988` | `0.0000` | 3 | 3 | reference |
| `D1-dense-poly2-silu-d2-adamw-240` | `0.8611` | `0.8132` | `+0.0148` | 3 | 2 | 1 |
| `D2-dense-poly2-silu-d3-warmup-smooth-240` | `0.8644` | `0.8218` | `+0.0180` | 3 | 2 | 1 |
| `D3-dense-poly2-gate-d3-warmup-smooth-240` | `0.8670` | `0.8218` | `+0.0206` | 3 | 3 | 1 |
| `D4-dense-poly3-d2-warmup-smooth-240` | `0.8615` | `0.8177` | `+0.0152` | 3 | 2 | 1 |

最佳 dense task candidate 是 `D3`：

| dataset | MLP val acc | D3 val acc | gap | D3 test acc |
|---|---:|---:|---:|---:|
| MNIST | `0.8861` | `0.9167` | `+0.0306` | `0.9225` |
| Fashion-MNIST | `0.8268` | `0.8281` | `+0.0013` | `0.8125` |
| KMNIST | `0.8262` | `0.8561` | `+0.0299` | `0.7305` |

D3 calibration / NLL：

| candidate | val ECE mean | val NLL mean | test ECE mean | test NLL mean |
|---|---:|---:|---:|---:|
| `D0-MLP-autograd-fulltrain-240` | `0.0603` | `0.5516` | `0.0906` | `0.7986` |
| `D3-dense-poly2-gate-d3-warmup-smooth-240` | `0.0537` | `0.4675` | `0.0481` | `0.5968` |

判断：

- `D3` 真实满足 basic Beyond-MLP task gate：all primary datasets within 1% MLP，且 3/3 dataset mean `KAN >= MLP`。
- `D3` 在 MNIST/KMNIST 两个 dataset 上超过 `MLP + 0.005`，mean ECE/NLL 也优于 MLP。
- 但 targeted dense run 没有测 ValLossAUC/time AUC，也不是 official kernel-native route；其 linear readout/head 也没有作为 strict zero-non-KAN hard gate 重新证明，因此不能宣称 strong/full Beyond goal 达成。

## 5. Dense Efficiency Probe

运行目录：

```text
results/real_rerun_20260505/v70_densekan_efficiency_probe_all_20260505T205300Z
```

覆盖：

```text
3 datasets x 3 batch sizes x 3 candidates = 27 detail rows
real train batch CE full-step
warmup/reps = 10/60
batch sizes = 128,256,512
```

Summary：

| candidate | step ratio mean | memory ratio mean | forward ratio mean | backward ratio mean | update ms mean | survivor |
|---|---:|---:|---:|---:|---:|---|
| `MLP-autograd-reference-depth2` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.1932` | REF |
| `D1-dense-poly2-silu-d2-adamw-240` | `2.6156` | `1.3101` | `2.3765` | `1.7818` | `0.9056` | FAIL 9/9 |
| `D3-dense-poly2-gate-d3-warmup-smooth-240` | `2.7489` | `1.1982` | `2.5314` | `1.9199` | `0.9273` | FAIL 9/9 |

判断：

- dense task improvement 不是免费的；原始 dense manual optimizer 路径 step 明显慢于 MLP。
- D3 memory ratio mean `1.1982`，超过 S2 diagnostic 的 `<=1.05`，更不用说 S1 `<1.00`。
- D3 step ratio mean `2.7489`，超过 S2 diagnostic 的 `<=1.50`。

## 6. FastOpt 修复尝试

运行目录：

```text
results/real_rerun_20260505/v70_densekan_fastopt_efficiency_probe_all_20260505T205700Z
```

修复内容：

```text
用 torch._foreach_* 实现测量脚本内的 dense AdamW fast update；
去掉 ManualOptimizer step 中 CPU norm / role share 诊断路径；
模型 forward/backward 数学不变。
```

Summary：

| candidate | step ratio mean | memory ratio mean | forward ratio mean | backward ratio mean | update ms mean | survivor |
|---|---:|---:|---:|---:|---:|---|
| `MLP-autograd-reference-depth2` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.1889` | REF |
| `D1-fastopt-dense-poly2-silu-d2-adamw-240` | `1.5948` | `1.2905` | `2.3045` | `1.8721` | `0.1148` | FAIL 9/9 |
| `D3-fastopt-dense-poly2-gate-d3-warmup-smooth-240` | `1.6810` | `1.1778` | `2.4131` | `1.9863` | `0.1171` | FAIL 9/9 |

判断：

- FastOpt 真实解决了大部分 update overhead：D3 update mean 从 `0.9273 ms` 降到 `0.1171 ms`。
- 但 D3 仍然没有进入 S2：step ratio mean `1.6810 > 1.50`，memory ratio mean `1.1778 > 1.05`。
- 这说明剩余 blocker 不是 optimizer update，而是 dense forward/backward live set 与 dense elementwise/kernelization。

## 7. No-Fake / No-Proxy 审计

Official full run 审计：

```text
results/real_rerun_20260505/v70_beyond_repair_seeded_all_20260505T195718Z/*.csv
fake_data_used / uses_fake_data / proxy_row_used / proxy_rows_used nonzero count = 0
route_decision.json: no_fake=true, no_proxy=true
```

追加 targeted artifacts：

| artifact | rows | SHA256 | fake/proxy nonzero |
|---|---:|---|---:|
| `v70_targeted_probe_kmnist_seed0_20260505T203500Z/targeted_probe_summary.csv` | 54 | `af8c57efa898c027280c76dae3f080ed6382ccdaa52a5aa0fb18ba73f055dffa` | 0 |
| `v70_targeted_probe_structure_kmnist_seed0_20260505T203620Z/targeted_probe_summary.csv` | 29 | `b589b8bb1427bf3f944236c214148bb44ddabd4752192c988ebf1afd091f4622` | 0 |
| `v70_targeted_probe_kan_head_kmnist_seed0_20260505T203700Z/targeted_probe_summary.csv` | 28 | `c833b6482131950517d060a1803aa3b7f56a7577e3ffe0a84602e0f006887e9f` | 0 |
| `v70_densekan_taskgate_all_20260505T204500Z/densekan_taskgate_summary.csv` | 5 | `e0187ba74270332aacc807917da286c7110689c135b75e46bc55c8de13d7d766` | 0 |
| `v70_densekan_taskgate_all_20260505T204500Z/densekan_taskgate_task.csv` | 45 | `951eb28c0596146bd3db0b2a884208a394105d94c4e765e9ee69c2eb1ca94207` | 0 |
| `v70_densekan_taskgate_all_20260505T204500Z/densekan_taskgate_trace.csv` | 72 | `f80566fe0ecab7c9357ec33db52444c977412556f6557f36ed8535b8d95a25cc` | 0 |
| `v70_densekan_efficiency_probe_all_20260505T205300Z/densekan_efficiency_summary.csv` | 3 | `21dfb7b1a6a419efdbd3550e16cb9d73b3f4c8ea6ab86f3226dabb817d1fba67` | 0 |
| `v70_densekan_efficiency_probe_all_20260505T205300Z/densekan_efficiency_detail.csv` | 27 | `1805724ef4e2bb3e3213790ce0ae4f3d138dc09f56feecb3fc1d320e350d2484` | 0 |
| `v70_densekan_fastopt_efficiency_probe_all_20260505T205700Z/densekan_fastopt_efficiency_summary.csv` | 3 | `4ab802bf1a2a668eb835eed277dea94b9285d269fec6bee0268a4fc637398f72` | 0 |
| `v70_densekan_fastopt_efficiency_probe_all_20260505T205700Z/densekan_fastopt_efficiency_detail.csv` | 27 | `4355ec40fbf1882fda763f920e305231caedec63deb29f987be8cb967c92e720` | 0 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v70_real.py
```

已通过。

## 8. 结论

本轮 v7.0 支持以下真实结论：

1. Official seeded full run 已跑通，但 grouped/T3/O7 路线未达成 Beyond-MLP basic task gate。
2. 用户要求继续修后，dense manual KAN targeted run 首次真实打开 basic Beyond-MLP task gate。
3. 最佳 dense task candidate 是 `D3-dense-poly2-gate-d3-warmup-smooth-240`：val acc mean `0.8670`，test acc mean `0.8218`，val gap vs MLP `+0.0206`，3/3 datasets `KAN >= MLP`。
4. Dense D3 的 calibration/NLL 也优于 MLP：val ECE `0.0537 < 0.0603`，val NLL `0.4675 < 0.5516`。
5. 但 dense D3 没有达成完整 v7.0 硬目标：它不是 official kernel-native S1/S0 candidate，strict zero non-KAN trainable 条件未作为 official gate 证明，且原始 efficiency probe 9/9 shapes FAIL。
6. FastOpt 修复真实降低 update overhead，但仍没有让 D3 进入 S2：step ratio mean `1.6810`，memory ratio mean `1.1778`。
7. 当前最有价值的新方向不是继续调 optimizer，而是把 dense `poly2_gate` 的表达力迁移回 kernel-native/grouped/fused 实现，并同步做 live-set memory repair。

最终一句话：

> v7.0 的 **accuracy Beyond-MLP** 已经在 dense manual KAN targeted run 中达成；但计划定义的 **完整 PureKAN-NG Beyond-MLP 硬目标** 尚未达成。下一步 blocker 是 kernelization 和 efficiency：需要把 dense `poly2_gate` 的表达力做成 graph-free、kernel-native、低内存的 S2/S1 candidate。
