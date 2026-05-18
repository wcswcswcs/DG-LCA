# DG-KAN v9.2.25 Dataset-Specific Direction/Event Selection 实验复盘

> 本复盘记录本轮针对 v9.2.24 `ActivationRepairedButControlEquivalent` 后续的 dataset-specific direction/event selection 实验。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 delayed local signal 写成 strict functional success。

## 0. 最新结论

截至本轮，v9.2.25 执行到一个可审计 terminal route：

```text
route = R2-DelayedFashionSignalOnly
base_candidate = LQ-t2-h256
success_v9225_local_direction_signal = true
success_v9225_strict_purekan_functional = false
success_v9225_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9225_dataset_specific_direction_event_selection_first_20260510T150000Z/
```

核心结论：

1. v9.2.24 boundary 被复现：N2a event-time activation 已不再 silent，但 global paired replay 仍输给 controls。
2. P1 source autopsy 显示分裂非常明确：Fashion-MNIST 有 delayed h80 CEp99 正信号；KMNIST 即使放大到 cap `0.5` 仍近似 0 effect，且 margin 多为负。
3. P2 重新做 dataset-specific direction/event matrix，覆盖 Fashion O1/O2、MNIST O2、KMNIST O6 low/high cap、KMNIST orthogonal variants。
4. 全 horizon 没有任何 survivor：`all_horizon_pass_count = 0`。
5. delayed h80 发现 2 个 local pass，均来自 Fashion：`F1-Fashion-O1O2-Cap0.5` 与 `F3-Fashion-O2-Cap0.5`。
6. best row 是 `F3-Fashion-O2-Cap0.5` / `h80_only`：CEp99 delta `-0.036434`，beats AdamWParallel `0.666667`，beats best LR `0.666667`，但 margin delta `-0.006796`，不能打开 short/full validation。
7. KMNIST 的核心问题不再是 activation silent，而是方向/target 不可归因：`K3-KMNIST-O6-OrthogonalCap0.5` h80 beat rate 达到 `0.666667`，但 CEp99 delta `+3.18e-07`、margin delta `-1.19e-06`，没有真实 tail/margin 收益。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9225_dataset_specific_direction_event_selection.py` | v9.2.25 runner；执行 source autopsy、dataset-specific direction/event matrix、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9225_dataset_specific_direction_event_selection.py
```

已通过。

正式运行：

```bash
python experiments/run_v9225_dataset_specific_direction_event_selection.py \
  --out-dir results/real_rerun_20260506/v9225_dataset_specific_direction_event_selection_first_20260510T150000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R2-DelayedFashionSignalOnly",
  "base_candidate": "LQ-t2-h256",
  "source_route": "R3-ActivationRepairedButControlEquivalent",
  "all_horizon_pass_count": 0,
  "h80_delayed_pass_count": 2,
  "best_candidate": "F3-Fashion-O2-Cap0.5",
  "best_scope": "h80_only",
  "best_real_beats_adamwparallel_rate": 0.6666666666666666,
  "best_real_beats_best_lr_rate": 0.6666666666666666,
  "best_CEp99_delta": -0.03643353780110677,
  "best_margin_delta": -0.006796201070149739,
  "primary_blocker": "only_delayed_h80_local_signal_no_all_horizon_survivor",
  "next_required_implementation": "design_delayed_event_controller_or_short_run_ablation_for_fashion_signal",
  "success_v9225_local_direction_signal": 2,
  "success_v9225_strict_purekan_functional": 0,
  "success_v9225_external_ready": 0
}
```

判断：

1. 这轮不是 strict functional success，因为没有 all-horizon survivor。
2. 这轮也不是完全失败，因为 delayed Fashion h80 signal 在 strong controls 下留下了局部正证据。
3. 当前 route 的含义是：可以设计 Fashion delayed event controller 或做短程 ablation，但不能把这个 local signal 外推到 KMNIST 或 full-run。

## 3. P0 v9.2.24 boundary reproduction

Artifact：

```text
p0_v9224_boundary_reproduction.csv
```

关键值：

| metric | value |
|---|---:|
| source route | `R3-ActivationRepairedButControlEquivalent` |
| source activation policy | `MNIST=0.5 / Fashion-MNIST=0.5 / KMNIST=0.5` |
| source Real beats AdamWParallel | `0.202381` |
| source Real beats best LR | `0.333333` |
| source Real CEp99 delta | `-0.002547` |
| source Real margin delta | `+0.001134` |
| fake/proxy count | `0` |

判断：v9.2.24 的结论被正确接入本轮：activation repair 已完成，但 paired replay 仍 control-equivalent。

## 4. P1 source signal autopsy

Artifact：

```text
p1_v9224_signal_autopsy.csv
```

Fashion-MNIST delayed signal：

| target | horizon | CEp99 delta | margin delta | beats AdamWParallel | beats best LR |
|---|---:|---:|---:|---:|---:|
| O1 HardTailLogitCorrection | 1 | `-7.15e-06` | `+7.57e-06` | `0.666667` | `0.666667` |
| O1 HardTailLogitCorrection | 80 | `-0.031826` | `+0.026151` | `0.333333` | `0.333333` |
| O2 MarginTailExpansion | 1 | `-2.30e-05` | `-4.41e-06` | `0.666667` | `0.666667` |
| O2 MarginTailExpansion | 80 | `-0.036434` | `-0.006796` | `0.666667` | `0.666667` |

KMNIST amplified but non-causal signal：

| target | horizon | CEp99 delta | margin delta | beats AdamWParallel | beats best LR |
|---|---:|---:|---:|---:|---:|
| O1 HardTailLogitCorrection | 80 | `+4.77e-07` | `-7.95e-07` | `0.333333` | `0.333333` |
| O2 MarginTailExpansion | 80 | `+7.95e-07` | `-1.31e-06` | `0.000000` | `0.000000` |
| O6 KMNISTHardModeOutputTarget | 80 | `+4.77e-07` | `-1.67e-06` | `0.333333` | `0.333333` |

MNIST weak delayed reference：

| target | horizon | CEp99 delta | margin delta | beats AdamWParallel | beats best LR |
|---|---:|---:|---:|---:|---:|
| O2 MarginTailExpansion | 80 | `-2.06e-05` | `+3.28e-05` | `0.333333` | `0.666667` |

判断：

1. Fashion 的 CEp99 信号是真实 delayed tail signal，尤其 O2 h80 的 CEp99 改善最大。
2. Fashion O1 h80 同时改善 CEp99 和 margin，但 beat rate 只有 `0.333333`。
3. Fashion O2 h80 beat rate 更强，但 margin 为负，说明它更像 tail compression，不是完整 margin repair。
4. KMNIST 不是 silent 问题：即使 O6 h80 被激活，CEp99/margin 仍接近 0 或负向。

## 5. P2/P3 dataset-specific matrix

Artifacts：

```text
p2_dataset_specific_direction_event_matrix.csv
p3_direction_event_selection_summary.csv
```

候选范围：

```text
D0 global O1/O2/O6 cap0.5
F1 Fashion O1/O2 cap0.5
F2 Fashion O1 cap0.5
F3 Fashion O2 cap0.5
M1 MNIST O2 cap0.5
K1 KMNIST O6 cap0.1
K2 KMNIST O6 cap0.5
K3 KMNIST O6 orthogonal cap0.5
K4 KMNIST O1/O2/O6 orthogonal cap0.5
```

Summary：

| candidate | scope | dataset | CEp99 delta | margin delta | beats AdamWParallel | beats best LR | local pass |
|---|---|---|---:|---:|---:|---:|---:|
| D0 GlobalCap0.5 | all | all | `-0.002547` | `+0.001134` | `0.202381` | `0.333333` | 0 |
| D0 GlobalCap0.5 | h80 | all | `-0.009754` | `+0.002769` | `0.333333` | `0.428571` | 0 |
| F1 Fashion O1/O2 | all | Fashion | `-0.008912` | `+0.003964` | `0.333333` | `0.333333` | 0 |
| F1 Fashion O1/O2 | h80 | Fashion | `-0.034130` | `+0.009677` | `0.500000` | `0.500000` | 1 |
| F2 Fashion O1 | h80 | Fashion | `-0.031826` | `+0.026151` | `0.333333` | `0.333333` | 0 |
| F3 Fashion O2 | h80 | Fashion | `-0.036434` | `-0.006796` | `0.666667` | `0.666667` | 1 |
| M1 MNIST O2 | all | MNIST | `-4.53e-06` | `+7.97e-06` | `0.083333` | `0.500000` | 0 |
| M1 MNIST O2 | h80 | MNIST | `-2.06e-05` | `+3.28e-05` | `0.333333` | `0.666667` | 0 |
| K1 KMNIST O6 cap0.1 | all | KMNIST | `-1.19e-07` | `-2.98e-07` | `0.333333` | `0.500000` | 0 |
| K2 KMNIST O6 cap0.5 | all | KMNIST | `+2.38e-07` | `-5.86e-07` | `0.250000` | `0.250000` | 0 |
| K3 KMNIST O6 orthogonal | h80 | KMNIST | `+3.18e-07` | `-1.19e-06` | `0.666667` | `0.666667` | 0 |
| K4 KMNIST O1/O2/O6 orthogonal | all | KMNIST | `+4.11e-07` | `-5.00e-07` | `0.166667` | `0.305556` | 0 |

判断：

1. Fashion delayed event 是本轮唯一可靠方向，但它只在 h80 成立。
2. F1 比 F3 更均衡：h80 CEp99 `-0.034130`、margin `+0.009677`、beat rates `0.5/0.5`。
3. F3 是 best CEp99 row，但 margin 变差；它适合做 tail-compression ablation，不适合直接宣称 full margin repair。
4. KMNIST orthogonalization 没解决问题：beat rates 有时变高，但 effect size 近似 0，说明 controls 的排序不等于 functional causality。

## 6. Downstream boundary

这些 artifact 已落盘，但明确为 `not_run`：

| artifact | reason |
|---|---|
| `p4_short_run_direction_event_validation.csv` | `only_delayed_h80_local_signal_no_all_horizon_survivor` |
| `p5_full_functional_direction_event_validation.csv` | same |
| `p6_external_ready.csv` | same |

判断：没有把 delayed h80 local signal 倒灌成 short-run/full-run success。

## 7. No-fake audit

`v9225_provenance_audit.csv`：

```text
rows_checked = 2650
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

说明：

1. P0/P1 只读取 v9.2.24 source artifact，不补造 source 没有的值。
2. P2/P3 是本轮真实 paired replay rows。
3. P4-P6 是明确 gate-blocked `not_run`。

## 8. Hash

| artifact | SHA256 |
|---|---|
| `experiments/run_v9225_dataset_specific_direction_event_selection.py` | `fd18b5f57d53eb2431d95c6c9dd943ca0ae87deaef277299962f67bfc709709d` |
| route | `6da8defc43fdc594686c7b31fbebeb4e0d69fb53d3a0fea1f9b5da47984f609d` |
| P0 recap | `02a6513d7f8d065a87b259e145652ad97be77feff4a4a5d4c855b5b53f1278bb` |
| P1 autopsy | `4413749e98bc30baafe7fb983364919d4c58a94ce830e9762d3baf1c002ad0d1` |
| P2 matrix | `2799f5f95309b90badd2e271edf1a2116e1623c1fabec9df4cc5e5d96eec48f7` |
| P3 summary | `f3b434f98365b239c1280a33208d6e9fedb5d8083a281726431c36ce45f5dbed` |
| failure table | `3235bef4aad6d2564f9cfc42afcc85979772f0cddcc7930cd1277f77427a96bb` |
| provenance audit | `1843410d85c5c2ecd481234cc57e5da496142b6152e95360eba5083702531081` |

## 9. 最终分析结论

v9.2.25 的真实推进是：

```text
v9.2.24: activation scale repair 后，全局 RealFunctional 仍 control-equivalent。
v9.2.25: 分 dataset 重做 direction/event selection，发现 Fashion delayed h80 局部信号；
          KMNIST 放大/orthogonal 后仍没有可归因收益。
```

机制判断：

1. N2a 的 silent 问题已经不是当前主要 blocker；如果只是 silent，KMNIST 在 cap0.5 或 orthogonal cap0.5 下应有明显 CEp99/margin 位移，但实际没有。
2. Fashion-MNIST 的 delayed h80 signal 值得保留：它有真实 CEp99 改善，并且在 F1/F3 下分别体现为 balanced tail+margin 与 stronger tail-only 两种形态。
3. KMNIST 的问题更深：现有 O6 hard-mode target、O1/O2 mixed target、orthogonal direction 都没有产生可观 effect size；下一步不能再简单放大 KMNIST。
4. 当前最合理的延续不是全局 functional patch，而是单独设计 Fashion delayed event controller，并把 KMNIST 返回到 target/primitive 诊断。
5. 因没有 all-horizon survivor，本轮不能打开 short-run、full-run 或 external-ready。

最终一句话：

> v9.2.25 真实执行后停在 `R2-DelayedFashionSignalOnly`：Fashion-MNIST 的 h80 delayed CEp99 信号成立但只局部成立；KMNIST 放大与 orthogonal 后仍不赢 controls。因此 strict PureKAN functional 仍未成功，下一步应围绕 Fashion delayed controller 做短程验证，同时重新诊断 KMNIST target/primitive。
