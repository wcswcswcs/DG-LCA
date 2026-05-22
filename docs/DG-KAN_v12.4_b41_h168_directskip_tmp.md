# DG-KAN v12.4 Multi-Basis Functional Dual 结果复盘

> 本复盘记录 `DG-KAN_v12.4_多基函数效率优先与Functional双线计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位数据或 CPU offload。Base 未合格时 Functional 只允许 diagnostic。

## 0. 最新结论

```text
route = R1-EfficiencyAllFail
base_qualified = False
functional_open = False
functional_diagnostic_positive = False
next_recommended_action = implement fused/manual kernel for top two families only
```

最终 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b41_h168_directskip_compilewarm_20260520T170000Z
```

## 1. 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增统一 strict edge-basis KAN primitive：Activation/RBF/OrthogonalPolynomial/Fourier/Wavelet/BSpline/Rational | 所有 learnable tensor 都是 edge basis 权重；fixed centers/normalizer 不按 dataset name 或 label 分支。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 新增 v12.4 A0/A1/A2/A3/B1 runner | 按 efficiency-first 执行；functional 在 base 未过时标记 `diagnostic_base_not_qualified`。 |
| `third_party/MJKAN` | 作为 RBF/FastKAN basis 设计来源读取 | 没有直接运行其 Colab 脚本，也没有引入 base update MLP shortcut。 |
| `third_party/KANbeFair` | 作为 local B-spline basis 设计来源读取 | 本轮只实现 order1 local basis 进行 microbench，不使用其 shortcut/base_fun。 |
| `third_party/rational_kat_cu` | 作为 Rational/KAT safe denominator 设计来源读取 | 本轮使用 torch-safe rational lite diagnostic，未依赖 CUDA extension 编译。 |

本轮没有做：

```text
1. 没有调低 base 或 functional gate。
2. 没有按 MNIST/Fashion/KMNIST 名称分支。
3. 没有 teacher/distillation/loss modification/class weights。
4. 没有把 autograd-only diagnostic 写成 final manual-kernel claim。
```

## 2. Route

```json
{
  "stage": "V124_ROUTE_DECISION",
  "route": "R1-EfficiencyAllFail",
  "base_blocker_route": "efficiency_all_fail",
  "A1_exploratory_survivors": [],
  "A2_expression_pass": [],
  "A3_task_pass": [],
  "functional_diagnostic_positive": false,
  "base_qualified": false,
  "functional_open": false,
  "next_recommended_action": "implement fused/manual kernel for top two families only",
  "no_fake": true,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. A1 Efficiency Microbench batch=128

| candidate | family | step ratio q90 vs MLP | memory ratio | exploratory pass |
|---|---|---:|---:|---:|
| `B41a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-cosine-lr-final075` | `GatedHybrid` | `4.728449549786577` | `1.0883296913411635` | `0` |
| `B41a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-cosine-lr-final075` | `GatedHybrid` | `2.004490840320976` | `0.4303127560775744` | `0` |
| `B41b-GatedLegendreQuadratic-h168-lowboost-temp065-directskip-cosine-lr-final050` | `GatedHybrid` | `4.803313537707502` | `1.121295411089866` | `0` |
| `B41b-GatedLegendreQuadratic-h168-lowboost-temp065-directskip-cosine-lr-final050` | `GatedHybrid` | `1.989416174851358` | `0.4306200491668943` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = []
A2 expression pass = []
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `not_run` |  |  |  |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `None` | `None` | `None` | `None` | `None` | `None` |

## 6. No-Fake / Hash

```text
rows_checked = 61
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `d95466f52a01a77d542c8b11712b777392025625938ae9c96ba785083c7dee99` |
| `v124_control_matrix.csv` | `006a7aededa28db873fe47ee3f27dd65cfe1e883f7c9029e997e1a607b4eaad6` |
| `v124_efficiency_microbench.csv` | `863f41a7dc43e207881d77c86d76ab6ecbc1f3dd616a4fdc2a472638924be8bb` |
| `v124_expression_battery.csv` | `71c0889a5a87571d2ea2990cf54fdb85f3e8768a687880c158cffdba4d90f1c2` |
| `v124_provenance_audit.csv` | `395e6b2feee8682274c199ba6bd2dbdab56fab561f0f83b013be24a8430fdaaf` |
| `v124_route_decision.json` | `4798272c073de65d9a0ee5d0a9e61634099acbd1a26916774e5b659ca2635858` |
| `v124_task_triage.csv` | `399cafc145d1e2ff5272744855d28d2e2eeb61cb6702a205f92dea74a61714c1` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
