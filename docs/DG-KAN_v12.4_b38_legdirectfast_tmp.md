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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b38_legdirectfast_compilewarm_20260520T130000Z
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
| `B38a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-legfast-final075` | `GatedHybrid` | `6.790452402501039` | `1.0962851679868888` | `0` |
| `B38a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-legfast-final075` | `GatedHybrid` | `2.3515869810362546` | `0.4423654739142311` | `0` |
| `B38b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-legfast-final050` | `GatedHybrid` | `6.1967192535653774` | `1.1313336520076482` | `0` |
| `B38b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-legfast-final050` | `GatedHybrid` | `2.422857574375153` | `0.44267276700355096` | `0` |
| `B38c-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-legfast-final050` | `GatedHybrid` | `6.970424344023157` | `1.1154056268779022` | `0` |
| `B38c-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-legfast-final050` | `GatedHybrid` | `2.1287205147748334` | `0.41839661294728214` | `0` |

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
rows_checked = 67
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `33a9d9b7e4613841a9ea7679e6e4e495bc1d23e59681e9319ece91c034ba84fd` |
| `v124_control_matrix.csv` | `006a7aededa28db873fe47ee3f27dd65cfe1e883f7c9029e997e1a607b4eaad6` |
| `v124_efficiency_microbench.csv` | `406e8faba06b9b566cf3acecd95659938d15d6893eeb48a7141c07e221037bad` |
| `v124_expression_battery.csv` | `5b9da370867a7e4c02136e2f17c0de7035865ebbda4a737e98f704c3957acb56` |
| `v124_provenance_audit.csv` | `f660b0802881078841568cb1a20dcebb0e9ded56c5b1d4ed5b04650ef64aed19` |
| `v124_route_decision.json` | `4798272c073de65d9a0ee5d0a9e61634099acbd1a26916774e5b659ca2635858` |
| `v124_task_triage.csv` | `399cafc145d1e2ff5272744855d28d2e2eeb61cb6702a205f92dea74a61714c1` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
