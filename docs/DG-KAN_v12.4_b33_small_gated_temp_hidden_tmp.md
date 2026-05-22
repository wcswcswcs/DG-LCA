# DG-KAN v12.4 Multi-Basis Functional Dual 结果复盘

> 本复盘记录 `DG-KAN_v12.4_多基函数效率优先与Functional双线计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位数据或 CPU offload。Base 未合格时 Functional 只允许 diagnostic。

## 0. 最新结论

```text
route = R3-ExpressionPassTaskFail
base_qualified = False
functional_open = False
functional_diagnostic_positive = False
next_recommended_action = global optimizer/initialization repair; no dataset-specific tuning
```

最终 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b33_small_gated_temp_hidden_20260520T080000Z
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
  "route": "R3-ExpressionPassTaskFail",
  "base_blocker_route": "task_fail",
  "A1_exploratory_survivors": [
    "B33a-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final075",
    "B33b-GatedLegendreQuadratic-h176-lowboost-temp050-cosine-lr-final050",
    "B33c-GatedLegendreQuadratic-h176-lowboost-temp075-cosine-lr-final075"
  ],
  "A2_expression_pass": [
    "B33a-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final075",
    "B33b-GatedLegendreQuadratic-h176-lowboost-temp050-cosine-lr-final050",
    "B33c-GatedLegendreQuadratic-h176-lowboost-temp075-cosine-lr-final075"
  ],
  "A3_task_pass": [],
  "functional_diagnostic_positive": false,
  "base_qualified": false,
  "functional_open": false,
  "next_recommended_action": "global optimizer/initialization repair; no dataset-specific tuning",
  "no_fake": true,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. A1 Efficiency Microbench batch=128

| candidate | family | step ratio q90 vs MLP | memory ratio | exploratory pass |
|---|---|---:|---:|---:|
| `B33a-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final075` | `GatedHybrid` | `3.5315320687813045` | `1.0106357552581262` | `0` |
| `B33a-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final075` | `GatedHybrid` | `1.3965785478084656` | `0.37595602294455066` | `1` |
| `B33b-GatedLegendreQuadratic-h176-lowboost-temp050-cosine-lr-final050` | `GatedHybrid` | `4.490369130625566` | `1.048910816716744` | `0` |
| `B33b-GatedLegendreQuadratic-h176-lowboost-temp050-cosine-lr-final050` | `GatedHybrid` | `1.4382273009434492` | `0.40082969134116364` | `1` |
| `B33c-GatedLegendreQuadratic-h176-lowboost-temp075-cosine-lr-final075` | `GatedHybrid` | `4.183797689057333` | `1.0534348538650642` | `0` |
| `B33c-GatedLegendreQuadratic-h176-lowboost-temp075-cosine-lr-final075` | `GatedHybrid` | `0.8106067708434458` | `0.40113698443048346` | `1` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B33a-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final075', 'B33b-GatedLegendreQuadratic-h176-lowboost-temp050-cosine-lr-final050', 'B33c-GatedLegendreQuadratic-h176-lowboost-temp075-cosine-lr-final075']
A2 expression pass = ['B33a-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final075', 'B33b-GatedLegendreQuadratic-h176-lowboost-temp050-cosine-lr-final050', 'B33c-GatedLegendreQuadratic-h176-lowboost-temp075-cosine-lr-final075']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B33a-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final075` | `-0.004123263888888889` | `-0.025390625` | `-0.006930961377090878` |
| `B33b-GatedLegendreQuadratic-h176-lowboost-temp050-cosine-lr-final050` | `-0.0010850694444444445` | `-0.015625` | `-0.02098911917871899` |
| `B33c-GatedLegendreQuadratic-h176-lowboost-temp075-cosine-lr-final075` | `0.005208333333333333` | `-0.01171875` | `-0.006462441136439641` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B33a-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final075` | `diagnostic_base_not_qualified` | `-0.005409632789707963` | `0.0` | `-0.005409632789707963` | `0` |
| `B33b-GatedLegendreQuadratic-h176-lowboost-temp050-cosine-lr-final050` | `diagnostic_base_not_qualified` | `-0.00029669009977273397` | `0.0026809306095643137` | `-0.0029776207093370477` | `0` |
| `B33c-GatedLegendreQuadratic-h176-lowboost-temp075-cosine-lr-final075` | `diagnostic_base_not_qualified` | `0.0007607209457951569` | `0.0036883751360026196` | `-0.0029276541902074626` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 757
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `b5567cda7bc37f05aa61150780e3913c115f839ec0a40369c662fc4e265bf736` |
| `v124_control_matrix.csv` | `ac4cf9fa14b106a0aa24589dc853cd02e2f6274fec775345b7ab428d095f5c0b` |
| `v124_efficiency_microbench.csv` | `1bdc934d1aedfe61c7293bc2d7f2f1e418a00f5d91386ada203c5cf1116b46b6` |
| `v124_expression_battery.csv` | `20063caa4613d39b2b659a1899e8a4cb7e4d4d7b5c60b717a0520d3825b65c60` |
| `v124_provenance_audit.csv` | `2fb776b5099c581b419d0317c8ee053cc910c670aeb0b9843a5fa46166ef67a6` |
| `v124_route_decision.json` | `cabefb3aa86d9974a599d0a18b294b8b5b05fcdf46972f01c8a3967624288b7d` |
| `v124_task_triage.csv` | `2b30e3d235a71f9a805b13e1d1b9abfea62484c370b1237ee74edb3a338e6ad6` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
