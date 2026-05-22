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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b42_fastreuse_compilewarm_20260520T180000Z
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
    "B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075"
  ],
  "A2_expression_pass": [
    "B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075"
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
| `B42a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-fastreuse-final075` | `GatedHybrid` | `4.808609177169781` | `1.0314633979786942` | `0` |
| `B42a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-fastreuse-final075` | `GatedHybrid` | `1.566701348269234` | `0.4423654739142311` | `0` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `GatedHybrid` | `4.544915279133693` | `1.0590344168260037` | `0` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `GatedHybrid` | `1.337654524030241` | `0.4306200491668943` | `1` |
| `B42c-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-fastreuse-final050` | `GatedHybrid` | `4.378752817420296` | `1.0647364108167168` | `0` |
| `B42c-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-fastreuse-final050` | `GatedHybrid` | `1.71016329447941` | `0.4429800600928708` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075']
A2 expression pass = ['B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `-0.009331597222222222` | `-0.017578125` | `0.0016034146149953206` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `diagnostic_base_not_qualified` | `4.012296239519486e-05` | `0.0004065333827725226` | `-0.00036641042037732774` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 367
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `f8ce17de67de647546ec888a857746db448ad104997047425f89693d69b56215` |
| `v124_control_matrix.csv` | `d43dd8cbe419be28a5655f390d2115a76764615dbb2df3ef569ecd4e9cf0b55a` |
| `v124_efficiency_microbench.csv` | `a6c8d16c8130f34ac1babe7fc094ea705b45b0bc96bbb83c28f8c08d8667497e` |
| `v124_expression_battery.csv` | `6772c53484a408b1f63ccfcccff33ceb239c953c22c150e06e39d42f29076b91` |
| `v124_provenance_audit.csv` | `7681d45fadc5a6211167234f3c37d2ae05de0a61645be58295b818fcb10b7b26` |
| `v124_route_decision.json` | `8ffcdffccce8598ea64f15ccb095d77692b9a27afae7c18c87ae3d0d4a394b12` |
| `v124_task_triage.csv` | `8f6df927f0e8bddf457314025f1a50dcbd668019db35019e74ce817601383c24` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
