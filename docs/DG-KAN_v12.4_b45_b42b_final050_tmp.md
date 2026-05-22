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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b45_b42b_final050_20260520T210000Z
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
    "B45a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final050"
  ],
  "A2_expression_pass": [
    "B45a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final050"
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
| `B45a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final050` | `GatedHybrid` | `3.3857557024052967` | `1.0239859328052445` | `0` |
| `B45a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final050` | `GatedHybrid` | `1.178873745280927` | `0.4303127560775744` | `1` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B45a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final050']
A2 expression pass = ['B45a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final050']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B45a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final050` | `-0.009331597222222222` | `-0.0234375` | `-0.0027160458266735077` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B45a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final050` | `diagnostic_base_not_qualified` | `4.012296239519486e-05` | `0.0004065333827725226` | `-0.00036641042037732774` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 355
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `4586cedc46551c25e67fd3d202cc3e74b100e84c65ab446d63181cb2bf7023e1` |
| `v124_control_matrix.csv` | `a38b9010410aa0fe7e5a3351d2bb19c4faf8f88a57566106b089d4c22da745b6` |
| `v124_efficiency_microbench.csv` | `84b6384852c98473709d1f9550a5bf0c02c0e7ef7e0dbdc3431ebfdae2862f3c` |
| `v124_expression_battery.csv` | `1f7c4baeef340b0771380c0a28cbe29e365aa0480de7e0795a5539640415f234` |
| `v124_provenance_audit.csv` | `f3d91166ebeb3e44a1e4300f5cea824b5fb8d7150aba02817e60a4532288f1e9` |
| `v124_route_decision.json` | `b489c8aac4e18654eb9ec5f4996b6d34b5815ab4ed1ca70029bbec82cf742d5e` |
| `v124_task_triage.csv` | `3d7709a864b1f09b67dbeb6e4d376e4dd8845bd8d7dec7cc97ce497ca241f62e` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
