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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b43_h172_fastreuse_compilewarm_20260520T190000Z
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
    "B43b-GatedLegendreQuadratic-h172-lowboost-temp065-directskip-fastreuse-final050"
  ],
  "A2_expression_pass": [
    "B43b-GatedLegendreQuadratic-h172-lowboost-temp065-directskip-fastreuse-final050"
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
| `B43a-GatedLegendreQuadratic-h172-lowboost-temp075-directskip-fastreuse-final075` | `GatedHybrid` | `6.180559840845385` | `1.027775880906856` | `0` |
| `B43a-GatedLegendreQuadratic-h172-lowboost-temp075-directskip-fastreuse-final075` | `GatedHybrid` | `1.7698200669445636` | `0.4364244741873805` | `0` |
| `B43b-GatedLegendreQuadratic-h172-lowboost-temp065-directskip-fastreuse-final050` | `GatedHybrid` | `3.8750971939672625` | `1.0618000546298825` | `0` |
| `B43b-GatedLegendreQuadratic-h172-lowboost-temp065-directskip-fastreuse-final050` | `GatedHybrid` | `1.0976061936748942` | `0.4367317672767004` | `1` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B43b-GatedLegendreQuadratic-h172-lowboost-temp065-directskip-fastreuse-final050']
A2 expression pass = ['B43b-GatedLegendreQuadratic-h172-lowboost-temp065-directskip-fastreuse-final050']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B43b-GatedLegendreQuadratic-h172-lowboost-temp065-directskip-fastreuse-final050` | `-0.011935763888888888` | `-0.037109375` | `-0.011586784074703852` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B43b-GatedLegendreQuadratic-h172-lowboost-temp065-directskip-fastreuse-final050` | `diagnostic_base_not_qualified` | `1.318335971234319e-05` | `0.002771727584360395` | `-0.002758544224648052` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 361
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `e4773b174b57c367cee7fd3bce09d25899b467cffe793b61162b1a1a7a3d66c4` |
| `v124_control_matrix.csv` | `129032c0ba0191174a63f53b1bf2b676b5193142ab60d3d12448e2803dcfe82b` |
| `v124_efficiency_microbench.csv` | `b80d853ed3f046627bf573988ae88d7a04f369592fa78b0251ababaedc911de4` |
| `v124_expression_battery.csv` | `8a4d48425711665966413f38d9520ce46616f47e8c1002c480ebaac4f816d0d9` |
| `v124_provenance_audit.csv` | `5a568c697f74b283c63899d596fcab99728d5d0fd9e4047ca74b5ab5e1298dc3` |
| `v124_route_decision.json` | `77ac248d5241745bf5178a3806ced47c2d10b3b406aad78869a8ff8b9c09b7a4` |
| `v124_task_triage.csv` | `987f6e49b9505041a7b12b39be56ad05e5c28ab25538d4611f3445aa9330fecc` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
