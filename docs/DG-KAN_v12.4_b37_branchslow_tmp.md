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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b37_branchslow_compilewarm_20260520T120000Z
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
    "B37b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-branchslow-final050"
  ],
  "A2_expression_pass": [
    "B37b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-branchslow-final050"
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
| `B37a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-branchslow-final075` | `GatedHybrid` | `6.393630210909597` | `1.0962851679868888` | `0` |
| `B37a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-branchslow-final075` | `GatedHybrid` | `2.0786664485771964` | `0.4423654739142311` | `0` |
| `B37b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-branchslow-final050` | `GatedHybrid` | `6.817794261487961` | `1.1313336520076482` | `0` |
| `B37b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-branchslow-final050` | `GatedHybrid` | `1.4967179505370454` | `0.44267276700355096` | `1` |
| `B37c-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-branchslow-final050` | `GatedHybrid` | `4.339421445420811` | `1.1154056268779022` | `0` |
| `B37c-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-branchslow-final050` | `GatedHybrid` | `2.1086892231613645` | `0.41839661294728214` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B37b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-branchslow-final050']
A2 expression pass = ['B37b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-branchslow-final050']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B37b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-branchslow-final050` | `-0.03624131944444445` | `-0.05859375` | `0.027439401381545596` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B37b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-branchslow-final050` | `diagnostic_base_not_qualified` | `-0.0005323676143538236` | `0.0030359667877211116` | `-0.0035683344020749352` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 367
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `df589541f6f15028eb13ee6548d1c676ff35b6e98c41cf1c0f4de22ebc5ea922` |
| `v124_control_matrix.csv` | `49abb648bf408cfd87a0299c340f26628e1777cb75c9f9cffa57b7c2c2da7d2d` |
| `v124_efficiency_microbench.csv` | `5c826531e7547fdea1edb315a6678c907c0959604abcf6835975bc859ca9b04d` |
| `v124_expression_battery.csv` | `59f41ce83039eaacc7e50a2b788c7ca40ea13fdc7ef5e6699ade5521edfbbbed` |
| `v124_provenance_audit.csv` | `7681d45fadc5a6211167234f3c37d2ae05de0a61645be58295b818fcb10b7b26` |
| `v124_route_decision.json` | `f7373aabda0f05ab687eaa8286ad611a2dce8a910ddb02f6ca0b9d5070e33ffe` |
| `v124_task_triage.csv` | `d794dbc7d06fb7ee8eac6a9e656bad9d7fdba6747e78f64ea27e02ab9514adcb` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
