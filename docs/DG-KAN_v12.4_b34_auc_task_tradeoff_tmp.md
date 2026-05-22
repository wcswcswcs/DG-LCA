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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b34_auc_task_tradeoff_20260520T090000Z
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
    "B34b-GatedLegendreQuadratic-h168-lowboost-temp075-cosine-lr-final050",
    "B34c-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final050"
  ],
  "A2_expression_pass": [
    "B34b-GatedLegendreQuadratic-h168-lowboost-temp075-cosine-lr-final050",
    "B34c-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final050"
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
| `B34a-GatedLegendreQuadratic-h168-lowboost-temp050-cosine-lr-final050` | `GatedHybrid` | `3.9233326578474896` | `1.0187107347719202` | `0` |
| `B34a-GatedLegendreQuadratic-h168-lowboost-temp050-cosine-lr-final050` | `GatedHybrid` | `1.5070041858610057` | `0.38848675225348267` | `0` |
| `B34b-GatedLegendreQuadratic-h168-lowboost-temp075-cosine-lr-final050` | `GatedHybrid` | `4.425806872143437` | `1.0433112537558045` | `0` |
| `B34b-GatedLegendreQuadratic-h168-lowboost-temp075-cosine-lr-final050` | `GatedHybrid` | `1.2436482172982029` | `0.3887940453428025` | `1` |
| `B34c-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final050` | `GatedHybrid` | `2.957911251485961` | `1.0355435673313302` | `0` |
| `B34c-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final050` | `GatedHybrid` | `1.1364238316568809` | `0.3765706091231904` | `1` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B34b-GatedLegendreQuadratic-h168-lowboost-temp075-cosine-lr-final050', 'B34c-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final050']
A2 expression pass = ['B34b-GatedLegendreQuadratic-h168-lowboost-temp075-cosine-lr-final050', 'B34c-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final050']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B34b-GatedLegendreQuadratic-h168-lowboost-temp075-cosine-lr-final050` | `-0.011501736111111112` | `-0.029296875` | `-0.003883904880947537` |
| `B34c-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final050` | `-0.013454861111111112` | `-0.03125` | `-0.009960176216231452` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B34b-GatedLegendreQuadratic-h168-lowboost-temp075-cosine-lr-final050` | `diagnostic_base_not_qualified` | `-0.0022114324683168984` | `0.0` | `-0.0022114324683168984` | `0` |
| `B34c-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final050` | `diagnostic_base_not_qualified` | `-0.005409632789707963` | `0.0` | `-0.005409632789707963` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 562
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `fbb66b7a39fbd5b9c4dec52396ff42882278f9c9f5d663a56b1bb9844213d43e` |
| `v124_control_matrix.csv` | `649f98d3e59de9895c073ffdcba858ebc6277b174679dd11e88e9512f1fbc862` |
| `v124_efficiency_microbench.csv` | `c0278cb2c504c8396caf51b2ac0226c05c49fa2a57ac074dd4b2cee2b4864333` |
| `v124_expression_battery.csv` | `72b00fd06b4656e32ef3ccc5e627ab82130b64a5d1828fd90b888078daaef637` |
| `v124_provenance_audit.csv` | `32eddbffab9ce79a5e8ffd8a4e55df04e786e8dd85a3733cd459f5e1e983524d` |
| `v124_route_decision.json` | `2c75289ef727b0a96656211f6297a602802fdff6d1b8ef092bb25c2de67d8602` |
| `v124_task_triage.csv` | `ca9a959d1661d9221b5be9834af066d7af72aca29a98e0cc524ba17d64d5dc4a` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
