# DG-KAN v12.4 Multi-Basis Functional Dual 结果复盘

> 本复盘记录 `DG-KAN_v12.4_多基函数效率优先与Functional双线计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位数据或 CPU offload。Base 未合格时 Functional 只允许 diagnostic。

## 0. 最新结论

```text
route = R5-FunctionalDiagnosticPositiveBaseNotYetQualified
base_qualified = False
functional_open = False
functional_diagnostic_positive = True
next_recommended_action = continue base repair; do not write functional success
```

最终 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b32_small_gated_inputnorm_20260520T070000Z
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
  "route": "R5-FunctionalDiagnosticPositiveBaseNotYetQualified",
  "base_blocker_route": "task_fail",
  "A1_exploratory_survivors": [
    "B32a-GatedLegendreQuadratic-h160-lowboost-temp050-cosine-lr-final050",
    "B32c-GatedLegendreQuadratic-h192-lowboost-temp075-cosine-lr-final075"
  ],
  "A2_expression_pass": [
    "B32a-GatedLegendreQuadratic-h160-lowboost-temp050-cosine-lr-final050",
    "B32c-GatedLegendreQuadratic-h192-lowboost-temp075-cosine-lr-final075"
  ],
  "A3_task_pass": [],
  "functional_diagnostic_positive": true,
  "base_qualified": false,
  "functional_open": false,
  "next_recommended_action": "continue base repair; do not write functional success",
  "no_fake": true,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. A1 Efficiency Microbench batch=128

| candidate | family | step ratio q90 vs MLP | memory ratio | exploratory pass |
|---|---|---:|---:|---:|
| `B32a-GatedLegendreQuadratic-h160-lowboost-temp050-cosine-lr-final050` | `GatedHybrid` | `4.7533844868315525` | `1.0106357552581262` | `0` |
| `B32a-GatedLegendreQuadratic-h160-lowboost-temp050-cosine-lr-final050` | `GatedHybrid` | `1.3938805719373952` | `0.37595602294455066` | `1` |
| `B32b-GatedLegendreQuadratic-h192-lowboost-temp050-cosine-lr-final050` | `GatedHybrid` | `4.4413725695249555` | `1.0647364108167168` | `0` |
| `B32b-GatedLegendreQuadratic-h192-lowboost-temp050-cosine-lr-final050` | `GatedHybrid` | `2.001815264966266` | `0.4254131384867523` | `0` |
| `B32c-GatedLegendreQuadratic-h192-lowboost-temp075-cosine-lr-final075` | `GatedHybrid` | `5.1586388091370665` | `1.0734942638623326` | `0` |
| `B32c-GatedLegendreQuadratic-h192-lowboost-temp075-cosine-lr-final075` | `GatedHybrid` | `1.358197619935907` | `0.42573750341436767` | `1` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B32a-GatedLegendreQuadratic-h160-lowboost-temp050-cosine-lr-final050', 'B32c-GatedLegendreQuadratic-h192-lowboost-temp075-cosine-lr-final075']
A2 expression pass = ['B32a-GatedLegendreQuadratic-h160-lowboost-temp050-cosine-lr-final050', 'B32c-GatedLegendreQuadratic-h192-lowboost-temp075-cosine-lr-final075']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B32a-GatedLegendreQuadratic-h160-lowboost-temp050-cosine-lr-final050` | `-0.016927083333333332` | `-0.037109375` | `-0.014770552722944153` |
| `B32c-GatedLegendreQuadratic-h192-lowboost-temp075-cosine-lr-final075` | `-0.004557291666666667` | `-0.021484375` | `-0.002362676585714022` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B32a-GatedLegendreQuadratic-h160-lowboost-temp050-cosine-lr-final050` | `diagnostic_base_not_qualified` | `0.005522903209947749` | `0.002833909596476447` | `0.002688993613471302` | `1` |
| `B32c-GatedLegendreQuadratic-h192-lowboost-temp075-cosine-lr-final075` | `diagnostic_base_not_qualified` | `-0.00044954537476105116` | `0.0025241421409871734` | `-0.0029736875157482245` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 562
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `c830f62aa11f4611da84dab69f83cf1d9bc3fc80fb768182dbc8df43b36355e7` |
| `v124_control_matrix.csv` | `c6ecffe44e7fe658365a22160178c375d66ad6d3a667470c0977afa8b2a29ade` |
| `v124_efficiency_microbench.csv` | `3b33307f0e135c64e2eabee12467c4f18c34c7765470771ba769337a04b3e88f` |
| `v124_expression_battery.csv` | `1107998a241262dca83753afb97dd743c1a103495a7ff5797538ac61bf8d3039` |
| `v124_provenance_audit.csv` | `32eddbffab9ce79a5e8ffd8a4e55df04e786e8dd85a3733cd459f5e1e983524d` |
| `v124_route_decision.json` | `42d6de5df574904c9d96a849091026faf4d9b13aa743b201049d8c48311c8ff1` |
| `v124_task_triage.csv` | `41f531e2b1f9f10d31742476b57a1cae4239273b09d26c0137b7f80ddd25d6ee` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
