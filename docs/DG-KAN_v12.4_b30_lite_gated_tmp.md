# DG-KAN v12.4 Multi-Basis Functional Dual 结果复盘

> 本复盘记录 `DG-KAN_v12.4_多基函数效率优先与Functional双线计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位数据或 CPU offload。Base 未合格时 Functional 只允许 diagnostic。

## 0. 最新结论

```text
route = R2-EfficiencyPassExpressionFail
base_qualified = False
functional_open = False
functional_diagnostic_positive = False
next_recommended_action = increase K within efficiency budget, improve init/normalization, or try lite hybrid
```

最终 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b30_lite_gated_inputnorm_20260520T050000Z
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
  "route": "R2-EfficiencyPassExpressionFail",
  "base_blocker_route": "expression_fail",
  "A1_exploratory_survivors": [
    "B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050"
  ],
  "A2_expression_pass": [],
  "A3_task_pass": [],
  "functional_diagnostic_positive": false,
  "base_qualified": false,
  "functional_open": false,
  "next_recommended_action": "increase K within efficiency budget, improve init/normalization, or try lite hybrid",
  "no_fake": true,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. A1 Efficiency Microbench batch=128

| candidate | family | step ratio q90 vs MLP | memory ratio | exploratory pass |
|---|---|---:|---:|---:|
| `B30a-LiteGatedLegendreQuadratic-h192-temp050-cosine-lr-final050` | `LiteGatedHybrid` | `2.7374577897056165` | `0.9695267686424475` | `0` |
| `B30a-LiteGatedLegendreQuadratic-h192-temp050-cosine-lr-final050` | `LiteGatedHybrid` | `1.684879731220057` | `0.3491020213056542` | `0` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `LiteGatedHybrid` | `3.3433490701123563` | `1.0160304561595193` | `0` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `LiteGatedHybrid` | `1.4590838571380897` | `0.4114654465992898` | `1` |
| `B30c-LiteGatedLegendreQuadratic-h296-temp075-cosine-lr-final075` | `LiteGatedHybrid` | `3.5473112304412378` | `1.0493205408358373` | `0` |
| `B30c-LiteGatedLegendreQuadratic-h296-temp075-cosine-lr-final075` | `LiteGatedHybrid` | `1.986537525737511` | `0.45098675225348267` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050']
A2 expression pass = []
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `0.0015190972222222222` | `-0.017578125` | `-0.000246293842792511` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `diagnostic_base_not_qualified` | `-1.234620362833283e-06` | `0.0` | `-1.234620362833283e-06` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 367
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `4809f31bfcf354a79840a09fa9cf5f0b32ca9cc514ad6642639ff136286833fd` |
| `v124_control_matrix.csv` | `b7b88169065fdb78b904c6fe4a8599d7ec922950b37ad02a6d2ac46ad1fb2f1f` |
| `v124_efficiency_microbench.csv` | `f1ab39b7f69371902036f4bf3c8164093dbad90063500376526b36109532178e` |
| `v124_expression_battery.csv` | `b7e90dc5358d1627697b6fcabe7ad09d5c1ab41ad60c28c5a0ad36826542a6d6` |
| `v124_provenance_audit.csv` | `7681d45fadc5a6211167234f3c37d2ae05de0a61645be58295b818fcb10b7b26` |
| `v124_route_decision.json` | `9a130772664e4e83742dc496d0221a9a3f0a174ac5cb149401fb3ca73482aab2` |
| `v124_task_triage.csv` | `56515212ba973fb31d5d059fa9610a1ec5f1ac20d0103e757f789c26aa9b553e` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
