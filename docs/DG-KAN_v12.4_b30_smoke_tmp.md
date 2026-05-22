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
results/v12_4_multibasis_functional_dual/smoke_b30_lite_gated
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
  "base_blocker_route": "expression_fail",
  "A1_exploratory_survivors": [
    "B30a-LiteGatedLegendreQuadratic-h192-temp050-cosine-lr-final050"
  ],
  "A2_expression_pass": [],
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


## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B30a-LiteGatedLegendreQuadratic-h192-temp050-cosine-lr-final050']
A2 expression pass = []
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B30a-LiteGatedLegendreQuadratic-h192-temp050-cosine-lr-final050` | `0.0` | `0.0` | `0.1796417459845543` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B30a-LiteGatedLegendreQuadratic-h192-temp050-cosine-lr-final050` | `diagnostic_base_not_qualified` | `0.005579818779630763` | `0.0` | `0.005579818779630763` | `1` |

## 6. No-Fake / Hash

```text
rows_checked = 59
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `20fb4b9c7200adce4017248003932ecc354f1f30d6282865c77d70dbc3668502` |
| `v124_control_matrix.csv` | `2b45ceaa321d7c17b1999b4752c5736b5eb3f5772bd427c8c561b7eb810c21bd` |
| `v124_efficiency_microbench.csv` | `6cfec771def4d8747039c0e8d9c2936000d14d99fd0b7998f5539b7e2f7a27d7` |
| `v124_expression_battery.csv` | `3c2e34cdcceabbbe0feb5a64d7a4e6c8583d35b1c13e22f964b49ab661ff00a3` |
| `v124_provenance_audit.csv` | `788404fe78edcf092d8a578aa7a3cd0ac4ebcc86893b97e3bab66f81ab24117e` |
| `v124_route_decision.json` | `3f5a7b0eb60ce6c6bfbd393b395d1c6c896357acc604c2b0e96af5f12932c1b7` |
| `v124_task_triage.csv` | `28562cac9ed04fe0a1d3335f3e3439abd45cb9726c6b596250518751cf0c2c26` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
