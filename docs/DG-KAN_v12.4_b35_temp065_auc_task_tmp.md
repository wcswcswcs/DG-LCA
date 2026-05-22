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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b35_temp065_auc_task_20260520T100000Z
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
    "B35a-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final075",
    "B35b-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final050",
    "B35c-GatedLegendreQuadratic-h168-lowboost-temp065-cosine-lr-final075"
  ],
  "A2_expression_pass": [
    "B35a-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final075",
    "B35b-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final050",
    "B35c-GatedLegendreQuadratic-h168-lowboost-temp065-cosine-lr-final075"
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
| `B35a-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final075` | `GatedHybrid` | `3.9442518228579404` | `1.026461349358099` | `0` |
| `B35a-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final075` | `GatedHybrid` | `1.2763615057504591` | `0.4005394700901393` | `1` |
| `B35b-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final050` | `GatedHybrid` | `4.0239880319577335` | `1.05314463261404` | `0` |
| `B35b-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final050` | `GatedHybrid` | `0.9842848973937696` | `0.40084676317945916` | `1` |
| `B35c-GatedLegendreQuadratic-h168-lowboost-temp065-cosine-lr-final075` | `GatedHybrid` | `3.599152599250454` | `1.0457013111171811` | `0` |
| `B35c-GatedLegendreQuadratic-h168-lowboost-temp065-cosine-lr-final075` | `GatedHybrid` | `1.0773252848383323` | `0.38910133843212236` | `1` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B35a-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final075', 'B35b-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final050', 'B35c-GatedLegendreQuadratic-h168-lowboost-temp065-cosine-lr-final075']
A2 expression pass = ['B35a-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final075', 'B35b-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final050', 'B35c-GatedLegendreQuadratic-h168-lowboost-temp065-cosine-lr-final075']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B35a-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final075` | `0.002170138888888889` | `-0.021484375` | `-0.009280000709825091` |
| `B35b-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final050` | `-0.00043402777777777775` | `-0.01171875` | `-0.018908972748451762` |
| `B35c-GatedLegendreQuadratic-h168-lowboost-temp065-cosine-lr-final075` | `-0.007161458333333333` | `-0.029296875` | `-0.003160616796877649` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B35a-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final075` | `diagnostic_base_not_qualified` | `0.00013740584162391656` | `0.002665657697620283` | `-0.0025282518559963663` | `0` |
| `B35b-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final050` | `diagnostic_base_not_qualified` | `0.00013740584162391656` | `0.002665657697620283` | `-0.0025282518559963663` | `0` |
| `B35c-GatedLegendreQuadratic-h168-lowboost-temp065-cosine-lr-final075` | `diagnostic_base_not_qualified` | `-0.000512307167539916` | `0.002531306120541643` | `-0.003043613288081559` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 757
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `0dad7e10dc7b3a80927194a8e43df769256e87d469459e87769c686b06ff1bc7` |
| `v124_control_matrix.csv` | `c4adecd7a61917797abc665c23844d3d5c1c65343403a82427a469297f078b58` |
| `v124_efficiency_microbench.csv` | `284edb6ff665cf7f9260a0abcbab681a0da7e961a61e23c094eca0515b0f660c` |
| `v124_expression_battery.csv` | `7056eb4e9a970b742cfb961fcdc0f36acd01f1ff0b2f95c1558ca1a94935cdae` |
| `v124_provenance_audit.csv` | `2fb776b5099c581b419d0317c8ee053cc910c670aeb0b9843a5fa46166ef67a6` |
| `v124_route_decision.json` | `c6e51907dddf9851c64c745e8ea5c2ab9bab1a177b577098474786a8ded9b56e` |
| `v124_task_triage.csv` | `6a57acb5bade8e0248779b2e7add8d71b4fe89e1d3903d1eeec8999945bc538e` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
