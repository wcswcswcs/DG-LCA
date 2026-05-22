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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b31_lite_quadboost_inputnorm_20260520T060000Z
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
    "B31a-LiteGatedLegendreQuadratic-h224-midboost-temp100-cosine-lr-final050",
    "B31b-LiteGatedLegendreQuadratic-h256-quadboost-temp075-cosine-lr-final075",
    "B31c-LiteGatedLegendreQuadratic-h256-quadboost-temp100-cosine-lr-final050"
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
| `B31a-LiteGatedLegendreQuadratic-h224-midboost-temp100-cosine-lr-final050` | `LiteGatedHybrid` | `2.1116982186506683` | `0.9901154056268779` | `0` |
| `B31a-LiteGatedLegendreQuadratic-h224-midboost-temp100-cosine-lr-final050` | `LiteGatedHybrid` | `1.2875099311280598` | `0.3799849767822999` | `1` |
| `B31b-LiteGatedLegendreQuadratic-h256-quadboost-temp075-cosine-lr-final075` | `LiteGatedHybrid` | `2.332150376262163` | `1.0194618956569244` | `0` |
| `B31b-LiteGatedLegendreQuadratic-h256-quadboost-temp075-cosine-lr-final075` | `LiteGatedHybrid` | `1.2838843423995487` | `0.4115508057907675` | `1` |
| `B31c-LiteGatedLegendreQuadratic-h256-quadboost-temp100-cosine-lr-final050` | `LiteGatedHybrid` | `2.439300827673511` | `1.0235762086861513` | `0` |
| `B31c-LiteGatedLegendreQuadratic-h256-quadboost-temp100-cosine-lr-final050` | `LiteGatedHybrid` | `0.8262067292631053` | `0.4123190385140672` | `1` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B31a-LiteGatedLegendreQuadratic-h224-midboost-temp100-cosine-lr-final050', 'B31b-LiteGatedLegendreQuadratic-h256-quadboost-temp075-cosine-lr-final075', 'B31c-LiteGatedLegendreQuadratic-h256-quadboost-temp100-cosine-lr-final050']
A2 expression pass = []
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B31a-LiteGatedLegendreQuadratic-h224-midboost-temp100-cosine-lr-final050` | `0.008246527777777778` | `-0.0078125` | `0.0063912661539183725` |
| `B31b-LiteGatedLegendreQuadratic-h256-quadboost-temp075-cosine-lr-final075` | `0.004340277777777778` | `-0.017578125` | `0.012863874435424805` |
| `B31c-LiteGatedLegendreQuadratic-h256-quadboost-temp100-cosine-lr-final050` | `0.0015190972222222222` | `-0.025390625` | `0.017696744451920193` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B31a-LiteGatedLegendreQuadratic-h224-midboost-temp100-cosine-lr-final050` | `diagnostic_base_not_qualified` | `-1.4964683010099122e-05` | `0.0009831421551194097` | `-0.0009981068381295088` | `0` |
| `B31b-LiteGatedLegendreQuadratic-h256-quadboost-temp075-cosine-lr-final075` | `diagnostic_base_not_qualified` | `-0.00010467095263666515` | `0.0` | `-0.00010467095263666515` | `0` |
| `B31c-LiteGatedLegendreQuadratic-h256-quadboost-temp100-cosine-lr-final050` | `diagnostic_base_not_qualified` | `-0.00041203573346138` | `3.4712790188962117e-06` | `-0.0004155070124802762` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 757
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `c9c941f594e0bd761e5441f662217c76c0f87c576a52a600f713e736e005a59f` |
| `v124_control_matrix.csv` | `529fe518ce9aab396e02d9a367f3fc5fa17fcc432061f83101ae79e5b0b71d50` |
| `v124_efficiency_microbench.csv` | `bbc98e9eebaf1b61023594fce868df4875172e0eb4ea44f4c66e936e3ecbc64b` |
| `v124_expression_battery.csv` | `3d1b913a0cb0ef1e1e699d197d84d24bf96105396f9fb5660dc7b19940336a7d` |
| `v124_provenance_audit.csv` | `2fb776b5099c581b419d0317c8ee053cc910c670aeb0b9843a5fa46166ef67a6` |
| `v124_route_decision.json` | `6d5aec8f8ffaf332962f7e32a8e6d3ddf7efbaefbf9715b4de1a04c22aba52f1` |
| `v124_task_triage.csv` | `773553980f7de80663c79fa864184e65bac9f2190baf26e1079cabe8d62464c0` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
