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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b36_directskip_task_stability_20260520T110000Z
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
    "B36a-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-cosine-lr-final050",
    "B36b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-cosine-lr-final050",
    "B36c-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final075"
  ],
  "A2_expression_pass": [
    "B36a-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-cosine-lr-final050",
    "B36b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-cosine-lr-final050",
    "B36c-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final075"
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
| `B36a-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-cosine-lr-final050` | `GatedHybrid` | `6.492918942627123` | `1.080049849767823` | `0` |
| `B36a-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-cosine-lr-final050` | `GatedHybrid` | `1.378378908100737` | `0.41778202676864246` | `1` |
| `B36b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-cosine-lr-final050` | `GatedHybrid` | `4.832525423315527` | `1.1270998361103524` | `0` |
| `B36b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-cosine-lr-final050` | `GatedHybrid` | `1.3463464719410614` | `0.4426556951652554` | `1` |
| `B36c-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final075` | `GatedHybrid` | `5.4943160790915355` | `1.1316238732586725` | `0` |
| `B36c-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final075` | `GatedHybrid` | `1.0688823926853201` | `0.44296298825457525` | `1` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B36a-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-cosine-lr-final050', 'B36b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-cosine-lr-final050', 'B36c-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final075']
A2 expression pass = ['B36a-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-cosine-lr-final050', 'B36b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-cosine-lr-final050', 'B36c-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final075']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B36a-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-cosine-lr-final050` | `-0.011501736111111112` | `-0.033203125` | `-0.016907722792691655` |
| `B36b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-cosine-lr-final050` | `-0.0008680555555555555` | `-0.01171875` | `-0.019557193542520206` |
| `B36c-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final075` | `0.005208333333333333` | `-0.01171875` | `-0.00795710107518567` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B36a-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-cosine-lr-final050` | `diagnostic_base_not_qualified` | `-4.34247180880476e-06` | `0.0002523078626324171` | `-0.00025665033444122187` | `0` |
| `B36b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-cosine-lr-final050` | `diagnostic_base_not_qualified` | `-0.0005323676143538236` | `0.0030359667877211116` | `-0.0035683344020749352` | `0` |
| `B36c-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final075` | `diagnostic_base_not_qualified` | `-5.8277758734348595e-06` | `0.0031499523357334525` | `-0.0031557801116068873` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 757
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `e3f795cd0c6b83cb95b82abc4eec109a355360aa7f1aede1aa974e0b8ebbc4f6` |
| `v124_control_matrix.csv` | `e8c32054d9f323ecec147c7ab7773fb94a39b9f69f71980c551e9bda448a5e66` |
| `v124_efficiency_microbench.csv` | `352e6ab375a86ff59005f1908785834c55cedd264bfff640ec7b2c0c7e287e86` |
| `v124_expression_battery.csv` | `c0aa8e1ecd15a4b400890525be745b2145efe783e6e6760e79ed513a72280bee` |
| `v124_provenance_audit.csv` | `2fb776b5099c581b419d0317c8ee053cc910c670aeb0b9843a5fa46166ef67a6` |
| `v124_route_decision.json` | `75d7d78e9269185db3aad850e1a43bed5723502fd51101810370d090adf060be` |
| `v124_task_triage.csv` | `da688b93be0661718af6053139f1c4ed2d5e92d571e46bf0d3f816cdcea65dc2` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
