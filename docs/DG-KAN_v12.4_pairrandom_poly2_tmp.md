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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_pairrandom_poly2_20260519T235500Z
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
    "B1r-ReLU-KAN-stream-K2-repair",
    "B2b-FastKAN-RBF-K8"
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.7131453698874584` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `0.9843010213245061` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `1.9892010959901554` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `2.336927213102334` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `2.062798512788326` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `1.5349404480894386` | `1.1750887735591369` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.5974357390579288` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.001643750567284` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `1.6096904933254743` | `1.0923415733406172` | `0` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `1.9721751656636237` | `1.1100450696531003` | `0` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `1.900373194151892` | `1.1635823545479378` | `0` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `3.182700281657703` | `1.1368307839388145` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `2.0525105936970927` | `1.0921025676044795` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.8481590337264293` | `1.1769837476099427` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.861969203625353` | `0.5075628243649276` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `2.5480563422682083` | `1.310878175361923` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `2.3023210381550774` | `0.5061458617863972` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `4.389431688725118` | `1.1455032777929528` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `2.457149719922101` | `0.577711007921333` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `3.8020699037761005` | `1.1376160885004096` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `2.5565650750650004` | `0.515023217700082` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `4.011027200371711` | `0.9338124829281617` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `1.6545592769743591` | `0.3053981152690522` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `2.63792515483128` | `0.8576720841300192` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `2.4762985382244294` | `0.24465651461349358` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `2.107669140240427` | `0.8511506419011199` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `2.1666330451943825` | `0.24482723299644907` | `0` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `4.0064400693974` | `1.0737161977601748` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B1r-ReLU-KAN-stream-K2-repair', 'B2b-FastKAN-RBF-K8']
A2 expression pass = []
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B1r-ReLU-KAN-stream-K2-repair` | `-0.294921875` | `-0.341796875` | `0.32095512623588246` |
| `B2b-FastKAN-RBF-K8` | `-0.2339409722222222` | `-0.31640625` | `0.3623969443142414` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B1r-ReLU-KAN-stream-K2-repair` | `diagnostic_base_not_qualified` | `-0.002065216998578867` | `0.0` | `-0.002065216998578867` | `0` |
| `B2b-FastKAN-RBF-K8` | `diagnostic_base_not_qualified` | `-0.013029543715944447` | `0.0` | `-0.013029543715944447` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 629
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `4f0262f1fad9f0323f67541efdbdfcf459e6a9163f1a321e5a63844a76afee4b` |
| `v124_control_matrix.csv` | `9fcf3760a2aa5e10fb5d1ae33b9b9340e14865081d75e9a71dd67cb672568f98` |
| `v124_efficiency_microbench.csv` | `72e710488e5fdcc00ad7c583622db6035e6c591e5d52ef43804cc77c1a195cdc` |
| `v124_expression_battery.csv` | `1831f8789ce7edd5b76f017e0868ab8b01a75d75764ecbe24d9ef4656ebb628e` |
| `v124_provenance_audit.csv` | `f95ee173e4b102e65c55c44cc7fb3bae1e87d6a49a787cd9d5cd3a1e60f1af67` |
| `v124_route_decision.json` | `360836bd8610ebc454e52731089301dcf9f736ee8bb7944c4db3112ba7ffd222` |
| `v124_task_triage.csv` | `c91c89aa13d81f6a672cc75605c38b91e670bd42c3f67bec9d1a3fdeb65fb0b6` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
