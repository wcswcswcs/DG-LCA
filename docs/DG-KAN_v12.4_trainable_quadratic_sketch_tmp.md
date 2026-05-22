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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_trainable_quadratic_sketch_20260520T020000Z
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
    "B10a-QuadraticSketch-h64-diagnostic",
    "B10b-QuadraticSketch-h128-diagnostic",
    "B10c-QuadraticSketch-h256-diagnostic",
    "B10d-QuadraticSketch-h512-diagnostic",
    "B11a-TrainableQuadraticSketch-h128",
    "B11b-TrainableQuadraticSketch-h256",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B2b-FastKAN-RBF-K8",
    "B3a-ChebyKAN-K4",
    "B3b-LegendreKAN-K4",
    "B4a-FourierKAN-lowfreq-K4",
    "B6a-BSpline-order1-local-K4",
    "B6b-BSpline-order1-local-K8-expression-repair",
    "B8d-BSpline-ReLU-lite-combo-K4-repair",
    "B9a-Poly2SignedPair-stream-K3-repair",
    "B9b-Poly2SignedPair-h64-diagnostic-repair",
    "B9c-Poly2SignedPair-h64-K2-diagnostic-repair",
    "B9d-Poly2PairRandom-h64-K2-diagnostic-repair"
  ],
  "A2_expression_pass": [
    "B3b-LegendreKAN-K4"
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.8116536487220793` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.0696056591752314` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `1.7317526371525374` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `2.524658431192297` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `1.8417898733500553` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `1.7089824805145573` | `1.1750887735591369` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.4729594185881316` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `0.8466751081529773` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `1.425754266376073` | `1.0923415733406172` | `1` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `1.4598708576359614` | `1.1100450696531003` | `1` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `1.188554645413467` | `1.1635823545479378` | `1` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `2.7041549150108577` | `1.1368307839388145` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `1.6253219704625224` | `1.0921025676044795` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.463146881993472` | `1.1769837476099427` | `1` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `0.9180248028019378` | `0.5075628243649276` | `1` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.4693043305035436` | `1.310878175361923` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `0.8490901088066943` | `0.5061458617863972` | `1` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `1.6046584548920086` | `1.1455032777929528` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `0.8264647179095694` | `0.577711007921333` | `1` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `1.5444729045800876` | `1.1376160885004096` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `0.7739666958480341` | `0.515023217700082` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `1.3898940346376556` | `0.9338124829281617` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `0.7148536233370536` | `0.3053981152690522` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.0046118322918844` | `0.8576720841300192` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `0.8118605603744237` | `0.24465651461349358` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.2407855190546342` | `0.8511506419011199` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `0.7851608598202849` | `0.24482723299644907` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `0.8490591016834825` | `0.7632989620322316` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `1.3243892156303811` | `0.15120527178366566` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.3313093998719423` | `0.7644769188746244` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.9333881411117717` | `0.16595534007101884` | `0` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.1356641956205826` | `0.7867727396886097` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.4763661205243586` | `0.1954554766457252` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.3512768022347246` | `0.815521715378312` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.4718108383786288` | `0.25445574979513796` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.9603630874878815` | `0.8579623053810435` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `0.996431284199178` | `0.24661977601748156` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.1481194476018457` | `0.948784485113357` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `0.7702694749420969` | `0.3704930346899754` | `1` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `1.6798676476416348` | `1.0881077574433216` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10a-QuadraticSketch-h64-diagnostic', 'B10b-QuadraticSketch-h128-diagnostic', 'B10c-QuadraticSketch-h256-diagnostic', 'B10d-QuadraticSketch-h512-diagnostic', 'B11a-TrainableQuadraticSketch-h128', 'B11b-TrainableQuadraticSketch-h256', 'B1r-ReLU-KAN-stream-K2-repair', 'B2b-FastKAN-RBF-K8', 'B3a-ChebyKAN-K4', 'B3b-LegendreKAN-K4', 'B4a-FourierKAN-lowfreq-K4', 'B6a-BSpline-order1-local-K4', 'B6b-BSpline-order1-local-K8-expression-repair', 'B8d-BSpline-ReLU-lite-combo-K4-repair', 'B9a-Poly2SignedPair-stream-K3-repair', 'B9b-Poly2SignedPair-h64-diagnostic-repair', 'B9c-Poly2SignedPair-h64-K2-diagnostic-repair', 'B9d-Poly2PairRandom-h64-K2-diagnostic-repair']
A2 expression pass = ['B3b-LegendreKAN-K4']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B3b-LegendreKAN-K4` | `-0.3294270833333333` | `-0.4296875` | `0.3057285402384069` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B10a-QuadraticSketch-h64-diagnostic` | `diagnostic_base_not_qualified` | `1.3359263539403088e-05` | `0.0008205875754356828` | `-0.0008072283118962797` | `0` |
| `B10b-QuadraticSketch-h128-diagnostic` | `diagnostic_base_not_qualified` | `0.001973967254161657` | `0.002069243043661295` | `-9.527578949963811e-05` | `0` |
| `B10c-QuadraticSketch-h256-diagnostic` | `diagnostic_base_not_qualified` | `-0.0019781291484832764` | `0.0` | `-0.0019781291484832764` | `0` |
| `B10d-QuadraticSketch-h512-diagnostic` | `diagnostic_base_not_qualified` | `-5.185604097235341e-07` | `0.0` | `-5.185604097235341e-07` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `diagnostic_base_not_qualified` | `-0.00015688857346418672` | `0.00289104945078833` | `-0.0030479380242525167` | `0` |
| `B11b-TrainableQuadraticSketch-h256` | `diagnostic_base_not_qualified` | `-0.00013237509664576486` | `6.053881910617065e-05` | `-0.0001929139157519355` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `diagnostic_base_not_qualified` | `-0.002065216998578867` | `0.0` | `-0.002065216998578867` | `0` |
| `B2b-FastKAN-RBF-K8` | `diagnostic_base_not_qualified` | `-0.013029543715944447` | `0.0` | `-0.013029543715944447` | `0` |
| `B3a-ChebyKAN-K4` | `diagnostic_base_not_qualified` | `0.0016249120845213127` | `0.0009381397969456806` | `0.0006867722875756321` | `1` |
| `B3b-LegendreKAN-K4` | `diagnostic_base_not_qualified` | `-0.006152881437051727` | `0.003726346148791748` | `-0.009879227585843475` | `0` |
| `B4a-FourierKAN-lowfreq-K4` | `diagnostic_base_not_qualified` | `0.003803054025772079` | `0.001866741251363102` | `0.001936312774408977` | `1` |
| `B6a-BSpline-order1-local-K4` | `diagnostic_base_not_qualified` | `0.002269369158875989` | `0.006272526008597623` | `-0.004003156849721634` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `diagnostic_base_not_qualified` | `0.02617166891475997` | `0.07238561982052971` | `-0.04621395090576974` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `diagnostic_base_not_qualified` | `0.0014408970357095985` | `0.0037932873636172815` | `-0.002352390327907683` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `diagnostic_base_not_qualified` | `-4.055223265186925e-06` | `1.016035103074131e-05` | `-1.4215574295928235e-05` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `diagnostic_base_not_qualified` | `4.14654969960182e-05` | `1.1681385773343322e-05` | `2.978411122267488e-05` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `-7.393090586393924e-07` | `2.451584039064869e-05` | `-2.5255149449288083e-05` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `5.402795305364805e-06` | `0.0` | `5.402795305364805e-06` | `1` |

## 6. No-Fake / Hash

```text
rows_checked = 1761
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `b93ed092ea63eb654b6bea198399f32177abb7eb0e72c68269a2895eb9af4167` |
| `v124_control_matrix.csv` | `650854c7f2c4edef97d3c7ca72cd8b90ffbca71bf2b808e1d5484d9c046f8b81` |
| `v124_efficiency_microbench.csv` | `7c0993e0bb858e38e72e5805c4a2c1b4200bdb83b407db00f26a21dc3660a32e` |
| `v124_expression_battery.csv` | `d2618f77cf105f7fbc2e41051f717bef614ba1e30991827ec1c8bdff71e29c0a` |
| `v124_provenance_audit.csv` | `e2fcb395edb461a20cd3d7c352221a93195f291354e1b27e9363e366253e1986` |
| `v124_route_decision.json` | `f9faf9ab467765b40accb9dcf050581b9716959a099b71dcef34e305f8192780` |
| `v124_task_triage.csv` | `e45f081cae1e7f49f5b0b95bcd555b4deb745e04c7cbf0600ea5990d25eec29d` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
