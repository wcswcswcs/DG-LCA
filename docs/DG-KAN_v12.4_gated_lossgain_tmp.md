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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_gated_lossgain_20260520T080000Z
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
    "B12a-LegendreKAN-K4-fan-scale-repair",
    "B12b-ChebyKAN-K4-fan-scale-repair",
    "B13a-LegendreKAN-K4-identity-residual-repair",
    "B13b-ChebyKAN-K4-identity-residual-repair",
    "B14a-GatedLegendreQuadratic-h160",
    "B14b-GatedLegendreQuadratic-h224",
    "B14c-GatedLegendreQuadratic-h224-quad-boost",
    "B14d-GatedLegendreQuadratic-h228-quad-boost",
    "B15a-GatedLegendreQuadratic-h224-loss-gain",
    "B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B2b-FastKAN-RBF-K8",
    "B6a-BSpline-order1-local-K4",
    "B6b-BSpline-order1-local-K8-expression-repair",
    "B8d-BSpline-ReLU-lite-combo-K4-repair",
    "B9a-Poly2SignedPair-stream-K3-repair",
    "B9b-Poly2SignedPair-h64-diagnostic-repair",
    "B9c-Poly2SignedPair-h64-K2-diagnostic-repair",
    "B9d-Poly2PairRandom-h64-K2-diagnostic-repair"
  ],
  "A2_expression_pass": [
    "B14b-GatedLegendreQuadratic-h224"
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.52280608254764` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.4241881690384535` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `1.7832177233461326` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `3.0502176515209967` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `2.320911567771821` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `1.538986897119438` | `1.1750887735591369` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.5358431456525632` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.384128233766518` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `2.4254079948050933` | `1.0923415733406172` | `0` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `2.5465593757568` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `2.823576440501152` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `0.8280594370351224` | `0.5075628243649276` | `1` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.6071132383336886` | `1.1272193389784213` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `0.6756603173068938` | `0.5418430756623873` | `1` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.2299371570116149` | `1.1451959847036328` | `1` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `0.7756947079293613` | `0.5075628243649276` | `1` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.4952915820140247` | `1.0854616225075115` | `1` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `0.7832923520649191` | `0.505736137667304` | `1` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `2.287649921619819` | `1.046503687517072` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `0.7533843256193848` | `0.36916143130292267` | `1` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `2.444893967142537` | `1.0786670308658837` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `0.8921441458375997` | `0.4680927342256214` | `1` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `2.374316559805639` | `1.0959266593826824` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `0.8737598482540483` | `0.4684512428298279` | `1` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `2.8300684715157263` | `1.1001775471182738` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `1.0717744706767875` | `0.4750751160885004` | `1` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `2.195225257242123` | `1.0977192024037148` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `0.9053869385590702` | `0.4691682600382409` | `1` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `2.589205216160282` | `1.1008945643266868` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `0.95623846692586` | `0.4757921332969134` | `1` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `1.7758091875829558` | `1.1349187380497132` | `0` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `2.92606890620465` | `1.10816716744059` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `2.0961705187456925` | `1.0815692433761268` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.24639185497624` | `1.16629677683693` | `1` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.045415046484747` | `0.5078018301010653` | `1` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.5141760522182082` | `1.3129438677956842` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `0.9710607451070126` | `0.506111718109806` | `1` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `2.0041810773076096` | `1.1291996722207047` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `1.0256955757051613` | `0.5442502048620596` | `1` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `2.000890404722623` | `1.0787182463807703` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `0.8542896961369557` | `0.4717290357825731` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `1.993848604362856` | `0.9118410270417918` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `0.9625070122959825` | `0.30746380770281345` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.5862995022705018` | `0.8597377765637804` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `0.9501974790712503` | `0.24672220704725484` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.594117828467158` | `0.8532163343348812` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `0.6785335229801717` | `0.24689292543021032` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `1.098558570994874` | `0.7653646544659929` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `0.9558646558120818` | `0.15327096421742692` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `0.6889361522607428` | `0.7665426113083856` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `0.7463771027876429` | `0.16802103250478012` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `0.8577160087546561` | `0.7888384321223709` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `0.7376156378005273` | `0.19752116907948647` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.059036122702964` | `0.8175874078120732` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.1115736059448782` | `0.2565214422288992` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.3796489753436858` | `0.8600279978148047` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `0.8579745391910245` | `0.24868546845124284` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.1094125771953325` | `0.9508501775471183` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.3936915235926877` | `0.37255872712373667` | `1` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `2.131008312758146` | `1.0901734498770828` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10a-QuadraticSketch-h64-diagnostic', 'B10b-QuadraticSketch-h128-diagnostic', 'B10c-QuadraticSketch-h256-diagnostic', 'B10d-QuadraticSketch-h512-diagnostic', 'B11a-TrainableQuadraticSketch-h128', 'B11b-TrainableQuadraticSketch-h256', 'B12a-LegendreKAN-K4-fan-scale-repair', 'B12b-ChebyKAN-K4-fan-scale-repair', 'B13a-LegendreKAN-K4-identity-residual-repair', 'B13b-ChebyKAN-K4-identity-residual-repair', 'B14a-GatedLegendreQuadratic-h160', 'B14b-GatedLegendreQuadratic-h224', 'B14c-GatedLegendreQuadratic-h224-quad-boost', 'B14d-GatedLegendreQuadratic-h228-quad-boost', 'B15a-GatedLegendreQuadratic-h224-loss-gain', 'B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain', 'B1r-ReLU-KAN-stream-K2-repair', 'B2b-FastKAN-RBF-K8', 'B6a-BSpline-order1-local-K4', 'B6b-BSpline-order1-local-K8-expression-repair', 'B8d-BSpline-ReLU-lite-combo-K4-repair', 'B9a-Poly2SignedPair-stream-K3-repair', 'B9b-Poly2SignedPair-h64-diagnostic-repair', 'B9c-Poly2SignedPair-h64-K2-diagnostic-repair', 'B9d-Poly2PairRandom-h64-K2-diagnostic-repair']
A2 expression pass = ['B14b-GatedLegendreQuadratic-h224']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B14b-GatedLegendreQuadratic-h224` | `0.0026041666666666665` | `-0.037109375` | `-0.012105057636896769` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B10a-QuadraticSketch-h64-diagnostic` | `diagnostic_base_not_qualified` | `1.3359263539403088e-05` | `0.0008205875754356828` | `-0.0008072283118962797` | `0` |
| `B10b-QuadraticSketch-h128-diagnostic` | `diagnostic_base_not_qualified` | `0.001973967254161657` | `0.002069243043661295` | `-9.527578949963811e-05` | `0` |
| `B10c-QuadraticSketch-h256-diagnostic` | `diagnostic_base_not_qualified` | `-0.0019781291484832764` | `0.0` | `-0.0019781291484832764` | `0` |
| `B10d-QuadraticSketch-h512-diagnostic` | `diagnostic_base_not_qualified` | `-5.185604097235341e-07` | `0.0` | `-5.185604097235341e-07` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `diagnostic_base_not_qualified` | `-0.00015688857346418672` | `0.00289104945078833` | `-0.0030479380242525167` | `0` |
| `B11b-TrainableQuadraticSketch-h256` | `diagnostic_base_not_qualified` | `-0.00013237509664576486` | `6.053881910617065e-05` | `-0.0001929139157519355` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `diagnostic_base_not_qualified` | `2.5516345816344996` | `0.02923315749439226` | `2.5224014241401074` | `1` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `diagnostic_base_not_qualified` | `0.010721973229378179` | `0.059079261395131866` | `-0.04835728816575369` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `diagnostic_base_not_qualified` | `0.01466868295428947` | `0.015135731875359326` | `-0.00046704892106985696` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `diagnostic_base_not_qualified` | `0.0009064327380206016` | `0.00924675009415954` | `-0.008340317356138938` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `diagnostic_base_not_qualified` | `-0.0030250690793711676` | `0.00018007950541143458` | `-0.0032051485847826022` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `diagnostic_base_not_qualified` | `-9.739293402599714e-05` | `0.0009702653393306448` | `-0.001067658273356642` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `diagnostic_base_not_qualified` | `-7.763039190411547e-05` | `0.0` | `-7.763039190411547e-05` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `diagnostic_base_not_qualified` | `-7.895683709024937e-05` | `0.001226452343282336` | `-0.0013054091803725854` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `diagnostic_base_not_qualified` | `-0.00020563623901370676` | `0.0006638135429084535` | `-0.0008694497819221603` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `diagnostic_base_not_qualified` | `-8.342782527348547e-05` | `0.0008029128400779406` | `-0.0008863406653514261` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `diagnostic_base_not_qualified` | `-0.002065216998578867` | `0.0` | `-0.002065216998578867` | `0` |
| `B2b-FastKAN-RBF-K8` | `diagnostic_base_not_qualified` | `-0.013029543715944447` | `0.0` | `-0.013029543715944447` | `0` |
| `B6a-BSpline-order1-local-K4` | `diagnostic_base_not_qualified` | `0.002269369158875989` | `0.006272526008597623` | `-0.004003156849721634` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `diagnostic_base_not_qualified` | `0.02617166891475997` | `0.07238561982052971` | `-0.04621395090576974` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `diagnostic_base_not_qualified` | `0.0014408970357095985` | `0.0037932873636172815` | `-0.002352390327907683` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `diagnostic_base_not_qualified` | `-4.055223265186925e-06` | `1.016035103074131e-05` | `-1.4215574295928235e-05` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `diagnostic_base_not_qualified` | `4.14654969960182e-05` | `1.1681385773343322e-05` | `2.978411122267488e-05` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `-7.393090586393924e-07` | `2.451584039064869e-05` | `-2.5255149449288083e-05` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `5.402795305364805e-06` | `0.0` | `5.402795305364805e-06` | `1` |

## 6. No-Fake / Hash

```text
rows_checked = 2353
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `df369bf862589fa4bca10de49596e3b8549cd0ddc5fec8527afc9390b63c5945` |
| `v124_control_matrix.csv` | `7bdff5e122250075ddb6a39a41a7f2cbdfbe56e3fee0193a1230f3d302082c3f` |
| `v124_efficiency_microbench.csv` | `4196e20c255c6d962e0d104129f2da7fe07d317badb15dad40462816dbd9f7de` |
| `v124_expression_battery.csv` | `b4e7d0b6a9b7a66c3f58cf8f4ede71f2a63bfb550fae2e8e6fee9af81fe971b0` |
| `v124_provenance_audit.csv` | `65deab482ab96286de0a1000e63f7a5c09f583673f08d05dc14bebab4fca18c3` |
| `v124_route_decision.json` | `1d6089fa7a3ccf6a567e08aea00fa24dc592b4eaf3365df2671c2e50179cff64` |
| `v124_task_triage.csv` | `b3216c190b06d354df1db9eddd9704dae8f65783348d67e57381d5385cc88ed7` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
