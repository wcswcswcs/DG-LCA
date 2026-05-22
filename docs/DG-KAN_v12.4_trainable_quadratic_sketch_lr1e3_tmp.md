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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_trainable_quadratic_sketch_lr1e3_20260520T030000Z
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
    "B10a-QuadraticSketch-h64-diagnostic",
    "B10b-QuadraticSketch-h128-diagnostic",
    "B10c-QuadraticSketch-h256-diagnostic",
    "B10d-QuadraticSketch-h512-diagnostic",
    "B11a-TrainableQuadraticSketch-h128",
    "B11b-TrainableQuadraticSketch-h256",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B2a-GaussianRBF-K4-compact",
    "B2b-FastKAN-RBF-K8",
    "B3a-ChebyKAN-K4",
    "B4a-FourierKAN-lowfreq-K4",
    "B6a-BSpline-order1-local-K4",
    "B6b-BSpline-order1-local-K8-expression-repair",
    "B8d-BSpline-ReLU-lite-combo-K4-repair",
    "B9a-Poly2SignedPair-stream-K3-repair",
    "B9b-Poly2SignedPair-h64-diagnostic-repair",
    "B9c-Poly2SignedPair-h64-K2-diagnostic-repair",
    "B9d-Poly2PairRandom-h64-K2-diagnostic-repair"
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.3997666023588065` | `1.017242556678503` | `1` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `0.9400972439830766` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `1.5427392440391112` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `1.8758796156026565` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `1.7453230639343307` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `1.469649870209081` | `1.1750887735591369` | `1` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.5544223015200893` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `0.5723904515794934` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `1.11331961145318` | `1.0923415733406172` | `1` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `1.5112425401337393` | `1.1100450696531003` | `0` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `1.2461581148461376` | `1.1635823545479378` | `1` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `2.261998720786367` | `1.1368307839388145` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `1.6209060502888646` | `1.0921025676044795` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.2797033103428306` | `1.1769837476099427` | `1` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `0.7291050056081466` | `0.5075628243649276` | `1` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.1749937585998265` | `1.310878175361923` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `0.6521423182971399` | `0.5061458617863972` | `1` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `1.533579722391269` | `1.1455032777929528` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `0.648258537386586` | `0.577711007921333` | `1` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `1.4207151577025021` | `1.1376160885004096` | `1` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `0.8895221654851663` | `0.515023217700082` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `1.2668634196248785` | `0.9338124829281617` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `0.6718911291502706` | `0.3053981152690522` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.3161415783904757` | `0.8576720841300192` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `0.7096534489243075` | `0.24465651461349358` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.2893518328052573` | `0.8511506419011199` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `0.5250607970182094` | `0.24482723299644907` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `0.8308580032248014` | `0.7632989620322316` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `0.5501442474287879` | `0.15120527178366566` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `0.7165863243709568` | `0.7644769188746244` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `0.6030622074652874` | `0.16595534007101884` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `0.8847390709359475` | `0.7867727396886097` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `0.8499048713750795` | `0.1954554766457252` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `0.6065678589919542` | `0.815521715378312` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `0.5677734298308827` | `0.25445574979513796` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.108333573754394` | `0.8579623053810435` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `0.5971592021048178` | `0.24661977601748156` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `0.834096553028571` | `0.948784485113357` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `0.8432953926332833` | `0.3704930346899754` | `1` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `1.6119273391658613` | `1.0881077574433216` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10a-QuadraticSketch-h64-diagnostic', 'B10b-QuadraticSketch-h128-diagnostic', 'B10c-QuadraticSketch-h256-diagnostic', 'B10d-QuadraticSketch-h512-diagnostic', 'B11a-TrainableQuadraticSketch-h128', 'B11b-TrainableQuadraticSketch-h256', 'B1r-ReLU-KAN-stream-K2-repair', 'B2a-GaussianRBF-K4-compact', 'B2b-FastKAN-RBF-K8', 'B3a-ChebyKAN-K4', 'B4a-FourierKAN-lowfreq-K4', 'B6a-BSpline-order1-local-K4', 'B6b-BSpline-order1-local-K8-expression-repair', 'B8d-BSpline-ReLU-lite-combo-K4-repair', 'B9a-Poly2SignedPair-stream-K3-repair', 'B9b-Poly2SignedPair-h64-diagnostic-repair', 'B9c-Poly2SignedPair-h64-K2-diagnostic-repair', 'B9d-Poly2PairRandom-h64-K2-diagnostic-repair']
A2 expression pass = []
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B10a-QuadraticSketch-h64-diagnostic` | `-0.4294704861111111` | `-0.525390625` | `0.14302263470987478` |
| `B10b-QuadraticSketch-h128-diagnostic` | `-0.2628038194444444` | `-0.384765625` | `0.2262862686895662` |
| `B10c-QuadraticSketch-h256-diagnostic` | `-0.1410590277777778` | `-0.197265625` | `0.2095145167162021` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B10a-QuadraticSketch-h64-diagnostic` | `diagnostic_base_not_qualified` | `2.3183226585343775e-05` | `4.632025957018726e-06` | `1.855120062832505e-05` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `diagnostic_base_not_qualified` | `3.1049549579442726e-05` | `3.586187958681819e-05` | `-4.812330007375465e-06` | `0` |
| `B10c-QuadraticSketch-h256-diagnostic` | `diagnostic_base_not_qualified` | `6.890296936123974e-06` | `0.00018443167209625244` | `-0.00017754137516012847` | `0` |
| `B10d-QuadraticSketch-h512-diagnostic` | `diagnostic_base_not_qualified` | `-4.020333290055689e-06` | `0.0` | `-4.020333290055689e-06` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `diagnostic_base_not_qualified` | `-7.276062565719776e-05` | `0.0` | `-7.276062565719776e-05` | `0` |
| `B11b-TrainableQuadraticSketch-h256` | `diagnostic_base_not_qualified` | `-2.4310526315574066e-05` | `0.0002481965280458631` | `-0.0002725070543614372` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `diagnostic_base_not_qualified` | `0.017852717013123076` | `0.004026723359367601` | `0.013825993653755475` | `1` |
| `B2a-GaussianRBF-K4-compact` | `diagnostic_base_not_qualified` | `0.03789133035957093` | `0.02505277145135487` | `0.012838558908216058` | `1` |
| `B2b-FastKAN-RBF-K8` | `diagnostic_base_not_qualified` | `0.008765733814205312` | `0.005471124919033343` | `0.0032946088951719688` | `1` |
| `B3a-ChebyKAN-K4` | `diagnostic_base_not_qualified` | `0.0008717792309687056` | `0.0` | `0.0008717792309687056` | `1` |
| `B4a-FourierKAN-lowfreq-K4` | `diagnostic_base_not_qualified` | `0.00029402678695955586` | `0.0` | `0.00029402678695955586` | `1` |
| `B6a-BSpline-order1-local-K4` | `diagnostic_base_not_qualified` | `-0.01391883570537189` | `0.0` | `-0.01391883570537189` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `diagnostic_base_not_qualified` | `0.1406535829418052` | `0.12223579896823544` | `0.018417783973569746` | `1` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `diagnostic_base_not_qualified` | `0.0046931367618221476` | `0.005753795710059073` | `-0.0010606589482369255` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `diagnostic_base_not_qualified` | `4.1797515049069034e-07` | `0.0` | `4.1797515049069034e-07` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `diagnostic_base_not_qualified` | `-2.9921119201858914e-05` | `3.5351276263639875e-05` | `-6.527239546549879e-05` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `8.130204559364529e-06` | `2.324718873447651e-05` | `-1.511698417511198e-05` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `7.523389626840071e-07` | `0.0` | `7.523389626840071e-07` | `1` |

## 6. No-Fake / Hash

```text
rows_checked = 1994
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `b93ed092ea63eb654b6bea198399f32177abb7eb0e72c68269a2895eb9af4167` |
| `v124_control_matrix.csv` | `9b282747797c9847861c8bd50304ab599bca584b206a50ed97b1d0e1204ab7e5` |
| `v124_efficiency_microbench.csv` | `e3d9f4ac46972e458db50d54b2ae00f24390214f2b3722ddb10f33da9fefe870` |
| `v124_expression_battery.csv` | `6c97fa0c7113e73971be890c809ccc07bfc20798dc9ab9e66e95a1297ae6f21e` |
| `v124_provenance_audit.csv` | `b30f944b87b88ce2f5ac6955ec21fba937412edfa51933c2b8617e10607807ff` |
| `v124_route_decision.json` | `cb38fb6a8af57708306a9a28c86daf2f8044a00147cc8c15f474e707e3f915f9` |
| `v124_task_triage.csv` | `0729b6052d98698b9293b760df2894a9e18687af9ffbe149ba5ddbd1161ee679` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
