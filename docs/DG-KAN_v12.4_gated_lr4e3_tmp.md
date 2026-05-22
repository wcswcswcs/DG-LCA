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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_gated_lr4e3_20260520T090000Z
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
    "B11b-TrainableQuadraticSketch-h256",
    "B12a-LegendreKAN-K4-fan-scale-repair",
    "B12b-ChebyKAN-K4-fan-scale-repair",
    "B13a-LegendreKAN-K4-identity-residual-repair",
    "B13b-ChebyKAN-K4-identity-residual-repair",
    "B14a-GatedLegendreQuadratic-h160",
    "B14b-GatedLegendreQuadratic-h224",
    "B14c-GatedLegendreQuadratic-h224-quad-boost",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B6a-BSpline-order1-local-K4",
    "B6b-BSpline-order1-local-K8-expression-repair",
    "B8d-BSpline-ReLU-lite-combo-K4-repair",
    "B9a-Poly2SignedPair-stream-K3-repair",
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `2.425835791448599` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.0983815101266663` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `1.894763367125867` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `3.423858693533015` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `3.2897627189821677` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `2.6535488298622303` | `1.1750887735591369` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `2.715006542333223` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.7288860699332405` | `0.506111718109806` | `0` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `2.7324469421448883` | `1.0923415733406172` | `0` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `2.872166235971322` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `2.8444459844217014` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.095492668045533` | `0.5075628243649276` | `1` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.7795580670192845` | `1.1272193389784213` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.0257074656051894` | `0.5418430756623873` | `1` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.7669711447180714` | `1.1451959847036328` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.455722910591007` | `0.5075628243649276` | `1` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.6367278722655572` | `1.0854616225075115` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `0.9511348745967034` | `0.505736137667304` | `1` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `4.007363629952148` | `1.046503687517072` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `1.0974703324377644` | `0.36916143130292267` | `1` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `4.407835807121118` | `1.0786670308658837` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `1.2108921752297315` | `0.4680927342256214` | `1` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `4.074345811523586` | `1.0959266593826824` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `1.2210708164731778` | `0.4684512428298279` | `1` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `4.454219670391782` | `1.1001775471182738` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `1.7044404310622927` | `0.4750751160885004` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `4.558907292980869` | `1.0977192024037148` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `2.0079679661231844` | `0.4691682600382409` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `5.246447554128496` | `1.1008945643266868` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `1.8572444857510813` | `0.4757921332969134` | `0` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `3.1871971179178717` | `1.1349187380497132` | `0` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `4.543706364185757` | `1.10816716744059` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `3.3017686250677034` | `1.0815692433761268` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `2.263373498730666` | `1.16629677683693` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.1734558106200579` | `0.5078018301010653` | `1` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `2.4564852937469364` | `1.3129438677956842` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.171588137110385` | `0.506111718109806` | `1` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `3.218197174848775` | `1.1291996722207047` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `1.2993612881376855` | `0.5442502048620596` | `1` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `2.5199271850916167` | `1.0787182463807703` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `1.45274173281619` | `0.4717290357825731` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `2.870037227901201` | `0.9118410270417918` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `2.385059386107366` | `0.30746380770281345` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.8234454677005865` | `0.8597377765637804` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.861595876125529` | `0.24672220704725484` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.9643437882578254` | `0.8532163343348812` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.1100358230391143` | `0.24689292543021032` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `1.3575434438022582` | `0.7653646544659929` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `1.8458101546037498` | `0.15327096421742692` | `0` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.1713188718768601` | `0.7665426113083856` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.7736986213941035` | `0.16802103250478012` | `0` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `0.9644873900881875` | `0.7888384321223709` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.4799863328067293` | `0.19752116907948647` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.4464445895673685` | `0.8175874078120732` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `2.0546764205133643` | `0.2565214422288992` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.6605324483961008` | `0.8600279978148047` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.9869291178155288` | `0.24868546845124284` | `0` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.4934989723641525` | `0.9508501775471183` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.025168179835942` | `0.37255872712373667` | `1` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `3.054343993133546` | `1.0901734498770828` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10a-QuadraticSketch-h64-diagnostic', 'B10b-QuadraticSketch-h128-diagnostic', 'B10c-QuadraticSketch-h256-diagnostic', 'B10d-QuadraticSketch-h512-diagnostic', 'B11b-TrainableQuadraticSketch-h256', 'B12a-LegendreKAN-K4-fan-scale-repair', 'B12b-ChebyKAN-K4-fan-scale-repair', 'B13a-LegendreKAN-K4-identity-residual-repair', 'B13b-ChebyKAN-K4-identity-residual-repair', 'B14a-GatedLegendreQuadratic-h160', 'B14b-GatedLegendreQuadratic-h224', 'B14c-GatedLegendreQuadratic-h224-quad-boost', 'B1r-ReLU-KAN-stream-K2-repair', 'B6a-BSpline-order1-local-K4', 'B6b-BSpline-order1-local-K8-expression-repair', 'B8d-BSpline-ReLU-lite-combo-K4-repair', 'B9a-Poly2SignedPair-stream-K3-repair', 'B9d-Poly2PairRandom-h64-K2-diagnostic-repair']
A2 expression pass = ['B14b-GatedLegendreQuadratic-h224']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B14b-GatedLegendreQuadratic-h224` | `-0.0026041666666666665` | `-0.02734375` | `0.009255512721008725` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B10a-QuadraticSketch-h64-diagnostic` | `diagnostic_base_not_qualified` | `-2.395063638682693e-05` | `0.0019362737236163774` | `-0.0019602243600032043` | `0` |
| `B10b-QuadraticSketch-h128-diagnostic` | `diagnostic_base_not_qualified` | `-6.103515625355271e-06` | `0.00029046833515167236` | `-0.00029657185077702763` | `0` |
| `B10c-QuadraticSketch-h256-diagnostic` | `diagnostic_base_not_qualified` | `0.0019359290599822998` | `0.0005485415458679199` | `0.0013873875141143799` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `diagnostic_base_not_qualified` | `-2.596974372881533e-05` | `0.0006458878517152655` | `-0.0006718575954440809` | `0` |
| `B11b-TrainableQuadraticSketch-h256` | `diagnostic_base_not_qualified` | `-2.9327598193962245e-05` | `1.690340080218178e-05` | `-4.6230998996144024e-05` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `diagnostic_base_not_qualified` | `0.1485627100959086` | `0.0` | `0.1485627100959086` | `1` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `diagnostic_base_not_qualified` | `0.05201941053095638` | `0.03266001299611432` | `0.01935939753484206` | `1` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `diagnostic_base_not_qualified` | `0.005540666608441214` | `0.0` | `0.005540666608441214` | `1` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `diagnostic_base_not_qualified` | `-0.007478621321645562` | `0.0015807997764187842` | `-0.009059421098064346` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `diagnostic_base_not_qualified` | `0.0013006632173535593` | `0.010244217655641208` | `-0.008943554438287649` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `diagnostic_base_not_qualified` | `0.0002149408302356548` | `0.008368806435596454` | `-0.0081538656053608` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `diagnostic_base_not_qualified` | `-0.0002367768435458384` | `0.0019201318846517879` | `-0.0021569087281976262` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `diagnostic_base_not_qualified` | `0.017811575612310726` | `0.022696443791723198` | `-0.0048848681794124715` | `0` |
| `B6a-BSpline-order1-local-K4` | `diagnostic_base_not_qualified` | `0.01231237298410437` | `0.00593157343345041` | `0.006380799550653959` | `1` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `diagnostic_base_not_qualified` | `0.14070629220033726` | `0.0` | `0.14070629220033726` | `1` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `diagnostic_base_not_qualified` | `0.0015569467396112646` | `0.0` | `0.0015569467396112646` | `1` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `diagnostic_base_not_qualified` | `1.1246309530532628e-05` | `1.6904392256655854e-05` | `-5.658082726123226e-06` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `-1.6521377510514412e-06` | `6.523187332518887e-06` | `-8.175325083570328e-06` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 1818
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `df369bf862589fa4bca10de49596e3b8549cd0ddc5fec8527afc9390b63c5945` |
| `v124_control_matrix.csv` | `f4eb4040d10ce98b64f0fc382a989079743885fd5fdd9c4099029b2c7ee13e81` |
| `v124_efficiency_microbench.csv` | `147bd2d57fe8ed3ccce3ffda39a0723e03852dbbafc8e388a0f25c6723cf6e79` |
| `v124_expression_battery.csv` | `87a06cd7259563bfc3f76105bf4c49723681aee4c8bc075f91777dd1ac2229d0` |
| `v124_provenance_audit.csv` | `667b93995b25db206add122bb56d2adef0a3c680f5d4a67387ae3afcd9c3a8a0` |
| `v124_route_decision.json` | `cd13819d3ae97dfe01abcd3c3041acd2cf1ba29f118599a7e20389fae706de06` |
| `v124_task_triage.csv` | `19ad9524e0ff613cfadc54b9688b40f2c5fafecc51f3a5d458023bece06e8532` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
