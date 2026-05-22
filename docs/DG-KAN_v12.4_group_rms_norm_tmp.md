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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_group_rms_norm_20260520T130000Z
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
    "B12a-LegendreKAN-K4-fan-scale-repair",
    "B14c-GatedLegendreQuadratic-h224-quad-boost",
    "B14d-GatedLegendreQuadratic-h228-quad-boost",
    "B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B6a-BSpline-order1-local-K4",
    "B6b-BSpline-order1-local-K8-expression-repair",
    "B8d-BSpline-ReLU-lite-combo-K4-repair",
    "B9a-Poly2SignedPair-stream-K3-repair",
    "B9b-Poly2SignedPair-h64-diagnostic-repair",
    "B9c-Poly2SignedPair-h64-K2-diagnostic-repair"
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.7088441857729488` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.3018644443137424` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `2.2775911558587847` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `3.528589356624462` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `3.580508049769014` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `2.527186461808823` | `1.1750887735591369` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `2.6670673505361164` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.9645201507114953` | `0.506111718109806` | `0` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `2.5619425849088886` | `1.0923415733406172` | `0` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `2.794244210190493` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `3.3831159369920436` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.3603003491530499` | `0.5075628243649276` | `1` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `2.4409204611885817` | `1.1272193389784213` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.94835232822169` | `0.5418430756623873` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `3.093957861023185` | `1.1451959847036328` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.9104511096615053` | `0.5075628243649276` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `3.1134353077679875` | `1.0854616225075115` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.6949381050916819` | `0.505736137667304` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `4.323422283592836` | `1.0465549030319585` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `2.1729798047309434` | `0.369263862332696` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `3.811745529094305` | `1.078769461895657` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `1.9698895596287107` | `0.4681951652553947` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `3.927728638213991` | `1.0960290904124557` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `1.4173026144594634` | `0.4685536738596012` | `1` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `3.7241805641700174` | `1.100279978148047` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `1.2617341223586278` | `0.4751775471182737` | `1` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `3.6973817531439814` | `1.097821633433488` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `1.6806900451278983` | `0.4692706910680142` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `3.4783220780933646` | `1.10099699535646` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `1.3849083762868226` | `0.4758945643266867` | `1` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `4.770890812587333` | `1.07450150232177` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `2.222239525408504` | `0.4699877082764272` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `4.563237539032424` | `1.0776768642447419` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `2.2289588056434826` | `0.4766457252116908` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `6.820111545398808` | `1.0874931712646818` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `1.9057032787235864` | `0.4778748975689702` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `5.576281765085408` | `1.094390193936083` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `1.992752267088716` | `0.4844987708276427` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `6.143200427039667` | `1.0917952745151598` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `1.5977795630507314` | `0.4785919147773832` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `5.4867209555992815` | `1.095107211144496` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `1.612957650711495` | `0.4852499317126468` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `5.050413093150256` | `1.1137155148866429` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `1.504425568376251` | `0.4722753346080306` | `0` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `4.567221876158497` | `1.1135106528270964` | `0` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `1.6249372852325548` | `0.4788992078667031` | `0` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `4.980028855476659` | `1.1109157334061732` | `0` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `2.47454326271851` | `0.4729923518164436` | `0` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `5.1507547159722815` | `1.1142276700355094` | `0` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `1.5645713696281405` | `0.4796503687517072` | `0` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `2.6200005302349787` | `1.1387257579896204` | `0` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `4.366107647503351` | `1.1119741873804971` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `3.147573863069268` | `1.0853762633160338` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `2.929310543031911` | `1.1701037967768368` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.3829713477058203` | `0.5115917782026769` | `1` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `2.9027912193087517` | `1.3167338158972959` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.4262435346978641` | `0.5099016662114176` | `1` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `3.056688245893575` | `1.1329896203223162` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `1.4092134120041635` | `0.5480401529636711` | `1` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `2.874911553240939` | `1.0825081944823818` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `1.232458245109362` | `0.4755189838841847` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `2.8707426364174293` | `0.9156309751434034` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `1.204429979574814` | `0.31125375580442505` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.7617726646491003` | `0.863527724665392` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.308936397645935` | `0.25051215514886643` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.9646292053426297` | `0.8570062824364928` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.9878641912267945` | `0.2506828735318219` | `0` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `1.19664789011036` | `0.7691546025676045` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `1.083206789859412` | `0.15706091231903851` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.574149574386175` | `0.7703325594099972` | `0` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.448413817656201` | `0.1718109806063917` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.9517114113696639` | `0.7926283802239825` | `0` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.2513371724128097` | `0.20131111718109806` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.908573231243049` | `0.8213773559136848` | `0` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.2891330123406621` | `0.2603113903305108` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `2.333852651709585` | `0.8638179459164163` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.285870841888385` | `0.25247541655285444` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `2.517282541278125` | `0.9546401256487299` | `0` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.1452418317078734` | `0.37634867522534826` | `1` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `2.7654316201842186` | `1.0939633979786942` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10a-QuadraticSketch-h64-diagnostic', 'B10b-QuadraticSketch-h128-diagnostic', 'B10c-QuadraticSketch-h256-diagnostic', 'B10d-QuadraticSketch-h512-diagnostic', 'B11a-TrainableQuadraticSketch-h128', 'B11b-TrainableQuadraticSketch-h256', 'B12a-LegendreKAN-K4-fan-scale-repair', 'B14c-GatedLegendreQuadratic-h224-quad-boost', 'B14d-GatedLegendreQuadratic-h228-quad-boost', 'B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain', 'B1r-ReLU-KAN-stream-K2-repair', 'B6a-BSpline-order1-local-K4', 'B6b-BSpline-order1-local-K8-expression-repair', 'B8d-BSpline-ReLU-lite-combo-K4-repair', 'B9a-Poly2SignedPair-stream-K3-repair', 'B9b-Poly2SignedPair-h64-diagnostic-repair', 'B9c-Poly2SignedPair-h64-K2-diagnostic-repair']
A2 expression pass = []
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B10a-QuadraticSketch-h64-diagnostic` | `-0.2810329861111111` | `-0.376953125` | `0.187868672526545` |
| `B10b-QuadraticSketch-h128-diagnostic` | `-0.1560329861111111` | `-0.23046875` | `0.16646509576174948` |
| `B10c-QuadraticSketch-h256-diagnostic` | `-0.06966145833333333` | `-0.130859375` | `0.10470389781726731` |

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
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `diagnostic_base_not_qualified` | `-7.763039190411547e-05` | `0.0` | `-7.763039190411547e-05` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `diagnostic_base_not_qualified` | `-7.895683709024937e-05` | `0.001226452343282336` | `-0.0013054091803725854` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `diagnostic_base_not_qualified` | `-8.342782527348547e-05` | `0.0008029128400779406` | `-0.0008863406653514261` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `diagnostic_base_not_qualified` | `-0.002065216998578867` | `0.0` | `-0.002065216998578867` | `0` |
| `B6a-BSpline-order1-local-K4` | `diagnostic_base_not_qualified` | `0.002269369158875989` | `0.006272526008597623` | `-0.004003156849721634` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `diagnostic_base_not_qualified` | `0.02617166891475997` | `0.07238561982052971` | `-0.04621395090576974` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `diagnostic_base_not_qualified` | `0.0014408970357095985` | `0.0037932873636172815` | `-0.002352390327907683` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `diagnostic_base_not_qualified` | `-4.055223265186925e-06` | `1.016035103074131e-05` | `-1.4215574295928235e-05` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `diagnostic_base_not_qualified` | `4.14654969960182e-05` | `1.1681385773343322e-05` | `2.978411122267488e-05` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `-7.393090586393924e-07` | `2.451584039064869e-05` | `-2.5255149449288083e-05` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 2076
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `1d4cc2818356c4135e4d708e4c3e4e4bba1b8c971e84869c4ad1e74a8b1ac32e` |
| `v124_control_matrix.csv` | `cccd700c3f930cb340dc6c891136528d00fcdf437a1d4b5a13f91210a1ce5a0a` |
| `v124_efficiency_microbench.csv` | `f40aa6cb3b188eca7a69d2f230d243a18afab5cfb77664ca1520cb3e1343a047` |
| `v124_expression_battery.csv` | `fab880c46926dc31f097ae4eb531252a1f3af9d83c6160e55c0760b357d4ca08` |
| `v124_provenance_audit.csv` | `1d708fabe156fa217d3974e2dfa7e1d7415e7c2f61cba03a5a7c9a2c41cc946e` |
| `v124_route_decision.json` | `5e76c857b0f1d753fb509a85ef7c199a38450c27ac23b9020c36326a9a31ec66` |
| `v124_task_triage.csv` | `2142719bd89fc37baf070e14c48777cfc72bd6be63739d19a443e51f45495e40` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
