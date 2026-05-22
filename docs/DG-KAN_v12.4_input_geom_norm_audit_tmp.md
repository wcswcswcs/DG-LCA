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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_input_geom_norm_audit_20260520T120000Z
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
    "B12b-ChebyKAN-K4-fan-scale-repair",
    "B13a-LegendreKAN-K4-identity-residual-repair",
    "B13b-ChebyKAN-K4-identity-residual-repair",
    "B14b-GatedLegendreQuadratic-h224",
    "B14c-GatedLegendreQuadratic-h224-quad-boost",
    "B14d-GatedLegendreQuadratic-h228-quad-boost",
    "B15a-GatedLegendreQuadratic-h224-loss-gain",
    "B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B2a-GaussianRBF-K4-compact",
    "B2b-FastKAN-RBF-K8",
    "B6a-BSpline-order1-local-K4",
    "B6b-BSpline-order1-local-K8-expression-repair",
    "B8d-BSpline-ReLU-lite-combo-K4-repair",
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `2.11784596659473` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `0.8243372140984956` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `1.8649275997557462` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `2.249220683393926` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `1.7078445996446565` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `1.3270305192477942` | `1.1750887735591369` | `1` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.6340302499133033` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `0.8591635871855309` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `1.7316714380036453` | `1.0923415733406172` | `0` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `1.8478108242883888` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.6288746454299297` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `0.9111530959431164` | `0.5075628243649276` | `1` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.6060073758049518` | `1.1272193389784213` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `0.8041434888844057` | `0.5418430756623873` | `1` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.7221789677197006` | `1.1451959847036328` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.0698530572406963` | `0.5075628243649276` | `1` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `2.25506111475534` | `1.0854616225075115` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.0449579462064194` | `0.505736137667304` | `1` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `3.1444529062638837` | `1.0465378311936628` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `1.6739292880464391` | `0.3692297186561049` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `3.903054373997699` | `1.0787353182190658` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `0.8997853832720459` | `0.4681610215788036` | `1` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `3.708783652199553` | `1.0959949467358645` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `1.4517816114758917` | `0.4685195301830101` | `1` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `3.8407371956180305` | `1.1002458344714559` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `1.4574127460023072` | `0.4751434034416826` | `1` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `3.0661094907128907` | `1.097787489756897` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `1.1352640436228048` | `0.4692365473914231` | `1` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `2.6778162292678247` | `1.1009628516798688` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `1.949142698057269` | `0.4758604206500956` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `4.844348122662259` | `1.074467358645179` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `1.9250186931814162` | `0.4699535645998361` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `4.808914613809567` | `1.0776427205681507` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `2.0810171190020292` | `0.4766115815350997` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `4.014920669581289` | `1.0874590275880907` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `1.6313675202324913` | `0.4778407538923791` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `4.552497602270966` | `1.094356050259492` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `1.4100122612002235` | `0.48446462715105165` | `1` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `4.6100271950701375` | `1.0917611308385686` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `1.8609649554238297` | `0.4785577711007921` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `4.929281525793012` | `1.095073067467905` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `1.5212885973541603` | `0.4852157880360557` | `0` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `1.8749056991321418` | `1.1407573067467904` | `0` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `3.3499536914848553` | `1.1140057361376674` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `2.2467938897729316` | `1.087407812073204` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.7111767293091351` | `1.1721353455340071` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.3921566647245216` | `0.5100894564326687` | `1` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `2.230873016699772` | `1.3152314941272876` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `0.8780688486391849` | `0.5083993444414094` | `1` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `2.380220383523727` | `1.1314872985523081` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `1.4388553848727852` | `0.546537831193663` | `1` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `2.293405614960143` | `1.0810058727123737` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `1.6780560618112716` | `0.4740166621141765` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `4.425624198785553` | `0.9141286533733952` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `1.5292297105127366` | `0.30975143403441685` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `2.339308259646719` | `0.8620254028953838` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.5707361674234726` | `0.24900983337885824` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `2.155419281121036` | `0.8555039606664846` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `0.8290603884189827` | `0.24918055176181372` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `1.3078004182914937` | `0.7676522807975963` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `0.8054038025386148` | `0.15555859054903032` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `0.8464018876157107` | `0.768830237639989` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.3223653150613168` | `0.1703086588363835` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.0165931357794928` | `0.7911260584539743` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.1433342973816982` | `0.19980879541108987` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `0.8369619801007376` | `0.8198750341436766` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `0.9081051710234257` | `0.2588090685605026` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.565605373203214` | `0.862315624146408` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.2015208850329782` | `0.25097309478284624` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.4787995904080025` | `0.9531378038787217` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `0.9375302225981578` | `0.37484635345534006` | `1` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `2.1822013239933296` | `1.0924610762086862` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10a-QuadraticSketch-h64-diagnostic', 'B10b-QuadraticSketch-h128-diagnostic', 'B10c-QuadraticSketch-h256-diagnostic', 'B10d-QuadraticSketch-h512-diagnostic', 'B11a-TrainableQuadraticSketch-h128', 'B11b-TrainableQuadraticSketch-h256', 'B12a-LegendreKAN-K4-fan-scale-repair', 'B12b-ChebyKAN-K4-fan-scale-repair', 'B13a-LegendreKAN-K4-identity-residual-repair', 'B13b-ChebyKAN-K4-identity-residual-repair', 'B14b-GatedLegendreQuadratic-h224', 'B14c-GatedLegendreQuadratic-h224-quad-boost', 'B14d-GatedLegendreQuadratic-h228-quad-boost', 'B15a-GatedLegendreQuadratic-h224-loss-gain', 'B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm', 'B1r-ReLU-KAN-stream-K2-repair', 'B2a-GaussianRBF-K4-compact', 'B2b-FastKAN-RBF-K8', 'B6a-BSpline-order1-local-K4', 'B6b-BSpline-order1-local-K8-expression-repair', 'B8d-BSpline-ReLU-lite-combo-K4-repair', 'B9d-Poly2PairRandom-h64-K2-diagnostic-repair']
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
| `B12b-ChebyKAN-K4-fan-scale-repair` | `diagnostic_base_not_qualified` | `0.010721973229378179` | `0.059079261395131866` | `-0.04835728816575369` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `diagnostic_base_not_qualified` | `0.01466868295428947` | `0.015135731875359326` | `-0.00046704892106985696` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `diagnostic_base_not_qualified` | `0.0009064327380206016` | `0.00924675009415954` | `-0.008340317356138938` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `diagnostic_base_not_qualified` | `-9.739293402599714e-05` | `0.0009702653393306448` | `-0.001067658273356642` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `diagnostic_base_not_qualified` | `-7.763039190411547e-05` | `0.0` | `-7.763039190411547e-05` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `diagnostic_base_not_qualified` | `-7.895683709024937e-05` | `0.001226452343282336` | `-0.0013054091803725854` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `diagnostic_base_not_qualified` | `-0.00020563623901370676` | `0.0006638135429084535` | `-0.0008694497819221603` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `diagnostic_base_not_qualified` | `-7.95033345211138e-05` | `0.00042811441086065827` | `-0.0005076177453817721` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `diagnostic_base_not_qualified` | `-0.002065216998578867` | `0.0` | `-0.002065216998578867` | `0` |
| `B2a-GaussianRBF-K4-compact` | `diagnostic_base_not_qualified` | `0.016744405456739386` | `0.0` | `0.016744405456739386` | `1` |
| `B2b-FastKAN-RBF-K8` | `diagnostic_base_not_qualified` | `-0.013029543715944447` | `0.0` | `-0.013029543715944447` | `0` |
| `B6a-BSpline-order1-local-K4` | `diagnostic_base_not_qualified` | `0.002269369158875989` | `0.006272526008597623` | `-0.004003156849721634` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `diagnostic_base_not_qualified` | `0.02617166891475997` | `0.07238561982052971` | `-0.04621395090576974` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `diagnostic_base_not_qualified` | `0.0014408970357095985` | `0.0037932873636172815` | `-0.002352390327907683` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `5.402795305364805e-06` | `0.0` | `5.402795305364805e-06` | `1` |

## 6. No-Fake / Hash

```text
rows_checked = 2432
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `eaa22b2d2ba1f94aac1ef2541c6dbb722515d209137e7c4bd34972b9174d60e7` |
| `v124_control_matrix.csv` | `d6a483d2674e43872ba5c346eec3c42cc69e6fdf338c48fb6bda8f22076fe36a` |
| `v124_efficiency_microbench.csv` | `c7ab5ee4bd0c695cdfe4f1b8ab6012199f385d29abd4672c2ac51a6c13f85137` |
| `v124_expression_battery.csv` | `ee33bc78c32b8d4d7379d1ce55471e43d7a21925763c00c78cb39cf7248c260b` |
| `v124_provenance_audit.csv` | `a58556662d0ab39a6536a5a7295e65da643e8a4a7c091a637859ca01aea54315` |
| `v124_route_decision.json` | `946bb37080be2ea807970122522d2d19ee845c39b0d2ab11e1c406e13136112e` |
| `v124_task_triage.csv` | `6237c6bb58c05f45090bfa046d96c67774f0663e3bc9ea47a98730c08a7eb377` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
