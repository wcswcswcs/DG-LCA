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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_steady_auc_time_20260520T200000Z
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
    "B10c-QuadraticSketch-h256-diagnostic",
    "B11a-TrainableQuadraticSketch-h128",
    "B11b-TrainableQuadraticSketch-h256",
    "B12b-ChebyKAN-K4-fan-scale-repair",
    "B13b-ChebyKAN-K4-identity-residual-repair",
    "B14a-GatedLegendreQuadratic-h160",
    "B14d-GatedLegendreQuadratic-h228-quad-boost",
    "B18a-GatedLegendreQuadratic-h224-group-rms-norm",
    "B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15",
    "B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm",
    "B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm",
    "B2b-FastKAN-RBF-K8",
    "B6a-BSpline-order1-local-K4",
    "B8d-BSpline-ReLU-lite-combo-K4-repair"
  ],
  "A2_expression_pass": [
    "B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm",
    "B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm"
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.9735767308823955` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.2058068688642145` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `2.2532674615654233` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `2.5841273879922695` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `2.608103578009087` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `2.8907944726882175` | `1.1750887735591369` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `2.3572926413984048` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.4306676329574288` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `3.218900229272983` | `1.0923415733406172` | `0` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `2.6720242746388068` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `2.5098369525022273` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.590570264290974` | `0.5075628243649276` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `2.659051298624041` | `1.1272193389784213` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.1707878726489873` | `0.5418430756623873` | `1` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `2.0649380351980655` | `1.1451959847036328` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.628291779614963` | `0.5075628243649276` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `2.7516272995629985` | `1.0854616225075115` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.1911243942594933` | `0.505736137667304` | `1` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `3.9196348755398214` | `1.0465890467085497` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `1.3818952724103892` | `0.3693321496858782` | `1` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `3.823134423746636` | `1.078837749248839` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `1.5854960319374372` | `0.4682634526085769` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `4.988543987935032` | `1.0960973777656378` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `2.0156937839045868` | `0.4686219612127834` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `7.049127082509334` | `1.100348265501229` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `1.4639568320454817` | `0.4752458344714559` | `1` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `4.157857101513588` | `1.0978899207866704` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `2.0977159523189375` | `0.4693389784211964` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `4.99974335413833` | `1.101065282709642` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `1.6133747664663343` | `0.4759628516798689` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `6.053196637758334` | `1.0745697896749522` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `1.9236740834708008` | `0.47005599562960937` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `5.950167091127168` | `1.0777451515979242` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `2.048678102983122` | `0.476714012564873` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `5.731077944775118` | `1.087561458617864` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `1.8380827890913656` | `0.47794318492215243` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `6.662146725062144` | `1.0944584812892653` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `1.7578440782936688` | `0.4845670581808249` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `7.166501665685906` | `1.0918635618683419` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `2.4542860863777123` | `0.4786602021305654` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `6.273818022207003` | `1.0951754984976783` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `2.278906523829277` | `0.485318219065829` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `7.117023277799753` | `1.1137838022398252` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `1.1661376988452745` | `0.47234362196121277` | `1` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `4.702485545833695` | `1.1135789401802787` | `0` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `1.8448134624722252` | `0.4789674952198853` | `0` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `5.959274100191972` | `1.1109840207593553` | `0` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `2.775708398851384` | `0.47306063916962576` | `0` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `5.283668440075503` | `1.1142959573886917` | `0` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `2.3076733696054124` | `0.4797186561048894` | `0` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `GatedHybrid` | `8.4826047565301` | `1.1170615951925704` | `0` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `GatedHybrid` | `2.4883508503680267` | `0.4809478284621688` | `0` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `GatedHybrid` | `8.393541407408122` | `1.1239586178639716` | `0` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `GatedHybrid` | `2.6713128209580588` | `0.4875717017208413` | `0` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `GatedHybrid` | `5.655986822430808` | `1.1213636984430484` | `0` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `GatedHybrid` | `1.2446237012955694` | `0.4816648456705818` | `1` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `GatedHybrid` | `4.923145517600641` | `1.1246756350723845` | `0` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `GatedHybrid` | `2.109486491222993` | `0.4883228626058454` | `0` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `GatedHybrid` | `6.782532927844931` | `1.1221148593280525` | `0` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `GatedHybrid` | `2.1805319744166085` | `0.482450150232177` | `0` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `GatedHybrid` | `5.978977749587598` | `1.1214319857962305` | `0` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `GatedHybrid` | `2.249600259499716` | `0.4828086588363835` | `0` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `GatedHybrid` | `6.977396071489439` | `1.1258194482381862` | `0` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `GatedHybrid` | `1.3745164526678306` | `0.489432532095056` | `1` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `GatedHybrid` | `4.210869645607637` | `1.1272534826550122` | `0` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `GatedHybrid` | `1.915170001254088` | `0.4897910406992625` | `0` |
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `GatedHybrid` | `5.563848948334798` | `1.0848470363288718` | `0` |
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `GatedHybrid` | `1.3547273886353917` | `0.47678229991805515` | `1` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `GatedHybrid` | `3.918462182820283` | `1.08447145588637` | `0` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `GatedHybrid` | `1.8815526089850572` | `0.4834232450150232` | `0` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `GatedHybrid` | `7.606903954557001` | `1.1207661841027041` | `0` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `GatedHybrid` | `2.534557556200822` | `0.4846353455340071` | `0` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `GatedHybrid` | `6.702003807282071` | `1.1276461349358098` | `0` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `GatedHybrid` | `1.3721196174710515` | `0.49127629063097517` | `1` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `GatedHybrid` | `4.910717788432166` | `1.1290972411909315` | `0` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `GatedHybrid` | `2.5757706209021083` | `0.4916518710734772` | `0` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `GatedHybrid` | `6.507286370959187` | `1.1294728216334335` | `0` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `GatedHybrid` | `1.7531802381480661` | `0.49202745151597926` | `0` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `GatedHybrid` | `5.121056858413483` | `1.1298484020759356` | `0` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `GatedHybrid` | `1.7224826196287257` | `0.4924030319584813` | `0` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `GatedHybrid` | `6.349906787509286` | `1.1302239825184375` | `0` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `GatedHybrid` | `1.662602135808067` | `0.49277861240098336` | `0` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `2.9096815581559126` | `1.1482689155968315` | `0` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `4.421724820970922` | `1.1215173449877083` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `2.7167954116462956` | `1.0949194209232451` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `2.3430581414418152` | `1.179646954384048` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.3138833151651377` | `0.5175327779295275` | `1` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.9529270892621147` | `1.3226748156241463` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.9638657900914203` | `0.5158426659382682` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `3.071226562223659` | `1.1380770281343895` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `1.4147515700215125` | `0.5531275607757443` | `1` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `2.630274203091487` | `1.0885516252390057` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `1.9475978188518344` | `0.4575252663206774` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `3.312504823414785` | `0.921571974870254` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `2.3370493724737367` | `0.31719475553127563` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `2.244145580305033` | `0.8694687243922425` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `2.1360486344912744` | `0.256453154875717` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `2.736874885443899` | `0.8629472821633434` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `2.200993100909053` | `0.2566238732586725` | `0` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `2.315714604978407` | `0.7750956022944551` | `0` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `2.731886268702791` | `0.1630019120458891` | `0` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `3.0170986034606395` | `0.7762735591368478` | `0` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `2.059649964789072` | `0.17775198033324227` | `0` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.3483962547792` | `0.7985693799508331` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.1680704813446394` | `0.20725211690794865` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.92100392143622` | `0.8273183556405354` | `0` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.9416400976902275` | `0.2662523900573614` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `2.0204080689298127` | `0.8697589456432668` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.3759582517372333` | `0.258416416279705` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.3543037320367994` | `0.9605811253755805` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.4984723843425525` | `0.38228967495219884` | `1` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `3.306731999819926` | `1.099904397705545` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10c-QuadraticSketch-h256-diagnostic', 'B11a-TrainableQuadraticSketch-h128', 'B11b-TrainableQuadraticSketch-h256', 'B12b-ChebyKAN-K4-fan-scale-repair', 'B13b-ChebyKAN-K4-identity-residual-repair', 'B14a-GatedLegendreQuadratic-h160', 'B14d-GatedLegendreQuadratic-h228-quad-boost', 'B18a-GatedLegendreQuadratic-h224-group-rms-norm', 'B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm', 'B1r-ReLU-KAN-stream-K2-repair', 'B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15', 'B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm', 'B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm', 'B2b-FastKAN-RBF-K8', 'B6a-BSpline-order1-local-K4', 'B8d-BSpline-ReLU-lite-combo-K4-repair']
A2 expression pass = ['B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm', 'B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `-0.0015190972222222222` | `-0.0078125` | `0.006518384648693932` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `-0.009114583333333334` | `-0.01953125` | `0.027369055483076308` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B10c-QuadraticSketch-h256-diagnostic` | `diagnostic_base_not_qualified` | `-0.0019781291484832764` | `0.0` | `-0.0019781291484832764` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `diagnostic_base_not_qualified` | `-0.00015688857346418672` | `0.00289104945078833` | `-0.0030479380242525167` | `0` |
| `B11b-TrainableQuadraticSketch-h256` | `diagnostic_base_not_qualified` | `-0.00013237509664576486` | `6.053881910617065e-05` | `-0.0001929139157519355` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `diagnostic_base_not_qualified` | `0.010721973229378179` | `0.059079261395131866` | `-0.04835728816575369` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `diagnostic_base_not_qualified` | `0.0009064327380206016` | `0.00924675009415954` | `-0.008340317356138938` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `diagnostic_base_not_qualified` | `-0.0030250690793711676` | `0.00018007950541143458` | `-0.0032051485847826022` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `diagnostic_base_not_qualified` | `-7.895683709024937e-05` | `0.001226452343282336` | `-0.0013054091803725854` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `diagnostic_base_not_qualified` | `-0.002098617777624767` | `6.056698130052496e-05` | `-0.002159184758925292` | `0` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `diagnostic_base_not_qualified` | `0.003768859952016257` | `0.0054622161295045935` | `-0.0016933561774883366` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `diagnostic_base_not_qualified` | `-0.002065216998578867` | `0.0` | `-0.002065216998578867` | `0` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `diagnostic_base_not_qualified` | `-7.82757502371112e-05` | `0.001178393978362724` | `-0.0012566697285998352` | `0` |
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `diagnostic_base_not_qualified` | `-0.00011866015135186814` | `0.00691744941561101` | `-0.007036109566962878` | `0` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `diagnostic_base_not_qualified` | `-0.00010235719731266357` | `0.0027740255847445994` | `-0.002876382782057263` | `0` |
| `B2b-FastKAN-RBF-K8` | `diagnostic_base_not_qualified` | `-0.013029543715944447` | `0.0` | `-0.013029543715944447` | `0` |
| `B6a-BSpline-order1-local-K4` | `diagnostic_base_not_qualified` | `0.002269369158875989` | `0.006272526008597623` | `-0.004003156849721634` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `diagnostic_base_not_qualified` | `0.0014408970357095985` | `0.0037932873636172815` | `-0.002352390327907683` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 1976
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `1475352572ad5b66122b5eb37a84e5c24362c95f5019fc176df0178be088bbde` |
| `v124_control_matrix.csv` | `cd1f60dcc9057b1872542267fe94c76f0efc28f12a11784bcaf3c34340dd04e7` |
| `v124_efficiency_microbench.csv` | `861f6a1528ce108c207997274f9c4edc8aa4c68173c1190e3604238946870f6a` |
| `v124_expression_battery.csv` | `4e5aebc4f7ee1ff447aac24c740029256fbe49c9b3ec42d80ab31bc0b850be2c` |
| `v124_provenance_audit.csv` | `1d189287200199ede4fc9cffb7340f25fdef315e31461c4769a81c262a9f72de` |
| `v124_route_decision.json` | `887faf07bcfd095306ac66afe9200ae253b179a8eb496dea0002bb14eab687f6` |
| `v124_task_triage.csv` | `8d7add6380e1e87105729961c90b01f08d97c677ce7685e52c5217d4df71b7ba` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
