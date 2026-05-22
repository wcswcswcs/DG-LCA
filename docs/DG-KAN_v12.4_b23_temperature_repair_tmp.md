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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b23_temperature_repair_20260520T180000Z
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
    "B10c-QuadraticSketch-h256-diagnostic",
    "B13a-LegendreKAN-K4-identity-residual-repair",
    "B13b-ChebyKAN-K4-identity-residual-repair",
    "B14a-GatedLegendreQuadratic-h160",
    "B14b-GatedLegendreQuadratic-h224",
    "B14c-GatedLegendreQuadratic-h224-quad-boost",
    "B14d-GatedLegendreQuadratic-h228-quad-boost",
    "B15a-GatedLegendreQuadratic-h224-loss-gain",
    "B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain",
    "B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm",
    "B17a-GatedLegendreQuadratic-h224-input-geom-norm",
    "B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm",
    "B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm",
    "B18a-GatedLegendreQuadratic-h224-group-rms-norm",
    "B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B20b-GatedLegendreQuadratic-h224-residual-mix25",
    "B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25",
    "B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm",
    "B2b-FastKAN-RBF-K8"
  ],
  "A2_expression_pass": [
    "B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm"
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.4210945928108032` | `1.017242556678503` | `1` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `0.9714962260444222` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `1.8098891220258857` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `2.867143972712869` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `2.588431406584215` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `2.092056162276709` | `1.1750887735591369` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.4128447659102223` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `0.8624941461796521` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `1.8937407934611876` | `1.0923415733406172` | `0` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `1.630924819705673` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `2.409505260026512` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `2.151737869744576` | `0.5075628243649276` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `3.0189063055611984` | `1.1272193389784213` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `2.0003169349551118` | `0.5418430756623873` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `2.7289907233071844` | `1.1451959847036328` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `0.9859187703057328` | `0.5075628243649276` | `1` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `2.4599117775316994` | `1.0854616225075115` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.0373734441403395` | `0.505736137667304` | `1` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `3.222208640885895` | `1.0465890467085497` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `1.3298770223910872` | `0.3693321496858782` | `1` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `4.488756615578856` | `1.078837749248839` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `1.1130818242395293` | `0.4682634526085769` | `1` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `4.610120244086672` | `1.0960973777656378` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `1.1647832998990333` | `0.4686219612127834` | `1` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `3.2804529531516535` | `1.100348265501229` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `1.3850480007006634` | `0.4752458344714559` | `1` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `3.528369560852221` | `1.0978899207866704` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `1.3648529153799533` | `0.4693389784211964` | `1` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `5.353238869016445` | `1.101065282709642` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `1.4279216355780695` | `0.4759628516798689` | `1` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `3.9219828205557867` | `1.0745697896749522` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `2.1381481564045615` | `0.47005599562960937` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `6.4249621009748425` | `1.0777451515979242` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `1.1089184210594427` | `0.476714012564873` | `1` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `4.49148496132784` | `1.087561458617864` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `1.0067652588192202` | `0.47794318492215243` | `1` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `4.505927042675426` | `1.0944584812892653` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `2.5303102447554404` | `0.4845670581808249` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `5.692062875019172` | `1.0918635618683419` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `1.2287038871537896` | `0.4786602021305654` | `1` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `4.114880989504996` | `1.0951754984976783` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `1.4489172404850843` | `0.485318219065829` | `1` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `4.506747433315291` | `1.1137838022398252` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `1.4793869128740298` | `0.47234362196121277` | `1` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `4.221435005666173` | `1.1135789401802787` | `0` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `2.587926782013958` | `0.4789674952198853` | `0` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `4.895052783947127` | `1.1109840207593553` | `0` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `1.1829555453639873` | `0.47306063916962576` | `1` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `3.2073624591213603` | `1.1142959573886917` | `0` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `2.5520235453616493` | `0.4797186561048894` | `0` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `GatedHybrid` | `7.791988391965048` | `1.1170615951925704` | `0` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `GatedHybrid` | `2.3403761189678307` | `0.4809478284621688` | `0` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `GatedHybrid` | `5.441612874973251` | `1.1239586178639716` | `0` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `GatedHybrid` | `2.474039929629753` | `0.4875717017208413` | `0` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `GatedHybrid` | `5.218318506438179` | `1.1213636984430484` | `0` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `GatedHybrid` | `2.590852752904868` | `0.4816648456705818` | `0` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `GatedHybrid` | `6.735371669271195` | `1.1246756350723845` | `0` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `GatedHybrid` | `2.1191129394706834` | `0.4883228626058454` | `0` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `GatedHybrid` | `7.627774172322612` | `1.1221148593280525` | `0` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `GatedHybrid` | `2.3047705306598254` | `0.482450150232177` | `0` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `GatedHybrid` | `4.5772385764611005` | `1.1214319857962305` | `0` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `GatedHybrid` | `1.3435483072408867` | `0.4828086588363835` | `1` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `GatedHybrid` | `4.2297295841831755` | `1.1258194482381862` | `0` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `GatedHybrid` | `1.5366999573607267` | `0.489432532095056` | `0` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `GatedHybrid` | `4.08160390461193` | `1.1272534826550122` | `0` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `GatedHybrid` | `1.1141614569857232` | `0.4897910406992625` | `1` |
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `GatedHybrid` | `3.249557221996656` | `1.0848470363288718` | `0` |
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `GatedHybrid` | `1.6231119889997856` | `0.47678229991805515` | `0` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `GatedHybrid` | `4.042550855494777` | `1.08447145588637` | `0` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `GatedHybrid` | `1.3184085857645944` | `0.4834232450150232` | `1` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `GatedHybrid` | `5.434490772108155` | `1.1207661841027041` | `0` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `GatedHybrid` | `2.689253458711332` | `0.4846353455340071` | `0` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `GatedHybrid` | `8.31776528127852` | `1.1276461349358098` | `0` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `GatedHybrid` | `2.256288750119915` | `0.49127629063097517` | `0` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `GatedHybrid` | `4.118415281808693` | `1.1290972411909315` | `0` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `GatedHybrid` | `2.4829354822792364` | `0.4916518710734772` | `0` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `GatedHybrid` | `7.5804849054718115` | `1.1294728216334335` | `0` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `GatedHybrid` | `2.0606983873805116` | `0.49202745151597926` | `0` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `GatedHybrid` | `7.4382992515542625` | `1.1298484020759356` | `0` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `GatedHybrid` | `2.0277726360730925` | `0.4924030319584813` | `0` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `GatedHybrid` | `5.955971005294133` | `1.1302239825184375` | `0` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `GatedHybrid` | `2.066922034752469` | `0.49277861240098336` | `0` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `3.9469998925460255` | `1.1482689155968315` | `0` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `6.777975458580918` | `1.1215173449877083` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `3.8659517094992237` | `1.0949194209232451` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `3.740677157080094` | `1.179646954384048` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.9280511731394119` | `0.5175327779295275` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `3.3551683650155573` | `1.3226748156241463` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `2.513128963722052` | `0.5158426659382682` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `3.7670910472471033` | `1.1380770281343895` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `2.212151216045859` | `0.5531275607757443` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `4.739392551911666` | `1.0885516252390057` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `1.9085481515286393` | `0.4575252663206774` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `4.443466476906455` | `0.921571974870254` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `2.262832839103495` | `0.31719475553127563` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `3.192161687590135` | `0.8694687243922425` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.836781336839322` | `0.256453154875717` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `2.696681976562009` | `0.8629472821633434` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.873842187599277` | `0.2566238732586725` | `0` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `2.184987986779237` | `0.7750956022944551` | `0` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `2.129257947231666` | `0.1630019120458891` | `0` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.8353297480275266` | `0.7762735591368478` | `0` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.9729816290439994` | `0.17775198033324227` | `0` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.2289795771489898` | `0.7985693799508331` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `2.10169687943472` | `0.20725211690794865` | `0` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.7506918341074278` | `0.8273183556405354` | `0` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.830808498899703` | `0.2662523900573614` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `2.1890385925082287` | `0.8697589456432668` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `2.0022681388278634` | `0.258416416279705` | `0` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `2.584301483624624` | `0.9605811253755805` | `0` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `2.1966727507157127` | `0.38228967495219884` | `0` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `3.2846719618849813` | `1.099904397705545` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10c-QuadraticSketch-h256-diagnostic', 'B13a-LegendreKAN-K4-identity-residual-repair', 'B13b-ChebyKAN-K4-identity-residual-repair', 'B14a-GatedLegendreQuadratic-h160', 'B14b-GatedLegendreQuadratic-h224', 'B14c-GatedLegendreQuadratic-h224-quad-boost', 'B14d-GatedLegendreQuadratic-h228-quad-boost', 'B15a-GatedLegendreQuadratic-h224-loss-gain', 'B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain', 'B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm', 'B17a-GatedLegendreQuadratic-h224-input-geom-norm', 'B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm', 'B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm', 'B18a-GatedLegendreQuadratic-h224-group-rms-norm', 'B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm', 'B1r-ReLU-KAN-stream-K2-repair', 'B20b-GatedLegendreQuadratic-h224-residual-mix25', 'B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25', 'B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm', 'B2b-FastKAN-RBF-K8']
A2 expression pass = ['B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `-0.005425347222222222` | `-0.029296875` | `0.023337364610698488` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B10c-QuadraticSketch-h256-diagnostic` | `diagnostic_base_not_qualified` | `-0.0019781291484832764` | `0.0` | `-0.0019781291484832764` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `diagnostic_base_not_qualified` | `0.01466868295428947` | `0.015135731875359326` | `-0.00046704892106985696` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `diagnostic_base_not_qualified` | `0.0009064327380206016` | `0.00924675009415954` | `-0.008340317356138938` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `diagnostic_base_not_qualified` | `-0.0030250690793711676` | `0.00018007950541143458` | `-0.0032051485847826022` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `diagnostic_base_not_qualified` | `-9.739293402599714e-05` | `0.0009702653393306448` | `-0.001067658273356642` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `diagnostic_base_not_qualified` | `-7.763039190411547e-05` | `0.0` | `-7.763039190411547e-05` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `diagnostic_base_not_qualified` | `-7.895683709024937e-05` | `0.001226452343282336` | `-0.0013054091803725854` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `diagnostic_base_not_qualified` | `-0.00020563623901370676` | `0.0006638135429084535` | `-0.0008694497819221603` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `diagnostic_base_not_qualified` | `-8.342782527348547e-05` | `0.0008029128400779406` | `-0.0008863406653514261` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `diagnostic_base_not_qualified` | `-0.00013990018542164862` | `0.0053534928810177185` | `-0.005493393066439367` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `diagnostic_base_not_qualified` | `-8.645000800155955e-05` | `0.0` | `-8.645000800155955e-05` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `diagnostic_base_not_qualified` | `-0.00012400512670041053` | `0.0014595131279211415` | `-0.001583518254621552` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `diagnostic_base_not_qualified` | `-0.00023029879443514645` | `0.0029624343609584436` | `-0.00319273315539359` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `diagnostic_base_not_qualified` | `-0.002098617777624767` | `6.056698130052496e-05` | `-0.002159184758925292` | `0` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `diagnostic_base_not_qualified` | `-0.002182152770044077` | `0.005223042765916119` | `-0.007405195535960196` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `diagnostic_base_not_qualified` | `-0.002065216998578867` | `0.0` | `-0.002065216998578867` | `0` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `diagnostic_base_not_qualified` | `-0.002067623863291068` | `0.0` | `-0.002067623863291068` | `0` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `diagnostic_base_not_qualified` | `-6.60671930798884e-05` | `2.7757184865917495e-05` | `-9.382437794580589e-05` | `0` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `diagnostic_base_not_qualified` | `0.0010797791264733902` | `0.0002384023627388654` | `0.0008413767637345249` | `1` |
| `B2b-FastKAN-RBF-K8` | `diagnostic_base_not_qualified` | `-0.013029543715944447` | `0.0` | `-0.013029543715944447` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 2166
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `1475352572ad5b66122b5eb37a84e5c24362c95f5019fc176df0178be088bbde` |
| `v124_control_matrix.csv` | `1cdf36f47afc6dcea36bd620213a9fe5eceab93d37ac124889f96f2c70656d68` |
| `v124_efficiency_microbench.csv` | `a42155ca2301eaffdd42ac3bf556832ad8cbbee6f114ad59340ee496fb33ae8d` |
| `v124_expression_battery.csv` | `257c703983cae10a1d3bdad137b034c12a3c3d9a88a24bd3dade10069cec5f8b` |
| `v124_provenance_audit.csv` | `b84a9853ca6979ac013c26980be318a0bd32bfa7aeffdc97e3b53399780775d4` |
| `v124_route_decision.json` | `fb832c6b82c3cf8150bd8a252add89a72a51a54e6054b820cf1eb9d86309ed94` |
| `v124_task_triage.csv` | `23ca60ed0a1d7e6ffc95236d1d3d6623abc35b164563576cfc7b856ba41e385f` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
