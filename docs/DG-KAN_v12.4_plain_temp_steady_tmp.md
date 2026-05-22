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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_plain_temp_steady_20260520T210000Z
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
    "B13b-ChebyKAN-K4-identity-residual-repair",
    "B14b-GatedLegendreQuadratic-h224",
    "B14d-GatedLegendreQuadratic-h228-quad-boost",
    "B16a-GatedLegendreQuadratic-h224-basis-norm",
    "B17a-GatedLegendreQuadratic-h224-input-geom-norm",
    "B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm",
    "B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm",
    "B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm",
    "B18a-GatedLegendreQuadratic-h224-group-rms-norm",
    "B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm",
    "B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm",
    "B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B20a-GatedLegendreQuadratic-h224-residual-mix15",
    "B20b-GatedLegendreQuadratic-h224-residual-mix25",
    "B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15",
    "B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25",
    "B2b-FastKAN-RBF-K8",
    "B9b-Poly2SignedPair-h64-diagnostic-repair",
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.7162636502536253` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.1391034294671714` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `1.9664188808370573` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `3.2089659385698392` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `2.6869160318747904` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `1.9314164876121247` | `1.1750887735591369` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.930407551295232` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `0.8983340055688087` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `2.0641339299652257` | `1.0923415733406172` | `0` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `2.181828950308054` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `2.5692546649560066` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `0.9749796788297138` | `0.5075628243649276` | `1` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.8936831690063298` | `1.1272193389784213` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.1644449575758606` | `0.5418430756623873` | `1` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `2.314906584259094` | `1.1451959847036328` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.6106053934443467` | `0.5075628243649276` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.9925322718843472` | `1.0854616225075115` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `0.9908479346710403` | `0.505736137667304` | `1` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `3.423236956257713` | `1.0465890467085497` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `1.9100677216848503` | `0.3693321496858782` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `3.979994872017861` | `1.078837749248839` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `1.2005760733151505` | `0.4682634526085769` | `1` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `3.31344016129653` | `1.0960973777656378` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `2.1240673461659503` | `0.4686219612127834` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `5.425765961094309` | `1.100348265501229` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `1.3132397401622464` | `0.4752458344714559` | `1` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `2.9620246099826386` | `1.0978899207866704` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `1.8381651820008826` | `0.4693389784211964` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `4.9118120352974435` | `1.101065282709642` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `2.0779461615187755` | `0.4759628516798689` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `5.057328941061693` | `1.0745697896749522` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `1.1150635558241166` | `0.47005599562960937` | `1` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `3.704535765176403` | `1.0777451515979242` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `2.3817244580924077` | `0.476714012564873` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `4.1018582521111275` | `1.087561458617864` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `1.1367706807371298` | `0.47794318492215243` | `1` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `5.1024248624832795` | `1.0944584812892653` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `1.1677236674035434` | `0.4845670581808249` | `1` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `4.609341890632702` | `1.0918635618683419` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `1.1134250339158207` | `0.4786602021305654` | `1` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `4.125830485703876` | `1.0951754984976783` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `1.49914475320628` | `0.485318219065829` | `1` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `4.891202811693598` | `1.1137838022398252` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `1.4315052027027364` | `0.47234362196121277` | `1` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `3.326645631610628` | `1.1135789401802787` | `0` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `0.9727350620864024` | `0.4789674952198853` | `1` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `3.7750936673085813` | `1.1109840207593553` | `0` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `1.8526853016671274` | `0.47306063916962576` | `0` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `4.421344534259102` | `1.1142959573886917` | `0` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `2.0826718019792545` | `0.4797186561048894` | `0` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `GatedHybrid` | `4.915955404881513` | `1.1170615951925704` | `0` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `GatedHybrid` | `2.0664130331381267` | `0.4809478284621688` | `0` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `GatedHybrid` | `6.793168221638119` | `1.1239586178639716` | `0` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `GatedHybrid` | `1.2682899279133661` | `0.4875717017208413` | `1` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `GatedHybrid` | `3.324534628956839` | `1.1213636984430484` | `0` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `GatedHybrid` | `1.0218825558809277` | `0.4816648456705818` | `1` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `GatedHybrid` | `4.058925646091597` | `1.1246756350723845` | `0` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `GatedHybrid` | `1.886919497177861` | `0.4883228626058454` | `0` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `GatedHybrid` | `4.2569189024119005` | `1.1221148593280525` | `0` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `GatedHybrid` | `1.1718450456412068` | `0.482450150232177` | `1` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `GatedHybrid` | `4.936781043326118` | `1.1214319857962305` | `0` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `GatedHybrid` | `1.3637310385030157` | `0.4828086588363835` | `1` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `GatedHybrid` | `4.645014353518557` | `1.1258194482381862` | `0` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `GatedHybrid` | `1.4413506618478962` | `0.489432532095056` | `1` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `GatedHybrid` | `4.6098621858965165` | `1.1272534826550122` | `0` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `GatedHybrid` | `1.281351954672503` | `0.4897910406992625` | `1` |
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `GatedHybrid` | `3.647107529853669` | `1.0848470363288718` | `0` |
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `GatedHybrid` | `2.0297556244950945` | `0.47678229991805515` | `0` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `GatedHybrid` | `5.139667874024904` | `1.08447145588637` | `0` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `GatedHybrid` | `2.0231348964182594` | `0.4834232450150232` | `0` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `GatedHybrid` | `6.775970322344824` | `1.1207661841027041` | `0` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `GatedHybrid` | `2.07357238260078` | `0.4846353455340071` | `0` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `GatedHybrid` | `6.707005793220368` | `1.1276461349358098` | `0` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `GatedHybrid` | `1.9149971153017955` | `0.49127629063097517` | `0` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `GatedHybrid` | `5.113424200910275` | `1.1290972411909315` | `0` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `GatedHybrid` | `2.087782291001824` | `0.4916518710734772` | `0` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `GatedHybrid` | `6.172397494919083` | `1.1294728216334335` | `0` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `GatedHybrid` | `2.14932640672562` | `0.49202745151597926` | `0` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `GatedHybrid` | `7.15857026926571` | `1.1298484020759356` | `0` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `GatedHybrid` | `2.089243049526678` | `0.4924030319584813` | `0` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `GatedHybrid` | `6.723873988918694` | `1.1302239825184375` | `0` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `GatedHybrid` | `2.0654034304167976` | `0.49277861240098336` | `0` |
| `B26a-GatedLegendreQuadratic-h224-basis-specific-temp075` | `GatedHybrid` | `3.8752594187520546` | `1.0878346080305927` | `0` |
| `B26a-GatedLegendreQuadratic-h224-basis-specific-temp075` | `GatedHybrid` | `2.0069039499623567` | `0.47978694345807155` | `0` |
| `B26b-GatedLegendreQuadratic-h224-basis-specific-temp050` | `GatedHybrid` | `5.616225348658636` | `1.0835837202950014` | `0` |
| `B26b-GatedLegendreQuadratic-h224-basis-specific-temp050` | `GatedHybrid` | `1.9945619731968804` | `0.4801625239005736` | `0` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `3.389208346515646` | `1.144393608303742` | `0` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `5.550222820653406` | `1.117642037694619` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `3.5678143983675756` | `1.0910441136301556` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `2.8579325702002927` | `1.1757716470909587` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.6674981728023357` | `0.5182839388145315` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `2.884328350127558` | `1.3234259765091505` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.663651686394737` | `0.5165938268232724` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `3.004660832629027` | `1.139681780934171` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `1.6992220228006947` | `0.5547323135755258` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `3.295409589659802` | `1.0892003550942366` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `1.686325597544033` | `0.48221114449603936` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `3.226520697438891` | `0.9223231357552581` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `1.157946181510909` | `0.31794591641627973` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.8375046086031812` | `0.8702198852772467` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.6529882158037479` | `0.2572043157607211` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `2.0320530537900003` | `0.8636984430483474` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `0.9086266220908739` | `0.2573750341436766` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `0.8769934030958818` | `0.7758467631794591` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `0.8153653204130841` | `0.1637530729308932` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.0739708924538194` | `0.777024720021852` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `0.8205232907517559` | `0.17850314121824637` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.0431070373807907` | `0.7993205408358373` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `0.956272040285481` | `0.20800327779295275` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.1952478366429478` | `0.8280695165255395` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `0.9694871734639087` | `0.2670035509423655` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.2637528799084228` | `0.870510106528271` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `0.9044900831522795` | `0.2591675771647091` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.29677955058018` | `0.9613322862605845` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `0.976755646653042` | `0.38304083583720294` | `1` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `2.172437145983056` | `1.100655558590549` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10a-QuadraticSketch-h64-diagnostic', 'B10b-QuadraticSketch-h128-diagnostic', 'B10c-QuadraticSketch-h256-diagnostic', 'B10d-QuadraticSketch-h512-diagnostic', 'B11a-TrainableQuadraticSketch-h128', 'B11b-TrainableQuadraticSketch-h256', 'B12a-LegendreKAN-K4-fan-scale-repair', 'B12b-ChebyKAN-K4-fan-scale-repair', 'B13b-ChebyKAN-K4-identity-residual-repair', 'B14b-GatedLegendreQuadratic-h224', 'B14d-GatedLegendreQuadratic-h228-quad-boost', 'B16a-GatedLegendreQuadratic-h224-basis-norm', 'B17a-GatedLegendreQuadratic-h224-input-geom-norm', 'B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm', 'B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm', 'B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm', 'B18a-GatedLegendreQuadratic-h224-group-rms-norm', 'B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm', 'B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm', 'B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm', 'B1r-ReLU-KAN-stream-K2-repair', 'B20a-GatedLegendreQuadratic-h224-residual-mix15', 'B20b-GatedLegendreQuadratic-h224-residual-mix25', 'B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15', 'B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25', 'B2b-FastKAN-RBF-K8', 'B9b-Poly2SignedPair-h64-diagnostic-repair', 'B9d-Poly2PairRandom-h64-K2-diagnostic-repair']
A2 expression pass = []
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B10a-QuadraticSketch-h64-diagnostic` | `-0.2810329861111111` | `-0.376953125` | `0.18786867583791414` |
| `B10b-QuadraticSketch-h128-diagnostic` | `-0.1560329861111111` | `-0.23046875` | `0.16646509907311863` |
| `B10c-QuadraticSketch-h256-diagnostic` | `-0.06966145833333333` | `-0.130859375` | `0.10470389285021359` |

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
| `B13b-ChebyKAN-K4-identity-residual-repair` | `diagnostic_base_not_qualified` | `0.0009064327380206016` | `0.00924675009415954` | `-0.008340317356138938` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `diagnostic_base_not_qualified` | `-9.739293402599714e-05` | `0.0009702653393306448` | `-0.001067658273356642` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `diagnostic_base_not_qualified` | `-7.895683709024937e-05` | `0.001226452343282336` | `-0.0013054091803725854` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `diagnostic_base_not_qualified` | `-0.00013499184002574438` | `0.00449900838266748` | `-0.004634000222693224` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `diagnostic_base_not_qualified` | `-8.645000800155955e-05` | `0.0` | `-8.645000800155955e-05` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `diagnostic_base_not_qualified` | `-7.95033345211138e-05` | `0.00042811441086065827` | `-0.0005076177453817721` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `diagnostic_base_not_qualified` | `-0.00012400512670041053` | `0.0014595131279211415` | `-0.001583518254621552` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `diagnostic_base_not_qualified` | `-0.00023029879443514645` | `0.0029624343609584436` | `-0.00319273315539359` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `diagnostic_base_not_qualified` | `-0.002098617777624767` | `6.056698130052496e-05` | `-0.002159184758925292` | `0` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `diagnostic_base_not_qualified` | `-0.00010431500322383158` | `0.0` | `-0.00010431500322383158` | `0` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `diagnostic_base_not_qualified` | `-7.177283231518672e-05` | `0.0010859284882380749` | `-0.0011577013205532616` | `0` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `diagnostic_base_not_qualified` | `0.003768859952016257` | `0.0054622161295045935` | `-0.0016933561774883366` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `diagnostic_base_not_qualified` | `-0.002065216998578867` | `0.0` | `-0.002065216998578867` | `0` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `diagnostic_base_not_qualified` | `-0.002052105891147349` | `0.00018282349310272394` | `-0.002234929384250073` | `0` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `diagnostic_base_not_qualified` | `-0.002067623863291068` | `0.0` | `-0.002067623863291068` | `0` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `diagnostic_base_not_qualified` | `-7.82757502371112e-05` | `0.001178393978362724` | `-0.0012566697285998352` | `0` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `diagnostic_base_not_qualified` | `-6.60671930798884e-05` | `2.7757184865917495e-05` | `-9.382437794580589e-05` | `0` |
| `B2b-FastKAN-RBF-K8` | `diagnostic_base_not_qualified` | `-0.013029543715944447` | `0.0` | `-0.013029543715944447` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `diagnostic_base_not_qualified` | `4.14654969960182e-05` | `1.1681385773343322e-05` | `2.978411122267488e-05` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `5.402795305364805e-06` | `0.0` | `5.402795305364805e-06` | `1` |

## 6. No-Fake / Hash

```text
rows_checked = 3026
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `0c50c6db9b226a2b2799ce4b220e950e554ea31d13e386f8cfa2c7689c0ac1b3` |
| `v124_control_matrix.csv` | `116716c7437d1fc8ac5755e5e59c983d60d7851f1ac06047d11ab58a89c8d818` |
| `v124_efficiency_microbench.csv` | `65caa665c1b9f5b482acbc85133cfc9e049e099036fe9585147e2d70055d6f16` |
| `v124_expression_battery.csv` | `d0453f02056fb279f6a4e3b4c7b990bda5382cc6608f96ca2df5b7dfd0098dc0` |
| `v124_provenance_audit.csv` | `20a4cf4e2dd8cdf409039c4f1a0d59829c56a9188579f2af411b0bf5098be196` |
| `v124_route_decision.json` | `27699bd36592041f942431613b8f3ef83bad67d5c431c23c24fa373c02fba080` |
| `v124_task_triage.csv` | `c7ac7c82ced882c20aff4b8581eed49d5e2b30cb7a9e454c2666a3571acea79a` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
