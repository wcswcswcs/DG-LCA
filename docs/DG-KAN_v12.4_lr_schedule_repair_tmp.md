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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_lr_schedule_repair_20260520T220000Z
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
    "B16a-GatedLegendreQuadratic-h224-basis-norm",
    "B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm",
    "B17a-GatedLegendreQuadratic-h224-input-geom-norm",
    "B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm",
    "B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm",
    "B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm",
    "B18a-GatedLegendreQuadratic-h224-group-rms-norm",
    "B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm",
    "B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm",
    "B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm",
    "B19a-GatedLegendreQuadratic-h224-residual-geom-norm",
    "B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm",
    "B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm",
    "B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B20a-GatedLegendreQuadratic-h224-residual-mix15",
    "B20b-GatedLegendreQuadratic-h224-residual-mix25",
    "B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15",
    "B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25",
    "B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm",
    "B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm",
    "B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm",
    "B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm",
    "B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost",
    "B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost",
    "B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075",
    "B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050",
    "B26a-GatedLegendreQuadratic-h224-basis-specific-temp075",
    "B26b-GatedLegendreQuadratic-h224-basis-specific-temp050",
    "B2a-GaussianRBF-K4-compact",
    "B2b-FastKAN-RBF-K8",
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
    "B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm",
    "B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm",
    "B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm",
    "B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm",
    "B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost",
    "B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost",
    "B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075",
    "B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050",
    "B26a-GatedLegendreQuadratic-h224-basis-specific-temp075",
    "B26b-GatedLegendreQuadratic-h224-basis-specific-temp050"
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.7830733791706255` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.2044836544292254` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `1.8681157690954893` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `2.6585181465639476` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `1.5272412604590166` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `1.269919558715269` | `1.1750887735591369` | `1` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.7080107747365743` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `0.7798704973643489` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `1.6560453274157267` | `1.0923415733406172` | `0` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `1.62107657499344` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.76095728617601` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `0.7871931154195396` | `0.5075628243649276` | `1` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.811560483817872` | `1.1272193389784213` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `0.6772820329244955` | `0.5418430756623873` | `1` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.2656376014034363` | `1.1451959847036328` | `1` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `0.7480081086096487` | `0.5075628243649276` | `1` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.8349911782964239` | `1.0854616225075115` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.0224648343424299` | `0.505736137667304` | `1` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `3.2776059863508964` | `1.0465890467085497` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `1.4653743553070246` | `0.3693321496858782` | `1` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `4.126815060250534` | `1.078837749248839` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `1.263247182410187` | `0.4682634526085769` | `1` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `3.280745472980284` | `1.0960973777656378` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `0.9094406006876272` | `0.4686219612127834` | `1` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `3.007925890359467` | `1.100348265501229` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `1.4589870536731804` | `0.4752458344714559` | `1` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `3.1215935782722064` | `1.0978899207866704` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `1.2209767174086006` | `0.4693389784211964` | `1` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `3.128660036372813` | `1.101065282709642` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `1.1279159914808832` | `0.4759628516798689` | `1` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `3.1534999499761835` | `1.0745697896749522` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `0.7982818413814664` | `0.47005599562960937` | `1` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `2.875802060981273` | `1.0777451515979242` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `1.4135221233556112` | `0.476714012564873` | `1` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `4.294713317371867` | `1.087561458617864` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `1.4352081608461884` | `0.47794318492215243` | `1` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `3.8986001611893646` | `1.0944584812892653` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `1.380042275216246` | `0.4845670581808249` | `1` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `3.9958119815798323` | `1.0918635618683419` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `1.046241012172553` | `0.4786602021305654` | `1` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `3.6386671754570807` | `1.0951754984976783` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `1.2174264517529445` | `0.485318219065829` | `1` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `2.96569832230841` | `1.1137838022398252` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `0.9928747622171805` | `0.47234362196121277` | `1` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `3.7520839976591303` | `1.1135789401802787` | `0` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `0.9088457448736448` | `0.4789674952198853` | `1` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `3.7843583871886906` | `1.1109840207593553` | `0` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `0.9199500820675679` | `0.47306063916962576` | `1` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `3.5187058533024755` | `1.1142959573886917` | `0` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `0.9215080830208032` | `0.4797186561048894` | `1` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `GatedHybrid` | `4.261860226259694` | `1.1170615951925704` | `0` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `GatedHybrid` | `0.9330918886617794` | `0.4809478284621688` | `1` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `GatedHybrid` | `3.8204052648676297` | `1.1239586178639716` | `0` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `GatedHybrid` | `0.9335106769288118` | `0.4875717017208413` | `1` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `GatedHybrid` | `4.226944959275726` | `1.1213636984430484` | `0` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `GatedHybrid` | `0.9483146046061822` | `0.4816648456705818` | `1` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `GatedHybrid` | `3.6715745665066324` | `1.1246756350723845` | `0` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `GatedHybrid` | `0.980064999740039` | `0.4883228626058454` | `1` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `GatedHybrid` | `3.501969526603644` | `1.1221148593280525` | `0` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `GatedHybrid` | `1.4653709615609385` | `0.482450150232177` | `1` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `GatedHybrid` | `4.595578410399198` | `1.1214319857962305` | `0` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `GatedHybrid` | `0.9512485659725913` | `0.4828086588363835` | `1` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `GatedHybrid` | `4.674654458954746` | `1.1258194482381862` | `0` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `GatedHybrid` | `0.9451007270625865` | `0.489432532095056` | `1` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `GatedHybrid` | `3.564881026800956` | `1.1272534826550122` | `0` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `GatedHybrid` | `0.9135730974218254` | `0.4897910406992625` | `1` |
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `GatedHybrid` | `3.086962028461042` | `1.0848470363288718` | `0` |
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `GatedHybrid` | `1.2375201435798944` | `0.47678229991805515` | `1` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `GatedHybrid` | `3.949049961508132` | `1.08447145588637` | `0` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `GatedHybrid` | `1.183096673843761` | `0.4834232450150232` | `1` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `GatedHybrid` | `4.04299767691293` | `1.1207661841027041` | `0` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `GatedHybrid` | `0.8885290160539122` | `0.4846353455340071` | `1` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `GatedHybrid` | `3.84729418660798` | `1.1276461349358098` | `0` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `GatedHybrid` | `1.0315781285825232` | `0.49127629063097517` | `1` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `GatedHybrid` | `4.557079754797773` | `1.1290972411909315` | `0` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `GatedHybrid` | `0.8600029512015965` | `0.4916518710734772` | `1` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `GatedHybrid` | `4.078807671115054` | `1.1294728216334335` | `0` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `GatedHybrid` | `0.8026880098000527` | `0.49202745151597926` | `1` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `GatedHybrid` | `3.547989809530753` | `1.1298484020759356` | `0` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `GatedHybrid` | `1.1619621879671067` | `0.4924030319584813` | `1` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `GatedHybrid` | `3.877759845223459` | `1.1302239825184375` | `0` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `GatedHybrid` | `1.2447705427183715` | `0.49277861240098336` | `1` |
| `B26a-GatedLegendreQuadratic-h224-basis-specific-temp075` | `GatedHybrid` | `3.019459197058681` | `1.0878346080305927` | `0` |
| `B26a-GatedLegendreQuadratic-h224-basis-specific-temp075` | `GatedHybrid` | `1.360594888393945` | `0.47978694345807155` | `1` |
| `B26b-GatedLegendreQuadratic-h224-basis-specific-temp050` | `GatedHybrid` | `3.528416039604202` | `1.0835837202950014` | `0` |
| `B26b-GatedLegendreQuadratic-h224-basis-specific-temp050` | `GatedHybrid` | `1.09410694997116` | `0.4801625239005736` | `1` |
| `B27a-GatedLegendreQuadratic-h224-basis-specific-cosine-lr` | `GatedHybrid` | `3.8090592929523712` | `1.0839593007375035` | `0` |
| `B27b-GatedLegendreQuadratic-h228-basis-specific-residual-cosine-lr` | `GatedHybrid` | `4.226548298233175` | `1.1267242556678503` | `0` |
| `B27c-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr` | `GatedHybrid` | `3.424375866338532` | `1.1267242556678503` | `0` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `1.3100271214612222` | `1.144393608303742` | `1` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `2.672733598669978` | `1.135226031139033` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `2.185872486642555` | `1.0910441136301556` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.8662057687981972` | `1.193765364654466` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.0791786754371928` | `0.5188985249931712` | `1` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.7728807379252889` | `1.3496141764545206` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.3776236202555598` | `0.5174474187380497` | `1` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `2.4423187376459157` | `1.1574364927615406` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `0.8796084865915128` | `0.5197862605845397` | `1` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `2.156425902101829` | `1.1129302103250478` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `0.7869164572385957` | `0.4577642720568151` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `2.158689530741299` | `0.9463602840753892` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `0.8254962912464021` | `0.31794591641627973` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.8715862138431991` | `0.8702198852772467` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.2469472914580362` | `0.2572043157607211` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.5641038589472247` | `0.8636984430483474` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.2526067024313203` | `0.2573750341436766` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `1.2103981393583458` | `0.7758467631794591` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `0.7762782850070679` | `0.1637530729308932` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.2206439945423133` | `0.777024720021852` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.1521260258106607` | `0.17850314121824637` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.1734128230116618` | `0.7993205408358373` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.1851375369901356` | `0.20800327779295275` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `0.8225274421702274` | `0.8280695165255395` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.1258650828460932` | `0.2670035509423655` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.6505420287624053` | `0.870510106528271` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `0.6784228746088199` | `0.2591675771647091` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.4762977379596336` | `0.9613322862605845` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `0.6851107263960547` | `0.38304083583720294` | `1` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `2.4314189754660664` | `1.100655558590549` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10a-QuadraticSketch-h64-diagnostic', 'B10b-QuadraticSketch-h128-diagnostic', 'B10c-QuadraticSketch-h256-diagnostic', 'B10d-QuadraticSketch-h512-diagnostic', 'B11a-TrainableQuadraticSketch-h128', 'B11b-TrainableQuadraticSketch-h256', 'B12a-LegendreKAN-K4-fan-scale-repair', 'B12b-ChebyKAN-K4-fan-scale-repair', 'B13a-LegendreKAN-K4-identity-residual-repair', 'B13b-ChebyKAN-K4-identity-residual-repair', 'B14a-GatedLegendreQuadratic-h160', 'B14b-GatedLegendreQuadratic-h224', 'B14c-GatedLegendreQuadratic-h224-quad-boost', 'B14d-GatedLegendreQuadratic-h228-quad-boost', 'B15a-GatedLegendreQuadratic-h224-loss-gain', 'B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain', 'B16a-GatedLegendreQuadratic-h224-basis-norm', 'B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm', 'B17a-GatedLegendreQuadratic-h224-input-geom-norm', 'B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm', 'B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm', 'B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm', 'B18a-GatedLegendreQuadratic-h224-group-rms-norm', 'B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm', 'B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm', 'B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm', 'B19a-GatedLegendreQuadratic-h224-residual-geom-norm', 'B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm', 'B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm', 'B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm', 'B1r-ReLU-KAN-stream-K2-repair', 'B20a-GatedLegendreQuadratic-h224-residual-mix15', 'B20b-GatedLegendreQuadratic-h224-residual-mix25', 'B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15', 'B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25', 'B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm', 'B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm', 'B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm', 'B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm', 'B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost', 'B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost', 'B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075', 'B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050', 'B26a-GatedLegendreQuadratic-h224-basis-specific-temp075', 'B26b-GatedLegendreQuadratic-h224-basis-specific-temp050', 'B2a-GaussianRBF-K4-compact', 'B2b-FastKAN-RBF-K8', 'B4a-FourierKAN-lowfreq-K4', 'B6a-BSpline-order1-local-K4', 'B6b-BSpline-order1-local-K8-expression-repair', 'B8d-BSpline-ReLU-lite-combo-K4-repair', 'B9a-Poly2SignedPair-stream-K3-repair', 'B9b-Poly2SignedPair-h64-diagnostic-repair', 'B9c-Poly2SignedPair-h64-K2-diagnostic-repair', 'B9d-Poly2PairRandom-h64-K2-diagnostic-repair']
A2 expression pass = ['B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm', 'B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm', 'B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm', 'B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm', 'B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost', 'B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost', 'B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075', 'B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050', 'B26a-GatedLegendreQuadratic-h224-basis-specific-temp075', 'B26b-GatedLegendreQuadratic-h224-basis-specific-temp050']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `0.00043402777777777775` | `-0.015625` | `0.003979441192415025` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `-0.010850694444444444` | `-0.025390625` | `0.023420354972283047` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `-0.0013020833333333333` | `-0.015625` | `0.004558134824037552` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `-0.008897569444444444` | `-0.033203125` | `0.021551020443439484` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `-0.006510416666666667` | `-0.025390625` | `0.011439600752459632` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `-0.0032552083333333335` | `-0.025390625` | `0.011057673229111565` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `0.00390625` | `-0.017578125` | `-0.00032650637957784865` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `0.004340277777777778` | `-0.009765625` | `-0.0010031681093904707` |
| `B26a-GatedLegendreQuadratic-h224-basis-specific-temp075` | `0.0030381944444444445` | `-0.01171875` | `-0.0017379547158877056` |
| `B26b-GatedLegendreQuadratic-h224-basis-specific-temp050` | `-0.001736111111111111` | `-0.013671875` | `-0.002380286239915424` |

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
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `diagnostic_base_not_qualified` | `-0.00013499184002574438` | `0.00449900838266748` | `-0.004634000222693224` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `diagnostic_base_not_qualified` | `-0.00013990018542164862` | `0.0053534928810177185` | `-0.005493393066439367` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `diagnostic_base_not_qualified` | `-8.645000800155955e-05` | `0.0` | `-8.645000800155955e-05` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `diagnostic_base_not_qualified` | `-7.95033345211138e-05` | `0.00042811441086065827` | `-0.0005076177453817721` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `diagnostic_base_not_qualified` | `-0.00012400512670041053` | `0.0014595131279211415` | `-0.001583518254621552` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `diagnostic_base_not_qualified` | `-0.00023029879443514645` | `0.0029624343609584436` | `-0.00319273315539359` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `diagnostic_base_not_qualified` | `-0.002098617777624767` | `6.056698130052496e-05` | `-0.002159184758925292` | `0` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `diagnostic_base_not_qualified` | `-0.00010431500322383158` | `0.0` | `-0.00010431500322383158` | `0` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `diagnostic_base_not_qualified` | `-0.002182152770044077` | `0.005223042765916119` | `-0.007405195535960196` | `0` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `diagnostic_base_not_qualified` | `-0.00014436169623310846` | `0.006889200536788209` | `-0.007033562233021318` | `0` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `diagnostic_base_not_qualified` | `-0.004002655797295862` | `0.0` | `-0.004002655797295862` | `0` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `diagnostic_base_not_qualified` | `-7.177283231518672e-05` | `0.0010859284882380749` | `-0.0011577013205532616` | `0` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `diagnostic_base_not_qualified` | `0.003768859952016257` | `0.0054622161295045935` | `-0.0016933561774883366` | `0` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `diagnostic_base_not_qualified` | `-0.00014175206701105836` | `0.0008730847249256435` | `-0.0010148367919367018` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `diagnostic_base_not_qualified` | `-0.002065216998578867` | `0.0` | `-0.002065216998578867` | `0` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `diagnostic_base_not_qualified` | `-0.002052105891147349` | `0.00018282349310272394` | `-0.002234929384250073` | `0` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `diagnostic_base_not_qualified` | `-0.002067623863291068` | `0.0` | `-0.002067623863291068` | `0` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `diagnostic_base_not_qualified` | `-7.82757502371112e-05` | `0.001178393978362724` | `-0.0012566697285998352` | `0` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `diagnostic_base_not_qualified` | `-6.60671930798884e-05` | `2.7757184865917495e-05` | `-9.382437794580589e-05` | `0` |
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `diagnostic_base_not_qualified` | `-0.00011866015135186814` | `0.00691744941561101` | `-0.007036109566962878` | `0` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `diagnostic_base_not_qualified` | `0.0010797791264733902` | `0.0002384023627388654` | `0.0008413767637345249` | `1` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `diagnostic_base_not_qualified` | `0.0003406812856390218` | `0.006067395143654508` | `-0.005726713858015486` | `0` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `diagnostic_base_not_qualified` | `-0.00010235719731266357` | `0.0027740255847445994` | `-0.002876382782057263` | `0` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `diagnostic_base_not_qualified` | `-0.00011949281749057405` | `0.0014906414233433196` | `-0.0016101342408338937` | `0` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `diagnostic_base_not_qualified` | `-0.00011516107177200752` | `0.005180975999423687` | `-0.0052961370711956945` | `0` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `diagnostic_base_not_qualified` | `-0.0001487372966906264` | `0.0033333992106774346` | `-0.003482136507368061` | `0` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `diagnostic_base_not_qualified` | `-0.00014872105464114327` | `0.0029415645725627826` | `-0.003090285627203926` | `0` |
| `B26a-GatedLegendreQuadratic-h224-basis-specific-temp075` | `diagnostic_base_not_qualified` | `-0.00014203292268177847` | `0.0` | `-0.00014203292268177847` | `0` |
| `B26b-GatedLegendreQuadratic-h224-basis-specific-temp050` | `diagnostic_base_not_qualified` | `-0.0001764363926364254` | `0.003266889435511011` | `-0.0034433258281474366` | `0` |
| `B2a-GaussianRBF-K4-compact` | `diagnostic_base_not_qualified` | `0.016744405456739386` | `0.0` | `0.016744405456739386` | `1` |
| `B2b-FastKAN-RBF-K8` | `diagnostic_base_not_qualified` | `-0.013029543715944447` | `0.0` | `-0.013029543715944447` | `0` |
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
rows_checked = 5944
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `d7d9021b600a1ee0d26bf69ef38522a849916cc42e99c2cb73efaec18579cbe0` |
| `v124_control_matrix.csv` | `361f1e9e74b24d679e018e2319fca0dc0f0c77bcd4874b057223059827239de7` |
| `v124_efficiency_microbench.csv` | `3fb52ef524b9fab9b27c2d6bce1e77ffdd5cbaebac9abf92834cb935a848c4f5` |
| `v124_expression_battery.csv` | `50ff2941a33142afbcc0bf395e62d7ba0c0bcb3706c61e5e8fb7c37897133bbd` |
| `v124_provenance_audit.csv` | `eb23bcede725585236e2b050bc11e2aecae03a02f771f02f66f53ba98c1fd015` |
| `v124_route_decision.json` | `b2e52772af31993dee48d99b8f60b3b957b986a5922ae208de2952b1b3701733` |
| `v124_task_triage.csv` | `5b253c50a1474cbb3f51ea3c7d9d1544718739ea8eee2651553dd1e354ec9413` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
