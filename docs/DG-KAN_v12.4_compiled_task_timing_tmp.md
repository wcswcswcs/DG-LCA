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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_compiled_task_timing_20260520T190000Z
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
    "B1a-ReLU-KAN-local-hinge-K4",
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
    "B2a-GaussianRBF-K4-compact",
    "B2b-FastKAN-RBF-K8",
    "B4a-FourierKAN-lowfreq-K4",
    "B6a-BSpline-order1-local-K4",
    "B6b-BSpline-order1-local-K8-expression-repair",
    "B6r-BSpline-order1-stream-K2-repair",
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
    "B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050"
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.6745537605622725` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `0.6536131653106044` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `1.1339493618354575` | `1.0742112810707456` | `1` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `2.597679909654109` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `1.9795781533120596` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `1.446054894928057` | `1.1750887735591369` | `1` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.943267592753963` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.0972858505679064` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `1.787820329462581` | `1.0923415733406172` | `0` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `1.872785412256513` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.2624779346487143` | `1.1100450696531003` | `1` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `0.6224732436668481` | `0.5075628243649276` | `1` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.0575277623909722` | `1.1272193389784213` | `1` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.0446323078147794` | `0.5418430756623873` | `1` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.4353599545565578` | `1.1451959847036328` | `1` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `0.6912466088555154` | `0.5075628243649276` | `1` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.7528958864217665` | `1.0854616225075115` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `0.8520702902227669` | `0.505736137667304` | `1` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `2.406182023621241` | `1.0465890467085497` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `0.9758262576108332` | `0.3693321496858782` | `1` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `2.2561576848521505` | `1.078837749248839` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `0.6921543150330071` | `0.4682634526085769` | `1` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `2.237431662514108` | `1.0960973777656378` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `0.7892809522052537` | `0.4686219612127834` | `1` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `2.426782955719575` | `1.100348265501229` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `1.0050528304926634` | `0.4752458344714559` | `1` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `2.9272770867128113` | `1.0978899207866704` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `1.0460764990706422` | `0.4693389784211964` | `1` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `2.4665854741622875` | `1.101065282709642` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `0.8908938598321804` | `0.4759628516798689` | `1` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `2.5786845177073885` | `1.0745697896749522` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `1.2435137447310984` | `0.47005599562960937` | `1` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `3.4837663435457293` | `1.0777451515979242` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `1.2258339394792155` | `0.476714012564873` | `1` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `3.2774728230919687` | `1.087561458617864` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `0.8107872177032007` | `0.47794318492215243` | `1` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `2.72139138745695` | `1.0944584812892653` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `0.9577624182592849` | `0.4845670581808249` | `1` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `4.277586063916229` | `1.0918635618683419` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `1.142858075019881` | `0.4786602021305654` | `1` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `3.536098908771317` | `1.0951754984976783` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `0.8133997054670818` | `0.485318219065829` | `1` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `3.3277855018288776` | `1.1137838022398252` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `1.0578128516530965` | `0.47234362196121277` | `1` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `2.795536425165729` | `1.1135789401802787` | `0` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `1.0548079657475276` | `0.4789674952198853` | `1` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `3.4271897091674286` | `1.1109840207593553` | `0` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `1.186511493320808` | `0.47306063916962576` | `1` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `3.25970297091959` | `1.1142959573886917` | `0` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `1.2956588605042696` | `0.4797186561048894` | `1` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `GatedHybrid` | `3.6116200983302877` | `1.1170615951925704` | `0` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `GatedHybrid` | `0.868045847763283` | `0.4809478284621688` | `1` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `GatedHybrid` | `2.682982757379072` | `1.1239586178639716` | `0` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `GatedHybrid` | `1.3765946105435094` | `0.4875717017208413` | `1` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `GatedHybrid` | `3.716814298928963` | `1.1213636984430484` | `0` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `GatedHybrid` | `0.8995221699907899` | `0.4816648456705818` | `1` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `GatedHybrid` | `3.0033987077139837` | `1.1246756350723845` | `0` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `GatedHybrid` | `0.8730923903945679` | `0.4883228626058454` | `1` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `GatedHybrid` | `2.6503249104068733` | `1.1221148593280525` | `0` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `GatedHybrid` | `1.1670212585474873` | `0.482450150232177` | `1` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `GatedHybrid` | `3.800585281255446` | `1.1214319857962305` | `0` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `GatedHybrid` | `0.9597683053584798` | `0.4828086588363835` | `1` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `GatedHybrid` | `3.1927867196571236` | `1.1258194482381862` | `0` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `GatedHybrid` | `0.8370815710019966` | `0.489432532095056` | `1` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `GatedHybrid` | `3.280896741570914` | `1.1272534826550122` | `0` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `GatedHybrid` | `1.3723624645996368` | `0.4897910406992625` | `1` |
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `GatedHybrid` | `3.568288903265286` | `1.0848470363288718` | `0` |
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `GatedHybrid` | `1.213084114145327` | `0.47678229991805515` | `1` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `GatedHybrid` | `2.590516255960566` | `1.08447145588637` | `0` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `GatedHybrid` | `1.1194473586770528` | `0.4834232450150232` | `1` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `GatedHybrid` | `3.6303412564736797` | `1.1207661841027041` | `0` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `GatedHybrid` | `0.9852533634423026` | `0.4846353455340071` | `1` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `GatedHybrid` | `2.9486886783733994` | `1.1276461349358098` | `0` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `GatedHybrid` | `1.001710713531086` | `0.49127629063097517` | `1` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `GatedHybrid` | `2.9284888643754576` | `1.1290972411909315` | `0` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `GatedHybrid` | `0.9614867897371187` | `0.4916518710734772` | `1` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `GatedHybrid` | `3.3816001634369397` | `1.1294728216334335` | `0` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `GatedHybrid` | `1.1320934342981956` | `0.49202745151597926` | `1` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `GatedHybrid` | `4.002909737512633` | `1.1298484020759356` | `0` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `GatedHybrid` | `0.8137566305794832` | `0.4924030319584813` | `1` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `GatedHybrid` | `2.7798353387063304` | `1.1302239825184375` | `0` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `GatedHybrid` | `0.9404647488847646` | `0.49277861240098336` | `1` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `1.4028687359108902` | `1.1482689155968315` | `1` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `2.2002156499175043` | `1.1215173449877083` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `1.41800160138779` | `1.0949194209232451` | `1` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.3720845529905958` | `1.179646954384048` | `1` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `0.9763678441616435` | `0.5175327779295275` | `1` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.6569753915647032` | `1.3226748156241463` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.0893195454374505` | `0.5158426659382682` | `1` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `1.6562620752149233` | `1.1380770281343895` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `0.7593910692674367` | `0.5531275607757443` | `1` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `1.3963087762291313` | `1.0885516252390057` | `1` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `0.7365514805600087` | `0.4575252663206774` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `1.2739610525120657` | `0.921571974870254` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `0.8197197773053049` | `0.31719475553127563` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.105491509666875` | `0.8694687243922425` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `0.7092797800529816` | `0.256453154875717` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.0193845275188251` | `0.8629472821633434` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `0.9704726182025992` | `0.2566238732586725` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `0.6817818351752218` | `0.7750956022944551` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `0.6965217093159648` | `0.1630019120458891` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `0.9807002399708906` | `0.7762735591368478` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `0.5812896498598459` | `0.17775198033324227` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `0.9916367884498866` | `0.7985693799508331` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `0.5751843737390909` | `0.20725211690794865` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.0163361011391825` | `0.8273183556405354` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `0.5826908345166574` | `0.2662523900573614` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `0.8021392728076451` | `0.8697589456432668` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `0.6090299146784666` | `0.258416416279705` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.2678906562261274` | `0.9605811253755805` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `0.6307614161165479` | `0.38228967495219884` | `1` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `1.7305218960535838` | `1.099904397705545` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10a-QuadraticSketch-h64-diagnostic', 'B10b-QuadraticSketch-h128-diagnostic', 'B10c-QuadraticSketch-h256-diagnostic', 'B10d-QuadraticSketch-h512-diagnostic', 'B11a-TrainableQuadraticSketch-h128', 'B11b-TrainableQuadraticSketch-h256', 'B12a-LegendreKAN-K4-fan-scale-repair', 'B12b-ChebyKAN-K4-fan-scale-repair', 'B13a-LegendreKAN-K4-identity-residual-repair', 'B13b-ChebyKAN-K4-identity-residual-repair', 'B14a-GatedLegendreQuadratic-h160', 'B14b-GatedLegendreQuadratic-h224', 'B14c-GatedLegendreQuadratic-h224-quad-boost', 'B14d-GatedLegendreQuadratic-h228-quad-boost', 'B15a-GatedLegendreQuadratic-h224-loss-gain', 'B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain', 'B16a-GatedLegendreQuadratic-h224-basis-norm', 'B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm', 'B17a-GatedLegendreQuadratic-h224-input-geom-norm', 'B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm', 'B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm', 'B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm', 'B18a-GatedLegendreQuadratic-h224-group-rms-norm', 'B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm', 'B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm', 'B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm', 'B19a-GatedLegendreQuadratic-h224-residual-geom-norm', 'B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm', 'B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm', 'B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm', 'B1a-ReLU-KAN-local-hinge-K4', 'B1r-ReLU-KAN-stream-K2-repair', 'B20a-GatedLegendreQuadratic-h224-residual-mix15', 'B20b-GatedLegendreQuadratic-h224-residual-mix25', 'B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15', 'B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25', 'B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm', 'B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm', 'B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm', 'B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm', 'B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost', 'B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost', 'B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075', 'B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050', 'B2a-GaussianRBF-K4-compact', 'B2b-FastKAN-RBF-K8', 'B4a-FourierKAN-lowfreq-K4', 'B6a-BSpline-order1-local-K4', 'B6b-BSpline-order1-local-K8-expression-repair', 'B6r-BSpline-order1-stream-K2-repair', 'B8d-BSpline-ReLU-lite-combo-K4-repair', 'B9a-Poly2SignedPair-stream-K3-repair', 'B9b-Poly2SignedPair-h64-diagnostic-repair', 'B9c-Poly2SignedPair-h64-K2-diagnostic-repair', 'B9d-Poly2PairRandom-h64-K2-diagnostic-repair']
A2 expression pass = ['B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm', 'B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm', 'B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm', 'B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm', 'B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost', 'B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost', 'B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075', 'B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `0.007595486111111111` | `-0.0078125` | `0.007171362638473511` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `0.00021701388888888888` | `-0.013671875` | `0.023001437799798116` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `0.006510416666666667` | `-0.009765625` | `0.010234913478295008` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `-0.004123263888888889` | `-0.02734375` | `0.026371407839987013` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `0.0008680555555555555` | `-0.01171875` | `0.014450787256161371` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `0.0006510416666666666` | `-0.017578125` | `0.014616372684637705` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `0.009548611111111112` | `-0.005859375` | `0.007062929785913891` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `0.009982638888888888` | `-0.0078125` | `0.0017214732037650214` |

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
| `B1a-ReLU-KAN-local-hinge-K4` | `diagnostic_base_not_qualified` | `0.002096993869568742` | `0.005597837989548715` | `-0.0035008441199799734` | `0` |
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
| `B2a-GaussianRBF-K4-compact` | `diagnostic_base_not_qualified` | `0.016744405456739386` | `0.0` | `0.016744405456739386` | `1` |
| `B2b-FastKAN-RBF-K8` | `diagnostic_base_not_qualified` | `-0.013029543715944447` | `0.0` | `-0.013029543715944447` | `0` |
| `B4a-FourierKAN-lowfreq-K4` | `diagnostic_base_not_qualified` | `0.003803054025772079` | `0.001866741251363102` | `0.001936312774408977` | `1` |
| `B6a-BSpline-order1-local-K4` | `diagnostic_base_not_qualified` | `0.002269369158875989` | `0.006272526008597623` | `-0.004003156849721634` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `diagnostic_base_not_qualified` | `0.02617166891475997` | `0.07238561982052971` | `-0.04621395090576974` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `diagnostic_base_not_qualified` | `-0.003070495001477269` | `0.021122178706267647` | `-0.024192673707744916` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `diagnostic_base_not_qualified` | `0.0014408970357095985` | `0.0037932873636172815` | `-0.002352390327907683` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `diagnostic_base_not_qualified` | `-4.055223265186925e-06` | `1.016035103074131e-05` | `-1.4215574295928235e-05` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `diagnostic_base_not_qualified` | `4.14654969960182e-05` | `1.1681385773343322e-05` | `2.978411122267488e-05` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `-7.393090586393924e-07` | `2.451584039064869e-05` | `-2.5255149449288083e-05` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `5.402795305364805e-06` | `0.0` | `5.402795305364805e-06` | `1` |

## 6. No-Fake / Hash

```text
rows_checked = 5671
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `1475352572ad5b66122b5eb37a84e5c24362c95f5019fc176df0178be088bbde` |
| `v124_control_matrix.csv` | `1ea62ff471f0ce252c27e38209d62be9ac754e3ff8fd4ca6d4a7764cdf0a06fc` |
| `v124_efficiency_microbench.csv` | `fe780069266f96cd1a48b51eec12421f7d5110b0afb9392272f53412ea216713` |
| `v124_expression_battery.csv` | `d7df3a5764ba56dccf9514e059f751bb3c1490753c3bf058ae0ded78d66fe28f` |
| `v124_provenance_audit.csv` | `509cc59e2366f8b25d7080dd84730fc632218134ac6f6e0ddbd263a202bd7d83` |
| `v124_route_decision.json` | `7f3879b070c59ef161e7b943948e9d094456c9ff1218139a963321c495f55ad7` |
| `v124_task_triage.csv` | `34e166e5f91263c4cca4de17eb633cd01992d5b5a252df07d891f14fbbe98eac` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
