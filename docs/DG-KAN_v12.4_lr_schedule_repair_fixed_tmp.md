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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_lr_schedule_repair_fixed_20260520T230000Z
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
    "B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm",
    "B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost",
    "B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075",
    "B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050",
    "B26b-GatedLegendreQuadratic-h224-basis-specific-temp050",
    "B27a-GatedLegendreQuadratic-h224-basis-specific-cosine-lr",
    "B27b-GatedLegendreQuadratic-h228-basis-specific-residual-cosine-lr",
    "B27c-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr",
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
    "B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm",
    "B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm",
    "B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm",
    "B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost",
    "B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075",
    "B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050",
    "B26b-GatedLegendreQuadratic-h224-basis-specific-temp050",
    "B27a-GatedLegendreQuadratic-h224-basis-specific-cosine-lr",
    "B27b-GatedLegendreQuadratic-h228-basis-specific-residual-cosine-lr",
    "B27c-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr"
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.548713757010195` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.2613969586540426` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `2.3719233192561813` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `2.9237959210317195` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `2.8512450900256567` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `2.2308671355266445` | `1.1750887735591369` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.9155123969756829` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.2206665928751448` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `1.8108080455463977` | `1.0923415733406172` | `0` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `1.7950362429898061` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.4053615216947075` | `1.1100450696531003` | `1` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.1138752185364307` | `0.5075628243649276` | `1` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.9500298004223149` | `1.1272193389784213` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.1177098233544491` | `0.5418430756623873` | `1` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `2.2560621154326457` | `1.1451959847036328` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.0351108575710104` | `0.5075628243649276` | `1` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.9085975637445225` | `1.0854616225075115` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `0.8238393445042345` | `0.505736137667304` | `1` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `2.807309759780215` | `1.0465890467085497` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `1.3859329235065732` | `0.3693321496858782` | `1` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `2.968227357355312` | `1.078837749248839` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `1.0730798594555322` | `0.4682634526085769` | `1` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `3.3701131564607323` | `1.0960973777656378` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `1.4757868730558772` | `0.4686219612127834` | `1` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `3.420374832550008` | `1.100348265501229` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `1.5042048395885839` | `0.4752458344714559` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `3.052946836046591` | `1.0978899207866704` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `1.0700041720591242` | `0.4693389784211964` | `1` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `2.2614680539472793` | `1.101065282709642` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `1.0374057739027769` | `0.4759628516798689` | `1` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `2.4385354937220445` | `1.0745697896749522` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `1.401875439910996` | `0.47005599562960937` | `1` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `2.830128794587108` | `1.0777451515979242` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `0.8662315805008741` | `0.476714012564873` | `1` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `2.98576887927707` | `1.087561458617864` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `1.1963131201325976` | `0.47794318492215243` | `1` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `3.525113525418341` | `1.0944584812892653` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `0.939684229048884` | `0.4845670581808249` | `1` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `3.02166334604818` | `1.0918635618683419` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `1.1786418954203846` | `0.4786602021305654` | `1` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `4.592503348999841` | `1.0951754984976783` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `1.0333883931612289` | `0.485318219065829` | `1` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `3.1578098392480083` | `1.1137838022398252` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `0.9791090525168585` | `0.47234362196121277` | `1` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `2.9683742308652907` | `1.1135789401802787` | `0` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `1.186888239901914` | `0.4789674952198853` | `1` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `3.6513422961651125` | `1.1109840207593553` | `0` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `0.9699827441364122` | `0.47306063916962576` | `1` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `3.5822541323252284` | `1.1142959573886917` | `0` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `1.0112861294643871` | `0.4797186561048894` | `1` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `GatedHybrid` | `3.20653948981677` | `1.1170615951925704` | `0` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `GatedHybrid` | `0.9759869616057035` | `0.4809478284621688` | `1` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `GatedHybrid` | `3.146641350498377` | `1.1239586178639716` | `0` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `GatedHybrid` | `1.0608313182571578` | `0.4875717017208413` | `1` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `GatedHybrid` | `3.734579984333493` | `1.1213636984430484` | `0` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `GatedHybrid` | `0.8422601207910451` | `0.4816648456705818` | `1` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `GatedHybrid` | `3.1041643961583` | `1.1246756350723845` | `0` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `GatedHybrid` | `0.9627907669777264` | `0.4883228626058454` | `1` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `GatedHybrid` | `3.2562173614422267` | `1.1221148593280525` | `0` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `GatedHybrid` | `1.080614967191154` | `0.482450150232177` | `1` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `GatedHybrid` | `3.2192473832391073` | `1.1214319857962305` | `0` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `GatedHybrid` | `1.0241908475807733` | `0.4828086588363835` | `1` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `GatedHybrid` | `3.8220672127239297` | `1.1258194482381862` | `0` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `GatedHybrid` | `0.8505222169243694` | `0.489432532095056` | `1` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `GatedHybrid` | `3.0628253922303204` | `1.1272534826550122` | `0` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `GatedHybrid` | `1.2727395670140544` | `0.4897910406992625` | `1` |
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `GatedHybrid` | `2.9339809958449696` | `1.0848470363288718` | `0` |
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `GatedHybrid` | `1.280869973662103` | `0.47678229991805515` | `1` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `GatedHybrid` | `3.5363806393751562` | `1.08447145588637` | `0` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `GatedHybrid` | `1.138028602729151` | `0.4834232450150232` | `1` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `GatedHybrid` | `3.7071502565674455` | `1.1207661841027041` | `0` |
| `B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm` | `GatedHybrid` | `1.503278330268147` | `0.4846353455340071` | `0` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `GatedHybrid` | `4.605140430942488` | `1.1276461349358098` | `0` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `GatedHybrid` | `1.0323608462184684` | `0.49127629063097517` | `1` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `GatedHybrid` | `3.6506584474263795` | `1.1290972411909315` | `0` |
| `B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost` | `GatedHybrid` | `1.57199356310878` | `0.4916518710734772` | `0` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `GatedHybrid` | `3.813567564652726` | `1.1294728216334335` | `0` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `GatedHybrid` | `1.0443353654383216` | `0.49202745151597926` | `1` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `GatedHybrid` | `3.9030605033717047` | `1.1298484020759356` | `0` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `GatedHybrid` | `1.0033857536952524` | `0.4924030319584813` | `1` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `GatedHybrid` | `3.821167523783575` | `1.1302239825184375` | `0` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `GatedHybrid` | `1.2819356935267807` | `0.49277861240098336` | `1` |
| `B26a-GatedLegendreQuadratic-h224-basis-specific-temp075` | `GatedHybrid` | `2.748311593215721` | `1.0878346080305927` | `0` |
| `B26a-GatedLegendreQuadratic-h224-basis-specific-temp075` | `GatedHybrid` | `1.5383361147060828` | `0.47978694345807155` | `0` |
| `B26b-GatedLegendreQuadratic-h224-basis-specific-temp050` | `GatedHybrid` | `3.3384984560543107` | `1.0835837202950014` | `0` |
| `B26b-GatedLegendreQuadratic-h224-basis-specific-temp050` | `GatedHybrid` | `1.2026830313557206` | `0.4801625239005736` | `1` |
| `B27a-GatedLegendreQuadratic-h224-basis-specific-cosine-lr` | `GatedHybrid` | `4.099709233022273` | `1.0839593007375035` | `0` |
| `B27a-GatedLegendreQuadratic-h224-basis-specific-cosine-lr` | `GatedHybrid` | `1.3239846570397114` | `0.48053810434307564` | `1` |
| `B27b-GatedLegendreQuadratic-h228-basis-specific-residual-cosine-lr` | `GatedHybrid` | `4.254097274254706` | `1.1270998361103524` | `0` |
| `B27b-GatedLegendreQuadratic-h228-basis-specific-residual-cosine-lr` | `GatedHybrid` | `1.192250613037259` | `0.49428093417099156` | `1` |
| `B27c-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr` | `GatedHybrid` | `4.313570970415276` | `1.1321018847309479` | `0` |
| `B27c-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr` | `GatedHybrid` | `1.433143036350839` | `0.4946565146134936` | `1` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `1.917590337851645` | `1.1501468178093417` | `0` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `2.996902600867334` | `1.1233952472002184` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `1.7689391617283112` | `1.0967973231357553` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.5589721976250484` | `1.1815248565965584` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.1466528449469837` | `0.5194106801420377` | `1` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.7867595304588701` | `1.3245527178366567` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.0595579887836886` | `0.5177205681507785` | `1` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `2.320627994232909` | `1.1399549303468997` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `1.1103125070953386` | `0.5550054629882546` | `1` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `2.4138645755284607` | `1.090429527451516` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `0.8900047964489249` | `0.45940316853318763` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `1.6698391344822106` | `0.9234498770827643` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `1.360406307472243` | `0.31907265774378585` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `2.116440183457076` | `0.8713466266047528` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `0.9542434382308199` | `0.25833105708822723` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.502413692300706` | `0.8648251843758535` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.2027643439366074` | `0.25850177547118275` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `0.9219265831119587` | `0.7769735045069653` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `0.6800694775560248` | `0.16487981425839934` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `0.8919757396181005` | `0.7781514613493581` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `0.9315411926980451` | `0.17962988254575252` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.2412535760506778` | `0.8004472821633434` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `0.7884740480893672` | `0.2091300191204589` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.0363778012396976` | `0.8291962578530456` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `0.9384758928774153` | `0.2681302922698716` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.5280534080330586` | `0.8716368478557771` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `0.8233591319846514` | `0.26029431849221524` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.30090096610131` | `0.9624590275880907` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `0.8325125445587267` | `0.3841675771647091` | `1` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `1.9144024021978525` | `1.101782299918055` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10a-QuadraticSketch-h64-diagnostic', 'B10b-QuadraticSketch-h128-diagnostic', 'B10c-QuadraticSketch-h256-diagnostic', 'B10d-QuadraticSketch-h512-diagnostic', 'B11a-TrainableQuadraticSketch-h128', 'B11b-TrainableQuadraticSketch-h256', 'B12a-LegendreKAN-K4-fan-scale-repair', 'B12b-ChebyKAN-K4-fan-scale-repair', 'B13a-LegendreKAN-K4-identity-residual-repair', 'B13b-ChebyKAN-K4-identity-residual-repair', 'B14a-GatedLegendreQuadratic-h160', 'B14b-GatedLegendreQuadratic-h224', 'B14c-GatedLegendreQuadratic-h224-quad-boost', 'B15a-GatedLegendreQuadratic-h224-loss-gain', 'B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain', 'B16a-GatedLegendreQuadratic-h224-basis-norm', 'B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm', 'B17a-GatedLegendreQuadratic-h224-input-geom-norm', 'B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm', 'B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm', 'B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm', 'B18a-GatedLegendreQuadratic-h224-group-rms-norm', 'B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm', 'B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm', 'B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm', 'B19a-GatedLegendreQuadratic-h224-residual-geom-norm', 'B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm', 'B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm', 'B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm', 'B1r-ReLU-KAN-stream-K2-repair', 'B20a-GatedLegendreQuadratic-h224-residual-mix15', 'B20b-GatedLegendreQuadratic-h224-residual-mix25', 'B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15', 'B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25', 'B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm', 'B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm', 'B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm', 'B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost', 'B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075', 'B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050', 'B26b-GatedLegendreQuadratic-h224-basis-specific-temp050', 'B27a-GatedLegendreQuadratic-h224-basis-specific-cosine-lr', 'B27b-GatedLegendreQuadratic-h228-basis-specific-residual-cosine-lr', 'B27c-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr', 'B2b-FastKAN-RBF-K8', 'B6a-BSpline-order1-local-K4', 'B6b-BSpline-order1-local-K8-expression-repair', 'B8d-BSpline-ReLU-lite-combo-K4-repair', 'B9a-Poly2SignedPair-stream-K3-repair', 'B9b-Poly2SignedPair-h64-diagnostic-repair', 'B9c-Poly2SignedPair-h64-K2-diagnostic-repair', 'B9d-Poly2PairRandom-h64-K2-diagnostic-repair']
A2 expression pass = ['B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm', 'B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm', 'B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm', 'B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost', 'B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075', 'B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050', 'B26b-GatedLegendreQuadratic-h224-basis-specific-temp050', 'B27a-GatedLegendreQuadratic-h224-basis-specific-cosine-lr', 'B27b-GatedLegendreQuadratic-h228-basis-specific-residual-cosine-lr', 'B27c-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm` | `0.00043402777777777775` | `-0.015625` | `0.003979441192415025` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `-0.010850694444444444` | `-0.025390625` | `0.023420354972283047` |
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `-0.011501736111111112` | `-0.03125` | `0.02323410411675771` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `-0.0013020833333333333` | `-0.01953125` | `0.0066630372570620645` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `0.00021701388888888888` | `-0.01953125` | `0.003853798740439945` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `0.002170138888888889` | `-0.013671875` | `-0.0038603341413868796` |
| `B26b-GatedLegendreQuadratic-h224-basis-specific-temp050` | `0.0013020833333333333` | `-0.017578125` | `-0.003882397794061237` |
| `B27a-GatedLegendreQuadratic-h224-basis-specific-cosine-lr` | `-0.006944444444444444` | `-0.02734375` | `-0.012792886959181892` |
| `B27b-GatedLegendreQuadratic-h228-basis-specific-residual-cosine-lr` | `-0.011935763888888888` | `-0.03515625` | `0.004631936136219237` |
| `B27c-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr` | `-0.006727430555555556` | `-0.02734375` | `-0.018747775091065302` |

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
| `B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm` | `diagnostic_base_not_qualified` | `-0.00010235719731266357` | `0.0027740255847445994` | `-0.002876382782057263` | `0` |
| `B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost` | `diagnostic_base_not_qualified` | `-0.00011516107177200752` | `0.005180975999423687` | `-0.0052961370711956945` | `0` |
| `B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075` | `diagnostic_base_not_qualified` | `-0.0001487372966906264` | `0.0033333992106774346` | `-0.003482136507368061` | `0` |
| `B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050` | `diagnostic_base_not_qualified` | `-0.00014872105464114327` | `0.0029415645725627826` | `-0.003090285627203926` | `0` |
| `B26b-GatedLegendreQuadratic-h224-basis-specific-temp050` | `diagnostic_base_not_qualified` | `-0.0001764363926364254` | `0.003266889435511011` | `-0.0034433258281474366` | `0` |
| `B27a-GatedLegendreQuadratic-h224-basis-specific-cosine-lr` | `diagnostic_base_not_qualified` | `-0.00011866015135186814` | `0.00691744941561101` | `-0.007036109566962878` | `0` |
| `B27b-GatedLegendreQuadratic-h228-basis-specific-residual-cosine-lr` | `diagnostic_base_not_qualified` | `-0.00010235719731266357` | `0.0027740255847445994` | `-0.002876382782057263` | `0` |
| `B27c-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr` | `diagnostic_base_not_qualified` | `-0.00014872105464114327` | `0.0029415645725627826` | `-0.003090285627203926` | `0` |
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
rows_checked = 5718
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `d7d9021b600a1ee0d26bf69ef38522a849916cc42e99c2cb73efaec18579cbe0` |
| `v124_control_matrix.csv` | `2240ed5447e9ee60c88ce5694cbbbc98533920df4a9b6553a2a8265ec8efb1f5` |
| `v124_efficiency_microbench.csv` | `eeed21448667738988c3f16a2307cd66a0c03dac3e6ed4854e25b6ba019d5de5` |
| `v124_expression_battery.csv` | `dde083aae63ea6dfe4fa24780cf1e0a7e6618830bd3365489b540fa07a47be4b` |
| `v124_provenance_audit.csv` | `2f573df00a1d95d1d26361a9d1b3b8bc50558a75cbad7197e4e43755bf05fe58` |
| `v124_route_decision.json` | `4ac3d1b0d0e6a6e784a7e4c370a071b7c8c8f1b84ba48fdd36f1a26aa2c7c1a6` |
| `v124_task_triage.csv` | `ee899fc67bc679c0c80addf38e6f23e6f767bcb59a7a72b5e0d480c85d932fd2` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
