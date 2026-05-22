# DG-KAN v12.4 Multi-Basis Functional Dual 结果复盘

> 本复盘记录 `DG-KAN_v12.4_多基函数效率优先与Functional双线计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位数据或 CPU offload。Base 未合格时 Functional 只允许 diagnostic。

## 0. 最新结论

```text
route = R2-EfficiencyPassExpressionFail
base_qualified = False
functional_open = False
functional_diagnostic_positive = False
next_recommended_action = increase K within efficiency budget, improve init/normalization, or try lite hybrid
```

最终 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_basisnorm_20260520T100000Z
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
  "route": "R2-EfficiencyPassExpressionFail",
  "base_blocker_route": "expression_fail",
  "A1_exploratory_survivors": [
    "B10b-QuadraticSketch-h128-diagnostic",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B2b-FastKAN-RBF-K8",
    "B9c-Poly2SignedPair-h64-K2-diagnostic-repair"
  ],
  "A2_expression_pass": [],
  "A3_task_pass": [],
  "functional_diagnostic_positive": false,
  "base_qualified": false,
  "functional_open": false,
  "next_recommended_action": "increase K within efficiency budget, improve init/normalization, or try lite hybrid",
  "no_fake": true,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. A1 Efficiency Microbench batch=128

| candidate | family | step ratio q90 vs MLP | memory ratio | exploratory pass |
|---|---|---:|---:|---:|
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.8942291154371216` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.2125899797194484` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `2.0569835488713797` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `2.6572016210870855` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `2.622692710731606` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `2.185777439477301` | `1.1750887735591369` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.9896297470569966` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.2917316272461967` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `3.067790316347613` | `1.0923415733406172` | `0` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `3.23359819258721` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `2.8646229233047253` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `2.308781395557318` | `0.5075628243649276` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `4.47032611007671` | `1.1272193389784213` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `2.5947482757624534` | `0.5418430756623873` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `3.9059204262586005` | `1.1451959847036328` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `2.689994389873274` | `0.5075628243649276` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `3.6430634232177` | `1.0854616225075115` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `2.0049051855685374` | `0.505736137667304` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `5.9780611866718525` | `1.0465378311936628` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `2.2776567579101297` | `0.3692297186561049` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `7.290379110713183` | `1.0787353182190658` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `2.5924451115872404` | `0.4681610215788036` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `6.142490639251169` | `1.0959949467358645` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `2.4021841970019078` | `0.4685195301830101` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `5.945381812970486` | `1.1002458344714559` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `2.6072244080504134` | `0.4751434034416826` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `5.234219860835548` | `1.097787489756897` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `2.872447081351361` | `0.4692365473914231` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `6.5850002050980905` | `1.1009628516798688` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `2.752420800010034` | `0.4758604206500956` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `7.541824846641629` | `1.074467358645179` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `2.596406640325698` | `0.4699535645998361` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `7.021558019679753` | `1.0776427205681507` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `2.775313447968259` | `0.4766115815350997` | `0` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `3.699540343822533` | `1.1357040426113083` | `0` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `6.320143829790997` | `1.1089524720021853` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `4.072579022649512` | `1.082354547937722` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `3.526133352002914` | `1.167082081398525` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `2.185679362495506` | `0.5085871346626605` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `3.451932255329543` | `1.3137291723572795` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `2.20152434658267` | `0.5068970226714012` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `4.025463519115194` | `1.1299849767822998` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `2.0353820422280005` | `0.5450355094236548` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `3.778987839379512` | `1.0795035509423654` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `1.9997639572849042` | `0.4725143403441683` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `3.8193100212366478` | `0.912626331603387` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `2.4218705295427094` | `0.30824911226440865` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `3.491622488332283` | `0.8605230811253756` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.2066221907087378` | `0.24750751160885004` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `2.2889179281741345` | `0.8540016388964764` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `2.238060078422519` | `0.24767822999180553` | `0` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `1.9226291868667724` | `0.7661499590275881` | `0` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `2.0648632221907293` | `0.15405626877902212` | `0` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.0914594584814126` | `0.7673279158699808` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `2.068677635456494` | `0.1688063370663753` | `0` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.9818290287921738` | `0.7896237366839661` | `0` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `2.111593717028169` | `0.19830647364108167` | `0` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.988044888827068` | `0.8183727123736684` | `0` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.914491161917181` | `0.2573067467904944` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `2.657522067577919` | `0.8608133023763999` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.8111637718022098` | `0.24947077301283802` | `0` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `2.5594147579950475` | `0.9516354821087135` | `0` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.9755086149936676` | `0.37334403168533187` | `0` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `4.74475958956942` | `1.0909587544386778` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10b-QuadraticSketch-h128-diagnostic', 'B1r-ReLU-KAN-stream-K2-repair', 'B2b-FastKAN-RBF-K8', 'B9c-Poly2SignedPair-h64-K2-diagnostic-repair']
A2 expression pass = []
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B10b-QuadraticSketch-h128-diagnostic` | `-0.1560329861111111` | `-0.23828125` | `0.1669517643749714` |
| `B1r-ReLU-KAN-stream-K2-repair` | `-0.2938368055555556` | `-0.34375` | `0.3115432175497214` |
| `B2b-FastKAN-RBF-K8` | `-0.21961805555555555` | `-0.30859375` | `0.36783884424302316` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B10b-QuadraticSketch-h128-diagnostic` | `diagnostic_base_not_qualified` | `0.001973967254161657` | `0.002069243043661295` | `-9.527578949963811e-05` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `diagnostic_base_not_qualified` | `-0.002065216998578867` | `0.0` | `-0.002065216998578867` | `0` |
| `B2b-FastKAN-RBF-K8` | `diagnostic_base_not_qualified` | `-0.013029543715944447` | `0.0` | `-0.013029543715944447` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `-7.393090586393924e-07` | `2.451584039064869e-05` | `-2.5255149449288083e-05` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 990
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `aac297cb798be5e8a0b89e26792da7a6ba03be658f9641acbb41f3fd6a0816f2` |
| `v124_control_matrix.csv` | `ad08c83130627302930559cafbf1f2ea68952b2b7862332a5e96882b3a6b2faa` |
| `v124_efficiency_microbench.csv` | `d70a9611cbca50200fc2b9fef81c046a30345bd6fb7ac3e8f2a98e9563bc6e10` |
| `v124_expression_battery.csv` | `f45d7ef163b3c3112d8bbc53f0816190a3c34866d10e3967fa5fc31aabfd18e9` |
| `v124_provenance_audit.csv` | `214254f8bdf0808305813e612c4074ecd076dde25cff43a0f2348e1db5c37a25` |
| `v124_route_decision.json` | `bb8336e7b5979fdaaad2d0992476d6ac6d2cb31d5d2200c09d798e838c3e3724` |
| `v124_task_triage.csv` | `7d2cd6b679a65be7db639cddd6ae7459758629c8288c6684c59d6eecfa8c516b` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
