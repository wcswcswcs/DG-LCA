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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_residual_mix_grid_20260520T150000Z
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
    "B10c-QuadraticSketch-h256-diagnostic",
    "B12a-LegendreKAN-K4-fan-scale-repair",
    "B15a-GatedLegendreQuadratic-h224-loss-gain",
    "B16a-GatedLegendreQuadratic-h224-basis-norm",
    "B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm",
    "B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm",
    "B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm",
    "B18a-GatedLegendreQuadratic-h224-group-rms-norm",
    "B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm",
    "B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm",
    "B19a-GatedLegendreQuadratic-h224-residual-geom-norm",
    "B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm",
    "B20b-GatedLegendreQuadratic-h224-residual-mix25",
    "B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25",
    "B2b-FastKAN-RBF-K8",
    "B3a-ChebyKAN-K4"
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.7384296063311135` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.578811863953962` | `0.443765364654466` | `0` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `2.103626662146233` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `2.684819334798835` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `2.3071616217640667` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `1.8920851681167534` | `1.1750887735591369` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.758231090371568` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `0.8789534206347064` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `1.4481055876982756` | `1.0923415733406172` | `1` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `1.5267297404581912` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.5307973722858825` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.4534758260429723` | `0.5075628243649276` | `1` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.8747073267105752` | `1.1272193389784213` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `2.0041703574609437` | `0.5418430756623873` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `2.880541991054758` | `1.1451959847036328` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `2.0131795092155884` | `0.5075628243649276` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `2.708494371241796` | `1.0854616225075115` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.6098190846183262` | `0.505736137667304` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `4.736494725436747` | `1.046571974870254` | `0` |
| `B14a-GatedLegendreQuadratic-h160` | `GatedHybrid` | `1.7742539389783962` | `0.3692980060092871` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `4.701160670774883` | `1.078803605572248` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `GatedHybrid` | `1.6527439949854728` | `0.4682293089319858` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `4.616431582908759` | `1.0960632340890466` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `GatedHybrid` | `1.80944311638185` | `0.4685878175361923` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `4.3857523413105035` | `1.1003141218246382` | `0` |
| `B14d-GatedLegendreQuadratic-h228-quad-boost` | `GatedHybrid` | `1.8859156409518505` | `0.4752116907948648` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `4.399084915510252` | `1.0978557771100792` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `GatedHybrid` | `1.0168809677527937` | `0.4693048347446053` | `1` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `2.9629632016312377` | `1.1010311390330512` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `GatedHybrid` | `1.903972911970728` | `0.4759287080032778` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `4.5691749107462085` | `1.074535645998361` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `GatedHybrid` | `1.0281648668327057` | `0.4700218519530183` | `1` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `3.042326675701468` | `1.077711007921333` | `0` |
| `B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm` | `GatedHybrid` | `1.9110173886067492` | `0.4766798688882819` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `3.9603678441930374` | `1.087527314941273` | `0` |
| `B17a-GatedLegendreQuadratic-h224-input-geom-norm` | `GatedHybrid` | `1.967716782927318` | `0.47790904124556133` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `3.971367219803137` | `1.094424337612674` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `GatedHybrid` | `1.2207375230431412` | `0.4845329145042338` | `1` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `3.0880850698640443` | `1.091829418191751` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `GatedHybrid` | `1.088712492255776` | `0.4786260584539743` | `1` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `3.670695962753732` | `1.095141354821087` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `GatedHybrid` | `1.012700299822387` | `0.4852840753892379` | `1` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `4.0870874561312265` | `1.113749658563234` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `GatedHybrid` | `1.1413025579971487` | `0.47230947828462166` | `1` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `3.2104433532193997` | `1.1135447965036875` | `0` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `GatedHybrid` | `1.0018699097742947` | `0.4789333515432942` | `1` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `3.017286866675623` | `1.1109498770827644` | `0` |
| `B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm` | `GatedHybrid` | `1.8509262288942423` | `0.4730264954930347` | `0` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `3.55476489248758` | `1.1142618137121005` | `0` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `GatedHybrid` | `1.2299337794936285` | `0.4796845124282983` | `1` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `GatedHybrid` | `4.452131977648115` | `1.1170274515159793` | `0` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `GatedHybrid` | `1.251428000020621` | `0.4809136847855777` | `1` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `GatedHybrid` | `3.9278172372247977` | `1.1239244741873804` | `0` |
| `B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm` | `GatedHybrid` | `1.6629964680577112` | `0.4875375580442502` | `0` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `GatedHybrid` | `5.555343946647566` | `1.1213295547664572` | `0` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `GatedHybrid` | `1.2275231765707317` | `0.4816307019939907` | `1` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `GatedHybrid` | `4.982268570141063` | `1.1246414913957934` | `0` |
| `B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm` | `GatedHybrid` | `1.7409311081722905` | `0.4882887189292543` | `0` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `GatedHybrid` | `5.713027869502207` | `1.1220807156514614` | `0` |
| `B20a-GatedLegendreQuadratic-h224-residual-mix15` | `GatedHybrid` | `1.8504636223893665` | `0.4824160065555859` | `0` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `GatedHybrid` | `5.8653921207998865` | `1.1213978421196393` | `0` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `GatedHybrid` | `0.8934151428652006` | `0.4827745151597924` | `1` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `GatedHybrid` | `4.046107812940185` | `1.125785304561595` | `0` |
| `B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15` | `GatedHybrid` | `1.8515823841383738` | `0.4893983884184649` | `0` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `GatedHybrid` | `5.994285573924701` | `1.1272193389784213` | `0` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `GatedHybrid` | `1.0644790225917549` | `0.4897568970226714` | `1` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `2.013573733047996` | `1.1452642720568151` | `0` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `3.8635233482097235` | `1.1185127014476919` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `2.39825780358495` | `1.0919147773832287` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `2.0473112570008847` | `1.1766423108440316` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.614074427640064` | `0.5145281343895111` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `2.7595048882238973` | `1.31967017208413` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.5535681957193963` | `0.5128380223982518` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `3.2500360866431204` | `1.1359259765091505` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `1.5168396500626664` | `0.5509765091505053` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `3.0193761771751073` | `1.085444550669216` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `1.7299122488073975` | `0.47845534007101886` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `3.0027779892097906` | `0.9185673313302376` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `1.6197833194192537` | `0.31419011199125924` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `2.5378984806916747` | `0.8664640808522261` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.568170322284048` | `0.2534485113357006` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.8589114600564831` | `0.859942638623327` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.5860741492573884` | `0.2536192297186561` | `0` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `1.5657129431871906` | `0.7720909587544387` | `0` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `1.5271609606810248` | `0.1599972685058727` | `0` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.5945350266358849` | `0.7732689155968314` | `0` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.677269569346853` | `0.1747473367932259` | `0` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.1051962794974193` | `0.7955647364108167` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.5745819180445044` | `0.20424747336793225` | `0` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.5990629906001876` | `0.824313712100519` | `0` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `1.5598652391316281` | `0.263247746517345` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `2.0933500807703984` | `0.8667543021032504` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.6113561785033383` | `0.2554117727396886` | `0` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `2.077545116643251` | `0.9575764818355641` | `0` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.606311174820378` | `0.37928503141218245` | `0` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `2.922869955564909` | `1.0968997541655285` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10c-QuadraticSketch-h256-diagnostic', 'B12a-LegendreKAN-K4-fan-scale-repair', 'B15a-GatedLegendreQuadratic-h224-loss-gain', 'B16a-GatedLegendreQuadratic-h224-basis-norm', 'B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm', 'B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm', 'B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm', 'B18a-GatedLegendreQuadratic-h224-group-rms-norm', 'B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm', 'B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm', 'B19a-GatedLegendreQuadratic-h224-residual-geom-norm', 'B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm', 'B20b-GatedLegendreQuadratic-h224-residual-mix25', 'B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25', 'B2b-FastKAN-RBF-K8', 'B3a-ChebyKAN-K4']
A2 expression pass = []
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B10c-QuadraticSketch-h256-diagnostic` | `-0.06684027777777778` | `-0.12109375` | `0.10396126616332266` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `-0.2092013888888889` | `-0.32421875` | `0.3050468609564834` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `-0.0010850694444444445` | `-0.01171875` | `0.013859681785106659` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B10c-QuadraticSketch-h256-diagnostic` | `diagnostic_base_not_qualified` | `-0.0019781291484832764` | `0.0` | `-0.0019781291484832764` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `diagnostic_base_not_qualified` | `2.5516345816344996` | `0.02923315749439226` | `2.5224014241401074` | `1` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `diagnostic_base_not_qualified` | `-0.00020563623901370676` | `0.0006638135429084535` | `-0.0008694497819221603` | `0` |
| `B16a-GatedLegendreQuadratic-h224-basis-norm` | `diagnostic_base_not_qualified` | `-0.00013499184002574438` | `0.00449900838266748` | `-0.004634000222693224` | `0` |
| `B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm` | `diagnostic_base_not_qualified` | `-7.95033345211138e-05` | `0.00042811441086065827` | `-0.0005076177453817721` | `0` |
| `B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm` | `diagnostic_base_not_qualified` | `-0.00012400512670041053` | `0.0014595131279211415` | `-0.001583518254621552` | `0` |
| `B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm` | `diagnostic_base_not_qualified` | `-0.00023029879443514645` | `0.0029624343609584436` | `-0.00319273315539359` | `0` |
| `B18a-GatedLegendreQuadratic-h224-group-rms-norm` | `diagnostic_base_not_qualified` | `-0.002098617777624767` | `6.056698130052496e-05` | `-0.002159184758925292` | `0` |
| `B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm` | `diagnostic_base_not_qualified` | `-0.00010431500322383158` | `0.0` | `-0.00010431500322383158` | `0` |
| `B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm` | `diagnostic_base_not_qualified` | `-0.00014436169623310846` | `0.006889200536788209` | `-0.007033562233021318` | `0` |
| `B19a-GatedLegendreQuadratic-h224-residual-geom-norm` | `diagnostic_base_not_qualified` | `-0.004002655797295862` | `0.0` | `-0.004002655797295862` | `0` |
| `B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm` | `diagnostic_base_not_qualified` | `0.003768859952016257` | `0.0054622161295045935` | `-0.0016933561774883366` | `0` |
| `B20b-GatedLegendreQuadratic-h224-residual-mix25` | `diagnostic_base_not_qualified` | `-0.002067623863291068` | `0.0` | `-0.002067623863291068` | `0` |
| `B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25` | `diagnostic_base_not_qualified` | `-6.60671930798884e-05` | `2.7757184865917495e-05` | `-9.382437794580589e-05` | `0` |
| `B2b-FastKAN-RBF-K8` | `diagnostic_base_not_qualified` | `-0.013029543715944447` | `0.0` | `-0.013029543715944447` | `0` |
| `B3a-ChebyKAN-K4` | `diagnostic_base_not_qualified` | `0.0016249120845213127` | `0.0009381397969456806` | `0.0006867722875756321` | `1` |

## 6. No-Fake / Hash

```text
rows_checked = 2046
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `dd711b0105f7ab191d609371722a2d5f4ee85e5328e68fc0303b4761a223aed6` |
| `v124_control_matrix.csv` | `ca3119c03e34fe9ed2f3569e5e64cd408c110beb4dfd509bfaebaa5fd2a122b2` |
| `v124_efficiency_microbench.csv` | `da2234f799582d253e116f50e5eadb9a786621c2a6fdce014ad585801de609c9` |
| `v124_expression_battery.csv` | `e7f67d40f18146432a2e6c0845e595358c2749df285e2f50b1caaa86a97329dd` |
| `v124_provenance_audit.csv` | `e24a30dc98527decdd26b553bfdfd15908a3170b620ac0ae001e59ea0fedaf9d` |
| `v124_route_decision.json` | `028aa19ea775b5911614cd25beadcff6a60e9e896ecb00eec74c34e20201f5b8` |
| `v124_task_triage.csv` | `5fd718347ddfdf213dfe78434dcbb57164939b56d9c0c7e99bbe663627d3bfc0` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
