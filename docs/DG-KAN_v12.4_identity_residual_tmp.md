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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_identity_residual_20260520T050000Z
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
    "B1a-ReLU-KAN-local-hinge-K4",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B2b-FastKAN-RBF-K8",
    "B3a-ChebyKAN-K4",
    "B3b-LegendreKAN-K4",
    "B6a-BSpline-order1-local-K4",
    "B6b-BSpline-order1-local-K8-expression-repair",
    "B8d-BSpline-ReLU-lite-combo-K4-repair",
    "B9a-Poly2SignedPair-stream-K3-repair",
    "B9b-Poly2SignedPair-h64-diagnostic-repair",
    "B9c-Poly2SignedPair-h64-K2-diagnostic-repair",
    "B9d-Poly2PairRandom-h64-K2-diagnostic-repair"
  ],
  "A2_expression_pass": [
    "B3b-LegendreKAN-K4"
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.722791524094679` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `0.8796300922131879` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `1.4614031783248584` | `1.0742112810707456` | `1` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `1.853243162983404` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `2.0219289086187384` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `1.7454882501650046` | `1.1750887735591369` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.615832976741747` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `0.9117752179035766` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `1.3234541483212061` | `1.0923415733406172` | `1` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `1.4688537134604815` | `1.1100450696531003` | `1` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `1.5700997411750177` | `1.1100450696531003` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `0.9945558937529808` | `0.5075628243649276` | `1` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `2.0192795288088248` | `1.1272193389784213` | `0` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `OrthogonalPolynomial` | `0.7670751411652741` | `0.5418430756623873` | `1` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.6228427553674147` | `1.1451959847036328` | `0` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `0.744817558021511` | `0.5075628243649276` | `1` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.767676883649156` | `1.0854616225075115` | `0` |
| `B13b-ChebyKAN-K4-identity-residual-repair` | `OrthogonalPolynomial` | `1.0570103837194402` | `0.505736137667304` | `1` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `1.688485127570834` | `1.1626263316033871` | `0` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `3.0675961734490444` | `1.1358747609942639` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `1.5055388422104727` | `1.0910953291450423` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.7674836260073647` | `1.176437448784485` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `0.7904570816296324` | `0.5061458617863972` | `1` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.7346074229180353` | `1.3602157880360557` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `0.8301081007782126` | `0.5027656378038787` | `1` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `1.587302898461231` | `1.109976782299918` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `0.9867827660605076` | `0.5074774651734498` | `1` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `1.8059988065688979` | `1.1353626058453974` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `1.104586455643058` | `0.4798040152963671` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `1.710743331323595` | `0.9097753346080306` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `1.155511333716663` | `0.3053981152690522` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.2097508223844977` | `0.8576720841300192` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `0.9285769652902697` | `0.24465651461349358` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.4027970226903959` | `0.8511506419011199` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `0.9694490326412407` | `0.24482723299644907` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `0.6927526212196766` | `0.7632989620322316` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `0.8199957115918279` | `0.15120527178366566` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.0339229842852493` | `0.7644769188746244` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `0.7457278672989017` | `0.16595534007101884` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `0.8092363995865502` | `0.7867727396886097` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `QuadraticSketch` | `1.0951624459565927` | `0.1954554766457252` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `0.8327880145507985` | `0.815521715378312` | `1` |
| `B10d-QuadraticSketch-h512-diagnostic` | `QuadraticSketch` | `0.8820164579300022` | `0.25445574979513796` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `1.2333582300976125` | `0.8579623053810435` | `1` |
| `B11a-TrainableQuadraticSketch-h128` | `QuadraticSketch` | `0.8246505893768497` | `0.24661977601748156` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.2658783277442323` | `0.948784485113357` | `1` |
| `B11b-TrainableQuadraticSketch-h256` | `QuadraticSketch` | `1.130369718501068` | `0.3704930346899754` | `1` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `2.272137107422142` | `1.0881077574433216` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10a-QuadraticSketch-h64-diagnostic', 'B10b-QuadraticSketch-h128-diagnostic', 'B10c-QuadraticSketch-h256-diagnostic', 'B10d-QuadraticSketch-h512-diagnostic', 'B11a-TrainableQuadraticSketch-h128', 'B11b-TrainableQuadraticSketch-h256', 'B12a-LegendreKAN-K4-fan-scale-repair', 'B12b-ChebyKAN-K4-fan-scale-repair', 'B13a-LegendreKAN-K4-identity-residual-repair', 'B13b-ChebyKAN-K4-identity-residual-repair', 'B1a-ReLU-KAN-local-hinge-K4', 'B1r-ReLU-KAN-stream-K2-repair', 'B2b-FastKAN-RBF-K8', 'B3a-ChebyKAN-K4', 'B3b-LegendreKAN-K4', 'B6a-BSpline-order1-local-K4', 'B6b-BSpline-order1-local-K8-expression-repair', 'B8d-BSpline-ReLU-lite-combo-K4-repair', 'B9a-Poly2SignedPair-stream-K3-repair', 'B9b-Poly2SignedPair-h64-diagnostic-repair', 'B9c-Poly2SignedPair-h64-K2-diagnostic-repair', 'B9d-Poly2PairRandom-h64-K2-diagnostic-repair']
A2 expression pass = ['B3b-LegendreKAN-K4']
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B3b-LegendreKAN-K4` | `-0.3294270833333333` | `-0.4296875` | `0.3057285402384069` |

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
| `B1a-ReLU-KAN-local-hinge-K4` | `diagnostic_base_not_qualified` | `0.002096993869568742` | `0.005597837989548715` | `-0.0035008441199799734` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `diagnostic_base_not_qualified` | `-0.002065216998578867` | `0.0` | `-0.002065216998578867` | `0` |
| `B2b-FastKAN-RBF-K8` | `diagnostic_base_not_qualified` | `-0.013029543715944447` | `0.0` | `-0.013029543715944447` | `0` |
| `B3a-ChebyKAN-K4` | `diagnostic_base_not_qualified` | `0.0016249120845213127` | `0.0009381397969456806` | `0.0006867722875756321` | `1` |
| `B3b-LegendreKAN-K4` | `diagnostic_base_not_qualified` | `-0.006152881437051727` | `0.003726346148791748` | `-0.009879227585843475` | `0` |
| `B6a-BSpline-order1-local-K4` | `diagnostic_base_not_qualified` | `0.002269369158875989` | `0.006272526008597623` | `-0.004003156849721634` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `diagnostic_base_not_qualified` | `0.02617166891475997` | `0.07238561982052971` | `-0.04621395090576974` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `diagnostic_base_not_qualified` | `0.0014408970357095985` | `0.0037932873636172815` | `-0.002352390327907683` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `diagnostic_base_not_qualified` | `-4.055223265186925e-06` | `1.016035103074131e-05` | `-1.4215574295928235e-05` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `diagnostic_base_not_qualified` | `4.14654969960182e-05` | `1.1681385773343322e-05` | `2.978411122267488e-05` | `1` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `-7.393090586393924e-07` | `2.451584039064869e-05` | `-2.5255149449288083e-05` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `5.402795305364805e-06` | `0.0` | `5.402795305364805e-06` | `1` |

## 6. No-Fake / Hash

```text
rows_checked = 2090
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `437a559c4ba60be135943567a7935e6dc30a9a646f3c54e3e0a0188131c18045` |
| `v124_control_matrix.csv` | `c5ad8214d650d379a68040e941e11d2ac853794ee91a89e9ccd1e70180862602` |
| `v124_efficiency_microbench.csv` | `012e97ade36fbe36db9d46397ed2b5987588c8f674d89a353af554177bbcc688` |
| `v124_expression_battery.csv` | `4d45d6ad6b0aa95088a2e5c2cee4ac712886288ee5f9811fc672dd2460d1ef2b` |
| `v124_provenance_audit.csv` | `b34a9bf50449710e312f64799fb1097388ae9c8cf06cda4f4b87f2cae0cd616f` |
| `v124_route_decision.json` | `defd372c44e0eba09e721024ebee2c974daf72424832784dba73d99297994f0b` |
| `v124_task_triage.csv` | `75efc88b7ca9db9383698c9d073345dd5e0bebdf9b0f8e199df84c291b0e20c0` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
