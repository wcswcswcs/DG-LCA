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
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_quadratic_sketch_scale_repair_20260520T003000Z
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
    "B10b-QuadraticSketch-h128-diagnostic",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B2b-FastKAN-RBF-K8",
    "B6a-BSpline-order1-local-K4",
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
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.9219063828122456` | `1.017242556678503` | `0` |
| `B1r-ReLU-KAN-stream-K2-repair` | `Activation` | `1.2256174787597542` | `0.443765364654466` | `1` |
| `B1a-ReLU-KAN-local-hinge-K4` | `Activation` | `2.014126356099236` | `1.0742112810707456` | `0` |
| `B1b-RSWAF-hinge-K8` | `Activation` | `2.8709123756209824` | `1.214336929800601` | `0` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `RBF` | `2.544827492882627` | `1.1265364654465992` | `0` |
| `B2a-GaussianRBF-K4-compact` | `RBF` | `1.7219844665818749` | `1.1750887735591369` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.9735582337044009` | `1.3009082217973231` | `0` |
| `B2b-FastKAN-RBF-K8` | `RBF` | `1.23555085165303` | `0.506111718109806` | `1` |
| `B3a-ChebyKAN-K4` | `OrthogonalPolynomial` | `2.863846316993041` | `1.0923415733406172` | `0` |
| `B3b-LegendreKAN-K4` | `OrthogonalPolynomial` | `3.0240074096513307` | `1.1100450696531003` | `0` |
| `B4a-FourierKAN-lowfreq-K4` | `Fourier` | `1.9523803558492288` | `1.1635823545479378` | `0` |
| `B5c-RickerWaveletKAN-lite-K4` | `Wavelet` | `3.280240411727089` | `1.1368307839388145` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `BSpline` | `1.8903295502184618` | `1.0921025676044795` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.6094220501437677` | `1.1769837476099427` | `0` |
| `B6a-BSpline-order1-local-K4` | `BSpline` | `1.4978845836673353` | `0.5075628243649276` | `1` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `2.6204291418596397` | `1.310878175361923` | `0` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `BSpline` | `1.851241375180053` | `0.5061458617863972` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `3.4788048386748858` | `1.1455032777929528` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `Hybrid` | `1.9210339695213376` | `0.577711007921333` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `3.078011956141788` | `1.1376160885004096` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `HybridPoly` | `1.5794521113232318` | `0.515023217700082` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `2.646075935528102` | `0.9338124829281617` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `HybridPoly` | `1.791334188753445` | `0.3053981152690522` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `2.4386140720404503` | `0.8576720841300192` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `HybridPoly` | `1.5258579467586058` | `0.24465651461349358` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.715391559691143` | `0.8511506419011199` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `HybridPoly` | `1.4605619351805568` | `0.24482723299644907` | `1` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `1.5242438850346454` | `0.7632989620322316` | `0` |
| `B10a-QuadraticSketch-h64-diagnostic` | `QuadraticSketch` | `1.5155892879464947` | `0.15120527178366566` | `0` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `1.3934256359948494` | `0.7644769188746244` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `QuadraticSketch` | `0.6451118126621916` | `0.16595534007101884` | `1` |
| `B7a-RationalKAT-lite-safe-den-K4` | `Rational` | `1.6097448088964048` | `1.0740405626877902` | `0` |

## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10b-QuadraticSketch-h128-diagnostic', 'B1r-ReLU-KAN-stream-K2-repair', 'B2b-FastKAN-RBF-K8', 'B6a-BSpline-order1-local-K4', 'B9d-Poly2PairRandom-h64-K2-diagnostic-repair']
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
| `B6a-BSpline-order1-local-K4` | `diagnostic_base_not_qualified` | `0.002269369158875989` | `0.006272526008597623` | `-0.004003156849721634` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `5.402795305364805e-06` | `0.0` | `5.402795305364805e-06` | `1` |

## 6. No-Fake / Hash

```text
rows_checked = 987
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `3abf969c06db1663cd4467ab5c19aa08f1bef7c56920e69e3f3b713459380d0f` |
| `v124_control_matrix.csv` | `06eeabc834ab3e42be0208fc0eae496c5eb01eecb527f66dfe9299441093810f` |
| `v124_efficiency_microbench.csv` | `2b8f42ec34e3279e9d6e0636e501304701b6af3f70dfd9bad818ab94d026ce3f` |
| `v124_expression_battery.csv` | `3bcfe86213aa9d03cdac4da30bf9cbfe83387fc4ee9ed59fa1f73ccb379fe1ad` |
| `v124_provenance_audit.csv` | `edc94caac8548dc5083014c4840f155cb8c78c2f7cba780e612b468162d32ef4` |
| `v124_route_decision.json` | `5b37e9365a01003bb82a61858623e68ce155ffc546de4379a831cf07b23798a3` |
| `v124_task_triage.csv` | `8fd75d6aeb9daaa625ee8d9e3e65fc971c09b8f6077df0f49988c26b1645e209` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
