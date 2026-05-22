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
results/v12_4_multibasis_functional_dual/smoke_b11_trainable_quadratic_sketch
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
    "B1a-ReLU-KAN-local-hinge-K4",
    "B1b-RSWAF-hinge-K8",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B2a-GaussianRBF-K4-compact",
    "B2b-FastKAN-RBF-K8",
    "B2r-FastKAN-RBF-stream-K2-repair",
    "B3a-ChebyKAN-K4",
    "B3b-LegendreKAN-K4",
    "B4a-FourierKAN-lowfreq-K4",
    "B5c-RickerWaveletKAN-lite-K4",
    "B6a-BSpline-order1-local-K4",
    "B6r-BSpline-order1-stream-K2-repair",
    "B8d-BSpline-ReLU-lite-combo-K4-repair",
    "B9a-Poly2SignedPair-stream-K3-repair",
    "B9b-Poly2SignedPair-h64-diagnostic-repair",
    "B9c-Poly2SignedPair-h64-K2-diagnostic-repair",
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


## 4. A2/A3 Survivors

```text
A1 exploratory survivors = ['B10a-QuadraticSketch-h64-diagnostic', 'B10b-QuadraticSketch-h128-diagnostic', 'B10c-QuadraticSketch-h256-diagnostic', 'B10d-QuadraticSketch-h512-diagnostic', 'B11a-TrainableQuadraticSketch-h128', 'B11b-TrainableQuadraticSketch-h256', 'B1a-ReLU-KAN-local-hinge-K4', 'B1b-RSWAF-hinge-K8', 'B1r-ReLU-KAN-stream-K2-repair', 'B2a-GaussianRBF-K4-compact', 'B2b-FastKAN-RBF-K8', 'B2r-FastKAN-RBF-stream-K2-repair', 'B3a-ChebyKAN-K4', 'B3b-LegendreKAN-K4', 'B4a-FourierKAN-lowfreq-K4', 'B5c-RickerWaveletKAN-lite-K4', 'B6a-BSpline-order1-local-K4', 'B6r-BSpline-order1-stream-K2-repair', 'B8d-BSpline-ReLU-lite-combo-K4-repair', 'B9a-Poly2SignedPair-stream-K3-repair', 'B9b-Poly2SignedPair-h64-diagnostic-repair', 'B9c-Poly2SignedPair-h64-K2-diagnostic-repair', 'B9d-Poly2PairRandom-h64-K2-diagnostic-repair']
A2 expression pass = []
A3 task pass = []
```

Task triage summary：

| candidate | mean val acc delta vs MLP | worst row delta | mean ECE delta |
|---|---:|---:|---:|
| `B10a-QuadraticSketch-h64-diagnostic` | `-0.09375` | `-0.09375` | `0.011772505939006805` |
| `B10b-QuadraticSketch-h128-diagnostic` | `-0.21875` | `-0.21875` | `-0.06995096430182457` |
| `B10c-QuadraticSketch-h256-diagnostic` | `-0.1875` | `-0.1875` | `-0.04039578139781952` |

## 5. Functional Diagnostic

| candidate | status | best functional score | best control score | control gap | beats controls |
|---|---|---:|---:|---:|---:|
| `B10a-QuadraticSketch-h64-diagnostic` | `diagnostic_base_not_qualified` | `0.0166427344083786` | `0.016641258448362173` | `1.475960016428246e-06` | `1` |
| `B10b-QuadraticSketch-h128-diagnostic` | `diagnostic_base_not_qualified` | `0.00010989606380462646` | `4.056096076787696e-06` | `0.00010583996772783877` | `1` |
| `B10c-QuadraticSketch-h256-diagnostic` | `diagnostic_base_not_qualified` | `1.8504261970697655e-05` | `0.0002768069505689752` | `-0.00025830268859827754` | `0` |
| `B10d-QuadraticSketch-h512-diagnostic` | `diagnostic_base_not_qualified` | `6.693750619835015e-05` | `0.00039191842079144834` | `-0.0003249809145930982` | `0` |
| `B11a-TrainableQuadraticSketch-h128` | `diagnostic_base_not_qualified` | `-8.545219898170586e-05` | `0.017781123715277403` | `-0.01786657591425911` | `0` |
| `B11b-TrainableQuadraticSketch-h256` | `diagnostic_base_not_qualified` | `0.024887384123107203` | `0.07554710970275202` | `-0.05065972557964482` | `0` |
| `B1a-ReLU-KAN-local-hinge-K4` | `diagnostic_base_not_qualified` | `0.008508060576069454` | `0.0419110891304757` | `-0.033403028554406244` | `0` |
| `B1b-RSWAF-hinge-K8` | `diagnostic_base_not_qualified` | `0.04703033000897694` | `0.040793497313726945` | `0.006236832695249994` | `1` |
| `B1r-ReLU-KAN-stream-K2-repair` | `diagnostic_base_not_qualified` | `0.040246786130768264` | `0.0551485200918993` | `-0.014901733961131036` | `0` |
| `B2a-GaussianRBF-K4-compact` | `diagnostic_base_not_qualified` | `0.024343983339560005` | `0.05816595074328279` | `-0.033821967403722786` | `0` |
| `B2b-FastKAN-RBF-K8` | `diagnostic_base_not_qualified` | `0.051331550862571795` | `0.007549709524673265` | `0.04378184133789853` | `1` |
| `B2r-FastKAN-RBF-stream-K2-repair` | `diagnostic_base_not_qualified` | `0.01811394733605365` | `0.006070552788147943` | `0.012043394547905706` | `1` |
| `B3a-ChebyKAN-K4` | `diagnostic_base_not_qualified` | `0.018084498680498884` | `0.030471011106265156` | `-0.012386512425766272` | `0` |
| `B3b-LegendreKAN-K4` | `diagnostic_base_not_qualified` | `0.001505215506917068` | `0.0` | `0.001505215506917068` | `1` |
| `B4a-FourierKAN-lowfreq-K4` | `diagnostic_base_not_qualified` | `0.02821786804079096` | `0.04423261131462142` | `-0.016014743273830456` | `0` |
| `B5c-RickerWaveletKAN-lite-K4` | `diagnostic_base_not_qualified` | `0.028047908409426725` | `0.021430940066271553` | `0.006616968343155172` | `1` |
| `B6a-BSpline-order1-local-K4` | `diagnostic_base_not_qualified` | `0.016291990105909093` | `0.0376115843036855` | `-0.021319594197776404` | `0` |
| `B6r-BSpline-order1-stream-K2-repair` | `diagnostic_base_not_qualified` | `0.022254701659678133` | `0.028276027545206084` | `-0.0060213258855279506` | `0` |
| `B8d-BSpline-ReLU-lite-combo-K4-repair` | `diagnostic_base_not_qualified` | `0.04376092278161092` | `0.05073540173203339` | `-0.006974478950422469` | `0` |
| `B9a-Poly2SignedPair-stream-K3-repair` | `diagnostic_base_not_qualified` | `0.028461294461661346` | `0.029036888680612938` | `-0.0005755942189515917` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `diagnostic_base_not_qualified` | `-0.0019951099254882365` | `0.0` | `-0.0019951099254882365` | `0` |
| `B9c-Poly2SignedPair-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `-0.014495285263127045` | `0.0` | `-0.014495285263127045` | `0` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `-0.015432561070102935` | `0.030110018374128522` | `-0.04554257944423146` | `0` |

## 6. No-Fake / Hash

```text
rows_checked = 1097
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `b93ed092ea63eb654b6bea198399f32177abb7eb0e72c68269a2895eb9af4167` |
| `v124_control_matrix.csv` | `193bb3e2f2d4f57f566683246bf44b528da955fe7372473c3607616e10dc549c` |
| `v124_efficiency_microbench.csv` | `9c40242624211421ab6652a5e89dc99f21406eec839db6da7adfd3017aa82c47` |
| `v124_expression_battery.csv` | `eac9ee65bb9729433c628fd468d481b8ec9837b3aef9623115db5bd962fe1bcb` |
| `v124_provenance_audit.csv` | `3f18d3d9e5fd0632d10422cf3bd59c635627aaf40fa643700659eef9efebade9` |
| `v124_route_decision.json` | `6297ca6673c09b6c2901adccf06e16038fd55ff3271ced8ab085784b01235d15` |
| `v124_task_triage.csv` | `806ef77a056c8fba40a062cd26ae44d76910976484ff300105f03693b2c7d56e` |

## 7. 最终分析结论

```text
1. v12.4 已按 plan 从 LQ frame 小修转向多 basis primitive-level screen。
2. Efficiency、expression、task、functional diagnostic 均分线落盘；functional 没有越过 base gate。
3. 若 route 是 R2/R3，下一步按 route action 做全局 basis/initialization repair，而不是 dataset-specific tuning。
4. 若 route 是 R1，应优先 fused/manual kernel；若 route 是 R4/R5，应继续 functional target repair但不写 official success。
```
