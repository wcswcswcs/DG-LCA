# DG-KAN v12.13 B320Locked FunctionalMechanism NoBSpline 执行复盘

生成时间：`2026-05-23T21:58:37Z`

## 1. 执行入口

```text
script = experiments/run_v1213_b320locked_functional_mechanism_nobspline.py
command = experiments/run_v1213_b320locked_functional_mechanism_nobspline.py --aggregate-input-dirs results/v12_13_b320locked_functional_mechanism_nobspline/batch1_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch1_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch1_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1ext_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1ext_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1ext_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1role_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1role_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1role_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm5_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm5_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm5_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch3_bm2_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch3_bm2_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch3_bm2_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch4_bm2signed_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch4_bm2signed_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch4_bm2signed_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch5_noisecontrast_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch5_noisecontrast_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch5_noisecontrast_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_KMNIST_gpu2 --out-dir results/v12_13_b320locked_functional_mechanism_nobspline/batch11_aggregate_lossagnostic_official --report-path docs/DG-KAN_v12.13_B320Locked_FunctionalMechanism_NoBSpline_执行复盘.md
plan_doc = docs/DG-KAN_v12.13_B320Locked_FunctionalMechanism_NoBSpline_独立分析与下一步计划.md
out_dir = results/v12_13_b320locked_functional_mechanism_nobspline/batch11_aggregate_lossagnostic_official
```

本轮按 v12.13 计划先执行 Batch 1：B320 anchor monitor、Line C calibration/null distribution、No-BSpline family status refresh。没有 fake/proxy/CPU offload；没有 teacher/distillation/loss modification/sampler/class weight/dataset-name branch。

## 2. 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `experiments/run_v1213_b320locked_functional_mechanism_nobspline.py` | 新增 v12.13 runner | 复用真实 v12.10/v12.12 artifacts，新增 Line C repeated calibration / null distribution / BM1b/BM3a repeated probe；不改模型、不改 loss、不进入 P4。 |
| `docs/DG-KAN_v12.13_B320Locked_FunctionalMechanism_NoBSpline_执行复盘.md` | 新增 v12.13 复盘文件 | 记录命令、source、指标、route 和 blocker，防止后续复现再次读全代码。 |

## 3. B320 Anchor Monitor

```text
B320_anchor_locked = 1
step_ratio_q90 = 0.39931987348441034
memory_ratio_q90 = 0.1295238095238095
mean_delta = 0.027669270833333332
worst_delta = -0.001953125
near_pass_rate = 1.0
AUC_step_ratio = 0.9336784156141542
AUC_time_ratio = 0.7424718100091173
ECE_delta = 0.01767905056476593
LineC_nontearing_pass = 1
```

## 4. Line C Calibration

```text
calibration_rows = 3240
linec_calibration_pass = 1
calibration_fail_reason = 
noop_false_positive_rows = 0
random_false_positive_rows = 0
bm1_full_3x3_pass_candidate_count = 0
bm2_full_3x3_pass_candidate_count = 0
bm3_full_3x3_pass_candidate_count = 0
bm5_full_3x3_pass_candidate_count = 0
bm1_partial_pass_rows = 1
bm2_partial_pass_rows = 2
bm3_partial_pass_rows = 0
bm5_partial_pass_rows = 0
fail_reason_counts = {"CouplingR2_delta<0.02": 1111, "NoiseSignalLeak_delta>-0.01": 3145, "RealSignalReservoirRatio_delta>-0.01": 3076, "control_gap<0.005": 2757, "logit_max_abs_drift>0.05": 691, "holdout_loss_ratio>1.002": 120}
```

解释：若是单次运行，runner 参数为 `methods=NoOp,TaskOnlyAdamW,AdamWParallelDirection,RandomMatchedNorm,BM1b-SoftSNRGate,BM3a-BranchRebalance`、`functional_batch_sizes=16`、`windows=5`、`probe_splits=5`；若是 aggregate 报告，则以三数据集 shard 的 CSV 汇总表为准。若 NoOp/Random false positive 或 NoOp noise/reservoir 方差超过计划阈值，route 必须是 `R2-LineCMetricNotReliable`，不能把 BM partial pass 推进 P4。

如果这是 aggregate 报告，实际执行矩阵以 `v1213_linec_null_distribution.csv` 为准：

| method | batch | window | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `AdamWParallelDirection` | `16` | `5` | `45` | `0.40077063390794115` | `-0.00027851429250505235` | `-0.002844991617732578` | `0` |
| `AdamWParallelDirection` | `32` | `3` | `45` | `0.17350704117198365` | `0.001292883785855439` | `-0.0013407664166556464` | `0` |
| `AdamWParallelDirection` | `32` | `10` | `45` | `0.2829341051942559` | `-0.0020149900681442684` | `-0.004780732095241547` | `0` |
| `BM1b-SoftSNRGate` | `16` | `5` | `45` | `0.3763169309966251` | `-0.00010306388139724731` | `-0.0031816853417290583` | `1` |
| `BM1b-SoftSNRGate` | `32` | `3` | `45` | `0.1593876228496969` | `0.001174013859902819` | `-0.0013699978590011597` | `0` |
| `BM1b-SoftSNRGate` | `32` | `5` | `45` | `0.2136147570950757` | `0.00017672176990244124` | `-0.002479976746771071` | `0` |
| `BM1b-SoftSNRGate` | `32` | `10` | `45` | `0.2752052408534761` | `-0.00198145281109545` | `-0.004790632923444112` | `0` |
| `BM1h-RoleWiseSNR-direct` | `32` | `5` | `45` | `0.015868592610844685` | `-3.99228185415268e-05` | `-0.0005676911936865912` | `0` |
| `BM1i-RoleWiseSNR-quad` | `32` | `5` | `45` | `0.21774649796769718` | `-0.001746921158499188` | `-0.001116353604528639` | `0` |
| `BM1j-RoleWiseSNR-branch` | `32` | `5` | `45` | `0.11547631639673457` | `-0.0006927752453419897` | `-0.003169774015744527` | `0` |
| `BM2e-LowRankLineCGrid-balanced` | `32` | `5` | `45` | `0.20165167372544876` | `0.00013725755529271232` | `-0.002332194315062629` | `0` |
| `BM2g-LineCGrid-withNoiseVeto` | `32` | `5` | `45` | `0.2120803797765593` | `0.0001860047173168924` | `-0.0021824253929985894` | `0` |
| `BM2h-LineCGrid-withReservoirVeto` | `32` | `5` | `45` | `0.18689091708024527` | `-0.0001542907829085986` | `-0.0029072622458140057` | `0` |
| `BM2i-LineCGrid-signed` | `32` | `5` | `45` | `0.21702432380318065` | `1.1947212947739496e-05` | `-0.0021996693478690253` | `0` |
| `BM2j-LineCGrid-signedAdamWOrthogonal` | `32` | `5` | `45` | `0.1983931336065015` | `-0.0007064087523354425` | `-0.0016072054704030355` | `0` |
| `BM2k-LineCGrid-noiseContrast` | `32` | `5` | `45` | `0.2541957760602528` | `-0.00029134597215387556` | `-0.0028791122966342502` | `0` |
| `BM2l-LineCGrid-noiseContrastOrth` | `32` | `5` | `45` | `0.2571067341777921` | `-0.0005828314564294286` | `-0.0007246166467666626` | `0` |
| `BM2s-LogitCotangentTail` | `32` | `5` | `45` | `0.1570992316187467` | `-0.002331516539884938` | `-0.00021885534127553304` | `0` |
| `BM2t-LogitCotangentTailFlip` | `32` | `5` | `45` | `0.15743122558542647` | `9.472303920321994e-05` | `0.0008008400599161784` | `0` |
| `BM2u-LogitCotangentBalanced` | `32` | `5` | `45` | `0.1682587748191264` | `-0.00013036612007353042` | `-0.0002749005953470866` | `0` |
| `BM2v-LogitCotangentBalancedOrth` | `32` | `5` | `45` | `0.17148056130813485` | `-0.00016539345184961955` | `-0.00030705398983425563` | `0` |
| `BM2w-LogitCotangentSpectralTopDamp` | `32` | `5` | `45` | `0.14061654825032185` | `9.042728278372023e-05` | `0.0009154842959509955` | `0` |
| `BM2x-LogitCotangentSpectralTopDampOrth` | `32` | `5` | `45` | `0.14608559390755904` | `2.8124203284581504e-05` | `0.0006641719076368544` | `0` |
| `BM2y-LogitCotangentSpectralIso` | `32` | `3` | `45` | `0.10537258410245963` | `4.794582103689512e-05` | `4.6187970373365615e-05` | `0` |
| `BM2y-LogitCotangentSpectralIso` | `32` | `5` | `45` | `0.14470738751521395` | `-0.00048750667936272094` | `0.0002517670392990112` | `1` |
| `BM2y-LogitCotangentSpectralIso` | `32` | `10` | `45` | `0.20010131891374985` | `-0.0006749643219841851` | `0.0014614009194903904` | `0` |
| `BM2z-LogitCotangentSpectralIsoOrth` | `32` | `3` | `45` | `0.10838210872268408` | `3.224128029412694e-05` | `-1.4816721280415853e-05` | `0` |
| `BM2z-LogitCotangentSpectralIsoOrth` | `32` | `5` | `45` | `0.15019762282478877` | `-0.0005318532387415568` | `4.866222540537516e-05` | `1` |
| `BM2z-LogitCotangentSpectralIsoOrth` | `32` | `10` | `45` | `0.20998919708733116` | `-0.0008394607653220494` | `0.0012610709501637352` | `0` |
| `BM3a-BranchRebalance` | `16` | `5` | `45` | `0.6646413285593893` | `0.000426774885919359` | `0.007577203048600091` | `0` |
| `BM3a-BranchRebalance` | `32` | `3` | `45` | `0.3173873356126252` | `0.0013470250699255202` | `0.0037208212746514214` | `0` |
| `BM3a-BranchRebalance` | `32` | `10` | `45` | `0.323897284170842` | `0.0014847979363467958` | `0.014852123790317112` | `0` |
| `BM5a-ReservoirVJP` | `32` | `5` | `45` | `0.21163811196231413` | `0.0004035065985388226` | `0.00071099234951867` | `0` |
| `BM5b-ReservoirVJP-AdamWOrthogonal` | `32` | `5` | `45` | `0.21332445317792942` | `0.0003549836162063811` | `0.0009675347142749362` | `0` |
| `BM5c-NoiseVetoVJP` | `32` | `5` | `45` | `0.2542753834003061` | `-0.00041285134438011383` | `-0.002257476250330607` | `0` |
| `BM5d-NoiseVetoVJP-AdamWOrthogonal` | `32` | `5` | `45` | `0.25625326788018277` | `-0.0006122012519174152` | `-0.0012898875607384576` | `0` |
| `NoOp` | `16` | `5` | `45` | `0.0` | `0.0` | `0.0` | `0` |
| `NoOp` | `32` | `3` | `90` | `0.0` | `0.0` | `0.0` | `0` |
| `NoOp` | `32` | `5` | `315` | `0.0` | `0.0` | `0.0` | `0` |
| `NoOp` | `32` | `10` | `90` | `0.0` | `0.0` | `0.0` | `0` |
| `RandomMatchedNorm` | `16` | `5` | `45` | `0.006147675907033041` | `-5.200770166185167e-05` | `1.4420681529574925e-05` | `0` |
| `RandomMatchedNorm` | `32` | `3` | `90` | `0.0012402161738465584` | `1.059751957654953e-06` | `-4.595849249098036e-06` | `0` |
| `RandomMatchedNorm` | `32` | `5` | `315` | `0.003406223269465952` | `-1.556078592936198e-05` | `-3.55574819776747e-06` | `0` |
| `RandomMatchedNorm` | `32` | `10` | `90` | `0.01295159033381295` | `2.9197438723511166e-05` | `-3.219727012846205e-05` | `0` |
| `TaskOnlyAdamW` | `16` | `5` | `45` | `0.40077063390794115` | `-0.00027851429250505235` | `-0.002844991617732578` | `0` |
| `TaskOnlyAdamW` | `32` | `3` | `90` | `0.1735070411719837` | `0.001292883785855439` | `-0.0013407664166556464` | `0` |
| `TaskOnlyAdamW` | `32` | `5` | `315` | `0.22580067458671163` | `-0.001761377209590541` | `-0.0021269884374406603` | `0` |
| `TaskOnlyAdamW` | `32` | `10` | `90` | `0.28293410519425594` | `-0.0020149900681442684` | `-0.004780732095241547` | `0` |

BM1-ext 追加了 `window=3/10`、`batch=32` 和 role-wise direct/quad/branch；BM5 追加了 logits-vs-onehot residual VJP 及 AdamW-orthogonal 版本。二者都没有形成 3x3 survivor，因此 P4 按计划保持关闭。

## 5. No-BSpline Family Status

| family | status | best_task_candidate | blocker |
|---|---|---|---|
| `Rational` | `TaskBlocked` | `B7kd-RationalKAT-flashgroup-G16-h32-linearresGain14400-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3` | `official L3 and A4 pass exist, but A5 task gate failed` |
| `Chebyshev` | `TaskBlocked` | `B3ao-ChebyKAN-K3-h112-inputcrossL4P128-linearraw010-tritonL3-gradbuf` | `official L3 and A4 pass exist, but A5 task gate failed` |
| `Wavelet` | `TaskBlocked` | `B5n-HatWaveletKAN-K4-inputcrossL4P128-linearraw010-tritonL3` | `official L3 and A4 pass exist, but A5 task gate failed` |
| `RBF` | `ExpressionBlocked` | `` | `official L3 pass exists, but A4 expression gate failed` |
| `Fourier` | `ExpressionBlocked` | `` | `official L3 pass exists, but A4 expression gate failed` |
| `BSpline` | `RejectedForThisVersion` | `` | `B-spline frozen by v12.12 plan` |

## 6. Route

```text
route = R3-FunctionalMechanismPartialOnly
official_functional_success = 0
next_recommended_action = continue constrained BM2/BM5 noise-reservoir solve; no P4
```

## 7. 复现命令

```text
conda run -n kan python experiments/run_v1213_b320locked_functional_mechanism_nobspline.py \
  --out-dir results/v12_13_b320locked_functional_mechanism_nobspline/batch11_aggregate_lossagnostic_official \
  --report-path docs/DG-KAN_v12.13_B320Locked_FunctionalMechanism_NoBSpline_执行复盘.md
```

本轮完整命令：

```text
experiments/run_v1213_b320locked_functional_mechanism_nobspline.py --aggregate-input-dirs results/v12_13_b320locked_functional_mechanism_nobspline/batch1_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch1_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch1_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1ext_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1ext_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1ext_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1role_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1role_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1role_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm5_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm5_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm5_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch3_bm2_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch3_bm2_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch3_bm2_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch4_bm2signed_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch4_bm2signed_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch4_bm2signed_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch5_noisecontrast_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch5_noisecontrast_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch5_noisecontrast_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_KMNIST_gpu2 --out-dir results/v12_13_b320locked_functional_mechanism_nobspline/batch11_aggregate_lossagnostic_official --report-path docs/DG-KAN_v12.13_B320Locked_FunctionalMechanism_NoBSpline_执行复盘.md
```

## 8. Hash

| artifact | sha256 |
|---|---|
| `fig_A_b320_auc_seed_matrix.svg` | `675f4805a71bd8e8bff0eafc2ac9815e9d752eb1c9370350fff66a91bd2cb5f9` |
| `fig_A_b320_efficiency_memory_bar.svg` | `ffabd371ffbe74b0e5c4b883a55f3bde4d6a8f6ceae0d918aeb691eaf6768e2d` |
| `fig_A_b320_linec_vs_mlp.svg` | `16dc0708478cffe5662d1f9f197d2980d292f18d9b654d0368dcf54c4004d338` |
| `fig_A_b320_scorecard.svg` | `d04bcaedb606e7b9c25dfaadc5f8fcdca74a1db671481172bb13e6347987f295` |
| `fig_B_control_gap_by_candidate.svg` | `9b3f673805caca6cf1829804d352f0e90765458a09233e158cc608c357a12a77` |
| `fig_B_event_accept_reject_timeline.svg` | `6eb621b21adc1a427108ffdc5fda59811befa880b01e5c9af920b8cebb942e8b` |
| `fig_B_functional_pareto.svg` | `d80e742280a33ab60f9f525b3853226ad19a3bc0493889542d344fd0c5c348ad` |
| `fig_B_moment_transport_effect.svg` | `d903f39f1af999f279f0c5e0290fb30610381aea622f5c599897c8ef5d3a2bf7` |
| `fig_B_noise_leak_vs_coupling.svg` | `5ac78942263a2b83f82ba8637cb313fb11bea4f6872dce2bd5f540c7c22ca49f` |
| `fig_B_p3_to_p4_prediction.svg` | `5df459b77db99f458d31a6948d4ce2d7bced9ee199a64aa19d318186bebd96ae` |
| `fig_B_qp_constraint_activity.svg` | `4f60807d95ce34697e03408c93340581275e9bf426fc9adf53a23faeb248523a` |
| `fig_B_reservoir_vs_auc_time.svg` | `8c1f9a825210bf6a8acb6954a4cbbfea70447bd29fdfdfc4929b271bd508c966` |
| `fig_C_coupling_vs_noise_tradeoff.svg` | `322a0d42cb13021add522b2e60b7b130b543cb93076810c2fdb735fbcde9e790` |
| `fig_C_metric_std_by_batch_size.svg` | `152190c2d6343a7e0a6abbac9234d48214d0399fd1a73decc7576c11e8aab482` |
| `fig_C_noise_leak_ci_by_method.svg` | `53a127f5e4b78cda7907760e2bf5ec5a2a9095bf7a78a436fd5b41f25dbbee23` |
| `fig_C_noop_null_distribution.svg` | `2c36ca1d569aa99174beab53726cfbed090544b5f03332915f4d9509a3700d17` |
| `fig_C_reservoir_ci_by_method.svg` | `2a66825da1cd3523acf72b333a24b9f90e92e5a9e8f208eb5b337779300743a3` |
| `fig_D_cheby_degree_energy.svg` | `30a2174835a9dad71beed6aa4a14a1088defc5f979a0f14b8d1756145c64e7dd` |
| `fig_D_family_efficiency_expression_task.svg` | `8f1b3423111b0427fc33ad02bc3a5abd14a4568d574d9e83fb1eb72227cb95c9` |
| `fig_D_family_status_grid.svg` | `530ccbaecd92d0ce3a31eb53d7895595965e814b943b270ed1bb26f683500ee9` |
| `fig_D_fourier_spectrum_expression.svg` | `cb418793434c3939fd4eed64c44d908bf93c5f816dca6d6e3a47a5201080719e` |
| `fig_D_rational_linec_collapse.svg` | `c0cdccbb8e2478dd4dcbd3239c824e03f5886d1f37f8bbb7a9ad7f97e0cf7e87` |
| `fig_D_rbf_expression_r2.svg` | `c60be104bd36a7d9422f7b4437ff638f0fddea55fa89ac2e133fe49ce2937877` |
| `fig_D_wavelet_local_coverage.svg` | `79a397dc4716df23b090a2672342d1982446c418830fba9129fd649a7763d1c5` |
| `v1212_b320_anchor_hardening.csv` | `c2d7b6532734df63b14603b631626d37c5fce0fc3b8e6b814be167b4a1ad8379` |
| `v1212_family_efficiency.csv` | `2467a033ebfc2559c57f70592e82e701b048c0288395b5b996b1bee0469a7852` |
| `v1212_family_expression.csv` | `e1200e42ce05b7a4f4e7f4a7b2f812bd002fce63052ab0f27d2b4dab36078de8` |
| `v1212_family_failure_table.csv` | `35f90d8fc445929e2412dc7ba7dfaaf897a95678cbceab37eba66af3ad5f4605` |
| `v1212_family_linec.csv` | `ac3d78c16f13e1ea4cc838d14555cf11996478098e4cd457e0eae7102158afb3` |
| `v1212_family_manifest.csv` | `b117801b9231d86c1e8e2a8f22c6c086826ca610da9e03c5889aa419c7cdcd55` |
| `v1212_family_status.csv` | `b117801b9231d86c1e8e2a8f22c6c086826ca610da9e03c5889aa419c7cdcd55` |
| `v1212_family_status.json` | `9585e751b3bdaa09f00b029506dc717c7cbec5634be25e7cbc9f88437e3201d5` |
| `v1212_family_task_triage.csv` | `e7d2f274379b287486d835529d1462b6cf710064583f803316c09ec592bbc5b1` |
| `v1213_calibration_decision.json` | `221de839890e3b61b561438ae934f86d026c6d85771a2288041e55489ecfdb36` |
| `v1213_functional_p3_linec.csv` | `274fd6bb60c1d324393251d175789bc382ef1954559a6d5db1518fef04225885` |
| `v1213_linec_calibration.csv` | `274fd6bb60c1d324393251d175789bc382ef1954559a6d5db1518fef04225885` |
| `v1213_linec_null_distribution.csv` | `73df2a803177e931891de567af08968e7dffdeaceb6bc2876a6adec14db18ac6` |
| `v1213_route_decision.json` | `5e1ed48b2c6183df1020f616fca32f25c41756d3bf9720c445d7d8b106b7388a` |
| `v1213_run_manifest.json` | `676f76fab3cf2533e345c661029b64b0ff4993fa5a940c0ab1e9b9c4d98e53c2` |

## 9. Batch 9-11：loss-agnostic cotangent / spectral shaping

本轮修正 Batch 6-8 的 CE-specific 问题：新增方法不再使用 `CE(real label)`、`CE(permuted label)` 或 per-example CE vector 构造 functional 方向。方向来自当前训练 batch 的 label-free logit cotangent ensemble，以及由该 ensemble 生成的 gradient sketch sample modes / spectral objective。CE、ECE、CEp99 只作为审计指标和坏化约束。

### 9.1 代码修改

| file | 修改 | 审计说明 |
|---|---|---|
| `experiments/run_v1213_b320locked_functional_mechanism_nobspline.py` | 新增 cotangent-only 快路径 | 当 `--calibration-methods` 只含 control + `BM2s-z` 时，不再无条件计算旧 CE-specific projector 方向。 |
| `experiments/run_v1213_b320locked_functional_mechanism_nobspline.py` | 新增 `logit_cotangent_sketch_delta` | 用 label-free logit cotangent 和 sample-mode eigenspace 生成一阶 VJP；记录 `loss_agnostic_direction=1`、`ce_vector_used_for_direction=0`。 |
| `experiments/run_v1213_b320locked_functional_mechanism_nobspline.py` | 新增 `logit_cotangent_spectral_delta` | 用 label-free logit-cotangent gradient sketch 的谱目标做二阶 top-damp / isotropic shaping；不使用标签 CE 向量。 |

### 9.2 运行命令

Batch 9，一阶 cotangent sample-mode VJP：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1213_b320locked_functional_mechanism_nobspline.py --probe-device auto --probe-datasets MNIST --functional-batch-sizes 32 --windows 5 --probe-splits 5 --calibration-methods NoOp,TaskOnlyAdamW,RandomMatchedNorm,BM2s-LogitCotangentTail,BM2t-LogitCotangentTailFlip,BM2u-LogitCotangentBalanced,BM2v-LogitCotangentBalancedOrth --out-dir results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_MNIST_gpu0 --report-path results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_MNIST_gpu0/tmp_report.md
CUDA_VISIBLE_DEVICES=1 conda run -n kan python experiments/run_v1213_b320locked_functional_mechanism_nobspline.py --probe-device auto --probe-datasets Fashion-MNIST --functional-batch-sizes 32 --windows 5 --probe-splits 5 --calibration-methods NoOp,TaskOnlyAdamW,RandomMatchedNorm,BM2s-LogitCotangentTail,BM2t-LogitCotangentTailFlip,BM2u-LogitCotangentBalanced,BM2v-LogitCotangentBalancedOrth --out-dir results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_Fashion_gpu1 --report-path results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_Fashion_gpu1/tmp_report.md
CUDA_VISIBLE_DEVICES=2 conda run -n kan python experiments/run_v1213_b320locked_functional_mechanism_nobspline.py --probe-device auto --probe-datasets KMNIST --functional-batch-sizes 32 --windows 5 --probe-splits 5 --calibration-methods NoOp,TaskOnlyAdamW,RandomMatchedNorm,BM2s-LogitCotangentTail,BM2t-LogitCotangentTailFlip,BM2u-LogitCotangentBalanced,BM2v-LogitCotangentBalancedOrth --out-dir results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_KMNIST_gpu2 --report-path results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_KMNIST_gpu2/tmp_report.md
```

Batch 10，二阶 spectral shaping：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1213_b320locked_functional_mechanism_nobspline.py --probe-device auto --probe-datasets MNIST --functional-batch-sizes 32 --windows 5 --probe-splits 5 --calibration-methods NoOp,TaskOnlyAdamW,RandomMatchedNorm,BM2w-LogitCotangentSpectralTopDamp,BM2x-LogitCotangentSpectralTopDampOrth,BM2y-LogitCotangentSpectralIso,BM2z-LogitCotangentSpectralIsoOrth --out-dir results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_MNIST_gpu0 --report-path results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_MNIST_gpu0/tmp_report.md
CUDA_VISIBLE_DEVICES=1 conda run -n kan python experiments/run_v1213_b320locked_functional_mechanism_nobspline.py --probe-device auto --probe-datasets Fashion-MNIST --functional-batch-sizes 32 --windows 5 --probe-splits 5 --calibration-methods NoOp,TaskOnlyAdamW,RandomMatchedNorm,BM2w-LogitCotangentSpectralTopDamp,BM2x-LogitCotangentSpectralTopDampOrth,BM2y-LogitCotangentSpectralIso,BM2z-LogitCotangentSpectralIsoOrth --out-dir results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_Fashion_gpu1 --report-path results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_Fashion_gpu1/tmp_report.md
CUDA_VISIBLE_DEVICES=2 conda run -n kan python experiments/run_v1213_b320locked_functional_mechanism_nobspline.py --probe-device auto --probe-datasets KMNIST --functional-batch-sizes 32 --windows 5 --probe-splits 5 --calibration-methods NoOp,TaskOnlyAdamW,RandomMatchedNorm,BM2w-LogitCotangentSpectralTopDamp,BM2x-LogitCotangentSpectralTopDampOrth,BM2y-LogitCotangentSpectralIso,BM2z-LogitCotangentSpectralIsoOrth --out-dir results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_KMNIST_gpu2 --report-path results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_KMNIST_gpu2/tmp_report.md
```

Batch 11，非数据集特化 window 复查：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1213_b320locked_functional_mechanism_nobspline.py --probe-device auto --probe-datasets MNIST --functional-batch-sizes 32 --windows 3,10 --probe-splits 5 --calibration-methods NoOp,TaskOnlyAdamW,RandomMatchedNorm,BM2y-LogitCotangentSpectralIso,BM2z-LogitCotangentSpectralIsoOrth --out-dir results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_MNIST_gpu0 --report-path results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_MNIST_gpu0/tmp_report.md
CUDA_VISIBLE_DEVICES=1 conda run -n kan python experiments/run_v1213_b320locked_functional_mechanism_nobspline.py --probe-device auto --probe-datasets Fashion-MNIST --functional-batch-sizes 32 --windows 3,10 --probe-splits 5 --calibration-methods NoOp,TaskOnlyAdamW,RandomMatchedNorm,BM2y-LogitCotangentSpectralIso,BM2z-LogitCotangentSpectralIsoOrth --out-dir results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_Fashion_gpu1 --report-path results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_Fashion_gpu1/tmp_report.md
CUDA_VISIBLE_DEVICES=2 conda run -n kan python experiments/run_v1213_b320locked_functional_mechanism_nobspline.py --probe-device auto --probe-datasets KMNIST --functional-batch-sizes 32 --windows 3,10 --probe-splits 5 --calibration-methods NoOp,TaskOnlyAdamW,RandomMatchedNorm,BM2y-LogitCotangentSpectralIso,BM2z-LogitCotangentSpectralIsoOrth --out-dir results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_KMNIST_gpu2 --report-path results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_KMNIST_gpu2/tmp_report.md
```

最终合规聚合（排除 CE-specific Batch 6-8）：

```bash
conda run -n kan python experiments/run_v1213_b320locked_functional_mechanism_nobspline.py --aggregate-input-dirs results/v12_13_b320locked_functional_mechanism_nobspline/batch1_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch1_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch1_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1ext_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1ext_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1ext_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1role_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1role_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm1role_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm5_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm5_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch2_bm5_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch3_bm2_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch3_bm2_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch3_bm2_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch4_bm2signed_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch4_bm2signed_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch4_bm2signed_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch5_noisecontrast_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch5_noisecontrast_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch5_noisecontrast_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch9_cotangent_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch10_spectral_KMNIST_gpu2,results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_MNIST_gpu0,results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_Fashion_gpu1,results/v12_13_b320locked_functional_mechanism_nobspline/batch11_spectralwin_KMNIST_gpu2 --out-dir results/v12_13_b320locked_functional_mechanism_nobspline/batch11_aggregate_lossagnostic_official --report-path docs/DG-KAN_v12.13_B320Locked_FunctionalMechanism_NoBSpline_执行复盘.md
```

### 9.3 结论

Batch 9-11 仍没有 3x3 survivor，最终 route 仍为 `R3-FunctionalMechanismPartialOnly`，P4 关闭。`BM2y-LogitCotangentSpectralIso` 与 `BM2z-LogitCotangentSpectralIsoOrth` 在 `window=5` 各出现 1 个 partial pass，但只发生在 MNIST 单一 seed/split，window=3/10 没有扩展成稳定通过。该结果说明 loss-agnostic projector shaping 可以产生局部 Line C 改善，但当前 cotangent ensemble / spectral objective 还不足以稳定跨过 noise 与 reservoir 双门槛。
