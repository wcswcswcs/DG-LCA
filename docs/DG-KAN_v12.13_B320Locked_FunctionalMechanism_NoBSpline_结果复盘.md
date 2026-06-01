# DG-KAN v12.13 B320Locked FunctionalMechanism NoBSpline 结果复盘

生成时间：`2026-05-23T18:25:00Z`

对应计划：`docs/DG-KAN_v12.13_B320Locked_FunctionalMechanism_NoBSpline_独立分析与下一步计划.md`

对应执行审计：`docs/DG-KAN_v12.13_B320Locked_FunctionalMechanism_NoBSpline_执行复盘.md`

结果来源：

```text
results/v12_13_b320locked_functional_mechanism_nobspline/batch2_aggregate_bm1_bm5/v1213_calibration_decision.json
results/v12_13_b320locked_functional_mechanism_nobspline/batch2_aggregate_bm1_bm5/v1213_linec_null_distribution.csv
results/v12_13_b320locked_functional_mechanism_nobspline/batch2_aggregate_bm1_bm5/v1213_functional_p3_linec.csv
```

本文件只做结果解释，不新增实验数据，不修改 gate，不把 partial positive 写成 survivor。

## 1. 最终结论

v12.13 没有得到可进入 P4 的 functional survivor。

```text
route = R3-FunctionalMechanismPartialOnly
official_functional_success = 0
linec_calibration_pass = 1
bm1_full_3x3_pass_candidate_count = 0
bm3_full_3x3_pass_candidate_count = 0
bm5_full_3x3_pass_candidate_count = 0
```

解释：

```text
Line C 测量本身没有在 NoOp / RandomMatchedNorm 上出现 false positive；
但 BM1-ext / BM3 / BM5 都没有满足 3x3 P3 survivor 标准；
因此 P4 short-run 按 v12.13 规则保持关闭。
```

## 2. B320 Anchor 状态

B320 仍然是 locked anchor，不是本轮 blocker。

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

结论：v12.13 的失败不是 base regression，也不是 B320 需要继续小修。

## 3. Line C Calibration 结果

本轮聚合结果：

```text
calibration_rows = 1350
summary_rows = 27
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
noop_false_positive_rows = 0
random_false_positive_rows = 0
calibration_fail_reason =
```

NoOp / RandomMatchedNorm 的 null distribution：

| method | batch | window | rows | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|---:|
| `NoOp` | `16` | `5` | `45` | `0.0` | `0.0` | `0` |
| `NoOp` | `32` | `3` | `45` | `0.0` | `0.0` | `0` |
| `NoOp` | `32` | `5` | `90` | `0.0` | `0.0` | `0` |
| `NoOp` | `32` | `10` | `45` | `0.0` | `0.0` | `0` |
| `RandomMatchedNorm` | `16` | `5` | `45` | `-0.00005200770166185167` | `0.000014420681529574925` | `0` |
| `RandomMatchedNorm` | `32` | `3` | `45` | `0.000001059751957654953` | `-0.000004595849249098036` | `0` |
| `RandomMatchedNorm` | `32` | `5` | `90` | `-0.00001556078592936198` | `-0.00000355574819776747` | `0` |
| `RandomMatchedNorm` | `32` | `10` | `45` | `0.000029197438723511166` | `-0.00003219727012846205` | `0` |

结论：在本轮设置下，Line C 的 hard gate 没有被 NoOp 或随机匹配范数误触发。当前主要 blocker 不是 metric false positive，而是 functional 方向没有稳定产生 noise/reservoir release。

## 4. Functional Mechanism 结果

总体失败计数：

```text
fail_reason_counts = {
  "CouplingR2_delta<0.02": 485,
  "NoiseSignalLeak_delta>-0.01": 1301,
  "RealSignalReservoirRatio_delta>-0.01": 1248,
  "control_gap<0.005": 1157,
  "logit_max_abs_drift>0.05": 362,
  "holdout_loss_ratio>1.002": 81
}
```

这说明最主要 blocker 仍然是：

```text
1. NoiseSignalLeak 没有稳定下降到 -0.01；
2. RealSignalReservoirRatio 没有稳定下降到 -0.01；
3. control_gap 经常不足；
4. 部分 coupling 提升伴随 logit drift 或 task-safety 问题。
```

### 4.1 BM1-ext

BM1b 仍只有局部 positive，不能 promotion。

| method | batch | window | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BM1b-SoftSNRGate` | `16` | `5` | `45` | `0.3763169309966251` | `-0.00010306388139724731` | `-0.0031816853417290583` | `1` |
| `BM1b-SoftSNRGate` | `32` | `3` | `45` | `0.1593876228496969` | `0.001174013859902819` | `-0.0013699978590011597` | `0` |
| `BM1b-SoftSNRGate` | `32` | `5` | `45` | `0.2136147570950757` | `0.00017672176990244124` | `-0.002479976746771071` | `0` |
| `BM1b-SoftSNRGate` | `32` | `10` | `45` | `0.2752052408534761` | `-0.00198145281109545` | `-0.004790632923444112` | `0` |

结论：BM1b 可以打开 CouplingR2，但 noise/reservoir 的均值改善远小于 `-0.01` gate。`batch=16/window=5` 的 1 行 pass 仍是 partial，不是 3x3 survivor。

### 4.2 Role-wise SNR

| method | batch | window | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BM1h-RoleWiseSNR-direct` | `32` | `5` | `45` | `0.015868592610844685` | `-0.0000399228185415268` | `-0.0005676911936865912` | `0` |
| `BM1i-RoleWiseSNR-quad` | `32` | `5` | `45` | `0.21774649796769718` | `-0.001746921158499188` | `-0.001116353604528639` | `0` |
| `BM1j-RoleWiseSNR-branch` | `32` | `5` | `45` | `0.11547631639673457` | `-0.0006927752453419897` | `-0.003169774015744527` | `0` |

结论：

```text
direct role 基本接近 no-op；
quad role 有 coupling，但 noise/reservoir release 不足；
branch role 对 reservoir 略好于 direct/quad，但仍远低于 -0.01 gate。
```

### 4.3 BM3 Branch Rebalance

| method | batch | window | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BM3a-BranchRebalance` | `16` | `5` | `45` | `0.6646413285593893` | `0.000426774885919359` | `0.007577203048600091` | `0` |
| `BM3a-BranchRebalance` | `32` | `3` | `45` | `0.3173873356126252` | `0.0013470250699255202` | `0.0037208212746514214` | `0` |
| `BM3a-BranchRebalance` | `32` | `10` | `45` | `0.323897284170842` | `0.0014847979363467958` | `0.014852123790317112` | `0` |

结论：BM3a 是典型 coupling-only / coordinate move。它能显著提高 CouplingR2，但 NoiseSignalLeak 与 RealSignalReservoirRatio 反而变差，因此不能作为 functional value source。

### 4.4 BM5 Reservoir VJP

| method | batch | window | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BM5a-ReservoirVJP` | `32` | `5` | `45` | `0.21163811196231413` | `0.0004035065985388226` | `0.00071099234951867` | `0` |
| `BM5b-ReservoirVJP-AdamWOrthogonal` | `32` | `5` | `45` | `0.21332445317792942` | `0.0003549836162063811` | `0.0009675347142749362` | `0` |

结论：当前 BM5 原型没有释放 reservoir。AdamW-orthogonal 后 RealSignalReservoirRatio_delta 仍为正，说明简单 logits-vs-onehot residual VJP 不是足够的 reservoir-release basis，下一步应进入 BM2 constrained solve，而不是把 BM5 单独 promotion。

## 5. Classic No-BSpline Family 状态

| family | status | blocker |
|---|---|---|
| `Rational` | `TaskBlocked` | L3/A4 pass，但 A5 task gate fail |
| `Chebyshev` | `TaskBlocked` | L3/A4 pass，但 A5 task gate fail |
| `Wavelet` | `TaskBlocked` | L3/A4 pass，但 A5 task gate fail |
| `RBF` | `ExpressionBlocked` | L3 pass，但 A4 expression gate fail |
| `Fourier` | `ExpressionBlocked` | L3 pass，但 A4 expression gate fail |
| `BSpline` | `RejectedForThisVersion` | v12.12 起冻结 |

结论：Line D 没有改变 v12.13 主线判断。Classic family 仍是并行 portfolio，不替代 B320 functional 主线。

## 6. 为什么不能进入 P4

v12.13 规定 P4 只对 P3 survivor 运行。P3 survivor 要求：

```text
3x3 all-row pass；
或 8/9 pass 且 bootstrap CI 有效，失败不集中在单一 dataset。
```

本轮实际是：

```text
BM1 partial pass rows = 1
BM3 partial pass rows = 0
BM5 partial pass rows = 0
full_3x3_pass_candidate_count = 0
```

因此继续 P4 会违反计划，不是保守，而是避免把 clone one-step artifact 或 partial row 写成成功。

## 7. 下一步建议

当前最合理的下一步是 BM2：

```text
BM2-Constrained Signal-Reservoir Projector
目标：直接把 NoiseSignalLeak 和 RealSignalReservoirRatio 作为 hard constraints。
不要再单独加大 BM1/BM3/BM5 强度；
不要按 dataset 调 threshold；
不要进入 P4；
先构造低秩 QP / constrained solve，并记录 constraint_active_count、solve_status、functional_norm_ratio、control_gap。
```

具体优先级：

```text
1. 用 TaskOnlyAdamW / BM1 / BM5 / branch roles 组成低秩 basis；
2. 在 basis 内求满足 logit drift、holdout loss、CEp99 的 constrained update；
3. 若 coupling 过但 noise 不过，提高 beta，不提高 alpha；
4. 若 reservoir 不过，加入 reservoir-release VJP basis 或提高 gamma；
5. 若 control_gap 小，移除 AdamW 平行分量。
```

最终 route：

```text
R3-FunctionalMechanismPartialOnly
```

## 8. 追加结果：BM2 Constrained Low-Rank Solve

追加时间：`2026-05-23T18:31:00Z`

结果来源：

```text
results/v12_13_b320locked_functional_mechanism_nobspline/batch3_aggregate_bm2/v1213_calibration_decision.json
results/v12_13_b320locked_functional_mechanism_nobspline/batch3_aggregate_bm2/v1213_linec_null_distribution.csv
results/v12_13_b320locked_functional_mechanism_nobspline/batch3_aggregate_bm2/v1213_functional_p3_linec.csv
```

### 8.1 BM2 做了什么

实现的候选：

```text
BM2e-LowRankLineCGrid-balanced
BM2g-LineCGrid-withNoiseVeto
BM2h-LineCGrid-withReservoirVeto
```

低秩 basis 固定为：

```text
bm1 = BM1b-SoftSNRGate
bm5_orth = BM5b-ReservoirVJP-AdamWOrthogonal
branch = BM1j-RoleWiseSNR-branch
quad = BM1i-RoleWiseSNR-quad
```

求解方式：

```text
在训练 functional batch 内部分成 solve/holdout split；
在固定低秩 basis 上做固定系数 grid solve；
内部选择只使用训练 batch holdout，不使用 validation/test 作为 commit-time 特征；
外部仍按 v12.13 P3 Line C gate 评估。
```

记录字段已写入 CSV：

```text
subspace_id
rank
alpha_beta_gamma_eta
constraint_active_count
solve_status
functional_norm_ratio
task_orthogonal_fraction
bm2_coeff_*
bm2_internal_*
```

### 8.2 BM2 聚合结果

```text
calibration_rows = 1620
summary_rows = 30
linec_calibration_pass = 1
noop_false_positive_rows = 0
random_false_positive_rows = 0
bm2_full_3x3_pass_candidate_count = 0
bm2_partial_pass_rows = 0
route = R3-FunctionalMechanismPartialOnly
```

BM2 mean result：

| method | batch | window | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BM2e-LowRankLineCGrid-balanced` | `32` | `5` | `45` | `0.20165167372544876` | `0.00013725755529271232` | `-0.002332194315062629` | `0` |
| `BM2g-LineCGrid-withNoiseVeto` | `32` | `5` | `45` | `0.2120803797765593` | `0.0001860047173168924` | `-0.0021824253929985894` | `0` |
| `BM2h-LineCGrid-withReservoirVeto` | `32` | `5` | `45` | `0.18689091708024527` | `-0.0001542907829085986` | `-0.0029072622458140057` | `0` |

### 8.3 BM2 结论

BM2 比 BM3 更像 task-safe functional direction：CouplingR2 保持正值，reservoir 均值比 BM3 明显更好，没有 BM3 那种 reservoir 正向恶化。但它仍未解决 v12.13 的 hard blocker：

```text
NoiseSignalLeak_delta 仍远高于 -0.01；
RealSignalReservoirRatio_delta 仍远高于 -0.01；
pass_rows = 0；
bm2_full_3x3_pass_candidate_count = 0。
```

失败计数更新为：

```text
fail_reason_counts = {
  "CouplingR2_delta<0.02": 575,
  "NoiseSignalLeak_delta>-0.01": 1562,
  "RealSignalReservoirRatio_delta>-0.01": 1504,
  "control_gap<0.005": 1387,
  "holdout_loss_ratio>1.002": 81,
  "logit_max_abs_drift>0.05": 430
}
```

### 8.4 v12.13 是否完成

按当前计划可认为 v12.13 的主线执行已经推进到 BM2 第一版，但没有达到 functional success：

```text
B320 anchor：完成，locked；
Line C calibration：完成，metric 本轮可信；
BM1-ext：完成，无 survivor；
BM3-ext baseline：完成，coupling-only；
BM5 first VJP：完成，无 survivor；
BM2 first constrained low-rank solve：完成，无 survivor；
P4：未开启，因为没有 P3 survivor。
```

因此当前最终 route 仍是：

```text
R3-FunctionalMechanismPartialOnly
```

下一步不应进入 P4，也不应继续调 BM1/BM3/BM5 强度。若继续 v12.13 后续工作，应改进 BM2 的 basis 本身：当前 basis 只能产生 coupling 与轻微 reservoir 改善，但没有提供足够强的 noise-release / reservoir-release 方向。

## 9. 追加结果：BM2 Signed / Noise-Contrast

追加时间：`2026-05-23T19:49:00Z`

结果来源：

```text
results/v12_13_b320locked_functional_mechanism_nobspline/batch5_aggregate_noisecontrast/v1213_calibration_decision.json
results/v12_13_b320locked_functional_mechanism_nobspline/batch5_aggregate_noisecontrast/v1213_linec_null_distribution.csv
results/v12_13_b320locked_functional_mechanism_nobspline/batch5_aggregate_noisecontrast/v1213_functional_p3_linec.csv
```

### 9.1 追加机制

新增机制：

```text
BM2i-LineCGrid-signed
BM2j-LineCGrid-signedAdamWOrthogonal
BM5c-NoiseVetoVJP
BM5d-NoiseVetoVJP-AdamWOrthogonal
BM2k-LineCGrid-noiseContrast
BM2l-LineCGrid-noiseContrastOrth
```

设计目的：

```text
BM2i/BM2j:
  允许 low-rank basis 的 signed coefficients，并测试移除 AdamW 平行分量是否能打开 noise/reservoir release。

BM5c/BM5d:
  使用当前训练 batch 的 permuted-label VJP 做 noise veto，构造 real-label descent minus noise-label descent 的 functional correction。

BM2k/BM2l:
  把 noise-veto basis 加入 BM2 low-rank solve，测试显式 noise-contrast 是否能改善 NoiseSignalLeak。
```

这些机制仍满足：

```text
no validation/test commit-time feature;
no dataset-name branch;
no teacher / distillation;
no loss modification;
no sampler / class weight;
no P4 without P3 survivor.
```

### 9.2 最终聚合结果

```text
calibration_rows = 2160
summary_rows = 36
linec_calibration_pass = 1
noop_false_positive_rows = 0
random_false_positive_rows = 0
bm1_full_3x3_pass_candidate_count = 0
bm2_full_3x3_pass_candidate_count = 0
bm3_full_3x3_pass_candidate_count = 0
bm5_full_3x3_pass_candidate_count = 0
bm1_partial_pass_rows = 1
bm2_partial_pass_rows = 0
bm3_partial_pass_rows = 0
bm5_partial_pass_rows = 0
route = R3-FunctionalMechanismPartialOnly
```

追加方法均值：

| method | batch | window | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BM2i-LineCGrid-signed` | `32` | `5` | `45` | `0.21702432380318065` | `0.000011947212947739496` | `-0.0021996693478690253` | `0` |
| `BM2j-LineCGrid-signedAdamWOrthogonal` | `32` | `5` | `45` | `0.1983931336065015` | `-0.0007064087523354425` | `-0.0016072054704030355` | `0` |
| `BM5c-NoiseVetoVJP` | `32` | `5` | `45` | `0.2542753834003061` | `-0.00041285134438011383` | `-0.002257476250330607` | `0` |
| `BM5d-NoiseVetoVJP-AdamWOrthogonal` | `32` | `5` | `45` | `0.25625326788018277` | `-0.0006122012519174152` | `-0.0012898875607384576` | `0` |
| `BM2k-LineCGrid-noiseContrast` | `32` | `5` | `45` | `0.2541957760602528` | `-0.00029134597215387556` | `-0.0028791122966342502` | `0` |
| `BM2l-LineCGrid-noiseContrastOrth` | `32` | `5` | `45` | `0.2571067341777921` | `-0.0005828314564294286` | `-0.0007246166467666626` | `0` |

### 9.3 结论

noise-contrast 有真实诊断价值：相比 BM2 第一版，`BM2k/BM2l` 和 `BM5c/BM5d` 的 CouplingR2 更高，NoiseSignalLeak 均值也从接近 0 或正数变成小幅负数。

但它仍不是 survivor：

```text
NoiseSignalLeak_delta 仍远高于 -0.01；
RealSignalReservoirRatio_delta 仍远高于 -0.01；
pass_rows = 0；
full_3x3_pass_candidate_count = 0。
```

最终失败计数：

```text
fail_reason_counts = {
  "CouplingR2_delta<0.02": 755,
  "NoiseSignalLeak_delta>-0.01": 2094,
  "RealSignalReservoirRatio_delta>-0.01": 2031,
  "control_gap<0.005": 1777,
  "holdout_loss_ratio>1.002": 98,
  "logit_max_abs_drift>0.05": 540
}
```

### 9.4 当前 v12.13 状态

当前可以认为 v12.13 主线已按计划推进到以下节点：

```text
B320 anchor monitor：完成，locked；
Line C calibration：完成，本轮 NoOp/Random false positive = 0；
BM1-ext：完成，无 3x3 survivor；
BM3：完成，coupling-only；
BM5 reservoir VJP：完成，无 survivor；
BM2 first grid：完成，无 survivor；
BM2 signed / AdamW-orth：完成，无 survivor；
BM5 noise-veto / BM2 noise-contrast：完成，无 survivor；
P4：按计划关闭，因为没有 P3 survivor。
```

因此最终 route 仍是：

```text
R3-FunctionalMechanismPartialOnly
```

更深层 blocker：

```text
现有 functional directions 可以稳定打开 CouplingR2，
也能让 NoiseSignalLeak 出现很小的负向均值，
但无法把 NoiseSignalLeak_delta 或 RealSignalReservoirRatio_delta 推到 -0.01。
这说明当前 basis 主要改变 update/readout direction，
还没有真正改变 Line C 使用的 per-example gradient sketch signal/reservoir 子空间。
```

下一步如果继续，不应进入 P4，也不应再只改组合系数；需要构造能改变 `grad_sketch @ grad_sketch.T` 谱结构的 loss-agnostic basis。注意：不能把 per-example CE vector 作为 functional 方向的优化目标；CE / ECE / CEp99 只能作为审计指标或约束，不应作为上游 loss surrogate。

## 10. Batch 6：projector CE-vector surrogate VJP（非合规诊断）

### 10.1 动机

Batch 5 后的 blocker 是：现有 basis 对 CouplingR2 有效，对 NoiseSignalLeak 有弱负向效果，但始终不能把 `RealSignalReservoirRatio_delta` 推到 `-0.01`。本轮为了诊断 fixed projector 是否能解释 reservoir release，固定当前训练 batch 的 Line C projector：

```text
grad_sketch -> K = grad_sketch @ grad_sketch.T
K eigenspace -> p_sig / p_res
r_real = CE(real label) centered
r_noise = CE(permuted label) centered
surrogate = ||p_res r_real||^2 / ||r_real||^2
          + ||p_sig r_noise||^2 / ||r_noise||^2
```

新增诊断方法：

```text
BM5e-LineCProjectorCEVJP：只压 reservoir 项；
BM5f-LineCProjectorCEVJP-balanced：同时压 reservoir/noise；
BM5g-LineCProjectorCEVJP-balancedOrth：balanced 后去除 AdamW 平行分量；
BM2m-LineCGrid-projectorCE：把 projector CE-vector VJP 作为 BM2 basis；
BM2n-LineCGrid-projectorCEOrth：BM2m 的 AdamW-orthogonal 版本。
```

合规性审计：这些方法直接用 `CE(real label)` / `CE(permuted label)` 构造 VJP surrogate，违反“不能只针对 CE / functional 应 loss-agnostic”的约束，因此不能作为 official functional mechanism 或 P3 survivor 证据。这里只保留为错误路线的诊断记录，避免后续误用。实现 bug 审计：第一次运行时 `_sample_grad_sketch` 被误放入 `no_grad`，导致内部 per-example gradient 不能计算；修复后只 detach 已生成的 `grad_sketch` 与 projector，三数据集重跑均成功。

### 10.2 聚合结果

聚合目录：

```text
results/v12_13_b320locked_functional_mechanism_nobspline/batch6_aggregate_projector
```

关键 summary：

| method | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|
| `BM5e-LineCProjectorCEVJP` | 45 | 0.128310 | 0.006051 | -0.018313 | 0 |
| `BM5f-LineCProjectorCEVJP-balanced` | 45 | 0.141640 | 0.009075 | -0.016331 | 0 |
| `BM5g-LineCProjectorCEVJP-balancedOrth` | 45 | 0.170483 | 0.008888 | -0.015638 | 0 |
| `BM2m-LineCGrid-projectorCE` | 45 | 0.228680 | 0.007772 | -0.016844 | 0 |
| `BM2n-LineCGrid-projectorCEOrth` | 45 | 0.226375 | 0.001773 | -0.001717 | 0 |

最终 decision：

```json
{
  "linec_calibration_pass": 1,
  "noop_false_positive_rows": 0,
  "random_false_positive_rows": 0,
  "bm1_full_3x3_pass_candidate_count": 0,
  "bm2_full_3x3_pass_candidate_count": 0,
  "bm3_full_3x3_pass_candidate_count": 0,
  "bm5_full_3x3_pass_candidate_count": 0,
  "bm5_partial_pass_rows": 0,
  "route": "R3-FunctionalMechanismPartialOnly"
}
```

### 10.3 解释

本轮是一个非合规诊断：它第一次明确把 `RealSignalReservoirRatio_delta_mean` 打到 `-0.01` 以下：`BM5e=-0.018313`，`BM2m=-0.016844`。这只能说明“CE-specific projector surrogate”可以解释 reservoir release 的一个方向，不能说明 functional update 已满足 loss-agnostic 约束。

但它仍然不是 P3 survivor，因为代价是 `NoiseSignalLeak_delta_mean` 转为明显正值，`BM2m=+0.007772`、`BM5f=+0.009075`。换言之，projector CE surrogate 找到了 reservoir release 的一半机制，但没有同时做到 noise-safe；`BM2n` 的 task-orthogonal 版本能压低 noise 增幅到 `+0.001773`，但 reservoir release 同时退回 `-0.001717`，也失败。

因此 v12.13 仍未完成，P4 仍然关闭。更正后的 blocker 不是“找到可用方向”，而是：

```text
CE-specific projector VJP 可以作为诊断显示 reservoir release 是可达的，
但它不满足 loss-agnostic functional 约束，不能计入 official success。
下一步必须构造不依赖 CE label loss 的 projector / gradient-sketch shaping basis。
```

## 11. Batch 7-8：noise-hard 与 projector Pareto grid（非合规诊断）

### 11.1 Batch 7 noise-hard

为检验 Batch 6 的冲突是否只是 surrogate 组合方式问题，本轮把 reservoir projector VJP 投影到 noise surrogate 的非增半空间/nullspace。由于这些方向仍由 CE-vector surrogate 生成，仍属于非合规诊断，不计入 official functional success：

```text
BM5h-LineCProjectorCEVJP-noiseNull
BM5i-LineCProjectorCEVJP-noiseMargin
BM5j-LineCProjectorCEVJP-noiseNullOrth
BM2o-LineCGrid-projectorNoiseHard
BM2p-LineCGrid-projectorNoiseHardOrth
```

聚合结果：

| method | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|
| `BM5h-LineCProjectorCEVJP-noiseNull` | 45 | 0.130042 | 0.006071 | -0.017939 | 0 |
| `BM5i-LineCProjectorCEVJP-noiseMargin` | 45 | 0.130863 | 0.006061 | -0.017836 | 0 |
| `BM5j-LineCProjectorCEVJP-noiseNullOrth` | 45 | 0.165792 | 0.005856 | -0.017118 | 0 |
| `BM2o-LineCGrid-projectorNoiseHard` | 45 | 0.224857 | 0.004851 | -0.017916 | 0 |
| `BM2p-LineCGrid-projectorNoiseHardOrth` | 45 | 0.225207 | -0.001243 | -0.002937 | 0 |

结论：noise-hard 约束能把 `BM2p` 的 NoiseSignalLeak_delta_mean 压到负值，但 reservoir release 同时退回不到门槛；未形成 P3 survivor。更重要的是，该路线仍 CE-specific，不能作为合规 functional 方向。

### 11.2 Batch 8 projector Pareto grid

Batch 7 说明手工半空间投影不能保持两端都过门槛，因此进一步将 pure reservoir-projector VJP 与 pure noise-projector VJP 分开作为 BM2 basis，让内部 Line C / holdout 评估直接选 Pareto 折中。该实验同样只作为非合规诊断：

```text
BM5k-LineCProjectorCEVJP-noiseOnly
BM2q-LineCGrid-projectorPareto
BM2r-LineCGrid-projectorParetoOrth
```

最终聚合目录：

```text
results/v12_13_b320locked_functional_mechanism_nobspline/batch8_aggregate_projector_pareto
```

关键结果：

| method | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|
| `BM5k-LineCProjectorCEVJP-noiseOnly` | 45 | 0.141777 | 0.000065 | 0.002880 | 0 |
| `BM2q-LineCGrid-projectorPareto` | 45 | 0.168517 | 0.001561 | -0.001445 | 0 |
| `BM2r-LineCGrid-projectorParetoOrth` | 45 | 0.192934 | 0.001606 | -0.000740 | 0 |

最终 decision：

```json
{
  "linec_calibration_pass": 1,
  "noop_false_positive_rows": 0,
  "random_false_positive_rows": 0,
  "bm1_full_3x3_pass_candidate_count": 0,
  "bm2_full_3x3_pass_candidate_count": 0,
  "bm3_full_3x3_pass_candidate_count": 0,
  "bm5_full_3x3_pass_candidate_count": 0,
  "route": "R3-FunctionalMechanismPartialOnly"
}
```

### 11.3 当前最终判断

v12.13 仍未完成 functional success，P4 仍关闭。Batch 6-8 的正确解释应是：

```text
CE-specific projector surrogate 可以作为诊断释放 reservoir；
但它违反 loss-agnostic 约束，不能算 official mechanism；
并且即便在 CE-specific 设定下，也没有同时满足 noise/reservoir gate。
```

因此继续推进时不应再增加任意 coefficient grid，也不应继续使用 CE-vector VJP。更合理的下一步是 loss-agnostic projector shaping：用输出/logit Jacobian 的任意上游 cotangent ensemble、或 primitive/branch-level basis 来改变 per-example gradient sketch 的 eigenspace/谱结构；CE / ECE / CEp99 只保留为审计指标和坏化约束。

## 12. Batch 9-11：loss-agnostic cotangent / spectral shaping

### 12.1 合规性修正

Batch 6-8 暴露出一个审计错误：虽然没有改训练 loss，但 functional 方向直接用了 per-example CE vector，因此不满足“应对任意上游 loss”的要求。本轮新增专用 fast path：当方法只包含 control + `BM2s-z` 时，runner 不再无条件计算旧的 CE-specific projector 方向。

新增方向全部只使用：

```text
当前训练 batch 输入 x；
model(x) 的 logit；
label-free random cotangent ensemble；
由 cotangent VJP 形成的 gradient sketch / spectral objective。
```

不使用：

```text
CE(real label)；
CE(permuted label)；
per-example CE vector；
validation/test/future outcome；
dataset-name branch。
```

CSV 中新增审计字段：

```text
loss_agnostic_direction = 1
ce_vector_used_for_direction = 0
```

### 12.2 Batch 9：一阶 cotangent sample-mode VJP

方法：

```text
BM2s-LogitCotangentTail
BM2t-LogitCotangentTailFlip
BM2u-LogitCotangentBalanced
BM2v-LogitCotangentBalancedOrth
```

结果：

| method | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|
| `BM2s-LogitCotangentTail` | 45 | 0.157099 | -0.002332 | -0.000219 | 0 |
| `BM2t-LogitCotangentTailFlip` | 45 | 0.157431 | 0.000095 | 0.000801 | 0 |
| `BM2u-LogitCotangentBalanced` | 45 | 0.168259 | -0.000130 | -0.000275 | 0 |
| `BM2v-LogitCotangentBalancedOrth` | 45 | 0.171481 | -0.000165 | -0.000307 | 0 |

解释：一阶 cotangent tail 方向是合规的，且能带来弱负向 NoiseSignalLeak，但 reservoir release 几乎没有打开。

### 12.3 Batch 10：二阶 spectral shaping

方法：

```text
BM2w-LogitCotangentSpectralTopDamp
BM2x-LogitCotangentSpectralTopDampOrth
BM2y-LogitCotangentSpectralIso
BM2z-LogitCotangentSpectralIsoOrth
```

结果：

| method | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|
| `BM2w-LogitCotangentSpectralTopDamp` | 45 | 0.140617 | 0.000090 | 0.000915 | 0 |
| `BM2x-LogitCotangentSpectralTopDampOrth` | 45 | 0.146086 | 0.000028 | 0.000664 | 0 |
| `BM2y-LogitCotangentSpectralIso` | 45 | 0.144707 | -0.000488 | 0.000252 | 1 |
| `BM2z-LogitCotangentSpectralIsoOrth` | 45 | 0.150198 | -0.000532 | 0.000049 | 1 |

`BM2y/BM2z` 的 pass 行来自 MNIST 单一 seed/split：

```text
BM2y MNIST seed=2 split=2 window=5:
CouplingR2_delta=0.165617
NoiseSignalLeak_delta=-0.017918
RealSignalReservoirRatio_delta=-0.013195

BM2z MNIST seed=2 split=2 window=5:
CouplingR2_delta=0.191150
NoiseSignalLeak_delta=-0.017727
RealSignalReservoirRatio_delta=-0.013122
```

这是 v12.13 中第一次出现 loss-agnostic projector/spectral 方向的 partial pass，但只是一行局部信号，不能进入 P4。

### 12.4 Batch 11：window 复查

为避免只报告 `window=5` 的偶然局部，本轮对 `BM2y/BM2z` 追加 `window=3,10`，不按数据集调参。

| method | window | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|---:|
| `BM2y-LogitCotangentSpectralIso` | 3 | 45 | 0.105373 | 0.000048 | 0.000046 | 0 |
| `BM2y-LogitCotangentSpectralIso` | 5 | 45 | 0.144707 | -0.000488 | 0.000252 | 1 |
| `BM2y-LogitCotangentSpectralIso` | 10 | 45 | 0.200101 | -0.000675 | 0.001461 | 0 |
| `BM2z-LogitCotangentSpectralIsoOrth` | 3 | 45 | 0.108382 | 0.000032 | -0.000015 | 0 |
| `BM2z-LogitCotangentSpectralIsoOrth` | 5 | 45 | 0.150198 | -0.000532 | 0.000049 | 1 |
| `BM2z-LogitCotangentSpectralIsoOrth` | 10 | 45 | 0.209989 | -0.000839 | 0.001261 | 0 |

### 12.5 当前合规结论

最终 official 聚合目录：

```text
results/v12_13_b320locked_functional_mechanism_nobspline/batch11_aggregate_lossagnostic_official
```

最终 decision：

```json
{
  "linec_calibration_pass": 1,
  "noop_false_positive_rows": 0,
  "random_false_positive_rows": 0,
  "bm1_full_3x3_pass_candidate_count": 0,
  "bm2_full_3x3_pass_candidate_count": 0,
  "bm2_partial_pass_rows": 2,
  "bm3_full_3x3_pass_candidate_count": 0,
  "bm5_full_3x3_pass_candidate_count": 0,
  "route": "R3-FunctionalMechanismPartialOnly"
}
```

因此 v12.13 仍未完成 functional success，P4 继续关闭。新的、合规的结论是：

```text
loss-agnostic logit-cotangent spectral shaping 可产生局部 Line C pass；
但信号只出现在 MNIST 单一 seed/split，不能跨 dataset/seed 稳定复现；
当前 blocker 是：label-free cotangent ensemble 与 B320 的实际 CE-gradient Line C projector 之间耦合太弱。
```

继续推进时，应优先考虑不含 CE 的更强 cotangent ensemble 或 primitive/branch-level gradient-sketch basis，例如 class-mean-free logit Jacobian 多 cotangent、layer/branch role-conditioned cotangent、或直接对 direct/quad/branch 子模块的 label-free gradient sketch 做谱 shaping；不应回到 CE-vector surrogate。
