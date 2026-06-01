# DG-KAN v13.8：Generic PopRisk Optimizer 与 KAN SNR-to-Basis Lift 分线完整计划

> 版本：v13.8 execution plan  
> 基于：v13.7 `BoundaryConditionedPopRiskTraining` 真实结果复盘；`A Theory of Generalization in Deep Learning`；`Deep Manifold Part 2: Neural Network Mathematics`  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 硬约束：strict FC-PureKAN；no active B-spline；不使用 label-informed initialization；不使用 teacher / distillation / loss modification / sampler / class weight / dataset-name branch；functional direction 不使用 validation/test/future/query batch；CEp99/NLL/ECE/LineC hard target 只能作为 audit/gate，不能生成 direction；允许通过统一 loss interface 读取当前训练 loss 的 output cotangent，不允许写死 CE-specific 规则。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的最终目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找到一个单独好看的 optimizer trick。项目目标是：

$$
\boxed{
\text{构建 label-free strict FC-PureKAN substrate/base，并通过 loss-interface-generic functional update 得到比 ordinary backprop/AdamW 更好的训练几何与模型。}
}
$$

最终 claim 必须同时满足：

```text
1. 表达力不打折，最好优于同规模 MLP；
2. forward / backward / step / memory 与 MLP 可比；
3. training trajectory 健康，AUC-step / AUC-time 不输；
4. geometry 更好：train-probe coupling、signal/reservoir/noise、tail、calibration 不坏；
5. functional update 的收益必须打过 NoOp / RandomMatchedNorm / AdamWParallel / SNR-only / MLP analog controls；
6. 不使用 label-informed initialization；
7. 不把 generic MLP optimizer positive 写成 KAN-specific success。
```

当前最重要的路线重定义是：

$$
\boxed{
\text{loss-agnostic 不应被执行成 label-blind，而应是 loss-interface-generic。}
}
$$

也就是说，functional update 可以通过统一 loss interface 使用当前训练 batch 的 output cotangent / per-example gradients；但不能针对 CEp99、NLL、ECE、LineC hard target、validation/test/future/query batch 或 dataset name 设计方向。

## 0.2 当前进展摘要

v13.7 改变了项目状态。v13.6 的 one-shot PopRisk-SNR 失败后，v13.7 把它改成持续训练过程里的 boundary-conditioned PopRisk-SNR，并得到一个关键结果：

```text
MLP generic SNR 在 synthetic plan-scale 上 7/7 task family pass；
MLP real triage 3x3 和 5-seed 可通过；
MLP real triage 10-seed 未确认，只有 KMNIST robust，MNIST borderline，Fashion-MNIST weak；
KAN/Rational parameter/basis/cover SNR 仍未达到 S3/S4；
Non-RAT 仍不是 functional substrate。
```

因此 v13.7 的正确结论不是“PopRisk-SNR 全失败”，而是：

$$
\boxed{
\text{population-risk SNR 是一个真实 generic optimizer signal；}
\text{但当前 KAN/Rational basis/group telemetry 没有把 generic signal 转成 KAN-specific advantage。}
}
$$

v13.8 因此不再继续做 cover threshold / SNR tau / EMA / role norm 小网格。下一步要把路线拆成两条：

```text
Line G:
  Generic MLP PopRisk optimizer line。
  它独立验证 SNR/Blend 是否是通用 optimizer 机制，不作为 KAN promotion。

Line K:
  KAN-specific SNR-to-Basis Lift line。
  它研究为什么 generic parameter-level SNR 在 Rational/KAN 上不能通过 basis/group telemetry，并把 parameter SNR signal 映射进 basis cover。
```

---

# 1. v13.7 结果独立分析

## 1.1 v13.7 有什么进展

v13.7 的 official small-budget route 是：

```text
route = R5-GenericSNRPositiveKANSpecificFailed
minimum_success = S2-MLPGenericSNRPositive
s1_implementation_sanity_pass = 1
mlp_s2_task_pass_count = 6 / 7
rat_basis_s3_task_pass_count = 0 / 7
cover_s4_task_pass_count = 0 / 7
nonrat_substrate_health_pass_count = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
```

随后补跑 plan-scale synthetic official，结果更强：

```text
MLP-SNR task pass = 7 / 7
RAT parameter task pass = X1, X7 only
RAT basis S3 task pass = X1 only
Cover S4 task pass = X1 only
Non-RAT substrate repair scout = 0 pass
```

这说明 v13.7 的第一层重大进展是：**continuous PopRisk-SNR 与 v13.6 one-shot 不同；SNR 作为训练过程机制，在 MLP 上确实有 synthetic positive。**

v13.7 的第二层进展是 MLP real triage。plain real triage 没过；SNR-Blend 修复后 3x3 real triage 通过，5-seed 也通过；但 10-seed 没确认，Fashion-MNIST 只有 3/10，MNIST 5/10，KMNIST 9/10。这个结果说明：**generic SNR optimizer 有真实信号，但 robustness / tail safety 还不足。**

## 1.2 v13.7 没有什么进展

v13.7 没有产生 KAN-specific functional success：

```text
KAN S3 = 0
KAN S4 = 0
KAN S5 = 0
Rational basis/group telemetry = fail
Non-RAT substrate = fail
KAN real short-run = closed
```

Rational 的失败不是因为 cover boundary 太严。Case B 里 no-cover / weak-cover / group-soft 都没有打开 task-level S3；rejection audit 中 cover-boundary rejection 不是主导，den/r'/r''/group diversity 也没有显示明显数值 catastrophe。

因此不能再说：

```text
也许把 cover guard 调弱一点就能成功；
也许 group gate 太硬；
也许 Rational denominator 爆了；
也许再跑 O7/O8/O9 或 BN/BM 就行。
```

这些解释都不够。

## 1.3 本轮真正发现的问题

v13.7 把 blocker 精确定位为：

$$
\boxed{
\text{generic population-risk signal 存在，但 KAN basis telemetry 没有正确承接它。}
}
$$

MLP 的参数空间是普通 dense coordinate，SNR mask 直接作用在参数上可以有效；KAN/Rational 的显式 basis coordinate 增加了额外结构，但当前使用的 group / basis telemetry 反而丢失了 signal。

这意味着 KAN 线的下一步不是继续“设计一个 basis-SNR update”，而是先回答：

```text
parameter-level SNR signal 在投影到 basis/group coordinate 时，到底丢了多少？
丢在 numerator / denominator / readout / residual / projection 的哪个 role？
是 group 划分错了，还是 basis substrate 不能表达这些 update？
是 Rational-only 问题，还是所有 KAN basis 都有？
```

这就是 v13.8 的核心。

---

# 2. 各线当前进展百分比

| 线 | v13.6 后估计 | v13.7 后估计 | 变化 | 判断 |
|---|---:|---:|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 99% | 99% | 0 | 工程闭包强，不是 blocker |
| Historical FHQ / B320-current 工程能力 | 85% frozen | 85% frozen | 0 | 历史强，但 label-informed init 禁用 |
| Label-free FHQ / A-DYN monitor | 12%-15% | 10%-12% | -2 | 已非主线，只低预算监控 |
| Line C 几何审计 | 88% | 88% | 0 | 审计稳定，但不能做 direction source |
| Rational S1 efficient substrate | 85% | 85% | 0 | 仍是唯一稳定 substrate family |
| Rational S1C channel controllability | 60% | 60% | 0 | 可执行 basis-channel movement，但 value 未成立 |
| Non-RAT substrate / S1C | 15%-25% | 10%-20% | -5 | v13.7 Non-RAT substrate repair 仍 0 pass |
| Full basis-param writeback | 82% | 85% | +3 | v13.7 真实 named parameter update / optimizer state 路径更成熟 |
| Population-risk SNR 实现 | 55% | 80% | +25 | continuous training loop、SNRState、per-example gradients 已可用 |
| Population-risk SNR generic mechanism | 5% | 55% | +50 | MLP synthetic 7/7，real 5-seed pass；10-seed 未确认 |
| KAN-specific basis-SNR mechanism | 5% | 8%-10% | +3 | 有局部 X1/X7 signal，但未达 S3/S4 |
| Basis-cover boundary schedule | 5%-10% | 8%-10% | 0 | cover rows 有局部 pass，但非主 blocker |
| MLP real optimizer line | 新增 | 50% | +50 | 3x3/5-seed pass，10-seed fail |
| MLP final generic optimizer claim | 新增 | 35% | +35 | 10-seed 未过，不能 final claim |
| Basis-specific functional official | 0%-5% | 0%-5% | 0 | S5 仍为 0 |
| 整体 next-gen MLP claim | 36%-42% | 42%-48% | +6 | generic SNR 有真实进展；KAN-specific 仍卡住 |

这次整体完成度可以略上调，不是因为 KAN 成功了，而是因为我们终于确认了一件重要事：**population-risk SNR 不是空想，它在 MLP 上可工作；但 KAN 的优势还没有出现。**

---

# 3. 结合文档后的新理论判断

## 3.1 Generalization paper 给出的核心方向

Generalization paper 的重点不是“再造一个 optimizer trick”，而是它在 output-space train-test coupling 下把训练方向分成 signal channel 与 reservoir。它的核心工程启发是：训练方向应该由 per-example gradient 的 drift-vs-diffusion 决定。

对 batch $B$，记 per-example gradient 为：

$$
g_i = J_\theta(x_i)^T\delta_i,
$$

其中 $\delta_i$ 来自统一 loss interface。定义：

$$
\mu_k = \frac{1}{b}\sum_i g_{i,k},
$$

$$
\sigma_k^2 = \frac{1}{b-1}\sum_i (g_{i,k}-\mu_k)^2.
$$

最朴素的 population-risk gate 是：

$$
q_k = 1\left\{\mu_k^2 > \tau\frac{\sigma_k^2}{b-1}\right\}.
$$

v13.7 已经验证：这个思想在 MLP 上有明显信号。因此下一步不能再质疑 “SNR 原理全错”。真正问题是：**KAN basis coordinate 没有正确实现这个 population-risk gate。**

## 3.2 Deep Manifold 给出的核心方向

Deep Manifold 的启发是：神经网络的 node covers / coordinate charts 会随训练移动，boundary condition 是迭代方向的来源。对 KAN 来说，每个 basis family 都是 cover：

```text
Rational:
  denominator / slope / curvature / group function cover

Fourier:
  frequency-band cover

Chebyshev:
  degree-energy cover

RBF/FastKAN:
  center / width / occupancy cover

Wavelet:
  scale / support / local-tail cover
```

v13.7 告诉我们，单纯在 basis group 上硬做 SNR 不够。Deep Manifold 的视角说明：cover 不是静态分组，而是在训练过程中移动的局部坐标。固定 group telemetry 可能把 MLP 参数级 signal 错误压扁。

因此 v13.8 要做的是：

$$
\boxed{
\text{从 generic parameter-SNR 信号出发，学习或构造一个动态 basis-cover lift。}
}
$$

不是：

$$
\boxed{
\text{直接用固定 basis group SNR 替代 parameter SNR。}
}
$$

---

# 4. v13.8 总体方案

v13.8 不再继续 cover threshold / tau / EMA 小网格。它拆成四条主线：

```text
Line G：Generic PopRisk optimizer line
  把 MLP-SNR/Blend 当成独立通用优化器线确认或关闭。

Line K：KAN SNR-to-Basis Lift line
  研究 generic parameter-level SNR 为什么无法迁移成 KAN basis/group advantage。

Line D：Basis substrate line
  Rational 继续作为主 substrate；Non-RAT 先做 substrate repair，不直接 functional。

Line C/R/Z：Geometry audit / code audit / finalizer
  保持审计，不生成 direction。
```

核心目标不是得到一个小 row pass，而是回答三个决定性问题：

```text
Q1: MLP PopRisk-SNR 是否是一个真实、可 10-seed 确认的 generic optimizer？
Q2: Rational/KAN 的失败是因为 parameter-SNR signal 在 basis/group lift 中丢失，还是 substrate 本身不可承载？
Q3: KAN 是否能在 generic optimizer 之上，通过显式 basis cover 提供额外优势？
```

---

# 5. Line G：Generic MLP PopRisk optimizer confirmation

## 5.1 目标

把 v13.7 中 MLP generic SNR 的发现单独确认为一个通用 optimizer line。这个线不属于 KAN promotion。

## 5.2 假设

### H-G1

MLP-SNR/Blend 的 5-seed pass 不是偶然，但 10-seed fail 来自 tail-safety / trust-region 不够，而不是 population-risk signal 消失。

### H-G2

若加入 train-stream trust / loss-interface quantile safety / logit-norm trust，MLP-SNR/Blend 可以达到 10-seed confirmation。

### H-G3

如果 H-G2 失败，则当前 MLP generic SNR 只能算 synthetic + small-seed positive，不能作为稳定 optimizer claim。

## 5.3 方法

候选：

```text
G0: MLP-AdamW baseline
G1: MLP-AdamW-SNRSoft
G2: MLP-AdamW-SNREMA
G3: MLP-AdamW-SNRRoleNorm
G4: MLP-AdamW-SNREMA-Blend25/50/75
G5: MLP-AdamW-SNRRoleNorm-Blend25/50/75
G6: MLP-AdamW-SNRBlend + TrainLossQuantileTrust
G7: MLP-AdamW-SNRBlend + LogitNormTrust
G8: MLP-AdamW-SNRBlend + ActiveFractionSchedule
G9: MLP-AdamW-SNRBlend + PerExampleGradientClip
```

合法性：

```text
1. G6 的 TrainLossQuantileTrust 使用 current train-stream per-example loss distribution，走 generic loss interface；不使用 validation/test/future。
2. G7 使用 current train-stream logits，不使用 CEp99/NLL/ECE audit。
3. 所有 CEp99/NLL/ECE 只作为 audit/gate。
```

## 5.4 记录字段

必须写出：

```text
v138_mlp_generic_optimizer_training.csv
v138_mlp_generic_optimizer_summary.csv
v138_mlp_generic_optimizer_10seed.csv
v138_mlp_generic_failure_table.csv
```

字段：

```text
method
dataset
seed
train_steps / epochs
source_vs_adamw
AUC_time_ratio
AUC_step_ratio
CEp99_delta
NLL_delta
ECE_delta
Brier_delta
margin_p10_delta
active_fraction_mean
active_fraction_p90
removed_update_norm_fraction
cos_snr_adamw
snr_blend_alpha
trust_reject_fraction
train_loss_quantile_p95
logit_norm_p95
per_example_grad_clip_fraction
real_triage_pass
```

## 5.5 Gate

10-seed generic optimizer confirm：

```text
For each dataset:
  seed_pass_count >= 6 / 10

Across datasets:
  dataset_pass_count = 3 / 3
```

单 seed pass：

$$
source\_vs\_adamw \ge 0.005,
$$

$$
AUCtime\_ratio \le 1.0,
$$

$$
CEp99\_delta \le 0.05,
$$

$$
NLL\_delta \le 0.02,
$$

$$
ECE\_delta \le 0.02.
$$

若只过 2/3 dataset，记为 `GenericOptimizerNearPass`，不能 final claim。

## 5.6 可视化

```text
fig_v138_mlp_seed_pass_heatmap.svg
fig_v138_mlp_source_vs_tail_scatter.svg
fig_v138_mlp_auc_vs_ce_p99.svg
fig_v138_mlp_active_fraction_trace.svg
fig_v138_mlp_blend_alpha_pareto.svg
```

## 5.7 失败后 Codex 应先尝试

```text
Case G-A: source_vs_adamw positive but CEp99 fail
  Try TrainLossQuantileTrust -> LogitNormTrust -> PerExampleGradientClip。

Case G-B: AUC_time fail but source positive
  Try active fraction schedule and smaller blend alpha。

Case G-C: KMNIST weak but MNIST/Fashion strong
  Do not dataset-tune. Run same trust policy across all datasets and report slice failure。

Case G-D: 10-seed fails after G6-G9
  Record GenericPopRiskSNRRealNoGo_v13_8; stop MLP optimizer expansion。
```

---

# 6. Line K0：KAN SNR signal transfer audit

## 6.1 目标

解释为什么 MLP parameter-SNR works，但 Rational basis/group SNR 不 work。

核心问题：

$$
\boxed{
\text{generic parameter-level SNR signal 在 basis/group lift 中是否被丢失？}
}
$$

## 6.2 假设

### H-K0a

Rational 的 group/basis telemetry 把有效 parameter-SNR update 投影到错误子空间，导致 signal mass loss。

### H-K0b

Rational 的 role grouping 过粗；readout、numerator、denominator、residual、projection 的 SNR 需要分开处理。

### H-K0c

如果 parameter-SNR 在 Rational 上也稳定失败，则问题不是 basis lift，而是 Rational substrate 与 generic PopRisk 不匹配。

## 6.3 实验对象

```text
RAT-AdamW
RAT-ParameterSNRSoft
RAT-ParameterSNREMA
RAT-GroupSNR
RAT-GroupSNREMA
RAT-BasisSNR
RAT-BasisSNR-CoverPhaseSchedule
```

任务：

```text
X1..X7
seeds = 0,1,2
loss = CE,Brier
train_steps = 200
```

## 6.4 必须记录

```text
v138_snr_transfer_audit.csv
v138_rolewise_snr_audit.csv
v138_basis_group_projection_loss.csv
```

字段：

```text
method
task
seed
loss
role
param_snr_active_fraction
group_snr_active_fraction
basis_snr_active_fraction
param_update_norm
group_update_norm
basis_update_norm
cos_group_vs_param
cos_basis_vs_param
signal_retention_group
signal_retention_basis
false_drop_fraction
false_keep_fraction
role_signal_mass_fraction
role_noise_mass_fraction
readout_signal_mass
numerator_signal_mass
denominator_signal_mass
projection_signal_mass
residual_signal_mass
snr_entropy_param
snr_entropy_group
snr_entropy_basis
source_vs_adamw
CEp99_delta
LineC_pass
```

定义：

$$
signal\_retention\_group =
\frac{\|P_{group} u_{paramSNR}\|_2}{\|u_{paramSNR}\|_2+\epsilon}.
$$

$$
cos\_group\_vs\_param =
\frac{u_{groupSNR}^T u_{paramSNR}}
{\|u_{groupSNR}\|\|u_{paramSNR}\|+\epsilon}.
$$

## 6.5 Gate

Signal transfer audit pass：

$$
\operatorname{median}(signal\_retention\_group) \ge 0.70,
$$

$$
\operatorname{median}(cos\_group\_vs\_param) \ge 0.60.
$$

如果 group/basis lift 不满足这个 gate，就不能继续把 group/basis-SNR 当 functional source。

## 6.6 失败后 Codex 应先尝试

```text
Case K0-A: group retention low
  Implement role-wise SNR lift; do not tune cover threshold。

Case K0-B: role-wise readout dominates signal
  Do not force denominator/group update; use parameter-SNR base + readout-rational decoupling safety。

Case K0-C: denominator role has high false-keep
  Add denominator safety as projection, not as primary signal source。

Case K0-D: parameter-SNR itself fails across tasks
  Stop KAN SNR lift; return to substrate architecture.
```

---

# 7. Line K1：KAN SNR-to-Basis Lift

## 7.1 目标

把 v13.7 中有效的 generic parameter-SNR 作为 source，构造能被 KAN basis cover 承载的 update，而不是直接用固定 group-SNR 替代 parameter-SNR。

## 7.2 候选

```text
K1-RAT-ParamSNR-Only
K2-RAT-ParamSNR-MinCoverGuard
K3-RAT-RoleWiseSNRLift
K4-RAT-DynamicSNRClusterLift
K5-RAT-LowRankSNRCorrector
K6-RAT-ParamSNR-Then-BasisConsolidation
K7-RAT-ParamSNR-BlendAdamW-Then-CoverPhase
```

解释：

```text
K2:
  使用 parameter-SNR 作为主方向；basis cover guard 只做安全投影。

K3:
  readout / numerator / denominator / residual / projection 分 role 处理，避免 group telemetry 混淆。

K4:
  用 per-example gradient covariance 动态聚类参数，得到 data-driven cover groups。

K5:
  在 parameter-SNR update 上加入 low-rank correction，补偿 group projection loss。

K6:
  前半段只做 parameter-SNR plasticity，后半段才加入 basis consolidation。

K7:
  将 SNR update 与 AdamW 同 batch gradient blend，参考 MLP line 的成功结构。
```

## 7.3 记录字段

```text
v138_rat_snr_lift_training.csv
v138_rat_snr_lift_summary.csv
v138_rat_snr_lift_linec.csv
v138_rat_snr_lift_controls.csv
v138_rat_snr_lift_failure_table.csv
```

字段：

```text
method
task
seed
loss
phase
source_vs_adamw
source_vs_best_control
AUC_time_ratio
CEp99_delta
NLL_delta
ECE_delta
CouplingR2_delta
NoiseSignalLeak_delta
Reservoir_delta
active_fraction
blend_alpha
signal_retention
cos_update_adamw
cos_update_param_snr
cover_reject_fraction
den_p01_min
r_prime_p99
r_double_prime_p99
group_diversity
pass_s3
pass_s4
```

## 7.4 Gate

Synthetic KAN-specific S3：

```text
>= 5 / 7 synthetic task families pass;
within each passed family, >= 2 / 3 seeds or >=2 loss interfaces pass;
source_vs_best_control >= 0.005;
AUC_time_ratio <= 1.0;
CEp99/NLL/ECE non-harm;
LineC majority pass;
```

KAN-specific S4 cover pass：

```text
S3 pass;
plus basis-cover boundary schedule improves LineC / tail over K1 without source loss.
```

## 7.5 失败后 Codex 应先尝试

```text
Case K1-A: parameter-SNR works but cover guard hurts source
  Delay cover to phase 2; use K6.

Case K1-B: role-wise lift works only on X1/X7
  Run task-family autopsy; do not promote.

Case K1-C: dynamic cluster unstable across seeds
  Freeze clusters after warmup; compare cluster drift.

Case K1-D: source positive but LineC fail
  Add basis consolidation phase; do not add LineC target direction.

Case K1-E: all K1-K7 fail
  Record KAN_SNRtoBasisLiftNoGo_v13_8 and return to substrate/base architecture.
```

---

# 8. Line D：Basis substrate portfolio

## 8.1 目标

非 B-spline active basis 仍需并行推进，但不能直接进入 functional proof，除非先成为 usable substrate。

Active families：

```text
Rational
Chebyshev
Fourier
RBF/FastKAN
Wavelet
```

B-spline：

```text
Frozen / no active budget。
```

## 8.2 当前状态

```text
Rational:
  stable substrate; main KAN line.

Chebyshev / Fourier / RBF / Wavelet:
  WorkspaceOnly_NotFunctionalSubstrate in v13.7 Line S。
```

## 8.3 每轮必须推进的最小 vertical slice

每轮至少执行：

```text
1. Rational: SNR-to-basis lift main run。
2. Two Non-RAT families: substrate-health repair scout。
3. One Non-RAT family: exact lifetime / task-health vertical slice。
```

优先级：

```text
Rational 60%
Fourier / Chebyshev 25%
RBF / Wavelet 15%
```

## 8.4 Non-RAT substrate-health gate

A Non-RAT candidate 进入 functional proof 前必须满足：

$$
workspace\_raw\_ratio \le 1.25,
$$

$$
workspace\_incremental\_ratio \le 2.00,
$$

$$
step\_ratio \le 1.75,
$$

$$
mean\_delta\_vs\_MLP \ge -0.08,
$$

$$
AUCtime\_ratio \le 2.0,
$$

$$
LineC\_pass\_rate \ge 0.25.
$$

这是 substrate-health gate，不是 final base gate。

## 8.5 Family-specific repair

```text
Fourier:
  frequency-band energy audit;
  high-frequency quarantine;
  lowfreq + residual capacity;
  no high-frequency sweep without noise-leak audit.

Chebyshev:
  degree-energy damping;
  high-degree late-enable;
  recurrence stability;
  no blind K increase.

RBF/FastKAN:
  center occupancy;
  width condition;
  out-of-grid fraction;
  compact capacity repair while preserving workspace.

Wavelet:
  scale/support occupancy;
  local-tail coverage;
  no heavy Morlet/MexicanHat unless local hat/triangle has substrate-health.
```

---

# 9. Line C：Manifold-Channel Audit

Line C 保持审计角色，不生成 direction。

必须记录：

```text
CouplingR2
CouplingCorr
NoiseSignalLeak
RealSignalReservoirRatio
KernelDrift
CEp99
NLL
ECE
Brier
margin_p10
signal_mass_topk
reservoir_fraction
train_probe_coupling_r2
```

核心判断不再是 CouplingR2 单指标，而是：

$$
\boxed{
\text{clean signal 进入 signal channel，noise 不进入 signal channel，真实 signal 不困在 reservoir。}
}
$$

Line C 可用于：

```text
1. gate / audit;
2. failure explanation;
3. no-go boundary;
```

不可用于：

```text
1. functional direction source;
2. parameter update objective;
3. dataset-specific controller。
```

---

# 10. Line R：代码与实现审查

Codex 每轮必须在复盘中写 Implementation Readback，不得只交 CSV。

必须指出：

```text
1. per-example gradient 是如何计算的；
2. loss-interface cotangent 是否 generic；
3. 是否写真实 named parameters；
4. 是否更新 optimizer state；
5. SNR state 是否 persistent；
6. basis/group telemetry 文件和核心函数位置；
7. cover boundary 是否只用 train-stream telemetry；
8. CEp99/NLL/ECE/LineC 是否只做 audit；
9. MLP line 与 KAN line 的 route 是否分开；
10. Non-RAT 未过 substrate 时是否被禁止进入 functional proof。
```

缺少实现读回时，route 必须降级：

```text
R0-ImplementationReadbackIncomplete
```

---

# 11. v13.8 主要 artifacts

必须落盘：

```text
v138_progress_table.csv
v138_route_decision.json
v138_loss_interface_audit.csv
v138_forbidden_information_audit.csv
v138_mlp_generic_optimizer_training.csv
v138_mlp_generic_optimizer_summary.csv
v138_mlp_generic_optimizer_10seed.csv
v138_snr_transfer_audit.csv
v138_rolewise_snr_audit.csv
v138_basis_group_projection_loss.csv
v138_rat_snr_lift_training.csv
v138_rat_snr_lift_summary.csv
v138_rat_snr_lift_linec.csv
v138_rat_snr_lift_controls.csv
v138_nonrat_substrate_health.csv
v138_linec_audit.csv
v138_failure_table.csv
v138_no_go_boundary.md
v138_next_hypothesis_queue.md
v138_implementation_readback.md
v138_code_review_packet.zip
```

---

# 12. 必须生成的可视化

```text
fig_v138_progress_delta.svg
fig_v138_mlp_10seed_pass_heatmap.svg
fig_v138_mlp_source_tail_pareto.svg
fig_v138_snr_transfer_retention_by_role.svg
fig_v138_param_vs_group_snr_cosine.svg
fig_v138_false_drop_false_keep_by_role.svg
fig_v138_rat_snr_lift_task_family_heatmap.svg
fig_v138_linec_signal_reservoir_noise_matrix.svg
fig_v138_nonrat_substrate_status.svg
fig_v138_failure_taxonomy.svg
```

---

# 13. v13.8 Route 定义

```text
R0-ImplementationOrArtifactFailure:
  required artifact 缺失、implementation readback 缺失、forbidden information violation。

R1-GenericSNROptimizerNoGo:
  MLP generic line 在 10-seed confirmation 失败，且 trust/blend repair 已执行。

R2-GenericSNROptimizerConfirmed_KANNotSpecific:
  MLP generic line 过 10-seed，但 KAN SNR-to-basis lift 未过 S3。

R3-SNRSignalLostInBasisLift:
  parameter-SNR 有 signal，但 group/basis retention / cosine 不过。

R4-KANBasisLiftNoGo:
  K0/K1 全部执行后仍无 S3。

R5-KANSpecificS3Opened:
  KAN SNR-to-basis lift synthetic >=5/7 pass，但 S4/S5 未开。

R6-KANSpecificS4Opened:
  KAN SNR-to-basis + cover schedule pass，但 real short-run 未确认。

S5-OfficialFunctionalSuccess:
  label-free substrate/base + KAN-specific functional 在 real 3x3 或更强确认中打过 controls。
```

---

# 14. Stop / Go 规则

## 14.1 如果 MLP generic 10-seed 失败

停止扩展 MLP SNR 小修，记录：

```text
GenericPopRiskSNRRealNoGo_v13_8
```

但仍可以继续 KAN basis line，因为 KAN 可能需要 explicit basis coordinate。

## 14.2 如果 MLP generic 10-seed 成功但 KAN S3 失败

结论：

```text
functional signal is generic optimizer, not KAN-specific。
```

不得写 KAN promotion。继续 Line K0/K1 或转 basis substrate redesign。

## 14.3 如果 K0 显示 SNR signal 在 basis lift 中严重丢失

不得继续 group/cover threshold 小修。必须做：

```text
role-wise lift;
dynamic SNR cluster;
low-rank SNR corrector;
```

如果仍失败，记录：

```text
KAN_SNRtoBasisLiftNoGo_v13_8
```

## 14.4 如果 KAN S3 打开

才能进入 KAN real short-run。否则禁止 real short-run。

## 14.5 如果 Non-RAT substrate-health 仍 0

Non-RAT 不进入 functional proof。继续 substrate architecture repair，不写 family failure。

---

# 15. 这版计划的本质改变

v13.8 不再问：

```text
能不能再调一个 SNR threshold / cover guard / group gate？
```

它问：

$$
\boxed{
\text{MLP 上已经存在的 generic PopRisk signal，为什么不能被 KAN basis coordinate 承接？}
}
$$

这比继续小修更接近本质。若 KAN 不能承接 generic signal，它就不是 next-gen MLP；若 KAN 能承接并进一步改善 signal/reservoir/noise，那么才有 functional update 的真正研究价值。
