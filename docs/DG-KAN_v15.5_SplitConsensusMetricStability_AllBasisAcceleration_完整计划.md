# DG-KAN v15.5：Split-Consensus Metric Stability + All-Basis 并行加速完整计划

> 版本：v15.5 execution plan  
> 基于：v15.04 `SplitConsensusSignalSubspace FunctionalUpdate AllBasisAcceleration` 真实结果复盘  
> 生成时间：2026-05-31（Asia/Singapore）  
> 公式格式：Typora 友好，仅使用 `$...$` 与 `$$...$$`  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no fake / proxy / CPU offload；functional direction 不使用 validation/test/future/query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate / failure explanation，不能生成 direction。  
> 执行原则：promotion fail-closed；exploration continue-open；不允许 action bank / controller / reset route；不允许 G9/G10 / F-CHE8/F-CHE9 式 token search。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是寻找一个局部 positive 的 update trick。总目标是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 functional update 改善训练几何与真实训练轨迹，}
\text{使其优于同 base 的 AdamW / ordinary backprop controls。}
}
$$

最终必须证明：

$$
\boxed{
\text{PureKAN base + functional update}
>
\text{PureKAN base + AdamW / Cautious / MGUP / matched controls}
}
$$

并且收益不能由 MLP/generic optimizer controls、random matched direction、same-active-fraction control、same-rank random subspace、decoupled decay、reset / optimizer-state trick 解释。

## 0.2 v15.04 真实状态

v15.04 不是能力成功，但也不是完全没信号。最新可确认状态：

```text
route = R4-LineCOrTailDominated
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_rows = 0
forbidden_information_violation_sum = 0
no_action_search_violation_sum = 0
direction_source = train_stream_only / optimizer_state / split_gradient
```

核心结果：

```text
Line S:
  line_s_gate_pass = 1
  line_s_best_snr = 1.9560
  k2_best_signal_to_noise_ratio = 10.5196
  k4_best_signal_to_noise_ratio = 1.9560

Line G:
  best_method = G7-D-CHE-SplitConsensusMetricNoProjection [M0-identity/S0-diagonal]
  real_lite = 3/9
  source_mean = 0.4495
  control_equivalent_fraction = 0.0
  bad_event_fraction = 1.0
  tail_fail = 1.0
  linec_fail = 0.6667
  auc_median = 0.8296

Line D:
  best_non_dche_family = D-FOU
  best_non_dche_dataset_seed_pass_count = 0/9

Line M:
  generic_split_consensus_explains = 0
  kan_specific_pass_count = 6
```

因此 v15.04 的真实含义是：

```text
1. split-consensus signal subspace 可观测；这不是纯噪声。
2. metric-only G7 比 projection variants 更强。
3. best row 有正 source，并且没有被 generic controls 全部解释。
4. 但收益进入 tail / LineC 不稳定区域。
5. 当前不能 promotion，也不能继续 G9/G10/action/controller/reset。
```

## 0.3 当前项目进度百分数

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| 代码 / provenance / finalizer 审计 | 99% | 工程闭包强，不是 blocker |
| Historical B320 / FHQ anchor | 85% frozen | 历史强，但 label-informed init 禁用，不能作最终 official claim |
| Line C 几何审计 | 88% | 审计成熟；只能解释失败，不能生成方向 |
| PopRisk / FU / FMS infrastructure | 95% | per-example gradient、second moment、projection telemetry、controls 成熟 |
| Current FU / FMS / proximal / ST-FU family | 0%-3% | v15.0-v15.3 已基本 no-go |
| Split-consensus signal observability | 45%-55% | v15.04 Line S gate pass，是本轮最大正信号 |
| Split-consensus metric-only FU | 25%-35% | G7 real-lite 3/9、source positive、controls 未解释，但 bad event dominates |
| Split-consensus projection variants | 5%-10% | projection 可能杀 value 或放大 bad event |
| D-CHE substrate | 85% | 当前最强 Non-RAT substrate，历史 9/9 eligibility |
| D-CHE split-consensus real transfer | 25%-35% | 3/9 weak signal；不是 S4/S5 |
| Rational substrate | 85% | 稳定，但 reset / optimizer-state route 被 generic confound 打回 |
| D-FOU substrate | 20%-30% | v15.04 best non-D-CHE 仍 0/9；历史信号未复现 |
| D-RBF / FastKAN substrate | 20%-30% | workspace/telemetry 有过信号，task-health 未稳 |
| D-WAV substrate | 15%-20% | 弱线索，低预算保留 |
| Non-D-CHE official FU proof | 0%-5% | 目前不可打开 |
| Official S5 functional success | 0% | 尚未达成 |
| 整体 next-gen MLP claim | 28%-36% | 比 v15.3 略上调；原因是 v15.04 找到 source-positive / control-resistant weak signal，但仍远离 official success |

---

# 1. v15.04 独立分析

## 1.1 这次有没有进展？

有，但不是 S5 能力进展，而是 **signal recovery 进展**。

过去几轮的状态是：

```text
v15.0:
  optimizer-aware FU 无法打开。

v15.1:
  control-residual FU 仍 1/9，source 为负。

v15.2.1:
  function-metric / proximal 全 fail，local positive no transfer。

v15.3:
  split-transfer operator FU 仍 0/9，source 明显为负。

v15.04:
  split-consensus metric-only G7 达到 real-lite 3/9，source_mean = +0.4495，control_equivalent_fraction = 0。
```

这说明 v15.04 **不是完全没进展**。它第一次在最近 v15 系列里把 source 从明显负值推到显著正值，并且 best method 不被 generic controls 解释。

但 v15.04 也没有成功，因为：

```text
bad_event_fraction = 1.0
tail_fail = 1.0
linec_fail = 0.6667
weak_gate_overall = 0
S4/S5 = 0
promotion_allowed = 0
```

所以正确结论不是“SplitConsensus 成功”，而是：

$$
\boxed{
\text{Split-consensus metric-only 找到了有价值 signal，}
\text{但当前 update 不能把它稳定限制在 tail-safe / geometry-safe 区域。}
}
$$

## 1.2 为什么 v15.04 比 v15.3 更值得继续？

v15.3 的 ST-FU 失败是强 no-go：

```text
real_lite = 0/9
source_vs_best_control_mean = -0.4604
control_equivalent_fraction = 1.0
transfer_supported_count = 0
```

v15.04 则变成：

```text
real_lite = 3/9
source_mean = +0.4495
control_equivalent_fraction = 0.0
generic_split_consensus_explains = 0
```

这意味着当前不是“没有 functional signal”，而是“functional signal 没有稳定化”。这比继续修旧 FU / proximal / ST-FU 更有希望。

## 1.3 不能误读的地方

v15.04 不能被写成：

```text
1. S4 success；
2. S5 success；
3. D-CHE-FU official success；
4. all-basis success；
5. LineC/tail-safe success；
6. 可以用 LineC/tail audit metric 设计下一步方向。
```

v15.04 只能写成：

```text
S2/S3 前的 weak exploration signal：
  split-consensus metric-only 有 source-positive、control-resistant signal，
  但被 tail / LineC bad-event gate 阻断。
```

---

# 2. 当前真正卡点

当前 blocker 不是：

```text
1. 缺二阶矩；
2. 缺 cautious alignment；
3. 缺 decoupled decay；
4. 缺 proximal solver；
5. 缺 split-transfer objective；
6. 缺 fallback ladder；
7. 缺 matched controls；
8. 缺 signal subspace observability。
```

这些基本已被 v15.0-v15.04 覆盖。

当前 blocker 是：

$$
\boxed{
\text{split-consensus signal subspace 可见，}
\text{但当前 metric-only FU 缺少 train-stream-only stability constraint，}
\text{导致 source gain 进入 tail / LineC bad-event 区域。}
}
$$

更白话地说：

```text
我们终于找到一个看起来真有用的信号方向；
但它像一脚油门，能让 source 变好，
却同时把模型推到 tail / geometry 不稳定区域。
```

下一步不能继续问：

```text
再加哪个 G9/G10？
再调 projection strength？
再调 alpha grid？
再上 controller/action bank？
```

下一步应该问：

$$
\boxed{
\text{能否在不使用 audit metric 生成方向的情况下，}
\text{用 train-stream-only stability constraint 保留 G7 的 source gain，}
\text{同时降低 tail / LineC bad events？}
}
$$

---

# 3. 下一步总策略

v15.5 的核心不是换一个新 FU family，而是 **稳定化 v15.04 中第一次出现的有效信号**：

```text
主线：
  Split-consensus metric-only signal stabilization。

禁止：
  G9/G10 token search；
  action bank / controller；
  reset route；
  audit-directed direction；
  dataset/seed branch。

允许：
  同一个 split-consensus metric-only mechanism 内的 stability/failure-class fallback；
  train-stream-only stability proxy；
  matched controls；
  all-basis substrate 并行。
```

v15.5 的核心假设：

$$
\boxed{
\text{G7 metric-only 的 source gain 是真实的，}
\text{但需要非 audit-directed stability boundary 才能变成 functional progress。}
}
$$

---

# 4. Hypotheses

## H1：G7 metric-only 有真实 signal，projection variants 破坏或污染它

v15.04 中 best method 是 G7 no-projection metric-only，而不是带 projection 的 variants。因此假设：

```text
hard projection into signal subspace 会导致：
  1. value retention 下降；
  2. tail / LineC risk 上升；
  3. metric-only 比 projection 更稳定。
```

验证方式：同一 split-consensus subspace 下，固定 source signal，比较：

```text
metric-only
soft metric shrink
hard projection
random same-rank projection
same-retention random projection
```

## H2：tail / LineC bad event 可由 train-stream-only stability proxy 预测

注意：不能用 LineC/tail audit metric 生成方向。只能使用 train-stream features：

```text
split loss disagreement
micro-horizon B1/B2 recovery lag
logit RMS drift
train entropy collapse
train margin p10 drift
update cosine with AdamW
per-example loss q90 / q95 generic loss-interface statistic
split-consensus eigengap
projection retention
degree energy drift
```

若这些 proxy 不能预测 bad event，则不能继续 stability gating。

## H3：stability constraint 应该是 trust scalar / abstention，而不是新 direction

我们不再新增 update direction，而是让 G7 metric-only update乘一个 trust scalar：

$$
\Delta\theta_t
=
\eta_{scs}\,s_t\,u_{G7,t}.
$$

其中：

$$
s_t\in[0,1]
$$

只由 train-stream stability proxy 决定。若 proxy 不确定，则 NoOp / smaller commit，而不是换方向。

## H4：D-CHE 是当前唯一 carrier，但 all-basis 不能停

D-CHE 是当前主 carrier；D-FOU/RBF/WAV 仍未过 substrate gate。但 all-basis 必须继续并行，不允许因为 D-CHE 有信号而停止其它 basis。

---

# 5. 实验总览

v15.5 分为九条线：

```text
Line R: Implementation / provenance / no-action-search audit。
Line S: Split-consensus subspace reproducibility and robustness。
Line G: G7 metric-only stabilization on D-CHE。
Line B: Train-stream bad-event proxy validation。
Line C: Causal controls and KAN-specificity audit。
Line P: Micro-horizon path / recovery-lag audit。
Line D: All-basis substrate acceleration。
Line M: MLP / generic controls。
Line Z: final route / no-go / next queue。
```

执行必须并行，不再串行等单线 gate。

---

# 6. Line R：Implementation / provenance / no-action-search audit

## 6.1 目标

确认 v15.5 不重犯 v9/v12 旧错误：

```text
1. 不新增 action token；
2. 不启动 controller；
3. 不使用 action bank；
4. 不使用 reset route；
5. 不用 audit metric 生成 direction；
6. 不按 dataset / seed 分支。
```

## 6.2 必须记录

```text
required_artifact_manifest.csv
forbidden_information_audit.csv
no_action_search_audit.csv
direction_provenance.csv
method_surface_manifest.csv
implementation_readback.md
code_review_packet.zip
```

## 6.3 Hard stop

若出现以下任一项，必须 hard stop：

```text
required_artifact_missing_count > 0
forbidden_information_violation_count > 0
no_action_search_violation_count > 0
direction_uses_validation_test_future_query = 1
direction_uses_linec_tail_auc_calibration_audit = 1
dataset_name_branch = 1
seed_specific_scale = 1
action_token_or_controller_used = 1
```

---

# 7. Line S：Split-consensus subspace reproducibility and robustness

## 7.1 目标

确认 v15.04 的 split-consensus signal subspace 不是偶然 artifact。

## 7.2 方法

继续使用当前 train stream split：

$$
B = B_1 \cup \cdots \cup B_K.
$$

每个 split 梯度：

$$
g_j=\nabla_\theta \mathcal L_{B_j}(\theta).
$$

用 AdamV / role-second-moment whiten：

$$
h_j=M_t^{-1/2}g_j.
$$

构造：

$$
C_{split}=\frac{1}{K(K-1)}\sum_{a\ne b}h_a h_b^T,
$$

$$
N_{split}=\frac{1}{K}\sum_j(h_j-\bar h)(h_j-\bar h)^T,
$$

$$
A_{split}=C_{split}-\lambda_n N_{split}.
$$

## 7.3 必测配置

```text
K = 2, 4, 8
batch_size = 128, 256
metric = identity, AdamVDiag, DegreeRoleSecondMoment
rank = diagonal, role-block, lowrank-r4, lowrank-r8
```

注意：这不是 token search。Line S 只做 subspace observability，不直接 promotion。

## 7.4 记录指标

```text
line_s_rows
line_s_k_sensitivity_rows
signal_to_noise_ratio
effective_rank_A_split
top_eigenvalue_A_split
eigengap_A_split
negative_eigen_fraction
projection_retention_adam_grad
rolewise_consensus
degreewise_consensus
subspace_build_time_ms
subspace_memory_mb
subspace_stability_across_steps
subspace_stability_across_seeds
```

## 7.5 Gate

Exploration subspace gate：

```text
signal_to_noise_ratio >= 1.10
projection_retention_adam_grad >= 0.20
negative_eigen_fraction <= 0.40
effective_rank_A_split >= 1
```

Strong subspace gate：

```text
signal_to_noise_ratio >= 1.50
projection_retention_adam_grad >= 0.30
negative_eigen_fraction <= 0.30
subspace_stability_across_steps >= 0.50
```

## 7.6 Fallback ladder

若 Line S fail，Codex 不能 stop，必须执行：

```text
S-FB1: K=2 vs K=4 vs K=8 sensitivity。
S-FB2: identity vs AdamV vs DegreeRoleSecondMoment。
S-FB3: diagonal vs role-block vs lowrank-r4/r8。
S-FB4: batch128 vs batch256。
S-FB5: subspace no-go certificate。
```

---

# 8. Line G：D-CHE G7 metric-only stabilization

## 8.1 目标

保留 v15.04 的正 signal，同时降低 tail / LineC bad event。只允许同一 G7 metric-only mechanism 的 stability treatment，不允许新增 G9/G10。

## 8.2 更新形式

Base metric-only update：

$$
u_t = -M_{split,t}^{-1/2}g_t.
$$

Stability trust scalar：

$$
\Delta\theta_t = \eta\,s_t\,\nu_t + \Delta\theta_{decoupled}.
$$

其中：

$$
s_t\in[0,1].
$$

`s_t` 不能使用 LineC/CEp99/NLL/ECE/AUCtime/Brier audit；只能使用 train-stream stability proxy。

## 8.3 预注册 variants

```text
G7R: reproduce v15.04 G7 exact semantics。
G7S1: G7 + split-agreement trust scalar。
G7S2: G7 + recovery-lag trust scalar。
G7S3: G7 + logit-RMS / entropy-collapse trust scalar。
G7S4: G7 + per-example-loss-quantile generic trust scalar。
G7S5: G7 + combined train-stream trust scalar。
```

这些不算 G9/G10，因为它们不改变 direction family，不新增 action；只是同一个 G7 metric-only 的 precommit trust scaling。

## 8.4 Matched controls

```text
C0-AdamW
C1-CautiousAdamW
C2-MGUP
C3-RandomSubspaceSameRank
C4-RandomSubspaceSameProjectionRetention
C5-SameActiveFractionRandomMask
C6-SameTrustScalarRandomDirection
C7-NoOpMatchedOverhead
C8-MLP-SameSplitConsensusMetric
```

## 8.5 记录指标

```text
real_lite_pass_count
source_vs_best_control_mean
source_vs_best_control_p10
control_equivalent_fraction
bad_event_fraction
tail_fail_fraction
linec_fail_fraction
auc_fail_fraction
NLL_delta_mean
ECE_delta_mean
CEp99_delta_mean
LineC_pass_rate
AUCtime_median
step_time_ratio
memory_ratio
trust_scalar_mean
trust_scalar_p10
trust_scalar_p90
trust_abstention_fraction
source_retention_vs_G7R
bad_event_reduction_vs_G7R
```

## 8.6 Gate

Weak exploration：

```text
real_lite_pass_count >= 3/9
source_vs_best_control_mean > 0
control_equivalent_fraction <= 0.50
bad_event_fraction < 1.0
```

Meaningful exploration：

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean >= 0.005
control_equivalent_fraction <= 0.50
bad_event_fraction <= 0.60
```

S4 exploration：

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail_fail_fraction <= 0.35
linec_fail_fraction <= 0.35
```

Official S5 不降低，见第 17 节。

## 8.7 Fallback ladder

若 G line fail，Codex 不许 stop，必须执行：

```text
G-FB1: Source-retention vs bad-event tradeoff decomposition。
G-FB2: Trust scalar proxy ablation。
G-FB3: NoOp-abstention audit。
G-FB4: Same-trust random-direction control。
G-FB5: Scale sanity only within fixed eta {0.25x, 0.5x, 1.0x of v15.04 eta}。
G-FB6: failure taxonomy and exhaustion certificate。
```

注意：`G-FB5` 不是无限 strength grid。它只用于确认 v15.04 bad event 是否是 gross scale issue。

---

# 9. Line B：Train-stream bad-event proxy validation

## 9.1 目标

验证 tail / LineC bad event 是否有合法 train-stream proxy 可见。Line B 不生成 direction；它只决定 G7S trust scalar 是否有证据。

## 9.2 Candidate proxies

```text
B1 split_loss_disagreement
B2 recovery_lag_h1/h2/h4
B3 logit_rms_drift
B4 entropy_collapse
B5 margin_p10_drift
B6 update_cosine_to_adam
B7 per_example_loss_q90_q95_generic
B8 split_consensus_eigengap
B9 degree_energy_drift
B10 projection_retention_drift
```

## 9.3 记录指标

```text
proxy_name
auc_predict_bad_event
auc_predict_tail_fail
auc_predict_linec_fail
auc_predict_auc_fail
spearman_proxy_to_source
precision_at_topk
false_positive_rate_on_controls
leave_dataset_out_auc
leave_seed_out_auc
leave_method_out_auc
```

## 9.4 Gate

Proxy exploration gate：

```text
auc_predict_bad_event >= 0.60
leaveout_auc_min >= 0.55
false_positive_rate_on_controls <= 0.40
```

Promotion-enabling proxy gate：

```text
auc_predict_bad_event >= 0.70
leaveout_auc_min >= 0.65
false_positive_rate_on_controls <= 0.25
```

If Line B fails：

```text
Trust scalar cannot use that proxy.
Line G still runs G7R and pure controls.
No hard stop.
```

---

# 10. Line C：Causal controls and KAN-specificity audit

## 10.1 目标

验证 G7 / G7S 的 positive 不是 generic optimizer / MLP / random subspace effect。

## 10.2 Controls

```text
MLP-AdamW
MLP-SplitConsensusMetric
MLP-SameTrustScalar
D-CHE-AdamW
D-CHE-CautiousAdamW
D-CHE-MGUP
D-CHE-RandomSameRank
D-CHE-RandomSameEigenSpectrum
D-CHE-SameProjectionRetentionRandom
D-CHE-SameTrustScalarRandomDirection
D-CHE-NoOpMatchedOverhead
```

## 10.3 Difference-in-differences

For candidate $H$:

$$
\Delta_{KAN-specific}
=
(H_{D-CHE}-Control_{D-CHE})
-
(H_{MLP}-Control_{MLP}).
$$

Require:

```text
Delta_KAN_specific_source > 0
MLP/generic controls do not pass same gate
random subspace controls do not pass same gate
```

---

# 11. Line P：Micro-horizon path / recovery-lag audit

## 11.1 目标

解释 v15.04 的 bad event：它是 early spike、recovery lag、tail drift、LineC drift，还是 source/tail non-colocation。

## 11.2 方法

只对 top methods and controls 跑 train-stream B1/B2 micro-horizon audit：

```text
horizon = 1, 2, 4
```

不提交 probe update，不使用 validation/test/future/query，不用 audit metrics 生成 direction。

## 11.3 记录指标

```text
B1_loss_delta_h1/h2/h4
B2_loss_delta_h1/h2/h4
micro_horizon_integral
recovery_lag
logit_rms_drift_h
entropy_drift_h
margin_p10_drift_h
source_proxy_h
tail_proxy_h
matched_random_effect_h
NoOp_overhead_effect_h
```

## 11.4 Output taxonomy

```text
P1-ImmediateGainRecoveryFail
P2-SplitGainMismatch
P3-TailProxySpike
P4-LineCProxyUnobservable
P5-ControlEquivalentPath
P6-OverheadDominated
```

---

# 12. Line D：All-basis substrate acceleration

## 12.1 目标

D-CHE 不能成为唯一 carrier。v15.5 继续并行推动 D-FOU / D-RBF / D-WAV，但不允许没过 substrate gate 就进入 official FU proof。

## 12.2 D-FOU

预注册方向：

```text
D-FOU62-LowFreqIdentityResidualV5
D-FOU63-BandwiseConsensusMetric
D-FOU64-PhaseStableBandMixV2
D-FOU65-NoMaterializeLifetimeV3
D-FOU66-HighFrequencyQuarantineNoAuditDirection
```

记录：

```text
low_band_energy
mid_band_energy
high_band_energy
phase_drift
high_freq_ratio
bandwise_snr
step_ratio
memory_ratio
AUCtime_ratio
LineC_pass_rate
```

## 12.3 D-RBF / FastKAN

预注册方向：

```text
D-RBF60-CompactBumpIdentityResidualV2
D-RBF61-ActiveCenterConsensusOccupancy
D-RBF62-WidthConditionGuardV2
D-RBF63-GaussianLocalK4NoDenseMaterialization
D-RBF64-CenterDropoutNoTaskBranchDiagnostic
```

记录：

```text
active_center_fraction
center_occupancy_entropy
width_p01
width_p99
oog_fraction
center_snr
step_ratio
memory_ratio
AUCtime_ratio
LineC_pass_rate
```

## 12.4 D-WAV

预注册方向：

```text
D-WAV53-TriangularSupportV5
D-WAV54-ScaleOccupancyConsensus
D-WAV55-SupportOverlapDampingV2
D-WAV56-LocalTailCoverageAuditOnly
```

记录：

```text
scale_occupancy
support_overlap
local_tail_proxy_train_stream
wavelet_energy_by_scale
step_ratio
memory_ratio
AUCtime_ratio
LineC_pass_rate
```

## 12.5 Gate

Exploration substrate gate：

```text
family_dataset_seed_pass_count >= 6/9
step_ratio <= 1.75
memory_ratio <= 1.75
mean_delta_vs_MLP >= -0.05
worst_delta_vs_MLP >= -0.10
LineC_pass_rate >= 0.30
```

Official FMS eligibility：

```text
family_dataset_seed_pass_count = 9/9
step_ratio <= 1.25
memory_ratio <= 1.25
mean_delta_vs_MLP >= -0.02
LineC_pass_rate >= 0.60
```

---

# 13. Line M：MLP / generic controls

每个 positive-looking result 必须跑：

```text
MLP-AdamW
MLP-CautiousAdamW
MLP-MGUP
MLP-SplitConsensusMetric
MLP-SameTrustScalar
MLP-RandomSubspaceSameRank
MLP-NoOpMatchedOverhead
```

If MLP / generic controls explain positive result：

```text
route = R3-GenericControlsExplainResult
KAN-specific claim = 0
promotion_allowed = 0
```

---

# 14. Line X：Transfer operator audit

Line X 不生成 direction，只判断 local positive 是否支持 transfer：

```text
train_motion_to_probe_motion_R2
signal_channel_energy_delta
reservoir_energy_delta
noise_leakage_proxy_delta
source_vs_best_control
local_positive_no_transfer_count
transfer_supported_count
```

Gate：

```text
transfer_supported_count >= 3/9 for exploration
transfer_supported_count >= 6/9 for S4 exploration
```

If local positives remain no-transfer：

```text
R-X-LocalPositiveNoTransfer
```

---

# 15. Visualizations

必须生成：

```text
fig_v155_progress_dashboard.svg
fig_line_s_signal_to_noise_by_K_metric.svg
fig_g7_source_vs_bad_event_scatter.svg
fig_g7_trust_scalar_ablation.svg
fig_bad_event_proxy_auc.svg
fig_source_tail_linec_failure_heatmap.svg
fig_controls_kan_vs_mlp_difference_in_difference.svg
fig_micro_horizon_recovery_lag.svg
fig_allbasis_substrate_pareto.svg
fig_transfer_operator_local_vs_transfer.svg
fig_route_gate_dashboard.svg
```

---

# 16. Route definitions

```text
S5-OfficialFunctionalSuccess:
  Full 9/9 real gate pass, controls fail, promotion allowed.

S4-SplitConsensusMetricRealTransferPositive:
  real_lite >= 6/9 with source, AUC, tail, LineC stable.

S3-SplitConsensusMetricMeaningfulPositive:
  real_lite >= 4/9, source positive, bad_event <= 0.60.

S2-SplitConsensusMetricWeakPositive:
  real_lite >= 3/9, source positive, controls not explain, but not stable enough.

R1-NoSplitConsensusSignalSubspace:
  Line S gate fails after fallback ladder.

R2-MetricSignalProjectionKillsValue:
  projection variants fail while metric-only positive.

R3-GenericControlsExplainResult:
  MLP/generic/random controls explain positive result.

R4-LineCOrTailDominated:
  source positive and controls not explain, but bad_event/tail/LineC dominates.

R5-StabilityProxyUnobservable:
  bad-event stability has no legal train-stream proxy.

R6-AllBasisSubstrateBlocked:
  D-FOU/RBF/WAV fail substrate exploration.

R7-SplitConsensusCurrentDefinitionNoGo:
  Line S/G/B/C/P/D/M all complete, no weak or meaningful positive remains.

R0-ArtifactOrProvenanceViolation:
  required artifacts missing or forbidden/no-action violation exists.
```

---

# 17. Success criteria

## 17.1 Weak exploration

```text
real_lite_pass_count >= 3/9
source_vs_best_control_mean > 0
control_equivalent_fraction <= 0.50
bad_event_fraction < 1.0
```

## 17.2 Meaningful exploration

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean >= 0.005
control_equivalent_fraction <= 0.50
bad_event_fraction <= 0.60
tail_fail_fraction <= 0.60
linec_fail_fraction <= 0.60
```

## 17.3 S4 exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail_fail_fraction <= 0.35
linec_fail_fraction <= 0.35
step_time_ratio <= 1.50
memory_ratio <= 1.50
```

## 17.4 Official S5

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
controls fail
forbidden audit pass
no-action-search audit pass
code review pass
promotion_allowed = 1
```

---

# 18. Stop / continue contract

## 18.1 Promotion fail-closed

Promotion 只有 S5 达成才允许。任何 diagnostic、weak positive、source-only、real-lite、substrate-only、MLP/generic positive 都不能 promotion。

## 18.2 Exploration continue-open

这些情况不能 hard stop：

```text
Line S fail
Line G fail
Line B proxy fail
Line C controls positive
Line P path audit fail
Line D all-basis fail
MLP/generic controls positive
D-CHE real-lite <4/9
bad_event high
LineC/tail/AUC fail
overhead high
```

必须进入 fallback ladder / failure taxonomy / exhaustion certificate。

## 18.3 True hard stop

只有这些可以 hard stop：

```text
required_artifact_missing_count > 0
forbidden_information_violation_count > 0
no_action_search_violation_count > 0
direction uses validation/test/future/query
direction uses LineC/CEp99/NLL/ECE/AUCtime/Brier
dataset-name branch
seed-specific scale
action token / controller / action bank / reset route
fake / proxy / CPU offload
```

## 18.4 Exhaustion certificate

No-go 前必须写：

```text
line_name
planned_methods
executed_methods
fallbacks_executed
deferred_items
deferred_reason
budget_kind
planned_budget
consumed_budget
final_stop_allowed
next_hypothesis_queue
```

No line may use “budget exhausted” without this certificate.

---

# 19. Parallel acceleration plan

v15.5 必须并行执行：

```text
GPU group 0:
  Line S + Line G D-CHE metric-only stabilization.

GPU group 1:
  Line B bad-event proxy + Line P micro-horizon audit.

GPU group 2:
  Line M MLP/generic controls + Line X transfer audit.

GPU group 3:
  Line D D-FOU + D-RBF substrate acceleration.

GPU group 4 if available:
  D-WAV monitor + Rational/D-CHE no-regression + figure generation.
```

最低完成合同：

```text
Line R:
  required / forbidden / no-action / method surface complete.

Line S:
  K=2/4/8 × metric × rank sensitivity complete.

Line G:
  G7R + G7S1..G7S5 + controls complete.

Line B:
  bad-event proxy AUC table + leaveout table complete.

Line C/M:
  every positive-looking row has matched controls.

Line P:
  top-2 G methods + controls micro-horizon audit.

Line D:
  D-FOU and D-RBF full rows；D-WAV low-budget；D-CHE/Rational no-regression.

Line Z:
  route / no-go / failure taxonomy / exhaustion certificate.
```

---

# 20. Next-hypothesis policy

If v15.5 ends with R4 again but source remains positive：

```text
Do not add G9/G10.
Next hypothesis must be theory-level:
  stability-preserving substrate;
  train-stream boundary condition;
  lower-level basis carrier;
  or explicit signal/reservoir train-stream observable.
```

If v15.5 loses source under all stability constraints：

```text
Split-consensus metric-only value is fragile.
Route = R7-SplitConsensusCurrentDefinitionNoGo.
Stop current split-consensus FU definition.
```

If v15.5 gets real_lite >=4/9：

```text
Continue to S4 exploration with same mechanism only;
no token expansion;
increase seeds/budget only after controls pass.
```

If v15.5 gets real_lite >=6/9：

```text
Open S4 confirm;
prepare official 3x3 S5 run;
strict promotion gates unchanged.
```

---

# 21. 最终判断

v15.04 是最近几轮里第一个重新出现 **source-positive + control-resistant** signal 的结果。它不是 S5，也不是 S4，但它说明 split-consensus metric-only 不应该被简单丢掉。

下一步加速的关键不是继续补 method token，而是：

$$
\boxed{
\text{保留 G7 metric-only 的 source gain，}
\text{用合法 train-stream stability boundary 降低 tail / LineC bad event。}
}
$$

如果 v15.5 仍然 source-positive 但 bad-event 不可控，那么当前 split-consensus FU 不能 promotion，但它会给出明确的 theory-level 下一步：需要更稳定的 substrate / boundary condition，而不是继续找好动作。
