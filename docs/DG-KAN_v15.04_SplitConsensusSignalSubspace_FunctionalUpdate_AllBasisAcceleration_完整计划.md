# DG-KAN v15.4：Split-Consensus Signal Subspace Functional Update + All-Basis 并行加速完整计划

生成时间：2026-05-31（Asia/Singapore）

> 本计划基于 v15.3 `SplitTransferOperatorFU AllBasisAcceleration` 真实结果复盘修订。v15.3 已经证明：仅把 B1/B2 split-transfer loss 写进局部 operator solve，仍然不能产生 real-lite / transfer 支持；当前 route 为 `R1-STFUStillLocalPositiveNoTransfer`，promotion 仍关闭。v15.4 不再继续 T6/T7、FU9/FU10、F-CHE8/F-CHE9、action bank、controller、reset route 或 alpha-grid 小修，而是把 functional update 重定义为 **split-consensus signal subspace metric**：先用多个 train split 估计当前训练流中“跨 split 一致”的 signal subspace，再让普通 train-stream gradient 在这个 signal subspace / metric 中更新。

---

# 0. 项目总目标与当前进展

## 0.1 总目标

DG-KAN 的总目标仍然是：

$$
\boxed{
\text{strict FC-PureKAN base} + \text{functional update}
>
\text{same PureKAN base} + \text{ordinary AdamW/backprop controls}
}
$$

这里的 “>” 不只是某个 diagnostic score 更好，而是必须同时满足：

```text
1. 表达力不低于同规模 MLP / AdamW baseline；
2. forward / backward / step / memory 与 MLP 可比；
3. AUC-step / AUC-time 不差；
4. CEp99 / NLL / ECE / calibration / tail 不坏；
5. LineC / signal-reservoir-noise 几何不撕裂；
6. functional update 的收益不能被 AdamW / Cautious / MGUP / random / same-active / same-projection / MLP-generic controls 解释；
7. functional direction 只能使用 train-stream / loss-interface generic 信息，不使用 validation/test/future/query，也不使用 LineC/CEp99/NLL/ECE/AUCtime 生成方向。
```

最终希望证明：KAN 显式 basis / function coordinate 能提供 ordinary optimizer 没有的训练几何优势。

## 0.2 当前进展锚点

截至 v15.3：

```text
D-CHE substrate:
  历史 9/9 eligibility，仍是当前最强 Non-RAT carrier。

D-CHE old FU / FMS / function-metric / proximal / CR-FU / ST-FU:
  均未形成 official real-transfer success。

v15.3 Split-Transfer Operator FU:
  S1 executed，但 S2/S3/S4/S5 = 0。

All-basis:
  D-FOU / D-RBF / D-WAV v15.3 reconfirmation 仍为 0/9。

Official S5:
  0。
```

v15.3 的核心失败不是 artifact 缺失，也不是 Codex 没继续。v15.3 已经执行：Line R/T/M/X/D/C/Z，修复了 objective semantics、commit scale、metric choice、Line X denominator、full-gradient duplicate backward 等实现问题，并补齐 execution contract。最新结果仍然是：

```text
route = R1-STFUStillLocalPositiveNoTransfer
line_t_real_lite_pass_count = 0 / 9
line_t_source_vs_best_control_mean = -0.46042725774976945
line_t_control_equivalent_fraction = 1.0
line_t_bad_event_fraction = 1.0
line_x_gate_pass = 0
generic_stfu_explains = 1
promotion_allowed = 0
```

因此 v15.4 的问题不是“继续修 v15.3 的某个参数”，而是：

$$
\boxed{
\text{为什么 local/split functional solve 仍然不能产生 transfer-visible update？}
}
$$

---

# 1. 对 v15.3 的独立判断

## 1.1 v15.3 不是浅尝辄止

v15.3 已经完成了计划内的主线与 fallback：

```text
1. T0-T5 + TCTRL controls；
2. metric choices M0/M1/M2；
3. Line X transfer operator audit；
4. Line M generic / MLP ST-FU controls；
5. Line D D-FOU52..56 / D-RBF50..54 / D-WAV45..48 substrate reconfirmation；
6. Line C geometry/tail/AUC audit；
7. Line R/Z required manifest、forbidden/no-action audit、gate recompute、execution contract；
8. T-FB1..T-FB6 fallback ladder；
9. full-gradient duplicate backward 修复；
10. pure $U\alpha$ objective 和 commit semantics 修复。
```

所以 v15.3 的 no-go 是可信的。

## 1.2 v15.3 的本质失败

v15.3 的关键数据：

```text
Line T:
  real-lite 0/9；
  source_vs_best_control_mean = -0.4604；
  control_equivalent_fraction = 1.0；
  bad_event_fraction = 1.0；
  best method 仍 LineC fail 9/9。

Line X:
  transfer_supported_dataset_seed_count = 0；
  local_positive_any_dataset_seed_count = 9；
  local_positive_no_transfer_raw_count = 135。

Line M:
  generic_stfu_explains = 1。

Line D:
  best Non-D-CHE family = D-FOU，0/9；
  D-RBF 0/9；
  D-WAV 0/9。
```

这说明：

$$
\boxed{
\text{v15.3 能产生 local/split-looking signal，但这个 signal 不进入 transfer-visible channel。}
}
$$

旧的问题链条已经非常清楚：

```text
v15.0:
  optimizer-aware FU wrapper 不够。

v15.1:
  control-residual FU 仍无剩余 value。

v15.2.1:
  function-metric / proximal 仍 control-equivalent。

v15.3:
  split-transfer operator solve 仍 local-positive-no-transfer。
```

所以 v15.4 不能再写：

```text
T6/T7；
FU9/FU10；
F-CHE8/F-CHE9；
alpha grid 扩展；
再调 strength / lambda / interval；
再开 controller / action bank；
再用 real fail pattern 改 direction。
```

---

# 2. 新核心假设：functional update 缺的不是 another direction，而是 signal-subspace identification

## 2.1 为什么 split-transfer operator 仍失败

v15.3 在 B1/B2 上求解：

$$
\alpha^*
=
\arg\min_\alpha
\mathcal L_{B1}(f_\theta+J_{B1}U\alpha)
+
\lambda_2\mathcal L_{B2}(f_\theta+J_{B2}U\alpha)
+
\rho\|\alpha\|^2
+
\tau\|U\alpha\|_{M_t}^2.
$$

这比单 batch local update 更好，但仍然失败。我的解释是：

```text
B1/B2 split loss 是 local consistency，不等于 population signal channel。
如果 U 子空间本身混有大量 local/noisy/control-equivalent directions，
B1/B2 objective 仍会找到 local-improving 但 transfer-invisible 的方向。
```

也就是说，我们不能先给一个 U，再在 U 内求解。我们必须先问：

$$
\boxed{
U \text{ 是否是跨 train split 一致的 signal subspace？}
}
$$

## 2.2 从 generalization theory 得到的新方向

根据 population-risk / signal-channel 的思路，真正有用的是跨样本 / 跨 minibatch 一致的 drift，而不是单 split local descent。

对 train split $B_j$，定义当前 loss-interface generic gradient：

$$
g_j = \nabla_\theta \mathcal L_{B_j}(\theta).
$$

用 Adam 二阶矩或 role-wise second moment 定义 whitened gradient：

$$
h_j = M_t^{-1/2} g_j.
$$

如果某个方向 $u$ 是 population signal，它应该在多个 splits 上同向：

$$
\langle u,h_a\rangle \langle u,h_b\rangle > 0,
\quad a\ne b.
$$

如果某方向只是 split-specific noise，它可能在单个 split 上好，但跨 split 不一致。

所以 v15.4 的核心是构造 **Split-Consensus Signal Operator**：

$$
C_{split}
=
\frac{1}{K(K-1)}\sum_{a\ne b} h_a h_b^T.
$$

并估计 split-noise operator：

$$
N_{split}
=
\frac{1}{K}\sum_{j=1}^K (h_j-\bar h)(h_j-\bar h)^T.
$$

定义 signal operator：

$$
A_{split}=C_{split}-\lambda_n N_{split}.
$$

functional update 不再是额外方向，而是：

$$
\boxed{
\text{ordinary train gradient 在 split-consensus signal subspace 中更新。}
}
$$

---

# 3. v15.4 总体实验目标

v15.4 的目标不是让某个新 token 过，而是一次性回答以下问题：

```text
Q1. 当前 train-stream 中是否存在可观测的 split-consensus signal subspace？
Q2. 把普通 gradient 限制到这个 signal subspace，是否比 AdamW / Cautious / MGUP / random controls 更好？
Q3. 若 Q2 失败，是 subspace 构造失败、projection kills value、LineC 撕裂、tail blocker、overhead blocker，还是 D-CHE carrier 不适合？
Q4. D-FOU / D-RBF / D-WAV 是否能重新打开 substrate carrier？
```

一句话：

$$
\boxed{
\text{从“找 update direction”改成“识别 signal subspace 并作为 metric 使用”。}
}
$$

---

# 4. v15.4 实验线总览

```text
Line R:
  Provenance / forbidden / no-action-search / execution-contract audit。

Line S:
  Split-consensus signal subspace estimation。

Line G:
  Gradient-aligned signal-subspace functional update on D-CHE。

Line P:
  Path / micro-horizon audit，不生成 direction，只解释 update 是否 transfer。

Line M:
  MLP / generic optimizer controls。

Line D:
  All-basis substrate acceleration，尤其 D-FOU / D-RBF。

Line C:
  LineC / signal-reservoir-noise / tail / calibration audit。

Line Z:
  Route / no-go / exhaustion certificate / next hypothesis。
```

---

# 5. Line S：Split-Consensus Signal Subspace Estimation

## 5.1 目标

判断当前 train stream 中是否存在跨 split 一致的 signal subspace。

## 5.2 数据与 splits

每个 event 使用当前 train stream，拆成 $K$ 个 micro-splits：

```text
K = 4 primary；
K = 2 diagnostic；
K = 8 low-budget diagnostic if overhead allows。
```

每个 split $B_j$ 计算：

```text
g_j = train loss gradient；
h_j = second-moment-whitened gradient；
rolewise h_j；
degreewise h_j for D-CHE；
```

禁止：

```text
validation / test / future / query；
LineC / CEp99 / NLL / ECE / AUCtime / Brier 作为 direction；
dataset-name branch；
seed-specific scale。
```

## 5.3 Operator construction

Whitened gradients：

$$
h_j=M_t^{-1/2}g_j.
$$

Cross-split consensus：

$$
C_{split}=\frac{1}{K(K-1)}\sum_{a\ne b}h_a h_b^T.
$$

Noise / disagreement：

$$
N_{split}=\frac{1}{K}\sum_j(h_j-\bar h)(h_j-\bar h)^T.
$$

Signal operator：

$$
A_{split}=C_{split}-\lambda_n N_{split}.
$$

由于 full matrix 太贵，必须用 sketch：

```text
S0-diagonal:
  只记录 diagonal consensus。

S1-role-block:
  role / degree / basis group block consensus。

S2-lowrank-r4:
  random projection sketch rank 4。

S3-lowrank-r8:
  random projection sketch rank 8。
```

## 5.4 必须记录指标

```text
split_count
sketch_rank
metric_choice
consensus_score_mean
consensus_score_p10
noise_score_mean
signal_to_noise_ratio
effective_rank_A_split
top_eigenvalue_A_split
eigengap_A_split
negative_eigen_fraction
projection_retention_adam_grad
projection_retention_fu_prior
rolewise_consensus
rolewise_noise
degreewise_consensus
high_degree_noise_fraction
subspace_build_time_ms
subspace_memory_mb
```

## 5.5 Line S exploration gate

Line S 不是 promotion gate，只决定是否值得进入 Line G：

```text
signal_to_noise_ratio >= 1.10
projection_retention_adam_grad >= 0.20
negative_eigen_fraction <= 0.40
effective_rank_A_split >= 1
```

如果 Line S 不过，不能 hard stop。必须执行 fallback ladder：

```text
S-FB1: check K=2 vs K=4 split sensitivity；
S-FB2: check diagonal vs role-block vs lowrank-r4；
S-FB3: check AdamV vs DegreeRoleSecondMoment metric；
S-FB4: write subspace no-go certificate。
```

---

# 6. Line G：Gradient-Aligned Signal-Subspace Functional Update

## 6.1 目标

不再新增 residual direction，而是把 ordinary gradient 限制 / 预条件到 signal subspace。

普通 full train gradient：

$$
g=\nabla_\theta\mathcal L_B(\theta).
$$

Whitened gradient：

$$
h=M_t^{-1/2}g.
$$

Signal projection：

$$
h_{sig}=P_{A}h.
$$

更新：

$$
\Delta\theta=-\eta M_t^{-1/2}h_{sig}+\Delta\theta_{decoupled}.
$$

这里 $P_A$ 是 $A_{split}$ 的 positive signal subspace projection。

## 6.2 预注册 methods

```text
G0-D-CHE-AdamW
G1-D-CHE-CautiousAdamW
G2-D-CHE-MGUP
G3-D-CHE-SplitConsensusDiagMetric
G4-D-CHE-SplitConsensusRoleBlockMetric
G5-D-CHE-SplitConsensusLowRank-r4
G6-D-CHE-SplitConsensusLowRank-r8
G7-D-CHE-SplitConsensusMetricNoProjection
G8-D-CHE-SplitConsensusProjectionPlusAdamV
```

解释：

```text
G3-G8 不是新 action token；
它们是同一 split-consensus signal-subspace mechanism 的不同预注册实现层级。
不允许新增 G9/G10。
```

## 6.3 Matched controls

每个 G3-G8 必须配：

```text
C0-AdamW
C1-CautiousAdamW
C2-MGUP
C3-RandomSubspaceSameRank
C4-RandomSubspaceSameProjectionRetention
C5-SameActiveFractionRandomMask
C6-SameDegreeRoleEnergyRandom
C7-NoOpMatchedOverhead
C8-MLP-SameSplitConsensusControl
```

## 6.4 关键指标

```text
real_lite_pass_count
source_vs_best_control_mean
control_equivalent_fraction
bad_event_fraction
AUCtime_ratio_median
CEp99_delta_mean
NLL_delta_mean
ECE_delta_mean
LineC_fail_count
projection_retention
projection_rejection_fraction
signal_subspace_rank
subspace_noise_ratio
step_time_ratio
memory_ratio
```

## 6.5 Exploration gates

Weak exploration：

```text
real_lite_pass_count >= 3/9
source_vs_best_control_mean > 0
control_equivalent_fraction <= 0.70
bad_event_fraction <= 0.60
```

Meaningful exploration：

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean >= 0.005
control_equivalent_fraction <= 0.50
bad_event_fraction <= 0.40
```

S4 exploration：

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail fail reduced vs AdamW
LineC fail <= AdamW
```

Official S5 仍然：

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
promotion_allowed = 1
```

## 6.6 Line G fallback ladder

若 G3-G8 全 fail，不允许直接 stop。必须执行：

```text
G-FB1: ProjectionKillsValue audit
  比较 projection 前后的 train loss delta / B1-B2 delta / source proxy。

G-FB2: SubspaceNoiseAudit
  若 negative_eigen_fraction 或 noise_score 高，记录 signal subspace no-go。

G-FB3: MetricChoiceAudit
  比较 identity / AdamV / DegreeRoleSecondMoment。

G-FB4: ControlResidualizationAudit
  判断 positive-looking rows 是否被 C3-C7 解释。

G-FB5: OverheadAudit
  分解 subspace build / projection / optimizer update / logging 时间。

G-FB6: G-line exhaustion certificate。
```

---

# 7. Line P：Path / Micro-Horizon Audit

## 7.1 目标

Line P 不生成方向，只解释 signal-subspace update 是否真的改善 train path。

## 7.2 记录

对 top-2 G methods + controls，执行 train-stream micro-horizon diagnostic：

```text
horizon = 1, 2, 4
B1/B2/B3 train splits
No commit for audit-only rows unless method already selected by Line G
```

记录：

```text
micro_horizon_loss_integral
recovery_lag
B1_gain
B2_gain
B3_gain
B2_over_B1_gain
B3_over_B1_gain
loss_spike_count
margin_p10_delta_train
entropy_delta_train
logit_rms_delta_train
```

## 7.3 Failure classes

```text
P-B1 LocalOnly:
  B1 gain >0 but B2/B3 <=0。

P-B2 ProjectionKillsTransfer:
  unprojected gradient transfers better than projected update。

P-B3 NoiseSubspace:
  random same-rank subspace matches signal subspace。

P-B4 OverheadDominated:
  source positive but step_time gate impossible。

P-B5 TailDominated:
  source positive but train-tail proxy and CEp99/NLL audit fail。
```

---

# 8. Line D：All-Basis Substrate Acceleration

## 8.1 目标

D-CHE 不能成为唯一 carrier。v15.4 必须继续 D-FOU / D-RBF / D-WAV，但不能把 substrate-only rows 写成 FU success。

## 8.2 D-FOU

预注册：

```text
D-FOU57-LowFreqIdentityResidualV5
D-FOU58-BandwiseSNRWarmupV3
D-FOU59-PhaseStableBandMixV3
D-FOU60-NoMaterializeLifetimeV3
D-FOU61-HighFrequencyQuarantineV3
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

## 8.3 D-RBF / FastKAN

预注册：

```text
D-RBF55-CompactBumpIdentityResidualV4
D-RBF56-ActiveCenterOccupancyRepairV4
D-RBF57-WidthConditionGuardV4
D-RBF58-GaussianLocalK4NoDenseV3
D-RBF59-CenterSNRWarmupV2
```

记录：

```text
center_occupancy_entropy
empty_center_fraction
width_p01
width_p99
active_center_snr
out_of_grid_fraction
step_ratio
memory_ratio
AUCtime_ratio
LineC_pass_rate
```

## 8.4 D-WAV

预注册：

```text
D-WAV49-TriangularSupportV5
D-WAV50-ScaleOccupancyV4
D-WAV51-SupportOverlapDampingV3
D-WAV52-LocalTailCoverageAuditV3
```

记录：

```text
scale_occupancy
support_overlap
local_tail_proxy
wavelet_high_scale_energy
step_ratio
memory_ratio
AUCtime_ratio
LineC_pass_rate
```

## 8.5 Substrate gates

Exploration substrate gate：

```text
family_dataset_seed_pass_count >= 6/9
step_ratio <= 1.75
memory_ratio <= 1.75
mean_delta_vs_mlp >= -0.05
worst_delta_vs_mlp >= -0.10
LineC_pass_rate >= 0.30
```

Official FU eligibility：

```text
family_dataset_seed_pass_count = 9/9
step_ratio <= 1.25
memory_ratio <= 1.25
LineC_pass_rate >= 0.70
```

D-FOU/RBF/WAV 未达 6/9 不能 hard stop 全轮，必须写 family failure taxonomy。

---

# 9. Line M：MLP / Generic Controls

每个 positive-looking G result 必须比较：

```text
MLP-AdamW
MLP-CautiousAdamW
MLP-MGUP
MLP-SplitConsensusDiagMetric
MLP-SplitConsensusLowRank-r4
MLP-RandomSubspaceSameRank
MLP-NoOpMatchedOverhead
```

如果 MLP / generic controls 解释 gain：

```text
KAN-specific claim = 0
route = R-M-GenericSplitConsensusExplainsGain
```

如果 D-CHE gain > MLP gain，记录 difference-in-differences：

$$
\Delta_{KAN-specific}
=
(G_{D-CHE}-C_{D-CHE})
-
(G_{MLP}-C_{MLP}).
$$

---

# 10. Line C：Geometry / Tail / Transfer Audit

Line C 只做 audit，不生成方向。

记录：

```text
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
NLL_delta
ECE_delta
Brier_delta
margin_p10_delta
signal_like_displacement_fraction
reservoir_like_displacement_fraction
noise_leakage_proxy_delta
LineC_pass
```

必须画：

```text
fig_split_consensus_spectrum.svg
fig_projection_retention_vs_source.svg
fig_signal_noise_ratio_by_method.svg
fig_real_lite_heatmap_3x3.svg
fig_linec_tail_failure_heatmap.svg
fig_allbasis_substrate_status.svg
fig_controls_explain_fraction.svg
```

---

# 11. Stop / Continue 制度

## 11.1 Promotion fail-closed

S5 不降低：

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

## 11.2 Exploration continue-open

这些不能 hard stop：

```text
Line S fail；
Line G fail；
Line P fail；
Line D fail；
Line M controls positive；
D-FOU/RBF/WAV <6/9；
D-CHE real-lite <4/9；
LineC/tail/AUC fail；
overhead high；
subspace noisy；
projection kills value。
```

它们必须进入 fallback ladder / failure taxonomy / exhaustion certificate。

## 11.3 真正 hard stop

只有这些可以 hard stop：

```text
required_artifact_missing_count > 0；
forbidden_information_violation_count > 0；
no_action_search_violation_count > 0；
direction uses validation/test/future/query；
direction uses LineC/CEp99/NLL/ECE/AUCtime/Brier；
dataset-name branch；
seed-specific scale；
action token / controller / action bank / reset route；
fake / proxy / CPU offload。
```

---

# 12. 加速执行安排

v15.4 必须并行，不再串行等待一个 gate。

```text
GPU group 0:
  Line S + Line G D-CHE split-consensus signal-subspace update。

GPU group 1:
  Line M MLP / generic controls。

GPU group 2:
  Line D D-FOU + D-RBF substrate acceleration。

GPU group 3:
  Line D D-WAV + D-CHE/Rational no-regression。

GPU group 4 if available:
  Line P micro-horizon audit + Line C geometry/tail audit。
```

最低完成合同：

```text
Line S:
  S0/S1/S2/S3 + S-FB1..S-FB4。

Line G:
  G0..G8 + C0..C8 + G-FB1..G-FB6。

Line P:
  top-2 G methods + controls。

Line D:
  D-FOU and D-RBF full rows；D-WAV low-budget；D-CHE no-regression。

Line M:
  every positive-looking result matched controls。

Line C:
  geometry/tail/failure heatmap。

Line Z:
  route/no-go/exhaustion certificate/next queue。
```

---

# 13. Route definitions

```text
S1-SplitConsensusSubspaceObservable:
  Line S gate pass。

S2-SplitConsensusFUExplorationPositive:
  Line G weak exploration pass。

S3-SplitConsensusFUMeaningfulPositive:
  Line G meaningful exploration pass。

S4-SplitConsensusFURealLitePositive:
  Line G S4 exploration pass。

S5-OfficialFunctionalSuccess:
  full 9/9 official gate pass。

R1-NoSplitConsensusSignalSubspace:
  Line S and S-FB all fail。

R2-SubspaceExistsProjectionKillsValue:
  Line S pass but projection/update source negative and P-FB confirms projection kills transfer。

R3-GenericControlsExplainSplitConsensus:
  MLP/generic/random subspace controls explain positive-looking results。

R4-LineCOrTailDominated:
  source positive but LineC/tail failure dominates and cannot be separated without audit-directed direction。

R5-AllBasisSubstrateBlocked:
  D-CHE only carrier; D-FOU/RBF/WAV all <6/9。

R6-SplitConsensusFUCurrentDefinitionNoGo:
  Line S/G/P/D/M/C all completed, no exploration pass, no legal fallback remains。
```

---

# 14. 最终判断

v15.3 的关键教训是：

```text
local/split proximal objective 仍然可以产生 local positive，
但这不等于 transfer-visible functional update。
```

v15.4 的根本改变是：

$$
\boxed{
\text{先识别跨 split 一致的 signal subspace，}
\text{再让普通 gradient 在这个 subspace / metric 中更新。}
}
$$

如果 v15.4 也失败，并且 random / MLP / generic controls 继续解释所有 positive，那么必须正式关闭当前 train-stream functional-update family，转向更底层 substrate / graph-free primitive / theoretical update definition，而不是继续堆 FU token。

