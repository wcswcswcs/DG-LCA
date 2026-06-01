# DG-KAN v15.3：Split-Transfer Operator Functional Update + All-Basis 并行加速完整计划

> 版本：v15.3 execution plan  
> 日期：2026-05-31 Asia/Singapore  
> 目标：停止 current FU / FMS / function-metric / proximal 小修；继续推进 functional update 主线，但把它重定义为 **train-split transfer operator functional update**。同时保留 all-basis substrate 并行加速，不让 D-CHE 独占路线。  
> 公式格式：Typora 友好，仅使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no fake/proxy/CPU offload；functional direction 不使用 validation/test/future/query batch；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit/gate，不能作为 direction source；不新增 action token / controller / action bank / reset route。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找一个只在某个 diagnostic 上好看的 update。项目目标是：

$$
\boxed{
\text{在 strict FC-PureKAN substrate 上，通过 functional update 获得比普通 AdamW/backprop 更好的训练轨迹、泛化几何和效率。}
}
$$

最终要证明：

$$
\boxed{
\text{PureKAN base + functional update}
>
\text{PureKAN base + ordinary optimizer controls}
}
$$

并且同时满足：

```text
1. task/source gain 不被 NoOp / Random / AdamW / Cautious / MGUP / same-active / same-projection / same-value-retention controls 解释；
2. real 3x3 或预注册 real-lite gate 成立；
3. LineC / tail / calibration / AUCtime 不坏；
4. step/memory overhead 不越界；
5. 不使用 validation/test/future/query/audit target 生成方向；
6. 不靠 dataset/seed 分支或局部 positive row 拼接成功。
```

## 0.2 当前进展摘要

v15.02.1 已经完成 Explore-Open execution contract，不是 Codex 单线失败后早停。Line R/G/P/X/D/M/C/Z 都落盘，G/P fallback ladder 与 exhaustion certificate 已完成，Line M 对 283 个 positive-looking rows 做 matched control map，generic controls 解释率为 100%。最终 route 为：

```text
route = R15_2_1-CurrentFunctionalDefinitionNoGo
minimum_success = S1-ExploreOpenExecutionContractExecuted
S2/S3/S4/S5 = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

这说明：

```text
1. current function-metric / proximal / CR-FU / optimizer-aware FU family 已经不应继续小修；
2. failure 不是 artifact 缺失，也不是没有执行 fallback；
3. D-CHE substrate 仍是当前最强 carrier，但 D-CHE 上的旧 FU 定义没有 real-transfer value；
4. D-FOU / D-RBF / D-WAV 在 v15.02.1 中仍是 0/9，不允许 official FU proof；
5. 下一步必须换 functional definition，而不是继续 FU9/FU10/F-CHE8/action/controller/reset。
```

## 0.3 当前各线完成度

| 线 | 完成度 | 当前状态 |
|---|---:|---|
| Code / provenance / finalizer | 99% | required / forbidden / no-action-search audit 成熟 |
| Historical B320/FHQ anchor | 85% frozen | 历史强，但 label-informed init 禁用，不能做 final official claim |
| Line C geometry audit | 88% | 审计稳定；不能生成方向 |
| PopRisk / FU / FMS infrastructure | 95% | per-example gradient、alignment、second moment、projection telemetry、controls 已成熟 |
| Current FU direction family | 0%-3% | v15.0 / v15.1 / v15.02.1 已基本 no-go |
| Function-metric update | 0%-3% | v15.02.1 Line G 0/9，control-equivalent=1 |
| Function-space proximal update | 0%-3% | v15.02.1 Line P 0/9，control-equivalent≈0.933 |
| D-CHE substrate | 85% | 当前最强 Non-RAT substrate，历史 9/9 eligibility |
| D-CHE old FU real-transfer | 0%-5% | old FU / FMS / function-metric 不能再小修 |
| Split-transfer operator FU | 0% | v15.3 新主线，尚未执行 |
| Rational substrate | 85% | 稳定；reset/optimizer-state route 被 generic confound 打回 |
| D-FOU substrate | 20%-30% | 历史有 6/9，但 v15.02.1 0/9，需要重新做 substrate acceleration |
| D-RBF / FastKAN substrate | 20%-30% | workspace/telemetry 有信号，task-health 仍不稳 |
| D-WAV substrate | 15%-20% | 弱线索，低预算保留 |
| Non-D-CHE official FU proof | 0%-5% | 目前不可打开 |
| Official S5 functional success | 0% | 尚未达成 |
| Overall next-gen MLP claim | 25%-33% | 当前 FU family no-go 后必须下调；下一步靠新 functional definition 与 substrate carrier |

---

# 1. v15.02.1 的独立分析

## 1.1 v15.02.1 有没有进展？

有，但只是 **执行合同闭合与机制排除进展**，不是能力进展。

v15.02.1 的进展是：它证明 Codex 没有继续“浅尝辄止”。Line G/P fallback ladder、Line D family-specific fallback certificate、Line M matched control map、Line X transfer audit、Line C tail audit、Line Z no-go taxonomy 都已完成。也就是说，这次 no-go 不是因为没跑够，而是因为 current function-metric / proximal definition 本身没有打开。

## 1.2 Line G 说明什么？

Line G 的 function-metric update 全部失败：

```text
line_g_candidate_count = 6
real_lite_pass_count = 0/9
source_vs_best_control_mean = -0.07151905742902605
control_equivalent_fraction = 1.0
bad_event_fraction = 1.0
best_g_method = G8-D-CHE-CombinedAdamV_FunctionMetric_Alignment
```

其中 G6/G8 的 AUCtime 接近 1.02，比 G3/G4/G5/G7 的 1.275 好，但 source 仍然为负，且 dataset-seed pass 为 0。结论：**把 function/basis 信息作为 preconditioner 包在普通 gradient 外面，当前实现没有独立 value。**

## 1.3 Line P 说明什么？

Line P 的 function-space proximal update 也失败：

```text
line_p_candidate_count = 5
real_lite_pass_count = 0/9
source_vs_best_control_mean = -0.007147685686747233
control_equivalent_fraction = 0.9333333333333333
bad_event_fraction = 1.0
best_p_method = P2-D-CHE-PerExampleGradLowRankProx
```

它比 Line G 的 source 均值更接近 0，但依然没有通过任何 real-lite gate，而且 matched controls 能解释大部分 apparent positive。结论：**仅在参数子空间或低秩 function approximation 上做 proximal，不足以产生 transfer gain。**

## 1.4 Line X 说明什么？

Line X 有 10 个 positive-looking rows，但：

```text
x_transfer_supported_count = 0
x_local_positive_no_transfer_rows = 10
line_x_route = R-X-LocalPositiveNoTransfer
```

这非常关键。它说明当前 local positive 仍不能保证 train-to-probe / train-to-transfer 方向成立。换句话说：**旧方法最核心的问题不是局部 loss 或局部 source，而是 local effect 不转移。**

## 1.5 Line D 说明什么？

Line D 中 D-FOU / D-RBF / D-WAV 全部仍是 0/9：

```text
D-FOU: 45 rows, 0/9, max mean delta vs MLP = -0.109375
D-RBF: 45 rows, 0/9, max mean delta vs MLP = -0.328125
D-WAV: 36 rows, 0/9, max mean delta vs MLP = -0.0234375
```

虽然 Line D 的 fallback certificate 已覆盖 lineage replay、gate mismatch、budget mismatch、family-specific decomposition，但没有打开新的 official FU carrier。因此 D-CHE 仍是唯一可继续 functional proof 的 Non-RAT carrier。

## 1.6 Line M 说明什么？

Line M 对所有 positive-looking rows 做了 matched controls：

```text
positive_looking_rows = 283
positive_rows_checked_by_m = 283
generic_control_explains_positive_fraction = 1.0
generic_control_explains_positive = 1
```

这说明 v15.02.1 的局部 positive 不能被写成 KAN-specific functional value。所有看起来有希望的行，都被 generic / MLP / matched controls 解释。

## 1.7 当前真正 blocker

当前 blocker 不是：

```text
没有二阶矩；
没有 cautious alignment；
没有 decoupled decay；
没有 proximal solver；
没有 fallback ladder；
没有 matched controls；
没有 all-basis fallback；
Codex 没继续。
```

这些都已经覆盖。

当前真正 blocker 是：

$$
\boxed{
\text{functional update 仍没有被定义成一个能产生 train-to-transfer effect 的 operator。}
}
$$

旧 FU family 的共同问题是：它们都在参数空间或参数子空间里生成 update，然后再看是否 transfer。v15.02.1 证明这个方向在当前 substrate / metric / proximal family 下已经耗尽。

---

# 2. v15.3 的核心重置：Split-Transfer Operator Functional Update

## 2.1 为什么必须换成 split-transfer operator？

`A Theory of Generalization in Deep Learning` 给出的核心启发是：训练成功不是单点 loss descent，而是 train motion 是否落在 test-visible signal channel。当前 v15.02.1 的 Line X 已经直接显示 local positive 不 transfer。因此下一步 functional update 必须把 **transfer** 放进更新定义本身。

不能再做：

```text
先造一个参数 update -> 看是否 transfer。
```

必须改成：

```text
用 train split B1 / B2 直接估计 transfer-supporting update。
```

其中 B1 和 B2 都来自当前 train stream，不是 validation/test/future/query；因此合法。

## 2.2 基本定义

取当前训练 batch，拆成两个 train split：

```text
B1 = inner update split
B2 = transfer-observability split
```

当前模型为 $f_\theta$。选择一个预注册参数子空间 $U$，例如：

```text
U0: AdamW update direction span
U1: per-example gradient top-r low-rank span
U2: D-CHE degree-role span
U3: output-Jacobian sketch span
U4: random matched-norm control span
```

令候选 update 为：

$$
\Delta\theta = U\alpha.
$$

一阶 function displacement：

$$
\Delta f_{B1}=J_{B1}U\alpha,
$$

$$
\Delta f_{B2}=J_{B2}U\alpha.
$$

v15.3 的核心 solver 不再只优化 B1，而是要求 B1 improvement 与 B2 transfer 同时成立：

$$
\alpha^*
=
\arg\min_\alpha
\mathcal L_{B1}(f_\theta + J_{B1}U\alpha)
+
\lambda_2 \mathcal L_{B2}^{proxy}(f_\theta + J_{B2}U\alpha)
+
\rho\|\alpha\|_2^2
+
\tau R_{trust}(\alpha).
$$

这里 $\mathcal L_{B2}^{proxy}$ 仍然是 train loss interface，不是 validation/test。它只回答：**这个 update 是否在另一个 train split 上也有同向价值。**

## 2.3 为什么它不是 action search？

它不是从一堆动作里选一个。它是一个固定的 operator solve：

```text
fixed split protocol
fixed subspace U
fixed objective
fixed alpha grid or closed-form low-rank solve
fixed controls
```

禁止：

```text
新增 ST-FU6/ST-FU7；
根据 dataset/seed fail pattern 改 objective；
根据 LineC/CEp99/NLL/ECE/AUCtime 反推 objective；
controller；
action bank；
reset route。
```

允许：

```text
同一 solver 内做 split-size sanity；
subspace quality audit；
B1/B2 disagreement decomposition；
matched controls；
runtime breakdown。
```

---

# 3. v15.3 假设

## H1：旧 FU family 失败是因为只优化 local effect，不优化 transfer effect

证据：v15.02.1 Line X 出现 `LocalPositiveNoTransfer`。如果 H1 成立，split-transfer objective 应该至少提升：

```text
x_transfer_supported_count
real_lite_pass_count
source_vs_best_control_mean
control_equivalent_fraction
```

## H2：D-CHE 是可用 carrier，但需要 transfer-aware FU，而不是 function-metric wrapper

D-CHE substrate 历史 9/9，synthetic FMS 历史 5/7，但 real transfer 不成立。若 H2 成立，ST-FU 在 D-CHE 上应超过 AdamW / generic controls，并至少达到 real-lite >=3/9 exploration。

## H3：如果 ST-FU 在 MLP 上同样成功，则它是 generic optimizer，不是 KAN-specific

这不是失败，但不能写成 KAN-specific success。必须记录：

$$
\Delta_{KAN-specific}
=
(D\text{-}CHE+STFU - D\text{-}CHE+AdamW)
-
(MLP+STFU - MLP+AdamW).
$$

## H4：Non-D-CHE basis 当前不是 functional blocker 的主因，但仍必须并行推进

D-FOU / D-RBF / D-WAV 目前 0/9，不能 official FU proof。但它们不能停。v15.3 中 Line D 继续 substrate acceleration，不抢 ST-FU 主线，但必须跑到 failure taxonomy 而不是 0/9 早停。

---

# 4. 实验总结构

v15.3 分为七条线：

```text
Line R: Implementation / provenance / no-action-search audit
Line T: Split-transfer operator FU on D-CHE
Line M: MLP / generic ST-FU controls
Line X: Transfer operator audit
Line D: All-basis substrate acceleration
Line C: Geometry / tail / AUC audit
Line Z: Route / no-go / exhaustion certificate / next hypothesis
```

并行执行，不再串行等待一个 gate：

```text
GPU group 0:
  Line T D-CHE ST-FU main + controls

GPU group 1:
  Line M MLP ST-FU controls

GPU group 2:
  Line D D-FOU / D-RBF substrate acceleration

GPU group 3:
  Line X transfer operator audit + Line C metrics

GPU group 4 if available:
  D-CHE no-regression + Rational monitor + D-WAV low-budget
```

---

# 5. Line R：Implementation / provenance / no-action-search audit

## 5.1 目标

确保本轮不是旧 action search 复燃。

## 5.2 必须记录 artifact

```text
v153_method_surface_manifest.csv
v153_direction_provenance.csv
v153_forbidden_information_audit.csv
v153_no_action_search_audit.csv
v153_required_artifact_manifest.csv
v153_code_review_manifest.csv
v153_execution_contract_coverage_audit.csv
```

## 5.3 硬禁止项

任何行出现以下任一项，立即 hard stop：

```text
uses_validation_for_direction = 1
uses_test_for_direction = 1
uses_future_for_direction = 1
uses_query_batch_for_direction = 1
uses_LineC_as_direction = 1
uses_CEp99_as_direction = 1
uses_NLL_as_direction = 1
uses_ECE_as_direction = 1
uses_AUCtime_as_direction = 1
uses_dataset_name_branch = 1
uses_seed_specific_scale = 1
is_action_token_extension = 1
controller_executed = 1
action_bank_used_as_search_space = 1
reset_route_used = 1
fake_or_proxy_row = 1
cpu_offload_used = 1
```

---

# 6. Line T：D-CHE Split-Transfer Operator FU

## 6.1 方法列表

只允许以下预注册 methods：

```text
T0-D-CHE-AdamW
T1-D-CHE-STFU-AdamSubspace
T2-D-CHE-STFU-PerExampleGradLowRank-r4
T3-D-CHE-STFU-PerExampleGradLowRank-r8
T4-D-CHE-STFU-DegreeRoleSubspace
T5-D-CHE-STFU-OutputJacobianSketch-r4
TCTRL-D-CHE-STFU-RandomMatchedSubspace
TCTRL-D-CHE-STFU-SameNormNoTransfer
TCTRL-D-CHE-STFU-B1OnlyProximal
TCTRL-D-CHE-STFU-B2ShuffledCotangent
```

禁止 T6/T7。

## 6.2 Split protocol

每个 train batch 拆为：

```text
B1: 50%
B2: 50%
```

如果 batch 太小，允许全局 fallback：

```text
B1/B2 = 75%/25%
```

但必须是全局设置，不能按 dataset / seed 分支。

## 6.3 Objective

基本 objective：

$$
J(\alpha)
=
\mathcal L_{B1}(f+J_{B1}U\alpha)
+
\lambda_2 \mathcal L_{B2}(f+J_{B2}U\alpha)
+
\rho \|\alpha\|_2^2
+
\tau \|U\alpha\|_{M_t}^2.
$$

其中 $M_t$ 可选：

```text
M0: identity
M1: Adam v_t diagonal
M2: D-CHE degree-role second moment
```

为避免 method explosion，metric choice 只作为 factor，不生成新 method token。

## 6.4 Alpha solve

第一版使用固定 alpha candidates：

```text
alpha_scale = 0, 0.025, 0.05, 0.10, 0.20
```

如果低秩 subspace 维度 $r > 1$，使用 ridge closed-form / small Cholesky solve，并记录：

```text
solver_type = alpha_grid / ridge_closed_form / cholesky
subspace_rank
condition_number
solve_time_ms
```

## 6.5 必须记录指标

```text
method
dataset
seed
step
split_protocol
subspace_type
subspace_rank
alpha_norm
B1_loss_delta_proxy
B2_loss_delta_proxy
B1_B2_agreement
B2_transfer_gain_proxy
J_B1_U_norm
J_B2_U_norm
subspace_condition
update_norm
update_over_adam_norm
cos_update_adam
cos_update_neg_grad
same_norm_random_gap
b1_only_gap
b2_shuffled_gap
source_vs_best_control
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass
step_time_ratio
memory_ratio
strict_pass
real_lite_pass
fail_reason
```

## 6.6 Exploration gate

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
tail_fail_reduced_vs_AdamW = 1
LineC_fail_count <= AdamW
```

S5 official 不降低：

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

## 6.7 Fallback ladder

Line T 主方法失败不能立刻 no-go。必须执行：

```text
T-FB1: Split agreement decomposition
  判断 B1 gain 是否为正、B2 是否反向。

T-FB2: Subspace quality audit
  判断 U 是否太弱、condition 是否太差、J_B2_U 是否近 0。

T-FB3: Control residualization audit
  判断 update 是否被 B1Only / Random / SameNorm / AdamW controls 解释。

T-FB4: Scale sanity
  只在 alpha_scale 全局 grid 内复核，不允许新增 grid。

T-FB5: Runtime overhead decomposition
  拆分 JVP/VJP / solve / commit / logging cost。

T-FB6: Exhaustion certificate
  写明 T0-T5 + controls + fallback 是否完成。
```

---

# 7. Line M：MLP / generic ST-FU controls

## 7.1 目标

判断 ST-FU 是否只是 generic optimizer。

## 7.2 方法

```text
M0-MLP-AdamW
M1-MLP-STFU-AdamSubspace
M2-MLP-STFU-GradLowRank-r4
M3-MLP-STFU-OutputJacobianSketch-r4
MCTRL-MLP-RandomMatchedSubspace
MCTRL-MLP-B1OnlyProximal
MCTRL-MLP-B2ShuffledCotangent
```

## 7.3 判断

如果 MLP-STFU 与 D-CHE-STFU 同样成功，则结论为：

```text
generic ST-FU positive；KAN-specific claim not established。
```

如果 D-CHE-STFU 显著强于 MLP-STFU，才可讨论：

```text
KAN basis/substrate gives transfer operator leverage。
```

需要记录：

$$
\Delta_{KAN-specific}
=
(D\text{-}CHE+STFU - D\text{-}CHE+AdamW)
-
(MLP+STFU - MLP+AdamW).
$$

---

# 8. Line X：Transfer operator audit

## 8.1 目标

Line X 只审计，不生成方向。

它要回答：

```text
ST-FU 是否真的提升 train split transfer，而不是 B1 local positive？
```

## 8.2 指标

```text
train_to_train_split_R2
B1_to_B2_transfer_R2
B1_to_B2_transfer_cosine
local_positive_no_transfer_count
transfer_supported_count
reservoir_like_displacement_fraction
signal_like_displacement_fraction
noise_leakage_proxy_delta
source_transfer_gap
```

## 8.3 Gate

Exploration：

```text
transfer_supported_count >= 3/9
local_positive_no_transfer_count reduced vs v15.02.1
```

如果 Line X 继续全是 LocalPositiveNoTransfer，则 ST-FU route 必须降级：

```text
R-X-SplitTransferStillNoTransfer
```

---

# 9. Line D：All-Basis substrate acceleration

## 9.1 目标

不允许 D-CHE 成为唯一 carrier。D-FOU / D-RBF / D-WAV 继续 substrate acceleration，但未过 gate 不进入 official FU proof。

## 9.2 D-FOU

候选：

```text
D-FOU52-LowFreqIdentityResidualV5
D-FOU53-BandwiseSNRWarmupV3
D-FOU54-PhaseStableBandMixV3
D-FOU55-NoMaterializeLifetimeV3
D-FOU56-HighFrequencyQuarantineV2
```

必须记录：

```text
low_band_energy
mid_band_energy
high_band_energy
phase_drift
bandwise_snr
high_freq_ratio
step_ratio
memory_ratio
mean_delta_vs_MLP
worst_delta_vs_MLP
AUCtime_ratio
LineC_pass_rate
```

Fallback：

```text
FOU-FB1 lineage replay vs v14.9/v15.02.1
FOU-FB2 low-frequency-only ablation
FOU-FB3 phase drift decomposition
FOU-FB4 high-frequency quarantine audit
FOU-FB5 exhaustion certificate
```

## 9.3 D-RBF / FastKAN

候选：

```text
D-RBF50-CompactBumpIdentityResidualV4
D-RBF51-ActiveCenterOccupancyRepairV3
D-RBF52-WidthConditionGuardV3
D-RBF53-GaussianLocalK4NoDenseV2
D-RBF54-CenterSNRWarmupV2
```

必须记录：

```text
center_occupancy_entropy
empty_center_fraction
width_p01
width_p99
out_of_grid_fraction
active_center_snr
identity_residual_norm
step_ratio
memory_ratio
mean_delta_vs_MLP
AUCtime_ratio
LineC_pass_rate
```

Fallback：

```text
RBF-FB1 center occupancy collapse audit
RBF-FB2 width condition collapse audit
RBF-FB3 identity residual ablation
RBF-FB4 compact support replay
RBF-FB5 exhaustion certificate
```

## 9.4 D-WAV

候选：

```text
D-WAV45-TriangularSupportV5
D-WAV46-ScaleOccupancyHardeningV3
D-WAV47-SupportOverlapDampingV3
D-WAV48-LocalTailCoverageAuditV2
```

Fallback：

```text
WAV-FB1 scale occupancy audit
WAV-FB2 support overlap audit
WAV-FB3 low-scale-only replay
WAV-FB4 local-tail coverage decomposition
WAV-FB5 exhaustion certificate
```

## 9.5 Substrate gates

Exploration substrate：

```text
family_dataset_seed_pass_count >= 6/9
step_ratio <= 1.75
memory_ratio <= 1.75
mean_delta >= -0.05
worst_delta >= -0.10
LineC_pass_rate >= 0.30
```

Official FU eligibility：

```text
family_dataset_seed_pass_count = 9/9
step_ratio <= 1.25
memory_ratio <= 1.25
mean_delta >= -0.02
worst_delta >= -0.05
LineC_pass_rate >= 0.70
```

---

# 10. Line C：Geometry / tail / AUC audit

Line C 仍只做 audit / gate / failure explanation，不生成方向。

必须记录：

```text
source_vs_best_control
AUCtime_ratio
AUCstep_ratio
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
NLL_delta
ECE_delta
Brier_delta
margin_p10_delta
LineC_pass
LineC_fail_reason
```

必须可视化：

```text
fig_v153_transfer_operator_heatmap.svg
fig_v153_B1_B2_agreement_scatter.svg
fig_v153_STFU_vs_controls_source.svg
fig_v153_AUCtime_tail_failure_matrix.svg
fig_v153_allbasis_substrate_matrix.svg
fig_v153_runtime_breakdown.svg
```

---

# 11. Stop / continue 制度

## 11.1 Promotion fail-closed

Promotion 仍严格：

```text
S5 = 9/9 real 3x3
source / AUCtime / CEp99 / NLL / ECE / LineC / step / memory 全过
controls fail
provenance pass
code review pass
promotion_allowed = 1
```

## 11.2 Exploration continue-open

以下情况不能 hard stop：

```text
Line T fail
Line M controls positive
Line X transfer unsupported
Line D family <6/9
Line C tail/AUC fail
runtime overhead high
D-CHE ST-FU <3/9
MLP ST-FU positive
```

它们只进入 route / failure taxonomy / next hypothesis，不阻止其他线执行。

## 11.3 真正 hard stop

只有以下情况 hard stop：

```text
required_artifact_missing_count > 0
forbidden_information_violation_count > 0
no_action_search_violation_count > 0
direction uses validation/test/future/query
direction uses LineC/CEp99/NLL/ECE/AUCtime/Brier
dataset-name branch / seed-specific scale
action token / controller / action bank / reset route
fake/proxy/CPU offload
```

---

# 12. Exhaustion certificate

每条线结束必须写：

```text
line_id
main_surface_executed
fallback_ladder_executed
controls_executed
failure_taxonomy_complete
budget_kind
planned_budget
consumed_budget
deferred_items
deferred_reason
final_stop_allowed
```

不允许只写：

```text
budget exhausted
```

必须写明：

```text
which budget exhausted?
which fallbacks executed?
which items deferred?
why deferred?
does deferral affect route?
```

---

# 13. 最终 route 决策

## Success routes

```text
S2-STFUTransferExplorationPositive:
  D-CHE ST-FU real_lite >=3/9 且 source mean positive，controls 不完全解释。

S3-STFUMeaningfulRealLite:
  D-CHE ST-FU real_lite >=4/9，source >=0.005，control_equivalent <=0.50。

S4-STFURealTransferExploration:
  real_lite >=6/9，AUCtime/tail/LineC 大体同位。

S5-OfficialFunctionalSuccess:
  real 3x3 = 9/9，official gates 全过。
```

## No-go routes

```text
R1-STFUStillLocalPositiveNoTransfer:
  ST-FU 仍不能提升 B1->B2 transfer。

R2-STFUControlEquivalent:
  ST-FU positive 被 random / B1-only / MLP / same-norm controls 解释。

R3-STFUSubspaceNoSignal:
  J_B2 U 近 0 或 subspace condition 太差。

R4-STFUOverheadBlocked:
  有 value，但 step/memory overhead 无法接受。

R5-AllBasisSubstrateBlocked:
  D-FOU/RBF/WAV 仍 <6/9。

R6-CurrentFunctionalFamilyNoGo:
  ST-FU 与所有 substrate / controls 都失败。
```

---

# 14. 本轮最重要的判断标准

v15.3 不是为了再找一个好 update。它要回答：

$$
\boxed{
\text{把 transfer 放进 functional update 定义本身后，local positive 是否能转成 real transfer?}
}
$$

如果答案仍然是否定的，那么当前 DG-KAN functional update 主线需要更深层的理论重建，不能再靠 function-metric、proximal、CR-FU、FMS 或 optimizer wrapper 延续。

