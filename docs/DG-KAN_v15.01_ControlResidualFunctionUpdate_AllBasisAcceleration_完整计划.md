# DG-KAN v15.1：Control-Residual Function Update + All-Basis 加速并行完整计划

> 版本：v15.1 execution plan  
> 生成时间：2026-05-30  
> 核心修正：v15.0 已经证明“optimizer-aware wrapper 形式的 FU”没有打开 S2/S3/S4/S5；下一步不能继续 FU9/FU10，也不能回到 action / controller / reset。v15.1 改成 **Control-Residual Function Update, CR-FU**：先让现代 optimizer 负责一阶下降，再只在“optimizer 不能解释的剩余函数空间”里做 functional update。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能做 audit / gate，不能生成方向；不能启动 action bank / controller / reset route。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找一个普通 optimizer trick。项目目标是：

$$
\boxed{
\text{在 strict FC-PureKAN base/substrate 上，通过 functional update 获得比 ordinary backprop / AdamW 更好的训练轨迹、几何和泛化。}
}
$$

最终要证明的是：

$$
\boxed{
\text{PureKAN base}+\text{functional update}
>
\text{PureKAN base}+\text{ordinary AdamW/backprop controls}.
}
$$

同时必须满足：

```text
1. 表达力不打折；
2. forward / backward / step / memory 接近 MLP；
3. 收敛轨迹健康，AUCtime 不输；
4. tail / calibration 不坏；
5. signal / reservoir / noise geometry 不坏；
6. FU gain 必须超过 matched controls；
7. 不能把 MLP/generic optimizer positive 写成 KAN-specific success。
```

## 0.2 v15.0 当前真实状态

v15.0 已经按计划把 function update 放进 optimizer-aware 语境，测试 second moment、alignment、decoupled regularization、curvature / block preconditioner 和 basis-safe projection。它完成了 Line R/O/FU/M/K/D/C/Z，required artifacts、forbidden audit 和 no-action-search audit 均闭合。

但是 v15.0 最终 route 是：

```text
route = R8-CurrentFUDefinitionNoGo
minimum_success = S1-OptimizerMechanismImplemented
official_s5_reached = 0
promotion_allowed = 0
```

核心事实：

```text
1. D-CHE FU real-lite pass = 0 / 9。
2. source_vs_best_control_mean = -0.19696081876754762。
3. control_equivalent_fraction = 0.8222222222222222。
4. best FU method = FU4-D-CHE-MGUPFunctionUpdate。
5. best FU mean source_vs_best_control = 0.2216603292359246。
6. generic_optimizer_explains_gain = 1。
7. best MLP control = M2-MLP-CautiousAdamW。
8. Non-D-CHE best substrate = D-FOU 0 / 9。
9. D-FOU / D-RBF / D-WAV v15 substrate-only reconfirmation 全部 0 / 9。
```

这说明：

$$
\boxed{
\text{现代 optimizer 组件本身有信号，但 current FU 仍没有独立 causal value。}
}
$$

更具体地说：alignment、second moment、MGUP、SophiaDiag、SOAPLite、decoupled decay 都作为 wrapper 测过；它们没有把 D-CHE FU 变成 control-resistant functional success。v15.1 不能继续新增 FU9 / FU10，而要换 functional update 的因果结构。

---

# 1. 独立分析：v15.0 到底说明了什么

## 1.1 不是“没实现 optimizer-aware FU”

v15.0 已经实现并记录：

```text
decoupled_decay_implemented = 1
second_moment_implemented = 1
alignment_trace_available = 1
controls_available = 1
s1_optimizer_mechanism_implemented = 1
```

FU surface 覆盖：

```text
FU0-D-CHE-AdamW
FU1-D-CHE-RawFunctionUpdate
FU2-D-CHE-AdamV2FunctionUpdate
FU3-D-CHE-CautiousFunctionUpdate hard / soft
FU4-D-CHE-MGUPFunctionUpdate
FU5-D-CHE-SophiaDiagClippedFunctionUpdate
FU6-D-CHE-BlockSecondMomentFunctionUpdate
FU7-D-CHE-DecoupledDecayPlusFunctionUpdate
FU8-D-CHE-SOAPLiteDiagnosticFunctionUpdate
```

因此当前失败不能解释为：

```text
没有考虑二阶矩；
没有考虑 alignment；
没有考虑 decoupled decay；
没有考虑 MLP controls；
没有记录 LineC/tail audit；
没有补齐 hard/soft cautious。
```

这些都已覆盖。

## 1.2 当前 FU 的最大问题：仍然在和 optimizer 抢“一阶下降”这件事

现代 optimizer 已经很强。AdamW / CautiousAdamW / MGUP / second moment scaling 已经能处理很多一阶下降与噪声缩放问题。v15.0 中 MLP-CautiousAdamW 作为 generic control 有明显正信号，而 D-CHE FU 没有打开 3x3 pass。这说明：

$$
\boxed{
\text{如果 FU 只是另一个一阶下降方向或一阶下降 wrapper，它会被 generic optimizer controls 解释。}
}
$$

因此 v15.1 不再让 FU 与 AdamW / CautiousAdamW 竞争“一阶下降”。FU 只允许在 **control-residual subspace** 中发挥作用：

```text
ordinary optimizer:
  负责当前 train-stream loss 的主下降方向；

functional update:
  只负责 optimizer / matched controls 解释不了的剩余 function displacement。
```

## 1.3 Decoupled weight decay 的启发

AdamW 的核心启发不是“加 weight decay”，而是：

$$
\boxed{
\text{regularization / decay 不能混进 loss-gradient update。}
}
$$

对 DG-KAN 来说，degree damping、frequency damping、RBF width shrink、wavelet scale shrink、rational denominator damping 都不能被写成 FU value。它们必须作为 decoupled constraint / shrink 独立执行，并配套 decay-only、random-matched decay、FU+same-decay controls。

v15.0 已经显示 decoupled decay 不足以解释全部，但也能解释一部分：`decoupled_decay_explains_fraction = 0.2222`。因此 v15.1 继续保持：

```text
FU value path 与 decoupled regularization path 分离；
decay / damping 只能做 constraint，不做 value source。
```

## 1.4 Cautious / MGUP 的启发

Cautious Optimizers 的直接启发是：proposed update 与当前 gradient 冲突时，不应盲目更新。v15.0 的 FU4 / MGUP 有 best mean source 正值，但仍 0/9 pass。这说明 alignment 有用，但单独不够。v15.1 将 alignment 从“一个 FU variant”升级为 **所有 FU 的硬审计层**：

$$
a_i = u_i(-g_i).
$$

若 $a_i < 0$，该坐标的 FU 与当前 train descent 反向，必须记录：

```text
anti_alignment_fraction
rolewise_anti_alignment
basis_group_anti_alignment
post_projection_anti_alignment
```

但 v15.1 不再只做 `HardCautiousFU` / `SoftCautiousFU` 小变体；alignment 只作为 control-residual solver 的约束与诊断。

## 1.5 第二矩 / 曲率的启发

v15.0 的 AdamV2、SophiaDiag、BlockSecondMoment、SOAPLite 没有打开 success，说明“加二阶矩 wrapper”不够。但这不等于二阶矩无用，而是说明二阶矩必须进入 **control residual inner product**：

$$
\langle u,v\rangle_{M_2}=u^\top M_2 v.
$$

其中 $M_2$ 可以是：

```text
diag(sqrt(v_t)+eps)
role-wise Adam v
degree-wise second moment
basis-group second moment
block low-rank second moment
```

FU 不应只被二阶矩缩放；它必须先在这个 metric 中投掉 ordinary optimizer 已经解释的方向。

---

# 2. v15.1 核心假设

v15.1 的主假设是：

$$
\boxed{
\text{FU 失败不是因为没有 optimizer wrapper，}
\text{而是因为 FU 没有与 generic optimizer controls 做 residualization。}
}
$$

也就是说，当前 FU 大量 effect 被以下 controls 解释：

```text
AdamW / AdamWParallel
CautiousAdamW
MGUP-like alignment
second moment scaling
decoupled decay
matched random norm
same active fraction
same projection rejection
same value retention
```

所以 v15.1 只问一个更尖锐的问题：

$$
\boxed{
\text{在 optimizer 和 matched controls 已经解释的部分被投掉后，FU 还有没有剩余 causal value？}
}
$$

若答案是否定的，就应该正式关闭当前 FU family，不能再以 FU9/FU10 拖延。

---

# 3. v15.1 新机制：Control-Residual Function Update, CR-FU

## 3.1 基础对象

令当前 train-stream 梯度为：

$$
g_t=\nabla_\theta \mathcal L_{train}(\theta_t).
$$

令 generic optimizer update 为：

$$
a_t = \operatorname{OptimizerStep}(g_t, m_t, v_t).
$$

它可以是 AdamW、CautiousAdamW 或 MGUP-style optimizer control。

令 raw FU proposal 为：

$$
u_t = \Delta\theta_{FU}.
$$

v15.0 失败说明，直接用 $u_t$ 或简单 preconditioned $u_t$ 不够。

## 3.2 在二阶矩 metric 下投掉 optimizer 可解释分量

定义 metric：

$$
M_t = \operatorname{diag}(\sqrt{v_t}+\epsilon).
$$

内积：

$$
\langle x,y\rangle_{M_t}=x^\top M_t y.
$$

将 FU 投影到 optimizer residual subspace：

$$
u_{\perp}
=
u_t
-
\frac{\langle u_t,a_t\rangle_{M_t}}
{\langle a_t,a_t\rangle_{M_t}+\epsilon}
a_t.
$$

若有多个 control directions $c_j$，用矩阵 $C=[a_t,c_1,c_2,\ldots,c_k]$：

$$
u_{\perp}
=
u_t
-
C(C^\top M_t C+\lambda I)^{-1}C^\top M_t u_t.
$$

这一步的意义是：

```text
如果 FU 只是普通 optimizer direction 的别名，会被投掉；
只有 FU 中不能被 optimizer / matched controls 解释的部分留下。
```

## 3.3 CR-FU 更新

最终更新不是替代 AdamW，而是：

$$
\Delta\theta_t
=
a_t
+
\eta_{fu}
P_{safe}
P_{align}
P_{trust}
u_{\perp}.
$$

其中：

```text
P_align:
  保证 u_perp 与 -g_t 至少不大规模反向。

P_safe:
  basis-safe projection，只做安全约束，不定义 value。

P_trust:
  train-stream trust region，不使用 audit metric。
```

若 $u_{\perp}$ norm 很小或 control-residual value 不足，CR-FU 自动退化为 ordinary optimizer：

$$
\Delta\theta_t=a_t.
$$

这不是 NoOp success，也不是 action selection；这是固定公式下的 abstention。

## 3.4 CR-FU 与 action search 的区别

v15.1 禁止 action bank。CR-FU 不是在一堆动作中选择，也不是新增 FU9/FU10 token。它是一个固定求解器：

```text
输入：
  当前 train batch；
  current optimizer state；
  FU proposal；
  matched control directions；
  basis telemetry；

输出：
  single residualized FU update 或 FU abstention。
```

不会根据 dataset / seed / audit fail pattern 选择不同方法。

---

# 4. v15.1 并行实验总览

v15.1 不再串行等待一个 line 失败后再执行下一条。必须并行执行：

```text
Line R:
  provenance / no-action-search / forbidden audit。

Line O:
  optimizer-state and direction decomposition readback。

Line C0:
  v15.0 FU failure autopsy and residualizability audit。

Line F:
  CR-FU on D-CHE。

Line M:
  MLP and generic optimizer controls。

Line K:
  KAN-specificity / control-residual causal audit。

Line D:
  all-basis substrate acceleration and reconciliation。

Line C:
  LineC / tail / calibration audit。

Line Z:
  route / no-go / next hypothesis.
```

---

# 5. Line R：审计与硬约束

## 5.1 必须检查

```text
required_artifact_missing_count
forbidden_information_violation_count
no_action_search_violation_count
direction_uses_validation_test_future_query
direction_uses_LineC_CEp99_NLL_ECE_AUCtime
dataset_name_branch
seed_specific_scale
action_token_extension
controller_executed
reset_route_used
fake_data_used
proxy_row_used
cpu_offload_used
```

## 5.2 Hard stop 条件

只有以下情况允许 hard stop：

```text
1. required_artifact_missing_count > 0；
2. forbidden_information_violation_count > 0；
3. no_action_search_violation_count > 0；
4. direction 使用 validation/test/future/query；
5. direction 使用 LineC/CEp99/NLL/ECE/AUCtime/Brier；
6. dataset-name branch；
7. seed-specific scale；
8. action token / controller / action bank / reset route；
9. fake / proxy / CPU offload。
```

任何实验 gate fail 都不能 hard stop whole run。

---

# 6. Line O：optimizer-state and direction decomposition

## 6.1 目标

判断 FU 与 optimizer controls 的关系：

```text
FU 与 AdamW 是否几乎同向？
FU 与 CautiousAdamW 是否几乎同向？
FU 的正信号是否来自 decoupled decay？
FU 的 anti-alignment 是否集中在某些 basis role？
FU 经 second moment metric 后是否被 controls 解释？
```

## 6.2 记录指标

```text
fu_norm
grad_norm
adamw_update_norm
cautious_update_norm
mgup_update_norm
decay_update_norm

cos_fu_neg_grad
cos_fu_adamw
cos_fu_cautious
cos_fu_mgup
cos_fu_decay

metric_cos_fu_adamw
metric_cos_fu_cautious
metric_cos_fu_mgup

control_projection_norm_fraction
control_residual_norm_fraction
anti_alignment_fraction
rolewise_anti_alignment
basis_group_anti_alignment

decoupled_decay_explains_fraction
random_matched_decay_explains_fraction
second_moment_condition
curvature_clip_fraction
```

## 6.3 可视化

```text
fig_v151_direction_cosine_matrix.svg
fig_v151_control_projection_fraction.svg
fig_v151_alignment_by_role.svg
fig_v151_decay_vs_fu_norm.svg
fig_v151_second_moment_condition.svg
```

---

# 7. Line C0：v15.0 FU failure autopsy

## 7.1 目标

在不新增方法的前提下，对 v15.0 失败做归因：

```text
Failure A:
  FU direction 被 optimizer controls 解释。

Failure B:
  FU residual norm 过小。

Failure C:
  FU residual norm 不小，但 train-stream effect control-equivalent。

Failure D:
  FU residual train-stream positive，但 real-lite fail。

Failure E:
  basis-safe projection 杀掉 value。

Failure F:
  decoupled decay / regularization 解释 gain。
```

## 7.2 必须输出

```text
v151_fu_failure_attribution.csv
v151_fu_residualizability_summary.csv
v151_projection_value_loss.csv
v151_decay_deconfound_extended.csv
v151_current_fu_no_go_boundary.md
```

---

# 8. Line F：D-CHE CR-FU

## 8.1 实验方法

只允许预注册方法：

```text
F0-D-CHE-AdamW
F1-D-CHE-CautiousAdamW
F2-D-CHE-MGUPControl
F3-D-CHE-RawFUResidualizedAgainstAdamW
F4-D-CHE-RawFUResidualizedAgainstCautious
F5-D-CHE-MGUPFUResidualizedAgainstMGUPControl
F6-D-CHE-FunctionSpaceProximalResidual
F7-D-CHE-CRFU-AbstentionEnabled
FCTRL-RandomResidualMatchedNorm
FCTRL-SameActiveFractionResidual
FCTRL-SameProjectionRejectionResidual
FCTRL-SameValueRetentionResidual
```

注意：

```text
F3-F7 不是 action bank；
它们是不同的 residualization / solver definitions。
不允许新增 F8/F9/F10。
```

## 8.2 Function-space proximal residual

F6 只在 residual subspace 中解：

$$
\alpha^\star
=
\arg\min_\alpha
\mathcal L_{B_1}
\left(
\theta_t+a_t+U_\perp\alpha
\right)
+
\lambda \|U_\perp\alpha\|_{M_t}^2.
$$

其中：

```text
B1 是 train split，不是 validation/test。
U_perp 是已对 optimizer controls residualized 的 function-update basis。
```

然后：

$$
\Delta\theta_t=a_t+U_\perp\alpha^\star.
$$

这不是 future oracle；它只使用当前 train-stream split。

## 8.3 通过标准

### Exploration gate

```text
real_lite_pass_count >= 3/9
source_vs_best_control_mean > 0
control_equivalent_fraction <= 0.60
bad_event_fraction <= 0.50
step_time_ratio <= 1.50 exploratory
```

### Meaningful gate

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean >= 0.005
control_equivalent_fraction <= 0.50
bad_event_fraction <= 0.35
step_time_ratio <= 1.35
```

### S4 exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime median <= 1.05
tail fail reduced vs F0
LineC fail <= F0
```

### S5 official

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

---

# 9. Line M：MLP / generic controls

## 9.1 目标

防止把 generic optimizer gain 写成 KAN-specific functional gain。

## 9.2 必跑 controls

```text
M0-MLP-AdamW
M1-MLP-CautiousAdamW
M2-MLP-MGUPControl
M3-MLP-FunctionSpaceProximalResidual
M4-MLP-CRFU-AbstentionEnabled
M5-MLP-RandomResidualMatchedNorm
M6-MLP-SameActiveFractionResidual
M7-MLP-NoOpMatchedOverhead
```

## 9.3 判断

若 MLP 与 D-CHE 同时 positive：

```text
generic functional optimizer positive；
不能 claim KAN-specific。
```

若 MLP controls 解释 D-CHE CR-FU：

```text
route = R3-GenericOptimizerExplainsCRFU
```

若 D-CHE positive 且 MLP controls 不解释：

```text
KAN-specific candidate 才可进入 S4/S5。
```

---

# 10. Line K：KAN-specificity / control-residual causal audit

## 10.1 Difference-in-differences

核心估计：

$$
\Delta_{KAN}
=
(D\text{-}CHE+CRFU - D\text{-}CHE+Control)
-
(MLP+CRFU - MLP+Control).
$$

通过探索要求：

$$
\Delta_{KAN} > 0.
$$

并且：

```text
D-CHE CRFU real_lite_pass_count > MLP CRFU real_lite_pass_count
```

## 10.2 记录指标

```text
dche_crfu_source
dche_control_source
mlp_crfu_source
mlp_control_source
delta_kan_specific
delta_generic
kan_specific_pass
```

---

# 11. Line D：all-basis substrate acceleration

## 11.1 当前状态

v15.0 的 all-basis reconfirmation 显示：

```text
D-FOU = 0/9
D-RBF = 0/9
D-WAV = 0/9
```

这意味着 Non-D-CHE 当前不能进入 official FU proof。但不能停止 all-basis，因为 D-FOU / D-RBF 历史上曾有过 6/9 exploration signal，说明存在 reproducibility / candidate / budget / gate mismatch。

## 11.2 v15.1 并行候选

### D-FOU

```text
D-FOU42-LowFreqResidualV5
D-FOU43-BandwiseSecondMomentWarmup
D-FOU44-PhaseStableLowBandOnly
D-FOU45-NoMaterializeLifetimeV4
D-FOU46-HighFreqQuarantineLateEnable
```

### D-RBF / FastKAN

```text
D-RBF40-CompactBumpIdentityResidualV2
D-RBF41-ActiveCenterOccupancySecondMoment
D-RBF42-WidthFloorTrustRegion
D-RBF43-GaussianLocalK4NoDenseV2
D-RBF44-CenterReadoutDecoupledWarmup
```

### D-WAV

```text
D-WAV37-TriangularSupportV5
D-WAV38-ScaleOccupancySecondMoment
D-WAV39-LocalSupportOverlapTrust
D-WAV40-FineScaleLateEnable
```

### D-CHE

```text
D-CHE no-regression monitor
D-CHE CR-FU carrier
```

## 11.3 Gate

Exploration substrate:

```text
family_dataset_seed_pass_count >= 6/9
step_ratio <= 1.75
memory_ratio <= 1.75
mean_delta >= -0.05
worst_delta >= -0.10
LineC_pass_rate >= 0.30
```

Official FMS eligibility:

```text
family_dataset_seed_pass_count = 9/9
step_ratio <= 1.25
memory_ratio <= 1.25
LineC_pass_rate >= 0.50
```

If not reached, no official FU proof.

---

# 12. Line C：Geometry / tail / calibration audit

## 12.1 指标

```text
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
NLL_delta
ECE_delta
Brier_delta
margin_p10_delta
signal_channel_energy
reservoir_energy
noise_leakage_proxy
```

## 12.2 规则

```text
1. These metrics are audit/gate only.
2. They cannot generate FU direction.
3. They cannot determine dataset/seed branch.
4. They cannot be optimized directly.
```

## 12.3 可视化

```text
fig_v151_linec_tail_matrix.svg
fig_v151_signal_reservoir_noise_plane.svg
fig_v151_tail_calibration_tradeoff.svg
```

---

# 13. 并行加速执行

## 13.1 GPU 分组

```text
GPU group 0:
  Line F D-CHE CR-FU + matched controls

GPU group 1:
  Line M MLP / generic controls

GPU group 2:
  Line D D-FOU substrate acceleration

GPU group 3:
  Line D D-RBF / D-WAV substrate acceleration

GPU group 4 if available:
  Line O/C0 decomposition + Line C audit + Rational no-regression
```

## 13.2 最低完成要求

```text
Line R:
  forbidden/no-action/provenance audit complete

Line O:
  direction decomposition complete

Line C0:
  v15.0 failure attribution complete

Line F:
  F0-F7 + controls complete

Line M:
  M0-M7 complete

Line K:
  difference-in-differences complete

Line D:
  D-FOU and D-RBF full rows complete
  D-WAV low-budget monitor complete
  D-CHE no-regression complete

Line Z:
  route/no-go/failure taxonomy/next hypothesis complete
```

No single line failure may stop the others.

---

# 14. Stop / continue contract

## 14.1 Promotion fail-closed

S5 only if:

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

## 14.2 Exploration continue-open

These do not hard stop:

```text
Line F CR-FU fail
Line M generic controls positive
Line K KAN-specificity fail
Line D all-basis fail
D-CHE real-lite <4/9
D-FOU / D-RBF / D-WAV <6/9
LineC/tail/AUC fail
step_time > exploratory gate
```

They must produce route, failure taxonomy and next hypothesis, but not stop the entire run.

## 14.3 Hard stop only for violations

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

---

# 15. Expected routes

```text
S2-CRFUExplorationPositive:
  real_lite_pass_count >= 3/9 and source mean > 0.

S3-CRFUMeaningfulPositive:
  real_lite_pass_count >= 4/9 and source_vs_best_control >= 0.005.

S4-CRFURealLitePositive:
  real_lite_pass_count >= 6/9.

S5-OfficialFunctionalSuccess:
  real_dataset_seed_pass_count = 9/9 and all official gates pass.

R1-CRFUControlEquivalent:
  CR-FU <= matched controls.

R2-CRFUResidualNormTooSmall:
  residualized FU norm too small after control projection.

R3-GenericOptimizerExplainsCRFU:
  MLP/generic controls explain gain.

R4-DecoupledDecayExplainsGain:
  decay-only or random-matched decay explains gain.

R5-BasisProjectionKillsResidualValue:
  residual FU value lost after basis-safe projection.

R6-AllBasisSubstrateBlocked:
  non-D-CHE families <6/9.

R7-CurrentFUFamilyNoGo:
  FU, CR-FU, and substrate alternatives all fail.
```

---

# 16. Final decision rule

v15.1 is decisive. If:

```text
Line F CR-FU fail
Line M/generic controls explain positive rows
Line K KAN-specificity fail
Line D no non-D-CHE substrate >=6/9
```

then write:

```text
route = R7-CurrentFUFamilyNoGo
```

and do not continue FU9/FU10 / F-CHE8/F-CHE9 / action / controller / reset.

If CR-FU opens S2/S3/S4, continue only through pre-registered confirm path, not token search.

---

# 17. One-sentence summary

v15.1 does not search for another “good update”. It asks whether any functional update remains after subtracting what AdamW, Cautious/MGUP, second moments, and decoupled decay already explain. If nothing remains, the current FU family is no-go.
