# DG-KAN v15.0：Function Update + All-Basis 并行加速完整计划

> 版本：v15.0 execution plan  
> 目标：在不重蹈 v9 / v12 / v14 action-search 错误的前提下，继续以 **functional update** 为突破口，同时把最新优化器研究中的二阶矩、方向一致性、解耦权重衰减、矩阵预条件、schedule-free 训练思想系统纳入实验。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline budget；no teacher / distillation / loss modification / sampler / class weight / dataset-name branch；no label-informed initialization；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime 只能作为 audit / gate，不能生成方向；不能新增 action bank / controller / reset route；不能按 dataset / seed 调参；不能把 diagnostic / real-lite / local positive / control-equivalent 写成 promotion。  

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找到一个只在局部 metric 上好看的 functional event。项目总目标是：

$$
\boxed{
\text{PureKAN substrate} + \text{Function Update}
>
\text{同一 substrate} + \text{普通 AdamW / backprop controls}
}
$$

并且这种优势必须同时满足：

```text
1. 真实参数更新，不是 readout-feature proxy；
2. 不使用 validation/test/future/query 作为 direction；
3. 不用 CEp99/NLL/ECE/LineC/AUCtime 作为 direction；
4. 打过 RandomMatched / SameActiveFraction / AdamWParallelDirection / NoOpMatchedOverhead / generic optimizer controls；
5. source、AUCtime、tail、LineC、runtime 同时不坏；
6. 在多个 basis substrate 上可复核，而不是只绑定一个偶然 carrier；
7. forward/backward/step/memory 与 MLP-like envelope 相容。
```

最终要证明：

$$
\boxed{
\text{Function Update 不是“好动作搜索”，而是一个可解释、可审计、可迁移的训练机制。}
}
$$

## 0.2 当前实验事实

截至 v14.16，当前事实非常清楚：

```text
1. D-CHE substrate 是当前最强 Non-RAT carrier，历史上已有 9/9 substrate eligibility。
2. D-CHE-FMS synthetic 历史 best 为 5/7，但 real transfer 不可靠。
3. current FMS direction 在 decisive fork 中 no-go：
   Line A mean_source_vs_best_control < 0，
   control_equivalent_fraction 高，
   real_lite_pass_count = 0/9。
4. Metric-state FMS MS1/MS2/MS3 也没有打开 exploration gate。
5. Transfer predictor 不稳健，不能用于 gating / abstention / boundary。
6. D-FOU / D-RBF / D-WAV v14.16 实跑 reconfirmation 仍未达到 substrate exploration gate。
7. Rational 仍是稳定 substrate，但 reset / optimizer-state route 已被 generic optimizer-state confound 打回。
8. Official S5 functional success 仍为 0。
```

因此 v15.0 不能继续：

```text
FMS-M6 / FMS-M7；
F-CHE8 / F-CHE9；
action token；
controller；
action bank；
reset route；
strength / lambda / interval / topK grid；
dataset / seed branch；
audit-direction。
```

## 0.3 v15.0 的战略重置

用户指出得对：我们之前没有充分吸收优化器研究进展，尤其没有认真处理：

```text
1. Adam / AdamW 的二阶矩；
2. update direction 与当前 gradient 的方向一致性；
3. decoupled weight decay 与 L2 regularization 的区别；
4. matrix / block / input-second-moment preconditioning；
5. schedule-free / momentum / older-gradient mixing；
6. functional update 与普通 optimizer state 的耦合关系。
```

v15.0 因此不再叫 Function-Space Proximal Update 单线计划，而是：

$$
\boxed{
\text{Function Update}
=
\text{loss-interface generic value}
+
\text{optimizer-aware second-moment / alignment gate}
+
\text{basis-safe projection}
+
\text{decoupled regularization}
}
$$

这不是把 functional update 降级为 optimizer trick，而是承认：一个可用的 function update 必须和 optimizer 的一阶矩、二阶矩、权重衰减、方向一致性、矩阵预条件一起设计。

---

# 1. 外部优化器研究给 v15.0 的启发

## 1.1 AdamW / Decoupled Weight Decay 的启发

Decoupled Weight Decay 的核心教训是：在 adaptive optimizer 中，L2 regularization 和 weight decay 不等价。若把 decay 项混进 gradient，再被 Adam 的二阶矩预条件，会改变 regularization 的实际含义。

对 DG-KAN 的直接启发：

```text
1. basis smoothness / degree damping / denominator damping / frequency damping 不能混进 function-update direction；
2. regularization 必须 decoupled，作为独立 shrink / constraint step；
3. 如果一个 improvement 来自 decoupled decay，而不是 function update direction，必须写成 regularization optimizer gain，不能写成 functional gain；
4. controls 必须包含：
   coupled L2-style functional regularization；
   decoupled role-wise weight decay；
   random-matched decoupled decay；
   no-decay baseline。
```

v15.0 定义：

$$
\theta_{t+1/2}=\theta_t+\Delta\theta_{FU},
$$

$$
\theta_{t+1}=(1-\eta\lambda_r)\theta_{t+1/2,r}
$$

其中 $\lambda_r$ 是 role-level decoupled decay，只能是预注册 role schedule，不允许 dataset/seed branch。

## 1.2 Adam 二阶矩 / Sophia / SOAP 的启发

过去 FMS 的一个根本问题是：我们常常只看 direction，却没有认真处理第二矩或曲率。AdamW 的关键不只是 momentum $m_t$，还有二阶矩 $v_t$：

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2.
$$

若 functional update 产生的方向没有经过 $v_t$ 或 curvature-aware scale，它可能在 high-variance / high-curvature 坐标上过度移动，表现为 AUCtime debt、tail fail 或 control-equivalent。

Sophia 的启发是使用轻量 diagonal Hessian / curvature estimate 和 clipping，避免每步昂贵二阶优化。SOAP / Shampoo 系列的启发是：在合适的旋转坐标或预条件坐标中更新 second moment，而不是只在原始坐标中做 diagonal Adam。

v15.0 因此要求每个 FU candidate 同时记录：

```text
raw_direction_norm
adam_preconditioned_norm
v2_preconditioned_norm
hessian_diag_or_gn_diag_norm
clipped_fraction
rolewise_curvature_p50/p90/p99
cos_raw_vs_adam_preconditioned
cos_raw_vs_gradient
cos_preconditioned_vs_gradient
```

并且比较：

```text
FU-raw
FU-AdamV2
FU-SophiaDiag
FU-BlockSecondMoment
FU-SOAP-lite, diagnostic
```

## 1.3 Cautious Optimizer / MGUP 的启发

用户提到的“functional update 的方向和梯度反传方向相同时更新，相反的话不更新”，正好对应 Cautious Optimizer / MGUP 的核心思想。

定义当前 train-stream supervised gradient：

$$
g_t=\nabla_\theta L_{train}(\theta_t).
$$

定义 functional update 作为参数更新向量：

$$
u_{FU}.
$$

因为 $-g_t$ 是普通下降方向，所以 coordinate-wise descent alignment 为：

$$
a_i=u_{FU,i}(-g_{t,i}).
$$

若：

$$
a_i>0,
$$

说明 functional update 与当前 train-stream descent direction 同向；若：

$$
a_i<0,
$$

说明它在该坐标上逆着当前 train loss。

v15.0 不直接假设“相反就一定不更新”，而是并行验证三种 gate：

```text
Cautious-hard:
  a_i <= 0 的坐标置零。

Cautious-soft:
  a_i <= 0 的坐标降低权重，不完全置零。

MGUP-style:
  按 a_i 排序或分位数重加权，同向强、反向弱。
```

注意：alignment 必须在 **preconditioned space** 里同时记录：

$$
a_i^{raw}=u_i(-g_i),
$$

$$
a_i^{adam}=\frac{u_i}{\sqrt{v_i}+\epsilon}\frac{-g_i}{\sqrt{v_i}+\epsilon}.
$$

如果 raw alignment positive 但 preconditioned alignment negative，则不允许 promotion，必须写成 alignment ambiguity。

## 1.4 Muon / Newton-Muon / matrix preconditioning 的启发

Muon 系列的启发不是“照搬 matrix sign”，而是：对矩阵参数，优化方向可能需要满足结构化的 matrix geometry；Newton-Muon 更进一步指出，input second moment $ZZ^\top$ 的右预条件很重要。

对 KAN 的启发：

```text
1. readout / mixing / degree projection / basis-role matrices 不能只做 coordinate-wise update；
2. D-CHE 的 degree/readout 部分应测试 role-matrix preconditioning；
3. D-FOU 的 band mixing、D-RBF 的 center/readout、Rational 的 group/readout 都应记录 input-second-moment；
4. 不能让 matrix preconditioner 变成昂贵 route；只做 low-rank / block / infrequent update。
```

v15.0 引入：

$$
G_{in}=ZZ^\top+\rho I,
$$

$$
\Delta W_{right-precond}=\Delta W G_{in}^{-1}.
$$

但它只允许作为预注册 diagnostic / low-rank path，不能在 positive row 后无限扩 rank。

## 1.5 Schedule-Free / AdEMAMix / Adan / Lion 的启发

这些工作共同提醒我们：训练过程不是单点 update，而是 momentum、长期平均、短期梯度、旧梯度信息共同作用的动态系统。

v15.0 只借鉴三件事：

```text
1. schedule-free 作为 control，不再手调 LR schedule；
2. old-gradient / long EMA 作为 diagnostic，检查 functional value 是否来自更长时间尺度；
3. sign / momentum optimizers 只做 MLP/generic controls，不能写成 KAN-specific。
```

---

# 2. v15.0 核心假设

## H1：当前 FMS 失败，是因为缺少 optimizer-aware second moment / alignment gate

过去 current FMS direction no-go，可能不是因为 functional idea 完全错，而是因为：

```text
1. direction 未经过 Adam second moment；
2. direction 与 current gradient 反向坐标太多；
3. direction 与 optimizer momentum / v2 state 不一致；
4. role/basis telemetry 被用成 value source，而不是 safety projection。
```

判断标准：

```text
若 FU + V2 + Cautious/MGUP gate 显著超过 controls，
则 current FMS no-go 主要是 optimizer-awareness 缺失。

若仍然 control-equivalent，
则 direction family 仍 no-go。
```

## H2：decoupled regularization 能防止 basis safety 伪装成 direction

过去很多 basis constraint / boundary / degree damping 可能混进 direction，导致 control-equivalent。v15.0 把它们全部 decouple：

```text
value direction:
  只来自 train-stream loss-interface gradient / FU solver。

regularization / safety:
  只作为 decoupled shrink / trust-region / projection。
```

如果 improvement 来自 decoupled regularization本身，必须写成 regularization optimizer result，不是 functional update result。

## H3：二阶矩必须区分 parameter-level、role-level、basis-level

仅 parameter-level $v_t$ 不够。KAN 的 basis 结构有 role：

```text
D-CHE:
  low-degree / high-degree / readout / residual。

D-FOU:
  low-frequency / high-frequency / phase / amplitude / readout。

D-RBF:
  centers / widths / readout / residual.

Rational:
  numerator / denominator / readout / group residual.

Wavelet:
  scale / support / readout.
```

v15.0 记录三层 second moment：

$$
v_i^{param},
$$

$$
v_r^{role}=\operatorname{mean}_{i\in r} v_i,
$$

$$
v_b^{basis}=\operatorname{mean}_{i\in b} v_i.
$$

并测试：

```text
param-v2 preconditioner
role-v2 preconditioner
basis-v2 preconditioner
block-v2 preconditioner
```

## H4：若 FU 方向与 backprop 方向相反，可能应 abstain 或 downweight

这不是手工规则，而是 optimizer literature 支持的 hypothesis。v15.0 将其作为可证伪实验：

```text
Hard Cautious:
  opposite coordinates zero.

Soft Cautious:
  opposite coordinates downweight.

MGUP:
  alignment high coordinates upweight, low coordinates downweight.

No gate:
  baseline FU.
```

如果 Cautious/MGUP 改善 source/AUC/tail/LineC，说明 current FU 的主要问题是 direction conflict。  
如果不改善，说明问题不是 simple alignment。

## H5：All-basis 不应停；D-CHE 是 carrier，但不是唯一 carrier

v15.0 继续 all-basis parallel：

```text
D-CHE:
  主 FU carrier。

D-FOU:
  low-frequency / phase-stable / no-materialize substrate repair。

D-RBF / FastKAN:
  active center / width condition / compact bump substrate repair。

D-WAV:
  low-budget monitor。

Rational:
  no-regression monitor；不继续 reset route。
```

---

# 3. 总体实验设计

v15.0 分成 8 条线，必须并行执行：

```text
Line R:
  implementation / provenance / no-action-search audit。

Line O:
  optimizer literature-inspired mechanism implementation audit。

Line FU:
  optimizer-aware function update on D-CHE。

Line M:
  MLP / generic optimizer controls。

Line K:
  KAN-specificity and decoupled regularization audit。

Line D:
  all-basis substrate acceleration。

Line C:
  geometry / tail / LineC audit。

Line Z:
  final route / no-go / next hypothesis。
```

预算分配：

```text
Line FU D-CHE optimizer-aware update:
  35%

Line M controls:
  15%

Line D all-basis substrate:
  30%

Line C/R/Z audit:
  20%
```

并行 GPU 分配：

```text
GPU group 0:
  D-CHE FU main matrix.

GPU group 1:
  MLP / generic controls.

GPU group 2:
  D-FOU + D-RBF substrate acceleration.

GPU group 3:
  D-CHE second-moment / alignment ablation.

GPU group 4 if available:
  Rational monitor + D-WAV low-budget.
```

---

# 4. Line R：Implementation / Provenance / No-Action-Search Audit

## 4.1 目标

确保 v15.0 不重蹈 v9/v12 错误：

```text
1. 不找 action；
2. 不启 controller；
3. 不用 audit metric 生成方向；
4. 不把 regularization / optimizer trick 写成 functional gain；
5. 不把 MLP/generic success 写成 KAN-specific。
```

## 4.2 必须记录

```text
required_artifact_missing_count
forbidden_information_violation_count
no_action_search_violation_count
direction_uses_validation_test_future_query
direction_uses_linec_cep99_nll_ece_auctime
dataset_name_branch_used
seed_specific_scale_used
new_action_token_count
controller_executed
reset_route_used
fche_token_extension_count
fms_method_extension_count
optimizer_grid_search_count
decoupled_decay_implemented
second_moment_implemented
alignment_gate_implemented
```

## 4.3 Hard stop

只有以下情况 hard stop：

```text
required_artifact_missing_count > 0
forbidden_information_violation_count > 0
no_action_search_violation_count > 0
direction_uses_validation_test_future_query = 1
direction_uses_linec_cep99_nll_ece_auctime = 1
dataset_name_branch_used = 1
seed_specific_scale_used = 1
new_action_token_count > 0
controller_executed = 1
reset_route_used = 1
```

其它 fail 不 hard stop，必须继续其它独立线。

---

# 5. Line O：Optimizer Mechanism Implementation Audit

## 5.1 目标

先确认优化器启发被正确实现，不让 Codex 把它们写成名字而没有机制。

## 5.2 需要审计的机制

```text
O1-DecoupledWeightDecay:
  regularization 不进入 gradient / direction；
  update 后单独 shrink。

O2-SecondMomentPrecondition:
  param / role / basis v2 均有 trace。

O3-CautiousAlignmentGate:
  raw 与 Adam-preconditioned alignment 均记录。

O4-MGUPReweight:
  alignment 分位数重加权，不是 hard-coded dataset branch。

O5-SophiaDiagClip:
  diagonal curvature / Gauss-Newton proxy + clip。

O6-BlockSecondMoment:
  D-CHE degree/readout block second moment。

O7-ScheduleFreeControl:
  schedule-free 或 averaged iterate 只作为 control。

O8-WeightDecayControls:
  coupled L2 vs decoupled decay vs cautious decay vs no decay。
```

## 5.3 输出 artifact

```text
v150_optimizer_mechanism_manifest.csv
v150_decoupled_decay_audit.csv
v150_second_moment_trace.csv
v150_alignment_trace.csv
v150_curvature_diag_trace.csv
v150_block_preconditioner_trace.csv
v150_schedulefree_control_trace.csv
```

---

# 6. Line FU：D-CHE Optimizer-Aware Function Update

## 6.1 目标

在 D-CHE 9/9 substrate 上，测试 optimizer-aware FU 是否比 AdamW / random / generic controls 有独立 causal value。

## 6.2 候选 family

v15.0 只允许这些预注册 FU candidates，不允许 FU6/FU7：

```text
FU0-D-CHE-AdamW
FU1-D-CHE-RawFunctionUpdate
FU2-D-CHE-AdamV2FunctionUpdate
FU3-D-CHE-CautiousFunctionUpdate
FU4-D-CHE-MGUPFunctionUpdate
FU5-D-CHE-SophiaDiagClippedFunctionUpdate
FU6-D-CHE-BlockSecondMomentFunctionUpdate
FU7-D-CHE-DecoupledDecayPlusFunctionUpdate
FU8-D-CHE-SOAPLiteDiagnosticFunctionUpdate
```

注意：FU8 是 diagnostic，若成本过高不能 promotion。

## 6.3 FU direction 定义

基础 FU direction 来自 train-stream loss-interface gradient / population risk signal：

$$
g_t=\nabla_\theta L_{train}(\theta_t),
$$

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2.
$$

基础 descent proposal：

$$
u_{base}=-\frac{m_t}{\sqrt{v_t}+\epsilon}.
$$

Functional augmentation 不是另找 action，而是在 D-CHE degree/basis subspace 中形成：

$$
u_{FU}=P_{D-CHE}(u_{base}),
$$

其中 $P_{D-CHE}$ 是预注册 basis-safe projection，只能使用 D-CHE degree telemetry 和 train-stream parameter roles，不能使用 audit metric。

## 6.4 Second-moment variants

### FU2 AdamV2FunctionUpdate

$$
u=-\frac{m_t}{\sqrt{v_t^{param}}+\epsilon}.
$$

### FU5 SophiaDiagClippedFunctionUpdate

定义 diagonal curvature proxy：

$$
h_t=\operatorname{EMA}(\widehat{\operatorname{diag}}(J^\top H_y J)).
$$

更新：

$$
u=-\operatorname{clip}\left(\frac{m_t}{h_t+\epsilon}, -c, c\right).
$$

### FU6 BlockSecondMomentFunctionUpdate

对 D-CHE role block $r$：

$$
V_r=\operatorname{EMA}(g_r g_r^\top)+\rho I.
$$

更新：

$$
u_r=-V_r^{-1/2}m_r.
$$

如果 full block 太贵，只允许 low-rank sketch：

$$
V_r\approx D_r+U_rU_r^\top.
$$

## 6.5 Alignment variants

### FU3 CautiousFunctionUpdate

Coordinate-wise alignment：

$$
a_i=u_i(-g_i).
$$

Hard gate：

$$
u_i^{cautious}=u_i\mathbf{1}[a_i>0].
$$

Soft gate：

$$
u_i^{soft}=u_i \cdot \sigma(\tau a_i).
$$

默认必须同时测试 hard 与 soft，但作为同一个 FU3 family，不允许继续扩 token。

### FU4 MGUPFunctionUpdate

定义 alignment score：

$$
s_i=u_i(-g_i).
$$

按 $s_i$ 分位数重加权：

$$
u_i^{mgup}=
\begin{cases}
\alpha u_i, & s_i \in \text{top } \tau \\
\gamma u_i, & \text{otherwise}
\end{cases}
$$

固定：

```text
tau = 0.5
alpha = 2.0
gamma = 0.5
```

不允许 dataset/seed tuning。

## 6.6 Decoupled decay variants

### FU7 DecoupledDecayPlusFunctionUpdate

先应用 function update：

$$
\theta_{t+1/2}=\theta_t+\eta u_{FU}.
$$

再 decoupled role decay：

$$
\theta_{t+1,r}=(1-\eta\lambda_r)\theta_{t+1/2,r}.
$$

Role decay schedule 预注册：

```text
low_degree: 0
high_degree: lambda
readout: lambda_readout
bias: 0
```

不能把 decay gradient 混入 $g_t$。

## 6.7 Matched controls

每个 FU candidate 必须配：

```text
C0-D-CHE-AdamW
C1-NoOpMatchedOverhead
C2-RandomMatchedNorm
C3-SameActiveFractionRandomMask
C4-SameAlignmentMaskRandomDirection
C5-SameSecondMomentScaleRandomDirection
C6-SameDecoupledDecayNoFU
C7-CoupledL2RegularizationControl
C8-AdamWParallelDirection
C9-GenericOptimizerStateControl
```

只有超过所有 controls 才能说 FU 有 value。

## 6.8 指标

必须记录：

```text
source_vs_best_control
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass
step_time_ratio
memory_ratio
cos_fu_vs_gradient
cos_fu_vs_adamw
alignment_positive_fraction_raw
alignment_positive_fraction_preconditioned
second_moment_condition
curvature_clip_fraction
decoupled_decay_norm_fraction
projection_rejection_fraction
value_retention_after_projection
control_equivalent_fraction
bad_event_fraction
```

## 6.9 Gate

Exploration：

```text
real_lite_pass_count >= 3/9
source_vs_best_control_mean > 0
control_equivalent_fraction <= 0.60
bad_event_fraction <= 0.50
```

Meaningful exploration：

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean >= 0.002
control_equivalent_fraction <= 0.50
```

S4 exploration：

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
control_equivalent_fraction <= 0.40
```

S5 official：

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
```

---

# 7. Line M：MLP / Generic Optimizer Controls

## 7.1 目标

判断 FU 是否 KAN-specific，还是普通 optimizer trick。

## 7.2 MLP controls

```text
M0-MLP-AdamW
M1-MLP-AdamW-DecoupledDecay
M2-MLP-CautiousAdamW
M3-MLP-MGUPAdamW
M4-MLP-SophiaDiagAdamW
M5-MLP-BlockSecondMomentAdamW
M6-MLP-ScheduleFreeAdamW
M7-MLP-SameOverheadNoUpdate
```

## 7.3 判断

若：

```text
MLP-Cautious/MGUP/Sophia 过，
D-CHE-FU 不过，
```

则说明 optimizer insight 是 generic optimizer value，不是 KAN-specific functional update。

若：

```text
D-CHE-FU 过，
MLP/generic controls 不过，
```

才可写 KAN-specific functional leverage。

若：

```text
二者都过，
```

写成 generic optimizer + KAN carrier amplification，不写成纯 KAN-specific。

---

# 8. Line K：KAN-specificity and Decoupled Regularization Audit

## 8.1 Difference-in-differences

定义：

$$
\Delta_{KAN-specific}
=
[(D\text{-}CHE+FU)-(D\text{-}CHE+AdamW)]
-
[(MLP+FU_{analog})-(MLP+AdamW)].
$$

要求：

$$
\Delta_{KAN-specific}>0.
$$

否则不能声称 KAN-specific。

## 8.2 Decay deconfounding

必须回答：

```text
1. improvement 是否来自 decoupled decay？
2. coupled L2 是否更差？
3. cautious decay 是否解释结果？
4. no-decay FU 是否仍有 value？
```

如果 `SameDecoupledDecayNoFU` 解释了 FU gain，route：

```text
R-K-DecoupledRegularizationExplainsGain
```

不能 promotion functional update。

---

# 9. Line D：All-Basis Substrate 并行加速

## 9.1 目标

不能因为 D-CHE 是当前 carrier 就放弃其它 basis。

## 9.2 D-CHE

状态：主 carrier，做 no-regression 和 FU main。

必须记录：

```text
degree_energy
high_degree_fraction
degree_entropy
degree_projection_rejection_fraction
degree_role_second_moment
```

## 9.3 D-FOU

方向：

```text
D-FOU37-LowFreqIdentityResidualV4
D-FOU38-BandwiseSecondMomentWarmup
D-FOU39-PhaseStableCautiousUpdate
D-FOU40-NoMaterializeLifetimeV4
D-FOU41-HighFrequencyQuarantineV2
```

目标：从 current 0/9 or historical 6/9 reconfirmation gap 中找出 reproducible substrate。

## 9.4 D-RBF / FastKAN

方向：

```text
D-RBF35-ActiveCenterSecondMoment
D-RBF36-WidthConditionDecoupledDecay
D-RBF37-CompactBumpIdentityResidualV2
D-RBF38-GaussianLocalK4TaskHealthV2
D-RBF39-NoDenseCenterUpdate
```

重点：RBF/FastKAN 不能只看 workspace；必须解决 task collapse。

## 9.5 D-WAV

方向：

```text
D-WAV33-TriangularSupportV4
D-WAV34-ScaleSecondMomentOccupancy
D-WAV35-SupportOverlapCautiousUpdate
D-WAV36-LocalTailCoverageAuditOnly
```

低预算 monitor，不进入 official FMS unless >=6/9 exploration.

## 9.6 Substrate gates

Exploration substrate：

```text
family_dataset_seed_pass_count >= 6/9
step_ratio <= 1.75
memory_ratio <= 1.75
mean_delta >= -0.05
worst_delta >= -0.10
LineC_pass_rate >= 0.30
```

Official FMS eligibility：

```text
family_dataset_seed_pass_count = 9/9
step_ratio <= 1.25
memory_ratio <= 1.25
LineC_pass_rate >= 0.70
```

未达到 6/9 不 hard stop 全轮；但不得进入 official FU proof。

---

# 10. Line C：Geometry / Tail / LineC Audit

## 10.1 目标

Line C 只做审计，不生成 direction。

## 10.2 必须记录

```text
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
NLL_delta
ECE_delta
Brier_delta
margin_p10_delta
LineC_pass
signal_channel_energy
reservoir_energy
noise_leakage_proxy
```

## 10.3 可视化

```text
fig_v150_linec_signal_reservoir_noise.svg
fig_v150_tail_calibration_dashboard.svg
fig_v150_alignment_vs_linec.svg
fig_v150_second_moment_vs_tail.svg
```

---

# 11. Unified success / failure routes

## S1：Optimizer mechanism implemented

```text
decoupled_decay_implemented = 1
second_moment_implemented = 1
alignment_trace_available = 1
controls_available = 1
```

## S2：FU exploration positive

```text
real_lite_pass_count >= 3/9
source_vs_best_control_mean > 0
controls not explaining
```

## S3：D-CHE FU meaningful positive

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean >= 0.002
control_equivalent_fraction <= 0.50
```

## S4：D-CHE FU exploration strong

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
control_equivalent_fraction <= 0.40
```

## S5：Official success

```text
real_dataset_seed_pass_count = 9/9
all source / AUC / tail / LineC / efficiency gates pass
promotion_allowed = 1
```

## R routes

```text
R1-AlignmentDoesNotHelp:
  cautious/MGUP 不改善。

R2-SecondMomentDoesNotHelp:
  AdamV2/Sophia/BlockSecondMoment 不改善。

R3-DecoupledDecayExplainsGain:
  gain 被 decoupled decay controls 解释。

R4-FUControlEquivalent:
  random/matched controls 解释 result。

R5-FUValueButGeneric:
  MLP/generic optimizer 同样解释。

R6-FUValueButCostBlocked:
  value pass 但 step/memory fail。

R7-AllBasisCarrierBlocked:
  D-CHE 外仍无 substrate。

R8-CurrentFUDefinitionNoGo:
  FU1..FU8 全 fail。
```

---

# 12. Stop / continue contract

## 12.1 Promotion fail-closed

Promotion 只能在 S5 时打开。其它任何情况：

```text
promotion_allowed = 0
official_s5_reached = 0
```

## 12.2 Exploration continue-open

以下情况不能 hard stop：

```text
FU1 fail
FU2 fail
FU3 fail
FU4 fail
FU5 fail
FU6 fail
FU7 fail
FU8 fail
Line M positive
Line D family <6/9
LineC fail
tail fail
AUC fail
overhead fail
```

它们只能写 route / failure taxonomy / next hypothesis，并继续其它独立线。

## 12.3 Hard stop only

只有以下情况 hard stop：

```text
required_artifact_missing_count > 0
forbidden_information_violation_count > 0
no_action_search_violation_count > 0
direction uses validation/test/future/query
direction uses LineC/CEp99/NLL/ECE/AUCtime
dataset-name branch
seed-specific scale
action token / controller / action bank / reset route
fake / proxy / CPU offload
```

---

# 13. Required artifacts

```text
v150_route_decision.json
v150_required_manifest.csv
v150_forbidden_information_audit.csv
v150_no_action_search_audit.csv
v150_optimizer_mechanism_manifest.csv
v150_decoupled_decay_audit.csv
v150_second_moment_trace.csv
v150_alignment_trace.csv
v150_curvature_diag_trace.csv
v150_dche_fu_results.csv
v150_dche_fu_controls.csv
v150_mlp_controls.csv
v150_kan_specificity_summary.csv
v150_decoupled_decay_deconfound.csv
v150_allbasis_substrate_results.csv
v150_linec_tail_audit.csv
v150_failure_taxonomy.csv
v150_code_review_packet.zip
```

---

# 14. Required visualizations

```text
fig_v150_progress_table.svg
fig_v150_fu_vs_controls.svg
fig_v150_alignment_histogram.svg
fig_v150_second_moment_scale_distribution.svg
fig_v150_decoupled_decay_deconfound.svg
fig_v150_cautious_vs_mgup_ablation.svg
fig_v150_sophia_block_moment_ablation.svg
fig_v150_mlp_vs_dche_optimizer_specificity.svg
fig_v150_allbasis_substrate_heatmap.svg
fig_v150_linec_tail_dashboard.svg
fig_v150_runtime_breakdown.svg
```

---

# 15. Codex execution order

Codex 必须按顺序执行：

```text
1. Line R audit；
2. Line O mechanism implementation audit；
3. Line FU D-CHE optimizer-aware FU；
4. Line M MLP/generic controls；
5. Line K KAN-specificity / decoupled deconfound；
6. Line D all-basis substrate；
7. Line C geometry/tail audit；
8. Line Z route/no-go/next hypothesis。
```

但可以并行执行 FU/M/D/C，只要最终 route 在 Line R/O 完成后统一决策。

Codex 不许：

```text
1. 新增 FU9/FU10；
2. 新增 F-CHE8/F-CHE9；
3. 启动 controller；
4. 使用 action bank；
5. 新增 reset route；
6. 用 audit metric 生成 direction；
7. 按 dataset/seed 写 branch；
8. 把 decoupled regularization 写成 functional gain；
9. 把 MLP/generic success 写成 KAN-specific；
10. 把 local positive row 写成 promotion。
```

---

# 16. 最终判断

v15.0 的核心不是放弃 FPU，也不是回到旧 FMS，而是把 function update 放回现代 optimizer 语境中重新定义：

$$
\boxed{
\text{Function Update must be optimizer-aware:}
\quad
\text{second moment}
+
\text{alignment}
+
\text{decoupled regularization}
+
\text{basis-safe projection}.
}
$$

如果 v15.0 成功，说明之前 FMS no-go 的根因是缺少 optimizer-aware functional machinery。  
如果 v15.0 失败，则 current train-stream function update family 很可能 no-go，应冻结这一 functional family，转向 substrate/base architecture 或更强 theoretical reset。  
