# DG-KAN v8.7 Stability-First Functional Architecture：从“可调参达标”到“稳定方法”的完整实验计划

> 本计划基于 v8.4、v8.5、v8.6 的真实结果重新制定。  
> 当前结论不是“路线失败”，而是：**方法还不够稳健**。  
> v8.5 已经出现了强的外部公平成功点，但它仍依赖特定配置，例如 `hidden28 + FT7 + stride128`，并且此前 v8.4 也显示不同 timing protocol 下 gate 会不稳定。  
> 因此 v8.7 不再以“继续调到某个点过线”为目标，而是把当前方法升级成一个**稳定、自动、可复现、跨任务公平**的 functional architecture method。  
> 本轮继续遵守：**no teacher、no self-teacher、no distillation、no loss modification、no sampler/class weight、no CPU offload、no optimizer hyperparameter sweep**。Functional update 仍然必须是 update rule，不是 loss。

---

## 0. 当前判断：我同意“方法还需要改进优化”

### 0.1 为什么不能直接满足于 v8.5 strong route

v8.5 按文档 route-level 判据已经达成了 strong success，但这并不等于方法已经成熟。当前成功点最核心的是：

```text
DG1-FT7-KW6 hidden28 stride128
KANbeFair MNIST full protocol
accuracy > KB-MLP
params < KB-MLP
forward FLOPs < KB-MLP
step ratio <= 1.50
functional curvature ratio <= 0.90
```

这个结果很重要，但它仍然暴露出两个问题。

第一，它是一个具体配置点，而不是一个自动稳定方法。`hidden28`、`stride128`、FT7 的 functional event cadence、role budget、guard 机制共同构成了成功条件。早期 hidden28 原始 FT7 有 task / params / FLOPs / geometry 优势，但 wall-clock 失败；后来通过 stride screen 才找到 stride128 这个更稳定点。这说明方法仍有配置敏感性。

第二，它仍主要在 KANbeFair MNIST full protocol 上最硬，尚不能推出“所有任务族广泛优于 MLP”。KANbeFair 本身的外部结论是：在参数量或 FLOPs 公平设置下，KAN 通常只在 symbolic formula representation 上优于 MLP，而在 machine learning、computer vision、NLP、audio 等任务上多数情况下弱于 MLP。因此，v8.5 的成功应该被看作一个重要突破点，而不是最终普适定理。

### 0.2 当前方法的不稳定具体表现

当前不稳定不是一个抽象担忧，而是来自三类真实现象。

**第一类：timing protocol 敏感。**  
v8.4 中，`KW6 hidden68` 在原 `50 warmup / 200 reps` official-style fresh/repeat reproduction 下 H0 step gate 不稳；fresh/repeat 都没有通过 H0。但在 high-rep timing-stabilized diagnostic 下，H0/P7/P8/P9 链条可以闭合。这说明当前系统 gate 对 benchmark protocol、warmup/reps、sync、phase timing 较敏感。

**第二类：configuration 敏感。**  
v8.5 中，hidden68 task 很强但参数/FLOPs 不公平；hidden28 进入参数/FLOPs fair envelope，但原始 FT7 太慢；stride32 曾经过但 repeat 不稳；stride64/stride128 才形成更可接受的 wall-clock/geometry 平衡。这说明 fixed hyperparameters 仍然是成功的重要条件。

**第三类：functional update 的作用需要从“可用”升级到“自适应”。**  
FT7 目前靠 role-wise guarded correction、event stride、alpha multiplier、pre-holdout / train-budget fallback 等机制控制风险。它已经比早期 naked functional update 成熟，但这些超参如果仍要人工挑选，就说明 functional update 还不是一个稳定方法。

因此我同意：

$$
\boxed{
\text{当前方法已经有强信号，但还没有足够稳健；下一步必须改进方法本身，而不是继续手工调参过 gate。}
}
$$

---

## 1. v8.7 总体目标

v8.7 的目标不是再找一个更好的 stride、hidden、alpha 或 role budget。v8.7 的目标是把当前路线从：

```text
一个可调参后达标的 functional KAN recipe
```

升级成：

```text
一个无需人工阈值追逐、可自动选择 event / role / budget、可跨 seed / protocol / task 复现的稳定方法。
```

正式写成：

$$
\boxed{
\text{DG-KAN + functional update must become stability-first, adaptive, and externally fair.}
}
$$

具体目标分四层。

### 1.1 Stability Success

给定固定规则而不是手工挑参，candidate 在多个 seed、多个 rerun、多个 timing protocol 中稳定满足：

$$
Acc_{\text{DG}}\geq Acc_{\text{MLP}},
$$

$$
Params_{\text{DG}}\leq1.05Params_{\text{MLP}},
$$

$$
FLOPs_{\text{DG}}\leq1.05FLOPs_{\text{MLP}},
$$

$$
StepTime_{\text{DG}}\leq1.50StepTime_{\text{MLP}},
$$

$$
R_{\text{curv,DG}}\leq0.90R_{\text{base}}.
$$

并且 pass rate 满足：

$$
P(\text{all gates pass})\geq0.80
$$

across independent reruns.

### 1.2 Adaptive Functional Success

Functional update 不再使用固定 `stride / alpha / role_budget` 作为主成功 recipe，而是使用可解释的 adaptive controller：

$$
\Delta\theta_t
=
\Delta\theta_{\text{task},t}
+
\sum_r \lambda_r(t) P_{\mathcal{T}}(\Delta\theta_{\text{func},r,t}),
$$

其中 $r$ 表示 role，例如 stack、head、basis、mix、residual。$\lambda_r(t)$ 由 task descent、geometry pressure、role stability、system budget 自动决定。

Adaptive success 要求：

$$
Acc_{\text{adaptive}}\geq Acc_{\text{fixed-best}}-0.002,
$$

$$
R_{\text{curv,adaptive}}\leq R_{\text{fixed-best}}+0.05,
$$

$$
StepTime_{\text{adaptive}}\leq1.05StepTime_{\text{fixed-best}},
$$

并且 adaptive controller 的 pass variance 更低：

$$
Var(\text{gate margin}_{\text{adaptive}})<Var(\text{gate margin}_{\text{fixed}}).
$$

### 1.3 External Fair Success

不仅在 MNIST full protocol 上成立，还要在更多 KANbeFair task family 上给出清晰结论：

```text
broad external fair win
task-family-limited fair win
symbolic / geometry-only win
negative external result
```

v8.7 不强行要求所有任务都赢，但必须明确边界。

### 1.4 No Manual Tuning Success

v8.7 的 route 不允许通过“看结果再挑 stride/alpha/hidden”写成功。必须使用预注册选择规则：

```text
hidden 由 params/FLOPs envelope 自动确定；
functional event 由 controller 自动触发；
role budget 由 curvature/task contribution 自动分配；
timing protocol 使用固定 robust gate；
candidate 选择使用 pre-registered score。
```

---

## 2. 保持不变的硬约束

v8.7 继续严格禁止：

```text
external teacher
self teacher
teacher logits
distillation
self-distillation
label smoothing
focal loss
margin loss
calibration loss
NLL-balanced loss
geometry loss as training objective
sampler / class weight
oversampling / undersampling
test-based selection
CPU offload
optimizer hyperparameter sweep
```

允许 functional update，但它必须是 update rule，而不是 loss：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

允许：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\Delta\theta_{\text{functional}}.
$$

不允许：

$$
L = CE+\lambda L_{\text{geo}}.
$$

所有 official rows 必须记录：

```text
loss_type = CE
geometry_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
teacher_logits_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
fake_data_used = 0
proxy_row_used = 0
```

---

## 3. 当前方法为什么会不稳定：本质假设

### H1：不稳定主要来自 fixed functional event schedule，而不是 functional update 本身

当前 FT7 需要 stride、alpha、role budget、pre-holdout、fallback 等条件。H1 假设：functional direction 是有价值的，但 fixed schedule 会在不同 seed / protocol / task 下触发过多或过少，导致有时 wall-clock 过慢，有时几何不足，有时 task 波动。

H1 成立标准：

在 fixed schedule sensitivity grid 中，若 pass/fail 对 stride 或 alpha 敏感：

$$
\max_{s,a} Acc(s,a)-\min_{s,a} Acc(s,a)\geq0.003
$$

或：

$$
\max_{s,a} StepRatio(s,a)-\min_{s,a} StepRatio(s,a)\geq0.20,
$$

则 fixed schedule 是不稳定来源。

如果 adaptive controller 能降低该 variance：

$$
Var_{\text{adaptive}}(\text{gate margin})<0.5Var_{\text{fixed}}(\text{gate margin}),
$$

则 H1 支持 adaptive schedule 路线。

### H2：functional update 需要“task descent budget”，不能只按几何事件触发

几何改善与任务下降存在冲突。H2 假设：functional update 必须受 task descent budget 约束，而不是只看 curvature。

定义每步 task descent budget：

$$
B_t=
\max(0,L_{\text{holdout}}(\theta_t)-L_{\text{holdout}}(\theta_t+\Delta\theta_{\text{task}})).
$$

functional correction 后必须满足：

$$
L_{\text{holdout}}(\theta_t+\Delta\theta_{\text{task}}+\Delta\theta_{\text{func}})
\leq
L_{\text{holdout}}(\theta_t)-\rho B_t,
$$

其中：

$$
\rho\geq0.95.
$$

H2 成立标准：

adaptive budget controller 相比 fixed FT7：

$$
BadStepRate_{\text{adaptive}}\leq BadStepRate_{\text{fixed}},
$$

$$
HoldoutDescentRatio_{\text{adaptive}}\geq0.95,
$$

并且：

$$
R_{\text{curv,adaptive}}\leq0.90R_{\text{base}}.
$$

### H3：role budget 应该由 role contribution 自动分配

当前 role budget 是固定的，例如 stack/head 各自手动给 budget。H3 假设：不同任务、不同阶段、不同数据集里，stack/head/basis/mix 的贡献不同，固定 role budget 造成不稳定。

定义 role score：

$$
S_r(t)
=
\operatorname{ReLU}(\Delta R_r(t))
\cdot
\operatorname{ReLU}(\cos(d_{\text{func},r},d_{\text{task},r}))
\cdot
G_r(t)
\cdot
C_r(t),
$$

其中：

```text
Delta R_r(t): role r 的可降低几何量
cos(...): functional direction 与 task direction 一致性
G_r(t): role r 的历史 geometry gain
C_r(t): role r 的系统成本惩罚因子
```

role budget 自动分配：

$$
b_r(t)=b_{\text{total}}\frac{S_r(t)}{\sum_j S_j(t)+\epsilon}.
$$

H3 成立标准：

adaptive role budget 相比 fixed role budget：

$$
R_{\text{curv,adaptive}}\leq R_{\text{curv,fixed}}+0.05,
$$

$$
StepRatio_{\text{adaptive}}\leq StepRatio_{\text{fixed}},
$$

且：

$$
AccDrop_{\text{adaptive}}\leq0.002.
$$

### H4：hidden / width 不应该再手动试，而应由公平 envelope 自动确定

v8.5 的 hidden28 是有效点，但如果每个任务都手动调 hidden，会削弱方法可信度。H4 假设：应根据参数/FLOPs/step 预算自动确定 hidden。

定义 candidate hidden 选择规则：

$$
h^*
=
\max_h
\left\{
h:
Params(h)\leq\alpha_p Params_{\text{MLP}},
FLOPs(h)\leq\alpha_f FLOPs_{\text{MLP}},
StepPilot(h)\leq\alpha_t Step_{\text{MLP}}
\right\}.
$$

默认：

$$
\alpha_p=1.00,\quad \alpha_f=1.00,\quad \alpha_t=1.50.
$$

H4 成立标准：

自动 $h^*$ 的 performance 不低于人工 hidden28：

$$
Acc(h^*)\geq Acc(h=28)-0.002,
$$

并且公平 envelope 自动通过：

$$
Params(h^*)\leq1.05Params_{\text{MLP}},
$$

$$
FLOPs(h^*)\leq1.05FLOPs_{\text{MLP}}.
$$

### H5：timing instability 必须用 robust gate，而不是选择对自己有利的 protocol

v8.4 说明 official-style 50/200 与 high-rep timing 可能给出不同 gate 结果。H5 假设：单一 timing protocol 不可靠，必须用多 protocol robust gate。

定义 protocols：

```text
T0: 50 warmup / 200 reps
T1: 100 warmup / 500 reps
T2: 200 warmup / 1000 reps
T3: train-step-only phase-clean
T4: full-loop including guard / validation sync
```

Robust timing gate：

$$
Q_{90}(\text{StepRatio across protocols/reruns})\leq1.50.
$$

H5 成立标准：

如果 candidate 只在 T2 high-rep 下通过，但 T0/T1/T4 失败，则只能写：

```text
timing-protocol-sensitive success
```

不能写 formal system success。

### H6：如果跨任务不稳，必须输出边界而不是继续调参

H6 允许 negative result。若 DG-KAN 在多数非 symbolic tasks 上无法通过 fairness envelope，则应写明任务族边界。

H6 触发标准：

$$
Acc_{\text{DG}} < Acc_{\text{MLP}} - 0.01
$$

in most non-symbolic tasks under fair envelope。

此时 route 不是失败，而是：

```text
TaskFamilyLimitedFunctionalAdvantage
```

---

## 4. 方法改进：v8.7 新机制

v8.7 不再新增 teacher/loss/optimizer，而是新增三个方法层模块。

### 4.1 Adaptive Functional Event Controller

当前固定 event stride 的问题是：不同阶段需要不同频率。早期训练 task descent 更重要，functional event 应少；中后期表示稳定后，functional geometry correction 更有用。v8.7 用 event score 自动触发：

$$
E(t)
=
w_g \cdot \operatorname{Norm}(\Delta R(t))
+
w_m \cdot \operatorname{Norm}(\Delta MarginStability(t))
-
w_c \cdot \operatorname{Norm}(Cost(t))
-
w_b \cdot \operatorname{Norm}(BadStepRisk(t)).
$$

触发条件：

$$
E(t)>\tau_t,
$$

其中 $\tau_t$ 不是固定值，而是用滑动窗口分位数：

$$
\tau_t=Q_{75}(E(t-k),...,E(t-1)).
$$

这样 event frequency 会自动适配任务和训练阶段。

必须记录：

```text
event_score
event_threshold
event_triggered
event_interval
functional_update_norm
task_budget
holdout_descent_ratio
bad_step_flag
functional_update_time
```

### 4.2 Task-Budgeted Role-Wise Functional Update

对每个 role 单独计算 functional correction，但不允许超过 task budget。

role-wise update：

$$
\Delta\theta_{\text{func},r}
=
-\lambda_r(t)
P_{\mathcal{T}}
\left(
M_r^{-1}\nabla_{\theta_r}R_{\text{geo}}
\right).
$$

其中 $M_r$ 是 role-local diagonal / low-rank metric approximation，不能 full Jacobian materialization。

约束：

$$
\frac{\|\Delta\theta_{\text{func},r}\|}
{\|\theta_r\|+\epsilon}
\leq b_r(t).
$$

并且整体 corrected direction 要满足：

$$
\cos(d_{\text{corrected}},d_{\text{task}})\geq0.85.
$$

### 4.3 Stability-Normalized Base Architecture Selection

不再手动选 hidden，而是每个 task 根据公平 envelope 自动确定。

候选集：

```text
hidden candidates = {16,20,24,28,32,36,40,48,56,68}
event controller = adaptive
functional rule = same across hidden
```

选择规则：

```text
先过滤 params/FLOPs/step pilot 不合格；
再选择 H0Base score 最高；
不得使用 test metric 选 hidden；
不得使用 final test 或 robustness 结果调整 hidden。
```

score：

$$
Score(h)
=
Acc_{\text{val,pilot}}(h)
-
\lambda_p\max(0,ParamsRatio(h)-1)
-
\lambda_f\max(0,FLOPsRatio(h)-1)
-
\lambda_t\max(0,StepRatio(h)-1.5)
+
\lambda_g GeometryScore(h).
$$

pilot 只允许使用 train/val，不允许使用 test。

---

## 5. 实验阶段总览

v8.7 分为九个 Wave：

```text
Wave 0:
  accepted route reproduction and instability audit

Wave 1:
  fixed hyperparameter sensitivity map

Wave 2:
  adaptive functional controller implementation

Wave 3:
  stability-normalized architecture selection

Wave 4:
  robust timing and training-compute fairness

Wave 5:
  KANbeFair multi-task external validation

Wave 6:
  continual balance and robustness validation

Wave 7:
  mechanism attribution and boundary analysis

Wave 8:
  final route decision
```

---

## 6. Wave 0：accepted route reproduction and instability audit

### P0：fresh reproduction of current accepted route

#### 目标

先复现 `DG1-FT7-KW6 hidden28 stride128`，并量化不稳定性，而不是只报告 pass/fail。

#### 必跑对象

```text
KB-MLP
KB-KAN
DG0-KW6 hidden28 stride128 base
DG1-FT7 hidden28 stride128
DG-NoOp
DG-RandomFunc
```

#### 设置

```text
datasets = KANbeFair MNIST full protocol
seeds = 5 independent seeds
reruns = 3 independent out-dirs
timing protocols = T0,T1,T2,T3,T4
```

#### 必须记录

```text
run_id
seed
candidate
test_acc
delta_vs_KB_MLP
params
forward_FLOPs
backward_FLOPs_estimate
step_time_ms
step_ratio
train_time_s
peak_memory_MB
curvature
curvature_ratio_vs_DG0
functional_events
event_intervals
functional_update_time_ratio
NoTeacherNoLossNoOffloadPass
FunctionalCausalityPass
fake_data_used
proxy_row_used
```

#### 判断标准

Reproduction pass：

$$
\operatorname{mean}(Acc_{\text{DG1}}-Acc_{\text{KB-MLP}})>0.
$$

Robust timing pass：

$$
Q_{90}(StepRatio)\leq1.50.
$$

Functional geometry pass：

$$
\operatorname{mean}(R_{\text{curv,DG1}}/R_{\text{curv,DG0}})\leq0.90.
$$

#### 必须可视化

```text
p0_acc_delta_rerun_boxplot.svg
p0_step_ratio_by_protocol.svg
p0_curvature_ratio_by_rerun.svg
p0_event_interval_distribution.svg
p0_pass_fail_waterfall.svg
```

---

## 7. Wave 1：fixed hyperparameter sensitivity map

### P1：fixed FT7 sensitivity grid

#### 目标

量化当前 success 对 stride、alpha、role budget、hidden 的依赖，为 adaptive controller 提供证据。

#### 不允许

P1 不能用于“挑最好点写成功”。它是敏感性分析，不是最终 route selection。

#### Grid

```text
hidden = {24,28,32}
stride = {32,64,128,256}
alpha = {5,10,15,20}
role_budget = {0.05,0.10,0.15,0.25}
seeds = 0,1,2
```

#### 必须记录

```text
hidden
stride
alpha
role_budget
test_acc
delta_vs_KB_MLP
curvature_ratio
step_ratio
memory_ratio
functional_events
bad_step_rate
holdout_descent_ratio
functional_update_time_ratio
pass_primary
pass_params
pass_flops
pass_wallclock
pass_geometry
```

#### 判断标准

Sensitivity evidence：

如果任一维度改变导致：

$$
\Delta Acc\geq0.003
$$

或：

$$
\Delta StepRatio\geq0.20
$$

或：

$$
\Delta CurvatureRatio\geq0.20,
$$

则说明 fixed recipe 不稳，adaptive controller 有必要。

#### 必须可视化

```text
p1_stride_alpha_accuracy_heatmap.svg
p1_stride_alpha_step_heatmap.svg
p1_role_budget_curvature_heatmap.svg
p1_hidden_fairness_pareto.svg
p1_hyperparam_pass_rate_matrix.svg
p1_sensitivity_sobol_bar.svg
```

---

## 8. Wave 2：adaptive functional controller implementation

### P2：one-step adaptive controller audit

#### 目标

验证 adaptive controller 在单步上比 fixed FT7 更安全、更少依赖超参。

#### 必跑对象

```text
Fixed-FT7-stride128
Adaptive-FT-A-event-score
Adaptive-FT-B-task-budgeted
Adaptive-FT-C-role-adaptive
Adaptive-FT-D-full-controller
NoOp-control
RandomFunc-control
```

#### 必须记录

```text
event_score
event_threshold
triggered
role_score_stack
role_score_head
role_budget_stack
role_budget_head
cos_corrected_task
holdout_descent_ratio
bad_step_flag
functional_update_norm
curvature_reduction
functional_update_time_ms
```

#### 判断标准

One-step adaptive pass：

$$
\cos(d_{\text{corrected}},d_{\text{task}})\geq0.85,
$$

$$
HoldoutDescentRatio\geq0.95,
$$

$$
BadStepRate\leq0.02.
$$

Adaptive better than fixed if：

$$
BadStepRate_{\text{adaptive}}\leq BadStepRate_{\text{fixed}},
$$

and:

$$
CurvatureReduction_{\text{adaptive}}\geq0.8CurvatureReduction_{\text{fixed}}.
$$

#### 必须可视化

```text
p2_event_score_threshold_timeline.svg
p2_role_budget_timeline.svg
p2_cosine_holdout_scatter.svg
p2_bad_step_rate_bar.svg
p2_curvature_reduction_vs_time.svg
```

### P3：multi-step adaptive smoke

#### 目标

检验 adaptive controller 在 20/50/240 step 中是否稳定。

#### 设置

```text
datasets = MNIST full protocol
seeds = 0,1,2
steps = 20,50,240
```

#### 必须记录

```text
train_loss_curve
test_acc_curve
ECE_curve
NLL_curve
curvature_curve
step_time_curve
event_count_curve
event_interval_curve
role_budget_curve
bad_step_rate_curve
holdout_descent_ratio_curve
```

#### 判断标准

Multi-step pass：

$$
Acc_{\text{adaptive}}\geq Acc_{\text{fixed}}-0.002,
$$

$$
CurvatureRatio_{\text{adaptive}}\leq0.90,
$$

$$
StepRatio_{\text{adaptive}}\leq1.50.
$$

#### 必须可视化

```text
p3_loss_acc_curve.svg
p3_curvature_curve.svg
p3_event_frequency_curve.svg
p3_step_time_curve.svg
p3_adaptive_vs_fixed_pareto.svg
```

---

## 9. Wave 3：stability-normalized architecture selection

### P4：automatic hidden / budget selection

#### 目标

替代手工 hidden28，建立公平 envelope 下自动选 architecture 的规则。

#### 候选

```text
hidden = {16,20,24,28,32,36,40,48,56,68}
basis / role config = current accepted default
controller = adaptive
```

#### 必须记录

```text
hidden
params
forward_FLOPs
backward_FLOPs_estimate
step_pilot
memory_pilot
pilot_val_acc
pilot_curvature
Score(h)
selected_by_rule
test_acc_final
```

#### 判断标准

Auto selection pass：

$$
Params(h^*)\leq1.05Params_{\text{MLP}},
$$

$$
FLOPs(h^*)\leq1.05FLOPs_{\text{MLP}},
$$

$$
StepPilot(h^*)\leq1.50Step_{\text{MLP}},
$$

and:

$$
Acc(h^*)\geq Acc(h=28)-0.002.
$$

#### 必须可视化

```text
p4_hidden_score_curve.svg
p4_params_flops_hidden_curve.svg
p4_hidden_task_system_pareto.svg
p4_auto_selected_hidden_marker.svg
```

---

## 10. Wave 4：robust timing and training-compute fairness

### P5：multi-protocol timing audit

#### 目标

解决当前 timing protocol 敏感问题，不再只依赖单一 warmup/reps。

#### Protocols

```text
T0 = 50 warmup / 200 reps
T1 = 100 warmup / 500 reps
T2 = 200 warmup / 1000 reps
T3 = train-step-only phase-clean
T4 = full-loop with guard / validation sync
```

#### 必须记录

```text
protocol_id
candidate
step_time_ms
step_ratio
forward_time_ms
backward_time_ms
base_update_time_ms
functional_update_time_ms
guard_time_ms
cuda_sync_time_ms
validation_time_ms
unknown_time_fraction
kernel_time_ms
```

#### 判断标准

Robust timing pass：

$$
Q_{90}(StepRatio)\leq1.50.
$$

Strict timing pass：

$$
\max(StepRatio)\leq1.50.
$$

Time accounting pass：

$$
unknown\_time\_fraction\leq0.10.
$$

#### 必须可视化

```text
p5_step_ratio_protocol_boxplot.svg
p5_time_breakdown_stacked.svg
p5_q90_timing_gate.svg
p5_unknown_time_fraction_bar.svg
```

### P6：training compute counter

#### 目标

补齐 forward-only FLOPs 的弱点。

#### 必须记录

```text
forward_FLOPs
backward_FLOPs_estimate
update_FLOPs_estimate
functional_update_FLOPs_estimate
kernel_time_forward
kernel_time_backward
kernel_time_update
kernel_time_functional
train_step_energy_proxy_if_available
```

#### 判断标准

Training compute fair pass：

至少满足两项：

$$
ForwardFLOPs_{\text{DG}}\leq1.05ForwardFLOPs_{\text{MLP}},
$$

$$
BackwardFLOPs_{\text{DG}}\leq1.50BackwardFLOPs_{\text{MLP}},
$$

$$
KernelTime_{\text{DG}}\leq1.50KernelTime_{\text{MLP}},
$$

$$
StepTime_{\text{DG}}\leq1.50StepTime_{\text{MLP}}.
$$

#### 必须可视化

```text
p6_forward_backward_flops_bar.svg
p6_kernel_time_by_phase.svg
p6_training_compute_vs_accuracy.svg
```

---

## 11. Wave 5：KANbeFair multi-task external validation

### P7：broad task transfer

#### 目标

检查 adaptive DG-KAN 是否跨任务成立。

#### 任务族

```text
Vision:
  MNIST, Fashion-MNIST, KMNIST, CIFAR-10 if runnable

Tabular / machine learning:
  at least two KANbeFair tabular datasets

Symbolic:
  KANbeFair symbolic formula tasks

NLP:
  one text classification task if runnable

Audio:
  one audio classification task if runnable
```

#### 必跑对象

```text
KB-MLP
KB-KAN
DG-Base-Auto
DG-Fixed-FT7
DG-Adaptive-FT
DG-NoOp
DG-RandomFunc
```

#### 必须记录

```text
task_family
task_name
candidate
seed
metric
delta_vs_KB_MLP
delta_vs_DG_Base
params_ratio
forward_flops_ratio
training_compute_ratio
step_ratio
memory_ratio
curvature_ratio
ECE
NLL
functional_events
```

#### 判断标准

Broad fair success：

至少两个非 symbolic task families 满足：

$$
Metric_{\text{DG-Adaptive}}\geq Metric_{\text{KB-MLP}},
$$

and joint fair envelope pass.

Task-family-limited success：

MNIST-like / symbolic / geometry-sensitive tasks pass，但多数 non-symbolic tasks 不 pass。

#### 必须可视化

```text
p7_task_family_win_loss_heatmap.svg
p7_delta_vs_mlp_by_task.svg
p7_task_family_pareto.svg
p7_boundary_map.svg
```

---

## 12. Wave 6：continual balance and robustness validation

### P8：balanced continual learning formalization

#### 目标

确认 anti-forgetting update-rule 不只是降低 forgetting，也不偏向旧 task。

#### Splits

```text
Split A: digits 0-2 / 3-5 / 6-9
Split B: digits 0-4 / 5-9
Split C: random balanced 3-task split
Split D: interleaved hard split
```

#### 必跑对象

```text
KB-MLP
KB-KAN
DG-Base
DG-Adaptive-FT
DG-Adaptive-no-old-head-restore
DG-Adaptive-no-stack-anchor
DG-Adaptive-new-task-bias-control
```

#### 必须记录

```text
split_id
task_id
candidate
acc_after_each_task
final_task_acc_vector
final_avg_acc
forgetting_score
backward_transfer
forward_transfer
old_task_acc
new_task_acc
max_min_task_acc_gap
curvature_after_task
functional_events
anchor_strength
```

#### 判断标准

Balanced continual pass：

$$
Forgetting_{\text{DG-Adaptive}}\leq Forgetting_{\text{DG-Base}},
$$

$$
FinalAvgAcc_{\text{DG-Adaptive}}\geq FinalAvgAcc_{\text{DG-Base}},
$$

and:

$$
\max_i Acc_i-\min_i Acc_i\leq0.20.
$$

#### 必须可视化

```text
p8_continual_accuracy_matrix.svg
p8_forgetting_by_split.svg
p8_task_balance_bar.svg
p8_curvature_forgetting_scatter.svg
```

### P9：robustness and perturbation

#### 目标

检查 functional geometry 是否真的带来鲁棒性。

#### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10,0.20
random_erasing = small,medium
affine_shift = mild
```

#### 必须记录

```text
noise_type
noise_level
clean_acc
noisy_acc
accuracy_drop
ECE_under_noise
NLL_under_noise
curvature_under_noise
robustness_auc
functional_events
step_ratio
memory_ratio
```

#### 判断标准

Robustness pass：

$$
AccDrop_{\text{DG-Adaptive}}\leq AccDrop_{\text{DG-Base}}
$$

for at least 3 settings.

Strong robustness pass：

$$
RobustnessAUC_{\text{DG-Adaptive}}\geq RobustnessAUC_{\text{DG-Base}}.
$$

#### 必须可视化

```text
p9_accuracy_drop_bar.svg
p9_robustness_auc.svg
p9_ece_under_noise.svg
p9_curvature_under_noise.svg
```

---

## 13. Wave 7：mechanism attribution and boundary analysis

### P10：functional mechanism attribution

#### 目标

解释 adaptive functional update 何时有效、靠什么有效。

#### 必须记录

```text
task_name
task_family
role
role_score
role_budget
role_accept_rate
role_update_norm
role_curvature_delta
role_task_delta
role_system_cost
event_score
event_threshold
event_interval
```

#### 判断标准

Mechanism attribution pass：

至少 70% 的 curvature reduction 可以归因到 top roles/events：

$$
\frac{\sum_{i=1}^{k}\Delta R_i}{\Delta R_{\text{total}}}\geq0.70.
$$

#### 必须可视化

```text
p10_role_curvature_waterfall.svg
p10_event_score_timeline.svg
p10_role_budget_by_task.svg
p10_update_norm_vs_geometry_delta.svg
```

### P11：negative-result boundary audit

#### 目标

如果某些任务失败，输出边界而不是继续调参。

#### 必须记录

```text
task_family
task_name
DG_vs_MLP_delta
params_ratio
flops_ratio
training_compute_ratio
step_ratio
geometry_delta
failure_reason
boundary_label
```

Boundary labels：

```text
broad_fair_win
mnist_like_win
symbolic_win
geometry_robustness_win
forward_flops_only
training_compute_fail
wallclock_fail
mlp_dominates
continual_unbalanced
adapter_unstable
```

#### 必须可视化

```text
p11_boundary_matrix.svg
p11_route_by_task_family.svg
p11_negative_result_table.md
```

---

## 14. Wave 8：final route decision

### P12：route decision

#### Route cases

```text
R1-StableBroadExternalFunctionalAdvantage:
  adaptive method across at least two non-symbolic task families with joint fairness and functional causality.

R2-StableTaskFamilyLimitedAdvantage:
  stable MNIST-like / symbolic / geometry-sensitive success, broad non-symbolic not enough.

R3-StableGeometryOnlyAdvantage:
  functional geometry robustly improves but task accuracy not broadly better.

R4-ConfigSensitiveSuccess:
  fixed recipe can pass but adaptive/stability criteria fail.

R5-TimingProtocolSensitiveSuccess:
  high-rep or one protocol pass but robust timing gate fails.

R6-ForwardFairButTrainingComputeOpen:
  forward FLOPs fair but training compute fairness not closed.

R7-ContinualUnbalanced:
  forgetting improves but task balance fails.

R8-NegativeExternalResult:
  DG-KAN loses to MLP in most external tasks.

R9-ReproductionFail:
  v8.5 accepted route cannot fresh reproduce.

R10-CodeOrContractFail:
  adapter / counter / no-teacher / no-loss / no-fake contract fails.
```

#### route_decision.json 必须记录

```text
route
v85_reproduction_pass
adaptive_controller_pass
no_manual_tuning_pass
robust_timing_pass
training_compute_fair_pass
multi_task_fair_pass_count
functional_causality_pass_count
continual_balance_pass
robustness_pass
geometry_generalization_pass
best_candidate
best_task_family
boundary_label
primary_blocker
next_required_implementation
success_v87_stability
success_v87_external_formal
success_v87_broad_strong
```

---

## 15. Required artifacts

v8.7 必须落盘：

```text
run_manifest.json
v85_fresh_reproduction_stability.csv
fixed_hyperparam_sensitivity_grid.csv
adaptive_functional_one_step.csv
adaptive_functional_multistep.csv
auto_architecture_selection.csv
robust_timing_protocols.csv
training_compute_counter.csv
kanbefair_multitask_transfer.csv
joint_fair_envelope.csv
functional_causality_multitask.csv
continual_balance_multisplit.csv
robustness_perturbation.csv
functional_mechanism_attribution.csv
negative_boundary_audit.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_reproduction_fail
F2_fixed_recipe_instability
F3_adaptive_controller_fail
F4_auto_arch_selection_fail
F5_robust_timing_fail
F6_training_compute_fair_fail
F7_multitask_external_fail
F8_functional_causality_fail
F9_continual_unbalanced
F10_geometry_no_downstream_gain
F11_teacher_or_loss_violation
F12_fake_or_proxy_violation
F13_artifact_missing
F14_counter_scope_mismatch
```

---

## 16. 第一轮执行顺序

### Step 1：复现与不稳定性审计

先执行 P0。不要直接改方法。先把 v8.5 accepted route 的 seed/rerun/protocol variance 量化清楚。

### Step 2：fixed sensitivity map

执行 P1。目标不是挑最好，而是证明 fixed recipe 的不稳定维度，为 adaptive controller 提供依据。

### Step 3：实现 adaptive functional controller

执行 P2/P3。只有 one-step 与 multistep 都通过，才进入正式外部任务。

### Step 4：automatic architecture selection

执行 P4。停止手工 hidden 选择，让公平 envelope 选择 architecture。

### Step 5：robust timing / training compute

执行 P5/P6。解决 high-rep vs official-style timing 与 forward-only FLOPs 的公平性问题。

### Step 6：external multi-task validation

执行 P7/P9。验证 task-family boundary 和 robustness。

### Step 7：continual balance

执行 P8。把 P11 的 anti-forgetting result 扩成 balanced continual formalization。

### Step 8：mechanism / boundary route

执行 P10/P11/P12，输出成功、边界或反证。

---

## 17. 停止条件

### 17.1 成功停止

Stable broad success：

```text
v85 reproduction pass
adaptive controller pass
auto architecture selection pass
robust timing pass
training compute fair pass
at least two non-symbolic task families joint fair pass
functional causality pass in majority tasks
continual balance pass
```

Stable task-family success：

```text
v85 reproduction pass
adaptive controller pass
MNIST-like / symbolic / geometry-sensitive tasks pass
but broad non-symbolic tasks fail
```

### 17.2 失败停止

```text
1. v8.5 accepted route cannot reproduce；
2. fixed recipe is highly unstable and adaptive controller cannot reduce variance；
3. adaptive controller fails one-step descent gate；
4. auto hidden selection cannot match hidden28 within 0.2%；
5. robust timing Q90 step ratio > 1.50；
6. training compute fairness cannot be established；
7. DG-KAN loses to MLP by > 1% on most non-symbolic tasks；
8. functional causality disappears outside MNIST；
9. continual balance fails across splits；
10. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 18. 最终解释规则

### Case A：adaptive method broad pass

可以声明：

```text
DG-KAN + adaptive functional update establishes stable external fair advantage on tested task families.
```

### Case B：fixed recipe pass, adaptive fail

必须声明：

```text
Current method remains configuration-sensitive; v8.5 success is real but not yet a robust method.
```

### Case C：MNIST-like / symbolic pass only

必须声明：

```text
DG-KAN functional advantage is task-family-limited.
```

### Case D：geometry improves but downstream does not

必须声明：

```text
Functional update gives geometry improvement, but not broad generalization advantage.
```

### Case E：timing unstable

必须声明：

```text
Current system success is timing-protocol-sensitive and requires kernel/timing formalization.
```

### Case F：external negative

必须声明：

```text
Internal and MNIST external success do not generalize to broader KANbeFair tasks.
```

---

## 19. 最终建议

v8.7 的一句话策略是：

$$
\boxed{
\text{把当前“可调参达标的成功点”升级成“自适应、稳健、跨任务公平的方法”。}
}
$$

现在不应该继续：

```text
只在 MNIST full protocol 上找更好 stride；
手工调整 hidden；
手工调整 alpha / role budget；
回 teacher；
改 loss；
做 optimizer sweep；
把 single-route strong success 直接当最终结论。
```

现在应该做：

```text
1. 量化不稳定性；
2. 建 adaptive functional event controller；
3. 自动选择 architecture；
4. 用 robust timing gate 替代单 protocol gate；
5. 补 training compute fairness；
6. 做 multi-task external validation；
7. 做 continual balance；
8. 输出边界清楚的 route。
```

最终要证明的不是：

$$
\text{某个 hidden28 + stride128 可以过。}
$$

而是：

$$
\boxed{
\text{DG-KAN 的 functional architecture method 在不靠人工调参的情况下稳定优于 MLP 或给出清楚任务边界。}
}
