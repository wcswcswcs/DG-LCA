# DG-KAN v4.6 实验计划：Functional Learning Dynamics，重新设计 PureKAN functional optimizer

> 版本：v4.6 draft  
> 日期：2026-05-03  
> 目标：在 v4.5 AFO 失败之后，不再继续小修 Sobolev、warmup、trust radius 或固定 depth-wise 配置，而是重新设计 PureKAN 的 functional optimizer。  
> 公式格式：Typora 友好，全文只使用 `$...$` 和 `$$...$$`。

---

## 0. 一句话结论

v4.5 的结果说明，当前 functional update 已经不再是“方向不下降”的早期问题。`FCAdam-dataSob` 已经能在 20-step horizon 上产生非常强的 holdout descent，但它的函数空间轨迹、rank / margin 形成和 geometry 预算无法同时满足。因此，PureKAN functional optimizer 的核心瓶颈已经变成：

$$
\boxed{
\text{local descent exists, but functional learning dynamics are wrong.}
}
$$

下一阶段不应该继续寻找更好的静态 preconditioner：

$$
\Delta a=-\eta M^{-1}g.
$$

而应该把 PureKAN functional optimizer 重新定义为一个带状态的深度网络优化系统：

$$
\boxed{
\text{task-learning dynamics}
+
\text{role-wise temporal state}
+
\text{representation-phase control}
+
\text{delayed geometry consolidation}
+
\text{Lyapunov / restart safeguard}
}
$$

本计划将这个方向命名为 **FLD: Functional Learning Dynamics**。

---

## 1. 为什么 v4.5 改变了问题定义

### 1.1 P0 已经排除实现问题

v4.5 的 P0 显示 function-coordinate whitening、roundtrip、rollback 都是数值正确的，并且 strict PureKAN 仍然保持：

```text
nonKAN params = 0
functional coverage = 1.0
roundtrip error = 0
rollback error = 0
```

所以当前失败不能继续主要归因于：

```text
alpha 没固定
参数没覆盖
u-space whitening 写错
rollback 写错
还有隐藏 non-KAN 参数
```

这些实现层面的借口已经基本排除。

---

### 1.2 AdamW 不是早期 geometry-improving trajectory

v4.5 的 AdamW trajectory forensic 非常关键。AdamW 在前 20 step 中能降低 holdout loss，但它本身并不是一个纯几何改善轨迹。它的早期行为更像：

```text
任务 loss 下降；
effective rank 下降；
phi 稍微上升；
input role 承担大部分更新；
block/output role 更新较小。
```

典型 role update share 是：

```text
input share 约 0.68 - 0.71
block share 约 0.20 - 0.21
output share 约 0.09 - 0.11
```

这说明一个非常重要的事实：

$$
\boxed{
\text{PureKAN 的早期优化必须先学任务和表征，而不是先追求几何下降。}
}
$$

因此，早期要求 functional optimizer 同时满足 Adam-like task descent、rank/margin 形成、并且立刻做到 strict geometry improvement，是不合理的。几何应该是 phased objective，而不是第一步就硬压。

---

### 1.3 FCAdam-dataSob 的正信号是真实的，但不是 solved optimizer

v4.5 中 `FCAdam-dataSob` 在 MNIST、Fashion-MNIST、KMNIST 上都是最强 20-step holdout descent 点之一。它说明：

$$
\boxed{
\text{functional-coordinate adaptive dynamics 可以产生真实任务下降。}
}
$$

但它没有通过 P2，因为：

```text
cos / R2 to AdamW function trajectory 很低；
rank / margin 和 Adam-like alignment 不能同时保持；
phi/A 常高于 1；
没有候选进入 P3。
```

所以 v4.5 的正确解读不是：

```text
FCAdam 失败，换下一个组件。
```

而是：

```text
FCAdam 证明了 task-learning signal 存在；
但当前 coordinate / temporal dynamics 还不能稳定塑造 PureKAN 表征。
```

---

### 1.4 当前 gate 本身也需要重构

v4.5 的 P2 gate 要求候选同时满足：

```text
20-step holdout descent；
AdamW function trajectory cos / R2；
rank / margin retention；
Phase-I phi budget。
```

这个 gate 的优点是严格，缺点是它可能把两个阶段的目标混在了一起。AdamW 自己早期也会降低 rank 并略升 phi。如果我们要求 functional optimizer 早期同时比 AdamW 更几何、更保 rank、更像 AdamW，它可能反而压死了真实的 task-learning trajectory。

因此 v4.6 的 gate 要改成 **phase-aware gate**：

```text
Phase I: task descent + role dynamics + margin formation；
Phase II: representation retention + stabilization；
Phase III: geometry consolidation。
```

---

## 2. 当前 functional update 的深层缺陷

### 2.1 过去的问题：把 smoothness prior 当成 optimizer

最早的 U-FULL 是：

$$
\Delta a=-\eta S^{-1}g,
$$

其中 $S$ 是 Sobolev smoothness metric。

这个 update 很擅长做：

```text
降低 phi_prime；
降低 Jacobian condition；
降低函数曲率；
让 KAN branch 更规整。
```

但它不擅长做：

```text
形成 feature rank；
扩大 class margin；
构造 class boundary；
在 input/block/output 之间形成长期 co-adaptation。
```

所以 Sobolev-only functional update 在 Hybrid-DGKAN residual branch 上有用，因为 hybrid 的 stem/head/LN/AdamW 已经承担了大部分表示学习；但 PureKAN 中所有表示学习都压到 KAN coefficients 上，Sobolev-only 就不够了。

---

### 2.2 TFU / FNG 的问题：局部 task-aware，不是全局学习动力学

TFU / FNG 试图把 update 改成：

$$
\Delta a_l=-\eta_l(G_l+\lambda_l S_l+\rho I)^{-1}g_l.
$$

这个方向是合理的，但目前的 $G_l$ 仍然是 layer-local 的。它不能完整描述：

```text
input KAN 的变化如何影响后面所有 blocks；
block KAN 的变化如何影响 output boundary；
output KAN 的变化如何反过来塑造隐藏层梯度；
多层同时更新时 representation drift 如何累积。
```

这解释了为什么 v4.1 FTF one-step 很强但训练崩，v4.2 BFT 把它变安全但变弱，v4.3/v4.4/v4.5 的 FC/FNG 系列可以产生局部 descent，却没有成为完整 optimizer。

---

### 2.3 FCAdam 的问题：function-coordinate 正确，但 trajectory controller 不正确

FCAdam 把 coefficient 变换到 functional coordinate：

$$
a=L^{-T}u,
$$

然后在 $u$ 上跑 Adam-like dynamics。这个方向比 $S^{-1}g$ 更接近正确，因为它保留了 Adam 的时间尺度。但 v4.5 显示：

```text
FCAdam-dataSob 能强下降；
但 20-step function trajectory 与 AdamW 目标显著偏离；
rank / margin / geometry 不能同时满足。
```

这说明当前 FCAdam 的问题不是 coordinate correctness，而是 **temporal controller** 错了。它没有稳定地控制：

```text
每个 role 的更新份额；
momentum 的方向漂移；
rank 和 margin 的形成速度；
geometry 何时开始收束。
```

---

### 2.4 直接匹配 AdamW 也不是最终答案

AdamW 是强 baseline，但它不是理论最优的 function-space trajectory。v4.5 已经显示 AdamW 早期会让 rank 降低、phi 上升。因此，我们不能把目标写成：

$$
\Delta f_{method}\approx \Delta f_{AdamW}
$$

然后再硬要求 geometry 更好。更合理的是：

$$
\Delta f_{method}
\in
\mathcal E_{AdamW},
$$

其中 $\mathcal E_{AdamW}$ 是一个 **trajectory envelope**，它约束任务下降、role share、margin/rank 区间，而不是逐点复制 AdamW displacement。

也就是说，AdamW 应该是：

```text
teacher of dynamics profile,
not exact target displacement.
```

---

### 2.5 加速梯度报告给出的关键启发

你上传的梯度下降研究报告有三个对当前项目特别重要的启发。

第一，近五年加速优化的核心不是静态 preconditioner，而是 **Lyapunov / restart / adaptive dynamics / noise-robust momentum**。这说明我们不能继续把 optimizer 当作一个矩阵 $M^{-1}$。

第二，深度学习中的成功路线不是纯 NAG，而是 Adan、Win 这类 **Nesterov 化 adaptive optimizer**。它们把加速、动量、自适应尺度和训练 recipe 结合起来，而不是只换一个下降方向。

第三，报告指出一般 PL 类并不存在通用多项式级加速，这提醒我们：PureKAN 不是普通“弱非凸”问题，是否能加速取决于具体结构。我们必须利用 KAN 的 function-space structure，但不能幻想一个 Sobolev inverse 就能普适解决。

因此，v4.6 的思想是：

$$
\boxed{
\text{functional coordinate}
+
\text{adaptive accelerated dynamics}
+
\text{phase-aware Lyapunov controller}
}
$$

---

## 3. v4.6 新方向：FLD, Functional Learning Dynamics

### 3.1 总体形式

v4.6 不再把 update 定义为一个单步公式，而定义为一个带状态的动力系统。

对第 $l$ 个 KAN role 或 layer，维护 functional coordinate $u_l$：

$$
a_l=L_l^{-T}u_l.
$$

其中 $L_l$ 来自一个 functional metric：

$$
M_l=L_lL_l^T.
$$

每步先在 $u$-space 计算梯度：

$$
g^u_l=L_l^{-1}g^a_l.
$$

然后用 adaptive accelerated dynamics 更新：

$$
m_{l,t}=\beta_1m_{l,t-1}+(1-\beta_1)g^u_{l,t},
$$

$$
v_{l,t}=\beta_2v_{l,t-1}+(1-\beta_2)(g^u_{l,t})^2.
$$

候选方向为：

$$
d^u_{l,t}=\frac{m_{l,t}}{\sqrt{v_{l,t}}+\epsilon}.
$$

但这个方向不直接写回。它还要经过：

```text
role-wise scale controller；
trajectory envelope controller；
rank / margin controller；
phased geometry controller；
Lyapunov restart controller。
```

最后写回 coefficient：

$$
\Delta a_l=L_l^{-T}\Delta u_l.
$$

---

### 3.2 不再使用“硬匹配 AdamW displacement”作为主要目标

v4.6 只把 AdamW 用作 **profile teacher**。需要从 AdamW forensic 中提取：

```text
holdout descent envelope；
role update share envelope；
rank ratio envelope；
margin ratio envelope；
phi ratio envelope；
accepted step norm envelope。
```

定义 AdamW 的 early phase envelope：

$$
\mathcal E_t=
\left\{
\Delta L_t,
\ r_{rank,t},
\ r_{margin,t},
\ r_{phi,t},
\ s_{input,t},
\ s_{block,t},
\ s_{output,t}
\right\}.
$$

候选 optimizer 不需要逐点满足：

$$
\cos(\Delta f,\Delta f_{AdamW})>c.
$$

而要满足：

$$
\Delta L_{holdout}\geq \tau_L \Delta L_{AdamW},
$$

$$
r_{rank}\geq r_{rank}^{min},
$$

$$
r_{margin}\geq r_{margin}^{min},
$$

$$
r_{phi}\leq r_{phi}^{budget}(t).
$$

这里 $r_{phi}^{budget}(t)$ 是 phased 的。

---

### 3.3 三阶段 geometry policy

v4.6 将训练分为三阶段。

#### Phase I: Representation formation

目标：

```text
任务下降；
margin 形成；
role update share 接近 AdamW profile；
允许 phi 上升。
```

约束：

$$
r_{phi}\leq 1.25,
$$

$$
r_{rank}\geq 0.65,
$$

$$
r_{margin}\geq 0.80.
$$

#### Phase II: Stabilization

目标：

```text
继续 task descent；
保持 rank / margin；
开始压 geometry。
```

约束：

$$
r_{phi}\leq 1.10,
$$

$$
r_{rank}\geq 0.80,
$$

$$
r_{margin}\geq 0.90.
$$

#### Phase III: Geometry consolidation

目标：

```text
保持 accuracy；
降低 phi/J；
改善 ECE；
避免 late overfit。
```

约束：

$$
r_{phi}\leq 0.95,
$$

或：

$$
\phi'_{p95,method}<\phi'_{p95,AdamW}.
$$

---

### 3.4 Role-wise update share controller

v4.5 的 AdamW forensic 表明早期 input role 的更新份额最高，block 次之，output 最小。PureKAN functional optimizer 必须显式控制 role share。

定义每步 role update share：

$$
s_l(t)=\frac{\|\Delta u_l(t)\|}{\sum_j\|\Delta u_j(t)\|+\epsilon}.
$$

设定 early target：

```text
input share target: 0.60 - 0.75
block share target: 0.15 - 0.30
output share target: 0.05 - 0.15
```

如果某个 role 偏离 target，就调整 role LR multiplier：

$$
\eta_l(t+1)=\eta_l(t)\exp\left(\gamma_s(s_l^\star-s_l(t))\right).
$$

这不是为了模仿 AdamW 的每个方向，而是为了保持 PureKAN 的表示学习节奏。

---

### 3.5 Lyapunov energy 与 restart

基于加速梯度报告中 Lyapunov / restart 的启发，v4.6 引入一个可记录的训练能量：

$$
\mathcal V_t
=
L_{holdout,t}
+
\lambda_{rank}[r_{rank}^{min}(t)-r_{rank,t}]_+^2
+
\lambda_{margin}[r_{margin}^{min}(t)-r_{margin,t}]_+^2
+
\lambda_{phi}[r_{phi,t}-r_{phi}^{budget}(t)]_+^2
+
\lambda_{step}\|\Delta u_t\|^2.
$$

如果出现：

$$
\mathcal V_t>\mathcal V_{t-1}+\epsilon_V,
$$

或者：

$$
\langle m_t,g_t\rangle<0,
$$

则触发 restart：

```text
清空或衰减 momentum；
降低 eta；
提高 geometry damping；
重置 role share controller 的积分项。
```

这比单纯 trust clipping 更有意义，因为它直接关心 loss、rank、margin、geometry 的联合状态。

---

## 4. v4.6 候选方法

### 4.1 FLD-AdamCoord

这是最小可行版本。

```text
functional coordinate whitening；
AdamW-style m/v in u-space；
role-wise LR multiplier；
three-phase geometry budget；
no Nesterov；
no Lyapunov restart。
```

目的：确认 role-share control + phased geometry 是否已经足够。

---

### 4.2 FLD-Nesterov

加入 lookahead：

$$
\tilde u_t=u_t+\beta m_{t-1}.
$$

在 $\tilde u_t$ 处计算梯度或近似梯度，然后更新。由于真实二次 forward 成本高，可以先用 one-extra-forward 版本做 audit，再做 cheap approximation。

目的：验证 functional-coordinate Nesterov 是否优于普通 FCAdam。

---

### 4.3 FLD-AdanLite

借鉴 Adan 的思想，显式追踪梯度差：

$$
d_t=g_t-g_{t-1}.
$$

维护：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
n_t=\beta_2n_{t-1}+(1-\beta_2)d_t,
$$

$$
v_t=\beta_3v_{t-1}+(1-\beta_3)(g_t+(1-\beta_2)d_t)^2.
$$

候选方向：

$$
p_t=\frac{m_t+(1-\beta_2)n_t}{\sqrt{v_t}+\epsilon}.
$$

所有变量都在 functional coordinate $u$ 中。

目的：验证梯度变化项是否能修复 20-step trajectory drift。

---

### 4.4 FLD-WinLite

借鉴 Win / weight-decay-integrated Nesterov 的思想，不直接把 geometry penalty 当作 update 后处理，而把 functional decay / Sobolev penalty 融进 lookahead step。

形式：

$$
u_{t+1/2}=u_t-\eta \lambda_S S_u u_t,
$$

$$
\tilde u_t=u_{t+1/2}+\beta m_t,
$$

$$
u_{t+1}=\tilde u_t-\eta p_t.
$$

目的：验证 geometry regularization 是否应该通过 optimizer dynamics 耦合，而不是训练后期硬拉回。

---

### 4.5 FLD-LyapunovRestart

在 `FLD-AdanLite` 或 `FLD-Nesterov` 基础上加入 Lyapunov restart。

```text
如果 holdout loss 上升：restart；
如果 rank collapse：restart；
如果 margin collapse：restart；
如果 phi 超预算：geometry restart；
如果 momentum-gradient cosine < 0：momentum restart。
```

目的：验证 restart / Lyapunov controller 是否能避免 20-step drift。

---

### 4.6 FLD-TeacherEnvelope

将 AdamW forensic 转化为 soft envelope loss，而不是硬 cos/R2 gate。

额外 controller loss：

$$
\mathcal R_{env}
=
\alpha_s\sum_l(s_l-s_l^\star)^2
+
\alpha_r[r_{rank}^{min}-r_{rank}]_+^2
+
\alpha_m[r_{margin}^{min}-r_{margin}]_+^2
+
\alpha_\phi[r_{phi}-r_{phi}^{budget}]_+^2.
$$

这个 penalty 不进入 autograd loss，而用于调整 optimizer state 和 step acceptance。

目的：验证 “AdamW as profile teacher” 是否优于 “AdamW as exact displacement target”。

---

## 5. 实验总流程

本轮不直接做 10-seed。所有方法必须先通过 trajectory-dynamics 诊断。

```text
P0: implementation invariants and coordinate/state smoke
P1: AdamW trajectory envelope construction
P2: one-step and 20-step dynamics audit
P3: 100-step trajectory audit
P4: phase-controller ablation
P5: 3-seed short full-budget selection
P6: 5-seed confirm
P7: 10-seed final confirm
P8: failure diagnosis and theory update
```

---

## 6. P0：实现与不变量检查

### 6.1 目标

确认所有 FLD 方法仍然是 strict PureKAN functional optimizer，不引入 non-KAN 参数，也不复用普通 AdamW 参数更新。

### 6.2 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### 6.3 模型

```text
model = PureKAN
hidden_dim = 64
basis_count = 16
深度 = 2 和 4 都做 smoke
alphaFixed1
FixedNorm
nonKAN params = 0
```

### 6.4 方法

```text
PureKAN-AdamW
D0-allFullSobolev
D6-allTaskAware
FCAdam-dataSob
FLD-AdamCoord
FLD-Nesterov
FLD-AdanLite
FLD-WinLite
FLD-LyapunovRestart
FLD-TeacherEnvelope
```

### 6.5 必须记录

```text
implementation/nonKAN_param_count
implementation/functional_coverage
implementation/u_roundtrip_error
implementation/rollback_error
implementation/state_m_finite
implementation/state_v_finite
implementation/restart_state_finite
implementation/role_lr_multiplier_finite
implementation/phase_state
implementation/geometry_budget
implementation/NaN_count
```

### 6.6 通过条件

$$
\text{nonKAN params}=0
$$

$$
\text{functional coverage}=1.0
$$

$$
\text{roundtrip error}<10^{-8}
$$

$$
\text{NaN count}=0
$$

---

## 7. P1：AdamW trajectory envelope construction

### 7.1 目标

重新定义 AdamW 作为 **profile teacher**，而不是 exact target。

### 7.2 设置

```text
methods = PureKAN-AdamW
seeds = 0,1,2
steps recorded = 1, 5, 20, 50, 100, full epoch
```

### 7.3 记录指标

#### Task trajectory

```text
loss/train
loss/holdout
loss/val
holdout_descent_1
holdout_descent_5
holdout_descent_20
holdout_descent_50
holdout_descent_100
val_loss_auc_partial
```

#### Role update profile

```text
role/input_update_norm
role/block_update_norm
role/output_update_norm
role/input_share
role/block_share
role/output_share
role/share_entropy
role/share_ema
```

#### Representation profile

```text
repr/effective_rank_input
repr/effective_rank_block
repr/effective_rank_output
repr/class_centroid_separation
repr/class_within_scatter
repr/fisher_ratio
repr/logit_margin_mean
repr/logit_margin_p10
repr/logit_margin_p50
repr/logit_entropy
```

#### Geometry profile

```text
geometry/phi_prime_p95
geometry/phi_prime_max
geometry/jacobian_condition
geometry/curvature_energy
geometry/sobolev_norm
```

#### Optimizer state

```text
adam/m_norm_by_role
adam/v_norm_by_role
adam/update_over_param_by_role
adam/m_grad_cos_by_role
```

### 7.4 输出

产生一个 `adamw_trajectory_envelope.json`，至少包含：

```text
role_share_target_by_phase
rank_budget_by_phase
margin_budget_by_phase
phi_budget_by_phase
holdout_descent_target_by_phase
restart_reference_events
```

### 7.5 可视化

```text
figures/p1_adamw_loss_rank_phi_phase.png
figures/p1_adamw_role_share_stack.png
figures/p1_adamw_margin_rank_trajectory.png
figures/p1_adamw_phi_vs_holdout_descent.png
figures/p1_adamw_update_state_by_role.png
```

---

## 8. P2：one-step and 20-step dynamics audit

### 8.1 目标

先验证候选是否能保持 20-step learning dynamics。不要直接进入 full training。

### 8.2 方法

```text
PureKAN-AdamW
D0-allFullSobolev
D6-allTaskAware
FCAdam-dataSob
FLD-AdamCoord
FLD-Nesterov
FLD-AdanLite
FLD-WinLite
FLD-LyapunovRestart
FLD-TeacherEnvelope
```

### 8.3 记录指标

#### Descent

```text
dynamics/train_descent_1
dynamics/holdout_descent_1
dynamics/train_descent_5
dynamics/holdout_descent_5
dynamics/train_descent_20
dynamics/holdout_descent_20
dynamics/bad_step_rate
dynamics/negative_holdout_steps
```

#### AdamW envelope matching

```text
envelope/role_share_l2_error
envelope/rank_budget_violation
envelope/margin_budget_violation
envelope/phi_budget_violation
envelope/trajectory_energy
```

#### Function displacement diagnostics

```text
traj/cos_with_adam_function
traj/R2_with_adam_function
traj/cos_with_adam_coeff
traj/R2_with_adam_coeff
traj/logit_delta_cos
traj/activation_delta_cos_input
traj/activation_delta_cos_block
traj/activation_delta_cos_output
```

注意：cos/R2 只作为诊断，不作为单独硬 gate。

#### Optimizer state

```text
state/m_norm
state/v_norm
state/n_norm, for AdanLite
state/update_norm
state/update_over_param
state/accepted_eta
state/restart_count
state/restart_reason
state/role_lr_multiplier
```

### 8.4 通过条件

P2 不要求 geometry 优于 AdamW，只要求不超出 early budget。

候选必须满足：

$$
\Delta L_{holdout,20}
\geq
0.8\Delta L_{holdout,20}^{AdamW}
$$

或：

$$
\Delta L_{holdout,20}
>
\Delta L_{holdout,20}^{D6}+0.1.
$$

并且：

$$
r_{phi,20}<1.25,
$$

$$
r_{rank,20}>0.65,
$$

$$
r_{margin,20}>0.80,
$$

$$
\text{bad step rate}<0.05.
$$

### 8.5 可视化

```text
figures/p2_holdout_descent_20_bar.png
figures/p2_role_share_error_heatmap.png
figures/p2_rank_margin_phi_scatter.png
figures/p2_trajectory_energy_by_method.png
figures/p2_restart_reason_stack.png
figures/p2_momentum_cos_trace.png
```

---

## 9. P3：100-step trajectory audit

### 9.1 目标

v4.5 的核心失败是 20-step temporal mismatch。v4.6 必须直接检查 100-step trajectory。

### 9.2 方法

只进入 P2 通过的候选。若没有候选通过，则强制保留以下方法做 diagnostic：

```text
FCAdam-dataSob
FLD-AdanLite
FLD-LyapunovRestart
```

### 9.3 记录指标

```text
loss/train_curve_100
loss/holdout_curve_100
repr/rank_curve_100
repr/margin_curve_100
geometry/phi_curve_100
geometry/J_curve_100
role/share_curve_100
state/momentum_norm_curve_100
state/restart_markers
state/eta_curve_100
state/lyapunov_curve_100
```

### 9.4 通过条件

候选必须满足：

$$
\Delta L_{holdout,100}
\geq
0.75\Delta L_{holdout,100}^{AdamW}
$$

并且：

$$
\text{trajectory energy}_{100}<\text{D6 trajectory energy}_{100}.
$$

同时：

$$
r_{phi,100}<1.15,
$$

$$
r_{rank,100}>0.75,
$$

$$
r_{margin,100}>0.85.
$$

### 9.5 可视化

```text
figures/p3_100step_loss_rank_phi.png
figures/p3_lyapunov_with_restart_markers.png
figures/p3_role_share_vs_adamw.png
figures/p3_energy_components.png
figures/p3_method_phase_transition.png
```

---

## 10. P4：phase-controller ablation

### 10.1 目标

验证 v4.6 的关键思想：几何不应该早期硬压，而应该 phased consolidation。

### 10.2 方法

对 P3 最好候选做 ablation：

```text
base candidate
no role controller
no Lyapunov restart
no phased geometry
strict early geometry
no geometry until late
fixed role share
AdamW role share envelope
```

### 10.3 记录指标

```text
phase/phase_enter_step
phase/phase_exit_step
phase/geometry_budget
phase/phi_violation_count
phase/rank_violation_count
phase/margin_violation_count
phase/restart_count
phase/restart_reason
phase/controller_adjustment_norm
```

### 10.4 判定

如果 `strict early geometry` 任务下降差，而 `phased geometry` 任务下降好且最终 geometry 能收回来，则确认：

$$
\boxed{
\text{PureKAN functional optimizer needs delayed geometry consolidation.}
}
$$

如果 `no Lyapunov restart` 出现 100-step drift，而 `with restart` 稳定，则确认 restart 是必要组件。

---

## 11. P5：3-seed short full-budget selection

### 11.1 目标

只对 P3/P4 通过的候选进行小规模训练，避免浪费 5/10 seed。

### 11.2 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### 11.3 对照方法

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
D0-allFullSobolev
D6-allTaskAware
FCAdam-dataSob
best FLD candidate 1
best FLD candidate 2
```

### 11.4 记录指标

#### Task

```text
task/train_loss_curve
task/val_loss_curve
task/test_acc
task/best_val_acc
task/val_loss_auc
task/val_acc_auc
task/acc_gap_vs_purekan_adamw
task/acc_gap_vs_mlp_adamw
task/acc_gap_vs_hybrid_ufull
```

#### Representation

```text
repr/effective_rank_by_epoch
repr/rank_ratio_vs_adamw
repr/class_centroid_separation
repr/fisher_ratio
repr/logit_margin_mean
repr/logit_margin_p10
repr/logit_entropy
repr/feature_norm
```

#### Geometry

```text
geometry/phi_prime_p95
geometry/phi_prime_max
geometry/jacobian_condition
geometry/curvature_energy
geometry/sobolev_norm
geometry/final_phi_ratio_vs_adamw
geometry/final_J_ratio_vs_adamw
```

#### Optimizer dynamics

```text
state/restart_count
state/eta_mean
state/eta_p95
state/m_norm_by_role
state/v_norm_by_role
state/update_share_by_role
state/role_lr_multiplier
state/trajectory_energy_auc
state/phi_budget_violation_auc
```

### 11.5 通过条件

进入 P6 的候选必须：

$$
\operatorname{Acc}_{method}
\geq
\operatorname{Acc}_{PureKAN-AdamW}-1\%
$$

在 MNIST/Fashion 上，并且：

$$
\operatorname{Acc}_{method}
\geq
\operatorname{Acc}_{PureKAN-AdamW}-2\%
$$

在 KMNIST 上。

还必须满足：

$$
\operatorname{AUC}_{val,method}
<
\operatorname{AUC}_{val,D6},
$$

$$
\phi'_{p95,method}
<
1.05\phi'_{p95,AdamW},
$$

以及：

$$
\text{restart rate}<0.25.
$$

---

## 12. P6：5-seed confirm

### 12.1 目标

确认 P5 候选是否稳定，而不是单 seed 或 3-seed 偶然。

### 12.2 方法

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
best FLD candidate
second FLD candidate, optional
```

### 12.3 Seeds

```text
seeds = 0,1,2,3,4
```

### 12.4 通过条件

候选必须至少满足 Pareto pass：

```text
accuracy gap vs PureKAN-AdamW <= 1% on MNIST/Fashion
accuracy gap vs PureKAN-AdamW <= 2% on KMNIST
val-loss AUC better than D6 and D0
final phi/J better than AdamW or not worse than 1.05x AdamW
ECE not worse than AdamW
```

如果想 claim strong PureKAN functional optimizer，则必须：

$$
\operatorname{Acc}_{method}
\geq
\operatorname{Acc}_{PureKAN-AdamW}
$$

在至少两个数据集上，并且第三个数据集 gap 小于 1%。

---

## 13. P7：10-seed final confirm

### 13.1 触发条件

只有 P6 通过才运行。

### 13.2 Seeds

```text
seeds = 0..9
```

### 13.3 最终 claim 条件

如果候选要成为 PureKAN 默认 functional optimizer，必须满足：

$$
\operatorname{Acc}_{FLD}
\geq
\max(
\operatorname{Acc}_{PureKAN-AdamW},
\operatorname{Acc}_{MLP-AdamW},
\operatorname{Acc}_{Hybrid-DGKAN-UFULL}
)-\epsilon
$$

其中：

$$
\epsilon=0.005
$$

同时必须满足：

```text
val-loss AUC better than PureKAN-AdamW or within 2% while geometry better；
phi_prime_p95 lower than AdamW；
Jacobian condition lower than AdamW；
ECE lower than AdamW；
role update share stable；
restart count finite and interpretable。
```

如果做不到，则不能 claim solved，只能 claim：

```text
FLD improves functional optimizer dynamics but does not yet surpass AdamW.
```

---

## 14. P8：failure diagnosis

如果 P7 不通过，必须输出一个明确失败类型，而不是继续调参。

### 14.1 失败类型

```text
F1: role share mismatch
F2: trajectory energy high
F3: rank formation failure
F4: margin formation failure
F5: early geometry overconstraint
F6: late geometry cannot consolidate
F7: momentum drift
F8: restart over-triggered
F9: basis occupancy failure
F10: compute overhead unacceptable
```

### 14.2 必须输出表

```text
failure_table.csv
failure_by_dataset.csv
failure_by_role.csv
failure_by_phase.csv
failure_by_method.csv
role_share_trace.csv
trajectory_energy_trace.csv
rank_margin_phi_trace.csv
restart_trace.csv
optimizer_state_trace.csv
```

### 14.3 必须画图

```text
figures/failure_taxonomy_heatmap.png
figures/role_share_mismatch.png
figures/trajectory_energy_vs_accuracy.png
figures/rank_margin_phi_phase_plot.png
figures/restart_timeline.png
figures/optimizer_state_norms.png
figures/basis_occupancy_failure.png
```

---

## 15. 本轮必须避免的误区

### 15.1 不再继续调 Sobolev alpha / beta 作为主线

这些已经不是根因。它们只能在 P4 phase-controller ablation 中作为辅助。

### 15.2 不再用 strict early geometry gate 杀掉任务学习

早期允许 phi 上升，只要不超过 phase budget。

### 15.3 不再把 AdamW displacement 当作唯一目标

AdamW 是 profile teacher，不是 exact target。

### 15.4 不再只看 holdout descent

v4.5 已经证明 holdout descent 可以很强但 trajectory mismatch。必须同时看：

```text
rank
margin
role share
phi
Lyapunov energy
restart behavior
```

### 15.5 不再直接跑 5/10 seed

如果 P2/P3 的 trajectory dynamics 不过，继续扩 seed 只是浪费。

---

## 16. 预期结论模板

### 情况 A：FLD 成功

```text
PureKAN functional optimization is solved by Functional Learning Dynamics. The key was not a better Sobolev preconditioner, but a stateful optimizer that combines functional-coordinate adaptive dynamics, role-wise update control, phased geometry consolidation, and Lyapunov restart. FLD matches or exceeds PureKAN-AdamW while improving geometry and calibration.
```

### 情况 B：FLD 部分成功

```text
FLD substantially improves PureKAN functional training dynamics and closes much of the gap to PureKAN-AdamW, but does not yet exceed all baselines. The remaining bottleneck is representation formation on KMNIST, especially role-share mismatch or margin formation.
```

### 情况 C：FLD 失败

```text
Even with functional-coordinate adaptive dynamics, phased geometry, and Lyapunov restart, PureKAN functional training does not match AdamW. This suggests that current functional-coordinate optimizers still lack the implicit bias or noise-scale dynamics required for full-network PureKAN training. Functional update should remain a hybrid KAN-branch optimizer until a stronger theory is found.
```

---

## 17. 总结

v4.5 后，我们不能再说：

```text
只要调好 functional update 的 metric，PureKAN 就能超过 AdamW。
```

更准确的说法是：

$$
\boxed{
\text{PureKAN needs functional learning dynamics, not just functional preconditioning.}
}
$$

v4.6 的目标是验证：

$$
\boxed{
\text{functional-coordinate adaptive acceleration}
+
\text{role-wise representation control}
+
\text{Lyapunov restart}
+
\text{phased geometry consolidation}
}
$$

是否足以把 PureKAN functional optimizer 从局部 descent proposal 推进成真正可以训练完整网络的 optimizer。
