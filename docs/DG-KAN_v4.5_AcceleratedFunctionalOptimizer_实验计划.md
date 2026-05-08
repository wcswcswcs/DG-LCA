# DG-KAN v4.5：从 Functional Update 到 Accelerated Functional Optimizer 的重新设计与实验计划

## 0. 本计划的定位

v4.5 不再把问题理解为“继续调 Sobolev 系数、warmup、trust radius 或 shallow/deep 分工”。这些方向已经多轮验证过，最多能带来局部改善，不能解决 PureKAN functional optimizer 的根问题。

本计划的核心判断是：

$$
\boxed{
\text{当前 functional update 不是一个完整 optimizer，而只是一个局部 proposal / preconditioner。}
}
$$

PureKAN 需要的是一个真正的深度网络优化器：它必须同时具备任务学习动力学、函数空间几何约束、长期时间尺度、自适应步长、restart / rejection 机制，以及 rank / margin 形成能力。

因此 v4.5 的目标不是继续找更好的 $S^{-1}g$，而是设计：

$$
\boxed{
\text{AFO: Accelerated Functional Optimizer}
}
$$

AFO 的基本思想是：

$$
\text{Adam-like task trajectory}
+
\text{functional coordinate geometry}
+
\text{restart / Lyapunov controller}
+
\text{rank-margin preservation}
+
\text{late-stage geometry consolidation}.
$$

这版计划必须回答一个最根本的问题：

$$
\boxed{
\text{PureKAN 的 functional optimizer 是否能在不依赖非 KAN 参数的情况下，超过 PureKAN-AdamW、Hybrid-DGKAN-UFULL 和 MLP-AdamW？}
}
$$

如果不能，则必须承认当前 functional update 的正确定位只是 hybrid branch optimizer，而不是 PureKAN full-network optimizer。

---

## 1. v4.4 结果的深层复盘

### 1.1 实现问题基本排除

v4.4 的 P0 smoke 说明当前实验已经不是 hook / coverage / rollback 问题。所有 successful rows 保持：

```text
strict PureKAN nonKAN params = 0
coefficient coverage = 1.0
fixed alpha
exact rollback
errors = 0
```

这意味着如果 PureKAN functional optimizer 失败，不能再主要归因于：

```text
input/output KAN 没被更新
alphaFixed1 没生效
还有隐藏非 KAN 参数
rollback / temporary apply 实现错误
```

当前问题已经进入 optimizer 设计层。

---

### 1.2 FCAdam / AB-RBF-FCAdam 给出了最重要的正信号

v4.4 P1 中，FCAdam 和 AB-RBF-FCAdam 能明显接近 AdamW 的 function-space displacement。

典型信号：

```text
AB-RBF-FCAdam-H1-low:
  cos_f = 0.9746
  R2 = 0.9345

FCAdam-L2:
  cos_f = 0.8949
  R2 = 0.7862

FCAdam-H1-low:
  cos_f = 0.8620
  R2 = 0.7100

FCAdam-dataSob:
  cos_f = 0.8752
  R2 = 0.7278
```

这说明：

$$
\boxed{
\text{functional-coordinate Adam 是当前最接近 AdamW 学习动力学的方向。}
}
$$

但这些方法仍然没有通过 P1 / P2 gate，因为 rank、margin、geometry 无法同时满足。

这不是无效信号，而是说明 FCAdam 的底层方向比旧 Sobolev / TFU / FNG 更对，但还缺少 trajectory-level controller。

---

### 1.3 P2 的真正失败点不是 loss descent，而是 representation retention

v4.4 P2 中很多方法有正的 5-step holdout descent，尤其 `FCAdam-dataSob`：

```text
Fashion-MNIST:
  hold5 = 0.6420
  R2 = 0.2358
  rank/A = 0.6949
  margin/A = 1.1497
  phi/A = 1.0642

KMNIST:
  hold5 = 0.2997
  R2 = 0.4761
  rank/A = 0.9011
  margin/A = 1.2821
  phi/A = 1.0820

MNIST:
  hold5 = 0.2588
  R2 = 0.4826
  rank/A = 1.0451
  margin/A = 1.2523
  phi/A = 1.1053
```

这说明它不是完全不会下降，而是没有同时满足：

```text
1. Adam-like function trajectory
2. representation rank retention
3. margin formation
4. geometry improvement
```

v4.4 的 failure diagnosis 也指出，persistent blocker 是 geometry gate + representation retention：候选能局部学习，但没有同时组合 Adam-like trajectory、feature formation 和 strict geometry improvement。

因此下一步不应该问：

```text
哪个 update 能让 loss 降一点？
```

而应该问：

```text
哪个 optimizer 能形成 AdamW 类似的表示学习轨迹，同时逐步收紧函数几何？
```

---

### 1.4 GFK 的失败说明 global kernel step 不能太小

GFK 非常稳定，但几乎没有 function displacement：

```text
GFK-output-only / block-output / all-lowrank:
  R2 ≈ 0.0006 - 0.0009
  hold5 ≈ 0.003 - 0.008
  margin/A ≈ 0.01 - 0.02
```

这说明当前 GFK 不是危险，而是太弱。它给了一个重要边界：

$$
\boxed{
\text{全局 functional kernel 如果只做小型低秩 diagnostic，不足以承担训练。}
}
$$

后续如果继续 global kernel，必须扩大 Krylov / CG step、target dimension 或用 AdamW trajectory 作为 teacher，而不是继续 output-only tiny displacement。

---

### 1.5 SFD 的失败说明“拟合 AdamW displacement”不能只做单步

SFD 能安全地拟合一部分 AdamW function displacement，但 R2 明显不足：

```text
SFD-direct:
  cos coeff ≈ 0.6115
  R2 ≈ 0.3033 in P1 summary

SFD-prox / residual:
  R2 更低
```

这说明 AdamW displacement 可以作为 teacher，但当前 SFD 的 layer-wise fitting 太弱。它没有捕捉 AdamW 多步轨迹中的 momentum / RMS / noise-averaging 效果。

因此 SFD 不能再是：

$$
\text{fit one AdamW step}
$$

而应该变成：

$$
\text{fit a short AdamW rollout in function space}
$$

也就是把 teacher 从 one-step displacement 改成 $K$-step trajectory target。

---

## 2. 梯度下降研究报告带来的启发

你额外上传的加速梯度下降报告对这里非常有启发。它不是直接告诉我们用哪个现成 optimizer，而是提供了三个重要原则。

### 2.1 加速不是单步 preconditioner，而是动态系统

近五年的加速梯度研究强调 Lyapunov、high-resolution ODE、restart、momentum、参数无关步长和噪声鲁棒性。这说明真正的加速优化器不是一个静态矩阵 $M^{-1}$，而是一个带状态的动态系统。

当前 functional update 最大的问题正是：

$$
\boxed{
\text{它把 optimizer 简化成了静态 function-space preconditioner。}
}
$$

这对 toy function fitting 可以成立，但对 PureKAN 深度网络不够。

---

### 2.2 深度学习里有效的“加速”通常和 adaptive optimizer 结合

报告里提到的 Adan、Win、IRE 等方向都说明，深度学习实践中真正有效的加速，通常不是直接套经典 NAG，而是把 Nesterov / restart / flatness / implicit regularization 与 AdamW-style adaptive dynamics 结合。

这对我们意味着：

$$
\boxed{
\text{PureKAN functional optimizer 不应该试图抛弃 AdamW 的时间尺度，}
}
$$

而应该把 AdamW 的长期学习动力学搬到 functional coordinate 里。

这正是 v4.5 选择 FCAdam / FAdamNAG / restart controller 的原因。

---

### 2.3 一般非凸没有免费加速，必须利用结构

报告里也强调，一般 PL / 非凸问题并不存在普遍免费加速；有效加速依赖强几何结构、restart、variance modeling、flatness 或问题特定结构。

PureKAN 的结构是：

$$
Y_l = \Phi_l(X_l) A_l^\top.
$$

因此真正应该利用的是：

```text
basis feature covariance
edge/channel structure
layer output tangent
class margin dynamics
function-space whitening
Sobolev smoothness budget
```

而不是只用一个 basis-only Sobolev matrix。

---

## 3. 当前 functional update 的根本缺陷

### 3.1 缺陷一：把 smoothness prior 当成 optimizer geometry

旧 U-FULL 本质是：

$$
\Delta a=-\eta S^{-1}g.
$$

其中 $S$ 是 Sobolev metric。这个 $S$ 关心函数平滑、导数、曲率，但不关心 class margin、feature rank、downstream sensitivity。

因此它擅长让几何漂亮，却不擅长形成分类表示。

这解释了多轮实验中反复出现的模式：

```text
phi / J 很好
ECE 有时很好
但 accuracy / margin / rank 不够
```

---

### 3.2 缺陷二：local descent 不等于 representation learning

D6 / FNG / FPA / FCAdam 多次出现 positive local descent，但 P2 / micro-run 不过。原因是 PureKAN 的目标不是只让当前 batch loss 下降，而是形成长期特征：

```text
input KAN 要学特征提取
block KAN 要学组合变换
output KAN 要学分类边界
```

局部下降可能短期有用，但如果破坏 feature rank 或 class separation，长期 accuracy 会输 AdamW。

---

### 3.3 缺陷三：当前 optimizer 没有“能量函数”

我们现在记录了很多指标，但 optimizer 本身没有一个统一 energy 来决定：

```text
该不该加速
该不该 restart
该不该放松 geometry
该不该收紧 Sobolev
该不该保留 Adam-like direction
```

这和加速梯度文献中的 Lyapunov / restart 思想相反。优化器必须有一个可监控的能量。

v4.5 需要引入 functional Lyapunov energy：

$$
\mathcal E_t
=
L_t
+
\lambda_\phi \max(0, \phi_t/\phi_{ref}-r_\phi)^2
+
\lambda_R \max(0, r_R-R_t/R_{ref})^2
+
\lambda_m \max(0, r_m-m_t/m_{ref})^2
+
\lambda_s \|\Delta f_t\|^2.
$$

其中：

```text
L_t = train/holdout loss
phi_t = geometry roughness
R_t = effective rank
m_t = margin statistic
Delta f_t = function displacement
```

---

### 3.4 缺陷四：几何 gate 的时序可能错了

v4.4 P2 gate 要求候选在早期 horizon 就满足：

$$
\phi/A \leq 0.9.
$$

这个条件可能过早。AdamW 的成功很可能先允许函数 roughness 上升，形成有效 feature 和 margin，再通过后期正则或 averaging 收几何。

所以下一版不能要求：

```text
每个早期阶段都比 AdamW 更平滑。
```

应该改成：

```text
early: 保 rank / margin / loss trajectory
mid: 限制 phi 不爆
late: consolidate geometry，使 phi/J 优于 AdamW
```

也就是把 geometry 从 immediate hard gate 改成 phased budget。

---

### 3.5 缺陷五：缺少 restart 和 anti-stall 机制

AdamW 不一定每一步方向都优雅，但它的 momentum / RMS / weight decay / step schedule 给了长期稳定学习动力学。

当前 functional update 即使加入 trust gate，也只是在防坏步；它没有：

```text
stagnation detection
restart
momentum reset
geometry relaxation
rank rescue
step-size recovery
```

这就是为什么 BFT 变安全后也变弱。v4.5 必须加入 restart / acceleration state。

---

## 4. v4.5 的核心新方向：AFO

AFO 的完整名称：

```text
Accelerated Functional Optimizer
```

核心目标：

$$
\boxed{
\text{在 function-whitened coordinate 中保留 AdamW 的任务学习动力学，}
}
$$

同时用 Sobolev 和 rank / margin constraints 做后期几何控制。

---

## 5. AFO 的三个核心组件

### 5.1 Functional-coordinate Adam backbone

对每个 KAN layer，令 Sobolev / data-Sobolev metric 为：

$$
M_l = S_l + \rho I.
$$

做 Cholesky：

$$
M_l = L_l L_l^\top.
$$

定义 whitened coordinate：

$$
a_l = L_l^{-\top}u_l.
$$

在 $u_l$ 上运行 AdamW / AdamW-like update：

$$
g_{u,l}=L_l^{-1}g_{a,l}.
$$

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_{u,t}.
$$

$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_{u,t}^2.
$$

$$
\Delta u_t=-\eta \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

再映射回 coefficient：

$$
\Delta a_t=L_t^{-\top}\Delta u_t.
$$

这和旧 U-FULL 的区别是：

```text
旧 U-FULL:
  每步直接 S^{-1}g

FC-Adam:
  用 S 定义函数坐标，在函数坐标里保留 AdamW 时间动力学
```

这应该成为 v4.5 主线。

---

### 5.2 Functional Nesterov / lookahead dynamics

加速报告启发我们：有效加速不是裸 momentum，而是 lookahead + restart + Lyapunov。

在 functional coordinate $u$ 中做 lookahead：

$$
\tilde u_t=u_t+\mu_t(u_t-u_{t-1}).
$$

在 $\tilde u_t$ 对应的参数上计算 gradient：

$$
g_t=\nabla_u L(\tilde u_t).
$$

然后进行 Adam-like 或 normalized update。

为了避免旧 Adam moment 在 coefficient space 里失控，momentum 必须存在于 function-whitened coordinate，而不是 raw coefficient coordinate。

---

### 5.3 Functional restart / Lyapunov controller

定义能量：

$$
\mathcal E_t
=
L^{hold}_t
+
\lambda_R \mathcal P_R(t)
+
\lambda_m \mathcal P_m(t)
+
\lambda_\phi \mathcal P_\phi(t)
+
\lambda_J \mathcal P_J(t).
$$

其中：

$$
\mathcal P_R(t)=\max(0, r_R-R_t/R_{Adam,t})^2,
$$

$$
\mathcal P_m(t)=\max(0, r_m-m_t/m_{Adam,t})^2,
$$

$$
\mathcal P_\phi(t)=\max(0, \phi_t/\phi_{Adam,t}-r_\phi(t))^2,
$$

$$
\mathcal P_J(t)=\max(0, J_t/J_{Adam,t}-r_J(t))^2.
$$

如果出现：

$$
\mathcal E_{t+1} > \mathcal E_t + \epsilon_E,
$$

则触发 restart：

```text
reset functional momentum
shrink lr
relax / tighten geometry depending on phase
restore last accepted state if necessary
```

---

## 6. 阶段性 geometry policy

v4.5 不再把 geometry 作为早期 hard gate。采用三阶段策略：

### Phase I：representation formation

目标：

```text
rank / margin / loss trajectory 接近 AdamW
```

允许：

$$
\phi/A \leq 1.15.
$$

不要求早期 $\phi$ 低于 AdamW。

---

### Phase II：geometry stabilization

目标：

```text
保持 accuracy / margin，同时开始降低 phi/J
```

要求：

$$
\phi/A \leq 1.05.
$$

---

### Phase III：geometry consolidation

目标：

```text
最终 geometry 优于 AdamW
```

要求：

$$
\phi/A \leq 0.90,
$$

或至少：

$$
\phi\text{ reduction} > 10\%.
$$

---

## 7. v4.5 实验总览

本轮实验按以下顺序执行：

```text
P0: implementation smoke and coordinate correctness
P1: AdamW trajectory forensic audit
P2: FC-Adam backbone one-step and short-horizon audit
P3: Functional Nesterov / Adan-like dynamics
P4: Lyapunov restart and phased geometry controller
P5: rank / margin preservation ablation
P6: 3-seed full-budget candidate selection
P7: 5-seed confirm
P8: 10-seed final confirm
P9: failure diagnosis and theory update
```

---

## 8. P0：实现与坐标正确性检查

### 8.1 目标

确认 function-whitened coordinate 的数学实现正确。

### 8.2 必跑模型

```text
PureKAN-FixedNorm
basis_count = 16 / 24
hidden_dim = 64 / 96
depth = 2 / 4
```

### 8.3 必查不变量

每个 run 必须记录：

```text
learnable_nonKAN_params
functional_coverage
alpha_trainable
input_coeff_seen
block_coeff_seen
output_coeff_seen
metric_condition_L2
metric_condition_H1
metric_condition_dataSob
whiten_reconstruction_error
u_to_a_roundtrip_error
adam_state_shape_match
rollback_error
```

### 8.4 数学检查

检查：

$$
\|a-L^{-\top}u\|/\|a\| < 10^{-6}.
$$

检查 update 映射：

$$
\Delta a = L^{-\top}\Delta u.
$$

检查 gradient 变换：

$$
g_u = L^{-1}g_a.
$$

### 8.5 通过标准

```text
nonKAN params = 0
functional coverage = 1.0
roundtrip error < 1e-6
rollback error = 0
no NaN / Inf
metric condition finite
```

### 8.6 可视化

```text
metric spectrum per role
roundtrip error histogram
u-space grad norm vs a-space grad norm
```

---

## 9. P1：AdamW trajectory forensic audit

### 9.1 目标

先搞清楚 PureKAN-AdamW 到底为什么能训练。不要再只把 AdamW 当黑盒 baseline。

### 9.2 方法

训练 PureKAN-AdamW，并在每个 epoch / selected steps 记录 function-space trajectory。

### 9.3 必记指标

```text
train_loss
holdout_loss
test_acc
val_auc
feature_effective_rank_input
feature_effective_rank_block_l
feature_effective_rank_output
class_centroid_separation
margin_mean
margin_p10
margin_p50
phi_prime_p95
jacobian_condition
basis_occupancy_entropy
dead_basis_fraction
coefficient_norm_by_role
function_displacement_norm_by_role
AdamW_m_norm_by_role
AdamW_v_norm_by_role
AdamW_update_norm_by_role
AdamW_update_over_param_by_role
```

### 9.4 新增关键诊断

记录 AdamW 的每步 function displacement：

$$
\Delta f_{Adam,l}(x)=f_l(a_t+\Delta a^{Adam}_t,x)-f_l(a_t,x).
$$

并记录：

```text
Delta f norm
Delta f alignment across steps
role update share
rank change after update
margin change after update
phi change after update
```

### 9.5 可视化

```text
AdamW trajectory: loss / rank / margin / phi over epochs
role-wise update share stacked area
rank vs margin scatter
phi vs accuracy scatter
AdamW m/v norm curves
function displacement norm by role
```

### 9.6 产出

得到 AdamW 的 target profile：

```text
rank_target(t)
margin_target(t)
phi_budget(t)
role_update_share_target(t)
function_displacement_target(t)
```

后续所有 functional optimizer 必须与这个 target profile 对齐。

---

## 10. P2：FC-Adam backbone audit

### 10.1 目标

验证 functional-coordinate Adam 是否能成为 PureKAN functional optimizer 的 backbone。

### 10.2 候选

```text
FCAdam-L2
FCAdam-H1-low
FCAdam-dataSob
AB-RBF-FCAdam-H1-low
FCAdam-dataSob-noGeometryGate
FCAdam-dataSob-phasedGeometry
```

### 10.3 配置说明

`dataSob` 的 metric：

$$
M_l = G_{data,l}+\lambda S_l+\rho I.
$$

其中：

$$
G_{data,l}=\mathbb E[\Phi_l^\top \Phi_l].
$$

如果 full $G_{data,l}$ 太贵，先用：

```text
diagonal data covariance
low-rank + diagonal covariance
EMA covariance
```

### 10.4 必记指标

```text
cos_function_with_adam
function_R2_with_adam
cos_coeff_with_adam
holdout_1step_descent
holdout_5step_descent
holdout_20step_descent
rank/A
margin/A
phi/A
jac/A
ECE
accepted_lr
u_m_norm
u_v_norm
u_update_norm
restart_count
```

### 10.5 通过标准

P2 不要求最终超过 AdamW，但必须满足：

```text
cos_function_with_adam >= 0.85 on at least 2/3 datasets
function_R2 >= 0.70 on at least 2/3 datasets
rank/A >= 0.85
margin/A >= 0.85
holdout_5step_descent positive on all datasets
early phi/A <= 1.15
bad_step_rate <= 0.05
```

### 10.6 可视化

```text
function_R2_vs_rank_retention
cos_function_with_adam_vs_holdout_descent
rank/A over 20 steps
margin/A over 20 steps
phi/A over 20 steps
```

---

## 11. P3：Functional Nesterov / Adan-like dynamics

### 11.1 目标

测试加速报告启发的方向：function coordinate 中加入 Nesterov / Adan-like temporal dynamics。

### 11.2 候选

```text
FAdam:
  FCAdam without lookahead

FNAG:
  FCAdam + Nesterov lookahead

FAdan-lite:
  functional-coordinate Adam + gradient-difference momentum

FWin-lite:
  functional-coordinate AdamW + weight-decay-integrated lookahead
```

### 11.3 FNAG 公式

在 $u$ 坐标：

$$
\tilde u_t = u_t + \mu_t(u_t-u_{t-1}).
$$

在 $\tilde u_t$ 处计算梯度：

$$
g_t=\nabla_u L(\tilde u_t).
$$

然后执行 Adam-like update。

### 11.4 FAdan-lite 公式

记录 gradient difference：

$$
d_t=g_t-g_{t-1}.
$$

更新：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)(g_t+\gamma d_t).
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)(g_t+\gamma d_t)^2.
$$

### 11.5 关键安全机制

所有 momentum 必须位于 function-whitened coordinate $u$，不能位于 raw coefficient $a$。

### 11.6 必记指标

```text
lookahead_loss
post_update_loss
lookahead_descent
momentum_norm
momentum_cos_current_grad
momentum_cos_adamw_update
restart_triggered
rank_change_after_lookahead
margin_change_after_lookahead
phi_change_after_lookahead
```

### 11.7 通过标准

```text
holdout_20step_descent > FCAdam backbone
rank/A not lower than FCAdam by > 5%
margin/A not lower than FCAdam by > 5%
phi/A <= 1.15 early, <= 1.05 mid
no catastrophic loss spike
```

---

## 12. P4：Functional Lyapunov restart controller

### 12.1 目标

验证 restart / Lyapunov controller 是否能解决长期 trajectory drift。

### 12.2 Energy 定义

$$
\mathcal E_t
=
L^{hold}_t
+\lambda_R \max(0,r_R-R_t/R_{ref,t})^2
+\lambda_m \max(0,r_m-m_t/m_{ref,t})^2
+\lambda_\phi \max(0,\phi_t/\phi_{ref,t}-r_\phi(t))^2
+\lambda_J \max(0,J_t/J_{ref,t}-r_J(t))^2.
$$

其中 reference 可以是：

```text
PureKAN-AdamW trajectory
或当前 run 的 EMA target
```

### 12.3 Restart 条件

```text
E increases for k consecutive audits
holdout loss worsens beyond epsilon
rank/A drops below threshold
margin/A drops below threshold
phi/J exceeds phase budget
```

### 12.4 Restart 动作

```text
reset momentum
halve lr
increase damping rho
optionally relax Sobolev lambda in early phase
restore last accepted state if loss spike severe
```

### 12.5 必记指标

```text
energy_total
energy_loss_term
energy_rank_term
energy_margin_term
energy_phi_term
energy_jac_term
restart_count
restart_reason
post_restart_recovery_steps
lr_after_restart
rho_after_restart
```

### 12.6 可视化

```text
Lyapunov energy curves
restart markers on loss curve
restart reason stacked bar
rank/margin before-after restart
phi/J before-after restart
```

---

## 13. P5：rank / margin preservation ablation

### 13.1 目标

验证 rank / margin 是否是 functional optimizer 追不上 AdamW 的核心缺口。

### 13.2 候选

```text
FCAdam-dataSob
FCAdam-dataSob + rank preservation
FCAdam-dataSob + margin preservation
FCAdam-dataSob + rank + margin preservation
FNAG + rank + margin preservation
```

### 13.3 Rank preservation penalty

不是直接加入训练 loss，而是作为 update acceptance / energy penalty：

$$
\mathcal P_R=\max(0,r_R-R_t/R_{Adam,t})^2.
$$

### 13.4 Margin preservation penalty

$$
\mathcal P_m=\max(0,r_m-m_t/m_{Adam,t})^2.
$$

### 13.5 必记指标

```text
effective_rank_input
block_effective_rank_l
output_effective_rank
class_centroid_separation
margin_mean
margin_p10
margin_p50
classwise_margin
classwise_accuracy
rank_penalty_value
margin_penalty_value
```

### 13.6 通过标准

```text
rank/A >= 0.90
margin/A >= 0.90
accuracy gap vs AdamW decreases by >= 50% compared with FCAdam backbone
phi/A <= 1.10 early and <= 1.00 late
```

---

## 14. P6：3-seed full-budget candidate selection

### 14.1 候选来源

只有 P2-P5 通过 gate 的方法进入 P6。

预期候选最多 4 个：

```text
FCAdam-dataSob-phasedGeometry
FNAG-dataSob-restart
FAdan-lite-dataSob-restart
FCAdam-dataSob-rankMargin
```

### 14.2 Baselines

必须包含：

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
D0-allFullSobolev
D6-allTaskAware
F4-FNG-leftFullRight
```

### 14.3 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### 14.4 主要指标

```text
test_acc
val_loss_auc
train_loss_auc
ECE
NLL
phi_prime_p95
Jacobian condition
curvature_energy
rank/A
margin/A
cos_function_with_adam
function_R2_with_adam
step_time
memory_peak
```

### 14.5 P6 通过标准

每个候选必须满足：

```text
MNIST acc >= PureKAN-AdamW - 0.5%
Fashion acc >= PureKAN-AdamW - 0.5%
KMNIST acc >= PureKAN-AdamW - 1.0%
val_loss_auc >= PureKAN-AdamW or within 2%
ECE <= PureKAN-AdamW
late phi_prime_p95 <= PureKAN-AdamW
rank/A >= 0.90
margin/A >= 0.90
```

如果没有候选满足，不进入 5-seed。

---

## 15. P7：5-seed confirm

### 15.1 目标

验证候选不是 3-seed 偶然。

### 15.2 统计

记录 paired delta vs PureKAN-AdamW：

```text
paired_acc_delta
paired_auc_delta
paired_ece_delta
paired_phi_delta
paired_rank_delta
paired_margin_delta
```

bootstrap CI：

```text
95% CI for acc delta
95% CI for AUC delta
95% CI for phi reduction
```

### 15.3 通过标准

```text
acc delta mean >= 0 on at least 2/3 datasets
KMNIST acc delta >= -0.5%
AUC delta >= 0 on all datasets
ECE delta <= 0 on all datasets
late phi reduction >= 10% on at least 2/3 datasets
rank/A >= 0.90
margin/A >= 0.90
```

---

## 16. P8：10-seed final confirm

### 16.1 目标

只有 P7 通过才跑。

### 16.2 最终 claim 分类

#### Clean PureKAN functional optimizer

需要：

```text
acc >= PureKAN-AdamW on mean and paired CI not strongly negative
AUC >= PureKAN-AdamW
ECE <= PureKAN-AdamW
phi/J better than AdamW late phase
rank/margin preserved
strict PureKAN nonKAN params = 0
```

#### Pareto PureKAN functional optimizer

如果：

```text
accuracy roughly matches AdamW
AUC / ECE / geometry better
rank/margin preserved
```

但 accuracy CI 跨 0，则只称 Pareto。

#### Functional branch optimizer only

如果仍不能接近 PureKAN-AdamW，则结论保持：

```text
functional update works for hybrid KAN residual branches,
but not yet as a full PureKAN optimizer.
```

---

## 17. P9：失败诊断

如果 v4.5 仍失败，必须输出 failure taxonomy，而不是继续调参。

### 17.1 Failure classes

```text
F1 coordinate mismatch:
  function R2 low, cos_function low

F2 temporal dynamics failure:
  one-step good, 20-step bad, restart high

F3 rank collapse:
  rank/A < 0.85

F4 margin failure:
  margin/A < 0.85

F5 geometry conflict:
  rank/margin good but phi/J too high

F6 over-regularization:
  phi/J good but loss/rank/margin bad

F7 task overfit:
  train descent good, holdout bad

F8 acceleration instability:
  lookahead loss spikes, restart frequent
```

### 17.2 必画图

```text
failure radar by method
failure type heatmap by dataset
rank vs accuracy scatter
margin vs accuracy scatter
phi vs accuracy scatter
function R2 vs accuracy gap
restart count vs accuracy gap
energy terms over time
```

---

## 18. 最终可视化清单

必须生成以下图表：

```text
1. AdamW trajectory forensic dashboard
2. FCAdam function R2 vs rank retention
3. holdout descent vs accuracy gap
4. rank/margin/phi phase plot
5. Lyapunov energy with restart markers
6. role-wise update share stacked area
7. momentum norm and cosine trace
8. geometry consolidation curve
9. accuracy-geometry Pareto plot
10. failure taxonomy heatmap
```

每张图必须同时显示：

```text
PureKAN-AdamW
D0 Sobolev
D6 task-aware
best v4.5 candidate
```

---

## 19. 本轮最重要的禁止事项

v4.5 不允许继续做以下事情作为主线：

```text
继续扫 Sobolev alpha / beta
继续扫 branch_final_scale
继续扫 diagwarmup length
继续扫 shallow/deep 固定分工
继续单独跑 FTF / BFT 小步验收
继续把 early phi < AdamW 作为硬门槛
```

这些都已经证明不能解决核心问题。

---

## 20. v4.5 的最终目标

v4.5 要回答：

$$
\boxed{
\text{PureKAN functional optimization 是否可以通过 functional-coordinate adaptive dynamics 成立？}
}
$$

如果答案是 yes，项目进入真正 PureKAN optimizer 主线。

如果答案是 no，则应该正式写清楚：

$$
\boxed{
\text{当前 functional update 的有效范围是 hybrid KAN branch，不是 full PureKAN training。}
}
$$

这两个结果都比继续小修小补更有科学价值。
