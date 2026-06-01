# DG-KAN v4.8 Functional Geometry Feasibility Redesign 实验计划

## 0. 版本定位

v4.8 的目标不是继续微调 `Sobolev alpha / beta`、`branch scale`、`warmup`、`trust radius` 或固定的 phase schedule。v4.7 已经说明，PSFT 的 Phase-I 能产生一定任务学习信号，Phase-II 也能做 teacher-preserving consolidation 的 sweep，但没有任何配置同时通过 teacher preservation 和 geometry reduction。更关键的是，当 teacher preservation 做得较好时，geometry reduction 非常弱；当 geometry reduction 稍微出现时，accuracy drop / KL 又很大。

所以 v4.8 的问题要重新定义为：

$$
\boxed{
\text{PureKAN 是否存在“准确且几何好”的可达解？如果存在，如何沿着不破坏任务函数的方向降低几何？}
}
$$

这和之前的 U-FULL / TFU / FNG / FTF / BFT / PSFT 都不同。过去我们默认认为 “AdamW 学到的函数可以被 functional Sobolev projection 平滑化”；v4.7 的 P2 结果说明这个假设没有被验证，甚至可能是错的。因此 v4.8 必须先做 **feasibility-first**：先证明目标存在，再设计 optimizer。

本计划把下一步命名为：

```text
FGF: Functional Geometry Feasibility
NFS: Nullspace Functional Smoothing
TAN: Task-then-Nullspace functional training
```

核心路线是：

```text
1. 先找 PureKAN 的 accuracy-geometry Pareto frontier。
2. 再测试是否能在保持 logits / hidden features 的同时降低 geometry。
3. 如果可以，再把 NFS 作为周期性 projection 接入 task-learning phase。
4. 如果不可以，说明当前 PureKAN/RBF 参数化下“准确 + 几何好”可能不可达，需要改 architecture / basis，而不是继续改 optimizer。
```

---

## 1. 当前结果的深层解读

### 1.1 v4.7 的关键事实

v4.7 的 P0 说明实现路径干净：PSFT snapshot、distillation、rollback、strict PureKAN non-KAN count、coverage 都没有问题。也就是说，当前失败不能再优先归因于参数没覆盖、alpha 没固定、rollback 错误或 teacher snapshot 错误。

P1 Phase-I dynamics 显示，functional-coordinate methods 仍能产生任务学习信号。例如 MNIST 上 `FCAdam-dataSob`、`FLD-AdanLite` 的 `hold100` 接近或超过 AdamW；但它们的 rank、margin、phi 和 role dynamics 不满足 P1 gate。也就是说，它们能学局部任务，但形成的表示不符合稳定的 PureKAN-AdamW 轨迹。

P2 consolidation sweep 是最重要的诊断。它直接测试：给定一个 teacher function，能否用 KL / feature distillation 保持 teacher，同时加 Sobolev-style consolidation 降低 geometry。结果是没有 all-dataset survivor。许多 row 在 Fashion 上有少量 `phi red`，但伴随很大的 `acc drop` 和 KL；更安全的 teacher-preserving row 则只有很弱的 geometry reduction。

因此当前最准确的判断是：

$$
\boxed{
\text{PSFT 证明了 phase separation 的诊断价值，但也证明了当前 Sobolev consolidation 不是有效的 function-preserving projection。}
}
$$

### 1.2 这不是“小修小补”能解决的问题

过去几轮已经尝试过许多局部修复：

```text
fixed Sobolev U-FULL
D6 allTaskAware
role-aware TFU
FNG / KFAC-like metric
FTF layer-local target fitting
BFT trust region
FCAdam / FLD / AdanLite / WinLite
PSFT learn -> consolidate
normalization ablation
```

这些实验共同说明：

```text
1. Sobolev-only 能改善几何，但不能训练好 PureKAN。
2. Task-aware / FCAdam 能改善短程任务下降，但长期 trajectory 不稳。
3. FTF 能拟合局部 target，但跨层组合会导致函数漂移。
4. BFT 能防止灾难，但变得太保守。
5. PSFT 暴露了 learn 和 smooth projection 之间缺少可保持任务函数的桥。
```

所以 v4.8 不再继续寻找一个新的单步方向：

$$
\Delta a = -\eta M^{-1}g.
$$

而是研究 functional optimizer 是否需要解决一个约束问题：

$$
\begin{aligned}
\min_{\Delta \theta} \quad & \Delta R_{geo}(\theta; \Delta\theta) \\
\text{s.t.} \quad & f_{\theta + \Delta\theta}(x) \approx f_\theta(x), \\
& h_{l,\theta + \Delta\theta}(x) \approx h_{l,\theta}(x), \\
& \text{task loss does not increase.}
\end{aligned}
$$

这就是 NFS 的动机。

---

## 2. 新理论假设

### 2.1 Hypothesis A: 当前 geometry target 可能不可达

PureKAN-AdamW 能训练，说明 PureKAN 架构有表达力。但它的 $$geometry 可能很差。这并不自动意味着存在一个同等准确、但 geometry 很好的 PureKAN 解。

有两种可能：

```text
A1. 可达：存在准确且平滑的 PureKAN 解，只是当前 optimizer 找不到。
A2. 不可达：当前 PureKAN/RBF basis/width/depth 需要高曲率才能完成分类，几何好和准确率存在硬 trade-off。
```

v4.7 consolidation 失败不能区分 A1 和 A2。因此 v4.8 第一任务是画出 PureKAN 的 accuracy-geometry Pareto frontier。

### 2.2 Hypothesis B: 当前 Sobolev projection 不是 function-preserving

当前 consolidation 大致是通过 teacher KL / feature loss / Sobolev penalty 联合优化。问题是，这不是显式地沿 teacher function 的 nullspace 下降。它可能在降低 coefficient Sobolev energy 时改变 logits 或 hidden representation。

真正的 geometry projection 应该优先在 teacher-preserving nullspace 中进行：

$$
J_f \Delta\theta \approx 0,
$$

其中 $J_f$ 是 logits 或 hidden features 对 coefficient 的 Jacobian。目标是降低 geometry regularizer $R(\theta)$：

$$
\Delta\theta_{NFS}
=
- P_{\mathrm{null}(J_f)} M^{-1}\nabla R(\theta).
$$

一个实用形式是：

$$
P_M
=
I
-
M^{-1}J_f^T
\left(J_fM^{-1}J_f^T + \mu I\right)^{-1}
J_f.
$$

然后：

$$
\Delta\theta
=
-\eta P_M M^{-1}\nabla R(\theta).
$$

这个方向的语义是：

```text
先保持当前函数，再寻找能降低 geometry 的自由度。
```

这和之前 “直接加 Sobolev penalty 训练” 完全不同。

### 2.3 Hypothesis C: PureKAN 需要 task phase 和 smoothing phase 的不同参数化

Phase I 的 task learning 更像 AdamW / FCAdam / FLD：它需要 rank、margin、role share、temporal dynamics。Phase II 的 geometry consolidation 更像 constrained smoothing：它需要保持 logits / features，不能再改任务边界。

所以 v4.8 不再试图用同一种 optimizer 完成全部工作，而是验证：

$$
\boxed{
\text{task-learning coordinate 和 geometry-projection coordinate 是否必须分离。}
}
$$

---

## 3. 新方法族

### 3.1 FGF: Functional Geometry Feasibility

FGF 是 v4.8 的第一部分，不是 optimizer，而是存在性测试。它回答：

```text
在当前 PureKAN 架构下，是否存在 accuracy 接近 AdamW、geometry 明显优于 AdamW 的解？
```

训练方法包括：

```text
PureKAN-AdamW
PureKAN-AdamW + coefficient Sobolev penalty
PureKAN-AdamW + derivative/Jacobian penalty
PureKAN-FCAdam task phase
PureKAN-FCAdam + weak Sobolev penalty
PureKAN with wider hidden / larger basis / AB-RBF basis
```

如果任何 AdamW-based geometry regularization 都找不到准确且低 geometry 的解，那说明 functional optimizer 不应该继续承诺 “PureKAN 既准确又几何好”，除非改架构。

### 3.2 NFS: Nullspace Functional Smoothing

NFS 是真正的新 Phase-II projection。

给定 teacher $\theta_T$，我们冻结 teacher outputs：

$$
z_T=f_{\theta_T}(x),
\quad
h_{l,T}=h_l(x;\theta_T).
$$

定义约束 residual：

$$
\epsilon_z(\Delta\theta)=
\frac{\|f_{\theta+\Delta\theta}(x)-z_T\|}{\|z_T\|+\epsilon},
$$

$$
\epsilon_h(\Delta\theta)=
\frac{\sum_l\|h_l(\theta+\Delta\theta)-h_{l,T}\|}{\sum_l\|h_{l,T}\|+\epsilon}.
$$

目标是降低：

$$
R_{geo}(\theta)=
\sum_l
\left(
\gamma_1\phi'_{p95,l}
+
\gamma_2\log\kappa(J_l)
+
\gamma_3\|A_l\|_{S_l}^2
\right).
$$

NFS 每次 proposal 必须满足：

$$
\epsilon_z < r_z,
\quad
\epsilon_h < r_h,
\quad
\mathrm{KL}(p_T\|p_{\theta+\Delta}) < r_{KL}.
$$

如果 proposal 降低了 geometry 但破坏任务函数，就 reject 或 backtrack。

### 3.3 TAN: Task-then-Nullspace Functional Training

TAN 是把 NFS 接入训练循环：

```text
Phase A: task learning
  用 FCAdam / FLD / AdamW-diagnostic 产生表示和 margin。

Phase B: nullspace smoothing
  用 NFS 在保持 logits/features 的同时降低 geometry。

Phase C: small-step task refresh
  用小步 task-aware update 修复 projection 造成的轻微 loss drift。
```

TAN 的关键不是一次性 projection，而是交替循环：

```text
repeat:
  task steps K_task
  snapshot teacher
  NFS smoothing K_smooth
  refresh steps K_refresh
```

但 v4.8 初期必须先用 diagnostic 模式，不直接跑大 seed。

---

## 4. 实验总览

v4.8 分成九个阶段。每个阶段都有明确 stop/go gate，避免继续把资源花在无效 seed confirm 上。

```text
P0: implementation smoke and invariant checks
P1: feasibility frontier with AdamW geometry regularization
P2: offline NFS projection audit from AdamW teachers
P3: offline NFS projection audit from FCAdam/FLD teachers
P4: TAN one-cycle micro-run
P5: TAN alternating-cycle ablation
P6: capacity / basis feasibility expansion
P7: 3-seed full-budget selection
P8: 5-seed confirm
P9: 10-seed final confirm and failure diagnosis
```

---

## 5. P0: Implementation smoke and invariants

### 5.1 目的

P0 只确认实现路径，不看方法优劣。v4.8 引入 NFS，需要确认 projection / rollback / teacher snapshot / Jacobian-vector products 都可靠。

### 5.2 必跑配置

```text
PureKAN-AdamW-smoke
PureKAN-FCAdam-smoke
NFS-logit-only-smoke
NFS-hidden-only-smoke
NFS-logit-hidden-smoke
NFS-random-nullspace-smoke
TAN-one-cycle-smoke
```

每个数据集先跑：

```text
MNIST
Fashion-MNIST
KMNIST
```

只用：

```text
seed = 0
small train_size = 1024
steps = 5 to 20
```

### 5.3 必须记录

```text
implementation/nonkan_param_count
implementation/functional_coverage
implementation/input_coeff_seen
implementation/block_coeff_seen
implementation/output_coeff_seen
implementation/teacher_snapshot_error
implementation/rollback_max_abs_error
implementation/temp_apply_max_abs_error
implementation/jvp_finite
implementation/vjp_finite
implementation/cg_residual
implementation/cg_iters
implementation/nfs_projection_finite
implementation/nfs_constraint_residual
implementation/nfs_geometry_delta
implementation/no_nan_inf
```

### 5.4 通过标准

```text
nonKAN_param_count = 0
functional_coverage = 1.0
rollback_max_abs_error < 1e-8
teacher_snapshot_error = 0
jvp/vjp finite
cg_residual < 1e-3, 或明确记录 fallback
no NaN / Inf
```

如果 P0 不过，不进入 P1。

---

## 6. P1: Feasibility frontier with AdamW geometry regularization

### 6.1 目的

P1 是 v4.8 最关键的诊断之一。它回答：

$$
\boxed{
\text{准确且几何好的 PureKAN 解是否存在？}
}
$$

如果不存在，后续任何 functional optimizer 都不应被要求在 PureKAN 上同时超过 AdamW 并显著降低 geometry。

### 6.2 方法

用 AdamW 作为强 task optimizer，然后加入不同几何正则。这里不追求 strict functional claim，只追求 feasibility。

候选：

```text
A0: PureKAN-AdamW
A1: AdamW + coeff Sobolev penalty lambda = 1e-6
A2: AdamW + coeff Sobolev penalty lambda = 3e-6
A3: AdamW + coeff Sobolev penalty lambda = 1e-5
A4: AdamW + derivative penalty phi-prime proxy lambda = 1e-5
A5: AdamW + Jacobian penalty lambda = 1e-5
A6: AdamW + mixed geometry penalty
A7: AdamW + late geometry penalty only
A8: AdamW then post-hoc L2/Sobolev weight decay fine-tune
```

其中 mixed geometry penalty 为：

$$
\mathcal L
=
\mathcal L_{CE}
+
\lambda_S\sum_l\|A_l\|_{S_l}^2
+
\lambda_\phi\sum_l \widehat{\phi'_{p95,l}}
+
\lambda_J\sum_l \log \widehat{\kappa(J_l)}.
$$

### 6.3 运行设置

先做 1-seed feasibility：

```text
seed = 0
epochs = current PureKAN full budget
hidden_dim = 64
basis_count = 16
depth = 2 or 4, 取当前 PureKAN-AdamW baseline 配置
```

如果某个候选接近 gate，再做 3-seed：

```text
seeds = 0,1,2
```

### 6.4 必须记录

```text
task/train_loss_curve
task/val_loss_curve
task/test_acc
task/val_auc
task/nll
task/ece
representation/effective_rank_input
representation/effective_rank_block_mean
representation/effective_rank_output
representation/class_centroid_separation
representation/margin_mean
representation/margin_p10
representation/margin_p50
geometry/phi_prime_p95
geometry/phi_prime_max
geometry/jacobian_condition_max
geometry/sobolev_norm_total
geometry/sobolev_norm_by_role
geometry/curvature_energy
basis/basis_occupancy_entropy
basis/dead_basis_fraction
basis/out_of_grid_fraction
role/update_share_input
role/update_share_block
role/update_share_output
compute/step_time_ms
compute/memory_peak_mb
```

### 6.5 必须可视化

```text
1. accuracy vs phi_prime_p95 Pareto frontier
2. accuracy vs max_jacobian_condition Pareto frontier
3. accuracy vs sobolev_norm_total Pareto frontier
4. val_loss_curve by lambda
5. margin_p10 vs phi_prime_p95 scatter
6. effective_rank vs accuracy scatter
7. geometry penalty lambda vs acc drop line plot
8. basis occupancy heatmap by role
```

### 6.6 P1 判定

定义 AdamW baseline 为 $A_0$。候选 $A_i$ 被认为证明 feasibility，如果：

$$
Acc(A_i) \ge Acc(A_0)-0.01,
$$

并且：

$$
\phi_{p95}(A_i) \le 0.8\phi_{p95}(A_0)
$$

或：

$$
\kappa_J(A_i) \le 0.5\kappa_J(A_0),
$$

同时：

$$
ECE(A_i) \le ECE(A_0)+0.02.
$$

如果没有任何候选在三个数据集上通过，则说明当前 PureKAN 架构可能没有明显的 “smooth accurate solution”。这时进入 P6 capacity expansion，而不是继续改 optimizer。

---

## 7. P2: Offline NFS projection audit from AdamW teachers

### 7.1 目的

P2 测试：给定一个强 teacher，比如 PureKAN-AdamW，NFS 能否在不破坏 logits/features 的情况下降低 geometry。

这直接针对 v4.7 的失败点：当前 consolidation operator 无法同时 teacher-preserve 和 geometry-reduce。

### 7.2 Teacher 来源

```text
TeacherA: PureKAN-AdamW at step 20
TeacherB: PureKAN-AdamW at step 100
TeacherC: PureKAN-AdamW final checkpoint
TeacherD: AdamW + geometry penalty best checkpoint from P1
```

### 7.3 NFS 变体

```text
NFS-Z:
  constrain logits only

NFS-H:
  constrain selected hidden layers only

NFS-ZH:
  constrain logits + hidden features

NFS-ZH-margin:
  constrain logits + hidden + correct-class margin

NFS-role-input:
  smooth input KAN only

NFS-role-block:
  smooth residual block KAN only

NFS-role-output:
  smooth output KAN only

NFS-role-cycle:
  input -> block -> output sequential smoothing
```

### 7.4 NFS 公式

令 $R(\theta)$ 是几何目标，$c(\theta)$ 是 teacher-preserving constraints：

$$
c(\theta)=
\begin{bmatrix}
\sqrt{\tau_z}(z_\theta-z_T) \\
\sqrt{\tau_h}(h_\theta-h_T) \\
\sqrt{\tau_m}(m_\theta-m_T)
\end{bmatrix}.
$$

NFS 近似解：

$$
\Delta\theta
= -\eta
\left(I - M^{-1}J_c^T(J_cM^{-1}J_c^T+\mu I)^{-1}J_c\right)
M^{-1}\nabla R.
$$

如果这个完整投影太贵，先实现 CG / low-rank 近似：

```text
NFS-CG-5
NFS-CG-10
NFS-lowrank-32
NFS-lowrank-64
NFS-diag-projector
```

### 7.5 必须记录

```text
teacher/teacher_acc
teacher/teacher_loss
teacher/teacher_ece
teacher/teacher_phi_p95
teacher/teacher_jacobian
projection/geometry_before
projection/geometry_after
projection/phi_reduction
projection/jacobian_reduction
projection/sobolev_reduction
projection/kl_teacher_student
projection/logit_relative_drift
projection/hidden_relative_drift
projection/margin_relative_drift
projection/constraint_residual_predicted
projection/constraint_residual_actual
projection/predicted_geometry_delta
projection/actual_geometry_delta
projection/cg_iters
projection/cg_residual
projection/nullspace_fraction
projection/backtrack_count
projection/accepted_eta
projection/reject_reason
projection/role
projection/update_norm_by_role
projection/update_sobolev_norm
```

### 7.6 必须可视化

```text
1. teacher KL vs phi reduction scatter
2. logit drift vs phi reduction scatter
3. hidden drift vs phi reduction scatter
4. predicted vs actual geometry reduction
5. predicted vs actual logit drift
6. NFS constraint residual histogram
7. accepted eta distribution
8. role-wise geometry reduction bar chart
9. before/after edge function plots for selected edges
10. before/after basis coefficient spectrum
```

### 7.7 P2 判定

P2 通过条件不是最终 accuracy，而是 projection 可行性。

一个 NFS 配置通过，如果在每个数据集上存在 teacher checkpoint 使得：

$$
\mathrm{KL}(p_T\|p_{after}) < 0.05,
$$

$$
\frac{\|z_{after}-z_T\|}{\|z_T\|+\epsilon}<0.03,
$$

$$
Acc_{after} \ge Acc_T - 0.005,
$$

并且：

$$
\phi_{red} > 0.10
\quad\text{or}\quad
J_{red} > 0.20.
$$

如果 P2 全部失败：

```text
结论不是 optimizer 没调好，而是当前参数化下缺少 function-preserving geometry degrees of freedom。
```

此时进入 P6 architecture/basis expansion。

---

## 8. P3: Offline NFS from functional teachers

### 8.1 目的

P2 用 AdamW teacher，只是证明 projection 是否存在。P3 用 functional teacher，测试能否把 functional Phase-I 学到的粗糙函数投影成更好几何。

Teacher：

```text
FCAdam-dataSob step 20 / 100
FLD-AdanLite step 20 / 100
D6 allTaskAware step 100
FNG-leftFullRight step 100
```

### 8.2 重点问题

P3 判断：functional teacher 是否比 AdamW teacher 更容易被 NFS 平滑化。

可能结果有三种：

```text
1. AdamW teacher 可平滑，functional teacher 不可平滑：functional Phase-I 学到的函数太粗糙/错位。
2. functional teacher 可平滑，AdamW teacher 不可平滑：functional trajectory 更接近 smoothable solution。
3. 两者都不可平滑：当前 architecture/basis 没有足够 nullspace。
```

### 8.3 记录和图

沿用 P2 全部指标，并额外记录：

```text
teacher_type
teacher_holdout_descent
teacher_rank
teacher_margin
teacher_phi_ratio_to_adamw
teacher_role_share_input
teacher_role_share_block
teacher_role_share_output
smoothability_score
```

定义：

$$
\mathrm{smoothability}
=
\frac{\phi_{red}}{\mathrm{KL}+0.01}
\cdot
\mathbf{1}[Acc_{drop}<0.01].
$$

---

## 9. P4: TAN one-cycle micro-run

### 9.1 目的

如果 P2/P3 显示 NFS 可以 function-preserving 降几何，就进入一轮 TAN：

```text
Task phase -> NFS smoothing -> refresh phase
```

### 9.2 配置

Task phase 候选：

```text
T0: FCAdam-dataSob
T1: FLD-AdanLite
T2: FCAdam-L2
T3: D6 allTaskAware
T4: PureKAN-AdamW diagnostic teacher, not final strict functional claim
```

Smoothing phase：

```text
Best NFS-ZH from P2/P3
Best role-specific NFS from P2/P3
```

Refresh phase：

```text
R0: none
R1: 5 steps FCAdam small lr
R2: 10 steps FCAdam small lr
R3: 5 steps task-diag D6 small lr
```

### 9.3 一个 cycle 的形式

$$
\theta_0
\xrightarrow{K_{task}}
\theta_T
\xrightarrow{K_{NFS}}
\theta_S
\xrightarrow{K_{refresh}}
\theta_R.
$$

### 9.4 必须记录

```text
cycle/task_acc_before
cycle/task_acc_after_task
cycle/task_acc_after_smooth
cycle/task_acc_after_refresh
cycle/val_loss_after_task
cycle/val_loss_after_smooth
cycle/val_loss_after_refresh
cycle/phi_after_task
cycle/phi_after_smooth
cycle/phi_after_refresh
cycle/teacher_KL_after_smooth
cycle/refresh_recovered_loss
cycle/refresh_recovered_margin
cycle/rank_after_each_phase
cycle/margin_after_each_phase
cycle/ece_after_each_phase
cycle/nfs_acceptance_rate
cycle/nfs_reject_reason
```

### 9.5 可视化

```text
1. task -> smooth -> refresh phase plot: loss, acc, phi, rank, margin
2. KL after smooth vs refresh recovery scatter
3. per-cycle geometry reduction bar chart
4. per-role update share in each phase
5. before/after confusion matrix change
```

### 9.6 P4 判定

P4 通过条件：

```text
After one cycle:
  accuracy drop from task phase < 1%
  phi reduction from task phase > 10%
  refresh recovers at least 80% of loss increase caused by smoothing
  ECE not worse than task phase by more than 0.02
```

---

## 10. P5: Alternating TAN cycles

### 10.1 目的

P4 只看一个 cycle。P5 看多 cycle 是否稳定。

### 10.2 配置

只取 P4 最好的 2 到 3 个组合。

```text
cycle_count = 3 or 5
K_task = 50 or 100 steps
K_smooth = 5 or 10 accepted NFS steps
K_refresh = 5 or 10 steps
```

### 10.3 必须记录

```text
cycle_index
acc_by_cycle
val_loss_by_cycle
phi_by_cycle
jacobian_by_cycle
rank_by_cycle
margin_by_cycle
KL_to_cycle_teacher
NFS_accept_rate_by_cycle
refresh_recovery_by_cycle
geometry_gain_per_task_loss
```

### 10.4 P5 判定

P5 通过，如果多 cycle 后：

$$
Acc_{TAN} \ge Acc_{FCAdam\ phase} - 0.01,
$$

且：

$$
\phi_{TAN} \le 0.85\phi_{FCAdam\ phase},
$$

并且：

$$
Acc_{TAN} \ge Acc_{D6} + 0.03
$$

在 Fashion 和 KMNIST 上至少同时成立。

---

## 11. P6: Capacity and basis feasibility expansion

### 11.1 触发条件

如果 P1 显示没有 smooth accurate frontier，或者 P2 显示 NFS 不存在可靠 function-preserving smoothing direction，就触发 P6。

### 11.2 目的

判断失败是 optimizer 问题，还是 PureKAN/RBF 参数化没有足够低几何自由度。

### 11.3 变量

```text
hidden_dim: 64, 96, 128
basis_count: 16, 24, 32, 48
depth: 2, 4
basis_type:
  RBF only
  AB-RBF: constant + linear + RBF
  wide-grid RBF
  multi-scale RBF
residual_alpha:
  fixed1
  smaller fixed alpha 0.5
normalization:
  FixedNorm
  ScalarGainNorm functional
```

### 11.4 必须对照

```text
PureKAN-AdamW
PureKAN-AdamW + geometry regularization
FCAdam-dataSob
D6 allTaskAware
NFS projection from AdamW teacher
```

### 11.5 记录

```text
param_count
basis_count
hidden_dim
depth
basis_type
basis_occupancy_entropy
basis_dead_frac
out_of_grid_frac
acc
val_auc
ece
rank
margin
phi
jacobian
sobolev_norm
smoothability_score
step_time
memory
```

### 11.6 可视化

```text
1. capacity vs accuracy-geometry frontier
2. basis_count vs smoothability_score
3. hidden_dim vs phi at fixed accuracy
4. basis occupancy heatmap by basis_type
5. param_count vs acc/phi Pareto
6. edge function examples for RBF vs AB-RBF vs multiscale
```

### 11.7 P6 判定

如果更大 capacity / richer basis 出现 smooth accurate frontier，说明之前失败是 architecture capacity / basis issue。下一步应优先改 PureKAN primitive。

如果更大 capacity 仍然没有 frontier，说明 PureKAN 本身可能需要 hybrid components，或者几何目标过强。

---

## 12. P7: 3-seed full-budget candidate selection

### 12.1 进入条件

只有满足以下之一才进入 P7：

```text
1. P1 找到 smooth accurate frontier，并且 P2 找到可靠 NFS。
2. P5 多 cycle TAN 在 seed0 上接近 PureKAN-AdamW。
3. P6 找到新的 basis/capacity 使 smooth accurate solution 可达。
```

### 12.2 候选数量限制

最多 4 个：

```text
C1: best TAN functional-only candidate
C2: best NFS-projection candidate
C3: best capacity/basis candidate
C4: best diagnostic AdamW+geometry candidate, not final functional claim
```

### 12.3 对照

```text
PureKAN-AdamW
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
D6 allTaskAware
FNG-leftFullRight
```

### 12.4 3-seed gate

必须满足：

```text
MNIST:
  acc >= PureKAN-AdamW - 1.0%
Fashion:
  acc >= PureKAN-AdamW - 1.0%
KMNIST:
  acc >= PureKAN-AdamW - 1.5%
```

并且至少满足其中一个 geometry 条件：

```text
phi reduction vs PureKAN-AdamW > 15%
或 J reduction vs PureKAN-AdamW > 30%
```

并且：

```text
ECE not worse by more than 0.02
val-loss AUC not worse by more than 5%
```

---

## 13. P8: 5-seed confirm

P8 只运行 P7 中最多两个候选。

必须额外做 paired seed analysis：

```text
paired acc delta vs PureKAN-AdamW
paired val_auc delta vs PureKAN-AdamW
paired phi delta vs PureKAN-AdamW
paired ECE delta vs PureKAN-AdamW
paired smoothability score
```

P8 通过：

$$
\Delta Acc_{paired} > -0.01
$$

且：

$$
\Delta \phi_{paired} < -0.15\phi_{AdamW}
$$

或：

$$
\Delta J_{paired} < -0.30J_{AdamW}.
$$

---

## 14. P9: 10-seed final confirm and failure diagnosis

### 14.1 成功判定

v4.8 final success 不是只超过旧 U-FULL。必须比较：

```text
PureKAN-AdamW
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
```

最终候选必须：

```text
1. accuracy 接近或超过 PureKAN-AdamW。
2. geometry 明显优于 PureKAN-AdamW。
3. 没有 non-KAN trainable params。
4. functional coverage = 1.0。
5. ECE 不显著变差。
6. wall-clock 可以慢，但需要明确报告。
```

### 14.2 如果失败，必须输出明确 failure type

```text
F1: no smooth accurate frontier exists
F2: NFS cannot preserve function while reducing geometry
F3: task phase cannot reach AdamW-level representation
F4: smoothing phase destroys logits/features
F5: refresh phase cannot recover from smoothing
F6: capacity/basis insufficient
F7: geometry metric too strict / wrong metric
F8: PureKAN requires hybrid-like non-KAN stabilizers
```

### 14.3 最终报告必须包含

```text
accuracy-geometry Pareto frontier
teacher-preserving smoothing feasibility
TAN cycle dynamics
capacity/basis smoothability map
final 10-seed scorecard
failure taxonomy
```

---

## 15. 本轮必须停止的方向

v4.8 不再继续投入以下方向，除非作为对照：

```text
1. 固定 Sobolev U-FULL 直接训练 PureKAN。
2. 再扫 branch_final_scale。
3. 再扫 diagwarmup / smooth transition。
4. 再扫固定 shallow/deep hand-crafted metric 分工。
5. 再做单独 FTF layer-local target fitting。
6. 再做只有 trust gate 的 BFT。
7. 再把 D6 allTaskAware 当主线。
```

原因是这些方向已经反复证明：要么能短程下降但长期不行，要么保几何但任务不足，要么能学任务但无法被平滑投影。

---

## 16. 预期解释路径

v4.8 的结果会把项目带向三种可能结论。

### 16.1 最好情况

```text
P1 找到 smooth accurate solution。
P2 NFS 能保持 teacher 并降 geometry。
P5 TAN 多 cycle 成功。
```

结论：

$$
\boxed{
\text{PureKAN functional optimization 可行，但需要 task phase + nullspace smoothing，而不是单一 Sobolev update。}
}
$$

### 16.2 中间情况

```text
P1 有 smooth accurate frontier，但 P2 NFS 失败。
```

结论：

$$
\boxed{
\text{目标存在，但当前 projection operator 错；需要更强的 constrained optimization / better tangent model。}
}
$$

### 16.3 严重情况

```text
P1/P6 都找不到 smooth accurate frontier。
```

结论：

$$
\boxed{
\text{当前 PureKAN/RBF 架构下，准确和低 geometry 可能是硬 trade-off；之前 Hybrid 成功依赖 non-KAN interface。}
}
$$

这时应该停止 PureKAN functional optimizer 主线，转向：

```text
Rational/KAT primitive-specific metric
Hybrid-DGKAN branch optimizer
更大/多尺度 PureKAN architecture
```

---

## 17. 最终执行建议

第一批只跑：

```text
P0
P1 seed0 feasibility frontier
P2 AdamW teacher NFS projection
```

不要直接跑 P3-P9。只有当 P1 或 P2 出现明确正信号，才继续。

优先级如下：

```text
Priority 1: P1 feasibility frontier
Priority 2: P2 NFS projection from AdamW teacher
Priority 3: P6 capacity/basis expansion if P1/P2 fail
Priority 4: P4/P5 TAN cycles only if NFS passes
Priority 5: seed confirm only after candidate is mechanistically justified
```

一句话：

$$
\boxed{
\text{v4.8 先证明 smooth accurate PureKAN 是否存在，再讨论 functional optimizer 如何找到它。}
}
$$
