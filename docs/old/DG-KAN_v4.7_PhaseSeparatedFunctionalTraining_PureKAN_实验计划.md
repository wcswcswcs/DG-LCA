# DG-KAN v4.7：Phase-Separated Functional Training for PureKAN 详细实验计划

> 目标：重新设计 PureKAN functional optimizer。  
> 版本：v4.7  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 核心结论预设：不要继续把 functional update 当成一个静态 preconditioner；下一轮要把它改成 **分阶段的 functional learning dynamics**。

---

## 0. 当前结论：v4.6 不是普通失败，而是暴露了 optimizer 定义问题

v4.6 的结果非常有价值。它说明我们已经不在“实现有没有接对”的阶段，而是在“functional optimizer 本身如何定义”的阶段。

P0 已经通过：strict PureKAN 仍然满足：

```text
nonKAN params = 0
coefficient coverage = 1.0
u -> a roundtrip = 0
rollback = 0
run errors = 0
```

所以这轮不能再主要怀疑：

```text
input/output KAN 没被更新
alpha 没 fixed
rollback 有 bug
functional coordinate whitening 错
```

这些实现问题已经基本排除。

P1 给出了 AdamW 早期轨迹 envelope：

| dataset | holdout Δ | rank ratio | phi ratio | input share | block share | output share |
|---|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | 1.7950 | 0.7919 | 1.1583 | 0.6774 | 0.2152 | 0.1074 |
| KMNIST | 1.7909 | 0.7285 | 1.1776 | 0.6796 | 0.2149 | 0.1054 |
| MNIST | 1.9020 | 0.5084 | 1.2107 | 0.6830 | 0.2131 | 0.1039 |

这个表给出一个重要事实：AdamW 早期不是几何收缩器。它会允许 $\phi$ 上升、rank 变化，并且大部分更新集中在 input KAN role 上。也就是说，PureKAN-AdamW 的成功更像：

```text
先形成表示和 margin，再谈几何。
```

P2 的结果进一步说明：FLD / FCAdam / AdanLite 已经能产生很强的 20-step holdout descent，甚至比 AdamW 的短程下降更大，但仍然没有 survivor。以 `FLD-AdanLite` 为例：

| dataset | hold20 | hold/A | rank | margin | phi |
|---|---:|---:|---:|---:|---:|
| MNIST | 1.4673 | 5.5328 | 0.3243 | 1.6747 | 1.4171 |
| Fashion-MNIST | 1.5464 | 1.4193 | 0.3350 | 1.6167 | 1.3475 |
| KMNIST | 1.3529 | 2.5968 | 0.3574 | 1.3924 | 1.3747 |

这说明：

```text
短程任务下降有了；
但是 rank / margin / phi 没有形成可持续的表示学习轨迹。
```

P3 的 100-step audit 继续支持这个判断。`FCAdam-dataSob / FLD-AdanLite / FLD-LyapunovRestart` 不再是完全不学习；它们甚至在部分数据集上有接近 AdamW 的 holdout descent。但它们的共同问题是：

```text
rank 偏低或不稳定；
margin 过大但未转化为 clean accuracy；
phi 过高；
100-step 轨迹不成为可确认的 optimizer。
```

因此，当前结论不是：

```text
functional update 没有下降方向。
```

而是：

```text
functional update 有短程下降能力，但没有形成深度表示学习所需的长期轨迹。
```

---

## 1. 重新定义问题：functional update 难调的根因

过去我们一直在尝试把 functional update 写成：

$$
\Delta a = -\eta M^{-1}g.
$$

不同版本只是改变 $M$：

```text
Sobolev Gram
task-aware diagonal
FNG / KFAC
functional-coordinate Adam
AdanLite / WinLite
BFT / trust gate
```

但 v4.6 说明，核心问题不是某个 $M$ 还没调对，而是：

$$
\boxed{
\text{单步 metric-preconditioned update 不是完整的深度网络 optimizer。}
}
$$

PureKAN 要训练的是一个完整深层函数：

$$
x
\rightarrow
\operatorname{KAN}_{in}
\rightarrow
\operatorname{KAN}_{block,1}
\rightarrow
\cdots
\rightarrow
\operatorname{KAN}_{out}
\rightarrow
z.
$$

其中 input KAN、middle blocks、output KAN 的角色完全不同：

```text
input KAN:
  raw pixels -> hidden feature
  主要负责 early representation formation

block KAN:
  hidden feature -> composed hidden feature
  主要负责 feature transformation / residual refinement

output KAN:
  hidden feature -> logits
  主要负责 margin / class boundary
```

一个统一的静态 functional metric 无法同时完成这三件事。

更严重的是，之前的 gate 同时要求：

```text
短程 loss descent
Adam-like function trajectory
rank / margin retention
geometry improvement
```

但 AdamW 早期自身也不是 geometry-improving trajectory。因此下一步不能把 geometry 作为所有阶段的硬约束，而要把训练过程拆成阶段：

```text
Phase I:
  表示形成，允许 geometry 变粗糙

Phase II:
  函数空间几何收束，同时保持已经学到的 logits / features

Phase III:
  小步任务微调与几何保持
```

这就是 v4.7 的核心方向：

$$
\boxed{
\text{Phase-Separated Functional Training，简称 PSFT。}
}
$$

---

## 2. 新核心假设：学习和几何不能再在同一步里硬兼容

v4.6 的 `FCAdam-dataSob / FLD-AdanLite` 证明了强任务下降方向存在，但它们被 early geometry / rank / trajectory gate 拦住。与其继续要求一个 update 同时做到所有事情，不如把它拆开：

### 2.1 Phase I：Functional Representation Learning

Phase I 的目标是学任务，不是压几何。它应该允许：

$$
\phi/A \leq 1.5
$$

甚至短期内更高，但必须满足：

$$
L_{\text{holdout}}\downarrow,
\quad
\text{margin}_{p10}\uparrow,
\quad
\text{class separation}\uparrow.
$$

这里的 update 使用 function-coordinate adaptive dynamics：

$$
a_l = L_l^{-T}u_l.
$$

在 $u_l$ 空间中做 Adam / Adan / Win-like update：

$$
u_{l,t+1}
=
u_{l,t}
+
\Delta u_{l,t}.
$$

Phase I 的 update 形式可以写成：

$$
\Delta u_{l,t}
=
\operatorname{AdaptiveStep}
\left(
\nabla_{u_l}
\left[
\mathcal L_{\text{CE}}
+
\lambda_{rank}\mathcal R_{rank}
+
\lambda_{sep}\mathcal R_{sep}
+
\lambda_{margin}\mathcal R_{margin}
\right]
\right).
$$

Phase I 中 Sobolev 只做弱 damping，不作为主目标：

$$
\lambda_S^{early} \approx 0.
$$

### 2.2 Phase II：Functional Geometry Consolidation

Phase II 的目标不是继续猛烈学任务，而是把 Phase I 学到的函数进行几何压缩 / 平滑，同时不破坏预测。

设 Phase I checkpoint 为 teacher $\theta^T$，它产生 teacher logits $z^T$ 和 hidden states $h_l^T$。

Phase II 解的是：

$$
\min_\theta
\quad
\mathcal L_{\text{CE}}(\theta)
+
\tau_z
\operatorname{KL}
\left(
p_{\theta^T}(y|x)
\| 
p_\theta(y|x)
\right)
+
\tau_h
\sum_l
\|h_l(\theta)-h_l(\theta^T)\|_2^2
+
\lambda_S
\sum_l
\|a_l\|_{S_l}^2.
$$

直觉：

```text
不要在几何 consolidation 阶段重新发明分类边界；
先保持已学到的 logits / features，
再用 Sobolev 把函数变平滑。
```

这和之前的 U-FULL 完全不同。之前是：

```text
一开始就用 Sobolev 管住所有层。
```

现在是：

```text
先学出有用函数，再做函数空间投影和压缩。
```

### 2.3 Phase III：Small-Step Functional Fine-Tuning

Phase III 的目标是小步微调：

$$
\Delta a_l
=
-\eta_l
\left(
G_l+\lambda_lS_l+\rho I
\right)^{-1}
\nabla_{a_l}
\left[
\mathcal L_{\text{CE}}
+
\tau_z\operatorname{KL}
+
\tau_h\mathcal L_{feat}
\right].
$$

这里 $\eta_l$ 小，$\lambda_l$ 中等，几何 gate 才变成硬约束。

---

## 3. 为什么这是“重新设计”，不是小修小补

这个 v4.7 计划不再问：

```text
Sobolev alpha 取多少？
branch scale 取多少？
Nesterov / Adan / Win 哪个更好？
```

而是问：

$$
\boxed{
\text{PureKAN functional optimizer 是否必须是一个分阶段训练协议？}
}
$$

这和之前所有尝试不同：

| 旧方法 | 核心问题 |
|---|---|
| U-FULL / Sobolev-only | 几何好，但不是任务 descent optimizer |
| D6 / Global TFU | one-step descent 好，但长程表示学习不足 |
| FNG / EK-FNG | curvature 更合理，但仍是局部 preconditioner |
| FTF | 局部 target 强，但跨层漂移严重 |
| BFT | 变安全，但变弱 |
| FCAdam / FLD | 短程下降强，但 rank / phi / role dynamics 不闭环 |

v4.7 的新点是：

```text
把学习阶段和几何阶段分离，
并用 teacher-preserving functional projection 连接它们。
```

---

## 4. 主要方法族

### 4.1 PSFT-A：FCAdam Learn -> Sobolev Distill

这是最小可行版本。

```text
Phase I:
  FCAdam-dataSob 或 FLD-AdanLite
  no / low Sobolev
  role-share controller 开启

Phase II:
  freeze teacher logits/features
  Sobolev consolidation with KL + feature preservation

Phase III:
  low-lr TFU / FNG fine-tune
```

推荐代号：

```text
PSFT-A-FCAdamDistill
PSFT-A-AdanDistill
```

### 4.2 PSFT-B：AdamW Teacher -> Functional Distillation

这是诊断版本，不作为最终 claim，但非常重要。它用 PureKAN-AdamW 跑出 teacher checkpoint，然后问：

```text
functional consolidation 能否在不使用 AdamW 继续训练的情况下，
复现并平滑 AdamW 学到的 PureKAN function？
```

公式：

$$
\min_\theta
\quad
\tau_z\operatorname{KL}(p_{AdamW}\|p_\theta)
+
\tau_h\sum_l\|h_l-h_l^{AdamW}\|^2
+
\lambda_S\sum_l\|a_l\|_S^2.
$$

如果这个也失败，说明当前 basis / functional projection 本身有问题。

如果这个成功，但 PSFT-A 失败，说明：

```text
functional projection 可以；
functional representation learning 还不行。
```

### 4.3 PSFT-C：Alternating Learn-Consolidate Cycles

不是只做一次 Phase I -> Phase II，而是循环：

```text
for cycle in 1..K:
  learn for T_learn steps
  consolidate for T_smooth steps
  validate holdout / rank / phi
```

每个 cycle 都用当前模型作为 teacher。

这对应一种 operator splitting：

$$
\theta_{t+1/2}
=
\operatorname{LearnStep}(\theta_t)
$$

$$
\theta_{t+1}
=
\operatorname{SmoothProject}(\theta_{t+1/2}).
$$

### 4.4 PSFT-D：Role-Staged Learning

AdamW envelope 显示早期 input update share 约为 $0.68$，block share 约为 $0.21$，output share 约为 $0.10$。因此 Phase I 不应该让 output 或 block 抢走学习。

Role-staged version：

```text
Stage 1:
  input-heavy learning
  target share: input 0.65, block 0.25, output 0.10

Stage 2:
  block refinement
  target share: input 0.35, block 0.45, output 0.20

Stage 3:
  output / margin tuning
  target share: input 0.20, block 0.40, output 0.40

Stage 4:
  Sobolev consolidation
```

---

## 5. 总体成功标准

v4.7 不是为了证明某个 proposal one-step 好，而是要真正挑战 PureKAN-AdamW。

最终必须同时比较：

```text
PureKAN-AdamW
PureKAN-UFULL / D0
D6-allTaskAware
FCAdam-dataSob
FLD-AdanLite
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
```

### 5.1 最终 5-seed gate

对 MNIST / Fashion-MNIST / KMNIST，候选必须满足：

$$
\operatorname{Acc}_{candidate}
\geq
\operatorname{Acc}_{PureKAN-AdamW}
-
0.5\%
$$

并且至少一个数据集超过 AdamW：

$$
\operatorname{Acc}_{candidate}
>
\operatorname{Acc}_{PureKAN-AdamW}.
$$

同时：

$$
\text{AUC improvement vs D6} > 0.
$$

$$
\text{ECE reduction vs AdamW} > 0.
$$

$$
\phi/A_{\text{final}} \leq 1.0.
$$

$$
J/A_{\text{final}} \leq 1.0.
$$

### 5.2 Strong gate

如果要 claim PureKAN functional optimizer solved：

$$
\operatorname{Acc}_{candidate}
\geq
\max(
\operatorname{Acc}_{PureKAN-AdamW},
\operatorname{Acc}_{Hybrid-DGKAN-UFULL},
\operatorname{Acc}_{MLP-AdamW}
).
$$

并且：

$$
\text{final geometry better than PureKAN-AdamW}.
$$

---

## 6. 实验阶段设计

## P0：Implementation Smoke and Invariants

### 目标

确认 PSFT 的三阶段机制、teacher snapshot、functional coordinate、distillation loss、role budget、rollback 和 logging 都没有问题。

### 方法

```text
datasets:
  MNIST smoke
  Fashion-MNIST smoke
  KMNIST smoke

train / val / test:
  512 / 128 / 128

epochs:
  1

seeds:
  0
```

### 必跑方法

```text
PureKAN-AdamW
D0-allFullSobolev
FCAdam-dataSob
FLD-AdanLite
PSFT-A-FCAdamDistill-smoke
PSFT-A-AdanDistill-smoke
PSFT-B-AdamTeacherDistill-smoke
PSFT-C-Alternating-smoke
```

### 必须记录

```text
strict_purekan_nonKAN_params
coefficient_coverage
teacher_snapshot_ok
teacher_logits_finite
teacher_features_finite
distill_loss_finite
phase_trace
role_share_trace
functional_coordinate_roundtrip
rollback_error
NaN / Inf count
step_time
memory_peak
```

### P0 通过条件

$$
\text{run failures}=0.
$$

$$
\text{nonKAN params}=0.
$$

$$
\text{coefficient coverage}=1.
$$

$$
\text{all losses finite}.
$$

---

## P1：Phase-I Learning Dynamics Audit

### 目标

确认 Phase I 是否真的能形成任务表示，而不是只降低短程 holdout loss。

### 方法

```text
PureKAN-AdamW
FCAdam-dataSob
FLD-AdanLite
FLD-WinLite
FLD-LyapunovRestart
PSFT-A-FCAdam-PhaseI-only
PSFT-A-Adan-PhaseI-only
PSFT-D-role-staged-PhaseI-only
```

### 步长

```text
steps:
  1, 5, 20, 50, 100
```

### 记录指标

#### 任务轨迹

```text
train_loss_delta
holdout_loss_delta
val_loss_delta
train_acc
val_acc
test_acc_probe
```

#### 表示轨迹

```text
effective_rank_input
effective_rank_block_mean
effective_rank_output_input
class_centroid_separation
within_class_variance
between_class_variance
margin_mean
margin_p10
margin_p50
margin_p90
```

#### role update

```text
role_update_share_input
role_update_share_block
role_update_share_output
role_update_norm_input
role_update_norm_block
role_update_norm_output
role_update_over_param_input
role_update_over_param_block
role_update_over_param_output
```

#### 函数几何

```text
phi_prime_p95
phi_prime_max
Jacobian_condition
Sobolev_energy
basis_occupancy_entropy
dead_basis_fraction
out_of_grid_fraction
```

#### AdamW envelope alignment

```text
holdout_delta_over_adamw
rank_ratio_over_adamw
margin_ratio_over_adamw
phi_ratio_over_adamw
role_share_L1_distance_to_adamw
function_R2_to_adamw
function_cos_to_adamw
```

### 可视化

```text
p1_holdout_delta_vs_step.svg
p1_rank_margin_vs_step.svg
p1_phi_vs_step.svg
p1_role_share_stacked_area.svg
p1_role_share_distance_to_adamw.svg
p1_function_R2_to_adamw.svg
p1_loss_rank_phi_3axis.svg
```

### P1 通过条件

候选进入 P2 必须满足：

$$
\text{holdout}_{100}/\text{AdamW}_{100} > 0.75.
$$

$$
\text{role share distance to AdamW} < 0.35.
$$

$$
\text{margin}_{p10}\text{ improves}.
$$

$$
\phi/A \leq 1.7
$$

其中 $\phi/A$ 在 P1 不再要求小于 $1.0$，因为 P1 是表示形成阶段。

---

## P2：Geometry Consolidation-Only Test

### 目标

这是 v4.7 的关键诊断。它回答：

```text
如果已经有一个学得不错的 teacher，
functional Sobolev projection 能不能在保持 logits/features 的同时改善 geometry？
```

### Teacher 来源

```text
Teacher A:
  PureKAN-AdamW checkpoint at step 100

Teacher B:
  PSFT-A Phase-I checkpoint at step 100

Teacher C:
  FCAdam-dataSob checkpoint at step 100

Teacher D:
  FLD-AdanLite checkpoint at step 100
```

### Student 初始化

```text
same checkpoint as teacher
then apply consolidation phase
```

### Consolidation objective

$$
\mathcal L_{\text{consolidate}}
=
\tau_z
\operatorname{KL}(p_T\|p_\theta)
+
\tau_h
\sum_l
\|h_l^T-h_l\|_2^2
+
\lambda_{CE}
\mathcal L_{CE}
+
\lambda_S
\sum_l
\|a_l\|_{S_l}^2.
$$

### Sweep

```text
tau_z in {0.5, 1.0, 2.0}
tau_h in {0.1, 0.5}
lambda_S in {1e-4, 3e-4, 1e-3, 3e-3}
consolidation_steps in {20, 50, 100}
```

### 必须记录

```text
teacher_acc
student_acc_before
student_acc_after
KL_teacher_student
feature_mse_input
feature_mse_block_mean
feature_mse_output
phi_before
phi_after
J_before
J_after
Sobolev_energy_before
Sobolev_energy_after
margin_before
margin_after
rank_before
rank_after
```

### 关键可视化

```text
p2_geometry_reduction_vs_kl.svg
p2_acc_retention_vs_phi_reduction.svg
p2_feature_mse_vs_acc_drop.svg
p2_teacher_student_logit_correlation.svg
p2_sobolev_energy_curve.svg
```

### P2 通过条件

$$
\operatorname{Acc}_{after}
\geq
\operatorname{Acc}_{before}
-
0.5\%.
$$

$$
\phi_{after}
\leq
0.85\phi_{before}.
$$

$$
J_{after}
\leq
0.85J_{before}.
$$

$$
\operatorname{KL}(p_T\|p_\theta)
\leq
0.05.
$$

如果 P2 失败，说明：

```text
functional geometry projection 本身无法保持学到的函数；
PureKAN functional optimizer 需要重新考虑 basis / architecture。
```

如果 P2 成功，进入 P3。

---

## P3：One-Cycle PSFT Micro-Run

### 目标

验证一次完整的：

```text
Phase I learn
Phase II consolidate
Phase III fine-tune
```

能不能形成比 D6/FCAdam 更稳定的 trajectory。

### 配置

```text
train / val / test = 6000 / 1000 / 1000
seeds = 0,1,2
datasets = MNIST, Fashion-MNIST, KMNIST
```

### 方法

```text
PureKAN-AdamW
D6-allTaskAware
FCAdam-dataSob
FLD-AdanLite
PSFT-A-FCAdamDistill
PSFT-A-AdanDistill
PSFT-D-role-staged-FCAdamDistill
```

### Phase schedule

```text
Phase I:
  100 steps learn

Phase II:
  50 steps consolidate

Phase III:
  50 steps fine-tune
```

### 记录

```text
phase_id
step_in_phase
train_loss
val_loss
test_probe_acc
holdout_delta
KL_to_phaseI_teacher
feature_MSE_to_phaseI_teacher
role_share
rank
margin
phi
J
ECE
NLL
```

### 可视化

```text
p3_phase_colored_loss_curve.svg
p3_phase_colored_accuracy_curve.svg
p3_phase_colored_phi_curve.svg
p3_rank_margin_phi_phase_plot.svg
p3_role_share_by_phase.svg
p3_distill_preservation_curve.svg
```

### P3 通过条件

候选进入 P4 必须满足：

$$
\text{acc gap vs PureKAN-AdamW} < 2\%.
$$

$$
\text{AUC vs D6} > 0.
$$

$$
\phi_{\text{final}}/A \leq 1.15.
$$

$$
\text{ECE reduction vs AdamW} \geq 0.
$$

---

## P4：Alternating PSFT Cycles

### 目标

一次 Learn -> Consolidate 可能不够。P4 验证多轮 operator splitting 是否能更接近 AdamW 的长期轨迹。

### 方法

```text
PSFT-C-FCAdam cycles
PSFT-C-Adan cycles
PSFT-C-role-staged cycles
```

### Schedules

```text
Cycle short:
  learn 50 steps
  consolidate 25 steps
  repeat 4 cycles

Cycle medium:
  learn 100 steps
  consolidate 50 steps
  repeat 3 cycles

Cycle long:
  learn 150 steps
  consolidate 75 steps
  repeat 2 cycles
```

### 记录

```text
cycle_id
phase_id
cycle_holdout_delta
cycle_acc_delta
cycle_phi_delta
cycle_rank_delta
cycle_margin_delta
teacher_preservation_KL
feature_preservation_MSE
consolidation_accept_rate
```

### 可视化

```text
p4_cycle_dashboard.svg
p4_cycle_rank_phi_tradeoff.svg
p4_cycle_acc_vs_phi.svg
p4_consolidation_effect_per_cycle.svg
```

### P4 通过条件

至少一个 candidate 满足：

$$
\text{acc gap vs PureKAN-AdamW} < 1.5\%.
$$

$$
\phi/A_{\text{final}} < 1.05.
$$

$$
\text{holdout AUC vs D6} > 0.
$$

并且不能出现：

```text
chance accuracy collapse
KL explosion
rank collapse
```

---

## P5：Role-Staging Ablation

### 目标

验证 AdamW envelope 中的 input-heavy update 是否是 PureKAN 训练的关键。

### 对照

```text
No role controller
AdamW-envelope role controller
Input-heavy only
Block-heavy only
Output-heavy only
Adaptive role controller
```

### Role share targets

```text
AdamW-envelope:
  input = 0.68
  block = 0.21
  output = 0.11

Input-heavy:
  input = 0.75
  block = 0.15
  output = 0.10

Block-heavy:
  input = 0.30
  block = 0.55
  output = 0.15

Output-heavy:
  input = 0.25
  block = 0.25
  output = 0.50
```

### 记录

```text
actual_role_share
role_share_error
role_specific_loss_proxy
role_specific_update_norm
role_specific_phi
role_specific_rank_effect
role_specific_margin_effect
```

### 判定

如果 AdamW-envelope role controller 明显好于 no-controller，说明：

```text
PureKAN functional update 的问题之一是 role-wise credit allocation。
```

如果没有差异，则 role share 不是主瓶颈。

---

## P6：Consolidation Objective Ablation

### 目标

理解 Phase II 中哪些项是必要的。

### 对照

```text
KL only
feature MSE only
KL + feature MSE
KL + CE
KL + feature MSE + CE
KL + feature MSE + CE + Sobolev
```

### 记录

```text
acc_retention
phi_reduction
J_reduction
KL_after
feature_MSE_after
margin_after
rank_after
```

### 可视化

```text
p6_objective_ablation_radar.svg
p6_kl_feature_phi_scatter.svg
p6_acc_retention_bar.svg
```

### 判定

理想情况：

```text
KL + feature + Sobolev 保持 acc，同时降低 phi/J。
```

如果 KL only 保持 logits 但 feature drift 大，说明 hidden representation preservation 必须加入。  
如果 feature only 保持 rank 但 logits 退化，说明 output constraint 必须加入。

---

## P7：3-Seed Full-Budget Candidate Selection

### 目标

用完整训练预算筛选最终候选。

### 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### 方法

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
D6-allTaskAware
FCAdam-dataSob
FLD-AdanLite
Best PSFT-A
Best PSFT-C
Best PSFT-D
```

### seeds

```text
0, 1, 2
```

### 记录

```text
test_acc
val_loss_auc
ECE
NLL
phi_prime_p95
Jacobian_condition
effective_rank
margin_p10 / mean / p90
role_share_auc
phase_transition_metrics
step_time
memory_peak
```

### P7 通过条件

候选进入 5-seed confirm 必须满足：

$$
\text{mean acc gap vs PureKAN-AdamW} < 1.0\%.
$$

$$
\text{val AUC vs D6} > 0.
$$

$$
\phi/A < 1.0.
$$

$$
\text{ECE reduction vs AdamW} > 0.
$$

---

## P8：5-Seed Confirm

### 目标

确认 P7 survivor 是否稳定。

### seeds

```text
0,1,2,3,4
```

### 方法

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
Best PSFT candidate
```

### 统计

```text
mean
std
paired delta
bootstrap 95% CI
failure count by seed
```

### 通过条件

$$
\Delta acc_{\text{candidate}-AdamW}
\geq
-0.5\%.
$$

$$
CI_{\Delta acc}^{lo}
>
-1.0\%.
$$

$$
\Delta AUC > 0.
$$

$$
\Delta ECE > 0.
$$

$$
\phi/A < 1.0.
$$

---

## P9：10-Seed Final Confirm

只对 P8 通过者运行。

### seeds

```text
0..9
```

### 最终 claim 条件

弱 claim：

```text
PureKAN-PSFT matches PureKAN-AdamW accuracy while improving geometry / calibration.
```

medium claim：

```text
PureKAN-PSFT slightly exceeds AdamW on at least one dataset and matches on others.
```

strong claim：

```text
PureKAN-PSFT exceeds PureKAN-AdamW, Hybrid-DGKAN-UFULL, and MLP-AdamW on all MNIST-family datasets.
```

---

## 7. 指标总表

### 7.1 任务指标

```text
train_loss
val_loss
test_loss
train_acc
val_acc
test_acc
val_loss_auc
val_acc_auc
NLL
ECE
classwise_acc
worst_class_acc
```

### 7.2 表示学习指标

```text
effective_rank_input
effective_rank_block_l
effective_rank_output_input
class_centroid_distance
within_class_variance
between_class_variance
separation_ratio
margin_mean
margin_p10
margin_p50
margin_p90
```

### 7.3 functional geometry

```text
phi_prime_p95
phi_prime_max
Jacobian_condition
Sobolev_energy
curvature_energy
basis_occupancy_entropy
dead_basis_fraction
out_of_grid_fraction
```

### 7.4 teacher preservation

```text
KL_teacher_student
feature_MSE_input
feature_MSE_block_mean
feature_MSE_output
logit_cos_teacher_student
logit_R2_teacher_student
prediction_agreement
```

### 7.5 optimizer dynamics

```text
phase_id
cycle_id
role_update_share
update_norm_by_role
update_over_param_by_role
function_delta_norm
function_delta_R2_to_teacher
AdamW_envelope_distance
accepted_step_ratio
backtrack_count
restart_count
step_time_ms
memory_peak_mb
```

---

## 8. 可视化总表

必须生成以下图：

```text
figures/p1_holdout_delta_vs_step.svg
figures/p1_rank_margin_phi_trajectory.svg
figures/p1_role_share_stacked_area.svg
figures/p1_function_alignment_to_adamw.svg

figures/p2_geometry_reduction_vs_acc_retention.svg
figures/p2_teacher_student_KL_curve.svg
figures/p2_feature_preservation_curve.svg
figures/p2_phi_reduction_curve.svg

figures/p3_phase_colored_loss_curve.svg
figures/p3_phase_colored_acc_curve.svg
figures/p3_phase_colored_phi_curve.svg
figures/p3_rank_margin_phi_phase_plot.svg

figures/p4_cycle_dashboard.svg
figures/p4_cycle_acc_phi_tradeoff.svg
figures/p4_role_share_by_cycle.svg

figures/p5_role_share_ablation.svg
figures/p6_consolidation_objective_radar.svg
figures/p7_candidate_scorecard.svg
figures/p8_paired_delta_ci.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 9. 失败解释表

每个失败 run 必须归到一个主类型：

| code | failure type | 判断标准 |
|---|---|---|
| F1 | Phase-I under-learning | holdout descent < 0.75 AdamW |
| F2 | role allocation mismatch | role share L1 distance > 0.35 |
| F3 | rank collapse | effective rank below 0.7 AdamW envelope |
| F4 | margin failure | margin p10 不升或 class separation 不升 |
| F5 | geometry explosion | phi/A > 1.7 in Phase I or > 1.0 final |
| F6 | consolidation destroys function | KL > 0.05 or acc drop > 0.5% after Phase II |
| F7 | feature drift | hidden feature MSE too high after consolidation |
| F8 | no final advantage | trajectory OK but final acc/ECE/AUC not competitive |
| F9 | compute failure | step time > 2x AdamW without accuracy benefit |

---

## 10. 本轮之后的决策规则

### 情况 A：P2 consolidation-only 失败

结论：

```text
Current functional geometry projection cannot preserve learned PureKAN functions.
Stop optimizer search and inspect basis / architecture / parameterization.
```

### 情况 B：P2 成功，但 P3/P4 失败

结论：

```text
Functional projection works, but functional representation learning still fails.
Next work should focus on Phase-I learning dynamics / role allocation.
```

### 情况 C：P4/P7 成功但 P8 不稳

结论：

```text
PSFT is a promising but unstable optimizer.
Need seed-stability controller and adaptive phase lengths.
```

### 情况 D：P8/P9 成功

结论：

```text
PureKAN functional optimizer is solved at MNIST-family scale.
Functional training requires phase separation:
task-learning phase + geometry consolidation phase.
```

---

## 11. 最终一句话

v4.7 的核心不是继续寻找更好的单步 update，而是验证：

$$
\boxed{
\text{PureKAN functional training 是否必须是}
\quad
\text{Learn} \rightarrow \text{Functional Smooth Projection} \rightarrow \text{Fine Tune}
}
$$

如果这个成立，我们就能解释过去所有现象：

```text
Sobolev-only:
  一开始就 smooth，所以学不起来。

FCAdam / FLD:
  能学，但 roughness / margin / rank 不受控。

BFT:
  很安全，但太弱。

PSFT:
  先让函数学会任务，再把函数投影回好的几何。
```

这才是下一步真正值得验证的方向。
