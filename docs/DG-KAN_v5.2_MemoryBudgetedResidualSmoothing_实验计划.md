# DG-KAN v5.2：Memory-Budgeted AB-RBF Residual Smoothing 实验计划

## 0. 当前结论：AB-RBF 是主线，强残差平滑验收器必须降级

v5.1 的实验结果给出了一个很清楚的边界：**AB-RBF edge 分解是正确方向，但强残差平滑验收器不适合进入训练主线**。这一轮不应该被解读成 AB-RBF 失败，而应该被解读成：我们已经找到了更合理的 PureKAN edge 坐标，但随后加入的 strong residual smoothing acceptor 太重、太复杂、重复计算过多，显存与时间成本没有换来对应的训练收益。

当前最重要的判断是：

$$
\boxed{\text{AB-RBF 继续作为 PureKAN 主 primitive；strong acceptor 只保留为离线诊断。}}
$$

v5.1 的 P0 显示，AB-RBF core smoke 通过，`nonKAN = 0`，edge/base/RBF coverage 都为 `1.0`，rollback 为 `0`。这说明 AB-RBF 本身没有破坏 strict PureKAN 的定义。P1 进一步确认 `ABRBF-linear+silu` 和 `ABRBF-silu` 是 architecture-positive，base path 被稳定使用，BaseOnly 仍然不足以单独替代 AB-RBF，尤其在 KMNIST 上。P2 eigenmode audit 也继续支持之前的解释：RBF-only 过去把低阶结构压在 high Sobolev eigen modes 上，而 AB-RBF 把一部分能量转移到 explicit base modes，降低了 RBF residual 的高模态压力。

但是 P3 到 P6 暴露了另一个问题：strong residual smoothing acceptor 太重。P3 的 residual smoothing audit 有大量 proposal/controller 组合，很多 top rows 在不同 controller 下给出完全相同或近似相同的 `acc drop / phiR red / curvR red / KL / score`，这说明复杂 controller 网格没有提供足够的额外决策信息。P4 的 accepted smoothing 可以大幅降低 residual geometry，比如 `curvR red` 接近 `0.86-0.98`，但多步 smoothing 的 accuracy drop 过大，不能通过 joint task gate。P5/P6 基本没有展开，说明强验收器没有形成可持续训练路径。

因此 v5.2 的设计目标不再是“更强 smoothing”，而是：

$$
\boxed{\text{在显存和时间预算内，找到可持续、轻量、低频触发的 residual smoothing。}}
$$

也就是说，v5.2 的问题不是“能不能把 RBF residual 压得更平滑”，而是：

$$
\boxed{\text{能不能用很小的训练成本，稳定改善 }\phi_{rbf}\text{ / curvature}_{rbf}\text{，且不伤任务性能。}}
$$

---

## 1. 对当前代码实现的审视

### 1.1 `RBFDense` 的问题已经不是唯一核心，但仍然是基础边界

上传的 `dgkan_core(5).py` 里可以看到，基础 `RBFDense` 仍然是固定 centers、固定 width，只训练 `coeff` 的结构：

```python
centers = torch.linspace(-2.5, 2.5, basis_count)
self.register_buffer("centers", centers)
self.width = float((centers[1] - centers[0]).abs() * 1.4)
self.coeff = nn.Parameter(...)
```

这说明基础 RBF-only PureKAN 的确是 fixed-grid RBF coefficient-only 网络。v4.9/v5.0 已经说明，这种坐标对 PureKAN functional training 很不友好：低阶结构被迫由 RBF 高模态承担，而 Sobolev functional update 又天然惩罚这些高模态。

v5.1 结果说明 AB-RBF 是修正这个问题的正确方向。但在上传的 `dgkan_core(5).py` 中，我没有看到稳定的 core-level `ABRBFDense`、`edge_named_params()`、`base_named_params()`、`rbf_residual_named_params()` 定义。这和 v5.1 replay 中“Added core ABRBFDense”的描述存在一个实现版本不一致风险。因此 v5.2 的 P0 必须首先确认：当前真正运行的代码里，AB-RBF 是否已经成为 core primitive，而不是只在某个 runner 中临时定义。

如果 AB-RBF 仍然只在 runner 中临时定义，那么 v5.2 不能直接进入训练实验。必须先把它提升为 core 模块，否则后续所有 audit 都容易因为参数分组、coverage、geometry 统计、rollback、smoothing proposal 对象不一致而产生假结论。

### 1.2 AB-RBF 必须拆分 base 与 residual 两套参数和两套几何

AB-RBF 的 edge 应写成：

$$
f_{ij}(x)=b_{ij}+w_{ij}x+u_{ij}\operatorname{silu}(x)+\sum_k c_{ijk}B_k(x).
$$

其中：

```text
base path:
  b_ij, w_ij, u_ij

RBF residual path:
  c_ijk
```

v5.1 结果表明 base path 在任务中真实参与，base/RBF norm 大约在 `0.16-0.67` 范围内，且 BaseOnly 不足以独立解决任务。因此 v5.2 必须避免两种错误：

第一，不能把 base path 当成普通 non-KAN 参数，因为那会破坏 strict PureKAN 叙事。base path 应该是 KAN edge 内部的低阶通道。

第二，不能把 base path 和 RBF residual path 用同一个 Sobolev 几何指标评价。线性项天然有常数导数，不能把它当成坏几何；真正需要控制的是 RBF residual 的粗糙性。

因此 v5.2 中几何指标必须分解为：

$$
\phi_{total},\quad \phi_{base},\quad \phi_{rbf},
$$

$$
\operatorname{curv}_{total},\quad \operatorname{curv}_{base},\quad \operatorname{curv}_{rbf},
$$

以及：

$$
\|c\|_{S_{rbf}}^2.
$$

默认的 smoothing 目标只作用于 $c_{ijk}$，不作用于 $b,w,u$。

### 1.3 强验收器重在哪里

v5.1 的 strong acceptor 重在三个方面。

第一，它同时追踪 teacher logits、hidden features、margin、KL、activation drift、geometry reduction 和 temporary apply/rollback。这让每个 smoothing proposal 都需要额外 forward、缓存 teacher、计算多个约束、再进行 rollback。

第二，它用 proposal × controller 网格。P3 中 S0-S7 proposals 乘 C0-C5 controllers 导致大量重复计算，而许多 controller 的结果完全相同，说明决策冗余很高。

第三，它把 smoothing 设计成训练中的强过程，但 P4 结果显示多步 accepted smoothing 的 `acc drop` 太大。也就是说，强 smoothing 虽然能降 geometry，但训练过程中不可持续。

v5.2 因此要把 smoothing 从“训练主循环里的强验收器”改成“低频、轻量、受显存预算约束的维护步骤”。

---

## 2. v5.2 的核心假设

v5.2 不再提出一个新的大 optimizer，而是测试一个更克制的假设：

$$
\boxed{\text{AB-RBF task learning 由 AdamW 完成；RBF residual smoothing 低频、轻量、只做几何维护。}}
$$

具体来说，训练主路径是：

```text
AB-RBF task phase:
  edge/base/RBF 参数用 AdamW 或 edge-only AdamW 学任务

light residual smoothing event:
  只对 RBF residual coeff 做轻量 smoothing proposal
  只使用 logit/KL/score gate
  不默认 hidden trust
  不默认 margin trust
  不默认 exact nullspace projection

refresh phase:
  少量 AdamW / base+RBF task step 恢复 task boundary
```

这种设计接受一个现实：**AdamW 是当前唯一稳定学习 PureKAN / AB-RBF 任务表示的路径**。functional smoothing 不应该再抢主任务学习，而应该成为一个 geometry maintenance operator。

因此 v5.2 的目标不是打败 AdamW 的初期训练速度，而是证明：

$$
\boxed{\text{AB-RBF + light smoothing 可以接近 AdamW accuracy，同时更好地控制 RBF residual geometry。}}
$$

---

## 3. v5.2 的默认方法定义

### 3.1 ABRBF-AdamW baseline

这是主 baseline。它训练整个 AB-RBF edge：

$$
\theta_{edge}=(\theta_{base}, c_{rbf}).
$$

更新为：

$$
\theta_{edge,t+1}=\operatorname{AdamW}(\theta_{edge,t},\nabla_{\theta}L).
$$

它回答：AB-RBF 架构在没有 smoothing 的情况下能达到什么任务上限。

### 3.2 ABRBF-LightSmooth-logitOnly

这是 v5.2 的核心候选。训练大部分时间仍是 AdamW，每隔若干 epoch 或事件触发一次 smoothing。smoothing 只更新 RBF residual coefficient：

$$
c \leftarrow c + \eta d_{smooth}.
$$

默认 proposal：

$$
d_{smooth}=\operatorname{Laplacian}(c)-\alpha c.
$$

或者：

$$
d_{smooth}=-\nabla_c R_{rbf}(c),
$$

其中：

$$
R_{rbf}(c)=\|c\|_{S_{rbf}}^2.
$$

验收只看轻量约束：

$$
\operatorname{KL}(p_T,p_S)<\epsilon_{KL},
$$

$$
\frac{\|z_S-z_T\|}{\|z_T\|+\epsilon}<\epsilon_z,
$$

$$
\Delta Acc_{holdout}\geq -\epsilon_{acc},
$$

并要求至少一个 residual geometry 指标改善：

$$
\Delta\phi_{rbf}>0
\quad\text{or}\quad
\Delta\operatorname{curv}_{rbf}>0.
$$

### 3.3 ABRBF-LightSmooth-scoreOnly

这个版本不做多重 trust controller，只计算一个 score：

$$
\operatorname{score}
=
\omega_\phi\Delta\phi_{rbf}
+
\omega_c\Delta\operatorname{curv}_{rbf}
+
\omega_s\Delta\|c\|_{S_{rbf}}^2
-
\omega_{KL}\operatorname{KL}
-
\omega_z\operatorname{drift}_z
-
\omega_a\max(0,-\Delta Acc_{holdout}).
$$

只有当 score 为正且 memory/time gate 通过时接受。这个版本的价值是验证：复杂 controller 是否可以被一个简单 score 取代。

### 3.4 StrongSmooth-reference

保留强验收器作为离线参考，不进入训练默认。它只用于说明 light smoother 的 geometry reduction 与 memory/time trade-off 和 strong smoother 相比损失多少。

---

## 4. 实验阶段总览

v5.2 分成九个阶段。每个阶段都有明确停止条件，避免再次形成大规模无效 sweep。

```text
P0: Core implementation and memory smoke
P1: Single smoothing event audit
P2: One-cycle train -> light smooth -> refresh
P3: Event-driven multi-cycle training
P4: 3-seed candidate selection
P5: 5-seed confirm
P6: 10-seed final confirm
P7: CIFAR-small memory / scaling precheck
P8: failure diagnosis and final decision
```

---

## 5. P0：Core implementation and memory smoke

### 5.1 目的

P0 不看 accuracy，专门确认两件事：第一，AB-RBF 已经是 core-level primitive；第二，light smoothing 的显存 / 时间开销在预算内。

### 5.2 必跑方法

```text
ABRBF-AdamW
ABRBF-LightSmooth-logitOnly-smoke
ABRBF-LightSmooth-scoreOnly-smoke
ABRBF-StrongSmooth-reference-smoke
RBFOnly-AdamW, diagnostic
```

### 5.3 必查代码不变量

每个 run 必须记录：

```text
learnable_nonkan_params
edge_param_count
base_param_count
rbf_residual_param_count
edge_coverage
base_coverage
rbf_residual_coverage
rollback_max_abs_error
teacher_snapshot_error
smoothing_applies_to_base, must be 0 by default
smoothing_applies_to_rbf, must be 1
```

核心 gate：

$$
\operatorname{nonKAN}=0,
$$

$$
\operatorname{edge\ coverage}=1.0,
$$

$$
\operatorname{base\ coverage}=1.0,
$$

$$
\operatorname{rbf\ coverage}=1.0,
$$

$$
\operatorname{rollback\ error}<10^{-8}.
$$

### 5.4 显存指标

P0 必须新增显存指标：

```text
peak_cuda_allocated_mb
peak_cuda_reserved_mb
teacher_cache_mb
proposal_temp_mb
smoothing_forward_count
smoothing_backward_count
smoothing_time_sec
training_step_time_ms
memory_ratio_vs_ABRBF_AdamW
time_ratio_vs_ABRBF_AdamW
```

显存 gate：

$$
\operatorname{memory\ ratio}\leq1.25.
$$

时间 gate：

$$
\operatorname{amortized\ time\ ratio}\leq1.20.
$$

如果 light smoother 超过这两个 gate，不能进入 P1。strong reference 可以超过，但必须标记为 offline-only。

### 5.5 P0 可视化

P0 输出三张图。

第一张是 memory dashboard：横轴为方法，纵轴为 peak allocated / reserved MB。用堆叠条显示 teacher cache 和 proposal temp memory。

第二张是 time dashboard：横轴为方法，纵轴为 training step time 和 smoothing time。要画出 amortized time ratio。

第三张是 coverage dashboard：edge/base/RBF coverage 三个柱状图，必须全部为 1.0。

---

## 6. P1：Single smoothing event audit

### 6.1 目的

P1 从已经训练好的 ABRBF-AdamW teacher 出发，只做一次 smoothing。它回答：在单次事件中，light smoother 能否用很小 task drift 换来足够 residual geometry reduction。

### 6.2 Teacher

默认 teacher：

```text
PureKAN-ABRBF-linear+silu-AdamW
```

备选 teacher：

```text
PureKAN-ABRBF-silu-AdamW
PureKAN-ABRBF-linear-AdamW
```

只有当 linear+silu 在某个数据集明显不稳定时，才使用备选。

### 6.3 Smoothing candidates

P1 只允许小矩阵：

```text
S0-laplacian + logitOnly
S0-laplacian + scoreOnly
S1-sobolev-grad + logitOnly
S1-sobolev-grad + scoreOnly
StrongSmooth-reference
```

不要在 P1 继续跑 S0-S7 × C0-C5。

### 6.4 记录指标

任务保持：

```text
teacher_acc
student_acc_after_smooth
acc_drop
teacher_val_loss
student_val_loss_after_smooth
KL_teacher_student
logit_relative_drift
margin_mean_before
margin_mean_after
margin_p10_before
margin_p10_after
```

残差几何：

```text
phi_rbf_before
phi_rbf_after
phi_rbf_reduction
curvature_rbf_before
curvature_rbf_after
curvature_rbf_reduction
sobolev_rbf_before
sobolev_rbf_after
sobolev_rbf_reduction
high_eig_energy_before
high_eig_energy_after
high_eig_energy_reduction
```

base/RBF 分工：

```text
base_norm
rbf_norm
base_over_rbf
base_ablation_drop
rbf_ablation_drop
```

验收过程：

```text
accepted
rejected
reject_reason
backtrack_count
accepted_eta
score
```

显存/时间：

```text
peak_cuda_allocated_mb
peak_cuda_reserved_mb
smoothing_time_sec
memory_ratio_vs_teacher
time_ratio_vs_teacher
```

### 6.5 P1 通过标准

P1 单个数据集通过：

$$
\Delta Acc \leq 0.005,
$$

$$
\operatorname{KL}\leq0.005,
$$

$$
\operatorname{logit\ drift}\leq0.03,
$$

且满足：

$$
\Delta\phi_{rbf}\geq0.05
\quad\text{or}\quad
\Delta\operatorname{curv}_{rbf}\geq0.20.
$$

并且：

$$
\operatorname{memory\ ratio}\leq1.25.
$$

P1 all-dataset survivor 才能进入 P2。

### 6.6 P1 可视化

必须画四张图。

第一张是 geometry-cost Pareto：横轴 memory ratio，纵轴 curvature_rbf_reduction，点大小为 acc_drop，颜色为 proposal。

第二张是 function-preservation scatter：横轴 KL，纵轴 logit drift，点大小为 phi_rbf_reduction。

第三张是 residual eigmode plot：每个方法画 before/after high Sobolev eigen energy。

第四张是 acceptance waterfall：按 teacher -> proposal -> eta -> accept/reject 展示每一步为何被接受或拒绝。

---

## 7. P2：One-cycle train -> light smooth -> refresh

### 7.1 目的

P1 是单次 smoothing audit，P2 要验证 smoothing 是否能嵌入训练流程。流程是：

```text
Task train phase
-> light smooth event
-> refresh phase
```

关键问题是：smoothing 后是否可以通过短 refresh 恢复 task performance，同时保留一部分 residual geometry improvement。

### 7.2 方法

```text
ABRBF-AdamW, no smoothing
ABRBF-LightSmooth-logitOnly-oneCycle
ABRBF-LightSmooth-scoreOnly-oneCycle
ABRBF-StrongSmooth-reference-oneCycle, offline only
```

Task train phase 使用 AdamW。Refresh phase 也先使用 AdamW，不引入新的 functional optimizer，避免变量过多。

### 7.3 推荐训练流程

```text
train_steps_task = 100 or 1/3 full budget
smooth_events = 1
refresh_steps = 20 or 1/5 task phase
```

如果 full-budget 是 epoch 形式，则：

```text
task phase = 60% epochs
smooth event = once
refresh phase = 20% epochs
final fine tune = optional 20% epochs, no smoothing
```

### 7.4 记录指标

每个阶段都要记录：

```text
acc
val_loss
val_auc_proxy
ECE
NLL
margin_mean
margin_p10
phi_base
phi_rbf
phi_total
curvature_base
curvature_rbf
curvature_total
sobolev_rbf_norm
base_over_rbf
base_ablation_drop
rbf_ablation_drop
high_eig_energy
```

另外记录 smoothing event 前后：

```text
pre_smooth_acc
post_smooth_acc
post_refresh_acc
pre_smooth_phi_rbf
post_smooth_phi_rbf
post_refresh_phi_rbf
pre_smooth_curv_rbf
post_smooth_curv_rbf
post_refresh_curv_rbf
refresh_recovery_ratio
geometry_retention_ratio
```

定义：

$$
\operatorname{refresh\ recovery}
=
\frac{Acc_{refresh}-Acc_{smooth}}{Acc_{pre}-Acc_{smooth}+\epsilon}.
$$

$$
\operatorname{geometry\ retention}
=
\frac{\phi_{pre}-\phi_{refresh}}{\phi_{pre}-\phi_{smooth}+\epsilon}.
$$

### 7.5 P2 通过标准

P2 单个数据集通过：

$$
Acc_{final}\geq Acc_{AdamW}-0.005,
$$

$$
\Delta \phi_{rbf,final}\geq0.05
\quad\text{or}\quad
\Delta \operatorname{curv}_{rbf,final}\geq0.20,
$$

$$
\operatorname{refresh\ recovery}\geq0.80,
$$

$$
\operatorname{geometry\ retention}\geq0.50,
$$

$$
\operatorname{memory\ ratio}\leq1.25.
$$

只有 all-dataset survivor 进入 P3。

### 7.6 P2 可视化

必须画：

```text
train/smooth/refresh 三阶段 acc 曲线
val loss 曲线
phi_rbf 曲线
curvature_rbf 曲线
base/RBF norm 曲线
refresh recovery bar
geometry retention bar
```

还要画一张 two-axis 图：左轴 accuracy，右轴 phi_rbf，标出 smoothing event 的垂直线。这样可以直接看 smoothing 是否只是瞬时压几何，还是 refresh 后仍然保留改善。

---

## 8. P3：Event-driven multi-cycle training

### 8.1 目的

P2 只做 one-cycle。P3 测试低频 smoothing 是否能多次触发，而不造成 P4 那种 accumulated accuracy drop。

### 8.2 触发条件

默认不要固定每 epoch smoothing，而是 event-driven：

```text
if val_loss plateau and phi_rbf above budget:
    trigger smoothing
```

具体条件：

$$
\frac{|L_{val,t}-L_{val,t-k}|}{L_{val,t-k}+\epsilon}<\epsilon_{plateau},
$$

且：

$$
\phi_{rbf,t}>\phi_{budget}.
$$

推荐初始值：

```text
plateau_window = 3 evals
epsilon_plateau = 0.01
phi_budget = AdamW_teacher_phi_rbf * 0.95
max_smoothing_events = 3
minimum_gap_between_events = 3 epochs
```

如果没有 plateau，不触发 smoothing。

### 8.3 方法

```text
ABRBF-AdamW
ABRBF-LightSmooth-fixedInterval
ABRBF-LightSmooth-eventDriven
ABRBF-LightSmooth-eventDriven-refreshStrong
```

其中 `refreshStrong` 只加长 refresh，不加强 smoothing。

### 8.4 记录指标

除了 P2 指标，P3 额外记录：

```text
smoothing_event_count
accepted_event_count
rejected_event_count
event_epoch
event_reason
pre_event_val_loss
post_event_val_loss
pre_event_phi_rbf
post_event_phi_rbf
pre_event_acc
post_event_acc
recovery_epochs
accumulated_acc_drop
accumulated_phi_reduction
```

显存与时间要记录 amortized 版本：

```text
amortized_step_time_ms
amortized_memory_ratio
smoothing_time_total_sec
training_time_total_sec
smoothing_time_fraction
```

### 8.5 P3 通过标准

P3 通过：

$$
Acc_{final}\geq Acc_{ABRBF-AdamW}-0.005,
$$

$$
\Delta \phi_{rbf,final}\geq0.05
\quad\text{or}\quad
\Delta \operatorname{curv}_{rbf,final}\geq0.20,
$$

$$
\operatorname{smoothing\ time\ fraction}\leq0.15,
$$

$$
\operatorname{amortized\ memory\ ratio}\leq1.25,
$$

$$
\operatorname{accepted\ event\ count}\leq3.
$$

如果 fixedInterval 过但 eventDriven 不过，说明 trigger 设计错；如果 eventDriven 过而 fixedInterval 不过，说明低频自适应是必要的。

### 8.6 P3 可视化

必须画：

```text
epoch vs acc / val_loss / phi_rbf / curv_rbf
event markers over training curves
event reason stacked bar
accepted eta over events
acc drop per event
geometry gain per event
amortized time ratio over training
```

---

## 9. P4：3-seed candidate selection

### 9.1 目的

P4 才开始多 seed。P4 不再跑强验收器，只比较最轻 candidate。

### 9.2 方法

```text
ABRBF-AdamW
ABRBF-LightSmooth-best-from-P3
RBFOnly-AdamW
BaseOnly-linear diagnostic
BaseOnly-silu diagnostic
```

如果 P3 没有 survivor，则 P4 不运行。

### 9.3 datasets

```text
MNIST
Fashion-MNIST
KMNIST
```

### 9.4 seeds

```text
seeds = 0, 1, 2
```

### 9.5 P4 通过标准

对每个数据集：

$$
\Delta Acc_{paired}\geq -0.005,
$$

$$
\Delta \phi_{rbf}\geq0.05
\quad\text{or}\quad
\Delta \operatorname{curv}_{rbf}\geq0.20,
$$

$$
\Delta ECE\geq -0.01,
$$

$$
\operatorname{time\ ratio}\leq1.20,
$$

$$
\operatorname{memory\ ratio}\leq1.25.
$$

P4 all-dataset survivor 才进入 P5。

### 9.6 P4 可视化

```text
paired acc delta by seed
paired phi_rbf reduction by seed
paired curvature_rbf reduction by seed
accuracy-geometry Pareto plot
memory-time Pareto plot
base/RBF ablation drop comparison
```

---

## 10. P5：5-seed confirm

### 10.1 目的

P5 确认 P4 survivor 是否稳定。P5 不再引入新方法。

### 10.2 方法

```text
ABRBF-AdamW
ABRBF-LightSmooth-best
RBFOnly-AdamW
Hybrid-DGKAN-UFULL-f085, optional reference
```

### 10.3 seeds

```text
seeds = 0,1,2,3,4
```

### 10.4 记录统计

P5 必须输出：

```text
mean
std
paired delta
bootstrap CI95
failure count
memory mean/std
time mean/std
```

主要 paired deltas：

```text
acc_delta_vs_ABRBF_AdamW
val_auc_delta_vs_ABRBF_AdamW
ECE_delta_vs_ABRBF_AdamW
phi_rbf_reduction_vs_ABRBF_AdamW
curv_rbf_reduction_vs_ABRBF_AdamW
memory_ratio_vs_ABRBF_AdamW
time_ratio_vs_ABRBF_AdamW
```

### 10.5 P5 通过标准

P5 通过要求：

$$
CI95(\Delta Acc)_{lo}\geq -0.01,
$$

$$
\operatorname{mean}(\Delta \phi_{rbf})\geq0.05
\quad\text{or}\quad
\operatorname{mean}(\Delta\operatorname{curv}_{rbf})\geq0.20,
$$

$$
\operatorname{mean}(\operatorname{memory\ ratio})\leq1.25,
$$

$$
\operatorname{mean}(\operatorname{time\ ratio})\leq1.20.
$$

如果 P5 只在 MNIST/Fashion 过而 KMNIST 不过，需要记录为 dataset-specific smoothing，不允许 claim unified smoothing。

---

## 11. P6：10-seed final confirm

### 11.1 目的

P6 是最终 confirm，只在 P5 过线后运行。

### 11.2 方法

```text
ABRBF-AdamW
ABRBF-LightSmooth-best
RBFOnly-AdamW
Hybrid-DGKAN-UFULL-f085 reference
```

### 11.3 seeds

```text
seeds = 0..9
```

### 11.4 最终可 claim 的结论

如果 P6 通过，可以 claim：

```text
AB-RBF PureKAN with memory-budgeted light residual smoothing preserves AdamW-level accuracy while improving RBF-residual geometry under strict edge-only parameterization.
```

如果 P6 不过，但 P5 有局部信号，可以 claim：

```text
AB-RBF architecture is confirmed, but train-time residual smoothing remains an offline diagnostic / dataset-specific maintenance step.
```

如果 P4 都不过，则 claim：

```text
AB-RBF architecture is the useful contribution; residual smoothing should not be used in training with the current implementation.
```

---

## 12. P7：CIFAR-small memory / scaling precheck

### 12.1 目的

CIFAR-small 不用于直接做 full confirm。P7 只检查 light smoothing 是否可扩展到更大输入和 ConvStem/vision setting。

### 12.2 方法

```text
ConvStem-ABRBF-AdamW
ConvStem-ABRBF-LightSmooth-best
ConvStem-RBFOnly-AdamW
```

### 12.3 记录指标

```text
acc
val_auc
ECE
phi_rbf
curv_rbf
branch/base/RBF ratio
basis occupancy
out_of_grid_fraction
peak memory
time ratio
smoothing event count
```

### 12.4 判定

P7 通过条件较低：

$$
\operatorname{memory\ ratio}\leq1.30,
$$

$$
\operatorname{time\ ratio}\leq1.30,
$$

且 smoothing 不造成：

$$
\Delta Acc<-0.02.
$$

CIFAR-small 过 P7 只代表 scaling precheck 通过，不代表方法在 CIFAR 上已成立。

---

## 13. P8：Failure diagnosis

如果 P3 或 P4 没有 survivor，P8 必须输出失败诊断，而不是继续盲目调参。

### 13.1 失败类型

```text
F1: memory too high
F2: smoothing too weak
F3: smoothing task drift too high
F4: refresh cannot recover task
F5: geometry gain not retained
F6: base path absorbs all task, RBF residual becomes irrelevant
F7: KMNIST-specific failure
F8: implementation mismatch, AB-RBF not in core or coverage broken
```

### 13.2 每个失败类型的判定

F1：

$$
\operatorname{memory\ ratio}>1.25.
$$

F2：

$$
\Delta\phi_{rbf}<0.03
\quad\text{and}\quad
\Delta\operatorname{curv}_{rbf}<0.10.
$$

F3：

$$
\operatorname{KL}>0.005
\quad\text{or}\quad
\operatorname{logit\ drift}>0.03
\quad\text{or}\quad
\Delta Acc<-0.005.
$$

F4：

$$
\operatorname{refresh\ recovery}<0.80.
$$

F5：

$$
\operatorname{geometry\ retention}<0.50.
$$

F6：

$$
\operatorname{rbf\ ablation\ drop}<0.05
$$

or

$$
\operatorname{base\ share}>0.98
$$

for most of training.

F7：KMNIST fails while MNIST/Fashion pass.

F8：P0 invariant fails.

### 13.3 P8 输出

必须生成：

```text
failure_table.csv
failure_by_dataset.csv
failure_by_method.csv
failure_heatmap.png
recommendation.json
```

---

## 14. 总指标清单

### 14.1 任务指标

```text
test_acc
val_acc
train_loss
val_loss
val_auc_proxy
ECE
NLL
margin_mean
margin_p10
classwise_accuracy
paired_acc_delta
paired_auc_delta
```

### 14.2 AB-RBF 分解指标

```text
base_norm
rbf_norm
base_over_rbf
base_ablation_drop
rbf_ablation_drop
base_update_norm
rbf_update_norm
base_update_share
rbf_update_share
```

### 14.3 residual geometry 指标

```text
phi_base_p95
phi_rbf_p95
phi_total_p95
curvature_base
curvature_rbf
curvature_total
sobolev_rbf_norm
high_eig_energy
high_grad_energy
basis_occupancy_entropy
dead_basis_fraction
out_of_grid_fraction
```

### 14.4 smoothing 验收指标

```text
teacher_acc
student_acc
acc_drop
KL_teacher_student
logit_relative_drift
hidden_sketch_drift_optional
margin_relative_drift_optional
accepted
rejected
reject_reason
backtrack_count
accepted_eta
score
phi_rbf_reduction
curv_rbf_reduction
sobolev_rbf_reduction
geometry_retention
refresh_recovery
```

### 14.5 显存/时间指标

```text
peak_cuda_allocated_mb
peak_cuda_reserved_mb
teacher_cache_mb
proposal_temp_mb
smoothing_time_sec
training_time_sec
smoothing_time_fraction
step_time_ms
amortized_step_time_ms
memory_ratio_vs_baseline
time_ratio_vs_baseline
```

---

## 15. 必须生成的图

### 15.1 Memory dashboard

画四组柱状图：

```text
peak allocated MB
peak reserved MB
teacher cache MB
proposal temp MB
```

按方法分组。目标是证明 light smoother 相比 strong smoother 是否真的节省显存。

### 15.2 Geometry-cost Pareto

横轴：

$$
\operatorname{memory\ ratio}
$$

纵轴：

$$
\Delta\operatorname{curv}_{rbf}
$$

点大小：

$$
\Delta Acc
$$

颜色：proposal 类型。

这张图用于判断哪个 smoothing proposal 最划算。

### 15.3 Task-geometry training curve

每个 dataset 画：

```text
epoch vs test_acc
epoch vs val_loss
epoch vs phi_rbf
epoch vs curvature_rbf
```

用竖线标记 smoothing event。

### 15.4 Refresh recovery plot

横轴是方法，纵轴是：

$$
\operatorname{refresh\ recovery}
$$

和：

$$
\operatorname{geometry\ retention}.
$$

这张图判断 smoothing 后 refresh 是否能恢复任务性能。

### 15.5 Base/RBF role plot

画：

```text
epoch vs base_over_rbf
epoch vs base_ablation_drop
epoch vs rbf_ablation_drop
```

这张图用于检查 smoothing 是否让 RBF residual 退化成无任务贡献的几何装饰。

### 15.6 Failure heatmap

行是方法，列是失败类型：

```text
F1 memory
F2 weak smoothing
F3 task drift
F4 refresh fail
F5 geometry not retained
F6 RBF residual irrelevant
F7 KMNIST only
F8 implementation
```

每个格子填失败次数或比例。

---

## 16. 不再建议继续做的事情

v5.2 明确停止以下方向，除非 P8 失败分析强烈要求：

```text
S0-S7 × C0-C5 full grid
默认 hidden trust
默认 margin trust
exact dense NFS
strong projection in train loop
每个 epoch 都 smoothing
继续单纯调 Sobolev alpha/beta
继续调 branch_final_scale
继续把 base path 冻住
```

这些方向要么已经太重，要么已经证明会伤任务性能，要么没有提供足够新增信息。

---

## 17. 最终决策规则

v5.2 最终有四种可能结论。

### 17.1 结论 A：Light smoothing 成功

条件：P6 通过。

结论：

```text
AB-RBF is the PureKAN edge primitive, and memory-budgeted light residual smoothing is a practical geometry maintenance step.
```

此时可以进入 CIFAR-small / Rational/KAT scaling。

### 17.2 结论 B：Light smoothing 仅 dataset-specific 成功

条件：MNIST/Fashion 通过，KMNIST 不过，或者反过来。

结论：

```text
Residual smoothing is dataset-specific; AB-RBF remains the architecture contribution, but smoothing is not unified.
```

### 17.3 结论 C：Light smoothing 仅离线有效

条件：P1 单次 smoothing 通过，但 P2/P3 训练集成失败。

结论：

```text
Residual smoothing is a diagnostic/offline projection tool, not a train-time optimizer component.
```

### 17.4 结论 D：Light smoothing 不值得继续

条件：P1 就不过，或 memory/time gate 不过。

结论：

```text
AB-RBF should be trained with AdamW; residual geometry should be reported but not optimized with current smoothing tools.
```

---

## 18. 最后判断

v5.2 的价值不是再提出一个复杂 optimizer，而是把项目从“强 functional smoothing 越强越好”的误区中拉出来。当前证据支持：

$$
\boxed{\text{AB-RBF 是有效的 edge 坐标修复。}}
$$

但也支持：

$$
\boxed{\text{strong smoothing acceptor 作为训练主线太重、太贵、不可持续。}}
$$

因此 v5.2 的实验目标是验证一个更现实的方案：

$$
\boxed{\text{AB-RBF 用 AdamW 学任务，轻量 RBF residual smoothing 只做低频几何维护。}}
$$

如果这个方案仍然失败，就应该把 residual smoothing 从训练主线中移除，只保留 AB-RBF 作为 PureKAN 架构改进，并把 geometry control 作为离线分析或后处理工具。
