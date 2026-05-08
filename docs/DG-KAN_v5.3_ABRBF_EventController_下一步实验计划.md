# DG-KAN v5.3：AB-RBF 轻量残差平滑控制器与可持续训练计划

## 0. 当前阶段一句话结论

v5.2 不是没有进展。它把问题从：

```text
强 residual smoothing 显存太高、训练不可用
```

推进到了：

```text
轻量 residual smoothing 单次和 one-cycle 可用，但 multicycle controller 还不会在正确时间、用正确强度、配合正确 refresh 使用它。
```

所以当前不是 AB-RBF 失败，也不是 smoothing operator 失败，而是 **event scheduling / recovery / controller 失败**。

更精确地说：

$$
\boxed{
\text{AB-RBF 是正确的 PureKAN edge primitive；LightSmooth 是可用的局部几何维护算子；}
}
$$

$$
\boxed{
\text{但目前还没有一个稳定的训练期控制器把 LightSmooth 变成可持续 multicycle 方法。}
}
$$

v5.3 的目标不是再发明更强 smoothing，也不是继续堆 NFS / trust-region / hidden constraints，而是把 v5.2 已经验证的轻量算子接入一个真正可用的 **budgeted event controller**。

---

## 1. v5.2 到底有没有进展

### 1.1 有明确进展

v5.1 的强 residual smoothing acceptor 显存和时间都太重。v5.2 先解决了这个工程瓶颈。

在 P0 memory smoke 中，LightSmooth 的显存相比 ABRBF-AdamW 大约是：

```text
Fashion-MNIST: 1.1179x
KMNIST:        1.1179x
MNIST:         1.3206x, 但这是 P0 smoke 的 borderline outlier
```

而 StrongSmooth 是：

```text
Fashion-MNIST: 3.7551x
KMNIST:        3.7551x
MNIST:         4.4360x
```

同时 StrongSmooth 单次 smoothing 需要约 `2.7s`，LightSmooth 只需要约 `0.04s`。所以 v5.2 已经把 smoothing 从“只能离线诊断”推进到“有机会进训练循环”的成本级别。

P1 single-event 结果更关键。以 ABRBF-linear+silu teacher 为例，Fashion-MNIST 与 KMNIST 的 LightSmooth 都能在 `acc drop = 0`、KL 约 `0.0006-0.0007`、logit drift 约 `0.025` 的情况下获得约：

$$
\phi_{rbf}\text{ reduction}\approx 0.34,
$$

$$
\operatorname{curv}_{rbf}\text{ reduction}\approx 0.56.
$$

这证明 LightSmooth 作为单次 residual 几何维护算子是真实有效的。

P2 one-cycle 也通过了。ABRBF-LightSmooth 在三组数据集上都能完成：

```text
train -> light smooth -> refresh
```

并且保持或略微提高 accuracy，同时留下约 `5%` 的 residual phi reduction 和约 `53%-54%` 的 residual curvature reduction。KMNIST one-cycle 从 ABRBF-AdamW 的 `0.6797` 提到 `0.6973`，Fashion 从 `0.7949` 到 `0.7988`，MNIST 从 `0.8770` 到 `0.8809`。

因此，v5.2 的实质进展是：

$$
\boxed{
\text{我们已经有了 memory-budgeted、function-preserving、one-cycle 有效的 AB-RBF residual smoother。}
}
$$

这不是小进展。

---

### 1.2 也有明确失败

失败发生在 P3 multicycle integration。

P3 中 fixed-interval smoothing 能保留几何收益，但 task cost 累积太大：

```text
Fashion-MNIST:
  ABRBF-AdamW acc = 0.8008
  fixedInterval acc = 0.7969
  gap = 0.0039
  phiR red = 0.1139
  curvR red = 0.7852
  Fashion 单独基本可接受

KMNIST:
  ABRBF-AdamW acc = 0.6934
  fixedInterval acc = 0.6699
  gap = 0.0234
  phiR red = 0.1233
  curvR red = 0.7796
  KMNIST 任务代价过大

MNIST:
  ABRBF-AdamW acc = 0.8906
  fixedInterval acc = 0.8711
  gap = 0.0195
  phiR red = 0.1000
  curvR red = 0.7708
  MNIST 任务代价也过大
```

而 event-driven smoothing 完全没有触发：

```text
Fashion/KMNIST/MNIST:
  events = 0
  accepted = 0
  phiR red = 0
  curvR red = negative or 0
```

这说明当前 controller 的触发条件太保守或定义错了。它没有识别到“该做 smoothing 的时机”。

所以 v5.2 的失败不是 smoothing 算子失败，而是：

$$
\boxed{
\text{fixed schedule 太粗暴，event trigger 太迟钝。}
}
$$

---

## 2. 代码实现审视

### 2.1 当前 core 文件仍显示 RBFDense 是固定 centers / width / 只训练 coeff

上传的 `dgkan_core(5).py` 中，`RBFDense` 的结构仍然是：

```python
centers = torch.linspace(-2.5, 2.5, basis_count)
self.register_buffer("centers", centers)
self.width = float((centers[1] - centers[0]).abs() * 1.4)
self.coeff = nn.Parameter(...)
```

这意味着基础 RBF 层仍然是：

$$
f_{ij}(x)=\sum_k c_{ijk}B_k(x),
$$

其中被训练的是 $c_{ijk}$，centers 与 width 固定。

这本身不一定错，但对 PureKAN 来说已经暴露过明显限制：RBFOnly 会把低阶结构压到 RBF 高 Sobolev eigenmodes 里。v5.0/v5.1 已经说明 AB-RBF 能把相当一部分能量转移到 explicit base path，从而改善 architecture frontier。

### 2.2 当前上传的 core 中没有明显的 ABRBFDense 定义，需要检查实现是否分散在 runner 内

我在上传的 `dgkan_core(5).py` 中没有看到 `ABRBFDense`、`edge_named_params`、`base_named_params`、`rbf_residual_named_params` 这些 core-level 实现痕迹。v5.1 结果文档写到已经添加了 core AB-RBF edge paths，但当前上传代码更像仍以 `RBFDense` / `PureKANClassifier` 为主。

这可能有两种解释：

```text
1. 你上传的 dgkan_core(5).py 不是最终实际跑 v5.1/v5.2 的 core；
2. AB-RBF 仍然主要实现在 run_gafu_v5x.py 的临时模块里，而不是统一进 dgkan_core.py。
```

不管哪种，下一步都必须先做 **core consistency check**。

v5.3 不能继续让 AB-RBF 实现分散在 runner 里。否则后续 smoothing、edge decomposition、base/RBF 参数分组、memory audit、CIFAR scaling 都很容易出现不一致。

### 2.3 v5.2 controller 的核心问题不是单次 smoothing，而是触发/恢复逻辑

从结果看，LightSmooth 本身工作正常：P1 单次与 P2 one-cycle 都过。P3 失败有两类：

```text
fixedInterval:
  会触发，也能降 residual geometry，但 task cost 累积。

eventDriven:
  几乎不触发，所以没有 geometry gain。
```

这说明当前 controller 不是“验收算子不行”，而是缺少：

```text
1. 什么时候 probe smoothing？
2. probe 后怎么选择 eta？
3. 什么时候允许 smoothing？
4. smoothing 后怎么 refresh？
5. 如果上一轮 smoothing 没恢复，多久不能再 smooth？
```

v5.3 应该围绕这五个问题重做 controller，而不是继续增加 proposal 或 hidden constraints。

---

## 3. 当前改进方向

### 3.1 主线保留 AB-RBF，不回退 RBFOnly

AB-RBF 已经多轮证明是 architecture-positive。Base path 不是装饰，而是承担低阶结构；RBF residual 才适合被 Sobolev / residual smoothing 约束。

v5.3 默认 edge 应该固定为：

```text
PureKAN-ABRBF-linear+silu
```

并保留：

```text
PureKAN-ABRBF-linear
PureKAN-ABRBF-silu
```

作为 architecture ablation。

RBFOnly 只保留为负对照和 eigenmode audit，不再作为主线。

---

### 3.2 smoothing 从训练主路径降级为低频 maintenance event

v5.2 已经证明频繁 smoothing 会积累 task cost。下一步应该明确：

$$
\boxed{
\text{LightSmooth 不是每步 optimizer，而是低频 residual geometry maintenance event。}
}
$$

它的默认职责不是学任务，而是：

```text
当 RBF residual 的 phi/curvature 超预算，且 task trajectory 有足够 slack 时，做一次小步 residual smoothing。
```

---

### 3.3 event trigger 不能只看 plateau，要看 geometry debt + task slack + probe score

v5.2 的 event-driven 没触发，说明 plateau trigger 太保守或不相关。

新的 event controller 应该计算：

$$
D_{geo}(t)
=
\max\left(0,\frac{\phi_{rbf}(t)}{\phi_{target}}-1\right)
+
\omega_c
\max\left(0,\frac{\operatorname{curv}_{rbf}(t)}{c_{target}}-1\right).
$$

同时计算 task slack：

$$
S_{task}(t)
=
\min\left(
Acc(t)-Acc_{floor},
\epsilon_{loss}-\Delta L_{recent}
\right).
$$

但只看 debt 和 slack 还不够。应该每隔若干 epoch 做一个 **cheap probe**：在 holdout mini-batch 上尝试多个小 eta，不写回参数，只记录：

$$
Score(\eta)
=
\Delta \phi_{rbf}
+
\omega_k\Delta \operatorname{curv}_{rbf}
-
\lambda_{KL}KL
-
\lambda_z d_z
-
\lambda_a\Delta Acc_{drop}
-
\lambda_t T_{smooth}.
$$

然后只有当：

$$
D_{geo}(t)>D_{min},
$$

$$
S_{task}(t)>0,
$$

$$
\max_{\eta}Score(\eta)>0,
$$

才真正执行 smoothing。

---

### 3.4 refresh 不能只有一种，需要比较 base-only / RBF-frozen / full refresh

v5.2 的 P2 one-cycle 能 recover，但 P3 multicycle 任务成本累积。说明 refresh policy 还不够成熟。

下一步必须系统比较：

```text
refresh-none:
  smoothing 后不 refresh，只看真实 damage。

refresh-full:
  smoothing 后 full AB-RBF AdamW refresh。

refresh-base-only:
  smoothing 后只更新 base path，让 base path 修复 logits。

refresh-rbf-frozen:
  smoothing 后冻结 RBF residual，更新 base path + normalization-like fixed components if any。

refresh-rbf-task-small:
  smoothing 后 RBF residual 用很小 AdamW/task step 恢复，base path 正常更新。
```

我当前最看好的是：

$$
\boxed{
\text{smoothing 后短期冻结 RBF residual，只让 base path 做 task recovery。}
}
$$

原因是 smoothing 目标是降低 RBF residual roughness，如果 refresh 立刻让 RBF residual 重新大步学习，几何收益会被冲掉；但如果完全不 refresh，task cost 会保留。base path 是低阶任务通道，适合作为 smoothing 后的 recovery channel。

---

### 3.5 multicycle 需要 cooldown 和 recovery accounting

每次 smoothing event 后都要进入 cooldown。

默认：

```text
cooldown_epochs = 3
```

在 cooldown 内禁止再次 smoothing，除非出现极端 geometry explosion。

并且记录：

$$
Recovery(t)=
\frac{Acc_{after\ refresh}-Acc_{after\ smooth}}
{Acc_{before\ smooth}-Acc_{after\ smooth}+\epsilon}.
$$

如果：

$$
Recovery < 0.8,
$$

则下一次 smoothing 的 eta 上限减半，或者直接禁用 smoothing。

---

## 4. v5.3 实验总目标

v5.3 不是证明 smoothing 越强越好，而是证明：

$$
\boxed{
\text{在显存/时间预算内，LightSmooth 能作为低频 maintenance event，稳定改善 residual geometry 而不伤 task。}
}
$$

最终要求：

```text
1. AB-RBF remains task-competitive.
2. LightSmooth memory ratio <= 1.25 on main runs.
3. Event controller produces at most 1-2 accepted smoothing events.
4. Final acc drop vs ABRBF-AdamW <= 0.5%.
5. Final phi_rbf reduction >= 5% or curvature_rbf reduction >= 30%.
6. Refresh recovery >= 0.8 after each smoothing event.
```

---

## 5. v5.3 实验阶段

## P0: Core consistency and memory smoke

### 目标

P0 先确认代码结构和 v5.2 的关键实现一致。尤其要检查 AB-RBF 是否已经进入 core，而不是只在 runner 内部临时定义。

### 方法

```text
methods:
  ABRBF-AdamW
  ABRBF-LightSmooth-logitOnly-smoke
  ABRBF-LightSmooth-scoreOnly-smoke
  ABRBF-StrongSmooth-reference-smoke, offline only

datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0

model:
  hidden_dim = 64
  basis_count = 16
  depth = 4
  edge = ABRBF-linear+silu
```

### 必须记录

```text
nonKAN_param_count
edge_param_count
base_param_count
rbf_residual_param_count
edge_coverage
base_coverage
rbf_coverage
ABRBFDense_in_core
ABRBF_defined_in_runner_only
rollback_max_abs_error
peak_cuda_allocated_mb
peak_cuda_reserved_mb
memory_ratio_vs_ABRBF_AdamW
smoothing_time_sec
teacher_cache_mb
proposal_temp_mb
```

### 通过条件

$$
\text{nonKAN param count}=0.
$$

$$
\text{edge coverage}=\text{base coverage}=\text{rbf coverage}=1.
$$

$$
\text{rollback error}<10^{-8}.
$$

$$
\text{LightSmooth memory ratio}\leq1.25
$$

允许 MNIST P0 smoke 出现一次 `1.25-1.35` 的 borderline，但 P1/P2 不能持续超过预算。

### P0 图

```text
memory_dashboard_p0.png:
  method vs peak allocated MB
  method vs peak reserved MB
  teacher cache / proposal temp stacked bar

param_coverage_manifest.png:
  edge/base/rbf coverage heatmap
```

---

## P1: Single-event smoother verification

### 目标

P1 复现 v5.2 的单次 smoothing 正信号，并去掉重复 controller。

### 方法

只保留两个 proposal：

```text
S0-laplacian
S1-sobolev-grad
```

只保留两个 controller：

```text
logitOnly
scoreOnly
```

teachers：

```text
ABRBF-linear+silu-AdamW
ABRBF-linear-AdamW
ABRBF-silu-AdamW
```

### 必须记录

```text
teacher_acc
student_acc_after_smooth
acc_drop
KL_teacher_student
logit_relative_drift
phi_rbf_before
phi_rbf_after
phi_rbf_reduction
curvature_rbf_before
curvature_rbf_after
curvature_rbf_reduction
sobolev_rbf_before
sobolev_rbf_after
sobolev_rbf_reduction
accepted_eta
backtrack_count
memory_ratio
smoothing_time_sec
```

### 通过条件

$$
\Delta Acc\leq0.005.
$$

$$
KL\leq0.005.
$$

$$
d_z\leq0.03.
$$

$$
\Delta\phi_{rbf}\geq0.05
\quad\text{or}\quad
\Delta\operatorname{curv}_{rbf}\geq0.30.
$$

$$
\text{memory ratio}\leq1.25.
$$

### P1 图

```text
single_event_geometry_cost_pareto.png:
  x = memory ratio
  y = phi_rbf reduction
  point size = acc drop
  color = proposal

single_event_logit_vs_phi.png:
  x = logit drift
  y = phi_rbf reduction

single_event_curvature_bar.png:
  proposal/controller grouped bar of curvature_rbf reduction
```

---

## P2: Event-probe audit without writing updates

### 目标

P2 是 v5.3 的关键新增阶段。它不真正 smoothing，只在训练过程中定期 probe：如果我们使用新的 controller，它会不会在正确时间触发？选择什么 eta？预期 task cost 是多少？

这一步解决 v5.2 的问题：event-driven 没触发，而 fixedInterval 太粗暴。

### 训练设置

```text
method:
  ABRBF-linear+silu-AdamW

datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

probe_interval_epochs:
  2

eta_grid:
  0.005, 0.01, 0.02, 0.035, 0.05
```

### Probe 逻辑

每次 probe 计算：

```text
current phi_rbf
current curvature_rbf
current val_loss trend
current acc slack
base/RBF norm ratio
base/RBF ablation drop
```

然后对每个 eta 做 temporary smoothing，不写回，记录：

```text
acc_drop_probe
KL_probe
logit_drift_probe
phi_gain_probe
curv_gain_probe
score_probe
```

分数定义：

$$
Score(\eta)=
\Delta\phi_{rbf}
+0.5\Delta\operatorname{curv}_{rbf}
-10KL
-2d_z
-5\max(0,\Delta Acc_{drop})
-0.1T_{smooth}.
$$

### 必须记录

```text
probe_epoch
probe_step
geometry_debt
phi_rbf
curvature_rbf
val_loss_slope
val_acc
task_slack
eta
probe_score
probe_acc_drop
probe_KL
probe_logit_drift
probe_phi_gain
probe_curv_gain
would_trigger_by_plateau_old
would_trigger_by_geometry_debt
would_trigger_by_score
best_eta
best_score
```

### 通过条件

P2 不要求真正提升 accuracy。它要求 controller 行为合理：

```text
1. 至少一个非初始 probe 点有 best_score > 0。
2. 新 geometry-debt/score trigger 在至少两个数据集上会触发。
3. 旧 plateau trigger 与新 trigger 的差异必须可解释。
4. best eta 不能总是 0.05；如果总是最大 eta，说明 eta grid 太小。
5. probe task cost 的 P90 acc_drop <= 0.005。
```

### P2 图

```text
probe_score_over_time.png:
  epoch vs best_score, with trigger threshold

would_fire_comparison.png:
  old plateau trigger vs new score trigger count

eta_selection_heatmap.png:
  dataset x epoch, color = selected eta

geometry_debt_vs_score.png:
  x = geometry debt
  y = best score

probe_cost_gain_pareto.png:
  x = acc_drop_probe
  y = phi_gain_probe or curv_gain_probe
  color = eta
```

---

## P3: One-cycle controller variants

### 目标

P3 真正写回一次 smoothing，并比较不同 refresh policy。P3 的目标不是 multicycle，而是找出最安全的 smoothing + refresh 组合。

### 方法

```text
baseline:
  ABRBF-AdamW

one-cycle methods:
  oneShot-late-logitOnly-fullRefresh
  oneShot-late-logitOnly-baseOnlyRefresh
  oneShot-late-logitOnly-rbfFrozenRefresh
  oneShot-late-scoreOnly-fullRefresh
  oneShot-late-scoreOnly-baseOnlyRefresh
  oneShot-late-scoreOnly-rbfFrozenRefresh
```

smoothing 时间：

```text
late_epoch = 70% training progress
```

refresh 长度：

```text
refresh_steps = 5 or 10
```

### Refresh 定义

#### fullRefresh

```text
base path + RBF residual 全部 AdamW/task refresh
```

#### baseOnlyRefresh

```text
只更新 base path
RBF residual 冻结
```

#### rbfFrozenRefresh

```text
base path 更新
RBF residual 冻结若干步
之后恢复 RBF residual task update
```

### 必须记录

```text
acc_before_smooth
acc_after_smooth
acc_after_refresh
val_loss_before_smooth
val_loss_after_smooth
val_loss_after_refresh
phi_rbf_before
phi_rbf_after_smooth
phi_rbf_after_refresh
curv_rbf_before
curv_rbf_after_smooth
curv_rbf_after_refresh
recovery_ratio
geometry_retention
refresh_steps_to_recover
base_update_norm_during_refresh
rbf_update_norm_during_refresh
base/RBF_norm_before_after
```

定义：

$$
Recovery=
\frac{Acc_{refresh}-Acc_{smooth}}
{Acc_{before}-Acc_{smooth}+\epsilon}.
$$

$$
GeometryRetention=
\frac{\phi_{before}-\phi_{refresh}}
{\phi_{before}-\phi_{smooth}+\epsilon}.
$$

### 通过条件

$$
Acc_{refresh}\geq Acc_{ABRBF-AdamW}-0.005.
$$

$$
\phi_{rbf}\text{ reduction after refresh}\geq0.05
\quad\text{or}\quad
\operatorname{curv}_{rbf}\text{ reduction after refresh}\geq0.30.
$$

$$
Recovery\geq0.8.
$$

$$
GeometryRetention\geq0.5.
$$

### P3 图

```text
refresh_recovery_plot.png:
  smooth event -> refresh trajectory for acc and val loss

geometry_retention_plot.png:
  phi_rbf and curvature_rbf before/smooth/refresh

refresh_role_update_plot.png:
  base update norm vs rbf update norm during refresh

one_cycle_method_pareto.png:
  x = final acc gap
  y = final phi_rbf reduction
  size = memory ratio
```

---

## P4: Event-driven multicycle v2

### 目标

P4 是 v5.3 的主实验。它测试新的 controller 是否能解决 v5.2 P3 的问题。

### 方法

只扩 P3 通过的前两个 one-cycle refresh policies。

候选：

```text
EventV2-scoreProbe-max1
EventV2-scoreProbe-max2
EventV2-geometryDebt-scoreProbe-max2
FixedInterval-reference
ABRBF-AdamW
```

### Controller 默认参数

```text
probe_interval_epochs = 2
cooldown_epochs = 3
max_events = 2
eta_grid = 0.005, 0.01, 0.02, 0.035, 0.05
score_threshold = 0.02
min_geometry_debt = 0.05
max_probe_acc_drop = 0.005
max_probe_KL = 0.005
max_probe_logit_drift = 0.03
```

如果上一次 event：

$$
Recovery < 0.8,
$$

则：

```text
next eta max = previous eta / 2
cooldown_epochs += 2
```

如果：

$$
GeometryRetention < 0.3,
$$

则：

```text
下一次 smoothing 不触发，直到 phi_rbf 再次超预算且 val loss plateau。
```

### 必须记录

```text
event_id
event_epoch
event_reason
geometry_debt
task_slack
best_eta
best_score
accepted_eta
acc_before
acc_after_smooth
acc_after_refresh
KL
logit_drift
phi_gain_probe
phi_gain_actual
curv_gain_probe
curv_gain_actual
recovery_ratio
geometry_retention
cooldown_remaining
reject_reason
```

### 通过条件

三数据集都需要：

$$
Acc_{final}\geq Acc_{ABRBF-AdamW}-0.005.
$$

并且：

$$
\phi_{rbf}\text{ reduction}\geq0.05
\quad\text{or}\quad
\operatorname{curv}_{rbf}\text{ reduction}\geq0.30.
$$

同时：

$$
\text{memory ratio}\leq1.25,
$$

$$
\text{amortized time ratio}\leq1.20.
$$

### P4 图

```text
event_timeline.png:
  epoch vs acc / val loss / phi_rbf / curv_rbf with event markers

accepted_eta_trace.png:
  event index vs eta and score

event_recovery_dashboard.png:
  recovery_ratio and geometry_retention per event

fixed_vs_event_pareto.png:
  final acc gap vs phi_rbf reduction, color = event count

rejection_reason_stacked_bar.png:
  counts of no_geometry_debt / no_task_slack / negative_score / cooldown / task_cost
```

---

## P5: 3-seed candidate selection

### 目标

P5 只扩 P4 通过的候选，不再跑所有 variants。

### 方法

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

methods:
  ABRBF-AdamW
  best EventV2 candidate
  best OneCycle candidate
  FixedInterval-reference
  RBFOnly-AdamW, diagnostic only
```

### 必须记录

所有 P4 指标继续记录，并额外聚合：

```text
mean_acc
std_acc
paired_acc_delta_vs_ABRBF
paired_acc_CI
mean_phi_rbf_reduction
mean_curv_rbf_reduction
mean_recovery_ratio
mean_geometry_retention
mean_event_count
mean_memory_ratio
mean_time_ratio
```

### 通过条件

```text
1. mean acc gap vs ABRBF-AdamW <= 0.005 on all datasets
2. paired acc CI lower bound >= -0.010
3. phi_rbf reduction >= 0.05 or curvature_rbf reduction >= 0.30
4. memory ratio <= 1.25
5. time ratio <= 1.20
6. no dataset-specific hard failure, especially KMNIST
```

---

## P6: 5-seed confirm

### 目标

只确认一个 candidate。

### 方法

```text
methods:
  ABRBF-AdamW
  selected EventV2 or OneCycle method
  RBFOnly-AdamW diagnostic

seeds:
  0..4
```

### 成功标准

P6 若要称为 train-time geometry maintenance success：

$$
\Delta Acc_{paired}\geq -0.005
$$

并且：

$$
\Delta \phi_{rbf}\geq 0.05
\quad\text{or}\quad
\Delta \operatorname{curv}_{rbf}\geq0.30.
$$

同时 ECE 不能恶化超过：

$$
\Delta ECE\leq0.02.
$$

---

## P7: 10-seed final only if P6 passes

### 目标

最终确认，不再调参。

### 方法

```text
methods:
  ABRBF-AdamW
  selected geometry-maintenance method

seeds:
  0..9
```

### 报告指标

```text
paired_acc_delta_mean
paired_acc_CI95
paired_val_auc_delta
paired_ECE_delta
paired_phi_rbf_reduction
paired_curv_rbf_reduction
paired_memory_ratio
paired_time_ratio
```

### 最终 claim 规则

如果 P7 通过：

```text
AB-RBF + EventV2 LightSmooth is a train-time residual geometry maintenance method.
```

如果 P7 不通过但 P3 one-cycle 稳定：

```text
AB-RBF + LightSmooth is a one-cycle / post-task geometry maintenance method, not multicycle training component.
```

如果 P3/P4 都不稳定：

```text
AB-RBF remains the main PureKAN primitive; residual smoothing should remain offline/posthoc diagnostic only.
```

---

## 6. 必须统一记录的指标

### 6.1 任务指标

```text
test_acc
val_acc
train_acc
val_loss
val_loss_auc
train_loss_auc
ECE
NLL
margin_mean
margin_p10
classwise_acc
train_test_gap
```

### 6.2 AB-RBF edge 分解指标

```text
base_norm
rbf_norm
base_over_rbf
base_ablation_drop
rbf_ablation_drop
phi_base
phi_rbf
phi_total
curvature_base
curvature_rbf
curvature_total
sobolev_rbf_norm
high_sobolev_eigen_energy
basis_occupancy_entropy
dead_basis_fraction
out_of_grid_fraction
```

### 6.3 Smoothing 指标

```text
proposal
controller
eta
accepted_eta
backtrack_count
acc_drop
KL_teacher_student
logit_relative_drift
phi_rbf_reduction
curvature_rbf_reduction
sobolev_rbf_reduction
smooth_score
accepted
rejected
reject_reason
```

### 6.4 Controller 指标

```text
probe_epoch
geometry_debt
task_slack
val_loss_slope
best_probe_score
best_probe_eta
would_trigger_old_plateau
would_trigger_new_score
cooldown_remaining
last_recovery_ratio
last_geometry_retention
event_count
accepted_event_count
rejected_event_count
```

### 6.5 Refresh 指标

```text
refresh_policy
refresh_steps
refresh_recovered_loss
refresh_recovered_acc
recovery_ratio
geometry_retention
base_update_norm_refresh
rbf_update_norm_refresh
base_update_share_refresh
rbf_update_share_refresh
```

### 6.6 显存与时间指标

```text
peak_cuda_allocated_mb
peak_cuda_reserved_mb
teacher_cache_mb
proposal_temp_mb
smoothing_time_sec
refresh_time_sec
total_training_time_sec
amortized_step_time_ms
memory_ratio_vs_ABRBF_AdamW
time_ratio_vs_ABRBF_AdamW
```

---

## 7. 必须生成的可视化

### 7.1 Memory dashboard

```text
memory_dashboard.png
```

内容：

```text
method vs peak allocated MB
method vs peak reserved MB
teacher cache / proposal temp stacked bars
smoothing time per event
```

### 7.2 Single-event cost-benefit Pareto

```text
single_event_pareto.png
```

横轴：

$$
\Delta Acc_{drop}
$$

纵轴：

$$
\Delta\phi_{rbf}
$$

点大小：

$$
\Delta\operatorname{curv}_{rbf}
$$

颜色：proposal / controller。

### 7.3 Probe trigger dashboard

```text
probe_trigger_dashboard.png
```

包含：

```text
epoch vs geometry_debt
epoch vs best_probe_score
epoch vs selected_eta
epoch vs would_trigger_old/new
```

### 7.4 Event timeline

```text
event_timeline.png
```

同图显示：

```text
accuracy
val loss
phi_rbf
curvature_rbf
smoothing event markers
cooldown windows
```

### 7.5 Recovery plot

```text
refresh_recovery.png
```

每个 event 画：

```text
acc_before -> acc_after_smooth -> acc_after_refresh
phi_before -> phi_after_smooth -> phi_after_refresh
```

### 7.6 Geometry retention plot

```text
geometry_retention.png
```

展示：

```text
method vs geometry_retention
method vs recovery_ratio
```

### 7.7 Base/RBF role plot

```text
base_rbf_role_trace.png
```

展示：

```text
epoch vs base_over_rbf
epoch vs base_ablation_drop
epoch vs rbf_ablation_drop
```

### 7.8 Failure heatmap

```text
failure_heatmap.png
```

行：method。列：dataset。颜色：failure category。

Failure categories：

```text
memory_over_budget
task_cost_accumulation
no_trigger
poor_recovery
poor_geometry_retention
KMNIST_specific_failure
MNIST_specific_failure
```

---

## 8. 决策规则

### 情况 A：EventV2 通过 P6/P7

结论：

```text
AB-RBF + EventV2 LightSmooth becomes train-time residual geometry maintenance.
```

这时主线是：

```text
AB-RBF AdamW for task learning
+ event-driven LightSmooth for residual geometry maintenance
```

### 情况 B：OneCycle 通过，但 EventV2 不通过

结论：

```text
AB-RBF + LightSmooth is one-cycle / late-stage geometry maintenance only.
```

这时不要强行做 multicycle。报告重点是：

```text
one-cycle smoothing is useful as post-task or late-stage consolidation.
```

### 情况 C：只有 P1 单次通过，P2/P3 不稳

结论：

```text
LightSmooth remains offline/posthoc diagnostic, not train-time component.
```

主线退回：

```text
AB-RBF AdamW is the PureKAN default.
```

### 情况 D：AB-RBF 自身在后续 full-budget 不稳定

结论：

```text
需要重新审视 AB-RBF capacity / basis / normalization，而不是 smoothing。
```

---

## 9. 当前推荐执行顺序

第一优先级：

```text
P0 -> P1 -> P2
```

先证明 code consistency、single-event、probe trigger 都正确。

第二优先级：

```text
P3 one-cycle refresh policy
```

选出最安全 refresh。

第三优先级：

```text
P4 event-driven multicycle
```

只扩 P3 胜出的 refresh policy。

第四优先级：

```text
P5 3-seed selection
```

如果 P4 失败，不能进入 P5。

---

## 10. 最终总结

v5.2 已经有进展：

$$
\boxed{
\text{LightSmooth 已经从“太重的 strong acceptor”变成“可用的一次性 residual geometry maintenance”。}
}
$$

但 v5.2 也明确失败：

$$
\boxed{
\text{固定周期 smoothing 会累积 task cost；旧 event trigger 又完全不触发。}
}
$$

所以 v5.3 的核心不是新 smoother，而是新 controller：

$$
\boxed{
\text{geometry debt + task slack + cheap probe score + cooldown + refresh recovery。}
}
$$

如果这个 controller 仍然失败，就应该停止把 smoothing 放进训练循环，只把它作为 one-cycle late-stage/posthoc 工具。届时主线应明确收束为：

```text
PureKAN default = AB-RBF AdamW
LightSmooth = optional one-cycle geometry maintenance / offline diagnostic
```
