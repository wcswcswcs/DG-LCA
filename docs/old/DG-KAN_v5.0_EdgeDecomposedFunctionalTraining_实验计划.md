# DG-KAN v5.0：Edge-Decomposed Functional Training 实验计划

## 0. 本轮结论先行

v4.9 之后，我认为 PureKAN functional update 的问题不能再被理解为“再找一个更好的 Sobolev 系数、trust radius 或 warmup”。现在更准确的判断是：

$$
\boxed{\text{PureKAN 的主要瓶颈是 edge 参数化、几何定义、functional update 目标三者错配。}}
$$

当前 strict PureKAN 主要是 fixed-grid RBF-only 网络：每条边是

$$
f_{ij}(x)=\sum_k c_{ijk}B_k(x),
$$

其中真正被训练的是 $c_{ijk}$。RBF centers 和 width 基本是固定的；PureKAN 里没有普通 stem、head、bias、learnable LayerNorm，也没有 learnable alpha。这个设置很干净，但也非常苛刻。它要求 RBF 系数同时承担低阶线性结构、尺度调整、feature formation、class boundary 和非线性修正。过去的 U-FULL / TFU / FNG / FCAdam / NFS 都是在这个坐标系上试图优化系数，但它们没有给低阶结构一个自然通道。

v4.9 的结果已经改变了我们对问题的理解。AB-RBF 在 AdamW 下是 architecture-positive：`ABRBF-silu` 在 MNIST、Fashion、KMNIST 都提升 mean accuracy；`ABRBF-linear` 在 MNIST/KMNIST 提升，并在 Fashion 基本持平。eigenmode audit 显示，RBFOnly 大约有 $26\%-27\%$ coefficient energy 落在 high Sobolev-eigen band；ABRBF-linear 把约 $24\%-26\%$ 能量转移到 explicit base modes，并把 high-mode energy 降到约 $22\%-23\%$。这说明 fixed-grid RBF-only 很可能把低阶结构强行塞进了 RBF 高模态。

同时，Exact NFS 的结果也非常重要。Exact NFS 的 KKT residual 非常小，KL/logit drift 几乎为零，但 geometry reduction 也几乎为零。这说明 Exact NFS 并不是数值不稳定，而是过于保守，或者说它的约束空间把真正能降几何的方向投掉了。相比之下，Heuristic NFS-role-block 还能给出小的 $\phi$ reduction 和较大的 Jacobian reduction。这说明 v4.8/v4.9 的 NFS 信号是真实的，但“严格 nullspace projection”不是正确实现方式。

因此 v5.0 的核心不再是“继续发明一个单步 functional optimizer”，而是建立一个新的训练框架：

$$
\boxed{\text{Edge-Decomposed Functional Training}}
$$

它的基本思想是：每条 KAN edge 不应该只有 RBF residual，而应该被分成低阶 base path 和 RBF residual path。base path 负责低频、线性、尺度和稳定表示；RBF residual 负责非线性修正；几何约束主要作用在 RBF residual 上，而不是无差别压制整个 edge。

---

## 1. v4.9 的关键证据与新的解释

### 1.1 P0：实现不是主要问题

v4.9 P0 通过：21 rows、0 errors、max nonKAN 为 0、functional coverage 为 1.0、rollback 为 0、max KKT residual 约 $10^{-6}$。这说明新 edge 参数化和 Exact NFS 路径至少在 smoke 层面是可审计的。后续失败不能优先归因于参数没被更新、rollback 出错、hidden non-KAN 参数混入等低级实现问题。

### 1.2 P1：AB-RBF 是 architecture-positive，但当前几何指标会误判它

P1 里 `ABRBF-silu` 在三数据集都提升 mean accuracy；`ABRBF-linear` 也改善 MNIST/KMNIST，并且 base/RBF norm 大约在 $0.55-0.59$，说明 base path 不是摆设，而是真正在用。这个结果强烈支持：RBF-only 的边函数缺少低阶通道。

但是 P1 里 AB-RBF 的 `phi red` 和 `J red` 往往是负的，尤其 `ABRBF-linear` 的 Jacobian reduction 数字非常差。这里不能简单解读为“AB-RBF 几何坏”。因为现有几何指标把 base path 的导数也算作“坏导数”。如果 edge 里有

$$
f_{ij}(x)=b_{ij}+w_{ij}x+\sum_k c_{ijk}B_k(x),
$$

那么 $w_{ij}$ 带来的导数是稳定低阶结构，不等同于 RBF residual 的高频粗糙性。当前 $\phi'$ / Jacobian gate 没有区分“有用低阶 derivative”和“危险高频 derivative”。因此 v5.0 必须重新定义几何审计：

$$
\text{edge derivative}=	ext{base derivative}+\text{RBF residual derivative}.
$$

新的 geometry gate 应该主要约束 RBF residual 的粗糙性，同时单独记录 full-network Jacobian / Lipschitz，而不是把二者混在一个分数里。

### 1.3 P2：RBF-only 确实在用高 Sobolev eigenmodes 承担低阶结构

RBFOnly 的 high eigenmode coefficient energy 约 $26\%-27\%$。ABRBF-linear 把约 $24\%-26\%$ 能量转移到 base modes，并将 high-mode energy 降到 $22\%-23\%$。这支持如下解释：

$$
\boxed{\text{RBF-only 不是没有表达力，而是用不自然的 RBF 高模态表达低阶结构。}}
$$

这也解释了为什么 Sobolev functional update 很难训练 PureKAN。Sobolev metric 正在惩罚的，很可能正是 AdamW 用来拼出低阶结构的 coefficient modes。功能上，这些 modes 可能是任务需要的；几何上，它们被 Sobolev 视作高代价方向。

### 1.4 P3：Exact NFS 过于保守，Heuristic NFS 的“小松弛”反而有效

Exact NFS-logit / logit-hidden / margin 的 KKT residual 很小，KL 和 logit drift 基本为零，但 $\phi$ reduction 几乎为零。这说明 exact projection 不是数值坏掉，而是投影后几何梯度几乎没有剩余可动方向。

所以 v5.0 不应该继续追求“更 exact 的 nullspace”。更应该做的是 **relaxed constrained smoothing**：允许极小的 logit / hidden / margin drift，换取可观的 geometry reduction。形式上不是硬约束

$$
J_f\Delta\theta=0,
$$

而是 trust-region/penalty：

$$
\min_{\Delta\theta}
\quad
\nabla R(\theta)^\top\Delta\theta
+
\frac{1}{2\eta}\|\Delta\theta\|_M^2
+
\frac{\mu_z}{2}\|J_z\Delta\theta\|^2
+
\frac{\mu_h}{2}\|J_h\Delta\theta\|^2.
$$

这样不会把所有 smoothing direction 直接杀掉。

### 1.5 P6：单纯扩大 capacity / basis 不能解决 geometry

P6 显示：更大的 hidden、basis 或 learnWidth 并没有稳定打开 smooth accurate frontier。ABRBF-silu h96/b24/d4 在 MNIST 上很好；ABRBF-linear 在 Fashion/KMNIST 上改善或持平；learnWidth alone 仍然不是 clear positive。更重要的是，capacity 扩大后 $\phi$ 通常升高。因此“多给 basis / 多给 width 自由度”不是根因解法。

---

## 2. 代码实现审计：必须改的地方

### 2.1 当前 core 的 RBFDense 仍是 fixed-grid RBF coefficient layer

当前 `RBFDense` 的核心实现是固定 centers、固定 width、训练 `coeff`。这意味着 PureKAN strict 版本中的 input/block/output KAN 都只训练 RBF coefficient，而 centers/width/base/bias 都没有参与 functional update。v5.0 的第一件事，是把 AB-RBF 从 runner-level 实验配置提升成 core-level edge module。

### 2.2 AB-RBF 必须是 edge-internal 参数，不是回退到 MLP

v5.0 不能简单加回 Linear stem/head。目标是把低阶通道放回每条 KAN edge 内部：

$$
f_{ij}(x)=b_{ij}+w_{ij}x+u_{ij}\operatorname{silu}(x)+\sum_k c_{ijk}B_k(x).
$$

这里 $b_{ij}$、$w_{ij}$、$u_{ij}$ 都属于 edge 参数，应该计入 functional coverage。它们不是 non-KAN 参数。这样才能保持 strict edge-only PureKAN。

### 2.3 coefficient_named_params 必须支持 edge groups

当前 functional coverage 不能只说 “coeff seen = 1”。需要进一步拆成：

```text
base_const_seen
base_linear_seen
base_silu_seen
rbf_coeff_seen
width_seen, if learnable width enabled
center_seen, if learnable center enabled
```

v5.0 P0 里每一个 strict row 都必须记录这些字段，并保证总 coverage 为 1.0。否则我们无法判断 functional optimizer 是否真正覆盖了所有 edge 参数。

### 2.4 几何指标需要拆分 base geometry 与 residual geometry

新 edge 的 derivative 是：

$$
f'_{ij}(x)=w_{ij}+u_{ij}\operatorname{silu}'(x)+\sum_k c_{ijk}B'_k(x).
$$

当前 $\phi'$ 把这些全部混在一起。v5.0 必须记录：

```text
phi_base_p95
phi_rbf_p95
phi_total_p95
curvature_base_p95
curvature_rbf_p95
curvature_total_p95
residual_sobolev_norm
base_norm
rbf_norm
base_over_rbf
```

如果 AB-RBF 的 total $\phi'$ 上升，但 RBF residual roughness 下降、accuracy 上升、ECE 不坏，那这不是失败。

### 2.5 Exact NFS 需要记录有效 nullspace 信息

P3 的 Exact NFS 结果显示 KKT 很小但 geometry 没动。下一步不能只记录 KKT residual。必须记录：

```text
constraint_rank
constraint_condition
nullspace_dim_estimate
raw_smoothing_grad_norm
projected_smoothing_grad_norm
projected_over_raw
angle_smoothing_to_constraint_rowspace
active_constraint_count
```

如果 `projected_over_raw` 接近 0，就说明 exact constraints 几乎完全消灭 smoothing direction。这是设计失败，不是优化失败。

---

## 3. v5.0 的核心假设

v5.0 将验证三个假设。

### 假设 H1：PureKAN 需要 edge-level base path

RBF-only PureKAN 需要用 RBF coefficient 表达低阶结构；这让 Sobolev / functional update 天然与任务表达冲突。AB-RBF 给出 base path 后，functional optimizer 的任务难度应下降。

验证标准：AB-RBF 在 AdamW 下应稳定优于 RBFOnly；更重要的是 AB-RBF 的 high Sobolev eigenmode energy 应下降，并且 base path ablation 应显示 base 对任务有实际贡献。

### 假设 H2：几何应该约束 RBF residual，而不是压制整个 edge

base path 是低阶函数，不应该被强 Sobolev 惩罚。RBF residual 才是主要 smoothing 对象。

验证标准：split-geometry gate 下，AB-RBF 应比 total-geometry gate 更合理；如果 total $\phi'$ 上升但 residual roughness 下降、accuracy/ECE 保持，则应判为成功。

### 假设 H3：Exact nullspace 太保守，需要 relaxed NFS

严格投影 $J_f\Delta=0$ 使 geometry reduction 近乎为零。有效的 geometry consolidation 应该允许极小函数漂移，换取显著 residual smoothing。

验证标准：Relaxed NFS 在相同 KL/logit drift 预算下，$\phi_{rbf}$ / residual Sobolev / curvature reduction 应显著强于 Exact NFS，并且 accuracy drop 不超过阈值。

---

## 4. v5.0 实验阶段

## P0：Core implementation smoke and invariants

P0 的目标不是看 accuracy，而是确认新的 edge module、coverage、geometry decomposition 和 relaxed NFS instrumentation 都正确。

需要实现：

```text
ABRBFDense:
  base_mode in {none, linear, silu, linear+silu}
  rbf_coeff
  optional learnable log_width
  optional data-quantile centers, initially frozen

PureKANClassifier(edge_type=...):
  RBFOnly
  ABRBF-linear
  ABRBF-silu
  ABRBF-linear+silu

coefficient_named_params:
  returns named edge parameter groups, not only coeff
```

P0 必跑配置：

```text
RBFOnly-smoke
ABRBF-linear-smoke
ABRBF-silu-smoke
ABRBF-linear+silu-smoke
ABRBF-linear-learnWidth-smoke
RelaxedNFS-smoke
ExactNFS-smoke
```

每个数据集 MNIST / Fashion-MNIST / KMNIST 都跑 seed0，1 epoch 或 2 smoke epochs。

必须记录：

```text
learnable_nonkan_params
functional_coverage_total
base_const_seen
base_linear_seen
base_silu_seen
rbf_coeff_seen
width_seen
center_seen
rollback_max_abs_error
nan_inf_count
edge_type
base_mode
width_mode
center_mode
```

几何 decomposition 记录：

```text
phi_base_p95
phi_rbf_p95
phi_total_p95
curvature_base_p95
curvature_rbf_p95
curvature_total_p95
sobolev_rbf_norm
base_norm
rbf_norm
base_over_rbf
```

NFS instrumentation 记录：

```text
constraint_rank
constraint_condition
projected_over_raw
KKT_residual
KL
logit_drift
hidden_drift
margin_drift
```

P0 通过标准：

```text
learnable_nonkan_params = 0
functional_coverage_total = 1.0
all intended edge groups seen = 1
rollback_max_abs_error < 1e-8
no NaN / Inf
RelaxedNFS and ExactNFS both produce finite diagnostics
```

---

## P1：Architecture frontier under AdamW

P1 回答：新的 AB-RBF edge 是否真的比 RBF-only 更适合作为 PureKAN 参数化。这里先用 AdamW，不讨论 functional optimizer。

方法：

```text
PureKAN-RBFOnly-AdamW
PureKAN-ABRBF-linear-AdamW
PureKAN-ABRBF-silu-AdamW
PureKAN-ABRBF-linear+silu-AdamW
PureKAN-ABRBF-linear-learnWidth-AdamW
PureKAN-RBF-quantileCenters-AdamW
PureKAN-BaseOnly-linear-AdamW
PureKAN-BaseOnly-silu-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
```

配置：

```text
hidden_dim = 64, basis_count = 16, depth = 4
train_size = current standard v4.9 setting
seeds = 0,1,2 first
```

如果 P1 有 clear positive，再扩 seeds 0..4。

记录指标：

```text
test_acc
val_loss_auc
ECE / NLL
train_loss_curve
val_loss_curve
feature_effective_rank by role
class_centroid_separation
margin_mean / margin_p10
phi_base_p95 / phi_rbf_p95 / phi_total_p95
curvature_base_p95 / curvature_rbf_p95 / curvature_total_p95
jacobian_condition
base_norm
rbf_norm
base_over_rbf
base_ablation_acc_drop
rbf_ablation_acc_drop
base_logit_delta_norm
rbf_logit_delta_norm
```

判定：

```text
AB-RBF architecture positive:
  mean acc >= RBFOnly + 0.5% on at least two datasets
  no dataset worse than RBFOnly by more than 0.5%
  base_ablation_acc_drop > 0.5% or base_logit_delta nontrivial
  high Sobolev eigenmode energy lower than RBFOnly
```

如果 `BaseOnly` 本身已经接近 AB-RBF，则说明 RBF residual 对任务贡献仍不足，需要重新设计 residual regularization。若 AB-RBF 明显强于 BaseOnly，则说明 base + residual 组合有效。

可视化：

```text
architecture frontier: acc vs residual_phi_rbf_p95
architecture frontier: acc vs total_phi_p95
base_over_rbf vs acc scatter
base_ablation_drop vs acc scatter
rank / margin curves by method
ECE vs acc scatter
```

---

## P2：Eigenmode and basis-use audit

P2 回答：AB-RBF 是否真正缓解 RBF high-mode pressure，以及 AdamW 使用了哪些 RBF 模式。

对 P1 每个 trained teacher 运行 eigenmode audit：

$$
S=Q\Lambda Q^\top,
$$

$$
\tilde c=Q^\top c.
$$

记录：

```text
low_eigen_coeff_energy
mid_eigen_coeff_energy
high_eigen_coeff_energy
low_eigen_grad_energy
mid_eigen_grad_energy
high_eigen_grad_energy
base_mode_energy
base_mode_grad_energy
basis_occupancy_entropy
dead_basis_fraction
out_of_grid_fraction
center_coverage
width_effective_scale
```

新增两个重要量：

$$
\text{high-mode task pressure}
=
\frac{\|Q_{high}^\top \nabla_c L\|^2}{\|\nabla_c L\|^2},
$$

$$
\text{high-mode learned energy}
=
\frac{\|Q_{high}^\top c\|^2}{\|c\|^2}.
$$

如果 high-mode task pressure 很低但 learned energy 高，说明 coefficient 高模态可能在表达低阶结构或 compensating basis mismatch。若 AB-RBF 降低 high-mode learned energy，同时 acc 上升，则支持 base path 假设。

可视化：

```text
Sobolev eigenvalue spectrum
coefficient energy spectrum by role
gradient energy spectrum by role
base mode energy vs high-mode energy scatter
basis occupancy heatmap input/block/output
out-of-grid fraction by layer
```

---

## P3：Split-metric functional update on AB-RBF

P3 是 v5.0 的第一个 functional optimizer 测试。重点不是继续用一个全局 Sobolev metric，而是 split update：

$$
\theta_{edge}=(\theta_{base}, c_{rbf}).
$$

base path 用任务学习动力学；RBF residual 用 geometry-aware functional update。

候选方法：

```text
ABRBF-AdamW
ABRBF-allAdamW-edgeOnly
ABRBF-baseAdam-rbfUFULL
ABRBF-baseFCAdam-rbfUFULL
ABRBF-baseAdam-rbfD6
ABRBF-baseAdam-rbfFCAdam-dataSob
ABRBF-baseAdam-rbfRelaxedNFSRefresh
ABRBF-baseFrozen-rbfUFULL
ABRBF-baseOnlyAdam-rbfFrozen
```

这里 `baseAdam` 不是普通 non-KAN AdamW，而是 edge-internal base 参数的 Adam-like update。它仍然属于 edge-only 参数化。为了保持严格性，需要在 row 中记录：

```text
nonKAN = 0
base_params_are_edge_params = 1
```

更新原则：

$$
\Delta \theta_{base} = \operatorname{AdamLike}(\nabla_{\theta_{base}}L),
$$

$$
\Delta c_{rbf} = -\eta M_{rbf}^{-1}\nabla_{c_{rbf}}L.
$$

其中 $M_{rbf}$ 可以是：

```text
identity
data-diag
dataSob
Sobolev full
Sobolev diag
```

P3 记录：

```text
base_update_norm
rbf_update_norm
base_update_share
rbf_update_share
cos_base_with_AdamW
cos_rbf_with_AdamW
function_R2_with_AdamW
holdout_descent_1/5/20
rank_change
margin_change
phi_base_change
phi_rbf_change
phi_total_change
curvature_rbf_change
jacobian_condition
ECE
```

P3 通过标准：

```text
on all datasets:
  accuracy gap vs ABRBF-AdamW after short run < 3%
  holdout_descent_20 positive and >= 0.75 * ABRBF-AdamW
  residual_phi_rbf <= ABRBF-AdamW or residual_curvature <= ABRBF-AdamW
  total phi allowed to rise if base derivative explains it
  base_ablation and rbf_ablation both nontrivial, unless BaseOnly baseline dominates
```

可视化：

```text
base/rbf update share over time
holdout descent vs residual_phi_rbf scatter
function_R2_with_AdamW vs accuracy gap
rank trajectory
margin trajectory
base/rbf ablation bars
```

---

## P4：Relaxed NFS projection audit

P4 不再追求 exact nullspace。目标是验证 relaxed constraints 是否能在保持 function 的同时带来有效 residual smoothing。

比较：

```text
Heuristic-NFS-role-block
ExactNFS-logit-hidden
RelaxedNFS-logit-hidden-muLow
RelaxedNFS-logit-hidden-muMed
RelaxedNFS-logit-hidden-muHigh
RelaxedNFS-role-block
RelaxedNFS-role-cycle
RelaxedNFS-rbfResidualOnly
RelaxedNFS-baseFrozen-rbfOnly
```

Relaxed NFS 解的形式：

$$
\Delta
=
-\eta
\left(
M+
\mu_zJ_z^TJ_z+
\mu_hJ_h^TJ_h+
\mu_mJ_m^TJ_m+ho I
\right)^{-1}
\nabla R.
$$

其中 $R$ 只对 RBF residual geometry 施加：

$$
R(c_{rbf})=\|c_{rbf}\|_{S}^2.
$$

必须记录：

```text
raw_geometry_grad_norm
projected_geometry_grad_norm
projected_over_raw
constraint_rank
constraint_condition
constraint_singular_values
KL_teacher_student
logit_relative_drift
hidden_relative_drift
margin_relative_drift
acc_drop
phi_rbf_reduction
curvature_rbf_reduction
phi_total_reduction
jacobian_reduction
smoothability_score
accepted_eta
backtrack_count
```

通过标准：

```text
acc_drop <= 0.5%
KL < 0.01
logit_drift < 0.03
hidden_drift < 0.05
phi_rbf_reduction > 10% or curvature_rbf_reduction > 20%
projected_over_raw > 0.05
```

如果 RelaxedNFS 成功而 ExactNFS 失败，就能证明问题不是没有 smoothing direction，而是 exact constraints 太强。

可视化：

```text
projected_over_raw histogram
constraint singular value spectrum
KL vs phi_rbf_reduction scatter
acc_drop vs phi_rbf_reduction scatter
Exact vs Relaxed NFS paired bar
role-wise NFS smoothability heatmap
```

---

## P5：AB-RBF Learn -> Relaxed Smooth -> Refresh

P5 测试一个完整 cycle：先学任务，再只平滑 RBF residual，最后小步刷新任务。

流程：

```text
Phase I: ABRBF task learning
  candidate: ABRBF-AdamW, ABRBF-baseAdam-rbfFCAdam, ABRBF-baseAdam-rbfD6

Phase II: RelaxedNFS residual smoothing
  target: RBF residual only
  constraints: logits + selected hidden sketch + margin budget

Phase III: Refresh
  option A: base path only refresh
  option B: output edge only refresh
  option C: full edge small-step refresh
```

记录：

```text
phaseI_acc
phaseI_phi_base / phi_rbf / phi_total
phaseI_rank
phaseI_margin
smooth_acc_drop
smooth_KL
smooth_phi_rbf_reduction
refresh_acc_recovery
refresh_phi_rbf_rebound
refresh_margin_recovery
cycle_net_score
```

cycle net score：

$$
\text{cycle\_score}
=
\Delta\text{acc}_{refresh}
+
\alpha \Delta\text{AUC}
+
\beta \Delta\text{ECE}
+
\gamma \Delta\text{geometry}_{rbf}
-
\delta \text{drift}.
$$

P5 通过标准：

```text
refresh_acc >= phaseI_acc - 0.5%
phi_rbf_refresh <= 0.9 * phi_rbf_phaseI
ECE not worse by > 0.02
rank not reduced by > 10%
margin_p10 not reduced by > 10%
```

可视化：

```text
three-phase acc / phi_rbf / phi_total curves
KL and logit drift over cycle
rank/margin before-smooth-after-refresh bars
cycle score heatmap
```

---

## P6：Event-driven multi-cycle TAN v2

只有 P5 有 all-dataset survivor 时才跑 P6。P6 不使用固定 cycle，而是根据事件触发 smoothing。

触发条件：

```text
if task_loss_plateau and phi_rbf_over_budget:
  trigger RelaxedNFS

if acc_drop_after_smooth > threshold:
  trigger refresh

if geometry_rebound_after_refresh:
  reduce refresh eta or increase residual Sobolev
```

事件状态：

```text
LEARN
SMOOTH_RBF
REFRESH_BASE
REFRESH_FULL_EDGE
RECOVERY
```

记录：

```text
event_type
event_step
trigger_reason
pre_event_metrics
post_event_metrics
acc_drop
geometry_gain
refresh_recovery
event_success
```

P6 通过标准：

```text
all datasets:
  final acc >= ABRBF-AdamW - 1%
  residual_phi_rbf <= ABRBF-AdamW * 0.85
  ECE <= ABRBF-AdamW + 0.02
  no catastrophic rank collapse
  smooth events success rate > 60%
```

可视化：

```text
event timeline
acc/loss/phi_rbf curves with event markers
smooth event success waterfall
refresh recovery violin
```

---

## P7：Capacity and basis follow-up

P7 只在 P1-P6 有正信号后进行。它不再盲目扩 capacity，而是围绕最好的 AB-RBF edge 做小范围验证。

候选：

```text
ABRBF-linear h64/b16/d4
ABRBF-linear h96/b24/d4
ABRBF-silu h64/b16/d4
ABRBF-silu h96/b24/d4
ABRBF-linear+silu h64/b16/d4
ABRBF-linear+silu h96/b24/d4
ABRBF-linear quantileCenters
ABRBF-linear learnWidth with width regularization
```

记录：

```text
all P1/P2 metrics
parameter_count
step_time_ms
peak_memory
base/rbf contribution
center_coverage
basis_occupancy
```

判定：

```text
Architecture scaling positive if:
  acc improves without residual_phi_rbf exploding
  base/RBF contribution remains balanced
  memory/time overhead acceptable
```

---

## P8：3-seed candidate selection

P8 把最好的方法组合跑 3-seed full budget。

候选最多 6 个：

```text
RBFOnly-AdamW
ABRBF-best-AdamW
ABRBF-best-splitFunctional
ABRBF-best-LearnSmoothRefresh
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
```

指标：

```text
test_acc
val_auc
ECE / NLL
phi_base_p95
phi_rbf_p95
phi_total_p95
curvature_rbf
jacobian_condition
rank
margin_p10
base_ablation_drop
rbf_ablation_drop
step_time_ms
```

P8 通过标准：

```text
ABRBF functional candidate:
  acc >= RBFOnly-AdamW - 0.5%
  acc >= Hybrid-DGKAN-UFULL-f085 - 1%
  val_auc positive vs RBFOnly-AdamW or no worse by more than 3%
  phi_rbf_reduction > 10% or curvature_rbf_reduction > 20%
  ECE not worse by > 0.02
```

如果 functional candidate 不能接近 ABRBF-AdamW，但 ABRBF-AdamW 很强，则说明 architecture 修复成功，optimizer 仍未解决。若 ABRBF functional 也明显改善，则进入 P9。

---

## P9：5-seed confirm

P9 对 P8 survivor 做 5-seed confirm。

必须包含：

```text
PureKAN-RBFOnly-AdamW
PureKAN-ABRBF-best-AdamW
PureKAN-ABRBF-functional-best
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
```

5-seed gate：

```text
acc mean not below best baseline by > 0.5% on any dataset
paired acc CI not strongly negative
val_auc mean positive or comparable
residual_phi_rbf reduced by > 10%
ECE comparable or better
base/rbf ablation both interpretable
```

可视化：

```text
paired delta vs RBFOnly-AdamW
paired delta vs ABRBF-AdamW
acc vs residual geometry Pareto
ECE vs acc Pareto
base/rbf contribution dashboard
```

---

## P10：10-seed final confirm

只有 P9 通过才跑 P10。P10 的目标不是继续搜索，而是最终确认。

最终结论分类：

### A. Strong success

```text
ABRBF functional candidate >= ABRBF-AdamW or within 0.5%
and geometry residual clearly better
and ECE/AUC not worse
```

可 claim：

$$
\boxed{\text{Edge-decomposed functional optimizer works for PureKAN.}}
$$

### B. Architecture success, optimizer partial

```text
ABRBF-AdamW strong;
ABRBF functional improves over RBF functional but does not match AdamW.
```

可 claim：

$$
\boxed{\text{RBF-only was a bottleneck; functional optimizer still needs Adam-like task dynamics.}}
$$

### C. NFS success only

```text
ABRBF-AdamW good;
RelaxedNFS can smooth posthoc but cannot train from scratch.
```

可 claim：

$$
\boxed{\text{Functional update is currently better as geometry consolidation than as primary optimizer.}}
$$

### D. No success

```text
ABRBF not stable or functional still fails.
```

可 claim：

$$
\boxed{\text{Current PureKAN parameterization remains insufficient; revisit primitive, e.g. Rational/KAT or spline.}}
$$

---

## 5. Required artifacts

每个阶段输出：

```text
runs.csv
curves.csv
edge_decomposition_audit.csv
eigenmode_audit.csv
basis_occupancy_audit.csv
nfs_projection_audit.csv
failure_table.csv
aggregate_decision.json
```

最终文档输出：

```text
docs/DG-KAN_v5.0_EdgeDecomposedFunctionalTraining_结果复盘.md
```

图表输出：

```text
figures/architecture_frontier_acc_vs_phi_rbf.svg
figures/base_over_rbf_vs_acc.svg
figures/eigenmode_energy_spectrum.svg
figures/basis_occupancy_heatmap.svg
figures/exact_vs_relaxed_nfs_scatter.svg
figures/projected_over_raw_histogram.svg
figures/learn_smooth_refresh_cycle.svg
figures/event_timeline.svg
figures/paired_delta_vs_baselines.svg
```

---

## 6. 现在最重要的失败标签

v5.0 failure table 必须包含以下标签：

```text
F1_base_path_unused
F2_base_path_helps_accuracy_but_geometry_bad
F3_residual_high_mode_pressure_persists
F4_exact_nfs_overconstrained
F5_relaxed_nfs_destroys_function
F6_relaxed_nfs_no_geometry_gain
F7_refresh_cannot_recover_accuracy
F8_capacity_increases_phi_without_accuracy
F9_width_learning_unstable_or_unused
F10_functional_candidate_lags_ABRBF_AdamW
```

其中最重要的是 F4。若 Exact NFS 的 projected_over_raw 接近 0，就说明 strict nullspace direction 基本没有可用 smoothing 方向，下一步必须使用 relaxed projection，而不是继续改 KKT solver。

---

## 7. 当前不建议继续做的事情

v5.0 暂停以下方向：

```text
继续调 Sobolev alpha / beta
继续调 branch_final_scale
继续固定 shallow/deep TFU 分工
继续 exact NFS KKT solve without relaxation
继续单纯扩 basis_count / hidden_dim
继续 learnWidth alone
继续 FCAdam / FNG / BFT 无参数化改变的版本
```

这些方向已经多轮显示不是主矛盾。

---

## 8. 最终总结

v4.9 的结果给了一个新的、更有希望的解释：

$$
\boxed{\text{PureKAN functional update 失败，不只是 optimizer 失败，而是 RBF-only edge coordinate 不适合当前任务。}}
$$

v5.0 的目标是把低阶通道放回 edge 内部，并且只对 RBF residual 施加几何 smoothing：

$$
f_{ij}(x)=b_{ij}+w_{ij}x+u_{ij}\operatorname{silu}(x)+r_{ij}(x),
$$

$$
r_{ij}(x)=\sum_k c_{ijk}B_k(x).
$$

如果这条路线成功，项目就能从 “Hybrid branch functional update” 推进到真正更干净的 “edge-decomposed PureKAN functional optimizer”。如果它仍然失败，我们也会得到一个清楚结论：当前 RBF-style PureKAN 需要更换 primitive 或接受 AdamW 作为主训练动力学。
