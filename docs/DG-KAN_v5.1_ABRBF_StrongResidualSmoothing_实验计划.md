# DG-KAN v5.1：AB-RBF Edge-Decomposed Functional Training 与 Strong Residual Smoothing 实验计划

## 0. 当前结论先冻结：v5.0 不是失败，而是把问题定位清楚了

v5.0 的核心贡献不是把 PureKAN functional optimizer 一步做成，而是把过去混在一起的三个问题拆开了：

```text
1. RBF-only edge 坐标是否合适？
2. base path 是否真的承担任务？
3. RBF residual 是否可以在不破坏任务函数的情况下被 geometry smoothing？
```

v5.0 的答案非常明确：

```text
RBF-only 是瓶颈；
AB-RBF 是 architecture-positive；
base path 真的承担任务；
RBF residual functional update 有局部收益；
但当前 relaxed NFS / residual smoothing 还太弱，不能形成 all-dataset survivor。
```

所以 v5.1 不再继续问：

```text
有没有另一个 Sobolev / TFU / FNG / FCAdam 小变体？
```

而是问：

$$
\boxed{
\text{在 AB-RBF edge 坐标下，如何让 base path 学任务，RBF residual 负责可控非线性，并把 geometry 主要约束到 residual 上？}
}
$$

这意味着 v5.1 的主线从 **PureKAN coefficient-only functional optimizer** 改成 **Edge-decomposed functional training**。

---

## 1. v5.0 结果复盘与深层解释

### 1.1 P0 说明实现基本干净，但 AB-RBF 仍应进入 core-level 实现

v5.0 的 P0 通过：

```text
rows = 27
errors = 0
max nonKAN = 0
min coverage = 1.0
rollback = 0
max KKT ≈ 1e-6
max CG ≈ 1e-6
```

这说明 edge-only 参数计数、functional coverage、rollback、Relaxed NFS row-space / Woodbury solver 都能被审计。也就是说，当前结论不能主要归因于 hidden non-KAN 参数、coverage 漏掉、rollback 错误或 NFS smoke 不稳定。

但是从代码组织上看，v5.0 的 edge-decomposed PureKAN 主要是在 `run_gafu_v50.py` 中新增的实验模块，而 `dgkan_core.py` 的标准 `RBFDense` 仍是：

```python
centers = torch.linspace(-2.5, 2.5, basis_count)
self.register_buffer("centers", centers)
self.width = fixed float
self.coeff = nn.Parameter(...)
```

也就是说，core 里的默认 RBF 仍然是 fixed centers、fixed width、only coeff trainable 的实现。v5.1 的第一件事应该是把 `ABRBFDense` 提升为 core-level 模块，而不是继续只在 runner 里临时定义。否则后续不同 runner、audit、coefficient collection、geometry metrics 很容易出现不一致。

v5.1 的 core-level 原则：

```text
ABRBFDense 是正式 KAN edge primitive；
base 参数和 RBF coeff 都属于 edge 参数；
strict PureKAN 中 learnable nonKAN params 必须仍为 0；
coefficient_named_params / edge_named_params 必须按 role 收集 input / block / output 和 base / rbf。
```

---

### 1.2 P1 说明 AB-RBF 是明确正信号

v5.0 的 P1 architecture frontier 显示，AB-RBF 不是摆设。

三个 all-dataset positive 架构是：

```text
PureKAN-ABRBF-linear+silu-AdamW
PureKAN-ABRBF-linear-AdamW
PureKAN-ABRBF-linear-learnWidth-AdamW
PureKAN-ABRBF-silu-AdamW
```

其中 `ABRBF-silu` 在 MNIST、Fashion-MNIST、KMNIST 上都提高 mean accuracy；`ABRBF-linear` 也改善 MNIST / KMNIST，并且 Fashion 基本打平。BaseOnly 不是充分的，尤其在 KMNIST 上明显不够；这说明 base path 重要，但 RBF residual 仍然需要承担非线性。

这个结果把前几轮的问题重新解释成：

$$
\boxed{
\text{PureKAN-UFULL 不是单纯 optimizer 失败，而是 fixed-grid RBF-only edge 坐标也不合适。}
}
$$

RBF-only 的 edge 是：

$$
f_{ij}(x)=\sum_k c_{ijk}B_k(x).
$$

AB-RBF 的 edge 是：

$$
f_{ij}(x)=b_{ij}+w_{ij}x+u_{ij}\operatorname{silu}(x)+\sum_k c_{ijk}B_k(x).
$$

后者把低阶、尺度、近似线性、base activation 的表达通道放回了 KAN edge 内部，而不是放回 MLP stem/head。因此它仍然可以保持 strict edge-only claim。

---

### 1.3 P2 说明 RBF-only 的确在用高 Sobolev modes 承担低阶结构

P2 eigenmode audit 显示：

```text
RBFOnly high Sobolev-eigen coefficient energy ≈ 26%-27%
ABRBF-linear base-mode energy ≈ 24%-26%
ABRBF-linear high Sobolev-eigen energy ≈ 22%-23%
```

这支持一个重要解释：

$$
\boxed{
\text{RBFOnly 被迫用高 Sobolev eigenmodes 拼低阶结构。}
}
$$

这正好解释为什么之前 U-FULL / Sobolev functional update 在 PureKAN 上表现差：

```text
AdamW 可以利用高 mode 拼出任务所需的低阶结构；
Sobolev functional update 会压制这些 high-mode coefficient；
于是 geometry 变好了，但 task representation 学不起来。
```

AB-RBF 通过显式 base modes 减轻了这个问题，但没有完全消除，尤其是 KMNIST 仍然需要 RBF residual 承担更强任务压力。

---

### 1.4 P3 说明 split functional update 是正方向，但 base path 的 task dynamics 不能丢

P3 split functional update 是本轮最重要的 optimizer 结果。

它显示：

```text
baseFrozen-rbfUFULL 在所有数据集上硬失败；
baseOnlyAdam-rbfFrozen 在很多设置下很强；
baseAdam + rbf functional variants 能保留较多 holdout descent，同时降低 RBF residual phi；
baseFCAdam-rbfUFULL 在 Fashion/MNIST 上甚至超过 ABRBF-AdamW，但 KMNIST 仍滞后。
```

这说明两件事。

第一，base path 不是辅助项，而是主要 task carrier。冻结 base 后，Fashion 从 `0.7754` 掉到 `0.6048`，KMNIST 从 `0.6517` 掉到 `0.2682`，MNIST 从 `0.8626` 掉到 `0.4954`。这说明 PureKAN 里的低阶任务通道必须主动学习。

第二，RBF residual functional update 确实能降低 residual geometry。很多 `baseAdam + rbf functional` 变体都有约 `0.17-0.19` 的 `phiR red`。但在 KMNIST 上它们仍然落后 ABRBF-AdamW，说明 residual functional update 目前还在牺牲一部分 task contribution。

所以 v5.1 不能追求“所有 edge 参数都同一种 functional update”。更合理的是：

$$
\boxed{
\text{base path 用 task-learning dynamics，RBF residual 用 geometry-aware functional dynamics。}
}
$$

也就是：

$$
\Delta \theta_{base}\leftarrow \operatorname{AdamLike}(\nabla_{base}L),
$$

$$
\Delta c_{rbf}\leftarrow \operatorname{FunctionalResidualUpdate}(\nabla_{c}L, S_{rbf}, \text{teacher / trust constraints}).
$$

---

### 1.5 P4 说明 Relaxed NFS 的当前实现仍太弱

P4 relaxed NFS 没有 all-dataset survivor。现象非常有信息量：

```text
Heuristic-NFS-role-block 有小的 phiR red / curvR red；
ExactNFS-logit / logit-hidden 几乎 phiR red = 0；
RelaxedNFS 的 projected/raw 不为 0，但仍几乎没有 residual phi reduction。
```

这说明 strict nullspace projection 太保守，而当前 relaxed projection 虽然保留了投影质量，却没有转化成有效的 residual smoothing。也就是说，问题不是“投影方向完全不存在”，而是：

$$
\boxed{
\text{当前 residual smoothing proposal / step controller 没有把可用方向转化成足够的 geometry movement。}
}
$$

v5.1 不应该回到 exact nullspace。Exact NFS 已经证明会把方向投掉。v5.1 应该做 **score-based accepted residual smoothing**：允许极小 function drift，用多目标评分选择步长，而不是硬约束为 $J\Delta=0$。

---

### 1.6 P7 说明扩大容量不是直接解法

P7 capacity follow-up 支持 architecture signal，但没有解决 geometry。比如：

```text
ABRBF-linear+silu h64/b16/d4 在 KMNIST seed0 强；
ABRBF-linear 在 Fashion/KMNIST 上相对大 RBF-only 有改善；
但 h96/b24 并不会自动改善 geometry，phi_rbf 往往上升；
quantile centers 不是 clean fix。
```

所以 v5.1 不应该把主要资源放在盲目扩 hidden / basis / depth。容量可以作为 follow-up，但核心仍是 residual smoothing 与 split dynamics。

---

## 2. 代码实现审计：v5.1 必须修正和固化的点

### 2.1 `ABRBFDense` 必须进入 `dgkan_core.py`

当前 core 仍以 `RBFDense` 为主，其 centers 固定、width 固定、只有 coeff 是 `nn.Parameter`。这对 RBF-only 结论是清楚的，但对 v5.0 之后的主线不够。

v5.1 应新增：

```python
class ABRBFDense(nn.Module):
    base_kind: none / linear / silu / linear+silu
    base_bias: Optional[nn.Parameter]
    base_linear: Optional[nn.Parameter]
    base_silu: Optional[nn.Parameter]
    rbf_coeff: nn.Parameter
    centers: buffer or optional parameter depending mode
    width: buffer / constrained parameter depending mode
```

建议默认先保持：

```text
centers fixed
width fixed
base parameters trainable
rbf_coeff trainable
```

不要马上把 centers/width 也学起来，因为 v5.0 已经显示 learnWidth alone 不是稳定正信号，quantile centers 也不是 clean fix。等 base + residual 更新稳定后，再做 center/width。

---

### 2.2 `coefficient_named_params()` 要升级为 `edge_named_params()`

过去 `coefficient_named_params()` 的语义是“找 RBF coeff”。但 AB-RBF 后，edge 参数不只有 RBF coeff。

建议拆成：

```python
def edge_named_params(model):
    return all edge-owned learnable params


def rbf_residual_named_params(model):
    return only RBF residual coeff


def base_named_params(model):
    return base bias / linear / silu params
```

并且每个参数必须记录 role：

```text
role_depth = input / block / output
role_edge = base / rbf
role_block_index = 0,1,2,...
```

v5.1 的 functional coverage 不能只报告一个 `coverage=1.0`。必须报告：

```text
coverage_edge_total
coverage_base
coverage_rbf
coverage_input
coverage_block
coverage_output
learnable_nonKAN_params
```

这样才能避免“base 参数是否被算成 non-KAN”这种解释风险。

---

### 2.3 split geometry 指标要从审计变成训练内指标

v5.0 已经记录：

```text
phi_base_p95
phi_rbf_p95
phi_total_p95
curvature_base
curvature_rbf
curvature_total
sobolev_rbf_norm
```

v5.1 应把它们变成训练 controller 的一部分。

关键原则：

```text
base derivative 不应该被当成坏 geometry；
RBF residual roughness 才是主要 geometry-control 对象。
```

因此 gate 不应再只看 `phi_total`。应主要看：

$$
\phi_{rbf,p95},\quad \operatorname{curvature}_{rbf},\quad \|c_{rbf}\|_{S_{rbf}}.
$$

同时保留 total geometry 作为安全指标。

---

### 2.4 `base/RBF` 比值在 BaseOnly 行需要修正显示

v5.0 表里 BaseOnly 的 `base/RBF` 出现极大数值，这是因为 RBF norm 为 0。这个值不能参与均值、gate 或图表坐标轴。v5.1 应处理为：

```text
if rbf_norm < eps:
    base_over_rbf = NaN
    base_fraction = 1.0
else:
    base_over_rbf = base_norm / rbf_norm
    base_fraction = base_norm / (base_norm + rbf_norm)
```

主报告应使用 `base_fraction`，而不是单独使用 `base/RBF`。

---

### 2.5 Relaxed NFS 的当前 projection 不应只报告 `proj/raw`

v5.0 的 `proj/raw` 说明投影后仍有方向，但不说明这个方向是否真的降低 residual geometry。因此 v5.1 必须记录：

```text
raw proposal geometry descent
projected proposal geometry descent
accepted proposal geometry descent
predicted phiR reduction
actual phiR reduction
predicted curvature reduction
actual curvature reduction
score before/after line search
```

否则我们无法判断 Relaxed NFS 是 proposal 不对，还是 line search / trust radius 太保守。

---

## 3. v5.1 的核心方法：AB-RBF + Strong Accepted Residual Smoothing

v5.1 不再把 RBF residual smoothing 写成硬 nullspace problem，而是写成一个 constrained score-maximization step。

给定 teacher 模型 $\theta_T$，当前模型 $\theta$，只更新 RBF residual coefficient $c$。定义 proposal $d_c$ 后，用步长 $\eta$ 得到候选：

$$
c' = c + \eta d_c.
$$

候选需要满足软约束：

$$
\operatorname{KL}(p_T\|p_{\theta'}) \leq \epsilon_{KL},
$$

$$
\frac{\|z_{\theta'}-z_T\|}{\|z_T\|+\epsilon} \leq \epsilon_z,
$$

$$
\frac{\|h_{\theta'}-h_T\|}{\|h_T\|+\epsilon} \leq \epsilon_h,
$$

$$
\Delta Acc \leq \epsilon_{acc}.
$$

但目标不是让 drift 等于 0，而是最大化：

$$
\operatorname{Score}(\eta)
=
\alpha_\phi \operatorname{Red}_{\phi_{rbf}}(\eta)
+
\alpha_\kappa \operatorname{Red}_{\kappa_{rbf}}(\eta)
+
\alpha_S \operatorname{Red}_{S_{rbf}}(\eta)
-
\beta_z D_z(\eta)
-
\beta_h D_h(\eta)
-
\beta_m D_m(\eta)
-
\beta_L \max(0, L_{hold}(\eta)-L_{hold}(0)).
$$

接受条件：

$$
\operatorname{Score}(\eta)>0,
$$

并且硬安全约束不被违反。这样 relaxed smoothing 不再追求严格 nullspace，而是允许极小 function drift 换取真实 residual geometry movement。

---

## 4. v5.1 实验计划总览

v5.1 的目标不是立刻 10-seed confirm，而是把 v5.0 的两个正信号变成可持续训练方案：

```text
正信号 1: AB-RBF architecture-positive
正信号 2: split functional update 可以降低 residual phi
```

v5.1 的阶段：

```text
P0: core implementation smoke
P1: AB-RBF architecture confirm, 5-seed
P2: split geometry calibration and gate validation
P3: residual smoothing proposal audit
P4: score-based accepted residual smoothing
P5: learn -> smooth -> refresh v3
P6: event-driven multi-cycle controller
P7: residual functional training from scratch
P8: capacity / basis follow-up
P9: 3-seed full-budget candidate selection
P10: 5-seed confirm
P11: 10-seed final confirm
```

---

## 5. P0：core implementation smoke

### 5.1 目标

P0 只验证实现是否干净，不做性能结论。

### 5.2 必跑模型

```text
PureKAN-RBFOnly
PureKAN-ABRBF-linear
PureKAN-ABRBF-silu
PureKAN-ABRBF-linear+silu
PureKAN-BaseOnly-linear
PureKAN-BaseOnly-silu
```

### 5.3 必查 invariant

每个 dataset / model 都记录：

```text
learnable_nonKAN_params
edge_param_count_total
base_param_count
rbf_param_count
coverage_edge_total
coverage_base
coverage_rbf
coverage_input
coverage_block
coverage_output
rollback_max_abs_error
finite_forward
finite_backward
finite_split_geometry
```

通过条件：

$$
\text{learnable\_nonKAN\_params}=0.
$$

$$
\text{coverage\_edge\_total}=1.
$$

$$
\text{coverage\_base}=1,\quad \text{coverage\_rbf}=1.
$$

$$
\text{rollback\_max\_abs\_error}<10^{-8}.
$$

P0 还要输出 `edge_param_manifest.csv`，逐参数列出：

```text
name
shape
role_depth
role_edge
optimizer_group
requires_grad
included_in_edge_update
included_in_base_update
included_in_rbf_update
```

---

## 6. P1：AB-RBF architecture confirm，5-seed

### 6.1 目标

v5.0 的 architecture frontier 是 3-seed / compact setting。v5.1 要先确认 AB-RBF 不是 seed artifact。

### 6.2 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### 6.3 方法

```text
PureKAN-RBFOnly-AdamW
PureKAN-ABRBF-linear-AdamW
PureKAN-ABRBF-silu-AdamW
PureKAN-ABRBF-linear+silu-AdamW
PureKAN-BaseOnly-linear-AdamW
PureKAN-BaseOnly-silu-AdamW
MLP-AdamW
```

### 6.4 设置

```text
hidden_dim = 64
basis_count = 16
depth = 4
train / val / test = current compact setting first
seeds = 0..4
```

### 6.5 记录指标

任务指标：

```text
test_acc
val_loss_auc
train_loss_auc
ECE
NLL
classwise_acc
margin_mean
margin_p10
rank_block
rank_input
rank_output
```

edge decomposition：

```text
base_norm
rbf_norm
base_fraction
base_over_rbf, only if rbf_norm > eps
base_ablation_drop
rbf_ablation_drop
base_only_forward_acc
rbf_only_forward_acc
base_logit_delta_norm
rbf_logit_delta_norm
base_margin_contribution
rbf_margin_contribution
```

geometry：

```text
phi_base_p95
phi_rbf_p95
phi_total_p95
curvature_base
curvature_rbf
curvature_total
sobolev_rbf_norm
jacobian_condition_total
```

### 6.6 可视化

必须画：

```text
1. acc by method and dataset, with seed std/CI
2. acc vs phi_rbf scatter
3. acc vs phi_total scatter
4. base_fraction distribution by layer and dataset
5. base_ablation_drop vs rbf_ablation_drop scatter
6. rank_block vs acc scatter
7. classwise accuracy heatmap for RBFOnly vs ABRBF
```

### 6.7 判定

P1 通过不是要求 AB-RBF 全面超过所有 baseline，而是要求：

```text
ABRBF-linear/silu/linear+silu 至少一个在三个数据集上 mean acc >= RBFOnly - 0.5%
并且至少两个数据集上 mean acc > RBFOnly
base_fraction 非平凡，即 0.1 < base_fraction < 0.9
BaseOnly 不能全面替代 AB-RBF
```

如果 P1 不通过，则 v5.0 的 architecture-positive 信号不稳，v5.1 不进入 optimizer 阶段。

---

## 7. P2：split geometry calibration

### 7.1 目标

重新定义 geometry gate。过去 total phi 会把 base path 的线性导数也算成坏几何，这对 AB-RBF 不公平。

P2 要建立三个分开的结论：

```text
base geometry: 是否只是合理低阶导数；
RBF residual geometry: 是否真的粗糙；
total geometry: 是否会造成 credit/Jacobian 风险。
```

### 7.2 方法

使用 P1 训练好的 teacher：

```text
RBFOnly-AdamW
ABRBF-linear-AdamW
ABRBF-silu-AdamW
ABRBF-linear+silu-AdamW
```

### 7.3 指标

每层记录：

```text
phi_base_p50/p95/max
phi_rbf_p50/p95/max
phi_total_p50/p95/max
curvature_base_p95
curvature_rbf_p95
curvature_total_p95
sobolev_rbf_norm
jacobian_condition_total
jacobian_condition_without_rbf
jacobian_condition_without_base
```

还要记录 RBF residual 的 eigenmode：

```text
low_mode_coeff_energy
mid_mode_coeff_energy
high_mode_coeff_energy
low_mode_grad_energy
mid_mode_grad_energy
high_mode_grad_energy
base_mode_energy
```

### 7.4 可视化

```text
1. phi_base / phi_rbf / phi_total stacked bar
2. curvature_base / curvature_rbf / curvature_total stacked bar
3. high Sobolev eigenmode energy by method
4. layerwise phi_rbf heatmap
5. jacobian total vs residual-only scatter
6. base_fraction vs high_mode_energy scatter
```

### 7.5 输出新 gate

P2 要产出 `split_geometry_gate.json`，建议默认：

```text
primary geometry gate:
  phi_rbf_reduction or curvature_rbf_reduction

secondary safety gate:
  jacobian_condition_total not worse than teacher by > 20%

not primary:
  phi_total if base path contains linear/silu derivative
```

---

## 8. P3：residual smoothing proposal audit

### 8.1 目标

先不训练，只看单次 residual smoothing proposal 是否有真实 geometry movement。

### 8.2 Teacher

```text
ABRBF-linear-AdamW
ABRBF-silu-AdamW
ABRBF-linear+silu-AdamW
```

### 8.3 Proposal variants

```text
S0: residual Laplacian smoothing
S1: residual Sobolev-gradient smoothing
S2: residual eigenmode shrink, high modes only
S3: residual eigenmode shrink, high+mid modes
S4: residual coefficient L2 shrink
S5: data-aware residual smoothing
S6: heuristic role-block residual smoother, v5.0 reference
S7: score-gradient proposal, maximize split-geometry score approximation
```

### 8.4 Projector / controller variants

```text
C0: no projection, line search only
C1: logit KL trust
C2: logit + hidden trust
C3: logit + hidden + margin trust
C4: score-based accepted step
C5: score-based accepted step + adaptive eta grid
```

### 8.5 指标

每个 proposal 记录 before / after / projected / accepted：

```text
raw_direction_norm
projected_direction_norm
projected_over_raw
predicted_phiR_red
actual_phiR_red
predicted_curvR_red
actual_curvR_red
predicted_sobolev_red
actual_sobolev_red
KL_teacher_student
logit_relative_drift
hidden_relative_drift
margin_relative_drift
acc_drop
holdout_loss_delta
score
accepted_eta
accept_rate
reject_reason
```

### 8.6 可视化

```text
1. predicted vs actual phiR reduction scatter
2. projected_over_raw vs actual phiR reduction scatter
3. KL/logit drift vs phiR reduction Pareto
4. accepted_eta histogram
5. rejection reason stacked bar
6. proposal score heatmap by dataset / teacher / proposal / controller
```

### 8.7 判定

P3 survivor 条件：

$$
\Delta Acc \leq 0.5\%.
$$

$$
\operatorname{KL}<0.01.
$$

$$
\operatorname{logit\_drift}<0.03.
$$

$$
\operatorname{phiR\_red}>5\% \quad \text{or} \quad \operatorname{curvR\_red}>10\%.
$$

并且至少在两个数据集上满足，才进入 P4。

---

## 9. P4：score-based accepted residual smoothing

### 9.1 目标

P4 把 P3 的 proposal 放进一个小步 accepted update 中，验证它是否能连续执行 5 到 20 步而不破坏 task。

### 9.2 更新形式

只更新 RBF residual：

$$
c_{t+1}=c_t+\eta_t d_t.
$$

base path 冻结，teacher 固定。每步通过 score-based accept：

$$
\operatorname{Score}
=
\alpha_\phi \operatorname{Red}_{\phi_{rbf}}
+
\alpha_C \operatorname{Red}_{curv_{rbf}}
+
\alpha_S \operatorname{Red}_{S_{rbf}}
-
\beta_z D_z
-
\beta_h D_h
-
\beta_m D_m
-
\beta_L \max(0,\Delta L_{hold}).
$$

默认权重：

```text
alpha_phi = 1.0
alpha_C = 0.5
alpha_S = 0.2
beta_z = 2.0
beta_h = 1.0
beta_m = 1.0
beta_L = 3.0
```

### 9.3 方法

```text
P4-A: best P3 proposal, 5 steps
P4-B: best P3 proposal, 20 steps
P4-C: adaptive proposal choice among S0/S2/S6/S7
P4-D: role-block only
P4-E: role-cycle input/block/output residual smoothing
```

### 9.4 指标

```text
stepwise accepted_eta
accept_rate
score_curve
phiR_curve
curvR_curve
sobolev_rbf_curve
KL_curve
logit_drift_curve
hidden_drift_curve
margin_curve
acc_curve
holdout_loss_curve
rank_curve
base_fraction_curve
```

### 9.5 可视化

```text
1. phiR / acc two-axis curve over smoothing steps
2. score curve with accepted/rejected markers
3. KL and logit drift curves
4. margin and rank curves
5. proposal selection stacked area plot
6. residual geometry Pareto front before/after smoothing
```

### 9.6 判定

P4 survivor：

```text
acc drop <= 0.5%
KL < 0.02
phiR red > 8% or curvR red > 15%
holdout loss not worse by > 2%
```

P4 如果没有 survivor，就不进入 Learn-Smooth-Refresh。此时说明 residual smoothing proposal 仍然不足，需要回到 proposal 设计。

---

## 10. P5：Learn -> Smooth -> Refresh v3

### 10.1 目标

验证完整 one-cycle：

```text
learn task with AB-RBF
smooth RBF residual with accepted controller
refresh task while preserving residual geometry
```

### 10.2 Phase A：Learn

方法：

```text
ABRBF-AdamW
ABRBF-baseAdam-rbfAdamW
ABRBF-baseAdam-rbfFCAdam
ABRBF-baseFCAdam-rbfAdamW
```

学习到 teacher snapshot：

```text
T20
T100
Tfinal
```

### 10.3 Phase B：Smooth

使用 P4 survivor，只更新 RBF residual。base path 固定。

### 10.4 Phase C：Refresh

Refresh variants：

```text
R0: base-only AdamW refresh, RBF frozen
R1: base AdamW + tiny RBF task update
R2: base FCAdam + tiny RBF task update
R3: D6-style small task refresh on RBF only
R4: no refresh, smoothing only
```

### 10.5 记录指标

每个 phase 都记录：

```text
test_acc
val_loss_auc
ECE
NLL
rank_block
margin_mean/p10
base_fraction
base_ablation_drop
rbf_ablation_drop
phi_rbf_p95
curvature_rbf
sobolev_rbf_norm
jacobian_total
KL_to_teacher
logit_drift
hidden_drift
```

### 10.6 可视化

```text
1. learn-smooth-refresh phase plot: acc / phiR / KL / rank
2. refresh recovery plot: acc recovered vs phiR retained
3. base vs RBF ablation drop before and after smoothing
4. task loss vs residual geometry Pareto by phase
```

### 10.7 判定

P5 survivor 条件：

```text
After Refresh:
  acc >= Learn_acc - 0.5%
  phiR <= Learn_phiR * 0.90
  ECE not worse by > 0.02
  rank >= Learn_rank * 0.90
  KL_to_learn_teacher < 0.03
```

必须三个数据集都至少有一个 survivor 才进入 P6。

---

## 11. P6：event-driven multi-cycle controller

### 11.1 目标

v4.8/v5.0 都说明固定 cycle 不稳定。v5.1 改用 event-driven controller。

### 11.2 触发条件

只在以下条件满足时 smoothing：

```text
1. validation loss plateau: recent improvement < threshold
2. rank and margin stable: rank drop < threshold, margin p10 stable
3. residual geometry over budget: phiR > target or curvR > target
4. base path contribution stable: base_fraction variation small
```

形式上，若：

$$
\Delta L_{val}^{recent}<\epsilon_L,
$$

$$
\phi_{rbf}>\phi_{target},
$$

$$
\Delta rank > -\epsilon_r,
$$

才触发 smoothing。

### 11.3 Controller variants

```text
E0: no smoothing, ABRBF-AdamW baseline
E1: fixed cycle, v5.0 reference
E2: event-driven smoothing, conservative
E3: event-driven smoothing, aggressive
E4: event-driven smoothing + base-only refresh
E5: event-driven smoothing + adaptive proposal selection
```

### 11.4 指标

```text
num_smoothing_events
trigger_reason
accepted_steps_per_event
phiR_reduction_per_event
acc_drop_per_event
refresh_recovery_rate
cycle_stability_score
failure_reason
```

### 11.5 可视化

```text
1. timeline with smoothing events marked
2. acc/val-loss/phiR/rank/margin over time
3. event waterfall: before -> after smooth -> after refresh
4. smoothing trigger heatmap
5. per-event accepted eta histogram
```

### 11.6 判定

P6 survivor：

```text
final acc >= ABRBF-AdamW - 0.75%
final phiR red >= 10%
final ECE not worse by > 0.02
no more than one catastrophic event with acc drop > 1%
```

---

## 12. P7：residual functional training from scratch

### 12.1 目标

P5/P6 是 posthoc / phase-separated。P7 测训练时就使用 split dynamics：base 学任务，RBF residual 受 geometry 控制。

### 12.2 方法

```text
ABRBF-AdamW
ABRBF-baseAdam-rbfUFULL
ABRBF-baseAdam-rbfD6
ABRBF-baseAdam-rbfFCAdam-dataSob
ABRBF-baseAdam-rbfResidualProxAdam
ABRBF-baseFCAdam-rbfUFULL
ABRBF-baseAdam-rbfAdamW+lateResidualSmooth
```

新增 `rbfResidualProxAdam`：

先算 RBF residual 的 AdamW proposal $d_{Adam}$，再解：

$$
\Delta c
=
\arg\min_{\Delta}
\frac{1}{2}\|\Delta-d_{Adam}\|_{D_t}^2
+
\frac{\lambda}{2}\|c+\Delta\|_{S_{rbf}}^2.
$$

闭式形式近似为：

$$
\Delta
=
(D_t+\lambda S_{rbf}+\rho I)^{-1}D_t d_{Adam}.
$$

直觉：

```text
AdamW proposal 负责 task learning；
Sobolev prox 只修正 residual geometry。
```

这比直接 $S^{-1}g$ 更符合 v5.0 的经验。

### 12.3 指标

除了常规指标，P7 必须记录：

```text
cos_rbf_update_with_adam
rbf_prox_projection_error
rbf_sobolev_penalty_before_after
base_update_norm
rbf_update_norm
base_update_share
rbf_update_share
high_mode_energy_over_time
```

### 12.4 判定

P7 survivor：

```text
acc >= ABRBF-AdamW - 0.75%
val_loss_auc >= ABRBF-AdamW - 5%
phiR red >= 10%
ECE not worse by > 0.02
base_fraction remains nontrivial
```

如果 P7 survivor 出现，再进入 P9/P10 seed selection。

---

## 13. P8：capacity / basis follow-up，只作为辅助

P8 不做大扫，只测 v5.0 已经显示有意义的组合：

```text
ABRBF-linear h64/b16/d4
ABRBF-linear+silu h64/b16/d4
ABRBF-silu h64/b16/d4
ABRBF-linear h96/b24/d4
ABRBF-linear+silu h96/b24/d4
```

不优先测：

```text
quantile centers, unless P2 basis coverage shows severe out-of-grid
learnWidth alone, unless P1/P2 shows fixed width is primary blocker
```

P8 只回答：

```text
candidate 是否对 capacity 敏感？
是否需要 h96/b24 才能过 gate？
更大容量是否只是提高 acc 但恶化 phiR？
```

---

## 14. P9：3-seed full-budget candidate selection

P9 只允许从 P5/P6/P7 survivors 进入。

候选最多 4 个：

```text
1. ABRBF-AdamW baseline
2. best P5 one-cycle candidate
3. best P6 event-driven candidate
4. best P7 training-from-scratch candidate
```

数据集：

```text
MNIST
Fashion-MNIST
KMNIST
```

seeds：

```text
0,1,2
```

判定：

```text
进入 P10 的候选必须：
  acc gap <= 0.75% vs ABRBF-AdamW on all datasets
  phiR red >= 10% on at least two datasets
  no catastrophic failure
```

---

## 15. P10：5-seed confirm

P10 使用 seeds 0..4。

必须报告 paired delta vs ABRBF-AdamW：

```text
paired_acc_delta
paired_val_auc_delta
paired_phiR_delta
paired_curvR_delta
paired_ECE_delta
paired_rank_delta
```

通过条件：

$$
\operatorname{mean}(\Delta Acc) \geq -0.75\%.
$$

$$
\operatorname{mean}(\Delta \phi_{rbf}) \leq -10\%.
$$

$$
\operatorname{ECE}_{candidate}\leq \operatorname{ECE}_{AdamW}+0.02.
$$

如果 P10 通过，再进入 P11。

---

## 16. P11：10-seed final confirm

P11 使用 seeds 0..9，只确认最终 1 个 candidate。

输出最终结论分三档。

### 16.1 strong pass

```text
acc gap <= 0.5%
phiR red >= 15%
curvR red >= 15%
ECE not worse
rank retained >= 90%
```

可写：

```text
AB-RBF edge-decomposed functional training provides a geometry-improving PureKAN optimizer.
```

### 16.2 Pareto pass

```text
acc gap <= 1.0%
phiR red >= 10%
ECE not worse by > 0.02
```

可写：

```text
AB-RBF split functional training is a geometry Pareto optimizer, but AdamW remains accuracy baseline.
```

### 16.3 fail

如果不满足：

```text
AB-RBF improves architecture, but functional residual smoothing remains a posthoc diagnostic rather than a train-time optimizer.
```

---

## 17. v5.1 必须产出的文件

```text
results/v5_1/
  p0_core_smoke.csv
  edge_param_manifest.csv
  p1_architecture_confirm5.csv
  p2_split_geometry_calibration.csv
  p3_residual_smoothing_proposal_audit.csv
  p4_accepted_residual_smoothing.csv
  p5_learn_smooth_refresh_v3.csv
  p6_event_driven_tan_v2.csv
  p7_split_residual_training.csv
  p8_capacity_basis_followup.csv
  p9_candidate_selection3.csv
  p10_confirm5.csv
  p11_confirm10.csv
  paired_delta_vs_ABRBF_AdamW.csv
  failure_table.csv
  aggregate_decision.json
  figures/
```

Figures：

```text
figures/p1_architecture_frontier_acc_phiR.svg
figures/p1_base_rbf_ablation.svg
figures/p2_split_geometry_stacked.svg
figures/p2_eigenmode_energy.svg
figures/p3_predicted_vs_actual_phiR.svg
figures/p3_drift_vs_geometry_pareto.svg
figures/p4_score_eta_acceptance.svg
figures/p5_learn_smooth_refresh_timeline.svg
figures/p6_event_waterfall.svg
figures/p7_training_curves_acc_phiR.svg
figures/p10_paired_delta_forest.svg
```

---

## 18. 失败诊断表

每个失败 row 必须打标签：

```text
F1_ABRBF_not_architecture_positive
F2_base_path_unused
F3_base_path_overdominates_rbf
F4_residual_high_mode_pressure_persists
F5_smoothing_no_geometry_gain
F6_smoothing_breaks_task
F7_refresh_cannot_recover_task
F8_event_controller_over_smooths
F9_candidate_lags_ABRBF_AdamW
F10_capacity_increases_phiR
F11_implementation_invariant_failed
```

最终报告要按 dataset / method / failure_type 做 heatmap。

---

## 19. 预期结论模板

### 如果结果好

```text
v5.1 confirms that PureKAN functional training requires edge decomposition.
Base paths learn low-order task structure, while RBF residuals can be geometry-controlled through score-based accepted residual smoothing.
The resulting AB-RBF split functional optimizer matches ABRBF-AdamW accuracy within a small gap while reducing residual roughness and preserving calibration.
```

### 如果结果中等

```text
v5.1 confirms AB-RBF as the correct PureKAN architecture, but residual functional smoothing remains a Pareto/posthoc component.
It can reduce residual roughness with small task drift, yet train-time candidates still lag ABRBF-AdamW on KMNIST.
```

### 如果结果失败

```text
v5.1 shows that AB-RBF fixes part of the parameterization bottleneck, but current RBF residual smoothing is not strong enough to serve as a training optimizer.
PureKAN functional optimization should be paused or reframed around a different primitive / residual parameterization.
```

---

## 20. 最终一句话

v5.1 的核心不是再证明 RBF-only U-FULL，而是验证：

$$
\boxed{
\text{AB-RBF base path 学任务，RBF residual 做可控几何，是否能形成真正 PureKAN functional training。}
}
$$

如果这个方向也失败，那么我们应明确承认：

```text
当前 functional update 的有效范围仍主要是 Hybrid-DGKAN branch / posthoc smoothing，
而不是完整 PureKAN train-time optimizer。
```
