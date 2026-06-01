# DG-KAN v4.9：RBF 参数化重审、Exact NFS 与 PureKAN 功能几何路线

> 目标：基于 v4.8 结果和当前 `dgkan_core / run_gafu_v48 / analyze_gafu_v48` 实现，重新审视 functional update 的失败原因，明确区分“optimizer 失败”和“RBF-only 参数化失败”，并给出下一步可执行实验计划。  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`，不使用 `\[\]`。

---

## 0. 当前结论先行

v4.8 的核心进展不是找到最终 optimizer，而是把问题边界显著缩窄了。当前最重要的判断是：

$$
\boxed{
\text{PureKAN functional update 的失败，不能再只归因于某个 optimizer 超参。}
}
$$

更准确地说，当前失败同时涉及三件事：

```text
1. 当前 PureKAN 是 fixed-grid RBF-only coefficient 网络；
2. 当前 functional update 主要作用在 coefficient 上，centers / width / base path 均未参与学习；
3. 当前 v4.8 的 NFS 更像 teacher-constrained smoothing proposal，不是真正的 Jacobian-nullspace projection。
```

因此，下一轮不能继续只调：

```text
Sobolev alpha / beta
branch scale
warmup length
trust radius
refresh steps
```

而应该改成：

$$
\boxed{
\text{先验证 RBF 参数化是否给 functional optimizer 提供了合适坐标，再验证真正的函数保持型 smoothing。}
}
$$

本计划将 v4.9 的主问题定义为：

$$
\boxed{
\text{PureKAN 失败到底是 optimizer 问题，还是 RBF-only 参数化问题，还是 NFS 实现过弱？}
}
$$

---

## 1. v4.8 实验结果复盘

### 1.1 P0：实现 smoke 通过，但只说明路径可运行

v4.8 的 P0 显示：

```text
rows = 21
errors = 0
max nonKAN = 0
min coverage = 1.0
max rollback = 0
max CG residual = 0.0533
pass = true
```

这说明：

```text
1. strict PureKAN 仍然没有 non-KAN trainable params；
2. coefficient coverage 是完整的；
3. NFS temporary apply / rollback / teacher snapshot 基本可用；
4. finite projection diagnostics 没有明显数值爆炸。
```

所以这一轮不能再主要归因于：

```text
alpha 没 fixed
input/output KAN 没被更新
rollback 出错
teacher snapshot 出错
```

但 P0 不证明 NFS 是数学意义上的 nullspace projection，也不证明 PureKAN 存在准确且几何好的解。

---

### 1.2 P1：AdamW + 简单几何正则没有暴露 smooth accurate frontier

P1 的目标是问：

$$
\boxed{
\text{PureKAN 是否存在 AdamW 能找到的准确且几何更好的解？}
}
$$

结果是没有 all-dataset survivor。

典型结果如下。

Fashion-MNIST：

```text
A0-PureKAN-AdamW acc = 0.7754
A1/A2/A3 Sobolev regularized acc = 0.7754
phi red ≈ 0
J red ≈ 0
A8 posthoc geometry acc = 0.5664
```

KMNIST：

```text
A0-PureKAN-AdamW acc = 0.6797
A1/A2/A3 Sobolev regularized acc = 0.6797
phi red ≈ 0
J red ≈ 0
A8 posthoc geometry acc = 0.1973
```

MNIST：

```text
A0-PureKAN-AdamW acc = 0.8711
A1/A2/A3 Sobolev regularized acc = 0.8711
small J reduction only on some rows
A8 posthoc geometry acc = 0.2148
```

这说明两件事。

第一，当前弱几何正则几乎不起作用：

$$
\text{AdamW} + \lambda R_{geom}
\approx
\text{AdamW}
$$

其中 $\lambda$ 很小时，accuracy 保住了，但 geometry 没明显改善。

第二，强行 posthoc smoothing 会破坏 task function：

$$
\text{geometry push too hard}
\Rightarrow
\text{accuracy collapse}.
$$

所以 P1 的结论不是“smooth accurate PureKAN 不存在”，而是：

$$
\boxed{
\text{当前实现中的简单几何正则，没有找到 smooth accurate frontier。}
}
$$

这仍然留下两个开放可能：

```text
1. 需要更好的 RBF 参数化；
2. 需要真正 function-preserving projection，而不是普通几何正则。
```

---

### 1.3 P2：NFS-role-block 是本轮最强正信号

P2 从 AdamW teacher 出发，做 offline smoothing projection。这里出现了大量 `NFS-role-block` pass。

Fashion-MNIST 上，典型成功行为是：

```text
TeacherA-AdamW20 + NFS-role-block + diag + eta=0.05:
  acc drop = 0.0039
  KL = 0.0001
  logit drift = 0.0147
  phi red = 0.1482
  J red = 0.4284
  pass = yes
```

KMNIST 上，典型成功行为是：

```text
TeacherI-D6-100 + NFS-role-block + diag:
  acc drop = 0.0020
  KL = 0.0004
  phi red = 0.1114
  smoothability = 10.7330
  pass = yes
```

这说明：

$$
\boxed{
\text{在 teacher 附近，block-role smoothing 可以在较小函数漂移下改善几何。}
}
$$

这非常重要，因为它证明 functional geometry 并不是完全没希望。至少在局部，存在类似：

$$
\Delta \theta
\quad\text{s.t.}\quad
\Delta f \approx 0,
\quad
\Delta R_{geom} < 0.
$$

也就是说，PureKAN 的某些 teacher function 附近确实存在可用的 smoothing direction。

---

### 1.4 P3：functional teachers 也能被 NFS 平滑，说明 NFS 不是只适配 AdamW teacher

P3 从 functional teacher 出发，例如：

```text
TeacherE-FCAdam20
TeacherI-D6-100
TeacherJ-FNG-100
```

也出现了 all-dataset survivor，例如：

```text
TeacherE-FCAdam20 | NFS-role-block | diag | 0.035
TeacherI-D6-100   | NFS-role-block | diag | 0.035
TeacherJ-FNG-100  | NFS-role-block | diag | 0.035
```

这说明：

$$
\boxed{
\text{NFS-role-block 是一个真实的局部几何 projection primitive。}
}
$$

它不是只对 AdamW teacher 工作，也不是纯偶然。

但它仍然只是局部 primitive，不是完整 optimizer。

---

### 1.5 P4：TAN one-cycle 成立，说明 Learn -> Smooth -> Refresh 有一次性价值

P4 显示，一次：

```text
Task teacher
-> NFS-role-block smoothing
-> D6 or FCAdam refresh
```

可以在多个数据集上通过。

典型结果：

```text
Fashion, PureKAN-AdamW teacher + NFS-role-block + D6-5:
  task acc = 0.7734
  final acc = 0.7852
  acc drop = -0.0117
  phi red = 0.1370
  pass = yes

KMNIST, PureKAN-AdamW teacher + NFS-role-block + D6-5:
  task acc = 0.6953
  final acc = 0.7031
  acc drop = -0.0078
  phi red = 0.1319
  pass = yes
```

这说明：

$$
\boxed{
\text{一次 Learn -> Smooth -> Refresh 是可行的。}
}
$$

它也说明之前的想法“先学任务，再做几何 projection”不是错的。

---

### 1.6 P5：多 cycle 失败，核心是 controller 不稳定，不是 NFS 完全无效

P5 的 alternating TAN cycles 结果是：

```text
Fashion: pass
KMNIST: pass
MNIST: fail
```

典型 P5 最终结果：

```text
Fashion:
  final acc = 0.7773
  final phi = 0.1143 / 0.1199
  pass = yes

KMNIST:
  final acc = 0.6621
  final phi = 0.1295 / 0.1319
  pass = yes

MNIST:
  final acc = 0.8242
  final phi = 0.1483 / 0.1489
  pass = no
```

所以 P5 的结论不是：

```text
NFS/TAN 完全失败。
```

而是：

$$
\boxed{
\text{当前 multi-cycle controller 不能稳定保持 task/geometry tradeoff。}
}
$$

特别是 MNIST 上，cycles 后的 task/geometry 平衡不能保持。下一步如果继续 TAN，不应该继续扩大 seed，而应该重新设计 cycle controller。

---

## 2. 代码实现审计：当前还有哪些关键限制

### 2.1 当前 strict PureKAN 是 fixed-grid RBF coefficient-only 网络

当前 `RBFDense` 的核心是：

```python
centers = torch.linspace(-2.5, 2.5, basis_count)
self.register_buffer("centers", centers)
self.width = float((centers[1] - centers[0]).abs() * 1.4)
self.coeff = nn.Parameter(...)
```

也就是说：

```text
centers 是 buffer，不训练；
width 是 float，不训练；
coeff 是 Parameter，训练；
PureKAN 中 bias=False；
PureKAN 中没有 Linear stem/head，没有 learnable LayerNorm，没有 alpha 参数。
```

PureKAN 的结构是：

```text
input_kan: RBFDense(input_dim -> hidden_dim, bias=False)
blocks: PureResidualKANBlock(... RBFDense(hidden -> hidden, bias=False))
output_kan: RBFDense(hidden_dim -> num_classes, bias=False)
```

所以 strict PureKAN 实际是：

$$
\boxed{
\text{fixed centers + fixed width + coeff-only RBF edge network.}
}
$$

这非常关键。因为现在我们说“PureKAN functional update 失败”，其实更准确是：

$$
\boxed{
\text{fixed-grid RBF-only PureKAN 在当前 functional update 下失败。}
}
$$

它不等于所有 KAN 参数化都失败。

---

### 2.2 当前 PureKAN 缺少 base / linear / identity 通道

当前 RBF edge 是：

$$
f_{ij}(x)=\sum_{k=1}^K c_{ijk}B_k(x).
$$

没有：

$$
b_{ij},\quad w_{ij}x,\quad w_{ij}^{base}\operatorname{silu}(x).
$$

但许多 KAN / spline-KAN 实现都有 base path 或 residual base function：

$$
f_{ij}(x)=w_{ij}^{base}b(x)+\sum_k c_{ijk}B_k(x).
$$

当前 PureKAN 要用 RBF basis 去拼出所有低阶、线性、尺度变换和分类 boundary。AdamW 能训练，是因为它可以自由地在 coefficient space 里拼；functional/Sobolev 更新会倾向于平滑、压制高频和某些组合模式。

这可能解释为什么：

```text
Hybrid-DGKAN 成功；
PureKAN-UFULL/TFU/FNG/FLD 反复失败。
```

Hybrid-DGKAN 有 stem/head/LN/residual identity，提供了便宜的低阶通道。PureKAN 没有。

因此，下一步必须验证：

$$
\boxed{
\text{把 base path 放回 KAN edge 内部，是否能让 PureKAN functional update 成立？}
}
$$

---

### 2.3 当前 NFS 不是严格的 Jacobian nullspace projection

从 `run_gafu_v48.py` 的实现看，`_nfs_project_once` 大致流程是：

```text
1. 用 _geometry_smooth_updates 生成 smoothing proposal；
2. 根据 projector 类型乘一个 projector_scale；
3. temporary apply；
4. 检查 KL / logit drift / hidden drift / margin drift；
5. backtracking；
6. accept 或 rollback。
```

其中 projector 是：

```text
diag -> 1.0
cg5 -> 0.85
cg10 -> 0.70
lowrank32 -> 0.55
lowrank64 -> 0.45
```

这说明当前 NFS 更准确地说是：

$$
\boxed{
\text{teacher-constrained smoothing proposal with backtracking.}
}
$$

而不是真正求解：

$$
\Delta\theta
=
-\eta
\left(
I
-
M^{-1}J_f^T(J_fM^{-1}J_f^T+\mu I)^{-1}J_f
\right)
M^{-1}\nabla R.
$$

这个差异很大。当前 P2/P3 的 positive signal 仍然有价值，但它还没有证明真正的 nullspace smoothing 可以训练。

所以 v4.9 必须实现 **Exact / Sketch NFS**，而不是继续使用 projector scale 名义上的 `cg5/cg10/lowrank`。

---

### 2.4 P1 的几何正则也不是完整几何 frontier

P1 的 `AdamW-Sobolev / PhiProxy / JacProxy` 是弱正则和 proxy 正则。它能说明当前简单正则没找到 frontier，但不能证明：

```text
准确且几何好的 PureKAN 解不存在。
```

因为真正的 geometry objective 应该至少包含：

```text
1. coefficient Sobolev norm；
2. actual phi_prime_p95 或可微 surrogate；
3. block Jacobian proxy；
4. basis occupancy / out-of-grid penalty；
5. function-preserving smoothing phase。
```

v4.8 的 P1 只完成了第一层可行性检查，不能当作最终 capacity 结论。

---

### 2.5 P6 capacity / basis expansion 没有被真正用来回答问题

`run_gafu_v48.py` 的 P6 已经预留了：

```text
(64,16,4,RBF)
(96,24,4,RBF)
(96,32,2,RBF)
(96,24,4,AB-RBF)
```

并且方法包括：

```text
A0-PureKAN-AdamW
A6-AdamW-MixedGeom
```

但 v4.8 的决策在 P5 后停止，P6 没有真正用于最终判断。

这意味着：

$$
\boxed{
\text{capacity / basis / AB-RBF 是否能改变结论，仍然是开放问题。}
}
$$

下一轮不应该跳过这个问题。

---

## 3. 重新定义当前失败模式

### 3.1 失败模式一：RBF-only 参数化可能没有给 functional optimizer 合适的低阶通道

当前 strict PureKAN 的 edge 是：

$$
f(x)=\sum_k c_kB_k(x).
$$

functional update 一直在 coefficient basis 上做文章。但如果任务需要简单线性或 affine transform，这个结构非常别扭。

因此现在要问：

$$
\boxed{
\text{PureKAN-AdamW 的成功是否依赖 RBF coefficient 拼出的低阶通道？}
}
$$

如果是，那么 functional update 压制这些通道就会导致训练失败。

---

### 3.2 失败模式二：geometry smoothing 可以局部 function-preserving，但 multi-cycle controller 不稳定

P2/P3/P4 都显示 NFS-role-block 可行；P5 显示 alternating cycles 不稳定。

这说明问题不是：

```text
完全没有平滑方向。
```

而是：

```text
如何决定什么时候平滑、平滑哪个 role、平滑多大、平滑后如何 refresh。
```

当前 controller 太粗：

```text
固定 cycles
固定 role-block
固定 refresh D6-5
固定 NFS eta
```

下一步需要事件驱动 controller，而不是固定周期。

---

### 3.3 失败模式三：当前 NFS 没有真正投影到 teacher function 的 nullspace

当前 NFS 是 proposal + constraints，不是 KKT projection。这会导致：

```text
1. 可解释性不足；
2. projector 名称和数学含义不一致；
3. 难以知道失败来自 proposal，还是来自 nullspace 不存在；
4. 多 cycle 时漂移累积更难控制。
```

因此必须实现真正的 projected smoothing。

---

### 3.4 失败模式四：centers / width 固定可能造成 basis coverage bottleneck

当前 centers 固定在：

$$
[-2.5,2.5].
$$

width 固定为中心间距的 $1.4$ 倍。

但 PureKAN 的不同层输入分布不同：

```text
input_kan: raw normalized pixels
block_kan: hidden feature
output_kan: FixedNorm(hidden)
```

一个统一固定 grid 未必适合所有 role。

所以必须记录并测试：

```text
basis occupancy entropy
out-of-grid fraction
dead basis fraction
per-role center coverage
per-role width adequacy
```

如果 basis coverage 差，那么继续优化 coefficient 没有意义。

---

## 4. v4.9 新方向

v4.9 不再继续设计“新 functional optimizer”，而是先解决三个基础问题。

### 4.1 Base-RBF：把低阶通道放回 KAN edge 内部

实现：

$$
f_{ij}(x)
=
\beta_{ij}^{0}
+
\beta_{ij}^{1}x
+
\sum_{k=1}^K c_{ijk}B_k(x).
$$

或：

$$
f_{ij}(x)
=
w_{ij}^{base}\operatorname{silu}(x)
+
\sum_{k=1}^K c_{ijk}B_k(x).
$$

注意，这不是加回 MLP，而是把 base path 作为 KAN edge 的一部分。

参数仍然属于 edge function：

```text
base_const
base_linear
base_silu_weight
rbf_coeff
```

这样 PureKAN 仍可保持“所有可学习参数都是 KAN edge function 参数”的定义。

---

### 4.2 Exact NFS：实现真正的 function-preserving geometry projection

目标是求解：

$$
\min_{\Delta\theta}
\quad
\nabla R(\theta)^T\Delta\theta
+
\frac{1}{2\eta}\|\Delta\theta\|_M^2
$$

约束：

$$
J_f\Delta\theta\approx 0.
$$

其投影形式为：

$$
\Delta\theta
=
-\eta
\left(
M^{-1}g_R
-
M^{-1}J_f^T
(J_fM^{-1}J_f^T+\mu I)^{-1}
J_fM^{-1}g_R
\right).
$$

其中：

```text
g_R = geometry / Sobolev / roughness gradient
J_f = teacher function Jacobian，可以是 logits、hidden features、margin 的组合
M = functional metric / diagonal metric / identity
```

这才是真正的 nullspace smoothing。

---

### 4.3 Event-driven TAN：只在需要时 smoothing，而不是固定 cycle

当前 P5 失败说明固定 alternating cycles 不稳定。

v4.9 应该改为事件驱动：

```text
if geometry exceeds budget and task plateau is acceptable:
    do Exact NFS block smoothing
    run short task refresh
else:
    continue task learning
```

触发条件可以是：

$$
\phi'_{p95} / \phi'_{teacher} > r_{\phi}
$$

或：

$$
\Delta L_{holdout}\text{ plateau for }K\text{ steps}.
$$

接受条件是：

$$
\Delta Acc > -\epsilon_{acc},
\quad
KL < \epsilon_{KL},
\quad
\Delta \phi < 0.
$$

---

### 4.4 Basis / center / width feasibility

至少要测试：

```text
fixed uniform RBF
AB-RBF: constant + linear + RBF
learnable width only
per-role width
quantile centers
AB-RBF + learnable width
```

优先级是：

```text
1. AB-RBF
2. per-role / learnable width
3. quantile centers
```

原因：AB-RBF 直接补 low-order path，成本最低，解释最清楚。

---

## 5. v4.9 实验计划

### P0：Code audit 与参数化 smoke

#### 目的

确认新参数化和 Exact NFS 实现没有破坏 strict PureKAN 定义。

#### 需要实现

新增 edge 类型：

```text
RBFOnly:
  f(x)=sum_k c_k B_k(x)

AB-RBF-linear:
  f(x)=b+w x+sum_k c_k B_k(x)

AB-RBF-silu:
  f(x)=w_base silu(x)+sum_k c_k B_k(x)

RBF-learnWidth:
  centers fixed, width per layer trainable or functional-updated

RBF-quantileCenters:
  centers initialized from activation quantiles, fixed during run
```

需要记录：

```text
learnable_nonKAN_params
learnable_edge_params
rbf_coeff_params
base_params
center_params
width_params
functional_coverage
base_coverage
center_width_coverage
rollback_error
NFS_kkt_residual
NFS_constraint_residual
CG_residual
```

#### 判定

通过标准：

```text
nonKAN params = 0 for strict PureKAN variants
functional coverage = 1.0
base/width params either included in edge functional group or explicitly marked AdamW-control ablation
rollback_error < 1e-8
NFS constraint residual finite
```

#### 可视化

```text
parameter-count stacked bar: coeff/base/center/width
coverage heatmap by role: input/block/output
NFS residual histogram
```

---

### P1：RBF 参数化上限：AdamW feasibility frontier

#### 目的

先问架构本身：

$$
\boxed{
\text{Base-RBF / learnWidth 是否让 PureKAN-AdamW 更强或更平滑？}
}
$$

#### 方法

数据集：

```text
MNIST
Fashion-MNIST
KMNIST
```

模型：

```text
PureKAN-RBFOnly-AdamW
PureKAN-ABRBF-linear-AdamW
PureKAN-ABRBF-silu-AdamW
PureKAN-RBF-learnWidth-AdamW
PureKAN-ABRBF-linear-learnWidth-AdamW
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
```

seeds：

```text
seed = 0,1,2
```

训练预算：

```text
same as v4.8 P1/P6 short budget
```

#### 记录指标

任务：

```text
test_acc
val_loss
val_loss_auc
ECE
NLL
margin_mean
margin_p10
classwise_acc
```

几何：

```text
phi_prime_p95
phi_prime_max
curvature_energy
jacobian_condition
sobolev_norm_total
coefficient_roughness
```

表示：

```text
rank_input
rank_block
rank_output
class_centroid_separation
feature_norm_mean/p95
```

basis：

```text
basis_occupancy_entropy
dead_basis_fraction
out_of_grid_fraction
per_role_basis_mean
center_coverage
width_value / width_softplus
```

base path：

```text
base_output_norm
rbf_output_norm
base_over_rbf_norm
base_margin_contribution
rbf_margin_contribution
base_coeff_norm
linear_base_slope_distribution
```

#### 判定

P1 通过标准不是 functional optimizer 成功，而是回答 architecture feasibility。

```text
AB-RBF positive if:
  acc >= RBFOnly-AdamW - 0.5%
  and either phi/J improves meaningfully
  or basis occupancy / rank improves
  or functional candidates later get better direction.
```

强 positive：

```text
AB-RBF-AdamW improves acc and reduces phi/J vs RBFOnly-AdamW.
```

如果 AB-RBF-AdamW 不如 RBFOnly-AdamW：

```text
base path 不是当前瓶颈，回到 optimizer/NFS。
```

#### 可视化

```text
accuracy-geometry Pareto: acc vs phi_prime_p95
accuracy-geometry Pareto: acc vs jacobian_condition
base/RBF output norm stacked bar by layer
basis occupancy heatmap by layer
center-width coverage histogram
rank vs acc scatter
margin p10 vs phi scatter
```

---

### P2：RBF eigenmode audit：AdamW 到底用了哪些模式

#### 目的

验证 functional update 是否压掉了 AdamW 需要的 RBF modes。

#### 方法

对 P1 中的 AdamW teacher，计算 Sobolev Gram：

$$
S=Q\Lambda Q^T.
$$

对 coefficient：

$$
\tilde c=Q^Tc.
$$

对 gradient：

$$
\tilde g=Q^Tg.
$$

记录不同 eigenvalue 区间的能量：

$$
E_c(m)=\|\tilde c_m\|^2,
$$

$$
E_g(m)=\|\tilde g_m\|^2.
$$

#### 记录指标

```text
coeff_energy_by_eigenmode
grad_energy_by_eigenmode
update_energy_by_eigenmode
energy_low/mid/high
high_mode_fraction
sobolev_penalty_contribution_by_mode
role-wise eigen energy
base-vs-rbf energy
```

#### 判定

如果 AdamW 的 useful modes 主要落在 high Sobolev eigenvalue 区域，则说明：

$$
\boxed{
\text{Sobolev U-FULL 把 AdamW 需要的表达模式压掉了。}
}
$$

如果 AB-RBF 把能量从 high modes 转移到 base / low modes，则 AB-RBF 是强候选。

#### 可视化

```text
eigenmode energy spectrum
cumulative energy vs eigenvalue
AdamW vs U-FULL coefficient spectrum
role-wise high-mode fraction bar
base path contribution vs high-mode fraction
```

---

### P3：Exact NFS projection audit

#### 目的

把 v4.8 的 proposal/backtracking NFS 升级为真正的 constrained projection。

#### 方法

Teacher：

```text
PureKAN-RBFOnly-AdamW
PureKAN-ABRBF-linear-AdamW
PureKAN-ABRBF-silu-AdamW
D6-allTaskAware
FCAdam-dataSob
```

Projection variants：

```text
Heuristic-NFS-role-block  # v4.8 current baseline
ExactNFS-logit
ExactNFS-hidden
ExactNFS-logit-hidden
ExactNFS-margin
ExactNFS-role-block
ExactNFS-role-cycle
```

Metric：

```text
M = identity
M = Sobolev diag
M = Sobolev full
M = dataSob diag
```

Projector solve：

```text
KKT direct on small sketch
CG with VJP/JVP
low-rank randomized SVD
```

#### Core formula

Use:

$$
\Delta\theta
=
-\eta
\left(
M^{-1}g_R
-
M^{-1}J_f^T
(J_fM^{-1}J_f^T+\mu I)^{-1}
J_fM^{-1}g_R
\right).
$$

where:

```text
g_R = gradient of roughness / Sobolev / phi proxy
J_f = Jacobian of logits, hidden features, or margins
```

#### 记录指标

Projection quality：

```text
KKT_residual
constraint_residual
projected_component_norm
nullspace_component_norm
J_delta_norm
M_norm_delta
CG_iterations
CG_residual
```

Function preservation：

```text
acc_drop
KL_teacher_student
logit_relative_drift
hidden_relative_drift
margin_relative_drift
argmax_flip_rate
classwise_flip_rate
```

Geometry：

```text
phi_reduction
J_reduction
sobolev_reduction
roughness_reduction
curvature_reduction
```

#### 判定

Exact NFS must beat heuristic NFS on at least one of:

```text
same geometry reduction with smaller drift
or stronger geometry reduction with same drift
or lower cycle instability in P5
```

Pass gate:

```text
acc_drop <= 0.5%
KL < 0.005
logit_drift < 0.03
argmax_flip_rate < 2%
phi_reduction > 10% or J_reduction > 20%
KKT_residual < 1e-3 or CG_residual < 1e-2
```

#### 可视化

```text
geometry reduction vs logit drift Pareto
phi reduction vs acc drop scatter
KKT residual histogram
CG residual by projector
argmax flip heatmap by class
heuristic vs exact NFS paired bar
```

---

### P4：One-cycle TAN v2 with exact NFS and AB-RBF

#### 目的

验证：

$$
\boxed{
\text{Task teacher} \rightarrow \text{Exact NFS} \rightarrow \text{Refresh}
}
$$

是否能稳定通过一 cycle，并且比 v4.8 one-cycle 更稳。

#### Methods

Task teachers：

```text
RBFOnly-AdamW
ABRBF-linear-AdamW
ABRBF-silu-AdamW
FCAdam-dataSob
D6-allTaskAware
```

NFS：

```text
Best Heuristic NFS-role-block from v4.8
Best ExactNFS-role-block
Best ExactNFS-logit-hidden
```

Refresh：

```text
none
D6-5
FCAdam5
AdamW-small-5  # diagnostic, not strict functional final
```

#### 记录指标

Cycle metrics：

```text
task_acc
smooth_acc
refresh_acc
acc_drop_from_task
phi_reduction_from_task
J_reduction_from_task
refresh_recovered_loss
refresh_recovered_acc
KL_after_smooth
KL_after_refresh
rank_after_smooth
rank_after_refresh
```

#### 判定

One-cycle pass:

```text
final_acc >= task_acc - 0.5%
phi_reduction_from_task > 10%
KL_after_refresh < 0.01
rank_after_refresh >= 0.85 * task_rank
```

All-dataset survivor required before P5.

#### 可视化

```text
one-cycle trajectory plot: task -> smooth -> refresh
acc drop vs phi reduction scatter
rank preservation after smoothing
refresh recovery bar
role contribution during refresh
```

---

### P5：Event-driven TAN multi-cycle controller

#### 目的

修复 v4.8 P5 的核心失败：固定 alternating cycles 不稳定，尤其 MNIST 不过。

#### Controller

不要固定每个 cycle 都 smoothing。改为事件触发：

```text
if geometry_bad and task_plateau and function_confidence_high:
    smooth role-block
    refresh
else:
    continue task learning
```

Trigger variables：

```text
phi_ratio_to_teacher
jac_ratio_to_teacher
holdout_loss_plateau
margin_p10_plateau
rank_drop
basis_occupancy_low
```

Pseudo-rule:

```text
smooth if:
  phi_ratio > 1.10
  and holdout_loss_improvement_last_K < threshold
  and rank_ratio > 0.75

skip smooth if:
  rank is still forming
  or margin_p10 is too low
  or recent smoothing caused acc drop
```

#### Methods

Only use P4 survivors.

Run:

```text
cycles up to 3
but smoothing can be skipped
```

#### 记录指标

```text
cycle_index
smooth_triggered
trigger_reason
skip_reason
teacher_acc
pre_smooth_acc
post_smooth_acc
post_refresh_acc
acc_drop
phi_reduction
rank_change
margin_change
KL
argmax_flip
refresh_steps_used
```

#### 判定

Pass:

```text
all datasets pass 3-cycle gate
final_acc >= initial_task_acc - 1%
final_phi_reduction > 10%
rank_final >= 0.75 * initial_task_rank
no cycle has acc collapse > 2%
```

#### 可视化

```text
cycle timeline per dataset
trigger reason stacked bar
acc/phi/rank over cycles
smoothing accepted vs rejected plot
MNIST failure drilldown if fail
```

---

### P6：Capacity / basis expansion full run

#### 目的

真正回答：

$$
\boxed{
\text{RBF-only 是否容量/参数化不足？}
}
$$

#### Config grid

```text
RBFOnly h64 b16 d4
RBFOnly h96 b24 d4
RBFOnly h96 b32 d2
ABRBF-linear h96 b24 d4
ABRBF-silu h96 b24 d4
ABRBF-linear h128 b24 d4
ABRBF-linear h96 b32 d4
RBF-learnWidth h96 b24 d4
ABRBF-linear-learnWidth h96 b24 d4
```

Methods:

```text
AdamW
AdamW-MixedGeom
Best functional task learner from earlier, if any
Best TAN one-cycle/multi-cycle, if P4/P5 passes
```

Seeds:

```text
0,1,2 initially
```

#### 记录指标

```text
param_count
step_time_ms
peak_memory
acc
val_auc
ECE
phi/J
rank
margin
basis coverage
base contribution
width distribution
```

#### 判定

If AB-RBF-AdamW improves smooth frontier:

```text
keep AB-RBF as new PureKAN default.
```

If AB-RBF functional closes gap:

```text
continue functional optimizer on AB-RBF.
```

If AB-RBF AdamW itself fails:

```text
RBF parameterization is not the main issue; return to optimizer dynamics.
```

---

### P7：3-seed candidate selection

Only enter if P4/P5 or P6 yields a candidate.

Candidate examples:

```text
ABRBF-linear-AdamW + ExactNFS one-cycle
ABRBF-linear-FCAdam + ExactNFS one-cycle
RBFOnly-AdamW + ExactNFS event-driven TAN
```

Baselines:

```text
PureKAN-RBFOnly-AdamW
PureKAN-ABRBF-AdamW
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
D6-allTaskAware
```

Gate:

```text
acc >= PureKAN-AdamW - 0.5% on all datasets
phi reduction > 10% or J reduction > 20%
ECE not worse by > 0.02
rank >= 0.75 * PureKAN-AdamW
val_auc not worse by > 5%
```

---

### P8：5-seed confirm

Run final candidate vs baselines.

Record paired deltas:

```text
paired_acc_delta_vs_PureKAN_AdamW
paired_auc_delta_vs_PureKAN_AdamW
paired_phi_delta
paired_jac_delta
paired_ece_delta
paired_rank_delta
```

Pass:

```text
accuracy CI not worse than AdamW by > 0.5%
geometry improves with CI positive
ECE not worse
no catastrophic seed
```

---

### P9：10-seed final confirm

Only if P8 passes.

Final claim allowed only if:

```text
PureKAN functional / TAN candidate matches or beats PureKAN-AdamW accuracy;
geometry improves materially;
model remains strict KAN-edge parameterization;
no hidden MLP stem/head is introduced.
```

---

## 6. Required artifacts

Under `results/v4_9/`:

```text
p0_code_audit.csv
p1_architecture_frontier.csv
p1_training_trace.csv
p2_eigenmode_audit.csv
p3_exact_nfs_projection.csv
p4_tan_one_cycle_v2.csv
p5_event_tan_cycles.csv
p6_capacity_basis_expansion_full.csv
p7_candidate_selection.csv
p8_confirm5.csv
p9_confirm10.csv
failure_table.csv
aggregate_decision.json
figures/
```

Figures:

```text
figures/acc_phi_pareto.svg
figures/acc_jac_pareto.svg
figures/base_rbf_contribution_by_layer.svg
figures/basis_occupancy_heatmap.svg
figures/eigenmode_energy_spectrum.svg
figures/nfs_geometry_vs_drift.svg
figures/nfs_constraint_residual_hist.svg
figures/tan_one_cycle_trajectory.svg
figures/tan_cycle_timeline.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 7. Failure taxonomy for v4.9

```text
F1_architecture_frontier_absent:
  AB-RBF / learnWidth / capacity expansion still cannot expose accurate smoother solutions.

F2_nfs_not_exact:
  Exact NFS cannot reduce geometry under low drift.

F3_nfs_worse_than_heuristic:
  Exact projection is worse than v4.8 heuristic NFS.

F4_smoothing_destroys_function:
  KL/logit drift/acc drop too large.

F5_refresh_cannot_recover:
  smoothing works, but refresh cannot restore task loss.

F6_multicycle_instability:
  one-cycle works, repeated event-driven cycles fail.

F7_basis_coverage_bad:
  dead basis / out-of-grid / low occupancy dominate.

F8_base_path_not_used:
  AB-RBF base contribution remains near zero.

F9_functional_candidate_acc_gap:
  final candidate geometry improves but accuracy gap too large.
```

---

## 8. Expected interpretations

### Case A：AB-RBF AdamW exposes better frontier

Then current PureKAN failure is partly parameterization-driven.

Conclusion:

$$
\boxed{
\text{RBF-only PureKAN lacked low-order edge channels.}
}
$$

Next default becomes AB-RBF PureKAN.

---

### Case B：Exact NFS beats heuristic NFS

Then v4.8 was limited by approximate proposal/backtracking.

Conclusion:

$$
\boxed{
\text{function-preserving geometry projection is real and should become a primitive.}
}
$$

---

### Case C：One-cycle works but multi-cycle fails again

Then the controller is still the bottleneck.

Conclusion:

$$
\boxed{
\text{Learn-Smooth-Refresh is valid locally, but online scheduling is unresolved.}
}
$$

---

### Case D：AB-RBF + Exact NFS still fails

Then strict PureKAN functional training remains unresolved.

Conclusion:

$$
\boxed{
\text{Current KAN-edge parameterization plus coefficient-only functional control is insufficient.}
}
$$

At that point, the project should either:

```text
1. return to Hybrid-DGKAN as the validated contribution;
2. move to spline / efficient-KAN / rational primitive with primitive-specific metric;
3. or allow a small set of non-KAN normalization/base parameters as necessary infrastructure.
```

---

## 9. Current recommendation

Do not continue another general functional optimizer sweep.

The next run should be exactly:

```text
v4.9-P0 code audit
v4.9-P1 AB-RBF / learnWidth AdamW frontier
v4.9-P2 eigenmode audit
v4.9-P3 Exact NFS projection
```

Only if P3 passes should we continue to TAN cycles.

The strongest current hypothesis is:

$$
\boxed{
\text{PureKAN needs a better edge parameterization first: base + RBF, then function-preserving smoothing.}
}
$$

