# DG-KAN Optimizer v3 完整计划：Phase-Aligned GA-FU

> 版本：v3.1 draft, added Rational / GR-KAN scaling track  
> 日期：2026-05-01  
> 目标：把当前 GA-FU 从“tuned weak-pass optimizer candidate”升级为一个实现清晰、诊断充分、可以继续冲 medium/strong gate 的 **phase-aligned functional optimizer**。  
> 公式格式：Typora 友好，全文只使用 `$...$` 和 `$$...$$`。  
> 本计划默认不加入 data metric、SNR、FSAM、prox/spectral、target solve、hard branch controller、naive Adam momentum。它们只保留为后续 profile / ablation，不进入 v3 core。

---

## 0. 总结一句话

我建议把 v3 定义为：

$$
\boxed{
\text{GA-FU-v3}
=
\text{phase-aligned functional metric}
+
\text{true shared Sobolev Gram}
+
\text{smooth branch transition}
+
\text{coordinated rest/coeff schedule}
+
\text{EMA geometry feedback}
+
\text{transparent trust accounting}
}
$$

它不是继续堆 optimizer 全家桶，而是把当前 GA-FU 中已经证明有价值、但实现仍偏粗的部分做扎实：让 branch phase、metric phase、coefficient LR phase、rest/head LR phase 和 geometry feedback 真正对齐。

当前 tuned GA-FU 已经在 Fashion-MNIST 与 KMNIST 上取得 5-seed weak pass，但它还没达到 medium/strong。最主要瓶颈不是 accuracy，也不是几何，而是 **validation-loss AUC 与 early trajectory**。这意味着 v3 的核心目标不是继续盲目增强 branch，而是修复学习动力学：让 KAN branch 在早期足够参与任务，同时让 loss trajectory 更平滑、更快、更不被突兀 switch 或 metric mismatch 拖慢。

---

# 第一部分：我的思考与判断

## 1. 当前 optimizer 实际是什么

现在的训练方式本质上是 hybrid optimizer：

$$
\boxed{
\text{KAN coefficients: manual functional update}
+
\text{non-KAN parameters: AdamW}
}
$$

非 KAN 参数包括 stem、head、LayerNorm、residual alpha、KAN bias 等；KAN edge coefficient `kan.coeff` 由手写 functional update 更新。这个方向是对的，因为之前实验已经说明 functional update 不适合粗暴替代所有参数，真正适合它的是 KAN edge function coefficient。

模型 block 是：

$$
h_{k+1}
=
h_k
+
\alpha_k b_t \operatorname{KAN}_k(\operatorname{LN}(h_k)).
$$

其中 $b_t$ 是 branch scale，由 schedule 控制。KAN edge function 写成：

$$
\phi_{ji}(t)=\sum_{m=1}^{M}a_{jim}B_m(t).
$$

普通 coefficient update 是：

$$
a\leftarrow a-\eta\nabla_aL.
$$

functional update 是：

$$
a\leftarrow a-\eta P_t\nabla_aL.
$$

当前 GA-FU 的语义是：早期用 diagonal functional update 激活 branch，后期用 Sobolev metric 收住几何，并通过 online geometry trigger 控制切换。

问题在于，当前代码中的 $P_t$ 还没有真正实现完整 Sobolev Gram inverse。现在更像一个 basis-index diagonal grid penalty：

$$
D_m
=
1
+
\alpha_s(\pi k_m)^2
+
\beta_s(\pi k_m)^4.
$$

然后更新近似为：

$$
\Delta a_m
=
-\eta\frac{\nabla_{a_m}L}{D_m}.
$$

这对稳定性有帮助，但它不是：

$$
(M+\rho I)^{-1}\nabla_aL.
$$

因此它无法处理 basis 之间的相关性，也无法真正兑现“diag-to-Sobolev”的实现语义。

---

## 2. 当前 optimizer 已经做对的事情

当前 GA-FU 的最大价值是，它已经抓住了分类任务的核心矛盾：

$$
\boxed{
\text{KAN branch 太弱，则表达力没有释放；branch 太强，则几何失控。}
}
$$

这个判断已经被多轮实验验证。早期 functional update 失败时，hybrid branch ratio 远低于 AdamW，关闭 KAN branch 后几乎不掉精度；提高 coefficient LR 后，accuracy 可以追上或超过 AdamW，但几何风险上升。

现在的 tuned GA-FU 已经能在 Fashion-MNIST 和 KMNIST 上做到：

$$
\text{accuracy 不输 AdamW，甚至略高于 AdamW，}
$$

同时：

$$
\phi'_{p95}\downarrow,
\quad
\kappa(J)\downarrow,
\quad
\operatorname{ECE}\downarrow,
\quad
\text{branch/AdamW 进入合理区间。}
$$

这说明方向不是错的。现在的问题不是“要不要 functional update”，而是“functional update 的 phase、metric、schedule、trigger 是否够精细”。

---

## 3. 当前 optimizer 的主要问题

### 3.1 metric phase 与 branch phase 没有严格对齐

当前代码中 metric 的 early/late 判断主要来自 progress / warmup fraction，而 branch 的 active/geometry phase 又来自 geometry switch 或 max active fraction。这会导致：

$$
\text{branch 已经进入 late phase，metric 仍可能处在 early mode；}
$$

或者：

$$
\text{metric 已经进入 late mode，branch 仍在 active mode。}
$$

这会直接影响 validation-loss AUC。一个好的 schedule 应该是：

$$
\text{ACTIVE phase}
\Rightarrow
\text{diag metric + high branch + high coeff LR},
$$

$$
\text{TRANSITION phase}
\Rightarrow
\text{smooth metric mix + smooth branch decay},
$$

$$
\text{GEOMETRY phase}
\Rightarrow
\text{full Sobolev metric + stable branch scale}.
$$

当前实现更像是几个 knob 并排存在，还没有形成一个统一 state machine。

### 3.2 当前 Sobolev 只是 diagonal approximation

真正的 Sobolev metric 应该是：

$$
M
=
\int B(t)B(t)^\top dt
+
\alpha_s\int B'(t)B'(t)^\top dt
+
\beta_s\int B''(t)B''(t)^\top dt.
$$

更新为：

$$
\Delta a
=
-\eta(M+\rho I)^{-1}\nabla_aL.
$$

basis_count 当前是 16 或 24，所以完整 $M\in\mathbb R^{M\times M}$ 很小，做 Cholesky solve 代价不大。当前使用 basis-index diagonal penalty，工程上简单，但会损失两个能力：

1. basis 相关性不能被处理；
2. update direction 可能不是自然的 function-space steepest descent。

v3 应该把这个补上。

### 3.3 geometry trigger 还不是闭环控制

当前 trigger 可以切换 active phase，但还不是一个完整闭环。它缺少三件事：

第一，触发后没有平滑 transition，而是接近 abrupt switch。

第二，触发后如果几何继续恶化，shrink 逻辑没有成为真正的连续 feedback controller。

第三，触发信号缺少 EMA / hysteresis，容易受单次 audit 噪声影响。

更合理的 v3 是：

$$
\bar \phi_t=(1-\beta)\phi_t+\beta\bar\phi_{t-1},
$$

$$
\bar \kappa_t=(1-\beta)\kappa_t+\beta\bar\kappa_{t-1},
$$

$$
\bar r_t=(1-\beta)r_t+\beta\bar r_{t-1}.
$$

再用这些 EMA 信号控制 phase transition 和 recovery。

### 3.4 rest/head AdamW 与 KAN coeff schedule 不够协同

Fashion 的结果很关键：单独调 coeff schedule 不够，真正让 Fashion weak pass 站住的是 rest/head AdamW 与 KAN coefficient 的 schedule 协同，尤其 rest+coeff cosine 的组合。

这说明：

$$
\boxed{
\text{AUC gate 不是只靠 KAN coeff LR 解决的。}
}
$$

KAN branch 学得再好，如果 stem/head/rest 参数在 early trajectory 上没有配合，val loss AUC 仍可能输给 AdamW。v3 必须把 rest/head schedule 当成 optimizer 的一部分，而不是 runner 里的后补实验项。

### 3.5 trust clipping 现在是隐藏变量

当前配置里有 `trust_radius=0.15` 和 `trust_mode=safety_only`。这意味着 functional update 可能被 metric norm clipping 影响。

如果 clip rate 很低，它是安全网；如果 clip rate 较高，它就是实际 optimizer component。v3 不能让它隐藏。每个实验都必须记录：

$$
\operatorname{clip\_rate}
=
\frac{\#\text{clipped updates}}{\#\text{coefficient updates}}.
$$

并且至少比较：

```text
trust_mode = off
trust_mode = safety_only, radius = 0.15
trust_mode = safety_only, radius = 0.30
```

### 3.6 branch 指标需要 final / per-layer / no-KAN 联合诊断

当前 `branch_output_norm_ratio` 更多是训练过程平均值。对 schedule 方法来说，这个均值会混合 active phase 和 late phase。我们真正关心的是：

$$
\text{final branch contribution to task}.
$$

因此必须记录：

$$
r_{branch}^{final},
\quad
r_{branch,k}^{final},
\quad
\Delta_{noKAN}^{final},
\quad
\Delta_{logit}^{KAN},
\quad
\Delta_{margin}^{KAN}.
$$

如果 branch ratio 高，但 no-KAN drop 不高，说明 branch 变大了但没有做有用任务。

---

## 4. v3 的设计目标

v3 不是为了再创造一个复杂 optimizer，而是为了同时满足你的三个要求：

$$
\boxed{
\text{高模型表达力}
+
\text{快速收敛}
+
\text{几何好}
}
$$

这三个目标对应到实验指标上是：

### 4.1 高模型表达力

表达力不能只看 test accuracy。它要同时看：

$$
\operatorname{Acc}_{test},
\quad
\Delta_{noKAN},
\quad
r_{branch},
\quad
\Delta_{margin}^{KAN},
\quad
\text{active basis fraction}.
$$

v3 需要证明 KAN branch 不是装饰，而是在做任务。

### 4.2 快速收敛

快速收敛不能只看 final loss。它要看：

$$
\operatorname{AUC}_{val\ loss},
\quad
\operatorname{steps\_to\_target},
\quad
\operatorname{time\_to\_target},
\quad
L_{10\%},L_{25\%},L_{50\%},L_{75\%}.
$$

如果 v3 只是 final accuracy 好，但 AUC 不好，就不能 claim fast convergence。

### 4.3 几何好

几何至少包括：

$$
\phi'_{p95},
\quad
\phi'_{max},
\quad
\kappa(J),
\quad
\text{credit amplification},
\quad
\text{curvature energy},
\quad
\text{Sobolev norm}.
$$

这里最关键的不是让 branch 永远很弱，而是在 branch 有任务贡献的情况下仍然控制这些几何量。

---

# 第二部分：GA-FU-v3 方法设计

## 5. v3 core 定义

GA-FU-v3 的训练循环由一个明确的 state machine 控制：

```text
ACTIVE -> TRANSITION -> GEOMETRY -> RECOVERY(optional)
```

其中每个 phase 同时决定：

```text
branch_scale
coeff_lr_multiplier
functional metric mode
rest/head lr schedule
trust behavior
```

形式化写成：

$$
a_{t+1}=a_t-\eta_a(t)P_{s_t}\nabla_aL_t,
$$

$$
\theta_{t+1}=\operatorname{AdamW}(\theta_t,\nabla_\theta L_t;\eta_\theta(t)),
$$

$$
h_{k+1}=h_k+\alpha_k b_t\operatorname{KAN}_k(\operatorname{LN}(h_k)).
$$

这里 $s_t$ 是当前 phase。

---

## 6. Phase 1：ACTIVE

ACTIVE phase 的目标是释放表达力，而不是追求极限几何。它要解决 branch under-active。

配置原则：

```text
metric = diag(shared Sobolev Gram) 或 low-alpha diag Gram
branch_scale = branch_boost
coeff_lr_multiplier = coeff_lr_boost
rest/head lr = base_lr
```

更新为：

$$
P_{ACTIVE}
=
\operatorname{diag}(M_{fast}+\rho I)^{-1}.
$$

其中：

$$
M_{fast}
=
\int BB^\top dt
+
\alpha_{fast}\int B'B'^\top dt
+
\beta_{fast}\int B''B''^\top dt.
$$

建议先用：

```text
alpha_fast = 0.00 or 0.01
beta_fast = 0.00
rho = 1e-3
```

ACTIVE phase 不是越长越好。它的退出条件是：

$$
\bar \phi_t>\tau_\phi
\quad\text{or}\quad
\bar\kappa_t>\tau_J
\quad\text{or}\quad
\bar r_t>\tau_b
\quad\text{or}\quad
t/T>T_{max\ active}.
$$

---

## 7. Phase 2：TRANSITION

当前实现最大的问题之一是 active 到 geometry 的切换太硬。v3 使用 smooth transition。

设 transition length 为 $T_{tr}$，transition 内局部进度为：

$$
\tau=\frac{t-t_{switch}}{T_{tr}}.
$$

定义平滑系数：

$$
\lambda(\tau)=\frac{1}{2}-\frac{1}{2}\cos(\pi\tau).
$$

branch scale 从 $b_{boost}$ 平滑到 $b_{final}$：

$$
b_t=(1-\lambda)b_{old}+\lambda b_{final}.
$$

coeff LR multiplier 从 $c_{boost}$ 平滑到 $c_{final}$：

$$
c_t=(1-\lambda)c_{old}+\lambda c_{final}.
$$

metric direction 使用 direction mix，而不是直接切换：

$$
d_t=(1-\lambda)d_{diag}+\lambda d_{full},
$$

其中：

$$
d_{diag}=\operatorname{diag}(M+\rho I)^{-1}\nabla_aL,
$$

$$
d_{full}=(M+\rho I)^{-1}\nabla_aL.
$$

然后：

$$
a_{t+1}=a_t-\eta_a(t)d_t.
$$

这样做的好处是，loss trajectory 不会因为 optimizer geometry 突然变化而出现断点。

---

## 8. Phase 3：GEOMETRY

GEOMETRY phase 的目标是保持最终表达力，同时收住函数几何。

配置原则：

```text
metric = full shared Sobolev Gram
branch_scale = branch_final_scale
coeff_lr_multiplier = coeff_lr_final_mult
rest/head lr = cosine-decayed value
```

更新为：

$$
P_{GEOMETRY}=(M_{geo}+\rho I)^{-1}.
$$

其中：

$$
M_{geo}
=
\int BB^\top dt
+
\alpha_{geo}\int B'B'^\top dt
+
\beta_{geo}\int B''B''^\top dt.
$$

建议先用：

```text
alpha_geo = 0.15
beta_geo = 0.02
rho = 1e-3
```

但要保留 sweep：

```text
alpha_geo in {0.05, 0.10, 0.15}
beta_geo in {0.00, 0.01, 0.02}
```

这里不建议大范围扫，因为 Stage B 早已说明过强 Sobolev smoothing 会伤高频拟合。

---

## 9. Phase 4：RECOVERY

RECOVERY 不是默认新 optimizer 组件，而是 geometry feedback 的安全状态。如果 GEOMETRY phase 后仍出现：

$$
\bar\phi_t>\tau_\phi^{recover}
\quad\text{or}\quad
\bar\kappa_t>\tau_J^{recover},
$$

则执行：

$$
b_t\leftarrow\max(b_{min},\gamma b_t),
$$

$$
\eta_a(t)\leftarrow\gamma_\eta\eta_a(t).
$$

建议：

```text
b_min = 0.65 or 0.70
gamma = 0.90
gamma_eta = 0.80
recovery_cooldown = 1 epoch
```

RECOVERY 必须记录 shrink events。如果 shrink events 很多，说明 upstream schedule 太激进，不能把 recovery 当成功。

---

## 10. True shared Sobolev Gram 实现

### 10.1 共享 Gram 的基本思想

每个 KAN edge 共享同一组 basis：

$$
\phi_{ji}(t)=\sum_m a_{jim}B_m(t).
$$

所以 Sobolev Gram 只需要按 basis_count 构建一次：

$$
M\in\mathbb R^{M\times M}.
$$

然后对每个 edge 的 coefficient gradient 做同一个 solve：

$$
D_{ji,:}=(M+\rho I)^{-1}G_{ji,:}.
$$

这里 $G_{ji,:}$ 是 $
abla_{a_{ji,:}}L$。

因为 $M=16$ 或 $24$，这个 solve 很便宜。

### 10.2 数值构造

在 grid 上做 quadrature：

$$
t_q\in[t_{min},t_{max}],\quad q=1,\dots,Q.
$$

basis matrix：

$$
B_{qm}=B_m(t_q).
$$

一阶导：

$$
B'_{qm}=\frac{dB_m(t_q)}{dt}.
$$

二阶导：

$$
B''_{qm}=\frac{d^2B_m(t_q)}{dt^2}.
$$

Gram：

$$
M
=\Delta t\cdot B^\top B
+\alpha_s\Delta t\cdot B'^\top B'
+\beta_s\Delta t\cdot B''^\top B''.
$$

Damped matrix：

$$
A=M+\rho I.
$$

使用 Cholesky：

$$
A=LL^\top.
$$

方向计算：

$$
d=\operatorname{solve}(A,g).
$$

### 10.3 metric modes

v3 至少实现四个 metric mode：

```text
identity
basis_diag_gram
full_sobolev_gram
diag_to_full_sobolev
```

它们含义如下：

| mode | 作用 | 使用阶段 |
|---|---|---|
| `identity` | 不做函数空间预条件 | sanity / ablation |
| `basis_diag_gram` | 用真实 Gram 的 diagonal，而不是 index penalty | ACTIVE |
| `full_sobolev_gram` | 完整 Sobolev inverse | GEOMETRY |
| `diag_to_full_sobolev` | direction mix | TRANSITION |

当前代码里的 index diagonal penalty 可以保留为 `grid_index_diag_legacy`，只用于对照。

---

## 11. Coordinated schedule

v3 不再把 rest/head schedule 看作独立实验项，而是纳入 optimizer profile。

### 11.1 Fashion profile

当前 Fashion 的核心发现是 rest/head schedule 很重要。因此 v3 Fashion default：

```text
hidden_dim = 96
basis_count = 24
depth = 4
alpha_init = 1.5
epochs = 30
coeff_lr = 0.05
rest_lr = 0.001
branch_boost = 1.2
coeff_lr_boost = 1.0
branch_final_scale = 0.8
geometry_min_epochs = 0
rest_lr_schedule = cosine
rest_lr_final_mult = 0.3
coeff_lr_schedule = cosine
coeff_lr_decay_final_mult = 0.5
metric_active = basis_diag_gram
metric_transition = diag_to_full_sobolev
metric_geometry = full_sobolev_gram
```

v3 需要验证：当前 bothcos 的 AUC gain 是否来自 LR schedule 本身，还是来自 schedule 与 legacy diagonal metric 的偶然配合。加入 true Gram 后，可能出现两种结果：

1. AUC 继续改善，说明 true Gram 修复 direction；
2. AUC 下降，说明 full Gram 太早或太强，需要更长 transition 或更小 $\alpha_{geo}$。

### 11.2 KMNIST profile

当前 KMNIST 的 c0.14 profile 已经弱通过，但 AUC 只有 $7\%$，未过 medium 的 $8\%$。

v3 KMNIST default：

```text
hidden_dim = 96
basis_count = 24
depth = 4
alpha_init = 1.5
epochs = 20
coeff_lr = 0.14
rest_lr = 0.003
branch_boost = 1.2
coeff_lr_boost = 1.0
branch_final_scale = 0.8
geometry_min_epochs = 0
rest_lr_schedule = none or cosine
rest_lr_final_mult = 0.7
coeff_lr_schedule = none or cosine
coeff_lr_decay_final_mult = 0.7
metric_active = basis_diag_gram
metric_transition = diag_to_full_sobolev
metric_geometry = full_sobolev_gram
```

KMNIST 不建议再提高 branch boost。它更需要：

$$
\text{better direction} + \text{milder rest/coeff schedule} + \text{stable final geometry}.
$$

---

## 12. Trust accounting

v3 core 默认把 trust 当 safety，而不是主优化机制。

每个 run 必须记录：

```text
trust/mode
trust/radius
trust/clip_count
trust/clip_rate
trust/update_metric_norm_mean
trust/update_metric_norm_p95
trust/update_over_coeff_norm_mean
```

判定规则：

$$
\operatorname{clip\_rate}<0.02
$$

可以视为 safety-only。

如果：

$$
\operatorname{clip\_rate}>0.05,
$$

则该 run 中 trust 已经实质参与 optimizer，需要单独报告，不能把结果归因给 metric/schedule。

---

## 13. v3 不做什么

v3 core 明确不做：

```text
data metric
SNR
FSAM
prox / spectral smoothing
target solve
hard branch target controller
naive Adam momentum
```

原因很简单：这些组件都有局部信号，但会污染 clean optimizer 结论。v3 的目标是先把 GA-FU 的 core 做干净。如果 core 过不了，再考虑 profile；如果 core 过了，再把这些组件作为 stress/noise/hard-task ablation。

唯一可以预注册但不进入 core 的扩展是 **function-normalized momentum**。它必须满足：

$$
u_t=P_t\nabla_aL,
$$

$$
\tilde u_t=\frac{u_t}{\|u_t\|_M+\epsilon},
$$

$$
m_t=\beta m_{t-1}+(1-\beta)\tilde u_t.
$$

也就是说，momentum 只能平滑 direction，不能放大 magnitude。

---

# 第三部分：实现计划

## 14. 代码改动总览

建议新增或修改：

```text
experiments/dgkan_core.py
  add true Sobolev Gram builder
  add metric modes
  add GA-FU-v3 phase state machine
  add smooth transition
  add EMA geometry feedback
  add per-epoch/per-phase detailed logging

experiments/run_gafu_v3.py
  implement packages V3_E0 ... V3_E9
  aggregate runs
  generate summary/failure tables

experiments/plot_gafu_v3.py
  optional: generate local png/pdf dashboards from csv
```

如果不想新增 runner，也可以扩展 `run_gafu_consolidation.py`，但我建议新增 `run_gafu_v3.py`。原因是 v3 的实验组织会更复杂，和当前 GA-FU consolidation 混在一起容易污染旧结果。

---

## 15. Sobolev Gram builder

### 15.1 函数接口

建议实现：

```python
def build_rbf_sobolev_gram(
    centers: torch.Tensor,
    width: float,
    alpha: float,
    beta: float,
    rho: float,
    grid_min: float = -2.5,
    grid_max: float = 2.5,
    grid_size: int = 512,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    ...
```

返回：

```text
M
A = M + rho I
chol
A_inv_diag
condition
eig_min
eig_max
```

### 15.2 二阶导公式

RBF basis：

$$
B_m(t)=\exp\left(-\frac{1}{2}\left(\frac{t-c_m}{\sigma}\right)^2\right).
$$

一阶导：

$$
B'_m(t)=-\frac{t-c_m}{\sigma^2}B_m(t).
$$

二阶导：

$$
B''_m(t)=\left(\frac{(t-c_m)^2}{\sigma^4}-\frac{1}{\sigma^2}\right)B_m(t).
$$

这可以直接向量化实现。

### 15.3 batched solve

gradient shape 是：

```text
[out_dim, in_dim, basis_count]
```

可以 reshape 为：

```text
[out_dim * in_dim, basis_count]
```

然后做：

```python
direction = torch.cholesky_solve(grad.T, chol).T
```

或用：

```python
direction = torch.linalg.solve(A, grad_flat.T).T
```

最后 reshape 回原 shape。

---

## 16. Phase state machine

建议新增状态：

```python
class GAFUPhase(Enum):
    ACTIVE = 0
    TRANSITION = 1
    GEOMETRY = 2
    RECOVERY = 3
```

runtime state 记录：

```text
phase
phase_enter_step
switch_step
switch_reason
transition_progress
ema_phi
ema_jac
ema_branch
ema_val_loss
recovery_events
current_branch_scale
current_coeff_lr_mult
current_metric_mix
```

### 16.1 phase update 逻辑

每个 step 先更新 schedule：

```text
update_ema_if_audit_available()
maybe_switch_phase()
compute_branch_scale()
compute_coeff_lr_multiplier()
compute_metric_mode()
```

再 forward/backward/update。

### 16.2 audit interval

full Jacobian audit 不能每 step 做。建议：

```text
cheap branch proxy: every step
phi prime estimate: every N steps or every epoch
Jacobian condition: every epoch, or every 0.1 epoch for smoke/small runs
```

对于 v3 初期实验，建议每 epoch audit 即可。对于 schedule correctness smoke，可以用小 batch 每 50 steps audit。

---

## 17. Smooth transition 实现

需要新增：

```text
transition_frac
transition_min_steps
transition_max_steps
transition_curve = cosine
```

建议：

```text
transition_frac = 0.10
transition_min_steps = 50
transition_max_steps = 300
```

transition length：

$$
T_{tr}=\operatorname{clip}(0.1T,50,300).
$$

如果总训练很短，例如 P0 smoke，允许：

$$
T_{tr}=\max(5,0.1T).
$$

---

## 18. Logging 改动

每个 run 的 `summary.json` 和 CSV 必须新增：

```text
v3/phase_final
v3/phase_switch_step
v3/phase_switch_epoch
v3/phase_switch_reason
v3/transition_length_steps
v3/metric_active
v3/metric_geometry
v3/metric_condition_active
v3/metric_condition_geometry
v3/metric_mix_auc
v3/ema_phi_final
v3/ema_jac_final
v3/ema_branch_final
v3/recovery_events
v3/trust_clip_rate
v3/precond_solve_time_ms
v3/metric_build_time_ms
```

并记录 curve：

```text
curve/epoch
curve/train_loss
curve/val_loss
curve/val_acc
curve/branch_ratio
curve/branch_scale
curve/coeff_lr
curve/rest_lr
curve/phi_prime_p95
curve/jac_condition
curve/phase_id
curve/no_kan_drop
curve/ece
```

如果 CSV 不方便存长 curve，建议输出单独文件：

```text
runs/{run_id}/curves.csv
```

---

# 第四部分：详细实验计划

## 19. 实验总览

v3 实验分成十个包：

```text
V3-E0: implementation correctness smoke
V3-E1: true Gram metric sanity
V3-E2: phase alignment ablation
V3-E3: full Gram vs diagonal metric
V3-E4: coordinated schedule ablation
V3-E5: EMA trigger and recovery ablation
V3-E6: trust transparency
V3-E7: 5-seed and 10-seed clean confirm
V3-E8: target-matched convergence
V3-E9: branch / geometry mechanism audit
V3-E10: generalization sanity
```

这些不是平行乱扫，而是逐步收敛。前一包没过时，后一包不扩大。

---

## 20. V3-E0：implementation correctness smoke

### 20.1 目标

E0 只确认代码没有数值 bug，不做方法结论。重点验证 true Sobolev Gram、phase machine、transition、logging 都能正常跑。

### 20.2 设置

```text
datasets = Fashion-MNIST, KMNIST
train / val / test = 512 / 128 / 128
epochs = 1
seeds = 0
model = h32 / depth2 / basis8
methods = AdamW, GA-FU-v2-current, GA-FU-v3-smoke
```

### 20.3 必须记录

```text
run_failed
nan_or_inf_count
metric_condition_active
metric_condition_geometry
phase_switch_step
phase_switch_reason
transition_length_steps
final_branch_scale
final_metric_mode
trust_clip_rate
precond_solve_time_ms
```

### 20.4 通过条件

E0 通过要求：

$$
\text{run failures}=0.
$$

$$
\text{all key metrics finite}.
$$

$$
1 < \operatorname{cond}(M+\rho I) < 10^4.
$$

$$
\text{GA-FU-v3 smoke acc gap vs v2}<5\%.
$$

如果这里失败，优先查实现，不做调参。

---

## 21. V3-E1：true Gram metric sanity

### 21.1 目标

E1 验证 true shared Gram 是否在简单可控场景中提供合理方向。这个包避免直接在 Fashion/KMNIST 上解释复杂现象。

### 21.2 实验一：1D edge regression

任务：

$$
y=\sin x+0.3\sin(5x).
$$

以及：

$$
y=\operatorname{smoothstep}(x).
$$

方法：

```text
coeff AdamW
legacy grid-index diagonal
basis_diag_gram
full_sobolev_gram
diag_to_full_sobolev
```

记录：

```text
train_mse
val_mse
loss_auc
phi_prime_p95
curvature_energy
predicted_descent
actual_descent
pred_actual_ratio
metric_condition
```

判定：

$$
\text{diag_to_full final loss}\leq1.1\times\text{full Sobolev final loss},
$$

且：

$$
\text{diag_to_full AUC}\leq1.1\times\text{basis diag AUC}.
$$

这复用 Stage B 的逻辑：diag 负责 early speed，full Sobolev 负责 final quality。

### 21.3 实验二：coefficient update direction audit

在 Fashion/KMNIST 小 batch 上，冻结模型，计算同一个 gradient 下不同 metric 的 update direction。

记录：

```text
cos(raw_grad, legacy_diag_direction)
cos(raw_grad, basis_diag_direction)
cos(raw_grad, full_gram_direction)
cos(basis_diag_direction, full_gram_direction)
update_norm_raw
update_norm_metric
update_amplification
```

如果 full Gram direction 与 diag direction cosine 过低，例如：

$$
\cos(d_{diag},d_{full})<0.5,
$$

需要检查 Gram 构造和 damping。

---

## 22. V3-E2：phase alignment ablation

### 22.1 目标

E2 验证 v3 最核心的假设：当前 AUC 卡点有一部分来自 phase mismatch。只改 phase alignment，不同时引入 full Gram。

### 22.2 设置

```text
datasets = Fashion-MNIST, KMNIST
model = h96 / depth4 / basis24 / alpha1.5
Fashion epochs = 30
KMNIST epochs = 20
seeds = 0,1,2
```

### 22.3 方法

| method | 说明 |
|---|---|
| `AdamW` | 全参数 AdamW baseline |
| `GA-FU-v2-tuned` | 当前 best tuned profile |
| `V3-phase-aligned-hard` | metric phase 跟随 branch switch，但无 transition |
| `V3-phase-aligned-smooth` | metric phase 跟随 branch switch，并加 smooth transition |

### 22.4 需要固定的东西

为了只测 phase alignment，先仍使用 legacy diagonal grid metric。也就是说：

```text
metric = legacy grid diagonal
not true full Gram yet
```

### 22.5 记录指标

除标准指标外，重点记录：

```text
phase_mismatch_count
phase_mismatch_fraction
transition_loss_jump
val_loss_delta_before_after_switch
branch_scale_jump
coeff_lr_jump
```

定义 transition loss jump：

$$
\Delta L_{switch}=L_{val}(e_{switch}+1)-L_{val}(e_{switch}).
$$

### 22.6 判定

如果 smooth phase alignment 能让：

$$
\operatorname{AUCImprove}_{v3}
>
\operatorname{AUCImprove}_{v2}+2\%,
$$

并且：

$$
\operatorname{AccGap}_{v3}\leq\operatorname{AccGap}_{v2}+0.3\%,
$$

$$
\phi\text{ red},J\text{ red 不下降超过 }10\%,
$$

则 phase alignment 是 confirmed improvement。

---

## 23. V3-E3：full Gram vs diagonal metric

### 23.1 目标

E3 验证 true shared Sobolev Gram 是否优于当前 diagonal approximation。

### 23.2 设置

沿用 E2 的最佳 phase setting。

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2
model = h96 / depth4 / basis24
```

### 23.3 方法矩阵

| method | ACTIVE metric | TRANSITION metric | GEOMETRY metric |
|---|---|---|---|
| `legacy-v2` | legacy index diag | none | legacy index diag |
| `diag-gram` | basis diag Gram | basis diag Gram | basis diag Gram |
| `full-from-start` | full Gram | full Gram | full Gram |
| `diag-to-full-hard` | basis diag Gram | hard switch | full Gram |
| `diag-to-full-smooth` | basis diag Gram | direction mix | full Gram |

### 23.4 参数

```text
alpha_fast = 0.00
beta_fast = 0.00
alpha_geo = 0.15
beta_geo = 0.02
rho = 1e-3
transition_frac = 0.10
```

如果 full Gram 过强，再加一组：

```text
alpha_geo = 0.05
beta_geo = 0.00
```

### 23.5 记录指标

```text
metric/eig_min
metric/eig_max
metric/condition
metric/update_amplification
metric/diag_full_direction_cos
metric/precond_solve_time_ms
metric/cholesky_success
```

### 23.6 判定

`diag-to-full-smooth` 是候选默认，需满足：

$$
\text{AUC improvement} > \text{v2 tuned AUC improvement},
$$

或至少：

$$
\text{AUC improvement} \geq \text{v2 tuned AUC improvement}-1\%,
$$

同时：

$$
\phi\text{ red and }J\text{ red}\geq\text{v2 tuned}-10\%.
$$

如果 full Gram 提升几何但显著伤 AUC，则保留 `basis_diag_gram` 作为 v3 default，并把 full Gram 降级为 late-only 或 ablation。

---

## 24. V3-E4：coordinated schedule ablation

### 24.1 目标

E4 只研究 rest/head schedule 与 coefficient schedule 的协同，不再动 metric。Fashion 已经证明 rest/head schedule 对 AUC 关键；E4 要判断 v3 中应固定哪种 schedule。

### 24.2 设置

使用 E3 的最佳 metric/phase setting。

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2
Fashion epochs = 30
KMNIST epochs = 20
```

### 24.3 Fashion schedule matrix

| label | rest schedule | rest final | coeff schedule | coeff final |
|---|---|---:|---|---:|
| `none` | none | 1.0 | none | 1.0 |
| `restcos` | cosine | 0.3 | none | 1.0 |
| `coeffcos` | none | 1.0 | cosine | 0.5 |
| `bothcos` | cosine | 0.3 | cosine | 0.5 |
| `restcos-softcoeff` | cosine | 0.3 | cosine | 0.7 |

### 24.4 KMNIST schedule matrix

| label | rest schedule | rest final | coeff schedule | coeff final |
|---|---|---:|---|---:|
| `none` | none | 1.0 | none | 1.0 |
| `restcos07` | cosine | 0.7 | none | 1.0 |
| `coeffcos07` | none | 1.0 | cosine | 0.7 |
| `bothcos07` | cosine | 0.7 | cosine | 0.7 |
| `restcos05` | cosine | 0.5 | none | 1.0 |

### 24.5 记录指标

重点记录 loss trajectory：

```text
val_loss_epoch_1
val_loss_epoch_2
val_loss_epoch_5
val_loss_mid
val_loss_final
val_auc
final_val_loss
```

以及：

```text
AUC_first_half
AUC_second_half
early_loss_gap_vs_adamw
late_loss_gap_vs_adamw
```

定义：

$$
\operatorname{AUC}_{first}=\frac{1}{T/2}\sum_{t=1}^{T/2}L_{val,t}.
$$

### 24.6 判定

Fashion 如果 `bothcos` 仍是唯一过 AUC gate，则 v3 Fashion default 固定 bothcos。

KMNIST 如果 `restcos07` 或 `bothcos07` 把 AUC 从 $7\%$ 推到 $8\%$ 以上，同时 accuracy 不掉超过 $0.5\%$，则进入 medium candidate。

---

## 25. V3-E5：EMA trigger and recovery ablation

### 25.1 目标

E5 验证 online geometry trigger 是否应该使用 EMA / hysteresis / recovery，而不是一次性 hard switch。

### 25.2 方法

| method | trigger | transition | recovery |
|---|---|---|---|
| `hard-current` | raw epoch audit | no/short | no |
| `ema-trigger` | EMA phi/J/branch | smooth | no |
| `ema-hysteresis` | EMA + hysteresis | smooth | no |
| `ema-recovery` | EMA + hysteresis | smooth | yes |

### 25.3 EMA 设置

```text
ema_beta = 0.8
min_active_frac = 0.10
max_active_frac = Fashion 0.50, KMNIST 0.35
hysteresis_low_mult = 0.85
recovery_cooldown_epochs = 1
```

### 25.4 记录指标

```text
switch_epoch
switch_reason
switch_reason_counts
shrink_events
geometry_recovery_after_switch
phi_peak_before_switch
phi_final_over_peak
jac_peak_before_switch
jac_final_over_peak
```

定义 geometry recovery：

$$
\operatorname{Recovery}_\phi
=
\frac{\phi_{peak}-\phi_{final}}{\phi_{peak}+\epsilon}.
$$

$$
\operatorname{Recovery}_J
=
\frac{J_{peak}-J_{final}}{J_{peak}+\epsilon}.
$$

### 25.5 判定

EMA trigger 成功要求：

$$
\operatorname{std}(switch\_epoch)\text{ 下降},
$$

$$
\text{seed acc std 不上升},
$$

$$
\text{AUC 不下降超过 }1\%,
$$

$$
\phi/J\text{ reduction 不下降超过 }10\%.
$$

如果 recovery events 频繁出现，说明默认 boost 太激进，不应把 recovery 作为成功点。

---

## 26. V3-E6：trust transparency

### 26.1 目标

确认 trust clipping 是否在拖慢或稳定训练。

### 26.2 设置

使用 E5 后的最佳 v3 core。

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2
methods = v3-trust-off, v3-trust-r015, v3-trust-r030
```

### 26.3 记录指标

```text
trust_clip_rate
trust_clip_epoch_hist
AUC_improvement
accuracy
phi/J reduction
bad_step_count
pred_actual_descent_ratio
```

### 26.4 判定

如果：

$$
\text{trust-off AUC 更好且 geometry 不坏},
$$

则 v3 default 设为 trust off。

如果：

$$
\text{trust-off 偶发 J spike},
$$

则 v3 default 用 safety-only，但 clip rate 必须：

$$
\operatorname{clip\_rate}<0.05.
$$

如果 `r0.15` clip rate 高于 $5\%$ 且 AUC 差，就说明 trust 太保守，默认改为 `r0.30` 或 off。

---

## 27. V3-E7：clean confirm

### 27.1 目标

E7 是正式 clean confirm。只比较少数固定方法，不再搜索。

### 27.2 方法

```text
AdamW
StaticFunctional
GA-FU-v2-tuned
GA-FU-v3-core
GA-FU-v3-nearby-1
```

其中 `GA-FU-v3-nearby-1` 只能是一个近邻备选，例如 KMNIST 的 `restcos07` 或 Fashion 的 `restcos0.3`。不能放入多个 profile 分散结论。

### 27.3 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2,3,4
optional confirm = 0..9
Fashion epochs = 30
KMNIST epochs = 20
model = h96 / depth4 / basis24 / alpha1.5
```

### 27.4 主要 scorecard

| metric | target |
|---|---|
| test acc gap vs AdamW | < 0.5% preferred, < 1.0% accepted |
| val-loss AUC improvement | > 8% medium, > 10% strong |
| phi reduction | > 25% |
| J reduction | > 20% |
| branch/AdamW | 0.55 to 0.85 |
| ECE reduction | > 10% |
| time-to-target | <= 1.1x AdamW for strong fast claim |

### 27.5 pass levels

Weak pass：

$$
\text{acc gap}<1.0\%,
$$

$$
\text{AUC improvement}>6\%,
$$

$$
\phi\text{ reduction}>20\%,
$$

$$
J\text{ reduction}>15\%,
$$

$$
0.45<\text{branch/AdamW}<0.95.
$$

Medium pass：

$$
\text{Fashion acc}\geq\text{AdamW acc},
$$

$$
\text{KMNIST acc gap}<0.5\%,
$$

$$
\text{AUC improvement}>8\%,
$$

$$
\phi\text{ reduction}>25\%,
$$

$$
J\text{ reduction}>20\%.
$$

Strong pass：

$$
\text{acc gap}<0.3\%,
$$

$$
\text{AUC improvement}>10\%,
$$

$$
\phi\text{ reduction}>30\%,
$$

$$
J\text{ reduction}>30\%,
$$

$$
\text{ECE reduction}>10\%,
$$

$$
\text{time-to-target}_{v3}\leq1.1\times\text{AdamW}.
$$

---

## 28. V3-E8：target-matched convergence

### 28.1 目标

这个包用于决定能不能 claim fast convergence。

### 28.2 target 定义

对每个 seed，AdamW 的 final val loss 是：

$$
L^*_{A,s}=L_{val,A,s}^{final}.
$$

AdamW final val acc 是：

$$
A^*_{A,s}=A_{val,A,s}^{final}.
$$

relaxed loss target：

$$
L^{relaxed}_{s}=1.05L^*_{A,s}.
$$

relaxed acc target：

$$
A^{relaxed}_{s}=A^*_{A,s}-0.005.
$$

记录每个 method 首次达到 target 的 step 和 wall-clock time。

### 28.3 指标

```text
target/reached_adamw_final_loss
target/steps_to_adamw_final_loss
target/time_to_adamw_final_loss
target/reached_relaxed_loss
target/steps_to_relaxed_loss
target/time_to_relaxed_loss
target/reached_adamw_final_acc
target/steps_to_adamw_final_acc
target/time_to_adamw_final_acc
```

### 28.4 可视化

```text
Kaplan-style target reach curve
steps_to_target paired bar
time_to_target paired bar
AUC vs time_to_target scatter
```

### 28.5 判定

只有当：

$$
\operatorname{time\_to\_target}_{v3}
<
\operatorname{time\_to\_target}_{AdamW},
$$

或至少：

$$
\operatorname{steps\_to\_target}_{v3}
<
\operatorname{steps\_to\_target}_{AdamW},
$$

且：

$$
\operatorname{step\_time\_overhead}<1.2,
$$

才可以 claim fast convergence。

如果 v3 只有 AUC 好但 time-to-target 不好，结论应写为 smoother trajectory，不写 faster convergence。

---

## 29. V3-E9：branch / geometry mechanism audit

### 29.1 目标

解释 v3 是否真正解决 effective functional step control problem。

### 29.2 需要记录的曲线

每个 epoch 记录：

```text
branch_over_adamw
branch_ratio_per_layer
no_kan_drop
kan_logit_delta_norm
kan_margin_contribution
phi_prime_p95
jacobian_condition
credit_amplification
phase_id
branch_scale
metric_mode
```

### 29.3 核心机制判定

v3 机制成功要求：

$$
\operatorname{branch/AdamW}_{final}>0.55,
$$

$$
\Delta_{noKAN}^{v3}>0.7\Delta_{noKAN}^{AdamW},
$$

$$
\phi'_{v3}<0.75\phi'_{AdamW},
$$

$$
\kappa(J)_{v3}<0.8\kappa(J)_{AdamW}.
$$

### 29.4 关键可视化

```text
branch_over_adamw vs epoch, with phase vertical line
no_kan_drop vs epoch, with phase vertical line
phi_prime_p95 vs epoch, with phase vertical line
jacobian_condition vs epoch, with phase vertical line
branch_ratio vs no_kan_drop scatter
branch_ratio vs phi_prime scatter
branch_ratio vs test_acc scatter
```

如果 branch ratio 上升但 no-KAN drop 不上升，则 v3 没有真正释放任务表达力。

---

## 30. V3-E10：generalization sanity

### 30.1 目标

确认 v3 没有牺牲泛化与校准优势。

### 30.2 small-data

```text
datasets = Fashion-MNIST, KMNIST
train_size = 500, 1000, 2000, 6000
methods = AdamW, GA-FU-v2-tuned, GA-FU-v3-core
seeds = 0,1,2,3,4
epochs = 20 or matched to default
```

### 30.3 label-noise sanity

不加 SNR，只测 v3 core：

```text
noise = 0.2, 0.4
methods = AdamW, GA-FU-v3-core
seeds = 0,1,2,3,4
```

### 30.4 指标

```text
clean_test_acc
noisy_train_acc_against_noisy_labels
noisy_train_acc_against_clean_labels
noise_memorization_rate
ECE
NLL
Brier score
train_test_gap_loss
train_test_gap_acc
phi_prime_p95
jacobian_condition
curvature_energy
```

### 30.5 判定

Fashion：

$$
\operatorname{Acc}_{v3}\geq\operatorname{Acc}_{AdamW}-0.5\%.
$$

KMNIST：

$$
\operatorname{AccGap}_{v3}<1.0\%.
$$

同时：

$$
\operatorname{ECE}_{v3}<\operatorname{ECE}_{AdamW},
$$

$$
\operatorname{train\_test\_gap}_{loss,v3}<\operatorname{train\_test\_gap}_{loss,AdamW}.
$$

---

# 第五部分：指标与可视化规范

## 31. 每个 run 必须记录的指标

### 31.1 任务性能

```text
task/train_loss
task/val_loss
task/test_loss
task/train_acc
task/val_acc
task/test_acc
task/best_val_loss
task/best_val_acc
task/best_epoch
```

### 31.2 收敛

```text
conv/train_loss_auc
conv/val_loss_auc
conv/val_acc_auc
conv/val_loss_auc_first_half
conv/val_loss_auc_second_half
conv/loss_at_10pct
conv/loss_at_25pct
conv/loss_at_50pct
conv/loss_at_75pct
conv/val_auc_improvement_vs_adamw
```

AUC improvement：

$$
\operatorname{AUCImprove}
=
\frac{\operatorname{AUC}_{AdamW}-\operatorname{AUC}_{method}}{\operatorname{AUC}_{AdamW}+\epsilon}.
$$

### 31.3 branch

```text
branch/global/mean_output_norm_ratio
branch/global/final_output_norm_ratio
branch/global/branch_over_adamw
branch/layer_k/ratio_mean
branch/layer_k/ratio_final
branch/layer_k/effective_alpha
branch/layer_k/branch_scale
ablation/no_kan_acc
ablation/no_kan_drop
ablation/no_kan_drop_over_adamw
ablation/kan_logit_delta_norm
ablation/kan_margin_contribution
```

No-KAN drop：

$$
\Delta_{noKAN}=\operatorname{Acc}_{full}-\operatorname{Acc}_{noKAN}.
$$

### 31.4 geometry

```text
kan/phi_prime_p95
kan/phi_prime_max
kan/curvature_mean
kan/curvature_p95
kan/sobolev_norm_mean
kan/sobolev_norm_p95
jacobian/condition_mean
jacobian/condition_p95
jacobian/condition_max
credit/amplification_mean
credit/amplification_p95
credit/noise_gain_p95
```

### 31.5 metric/update

```text
metric/mode_active
metric/mode_geometry
metric/condition_active
metric/condition_geometry
metric/eig_min
metric/eig_max
metric/update_amplification
metric/diag_full_direction_cos
update/raw_grad_norm
update/precond_direction_norm
update/functional_norm
update/delta_over_coeff_norm
update/predicted_descent
update/actual_descent
update/pred_actual_ratio
update/bad_step_count
```

### 31.6 schedule

```text
schedule/phase_id
schedule/branch_scale
schedule/coeff_lr_multiplier
schedule/effective_coeff_lr
schedule/rest_lr
schedule/metric_mix
schedule/switch_step
schedule/switch_epoch
schedule/switch_reason
schedule/transition_progress
schedule/recovery_events
schedule/shrink_events
```

### 31.7 trust

```text
trust/mode
trust/radius
trust/clip_rate
trust/clip_count
trust/update_metric_norm_mean
trust/update_metric_norm_p95
```

### 31.8 calibration/generalization

```text
calibration/ece
calibration/nll
calibration/brier
generalization/train_test_gap_acc
generalization/train_test_gap_loss
generalization/val_test_gap_loss
```

### 31.9 compute

```text
compute/step_time_ms
compute/epoch_time_sec
compute/total_train_time_sec
compute/precond_solve_time_ms
compute/metric_build_time_ms
compute/throughput_samples_per_sec
memory/peak_allocated_mb
```

---

## 32. Dashboard 设计

### Page 1：v3 scorecard

表格：

```text
dataset
method
test_acc_mean ± std
acc_gap_vs_adamw
val_auc_improvement
phi_reduction
J_reduction
branch_over_adamw
no_kan_drop
ECE_reduction
step_time_ratio
pass_level
```

### Page 2：loss trajectory

图：

```text
train_loss vs epoch
val_loss vs epoch
val_acc vs epoch
AUC first/second half bar
```

### Page 3：phase behavior

图：

```text
phase_id vs epoch
branch_scale vs epoch
coeff_lr_multiplier vs epoch
metric_mix vs epoch
switch_step histogram
switch_reason count
```

### Page 4：branch mechanism

图：

```text
branch_over_adamw vs epoch
no_kan_drop vs epoch
per-layer branch ratio heatmap
branch_ratio vs no_kan_drop scatter
branch_ratio vs test_acc scatter
```

### Page 5：geometry safety

图：

```text
phi_prime_p95 vs epoch
jacobian_condition vs epoch
credit_amplification_p95 vs epoch
curvature_energy vs epoch
```

### Page 6：metric analysis

图：

```text
metric condition by method
update amplification by method
diag-full direction cosine by epoch
precond solve time by method
```

### Page 7：target-matched convergence

图：

```text
steps_to_target loss bar
time_to_target loss bar
steps_to_target acc bar
time_to_target acc bar
Kaplan target reach curve
```

### Page 8：Pareto plots

至少画：

```text
x = val_loss_auc, y = test_acc, size = J_reduction, color = method
x = phi_prime_p95, y = test_acc, size = branch_over_adamw
x = step_time_ms, y = test_acc, size = memory_peak
x = branch_over_adamw, y = no_kan_drop, size = test_acc
```

---

# 第六部分：失败诊断与行动规则

## 33. failure table

每个 run 都要打 failure label。

| failure type | 触发条件 | 含义 | 优先行动 |
|---|---|---|---|
| `accuracy_gap_too_large` | acc gap > gate | 表达力或优化不足 | 看 branch/no-KAN，再调 coeff/rest schedule |
| `no_convergence_gain` | AUC improvement <= 0 | trajectory 不够快 | 调 rest/head schedule、smooth transition |
| `wall_clock_no_gain` | time-to-target 不优 | 单步太慢或 target 达不到 | 看 solve time / overhead |
| `branch_underactive` | branch/AdamW < 0.5 且 no-KAN drop 小 | KAN 没用起来 | 增 coeff_lr 或 branch boost，延长 ACTIVE |
| `branch_overactive` | branch/AdamW > 1.1 | branch 太强 | 提前 switch、降低 boost、提高 full Gram |
| `geometry_not_preserved` | phi/J reduction 不达标 | 几何优势消失 | 增 full Gram、降低 final scale、开启 recovery |
| `switch_too_early` | switch < 10% training | 表达力释放不足 | 提高 min_active_frac 或阈值 |
| `switch_too_late` | switch > max_active_frac + tolerance | Sobolev 介入过晚 | 修 phase logic |
| `trust_too_active` | clip_rate > 5% | trust 在主导 optimizer | 放宽 radius 或关 trust |
| `metric_condition_bad` | cond > 1e4 | Gram/damping 有问题 | 增 rho 或降 alpha/beta |
| `full_gram_too_slow` | overhead > 1.3x | 工程成本高 | cache/cholesky/diag fallback |
| `no_kan_no_contribution` | no-KAN drop < 0.7 AdamW | branch 没做任务 | 调 branch learning，不只看 norm |
| `seed_unstable` | acc std > 0.8% | profile 不稳 | 复查 switch/recovery seed 分布 |

---

## 34. 诊断决策树

### 34.1 accuracy 不够，geometry 很好

如果：

$$
\text{acc gap}>1\%,
\quad
\phi/J\text{ red 很高},
\quad
\text{branch/AdamW}<0.5,
$$

这是 branch under-active。行动：

```text
increase coeff_lr slightly
increase branch_boost slightly
lengthen ACTIVE phase
reduce alpha_geo in early geometry
```

不要先加 momentum。

### 34.2 accuracy 好，AUC 不好

如果：

$$
\text{acc gap}<0.5\%,
\quad
\text{AUC improvement}<0,
$$

这是 trajectory 问题。行动：

```text
inspect early val_loss gap
use rest/head cosine schedule
use smooth transition
reduce abrupt metric switch
compare AUC first_half vs second_half
```

### 34.3 branch 高但 no-KAN drop 小

如果：

$$
\text{branch/AdamW}>0.7,
\quad
\Delta_{noKAN}\text{ 小},
$$

说明 branch norm 变大但任务贡献不足。行动：

```text
check per-layer branch
check logit delta / margin contribution
reduce branch boost, improve coeff direction
try full Gram or diag-to-full transition
```

### 34.4 full Gram 伤 AUC

如果：

$$
\text{full Gram AUC worse than diag},
$$

不要立刻放弃 full Gram。先试：

```text
full Gram only in late GEOMETRY
smaller alpha_geo
longer transition
basis_diag_gram active + full_gram late
```

如果仍然差，v3 default 用 `basis_diag_gram`，full Gram 作为 geometry ablation。

### 34.5 trust clip rate 高

如果：

$$
\operatorname{clip\_rate}>0.05,
$$

则当前 trust 已不是 safety-only。行动：

```text
compare trust off / radius 0.30
report trust as active component
avoid using that run as clean v3 default unless necessary
```

---

# 第七部分：推荐执行命令

## 35. smoke

```bash
python experiments/run_gafu_v3.py \
  --packages V3_E0 \
  --datasets Fashion-MNIST,KMNIST \
  --seeds 0 \
  --out-dir results/gafu_v3_e0_smoke \
  --fresh --device auto --no-download \
  --epochs 1 --train-size 512 --val-size 128 --test-size 128 \
  --hidden-dim 32 --depth 2 --basis-count 8 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 64 \
  --continue-on-error
```

## 36. phase / metric ablation

```bash
python experiments/run_gafu_v3.py \
  --packages V3_E2,V3_E3 \
  --datasets Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --out-dir results/gafu_v3_phase_metric_seed012 \
  --fresh --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

Fashion 需要 30 epochs 的正式版本：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_E2,V3_E3,V3_E4 \
  --datasets Fashion-MNIST \
  --seeds 0,1,2 \
  --out-dir results/gafu_v3_fashion_seed012_e30 \
  --fresh --device auto --no-download \
  --epochs 30 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

## 37. clean confirm

```bash
python experiments/run_gafu_v3.py \
  --packages V3_E7 \
  --datasets Fashion-MNIST,KMNIST \
  --methods AdamW,StaticFunctional,GA-FU-v2-tuned,GA-FU-v3-core \
  --seeds 0,1,2,3,4 \
  --out-dir results/gafu_v3_clean_confirm_5seed \
  --fresh --device auto --no-download \
  --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

如果 5 seeds 过 medium 附近，再跑 10 seeds：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_E7,V3_E8,V3_E9 \
  --datasets Fashion-MNIST,KMNIST \
  --methods AdamW,GA-FU-v3-core \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --out-dir results/gafu_v3_clean_confirm_10seed \
  --device auto --no-download \
  --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

---

# 第八部分：最终交付文件

## 38. 每个实验目录应输出

```text
results/gafu_v3_*/
  runs.csv
  summary_by_method.csv
  summary_by_dataset.csv
  target_matched.csv
  branch_geometry_audit.csv
  metric_audit.csv
  trust_audit.csv
  failure_table.csv
  failure_summary_by_method.csv
  aggregate_summary.json
  runs/{run_id}/summary.json
  runs/{run_id}/curves.csv
```

## 39. aggregate_summary.json 必须包含

```text
v3_core_pass_level
fashion_pass_level
kmnist_pass_level
v3_vs_v2_auc_gain
v3_vs_v2_acc_gain
v3_vs_v2_phi_gain
v3_vs_v2_jac_gain
fast_convergence_claim_allowed
main_failure_type
recommended_stage_i_optimizer
recommended_cifar_small_default
```

---

# 第九部分：最终决策规则

## 40. 如果 v3 过 medium

如果满足：

$$
\text{Fashion acc}\geq\text{AdamW},
$$

$$
\text{KMNIST acc gap}<0.5\%,
$$

$$
\text{AUC improvement}>8\%,
$$

$$
\phi\text{ red}>25\%,
$$

$$
J\text{ red}>20\%,
$$

则：

```text
Stage I / CIFAR clean default = GA-FU-v3-core
AdamW remains accuracy baseline
GA-FU-v2 becomes historical baseline
```

## 41. 如果 v3 只过 weak

如果 v3 只满足 weak，但比 v2 有稳定提升，则：

```text
GA-FU-v3 = geometry-stable Pareto optimizer
AdamW = accuracy/convergence baseline
Stage I clean scaling reports both AdamW and v3
```

不能 claim：

```text
v3 converges faster than AdamW
```

除非 E8 target-matched time 通过。

## 42. 如果 v3 没超过 v2

如果 v3 没有超过当前 tuned v2，则不继续在 optimizer 组件上堆东西。下一步回到：

```text
model capacity
basis parameterization
rest/head architecture
ConvStem-DGKAN for CIFAR small
training objective / label smoothing
```

也就是说，不把失败解释为“还缺一个新 optimizer 组件”。

---

# 第十部分：我对 v3 最可能成功的判断

我认为最有希望带来真实提升的不是单独 full Sobolev Gram，而是下面三件事的组合：

$$
\boxed{
\text{smooth phase alignment}
+
\text{rest/head coordinated cosine schedule}
+
\text{basis diag active -> full Gram late}
}
$$

其中 Fashion 的关键是 rest/head schedule；KMNIST 的关键是更自然的 coefficient direction 和更稳的 late geometry。full Gram 如果从一开始使用，可能会慢；但如果只在 late phase 使用，应该更符合 Stage B 里“diag 快、Sobolev 稳”的经验。

所以我建议 v3 的默认实验顺序是：

```text
先做 phase alignment，不改 metric；
再引入 true Gram，不改 schedule；
再做 rest/coeff schedule；
最后做 EMA/recovery/trust。
```

这样每一步都有明确因果，不会又变成 FGO 式大杂烩。

最终，v3 成功的形态应该不是“branch 越大越好”，而是：

$$
\boxed{
\text{branch 有任务贡献，loss trajectory 更快，几何仍明显优于 AdamW。}
}
$$

这正好对应你的三项要求：

$$
\boxed{
\text{高表达力：accuracy + no-KAN drop + branch contribution}
}
$$

$$
\boxed{
\text{快速收敛：AUC + target-matched steps/time}
}
$$

$$
\boxed{
\text{几何好：}\phi'_{p95}+\kappa(J)+credit\ amplification
}
$$

---

# 第十一部分：Rational / GR-KAN 替代 RBF 的必要性评估与 v3-R 实验计划

> 新增目的：回答一个单独但非常关键的问题：当前 DG-KAN 的 RBF edge primitive 明显比 MLP 重，是否应该把 RBF-KAN primitive 换成 Rational / Group-Rational KAN primitive。  
> 本节不是替代 GA-FU-v3 optimizer 计划，而是给 v3 增加一个 **Rational primitive scaling track**，记为 **v3-R**。  
> 结论先行：**Rational 必须进入下一轮实验计划，但不应直接替换当前 RBF-DG-KAN default。**它应作为 scaling-critical candidate，与 RBF-DG-KAN 并行验证。

---

## 43. 为什么这个问题现在必须处理

当前 DG-KAN 的 RBF primitive 是研究上最干净的版本，因为它保留了 KAN 的核心性质：

$$
\text{每条 edge 对应一个可学习一维函数。}
$$

当前 RBF-DG-KAN 的 edge function 为：

$$
\phi_{ji}(t)=\sum_{m=1}^{M}a_{jim}B_m(t).
$$

对应 layer 为：

$$
y_j=\sum_i\phi_{ji}(x_i).
$$

所以参数规模是：

$$
\operatorname{Params}_{RBF}
=
 c_{out}c_{in}M+c_{out}.
$$

这正好支持当前 DG-KAN 主线里的三件事：

```text
edge-wise function learning
analytic adjoint
functional-space update
```

RBF 版本的 analytic adjoint 很清楚：

$$
g_{x_i}=\sum_j g_{y_j}\phi'_{ji}(x_i).
$$

functional update 也很清楚：

$$
a\leftarrow a-\eta(M_{Sob}+\rho I)^{-1}\nabla_aL.
$$

但是它的工程代价也非常明确：

```text
coefficient tensor: [c_out, c_in, basis_count]
basis tensor:       [batch, c_in, basis_count]
weighted sum:       einsum("bik,oik->bo")
```

因此，RBF-DG-KAN 的主要瓶颈不是理论，而是 scaling：

$$
\operatorname{Params}_{RBF}=O(c_{out}c_{in}M),
$$

$$
\operatorname{FLOPs}_{RBF}=O(Bc_{out}c_{in}M),
$$

$$
\operatorname{activation\ memory}_{RBF}=O(Bc_{in}M),
$$

$$
\operatorname{dense\ edge\ stats}=O(c_{out}c_{in}M).
$$

这意味着：在 Fashion / KMNIST 小模型上，RBF-DG-KAN 是合理的机制验证工具；但如果进入 CIFAR-small、ConvStem-DGKAN、DG-KANFormer 或 Transformer FFN scaling，RBF dense edge bank 很可能成为主瓶颈。

所以 Rational 的必要性不是来自“RBF 不好”，而是来自另一个问题：

$$
\boxed{
\text{RBF-DG-KAN 能证明机制，但未必是可扩展 primitive。}
}
$$

---

## 44. Rational / GR-KAN 和当前 RBF-DG-KAN 的本质差别

Rational / GR-KAN 的形式更接近：

$$
z_i=R_{q(i)}(x_i),
$$

$$
y_j=\sum_i W_{ji}z_i+b_j.
$$

其中 $q(i)$ 表示 channel $i$ 所属 group，$R_q$ 是该 group 共享的 rational function。

一个常见 rational function 写法是：

$$
R_q(t)=\frac{P_q(t)}{Q_q(t)}.
$$

例如：

$$
P_q(t)=\sum_{r=0}^{m}p_{qr}t^r,
$$

$$
Q_q(t)=1+\sum_{s=1}^{n}\tilde q_{qs}t^s.
$$

实际实现应使用 safe denominator，例如：

$$
Q_q(t)=1+\sum_{s=1}^{n}|q_{qs}||t|^s,
$$

或其他 guaranteed-positive parameterization，避免 pole / denominator collapse。

Rational / GR-KAN 的参数规模约为：

$$
\operatorname{Params}_{Rat}
=
 c_{out}c_{in}+O(g(m+n))+c_{out}.
$$

严格计数会因是否固定 $Q_0=1$、是否共享 denominator、是否加入 base scale 而有 $O(g)$ 的差异，但量级结论不变：

$$
\operatorname{Params}_{Rat}\approx O(c_{out}c_{in}).
$$

所以它和 RBF 的差异是：

| primitive | 学到的函数对象 | 参数量 | GPU 友好性 | 当前 DG-KAN 叙事适配度 |
|---|---|---:|---:|---:|
| MLP / Linear | 固定激活 + 参数矩阵 | $O(c_{out}c_{in})$ | 高 | 低 |
| RBF-DG-KAN | 每条 edge 一个函数 | $O(c_{out}c_{in}M)$ | 中低 | 最高 |
| Rational / GR-KAN | 每个 group 一个可学习函数 + linear mixing | $O(c_{out}c_{in}+g(m+n))$ | 高 | 中高 |

RBF-DG-KAN 是最忠实的 edge-wise KAN；Rational / GR-KAN 是更工程化、更适合 scaling 的 activation/group-wise KAN。

---

## 45. 是否“必须换成 Rational”：我的判断

我的判断分成两层。

### 45.1 不应该直接替换当前 v3 default

当前 v3 的核心是：

$$
\boxed{
\text{phase-aligned functional metric}
+
\text{true Sobolev Gram}
+
\text{branch schedule}
+
\text{geometry feedback}
}
$$

这套 optimizer 的机制依赖 RBF edge coefficients：

$$
a_{jim}
$$

以及 edge-wise basis Gram：

$$
M_{mn}
=
\int B_mB_n
+
\alpha\int B'_mB'_n
+
\beta\int B''_mB''_n.
$$

如果直接把 RBF 换成 Rational，v3 里的 functional update 不再是同一个对象。Rational 的 learnable function 是 group-wise activation $R_q$，而不是 edge-wise function $\phi_{ji}$。因此 functional metric 也要改成 rational-parameter metric：

$$
G_q(\theta)
=
\int J_{\theta_q}R_q(t)^\top J_{\theta_q}R_q(t)dt
+
\alpha\int J_{\theta_q}R'_q(t)^\top J_{\theta_q}R'_q(t)dt
+
\beta\int J_{\theta_q}R''_q(t)^\top J_{\theta_q}R''_q(t)dt.
$$

这里 $\theta_q$ 是 rational numerator / denominator 参数。这个 metric 是 state-dependent 的，因为 $R_q$ 对 denominator 参数是非线性的。

所以，Rational 不是简单替换 basis。它会改变：

```text
function object
functional metric
adjoint formula
geometry risk
branch contribution semantics
```

因此不能在没有实验 gate 的情况下把它变成 v3 default。

### 45.2 但必须加入 v3-R scaling track

虽然不能直接替换，但 Rational 必须加入下一轮计划。原因很直接：

$$
\boxed{
\text{当前 RBF primitive 的 } O(c_{out}c_{in}M) \text{ 成本会阻碍 scaling。}
}
$$

如果我们的目标只是继续在 Fashion / KMNIST 上证明 functional update 的机制，RBF 足够好；但你的目标包括：

```text
高模型表达力
快速收敛
几何好
```

其中“快速收敛”不能只看 step 数，还要看 wall-clock。RBF 的单步时间、basis tensor、edge coefficient、dense stats 都比 MLP 重很多。即使 AUC 好，如果 wall-clock 到 target 变慢，fast convergence claim 仍然不成立。

Rational / GR-KAN 是必须测试的原因是：它有机会在保持 learnable function expressivity 的同时，把参数 / FLOPs / memory 拉回接近 MLP 的量级。

所以本计划新增一个明确分支：

$$
\boxed{
\text{v3-R = Rational-DG-KAN primitive + GA-FU-compatible optimizer audit}
}
$$

它的目标不是推翻 RBF，而是回答：

$$
\boxed{
\text{Rational 能否成为 DG-KAN 的 scalable primitive？}
}
$$

---

## 46. Rational-DG-KAN 的候选结构

### 46.1 R-DGKAN-Act：group rational activation + linear

这是最应该先实现的版本，因为它最接近 KAT / GR-KAN 的工程形式。

输入：

$$
x\in\mathbb R^{c_{in}}.
$$

Group rational activation：

$$
z_i=R_{q(i)}(x_i).
$$

Linear mixing：

$$
y=Wz+b.
$$

Residual block：

$$
h_{k+1}=h_k+\alpha_k b_t W_kR_k(\operatorname{LN}(h_k)).
$$

这里 $R_k$ 表示逐 channel 应用 group rational function。

参数：

$$
W_k\in\mathbb R^{d\times d},
$$

$$
\theta_R\in\mathbb R^{g\times(m+n+O(1))}.
$$

所以每层参数约为：

$$
d^2+g(m+n)+d.
$$

这比 RBF-DG-KAN 的：

$$
d^2M+d
$$

轻很多。

### 46.2 R-DGKAN-EdgeLite：linear edge weight + group rational function

R-DGKAN-Act 可以解释为：

$$
\phi_{ji}(t)=W_{ji}R_{q(i)}(t).
$$

这仍然是 edge function，但它是低秩共享形式：

$$
\phi_{ji}\in\operatorname{span}\{R_{q(i)}\}.
$$

这说明 Rational 不是完全丢掉 KAN，而是把 edge-wise 函数族限制为 group-shared activation，再用 $W$ 做 mixing。

这个结构的优点是：

```text
参数接近 MLP
主计算是 GEMM
derivative analytic
branch schedule 可以复用
```

缺点是：

```text
每条 edge 不再有独立函数
表达力低于 RBF edge bank
functional update 的对象从 edge coefficient 变成 group rational parameters
```

### 46.3 R-DGKAN-Hybrid：小 RBF residual + Rational 主支路

如果纯 Rational 在表达力上不足，可以考虑 hybrid，但不进入第一轮默认。

形式：

$$
\operatorname{Branch}(x)
=
W R_g(x)+\lambda\operatorname{RBF}_{narrow}(x).
$$

其中 $\operatorname{RBF}_{narrow}$ 使用小 basis 或 grouped connectivity。

这个版本只在 R0-R4 证明 Rational under-expressive 后再做，不应先加入，避免 v3-R 也变成大杂烩。

---

## 47. Rational analytic adjoint

Rational branch：

$$
y_j=\sum_i W_{ji}R_{q(i)}(x_i)+b_j.
$$

给定输出 credit $g_y$，输入 credit 为：

$$
g_{x_i}
=
\sum_j g_{y_j}W_{ji}R'_{q(i)}(x_i).
$$

即：

$$
g_x
=
R'(x)\odot W^\top g_y.
$$

其中 rational derivative 为：

$$
R'_q(t)=\frac{P'_q(t)Q_q(t)-P_q(t)Q'_q(t)}{Q_q(t)^2}.
$$

因此 Rational 仍然支持 analytic adjoint。它不会破坏 DG-KAN 的 primitive-adjoint co-design 主线。

但是 Rational 额外引入一个必须监控的几何风险：denominator。

必须记录：

$$
Q_{min}=\min_{q,t}|Q_q(t)|,
$$

$$
Q_{p01}=\operatorname{quantile}_{0.01}(|Q_q(t)|),
$$

以及：

$$
\max |R'_q(t)|,
\quad
\operatorname{p95}|R'_q(t)|,
\quad
\max |R''_q(t)|.
$$

如果 denominator 接近 0，Rational 会比 RBF 更容易产生 derivative spike。

所以 Rational 的 geometry gate 不能只看 $\phi'_{p95}$ 和 Jacobian condition，还要看：

$$
Q_{p01}>\tau_Q.
$$

---

## 48. Rational functional update 的定义

Rational 参数记为：

$$
\theta_q=(p_{q0},\dots,p_{qm},q_{q1},\dots,q_{qn}).
$$

普通 AdamW / SGD 更新是：

$$
\theta_q\leftarrow\theta_q-\eta\nabla_{\theta_q}L.
$$

如果要保留 functional update 叙事，应定义 group-level function metric：

$$
G_q(\theta)
=
\int J_{\theta_q}R_q(t)^\top J_{\theta_q}R_q(t)dt
+
\alpha_s\int J_{\theta_q}R'_q(t)^\top J_{\theta_q}R'_q(t)dt
+
\beta_s\int J_{\theta_q}R''_q(t)^\top J_{\theta_q}R''_q(t)dt.
$$

更新为：

$$
\theta_q
\leftarrow
\theta_q-
\eta(G_q(\theta)+\rho I)^{-1}\nabla_{\theta_q}L.
$$

这个 metric 维度很小：

$$
G_q\in\mathbb R^{(m+n+O(1))\times(m+n+O(1))}.
$$

所以从计算量上它比 RBF edge Sobolev Gram 更轻。

但因为 $G_q$ 依赖当前 rational 参数，v3-R 第一轮不要直接假设它一定优于 AdamW。必须分开测试：

```text
Rational-AdamW
Rational-GA-schedule-only
Rational-functional-diag
Rational-functional-full
Rational-diag-to-full
```

这样才能判断 Rational 的收益来自：

```text
primitive 本身
branch schedule
functional metric
rest/head schedule
```

而不是混在一起。

---

## 49. v3-R 的核心假设

v3-R 只验证四个假设。

### 49.1 H-R1：Rational 显著降低工程成本

目标：

$$
\operatorname{step\ time}_{Rat}
<
0.7\operatorname{step\ time}_{RBF}
$$

或：

$$
\operatorname{peak\ memory}_{Rat}
<
0.6\operatorname{peak\ memory}_{RBF}.
$$

如果 Rational 速度 / memory 没明显好，就没有替换 RBF 的工程理由。

### 49.2 H-R2：Rational 不显著牺牲表达力

目标：

$$
\operatorname{Acc}_{Rat}
\geq
\operatorname{Acc}_{RBF}-1.0\%.
$$

并且：

$$
\Delta_{noKAN}^{Rat}
\geq
0.7\Delta_{noKAN}^{RBF}.
$$

如果 Rational 参数少但 KAN branch 没任务贡献，就不能作为 DG-KAN default。

### 49.3 H-R3：Rational 几何可控

目标：

$$
\phi'_{p95,Rat}
\leq
1.1\phi'_{p95,RBF},
$$

$$
\kappa(J)_{Rat}
\leq
1.2\kappa(J)_{RBF},
$$

$$
Q_{p01}>\tau_Q.
$$

其中 $\tau_Q$ 初始可设为：

$$
\tau_Q=0.1.
$$

### 49.4 H-R4：Rational 能改善 wall-clock convergence

目标不只是 step AUC：

$$
\operatorname{AUC}_{val,Rat}<\operatorname{AUC}_{val,RBF}
$$

还要：

$$
\operatorname{time\ to\ target}_{Rat}<\operatorname{time\ to\ target}_{RBF}.
$$

如果 Rational 只是在参数上轻，但收敛更慢或不稳定，不能替代 RBF。

---

## 50. v3-R 实验包总览

新增实验包：

```text
V3-R0: Rational implementation smoke
V3-R1: parameter / FLOPs / wall-clock accounting
V3-R2: width-matched and param-matched task comparison
V3-R3: Rational optimizer compatibility
V3-R4: geometry and denominator safety audit
V3-R5: expression mechanism audit
V3-R6: CIFAR-small scaling precheck
V3-R7: final decision gate
```

这些包不替代 V3-E0 到 V3-E10，而是并行放在 v3 后半段。建议执行顺序是：

```text
先完成 V3-E0/E1/E2，确保 RBF v3 core 正常；
再启动 V3-R0/R1/R2；
只有 Rational 通过 R0-R2，才做 R3/R4；
只有 R3/R4 通过，才进入 CIFAR-small R6。
```

---

## 51. V3-R0：Rational implementation smoke

### 51.1 目标

确认 Rational primitive 的 forward、backward、analytic adjoint、denominator safety 和 logging 都正确。

### 51.2 设置

```text
datasets = Fashion-MNIST, KMNIST
train / val / test = 512 / 128 / 128
epochs = 1
seeds = 0
model = h32 / depth2
rational degree = m5 n4
groups = 4 or 8
methods = RBF-DGKAN-smoke, Rational-DGKAN-AdamW, Rational-DGKAN-GA-schedule-only
```

### 51.3 必须记录

```text
run_failed
nan_or_inf_count
rational/denominator_min
rational/denominator_p01
rational/r_prime_p95
rational/r_prime_max
rational/r_double_prime_p95
rational/group_function_norm_mean
rational/group_function_norm_max
rational/analytic_vjp_cos
rational/analytic_vjp_relerr
compute/step_time_ms
memory/peak_allocated_mb
```

### 51.4 通过条件

$$
\text{run failures}=0.
$$

$$
\text{all metrics finite}.
$$

$$
\operatorname{analytic\ vjp\ cos}>0.99999.
$$

$$
\operatorname{analytic\ vjp\ relerr}<10^{-5}.
$$

$$
Q_{p01}>0.1.
$$

如果 R0 失败，优先修 rational parameterization 和 safe denominator，不做调参。

---

## 52. V3-R1：parameter / FLOPs / wall-clock accounting

### 52.1 目标

确认 Rational 是否真的解决了 RBF-DG-KAN 的工程负担。

### 52.2 对照设置

对每个 dataset 记录以下模型：

```text
MLP h96 d4
RBF-DGKAN h96 d4 basis24
Rational-DGKAN h96 d4 groups8 m5 n4
Rational-DGKAN h192 d4 groups16 m5 n4
Rational-DGKAN h256 d4 groups16 m5 n4
```

这里要同时做三种公平性：

```text
width-matched: hidden_dim 相同
parameter-matched: 参数量接近
wall-clock-matched: step_time 接近
```

RBF 与 Rational 的参数近似关系：

$$
\operatorname{Params}_{RBF}\approx d^2M,
$$

$$
\operatorname{Params}_{Rat}\approx d_R^2.
$$

若要粗略 parameter-match：

$$
d_R\approx d\sqrt{M}.
$$

但 $M=24$ 时 $\sqrt{M}\approx4.9$，直接 parameter-match 会让 Rational hidden 过大。因此实际可采用分级：

```text
h96: width-matched
h192: moderate upscaling
h256: aggressive but still manageable
```

### 52.3 指标

```text
params/total
params/kan_or_rational
params/nonkan
compute/step_time_ms
compute/throughput_samples_per_sec
memory/peak_allocated_mb
memory/activation_proxy_mb
flops/proxy_forward
flops/proxy_backward
```

### 52.4 可视化

```text
params vs test_acc
step_time_ms vs test_acc
memory_peak vs test_acc
params vs val_auc
step_time_ms vs val_auc
```

### 52.5 判定

Rational 必须至少满足：

$$
\operatorname{step\ time}_{Rat,h96}<0.7\operatorname{step\ time}_{RBF,h96}
$$

或：

$$
\operatorname{memory}_{Rat,h96}<0.6\operatorname{memory}_{RBF,h96}.
$$

否则它没有工程替换意义。

---

## 53. V3-R2：width-matched / capacity-matched task comparison

### 53.1 目标

判断 Rational 是否因为共享函数而表达力不足。

### 53.2 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2
models:
  MLP h96 d4
  RBF-DGKAN h96 d4 basis24
  Rational h96 d4 g8 m5 n4
  Rational h192 d4 g16 m5 n4
  Rational h256 d4 g16 m5 n4
methods:
  AdamW
  GA-FU-v3-RBF-core
  Rational-AdamW
```

先只跑 Rational-AdamW，原因是要先分离 architecture effect。如果 Rational-AdamW 已经明显弱，则不应该急着上 functional update。

### 53.3 指标

```text
test_acc
val_loss_auc
train_acc
train_loss
ece
no_kan_drop
branch_over_rbf
branch_over_mlp_if_defined
rational/r_prime_p95
rational/denominator_p01
jacobian/condition_max
compute/step_time_ms
memory/peak_allocated_mb
```

### 53.4 判定

Rational architecture 通过 R2 的条件：

$$
\operatorname{Acc}_{Rat,best}
\geq
\operatorname{Acc}_{RBF}-1.0\%.
$$

并且：

$$
\operatorname{ValAUC}_{Rat,best}
\leq
1.1\operatorname{ValAUC}_{RBF}.
$$

同时必须满足：

$$
Q_{p01}>0.1.
$$

如果 h96 Rational 弱，但 h192 / h256 能追上 RBF，说明 Rational 可以通过宽度换回表达力。只要 step-time / memory 仍优于 RBF，就值得继续。

---

## 54. V3-R3：Rational optimizer compatibility

### 54.1 目标

判断 GA-FU 的核心思想能否迁移到 Rational primitive。

### 54.2 方法矩阵

| method | Rational params | Linear / non-rational params | branch schedule | metric |
|---|---|---|---|---|
| `R-AdamW` | AdamW | AdamW | none | none |
| `R-GA-schedule-only` | AdamW | AdamW | GA-FU branch schedule | none |
| `R-FuncDiag` | group functional diag | AdamW | GA-FU branch schedule | diag $G_q$ |
| `R-FuncFull` | group functional full | AdamW | GA-FU branch schedule | full $G_q$ |
| `R-DiagToFull` | diag-to-full group metric | AdamW | GA-FU branch schedule | diag-to-full |
| `RBF-GA-FU-v3` | RBF functional | AdamW | GA-FU-v3 | Sobolev Gram |

### 54.3 Rational functional metric

对每个 group 构造：

$$
G_q(\theta)
=
\Delta t\cdot J_R^\top J_R
+
\alpha_s\Delta t\cdot J_{R'}^\top J_{R'}
+
\beta_s\Delta t\cdot J_{R''}^\top J_{R''}.
$$

其中：

$$
J_R(t)=\frac{\partial R_q(t)}{\partial\theta_q}.
$$

更新：

$$
\Delta\theta_q
=
-\eta(G_q+\rho I)^{-1}\nabla_{\theta_q}L.
$$

### 54.4 记录指标

```text
metric/group_condition_mean
metric/group_condition_max
metric/update_amplification
metric/predicted_descent
metric/actual_descent
metric/pred_actual_ratio
rational/denominator_p01
rational/r_prime_p95
rational/r_double_prime_p95
schedule/branch_scale
schedule/switch_epoch
trust/clip_rate
```

### 54.5 判定

如果 `R-GA-schedule-only` 已经和 `R-FuncDiag/Full` 一样好，说明 Rational 上 functional metric 不是关键，Rational 的默认 optimizer 应保持 AdamW + branch schedule。

如果 `R-DiagToFull` 明显优于 `R-AdamW`，并保留 geometry，那么 Rational 可以进入 GA-FU-v3 的 functional optimizer 主线。

默认晋级条件：

$$
\operatorname{AUCImprove}_{R-DiagToFull}>5\%,
$$

$$
\operatorname{AccGap}_{R-DiagToFull}<1.0\%,
$$

$$
Q_{p01}>0.1,
$$

$$
\kappa(J)_{R-DiagToFull}<1.2\kappa(J)_{R-AdamW}.
$$

---

## 55. V3-R4：geometry and denominator safety audit

### 55.1 目标

Rational 最大的新增风险是 denominator 和 derivative spike。R4 专门检查这个问题。

### 55.2 必须记录曲线

```text
rational/denominator_min vs epoch
rational/denominator_p01 vs epoch
rational/r_prime_p95 vs epoch
rational/r_prime_max vs epoch
rational/r_double_prime_p95 vs epoch
jacobian/condition_max vs epoch
credit/amplification_p95 vs epoch
branch/branch_over_rbf vs epoch
branch/no_kan_drop vs epoch
```

### 55.3 Pole-risk failure

新增 failure type：

```text
rational_pole_risk
```

触发条件：

$$
Q_{p01}<0.05
$$

或：

$$
\max |R'|>5\times \operatorname{p95}|R'|.
$$

新增 failure type：

```text
rational_derivative_spike
```

触发条件：

$$
R'_{p95}>1.2R'_{p95,AdamW}
$$

并且：

$$
\kappa(J)>1.2\kappa(J)_{AdamW}.
$$

### 55.4 判定

Rational 不能只靠 accuracy 过关。它必须满足：

$$
Q_{p01}>0.1,
$$

$$
R'_{p95}\text{ finite and stable},
$$

$$
\kappa(J)\text{ no worse than RBF by more than }20\%.
$$

否则它只能作为 fast but unstable ablation，不能作为 DG-KAN default。

---

## 56. V3-R5：expression mechanism audit

### 56.1 目标

Rational 的主要风险是 group sharing 导致表达力不足。因此要单独审计：它到底有没有学到有用的 KAN branch。

### 56.2 1D function fitting

任务：

$$
y=\sin x+0.3\sin(5x),
$$

$$
y=\operatorname{smoothstep}(x),
$$

$$
y=\sin(3x)+0.2\operatorname{sign}(\sin(7x)).
$$

方法：

```text
MLP small
RBF edge KAN
Rational group KAN g4/g8/g16
Rational higher degree m7 n6
```

记录：

```text
train_mse
val_mse
loss_auc
function_l2_error
slope_p95
curvature_energy
rational_denominator_p01
```

### 56.3 Classification mechanism

对 Fashion / KMNIST，记录：

```text
no_kan_drop
kan_logit_delta_norm
kan_margin_contribution
branch_ratio_per_layer
group_function_diversity
active_group_fraction
linear_weight_rank
```

Group function diversity：

$$
D_R
=
\frac{1}{g(g-1)}\sum_{q\ne q'}
\frac{\langle R_q,R_{q'}\rangle}{\|R_q\|\|R_{q'}\|+\epsilon}.
$$

如果 $D_R$ 过高，说明不同 group 学到的函数太相似，实际退化成一个共享激活 MLP。

### 56.4 判定

Rational mechanism pass：

$$
\Delta_{noKAN}^{Rat}>0.7\Delta_{noKAN}^{RBF},
$$

$$
\operatorname{branch/AdamW}_{Rat}\in[0.45,0.95],
$$

$$
D_R<0.95.
$$

如果 Rational accuracy 高但 no-KAN drop 很小，说明它只是更像 MLP，不应作为 DG-KAN 的函数空间主线。

---

## 57. V3-R6：CIFAR-small scaling precheck

### 57.1 目标

Rational 的真正价值在 scaling，因此必须做一个小型 CIFAR precheck。

### 57.2 设置

```text
dataset = CIFAR-10 small
train / val / test = 10000 / 2000 / 2000
model = ConvStem + residual blocks
methods:
  ConvStem-MLP
  ConvStem-RBF-DGKAN-GA-FU-v3
  ConvStem-Rational-DGKAN-AdamW
  ConvStem-Rational-DGKAN-v3-R
seeds = 0,1,2
epochs = 30 or 50
```

### 57.3 指标

```text
test_acc
val_auc
time_to_target
step_time_ms
peak_memory_mb
branch_over_adamw
no_kan_drop
phi_prime_p95 or r_prime_p95
jacobian_condition_proxy
ece
```

### 57.4 判定

CIFAR-small 上，Rational 通过 scaling precheck 需要：

$$
\operatorname{Acc}_{Rat}
\geq
\operatorname{Acc}_{RBF}-1.0\%,
$$

$$
\operatorname{time\ to\ target}_{Rat}
<
\operatorname{time\ to\ target}_{RBF},
$$

$$
\operatorname{memory}_{Rat}
<
0.7\operatorname{memory}_{RBF}.
$$

如果它只在 Fashion / KMNIST 小图上快，但 CIFAR-small 不快，则不能 claim scaling primitive。

---

## 58. V3-R7：最终决策规则

### 58.1 Rational 成为 scaling default

如果 Rational 满足：

$$
\text{acc gap vs RBF}<1.0\%,
$$

$$
\text{wall-clock target time}<\text{RBF},
$$

$$
\text{memory}<0.7\times\text{RBF},
$$

$$
\text{geometry no worse than RBF by more than }20\%,
$$

则：

```text
Fashion / KMNIST mechanism paper default仍可用RBF
CIFAR / DG-KANFormer scaling default = Rational-DGKAN-v3-R
```

### 58.2 Rational 作为 fast ablation

如果 Rational 快，但：

$$
\text{acc gap}>1.0\%
$$

或：

$$
\Delta_{noKAN}^{Rat}<0.7\Delta_{noKAN}^{RBF},
$$

则：

```text
Rational = fast engineering ablation
RBF = main DG-KAN functional-update primitive
```

### 58.3 Rational 不采用

如果 Rational 没有明显速度 / memory 优势，或者 denominator / derivative 不稳定，则：

```text
Do not replace RBF.
Continue RBF-v3.
Scaling should use grouped / low-rank / sparse RBF instead.
```

---

## 59. v3-R 新增 failure table

新增 failure labels：

| failure type | 触发条件 | 含义 | 行动 |
|---|---|---|---|
| `rational_underexpressive` | acc gap vs RBF > 1% and train acc also lower | group sharing 伤表达力 | 增 groups / hidden / degree，或保留 RBF |
| `rational_pole_risk` | $Q_{p01}<0.05$ | denominator 不安全 | safe denominator / clamp / smaller lr |
| `rational_derivative_spike` | $R'_{p95}$ 和 J 同时高于 AdamW | derivative geometry 失控 | stronger denominator regularization / lower branch |
| `group_sharing_too_strong` | group diversity $D_R>0.95$ | 退化成共享激活 | 增 groups / group-specific numerator |
| `functional_metric_mismatch` | functional metric 比 AdamW 差且 condition 高 | rational metric 定义不合适 | 使用 AdamW on rational params |
| `schedule_only_explains_gain` | schedule-only 与 functional 同级 | functional update 不是关键 | Rational default 用 AdamW + branch schedule |
| `no_wall_clock_gain` | step/time-to-target 不优于 RBF | 没有工程价值 | 不替换 RBF |

---

## 60. v3-R Dashboard 增补

新增 Dashboard 页面：

### Page R1：Primitive accounting

```text
params by primitive
step_time by primitive
peak_memory by primitive
throughput by primitive
```

### Page R2：Accuracy / cost Pareto

```text
x = step_time_ms, y = test_acc, size = memory, color = primitive
x = params_total, y = test_acc, size = val_auc, color = primitive
```

### Page R3：Rational geometry safety

```text
denominator_p01 vs epoch
r_prime_p95 vs epoch
r_double_prime_p95 vs epoch
jacobian_condition vs epoch
```

### Page R4：Rational expression mechanism

```text
no_kan_drop by primitive
branch_over_adamw by primitive
group_function_diversity by epoch
active_group_fraction by epoch
```

### Page R5：Scaling precheck

```text
CIFAR-small time_to_target
CIFAR-small memory_peak
CIFAR-small acc / ECE / AUC
```

---

## 61. 我对 Rational 替换的最终建议

我的建议是：

$$
\boxed{
\text{RBF 不要删，Rational 必须加。}
}
$$

原因是：

```text
RBF = 当前最清楚的 mechanism primitive
Rational = 最有希望的 scaling primitive
```

RBF-DG-KAN 继续承担：

```text
functional update 机制证明
Sobolev metric 验证
branch / geometry 解释
analytic adjoint correctness
```

Rational-DG-KAN 负责回答：

```text
这些机制能不能以接近 MLP 的成本保留下来？
```

所以 v3 的最终路线应改成双轨：

$$
\boxed{
\text{v3-RBF：机制主线}
\quad+
\quad
\text{v3-Rational：scaling 主线}
}
$$

如果 Rational 过 R0-R7 gate，那么 Stage I / CIFAR / DG-KANFormer 应默认优先测试 Rational-DGKAN；如果 Rational 没过，则继续用 RBF-DG-KAN，并转向 grouped / sparse / low-rank RBF 来降低成本。

一句话：

$$
\boxed{
\text{换 Rational 的必要性很高，但它必须以实验 gate 晋级，而不是凭参数量直接替换。}
}
$$
