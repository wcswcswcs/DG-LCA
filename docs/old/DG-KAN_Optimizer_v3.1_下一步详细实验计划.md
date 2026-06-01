# DG-KAN Optimizer v3.1 下一步详细实验计划：True-Gram Default 重定义与扩 seed 验证

> 版本：v3.1 next-action plan  
> 日期：2026-05-01  
> 适用范围：基于 GA-FU-v3 初始实验之后的下一轮执行计划。  
> 公式格式：Typora 友好，全文公式只使用 `$...$` 与 `$$...$$`，不使用 display bracket 公式。  
> 核心目标：修正 `diag_to_full_sobolev` 的 transition 语义，补齐 direction audit，重新定义 v3 default candidate，并用窄实验确认 KMNIST 与 Fashion-MNIST 的最优路线。

---

## 0. 当前结论先固定下来

这份计划建立在当前 v3 初始实验的三个事实之上。

第一，v3 的实现 smoke 已经通过。`build_rbf_sobolev_gram()`、`basis_diag_gram`、`full_sobolev_gram`、`diag_to_full_sobolev`、`ACTIVE -> TRANSITION -> GEOMETRY` state machine、smooth transition、trust accounting、preconditioner build/solve timing 和 metric condition logging 都已经能跑。E0 smoke 没有 run failure，没有 NaN / Inf，Gram condition 大约在 `6.5e3` 到 `6.9e3`，trust clip 没有实际参与。这说明 v3 的 true Gram 路线在数值上不是死路。

第二，E2 已经基本否定了 “phase alignment alone 就能修复 AUC” 这个假设。`V3-phase-aligned-hard` 接近 v2，但 AUC 没超过 v2；`V3-phase-aligned-smooth` 在 legacy diagonal metric 下反而明显伤 AUC。这里不能说明 phase alignment 没价值，但能说明它不是主因。当前最需要解释的是 functional metric direction，而不是只调 switch 形式。

第三，E3 seed0 显示 true Gram 有真实信号，尤其是 KMNIST 的 `V3-full-from-start`。KMNIST seed0 中它几乎不掉 accuracy，却把 AUC improvement 推到 `10.85%`，同时保留正的 geometry / calibration signal。Fashion seed0 中，`V3-diag-to-full-hard` 的 AUC 最强，但 accuracy 掉了；`V3-diag-gram` accuracy / geometry 好但 AUC 不如 v2。这说明 v3 不能简单沿用原计划里的 `diag-to-full-smooth` 作为默认，而应该按 dataset 重新定义 candidate。

因此，当前行动原则是：

$$
\boxed{
\text{先修 transition 语义，再解释 direction，再扩最少候选。}
}
$$

这轮不做大扫，不引入新 optimizer 组件，不把 SNR / momentum / prox / data metric 拉回主线。v3 的核心问题现在是：

$$
\boxed{
\text{true Gram direction 是否能在不牺牲表达力的情况下改善 loss trajectory 和几何。}
}
$$

---

## 1. 为什么必须先修 `diag_to_full_sobolev` transition 语义

当前代码里 `diag_to_full_sobolev` 的实现存在一个重要语义风险：transition phase 中构建 full direction 时，Gram 参数可能仍然来自 fast / active 参数，而不是 geometry 参数。

计划语义本来应该是：

$$
d_{diag-fast}
=
-\operatorname{diag}(M_{fast}+\rho I)^{-1}g,
$$

$$
d_{full-geo}
=
-(M_{geo}+\rho I)^{-1}g,
$$

然后在 transition 内做 direction mix：

$$
d_t
=
(1-\lambda_t)d_{diag-fast}+\lambda_t d_{full-geo}.
$$

其中：

$$
\lambda_t
=
\frac{1}{2}-\frac{1}{2}\cos(\pi\tau_t),
$$

$$
\tau_t
=
\frac{t-t_{switch}}{T_{transition}}.
$$

也就是说，transition 的含义不是 “diag fast 到 full fast”，而是 “fast diagonal direction 到 geometry full direction”。

如果当前实现实际做成：

$$
d_t
=
(1-\lambda_t)d_{diag-fast}+\lambda_t d_{full-fast},
$$

然后 transition 结束后突然切到：

$$
d_{full-geo},
$$

那 smooth transition 就不是 smooth。它只是在 transition 内平滑了一个不该平滑的对象，最后仍然会在 transition 末端发生 hard jump。这会解释两个现象：

1. E2 的 smooth alignment 伤 AUC，因为它削弱了 early effective step，却没有真正引入更好的 geometry direction；
2. E3 Fashion 的 `V3-diag-to-full-smooth` AUC 很好但 J reduction 为负，因为 metric 可能在错误阶段或错误强度下突变。

所以这轮实验的第一个结论不是 “smooth transition 不行”，而是：

$$
\boxed{
\text{当前 smooth transition 还不能代表计划中的 smooth transition。}
}
$$

必须先修实现，再重跑 seed0 probe，然后才决定 smooth 是否降级。

---

## 2. v3 default 候选重新定义

当前不要再把 `diag-to-full-smooth` 当默认。v3 default 需要拆成 dataset-specific candidate，因为 Fashion 和 KMNIST 的 E3 信号不同。

### 2.1 KMNIST default candidate

KMNIST 当前最像 default 的是：

```text
V3-full-from-start
```

它的语义是：从训练开始就使用 full shared Sobolev Gram：

$$
P_t=(M_{geo}+\rho I)^{-1}.
$$

它没有 diag warmup，也没有 transition。这个结果一开始看起来违反 Stage B 的 “diag early speed + Sobolev final quality” 经验，但 E3 seed0 给出很强信号：KMNIST 上 full Gram 可能并不保守，反而提供了更自然的 function-space direction。

因此 KMNIST 的假设改成：

$$
\boxed{
\text{KMNIST 的 AUC 卡点主要来自 diagonal approximation direction 不够好。}
}
$$

接下来应该扩 seed 的不是所有 true Gram 版本，而是 `full_sobolev_gram from start` 的窄变体。

### 2.2 Fashion-MNIST default candidate

Fashion 当前最像 default 的不是 full-from-start，而是：

```text
V3-diag-to-full-hard
```

原因是它在 seed0 上给出最强 AUC improvement，但 accuracy 低于 v2。Fashion 的核心问题不是 AUC 没信号，而是 true Gram 倾向于更强地改善 trajectory / geometry，同时牺牲了一点 final fitting。因此 Fashion 的下一步不是继续找更强 metric，而是做 accuracy rescue。

Fashion 的假设改成：

$$
\boxed{
\text{Fashion true Gram 改善优化路径，但需要更长训练和更轻的 final branch / coeff rescue 来恢复表达力。}
}
$$

Fashion 的实验必须回到 `epochs = 30`，因为之前 weak pass 的 tuned profile 就是在 30 epoch + rest/coeff schedule 下站住的。用 20 epoch 去判断 Fashion v3，会系统性低估它。

### 2.3 暂停作为默认的候选

`V3-diag-to-full-smooth` 暂停作为默认。它不是被永久否定，而是必须满足两个条件后才能回来：

1. transition 语义修复后，Fashion 的 J reduction 不再为负；
2. 3-seed 下 AUC 不低于 hard transition，且 accuracy 不差于 hard transition。

在这两个条件前，它只作为 diagnostic，不进入扩 seed default。

---

## 3. 总体执行顺序

这轮分为九个连续阶段，不能跳过前面的实现验证直接扩 seed。

```text
R0: transition 语义修复与单元检查
R1: E0 smoke after patch
R2: E1 direction audit
R3: E3 seed0 rerun after patch
R4: KMNIST full-from-start 3-seed expansion
R5: Fashion 30-epoch accuracy rescue 3-seed expansion
R6: dataset-specific default selection, seed012 paired comparison
R7: 5-seed / 10-seed clean confirm
R8: target-matched convergence and wall-clock accounting
R9: mechanism audit and final decision
```

R0 到 R3 的目标是确认 “代码和解释” 正确；R4 到 R6 的目标是确定 “候选”；R7 到 R9 才是正式 claim。

---

# 第一部分：实现修复与审计

## 4. R0：transition 语义修复

### 4.1 修改目标

当前 `functional_coeff_step()` 里对 `basis_diag_gram`、`full_sobolev_gram`、`diag_to_full_sobolev` 的处理应该拆成一个清晰的 direction builder。

建议新增 helper：

```python
def _v3_precondition_direction(
    grad,
    layer,
    cfg,
    state,
    *,
    metric_mode,
    phase,
    device,
    dtype,
):
    ...
```

这个 helper 返回：

```text
direction
trust_metric
active_gram_stats
geometry_gram_stats
direction_audit_stats
```

这样可以避免所有 metric mode 都塞在 `functional_coeff_step()` 的大分支里，也能让 direction audit 直接复用同一套逻辑。

### 4.2 修复后的 metric 语义

#### `basis_diag_gram`

使用 fast Gram 或当前 phase 指定的 Gram diagonal。默认 ACTIVE 用 fast：

$$
d_{basis-diag-fast}
=
-\operatorname{diag}(M_{fast}+\rho I)^{-1}g.
$$

这里建议先固定：

```text
alpha_fast = 0.0
beta_fast = 0.0
rho = 1e-3
```

#### `full_sobolev_gram`

无论 phase 是什么，只要 mode 明确是 `full_sobolev_gram`，就使用 geometry Gram：

$$
d_{full-geo}
=
-(M_{geo}+\rho I)^{-1}g.
$$

默认：

```text
alpha_geo = 0.15
beta_geo = 0.02
rho = 1e-3
```

#### `diag_to_full_sobolev`

必须显式构造两个方向：

$$
d_{diag-fast}
=
-\operatorname{diag}(M_{fast}+\rho I)^{-1}g,
$$

$$
d_{full-geo}
=
-(M_{geo}+\rho I)^{-1}g.
$$

然后做：

$$
d_t=(1-\lambda_t)d_{diag-fast}+\lambda_t d_{full-geo}.
$$

这里的 `trust_metric` 不应该含糊。建议 transition 中 trust metric 使用混合 diagonal：

$$
D_t=(1-\lambda_t)\operatorname{diag}(M_{fast}+\rho I)+\lambda_t\operatorname{diag}(M_{geo}+\rho I).
$$

如果为了简化，也可以用 geometry diagonal 作为 conservative trust metric：

$$
D_t=\operatorname{diag}(M_{geo}+\rho I).
$$

但必须在日志里记录使用哪一种。

### 4.3 新增 runtime state 字段

建议新增以下字段，不一定全部写入每步 CSV，但 summary 必须有 mean / p95：

```text
v3_direction_cos_diagfast_fullgeo
v3_direction_norm_ratio_fullgeo_diagfast
v3_direction_metric_norm_diagfast
v3_direction_metric_norm_fullgeo
v3_update_metric_norm_current
v3_update_over_coeff_norm_current
v3_transition_mix_mean
v3_transition_mix_auc
v3_diagfast_condition
v3_fullgeo_condition
v3_diagfast_eig_min
v3_fullgeo_eig_min
v3_diagfast_eig_max
v3_fullgeo_eig_max
```

这能回答三个问题：

1. full Gram 是否只是把步长放大了；
2. full Gram 是否真的改变了方向；
3. transition 是否在正确范围内平滑引入 full geometry direction。

### 4.4 R0 单元检查

R0 不需要完整训练，必须有三个 deterministic check。

#### Check A：direction endpoint check

构造一个随机 gradient `g`，用同一个 layer 的 RBF centers 构造 fast / geo Gram。检查：

当：

$$
\lambda=0,
$$

有：

$$
d_{diag-to-full}=d_{diag-fast}.
$$

当：

$$
\lambda=1,
$$

有：

$$
d_{diag-to-full}=d_{full-geo}.
$$

误差要求：

$$
\frac{\|d_{actual}-d_{expected}\|}{\|d_{expected}\|+\epsilon}<10^{-6}.
$$

#### Check B：Gram parameter check

记录 fast 和 geo 的 condition：

```text
cond_fast
cond_geo
eig_min_fast
eig_min_geo
eig_max_fast
eig_max_geo
```

如果 `alpha_fast=0,beta_fast=0`，`alpha_geo=0.15,beta_geo=0.02`，则两者 condition 通常不应该完全相同。完全相同说明 cache key 或参数传递有问题。

#### Check C：transition curve check

用 fake state 模拟 transition，确认 `current_metric_mix` 按 cosine 从 0 到 1。

记录：

```text
mix_first
mix_mid
mix_last
transition_length_steps
```

要求：

$$
mix_{first}>0
$$

当当前 switch step 被计入 transition 第一步时，`mix_first` 不应该永远等于 0。

$$
mix_{last}=1.
$$

### 4.5 R0 通过条件

R0 通过才允许进入 R1。

```text
py_compile pass
endpoint relative error < 1e-6
fast/geo Gram stats finite
transition mix reaches 1.0
no NaN / Inf in direction norms
```

---

## 5. R1：E0 smoke after patch

### 5.1 目标

R1 只确认修复后代码还能跑，不判断方法好坏。R1 的重点是看 transition 修复有没有引入数值问题。

### 5.2 设置

```text
datasets = Fashion-MNIST, KMNIST
train / val / test = 512 / 128 / 128
epochs = 1
seeds = 0
model = h32 / depth2 / basis8
methods = AdamW, GA-FU-v2-current, GA-FU-v3-smoke-fixed
```

### 5.3 推荐命令

```bash
python experiments/run_gafu_v3.py \
  --packages V3_E0 \
  --datasets Fashion-MNIST,KMNIST \
  --seeds 0 \
  --out-dir results/gafu_v3_r1_e0_after_transition_fix \
  --fresh --device auto --no-download \
  --epochs 1 --train-size 512 --val-size 128 --test-size 128 \
  --hidden-dim 32 --depth 2 --basis-count 8 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 64 \
  --continue-on-error --wandb --wandb-project DG-KAN
```

如果新增了单独 package，建议命名：

```text
V3_R1_SMOKE_FIXED
```

### 5.4 必须记录

```text
run_failed
nan_or_inf_count
v3_phase_final
v3_metric_mix_auc
v3_metric_condition_active
v3_metric_condition_geometry
v3_diagfast_condition
v3_fullgeo_condition
v3_direction_cos_diagfast_fullgeo_mean
v3_direction_norm_ratio_fullgeo_diagfast_mean
trust_clip_rate
precond_solve_time_ms
metric_build_time_ms
```

### 5.5 R1 通过条件

$$
\text{run failures}=0.
$$

$$
\text{all key metrics finite}.
$$

$$
1<\operatorname{cond}(M+\rho I)<10^4.
$$

$$
\text{trust clip rate}<0.05.
$$

如果 R1 失败，不能调模型超参，必须回 R0 修实现。

---

## 6. R2：E1 direction audit

### 6.1 为什么 R2 是必须的

E3 已经显示 full Gram 有信号，但我们还不知道它为什么有信号。可能性至少有四种：

1. full Gram direction 更接近真实 function-space steepest descent；
2. full Gram 只是让 update norm 变大；
3. full Gram 提供了额外 smoothing，类似隐式正则；
4. seed0 偶然。

如果不做 direction audit，后面即使 3-seed 过了，也很难解释结果。R2 的任务是把 update direction 本身测清楚。

### 6.2 审计对象

对同一个 coefficient gradient $g$，比较以下方向：

```text
raw_grad
legacy_grid_index_diag
basis_diag_fast
basis_diag_geo
full_fast
full_geo
diagfast_to_fullgeo_mix_0.25
diagfast_to_fullgeo_mix_0.50
diagfast_to_fullgeo_mix_0.75
current_v3_direction
```

公式上：

$$
d_{raw}=-g.
$$

$$
d_{legacy}=-D_{legacy}^{-1}g.
$$

$$
d_{diag-fast}=-\operatorname{diag}(M_{fast}+\rho I)^{-1}g.
$$

$$
d_{diag-geo}=-\operatorname{diag}(M_{geo}+\rho I)^{-1}g.
$$

$$
d_{full-fast}=-(M_{fast}+\rho I)^{-1}g.
$$

$$
d_{full-geo}=-(M_{geo}+\rho I)^{-1}g.
$$

$$
d_{mix}(\lambda)=(1-\lambda)d_{diag-fast}+\lambda d_{full-geo}.
$$

### 6.3 审计 checkpoint

R2 不应该只在 init 做，因为 init 的 gradient 不能代表训练后期。每个 dataset 至少审计四个状态：

```text
C0_init
C1_after_epoch_1
C2_switch_or_active_end
C3_final
```

如果 runner 暂时不方便保存 checkpoint，可以在训练过程中每个 epoch 后调用 audit，至少记录：

```text
epoch = 0, 1, switch_epoch, final_epoch
```

### 6.4 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2
model = h96 / depth4 / basis24 / alpha1.5
train / val / test = 6000 / 1000 / 1000
batch_size = 256
audit_batch_size = 256
```

R2 不需要跑太多 method。建议只审计：

```text
AdamW checkpoint
GA-FU-v2-tuned checkpoint
V3-full-from-start checkpoint
V3-diag-to-full-hard checkpoint
V3-diag-to-full-smooth-fixed checkpoint
```

### 6.5 必须记录的 direction 指标

#### Cosine

$$
\cos(d_a,d_b)
=
\frac{\langle d_a,d_b\rangle}{\|d_a\|\|d_b\|+\epsilon}.
$$

记录：

```text
cos/raw_fullgeo
cos/legacy_fullgeo
cos/diagfast_fullgeo
cos/diaggeo_fullgeo
cos/fullfast_fullgeo
cos/current_fullgeo
```

#### Norm ratio

$$
R_{norm}(d_a,d_b)
=
\frac{\|d_a\|}{\|d_b\|+\epsilon}.
$$

记录：

```text
norm/fullgeo_over_diagfast
norm/fullgeo_over_legacy
norm/current_over_diagfast
norm/current_over_raw
```

#### Metric norm

对 update $u$，记录：

$$
\|u\|_M=\sqrt{u^T(M+\rho I)u}.
$$

实际 coefficient tensor 是 `[out,in,basis]`，可以 flatten edge 后逐 edge 求和。

记录：

```text
metric_norm/diagfast
metric_norm/fullgeo
metric_norm/current
metric_norm/update_over_coeff
```

#### Predicted descent

对 direction $d$，小步长 $\eta$ 的一阶预测下降为：

$$
\Delta L_{pred}(d)=\langle \nabla_aL, \eta d\rangle.
$$

因为 $d$ 已经是负梯度方向，理想情况：

$$
\Delta L_{pred}<0.
$$

记录：

```text
pred_descent/raw
pred_descent/diagfast
pred_descent/fullgeo
pred_descent/current
```

#### Actual shadow descent

复制当前 coefficient，不真正更新训练模型，用一个小 shadow step：

$$
a'=a+\eta d,
$$

然后在同一 mini-batch 或 audit batch 上计算：

$$
\Delta L_{actual}=L(a')-L(a).
$$

记录：

```text
actual_delta/raw
actual_delta/diagfast
actual_delta/fullgeo
actual_delta/current
pred_actual_ratio
sign_agreement
bad_shadow_step
```

其中：

$$
\operatorname{signAgreement}
=
\mathbb 1[\Delta L_{pred}\cdot\Delta L_{actual}>0].
$$

这里的 sign convention 要统一。如果 predicted 和 actual 都是 loss delta，则两者都应为负。

### 6.6 可视化

R2 至少生成这些图：

```text
1. direction cosine heatmap by dataset / checkpoint / layer
2. fullgeo_over_diagfast norm ratio over epoch
3. predicted vs actual descent scatter
4. Gram condition by alpha/beta/rho
5. layer-wise direction cosine violin plot
6. cosine(diagfast, fullgeo) vs AUC improvement scatter
7. update_metric_norm vs trust_clip_rate scatter
```

### 6.7 R2 判定

如果：

$$
\cos(d_{diag-fast},d_{full-geo})<0.5
$$

且 full Gram 的 actual shadow descent 不稳定，则 full Gram 风险很高，需要调 `rho` 或 `alpha_geo/beta_geo`。

如果：

$$
\cos(d_{diag-fast},d_{full-geo})<0.5
$$

但 full Gram actual descent 更好，说明 true Gram 确实提供了新方向，值得扩 seed。

如果：

$$
\frac{\|d_{full-geo}\|}{\|d_{diag-fast}\|}>3
$$

且 trust clip rate 上升，说明 full Gram 可能只是步长放大，需要优先调 `rho`，而不是继续调 branch scale。

如果：

$$
\cos(d_{diag-fast},d_{full-geo})>0.8
$$

但 AUC 差异很大，说明差异更可能来自 norm / schedule / transition，而不是 direction。

---

# 第二部分：patched seed0 probe

## 7. R3：E3 seed0 rerun after patch

### 7.1 目标

R3 是修复 transition 语义后的 seed0 probe。它不是正式扩 seed，而是回答两个问题：

1. `diag-to-full-smooth-fixed` 的 Fashion negative J reduction 是否消失；
2. `full-from-start` 的 KMNIST strong AUC signal 是否保持。

### 7.2 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0
model = h96 / depth4 / basis24 / alpha1.5
train / val / test = 6000 / 1000 / 1000
Fashion epochs = 30 preferred, 20 only as diagnostic
KMNIST epochs = 20
batch_size = 256
audit_batch_size = 256
```

Fashion 在 R3 就建议直接跑 30 epoch，因为后续 rescue 会用 30 epoch。可以额外保留 20 epoch 复现对照，但不作为 Fashion 决策依据。

### 7.3 方法矩阵

| method | 用途 |
|---|---|
| `AdamW` | baseline |
| `GA-FU-v2-tuned` | 当前最强稳定 baseline |
| `V3-diag-gram` | accuracy / geometry safety candidate |
| `V3-full-from-start` | KMNIST primary candidate, Fashion secondary |
| `V3-diag-to-full-hard` | Fashion primary candidate |
| `V3-diag-to-full-smooth-fixed` | transition 修复后重新评估 |
| `V3-diag-to-full-smooth-fixed-longtr` | 如果 fixed smooth 仍不稳定，测试更长 transition |

`longtr` 建议：

```text
v3_transition_frac = 0.20
v3_transition_min_steps = 100
v3_transition_max_steps = 500
```

### 7.4 必须记录

除了标准指标，还必须记录：

```text
v3_direction_cos_diagfast_fullgeo_mean
v3_direction_norm_ratio_fullgeo_diagfast_mean
v3_transition_mix_auc
v3_transition_length_steps
v3_phase_switch_step
v3_phase_switch_reason
val_loss_before_switch
val_loss_after_switch
transition_loss_jump
phi_before_switch
phi_after_switch
jac_before_switch
jac_after_switch
```

定义 switch loss jump：

$$
\Delta L_{switch}=L_{val}(e_{switch}+1)-L_{val}(e_{switch}).
$$

如果 transition 是 step-level，则也记录最近一次 epoch-level proxy。

### 7.5 R3 决策规则

#### Smooth 是否恢复

如果修复后：

$$
JRed_{smooth-fixed}>0.2
$$

并且：

$$
AUCImprove_{smooth-fixed}\geq AUCImprove_{hard}-0.01,
$$

则 smooth 可以作为 secondary candidate。

如果修复后 Fashion 仍然：

$$
JRed_{smooth-fixed}<0,
$$

则 smooth 从本轮默认路线中移除，只保留为实现诊断。

#### KMNIST full-from-start 是否扩 seed

如果 seed0 仍满足：

$$
AUCImprove_{full-start}>0.08,
$$

$$
AccGap_{full-start}<0.005,
$$

$$
PhiRed_{full-start}>0.25,
$$

则进入 R4 扩 seed。

#### Fashion hard 是否进入 rescue

如果 Fashion `diag-to-full-hard` 仍满足：

$$
AUCImprove_{hard}>AUCImprove_{v2},
$$

且：

$$
JRed_{hard}>0.2,
$$

即使 acc 低，也进入 R5 做 accuracy rescue。

---

# 第三部分：dataset-specific 扩 seed

## 8. R4：KMNIST `full_sobolev_gram from start` 3-seed expansion

### 8.1 目标

R4 只验证 KMNIST。目标不是找所有 v3 组合，而是确认 `V3-full-from-start` 是否能稳定成为 KMNIST default candidate，并冲 medium AUC gate。

KMNIST 当前最关键的 gate 是：

$$
AUCImprove>8\%.
$$

v2 在 3-seed E2 中已经达到 `8.36%`，但 5-seed tuned 记录大约是 `7.00%`。因此 R4 必须用 paired seed 和后续 5-seed 看稳定性，不能只看 seed0。

### 8.2 设置

```text
dataset = KMNIST
seeds = 0,1,2
model = h96 / depth4 / basis24 / alpha1.5
epochs = 20
train / val / test = 6000 / 1000 / 1000
batch_size = 256
eval_batch_size = 512
audit_batch_size = 256
```

### 8.3 方法矩阵

| label | metric | schedule | 目的 |
|---|---|---|---|
| `AdamW` | AdamW | default | baseline |
| `GA-FU-v2-tuned` | legacy grid diag | current tuned | v2 baseline |
| `K-V3-FULL-base` | full Gram all phases | no rest/coeff decay | primary |
| `K-V3-FULL-restcos07` | full Gram all phases | rest cosine final 0.7 | 测 rest/head AUC 协同 |
| `K-V3-FULL-bothcos07` | full Gram all phases | rest cosine 0.7 + coeff cosine 0.7 | 测 mild coordinated schedule |
| `K-V3-FULL-rho3e-3` | full Gram all phases | base | 如果 R2 显示 full norm 太大才跑 |

默认先跑前五个。`rho3e-3` 是 conditional，不要一开始加大实验空间。

### 8.4 推荐配置

`K-V3-FULL-base`：

```text
coeff_lr = 0.14
rest_lr = 0.003
branch_boost = 1.2
coeff_lr_boost = 1.0
branch_max_active_frac = 0.35
branch_final_scale = 0.8
geometry_min_epochs = 0
v3_phase_mode = hard
v3_metric_active = full_sobolev_gram
v3_metric_transition = full_sobolev_gram
v3_metric_geometry = full_sobolev_gram
v3_alpha_geo = 0.15
v3_beta_geo = 0.02
v3_gram_rho = 1e-3
rest_lr_schedule = none
coeff_lr_schedule = none
```

`K-V3-FULL-restcos07`：

```text
rest_lr_schedule = cosine
rest_lr_final_mult = 0.7
coeff_lr_schedule = none
coeff_lr_decay_final_mult = 1.0
```

`K-V3-FULL-bothcos07`：

```text
rest_lr_schedule = cosine
rest_lr_final_mult = 0.7
coeff_lr_schedule = cosine
coeff_lr_decay_final_mult = 0.7
```

### 8.5 必须记录

KMNIST R4 最重要的是同时记录 convergence 和 geometry，不要只看 accuracy。

```text
test_acc
val_loss_auc
val_auc_improvement_vs_adamw
val_auc_improvement_vs_v2
phi_prime_p95
phi_prime_reduction_vs_adamw
jac_condition_max
jac_reduction_vs_adamw
branch_over_adamw
no_kan_acc_drop
ECE
ECE_reduction_vs_adamw
step_time_ms
precond_solve_time_ms
metric_build_time_ms
trust_clip_rate
```

### 8.6 可视化

```text
KMNIST val loss curve, seed mean ± std
KMNIST test acc paired by seed
AUC improvement paired by seed
phi/J reduction paired by seed
branch_over_adamw over epoch
full Gram direction norm ratio over epoch
AUC vs JRed Pareto scatter
AUC vs branch_over_adamw scatter
```

### 8.7 R4 通过条件

Primary candidate 通过 R4，需要：

$$
\operatorname{AccGap}_{v3}<0.005.
$$

$$
\operatorname{AUCImprove}_{v3}>0.08.
$$

$$
\operatorname{PhiRed}_{v3}>0.25.
$$

$$
\operatorname{JRed}_{v3}>0.20.
$$

$$
0.55<\operatorname{branch/AdamW}<0.85.
$$

$$
\operatorname{ECERed}_{v3}>0.10.
$$

如果 AUC 和 accuracy 过了，但 JRed 只有 `0.15-0.20`，则标记为：

```text
KMNIST-v3 convergence candidate, geometry borderline
```

这时不能 claim geometry default，只能作为 AUC candidate 进入 5-seed，同时需要检查 R2 的 norm / rho。

### 8.8 R4 失败诊断

如果 full-from-start accuracy 掉：

```text
优先检查 branch_over_adamw 是否 < 0.55。
如果 branch under-active，则提高 branch_final_scale 到 0.85 或 coeff_lr 到 0.16。
如果 branch 正常但 no-KAN drop 低，则检查 final branch task contribution。
```

如果 AUC 没过 8%：

```text
检查 restcos07 / bothcos07 是否改善 first-half AUC。
如果 schedule 没改善，说明 seed0 AUC 是偶然，v3 full 不进入 default。
```

如果 geometry 不稳：

```text
检查 fullgeo_over_diagfast norm ratio。
如果 norm ratio > 3，优先试 rho=3e-3。
如果 direction cosine 很低且 bad shadow step 多，调低 alpha_geo/beta_geo。
```

---

## 9. R5：Fashion 30-epoch true-Gram accuracy rescue 3-seed expansion

### 9.1 目标

Fashion 的目标不同。Fashion true Gram 已经显示 AUC signal，但 accuracy 有掉点。R5 的核心不是证明 full Gram 能不能降低 loss，而是证明它能否在 30 epoch + mild branch/coeff rescue 下恢复 final accuracy。

Fashion 的假设：

$$
\boxed{
\text{diag-to-full-hard 改善 trajectory，但 late expression 需要轻微增强。}
}
$$

### 9.2 设置

```text
dataset = Fashion-MNIST
seeds = 0,1,2
model = h96 / depth4 / basis24 / alpha1.5
epochs = 30
train / val / test = 6000 / 1000 / 1000
batch_size = 256
eval_batch_size = 512
audit_batch_size = 256
```

### 9.3 方法矩阵

Fashion 不做全 factorial，只跑窄 rescue matrix。

| label | metric | branch_final | coeff_lr | coeff decay final | 目的 |
|---|---|---:|---:|---:|---|
| `AdamW` | AdamW | - | - | - | baseline |
| `GA-FU-v2-tuned-bothcos` | legacy grid | 0.80 | 0.05 | 0.50 | v2 tuned baseline |
| `F-V3-HARD-base30` | diag-fast -> full-geo hard | 0.80 | 0.05 | 0.50 | 复核 E3 hard |
| `F-V3-HARD-f085` | diag-fast -> full-geo hard | 0.85 | 0.05 | 0.50 | 最小 branch rescue |
| `F-V3-HARD-f090` | diag-fast -> full-geo hard | 0.90 | 0.05 | 0.50 | 更强 branch rescue |
| `F-V3-HARD-c006-softcoeff` | diag-fast -> full-geo hard | 0.85 | 0.06 | 0.70 | coeff rescue |
| `F-V3-FULL-f085` | full Gram from start | 0.85 | 0.05 | 0.70 | secondary full candidate |
| `F-V3-DIAG-safety` | basis diag Gram | 0.80 | 0.05 | 0.50 | accuracy/geometry safety |

如果资源不足，第一轮可以只跑：

```text
AdamW
GA-FU-v2-tuned-bothcos
F-V3-HARD-base30
F-V3-HARD-f085
F-V3-HARD-c006-softcoeff
F-V3-FULL-f085
```

### 9.4 Fashion schedule 固定

Fashion 全部 v3 rescue 默认使用 bothcos，因为之前 Fashion weak pass 的关键就是 rest/head 和 coefficient schedule 协同。

```text
rest_lr_schedule = cosine
rest_lr_final_mult = 0.3
coeff_lr_schedule = cosine
coeff_lr_decay_final_mult = 0.5 or 0.7
lr_decay_start_frac = 0.0
```

这里不再测试 `none` / `restcos` / `coeffcos` 的完整矩阵。那是上一版 E4 的任务；现在 R5 是 rescue，不是重新做 schedule ablation。

### 9.5 推荐配置

`F-V3-HARD-base30`：

```text
coeff_lr = 0.05
rest_lr = 0.001
branch_boost = 1.2
coeff_lr_boost = 1.0
branch_max_active_frac = 0.35
branch_final_scale = 0.8
geometry_min_epochs = 0
v3_phase_mode = hard
v3_metric_active = basis_diag_gram
v3_metric_transition = full_sobolev_gram
v3_metric_geometry = full_sobolev_gram
v3_alpha_fast = 0.0
v3_beta_fast = 0.0
v3_alpha_geo = 0.15
v3_beta_geo = 0.02
v3_gram_rho = 1e-3
rest_lr_schedule = cosine
rest_lr_final_mult = 0.3
coeff_lr_schedule = cosine
coeff_lr_decay_final_mult = 0.5
```

`F-V3-HARD-f085` 只改：

```text
branch_final_scale = 0.85
```

`F-V3-HARD-f090` 只改：

```text
branch_final_scale = 0.90
```

`F-V3-HARD-c006-softcoeff`：

```text
branch_final_scale = 0.85
coeff_lr = 0.06
coeff_lr_decay_final_mult = 0.7
```

`F-V3-FULL-f085`：

```text
v3_metric_active = full_sobolev_gram
v3_metric_transition = full_sobolev_gram
v3_metric_geometry = full_sobolev_gram
v3_phase_mode = hard
branch_final_scale = 0.85
coeff_lr_decay_final_mult = 0.7
```

### 9.6 必须记录

Fashion R5 特别要记录 final expression 是否恢复。

```text
test_acc
train_acc
val_loss_auc
val_loss_auc_first_half
val_loss_auc_second_half
best_val_acc
best_epoch
branch_over_adamw_final
branch_over_adamw_train_mean
no_kan_acc_drop_final
no_kan_drop_over_adamw
kan_logit_delta_norm
kan_margin_contribution
phi_prime_p95
jac_condition_max
curvature_energy
ECE
trust_clip_rate
precond_solve_time_ms
step_time_ms
```

### 9.7 可视化

```text
Fashion val loss curve, 30 epochs
Fashion test acc paired by seed
AUC improvement vs acc gap scatter
branch_final_scale vs no-KAN drop bar
branch_over_adamw vs test_acc scatter
branch_over_adamw vs phi_prime_p95 scatter
F-HARD variants Pareto: x=AUC, y=test_acc, size=JRed
first-half AUC vs second-half AUC stacked bar
```

### 9.8 R5 通过条件

Fashion candidate 通过 R5，需要：

$$
\operatorname{Acc}_{v3}\geq\operatorname{Acc}_{AdamW}.
$$

或者至少：

$$
\operatorname{Acc}_{v3}\geq\operatorname{Acc}_{v2}-0.003.
$$

同时：

$$
\operatorname{AUCImprove}_{v3}>\operatorname{AUCImprove}_{v2}+0.02
$$

或：

$$
\operatorname{AUCImprove}_{v3}>0.08.
$$

并且：

$$
\operatorname{PhiRed}_{v3}>0.25,
$$

$$
\operatorname{JRed}_{v3}>0.20,
$$

$$
0.55<\operatorname{branch/AdamW}<0.85.
$$

如果 Fashion candidate 只有 AUC 强但 accuracy 仍低，则不能作为 default，只能标记为：

```text
Fashion true-Gram convergence profile, not clean default
```

### 9.9 R5 失败诊断

如果 accuracy 低但 AUC 好：

```text
检查 branch_over_adamw_final 与 no_kan_drop_final。
如果 branch_over_adamw < 0.55，继续 f0.90 或 coeff_lr 0.06。
如果 branch_over_adamw 正常但 no_kan_drop 低，说明 branch norm 没转化为 task contribution。
如果 train_acc 也低，说明 underfit，需要 coeff schedule 更少 decay。
如果 train_acc 高但 test_acc 低，说明 true Gram rescue 可能伤泛化，不能继续加 branch。
```

如果 geometry 掉：

```text
检查 branch_final_scale 是否过大。
如果 f0.90 导致 phi/J 接近 AdamW，则 f0.90 不进入候选。
如果 c0.06 导致 J spike，回到 c0.05/f0.85。
```

如果 AUC 没有超过 v2：

```text
说明 Fashion true Gram 的 seed0 AUC signal 不稳定。
此时保留 v2 as Fashion default，v3 只在 KMNIST 继续。
```

---

## 10. R6：dataset-specific default selection

R6 的目标是从 R4 / R5 中各选一个 default candidate，然后做一张干净的 seed012 paired comparison。

### 10.1 候选选择规则

KMNIST 选择规则：

1. 先看 AUC improvement；
2. 再看 accuracy gap；
3. 再看 PhiRed / JRed；
4. 如果多个候选都过 gate，选 step-time 更低的。

Fashion 选择规则：

1. 先看 accuracy 是否恢复；
2. 再看 AUC 是否超过 v2；
3. 再看 no-KAN drop 是否说明 branch 有任务贡献；
4. 如果 true Gram 候选没恢复 accuracy，Fashion default 保留 v2。

### 10.2 R6 方法

```text
AdamW
GA-FU-v2-tuned
V3-KMNIST-selected
V3-Fashion-selected
```

注意：`V3-KMNIST-selected` 和 `V3-Fashion-selected` 可以是不同 config。报告时不要强行叫同一个 universal default，而是：

$$
\boxed{
\text{v3 candidate can be dataset-specific until 5-seed confirms otherwise.}
}
$$

### 10.3 关键表格

R6 需要输出 paired seed table：

| dataset | seed | AdamW acc | v2 acc | v3 acc | v3-v2 acc | v3 AUC improve | v3-v2 AUC | v3 PhiRed | v3 JRed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

以及 summary：

| dataset | selected v3 | acc mean | acc gap vs AdamW | AUC improve | PhiRed | JRed | branch/AdamW | ECE red | pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|

### 10.4 R6 决策

如果两个 dataset 都有 v3 selected candidate 通过 weak / medium：进入 R7。

如果 KMNIST 过而 Fashion 不过：

```text
KMNIST v3 true-Gram default candidate confirmed for expansion.
Fashion remains v2 tuned default.
```

如果 Fashion 过而 KMNIST 不过：

```text
Fashion v3 true-Gram rescue candidate confirmed.
KMNIST remains v2 tuned default.
```

如果两个都不过：

```text
v3 true-Gram has diagnostic value, but v2 remains optimizer default.
```

---

# 第四部分：正式确认与机制解释

## 11. R7：5-seed / 10-seed clean confirm

### 11.1 目标

R7 才是正式确认。R7 不能再搜索，只能比较固定候选。

### 11.2 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2,3,4
optional final = 0,1,2,3,4,5,6,7,8,9
model = h96 / depth4 / basis24 / alpha1.5
Fashion epochs = 30
KMNIST epochs = 20
train / val / test = 6000 / 1000 / 1000
```

### 11.3 方法

```text
AdamW
GA-FU-v2-tuned
GA-FU-v3-selected
```

如果 Fashion 和 KMNIST selected config 不同，在同一个 runner 中要通过 dataset-specific override 实现，不要手动改命令导致记录不一致。

### 11.4 统计要求

所有关键指标报告：

$$
\mu\pm\sigma.
$$

并记录 paired seed difference：

$$
\Delta_s=Metric_{v3,s}-Metric_{v2,s}.
$$

报告：

$$
\bar\Delta,
\quad
\operatorname{std}(\Delta),
\quad
CI_{95\%}(\Delta).
$$

Bootstrap CI 建议使用 seed-level bootstrap，重复 `10000` 次即可。

### 11.5 R7 pass levels

#### Weak pass

两个 dataset 都满足：

$$
\text{acc gap vs AdamW}<1.0\%,
$$

$$
\text{AUC improvement}>6\%,
$$

$$
\phi'\text{ reduction}>20\%,
$$

$$
J\text{ reduction}>15\%,
$$

$$
0.45<\text{branch/AdamW}<0.95.
$$

#### Medium pass

Fashion：

$$
\operatorname{Acc}_{v3}\geq\operatorname{Acc}_{AdamW}.
$$

KMNIST：

$$
\operatorname{AccGap}_{v3}<0.5\%.
$$

两个 dataset：

$$
\text{AUC improvement}>8\%,
$$

$$
\phi'\text{ reduction}>25\%,
$$

$$
J\text{ reduction}>20\%.
$$

#### Strong pass

两个 dataset：

$$
\text{acc gap vs AdamW}<0.3\%,
$$

$$
\text{AUC improvement}>10\%,
$$

$$
\phi'\text{ reduction}>30\%,
$$

$$
J\text{ reduction}>30\%,
$$

$$
\text{ECE reduction}>10\%.
$$

同时：

$$
\text{time-to-target}_{v3}\leq1.1\times\text{time-to-target}_{AdamW}.
$$

---

## 12. R8：target-matched convergence and wall-clock accounting

### 12.1 目标

AUC 更好不等于真正更快。R8 要决定能否写 “fast convergence”。

### 12.2 Target 定义

对每个 dataset 和 seed，AdamW final val loss：

$$
L^*_{A,s}=L^{final}_{val,A,s}.
$$

AdamW final val acc：

$$
A^*_{A,s}=A^{final}_{val,A,s}.
$$

定义 relaxed loss target：

$$
L^{relaxed}_s=1.05L^*_{A,s}.
$$

定义 relaxed acc target：

$$
A^{relaxed}_s=A^*_{A,s}-0.005.
$$

记录每个 method 首次达到 target 的 step 和 wall-clock time。

### 12.3 指标

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
compute/step_time_ms
compute/precond_solve_time_ms
compute/metric_build_time_ms
compute/time_ratio_vs_adamw
memory/peak_allocated_mb
```

### 12.4 可视化

```text
Kaplan-style target reach curve
steps_to_target paired bar
time_to_target paired bar
AUC vs time_to_target scatter
step_time_ms by method
precond_solve_time stacked with train step time
```

### 12.5 判断

只有当：

$$
\operatorname{time\_to\_target}_{v3}<\operatorname{time\_to\_target}_{AdamW}
$$

或至少：

$$
\operatorname{steps\_to\_target}_{v3}<\operatorname{steps\_to\_target}_{AdamW},
$$

且：

$$
\operatorname{step\_time\_overhead}<1.2,
$$

才可以写 fast convergence。

如果 v3 只有 AUC 更好但 target time 不好，结论写成：

```text
v3 improves validation-loss trajectory and geometry, but does not yet establish wall-clock faster convergence.
```

---

## 13. R9：mechanism audit

### 13.1 目标

R9 解释 v3 selected 是否真的满足你的三目标：

$$
\boxed{
\text{高表达力} + \text{快速收敛} + \text{好几何}
}
$$

这里的关键是不能只看 final accuracy。必须证明 KAN branch 真正在做任务，同时 geometry 没坏。

### 13.2 表达力指标

```text
branch/final_output_norm_ratio
branch/layer_k_final_output_norm_ratio
branch/branch_over_adamw_final
ablation/no_kan_test_acc
ablation/no_kan_acc_drop
ablation/no_kan_drop_over_adamw
ablation/kan_logit_delta_norm
ablation/kan_margin_contribution
basis/active_basis_fraction
basis/dead_basis_fraction
```

机制成功要求：

$$
\operatorname{branch/AdamW}_{final}>0.55.
$$

$$
\Delta_{noKAN}^{v3}>0.7\Delta_{noKAN}^{AdamW}.
$$

如果 branch ratio 高但 no-KAN drop 低，说明 branch norm 变大了，但任务贡献不足。

### 13.3 快速收敛指标

```text
val_loss_auc
val_loss_auc_first_half
val_loss_auc_second_half
loss_at_10pct_steps
loss_at_25pct_steps
loss_at_50pct_steps
steps_to_target
time_to_target
```

机制成功要求：

$$
AUCImprove_{v3}>AUCImprove_{v2}
$$

或：

$$
AUCImprove_{v3}>8\%.
$$

如果 first-half AUC 好但 second-half 不好，说明 v3 早期方向好但 late fitting 不足。Fashion 可能出现这种情况。

### 13.4 几何指标

```text
kan/phi_prime_p95
kan/phi_prime_max
kan/curvature_energy
kan/sobolev_norm_mean
jacobian/condition_mean
jacobian/condition_max
credit/amplification_p95
credit/noise_gain_p95
```

机制成功要求：

$$
\phi'_{v3}<0.75\phi'_{AdamW}.
$$

$$
\kappa(J)_{v3}<0.8\kappa(J)_{AdamW}.
$$

### 13.5 关键图

```text
branch_over_adamw vs epoch, with phase vertical line
no_kan_drop vs epoch, with phase vertical line
phi_prime_p95 vs epoch, with phase vertical line
jacobian_condition vs epoch, with phase vertical line
branch_ratio vs no_kan_drop scatter
branch_ratio vs phi_prime_p95 scatter
branch_ratio vs test_acc scatter
AUC improvement vs J reduction Pareto plot
```

### 13.6 R9 输出结论模板

R9 最终要给出四种可能之一。

#### 结论 A：v3 成为 clean default

条件：两个 dataset 都过 medium。

```text
GA-FU-v3 selected becomes clean default for Stage-I small vision scaling.
```

#### 结论 B：v3 是 dataset-specific default

条件：KMNIST 过，Fashion 不过，或反过来。

```text
GA-FU-v3 true Gram is adopted for dataset X; GA-FU-v2 remains default for dataset Y.
```

#### 结论 C：v3 是 convergence/geometry profile

条件：AUC/geometry 好，但 accuracy 没恢复。

```text
GA-FU-v3 true Gram is a convergence/geometry profile, not the clean accuracy default.
```

#### 结论 D：v3 不进入 default

条件：3-seed / 5-seed 不稳定。

```text
GA-FU-v3 true Gram provides diagnostic insight, but GA-FU-v2 remains the optimizer default.
```

---

# 第五部分：统一指标与日志规范

## 14. Run-level 必须记录

每个 run 的 summary 必须包含以下字段。

### 14.1 Config

```text
config/dataset
config/seed
config/method
config/hidden_dim
config/depth
config/basis_count
config/alpha_init
config/epochs
config/train_size
config/val_size
config/test_size
config/coeff_lr
config/rest_lr
config/branch_boost
config/branch_final_scale
config/rest_lr_schedule
config/rest_lr_final_mult
config/coeff_lr_schedule
config/coeff_lr_decay_final_mult
config/v3_metric_active
config/v3_metric_transition
config/v3_metric_geometry
config/v3_alpha_fast
config/v3_beta_fast
config/v3_alpha_geo
config/v3_beta_geo
config/v3_gram_rho
```

### 14.2 Task performance

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
compare/test_acc_gap_vs_adamw
compare/test_acc_gap_vs_v2
```

### 14.3 Convergence

```text
conv/train_loss_auc
conv/val_loss_auc
conv/val_acc_auc
conv/val_loss_auc_first_half
conv/val_loss_auc_second_half
conv/val_auc_improvement_vs_adamw
conv/val_auc_improvement_vs_v2
conv/loss_at_10pct_steps
conv/loss_at_25pct_steps
conv/loss_at_50pct_steps
conv/loss_at_75pct_steps
```

### 14.4 Geometry

```text
kan/phi_prime_p95
kan/phi_prime_max
kan/curvature_energy
kan/sobolev_norm_mean
jacobian/condition_mean
jacobian/condition_max
credit/amplification_p95
compare/phi_prime_reduction_vs_adamw
compare/jac_reduction_vs_adamw
```

### 14.5 Branch and expression

```text
branch/global_mean_output_norm_ratio
branch/final_output_norm_ratio
branch/branch_over_adamw
branch/layer_0_output_norm_ratio
branch/layer_1_output_norm_ratio
branch/layer_2_output_norm_ratio
branch/layer_3_output_norm_ratio
ablation/no_kan_test_acc
ablation/no_kan_acc_drop
ablation/no_kan_drop_over_adamw
ablation/kan_logit_delta_norm
ablation/kan_margin_contribution
basis/active_basis_fraction
basis/dead_basis_fraction
```

### 14.6 v3 metric and transition

```text
v3/phase_final
v3/phase_switch_step
v3/phase_switch_epoch
v3/phase_switch_reason
v3/transition_length_steps
v3/metric_mix_auc
v3/diagfast_condition
v3/fullgeo_condition
v3/diagfast_eig_min
v3/fullgeo_eig_min
v3/diagfast_eig_max
v3/fullgeo_eig_max
v3/direction_cos_diagfast_fullgeo_mean
v3/direction_norm_ratio_fullgeo_diagfast_mean
v3/update_metric_norm_mean
v3/update_metric_norm_p95
v3/update_over_coeff_norm_mean
```

### 14.7 Trust and compute

```text
trust/mode
trust/radius
trust/clip_count
trust/clip_rate
compute/step_time_ms
compute/epoch_time_sec
compute/total_train_time_sec
compute/precond_solve_time_ms
compute/metric_build_time_ms
compute/throughput_samples_per_sec
memory/peak_allocated_mb
```

---

## 15. Curve-level 必须记录

每个 run 建议输出：

```text
runs/{run_id}/curves.csv
```

字段：

```text
epoch
step
phase_id
train_loss
val_loss
val_acc
test_acc_optional
branch_ratio_train_mean
branch_ratio_final_eval
branch_scale
coeff_lr_effective
rest_lr_effective
metric_mix
phi_prime_p95
jac_condition
curvature_energy
ece
no_kan_acc_drop
trust_clip_rate_epoch
```

这份 curve 文件很重要，因为 AUC / switch / geometry recovery 都需要 epoch-level 或 step-level 解释。

---

## 16. Direction audit 文件

R2 单独输出：

```text
direction_audit.csv
```

字段：

```text
dataset
seed
method
checkpoint_tag
epoch
layer
metric_a
metric_b
cosine
norm_ratio
metric_norm_a
metric_norm_b
pred_delta_a
actual_delta_a
pred_actual_ratio_a
sign_agreement_a
bad_shadow_step_a
gram_condition_fast
gram_condition_geo
rho
alpha_fast
beta_fast
alpha_geo
beta_geo
```

并输出聚合：

```text
direction_audit_summary_by_dataset.csv
direction_audit_summary_by_layer.csv
direction_audit_summary_by_checkpoint.csv
```

---

# 第六部分：可视化 dashboard 规范

## 17. Scorecard dashboard

第一张图不是曲线，而是一张 scorecard 表。

列：

```text
dataset
method
runs
test_acc_mean
test_acc_std
acc_gap_vs_adamw
val_auc_improvement
phi_prime_reduction
jac_reduction
branch_over_adamw
no_kan_drop_over_adamw
ECE_reduction
step_time_ratio
pass_level
```

这张表用于快速判断 v3 是否比 v2 更值得继续。

## 18. Loss trajectory dashboard

图：

```text
val_loss vs epoch, mean ± std
val_acc vs epoch, mean ± std
first-half AUC bar
second-half AUC bar
loss_at_10/25/50/75pct bar
```

Fashion 特别要看 first-half 和 second-half。如果 true Gram early AUC 好但 late acc 不够，second-half 会暴露出来。

## 19. Direction audit dashboard

图：

```text
cos(diagfast, fullgeo) heatmap
norm(fullgeo) / norm(diagfast) over epoch
predicted vs actual descent scatter
bad_shadow_step rate bar
condition number by metric
```

这张 dashboard 是解释 v3 的关键。如果没有 direction audit 图，v3 的结果只能算经验调参。

## 20. Branch and expression dashboard

图：

```text
branch_over_adamw over epoch
branch ratio per layer final bar
no_kan_drop bar
kan_logit_delta_norm bar
kan_margin_contribution bar
branch_over_adamw vs no_kan_drop scatter
branch_over_adamw vs test_acc scatter
```

这张 dashboard 回答：KAN branch 是否真的有用。

## 21. Geometry safety dashboard

图：

```text
phi_prime_p95 over epoch
jacobian_condition over epoch
curvature_energy over epoch
credit_amplification_p95 over epoch
AUC improvement vs J reduction scatter
branch_over_adamw vs phi_prime_p95 scatter
```

这张 dashboard 回答：v3 是否牺牲了几何。

## 22. Transition dashboard

图：

```text
phase_id over epoch
metric_mix over step / epoch
branch_scale over epoch
coeff_lr_effective over epoch
rest_lr_effective over epoch
switch_step histogram
switch_reason counts
transition_loss_jump bar
```

如果 transition 修复后 smooth 仍失败，这张图会说明失败发生在 switch 前、transition 内还是 geometry phase。

## 23. Compute dashboard

图：

```text
step_time_ms by method
precond_solve_time_ms by method
metric_build_time_ms by method
memory_peak_allocated_mb by method
time_to_target by method
AUC improvement vs step_time_ratio scatter
```

v3 true Gram 如果 step time 太高，就算 AUC 好也不能直接 claim fast convergence。

---

# 第七部分：命令模板与输出目录

## 24. R0 / R1

```bash
python -m py_compile experiments/dgkan_core.py experiments/run_gafu_v3.py
```

```bash
python experiments/run_gafu_v3.py \
  --packages V3_E0 \
  --datasets Fashion-MNIST,KMNIST \
  --seeds 0 \
  --out-dir results/gafu_v3_r1_e0_after_transition_fix \
  --fresh --device auto --no-download \
  --epochs 1 --train-size 512 --val-size 128 --test-size 128 \
  --hidden-dim 32 --depth 2 --basis-count 8 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 64 \
  --continue-on-error --wandb --wandb-project DG-KAN
```

## 25. R2 direction audit

建议新增 package：

```text
V3_R2_DIRECTION_AUDIT
```

命令：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_R2_DIRECTION_AUDIT \
  --datasets Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --out-dir results/gafu_v3_r2_direction_audit \
  --fresh --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error --wandb --wandb-project DG-KAN
```

如果 runner 暂时不支持训练中 direction audit，可以先新增一个独立脚本：

```text
experiments/audit_gafu_v3_directions.py
```

它读取 checkpoint / config，输出 `direction_audit.csv`。

## 26. R3 seed0 probe

```bash
python experiments/run_gafu_v3.py \
  --packages V3_R3_E3_FIXED \
  --datasets Fashion-MNIST,KMNIST \
  --seeds 0 \
  --out-dir results/gafu_v3_r3_e3_fixed_seed0 \
  --fresh --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error --wandb --wandb-project DG-KAN
```

Fashion 30-epoch probe 可单独跑：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_R3_E3_FIXED \
  --datasets Fashion-MNIST \
  --seeds 0 \
  --out-dir results/gafu_v3_r3_fashion_e3_fixed_seed0_e30 \
  --fresh --device auto --no-download \
  --epochs 30 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error --wandb --wandb-project DG-KAN
```

## 27. R4 KMNIST expansion

建议新增 package：

```text
V3_R4_KMNIST_FULL_EXPAND
```

命令：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_R4_KMNIST_FULL_EXPAND \
  --datasets KMNIST \
  --seeds 0,1,2 \
  --out-dir results/gafu_v3_r4_kmnist_full_expand_s012 \
  --fresh --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error --wandb --wandb-project DG-KAN
```

## 28. R5 Fashion rescue

建议新增 package：

```text
V3_R5_FASHION_RESCUE
```

命令：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_R5_FASHION_RESCUE \
  --datasets Fashion-MNIST \
  --seeds 0,1,2 \
  --out-dir results/gafu_v3_r5_fashion_rescue_s012_e30 \
  --fresh --device auto --no-download \
  --epochs 30 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error --wandb --wandb-project DG-KAN
```

## 29. R7 clean confirm

5-seed：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_R7_CLEAN_CONFIRM \
  --datasets Fashion-MNIST,KMNIST \
  --seeds 0,1,2,3,4 \
  --out-dir results/gafu_v3_r7_clean_confirm_s01234 \
  --fresh --device auto --no-download \
  --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error --wandb --wandb-project DG-KAN
```

10-seed：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_R7_CLEAN_CONFIRM \
  --datasets Fashion-MNIST,KMNIST \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --out-dir results/gafu_v3_r7_clean_confirm_s0to9 \
  --fresh --device auto --no-download \
  --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error --wandb --wandb-project DG-KAN
```

注意：Fashion 和 KMNIST epochs 不同。runner 需要支持 dataset-specific epochs override，否则建议分两个命令跑：Fashion `epochs=30`，KMNIST `epochs=20`。

---

# 第八部分：failure table 与停止规则

## 30. Failure tags

每个 run 自动打标签。

| failure type | 触发条件 | 解释 |
|---|---|---|
| `transition_semantics_failed` | endpoint check 不通过 | transition 实现仍错 |
| `direction_bad_shadow_descent` | full/current bad shadow step rate 高 | metric direction 不可靠 |
| `fullgeo_norm_explosion` | norm ratio > 3 且 clip rate 高 | full Gram 步长放大 |
| `accuracy_gap_too_large` | acc gap 超阈值 | final fitting 不够 |
| `no_convergence_gain` | AUC improvement <= 0 | 没有收敛优势 |
| `branch_underactive` | branch/AdamW < 0.55 | 表达力没释放 |
| `branch_overactive` | branch/AdamW > 0.95 | branch 过强 |
| `no_kan_no_contribution` | no-KAN drop 低于 AdamW 0.7x | branch 无任务贡献 |
| `geometry_not_preserved` | phi/J reduction 不达标 | 几何没守住 |
| `jacobian_spike` | J condition 高于 AdamW 或 JRed < 0 | Jacobian 失控 |
| `phi_prime_spike` | phi 高于 AdamW | edge derivative 失控 |
| `trust_not_safety_only` | clip rate > 0.05 | trust 变成优化组件 |
| `wall_clock_no_gain` | time-to-target 不好 | 不能 claim fast |
| `seed_unstable` | acc std > 0.8% | seed 不稳定 |

## 31. 停止规则

### 31.1 transition 没修好

如果 R0 endpoint check 失败，不跑 R1-R9。

### 31.2 full Gram direction 不可靠

如果 R2 中 fullgeo bad shadow step rate 明显高，且 R3 seed0 geometry 失控，则不扩 full-from-start。先调：

```text
rho = 3e-3
alpha_geo = 0.10 or 0.05
beta_geo = 0.01 or 0.00
```

但这只允许在 R2/R3 阶段做，不进入 R4/R5 大扫。

### 31.3 KMNIST full-from-start 不稳定

如果 R4 中 `K-V3-FULL-base/restcos/bothcos` 都没有过 AUC `8%`，则 KMNIST v3 true Gram 不进入 default。KMNIST 保留 v2 tuned。

### 31.4 Fashion rescue 失败

如果 R5 中所有 Fashion true Gram candidate 都 accuracy 低于 v2 超过 `0.3%`，则 Fashion 保留 v2 tuned。true Gram 只作为 convergence profile，不作为 clean default。

### 31.5 5-seed 退化

如果 R7 5-seed 中 3-seed 结论消失，不能继续跑 10-seed试图“赌回来”。先回到 paired seed table 和 failure table 解释退化。

---

# 第九部分：最终报告结构

## 32. 最终报告应该怎么写

R7-R9 完成后，最终报告建议按下面结构写。

### 32.1 Main result

先给 scorecard，不先给曲线。

```text
Table 1: AdamW vs GA-FU-v2 vs GA-FU-v3-selected
```

列：

```text
dataset
method
test acc
acc gap
AUC improvement
PhiRed
JRed
branch/AdamW
no-KAN drop
ECE reduction
step-time ratio
pass level
```

### 32.2 Why v3 changed default

解释：

```text
Phase alignment alone did not solve AUC.
True Gram direction produced the real signal.
KMNIST selected full-from-start.
Fashion selected hard transition + rescue, or retained v2.
```

### 32.3 Direction audit

给 direction heatmap 和 predicted/actual descent，说明 full Gram 的作用不是玄学。

### 32.4 Mechanism

给 branch/no-KAN/geometry 曲线，说明表达力、收敛、几何三目标如何同时满足或哪里没满足。

### 32.5 Decision

只允许四种结论：

```text
v3 clean default
v3 dataset-specific default
v3 convergence/geometry profile
v2 remains default, v3 diagnostic only
```

---

## 33. 这轮计划的核心判断

这轮实验不应该再扩成大网格。现在已经有足够多的信息说明：

$$
\boxed{
\text{v3 的价值主要来自 true Gram direction，而不是 phase alignment alone。}
}
$$

但 true Gram 是否能成为 default，还要解决两个具体问题：

$$
\boxed{
\text{KMNIST: full-from-start 的 AUC/accuracy/geometry 是否能 3-seed 稳定。}
}
$$

$$
\boxed{
\text{Fashion: diag-to-full-hard 的 AUC 信号能否通过 30 epoch rescue 恢复 accuracy。}
}
$$

所以本轮最重要的不是继续发明新 optimizer，而是把当前两个问题做干净。最终如果得到的结论是 dataset-specific，也完全可以接受。当前阶段不需要强行把 Fashion 和 KMNIST 压成同一个 v3 default。

真正的成功标准是：

$$
\boxed{
\text{表达力由 branch/no-KAN 保证，收敛由 AUC/target 保证，几何由 }\phi'\text{ 和 }J\text{ 保证。}
}
$$

只要这三个同时站住，v3 才能进入 Stage-I scaling；否则 v2 仍然是 clean default，v3 作为 true-Gram diagnostic profile 保留。
