# DG-KAN Optimizer v3.3 实验计划：统一 Full-Start Profile 与统一 Optimizer 设定

> 版本：v3.3 draft  
> 日期：2026-05-02  
> 目标：在 v3.2 结果基础上，验证 `full_sobolev_gram from start + alphaFixed1` 是否能成为跨 Fashion-MNIST / KMNIST 的统一 profile，并额外评估“非 KAN 参数与 KAN 参数使用统一 optimizer 动力学”的必要性与风险。  
> 公式格式：Typora 友好，全文公式只使用 `$...$` 与 `$$...$$`。

---

## 0. 当前判断与下一轮核心问题

v3.2 已经给出了一个很清楚的局面。Fashion-MNIST 上，`F-V3-HARD-base30-final-alphaFixed1` 已经通过 10-seed confirm，可以写成 clean default。它同时提高 accuracy、validation-loss AUC、几何、calibration，并且通过 no-KAN expression audit。

KMNIST 上，`K-V3-FULL-base-final-alphaFixed1` 几乎打平 AdamW accuracy，同时改善 AUC、几何和 ECE，但 no-KAN ratio 只有约 $0.66$，低于 expression threshold $0.7$。这说明它不是失败，而是一个 geometry / convergence Pareto candidate。它的关键短板不是几何，而是 KAN branch 的任务表达力释放还差一点。

因此 v3.3 的核心问题不是“继续发明 optimizer 组件”，而是：

$$
\boxed{
\text{full-start fixed1 能否通过轻量 expression rescue 成为统一 profile？}
}
$$

同时，用户提出一个新的重要设定：

$$
\boxed{
\text{能否把非 KAN 参数与 KAN 参数纳入统一 optimizer 动力学？}
}
$$

这个设定值得加入，但不能直接替换主线。原因是过去实验已经说明，KAN coefficients 适合 functional update，非 KAN 参数适合 AdamW。如果粗暴把所有参数都用同一种手写 functional/SGD update，容易欠拟合。因此 v3.3 会把“统一 optimizer”分成几个层级，从最安全到最激进逐步验证。

---

## 1. 本轮实验要回答的两个问题

### 1.1 问题 A：统一 profile 是否可行

当前 profile 仍然是 dataset-specific：

```text
Fashion-MNIST:
  hard diag-to-full true Gram
  alphaFixed1
  30 epochs

KMNIST:
  full_sobolev_gram from start
  alphaFixed1
  20 epochs
```

v3.3 希望优先验证一个更统一的 profile：

```text
U-FULL:
  alpha_mode = fixed1
  metric_active = full_sobolev_gram
  metric_transition = full_sobolev_gram
  metric_geometry = full_sobolev_gram
  phase_mode = hard or none
  no transition needed
```

直观上，这个 profile 的优点是非常清楚：它最简洁、最稳定、几何最好，也最适合写成统一方法。但它可能偏保守，尤其在 KMNIST 上可能压住 branch 表达力。

所以问题 A 的判定不是只看 accuracy，而是同时看：

$$
\operatorname{Acc}_{test},
\quad
\operatorname{AUCImprove}_{val},
\quad
\phi'_{p95}\text{ reduction},
\quad
\kappa(J)\text{ reduction},
\quad
\operatorname{ECE}\text{ reduction},
\quad
\frac{\Delta_{noKAN}^{method}}{\Delta_{noKAN}^{AdamW}}.
$$

其中最关键的新 gate 是：

$$
\frac{\Delta_{noKAN}^{method}}{\Delta_{noKAN}^{AdamW}}>0.7.
$$

### 1.2 问题 B：统一 optimizer 动力学是否可行

当前默认训练是 hybrid：

$$
\boxed{
\text{KAN coefficients: functional update}
+
\text{non-KAN parameters: AdamW}
}
$$

这个 hybrid 不是随便来的，而是之前分类实验反复证明：非 KAN 参数如果不用 AdamW，容易欠拟合；KAN coefficients 如果只用普通 AdamW，则缺少函数空间几何控制。

但从方法美观和工程实现看，确实希望进一步探索一个统一 optimizer 设定。这里的“统一”可以有三种不同强度：

```text
Level 1: 统一 controller / schedule，但 KAN 与 non-KAN 仍使用不同 update geometry。
Level 2: 单一 AdamW optimizer 动力学，KAN gradient 先做 functional preconditioning。
Level 3: 所有参数都使用同一种手写 functional-style update。
```

Level 3 风险最大，过去已有欠拟合信号，因此只作为负对照或极小 smoke。v3.3 主要研究 Level 2：

$$
\boxed{
\text{AdamW on all parameters, but KAN gradients are transformed by Sobolev preconditioner before AdamW moments.}
}
$$

也就是对 KAN coefficient gradient 做：

$$
\tilde g_a=P_{Sob}g_a,
$$

然后把 $\tilde g_a$ 交给 AdamW；非 KAN 参数则使用原始梯度 $g_\theta$。

这样训练表面上是一个统一 AdamW optimizer，只是 KAN 参数的梯度进入 AdamW 之前被函数空间 metric 改写。

---

## 2. v3.3 的主假设

本轮有三个预注册假设。

### 2.1 H1：full-start fixed1 可以成为 unified Pareto profile

如果 `full_sobolev_gram from start + alphaFixed1` 在 Fashion 和 KMNIST 上都满足：

$$
\text{acc gap vs AdamW}<0.5\%,
$$

$$
\text{AUC improvement}>8\%,
$$

$$
\phi'\text{ reduction}>25\%,
$$

$$
J\text{ reduction}>20\%,
$$

$$
\text{ECE reduction}>10\%,
$$

并且 no-KAN ratio 超过 $0.7$，则它可以升级为 unified clean profile。

如果它只在 KMNIST 上是 Pareto，而 Fashion 明显弱于 hard profile，则它只能称为 unified Pareto profile，而不是 clean default。

### 2.2 H2：KMNIST 当前短板是 expression 不足，而不是 geometry 不足

KMNIST v3.2 的 full-start fixed1 已经有很好的 AUC、geometry、ECE，但 no-KAN ratio 不足。因此 v3.3 的 expression rescue 不是为了继续降低 $\phi'$ 或 Jacobian，而是为了提高：

$$
\Delta_{noKAN},
\quad
\operatorname{branch/AdamW},
\quad
\operatorname{KAN\ margin\ contribution}.
$$

目标是把 KMNIST 的 no-KAN ratio 从约 $0.66$ 推到 $0.70+$，同时保持 AUC 和几何优势。

### 2.3 H3：统一 AdamW 动力学可能提升工程统一性，但有 moment 放大风险

如果对 KAN gradient 做 Sobolev preconditioning 后再交给 AdamW，AdamW 的 moment 可能积累被 preconditioned direction 放大的方向，导致 branch over-active 或几何 spike。因此统一 optimizer 设定必须记录 moment 方向、norm amplification、trust clipping 和 bad step。

成功的 unified optimizer 不能只是“能跑”，必须满足：

$$
\text{acc 不明显低于 current hybrid},
$$

$$
\text{AUC 不明显低于 current hybrid},
$$

$$
\phi'/J\text{ 不显著恶化},
$$

$$
\text{branch expression 不低于 current hybrid},
$$

$$
\text{moment amplification 可控}.
$$

---

## 3. 方法定义

### 3.1 Baseline A：AdamW-alphaFixed1

这是全参数 AdamW baseline。

```text
alpha_mode = fixed1
KAN coeff = AdamW
non-KAN params = AdamW
metric = none
```

它用于定义 accuracy、AUC、branch、no-KAN、ECE 和 geometry 的比较基准。

### 3.2 Baseline B：GA-FU-v2-alphaFixed1

这是 v2 对照。它用于判断 true-Gram v3 是否真正优于旧 GA-FU。

```text
alpha_mode = fixed1
KAN coeff = v2 functional update
non-KAN params = AdamW
metric = legacy diagonal / grid style
```

### 3.3 Baseline C：v3.2 dataset-specific best

这一组不是 unified 候选，而是当前 best upper bound。

```text
Fashion:
  F-V3-HARD-base30-final-alphaFixed1

KMNIST:
  K-V3-FULL-base-final-alphaFixed1
```

它回答：统一 full-start profile 与当前 dataset-specific profile 的差距有多大。

### 3.4 Candidate U-FULL：统一 full Sobolev from start

这是 v3.3 的主候选。

```text
alpha_mode = fixed1
metric_active = full_sobolev_gram
metric_transition = full_sobolev_gram
metric_geometry = full_sobolev_gram
v3_phase_mode = hard or none
branch schedule = no active diag warmup
```

更新为：

$$
a_{t+1}=a_t-\eta_a(t)(M_{geo}+\rho I)^{-1}\nabla_aL_t.
$$

它的表达力 rescue 只允许小幅调：

```text
branch_final_scale
coeff_lr
Sobolev alpha/beta
rest/head schedule
```

不允许加入新组件。

### 3.5 Candidate UO-PGAdam：统一 AdamW with preconditioned KAN gradient

这是本轮新增的 unified optimizer 设定。

非 KAN 参数：

$$
g_\theta=\nabla_\theta L.
$$

KAN coefficient：

$$
g_a=\nabla_a L,
$$

$$
\tilde g_a=(M_{geo}+\rho I)^{-1}g_a.
$$

然后所有参数统一交给 AdamW：

$$
(\theta,a)\leftarrow\operatorname{AdamW}\left((\theta,a),(g_\theta,\tilde g_a)\right).
$$

这个方法的优点是工程上更统一；风险是 AdamW moment 会在 $\tilde g_a$ 上积累，可能造成 branch 放大。

### 3.6 Candidate UO-PostAdam：Adam moment first, then Sobolev postcondition

另一种统一动力学是先让 AdamW 产生 KAN 的 Adam direction：

$$
d_a^{Adam}=\frac{\hat m_a}{\sqrt{\hat v_a}+\epsilon},
$$

然后对这个 direction 做 Sobolev postconditioning：

$$
\tilde d_a=(M_{geo}+\rho I)^{-1}d_a^{Adam}.
$$

更新为：

$$
a_{t+1}=a_t-\eta_a\tilde d_a.
$$

非 KAN 参数仍然是标准 AdamW direction。这个方法不是严格单一 optimizer object，但它统一了 moment dynamics，同时保留 KAN functional geometry。

它的风险是 $d_a^{Adam}$ 已经被 $v$ 归一化，再经过 Sobolev inverse 后方向可能过强。因此需要 trust / norm audit。

### 3.7 Candidate UO-NormPGAdam：normalized preconditioned-gradient AdamW

为了避免 UO-PGAdam 的 moment 放大，加入 direction norm normalization：

$$
\tilde g_a=P_{Sob}g_a,
$$

$$
\tilde g_a^{norm}=\tilde g_a\cdot\frac{\|g_a\|_2}{\|\tilde g_a\|_2+\epsilon}.
$$

然后 AdamW 使用 $\tilde g_a^{norm}$。

这个方法牺牲一部分 functional step magnitude，但可能保留 direction benefit，并降低 moment explosion 风险。

### 3.8 Negative Control：AllFunctionalRestSGD

这个只做 smoke，不进入正式搜索。它把非 KAN 参数也用手写 SGD / functional-style update 更新。过去类似设置已经表现出欠拟合风险，因此只用于确认失败边界，不作为主线。

---

## 4. 统一 profile 的严格程度

为了避免“统一 profile”定义混乱，本轮把统一分成两个等级。

### 4.1 Core-unified profile

Core-unified 要求以下项完全一致：

```text
alpha_mode
metric mode
phase mode
Sobolev alpha/beta/rho
branch_final_scale
branch schedule type
optimizer family
```

但允许 dataset-specific base learning rate 和 epochs，因为 Fashion 与 KMNIST 的 loss scale / difficulty 不同。

这是实际最合理的统一定义。

### 4.2 Strict-unified profile

Strict-unified 要求连这些也一致：

```text
coeff_lr
rest_lr
rest schedule
coeff schedule
branch_final_scale
epochs, except reporting can also include equal-step comparison
```

Strict-unified 作为探索目标，不作为第一轮必须达成的硬目标。原因是过早要求所有数值超参完全相同，可能会让一个本来有效的 optimizer family 被误判失败。

---

## 5. 实验阶段总览

v3.3 分为九个阶段：

```text
P0: implementation smoke and exact reproduction
P1: unified full-start seed0 sanity
P2: unified full-start expression rescue 3-seed
P3: unified optimizer smoke
P4: unified optimizer 3-seed comparison
P5: 5-seed confirm
P6: 10-seed final confirm
P7: mechanism and expression audit
P8: direction / moment audit for unified optimizer
P9: wall-clock, memory, and decision report
```

这些阶段不是并行大扫。P1/P2 先确定 unified full-start 是否值得，P3/P4 再判断 unified optimizer 是否值得，P5 以后只确认少数候选。

---

## 6. P0：implementation smoke and reproduction

### 6.1 目标

P0 确认新 package、alpha mode、optimizer method、full-start profile 和 unified optimizer path 没有实现错误。P0 不做方法结论。

### 6.2 设置

```text
datasets = Fashion-MNIST, KMNIST
train / val / test = 512 / 128 / 128
epochs = 1
seeds = 0
model = h32 / depth2 / basis8
alpha_mode = fixed1
```

### 6.3 方法

```text
AdamW-alphaFixed1
U-FULL-smoke
UO-PGAdam-smoke
UO-NormPGAdam-smoke
AllFunctionalRestSGD-smoke
```

### 6.4 必须记录

```text
run_failed
nan_or_inf_count
trust_clip_rate
v3_metric_condition_geometry
v3_metric_eig_min_geometry
v3_metric_eig_max_geometry
precond_solve_time_ms
step_time_ms
branch_over_adamw
no_kan_acc_drop
optimizer_path
kan_grad_transform_mode
```

### 6.5 通过条件

P0 通过要求：

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

如果 UO 方法在 smoke 中出现 NaN、branch/A 大于 $1.2$ 或 Jacobian spike，则 UO 进入 quarantine，不参与 P4。

---

## 7. P1：unified full-start seed0 sanity

### 7.1 目标

P1 用 seed0 快速判断 full-start fixed1 在两个数据集上的趋势，尤其是 Fashion 上是否明显弱于 hard diag-to-full，以及 KMNIST expression 是否有被 rescue 的迹象。

### 7.2 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0
Fashion epochs = 30
KMNIST epochs = 20
model = h96 / depth4 / basis24
alpha_mode = fixed1
```

### 7.3 方法矩阵

```text
AdamW-alphaFixed1
GA-FU-v2-alphaFixed1
DatasetSpecificBest-v3.2
U-FULL-base
U-FULL-f085
U-FULL-f090
U-FULL-cplus
U-FULL-f085-cplus
U-FULL-softgeo
```

各候选含义：

```text
U-FULL-base:
  branch_final_scale = 0.80
  coeff_lr = current dataset base
  alpha_geo = 0.15
  beta_geo = 0.02

U-FULL-f085:
  branch_final_scale = 0.85

U-FULL-f090:
  branch_final_scale = 0.90

U-FULL-cplus:
  coeff_lr = 1.10x current base

U-FULL-f085-cplus:
  branch_final_scale = 0.85
  coeff_lr = 1.10x current base

U-FULL-softgeo:
  alpha_geo = 0.10
  beta_geo = 0.01
```

这里的 `current dataset base` 先沿用 v3.2：Fashion 使用当前 Fashion full-start candidate 的 base lr；KMNIST 使用 `K-V3-FULL-base` 的 base lr。

### 7.4 P1 判定

P1 不是正式结论，只用于筛选进入 P2 的候选。

进入 P2 的条件是：

Fashion：

$$
\operatorname{Acc}_{U-FULL}\geq\operatorname{Acc}_{AdamW}-0.003.
$$

$$
\operatorname{AUCImprove}>0.08.
$$

KMNIST：

$$
\operatorname{AccGap}<0.007.
$$

$$
\operatorname{AUCImprove}>0.08.
$$

$$
\frac{\Delta_{noKAN}^{U-FULL}}{\Delta_{noKAN}^{AdamW}}>0.68.
$$

如果没有任何 U-FULL candidate 在 KMNIST 上把 no-KAN ratio 推到 $0.68$ 以上，则 expression rescue 需要转向 hard/diag warmup，不再强推 full-start。

---

## 8. P2：unified full-start expression rescue 3-seed

### 8.1 目标

P2 是本轮核心搜索，但搜索空间必须很窄。它只验证 P1 筛出的 2 到 3 个 U-FULL 候选能否在 seed0/1/2 上稳定提高 expression，同时保留 AUC / geometry。

### 8.2 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2
Fashion epochs = 30
KMNIST epochs = 20
model = h96 / depth4 / basis24
alpha_mode = fixed1
```

### 8.3 方法

固定包含：

```text
AdamW-alphaFixed1
GA-FU-v2-alphaFixed1
DatasetSpecificBest-v3.2
U-FULL-base
```

额外包含 P1 最好的两个 expression rescue candidate，例如：

```text
U-FULL-f085
U-FULL-f085-cplus
```

或：

```text
U-FULL-f090
U-FULL-softgeo
```

### 8.4 主要记录

P2 必须每个 epoch 记录：

```text
val_loss
val_acc
test_acc if enabled
branch_over_adamw
branch_ratio_per_layer
no_kan_val_acc
no_kan_test_acc
no_kan_acc_drop
kan_logit_delta_norm_mean
kan_logit_delta_norm_p95
kan_margin_contribution_mean
kan_margin_contribution_p95
phi_prime_p95
phi_prime_max
jacobian_condition_max
curvature_energy
sobolev_norm
ECE
step_time_ms
```

尤其要记录 final no-KAN ratio：

$$
R_{noKAN}=
\frac{\Delta_{noKAN}^{method}}{\Delta_{noKAN}^{AdamW}+\epsilon}.
$$

还要记录 branch-final relation：

$$
R_{branch}=
\frac{r_{branch}^{method}}{r_{branch}^{AdamW}+\epsilon}.
$$

### 8.5 P2 成功标准

P2 候选进入 P5 的条件：

Fashion：

$$
\text{acc gap vs AdamW}<0.
$$

$$
\text{AUC improvement}>0.08.
$$

$$
R_{noKAN}>0.7.
$$

$$
\phi'\text{ red}>0.25,
\quad
J\text{ red}>0.20.
$$

KMNIST：

$$
\text{acc gap vs AdamW}<0.005.
$$

$$
\text{AUC improvement}>0.085.
$$

$$
R_{noKAN}>0.7.
$$

$$
\text{ECE reduction}>0.10.
$$

如果某个 U-FULL 候选在两个数据集上都满足上述条件，则进入 unified profile 5-seed confirm。

---

## 9. P3：unified optimizer smoke

### 9.1 目标

P3 验证统一非 KAN / KAN optimizer 设定是否能稳定运行。这个阶段不追求最优，只看它是否有资格进入 3-seed 正式对比。

### 9.2 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0
Fashion epochs = 10 for smoke, then 30 if stable
KMNIST epochs = 10 for smoke, then 20 if stable
model = h96 / depth4 / basis24
alpha_mode = fixed1
metric = full_sobolev_gram from start
```

### 9.3 方法

```text
Hybrid-U-FULL-base
UO-PGAdam
UO-NormPGAdam
UO-PostAdam
UO-PostAdam-trust030
AllFunctionalRestSGD-negative
```

### 9.4 统一 optimizer 特有记录

除了常规指标，P3 必须记录：

```text
optimizer/unified_mode
optimizer/kan_grad_transform
optimizer/kan_raw_grad_norm
optimizer/kan_precond_grad_norm
optimizer/kan_precond_over_raw_norm
optimizer/rest_grad_norm
optimizer/kan_update_norm
optimizer/rest_update_norm
optimizer/kan_update_over_param_norm
optimizer/rest_update_over_param_norm
optimizer/kan_update_over_rest_update
optimizer/adam_m_norm_kan
optimizer/adam_v_norm_kan
optimizer/moment_cos_raw_precond
optimizer/moment_cos_precond_current
optimizer/moment_amplification_ratio
optimizer/bad_step_count
optimizer/trust_clip_rate
```

关键 direction cosine：

$$
\cos(g_a,Pg_a),
$$

$$
\cos(m_t,Pg_a),
$$

$$
\cos(d_t^{Adam},d_t^{Hybrid}).
$$

这些指标用于判断 Adam moment 是否正在偏离 functional direction。

### 9.5 P3 失败条件

如果出现以下任一情况，该 UO 方法停止：

$$
\text{branch/AdamW}>1.1.
$$

$$
\phi'_{p95}>1.2\times\phi'_{AdamW}.
$$

$$
\kappa(J)>1.2\times\kappa(J)_{AdamW}.
$$

$$
\text{trust clip rate}>0.10.
$$

$$
\text{test acc gap}>2\%.
$$

$$
\text{bad step count}>10\%\text{ of updates}.
$$

只有通过 P3 的 UO candidate 才能进入 P4。

---

## 10. P4：unified optimizer 3-seed comparison

### 10.1 目标

P4 判断统一 optimizer 动力学是否有实际价值。它不是为了替代 U-FULL 主线，而是回答：是否可以把 hybrid update 简化成统一 AdamW-style optimizer，同时保留 functional geometry。

### 10.2 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2
alpha_mode = fixed1
base profile = P2 best U-FULL or U-FULL-base
```

### 10.3 方法

```text
AdamW-alphaFixed1
Hybrid-U-FULL-best
UO-PGAdam
UO-NormPGAdam
UO-PostAdam
```

如果 P3 中只有一个 UO 方法稳定，则 P4 只扩那个方法。

### 10.4 P4 判定

统一 optimizer 被认为有价值，需要满足：

$$
\operatorname{Acc}_{UO}\geq\operatorname{Acc}_{Hybrid}-0.003.
$$

$$
\operatorname{AUCImprove}_{UO}\geq\operatorname{AUCImprove}_{Hybrid}-0.01.
$$

$$
\phi\text{ red}_{UO}\geq\phi\text{ red}_{Hybrid}-0.05.
$$

$$
J\text{ red}_{UO}\geq J\text{ red}_{Hybrid}-0.05.
$$

$$
R_{noKAN,UO}\geq R_{noKAN,Hybrid}-0.05.
$$

如果 UO 不能达到这些条件，则保留 hybrid optimizer，不把统一 AdamW-style optimizer 写成主线。

---

## 11. P5：5-seed confirm

### 11.1 目标

P5 是正式确认阶段，只确认极少数候选。

### 11.2 方法选择

进入 P5 的最多 4 个方法：

```text
AdamW-alphaFixed1
GA-FU-v2-alphaFixed1
DatasetSpecificBest-v3.2
U-FULL-best-v3.3
```

如果 P4 中 UO 方法通过，则加入一个：

```text
UO-best-v3.3
```

不能加入多个 UO 候选，避免结论分散。

### 11.3 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2,3,4
Fashion epochs = 30
KMNIST epochs = 20
model = h96 / depth4 / basis24
alpha_mode = fixed1
```

### 11.4 P5 主要 scorecard

每个方法输出：

```text
test_acc_mean ± std
paired_acc_delta_vs_adamw + CI95
val_auc_improvement_vs_adamw
paired_auc_delta_vs_adamw + CI95
phi_prime_reduction
jacobian_reduction
branch_over_adamw
no_kan_drop
no_kan_drop_over_adamw
kan_margin_contribution
ECE_reduction
step_time_ms
time_to_relaxed_loss
```

### 11.5 P5 通过条件

Unified full-start clean pass：

$$
\text{Fashion acc}\geq\text{AdamW acc}.
$$

$$
\text{KMNIST acc gap}<0.005.
$$

$$
\text{AUC improvement}>0.085\text{ on both datasets}.
$$

$$
R_{noKAN}>0.7\text{ on both datasets}.
$$

$$
\phi\text{ red}>0.25,
\quad
J\text{ red}>0.20.
$$

Unified optimizer pass：

$$
\text{UO matches or improves Hybrid-U-FULL on at least 4 of 6 main axes:}
$$

```text
accuracy
AUC
phi/J geometry
ECE
no-KAN expression
step-time / implementation simplicity
```

If UO is worse on accuracy or expression, it remains an ablation.

---

## 12. P6：10-seed final confirm

### 12.1 目标

P6 是最终结论阶段。只扩一个 unified full-start candidate，以及一个 UO candidate 如果它在 P5 通过。

### 12.2 设置

```text
seeds = 0..9
datasets = Fashion-MNIST, KMNIST
methods = AdamW, current v3.2 best, U-FULL-best, optional UO-best
```

### 12.3 最终决策

P6 后允许四种结论。

#### 结论 A：Unified clean profile 成立

条件：U-FULL 在 Fashion 和 KMNIST 上都通过 clean gates。

写法：

```text
full_sobolev_gram from start + alphaFixed1 becomes the unified GA-FU-v3.3 profile.
```

#### 结论 B：Unified Pareto profile 成立，但 clean 仍 dataset-specific

条件：U-FULL 在两个数据集上都匹配 accuracy 并改善 AUC/geometry/ECE，但至少一个数据集 expression 或 AUC 没过 strict gate。

写法：

```text
full-start fixed1 is the unified Pareto profile, while hard diag-to-full remains the Fashion clean profile.
```

#### 结论 C：Unified optimizer 动力学成立

条件：UO-best 匹配 hybrid U-FULL，并且没有 moment / geometry 风险。

写法：

```text
KAN functional geometry can be integrated into a unified AdamW-style optimizer through Sobolev-preconditioned gradients.
```

#### 结论 D：Hybrid 仍是主线

条件：UO 方法不稳定或 expression/accuracy 明显变差。

写法：

```text
The hybrid design remains necessary: AdamW for non-KAN parameters and functional update for KAN coefficients.
```

---

## 13. P7：mechanism and expression audit

### 13.1 目标

P7 解释 unified profile 是否真的解决了 expression 问题，而不是只靠 final accuracy。

### 13.2 必须记录的 expression 指标

```text
branch/global/final_output_norm_ratio
branch/global/branch_over_adamw
branch/layer_k/final_output_norm_ratio
branch/layer_k/branch_scale
ablation/no_kan_val_acc
ablation/no_kan_test_acc
ablation/no_kan_acc_drop
ablation/no_kan_drop_over_adamw
ablation/no_kan_loss_increase
ablation/kan_logit_delta_norm_mean
ablation/kan_logit_delta_norm_p95
ablation/kan_margin_contribution_mean
ablation/kan_margin_contribution_p95
basis/active_basis_fraction
basis/dead_basis_fraction
```

### 13.3 表达力成功定义

Unified profile 表达力成功要求：

$$
0.55<\operatorname{branch/AdamW}<0.85.
$$

$$
R_{noKAN}>0.7.
$$

$$
\operatorname{KAN\ margin\ contribution}>0.
$$

$$
\operatorname{active\ basis\ fraction}>0.70.
$$

如果 branch/A 提高但 no-KAN ratio 没提高，说明只是 branch norm 变大，不是任务贡献变强。

### 13.4 P7 可视化

必须画：

```text
branch_over_adamw vs epoch
no_kan_drop vs epoch
no_kan_drop_over_adamw bar
branch_over_adamw vs no_kan_drop scatter
KAN margin contribution distribution
per-layer branch ratio heatmap
active basis fraction heatmap
```

最关键的图是：

```text
x = branch_over_adamw
 y = no_kan_drop_over_adamw
 color = method
 size = AUC improvement
```

它能直接显示 expression rescue 是否成功。

---

## 14. P8：direction and moment audit for unified optimizer

### 14.1 目标

P8 解释 UO 方法是否真的把 functional geometry 和 AdamW moment 融合好了。

### 14.2 方向指标

每个 KAN coeff update 记录：

$$
\cos(g_a,Pg_a),
$$

$$
\cos(Pg_a,d_{Hybrid}),
$$

$$
\cos(d_{UO},d_{Hybrid}),
$$

$$
\frac{\|Pg_a\|}{\|g_a\|+\epsilon},
$$

$$
\frac{\|d_{UO}\|}{\|d_{Hybrid}\|+\epsilon}.
$$

### 14.3 Moment 指标

```text
adam/m_norm_kan
adam/v_norm_kan
adam/m_over_grad_norm_kan
adam/update_over_grad_norm_kan
adam/moment_cos_with_current_precond_grad
adam/moment_cos_with_raw_grad
adam/moment_lag_cosine
adam/moment_amplification_ratio
```

moment lag cosine 定义为：

$$
\cos(m_t,\tilde g_t).
$$

如果这个值持续低于 $0.5$，说明 Adam moment 正在积累旧方向，可能不适合快速变化的 functional metric。

### 14.4 Bad-step 指标

每个 epoch 记录：

```text
bad_step_count
predicted_descent_proxy
actual_loss_delta_shadow
pred_actual_sign_agreement
```

如果 UO 方法的 bad step rate 高于 hybrid，则说明统一 AdamW-style moment 破坏了 functional descent property。

### 14.5 P8 可视化

```text
cos(d_UO, d_Hybrid) over epoch
moment_cos_with_precond_grad over epoch
norm ratio violin plot
update_over_param_norm by group
KAN update norm vs rest update norm
moment amplification vs phi_prime scatter
bad_step_rate vs AUC improvement scatter
```

---

## 15. P9：wall-clock, memory, and deployment decision

### 15.1 目标

v3.2 已经显示 true-Gram step time 约为 AdamW 的 $1.1\times$，且 relaxed target time 没有赢 AdamW。因此 v3.3 必须继续诚实报告 wall-clock。

### 15.2 指标

```text
compute/step_time_ms
compute/epoch_time_sec
compute/total_train_time_sec
compute/precond_solve_time_ms
compute/metric_build_time_ms
compute/time_ratio_vs_adamw
memory/peak_allocated_mb
memory/activation_memory_mb_if_available
memory/optimizer_state_mb
```

Unified optimizer 还要记录 optimizer state memory：

```text
memory/adam_state_kan_mb
memory/adam_state_rest_mb
memory/total_optimizer_state_mb
```

因为 UO-PGAdam 会给 KAN coeff 增加 Adam moments，可能比 manual functional update 使用更多 optimizer state。

### 15.3 Target-matched convergence

记录：

```text
target/reached_relaxed_loss
target/epochs_to_relaxed_loss
target/time_to_relaxed_loss
target/reached_adamw_final_loss
target/epochs_to_adamw_final_loss
target/time_to_adamw_final_loss
```

只有当：

$$
\operatorname{time\_to\_target}_{method}<\operatorname{time\_to\_target}_{AdamW}
$$

或至少：

$$
\operatorname{steps\_to\_target}_{method}<\operatorname{steps\_to\_target}_{AdamW}
$$

且 step time overhead 不超过 $1.1\times$，才允许 claim faster convergence。

### 15.4 P9 可视化

```text
step_time_ms by method
precond_solve_time_ms stacked bar
optimizer_state_memory by method
time_to_relaxed_loss bar
AUC improvement vs time_to_target scatter
accuracy vs step_time Pareto plot
```

---

## 16. Failure table

每个 run 必须打 failure tags。v3.3 需要比 v3.2 更重视 expression 和 UO moment failure。

### 16.1 通用 failure tags

| failure type | 触发条件 | 解释 |
|---|---|---|
| `accuracy_gap_too_large` | acc gap vs AdamW > 1% | 任务性能不够 |
| `no_convergence_gain` | AUC improvement <= 0 | 没有 loss trajectory 优势 |
| `auc_below_unified_gate` | AUC improvement < 8% | 不满足 unified gate |
| `expression_incomplete` | no-KAN ratio < 0.7 | KAN branch 任务贡献不足 |
| `branch_underactive` | branch/A < 0.5 | branch 太弱 |
| `branch_overactive` | branch/A > 0.95 | branch 过强 |
| `geometry_not_preserved` | phi/J reduction 不达标 | 几何优势不足 |
| `calibration_worse` | ECE 不优于 AdamW | 校准失败 |
| `wall_clock_slower` | time-to-target 慢于 AdamW | 不能 claim faster |
| `seed_unstable` | acc std > 1.5% | seed 不稳定 |

### 16.2 Unified optimizer failure tags

| failure type | 触发条件 | 解释 |
|---|---|---|
| `moment_explosion` | moment amplification ratio > 3 | Adam moment 放大 |
| `moment_direction_drift` | cos(moment, current precond grad) < 0.5 | moment 偏离当前 functional direction |
| `kan_update_too_large` | KAN update / param norm > threshold | KAN 更新过大 |
| `rest_kan_update_imbalance` | KAN/rest update ratio 异常 | 统一 optimizer 组间失衡 |
| `trust_not_safety_only` | clip rate > 5% | trust 实质参与优化 |
| `bad_step_rate_high` | bad step rate > 10% | descent property 损坏 |
| `optimizer_state_too_large` | state memory 明显增大 | 工程不划算 |

### 16.3 Failure visualization

```text
failure type count by method
failure type count by dataset
expression_incomplete vs branch_underactive scatter
moment_explosion vs geometry_not_preserved scatter
accuracy_gap vs no_kan_ratio scatter
AUC improvement vs no_kan_ratio scatter
```

---

## 17. 推荐执行顺序

### 17.1 第一轮执行

```text
P0 smoke
P1 seed0 unified full-start sanity
P3 unified optimizer smoke
```

这一步只需要确认候选是否安全，不要扩 seed。

### 17.2 第二轮执行

```text
P2 unified full-start expression rescue, seeds 0,1,2
P4 unified optimizer comparison, seeds 0,1,2 only for P3 survivors
```

这一步筛出最多两个 U-FULL 候选和最多一个 UO 候选。

### 17.3 第三轮执行

```text
P5 5-seed confirm
P7 expression audit
P8 direction/moment audit
```

如果 P5 没有 unified 候选过 gate，不继续 P6。

### 17.4 第四轮执行

```text
P6 10-seed final confirm
P9 wall-clock and memory report
```

P6 只跑最终候选，不再加入新搜索项。

---

## 18. 建议实现的 runner packages

建议在 `run_gafu_v3.py` 中新增：

```text
V3_3_P0_SMOKE
V3_3_P1_UFULL_SANITY
V3_3_P2_UFULL_EXPR3
V3_3_P3_UO_SMOKE
V3_3_P4_UO_EXPR3
V3_3_P5_CONFIRM5
V3_3_P6_CONFIRM10
```

建议在 `dgkan_core.py` 中新增配置字段：

```text
optimizer_method
kan_grad_transform
unified_optimizer_mode
kan_precondition_before_adam
kan_precondition_after_adam
kan_precond_grad_normalize
uo_trust_radius
uo_record_moment_audit
```

建议每个 run 输出：

```text
runs/{run_id}/curves.csv
runs/{run_id}/branch_audit.csv
runs/{run_id}/direction_audit.csv
runs/{run_id}/moment_audit.csv
runs/{run_id}/target_summary.json
```

---

## 19. 最终写法预案

### 19.1 如果 unified full-start 成功

可以写：

```text
We found that a full-Sobolev-from-start GA-FU profile with fixed residual branch scale provides a unified optimizer profile across Fashion-MNIST and KMNIST. Compared with AdamW, it preserves or improves accuracy while improving validation-loss trajectory, functional geometry, calibration, and branch expression.
```

中文：

```text
我们发现 full_sobolev_gram from start + alphaFixed1 可以作为 Fashion-MNIST 与 KMNIST 的统一 GA-FU profile。它在不牺牲 accuracy 的情况下改善 validation-loss trajectory、函数几何、校准和 KAN branch 表达力。
```

### 19.2 如果 unified full-start 只达到 Pareto

可以写：

```text
The full-start profile is a unified Pareto optimizer: it improves geometry, calibration, and validation-loss AUC across datasets, but dataset-specific profiles remain stronger for clean-default accuracy and branch expression.
```

中文：

```text
full-start profile 是统一 Pareto optimizer：它跨数据集改善几何、校准和 AUC，但 clean default 仍需要 dataset-specific profile。
```

### 19.3 如果 unified optimizer 成功

可以写：

```text
Sobolev-preconditioned KAN gradients can be integrated into a unified AdamW-style optimizer without losing the benefits of functional geometry.
```

中文：

```text
KAN 的 Sobolev-preconditioned gradient 可以并入统一 AdamW-style optimizer，同时保留 functional geometry 的收益。
```

### 19.4 如果 unified optimizer 失败

可以写：

```text
The hybrid optimizer remains necessary: AdamW dynamics are well-suited for non-KAN parameters, while KAN edge coefficients benefit from explicit functional-space updates. Unified AdamW-style moments introduce either expression loss or moment amplification.
```

中文：

```text
hybrid optimizer 仍然必要：非 KAN 参数适合 AdamW，KAN edge coefficients 需要显式函数空间更新。统一 AdamW-style moment 要么损失表达力，要么带来 moment amplification。
```

---

## 20. 本轮最重要的不要做什么

本轮不做以下事情：

```text
不加入 SNR
不加入 FSAM
不加入 data metric
不加入 prox/spectral
不换 Rational/KAT
不改模型容量
不重新打开 learnable alpha 作为主线
不大范围扫几十组 learning rate
```

原因是当前问题已经很明确：

$$
\boxed{
\text{验证 unified full-start fixed1，并做最小 expression rescue。}
}
$$

统一 optimizer 设定也只作为明确限定的实验包，不允许污染主线结论。

---

## 21. 最终一句话

v3.3 的目标是把当前结果从：

```text
Fashion clean default + KMNIST Pareto candidate + dataset-specific profile
```

推进到：

```text
Unified full-start profile, if expression rescue succeeds;
or a clear proof that hybrid/dataset-specific profiles are still necessary.
```

最理想结果是：

$$
\boxed{
\text{full_sobolev_gram from start + alphaFixed1 + minimal expression rescue}
}
$$

成为统一 profile。

第二理想结果是：

$$
\boxed{
\text{full-start fixed1 是统一 Pareto profile，但 clean default 仍 dataset-specific。}
}
$$

无论哪种结果，都能让项目从“继续调 optimizer”进入“明确方法边界并准备 scaling”的阶段。
