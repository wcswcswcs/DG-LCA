# DG-KAN 重构研究计划：Primitive-Adjoint Co-Design, Functional-Space Update, Analytic Local Credit and Low-Memory Sufficient Statistics

> 版本：v2.0  
> 定位：根据当前 Stage A/B/C 系列实验结果重写的主线研究计划  
> 核心变化：不再把 learned router 作为主贡献；将主线收缩并强化为 **KAN primitive-adjoint co-design + functional-space update + analytic local credit + low-memory sufficient statistics**  
> 目标场景：视觉任务为最终主应用，包括 MNIST / Fashion-MNIST / KMNIST / CIFAR-10 / CIFAR-100 / Tiny-ImageNet small 等逐级扩展  
> 公式格式：Typora 友好，使用 `$...$` 和 `$$...$$`

---

## 0. 当前阶段性结论

我们最初希望学习 local credit router 来替代局部 VJP。但经过 Stage A/B/C 以及 Stage C-next v2/v3/v4 的实验，主线需要重新定位。

当前最稳的实验结论是：

1. **DG-KAN 架构是可训练的，并且 residual / LayerNorm / KAN-like block 显著改善局部几何稳定性。**  
   普通 KAN 在部分设置中 Jacobian condition 极高，而 residual DG-KAN 可以把局部谱性质控制到更健康的范围。

2. **KAN-like primitive 提供了精确且高效的 analytic adjoint。**  
   对 KAN primitive：

   $$
   y_j = \sum_i \phi_{ji}(x_i)
   $$

   局部 VJP 为：

   $$
   g_{x_i} = \sum_j g_{y_j}\phi'_{ji}(x_i)
   $$

   即：

   $$
   g_x = \Phi'(x)^\top g_y
   $$

   实验中 analytic VJP 与 autograd VJP 对齐到数值精度，并且 analytic VJP 更快。

3. **Functional-space update 对 KAN edge coefficients 有价值。**  
   普通 coefficient update 是：

   $$
   a \leftarrow a - \eta \nabla_a \mathcal L
   $$

   Functional-space update 是：

   $$
   a \leftarrow a - \eta (M + \rho I)^{-1}\nabla_a \mathcal L
   $$

   实验显示：
   - `diag` 早期下降快；
   - `sobolev` 最终函数拟合质量好；
   - `diag_to_sobolev` 是当前最稳 schedule；
   - RKHS 当前不进入主线。

4. **Classification 中 functional update 必须 hybrid。**  
   KAN coefficients 使用 functional update，非 KAN 参数继续使用 AdamW / Adam：

   $$
   \boxed{
   \text{KAN coefficients: functional update}
   +
   \text{non-KAN parameters: AdamW}
   }
   $$

5. **Branch activity 是解释分类性能的关键。**  
   Residual DG-KAN block 为：

   $$
   h_{k+1} = h_k + \alpha_k \operatorname{KAN}_k(\operatorname{LN}(h_k))
   $$

   Branch ratio 定义为：

   $$
   r_{\text{branch},k}
   =
   \frac{
   \|\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))\|
   }{
   \|h_k\| + \epsilon
   }
   $$

   实验显示：branch 太弱会 under-active，branch 太强会牺牲几何稳定性，中间存在 Pareto regime。

6. **Learned router 不是当前主贡献。**  
   Hidden-only router 学不好 non-identity correction；derivative-aware / decomposition-aware router 可以高精度近似 span correction，但 micro-training 收益不足，并且 deterministic first-order decomposition 更便宜。因此 learned router 暂时降级为可选压缩 / 近似 / 消融模块。

因此，新的主线不再是：

$$
\text{learned router replaces local VJP}
$$

而是：

$$
\boxed{
\text{KAN primitive-adjoint co-design}
+
\text{functional-space update}
+
\text{analytic local credit}
+
\text{low-memory sufficient statistics}
}
$$

---

## 1. 新研究命题

### 1.1 核心命题

DG-KAN 的核心命题是：

> 在 KAN-like full hidden-state networks 中，每个 forward primitive 都可以配套一个精确、低成本、谱可控的 adjoint primitive；每个 KAN edge function 又拥有自然的函数空间更新几何。因此，DG-KAN 可以把 forward、adjoint、update 和 sufficient statistics 设计成一个闭环，从而带来更快收敛、更稳定 credit transport、更可控函数复杂度、更低 activation memory，以及潜在的更好泛化和抗遗忘能力。

用四个公式概括这个闭环：

### Forward primitive

$$
 y_j = \sum_i \phi_{ji}(x_i)
$$

### Analytic adjoint

$$
 g_{x_i} = \sum_j g_{y_j}\phi'_{ji}(x_i)
$$

### Functional-space update

$$
 a \leftarrow a - \eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

### Low-memory sufficient statistics

若：

$$
\phi_{ji}(t)=\sum_m a_{jim}B_m(t)
$$

则 edge coefficient gradient 可写为：

$$
\nabla_{a_{jim}}\mathcal L
=
\frac{1}{B}\sum_{n=1}^{B}g_j^{(n)}B_m(x_i^{(n)})
$$

因此可以累积统计量：

$$
s_{jim}=\sum_n g_j^{(n)}B_m(x_i^{(n)})
$$

而不是保存完整 autograd activation graph。

---

## 2. 研究目标

新的研究目标分为六类。

### 2.1 快收敛

验证 functional-space update 是否能改善训练动力学。

关注问题：

1. `diag` 是否提供更快 early-stage descent；
2. `sobolev` 是否提供更好的 final solution；
3. `diag_to_sobolev` 是否同时保留二者优势；
4. 在视觉任务中，这些优势是否仍然存在。

核心指标包括：

$$
\text{loss AUC}
$$

$$
\text{steps-to-target}
$$

$$
\text{time-to-target}
$$

$$
\text{largest stable learning rate}
$$

---

### 2.2 泛化与不易过拟合

验证 functional-space update 是否能通过控制 KAN edge function 的斜率、曲率和函数复杂度，改善泛化。

核心问题：

1. Sobolev / diag-to-Sobolev 是否降低 train-test gap；
2. 在小数据设置中是否优于 AdamW；
3. 在 label noise 或 corruptions 下是否更稳；
4. 泛化提升是否是真泛化，而不是欠拟合。

关键指标：

$$
\text{train-test gap}_{loss}=\mathcal L_{test}-\mathcal L_{train}
$$

$$
\text{train-test gap}_{acc}=\operatorname{Acc}_{train}-\operatorname{Acc}_{test}
$$

函数复杂度指标：

$$
\phi'_{p95},\quad \int |\phi''(t)|^2dt,\quad \|\phi\|_{\mathcal H}
$$

---

### 2.3 抗遗忘

验证 functional-space update 是否因为限制函数空间变化而减少旧任务遗忘。

Sequential task 中，forgetting 定义为：

$$
F_i=
\max_{t< T} \operatorname{Acc}_{i,t}
-
\operatorname{Acc}_{i,T}
$$

平均遗忘：

$$
F_{avg}=\frac{1}{N}\sum_i F_i
$$

同时记录函数漂移：

$$
D_{\phi}^{A\rightarrow B}
=
\|\phi_{after\ B}-\phi_{after\ A}\|_{\mathcal H}
$$

以及 edge function drift：

$$
\Delta\phi' = \int |\phi'_{after\ B}(t)-\phi'_{after\ A}(t)|^2dt
$$

---

### 2.4 表达力与 branch utilization

验证 DG-KAN 是否真正利用了 KAN branch 的函数表达能力，而不是依赖 stem/head 或 identity path。

必须同时记录任务性能和 branch contribution。

No-KAN ablation：

$$
\Delta_{noKAN}
=
\operatorname{Acc}_{full}-\operatorname{Acc}_{noKAN}
$$

如果：

$$
\Delta_{noKAN}\approx0
$$

说明 KAN branch 没有实际参与任务。

Branch utilization 必须和 accuracy 一起报告：

$$
\text{branch / AdamW}
=
\frac{r_{branch}^{method}}{r_{branch}^{AdamW}}
$$

---

### 2.5 更稳定的 credit transport

验证 functional-space update 和 Sobolev control 是否降低：

1. credit amplification；
2. noise gain；
3. Jacobian condition；
4. $
abla$ credit rank / instability。

Credit amplification：

$$
A_k=\frac{\|g_k\|}{\|g_{k+1}\|+\epsilon}
$$

Noise gain：

$$
G_{noise}
=
\frac{
\|C(g+\sigma\xi)-C(g)\|
}{
\|\sigma\xi\|+\epsilon
}
$$

Jacobian condition：

$$
\kappa(J)=\frac{\sigma_{max}(J)}{\sigma_{min}(J)+\epsilon}
$$

---

### 2.6 低显存 sufficient statistics

验证 DG-KAN 是否能减少 activation memory，而不仅仅是改 optimizer。

比较：

1. full BP；
2. checkpointing；
3. analytic local adjoint；
4. first-order local adjoint；
5. sufficient statistics accumulation。

核心 claim 只有在下面条件满足时才成立：

$$
M_{DG-KAN}<M_{checkpoint}
$$

或者至少：

$$
M_{DG-KAN}<M_{full\ BP}
$$

并且性能 gap 可控。

---

## 3. 主架构设计

### 3.1 Residual DG-KAN block

主 block 继续使用 residual 形式：

$$
 h_{k+1}=h_k+\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

其中：

$$
\operatorname{KAN}_k(x)_j=\sum_i \phi_{k,ji}(x_i)
$$

edge function 使用 RBF 或 spline basis：

$$
\phi_{ji}(t)=\sum_m a_{jim}B_m(t)
$$

当前默认使用 RBF：

$$
B_m(t)=\exp(-\gamma(t-c_m)^2)
$$

---

### 3.2 Vision architecture

最终应用目标是视觉任务。推荐主模型为：

$$
 x \rightarrow \operatorname{Stem}(x) \rightarrow h_0
$$

$$
 h_{k+1}=h_k+\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

$$
 z=W_{head}h_K
$$

其中 Stem 可以是：

1. linear stem，用于 MNIST / Fashion-MNIST / KMNIST；
2. conv stem，用于 CIFAR-10 / CIFAR-100；
3. patch stem，用于更大图像任务。

---

### 3.3 Dense KAN 与 grouped KAN

小模型使用 dense KAN：

$$
\operatorname{KAN}(x)_j=\sum_{i=1}^{d}\phi_{ji}(x_i)
$$

中等规模视觉模型使用 grouped KAN：

$$
\operatorname{KAN}(x)_j=\sum_{i\in\mathcal G(j)}\phi_{ji}(x_i)
$$

Grouped KAN 不是 adapter，它仍然输出完整 hidden state，只是为了降低计算与参数量。

---

## 4. 方法族

### 4.1 Baseline 方法

| 方法 | 说明 |
|---|---|
| MLP-AdamW | 参数量匹配 MLP baseline |
| ResMLP-AdamW | residual MLP baseline |
| KAN-AdamW | 普通 KAN coefficient AdamW |
| DG-KAN-AdamW | residual DG-KAN + AdamW |
| Checkpointed-DG-KAN | checkpointing memory baseline |

---

### 4.2 Functional update 方法

| 方法 | KAN coeff 更新 | 非 KAN 参数 |
|---|---|---|
| DG-KAN-AdamW | AdamW | AdamW |
| DG-KAN-Diag | diagonal functional preconditioner | AdamW |
| DG-KAN-Sobolev | Sobolev functional update | AdamW |
| DG-KAN-DiagToSobolev | diag warmup then Sobolev | AdamW |

其中 `diag_to_sobolev`：

前 $T_w$ steps：

$$
 d_t = \operatorname{diag}(M+\rho I)^{-1}\nabla_a\mathcal L
$$

之后：

$$
 d_t = (M+\rho I)^{-1}\nabla_a\mathcal L
$$

---

### 4.3 Local credit 方法

| 方法 | Credit 来源 | 当前地位 |
|---|---|---|
| FullBP | full autograd BP | baseline |
| IdentityAdj | $g_k=g_{k+1}$ | 强 baseline |
| AnalyticAdj | KAN analytic adjoint | 主线 |
| FirstOrderDecompAdj | deterministic first-order decomposition | 候选 |
| DecompRouterAdj | learned decomposition-aware router | ablation / candidate |
| HiddenRouterAdj | hidden-only router | 负对照 |

当前主线不再依赖 learned router。

---

## 5. 实验路线总览

新的实验路线按以下阶段推进：

$$
\boxed{
\text{D：Joint training with analytic / identity / first-order credit}
\rightarrow
\text{E：Low-memory sufficient statistics}
\rightarrow
\text{F：Convergence and optimization dynamics}
\rightarrow
\text{G：Generalization and overfitting}
\rightarrow
\text{H：Continual learning / forgetting}
\rightarrow
\text{I：Vision scaling}
}
$$

---

## 6. Stage D：Joint training with analytic / identity / first-order credit

### 6.1 目标

Stage D 验证：analytic local credit、identity credit 和 first-order decomposition credit 在真实训练中分别有多大价值。

不再把 learned router 作为主线，而是把它作为 ablation。

---

### 6.2 方法比较

| 方法 | Credit | Update | 目的 |
|---|---|---|---|
| FullBP-DGKAN | full BP | diag-to-Sobolev | 上界 / baseline |
| IdentityAdj-DGKAN | identity credit | diag-to-Sobolev | 强 baseline |
| AnalyticAdj-DGKAN | analytic local adjoint | diag-to-Sobolev | 主方法 |
| FirstOrderDecomp-DGKAN | deterministic first-order credit | diag-to-Sobolev | 低成本近似 |
| DecompRouter-DGKAN | learned router | diag-to-Sobolev | ablation |
| RandomCredit-DGKAN | random credit | diag-to-Sobolev | 负对照 |

---

### 6.3 数据集

第一批：

1. MNIST；
2. Fashion-MNIST；
3. KMNIST。

第二批：

1. CIFAR-10 small；
2. CIFAR-100 small；
3. Tiny-ImageNet small。

---

### 6.4 W&B metrics

#### 任务性能

| metric | 含义 |
|---|---|
| `train/loss` | train loss |
| `val/loss` | validation loss |
| `test/loss` | test loss |
| `train/acc` | train accuracy |
| `val/acc` | validation accuracy |
| `test/acc` | test accuracy |
| `compare/test_acc_gap_vs_fullbp` | 相对 FullBP 的 accuracy gap |
| `compare/val_auc_gap_vs_fullbp` | 相对 FullBP 的 val AUC gap |

#### 收敛

| metric | 含义 |
|---|---|
| `conv/loss_auc_train` | train loss AUC |
| `conv/loss_auc_val` | val loss AUC |
| `conv/steps_to_target_val_loss` | 达到目标 loss 的 step |
| `conv/steps_to_target_val_acc` | 达到目标 acc 的 step |
| `conv/time_to_target_sec` | 达到目标所需时间 |
| `conv/loss_at_10pct_steps` | 10% 训练步 loss |
| `conv/loss_at_25pct_steps` | 25% 训练步 loss |
| `conv/loss_at_50pct_steps` | 50% 训练步 loss |
| `conv/loss_at_75pct_steps` | 75% 训练步 loss |

#### Credit fidelity

| metric | 含义 |
|---|---|
| `credit/cos_vs_fullbp` | credit 与 FullBP cosine |
| `credit/relerr_vs_fullbp` | credit relative error |
| `credit/norm_ratio_vs_fullbp` | norm ratio |
| `credit/identity_relerr` | identity credit relerr |
| `credit/analytic_relerr` | analytic credit relerr |
| `credit/first_order_relerr` | first-order relerr |

#### Functional update quality

| metric | 含义 |
|---|---|
| `functional/update_norm_coeff` | coefficient update norm |
| `functional/update_norm_function` | function update norm |
| `functional/pred_descent` | predicted descent |
| `functional/actual_descent` | actual descent |
| `functional/descent_ratio` | actual / predicted |
| `functional/sign_agreement_rate` | descent sign agreement rate |

#### KAN geometry

| metric | 含义 |
|---|---|
| `kan/phi_prime_p95` | $\phi'_{p95}$ |
| `kan/phi_prime_max` | $\phi'_{max}$ |
| `kan/curvature_mean` | 平均曲率 |
| `kan/sobolev_norm_mean` | Sobolev norm |
| `jacobian/condition_mean` | mean Jacobian condition |
| `jacobian/condition_max` | max Jacobian condition |
| `credit/amplification_p95` | credit amplification p95 |
| `credit/noise_gain_p95` | credit noise gain p95 |

#### Branch utilization

| metric | 含义 |
|---|---|
| `branch/mean_output_norm_ratio` | branch ratio |
| `branch/branch_over_adamw` | branch / AdamW |
| `ablation/no_kan_test_acc` | no-KAN test acc |
| `ablation/no_kan_acc_drop` | full acc - no-KAN acc |
| `ablation/kan_logit_delta_norm` | KAN branch 对 logits 的贡献 |

---

### 6.5 可视化

W&B 页面应包含：

1. **Training curves**：train / val loss by method；
2. **Accuracy bar**：test acc mean ± std；
3. **Loss AUC bar**：val loss AUC by method；
4. **Credit fidelity scatter**：credit relerr vs val AUC；
5. **Branch utilization panel**：branch ratio、no-KAN drop；
6. **Geometry panel**：$\phi'_{p95}$、curvature、Jacobian condition；
7. **Credit amplification plot**：amplification p95 over epochs；
8. **Predicted vs actual descent scatter**；
9. **Method Pareto plot**：x = step time，y = test acc，size = memory；
10. **Failure table**。

---

### 6.6 成功标准

AnalyticAdj-DGKAN 成功：

$$
\text{test acc gap vs FullBP}<1\%-2\%
$$

$$
\text{val AUC gap vs FullBP}<5\%
$$

FirstOrderDecomp 成功：

$$
\text{test acc gap vs AnalyticAdj}<1\%
$$

$$
\text{step time}\leq\text{AnalyticAdj step time}
$$

IdentityAdj 成功如果：

$$
\text{test acc gap vs AnalyticAdj}<1\%
$$

则说明 residual DG-KAN 的 credit transport 已被结构稳定化。

DecompRouter 只有在同时满足下列条件时才进入主线：

$$
\text{training gain vs IdentityAdj}>3\%-5\%
$$

或：

$$
\text{time / memory reduction vs AnalyticAdj}>20\%
$$

否则只作为 ablation。

---

## 7. Stage E：Low-memory sufficient statistics

### 7.1 目标

验证 DG-KAN 是否可以不保存完整 activation graph，而通过 analytic local credit 和 edge sufficient statistics 完成训练。

---

### 7.2 训练模式

| 模式 | 保存内容 | 说明 |
|---|---|---|
| FullBP | full activation graph | 标准 BP |
| Checkpointing | checkpoint + recompute | 强 baseline |
| AnalyticAdj | block interfaces + local recompute | analytic local credit |
| SufficientStats | block interfaces + edge stats | 主低显存方法 |
| FirstOrderStats | first-order credit + edge stats | 低成本近似 |

---

### 7.3 Memory accounting

必须拆分：

| metric | 含义 |
|---|---|
| `memory/params_mb` | 参数显存 |
| `memory/optimizer_mb` | optimizer state |
| `memory/full_activations_mb` | full BP activation graph |
| `memory/checkpoints_mb` | checkpoint memory |
| `memory/interfaces_mb` | block interface memory |
| `memory/edge_stats_mb` | KAN edge sufficient statistics |
| `memory/local_recompute_mb` | local recompute temp memory |
| `memory/router_mb` | router memory if used |
| `memory/peak_allocated_mb` | peak allocated memory |
| `memory/peak_reserved_mb` | peak reserved memory |
| `memory/reduction_vs_fullbp_pct` | 相对 FullBP 降低 |
| `memory/reduction_vs_checkpoint_pct` | 相对 checkpoint 降低 |

---

### 7.4 Compute accounting

| metric | 含义 |
|---|---|
| `compute/forward_time_ms` | forward time |
| `compute/backward_time_ms` | backward time |
| `compute/analytic_adj_time_ms` | analytic adjoint time |
| `compute/stats_accum_time_ms` | sufficient stats accumulation time |
| `compute/local_update_time_ms` | local update time |
| `compute/recompute_time_ms` | recompute time |
| `compute/total_step_time_ms` | total step time |
| `compute/overhead_vs_fullbp_pct` | 相对 FullBP overhead |
| `compute/overhead_vs_checkpoint_pct` | 相对 checkpoint overhead |

---

### 7.5 统计量正确性

对 sufficient statistics 更新产生的 gradient / update 与 full BP 产生的 gradient / update 比较：

| metric | 含义 |
|---|---|
| `stats/coeff_grad_cos_vs_fullbp` | coefficient gradient cosine |
| `stats/coeff_grad_relerr_vs_fullbp` | coefficient gradient relerr |
| `stats/precond_update_cos_vs_fullbp` | preconditioned update cosine |
| `stats/precond_update_relerr_vs_fullbp` | update relerr |
| `stats/compressed_stats_relerr` | compressed stats 误差 |
| `stats/stats_mb` | stats memory |

---

### 7.6 可视化

1. **Memory stacked bar**：params / optimizer / activation / interface / stats；
2. **Memory vs accuracy Pareto**；
3. **Step time vs accuracy Pareto**；
4. **Stats gradient fidelity heatmap**；
5. **FullBP vs sufficient stats update scatter**；
6. **Batch size scaling curve**；
7. **Depth scaling curve**。

---

### 7.7 成功标准

Weak success：

$$
M_{SufficientStats}<M_{FullBP}
$$

且：

$$
\text{accuracy gap}<3\%
$$

Medium success：

$$
M_{SufficientStats}<M_{Checkpointing}
$$

且：

$$
\text{accuracy gap}<2\%
$$

Strong success：

$$
\text{memory reduction vs FullBP}>30\%
$$

$$
\text{memory reduction vs Checkpointing}>10\%
$$

$$
\text{wall-clock overhead}<1.5\times
$$

---

## 8. Stage F：Fast convergence and optimization dynamics

### 8.1 目标

系统验证 functional-space update 是否比 AdamW / coefficient update 收敛更快。

---

### 8.2 任务

1. High-frequency regression；
2. MNIST / Fashion-MNIST；
3. KMNIST；
4. CIFAR-10 small。

---

### 8.3 比较方法

| 方法 | 说明 |
|---|---|
| AdamW | 标准 optimizer |
| Diag | diagonal functional update |
| Sobolev | Sobolev update |
| DiagToSobolev | 当前默认 schedule |
| Shampoo / K-FAC-like baseline | 若可实现，作为强 preconditioner baseline |

---

### 8.4 指标

| metric | 含义 |
|---|---|
| `conv/loss_auc` | loss AUC |
| `conv/steps_to_target` | steps-to-target |
| `conv/time_to_target` | time-to-target |
| `conv/largest_stable_lr` | largest stable LR |
| `conv/plateau_step` | plateau step |
| `stability/loss_spike_count` | loss spikes |
| `stability/bad_steps` | bad steps |

---

### 8.5 关键分析

必须区分：

1. 早期下降速度；
2. 最终测试性能；
3. 稳定学习率范围；
4. 是否只是欠拟合导致曲线平滑。

---

## 9. Stage G：Generalization and overfitting

### 9.1 目标

验证 DG-KAN functional update 是否减少过拟合，提高泛化。

---

### 9.2 实验设置

#### Small-data regime

训练集大小：

$$
N\in\{500,1000,2000,5000,10000\}
$$

#### Label noise

噪声比例：

$$
\eta_{noise}\in\{0.1,0.2,0.4\}
$$

#### Corruption / OOD

视觉任务可用：

1. Gaussian noise；
2. blur；
3. rotation；
4. contrast shift；
5. occlusion。

---

### 9.3 指标

| metric | 含义 |
|---|---|
| `generalization/train_test_gap_loss` | test loss - train loss |
| `generalization/train_test_gap_acc` | train acc - test acc |
| `generalization/val_test_gap` | val-test gap |
| `generalization/noise_robust_acc` | noisy label setting acc |
| `generalization/corruption_acc_mean` | corruption 平均 acc |
| `generalization/corruption_acc_worst` | 最差 corruption acc |
| `generalization/ece` | calibration error |
| `kan/curvature_mean` | function curvature |
| `kan/phi_prime_p95` | derivative strength |

---

### 9.4 关键判定

如果 Sobolev / diag-to-Sobolev 的 test acc 更好，但 train acc 更差很多，则可能只是欠拟合。

只有当：

$$
\text{test acc higher or similar}
$$

同时：

$$
\text{train-test gap lower}
$$

并且 branch utilization 正常，才可以说泛化更好。

---

## 10. Stage H：Continual learning and forgetting

### 10.1 目标

验证 functional-space update 是否降低 sequential learning 中的遗忘。

---

### 10.2 任务

| 任务 | 说明 |
|---|---|
| Split MNIST | 5 个二分类 task |
| Split Fashion-MNIST | 5 个二分类 task |
| Split KMNIST | 稍难 sequential task |
| Split CIFAR-10 | 5 个二分类视觉 task |
| CIFAR-100 superclass split | 更难设置 |

---

### 10.3 Baselines

| 方法 | 说明 |
|---|---|
| AdamW fine-tune | 普通 sequential training |
| AdamW + replay small buffer | 强 baseline |
| EWC / L2 regularization | continual baseline |
| DG-KAN-Sobolev | functional update |
| DG-KAN-DiagToSobolev | 当前主方法 |
| DG-KAN-Sobolev + function drift penalty | 增强版本 |

---

### 10.4 指标

Average accuracy：

$$
A_{avg}=\frac{1}{T}\sum_{i=1}^{T}\operatorname{Acc}_{i,T}
$$

Forgetting：

$$
F_i=
\max_{t<T}\operatorname{Acc}_{i,t}
-
\operatorname{Acc}_{i,T}
$$

Backward transfer：

$$
BWT=\frac{1}{T-1}\sum_{i=1}^{T-1}
\left(
\operatorname{Acc}_{i,T}-\operatorname{Acc}_{i,i}
\right)
$$

Function drift：

$$
D_{\phi}=\|\phi_{after}-\phi_{before}\|_{\mathcal H}
$$

W&B metrics：

| metric | 含义 |
|---|---|
| `cl/avg_acc` | average accuracy |
| `cl/forgetting_mean` | mean forgetting |
| `cl/forgetting_max` | max forgetting |
| `cl/bwt` | backward transfer |
| `cl/new_task_acc` | new task acquisition |
| `cl/function_drift_hnorm` | function-space drift |
| `cl/phi_prime_drift` | derivative drift |
| `cl/branch_ratio_by_task` | branch ratio over tasks |

---

### 10.5 判定

抗遗忘 claim 成立需要同时满足：

$$
F_{DG-KAN}<F_{AdamW}
$$

并且：

$$
\text{new task acc not significantly lower}
$$

否则可能只是学得慢或欠拟合。

---

## 11. Stage I：Vision scaling

### 11.1 目标

把 DG-KAN 从 MNIST / Fashion-MNIST 推进到更真实视觉任务。

---

### 11.2 数据集

| 阶段 | 数据集 |
|---|---|
| I1 | CIFAR-10 small |
| I2 | CIFAR-10 full |
| I3 | CIFAR-100 small |
| I4 | Tiny-ImageNet small |
| I5 | SVHN / STL-10 optional |

---

### 11.3 模型

| 模型 | 说明 |
|---|---|
| ConvStem-DGKAN | conv stem + DG-KAN blocks |
| PatchStem-DGKAN | patch embedding + DG-KAN blocks |
| Hybrid CNN-DGKAN | CNN backbone + DG-KAN head/block |
| Grouped DG-KAN | scale hidden dimension |

---

### 11.4 Baselines

1. MLP / ResMLP；
2. small CNN；
3. small ResNet；
4. ConvMixer / MLP-Mixer small；
5. KAN-AdamW；
6. DG-KAN-AdamW；
7. DG-KAN-DiagToSobolev；
8. DG-KAN-AnalyticAdj / SufficientStats。

---

### 11.5 指标

| metric | 含义 |
|---|---|
| `vision/test_acc` | test accuracy |
| `vision/top5_acc` | top-5 accuracy for larger datasets |
| `vision/params` | parameter count |
| `vision/flops` | FLOPs |
| `vision/throughput` | samples/sec |
| `vision/memory_peak` | peak memory |
| `vision/acc_per_param` | accuracy per parameter |
| `vision/acc_per_flop` | accuracy per FLOP |
| `vision/corruption_acc` | corruption robustness |

---

### 11.6 成功标准

DG-KAN 在视觉任务上不一定要一开始超过 ResNet。第一目标是：

1. 性能接近合理 baseline；
2. 几何更稳定；
3. memory 更低或收敛更快；
4. branch utilization 正常；
5. sufficient stats 方案可扩展。

---

## 12. W&B Dashboard 总设计

### Page 1：Research overview

包含：

1. stage summary table；
2. method vs test acc；
3. method vs val AUC；
4. method vs peak memory；
5. method vs step time；
6. pass/fail scorecard。

---

### Page 2：Functional update dynamics

包含：

1. train / val loss curves；
2. loss AUC bar；
3. steps-to-target；
4. $\phi'_{p95}$ over time；
5. curvature over time；
6. Jacobian condition over time；
7. descent predicted vs actual scatter。

---

### Page 3：Credit and adjoint

包含：

1. analytic vs BP credit fidelity；
2. identity vs analytic gap；
3. first-order vs analytic gap；
4. credit amplification；
5. noise gain；
6. credit rank。

---

### Page 4：Branch utilization

包含：

1. branch ratio over training；
2. branch / AdamW；
3. no-KAN ablation drop；
4. logit delta norm；
5. branch ratio vs test acc；
6. branch ratio vs Jacobian condition。

---

### Page 5：Memory accounting

包含：

1. stacked memory bar；
2. peak memory over depth；
3. memory vs accuracy Pareto；
4. checkpointing comparison；
5. sufficient stats memory breakdown；
6. batch-size scaling。

---

### Page 6：Generalization

包含：

1. train-test gap；
2. small-data accuracy curve；
3. label-noise robustness；
4. corruption robustness；
5. curvature vs generalization gap；
6. $\phi'_{p95}$ vs test loss。

---

### Page 7：Continual learning

包含：

1. task accuracy matrix；
2. forgetting curve；
3. average accuracy over tasks；
4. function drift over tasks；
5. branch ratio over tasks；
6. new-task vs old-task tradeoff。

---

### Page 8：Vision scaling

包含：

1. accuracy vs params；
2. accuracy vs FLOPs；
3. accuracy vs memory；
4. throughput curves；
5. CIFAR / Tiny-ImageNet progression；
6. geometry metrics by model size。

---

### Page 9：Failure analysis

包含 failure table，至少记录：

| failure type | 含义 |
|---|---|
| `branch_underactive` | branch ratio 太低 |
| `branch_overactive` | branch ratio 太高 |
| `phi_prime_explosion` | $\phi'$ 爆炸 |
| `jacobian_unstable` | Jacobian condition 失控 |
| `over_smoothing` | train loss 降不下去且 curvature 低 |
| `no_generalization_gain` | 泛化不优于 baseline |
| `forgetting_not_improved` | 抗遗忘失败 |
| `memory_no_gain` | 显存不如 checkpointing |
| `stats_update_mismatch` | sufficient stats 更新不匹配 full BP |
| `router_no_training_gain` | router fidelity 好但训练无收益 |

---

## 13. 关键成功标准

### 13.1 主线最低成功

满足：

1. AnalyticAdj-DGKAN 接近 FullBP；
2. functional update 在收敛或 final test loss 上优于 AdamW；
3. KAN geometry 更稳定；
4. low-memory sufficient stats 至少比 full BP 省显存。

---

### 13.2 中等成功

满足：

1. diag-to-Sobolev 在视觉任务上稳定优于 AdamW AUC；
2. generalization gap 更低；
3. credit amplification 更低；
4. sufficient stats memory 低于 checkpointing 或接近 checkpointing 但更快；
5. branch utilization 正常。

---

### 13.3 强成功

满足：

1. CIFAR-10 / CIFAR-100 上性能接近强 baseline；
2. memory reduction vs full BP > $30\%$；
3. memory reduction vs checkpointing > $10\%$；
4. wall-clock overhead < $1.5\times$；
5. continual learning forgetting 明显低于 AdamW；
6. small-data / noisy-label / corruption 泛化更好。

---

## 14. Router 的新定位

Learned router 不再是主贡献。

当前定位：

$$
\boxed{
\text{learned router = optional compression / approximation / diagnostic module}
}
$$

进入主线的条件非常严格：

$$
\text{training gain vs IdentityAdj}>3\%-5\%
$$

或：

$$
\text{time / memory reduction vs AnalyticAdj}>20\%
$$

并且：

$$
\text{accuracy gap}<2\%
$$

否则只作为 ablation。

当前更值得保留的是：

$$
\boxed{
\text{FirstOrderDecompAdj}
}
$$

因为它接近 analytic credit、成本低、不需要训练 router。

---

## 15. 研究产出预期

### 15.1 论文主标题方向

候选题目：

> **DG-KAN: Primitive-Adjoint Co-Design and Functional-Space Updates for Efficient Visual Learning**

中文：

> **DG-KAN：面向高效视觉学习的 Primitive-Adjoint 协同设计与函数空间更新**

---

### 15.2 核心贡献

1. 提出 residual DG-KAN 作为 KAN-like full hidden-state visual architecture；
2. 系统验证 KAN primitive 的 analytic adjoint，与 autograd VJP 对齐且更高效；
3. 提出并验证 KAN edge function 的 functional-space update，尤其 diag-to-Sobolev schedule；
4. 证明 functional update 可以改善 credit geometry，包括 $\phi'$、Jacobian condition、credit amplification；
5. 提出 low-memory sufficient statistics 训练路径；
6. 系统分析 learned router 的边界，表明其当前更适合作为 optional approximation，而非主贡献；
7. 在视觉任务上验证收敛、泛化、抗遗忘和 memory tradeoff。

---

## 16. 最终研究叙事

最终研究叙事应该从：

$$
\text{Can we learn a router to replace BP?}
$$

转变为：

$$
\text{Can we design neural primitives whose forward, adjoint, update geometry, and memory statistics are aligned?}
$$

DG-KAN 的回答是：

$$
\boxed{
\text{Yes, KAN-like primitives provide a controlled setting for this co-design.}
}
$$

前向：

$$
 y_j = \sum_i \phi_{ji}(x_i)
$$

反向：

$$
 g_{x_i}=\sum_j g_{y_j}\phi'_{ji}(x_i)
$$

更新：

$$
 a\leftarrow a-\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

统计量：

$$
 s_{jim}=\sum_n g_j^{(n)}B_m(x_i^{(n)})
$$

这个闭环比 learned router 更扎实，也更符合当前实验结果。
