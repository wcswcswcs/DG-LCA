# DG-KAN Stage C-next v4 详细实验计划：把 derivative-aware router 从 one-step 推进到真实 functional-update 训练

> 文件名：`DG-KAN_StageC-next_v4_详细实验计划.md`  
> 对象：DG-KAN / DG-LCA 的 Stage C-next v4 实验执行方案  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`  
> 核心目标：在 v3 已证明 decomposition-aware router 的 fidelity 与 one-step usefulness 后，验证它是否能在真实 functional-update micro-training / dry-run training 中产生稳定收益，并决定 RouterAdj 是否进入 Stage D 主线。

---

## 0. 当前状态与 v4 的定位

Stage C-next v3 已经给出一个非常清楚的新结论：hidden-only router 不够，router 必须显式利用 KAN 的一阶导数结构或 branch correction decomposition。v3 中，span2 correction 几乎完全由一阶 branch correction 解释，cross-term 只有约 $7\%-11\%$；hidden-only feature probe 几乎没有方向信号，而 derivative / first-order sketch 能把 correction direction 的预测质量大幅提高；`decomposition_aware` router 在 4 个固定 setting 上都通过 fidelity gate，并且 one-step hidden update 的收益接近 analytic oracle。

但是 v3 的 micro-training precheck 没有稳定通过。也就是说，当前证据链是：

$$
\text{decomposition-aware router fidelity succeeds}
$$

$$
\text{one-step hidden descent succeeds}
$$

但还不能推出：

$$
\text{router-based functional-update training succeeds}
$$

因此 Stage C-next v4 不再继续盲目扩大 router sweep，也不再继续证明 decomposition-aware router 的静态 fidelity。v4 的核心任务是把问题推进到 **credit-to-update coupling** 和 **真实 functional-update micro-training**：

$$
\boxed{
\text{Does decomposition-aware credit improve actual KAN coefficient functional updates over identity credit?}
}
$$

如果 v4 成功，RouterAdj 可以作为 Stage D 的候选方法进入完整训练对比。如果 v4 失败，Stage D 应以 AnalyticAdj / IdentityAdj 为主，RouterAdj 降级为 ablation 或 later work。

---

## 1. v4 的核心研究问题

v4 只回答四个问题。

第一，v3 中 one-step hidden-state descent 的收益，是否能转化为 KAN coefficient functional update 的收益？也就是说，用不同 credit source 计算出来的 KAN coefficient gradient 是否真的更接近 BP / analytic teacher，并且经过 Sobolev / diag-to-Sobolev preconditioner 后仍然保持正确方向。

第二，decomposition-aware router 与 deterministic first-order decomposition 谁更适合作为 Stage D 的候选？如果 first-order decomposition 已经足够好，learned router 的价值必须来自压缩、泛化或 cost，而不是 fidelity 本身。

第三，router 的训练目标是否应该从 hidden-credit matching 改成 update-aware matching？v3 中 hidden correction fidelity 很好，但 micro-training 不稳定，说明可能需要直接优化 coefficient-gradient 或 preconditioned-update alignment。

第四，router 是否有实际 cost 优势？如果 decomposition-aware router 需要先计算大量 first-order sketch，且成本不低于 analytic adjoint，那么即使 fidelity 很高，也不应作为主线训练方法。

---

## 2. v4 的总体执行顺序

v4 分为六个阶段执行，必须按顺序推进。

1. **V4-A：固定 setting 复核与数据准备**  
   复用 v3 的四个主 setting，但只把其中两个作为主线，另外两个作为 stress / ablation。

2. **V4-B：Credit-to-coefficient-gradient audit**  
   不直接训练，先检查不同 credit source 导出的 KAN coefficient gradient 是否接近 teacher。

3. **V4-C：Functional-update micro-training**  
   使用真正的 KAN coefficient functional update，而不是 v3 的轻量 `prefix_hidden_credit_kan_coeff_adamw` precheck。

4. **V4-D：Update-aware router training**  
   如果 B/C 显示 hidden fidelity 不足以保证 update usefulness，则训练 update-aware router。

5. **V4-E：Cost accounting**  
   比较 AnalyticAdj、FirstOrderDecompAdj、DecompositionAwareRouterAdj、IdentityAdj 的 compute / memory / stats 成本。

6. **V4-F：Stage D readiness dry-run**  
   在短训练预算下比较 FullBP、IdentityAdj、AnalyticAdj、FirstOrderDecompAdj、DecompositionAwareRouterAdj，给 Stage D 做最终决策。

---

## 3. 固定 setting 与数据集

v4 不再大范围搜索数据集和超参。v3 已经证明可以构造 router-worthy setting。v4 的目标是把这些 setting 做透。

### 3.1 主线 setting

v4 主线只使用两个 setting：

| setting | dataset | regime | depth | hidden | basis | span | 用途 |
|---|---|---|---:|---:|---:|---:|---|
| `F8_P_s2` | Fashion-MNIST | Pareto | 8 | 64 | 16 | 2 | 最干净的 stable setting |
| `K8_P_s2` | KMNIST | Pareto | 8 | 64 | 16 | 2 | 更难数据的 stable-ish setting |

这两个 setting 是 v4 的核心。所有主结论优先基于它们。

### 3.2 Stress setting

v4 保留两个 stress setting：

| setting | dataset | regime | depth | hidden | basis | span | 用途 |
|---|---|---|---:|---:|---:|---:|---|
| `F8_A_s2` | Fashion-MNIST | AdamW | 8 | 64 | 16 | 2 | 强 baseline / non-functional regime |
| `K8_A_s2` | KMNIST | active | 8 | 64 | 16 | 2 | 高 branch activity stress |

Stress setting 不作为方法是否通过的唯一依据，但用于判断方法是否泛化到更强 correction / 更高 amplification 场景。

### 3.3 Seed 与 split

所有主实验使用：

$$
N_{seed}=5
$$

每个 setting 至少包含：

| split | 样本数 | 用途 |
|---|---:|---|
| train-router | 8192 | router 训练 |
| val-router | 2048 | router early stopping |
| test-router | 2048 | fidelity / update audit |
| micro-train | 4096 | micro-training |
| micro-val | 1024 | micro-training validation |
| micro-test | 1024 | final test |

如果显存或时间受限，可先用 3 seeds 做 smoke，但正式结论必须报告 5 seeds。

---

## 4. 关键 notation

### 4.1 Span2 VJP

设两个连续 residual DG-KAN blocks 为：

$$
h_{k+1}=F_k(h_k)
$$

$$
h_{k+2}=F_{k+1}(h_{k+1})
$$

span2 teacher credit 为：

$$
g_k^{teacher}=J_{k:k+2}^{\top}g_{k+2}
$$

其中：

$$
J_{k:k+2}=D(F_{k+1}\circ F_k)(h_k)
$$

identity baseline 为：

$$
g_k^{id}=g_{k+2}
$$

span correction 为：

$$
\Delta g^{teacher}=g_k^{teacher}-g_{k+2}
$$

### 4.2 First-order decomposition

v3 已经发现：

$$
\Delta g^{teacher}
\approx
\Delta g_0+\Delta g_1
$$

其中 $\Delta g_0$ 与 $\Delta g_1$ 是两个 block 的一阶 branch correction。记：

$$
\Delta g^{first}=\Delta g_0+\Delta g_1
$$

cross-term 为：

$$
\Delta g^{cross}=\Delta g^{teacher}-\Delta g^{first}
$$

v3 的结果说明：

$$
\frac{\|\Delta g^{cross}\|}{\|\Delta g^{teacher}\|}\approx0.07\sim0.11
$$

### 4.3 v4 的 credit sources

v4 比较以下 credit source：

| credit source | 定义 | 作用 |
|---|---|---|
| `full_bp` | full autograd BP | oracle baseline |
| `analytic_span` | exact analytic / hybrid analytic span VJP | main teacher |
| `identity` | $g_{k+2}$ | strong baseline |
| `first_order_decomp` | $g_{k+2}+\Delta g^{first}$ | deterministic approximation |
| `decomp_router` | $g_{k+2}+\widehat{\Delta g}$ | learned candidate |
| `hidden_only_router` | hidden-only learned router | negative / ablation |
| `random` | random credit matched norm | negative control |

---

## 5. V4-A：固定 setting 复核与数据准备

### 5.1 目标

V4-A 不是新实验结论，而是为了保证 v4 使用的 setting 与 v3 一致。每次 v4 运行前，必须重新复核 checkpoint 的 performance、branch activity、basis coverage、credit geometry。如果 setting 发生漂移，后续结果不能和 v3 对比。

### 5.2 必须记录的 W&B config

每个 run 必须记录：

| config key | 含义 |
|---|---|
| `exp/stage` | `stage_c_next_v4` |
| `exp/phase` | `A_setting_audit`, `B_grad_audit`, `C_micro_train`, `D_update_router`, `E_cost`, `F_dry_run` |
| `data/name` | `fashion_mnist`, `kmnist` |
| `setting/name` | `F8_P_s2`, `K8_P_s2`, `F8_A_s2`, `K8_A_s2` |
| `setting/regime` | `adamw`, `pareto`, `active` |
| `model/depth` | 8 |
| `model/hidden_dim` | 64 |
| `model/basis_count` | 16 |
| `span/length` | 2 |
| `update/coeff_lr` | checkpoint 的 KAN coefficient lr |
| `update/rest_lr` | non-KAN AdamW lr |
| `update/type` | `diag_to_sobolev`, `adamw` |
| `kan/alpha` | residual branch scale |
| `seed` | seed id |

### 5.3 必须记录的 metrics

| metric key | 含义 |
|---|---|
| `v4/audit/test_acc` | checkpoint test acc |
| `v4/audit/test_loss` | checkpoint test loss |
| `v4/audit/gap_vs_adamw` | test acc gap vs AdamW |
| `v4/audit/branch_over_adamw` | branch / AdamW |
| `v4/audit/phi_prime_p95_over_adamw` | $\phi'_{p95}$ / AdamW |
| `v4/audit/jcond_over_adamw` | Jacobian condition / AdamW |
| `v4/audit/identity_relerr_span2` | span2 identity relerr |
| `v4/audit/correction_ratio_span2` | span2 correction ratio |
| `v4/audit/amp_p95_span2` | span2 amplification p95 |
| `v4/audit/active_basis_fraction` | active basis fraction |
| `v4/audit/out_of_grid_fraction` | out-of-grid fraction |

### 5.4 通过标准

主线 setting 必须满足：

$$
\text{test gap}<2\%
$$

$$
\text{identity relerr}_{span2}>0.25
$$

$$
\text{correction ratio}_{span2}>0.30
$$

$$
\text{amp p95}_{span2}<1.7
$$

如果 `F8_P_s2` 或 `K8_P_s2` 不满足这些条件，需要回到 v2/v3 的 Pareto search 重新生成 checkpoint，不进入 v4 主实验。

### 5.5 可视化

W&B Page `V4-A Setting Map` 包含：

1. setting × seed 的 test accuracy heatmap；
2. branch / AdamW bar chart；
3. identity relerr / correction ratio scatter；
4. amplification p95 by setting；
5. basis active fraction 和 out-of-grid fraction table；
6. Pareto gate pass/fail scorecard。

---

## 6. V4-B：Credit-to-coefficient-gradient audit

### 6.1 为什么要做 B

v3 说明 decomposition-aware credit 在 hidden space 上很准，并且 one-step hidden descent 有收益。但 KAN 参数真正更新的是 edge coefficients $a_{jim}$。因此 v4 必须检查：

$$
\text{hidden credit good}
\Rightarrow
\text{coefficient gradient good?}
$$

对 KAN edge function：

$$
\phi_{ji}(t)=\sum_m a_{jim}B_m(t)
$$

给定输出端 credit $g_j^{(n)}$，coefficient gradient 为：

$$
\nabla_{a_{jim}}\mathcal L
=
\frac{1}{B}\sum_n g_j^{(n)}B_m(x_i^{(n)})
$$

所以即使 hidden credit cosine 很高，也可能在经过 basis statistics 和 functional preconditioner 后，coefficient update 方向发生变化。

### 6.2 实验方法

固定 checkpoint 和 audit batch，分别用不同 credit source 计算 KAN coefficient gradient：

$$
\nabla_a^{source}
$$

teacher 为：

$$
\nabla_a^{teacher}
$$

然后比较未预条件的 coefficient gradient 和预条件后的 update direction。

Sobolev / diag-to-Sobolev 预条件方向：

$$
d_a^{source}=(M+\rho I)^{-1}\nabla_a^{source}
$$

teacher 预条件方向：

$$
d_a^{teacher}=(M+\rho I)^{-1}\nabla_a^{teacher}
$$

### 6.3 Credit sources

V4-B 比较：

| source | 说明 |
|---|---|
| `identity` | $g_{k+2}$ |
| `first_order_decomp` | $g_{k+2}+\Delta g^{first}$ |
| `decomp_router` | $g_{k+2}+\widehat{\Delta g}$ |
| `hidden_only_router` | hidden-only router output |
| `random` | norm matched random |
| `analytic_span` | teacher / oracle |

### 6.4 必须记录的 metrics

Hidden-credit metrics：

| metric key | 含义 |
|---|---|
| `v4/b/hidden_credit_cos` | hidden credit vs teacher cosine |
| `v4/b/hidden_credit_relerr` | hidden credit relative error |
| `v4/b/correction_cos` | correction direction cosine |
| `v4/b/correction_relerr` | correction relative error |
| `v4/b/correction_norm_ratio` | correction norm ratio |

Coefficient-gradient metrics：

| metric key | 含义 |
|---|---|
| `v4/b/coeff_grad_cos` | $\nabla_a^{source}$ vs $\nabla_a^{teacher}$ cosine |
| `v4/b/coeff_grad_relerr` | coefficient gradient relative error |
| `v4/b/coeff_grad_norm_ratio` | coefficient gradient norm ratio |
| `v4/b/coeff_grad_cos_p05` | low-percentile cosine |
| `v4/b/coeff_grad_relerr_p95` | high-percentile relerr |

Preconditioned-update metrics：

| metric key | 含义 |
|---|---|
| `v4/b/precond_update_cos` | $d_a^{source}$ vs $d_a^{teacher}$ cosine |
| `v4/b/precond_update_relerr` | preconditioned update relative error |
| `v4/b/precond_update_norm_ratio` | preconditioned update norm ratio |
| `v4/b/precond_update_cos_p05` | low-percentile cosine |
| `v4/b/precond_update_relerr_p95` | high-percentile relerr |
| `v4/b/precond_noise_amplification` | preconditioner 对 gradient error 的放大 |

Functional metric metrics：

| metric key | 含义 |
|---|---|
| `v4/b/metric_condition` | $\kappa(M+\rho I)$ |
| `v4/b/metric_eig_min` | min eigenvalue |
| `v4/b/metric_eig_max` | max eigenvalue |
| `v4/b/update_norm_function` | function-space update norm |
| `v4/b/update_norm_coeff` | coefficient update norm |

### 6.5 关键诊断

V4-B 必须计算两个 gap：

$$
\Delta_{hidden\rightarrow coeff}
=
\cos_{hidden}-\cos_{coeff\_grad}
$$

$$
\Delta_{coeff\rightarrow update}
=
\cos_{coeff\_grad}-\cos_{precond\_update}
$$

如果 $\Delta_{hidden\rightarrow coeff}$ 大，说明 hidden credit fidelity 没有很好转化成 coefficient gradient fidelity。如果 $\Delta_{coeff\rightarrow update}$ 大，说明 Sobolev / diag-to-Sobolev preconditioner 放大了 credit error。

### 6.6 通过标准

`decomp_router` 在主线 setting 上应满足：

$$
\text{coeff grad cosine}>0.90
$$

$$
\text{precond update cosine}>0.85
$$

$$
0.8<\text{precond update norm ratio}<1.2
$$

并且相比 identity：

$$
\text{precond update relerr reduction}>20\%
$$

如果 B 不通过，C/D/E/F 不应把 router 作为主线，只能保留 `first_order_decomp` 和 `analytic_span`。

### 6.7 可视化

W&B Page `V4-B Credit to Update` 包含：

1. hidden credit cosine vs coefficient gradient cosine scatter；
2. coefficient gradient cosine vs preconditioned update cosine scatter；
3. source × setting 的 preconditioned update cosine heatmap；
4. source × setting 的 preconditioned update relerr heatmap；
5. identity / first-order / router / analytic 的 update norm ratio bar chart；
6. preconditioner condition vs update error scatter；
7. error decomposition waterfall：hidden error、coefficient gradient error、preconditioned update error。

---

## 7. V4-C：真正的 functional-update micro-training

### 7.1 为什么 v4 的 micro-training 要重做

v3 的 micro-training 是轻量 precheck，并不是完整 functional-update micro-training。v4 必须使用和 Stage D 一致的 KAN coefficient functional update：

$$
a\leftarrow a-\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

而不是只用 AdamW 更新 prefix KAN coefficients。v4-C 的目标是检验：不同 credit source 进入真实 functional update 后，是否能稳定改善 validation loss、accuracy、branch geometry 和 update quality。

### 7.2 训练范围

v4-C 使用两级 micro-training。

#### Level 1：Span-local micro-training

固定整个模型，只更新一个 span 内的 KAN coefficients。对于 span2，更新 blocks $k$ 和 $k+1$ 的 KAN edge coefficients，其余参数冻结。

这个级别用于回答：

$$
\text{credit source 是否能正确更新指定 span?}
$$

#### Level 2：Prefix micro-training

更新从输入到某个 layer 的 KAN coefficients，例如前 $L_p$ 个 blocks：

$$
L_p\in\{2,4\}
$$

stem、head、LayerNorm、non-KAN 参数保持 AdamW 或冻结，具体作为 ablation 比较。

这个级别更接近 Stage D，但仍然比完整训练便宜。

### 7.3 方法比较

v4-C 必须比较以下方法：

| method | credit source | update | 目的 |
|---|---|---|---|
| `identity_func` | identity | diag-to-Sobolev | 强 baseline |
| `analytic_func` | analytic span adjoint | diag-to-Sobolev | oracle teacher |
| `first_order_func` | first-order decomposition | diag-to-Sobolev | deterministic approximation |
| `decomp_router_func` | decomposition-aware router | diag-to-Sobolev | learned candidate |
| `hidden_router_func` | hidden-only router | diag-to-Sobolev | negative learned baseline |
| `random_func` | random credit | diag-to-Sobolev | negative control |
| `full_bp_func` | full BP credit | diag-to-Sobolev | upper baseline |

### 7.4 Training budgets

每个 setting 运行：

$$
T\in\{50,100,200,500\}\ \text{steps}
$$

Micro learning rate：

$$
\eta_{micro}\in\{3\times10^{-4},10^{-3},3\times10^{-3}\}
$$

正式主结果先使用：

$$
\eta_{micro}=10^{-3}
$$

每个方法 5 seeds。

### 7.5 必须记录的 metrics

Training metrics：

| metric key | 含义 |
|---|---|
| `v4/c/train_loss_curve` | micro-train loss curve |
| `v4/c/val_loss_curve` | micro-val loss curve |
| `v4/c/test_acc_final` | final test acc |
| `v4/c/val_loss_auc` | validation loss AUC |
| `v4/c/steps_to_target` | 达到 target val loss 的 step |
| `v4/c/gain_vs_identity_val_loss` | 相对 identity 的 val loss gain |
| `v4/c/gain_vs_identity_auc` | 相对 identity 的 AUC gain |
| `v4/c/gap_vs_analytic` | 相对 analytic 的 final gap |

Descent metrics：

| metric key | 含义 |
|---|---|
| `v4/c/pred_descent` | predicted descent |
| `v4/c/actual_descent` | actual descent |
| `v4/c/descent_ratio` | actual / predicted |
| `v4/c/sign_agreement_rate` | sign agreement rate |
| `v4/c/bad_step_count` | predicted negative actual positive count |
| `v4/c/bad_step_total_increase` | bad steps loss increase 总量 |

Update metrics：

| metric key | 含义 |
|---|---|
| `v4/c/coeff_update_cos_vs_teacher` | coefficient update vs teacher cosine |
| `v4/c/precond_update_cos_vs_teacher` | preconditioned update vs teacher cosine |
| `v4/c/update_norm_coeff` | coefficient update norm |
| `v4/c/update_norm_function` | function update norm |
| `v4/c/update_norm_ratio_vs_teacher` | update norm ratio vs teacher |
| `v4/c/update_clip_rate` | update clipping rate |

KAN geometry metrics：

| metric key | 含义 |
|---|---|
| `v4/c/branch_ratio` | branch ratio |
| `v4/c/branch_over_adamw` | branch / AdamW |
| `v4/c/phi_prime_p95` | $\phi'_{p95}$ |
| `v4/c/phi_prime_max` | $\phi'_{max}$ |
| `v4/c/curvature` | $\int |\phi''|^2$ |
| `v4/c/jacobian_condition` | Jacobian condition |
| `v4/c/credit_amplification_p95` | credit amplification p95 |

### 7.6 成功标准

For `decomp_router_func`，主线通过条件是：

$$
\text{gain vs identity AUC}>5\%
$$

$$
\text{final val loss gain vs identity}>3\%
$$

$$
\text{gap vs analytic}<2\%
$$

$$
\text{sign agreement rate}>0.75
$$

并且：

$$
\text{bad step count}\leq \text{identity bad step count}+1
$$

如果 `first_order_func` 通过而 `decomp_router_func` 不通过，则 Stage D 可以使用 deterministic first-order decomposition，但 learned router 继续作为 ablation。

### 7.7 可视化

W&B Page `V4-C Functional Micro-training` 包含：

1. validation loss curves by credit source；
2. val loss AUC bar chart；
3. gain vs identity bar chart；
4. gap vs analytic bar chart；
5. predicted vs actual descent scatter；
6. coefficient update cosine over steps；
7. preconditioned update cosine over steps；
8. branch ratio / $\phi'$ / Jacobian condition over steps；
9. bad step count table；
10. method × budget heatmap，颜色为 val loss gain。

---

## 8. V4-D：Update-aware router training

### 8.1 为什么需要 D

如果 V4-B 或 V4-C 显示 hidden correction fidelity 很高，但 coefficient update quality 不稳定，说明 router 训练目标不应该只匹配 hidden credit。它需要直接对 coefficient gradient 或 preconditioned update 负责。

### 8.2 Router 目标

原 hidden correction loss：

$$
\mathcal L_{hidden}
=
1-\\cos(\widehat{\Delta g},\Delta g^{teacher})
$$

新增 coefficient-gradient loss：

$$
\mathcal L_{coeff}
=
1-
\cos(\nabla_a^{router},\nabla_a^{teacher})
$$

新增 preconditioned-update loss：

$$
\mathcal L_{update}
=
1-
\cos(d_a^{router},d_a^{teacher})
$$

其中：

$$
d_a=(M+\rho I)^{-1}\nabla_a
$$

总 loss：

$$
\mathcal L_{router}
=
\lambda_h\mathcal L_{hidden}
+
\lambda_c\mathcal L_{coeff}
+
\lambda_u\mathcal L_{update}
+
\lambda_n\mathcal L_{norm}
$$

建议 sweep：

| setting | $\lambda_h$ | $\lambda_c$ | $\lambda_u$ | 用途 |
|---|---:|---:|---:|---|
| hidden-only | 1 | 0 | 0 | v3 baseline |
| coeff-aware | 0.5 | 1 | 0 | gradient alignment |
| update-aware | 0.3 | 0.5 | 1 | functional update alignment |
| update-dominant | 0.1 | 0.3 | 2 | 强化最终 update |

### 8.3 Router variants

只比较三类 router，避免 v4 爆炸：

| router | 说明 |
|---|---|
| `decomposition_aware_v3` | v3 baseline |
| `decomposition_aware_coeff_loss` | 加 coefficient-gradient loss |
| `decomposition_aware_update_loss` | 加 preconditioned-update loss |

hidden-only router 不再作为主实验，只作为负对照。

### 8.4 记录指标

除了 V4-B / V4-C 的指标外，新增：

| metric key | 含义 |
|---|---|
| `v4/d/router_hidden_loss` | hidden correction loss |
| `v4/d/router_coeff_loss` | coefficient gradient loss |
| `v4/d/router_update_loss` | preconditioned update loss |
| `v4/d/coeff_grad_cos_train` | train coeff gradient cosine |
| `v4/d/coeff_grad_cos_val` | val coeff gradient cosine |
| `v4/d/precond_update_cos_train` | train preconditioned update cosine |
| `v4/d/precond_update_cos_val` | val preconditioned update cosine |
| `v4/d/micro_gain_after_router` | 用该 router 做 micro-training 的 gain |
| `v4/d/overfit_gap_hidden` | hidden train-val gap |
| `v4/d/overfit_gap_update` | update train-val gap |

### 8.5 通过标准

Update-aware router 必须相较 v3 router 满足：

$$
\text{precond update cosine improvement}>0.05
$$

并且：

$$
\text{micro-training gain improvement}>3\%
$$

如果 D 不能提高 micro-training gain，则说明 router fidelity 不是主要瓶颈，Stage D 不应再继续 router 主线。

### 8.6 可视化

W&B Page `V4-D Update-aware Router` 包含：

1. hidden cosine vs coeff gradient cosine scatter；
2. coeff gradient cosine vs precond update cosine scatter；
3. router loss component curves；
4. update-aware vs hidden-only micro gain bar chart；
5. train-val overfit gap by loss type；
6. loss weights sweep heatmap；
7. failure examples table。

---

## 9. V4-E：Cost accounting

### 9.1 目标

v3 的 decomposition-aware router 使用 first-order sketch。它高 fidelity，但可能不便宜。v4 必须回答：

$$
\boxed{
\text{Is DecompositionAwareRouter cheaper or more useful than analytic adjoint?}
}
$$

如果 router 不便宜，也没有训练收益，就不应进入 Stage D 主线。

### 9.2 比较方法

| method | 计算内容 |
|---|---|
| `identity` | no correction |
| `analytic_span` | exact analytic span VJP |
| `first_order_decomp` | compute layerwise first-order correction |
| `decomp_router` | first-order sketch + router |
| `autograd_span` | full autograd span VJP |

### 9.3 记录指标

Time metrics：

| metric key | 含义 |
|---|---|
| `v4/e/time_forward_ms` | forward time |
| `v4/e/time_credit_ms` | credit computation time |
| `v4/e/time_first_order_sketch_ms` | first-order sketch time |
| `v4/e/time_router_ms` | router time |
| `v4/e/time_functional_update_ms` | functional update time |
| `v4/e/time_total_step_ms` | total step time |
| `v4/e/time_ratio_vs_analytic` | method / analytic time |
| `v4/e/time_ratio_vs_identity` | method / identity time |

Memory metrics：

| metric key | 含义 |
|---|---|
| `v4/e/memory_params_mb` | params memory |
| `v4/e/memory_optimizer_mb` | optimizer memory |
| `v4/e/memory_interfaces_mb` | interface memory |
| `v4/e/memory_credit_mb` | credit buffers |
| `v4/e/memory_first_order_stats_mb` | first-order stats memory |
| `v4/e/memory_router_params_mb` | router params |
| `v4/e/memory_router_activations_mb` | router activations |
| `v4/e/memory_peak_allocated_mb` | peak allocated |
| `v4/e/memory_ratio_vs_analytic` | method / analytic memory |

Accuracy / usefulness metrics：

| metric key | 含义 |
|---|---|
| `v4/e/utility_val_loss_gain` | val loss gain vs identity |
| `v4/e/utility_auc_gain` | AUC gain vs identity |
| `v4/e/utility_per_ms` | AUC gain / extra ms |
| `v4/e/utility_per_mb` | AUC gain / extra MB |

### 9.4 成本成功标准

RouterAdj 只有在以下条件至少满足一组时才有工程意义。

#### 条件 A：比 analytic 便宜且效果接近

$$
\text{time ratio vs analytic}<0.8
$$

$$
\text{memory ratio vs analytic}<0.8
$$

$$
\text{micro-training gap vs analytic}<2\%
$$

#### 条件 B：比 identity 明显有用，且开销可控

$$
\text{AUC gain vs identity}>5\%
$$

$$
\text{time overhead vs identity}<1.3\times
$$

$$
\text{memory overhead vs identity}<1.3\times
$$

如果两组条件都不满足，RouterAdj 不进入 Stage D 主线。

### 9.5 可视化

W&B Page `V4-E Cost Utility` 包含：

1. method vs time stacked bar；
2. method vs memory stacked bar；
3. utility per ms bar chart；
4. utility per MB bar chart；
5. accuracy / loss vs time Pareto；
6. memory vs performance Pareto；
7. router overhead breakdown pie chart。

---

## 10. V4-F：Stage D readiness dry-run

### 10.1 目标

V4-F 是 Stage D 前的最终决策实验。它不是完整 Stage D，但要比 micro-training 更接近真实训练。目标是比较几种 credit source 在短训练预算下是否稳定。

### 10.2 方法

| method | credit | update | 地位 |
|---|---|---|---|
| `FullBP-Func` | full BP / autograd | diag-to-Sobolev | upper baseline |
| `IdentityAdj-Func` | identity credit | diag-to-Sobolev | strong baseline |
| `AnalyticAdj-Func` | analytic span / local adjoint | diag-to-Sobolev | main candidate |
| `FirstOrderDecomp-Func` | deterministic first-order correction | diag-to-Sobolev | deterministic candidate |
| `DecompRouter-Func` | learned decomposition-aware router | diag-to-Sobolev | learned candidate |
| `HiddenRouter-Func` | hidden-only router | diag-to-Sobolev | negative router |

### 10.3 Training protocol

每个 method 从同一 initialization 开始，训练：

$$
T\in\{500,1000,2000\}\ \text{steps}
$$

数据集：

| dataset | setting |
|---|---|
| Fashion-MNIST | depth8, hidden64, basis16 |
| KMNIST | depth8, hidden64, basis16 |

所有方法必须使用相同 batch order、相同 seed、相同 update schedule。

### 10.4 记录指标

Performance：

| metric key | 含义 |
|---|---|
| `v4/f/train_loss` | train loss curve |
| `v4/f/val_loss` | val loss curve |
| `v4/f/test_acc` | test accuracy |
| `v4/f/val_auc` | val loss AUC |
| `v4/f/gap_vs_fullbp` | performance gap vs FullBP |
| `v4/f/gap_vs_analytic` | performance gap vs AnalyticAdj |
| `v4/f/gain_vs_identity` | gain vs IdentityAdj |

Stability：

| metric key | 含义 |
|---|---|
| `v4/f/bad_step_count` | bad steps |
| `v4/f/loss_spike_count` | loss spikes |
| `v4/f/nan_count` | NaN / Inf |
| `v4/f/grad_norm` | gradient norm |
| `v4/f/update_norm` | update norm |

Geometry：

| metric key | 含义 |
|---|---|
| `v4/f/branch_ratio` | branch ratio |
| `v4/f/phi_prime_p95` | $\phi'_{p95}$ |
| `v4/f/jacobian_condition` | Jacobian condition |
| `v4/f/credit_amplification_p95` | credit amplification p95 |
| `v4/f/correction_ratio` | correction ratio |

Credit fidelity during training：

| metric key | 含义 |
|---|---|
| `v4/f/credit_cos_vs_analytic` | credit source vs analytic cosine |
| `v4/f/credit_relerr_vs_analytic` | credit source relative error |
| `v4/f/precond_update_cos_vs_analytic` | update direction cosine |
| `v4/f/precond_update_relerr_vs_analytic` | update direction relerr |

Cost：

| metric key | 含义 |
|---|---|
| `v4/f/step_time_ms` | step time |
| `v4/f/peak_memory_mb` | peak memory |
| `v4/f/samples_per_sec` | throughput |
| `v4/f/time_to_target` | time to target |

### 10.5 成功标准

`DecompRouter-Func` 进入 Stage D 主线的条件：

$$
\text{gain vs IdentityAdj AUC}>5\%
$$

$$
\text{gap vs AnalyticAdj}<2\%
$$

$$
\text{step time overhead vs IdentityAdj}<1.3\times
$$

$$
\text{peak memory overhead vs IdentityAdj}<1.3\times
$$

并且：

$$
\text{no additional divergence or loss spike}
$$

如果 `FirstOrderDecomp-Func` 成功而 `DecompRouter-Func` 不成功，Stage D 使用 FirstOrderDecomp 作为 deterministic approximation，RouterAdj 保留为 ablation。

如果 `IdentityAdj-Func` 与 `AnalyticAdj-Func` 差距小于 $1\%$，并且 RouterAdj 不能显著超过 IdentityAdj，则 Stage D 主线可以简化为：

$$
\boxed{
\text{IdentityAdj / AnalyticAdj + functional update + memory accounting}
}
$$

---

## 11. v4 Dashboard 总设计

v4 建议建立 6 个 W&B dashboard 页面。

### Page 1：V4 Overview

展示每个 setting 的最终结论：

1. setting pass/fail table；
2. method × stage scorecard；
3. router 是否进入 Stage D 主线；
4. identity / analytic / first-order / router 总体比较表。

### Page 2：Credit to Update Audit

展示 hidden credit、coefficient gradient、preconditioned update 之间的关系：

1. hidden cosine vs coeff gradient cosine；
2. coeff gradient cosine vs precond update cosine；
3. source × setting heatmap；
4. update norm ratio distribution；
5. preconditioner condition vs update error。

### Page 3：Functional Micro-training

展示 micro-training 是否真正有收益：

1. val loss curves；
2. val AUC bar；
3. gain vs identity bar；
4. gap vs analytic bar；
5. method × training budget heatmap；
6. predicted vs actual descent scatter。

### Page 4：Update-aware Router

展示 router loss 改造是否有效：

1. loss component curves；
2. hidden / coeff / update cosine curves；
3. micro gain by router objective；
4. train-val overfit gap；
5. router objective weight heatmap。

### Page 5：Cost Utility

展示工程价值：

1. time stacked bar；
2. memory stacked bar；
3. utility per ms；
4. utility per MB；
5. performance-memory Pareto；
6. performance-time Pareto。

### Page 6：Failure Analysis

展示所有失败模式：

1. failure table；
2. update mismatch cases；
3. router overfit cases；
4. micro no-gain cases；
5. cost no-gain cases；
6. recommended action table。

---

## 12. Failure table

每个 failure 必须结构化记录。

| failure type | 触发条件 | 解释 |
|---|---|---|
| `setting_not_reproduced` | V4-A setting 不满足 gate | checkpoint 不可用 |
| `hidden_to_coeff_drop` | hidden cosine 高但 coeff cosine 低 | hidden credit fidelity 未转化 |
| `precond_amplifies_error` | coeff cosine 高但 update cosine 低 | metric 放大 credit error |
| `micro_no_gain` | micro gain vs identity <= 0 | 训练无收益 |
| `micro_unstable` | loss spike / bad steps 增加 | credit source 不稳定 |
| `router_update_loss_no_help` | update-aware loss 未提升 micro gain | router loss 不是瓶颈 |
| `router_cost_too_high` | cost 高且无性能收益 | 工程价值不足 |
| `first_order_beats_router` | deterministic first-order 优于 learned router | router 不必要 |
| `identity_still_enough` | identity 与 analytic gap < 1% | learned correction 价值弱 |

Failure table 字段：

| field | 含义 |
|---|---|
| `failure/type` | failure 类型 |
| `failure/phase` | A/B/C/D/E/F |
| `failure/setting` | setting 名 |
| `failure/method` | 方法 |
| `failure/seed` | seed |
| `failure/metric_name` | metric |
| `failure/metric_value` | 数值 |
| `failure/threshold` | 阈值 |
| `failure/diagnosis` | 诊断 |
| `failure/recommended_action` | 建议动作 |

---

## 13. 第一轮最小执行包

为了避免 v4 过大，第一轮只做以下最小包。

### Run 1：V4-B Credit-to-update audit

Settings：

```text
F8_P_s2, K8_P_s2
```

Credit sources：

```text
identity, first_order_decomp, decomp_router, analytic_span, random
```

目标：判断 decomp_router 的 hidden fidelity 是否转化为 coefficient gradient / preconditioned update fidelity。

通过标准：

$$
\text{precond update cosine}>0.85
$$

$$
\text{precond update relerr reduction vs identity}>20\%
$$

---

### Run 2：V4-C Functional-update micro-training

Settings：

```text
F8_P_s2, K8_P_s2
```

Methods：

```text
identity_func
analytic_func
first_order_func
decomp_router_func
random_func
```

Budgets：

```text
steps = 50,100,200
```

通过标准：

$$
\text{decomp router AUC gain vs identity}>5\%
$$

$$
\text{gap vs analytic}<2\%
$$

---

### Run 3：V4-E Cost accounting

比较：

```text
identity
analytic_span
first_order_decomp
decomp_router
autograd_span
```

通过标准：router 必须要么比 analytic 便宜，要么比 identity 明显有收益且开销小于 $1.3\times$。

---

### Run 4：V4-F Stage D readiness dry-run

Settings：

```text
F8_P_s2, K8_P_s2
```

Methods：

```text
FullBP-Func
IdentityAdj-Func
AnalyticAdj-Func
FirstOrderDecomp-Func
DecompRouter-Func
```

Training budget：

```text
500, 1000 steps
```

通过标准：DecompRouter 必须在 AUC、final val loss、cost 三方面同时接近 AnalyticAdj 并超过 IdentityAdj。

---

## 14. v4 之后的决策规则

### 情况 1：decomp router 通过 B/C/F

如果 decomp router 的 coefficient update fidelity、micro-training gain、dry-run gain 都通过，则 Stage D 加入：

$$
\boxed{\text{DecompositionAwareRouterAdj-DGKAN}}
$$

作为正式 candidate。

### 情况 2：first-order decomp 通过，但 router 不通过

Stage D 使用：

$$
\boxed{\text{FirstOrderDecompAdj-DGKAN}}
$$

router 只作为 ablation。

### 情况 3：analytic 通过，identity 已经足够

如果：

$$
\text{IdentityAdj gap vs AnalyticAdj}<1\%
$$

且 router 无明显 gain，则 Stage D 主线简化为：

$$
\boxed{
\text{IdentityAdj / AnalyticAdj + functional update + memory accounting}
}
$$

### 情况 4：hidden fidelity 好但 update fidelity 差

说明 router 训练目标不对。优先研究 update-aware loss，而不是继续扩大 router architecture。

### 情况 5：router fidelity 和 update fidelity 都好，但 cost 不划算

说明 router 可作为理论结果，但不应作为工程主线。Stage D 以 analytic / first-order 为主。

---

## 15. 最终建议

Stage C-next v4 是 learned router 的关键转折实验。它不是继续证明 router 静态 fidelity，而是直接验证：

$$
\boxed{
\text{router credit can improve actual functional updates over identity credit}
}
$$

v4 成功，则 RouterAdj 进入 Stage D 候选。v4 失败，则应停止把 learned router 作为主线，转向：

$$
\text{AnalyticAdj}
+
\text{FirstOrderDecompAdj}
+
\text{IdentityAdj}
+
\text{functional update}
+
\text{memory accounting}
$$

这并不是削弱 DG-KAN 路线。相反，它会让 DG-KAN 的主贡献更清楚：

1. KAN primitive 的 analytic adjoint 是可靠的；
2. functional-space update 对 KAN coefficients 有价值；
3. residual DG-KAN 的 identity-dominant credit transport 很稳定；
4. first-order decomposition 可能比 learned hidden-router 更适合当前架构；
5. learned router 只有在能转化为 functional-update training gain 且成本合理时，才值得进入主线。
