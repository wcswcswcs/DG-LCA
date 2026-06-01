# DG-KAN Stage C 详细实验计划：Analytic KAN Adjoint 与 Learned Local Router

> 文件名：`DG-KAN_StageC_详细实验计划.md`  
> 对象：DG-KAN / DG-LCA 的 Stage C 实验执行方案  
> 版本：v1.0  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`  
> 核心目标：在 Stage A/B 已验证 DG-KAN 可训练、functional update 有价值的基础上，系统验证 **KAN analytic adjoint 是否正确低成本**，以及 **learned local credit router 是否能在完整 hidden-space 上逼近 KAN local VJP**。

---

## 0. 为什么 Stage C 要重新调整

Stage A 和 Stage B 已经给出了几个会直接影响 Stage C 的结论。

第一，DG-KAN 架构本身已经被证明是可训练的，并且 residual / LayerNorm / KAN-like block 组合显著改善了普通 KAN 的 Jacobian condition。普通 KAN 的 Jacobian condition 可以达到 $10^8$ 量级，而 residual DG-KAN 可以降到个位数或接近个位数。

第二，functional-space update 对 KAN edge function 是有价值的，但它不是通用 optimizer。实验已经说明，KAN coefficients 适合用 `diag`、`sobolev` 或 `diag_to_sobolev`；非 KAN 参数仍然应使用 AdamW / Adam。粗暴地把非 KAN 参数也换成 functional update 或弱 SGD 会欠拟合。

第三，MNIST small 上初始 hybrid failure 已经被解释为 KAN branch under-active，而不是泛化失败或 basis coverage 失败。提高 coefficient learning rate 后可以修复精度；中等 coefficient learning rate 下可以得到一个 Pareto regime，用小于 $1\%$ 的 accuracy gap 换取更低的 Jacobian condition 和更低的 $\phi'_{p95}$。

因此 Stage C 不能只是泛泛地问：

$$
\text{router 能不能学 VJP?}
$$

而应该问一个更具体、更能连接 Stage B 发现的问题：

$$
\boxed{
\text{functional update 造成的不同 branch activity regime，是否会改变 local credit transport 的稳定性和 router 可学习性？}
}
$$

Stage C 的核心目标是验证下面这条链条：

$$
\text{functional update}
\rightarrow
\text{healthier KAN geometry}
\rightarrow
\text{more stable credit transport}
\rightarrow
\text{easier local router}
$$

如果这条链条成立，那么 Stage B 的几何收益就不只是“指标好看”，而是会直接帮助 DG-LCA 的 credit assignment 主线。

---

## 1. Stage C 的总目标

Stage C 分成六个连续部分：

1. **C0：Regime checkpoint 准备与复核**  
   固定并复现 Stage B 中发现的 conservative、Pareto、accuracy-parity 三个 branch activity regime。

2. **C1：Core KAN analytic adjoint 正确性验证**  
   验证 KAN primitive 的解析 VJP 是否与 autograd VJP 一致。

3. **C2：Full residual DG-KAN block adjoint 验证**  
   在包含 residual、LayerNorm、residual scale 的完整 block 上验证 analytic / hybrid analytic VJP。

4. **C3：Credit geometry audit across regimes**  
   比较 conservative / Pareto / accuracy-parity 下 credit amplification、Jacobian spectrum、credit rank、noise sensitivity 和 adjoint residual。

5. **C4：Learned local router distillation**  
   训练条件线性 router 逼近 analytic / autograd teacher VJP，并比较不同 branch regime 下 router 的可学习性。

6. **C5：One-step local update sanity check**  
   用 analytic credit 或 router credit 做局部 functional update，检查 predicted descent 与 actual descent 是否一致，为 Stage D joint training 做门槛测试。

Stage C 不直接验证 low-memory sufficient statistics，也不直接做 full joint training。那些属于 Stage D / E。Stage C 的结束条件是：

$$
\text{analytic adjoint correctness}
+
\text{credit geometry audit}
+
\text{router fidelity}
+
\text{one-step local update sanity}
$$

全部有清楚结果。

---

## 2. Stage C 的核心 notation

### 2.1 DG-KAN residual block

当前分类实验的主 block 是：

$$
h_{k+1}
=
h_k
+
\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

记：

$$
x_k=\operatorname{LN}(h_k)
$$

$$
u_k=\operatorname{KAN}_k(x_k)
$$

则：

$$
h_{k+1}=h_k+\alpha_k u_k
$$

KAN primitive 为：

$$
u_{k,j}
=
\sum_i \phi_{k,ji}(x_{k,i})
$$

edge function 用 basis 展开：

$$
\phi_{ji}(t)
=
\sum_m a_{jim}B_m(t)
$$

当前 RBF basis 可写为：

$$
B_m(t)=\exp(-\gamma(t-c_m)^2)
$$

其导数为：

$$
B'_m(t)
=
-2\gamma(t-c_m)B_m(t)
$$

所以：

$$
\phi'_{ji}(t)
=
\sum_m a_{jim}B'_m(t)
$$

---

### 2.2 Core KAN VJP

对于 core KAN：

$$
y_j=\sum_i \phi_{ji}(x_i)
$$

若输出端 credit 为 $g_y$，则输入端 credit 为：

$$
g_{x_i}
=
\sum_j g_{y_j}\phi'_{ji}(x_i)
$$

向量形式为：

$$
g_x
=
\Phi'(x)^\top g_y
$$

其中：

$$
\Phi'(x)_{ji}=\phi'_{ji}(x_i)
$$

这是 Stage C 的核心解析公式。

---

### 2.3 Full residual block VJP

完整 block 为：

$$
h_{out}=h+\alpha\operatorname{KAN}(\operatorname{LN}(h))
$$

记：

$$
x=\operatorname{LN}(h)
$$

$$
u=\operatorname{KAN}(x)
$$

若输出端 credit 为 $g_{out}$，则 residual identity path 给出：

$$
g_h^{id}=g_{out}
$$

KAN branch backward 给出：

$$
g_x^{branch}
=
\alpha\Phi'(x)^\top g_{out}
$$

再通过 LayerNorm 的 VJP：

$$
g_h^{branch}
=
D\operatorname{LN}(h)^\top g_x^{branch}
$$

所以完整 hidden credit 为：

$$
g_h
=
g_{out}+D\operatorname{LN}(h)^\top\left(\alpha\Phi'(x)^\top g_{out}\right)
$$

Stage C 会分别报告：

1. core KAN primitive VJP；
2. full residual block VJP；
3. identity path 与 branch path 的 credit decomposition。

---

### 2.4 Branch activity regimes

Stage C 必须带入 Stage B 已经发现的三个 regime，而不是只用一个 checkpoint。

| Regime | 典型设置 | 目的 |
|---|---|---|
| conservative | coeff lr 较小，例如 $0.1$ | KAN branch under-active，几何最稳 |
| Pareto | coeff lr $0.3$，rest lr $0.003$ | 精度 gap < $1\%$，几何明显更稳 |
| accuracy-parity | coeff lr $1.0$，rest lr $0.003$ | 精度追平 / 略超 AdamW，但几何优势减弱 |
| AdamW baseline | all AdamW | 标准对照 |

Branch ratio 定义为：

$$
r_{branch,k}
=
\frac{
\|\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))\|
}{
\|h_k\|+\epsilon
}
$$

相对 AdamW 的 branch strength 为：

$$
\text{branch / AdamW}
=
\frac{r_{branch}^{method}}{r_{branch}^{AdamW}}
$$

Stage C 的所有结论都要按 regime 分开报告。否则无法判断：router 容易学，是因为几何健康，还是因为 KAN branch 太弱、近似 identity。

---

## 3. Stage C 实验输入：checkpoint 与数据

### 3.1 数据集顺序

Stage C 使用以下数据集推进：

| 优先级 | 数据集 | 用途 |
|---:|---|---|
| 1 | MNIST small | 与 Stage B 结果直接衔接，成本低 |
| 2 | Fashion-MNIST | 验证 branch under-active 与 Pareto regime 是否外推 |
| 3 | Two Moons | 快速 sanity / 可视化 |
| 4 | CIFAR-10 small | 仅在 C1-C4 通过后进入 |

正式 Stage C 主结论至少需要 MNIST small 和 Fashion-MNIST。Two Moons 只能作为 sanity，不作为主要证据。

---

### 3.2 模型与 checkpoint

优先使用已经在 Stage B 中稳定的 residual DG-KAN 分类架构：

```text
x
  -> Linear stem
  -> LayerNorm
  -> [h <- h + alpha_k * RBFKAN(LayerNorm(h))] * depth
  -> Linear head
```

第一轮 Stage C 使用：

| setting | value |
|---|---:|
| hidden dim | $32$ |
| depth | $2$ |
| basis count | $8$ |
| alpha init | $1.0$ |
| train / val / test | same as Stage B |
| seeds | $0,1,2,3,4$ |

第二轮扩展：

| setting | value |
|---|---:|
| hidden dim | $64,128$ |
| depth | $4,8$ |
| basis count | $8,16$ |

Stage C 不应一开始进入 8 层以上。只有 C1-C4 在 depth $2$ 和 $4$ 上稳定后，再进入 depth $8$。

---

### 3.3 Regime checkpoint 准备

每个数据集、每个 seed 至少准备四类 checkpoint：

| checkpoint name | 训练方式 | 典型参数 |
|---|---|---|
| `all_adamw` | 所有参数 AdamW | baseline |
| `hybrid_conservative` | KAN coeff functional，非 KAN AdamW | coeff lr $0.1$ |
| `hybrid_pareto` | KAN coeff functional，非 KAN AdamW | coeff lr $0.3$, rest lr $0.003$ |
| `hybrid_active` | KAN coeff functional，非 KAN AdamW | coeff lr $1.0$, rest lr $0.003$ |

对于 functional update，默认方法为：

$$
\boxed{\text{diag-to-Sobolev}}
$$

但需要同时保留 `hybrid_sobolev` 和 `hybrid_diag` 作为 ablation。RKHS 不进入 Stage C 主线。

---

### 3.4 C0 必须复核的 checkpoint metrics

在进入 C1 前，每个 checkpoint 必须上传并确认以下指标：

| metric key | 含义 |
|---|---|
| `regime/train_acc` | 训练准确率 |
| `regime/test_acc` | 测试准确率 |
| `regime/test_gap_vs_adamw` | 相对 AdamW 的 test gap |
| `regime/branch_ratio` | branch ratio |
| `regime/branch_over_adamw` | branch / AdamW |
| `regime/no_kan_acc_drop` | 关闭 KAN branch 后精度下降 |
| `regime/phi_prime_p95` | $\phi'_{p95}$ |
| `regime/jacobian_condition_max` | 最大 Jacobian condition |
| `regime/active_basis_fraction` | active basis fraction |
| `regime/out_of_grid_fraction` | out-of-grid fraction |

C0 通过条件：

1. `all_adamw` 性能复现 Stage B；
2. `hybrid_pareto` 满足 test gap < $1\%$；
3. `hybrid_pareto` 满足 $\phi'_{p95}$ 和 Jacobian condition 低于 AdamW；
4. `hybrid_active` 精度接近或超过 AdamW；
5. coverage 不触发 failure。

如果 C0 不能复现 Stage B 的 regime，则不能进入 Stage C 主实验。

---

## 4. C1：Core KAN analytic adjoint 正确性验证

### 4.1 目标

C1 验证最基本的 KAN primitive adjoint：

$$
g_x=\Phi'(x)^\top g_y
$$

是否与 autograd VJP 一致。

这是 Stage C 的硬门槛。如果这里失败，则后续 learned router、local training、memory claim 全部不能推进。

---

### 4.2 实验对象

对每个 checkpoint、每一层 KAN block，提取：

$$
x=\operatorname{LN}(h)
$$

$$
y=\operatorname{KAN}(x)
$$

对 core KAN primitive 做 VJP 比较。

Credit 输入 $g_y$ 使用三类：

| credit type | 含义 |
|---|---|
| `task_bp_credit` | 来自真实 BP 的任务 credit |
| `random_gaussian_credit` | 随机 Gaussian credit，归一化到任务 credit norm |
| `mixed_credit` | task credit 与 random credit 混合 |

混合形式：

$$
g_y^{mix}
=
\lambda g_y^{task}
+
(1-\lambda)g_y^{rand}
$$

建议：

$$
\lambda\in\{0.25,0.5,0.75\}
$$

使用 random / mixed credit 是为了避免 analytic adjoint 只在真实 credit 分布上看起来正确，同时也为 learned router 的 OOD credit 测试做准备。

---

### 4.3 计算方法

Analytic VJP：

$$
g_{x_i}^{analytic}
=
\sum_j g_{y_j}\phi'_{ji}(x_i)
$$

Autograd VJP：

$$
g_x^{autograd}
=
J_{KAN}(x)^\top g_y
$$

其中 autograd 只覆盖 core KAN primitive，不包括 residual 和 LayerNorm。

---

### 4.4 W&B metrics

每个 layer、每种 credit type、每个 regime 记录：

| metric key | 含义 |
|---|---|
| `c1/core/layer_{k}/cos_vs_autograd` | analytic 与 autograd cosine |
| `c1/core/layer_{k}/relerr_vs_autograd` | relative error |
| `c1/core/layer_{k}/max_abs_error` | 最大绝对误差 |
| `c1/core/layer_{k}/norm_ratio` | analytic / autograd norm ratio |
| `c1/core/layer_{k}/sign_match_rate` | 元素符号一致率 |
| `c1/core/layer_{k}/cos_p05` | cosine 5 分位 |
| `c1/core/layer_{k}/relerr_p95` | relerr 95 分位 |
| `c1/core/layer_{k}/credit_type` | credit 类型 |
| `c1/core/layer_{k}/regime` | checkpoint regime |

全局聚合：

| metric key | 含义 |
|---|---|
| `c1/core/global/mean_cos` | 全层平均 cosine |
| `c1/core/global/worst_cos` | 最差层 cosine |
| `c1/core/global/mean_relerr` | 全层平均 relerr |
| `c1/core/global/max_relerr` | 最大 relerr |

成本：

| metric key | 含义 |
|---|---|
| `c1/cost/analytic_vjp_time_ms` | analytic VJP 时间 |
| `c1/cost/autograd_vjp_time_ms` | autograd VJP 时间 |
| `c1/cost/analytic_vjp_memory_mb` | analytic VJP 临时显存 |
| `c1/cost/autograd_vjp_memory_mb` | autograd VJP 临时显存 |
| `c1/cost/time_ratio_analytic_over_autograd` | analytic / autograd 时间比 |
| `c1/cost/memory_ratio_analytic_over_autograd` | analytic / autograd 显存比 |

---

### 4.5 可视化

C1 dashboard 必须包含：

1. **Analytic vs Autograd cosine heatmap**  
   x-axis: layer；y-axis: regime × credit type；color: `cos_vs_autograd`。

2. **Relative error heatmap**  
   x-axis: layer；y-axis: regime × credit type；color: `relerr_vs_autograd`。

3. **Norm calibration scatter**  
   x-axis: $\|g_x^{autograd}\|$；y-axis: $\|g_x^{analytic}\|$；color: regime。

4. **Time / memory bar chart**  
   compare analytic VJP vs autograd VJP。

5. **Error histogram**  
   histogram of elementwise relative error and cosine distribution。

---

### 4.6 成功标准

Core KAN primitive 上，正式成功标准为：

$$
\cos(g_x^{analytic},g_x^{autograd})>0.9999
$$

$$
\operatorname{relerr}<10^{-5}
$$

弱成功标准为：

$$
\cos>0.999
$$

$$
\operatorname{relerr}<10^{-4}
$$

如果使用 bf16 / fp16，允许相对误差放宽，但必须在 fp32 下满足严格标准。

若 C1 失败，优先检查：

1. RBF derivative 符号；
2. batch / dimension transpose；
3. dense KAN 的 $i,j$ 索引顺序；
4. residual scale 是否误加；
5. LayerNorm 是否被误包含；
6. autograd target 是否只包含 core KAN。

---

## 5. C2：Full residual DG-KAN block adjoint 验证

### 5.1 目标

C2 验证完整 residual block 的 VJP：

$$
h_{out}=h+\alpha\operatorname{KAN}(\operatorname{LN}(h))
$$

完整 VJP 公式是：

$$
g_h
=
g_{out}
+
D\operatorname{LN}(h)^\top\left(\alpha\Phi'(x)^\top g_{out}\right)
$$

C2 的目标是验证这个 full block adjoint 是否能与 autograd full block VJP 对齐，同时分解 identity path 与 branch path 的贡献。

---

### 5.2 两种实现级别

C2 分两种实现级别。

#### Level 1：Hybrid analytic full block VJP

使用 analytic KAN VJP，但 LayerNorm VJP 仍用 autograd 或框架函数：

$$
g_h^{hybrid}
=
g_{out}
+
D\operatorname{LN}(h)^\top_{autograd}\left(\alpha\Phi'(x)^\top g_{out}\right)
$$

这是推荐第一步，因为可以把错误主要限制在 KAN analytic VJP。

#### Level 2：Fully hand-written full block VJP

手写 LayerNorm VJP，得到完整 analytic block VJP：

$$
g_h^{analytic}
=
g_{out}
+
D\operatorname{LN}(h)^\top_{manual}\left(\alpha\Phi'(x)^\top g_{out}\right)
$$

Level 2 可以后做，不应阻塞 C3/C4。

---

### 5.3 需要分解的 credit 量

对每个 block，记录：

Identity path credit：

$$
g_h^{id}=g_{out}
$$

Branch path credit：

$$
g_h^{branch}=D\operatorname{LN}(h)^\top\left(\alpha\Phi'(x)^\top g_{out}\right)
$$

Total credit：

$$
g_h^{total}=g_h^{id}+g_h^{branch}
$$

Backward branch ratio：

$$
r_{back,k}
=
\frac{
\|g_h^{branch}\|
}{
\|g_h^{id}\|+\epsilon
}
$$

Identity-branch alignment：

$$
\cos_{id,branch}
=
\frac{
\langle g_h^{id},g_h^{branch}\rangle
}{
\|g_h^{id}\|\|g_h^{branch}\|+\epsilon
}
$$

Credit amplification：

$$
A_k
=
\frac{\|g_h^{total}\|}{\|g_{out}\|+\epsilon}
$$

这些指标是 Stage C 的关键，因为它们直接连接 Stage B 的 forward branch ratio 与 backward credit stability。

---

### 5.4 W&B metrics

Full VJP fidelity：

| metric key | 含义 |
|---|---|
| `c2/full/layer_{k}/cos_vs_autograd` | full block analytic / hybrid VJP 与 autograd cosine |
| `c2/full/layer_{k}/relerr_vs_autograd` | relative error |
| `c2/full/layer_{k}/norm_ratio` | norm ratio |
| `c2/full/layer_{k}/cos_p05` | 低分位 cosine |
| `c2/full/layer_{k}/relerr_p95` | 高分位 relerr |

Credit decomposition：

| metric key | 含义 |
|---|---|
| `c2/decomp/layer_{k}/id_credit_norm` | identity path credit norm |
| `c2/decomp/layer_{k}/branch_credit_norm` | branch path credit norm |
| `c2/decomp/layer_{k}/backward_branch_ratio` | $r_{back,k}$ |
| `c2/decomp/layer_{k}/id_branch_cos` | identity 与 branch credit cosine |
| `c2/decomp/layer_{k}/credit_amplification` | $\|g_h\|/\|g_{out}\|$ |
| `c2/decomp/layer_{k}/credit_amplification_p95` | amplification 95 分位 |

Forward-backward relation：

| metric key | 含义 |
|---|---|
| `c2/relation/layer_{k}/forward_branch_ratio` | Stage B 的 forward branch ratio |
| `c2/relation/layer_{k}/backward_branch_ratio` | backward branch ratio |
| `c2/relation/layer_{k}/forward_backward_branch_corr` | forward branch ratio 与 backward branch ratio 相关性 |
| `c2/relation/layer_{k}/phi_p95` | $\phi'_{p95}$ |
| `c2/relation/layer_{k}/jacobian_condition` | Jacobian condition |

---

### 5.5 可视化

1. **Full VJP fidelity by regime**  
   bar chart: regime vs `c2/full/global/mean_cos` and `mean_relerr`。

2. **Backward branch ratio heatmap**  
   x-axis: layer；y-axis: regime；color: `backward_branch_ratio`。

3. **Credit amplification vs branch ratio scatter**  
   x-axis: forward branch / AdamW；y-axis: credit amplification p95；color: regime。

4. **Identity-branch cosine distribution**  
   histogram by regime。

5. **Jacobian condition vs credit amplification**  
   x-axis: Jacobian condition；y-axis: amplification p95。

---

### 5.6 成功标准

Hybrid analytic full block VJP：

$$
\cos>0.999
$$

$$
\operatorname{relerr}<10^{-4}
$$

如果包含 hand-written LayerNorm：

$$
\cos>0.995
$$

$$
\operatorname{relerr}<10^{-3}
$$

Credit decomposition 成功标准不是固定数值，而是要求不同 regime 呈现可解释结构：

1. conservative regime 的 backward branch ratio 应较低；
2. Pareto regime 应有中等 backward branch ratio 和较低 amplification；
3. accuracy-parity regime 可能有更高 backward branch ratio 和更高 amplification；
4. naive Adam-moment 类 over-active regime 若加入对照，应表现出 amplification 或 condition 失控。

---

## 6. C3：Credit geometry audit across regimes

### 6.1 目标

C3 的目标不是验证公式，而是回答：

$$
\boxed{
\text{哪个 branch activity regime 的 credit transport 最健康？}
}
$$

更具体地说，C3 要判断：

1. Pareto regime 是否真的比 AdamW 或 accuracy-parity 有更低的 credit amplification；
2. conservative regime 的几何稳定是不是只是因为 branch 太弱；
3. accuracy-parity regime 是否因为 $\phi'$ 增大导致 credit transport 变难；
4. 哪个 regime 最适合 learned router。

---

### 6.2 要审计的对象

对每个数据集、每个 seed、每个 regime、每一层 block，采样 audit batch：

$$
B_{audit}\in\{256,512,1024\}
$$

计算：

1. hidden credit $g_h$；
2. output-side credit $g_{out}$；
3. analytic VJP credit；
4. autograd VJP credit；
5. random credit 的 VJP；
6. mixed credit 的 VJP。

---

### 6.3 Credit amplification 指标

Mean amplification：

$$
A_{mean,k}=\operatorname{mean}\frac{\|g_{h,k}\|}{\|g_{out,k}\|+\epsilon}
$$

P95 amplification：

$$
A_{p95,k}=\operatorname{p95}\frac{\|g_{h,k}\|}{\|g_{out,k}\|+\epsilon}
$$

Max amplification：

$$
A_{max,k}=\max\frac{\|g_{h,k}\|}{\|g_{out,k}\|+\epsilon}
$$

W&B keys：

| metric key | 含义 |
|---|---|
| `c3/amplification/layer_{k}/mean` | amplification mean |
| `c3/amplification/layer_{k}/p95` | amplification p95 |
| `c3/amplification/layer_{k}/max` | amplification max |
| `c3/amplification/global/mean` | 全层平均 |
| `c3/amplification/global/worst_p95` | 最差层 p95 |

---

### 6.4 Noise sensitivity 指标

对输出 credit 加扰动：

$$
\tilde g_{out}=g_{out}+\sigma\xi
$$

其中：

$$
\|\xi\|\approx\|g_{out}\|
$$

测试：

$$
\sigma\in\{0.01,0.05,0.1,0.2\}
$$

定义 VJP noise gain：

$$
G_{noise}
=
\frac{
\|C(\tilde g_{out})-C(g_{out})\|
}{
\|\tilde g_{out}-g_{out}\|+\epsilon
}
$$

W&B keys：

| metric key | 含义 |
|---|---|
| `c3/noise/layer_{k}/gain_sigma_0_01` | noise gain for $\sigma=0.01$ |
| `c3/noise/layer_{k}/gain_sigma_0_05` | noise gain for $\sigma=0.05$ |
| `c3/noise/layer_{k}/gain_sigma_0_1` | noise gain for $\sigma=0.1$ |
| `c3/noise/layer_{k}/gain_sigma_0_2` | noise gain for $\sigma=0.2$ |
| `c3/noise/global/worst_gain` | 最差 noise gain |

如果 Pareto regime 的 noise gain 显著低于 accuracy-parity，而精度 gap < $1\%$，这是非常重要的 Stage C 正面证据。

---

### 6.5 Credit rank 与子空间稳定性

对每层 hidden credit 构造：

$$
\Lambda_k=[g_k^{(1)},\dots,g_k^{(B)}]
$$

做 SVD：

$$
\Lambda_k=U_kS_kV_k^\top
$$

令：

$$
p_i=\frac{s_i^2}{\sum_j s_j^2}
$$

记录：

$$
E_r^{(2)}=\sum_{i=1}^{r}p_i
$$

$$
r_{PR}=\frac{1}{\sum_i p_i^2}
$$

W&B keys：

| metric key | 含义 |
|---|---|
| `c3/rank/layer_{k}/E8` | top-8 energy |
| `c3/rank/layer_{k}/E16` | top-16 energy |
| `c3/rank/layer_{k}/E32` | top-32 energy |
| `c3/rank/layer_{k}/PR` | participation rank |
| `c3/rank/layer_{k}/rank90` | rank90 |
| `c3/rank/layer_{k}/rank95` | rank95 |

子空间 overlap：

$$
\operatorname{overlap}_r(A,B)
=
\frac{1}{r}\|U_A^{(r)\top}U_B^{(r)}\|_F^2
$$

Energy-weighted overlap：

$$
\operatorname{EWO}(A,B)
=
\sum_i p_i\|U_B^\top u_i^A\|^2
$$

W&B keys：

| metric key | 含义 |
|---|---|
| `c3/overlap/layer_{k}/top8_batch` | top-8 batch overlap |
| `c3/overlap/layer_{k}/top16_batch` | top-16 batch overlap |
| `c3/overlap/layer_{k}/energy_weighted_batch` | energy-weighted overlap |
| `c3/overlap/layer_{k}/top8_regime` | 跨 regime top-8 overlap |
| `c3/overlap/layer_{k}/energy_weighted_regime` | 跨 regime EWO |

---

### 6.6 Jacobian spectrum 与 $\phi'$ 关系

记录：

| metric key | 含义 |
|---|---|
| `c3/jacobian/layer_{k}/sigma_max` | 最大奇异值 |
| `c3/jacobian/layer_{k}/sigma_min` | 最小奇异值 |
| `c3/jacobian/layer_{k}/condition` | 条件数 |
| `c3/kan/layer_{k}/phi_prime_p95` | $\phi'_{p95}$ |
| `c3/kan/layer_{k}/phi_prime_max` | $\phi'_{max}$ |
| `c3/kan/layer_{k}/curvature` | 曲率能量 |

必须画：

1. $\phi'_{p95}$ vs credit amplification；
2. Jacobian condition vs router relerr；
3. branch ratio vs VJP noise gain；
4. branch ratio vs $\phi'_{p95}$。

---

### 6.7 C3 成功判据

C3 不是单一 pass/fail，而是要产生解释性结论。

理想结果是：

1. Pareto regime 相比 AdamW：
   - test gap < $1\%$；
   - amplification p95 更低；
   - noise gain 更低；
   - $\phi'_{p95}$ 更低；
   - Jacobian condition 更低。

2. Conservative regime：
   - amplification 低；
   - router 可能容易学；
   - 但 branch too weak，应避免作为主要性能 regime。

3. Accuracy-parity regime：
   - performance 好；
   - 但 amplification、$\phi'$、condition 可能更高；
   - 需要判断 router 是否变难。

若结果显示 Pareto regime 同时性能接近且 credit geometry 明显更健康，那么 Stage C 的核心 hypothesis 获得支持。

---

## 7. C4：Learned local router distillation

### 7.1 目标

C4 训练 learned local router：

$$
\hat g_x=C^\phi(x,y)[g_y]
$$

逼近 teacher：

$$
g_x^{teacher}=\Phi'(x)^\top g_y
$$

或 full block teacher：

$$
g_h^{teacher}=J_{block}(h)^\top g_{out}
$$

C4 的目标不是立刻替代 analytic adjoint，而是回答：

$$
\boxed{
\text{KAN local VJP 是否可以被条件线性 router 蒸馏，且在哪个 branch regime 下最容易？}
}
$$

---

### 7.2 Router 训练数据拆分

每个 block 的 router 数据为：

$$
(x,y,g_y,g_x^{teacher})
$$

或 full block：

$$
(h,h_{out},g_{out},g_h^{teacher})
$$

拆分：

| split | 来源 | 用途 |
|---|---|---|
| `train_router` | 同一 checkpoint 的训练 batch | router 训练 |
| `val_router` | 同 checkpoint heldout batch | early stopping |
| `test_same_regime` | 同 regime 不同 batch | 同分布泛化 |
| `test_cross_seed` | 同 regime 不同 seed | seed 泛化 |
| `test_cross_regime` | train on one regime, eval on another | regime 泛化 |
| `test_random_credit` | random Gaussian credit | OOD credit |
| `test_mixed_credit` | task + random mixed credit | mixed OOD |

---

### 7.3 Router 类型

至少比较以下 router。

#### Router 0：zero

$$
\hat g=0
$$

负对照。

#### Router 1：identity

$$
\hat g=g_y
$$

residual block 中可能很强，因为 block 接近 identity。

#### Router 2：static ridge

学习静态线性映射：

$$
\hat g=Wg_y
$$

其中：

$$
W=\arg\min_W\sum_n\|Wg_y^{(n)}-g_x^{(n)}\|^2+\lambda\|W\|_F^2
$$

如果 static ridge 很强，说明不一定需要 $h$-conditioned router。

#### Router 3：conditioned diagonal

$$
\hat g=D(x,y)\odot g_y
$$

#### Router 4：conditioned diagonal + low-rank

$$
\hat g
=
D(x,y)\odot g_y
+
U(x,y)(V(x,y)^\top g_y)
$$

rank：

$$
r\in\{4,8,16,32\}
$$

#### Router 5：derivative-compressed KAN router

直接学习压缩形式的 derivative matrix：

$$
\widehat{\Phi'}(x)
$$

使：

$$
\hat g_x=\widehat{\Phi'}(x)^\top g_y
$$

可以采用 low-rank derivative factorization：

$$
\widehat{\Phi'}(x)
=
A(x)B(x)^\top+\operatorname{diag}(d(x))
$$

#### Router 6：nonlinear MLP router

$$
\hat g=C(x,y,g_y)
$$

作为负对照。它可能训练集表现好，但如果线性性和尺度泛化差，不应进入 Stage D。

#### Router 7：analytic oracle

$$
\hat g=\Phi'(x)^\top g_y
$$

不是 learned router，而是 oracle 上限。

---

### 7.4 Router loss

主 loss：

$$
\mathcal L_{vjp}
=
\frac{
\|\hat g-g^{teacher}\|^2
}{
\|g^{teacher}\|^2+\epsilon
}
$$

Cosine loss：

$$
\mathcal L_{cos}
=
1-
\frac{
\langle \hat g,g^{teacher}\rangle
}{
\|\hat g\|\|g^{teacher}\|+\epsilon
}
$$

Norm ratio loss：

$$
\mathcal L_{norm}
=
\left(
\frac{\|\hat g\|}{\|g^{teacher}\|+\epsilon}-1
\right)^2
$$

Linearity residual：

$$
r_{lin}
=
\frac{
\|C(\alpha v_1+\beta v_2)-\alpha C(v_1)-\beta C(v_2)\|
}{
\alpha\|C(v_1)\|+\beta\|C(v_2)\|+\epsilon
}
$$

Total loss：

$$
\mathcal L_{router}
=
\mathcal L_{vjp}
+
\lambda_{cos}\mathcal L_{cos}
+
\lambda_{norm}\mathcal L_{norm}
+
\lambda_{lin}r_{lin}
$$

对于结构性保证线性的 router，$r_{lin}$ 只作为 audit；对 nonlinear router，必须加入正则并作为负对照报告。

---

### 7.5 Router W&B metrics

Fidelity：

| metric key | 含义 |
|---|---|
| `c4/router/train/loss_vjp_norm` | normalized MSE |
| `c4/router/val/cos_full` | validation cosine |
| `c4/router/val/relerr_full` | validation relerr |
| `c4/router/val/norm_ratio` | norm ratio |
| `c4/router/val/cos_p05` | cosine 5 分位 |
| `c4/router/val/relerr_p95` | relerr 95 分位 |
| `c4/router/test_same_regime/cos` | same-regime test cosine |
| `c4/router/test_cross_regime/cos` | cross-regime cosine |
| `c4/router/test_random_credit/cos` | random credit cosine |
| `c4/router/test_mixed_credit/cos` | mixed credit cosine |

Linearity：

| metric key | 含义 |
|---|---|
| `c4/router/lin_residual` | superposition residual |
| `c4/router/scale_error_a0_5` | $C(0.5g)$ vs $0.5C(g)$ |
| `c4/router/scale_error_a2` | $C(2g)$ vs $2C(g)$ |
| `c4/router/superposition_error` | $C(g_1+g_2)$ vs $C(g_1)+C(g_2)$ |

Adjoint identity：

$$
r_{adj}^{norm}
=
\frac{
|\langle C(v),u\rangle-\langle v,Ju\rangle|
}{
|\langle v,Ju\rangle|+\epsilon
}
$$

| metric key | 含义 |
|---|---|
| `c4/router/adjoint_residual_norm_mean` | normalized adjoint residual mean |
| `c4/router/adjoint_residual_norm_p95` | p95 |

Baseline gap：

| metric key | 含义 |
|---|---|
| `c4/router/gap_vs_identity_cos` | cosine 相对 identity 提升 |
| `c4/router/gap_vs_static_ridge_cos` | cosine 相对 ridge 提升 |
| `c4/router/gap_vs_diagonal_cos` | cosine 相对 diagonal 提升 |
| `c4/router/gap_to_analytic_oracle` | 距 oracle 差距 |

Cost：

| metric key | 含义 |
|---|---|
| `c4/cost/router_time_ms` | router time |
| `c4/cost/analytic_vjp_time_ms` | analytic VJP time |
| `c4/cost/autograd_vjp_time_ms` | autograd VJP time |
| `c4/cost/router_memory_mb` | router memory |
| `c4/cost/router_num_params` | router 参数量 |
| `c4/cost/router_over_analytic_time_ratio` | router / analytic VJP time |

---

### 7.6 Cross-regime router matrix

必须做 cross-regime matrix。训练 regime 和评估 regime 形成矩阵：

| train \ eval | conservative | Pareto | accuracy-parity | AdamW |
|---|---:|---:|---:|---:|
| conservative | cos | cos | cos | cos |
| Pareto | cos | cos | cos | cos |
| accuracy-parity | cos | cos | cos | cos |
| pooled | cos | cos | cos | cos |

这张表非常关键。它回答：

1. router 是否只在单一 branch regime 内有效；
2. Pareto regime 学到的 router 是否最泛化；
3. accuracy-parity 是否因为 $\phi'$ / condition 更高而难以泛化；
4. pooled training 是否可以解决 regime shift。

---

### 7.7 可视化

C4 dashboard 至少包含：

1. **Router fidelity curve**  
   step vs val cosine / relerr。

2. **Layerwise router heatmap**  
   layer × router type，颜色为 cosine / relerr。

3. **Rank Pareto plot**  
   x-axis: router rank；y-axis: cosine；point size: router memory；color: router time。

4. **Baseline comparison bar**  
   identity、ridge、diagonal、diag-lowrank、derivative-compressed、nonlinear、analytic oracle。

5. **Cross-regime matrix heatmap**  
   train regime × eval regime，颜色为 cosine。

6. **Cost vs fidelity Pareto**  
   x-axis: router time；y-axis: cosine；color: router type。

7. **Linearity audit plot**  
   router type vs `lin_residual`。

8. **Adjoint residual plot**  
   router type vs normalized adjoint residual。

---

### 7.8 C4 成功标准

Weak success：

$$
\cos>0.9
$$

$$
\operatorname{relerr}<0.35
$$

Medium success：

$$
\cos>0.95
$$

$$
\operatorname{relerr}<0.2
$$

$$
0.8<\frac{\|\hat g\|}{\|g^{teacher}\|+\epsilon}<1.2
$$

$$
r_{lin}<0.05
$$

Strong success：

1. same-regime cosine > $0.95$；
2. cross-regime cosine > $0.90$；
3. router beats identity, diagonal, static ridge；
4. router time < analytic VJP time or router memory < analytic VJP memory；
5. adjoint residual normalized mean < $0.05$。

如果 learned router 不能 beat analytic VJP in cost or cannot beat static ridge in fidelity，则 learned router 不应进入 Stage D 主方法。此时 Stage D 应以 analytic adjoint 为主。

---

## 8. C5：One-step local update sanity check

### 8.1 目标

C5 是进入 Stage D 前的最后门槛。它不做完整训练，只检查：

> 如果用 analytic adjoint 或 learned router 提供的 credit，对某个 KAN block 做一次 local functional update，实际 loss 是否按预测方向下降？

这一步能提前发现：router fidelity 看起来不错，但用于 update 时方向不可靠的问题。

---

### 8.2 方法

固定训练好的 checkpoint，取一个 audit batch。比较下列 credit：

| credit source | 含义 |
|---|---|
| BP teacher | full BP credit |
| analytic adjoint | analytic KAN VJP |
| learned router | C4 router |
| identity | identity baseline |
| static ridge | ridge router |
| synthetic gradient | credit baseline |
| random feedback | negative control |

使用同一 functional update：

$$
a\leftarrow a-\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

或者对局部 target：

$$
\Delta F(h)\approx -\eta g_{out}
$$

测 update 前后 loss。

---

### 8.3 Metrics

| metric key | 含义 |
|---|---|
| `c5/local_update/pred_descent` | predicted descent |
| `c5/local_update/actual_descent` | actual descent |
| `c5/local_update/descent_ratio` | actual / predicted |
| `c5/local_update/sign_agreement` | sign 是否一致 |
| `c5/local_update/sign_agreement_rate` | 多 batch sign agreement |
| `c5/local_update/loss_before` | update 前 loss |
| `c5/local_update/loss_after` | update 后 loss |
| `c5/local_update/update_norm_coeff` | coefficient update norm |
| `c5/local_update/update_norm_function` | function update norm |
| `c5/local_update/phi_prime_p95_after` | update 后 $\phi'_{p95}$ |
| `c5/local_update/jacobian_condition_after` | update 后 condition |

---

### 8.4 可视化

1. predicted vs actual descent scatter；
2. credit source vs actual loss change bar chart；
3. descent sign agreement by regime；
4. update norm vs actual descent scatter；
5. phi p95 before / after update；
6. Jacobian condition before / after update。

---

### 8.5 成功标准

Analytic adjoint：

$$
\text{sign agreement rate}>0.85
$$

Learned router：

$$
\text{sign agreement rate}>0.70
$$

并且 learned router 的 actual descent 不应比 identity / static ridge 更差。

如果 C4 fidelity 成功但 C5 local update 失败，则说明：

$$
\text{VJP matching metric 不足以保证 update usefulness}
$$

这时 Stage D 不能直接用 learned router，必须先改 router loss 或加入 descent-aware training。

---

## 9. Stage C W&B dashboard 总设计

Stage C 建议建立 8 个 dashboard 页面。

### Page C0：Regime checkpoint map

内容：

1. test accuracy by regime；
2. branch / AdamW by regime；
3. no-KAN acc drop；
4. $\phi'_{p95}$；
5. Jacobian condition；
6. active basis fraction。

目标：确认 C1-C5 使用的 checkpoint 确实对应 conservative / Pareto / accuracy-parity。

---

### Page C1：Core analytic adjoint correctness

内容：

1. analytic vs autograd cosine heatmap；
2. relerr heatmap；
3. norm calibration scatter；
4. error distribution histogram；
5. analytic vs autograd time / memory bar。

---

### Page C2：Full block VJP decomposition

内容：

1. full block VJP cosine by regime；
2. identity vs branch credit norm；
3. backward branch ratio heatmap；
4. identity-branch alignment distribution；
5. credit amplification by layer。

---

### Page C3：Credit geometry across regimes

内容：

1. amplification p95 vs regime；
2. noise gain vs regime；
3. credit rank heatmap；
4. energy-weighted overlap；
5. $\phi'_{p95}$ vs amplification scatter；
6. branch ratio vs credit amplification scatter。

---

### Page C4：Router fidelity

内容：

1. router train / val cosine curves；
2. layerwise router heatmap；
3. rank vs fidelity Pareto；
4. baseline comparison bar；
5. cross-regime train/eval matrix；
6. cost vs fidelity Pareto。

---

### Page C5：Adjoint identity and linearity

内容：

1. normalized adjoint residual by router；
2. linearly residual by router；
3. scale error plots；
4. superposition error plots；
5. random / mixed credit OOD fidelity。

---

### Page C6：One-step local update sanity

内容：

1. predicted vs actual descent scatter；
2. actual descent by credit source；
3. sign agreement rate by method；
4. update norm vs loss change；
5. before / after $\phi'$ and Jacobian condition。

---

### Page C7：Failure analysis

内容：

1. all failures table；
2. analytic adjoint mismatch cases；
3. router low cosine cases；
4. router norm explosion cases；
5. cross-regime failure cases；
6. local update descent mismatch cases。

---

## 10. Stage C failure table

每个 failure 都要记录为结构化表，而不是只写文字。

| failure type | 触发条件 | 解释 |
|---|---|---|
| `c1_core_adj_mismatch` | core KAN relerr > $10^{-4}$ | edge derivative 或索引错误 |
| `c2_full_block_adj_mismatch` | full block relerr > $10^{-3}$ | residual / LN adjoint 错误 |
| `credit_amplification_high` | amplification p95 过高 | credit transport 不稳定 |
| `noise_gain_high` | noise gain 过高 | VJP 对 credit noise 敏感 |
| `router_low_cos` | router cosine < $0.85$ | router 学不准 |
| `router_norm_bad` | norm ratio < $0.5$ 或 > $1.5$ | router norm 校准失败 |
| `router_linearity_fail` | $r_{lin}>0.1$ | router 不保持线性 |
| `router_static_ridge_not_beaten` | learned router 不优于 ridge | 条件 router 贡献不足 |
| `cross_regime_fail` | cross-regime cosine < $0.75$ | router 不泛化 |
| `descent_mismatch` | predicted descent < 0 but actual > 0 | credit/update 不可靠 |
| `router_cost_too_high` | router slower than analytic VJP and no fidelity gain | 不适合部署 |

Failure table 字段：

| 字段 | 含义 |
|---|---|
| `failure/type` | failure 类型 |
| `failure/stage` | C1 / C2 / C3 / C4 / C5 |
| `failure/dataset` | 数据集 |
| `failure/regime` | conservative / Pareto / active / AdamW |
| `failure/method` | router / adjoint method |
| `failure/layer` | 层号 |
| `failure/seed` | seed |
| `failure/metric_name` | 触发 metric |
| `failure/metric_value` | metric 值 |
| `failure/threshold` | 阈值 |
| `failure/diagnosis` | 诊断 |
| `failure/recommended_action` | 建议动作 |

---

## 11. Stage C 第一轮最小执行包

为了避免一次性展开太多，第一轮只做下面 5 组实验。

### Experiment C0：Regime checkpoint audit

数据集：MNIST small、Fashion-MNIST。  
模型：depth 2, hidden dim 32, basis count 8。  
Regime：AdamW、conservative、Pareto、accuracy-parity。  
Seeds：5。

目标：确认 branch ratio、no-KAN drop、$\phi'_{p95}$、Jacobian condition 复现 Stage B。

成功：Pareto regime test gap < $1\%$，且 geometry metrics 低于 AdamW。

---

### Experiment C1：Core KAN analytic adjoint

数据集：MNIST small、Fashion-MNIST。  
Credit：task BP、random、mixed。  
Regime：四类 checkpoint。  
Seeds：5。

目标：验证：

$$
g_x=\Phi'(x)^\top g_y
$$

成功：

$$
\cos>0.999
$$

$$
\operatorname{relerr}<10^{-4}
$$

---

### Experiment C2：Full residual block adjoint

使用 hybrid analytic：analytic KAN VJP + autograd LayerNorm VJP。

目标：验证 full block VJP，并分解 identity / branch credit。

成功：

$$
\cos>0.999
$$

$$
\operatorname{relerr}<10^{-4}
$$

并得到 backward branch ratio 与 amplification 的 regime 差异。

---

### Experiment C3：Credit geometry audit

目标：比较四个 regime 的 credit amplification、noise gain、credit rank、Jacobian condition。

核心图：branch ratio vs amplification，$\phi'_{p95}$ vs noise gain，regime × layer heatmap。

成功：Pareto regime 显示更低 amplification / noise gain，同时保持 < $1\%$ test gap。

---

### Experiment C4：Router distillation

Router：identity、static ridge、diagonal、diag-lowrank、derivative-compressed、nonlinear、analytic oracle。  
Rank：$0,4,8,16,32$。  
Train/eval：same-regime + cross-regime matrix。

成功：

$$
\cos>0.95
$$

$$
\operatorname{relerr}<0.2
$$

并且 learned router beat static ridge / diagonal baseline。

---

### Experiment C5：One-step local update sanity

Credit source：BP teacher、analytic adjoint、learned router、identity、ridge、synthetic gradient、random feedback。

目标：检查 routed credit 是否真的能支持 local functional update。

成功：analytic adjoint sign agreement > $0.85$；learned router sign agreement > $0.70$，且 actual descent 不差于 identity / ridge。

---

## 12. Stage C 结果如何决策 Stage D

### 情况 1：C1 / C2 成功，C4 失败

结论：analytic KAN adjoint 可用，但 learned router 暂时不可用。Stage D 应优先走：

$$
\text{AnalyticAdj-KAN-Sobolev}
$$

而不是 Router-KAN-Sobolev。

---

### 情况 2：C4 same-regime 成功，但 cross-regime 失败

结论：router 能拟合局部 VJP，但对 branch regime / checkpoint 分布敏感。Stage D 若使用 router，必须 online update 或 scheduled audit。

---

### 情况 3：C4 成功，C5 失败

结论：VJP fidelity 指标不足以保证 local update 有用。需要加入 descent-aware router loss 或 local update-aware validation。

---

### 情况 4：Pareto regime 的 C3 / C4 全部最好

这是最理想结果。可以支持：

$$
\text{functional update improves local credit geometry and router learnability}
$$

Stage D 应以 Pareto regime 作为默认训练设置。

---

### 情况 5：accuracy-parity router 最好

说明高 branch activity 虽然几何指标更高，但 credit signal 更强、更容易学。此时需要重新权衡 Stage B 的几何目标和 Stage C 的 routing 目标。

---

## 13. Stage C 预期论文级结论

若 Stage C 成功，可以写出下面这种较强但仍诚实的结论：

> In DG-KAN, KAN-like primitives provide an exact analytic local adjoint. Functional-space update changes the branch activity and credit geometry; in the Pareto regime, the model maintains near-AdamW accuracy while reducing Jacobian condition, $\phi'_{p95}$, and credit amplification. Learned local routers can approximate KAN VJP in selected regimes, suggesting that local credit transport can be distilled under controlled KAN geometry.

中文表述：

> 在 DG-KAN 中，KAN-like primitive 提供了精确的解析局部 adjoint。Functional-space update 会改变 branch activity 与 credit geometry；在 Pareto regime 中，模型保持接近 AdamW 的精度，同时降低 Jacobian condition、$\phi'_{p95}$ 和 credit amplification。Learned local router 能在部分 regime 中近似 KAN VJP，说明在受控 KAN 几何下，局部 credit transport 具有被蒸馏的可能。

如果 only analytic adjoint 成功而 router 不成功，则结论改成：

> DG-KAN supports exact primitive-adjoint co-design and functional-space updates, but learned router distillation remains unstable or not cost-effective.

这仍然是有效研究结果。
