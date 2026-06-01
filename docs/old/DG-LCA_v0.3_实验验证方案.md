# DG-LCA v0.3 实验验证方案

> 版本：v0.1  
> 对象：DG-LCA v0.3 保守实验版  
> 定位：实验计划，不涉及具体代码实现  
> 公式格式：Typora 友好，使用 `$...$` 与 `$$...$$`，不使用方括号形式的 display math  
> 核心目标：验证 **local VJP distillation + functional-space preconditioning + 结构化低显存训练** 是否在小规模、结构化模块中成立。

---

## 0. 一句话实验目标

DG-LCA v0.3 不再试图证明“通用替代 BP”，而是验证一个更小、更可实验的问题：

> 在 KAN-like / adapter / 小型 residual block 等结构化模块中，BP 的局部 VJP 操作是否可以被条件线性 router 蒸馏；函数空间预条件更新是否能改善局部优化稳定性；这些机制是否能在特定场景下降低 activation memory 或改善训练效率。

可以压缩为：

$$
\text{full BP}
\quad\rightarrow\quad
\text{local VJP distillation}
+
\text{functional-space preconditioning}
+
\text{strict audits}
$$

本方案不以“大规模替代 BP”为目标，而以“验证局部组件是否成立”为目标。

---

## 1. 总体研究问题

本实验方案要回答五个核心问题。

### Q1：真实 credit 是否可压缩？

如果真实 BP credit directions 高维、无结构、随训练快速变化，那么 learned router 很难比 BP 更高效。

核心问题：

$$
\text{rank}_{\epsilon}(\text{credit}) \ll \dim(h)
$$

是否在 KAN-like / adapter / 小模型中成立？

---

### Q2：single-block local VJP router 能否学准？

标准 BP 中单个 block 的局部 VJP 是：

$$
g_k^{\text{teacher}}
=
G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

DG-LCA 学一个 router：

$$
\hat g_k
=
C_k^\phi(h_k,h_{k+1})[g_{k+1}]
$$

实验要验证：

$$
\hat g_k
\approx
g_k^{\text{teacher}}
$$

---

### Q3：KAN-like functional-space preconditioning 是否优于普通参数更新？

普通 coefficient update：

$$
a\leftarrow a-\eta\nabla_a\mathcal L
$$

函数空间预条件更新：

$$
a\leftarrow a-\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

实验要验证它是否带来：

1. 更快收敛；
2. 更稳定的 Jacobian 谱；
3. 更小的函数斜率和曲率；
4. 更好的 predicted descent 与 actual descent 一致性。

---

### Q4：learned router 多层串联后误差是否可控？

单层 router 好，不代表多层稳定。多层 self-routed credit 可能出现误差累积：

$$
e_k
=
C_k^{\text{true}}e_{k+1}
+
E_kg_{k+1}^{\text{true}}
+
E_ke_{k+1}
$$

实验要验证：

$$
\cos(\hat g_k,g_k^{\text{BP}})
$$

是否随层数增长仍然可接受。

---

### Q5：adapter-only DG-LCA 是否有实际 memory / compute 优势？

不能只和 vanilla BP 比。必须和：

1. checkpointing；
2. adapter-only BP；
3. adapter-only BP + checkpointing；
4. local exact VJP；
5. synthetic gradient；
6. feedback alignment；

做完整比较。

只有当：

$$
M_{\text{DG}} < M_{\text{checkpoint}}
$$

或：

$$
M_{\text{DG}} < M_{\text{adapter+checkpoint}}
$$

才可以声称有相对强 baseline 的显存优势。

---

## 2. 实验总路线

推荐按 Gate 逐步推进：

$$
\boxed{
\text{Gate 0 credit rank}
\rightarrow
\text{Gate 1 single-block VJP router}
\rightarrow
\text{Gate 2 KAN functional update}
\rightarrow
\text{Gate 3 sequential error audit}
\rightarrow
\text{Gate 4 adapter-only DG-LCA}
}
$$

原则：

1. Gate 0 失败，不做 full hidden-space learned router。
2. Gate 1 失败，不做 sequential learned routing。
3. Gate 2 成功，即使其他 Gate 失败，也可形成独立贡献。
4. Gate 3 失败，不推进 macro-router / tree routing。
5. Gate 4 失败，不宣称低显存优势。

macro-router / tree routing 暂不进入主线，只作为后续探索。

---

## 3. Gate 0：Credit 子空间诊断

### 3.1 目标

验证真实 BP credit 是否低维、可压缩、稳定。

这是 learned router 是否值得做的必要条件。

---

### 3.2 小目标

1. 测每层 credit 的有效秩。
2. 测 top-$r$ directions 对 credit variance 的解释比例。
3. 测不同 batch、不同 epoch、不同 layer 的 credit 子空间稳定性。
4. 比较 full hidden space credit 与 adapter / LoRA / KAN-like subspace credit。
5. 判断后续实验是否应收缩到结构化低维子空间。

---

### 3.3 数据集

#### Toy 数据集

用于 sanity check 与可视化：

1. Two Moons classification；
2. Circles classification；
3. 1D function regression：

$$
y=\sin x
$$

$$
y=\sin x+0.3\sin 5x
$$

4. 2D function regression：

$$
y=\sin x_1+\cos x_2
$$

#### 小型真实数据集

1. MNIST；
2. Fashion-MNIST；
3. CIFAR-10。

#### 可选中等规模数据集

若前面实验通过，再加入：

1. CIFAR-100；
2. Tiny-ImageNet small subset；
3. AG News；
4. SST-2。

---

### 3.4 模型

建议从小到大：

1. 4-layer MLP；
2. 8-layer MLP；
3. small residual MLP；
4. small CNN；
5. KAN-like MLP；
6. frozen backbone + adapter；
7. frozen backbone + LoRA-style module。

---

### 3.5 收集对象

在 diagnostic mode 下，用 BP 或 local exact VJP 收集每层 credit：

$$
\lambda_k^{(n)}
$$

构造 credit 矩阵：

$$
\Lambda_k=[\lambda_k^{(1)},\dots,\lambda_k^{(B)}]
$$

做 SVD：

$$
\Lambda_k=U_kS_kV_k^\top
$$

奇异值为：

$$
s_1\geq s_2\geq\cdots
$$

---

### 3.6 指标

令：

$$
p_i=\frac{s_i^2}{\sum_j s_j^2}
$$

主指标为 variance energy：

$$
E_r^{(2)}=\sum_{i=1}^{r}p_i
$$

participation rank：

$$
r_{\text{PR}}=\frac{1}{\sum_i p_i^2}
$$

entropy rank：

$$
r_{\text{ent}}
=
\exp\left(
-\sum_i p_i\log(p_i+\epsilon)
\right)
$$

辅助指标为 nuclear energy：

$$
E_r^{(1)}
=
\frac{\sum_{i=1}^{r}s_i}{\sum_i s_i}
$$

还需要报告：

1. across-batch subspace overlap；
2. across-epoch subspace overlap；
3. layer-wise rank profile；
4. batch-size sensitivity；
5. hidden-space rank vs adapter-subspace rank。

---

### 3.7 Baseline / 对照

Gate 0 本身是诊断实验，不是训练方法对比。但需要比较不同 credit 来源：

1. full BP credit；
2. local exact VJP credit；
3. adapter / LoRA projected credit；
4. random feedback credit；
5. local loss credit；
6. synthetic gradient generated credit。

---

### 3.8 成功标准

#### 弱成功

多数层满足：

$$
E_{128}^{(2)}>0.8
$$

并且 credit 子空间在训练中没有完全漂移。

#### 中等成功

多数层满足：

$$
E_{64}^{(2)}>0.8
$$

或：

$$
E_{128}^{(2)}>0.9
$$

并且 adapter / KAN-like 子空间的 credit 比 full hidden space 明显更低维。

#### 强成功

多个数据集、多个模型上均观察到：

$$
E_{64}^{(2)}>0.8
$$

且 top directions 在不同 batch / epoch 间有较高重合度。

---

### 3.9 失败标准

如果多数层满足：

$$
E_{128}^{(2)}<0.5
$$

并且不同 batch / epoch 的 top directions 明显不稳定，则 full hidden-space learned router 不应作为主线。

此时应收缩到：

1. adapter / LoRA 子空间；
2. KAN-like 模块；
3. top-$r$ projected credit；
4. exact local VJP；
5. local functional update，不依赖 learned router。

---

## 4. Gate 1：Single-block Local VJP Router

### 4.1 目标

验证条件线性 router 是否能在单个 block 上拟合 local exact VJP。

目标公式：

$$
\hat g_k=C_k^\phi(h_k,h_{k+1})[g_{k+1}]
$$

teacher 为：

$$
g_k^{\text{teacher}}
=
G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

此阶段仍使用 local autograd / local exact VJP，不声称低成本。目标只是验证 router 结构是否有表达能力。

---

### 4.2 小目标

1. diagonal router 是否足够；
2. diagonal + low-rank router 是否足够；
3. rank $r=4,8,16,32$ 的效果；
4. router 是否严格保持对 $g$ 的线性；
5. router 是否具有 scale generalization；
6. block 参数更新后 router 是否 stale；
7. teacher-forced credit 与 self-generated credit 的分布差异。

---

### 4.3 数据集

优先使用：

1. Two Moons；
2. Circles；
3. MNIST；
4. Fashion-MNIST；
5. CIFAR-10。

若前面通过，再加入：

1. AG News；
2. SST-2；
3. adapter-only text classification。

---

### 4.4 模型与 block 类型

1. MLP block；
2. residual MLP block；
3. small CNN block；
4. KAN-like block；
5. adapter block；
6. LoRA-style block。

---

### 4.5 Router 结构

#### Router 0：identity baseline

$$
\hat g_k=g_{k+1}
$$

#### Router 1：diagonal router

$$
\hat g_k=D(h)\odot g_{k+1}
$$

#### Router 2：diagonal + low-rank router

$$
C(h)[g]
=
D(h)\odot g+U(h)(V(h)^\top g)
$$

其中 rank $r$ 从 $4,8,16,32$ 开始。

#### Router 3：block-diagonal router

按照 channel / head / token group 分块。

#### Router 4：attention-style credit router

用于 token 模型，形式如：

$$
C(h)[g]=A_{\text{tok}}(h)gB_{\text{chan}}(h)^\top
$$

注意 $A_{\text{tok}}(h)$ 与 $B_{\text{chan}}(h)$ 可以依赖 hidden states，但对 $g$ 的作用必须保持线性。

#### Router 5：nonlinear MLP router

作为负对照。它可能在训练集上拟合较好，但可能破坏线性性、尺度泛化和组合性。

---

### 4.6 Baseline

1. exact local VJP teacher；
2. identity pass-through；
3. diagonal scaling；
4. random feedback matrix；
5. learned random projection；
6. synthetic gradient style predictor；
7. feedback alignment；
8. direct feedback alignment。

---

### 4.7 指标

#### VJP fidelity

cosine：

$$
\cos_k
=
\frac{
\langle \hat g_k,g_k^{\text{teacher}}\rangle
}{
\|\hat g_k\|\|g_k^{\text{teacher}}\|+\epsilon
}
$$

relative error：

$$
\operatorname{relerr}_k
=
\frac{
\|\hat g_k-g_k^{\text{teacher}}\|
}{
\|g_k^{\text{teacher}}\|+\epsilon
}
$$

norm ratio：

$$
\rho_k
=
\frac{
\|\hat g_k\|
}{
\|g_k^{\text{teacher}}\|+\epsilon
}
$$

#### 线性性

scale test：

$$
C(ag)\approx aC(g)
$$

superposition test：

$$
C(g_1+g_2)\approx C(g_1)+C(g_2)
$$

normalized linearity residual：

$$
r_{\text{lin}}
=
\frac{
\|C(\alpha v_1+\beta v_2)-\alpha C(v_1)-\beta C(v_2)\|
}{
\alpha\|C(v_1)\|+\beta\|C(v_2)\|+\epsilon
}
$$

#### 稳定性

router stale 可通过若干训练步后 cosine / relerr 的下降估计：

$$
\Delta_{\text{stale}}
\approx
1-\cos(\hat g_k^{t+\tau},g_k^{\text{teacher},t+\tau})
$$

#### 下游局部更新

用 router credit 做一次 local update，报告：

$$
\Delta\mathcal L_{\text{actual}}
$$

以及：

$$
\frac{
\Delta\mathcal L_{\text{actual}}
}{
\Delta\mathcal L_{\text{pred}}+\epsilon
}
$$

---

### 4.8 成功标准

#### 弱成功

在 KAN-like / small MLP block 上：

$$
\cos_k>0.9
$$

$$
\operatorname{relerr}_k<0.35
$$

$$
0.7<\rho_k<1.3
$$

#### 中等成功

在 MLP / CNN / adapter block 上：

$$
\cos_k>0.95
$$

$$
\operatorname{relerr}_k<0.2
$$

$$
0.8<\rho_k<1.2
$$

并且：

$$
r_{\text{lin}}<0.05
$$

#### 强成功

多个 block、多层、多训练阶段均满足：

$$
\cos_k>0.95
$$

且 router stale 在 50 到 200 个训练 step 内不明显恶化。

---

### 4.9 失败标准

如果 single-block 长期满足：

$$
\cos_k<0.85
$$

或：

$$
\operatorname{relerr}_k>0.5
$$

或 norm ratio 经常爆炸 / 消失，则不应进入 sequential routing。

修正方向：

1. 增强 router 结构；
2. 增大 low-rank rank；
3. 限制到 adapter / LoRA 子空间；
4. 使用 top-$r$ credit projection；
5. 回退到 exact local VJP。

---

## 5. Gate 2：KAN-like Functional-Space Update

### 5.1 目标

验证函数空间预条件更新是否比普通 coefficient BP / Adam 更稳定。

KAN-like 层：

$$
y_j=\sum_i\phi_{ji}(x_i)
$$

其中每条边是一个一元函数：

$$
\phi_{ji}:\mathbb R\to\mathbb R
$$

---

### 5.2 小目标

1. functional preconditioning 是否改善收敛速度；
2. 是否降低 $\phi'$ 爆炸；
3. 是否降低 $\phi''$ 过大；
4. 是否改善 Jacobian 谱稳定性；
5. predicted descent 与 actual descent 是否一致；
6. damping $\rho$ 对稳定性的影响；
7. Sobolev metric 与 RKHS metric 哪个更合适。

---

### 5.3 数据集

#### 第一阶段：函数拟合

1. 平滑低频：

$$
y=\sin x
$$

2. 中频：

$$
y=\sin x+0.3\sin 5x
$$

3. 局部变化：

$$
y=\sin x+\mathbf 1_{x>0}
$$

也可以使用平滑 step function 替代硬阶跃。

4. 二维回归：

$$
y=\sin x_1+\cos x_2
$$

#### 第二阶段：分类

1. Two Moons；
2. Circles；
3. MNIST；
4. Fashion-MNIST；
5. CIFAR-10 small KAN-like classifier。

---

### 5.4 方法比较

#### Baseline 1：coefficient SGD

$$
a\leftarrow a-\eta\nabla_a\mathcal L
$$

#### Baseline 2：coefficient Adam

普通 Adam 更新 spline / basis coefficient。

#### Baseline 3：coefficient AdamW

带 weight decay 的 Adam。

#### Method 1：Sobolev preconditioned update

$$
a\leftarrow a-\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

其中：

$$
M_{mn}
=
\int B_mB_n
+
\alpha\int B_m'B_n'
+
\beta\int B_m''B_n''
$$

#### Method 2：RKHS preconditioned update

使用 kernel-induced Gram matrix：

$$
M_{mn}=\langle B_m,B_n\rangle_{\mathcal H}
$$

#### Method 3：diagonal preconditioner

只使用 $M$ 的 diagonal 近似。

#### Optional：K-FAC / natural-gradient-like local method

若资源允许，作为更强 baseline。

---

### 5.5 指标

#### 任务指标

1. train loss；
2. validation loss；
3. test accuracy；
4. generalization gap；
5. convergence steps to target loss。

#### 函数形状指标

最大斜率：

$$
\max_t |\phi'_{ji}(t)|
$$

曲率能量：

$$
\int |\phi''_{ji}(t)|^2dt
$$

函数空间变化大小：

$$
\|\Delta\phi\|_{\mathcal H}
$$

#### Jacobian 稳定性

$$
\sigma_{\max}(J)
$$

$$
\sigma_{\min}(J)
$$

condition number：

$$
\kappa(J)=\frac{\sigma_{\max}(J)}{\sigma_{\min}(J)+\epsilon}
$$

#### Descent alignment

predicted descent：

$$
\Delta\mathcal L_{\text{pred}}
=
D_F\mathcal L[\Delta F]
$$

actual descent：

$$
\Delta\mathcal L_{\text{actual}}
=
\mathcal L_{\text{after}}-\mathcal L_{\text{before}}
$$

descent ratio：

$$
r_{\text{descent}}
=
\frac{
\Delta\mathcal L_{\text{actual}}
}{
\Delta\mathcal L_{\text{pred}}+\epsilon
}
$$

#### Metric stability

condition number：

$$
\kappa(M+\rho I)
$$

---

### 5.6 成功标准

#### 弱成功

functional preconditioning 相比 Adam：

1. final validation loss 不差；
2. $\max|\phi'|$ 更低；
3. $\int|\phi''|^2$ 更低；
4. Jacobian spectrum 更稳定。

#### 中等成功

在至少两个数据集上满足至少一项：

1. 收敛步数减少 20% 以上；
2. 同样步数下 validation loss 降低 5% 以上；
3. 同等 accuracy 下 $\max|\phi'|$ / curvature 显著更小。

并且：

$$
\Delta\mathcal L_{\text{pred}}<0
\Rightarrow
\Delta\mathcal L_{\text{actual}}<0
$$

在至少 70% 到 80% 的 update 中成立。

#### 强成功

在 MNIST / Fashion-MNIST / CIFAR-10 small 上同时满足：

1. validation accuracy 不低于 Adam；
2. convergence 更快；
3. Jacobian spectrum 更稳定；
4. 函数更平滑；
5. 对学习率更不敏感。

---

### 5.7 失败标准

如果 functional preconditioning 出现：

1. loss 不下降；
2. 明显过平滑导致表达力下降；
3. $M+\rho I$ 条件数过大；
4. actual descent 经常与 predicted descent 相反；
5. 完全打不过 Adam；

则 functional update 应降级为：

> 一种可选 local preconditioner，而不是 DG-LCA 的核心贡献。

---

## 6. Gate 3：Sequential Learned Routing Audit

### 6.1 目标

验证 learned router 串联后，credit 误差是否可控。

真实递推：

$$
g_k^{\text{true}}
=
C_k^{\text{true}}g_{k+1}^{\text{true}}
$$

learned 递推：

$$
\hat g_k
=
(C_k^{\text{true}}+E_k)\hat g_{k+1}
$$

误差近似：

$$
e_k
=
C_k^{\text{true}}e_{k+1}
+
E_kg_{k+1}^{\text{true}}
+
E_ke_{k+1}
$$

---

### 6.2 小目标

1. 比较 teacher-forced routing 与 self-routed routing；
2. 测每层 gradient cosine；
3. 测误差随深度增长；
4. 测 predicted descent 与 actual descent 是否一致；
5. 与 synthetic gradient / feedback alignment / local loss 比较；
6. 判断是否进入 adapter-only 实验。

---

### 6.3 数据集

1. Two Moons；
2. Circles；
3. MNIST；
4. Fashion-MNIST；
5. CIFAR-10；
6. 可选 AG News / SST-2 small adapter model。

---

### 6.4 模型

1. 4-block MLP / residual MLP；
2. 8-block MLP / residual MLP；
3. 12-block MLP / residual MLP；
4. 4/8-block KAN-like model；
5. small CNN with block routers。

第一版不要做 24、48、96 层。

---

### 6.5 Baseline

1. exact BP；
2. local exact VJP sequential；
3. learned router teacher-forced；
4. learned router self-routed；
5. synthetic gradient；
6. feedback alignment；
7. direct feedback alignment；
8. local loss；
9. target propagation；
10. random feedback negative control。

---

### 6.6 指标

#### 每层 credit fidelity

$$
\cos(\hat g_k,g_k^{\text{BP}})
$$

$$
\frac{
\|\hat g_k-g_k^{\text{BP}}\|
}{
\|g_k^{\text{BP}}\|+\epsilon
}
$$

#### Teacher-forced vs self-routed drift

teacher-forced：

$$
\hat g_k
=
C_k(h_k,h_{k+1})[g_{k+1}^{\text{BP}}]
$$

self-routed：

$$
\hat g_k
=
C_k(h_k,h_{k+1})[\hat g_{k+1}]
$$

drift：

$$
\operatorname{drift}_k
=
\|\hat g_k^{\text{self}}-\hat g_k^{\text{teacher-forced}}\|
$$

#### 训练指标

1. train loss；
2. validation loss；
3. test accuracy；
4. training stability；
5. number of divergence events。

#### Descent 指标

$$
\Delta\mathcal L_{\text{pred}}
$$

$$
\Delta\mathcal L_{\text{actual}}
$$

$$
\frac{
\Delta\mathcal L_{\text{actual}}
}{
\Delta\mathcal L_{\text{pred}}+\epsilon
}
$$

#### 深度敏感性

分别报告 4、8、12 block 下的：

1. average cosine；
2. worst-layer cosine；
3. final loss gap；
4. drift growth；
5. divergence rate。

---

### 6.7 成功标准

#### 弱成功

4-block 模型中：

$$
\cos(\hat g_k,g_k^{\text{BP}})>0.8
$$

且 train loss 稳定下降。

#### 中等成功

8-block 模型中：

$$
\cos(\hat g_k,g_k^{\text{BP}})>0.8
$$

多数层：

$$
\operatorname{relerr}<0.5
$$

最终性能与 exact BP 差距满足：

1. classification accuracy gap < 3%；
2. 或 validation loss gap < 10%。

#### 强成功

12-block 模型中：

$$
\cos(\hat g_k,g_k^{\text{BP}})>0.85
$$

self-routed drift 可控，且性能接近 BP：

1. accuracy gap < 1% 到 2%；
2. validation loss gap < 5%；
3. actual descent 与 predicted descent 在 70% 到 80% 的 update 中方向一致。

---

### 6.8 失败标准

如果 8-block 时：

$$
\cos(\hat g_k,g_k^{\text{BP}})<0.5
$$

或 self-routed routing 明显崩溃，而 teacher-forced routing 仍然好，说明主要问题是：

$$
\text{distribution shift}+\text{error accumulation}
$$

此时不应扩展到更深模型，也不应进入 macro-router。

修正方向：

1. scheduled mixing；
2. more frequent exact VJP audit；
3. trust-region；
4. adapter-only subspace；
5. top-$r$ credit projection；
6. exact local VJP fallback。

---

## 7. Gate 4：Adapter-only DG-LCA Memory / Compute Test

### 7.1 目标

验证 DG-LCA 是否在结构化低维子空间中有实际 memory / compute / performance tradeoff。

此阶段不训练 full model，而是 frozen backbone + adapter：

$$
h_{k+1}
=
h_k
+
F_k^{\text{frozen}}(h_k)
+
A_k(h_k;\psi_k)
$$

只训练 adapter 参数 $\psi_k$。

---

### 7.2 小目标

1. adapter credit 是否比 full hidden credit 更低维；
2. learned router 在 adapter subspace 上是否更准；
3. local adapter update 是否接近 full BP adapter training；
4. memory 是否低于 checkpointed adapter BP；
5. router overhead 是否可接受；
6. exact VJP audit 是否可以低频进行。

---

### 7.3 数据集

#### 图像任务

1. CIFAR-10；
2. CIFAR-100；
3. Tiny-ImageNet small subset。

#### 文本任务

1. AG News；
2. SST-2；
3. IMDB sentiment；
4. small sequence classification task。

不建议第一版做 diffusion 或 language modeling，复杂度太高。

---

### 7.4 模型

1. frozen small ResNet + adapters；
2. frozen small ViT + adapters；
3. frozen small Transformer encoder + adapters；
4. LoRA-style low-rank adapters；
5. KAN-like adapters。

---

### 7.5 Baseline

1. full BP adapter training；
2. adapter BP + activation checkpointing；
3. local BP through adapter；
4. exact local VJP + local adapter update；
5. learned VJP router + local adapter update；
6. synthetic gradient adapter training；
7. feedback alignment adapter training；
8. local loss adapter training；
9. standard LoRA training。

---

### 7.6 指标

#### 性能指标

1. validation loss；
2. test accuracy；
3. convergence speed；
4. stability across seeds；
5. final performance gap vs adapter BP。

#### Memory accounting

标准 BP：

$$
M_{\text{BP}}
=
M_{\text{params}}
+
M_{\text{optim}}
+
M_{\text{activations}}
$$

DG-LCA：

$$
M_{\text{DG}}
=
M_{\text{params}}
+
M_{\text{optim}}
+
M_{\text{interfaces}}
+
M_{\text{router}}
+
M_{\text{stats}}
+
M_{\text{local}}
$$

必须报告：

1. peak GPU memory；
2. activation memory；
3. router parameter memory；
4. router activation memory；
5. local update memory；
6. audit buffer memory。

#### Compute accounting

DG-LCA 总计算：

$$
C_{\text{DG}}
=
C_{\text{forward}}
+
C_{\text{router}}
+
C_{\text{teacher/audit}}
+
C_{\text{probe}}
+
C_{\text{local update}}
+
C_{\text{metric}}
$$

必须报告：

1. wall-clock time；
2. throughput；
3. total FLOPs estimate；
4. router overhead；
5. exact VJP audit frequency cost。

#### Credit 指标

1. adapter-subspace gradient cosine；
2. hidden-space gradient cosine；
3. predicted vs actual descent；
4. router stale rate；
5. self-routed drift。

---

### 7.7 成功标准

#### 弱成功

DG-LCA adapter training 达到：

1. accuracy gap < 3%；
2. validation loss gap < 10%；
3. peak memory 比 adapter BP 降低 > 15%。

此时 wall-clock 可以略高。

#### 中等成功

相比 adapter BP + checkpointing：

1. peak memory 降低 > 10%；
2. accuracy gap < 2%；
3. wall-clock 不超过 1.5 倍；
4. router overhead < 20%。

#### 强成功

相比 adapter BP + checkpointing：

1. peak memory 降低 > 20%；
2. accuracy gap < 1%；
3. wall-clock 不超过 1.2 倍；
4. exact VJP audit 不需要每步做；
5. learned router 多数 block 满足：

$$
\cos>0.9
$$

---

### 7.8 失败标准

如果 DG-LCA 出现以下情况，则不能宣称低显存优势：

1. 性能明显低于 adapter BP；
2. memory 只比 vanilla BP 好，但不如 checkpointing；
3. wall-clock 超过 2 倍；
4. router / audit overhead 太大；
5. 需要每步 exact VJP teacher 才能稳定；
6. learned router 在 deployment mode 明显退化。

此时最多只能说：

> local VJP distillation 是一个可诊断的研究工具，但尚未形成实际训练收益。

---

## 8. 必须完整比较的 Baseline 清单

### 8.1 BP / Memory Baseline

1. vanilla BP；
2. activation checkpointing；
3. reversible / invertible block；
4. local exact VJP；
5. adapter-only BP；
6. adapter-only BP + checkpointing。

---

### 8.2 Credit Assignment Baseline

1. synthetic gradients；
2. feedback alignment；
3. direct feedback alignment；
4. local loss；
5. target propagation；
6. random feedback negative control。

---

### 8.3 KAN / Functional Update Baseline

1. coefficient SGD；
2. coefficient Adam；
3. coefficient AdamW；
4. Sobolev preconditioned update；
5. RKHS preconditioned update；
6. diagonal preconditioner；
7. K-FAC / natural-gradient-like local method。

---

### 8.4 Router Baseline

1. identity router；
2. diagonal router；
3. diagonal + low-rank router；
4. block-diagonal router；
5. attention-style linear-in-credit router；
6. nonlinear router as negative / ablation control。

---

## 9. 消融实验

### 9.1 Router rank 消融

比较：

$$
r\in\{0,4,8,16,32,64\}
$$

其中 $r=0$ 对应 diagonal-only。

报告：

1. cosine；
2. relative error；
3. memory；
4. compute；
5. downstream loss。

---

### 9.2 Metric 消融

比较：

1. $G_k=I$；
2. diagonal $G_k$；
3. whitened diagonal $G_k$；
4. damped diagonal $G_k$。

所有 inverse 必须使用 damping：

$$
G_k^{-1}
\rightarrow
(G_k+\rho I)^{-1}
$$

---

### 9.3 Functional metric 消融

比较：

1. no preconditioning；
2. diagonal preconditioning；
3. Sobolev metric；
4. RKHS metric；
5. different damping $\rho$；
6. different smoothness weights $\alpha,\beta$。

Sobolev metric：

$$
\|\phi\|_{\mathcal H}^2
=
\int \phi(t)^2dt
+
\alpha\int \phi'(t)^2dt
+
\beta\int \phi''(t)^2dt
$$

---

### 9.4 Teacher frequency 消融

比较：

1. every step exact VJP；
2. every 5 steps exact VJP；
3. every 20 steps exact VJP；
4. every 100 steps exact VJP；
5. no exact VJP after warmup。

报告：

1. router stale；
2. self-routed drift；
3. final performance；
4. compute overhead。

---

### 9.5 Teacher-forced vs self-routed

必须分开报告：

teacher-forced：

$$
\hat g_k=C_k(h_k,h_{k+1})[g_{k+1}^{\text{BP}}]
$$

self-routed：

$$
\hat g_k=C_k(h_k,h_{k+1})[\hat g_{k+1}]
$$

如果 teacher-forced 好而 self-routed 坏，不能 claim deployment 可用。

---

## 10. 推荐第一轮实验配置

若资源有限，第一轮只做四组实验。

---

### 实验 A：Credit rank diagnostic

#### 数据集

1. MNIST；
2. Fashion-MNIST；
3. CIFAR-10。

#### 模型

1. 4-layer MLP；
2. 8-layer MLP；
3. small residual MLP；
4. KAN-like MLP。

#### 目标指标

$$
E_{64}^{(2)},\quad E_{128}^{(2)},\quad r_{\text{PR}}
$$

#### 成功标准

至少在 KAN-like / adapter 子空间中：

$$
E_{128}^{(2)}>0.9
$$

---

### 实验 B：Single-block VJP distillation

#### 数据集

1. Two Moons；
2. MNIST；
3. CIFAR-10。

#### 模型

1. MLP block；
2. KAN-like block；
3. adapter block。

#### Router

1. diagonal；
2. diagonal + low-rank $r=8$；
3. diagonal + low-rank $r=16$；
4. diagonal + low-rank $r=32$。

#### 成功标准

$$
\cos>0.95
$$

$$
\operatorname{relerr}<0.2
$$

$$
0.8<\rho<1.2
$$

---

### 实验 C：KAN functional update

#### 数据集

1. 1D function regression；
2. 2D function regression；
3. MNIST；
4. Fashion-MNIST。

#### 比较

1. coefficient Adam；
2. Sobolev preconditioned update；
3. RKHS preconditioned update。

#### 成功标准

1. validation loss 不差；
2. 收敛更快；
3. $\max|\phi'|$ 更低；
4. $\int|\phi''|^2$ 更低；
5. Jacobian spectrum 更稳定。

---

### 实验 D：8-block sequential routing

#### 数据集

1. MNIST；
2. Fashion-MNIST；
3. CIFAR-10 small。

#### 比较

1. exact BP；
2. local exact VJP；
3. learned router teacher-forced；
4. learned router self-routed；
5. synthetic gradient；
6. feedback alignment。

#### 成功标准

$$
\cos(\hat g_k,g_k^{\text{BP}})>0.8
$$

accuracy gap < 3%。

若该实验失败，不进入 adapter-only 大实验。

---

## 11. 成功结果的分级解释

### 11.1 最低限度成功

若只证明：

1. KAN-like functional update 比 coefficient BP / Adam 更稳定；
2. single-block VJP router 在小模块上达到：

$$
\cos>0.9
$$

则可作为一个保守贡献：

> Local VJP Distillation and Functional-Space Updates for Structured Neural Modules.

不能声称替代 BP。

---

### 11.2 中等成功

若进一步证明：

1. credit 子空间在 adapter / KAN-like 模块中可压缩；
2. single-block router 在多个模块中稳定；
3. 4/8-block sequential routing 不崩；
4. KAN preconditioning 有实际优化收益；

则可以声称：

> DG-LCA 是一种有希望的局部化 BP 组件。

---

### 11.3 强成功

若进一步证明：

1. 8/12-block self-routed training 接近 BP；
2. adapter-only DG-LCA 相比 checkpointed adapter BP 有 memory 优势；
3. performance gap < 1% 到 2%；
4. compute overhead 可控；
5. exact VJP audit 不需要每步做；

则可以较强地声称：

> 在结构化低维模块中，learned local VJP + functional-space preconditioning 可以部分替代 full BP 的局部组件，并带来实际 memory / efficiency tradeoff。

---

## 12. 失败结果的解释

失败结果同样重要。至少需要记录：

1. 哪些层 credit rank 高；
2. 哪些层 router 学不准；
3. teacher-forced 好但 self-routed 坏的情况；
4. norm ratio 是否失控；
5. router stale 多快发生；
6. functional update 是否过平滑；
7. predicted descent 和 actual descent 是否不一致；
8. memory 优势是否被 router / audit overhead 吃掉；
9. 哪些 baseline 明显强于 DG-LCA；
10. 哪些结构化子空间仍然可行。

---

## 13. 最终判定表

| 实验结果 | 结论 |
|---|---|
| Gate 0 失败 | learned router 主线不适合 full hidden space，收缩到 adapter / KAN |
| Gate 1 失败 | router 架构不够，不能做 sequential routing |
| Gate 2 成功 | functional-space preconditioning 是独立可保留贡献 |
| Gate 3 失败 | 多层 learned credit routing 暂不可行，只能做 local / teacher-forced |
| Gate 4 失败 | 不能 claim 低显存优势 |
| Gate 1 + Gate 2 成功 | 可形成保守论文 |
| Gate 1 + Gate 2 + Gate 3 成功 | 可 claim 局部 BP 组件可蒸馏 |
| Gate 1 到 Gate 4 全部成功 | 才能 claim 在结构化模块中有实际训练收益 |

---

## 14. 最终建议

第一篇论文不要追求证明：

$$
\text{DG-LCA replaces BP}
$$

而应证明：

$$
\text{local VJP can be distilled}
+
\text{KAN / adapter functional update can be stabilized}
+
\text{multi-layer routing boundaries can be measured}
+
\text{cost tradeoff can be audited}
$$

最稳的论文标题可以是：

> **Local VJP Distillation and Functional-Space Updates for Structured Neural Modules**

或中文：

> **结构化神经模块中的局部 VJP 蒸馏与函数空间更新**

第一篇的最低成功目标是：

1. credit 子空间在结构化模块中可压缩；
2. single-block local VJP router 能高保真拟合；
3. KAN-like functional-space preconditioning 改善稳定性；
4. sequential routing 的成功 / 失败边界被系统刻画。

这已经是一个严肃、可信、可发表的研究结果。

