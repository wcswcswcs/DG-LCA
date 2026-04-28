# DG-LCA v0.2：双几何局部 Credit Assignment 研究方案（专家修订版）

> 版本：v0.2  
> 定位：研究方案草稿，不是已解决方法  
> 核心收缩：从“通用替代 BP”收缩为“局部 VJP / credit transport 蒸馏 + 结构化函数空间更新 + 低显存 adapter / KAN-like 模块训练”

---

## 0. 一句话定位

DG-LCA v0.2 的目标不是宣称“替代反向传播”，而是研究一个更小、更可验证的问题：

> **能否把全局 BP 的长程 VJP 链分解为可局部蒸馏的 learned VJP operators，并在结构化模块上用函数空间更新替代普通参数坐标梯度，从而在部分场景下降低 activation memory、提高模块并行度，或改善优化几何？**

更简洁地说：

> **DG-LCA v0.2 = learned local VJP + metric-aware credit transport + local functional update + low-memory structured modules。**

它不是一个“大一统 BP 替代范式”，而是一条可以逐关验证的研究路线。

---

## 1. 背景：BP 的缺陷到底在哪里？

BP 的问题不在于“算不出梯度”。在给定计算图、参数化和 loss 的情况下，BP 是高效且精确的梯度计算方法。问题在于它把学习过程高度绑定在以下几个结构上：

1. 当前网络的参数化方式；
2. 当前计算图的长路径链式传播；
3. 中间激活的存储；
4. 最终 loss 对所有层的远端监督；
5. 中间表示是否健康、信息是否保存、Jacobian 谱是否稳定。

标准网络可写为：

$$
h_{k+1}=F_k(h_k;\theta_k), \quad k=0,\dots,K-1
$$

最终损失为：

$$
\mathcal L=\ell(h_K,y)
$$

BP 的反向 credit 递推是：

$$
\lambda_K=\nabla_{h_K}\ell
$$

$$
\lambda_k=J_k^\top\lambda_{k+1}
$$

其中：

$$
J_k=\frac{\partial F_k}{\partial h_k}
$$

参数梯度是：

$$
\nabla_{\theta_k}\mathcal L
=
D_{\theta_k}F_k(h_k;\theta_k)^\top\lambda_{k+1}
$$

因此，BP 的全局痛点主要在于：

$$
\boxed{
\lambda_K\rightarrow\lambda_{K-1}\rightarrow\cdots\rightarrow\lambda_0
}
$$

也就是长程 cotangent / credit transport，而不是“单个 block 拿到输出端 credit 后如何更新参数”。

---

## 2. v0.1 方案的主要问题

专家意见指出，v0.1 的核心洞察有价值，但存在过度乐观之处。本版方案显式收缩这些 claim。

### 2.1 adjoint identity 正确，但不等于 router 低成本可学

v0.1 使用局部恒等式：

$$
\langle C_k^\phi(v),u\rangle_{G_k}
\approx
\langle v,J_ku\rangle_{G_{k+1}}
$$

这个式子是对的。但从 bilinear identity 到可靠 operator learning，中间还有：

- 样本复杂度；
- probe 分布选择；
- operator 结构约束；
- router 使用时的分布漂移；
- 高维状态空间下的可学习性；
- 多层误差累积。

所以 v0.2 不再宣称“adjoint probe 足以训练 router”。新的路线是：

> **先用 local exact VJP 蒸馏 router，验证 router 架构表达力；再逐步替换为 adjoint probe 或 mixed supervision。**

---

### 2.2 composition loss 不能单独训练 macro-router

v0.1 提出：

$$
C_{a:c}\approx C_{a:b}C_{b:c}
$$

并使用：

$$
\mathcal L_{\text{comp}}
=
\left\|C_{a:c}(v)-C_{a:b}(C_{b:c}(v))\right\|^2
$$

但这个 loss 有退化解。例如所有 router 都为零时，composition loss 也为零。

所以 v0.2 的修正是：

> **composition loss 只能作为辅助一致性约束，不能作为 macro-router 的主监督。**

macro-router 必须有真实锚点，例如 macro-level adjoint probe：

$$
\langle C_{a:b}(v),u\rangle_{G_a}
\approx
\langle v,J_{a:b}u\rangle_{G_b}
$$

其中：

$$
J_{a:b}=D(F_{b-1}\circ\cdots\circ F_a)(h_a)
$$

或者用 window-BP / local exact VJP 做蒸馏。

---

### 2.3 $O(\log K)$ 只能是 critical path，不是总计算量

v0.1 说 tree routing 可以把 credit 传播从 $O(K)$ 变成 $O(\log K)$。

v0.2 改为：

> **在足够硬件并行度、router 已训练好、macro-router 成本可控的前提下，credit routing 的 critical path latency 可能接近 $O(\log K)$；但总 router 调用数量和总计算量仍通常是 $O(K)$。**

因此，v0.2 不再把 tree routing 当作早期核心贡献。它被降级为后期探索目标。

---

### 2.4 仅凭 endpoint 不足以推断 span Jacobian

macro-router 若写成：

$$
C_{a:b}(h_a,h_b,g_b)\mapsto g_a
$$

通常信息不足。因为两个不同函数可能在一个样本上有相同输入输出，但局部导数不同。

所以 v0.2 要求 macro-router 至少包含额外条件信息：

$$
C_{a:b}(h_a,h_b,g_b,s_{a:b})\mapsto g_a
$$

其中 $s_{a:b}$ 可以是：

- span id；
- block parameter summary；
- intermediate activation sketch；
- low-rank Jacobian sketch；
- router hidden state；
- window-BP 蒸馏缓存。

早期实验中，先不做大跨度 macro-router，只做 span length $2,4,8$ 的小窗口。

---

### 2.5 随机方向 norm ratio 不能证明 bi-Lipschitz

v0.1 的信息保持证书：

$$
r_{\text{info},k}
=
\left|
\frac{\|F_k(h_k+\epsilon u)-F_k(h_k)\|}{\epsilon\|u\|}-1
\right|
$$

只能测试随机方向上的局部 norm ratio，不能证明最小奇异值和最大奇异值均受控。

v0.2 改为估计任务相关切空间上的局部谱：

$$
\sigma_{\min}(J_k|_{\mathcal T_k}),\quad
\sigma_{\max}(J_k|_{\mathcal T_k})
$$

其中 $\mathcal T_k$ 是任务相关切空间的估计，例如由以下向量张成：

- activation perturbation directions；
- observed credit directions；
- adapter update directions；
- low-rank PCA directions；
- data augmentation induced tangent directions。

对应证书改为：

$$
r_{\text{spec},k}
=
\max\left(
\left|\sigma_{\max}(J_k|_{\mathcal T_k})-1\right|,
\left|\sigma_{\min}(J_k|_{\mathcal T_k})-1\right|
\right)
$$

---

### 2.6 低显存 claim 必须与强 baseline 比较

v0.2 不再只和 vanilla BP 比较。显存收益必须与以下 baseline 比较：

1. vanilla BP；
2. activation checkpointing；
3. reversible / invertible blocks；
4. adapter-only training + checkpointing；
5. local BP through blocks；
6. DG-LCA v0.2 including router / probe / statistics overhead。

因此，v0.2 的显存 claim 改为：

> **DG-LCA 有潜力降低 full end-to-end BP 的 activation memory，但必须通过完整 cost accounting 证明其优于 checkpointing / reversible / adapter-only 等强 baseline。**

---

### 2.7 functional update certificate 要改写

v0.1 的：

$$
r_{\text{func},k}
=
D_{F_k}\mathcal L[\Delta F_k]+
\|\Delta F_k\|_{M_k}^2
$$

不是真正 residual。

如果局部函数空间更新来自：

$$
\Delta F_k^\star
=
\arg\min_{\Delta F}
D\mathcal L[\Delta F]
+
\frac{1}{2\eta}\|\Delta F\|_{M_k}^2
$$

最优性条件应为：

$$
\nabla_{M_k}\mathcal L+
\frac{1}{\eta}\Delta F_k=0
$$

所以 v0.2 的 functional update residual 改为：

$$
r_{\text{func},k}
=
\left\|
\nabla_{M_k}\mathcal L+
\frac{1}{\eta}\Delta F_k
\right\|_{M_k^{-1}}
$$

同时监控 predicted local descent：

$$
\Delta\mathcal L_{\text{pred},k}
=
D_{F_k}\mathcal L[\Delta F_k]
$$

要求：

$$
\Delta\mathcal L_{\text{pred},k}<0
$$

---

## 3. v0.2 的核心研究命题

v0.2 的命题是：

> **在结构化模块、低维 credit 子空间、局部 VJP 可蒸馏、以及函数空间更新可实现的条件下，全局 BP 的一部分 credit assignment 可以被局部化、蒸馏化和低显存化。**

形式化地说，标准 BP 的局部 VJP 是：

$$
g_k^{\text{true}}
=
G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

DG-LCA v0.2 学习一个结构化 router：

$$
\hat g_k
=
C_k^\phi(h_k,h_{k+1},g_{k+1})
$$

目标是：

$$
\hat g_k
\approx
G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

但这个近似只要求在训练相关的 credit 子空间上成立：

$$
g_{k+1}\in\mathcal S_{k+1}
$$

其中 $\mathcal S_{k+1}$ 是 observed credit subspace。

---

## 4. v0.2 的核心组件

DG-LCA v0.2 由五个组件组成。

### 4.1 组件 A：credit 子空间诊断

第一步不是训练 router，而是先问：

> 真实 BP credit 是否可压缩？

收集真实 BP 或 local BP credit：

$$
\lambda_k^{(n)}
$$

构造 credit 矩阵：

$$
\Lambda_k=[\lambda_k^{(1)},\dots,\lambda_k^{(B)}]
$$

做奇异值分解，得到奇异值：

$$
s_1\geq s_2\geq\cdots
$$

计算 effective rank：

$$
r_{\text{eff}}
=
\frac{\left(\sum_i s_i\right)^2}{\sum_i s_i^2}
$$

计算 top-$r$ energy：

$$
E_r
=
\frac{\sum_{i=1}^r s_i}{\sum_i s_i}
$$

如果 top 64 或 top 128 directions 不能解释大部分 credit energy，则 learned router 很可能不值得做。

---

### 4.2 组件 B：结构化 local VJP router

真实 pullback 对 $g_{k+1}$ 是线性的。因此 router 必须显式保持对 credit 输入的线性性。

错误形式：

$$
C_k^\phi(h_k,h_{k+1},g_{k+1})
$$

更合理形式：

$$
\hat g_k=A_k^\phi(h_k,h_{k+1})g_{k+1}
$$

其中 $A_k^\phi$ 不显式构造完整矩阵，而采用结构化形式。

#### Diagonal + low-rank 形式

$$
A_k^\phi g
=
D_k(h)g+U_k(h)V_k(h)^\top g
$$

其中：

- $D_k(h)$ 是 diagonal 或 block-diagonal；
- $U_k,V_k$ 是低秩因子；
- rank $r$ 是小常数，例如 $r=8,16,32$。

#### Attention-style credit router

对 token 状态，可使用：

$$
A_k^\phi g
=
\text{TokenMix}_k(h,g)+\text{ChannelMix}_k(h,g)
$$

但必须保持对 $g$ 线性。即 $h$ 可以决定 mixing weights，$g$ 只能被线性作用。

#### 线性性正则

$$
\mathcal L_{\text{lin}}
=
\left\|
C(\alpha v_1+\beta v_2)
-
\alpha C(v_1)-\beta C(v_2)
\right\|^2
$$

如果 router 本身结构已经保证线性性，则此项可作为监控指标。

---

### 4.3 组件 C：local exact VJP 蒸馏优先

v0.2 的训练顺序是：

1. 先用 local exact VJP 作为 teacher；
2. 再尝试 adjoint probe；
3. 最后尝试 finite-difference probe。

local exact VJP teacher：

$$
g_k^{\text{teacher}}
=
G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

router loss：

$$
\mathcal L_{\text{vjp},k}
=
\left\|
C_k^\phi(h_k,h_{k+1},g_{k+1})
-g_k^{\text{teacher}}
\right\|^2
$$

也可以加入 cosine loss：

$$
\mathcal L_{\text{cos},k}
=
1-
\frac{\left\langle \hat g_k,g_k^{\text{teacher}}\right\rangle}
{\|\hat g_k\|\|g_k^{\text{teacher}}\|}
$$

最终：

$$
\mathcal L_{\text{router},k}
=
\mathcal L_{\text{vjp},k}
+
\alpha\mathcal L_{\text{cos},k}
+
\beta\mathcal L_{\text{lin},k}
+
\gamma\mathcal L_{\text{norm},k}
$$

其中 norm regularization 可写为：

$$
\mathcal L_{\text{norm},k}
=
\left(
\frac{\|\hat g_k\|}{\|g_{k+1}\|}-c_k
\right)^2
$$

$c_k$ 可由 teacher VJP 的平均 norm ratio 估计。

---

### 4.4 组件 D：adjoint probe 作为后续替代监督

当 local VJP router 在 exact teacher 下表现足够好后，再引入 adjoint probe。

局部 identity：

$$
\langle C_k^\phi(v),u\rangle_{G_k}
\approx
\langle v,J_ku\rangle_{G_{k+1}}
$$

用 exact JVP：

$$
J_ku
$$

得到：

$$
\mathcal L_{\text{adj},k}
=
\left(
\langle C_k^\phi(v),u\rangle_{G_k}
-
\langle v,J_ku\rangle_{G_{k+1}}
\right)^2
$$

如果不能使用 exact JVP，再使用 finite difference：

$$
J_ku
\approx
\frac{F_k(h_k+\epsilon u)-F_k(h_k)}{\epsilon}
$$

finite difference 不应作为第一版实验的主方法，因为它成本高且对非平滑组件敏感。

---

### 4.5 组件 E：local functional update

当 block 拿到输出端 credit $g_{k+1}$ 后，不直接做参数梯度，而是先构造局部函数空间更新。

模块视为函数对象：

$$
F_k\in\mathcal F_k
$$

局部变分：

$$
D_{F_k}\mathcal L[\delta F_k]
=
\mathbb E
\left[
\left\langle
G_{k+1}g_{k+1},
\delta F_k(h_k)
\right\rangle
\right]
$$

选择函数空间 metric $M_k$，得到更新问题：

$$
\Delta F_k^\star
=
\arg\min_{\Delta F\in\mathcal F_k}
D_{F_k}\mathcal L[\Delta F]
+
\frac{1}{2\eta}\|\Delta F\|_{M_k}^2
$$

等价于：

$$
\Delta F_k^\star
=-\eta\nabla_{M_k}\mathcal L
$$

对于有限样本，也可写成 target fitting：

$$
\Delta F_k^\star
=
\arg\min_{\Delta F\in\mathcal F_k}
\sum_n
\left\|
\Delta F(h_k^{(n)})+
\eta g_{k+1}^{(n)}
\right\|_{G_{k+1}}^2
+
\Omega_k(\Delta F)
$$

这里 $\Omega_k$ 是函数空间正则，例如 Sobolev norm、RKHS norm、low-rank norm。

---

## 5. KAN-like 最小模块

KAN-like 模块适合作为最小实验对象，因为它的局部 Jacobian 和函数空间更新都很清楚。

一层 KAN-like block：

$$
y_j=\sum_i\phi_{ji}(x_i)
$$

其中：

$$
\phi_{ji}:\mathbb R\rightarrow\mathbb R
$$

前向扰动：

$$
\delta y_j
=
\sum_i\phi_{ji}'(x_i)\delta x_i
$$

因此输入端 credit：

$$
g_{x_i}
=
\sum_j g_j\phi_{ji}'(x_i)
$$

这说明 KAN-like primitive 的反向 credit transport 由一维函数导数 $\phi_{ji}'$ 决定。

---

## 6. KAN-like functional update

对某条边函数 $\phi_{ji}$，若输出端 credit 为 $g_j$，则：

$$
D_{\phi_{ji}}\mathcal L[\delta\phi]
=
\mathbb E[g_j\delta\phi(x_i)]
$$

若选择 RKHS kernel $K(t,s)$，functional gradient update 为：

$$
\Delta\phi_{ji}(t)
=
-\eta\mathbb E[g_jK(t,x_i)]
$$

mini-batch 形式：

$$
\Delta\phi_{ji}(t)
=
-\eta\frac{1}{B}\sum_{n=1}^B g_j^{(n)}K(t,x_i^{(n)})
$$

若用 spline basis：

$$
\phi_{ji}(t)=\sum_m a_{jim}B_m(t)
$$

普通 coefficient BP 是：

$$
a\leftarrow a-\eta\nabla_a\mathcal L
$$

函数空间更新则是：

$$
a\leftarrow a-\eta G^{-1}\nabla_a\mathcal L
$$

其中：

$$
G_{mn}=\langle B_m,B_n\rangle_{\mathcal H}
$$

若选择 Sobolev metric：

$$
\|\phi\|_{\mathcal H}^2
=
\int \phi(t)^2dt
+\alpha\int \phi'(t)^2dt
+\beta\int \phi''(t)^2dt
$$

则：

$$
G_{mn}
=
\int B_mB_n
+\alpha\int B_m'B_n'
+\beta\int B_m''B_n''
$$

这会自然控制函数平滑性、斜率和曲率，从而影响 credit transport 稳定性。

---

## 7. EML 在 v0.2 中的角色

EML 不再作为 DG-LCA 的直接技术支撑，而是作为一个原则性启发和反例。

EML primitive：

$$
\operatorname{eml}(x,y)=\exp(x)-\ln(y)
$$

它的局部 adjoint rule 是：

$$
\lambda_x=\exp(x)\lambda_z
$$

$$
\lambda_y=-\frac{1}{y}\lambda_z
$$

这说明：一个 primitive 的 forward 表达能力强，并不意味着它的 adjoint 稳定。$\exp(x)$ 和 $1/y$ 都可能导致 credit 爆炸或数值奇异。

因此，v0.2 从 EML 得到的原则是：

> **设计 forward primitive 时必须同时设计 adjoint primitive。**

一个适合 DG-LCA 的 primitive 应该满足：

1. forward expressive；
2. local adjoint low-cost；
3. cotangent amplification controllable；
4. composition stable；
5. local functional update cheap；
6. activation sufficient statistics small。

---

## 8. v0.2 的实验 Gate 设计

v0.2 采用 gate-based research plan。每个 gate 失败，就不推进下一步。

---

### Gate 0：credit 子空间诊断

目标：验证真实 credit 是否可压缩。

收集：

$$
\lambda_k
$$

指标：

$$
r_{\text{eff}}
=
\frac{\left(\sum_i s_i\right)^2}{\sum_i s_i^2}
$$

$$
E_r
=
\frac{\sum_{i=1}^r s_i}{\sum_i s_i}
$$

通过门槛示例：

- top 64 或 top 128 directions 解释大部分 credit energy；
- across layer 稳定；
- across training stage 稳定；
- across batch / data distribution 不完全崩坏。

如果 Gate 0 失败，则 learned router 不适合作为主路线，只能转向 adapter-only 或更强结构假设。

---

### Gate 1：single-block local VJP router

固定一个 block，用 exact local VJP 产生 teacher：

$$
g_k^{\text{teacher}}
=
G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

训练：

$$
\hat g_k=C_k^\phi(h_k,h_{k+1},g_{k+1})
$$

指标：

$$
\cos(\hat g_k,g_k^{\text{teacher}})
$$

$$
\frac{\|\hat g_k-g_k^{\text{teacher}}\|}{\|g_k^{\text{teacher}}\|}
$$

$$
\left\|C(\alpha v_1+\beta v_2)-\alpha C(v_1)-\beta C(v_2)\right\|
$$

建议通过门槛：

$$
\cos(\hat g_k,g_k^{\text{teacher}})>0.9
$$

如果 single-block router 都学不好，不进入多层实验。

---

### Gate 2：sequential routing error accumulation

比较以下方法：

1. exact BP；
2. local exact VJP；
3. learned local router；
4. synthetic gradient；
5. feedback alignment；
6. direct feedback alignment；
7. local loss；
8. target propagation。

核心指标：

$$
\cos(\hat g_k,g_k^{\text{BP}})
$$

以及 predicted descent：

$$
\Delta\mathcal L_{\text{pred}}
=
\sum_k
D_{F_k}\mathcal L[\Delta F_k]
$$

要求：

$$
\Delta\mathcal L_{\text{pred}}<0
$$

并且多步训练中保持稳定。

---

### Gate 3：KAN-like functional update

比较两种更新。

普通 coefficient update：

$$
a\leftarrow a-\eta\nabla_a\mathcal L
$$

函数空间更新：

$$
a\leftarrow a-\eta G^{-1}\nabla_a\mathcal L
$$

指标：

$$
\max_x |\phi'(x)|
$$

$$
\int |\phi''(x)|^2dx
$$

$$
\sigma_{\max}(J),\quad \sigma_{\min}(J)
$$

$$
\Delta\mathcal L_{\text{actual}}
$$

$$
\frac{\Delta\mathcal L_{\text{actual}}}{\Delta\mathcal L_{\text{pred}}}
$$

目标是验证 Sobolev/RKHS functional update 是否真的改善优化和 credit stability。

---

### Gate 4：small-span macro-router

不一开始做全树 macro-router。先做：

$$
C_{k:k+2},\quad C_{k:k+4},\quad C_{k:k+8}
$$

主监督必须是 macro adjoint probe 或 window exact VJP：

$$
\langle C_{a:b}(v),u\rangle_{G_a}
\approx
\langle v,J_{a:b}u\rangle_{G_b}
$$

composition loss 只作为辅助项：

$$
\mathcal L_{\text{comp}}
=
\left\|
C_{a:c}(v)-C_{a:b}(C_{b:c}(v))
\right\|^2
$$

若 small-span macro-router 无法稳定通过，则不推进 tree routing。

---

### Gate 5：完整 cost accounting

必须报告：

- activation memory；
- router parameter memory；
- router activation memory；
- probe cost；
- local VJP teacher cost；
- local functional update cost；
- wall-clock time；
- total FLOPs；
- communication overhead。

比较 baseline：

1. vanilla BP；
2. checkpointing；
3. reversible blocks；
4. adapter-only BP；
5. adapter-only + checkpointing；
6. synthetic gradient；
7. DG-LCA v0.2。

若 DG-LCA 只比 vanilla BP 省显存，但不如 checkpointing / reversible / adapter baseline，则不能宣称低显存优势。

---

## 9. v0.2 的训练流程

### Phase A：诊断

1. 用 BP 训练小模型；
2. 收集每层 credit；
3. 测 effective rank、top energy、credit stability；
4. 测 local Jacobian spectrum；
5. 确定是否值得训练 router。

---

### Phase B：single-block router distillation

1. 固定 block；
2. 用 local exact VJP 生成 teacher；
3. 训练 structured linear-in-credit router；
4. 测 cosine、relative error、linearity、scale generalization；
5. 若通过，再测试 adjoint probe supervision。

---

### Phase C：functional update

1. 使用 KAN-like block 或 adapter block；
2. 给定真实 credit；
3. 比较 coefficient BP 与 Sobolev/RKHS functional update；
4. 测 loss、Jacobian spectrum、函数平滑性、credit amplification。

---

### Phase D：multi-block learned routing

1. 用 learned router 顺序传播 credit；
2. 暂不做 macro-router；
3. 测误差累积；
4. 对比 synthetic gradient、feedback alignment、target propagation、local loss。

---

### Phase E：small-span macro-router

1. 训练 span length 2/4/8 macro-router；
2. 使用 macro adjoint probe 或 window-BP teacher；
3. 加 composition loss 作为辅助；
4. 测 critical path latency 与总 FLOPs。

---

### Phase F：adapter-only 低显存训练

1. 冻结主干；
2. 只训练 adapter / LoRA / KAN-like local modules；
3. 使用 learned local router 或 exact local VJP；
4. 测性能、显存、速度；
5. 与 adapter BP + checkpointing 比较。

---

## 10. 关键风险与修正策略

### 风险 1：credit 子空间不低维

如果 credit 分布高维无结构，则 router 学习困难。

修正：

- 限制到 adapter / LoRA 子空间；
- 限制到 head-wise / token-local 子空间；
- 只学习 top-$r$ credit projection；
- 使用 exact local VJP 而不是 learned router。

---

### 风险 2：router 使用时分布漂移

训练时：

$$
v\sim g_{k+1}^{\text{teacher}}
$$

使用时：

$$
v\sim \hat g_{k+1}
$$

修正：

- scheduled mixing teacher credit and predicted credit；
- 用 previous-router output 作为训练输入；
- periodic exact VJP audit；
- trust-region 限制 block 更新幅度。

---

### 风险 3：router 误差累积

若：

$$
C_k=C_k^{\text{true}}+E_k
$$

误差递推为：

$$
e_k
=
C_k^{\text{true}}e_{k+1}+E_kg_{k+1}+E_ke_{k+1}
$$

修正：

- 控制 router norm；
- 控制 interface spectrum；
- 缩短 learned routing span；
- 定期 window-BP 校准；
- 用 exact local VJP 替代高风险 block。

---

### 风险 4：macro-router 学不到长路径 Jacobian product

修正：

- 只做 span 2/4/8；
- 不用 composition loss 单独训练；
- 使用 macro adjoint probe 锚定；
- 引入 intermediate sketch；
- 把 tree routing 放到后期实验。

---

### 风险 5：functional update 无法投影回参数化模块

修正：

- 从 KAN-like / RKHS / small adapter 开始；
- 不直接用于完整 Transformer block；
- 用 target fitting 验证可实现性；
- 报告 projection residual：

$$
r_{\text{proj},k}
=
\left\|
F_k^{\text{new}}-(F_k+\Delta F_k^\star)
\right\|
$$

---

### 风险 6：显存收益不成立

修正：

- 完整统计 router / probe / local update overhead；
- 与 checkpointing / reversible / adapter BP 比较；
- 不宣称总计算更低，除非实测支持；
- 首先定位在 activation memory constrained 场景。

---

## 11. v0.2 与相关工作的关系

### Synthetic Gradients

DG-LCA v0.2 与 synthetic gradients 相近，但不是直接预测最终梯度：

$$
\hat g_k\approx\nabla_{h_k}\mathcal L
$$

而是学习局部 VJP operator：

$$
C_k\approx G_k^{-1}J_k^\top G_{k+1}
$$

因此它更像：

> **operator-level synthetic gradient / learned local VJP。**

---

### Target Propagation

Target propagation 传播 target，而 DG-LCA 传播 metric-aware credit。

两者可以通过 metric 关联：

$$
\tilde h_k=h_k-\eta g_k
$$

如果 $g_k$ 是 metric-aware credit，那么它可以生成局部 target。

---

### Feedback Alignment

DG-LCA 的 learned router 可以看成 learned feedback operator。它必须与：

- feedback alignment；
- direct feedback alignment；
- sign-symmetry feedback；

做 baseline 比较。

---

### Checkpointing / Reversible Nets

DG-LCA 的低显存目标必须与 checkpointing 和 reversible blocks 区分。

Checkpointing：仍然是 exact full BP，只是重算激活。  
Reversible nets：用可逆结构减少 activation storage。  
DG-LCA：尝试用 local credit router + local functional update 减少对 full BP graph 的依赖。

所以三者应作为强 baseline，而不是被忽略。

---

### KAN

KAN-like 模块是 DG-LCA 的最小实验场景。它的价值在于：

1. 局部 primitive 是一元函数；
2. local Jacobian 由 $\phi'(x)$ 给出；
3. functional update 可写为 RKHS/Sobolev update；
4. local sufficient statistics 较小。

---

### EML

EML 不作为直接技术支撑，而是提供一个警示：

> primitive 的 forward 表达力和 adjoint 稳定性必须同时考虑。

---

## 12. v0.2 的最终表述

DG-LCA v0.2 不再说：

> 我们提出一种通用替代 BP 的训练方法。

而应说：

> 我们研究一种局部化 BP 的路线：把全局 VJP 链拆成可蒸馏的 local VJP operators，并在结构化模块中使用 metric-corrected functional update。该方法旨在探索局部 credit assignment、低显存训练和优化几何改善之间的关系。

更正式的命题：

$$
\boxed{
\text{DG-LCA v0.2 studies whether learned local VJP operators plus metric-corrected functional updates can approximate selected parts of BP under structured, low-dimensional, and low-memory regimes.}
}
$$

中文：

> **DG-LCA v0.2 研究的是：在结构化、低维、低显存场景下，learned local VJP operator 与 metric-corrected functional update 能否近似 BP 的部分 credit assignment 功能。**

---

## 13. 最小可行课题标题

最靠谱的收缩版题目是：

> **用局部 VJP 蒸馏与 Sobolev/RKHS 函数空间更新实现结构化模块的低显存局部训练。**

英文可写为：

> **Local VJP Distillation and Functional-Space Updates for Memory-Efficient Training of Structured Neural Modules.**

或者更贴近 DG-LCA：

> **DG-LCA: Dual-Geometry Local Credit Assignment via Learned VJP Operators and Functional Updates.**

---

## 14. 当前最值得做的第一组实验

### 实验 A：KAN-like functional update

目标：验证函数空间更新是否优于 coefficient BP。

- 数据：函数拟合、CIFAR-small、toy classification；
- 模块：KAN-like layer；
- 比较：coefficient BP vs Sobolev/RKHS update；
- 指标：loss、generalization、$\phi'$、$\phi''$、Jacobian spectrum、credit amplification。

---

### 实验 B：single-block learned VJP

目标：验证 structured router 能否学到 local VJP。

- 模块：MLP block / KAN-like block / small residual block；
- teacher：local exact VJP；
- router：diagonal + low-rank / attention-linear-in-credit；
- 指标：cosine、relative error、linearity、scale generalization。

---

### 实验 C：multi-block sequential routing

目标：验证误差累积是否可控。

- 模型：4/8/12 block 小模型；
- 比较：BP、learned local router、synthetic gradient、feedback alignment、target propagation；
- 指标：layerwise gradient cosine、predicted descent、actual loss、training stability。

---

### 实验 D：adapter-only memory test

目标：验证低显存 claim。

- 冻结主干；
- 训练 small adapter / LoRA / KAN-like adapter；
- 比较：adapter BP、adapter BP + checkpointing、DG-LCA local update；
- 指标：memory、time、accuracy/loss。

---

## 15. 最终结论

DG-LCA v0.2 的可信版本不是“替代 BP”，而是一个分层研究项目：

1. **先诊断 credit 是否可压缩；**
2. **再验证 local VJP router 是否可蒸馏；**
3. **再验证 functional update 是否改善局部优化；**
4. **再测试多层误差累积；**
5. **最后才讨论 macro-router、tree routing 和低显存收益。**

它的核心洞察仍然保留：

$$
\text{BP 的全局痛点}
\approx
\text{long-range cotangent transport}
$$

但新版方案更谨慎：

$$
\text{replace BP generally}
\quad\longrightarrow\quad
\text{distill and approximate local VJP under structure}
$$

这使 DG-LCA 从一个过大的设想，变成一个可以严肃实验验证的研究方向。

