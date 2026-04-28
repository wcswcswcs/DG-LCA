# DG-LCA：双几何局部 Credit Assignment 研究方案

> 版本：v0.1  
> 目标：把反向传播从一个全局、串行、长路径、高显存的算法，分解成一组局部可学习、可并行、信息尽量无损、优化几何更好、显存更低的 credit assignment 机制。  
> 公式格式：Typora 友好，使用 `$...$` 与 `$$...$$`。

---

## 0. 一句话摘要

**DG-LCA**，即 **Dual-Geometry Local Credit Assignment**，试图把标准 BP 中的全局 cotangent transport 拆成：

$$
\text{信息保持 interface}
+
\text{metric-aware credit router}
+
\text{hierarchical composition}
+
\text{local functional update}
+
\text{low-memory sufficient statistics}
$$

它不是简单地“不要梯度”，也不是“每层加一个局部 loss”，而是把 BP 的核心对象：

$$
\lambda_k
=
J_k^\top \lambda_{k+1}
$$

重构为一组局部、可校准、可组合、可并行的 credit transport 机制，并且把每个模块拿到 credit 之后的参数更新，尽可能改写成函数空间中的局部更新。

---

## 1. 研究动机：BP 的结构性问题

BP 的根本局限不是“无法计算梯度”，而是它高度依赖：

1. 当前网络的计算图；
2. 当前参数化方式；
3. 长路径 Jacobian 链式传播；
4. 大量中间激活的保存；
5. 最终 loss 对所有早期层的远端监督；
6. 中间表示的健康性由架构被动保证，而不是由 BP 主动维护。

标准 BP 在数学上是正确的，但在深层模型、长序列模型、多模态模型、视频模型和复杂模块化系统中，会暴露几个问题。

### 1.1 参数化敏感

BP 在参数空间里做欧氏梯度下降。设网络为：

$$
f_\theta:X\to Y
$$

loss 为：

$$
\mathcal L(f_\theta)
$$

标准更新是：

$$
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta \mathcal L(f_\theta)
$$

如果换一个参数化：

$$
\theta=\phi(\eta)
$$

则梯度方向变为：

$$
\nabla_\eta \mathcal L
=
D\phi(\eta)^\top \nabla_\theta \mathcal L
$$

即使两个参数化表达的是相同函数族，优化路径也可能完全不同。ResNet 的残差参数化：

$$
H(x)=x+F(x)
$$

并没有改变 BP 的链式法则，但显著改变了优化几何。这说明 BP 本身不会自动选择“好参数化”。

### 1.2 显存开销高

标准 BP 需要保存大量中间激活，以便计算：

$$
\nabla_{\theta_k}\mathcal L
=
D_{\theta_k}F_k(h_k;\theta_k)^\top \lambda_{k+1}
$$

这通常需要保存：

- block 输入 $h_k$；
- block 内部中间激活；
- normalization 统计量；
- attention weights；
- mask、routing、dropout 等随机状态。

模型越深、序列越长、分辨率越高，activation memory 越成为训练瓶颈。

### 1.3 长路径 Jacobian 乘积不稳定

将网络分成 $K$ 个 block：

$$
h_{k+1}=F_k(h_k;\theta_k)
$$

最终损失为：

$$
\mathcal L=\ell(h_K,y)
$$

BP 的 credit 递推为：

$$
\lambda_K=\nabla_{h_K}\ell
$$

$$
\lambda_k
=
J_k^\top \lambda_{k+1}
$$

其中：

$$
J_k=\frac{\partial F_k}{\partial h_k}
$$

于是：

$$
\lambda_k
=
J_k^\top J_{k+1}^\top \cdots J_{K-1}^\top \lambda_K
$$

只要局部 Jacobian 的谱性质不稳定，长链乘积就会导致 credit 消失、爆炸、旋转或扭曲。

### 1.4 BP 不保证前向信息保留

BP 是误差信号分配机制，不是信息保持机制。即使最终 loss 能被降低，也不意味着中间层保持了足够健康、可逆、可控的表示。Residual stream、identity shortcut、normalization、gating、Attention Residual、Hyper-Connections、mHC 等结构，本质上都在为 BP 修建更稳定的信息通道。

### 1.5 远端 credit assignment 脆弱

早期层的更新依赖最终 loss 经过许多层间接传回。路径越长，credit 越容易混杂、衰减或变成噪声。复杂多模块系统中，这会使模块之间互相“甩锅”：一个模块没有学好，另一个模块可能用奇怪方式补偿它。

---

## 2. BP 的数学本质：global cotangent transport

BP 的核心不是“求梯度”这么笼统，而是一个 **cotangent transport** 问题。

前向传的是状态：

$$
h_0\to h_1\to \cdots \to h_K
$$

反向传的是 covector，也就是 cotangent vector：

$$
\lambda_k\in T_{h_k}^{*}\mathcal H_k
$$

标准 BP 做的是：

$$
\lambda_k
=
J_k^\top \lambda_{k+1}
$$

参数梯度是局部的：

$$
\nabla_{\theta_k}\mathcal L
=
D_{\theta_k}F_k(h_k;\theta_k)^\top \lambda_{k+1}
$$

因此，如果某个 block 已经拿到了正确的输出端 credit $\lambda_{k+1}$，它的参数更新其实是局部的。真正全局、串行、脆弱的是：

$$
\lambda_K
\to
\lambda_{K-1}
\to
\cdots
\to
\lambda_0
$$

也就是：

$$
\lambda_k
=
\left(D F_{k:K}\right)^\top \lambda_K
$$

其中：

$$
F_{k:K}=F_{K-1}\circ \cdots \circ F_k
$$

所以要分解 BP，真正要分解的是：

$$
\boxed{
\text{global cotangent transport}
}
$$

而不是简单地替换 loss 或给每层加辅助头。

---

## 3. 设计目标

DG-LCA 试图实现以下目标。

### 3.1 局部性

每个 block 的 credit transport 和参数更新主要依赖局部变量：

$$
h_k,\quad h_{k+1},\quad g_{k+1}
$$

而不是依赖完整网络的反向计算图。

### 3.2 可学习性

不要求显式计算完整 Jacobian，而是学习局部 credit router：

$$
C_k^\phi:
(h_k,h_{k+1},g_{k+1})
\mapsto
\hat g_k
$$

### 3.3 可并行性

通过 macro-router 和树状 credit routing，把 credit 传播深度从：

$$
O(K)
$$

降低到：

$$
O(\log K)
$$

### 3.4 信息尽量无损

每个 interface 在任务相关切空间上尽量保持 bi-Lipschitz：

$$
a\|\delta h_k\|
\leq
\|J_k\delta h_k\|
\leq
b\|\delta h_k\|
$$

其中 $a$ 不应太小，$b$ 不应太大。

### 3.5 优化几何更好

引入两套 metric：

- 状态空间 metric：$G_k$；
- 模块函数空间 metric：$M_k$。

credit transport 采用 metric-aware pullback：

$$
g_k
=
G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

模块更新采用 metric-corrected functional gradient。

### 3.6 显存更低

只长期保存 block interface：

$$
h_0,h_1,\dots,h_K
$$

而不是保存所有层内部 activation。block 内部状态通过重算、可逆结构或局部充分统计量处理。

---

## 4. 核心方案：DG-LCA

DG-LCA = **Dual-Geometry Local Credit Assignment**。

“双几何”指：

1. **状态空间几何**：隐藏状态如何传信息、如何传 credit；
2. **模块函数空间几何**：模块收到 credit 后，如何以函数空间方式更新。

---

## 5. 组件一：信息保持 interface

普通网络可以写为：

$$
h_{k+1}=F_k(h_k)
$$

DG-LCA 不希望 $F_k$ 是任意黑盒，而希望它至少在主信息通道上近似信息保持。

一种结构是拆分状态为：

$$
h_k=(m_k,c_k)
$$

其中：

- $m_k$ 是 public memory stream；
- $c_k$ 是 private computation stream。

更新为：

$$
m_{k+1}
=
m_k+\alpha_k U_k(m_k,c_k)
$$

其中 $\alpha_k$ 控制单层扰动幅度。

目标是在数据流形的任务相关切空间上满足：

$$
a\|\delta m_k\|
\leq
\|\delta m_{k+1}\|
\leq
b\|\delta m_k\|
$$

更理想地：

$$
a\approx b\approx 1
$$

这意味着：

- forward information 不容易丢；
- backward credit 不容易爆炸或消失；
- interface 可以作为低显存 checkpoint；
- 局部 router 的误差不容易被深度放大。

### 信息保持证书

用随机 perturbation $u$ 估计：

$$
r_{\text{info},k}
=
\left|
\frac{\|F_k(h_k+\epsilon u)-F_k(h_k)\|}
{\epsilon\|u\|}
-
1
\right|
$$

若 $r_{\text{info},k}$ 长期偏大，说明该 block 的前向/反向谱性质可能不健康。

---

## 6. 组件二：metric-aware credit transport

标准 BP 默认所有状态空间都是欧氏空间：

$$
\lambda_k=J_k^\top \lambda_{k+1}
$$

但如果每层状态空间有 metric $G_k$，则状态空间中的 gradient vector $g_k$ 应满足：

$$
D\mathcal L[h_k][\delta h_k]
=
\langle g_k,\delta h_k\rangle_{G_k}
$$

其中：

$$
\langle u,v\rangle_{G_k}=u^\top G_kv
$$

由：

$$
\delta h_{k+1}=J_k\delta h_k
$$

得到：

$$
\langle g_k,\delta h_k\rangle_{G_k}
=
\langle g_{k+1},J_k\delta h_k\rangle_{G_{k+1}}
$$

因此：

$$
g_k^\top G_k\delta h_k
=
g_{k+1}^\top G_{k+1}J_k\delta h_k
$$

所以：

$$
G_kg_k=J_k^\top G_{k+1}g_{k+1}
$$

即：

$$
\boxed{
g_k
=
G_k^{-1}J_k^\top G_{k+1}g_{k+1}
}
$$

DG-LCA 的 credit router 学习的是：

$$
C_k^\phi
\approx
G_k^{-1}J_k^\top G_{k+1}
$$

而不是裸的 $J_k^\top$。

这一步把 credit assignment 从欧氏参数空间中的链式法则，提升到状态空间几何中的 Riesz-corrected pullback。

---

## 7. 组件三：局部 credit router

每个 block 配一个 local credit router：

$$
C_k^\phi:
(h_k,h_{k+1},g_{k+1})
\mapsto
\hat g_k
$$

目标是：

$$
\hat g_k
\approx
G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

注意这里不要求 router 预测：

$$
\nabla_{h_k}\mathcal L
$$

的全局复杂结构。它只学习当前 block 的局部几何：

$$
h_k\mapsto h_{k+1}
$$

这和学习 value function 不同。value function 需要估计“从当前层往后的最终 loss”，而 local credit router 只需要估计当前 block 的 metric-aware pullback。

---

## 8. 组件四：用 adjoint identity 训练 router

如果用完整 BP 的 $\lambda_k$ 给 router 打标签，那仍然依赖全局 BP。因此 DG-LCA 使用局部 adjoint identity。

对任意输入扰动 $u$ 和输出 covector $v$，真实 Jacobian 满足：

$$
\langle J_k^\top v,u\rangle
=
\langle v,J_ku\rangle
$$

若使用 metric，目标为：

$$
\langle C_k^\phi(v),u\rangle_{G_k}
\approx
\langle v,J_ku\rangle_{G_{k+1}}
$$

而右边可以通过局部 forward probe 估计：

$$
J_ku
\approx
\frac{F_k(h_k+\epsilon u)-F_k(h_k)}{\epsilon}
$$

所以 router 的局部训练目标为：

$$
\mathcal L_{\text{adj},k}
=
\left(
\langle C_k^\phi(v),u\rangle_{G_k}
-
\left\langle
v,
\frac{F_k(h_k+\epsilon u)-F_k(h_k)}{\epsilon}
\right\rangle_{G_{k+1}}
\right)^2
$$

这只需要当前 block 的 forward perturbation，不需要全网络反传。

### Adjoint 一致性证书

$$
r_{\text{adj},k}
=
\left|
\langle C_k^\phi(v),u\rangle_{G_k}
-
\langle v,J_ku\rangle_{G_{k+1}}
\right|
$$

---

## 9. 组件五：只在真实 credit 子空间上学习

完整 $J_k^\top$ 太大，尤其在 Transformer 中，状态维度为：

$$
n_{\text{token}}\times d_{\text{model}}
$$

完整 Jacobian 是巨大的：

$$
(n_{\text{token}}d_{\text{model}})
\times
(n_{\text{token}}d_{\text{model}})
$$

DG-LCA 不要求 router 在整个空间上精确，只要求在真实训练会出现的 credit 子空间上精确：

$$
C_k^\phi v
\approx
G_k^{-1}J_k^\top G_{k+1}v,
\quad
v\in \mathcal S_k
$$

其中 $\mathcal S_k$ 是训练中出现的 credit directions 的低维结构化子空间。

因此第一批实验必须先测：

$$
\mathrm{rank}_\epsilon
\left(
\mathrm{Cov}(\lambda_k)
\right)
$$

以及：

$$
\mathrm{rank}_\epsilon
\left(
\mathrm{Cov}(\nabla_{\theta_k}\mathcal L)
\right)
$$

如果真实 credit directions 低秩、稀疏、head-wise、token-local 或低频，router 才有压缩空间。

---

## 10. 组件六：hierarchical macro-router

如果仍然逐层传播：

$$
g_K\to g_{K-1}\to \cdots \to g_0
$$

那么只是 learned BP，不够并行。

DG-LCA 引入 macro-router：

$$
C_{a:b}
\approx
G_a^{-1}
\left(D F_{a:b}\right)^\top
G_b
$$

其中：

$$
F_{a:b}=F_{b-1}\circ \cdots \circ F_a
$$

然后使用树状 credit routing：

$$
g_K
\to
g_{K/2}
\to
g_{K/4},g_{3K/4}
\to
\cdots
$$

这样 credit routing 深度由：

$$
O(K)
$$

变为：

$$
O(\log K)
$$

### 组合一致性

macro-router 必须满足：

$$
C_{a:c}
\approx
C_{a:b}C_{b:c}
$$

因此加入 composition loss：

$$
\mathcal L_{\text{comp}}
=
\left\|
C_{a:c}(v)
-
C_{a:b}(C_{b:c}(v))
\right\|^2
$$

### Composition 证书

$$
r_{\text{comp}}
=
\left\|
C_{a:c}(v)
-
C_{a:b}(C_{b:c}(v))
\right\|
$$

如果 $r_{\text{comp}}$ 大，则 tree credit routing 不可信，需要退回 sequential local router 或临时 window-BP 校准。

---

## 11. 组件七：local functional update

当 block 收到输出端 credit $g_{k+1}$ 后，标准 BP 做参数空间更新：

$$
\nabla_{\theta_k}\mathcal L
=
D_{\theta_k}F_k^\top G_{k+1}g_{k+1}
$$

DG-LCA 希望改成模块函数空间更新。

把当前 block 看成函数对象：

$$
F_k\in \mathcal F_k
$$

对 $F_k$ 的扰动 $\delta F_k$ 导致：

$$
h_{k+1}\mapsto h_{k+1}+\delta F_k(h_k)
$$

loss 的一阶变分为：

$$
D_{F_k}\mathcal L[\delta F_k]
=
\mathbb E
\left[
\left\langle
g_{k+1},
\delta F_k(h_k)
\right\rangle_{G_{k+1}}
\right]
$$

然后在函数空间 $\mathcal F_k$ 中选择 metric $M_k$，得到 functional gradient：

$$
\nabla_{\mathcal F_k}\mathcal L
$$

更新：

$$
F_k
\leftarrow
F_k
-
\eta\nabla_{\mathcal F_k}\mathcal L
$$

更实际地，可以解局部投影问题：

$$
\Delta F_k^\star
=
\arg\min_{\Delta F\in\mathcal F_k}
\sum_n
\left\|
\Delta F(h_k^{(n)})
+
\eta g_{k+1}^{(n)}
\right\|_{G_{k+1}}^2
+
\Omega_k(\Delta F)
$$

其中 $\Omega_k$ 是函数空间正则，例如 Sobolev norm、RKHS norm、low-rank norm 或 smoothness norm。

这一步把“局部参数梯度”改成“局部函数变化拟合”。

---

## 12. KAN-like primitive 的具体化

KAN 的启发在于：把标量权重替换为边上的可学习一元函数。一个 KAN-like 层可写为：

$$
y_j
=
\sum_i
\phi_{ji}(x_i)
$$

其中：

$$
\phi_{ji}:\mathbb R\to \mathbb R
$$

### 12.1 KAN 的局部 credit transport

前向微扰为：

$$
\delta y_j
=
\sum_i
\phi_{ji}'(x_i)\delta x_i
$$

所以局部 Jacobian 为：

$$
J_{ji}
=
\phi_{ji}'(x_i)
$$

输出端 credit 为 $g_j$ 时，输入端 credit 为：

$$
g_{x_i}
=
\sum_j
g_j\phi_{ji}'(x_i)
$$

也就是：

$$
g_x
=
\Phi'(x)^\top g_y
$$

这说明 KAN-like primitive 的 credit transport 由一元函数导数控制。

### 12.2 KAN 的 local functional update

对边函数 $\phi_{ji}$ 的扰动 $\delta\phi$：

$$
D_{\phi_{ji}}\mathcal L[\delta\phi]
=
\mathbb E[
g_j\delta\phi(x_i)
]
$$

如果选择 RKHS metric，kernel 为 $K(t,s)$，则 functional update 为：

$$
\Delta \phi_{ji}(t)
=
-\eta
\mathbb E[
g_jK(t,x_i)
]
$$

mini-batch 形式为：

$$
\Delta \phi_{ji}(t)
=
-\eta
\frac{1}{B}
\sum_{n=1}^{B}
g_j^{(n)}
K(t,x_i^{(n)})
$$

如果 $\phi$ 用 basis 表示：

$$
\phi(t)=\sum_m a_mB_m(t)
$$

普通 coefficient BP 为：

$$
a
\leftarrow
a-\eta\nabla_a\mathcal L
$$

函数空间更新应为：

$$
a
\leftarrow
a-\eta G^{-1}\nabla_a\mathcal L
$$

其中：

$$
G_{mn}
=
\langle B_m,B_n\rangle_{\mathcal H}
$$

如果选择 Sobolev metric：

$$
\|\phi\|_{\mathcal H}^2
=
\int \phi(t)^2dt
+
\alpha\int \phi'(t)^2dt
+
\beta\int \phi''(t)^2dt
$$

则：

$$
G_{mn}
=
\int B_mB_n
+
\alpha\int B'_mB'_n
+
\beta\int B''_mB''_n
$$

这会同时控制：

- edge function 的平滑性；
- $\phi'(t)$ 的大小；
- credit transport 的谱稳定性；
- local functional update 的条件数。

### 12.3 为什么 KAN-like block 适合作为最小实验对象

KAN-like primitive 具有几个优点：

1. 局部函数空间是一维的；
2. functional metric 容易定义；
3. local update 充分统计量很小，只需要 $(x_i,g_j)$；
4. credit transport 依赖 $\phi'(x_i)$，可直接监控；
5. Sobolev/RKHS metric 可以同时约束 forward 函数和 backward credit。

因此 DG-LCA 的最小实验应优先从 KAN-like block 开始，而不是直接挑战完整 Transformer。

---

## 13. EML 给出的 primitive-adjoint co-design 原则

EML 类工作的启发不是直接使用某个极端 primitive，而是提醒我们：

$$
\boxed{
\text{forward primitive 必须和 adjoint primitive 一起设计}
}
$$

假设一个二元 primitive：

$$
z=T(x,y)
$$

它的 forward rule 是：

$$
(x,y)\mapsto z
$$

但它的 credit rule 是：

$$
g_x
=
\left(\frac{\partial T}{\partial x}\right)^\top g_z
$$

$$
g_y
=
\left(\frac{\partial T}{\partial y}\right)^\top g_z
$$

如果 $T$ 的导数容易爆炸、消失或数值不稳定，那么这个 primitive 即使 forward 表达能力很强，也不适合深层局部 credit assignment。

因此 DG-LCA 的 primitive 设计原则是：

> 每个 forward primitive 都必须配一个低成本、谱稳定、可组合、metric-compatible 的 adjoint primitive。

一个 primitive 是否适合 DG-LCA，不只看表达能力，还要看：

$$
\left\|
\frac{\partial T}{\partial x}
\right\|,
\quad
\left\|
\frac{\partial T}{\partial y}
\right\|
$$

以及它们深层组合后的谱行为。

---

## 14. 完整训练流程

### Step 1：前向传播，只保存 block interface

计算：

$$
h_0,h_1,\dots,h_K
$$

只长期保存 block 边界状态，而不是保存所有内部 activation。

如果 block 可逆，可进一步降低保存量。若 block 不可逆，则在局部更新时重算内部状态。

---

### Step 2：顶部计算真实任务 credit

在最后一层计算：

$$
g_K
=
G_K^{-1}\nabla_{h_K}\ell(h_K,y)
$$

---

### Step 3：hierarchical credit routing

用 local-router 或 macro-router 生成各层 credit：

$$
\hat g_k
=
C_k^\phi(h_k,h_{k+1},\hat g_{k+1})
$$

或：

$$
\hat g_a
=
C_{a:b}^\phi(h_a,h_b,\hat g_b)
$$

树状传播使 credit routing 深度从 $O(K)$ 降低到 $O(\log K)$。

---

### Step 4：每个 block 并行做 local functional update

每个 block 解：

$$
\Delta F_k^\star
=
\arg\min_{\Delta F\in\mathcal F_k}
\sum_n
\left\|
\Delta F(h_k^{(n)})
+
\eta \hat g_{k+1}^{(n)}
\right\|_{G_{k+1}}^2
+
\Omega_k(\Delta F)
$$

然后：

$$
F_k\leftarrow F_k+\Delta F_k^\star
$$

对于 KAN-like primitive，这一步可分解成许多一维函数更新。

对于 Transformer，可以先只更新 adapter、LoRA 或小型 functional module。

---

### Step 5：局部训练 router

使用 local forward probe：

$$
\mathcal L_{\text{adj},k}
=
\left(
\langle C_k^\phi(v),u\rangle_{G_k}
-
\left\langle
v,
\frac{F_k(h_k+\epsilon u)-F_k(h_k)}{\epsilon}
\right\rangle_{G_{k+1}}
\right)^2
$$

并训练 macro-router 的 composition：

$$
\mathcal L_{\text{comp}}
=
\left\|
C_{a:c}(v)
-
C_{a:b}(C_{b:c}(v))
\right\|^2
$$

---

### Step 6：certificate audit

每轮训练监控：

$$
r_{\text{info},k}
$$

$$
r_{\text{adj},k}
$$

$$
r_{\text{comp}}
$$

以及函数更新证书：

$$
r_{\text{func},k}
=
D_{F_k}\mathcal L[\Delta F_k]
+
\|\Delta F_k\|_{M_k}^2
$$

如果某些 residual 失控，则采取：

1. 降低该 block 学习率；
2. 增加 router probe；
3. 缩小 block 粒度；
4. 调整 metric；
5. 暂时关闭 macro-router；
6. 使用 window-BP 做局部校准。

---

## 15. 低显存机制

标准 BP 的显存负担来自保存整条反向链所需的中间激活。DG-LCA 的长期状态主要是：

$$
h_0,h_1,\dots,h_K
$$

即 block interface。

block 内部有几种处理方式：

1. **重算**：局部更新时 recompute block 内部 activation；
2. **可逆**：通过 reversible block 从输出恢复输入；
3. **adapter-only**：主干冻结，只训练小模块；
4. **函数充分统计量**：对 KAN-like primitive，只保存 $(x_i,g_j)$；
5. **sketch**：保存压缩统计量而不是完整 activation。

因此显存结构从：

$$
O(\text{all layer activations})
$$

变成：

$$
O(\text{block interfaces}+\text{local sufficient statistics})
$$

DG-LCA 和 checkpointing 的区别是：checkpointing 仍然做 full global BP，只是用重算换显存；DG-LCA 改变了 credit assignment 的组织方式。

---

## 16. 最小实验路线

### 实验 1：KAN-like functional update

目的：验证函数空间更新是否比 coefficient-space BP 更稳定。

设置：

- 使用 KAN-like layer；
- credit 仍然用真实 BP 得到；
- 比较普通 coefficient BP 与 Sobolev/RKHS 更新。

比较：

$$
a\leftarrow a-\eta\nabla_a\mathcal L
$$

和：

$$
a\leftarrow a-\eta G^{-1}\nabla_a\mathcal L
$$

指标：

- 收敛速度；
- 泛化；
- $\phi'(t)$ 的稳定性；
- Jacobian spectrum；
- credit amplification；
- 函数平滑性；
- 训练显存。

若这一步无收益，DG-LCA 的 functional update 部分需要重构。

---

### 实验 2：单 block credit router

目的：验证 local router 是否能学习 metric-aware pullback。

固定一个 block，训练：

$$
C_k^\phi(v)
\approx
G_k^{-1}J_k^\top G_{k+1}v
$$

评价：

$$
\cos
\left(
C_k^\phi(v),
G_k^{-1}J_k^\top G_{k+1}v
\right)
$$

以及：

$$
r_{\text{adj},k}
$$

如果单 block router 学不好，就不进入多层实验。

---

### 实验 3：多 block sequential routing

目的：观察 router 误差如何随深度累积。

比较：

1. exact BP；
2. local router sequential；
3. synthetic gradient baseline；
4. local loss baseline。

指标：

- gradient cosine；
- loss gap；
- router residual；
- error accumulation；
- memory；
- wall-clock。

---

### 实验 4：macro-router tree routing

目的：验证并行 credit assignment 是否可行。

比较：

$$
C_{a:c}(v)
$$

与：

$$
C_{a:b}(C_{b:c}(v))
$$

指标：

- composition residual；
- tree routing depth；
- parallelism efficiency；
- training performance gap。

---

### 实验 5：adapter-only Transformer / diffusion block

目的：从受控环境进入真实模型。

设置：

- 冻结主干；
- 只训练 adapter、LoRA 或 memory module；
- 用 DG-LCA 做局部 credit assignment。

指标：

- 与 full BP adapter training 的性能差距；
- 显存下降；
- 是否支持异步/并行 block update；
- router 是否稳定。

---

## 17. 核心假设

DG-LCA 成立至少依赖以下假设。

### 假设一：真实 credit 子空间可压缩

训练中的 credit directions 不是任意高维随机分布，而具有低秩、稀疏、token-local、head-wise 或低频结构：

$$
\mathrm{rank}_\epsilon
\left(
\mathrm{Cov}(\lambda_k)
\right)
\ll
\dim(h_k)
$$

### 假设二：interface 可以做到近似信息保持

任务相关切空间上：

$$
a\|\delta h_k\|
\leq
\|J_k\delta h_k\|
\leq
b\|\delta h_k\|
$$

且 $a,b$ 不随深度恶化。

### 假设三：局部 pullback 可学习

存在结构化 router：

$$
C_k^\phi
$$

使其在真实 credit 子空间上逼近：

$$
G_k^{-1}J_k^\top G_{k+1}
$$

### 假设四：router 可组合

存在 macro-router 满足：

$$
C_{a:c}
\approx
C_{a:b}C_{b:c}
$$

且误差不会随深度爆炸。

### 假设五：local functional update 可实现

理想函数空间更新：

$$
\Delta F_k^\star
$$

能够被当前模块参数化近似实现。

---

## 18. 主要风险

### 风险一：credit 子空间并不低维

如果 $\lambda_k$ 高维且无结构，router 很难比 BP 更高效。

### 风险二：router 误差累积

若：

$$
C_k
=
G_k^{-1}J_k^\top G_{k+1}
+
E_k
$$

则 credit 误差可能满足：

$$
e_k
\approx
\sum_{j=k}^{K-1}
\left(
\prod_{i=k}^{j-1}
C_i
\right)
E_j g_{j+1}
$$

如果谱不稳定，局部误差会被深度放大。

### 风险三：metric 学错

如果 $G_k$ 或 $M_k$ 条件数差，metric correction 可能放大噪声。

### 风险四：macro-router 不可组合

若：

$$
C_{a:c}
\not\approx
C_{a:b}C_{b:c}
$$

则 tree routing 不可信。

### 风险五：functional update 难以投影回参数化

即使函数空间里有好更新，当前 block 的参数化也未必能实现它。

### 风险六：probe 开销过大

adjoint probe 虽然局部，但如果每步大量 probe，计算成本可能抵消并行和低显存收益。

---

## 19. 与已有方向的关系

### 19.1 与 synthetic gradients 的关系

Synthetic gradients 直接预测某层未来会收到的梯度。DG-LCA 不直接预测全局梯度，而是学习局部 metric-aware pullback：

$$
C_k^\phi
\approx
G_k^{-1}J_k^\top G_{k+1}
$$

因此它更局部、更可验证。

### 19.2 与 target propagation 的关系

Target propagation 传 target。DG-LCA 传 covector / gradient vector，并可通过 metric 转成 local target：

$$
\tilde h_k
=
h_k-\eta G_k^{-1}\lambda_k
$$

它不依赖完美 inverse，而依赖局部 adjoint consistency。

### 19.3 与 local loss 的关系

Local loss 给每层一个辅助目标，但不保证：

$$
\nabla_{h_k}\ell_k
\approx
(D F_{k:K})^\top\nabla_{h_K}\ell
$$

DG-LCA 的目标是逼近全局 cotangent pullback 的局部分解，而不是构造任意局部监督。

### 19.4 与 KAN 的关系

KAN 提供了 function-valued primitive，尤其适合作为 DG-LCA 的 local functional update 实验单元。KAN-like edge function 的 credit update 可写成一维函数空间中的 smoothing / regression。

### 19.5 与 EML 的关系

EML 的启发是：一个 forward primitive 的表达力不够，必须同时考虑它的 adjoint rule。DG-LCA 的 primitive 设计原则正是 forward-adjoint co-design。

---

## 20. 第一版研究命题

DG-LCA 的核心命题可以写成：

> 如果网络 interface 在任务相关切空间上近似信息保持，真实 credit 分布在低维结构化子空间内，并且局部 metric-aware pullback operator 可以通过 adjoint consistency 被学习和组合，那么全局 BP 的 cotangent transport 可以被分解成一组局部、可学习、可并行、低显存的 credit assignment 机制。

用公式概括：

$$
h_{k+1}=F_k(h_k)
$$

$$
g_k
\approx
C_k^\phi(h_k,h_{k+1},g_{k+1})
\approx
G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

$$
\Delta F_k
=
-\eta\nabla_{\mathcal F_k}\mathcal L
$$

$$
C_{a:c}
\approx
C_{a:b}C_{b:c}
$$

$$
a\|\delta h_k\|
\leq
\|J_k\delta h_k\|
\leq
b\|\delta h_k\|
$$

---

## 21. 当前最重要的下一步

不要直接训练大模型。先回答五个基础问题：

1. 真实网络中的 credit directions 是否低维可压缩？
2. KAN-like functional update 是否优于 coefficient BP？
3. 单个 local credit router 是否能通过 adjoint probe 学好？
4. router 误差是否会随深度可控累积？
5. macro-router 是否满足组合一致性并带来真实并行收益？

只有这些问题得到正面结果，DG-LCA 才值得向 Transformer、LLM、diffusion、video world model 扩展。

---

## 22. 参考与启发

1. 用户上传文档：`BP缺陷.md`。其中总结了 BP 对参数化敏感、计算与存储开销高、链式相乘怕深度、不保证前向信息保留、远端 credit assignment 脆弱、不主动维护中间表示健康性等结构性局限。
2. Ziming Liu et al., **KAN: Kolmogorov-Arnold Networks**, arXiv:2404.19756。KAN 将 MLP 中节点上的固定激活与线性权重结构，替换为边上的可学习一元函数，为 local functional update 提供了很好的 primitive。
3. Andrzej Odrzywołek, **All elementary functions from a single binary operator**, arXiv:2603.21852。该工作展示单一二元算子可生成初等函数 grammar，启发了 forward primitive 与 adjoint primitive 必须共同设计的原则。

---

## 23. 简短总结

DG-LCA 不是一个“完全不用 BP”的口号，而是一个更细的分解方案：

$$
\boxed{
\text{global BP}
\quad
\Rightarrow
\quad
\text{local credit transport}
+
\text{local functional update}
+
\text{information-preserving interface}
+
\text{hierarchical composition}
}
$$

它试图保留 BP 中最强的东西：精确的 credit assignment 思想；同时削弱 BP 中最麻烦的东西：全局串行反传、长路径 Jacobian 乘积、大量 activation 保存、参数空间几何不良、对中间表示健康性的被动依赖。

第一版最小可行方向是：

$$
\boxed{
\text{KAN-like block}
+
\text{Sobolev/RKHS functional update}
+
\text{local metric-aware credit router}
+
\text{adjoint consistency probe}
}
$$

如果这个最小系统能证明：

$$
\text{credit 近似可靠}
+
\text{functional update 更稳定}
+
\text{显存更低}
+
\text{误差可控}
$$

那么再继续扩展到 adapter-only Transformer、diffusion block 和 video memory module。
