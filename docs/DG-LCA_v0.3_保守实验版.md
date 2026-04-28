# DG-LCA v0.3：双几何局部 Credit Assignment 研究方案（保守实验版）

> 版本：v0.3  
> 状态：研究计划，不是已成立算法  
> 核心修订：进一步收缩 claim，把主线从“大规模替代 BP”改为“结构化模块上的 local VJP distillation + functional-space preconditioning + 低显存可行性验证”。  
> 建议论文定位：**Local VJP Distillation and Functional-Space Updates for Structured Neural Modules**。

---

## 0. 一句话定位

DG-LCA v0.3 不再试图直接提出一个“通用 BP 替代算法”。它研究的是一个更窄、更可验证的问题：

> **能否把全局 BP 中的局部 VJP 操作蒸馏为条件线性的 learned VJP router，并在 KAN-like / adapter 等结构化模块上，用函数空间预条件更新替代普通参数坐标梯度，从而在特定场景下降低 activation memory、改善局部优化几何，或提升模块训练的可并行性？**

换句话说，v0.3 的目标是：

$$
\text{full end-to-end BP}
\quad\longrightarrow\quad
\text{local VJP distillation} + \text{local functional update} + \text{strict audits}
$$

它不是：

$$
\text{BP replacement for all neural networks}
$$

而是：

$$
\text{a staged research program for amortizing local BP components}
$$

---

## 1. 本版相对 v0.2 的核心修订

### 1.1 主张进一步收缩

v0.2 的表述仍然容易被理解为：

$$
\text{DG-LCA can replace global BP}
$$

v0.3 改成：

$$
\text{DG-LCA tries to distill and amortize local VJP under structure}
$$

也就是说，本版只主张：

1. 在结构化模块中，local VJP 可能可被低成本 router 近似；
2. 在 KAN-like / adapter 模块中，函数空间更新可能比普通参数坐标梯度更稳定；
3. 低显存收益必须通过完整 cost accounting 证明；
4. macro-router / tree routing 暂时不是主线。

---

### 1.2 macro-router 和 tree routing 降级为后续探索

v0.2 中的 macro-router：

$$
C_{a:b}\approx G_a^{-1}(DF_{a:b})^\top G_b
$$

理论上有吸引力，但风险很高，因为它本质上压缩的是一段长 Jacobian product。v0.3 把它降级为 future work，只保留小窗口 span $2,4,8$ 的后期实验。

主线不再依赖：

$$
O(\log K)\ \text{tree routing}
$$

因为这个说法最多对应 critical path latency，不代表总计算量下降。

---

### 1.3 router 必须是“条件线性算子”

真实 VJP 对输出端 credit 是线性的。因此 router 不应写成普通非线性函数：

$$
C_k^\phi(h_k,h_{k+1},g_{k+1})
$$

而应写成条件线性算子：

$$
\hat g_k=C_k^\phi(h_k,h_{k+1})[g_{k+1}]
$$

其中 $h_k,h_{k+1}$ 可以决定算子本身，但算子对 $g_{k+1}$ 必须保持线性。

---

### 1.4 credit rank 诊断改用平方奇异值能量

若 credit 矩阵为：

$$
\Lambda_k=[\lambda_k^{(1)},\dots,\lambda_k^{(B)}]
$$

SVD 为：

$$
\Lambda_k=U_kS_kV_k^\top
$$

奇异值为 $s_i$。若要衡量 PCA / covariance 意义下的 explained variance，应使用平方奇异值：

$$
E_r^{(2)}=\frac{\sum_{i=1}^r s_i^2}{\sum_i s_i^2}
$$

可辅助报告 nuclear energy：

$$
E_r^{(1)}=\frac{\sum_{i=1}^r s_i}{\sum_i s_i}
$$

但主指标应为 $E_r^{(2)}$。

---

### 1.5 local exact VJP 的角色分成三种模式

v0.3 明确区分：

1. **diagnostic mode**：用 exact BP / exact VJP 测量真实 credit 与真实局部 VJP；
2. **distillation mode**：用 local exact VJP 训练 router；
3. **deployment mode**：减少 exact VJP 频率，改用 learned router、scheduled exact audit 或 adjoint probe。

这样可以避免“说不用 BP，但 teacher 还是 BP”的歧义。

---

### 1.6 functional update 改写为 metric preconditioning

KAN-like functional update 本质上不是魔法，而是函数空间 Gram matrix 对 coefficient gradient 的预条件：

$$
a\leftarrow a-\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

其中 $M$ 是函数空间 metric / Gram matrix，$\rho I$ 是 damping。它应与 Adam、K-FAC、natural gradient、Sobolev training 等做比较。

---

## 2. BP 的分解：我们到底要拆什么？

将模型分成 $K$ 个 block：

$$
h_{k+1}=F_k(h_k;\theta_k),\quad k=0,\dots,K-1
$$

最终损失为：

$$
\mathcal L=\ell(h_K,y)
$$

标准 BP 中，输出端 credit 为：

$$
\lambda_K=\nabla_{h_K}\ell
$$

局部 VJP 递推为：

$$
\lambda_k=J_k^\top\lambda_{k+1}
$$

其中：

$$
J_k=\frac{\partial F_k}{\partial h_k}
$$

参数梯度为：

$$
\nabla_{\theta_k}\mathcal L
=
D_{\theta_k}F_k(h_k;\theta_k)^\top\lambda_{k+1}
$$

这说明一个关键事实：

> **只要 block 已经拿到输出端 credit $\lambda_{k+1}$，它的参数更新本身是局部的。BP 的全局性主要来自 $\lambda_K\to\lambda_0$ 的长程 cotangent transport。**

因此 DG-LCA v0.3 研究的对象不是“梯度下降是否存在”，而是：

$$
\boxed{\text{how to localize, distill, and amortize VJP / cotangent transport}}
$$

---

## 3. metric-aware credit transport：正确公式与限制

如果第 $k$ 层状态空间有 metric：

$$
G_k\succ 0
$$

内积定义为：

$$
\langle u,v\rangle_{G_k}=u^\top G_kv
$$

若 $g_k$ 是 metric 下的 gradient vector，则 covector 为：

$$
\lambda_k=G_kg_k
$$

因为：

$$
D\mathcal L[h_k][\delta h_k]
=
\langle g_k,\delta h_k\rangle_{G_k}
=
g_k^\top G_k\delta h_k
$$

又因为：

$$
\delta h_{k+1}=J_k\delta h_k
$$

所以：

$$
g_k^\top G_k\delta h_k
=
g_{k+1}^\top G_{k+1}J_k\delta h_k
$$

于是：

$$
G_kg_k=J_k^\top G_{k+1}g_{k+1}
$$

即：

$$
\boxed{
g_k=G_k^{-1}J_k^\top G_{k+1}g_{k+1}
}
$$

这就是 metric-aware local VJP。

但这个公式不意味着 metric-aware 一定更好。它依赖：

1. $G_k$ 在相关子空间上可逆；
2. $G_k^{-1}$ 不病态；
3. metric 不放大 router 噪声；
4. metric 的估计或学习成本可控；
5. 若 $G_k$ 依赖状态 $h_k$，其变化不会引入新的不稳定性。

因此 v0.3 中，metric 首先采用简单、可控的形式：

$$
G_k=I
$$

或 diagonal / whitened / damped diagonal：

$$
G_k=\operatorname{diag}(\sigma_k^2+\epsilon)
$$

并要求所有 inverse 使用 damping 和 clipping：

$$
G_k^{-1}\quad\rightarrow\quad (G_k+\rho I)^{-1}
$$

---

## 4. DG-LCA v0.3 的主线：learned local VJP router

### 4.1 条件线性 router

真实 local VJP 是线性算子：

$$
g_{k+1}\mapsto G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

因此 router 定义为：

$$
\hat g_k=C_k^\phi(h_k,h_{k+1})[g_{k+1}]
$$

它不是一个任意非线性函数，而是由 $h_k,h_{k+1}$ 条件化出来的线性算子。

---

### 4.2 最小结构：diagonal + low-rank

最小 router 采用：

$$
C_k^\phi(h)[g]
=
D_k(h)\odot g+U_k(h)(V_k(h)^\top g)
$$

其中：

- $D_k(h)$ 是 diagonal gating；
- $U_k(h),V_k(h)$ 是低秩因子；
- rank $r$ 可以从 $8,16,32$ 开始；
- $h$ 表示 $(h_k,h_{k+1})$ 或其 compressed sketch。

这个结构便宜，但表达力有限。它适合 KAN-like、MLP block、小 adapter，不适合作为 full Transformer block 的默认假设。

---

### 4.3 attention-style credit router

对 token 状态，可以使用 attention-style credit router：

$$
C_k^\phi(h)[g]
=
A_{\text{tok}}(h)gB_{\text{chan}}(h)^\top
$$

这里 $A_{\text{tok}}(h)$ 和 $B_{\text{chan}}(h)$ 由 hidden states 决定，但它们对 $g$ 的作用必须保持线性。

若使用多头结构：

$$
C_k^\phi(h)[g]
=
\sum_{m=1}^M A_m(h)gB_m(h)^\top
$$

必须监控 router 的总计算和显存，防止它比 local VJP 本身更贵。

---

### 4.4 线性性证书

即使结构上已保证线性，也应监控：

$$
r_{\text{lin}}
=
\frac{
\|C(\alpha v_1+\beta v_2)-\alpha C(v_1)-\beta C(v_2)\|
}{
\alpha\|C(v_1)\|+\beta\|C(v_2)\|+\epsilon
}
$$

若 router 是非线性实现，则必须加入线性性正则：

$$
\mathcal L_{\text{lin}}
=
\|C(\alpha v_1+\beta v_2)-\alpha C(v_1)-\beta C(v_2)\|^2
$$

---

## 5. router 的三种使用模式

### 5.1 Diagnostic mode

使用 full BP 或 local exact VJP 测量真实 credit：

$$
g_k^{\text{true}}=G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

目的：

1. 测 credit rank；
2. 测真实 VJP 的结构；
3. 评估 router 逼近难度；
4. 计算评估指标，不用于 claim 低成本训练。

---

### 5.2 Distillation mode

使用 local exact VJP 作为 teacher：

$$
g_k^{\text{teacher}}=G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

训练 router：

$$
\hat g_k=C_k^\phi(h_k,h_{k+1})[g_{k+1}]
$$

损失：

$$
\mathcal L_{\text{vjp},k}
=
\frac{\|\hat g_k-g_k^{\text{teacher}}\|^2}
{\|g_k^{\text{teacher}}\|^2+\epsilon}
$$

cosine loss：

$$
\mathcal L_{\text{cos},k}
=
1-
\frac{
\langle \hat g_k,g_k^{\text{teacher}}\rangle
}{
\|\hat g_k\|\|g_k^{\text{teacher}}\|+\epsilon
}
$$

norm ratio loss：

$$
\mathcal L_{\text{norm},k}
=
\left(
\frac{\|\hat g_k\|}{\|g_k^{\text{teacher}}\|+\epsilon}-1
\right)^2
$$

总损失：

$$
\mathcal L_{\text{router},k}
=
\mathcal L_{\text{vjp},k}
+\alpha\mathcal L_{\text{cos},k}
+\beta\mathcal L_{\text{norm},k}
+\gamma\mathcal L_{\text{lin},k}
$$

这一步仍然用 local autograd，不声称低成本。它只验证：

$$
\text{router architecture can approximate local VJP}
$$

---

### 5.3 Deployment mode

减少 exact VJP 频率，使用 learned router 进行局部 credit transport：

$$
\hat g_k=C_k^\phi(h_k,h_{k+1})[\hat g_{k+1}]
$$

但必须保留 scheduled audit：

$$
\text{every }T\text{ steps, compute exact local VJP for a subset of blocks}
$$

如果 audit 失败，则触发：

1. 降低学习率；
2. 增加 exact VJP 蒸馏频率；
3. 重置或微调 router；
4. 缩小 block；
5. 回退到 local BP。

---

## 6. credit 子空间诊断：Gate 0

### 6.1 目标

先验证一个必要条件：

$$
\text{observed credit directions are compressible}
$$

如果真实 credit 在 hidden space 中没有低维结构，则 learned router 很难比 BP 更划算。

---

### 6.2 数据收集

在 diagnostic mode 下，收集每层 credit：

$$
\lambda_k^{(n)}
$$

构造：

$$
\Lambda_k=[\lambda_k^{(1)},\dots,\lambda_k^{(B)}]
$$

做 SVD：

$$
\Lambda_k=U_kS_kV_k^\top
$$

---

### 6.3 主指标

令：

$$
p_i=\frac{s_i^2}{\sum_j s_j^2}
$$

variance energy：

$$
E_r^{(2)}=\sum_{i=1}^r p_i
$$

participation rank：

$$
r_{\text{PR}}=\frac{1}{\sum_i p_i^2}
$$

entropy rank：

$$
r_{\text{ent}}=\exp\left(-\sum_i p_i\log(p_i+\epsilon)\right)
$$

辅助指标：

$$
E_r^{(1)}=\frac{\sum_{i=1}^r s_i}{\sum_i s_i}
$$

---

### 6.4 必须避免的假低秩

credit rank 诊断必须跨以下维度重复：

1. 不同 batch size；
2. 不同训练阶段；
3. 不同 layer；
4. 不同任务或数据分布；
5. teacher credit 与 self-routed credit；
6. hidden space credit 与 adapter / LoRA subspace credit。

小 batch 会天然导致低秩，因此不能只用单个 mini-batch 的 rank 作为证据。

---

### 6.5 建议通过条件

可作为非硬性通过门槛：

$$
E_{64}^{(2)}>0.8
$$

或：

$$
E_{128}^{(2)}>0.9
$$

并且该结构在多层、多训练阶段中稳定。如果不满足，主线应收缩到 adapter subspace 或 KAN-like small module。

---

## 7. single-block local VJP distillation：Gate 1

### 7.1 目标

验证：

$$
C_k^\phi(h_k,h_{k+1})[g_{k+1}]
\approx
G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

在单个 block 上是否可行。

---

### 7.2 评估指标

cosine：

$$
\cos_k
=
\frac{\langle \hat g_k,g_k^{\text{teacher}}\rangle}
{\|\hat g_k\|\|g_k^{\text{teacher}}\|+\epsilon}
$$

relative error：

$$
\operatorname{relerr}_k
=
\frac{\|\hat g_k-g_k^{\text{teacher}}\|}
{\|g_k^{\text{teacher}}\|+\epsilon}
$$

norm ratio：

$$
\rho_k
=
\frac{\|\hat g_k\|}{\|g_k^{\text{teacher}}\|+\epsilon}
$$

scale generalization：

$$
C(ag)\approx aC(g)
$$

superposition generalization：

$$
C(g_1+g_2)\approx C(g_1)+C(g_2)
$$

---

### 7.3 建议通过条件

早期可设：

$$
\cos_k>0.9
$$

更强目标：

$$
\cos_k>0.95,\quad \operatorname{relerr}_k<0.2,\quad |\rho_k-1|<0.2
$$

如果 single-block router 长期达不到这些门槛，不应继续做 sequential learned routing。

---

### 7.4 router stale 问题

block 参数更新后，$J_k$ 会变化，因此 router 会过期。需要监控：

$$
\Delta_{\text{stale},k}
=
\|C_k^{\phi,t+\tau}-C_k^{\text{teacher},t+\tau}\|
$$

实际用 cosine / relerr 近似评估。若 stale 太快，则必须增加 router update 频率或限制 block 参数变化：

$$
\|\theta_k^{t+1}-\theta_k^t\|\leq \delta
$$

---

## 8. KAN-like functional-space update：Gate 2

### 8.1 为什么选 KAN-like module

KAN-like 层：

$$
y_j=\sum_i\phi_{ji}(x_i)
$$

其中 $\phi_{ji}$ 是一元 edge function。该结构适合作为最小实验，因为：

1. local Jacobian 清晰；
2. credit transport 由 $\phi'_{ji}(x_i)$ 控制；
3. functional update 可写成一维函数更新；
4. Sobolev / RKHS metric 容易定义；
5. sufficient statistics 较小。

---

### 8.2 KAN 局部 VJP

前向扰动：

$$
\delta y_j=\sum_i\phi'_{ji}(x_i)\delta x_i
$$

输出端 credit 为 $g_j$ 时，输入端 credit 为：

$$
g_{x_i}=\sum_j g_j\phi'_{ji}(x_i)
$$

这说明 KAN 的 backward credit transport 由一维函数导数控制。

---

### 8.3 函数空间更新

对 edge function 扰动：

$$
\phi_{ji}\mapsto \phi_{ji}+\delta\phi_{ji}
$$

有：

$$
D_{\phi_{ji}}\mathcal L[\delta\phi]
=
\mathbb E[\tilde g_j\delta\phi(x_i)]
$$

其中 $\tilde g_j$ 是输出端 covector 分量；若 $g_j$ 是 metric-gradient，则：

$$
\tilde g=G_y g
$$

若选择 RKHS kernel $K(t,s)$，则函数空间更新为：

$$
\Delta\phi_{ji}(t)
=
-\eta\mathbb E[\tilde g_jK(t,x_i)]
$$

mini-batch 形式：

$$
\Delta\phi_{ji}(t)
=
-\eta\frac{1}{B}\sum_{n=1}^B \tilde g_j^{(n)}K(t,x_i^{(n)})
$$

若用 basis：

$$
\phi(t)=\sum_m a_mB_m(t)
$$

普通 coefficient BP：

$$
a\leftarrow a-\eta\nabla_a\mathcal L
$$

函数空间预条件更新：

$$
a\leftarrow a-\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

其中：

$$
M_{mn}=\langle B_m,B_n\rangle_{\mathcal H}
$$

---

### 8.4 Sobolev metric

可选：

$$
\|\phi\|_{\mathcal H}^2
=
\int \phi(t)^2dt
+\alpha\int \phi'(t)^2dt
+\beta\int \phi''(t)^2dt
$$

对应 Gram matrix：

$$
M_{mn}
=
\int B_mB_n
+\alpha\int B'_mB'_n
+\beta\int B''_mB''_n
$$

必须使用 damping：

$$
M^{-1}\rightarrow (M+\rho I)^{-1}
$$

并监控条件数：

$$
\kappa(M+\rho I)
$$

---

### 8.5 KAN update 评估指标

除了 loss，还要测：

$$
\max_t |\phi'_{ji}(t)|
$$

$$
\int |\phi''_{ji}(t)|^2dt
$$

局部 Jacobian 谱：

$$
\sigma_{\max}(J),\quad \sigma_{\min}(J)
$$

predicted descent：

$$
\Delta\mathcal L_{\text{pred}}
=
D_{F_k}\mathcal L[\Delta F_k]
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
\frac{\Delta\mathcal L_{\text{actual}}}
{\Delta\mathcal L_{\text{pred}}+\epsilon}
$$

若 predicted descent 为负但 actual loss 经常不降，说明 functional update 或 credit 近似不可信。

---

## 9. sequential learned routing audit：Gate 3

### 9.1 目标

在小模型中测试 learned router 串联后是否误差爆炸。

真实递推：

$$
g_k^{\text{true}}=C_k^{\text{true}}g_{k+1}^{\text{true}}
$$

learned 递推：

$$
\hat g_k=(C_k^{\text{true}}+E_k)\hat g_{k+1}
$$

误差近似：

$$
e_k=C_k^{\text{true}}e_{k+1}+E_kg_{k+1}^{\text{true}}+E_ke_{k+1}
$$

这说明即使单个 router 误差小，多层后仍可能累积。

---

### 9.2 必测指标

每层 cosine：

$$
\cos(\hat g_k,g_k^{\text{BP}})
$$

每层 relative error：

$$
\frac{\|\hat g_k-g_k^{\text{BP}}\|}{\|g_k^{\text{BP}}\|+\epsilon}
$$

sequential drift：

$$
\operatorname{drift}_k
=
\|\hat g_k^{\text{self}}-\hat g_k^{\text{teacher-forced}}\|
$$

predicted vs actual descent：

$$
\frac{\Delta\mathcal L_{\text{actual}}}{\Delta\mathcal L_{\text{pred}}+\epsilon}
$$

---

### 9.3 teacher-forced 与 self-routed 两种评估

teacher-forced：

$$
\hat g_k=C_k^\phi(h_k,h_{k+1})[g_{k+1}^{\text{BP}}]
$$

self-routed：

$$
\hat g_k=C_k^\phi(h_k,h_{k+1})[\hat g_{k+1}]
$$

若 teacher-forced 好而 self-routed 坏，说明主要问题是分布漂移与误差累积。

---

### 9.4 通过条件

小模型 $4,8,12$ block 上：

$$
\cos(\hat g_k,g_k^{\text{BP}})>0.8
$$

且 loss update 后 actual descent 与 predicted descent 大体一致。

若 $8$ block 就明显崩溃，应停止向更深模型扩展。

---

## 10. adapter-only DG-LCA：Gate 4

### 10.1 为什么从 adapter 开始

full hidden state 的 VJP 可能太复杂。adapter / LoRA 子空间更低维，更适合 learned local router。

设：

$$
h_{k+1}=h_k+F_k^{\text{frozen}}(h_k)+A_k(h_k;\psi_k)
$$

其中主干 frozen，只训练 adapter 参数 $\psi_k$。

此时 credit 可投影到 adapter 输出子空间：

$$
g_{k+1}^{\text{adapter}}=P_k g_{k+1}
$$

router 只需在该子空间中近似 VJP。

---

### 10.2 目标

比较：

1. full BP adapter training；
2. checkpointed BP adapter training；
3. local BP through adapter；
4. learned VJP router + local adapter update；
5. synthetic gradient baseline；
6. feedback alignment baseline。

指标：

1. accuracy / loss；
2. memory peak；
3. wall-clock；
4. router overhead；
5. gradient cosine；
6. predicted vs actual descent。

---

## 11. macro-router：暂时不是主线

### 11.1 为什么降级

macro-router 要学习：

$$
C_{a:b}\approx G_a^{-1}(DF_{a:b})^\top G_b
$$

这相当于压缩一段长 Jacobian product，是 BP 最困难部分之一。

composition loss：

$$
\|C_{a:c}(v)-C_{a:b}(C_{b:c}(v))\|^2
$$

只能提供自洽性，不能提供正确性。全零 router 也能满足 composition consistency。

---

### 11.2 若后期探索，必须有真实锚点

macro-router 的监督必须包含：

$$
\langle C_{a:b}(v),u\rangle_{G_a}
\approx
\langle v,J_{a:b}u\rangle_{G_b}
$$

或 window exact VJP teacher：

$$
g_a^{\text{teacher}}=G_a^{-1}J_{a:b}^\top G_bg_b
$$

composition loss 只能作为辅助项：

$$
\mathcal L_{\text{macro}}
=
\mathcal L_{\text{anchor}}+\beta\mathcal L_{\text{comp}}
$$

---

### 11.3 小 span 限制

仅考虑：

$$
|b-a|\in\{2,4,8\}
$$

并必须提供 intermediate sketch：

$$
s_{a:b}=\operatorname{sketch}(h_a,h_{a+1},\dots,h_b,\theta_{a:b})
$$

否则仅凭 $(h_a,h_b)$ 通常不足以推断 span Jacobian。

---

## 12. 信息保持：从随机 probe 改成子空间谱估计

随机 norm ratio：

$$
\frac{\|F_k(h_k+\epsilon u)-F_k(h_k)\|}{\epsilon\|u\|}
$$

只能测一个方向，不能证明 bi-Lipschitz。

v0.3 只要求在任务相关切空间 $\mathcal T_k$ 上估计谱：

$$
\sigma_{\min}(J_k|_{\mathcal T_k}),\quad \sigma_{\max}(J_k|_{\mathcal T_k})
$$

其中 $\mathcal T_k$ 可由以下方向张成：

1. observed credit directions；
2. activation perturbation directions；
3. adapter update directions；
4. augmentation-induced tangent directions；
5. PCA top directions；
6. random directions as control。

spectral certificate：

$$
r_{\text{spec},k}
=
\max\left(
|\sigma_{\max}(J_k|_{\mathcal T_k})-1|,
|\sigma_{\min}(J_k|_{\mathcal T_k})-1|
\right)
$$

这仍不是严格证明，只是比随机单方向 probe 更可信。

---

## 13. functional update 的最优性与 projection residual

局部函数空间更新定义为：

$$
\Delta F_k^\star
=
\arg\min_{\Delta F\in\mathcal F_k}
D\mathcal L[\Delta F]+
\frac{1}{2\eta}\|\Delta F\|_{M_k}^2
$$

最优性条件：

$$
\nabla_{M_k}\mathcal L+\frac{1}{\eta}\Delta F_k^\star=0
$$

functional residual：

$$
r_{\text{func},k}
=
\left\|
\nabla_{M_k}\mathcal L+\frac{1}{\eta}\Delta F_k
\right\|_{M_k^{-1}}
$$

但即使存在理想 $\Delta F_k^\star$，当前参数化也未必能实现它。若参数更新后得到 $F_k^{\text{new}}$，则 projection residual 为：

$$
r_{\text{proj},k}
=
\frac{
\|F_k^{\text{new}}-(F_k+\Delta F_k^\star)\|_{\mu_k}
}{
\|\Delta F_k^\star\|_{\mu_k}+\epsilon
}
$$

其中 $\mu_k$ 是当前 batch 或 hidden-state 分布。

若 $r_{\text{proj},k}$ 高，说明函数空间中的理想更新无法被当前模块参数化有效实现。

---

## 14. 低显存 claim 的严格 cost accounting

DG-LCA v0.3 不直接声称低显存优于 BP。必须比较完整成本。

### 14.1 memory accounting

标准 BP：

$$
M_{\text{BP}}
=
M_{\text{params}}+M_{\text{optim}}+M_{\text{activations}}
$$

DG-LCA：

$$
M_{\text{DG}}
=
M_{\text{params}}+M_{\text{optim}}+M_{\text{interfaces}}+M_{\text{router}}+M_{\text{stats}}+M_{\text{local}}
$$

其中：

- $M_{\text{interfaces}}$：block boundary states；
- $M_{\text{router}}$：router 参数和激活；
- $M_{\text{stats}}$：credit rank、metric、sketch、audit buffers；
- $M_{\text{local}}$：local update 所需短期激活。

只有当：

$$
M_{\text{DG}}<M_{\text{checkpoint}}\quad\text{or}\quad M_{\text{DG}}<M_{\text{adapter+checkpoint}}
$$

才可声称相对强 baseline 有显存优势。

---

### 14.2 compute accounting

DG-LCA 总计算：

$$
C_{\text{DG}}
=
C_{\text{forward}}+C_{\text{router}}+C_{\text{teacher/audit}}+C_{\text{probe}}+C_{\text{local update}}+C_{\text{metric}}
$$

必须与：

$$
C_{\text{BP}},\quad C_{\text{checkpoint}},\quad C_{\text{reversible}},\quad C_{\text{adapter BP}}
$$

比较。

若 exact VJP teacher 每步都需要，DG-LCA 可能只是“带额外 router 的 BP”，没有成本优势。

---

## 15. 实验路线：v0.3 推荐顺序

专家建议的最稳路线是：

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

macro-router / tree routing 暂不进入主线。

---

### Gate 0：credit 子空间诊断

目标：判断 learned router 是否有低维压缩空间。

失败条件：

$$
E_{128}^{(2)}\ll 0.9
$$

并且跨训练阶段、层、batch 都不稳定。

若失败，收缩到 adapter / LoRA 子空间。

---

### Gate 1：single-block VJP router

目标：验证 router 能否拟合 local exact VJP。

失败条件：

$$
\cos_k<0.9
$$

或 norm ratio 明显失控。

若失败，改 router 结构，或放弃 learned VJP 主线。

---

### Gate 2：KAN-like functional update

目标：验证函数空间预条件是否优于 coefficient BP / Adam。

对比：

$$
a\leftarrow a-\eta\nabla_a\mathcal L
$$

与：

$$
a\leftarrow a-\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

测 loss、泛化、Jacobian 谱、$\phi'$ 稳定性、actual descent。

---

### Gate 3：sequential routing audit

目标：验证 learned router 串联后的误差累积。

先做 $4,8,12$ block。若 8-block 已经崩溃，不扩展。

---

### Gate 4：adapter-only DG-LCA

目标：在低维结构化子空间测试是否有实际 memory / compute / performance tradeoff。

不要直接跳到 full Transformer / LLM full fine-tuning。

---

## 16. baseline 必须完整

至少包含：

1. vanilla BP；
2. activation checkpointing；
3. reversible / invertible block；
4. adapter-only BP；
5. adapter-only BP + checkpointing；
6. synthetic gradients；
7. feedback alignment / direct feedback alignment；
8. target propagation / local loss；
9. coefficient BP / Adam for KAN；
10. K-FAC / natural-gradient-like local preconditioning。

DG-LCA 不能只和 vanilla BP 比较。

---

## 17. EML 的位置：只作为 primitive-adjoint co-design 的提醒

EML 的核心启发不是直接支撑 DG-LCA，而是提醒：

> **forward primitive 的表达能力不等于 adjoint primitive 的稳定性。**

例如：

$$
\operatorname{eml}(x,y)=\exp(x)-\ln(y)
$$

其导数为：

$$
\frac{\partial}{\partial x}\operatorname{eml}(x,y)=\exp(x)
$$

$$
\frac{\partial}{\partial y}\operatorname{eml}(x,y)=-\frac{1}{y}
$$

对应 adjoint rule：

$$
\lambda_x=\exp(x)\lambda_z
$$

$$
\lambda_y=-\frac{1}{y}\lambda_z
$$

这说明它虽然 forward 表达力强，但 adjoint 可能有爆炸和奇异风险。因此 EML 在本方案中只作为反例型启发：

$$
\boxed{\text{design forward primitive together with its adjoint geometry}}
$$

不要把 EML 当成 DG-LCA 的技术依据。

---

## 18. 本方案当前最可能失败的地方

### 18.1 credit 子空间不低维

如果 observed credit 高维且快速变化，router 没有压缩优势。

### 18.2 router 从 teacher-forced 到 self-routed 分布漂移

训练时：

$$
g_{k+1}\sim g_{k+1}^{\text{BP}}
$$

使用时：

$$
g_{k+1}\sim \hat g_{k+1}
$$

分布不同，误差可能累积。

### 18.3 router 太弱或太贵

低秩 router 便宜但可能不准；attention-style router 准但可能不省。

### 18.4 metric 放大噪声

$G_k^{-1}$ 或 $M_k^{-1}$ 可能病态，必须 damping / clipping。

### 18.5 functional update 无法投影回当前参数化

函数空间理想更新存在，不代表当前 block 能实现。

### 18.6 低显存优势打不过强 baseline

checkpointing、reversible blocks、adapter-only BP 已经很强，DG-LCA 必须做完整 cost accounting。

### 18.7 macro-router 学不到

macro-router 是后期高风险探索，不进入 v0.3 主线。

---

## 19. v0.3 的最终研究命题

v0.3 的核心命题是：

> **在结构化神经模块中，BP 的局部 VJP 操作可能可以被条件线性 learned router 蒸馏；当局部模块具有自然函数空间 metric 时，其参数更新可以被改写成 metric-preconditioned functional update。若 credit 子空间可压缩、router 误差可控、functional update 可投影、且成本优于强 baseline，则 DG-LCA 可作为一种局部化 BP 组件，而非通用 BP 替代。**

形式化压缩为：

$$
\hat g_k=C_k^\phi(h_k,h_{k+1})[g_{k+1}]
\approx
G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

$$
\Delta F_k^\star
=
\arg\min_{\Delta F}
D\mathcal L[\Delta F]+
\frac{1}{2\eta}\|\Delta F\|_{M_k}^2
$$

$$
\theta_k^{\text{new}}
\approx
\operatorname{Project}_{\Theta_k}(F_k+
\Delta F_k^\star)
$$

但所有 claim 都必须通过 Gate 0 到 Gate 4 逐步验证。

---

## 20. 最小可发表主线

最稳的论文题目不应是：

> DG-LCA: A Replacement for Backpropagation

而应是：

> **Local VJP Distillation and Functional-Space Updates for Structured Neural Modules**

核心贡献：

1. 把 BP 的长程 credit assignment 分解为 local VJP distillation 问题；
2. 提出条件线性 learned VJP router，并系统评估其单 block 和 sequential 误差；
3. 在 KAN-like module 中实现 Sobolev/RKHS functional-space preconditioning；
4. 提供 credit rank、VJP fidelity、descent alignment、projection residual、cost accounting 等严格评估协议；
5. 证明或否定该路线在 adapter / KAN-like 结构化模块中的可行性。

---

## 21. 参考工作

- Decoupled Neural Interfaces using Synthetic Gradients.
- Difference Target Propagation.
- Random synaptic feedback weights support error backpropagation for deep learning.
- Training Deep Nets with Sublinear Memory Cost.
- The Reversible Residual Network: Backpropagation Without Storing Activations.
- KAN: Kolmogorov-Arnold Networks.
- All elementary functions from a single binary operator.

