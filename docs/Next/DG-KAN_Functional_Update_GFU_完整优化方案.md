# DG-KAN Functional Update 完整优化方案：从 Sobolev 预条件到 Geometry-aware Functional Optimizer

> 文件名：`DG-KAN_Functional_Update_GFU_完整优化方案.md`  
> 主题：用 functional-space update 增强 DG-KAN 的泛化能力、收敛速度与 credit geometry 稳定性  
> 核心方法：Geometry-aware Functional Update，简称 GFU  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`  
> 结论先行：当前最值得继续优化的不是 learned router，而是 functional update 本身。具体方向是把现有 `diag-to-Sobolev` update 升级为带有 trust-region、layer-wise branch target controller、data-weighted metric、M-aware momentum 和 online geometry trigger 的 functional optimizer。

---

## 0. 方案定位

DG-LCA / DG-KAN 目前已经从早期“能否学习一个 local router 替代 BP”的探索，逐步收敛到一个更清晰的主线：

$$
\text{AnalyticAdj}
+
\text{functional-space update}
+
\text{geometry-aware schedule}
+
\text{memory accounting}
$$

在这条主线上，KAN primitive 的 analytic adjoint 已经基本通过正确性验证，FirstOrderDecompAdj 也已经证明可以作为近似 credit transport；真正还没有完全优化好的部分，是 KAN edge function 的 functional-space update 在视觉分类任务中的 optimizer dynamics。

已有实验说明，functional update 确实可以显著降低 $\phi'_{p95}$、Jacobian condition、credit amplification，并且在一些任务上改善 validation loss AUC。然而它也暴露出一个稳定的瓶颈：如果 KAN branch under-active，模型精度会落后 AdamW；如果 branch 激活过强，Jacobian condition 和 $\phi'$ 会恶化。也就是说，functional update 当前不是方向错误，而是缺少一个能动态控制“函数空间步长、branch 活性、几何稳定性”的统一优化器。

因此本方案的目标不是重写整个 DG-KAN，而是在现有 `AnalyticAdj + diag-to-Sobolev functional update` 的基础上，设计一个更完整的 functional optimizer：

$$
\boxed{
\text{GFU}
=
\text{Trust-region functional step}
+
\text{Branch target controller}
+
\text{Metric continuation}
+
\text{M-aware momentum}
+
\text{Geometry-aware trigger}
}
$$

GFU 的目标是让 functional update 同时满足三件事：

$$
\text{fast descent}
\quad + \quad
\text{strong branch utilization}
\quad + \quad
\text{controlled } \phi' \text{ / Jacobian geometry}
$$

这不是一个单一 trick，而是把前面所有实验中已经观察到的正负经验，合成一个可实现、可审计、可逐步 ablate 的 optimizer 方案。

---

## 1. 之前实验给出的核心 insight

### 1.1 Residual DG-KAN 的信息保持 interface 是有效的

纯 KAN block 的局部 Jacobian condition 在多个任务上达到非常高的数量级，而 residual DG-KAN：

$$
h_{k+1}
=
h_k
+
\alpha_k \operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

显著改善了局部几何稳定性。这说明 residual interface 不是附属设计，而是 DG-KAN 能够承载 local credit assignment 的基础。它使 forward state 的主信息通道接近 identity，同时让 backward credit 不容易在单层内爆炸或消失。

这个结果带来两个后续含义。第一，DG-KAN 不应该为了让 router 显得更有价值而轻易去掉 residual。第二，所有 functional update 的 schedule 都不能只看 KAN edge function 本身，还必须看 residual branch 对整体 block 的实际扰动强度。

因此优化 functional update 时，真正应该控制的对象不是 raw coefficient norm，而是：

$$
r_{branch,k}
=
\frac{
\|\alpha_k \operatorname{KAN}_k(\operatorname{LN}(h_k))\|
}{
\|h_k\|+\epsilon
}
$$

以及更接近 backward geometry 的 effective derivative scale：

$$
s_k
=
|\alpha_k| \cdot \phi'_{p95,k}
$$

如果 $r_{branch,k}$ 太小，KAN branch 没有真正参与任务；如果 $s_k$ 太大，local Jacobian 和 credit amplification 会恶化。

---

### 1.2 Functional update 已经显示出优化和泛化潜力，但分类任务需要 hybrid

在回归任务中，Sobolev / diagonal functional update 已经显示出两个稳定现象。Diagonal metric 往往带来更快的 early descent，Sobolev full inverse 往往带来更好的最终函数拟合能力。Two Moons 分类任务中，KAN coefficient 使用 Sobolev functional update、非 KAN 参数继续使用 AdamW，可以追平 AdamW accuracy，同时降低 bad steps 和 Jacobian condition。

但在 MNIST / Fashion-MNIST 的小模型分类实验里，最初的 hybrid functional update 低于 AdamW。后续 drilldown 显示，主要原因不是泛化失败，也不是 basis coverage 失败，而是 KAN branch under-active。具体表现为 hybrid 的 branch ratio 只有 AdamW 的一小部分，关闭 KAN branch 后精度几乎不掉，说明 KAN branch 没有被充分利用。

当 coefficient learning rate 提高后，classification accuracy 可以恢复；但如果 coefficient learning rate 太高，又会让 $\phi'$ 和 Jacobian condition 变差。由此形成一个重要结论：

$$
\boxed{
\text{functional update 的核心难点不是方向是否可用，}
\text{而是有效步长和 branch 活性如何动态控制。}
}
$$

这也是本方案后续引入 trust-region 和 branch target controller 的直接原因。

---

### 1.3 Analytic adjoint 已经不是主要瓶颈

Stage C / Stage D 的实验已经证明，RBF-KAN primitive 的 analytic backward 与 autograd backward 在数值上对齐。对于 KAN primitive：

$$
y_j
=
\sum_i \phi_{ji}(x_i)
$$

local VJP 为：

$$
g_{x_i}
=
\sum_j g_{y_j}\phi'_{ji}(x_i)
$$

也就是：

$$
g_x
=
\Phi'(x)^\top g_y
$$

完整 residual block 的 VJP 为：

$$
g_h
=
g_{out}
+
D\operatorname{LN}(h)^\top
\left(
\alpha \Phi'(\operatorname{LN}(h))^\top g_{out}
\right)
$$

这些 analytic VJP 已经可以作为 reliable teacher 或直接训练路径。D0 中 AnalyticAdj-DGKAN 与 FullBP-DGKAN 的训练结果几乎一致，说明 credit correctness 在当前主线中不是主要瓶颈。

这意味着继续优化时，重点应该转向：收到正确 credit 之后，KAN coefficient 如何在函数空间中更好更新。

---

### 1.4 Learned router 的主要教训：hidden-only 不够，derivative-aware 才有信号

Stage C-next 系列实验给了一个很重要的反面和正面教训。由于 residual DG-KAN 的 block VJP 很接近 identity，直接让 router 学完整 $g_h$ 会产生伪成功。后来改成学习 correction：

$$
\Delta g
=
g_h^{teacher}-g_{out}
$$

并让 router 预测：

$$
\hat g_h
=
g_{out}+\hat{\Delta g}
$$

这个设计是正确的，但普通 scalar / ridge / low-rank / hidden-only router 仍然无法稳定学会 correction direction。v2 通过 span2 放大 correction 后，direction-norm router 有进展，但全局仍未达到进入 Stage D 主线的标准。

v3 的关键发现是，span2 correction 几乎完全由一阶 branch correction 解释，cross-term 只有较小比例。也就是说，问题不是 correction 本身太复杂，而是 hidden-only router 没有看到必要的 primitive derivative 信息。

这给 functional update 的优化也带来一个启示：不要寄希望于黑盒 optimizer 自动从 coefficient gradient 中学会所有函数几何。应该显式使用 primitive derivative、Sobolev metric、data-weighted basis statistics 和在线几何审计。

---

### 1.5 Stage F 的核心结论：schedule 能显著改善 functional update

Stage F 系列实验已经把问题推进到 optimizer dynamics。静态 functional update 可以获得更低 $\phi'$ 和 Jacobian condition，但 accuracy 常常低于 AdamW。branch-aware / loss-aware / geometry-aware schedule 后，Fashion-MNIST 已经出现非常强的结果：accuracy 高于 AdamW，同时 validation loss AUC 更好，$\phi'$ 和 Jacobian condition 仍显著低于 AdamW。

KMNIST 更难。固定大 boost 会导致 branch 和 Jacobian 失控；温和 boost 加 geometry-aware trigger 可以改善 static functional update，但仍有约 $1\%$ 左右的 accuracy gap。Threshold sweep 说明继续扫 `branch_high`、`jac_high`、`final_scale` 的收益开始平台化。

这说明下一步不应继续只调 trigger 阈值，而应该升级 optimizer 本身：引入 trust-region、layer-wise branch controller、M-aware momentum、data-weighted metric 和 late-stage regularization。

---

## 2. 当前 functional update 的数学诊断

### 2.1 当前 update 的基本形式

目前 KAN edge function 用 basis 表示：

$$
\phi(t)
=
\sum_{m=1}^{M} a_m B_m(t)
$$

普通 coefficient update 是：

$$
a_{t+1}
=
a_t
-
\eta \nabla_a L
$$

functional update 则使用函数空间 metric：

$$
a_{t+1}
=
a_t
-
\eta
(M+\rho I)^{-1}
\nabla_a L
$$

其中 Sobolev metric 可以写成：

$$
M
=
\int B B^\top dt
+
\alpha
\int B' B'^\top dt
+
\beta
\int B'' B''^\top dt
$$

这个更新的优点是它不再把 basis coefficient 当作普通欧氏参数，而是近似函数空间中的梯度下降。它自然会考虑 basis overlap、函数斜率和曲率结构。

但这个形式仍然有三个不足。

第一，它的 effective step size 由 $\eta$、$\rho$ 和 $M$ 的谱共同决定。如果 $\rho$ 太大或 $M$ 太强，KAN branch 会 under-active。如果 $\eta$ 太大或 branch boost 太强，则 $\phi'$ 和 Jacobian condition 可能变坏。

第二，Sobolev preconditioning 不等于显式 smoothness regularization。$M^{-1}$ 改变的是优化方向，不保证最终函数一定更平滑。高频任务上，Sobolev update 甚至可能更积极地拟合局部结构。

第三，当前 update 没有充分利用 actual descent feedback。已有实验中 sign agreement 经常接近 1，但这只能说明方向大体正确，不能说明步长合适。更重要的是 predicted descent 和 actual descent 的 ratio。

---

### 2.2 Functional update 的真正控制变量

对 residual DG-KAN 来说，一个 KAN block 的前向扰动为：

$$
\Delta h_k
=
\alpha_k \operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

其相对强度是：

$$
r_{branch,k}
=
\frac{\|\Delta h_k\|}{\|h_k\|+\epsilon}
$$

而 backward credit 的非 identity 修正大致受以下量影响：

$$
\alpha_k \Phi'_k(x_k)
$$

因此，一个更合理的控制目标不是 coefficient norm，而是下面几个 observable：

$$
r_{branch,k}
$$

$$
\phi'_{p95,k}
$$

$$
s_k
=
|\alpha_k|\phi'_{p95,k}
$$

$$
\kappa(J_k)
$$

$$
\text{credit amplification}_{k,p95}
$$

如果只增大 coefficient learning rate，可能短期提升 branch ratio，但同时推高 $\phi'$ 和 Jacobian condition。如果只加强 Sobolev damping，几何会很好，但 branch under-active。GFU 的目标就是把这些量纳入统一反馈。

---

## 3. GFU 的总体设计

GFU，即 Geometry-aware Functional Update，是对现有 functional update 的系统升级。它不是替代 AnalyticAdj，也不是替代 DG-KAN 架构，而是替代当前较静态的 `diag-to-Sobolev` coefficient update。

每一步训练中，GFU 做以下事情。首先使用 AnalyticAdj 或 FirstOrderDecompAdj 得到 block credit，然后计算 KAN coefficient gradient。接着根据当前 layer 的函数空间 metric，求出 functional descent direction。然后用 trust-region 限制该 direction 的函数空间步长，再用 branch target controller 调整每层 effective learning rate 和 branch scale。最后根据 online geometry trigger 决定是否切换 metric、增加 damping、或者执行 late-stage smoothness shrink。

完整形式可以写成：

$$
g_{a,k,t}
=
\nabla_{a_k} L_t
$$

$$
M_{k,t}
=
M_{grid,k}
+
\lambda_{data,t}M_{data,k,t}
+
\alpha_{t}R_{1,k}
+
\beta_{t}R_{2,k}
+
\rho_{k,t}I
$$

$$
p_{k,t}
=
-M_{k,t}^{-1}g_{a,k,t}
$$

然后做 M-norm trust clipping：

$$
\|p_{k,t}\|_{M_{k,t}}
=
\sqrt{p_{k,t}^\top M_{k,t}p_{k,t}}
$$

若：

$$
\|p_{k,t}\|_{M_{k,t}} > \tau_{k,t}
$$

则：

$$
p_{k,t}
\leftarrow
p_{k,t}
\frac{\tau_{k,t}}{\|p_{k,t}\|_{M_{k,t}}+\epsilon}
$$

最终更新：

$$
a_{k,t+1}
=
a_{k,t}
+
\eta_{k,t}p_{k,t}
$$

其中 $\eta_{k,t}$、$\rho_{k,t}$、$\tau_{k,t}$、metric mixing coefficient、branch scale 都由 online feedback 控制。

GFU 的核心不只是换一个矩阵，而是形成闭环：

$$
\text{credit}
\rightarrow
\text{functional direction}
\rightarrow
\text{actual descent}
\rightarrow
\text{branch / geometry audit}
\rightarrow
\text{next-step controller}
$$

---

## 4. Adaptive functional metric

### 4.1 基础 grid Sobolev metric

对每个 basis $B_m(t)$，预先计算 grid metric：

$$
M_0
=
\int B(t)B(t)^\top dt
$$

$$
R_1
=
\int B'(t)B'(t)^\top dt
$$

$$
R_2
=
\int B''(t)B''(t)^\top dt
$$

基础 Sobolev metric 为：

$$
M_{sob}
=
M_0
+
\alpha R_1
+
\beta R_2
+
\rho I
$$

这部分可以继续沿用当前实现。它稳定、便宜、可预计算，适合作为全局函数空间先验。

---

### 4.2 Data-weighted metric

当前 grid metric 对所有输入区间一视同仁，但实际训练中每条 edge 看到的是激活分布 $x_i \sim \mathcal D_k$。因此需要引入 data-weighted metric：

$$
M_{data,k,t}
=
\mathbb E_{x_i\sim\mathcal D_{k,t}}
\left[
B(x_i)B(x_i)^\top
\right]
$$

还可以加入 derivative data metric：

$$
R_{1,data,k,t}
=
\mathbb E
\left[
B'(x_i)B'(x_i)^\top
\right]
$$

$$
R_{2,data,k,t}
=
\mathbb E
\left[
B''(x_i)B''(x_i)^\top
\right]
$$

最终 metric 为：

$$
M_{k,t}
=
(1-\lambda_{data,t})M_{grid,k}
+
\lambda_{data,t}M_{data,k,t}
+
\alpha_t R_{1,k}
+
\beta_t R_{2,k}
+
\rho_{k,t}I
$$

其中 $M_{data,k,t}$ 用 EMA 更新：

$$
\widehat M_{data,k,t}
=
\mu \widehat M_{data,k,t-1}
+
(1-\mu)M_{data,k}^{batch}
$$

早期可以提高 $\lambda_{data,t}$，让更新更贴近实际训练分布；后期降低 $\lambda_{data,t}$ 或增加 grid Sobolev 权重，让函数在全域保持平滑和稳定。

这一步对泛化尤其重要，因为它避免两个极端：只用 grid metric 会忽略真实激活密度，只用 data metric 会在训练分布外缺乏约束。

---

### 4.3 Metric continuation

现有 `diag-to-Sobolev` 是 hard switch。GFU 建议改成 soft continuation：

$$
P_{k,t}
=
(1-\lambda_t)D_{k,t}^{-1}
+
\lambda_t M_{k,t}^{-1}
$$

其中：

$$
D_{k,t}=\operatorname{diag}(M_{k,t})
$$

functional direction 变成：

$$
p_{k,t}
=-P_{k,t}g_{a,k,t}
$$

早期 $\lambda_t\approx 0$，使用 diagonal update，加速 early descent 和 branch activation。后期 $\lambda_t\to 1$，使用 full Sobolev inverse，增强函数几何控制。

$\lambda_t$ 不应只按 epoch 增加，而应由状态决定：

$$
\lambda_t
=
\sigma
\left(
 c_1 u_{plateau,t}
+c_2 u_{phi,t}
+c_3 u_{jac,t}
+c_4 u_{branch,t}
\right)
$$

其中：

$$
u_{phi,t}
=
\frac{\phi'_{p95,t}}{\tau_{\phi}}
$$

$$
u_{jac,t}
=
\frac{\kappa(J_t)}{\tau_J}
$$

$$
u_{branch,t}
=
\frac{r_{branch,t}}{r_{target,t}}
$$

当模型已经充分激活 KAN branch，或者 geometry 指标接近阈值，就让 $\lambda_t$ 更接近 Sobolev full inverse。

---

## 5. Trust-region functional step

### 5.1 为什么需要 trust-region

现有实验里 functional update 的方向大多不是问题，问题是步长。小步长导致 branch under-active，大步长导致 $\phi'$ 和 Jacobian 失控。Trust-region 正好解决这个问题。

对每个 layer，先计算：

$$
p_{k,t}
=-M_{k,t}^{-1}g_{a,k,t}
$$

predicted descent 为：

$$
\Delta L_{pred,k,t}
=
\eta_{k,t} g_{a,k,t}^\top p_{k,t}
$$

由于 $p_{k,t}$ 是下降方向，理论上：

$$
\Delta L_{pred,k,t}<0
$$

实际下降为：

$$
\Delta L_{actual,t}
=
L(\theta_t + \Delta\theta_t)-L(\theta_t)
$$

定义 ratio：

$$
q_t
=
\frac{
\Delta L_{actual,t}
}{
\Delta L_{pred,t}+\epsilon
}
$$

理想情况下，$q_t$ 接近 1。如果 $q_t$ 很小，说明一阶模型过于乐观；如果 $q_t<0$，说明 predicted descent 与 actual loss 方向不一致。

---

### 5.2 Damping 和 step size 更新规则

GFU 中每个 layer 维护 $\eta_{k,t}$、$\rho_{k,t}$ 和 trust radius $\tau_{k,t}$。

如果 actual loss 上升：

$$
\Delta L_{actual,t}>0
$$

则执行：

$$
\eta_{k,t+1}=c_{\eta}^{down}\eta_{k,t}
$$

$$
\rho_{k,t+1}=c_{\rho}^{up}\rho_{k,t}
$$

$$
\tau_{k,t+1}=c_{\tau}^{down}\tau_{k,t}
$$

如果 $q_t$ 在合理区间，例如：

$$
0.5<q_t<1.5
$$

则保持或略微增加步长：

$$
\eta_{k,t+1}=c_{\eta}^{up}\eta_{k,t}
$$

$$
\rho_{k,t+1}=c_{\rho}^{down}\rho_{k,t}
$$

如果 $q_t$ 长期偏低，例如：

$$
0<q_t<0.25
$$

说明方向可能对，但步长太激进或 metric 太弱，应增加 damping：

$$
\rho_{k,t+1}=c_{\rho}^{up}\rho_{k,t}
$$

这套机制直接复用你已经记录的 descent audit，只是把 audit 从事后分析变成在线控制。

---

### 5.3 M-norm clipping

对 functional direction 做普通 Euclidean clipping 并不合适，因为 coefficient basis 不是正交的。应该使用 M-norm：

$$
\|p\|_{M}
=
\sqrt{p^\top M p}
$$

这对应函数空间中的更新幅度。限制：

$$
\|\eta p\|_{M}
\leq
\tau
$$

如果超过，则：

$$
p
\leftarrow
p
\frac{\tau}{\eta\|p\|_M+\epsilon}
$$

这可以防止 naive Adam moment 那种 branch explosion，也能让不同 layer 的 functional step 更可比。

---

## 6. Layer-wise branch target controller

### 6.1 为什么需要 layer-wise controller

Stage F 的 schedule 已经说明，全局 branch boost 可以改善 under-active，但不同任务、不同 layer 对 branch activity 的需求并不一样。KMNIST 上大 boost 会导致几何失控，而 Fashion 上较强 boost 反而可以同时赢 accuracy 和 AUC。

因此下一步不应只使用固定 branch boost，而应该对每层设定目标 branch ratio：

$$
r_{k}^{\star}(t)
$$

例如：

$$
r_k^\star(t)
=
\begin{cases}
0.6, & \text{early active phase} \\
0.45, & \text{middle phase} \\
0.35, & \text{late geometry phase}
\end{cases}
$$

对于更难任务或更深模型，可以提高 early target；对于 geometry 已经接近阈值的层，可以降低 target。

---

### 6.2 Feedback rule

定义当前 branch ratio：

$$
r_{k,t}
=
\frac{
\|\alpha_k \operatorname{KAN}_k(\operatorname{LN}(h_k))\|
}{
\|h_k\|+\epsilon
}
$$

每层维护 coefficient lr multiplier：

$$
m_{k,t}
$$

使用 multiplicative feedback：

$$
m_{k,t+1}
=
m_{k,t}
\exp
\left(
\gamma_r
\log
\frac{r_k^\star(t)+\epsilon}{r_{k,t}+\epsilon}
\right)
$$

如果当前 layer branch under-active，即 $r_{k,t}<r_k^\star$，则 $m_{k,t}$ 增大；如果 branch 过强，则减小。

然后加入 geometry safety factor：

$$
m_{k,t+1}
\leftarrow
m_{k,t+1}\cdot s_{geom,k,t}
$$

其中：

$$
s_{geom,k,t}
=
\min
\left(
1,
\frac{\tau_{\phi}}{\phi'_{p95,k,t}+\epsilon},
\frac{\tau_J}{\kappa(J_{k,t})+\epsilon},
\frac{\tau_s}{|\alpha_k|\phi'_{p95,k,t}+\epsilon}
\right)^{\gamma_g}
$$

最终 effective coefficient learning rate 为：

$$
\eta_{coeff,k,t}
=
\eta_{base,k}
\cdot m_{k,t}
$$

这比全局 schedule 更细，因为每层可以根据自己的 branch activity 和 geometry 独立调节。

---

### 6.3 Alpha 和 branch scale 的控制

除了调 coefficient lr，还可以调 branch scale：

$$
h_{k+1}
=
h_k
+
\alpha_k b_{k,t}\operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

其中 $b_{k,t}$ 是 schedule / controller 给出的 branch scale。建议把 $\alpha_k$ 作为可训练参数，但 $b_{k,t}$ 作为 optimizer state，由 controller 控制。

更新规则可以类似：

$$
b_{k,t+1}
=
b_{k,t}
\exp
\left(
\gamma_b
\log
\frac{r_k^\star(t)+\epsilon}{r_{k,t}+\epsilon}
\right)
$$

但如果 geometry 指标接近阈值，则 shrink：

$$
b_{k,t+1}
\leftarrow
\max(b_{min}, c_{shrink}b_{k,t})
$$

这与目前 geometry-aware trigger 中的 shrink 思路一致，只是从离散触发升级为连续控制。

---

## 7. M-aware momentum 与 Adam-like dynamics

### 7.1 为什么 naive Adam moment 会失败

已有实验显示，直接把 Adam-style moment 叠加到 Sobolev update 上，会导致 branch ratio 爆炸、$\phi'$ 极高、Jacobian condition 失控。这说明 Adam 的欧氏参数空间动量与 functional metric 不匹配。

在 coefficient space 中，两个方向的 Euclidean norm 可能相近，但它们对应的函数变化大小完全不同。因此 Adam moment 必须改成 M-aware。

---

### 7.2 Functional direction momentum

最简单安全的方式是先算 functional direction：

$$
p_{t}
=-M_t^{-1}g_t
$$

然后对 $p_t$ 做 momentum：

$$
v_t
=
\mu v_{t-1}
+
(1-\mu)p_t
$$

再做 M-norm clipping：

$$
v_t
\leftarrow
v_t
\min
\left(
1,
\frac{\tau}{\|v_t\|_{M_t}+\epsilon}
\right)
$$

更新：

$$
a_{t+1}
=
a_t+
\eta_t v_t
$$

这种方法的好处是 momentum 累积的是函数空间下降方向，而不是 raw coefficient gradient。

---

### 7.3 Whitened functional Adam

更完整的方式是在 whitened coordinate 中使用 Adam。若：

$$
M=Q\Lambda Q^\top
$$

定义：

$$
b=M^{1/2}a
$$

则：

$$
\|a\|_M^2=a^\top Ma=\|b\|^2
$$

在 $b$ 空间中，gradient 为：

$$
\nabla_b L
=
M^{-1/2}\nabla_a L
$$

然后执行 AdamW：

$$
b_{t+1}
=
\operatorname{AdamW}(b_t,\nabla_b L)
$$

再映射回：

$$
a_{t+1}=M^{-1/2}b_{t+1}
$$

这种方法更接近“函数空间 Adam”。它保留 Adam 的 adaptive moment 优势，但不再直接作用在不良 coefficient geometry 上。

实现上，basis count 通常较小，例如 8 或 16，所以对每个 layer / edge group 做 $M$ 的 eigendecomposition 是可行的。若 dense edge 太多，可以按 layer 共享 basis metric，而不是每条 edge 独立分解。

---

### 7.4 M-aware Adam with trust region

折中方案是：

$$
m_t
=
\beta_1m_{t-1}
+(1-\beta_1)g_t
$$

$$
v_t
=
\beta_2v_{t-1}
+(1-\beta_2)g_t^2
$$

$$
u_t
=
\frac{m_t}{\sqrt{v_t}+\epsilon}
$$

然后：

$$
p_t
= -M_t^{-1}u_t
$$

再做 M-norm clipping 和 trust-region ratio 检查。这比 naive Adam moment 安全，因为 Adam 只负责 gradient normalization，真正的函数空间 geometry 仍由 $M_t^{-1}$ 和 trust region 控制。

---

## 8. Explicit smoothness regularization 与 proximal shrink

### 8.1 Preconditioning 不等于 regularization

Sobolev preconditioner 改变的是下降方向：

$$
p=-M^{-1}g
$$

它并不等价于在 loss 里加入平滑正则。若想增强泛化能力，需要显式控制函数复杂度。

定义 edge smoothness penalty：

$$
\Omega(a)
=
\lambda_1 a^\top R_1 a
+
\lambda_2 a^\top R_2 a
$$

也就是：

$$
\Omega(\phi)
=
\lambda_1\int |\phi'(t)|^2dt
+
\lambda_2\int |\phi''(t)|^2dt
$$

可以直接把它加到训练 loss 中，也可以用 proximal step。

---

### 8.2 Proximal smoothness shrink

先执行 functional update 得到：

$$
\tilde a_{t+1}
=
a_t+
\eta p_t
$$

然后求：

$$
a_{t+1}
=
\arg\min_a
\frac{1}{2}\|a-\tilde a_{t+1}\|^2
+
\eta\lambda a^\top R a
$$

闭式解为：

$$
a_{t+1}
=
(I+2\eta\lambda R)^{-1}
\tilde a_{t+1}
$$

这里的 $R$ 可以是 $R_1$、$R_2$ 或二者组合。

建议只在 late phase 或 geometry trigger 后启用 proximal shrink。早期过强平滑会抑制 branch learning，尤其对 KMNIST 这类更复杂任务不利。

---

## 9. Spectral filtering

Sobolev metric 的曲率矩阵可以分解：

$$
R_2=Q\Lambda Q^\top
$$

其中大 $\Lambda_i$ 对应高曲率 / 高频函数方向。把 coefficient 写成：

$$
a=Qz
$$

则可以对不同频率使用不同 learning rate：

$$
z_{i,t+1}
=
z_{i,t}
-
\eta
w_i(t)
\nabla_{z_i}L
$$

其中：

$$
w_i(t)=
\frac{1}{1+\lambda_t\Lambda_i}
$$

早期 $\lambda_t$ 小，允许高频方向参与拟合；后期 $\lambda_t$ 大，逐步压制高频函数模式。

这比固定 $\beta$ 更灵活。它可以让模型先快速学任务，再逐渐把不必要的 high-frequency wiggle 收掉。对于泛化实验，尤其是 label noise 或 small-data setting，这个机制可能非常有价值。

---

## 10. 具体实现方案

### 10.1 新增 optimizer 类

建议新增一个独立模块，例如：

```text
experiments/functional_optim.py
```

核心类：

```text
GeometryFunctionalOptimizer
```

它负责管理 KAN coefficient 的 functional update。非 KAN 参数仍由 AdamW 管理。

该类需要维护以下 state：

```text
per layer:
  metric_mode
  rho
  eta_multiplier
  trust_radius
  branch_scale
  branch_target
  data_metric_ema
  momentum_buffer
  descent_ratio_ema
  last_phi_prime_p95
  last_jac_condition
  last_branch_ratio
```

对于每个 KAN layer，它需要提供：

```text
compute_metric(layer, batch_stats)
compute_direction(grad, metric)
apply_trust_region(direction, metric)
update_controller(audit_metrics)
step()
```

---

### 10.2 与现有训练代码的连接

当前 `stage_b_functional.py` 已经支持 functional update、rest optimizer、branch schedule 和 geometry-aware trigger。GFU 可以作为一个新的 update type：

```text
--update gfu
```

新增参数：

```text
--gfu-metric-mode grid,data,mixed
--gfu-data-metric-ema 0.95
--gfu-trust-radius-init 0.03
--gfu-trust-radius-min 0.005
--gfu-trust-radius-max 0.20
--gfu-rho-init 0.001
--gfu-rho-min 0.0001
--gfu-rho-max 0.1
--gfu-descent-ratio-low 0.25
--gfu-descent-ratio-good-low 0.5
--gfu-descent-ratio-good-high 1.5
--gfu-branch-target-early 0.65
--gfu-branch-target-mid 0.50
--gfu-branch-target-late 0.35
--gfu-controller-gamma 0.05
--gfu-use-m-momentum true
--gfu-momentum 0.9
--gfu-mnorm-clip true
--gfu-prox-smooth-late true
--gfu-prox-lambda 0.001
```

GFU 不应该一次性打开所有功能。实现时可以分阶段：

```text
GFU-lite:
  trust-region + branch controller + existing grid Sobolev metric

GFU-data:
  GFU-lite + data-weighted metric

GFU-momentum:
  GFU-data + M-aware momentum

GFU-prox:
  GFU-momentum + late proximal smoothness shrink
```

这样可以清楚知道每个组件是否真的有收益。

---

### 10.3 每个 batch 的训练流程

一次训练 step 可以写成如下过程。

第一步，前向：

$$
\hat y=f_\theta(x)
$$

计算 loss：

$$
L=\ell(\hat y,y)
$$

第二步，使用 AnalyticAdj / autograd 得到 KAN coefficient gradient：

$$
g_{a,k}=\nabla_{a_k}L
$$

第三步，收集当前 batch 的 basis statistics：

$$
B(x),\quad B'(x),\quad B''(x)
$$

更新 data metric EMA：

$$
\widehat M_{data,k,t}
=
\mu\widehat M_{data,k,t-1}
+(1-\mu)B^\top B
$$

第四步，构造 metric：

$$
M_{k,t}
=
M_{grid,k}
+
\lambda_{data,t}\widehat M_{data,k,t}
+
\alpha_tR_{1,k}
+
\beta_tR_{2,k}
+
\rho_{k,t}I
$$

第五步，计算 direction：

$$
p_{k,t}=-M_{k,t}^{-1}g_{a,k}
$$

第六步，M-norm clipping：

$$
p_{k,t}\leftarrow
\operatorname{clip}_{M}(p_{k,t},\tau_{k,t})
$$

第七步，应用 M-aware momentum，若启用：

$$
v_{k,t}=\mu v_{k,t-1}+(1-\mu)p_{k,t}
$$

第八步，更新 KAN coefficients：

$$
a_{k,t+1}=a_{k,t}+\eta_{k,t}v_{k,t}
$$

第九步，非 KAN 参数用 AdamW 更新。

第十步，周期性 audit：

$$
r_{branch,k},\quad \phi'_{p95,k},\quad \kappa(J_k),\quad q_t
$$

第十一步，更新 controller state：

$$
\eta_{k,t+1},\quad \rho_{k,t+1},\quad \tau_{k,t+1},\quad b_{k,t+1},\quad \lambda_{t+1}
$$

---

### 10.4 Logging keys

建议新增以下 W&B / CSV keys。

Functional metric：

```text
metric/layer_k/rho
metric/layer_k/cond
metric/layer_k/eig_min
metric/layer_k/eig_max
metric/layer_k/data_mix_lambda
metric/layer_k/update_m_norm
metric/layer_k/trust_radius
metric/layer_k/trust_clip_rate
```

Descent control：

```text
descent/predicted_delta
descent/actual_delta
descent/ratio
descent/ratio_ema
descent/bad_step_rate
descent/rho_increase_count
descent/trust_shrink_count
```

Branch controller：

```text
branch/layer_k/target_ratio
branch/layer_k/actual_ratio
branch/layer_k/controller_multiplier
branch/layer_k/branch_scale
branch/layer_k/effective_coeff_lr
branch/layer_k/effective_alpha_phi_p95
```

Geometry：

```text
geometry/layer_k/phi_prime_p95
geometry/layer_k/jacobian_condition
geometry/layer_k/credit_amplification_p95
geometry/global/max_effective_alpha_phi
geometry/global/max_jac_condition
```

Smoothness：

```text
smooth/layer_k/curvature_energy
smooth/layer_k/prox_shrink_norm
smooth/layer_k/high_freq_energy
```

Outcome：

```text
conv/val_loss_auc
conv/train_loss_auc
conv/steps_to_target_val_loss
task/test_acc
task/test_loss
generalization/train_test_gap_acc
generalization/train_test_gap_loss
```

---

## 11. 实验计划

### 11.1 GFU-0：实现正确性与 smoke test

GFU-0 的目标不是赢 AdamW，而是确认实现没有破坏现有 functional update。

数据集：

```text
Two Moons
MNIST small smoke
Fashion-MNIST smoke
```

模型：

```text
depth = 2
hidden_dim = 32
basis_count = 8
```

方法：

```text
AdamW
current diag_to_sobolev
GFU-lite
GFU-lite no trust region
GFU-lite no branch controller
```

成功标准：

$$
\text{NaN / Inf count}=0
$$

$$
\text{bad step rate}<\text{current functional update}
$$

$$
\text{test acc gap vs current functional}<1\%
$$

如果 GFU-lite 在 smoke 上不稳定，则不进入后续实验。

---

### 11.2 GFU-1：Fashion / KMNIST 主实验

这是最重要的一组实验。直接对标 Stage F5 / F6 的 best schedule。

数据集：

```text
Fashion-MNIST
KMNIST
```

模型：

```text
depth = 4
hidden_dim = 64
basis_count = 16
alpha_init = 1.5
train / val / test = 6000 / 1000 / 1000
seeds = 0,1,2,3,4
```

方法：

```text
AdamW
current static functional top
current geometry-aware best
GFU-lite
GFU-data
GFU-data + M-momentum
GFU-data + M-momentum + proximal smoothness
```

Fashion 的目标不是只赢一次，而是证明稳定性：

$$
\text{test acc} \geq \text{AdamW}
$$

$$
\text{val AUC improvement}>10\%
$$

$$
\phi'_{p95}\text{ reduction}>20\%
$$

$$
\text{Jacobian condition reduction}>30\%
$$

KMNIST 的目标是缩小目前的剩余 gap：

$$
\text{test acc gap vs AdamW}<0.5\%
$$

同时保持：

$$
\text{val AUC improvement}>8\%
$$

$$
\phi'_{p95}\text{ reduction}>25\%
$$

$$
\text{Jacobian condition reduction}>15\%
$$

---

### 11.3 GFU-2：trust-region ablation

目标：验证 trust-region 是否真正解决 step-size 问题。

固定：

```text
Fashion best config
KMNIST tuned config
```

比较：

```text
no trust region
M-norm clipping only
predicted/actual ratio only
M-norm clipping + ratio-based damping
```

主要指标：

```text
bad step rate
descent ratio median
descent ratio p10
branch / AdamW
phi_prime_p95
Jacobian condition
val AUC
test acc
```

预期：trust-region 不一定显著提升 final accuracy，但应该降低 bad steps、减少 geometry spike，并允许更大的 early coefficient lr。

---

### 11.4 GFU-3：M-aware momentum / Adam 实验

目标：找回 AdamW 的 optimizer dynamics，但不破坏 functional geometry。

比较：

```text
current Sobolev
naive Sobolev momentum
functional-direction momentum
whitened functional Adam
M-aware Adam + trust region
```

成功标准：

$$
\text{branch explosion 不出现}
$$

$$
\max_k \kappa(J_k) < 1.2 \times \kappa(J_k)_{AdamW}
$$

$$
\text{test acc gap vs AdamW}<0.5\%
$$

如果 naive momentum 仍然爆炸，而 M-aware momentum 不爆炸，就能形成一个很清楚的实验证据：momentum 需要放在函数空间里，而不是 coefficient Euclidean space 里。

---

### 11.5 GFU-4：泛化压力实验

如果 functional update 的价值是泛化和稳定性，那么小数据和噪声标签下应该更明显。

数据集：

```text
Fashion-MNIST
KMNIST
```

设置：

```text
train_size = 1000, 3000, 6000, 12000
label_noise = 0%, 5%, 10%
depth = 4, 8
hidden_dim = 64, 128
```

比较：

```text
AdamW
current geometry-aware
GFU best
```

重点看：

$$
\text{test acc}
$$

$$
\text{train-test gap}
$$

$$
\text{val loss AUC}
$$

$$
\phi'_{p95}
$$

$$
\kappa(J)
$$

如果 GFU 在 noisy label 或 small-data 下比 AdamW 更稳，就可以支撑“functional update enhances generalization”的 claim。

---

### 11.6 GFU-5：更大模型与 ConvStem-DGKAN

Fashion 和 KMNIST 是机制验证。下一步需要更接近真实视觉任务。

建议顺序：

```text
MNIST full
Fashion-MNIST full
KMNIST full
EMNIST Balanced small
CIFAR-10 small with ConvStem-DGKAN
```

CIFAR-10 small 的第一轮不要追求 SOTA，只看 functional update 是否仍然改善：

$$
\text{loss AUC}
$$

$$
\phi'_{p95}
$$

$$
\kappa(J)
$$

$$
\text{credit amplification}
$$

$$
\text{test acc gap vs AdamW}
$$

---

## 12. Failure table 设计

GFU 实验必须保留结构化 failure table，否则很容易只看到平均结果，看不到失败模式。

建议 failure type：

```text
branch_underactive
branch_overactive
geometry_spike
trust_region_too_conservative
trust_region_too_loose
descent_ratio_bad
momentum_explosion
data_metric_overfit
prox_too_strong
accuracy_gap_too_large
no_convergence_gain
no_generalization_gain
```

触发条件示例：

$$
\text{branch_underactive}:
\frac{r_{branch}}{r_{branch}^{AdamW}}<0.5
$$

$$
\text{branch_overactive}:
\frac{r_{branch}}{r_{branch}^{AdamW}}>1.2
$$

$$
\text{geometry_spike}:
\kappa(J)>1.5\kappa(J)_{AdamW}
$$

$$
\text{trust_region_too_conservative}:
\text{clip rate}>0.8 \text{ and branch underactive}
$$

$$
\text{descent_ratio_bad}:
q_{p10}<0
$$

$$
\text{prox_too_strong}:
\text{curvature reduced but train acc gap}>2\%
$$

这样每次失败都能被归类到 optimizer 的具体部件，而不是只说“GFU 不好”。

---

## 13. 预期结果与决策规则

### 13.1 最理想结果

最理想结果是 GFU 在 Fashion 和 KMNIST 上都达到：

$$
\text{test acc} \geq \text{AdamW} - 0.5\%
$$

$$
\text{val-loss AUC improvement}>10\%
$$

$$
\phi'_{p95}\text{ reduction}>25\%
$$

$$
\text{Jacobian condition reduction}>20\%
$$

如果这个结果成立，就可以形成非常强的结论：

$$
\boxed{
\text{GFU improves optimization geometry and convergence dynamics while matching AdamW-level accuracy.}
}
$$

如果在 small-data 或 noisy-label setting 下 GFU 的 test acc 明显优于 AdamW，则可以进一步支持 generalization claim。

---

### 13.2 中等成功

如果 GFU 不能全面超过 AdamW，但能稳定做到：

$$
\text{test acc gap}<1\%
$$

$$
\text{val-loss AUC improvement}>8\%
$$

$$
\phi'_{p95}\text{ reduction}>25\%
$$

$$
\text{Jacobian condition reduction}>15\%
$$

那也足以说明 functional update 是有效的 Pareto optimizer：它用很小的精度代价换来显著更好的函数几何和收敛轨迹。

---

### 13.3 失败但有信息量的结果

如果 GFU 仍然追不上 AdamW，需要看失败类型。

如果 failure table 主要是 `branch_underactive`，说明 controller 太保守，应提高 branch target 或 trust radius。

如果主要是 `geometry_spike`，说明 branch target 太高或 damping 太弱，应提高 $\rho$、降低 branch final scale 或提前 Sobolev phase。

如果主要是 `momentum_explosion`，说明 M-aware momentum 仍不够安全，应退回 functional-direction momentum 或加强 M-norm clipping。

如果主要是 `prox_too_strong`，说明 smoothness regularization 进入太早或强度太大。

如果主要是 `no_generalization_gain`，说明 functional update 的优势主要在 geometry / AUC，而不是泛化，需要调整 claim。

---

## 14. 与 Stage D / E / F 主线的关系

GFU 不改变 Stage D 的 credit 主线。AnalyticAdj 仍然是默认 credit source：

$$
\boxed{\text{default credit} = \text{AnalyticAdj}}
$$

FirstOrderDecompAdj 仍然是候选近似：

$$
\boxed{\text{approx credit} = \text{FirstOrderDecompAdj}}
$$

GFU 主要改变的是 KAN coefficient update：

$$
\boxed{\text{coefficient update} = \text{GFU}}
$$

因此 Stage D / E / F 可以重新组织为：

```text
Stage D:
  credit correctness and joint training
  AnalyticAdj vs FirstOrder vs Identity

Stage E:
  memory accounting
  Analytic primitive vs sufficient statistics vs checkpointing

Stage F:
  optimizer dynamics
  GFU vs AdamW vs current functional update
```

这样结构更清楚：credit transport、memory mechanism、optimizer dynamics 分开验证。

---

## 15. 为什么这个方向值得继续

从数学上看，KAN edge function 是一个函数对象，而不是普通权重。对它做欧氏 coefficient update，本质上忽略了 basis overlap、函数斜率、函数曲率和输入分布。Functional update 的优势，是把参数更新改写成：

$$
\Delta \phi
\approx
-\eta \nabla_{\mathcal H}L
$$

而不是：

$$
\Delta a
=
-\eta \nabla_a L
$$

这使得更新方向更接近函数空间中的自然方向，也更容易显式控制 smoothness、Jacobian 和 credit amplification。

从实验上看，你已经观察到 functional update 可以让 $\phi'$ 和 Jacobian condition 显著低于 AdamW，并且在 Fashion-MNIST 上通过 schedule 已经能同时赢 accuracy、AUC 和 geometry。KMNIST 上还差约 $1\%$，但 threshold sweep 已经显示简单调 trigger 进入平台期。因此继续优化的合理方向不是再扫一个 schedule 超参，而是升级 optimizer 机制。

GFU 正是把已有经验合成闭环 optimizer：

$$
\text{under-active}
\Rightarrow
\text{increase branch / functional step}
$$

$$
\text{geometry spike}
\Rightarrow
\text{increase damping / shrink branch / switch Sobolev}
$$

$$
\text{bad descent ratio}
\Rightarrow
\text{trust-region shrink}
$$

$$
\text{late overfitting}
\Rightarrow
\text{proximal smoothness / spectral filtering}
$$

这比固定 `coeff_lr` 或固定 `branch_boost` 更符合 DG-KAN 的数学结构。

---

## 16. 最终建议

我建议下一阶段把研究问题明确写成：

> Can geometry-aware functional optimization improve DG-KAN convergence and generalization while preserving stable credit geometry?

对应中文表述：

> 几何感知的函数空间优化，能否在保持 DG-KAN 稳定 credit geometry 的同时，提升收敛速度和泛化能力？

具体路线是：

$$
\boxed{
\text{AnalyticAdj-DGKAN}
+
\text{GFU coefficient update}
+
\text{AdamW rest optimizer}
+
\text{online branch / geometry controller}
}
$$

第一阶段不要一次性追求所有组件，而是按下面顺序推进：

```text
GFU-lite:
  trust-region + branch controller

GFU-data:
  add data-weighted metric

GFU-momentum:
  add M-aware momentum

GFU-prox:
  add late proximal smoothness
```

如果 GFU-lite 已经能把 KMNIST gap 从约 $1.06\%$ 压到 $0.5\%$ 内，同时保留 AUC 和 geometry 优势，那么这就是非常重要的结果。如果 GFU-data 或 GFU-momentum 进一步提升，则可以形成一个完整 optimizer story。

最终你可以形成的主张不是“functional update 总是比 AdamW 精度更高”，而是更有科学价值的：

$$
\boxed{
\text{Functional-space update, when coupled with geometry-aware control,}
\text{can achieve AdamW-level accuracy with better convergence geometry,}
\text{lower derivative growth, and more stable credit transport.}
}
$$

这和 DG-LCA 的总目标完全一致：不是简单替代 BP，而是把 credit transport 和 local update 都变成可解释、可审计、可控的几何过程。

---

## 17. 一句话总结

GFU 的核心不是再发明一个 optimizer 名字，而是把你前面所有实验已经证明的事实合起来：

$$
\boxed{
\text{用 AnalyticAdj 保证 credit 正确，}
\text{用 functional metric 保证更新几何，}
\text{用 branch controller 保证表达力，}
\text{用 trust-region 和 geometry trigger 保证稳定性。}
}
$$

如果这条路线跑通，DG-KAN 的 functional update 就不只是一个“比 coefficient SGD 更平滑”的技巧，而会成为 DG-LCA 里真正支撑泛化、收敛和低显存训练的核心机制。
