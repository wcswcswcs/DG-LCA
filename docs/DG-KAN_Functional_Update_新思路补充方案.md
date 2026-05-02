# DG-KAN Functional Update 新思路补充方案：从函数空间优化到泛化控制

> 文件名：`DG-KAN_Functional_Update_新思路补充方案.md`  
> 对象：DG-KAN / DG-LCA functional update 后续实验  
> 定位：在已有 GFU 方案之外，补充一组更偏机制创新的新方向  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`  
> 核心目标：把当前 DG-KAN functional update 从“几何预条件优化器”进一步推进为“函数空间泛化控制器”，在保持或提升精度的同时，获得更低 $\phi'$、更低 Jacobian condition、更好的 loss AUC、更稳定的 branch utilization 和更强小数据泛化。

---

## 0. 总体定位

前一份 GFU 方案的核心是把当前已经有效的 functional update 做成一个更成熟的 optimizer：引入 trust-region、branch controller、M-aware momentum、data-weighted metric 和 late-stage geometry control。那条路线更像是对现有 Stage F 结果的系统工程化增强。

这份补充方案关注另一类问题：是否可以围绕 functional update 设计新的训练机制，使 DG-KAN 不只是“用 Sobolev metric 更新 coefficient”，而是利用 KAN edge function 的函数结构来增强泛化能力、收敛速度和稳定性。

因此本方案的主题不是简单调参，而是：

$$
\boxed{
\text{Function-space generalization control for DG-KAN}
}
$$

它的核心判断是：DG-KAN 的优势不应该只来自 residual geometry 或 analytic adjoint，而应该进一步来自“函数值、函数导数、函数频率、edge-level credit 统计”这些普通 MLP optimizer 看不到的信息。

在这个视角下，KAN edge function 不是普通参数向量，而是一个可被正则、扰动、平均、投影、重网格化、频谱过滤和 SNR 控制的函数对象：

$$
\phi_{ji}(t)=\sum_m a_{ji,m}B_m(t)
$$

functional update 的目标也不只是降低训练 loss，而是找到一个在函数空间中稳定、平滑、低 sharpness、低噪声并且任务有效的更新轨迹。

---

## 1. 之前实验给出的关键 insight

当前实验已经给出了一条相当清晰的证据链。这个补充方案建立在这些 insight 之上。

首先，residual DG-KAN 的几何稳定性已经被反复验证。纯 KAN block 的 Jacobian condition 容易达到极高量级，而 residual DG-KAN 通过

$$
h_{out}=h+\alpha\operatorname{KAN}(\operatorname{LN}(h))
$$

显著降低局部 Jacobian condition，使 credit amplification 和 noise gain 更可控。这说明信息保持 interface 是有效的，DG-KAN 不是单纯换一个激活函数，而是在构造一个更适合 local credit transport 的 forward interface。

其次，functional update 在回归任务和分类任务上都显示了价值。回归任务中，diag update 通常有更好的早期 loss AUC，Sobolev update 通常有更好的最终 test loss。分类任务中，直接把所有参数都改成 functional / SGD-like update 会欠拟合，但如果只对 KAN coefficients 使用 functional update，非 KAN 参数使用 AdamW，就能得到合理结果。这说明 functional update 最适合先作用在 KAN edge function 这一类 function-valued primitive 上，而不是粗暴替代所有参数优化。

第三，MNIST / Fashion-MNIST / KMNIST 的补充实验说明，functional update 的主要瓶颈不是 basis coverage，也不是 analytic credit 错误，而是 KAN branch under-active。小步长 functional update 会让 KAN branch 对 logits 的贡献太小，关闭 KAN branch 后精度几乎不掉。增大 coefficient learning rate 或 branch scale 可以恢复精度，但过强 branch 又会推高 $\phi'_{p95}$ 和 Jacobian condition。因此 functional update 必须和 branch activity controller 配套，而不能只靠固定 lr。

第四，AnalyticAdj 和 FirstOrderDecompAdj 已经证明 credit correctness 不是当前主瓶颈。AnalyticAdj 与 FullBP 在训练结果上几乎等价，FirstOrderDecompAdj 也能在 joint training 中接近 AnalyticAdj。这说明后续重点不应该继续纠缠 hidden-only learned router，而应该把精力转向 functional optimizer dynamics、泛化控制和 memory-efficient implementation。

第五，Stage F 的 schedule 实验给出最重要的方向：branch-aware、loss-aware 和 geometry-aware schedule 可以显著改善 static functional update。Fashion-MNIST 上已经出现同时优于 AdamW accuracy、loss AUC 和几何指标的配置；KMNIST 上也把 static functional 的 gap 从约 $1.46\%$ 缩小到约 $1.06\%$，并保持较好的 AUC improvement 与 geometry reduction。

这些结果共同说明，functional update 不是一个单独的 update rule，而应当被设计成一个闭环系统：

$$
\boxed{
\text{credit correctness}
+
\text{function-space metric}
+
\text{branch controller}
+
\text{泛化正则}
+
\text{online geometry feedback}
}
$$

本方案接下来提出的新机制，就是围绕这个闭环系统展开。

---

## 2. 新方向一：Functional-SAM，在函数空间中做 sharpness-aware training

### 2.1 动机

普通 SAM 的目标是寻找参数空间中 flat 的解。它通常写成：

$$
\min_\theta \max_{\|\epsilon\|\leq \rho} L(\theta+\epsilon)
$$

但 DG-KAN 的核心观点是 coefficient space 不是最自然的几何。对于 KAN edge function，真正应该控制的是函数扰动，而不是 raw coefficient 扰动。因此更合理的 sharpness-aware 目标是：

$$
\min_a \max_{\|\delta a\|_M\leq \rho} L(a+\delta a)
$$

其中：

$$
\|\delta a\|_M^2=\delta a^\top M\delta a
$$

$M$ 可以是 Sobolev metric、data-weighted Sobolev metric，或者二者的混合。这个目标的含义是：模型不仅要在当前 edge function 上 loss 低，还要在函数空间邻域内保持 loss 低。

这比参数空间 SAM 更符合 DG-KAN 的泛化目标。因为 KAN 的过拟合通常表现为 edge function 局部变尖、导数变大、曲率变高或高频模式过多，而这些正是 Sobolev metric 能刻画的对象。

---

### 2.2 数学形式

设当前 KAN coefficient gradient 为：

$$
g=\nabla_a L(a)
$$

函数空间邻域约束为：

$$
\delta a^\top M\delta a\leq \rho^2
$$

一阶近似下，最坏扰动方向是：

$$
\delta a_{adv}
=
\rho
\frac{M^{-1}g}{\sqrt{g^\top M^{-1}g+\epsilon}}
$$

然后在扰动后的函数上重新计算 loss 和梯度：

$$
g_{sam}=\nabla_a L(a+\delta a_{adv})
$$

最后做 functional update：

$$
a_{t+1}=a_t-\eta M^{-1}g_{sam}
$$

这就是 Functional-SAM 的基本形式。

如果加入 trust-region，可以变成：

$$
p_t=-M^{-1}g_{sam}
$$

并限制：

$$
p_t^\top M p_t\leq \tau^2
$$

如果方向过大，则缩放：

$$
p_t\leftarrow p_t\frac{\tau}{\sqrt{p_t^\top Mp_t+\epsilon}}
$$

---

### 2.3 实现方式

Functional-SAM 不需要每一步都做，因为它会增加一次额外 forward / backward。推荐先做 sparse SAM：

```text
普通 step:
  geometry-aware functional update

每 sam_interval step:
  1. 计算当前 KAN coefficient gradient g
  2. 用 M^{-1}g 构造 function-space adversarial perturbation
  3. 临时 perturb KAN coefficients
  4. 重新 forward/backward 得到 g_sam
  5. 恢复原 coefficient
  6. 用 M^{-1}g_sam 做更新
```

推荐初始配置：

```text
sam_interval in {4, 8}
rho_func in {0.01, 0.03, 0.05, 0.1}
metric in {sobolev, data_weighted_sobolev}
apply_to = KAN coefficients only
rest optimizer = AdamW
```

为了避免 SAM 早期抑制 branch 激活，可以只在中后期打开：

```text
sam_start_frac in {0.25, 0.5}
```

或者只在 geometry trigger 后打开：

```text
if phi_prime_p95 > phi_threshold or jacobian_condition > jac_threshold:
    enable Functional-SAM
```

---

### 2.4 预期效果

Functional-SAM 主要服务于泛化，而不是单纯训练 loss。它应该优先在以下场景中显示价值：

1. Fashion-MNIST / KMNIST 的 20 epoch setting；
2. 小训练集；
3. label noise；
4. depth 更深时；
5. branch boost 较强、容易产生高 Jacobian 的 setting。

如果有效，应看到：

```text
test acc 提升或保持
val loss AUC 不显著变差
phi_prime_p95 下降
Jacobian condition 下降
label-noise setting 下泛化 gap 更小
```

最关键的判定不是 train loss 是否更快下降，而是：

$$
\text{test accuracy gain}
+
\text{geometry reduction}
+
\text{label-noise robustness}
$$

---

## 3. 新方向二：SNR-adaptive edge update，按 edge-level credit 信噪比调节更新

### 3.1 动机

KAN 的每条 edge 都有自己的函数：

$$
\phi_{ji}(t)
$$

对应的 coefficient gradient 可以写成 batch 统计：

$$
s_{ji,m}
=
\frac{1}{B}\sum_{n=1}^{B}g_j^{(n)}B_m(x_i^{(n)})
$$

这个统计本质上是 edge-level credit assignment。不同 edge 的 credit 质量可能完全不同：有些 edge 的统计方向在 batch 之间很稳定，说明它确实携带任务相关信息；有些 edge 的统计高度噪声化，说明大步更新可能只是拟合偶然 batch noise。

当前 functional update 对所有 edge 使用类似的 lr / metric / damping，这会浪费一个重要信息：

$$
\boxed{
\text{不同 edge 的 credit SNR 不同，应该用不同的 functional update strength。}
}
$$

---

### 3.2 SNR 定义

对每条 edge 的 coefficient gradient 向量：

$$
s_{ji}\in\mathbb R^M
$$

维护 EMA：

$$
\mu_{ji,t}=\lambda\mu_{ji,t-1}+(1-\lambda)s_{ji,t}
$$

$$
q_{ji,t}=\lambda q_{ji,t-1}+(1-\lambda)s_{ji,t}^2
$$

方差估计为：

$$
v_{ji,t}=q_{ji,t}-\mu_{ji,t}^2
$$

定义 edge-level SNR：

$$
\operatorname{SNR}_{ji}
=
\frac{\|\mu_{ji,t}\|_2}{\sqrt{\operatorname{mean}(v_{ji,t})}+\epsilon}
$$

也可以用 function-space norm 定义：

$$
\operatorname{SNR}_{ji}^{M}
=
\frac{\sqrt{\mu_{ji,t}^\top M^{-1}\mu_{ji,t}}}{\sqrt{\operatorname{tr}(M^{-1}\Sigma_{ji,t})}+\epsilon}
$$

第一版建议先用简单欧氏 SNR，避免实现复杂。

---

### 3.3 SNR-adaptive update rule

根据 edge SNR 调节 learning rate：

$$
\eta_{ji}
=
\eta_0
\cdot
\operatorname{clip}
\left(
\frac{\operatorname{SNR}_{ji}}{\tau_{snr}},
\eta_{min\_mult},
\eta_{max\_mult}
\right)
$$

低 SNR edge 使用更强 damping：

$$
\rho_{ji}
=
\rho_0
\cdot
\operatorname{clip}
\left(
\frac{\tau_{snr}}{\operatorname{SNR}_{ji}+\epsilon},
\rho_{min\_mult},
\rho_{max\_mult}
\right)
$$

也可以提高低 SNR edge 的 curvature penalty：

$$
\beta_{ji}
=
\beta_0
\cdot
\operatorname{clip}
\left(
\frac{\tau_{snr}}{\operatorname{SNR}_{ji}+\epsilon},
\beta_{min\_mult},
\beta_{max\_mult}
\right)
$$

最终 update 为：

$$
\Delta a_{ji}
=
-\eta_{ji}
\left(M_{ji}+\rho_{ji}I\right)^{-1}
g_{ji}
$$

低 SNR edge 更新更慢、更平滑；高 SNR edge 更新更快、更积极。

---

### 3.4 和 branch controller 的关系

现有 branch controller 主要看：

$$
r_{branch,k}
=
\frac{\|\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))\|}{\|h_k\|+\epsilon}
$$

但它不知道 branch 中哪些 edge 是有用的，哪些 edge 是噪声。SNR-adaptive update 可以补上这个维度。

理想行为是：

```text
branch ratio 上升
no-KAN acc drop 上升
low-SNR edge fraction 下降
phi_prime_p95 不爆
Jacobian condition 不爆
```

如果 branch ratio 上升但 no-KAN drop 不上升，说明 branch 只是变大但没有真正参与任务；如果 SNR-aware update 有效，应看到 no-KAN drop 更接近 AdamW 或超过 static functional。

---

### 3.5 Logging keys

建议新增：

```text
snr/layer_{k}/edge_snr_mean
snr/layer_{k}/edge_snr_p10
snr/layer_{k}/edge_snr_p50
snr/layer_{k}/edge_snr_p90
snr/layer_{k}/low_snr_edge_fraction
snr/layer_{k}/high_snr_edge_fraction
update/layer_{k}/snr_lr_mult_mean
update/layer_{k}/snr_lr_mult_p90
metric/layer_{k}/snr_rho_mult_mean
metric/layer_{k}/snr_beta_mult_mean
branch/layer_{k}/snr_weighted_branch_ratio
```

---

## 4. 新方向三：Block-level Functional Target Solve，直接解局部函数目标

### 4.1 动机

目前多数 functional update 仍然是 gradient step：

$$
a_{t+1}=a_t-\eta M^{-1}\nabla_a L
$$

它比普通 coefficient SGD 更好，但仍然只是小步下降。DG-LCA 总方案中还有一个更强形式：给 block 一个局部目标，直接解函数空间投影问题。

对 block：

$$
h_{out}=h+\alpha\operatorname{KAN}(z),\quad z=\operatorname{LN}(h)
$$

如果输出端 credit 是 $g_{out}$，那么一阶局部下降希望 block output 发生：

$$
\Delta h_{out}^{(n)}\approx -\eta g_{out}^{(n)}
$$

由于 residual identity 不能直接改，主要让 branch output 产生变化：

$$
\alpha\Delta b^{(n)}\approx -\eta g_{out}^{(n)}
$$

即：

$$
\Delta b^{(n)}\approx -\frac{\eta}{\alpha}g_{out}^{(n)}
$$

这可以转成 KAN edge function 的局部 regression 问题。

---

### 4.2 局部函数投影公式

KAN branch 输出为：

$$
b_j^{(n)}=\sum_i \phi_{ji}(z_i^{(n)})
$$

边函数增量为：

$$
\Delta\phi_{ji}(t)=\sum_m \Delta a_{ji,m}B_m(t)
$$

则 output dimension $j$ 的目标是：

$$
\sum_i\sum_m \Delta a_{ji,m}B_m(z_i^{(n)})
\approx
-\frac{\eta}{\alpha}g_j^{(n)}
$$

令 $X$ 是由 $B_m(z_i^{(n)})$ 组成的设计矩阵，令：

$$
y_j=-\frac{\eta}{\alpha}g_j
$$

则 Sobolev ridge 目标为：

$$
\Delta a_j^\star
=
\arg\min_{\Delta a_j}
\|X\Delta a_j-y_j\|^2
+
\lambda\Delta a_j^\top M\Delta a_j
$$

闭式解：

$$
\Delta a_j^\star
=
(X^\top X+\lambda M)^{-1}X^\top y_j
$$

这个 update 比 $M^{-1}g$ 更强，因为它利用了 empirical feature covariance：

$$
X^\top X
$$

也就是说，它不仅知道函数空间 metric，还知道当前 batch 中 basis activation 的相关性。

---

### 4.3 与普通 functional gradient 的关系

普通 gradient 是：

$$
g_a=X^\top (b-y)
$$

functional update 是：

$$
\Delta a=-\eta M^{-1}g_a
$$

Functional Target Solve 则是：

$$
\Delta a=(X^\top X+\lambda M)^{-1}X^\top y
$$

它更接近局部 Gauss-Newton / kernel ridge update。对于早期 branch under-active 的问题，这可能比单纯增大 coefficient lr 更有效，因为它直接求“branch output 应该如何变化”。

---

### 4.4 实现策略

由于 dense KAN 的 $X$ 很大，第一版不建议全量求解。可以逐步做：

```text
版本 1：只对最后一个 KAN block 做 target solve
版本 2：per output dimension solve
版本 3：grouped KAN solve
版本 4：low-rank approximate solve
版本 5：target solve + functional gradient 混合
```

混合形式：

$$
\Delta a
=
(1-\gamma)\Delta a_{grad}
+
\gamma\Delta a_{target}
$$

早期 $\gamma$ 大，直接激活 branch；后期 $\gamma$ 小，回到稳定的 Sobolev functional update。

推荐初始配置：

```text
apply_layers = last_1 or last_2
gamma_start = 0.5
gamma_final = 0.0
target_solve_frac = 0.25
ridge_lambda in {1e-3, 1e-2, 1e-1}
```

---

### 4.5 预期作用

这个方向主要解决收敛速度和 branch under-active。

如果有效，应看到：

```text
branch / AdamW 上升
no-KAN acc drop 上升
val-loss AUC 改善
test acc gap 缩小
phi_prime_p95 不显著上升
Jacobian condition 不显著上升
```

它尤其适合 KMNIST，因为 KMNIST 当前主要不是 geometry threshold 问题，而是 functional branch 的任务参与度仍不足。

---

## 5. 新方向四：Credit Curriculum，从 IdentityAdj 平滑过渡到 AnalyticAdj

### 5.1 动机

实验已经显示 IdentityAdj 早期非常稳定，但在更难任务上不够；AnalyticAdj 更正确，但 non-identity correction 会增加优化复杂度。既然两者各有优势，可以做 credit curriculum：

$$
g_h^{(\lambda)}=g_{out}+\lambda_t\Delta g_{analytic}
$$

其中：

$$
\Delta g_{analytic}=g_h^{analytic}-g_{out}
$$

早期使用 identity-dominant credit，稳定地训练 head、stem 和主 residual stream；中后期逐步打开 analytic correction，让 KAN branch 获得更真实的 non-identity credit。

---

### 5.2 基本 schedule

最简单版本：

$$
\lambda_t=\min\left(1,\frac{t}{T_{curr}}\right)
$$

也可以用 cosine：

$$
\lambda_t=\frac{1}{2}\left(1-\cos\frac{\pi t}{T_{curr}}\right)
$$

更合理的是 geometry-aware credit curriculum：

$$
\lambda_{t+1}
=
\operatorname{clip}
\left(
\lambda_t
+\gamma_1\mathbf 1[r_{branch}<r^\star]
-\gamma_2\mathbf 1[\kappa(J)>\tau_J]
-\gamma_3\mathbf 1[\phi'_{p95}>\tau_\phi],
0,1
\right)
$$

也可以根据 correction ratio 控制：

$$
r_\Delta=
\frac{\|\Delta g\|}{\|g_{out}\|+\epsilon}
$$

如果 $r_\Delta$ 很大且 amplification 高，说明 correction 可能会扰动训练，降低 $\lambda$；如果 branch under-active，增加 $\lambda$。

---

### 5.3 和 FirstOrderDecompAdj 的关系

Credit curriculum 不一定只用于 AnalyticAdj，也可以用于 FirstOrderDecompAdj：

$$
g_h^{(\lambda)}=g_{out}+\lambda_t\Delta g_{first}
$$

其中：

$$
\Delta g_{first}=\sum_{s\in\text{span}}\Delta g_s
$$

这样可以在低成本 FirstOrder correction 和 identity credit 之间平滑过渡。

推荐比较：

```text
IdentityAdj
AnalyticAdj
FirstOrderDecompAdj
Identity -> Analytic curriculum
Identity -> FirstOrder curriculum
geometry-aware credit curriculum
```

---

### 5.4 预期作用

Credit curriculum 的主要目标是减少 hard task 上的 early optimization friction。

成功信号：

```text
KMNIST accuracy gap 缩小
val AUC 改善
branch under-active 缓解
Jacobian condition 不比 AnalyticAdj 更差
IdentityAdj 的稳定性保留，AnalyticAdj 的最终能力接近
```

---

## 6. 新方向五：Function-space EMA / SWA，平均边函数而不是只平均参数

### 6.1 动机

普通 EMA / SWA 在参数空间中做平均：

$$
\theta_{ema}=\tau\theta_{ema}+(1-\tau)\theta
$$

对于 KAN，更自然的是平均 edge function：

$$
\phi_{ema}(t)=\tau\phi_{ema}(t)+(1-\tau)\phi(t)
$$

由于 basis 固定，平均 coefficient 等价于平均函数：

$$
a_{ema}=\tau a_{ema}+(1-\tau)a
$$

但可以进一步做 Sobolev-smoothed EMA，让 slow function 不只是平均，而是保持平滑。

---

### 6.2 Sobolev-smoothed EMA

设 fast coefficient 为 $a_t$，slow coefficient 为 $a_t^{slow}$。可以定义：

$$
a_{t+1}^{slow}
=
\arg\min_a
\|a-a_t\|_{D}^2
+\lambda a^\top R a
+\mu\|a-a_t^{slow}\|_M^2
$$

其中 $D$ 可以是 identity 或 data-weighted basis Gram，$R$ 是 curvature penalty，$M$ 是 Sobolev metric。

闭式解为：

$$
a_{t+1}^{slow}
=
(D+\lambda R+\mu M)^{-1}
(Da_t+\mu M a_t^{slow})
$$

第一版可以先不用这么复杂，只做 coefficient EMA：

```text
KAN coefficients: EMA
non-KAN params: normal EMA or no EMA
```

---

### 6.3 Fast / slow function 训练范式

可以维护两个模型：

```text
fast model:
  用于训练，允许 branch 更 active

slow function model:
  用于 eval，edge functions 更平滑
```

这个机制很适合 functional update 的矛盾：训练时需要 branch active 才能提高精度，但最终泛化时希望函数形状更平滑。

建议记录：

```text
ema/test_acc
ema/val_loss
ema/phi_prime_p95
ema/jacobian_condition
ema/raw_minus_ema_acc
ema/raw_minus_ema_jcond
```

如果 EMA 模型 test acc 更高且 $\phi'$ / Jacobian 更低，这会成为 functional update 泛化能力的低成本增强。

---

## 7. 新方向六：Spectral Dropout / Spectral Shrink，在 Sobolev 特征空间中控制高频模式

### 7.1 动机

KAN edge function 的 basis coefficient 可以在 Sobolev penalty matrix 的特征空间中展开。设：

$$
R=Q\Lambda Q^\top
$$

其中 $R$ 可以是：

$$
R=\int B'B'^\top+\beta\int B''B''^\top
$$

令：

$$
a=Qz
$$

大 $\Lambda_i$ 对应高频、高导数或高曲率模式。泛化差往往不是因为所有函数成分都不好，而是高频模式过度拟合。

---

### 7.2 Spectral dropout

训练时对 $z_i$ 做频率依赖 dropout：

$$
z_i'=m_i z_i
$$

其中：

$$
m_i\sim \operatorname{Bernoulli}(1-p_i)
$$

dropout probability 设为：

$$
p_i=p_{max}\frac{\Lambda_i}{\Lambda_i+\tau}
$$

大 $\Lambda_i$ 的高频模式更容易被 drop，低频模式更稳定保留。

---

### 7.3 Spectral shrink

也可以做 deterministic shrink：

$$
z_i\leftarrow \frac{1}{1+\lambda\Lambda_i}z_i
$$

或者作为 late-stage proximal step：

$$
a\leftarrow (I+\lambda R)^{-1}a
$$

Spectral dropout 更像训练正则，spectral shrink 更像后期平滑。

---

### 7.4 推荐实验

初始配置：

```text
spectral_dropout_pmax in {0.05, 0.1, 0.2}
spectral_tau in {median_lambda, p75_lambda}
apply_start_frac in {0.25, 0.5}
apply_to = KAN coefficients only
```

成功信号：

```text
phi_prime_p95 下降
curvature energy 下降
Jacobian condition 下降
test acc 不掉或提升
label-noise 泛化提升
```

---

## 8. 新方向七：Edge-function Mixup，在一维 edge activation domain 中做 credit mixup

### 8.1 动机

KAN edge update 的训练样本不是整张图像，而是 edge-level scalar pair：

$$
(t_n,g_n)=(z_i^{(n)},g_j^{(n)})
$$

functional update 使用这些点来更新：

$$
\phi_{ji}(t)
$$

这意味着可以在 edge activation domain 中做 mixup，而不改变原始图像输入或 label。

---

### 8.2 Edge mixup 形式

对 batch 内两个样本 $a,b$，采样：

$$
\lambda\sim \operatorname{Beta}(\alpha_{mix},\alpha_{mix})
$$

构造：

$$
\tilde t=\lambda t_a+(1-\lambda)t_b
$$

$$
\tilde g=\lambda g_a+(1-\lambda)g_b
$$

然后把 $(\tilde t,\tilde g)$ 也加入 edge sufficient statistics：

$$
\tilde s_{ji,m}=\tilde g_j B_m(\tilde t_i)
$$

最终 gradient 使用：

$$
s_{total}=(1-\gamma_{mix})s_{orig}+\gamma_{mix}s_{mix}
$$

这鼓励 edge function 在 activation domain 上具有局部线性和插值平滑性。

---

### 8.3 实现注意

Edge mixup 不需要改变 forward loss，只影响 KAN coefficient update 的 sufficient statistics：

```text
1. 正常 forward / loss / credit
2. 收集 z_i 和 g_j
3. 在 batch 内 sample pair
4. 构造 mixed z 和 mixed credit
5. 计算 mixed coefficient stats
6. 与 original stats 混合
7. 做 functional update
```

推荐初始配置：

```text
edge_mixup_alpha in {0.2, 0.5, 1.0}
edge_mixup_ratio in {0.25, 0.5}
start_frac in {0.0, 0.25}
```

Edge mixup 可能会降低训练速度，因此更适合配合 geometry-aware schedule 或 late-stage regularization。

---

## 9. 新方向八：Adaptive Grid / Basis Transport，根据激活分布移动 RBF centers

### 9.1 动机

当前 RBF centers 多数是固定 grid。虽然已有实验中 active basis fraction 和 out-of-grid fraction 通常不是主因，但随着任务更复杂、depth 更深、进入 CIFAR 或 ConvStem-DGKAN，固定 grid 可能会浪费 basis capacity。

更自然的做法是根据 KAN input：

$$
z=\operatorname{LN}(h)
$$

的实际分布调整 centers。

---

### 9.2 Activation quantile grid

每隔若干 epoch 收集 $z$ 的分布，估计分位数：

$$
q_1,q_2,\dots,q_M
$$

把 RBF centers 更新为：

$$
c_m^{new}=q_m
$$

但是直接移动 centers 会改变函数，因此需要 function-preserving projection。

旧函数为：

$$
\phi_{old}(t)=\sum_m a_m^{old}B_m^{old}(t)
$$

新 basis 为：

$$
\phi_{new}(t)=\sum_m a_m^{new}B_m^{new}(t)
$$

求：

$$
a^{new}
=
\arg\min_a
\sum_{t\in\mathcal G}
\left|
\sum_m a_mB_m^{new}(t)-\phi_{old}(t)
\right|^2
+\lambda a^\top M_{new}a
$$

这样可以在基本保持当前函数的同时，把 basis 移到更有用的 activation 区域。

---

### 9.3 预期作用

Adaptive grid 主要作用在更复杂任务或更大模型：

```text
提高 active basis fraction
降低 basis redundancy
降低 out-of-grid risk
改善 metric conditioning
提升 hard dataset accuracy
```

初始不建议频繁更新 grid。可以在训练早期做一次或两次：

```text
grid_update_epoch in {1, 3}
projection_grid_size = 512
center_momentum = 0.5
```

中心点不要完全替换，而是 EMA：

$$
c^{new}\leftarrow \tau c^{old}+(1-\tau)q
$$

避免训练不稳定。

---

## 10. 新方向九：Low-rank Shared Edge-function Bank，减少 dense KAN 统计量并增强泛化

### 10.1 动机

Dense KAN 为每条 edge 单独学习一元函数：

$$
\phi_{ji}(t)
$$

这表达力强，但也带来两个问题：

1. 参数量和 edge stats memory 高；
2. 每条 edge 独立拟合，可能过拟合或 credit 噪声高。

Stage E 的 sufficient statistics 实验已经说明，dense edge stats 是 memory reduction 的瓶颈之一。因此可以引入共享函数 bank。

---

### 10.2 Low-rank function bank 形式

设共享一元函数为：

$$
\psi_r(t),\quad r=1,\dots,R
$$

令：

$$
\phi_{ji}(t)=\sum_{r=1}^{R}u_{jr}v_{ir}\psi_r(t)
$$

其中 $R$ 远小于输入 / 输出维度。

这样 KAN branch 为：

$$
y_j=\sum_i\sum_r u_{jr}v_{ir}\psi_r(x_i)
$$

也可以写成：

$$
y=U\left(\sum_i v_i\odot \psi(x_i)\right)
$$

这类似 KAN 的 LoRA / tensor factorization。

---

### 10.3 Functional update

共享函数的 gradient 为：

$$
\nabla_{\psi_r}L
=
\sum_{j,i}u_{jr}v_{ir}\nabla_{\phi_{ji}}L
$$

coefficient update 作用在共享 $\psi_r$ 上，而不是每条 edge 的独立函数上。

这样 edge stats memory 从：

$$
O(d_{out}d_{in}M)
$$

降低到近似：

$$
O(RM+d_{out}R+d_{in}R)
$$

如果 $R\ll d$，这对 Stage E memory 路线很重要。

---

### 10.4 预期作用

Low-rank shared bank 可能同时改善：

```text
泛化：共享函数降低 edge-level 过拟合
memory：减少 edge stats
compute：减少 dense edge update
credit SNR：多个 edge 共享统计，提高稳定性
```

推荐实验：

```text
rank R in {4, 8, 16, 32}
compare Dense DG-KAN, Grouped DG-KAN, LowRankBank DG-KAN
```

指标：

```text
test acc
val AUC
params
edge_stats_MB
peak memory
phi_prime_p95
Jacobian condition
no-KAN acc drop
```

如果 LowRankBank 能保持精度并降低 memory，它可能成为 DG-KAN 扩展到更大模型的关键结构。

---

## 11. 新方向十：Hypergradient / Controller 学习 Sobolev metric 参数

### 11.1 动机

当前 Sobolev metric 的参数 $\alpha,\beta,\rho$ 多数靠 sweep。不同数据集、不同层、不同训练阶段需要的平滑强度明显不同。Fashion-MNIST 和 KMNIST 的 schedule 行为已经说明，固定 metric 参数不够灵活。

因此可以把 metric 参数变成可学习或可控制的对象。

---

### 11.2 Hypergradient 形式

每层 metric：

$$
M_k(\alpha_k,\beta_k,\rho_k)
=
M_0+\alpha_kR_1+\beta_kR_2+\rho_kI
$$

训练一步：

$$
a_{t+1}=a_t-\eta M_k^{-1}\nabla_a L_{train}
$$

然后用 validation loss 更新 metric 参数：

$$
\nabla_{\alpha_k}L_{val}(a_{t+1})
$$

为了稳定，使用 log 参数：

$$
\tilde\alpha_k=\log\alpha_k
$$

$$
\tilde\beta_k=\log\beta_k
$$

$$
\tilde\rho_k=\log\rho_k
$$

第一版不需要每步做 hypergradient，可以每个 epoch 或每 $N$ steps 用一个小 validation batch 做一次。

---

### 11.3 Controller 近似

如果 hypergradient 实现成本太高，可以先用规则 controller：

```text
if train loss 下降慢 and branch under-active:
    decrease rho or beta

if val loss 不降 but train loss 降:
    increase beta or rho

if phi_prime / Jacobian 超阈值:
    increase beta or rho

if no-KAN drop 太小:
    decrease rho or increase coeff lr multiplier
```

这相当于 metric-level geometry-aware controller。

---

## 12. 新方向十一：Functional Langevin，在函数空间中注入平滑噪声

### 12.1 动机

普通 SGD 的噪声在参数空间中，而 DG-KAN 更自然的噪声应该在函数空间中。可以在 functional update 中加入 Langevin noise：

$$
a_{t+1}
=
a_t-\eta M^{-1}g_t
+
\sqrt{2\eta T}M^{-1/2}\xi_t
$$

其中：

$$
\xi_t\sim\mathcal N(0,I)
$$

噪声协方差为：

$$
\operatorname{Cov}(\Delta a)=2\eta T M^{-1}
$$

这意味着噪声主要出现在低 Sobolev norm 的平滑函数方向，高频 / 高曲率方向噪声较小。这比普通参数噪声更符合函数空间先验。

---

### 12.2 使用方式

建议只在后期使用：

```text
前 70% steps:
  geometry-aware functional update

后 30% steps:
  functional Langevin + function-space EMA
```

推荐：

```text
temperature in {1e-5, 3e-5, 1e-4}
start_frac in {0.5, 0.7}
combine_with_ema = true
```

这个方向主要用于小数据 / label noise 泛化，不应优先用于追训练 loss。

---

## 13. 推荐优先级

这些新想法很多，但不应该全部同时做。按当前项目状态，我建议优先级如下。

第一优先级是 Functional-SAM。它直接针对泛化和 sharpness，且和现有 Sobolev metric 完全兼容。你已经有 $\phi'_{p95}$、curvature、Jacobian condition、loss AUC 等指标，很容易判断它是否有效。

第二优先级是 SNR-adaptive edge update。它非常贴合 DG-LCA 的 credit assignment 视角，可以回答一个更细的问题：哪些 edge 的 credit 可靠，哪些 edge 只是噪声。它也可能直接缓解 branch under-active 与 over-active 之间的矛盾。

第三优先级是 Block-level Functional Target Solve。它可能显著提升收敛速度，是对当前 $M^{-1}g$ update 的实质升级。尤其适合 KMNIST，因为 KMNIST 目前更像是 branch 任务参与度不足，而不是单纯 geometry threshold 没调好。

第四优先级是 Credit Curriculum。它利用 IdentityAdj 的稳定性和 AnalyticAdj / FirstOrderAdj 的正确性，可能让 hard task 早期训练更顺。

第五优先级是 Function-space EMA / SWA。它成本低，几乎可以作为所有实验的附加增强。

Low-rank function bank 和 adaptive grid 更偏结构升级，适合 Stage E / 更大模型 / CIFAR 之后推进。Functional Langevin 和 hypergradient metric 更偏研究型，可以作为后续泛化实验的探索线。

---

## 14. 建议的下一轮实验：Stage F7

如果下一轮只做一个新实验包，推荐做：

$$
\boxed{
\text{Stage F7：Functional-SAM + SNR-adaptive edge update}
}
$$

这个包直接面向当前最重要的问题：functional update 如何进一步增强泛化能力和收敛速度。

---

### 14.1 实验设置

数据集：

```text
Fashion-MNIST
KMNIST
```

模型：

```text
Residual DG-KAN
depth = 4
hidden_dim = 64
basis_count = 16
alpha_init = 1.5
```

训练：

```text
epochs = 20
seeds = 0,1,2,3,4
rest optimizer = AdamW
functional update = current best geometry-aware schedule
```

对照方法：

```text
AdamW
current best geometry-aware functional
geometry-aware + Functional-SAM
geometry-aware + SNR-adaptive update
geometry-aware + Functional-SAM + SNR-adaptive update
```

---

### 14.2 Functional-SAM 参数

```text
rho_func in {0.03, 0.05}
sam_interval in {4, 8}
sam_start_frac = 0.25
metric = Sobolev
```

为了避免矩阵求逆重复开销，第一版可以复用当前 Sobolev inverse：

$$
\delta a_{adv}
=
\rho
\frac{M^{-1}g}{\sqrt{g^\top M^{-1}g+\epsilon}}
$$

---

### 14.3 SNR-adaptive 参数

```text
snr_ema_beta = 0.95
snr_tau = median_snr_per_layer
eta_min_mult = 0.5
eta_max_mult = 1.5
rho_min_mult = 0.5
rho_max_mult = 2.0
```

第一版可以只调 lr，不调 $\rho$ / $\beta$：

$$
\eta_{ji}=\eta_0\cdot m_{ji}^{snr}
$$

第二版再加入 damping adaptation。

---

### 14.4 成功标准

Fashion-MNIST：

```text
test_acc >= current geometry-aware best
val_loss_auc improvement vs AdamW > 15%
J condition reduction vs AdamW > 35%
phi_prime_p95 reduction vs AdamW > 20%
```

KMNIST：

```text
test_acc gap vs AdamW < 0.75%
val_loss_auc improvement vs AdamW > 10%
J condition reduction vs AdamW > 15%
phi_prime_p95 reduction vs AdamW > 25%
```

如果 KMNIST gap 从当前约 $1.06\%$ 降到 $0.75\%$ 以内，同时保持 AUC 和 geometry 优势，就说明新机制真正超过了 schedule tuning。

---

## 15. 建议的后续实验：Stage F8 与 Stage F9

### 15.1 Stage F8：Block-level Functional Target Solve

目标：验证局部函数目标求解是否能改善 branch under-active 和收敛速度。

方法：

```text
current geometry-aware functional
last-block target solve
last-2-block target solve
target solve + Sobolev gradient hybrid
```

参数：

```text
target_solve_frac in {0.1, 0.25, 0.5}
ridge_lambda in {1e-3, 1e-2, 1e-1}
apply_start_frac in {0.0, 0.1}
apply_end_frac in {0.25, 0.5}
```

主要看：

```text
branch_over_adamw
no_kan_acc_drop
val AUC
test acc gap
Jacobian condition
```

---

### 15.2 Stage F9：Function-space EMA + Spectral regularization

目标：验证 function-space averaging 和 spectral high-frequency control 是否提升泛化。

方法：

```text
current best functional
+ coefficient EMA
+ Sobolev-smoothed EMA
+ spectral dropout
+ spectral shrink
+ EMA + spectral dropout
```

重点 stress：

```text
small train size
label noise
longer training
```

如果这些方法有效，应该在小数据和 label noise 下比 AdamW 或 raw functional update 更有优势。

---

## 16. 需要新增的实现模块

建议在现有代码中加入以下组件。

### 16.1 FunctionalSAMController

负责：

```text
compute adversarial perturbation in M-norm
apply temporary coefficient perturbation
recompute loss / gradient
restore coefficients
return SAM gradient
```

核心接口：

```text
sam_grad = functional_sam.compute_grad(model, batch, metric_cache)
```

---

### 16.2 EdgeSNRTracker

负责：

```text
maintain EMA mean / variance of edge gradient stats
compute edge-level SNR
return lr / rho / beta multipliers
```

核心接口：

```text
snr_stats = snr_tracker.update(layer_id, coeff_grad)
mult = snr_tracker.get_multiplier(layer_id)
```

---

### 16.3 FunctionalTargetSolver

负责：

```text
construct local design matrix or sufficient stats
solve ridge / Sobolev projection
return delta coefficients
```

核心接口：

```text
delta_a = target_solver.solve(z, credit, metric, alpha)
```

---

### 16.4 FunctionEMA

负责：

```text
maintain KAN coefficient EMA
optionally apply Sobolev-smoothed projection
swap EMA weights for evaluation
```

核心接口：

```text
function_ema.update(model)
function_ema.swap_to_ema(model)
function_ema.restore(model)
```

---

### 16.5 SpectralRegularizer

负责：

```text
precompute eigenbasis of derivative penalty matrix
apply spectral dropout or spectral shrink
record high-frequency energy
```

核心 metrics：

```text
spectral/high_freq_energy
spectral/dropout_active_fraction
spectral/shrink_norm
```

---

## 17. 新增 W&B / CSV logging keys

为了判断这些新机制是否真的有效，建议新增以下指标。

Functional-SAM：

```text
fsam/enabled
fsam/rho
fsam/interval
fsam/adv_norm_M
fsam/base_loss
fsam/adv_loss
fsam/sharpness_gap
fsam/grad_cos_base_adv
fsam/update_norm_M
```

SNR-adaptive update：

```text
snr/global/mean
snr/global/p10
snr/global/p90
snr/global/low_snr_fraction
snr/layer_{k}/mean
snr/layer_{k}/low_fraction
update/snr_lr_mult_mean
update/snr_lr_mult_p90
metric/snr_rho_mult_mean
```

Target solve：

```text
target_solve/layer_{k}/residual_mse
target_solve/layer_{k}/ridge_condition
target_solve/layer_{k}/delta_norm_M
target_solve/layer_{k}/actual_branch_delta_norm
target_solve/global/actual_descent
```

Function EMA：

```text
ema/test_acc
ema/val_loss
ema/phi_prime_p95
ema/jacobian_condition
ema/raw_acc_gap
ema/raw_jcond_gap
```

Spectral regularization：

```text
spectral/layer_{k}/high_freq_energy
spectral/layer_{k}/low_freq_energy
spectral/layer_{k}/dropout_rate_mean
spectral/layer_{k}/shrink_ratio
```

Edge mixup：

```text
edge_mixup/enabled
edge_mixup/ratio
edge_mixup/alpha
edge_mixup/stats_norm_ratio
edge_mixup/credit_cos_original_mixed
```

---

## 18. 最终建议

当前 DG-KAN functional update 已经证明了一个重要方向：函数空间训练几何可以带来更稳定的 $\phi'$、更低 Jacobian condition、更好的 loss AUC，并且在 Fashion-MNIST 上通过 geometry-aware schedule 达到甚至超过 AdamW accuracy。现在剩下的问题不是“这个方向是否有效”，而是如何把它从 schedule tuning 提升为真正的 function-space optimizer。

本补充方案中，最值得优先推进的是：

$$
\boxed{
\text{Functional-SAM}
+
\text{SNR-adaptive edge update}
}
$$

Functional-SAM 直接作用于函数空间 sharpness，最可能增强泛化；SNR-adaptive edge update 直接作用于 edge-level credit reliability，最可能缓解 branch under-active 和 noisy edge overfitting。

如果这两者有效，DG-KAN 的 functional update 主张会更强：

$$
\boxed{
\text{不是简单地用 Sobolev metric 替代 AdamW，}
\quad
\text{而是在 function-valued primitive 上建立一套泛化可控的训练几何。}
}
$$

这会把 DG-KAN / DG-LCA 的故事从“局部 credit 和低显存”扩展到更完整的命题：

$$
\boxed{
\text{function-valued neural primitive should be trained with function-space geometry,}
\quad
\text{not merely coefficient-space gradients.}
}
$$

