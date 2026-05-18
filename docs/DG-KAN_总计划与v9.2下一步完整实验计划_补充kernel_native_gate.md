# DG-KAN 总计划与 v9.2 下一步实验计划：从经验成功点到 Kernel-Native Clean FullEdge PureKAN Functional System

> 本文件分为两部分。  
> **第一部分是总计划**：重新定义终极目标，总结过去实验已经验证的事实，尤其说明此前最好的模型到底怎么做、为什么它强、为什么它还不是终极目标，并给出从当前状态走向 Next-Gen Beyond-MLP 的总体路线。  
> **第二部分是下一步计划**：聚焦 v9.2 的近期目标，即解决 v9.1 暴露出的 “basis-source conditioning / 基函数输入源与归一化” 问题，建立有 justification 的默认基函数与 materialization-free 前向/反传路径。  
> 本计划继续遵守：**no teacher、no self-teacher、no distillation、no loss modification、no label smoothing、no sampler/class weight、no CPU offload、no fake/proxy、KAN path 不使用 PyTorch loss.backward graph**。

---

# Part I. 总计划

## 1. 终极目标：Next-Gen Beyond-MLP

DG-KAN 的终极目标不是在 MNIST-family 上找到一个偶然超过 MLP 的小模型，也不是证明某个 KAN 变体在单点配置下能跑出更高 accuracy。终极目标是构建一个可以作为 MLP 替代甚至超越 MLP 的 **Clean FullEdge PureKAN Functional Training System**。它必须同时满足以下条件。

第一，模型结构必须是 PureKAN。所有官方候选的可学习参数必须属于 edge-function system。允许每条 edge 含有 identity basis、低阶残差 basis、RBF/rational/piecewise 等函数基，但不能把普通 Linear-SiLU stack 冒充为终极 FullEdge PureKAN。FullEdge 层的形式应为：

$$
y_j=\sum_i \phi_{ij}(x_i).
$$

如果采用 hybrid residual basis，也仍然必须写成 edge function：

$$
\phi_{ij}(x_i)=w^{(0)}_{ij}\tilde{x}_i+\rho_{ij}(\tilde{x}_i),
$$

其中 $w^{(0)}_{ij}\tilde{x}_i$ 是 edge 内部 identity basis channel，不是额外 non-KAN linear layer。

第二，训练路径必须 graph-free。官方 KAN path 必须使用 manual forward、manual backward、manual update，不依赖 PyTorch `loss.backward()` graph。基础 task update 可以是 AdamW-equivalent manual update：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\theta_{t+1}
=
\theta_t-\eta
\left(
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
+\lambda\theta_t
\right).
$$

第三，functional update 必须是 update rule，不是 loss。训练目标保持标准 CE：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

允许更新为：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\Delta\theta_{\text{functional}}.
$$

不允许变成：

$$
L=CE+\lambda L_{\text{geo}}.
$$

第四，系统效率必须和 MLP 可比。最低目标不是“能跑”，而是在参数量、FLOPs、显存、step time 上进入公平 envelope：

$$
Params_{\text{KAN}}\leq1.05Params_{\text{MLP}},
$$

$$
ForwardFLOPs_{\text{KAN}}\leq1.05ForwardFLOPs_{\text{MLP}},
$$

$$
BackwardFLOPs_{\text{KAN}}\leq1.50BackwardFLOPs_{\text{MLP}},
$$

$$
StepTime_{\text{KAN}}\leq1.50StepTime_{\text{MLP}},
$$

$$
PeakMemory_{\text{KAN}}\leq1.05PeakMemory_{\text{MLP}}.
$$

更强目标是：

$$
StepTime_{\text{KAN}}\leq1.20StepTime_{\text{MLP}},
$$

$$
PeakMemory_{\text{KAN}}<PeakMemory_{\text{MLP}}.
$$


进一步补充一条前置硬要求：**任何感兴趣的基函数架构，在进入 official full task 之前，必须先证明它存在与同参数量 MLP 可比的前向、反传、显存路径。** 这不是附属 profiler，而是 basis / architecture 的资格门槛。也就是说，一个 basis 即使通过 conditioning、fit、gradcheck、small overfit，只要其 forward/backward/memory 无法和同参数量 MLP 接近，就只能作为 diagnostic basis，不能进入 official route。

同参数 MLP 对照定义为：

$$
\left|
\frac{Params_{\text{KAN}}-Params_{\text{MLP-match}}}
{Params_{\text{MLP-match}}}
\right|
\leq0.05.
$$

对每个 basis architecture，必须单独记录并通过：

$$
T_{\text{forward,KAN}}\leq1.25T_{\text{forward,MLP-match}},
$$

$$
T_{\text{backward,KAN}}\leq1.50T_{\text{backward,MLP-match}},
$$

$$
T_{\text{step,KAN}}\leq1.50T_{\text{step,MLP-match}},
$$

$$
M_{\text{peak,KAN}}\leq1.05M_{\text{peak,MLP-match}}.
$$

强目标为：

$$
T_{\text{forward,KAN}}\leq1.10T_{\text{forward,MLP-match}},
$$

$$
T_{\text{backward,KAN}}\leq1.25T_{\text{backward,MLP-match}},
$$

$$
T_{\text{step,KAN}}\leq1.20T_{\text{step,MLP-match}},
$$

$$
M_{\text{peak,KAN}}<M_{\text{peak,MLP-match}}.
$$

如果某个 basis 的 naive PyTorch / einsum 实现出现下列任一情况：

$$
T_{\text{forward/backward,KAN}}>1.50T_{\text{MLP-match}},
$$

或：

$$
M_{\text{peak,KAN}}>1.05M_{\text{MLP-match}},
$$

但它已经显示出明确的 task / approximation / geometry 信号，则该 basis 不得直接进入 official task，而必须进入 `KernelizationRequired` 路线：先实现 `torch.compile` / Triton / CUDA fused forward-backward，再重新做 correctness 和 microbench。若 kernelization 后仍不能通过上述 gate，则该 basis 只能保留为 diagnostic，不得作为 Next-Gen Beyond-MLP official candidate。

因此，FullEdge official path 明确禁止 materialize：

$$
B\times d_{in}\times d_{out}\times K.
$$

任何候选如果必须构造这一 dense edge tensor，默认判为：

```text
materialization_violation = 1
official_eligible = 0
```

除非它已经被定制 kernel 证明实际 peak memory 和 step time 与同参数 MLP 可比。

第五，任务表现不能只是“平替”。最终目标是 **显著优势**：

$$
Acc_{\text{KAN}}\geq Acc_{\text{MLP}}+\delta,
$$

其中 vision / tabular classification 的强目标取：

$$
\delta\geq0.005.
$$

同时还应在至少一个功能性维度上超越 MLP：

```text
ECE / NLL 更好；
ValLossAUC_step 或 ValLossAUC_time 更好；
sample efficiency 更好；
robustness 更好；
catastrophic forgetting 更低；
symbolic / smooth function representation 更强；
geometry 更平滑且不损害 task。
```

第六，架构要能扩展。FC FullEdge primitive 做稳之后，要自然推广到：

```text
KANConv:
  用 patch-local edge function 替代卷积核中的 scalar weight；

KAN-FFN / KAN-Transformer:
  用 channel-wise / token-wise kernel-native edge function 替代 Transformer FFN 中的 MLP。
```

因此，终极目标可以写成：

$$
\boxed{
\text{Next-Gen Beyond-MLP}
=
\text{Clean FullEdge PureKAN}
+
\text{Graph-Free Manual Training}
+
\text{Kernel-Native Efficiency}
+
\text{Task-Safe Functional Update}
+
\text{External Fair Advantage}
+
\text{Scalable Conv/Transformer Extension}.
}
$$

---

## 2. 过去实验已经验证了什么

### 2.1 表达力不是空的：dense KAN 确实有 Beyond-MLP 信号

v7.0 最重要的事实是 dense manual KAN 的 `D3-dense-poly2-gate` 首次打开了 basic Beyond-MLP task accuracy gate。它在 3 个 primary datasets 上 mean val gap 达到约 `+0.0206`，并且 3/3 datasets 都不低于 MLP；其 ECE/NLL 也优于 MLP。但 dense D3 不是 kernel-native official candidate，full-step efficiency probe 9/9 shapes fail，FastOpt 后 step 和 memory 仍不过 S2。这说明 KAN-family 有真实表达力，但 dense/materialized 实现不可作为终点。

这一阶段的核心结论是：

$$
\boxed{
\text{KAN basis / edge nonlinearity 可以产生超过 MLP 的任务信号，但 naive dense 实现太重。}
}
$$

### 2.2 高效路线能快，但表达力容易掉

v7.0 official grouped/T3/G2 路线有很好的系统信号：memory ratio 约 `1.0312`，step ratio 约 `0.8666`，但 task gap vs MLP 约 `-0.0326`，尤其 KMNIST gap 很大。这说明：只做 grouped / compression / lightweight kernel path 会把有效表达力压坏。

### 2.3 C3 / M12 / M13 证明了“表达力 + 系统”有接近闭合的区域，但 teacher-free 和 formal 条件很关键

v7.5/v7.6 的 M12 在 strict、GradPass、MacroSignificantPass、FullGridS2 上出现过强成功，但它含有 external C3 teacher/distillation，只能作为 diagnostic assisted route。v7.8/v7.9 把 official route 改成 teacher-free 后，M13 仍有稳定 positive signal，但 macro gap 约 `+0.01784`，低于 `+0.0200` hard gate。因此可以确认：

```text
teacher-assisted route 证明 reachable signal；
teacher-free route 证明 autonomous signal 已接近；
但 official success 不能依赖 external teacher。
```

### 2.4 v8.3-v8.4 证明 functional update 可以 system-gated re-entry，但 timing formalization 是硬问题

v8.3 建立了 system-gated functional update re-entry：functional update 只有在 base candidate 过 H0 system gate 后才能进入 full task。后续 FT7 作为 role-wise guarded functional update 进入了 P7/P8/P9，并完成 minimum success；v8.4 进一步做 causality、role mechanism、TimeAUC/profiler，但也发现 50/200 official-style timing 不稳定，高-rep timing 下才能闭合 minimum。因此 functional update 有价值，但不能裸替代 AdamW，也不能跳过系统 gate。

### 2.5 v8.5-v8.7 证明 transitional route 很强，但不是终极 PureKAN

v8.5 把 DG-KAN 接入 KANbeFair 外部公平框架，在 MNIST full protocol 下出现重要成功点：`DG1-FT7-KW6 hidden28 stride128` 相比 KB-MLP 有更高 test accuracy、更少 params、更少 forward FLOPs，并在 step time gate 内，同时保留 functional geometry improvement。v8.7 进一步把 fixed recipe 推进为 selected / adaptive / no-manual route：自动选 `KW4 hidden28`，在 FMNIST/KMNIST 上也通过 external fair formal route，P5 timing 的 q90 step ratio 约 `0.9336`，P7 FMNIST/KMNIST 分别比 KB-MLP 高约 `+1.41%` 和 `+2.18%`，且 no-fake/no-proxy/offload audit 干净。

但这条路线的结构是：

```text
packed / cached linear-SiLU stack
+
poly2_silu KAN-style head
+
manual AdamW-equivalent update
+
adaptive functional update
```

它是非常强的 **transitional DG-KAN success**，不是终极 FullEdge PureKAN success。不能把 linear-SiLU stack 成功直接写成 full-edge PureKAN 成功。

### 2.6 v9.0 证明 clean code / clean CE 必须重做，且 transitional CleanCE 成功，但 FullEdge 失败

v9.0 完成了核心化整改、contract hardening、CleanCE transitional revalidation。重要的是，去掉 label smoothing 后 transitional CleanCE route 没有 collapse，clean functional 在 MNIST 上仍高于 KB-MLP。但 FullEdge manual implementation 虽然通过了 edge-layer gradcheck、functional causality、robust timing，却在外部公平 task 上失败：MNIST、FMNIST、KMNIST 的 best FullEdge functional 均低于 KB-MLP，并且 forward FLOPs、backward estimate、memory 都不公平。

v9.0 的结论是：

$$
\boxed{
\text{CleanCE transitional success 成立；Clean FullEdge PureKAN 尚未成功。}
}
$$

### 2.7 v9.1 证明 basis 选择没有 justification，raw input 直接上 basis 是当前核心 blocker

v9.1 已补齐 planned basis family 的 dense manual FullEdge forward/backward，BAS0-BAS9 的 P3 gradcheck 全部通过，P4 512-sample overfit 中 BAS0、BAS6、BAS7、BAS9 通过。但 P1 raw-input conditioning 是 0/10 basis 通过，所有 basis 都卡在 dominant component 或 condition number；没有任何 basis 同时通过 conditioning + fit + gradcheck + overfit。因此不能进入 basis-justified FullEdge external success。

v9.1 的核心结论是：

$$
\boxed{
\text{FullEdge 不是没实现，也不是反传错；真正 blocker 是 basis-source conditioning。}
}
$$

也就是说，直接把基函数作用在 raw flattened pixels 上是不合理的。下一步应优先研究：

```text
basis 输入源；
normalization；
fan-in scaling；
edge-owned affine normalization；
hybrid residual parameterization；
materialization-free compute。
```

---

## 3. 之前最好的模型是怎么做的

截至目前，最强的 practical / external-fair 模型不是 FullEdge，而是 v8.7 selected transitional route。它的实质如下。

### 3.1 架构

模型主体是 packed / cached linear-SiLU stack，后接 KAN-style `poly2_silu` head。其形式可以理解为：

```text
x
 -> packed linear / SiLU stack
 -> poly2_silu_base head
 -> logits
```

stack 近似：

$$
h_{l+1}=\operatorname{SiLU}(h_lW_l^\top),
$$

而 head 的 edge transform 近似包含：

$$
z_i
=
b_{i,0}x_i
+
b_{i,1}\operatorname{SiLU}(x_i)
+
s(p_{i,0}x_i+p_{i,1}x_i^2).
$$

这解释了它为什么强：linear-SiLU stack 保留了 MLP-like 的稳定 cross-channel feature mixing，poly2_silu head 提供 KAN-style edge nonlinear residual，functional update 提供 geometry correction。

但这也解释了它为什么不是终极目标：stack 不是 full-edge learnable function system。

### 3.2 训练

基础训练目标是 CE-only，不能使用 teacher、self-teacher、distillation、label smoothing、sampler/class weight、CPU offload。基础 task update 是 AdamW-equivalent manual/foreach/addcdiv/no-sync 实现，功能上不是新 optimizer，而是为了减少临时 tensor、CPU sync 和 Python overhead。

Functional update 是额外的 update-rule correction，不是 loss。v8.3-v8.7 中的 FT7 / Adaptive-FT 主要是 role-wise / event-triggered / task-budgeted geometry correction。其目标是降低 curvature，同时不破坏 CE task descent。

### 3.3 系统实现

v8.7 的成功依赖了 system-side repair：

```text
foreach AdamW / addcdiv / no-sync；
compiled fused CE/head backward；
real-batch prewarm；
packed parameter owner；
manual forward/backward；
部分 no-input-grad / recompute path。
```

它不是完整自定义 kernel-native FullEdge 实现，但已经证明 manual graph-free route 可以进入外部公平 timing envelope。

### 3.4 指标

v8.7 selected route 的关键结果包括：

```text
P4 selected base = KW4 hidden28
params ratio = 0.939568
FLOPs ratio = 0.936347
selected confirmation delta vs KB-MLP = +0.279999
curvature ratio = 0.772483
P5 q90 step ratio = 0.933570
P7 FMNIST delta vs KB-MLP = +1.410002
P7 KMNIST delta vs KB-MLP = +2.179998
```

因此，我们应把它定义为：

$$
\boxed{
\text{Best transitional external-fair DG-KAN route}
}
$$

而不是：

$$
\boxed{
\text{Terminal FullEdge PureKAN route}
}
$$

---

## 4. 总体技术路线：从 transitional success 到 terminal FullEdge success

基于以上事实，未来总路线应分为六条主线。

### 主线 A：Basis-source-conditioned FullEdge primitive

不再默认 `poly2_silu`，也不再直接让 basis 吃 raw flattened pixels。每个 basis 必须和输入源、归一化、fan-in scaling 一起设计。默认形式应从 hybrid residual edge 开始：

$$
\phi_{ij}(x_i)=w^{(0)}_{ij}\tilde{x}_i+\rho_{ij}(\tilde{x}_i).
$$

优先 basis：

```text
Vision default:
  normalized Legendre / Chebyshev residual basis

Local-support diagnostic:
  non-recursive piecewise linear active-k residual basis

Smooth-local diagnostic:
  shared / low-rank / active-center RBF residual basis

High-expressivity diagnostic:
  bounded rational residual basis

Symbolic:
  non-recursive spline / piecewise / B-spline compatible active-k basis

Later Conv/signal:
  wavelet / Fourier diagnostic
```

### 主线 B：Materialization-free kernel-native implementation

禁止 naive dense FullEdge materialization：

$$
B\times d_{in}\times d_{out}\times K.
$$

默认采用 identity base + low-rank nonlinear residual：

$$
Y=XW^{(0)}+RV^\top,
$$

其中：

$$
R_{b,r}=\sum_i u_{i,r}\psi_r(\tilde{x}_{b,i}),
$$

$$
\psi_r(\tilde{x})=\sum_k a_{r,k}B_k(\tilde{x}).
$$

这样前向复杂度从：

$$
O(Bd_{in}d_{out}K)
$$

降为：

$$
O(Bd_{in}RK+BRd_{out}),
$$

当 $R\ll d_{out}$ 时才有可能和 MLP 可比。

### 主线 C：先 AdamW 可训，再 functional update

每个 FullEdge basis 必须先证明：

$$
Acc_{\text{PureKAN-AdamW}}\geq Acc_{\text{MLP-AdamW}}-\epsilon.
$$

若 AdamW 下完全训不动，不允许用 functional update 去“救 task”。Functional update 只允许作为 geometry / robustness / forgetting improvement：

$$
Acc_{\text{Functional}}\geq Acc_{\text{AdamW}}-0.005,
$$

$$
R_{\text{curv,Functional}}\leq0.90R_{\text{AdamW}}.
$$

### 主线 D：External fair evaluation

继续使用 KANbeFair，但不只看 MNIST-family。至少要覆盖：

```text
MNIST / Fashion-MNIST / KMNIST
symbolic tasks
tabular / machine learning tasks if runnable
CIFAR-10 if runnable
continual splits
```

同时每次都要报告：

```text
params
forward FLOPs
backward estimate
step time
memory
kernel time
accuracy / RMSE
ECE / NLL
robustness
functional geometry
```

### 主线 E：Robustness / scaling / continual

functional update 的价值不能只停在 curvature。它必须至少在一类下游优势上成立：

```text
更好 calibration；
更好 robustness；
更好 sample efficiency；
更低 forgetting；
更好 symbolic function fit；
更快 convergence。
```

### 主线 F：Conv / Transformer extension

FullEdge FC primitive 稳定后再扩展。KANConv 不应是全连接 edge explosion，而应是 patch-local identity conv base + nonlinear residual edge。KAN-Transformer 应先替换 FFN，而不是改 attention。

---

## 5. 总路线里程碑

### Milestone 1：v9.2 Basis-source-conditioned hybrid residual primitive

目标：证明 raw input conditioning 失败可以通过 feature source / normalization / hybrid residual basis 修复，并找到至少一个 justified basis candidate。

成功标准：

$$
\kappa(B^\top B)\leq10^4,
$$

$$
DeadBasisFraction\leq0.30,
$$

$$
DominantBasisFraction\leq0.70,
$$

$$
TrainAcc_{512}\geq0.98,
$$

$$
GradRelErr_{\max}\leq10^{-4}.
$$

### Milestone 2：v9.3 Materialization-free FullEdge system closure

目标：将 v9.2 justified basis 做成 low-rank/shared/grouped materialization-free implementation。

成功标准：

$$
ForwardFLOPsRatio\leq1.05,
$$

$$
BackwardFLOPsRatio\leq1.50,
$$

$$
StepRatio\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### Milestone 3：v9.4 Clean FullEdge external fair validation

目标：在 KANbeFair MNIST/FMNIST/KMNIST 上重新与 KB-MLP / KB-KAN 比较。

成功标准：

$$
Acc_{\text{FullEdge}}\geq Acc_{\text{KB-MLP}},
$$

同时满足 external fair envelope。

### Milestone 4：v9.5 Functional advantage formalization

目标：证明 functional update 对 justified FullEdge primitive 有独特价值。

成功标准：

$$
R_{\text{curv,Functional}}\leq0.90R_{\text{Base}},
$$

$$
Acc_{\text{Functional}}\geq Acc_{\text{Base}}-0.005,
$$

并且至少一个下游指标改善：

```text
ECE / NLL / robustness / forgetting / sample efficiency / symbolic RMSE。
```

### Milestone 5：v10 KANConv / KAN-FFN prototype

目标：把 FullEdge primitive 推广到 convolution-like 和 Transformer-FFN-like 模块。

成功标准：

```text
在一个小型 vision scale task 上，与 Conv/MLP-FFN baseline 进行 params/FLOPs/time/memory fair comparison；
至少不劣于 baseline，并保留 geometry or robustness advantage。
```

---

# Part II. 下一步计划：v9.2 Basis-Source-Conditioned Hybrid Residual PureKAN

## 6. v9.2 的整体目标

v9.2 不继续盲目增加 basis，也不直接进入 external full task。v9.2 的目标是解决 v9.1 暴露出的最核心 blocker：

$$
\boxed{
\text{raw flattened input 上所有 basis conditioning fail。}
}
$$

因此 v9.2 的中心问题是：

$$
\boxed{
\text{如果给 basis 正确的输入源、归一化和 hybrid residual 参数化，能否得到一个 justified、可训练、可 kernelize 的 FullEdge basis？}
}
$$

v9.2 不把 final external fair success 作为唯一目标。它的 minimum success 是得到至少一个 **justified basis-source-parameterization combination**：

```text
conditioning pass
one-layer fit pass
small overfit pass
gradcheck pass
microbench near-pass
```

v9.2 的 strong success 是这个组合还能在 MNIST/FMNIST/KMNIST 上接近或超过 MLP，并且没有明显 system fail。

---

## 7. v9.2 核心假设

### H1：v9.1 的 0/10 conditioning fail 主要来自 basis source，而不是所有 basis family 本身失败

v9.1 只在 raw flattened input 上做 conditioning。H1 假设：如果引入 normalized / clipped / fan-in normalized / edge-owned affine source，则至少一个 basis family 会通过 conditioning。

H1 成立标准：

存在 source-basis 组合满足：

$$
\kappa(B^\top B)\leq10^4,
$$

$$
DeadBasisFraction\leq0.30,
$$

$$
DominantBasisFraction\leq0.70.
$$

如果所有 source-basis 组合仍失败，则说明问题更深：FullEdge scalar basis 与当前 vision input family 不匹配，需转 patch / conv source。

### H2：normalized orthogonal residual basis 是 vision-family 的默认候选

H2 假设：vision-family 中最适合第一批 official FullEdge 的不是 spline/RBF/rational，而是 identity edge base + normalized low-order orthogonal residual。

候选：

```text
Legendre3 residual
Chebyshev3 residual
Hermite3 residual diagnostic
```

形式：

$$
\phi_{ij}(x_i)
=
w^{(0)}_{ij}\tilde{x}_i
+
\sum_{r=1}^{R}u_{i,r}v_{j,r}
\sum_{k=2}^{K}a_{r,k}P_k(\tilde{x}_i).
$$

H2 成立标准：

相比 monomial compact poly：

$$
\kappa_{\text{orthogonal}}\leq0.5\kappa_{\text{monomial}},
$$

$$
Acc_{\text{orthogonal}}\geq Acc_{\text{monomial}}-0.002,
$$

且：

$$
StepRatio_{\text{orthogonal}}\leq StepRatio_{\text{monomial}}+0.05.
$$

### H3：shared / low-rank RBF 只在 materialization-free 形式下值得进入 official

RBF 值得研究，但不能 dense materialize。H3 成立标准：

$$
MemoryRatio_{\text{RBF-lowrank}}\leq1.05,
$$

$$
StepRatio_{\text{RBF-lowrank}}\leq1.50,
$$

且 local bump / symbolic-like diagnostic 上 RBF 优于 polynomial：

$$
MSE_{\text{RBF}}<MSE_{\text{poly}}.
$$

### H4：bounded rational basis 需要 stability gate，而不是直接 official

Rational basis 有 KAT 经验动机，但必须先过稳定性。

形式：

$$
\phi(x)=\frac{p(x)}{1+\operatorname{softplus}(q(x))}.
$$

H4 成立标准：

$$
\text{NaN/Inf rate}=0,
$$

$$
\max|\phi'(x)|\leq C,
$$

$$
BadStepRate\leq0.02,
$$

$$
\kappa(B^\top B)\leq10^4.
$$

如果 rational task acc 高但 bad step 或 gradient tail 失控，则不能进入 official。

### H5：functional update 应只作用在 nonlinear residual / high-curvature subspace

H5 假设：对 identity edge base 做 smoothing 会伤害 task；functional update 应只作用于 residual basis coefficient 或 high-curvature subspace。

H5 成立标准：

$$
Acc_{\text{ResidualFunctional}}\geq Acc_{\text{AllParamFunctional}},
$$

$$
R_{\text{curv,ResidualFunctional}}\leq0.90R_{\text{Base}},
$$

$$
BadStepRate_{\text{ResidualFunctional}}\leq0.02.
$$

### H6：前向/反传/显存要和同参数 MLP 可比，必要时必须定制 kernel

H6 是 v9.2 的硬门槛。一个 basis architecture 不能只因为 conditioning、fit、overfit、gradcheck 通过就进入 full task；它还必须证明前向、反传、step time 和 peak memory 与同参数 MLP 可比。

首先构造同参数 MLP：

$$
\left|
\frac{Params_{\text{KAN}}-Params_{\text{MLP-match}}}
{Params_{\text{MLP-match}}}
\right|
\leq0.05.
$$

然后对 KAN 与 MLP-match 进行同 batch、同 dtype、同 warmup/reps、同 optimizer-state 范围的 microbench。H6 成立标准为：

$$
T_{\text{forward,KAN}}\leq1.25T_{\text{forward,MLP-match}},
$$

$$
T_{\text{backward,KAN}}\leq1.50T_{\text{backward,MLP-match}},
$$

$$
T_{\text{step,KAN}}\leq1.50T_{\text{step,MLP-match}},
$$

$$
M_{\text{peak,KAN}}\leq1.05M_{\text{peak,MLP-match}}.
$$

此外，official candidate 中不得 materialize：

$$
B\times d_{in}\times d_{out}\times K.
$$

如果候选必须构造上述 dense edge tensor，则：

```text
materializes_dense_edge_tensor = 1
kernel_native_pass = 0
official_eligible = 0
```

如果某个 basis 已经通过 conditioning / fit / overfit / gradcheck，并且显示出 task 或 geometry 信号，但 PyTorch/einsum 实现下无法满足 forward/backward/memory gate，则它进入：

```text
KernelizationRequired
```

此时必须先实现定制 kernel 或 fused path，包括：

```text
torch.compile fused basis eval + residual projection
Triton fused forward
Triton fused backward
active-k fused gather/scatter
foreach/manual update fusion
```

然后重新测：

```text
GradRelErrMax
GradCosMin
forward_time_ms
backward_time_ms
step_time_ms
peak_memory_MB
kernel_count_total
small_kernel_count
unknown_time_fraction
```

只有 kernelized path 通过 H6，candidate 才能进入 AdamW trainability 和 external full task。

---

## 8. v9.2 候选设计

### 8.1 Feature source candidates

```text
S0-Raw:
  raw flattened input，只作为 v9.1 对照，不期望通过。

S1-StdClip:
  per-feature mean/std normalization + clipping。

S2-FanInNorm:
  fan-in normalized input，按 feature variance 和 layer fan-in 归一化。

S3-EdgeAffineNorm:
  edge-owned affine normalized input：
  \tilde{x}_{i}=\alpha_i(x_i-\mu_i)/(\sigma_i+\epsilon)+\beta_i。
  \alpha_i,\beta_i 属于 edge system / source normalization 参数。

S4-PatchPool:
  对 vision 输入做固定 patch pooling / local average / local contrast normalization 后再 flatten。
  这是进入 KANConv 前的 FC-compatible diagnostic。

S5-PrevLayerNormed:
  使用上一层 FullEdge 输出的 per-channel normalized source，用于多层模型。
```

### 8.2 Basis candidates

第一批 official-priority：

```text
B0-IdentityOnly:
  sanity baseline，验证 identity edge base 是否能等价 MLP-like linear path。

B1-Legendre3Residual:
  normalized Legendre residual basis。

B2-Chebyshev3Residual:
  normalized Chebyshev residual basis。

B3-PiecewiseLinearResidual:
  non-recursive active-k hat basis，active basis = 2。

B4-SharedRBF4Residual:
  low-rank/shared RBF residual，centers fixed by quantiles。

B5-BoundedRational2Residual:
  bounded rational diagnostic，必须先过 stability gate。
```

第二批 diagnostic，不进入第一轮 official：

```text
B6-FourierLowFreq:
  low-frequency Fourier diagnostic for signal / periodic targets。

B7-HaarWaveletPiecewise:
  wavelet diagnostic for patch / local contrast tasks。

B8-CubicSplineActiveK:
  non-recursive cubic active-k spline，symbolic diagnostic。
```

### 8.3 Parameterization candidates

```text
P0-DenseFullEdge:
  只作为 negative control，不进入 official efficiency route。

P1-IdentityPlusLowRankResidual:
  主线。Y = XW0 + R V^T。

P2-IdentityPlusGroupedResidual:
  output channels 分组，降低 residual mixing cost。

P3-SharedBasisLowRank:
  basis 先按 input/rank 计算，再 output projection。

P4-ActiveKSharedResidual:
  piecewise/RBF active-k 版本。
```

### 8.4 Functional update candidates

```text
F0-NoFunctional:
  AdamW-only，先证明 basis 可训。

F1-ResidualOnlyFunctional:
  只 smoothing nonlinear residual coefficient。

F2-HighCurvatureOnlyFunctional:
  只作用 curvature top-p 的 residual channels。

F3-AllParamFunctional:
  negative control，验证是否伤 task。

F4-NoOpMatchedOverhead:
  causality control。

F5-RandomDirection:
  causality control。
```

---

## 9. v9.2 实验阶段

## Wave 0：代码与合同审计

### P0：Candidate registry and contract audit

目标是确认 v9.2 不再混入旧 transitional route，不再使用 label smoothing / teacher / loss modification，不再把 dense materialization 误记为 official kernel-native。

必须记录：

```text
candidate_id
source_id
basis_id
parameterization_id
functional_id
model_level
loss_type
label_smoothing
external_teacher_used
self_teacher_used
geometry_loss_used
sampler_changed
class_weight_used
cpu_offload_used
uses_loss_backward
materializes_dense_edge_tensor
official_eligible
```

通过标准：

```text
loss_type = CE
label_smoothing = 0
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
uses_loss_backward = 0
materializes_dense_edge_tensor = 0 for official candidate
```

可视化：

```text
p0_contract_heatmap.svg
p0_candidate_lattice.svg
p0_dense_materialization_violation_matrix.svg
```

---

## Wave 1：Feature-source × Basis conditioning

### P1：basis-source conditioning audit

目标是验证 v9.1 的 raw-input 失败是否可以通过 source/norm 修复。

实验矩阵：

```text
Sources:
  S0,S1,S2,S3,S4

Basis:
  B0,B1,B2,B3,B4,B5

Datasets:
  MNIST,Fashion-MNIST,KMNIST

Seeds:
  0,1,2
```

必须记录：

```text
source_id
basis_id
dataset
seed
activation_mean
activation_std
activation_p01
activation_p99
basis_cov_condition
dead_basis_fraction
dominant_basis_fraction
basis_channel_corr_mean
grad_norm_mean
grad_norm_cv
slope_p95
curvature_p95
conditioning_pass
```

判断标准：

$$
\kappa(B^\top B)\leq10^4,
$$

$$
DeadBasisFraction\leq0.30,
$$

$$
DominantBasisFraction\leq0.70.
$$

如果没有任何 source-basis 通过，v9.2 停止 full task，进入 feature source redesign。

可视化：

```text
p1_condition_heatmap_source_basis_dataset.svg
p1_condition_number_by_source.svg
p1_dead_dominant_basis_scatter.svg
p1_activation_histogram_grid.svg
p1_slope_curvature_boxplot.svg
```

### P2：basis one-layer fit diagnostic

目标是确认 basis 的近似能力，而不是直接跑 classifier。

Targets：

```text
identity
silu
quadratic
piecewise ramp
local bump
low-frequency sine
transitional hidden projection
symbolic polynomial
```

必须记录：

```text
source_id
basis_id
target_type
fit_MSE
fit_R2
coeff_norm
curvature
condition_number
train_time
fit_pass
```

判断标准：

对 identity / silu / quadratic：

$$
R^2\geq0.95.
$$

对 local bump：

$$
MSE_{\text{RBF or piecewise}}<MSE_{\text{poly}}.
$$

可视化：

```text
p2_fit_r2_by_basis_target.svg
p2_example_fit_curves.svg
p2_fit_mse_vs_condition.svg
p2_coeff_norm_vs_fit.svg
```

---

## Wave 2：Hybrid residual FullEdge implementation

### P3：manual gradcheck and overfit smoke

目标是让通过 P1/P2 的 candidate 进入 minimal model implementation。

必须记录：

```text
candidate_id
source_id
basis_id
parameterization_id
GradRelErrMax
GradCosMin
OutputAbsDiffMax
DxAbsDiffMax
ParamGradAbsDiffMax
GradPass
TrainAcc512
TrainLossFinal
OverfitPass
```

通过标准：

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999,
$$

$$
TrainAcc_{512}\geq0.98.
$$

可视化：

```text
p3_gradcheck_lollipop.svg
p3_overfit_curve.svg
p3_grad_vs_overfit_matrix.svg
```

### P4：kernel-native feasibility gate：forward / backward / memory vs 同参数 MLP

目标是把“前向和反传是否与 MLP 可比”提升为硬 gate。P4 不只是 materialization-free microbench，而是每个 basis architecture 进入 AdamW trainability 前的资格审查。P4 不过，P5/P6/P7 全部不得打开。

必须记录：

```text
candidate_id
source_id
basis_id
parameterization_id
matched_mlp_id
params_kan
params_mlp_match
params_ratio_vs_mlp_match
materializes_dense_edge_tensor
materialized_tensor_shape
materialized_MB
forward_time_ms_kan
forward_time_ms_mlp_match
backward_time_ms_kan
backward_time_ms_mlp_match
step_time_ms_kan
step_time_ms_mlp_match
peak_memory_MB_kan
peak_memory_MB_mlp_match
activation_cache_MB_kan
activation_cache_MB_mlp_match
optimizer_state_MB_kan
optimizer_state_MB_mlp_match
functional_update_time_ms
forward_ratio_vs_mlp_match
backward_ratio_vs_mlp_match
step_ratio_vs_mlp_match
memory_ratio_vs_mlp_match
forward_FLOPs_ratio
backward_FLOPs_ratio
kernel_count_total
small_kernel_count
basis_eval_time
residual_projection_time
output_projection_time
kernelization_required
kernel_native_pass
```

通过标准：

$$
materializes\_dense\_edge\_tensor=0,
$$

$$
\left|
\frac{Params_{\text{KAN}}-Params_{\text{MLP-match}}}
{Params_{\text{MLP-match}}}
\right|
\leq0.05,
$$

$$
T_{\text{forward,KAN}}\leq1.25T_{\text{forward,MLP-match}},
$$

$$
T_{\text{backward,KAN}}\leq1.50T_{\text{backward,MLP-match}},
$$

$$
T_{\text{step,KAN}}\leq1.50T_{\text{step,MLP-match}},
$$

$$
M_{\text{peak,KAN}}\leq1.05M_{\text{peak,MLP-match}},
$$

$$
ForwardFLOPsRatio\leq1.05,
$$

$$
BackwardFLOPsRatio\leq1.50.
$$

如果 candidate 的 task / fit / geometry 信号存在，但 P4 失败，则 route 不允许继续 full task，而是写为：

```text
KernelizationRequired
```

并在下一轮先实现定制 kernel，再重新执行 P3/P4。

可视化：

```text
p4_microbench_time_breakdown.svg
p4_memory_ratio_by_candidate.svg
p4_compute_vs_accuracy_proxy.svg
p4_materialization_waterfall.svg
```

---

## Wave 3：AdamW trainability

### P5：AdamW-only full task smoke

目标是先证明 PureKAN-AdamW 能训练。Functional update 不能救一个 AdamW 下完全不成立的 basis。

设置：

```text
Datasets = MNIST,Fashion-MNIST,KMNIST
Candidates = P1/P2/P3/P4 survivors
Seeds = 0,1,2
Training = CE-only, AdamW-equivalent manual update
Functional = off
```

必须记录：

```text
candidate_id
dataset
seed
train_loss_curve
val_acc
test_acc
val_loss
test_loss
ECE
NLL
margin_p10
hard_sample_acc
params_ratio
forward_FLOPs_ratio
backward_FLOPs_ratio
step_ratio
memory_ratio
```

判断标准：

Minimum trainability:

$$
Acc_{\text{PureKAN-AdamW}}\geq Acc_{\text{MLP-AdamW}}-0.01.
$$

Strong trainability:

$$
Acc_{\text{PureKAN-AdamW}}\geq Acc_{\text{MLP-AdamW}}.
$$

可视化：

```text
p5_train_loss_curve.svg
p5_val_acc_curve.svg
p5_dataset_gap_bar.svg
p5_task_system_pareto.svg
p5_ece_nll_bar.svg
```

---

## Wave 4：Residual-subspace functional update

### P6：functional update causality and task preservation

目标是验证 functional update 对 justified basis 是否有独特价值。

必须跑：

```text
Base AdamW
ResidualOnlyFunctional
HighCurvatureOnlyFunctional
AllParamFunctional
NoOpMatchedOverhead
RandomDirection
```

必须记录：

```text
candidate_id
functional_mode
dataset
seed
test_acc
delta_vs_base
curvature_ratio
jacobian_norm_ratio
local_lipschitz_ratio
ECE_delta
NLL_delta
robustness_proxy
bad_step_rate
holdout_descent_ratio
functional_update_time_ratio
functional_events
```

判断标准：

Causality pass：

$$
R_{\text{curv,Functional}}<R_{\text{curv,NoOp}},
$$

$$
R_{\text{curv,Functional}}<R_{\text{curv,Random}},
$$

$$
Acc_{\text{Functional}}\geq Acc_{\text{NoOp}}-0.002.
$$

Functional useful：

$$
R_{\text{curv,Functional}}\leq0.90R_{\text{Base}},
$$

$$
Acc_{\text{Functional}}\geq Acc_{\text{Base}}-0.005.
$$

可视化：

```text
p6_functional_curvature_bar.svg
p6_task_geometry_pareto.svg
p6_noop_random_control_matrix.svg
p6_bad_step_rate_timeline.svg
p6_functional_event_timeline.svg
```

---

## Wave 5：External fair validation

### P7：KANbeFair comparison

目标是重新与 KB-MLP / KB-KAN 比较，不能继承 transitional route 结果。

必须跑：

```text
KB-MLP
KB-KAN
Best HybridResidual FullEdge AdamW
Best HybridResidual FullEdge Functional
Clean transitional reference
NoOp
RandomFunc
```

任务：

```text
MNIST
Fashion-MNIST
KMNIST
symbolic tasks if runnable
tabular tasks if runnable
```

必须记录：

```text
task_family
dataset
candidate_id
seed
val_metric
test_metric
delta_vs_KB_MLP
delta_vs_KB_KAN
params_ratio
forward_FLOPs_ratio
backward_FLOPs_ratio
step_ratio
memory_ratio
kernel_time_ratio
ECE
NLL
curvature_ratio
functional_causality_pass
external_fair_pass
```

判断标准：

FullEdge external fair pass：

$$
Acc_{\text{FullEdge}}\geq Acc_{\text{KB-MLP}},
$$

$$
ParamsRatio\leq1.05,
$$

$$
ForwardFLOPsRatio\leq1.05,
$$

$$
BackwardFLOPsRatio\leq1.50,
$$

$$
StepRatio\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

可视化：

```text
p7_external_fair_scorecard.svg
p7_acc_vs_params.svg
p7_acc_vs_flops.svg
p7_acc_vs_step_memory.svg
p7_boundary_by_task.svg
```

---

## Wave 6：Boundary and go/no-go decision

### P8：boundary audit

目标是诚实判断结果边界。

可能 route：

```text
R1-HybridResidualFullEdgeSuccess:
  justified basis + FullEdge + functional update 通过 external fair。

R2-HybridResidualFullEdgeNearPass:
  task 接近 MLP，但 compute 或 memory 未完全过。

R3-BasisSourceConditioningFail:
  所有 source-basis conditioning 失败。

R4-AdamWTrainabilityFail:
  conditioning 和 overfit 通过，但 AdamW full task 训不动。

R5-FunctionalGeometryOnly:
  functional update 改善 geometry，但 task 没有收益。

R6-ComputeFail:
  task 有信号，但 materialization-free implementation 不够快/省。

R7-TransitionalStillBest:
  clean transitional 仍最强，FullEdge 未能追上。

R8-TaskFamilyLimited:
  只在 symbolic / MNIST-like 上成立。
```

必须记录：

```text
route
best_candidate
best_source
best_basis
best_parameterization
best_functional_mode
primary_blocker
next_required_implementation
success_v92_basis_source
success_v92_full_edge_trainability
success_v92_functional_advantage
success_v92_external_fair
```

可视化：

```text
p8_route_decision_tree.svg
p8_failure_reason_stacked.svg
p8_boundary_matrix.svg
```

---

## 10. 必须落盘 artifacts

```text
run_manifest.json
candidate_registry_v92.csv
contract_audit_v92.csv
basis_source_conditioning.csv
basis_fit_diagnostics.csv
hybrid_residual_gradcheck.csv
hybrid_residual_overfit.csv
materialization_free_microbench.csv
kernel_native_feasibility_vs_mlp.csv
kernelization_required_candidates.csv
adamw_trainability_task.csv
adamw_trainability_trace.csv
functional_causality_v92.csv
functional_event_trace.csv
kanbefair_external_validation.csv
boundary_audit.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_dense_materialization_violation
F3_conditioning_fail
F4_basis_fit_fail
F5_grad_fail
F6_overfit_fail
F7_microbench_fail
F7a_kernel_native_forward_backward_memory_fail
F7b_kernelization_required
F8_adamw_trainability_fail
F9_functional_causality_fail
F10_external_fair_fail
F11_compute_fair_fail
F12_geometry_only
F13_fake_or_proxy_violation
F14_artifact_missing
```

---

## 11. 第一轮执行顺序

第一轮不要直接 full task。执行顺序固定为：

```text
Step 1:
  P0 contract and registry。

Step 2:
  P1 basis-source conditioning。
  如果没有 source-basis 通过，停止，不跑 full task。

Step 3:
  P2 one-layer fit。
  淘汰 fit 不通过的 basis。

Step 4:
  P3 gradcheck + small overfit。
  淘汰不能 overfit 512 samples 或 grad 不对的 candidate。

Step 5:
  P4 kernel-native feasibility gate。
  对每个 survivor 构造同参数 MLP-match，测 forward / backward / step / peak memory / kernel count。
  P4 不过的 candidate 不得进入 AdamW trainability；若 basis 有 task/geometry 信号，则标记为 KernelizationRequired，先做定制 kernel。

Step 6:
  P5 AdamW-only trainability。
  证明 expression / optimization 本身成立。

Step 7:
  P6 residual-subspace functional update。
  只在 AdamW trainability 成立后打开。

Step 8:
  P7 external fair validation。
  只对 P5/P6 survivors 打开。

Step 9:
  P8 route decision。
```

---

## 12. 最终建议

v9.2 的一句话策略是：

$$
\boxed{
\text{先修 basis 的输入源与归一化，再用 hybrid residual low-rank FullEdge 做 kernel-native 验证。}
}
$$

现在不应该继续：

```text
盲目增加 basis；
直接 full task；
直接上 recursive B-spline；
把 raw-input conditioning fail 的 basis 拿去写 official route；
把 clean transitional success 当终极 FullEdge success；
用 functional update 救 AdamW 下不可训的 basis。
```

现在应该做：

```text
1. feature-source conditioning；
2. normalized orthogonal residual basis；
3. piecewise/RBF/rational 作为 gated diagnostic；
4. identity edge base + low-rank nonlinear residual；
5. materialization-free forward/backward；
6. kernel-native feasibility gate：前向、反传、step、显存必须和同参数 MLP 可比，必要时先定制 kernel；
7. AdamW trainability first；
8. residual-subspace functional update second；
9. external fair validation last。
```

最终希望 v9.2 给出的不是“某个 basis 跑通了”，而是：

$$
\boxed{
\text{为什么这个 basis 适合这个输入分布，为什么它可训练，为什么它高效，为什么 functional update 在它上面有独特价值。}
}
