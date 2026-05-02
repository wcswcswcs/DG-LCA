# DG-KAN / DG-LCA 完整实验计划

> 版本：v1.0  
> 对象：以 **KAN-like full hidden-state architecture** 为主线的 DG-LCA 实验计划  
> 定位：实验执行方案，不涉及具体代码实现  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`  
> 核心目标：围绕 KAN-like 主架构验证 **functional-space update + local credit transport + learned router + low-memory sufficient statistics** 是否成立。

---

## 0. 实验计划的重新定位

本实验计划将主线重新拉回原始 DG-LCA 设计中最重要的方向：**KAN-like primitive 不是辅助模块，而是主实验架构**。本计划不以 adapter-only 或 LoRA 子空间作为主线，也不把 KAN-like 仅当成 toy 模块。主目标是构造并验证一个完整的 **DG-KAN** 系统：网络的隐藏状态仍然是 full hidden-space，但每个 block 的主要可学习变换由 KAN-like edge functions 构成。

本计划要验证的核心命题是：

$$
\boxed{
\text{DG-KAN}
=
\text{full hidden-state KAN-like blocks}
+
\text{analytic / learned local credit transport}
+
\text{functional-space update}
+
\text{low-memory sufficient statistics}
}
$$

这与只在 adapter 子空间中做局部训练不同。adapter / LoRA / bottleneck 模块可以作为对照或后续扩展，但不是本计划的主线。主线是：

$$
\hat g_k
=
C_k^\phi(h_k,h_{k+1})[g_{k+1}]
\approx
J_k^\top g_{k+1}
$$

其中 $g_k,g_{k+1}$ 是完整 hidden state 上的 credit，而不是 adapter 投影后的 credit。

同时，DG-KAN 的关键优势来自 KAN-like primitive 的函数形式。对于一条边函数：

$$
\phi_{ji}:\mathbb R\to\mathbb R
$$

我们不仅可以写出它的 local adjoint rule，还可以在函数空间中定义更新：

$$
a
\leftarrow
 a-
\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

这正是普通 MLP、CNN、Transformer block 不容易直接做到的地方。

---

## 1. 总体研究问题

本实验计划围绕六个研究问题展开。每个问题都对应后面的一个实验阶段。

### 1.1 DG-KAN 架构本身是否可训练？

在正式验证 local credit assignment 之前，必须先证明 DG-KAN 作为一个网络架构可以稳定训练。也就是说，残差形式的 KAN-like block 在标准 BP 下应当能够达到合理的训练和验证性能。如果 DG-KAN 本身训练不稳定，那么后续 learned router、functional update、low-memory training 都无法单独归因。

### 1.2 Functional-space update 是否优于普通 coefficient update？

KAN-like block 的最大优势是每条边是函数。普通训练更新的是 basis coefficient：

$$
a\leftarrow a-\eta\nabla_a\mathcal L
$$

而 DG-KAN 的核心实验要验证：如果用 Sobolev / RKHS metric 对函数空间进行预条件，是否能带来更稳定的训练：

$$
a\leftarrow a-\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

成功标准不能只看最后 accuracy，还要看函数斜率、曲率、Jacobian 谱、credit amplification、predicted descent 与 actual descent 是否一致。

### 1.3 KAN-like primitive 的 analytic adjoint 是否正确、稳定、低成本？

对于 KAN-like 层：

$$
y_j=\sum_i\phi_{ji}(x_i)
$$

局部 VJP 可以写成：

$$
g_{x_i}=\sum_j g_{y_j}\phi'_{ji}(x_i)
$$

这给 DG-LCA 提供了一个非常干净的 primitive-adjoint 设计场景。实验必须先验证 analytic KAN adjoint 与 autograd VJP 完全一致，然后再讨论 learned router 是否有意义。

### 1.4 Learned local credit router 是否能逼近 KAN adjoint？

如果 analytic KAN adjoint 已经可用，learned router 的意义不是取代数学公式，而是测试一个更一般的问题：局部 VJP 能否被条件线性 router 蒸馏、压缩、加速或低显存化。router 的目标仍然是完整 hidden-space credit：

$$
\hat g_x=C^\phi(x,y)[g_y]\approx \Phi'(x)^\top g_y
$$

其中 $\hat g_x\in\mathbb R^d$，不是投影到低维 adapter 子空间。

### 1.5 DG-KAN joint training 是否能同时利用 local credit 和 functional update？

单独验证 functional update 或 local router 还不够。最终必须组合成训练流程：前向只保存 block interface，顶部计算任务 credit，然后用 analytic adjoint 或 learned router 逐层传 credit，每个 block 使用 functional-space update 更新 edge functions。

核心比较是：

| 方法 | credit 来源 | update 方式 |
|---|---|---|
| BP-KAN-Adam | full BP | coefficient Adam |
| BP-KAN-Sobolev | full BP | Sobolev / RKHS update |
| AnalyticAdj-KAN-Sobolev | analytic local adjoint | Sobolev / RKHS update |
| Router-KAN-Sobolev | learned router | Sobolev / RKHS update |

### 1.6 Low-memory sufficient statistics 是否真的成立？

DG-KAN 的低显存 claim 不应只来自“少保存 activation”。对于 KAN-like functional update，理论上边函数更新可以依赖充分统计量：

$$
\Delta\phi_{ji}(t)
=
-
\eta
\frac{1}{B}
\sum_{n=1}^{B}
 g_j^{(n)}K(t,x_i^{(n)})
$$

或者 basis 形式：

$$
\nabla_{a_{jim}}\mathcal L
=
\frac{1}{B}
\sum_{n=1}^{B}
 g_j^{(n)}B_m(x_i^{(n)})
$$

因此实验必须拆分比较 full BP、checkpointing、analytic local adjoint、sufficient statistics training、learned router training 的显存和计算成本。

---

## 2. 主实验架构：DG-KAN

### 2.1 Core KAN-like primitive

最基础的 KAN-like 层定义为：

$$
y_j=\sum_{i=1}^{d}\phi_{ji}(x_i),\quad j=1,\dots,d
$$

其中：

$$
x,y\in\mathbb R^d
$$

每条边函数写成 basis 展开：

$$
\phi_{ji}(t)=\sum_{m=1}^{M}a_{jim}B_m(t)
$$

其中 $B_m(t)$ 可以是 B-spline、RBF、piecewise linear basis 或其他一维 basis。主实验优先使用 B-spline 或 RBF，因为它们容易定义 Sobolev / RKHS metric。

这不是 adapter，也不是低维投影。对于 dense DG-KAN，每个输出维度 $y_j$ 都由所有输入维度 $x_i$ 的一维 edge functions 聚合而来，因此它处理的是完整 hidden state。

### 2.2 Residual DG-KAN block

主 block 使用 residual 形式：

$$
h_{k+1}=h_k+\alpha_k\,\operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

其中：

$$
\operatorname{KAN}_k(x)_j=\sum_i\phi_{k,ji}(x_i)
$$

$\alpha_k$ 是 residual scale，用于控制单层扰动幅度。LayerNorm 的作用是稳定 edge functions 的输入范围，避免 spline / kernel basis 在训练中频繁落到未覆盖区域。

在小维度实验中使用 dense KAN：

$$
i=1,\dots,d
$$

在中等维度实验中使用 grouped KAN：

$$
\operatorname{KAN}_k(x)_j=\sum_{i\in\mathcal G(j)}\phi_{k,ji}(x_i)
$$

其中 $\mathcal G(j)$ 是与输出维度 $j$ 相关的输入 group。Grouped KAN 是计算结构化，不是 adapter 子空间，因为它仍然输出完整 hidden state $h_{k+1}\in\mathbb R^d$，也仍然对完整 hidden credit 进行 routing。

### 2.3 模型族

本计划中所有实验围绕以下三个模型族展开。

#### DG-KAN-Small

用于函数拟合、Two Moons、MNIST 初步实验：

$$
d\in\{16,32,64\},\quad K\in\{2,4,8\},\quad M\in\{8,16\}
$$

使用 dense KAN blocks。

#### DG-KAN-Medium

用于 Fashion-MNIST 和 CIFAR-10 small：

$$
d\in\{128,256\},\quad K\in\{4,8\},\quad M\in\{8,16,32\}
$$

使用 grouped KAN 或 sparse KAN connectivity，保持完整 hidden-state 输出。

#### ConvStem-DG-KAN

用于图像任务。输入图像先通过轻量 conv stem 或 patch embedding 得到 hidden vector：

$$
h_0=\operatorname{Stem}(x)
$$

然后进入 DG-KAN blocks：

$$
h_{k+1}=h_k+\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

最后接分类 head：

$$
z=W_{head}h_K
$$

这个结构用于检验 DG-KAN 是否能超出 toy / tabular 场景。

---

## 3. 全局 W&B 记录规范

所有实验必须使用统一的 W&B config 和 metrics，否则后续无法做跨方法比较。每个 run 至少使用 $5$ 个 seed；资源不足时可用 $3$ 个 seed 做探索，但正式结果必须报告 $5$ seed 的 mean、std 和 worst-case。

### 3.1 必须记录的 config

每个 run 必须记录以下字段。

| config key | 含义 | 示例 |
|---|---|---|
| `exp/name` | 实验名称 | `gateB_mnist_dgkan_sobolev` |
| `exp/stage` | 实验阶段 | `A_baseline`, `B_functional`, `C_adjoint`, `D_joint`, `E_memory` |
| `exp/seed` | 随机种子 | `0`, `1`, `2`, `3`, `4` |
| `data/name` | 数据集 | `MNIST`, `FashionMNIST`, `CIFAR10_small` |
| `data/train_size` | 训练集大小 | `12000` |
| `data/val_size` | 验证集大小 | `2000` |
| `data/test_size` | 测试集大小 | `2000` |
| `model/family` | 模型族 | `DG-KAN-Small`, `DG-KAN-Medium`, `ConvStem-DG-KAN` |
| `model/depth` | block 数 | `2`, `4`, `8`, `12` |
| `model/hidden_dim` | hidden dimension | `32`, `64`, `128`, `256` |
| `model/kan_connectivity` | KAN 连接方式 | `dense`, `grouped`, `sparse` |
| `model/group_size` | group 大小 | `16`, `32`, `64` |
| `kan/basis_type` | basis 类型 | `bspline`, `rbf`, `piecewise_linear` |
| `kan/basis_count` | basis 数量 $M$ | `8`, `16`, `32` |
| `kan/grid_range` | basis 覆盖范围 | `[-3,3]` |
| `kan/residual_alpha_init` | $\alpha_k$ 初始值 | `0.1`, `0.5`, `1.0` |
| `update/type` | 更新方式 | `adam`, `adamw`, `sobolev`, `rkhs`, `diag_precond` |
| `update/damping_rho` | damping 系数 | `1e-4` |
| `update/sobolev_alpha` | 一阶 Sobolev 权重 | `1e-3` |
| `update/sobolev_beta` | 二阶 Sobolev 权重 | `1e-4` |
| `credit/type` | credit 来源 | `full_bp`, `analytic_kan_adjoint`, `learned_router` |
| `router/type` | router 类型 | `none`, `diagonal`, `diag_lowrank`, `derivative_compressed` |
| `router/rank` | router rank | `0`, `4`, `8`, `16`, `32` |
| `memory/mode` | memory 模式 | `full_bp`, `checkpoint`, `interface_stats`, `router_stats` |
| `optim/name` | optimizer | `Adam`, `AdamW`, `PrecondSGD` |
| `optim/lr` | 学习率 | `3e-4` |
| `train/max_epochs` | epoch 数 | `5`, `20`, `50` |
| `train/batch_size` | batch size | `64`, `128`, `256` |
| `resource/precision` | 精度 | `fp32`, `bf16` |
| `resource/checkpointing` | 是否 checkpointing | `true`, `false` |

### 3.2 必须记录的通用 metrics

以下 metrics 所有阶段都要记录。

| metric key | 频率 | 含义 |
|---|---:|---|
| `train/loss` | step | 训练 loss |
| `train/acc` | step 或 epoch | 训练 accuracy |
| `val/loss` | epoch | 验证 loss |
| `val/acc` | epoch | 验证 accuracy |
| `test/loss` | final | 测试 loss |
| `test/acc` | final | 测试 accuracy |
| `train/grad_norm` | step | 参数梯度范数 |
| `train/update_norm` | step | 参数更新范数 |
| `train/lr` | step | 学习率 |
| `conv/loss_auc` | final | loss 曲线面积 |
| `conv/steps_to_target_val_loss` | final | 达到目标 val loss 的 step |
| `conv/steps_to_target_val_acc` | final | 达到目标 val acc 的 step |
| `conv/time_to_target_sec` | final | 达到目标的 wall-clock time |
| `conv/plateau_step` | final | plateau step |
| `stability/loss_spike_count` | epoch | loss spike 数 |
| `stability/nan_or_inf_count` | epoch | NaN / Inf 数 |
| `stability/diverged` | final | 是否发散 |
| `perf/step_time_ms` | step | 每步耗时 |
| `perf/samples_per_sec` | step | 吞吐 |
| `memory/peak_allocated_mb` | epoch / final | peak allocated memory |
| `memory/peak_reserved_mb` | epoch / final | peak reserved memory |

### 3.3 收敛性判定

所有方法都必须进行三类比较。

第一，step-matched comparison。所有方法训练相同 step 数，比较 final loss、final accuracy 和 loss AUC。

第二，wall-clock-matched comparison。所有方法使用相同训练时间，比较验证集性能。这用于判断 functional update、analytic adjoint 或 router 的额外开销是否抵消了优化收益。

第三，memory-budget-matched comparison。设定固定显存预算 $M_{max}$，比较不同方法在相同显存下可以使用的 batch size、模型深度和性能。

Plateau 判定使用最近窗口 $W$ 内的验证 loss 改善：

$$
\Delta_W
=
\frac{\mathcal L_{t-W}^{val}-\mathcal L_t^{val}}
{|\mathcal L_{t-W}^{val}|+\epsilon}
$$

若连续 $P$ 个窗口满足：

$$
\Delta_W<\tau
$$

则认为进入 plateau。默认：

$$
W=500,\quad P=3,\quad \tau=0.005
$$

---

## 4. Stage A：DG-KAN 架构与 BP baseline

### 4.1 目标

Stage A 的目标是建立 DG-KAN 的可靠 baseline。此阶段使用标准 BP 和普通 coefficient optimizer，先证明 DG-KAN 架构本身可训练，并且与参数量或 FLOPs 匹配的 MLP / standard KAN baseline 相比不明显失效。

如果 Stage A 失败，后续所有 DG-LCA claim 都没有意义，因为训练不稳定可能来自架构本身，而不是 credit assignment 或 functional update。

### 4.2 实验设置

第一组使用函数拟合和 toy classification，用于快速检查 edge functions、LayerNorm、residual scale 和 basis 范围是否合理。数据集包括：

| 数据集 | 任务 | 目标 |
|---|---|---|
| $y=\sin x$ | 1D regression | 检查单调平滑函数学习 |
| $y=\sin x+0.3\sin 5x$ | 1D regression | 检查中频成分 |
| $y=\sin x_1+\cos x_2$ | 2D regression | 检查多输入函数 |
| Two Moons | classification | 非线性分类 sanity check |
| Circles | classification | 环形边界 sanity check |

第二组使用小型真实数据：MNIST、Fashion-MNIST 和 CIFAR-10 small。CIFAR-10 small 建议先使用轻量 conv stem 将图像映射为 $d$ 维 hidden vector，再接 DG-KAN blocks。

模型配置从小到大推进：

| 配置 | hidden dim | depth | KAN connectivity | basis count |
|---|---:|---:|---|---:|
| DG-KAN-S1 | 32 | 2 | dense | 8 |
| DG-KAN-S2 | 64 | 4 | dense | 8 / 16 |
| DG-KAN-M1 | 128 | 4 | grouped | 8 / 16 |
| DG-KAN-M2 | 256 | 8 | grouped | 16 / 32 |

Stage A 不启用 learned router，不启用 local functional update。所有方法都用标准 BP credit。

### 4.3 Baseline

Stage A 至少比较以下 baseline。

| 方法 | 说明 |
|---|---|
| MLP-Adam | 参数量匹配 MLP，Adam 训练 |
| MLP-AdamW | 参数量匹配 MLP，AdamW 训练 |
| Residual-MLP | 使用 residual block 的 MLP |
| KAN-Coeff-Adam | KAN-like block，coefficient Adam |
| KAN-Coeff-AdamW | KAN-like block，coefficient AdamW |
| DG-KAN-Coeff-Adam | residual DG-KAN，coefficient Adam |
| DG-KAN-Coeff-AdamW | residual DG-KAN，coefficient AdamW |

为了公平，至少提供两种匹配方式：parameter-matched 和 compute-matched。parameter-matched 保证参数量接近；compute-matched 保证每步 FLOPs 接近。

### 4.4 W&B 指标

Stage A 除通用指标外，需要记录 DG-KAN 架构稳定性。

| metric key | 含义 |
|---|---|
| `dgkan/layer_{k}/activation_norm` | 每层 activation norm |
| `dgkan/layer_{k}/residual_update_norm` | KAN residual branch 的更新范数 |
| `dgkan/layer_{k}/residual_ratio` | $\|\alpha KAN(LN(h))\|/\|h\|$ |
| `dgkan/layer_{k}/alpha` | residual scale |
| `dgkan/layer_{k}/output_variance` | 每层输出方差 |
| `dgkan/layer_{k}/ln_input_mean` | LN 后输入均值 |
| `dgkan/layer_{k}/ln_input_std` | LN 后输入标准差 |
| `kan/phi_value_mean` | edge function value 均值 |
| `kan/phi_value_std` | edge function value 标准差 |
| `kan/phi_prime_mean` | $|\phi'|$ 均值 |
| `kan/phi_prime_max` | 最大 $|\phi'|$ |
| `kan/phi_prime_p95` | $|\phi'|$ 95 分位 |
| `kan/phi_double_prime_energy_mean` | 平均 $\int |\phi''|^2$ |
| `kan/edge_function_norm` | edge function norm |
| `kan/edge_function_sobolev_norm` | Sobolev norm |
| `kan/edge_saturation_rate` | 输入落到 basis 边界外或饱和区比例 |

还要记录局部 Jacobian 与 credit amplification：

| metric key | 含义 |
|---|---|
| `jacobian/layer_{k}/sigma_max` | 局部 Jacobian 最大奇异值估计 |
| `jacobian/layer_{k}/sigma_min` | 局部 Jacobian 最小奇异值估计 |
| `jacobian/layer_{k}/condition` | 条件数 |
| `credit/layer_{k}/amplification_mean` | $\|g_k\|/\|g_{k+1}\|$ 均值 |
| `credit/layer_{k}/amplification_p95` | credit amplification 95 分位 |

### 4.5 可视化

W&B Dashboard 中 Stage A 至少包含以下图：

1. train / val loss curve，比较 MLP、KAN、DG-KAN。
2. test accuracy bar chart，按 dataset 和 model family 分组。
3. residual ratio over depth，观察 residual branch 是否过大。
4. $\phi'$ p95 over time，监控 edge derivatives 是否爆炸。
5. $\int |\phi''|^2$ over time，监控函数曲率。
6. Jacobian condition over training，观察谱是否稳定。
7. edge saturation rate over time，检查 basis 范围是否合适。

### 4.6 成功标准

Stage A 的弱成功标准是：DG-KAN 可以稳定训练，无 NaN / Inf，无长期 loss spike，且在 MNIST / Fashion-MNIST 上不明显差于 parameter-matched MLP。具体要求：

$$
\text{accuracy gap vs MLP}<5\%
$$

并且：

$$
\text{kan/phi\_prime\_p95}
$$

不长期爆炸，Jacobian condition 不持续失控。

中等成功标准是：DG-KAN 在 MNIST 和 Fashion-MNIST 上达到或超过 KAN-Coeff-Adam，并且 residual DG-KAN 比 non-residual KAN 更稳定。

强成功标准是：4 层和 8 层 DG-KAN 在 MNIST / Fashion-MNIST 上稳定，CIFAR-10 small 上达到合理性能，并且 credit amplification 的 p95 长期可控。

如果 Stage A 失败，应先调整 residual scale、basis range、LayerNorm、KAN connectivity 和学习率，不进入 Stage B。

---

## 5. Stage B：Functional-space update with true BP credit

### 5.1 目标

Stage B 是本计划最重要的实验之一。它使用真实 BP credit，不引入 learned router，只验证 functional-space update 是否真的优于普通 coefficient update。

原因是：如果 functional-space update 本身无收益，那么 KAN-like 作为 DG-LCA 主架构的优势会大幅减弱。相反，如果它能显著改善稳定性、收敛速度或 Jacobian 谱，则即使 learned router 暂时失败，DG-KAN 仍然有独立贡献。

### 5.2 方法定义

KAN edge function 写为：

$$
\phi(t)=\sum_{m=1}^{M}a_mB_m(t)
$$

普通 coefficient update 为：

$$
a\leftarrow a-\eta\nabla_a\mathcal L
$$

Sobolev functional update 为：

$$
a\leftarrow a-\eta(M_{Sob}+\rho I)^{-1}\nabla_a\mathcal L
$$

其中：

$$
M_{mn}^{Sob}
=
\int B_mB_n
+
\alpha\int B_m'B_n'
+
\beta\int B_m''B_n''
$$

RKHS functional update 为：

$$
a\leftarrow a-\eta(M_{RKHS}+\rho I)^{-1}\nabla_a\mathcal L
$$

其中：

$$
M_{mn}^{RKHS}=\langle B_m,B_n\rangle_{\mathcal H}
$$

所有 inverse 都必须使用 damping：

$$
M^{-1}\rightarrow (M+\rho I)^{-1}
$$

### 5.3 实验矩阵

Stage B 先在低维函数拟合上调通，再进入分类任务。推荐顺序如下。

| 顺序 | 数据集 | 模型 | 目的 |
|---:|---|---|---|
| 1 | $y=\sin x$ | 1-layer KAN | 验证平滑函数更新 |
| 2 | $y=\sin x+0.3\sin 5x$ | 1-layer KAN | 验证中频函数 |
| 3 | $y=\sin x_1+\cos x_2$ | 2D KAN | 验证多输入函数 |
| 4 | Two Moons | DG-KAN-Small | 验证分类 sanity |
| 5 | MNIST | DG-KAN-Small / Medium | 主实验 |
| 6 | Fashion-MNIST | DG-KAN-Medium | 稍难主实验 |
| 7 | CIFAR-10 small | ConvStem-DG-KAN | 扩展实验 |

每个数据集至少比较以下方法：

| 方法 | credit | update |
|---|---|---|
| KAN-Coeff-SGD | full BP | coefficient SGD |
| KAN-Coeff-Adam | full BP | coefficient Adam |
| KAN-Coeff-AdamW | full BP | coefficient AdamW |
| KAN-DiagPrecond | full BP | diagonal approximation of $M$ |
| KAN-Sobolev | full BP | Sobolev preconditioned update |
| KAN-RKHS | full BP | RKHS preconditioned update |

### 5.4 Hyperparameter sweep

Sobolev metric sweep：

$$
\alpha\in\{0,10^{-4},10^{-3},10^{-2},10^{-1}\}
$$

$$
\beta\in\{0,10^{-5},10^{-4},10^{-3},10^{-2}\}
$$

Damping sweep：

$$
\rho\in\{10^{-6},10^{-5},10^{-4},10^{-3},10^{-2}\}
$$

Learning rate sweep：

$$
\eta\in\{10^{-4},3\times 10^{-4},10^{-3},3\times 10^{-3},10^{-2}\}
$$

正式比较时不能只选最优单点，应报告每种方法的 best、mean、largest stable learning rate 和 seed std。

### 5.5 W&B 指标

除通用指标外，Stage B 必须记录以下指标。

| metric key | 含义 |
|---|---|
| `kan/phi_prime_max` | 最大 $|\phi'|$ |
| `kan/phi_prime_mean` | 平均 $|\phi'|$ |
| `kan/phi_prime_p95` | $|\phi'|$ 95 分位 |
| `kan/phi_double_prime_energy_mean` | 平均 $\int |\phi''|^2$ |
| `kan/phi_double_prime_energy_max` | 最大 $\int |\phi''|^2$ |
| `kan/sobolev_norm_mean` | 平均 Sobolev norm |
| `kan/sobolev_norm_max` | 最大 Sobolev norm |
| `kan/coefficient_norm` | coefficient norm |
| `kan/update_norm_coeff` | coefficient update norm |
| `kan/update_norm_function` | function-space update norm |
| `jacobian/sigma_max` | 局部 Jacobian 最大奇异值 |
| `jacobian/sigma_min` | 局部 Jacobian 最小奇异值 |
| `jacobian/condition` | Jacobian 条件数 |
| `credit/amplification_mean` | credit amplification 均值 |
| `credit/amplification_p95` | credit amplification 95 分位 |
| `metric/precond_condition` | $\kappa(M+\rho I)$ |
| `metric/precond_min_eig` | $M+\rho I$ 最小特征值 |
| `metric/precond_max_eig` | $M+\rho I$ 最大特征值 |

还必须记录 predicted descent 和 actual descent：

$$
\Delta\mathcal L_{pred}=D_F\mathcal L[\Delta F]
$$

$$
\Delta\mathcal L_{actual}=\mathcal L_{after}-\mathcal L_{before}
$$

对应 W&B metrics：

| metric key | 含义 |
|---|---|
| `descent/pred` | predicted descent |
| `descent/actual` | actual descent |
| `descent/ratio` | actual / predicted |
| `descent/sign_agreement` | 单步 sign 是否一致 |
| `descent/sign_agreement_rate` | 最近窗口 sign agreement rate |
| `descent/pred_negative_actual_positive_count` | 预测下降但实际上升次数 |

### 5.6 可视化

Stage B 的 dashboard 至少包含：

1. train / val loss curves：Adam、AdamW、Sobolev、RKHS 对比。
2. loss AUC bar chart。
3. steps-to-target bar chart。
4. largest stable learning rate heatmap。
5. $\phi'$ p95 over time。
6. $\int |\phi''|^2$ over time。
7. Jacobian condition over time。
8. predicted vs actual descent scatter。
9. damping $\rho$ vs validation loss heatmap。
10. Pareto plot：x 为 curvature，y 为 validation loss，颜色为方法。

### 5.7 成功标准

弱成功：functional update 相比 coefficient Adam / AdamW，validation loss 不显著更差，同时 $\phi'$、$\phi''$ 和 Jacobian condition 更稳定。

中等成功：至少两个数据集上满足以下任一条件：

$$
\text{steps-to-target reduction}>20\%
$$

或：

$$
\text{loss AUC reduction}>10\%
$$

或：

$$
\text{curvature reduction}>30\%
$$

并且：

$$
\text{descent sign agreement rate}>0.75
$$

强成功：在 MNIST / Fashion-MNIST / CIFAR-10 small 上，functional update 不低于 Adam 的 validation accuracy，同时收敛更快、最大稳定学习率更大、Jacobian spectrum 更稳定、$\phi'$ 与 $\phi''$ 更可控。

失败判定：若 functional update loss 不下降、明显过平滑、actual descent 经常与 predicted descent 方向相反，或者在合理 damping 下完全打不过 Adam，则 functional update 降级为辅助 preconditioner，不作为主贡献继续推进。

---

## 6. Stage C：KAN analytic adjoint 与 learned local router

### 6.1 目标

Stage C 验证 KAN-like block 的 local credit transport。它分成三步：首先证明 analytic adjoint 与 autograd VJP 一致；然后训练 learned router 逼近 analytic adjoint；最后用 adjoint identity 和 forward probe 做校准。

这一步仍然关注完整 hidden-space credit，而不是 adapter 投影。对于 KAN primitive：

$$
y_j=\sum_i\phi_{ji}(x_i)
$$

local VJP 是：

$$
g_{x_i}=\sum_jg_{y_j}\phi'_{ji}(x_i)
$$

如果使用 residual 和 LayerNorm，完整 block 的 VJP 还包括 residual identity path 和 LayerNorm 的局部导数。Stage C 需要分别报告 core KAN primitive 的 VJP fidelity 和 full residual block 的 VJP fidelity。

### 6.2 C1：Analytic KAN adjoint vs autograd VJP

首先固定一个训练好的 DG-KAN block，采样 $(x,y,g_y)$，比较：

$$
g_x^{analytic}=\Phi'(x)^\top g_y
$$

与 autograd 得到的：

$$
g_x^{autograd}=J^\top g_y
$$

对于 core KAN primitive，期望二者几乎完全一致。对于包含 LayerNorm / residual 的完整 block，analytic 版本需要包括这些组件的局部 adjoint；如果暂时不手写 LayerNorm adjoint，可先用 autograd 处理 LayerNorm，仅手写 core KAN adjoint，并单独报告这两类结果。

W&B 指标：

| metric key | 含义 |
|---|---|
| `credit/core_kan_vjp_cos_vs_autograd` | core KAN analytic VJP 与 autograd cosine |
| `credit/core_kan_vjp_relerr_vs_autograd` | core KAN analytic VJP relative error |
| `credit/full_block_vjp_cos_vs_autograd` | full block analytic / hybrid VJP cosine |
| `credit/full_block_vjp_relerr_vs_autograd` | full block relative error |
| `credit/analytic_vjp_norm_ratio` | analytic / autograd norm ratio |
| `compute/analytic_kan_vjp_time_ms` | analytic VJP 耗时 |
| `compute/autograd_vjp_time_ms` | autograd VJP 耗时 |
| `memory/analytic_vjp_temp_mb` | analytic VJP 临时显存 |
| `memory/autograd_vjp_temp_mb` | autograd VJP 临时显存 |

成功标准：

$$
\cos(g_x^{analytic},g_x^{autograd})>0.999
$$

$$
\operatorname{relerr}<10^{-4}
$$

对于 full residual block，如果使用 hybrid analytic / autograd LayerNorm，成功标准可以放宽为：

$$
\cos>0.995
$$

$$
\operatorname{relerr}<10^{-3}
$$

如果 C1 失败，不允许进入 learned router 训练，应先检查 edge derivative、basis derivative、batch dimension、residual path 和 LayerNorm adjoint。

### 6.3 C2：Learned KAN router

learned router 的目标是：

$$
\hat g_x=C^\phi(x,y)[g_y]\approx\Phi'(x)^\top g_y
$$

router 必须对 $g_y$ 保持线性。推荐比较以下 router：

| router | 形式 | 作用 |
|---|---|---|
| identity | $\hat g_x=g_y$ | 最弱 baseline |
| diagonal | $\hat g_x=D(x,y)\odot g_y$ | 只学缩放 |
| diagonal + low-rank | $D\odot g_y+U(V^\top g_y)$ | 主 learned router |
| derivative-compressed | 学 $\widehat{\Phi'}(x)$ 的压缩形式 | 更贴合 KAN primitive |
| static ridge | $\hat g_x=Wg_y$ | 检查是否无需 condition |
| nonlinear MLP | 非线性 $C(x,y,g_y)$ | 负对照，检查线性破坏 |
| analytic derivative | $\Phi'(x)^\top g_y$ | oracle baseline |

W&B fidelity 指标：

| metric key | 含义 |
|---|---|
| `router/train/loss_vjp_norm` | normalized VJP MSE |
| `router/val/cos_full` | full hidden-space cosine |
| `router/val/relerr_full` | full hidden-space relative error |
| `router/val/norm_ratio_full` | predicted / teacher norm |
| `router/val/cos_p05` | cosine 5 分位 |
| `router/val/relerr_p95` | relerr 95 分位 |
| `router/val/lin_residual` | 线性性 residual |
| `router/val/scale_error_a0_5` | $C(0.5g)$ vs $0.5C(g)$ |
| `router/val/scale_error_a2` | $C(2g)$ vs $2C(g)$ |
| `router/val/superposition_error` | $C(g_1+g_2)$ vs $C(g_1)+C(g_2)$ |

Adjoint identity residual：

$$
r_{adj}
=
\left|
\langle C(v),u\rangle
-
\langle v,Ju\rangle
\right|
$$

normalized version：

$$
r_{adj}^{norm}
=
\frac{
\left|
\langle C(v),u\rangle
-
\langle v,Ju\rangle
\right|
}{
|\langle v,Ju\rangle|+\epsilon
}
$$

对应指标：

| metric key | 含义 |
|---|---|
| `router/adjoint_residual_mean` | adjoint residual 均值 |
| `router/adjoint_residual_p95` | adjoint residual 95 分位 |
| `router/adjoint_residual_norm_mean` | normalized residual 均值 |
| `router/adjoint_residual_norm_p95` | normalized residual 95 分位 |

成本指标：

| metric key | 含义 |
|---|---|
| `compute/router_time_ms` | router forward 耗时 |
| `compute/analytic_vjp_time_ms` | analytic VJP 耗时 |
| `compute/autograd_vjp_time_ms` | autograd VJP 耗时 |
| `memory/router_mb` | router 显存 |
| `router/num_params` | router 参数量 |

### 6.4 C2 成功标准

弱成功：learned router 在 core KAN primitive 上达到：

$$
\cos>0.9
$$

$$
\operatorname{relerr}<0.35
$$

中等成功：learned router 在 full residual DG-KAN block 上达到：

$$
\cos>0.95
$$

$$
\operatorname{relerr}<0.2
$$

$$
0.8<\rho<1.2
$$

$$
r_{lin}<0.05
$$

强成功：learned router 达到 analytic adjoint 的 90% 以上 fidelity，同时满足：

$$
T_{router}<0.5T_{analytic\ VJP}
$$

或显存显著更低。如果 router 比 analytic VJP 更慢、更占显存，且 fidelity 不高，则 learned router 不应进入 joint training 主方法，只保留 analytic adjoint。

---

## 7. Stage D：DG-KAN joint training

### 7.1 目标

Stage D 将前面验证过的 functional-space update 和 local credit transport 组合起来，测试 DG-KAN 是否可以在训练中使用局部 credit 和函数空间更新完成端到端任务。

训练流程为：

1. 前向传播只长期保存 block interface：

$$
h_0,h_1,\dots,h_K
$$

2. 顶部计算任务 credit：

$$
g_K=\nabla_{h_K}\ell(h_K,y)
$$

3. 使用 analytic KAN adjoint 或 learned router 逐层传播 credit：

$$
\hat g_k=C_k(h_k,h_{k+1})[\hat g_{k+1}]
$$

或 analytic 形式：

$$
\hat g_{x_i}=\sum_j\hat g_{y_j}\phi'_{ji}(x_i)
$$

4. 每个 KAN block 使用 functional update 更新 edge functions：

$$
a_{ji}\leftarrow a_{ji}-\eta(M+\rho I)^{-1}\nabla_{a_{ji}}\mathcal L
$$

### 7.2 方法组合

Stage D 必须比较以下组合。重点不是只报一个 DG-KAN 方法，而是拆清 credit 来源和 update 方式。

| 方法 | credit 来源 | update 方式 | 目的 |
|---|---|---|---|
| BP-KAN-Adam | full BP | coefficient Adam | 标准 KAN baseline |
| BP-KAN-Sobolev | full BP | Sobolev / RKHS | 测 functional update 上限 |
| AnalyticAdj-KAN-Adam | analytic local adjoint | coefficient Adam | 测 analytic adjoint 替代 BP 的影响 |
| AnalyticAdj-KAN-Sobolev | analytic local adjoint | Sobolev / RKHS | 主方法之一 |
| Router-KAN-Adam | learned router | coefficient Adam | 分离 router 效果 |
| Router-KAN-Sobolev | learned router | Sobolev / RKHS | 完整 DG-KAN 方法 |
| SyntheticGrad-KAN | synthetic gradient | Adam / Sobolev | credit baseline |
| FeedbackAlign-KAN | feedback alignment | Adam / Sobolev | credit baseline |
| LocalLoss-KAN | local loss | Adam / Sobolev | local objective baseline |

### 7.3 数据集与模型

第一阶段使用 MNIST 和 Fashion-MNIST，因为它们足以暴露 credit routing 和 functional update 问题，同时计算成本可控。第二阶段再进入 CIFAR-10 small。

| 数据集 | 模型 | depth | hidden dim | 目的 |
|---|---|---:|---:|---|
| MNIST | DG-KAN-Small | 2 / 4 / 8 | 64 / 128 | 主实验 |
| Fashion-MNIST | DG-KAN-Small / Medium | 4 / 8 | 128 / 256 | 稍难主实验 |
| CIFAR-10 small | ConvStem-DG-KAN | 4 / 8 | 128 / 256 | 图像扩展 |

Stage D 中不应一开始做 12 层以上。只有 8 层稳定后再考虑更深模型。

### 7.4 W&B 指标

任务性能：

| metric key | 含义 |
|---|---|
| `train/loss` | train loss |
| `val/loss` | val loss |
| `test/loss` | test loss |
| `train/acc` | train acc |
| `val/acc` | val acc |
| `test/acc` | test acc |
| `compare/acc_gap_vs_bp_kan_adam` | 相对 BP-KAN-Adam 的 acc gap |
| `compare/loss_gap_vs_bp_kan_adam` | 相对 BP-KAN-Adam 的 loss gap |

Credit fidelity：

| metric key | 含义 |
|---|---|
| `credit/layer_{k}/cos_vs_bp` | 层 $k$ credit 与 BP cosine |
| `credit/layer_{k}/relerr_vs_bp` | 层 $k$ relative error |
| `credit/layer_{k}/norm_ratio_vs_bp` | norm ratio |
| `credit/global/mean_cos_vs_bp` | 全层平均 cosine |
| `credit/global/worst_cos_vs_bp` | 最差层 cosine |
| `credit/global/mean_relerr_vs_bp` | 全层平均 relerr |

Functional update quality：

| metric key | 含义 |
|---|---|
| `functional/pred_descent` | predicted descent |
| `functional/actual_descent` | actual descent |
| `functional/descent_sign_agreement_rate` | sign agreement rate |
| `functional/update_norm_function` | function-space update norm |
| `functional/update_norm_coeff` | coefficient update norm |
| `functional/projection_residual` | ideal function update 投影回参数化的 residual |

KAN stability：

| metric key | 含义 |
|---|---|
| `kan/phi_prime_max` | 最大 $|\phi'|$ |
| `kan/phi_prime_p95` | $|\phi'|$ 95 分位 |
| `kan/phi_double_prime_energy` | 曲率能量 |
| `kan/sobolev_norm` | Sobolev norm |
| `kan/edge_update_norm` | edge update norm |
| `kan/edge_saturation_rate` | edge saturation rate |

Sequential routing：

| metric key | 含义 |
|---|---|
| `routing/self_vs_teacher_drift` | self-routed 与 teacher-forced drift |
| `routing/error_amplification` | 误差放大因子 |
| `routing/credit_norm_growth` | credit norm 随深度增长率 |
| `routing/global/mean_cos_self` | self-routed 平均 cosine |
| `routing/global/worst_cos_self` | self-routed 最差层 cosine |

### 7.5 可视化

Stage D dashboard 至少包含：

1. train / val loss curves，按方法分组。
2. accuracy vs method bar chart，显示 mean ± std。
3. layerwise credit cosine heatmap。
4. self-routed drift over depth。
5. predicted vs actual descent scatter。
6. $\phi'$ p95 和 $\phi''$ energy over time。
7. memory vs accuracy Pareto plot。
8. wall-clock vs validation loss curve。
9. credit norm growth over depth。
10. failure table：记录所有 divergence、norm explosion、descent mismatch。

### 7.6 成功标准

弱成功：AnalyticAdj-KAN-Sobolev 能够不依赖 full BP 反向链完成稳定训练，性能接近 BP-KAN-Adam，并且 functional update 的稳定性指标优于 coefficient Adam。

中等成功：Router-KAN-Sobolev 在 MNIST / Fashion-MNIST 上满足：

$$
\text{accuracy gap vs BP-KAN-Adam}<3\%
$$

$$
\text{credit/global/mean\_cos\_vs\_bp}>0.8
$$

$$
\text{descent sign agreement rate}>0.7
$$

强成功：Router-KAN-Sobolev 在 8 层 DG-KAN 上满足：

$$
\text{accuracy gap}<2\%
$$

$$
\text{memory reduction vs full BP}>20\%
$$

$$
\text{wall-clock overhead}<1.5\times
$$

并且不需要每步 full BP audit。

失败判定：如果 AnalyticAdj-KAN-Sobolev 都明显差于 BP-KAN-Adam，则问题可能在 local training protocol 或 functional update，而不是 learned router。如果 AnalyticAdj-KAN-Sobolev 成功但 Router-KAN-Sobolev 失败，则问题主要在 router fidelity、distribution shift 或 sequential error accumulation。

---

## 8. Stage E：Low-memory sufficient statistics

### 8.1 目标

Stage E 验证 DG-KAN 是否真的具有低显存机制。重点不是只报告 peak memory，而是拆分完整 memory accounting，并确认 KAN-like functional update 是否可以依赖局部充分统计量完成。

对于 basis 形式：

$$
\phi_{ji}(t)=\sum_m a_{jim}B_m(t)
$$

edge coefficient gradient 为：

$$
\nabla_{a_{jim}}\mathcal L
=
\frac{1}{B}
\sum_{n=1}^{B}g_j^{(n)}B_m(x_i^{(n)})
$$

所以只要保存或累计：

$$
s_{jim}=\sum_n g_j^{(n)}B_m(x_i^{(n)})
$$

就可以完成局部函数更新。这是 DG-KAN 低显存机制的核心。

### 8.2 比较模式

Stage E 比较以下训练模式。

| 模式 | 保存内容 | 说明 |
|---|---|---|
| full BP | full activation graph | 标准 BP |
| checkpointing | checkpoints + recompute | 强 memory baseline |
| analytic KAN adjoint | interfaces + local recompute | 不保存 full graph |
| DG-KAN sufficient stats | interfaces + edge sufficient stats | 主低显存模式 |
| router DG-KAN | interfaces + router states + stats | learned router 模式 |

### 8.3 Memory accounting

必须拆分上传以下 W&B metrics。

| metric key | 含义 |
|---|---|
| `memory/params_mb` | 参数显存 |
| `memory/optimizer_mb` | optimizer state 显存 |
| `memory/full_activations_mb` | full activation graph 显存 |
| `memory/checkpoint_mb` | checkpoint 显存 |
| `memory/interfaces_mb` | block interface 显存 |
| `memory/kan_edge_stats_mb` | KAN sufficient statistics 显存 |
| `memory/router_param_mb` | router 参数显存 |
| `memory/router_activation_mb` | router activation 显存 |
| `memory/local_recompute_mb` | 局部重算临时显存 |
| `memory/audit_buffer_mb` | audit buffer 显存 |
| `memory/peak_allocated_mb` | peak allocated memory |
| `memory/peak_reserved_mb` | peak reserved memory |
| `memory/reduction_vs_full_bp_pct` | 相对 full BP 下降比例 |
| `memory/reduction_vs_checkpoint_pct` | 相对 checkpointing 下降比例 |

### 8.4 Compute accounting

必须记录：

| metric key | 含义 |
|---|---|
| `compute/forward_time_ms` | forward 耗时 |
| `compute/full_bp_backward_time_ms` | full BP backward 耗时 |
| `compute/analytic_adj_time_ms` | analytic adjoint 耗时 |
| `compute/router_time_ms` | router 耗时 |
| `compute/functional_update_time_ms` | functional update 耗时 |
| `compute/recompute_time_ms` | local recompute 耗时 |
| `compute/total_step_time_ms` | 总 step time |
| `compute/overhead_vs_full_bp_pct` | 相对 full BP overhead |
| `compute/overhead_vs_checkpoint_pct` | 相对 checkpoint overhead |

### 8.5 成功标准

弱成功：

$$
M_{DG-KAN}<M_{full\ BP}
$$

且性能 gap 小于 $5\%$。

中等成功：

$$
M_{DG-KAN}<M_{checkpoint}
$$

且：

$$
\text{accuracy gap}<3\%
$$

强成功：

$$
\text{memory reduction vs full BP}>30\%
$$

$$
\text{memory reduction vs checkpoint}>10\%
$$

$$
\text{wall-clock overhead}<1.5\times
$$

如果 DG-KAN 只比 vanilla BP 省显存，但不如 checkpointing，则不能 claim 强低显存优势，只能 claim memory accounting 仍需优化。

---

## 9. Stage F：Sequential routing 与 small-span macro-router

### 9.1 目标

Stage F 是后期实验，不进入第一阶段主线。它用于测试 DG-KAN 中 learned router 串联后的误差累积，以及 small-span macro-router 是否有并行潜力。

只有在 Stage C 和 Stage D 成功后才进入 Stage F。

### 9.2 Sequential routing

深度设置：

$$
K\in\{4,8,12\}
$$

比较：

| 方法 | 说明 |
|---|---|
| BP-KAN | full BP oracle |
| AnalyticAdj sequential | 逐层 analytic KAN adjoint |
| Router teacher-forced | 输入 BP credit，测单层 router |
| Router self-routed | 输入上一层预测 credit，测部署模式 |
| Synthetic gradient | baseline |
| Feedback alignment | baseline |
| Local loss | baseline |

必须区分 teacher-forced 和 self-routed：

$$
\hat g_k^{TF}=C_k(h_k,h_{k+1})[g_{k+1}^{BP}]
$$

$$
\hat g_k^{self}=C_k(h_k,h_{k+1})[\hat g_{k+1}^{self}]
$$

如果 teacher-forced 好而 self-routed 坏，说明 router 拟合 local VJP 但 deployment distribution shift 失败。

指标：

| metric key | 含义 |
|---|---|
| `routing/layer_{k}/cos_bp_teacher_forced` | teacher-forced 与 BP cosine |
| `routing/layer_{k}/cos_bp_self` | self-routed 与 BP cosine |
| `routing/layer_{k}/relerr_self` | self-routed relative error |
| `routing/layer_{k}/norm_ratio_self` | self-routed norm ratio |
| `routing/layer_{k}/drift_self_vs_tf` | self vs teacher-forced drift |
| `routing/global/mean_cos_self` | self-routed 平均 cosine |
| `routing/global/worst_cos_self` | self-routed 最差层 cosine |
| `routing/global/error_amplification` | 误差放大因子 |
| `routing/global/credit_norm_growth` | credit norm 增长率 |

成功标准：8 层 self-routed DG-KAN 满足：

$$
\text{routing/global/mean\_cos\_self}>0.8
$$

$$
\text{routing/global/worst\_cos\_self}>0.6
$$

并且：

$$
\text{accuracy gap vs BP-KAN}<3\%
$$

### 9.3 Small-span macro-router

macro-router 暂时只做 small span：

$$
|b-a|\in\{2,4\}
$$

目标是：

$$
C_{a:b}\approx (D F_{a:b})^\top
$$

必须有真实锚点，例如 window exact VJP：

$$
g_a^{teacher}=J_{a:b}^\top g_b
$$

或 macro adjoint identity：

$$
\langle C_{a:b}(v),u\rangle
\approx
\langle v,J_{a:b}u\rangle
$$

composition loss 只能作为辅助项：

$$
\mathcal L_{macro}
=
\mathcal L_{anchor}+\beta\mathcal L_{comp}
$$

不能单独使用 composition loss，因为全零 router 也能满足 composition consistency。

---

## 10. DG-KAN credit geometry audit

虽然本计划不把普通 MLP 的 Gate 0 当主线，但仍然需要在 DG-KAN 上做 credit geometry audit。这个 audit 的目的不是决定是否放弃 full hidden-space，而是理解 DG-KAN 中 credit 是否可压缩、edge update 是否低秩、sufficient statistics 是否有压缩空间。

### 10.1 Hidden credit rank

对每层 hidden credit：

$$
g_k=\frac{\partial \mathcal L}{\partial h_k}
$$

构造矩阵：

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

W&B metrics：

| metric key | 含义 |
|---|---|
| `credit/hidden/layer_{k}/E8` | top-8 energy |
| `credit/hidden/layer_{k}/E16` | top-16 energy |
| `credit/hidden/layer_{k}/E32` | top-32 energy |
| `credit/hidden/layer_{k}/E64` | top-64 energy |
| `credit/hidden/layer_{k}/PR` | participation rank |
| `credit/hidden/layer_{k}/rank90` | 90% energy rank |
| `credit/hidden/layer_{k}/rank95` | 95% energy rank |

### 10.2 Edge update rank

对 edge coefficient gradient：

$$
\nabla_{a_{jim}}\mathcal L
$$

构造矩阵并做 SVD，记录 edge update 的压缩性。

W&B metrics：

| metric key | 含义 |
|---|---|
| `credit/edge_update/E8` | edge update top-8 energy |
| `credit/edge_update/E16` | edge update top-16 energy |
| `credit/edge_update/E32` | edge update top-32 energy |
| `credit/edge_update/PR` | edge update participation rank |
| `credit/edge_update/rank90` | edge update rank90 |

### 10.3 Sufficient statistics compression

对于 basis statistics：

$$
s_{jim}=\sum_n g_j^{(n)}B_m(x_i^{(n)})
$$

记录其存储量、秩和压缩后误差。

| metric key | 含义 |
|---|---|
| `stats/sufficient_stats_mb` | 原始 sufficient stats 显存 |
| `stats/compressed_stats_mb` | 压缩后显存 |
| `stats/compression_ratio` | 压缩比例 |
| `stats/compression_relerr` | 压缩误差 |
| `stats/kernel_sum_rank` | kernel / basis sum 统计秩 |
| `stats/x_g_joint_rank` | $(x_i,g_j)$ 联合统计秩 |

---

## 11. 第一轮最小可执行实验包

第一轮实验不再以 adapter-only 为主，而以 DG-KAN 为主。建议只做下面五个实验，避免一开始扩展过宽。

### Experiment 1：DG-KAN BP baseline

数据集：Two Moons、MNIST、Fashion-MNIST。

模型：DG-KAN-Small，depth $2,4,8$，hidden dim $32,64,128$，basis count $8,16$。

比较：MLP-Adam、KAN-Coeff-Adam、DG-KAN-Coeff-Adam。

目标：证明 DG-KAN 架构可以稳定训练。

成功标准：DG-KAN 在 MNIST / Fashion-MNIST 上不明显差于 matched MLP，且 $\phi'$、$\phi''$、Jacobian condition 不失控。

### Experiment 2：Functional update with true BP credit

数据集：1D / 2D regression、MNIST、Fashion-MNIST。

比较：KAN-Coeff-Adam、KAN-Coeff-AdamW、KAN-Sobolev、KAN-RKHS、diagonal preconditioner。

目标：验证 functional-space update 是否优于 coefficient update。

成功标准：validation loss 不差，收敛更快或 loss AUC 更小，$\phi'$ 和 $\phi''$ 更稳定，descent sign agreement rate 大于 $0.75$。

### Experiment 3：Analytic KAN adjoint

数据集：MNIST、Fashion-MNIST。

比较：autograd VJP 与 analytic KAN VJP。

目标：验证 KAN primitive 的 adjoint rule 正确且低成本。

成功标准：core KAN primitive 上：

$$
\cos>0.999
$$

$$
\operatorname{relerr}<10^{-4}
$$

并且 analytic VJP 的 memory 或 time 至少有一项优于 autograd VJP。

### Experiment 4：Learned KAN router

数据集：MNIST、Fashion-MNIST。

比较：analytic KAN adjoint、diagonal router、diagonal + low-rank router、derivative-compressed router、synthetic gradient、feedback alignment。

目标：验证 learned router 是否能逼近完整 hidden-space KAN adjoint。

成功标准：full hidden-space credit 上：

$$
\cos>0.95
$$

$$
\operatorname{relerr}<0.2
$$

并且 learned router 至少打过 identity、diagonal、static ridge baseline。

### Experiment 5：DG-KAN joint training

数据集：MNIST、Fashion-MNIST、CIFAR-10 small。

比较：BP-KAN-Adam、BP-KAN-Sobolev、AnalyticAdj-KAN-Sobolev、Router-KAN-Sobolev。

目标：验证完整 DG-KAN 训练是否成立。

成功标准：Router-KAN-Sobolev 与 BP-KAN-Adam 的 accuracy gap 小于 $2\%$ 到 $3\%$，mean credit cosine 大于 $0.8$，descent sign agreement rate 大于 $0.7$，并且 memory 低于 full BP。

---

## 12. Baseline 总表

本计划的 baseline 必须围绕 KAN 和 DG-KAN，而不是 adapter-only。

| baseline | 作用 |
|---|---|
| MLP-Adam | 参数化 baseline |
| MLP-Checkpoint | memory baseline |
| Residual-MLP | residual interface baseline |
| KAN-Coeff-Adam | standard KAN coefficient BP |
| KAN-Coeff-AdamW | standard KAN stronger optimizer |
| KAN-Sobolev-BP | full BP credit + functional update |
| KAN-RKHS-BP | full BP credit + RKHS update |
| KAN-AnalyticAdj-Sobolev | analytic local adjoint + functional update |
| DG-KAN-Router-Sobolev | learned router + functional update |
| KAN-SyntheticGradient | synthetic gradient baseline |
| KAN-FeedbackAlignment | feedback alignment baseline |
| KAN-LocalLoss | local objective baseline |
| Checkpointed KAN-BP | memory baseline |

---

## 13. Failure analysis

所有失败都必须记录到 W&B failure table。失败不是附注，而是决定下一步研究路线的核心证据。

| failure type | 触发条件 | 解释 |
|---|---|---|
| `dgkan_arch_unstable` | Stage A loss 发散或 NaN | 架构本身不稳定 |
| `phi_prime_explosion` | `kan/phi_prime_p95` 长期过高 | edge derivative 导致 credit 放大 |
| `phi_oversmooth` | train loss 降不下去且 curvature 很低 | functional update 过度平滑 |
| `precond_ill_conditioned` | $\kappa(M+\rho I)$ 过高 | metric 病态 |
| `descent_mismatch` | predicted descent < 0 but actual > 0 | 局部线性近似或 credit 有问题 |
| `analytic_adj_mismatch` | analytic VJP 与 autograd 不一致 | adjoint 公式或实现错误 |
| `router_low_cosine` | router cosine < 0.85 | router 学不准 |
| `router_norm_explosion` | norm ratio > 1.5 | router 放大 credit |
| `self_routed_drift` | self-routed drift 快速上升 | 分布漂移 / 误差累积 |
| `memory_no_gain` | 不如 checkpointing 省显存 | 低显存 claim 失败 |
| `overhead_too_high` | wall-clock 超 baseline 2 倍 | 计算开销过高 |

failure table 字段：

| 字段 | 含义 |
|---|---|
| `failure/type` | 失败类型 |
| `failure/stage` | 实验阶段 |
| `failure/dataset` | 数据集 |
| `failure/model` | 模型 |
| `failure/method` | 方法 |
| `failure/layer` | 层号 |
| `failure/step` | step |
| `failure/metric_name` | 触发 metric |
| `failure/metric_value` | metric 值 |
| `failure/threshold` | 阈值 |
| `failure/diagnosis` | 诊断 |
| `failure/recommended_action` | 建议动作 |

---

## 14. 最终论文级判定标准

### 14.1 最低可发表结果

如果 Stage A、B、C 成功，而 Stage D / E 还不完全成功，则可以形成一篇保守论文。可声称：

> DG-KAN provides a controlled setting where KAN-like primitives enable exact local adjoints and functional-space preconditioning; functional update improves stability, and learned local routers can approximate KAN adjoints in selected regimes.

中文表述：

> DG-KAN 提供了一个受控场景，使 KAN-like primitive 的局部 adjoint 和函数空间更新可以被明确实现；在部分设置中，functional update 改善稳定性，learned router 可以近似 KAN adjoint。

### 14.2 中等强度结果

如果 Stage D 中 AnalyticAdj-KAN-Sobolev 接近 BP-KAN-Adam，并且 Router-KAN-Sobolev 在 MNIST / Fashion-MNIST 上性能 gap 小于 $3\%$，则可以声称：

> DG-KAN can partially localize BP credit transport in KAN-like networks while retaining competitive performance.

### 14.3 强结果

如果 Stage E 进一步证明 DG-KAN sufficient statistics 训练显著省显存，且 Router-KAN-Sobolev 在 8 层模型上接近 BP，则可以声称：

> In KAN-like full hidden-state networks, local credit transport plus functional-space updates can replace selected components of global BP with practical memory tradeoffs.

要求至少满足：

$$
\text{accuracy gap}<2\%
$$

$$
\text{memory reduction vs full BP}>20\%
$$

$$
\text{wall-clock overhead}<1.5\times
$$

$$
\text{mean credit cosine}>0.8
$$

$$
\text{descent sign agreement rate}>0.7
$$

---

## 15. 最终建议

本实验计划的第一优先级不是证明 DG-LCA 通用替代 BP，而是把原始设计中最有优势的 KAN-like 路线做扎实。实验主线应当是：

$$
\boxed{
\text{DG-KAN baseline}
\rightarrow
\text{functional update with true BP credit}
\rightarrow
\text{analytic KAN adjoint}
\rightarrow
\text{learned KAN router}
\rightarrow
\text{joint DG-KAN training}
\rightarrow
\text{low-memory sufficient statistics}
}
$$

adapter / LoRA 不是本计划主线。它们可以作为后续扩展或对照，但不能替代 DG-KAN 主实验。

如果最终发现 analytic KAN adjoint 成功、functional update 有收益，但 learned router 不稳定，那么论文仍然可以聚焦于：

$$
\text{KAN-like primitive-adjoint co-design}
+
\text{functional-space update}
+
\text{low-memory sufficient statistics}
$$

如果 learned router 也成功，则 DG-KAN 将成为一个更完整的 DG-LCA 实验系统，能够同时展示局部 credit transport、函数空间更新和低显存训练的可行性。
