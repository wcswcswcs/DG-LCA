# DG-KAN v5.4 计划：Efficiency-First PureKAN Functional Training System

> 文件名：`DG-KAN_v5.4_EfficiencyFirst_PureKAN_FunctionalTraining_实验计划.md`  
> 主题：在终极目标约束下重新设计 PureKAN functional training system  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`  
> 当前结论：AB-RBF 与 LightSmooth 有机制进展，但还不是最终高效系统。v5.4 的主线不再是继续加 smoothing controller，而是先证明 PureKAN edge primitive 能接近 MLP 的时间/显存，并为 functional training 提供正确的高效坐标。

---

## 0. 终极目标重新固定

本项目的终极目标定义为：

$$
\boxed{
\text{构建一个无 non-KAN 参数的 PureKAN functional training system，}
}
$$

$$
\boxed{
\text{其 forward 时间/显存接近 MLP，backward 时间接近 MLP，backward 显存低于 MLP，收敛速度快，}
}
$$

$$
\boxed{
\text{并在 accuracy / AUC / ECE / geometry 上超过 MLP-AdamW 与 PureKAN-AdamW。}
}
$$

这个目标比前几轮更严格。它意味着一个候选方法不能只做到：

```text
accuracy 还行；
geometry 还行；
某次 smoothing 有效果。
```

它还必须满足：

```text
forward cost 接近 MLP；
backward memory 明显优于 MLP；
step time 不能明显慢；
训练过程不是 AdamW 后处理，而是可持续的 functional training system；
所有可学习参数都属于 KAN edge 系统，不允许普通 MLP stem/head/LN 参数作为隐藏支撑。
```

因此 v5.4 不能再继续只围绕 smoothing event controller 打转。现在的核心问题变成：

$$
\boxed{
\text{我们是否有一个足够高效、足够可训练、又足够几何可控的 PureKAN edge primitive？}
}
$$

---

## 1. 当前 v5.3 结果的真实含义

### 1.1 已经确认的进展

v5.3 的 P0 说明 AB-RBF core / memory smoke 没有错误。LightSmooth 相比 StrongSmooth 的成本明显降低。典型结果是：

```text
ABRBF-AdamW: memory ratio = 1.0
ABRBF-LightSmooth: memory ratio 大约 1.02 到 1.20
ABRBF-StrongSmooth: memory ratio 大约 3.4 到 4.0
```

这说明：

$$
\boxed{
\text{StrongSmooth 太重，但 LightSmooth 作为低成本 smoothing primitive 是可用的。}
}
$$

v5.3 的 P1 single-event verification 也说明 LightSmooth 可以在 `acc drop = 0`、KL/logit drift 很小的情况下带来 residual geometry reduction。不过与 v5.2 相比，v5.3 的 P1 reduction 幅度变小，典型值大约是：

```text
phiR reduction: 约 7%
curvR reduction: 约 14%
```

这说明 LightSmooth 是有效工具，但它不是强 optimizer。

### 1.2 仍然失败的部分

v5.3 的关键失败在 P3 / P4：

```text
P3 all-dataset survivors: none
P4 all-dataset survivors: none
P5/P6/P7 not run
```

最终诊断是：

```text
refresh 可以恢复 task accuracy；
但 final residual phi reduction 只剩大约 0.8% 到 1.8%，低于 gate；
因此 event-driven multicycle 不能进入 seed confirmation。
```

这说明：

$$
\boxed{
\text{LightSmooth 可以单次降低 residual geometry，但训练期多轮维护不能稳定保留几何收益。}
}
$$

更重要的是，这个失败不是简单的 trigger 问题。v5.3 已经加入 P2 cheap probe audit，并且新 score/debt trigger 能观察到触发机会；旧 plateau trigger 迟钝的问题已经被看见。真正的问题是：

$$
\boxed{
\text{smoothing 后的 refresh 会把 geometry gain 冲掉。}
}
$$

所以继续微调 event threshold、cooldown、eta cap 的边际收益很低。

---

## 2. 代码实现层面的关键审视

### 2.1 当前上传的 `dgkan_core(5).py` 与 v5.3 文档存在一致性风险

v5.1 / v5.3 结果文档显示：

```text
ABRBFDense 已经进入 core；
edge_named_params / base_named_params / rbf_residual_named_params 可以审计；
P0 core consistency passed。
```

但是这次上传的 `dgkan_core(5).py` 中，基础可见结构仍然主要是：

```text
RBFDense:
  fixed centers
  fixed width
  trainable coeff
  optional bias

PureKANClassifier:
  input_kan = RBFDense(..., bias=False)
  blocks = RBFDense(..., bias=False)
  output_kan = RBFDense(..., bias=False)
```

在该上传文件里未能明显看到 `ABRBFDense`、`edge_named_params`、`base_named_params`、`rbf_residual_named_params` 这些 core-level 定义。这有两种可能：

```text
1. 上传的 core 不是实际运行 v5.3 的最新版；
2. AB-RBF 仍然主要在 runner/helper 里定义，而不是核心库里统一定义。
```

不管是哪种，v5.4 第一优先级都是 **core consistency hardening**。如果 AB-RBF 不是 core primitive，后续 efficiency benchmark、custom backward、CIFAR、Rational/KAT、functional update 都会有实现分裂风险。

### 2.2 Dense RBF / AB-RBF 无法天然满足 MLP-like efficiency

标准 dense RBF edge 是：

$$
f_o(x)=\sum_{i,k} c_{oik} B_k(x_i).
$$

其核心计算是：

```python
basis: [B, c_in, K]
out = einsum("bik,oik->bo", basis, coeff)
```

计算复杂度为：

$$
O(Bc_{in}K)+O(Bc_{out}c_{in}K).
$$

MLP / Linear 的复杂度是：

$$
O(Bc_{out}c_{in}).
$$

因此当 $K=16$ 或 $K=24$ 时，dense RBF / dense AB-RBF 的 forward 不可能天然接近 MLP。即使 LightSmooth 变轻，主 forward/backward primitive 仍然偏重。

这意味着：

$$
\boxed{
\text{AB-RBF dense 是机制平台，不一定是终极高效 primitive。}
}
$$

### 2.3 backward 显存要低于 MLP，必须 custom backward / analytic adjoint

如果用 PyTorch autograd 保存完整 `basis: [B,c_in,K]`，则 backward memory 不可能优于 MLP。要满足终极目标，必须使用：

```text
basis recomputation;
streaming coeff-gradient accumulation;
activation checkpoint / no-save basis;
analytic VJP;
sufficient statistics;
```

目标是 forward 时不保存完整 basis，而在 backward 中根据输入重新计算：

$$
B_k(x_i),
$$

并流式累积：

$$
\nabla c_{oik}=\sum_b \delta_{bo}B_k(x_{bi}).
$$

如果没有这个方向，RBF/AB-RBF 很难满足：

$$
M_{backward,KAN} < M_{backward,MLP}.
$$

---

## 3. 目前路线的重新判断

### 3.1 保留的结论

当前仍然成立的结论是：

```text
1. RBF-only coefficient coordinate 是 PureKAN 的坏坐标；
2. AB-RBF edge decomposition 是正方向；
3. base path 不是装饰，而是承担低阶/尺度/线性结构；
4. RBF residual 可以通过 LightSmooth 做低成本单次 geometry maintenance；
5. StrongSmooth / exact-heavy NFS 太重，不适合作为训练主线。
```

### 3.2 需要降级的结论

需要降级的是：

```text
1. AB-RBF dense 不能直接视为终极高效 PureKAN primitive；
2. LightSmooth 还不是稳定 train-time multicycle optimizer；
3. functional update 目前更像 residual geometry maintenance，而不是 from-scratch functional optimizer；
4. 仅靠 event controller 不能解决 geometry-retention-after-refresh 问题。
```

### 3.3 新的关键问题

v5.4 必须回答三个更硬的问题：

$$
\boxed{
Q1: \text{哪种 PureKAN edge primitive 能在 forward/backward 上接近 MLP？}
}
$$

$$
\boxed{
Q2: \text{能否用 analytic/custom backward 让 backward 显存低于 MLP？}
}
$$

$$
\boxed{
Q3: \text{在高效 primitive 上，functional training 是否还能保留 geometry/AUC/ECE 优势？}
}
$$

---

## 4. v5.4 的主线：Efficiency-First PureKAN

v5.4 不再把 smoothing controller 当主线，而是把主线拆成三条并行但有顺序的路线。

### 4.1 路线 A：Dense AB-RBF 只作为 reference

Dense AB-RBF 保留，用于：

```text
机制 reference；
geometry reference；
AB-RBF edge decomposition 上限；
LightSmooth 后处理 reference。
```

但它不再默认是最终系统。

### 4.2 路线 B：高效 AB-RBF primitive

必须实现至少三种高效替代：

#### B1. Depthwise-ABRBF + LinearMix

先对每个 channel 做一维 edge function：

$$
\tilde x_i = b_i+w_i x_i+u_i\operatorname{silu}(x_i)+\sum_k c_{ik}B_k(x_i),
$$

再做 mixing：

$$
y=W\tilde x.
$$

复杂度：

$$
O(Bc_{in}K)+O(Bc_{in}c_{out}).
$$

这比 dense AB-RBF 的：

$$
O(Bc_{in}c_{out}K)
$$

更接近 MLP。

为了保持 “无 non-KAN 参数”，$W$ 必须被定义为 KAN edge mixing parameter，而不是普通外部 MLP head/stem。它可以被纳入 `edge_named_params`，并在 final claim 中计入 KAN edge system。

#### B2. LowRank / CP-ABRBF

把 dense RBF residual coefficient：

$$
c_{oik}
$$

分解为：

$$
c_{oik}=\sum_{r=1}^{R}U_{or}V_{ir}W_{kr}.
$$

当 $R \ll \min(c_{in},c_{out},K)$ 时，参数量和计算量显著下降。需要测试：

```text
R = 4, 8, 16
```

并记录是否出现 rank bottleneck。

#### B3. Rational / KAT-AB Edge

Rational/KAT 的基本结构是：

$$
\tilde x_i=r_i(x_i),
$$

$$
y=W\tilde x.
$$

它天然更接近 MLP，因为主计算仍然是 GEMM。前面 Rational-AdamW 有信号，但 Rational functional metric 没跑通。v5.4 不再先追 Rational functional update，而先测试：

```text
Rational/KAT-AB 是否满足终极 efficiency envelope；
accuracy / ECE / geometry 是否接近 AB-RBF；
是否能作为最终 high-efficiency primitive。
```

### 4.3 路线 C：custom backward / analytic adjoint

对 dense AB-RBF 和高效 AB-RBF 都要实现：

```text
Autograd version
CustomBackward-Recompute version
CustomBackward-StreamingStats version
```

目标是验证：

$$
M_{backward,KAN} \leq 0.8 M_{backward,MLP}.
$$

---

## 5. v5.4 实验计划总览

v5.4 分为九个阶段：

```text
P0: Core consistency and primitive manifest
P1: Pure primitive efficiency microbenchmark
P2: custom backward correctness and memory audit
P3: accuracy frontier under AdamW-like training
P4: functional / analytic update smoke on efficient primitives
P5: LightSmooth compatibility on efficient primitives
P6: 3-seed efficiency-aware candidate selection
P7: 5-seed confirm
P8: 10-seed final confirm
P9: failure diagnosis and route decision
```

每个阶段都必须同时记录：

```text
task metrics;
geometry metrics;
edge contribution metrics;
efficiency metrics;
coverage / strict PureKAN invariants.
```

---

## 6. P0: Core consistency and primitive manifest

### 6.1 目的

确保所有候选都是 strict PureKAN，且 AB-RBF / efficient edge primitive 不再散落在 runner helper 中。

### 6.2 必测方法

```text
MLP-AdamW-reference
RBFOnly-Dense
ABRBF-Dense-linear+silu
ABRBF-DepthwiseMix-linear+silu
ABRBF-CPRank4
ABRBF-CPRank8
ABRBF-CPRank16
RationalKAT-AB
```

### 6.3 必须检查的不变量

```text
learnable_nonKAN_params = 0
edge_param_coverage = 1.0
base_param_coverage = 1.0, if applicable
rbf_residual_param_coverage = 1.0, if applicable
mixing_param_coverage = 1.0, if applicable
rollback_error = 0
no hidden nn.Linear outside edge system
no learnable LayerNorm gamma/beta
no ordinary stem/head
```

对于 DepthwiseMix / RationalKAT-AB，必须明确：

```text
LinearMix / W mixing 是 edge system 内部参数；
不是 ordinary non-KAN head。
```

### 6.4 记录字段

```text
method
primitive_type
num_edge_params
num_base_params
num_rbf_params
num_mixing_params
num_nonkan_params
coverage_edge
coverage_base
coverage_rbf
coverage_mixing
rollback_max_abs
forward_shape_ok
backward_shape_ok
```

### 6.5 判定

任何方法如果：

```text
num_nonkan_params > 0
coverage_edge < 1.0
```

直接不能进入 P1。

---

## 7. P1: Pure primitive efficiency microbenchmark

### 7.1 目的

先不看最终 accuracy，先确认 primitive 是否可能满足终极效率目标。

### 7.2 测试维度

使用 synthetic input 和真实 batch 两类输入。

```text
batch_size = 64, 128, 256, 512
input_dim = 784, 1024, 2048
hidden_dim = 64, 96, 128, 256
basis_count = 8, 16, 24
num_classes = 10
```

### 7.3 对照方法

```text
MLP-Linear+SiLU
RBFOnly-Dense
ABRBF-Dense-linear+silu
ABRBF-DepthwiseMix
ABRBF-CPRank4/8/16
RationalKAT-AB
```

### 7.4 记录指标

#### 时间

```text
forward_time_ms_mean
forward_time_ms_p50
forward_time_ms_p95
backward_time_ms_mean
backward_time_ms_p50
backward_time_ms_p95
step_time_ms_mean
samples_per_second
```

#### 显存

```text
forward_peak_allocated_mb
forward_peak_reserved_mb
backward_peak_allocated_mb
backward_peak_reserved_mb
activation_saved_mb
basis_saved_mb
optimizer_state_mb
```

#### 参数/FLOPs

```text
num_params
edge_params
base_params
rbf_params
mixing_params
estimated_forward_flops
estimated_backward_flops
flop_ratio_vs_mlp
```

### 7.5 Gate

进入 P2 的硬门槛：

$$
\frac{T_{forward,KAN}}{T_{forward,MLP}} \leq 1.25,
$$

$$
\frac{M_{forward,KAN}}{M_{forward,MLP}} \leq 1.25,
$$

$$
\frac{T_{backward,KAN}}{T_{backward,MLP}} \leq 1.40,
$$

$$
\frac{M_{backward,KAN}}{M_{backward,MLP}} \leq 1.00
$$

作为 P1 初筛。最终目标更严格：

$$
\frac{M_{backward,KAN}}{M_{backward,MLP}} \leq 0.80.
$$

P1 阶段允许先放宽到 $1.0$，因为 custom backward 还没完全进入。

### 7.6 必须画图

```text
forward time ratio vs hidden_dim
backward memory ratio vs hidden_dim
step time ratio vs basis_count
FLOPs vs measured time scatter
activation_saved_mb stacked bar
throughput vs batch_size
```

---

## 8. P2: Custom backward correctness and memory audit

### 8.1 目的

证明 KAN backward 能不用保存完整 basis activation，靠 recomputation / streaming stats 达到低显存。

### 8.2 方法

对以下 primitive 实现两版：

```text
Autograd
CustomBackward-Recompute
CustomBackward-StreamingStats
```

候选：

```text
ABRBF-Dense-linear+silu
ABRBF-DepthwiseMix
ABRBF-CPRank8
RationalKAT-AB, if applicable
```

### 8.3 正确性检查

对同一 batch，比对 autograd 和 custom backward：

$$
\operatorname{relerr}(\nabla \theta)
=
\frac{\|g_{custom}-g_{autograd}\|}{\|g_{autograd}\|+\epsilon}.
$$

记录：

```text
grad_relerr_coeff
grad_relerr_base
grad_relerr_rbf
grad_relerr_mixing
grad_cos_coeff
grad_cos_base
grad_cos_rbf
grad_cos_mixing
```

目标：

$$
\operatorname{relerr} < 10^{-4}
$$

或至少：

$$
\cos(g_{custom},g_{autograd}) > 0.999.
$$

### 8.4 显存检查

记录：

```text
basis_tensor_saved = true/false
saved_tensor_count
saved_tensor_total_mb
recompute_time_ms
streaming_stats_time_ms
backward_peak_allocated_mb
backward_peak_reserved_mb
```

### 8.5 Gate

进入 P3 的方法必须满足：

```text
grad correctness pass;
backward memory <= MLP backward memory;
custom step time <= 1.5x MLP step time;
```

### 8.6 可视化

```text
grad relerr histogram by param group
custom vs autograd memory bar
custom vs autograd step time bar
saved activation breakdown
memory-time Pareto frontier
```

---

## 9. P3: Accuracy frontier under AdamW-like training

### 9.1 目的

先验证高效 primitive 是否有任务能力。这里暂时允许 AdamW，因为问题是 architecture + efficiency frontier，不是 functional optimizer。

### 9.2 方法

```text
MLP-AdamW
RBFOnly-Dense-AdamW
ABRBF-Dense-AdamW
ABRBF-DepthwiseMix-AdamW
ABRBF-CPRank8-AdamW
ABRBF-CPRank16-AdamW
RationalKAT-AB-AdamW
```

### 9.3 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

第一轮：

```text
seeds = 0,1,2
train/val/test = 6000/1000/1000
```

### 9.4 记录指标

```text
test_acc
val_loss
val_auc
ECE
NLL
margin_mean
margin_p10
classwise_acc
feature_effective_rank
class_centroid_separation
base_ablation_drop
rbf_ablation_drop
mixing_ablation_drop
phi_base
phi_rbf
phi_total
curvature_rbf
sobolev_rbf_norm
forward_time_ratio
backward_memory_ratio
step_time_ratio
```

### 9.5 Gate

候选进入 P4 需要：

```text
accuracy >= MLP-AdamW - 0.5%
val_auc <= MLP-AdamW + 5% relative loss-AUC cost
ECE <= MLP-AdamW + 0.02
forward_time <= 1.25x MLP
backward_memory <= 1.0x MLP
step_time <= 1.4x MLP
```

### 9.6 可视化

```text
accuracy vs step_time ratio Pareto
accuracy vs backward memory ratio Pareto
ECE vs geometry scatter
base/RBF/mixing ablation bar
feature rank trajectory
classwise accuracy heatmap
```

---

## 10. P4: Functional / analytic update smoke on efficient primitives

### 10.1 目的

在高效 primitive 上测试 functional training 是否仍然可行。这里不再用旧的 heavy U-FULL，而使用轻量 functional variants。

### 10.2 方法

```text
AdamW baseline
FunctionalCoord-Adam
ResidualOnly-FunctionalSmooth
AnalyticAdj-StatsUpdate
TaskAdam + LightSmooth maintenance
```

其中：

```text
FunctionalCoord-Adam:
  在 function-whitened coordinate 中用 Adam-like dynamics。

ResidualOnly-FunctionalSmooth:
  base/mixing 用 Adam-like task update，RBF residual 用 light geometry update。

AnalyticAdj-StatsUpdate:
  使用解析 VJP / sufficient statistics 更新 edge 参数，目标是减少 backward memory。

TaskAdam + LightSmooth:
  AdamW-like task learner + 低频 LightSmooth，作为 pragmatic bridge。
```

### 10.3 记录指标

除了 P3 指标，还要记录：

```text
functional_update_norm
functional_update_cos_with_adam
grad_storage_mb
stats_storage_mb
analytic_vjp_time_ms
basis_recompute_time_ms
smoothing_event_count
smoothing_accepted_count
smoothing_geometry_gain
smoothing_task_cost
```

### 10.4 Gate

```text
accuracy >= AdamW version of same primitive - 1.0%
val_auc not worse than AdamW version by > 5%
ECE <= AdamW version + 0.02
rbf geometry improves by >= 5%
backward memory <= AdamW-autograd version by 20%
step time <= AdamW-autograd version + 20%
```

### 10.5 可视化

```text
functional vs AdamW trajectory overlay
update cosine over time
geometry gain vs task cost scatter
memory breakdown by training path
analytic stats fidelity plot
```

---

## 11. P5: LightSmooth compatibility on efficient primitives

### 11.1 目的

v5.3 说明 LightSmooth 单次可用，但 refresh 后 geometry retention 不够。P5 在高效 primitive 上重新验证，不再先做 multicycle。

### 11.2 方法

```text
EfficientPrimitive-AdamW
EfficientPrimitive-AdamW + one-shot LightSmooth
EfficientPrimitive-AdamW + one-cycle SmoothRefresh
EfficientPrimitive-AdamW + posthoc LightSmooth only
```

### 11.3 Refresh policy

只保留两个最有解释性的 refresh：

```text
base/mixing-only refresh:
  smoothing 后冻结 RBF residual，只恢复 base/mixing task fit。

small-RBF refresh:
  smoothing 后 RBF residual 只允许 very small LR，避免几何收益被冲掉。
```

### 11.4 记录指标

```text
pre_smooth_acc
post_smooth_acc
post_refresh_acc
pre_smooth_phi_rbf
post_smooth_phi_rbf
post_refresh_phi_rbf
geometry_retention = post_refresh_phi_gain / post_smooth_phi_gain
refresh_recovery = recovered_acc_drop / smooth_acc_drop
KL
logit_drift
memory_ratio
smooth_time
```

### 11.5 Gate

```text
post_refresh_acc >= baseline_acc - 0.005
geometry_retention >= 0.50
memory_ratio <= 1.25
amortized_time_overhead <= 0.10
```

### 11.6 可视化

```text
smooth-refresh recovery plot
geometry retention bar
accuracy/geometry timeline with smoothing marker
smoothing cost waterfall
```

---

## 12. P6: 3-seed efficiency-aware candidate selection

### 12.1 候选来源

只选择同时通过 P3/P4/P5 的方法。最多保留 3 个：

```text
best efficient primitive AdamW
best efficient primitive functional/analytic update
best efficient primitive + LightSmooth maintenance
```

### 12.2 对照

```text
MLP-AdamW
PureKAN-RBFOnly-AdamW
PureKAN-ABRBF-Dense-AdamW
best efficient candidates
```

### 12.3 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### 12.4 Gate

候选必须满足：

$$
Acc \geq Acc_{MLP-AdamW},
$$

$$
AUC \leq AUC_{MLP-AdamW},
$$

或至少相对 MLP loss-AUC 不差于 $5\%$，并且：

$$
ECE \leq ECE_{MLP-AdamW},
$$

$$
\phi_{rbf}\text{ 或 }\operatorname{curv}_{rbf}\text{ 有明确改善},
$$

$$
T_{step} \leq 1.25 T_{MLP},
$$

$$
M_{backward} \leq 0.8 M_{MLP}.
$$

P6 是 v5.4 的关键阶段。如果没有候选过 P6，则不要进入 5-seed。

---

## 13. P7/P8: 5-seed and 10-seed confirm

只有 P6 通过后才运行。

### 13.1 5-seed confirm

```text
seeds = 0,1,2,3,4
```

必须记录 paired delta vs：

```text
MLP-AdamW
PureKAN-AdamW same primitive
ABRBF-Dense-AdamW
```

### 13.2 10-seed final

```text
seeds = 0..9
```

最终必须输出：

```text
paired_acc_delta
paired_auc_delta
paired_ece_delta
paired_geometry_delta
paired_step_time_ratio
paired_backward_memory_ratio
```

---

## 14. P9: Failure diagnosis and route decision

如果 v5.4 没有候选通过 P6，需要根据失败模式做明确决策。

### 14.1 失败模式 F1：效率失败

```text
accuracy 可以，但 forward/backward 仍远慢于 MLP。
```

结论：

```text
Dense / factorized RBF 不是最终 primitive；转 Rational/KAT 或更强 fused kernel。
```

### 14.2 失败模式 F2：accuracy 失败

```text
效率过了，但 accuracy 不如 MLP / ABRBF-Dense。
```

结论：

```text
efficient primitive 表达力不足；提高 rank / group / base path，或保留 dense ABRBF 做机制平台。
```

### 14.3 失败模式 F3：backward memory 未降

```text
custom backward 正确，但显存不低。
```

结论：

```text
检查是否仍保存 basis / hidden / teacher tensors；改 streaming stats / recomputation；必要时写 fused CUDA/Triton。
```

### 14.4 失败模式 F4：geometry 失败

```text
accuracy 和效率过，但 geometry 没优势。
```

结论：

```text
LightSmooth 作为 posthoc/maintenance，或改 residual geometry metric；但不牺牲效率主线。
```

### 14.5 失败模式 F5：functional training 失败但 AdamW 成功

```text
efficient primitive + AdamW 成功，functional/analytic update 不成功。
```

结论：

```text
短期主线用 AdamW-like task training + analytic memory-saving backward；functional update 保留为 residual maintenance，不再宣称 fully functional-from-scratch。
```

---

## 15. 最终产物要求

v5.4 必须输出以下文件：

```text
p0_core_manifest.csv
p1_efficiency_microbenchmark.csv
p1_efficiency_gate_summary.csv
p2_custom_backward_correctness.csv
p2_memory_audit.csv
p3_accuracy_frontier.csv
p4_functional_update_smoke.csv
p5_lightsmooth_compatibility.csv
p6_candidate_selection3.csv
p7_confirm5.csv
p8_confirm10.csv
p9_failure_diagnosis.csv
failure_table.csv
aggregate_decision.json
recommendation.json
figures/
```

必须生成以下图：

```text
fig_forward_time_ratio_vs_hidden.png
fig_backward_memory_ratio_vs_hidden.png
fig_step_time_vs_accuracy_pareto.png
fig_backward_memory_vs_accuracy_pareto.png
fig_saved_activation_breakdown.png
fig_custom_backward_relerr_hist.png
fig_base_rbf_mixing_contribution.png
fig_accuracy_geometry_efficiency_pareto.png
fig_smooth_refresh_geometry_retention.png
fig_failure_heatmap.png
```

---

## 16. 当前推荐优先级

我建议 v5.4 的执行顺序是：

```text
1. 修正 / 确认 ABRBFDense core-level 实现。
2. 先跑 P1 efficiency microbenchmark，不要先跑更多 smoothing。
3. 对通过 P1 的 primitive 做 custom backward P2。
4. 只对通过 P1/P2 的 primitive 做 P3 accuracy frontier。
5. 只在高效且 accuracy 有希望的 primitive 上做 functional / LightSmooth。
```

也就是说：

$$
\boxed{
\text{效率先行，几何随后；没有效率，不进入终极目标主线。}
}
$$

---

## 17. 最终判断

v5.3 后，我们的状态是：

```text
AB-RBF: 有机制价值；
LightSmooth: 单次有效，训练期 controller 未成；
Dense RBF/AB-RBF: 不一定满足终极效率；
functional-from-scratch: 仍未完成；
custom backward / efficient primitive: 必须成为下一步核心。
```

因此 v5.4 的核心不是继续问：

```text
怎么让 LightSmooth 多轮更稳？
```

而是先问：

```text
我们有没有一个 forward/backward 接近 MLP 的 PureKAN primitive？
```

如果没有，任何 optimizer 和 smoothing 都无法满足终极目标。

