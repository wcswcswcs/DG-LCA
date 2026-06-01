# DG-KAN v5.5 Kernel-First Efficient PureKAN 实验计划

## 0. 终极目标与本轮重新定位

当前终极目标定义为：构建一个无 non-KAN 参数的 PureKAN functional training system，其 forward 时间与显存接近 MLP，backward 时间接近 MLP，backward 显存低于 MLP，收敛速度快，并且在 accuracy、validation-loss AUC、ECE、geometry 上超过 MLP-AdamW 与 PureKAN-AdamW。

这个目标比此前的“PureKAN 能训练”严格得多。它要求我们同时满足：

$$
\text{task quality}
+
\text{functional geometry}
+
\text{MLP-like efficiency}
+
\text{memory advantage}.
$$

v5.4 的结果说明，当前路线不能继续只在 optimizer / smoothing 上小修。原因是所有候选 primitive 都没有通过效率 envelope，P3 也没有产生 accuracy + efficiency 的 all-dataset survivor。因此 v5.5 的核心不是继续追问 “functional update 怎么调”，而是先回答：

$$
\boxed{
\text{我们是否拥有一个真正接近 MLP 的 PureKAN edge primitive？}
}
$$

如果没有，那么后续 LightSmooth、functional update、event controller 都只能是机制研究，无法满足终极目标。

---

## 1. v5.4 结果复盘：现在到底卡在哪里

### 1.1 P0 的正面信号：efficient primitives 已经进入实验对象

v5.4 P0 中，候选 primitive 包括：

```text
RBFOnly-Dense
ABRBF-Dense-linear+silu
ABRBF-DepthwiseMix-linear+silu
ABRBF-CPRank4 / 8 / 16
RationalKAT-AB
```

它们都保持了 strict PureKAN 设定：learnable non-KAN 参数为 0，mixing 参数也被计入 KAN edge system。也就是说，本轮不是“混入 MLP 参数后变快”，而是在 PureKAN edge 内部尝试不同的高效参数化。

但 P0 只能说明参数归属正确，不能说明计算路径正确。v5.4 后续结果显示，虽然候选都进入了 core-level object，但计算效率距离 MLP 仍然很远。

---

### 1.2 P1 的核心失败：没有 primitive 通过效率门槛

P1 efficiency microbenchmark 中，所有方法 pass rate 都是 0。

最接近 MLP step time 的是 `RBFOnly-Dense`：

```text
forward ratio = 2.84
backward ratio = 1.35
backward memory ratio = 2.52
step ratio = 1.51
```

这说明 RBFOnly-Dense 虽然相对其它 KAN primitive 速度还算低，但仍然远不满足终极目标，尤其 backward memory 明显高于 MLP。

`RationalKAT-AB` 的 backward memory ratio 只有约 1.35，是所有候选里更接近 memory target 的一个，但它 forward、backward、step time 都很慢：

```text
forward ratio = 6.99
backward ratio = 6.00
step ratio = 4.38
```

这说明 Rational/KAT 的理论复杂度虽然接近 MLP，但当前实现路径并没有把这个优势释放出来。

`ABRBF-DepthwiseMix` 理论上应该是一个更接近 MLP 的结构，因为复杂度应当接近：

$$
O(BdK)+O(Bdh)
$$

而不是 Dense RBF 的：

$$
O(Bd h K).
$$

但实际 P1 结果是：

```text
forward ratio = 7.87
backward ratio = 8.29
step ratio = 5.71
```

这说明当前 DepthwiseMix 实现很可能没有走到真正 GEMM-native / fused path，或者 benchmark 被大量小 kernel、basis materialization、layout copy、einsum fallback、Python overhead、autograd 保存张量拖慢。

因此 v5.4 的第一结论是：

$$
\boxed{
\text{当前失败的主因是 primitive implementation / kernel path，而不是 functional optimizer。}
}
$$

---

### 1.3 P2 的关键教训：custom backward 正确，但没有降低 measured peak memory

P2 custom backward audit 显示，`CustomBackward-Recompute` 和 `CustomBackward-StreamingStats` 的 relerr 为 0、cos 为 1，说明梯度正确性没有问题。

但是 measured backward memory ratio 没有下降。例如 `ABRBF-DepthwiseMix`：

```text
Autograd saved MB = 7.27
Custom saved MB = 1.09
bmem ratio = 2.98 in both cases
```

这说明我们目前记录的 `saved MB` 和实际 CUDA peak memory 不是同一个瓶颈。可能的原因包括：

```text
1. peak memory 被 forward basis materialization 或 temporary tensors 主导；
2. benchmark 没有按 forward/backward/optimizer phase 分开 reset peak；
3. optimizer state / gradient buffers / module buffers 被计入 peak；
4. recompute backward 仍然在内部 materialize 了同等大小的 basis；
5. PyTorch allocator reserved memory 掩盖了 activation saving 的改善；
6. custom backward 降低了 saved tensors，但没有降低最大临时分配。
```

因此 v5.5 不能只写“custom backward 已省显存”。必须做 **phase-separated memory accounting**：

$$
M_{peak}=
M_{params}+M_{grads}+M_{optimizer}+M_{forward\ activations}+M_{backward\ temporaries}+M_{workspace}.
$$

如果不知道哪个项主导，就无法优化 backward memory。

---

### 1.4 P3 的任务质量信号：CP-ABRBF 有 accuracy 潜力，但不是效率解

P3 accuracy frontier 中，CP-ABRBF 是相对有任务质量的候选。

Fashion-MNIST：

```text
MLP-AdamW acc = 0.8001
ABRBF-CPRank16 acc = 0.7956
ABRBF-DepthwiseMix acc = 0.7956
```

MNIST：

```text
MLP-AdamW acc = 0.9076
ABRBF-CPRank8 acc = 0.8978
ABRBF-CPRank16 acc = 0.8893
```

KMNIST 最难：

```text
MLP-AdamW acc = 0.7194
ABRBF-Dense acc = 0.6979
ABRBF-CPRank16 acc = 0.6908
ABRBF-DepthwiseMix acc = 0.6191
RationalKAT-AB acc = 0.5814
```

这说明 CP-ABRBF 可以恢复一部分 accuracy，尤其在 MNIST/Fashion 上，但它效率远远不达标；DepthwiseMix 理论上高效，但当前任务质量和效率都不达标；RationalKAT 当前最接近 memory 目标，但任务质量和速度都差。

因此 v5.4 的第二结论是：

$$
\boxed{
\text{CP-ABRBF 是 architecture/accuracy signal，不是 final efficient primitive。}
}
$$

---

## 2. 代码实现审视：现在最可能的实现问题

### 2.1 Dense RBFDense 仍然是一个机制 reference，不可能满足最终效率目标

基础 RBF 层的实现路径是：

```python
basis = exp(-0.5 * ((x.unsqueeze(-1) - centers) / width)^2)
out = einsum("bik,oik->bo", basis, coeff)
```

这会 materialize：

$$
\text{basis}\in\mathbb R^{B\times d\times K}.
$$

然后执行 edge-wise aggregation：

$$
y_o=\sum_{i,k}B_k(x_i)c_{oik}.
$$

计算量是：

$$
O(B d h K).
$$

MLP 是：

$$
O(B d h).
$$

所以 Dense RBF / Dense AB-RBF 只能作为机制平台，不能作为终极高效 primitive。即使用 custom backward，它的 forward 本身仍然不接近 MLP。

---

### 2.2 DepthwiseMix 理论上应该快，但当前表现说明实现路径有问题

DepthwiseMix 理想形式应是：

$$
\tilde x_i=f_i(x_i),
$$

$$
y=W\tilde x.
$$

其中 $f_i$ 是一维 AB-RBF function，$W$ 是 edge-system 内部 mixing。复杂度应该是：

$$
O(BdK)+O(Bdh).
$$

如果 $K=16$，$d=h=64$，理论上它应该接近 MLP，而不是比 MLP 慢 5x。v5.4 P1 的结果说明当前实现大概率存在以下问题之一：

```text
1. depthwise basis 仍然以不友好的 layout materialize；
2. depthwise function 和 mixing 没有融合，产生多个小 kernel；
3. mixing 没走标准 GEMM 或使用了 slow einsum；
4. backward 保存/重算路径重复做 basis；
5. batch size 太小，kernel launch overhead 主导；
6. benchmark 没有充分 warmup 或没有 torch.cuda.synchronize 分段；
7. torch.compile / Triton fusion 没启用；
8. KAN edge bookkeeping 引入了额外 Python / autograd overhead。
```

因此 v5.5 第一优先级应是 **DepthwiseMix kernel audit**。如果 DepthwiseMix 不能被优化到接近 MLP，那么 RBF-family 很难满足终极目标。

---

### 2.3 CP-ABRBF 不是天然高效，除非写成真正低秩计算图

CP 分解的目标是：

$$
c_{oik}=\sum_{r=1}^R U_{or}V_{ir}W_{kr}.
$$

理论上可以把 full edge cost 从 $O(dhK)$ 降到与 $R(d+h+K)$ 相关。但如果实现中仍然先生成大 tensor 或用多次不连续 einsum，实际会更慢。

v5.4 中 CPRank4 forward ratio 反而达到 15.56，CPRank8 为 10.38，CPRank16 为 7.17。这不是 rank 越小越快的理想曲线，说明实现存在非理想 overhead。下一步必须把 CP 计算拆成明确的三步：

$$
T_{br}=\sum_{i,k}B_{bik}V_{ir}W_{kr},
$$

$$
y_{bo}=\sum_r T_{br}U_{or}.
$$

理想计算应通过两次连续 GEMM / bmm 完成，而不是 Python loop 或高阶 einsum fallback。

---

### 2.4 RationalKAT 的理论路线对，但当前实现/训练 recipe 不对

Rational/KAT 的理想复杂度是：

$$
O(Bd(m+n))+O(Bdh).
$$

因此它最符合最终 “MLP-like forward” 要求。但 v5.4 的 RationalKAT-AB 结果是：

```text
forward ratio = 6.99
backward ratio = 6.00
backward memory ratio = 1.35
step ratio = 4.38
accuracy 明显低于 MLP，尤其 KMNIST。
```

这说明 Rational/KAT 当前至少有两个问题：

```text
1. kernel path 没有发挥 elementwise + GEMM 的理论优势；
2. edge-decomposed RationalKAT-AB 的 task recipe / initialization / grouping 还没调好。
```

Rational/KAT 不应被放弃。它目前最接近 backward memory 目标，但要进入主线，必须先通过 kernel-level benchmark 和 accuracy repair。

---

## 3. v5.5 的核心方向：Kernel-First，不再先调 functional optimizer

v5.5 的主问题是：

$$
\boxed{
\text{哪一种 PureKAN primitive 能先满足 MLP-like efficiency envelope？}
}
$$

只有 primitive 先过效率门槛，才值得继续做 functional training / LightSmooth / geometry controller。

v5.5 不再把 Dense AB-RBF 当最终主线。Dense AB-RBF 保留为 mechanism reference。真正主线从以下三类中选择：

```text
1. Fused Depthwise-ABRBF + GEMM mixing
2. Optimized CP-ABRBF with true low-rank compute graph
3. RationalKAT-AB with GEMM-native fused activation
```

并且所有候选必须满足：

$$
\frac{T_{fwd}}{T_{MLP}}\leq 1.25,
$$

$$
\frac{M_{fwd}}{M_{MLP}}\leq 1.25,
$$

$$
\frac{T_{bwd}}{T_{MLP}}\leq 1.40,
$$

$$
\frac{M_{bwd}}{M_{MLP}}\leq 0.80
$$

作为最终目标。早期 exploratory gate 可以放宽为：

$$
\frac{T_{step}}{T_{MLP}}\leq 2.0,
\quad
\frac{M_{bwd}}{M_{MLP}}\leq 1.25.
$$

---

## 4. v5.5 实验计划总览

v5.5 分为九个阶段：

```text
P0: core / runner consistency hardening
P1: phase-separated efficiency profiler
P2: DepthwiseMix kernel repair
P3: CP-ABRBF compute-graph repair
P4: RationalKAT kernel and recipe repair
P5: efficient primitive accuracy frontier
P6: custom backward true memory audit
P7: functional / LightSmooth compatibility smoke
P8: 3-seed joint candidate selection
P9: 5-seed / 10-seed confirm
```

这不是一个大扫计划，而是一个 gate-driven pipeline。每一阶段如果不过，就不进入下一阶段。

---

## 5. P0：core / runner consistency hardening

### 5.1 目的

v5.4 结果文件说 efficient primitives 已经加入 core，但上传代码中仍然能看到旧的 `RBFDense` / `PureKANClassifier` 作为主要结构。因此 P0 必须保证实际跑实验的 core 与上传/审计代码一致。

P0 要求：

```text
ABRBFDepthwiseMixDense
CPABRBFDense
RationalKATDense
edge_named_params
base_named_params
rbf_residual_named_params
mixing_named_params
```

这些必须在 `dgkan_core.py` 中作为 core-level definitions 存在，而不是仅在 runner helper 中临时定义。

### 5.2 必跑检查

对每个 primitive：

```text
MLP-reference
RBFOnly-Dense
ABRBF-Dense
ABRBF-DepthwiseMix
ABRBF-CPRank4/8/16
RationalKAT-AB
```

记录：

```text
primitive_name
class_name
param_count_total
param_count_edge
param_count_base
param_count_rbf
param_count_mixing
param_count_nonkan
coverage_edge
coverage_base
coverage_rbf
coverage_mixing
rollback_max_abs_error
forward_output_shape
backward_grad_finite
functional_param_manifest_hash
```

### 5.3 判定标准

P0 通过条件：

```text
nonKAN = 0 for all PureKAN candidates
coverage_edge = 1.0
coverage_base = 1.0 when base exists
coverage_rbf = 1.0 when RBF residual exists
coverage_mixing = 1.0 when mixing exists
rollback_max_abs_error < 1e-8
all gradients finite
manifest hash stable across rerun
```

### 5.4 可视化

画：

```text
parameter stack bar:
  edge / base / rbf / mixing / nonKAN

coverage heatmap:
  primitive x parameter group

rollback error bar:
  primitive x max_abs_error
```

如果 P0 不过，停止全部后续实验。

---

## 6. P1：phase-separated efficiency profiler

### 6.1 目的

v5.4 的 P1 只给了 fwd/bwd/bmem/step ratio，但现在需要知道具体慢在哪里、显存在哪里。P1 要把 forward、backward、optimizer、temporary allocation 拆开。

### 6.2 基准设置

所有 primitive 统一跑：

```text
batch_size in {64, 128, 256, 512, 1024}
hidden_dim in {64, 128}
depth in {2, 4}
basis_count in {8, 16}
dtype in {fp32, amp/bf16 if available}
```

先只用 synthetic input，不跑 dataset，以避免 dataloader 噪声。

### 6.3 记录指标

每个 primitive、每个 shape 记录：

```text
forward_time_ms
backward_time_ms
optimizer_time_ms
step_time_ms
forward_peak_allocated_mb
backward_peak_allocated_mb
optimizer_peak_allocated_mb
peak_reserved_mb
activation_saved_bytes
basis_tensor_bytes
workspace_temp_bytes
param_bytes
grad_bytes
optimizer_state_bytes
num_cuda_kernels_forward
num_cuda_kernels_backward
num_exp_calls_or_equivalent
num_einsum_calls
num_gemm_calls
achieved_tfLOPs_estimate
memory_bandwidth_estimate
```

### 6.4 判定标准

早期 efficiency candidate：

$$
T_{fwd}/T_{MLP}\leq 2.0,
$$

$$
T_{bwd}/T_{MLP}\leq 2.0,
$$

$$
M_{bwd}/M_{MLP}\leq 1.25.
$$

最终 efficiency candidate：

$$
T_{fwd}/T_{MLP}\leq 1.25,
$$

$$
T_{bwd}/T_{MLP}\leq 1.40,
$$

$$
M_{bwd}/M_{MLP}\leq 0.80.
$$

### 6.5 可视化

必须画：

```text
shape scaling curve:
  hidden_dim / batch_size vs forward ratio

memory decomposition stacked bar:
  params / grads / optimizer / activations / temporary workspace

kernel count bar:
  primitive vs number of CUDA kernels

roofline-style scatter:
  arithmetic intensity vs achieved throughput

step-time breakdown:
  forward / backward / optimizer
```

P1 的目标不是立刻筛到最终方案，而是明确每个 primitive 的瓶颈类型：

```text
basis materialization bottleneck
kernel launch bottleneck
GEMM underuse
temporary allocation bottleneck
autograd saved tensor bottleneck
optimizer state bottleneck
```

---

## 7. P2：DepthwiseMix kernel repair

### 7.1 为什么优先修 DepthwiseMix

DepthwiseMix 理论上最接近 MLP 复杂度，v5.4 结果却非常慢。因此它是最值得首先 debug 的 candidate。它如果优化成功，能保留 AB-RBF 的 edge decomposition 思想，同时让主计算回到 GEMM。

### 7.2 待实现版本

比较以下实现：

```text
DWM-0-current:
  当前实现，作为 reference

DWM-1-vectorized:
  basis [B,d,K] -> channel function [B,d]
  mixing y = x_tilde @ W.T
  不使用高级 einsum

DWM-2-compiled:
  DWM-1 + torch.compile

DWM-3-fused-triton-forward:
  fused basis eval + channel reduction
  output x_tilde [B,d]
  mixing 仍用 torch GEMM

DWM-4-custom-backward-recompute:
  forward 不保存 basis
  backward recompute basis

DWM-5-streaming-backward:
  backward streaming accumulate gradients for basis/base path
```

### 7.3 记录指标

除 P1 指标外，额外记录：

```text
channel_function_time_ms
mixing_gemm_time_ms
basis_eval_time_ms
basis_recompute_time_ms
x_tilde_bytes
W_mixing_grad_bytes
basis_saved_bytes
basis_recomputed_count
```

### 7.4 判定标准

DepthwiseMix 只有在以下条件下继续进入 P5：

$$
T_{step}/T_{MLP}\leq 1.75
$$

并且：

$$
M_{bwd}/M_{MLP}\leq 1.10.
$$

如果 DWM-3 / DWM-4 仍然达不到，说明 RBF basis eval 本身太贵，Depthwise RBF 不应作为最终高效 primitive。

### 7.5 可视化

画：

```text
basis_eval vs mixing_gemm time split
DWM version vs step ratio
DWM version vs backward memory ratio
batch scaling plot
hidden_dim scaling plot
```

---

## 8. P3：CP-ABRBF compute-graph repair

### 8.1 目的

CP-ABRBF 有 accuracy signal，但 v5.4 中 rank 越小不一定越快，说明计算图没有利用低秩结构。

### 8.2 实现版本

比较：

```text
CP-0-current
CP-1-two-stage-gemm
CP-2-batched-bmm
CP-3-compiled
CP-4-custom-backward-recompute
```

推荐公式：

$$
T_{br}=\sum_{i,k}B_{bik}V_{ir}W_{kr},
$$

$$
y_{bo}=\sum_r T_{br}U_{or}.
$$

如果实现正确，复杂度应随 rank $R$ 单调增长，而不是 CPRank4 最慢。

### 8.3 记录指标

```text
rank
T_compute_ms
U_projection_ms
V_projection_ms
W_basis_projection_ms
intermediate_T_bytes
rank_scaling_slope
forward_ratio
backward_ratio
bmem_ratio
```

### 8.4 判定标准

CP 必须满足：

```text
rank4 faster than rank8 faster than rank16, allowing minor noise
rank16 accuracy gap <= 2% on MNIST/Fashion and <= 4% on KMNIST
step ratio <= 2.0 early gate
```

如果 CP 无法做到 rank-speed monotonicity，暂停 CP 路线。

---

## 9. P4：RationalKAT-AB kernel and recipe repair

### 9.1 目的

RationalKAT 的 backward memory 最接近目标，但当前速度和 accuracy 均差。v5.5 需要判断它是实现问题还是 primitive 本身不适合。

### 9.2 实现版本

比较：

```text
RK-0-current
RK-1-torch-eager-clean
RK-2-triton-fused
RK-3-compiled
RK-4-grouped-rational-plus-gemm
RK-5-custom-backward-recompute
```

### 9.3 训练 recipe repair

RationalKAT-AB 在 v5.4 accuracy 明显低，尤其 KMNIST。需要小范围调：

```text
groups in {4, 8, 16}
mode in {swish, gelu-like, identity-base}
base path in {linear, silu, linear+silu}
init scale in {0.5, 1.0, 1.5}
learning rate in {5e-4, 1e-3, 2e-3}
```

不要先做 functional update，只做 AdamW-like training，先证明 primitive 可训。

### 9.4 记录指标

```text
forward ratio
backward ratio
bmem ratio
step ratio
parameter count
rational_denominator_min
rational_denominator_p01
rational_derivative_p95
rational_double_derivative_p95
group_function_diversity
base/Rational norm ratio
ECE
accuracy
val_loss_auc
```

### 9.5 判定标准

RationalKAT 才能进入 P5，当且仅当：

$$
M_{bwd}/M_{MLP}\leq1.0
$$

并且：

$$
T_{step}/T_{MLP}\leq2.0
$$

并且 AdamW accuracy gap：

```text
MNIST <= 2%
Fashion <= 2%
KMNIST <= 4%
```

如果 RationalKAT 过 memory 但不过 task，则进入 architecture repair，不进入 functional optimizer。

---

## 10. P5：efficient primitive accuracy frontier

### 10.1 目的

只把 P2-P4 通过 early efficiency gate 的 primitive 放进 P5。P5 才跑真实任务。

### 10.2 方法

候选：

```text
MLP-AdamW-reference
RBFOnly-Dense-AdamW-reference
Dense-ABRBF-AdamW-reference
Best-DepthwiseMix-AdamW
Best-CPABRBF-AdamW
Best-RationalKAT-AdamW
```

数据：

```text
MNIST
Fashion-MNIST
KMNIST
```

先跑：

```text
seeds = 0,1,2
train_size = 6000
val_size = 1000
test_size = 1000
epochs = current compact budget
```

### 10.3 记录指标

```text
test_acc
val_loss_auc
ECE
NLL
margin_mean
margin_p10
effective_rank_input
effective_rank_block
effective_rank_output
class_centroid_separation
phi_edge_p95
phi_residual_p95
curvature_residual
jacobian_condition
base/RBF or base/nonlinear ratio
ablation_drop_base
ablation_drop_residual
forward ratio
backward ratio
bmem ratio
step ratio
```

### 10.4 判定标准

P5 survivor 必须满足：

```text
accuracy gap vs MLP <= 1.0% on MNIST/Fashion
accuracy gap vs MLP <= 2.0% on KMNIST
ECE <= MLP + 0.02
step ratio <= 2.0 early gate
bmem ratio <= 1.25 early gate
```

并且至少在一个 geometry 指标上优于 MLP：

```text
ECE better
or residual phi lower
or curvature lower
or calibration/margin more stable
```

---

## 11. P6：custom backward true memory audit

### 11.1 目的

P2 说明 saved MB 降了但 measured peak memory 没降。P6 要做更严格的 memory tracing。

### 11.2 方法

对 P5 survivor 做：

```text
Autograd
RecomputeBackward
StreamingStatsBackward
CheckpointedForward
FusedBackward, if available
```

### 11.3 记录指标

必须分 phase 记录：

```text
memory_before_forward
memory_after_forward
memory_peak_forward
memory_before_backward
memory_after_backward
memory_peak_backward
memory_after_optimizer
memory_peak_optimizer
reserved_peak
allocated_peak
activation_saved_bytes
estimated_tensor_lifetime_bytes
```

同时记录：

```text
forward_time_ms
backward_time_ms
optimizer_time_ms
step_time_ms
grad_relerr_vs_autograd
grad_cos_vs_autograd
```

### 11.4 判定标准

真正 memory win：

$$
M_{peak,bwd}^{custom} < M_{peak,bwd}^{autograd}
$$

并且：

$$
M_{peak,bwd}^{custom}/M_{MLP}\leq 1.0
$$

早期 gate 接受：

$$
M_{peak,bwd}^{custom}/M_{MLP}\leq 1.25.
$$

如果 custom backward 只减少 saved MB，不减少 peak allocated，则不能宣称 backward memory advantage。

### 11.5 可视化

```text
memory timeline plot:
  allocated MB vs phase time

peak memory stacked bar:
  forward / backward / optimizer

saved bytes vs actual peak scatter:
  x = saved MB
  y = peak allocated MB
```

---

## 12. P7：functional / LightSmooth compatibility smoke

### 12.1 目的

只有 efficient primitive 过 P5/P6 后，才测试 functional update 或 LightSmooth。否则 functional method 再好也不满足终极目标。

### 12.2 方法

对 P5/P6 survivor：

```text
AdamW baseline
AdamW + LightSmooth single event
AdamW + one-cycle LightSmooth
Functional-coordinate Adam, if compatible
Residual smoothing maintenance, low frequency only
```

### 12.3 记录指标

```text
acc_before_smooth
acc_after_smooth
acc_after_refresh
KL_teacher_student
logit_drift
residual_phi_reduction
curvature_reduction
ECE_change
LightSmooth_memory_overhead
LightSmooth_time_overhead
amortized_step_ratio
```

### 12.4 判定标准

LightSmooth compatible if：

```text
acc_after_refresh >= acc_baseline - 0.005
KL <= 0.005
logit_drift <= 0.03
residual_phi_reduction >= 0.05 or curvature_reduction >= 0.20
amortized_time_overhead <= 0.10
memory_overhead <= 1.15x candidate baseline
```

---

## 13. P8：3-seed joint candidate selection

### 13.1 目的

P8 是第一个真正 joint gate：task + geometry + efficiency。

### 13.2 方法

只选最多 3 个 candidate：

```text
Best efficient primitive AdamW
Best efficient primitive + LightSmooth
Best efficient primitive + functional-compatible update, if P7 passes
```

对比：

```text
MLP-AdamW
RBFOnly-Dense-AdamW
Dense-ABRBF-AdamW
```

数据：

```text
MNIST
Fashion-MNIST
KMNIST
```

### 13.3 记录指标

```text
test_acc
paired_acc_delta_vs_MLP
val_loss_auc
paired_auc_delta_vs_MLP
ECE
NLL
phi_residual
curvature_residual
jacobian_condition
forward_time_ratio
backward_time_ratio
backward_memory_ratio
step_time_ratio
convergence_epoch_to_target
wall_clock_to_target
```

### 13.4 判定标准

P8 survivor：

```text
accuracy >= MLP-AdamW - 0.5% on MNIST/Fashion
accuracy >= MLP-AdamW - 1.0% on KMNIST
AUC >= MLP-AdamW or within 2% with better ECE/geometry
ECE <= MLP-AdamW
forward ratio <= 1.5 exploratory gate
backward ratio <= 1.75 exploratory gate
bmem ratio <= 1.25 exploratory gate
step ratio <= 1.75 exploratory gate
```

如果没有 survivor，不进入 P9。

---

## 14. P9：5-seed / 10-seed final confirm

### 14.1 目的

只有 P8 有 survivor，才跑 P9。

### 14.2 5-seed confirm

```text
seeds = 0..4
methods:
  MLP-AdamW
  PureKAN-AdamW best efficient primitive
  PureKAN-functional or LightSmooth candidate
```

通过 5-seed 后再跑 10-seed。

### 14.3 最终成功标准

最终系统必须满足：

$$
\operatorname{Acc}_{KAN}\geq \operatorname{Acc}_{MLP-AdamW}
$$

或者至少在 paired CI 中不显著低于 MLP，同时：

$$
\operatorname{AUC}_{KAN}\geq \operatorname{AUC}_{MLP-AdamW},
$$

$$
\operatorname{ECE}_{KAN}\leq \operatorname{ECE}_{MLP-AdamW},
$$

$$
T_{fwd}/T_{MLP}\leq1.25,
$$

$$
T_{bwd}/T_{MLP}\leq1.40,
$$

$$
M_{bwd}/M_{MLP}\leq0.80.
$$

如果 task 达标但 efficiency 不达标，则判定为 mechanism success，不是 terminal system success。

---

## 15. 必须生成的图表

### 15.1 Efficiency dashboard

```text
forward ratio by primitive
backward ratio by primitive
step ratio by primitive
backward memory ratio by primitive
```

四张图必须使用相同 primitive 顺序，便于看谁是真正 bottleneck。

### 15.2 Accuracy-efficiency Pareto

横轴：

$$
\text{step time ratio}
$$

纵轴：

$$
\text{accuracy gap vs MLP}
$$

点颜色：primitive family。

点大小：backward memory ratio。

这张图决定 candidate 是否有继续价值。

### 15.3 Memory phase timeline

对每个 P6 candidate 画：

```text
allocated MB vs phase
```

并标注：

```text
forward start/end
backward start/end
optimizer step
peak allocation
```

### 15.4 Kernel breakdown plot

```text
basis eval time
channel function time
GEMM/mixing time
backward recompute time
optimizer time
```

### 15.5 Task / geometry curve

训练过程中画：

```text
epoch vs validation loss
epoch vs test/val accuracy
epoch vs ECE
epoch vs residual phi
epoch vs residual curvature
```

### 15.6 Primitive role plot

对 AB-RBF / RationalKAT-like edge 画：

```text
base norm
residual norm
mixing norm
base ablation drop
residual ablation drop
```

### 15.7 Failure taxonomy heatmap

行：primitive。

列：

```text
efficiency fail
memory fail
accuracy fail
ECE fail
geometry fail
implementation fail
```

---

## 16. v5.5 决策树

v5.5 最终不应该用一句“继续调”结束，而要严格分流：

### 情况 A：DepthwiseMix 被修到接近 MLP，并 accuracy 可接受

结论：

```text
Depthwise-ABRBF + GEMM mixing becomes main efficient PureKAN primitive.
```

下一步：在 DepthwiseMix 上接 LightSmooth / functional-coordinate update。

### 情况 B：RationalKAT 过 memory/speed，但 accuracy 差

结论：

```text
RationalKAT is the best efficiency primitive but needs architecture/recipe repair.
```

下一步：做 RationalKAT task frontier，不做 functional update。

### 情况 C：CP-ABRBF accuracy 最好，但效率仍不达标

结论：

```text
CP-ABRBF remains a mechanism/accuracy reference, not final system.
```

下一步：继续 low-rank kernel repair；不进入 functional confirm。

### 情况 D：没有 primitive 过 exploratory efficiency gate

结论：

```text
Current PureKAN edge families cannot meet terminal efficiency target.
```

下一步：暂停 functional optimizer，进入 new primitive design：spline-token KAN、piecewise-linear KAN、hashed/grouped edge functions、or rational fused kernel。

---

## 17. 最重要的原则

v5.5 的核心原则是：

$$
\boxed{
\text{先过 efficiency envelope，再谈 functional optimizer。}
}
$$

之前 AB-RBF 和 LightSmooth 的工作不是白做，它们告诉我们：

```text
1. edge decomposition 是必要的；
2. residual geometry maintenance 是可行的；
3. strong smoothing 太重，light smoothing 可以低成本；
4. dense RBF/AB-RBF 不能作为最终高效 primitive。
```

但终极目标要求接近 MLP 的 forward / backward / memory。因此 v5.5 必须以 kernel 和 primitive 为第一优先级。

