# DG-KAN v11.0：函数空间自然更新重建实验计划

> 目标：停止“动作筛选 / 动作生成 / 事后标签反推”的旧范式，回到 functional update 的第一性定义：在函数空间中直接计算一个受信任域和记忆约束保护的小更新。  
> 本计划不以 MNIST / Fashion-MNIST / KMNIST 打榜为目标；这些数据集只作为诊断压力测试。  
> 本计划不按数据集调参，不使用 teacher，不改交叉熵损失，不使用事后 outcome 标签训练控制器。  
> 所有公式均使用 Typora 友好的 `$...$` 和 `$$...$$`。

---

## 0. 为什么必须换范式

过去很多轮实验证明了一件事：好动作存在，但动作不是稳定对象。

我们之前的流程大致是：

```text
生成或收集一批候选动作；
对每个动作跑未来训练分支；
给动作贴上 FastGood / SlowBurnGood / BadPath 等标签；
再尝试用训练当下可见特征识别这些动作；
失败后再换一个分数、换一个生成器、换一个子空间。
```

这条路已经走到尽头。原因不是实验不够多，而是问题表述错了。

一个历史动作可能只在某个具体训练状态、某个优化器状态、某个 batch 顺序下有效。离开原始状态后，它可能不再是好动作。所以继续学习“哪个动作好”很容易学习到历史偶然事件。

从 v11.0 开始，functional update 不再被定义为“挑一个动作”，而定义为：

$$
\boxed{
\text{在函数空间信任域内，直接求一个最小且安全的参数更新。}
}
$$

更直白地说：

```text
不是先猜一个参数动作再看结果；
而是先问当前模型输出函数应该怎么移动，
然后求一个能实现这种函数移动的参数更新。
```

---

## 1. 新主假设

### 1.1 总假设

给定模型参数 $\theta$，模型输出为 $f_\theta(x)$。普通任务训练使用标准交叉熵，不引入 teacher、不改 loss。functional update 只改变更新方向。

我们希望求一个小参数更新 $\Delta\theta$，使得当前 batch 的输出函数沿着交叉熵下降方向移动，同时在 memory buffer、hard-tail samples、old-family samples 上不造成明显损害。

写成优化问题：

$$
\Delta\theta^*
=
\arg\min_{\Delta\theta}
\sum_{i\in B}
\ell(f_\theta(x_i)+J_i\Delta\theta,y_i)
+
\frac{1}{2\eta}\|J_R\Delta\theta\|_2^2
$$

并满足：

$$
\Delta L_{memory}(\Delta\theta)\le \epsilon_M,
$$

$$
\Delta L_{hardtail}(\Delta\theta)\le \epsilon_H,
$$

$$
\|J_R\Delta\theta\|_2\le \epsilon_f.
$$

这里：

```text
B 是当前训练 batch；
R 是函数空间信任域参考集合，可以包含当前 batch、memory buffer、hard-tail samples；
J_i 是第 i 个样本输出对参数的雅可比矩阵；
J_R 是参考集合上的雅可比矩阵；
\epsilon_M / \epsilon_H / \epsilon_f 是安全约束，不按数据集调参。
```

### 1.2 最小可实现形式

先不做复杂二次规划，第一版使用经验函数空间自然更新：

$$
K = JJ^\top,
$$

$$
r = \nabla_f L,
$$

$$
\Delta\theta
=
-J^\top(K+\lambda I)^{-1}r.
$$

这里 $J$ 是输出对参数的雅可比矩阵，$K$ 是经验输出空间核，$r$ 是输出上的交叉熵残差。

这个形式不是新造手工规则。它对应一个非常朴素的数学原则：

```text
先在输出函数空间里决定该往哪里走，
再找一个参数更新实现这个输出变化。
```

### 1.3 为什么这不是旧路线的变体

旧路线依赖：

```text
历史好动作标签；
未来路径 outcome 表；
动作池密度；
generated payload family；
FastGood / SlowBurnGood 事后标签；
手工分数阈值。
```

新路线不依赖这些。训练时只使用：

```text
当前 batch；
memory buffer；
hard-tail buffer；
模型输出；
雅可比向量积；
向量雅可比积；
函数空间信任域。
```

未来路径仍然记录，但只作为验证，不作为训练标签。

---

## 2. 实验整体目标

v11.0 不追求一轮直接 system pass。它要回答四个第一性问题。

### 问题 1：函数空间自然更新是否能在普通多层感知机上工作？

普通多层感知机是更简单、更成熟的神经网络。如果新 functional update 在普通多层感知机上都不能带来稳定信号，那么它不是 KAN 问题，而是 update 理论本身不成立。

### 问题 2：函数空间自然更新是否能迁移到当前边函数网络？

如果普通多层感知机上有效，但边函数网络上无效，说明当前边函数网络的手写梯度、核路径、basis 表示或 kernel-native 实现还有问题。

### 问题 3：这个更新是否真的进入信号方向，而不是当前 batch 噪声？

要用 split-batch、memory buffer、label-noise 对照检查：更新不能只让当前 batch 变好，也不能让随机标签噪声方向受益。

### 问题 4：这个更新是否值得进入长期训练路径？

只有当单步函数空间更新满足正确性、信号性、安全性、成本门槛后，才允许跑 h20 / h80 / h240 未来训练路径。

---

## 3. 实验结构总览

v11.0 分成八个阶段，三类 runner 并行。

```text
工程门 runner：验证雅可比、核求解、数值正确性、成本。
科学发现 runner：验证函数空间更新是否有单步和短程机制。
官方门 runner：只有前面过硬门，才进入未来路径、controller、runtime、paired replay。
```

阶段如下：

```text
P0：冻结旧路线，建立新路线边界；
P1：实现函数空间自然更新核心模块；
P2：单步函数空间正确性验证；
P3：信号方向与噪声水库诊断；
P4：memory / hard-tail 约束求解；
P5：短程训练 smoke；
P6：边函数网络 graph-free 版本移植；
P7：未来路径验证；
P8：通过才进入 controller / runtime / paired replay。
```

---

## 4. P0：冻结旧路线，建立边界

### 4.1 目标

确认 v10.8 的失败边界，不再继续小修旧动作路线。

必须明确停止：

```text
继续调 FPO / FPAU / GoodCone / LFRO 阈值；
继续扩大 natural action panel；
继续新增 generated action family；
继续用 FastGood / SlowBurnGood outcome 标签训练选择器；
继续寻找固定好子空间；
继续把 known-good action 当稳定对象。
```

### 4.2 必须记录

```text
source_boundary_version
known_good_direct_replay_pass
known_good_direct_replay_fastslow_like_count
state_hash_match_rate
optimizer_state_hash_match_rate
batch_sequence_hash_match_rate
payload_hash_match_rate
natural_density_status
best_fpo_precision
best_generated_fastslow_precision
no_fake
no_proxy
no_cpu_offload
```

### 4.3 判断标准

P0 通过条件：

```text
v10.8 boundary reproduced = 1；
known-good direct replay failure acknowledged = 1；
old action-selection route frozen = 1；
no_fake / no_proxy / no_cpu_offload = 1。
```

### 4.4 不满足时 Codex 先做什么

```text
如果 v10.8 artifact 找不到：
  先修 artifact path 和 route_decision 加载，不进入 P1。

如果 payload hash 不匹配：
  先修 payload replay identity，不进入函数空间更新。

如果 state / optimizer / batch sequence hash 没有记录：
  补 state identity ledger，不允许继续解释 known-good 失败。

如果旧路线仍被 runner 自动打开：
  加 hard stop flag：disable_action_selector_routes = true。
```

---

## 5. P1：实现函数空间自然更新核心模块

### 5.1 目标

实现一个最小可用模块：`functional_space_natural_step.py`。

它不生成动作、不读 outcome 标签、不使用数据集特化规则。它只做一件事：

```text
给定模型、当前 batch、memory buffer 和 hard-tail buffer，
计算一个函数空间信任域内的小参数更新。
```

### 5.2 必须实现的接口

```text
forward_outputs(model, batch)
  返回 logits 和必要的轻量缓存。

vjp_outputs_to_params(model, batch, output_vector)
  计算 J^T u，也就是把输出空间向量拉回参数空间。

jvp_params_to_outputs(model, batch, param_vector)
  计算 J v，也就是参数扰动造成的输出扰动。

solve_functional_step(model, train_batch, memory_batch, hard_tail_batch)
  求解函数空间自然更新。
```

### 5.3 先支持的模型

```text
普通多层感知机，使用 PyTorch 自动微分作 discovery reference；
当前 LQ 边函数网络，先允许 discovery reference；
graph-free 边函数网络，作为官方候选路径，不能使用 loss.backward。
```

这里要严格区分 discovery 和 official：

```text
普通多层感知机 discovery 可以使用 autograd 计算 JVP/VJP；
边函数网络 official 必须走手写解析反向传播或显式标记为 discovery-only。
```

### 5.4 必须记录

```text
model_type
parameter_count
batch_size
output_dim
reference_set_size
jvp_available
vjp_available
jvp_vjp_consistency_error
finite_difference_jvp_error
kernel_matrix_size
kernel_condition_number
kernel_rank
ridge_lambda
cg_iteration_count
cg_residual_norm
update_norm_parameter_space
update_norm_function_space
step_cost_ms
peak_memory_mb
```

### 5.5 判断标准

P1 discovery pass：

```text
finite_difference_jvp_error <= 1e-3；
jvp_vjp_consistency_error <= 1e-3；
cg_residual_norm <= 1e-4；
no NaN / Inf；
step_cost_ms <= 3x AdamW step for discovery。
```

P1 official candidate pass：

```text
loss.backward = 0；
uses_torch_autograd_graph = 0；
jvp/vjp correctness pass；
step_cost_ms <= 1.5x AdamW step or marked as periodic-only。
```

### 5.6 可视化

```text
jvp finite-difference predicted-vs-actual scatter；
vjp dot-product consistency scatter；
kernel eigenvalue spectrum；
CG residual vs iteration；
function-space update norm vs parameter-space update norm；
step cost breakdown bar。
```

### 5.7 不满足时 Codex 先做什么

```text
如果 JVP finite difference 不过：
  检查 logits scaling；
  降低 perturbation epsilon；
  对单层线性模型做单元测试；
  对每个参数 block 单独检查 JVP。

如果 VJP/JVP dot consistency 不过：
  检查 flatten/unflatten 参数顺序；
  检查 requires_grad / no_grad 边界；
  检查 batch reduction 是 sum 还是 mean。

如果 kernel condition number 太高：
  增加 ridge damping；
  对 logits 做中心化；
  使用低秩截断；
  限制 reference set size。

如果 cost 过高：
  降低 reference set size；
  减少 conjugate gradient iterations；
  使用 Nyström / random feature sketch；
  设置每 N 步一次 functional update。
```

---

## 6. P2：单步函数空间正确性验证

### 6.1 目标

验证求出来的 $\Delta\theta$ 是否真的实现预期输出变化，并且是否比同范数 AdamW / 随机扰动更合理。

### 6.2 实验对象

```text
普通多层感知机；
当前 LQ 边函数网络；
如果可用，graph-free 边函数网络。
```

任务：

```text
MNIST-like；
Fashion-MNIST-like；
KMNIST-like；
synthetic interaction；
一个非视觉 tabular 任务；
一个 label-noise 任务。
```

这些任务只作为诊断，不允许为某个任务调单独阈值。

### 6.3 对照组

```text
AdamW one step；
AdamW same function norm；
随机参数扰动 same parameter norm；
随机函数扰动 same function norm；
SAM-like sharpness perturbation；
SOAP-like block preconditioned update，如果已有实现；
Muon-like orthogonalized matrix update，如果已有实现。
```

### 6.4 必须记录

```text
method
model_type
dataset
seed
train_batch_loss_before
train_batch_loss_after
heldout_batch_loss_before
heldout_batch_loss_after
memory_loss_before
memory_loss_after
hard_tail_loss_before
hard_tail_loss_after
predicted_output_delta_norm
actual_output_delta_norm
predicted_actual_delta_cosine
predicted_loss_delta
actual_loss_delta
actual_heldout_loss_delta
memory_loss_delta
hard_tail_loss_delta
parameter_update_norm
function_update_norm
cosine_with_adamw_update
cosine_with_random_update
step_cost_ms
peak_memory_mb
```

### 6.5 判断标准

P2 weak pass：

```text
predicted_actual_delta_cosine >= 0.90；
train_batch_loss_delta < 0；
heldout_batch_loss_delta <= 0 on at least 60% trials；
memory_loss_delta <= tolerance；
hard_tail_loss_delta <= tolerance；
function-space update beats random same-norm perturbation。
```

P2 strong pass：

```text
predicted_actual_delta_cosine >= 0.95；
heldout_batch_loss_delta <= 0 on at least 75% trials；
memory/hard-tail no harm on at least 90% trials；
beats AdamW same function norm on heldout loss or memory safety；
step_cost_ms <= 1.5x AdamW, or periodic amortized cost <= 1.5x。
```

### 6.6 可视化

```text
predicted vs actual output delta scatter；
train/heldout/memory/hard-tail loss delta box plot；
cosine with AdamW histogram；
function norm vs loss improvement scatter；
method comparison Pareto: heldout loss delta vs memory loss delta；
cost vs improvement scatter。
```

### 6.7 不满足时 Codex 先做什么

```text
如果 predicted_actual_delta_cosine 低：
  降低 trust radius；
  增加 damping；
  检查 nonlinearization error；
  分层记录每个 layer 的 output drift。

如果 train loss 降但 heldout loss 升：
  加 split consistency；
  减小 update radius；
  使用 signal-channel eigenmode filtering；
  不调 dataset-specific threshold。

如果 memory/hard-tail 受伤：
  改为 constrained solve；
  memory batch 加入 reference set；
  hard-tail batch 加入 reference set；
  禁止简单 hard veto 后重跑同一方向。

如果和 AdamW cosine 接近 1：
  说明只是 AdamW 变体；
  加函数空间 trust metric或 signal filtering。

如果 random same-norm 也一样好：
  说明指标太弱；
  加 memory/hard-tail/heldout split/noise-label 对照。
```

---

## 7. P3：信号方向与噪声水库诊断

### 7.1 目标

验证函数空间更新是否沿跨样本稳定信号方向移动，而不是利用当前 batch 的噪声。

### 7.2 方法

把当前 batch 切成多个子集：

```text
split_1, split_2, ..., split_k
```

每个 split 计算一个函数空间更新方向，比较这些方向的一致性。

定义：

$$
\text{SignalRatio}(u)
=
\frac{\|\mathbb{E}_s[u_s]\|_2^2}{\mathbb{E}_s\|u_s-\bar{u}\|_2^2+\epsilon}.
$$

这里 $u_s$ 是第 $s$ 个 split 的函数空间更新方向。

### 7.3 必须记录

```text
split_count
split_size
per_split_function_direction_norm
mean_function_direction_norm
function_direction_variance
signal_ratio
heldout_split_loss_delta
noise_label_split_loss_delta
memory_loss_delta
hard_tail_loss_delta
kernel_top_eigenvalues
signal_eigenmode_count
reservoir_eigenmode_count
update_energy_in_signal_channel
update_energy_in_reservoir
```

### 7.4 判断标准

P3 weak pass：

```text
signal_ratio > random_direction_signal_ratio by 2x；
heldout split loss improves in at least 60% trials；
noise-label split does not improve more than real-label split；
memory/hard-tail not harmed。
```

P3 strong pass：

```text
signal_ratio > random by 3x；
heldout split loss improves in at least 75% trials；
update_energy_in_signal_channel >= 70%；
update_energy_in_reservoir <= 30%；
noise-label split benefit <= 20% of real-label benefit。
```

### 7.5 可视化

```text
signal ratio histogram；
function direction cosine matrix across splits；
kernel eigenvalue spectrum with signal/reservoir split；
update energy by eigenmode；
real-label vs noise-label loss delta bar；
heldout improvement by split count。
```

### 7.6 不满足时 Codex 先做什么

```text
如果 signal ratio 低：
  增加 split count；
  使用 larger reference set；
  对输出 residual 做 centering；
  只保留 top stable eigenmodes。

如果 noise-label split 也改善：
  说明更新进入 noise/reservoir；
  加强 split consistency；
  提高 damping；
  引入 memory/hard-tail reference。

如果 signal eigenmodes 太少：
  降低 batch noise；
  使用 accumulated reference batch；
  不要按数据集调阈值。

如果 reservoir energy 高：
  对低 eigenvalue modes 做 shrinkage；
  使用 truncated kernel inverse。
```

---

## 8. P4：memory / hard-tail 约束求解

### 8.1 目标

把 memory/offdiag 从“手工 veto”改成函数空间约束。

过去 memory/offdiag 是有效安全信号，但不是收益源。现在它只作为约束：

$$
\Delta L_{memory}(\Delta\theta)\le \epsilon_M,
$$

$$
\Delta L_{hardtail}(\Delta\theta)\le \epsilon_H.
$$

### 8.2 候选求解器

```text
unconstrained_functional_step；
memory_in_reference_set；
soft_lagrange_memory_penalty；
projected_constrained_step；
small_quadratic_program_on_output_space；
trust_region_with_memory_barrier。
```

### 8.3 必须记录

```text
solver_type
constraint_type
memory_loss_delta
hard_tail_loss_delta
train_loss_delta
heldout_loss_delta
constraint_violation_count
lambda_memory
lambda_hardtail
trust_radius
solver_iterations
solver_residual
step_cost_ms
```

### 8.4 判断标准

P4 weak pass：

```text
memory_loss_delta <= tolerance in at least 85% trials；
hard_tail_loss_delta <= tolerance in at least 85% trials；
heldout loss still improves in at least 55% trials；
constraint solver cost <= 3x AdamW in discovery。
```

P4 strong pass：

```text
memory/hard-tail no-harm in at least 95% trials；
heldout loss improves in at least 70% trials；
constraint solver does not collapse train improvement；
amortized cost <= 1.5x AdamW if run every N steps。
```

### 8.5 可视化

```text
train improvement vs memory harm Pareto；
heldout improvement vs hard-tail harm Pareto；
constraint violation over steps；
lambda_memory trajectory；
trust radius vs output drift。
```

### 8.6 不满足时 Codex 先做什么

```text
如果 memory harm 高：
  include memory outputs in kernel reference；
  increase memory penalty；
  reduce trust radius；
  switch from hard projection to constrained solve。

如果 constraints kill all train improvement：
  record conflict angle between train residual and memory residual；
  split layers and check where conflict is concentrated；
  allow smaller update rather than changing threshold。

如果 hard-tail harm persists：
  increase hard-tail sampling diversity；
  use CEp99-focused reference set；
  check if hard-tail set is stale。

如果 cost high：
  solve in output space not parameter space；
  use low-rank kernel sketch；
  amortize every 5 or 10 training steps。
```

---

## 9. P5：短程训练 smoke

### 9.1 目标

只有 P1-P4 至少 weak pass 后，才允许进入短程训练。目标不是证明最终超越普通多层感知机，而是判断函数空间自然更新是否有稳定训练价值。

### 9.2 方法

比较：

```text
普通 AdamW；
AdamW + function-space natural update every step；
AdamW + function-space natural update every 5 steps；
AdamW + function-space natural update every 10 steps；
AdamW + same-norm random function-space perturbation；
AdamW + signal-channel-filtered function-space update；
AdamW + memory-constrained function-space update。
```

模型：

```text
普通多层感知机；
当前 LQ 边函数网络；
如果 P6 可用，graph-free 边函数网络。
```

任务：

```text
MNIST-like；
Fashion-MNIST-like；
KMNIST-like；
synthetic interaction；
label-noise task；
非视觉 tabular task。
```

### 9.3 必须记录

```text
method
model_type
dataset
seed
update_period
train_loss_curve
val_loss_curve
test_acc_curve
ECE_curve
NLL_curve
Brier_curve
margin_p10_curve
CEp99_curve
memory_loss_curve
hard_tail_loss_curve
time_to_target_loss
val_loss_auc_step
val_loss_auc_time
step_time_ms
amortized_functional_update_cost_ms
peak_memory_mb
bad_step_rate
loss_spike_count
```

### 9.4 判断标准

P5 weak pass：

```text
在普通多层感知机上，至少 3 个任务中有 2 个满足：
  val_loss_auc_time 不差于 AdamW；
  memory/hard-tail 不差；
  calibration 不差；
  same-norm random control 不同样有效。
```

P5 strong pass：

```text
普通多层感知机至少 4 个任务中有 3 个通过；
边函数网络至少 3 个任务中有 2 个通过；
label-noise task 上，noise-label benefit 显著低于 real-label benefit；
amortized step ratio <= 1.5。
```

### 9.5 可视化

```text
validation loss vs step；
validation loss vs wall-clock time；
accuracy vs wall-clock time；
ECE vs step；
memory/hard-tail loss vs step；
time-to-target bar；
same-norm random control comparison；
functional update cost amortization plot。
```

### 9.6 不满足时 Codex 先做什么

```text
如果普通多层感知机失败：
  停止 KAN functional update；
  回到函数空间自然更新公式、damping、signal filtering。

如果普通多层感知机成功但 KAN 失败：
  检查 KAN JVP/VJP correctness；
  检查 KAN kernel condition；
  检查 basis / primitive / graph-free path；
  不继续调数据集阈值。

如果 step cost 太高：
  增大 update period；
  降低 reference size；
  使用 low-rank sketch；
  尝试 blockwise kernel。

如果 memory/hard-tail 不稳：
  回到 P4 constrained solve；
  禁止扩大训练预算掩盖风险。
```

---

## 10. P6：边函数网络 graph-free 版本移植

### 10.1 目标

如果 P5 在普通多层感知机上通过，必须把函数空间自然更新迁移到 graph-free 边函数网络路径。否则它只是普通 autograd 原型，不是项目目标。

### 10.2 必须记录

```text
uses_loss_backward
uses_torch_autograd_graph
manual_forward_available
manual_backward_available
manual_jvp_available
manual_vjp_available
jvp_error_vs_reference
vjp_error_vs_reference
manual_cache_mb
forward_time_ratio_vs_mlp
backward_time_ratio_vs_mlp
step_time_ratio_vs_mlp
functional_update_cost_ms
amortized_step_ratio
```

### 10.3 判断标准

P6 weak pass：

```text
loss.backward = 0；
manual JVP/VJP correctness pass；
functional update discovery works on at least one task；
amortized step ratio <= 2.0。
```

P6 strong pass：

```text
loss.backward = 0；
manual JVP/VJP correctness pass；
P5 strong behavior reproduces on edge-function network；
amortized step ratio <= 1.5；
backward memory <= MLP reference。
```

### 10.4 不满足时 Codex 先做什么

```text
如果 manual JVP/VJP 不正确：
  回到 primitive-level analytic adjoint tests；
  compare against autograd on tiny shapes；
  isolate edge parameter / base parameter / residual parameter roles。

如果 efficiency fails but correctness works：
  optimize kernel/cache；
  use blockwise solve；
  amortize periodic updates；
  do not claim system success。

如果 KAN task fails but MLP works：
  test compositional FullEdge base；
  test primitive trainability first；
  do not patch functional update.
```

---

## 11. P7：未来路径验证

### 11.1 目标

只有 P5/P6 至少 weak pass 后，才看未来路径。未来路径不再用于训练，只用于验证。

### 11.2 记录指标

```text
method
model_type
dataset
seed
V1
V5
V20
V80
V240
RAUV
longrisk
bad_event
null_event
memory_fail
offdiag_fail
FastGood
SlowBurnGood
RiskyHighAUV
SafeLowValue
BadPath
paired_adamw_delta
paired_random_delta
```

### 11.3 判断标准

P7 weak pass：

```text
FastGood + SlowBurnGood precision >= 0.25 among tested updates；
V240 LCB > 0；
longrisk UCB <= 0.10；
memory/offdiag UCB <= 0.10；
beats same-norm random perturbation。
```

P7 strong pass：

```text
FastGood + SlowBurnGood precision >= 0.50；
V240 LCB > 0；
longrisk UCB <= 0.05；
bad/null UCB <= 0.10；
memory/offdiag UCB <= 0.05；
beats AdamW-only paired replay。
```

### 11.4 可视化

```text
V1/V5/V20/V80/V240 path curves；
FastGood / SlowBurnGood / RiskyHighAUV counts；
longrisk vs V240 scatter；
paired AdamW comparison；
update period vs future path quality。
```

### 11.5 不满足时 Codex 先做什么

```text
如果 V1 好但 V240 坏：
  update still immediate-descent dominated；
  strengthen signal-channel filtering；
  add memory/hard-tail constraints；
  reduce trust radius。

如果 V240 好但 longrisk 高：
  use constrained solve；
  record conflict between value and risk residual；
  do not accept high-risk path。

如果 Fast/Slow still zero on MLP:
  functional space update paradigm fails；
  stop and return to optimizer/architecture theory。

If MLP passes but KAN fails:
  return to primitive / graph-free / compositional architecture line.
```

---

## 12. P8：官方 controller / runtime / paired replay

### 12.1 目标

只有 P7 strong pass 才允许进入 official path。否则不再提前开 controller。

### 12.2 必须记录

```text
accepted_count
precision
V240_LCB
RAUV_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
leave_dataset_out_drop
leave_stratum_out_drop
leave_template_out_drop
runtime_step_ratio_q90
paired_replay_vs_adamw
paired_replay_vs_best_lr
paired_replay_vs_noop
paired_replay_vs_random
shuffled_update_control
```

### 12.3 判断标准

```text
accepted_count >= 87；
precision >= 0.75；
V240_LCB > 0；
longrisk_UCB <= 0.05；
bad_UCB <= 0.05；
null_UCB <= 0.15；
memory_UCB <= 0.05；
offdiag_UCB <= 0.05；
leaveout drops <= 0.10；
runtime_step_ratio_q90 <= 1.50；
paired replay beats AdamW / best learning rate / no-op / random；
shuffled update control fails。
```

### 12.4 不满足时 Codex 先做什么

```text
如果 accepted_count 不够：
  do not lower quality threshold blindly；
  increase tested states / update periods；
  check if update is too rare or too costly。

如果 precision low：
  go back to P7; controller not ready。

如果 runtime fails：
  amortize update period；
  lower sketch rank；
  blockwise solve；
  periodic update only。

如果 paired replay fails：
  do not claim functional success；
  analyze whether update differs from AdamW or random.
```

---

## 13. 总体 stop rules

### Stop Rule A：普通多层感知机失败

如果 P1-P5 在普通多层感知机上都失败：

```text
停止当前 functional update 重建；
回到理论公式、函数空间核、自然梯度、信号通道理论；
不要继续 KAN 实验。
```

### Stop Rule B：普通多层感知机成功，边函数网络失败

如果普通多层感知机成功，边函数网络失败：

```text
functional update 方向保留；
边函数网络 primitive / graph-free / kernel-native / compositional base 是 blocker；
回到基础结构，不继续调 functional update。
```

### Stop Rule C：效果好但成本太高

如果 P5/P7 有信号但成本过高：

```text
转向 blockwise / Kronecker / low-rank / orthogonalized approximation；
不写成 system success；
不继续跑 full controller。
```

### Stop Rule D：P7 无未来路径收益

如果单步和短程都好，但未来路径失败：

```text
说明函数空间更新仍是 immediate descent；
需要重新加入未来动力学约束；
不要用 Fast/Slow 标签训练 selector。
```

---

## 14. 最重要的可视化清单

```text
1. JVP finite difference correctness scatter；
2. VJP/JVP dot consistency plot；
3. empirical function kernel eigen spectrum；
4. predicted vs actual output delta scatter；
5. train / heldout / memory / hard-tail loss delta box plot；
6. signal vs reservoir eigenmode energy plot；
7. real-label vs noise-label improvement plot；
8. memory-harm vs heldout-improvement Pareto；
9. validation loss vs wall-clock time；
10. functional update amortized cost plot；
11. V1/V5/V20/V80/V240 future path curve；
12. paired AdamW / random / no-op comparison；
13. runtime step ratio dashboard。
```

---

## 15. 这轮计划的最终判断逻辑

v11.0 必须输出以下 route 之一。

### Route A：函数空间自然更新在普通多层感知机和边函数网络上都有效

```text
P5 strong pass；
P6 weak/strong pass；
P7 future path pass。
```

下一步进入 official controller / runtime / paired replay。

### Route B：普通多层感知机有效，边函数网络失败

```text
P5 pass；
P6 fail。
```

结论：functional update 理论有希望，但当前边函数网络 primitive / graph-free / kernel-native path 不支持。回到基础结构路线。

### Route C：普通多层感知机也失败

```text
P5 fail on MLP。
```

结论：当前函数空间自然更新范式失败，不再继续 KAN 版本。

### Route D：效果存在但成本过高

```text
P5/P7 pass；
runtime fails。
```

结论：转向 Kronecker / low-rank / orthogonalized approximation。

### Route E：单步有效但未来路径失败

```text
P2/P5 pass；
P7 fail。
```

结论：仍是 immediate descent，不是真正 future-path functional update。

---

## 16. 为什么这是“范式实验”而不是“小修”

这轮不再问：

```text
哪个动作好？
哪个生成器好？
哪个分数能选动作？
哪个阈值能过 gate？
```

这轮只问：

```text
当前模型的输出函数应该怎么移动？
这个函数移动能否由参数更新实现？
这个移动是否跨样本稳定？
这个移动是否不伤 memory / hard-tail？
这个移动是否能在短程和未来路径上胜过 AdamW / random controls？
这个方法能否迁移到 graph-free 边函数网络？
```

所以它是 functional update 重建，而不是 action selection 继续。

---

## 17. 最终一句话

$$
\boxed{
\text{v11.0 的核心不是再找好动作，而是在函数空间里直接求一个安全、稳定、可验证的更新。}
}
$$

如果这条路失败，我们就应该诚实停止当前 functional update 主线，回到基础结构、组合式边函数网络和 kernel-native graph-free training；如果这条路成功，再谈 controller、runtime 和 external advantage。
