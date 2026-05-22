# DG-KAN v11.1 Function-Space Natural Update：短程失败诊断与训练动力学重建实验计划

> 目标：v11.1 不继续旧的动作筛选、好动作标签、生成动作、FastGood / SlowBurnGood 反推路线。v11.1 只回答一个问题：为什么 v11.0 中函数空间自然更新已经通过数值正确性、单步预测、安全约束，却在短程训练中完全没有通过？如果这个问题不能在普通多层感知机上解决，当前 functional update 重建路线必须暂停，不进入 KAN。

---

## 0. 当前结论与路线重置

v11.0 的核心结果是：

```text
route = RouteC-MLPFunctionalUpdateFails
primary_blocker = mlp_function_space_update_not_validated
secondary_blocker = short_training_smoke_not_passed
system_legal_controller_pass = 0
```

v11.0 不是旧路线的小失败。它证明了一个非常重要的边界：

```text
函数空间自然步的数值内核可以正确实现；
单步线性预测也能和真实输出变化高度一致；
memory / hard-tail 约束可以在小样本诊断里过；
但一进入短程训练，任务层面 0 / 6 通过。
```

因此 v11.1 的判断是：

$$
\boxed{
\text{当前失败不是 JVP/VJP 数值错误，而是函数空间自然步没有形成稳定训练动力学。}
}
$$

也就是说，下一步不能继续问：

```text
能不能调一个阈值？
能不能换一个动作族？
能不能把 P5 预算加大？
能不能先试 KAN？
```

必须问：

```text
为什么一个单步看起来正确的函数空间自然更新，重复训练时不能稳定改善验证损失？
```

---

## 1. v11.1 总体目标

v11.1 的整体目标不是 system pass，也不是 KAN pass，而是把 v11.0 的短程训练失败拆成可判定原因。

核心问题是：

$$
\boxed{
\text{函数空间自然更新到底是有效的训练更新，还是只是一阶局部拟合现象？}
}
$$

v11.1 必须在普通多层感知机上先回答这个问题。普通多层感知机是更简单、更成熟的网络。如果在普通多层感知机上都不能稳定优于 AdamW 或至少不输随机同范数扰动，那么不应该继续进入 KAN。

---

## 2. 核心假设

### H1：短程训练失败来自信任域半径 / 阻尼 / 更新频率错误

函数空间自然步在单步线性化里有效，但反复执行后，可能因为更新过大、阻尼不足、核矩阵病态、参考集太小，导致训练轨迹偏离。

要验证：

$$
\Delta\theta = -J^\top (K + \lambda I)^{-1} r
$$

是否需要更小 trust radius、更大 damping、周期性执行、或者线性化误差控制。

### H2：短程训练失败来自信号方向和噪声方向没有分开

函数空间自然步可能降低当前 batch loss，但把更新能量放进 reservoir / noise direction，而不是可泛化 signal direction。

要验证：

$$
\text{real-label benefit} > \text{noise-label benefit}
$$

以及：

$$
\text{heldout split improvement} > 0
$$

是否在多个 batch split 上稳定成立。

### H3：函数空间自然步与 AdamW 动力学冲突

AdamW 已经在参数空间积累 momentum / second moment。额外函数空间更新可能破坏 AdamW 的动量状态，使后续优化器状态与参数位置不一致。

要验证：

```text
AdamW + function-space additive update
AdamW with function-space preconditioned gradient
function-space update only
periodic function-space update
AdamW moment reset / transported moment
```

哪种机制导致短程失败。

### H4：memory / hard-tail 安全通过小样本诊断，但训练中仍累积损伤

P4 可以在单步约束里过，但多步训练中 memory / hard-tail 损伤可能累积。

要验证：

$$
\sum_t \Delta L_{memory,t}
$$

和

$$
\max_t CEp99_{hard-tail,t}
$$

是否随训练累积恶化。

### H5：函数空间自然步只是昂贵的 AdamW 近似

如果它和 AdamW 更新方向高度一致，或者随机同范数扰动同样有效，那么它不是一个新的 functional update 范式。

要验证：

$$
\cos(\Delta\theta_{FSN}, \Delta\theta_{AdamW})
$$

以及同范数随机函数扰动对照。

---

## 3. 实验总结构

v11.1 分成三个并行 runner。

```text
Engineering Runner:
  只验证数值、成本、核条件、JVP/VJP、函数输出线性化误差。

Science Runner:
  在普通多层感知机上验证单步到短程训练的动力学问题。

Official Gate Runner:
  只有普通多层感知机短程训练过硬门，才允许进入 KAN / graph-free / future path。
```

---

## 4. P0：复现 v11.0 边界并冻结旧路线

### 4.1 目标

确认本轮不是回到动作筛选路线，而是继续函数空间自然更新路线。

### 4.2 必须记录

```text
source_route_v1100
P1_correctness_pass
P1_discovery_pass
P1_official_candidate_pass
P2_weak_pass
P3_weak_pass
P4_strong_pass
P5_task_pass_count
P5_weak_pass
P6_P7_P8_status
fake_data_used
proxy_row_used
cpu_offload_used
```

### 4.3 判断标准

必须满足：

```text
source_route_v1100 = RouteC-MLPFunctionalUpdateFails
P1_correctness_pass = 1
P2_weak_pass = 1
P4_strong_pass = 1
P5_weak_pass = 0
old_action_selection_routes_disabled = 1
generated_action_routes_disabled = 1
```

### 4.4 不满足时 Codex 先做什么

```text
如果 source route 不一致：
  停止本轮，重新读取 v11.0 manifest。

如果 P1/P2/P4 没有复现：
  先修复 JVP/VJP / solver / constrained solve，不进入训练。

如果旧 action route 仍被调用：
  删除或 hard-gate FPO / FPAU / GoodCone / LFRO / generated-action 入口。
```

---

## 5. P1：单步函数空间自然更新稳定性矩阵

### 5.1 目标

确定函数空间自然步的单步成功是否依赖某个脆弱设置。这里不看长训练，只看一个 batch 上的线性化、实际输出变化、heldout、memory、hard-tail。

### 5.2 实验变量

```text
reference_size = 2, 4, 8, 16, 32
trust_radius = 0.01, 0.03, 0.1, 0.3, 1.0
damping_lambda = 1e-4, 1e-3, 1e-2, 1e-1, 1.0
kernel_rank = full, 4, 8, 16, 32
update_period = 1, 5, 10
```

不要按数据集调参。所有配置统一用于所有任务。

### 5.3 必须记录

```text
model_type
dataset
seed
batch_id
reference_size
trust_radius
damping_lambda
kernel_rank
kernel_condition_number
kernel_effective_rank
cg_iterations
predicted_output_delta_cosine
actual_output_delta_cosine
predicted_actual_delta_error
train_loss_delta
heldout_loss_delta
memory_loss_delta
hard_tail_CEp99_delta
same_norm_random_loss_delta
adamw_loss_delta
cosine_with_adamw
functional_update_cost_ms
amortized_cost_ratio
nan_inf_count
```

### 5.4 判断标准

P1 weak pass：

```text
predicted_actual_delta_cosine >= 0.95；
heldout_loss_delta <= 0 in at least 60% trials；
functional update beats same-norm random in at least 70% trials；
memory_loss_delta <= tolerance in at least 85% trials；
hard_tail_CEp99_delta <= tolerance in at least 85% trials。
```

P1 strong pass：

```text
predicted_actual_delta_cosine >= 0.98；
heldout_loss_delta <= 0 in at least 75% trials；
functional update beats same-norm random in at least 85% trials；
memory/hard-tail noharm >= 95%；
amortized_cost_ratio <= 1.5 using update_period >= 5。
```

### 5.5 可视化

```text
predicted vs actual output delta scatter；
heldout improvement vs trust radius heatmap；
heldout improvement vs damping heatmap；
kernel eigen spectrum；
kernel condition vs train/heldout delta scatter；
functional update vs same-norm random box plot；
cosine with AdamW distribution；
functional update cost amortization curve。
```

### 5.6 不满足时 Codex 先做什么

```text
如果 predicted_actual_delta_cosine 低：
  检查 JVP/VJP flatten 顺序；
  降低 trust_radius；
  增大 damping_lambda；
  检查 logits scaling；
  先用 finite difference 小模型复现。

如果 train 改善但 heldout 不改善：
  增大 reference_size；
  做 split-batch signal filter；
  移除低信号 eigenmodes；
  检查是否过拟合当前 batch。

如果 memory/hard-tail 受伤：
  把 memory/hard-tail 输出加入 kernel reference；
  从 post-hoc veto 改成 constrained solve；
  降低 trust_radius。

如果 cost 太高：
  使用 low-rank kernel sketch；
  使用 blockwise output kernel；
  提高 update_period；
  禁止进入官方路径，只保留 discovery。
```

---

## 6. P2：信号方向与噪声水库诊断

### 6.1 目标

确定函数空间自然步是否真的进入跨样本稳定的信号方向，而不是进入当前 batch 噪声方向。

### 6.2 方法

把同一个 batch 切成多个 split。每个 split 计算一个函数空间方向，再比较 split 之间的一致性。

同时加入 label-noise control：

```text
real-label batch；
random-label batch；
partial-noise batch；
heldout clean split；
memory split；
hard-tail split。
```

### 6.3 必须记录

```text
split_count
real_label_signal_ratio
random_label_signal_ratio
partial_noise_signal_ratio
heldout_improved_rate
noise_label_improved_rate
signal_mode_energy_top1
signal_mode_energy_top5
reservoir_mode_energy
train_to_heldout_transfer_rate
memory_harm_rate
hardtail_harm_rate
```

### 6.4 判断标准

P2 weak pass：

```text
real_label_signal_ratio >= 2 * random_label_signal_ratio；
heldout_improved_rate >= 0.65；
noise_label_improved_rate <= 0.35；
memory_harm_rate <= 0.15；
hardtail_harm_rate <= 0.15。
```

P2 strong pass：

```text
real_label_signal_ratio >= 4 * random_label_signal_ratio；
heldout_improved_rate >= 0.75；
noise_label_improved_rate <= 0.20；
memory/hardtail harm <= 0.05。
```

### 6.5 可视化

```text
real vs random signal ratio bar；
signal eigenmode energy spectrum；
heldout improvement vs signal ratio scatter；
noise-label benefit distribution；
reservoir energy vs memory harm scatter。
```

### 6.6 不满足时 Codex 先做什么

```text
如果 random_label_signal_ratio 接近 real_label_signal_ratio：
  当前方法没有分开 signal 和 noise；
  增加 split consistency filter；
  只保留跨 split 方向一致的 eigenmodes。

如果 heldout_improved_rate 低：
  增大 reference_size；
  降低 kernel_rank；
  加强 damping；
  检查 batch 是否类别不平衡。

如果 memory_harm 高：
  memory residual 必须进入约束，而不是只做评估。
```

---

## 7. P3：AdamW 交互方式诊断

### 7.1 目标

确定函数空间自然步失败是否因为它和 AdamW 的动量 / 二阶矩状态冲突。

### 7.2 对照方法

```text
AdamW baseline；
functional update only；
AdamW + additive functional update；
AdamW + functional preconditioned gradient；
AdamW + functional update every 5 steps；
AdamW + functional update every 10 steps；
AdamW + functional update + moment reset；
AdamW + functional update + moment transport diagnostic；
AdamW + same-norm random function-space perturbation。
```

### 7.3 必须记录

```text
cosine_functional_adamw
norm_functional
norm_adamw
update_over_param_norm
momentum_norm_before_after
second_moment_norm_before_after
loss_spike_count
bad_step_rate
train_loss_auc_step
val_loss_auc_step
val_loss_auc_time
ECE_curve
Brier_curve
margin_p10_curve
memory_loss_curve
hardtail_loss_curve
```

### 7.4 判断标准

P3 weak pass：

```text
至少一种 interaction mode 在 2 / 3 任务上 val_loss_auc_step 不差于 AdamW；
loss_spike_count 不高于 AdamW + random；
memory/hardtail 不坏；
same-norm random control 不同样有效。
```

P3 strong pass：

```text
至少一种 interaction mode 在 3 / 4 任务上 val_loss_auc_time 不差于 AdamW；
ECE 不差；
noise-label task 不受益；
amortized_step_ratio <= 1.5。
```

### 7.5 可视化

```text
validation loss vs step for interaction modes；
validation loss vs wall-clock；
update cosine with AdamW over time；
momentum norm before/after functional step；
loss spike timeline；
AdamW / functional / random comparison。
```

### 7.6 不满足时 Codex 先做什么

```text
如果 additive update 失败但 preconditioned gradient 成功：
  functional update 不能作为额外步，只能作为 AdamW 的函数空间预条件器。

如果 every-step 失败但 periodic 成功：
  functional update 应低频执行；
  不要扩大每步版本。

如果 moment reset 后成功：
  说明 optimizer state mismatch 是主因；
  下一步做 moment transport。

如果 random control 同样有效：
  指标不足或 step norm 过大；
  降低 trust_radius；
  加强 heldout/memory/hardtail 检查。
```

---

## 8. P4：约束求解器重建

### 8.1 目标

把 memory / hard-tail 从事后安全检查改成求解器里的约束，而不是硬 veto 或后验评估。

### 8.2 优化问题

求解：

$$
\Delta\theta^*
=
\arg\min_{\Delta\theta}
\left[
\nabla_f L_B^\top J_B \Delta\theta
+
\frac{1}{2\eta}\|J_R\Delta\theta\|^2
\right]
$$

subject to：

$$
\nabla_f L_M^\top J_M \Delta\theta \le \epsilon_M,
$$

$$
\nabla_f L_H^\top J_H \Delta\theta \le \epsilon_H,
$$

$$
\|J_R\Delta\theta\| \le \epsilon_f.
$$

其中 $B$ 是训练 batch，$M$ 是 memory buffer，$H$ 是 hard-tail set，$R$ 是函数空间参考集。

### 8.3 必须记录

```text
constraint_solver_type
lambda_memory
lambda_hardtail
lambda_trust
train_loss_delta
heldout_loss_delta
memory_constraint_violation
hardtail_constraint_violation
trust_constraint_violation
conflict_angle_train_memory
conflict_angle_train_hardtail
solver_iterations
solver_cost_ms
```

### 8.4 判断标准

P4 weak pass：

```text
constraint violation <= tolerance in 90% trials；
heldout improvement in >= 60% trials；
train improvement not fully collapsed；
solver_cost <= 3x AdamW in discovery。
```

P4 strong pass：

```text
constraint violation <= tolerance in 97% trials；
heldout improvement >= 70%；
real-label benefit > random-label benefit；
amortized cost <= 1.5x AdamW with update_period >= 5。
```

### 8.5 可视化

```text
train improvement vs memory harm Pareto；
train improvement vs hardtail harm Pareto；
constraint multiplier trajectory；
conflict angle histogram；
solver cost vs constraint violation scatter。
```

### 8.6 不满足时 Codex 先做什么

```text
如果 constraints kill all train improvement：
  记录 train-memory conflict angle；
  分层看哪个 layer / class / sample group 冲突最大；
  减小 update magnitude，而不是放松 memory threshold。

如果 solver cost 高：
  转 output-space small QP；
  使用 low-rank sketch；
  blockwise solve；
  periodic update。

如果 hard-tail 仍受伤：
  hard-tail set 可能 stale；
  更新 hard-tail sampling；
  使用 CEp99-focused reference。
```

---

## 9. P5：短程训练重跑，但只允许机制化变体

### 9.1 目标

重新跑短程训练，但禁止盲目扩大预算。只允许从 P1-P4 中找到明确机制的变体进入。

### 9.2 方法

候选最多保留 6 个：

```text
AdamW baseline；
Best-P1-damped-periodic；
Best-P2-signal-filtered；
Best-P3-interaction-mode；
Best-P4-constrained-solve；
Same-norm random function-space perturbation。
```

任务：

```text
MNIST-like；
Fashion-MNIST-like；
KMNIST-like；
synthetic interaction；
label-noise task；
non-vision tabular task。
```

### 9.3 必须记录

```text
method
model_type
dataset
seed
update_period
trust_radius
damping_lambda
reference_size
kernel_rank
val_loss_auc_step
val_loss_auc_time
test_acc_curve
ECE_curve
NLL_curve
Brier_curve
margin_p10_curve
CEp99_curve
memory_loss_curve
hard_tail_loss_curve
noise_label_benefit
same_norm_random_benefit
step_time_ms
amortized_functional_update_cost_ms
peak_memory_mb
bad_step_rate
loss_spike_count
```

### 9.4 判断标准

P5 weak pass：

```text
普通多层感知机上 6 个任务中至少 3 个满足：
  val_loss_auc_time 不差于 AdamW；
  memory/hard-tail 不差；
  ECE 不差；
  same-norm random control 不同样有效。
```

P5 strong pass：

```text
普通多层感知机上 6 个任务中至少 4 个满足 weak 条件；
label-noise task 上 noise-label benefit 明显低于 real-label benefit；
amortized step ratio <= 1.5。
```

### 9.5 可视化

```text
validation loss vs step；
validation loss vs wall-clock；
ECE vs step；
memory/hard-tail loss vs step；
label-noise benefit comparison；
same-norm random control comparison；
functional update cost amortization。
```

### 9.6 不满足时 Codex 先做什么

```text
如果 P5 全部失败：
  停止当前函数空间自然更新范式；
  不进入 KAN；
  回到更新理论或基础架构。

如果 P5 单步好但短程坏：
  说明仍是 one-step local descent；
  检查 P3 optimizer-state conflict；
  不扩大 horizon 掩盖失败。

如果 random control 同样好：
  说明 function-space update 没有独特价值；
  停止该变体。

如果 cost 是唯一失败：
  转 blockwise / low-rank / periodic update；
  不写成 system success。
```

---

## 10. P6：只有普通多层感知机通过后才进入边函数网络

### 10.1 目标

如果 P5 在普通多层感知机上过线，才迁移到当前 LQ 边函数网络和 graph-free 边函数网络。

### 10.2 必须记录

```text
manual_forward_available
manual_backward_available
manual_jvp_available
manual_vjp_available
jvp_error_vs_reference
vjp_error_vs_reference
kernel_condition
basis_occupancy_entropy
basis_effective_rank
forward_time_ratio
backward_time_ratio
step_time_ratio
amortized_functional_update_cost
val_loss_auc_time
memory_loss_curve
hardtail_loss_curve
```

### 10.3 判断标准

P6 weak pass：

```text
LQ/KAN 上 JVP/VJP correctness pass；
至少一个任务 val_loss_auc_time 不差；
memory/hard-tail 不坏；
amortized step ratio <= 2.0。
```

P6 strong pass：

```text
至少 3 个任务中有 2 个不差于 AdamW；
backward memory <= MLP reference；
amortized step ratio <= 1.5。
```

### 10.4 不满足时 Codex 先做什么

```text
如果 MLP 成功但 KAN 失败：
  检查 KAN JVP/VJP correctness；
  检查 kernel condition；
  检查 basis occupancy / dead basis；
  回到 primitive / graph-free / kernel-native，不继续调 functional update。

如果 KAN cost 太高：
  使用 blockwise kernel；
  降低 sketch rank；
  periodic update。
```

---

## 11. P7：未来路径验证

### 11.1 目标

只有 P5 或 P6 过线后，才看 h20 / h80 / h240。未来路径只作为验证，不作为训练标签。

### 11.2 必须记录

```text
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
FastGood_count
SlowBurnGood_count
RiskyHighAUV_count
SafeLowValue_count
BadPath_count
paired_AdamW_delta
paired_random_delta
paired_noop_delta
```

### 11.3 判断标准

P7 weak pass：

```text
V240_LCB > 0；
longrisk_UCB <= 0.10；
memory/offdiag_UCB <= 0.10；
paired random control 不同样有效。
```

P7 strong pass：

```text
V240_LCB > 0；
longrisk_UCB <= 0.05；
memory/offdiag_UCB <= 0.05；
paired AdamW / random / noop 均被击败；
shuffled update control fails。
```

### 11.4 不满足时 Codex 先做什么

```text
如果 V1 好但 V240 坏：
  说明仍是 immediate descent；
  回到 P2/P3，不改未来标签。

如果 V240 好但 longrisk 高：
  回到 P4 constrained solve。

如果 random control 同样有效：
  说明更新没有 functional specificity；
  停止该路线。
```

---

## 12. P8：controller / runtime / paired replay 只在强门后打开

### 12.1 目标

只有 P7 strong pass 后，才允许 controller/runtime/paired replay。否则继续打开这些阶段只是制造假进展。

### 12.2 判断标准

```text
accepted_count >= 87；
precision >= 0.75；
V240_LCB > 0；
longrisk_UCB <= 0.05；
bad_UCB <= 0.05；
null_UCB <= 0.15；
memory_UCB <= 0.05；
offdiag_UCB <= 0.05；
runtime_step_ratio_q90 <= 1.50；
paired replay beats AdamW / best learning rate / noop / random；
shuffled update control fails。
```

### 12.3 不满足时 Codex 先做什么

```text
如果 accepted_count 不够：
  不降低质量阈值；
  增加 tested states / update periods；
  检查 update 是否太稀有。

如果 precision 低：
  回到 P7；controller 不 ready。

如果 runtime 失败：
  amortize update period；
  lower sketch rank；
  blockwise solve。

如果 paired replay 失败：
  不声明 functional success；
  检查是否只是 AdamW 或 random 的变体。
```

---

## 13. 最终决策规则

### Case A：P5 在普通多层感知机上失败

结论：

```text
当前函数空间自然更新范式失败；
不进入 KAN；
停止 v11.x functional update，回到理论或基础结构。
```

### Case B：P5 成功，P6 KAN 失败

结论：

```text
functional update 方向保留；
当前 KAN primitive / graph-free JVP/VJP / kernel-native path 是 blocker；
回到基础结构路线。
```

### Case C：P5/P6 单步和短程成功，P7 future path 失败

结论：

```text
函数空间更新仍然只是 immediate descent；
需要重新研究未来训练动力学；
不要训练 Fast/Slow selector。
```

### Case D：P7 成功但 runtime 成本过高

结论：

```text
转 blockwise / Kronecker / low-rank / orthogonalized approximation；
不写成 system success。
```

### Case E：P7/P8 全过

结论：

```text
进入 official controller / runtime / paired replay / short-full confirm。
```

