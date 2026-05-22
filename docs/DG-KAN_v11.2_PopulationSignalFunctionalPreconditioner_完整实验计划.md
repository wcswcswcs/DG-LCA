# DG-KAN v11.2 Population-Signal Functional Preconditioner 完整实验计划

> 目标：停止继续把 functional update 做成“额外动作”或“额外一步函数空间自然步”。v11.2 直接验证一个新的范式：functional update 应该成为 **AdamW 任务更新的函数空间信号预条件器**，而不是一个和 AdamW 抢方向的独立更新。
>
> 本计划只使用当前训练可见信息：当前 batch、分割子 batch、memory buffer、hard-tail buffer、模型输出、参数到输出的雅可比向量积和向量雅可比积。不得使用 future outcome label、dataset-specific threshold、历史好动作标签或 generated action 标签作为训练信号。
>
> 公式使用 Typora 友好的 `$...$` 和 `$$...$$`。

---

# 0. 本轮核心判断

v11.0 说明函数空间自然步的数值实现、单步输出预测、memory / hard-tail 约束都能工作，但短程训练完全不过。v11.1 进一步做了 trust radius、damping、rank、period、split consistency、AdamW interaction、memory / hard-tail 约束求解等 repair，仍然 `P5 task pass = 0/6`。这说明问题不是某个单步公式的数值 bug，而是：

$$
\boxed{
\text{单独插入一个函数空间自然步，没有形成稳定训练动力学。}
}
$$

v11.1 的最刺眼事实是：

```text
P1 stability weak / strong = 1 / 0
P2 signal weak / strong = 0 / 0
P3 AdamW interaction weak / strong = 1 / 0
P4 constrained solver weak / strong = 0 / 0
P5 mechanized smoke weak / strong = 0 / 0
P5 task pass = 0 / 6
```

其中 P2 的 `noise improved = 0.8333333333333334` 是核心警报。它说明即使真实标签信号比随机信号看起来强，当前函数空间自然步仍然会大量改善噪声标签方向。这意味着它没有可靠区分“会泛化的信号方向”和“训练集内部可下降的噪声方向”。

所以 v11.2 不再问：

```text
怎样把函数空间自然步再调稳一点？
怎样加更多 hard gate？
怎样把 P5 预算扩大？
```

v11.2 要问：

$$
\boxed{
\text{能否把 functional update 改成 AdamW 的信号通道预条件器，而不是额外一步参数扰动？}
}
$$

也就是说，普通 AdamW 仍然负责主任务优化；functional update 只负责调整 AdamW 更新方向中哪些输出空间方向应该被放大，哪些应该被收缩。

---

# 1. 本轮理论假设

## 1.1 旧公式的问题

v11.0 / v11.1 使用的核心形式是函数空间自然步：

$$
K = JJ^\top,
$$

$$
r = \nabla_f L,
$$

$$
\Delta\theta_{func} = -J^\top(K+\lambda I)^{-1}r.
$$

这里 $J$ 是输出对参数的雅可比，$r$ 是输出上的损失残差。这一步在单步上合理，但它作为一个额外参数扰动加入 AdamW 后可能出现三个问题：

```text
1. 它在当前 batch 上下降，但也能沿噪声标签方向下降；
2. 它改变了 AdamW 的动量 / 二阶矩估计，但没有把这个改变写进 AdamW 状态；
3. 它作为外加 update 和 AdamW 主方向抢控制权，多步后变成不稳定动力学。
```

所以 v11.2 的主假设是：

$$
\boxed{
\Delta\theta_{func}\text{ 不应作为 additive step，而应作为 }g_{AdamW}\text{ 的函数空间预条件。}
}
$$

---

## 1.2 新对象：函数空间信号预条件器

设普通 AdamW 在当前步看到的参数梯度为：

$$
g = \nabla_\theta L.
$$

它在输出空间里造成的线性变化是：

$$
\delta f_g = J(-\eta g).
$$

我们不直接加另一个 $\Delta\theta$，而是构造一个输出空间预条件器 $P_{sig}$，只保留跨子 batch 稳定的信号方向，收缩噪声方向：

$$
\delta f_{target} = -\eta P_{sig} r.
$$

再把这个目标输出变化拉回参数空间：

$$
g_{func} = J^\top(K+\lambda I)^{-1}P_{sig}r.
$$

最后不是直接应用 $g_{func}$，而是和 AdamW 梯度组合成一个新的有效梯度：

$$
g_{eff} = (1-\beta)g + \beta g_{func}.
$$

或者只用函数空间方向修正 AdamW 中不稳定的部分：

$$
g_{eff} = g + \beta(g_{func} - \operatorname{Proj}_{g}(g_{func})).
$$

然后把 $g_{eff}$ 送进 AdamW 的正常动量和二阶矩流程，而不是另加一步参数扰动。

这就是本轮要验证的范式：

$$
\boxed{
\text{Functional update = signal-channel preconditioned AdamW gradient.}
}
$$

---

# 2. 整体实验目标

v11.2 的整体目标不是证明 KAN 成功，也不是进入 controller。它只回答一个硬问题：

$$
\boxed{
\text{函数空间信号预条件器能否在普通多层感知机短程训练上稳定胜过 AdamW 和随机同范数扰动？}
}
$$

如果普通多层感知机上都失败，就停止当前 function-space natural update 主线，不进入 KAN。

如果普通多层感知机上成功，再做 graph-free KAN 迁移。

如果普通多层感知机上成功但 KAN 失败，说明 blocker 是 KAN primitive / graph-free Jacobian-vector product / vector-Jacobian product / kernel-native implementation，而不是 functional update 理论本身。

---

# 3. P0：复现 v11.1 边界并冻结旧路线

## 3.1 目标

确认 v11.1 的失败边界，并防止 Codex 回到旧路线。

## 3.2 必须读取与记录

```text
source_route_v1110
P1_stability_weak_pass
P2_signal_weak_pass
P3_adamw_interaction_weak_pass
P4_constrained_solver_weak_pass
P5_mechanized_smoke_task_pass_count
P5_mechanized_smoke_total_count
P6_status
P7_status
P8_status
fake_data_used
proxy_row_used
cpu_offload_used
```

## 3.3 判断标准

P0 pass 当且仅当：

```text
source_route_v1110 = CaseA-MLPShortTrainingMechanizedVariantsFail
P5 task pass = 0 / 6
P6/P7/P8 = not_run
fake/proxy/cpu = 0 / 0 / 0
old action-selection route disabled = 1
generated-action route disabled = 1
FPO/FPAU/GoodCone/LFRO route disabled = 1
```

## 3.4 不满足条件时 Codex 先做什么

如果 P0 读取不到 v11.1 artifact：

```text
先修 artifact path / manifest loader；
不要继续跑 P1-P7；
不要用 v11.0 替代 v11.1。
```

如果 fake/proxy/cpu 非零：

```text
停止本轮；
定位 fake/proxy/cpu 来源；
重新跑 v11.1 boundary lock。
```

---

# 4. P1：v11.1 短程失败动力学归因

## 4.1 目标

v11.1 不能只记录 `task pass = 0/6`。P1 要解释短程训练失败的动力学原因。

要判断失败属于哪一类：

```text
F1-noise-reservoir-amplification：噪声标签方向也被改善；
F2-adamw-state-conflict：额外函数步破坏 AdamW 动量 / 二阶矩；
F3-reference-overfit：reference set 太小，当前 batch 下降但 heldout 不稳；
F4-trust-radius-drift：单步 trust radius 合理，多步累积漂移；
F5-memory-hardtail-latent-harm：单步不伤 memory/hard-tail，多步伤害；
F6-random-control-explains-gain：同范数随机函数扰动能解释收益。
```

## 4.2 实验设置

对 v11.1 P5 中 6 个短程任务逐步重放前 100 或 200 步。每一步记录 AdamW baseline、AdamW + additive function-space natural step、AdamW + matched-norm random function step。

## 4.3 必须记录指标

```text
task_id
dataset
seed
step
method
train_loss
heldout_loss
memory_loss
hardtail_loss
noise_label_loss
val_loss
val_acc
function_update_norm
parameter_update_norm
cos_func_adamw
cos_func_prev_update
cos_adamw_prev_update
adamw_m_norm
adamw_v_norm
adamw_m_delta_after_func
adamw_v_delta_after_func
kernel_condition
trust_radius
cg_iterations
reference_size
signal_ratio
random_signal_ratio
noise_improved_bit
memory_harm_bit
hardtail_harm_bit
bad_step_bit
```

## 4.4 可视化

必须生成：

```text
p1_loss_curves_by_method.svg
p1_noise_vs_real_improvement.svg
p1_cos_func_adamw_trace.svg
p1_memory_hardtail_drift_trace.svg
p1_kernel_condition_vs_bad_step.svg
p1_failure_mode_stacked_bar.svg
```

## 4.5 判断标准

如果满足：

```text
noise_label_improved_rate >= 0.50
且 real_label_improvement / noise_label_improvement < 2.0
```

则主要失败是 `F1-noise-reservoir-amplification`。

如果满足：

```text
cos_func_adamw median < 0
或 adamw_m_delta_after_func p95 明显高于 baseline
```

则主要失败是 `F2-adamw-state-conflict`。

如果满足：

```text
train_loss improves
heldout_loss worsens
memory/hardtail noharm single-step = 1
但 100-step memory/hardtail drift > tolerance
```

则主要失败是 `F3/F5`。

## 4.6 不满足条件时 Codex 先做什么

如果无法归因，Codex 不允许继续 P2，应先：

```text
增加 per-step trace 字段；
检查 random-function control 是否同范数；
检查 noise-label task 是否真正 shuffled；
检查 heldout batch 是否与 train batch disjoint；
检查 memory/hard-tail buffer 是否固定；
检查 AdamW state 是否在 additive function step 后被正确记录。
```

---

# 5. P2：实现 Population-Signal Functional Preconditioner

## 5.1 目标

把函数空间更新从“额外一步”改成 AdamW 梯度的信号通道预条件。

## 5.2 候选方法

所有候选都必须使用同一个训练 batch、同一个 memory buffer、同一个 hard-tail buffer，不允许 dataset-specific threshold。

### PSFP-A：AdamW baseline

普通 AdamW，不加函数空间预条件。

### PSFP-B：v11.1 additive function-space natural step

复现旧失败线，用作负对照。

### PSFP-C：gradient replacement

用函数空间自然梯度替换 AdamW 梯度：

$$
g_{eff}=g_{func}.
$$

### PSFP-D：gradient interpolation

把 AdamW 梯度与函数空间梯度混合：

$$
g_{eff}=(1-\beta)g+\beta g_{func}.
$$

测试：

```text
beta = 0.05, 0.10, 0.25, 0.50
```

### PSFP-E：orthogonal residual correction

只加入 AdamW 当前方向没有覆盖的函数空间残差：

$$
g_{eff}=g+\beta(g_{func}-\operatorname{Proj}_g(g_{func})).
$$

### PSFP-F：signal-channel shrinkage

用子 batch 一致性构造 $P_{sig}$，对不稳定输出方向收缩：

$$
g_{func}=J^\top(K+\lambda I)^{-1}P_{sig}r.
$$

### PSFP-G：memory-constrained signal shrinkage

在 PSFP-F 基础上加入 memory / hard-tail 的不伤害约束：

$$
\Delta L_{memory}\le \epsilon_M,
$$

$$
\Delta L_{hardtail}\le \epsilon_H.
$$

### PSFP-H：population-risk diagnostic adapter

只做 discovery，不写 official。使用 leave-one-out 或 split-one-out 近似 population-risk direction，验证它是否比当前 signal-channel shrinkage 更好。

## 5.3 必须记录指标

```text
method
beta
lambda
rank
reference_size
split_count
signal_mode_count
kernel_condition
cg_iterations
cost_ms_q50
cost_ms_q90
amortized_cost_ratio
cos_eff_adamw
cos_func_adamw
cos_noise_direction
real_signal_ratio
random_signal_ratio
noise_improved_rate
train_loss_delta_1step
heldout_loss_delta_1step
memory_loss_delta_1step
hardtail_loss_delta_1step
same_norm_random_delta
```

## 5.4 判断标准

P2 weak pass：

```text
noise_improved_rate <= 0.25
real_signal_ratio / random_signal_ratio >= 2.0
heldout_loss_delta_1step <= 0 on at least 70% trials
memory/hardtail harm rate <= 0.10
cost_ms_q90 <= 3x AdamW step for discovery
```

P2 strong pass：

```text
noise_improved_rate <= 0.10
real_signal_ratio / random_signal_ratio >= 4.0
heldout_loss_delta_1step <= 0 on at least 85% trials
memory/hardtail harm rate <= 0.05
amortized_cost_ratio <= 1.5x AdamW step
```

## 5.5 可视化

```text
p2_real_vs_noise_signal_ratio.svg
p2_method_pareto_value_noise_cost.svg
p2_cos_eff_adamw_hist.svg
p2_memory_hardtail_harm_bar.svg
p2_kernel_spectrum_signal_modes.svg
```

## 5.6 不满足条件时 Codex 先做什么

如果 noise_improved_rate 高：

```text
增加 split_count；
改用 split-consistency eigenmodes；
减少 beta；
增加 ridge damping；
不要调 dataset-specific threshold。
```

如果 heldout_loss 不改善：

```text
检查 P_sig 是否只来自 train split；
加入 leave-one-split-out validation；
降低 trust radius；
检查 residual r 是否使用 logits gradient 而不是 loss scalar gradient。
```

如果 memory/hardtail 受伤：

```text
不要 hard veto；
改 constrained least squares；
把 memory/hardtail 加入 reference kernel；
提高 memory sample weight；
检查 memory buffer 是否固定。
```

如果成本太高：

```text
降低 rank；
减少 conjugate gradient iterations；
使用 blockwise output kernel；
每 5 或 10 步 amortize；
禁止进入 official。
```

---

# 6. P3：AdamW 耦合方式实验

## 6.1 目标

验证 functional update 应该如何进入 AdamW：是替换梯度、混合梯度、残差修正，还是低频预条件。

## 6.2 方法

对 P2 过 weak 的候选，测试以下耦合方式：

```text
C0: baseline AdamW
C1: additive delta after AdamW step
C2: replace gradient before AdamW moment update
C3: interpolate gradient before AdamW moment update
C4: residual correction before AdamW moment update
C5: periodic every 5 steps
C6: periodic every 10 steps
C7: warmup 100 steps then enable
C8: late-only enable after loss plateau
```

## 6.3 必须记录指标

```text
coupling_mode
method
period
warmup_steps
val_loss_auc_step
val_loss_auc_time
train_loss_auc
memory_loss_auc
hardtail_loss_auc
bad_step_rate
loss_spike_count
adamw_m_norm_trace
adamw_v_norm_trace
cos_update_prev_trace
cos_update_grad_trace
step_cost_ratio
```

## 6.4 判断标准

P3 weak pass：

```text
至少 2 个 coupling mode 的 val_loss_auc_step <= AdamW baseline
bad_step_rate <= AdamW + 0.05
memory/hardtail AUC 不劣于 AdamW
cost amortized <= 2.0x
```

P3 strong pass：

```text
至少 1 个 coupling mode 的 val_loss_auc_time <= AdamW baseline
且 memory/hardtail/ECE 不劣于 AdamW
且 random-function control 不能复现该收益
```

## 6.5 不满足条件时 Codex 先做什么

如果只有 additive 好但 AdamW-coupled 不好：

```text
检查 AdamW moment 是否过度吸收 functional correction；
降低 beta；
只在 warmup 后启用；
不扩大训练预算。
```

如果 only periodic10 有信号：

```text
把 functional update 降级为 low-frequency geometry preconditioner；
不做 every-step official。
```

如果所有 coupling 都失败：

```text
停止 v11.2 function-space preconditioner；
回到 function-preserving geometry maintenance 或 base architecture。
```

---

# 7. P4：短程训练并行 smoke

## 7.1 目标

在普通多层感知机上验证 PSFP 是否能形成多步训练优势。只有普通多层感知机过线，才允许进入 KAN。

## 7.2 任务

保留 v11.1 的 6 个任务，另外增加两个不用于调参的诊断任务：

```text
MNIST-small
Fashion-MNIST-small
KMNIST-small
Synthetic-interaction
Synthetic-label-noise
Synthetic-input-noise
Tabular-small-diagnostic
Toy-sequence-diagnostic
```

诊断任务只用于判断机制，不用于 dataset-specific tuning。

## 7.3 对照方法

```text
AdamW baseline
AdamW + v11.1 additive FSNU
AdamW + same-norm random function update
AdamW + PSFP-D
AdamW + PSFP-E
AdamW + PSFP-F
AdamW + PSFP-G
AdamW + PSFP-H diagnostic, if P2 allows
```

## 7.4 必须记录指标

```text
task_id
dataset
seed
method
train_loss_curve
val_loss_curve
test_acc_curve
val_loss_auc_step
val_loss_auc_time
train_loss_auc
memory_loss_auc
hardtail_loss_auc
ECE_curve
NLL_curve
Brier_curve
margin_p10_curve
noise_label_loss_curve
bad_step_rate
loss_spike_count
step_time_ms_q50
step_time_ms_q90
amortized_cost_ratio
random_control_match_rate
```

## 7.5 判断标准

P4 weak pass：

```text
至少 4 / 8 任务满足：
  val_loss_auc_step <= AdamW baseline
  memory/hardtail AUC 不劣于 AdamW
  random-control 不能复现收益
  noise_label_loss 不同步改善
```

P4 strong pass：

```text
至少 5 / 8 任务满足 weak 条件
且至少 3 / 8 任务 val_loss_auc_time <= AdamW baseline
且 ECE 不劣于 AdamW
```

## 7.6 可视化

```text
p4_val_loss_vs_step_by_task.svg
p4_val_loss_vs_time_by_task.svg
p4_memory_hardtail_auc_bar.svg
p4_noise_real_gap_by_method.svg
p4_random_control_comparison.svg
p4_cost_vs_gain_pareto.svg
```

## 7.7 不满足条件时 Codex 先做什么

如果 step-AUC 好但 time-AUC 差：

```text
保留为 expensive diagnostic；
转 low-rank / blockwise / periodic approximation；
不要写 success。
```

如果 train loss 好但 val loss 差：

```text
增强 signal-channel shrinkage；
检查 noise-label improvement；
降低 beta；
不加 dataset-specific threshold。
```

如果 memory/hardtail 受伤：

```text
启用 PSFP-G constrained variant；
将 memory/hardtail 加入 reference set；
检查是否 reference weighting 太低。
```

如果所有方法都 0/8：

```text
停止 function-space natural update 主线；
转 function-preserving geometry maintenance 或 primitive/base architecture。
```

---

# 8. P5：与前沿优化器风格的对照

## 8.1 目标

判断 PSFP 是否只是 AdamW、随机扰动、sharpness 风格扰动或矩阵几何优化器的重复。

## 8.2 对照

```text
AdamW
AdamW + SAM-like small perturbation
AdamW + same-norm random function update
AdamW + diagonal SNR preconditioner
AdamW + SOAP-like block preconditioner diagnostic
AdamW + Muon-like orthogonalized matrix update diagnostic
AdamW + PSFP best
```

SOAP / Muon 对照若实现成本过高，可以先用简化 blockwise diagnostic，不写 official。

## 8.3 记录指标

```text
method
cos_with_adamw
cos_with_sam
cos_with_random
function_delta_norm
parameter_delta_norm
output_signal_ratio
output_noise_ratio
val_loss_auc_step
val_loss_auc_time
ECE
memory_loss_auc
hardtail_loss_auc
cost_ratio
```

## 8.4 判断标准

PSFP 有独立价值，当且仅当：

```text
PSFP val_loss_auc_step 优于 AdamW 和 same-norm random；
PSFP noise improvement 明显低于 additive FSNU；
PSFP memory/hardtail 不劣；
PSFP 不是仅由更大 norm 解释；
若 SOAP/Muon diagnostic 更好，则 PSFP 降级为理论参考，转 optimizer integration。
```

## 8.5 不满足条件时 Codex 先做什么

如果 SOAP/Muon-like 更好：

```text
停止手写 PSFP score；
转向把 functional metric 整合进 blockwise preconditioner。
```

如果 SAM-like 更好：

```text
检查是否问题只是 flatness / trust region；
不要继续复杂 function-space kernel。
```

---

# 9. P6：KAN graph-free 迁移门

## 9.1 目标

只有 P4/P5 在普通多层感知机上至少 weak pass，才允许迁移到边函数网络。迁移前先做 graph-free JVP / VJP readiness。

## 9.2 必须记录

```text
manual_jvp_available
manual_vjp_available
manual_jvp_fd_error
manual_vjp_dot_error
parameter_flatten_order_match
edge_param_coverage
nonKAN_param_count
uses_loss_backward
uses_autograd_graph
cost_ratio_vs_mlp
```

## 9.3 判断标准

```text
manual_jvp_fd_error <= 1e-5
manual_vjp_dot_error <= 1e-8
edge_param_coverage = 1.0
nonKAN_param_count = 0
uses_loss_backward = 0
cost_ratio_vs_mlp <= 3.0 discovery
```

## 9.4 不满足条件时 Codex 先做什么

如果 JVP/VJP 不可用：

```text
先实现 graph-free edge-function JVP/VJP；
不要用 autograd official candidate；
只允许 autograd discovery reference。
```

如果成本太高：

```text
blockwise JVP/VJP；
low-rank output sketch；
periodic update；
不进入 official。
```

---

# 10. P7：未来路径验证

## 10.1 目标

只有普通多层感知机短程训练过线，才验证未来路径。未来路径只作为验证，不作为训练标签。

## 10.2 记录指标

```text
method
task_id
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
FastGood_count
SlowBurnGood_count
RiskyHighAUV_count
BadPath_count
```

## 10.3 判断标准

```text
FastGood + SlowBurnGood precision >= 0.25 discovery
V240 LCB > 0
longrisk UCB <= 0.10
memory/offdiag UCB <= 0.10
same-norm random control 不复现
```

## 10.4 不满足条件时 Codex 先做什么

如果短程 val loss 好但 V240 坏：

```text
说明仍是 immediate descent；
加强 signal-channel shrinkage；
加入 longer-reference memory；
不要调 dataset-specific threshold。
```

如果 V240 好但 longrisk 高：

```text
改 constrained solve；
提高 memory/hardtail reference weight；
使用 trust radius schedule。
```

---

# 11. P8：controller / runtime / paired replay 边界

## 11.1 打开条件

P8 只有在以下条件全部满足时打开：

```text
P4 weak pass = 1
P5 independent-value pass = 1
P6 graph-free readiness pass = 1, if KAN path
P7 future path weak pass = 1
```

## 11.2 记录指标

```text
controller_candidate_count
accepted_count
precision
V240_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
step_time_ratio_q90
paired_replay_vs_AdamW
paired_replay_vs_random
paired_replay_vs_noop
```

## 11.3 判断标准

```text
accepted_count >= 87
precision >= 0.75
V240_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.05
step_time_ratio_q90 <= 1.50
paired replay beats AdamW / random / noop
```

---

# 12. 本轮硬退出规则

## 12.1 停止 function-space natural update

如果 P4 在普通多层感知机上仍然：

```text
weak pass = 0
且 task pass <= 1 / 8
```

则停止当前 function-space natural update 主线，不进入 KAN。

## 12.2 降级为 expensive diagnostic

如果 step-AUC 有用但 time-AUC 不过：

```text
route = ExpensiveFunctionalDiagnosticOnly
next = low-rank / blockwise / periodic approximation
```

## 12.3 转 optimizer integration

如果 SOAP/Muon/SAM-like 对照明显优于 PSFP：

```text
route = ExistingOptimizerGeometryDominates
next = integrate KAN geometry into optimizer preconditioner
```

## 12.4 转 function-preserving geometry maintenance

如果所有 current-loss functional update 都失败：

```text
route = CurrentLossFunctionalUpdateFails
next = function-preserving geometry maintenance
```

这条路线只允许做：

```text
edge-function refit
basis/knot redistribution
base-residual rebalance
coefficient whitening
optimizer-state transport
memory-preserving refit
```

目标是不改变当前函数输出，而改善表示条件。

---

# 13. 最终 route 格式

本轮必须输出以下之一：

```text
R1-PSFPShortTrainingWeakPass
R2-PSFPStrongButTooExpensive
R3-PSFPStillNoiseReservoirAmplifying
R4-AdamWInteractionConflictUnresolved
R5-ExistingOptimizerGeometryDominates
R6-MLPFunctionSpaceUpdateStillFails
R7-KANGraphFreeJVPVJPBlocked
R8-KANTransferFailAfterMLPPass
R9-FunctionPreservingGeometryMaintenancePivot
```

---

# 14. 最终判断

v11.2 的目标不是把 v11.1 调过，而是裁决一个更根本的问题：

$$
\boxed{
\text{functional update 是否应该作为 AdamW 的函数空间信号预条件器存在？}
}
$$

如果答案是否定的，就不要再围绕函数空间自然步小修。那时应转向函数保持的几何维护，或者回到边函数网络基础结构与 kernel-native 训练系统。
