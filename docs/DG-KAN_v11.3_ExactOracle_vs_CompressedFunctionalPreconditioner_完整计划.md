# DG-KAN v11.3 Exact Functional Oracle vs Compressed Functional Preconditioner 完整实验计划

> 本计划基于 v11.2 `Population-Signal Functional Preconditioner` 的真实结果制定。v11.3 不继续小修 `PSFP-C`、`beta`、`rank`、`periodic`、`FPO`、`FPAU`、`GoodCone`、`LFRO` 或 generated action family。v11.3 的目标是做一次决定性拆分：
>
> 1. 高成本的函数空间更新在理论上到底有没有真实训练价值？
> 2. 如果高成本版本有价值，能否压缩成低成本预条件器？
> 3. 如果高成本版本都没有价值，是否应停止当前函数空间自然更新主线？
>
> 全文公式使用 Typora 友好的 `$...$` 与 `$$...$$`。

---

## 0. 本轮一句话目标

v11.3 的目标是：

$$
\boxed{
\text{先判断 exact function-space oracle 是否真的有效，再判断是否值得压缩；不要直接继续修 cheap proxy。}
}
$$

v11.2 的结果说明：

```text
route = R6-MLPFunctionSpaceUpdateStillFails
primary_blocker = psfp_candidate_too_expensive_for_discovery_gate
secondary_blocker = short_training_smoke_not_validated_after_cost_repairs
P2 best = PSFP-C-gradient-replacement
P2 best noise improved = 0.0
P2 best real/random = 7.899966612920136
P4 task pass = 1 / 8
P5 independent value = 0
```

这不是普通小失败。它说明：

```text
函数空间信号预条件器确实能在某些单步诊断中压住 noise；
但它代价太高，且短程训练没有稳定成立；
因此不能直接进入 KAN、future path、controller 或 runtime。
```

所以 v11.3 不应问：

```text
PSFP-C 阈值再调一下？
rank 再降一点？
reference set 再换一下？
periodic10 换 periodic20？
```

而应问：

```text
如果不考虑成本，exact function-space oracle 是否真的能赢？
如果 exact oracle 也赢不了，函数空间自然更新主线应停止。
如果 exact oracle 能赢，cheap approximation 才有研究价值。
```

---

## 1. 核心假设

### H1：v11.2 的失败可能是成本与近似失败，不一定是函数空间目标失败

v11.2 的 `PSFP-C-gradient-replacement` 把 noise improved rate 降到 `0.0`，这是正信号。但它的 amortized cost 约为 `12.35`，远超 discovery gate；短程训练也没有稳定通过。

因此必须区分两种情况：

```text
情况 A：exact oracle 本身没有训练价值。
  那么函数空间自然更新范式失败。

情况 B：exact oracle 有训练价值，但太贵。
  那么真正问题是压缩 / 近似 / kernelization。
```

v11.3 必须先裁决 A/B。

### H2：函数空间更新不应作为额外参数步，而应作为 AdamW 梯度预条件

旧方式：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{AdamW}+\Delta\theta_{func}.
$$

新方式：

$$
g_{eff}=\mathcal{P}_{func}(g_{AdamW}),
$$

然后让 AdamW 用 $g_{eff}$ 更新自己的动量和二阶矩。函数空间模块不再绕开 AdamW 状态，而是改变 AdamW 看到的有效梯度。

### H3：函数空间更新必须证明不会奖励噪声标签

v11.1 与 v11.2 都显示 noise/reservoir 问题很严重。v11.2 P1 里 real/noise improvement 为 `0.6667 / 0.8333`，说明旧机制更容易改善噪声标签方向。任何新候选必须满足：

$$
Improve_{real} - Improve_{noise} \ge 0.25.
$$

如果真实标签与噪声标签同时受益，说明更新仍在训练集内部可下降方向里移动，而不是进入可泛化信号通道。

### H4：如果 exact oracle 有价值，应优先压缩到低秩 / 分块 / 对角，而不是手写新规则

压缩路线应该来自函数空间核：

$$
K = JJ^\top,
$$

而不是来自手工规则。候选近似包括：

```text
Nyström low-rank kernel；
blockwise output-space kernel；
diagonal signal-to-noise preconditioner；
K-FAC-like layerwise Fisher approximation；
SOAP/Shampoo-like eigenbasis approximation；
Muon-like orthogonalized matrix direction。
```

---

## 2. 本轮不做什么

v11.3 明确停止以下路线：

```text
1. 不继续 generated action family；
2. 不回到 FastGood / SlowBurnGood action selector；
3. 不训练 outcome-label classifier；
4. 不使用 future outcome 做训练信号；
5. 不按 dataset 调 threshold；
6. 不直接进入 KAN；
7. 不因为 P2 单步好看就打开 controller；
8. 不把 high-cost exact oracle 写成 system success。
```

---

## 3. 总体流程

v11.3 分为九个阶段：

```text
P0: v11.2 boundary lock and artifact audit
P1: exact oracle cost decomposition
P2: exact oracle unlimited-budget short training
P3: exact oracle noise/signal decomposition
P4: compressed functional preconditioner matrix
P5: AdamW coupling and optimizer-state compatibility
P6: short training decisive smoke
P7: compare against existing geometry optimizers
P8: KAN graph-free migration gate
P9: final route decision and stop/pivot rules
```

---

## 4. P0：v11.2 边界冻结

### 4.1 目标

确认 v11.2 的失败边界真实存在，不复用错误 artifact，不把 discovery diagnostic 写成 official。

### 4.2 必须检查

```text
source route = R6-MLPFunctionSpaceUpdateStillFails
P1 dominant failure = F1-noise-reservoir-amplification
P2 best = PSFP-C-gradient-replacement
P2 weak/strong = 0 / 0
P4 task pass = 1 / 8
P5 independent value = 0
P6/P7/P8 = not_run
fake/proxy/cpu = 0 / 0 / 0
```

### 4.3 通过标准

```text
boundary_lock_pass = 1
no_fake_pass = 1
old_action_routes_disabled = 1
```

### 4.4 不满足时 Codex 先尝试

```text
如果 artifact 缺失：
  先修 manifest / path / hash，不运行新实验。

如果 v11.2 boundary 不一致：
  重跑 v11.2 P0-P4 small profile。

如果 old action route 意外打开：
  立即关闭 generated/FPO/FPAU/GoodCone/LFRO flags。
```

---

## 5. P1：Exact oracle 成本分解

### 5.1 目标

判断 PSFP-C 为什么太贵。不能只记录一个总 cost，要拆成：

```text
forward output cost；
JVP cost；
VJP cost；
kernel construction cost；
linear solve cost；
reference set cost；
memory/hard-tail constraint cost；
AdamW coupling cost；
Python overhead；
GPU sync overhead。
```

### 5.2 记录字段

```text
method_id
dataset
seed
model_type
batch_size
reference_size
memory_size
hard_tail_size
kernel_rank
cg_iteration_count
forward_ms
jvp_ms
vjp_ms
kernel_build_ms
solve_ms
constraint_ms
adamw_coupling_ms
python_overhead_ms
sync_count
peak_gpu_mb
amortized_cost_ratio
noise_improved_rate
real_improved_rate
heldout_nonharm_rate
memory_noharm_rate
hardtail_noharm_rate
```

### 5.3 判断标准

P1 不要求 task 成功，只判断是否有压缩空间。

```text
如果 solve_ms / total_ms >= 0.5：
  压缩重点是 low-rank / iterative solve。

如果 JVP+VJP >= 0.5：
  压缩重点是 cached Jacobian sketch / layerwise approximation。

如果 reference_size 扩大导致 cost 近似线性爆炸：
  需要 reference subsampling / Nyström。

如果 python_overhead 或 sync_count 高：
  需要 vectorization / fused batch。
```

### 5.4 可视化

```text
p1_cost_waterfall_by_method.svg
p1_reference_size_vs_cost.svg
p1_kernel_rank_vs_cost.svg
p1_cost_vs_noise_suppression.svg
p1_cost_vs_real_signal_improvement.svg
```

### 5.5 不满足时 Codex 先尝试

```text
如果无法分解 cost：
  在 runner 内加入 CUDA event timers 和 torch.cuda.synchronize boundaries。

如果 sync_count 高：
  合并 JVP/VJP calls，避免逐 split 循环。

如果 kernel_build_ms 高：
  改成 implicit matvec，不显式构造 K。

如果 solve_ms 高：
  降低 CG iterations，加入 damping，试 Nyström rank 8/16/32。
```

---

## 6. P2：Exact oracle unlimited-budget 短程训练

### 6.1 目标

这是 v11.3 的第一硬门。允许 exact oracle 很贵，但要问它是否真的有效。

如果高成本 exact oracle 都没有稳定训练价值，则停止当前函数空间自然更新主线。

### 6.2 方法

候选：

```text
AdamW baseline
same-norm random function update
v11.1 additive function-space natural step
PSFP-C exact gradient replacement
PSFP-H exact population-risk adapter
PSFP-C + memory constrained solve
PSFP-C + split consistency only
```

任务：

```text
MNIST small
Fashion-MNIST small
KMNIST small
synthetic interaction
label-noise diagnostic
input-noise diagnostic
tabular diagnostic
toy sequence diagnostic
```

每个任务只跑小预算短程训练，不追榜，只看机制。

### 6.3 记录字段

```text
task_id
method_id
seed
train_loss_auc
val_loss_auc
val_acc_auc
noise_label_loss_auc
memory_loss_auc
hardtail_loss_auc
ECE_auc
Brier_auc
bad_step_rate
loss_spike_count
time_to_small_target
step_time_ratio
wall_clock_auc
real_improved_rate
noise_improved_rate
real_minus_noise
same_norm_random_delta
adamw_delta
```

### 6.4 通过标准

Exact oracle discovery pass：

```text
至少 5 / 8 task 满足：
  val_loss_auc <= AdamW - small_margin
  real_minus_noise >= 0.25
  memory_loss_auc <= AdamW + tolerance
  hardtail_loss_auc <= AdamW + tolerance
  same_norm_random 不具备同样收益
```

Strong pass：

```text
至少 6 / 8 task 满足 discovery pass；
并且 label-noise diagnostic 中 noise_label_loss 不明显下降；
并且 input-noise diagnostic 中 hardtail 不坏。
```

### 6.5 不满足时 Codex 先尝试

```text
如果 exact oracle 也 0/8 或 <=1/8：
  停止函数空间自然更新主线，进入 P9 Case A。

如果 exact oracle real 改善但 noise 也改善：
  增加 split-consistency projection；
  收缩 reservoir/noise eigenmodes；
  不调 dataset threshold。

如果 exact oracle val 下降但 memory/hardtail 受伤：
  改成 constrained solve；
  memory 作为约束，不作为 reward。

如果 exact oracle 只在 synthetic 有效：
  降级为 theory diagnostic，不进入 compression。
```

---

## 7. P3：Exact oracle 的信号 / 噪声分解

### 7.1 目标

如果 P2 出现任何正信号，需要判断它来自真实可泛化信号，还是来自训练集噪声。

### 7.2 方法

把每个 batch 分成多个子 batch，计算每个子 batch 的函数空间方向：

$$
u_s = J_s^\top (K_s + \lambda I)^{-1}r_s.
$$

定义跨 split 一致性：

$$
C_{split} = \frac{\|\mathbb{E}_s[\nu_s]\|^2}{\mathbb{E}_s\|\nu_s-\bar\nu\|^2+\epsilon}.
$$

并对比真实标签和随机标签。

### 7.3 记录字段

```text
split_count
label_mode = real / shuffled / random
mean_direction_norm
variance_direction_norm
split_consistency
cos_with_adamw
cos_between_splits
heldout_split_improvement
noise_split_improvement
signal_channel_rank
reservoir_energy_ratio
```

### 7.4 判断标准

```text
real split_consistency >= 3 * random split_consistency
heldout_split_improvement >= 0.7
noise_split_improvement <= 0.25
reservoir_energy_ratio decreases after projection
```

### 7.5 不满足时 Codex 先尝试

```text
如果 real/random consistency 接近：
  当前 reference set 太小或方向仍是 noise；增加 split 数，缩小 trust radius。

如果 heldout split 不改善：
  减小 step radius，增加 damping。

如果 noise split 也改善：
  对低一致性 eigendirections 做 shrinkage。
```

---

## 8. P4：Compressed functional preconditioner matrix

### 8.1 目标

只有 P2/P3 证明 exact oracle 有价值，才压缩。压缩不是调阈值，而是逼近 exact oracle。

### 8.2 候选

```text
C1: diagonal output-space SNR preconditioner
C2: Nyström rank 8 / 16 / 32 kernel approximation
C3: blockwise layer output kernel
C4: K-FAC-like layerwise Fisher approximation
C5: SOAP/Shampoo-like eigenbasis approximation
C6: Muon-like orthogonalized gradient direction
C7: low-rank residual correction to AdamW
C8: periodic exact oracle teacher -> cheap student preconditioner
```

### 8.3 记录字段

```text
candidate_id
approximation_family
rank
block_size
cos_with_exact_oracle
relative_error_to_exact_oracle
noise_improved_rate
real_improved_rate
heldout_nonharm
memory_noharm
hardtail_noharm
cost_ratio
peak_memory_mb
```

### 8.4 判断标准

Compression pass：

```text
cos_with_exact_oracle >= 0.75
real_minus_noise >= 0.20
heldout_nonharm >= 0.75
memory_noharm >= 0.90
hardtail_noharm >= 0.90
cost_ratio <= 3.0 for discovery
```

Production candidate gate：

```text
cost_ratio <= 1.5
```

### 8.5 不满足时 Codex 先尝试

```text
如果 cos_with_exact_oracle 低：
  增加 rank 或换 blockwise basis。

如果 cost 高：
  降 rank，减少 reference set，缓存 JVP/VJP。

如果 memory 受伤：
  加 constrained solve，不要调 reward。

如果 approximation 比 exact 差很多：
  回到 exact oracle，不做 cheap proxy。
```

---

## 9. P5：AdamW 耦合方式

### 9.1 目标

确定函数空间预条件器应该如何进入 AdamW。

### 9.2 候选耦合方式

```text
A0: AdamW baseline
A1: replace raw gradient before AdamW moment
A2: interpolate raw gradient before AdamW moment
A3: residual correction orthogonal to AdamW gradient
A4: precondition first moment m_t
A5: precondition second moment denominator
A6: periodic low-frequency functional correction
A7: late-plateau only
A8: exact oracle teacher for moment-shaping only
```

### 9.3 记录字段

```text
method_id
cos_effective_grad_with_adamw
momentum_norm_change
second_moment_norm_change
bad_step_rate
loss_spike_count
real_minus_noise
memory_noharm
hardtail_noharm
val_loss_auc
step_time_ratio
```

### 9.4 判断标准

```text
bad_step_rate <= AdamW + 0.02
loss_spike_count <= AdamW + 1
real_minus_noise >= 0.20
val_loss_auc <= AdamW
step_time_ratio <= 3.0 discovery
```

### 9.5 不满足时 Codex 先尝试

```text
如果 moment norm 爆：
  降 beta 或只预条件 raw gradient，不碰 moments。

如果 loss spikes：
  加 trust radius；
  改成 periodic/late-only。

如果和 AdamW cosine 太高：
  说明只是 AdamW 变体，转 residual correction。

如果和 AdamW cosine 太低且 bad steps 多：
  说明方向冲突，改成 small beta。
```

---

## 10. P6：短程训练 decisive smoke

### 10.1 目标

只让 P2-P5 过线候选进入短程训练。不能因为单步好看就进入。

### 10.2 任务

```text
MNIST small
Fashion-MNIST small
KMNIST small
synthetic interaction
label-noise diagnostic
input-noise diagnostic
tabular diagnostic
toy sequence diagnostic
```

### 10.3 对照

```text
AdamW
same-norm random update
SAM-like diagnostic
SOAP-like diagnostic
Muon-like diagnostic
diagonal SNR diagnostic
exact oracle, if P2 pass
compressed candidate
```

### 10.4 通过标准

Weak pass：

```text
至少 5 / 8 task：
  val_loss_auc <= AdamW
  real_minus_noise >= 0.20
  memory/hardtail 不坏
  same-norm random 不同样有效
```

Strong pass：

```text
至少 6 / 8 task；
并且 cost_ratio <= 3.0 discovery；
并且没有一个任务出现 catastrophic loss spike。
```

### 10.5 不满足时 Codex 先尝试

```text
如果 exact oracle pass 但 compressed fail：
  继续 compression，不进入 KAN。

如果 exact oracle fail：
  停止函数空间自然更新主线。

如果 only MLP MNIST pass：
  判定为 dataset-specific weak signal，不继续。

如果 synthetic pass but real fail：
  回到 signal/reservoir decomposition。
```

---

## 11. P7：现有几何优化器对照

### 11.1 目标

判断我们的方法是否真的有独立价值，而不是被现有优化器风格支配。

### 11.2 对照方法

```text
AdamW
SAM-like perturbation
K-FAC-like block approximation
SOAP/Shampoo-like eigenbasis approximation
Muon-like orthogonalized update
diagonal SNR preconditioner
```

### 11.3 判断标准

```text
如果 existing optimizer 在 >= 6/8 task 上更好且成本更低：
  PSFP 降级，不进入 KAN。

如果 PSFP 在 signal/noise separation 上明显更好：
  保留为 research candidate。

如果两者互补：
  做 hybrid，但必须先证明 hybrid 不是手工规则。
```

---

## 12. P8：KAN graph-free 迁移 gate

### 12.1 目标

只有普通多层感知机上通过，才迁移到 KAN。不能让 KAN 替 function-space 更新背锅。

### 12.2 条件

```text
P6 strong pass = 1
P7 independent value pass = 1
cost_ratio discovery <= 3.0
```

### 12.3 KAN 迁移前检查

```text
graph-free JVP available
graph-free VJP available
manual adjoint correct
kernel path not autograd official
cost estimate <= discovery budget
```

### 12.4 不满足时 Codex 先尝试

```text
如果 KAN JVP/VJP 缺失：
  不迁移；先实现 graph-free adjoint。

如果 KAN cost 过高：
  不迁移；做 blockwise / low-rank approximation。

如果 KAN primitive 本身效率不过：
  回到 primitive/kernel-native route。
```

---

## 13. P9：最终路线决策

### Case A：Exact oracle 失败

结论：

```text
当前函数空间自然更新主线停止。
不要再修 PSFP/FNSU。
转向 primitive / architecture / compositional base。
```

### Case B：Exact oracle 成功，compressed 失败

结论：

```text
函数空间目标有价值，但当前压缩失败。
继续研究 low-rank / Kronecker / blockwise approximation。
不进入 KAN official。
```

### Case C：Compressed 成功但 cost 高

结论：

```text
保留 discovery；不写 system success。
优先 kernelization / amortization。
```

### Case D：Compressed 成功且 cost 可接受

结论：

```text
进入 KAN graph-free migration。
```

### Case E：Existing optimizer dominates

结论：

```text
停止自研 PSFP 主线；
把研究重点转向将 KAN primitive 与现有几何优化器结合。
```

---

## 14. 必须可视化

```text
P1 cost waterfall
P2 exact oracle training curves
P2 real-vs-noise improvement bars
P3 split consistency heatmap
P4 exact-vs-compressed cosine plot
P4 cost-quality Pareto
P5 AdamW coupling loss spike plot
P6 short training val_loss_auc dashboard
P7 optimizer comparison Pareto
P9 route decision matrix
```

---

## 15. 最终判断标准

v11.3 不是为了“再救 v11.2”。它是为了裁决：

$$
\boxed{
\text{函数空间自然更新到底是理论上没用，还是只是当前近似太贵太差？}
}
$$

只要 exact oracle 都不能稳定赢，当前主线就应该停止。

只要 exact oracle 能赢但压缩不能赢，项目就应该转向近似和 kernelization。

只要 compressed preconditioner 能赢且成本可控，才允许进入 KAN 和 future path。

