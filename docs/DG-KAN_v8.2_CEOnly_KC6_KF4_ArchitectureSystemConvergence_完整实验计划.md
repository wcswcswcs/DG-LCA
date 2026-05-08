# DG-KAN v8.2 CE-Only Architecture-System Convergence：KC6 质量路径与 KF4 系统路径合流完整实验计划

> 本计划基于 v8.1 CE-only/no-teacher/no-loss 实验复盘重新制定。  
> 本轮必须继续坚持：**不要 teacher、不要 self-teacher、不要 distillation、不要改 loss、不要 sampler/class weight、不要 optimizer sweep**。  
> v8.2 的核心目标不是继续做零散修补，而是把 v8.1 暴露出的两个半成功路径合流：  
> **KC6 有 CE-only architecture macro advantage，但 S2 没闭合；KF4 有 GPU-native FullGridS2，但 macro advantage 掉了。**  
> v8.2 要回答的本质问题是：**为什么同样 CE-only/no-teacher/strict PureKAN 路线下，质量路径和系统路径分裂？能否构造一个同时具备 KC6 的 macro 与 KF4 的 S2 的候选？**

---

## 0. 当前状态判断

### 0.1 目标是否已经达到

没有。

v8.1 已经严格执行了 CE-only/no-teacher/no-loss contract。官方 route 不允许 external teacher，不允许 self-teacher，不允许 margin/calibration/NLL-balanced/focal/label smoothing 等特殊 loss。所有 official candidate 都必须满足：

```text
external_teacher_used = 0
self_teacher_used = 0
loss_type = CE
special_loss_used = 0
distill_loss_used = 0
margin_loss_used = 0
calibration_loss_used = 0
nll_balanced_loss_used = 0
label_smoothing_used = 0
focal_loss_used = 0
class_weight_used = 0
sampler_changed = 0
```

v8.1 minimum success 需要：

$$
\text{CodeNativePass}
\land
\text{NoTeacherNoLossModificationPass}
\land
\text{StrictPass}
\land
\text{GradPass}
\land
\text{TeacherFreeCEMacroPass}
\land
\text{FullGridS2Pass}.
$$

当前没有任何 candidate 同时满足 `TeacherFreeCEMacroPass` 和 `FullGridS2Pass`，因此 v8.1 没有达成 minimum success。

---

### 0.2 v8.1 的关键结果

v8.1 主 run 中，M13 复现了 teacher-free positive signal，但没有过 hard gate：

```text
M13 macro val gap = +0.017839
CI95 low = +0.012240
Holm p = 7.25e-06
test gap = +0.022982
GradPass = 1
S2 shapes = 6/9
```

A2S/M9 在 v8.1 主 run 中没有复现此前 5-seed 的 `+0.019921875`，10-seed 下回到：

```text
A2S macro val gap = +0.016992
M9  macro val gap = +0.016992
```

这说明 A2S/M9 的 near-pass 不够稳定，不能继续当作已经接近成功的主证据。

v8.1 architecture matrix smoke 证明 RR3/RR9/RR14 这些 CE-only architecture primitives 可以在 5-seed 下跨过 macro gate：

```text
RR3  macro val gap = +0.021224
RR14 macro val gap = +0.021094
RR9  macro val gap = +0.021094
```

但它们都没有 S2，且只测 batch128 smoke，因此不能作为 official success。RR14 进入 full-grid 后，5-seed 仍能过 macro，但 memory max `1.614004`、step max `2.738844`、S2 `0/9`；10-seed full-grid confirmation 回落到 `+0.019401`，也未过 macro gate，S2 仍是 `0/9`。

随后 v8.1 追加了真正 CE-only kernel-native repair path。KC1/KC3/KC6 证明 CE-only/no-teacher kernel-native stack 可以真实跨过 macro gate。最重要的是 KC6：

```text
KC6 macro val gap = +0.024349
CI95 low = +0.015234
Holm p = 0.001569
test gap = +0.023177
GradPass = 1
memory max = 1.053081
step max = 1.570884
S2 shapes = 5/9
```

KC6 的意义很大：它第一次在严格 CE-only/no-teacher 路线下把 macro 推到明显超过 `+0.0200`，说明 architecture-only advantage 已经真实出现。但 KC6 没有 FullGridS2：bs512 memory 仍卡在 `1.05308`，并且部分 step shape 也略超界。

v8.1 进一步做 GPU-native allocator/lifetime repair。KC7 用 in-place FP32 `addcdiv_` update 试图去掉额外 FP32 step tensor，但结果是：

```text
KC7 macro gap = +0.017969
memory max = 1.053081
step max = 1.363327
S2 shapes = 6/9
```

KC7 说明：FP32 AdamW step tensor不是 bs512 memory miss 的根因；它也没有保住 KC6 的 macro。

之后做 GPU-side instrumentation 与 fused backward。KF4 成为当前最好系统候选：

```text
KF4 macro gap = +0.018359
memory max = 1.046778
step max = 1.449435
FullGridS2 = 9/9
GradPass = 1
```

KF4 的意义也很大：它说明 CE-only/no-teacher/strict PureKAN 现在可以真正进入 FullGridS2。但它损失了 KC6 的 macro，尤其 KMNIST gap 明显下降。

最后 v8.1 做 KC6/KF4 同协议 co-run，并尝试 KF5/KF6 合并：

```text
KC6: macro +0.024349, memory 1.053081, step 1.570884, S2 5/9
KF4: macro +0.018359, memory 1.046778, step 1.449435, S2 9/9
KF5: macro +0.018229, memory 1.053081, step 1.570727, S2 5/9
KF6: macro +0.018099, memory 1.053081, step 1.518668, S2 5/9
```

KF5/KF6 没有完成合流：它们既没有恢复 KC6 的 macro，也没有恢复 KF4 的 FullGridS2。

---

## 1. 当前问题的本质

### 1.1 现在不是“有没有表达力”的问题

KC6 已经达到：

$$
\Delta Acc_{\text{macro,val}}=+0.024349.
$$

这说明 CE-only/no-teacher/strict architecture 路线已经能超过 `+0.0200` hard gate。项目已经不再是“PureKAN 结构是否有优势”的早期阶段，而是进入：

$$
\boxed{
\text{质量路径和系统路径怎样合流？}
}
$$

KC6 证明质量可以成立；KF4 证明系统 S2 可以成立。但二者没有同时成立。

---

### 1.2 现在也不是传统 optimizer 问题

v8.1 的关键对比已经排除了传统 optimizer sweep 的主导地位。

第一，KC7 用 in-place FP32 `addcdiv_` update 修掉了部分 update temp，但没有解决 bs512 memory，也没有保住 macro。说明单纯 update tensor / AdamW step tensor 不是根因。

第二，KF4 使用 ATen fused backward 与 addcdiv update，可以 FullGridS2，但 macro 下降到 `+0.018359`。这不是 lr/weight decay 小调可以解释的普通 optimizer 问题，因为 one-step backward delta 显示 KC6 与 KF4 在同 init、同 batch 的 loss/gradient 近乎一致，至少初始局部数学并没有明显错位。

第三，dataset-level macro 来源显示，KF4/KF5/KF6 相对 KC6 的损失主要来自 KMNIST：

```text
KC6 KMNIST gap = +0.026953
KF4 KMNIST gap = +0.014844
KF5 KMNIST gap = +0.016797
KF6 KMNIST gap = +0.019141
```

这更像是训练轨迹、数值路径、cache/fused update order、activation/backward实现细节对 hard dataset 的长期影响，而不是传统 optimizer 超参问题。

因此 v8.2 不允许 optimizer sweep。只允许做 **semantic-equivalent path audit** 和 **implementation-level trajectory audit**。

---

### 1.3 当前真正 blocker 是“质量路径与系统路径分裂”

当前有两个半成功候选：

```text
KC6:
  Quality path
  CE-only macro pass
  GradPass
  NoTeacherNoLossPass
  but S2 fail

KF4:
  System path
  FullGridS2Pass
  GradPass
  NoTeacherNoLossPass
  but macro fail
```

这说明：

$$
\boxed{
\text{我们不是缺一个新 loss，也不是缺 teacher，而是缺一个数学等价且系统等价的实现路径。}
}
$$

下一步必须围绕这个分裂做机制实验，而不是再从头扫很多 architecture primitives。

---

### 1.4 为什么 naive merge 失败很重要

KF5/KF6 的失败说明：简单把 KC6 的 explicit derivative 与 KF4 的 fused/in-place backward 或 addcdiv update拼在一起，不会自动合流。这里可能有几个深层原因：

```text
1. KC6 与 KF4 的 full-training trajectory 在很多 step 后分叉，虽然 one-step gradient 几乎一致；
2. fused backward / in-place update 改变了 tensor lifetime、rounding order 或 update order；
3. addcdiv update、fused SiLU backward、explicit derivative 三者在有限精度下不是完全轨迹等价；
4. KMNIST 更敏感，放大了微小数值轨迹差异；
5. S2 的 memory优化路径可能改变了 backward temp 的生存期和 allocator行为，间接改变 timing/seed/order；
6. 当前 co-run 没有完整记录每 step 参数轨迹、optimizer state轨迹、activation分布和hard-sample行为。
```

因此 v8.2 要从“试候选”切换到“轨迹归因”。

---

## 2. v8.2 总体目标

v8.2 的总体目标是：

$$
\boxed{
\text{构造一个同时继承 KC6 macro advantage 和 KF4 FullGridS2 的 CE-only/no-teacher PureKAN candidate。}
}
$$

更具体地，v8.2 要回答以下问题：

```text
Q1:
  KC6 与 KF4 的差异到底来自 forward function、backward gradient、optimizer update、参数初始化、数值精度、in-place lifetime，还是训练轨迹累积？

Q2:
  KC6 的 S2 fail 到底来自 bs512 root/checkpoint live-set、backward recompute temp、allocator block、optimizer state，还是 step measurement drift？

Q3:
  KF4 的 macro drop 到底来自 KMNIST hard samples、feature rank、margin、NLL/accuracy tradeoff，还是 fused path 的长期轨迹偏移？

Q4:
  能否做一个 semantic-equivalent fusion：在不改 loss、不用 teacher、不改 optimizer超参的情况下，把 KC6 的质量路径和 KF4 的 S2路径合并？

Q5:
  如果不能合并，是质量路径必须牺牲 S2，还是系统路径需要重新设计更低级 fused backward / static workspace？
```

---

## 3. v8.2 成功标准

### 3.1 Minimum Success

v8.2 minimum success 定义为：

$$
\boxed{
\text{NoTeacherNoLossPass}
\land
\text{StrictPass}
\land
\text{GradPass}
\land
\text{TeacherFreeCEMacroPass}
\land
\text{FullGridS2Pass}
}
$$

具体数值为：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200,
$$

$$
CI_{95\%,macro}^{low}>0,
$$

$$
p_{\text{Holm}}<0.05,
$$

$$
\Delta Acc_{\text{macro,test}}\geq0.015,
$$

同时：

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

且必须 9/9 shapes pass：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
batch sizes = 128, 256, 512
```

### 3.2 Formal System Success

在 minimum success 上进一步要求：

$$
r_{\text{mem,max}}<1.00,
$$

$$
r_{\text{step,max}}\leq1.35,
$$

$$
ValLossAUC_{\text{time,KAN}}\leq ValLossAUC_{\text{time,MLP}},
$$

并且：

```text
ProfilerPass = true
mapped_kernel_time_fraction >= 0.90
unknown_kernel_time_fraction <= 0.10
top3 phase time explain >= 0.70
```

### 3.3 Mechanism Success

即使没有一次达成 formal success，v8.2 至少必须给出机制闭环。Mechanism Success 要求：

```text
KC6 vs KF4 trajectory divergence source identified
bs512 memory source quantified
KMNIST macro drop source identified
candidate route can be classified as quality-path / system-path / merged-path
```

量化要求：

$$
\frac{\sum_{i=1}^{3}M_{\text{top memory source},i}}{M_{\text{peak gap}}}\geq0.70.
$$

若无法解释 `KC6 -> KF4` 的 macro 差异，则不能继续 blind merging。

---

## 4. 明确禁止事项

v8.2 继续严格禁止：

```text
external teacher
self teacher
distillation
self-distillation
margin loss
calibration loss
NLL-balanced loss
label smoothing
focal loss
classwise loss
sampler / class weight
test-based selection
CPU offload
optimizer hyperparameter sweep
```

官方候选必须满足：

```text
external_teacher_used = 0
self_teacher_used = 0
loss_type = CE
special_loss_used = 0
cpu_offload_used = 0
nonKAN_param_count = 0
manual_forward = 1
manual_backward = 1
manual_update = 1
uses_loss_backward = 0
```

允许的修改仅限：

```text
1. architecture implementation path；
2. backward implementation path；
3. cache / root checkpoint / workspace lifetime；
4. in-place vs out-of-place update implementation；
5. deterministic / numerical-equivalence instrumentation；
6. kernel fusion / static workspace；
7. allocator-level GPU-native memory repair。
```

这些都必须保持数学目标不变：

$$
L = CE(y,p_{\theta}(x)).
$$

---

## 5. 核心假设

### H1：KC6 与 KF4 的 forward function 是等价或近等价的，macro差异来自训练轨迹而不是初始函数类

H1 基于 one-step backward delta：KC6 与 KF4 在同 init、同 dataset、同 batch 下的 loss delta 为 0，gradient差异极小。因此两者初始局部数学几乎等价。

H1 成立标准：

在固定 init、固定 batch、固定 eval mode 下：

$$
\max_x |f_{\text{KC6}}(x)-f_{\text{KF4}}(x)|\leq10^{-6},
$$

并且：

$$
\frac{\|\nabla_{\theta}L_{\text{KC6}}-\nabla_{\theta}L_{\text{KF4}}\|_2}
{\|\nabla_{\theta}L_{\text{KC6}}\|_2}\leq10^{-5}.
$$

如果 H1 成立但 full training macro差异仍大，则要进入 H2 轨迹假设。

### H2：KC6/KF4 macro差异来自多步训练轨迹分叉

H2 假设微小数值/更新顺序差异经过 240 steps 后在 KMNIST 放大。

记录每个 seed、dataset 的 step-wise trajectory：

```text
parameter_delta_l2
optimizer_state_m_delta_l2
optimizer_state_v_delta_l2
grad_l2
grad_cos_vs_KC6
activation_mean/std
logit_margin_p10
train_loss
val_loss
KMNIST hard-sample accuracy
```

H2 成立标准：

如果在 early steps 出现：

$$
\cos(g_{\text{candidate}},g_{\text{KC6}})<0.999
$$

或：

$$
\frac{\|\theta_t^{candidate}-\theta_t^{KC6}\|_2}{\|\theta_t^{KC6}\|_2}>10^{-4}
$$

并且该分叉与 KMNIST val gap下降同步，则判定为 trajectory divergence。

### H3：KC6 的 S2 fail 来自 bs512 live-set，而不是宏观结构表达力

KC6 的质量过线，但 S2 fail 主要集中在 bs512 memory `1.053081`，超出 S2 gate `1.05` 很小：

$$
1.053081-1.05=0.003081.
$$

H3 假设：这是 live-set / allocator / root checkpoint / backward temp 问题，不是表达力必需状态。

H3 成立标准：

GPU live-set attribution 能解释至少 70% 的 KAN-over-S2 gap：

$$
\frac{M_{\text{top1}}+M_{\text{top2}}+M_{\text{top3}}}{M_{\text{peak gap}}}\geq0.70.
$$

并且至少一个 repair 能满足：

$$
r_{\text{mem,max,new}}\leq1.05,
$$

且：

$$
\Delta Acc_{\text{macro,val,new}}\geq0.0200.
$$

### H4：KF4 的 macro drop 主要来自 KMNIST hard samples，而不是全局表达力下降

dataset-level gap显示 KF4 相比KC6 主要损失在 KMNIST：

$$
\Delta_{\text{KMNIST}}(\text{KF4-KC6})=-0.012109.
$$

H4 成立标准：

若：

```text
KMNIST hard-sample overlap explains >= 60% of macro drop
```

或：

$$
margin_{p10,KF4}^{KMNIST}<0.95 \cdot margin_{p10,KC6}^{KMNIST},
$$

则说明 KF4 的 system path 伤害了 hard-sample margin。

### H5：正确合流路线必须先保持 KC6 训练轨迹，再引入 KF4 系统优化

H5 的核心是“先质量，再系统”。也就是说，候选必须先证明 trajectory 与 KC6 近似，再看是否进入 S2；不能反过来先用 KF4 S2 再试图恢复质量。

H5 成立标准：

candidate 在前 50 steps 内满足：

$$
\cos(g_{\text{candidate}},g_{\text{KC6}})\geq0.9999,
$$

$$
\frac{\|\theta_t^{candidate}-\theta_t^{KC6}\|_2}{\|\theta_t^{KC6}\|_2}\leq10^{-5},
$$

并且最终：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

---

## 6. Candidate 设计

### 6.1 Baselines

```text
B0-MLP-AdamW-CE
KC6-quality-path-baseline
KF4-system-path-baseline
KF5-naive-merge-diagnostic
KF6-naive-merge-diagnostic
```

### 6.2 Trajectory audit candidates

这些候选不一定为了成功，而是为了定位差异来源：

```text
TA0-KC6-reference
TA1-KF4-reference
TA2-KC6-with-KF4-update-order-only
TA3-KC6-with-KF4-fused-backward-only
TA4-KC6-with-KF4-addcdiv-only
TA5-KF4-with-KC6-explicit-derivative-only
TA6-KF4-with-KC6-update-order-only
TA7-KC6-deterministic-no-inplace-shadow
TA8-KF4-deterministic-no-inplace-shadow
```

### 6.3 Quality-preserving S2 candidates

这些候选必须优先保持 KC6 trajectory：

```text
QS0-KC6-current
QS1-KC6-static-workspace-root-input
QS2-KC6-root-checkpoint-view-no-copy
QS3-KC6-root-checkpoint-lifetime-split
QS4-KC6-backward-temp-ring-buffer
QS5-KC6-prefix-recompute-static-buffer
QS6-KC6-fused-explicit-silu-backward-no-temp
QS7-KC6-update-phase-preallocated-denom
QS8-KC6-quality-preserving-S2-combo
```

### 6.4 System-preserving quality recovery candidates

这些候选以 KF4 为系统基底，但只允许做数学等价修复，不许改 loss：

```text
SQ0-KF4-current
SQ1-KF4-KC6-init-policy
SQ2-KF4-KC6-update-order
SQ3-KF4-KC6-explicit-silu-derivative-numerics
SQ4-KF4-no-addcdiv-update-but-ATen-fused-backward
SQ5-KF4-KMNIST-trajectory-stability-mode
SQ6-KF4-quality-recovered-combo
```

注意：`SQ5` 不能改 loss、不能改 sampler、不能用 class weight。它只能改变 deterministic implementation path，例如 update order、dtype、in-place boundary、activation cache/lifetime。

### 6.5 Exact live-set candidates

用于确认 memory source：

```text
LS0-KC6-live-set-baseline
LS1-KC6-root-checkpoint-only
LS2-KC6-no-head-cache-overlap
LS3-KC6-stack-backward-temp-only
LS4-KC6-update-phase-only
LS5-KC6-allocator-prewarm
LS6-KC6-static-buffer-owner
LS7-KC6-record-stream-lifetime-safe
```

---

## 7. 实验阶段总览

v8.2 分为七个 Wave：

```text
Wave 0:
  Contract reproduction and exact co-run lock

Wave 1:
  KC6/KF4 trajectory divergence audit

Wave 2:
  GPU live-set and bs512 memory attribution

Wave 3:
  Quality-preserving S2 repair

Wave 4:
  System-preserving quality recovery

Wave 5:
  Unified 10-seed co-selection

Wave 6:
  Formal system closure and scaling diagnostics
```

---

## 8. Wave 0：Contract reproduction and exact co-run lock

### P0：CE-only/no-teacher/no-loss contract audit

目标：确认所有 v8.2 candidates 都保持官方 contract。

必须记录：

```text
candidate_id
external_teacher_used
self_teacher_used
loss_type
special_loss_used
class_weight_used
sampler_changed
cpu_offload_used
nonKAN_param_count
manual_forward
manual_backward
manual_update
uses_loss_backward
GradPass
fake_data_used
proxy_row_used
```

判断标准：

```text
external_teacher_used = 0
self_teacher_used = 0
loss_type = CE
special_loss_used = 0
cpu_offload_used = 0
fake/proxy nonzero = 0
```

可视化：

```text
p0_contract_heatmap.svg
p0_teacher_loss_violation_matrix.svg
p0_candidate_registry_graph.svg
```

### P1：KC6/KF4 reproduction under exact same protocol

目标：在完全相同 run、相同 seeds、相同数据、相同 benchmark 下复现：

```text
KC6 quality path
KF4 system path
```

必须记录：

```text
macro_gap
CI95_low
Holm_p
test_gap
ECE_delta
NLL_delta
GradPass
memory_ratio_max
step_ratio_max
S2_shape_count
per_dataset_gap
per_shape_efficiency
```

判断标准：

KC6 reproduction：

$$
|\Delta Acc_{\text{KC6}}-0.024349|\leq0.004.
$$

KF4 reproduction：

$$
|\Delta Acc_{\text{KF4}}-0.018359|\leq0.004.
$$

Efficiency reproduction：

$$
|r_{\text{mem,KF4}}-1.046778|\leq0.015.
$$

$$
|r_{\text{mem,KC6}}-1.053081|\leq0.015.
$$

可视化：

```text
p1_kc6_kf4_task_bar.svg
p1_per_dataset_gap_bar.svg
p1_s2_shape_heatmap.svg
p1_quality_system_pareto.svg
```

---

## 9. Wave 1：Trajectory divergence audit

### P2：one-step and multi-step exact delta

目标：确认 KC6/KF4 在第 0 step 是否等价，在多步后何时分叉。

设置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
steps = 0,1,2,5,10,20,50,120,240
batch schedule fixed
dropout = disabled if any
deterministic flag on
```

必须记录：

```text
step
dataset
seed
candidate
train_loss
val_loss
logit_abs_max_delta_vs_KC6
grad_abs_max_delta_vs_KC6
grad_rel_l2_vs_KC6
grad_cos_vs_KC6
param_rel_l2_vs_KC6
optimizer_m_rel_l2_vs_KC6
optimizer_v_rel_l2_vs_KC6
activation_mean_delta
activation_std_delta
margin_p10
hard_sample_acc
```

判断标准：

若 early steps 中：

$$
grad\_cos_{\text{candidate vs KC6}}<0.9999
$$

或：

$$
param\_rel\_l2>10^{-5},
$$

且该分叉在 KMNIST 上与 val gap下降同步，则 H2 成立。

可视化：

```text
p2_grad_cos_over_steps.svg
p2_param_divergence_over_steps.svg
p2_margin_p10_over_steps.svg
p2_kmnist_hard_sample_acc.svg
p2_loss_curve_overlay.svg
```

### P3：implementation factor isolation

目标：把差异拆成 backward、update、in-place、dtype、order 四个因素。

候选：

```text
TA0-KC6-reference
TA1-KF4-reference
TA2-KC6-with-KF4-update-order-only
TA3-KC6-with-KF4-fused-backward-only
TA4-KC6-with-KF4-addcdiv-only
TA5-KF4-with-KC6-explicit-derivative-only
TA6-KF4-with-KC6-update-order-only
TA7-KC6-deterministic-no-inplace-shadow
TA8-KF4-deterministic-no-inplace-shadow
```

必须记录：

```text
factor_changed
macro_gap_3seed
KMNIST_gap
grad_cos_vs_KC6
param_divergence_step50
memory_ratio_max
step_ratio_max
GradPass
```

判断标准：

单因素导致 macro 降低超过：

$$
0.003
$$

或 KMNIST gap 降低超过：

$$
0.006
$$

则该因素是主要风险因素。

可视化：

```text
p3_factor_macro_delta_waterfall.svg
p3_factor_kmnist_delta_bar.svg
p3_factor_efficiency_pareto.svg
```

---

## 10. Wave 2：GPU live-set and bs512 memory attribution

### P4：exact live-set attribution for KC6 bs512

目标：解释 KC6 bs512 memory `1.053081` 为什么只差 S2 `0.003081`。

设置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
batch = 512 primary
also batch = 128,256 for contrast
memory history = on
CUDA allocated/reserved = recorded
phase NVTX ranges = forward, backward_stack, backward_head, update
```

必须记录：

```text
phase
peak_allocated_MB
peak_reserved_MB
phase_start_allocated_MB
phase_end_allocated_MB
root_input_cache_MB
hidden_y_cache_MB
head_cache_MB
stack_backward_temp_MB
optimizer_state_MB
update_temp_MB
allocator_padding_MB
reserved_unallocated_MB
largest_live_tensor_MB
top1_memory_source
top2_memory_source
top3_memory_source
unknown_memory_fraction
```

判断标准：

MemoryAttributionPass：

```text
unknown_memory_fraction <= 0.10
top3 memory sources identified
```

Actionable source：

$$
\frac{M_{\text{source}}}{M_{\text{peak gap}}}\geq0.20.
$$

可视化：

```text
p4_bs512_memory_waterfall.svg
p4_phase_peak_timeline.svg
p4_top_memory_sources_bar.svg
p4_batch_memory_scaling.svg
```

### P5：allocator baseline and workspace sensitivity

目标：判断 S2 miss 是否来自 allocator/padding，而不是 tensor本身。

实验：

```text
allocator prewarm on/off
static workspace on/off
record_stream safe lifetime on/off
empty_cache before measure on/off
fixed warmup 20/50/100
reps 100/200
```

必须记录：

```text
allocator_config
memory_ratio_max
reserved_ratio_max
allocated_ratio_max
step_ratio_max
variance_across_reps
S2_shape_count
```

判断标准：

如果 allocator策略让：

$$
r_{\text{mem,max}}\leq1.05
$$

且不改变 macro/GradPass，则 S2 miss 是 allocator-sensitive。

可视化：

```text
p5_allocator_sensitivity_boxplot.svg
p5_allocated_vs_reserved_scatter.svg
p5_memory_variance_by_protocol.svg
```

---

## 11. Wave 3：Quality-preserving S2 repair

### P6：KC6 quality-preserving memory repair

目标：从 KC6 出发，只做不改变训练轨迹的 memory repair。

候选：

```text
QS0-KC6-current
QS1-KC6-static-workspace-root-input
QS2-KC6-root-checkpoint-view-no-copy
QS3-KC6-root-checkpoint-lifetime-split
QS4-KC6-backward-temp-ring-buffer
QS5-KC6-prefix-recompute-static-buffer
QS6-KC6-fused-explicit-silu-backward-no-temp
QS7-KC6-update-phase-preallocated-denom
QS8-KC6-quality-preserving-S2-combo
```

必须记录：

```text
candidate
repair_component
macro_gap_5seed
KMNIST_gap
GradPass
grad_relerr_max
trajectory_cos_vs_KC6_step50
param_rel_l2_vs_KC6_step50
memory_ratio_max
step_ratio_max
S2_shape_count
top_memory_source_reduction
```

判断标准：

Quality-preserving S2 repair pass：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200,
$$

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50,
$$

and:

$$
trajectory\_cos_{\text{step50}}\geq0.9999.
$$

可视化：

```text
p6_quality_preserving_pareto.svg
p6_memory_reduction_by_component.svg
p6_trajectory_preservation_bar.svg
```

---

## 12. Wave 4：System-preserving quality recovery

### P7：KF4 quality recovery without changing loss

目标：从 KF4 出发，恢复 KC6 的 macro，但保持 S2。

候选：

```text
SQ0-KF4-current
SQ1-KF4-KC6-init-policy
SQ2-KF4-KC6-update-order
SQ3-KF4-KC6-explicit-silu-derivative-numerics
SQ4-KF4-no-addcdiv-update-but-ATen-fused-backward
SQ5-KF4-KMNIST-stability-deterministic-path
SQ6-KF4-quality-recovered-combo
```

必须记录：

```text
candidate
changed_factor
macro_gap_5seed
per_dataset_gap
KMNIST_gap
ECE_delta
NLL_delta
GradPass
memory_ratio_max
step_ratio_max
S2_shape_count
grad_cos_vs_KC6
param_divergence_step50
```

判断标准：

System-preserving quality recovery pass：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200,
$$

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

可视化：

```text
p7_kf4_quality_recovery_bar.svg
p7_dataset_gap_recovery.svg
p7_s2_quality_pareto.svg
```

---

## 13. Wave 5：Unified official co-selection

### P8：10-seed official co-selection

目标：把所有候选放到同一 protocol 下比较，不能用 5-seed near-pass 做成功结论。

必跑：

```text
B0
KC6
KF4
best-QS
best-SQ
KF5/KF6 diagnostic if still relevant
```

设置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
batch task = 128
bench batch = 128,256,512
grad batch = 8,128
bootstrap reps = 10000
```

必须记录：

```text
candidate
macro_gap
CI95_low
Holm_p
test_gap
per_dataset_gap
ECE_delta
NLL_delta
GradPass
grad_relerr_max
memory_ratio_max
step_ratio_max
S2_shape_count
S1_shape_count
ValLossAUC_step
ValLossAUC_time
NoTeacherNoLossPass
StrictPass
```

判断标准：

Minimum success：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200,
$$

$$
CI_{95\%,macro}^{low}>0,
$$

$$
p_{\text{Holm}}<0.05,
$$

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

可视化：

```text
p8_official_scorecard.svg
p8_seedwise_gap_boxplot.svg
p8_quality_system_pareto.svg
p8_s2_heatmap.svg
p8_route_decision_tree.svg
```

---

## 14. Wave 6：Formal system closure

### P9：TimeAUC accounting

目标：minimum success 后，检查 wall-clock 是否真正优于 MLP。

必须记录：

```text
wall_clock_time_total
train_step_time
forward_time
backward_time
update_time
validation_time
logging_time
cuda_sync_time
data_loading_time
unknown_time_fraction
ValLossAUC_step
ValLossAUC_time_train_only
ValLossAUC_time_total
time_to_target_acc
```

判断标准：

TimeAccountingPass：

```text
unknown_time_fraction <= 0.10
```

TimeAUCPass：

$$
ValLossAUC_{\text{time,KAN}}\leq ValLossAUC_{\text{time,MLP}}.
$$

可视化：

```text
p9_time_breakdown_stacked.svg
p9_step_auc_vs_time_auc.svg
p9_time_to_target.svg
```

### P10：phase-mapped kernel profiler

目标：真实 kernel attribution。

必须记录：

```text
kernel_count_total
mapped_kernel_time_fraction
unknown_kernel_time_fraction
kernel_count_forward
kernel_count_backward
kernel_count_update
top_kernel_name_1
top_kernel_phase_1
top_kernel_time_1
small_kernel_count_under_10us
layout_conversion_count
cuda_memcpy_time
cuda_sync_time
```

判断标准：

```text
mapped_kernel_time_fraction >= 0.90
unknown_kernel_time_fraction <= 0.10
top3 phase time explain >= 0.70
```

可视化：

```text
p10_kernel_timeline.svg
p10_kernel_count_by_phase.svg
p10_top_kernel_time_bar.svg
p10_small_kernel_fragmentation.svg
```

### P11：S1 memory package

目标：如果 S2 成立，继续推进 S1。

必须记录：

```text
memory_ratio_max
step_ratio_max
root_input_cache_MB
backward_temp_MB
optimizer_state_MB
allocator_padding_MB
reserved_unallocated_MB
top_memory_source
```

S1 pass：

$$
r_{\text{mem,max}}<1.00,
$$

$$
r_{\text{step,max}}\leq1.35.
$$

可视化：

```text
p11_s1_memory_waterfall.svg
p11_s1_candidate_pareto.svg
```

---

## 15. Artifact 要求

必须落盘：

```text
run_manifest.json
candidate_registry_v82.csv
contract_no_teacher_no_loss.csv
p0_contract_audit.csv
p1_reproduction.csv
p2_trajectory_divergence.csv
p3_factor_isolation.csv
p4_gpu_liveset_attribution.csv
p5_allocator_sensitivity.csv
p6_quality_preserving_s2_repair.csv
p7_system_preserving_quality_recovery.csv
p8_official_coselection.csv
p9_time_accounting.csv
p10_phase_mapped_profiler.csv
p11_s1_memory_package.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

failure taxonomy：

```text
F1_teacher_or_loss_violation
F2_contract_fail
F3_grad_fail
F4_macro_fail
F5_s2_fail
F6_s1_fail
F7_time_auc_fail
F8_profiler_incomplete
F9_trajectory_divergence_unexplained
F10_memory_attribution_incomplete
F11_naive_merge_fail
F12_fake_or_proxy_violation
F13_artifact_missing
```

---

## 16. 第一轮执行顺序

第一轮不再跑大量新 architecture。直接围绕 KC6/KF4 分裂执行。

```text
Step 1:
  P0 + P1
  复现 KC6/KF4 的质量-系统分裂。

Step 2:
  P2 + P3
  做轨迹分叉和 factor isolation。

Step 3:
  P4 + P5
  精确解释 KC6 bs512 memory miss。

Step 4:
  P6
  从 KC6 出发做 quality-preserving S2 repair。

Step 5:
  P7
  从 KF4 出发做 system-preserving quality recovery。

Step 6:
  P8
  统一 10-seed co-selection。

Step 7:
  P9-P11
  只有 P8 minimum success 后进入 formal system closure。
```

---

## 17. 停止条件

### 成功停止

出现下列任一情况立即复盘：

```text
CE-only/no-teacher candidate:
  macro gap >= +0.0200
  CI95 low > 0
  Holm p < 0.05
  GradPass = 1
  FullGridS2 = 1
```

formal success：

```text
minimum success + FullGridS1 + TimeAUCPass + ProfilerPass
```

### 失败停止

出现下列情况停止对应路线：

```text
1. KC6 reproduction macro < +0.0200；
2. KF4 reproduction S2 != 9/9；
3. trajectory divergence cannot be measured；
4. top memory sources explain < 70% of KC6 bs512 gap；
5. all QS repairs fail to reduce memory below 1.05；
6. all SQ repairs fail to recover macro above +0.0200；
7. GradPass fails and cannot be fixed by numerically equivalent rewrite；
8. any official candidate uses teacher/loss/CPU offload；
9. no-fake/no-proxy audit fails。
```

---

## 18. 最终解释规则

### Case A：KC6-derived QS candidate succeeds

如果 QS candidate 同时达到 macro 与 S2，说明质量路径本身是正确的，S2 blocker 是 live-set / allocator / backward-temp 实现问题。

### Case B：KF4-derived SQ candidate succeeds

如果 SQ candidate 同时达到 macro 与 S2，说明系统路径是正确的，之前 macro 掉线来自可修复的 numerical / trajectory issue。

### Case C：KC6 保质量但永远不进 S2

说明当前 quality path 有不可忽略 live-set 代价，需要重新设计 architecture primitive，而不是再做 memory小修。

### Case D：KF4 保 S2但永远不恢复 macro

说明 fused system path改变了训练轨迹或有效函数类，需要回到 KC6语义路径做更低层 fused backward。

### Case E：KC6/KF4 one-step等价但 multi-step分叉

说明问题是长期训练轨迹而非局部梯度错误。下一步应围绕 deterministic update order、in-place boundary、floating-point rounding、optimizer state lifetime 做更细的轨迹对齐。

### Case F：memory gap无法解释

如果 GPU live-set attribution解释不了 KC6 bs512 miss，则不能继续盲写 kernel。必须先补 profiler/memory snapshot。

---

## 19. 最终建议

v8.2 的一句话策略是：

$$
\boxed{
\text{停止扩散式 architecture 搜索，集中合流 KC6 质量路径与 KF4 系统路径。}
}
$$

当前最重要的事实是：

```text
KC6:
  已经证明 CE-only/no-teacher architecture macro 可以过 +2%，但 S2 差一点。

KF4:
  已经证明 GPU-native fused path 可以 FullGridS2，但 macro 掉线。

KF5/KF6:
  naive merge 失败。
```

所以 v8.2 的主线不是 teacher、不是 loss、不是 optimizer sweep，而是：

```text
1. 轨迹归因；
2. bs512 live-set attribution；
3. quality-preserving S2 repair；
4. system-preserving quality recovery；
5. unified 10-seed co-selection。
```

最终目标保持不变：

$$
\boxed{
\text{PureKAN-NG 自身，在 CE-only/no-teacher/no-loss 条件下，系统性超过 MLP-AdamW。}
}
