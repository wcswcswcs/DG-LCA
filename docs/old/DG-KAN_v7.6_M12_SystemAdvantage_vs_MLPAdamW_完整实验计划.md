# DG-KAN v7.6：从 M12 S2-MacroSignificant 到 MLP-AdamW 全面优势的机制确认与系统闭环实验计划

> 本计划基于 v7.5 最新最终复盘制定。v7.5 的关键变化是：最终追加自修复后，`M12` 首次同时满足 `StrictPass + GradPass + MacroSignificantPass + FullGridS2Pass`。这意味着项目已经达到 v7.5 的最低成功标准，但还不能宣称完成终极目标。  
> v7.6 的目标不是继续做 loss、sampler、classwise 或小型 optimizer sweep，也不是为了再刷一点 validation accuracy。v7.6 要回答一个更本质的问题：**M12 相比 MLP-AdamW 的优势，是一个真实、可复现、可解释、可扩展、具备 wall-clock 意义的 PureKAN-NG 系统优势，还是仅在当前 5-seed / small-dataset / S2-gate 下成立的阶段性成功？**

---

## 0. v7.5 最新结果的判断

### 0.1 最低目标已经达成

v7.5 的最终确认 run 中，最佳 candidate 是：

```text
M12 = A2S-fused-linear-silu-packed-generic-head-M5init-logit-distill-from-C3-T4-alpha025
```

最终 route 为：

```text
route = S2-MacroSignificantSuccess
strict_pass = 1
grad_pass = 1
macro_pass_v75 = 1
fullgrid_s2_pass = 1
success_v75_minimum = 1
primary_blocker = none
```

关键 task 指标：

```text
M12 val acc  = 0.8651
B0 MLP val acc = 0.8443
macro val gap = +0.0208
CI95 low = +0.0129
Holm p = 1.40e-03
test gap = +0.0214
ECE delta = +0.0024
NLL delta = -0.0914
```

关键 correctness 指标：

```text
GradPass = 6/6
max relerr = 8.07e-05
```

关键 full-grid efficiency 指标：

```text
memory ratio mean = 1.0182
step ratio mean = 1.3688
forward ratio mean = 2.0993
backward ratio mean = 1.1763
S2 shapes = 9/9
```

逐 shape S2 全部通过：

```text
MNIST bs128:
  memory = 0.9952
  step = 1.4490

MNIST bs256:
  memory = 1.0110
  step = 1.3643

MNIST bs512:
  memory = 1.0482
  step = 1.3917

Fashion-MNIST bs128:
  memory = 0.9952
  step = 1.3589

Fashion-MNIST bs256:
  memory = 1.0110
  step = 1.3522

Fashion-MNIST bs512:
  memory = 1.0482
  step = 1.3338

KMNIST bs128:
  memory = 0.9952
  step = 1.3622

KMNIST bs256:
  memory = 1.0110
  step = 1.3599

KMNIST bs512:
  memory = 1.0482
  step = 1.3466
```

因此 v7.5 已经真正完成了阶段性闭环：

$$
\boxed{
\text{StrictPass}+\text{GradPass}+\text{MacroSignificantPass}+\text{FullGridS2Pass}
}
$$

这不是 loss trick，也不是 classwise 修补。M12 的突破来自：

```text
A2S fused stack
packed generic V63 head
C3 teacher distillation
M5 initialization trajectory
FP32 SiLU derivative stable equivalent form
```

其中 SiLU derivative 的最终修复采用：

```python
sig = torch.sigmoid(y_i)
silu_prime = sig + y_i * (sig - sig.square())
delta = delta * silu_prime
```

它不是放宽 gradient gate，而是 FP32 内的数学等价重排，避免 FP64 额外 live-set。

---

## 1. 但现在还不能宣布终极成功

v7.5 达成的是最低成功标准，不是完整终极目标。当前仍有五个根本问题没有闭合。

### 1.1 M12 对 MLP-AdamW 是否有显著优势？

在 v7.5 当前口径下，答案是：

$$
\boxed{
\text{有 macro accuracy 的统计显著优势，并且 test 同向；但还不是全面系统优势。}
}
$$

M12 的 macro val gap 是：

$$
\Delta Acc_{\text{macro,val}}=+0.0208.
$$

它超过预设阈值：

$$
0.0208 > 0.0200.
$$

但安全边际只有：

$$
0.0208-0.0200=0.0008.
$$

这说明 M12 已经过线，但还不是“大幅碾压”。因此 v7.6 必须做 10-seed / 20-seed confirm，确认它不是 5-seed 下刚好过线。

### 1.2 M12 是 S2，不是 S1

S2 已经达成：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

但 S1 还没有达成。按 mean ratio：

$$
r_{\text{mem,mean}}=1.0182,
$$

$$
r_{\text{step,mean}}=1.3688.
$$

若目标是 mean S1：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35,
$$

则 memory 还需降低：

$$
1-\frac{1.00}{1.0182}\approx1.79\%.
$$

step 还需提速：

$$
1-\frac{1.35}{1.3688}\approx1.37\%.
$$

若目标是 full-grid S1，则 worst-shape 更严。当前 worst memory 是：

$$
r_{\text{mem,worst}}=1.0482.
$$

要 full-grid memory `<1.00`，需要降低：

$$
1-\frac{1.00}{1.0482}\approx4.60\%.
$$

当前 worst step 是：

$$
r_{\text{step,worst}}=1.4490.
$$

要 full-grid step `<=1.35`，需要提速：

$$
1-\frac{1.35}{1.4490}\approx6.83\%.
$$

所以 v7.6 的 efficiency 目标不是“再大改架构”，而是非常明确的 S1 margin repair。

### 1.3 M12 的 forward 仍偏慢

M12 的 full-step 过了 S2，但 forward ratio 仍然是：

$$
r_{\text{forward}}=2.0993.
$$

backward ratio 是：

$$
r_{\text{backward}}=1.1763.
$$

这说明 full-step S2 主要依赖 backward 和 update 路径已经控制住，但 forward 仍有明显 overhead。若终极目标是 forward/backward 都接近 MLP，v7.6 必须记录并优化 forward path，不能只看 step mean。

### 1.4 M12 依赖 C3 teacher distillation，需要公平性验证

M12 的训练使用：

```text
C3 teacher logit distill
T = 4
alpha = 0.25
M5 initialization trajectory
```

这本身不是问题；distillation 可以是训练 objective 的一部分。但如果要说“对 MLP-AdamW 有系统性优势”，必须做公平对照：

```text
MLP-AdamW baseline
MLP + same C3 distillation
MLP + self-distillation
M12 teacher-free
M12 self-distillation
M12 C3-distillation
```

如果 MLP 在相同 teacher / distillation protocol 下也得到类似提升，那么 M12 的优势一部分来自 training recipe，而不是 KAN structure。反之，如果 M12 比 MLP-distill 仍显著更好，则可以说 KAN structure 对 teacher signal 有更高可达性或更好 inductive bias。

### 1.5 还缺 time-to-target / ValLossAUC_time / geometry Pareto

目前 v7.5 的成功主要基于：

```text
final macro accuracy
CI / Holm p
test gap
ECE / NLL
full-grid S2 ratio
GradPass
```

但终极目标要求不仅 accuracy 更好，还要：

```text
ValLossAUC_time 不差于或优于 MLP-AdamW
time_to_target 更有竞争力
geometry 不只是口号，要有 Pareto 证据
sample efficiency / robustness / scaling 至少初步成立
```

如果 M12 每 step 比 MLP 慢 `1.3688x`，但只带来 `+2.08%` accuracy，必须验证它在 wall-clock 上是否仍值得。

---

## 2. v7.6 总体目标

v7.6 的目标是把 v7.5 的最低成功推进到系统级结论：

$$
\boxed{
\text{M12 是否真正优于 MLP-AdamW，而不是只刚好通过 v7.5 S2+macro gate？}
}
$$

v7.6 的整体目标分为五层：

```text
1. Reproducibility:
   10-seed / 20-seed confirm，确认 M12 的 +2% macro gap 稳定存在。

2. Fairness:
   对比 MLP-AdamW、MLP-distill、M12 teacher-free、M12 distill，确认优势不是 distillation-only。

3. Efficiency:
   从 FullGridS2 推到 FullGridS1，或至少明确 S1 差距来自哪个 live-set / kernel source。

4. Convergence:
   验证 M12 的 ValLossAUC_time、time_to_target、samples/sec 是否优于或至少不劣于 MLP-AdamW。

5. Mechanism:
   解释 M12 为什么成功：packed generic head、FP32 stable SiLU derivative、C3 teacher signal、A2S fused stack 各自贡献多少。
```

v7.6 的最低成功标准：

$$
\boxed{
\text{StrictPass}+\text{GradPass}+\text{MacroSignificantPass}_{10seed}+\text{FullGridS2Pass}+\text{FairnessPass}
}
$$

v7.6 的正式成功标准：

$$
\boxed{
\text{StrictPass}+\text{GradPass}+\text{MacroSignificantPass}_{10seed}+\text{FullGridS1Pass}+\text{TimeAUCPass}+\text{FairnessPass}
}
$$

v7.6 的强成功标准：

$$
\boxed{
\text{S1}+\text{TimeAUCPass}+\text{ECE/NLL improvement}+\text{MLP-distill superiority}+\text{Scaling/RBT pass}
}
$$

其中 RBT 表示 robustness / budget / transfer。

---

## 3. v7.6 禁止事项

第一，不允许回到 loss、sampler、class weight、focal、target margin、classwise-safe 主线。v7.6 的主问题不是刷榜。

第二，不允许把 v7.5 的 5-seed 成功直接写成终极成功。v7.6 必须做 10-seed 或 20-seed confirm。

第三，不允许只比较 MLP-AdamW raw baseline。由于 M12 使用 C3 distillation，必须加入 MLP-distill 和 M12 teacher-free 对照。

第四，不允许只看 mean ratio。efficiency 必须看 full-grid 9/9 shapes，S1/S2 都要报告 worst shape。

第五，不允许只看 final accuracy。必须记录 ValLossAUC_step、ValLossAUC_time、time_to_target_loss、time_to_target_acc。

第六，不允许在没有 kernel-count / live-set attribution 的情况下宣称 S1 repair 成功原因。必须解释 S1 margin 来自哪里。

第七，不允许为了修 GradPass 放宽阈值。v7.5 的正确做法是数值等价重排，而不是放宽 gate，v7.6 继续保持：

$$
\operatorname{grad\_relerr}_{max}\leq10^{-4}.
$$

第八，不允许把 geometry 当硬约束。geometry 只作为 Pareto 维度：如果 geometry 伤害 task 或 time-to-target，就不能作为路线目标。

第九，不允许把 test accuracy 用于调参。candidate selection 只用 train/val/holdout 与 pre-defined gates，test 只做最终报告。

第十，不允许把 M12 的 success 外推到更大数据/更多任务，除非 scaling / robustness / transfer 实验完成。

---

## 4. 核心假设

### H1：M12 的 macro accuracy advantage 对 MLP-AdamW 稳定显著

H1 假设：M12 相比 MLP-AdamW 的优势不是 5-seed 偶然，而是跨 seeds 稳定存在。

H1 成立标准：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

$$
CI_{95\%,macro}^{low}>0.
$$

$$
p_{\text{Holm}}<0.05.
$$

并且 test 同向：

$$
\Delta Acc_{\text{macro,test}}\geq0.015.
$$

此外，seed-wise win rate 必须满足：

$$
\frac{\#\{seed:Acc_{M12}>Acc_{MLP}\}}{\#seed}\geq0.70.
$$

如果 10-seed 后 gap 掉到 `+0.015` 以下，v7.5 success 仍然成立，但不能宣称 M12 对 MLP-AdamW 有稳定显著优势。

---

### H2：M12 的优势不是 distillation-only，而是 KAN structure + C3 reachability 的共同结果

H2 假设：C3 teacher distillation 对 A2S/M12 有帮助，但 M12 的优势并不只是 teacher 给的，KAN structure 对 teacher signal 有更好的吸收能力。

需要比较：

```text
MLP-AdamW
MLP + C3 logit distill
MLP + self-distill
M12 teacher-free
M12 + self-distill
M12 + C3 distill
A2S + C3 distill
C3 teacher
```

H2 成立标准：

M12 + C3 distill 相比 MLP + C3 distill 仍满足：

$$
\Delta Acc_{\text{macro,val}}\geq0.010.
$$

且 NLL 不差：

$$
NLL_{M12}\leq NLL_{MLPdistill}+0.01.
$$

若 MLP + C3 distill 追平 M12，则当前优势主要是 teacher / objective，而不是 architecture。此时 v7.6 route 应转为：

```text
distillation_objective_advantage_not_architecture
```

---

### H3：M12 的 S2 成功可以进一步推到 S1，且不破坏 macro

H3 假设：M12 目前离 S1 很近，剩余差距来自 head cache、packed owner、manual cache 或 allocator padding，而不是必须增加表达力 live-set。

H3 成立标准：

FullGridS1Pass：

$$
r_{\text{mem,max}}<1.00,
$$

$$
r_{\text{step,max}}\leq1.35.
$$

同时 macro 保持：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

允许的 task 损失：

$$
Acc_{\text{M12-S1}} \geq Acc_{\text{M12-S2}}-0.003.
$$

如果 S1 repair 导致 macro 掉到 `+0.017` 以下，则说明 S1 修复破坏了 effective expressivity，不能作为成功路线。

---

### H4：M12 的 wall-clock advantage 需要通过 AUC/time-to-target 证明

H4 假设：M12 虽然 step 比 MLP 慢，但由于 final accuracy、NLL 或 early loss dynamics 更好，可能在 time-to-target 或 ValLossAUC_time 上不劣于 MLP-AdamW。

定义：

$$
AUC_{time}^{val\_loss}
=
\sum_i
\frac{L_i+L_{i-1}}{2}
(t_i-t_{i-1}).
$$

H4 成立标准：

$$
AUC_{time,M12}^{val\_loss}\leq AUC_{time,MLP}^{val\_loss}.
$$

或者至少：

$$
time\_to\_target\_acc(M12)\leq1.10\cdot time\_to\_target\_acc(MLP).
$$

其中 target acc 定义为：

$$
Acc_{target}=Acc_{MLP,final}+0.005.
$$

若 M12 final accuracy 更高但 wall-clock AUC 明显更差，则 M12 是 accuracy/quality success，不是 system-level convergence success。

---

### H5：M12 成功的机制来自 packed generic head + stable SiLU derivative + A2S fused stack 的组合

H5 假设：M12 成功不是随机组合，而是三个机制同时成立：

```text
1. packed generic V63 head 减少参数 owner / head overhead；
2. stable FP32 SiLU derivative 修复 gradient gate；
3. A2S fused linear-SiLU stack 保持 low live-set；
4. C3 teacher distillation 保住 macro expression。
```

H5 成立标准：

消融后必须看到对应退化：

```text
M12 - packed head:
  step or memory worse, or GradPass unstable

M12 - stable SiLU derivative:
  GradPass fail or relerr approaches 1e-4

M12 - C3 distill:
  macro gap drops by >=0.005

M12 - fused stack:
  step/memory leaves S2
```

如果某个组件移除后结果不变，说明该组件不是必要机制，v7.6 应简化实现。

---

### H6：M12 的优势需要至少在一个扩展维度成立

H6 假设：如果 M12 是真正有意义的 PureKAN-NG，不应只在 1536 train / 3 small datasets / hidden64 下成立。它至少要在一个扩展维度继续有价值：

```text
sample efficiency
noise robustness
larger train size
hidden width scaling
patch/token small task
```

H6 成立标准之一：

Sample efficiency：

$$
AUC_{data,M12}>AUC_{data,MLP}.
$$

Noise robustness：

$$
AccDrop_{M12}<AccDrop_{MLP}
$$

for at least two noise settings.

Scaling：

$$
\Delta Acc_{\text{macro,val}}\geq0.010
$$

at a larger train size or hidden size, without leaving S2 by more than a controlled margin.

---

## 5. Candidate 设计

### 5.1 Baselines

```text
B0-MLP-AdamW-reference
B1-MLP-AdanLite-reference
B2-MLP-C3-logit-distill
B3-MLP-self-distill
B4-C3-teacher
B5-A2S-supervised
B6-A2S-C3-distill
B7-M12-final-v75
```

### 5.2 M12 confirmation candidates

```text
M12-FINAL:
  packed generic V63 head
  FP32 stable SiLU derivative
  C3 distill T4 alpha0.25
  M5 initialization trajectory

M12-NODISTILL:
  same architecture, no C3 teacher

M12-SELFDISTILL:
  same architecture, self-distill teacher

M12-RESEED:
  same architecture, independent seeds

M12-LONGBUDGET:
  same architecture, 480 steps

M12-NOSTABLESILU:
  old derivative form, diagnostic only

M12-NOPACKEDHEAD:
  unpacked generic head, diagnostic only
```

### 5.3 S1 repair candidates

```text
S1-0-M12-current
S1-1-head-cache-trim
S1-2-packed-owner-state-trim
S1-3-manual-cache-lifetime-trim
S1-4-bs512-head-temp-streaming
S1-5-forward-temp-fusion
S1-6-update-buffer-lifetime-trim
S1-7-allocator-padding-control
S1-8-M12-S1-combo
```

### 5.4 Mechanism ablation candidates

```text
A0-M12-full
A1-no-C3-distill
A2-no-packed-head
A3-no-stable-SiLU
A4-no-M5-init
A5-unfused-stack
A6-generic-head-only
A7-packed-head-only
A8-distill-only-on-MLP
```

### 5.5 Scaling / robustness candidates

```text
R0-MLP-AdamW
R1-M12-final
R2-M12-S1-combo, if exists
R3-M12-no-distill
R4-MLP-distill
```

---

## 6. 实验阶段总览

v7.6 分为十三个阶段：

```text
P0: v7.5 M12 reproduction and provenance lock
P1: 10-seed / 20-seed M12 significance confirmation
P2: MLP-AdamW / MLP-distill fairness comparison
P3: teacher dependency and optimization reachability audit
P4: M12 mechanism ablation
P5: S1 memory/step margin attribution
P6: S1 repair package
P7: ValLossAUC / time-to-target verification
P8: ECE / NLL / calibration refinement
P9: sample efficiency and robustness expansion
P10: scaling smoke
P11: geometry Pareto diagnostic
P12: candidate co-selection and route decision
P13: artifact and failure audit
```

P0-P4 验证 M12 成功是否真实且公平。  
P5-P6 尝试从 S2 推到 S1。  
P7-P10 判断系统级优势。  
P11 只做 geometry Pareto，不做硬目标。  
P12-P13 输出最终 route。

---

## 7. P0：v7.5 M12 reproduction and provenance lock

### 7.1 目的

确认 v7.6 与 v7.5 最终确认 run 可比。P0 不寻找新结果，只锁定 baseline、artifact、candidate lineage。

### 7.2 必跑对象

```text
B0-MLP-AdamW-reference
M12-FINAL
C3-teacher
A2S-reference
```

### 7.3 必须记录字段

```text
run_id
candidate
family
status
fake_data_used
proxy_row_used
uses_loss_backward
uses_torch_autograd_graph
manual_forward_available
manual_backward_available
manual_update_available
nonKAN_param_count
head_type
head_is_kan
teacher_used
teacher_candidate
distill_temperature
distill_alpha
grad_relerr_max
grad_cos_min
macro_val_gap
ci95_low
holm_p
test_gap
ECE_delta
NLL_delta
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
S2_shape_count
S1_shape_count
reproduction_delta_macro_gap
reproduction_delta_memory
reproduction_delta_step
```

### 7.4 判断标准

P0 pass：

```text
fake_data_used = 0
proxy_row_used = 0
nonKAN_param_count = 0
manual_backward_available = 1
uses_loss_backward = 0
grad_relerr_max <= 1e-4
grad_cos_min >= 0.999
```

M12 reproduction：

$$
|\Delta Acc_{\text{M12,v76}}-0.0208|\leq0.005.
$$

Efficiency reproduction：

$$
|r_{\text{mem,M12,v76}}-1.0182|\leq0.02.
$$

$$
|r_{\text{step,M12,v76}}-1.3688|\leq0.10.
$$

### 7.5 可视化

```text
p0_m12_reproduction_task_bar.svg
p0_m12_reproduction_efficiency_bar.svg
p0_contract_heatmap.svg
p0_candidate_lineage_table.md
```

---

## 8. P1：10-seed / 20-seed M12 significance confirmation

### 8.1 目的

验证 M12 对 MLP-AdamW 的 macro advantage 是否稳定，而不是 5-seed 下刚过线。

### 8.2 设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0..9
  optional 0..19 if 10-seed margin is close

steps:
  240

train/val/test:
  1536/512/512

bootstrap:
  10000 samples
```

### 8.3 必跑对象

```text
MLP-AdamW-reference
M12-FINAL
C3-teacher, reference only
```

### 8.4 必须记录字段

```text
candidate
dataset
seed
train_acc
val_acc
test_acc
train_loss
val_loss
test_loss
ECE
NLL
Brier
val_gap_vs_MLP
test_gap_vs_MLP
seedwise_win
ValLossAUC_step
ValLossAUC_time
time_to_target_acc
time_to_target_loss
```

统计字段：

```text
macro_val_gap_mean
macro_val_gap_std
macro_val_gap_ci95_low
macro_val_gap_ci95_high
paired_t_p
wilcoxon_p
holm_p
cohen_h
seed_win_rate
dataset_gap_mean
dataset_gap_ci95_low
```

### 8.5 判断标准

10-seed MacroSignificantPass：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

$$
CI_{95\%,macro}^{low}>0.
$$

$$
p_{\text{Holm}}<0.05.
$$

$$
\Delta Acc_{\text{macro,test}}\geq0.015.
$$

Seed stability：

$$
win\_rate_{seed}\geq0.70.
$$

If 10-seed gap is:

$$
0.018\leq\Delta Acc_{\text{macro,val}}<0.020,
$$

then run 20-seed confirm rather than declare failure.

### 8.6 可视化

```text
p1_seedwise_val_gap_boxplot.svg
p1_bootstrap_ci_macro.svg
p1_dataset_gap_ci.svg
p1_val_test_gap_scatter.svg
p1_seed_win_rate_bar.svg
```

---

## 9. P2：MLP-AdamW / MLP-distill fairness comparison

### 9.1 目的

回答用户最关心的问题：M12 相比 MLP-AdamW 是否有显著优势，以及这个优势是否公平。

### 9.2 必跑对象

```text
MLP-AdamW
MLP-AdanLite
MLP-C3-logit-distill-T4-alpha0.25
MLP-self-distill
M12-no-distill
M12-C3-distill
M12-self-distill
C3-teacher
```

### 9.3 必须记录字段

```text
candidate
optimizer
teacher_used
teacher_type
distill_temperature
distill_alpha
train_acc
val_acc
test_acc
val_loss
test_loss
ECE
NLL
Brier
ValLossAUC_step
ValLossAUC_time
time_to_target_acc
time_to_target_loss
memory_ratio_mean
step_ratio_mean
forward_ratio_mean
backward_ratio_mean
```

### 9.4 判断标准

M12 vs MLP-AdamW pass：

$$
Acc_{M12} - Acc_{MLP-AdamW}\geq0.0200.
$$

M12 vs MLP-distill fairness pass：

$$
Acc_{M12-distill} - Acc_{MLP-distill}\geq0.0100.
$$

NLL fairness：

$$
NLL_{M12-distill}\leq NLL_{MLP-distill}+0.01.
$$

Efficiency fairness：

$$
r_{\text{step,M12}}\leq1.50.
$$

$$
r_{\text{mem,M12}}\leq1.05.
$$

If MLP-distill closes the entire gap, route becomes:

```text
teacher_objective_explains_advantage
```

If M12 remains ahead of MLP-distill, route becomes:

```text
KAN_structure_distillation_advantage
```

### 9.5 可视化

```text
p2_fairness_val_gap_bar.svg
p2_distill_effect_by_model.svg
p2_nll_ece_fairness_scatter.svg
p2_task_efficiency_fairness_pareto.svg
p2_mlp_vs_m12_learning_curves.svg
```

---

## 10. P3：teacher dependency and optimization reachability audit

### 10.1 目的

判断 M12 是否必须依赖 C3 teacher，还是 M12 structure 本身已经有足够 reachability。

### 10.2 Candidates

```text
M12-supervised-only
M12-C3-logit-distill
M12-C3-feature-distill
M12-C3-logit+feature
M12-self-distill
M12-distill-then-supervised-finetune
A2S-supervised
A2S-C3-distill
```

### 10.3 必须记录字段

```text
candidate
teacher_candidate
teacher_used
distill_loss
supervised_loss
feature_loss
teacher_student_KL
teacher_student_logit_cos
feature_CKA
val_acc
test_acc
val_gap_vs_MLP
val_gap_vs_C3
ECE
NLL
distill_improvement_vs_supervised
memory_ratio_inference
step_ratio_inference
memory_ratio_training
step_ratio_training
```

### 10.4 判断标准

Teacher dependency strong if：

$$
Acc_{M12-distill}-Acc_{M12-supervised}\geq0.005.
$$

Teacher dependency mild if：

$$
0.001\leq Acc_{M12-distill}-Acc_{M12-supervised}<0.005.
$$

Teacher-independent pass if：

$$
Acc_{M12-supervised}-Acc_{MLP}\geq0.015.
$$

If teacher dependency is strong, v7.6 should treat distillation as part of the official training recipe and compare fairly to MLP-distill.

### 10.5 可视化

```text
p3_teacher_dependency_bar.svg
p3_teacher_student_KL_curve.svg
p3_feature_CKA_heatmap.svg
p3_distill_vs_supervised_efficiency.svg
```

---

## 11. P4：M12 mechanism ablation

### 11.1 目的

解释 M12 为什么成功，避免把成功归因成黑箱。

### 11.2 Candidates

```text
A0-M12-full
A1-M12-no-C3-distill
A2-M12-no-packed-head
A3-M12-old-SiLU-derivative
A4-M12-FP64-SiLU-derivative
A5-M12-no-M5-init
A6-M12-unfused-stack
A7-M12-generic-head-only
A8-M12-packed-head-only
```

### 11.3 必须记录字段

```text
candidate
ablation_type
grad_relerr_max
grad_cos_min
macro_val_gap
test_gap
ECE_delta
NLL_delta
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
S2_shape_count
head_forward_ms
head_backward_ms
stack_forward_ms
stack_backward_ms
parameter_owner_count
materialized_tensor_count
```

### 11.4 判断标准

A component is essential if removing it causes any of:

$$
\Delta Acc_{\text{macro,val}}\leq-0.005.
$$

or:

$$
r_{\text{step}}>1.50.
$$

or:

$$
r_{\text{mem}}>1.05.
$$

or GradPass fail:

$$
\operatorname{grad\_relerr}_{max}>10^{-4}.
$$

Stable SiLU derivative is essential if old derivative fails GradPass while stable derivative passes under identical task/efficiency setting.

### 11.5 可视化

```text
p4_mechanism_ablation_task_bar.svg
p4_mechanism_ablation_efficiency_bar.svg
p4_grad_relerr_by_silu_form.svg
p4_parameter_owner_count_vs_step.svg
p4_m12_mechanism_dashboard.svg
```

---

## 12. P5：S1 memory/step margin attribution

### 12.1 目的

M12 已经 S2，但 S1 尚未闭合。P5 专门解释 S1 还差在哪里。

### 12.2 Shape grid

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

batch sizes:
  128
  256
  512

depths:
  native
```

### 12.3 必须记录字段

```text
candidate
dataset
batch_size
peak_allocated_MB
peak_reserved_MB
memory_ratio
step_ratio
forward_ratio
backward_ratio
update_ratio
head_temp_MB
stack_temp_MB
manual_cache_MB
optimizer_state_MB
packed_owner_state_MB
allocator_padding_MB
reserved_unallocated_MB
largest_live_tensor_MB
materialized_tensor_count
kernel_count_total
torch_op_count
triton_kernel_count
top1_memory_source
top2_memory_source
top3_memory_source
top1_time_source
top2_time_source
top3_time_source
unknown_memory_fraction
unknown_time_fraction
```

### 12.4 判断标准

Attribution pass：

```text
top3 memory sources identified
unknown_memory_fraction <= 0.10
top3 time sources identified
unknown_time_fraction <= 0.10
kernel_count_total measured
materialized_tensor_count measured
```

S1 source is actionable if one source satisfies：

$$
\frac{M_{source}}{M_{peak}}\geq0.02.
$$

or:

$$
\frac{T_{source}}{T_{step}}\geq0.03.
$$

因为 M12 离 mean S1 只差约 `1-2%`，所以 even small source is actionable。

### 12.5 可视化

```text
p5_s1_memory_waterfall.svg
p5_s1_step_time_waterfall.svg
p5_shape_memory_heatmap.svg
p5_shape_step_heatmap.svg
p5_s1_gap_by_shape.svg
```

---

## 13. P6：S1 repair package

### 13.1 目的

基于 P5 attribution，把 M12 从 S2 推到 S1。不得牺牲 macro signal。

### 13.2 Candidates

```text
S1-0-M12-current
S1-1-head-cache-trim
S1-2-packed-owner-state-trim
S1-3-manual-cache-lifetime-trim
S1-4-bs512-head-temp-streaming
S1-5-forward-temp-fusion
S1-6-update-buffer-lifetime-trim
S1-7-allocator-padding-control
S1-8-M12-S1-combo
```

### 13.3 必须记录字段

```text
candidate
components
implementation_status
grad_relerr_max
grad_cos_min
macro_val_gap
test_gap
ECE_delta
NLL_delta
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
S1_shape_count
S2_shape_count
memory_improvement_vs_M12
step_improvement_vs_M12
macro_delta_vs_M12
```

### 13.4 判断标准

S1 repair useful：

$$
r_{\text{mem,max,new}}<r_{\text{mem,max,M12}}-0.02.
$$

or:

$$
r_{\text{step,max,new}}<r_{\text{step,max,M12}}-0.05.
$$

FullGridS1Pass：

$$
r_{\text{mem,max}}<1.00.
$$

$$
r_{\text{step,max}}\leq1.35.
$$

Task preservation：

$$
\Delta Acc_{\text{macro,val,new}}\geq\Delta Acc_{\text{macro,val,M12}}-0.003.
$$

If S1 repair reaches S1 but macro falls below `+0.0200`, it is an efficiency diagnostic, not final success.

### 13.5 可视化

```text
p6_s1_repair_pareto.svg
p6_s1_shape_pass_heatmap.svg
p6_task_preservation_bar.svg
p6_s1_boundary_plot.svg
```

---

## 14. P7：ValLossAUC / time-to-target verification

### 14.1 目的

验证 M12 的系统价值。如果 per-step 慢于 MLP，就必须证明收敛质量或 AUC 可以补偿。

### 14.2 必跑对象

```text
MLP-AdamW
MLP-distill
M12-FINAL
M12-S1-combo, if exists
C3-teacher
```

### 14.3 设置

```text
steps:
  240
  optional 480

logging:
  every 10 steps

seeds:
  0..9
```

### 14.4 必须记录字段

```text
candidate
dataset
seed
wall_clock_time_sec
step
val_loss
val_acc
train_loss
train_acc
ValLossAUC_step
ValLossAUC_time
ValAccAUC_step
ValAccAUC_time
time_to_target_loss
time_to_target_acc
samples_per_second
early_loss_slope
mid_loss_slope
late_loss_slope
```

### 14.5 判断标准

TimeAUCPass：

$$
ValLossAUC_{time,M12}\leq ValLossAUC_{time,MLP}.
$$

Target pass：

$$
time\_to\_target\_acc(M12)\leq1.10\cdot time\_to\_target\_acc(MLP).
$$

Strong convergence pass：

$$
time\_to\_target\_acc(M12)\leq time\_to\_target\_acc(MLP).
$$

If M12 wins final accuracy but loses all time-AUC metrics badly, route becomes：

```text
quality_advantage_but_wallclock_not_yet
```

### 14.6 可视化

```text
p7_val_loss_vs_time.svg
p7_val_acc_vs_time.svg
p7_val_loss_auc_time_bar.svg
p7_time_to_target_bar.svg
p7_step_ratio_vs_step_saving_scatter.svg
```

---

## 15. P8：ECE / NLL / calibration refinement

### 15.1 目的

M12 的 NLL 明显优于 MLP，但 ECE delta 为 `+0.0024`，只是 within gate，不是更好。P8 判断能否在不伤 task/efficiency 的情况下使 calibration 也优于 MLP。

### 15.2 Candidates

```text
CAL0-M12-current
CAL1-M12-temperature-posthoc
CAL2-M12-logit-scale-edge-owned
CAL3-M12-weak-geometry-lambda1e-5
CAL4-M12-weak-geometry-lambda1e-4
CAL5-M12-distill-temperature-adjusted
```

### 15.3 必须记录字段

```text
candidate
calibration_method
posthoc_or_train_time
val_acc
test_acc
ECE
AdaptiveECE
NLL
Brier
macro_gap
memory_ratio
step_ratio
temperature_value
logit_norm_mean
confidence_mean
wrong_confidence_mean
```

### 15.4 判断标准

Calibration improvement：

$$
ECE_{new}<ECE_{M12}.
$$

and:

$$
Acc_{new}\geq Acc_{M12}-0.002.
$$

and:

$$
r_{\text{step,new}}\leq1.50.
$$

Strong calibration pass：

$$
ECE_{new}<ECE_{MLP}.
$$

If calibration only improves by posthoc temperature, it can be reported separately but should not be mixed with training-time architecture claim.

### 15.5 可视化

```text
p8_reliability_diagram.svg
p8_ece_nll_bar.svg
p8_confidence_histogram_correct_wrong.svg
p8_accuracy_calibration_pareto.svg
```

---

## 16. P9：sample efficiency and robustness expansion

### 16.1 目的

判断 M12 的优势是否代表函数空间结构优势，而不只是当前 fixed-size dataset 下的 result。

### 16.2 Data fractions

```text
train_size:
  256
  512
  1024
  1536
  4096, if available
```

### 16.3 Noise settings

```text
label_noise:
  5%
  10%
  20%

input_noise:
  gaussian sigma 0.05
  gaussian sigma 0.10

corruption:
  random erasing small
  mild affine shift
```

### 16.4 必跑对象

```text
MLP-AdamW
MLP-distill
M12-FINAL
M12-S1-combo, if exists
```

### 16.5 必须记录字段

```text
candidate
dataset
seed
train_size
noise_type
noise_level
val_acc
test_acc
val_loss
test_loss
ECE
NLL
accuracy_drop_vs_clean
sample_efficiency_auc
robustness_auc
memory_ratio
step_ratio
time_to_target_acc
```

### 16.6 判断标准

Sample efficiency pass：

$$
AUC_{data,M12}>AUC_{data,MLP}.
$$

Robustness pass：

$$
AccDrop_{M12}<AccDrop_{MLP}
$$

for at least two noise settings.

If M12 only wins clean fixed-size task but loses robustness/sample efficiency, route remains clean-task success, not broad advantage.

### 16.7 可视化

```text
p9_accuracy_vs_train_size.svg
p9_sample_efficiency_auc_bar.svg
p9_accuracy_under_noise.svg
p9_robustness_auc_bar.svg
p9_ece_under_noise.svg
```

---

## 17. P10：scaling smoke

### 17.1 目的

验证 M12 在 width / batch / task size 上是否有合理 scaling。

### 17.2 Settings

```text
hidden_dim:
  48
  64
  96
  128

batch:
  128
  256
  512

optional tasks:
  CIFAR10-small
  EMNIST-subset
  Fashion/KMNIST larger train split
```

### 17.3 必须记录字段

```text
candidate
hidden_dim
batch_size
dataset
train_size
val_acc
test_acc
macro_gap_vs_MLP
memory_ratio
step_ratio
forward_ratio
backward_ratio
S2_pass
S1_pass
samples_per_second
parameter_count
kernel_count_total
```

### 17.4 判断标准

Scaling pass：

$$
\Delta Acc_{\text{macro,val}}\geq0.010.
$$

and:

$$
r_{\text{mem}}\leq1.10.
$$

and:

$$
r_{\text{step}}\leq1.70.
$$

for first smoke. Full pass requires S2.

### 17.5 可视化

```text
p10_hidden_scaling_accuracy.svg
p10_hidden_scaling_efficiency.svg
p10_batch_scaling_heatmap.svg
p10_task_scaling_pareto.svg
```

---

## 18. P11：geometry Pareto diagnostic

### 18.1 目的

Geometry 不是 hard gate，但如果要 claim functional KAN advantage，需要证明至少一个 geometry dimension 有 Pareto benefit。

### 18.2 Geometry losses

```text
basis_smoothness
edge_curvature
jacobian_norm
local_lipschitz
coefficient_tv
path_length
```

### 18.3 Lambda sweep

```text
lambda_geo:
  0
  1e-6
  3e-6
  1e-5
  3e-5
  1e-4
```

### 18.4 必须记录字段

```text
candidate
lambda_geo
geometry_loss_type
val_acc
test_acc
ECE
NLL
ValLossAUC_step
ValLossAUC_time
geometry_metric_value
basis_smoothness
edge_curvature
jacobian_norm
local_lipschitz
memory_ratio
step_ratio
```

### 18.5 判断标准

Geometry Pareto useful if：

$$
Acc_{\lambda}\geq Acc_{\lambda=0}-0.005.
$$

and one of：

$$
ECE_{\lambda}<ECE_{\lambda=0},
$$

$$
NLL_{\lambda}<NLL_{\lambda=0},
$$

$$
ValLossAUC_{\lambda}<ValLossAUC_{\lambda=0}.
$$

Harmful if：

$$
Acc_{\lambda}<Acc_{\lambda=0}-0.02
$$

and no metric improves.

### 18.6 可视化

```text
p11_geometry_pareto_frontier.svg
p11_accuracy_vs_geometry.svg
p11_ece_nll_vs_geometry.svg
p11_val_loss_auc_vs_geometry.svg
```

---

## 19. P12：candidate co-selection and route decision

### 19.1 Survivor 类型

```text
S0:
  StrictPass + GradPass + FullGridS1Pass + 10seed MacroSignificantPass + TimeAUCPass + FairnessPass

S1:
  StrictPass + GradPass + FullGridS2Pass + 10seed MacroSignificantPass + FairnessPass

S2:
  FullGridS2 + MacroSignificant, but fairness vs MLP-distill unclear

S3:
  MacroSignificant but no TimeAUCPass

S4:
  TimeAUCPass but macro gap unstable

S5:
  MLP-distill closes the gap; teacher objective explains advantage

S6:
  S1 repair hurts expression

S7:
  Scaling/robustness fail

S8:
  Gradient fail

S9:
  no improvement
```

### 19.2 必须记录字段

```text
candidate
survivor_type
strict_pass
grad_pass
macro_10seed_pass
fullgrid_s2_pass
fullgrid_s1_pass
time_auc_pass
fairness_pass
calibration_pass
sample_efficiency_pass
robustness_pass
geometry_pareto_pass
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
val_gap_vs_MLP
test_gap_vs_MLP
val_gap_vs_MLP_distill
CI95_low
Holm_p
ECE_delta
NLL_delta
ValLossAUC_time_delta
route_recommendation
primary_blocker
next_required_implementation
```

### 19.3 Route cases

```text
R1-FullSystemAdvantage:
  S1 + 10seed MacroSignificant + TimeAUCPass + FairnessPass.
  Claim system-level PureKAN-NG advantage.

R2-S2QualityAdvantage:
  S2 + MacroSignificant + FairnessPass, but S1/time AUC incomplete.
  Claim efficient diagnostic success, continue S1/time work.

R3-DistillationObjectiveAdvantage:
  MLP-distill catches up.
  Claim teacher/objective explains much of gain; need architecture-independent comparison.

R4-AccuracyOnlyAdvantage:
  MacroSignificant but time AUC fails.
  Claim quality improvement but not wall-clock system advantage.

R5-S1RepairHurtsExpression:
  Efficiency repair breaks macro.
  Return to dense-preserving S1 design.

R6-ScalingFail:
  Clean small-data success only.
  Need stronger primitive or scaling redesign.

R7-NoReproduction:
  10seed/20seed does not reproduce v7.5.
  Treat v7.5 as promising but unstable.
```

### 19.4 可视化

```text
p12_system_pareto_accuracy_efficiency_time.svg
p12_survivor_type_dashboard.svg
p12_route_decision_tree.svg
p12_final_scorecard.svg
```

---

## 20. P13：artifact and failure audit

### 20.1 Required artifacts

```text
run_manifest.json
provenance_audit.csv
candidate_registry.csv
gate_config.json
p0_reproduction.csv
p1_10seed_significance.csv
p2_fairness_mlp_distill.csv
p3_teacher_dependency.csv
p4_m12_mechanism_ablation.csv
p5_s1_attribution.csv
p6_s1_repair.csv
p7_val_loss_auc_time.csv
p8_calibration_refinement.csv
p9_sample_robustness.csv
p10_scaling_smoke.csv
p11_geometry_pareto.csv
p12_candidate_selection.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

### 20.2 Failure taxonomy

```text
F1_10seed_macro_fail
F2_mlp_distill_closes_gap
F3_teacher_free_fail
F4_time_auc_fail
F5_s1_memory_fail
F6_s1_step_fail
F7_s1_repair_hurts_expression
F8_forward_ratio_fail
F9_calibration_fail
F10_scaling_fail
F11_robustness_fail
F12_geometry_hurts_expression
F13_gradient_correctness_fail
F14_fake_or_proxy_violation
F15_artifact_missing
```

### 20.3 Required figures

```text
figures/p12_system_pareto_accuracy_efficiency_time.svg
figures/p12_final_scorecard.svg
figures/p12_route_decision_tree.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 21. 第一轮推荐执行顺序

v7.6 第一轮不要全量展开。最重要的是先确认“v7.5 最低成功是否能升级成稳定系统结论”。

### Step 1：P1 10-seed confirm

先确认 M12 的 macro gap 是否稳定超过 `+0.0200`。如果 10-seed 不稳，不要做 S1 和 scaling。

### Step 2：P2 fairness vs MLP-distill

如果 MLP-distill 追平 M12，那么下一步应研究 distillation objective，而不是宣称 KAN architecture 全面优势。

### Step 3：P7 time-to-target / AUC

如果 M12 final accuracy win 但 wall-clock AUC 很差，系统 claim 必须降级。

### Step 4：P5-P6 S1 repair

只有在 M12 10-seed 和 fairness pass 后，S1 repair 才值得投入。否则 S1 repair 可能只是优化一个不稳的候选。

### Step 5：P4 mechanism ablation

如果 P1/P2 通过，做 mechanism ablation，把 M12 成功原因写清楚。

---

## 22. 停止条件

### 22.1 成功停止

出现以下任一情况立即写复盘：

```text
10seed MacroSignificantPass + FullGridS2Pass + FairnessPass
10seed MacroSignificantPass + FullGridS1Pass + FairnessPass
10seed MacroSignificantPass + FullGridS1Pass + FairnessPass + TimeAUCPass
```

### 22.2 失败停止

出现以下情况停止对应路线：

```text
1. 10seed macro gap < +0.015；
2. MLP-distill 与 M12 gap < +0.005；
3. M12 teacher-free 与 M12-distill 差距 > +0.010，且 MLP-distill 追平；
4. S1 repair 让 macro gap 掉到 < +0.018；
5. time AUC 比 MLP 差 >10%；
6. GradPass 任一 route candidate fail 且无法用等价数值重排修复；
7. no-fake/no-proxy audit fail。
```

---

## 23. 成功与失败解释规则

### Case A：M12 10-seed 稳定，MLP-distill 追不上

这是最好的机制结果。说明 M12 不是单纯 teacher objective 的产物，而是 KAN architecture + distillation reachability 的优势。

### Case B：M12 10-seed 稳定，但 MLP-distill 追平

说明 v7.5 的优势很大部分来自 C3 teacher / distillation objective。M12 仍是一个 efficient strict KAN implementation success，但不能声称 architecture 独有优势。

### Case C：M12 10-seed 不稳定

说明 v7.5 是一个真实但边际很薄的阶段性成功。下一步应回到 effective expressivity margin，不应直接做 S1。

### Case D：S1 repair 成功且 macro 保持

这是真正系统突破：从 S2 diagnostic 进入 S1 official candidate。

### Case E：S1 repair 破坏 macro

说明 M12 的剩余 live-set 不是纯冗余，可能承载 effective expressivity 或数值稳定性。需要 dense-preserving S1 设计。

### Case F：TimeAUCPass 失败

说明 M12 是 quality win，不是 wall-clock win。系统 claim 必须降级为 accuracy/NLL advantage。

### Case G：Scaling/robustness fail

说明 M12 可能是当前 small-task/small-data 的结构成功，还不能外推到更广任务。

---

## 24. 最终建议

v7.6 的一句话策略是：

$$
\boxed{
\text{把 M12 从 v7.5 最低成功，升级为可复现、公平、wall-clock 有意义、可解释的 MLP-AdamW 系统优势。}
}
$$

当前最准确的状态是：

```text
1. v7.5 达成最低成功标准；
2. M12 对 B0/MLP-AdamW reference 有 macro significant accuracy advantage；
3. M12 full-grid S2 成立，9/9 shapes pass；
4. M12 还不是 S1；
5. M12 的 ECE 只是 within gate，不是更优；
6. M12 使用 C3 distillation，必须做 MLP-distill fairness；
7. M12 还缺 time-to-target / ValLossAUC_time；
8. M12 的 generalization/scaling 还没证明。
```

因此，下一步不是小修小补，而是系统确认：

$$
\boxed{
\text{reproducibility}
+
\text{fairness}
+
\text{S1 margin}
+
\text{time-to-target}
+
\text{mechanism}
+
\text{scaling}
}
$$

只有这些闭合，才能从：

```text
M12 is the first S2-MacroSignificant PureKAN-NG candidate
```

升级为：

```text
M12 establishes a robust system-level advantage over MLP-AdamW.
```
