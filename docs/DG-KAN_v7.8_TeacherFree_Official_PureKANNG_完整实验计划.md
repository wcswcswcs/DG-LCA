# DG-KAN v7.8 Teacher-Free Official：去 Teacher 依赖、重建 M12/M13 自主优势与系统闭环完整实验计划

> 本计划基于 v7.5-v7.7 的真实结果重新修订。核心纠偏是：**C3 teacher distillation 不能再作为 official success 的主线条件**。  
> 如果项目目标是证明 PureKAN-NG 本身相对 MLP-AdamW 有结构优势，那么最终 candidate 必须在 **无外部 teacher** 条件下成立。C3 teacher / MLP-distill 只能作为诊断、上界、fairness control 或 reachability probe，不能作为最终成功定义。  
> 因此，v7.8 的 official 主线改为：**Teacher-Free M12/M13 是否能达到 MacroSignificantPass + FullGridS2/S1 + TimeAUCPass**。  
> C3 teacher 只保留为 diagnostic branch，用来回答“如果 teacher 能帮、为什么帮、帮了多少”，而不是证明最终模型必须依赖 teacher。

---

## 0. 为什么要纠偏：teacher 不能成为最终拐杖

### 0.1 当前疑问是合理的

如果我们说：

```text
M12 = PureKAN-NG + C3 teacher distillation
```

并把它当成最终成功，那么确实有跑偏风险。因为最终 claim 会变成：

```text
一个被更强 KAN teacher 教过的 KAN student 超过了 raw MLP-AdamW。
```

这不是最干净的架构结论。更干净的结论应该是：

```text
PureKAN-NG 在相同监督信息下，不依赖外部 teacher，也能超过 MLP-AdamW。
```

因此从 v7.8 开始，official route 必须区分：

```text
Official:
  teacher-free PureKAN-NG

Diagnostic:
  C3 teacher distillation
  MLP-distill fairness
  offline teacher timing
  reachability upper bound
```

---

### 0.2 teacher 曾经为什么被引入

teacher 不是为了刷榜，而是为了解决一个阶段性问题：A2S-family 在 v7.4/v7.5 已经低 live-set、接近 S2，但 macro gap 卡在 `+0.0199` 左右。C3 是表达力 teacher，distillation 用来测试：

```text
A2S/M12 结构是否有能力吸收 C3 的函数信号？
如果能，说明低 live-set student 不是完全表达力不足；
如果不能，说明结构约束太强。
```

v7.6 结果显示，teacher 对 M12 的增益是 **mild** 而不是主因：

```text
M13 teacher-free val gap vs MLP = +0.01784
M12 - M13 = +0.00273
teacher_dependency_mild = 1
teacher_independent_pass = 1
```

这说明 M12 family 本身已经有独立信号，C3 teacher 只是把它从 `+0.01784` 推到 `+0.02057`。这个结果非常重要：teacher 不是全部优势来源，但它目前确实是跨过 `+0.0200` hard gate 的最后一脚。

因此 v7.8 的核心问题应改为：

$$
\boxed{
\text{如何让 teacher-free M12/M13 自己跨过 } +0.0200 \text{ macro gate？}
}
$$

而不是继续把 C3 teacher 当 official recipe。

---

### 0.3 v7.7 当前状态重判

v7.7 clean run 说明：

```text
M12 + C3 distill:
  MacroSignificantPass = true
  FairnessPass vs MLP-distill = true
  GradPass = true
  FullGridS2Pass = true

但:
  FullGridS1Pass = false
  TimeAUCPass = false
  TimeAccountingPass = false
  ProfilerPass = false
```

M12 的质量优势仍成立：

$$
\Delta Acc_{\text{macro,val}}=+0.02057,
$$

$$
CI_{95\%,macro}^{low}=+0.01491,
$$

$$
p_{\text{Holm}}=1.35\times10^{-6},
$$

$$
\Delta Acc_{\text{macro,test}}=+0.01966.
$$

M12 相对 MLP-distill 仍有：

$$
\Delta Acc_{\text{M12-B2}}=+0.01517,
$$

$$
\Delta NLL_{\text{M12-B2}}=-0.02413.
$$

但是 v7.8 不能把这个作为 official final，因为 M12 用了外部 C3 teacher。v7.8 official target 应改为 teacher-free candidate。

---

## 1. v7.8 新目标

v7.8 的总目标是：

$$
\boxed{
\text{证明或证伪 PureKAN-NG 是否能在无外部 teacher 条件下系统性超过 MLP-AdamW。}
}
$$

v7.8 分成三条路线，但只有第一条是 official route。

### Route A：Official Teacher-Free Route

目标 candidate：

```text
M13 / M12-no-distill / M12-teacher-free-derived
```

必须满足：

$$
\text{StrictPass}
+
\text{GradPass}
+
\text{TeacherFreeMacroSignificantPass}
+
\text{FullGridS2Pass}
$$

正式目标还要求：

$$
\text{FullGridS1Pass}
+
\text{TimeAUCPass}.
$$

### Route B：Self-Distill / No External Teacher Route

允许使用同架构自身 checkpoint / EMA / previous epoch as teacher，但不允许 C3 外部 teacher。它不是最纯 official，但比 C3 teacher 更接近自主训练。

允许形式：

```text
M12 self-distill
M12 EMA teacher
M12 delayed self-teacher
M12 snapshot distill
```

这些必须明确标记为：

```text
no_external_teacher = true
```

### Route C：C3 Teacher Diagnostic Route

C3 teacher 只保留用于：

```text
1. 上界：teacher 可以把 student 推到哪里；
2. 差分分析：teacher-free 缺什么；
3. MLP-distill fairness：确保 teacher 本身不是唯一解释；
4. offline teacher timing diagnostic：解释 TimeAUC 中 teacher cost；
5. reachability probe：判断结构表达力是否足够。
```

C3 teacher route 不允许成为 final official success。

---

## 2. 成功标准

### 2.1 Teacher-Free Minimum Success

v7.8 teacher-free 最低成功：

$$
\boxed{
\text{StrictPass}
\land
\text{GradPass}
\land
\text{TeacherFreeMacroSignificantPass}
\land
\text{FullGridS2Pass}
}
$$

其中：

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

并且：

```text
external_teacher_used = false
teacher_logits_used = false
teacher_forward_used = false
```

### 2.2 Teacher-Free Formal Success

v7.8 teacher-free 正式成功：

$$
\boxed{
\text{TeacherFreeMinimumSuccess}
\land
\text{FullGridS1Pass}
\land
\text{TimeAUCPass}
}
$$

FullGridS1Pass：

$$
r_{\text{mem,max}}<1.00,
$$

$$
r_{\text{step,max}}\leq1.35.
$$

TimeAUCPass：

$$
ValLossAUC_{\text{time,KAN}}\leq ValLossAUC_{\text{time,MLP}}.
$$

### 2.3 Teacher-Free Strong Success

v7.8 强成功：

$$
\boxed{
\text{TeacherFreeFormalSuccess}
+
\text{ECE/NLL improvement}
+
\text{ProfilerPass}
+
\text{ScalingOrRobustnessPass}
}
$$

要求：

$$
ECE_{\text{KAN}}\leq ECE_{\text{MLP}},
$$

$$
NLL_{\text{KAN}}\leq NLL_{\text{MLP}},
$$

以及至少一个扩展维度成立：

```text
sample efficiency
noise robustness
hidden/batch scaling
larger train split
small patch/token task
```

### 2.4 Diagnostic Teacher Route 的成功不能等价 official success

如果只有：

```text
M12 + C3 teacher pass
```

而：

```text
M12 teacher-free fail
```

则 route 必须写成：

```text
TeacherAssistedSuccessOnly
```

不能写成：

```text
PureKAN-NG official success
```

---

## 3. 当前离 teacher-free 目标还差多远

v7.6 记录的 teacher-free `M13`：

```text
M13 val gap vs MLP = +0.01784
M12 - M13 = +0.00273
teacher_dependency_mild = 1
teacher_independent_pass = 1
```

v7.8 teacher-free macro gate 是：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

所以 M13 距离 gate 还差：

$$
0.0200-0.01784=0.00216.
$$

这个差距不大，但不能四舍五入成功。它说明下一步应做：

```text
teacher-free objective/initialization/architecture-preserving tuning
```

而不是继续依赖 C3 teacher。

当前真正问题是：

$$
\boxed{
\text{M12 family 已经几乎 teacher-free 过线，但还缺约 }0.2\%\text{ macro accuracy。}
}
$$

这比 “必须 teacher 才能学会” 要乐观得多。

---

## 4. 禁止事项

第一，禁止把 C3 teacher-distilled M12 作为 official final success。它只能是 diagnostic / assisted candidate。

第二，禁止把 MLP-distill 当主线目标。MLP-distill 只作为 fairness control，用来验证 teacher objective 本身的影响。

第三，禁止用 teacher 解决 teacher-free 未过线的问题。只要候选使用外部 C3 logits，就不能进入 official route。

第四，禁止回到 loss/sampler/class weight/focal/target-margin/classwise-safe 主线。当前缺口不是刷榜式 local repair。

第五，禁止普通 optimizer 大扫。只有围绕 teacher-free reachability 的小规模机制实验允许执行。

第六，禁止为了 teacher-free macro gap 破坏 GradPass / FullGridS2。任何 teacher-free repair 必须重新过：

$$
grad\_relerr_{max}\leq10^{-4}.
$$

第七，禁止只看 final val acc。必须同时记录：

```text
ValLossAUC_step
ValLossAUC_time
ECE
NLL
time_to_target
memory/step full-grid
profiler mapping
```

第八，禁止在 TimeAccountingPass 和 ProfilerPass 不成立时宣称 wall-clock system advantage。

第九，禁止用 test acc 调参。test 只用于最终报告。

第十，禁止把 self-distill 混同于 external teacher。self-distill 必须单独标记：

```text
external_teacher_used = false
self_teacher_used = true
```

---

## 5. 核心假设

### H1：M12 family 的主要优势不依赖外部 teacher

H1 基于 v7.6 的事实：M13 teacher-free 已经达到 `+0.01784`，C3 teacher 只带来 `+0.00273` 增益。

H1 假设：

$$
\text{M12 teacher-free family}
$$

已经拥有主要结构优势，只需小幅 teacher-free 训练/初始化/数值稳定修复即可过 `+0.0200`。

H1 成立标准：

$$
\Delta Acc_{\text{macro,val,M12-TF}}\geq0.0200.
$$

并且：

$$
\Delta Acc_{\text{macro,test,M12-TF}}\geq0.015.
$$

H1 强成立标准：

$$
\Delta Acc_{\text{M12-TF}} \geq \Delta Acc_{\text{M12-C3}} - 0.003.
$$

也就是 teacher-free 不比 C3-distilled M12 低超过 `0.003`。

---

### H2：C3 teacher 增益主要来自 early optimization / calibration，而不是不可替代的函数信息

如果 C3 teacher 只是帮 M12 早期收敛或校准，那么我们应该能用 teacher-free 方法复现类似效果。

候选 teacher-free 方法：

```text
M5 initialization trajectory stabilization
EMA self-teacher
delayed self-distill
confidence sharpening
NLL-balanced CE
warmup schedule
edge update normalization
calibration-aware CE
longer budget 480-step diagnostic
```

H2 成立标准：

任一 no-external-teacher candidate 达到：

$$
Acc_{\text{TF-repair}}-Acc_{\text{M13}}\geq0.002.
$$

且：

$$
\Delta Acc_{\text{macro,val,TF-repair}}\geq0.0200.
$$

如果所有 teacher-free repair 增益都低于：

$$
0.001,
$$

而 C3 teacher 稳定提供 `+0.002-0.003`，则 C3 teacher 携带了 teacher-free 当前无法获得的信息，official claim 需要降级。

---

### H3：teacher-free macro gap 不能靠牺牲系统效率换取

H3 要求 teacher-free repair 必须保留 M12 系统路径。

H3 成立标准：

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50,
$$

且 GradPass 成立：

$$
grad\_relerr_{max}\leq10^{-4}.
$$

如果 teacher-free candidate 过 macro 但 S2 失败，它只是 task diagnostic，不是 official candidate。

---

### H4：TimeAUC fail 必须在 teacher-free 和 teacher-assisted 两条路上分别分析

C3 teacher 会引入 teacher forward / distill loss / logit loading / trace cost。因此 v7.8 必须区分：

```text
M12 teacher-free TimeAUC
M12 C3-online TimeAUC
M12 C3-offline TimeAUC
M12 self-distill TimeAUC
MLP raw TimeAUC
MLP C3-distill TimeAUC
```

H4 成立标准：

如果 teacher-free M12 满足：

$$
ValLossAUC_{\text{time,M12-TF}}\leq ValLossAUC_{\text{time,MLP}},
$$

但 C3-teacher M12 不满足，则 TimeAUC fail 主要来自 teacher path / distillation accounting。

如果 teacher-free M12 也 TimeAUC fail，则需要 profiler/kernel/loop repair。

---

### H5：S1 memory repair 应在 teacher-free official candidate 上完成

如果 teacher-free M12/M13 是 official route，那么 S1 repair 不应只在 teacher-assisted M12 上验证。S1 memory source 可能与 teacher path 无关，但 final package 必须在 teacher-free candidate 上重跑。

H5 成立标准：

Teacher-free S1 candidate：

$$
r_{\text{mem,max}}<1.00,
$$

$$
r_{\text{step,max}}\leq1.35,
$$

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

---

### H6：如果 teacher-free route 失败，项目应明确降级为 teacher-assisted compression / distillation system

如果完成：

```text
teacher-free initialization repair
self-distill
longer budget
calibration-aware CE
edge update normalization
S1/cache/fused implementation
```

仍然无法让 teacher-free candidate 过 `+0.0200`，而 C3 teacher route 稳定过，则结论应改为：

```text
PureKAN-NG architecture can be an efficient student of a stronger KAN teacher,
but standalone teacher-free superiority is not yet established.
```

这不是失败，但不是最初目标。

---

## 6. Candidate 设计

### 6.1 Official Teacher-Free candidates

```text
TF0-M12-no-distill-baseline
TF1-M13-teacher-free-M5init
TF2-M12-no-distill-longbudget-480
TF3-M12-no-distill-M5init-stabilized
TF4-M12-no-distill-edge-update-normalized
TF5-M12-no-distill-NLL-balanced-CE
TF6-M12-no-distill-calibration-aware-CE
TF7-M12-no-distill-warmup-cosine
TF8-M12-no-distill-EMA-self-teacher
TF9-M12-no-distill-delayed-self-distill
TF10-M12-no-distill-snapshot-self-distill
```

### 6.2 External teacher diagnostic candidates

```text
EXT0-M12-C3-online-teacher
EXT1-M12-C3-offline-logits
EXT2-M12-C3-distill-alpha0125
EXT3-M12-C3-distill-alpha025
EXT4-M12-C3-distill-alpha050
EXT5-A2S-C3-distill
EXT6-MLP-C3-distill
```

### 6.3 Self-distill / no external teacher candidates

```text
SELF0-M12-self-distill-EMA
SELF1-M12-self-distill-delayed-40steps
SELF2-M12-self-distill-snapshot-bestval
SELF3-M12-self-distill-logit-temperature2
SELF4-M12-self-distill-logit-temperature4
```

### 6.4 S1 / fused / streaming candidates

These must be tested on teacher-free candidates first.

```text
S1TF0-TF-best-current
S1TF1-root-input-lifetime-trim
S1TF2-hidden-y-lifetime-trim
S1TF3-streaming-root-input-cache
S1TF4-streaming-hidden-y-cache
S1TF5-fused-head-forward
S1TF6-fused-linear-silu-backward
S1TF7-dense-preserving-fused-backward-combo
S1TF8-teacher-free-S1-combo
```

### 6.5 Profiler candidates

```text
PROF0-MLP-AdamW
PROF1-TF-best
PROF2-TF-best-self-distill
PROF3-EXT-M12-C3-online
PROF4-EXT-M12-C3-offline
PROF5-MLP-C3-distill
PROF6-S1TF-best
```

---

## 7. 实验阶段总览

v7.8 Teacher-Free 版分为十五个阶段：

```text
P0: contract and clean reproduction
P1: teacher-free baseline reproduction
P2: teacher-free gap attribution
P3: teacher-free repair without external teacher
P4: self-distill no-external-teacher branch
P5: external teacher diagnostic only
P6: official teacher-free 10-seed significance
P7: teacher-free time accounting and TimeAUC
P8: phase-mapped profiler for teacher-free path
P9: S1 memory attribution on teacher-free path
P10: teacher-free dense-preserving fused/streaming package
P11: full-grid S1/S2 profiler for teacher-free candidate
P12: gated scaling and robustness
P13: geometry Pareto diagnostic
P14: route decision
P15: artifact and failure audit
```

---

## 8. P0：contract and clean reproduction

### 8.1 目的

确认 v7.8 与 v7.7/v7.6 可比，并锁定 no-fake/no-proxy、strict/manual、MLP denominator、M12/M13 lineage。

### 8.2 必跑对象

```text
B0-MLP-AdamW
B2-MLP-C3-distill
M12-C3-distill
M13-teacher-free
M12-no-distill
C3-teacher
```

### 8.3 必须记录字段

```text
candidate
external_teacher_used
self_teacher_used
teacher_candidate
teacher_mode
teacher_logits_used
fake_data_used
proxy_row_used
nonKAN_param_count
uses_loss_backward
manual_backward_available
grad_relerr_max
grad_cos_min
val_acc
test_acc
val_gap_vs_MLP
val_gap_vs_MLP_distill
ECE
NLL
memory_ratio_max
step_ratio_max
S2_shape_count
S1_shape_count
ValLossAUC_step
ValLossAUC_time
```

### 8.4 判断标准

P0 pass：

```text
fake_data_used = 0
proxy_row_used = 0
nonKAN_param_count = 0 for PureKAN candidates
grad_relerr_max <= 1e-4
```

M13 reproduction：

$$
|\Delta Acc_{\text{M13}}-0.01784|\leq0.005.
$$

M12 reproduction：

$$
|\Delta Acc_{\text{M12-C3}}-0.02057|\leq0.004.
$$

### 8.5 可视化

```text
p0_teacher_dependency_bar.svg
p0_contract_heatmap.svg
p0_m12_m13_lineage.svg
p0_task_efficiency_position.svg
```

---

## 9. P1：teacher-free baseline reproduction

### 9.1 目的

建立 official teacher-free baseline，确认 M13/M12-no-distill 的真实水平。

### 9.2 必跑对象

```text
TF0-M12-no-distill-baseline
TF1-M13-teacher-free-M5init
B0-MLP-AdamW
B2-MLP-C3-distill
M12-C3-distill diagnostic only
```

### 9.3 设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0..9

steps:
  240

trace_every:
  10
```

### 9.4 必须记录字段

```text
candidate
dataset
seed
val_acc
test_acc
val_loss
test_loss
ECE
NLL
val_gap_vs_MLP
test_gap_vs_MLP
macro_val_gap_mean
ci95_low
holm_p
seed_win_rate
ValLossAUC_step
ValLossAUC_time
memory_ratio_max
step_ratio_max
```

### 9.5 判断标准

Teacher-free near pass：

$$
\Delta Acc_{\text{macro,val}}\geq0.018.
$$

Teacher-free official pass：

$$
\Delta Acc_{\text{macro,val}}\geq0.020.
$$

If M13 remains:

$$
0.017\leq \Delta Acc <0.020,
$$

then proceed to P2/P3 teacher-free gap attribution/repair.

If:

$$
\Delta Acc <0.015,
$$

then M13 is unstable; return to architecture stability before S1/time work.

### 9.6 可视化

```text
p1_teacher_free_seedwise_gap.svg
p1_teacher_free_bootstrap_ci.svg
p1_teacher_free_vs_c3_distill.svg
p1_teacher_free_auc_step_time.svg
```

---

## 10. P2：teacher-free gap attribution

### 10.1 目的

解释 teacher-free M13 与 C3-distilled M12 的 `~0.0027` 差距来自哪里。P2 不修模型，只做 attribution。

### 10.2 对比对象

```text
M13-teacher-free
M12-C3-distill
B0-MLP
B2-MLP-C3-distill
```

### 10.3 必须记录字段

```text
candidate
dataset
seed
logit_norm_mean
confidence_mean
wrong_confidence_mean
margin_mean
margin_p10
feature_rank
feature_CKA_M13_vs_M12
logit_KL_M12teacher_vs_M13
NLL_gap
ECE_gap
early_loss_slope
mid_loss_slope
late_loss_slope
hard_example_overlap
samplewise_gain_distribution
```

### 10.4 判断标准

If M13 vs M12 gap is mostly calibration/NLL：

```text
NLL gap explains most loss difference
accuracy errors overlap >= 80%
feature CKA >= 0.95
```

then use calibration-aware CE / self-distill.

If feature CKA lower：

$$
CKA(M13,M12)<0.90,
$$

then teacher changes representation; use structural/initialization repair.

If gap concentrated in early steps：

```text
early_loss_slope_M13 worse than M12
late_loss_slope similar
```

then use warmup / initialization / EMA.

### 10.5 可视化

```text
p2_m13_m12_feature_cka.svg
p2_logit_kl_distribution.svg
p2_error_overlap_venn.svg
p2_margin_distribution.svg
p2_loss_slope_comparison.svg
```

---

## 11. P3：teacher-free repair without external teacher

### 11.1 目的

让 teacher-free M13/M12-no-distill 自己跨过 `+0.0200`，不使用 C3 logits。

### 11.2 Candidates

```text
TF2-M12-no-distill-longbudget-480
TF3-M12-no-distill-M5init-stabilized
TF4-M12-no-distill-edge-update-normalized
TF5-M12-no-distill-NLL-balanced-CE
TF6-M12-no-distill-calibration-aware-CE
TF7-M12-no-distill-warmup-cosine
```

这些不是普通 optimizer 大扫，而是针对 P2 找到的 gap source。

### 11.3 必须记录字段

```text
candidate
repair_type
external_teacher_used
val_acc
test_acc
val_gap_vs_MLP
test_gap_vs_MLP
ECE
NLL
ValLossAUC_step
ValLossAUC_time
grad_relerr_max
memory_ratio_max
step_ratio_max
macro_delta_vs_M13
macro_delta_vs_M12_C3
```

### 11.4 判断标准

Teacher-free repair useful：

$$
Acc_{\text{repair}}-Acc_{\text{M13}}\geq0.002.
$$

Teacher-free official pass：

$$
\Delta Acc_{\text{macro,val,repair}}\geq0.0200.
$$

Efficiency preservation：

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

If repair requires external teacher, it is moved to P5 diagnostic.

### 11.5 可视化

```text
p3_teacher_free_repair_gain.svg
p3_repair_task_efficiency_pareto.svg
p3_repair_auc_step_time.svg
p3_repair_nll_ece_bar.svg
```

---

## 12. P4：self-distill no-external-teacher branch

### 12.1 目的

测试不依赖外部 C3 的 self-distill 是否能提供 C3 teacher 的小幅增益。

### 12.2 Candidates

```text
SELF0-M12-self-distill-EMA
SELF1-M12-self-distill-delayed-40steps
SELF2-M12-self-distill-snapshot-bestval
SELF3-M12-self-distill-logit-temperature2
SELF4-M12-self-distill-logit-temperature4
```

### 12.3 必须记录字段

```text
candidate
self_teacher_type
external_teacher_used
teacher_update_rule
temperature
val_acc
test_acc
val_gap_vs_MLP
ECE
NLL
ValLossAUC_step
ValLossAUC_time
memory_ratio_max
step_ratio_max
grad_relerr_max
```

### 12.4 判断标准

Self-distill useful：

$$
Acc_{\text{self-distill}}-Acc_{\text{M13}}\geq0.002.
$$

Official no-external pass：

$$
external\_teacher\_used=0,
$$

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

Time constraint：

$$
ValLossAUC_{\text{time,self}}\leq1.10\cdot ValLossAUC_{\text{time,M13}}.
$$

### 12.5 可视化

```text
p4_self_distill_gain.svg
p4_self_distill_time_overhead.svg
p4_self_distill_vs_external_teacher.svg
```

---

## 13. P5：external teacher diagnostic only

### 13.1 目的

C3 teacher 只用于诊断，不进入 official route。P5 回答 teacher 帮了什么。

### 13.2 Candidates

```text
EXT0-M12-C3-online-teacher
EXT1-M12-C3-offline-logits
EXT2-M12-C3-distill-alpha0125
EXT3-M12-C3-distill-alpha025
EXT4-M12-C3-distill-alpha050
EXT5-A2S-C3-distill
EXT6-MLP-C3-distill
```

### 13.3 必须记录字段

```text
candidate
external_teacher_used
teacher_mode
alpha
temperature
teacher_precompute_sec
teacher_storage_MB
teacher_forward_time
val_acc
test_acc
val_gap_vs_MLP
val_gap_vs_teacher_free
ECE
NLL
ValLossAUC_step
ValLossAUC_time_total
ValLossAUC_time_train_only
```

### 13.4 判断标准

Teacher diagnostic conclusion:

If:

$$
Acc_{\text{M12-C3}}-Acc_{\text{M13}}\leq0.003,
$$

teacher dependency mild.

If:

$$
Acc_{\text{M12-C3}}-Acc_{\text{M13}}\geq0.010,
$$

teacher dependency strong, official teacher-free route likely not solved.

If MLP-C3 reaches M12-C3 within:

$$
0.005,
$$

then teacher objective explains advantage. Current data says this did not happen, but v7.8 should keep it audited.

### 13.5 可视化

```text
p5_external_teacher_gain_bar.svg
p5_teacher_cost_breakdown.svg
p5_mlp_distill_vs_m12_distill.svg
p5_teacher_alpha_sweep.svg
```

---

## 14. P6：official teacher-free 10-seed significance

### 14.1 目的

对 P3/P4 产生的 best teacher-free candidate 做 10-seed official confirmation。

### 14.2 必跑对象

```text
B0-MLP-AdamW
B2-MLP-C3-distill diagnostic
best-teacher-free-P3
best-self-distill-P4
M12-C3 diagnostic
```

### 14.3 必须记录字段

```text
candidate
external_teacher_used
self_teacher_used
dataset
seed
val_acc
test_acc
val_gap_vs_MLP
test_gap_vs_MLP
CI95_low
Holm_p
ECE
NLL
ValLossAUC_step
ValLossAUC_time
memory_ratio_max
step_ratio_max
S2_shape_count
S1_shape_count
```

### 14.4 判断标准

Official teacher-free pass：

$$
external\_teacher\_used=0.
$$

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

If only external-teacher candidate passes, route is not official success.

### 14.5 可视化

```text
p6_teacher_free_bootstrap_ci.svg
p6_teacher_free_seedwise_gap.svg
p6_teacher_free_vs_mlp_distill.svg
p6_official_candidate_scorecard.svg
```

---

## 15. P7：teacher-free time accounting and TimeAUC

### 15.1 目的

重新评估 TimeAUC，不再让 C3 teacher 干扰 official path。teacher-free candidate 的 time accounting 是 official。

### 15.2 必跑对象

```text
B0-MLP
best-teacher-free
best-self-distill
M12-C3-online diagnostic
M12-C3-offline diagnostic
```

### 15.3 必须记录字段

```text
candidate
external_teacher_used
self_teacher_used
wall_clock_time_total
train_step_time
forward_time
backward_time
update_time
teacher_forward_time
self_teacher_time
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

### 15.4 判断标准

TimeAccountingPass：

```text
unknown_time_fraction <= 0.10
teacher/self-teacher/validation/logging/sync separated
```

Teacher-free TimeAUCPass：

$$
ValLossAUC_{\text{time,total,TF}}\leq ValLossAUC_{\text{time,B0}}.
$$

If teacher-free AUC passes but C3-teacher AUC fails, the blocker is teacher path, not model.

### 15.5 可视化

```text
p7_teacher_free_time_breakdown.svg
p7_auc_time_teacher_free_vs_teacher.svg
p7_time_to_target.svg
p7_total_vs_train_only_auc.svg
```

---

## 16. P8：phase-mapped profiler for teacher-free path

### 16.1 目的

建立 teacher-free official path 的真实 kernel attribution。C3 teacher path 可以作为 diagnostic，但不决定 official success。

### 16.2 必跑对象

```text
PROF0-MLP-AdamW
PROF1-best-teacher-free
PROF2-best-self-distill
PROF3-M12-C3-online diagnostic
PROF4-M12-C3-offline diagnostic
```

### 16.3 必须记录字段

```text
candidate
external_teacher_used
kernel_count_total
mapped_kernel_time_fraction
unknown_kernel_time_fraction
kernel_count_forward
kernel_count_backward
kernel_count_update
kernel_count_validation
top_kernel_name_1
top_kernel_phase_1
top_kernel_time_1
small_kernel_count_under_10us
layout_conversion_count
cuda_memcpy_time
cuda_sync_time
```

### 16.4 判断标准

ProfilerPass：

```text
mapped_kernel_time_fraction >= 0.90
unknown_kernel_time_fraction <= 0.10
top3 phase time explain >= 0.70
```

### 16.5 可视化

```text
p8_teacher_free_kernel_timeline.svg
p8_kernel_count_by_phase.svg
p8_top_kernel_time_bar.svg
p8_mapped_vs_unknown_kernel_time.svg
```

---

## 17. P9：S1 memory attribution on teacher-free path

### 17.1 目的

S1 memory repair 必须在 teacher-free official candidate 上做，不只在 M12-C3 上做。

### 17.2 必跑对象

```text
B0-MLP
best-teacher-free
best-self-distill
M12-C3 diagnostic
```

### 17.3 必须记录字段

```text
candidate
external_teacher_used
batch_size
memory_ratio
step_ratio
root_input_cache_MB
hidden_y_cache_MB
manual_cache_MB
optimizer_state_MB
head_temp_MB
allocator_padding_MB
top1_memory_source
top2_memory_source
top3_memory_source
unknown_memory_fraction
```

### 17.4 判断标准

S1 attribution pass：

```text
top3 memory sources identified
unknown_memory_fraction <= 0.10
```

Actionable source：

$$
\frac{M_{\text{source}}}{M_{\text{peak}}}\geq0.02.
$$

### 17.5 可视化

```text
p9_teacher_free_s1_memory_waterfall.svg
p9_bs512_memory_source.svg
p9_memory_ratio_by_batch.svg
```

---

## 18. P10：teacher-free dense-preserving fused/streaming package

### 18.1 目的

将 S1 / profiler repair 集成到 teacher-free official candidate，而不是只修 teacher-assisted M12。

### 18.2 Candidates

```text
S1TF0-TF-best-current
S1TF1-root-input-lifetime-trim
S1TF2-hidden-y-lifetime-trim
S1TF3-streaming-root-input-cache
S1TF4-streaming-hidden-y-cache
S1TF5-fused-head-forward
S1TF6-fused-linear-silu-backward
S1TF7-dense-preserving-fused-backward-combo
S1TF8-teacher-free-S1-combo
```

### 18.3 必须记录字段

```text
candidate
external_teacher_used
components
grad_relerr_max
grad_cos_min
macro_val_gap
test_gap
ECE
NLL
ValLossAUC_step
ValLossAUC_time
memory_ratio_max
step_ratio_max
S1_shape_count
S2_shape_count
kernel_count_total
materialized_tensor_count
macro_delta_vs_TF_best
```

### 18.4 判断标准

Package task preservation：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

and:

$$
\Delta Acc_{\text{new}} \geq \Delta Acc_{\text{TF-best}}-0.003.
$$

FullGridS2:

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

FullGridS1:

$$
r_{\text{mem,max}}<1.00,
$$

$$
r_{\text{step,max}}\leq1.35.
$$

### 18.5 可视化

```text
p10_teacher_free_package_pareto.svg
p10_s1_s2_shape_heatmap.svg
p10_task_preservation_bar.svg
p10_kernel_reduction_bar.svg
```

---

## 19. P11：full-grid S1/S2 profiler for teacher-free candidate

### 19.1 目的

确认 final teacher-free candidate 的 full-grid S2/S1，不允许只看 mean。

### 19.2 Shape grid

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

batch:
  128
  256
  512
```

### 19.3 必须记录字段

```text
candidate
external_teacher_used
dataset
batch
memory_ratio
step_ratio
forward_ratio
backward_ratio
S2_pass
S1_pass
top_memory_source
kernel_count_total
```

### 19.4 判断标准

FullGridS2Pass：

```text
S2 shapes = 9/9
```

FullGridS1Pass：

```text
S1 shapes = 9/9
```

### 19.5 可视化

```text
p11_teacher_free_memory_heatmap.svg
p11_teacher_free_step_heatmap.svg
p11_s1_s2_boundary.svg
```

---

## 20. P12：gated scaling and robustness

### 20.1 打开条件

Only open if:

```text
TeacherFreeMacroSignificantPass = true
FullGridS2Pass = true
GradPass = true
```

### 20.2 设置

```text
train_size:
  256
  512
  1024
  1536
  4096

noise:
  label_noise = 0.05, 0.10, 0.20
  input_noise = 0.05, 0.10

hidden_dim:
  48,64,96,128
```

### 20.3 必须记录字段

```text
candidate
external_teacher_used
train_size
noise_type
hidden_dim
val_acc
test_acc
ECE
NLL
accuracy_drop
sample_efficiency_auc
robustness_auc
memory_ratio
step_ratio
```

### 20.4 判断标准

Sample efficiency pass：

$$
AUC_{\text{data,KAN}}>AUC_{\text{data,MLP}}.
$$

Robustness pass：

$$
AccDrop_{\text{KAN}}<AccDrop_{\text{MLP}}
$$

for at least two noise settings.

---

## 21. P13：geometry Pareto diagnostic

### 21.1 打开条件

Only open after teacher-free MacroSignificant + S2.

### 21.2 目的

Geometry remains diagnostic, not hard gate.

### 21.3 必须记录字段

```text
lambda_geo
geometry_loss_type
val_acc
test_acc
ECE
NLL
ValLossAUC_step
ValLossAUC_time
geometry_metric_value
memory_ratio
step_ratio
```

### 21.4 判断标准

Useful geometry：

$$
Acc_{\lambda}\geq Acc_{\lambda=0}-0.005
$$

and one of:

$$
ECE_{\lambda}<ECE_{\lambda=0},
$$

$$
NLL_{\lambda}<NLL_{\lambda=0},
$$

$$
ValLossAUC_{\lambda}<ValLossAUC_{\lambda=0}.
$$

---

## 22. P14：route decision

### 22.1 Survivor types

```text
S0:
  TeacherFreeMacroSignificant + FullGridS1 + TimeAUCPass + ProfilerPass

S1:
  TeacherFreeMacroSignificant + FullGridS2 + TimeAUCPass

S2:
  TeacherFreeMacroSignificant + FullGridS2 but TimeAUC fail

S3:
  Teacher-free near-pass only, external teacher pass

S4:
  External teacher only success

S5:
  Self-distill success

S6:
  Teacher-free macro pass but S2 fail

S7:
  GradFail

S8:
  Profiler/time incomplete
```

### 22.2 Route cases

```text
R1-TeacherFreeFullSystemAdvantage:
  S0. Official PureKAN-NG success.

R2-TeacherFreeS2QualityAdvantage:
  S2. Quality/S2 success, system time still open.

R3-SelfDistillOfficialNoExternalTeacher:
  Self-distill passes; report as no-external-teacher route.

R4-TeacherAssistedOnly:
  Only C3 teacher passes; official standalone goal not met.

R5-TeacherFreeNearPass:
  Teacher-free remains around +0.018 to +0.020; continue architecture/optimization.

R6-KernelizationNeeded:
  Teacher-free macro pass but S1/time fail due to kernel/memory.

R7-NoReproduction:
  Teacher-free or M12 signals do not reproduce.
```

### 22.3 必须记录字段

```text
candidate
route
external_teacher_used
self_teacher_used
strict_pass
grad_pass
teacher_free_macro_pass
fullgrid_s2_pass
fullgrid_s1_pass
time_auc_pass
profiler_pass
val_gap_vs_MLP
test_gap_vs_MLP
val_gap_vs_MLP_distill
memory_ratio_max
step_ratio_max
primary_blocker
next_required_implementation
```

---

## 23. P15：artifact and failure audit

### 23.1 Required artifacts

```text
run_manifest.json
provenance_audit.csv
candidate_registry.csv
gate_config.json
p0_reproduction.csv
p1_teacher_free_baseline.csv
p2_teacher_free_gap_attribution.csv
p3_teacher_free_repair.csv
p4_self_distill.csv
p5_external_teacher_diagnostic.csv
p6_teacher_free_10seed.csv
p7_teacher_free_time_accounting.csv
p8_teacher_free_profiler.csv
p9_teacher_free_s1_attribution.csv
p10_teacher_free_fused_package.csv
p11_fullgrid_teacher_free.csv
p12_scaling_robustness.csv
p13_geometry_pareto.csv
p14_route_decision.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

### 23.2 Failure taxonomy

```text
F1_teacher_free_macro_fail
F2_external_teacher_only_success
F3_self_distill_time_fail
F4_s2_fail
F5_s1_fail
F6_time_auc_fail
F7_grad_fail
F8_profiler_incomplete
F9_teacher_dependency_strong
F10_fairness_fail_vs_mlp_distill
F11_fake_or_proxy_violation
F12_artifact_missing
```

---

## 24. 第一轮推荐执行顺序

### Step 1：P1 teacher-free baseline reproduction

先确定 M13 / M12-no-distill 的真实 10-seed gap。不要从 M12+C3 继续往下推 official success。

### Step 2：P2 teacher-free gap attribution

明确 teacher-free 缺的 `~0.002` 是 early optimization、calibration、NLL、margin、feature rank，还是 representation gap。

### Step 3：P3 teacher-free repair

只做 P2 指向的 repair，不做大 sweep。

### Step 4：P4 self-distill

如果 P3 无法过线，尝试 no-external-teacher self-distill。

### Step 5：P6 official teacher-free 10-seed

所有 teacher-free candidate 必须重新 10-seed confirm。

### Step 6：P7-P11 system path

只有 teacher-free MacroSignificantPass 后，再做 TimeAUC / profiler / S1 package 作为 official 系统路线。

### Step 7：P5 external teacher diagnostic

C3 teacher branch 仍保留，但只用于解释，不决定 official success。

---

## 25. 停止条件

### 成功停止

出现以下任一情况，立即复盘：

```text
TeacherFreeMacroSignificant + FullGridS2 + GradPass
TeacherFreeMacroSignificant + FullGridS1 + GradPass
TeacherFreeMacroSignificant + FullGridS1 + TimeAUCPass + ProfilerPass
```

### 失败停止

出现以下任一情况，停止对应路线：

```text
1. teacher-free 10-seed macro gap < +0.015；
2. all teacher-free repairs improve < +0.001；
3. only external teacher passes and MLP-distill closes gap;
4. S1 repair makes teacher-free macro gap < +0.0200；
5. self-distill adds large time overhead and fails macro;
6. GradPass fails and cannot be fixed by numerical equivalent rewrite;
7. no-fake/no-proxy audit fail。
```

---

## 26. 成功与失败解释规则

### Case A：teacher-free candidate crosses +0.020 and S2

这是最重要的成功。说明 M12 family 不需要外部 teacher，可以 official claim PureKAN-NG architecture advantage.

### Case B：self-distill crosses +0.020 but pure supervised does not

可写成 no-external-teacher success，但要明确 self-distill 是 training recipe。它仍然比 C3 teacher 更接近 official goal.

### Case C：only C3 teacher crosses +0.020

不能写 official success。只能写：

```text
PureKAN-NG is an efficient student under C3 teacher, but standalone teacher-free advantage remains unproven.
```

### Case D：teacher-free passes macro but TimeAUC fails

说明 quality advantage 成立，system wall-clock 尚未闭合。继续 profiler/kernel path.

### Case E：teacher-free passes macro but S1 fails

说明 architecture advantage 成立，S1 memory/kernel 仍需 repair.

### Case F：teacher-free fails badly

如果 gap < +0.015，说明 M12 的 official standalone signal 不稳定。需要回到 architecture / objective source，不再把 M12-C3 当终点.

---

## 27. 最终建议

v7.8 的一句话策略是：

$$
\boxed{
\text{把 official 目标从 “M12+C3 teacher 成功” 改回 “M12/M13 teacher-free 成功”。}
}
$$

当前应这样定位：

```text
M12+C3:
  diagnostic assisted success

M13 / M12-no-distill:
  official target, currently near-pass

MLP-distill:
  fairness control only

C3 teacher:
  expression teacher / upper-bound diagnostic only
```

下一步真正该做的是：

```text
1. 复现 teacher-free M13/M12-no-distill 的 10-seed gap；
2. 分析 teacher-free 到 C3-distilled 的小差距；
3. 用 no-external-teacher repair 或 self-distill 闭合这 0.2%-0.3%；
4. 只在 teacher-free 过 macro 后，再做 S1 / TimeAUC / Profiler official 系统闭环。
```

这样才能避免路线变成 “teacher-assisted student system”，重新回到最初目标：

$$
\boxed{
\text{PureKAN-NG 自身是否系统性优于 MLP-AdamW。}
}
