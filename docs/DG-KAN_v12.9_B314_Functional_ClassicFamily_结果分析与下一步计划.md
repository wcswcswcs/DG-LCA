# DG-KAN v12.9：B314 Base Hardening、Functional Official Re-entry、经典基函数高效率化并行计划

> 基于 v12.8.3 `B109 / 经典基函数全家族高效率化 / Functional Geometry` 结果复盘重新制定。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 重要修正：当前总目标不是“再找至少一个 base”。现在已经有 B314/B109 这个合格 anchor。下一步是把它确认成稳健 base，并在其上证明 functional update 的独立几何贡献；经典基函数支线继续推进，但不再拖住 B314 主线，也不能因为 B314 过了而放弃。

---

## 0. 一句话判断

这轮不是慢到没有进展，而是第一次出现了一个真正可以作为主线 base 的结果：

$$
\boxed{
\text{B314 / B109 已经打开 base gate：效率、显存、A5 task、AUC、ECE 全部过。}
}
$$

但它还没有完成整个项目目标，因为：

$$
\boxed{
\text{Functional official success 仍然是 0，经典基函数 portfolio 仍然没有 family success。}
}
$$

所以现在项目进入了一个新阶段：

```text
过去：
  找一个 strict FC-PureKAN base 是否可能。

现在：
  已有 B314 anchor，要验证它是否稳健；
  然后证明 functional update 是否在这个 base 上提供独立几何收益；
  同时继续推进 Fourier / Wavelet / B-spline / Chebyshev / RBF-FastKAN / Rational 的高效率化支线。
```

这不是小修阶段。下一步不能再围绕温度、epoch、optimizer wrapper 或 CE calibration 打转，而要围绕三个本质问题：

```text
Q1. B314 的成功是不是稳健的 strict FC-PureKAN base success？
Q2. Functional update 是否能在 B314 上击败 strong controls，并改善 signal-channel geometry？
Q3. 经典 KAN basis 是否能在 lower-level fused implementation 下进入 MLP-like efficiency / task / geometry envelope？
```

---

## 1. 当前事实锚点

### 1.1 B314 / B109 已经从 near-base 变成 base anchor

v12.6 时，B109 的状态是：

```text
F3 full-step pass；
A4 pass；
task mean / worst / near / ECE 过；
但 AUC-step / AUC-time fail；
base_qualified = false。
```

v12.8.3 中，B314/B315 在 10-seed confirm 下首次同时打开 F3 efficiency 和 A5 task gate。当前最好的是：

```text
B314b-directRamp105
F3 step ratio = 0.7029370517485489
memory ratio = 0.1295238095238095
backward ratio = 0.5194831389214438
update ratio = 0.5839953083298681
task mean delta = 0.023567708333333333
worst delta = 0.00390625
near pass = 1.0
ECE ok = 1
AUC ok = 1
A5 pass = 1
```

这说明 B314 不只是“最终 accuracy 好看”。它同时满足：

```text
单步效率过；
显存大幅低于 MLP baseline；
A5 task 过；
AUC-step / AUC-time 过；
ECE 过；
10-seed confirm 过；
no fake / proxy / cpu audit 过。
```

因此本轮最大的正进展是：

$$
\boxed{
\text{B109/FHQ 线第一次提供了一个可用于 functional official re-entry 的 base anchor。}
}
$$

### 1.2 但 Functional official 仍未成功

当前 route 中：

```text
base_qualified = 1
functional_open = 1
official_functional_success = 0
```

这句话非常关键。它表示：

```text
base 已经允许 functional 进入 official re-entry；
但 functional update 本身还没有通过 official strong-control gate。
```

原始 functional re-entry 没有 beat controls：

```text
B109_functional_reentry_measured = 1
B109_functional_beats_controls = 0
official_functional_success = 0
```

后来做了 functional delta-score repair，修正了一个重要评估退化：零更新在 ridge coupling 里可能被错误记成 `CouplingR2=1.0`。修复后，`F7-TaskPlusSNRGeometryResidual-w025` 在 corrected diagnostic metric 上略微击败 strong controls：

```text
best control score = 0.12789996150526006
best functional score = 0.12906943744962096
gap vs best control = 0.0011694759443608982
beats controls = 1
official success = 0
```

这个信号有价值，但不能过度解读。正确解释是：

$$
\boxed{
\text{Functional 出现了 corrected diagnostic positive，}
\text{但还没有在预注册 official gate 下证明成功。}
}
$$

所以下一步不是“宣布 functional 成功”，也不是“换个分数就过”，而是：

```text
1. 固化 corrected metric；
2. 加入 no-op / random / AdamWParallel / SNR-only / shuffled 等 controls；
3. 做 one-step / five-step / short-run official re-entry；
4. 证明功能性改善不是评价退化或 control 漏洞。
```

### 1.3 Rational 从 KernelBlocked 推进到了 TaskBlocked，但不是 family success

v12.8.3 中 Rational 支线不是完全没进展。它经历了多个阶段：

```text
B7ir:
  analytic gradcheck 过，但 L3 step ratio = 1.2507，略超 1.25 gate；
  KernelBlocked。

B7iv/B7iw:
  hiddenResVJPTriton 把 L3 step 降到约 1.12；
  L3/A4 打开；
  但 A5 失败，mean/worst/near/ECE/AUC 都不过。

B7jx:
  L3/A4 打开；
  A5 失败，mean = -0.013997，worst = -0.076172，near = 0.6，ECE/AUC fail。

B7kc:
  bounded rational residual amplitude 0.05；
  L3 step = 1.0464，memory = 0.7710，backward = 0.7993，update = 0.4837；
  L3/A4 打开；
  A5 仍失败，mean = -0.014453，worst = -0.076172，near = 0.6，ECE/AUC fail。
```

这说明 Rational 的当前 blocker 已经不是：

```text
数学梯度 correctness；
lower-level L3 kernel 完全不可行；
A4 expression 完全不可行。
```

而是：

$$
\boxed{
\text{Rational 的 generic-logit task geometry / calibration / AUC trajectory 不健康。}
}
$$

所以 Rational 后续不应该继续：

```text
CE-specific calibration；
只调 cap；
只调 hidden residual amplitude；
只用 task loss 最终准确率筛选。
```

而要回答：

```text
Rational 的 signal-channel 是否错位？
真实信号是否被困在 reservoir？
噪声是否进入 signal channel？
denominator / r' / r'' 是否造成 tail calibration 伤害？
```

### 1.4 经典基函数 portfolio 仍未完成

当前 route 显示：

```text
classic_family_pass_count = 0
```

这不能解释为“经典基函数都失败”。更准确是：

```text
当前已经深入推进 Rational；
其他 family 的 official L3 / A4 / A5 / Line C closure 仍未完成。
```

v12.8.2 计划已经要求 classic family 不是 optional，而是 mandatory portfolio：

```text
B-spline
Rational
RBF / FastKAN
Chebyshev
Fourier
Wavelet
```

每个 family 都必须被推进到明确状态：

```text
FamilyPass
FamilyNearPass
KernelBlocked
ExpressionBlocked
TaskBlocked
GeometryBlocked
RejectedForThisVersion
```

而不能因为 B314 过了，就把这些 family 写成“没跑所以失败”。

---

## 2. 这次到底有多大进展？

### 2.1 是重大进展：base 终于合格

从 v12.6 到 v12.8.3，路线变化很明显：

```text
v12.6:
  B109 是 best，但 AUC-step/time fail，base_qualified = false。

v12.8.3:
  B314/B315 10-seed confirm 同时过 F3 efficiency 与 A5 task；
  B314 成为当前 B109 base candidate；
  base_qualified = true。
```

这是项目阶段变化，不是普通小改。它意味着 functional update 不再卡在“base 未合格所以不能 official”的旧阻塞上。

### 2.2 为什么仍然感觉慢？

因为你的最终目标不是“有一个快的 KAN base”，而是：

$$
\boxed{
\text{PureKAN base + functional update} >
\text{PureKAN base + ordinary backprop}
}
$$

现在只完成了前半段：

```text
Base anchor:
  已经出现。

Functional independent advantage:
  还没有 official 成功。

经典 basis portfolio:
  仍未完成。

外部公平评估:
  还未打开。
```

所以感觉慢是合理的。它不是没有进度，而是终于到了更难的第二阶段：**证明 functional update 的因果价值**。

### 2.3 不要被报告结论牵着走：我自己的独立判断

报告 route 是：

```text
R4-B109NearOrPassClassicFamiliesKernelBlocked
```

这个 route 名字里有 “NearOrPass”，容易让人觉得 B109 还只是 near-pass。但从数据看，B314 已经在当前 protocol 下：

```text
F3 efficiency pass；
A5 pass；
Line C measured；
Line C nontearing pass；
10-seed confirm 通过；
no fake / proxy / cpu audit 通过。
```

所以我不会把 B314 还放在“near-base”里。我的判断是：

$$
\boxed{
\text{B314 是当前 protocol 下的合格 base anchor，}
\text{但仍需要更宽协议 hardening，才能作为最终论文 base。}
}
$$

同时，functional 的 corrected delta-score positive 不能被当作 success。因为 gap 只有 `0.001169`，且它是 corrected diagnostic，而不是完整 official short-run。我的判断是：

$$
\boxed{
\text{functional 出现了一个值得追的正信号，但证据等级仍低。}
}
$$

Rational 的判断也要精确：

$$
\boxed{
\text{Rational 不再是单纯 kernel fail；它已经进入 task-geometry fail。}
}
$$

这比“Rational 失败”更有信息量。

---

## 3. 当前真正卡在哪里？

### 3.1 主 blocker 1：Functional official success 还没有过 strong controls

现在 base 已经打开，functional 不能再用“base not qualified”解释失败。真正 blocker 变成：

$$
\boxed{
\text{functional update 是否有 control-resistant 的独立几何收益？}
}
$$

必须区分三种东西：

```text
task AdamW delta:
  由普通反传提供的任务下降。

geometry residual:
  希望由 functional update 提供的几何修复。

evaluation metric:
  用来判断这件事是否真的发生。
```

当前 corrected delta-score 说明旧评价有退化：no-op 零更新可能被 ridge coupling 高估。这是重要发现，但它也说明我们过去一部分 functional negative / positive 可能混有 evaluator bug。因此下一步必须先固化 evaluator，再谈 official。

### 3.2 主 blocker 2：B314 的成功机制还没有被解释

B314 名字很长：

```text
B314b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090
```

它不是一个纯粹单一 basis，而是多个 task-geometry 组件组合：

```text
learnableP
trainProbeP
signalBroad035
signalBlock015
quadReadInit125
directRamp105
direct050
fusedProjGradAdamW
absdiag050
manualAdamW
classbranch
fixedgain
gainramp075
identitytailquad030
hingeamp025
temp090
```

它过 gate 是好事，但如果我们不知道哪些组件是必要的，就会遇到两个风险：

```text
1. B314 只是 current protocol 下的组合偶然；
2. 后续 functional 或 classic basis 迁移时不知道该保留什么。
```

所以 B314 现在需要 **mechanism autopsy**，但不能变成小网格调参。Autopsy 只做解释和鲁棒性，不做 dataset-tune。

### 3.3 主 blocker 3：经典 family 不能再只做 Rational 局部小修

Rational 已经从 KernelBlocked 推到 TaskBlocked。继续调：

```text
cap = 1.00 / 1.25 / 1.50；
hidden residual = 0.05 / 0.10 / 0.15；
rank = 80 / 96 / 112 / 120 / 128 / 132 / 136；
```

可能会继续消耗很大，但不解决 task geometry。

下一步 Rational 要换问题：

```text
不是怎么让 Rational 过 L3 / A4；
而是 Rational 为什么过了 L3/A4 仍然 A5 坏。
```

同理，B-spline / RBF / Chebyshev / Fourier / Wavelet 也不能只是 naive PyTorch reference。它们要有 family-specific fused route。

### 3.4 主 blocker 4：Line C 需要从“测了”升级为“解释失败与验收成功”

B314 的 Line C：

```text
CouplingR2 = 0.23202819810379627
NoiseSignalLeak = 0.035404518246650696
LineC nontearing pass = 1
```

这说明 B314 没有明显 tearing。但现在 Line C 还没有足够解释：

```text
Functional 为什么原 official 不 beat controls？
Rational 为什么 task blocked？
B314 的成功是否因为 signal channel 更好？
Classic basis failure 是 kernel、expression、task 还是 geometry？
```

Line C 应该成为下一步所有路线的共同解释层。

---

## 4. v12.9 总目标

v12.9 的总目标不是“再找一个 base”，而是：

$$
\boxed{
\text{以 B314/B109 为当前 base anchor，证明或证伪 functional update 的独立几何收益，}
}
$$

同时：

$$
\boxed{
\text{继续推进经典 KAN basis portfolio 的 family-specific high-efficiency closure，}
\text{不因 B314 成功而放弃经典基函数。}
}
$$

更具体地说，v12.9 要完成三件事：

```text
Goal A:
  B314 hardening。
  判断 B314 是否是稳健 base，而不是 current protocol artifact。

Goal B:
  Functional official re-entry。
  判断 F7 / signal-geometry residual 是否能在 corrected official gate 下击败 controls，并转化为 task-safe short-run gain。

Goal D:
  Classic family closure。
  对 Rational、B-spline、RBF/FastKAN、Chebyshev、Fourier、Wavelet 给出可审计状态。
```

---

## 5. 核心假设

### H1：B314 是真实 base anchor，而不是 protocol artifact

假设：

$$
\boxed{
\text{B314 在更宽协议下仍保持 MLP-like efficiency、A5 task pass、Line C nontearing。}
}
$$

需要验证：

```text
更多 seed；
不同 train-size；
更长 epoch；
不同 batch size；
same-param / same-step / same-FLOP MLP baseline；
architecture clean timing vs diagnostic hook timing 分离；
no fake / proxy / CPU offload；
strict PureKAN parameter audit。
```

成立标准：

$$
T_{step,q90}/T_{MLP,q90}\le 0.85
$$

或至少：

$$
T_{step,q90}/T_{MLP,q90}\le 1.00
$$

并且：

$$
M_{q90}/M_{MLP}\le 0.30
$$

或至少：

$$
M_{q90}/M_{MLP}\le 0.60.
$$

Task gate：

$$
\Delta Acc_{mean}\ge 0,
$$

$$
\Delta Acc_{worst}\ge -0.003,
$$

$$
NearPass = 1.0,
$$

$$
AUCstep\le 1.00,\quad AUCtime\le 1.00,
$$

$$
ECE_{B314}\le ECE_{MLP}+0.02.
$$

Line C gate：

$$
CouplingR^2_{B314}\ge CouplingR^2_{MLP}-0.02,
$$

$$
NoiseSignalLeak_{B314}\le NoiseSignalLeak_{MLP}+0.02.
$$

如果 H1 不成立，functional official 必须暂停，先修 base robustness。

### H2：B314 的成功来自 label-free task-geometry trajectory，不是 CE-specific calibration

假设：

$$
\boxed{
\text{B314 的 directRamp / trainProbeP / signalBroad / signalBlock 组合提供的是通用 task-geometry trajectory，}
\text{不是 CE-specific 或 dataset-specific trick。}
}
$$

验证方式：

```text
component ablation；
same dL/dlogits generic backward check；
不读取 label 的 forward trajectory audit；
不同 loss-delta diagnostic，不改变训练 loss；
同样 protocol 下 shuffled / random-label diagnostic；
对 MNIST / Fashion / KMNIST 只做 failure slice，不做 dataset branch。
```

成立标准：

```text
移除关键 component 时，AUC 或 Line C 有可解释下降；
但不出现某个 dataset-specific 组件决定成败；
所有 backward/update 仍只接收 generic dL/dlogits；
没有 CE-only VJP 或 label branch。
```

### H3：Functional delta-score repair 揭示了真实可追 signal，但需要 official 化

假设：

$$
\boxed{
\text{F7 = TaskPlusSNRGeometryResidual-w025 在 corrected metric 上的 control gap 不是 evaluator artifact，}
\text{而是 functional update 的真实弱正信号。}
}
$$

要先冻结 corrected metric：

$$
Score_{func}
=
CouplingR^2
+
\alpha\cdot ReservoirGain
-
\beta\cdot NoiseIncrease
-
\gamma\cdot NLLIncrease
-
\lambda\cdot CostOverhead.
$$

其中默认：

$$
\alpha=1,\quad \beta=1,\quad \gamma=1,\quad \lambda=0.1.
$$

并且 zero-delta 规则必须固定：

$$
\Delta U = 0 \Rightarrow CouplingR^2=0.
$$

成立标准：

```text
F7 / F8 类 functional update 在 one-step、five-step、short-run 中都击败 strong controls；
gap 不只是 0.001 级别单次波动；
task non-harm；
Line C 指标改善；
amortized overhead 不超过 5%。
```

### H4：Rational 当前失败来自 task geometry / calibration，而不是 kernel 或 grad correctness

假设：

$$
\boxed{
\text{Rational 已经具备 L3/A4 能力，但 generic logits 下 signal-channel / calibration 结构不健康。}
}
$$

验证方式：

```text
B7kc / B7jx / B7iw 等 TaskBlocked candidate 做 Line C autopsy；
对比 B314、MLP、Rational；
测 denominator safety、r' / r''、NoiseSignalLeak、RealSignalReservoirRatio、CEp99、margin_p10；
不 CE-tune。
```

成立标准：

```text
如果 Rational 的 CouplingR2 低或 NoiseSignalLeak 高：
  说明 geometry / signal-channel 坏。

如果 Line C 正常但 task/ECE 坏：
  说明 calibration/logit scale 或 output geometry 坏。

如果 denominator / r' 爆：
  说明 Rational basis geometry 不稳。

如果这些都正常但 A5 坏：
  检查 optimizer trajectory / AUC-step attribution。
```

### H5：经典 family 必须被 family-specific kernel 公平评价

假设：

$$
\boxed{
\text{经典 basis 的旧失败不能代表 family 失败；每个 family 需要自己的 fused kernel 和几何诊断。}
}
$$

成立标准：

对每个 family 输出明确状态：

```text
FamilyPass
FamilyNearPass
KernelBlocked
ExpressionBlocked
TaskBlocked
GeometryBlocked
RejectedForThisVersion
```

状态必须由真实 artifacts 支撑：

```text
gradcheck
efficiency
expression
task
Line C
functional diagnostic
failure table
```

---

## 6. v12.9 并行实验总流程

v12.9 分四条线并行：

```text
Line A:
  B314 base hardening and mechanism autopsy。

Line B:
  Functional official re-entry on B314。

Line C:
  Manifold-Channel Geometry Diagnostics for all candidates。

Line D:
  Classic basis family closure。
```

执行顺序不是完全串行，但 official gate 必须受控：

```text
B314 hardening 可以和 classic family kernel 同时跑；
Functional one-step/five-step 可以并行跑；
Functional short-run official 只有 corrected gate + strong controls 通过后才跑；
Classic family task 只有 L3/A4 通过后才跑；
任何时候都不按 dataset name 调参。
```

---

# 7. Line A：B314 Base Hardening

## 7.1 目标

确认 B314 是稳健 base，而不是 10-seed 当前协议上的偶然结果。

## 7.2 实验设计

### A0：Artifact / contract audit

记录：

```text
candidate_id
parameter_roles
edge_param_count
nonKAN_param_count
uses_loss_backward
uses_torch_autograd_graph
uses_fused_FHQ
uses_generic_dlogits
ce_specific_backward
dataset_branch_used
teacher_used
loss_modified
sampler_changed
class_weight_used
fake_data_used
proxy_row_used
cpu_offload_used
```

通过标准：

```text
nonKAN_param_count = 0
uses_loss_backward = 0
ce_specific_backward = 0
dataset_branch_used = 0
teacher/loss/sampler/class_weight/fake/proxy/cpu = 0
```

### A1：Reconfirm 10-seed exact protocol

重跑：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0..9
train_size = 1024
val_size = 512
test_size = 512
epochs = 3
batch_size = 128
task_compile_warmup_steps = 12
```

对照：

```text
MLP same-param AdamW
MLP same-step / same-FLOP if available
B314
B315
B109 older protocolfix final075
```

记录字段：

```text
dataset
seed
method
val_acc
test_acc
val_loss
test_loss
AUC_step
AUC_time
ECE
NLL
CEp99
margin_p10
step_ratio_q50
step_ratio_q90
forward_ratio
backward_ratio
update_ratio
memory_ratio_q90
kernel_count
compile_warmup_removed
```

通过标准：

```text
B314 mean delta >= 0
B314 worst delta >= -0.003
near pass = 1.0
AUC step/time pass = 1
ECE ok = 1
step q90 <= 0.85, or at least <= 1.0
memory q90 <= 0.30, or at least <= 0.60
```

### A2：Wider protocol hardening

不要马上外推到大任务，但要扩大协议：

```text
train_size = 1024, 2048, 4096
epochs = 3, 5, 8
batch_size = 64, 128, 256
seeds = 0..4 for broad grid
```

这不是超参调优。B314 参数固定，不按数据集改。目标是看鲁棒性边界。

记录：

```text
protocol_id
train_size
epochs
batch_size
dataset
seed
B314_vs_MLP_acc_delta
B314_vs_MLP_AUC_step_ratio
B314_vs_MLP_AUC_time_ratio
ECE_delta
step_ratio
memory_ratio
LineC_CouplingR2
LineC_NoiseSignalLeak
```

通过标准：

```text
至少 80% protocol rows 维持 A5 pass；
没有一个数据集系统性 collapse；
batch_size 改变不导致 step/memory gate 失真；
Line C 不出现明显 tearing。
```

### A3：B314 component autopsy

目的不是调参，而是解释机制。

做 ablation：

```text
A3-0 full B314
A3-1 remove trainProbeP
A3-2 remove signalBroad035
A3-3 remove signalBlock015
A3-4 directRamp100 / 105 / 110 bracket
A3-5 remove fusedProjGradAdamW, keep math if possible
A3-6 remove identitytailquad030
A3-7 remove absdiag050
A3-8 classbranch fixedgain vs classgain
```

只用小预算：

```text
seeds = 0,1,2
epochs = 3
```

记录：

```text
component_removed
A1/F3 efficiency
A4 expression
A5 task
AUC-step/time
Line C
functional corrected score
```

解释规则：

```text
如果某组件 removal 只影响 timing，不影响 A5：
  它是 system component。

如果影响 AUC/Line C：
  它是 trajectory/geometry component。

如果影响 expression：
  它是 representational component。

如果只在某 dataset fail：
  记为 failure slice，不允许做 dataset-specific branch。
```

## 7.3 可视化

必须生成：

```text
fig_A_b314_reconfirm_acc_delta_ci.svg
fig_A_b314_auc_step_time_by_seed.svg
fig_A_b314_efficiency_memory_box.svg
fig_A_b314_protocol_robustness_heatmap.svg
fig_A_component_ablation_radar.svg
fig_A_component_ablation_auc_vs_lineC.svg
```

## 7.4 Codex failure action

如果 A1 exact protocol 失败：

```text
1. 检查 compile warmup / timing accounting；
2. 检查 candidate id 是否完整；
3. 检查 trainProbeP / directRamp schedule 是否按原协议执行；
4. 不允许改 dataset-specific 参数；
5. 如果复现仍失败，route 改为 B314NotStable，暂停 functional official short-run。
```

如果 A2 wider protocol 部分失败：

```text
1. 先分离 train-size / batch-size / epoch 引起的 failure；
2. 如果 batch-size failure 来自 kernel occupancy，修 kernel；
3. 如果 train-size failure 来自 Line C tearing，进入 C-line autopsy；
4. 如果 epochs 增加后退化，查 overfit/noise leak；
5. 不用单个 dataset 调参。
```

---

# 8. Line C：Manifold-Channel Geometry Diagnostics

## 8.1 目标

Line C 不再只是附属指标。它要解释：

```text
B314 为什么成功；
Functional 是否真的改善几何；
Rational 为什么 TaskBlocked；
Classic family 是 task fail 还是 geometry fail。
```

## 8.2 核心指标

对 update batch $B$ 和 probe batch $Q$，记录窗口位移：

$$
\Delta U_B=f_{\theta_{t+\Delta}}(B)-f_{\theta_t}(B)
$$

$$
\Delta U_Q=f_{\theta_{t+\Delta}}(Q)-f_{\theta_t}(Q)
$$

拟合 ridge predictor：

$$
A_t=\arg\min_A \|\Delta U_Q-A\Delta U_B\|_F^2+\lambda\|A\|_F^2
$$

定义：

$$
CouplingR^2=
1-
\frac{\|\Delta U_Q-A_t\Delta U_B\|_F^2}
{\|\Delta U_Q\|_F^2+\epsilon}
$$

构造 signal / reservoir projector：

$$
\hat W_B=\sum_{\tau=t}^{t+\Delta}\hat K_{BB}(\tau)
$$

真实信号困在 reservoir：

$$
RealSignalReservoirRatio=
\frac{\|P_{res}r_{real}\|^2}{\|r_{real}\|^2+\epsilon}
$$

噪声泄漏进 signal：

$$
NoiseSignalLeak=
\frac{\|P_{sig}r_{noise}\|^2}{\|r_{noise}\|^2+\epsilon}
$$

## 8.3 记录字段

`v129_lineC_train_probe_coupling.csv`：

```text
run_id
candidate_id
basis_family
method
dataset
seed
step
window_size
batch_size_B
batch_size_Q
ridge_lambda
train_logit_drift_l2
probe_logit_drift_l2
CouplingR2
CouplingCorr
coupling_residual_norm
KernelDrift
CEp99_delta
ECE_delta
margin_p10_delta
official_gate_open
```

`v129_lineC_signal_reservoir.csv`：

```text
run_id
candidate_id
basis_family
dataset
seed
step
sketch_dim
signal_effective_rank
signal_mass_topk
reservoir_fraction
top_eigen_share
RealSignalReservoirRatio
NoiseSignalLeak
SNR_positive_fraction
real_noise_gap
```

`v129_lineC_summary.csv`：

```text
candidate_id
base_qualified
functional_official_open
CouplingR2_mean
CouplingR2_worst
NoiseSignalLeak_mean
NoiseSignalLeak_worst
RealSignalReservoirRatio_mean
RealSignalReservoirRatio_worst
CEp99_mean
ECE_mean
NLL_mean
nontearing_pass
```

## 8.4 判断标准

Base nontearing pass：

$$
CouplingR^2_{KAN}\ge CouplingR^2_{MLP}-0.02
$$

$$
NoiseSignalLeak_{KAN}\le NoiseSignalLeak_{MLP}+0.02
$$

$$
CEp99_{KAN}\le CEp99_{MLP}+\epsilon_{tail}
$$

Functional improvement pass：

$$
CouplingR^2_{func}\ge CouplingR^2_{base}+0.02
$$

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{base}-0.02
$$

$$
RealSignalReservoirRatio_{func}\le RealSignalReservoirRatio_{base}-0.02
$$

且：

$$
Acc_{func}\ge Acc_{base}-0.003
$$

$$
AUCtime_{func}\le AUCtime_{base}
$$

$$
ECE_{func}\le ECE_{base}+0.01
$$

## 8.5 可视化

```text
fig_C_coupling_predicted_vs_actual.svg
fig_C_coupling_r2_by_candidate.svg
fig_C_noise_signal_leak_by_candidate.svg
fig_C_real_signal_reservoir_ratio.svg
fig_C_kernel_drift_vs_coupling.svg
fig_C_auc_vs_real_signal_reservoir.svg
fig_C_noise_leak_vs_ECE.svg
fig_C_rational_vs_b314_lineC.svg
```

## 8.6 Codex failure action

如果 CouplingR2 不稳定：

```text
1. 增大 probe batch；
2. classwise centering logits；
3. 调 ridge_lambda；
4. 使用 PCA sketch；
5. 缩短 window；
6. 不降低 gate。
```

如果 NoiseSignalLeak 高：

```text
1. 检查 shuffled-label residual；
2. 检查 signal projector 阈值；
3. 改为 top-energy signal projector；
4. 对 functional 尝试 AdamW-orthogonal residual；
5. 不做 dataset-specific filter。
```

如果 RealSignalReservoirRatio 高：

```text
1. 检查 basis/rank；
2. 检查 top-eigen collapse；
3. 尝试 condition-preserving init；
4. functional 尝试 signal-projected geometry repair；
5. 不用 teacher 或 loss 改动。
```

---

# 9. Line B：Functional Official Re-entry

## 9.1 目标

在 B314 base 上证明 functional update 不是 control 等价扰动，而是带来独立几何收益。

## 9.2 阶段 B0：固化 corrected metric

先冻结 metric，不允许根据结果反复改。

默认 corrected score：

$$
Score=
CouplingR^2
+
ReservoirGain
-
NoiseIncrease
-
NLLIncrease
-
0.1\cdot CostOverhead.
$$

其中：

$$
ReservoirGain=
RealSignalReservoirRatio_{base}
-
RealSignalReservoirRatio_{candidate}
$$

$$
NoiseIncrease=
NoiseSignalLeak_{candidate}
-
NoiseSignalLeak_{base}
$$

$$
NLLIncrease=
NLL_{candidate}
-
NLL_{base}
$$

zero-delta rule：

$$
\|\Delta U\|<\epsilon \Rightarrow CouplingR^2=0.
$$

通过标准：

```text
NoOp score 不得因为 zero delta 变成高分；
RandomMatchedNorm 不得系统性优于 task-safe functional；
AdamWParallel 必须作为强控制保留。
```

## 9.3 阶段 B1：one-step / five-step cloned audit

候选：

```text
F0 TaskOnlyAdamW
F1 NoOpMatchedOverhead
F2 RandomMatchedNorm
F3 AdamWParallelDirection
F4 SNR-only
F5 GeometryOnlyNoSNR
F6 ShuffledPayload/Event
F7 TaskPlusSNRGeometryResidual-w025
F8 TaskPlusOrthogonalGeometryResidual
F9 GeometryResidualOnly
F10 BasisAwareFunctionalResidual, if available
```

记录：

```text
candidate_id
functional_update_id
control_id
dataset
seed
step
delta_norm
cos_with_adamw
task_delta
probe_task_delta
CouplingR2
RealSignalReservoirRatio
NoiseSignalLeak
NLL_delta
ECE_delta
CEp99_delta
margin_p10_delta
Score
Score_gap_vs_best_control
beats_controls
bad_step
cost_overhead
```

通过标准：

```text
one-step beats_controls rate >= 0.70
five-step beats_controls rate >= 0.60
bad_step_rate <= 0.02
task_delta not worse than AdamWParallel by > 0.003
Score_gap_vs_best_control mean >= 0.005
or bootstrap CI95 lower >= 0
```

注意：当前 gap `0.001169` 是 positive diagnostic，但不足以作为强 official margin。它可以打开 B2，但不能直接宣布 success。

## 9.4 阶段 B2：short-run functional re-entry

只有 B1 过才跑。

方法：

```text
B314 base + AdamW/FHQ baseline
B314 + NoOp matched overhead
B314 + RandomMatchedNorm event
B314 + AdamWParallel event
B314 + SNR-only
B314 + F7 functional event
B314 + F8 functional event
MLP + analogous geometry maintenance
```

设置：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0..4 first
epochs = 3 / 5
event period = fixed K or geometry debt trigger
```

不能按 dataset 调 event。触发只能基于：

```text
geometry debt；
SignalReservoir metrics；
SNR positive fraction；
tail drift；
AUC plateau；
task slack。
```

记录：

```text
val_acc
test_acc
AUC_step
AUC_time
ECE
NLL
CEp99
margin_p10
LineC metrics
event_count
accepted_event_count
rejected_event_count
functional_overhead
control_gap
```

通过标准：

$$
Acc_{func}\ge Acc_{B314}-0.003
$$

$$
AUCtime_{func}\le AUCtime_{B314}
$$

$$
ECE_{func}\le ECE_{B314}+0.01
$$

$$
Score_{func}\ge Score_{bestControl}+0.005
$$

$$
T_{amortized}/T_{B314}\le 1.05
$$

并且：

```text
NoOp / Random / AdamWParallel / SNR-only 不得同样通过。
```

## 9.5 阶段 B3：10-seed official functional confirm

只有 B2 通过才跑。

```text
seeds = 0..9
methods:
  B314 baseline
  B314 + best functional
  B314 + best strong control
  MLP baseline
  MLP analog functional
```

通过标准：

```text
paired Score delta vs best control CI95 lower > 0
paired AUCtime delta vs B314 <= 0
paired ECE delta <= 0 or within +0.005
paired Acc delta >= -0.003
overhead <= 1.05
```

## 9.6 Codex failure action

如果 F7 只在 corrected metric 过、原 gate 不过：

```text
1. 先不要宣布 success；
2. 对原 gate 做 zero-update degenerate audit；
3. 如果原 gate 有确认 bug，预注册 corrected official gate 后重跑；
4. 如果原 gate 无 bug，F7 只保留 diagnostic；
5. 不允许临时改 score 权重。
```

如果 functional 改善 Line C 但 task 变差：

```text
1. 加 task-safe backtracking；
2. 降低 lambda；
3. 加 projection to AdamW non-harm cone；
4. 若仍 task harm，停止该 functional direction。
```

如果 functional 被 AdamWParallel 解释：

```text
1. 计算 AdamW-orthogonal component；
2. 限制 functional 到 task-gradient null/complement subspace；
3. 加 shuffled payload control；
4. 如果仍被解释，写为 no independent functional advantage。
```

---

# 10. Line D：经典基函数全家族高效率化 closure

## 10.1 目标

不让 B314/FHQ 成功替代经典 KAN basis 研究。每个 family 都必须被推进到明确状态。

```text
D1 B-spline
D2 Rational
D3 RBF / FastKAN
D4 Chebyshev
D5 Fourier
D6 Wavelet
```

共同流程：

```text
D-P0 manifest / implementation contract
D-P1 gradcheck
D-P2 L2/L3 efficiency
D-P3 expression A4
D-P4 task A5
D-P5 Line C geometry
D-P6 family status
```

共同记录：

```text
family
candidate_id
implementation_level L0/L1/L2/L3
uses_dense_basis_materialization
uses_loss_backward
gradcheck_relerr
gradcheck_cos
forward_ratio
backward_ratio
update_ratio
step_ratio
memory_ratio
kernel_count
A4_expression_pass
A5_task_pass
LineC_nontearing_pass
family_status
failure_reason
```

Family official pass：

$$
T_{step}/T_{MLP}\le1.10
$$

$$
M_{step}/M_{MLP}\le0.80
$$

$$
A4=1,\quad A5=1,\quad C_{nontearing}=1
$$

Exploratory pass：

$$
T_{step}/T_{MLP}\le1.35
$$

$$
M_{step}/M_{MLP}\le1.00
$$

$$
A4\ge A4_{B314}-0.02
$$

### 10.2 D1 B-spline

技术难点：

```text
Cox-De Boor recursion 不适合 GPU；
dense [B,d,K] basis materialization 太贵；
knot gradient scatter/reduce 可能成为 backward blocker；
learnable knots 会引入排序和稳定性问题。
```

候选：

```text
SPL1 UniformLinearSpline-local2
SPL2 TritonLocalSpline-degree1
SPL3 LocalCubicSpline-local4
SPL4 MatrixSpline-degree3 diagnostic
```

优先公式：

$$
q=\left\lfloor\frac{x-x_{min}}{\Delta}\right\rfloor
$$

$$
t=\frac{x-(x_{min}+q\Delta)}{\Delta}
$$

$$
s(x)=(1-t)c_q+t c_{q+1}
$$

实现要求：

```text
不构造 dense [B,d,K]；
block over B x O；
loop over d；
dC 用 block-level reduction；
先 fixed uniform knots，不学 knots。
```

记录：

```text
degree
num_knots
active_basis_per_input
out_of_grid_fraction
bin_occupancy_entropy
empty_bin_fraction
knot_grad_collision_rate
second_difference_energy
slope_p95
scatter_atomic_count
```

失败后 Codex 先试：

```text
forward 慢：
  linear degree1；
  local2；
  fixed uniform knots；
  Triton tiled kernel；
  bf16 coeff read + fp32 accumulation。

backward 慢：
  block-level reduction；
  不训练 knots；
  fused coefficient update。

A4 不够：
  local cubic；
  增 knots 但 active basis 不变；
  small identity/quadratic residual，重新过 A1。

A5 fail：
  查 occupancy；
  查 RealSignalReservoirRatio；
  quantile grid init，不按 dataset 调参。
```

### 10.3 D2 Rational

当前状态：

```text
L3/A4 可打开；
B7kc L3 step = 1.0464；
A5 task blocked；
当前 blocker 是 task geometry/calibration，不是 grad correctness。
```

下一步不继续 CE-tune。候选分两类：

```text
RAT-A: 更强但仍 cheap 的 task-geometry primitive
  RAT-A1 group rational + identity residual
  RAT-A2 denominator-safe tangent residual
  RAT-A3 low-rank rational + bounded output geometry
  RAT-A4 rational + class-agnostic signal broadening

RAT-B: lower-level L3 kernel/timing repair
  RAT-B1 deeper fused Horner VJP
  RAT-B2 fused denominator safety and derivative
  RAT-B3 fused readblock + update
  RAT-B4 reduce temporary allocation
```

记录：

```text
den_min
den_p01
den_condition
r_prime_p95
r_double_prime_p95
group_function_diversity
group_dead_fraction
task_mean_delta
worst_delta
near_pass
AUC_step
AUC_time
ECE
LineC_CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
```

Rational-specific gate：

$$
den_{p01}>0.50
$$

$$
r'_{p95}<r'_{p95,max}
$$

$$
NoiseSignalLeak_{Rational}\le NoiseSignalLeak_{MLP}+0.02
$$

失败后 Codex 先试：

```text
A5 fail + LineC bad：
  design signal-channel rational geometry；不 CE-tune。

A5 fail + LineC ok + ECE bad：
  generic logit calibration through label-free scale/rank, not CE branch。

denominator unsafe：
  stronger safe-den parameterization；
  no task run until safety pass。

L3 close fail：
  reduce temp allocations；
  fused update；
  do not open A4/A5 if L3 gate fail.
```

### 10.4 D3 RBF / FastKAN

技术难点：

```text
exp cost；
active center selection；
dense centers materialization；
width/center trainability；
center occupancy collapse。
```

候选：

```text
RBF1 fixed-center FastKAN K4
RBF2 compact active-center Kactive=2
RBF3 compact active-center Kactive=4
RBF4 exp2 approximation diagnostic
RBF5 RBF-as-spline approximation
```

记录：

```text
K_total
K_active
exp_count_per_sample
active_center_entropy
dead_center_fraction
width_p01
width_p99
out_of_grid_fraction
A4
A5
LineC
```

失败后 Codex 先试：

```text
exp 慢：
  K_active 2/4；
  exp2 approximation；
  precompute center grid；
  no dense basis。

dead centers：
  train-stream quantile centers；
  fixed centers first。

AUC fail：
  check active_center_entropy and NoiseSignalLeak。
```

### 10.5 D4 Chebyshev

技术难点：

```text
recurrence 便宜，但 global support 可能把噪声送进 signal channel；
高阶 degree energy 可能造成 condition 与 tail risk。
```

候选：

```text
CHEB1 K3 fused recurrence
CHEB2 K4 fused recurrence
CHEB3 K6 diagnostic
CHEB4 K4 degree-damped
```

公式：

$$
T_0(x)=1
$$

$$
T_1(x)=x
$$

$$
T_{k+1}(x)=2xT_k(x)-T_{k-1}(x)
$$

记录：

```text
degree_energy_k
high_degree_energy_ratio
recurrence_max_abs
input_norm_range
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
ECE
```

失败后 Codex 先试：

```text
NoiseSignalLeak 高：
  degree damping；
  K 降到 3/4；
  signal-projected degree update。

condition 坏：
  tanh/robust normalization；
  recurrence clamp；
  do not learn high-degree scale early。
```

### 10.6 D5 Fourier

技术难点：

```text
sin/cos cost；
global support；
high-frequency noise leakage；
learnable frequency/phase backward 复杂。
```

候选：

```text
FOU1 K2 fixed lowfreq fused sincos
FOU2 K4 fixed lowfreq fused sincos
FOU3 K4 lowfreq-plus-linear
FOU4 late-enable high-frequency diagnostic
```

公式：

$$
\phi_{2k-1}(x)=\sin(k\omega x)
$$

$$
\phi_{2k}(x)=\cos(k\omega x)
$$

记录：

```text
frequency_count
high_freq_energy_ratio
spectral_entropy
phase_drift
sincos_time_ms
NoiseSignalLeak
CEp99
ECE
```

失败后 Codex 先试：

```text
sin/cos slow：
  K=2；
  fused sincos；
  fixed frequency only。

noise leak 高：
  high-frequency damping；
  late-enable；
  no learnable frequency until lowfreq pass。
```

### 10.7 D6 Wavelet

技术难点：

```text
Morlet/MexicanHat 可能太慢；
scale/shift backward 重；
local support coverage 不均；
tail local overfit。
```

候选：

```text
WAV1 triangle / hat wavelet local support
WAV2 Haar diagnostic
WAV3 compact MexicanHat diagnostic
WAV4 Morlet diagnostic only
```

优先用 cheap local wavelet：

$$
\psi(x)=\max(1-|x|,0)
$$

记录：

```text
active_support_count
scale_energy
scale_dead_fraction
local_tail_coverage
CEp99
margin_p10
NoiseSignalLeak
```

失败后 Codex 先试：

```text
Morlet 慢：
  回到 triangle/hat；
  复用 spline local kernel。

tail overfit：
  scale-energy balance；
  Line C noise leak audit。

coverage 不均：
  quantile scale init；
  no dataset branch。
```

## 10.8 Family status 输出

每个 family 必须写：

`v129_family_status.json`

```json
{
  "family": "...",
  "best_candidate": "...",
  "status": "FamilyPass | FamilyNearPass | KernelBlocked | ExpressionBlocked | TaskBlocked | GeometryBlocked | RejectedForThisVersion",
  "evidence": {
    "gradcheck": "...",
    "efficiency": "...",
    "expression": "...",
    "task": "...",
    "lineC": "..."
  },
  "next_action": "..."
}
```

---

# 11. 统一 artifact contract

必须输出：

```text
v129_route_decision.json
v129_provenance_audit.csv
v129_modification_audit.csv

v129_b314_contract.csv
v129_b314_reconfirm10.csv
v129_b314_wider_protocol.csv
v129_b314_component_ablation.csv
v129_b314_fullstep_profile.csv
v129_b314_task_trace.csv

v129_lineC_train_probe_coupling.csv
v129_lineC_signal_reservoir.csv
v129_lineC_summary.csv

v129_functional_metric_audit.csv
v129_functional_onestep.csv
v129_functional_fivestep.csv
v129_functional_short_run.csv
v129_functional_control_matrix.csv
v129_functional_route.json

v129_family_manifest.csv
v129_family_gradcheck.csv
v129_family_efficiency.csv
v129_family_expression.csv
v129_family_task_triage.csv
v129_family_lineC.csv
v129_family_functional_diagnostic.csv
v129_family_status.json
v129_failure_table.csv
```

---

# 12. 必须可视化

```text
fig_v129_base_reconfirm_acc_delta_ci.svg
fig_v129_base_auc_step_time_ci.svg
fig_v129_base_efficiency_memory_box.svg
fig_v129_b314_component_ablation_radar.svg
fig_v129_b314_protocol_heatmap.svg

fig_v129_lineC_coupling_by_candidate.svg
fig_v129_lineC_noise_leak_by_candidate.svg
fig_v129_lineC_real_signal_reservoir_by_candidate.svg
fig_v129_lineC_kernel_drift_vs_coupling.svg

fig_v129_functional_score_gap_vs_controls.svg
fig_v129_functional_task_nonharm.svg
fig_v129_functional_overhead_bar.svg
fig_v129_functional_short_run_paired_delta.svg

fig_v129_family_efficiency_pareto.svg
fig_v129_family_expression_radar.svg
fig_v129_family_task_gate_heatmap.svg
fig_v129_family_lineC_dashboard.svg
fig_v129_rational_taskblocked_autopsy.svg
```

---

# 13. 最终判断规则

## Case 1：B314 hardening pass + functional official pass

可以写：

```text
在 strict FC-PureKAN B314 base 上，functional update 产生 control-resistant 的几何/训练轨迹收益。
```

仍不能直接写：

```text
DG-KAN 全面超过 MLP。
```

还需要 external fair / broader tasks。

## Case 2：B314 hardening pass + functional official fail

结论：

```text
B314 是合格 efficient PureKAN base；
但当前 functional update 仍无独立 official advantage。
```

下一步：

```text
停止围绕 F7 小修；
回到 basis-aware functional primitive；
保留 B314 作为 architecture contribution。
```

## Case 3：B314 hardening fail

结论：

```text
B314 是 current protocol success，但不是 robust base。
```

下一步：

```text
先修 base；
functional official 暂停；
classic family 继续并行。
```

## Case 4：Classic family 有 FamilyPass

结论：

```text
该 family 可作为 B314 之外的 classic KAN base candidate。
```

下一步：

```text
与 B314 同协议比较；
做 Line C 和 functional diagnostic；
判断 family-specific geometry advantage。
```

## Case 5：Classic family 都未 pass，但状态明确

结论：

```text
v12.9 完成了 classic portfolio audit；
当前最强 base 仍为 B314/FHQ；
classic basis 的失败被定位为 Kernel / Expression / Task / Geometry 之一。
```

这不是失败，而是 portfolio closure。

---

# 14. 给 Codex 的执行摘要

Codex 下一步不要继续温度、epoch、optimizer 小修。执行下面四线：

```text
1. Line A:
   B314 exact 10-seed reconfirm + wider protocol + component autopsy。

2. Line C:
   对 B314、functional、Rational、classic family 全部跑 Manifold-Channel diagnostics。

3. Line B:
   固化 corrected functional metric；
   跑 one-step/five-step strong controls；
   只有 control-resistant 后才跑 short-run。

4. Line D:
   继续 classic family portfolio；
   每个 family 完成 L1/L2/L3、A4、A5、LineC、status；
   Rational 从 TaskBlocked 进入 task-geometry autopsy，不继续 CE-tune。
```

禁止：

```text
1. 把 corrected functional diagnostic 直接写成 official success；
2. 把 B314 过 gate 当成 classic basis 失败证明；
3. 按 dataset name 调参；
4. teacher / distillation / loss modification / sampler / class weight；
5. 使用 loss.backward 作为 official KAN path；
6. fake/proxy/CPU offload；
7. 只因为某个 family 未完成就写 family fail；
8. 继续 dense random pair trajectory；
9. 继续 CE-specific calibration。
```

---

## 15. 最终总结

v12.8.3 的真正意义是：

$$
\boxed{
\text{Base problem 第一次基本打开：B314 是当前合格 anchor。}
}
$$

但项目的核心目标还没有完成，因为：

$$
\boxed{
\text{Functional update 的独立 advantage 还没有 official，}
\text{经典基函数 portfolio 还没有 closure。}
}
$$

因此 v12.9 的正确方向不是重新找 base，也不是只修 functional，而是：

```text
B314 主线：
  确认、解释、加固。

Functional 主线：
  固化 corrected metric，强 controls 下 official re-entry。

经典 family 支线：
  family-specific high-efficiency kernel + A4/A5/LineC closure。

几何诊断线：
  用 train-probe coupling / signal-reservoir / noise leak 解释成功和失败。
```

一句话：

$$
\boxed{
\text{我们现在终于有了一个能承载 functional update 验证的 PureKAN base；}
\text{下一步必须证明 functional 是否真的有独立价值，}
\text{同时不能放弃经典基函数的高效率化 portfolio。}
}
$$
