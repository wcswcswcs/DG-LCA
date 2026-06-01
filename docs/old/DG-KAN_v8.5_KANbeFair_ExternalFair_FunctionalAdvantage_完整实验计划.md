# DG-KAN v8.5 KANbeFair 外部公平评估与 Functional Advantage Formalization 完整实验计划

> 本计划基于 v8.4 真实复盘、`run_gafu_v83_real.py` 的 functional update 路线、以及论文 **KAN or MLP: A Fairer Comparison** 制定。  
> 本轮核心变化是：不再只在自建 MNIST/Fashion-MNIST/KMNIST 小框架里判断成功，而是把 `./third_party/` 下的 KANbeFair 评估框架纳入正式路线。  
> v8.5 的目标不是继续做小修补，而是把当前 DG-KAN 的 high-rep minimum success 放到外部公平基准下重新审计，回答一个更硬的问题：  
> **在参数量、FLOPs、wall-clock、同 CE 监督、无 teacher、无改 loss 条件下，DG-KAN + functional update 是否仍然能形成相对 MLP 的真实优势？**

---

## 0. 当前状态

### 0.1 v8.4 达到了什么

v8.4 已经完成 **high-rep timing-stabilized minimum success**，但没有完成 formal / strong success。当前结果必须这样写：

```text
official-style 50 warmup / 200 reps fresh reproduction:
  H0BasePass = 0
  success_v84_minimum = 0

official-style 50 warmup / 200 reps repeat reproduction:
  H0BasePass = 0
  success_v84_minimum = 0

high-rep timing-stabilized diagnostic:
  H0BasePass = 1
  P7Confirm10Pass = 1
  P8TimePass = 1
  P9ScalingRobustnessPass = 1
  FunctionalCausalityPass = 1
  RoleMechanismPass = 1
  FormalProfilerPass = 1
  FormalTimeAccountingPass = 0
  S1Pass = 0
  NearS1Pass = 0
  success_v84_minimum = 1
  success_v84_formal_time_profiler = 0
  success_v84_strong = 0
```

也就是说，当前可以说：

$$
\boxed{
\text{v8.4 在 high-rep timing-stabilized protocol 下完成 minimum success。}
}
$$

但不能说：

$$
\boxed{
\text{v8.4 已经完成 formal / strong success。}
}
$$

更不能说：

$$
\boxed{
\text{已经在外部公平评估框架下证明 KAN 全面优于 MLP。}
}
$$

### 0.2 现在最准确的问题定位

v8.4 之前的核心 blocker 是：

```text
functional route 是否能在 system-gated base 上真正进入 full task？
```

v8.4 之后，这个问题已经得到阶段性正面回答：

```text
KW6 hidden68 + FT7 accepted route
在 high-rep timing protocol 下可以复现到 P9；
FT7 有几何收益；
FT7 causality controls 初步通过；
P9 scaling / robustness 有正信号。
```

但新的 blocker 是：

```text
1. official 50/200 timing protocol 下 H0 step gate 不稳；
2. formal time accounting 仍未完成；
3. guard/stride ablation 没有证明 accepted setting 绝对支配；
4. S1 未完成；
5. 所有结论仍主要来自自建 MNIST-family 小框架；
6. 尚未在 KANbeFair 的参数/FLOPs公平评估体系中验证。
```

因此 v8.5 必须从 “内部成功” 进入 “外部公平基准审计”。

---

## 1. KANbeFair 对当前项目的意义

### 1.1 为什么必须接入 KANbeFair

论文 **KAN or MLP: A Fairer Comparison** 的核心结论是：在控制参数量或 FLOPs 的公平设置下，KAN 通常只在 symbolic formula representation 上优于 MLP，而在 machine learning、computer vision、NLP、audio 等任务上多数情况下弱于 MLP。论文还明确提出，公平比较必须控制参数量或 FLOPs，而不能只比较任意宽度/任意计算预算下的结果。

这对 DG-KAN 是一个很强的外部压力测试。我们之前的结论主要围绕：

```text
MNIST / Fashion-MNIST / KMNIST
固定 train/val/test 小样本
内部 MLP-AdamW baseline
内部 memory/step gate
functional geometry metrics
```

而 KANbeFair 提醒我们：如果不控制参数/FLOPs，KAN 的优势很容易被质疑为：

```text
参数量更多；
FLOPs 更多；
训练预算更多；
任务选择更偏向 KAN；
评估框架不公平。
```

所以 v8.5 必须把 DG-KAN 放到 KANbeFair 的公平比较范式下，至少同时回答：

```text
1. 参数量相同，DG-KAN 是否仍优于 MLP？
2. FLOPs 相同，DG-KAN 是否仍优于 MLP？
3. wall-clock / memory 近似相同，DG-KAN 是否仍优于 MLP？
4. functional update 的几何优势是否在 KANbeFair 任务上可见？
5. 如果 DG-KAN 只在 symbolic 或少数小视觉任务上有效，边界在哪里？
```

### 1.2 本轮不改变的原则

v8.5 继续严格禁止：

```text
external teacher
self teacher
distillation
self-distillation
label smoothing
focal loss
margin loss
calibration loss
NLL-balanced loss
geometry loss as training objective
sampler / class weight
oversampling / undersampling
test-based selection
CPU offload
optimizer hyperparameter sweep
```

v8.5 允许 functional update，但必须继续是 update rule，而不是 loss：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

functional update 允许形式：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\lambda_t\Delta\theta_{\text{functional}},
$$

但不允许：

$$
L = CE + \lambda L_{\text{geo}}.
$$

---

## 2. v8.5 总体目标

v8.5 的总体目标是：

$$
\boxed{
\text{在 KANbeFair 外部公平评估框架下，验证 DG-KAN functional route 是否仍有 task / geometry / system 优势。}
}
$$

v8.5 要同时完成四件事。

### 2.1 复现性目标

在 fresh run 中复现 v8.4 high-rep survivor：

```text
KW6 hidden68 base
FT7 accepted functional update
high-rep timing protocol
P7/P8/P9 chain
```

要求：

$$
H0BasePass=1,
$$

$$
P7Confirm10Pass=1,
$$

$$
P8TimePass=1,
$$

$$
P9ScalingRobustnessPass=1.
$$

### 2.2 外部框架接入目标

把 `./third_party/KANbeFair` 作为外部评估框架接入，完成：

```text
source audit
baseline reproduction
dataset adapter
model adapter
metric adapter
parameter counter
FLOPs counter
wall-clock profiler adapter
```

并保证：

```text
DG-KAN candidate 与 KANbeFair MLP/KAN baseline 在同一 train/eval protocol 下运行。
```

### 2.3 公平比较目标

在 KANbeFair protocol 下至少完成三种公平比较：

```text
parameter-matched
FLOPs-matched
wall-clock/memory-aware
```

最小成功标准：

$$
Acc_{\text{DG-KAN}}\geq Acc_{\text{MLP}}
$$

在至少 primary MNIST-family 任务的 macro mean 上成立，并且：

$$
FLOPs_{\text{DG-KAN}}\leq1.05FLOPs_{\text{MLP}},
$$

或：

$$
Params_{\text{DG-KAN}}\leq1.05Params_{\text{MLP}}.
$$

强成功标准：

$$
Acc_{\text{DG-KAN}}\geq Acc_{\text{MLP}}
$$

在至少两个任务族上成立，并且：

$$
ECE_{\text{DG-KAN}}\leq ECE_{\text{MLP}},
$$

$$
NLL_{\text{DG-KAN}}\leq NLL_{\text{MLP}},
$$

$$
TimeAUC_{\text{DG-KAN}}\leq1.05TimeAUC_{\text{MLP}}.
$$

### 2.4 functional geometry 目标

证明 functional update 的价值不是只在内部小数据上成立，而是在 KANbeFair 框架下仍能提供几何优势。

最小 functional success：

$$
Acc_{\text{functional}}\geq Acc_{\text{base}}-0.005,
$$

$$
R_{\text{curv,functional}}\leq0.90R_{\text{curv,base}},
$$

$$
r_{\text{mem,functional}}\leq1.05,
$$

$$
r_{\text{step,functional}}\leq1.50.
$$

强 functional success：

$$
Acc_{\text{functional}}\geq Acc_{\text{base}},
$$

$$
R_{\text{curv,functional}}\leq0.80R_{\text{curv,base}},
$$

并且至少满足：

$$
RobustnessAUC_{\text{functional}}\geq RobustnessAUC_{\text{base}},
$$

或：

$$
ECE_{\text{functional}}\leq ECE_{\text{base}}.
$$

---

## 3. 核心假设

### H0：当前 v8.4 high-rep minimum 可以独立复现

H0 是所有外部评估之前的前置条件。如果 v8.4 high-rep survivor 不能复现，则不应把它接入 KANbeFair 做更大声明。

H0 成立标准：

fresh run 中：

$$
\Delta Acc_{\text{macro,val,KW6}}\geq0.0200,
$$

$$
r_{\text{mem,max,KW6}}\leq1.05,
$$

$$
r_{\text{step,max,KW6}}\leq1.50,
$$

and:

$$
Acc_{\text{FT7}}\geq Acc_{\text{KW6}}-0.005,
$$

$$
R_{\text{curv,FT7}}\leq0.90R_{\text{curv,KW6}},
$$

$$
TimeAUC_{\text{FT7}}\leq1.05TimeAUC_{\text{KW6}}.
$$

如果 H0 失败，则 v8.5 只做 KANbeFair baseline audit，不做 DG-KAN superiority claim。

---

### H1：KANbeFair baseline 能被复现

H1 不是模型假设，而是外部框架可信性假设。只有先复现 KANbeFair 的 MLP/KAN baseline，我们才有资格在该框架下加入 DG-KAN。

H1 成立标准：

对 KANbeFair 中至少一个 vision subset 和一个 symbolic subset，复现实验结果在论文报告区间附近：

$$
|Acc_{\text{repro}}-Acc_{\text{reported}}|\leq\epsilon_{\text{acc}},
$$

其中：

$$
\epsilon_{\text{acc}}=0.02
$$

for classification tasks。

对 symbolic RMSE：

$$
\frac{|RMSE_{\text{repro}}-RMSE_{\text{reported}}|}{RMSE_{\text{reported}}}\leq0.20.
$$

如果 H1 不成立，则必须先修复第三方框架接入、数据版本、预处理、seed、训练 epoch、模型构造。

---

### H2：DG-KAN 的内部优势能在参数匹配下保持

H2 检查是否只是因为 DG-KAN 事实上用了更多参数。

成立标准：

在 parameter-matched envelope 下：

$$
Params_{\text{DG-KAN}}\leq1.05Params_{\text{MLP}},
$$

且：

$$
Acc_{\text{DG-KAN}}\geq Acc_{\text{MLP}}
$$

on primary macro mean。

如果 DG-KAN 只在参数更多时有效，route 降级为：

```text
R4-CapacityDrivenInternalAdvantage
```

不能写成 fair architecture advantage。

---

### H3：DG-KAN 的优势能在 FLOPs 匹配下保持

H3 是 KANbeFair 最关键的压力测试。论文指出 KAN 的 spline function FLOPs 高，部分参数匹配优势在 FLOPs 匹配下会消失。

成立标准：

$$
FLOPs_{\text{DG-KAN}}\leq1.05FLOPs_{\text{MLP}},
$$

且：

$$
Acc_{\text{DG-KAN}}\geq Acc_{\text{MLP}}
$$

on primary macro mean。

如果参数匹配过但 FLOPs 匹配不过，则 route 写为：

```text
R5-ParameterFairButFLOPsFail
```

不能写成完整公平优势。

---

### H4：functional update 的几何优势不是内部数据集 artifact

H4 检查 FT7 在 KANbeFair task 上是否仍能改善几何。

成立标准：

在 KANbeFair primary subset 上：

$$
R_{\text{curv,FT7}}\leq0.90R_{\text{base}},
$$

and:

$$
Acc_{\text{FT7}}\geq Acc_{\text{base}}-0.005.
$$

如果 geometry 改善只在 MNIST-family 内部 split 出现，而在 KANbeFair 任务上消失，则 route 降级为：

```text
R6-InternalGeometryOnly
```

---

### H5：如果 KANbeFair 显示 MLP 仍全面优于 DG-KAN，则当前路线边界要被承认

H5 是 falsification hypothesis。v8.5 必须允许 negative result。

如果在 KANbeFair framework 中：

$$
Acc_{\text{DG-KAN}}<Acc_{\text{MLP}}-0.01
$$

on most non-symbolic tasks under parameter and FLOPs matching，则必须写：

```text
DG-KAN current advantage is not externally general under KANbeFair.
```

这不是失败，而是边界明确化。此时 functional update 的价值只能保留为：

```text
geometry / symbolic / robustness / interpretability advantage
```

而不是广义 task superiority。

---

### H6：symbolic formula representation 是 functional update 最可能放大的方向

KANbeFair 认为 KAN 在 symbolic formula representation 上更有优势，并且指出 B-spline activation 是其核心来源。functional update 的几何目标也更贴近 symbolic / smooth function approximation。

H6 成立标准：

在 KANbeFair symbolic tasks 上：

$$
RMSE_{\text{DG-KAN+FT7}}\leq RMSE_{\text{KANbeFair-KAN}},
$$

或：

$$
RMSE_{\text{DG-KAN+FT7}}\leq RMSE_{\text{MLP}}.
$$

同时几何指标改善：

$$
R_{\text{curv,FT7}}\leq0.90R_{\text{base}}.
$$

如果 symbolic 也没有收益，则 functional update 的几何 metric 需要重新设计。

---

## 4. Candidate 设计

### 4.1 Baselines

```text
KB-MLP:
  KANbeFair MLP baseline

KB-KAN:
  KANbeFair official KAN baseline

KB-BSpline-MLP:
  KANbeFair B-spline MLP / spline activation ablation, if available

DG-Base:
  KW6 hidden68 base candidate

DG-Functional:
  KW6 hidden68 + FT7 accepted functional update

DG-NoOp:
  matched-overhead no-op control

DG-RandomFunc:
  random functional direction control
```

### 4.2 DG-KAN Adapter candidates

```text
DG0-KW6-hidden68-base
DG1-FT7-accepted-functional
DG2-FT7-noop-matched-overhead
DG3-FT7-random-functional-direction
DG4-FT7-role-head-only
DG5-FT7-role-stack-only
```

### 4.3 Fairness envelopes

每个 task 至少跑三条 envelope：

```text
E0-default:
  KANbeFair original setting

E1-parameter-matched:
  Params(DG-KAN) within 5% of MLP

E2-FLOPs-matched:
  FLOPs(DG-KAN) within 5% of MLP

E3-time-budget-matched:
  train wall-clock budget matched

E4-memory-aware:
  peak memory <= 1.05 x MLP
```

---

## 5. 实验阶段总览

v8.5 分为九个 Wave：

```text
Wave 0:
  source audit and third-party framework contract

Wave 1:
  reproduce KANbeFair baselines

Wave 2:
  DG-KAN adapter and fairness counters

Wave 3:
  internal survivor reproduction under v8.4 protocol

Wave 4:
  KANbeFair primary task transfer

Wave 5:
  parameter / FLOPs / wall-clock fair envelope

Wave 6:
  functional geometry causality under KANbeFair

Wave 7:
  symbolic and continual-learning stress tests

Wave 8:
  final route decision and artifact audit
```

---

## 6. Wave 0：source audit and third-party framework contract

### P0：third_party source audit

#### 目标

确认 `./third_party/KANbeFair` 的代码结构、任务、数据集、baseline、参数/FLOPs计数方式、训练协议。不得假设其 API；必须从源码落盘审计。

#### 必须记录

```text
third_party_path
git_commit_or_hash
repo_url_if_available
license
python_files_count
dataset_modules
model_modules
baseline_model_classes
KAN_model_classes
MLP_model_classes
training_entrypoints
evaluation_entrypoints
parameter_counter_found
FLOPs_counter_found
result_parser_found
```

#### 判断标准

SourceAuditPass：

```text
third_party_path exists
training_entrypoint found
MLP baseline found
KAN baseline found
at least one dataset task runnable
parameter/FLOPs counter either found or implementable
```

如果没有找到源码或入口，必须停止并先修接入，不允许臆造结果。

#### 可视化

```text
p0_third_party_source_tree.md
p0_framework_dependency_graph.svg
p0_entrypoint_matrix.md
```

---

## 7. Wave 1：reproduce KANbeFair baselines

### P1：baseline reproduction

#### 目标

复现 KANbeFair 自身 MLP/KAN baseline，至少覆盖：

```text
MNIST
Fashion-MNIST
KMNIST
one symbolic task
one tabular task if available
```

#### 必须记录

```text
task_name
dataset_name
split_protocol
model_name
params
FLOPs
epochs
batch_size
optimizer
lr
weight_decay
seed
val_metric
test_metric
reported_metric_from_paper
absolute_delta_from_reported
relative_delta_from_reported
```

#### 判断标准

Classification reproduction pass：

$$
|Acc_{\text{repro}}-Acc_{\text{paper}}|\leq0.02.
$$

Symbolic reproduction pass：

$$
\frac{|RMSE_{\text{repro}}-RMSE_{\text{paper}}|}
{RMSE_{\text{paper}}}\leq0.20.
$$

#### 可视化

```text
p1_reproduction_vs_paper_bar.svg
p1_reproduction_delta_table.md
p1_acc_vs_params_overlay.svg
p1_acc_vs_flops_overlay.svg
```

---

## 8. Wave 2：DG-KAN adapter and fairness counters

### P2：model adapter implementation

#### 目标

实现 `DGKANAdapter`，让 DG-KAN candidate 能在 KANbeFair 任务中训练/评估，同时保持 DG-KAN contract。

#### 必须记录

```text
candidate_id
adapter_class
input_adapter
output_adapter
loss_type
functional_update_used
external_teacher_used
self_teacher_used
geometry_loss_used
manual_forward
manual_backward
manual_update
uses_loss_backward
nonKAN_param_count
fake_data_used
proxy_row_used
```

#### 判断标准

AdapterPass：

```text
loss_type = CE or RMSE for symbolic
geometry_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
manual_forward = 1
manual_backward = 1
manual_update = 1
uses_loss_backward = 0
fake/proxy = 0
```

### P3：params / FLOPs / memory counter

#### 目标

实现和 KANbeFair 可比的参数量、FLOPs、memory、wall-clock计数。

#### 必须记录

```text
model_name
params_total
params_trainable
params_KAN_edge
params_nonKAN
FLOPs_formula
FLOPs_measured_or_estimated
FLOPs_forward
FLOPs_backward_if_available
peak_memory_MB
step_time_ms
wall_clock_per_epoch
```

#### 判断标准

CounterPass：

```text
params_total available for all models
FLOPs available for all models
FLOPs formula documented
memory/time measured with same protocol
```

#### 可视化

```text
p3_params_vs_accuracy.svg
p3_flops_vs_accuracy.svg
p3_memory_time_pareto.svg
```

---

## 9. Wave 3：internal survivor reproduction under v8.4 protocol

### P4：v8.4 survivor reproduction

#### 目标

在进入外部评估前，fresh run 复现 v8.4 high-rep survivor。

#### 必跑对象

```text
B0
KW6 hidden68 base
FT7 accepted
no-op matched overhead
random functional direction
```

#### 必须记录

```text
macro_gap
CI95_low
Holm_p
test_gap
curvature_ratio
memory_ratio_max
step_ratio_max
ValLossAUC_time_ratio
functional_update_time_ratio
unknown_time_fraction
FunctionalCausalityPass
P9ScalingRobustnessPass
```

#### 判断标准

InternalReproPass：

```text
H0BasePass = 1
FT7 reproduction pass = 1
FunctionalCausalityPass = 1
P8TimePass = 1
```

如果不通过，外部 KANbeFair claim 不打开。

---

## 10. Wave 4：KANbeFair primary task transfer

### P5：MNIST-family transfer

#### 目标

先在 KANbeFair 的 MNIST / Fashion-MNIST / KMNIST protocol 下测试 DG-KAN 是否保留优势。

#### 必跑对象

```text
KB-MLP
KB-KAN
DG-Base
DG-Functional
DG-NoOp
DG-RandomFunc
```

#### 必须记录

```text
dataset
seed
model
params
FLOPs
train_time
peak_memory
val_acc
test_acc
val_loss
test_loss
ECE
NLL
curvature
jacobian_norm
local_lipschitz
functional_update_time
```

#### 判断标准

Primary transfer pass：

$$
Acc_{\text{DG-Functional}}\geq Acc_{\text{KB-MLP}},
$$

and:

$$
Acc_{\text{DG-Functional}}\geq Acc_{\text{DG-Base}}-0.005.
$$

Functional geometry pass：

$$
R_{\text{curv,DG-Functional}}\leq0.90R_{\text{curv,DG-Base}}.
$$

#### 可视化

```text
p5_mnist_family_acc_bar.svg
p5_mnist_family_params_flops_pareto.svg
p5_functional_geometry_delta.svg
p5_datasetwise_win_loss_heatmap.svg
```

---

## 11. Wave 5：parameter / FLOPs / wall-clock fair envelope

### P6：parameter-matched envelope

#### 目标

确认 DG-KAN 不是靠更多参数赢。

#### 设置

对每个 task 建立模型族：

```text
MLP widths:
  KANbeFair default grid

KAN widths / basis:
  adjusted to match params

DG-KAN hidden/basis:
  adjusted to within 5% params
```

#### 判断标准

ParameterFairPass：

$$
Params_{\text{DG-KAN}}\leq1.05Params_{\text{MLP}},
$$

and:

$$
Acc_{\text{DG-KAN}}\geq Acc_{\text{MLP}}.
$$

#### 可视化

```text
p6_acc_vs_params_envelope.svg
p6_param_matched_table.md
```

### P7：FLOPs-matched envelope

#### 目标

确认 DG-KAN 不是靠更多 FLOPs 赢。

#### 判断标准

FLOPsFairPass：

$$
FLOPs_{\text{DG-KAN}}\leq1.05FLOPs_{\text{MLP}},
$$

and:

$$
Acc_{\text{DG-KAN}}\geq Acc_{\text{MLP}}.
$$

#### 可视化

```text
p7_acc_vs_flops_envelope.svg
p7_flops_matched_table.md
p7_taskwise_flops_gap_heatmap.svg
```

### P8：wall-clock / memory-aware envelope

#### 目标

将自建 S2 gate 与 KANbeFair wall-clock protocol 对齐。

#### 判断标准

SystemFairPass：

$$
PeakMemory_{\text{DG-KAN}}\leq1.05PeakMemory_{\text{MLP}},
$$

$$
TrainStepTime_{\text{DG-KAN}}\leq1.50TrainStepTime_{\text{MLP}},
$$

$$
TimeAUC_{\text{DG-KAN}}\leq1.05TimeAUC_{\text{MLP}}.
$$

#### 可视化

```text
p8_wallclock_accuracy_pareto.svg
p8_time_auc_table.md
p8_memory_accuracy_pareto.svg
```

---

## 12. Wave 6：functional geometry causality under KANbeFair

### P9：functional causality controls

#### 目标

在 KANbeFair tasks 上验证 FT7 几何收益不是 overhead/no-op/random artifact。

#### 必跑对象

```text
DG-Base
DG-Functional
DG-NoOp
DG-RandomFunc
DG-ShuffledRoleFunc
```

#### 必须记录

```text
task
seed
acc_delta_vs_base
ECE_delta_vs_base
NLL_delta_vs_base
curvature_ratio
jacobian_ratio
local_lipschitz_ratio
functional_update_time_ratio
bad_step_rate
holdout_descent_ratio
role_accept_rate
role_geometry_delta
```

#### 判断标准

CausalityPass：

$$
R_{\text{curv,DG-Functional}}<R_{\text{curv,DG-NoOp}},
$$

$$
R_{\text{curv,DG-Functional}}<R_{\text{curv,DG-RandomFunc}},
$$

and:

$$
Acc_{\text{DG-Functional}}\geq Acc_{\text{DG-NoOp}}-0.002.
$$

#### 可视化

```text
p9_causality_curvature_bar.svg
p9_functional_vs_noop_pareto.svg
p9_role_contribution_waterfall.svg
```

---

## 13. Wave 7：symbolic and continual-learning stress tests

### P10：symbolic function representation

#### 目标

验证 functional geometry update 是否在 KANbeFair 最支持 KAN 的任务族上更有意义。

#### 必须记录

```text
function_name
model
params
FLOPs
RMSE
MAE
curvature
slope_p95
jacobian_norm
train_time
memory
```

#### 判断标准

SymbolicPass：

$$
RMSE_{\text{DG-Functional}}\leq RMSE_{\text{KB-KAN}},
$$

or:

$$
RMSE_{\text{DG-Functional}}\leq RMSE_{\text{KB-MLP}}.
$$

Functional geometry pass：

$$
R_{\text{curv,DG-Functional}}\leq0.90R_{\text{DG-Base}}.
$$

#### 可视化

```text
p10_symbolic_rmse_vs_params.svg
p10_symbolic_rmse_vs_flops.svg
p10_symbolic_geometry_pareto.svg
```

### P11：continual learning stress

#### 目标

KANbeFair 认为 KAN 在标准 class-incremental continual learning 中遗忘更严重。v8.5 要检查 functional update 是否能缓解遗忘。

#### 设置

```text
tasks:
  digits 0-2
  digits 3-5
  digits 6-9

models:
  KB-MLP
  KB-KAN
  DG-Base
  DG-Functional
```

#### 必须记录

```text
task_id
acc_after_task
backward_transfer
forgetting_score
final_average_acc
curvature_after_task
functional_update_time
memory
```

#### 判断标准

ContinualPass：

$$
Forgetting_{\text{DG-Functional}}\leq Forgetting_{\text{KB-KAN}},
$$

and:

$$
FinalAcc_{\text{DG-Functional}}\geq FinalAcc_{\text{KB-KAN}}.
$$

Strong continual pass：

$$
Forgetting_{\text{DG-Functional}}\leq Forgetting_{\text{KB-MLP}}.
$$

#### 可视化

```text
p11_continual_acc_matrix.svg
p11_forgetting_bar.svg
p11_curvature_forgetting_scatter.svg
```

---

## 14. Wave 8：final route decision and artifact audit

### P12：route decision

v8.5 route 必须明确区分：

```text
internal success
KANbeFair parameter-fair success
KANbeFair FLOPs-fair success
wall-clock success
functional geometry success
symbolic-only success
negative external result
```

#### route cases

```text
R1-ExternalFairFunctionalStrongSuccess:
  参数、FLOPs、wall-clock、geometry、robustness 均通过。

R2-ExternalFairTaskSuccessButSystemOpen:
  参数/FLOPs公平任务通过，但 wall-clock 或 memory 未闭合。

R3-FunctionalGeometrySuccessOnly:
  task 不赢 MLP，但 geometry / robustness / symbolic 有真实优势。

R4-InternalOnlySuccess:
  内部 v8.4 success 复现，但 KANbeFair 任务不通过。

R5-ParameterFairButFLOPsFail:
  参数公平过，FLOPs公平不过。

R6-SymbolicOnlyKANAdvantage:
  只在 symbolic 任务上成立。

R7-NegativeExternalResult:
  KANbeFair 下 DG-KAN 仍弱于 MLP，多数非 symbolic task 不通过。

R8-ReproductionFail:
  v8.4 survivor 或 KANbeFair baseline 不能复现。

R9-CodeIntegrationFail:
  third_party adapter / counters / contract 不完整。
```

#### route_decision.json 必须记录

```text
route
internal_reproduction_pass
kanbefair_baseline_reproduction_pass
parameter_fair_pass
flops_fair_pass
wallclock_fair_pass
functional_causality_pass
symbolic_pass
continual_pass
geometry_pass
robustness_pass
best_candidate
best_task_family
primary_blocker
next_required_implementation
```

---

## 15. Required artifacts

v8.5 必须落盘：

```text
run_manifest.json
third_party_source_audit.csv
kanbefair_reproduction.csv
dgkan_adapter_contract.csv
params_flops_counter_audit.csv
v84_survivor_reproduction.csv
kanbefair_primary_transfer.csv
parameter_matched_envelope.csv
flops_matched_envelope.csv
wallclock_memory_envelope.csv
functional_causality_kanbefair.csv
symbolic_representation.csv
continual_learning_stress.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

failure taxonomy：

```text
F1_third_party_source_missing
F2_kanbefair_baseline_reproduction_fail
F3_adapter_contract_fail
F4_counter_mismatch
F5_internal_survivor_reproduction_fail
F6_parameter_fair_fail
F7_flops_fair_fail
F8_wallclock_fail
F9_functional_causality_fail
F10_symbolic_fail
F11_continual_fail
F12_teacher_or_loss_violation
F13_fake_or_proxy_violation
F14_artifact_missing
```

---

## 16. 第一轮执行顺序

### Step 1：source audit

先运行 `P0`。如果 `./third_party/KANbeFair` 不可读或入口不清楚，停止；不要写任何外部结论。

### Step 2：KANbeFair baseline reproduction

运行 `P1`。只要 baseline 复现不过，就先修数据、seed、训练 epoch、模型宽度、FLOPs counter。

### Step 3：DG-KAN adapter

运行 `P2/P3`。确认 DG-KAN 接入 KANbeFair 后仍然满足 no-teacher/no-loss/manual update contract。

### Step 4：v8.4 survivor reproduction

运行 `P4`。如果 KW6+FT7 high-rep survivor 无法 fresh reproduce，外部评估只做 exploratory，不做 success claim。

### Step 5：primary transfer

运行 `P5`。先在 MNIST-family 验证内外框架差异。

### Step 6：fair envelope

运行 `P6/P7/P8`。这是本轮最核心的外部公平检验。

### Step 7：functional causality on external tasks

运行 `P9`。证明 functional update 不是内部 artifact。

### Step 8：symbolic / continual stress

运行 `P10/P11`。如果非 symbolic 任务不强，至少验证 functional/KAN 在 symbolic 或几何任务中的真实边界。

### Step 9：final route

运行 `P12`，明确写成功、边界或 negative result。

---

## 17. 停止条件

### 17.1 成功停止

External fair strong success：

```text
KANbeFair baseline reproduced
DG-KAN adapter contract pass
parameter-fair pass
FLOPs-fair pass
functional causality pass
wall-clock system pass
```

Formal external success：

```text
parameter-fair or FLOPs-fair pass
functional geometry pass
no teacher/no loss contract pass
```

### 17.2 失败停止

```text
1. third_party source audit fail；
2. KANbeFair MLP/KAN baseline cannot reproduce；
3. DG-KAN adapter uses teacher/loss/autograd backward；
4. params/FLOPs counter cannot be aligned；
5. v8.4 survivor cannot reproduce；
6. DG-KAN loses to MLP by > 0.01 on most non-symbolic tasks；
7. functional update geometry gain disappears in external tasks；
8. wall-clock overhead > 1.50 x MLP；
9. no-fake/no-proxy audit fail。
```

---

## 18. 最终解释规则

### Case A：DG-KAN beats MLP under params and FLOPs on KANbeFair

这是最强结果。可以写：

```text
DG-KAN provides an externally fair task advantage over MLP on the tested task family.
```

### Case B：DG-KAN only wins under params but not FLOPs

只能写：

```text
DG-KAN shows parameter-efficient advantage, but not FLOPs-efficient advantage.
```

### Case C：DG-KAN only improves geometry / robustness

写：

```text
Functional update gives geometry or robustness advantage, but does not establish broad task superiority.
```

### Case D：DG-KAN only wins symbolic tasks

写：

```text
DG-KAN advantage is aligned with symbolic / smooth function representation, consistent with KANbeFair's broader conclusion.
```

### Case E：DG-KAN loses to MLP in most KANbeFair tasks

必须写：

```text
Current DG-KAN internal success does not externally generalize under KANbeFair fair comparison.
```

这不是失败的借口，而是项目边界的确定。

---

## 19. 最终建议

v8.5 的一句话策略是：

$$
\boxed{
\text{用 KANbeFair 外部框架审计 DG-KAN 的 functional advantage，到底是普适优势、任务族优势，还是内部 protocol 优势。}
}
$$

本轮不该继续做：

```text
teacher
self-teacher
loss modification
optimizer sweep
只在 MNIST-family 内继续阈值追逐
```

本轮应该做：

```text
1. KANbeFair baseline reproduction；
2. DG-KAN adapter；
3. 参数 / FLOPs / wall-clock 公平 envelope；
4. FT7 functional causality external check；
5. symbolic / continual stress；
6. 明确 route 边界。
```

最终目标不是强行证明 KAN 全面优于 MLP，而是给出可信结论：

$$
\boxed{
\text{DG-KAN + functional update 在哪些公平条件和任务族上真正优于 MLP，在哪些条件下不优。}
}
