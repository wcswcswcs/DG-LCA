# DG-KAN v9.0 代码核心化整改与 Clean PureKAN Functional Training System 完整计划

> 本计划的目标不是继续在旧 runner 上打补丁，也不是继续找某个 seed、hidden、stride、alpha 的可过线组合。  
> 本计划要完成两件事：  
> **第一，把当前分散在 `run_gafu_v7x/v8x` 和 `dgkan_core.py` 中的实现整理成一个可审计、可复用、可扩展的核心系统。**  
> **第二，在这个核心系统里实现一个完全干净的终极 PureKAN functional training system，并重新做与 MLP / KANbeFair-KAN 的外部公平比较。**  
> 本计划继续严格坚持：**no teacher、no self-teacher、no distillation、no loss modification、no label smoothing、no sampler/class weight、no CPU offload、no fake/proxy、no PyTorch loss.backward graph for KAN path**。

---

## 0. 当前代码状态的独立判断

### 0.1 当前已经有很强的阶段性结果

当前 v8.7 formal selected route 已经证明：在 KANbeFair vision-family 的 FMNIST / KMNIST 任务上，DG-KAN 路线可以在 params / FLOPs / timing / geometry 维度内形成外部公平优势。最终 artifact 中记录：

```text
success_v87_stability = true
success_v87_adaptive = true
success_v87_no_manual_tuning = true
success_v87_external_fair = true
success_v87_formal = true
success_v87_broad_strong = not_claimed
```

其中关键结果包括：

```text
P4 no-manual selection:
  selected base = KW4 hidden28
  params ratio = 0.939568
  FLOPs ratio = 0.936347

P4 selected confirmation:
  DG functional delta vs KB-MLP = +0.279999
  curvature ratio = 0.772483

P5 timing:
  q90 step ratio = 0.933570
  max step ratio = 0.957862
  unknown fraction max = 0.066499

P7 FMNIST:
  DG functional acc = 88.279998
  KB-MLP acc = 86.869997
  delta = +1.410002

P7 KMNIST:
  DG functional acc = 85.929996
  KB-MLP acc = 83.749998
  delta = +2.179998
```

这说明当前路线不是空想，task、geometry、system 和 external-fair 都已经有真实成功点。

### 0.2 但当前代码还不能支撑终极结论

当前代码的问题不是“没有效果”，而是“实现和实验系统还不够干净”。我把问题分为四类。

第一，**代码组织过于分散**。当前核心逻辑散落在：

```text
dgkan_core.py
run_gafu_v80_real.py
run_gafu_v83_real.py
run_gafu_v85_real.py
run_gafu_v86_real.py
run_gafu_v87_real.py
以及更早的 v71/v72/v73/v76/v79 runner
```

大量候选通过 runner 之间的 monkey patch、candidate registry override、继承旧函数来实现。这样会造成两个问题：一是实验结果很难复现到一个干净的核心系统；二是默认值容易漂移，例如 loss、label smoothing、candidate eligibility、manual contract 这些关键约束可能被旧版本默认值污染。

第二，**当前 selected route 还不是终极 full-edge PureKAN**。v8.7 选中的 `KW4 hidden28` 从代码结构看是：

```text
packed linear-SiLU stack
+
poly2_silu_base KAN-style head
+
adaptive functional update
```

这条路线非常有价值，但如果终极目标是“所有可学习函数都属于 KAN edge system”，那当前 linear-SiLU stack 还不够纯。它更接近一个 graph-free manual linear-SiLU stack 加 KAN-style head，而不是每一层都是 learnable edge function 的 full PureKAN。

第三，**当前 route 必须重新清洗 loss contract**。官方口径应该是：

$$
L_{\text{task}} = CE(y,p_\theta(x)).
$$

不允许：

$$
L = CE_{\text{smooth}}(y,p_\theta(x)).
$$

也不允许：

$$
L = CE + \lambda L_{\text{geo}}.
$$

当前代码中存在 `label_smoothing` 和 `_weighted_smooth_ce_and_grad` 路径。即使结果报告中写 `loss_type=CE`，整改后也必须把 official candidate 的 `label_smoothing` 强制为 `0.0`，并在 contract 中显式审计。凡是 label smoothing 不为 0 的 run，只能进入 diagnostic，不得进入 official route。

第四，**functional update 目前是有效的几何校正，但还不是完整 functional optimizer**。当前 FT7 / Adaptive-FT 的核心是 role-wise / event-triggered / task-budgeted second-difference curvature correction。它是一个真实的 functional update rule，但还不是完整 Sobolev / natural functional metric solve。终极系统要保留当前低成本 functional correction，同时建立更规范的 function-space geometry API，避免把 functional update 混成 loss、optimizer trick 或随机 smoothing。

### 0.3 本次整改的基本判断

本次整改不应该推倒当前成果。正确策略是：

$$
\boxed{
\text{保留当前有效路线作为 regression target，同时重构出一个干净、强 contract、full-edge PureKAN-ready 的核心系统。}
}
$$

也就是说：

```text
1. 先冻结当前 selected route 的行为作为旧系统对照；
2. 然后核心化重构，不再让 runner 互相 patch；
3. 清洗 loss contract，强制 label_smoothing = 0；
4. 把当前 transitional KW4 路线重实现为 clean baseline；
5. 再实现 full-edge PureKAN stack；
6. 最后重新做 KANbeFair 外部公平比较。
```

---

## 1. 终极目标定义

### 1.1 Clean PureKAN functional training system

本计划中的终极系统必须满足：

```text
1. 所有官方候选 no teacher / no self-teacher / no distillation。
2. 所有官方候选 loss_type = CE，label_smoothing = 0。
3. 所有官方候选不使用 sampler / class weight / CPU offload。
4. 所有官方 KAN path 不使用 loss.backward graph。
5. 所有 KAN trainable parameters 必须属于 edge-function system。
6. forward / backward / update 必须由核心模块实现，不在 runner 中临时拼接。
7. functional update 是 update rule，不是 geometry loss。
8. 所有 pass/fail 只来自落盘 CSV/JSON/log/manifest。
```

形式化写成：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

训练更新允许：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\Delta\theta_{\text{functional}}.
$$

但禁止：

$$
L_{\text{train}}=CE+\lambda L_{\text{geo}}.
$$

### 1.2 Full-edge PureKAN 定义

本计划把模型分成三个等级，避免混淆。

#### Level A：Transitional Clean DG-KAN

```text
linear-SiLU manual stack
+
KAN-style poly2_silu head
+
manual backward/update
+
functional update
```

这个等级可以作为 regression / transitional baseline，但不能作为终极 full PureKAN 结论。

#### Level B：Strict Head PureKAN

```text
linear or mixed stack
+
strict KAN edge-function head
+
manual backward/update
```

这个等级可以证明 KAN head 有价值，但仍不能证明整网都是 edge-function PureKAN。

#### Level C：Full-edge PureKAN

```text
每一层都是 learnable edge function；
stack 和 head 都是 KAN edge system；
没有普通 linear-SiLU stack 作为主要表达模块；
manual forward/backward/update；
functional update 作用在 edge-function coefficient / knot / basis / role parameters 上。
```

v9.0 的核心目标是建立 Level C 并重新验证。

### 1.3 External fair success

Clean system 必须重新与 MLP 比较，不能直接继承旧 route 结论。外部公平比较至少包含：

```text
KB-MLP baseline
KB-KAN official baseline
DG-Transitional clean baseline
DG-FullEdge-PureKAN
DG-FullEdge-PureKAN + functional update
NoOp functional control
Random functional control
```

公平 envelope 至少包括：

$$
Params_{\text{KAN}}\leq1.05Params_{\text{MLP}},
$$

$$
ForwardFLOPs_{\text{KAN}}\leq1.05ForwardFLOPs_{\text{MLP}},
$$

$$
StepTime_{\text{KAN}}\leq1.50StepTime_{\text{MLP}},
$$

$$
PeakMemory_{\text{KAN}}\leq1.05PeakMemory_{\text{MLP}}.
$$

Training compute 还要补：

$$
BackwardFLOPs_{\text{KAN}}\leq1.50BackwardFLOPs_{\text{MLP}},
$$

或：

$$
KernelTime_{\text{KAN}}\leq1.50KernelTime_{\text{MLP}}.
$$

---

## 2. 代码核心化整改目录

### 2.1 新核心目录

建立新目录：

```text
dgkan/
  __init__.py

  config.py
  specs.py
  contracts.py
  registry.py

  data/
    __init__.py
    vision.py
    kanbefair.py
    splits.py

  models/
    __init__.py
    edge_functions.py
    edge_layers.py
    full_edge_stack.py
    transitional_stack.py
    heads.py
    model.py

  kernels/
    __init__.py
    stack_backward.py
    head_backward.py
    compiled_head.py
    triton_edge.py

  optim/
    __init__.py
    manual_adamw.py
    foreach_adamw.py
    state.py

  functional/
    __init__.py
    geometry.py
    second_diff.py
    controller.py
    guards.py
    role_budget.py

  training/
    __init__.py
    step.py
    loop.py
    eval.py
    gradcheck.py

  external/
    __init__.py
    kanbefair_adapter.py
    counters.py
    baselines.py

  profiling/
    __init__.py
    timing.py
    memory.py
    kernels.py
    compute.py

  artifacts/
    __init__.py
    writer.py
    route.py
    audit.py

experiments/
  run_v90_refactor_audit.py
  run_v90_clean_revalidate.py
```

原则：

```text
1. runner 只负责 parse args、调用核心 API、落盘 artifact；
2. runner 不再定义模型 class；
3. runner 不再定义 optimizer class；
4. runner 不再定义 functional controller；
5. runner 不再 monkey patch 旧 runner；
6. v7/v8 历史 runner 只作为 legacy reproduction，不作为核心依赖。
```

### 2.2 `specs.py`

定义唯一候选配置：

```python
@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    model_level: Literal["transitional", "strict_head", "full_edge"]
    stack_type: str
    head_type: str
    edge_basis: str
    hidden_dim: int
    depth: int
    basis_count: int
    loss_type: str = "CE"
    label_smoothing: float = 0.0
    optimizer_type: str = "manual_adamw"
    functional_update: str | None = None
    official_eligible: bool = True
```

强规则：

```text
official_eligible = true 时：
  label_smoothing 必须等于 0.0；
  loss_type 必须等于 CE；
  model_level 如果要声明终极 PureKAN，必须是 full_edge。
```

### 2.3 `contracts.py`

必须硬 assert：

```python
assert spec.loss_type == "CE"
assert spec.label_smoothing == 0.0
assert external_teacher_used == 0
assert self_teacher_used == 0
assert geometry_loss_used == 0
assert sampler_changed == 0
assert class_weight_used == 0
assert cpu_offload_used == 0
assert uses_loss_backward == 0
assert fake_data_used == 0
assert proxy_row_used == 0
```

新增两个 contract：

```text
CleanCEContract:
  CE only, label_smoothing = 0, no class weights.

FullEdgePureKANContract:
  all trainable modules implement EdgeFunctionProvider;
  no standard Linear-SiLU stack counted as full_edge.
```

### 2.4 `models/edge_functions.py`

定义所有函数基，统一 forward / derivative / geometry。

#### Poly2-SiLU edge

$$
\phi(x)
=
a_0
+
a_1x
+
a_2\operatorname{SiLU}(x)
+
s(b_1x+b_2x^2).
$$

#### RBF edge

$$
\phi(x)
=
\sum_{k=1}^{K} c_k
\exp\left(-\frac{(x-\mu_k)^2}{2\sigma^2}\right).
$$

#### AB-RBF edge

$$
\phi(x)
=
a_0+a_1x+a_2\operatorname{SiLU}(x)
+
\sum_{k=1}^{K}c_kRBF_k(x).
$$

#### Piecewise / spline edge

$$
\phi(x)=\sum_k c_k B_k(x).
$$

每个 edge function 必须实现：

```python
forward(x)
backward(dy, cache)
geometry_metrics()
functional_direction()
parameter_count()
flops_count()
```

### 2.5 `models/edge_layers.py`

Full-edge layer：

$$
y_j=\sum_i \phi_{ij}(x_i).
$$

不再写成：

$$
y = \operatorname{SiLU}(xW^\top).
$$

Level C candidate 只能使用 `EdgeFunctionLayer` 或它的 kernelized等价实现。

### 2.6 `models/transitional_stack.py`

保留当前 KW4/KW6 类似路线作为 transitional baseline：

```text
PackedLinearSiluStack
PackedPrefixNoInputGradStack
CachedAtenLowerOnlySiluBackwardStack
```

但所有 artifact 必须标记：

```text
model_level = transitional
full_edge_purekan_pass = 0
```

避免把它当作终极 PureKAN。

### 2.7 `optim/manual_adamw.py`

只放 AdamW-equivalent task update：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\theta_{t+1}
=
\theta_t-\eta
\left(
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
+\lambda\theta_t
\right).
$$

禁止把 functional update 写在 optimizer 内部。optimizer 只负责 task direction。

### 2.8 `functional/controller.py`

functional update 必须独立于 optimizer。定义：

$$
\Delta\theta_{\text{functional}}
=
-\lambda_t P_{\mathcal{T}}
\left(
M^{-1}\nabla_\theta R_{\text{geo}}
\right).
$$

其中 $R_{\text{geo}}$ 可为 edge curvature、slope、Jacobian proxy 等。它不是 loss，只是 update rule。

必须记录：

```text
event_score
event_threshold
event_triggered
role_budget
holdout_descent_ratio
bad_step_flag
functional_update_norm
geometry_delta
functional_update_time
```

### 2.9 `external/kanbefair_adapter.py`

把 KANbeFair protocol 独立出来，不再散落在 runner。必须支持：

```text
KB-MLP
KB-KAN
DG-Transitional
DG-FullEdge
DG-FullEdge+Functional
NoOp
RandomFunc
```

并统一：

```text
params counter
forward FLOPs counter
training compute proxy
wall-clock timer
memory meter
metric parser
```

---

## 3. 整改阶段设计

## Phase A：代码冻结与真相审计

### A1：Legacy current route audit

#### 目标

确认当前旧系统中 selected route 的真实实现，不靠报告字段。

#### 必须记录

```text
candidate_id
source_file
stack_type
head_type
model_level
loss_type
label_smoothing
uses_smooth_ce
external_teacher_used
self_teacher_used
uses_loss_backward
manual_forward
manual_backward
manual_update
optimizer_type
functional_update_type
is_full_edge_purekan
```

#### 判断标准

如果发现：

```text
label_smoothing != 0
```

则 route 只能记为：

```text
LegacySmoothedCE
```

不能作为 CleanCE official result。

如果：

```text
stack_type = linear / linear_silu
```

则 route 只能记为：

```text
TransitionalDGKAN
```

不能作为 FullEdgePureKAN result。

#### 可视化

```text
A1_candidate_contract_heatmap.svg
A1_implementation_lineage_graph.svg
A1_loss_contract_violation_matrix.svg
```

### A2：Legacy behavior freeze

#### 目标

把旧 selected route 作为 regression target 冻结，避免重构后性能不可追踪。

#### 必须记录

```text
test_acc
delta_vs_MLP
params_ratio
FLOPs_ratio
step_ratio
memory_ratio
curvature_ratio
functional_event_count
```

#### 判断标准

重构后 transitional clean baseline 如果要算 parity，必须满足：

$$
|Acc_{\text{new}}-Acc_{\text{legacy}}|\leq0.003.
$$

$$
|StepRatio_{\text{new}}-StepRatio_{\text{legacy}}|\leq0.10.
$$

$$
|CurvatureRatio_{\text{new}}-CurvatureRatio_{\text{legacy}}|\leq0.05.
$$

#### 可视化

```text
A2_legacy_vs_refactor_parity_bar.svg
A2_step_memory_parity.svg
A2_curvature_parity.svg
```

---

## Phase B：核心文件重构

### B1：建立 dgkan core package

#### 目标

把模型、训练、functional、external、profiling 全部移出 runner。

#### 完成标准

```text
dgkan/ 包可 import；
pytest smoke 通过；
old runner 不再作为 new runner 依赖；
new runner 只调用 dgkan API。
```

#### 必须记录

```text
module_name
source_file
public_api_count
unit_test_count
import_pass
```

#### 可视化

```text
B1_module_dependency_graph.svg
B1_public_api_table.md
```

### B2：Contract hardening

#### 目标

让任何 official candidate 不可能绕过 clean constraints。

#### 必须测试

```text
label_smoothing != 0 should fail
loss_type != CE should fail
external_teacher_used = 1 should fail
uses_loss_backward = 1 should fail
model_level=full_edge but linear stack should fail
```

#### 判断标准

所有 negative tests 必须 fail：

```text
negative_contract_tests_pass = 1
```

#### 可视化

```text
B2_contract_test_matrix.svg
```

### B3：Core artifact writer

#### 目标

所有实验输出统一 schema，不再每个 runner 自己写不同字段。

#### 必须落盘

```text
run_manifest.json
candidate_registry.csv
contract_audit.csv
task_summary.csv
efficiency_summary.csv
functional_trace.csv
geometry_summary.csv
external_fair_summary.csv
route_decision.json
failure_table.csv
```

---

## Phase C：Clean CE-only transitional revalidation

### C1：Label smoothing removal audit

#### 目标

验证去掉 label smoothing 后，当前 transitional route 是否仍成立。

#### 必跑

```text
KB-MLP
DG-Transitional-LegacySmoothCE
DG-Transitional-CleanCE
DG-Transitional-CleanCE+Functional
NoOp
RandomFunc
```

#### 必须记录

```text
loss_type
label_smoothing
test_acc
delta_vs_KB_MLP
ECE
NLL
curvature_ratio
step_ratio
memory_ratio
functional_events
```

#### 判断标准

CleanCE task preservation：

$$
Acc_{\text{CleanCE}}\geq Acc_{\text{LegacySmoothCE}}-0.005.
$$

CleanCE external fair pass：

$$
Acc_{\text{CleanCE+Functional}}\geq Acc_{\text{KB-MLP}}.
$$

$$
ParamsRatio\leq1.05.
$$

$$
FLOPsRatio\leq1.05.
$$

$$
StepRatio\leq1.50.
$$

Functional geometry pass：

$$
R_{\text{curv,CleanCE+Functional}}\leq0.90R_{\text{CleanCE}}.
$$

If CleanCE fails but LegacySmoothCE passes, route must be:

```text
LegacyLossDependentSuccess
```

#### 可视化

```text
C1_clean_vs_smooth_accuracy.svg
C1_clean_vs_smooth_ece_nll.svg
C1_clean_vs_smooth_curvature.svg
C1_clean_vs_smooth_system_pareto.svg
```

### C2：Refactor parity for transitional model

#### 目标

确认新核心系统能复现旧 transitional model 行为。

#### 判断标准

$$
|Acc_{\text{core}}-Acc_{\text{legacy}}|\leq0.003.
$$

$$
GradRelErr_{\max}\leq10^{-4}.
$$

$$
GradCos_{\min}\geq0.999.
$$

#### 可视化

```text
C2_legacy_core_parity.svg
C2_gradcheck_lollipop.svg
```

---

## Phase D：Full-edge PureKAN 实现

### D1：FullEdge-Poly2Silu stack

#### 目标

实现第一条 Level C candidate：stack 与 head 都是 edge function。

Edge layer：

$$
y_j=\sum_i \phi_{ij}(x_i).
$$

Poly2-SiLU edge：

$$
\phi_{ij}(x)
=
a_{ij,0}
+
a_{ij,1}x
+
a_{ij,2}\operatorname{SiLU}(x)
+
s(b_{ij,1}x+b_{ij,2}x^2).
$$

#### 必须实现

```text
manual forward
manual backward
manual update
geometry metrics
functional direction
params counter
FLOPs counter
cache breakdown
```

#### 必须记录

```text
candidate_id
model_level = full_edge
edge_basis = poly2_silu
GradPass
params
FLOPs
memory_ratio
step_ratio
test_acc
curvature
```

#### 判断标准

Implementation pass：

$$
GradRelErr_{\max}\leq10^{-4}.
$$

$$
GradCos_{\min}\geq0.999.
$$

FullEdge contract pass：

```text
all trainable params implement EdgeFunctionProvider
no standard linear-silu stack counted in full_edge path
```

### D2：FullEdge-RBF / ABRBF stack

#### 目标

把已有 RBF / ABRBF primitive 迁移到核心 full-edge stack，判断函数基是否比 Poly2Silu 更稳。

RBF:

$$
\phi_{ij}(x)=\sum_k c_{ij,k}\exp\left(-\frac{(x-\mu_k)^2}{2\sigma^2}\right).
$$

ABRBF:

$$
\phi_{ij}(x)
=
a_0+a_1x+a_2\operatorname{SiLU}(x)
+
\sum_k c_kRBF_k(x).
$$

#### 判断标准

RBF / ABRBF candidate 进入 external fair candidate pool 需要：

$$
Acc_{\text{candidate}}\geq Acc_{\text{KB-MLP}}-0.005.
$$

$$
StepRatio\leq1.50.
$$

$$
MemoryRatio\leq1.05.
$$

### D3：FullEdge-Spline / Piecewise stack

#### 目标

对齐 KANbeFair 论文指出的 B-spline symbolic advantage，建立真正 spline/Piecewise edge candidate。

#### 判断标准

Symbolic task pass：

$$
RMSE_{\text{FullEdge-Spline}}\leq RMSE_{\text{KB-MLP}}.
$$

Geometry pass：

$$
R_{\text{curv,Functional}}\leq0.90R_{\text{base}}.
$$

---

## Phase E：Functional training system 重构

### E1：Functional update API

#### 目标

把当前 FT7 / Adaptive-FT 从 runner 中抽象成通用模块。

统一接口：

```python
controller.propose(params, grads, geometry, holdout_state) -> FunctionalUpdate
controller.apply(params, update)
controller.audit() -> dict
```

#### 必须记录

```text
event_score
event_threshold
triggered
role_budget
task_direction_cos
holdout_descent_ratio
bad_step_rate
functional_update_norm
functional_update_time
geometry_delta
```

#### 判断标准

One-step pass：

$$
\cos(d_{\text{corrected}},d_{\text{task}})\geq0.85.
$$

$$
HoldoutDescentRatio\geq0.95.
$$

$$
BadStepRate\leq0.02.
$$

### E2：Full-edge functional geometry

#### 目标

让 functional update 作用在 full-edge basis / coefficients / knots 上，而不是只对 linear stack 参数做 second-diff。

几何项包括：

```text
coefficient curvature
slope p95
edge finite-difference curvature
Jacobian norm
local Lipschitz
```

曲率：

$$
R_{\text{curv}}
=
\sum_{i,j,k}
(c_{ij,k+2}-2c_{ij,k+1}+c_{ij,k})^2.
$$

函数空间有限差分曲率：

$$
FD_{\text{curv}}
=
\mathbb{E}_{x,\epsilon}
\left[
\frac{\phi(x+\epsilon)-2\phi(x)+\phi(x-\epsilon)}{\epsilon^2}
\right]^2.
$$

#### 判断标准

Functional useful：

$$
Acc_{\text{Functional}}\geq Acc_{\text{Base}}-0.005.
$$

$$
R_{\text{curv,Functional}}\leq0.90R_{\text{Base}}.
$$

$$
StepRatio_{\text{Functional}}\leq1.50.
$$

---

## Phase F：Kernel / system closure

### F1：Manual core profiler

#### 目标

统一 timing，不再依赖每个 runner 的局部测量。

Protocols：

```text
T0 = 50 warmup / 200 reps
T1 = 100 warmup / 500 reps
T2 = 200 warmup / 1000 reps
T3 = train-step-only phase-clean
T4 = full-loop with functional guard / validation sync
```

#### 判断标准

Robust timing pass：

$$
Q_{90}(StepRatio)\leq1.50.
$$

Strict timing pass：

$$
\max(StepRatio)\leq1.50.
$$

Unknown time：

$$
unknown\_time\_fraction\leq0.10.
$$

#### 可视化

```text
F1_step_ratio_by_protocol.svg
F1_phase_time_breakdown.svg
F1_kernel_time_by_phase.svg
```

### F2：Kernel paths

#### 目标

把当前 ATen / torch.compile / foreach / Triton 路线统一到 kernels package。

候选：

```text
ATen SiLU backward
torch.compile CE/head backward
Triton edge backward
Triton/full-edge grouped kernel
foreach AdamW update
```

#### 判断标准

Kernel path can be official only if：

$$
GradRelErr_{\max}\leq10^{-4}.
$$

$$
StepRatio\leq1.50.
$$

$$
MemoryRatio\leq1.05.
$$

### F3：Training compute fairness

#### 目标

补齐 forward-only FLOPs 的弱点。

必须记录：

```text
forward_FLOPs
backward_FLOPs_estimate
update_FLOPs_estimate
kernel_time_forward
kernel_time_backward
kernel_time_update
kernel_time_functional
step_energy_proxy_if_available
```

Training fair pass：

至少满足：

$$
ForwardFLOPs_{\text{KAN}}\leq1.05ForwardFLOPs_{\text{MLP}},
$$

and one of:

$$
BackwardFLOPs_{\text{KAN}}\leq1.50BackwardFLOPs_{\text{MLP}},
$$

$$
KernelTime_{\text{KAN}}\leq1.50KernelTime_{\text{MLP}},
$$

$$
StepTime_{\text{KAN}}\leq1.50StepTime_{\text{MLP}}.
$$

---

## Phase G：外部公平重新验证

### G1：KANbeFair baseline reproduction

#### 目标

重新复现外部 baseline，不继承旧 artifact。

必跑：

```text
KB-MLP
KB-KAN
```

任务：

```text
MNIST
Fashion-MNIST
KMNIST
symbolic formula tasks
at least two tabular / ML tasks if runnable
CIFAR-10 if runnable
continual splits
```

#### 判断标准

Classification baseline reproduction：

$$
|Acc_{\text{repro}}-Acc_{\text{reported}}|\leq0.02.
$$

Symbolic reproduction：

$$
\frac{|RMSE_{\text{repro}}-RMSE_{\text{reported}}|}{RMSE_{\text{reported}}}\leq0.20.
$$

### G2：Clean transitional route revalidation

#### 目标

先验证重构后 Level A 是否仍能复现旧 selected route。

#### 成功标准

$$
Acc_{\text{DG-Transitional-Clean}}\geq Acc_{\text{KB-MLP}}.
$$

$$
ParamsRatio\leq1.05.
$$

$$
FLOPsRatio\leq1.05.
$$

$$
StepRatio\leq1.50.
$$

但 route 只能写：

```text
TransitionalCleanSuccess
```

不能写：

```text
FullEdgePureKANSuccess
```

### G3：Full-edge PureKAN external validation

#### 目标

验证 Level C 终极候选。

必跑：

```text
KB-MLP
KB-KAN
DG-Transitional-Clean
DG-FullEdge-Poly2Silu
DG-FullEdge-RBF
DG-FullEdge-ABRBF
DG-FullEdge-Spline
DG-FullEdge-best+Functional
NoOp
RandomFunc
```

#### 成功标准

FullEdge Minimum：

$$
Acc_{\text{FullEdge+Functional}}\geq Acc_{\text{KB-MLP}}.
$$

$$
ParamsRatio\leq1.05.
$$

$$
FLOPsRatio\leq1.05.
$$

$$
StepRatio\leq1.50.
$$

$$
MemoryRatio\leq1.05.
$$

$$
R_{\text{curv,Functional}}\leq0.90R_{\text{Base}}.
$$

FullEdge Strong：

至少两个 task families 满足 FullEdge Minimum。

### G4：Functional causality controls

#### 目标

证明 functional update 不是 no-op / random / overhead artifact。

必跑：

```text
Base
Functional
NoOp matched overhead
Random functional direction
Shuffled role direction
No task budget
No role score
No event score
```

#### 判断标准

Causality pass：

$$
R_{\text{curv,Functional}}<R_{\text{curv,NoOp}}.
$$

$$
R_{\text{curv,Functional}}<R_{\text{curv,Random}}.
$$

$$
Acc_{\text{Functional}}\geq Acc_{\text{NoOp}}-0.002.
$$

---

## Phase H：Robustness / Continual / Boundary

### H1：Robustness

#### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10,0.20
random_erasing = small,medium
affine_shift = mild
```

#### 判断标准

$$
AccDrop_{\text{Functional}}\leq AccDrop_{\text{Base}}
$$

for at least 3 settings.

### H2：Continual balance

#### Splits

```text
Split A: digits 0-2 / 3-5 / 6-9
Split B: digits 0-4 / 5-9
Split C: random balanced 3-task split
Split D: interleaved hard split
```

#### 判断标准

$$
Forgetting_{\text{Functional}}\leq Forgetting_{\text{Base}}.
$$

$$
FinalAvgAcc_{\text{Functional}}\geq FinalAvgAcc_{\text{Base}}.
$$

$$
\max_i Acc_i-\min_i Acc_i\leq0.20.
$$

### H3：Boundary audit

#### 目标

如果 FullEdge PureKAN 只在部分任务成功，必须输出边界。

Labels：

```text
full_edge_broad_win
full_edge_vision_win
full_edge_symbolic_win
transitional_only_win
geometry_only_win
training_compute_fail
timing_fail
mlp_dominates
continual_unbalanced
contract_fail
```

---

## 4. Route decision 设计

### 4.1 Route cases

```text
R1-CleanFullEdgePureKANBroadSuccess:
  Full-edge PureKAN + functional update 在至少两个 task families 中通过 external fair gate。

R2-CleanFullEdgePureKANVisionSuccess:
  Full-edge 在 MNIST/FMNIST/KMNIST vision-family 中通过，但 broad 不足。

R3-CleanTransitionalOnlySuccess:
  clean transitional route 成功，但 full-edge route 未成功。

R4-LegacyLossDependentSuccess:
  旧 label smoothing / smooth CE 成功，clean CE 失败。

R5-FunctionalGeometryOnly:
  functional update 改善 geometry，但 task/fairness 不过。

R6-SystemOnlyFail:
  task/geometry 有优势，但 timing/memory/training compute 不过。

R7-MLPDominatesExternal:
  MLP 在多数外部任务公平条件下显著优于 DG-KAN。

R8-ContractFail:
  no-teacher/no-loss/full-edge/manual contract 失败。

R9-RefactorParityFail:
  core refactor 无法复现 legacy 行为。
```

### 4.2 route_decision.json 必须记录

```text
route
legacy_parity_pass
clean_ce_pass
full_edge_contract_pass
full_edge_task_pass
functional_causality_pass
external_fair_pass
training_compute_fair_pass
robustness_pass
continual_balance_pass
best_candidate
best_model_level
primary_blocker
next_required_implementation
```

---

## 5. 必须落盘 artifacts

```text
run_manifest.json
code_audit_legacy_current.csv
candidate_registry_clean.csv
contract_audit_clean.csv
legacy_refactor_parity.csv
label_smoothing_removal_audit.csv
full_edge_implementation_audit.csv
gradcheck_full_edge.csv
clean_transitional_revalidation.csv
full_edge_external_validation.csv
functional_causality_controls.csv
robust_timing_protocols.csv
training_compute_counter.csv
kanbefair_baseline_reproduction.csv
external_fair_envelope.csv
robustness_perturbation.csv
continual_balance_multisplit.csv
boundary_audit.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_loss_violation
F2_label_smoothing_violation
F3_teacher_or_distill_violation
F4_full_edge_contract_fail
F5_grad_fail
F6_refactor_parity_fail
F7_clean_ce_regression
F8_full_edge_task_fail
F9_functional_causality_fail
F10_timing_fail
F11_memory_fail
F12_training_compute_fail
F13_external_fair_fail
F14_robustness_fail
F15_continual_unbalanced
F16_fake_or_proxy_violation
F17_artifact_missing
```

---

## 6. 第一轮执行顺序

### Step 1：Legacy truth audit

执行 Phase A。确认当前代码里：

```text
哪些 candidate 用 smooth CE；
哪些是 linear-SiLU stack；
哪些是真 full-edge KAN；
哪些 route 只是 transitional success。
```

如果 `label_smoothing != 0` 被确认，立即把旧结果降级为 legacy diagnostic，不再作为 clean official 结论。

### Step 2：Core package skeleton

执行 Phase B。先把代码结构拆出来，不改算法，只迁移 API。

### Step 3：Contract hardening

先跑 negative contract tests，确认所有违规 candidate 都会 fail。

### Step 4：Transitional clean CE revalidation

执行 Phase C。去掉 label smoothing 后重新验证当前 route。这个阶段告诉我们：旧成功是否依赖 smooth CE。

### Step 5：Full-edge implementation

执行 Phase D。实现 FullEdge-Poly2Silu / RBF / Spline stack，先做 gradcheck 和 small task smoke。

### Step 6：Functional update migration

执行 Phase E。让 functional update 作用在 full-edge basis / coefficients 上。

### Step 7：System closure

执行 Phase F。统一 timing、memory、kernel、training compute counter。

### Step 8：External fair validation

执行 Phase G。重新与 KB-MLP / KB-KAN 比较。

### Step 9：Robustness / continual / boundary

执行 Phase H。输出最终边界。

---

## 7. 停止条件

### 成功停止

Full clean success：

```text
clean CE pass
full-edge contract pass
GradPass
external fair pass
functional causality pass
robust timing pass
training compute fair pass
```

Strong success：

```text
Full clean success
+
at least two task families pass
+
robustness or continual balance pass
```

### 失败停止

```text
1. clean CE after removing label smoothing collapses below MLP；
2. full-edge candidates all fail task by >1% vs MLP；
3. full-edge backward cannot pass gradcheck；
4. robust timing Q90 step ratio > 1.50 for all candidates；
5. training compute fairness cannot be established；
6. functional update no-op/random controls match or beat real functional update；
7. external KANbeFair broad tasks show MLP dominates；
8. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 8. 最终解释规则

### Case A：Full-edge clean system succeeds

可以声明：

```text
DG-KAN has established a clean full-edge PureKAN functional training system with external fair advantage.
```

### Case B：Clean transitional succeeds, full-edge fails

必须声明：

```text
Current method is a clean transitional DG-KAN success, not yet a terminal full-edge PureKAN success.
```

### Case C：Legacy smoothing succeeds, clean CE fails

必须声明：

```text
Previous selected route depended on loss modification; terminal claim invalid until CleanCE route is restored.
```

### Case D：Full-edge geometry improves but task fails

必须声明：

```text
Functional geometry mechanism works, but does not yet produce task superiority.
```

### Case E：External broad tasks fail

必须声明：

```text
DG-KAN advantage is task-family-limited.
```

---

## 9. 最终建议

本整改计划的核心不是“重写代码让它更好看”，而是：

$$
\boxed{
\text{用代码结构保证科学结论不会被旧默认值、runner patch、loss drift 和非 full-edge 结构污染。}
}
$$

当前应优先做：

```text
1. 清理 label smoothing；
2. 分离 transitional success 与 full-edge PureKAN success；
3. 核心化模型 / optimizer / functional / external / profiling；
4. 实现 full-edge stack；
5. 重新做 clean CE external fair validation。
```

只有完成这套整改后，最终目标才可以写成：

$$
\boxed{
\text{一个完全干净的 PureKAN functional training system。}
}
