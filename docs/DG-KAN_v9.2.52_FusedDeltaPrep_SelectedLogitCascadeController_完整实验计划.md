# DG-KAN v9.2.52 Fused Functional Delta-Prep Kernel 与 Selected-Logit Cascade Controller 完整实验计划

> 本计划基于 v9.2.51 `Branch-Logit Delta Kernel 与 Coverage-Preserving Online Controller` 的真实复盘制定。  
> v9.2.51 的 terminal route 是：
>
> ```text
> route = R9-BranchDeltaPredictiveButTooExpensive
> base_candidate = LQ-t2-h256
> success_v9251_strict_purekan_functional = False
> success_v9251_full_functional = False
> success_v9251_external_ready = False
> ```
>
> v9.2.51 的关键事实是：
>
> ```text
> P1 M0 logit-prep attribution:
>   pass = 1
>   dominant subphase = L1-RealFunctional delta preparation
>   L1 ratio = 0.508055
>   L5 branch forward preparation ratio = 0.247635
>   branch-logit path ratio = 0.889772
>
> P2 branch-logit delta:
>   BLD1-DeltaViewExactBranchLogits:
>     AUC = 0.886903
>     accept agreement = 1.0
>     step ratio = 4.700044
>     system pass = 0
>
>   BLD2-LinearizedLogitDeltaJVP:
>     AUC = 0.534446
>     agreement = 0.401389
>     step ratio = 1.411116
>
>   BLD4-TopClassTailClassDelta:
>     AUC = 0.720016
>     agreement = 0.317708
>     step ratio = 1.616674
>
>   BLD5-CachedControlBaselineDelta:
>     AUC = 0.596259
>     agreement = 0.455816
>     step ratio = 1.308337
>
> P3 cheap recall:
>   best = CR3-FamilyReliability
>   recall = 0.346837
>   candidate rate = 0.25
>   candidate bad-event = 0.225347
>
> P4 cascade controller:
>   best = C4-LowRankDeltaController
>   precision = 0.322581
>   coverage = 0.008073
>   bad-event = 0.215054
>   official eligible = 0
>
> P5 oracle support:
>   precision = 1.0
>   coverage = 0.122135
>   bad-event = 0.0
>   measured signal strata = 3
>   balanced diagnostic rows = 18
>   support measurement pass = 0
>
> current blocker:
>   branch_logit_delta_predictive_but_not_system_legal
> ```
>
> v9.2.52 的核心判断是：  
> **exact branch-logit delta 路径已经证明 safe-good 可观测，但目前它本质上仍在做昂贵的 RealFunctional delta preparation 与 branch forward preparation。低成本近似虽然便宜，但没有保留 exact path 的 agreement / control-gap 信息。因此下一轮不能继续只试新的 cheap score；必须实现真正的 fused functional delta-prep + selected-logit branch delta kernel。**

---

## 0. 硬约束

本计划继续遵守：

```text
no teacher
no self-teacher
no distillation
no loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
official controller 不使用 dataset_name 分支
PureKANConv / PureKANFormer 继续 deferred
```

Functional update 仍然是 update rule，不是 loss：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\Delta\theta_{\text{functional}}.
$$

任务目标保持标准 CE：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

数据集只能作为诊断切片，不能作为 official controller 条件：

```text
allowed:
  report MNIST / Fashion-MNIST / KMNIST slice metrics
  report overhead / safety / support distribution by dataset
  report leave-dataset-out generalization
  report signal-stratum / family composition by dataset

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use threshold B
  if dataset == MNIST: abstain
  tune threshold separately per dataset
  choose branch-delta approximation separately per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

---

# Part I. 对 v9.2.51 的独立判断

## 1. v9.2.51 没有达到目标

v9.2.51 没有 strict PureKAN functional success。原因很明确：

```text
branch-logit delta predictivity pass = 1
branch-logit delta system pass = 0
cheap recall pass = 0
cascade controller pass = 0
support measurement pass = 0
LDO / LSO / paired replay = not_run
```

因此不能声明：

```text
strict PureKAN local causal evidence
full functional success
external-ready success
Beyond-MLP success
```

P2 的 exact/delta-view result 是强信号，但它不是 system-legal。P3/P4 的 cheap recall 与 cascade controller 都没有形成可部署 policy。P5 oracle support 高，但 support measurement 太窄，不能越过 official gate。

## 2. v9.2.51 的真实进展

v9.2.51 的进展非常重要，它不是原地失败。

第一，M0 logit-prep 已经进一步定位到 branch-logit path。`L1-RealFunctional delta preparation` 单项占 `0.508055`，`L5-branch forward preparation` 占 `0.247635`，branch-logit path 总占 `0.889772`。这说明 v9.2.50 之后继续压 scalar metric kernel 已经不是主线；真正瓶颈是 **functional delta preparation + branch logits**。

第二，exact / delta-view branch-logit delta 保留了 safe-good signal。`BLD1-DeltaViewExactBranchLogits` AUC `0.886903`，accept agreement `1.0`。这说明 v9.2.48 的 online gap-probe signal 没有消失，也说明 branch-output displacement 是正确方向。

第三，低成本近似目前没有保留 exact signal。`BLD2` step ratio 进入 `1.50` 以内，但 AUC 只有 `0.534446`，agreement `0.401389`；`BLD4` AUC 有 `0.720016`，但 agreement 只有 `0.317708`，step ratio 也略高于 gate；`BLD5` 系统便宜但 AUC / agreement 都不足。因此当前不是“随便找一个 cheap approximation”就能过。

第四，oracle support 仍然非常强：precision `1.0`，coverage `0.122135`，bad-event `0.0`。这说明 good events 没有消失。真正 blocker 是 legal/system path 不能识别它们。

## 3. 当前真正 blocker

当前 blocker 可以写成：

$$
\boxed{
\text{exact branch-output displacement 有价值，但当前实现没有 system-legal 的 fused delta-prep / selected-logit path。}
}
$$

更具体地说，有三层问题。

### 3.1 RealFunctional delta preparation 太贵

L1 占 `0.508055`。这不是 controller decision 造成的，也不是 logging 造成的。它说明每个 candidate event 中 functional delta 的构造、打包、role/channel 组合、参数视图创建、或 branch-specific delta materialization 太重。

### 3.2 Branch forward/logit path 太贵

L5 占 `0.247635`，L6 logit buffer allocation 也占 `0.070448`。这说明即使不用完整 metric reduction，多分支 logit generation 本身也不便宜。下一步必须避免完整 branch forward 或完整 logits materialization。

### 3.3 Cheap recall 与 cascade 没有形成 legal support

P3 recall 只有 `0.346837`，bad-event `0.225347`；P4 controller precision `0.322581`，coverage `0.008073`，bad-event `0.215054`。这不是调一个 threshold 可以解决的。Cheap features 暂时只能做 diagnostic 或 weak candidate prior，不能单独承担 official accept。

---

# Part II. v9.2.52 总体目标

v9.2.52 的总体目标是：

$$
\boxed{
\text{用 fused functional delta-prep + selected-logit branch delta kernel，将 exact gap-probe 的 safe-good predictivity 转化为 system-legal controller。}
}
$$

更细分地说，本轮必须同时完成五个目标：

```text
1. 把 L1 RealFunctional delta preparation 继续拆解到可实现层。
2. 实现真实 fused delta-prep kernel，避免每个 branch / event 重复 materialization。
3. 实现 selected-logit branch delta path，避免完整 logits / 完整 branch forward。
4. 用 high-recall candidate discovery + fused branch-delta confirm 建立 controller。
5. 在同一轮并行打开 LDO/LSO scout 与 paired replay scout，但 official success 仍受 gate 控制。
```

---

## 1. Fused delta-prep kernel success

把 functional delta preparation 写成一次性、可复用、可缓存的 fused path：

$$
\Delta\theta_F
=
\mathcal{F}_{\text{fused}}
(
\theta,
g_{\text{task}},
s_{\text{role}},
s_{\text{risk}},
s_{\text{support}}
).
$$

目标不是改变 functional update，而是改变 delta 的生成与复用方式。对于同一 train step，RealFunctional branch 所需的 $\Delta\theta_F$ 应只生成一次，然后被 branch-logit delta kernel 复用。

Delta-prep system gate：

$$
T_{\Delta\theta_F,\text{fused}}
\leq0.35
T_{\Delta\theta_F,\text{v9251-ref}}.
$$

或者按整体 step：

$$
StepRatio_{q90}\leq1.50.
$$

Memory gate：

$$
MemoryRatio\leq1.05.
$$

Correctness sanity：

$$
\frac{\|\Delta\theta_{F,\text{fused}}-\Delta\theta_{F,\text{ref}}\|}
{\|\Delta\theta_{F,\text{ref}}\|+\epsilon}
\leq10^{-4}
$$

for exact fused candidate.

For approximate candidate：

$$
Agreement_{\text{branch-confirm}}\geq0.90.
$$

---

## 2. Selected-logit branch delta success

不要为每个 branch 生成完整 logits。先只生成对 CE / margin / risk / gap 足够的 selected logits：

```text
true class
current top-1
current top-2
hard negative class
optional tail class
```

设 selected class set 为 $\mathcal{C}_s$，则：

$$
\widehat z_{b,\mathcal{C}_s}
=
z_{0,\mathcal{C}_s}
+
\widehat{\Delta z}_{b,\mathcal{C}_s}.
$$

Selected-logit pass：

$$
AUC(\widehat S_{\text{gap}},Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(\widehat S_{\text{gap}},V_{\text{safe-grounded}})\geq0.35.
$$

Agreement pass：

$$
Agreement_{\text{accept}}\geq0.90.
$$

System pass：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

If selected-logit path has AUC but bad agreement, it can be used only as candidate generator, not official accept.

---

## 3. High-recall candidate discovery success

v9.2.51 的 cheap recall failed，因此 v9.2.52 的 candidate discovery 不能只依赖 `CR3-FamilyReliability`。Candidate generator should combine:

```text
cheap role/risk/family features
selected-logit delta signal
support density
family reliability
risk-safe prior
```

Candidate generator pass：

$$
Recall_{\text{safe-good}}\geq0.80,
$$

$$
CandidateRate\leq0.35,
$$

$$
CandidateBadEventRate\leq0.20.
$$

如果 CandidateRate 需要高于 `0.35` 才能 recall >= `0.80`，说明 cheap discovery 仍不足；可以转入 branch-delta scan mode，但 system burden 要单独计入。

---

## 4. Coverage-preserving cascade controller success

Official controller：

$$
Accept(e)
=
Candidate(e)
\land
FusedBranchDeltaConfirm(e)
\land
RiskSafe(e)
\land
SupportBalanced(e).
$$

其中：

$$
Candidate(e)
=
\mathbb{1}[S_{\text{candidate}}(e)\geq\tau_c],
$$

$$
FusedBranchDeltaConfirm(e)
=
\mathbb{1}[\widehat S_{\text{gap}}(e)\geq\tau_g],
$$

$$
RiskSafe(e)
=
\mathbb{1}[S_{\text{risk}}(e)\leq\rho],
$$

$$
SupportBalanced(e)
=
\mathbb{1}[Rel(family(e))\geq r_0]
\cdot
\mathbb{1}[Density(e)\geq d_0].
$$

Heldout gate：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

Support gate：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

System gate：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

---

## 5. Support measurement success

v9.2.51 measured signal strata count was `3` and balanced diagnostic rows were only `18`。v9.2.52 必须扩大 support measurement，否则 controller 即使局部过也不能外推。

Measurement pass：

```text
natural_real_event_count >= 12000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 6
measured_family_count >= 12
```

Official split rule：

```text
natural rows:
  official controller / leave-out / paired replay basis

balanced diagnostic rows:
  attribution / support analysis only
```

Balanced diagnostic rows 不得直接混入 official heldout pass，除非显式分离 calibration / heldout / diagnostic roles。

---

# Part III. 核心假设

## H1：v9.2.51 的失败主因是 RealFunctional delta preparation materialization，而不是 safe-good 不可观测

证据：

```text
BLD1 AUC = 0.886903
BLD1 agreement = 1.0
BLD1 step ratio = 4.700044
L1 RealFunctional delta preparation ratio = 0.508055
oracle precision = 1.0
oracle coverage = 0.122135
oracle bad-event = 0.0
```

H1 成立标准：

fused delta-prep candidate 能把 L1 time 降低至少 `60%`，同时保留：

$$
AUC_{\text{safe-good}}\geq0.70
$$

or:

$$
Agreement_{\text{accept}}\geq0.90.
$$

H1 失败标准：

```text
L1 降低后整体 step 仍 > 2.0；
或者 L1 并非实际 dominant after precise profiling；
或者 fused delta-prep 破坏 safe-good signal。
```

---

## H2：selected-logit delta 比 full-logit branch forward 更可能 system-legal

BLD4 的 AUC 已达到 `0.720016`，说明 selected class / tail class 方向有 signal，但 agreement 低。H2 认为如果 selected-logit delta 从 formula diagnostic 变成真实 fused kernel，并且保留 risk/gap confirmation，则可能恢复 agreement。

H2 成立标准：

selected-logit fused candidate 达到：

$$
AUC_{\text{safe-good}}\geq0.70,
$$

$$
Agreement_{\text{accept}}\geq0.80
$$

作为 diagnostic；official 需要：

$$
Agreement_{\text{accept}}\geq0.90.
$$

H2 失败标准：

```text
selected-logit path 即使真实 fused 后 agreement 仍 < 0.80；
或 bad-event 无法降到 <= 0.05。
```

---

## H3：cheap recall 不足，但 selected-logit delta 可以补 candidate discovery

CR3 recall 只有 `0.346837`。H3 认为 pure cheap recall 不够，但 selected-logit delta / top-class delta 可以作为 high-recall candidate generator。

H3 成立标准：

candidate generator 达到：

$$
Recall_{\text{safe-good}}\geq0.80,
$$

$$
CandidateRate\leq0.35,
$$

$$
CandidateBadEventRate\leq0.20.
$$

H3 失败标准：

```text
candidate recall 仍 < 0.60；
或 candidate bad-event > 0.30；
或 candidate rate 必须接近 all-accept 才有 recall。
```

---

## H4：oracle support 仍存在，下一步不是 carrier reset

H4 成立标准：

fresh v9.2.52 natural rows 上：

$$
OraclePrecision_{\text{safe-good}}\geq0.75,
$$

$$
OracleCoverage_{\text{safe-good}}\in[0.03,0.15],
$$

$$
OracleBadEventRate\leq0.05.
$$

H4 失败标准：

```text
fresh oracle precision < 0.75
or coverage < 0.03
or bad-event > 0.05
```

如果 H4 失败，下一步回到 carrier/support reset；如果 H4 成立但 controller fail，则继续 legal/system/controller。

---

## H5：如果 fused exact path 仍太贵，必须进入 CUDA/Triton kernelization，而不是继续 Python-level diagnostic

H5 成立标准：

```text
Fused Python / torch.compile / Triton smoke branch 没有达到 system gate；
但 profiler 显示 bottleneck 可被 single custom kernel 合并。
```

此时 route 应明确写：

```text
R9-FusedDeltaPredictiveButNeedsCustomKernel
```

而不是继续试 threshold。

---

# Part IV. 并行执行设计

v9.2.52 runner 必须并行执行：

```text
Lane A:
  v9.2.51 boundary reproduction

Lane B:
  L1 RealFunctional delta-prep subphase attribution:
    role score
    functional delta compute
    delta tensor assembly
    branch packing
    parameter view setup
    device sync / logging

Lane C:
  fused delta-prep candidates:
    cached functional delta state
    preallocated delta buffers
    in-place delta view
    packed role-wise delta
    fused RealFunctional delta-prep

Lane D:
  selected-logit branch delta candidates:
    true/top/hard-negative logits
    top-class delta
    tail-class delta
    selected-logit CE/margin/risk/gap

Lane E:
  fused multi-branch selected-logit kernel:
    RealFunctional / AdamWParallel / bestLR in branch dimension

Lane F:
  high-recall candidate generator:
    cheap features + selected-logit signal + support density

Lane G:
  coverage-preserving cascade controller

Lane H:
  support expansion:
    natural rows
    balanced diagnostic rows

Lane I:
  LDO / LSO

Lane J:
  paired replay scout and official replay

Lane K:
  short-run scout if paired replay passes
```

Gate discipline：

```text
diagnostic rows may be measured early
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  fused branch-delta predictivity pass
  system gate pass
  cascade controller pass
  LDO/LSO pass
```

---

# Part V. Candidate designs

## 1. Base and carrier

Official base remains：

```text
R2-LQ-fanin-output-scale-confirmed
```

Carrier candidates：

```text
A0-current-v9251
A3-RoleWiseFT7ResetCarrier
A6-HybridRoleControlRiskCarrier
```

A1 risk-bounded carrier remains diagnostic unless it passes carrier-active gate：

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

---

## 2. Fused delta-prep candidates

### FDP0：v9.2.51 BLD1 reference

Reference only. Expected predictive but too expensive.

### FDP1：CachedFunctionalDeltaState

Compute $\Delta\theta_F$ once per step and store compact role-wise delta state:

```text
functional_role_delta_stack
functional_role_delta_head
risk_gate_state
support_state
```

No repeated per-branch reconstruction.

### FDP2：PreallocatedDeltaBuffers

Preallocate all delta buffers:

```text
delta_stack_buffer
delta_head_buffer
branch_delta_buffer
selected_logit_buffer
metric_buffer
```

Avoid dynamic allocation in L1/L6.

### FDP3：PackedRoleWiseDeltaPrep

Pack role-wise delta into contiguous tensor layout:

$$
\Delta\theta_F
=
[\Delta\theta_{\text{stack}},\Delta\theta_{\text{head}}].
$$

This targets loop / scatter overhead.

### FDP4：InPlaceDeltaViewNoCopy

Use reversible delta-view without full parameter materialization:

$$
\theta_b = \theta + \alpha_b\Delta\theta_b.
$$

### FDP5：FusedRealFunctionalDeltaPrep

Single fused kernel / compiled block computes:

```text
role weights
functional scale
risk cap
branch ratio cap
delta buffer
```

### FDP6：HybridFusedDeltaPrep

Pipeline:

```text
CachedFunctionalDeltaState
+ PreallocatedDeltaBuffers
+ PackedRoleWiseDeltaPrep
+ InPlaceDeltaViewNoCopy
```

---

## 3. Selected-logit branch delta candidates

### SLD0：Full-logit exact reference

Same as BLD1; reference only.

### SLD1：TrueTopHardNegativeDelta

Selected classes:

```text
true class
top-1 predicted class
top-2 predicted class
hard negative
```

### SLD2：CE-Margin Selected Delta

Computes only logits required for:

```text
CE approx
margin p10
wrong confidence
gap score
```

### SLD3：TopKClassDelta

Use top-k logits with:

$$
k\in\{2,3,5\}.
$$

### SLD4：SelectedLogitLowRankDelta

Low-rank output delta only on selected logits:

$$
\widehat{\Delta z}_{\mathcal{C}_s}
=
U_{k,\mathcal{C}_s}U_k^T\Delta z.
$$

### SLD5：CachedControlSelectedDelta

Uses selected-logit RealFunctional delta and cached control baseline:

$$
S_{\text{gap-cache-selected}}
=
\widehat{Gain}_{F,\mathcal{C}_s}
-
EMA(
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})
).
$$

### SLD6：FusedMultiBranchSelectedLogitKernel

Compute selected logits for all branches in one fused branch dimension:

```text
branch dimension = RealFunctional / AdamWParallel / bestLR / NoOp
class dimension = selected logits only
```

### SLD7：HybridSelectedExactBorderline

Pipeline:

```text
selected-logit delta for candidate generation
→ exact delta-view only for borderline rows
```

---

## 4. Candidate generator features

### CG0：v9.2.51 CR3 reference

Reference only.

### CG1：SelectedLogitRecall

Use selected-logit gap / margin approximation as recall signal.

### CG2：RiskTailLCB

$$
S_{\text{risk}}
=
CEp99
+
WrongConfidenceP95
-
MarginP10
+
Uncertainty
+
Curvature.
$$

### CG3：RoleBranchMass

```text
branch_ratio_tail
effective_derivative
tail_activation_mass
functional_step_norm / task_step_norm
role_gap
```

### CG4：FamilyReliability

Family must not include dataset name:

$$
family(e)
=
(stratum,horizon,risk\_bucket,role\_bucket,attach\_type,branch\_bucket,probe\_bucket).
$$

Reliability:

$$
Rel(f)
=
\mathbb{E}[Y_{\text{safe-good}}\mid f]
-
\kappa\sqrt{\operatorname{Var}(Y_{\text{safe-good}}\mid f)}.
$$

### CG5：SupportDensity

$$
Density(e)
=
\frac{k}{N\cdot Volume(\mathcal{N}_k(e))}.
$$

### CG6：HybridRecall

$$
S_{\text{candidate}}
=
a_1S_{\text{selected-gap}}
+
a_2S_{\text{risk-safe}}
+
a_3Rel(f)
+
a_4Density
+
a_5S_{\text{role-branch}}.
$$

Coefficient selection uses calibration split only, no dataset name.

---

## 5. Controller candidates

### C0：v9.2.51 reference

Reference only.

### C1：CheapRecallThenFusedDelta

$$
Accept(e)
=
CheapRecall(e)
\land
FusedDeltaConfirm(e)
\land
RiskSafe(e).
$$

### C2：SelectedLogitCascade

$$
Accept(e)
=
SelectedLogitRecall(e)
\land
SelectedLogitGapConfirm(e)
\land
RiskSafe(e)
\land
SupportStable(e).
$$

### C3：FamilyBalancedSelectedDelta

$$
Accept(e)
=
Candidate(e)
\land
SelectedDeltaConfirm(e)
\land
RiskSafe(e)
\land
Rel(family(e))\geq r_0
\land
FamilyBalance(e).
$$

### C4：HybridExactBorderlineController

$$
Accept(e)
=
HighConfidenceSelectedAccept(e)
\lor
[
BorderlineSelected(e)
\land
ExactDeltaViewConfirm(e)
].
$$

### C5：FusedMultiBranchController

Uses SLD6 fused multi-branch selected-logit kernel:

$$
Accept(e)
=
FusedMultiBranchGap(e)
\land
RiskSafe(e)
\land
SupportBalanced(e).
$$

### C6：ParetoCostAwareController

Accept if event lies on Pareto frontier of:

```text
low delta-prep cost
positive selected gap
low risk
high support density
family balance
system envelope
```

### C7：Oracle

Posthoc diagnostic only. Never official.

---

# Part VI. 实验阶段

## P0：v9.2.51 boundary reproduction

### 目标

确认 v9.2.51 boundary 稳定。

### 必须记录

```text
route
source_route_v9250
dominant_m0_subphase
branch_logit_path_ratio
best_branch_delta_id
branch_delta_auc
branch_delta_agreement
branch_delta_step_ratio
cheap_recall_pass
safe_good_recall
candidate_rate
candidate_bad_event
cascade_controller_pass
controller_precision
controller_coverage
controller_bad_event
oracle_support_pass
oracle_precision
oracle_coverage
oracle_bad_event
measured_signal_strata_count
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R9-BranchDeltaPredictiveButTooExpensive
branch_delta_predictivity_pass = 1
branch_delta_system_pass = 0
oracle support pass = 1
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_branch_delta_signal_vs_system_ladder.svg
p0_oracle_high_system_low.svg
```

---

## P1：L1 RealFunctional delta-prep subphase attribution

### 目标

把 L1 `RealFunctional delta preparation` 拆开，确认真实瓶颈。

### 必须记录

```text
row_id
subphase_id
subphase_name
time_ms
time_ratio
read_MB
write_MB
temp_alloc_MB
kernel_count
sync_count
delta_tensor_count
delta_buffer_MB
branch_count
role_count
workspace_reuse_used
unknown_fraction
```

Subphases：

```text
D0-role score / controller scalar preparation
D1-functional delta coefficient compute
D2-delta tensor assembly
D3-role-wise delta packing
D4-branch scale application
D5-parameter delta-view setup
D6-device temporary allocation
D7-host sync / logging
```

### 判断标准

P1 pass：

```text
unknown_fraction <= 0.10
dominant_delta_prep_subphase_identified = 1
L1 attribution sums to L1 within ±0.05
```

### 可视化

```text
p1_l1_delta_prep_waterfall.svg
p1_delta_prep_memory_traffic.svg
p1_delta_tensor_count_vs_time.svg
```

---

## P2：Fused delta-prep kernel matrix

### 目标

并行测试 FDP1-FDP6，降低 L1 成本。

### 必须记录

```text
fused_delta_id
candidate_status
delta_correctness_relerr
delta_correctness_cos
delta_prep_time_ratio_vs_ref
delta_prep_memory_ratio_vs_ref
per_probe_overhead_q90
step_ratio_q90
memory_ratio
read_MB
write_MB
kernel_count
sync_count
uses_dataset_name
uses_validation
uses_test
uses_posthoc_commit
```

### 判断标准

Fused delta-prep pass：

$$
T_{\Delta\theta_F,\text{fused}}
\leq0.35T_{\Delta\theta_F,\text{ref}}.
$$

and:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Correctness for exact candidate：

$$
RelErr_{\Delta\theta}\leq10^{-4},
$$

$$
Cos_{\Delta\theta}\geq0.999.
$$

### 可视化

```text
p2_delta_prep_cost_pareto.svg
p2_delta_correctness.svg
p2_delta_workspace_reuse.svg
```

---

## P3：Selected-logit branch delta matrix

### 目标

并行测试 SLD1-SLD7，判断 selected-logit 是否能保留 exact gap signal。

### 必须记录

```text
selected_delta_id
selected_class_count
classes_used
uses_full_logits
AUC_safe_good
corr_safe_grounded
agreement_exact_accept
gap_error_mean
gap_error_p95
CE_error
margin_error
risk_error
precision_at_gate
coverage_at_gate
bad_event_at_gate
step_ratio_q90
memory_ratio
```

### 判断标准

Official selected delta pass：

$$
AUC_{\text{safe-good}}\geq0.70
$$

or:

$$
Corr_{\text{safe-grounded}}\geq0.35.
$$

and:

$$
Agreement_{\text{accept}}\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Diagnostic pass：

$$
AUC_{\text{safe-good}}\geq0.60
$$

and:

$$
Agreement_{\text{accept}}\geq0.80.
$$

### 可视化

```text
p3_selected_delta_auc_cost_pareto.svg
p3_selected_delta_agreement.svg
p3_selected_class_count_ablation.svg
p3_selected_delta_bad_event_curve.svg
```

---

## P4：High-recall candidate generator

### 目标

避免 v9.2.51 cheap recall 低召回。Candidate generator 不直接 official accept，只负责把 safe-good events 送入 branch-delta confirm。

### 必须记录

```text
candidate_generator_id
features_used
calibration_split_id
heldout_split_id
recall_safe_good_cal
recall_safe_good_heldout
candidate_rate
candidate_bad_event_rate
candidate_precision
candidate_coverage
accepted_strata_candidate_count
accepted_family_candidate_count
feature_overhead
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

Candidate generator pass：

$$
Recall_{\text{safe-good}}\geq0.80,
$$

$$
CandidateRate\leq0.35,
$$

$$
CandidateBadEventRate\leq0.20.
$$

### 可视化

```text
p4_candidate_recall_candidate_rate.svg
p4_candidate_bad_event_curve.svg
p4_candidate_feature_ablation.svg
p4_candidate_family_coverage.svg
```

---

## P5：Coverage-preserving cascade controller

### 目标

用 candidate discovery + fused/selected branch-delta confirmation + risk/support gates 建立 official local controller。

### Split design

```text
calibration split:
  choose thresholds / coefficients

heldout natural split:
  official local controller gate

balanced diagnostic split:
  support/failure analysis only

leave-dataset-out:
  P7

leave-stratum-out:
  P7
```

### 必须记录

```text
controller_id
fused_delta_id
selected_delta_id
candidate_generator_id
features_used
thresholds
coefficients
calibration_split_id
heldout_split_id
precision_cal
coverage_cal
bad_event_cal
precision_heldout
coverage_heldout
bad_event_heldout
AUC_heldout
corr_heldout
safe_good_recall
accepted_strata_count
accepted_family_count
max_family_share
amortized_overhead
step_q90
memory_ratio
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
official_eligible
```

### 判断标准

Controller pass：

$$
AUC_{\text{heldout}}\geq0.70
$$

or:

$$
Corr_{\text{heldout}}\geq0.35.
$$

and:

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

### 可视化

```text
p5_controller_precision_coverage_bad.svg
p5_controller_cost_vs_value.svg
p5_controller_family_coverage.svg
p5_cascade_stage_sankey.svg
p5_oracle_legal_gap.svg
```

---

## P6：Online support and stratum expansion

### 目标

扩大 natural / balanced support，避免 measured support 太窄。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
horizons = 20,80,240,640
signal_strata = S1-S8
carriers = A0,A3,A6
row_sources = natural, balanced_diagnostic
```

### 必须记录

```text
row_source
row_id
dataset
seed
horizon
signal_stratum
event_family
carrier_id
fused_delta_id
selected_delta_id
safe_good
bad_event
oracle_accept
controller_accept
risk_safe
value_positive
control_resistant
feature_values
```

### 判断标准

Measurement pass：

```text
natural_real_event_count >= 12000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 6
measured_family_count >= 12
```

Official accepted support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

### 可视化

```text
p6_signal_strata_coverage.svg
p6_family_support_heatmap.svg
p6_oracle_support_by_stratum.svg
p6_natural_vs_balanced_distribution.svg
```

---

## P7：Leave-dataset-out / leave-stratum-out

### 目标

证明 controller 不是 dataset-specific 或 single-stratum overfit。

### 设置

Leave-dataset-out：

```text
calibrate on MNIST + Fashion, evaluate KMNIST
calibrate on MNIST + KMNIST, evaluate Fashion
calibrate on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
calibrate on all but one signal stratum
evaluate held-out stratum
```

### 必须记录

```text
split_type
heldout
controller_id
fused_delta_id
selected_delta_id
threshold
precision
coverage
bad_event_rate
task_safe
CEp99_delta
margin_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
dataset_name_used
shuffle_control_pass
```

### 判断标准

LDO pass：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two held-out datasets：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

LSO pass：

At least $70\%$ held-out strata task-safe and：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

### 可视化

```text
p7_leave_dataset_out_matrix.svg
p7_leave_stratum_out_matrix.svg
p7_hidden_dataset_tuning_audit.svg
p7_leaveout_failure_modes.svg
```

---

## P8：Official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
controller_id = best P7 survivor
fused_delta_id = best P7 survivor
selected_delta_id = best P7 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledFusedDelta, ShuffledSelectedDelta, ShuffledCandidateRecall,
           ShuffledRiskScore, ShuffledBranchRatio, ShuffledControlGap,
           ShuffledValueScore, ShuffledSignalChannel,
           FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled,
           DatasetRouteShuffled, EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
controller_id
fused_delta_id
selected_delta_id
dataset
seed
horizon
signal_stratum
event_family
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_random
real_beats_noop
task_safe
event_count
coverage
bad_event_rate
step_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Paired replay pass：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{each slice,Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Shuffle controls fail：

```text
ShuffledFusedDelta = fail
ShuffledSelectedDelta = fail
ShuffledCandidateRecall = fail
ShuffledRiskScore = fail
ShuffledBranchRatio = fail
ShuffledControlGap = fail
ShuffledValueScore = fail
ShuffledSignalChannel = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p8_official_paired_replay_pareto.svg
p8_macro_beat_rate.svg
p8_signal_stratum_win_matrix.svg
p8_shuffle_control_matrix.svg
p8_system_gate_distribution.svg
```

---

## P9：Short-run scout

### 目标

如果 P8 pass，验证局部 paired replay 优势能否在连续训练中保持。

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledFusedDelta
```

### 必须记录

```text
candidate
dataset
seed
steps
train_loss
holdout_loss
val_acc_proxy
CEp99
margin_p10
ECE_proxy
NLL_proxy
curvature
local_lipschitz
basis_usage_entropy
functional_channel_usage_entropy
event_count
coverage
bad_event_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Task safety：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

Mechanism pass, at least one：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

or:

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

### 可视化

```text
p9_short_run_task_mechanism_pareto.svg
p9_short_run_controls.svg
p9_event_timeline.svg
p9_ce_tail_margin_panel.svg
```

---

## P10：Full 10-seed validation

### 目标

如果 short-run pass，验证 full functional route。

### 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledFusedDelta
```

### 必须记录

```text
candidate
dataset
seed
val_acc
test_acc
delta_vs_mlp
delta_vs_adamw
delta_vs_adamwparallel
delta_vs_bestlr
delta_vs_quadratic_feature_mlp
near_pass
full_pass
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature_ratio
local_lipschitz_ratio
basis_usage_entropy
functional_channel_usage_entropy
event_count
coverage
bad_event_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Task safety：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Macro improvement：

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003.
$$

Hard-stratum repair：

$$
\Delta Acc_{\text{hard-stratum,functional}}
-
\Delta Acc_{\text{hard-stratum,AdamW}}
\geq0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

---

## P11：Robustness / strong baseline / external-ready

### 目标

确认 functional advantage 不是 clean MNIST-family artifact。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
strong baselines = MLP-match, hidden-bracket MLP, QuadraticFeatureMLP, repaired LQ AdamW-only
```

### 判断标准

Robustness pass：

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{AdamW}}.
$$

Noise-tail pass：

$$
CEp99_{\text{functional}}\leq CEp99_{\text{AdamW}}.
$$

Strong baseline pass：

$$
Acc_{\text{functional}}\geq Acc_{\text{QuadraticFeatureMLP}}-0.005
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{QuadraticFeatureMLP}}.
$$

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9252.csv
p0_v9251_boundary_reproduction.csv
p1_l1_delta_prep_subphase_attribution.csv
p2_fused_delta_prep_kernel_matrix.csv
p3_selected_logit_branch_delta_matrix.csv
p4_high_recall_candidate_generator.csv
p5_coverage_preserving_cascade_controller.csv
p6_online_support_stratum_expansion.csv
p7_leave_dataset_and_stratum_out.csv
p8_official_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_10seed_functional_validation.csv
p11_robustness_external_ready.csv
l1_delta_prep_trace_v9252.csv
fused_delta_prep_trace_v9252.csv
selected_logit_delta_trace_v9252.csv
candidate_generator_trace_v9252.csv
cascade_controller_trace_v9252.csv
support_density_trace_v9252.csv
leaveout_trace_v9252.csv
paired_replay_branch_trace_v9252.csv
system_fused_delta_overhead_trace_v9252.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9251_boundary_unstable
F3_dataset_tuning_detected
F4_l1_subphase_unattributed
F5_fused_delta_prep_not_correct
F6_fused_delta_prep_not_system_legal
F7_selected_logit_delta_not_predictive
F8_selected_logit_delta_agreement_fail
F9_candidate_generator_low_recall
F10_candidate_generator_high_bad_event
F11_cascade_precision_fail
F12_cascade_coverage_fail
F13_cascade_bad_event_fail
F14_support_measurement_too_narrow
F15_oracle_support_collapse
F16_leave_dataset_out_fail
F17_leave_stratum_out_fail
F18_paired_replay_control_equivalent
F19_shuffle_control_pass
F20_functional_lr_equivalent
F21_short_run_task_drop
F22_full_run_no_macro_hard_stratum_gain
F23_strong_baseline_explains_gain
F24_robustness_fail
F25_external_not_ready
F26_fake_or_proxy_violation
F27_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-L1DeltaPrepAttributed:
  L1 RealFunctional delta-prep cost is fully attributed.

R2-FusedDeltaPrepSystemPass:
  fused delta-prep reduces cost and passes system envelope.

R3-SelectedLogitDeltaPredictive:
  selected-logit branch delta preserves safe-good signal.

R4-SelectedLogitDeltaSystemPass:
  selected-logit branch delta passes system envelope.

R5-CandidateGeneratorPass:
  high-recall candidate generator finds enough safe-good candidates.

R6-CascadeControllerPass:
  coverage-preserving cascade passes heldout precision / coverage / bad-event / system gate.

R7-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R8-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R9-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R10-FusedDeltaPredictiveButTooExpensive:
  fused delta/selected delta predicts safe-good but remains outside system envelope.

R11-SelectedDeltaCheapButUninformative:
  selected-logit path is cheap but loses exact gap-probe signal.

R12-CandidateDiscoveryFailOracleHigh:
  oracle support exists but candidate generator cannot find enough events.

R13-ControllerStillCoverageLimited:
  precision and safety are acceptable but coverage remains below 0.03.

R14-ControllerUnsafe:
  coverage exists but bad-event remains above 0.05.

R15-OracleSupportCollapse:
  fresh natural replay no longer has enough safe-good support.

R16-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R17-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R18-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9251_boundary_pass
dataset_tuning_detected
l1_subphase_attribution_pass
dominant_l1_subphase
best_fused_delta_id
fused_delta_correctness_pass
fused_delta_system_pass
fused_delta_step_ratio_q90
fused_delta_memory_ratio
best_selected_delta_id
selected_delta_predictivity_pass
selected_delta_system_pass
selected_delta_auc
selected_delta_corr
selected_delta_accept_agreement
best_candidate_generator_id
candidate_generator_pass
safe_good_recall
candidate_rate
candidate_bad_event
best_controller_id
cascade_controller_pass
controller_auc
controller_corr
accepted_precision
accepted_coverage
accepted_bad_event_rate
accepted_signal_strata_count
accepted_family_count
max_family_share
oracle_support_pass
oracle_precision
oracle_coverage
oracle_bad_event
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9252_strict_purekan_functional
success_v9252_full_functional
success_v9252_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 L1 delta-prep subphase attribution
  P2 fused delta-prep kernel matrix
  P3 selected-logit branch delta matrix
  P4 high-recall candidate generator

Batch 2:
  P5 coverage-preserving cascade controller
  P6 online support / stratum expansion
  P7 leave-dataset-out / leave-stratum-out
  P8 paired replay scout

Batch 3:
  official P8 paired replay
  P9 short-run if paired replay passes

Batch 4:
  P10 full 10-seed
  P11 robustness / strong baseline
```

Gate rule：

```text
P5/P8 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  fused / selected branch delta predictivity pass
  branch delta system gate pass
  cascade controller pass
  LDO/LSO pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.51 boundary reproduced
L1 delta-prep subphase attribution completed
fused delta-prep matrix measured
selected-logit branch delta matrix measured
candidate generator measured
support expansion measured
no fake/proxy/offload/loss/teacher violation
```

## Branch-delta system success

```text
Minimum diagnostic success
+
at least one fused/selected branch-delta path predicts safe-good
+
system overhead gate pass
+
accept agreement pass
```

## Legal controller success

```text
Branch-delta system success
+
candidate generator or direct branch-delta scan finds sufficient candidates
+
cascade controller heldout pass
+
multi-stratum / multi-family pass
```

## Local functional success

```text
Legal controller success
+
LDO / LSO pass
+
official paired replay beats AdamWParallel / bestLR
```

## Full functional success

```text
Local functional success
+
short/full run task-safe mechanism gain
+
macro or hard-stratum improvement
```

## Failure stop

```text
1. v9.2.51 boundary cannot be reproduced；
2. L1 cost cannot be attributed；
3. fused delta-prep cannot reduce cost；
4. all selected-logit branch deltas lose safe-good predictivity；
5. selected-logit branch delta remains too expensive；
6. candidate generator cannot find enough safe-good events；
7. cascade controller cannot simultaneously meet precision / coverage / bad-event；
8. online support remains too narrow；
9. fresh natural oracle support collapses；
10. leave-dataset-out fails；
11. leave-stratum-out fails；
12. paired replay remains control-equivalent；
13. shuffle controls pass, indicating overfit；
14. short-run task drops；
15. full run gives no macro / hard-stratum / geometry gain；
16. functional breaks system gate；
17. gains are explained by QuadraticFeatureMLP；
18. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：fused delta-prep + selected-logit controller + LDO/LSO + paired replay pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation and external robustness。

## Case B：fused delta-prep works but selected-logit loses signal

必须声明：

```text
system cost can be reduced, but selected-logit approximation loses exact gap information.
```

下一步回到 exact branch-delta kernelization or richer output-delta features。

## Case C：selected-logit signal works but system fail

必须声明：

```text
safe-good is observable through selected branch-output displacement, but implementation is still not system-legal.
```

下一步做 lower-level fused CUDA/Triton branch-delta kernel。

## Case D：candidate generator fails but oracle high

必须声明：

```text
good events exist, but legal candidate discovery cannot find them.
```

下一步设计 richer recall features; do not tune dataset。

## Case E：controller local pass but LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

## Case F：oracle support collapses

必须声明：

```text
fresh natural replay no longer has enough safe-good support.
```

下一步回到 carrier/support mechanism。

---

# Part XII. 最终建议

v9.2.52 的一句话策略是：

$$
\boxed{
\text{不要继续试 cheap score；先把 RealFunctional delta-prep 与 branch logits 做成 fused selected-logit system path。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
exact gap-probe 有没有 AUC；
BLD1 是否有 signal；
C4 threshold 是否差一点；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. L1 RealFunctional delta preparation 具体贵在哪里？
2. functional delta 能否一次生成、多 branch 复用？
3. selected-logit branch delta 是否能保留 exact gap-probe 的 safe-good signal？
4. selected-logit branch delta 是否能进入 step <= 1.50 / memory <= 1.05？
5. candidate generator 能否把 safe-good recall 拉到 >=0.80？
6. cascade controller 能否同时满足 precision / coverage / bad-event / system？
7. support 是否能扩展到 >=6 measured strata？
8. controller 能否 LDO/LSO？
9. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.52 的结果将给出清晰分叉：

```text
if fused/selected branch delta + cascade + LDO/LSO + paired replay pass:
  strict PureKAN functional obtains local causal evidence.

if fused delta predicts but too expensive:
  custom CUDA/Triton kernelization remains blocker.

if selected-logit path cheap but loses signal:
  exact branch-delta information cannot be compressed by current selected-logit approximation.

if candidate discovery fails:
  legal recall / sufficient-statistic design remains blocker.

if controller local pass but leave-out fails:
  no dataset tuning; broaden signal-stratum/family support.

if oracle support collapses:
  carrier/support stability is blocker.
```
