# DG-KAN v9.2.56 True Branch-Delta Tensor Interface 与 K7d Control-Gain CUDA Fusion 完整实验计划

> 本计划基于 v9.2.55 `Lower-Level CUDA Branch-Delta Extension 与 Exact-Signal Controller Closure` 的真实复盘制定。  
> v9.2.55 的 terminal route 是：
>
> ```text
> route = R9-TrueCustomExtensionNotImplemented
> base_candidate = LQ-t2-h256
> success_v9255_strict_purekan_functional = False
> success_v9255_full_functional = False
> success_v9255_external_ready = False
> ```
>
> v9.2.55 的关键事实是：
>
> ```text
> P0:
>   v9.2.54 boundary reproduced
>   source route = R14-NeedsLowerLevelCUDAExtension
>   source true custom branch-delta implemented = 0
>   oracle support pass = 1
>   fake/proxy/offload = 0
>
> P1:
>   K7 residual attribution pass = 1
>   dominant subphase = K7d-control_gain_compute
>   ratio = 0.8661336752717377
>
> P2:
>   CBD4-CUDABranchDeltaMetricFusedExtension compiled and invoked
>   step ratio = 1.0236406340263784
>   but uses source-measured scalar gap input
>   uses_true_branch_delta = 0
>   official pass = 0
>
>   CBD0-V9254Reference:
>     AUC = 0.8884008136827201
>     agreement = 1.0
>     step ratio = 2.8632798851361203
>     system pass = 0
>
>   CBD1 formula negative control:
>     step ratio = 1.35
>     AUC = 0.5349884272995514
>     agreement = 0.38682208994708994
>     official pass = 0
>
>   true_custom_branch_delta_implemented = 0
>
> P3:
>   exact-signal controller official_eligible = 0 for all controllers
>
> P4:
>   oracle precision = 1.0
>   oracle coverage = 0.12169312169312169
>   oracle bad-event = 0.0
>   measured signal strata = 3
>   support measurement pass = 0
>
> P5-P9:
>   all not_run because P2_true_custom_branch_delta_not_implemented
>
> current blocker:
>   true_branch_delta_extension_not_implemented
> ```
>
> 本轮最核心的信息是：
>
> $$
> \boxed{
> \text{CUDA extension toolchain 已打通，但我们还没有实现真正的 branch-delta logits tensor interface。}
> }
> $$
>
> 因此 v9.2.56 不再允许继续用 `source-measured gap input`、formula proxy、posthoc oracle、或 controller threshold 作为替代。  
> v9.2.56 必须直接回答：
>
> $$
> \boxed{
> \text{能否从真实 branch delta 计算 selected/full logits，再融合 control-gain reduction，并在 system gate 内保留 exact signal？}
> }
> $$

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

数据集只能作为诊断切片，不能作为 official controller 条件。允许报告不同数据集上的 failure mode、support distribution、bad-event、coverage、LDO 矩阵；不允许使用 dataset name 选择 kernel、controller、threshold、candidate generator 或 abstention policy。

---

# Part I. 对 v9.2.55 的独立判断

## 1. v9.2.55 没有达到目标

v9.2.55 没有 strict PureKAN functional success。它甚至还没有进入真正的 controller 阶段，因为最前面的 true branch-delta implementation gate 没过：

```text
true_custom_branch_delta_implemented = 0
exact_signal_controller_pass = 0
support_measurement_pass = 0
LDO / LSO / paired replay = not_run
```

这意味着不能声明：

```text
strict PureKAN local causal evidence
full functional success
external-ready success
Beyond-MLP success
```

虽然 `CBD4` 有 CUDA extension 编译与调用成功，并且 step ratio 低到 `1.0236406340263784`，但它使用的是 `source-measured scalar gap input`，不是从 functional branch delta 真实计算 logits 或 selected logits。因此它只能证明 **CUDA extension plumbing 可运行**，不能证明 **branch-delta kernel 闭合**。

## 2. v9.2.55 的真实进展

v9.2.55 仍有进展，主要有三点。

第一，K7 residual path 被进一步定位。v9.2.54 发现 K7 是 dominant residual；v9.2.55 进一步把 K7 内部定位到：

```text
K7d-control_gain_compute
ratio = 0.8661336752717377
```

这说明真正要 fuse 的不是泛泛的 logits，也不是泛泛的 metric，而是 **control-gain computation**：

```text
RealFunctional gain
AdamWParallel gain
bestLR gain
max-control gain
gap = Real - max(control)
risk / support accept components
```

第二，CUDA extension 编译/调用路径已经打通。环境补齐 `ninja` 后，`CBD4-CUDABranchDeltaMetricFusedExtension` 可运行，且 step ratio 很低。这说明下一步不是“完全不知道如何写 extension”，而是要把 extension 的输入从 source-measured scalar gap 换成真实 branch-delta logits / selected logits / full logits tensor interface。

第三，exact signal 仍然强。`CBD0-V9254Reference` 的 AUC 仍为 `0.8884008136827201`，accept agreement 为 `1.0`。oracle support 也仍然强：precision `1.0`，coverage `0.12169312169312169`，bad-event `0.0`。这说明 safe-good events 没有消失，value signal 也没有消失。

## 3. 当前真正 blocker

当前 blocker 不是：

```text
base 不稳；
basis 不行；
attach 污染；
carrier silent；
safe-good events 消失；
exact branch-delta 没有信号；
CUDA extension 完全无法编译；
controller threshold 差一点。
```

当前 blocker 是：

$$
\boxed{
\text{缺少 true branch-delta tensor interface。}
}
$$

更具体地说，当前实现缺少以下链条：

```text
functional delta state
→ branch delta state
→ selected/full branch logits
→ Real / AdamWParallel / bestLR gains
→ control gap
→ risk/support accept components
```

现在的 `CBD4` 只是在后半段做 metric extension，它没有真实计算 branch logits；它吃的是 source-measured scalar gap，所以不能 official。  
`CBD1` 则证明 formula shortcut 不够：虽然 step ratio `1.35`，但 AUC `0.534988`、agreement `0.386822`，丢了 exact signal。

## 4. 为什么进展仍显慢

进展慢是真实的，但不是因为研究方向随机。最近几轮的 blocker 逐步收缩：

```text
v9.2.48:
  online gap_probe 实现，AUC = 0.884406，但 step ratio = 6.121495。

v9.2.49:
  cost attribution 到 F7 metric computation。

v9.2.50:
  F7 内部定位到 M0-logit preparation。

v9.2.51:
  M0 内部定位到 branch-logit path；
  exact BLD1 AUC = 0.886903，但 step ratio = 4.700044。

v9.2.52:
  L1 内部定位到 D0 controller scalar preparation；
  SLD0 AUC = 0.888401，但 step ratio = 6.096370。

v9.2.53:
  D0 vectorization 成功；
  FBD6 AUC = 0.888401，agreement = 1.0；
  但 step ratio = 2.863280。

v9.2.54:
  residual dominant subphase 定位到 K7-gap_risk_support_reduction；
  formula shortcut 快但丢 signal；
  true custom extension 未实现。

v9.2.55:
  CUDA extension compile/call path 成功；
  K7d-control_gain_compute 成为 dominant；
  但 extension 仍未从 true branch delta 计算 logits / gain。
```

所以 v9.2.55 不是没有进展，而是暴露了更严厉的事实：**现在不是实验设计问题，而是实现问题。**  
继续生成更多 controller matrix，不会解决 `true_custom_branch_delta_implemented = 0`。

---

# Part II. v9.2.56 总体目标

v9.2.56 的总体目标是：

$$
\boxed{
\text{实现 true branch-delta tensor interface，融合 K7d control-gain compute，并打开 exact-signal controller / LDO / paired replay。}
}
$$

本轮分成六个必须同时推进的目标：

```text
1. 定义并验证 true branch-delta tensor interface；
2. 实现 CUDA/C++ 或 Triton-v2 true branch-delta logits kernel；
3. 在 kernel 内或 extension 内融合 K7d control-gain compute；
4. 用 branch-delta primary signal 建立 exact-signal controller；
5. 扩大 support measurement；
6. 通过 LDO/LSO 和 official paired replay。
```

如果第 1/2 项不能实现，本轮必须 stop-go，不允许继续用 formula proxy 或 source-measured gap 来推进 downstream。

---

## 1. True branch-delta tensor interface success

v9.2.56 必须先定义一个不可含糊的接口。对每个 event $e$ 和 branch $b$：

$$
b\in
\{
RealFunctional,
AdamWParallel,
bestLR,
NoOp
\}.
$$

输入必须来自 commit-time train-stream：

```text
base activations / base logits
functional delta state
AdamW-equivalent delta state
bestLR delta state
selected class ids
labels
risk/support state
```

输出必须是真实 branch-delta logits 或 selected logits：

$$
\widehat z_{b,\mathcal{C}}
=
z_{0,\mathcal{C}}
+
\widehat{\Delta z}_{b,\mathcal{C}}.
$$

其中 $\mathcal{C}$ 可以是：

```text
full logits:
  all C classes

selected logits:
  true class
  current top-1
  current top-2
  hard negative
  optional tail class
```

True interface pass 的必要条件：

```text
uses_true_branch_delta = 1
uses_formula_proxy = 0
uses_source_measured_gap = 0
uses_posthoc_replay = 0
uses_dataset_name = 0
commit_time_order_valid = 1
```

Correctness gate：

$$
Agreement_{\text{accept}}\geq0.90.
$$

Predictivity gate：

$$
AUC(S_{\text{delta}},Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(S_{\text{delta}},V_{\text{safe-grounded}})\geq0.35.
$$

System gate：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

---

## 2. K7d control-gain fusion success

v9.2.55 显示 K7d-control_gain_compute 占 K7 residual ratio `0.8661336752717377`。因此 v9.2.56 的 kernel 不能只返回 logits，然后再回到 Python/Torch 做 control gain。它必须至少输出 compact branch gain components：

```text
gain_real
gain_adamwparallel
gain_bestlr
max_control_gain
control_gap
risk_score
support_score
accept_components
```

定义：

$$
Gain_b
=
-\Delta CE_b
+
\lambda_m\Delta Margin_b
-
\lambda_r Risk_b.
$$

控制差距：

$$
Gap(e)
=
Gain_{\text{RealFunctional}}(e)
-
\max(
Gain_{\text{AdamWParallel}}(e),
Gain_{\text{bestLR}}(e)
).
$$

K7d fusion pass：

$$
T_{\text{K7d,fused}}\leq0.35T_{\text{K7d,ref}}.
$$

End-to-end pass：

$$
StepRatio_{q90}\leq1.50.
$$

如果 K7d fusion pass 但 full step 仍失败，必须继续归因 residual phases，不允许直接说 controller 失败。

---

## 3. Exact-signal controller success

如果 true branch-delta system gate 通过，official controller 不再让 weak cheap candidate generator 主导。controller 以 branch-delta signal 为 primary signal：

$$
Accept(e)
=
BranchDeltaConfirm(e)
\land
RiskSafe(e)
\land
SupportBalanced(e).
$$

其中：

$$
BranchDeltaConfirm(e)
=
\mathbb{1}[Gap(e)\geq\tau_g],
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

Family 不能包含 dataset name：

$$
family(e)
=
(stratum,horizon,risk\_bucket,role\_bucket,attach\_type,branch\_bucket,probe\_bucket).
$$

Heldout controller pass：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

Support pass：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

If custom branch-delta is cheap enough, candidate gate may be all-pass:

$$
Candidate(e)=1.
$$

If branch-delta is still moderately expensive but system-legal only under sparse calls, candidate gate may be broad-pass, but must satisfy:

$$
Recall_{\text{safe-good}}\geq0.80,
$$

$$
CandidateRate\leq0.35,
$$

$$
CandidateBadEventRate\leq0.20.
$$

---

## 4. Support expansion success

v9.2.55 still has measured signal strata `3` and balanced diagnostic rows `36`。v9.2.56 必须同时扩大 support，否则即使 local controller pass，也不能说明 route 已经 dataset/stratum-agnostic。

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
  official controller / leave-out / paired replay

balanced diagnostic rows:
  failure analysis / support analysis only
```

Balanced rows 不得直接混入 official heldout pass，除非显式分离 calibration / heldout / diagnostic roles。

---

## 5. Leave-out success

Leave-dataset-out：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

至少两个 held-out datasets 满足：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

Leave-stratum-out：

至少 $70\%$ held-out strata task-safe and：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

---

## 6. Official paired replay success

Only after true branch-delta system pass + controller pass + LDO/LSO pass.

Paired replay pass：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{Real,slice}}\geq Acc_{\text{AdamW,slice}}-0.005.
$$

Shuffle controls must fail：

```text
ShuffledTrueBranchDelta = fail
ShuffledControlGain = fail
ShuffledRiskScore = fail
ShuffledSupportBalance = fail
ShuffledCandidateGate = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

---

# Part III. 核心假设

## H1：v9.2.55 的失败是 true tensor interface 缺失，不是 value signal 消失

证据：

```text
CBD0 AUC = 0.8884008136827201
CBD0 agreement = 1.0
oracle precision = 1.0
oracle coverage = 0.12169312169312169
oracle bad-event = 0.0
true_custom_branch_delta_implemented = 0
```

H1 成立标准：

true branch-delta interface implemented 后，至少一个 candidate 达到：

$$
AUC_{\text{safe-good}}\geq0.70
$$

or:

$$
Corr_{\text{safe-grounded}}\geq0.35,
$$

with:

$$
Agreement_{\text{accept}}\geq0.90.
$$

H1 失败标准：

所有 true branch-delta candidates：

```text
AUC < 0.60
or agreement < 0.80
```

如果 H1 失败，说明 exact reference signal 无法被当前 lower-level implementation 保留，下一步需要重新设计 output-delta sufficient statistics。

---

## H2：K7d control-gain compute 是当前主系统瓶颈

证据：

```text
K7d-control_gain_compute ratio = 0.8661336752717377
```

H2 成立标准：

K7d fusion 后：

$$
T_{\text{K7d,fused}}\leq0.35T_{\text{K7d,ref}},
$$

且 end-to-end：

$$
StepRatio_{q90}\leq1.50.
$$

H2 失败标准：

K7d fusion 后仍：

```text
K7d > 50% residual
or total step ratio > 2.00
```

则需要更底层 CUDA/C++ optimization 或重构 branch delta representation。

---

## H3：formula proxy 是 negative control，不可 official promotion

证据：

```text
CBD1 step ratio = 1.35
CBD1 AUC = 0.5349884272995514
CBD1 agreement = 0.38682208994708994
```

H3 成立标准：

所有 formula proxy rows 必须明确：

```text
uses_formula_proxy = 1
official_eligible = 0
```

除非它同时具有 true branch-delta correctness。  
任何把 formula proxy speed 当作 official success 的 route 都应判定为 invalid。

---

## H4：如果 true branch-delta 足够便宜，candidate generator 不应成为 blocker

v9.2.49-v9.2.55 多轮显示 cheap candidate / selected proxy recall 不稳定。H4 认为，若 true branch-delta step gate 过，可以直接采用 all-pass or broad-pass controller。

H4 成立标准：

All-pass branch-delta controller achieves：

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

H4 失败标准：

All-pass branch-delta controller high bad-event or low precision.  
此时 candidate/risk/support filters 仍必要，但不能 dataset-specific。

---

## H5：oracle support 仍存在；不回到 carrier reset，除非 fresh oracle collapse

H5 成立标准：

fresh v9.2.56 natural rows：

$$
OraclePrecision_{\text{safe-good}}\geq0.75,
$$

$$
OracleCoverage_{\text{safe-good}}\in[0.03,0.15],
$$

$$
OracleBadEventRate\leq0.05.
$$

H5 失败标准：

```text
fresh oracle precision < 0.75
or coverage < 0.03
or bad-event > 0.05
```

如果 H5 失败，下一步回到 carrier/support design。  
如果 H5 成立但 controller fail，继续 legal/system/controller route。

---

# Part IV. 并行执行设计

v9.2.56 runner 必须并行执行以下 lanes：

```text
Lane A:
  v9.2.55 boundary reproduction

Lane B:
  True branch-delta tensor interface implementation:
    selected logits
    full logits small-C
    branch dimension
    no source-measured gap input

Lane C:
  K7d control-gain fusion:
    real/control gain computation
    gap/risk/support reduction
    device-side accept components

Lane D:
  Formula proxy negative-control audit:
    CBD1/CBD4-like paths remain diagnostic only

Lane E:
  Exact-signal controller:
    branch-delta primary
    risk-safe
    support-balanced
    optional broad candidate

Lane F:
  Support expansion:
    natural rows
    balanced diagnostic rows
    strata / family coverage

Lane G:
  LDO / LSO

Lane H:
  paired replay scout and official replay

Lane I:
  short-run scout if paired replay passes

Lane J:
  system audit:
    kernel count
    sync count
    read/write MB
    temp allocation
    CUDA extension compile status
    correctness / agreement trace
```

Gate discipline：

```text
diagnostic rows may be measured early
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  uses_true_branch_delta = 1
  uses_source_measured_gap = 0
  uses_formula_proxy = 0
  custom branch-delta predictivity pass
  custom branch-delta agreement pass
  custom branch-delta system pass
  exact-signal controller pass
  LDO/LSO pass
```

---

# Part V. Candidate designs

## 1. Base and carrier

Official base remains：

```text
R2-LQ-fanin-output-scale-confirmed
LQ-t2-h256
```

Carrier candidates：

```text
A0-current-v9255
A3-RoleWiseFT7ResetCarrier
A6-HybridRoleControlRiskCarrier
```

A1 risk-bounded carrier remains diagnostic unless carrier-active gate passes：

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

---

## 2. True branch-delta interface candidates

### TBD0：CBD0 reference

Reference only. Predictive but too expensive.

```text
uses_true_branch_delta = reference_exact
uses_source_measured_gap = 0
official_eligible = 0 unless system pass
```

### TBD1：SelectedLogitTrueBranchDelta

Computes selected branch logits from branch delta state:

```text
true class
top-1
top-2
hard negative
tail class
```

Outputs selected logits for:

```text
RealFunctional
AdamWParallel
bestLR
NoOp
```

### TBD2：FullLogitSmallCTrueBranchDelta

For 10-class datasets, computes all logits. It is allowed as current FC validation, but must be labeled small-C specialization.

### TBD3：TrueBranchDeltaMetricFused

Computes true branch logits and K7d gain components in one extension:

```text
gain_real
gain_adamwparallel
gain_bestlr
control_gap
risk_score
support_score
accept_components
```

### TBD4：TrueBranchDeltaWithExactFallback

Selected logits first; full logits only for borderline rows:

$$
|Gap_{\text{selected}}-\tau_g|<\epsilon_b.
$$

### TBD5：TritonV2TrueBranchDelta

Allowed only if it computes true branch delta. Must not use source-measured scalar gap.

### TBD6：CUDAExtensionTrueBranchDelta

C++/CUDA implementation. This is the preferred official target if Triton cannot represent the needed branch-delta path faithfully.

---

## 3. K7d fusion candidates

### K7F0：Reference K7d

Reference only.

### K7F1：GainOnlyFusion

Fuse:

```text
real gain
control gains
max control gain
control gap
```

### K7F2：GainRiskFusion

Fuse gain + risk score.

### K7F3：GainRiskSupportFusion

Fuse gain + risk + support / family ids.

### K7F4：AcceptComponentFusion

Kernel outputs accept components directly:

```text
branch_delta_confirm
risk_safe
support_balanced
candidate_accept
```

### K7F5：DelayedBulkLogging

No per-row logging / hash / timestamp during kernel path. Bulk flush only after block.

### K7F6：FullK7dFused

Combines K7F1-K7F5.

---

## 4. Controller candidates

### C0：v9.2.55 reference

Diagnostic only.

### C1：AllPassExactSignalController

No candidate gate:

$$
Accept(e)
=
BranchDeltaConfirm(e)
\land
RiskSafe(e)
\land
SupportBalanced(e).
$$

### C2：BroadCandidateExactSignalController

Only removes obviously inactive/unsafe rows:

$$
Accept(e)
=
BroadCandidate(e)
\land
BranchDeltaConfirm(e)
\land
RiskSafe(e)
\land
SupportBalanced(e).
$$

### C3：FamilyBalancedExactSignalController

$$
Accept(e)
=
BranchDeltaConfirm(e)
\land
RiskSafe(e)
\land
Rel(family(e))\geq r_0
\land
FamilyBalance(e).
$$

### C4：BorderlineFallbackExactSignalController

$$
Accept(e)
=
HighConfidenceDelta(e)
\lor
[
Borderline(e)
\land
ExactFallback(e)
].
$$

### C5：Oracle

Posthoc diagnostic only. Never official.

---

# Part VI. 实验阶段

## P0：v9.2.55 boundary reproduction

### 目标

确认 v9.2.55 boundary 稳定，避免把 implementation gap 误读为数据波动。

### 必须记录

```text
route
source_route_v9254
k7_residual_attribution_pass
dominant_k7_subphase
CBD0_AUC
CBD0_agreement
CBD0_step_ratio
CBD1_AUC
CBD1_agreement
CBD1_step_ratio
CBD4_step_ratio
CBD4_uses_true_branch_delta
true_custom_branch_delta_implemented
oracle_support_pass
oracle_precision
oracle_coverage
oracle_bad_event
support_measurement_pass
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R9-TrueCustomExtensionNotImplemented
K7d-control_gain_compute dominant
CBD0 exact signal retained
CBD1 formula negative control fail
CBD4 fast but not true branch-delta
oracle support pass = 1
fake/proxy/offload = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_exact_signal_vs_proxy_speed.svg
p0_route_ladder.svg
```

---

## P1：true branch-delta tensor interface gate

### 目标

本阶段不是测 controller，而是验证接口本身是否真实。必须阻止 `source-measured gap input` 和 formula proxy 混入 official path。

### 必须记录

```text
candidate_id
implementation_status
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
uses_selected_logits
uses_full_logits
uses_full_branch_forward
uses_autograd_graph
uses_dataset_name
uses_validation
uses_test
uses_posthoc_replay
commit_time_order_valid
input_tensor_shapes
output_tensor_shapes
branch_count
class_count
selected_class_count
```

### 判断标准

P1 pass：

```text
implementation_status = implemented
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
uses_posthoc_replay = 0
commit_time_order_valid = 1
```

If no candidate passes P1, downstream remains not_run.

### 可视化

```text
p1_interface_legality_matrix.svg
p1_tensor_interface_flow.svg
p1_proxy_rejection_table.svg
```

---

## P2：K7d control-gain subphase attribution and fusion

### 目标

把 K7d-control_gain_compute 拆成真实的 score/reduction operations，并测试 K7F candidates。

### 必须记录

```text
row_id
candidate_id
subphase_id
subphase_name
time_ms
time_ratio
read_MB
write_MB
temp_alloc_MB
kernel_count
sync_count
branch_count
class_count
selected_class_count
host_item_count
python_dispatch_count
unknown_fraction
```

Subphases：

```text
K7d0-branch_gain_prepare
K7d1-real_gain_compute
K7d2-adamwparallel_gain_compute
K7d3-bestlr_gain_compute
K7d4-max_control_gain
K7d5-control_gap_compute
K7d6-risk_score_compute
K7d7-support_score_compute
K7d8-accept_component_construct
K7d9-temp_allocation
K7d10-kernel_launch_sync
K7d11-logging_hash_timestamp
```

### 判断标准

Attribution pass：

```text
unknown_fraction <= 0.10
dominant_k7d_subphase_identified = 1
subphase sum within ±0.05 of K7d total
```

Fusion pass：

$$
T_{\text{K7d,fused}}\leq0.35T_{\text{K7d,ref}}.
$$

### 可视化

```text
p2_k7d_subphase_waterfall.svg
p2_k7d_kernel_count.svg
p2_k7d_memory_traffic.svg
p2_k7d_fusion_speedup.svg
```

---

## P3：true custom branch-delta implementation matrix

### 目标

并行测试 TBD1-TBD6。P3 是 v9.2.56 的核心 gate。

### 必须记录

```text
custom_delta_id
kernel_level
uses_cuda_extension
uses_triton
uses_true_branch_delta
uses_formula_proxy
uses_source_measured_gap
uses_selected_logits
uses_full_logits
uses_full_branch_forward
uses_autograd_graph
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
per_probe_overhead_q90
amortized_overhead
step_ratio_q50
step_ratio_q90
memory_ratio
read_MB
write_MB
kernel_count
sync_count
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

Implementation pass：

```text
uses_true_branch_delta = 1
uses_formula_proxy = 0
uses_source_measured_gap = 0
```

Predictivity pass：

$$
AUC_{\text{safe-good}}\geq0.70
$$

or:

$$
Corr_{\text{safe-grounded}}\geq0.35.
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

### 可视化

```text
p3_true_delta_auc_cost_pareto.svg
p3_true_delta_agreement.svg
p3_true_delta_error_distribution.svg
p3_formula_vs_true_delta.svg
p3_memory_traffic_kernel_count.svg
```

---

## P4：formula proxy negative-control audit

### 目标

确保 CBD1/CBD4-like shortcut 不会被误升为 official。

### 必须记录

```text
proxy_candidate_id
proxy_type
step_ratio_q90
AUC_safe_good
agreement_exact_accept
uses_true_branch_delta
uses_formula_proxy
uses_source_measured_gap
official_eligible
reason_not_official
```

### 判断标准

P4 pass：

```text
all formula/source-measured proxy rows official_eligible = 0
no proxy row can set success_v9256_strict_purekan_functional = 1
```

### 可视化

```text
p4_proxy_speed_signal_matrix.svg
p4_negative_control_summary.svg
```

---

## P5：exact-signal controller calibration

### 目标

如果 P3 true branch-delta system pass，建立 official exact-signal controller。

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
custom_delta_id
candidate_mode
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
p5_branch_delta_threshold_curve.svg
p5_oracle_legal_gap.svg
```

---

## P6：support expansion

### 目标

扩大 support measurement，避免只有 3 个 signal strata 的 narrow conclusion。

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
custom_delta_id
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
custom_delta_id
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
custom_delta_id = best P7 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledTrueBranchDelta, ShuffledControlGain, ShuffledRiskScore,
           ShuffledSupportBalance, ShuffledCandidateGate, ShuffledBranchRatio,
           ShuffledValueScore, ShuffledSignalChannel, FunctionalChannelShuffled,
           TailMaskShuffled, RoleScoreShuffled, DatasetRouteShuffled,
           EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
controller_id
custom_delta_id
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
ShuffledTrueBranchDelta = fail
ShuffledControlGain = fail
ShuffledRiskScore = fail
ShuffledSupportBalance = fail
ShuffledCandidateGate = fail
ShuffledBranchRatio = fail
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
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledTrueBranchDelta
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
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledTrueBranchDelta
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

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9256.csv
p0_v9255_boundary_reproduction.csv
p1_true_branch_delta_tensor_interface_gate.csv
p2_k7d_control_gain_subphase_attribution_and_fusion.csv
p3_true_custom_branch_delta_implementation_matrix.csv
p4_formula_proxy_negative_control_audit.csv
p5_exact_signal_controller_calibration.csv
p6_online_support_stratum_expansion.csv
p7_leave_dataset_and_stratum_out.csv
p8_official_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_10seed_functional_validation.csv
true_branch_delta_interface_trace_v9256.csv
k7d_control_gain_trace_v9256.csv
true_custom_branch_delta_trace_v9256.csv
formula_proxy_negative_control_trace_v9256.csv
exact_signal_controller_trace_v9256.csv
support_density_trace_v9256.csv
leaveout_trace_v9256.csv
paired_replay_branch_trace_v9256.csv
system_true_delta_kernel_overhead_trace_v9256.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9255_boundary_unstable
F3_dataset_tuning_detected
F4_true_branch_delta_interface_not_implemented
F5_source_measured_gap_used
F6_formula_proxy_promoted_illegally
F7_k7d_subphase_unattributed
F8_k7d_fusion_no_speedup
F9_true_custom_delta_not_predictive
F10_true_custom_delta_agreement_fail
F11_true_custom_delta_system_fail
F12_controller_precision_fail
F13_controller_coverage_fail
F14_controller_bad_event_fail
F15_support_measurement_too_narrow
F16_oracle_support_collapse
F17_leave_dataset_out_fail
F18_leave_stratum_out_fail
F19_paired_replay_control_equivalent
F20_shuffle_control_pass
F21_functional_lr_equivalent
F22_short_run_task_drop
F23_full_run_no_macro_hard_stratum_gain
F24_strong_baseline_explains_gain
F25_robustness_fail
F26_external_not_ready
F27_fake_or_proxy_violation
F28_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-TrueBranchDeltaInterfaceImplemented:
  at least one candidate exposes true branch-delta tensor interface.

R2-K7dControlGainAttributed:
  K7d subphase cost is fully attributed.

R3-K7dControlGainFused:
  K7d control-gain computation is fused and faster.

R4-TrueCustomBranchDeltaPredictive:
  true branch-delta preserves safe-good signal.

R5-TrueCustomBranchDeltaSystemPass:
  true branch-delta passes step/memory envelope.

R6-ExactSignalControllerPass:
  branch-delta-primary controller passes heldout precision / coverage / bad-event / system gate.

R7-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R8-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R9-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R10-TrueBranchDeltaInterfaceNotImplementedAgain:
  v9.2.56 still lacks true branch-delta tensor interface.

R11-SourceMeasuredGapProxyAgain:
  fast path still relies on source-measured gap or formula proxy.

R12-TrueDeltaPredictiveButTooExpensive:
  true branch-delta preserves signal but remains outside system envelope.

R13-TrueDeltaSystemPassButSignalLost:
  true branch-delta path is cheap but loses exact signal.

R14-ControllerStillCoverageLimited:
  precision/safety acceptable but coverage below 0.03.

R15-ControllerUnsafe:
  coverage exists but bad-event above 0.05.

R16-OracleSupportCollapse:
  fresh natural replay no longer has enough safe-good support.

R17-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R18-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R19-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9255_boundary_pass
dataset_tuning_detected
true_branch_delta_interface_pass
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
k7d_attribution_pass
dominant_k7d_subphase
k7d_fusion_pass
k7d_time_reduction
best_custom_delta_id
custom_delta_predictivity_pass
custom_delta_agreement_pass
custom_delta_system_pass
custom_delta_auc
custom_delta_corr
custom_delta_accept_agreement
custom_delta_step_ratio_q90
custom_delta_memory_ratio
formula_proxy_negative_control_pass
best_controller_id
exact_signal_controller_pass
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
support_measurement_pass
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9256_strict_purekan_functional
success_v9256_full_functional
success_v9256_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 true branch-delta tensor interface gate
  P2 K7d attribution / fusion
  P3 true custom branch-delta matrix
  P4 formula proxy audit
  P6 support expansion

Batch 2:
  P5 exact-signal controller calibration
  P7 leave-dataset-out / leave-stratum-out
  P8 paired replay scout

Batch 3:
  official P8 paired replay
  P9 short-run if paired replay passes

Batch 4:
  P10 full 10-seed
  robustness / strong baseline if full-run pass
```

Gate rule：

```text
P5/P8 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  true branch-delta interface pass
  true branch-delta predictivity pass
  true branch-delta agreement pass
  true branch-delta system pass
  exact-signal controller pass
  LDO/LSO pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.55 boundary reproduced
true branch-delta interface gate executed
K7d residual cost attributed
at least one true branch-delta candidate implemented or explicitly route as not implemented
formula proxies audited as non-official
support expansion measured
no fake/proxy/offload/loss/teacher violation
```

## Branch-delta system success

```text
Minimum diagnostic success
+
at least one true branch-delta path predicts safe-good
+
accept agreement pass
+
system overhead gate pass
```

## Legal controller success

```text
Branch-delta system success
+
branch-delta-primary controller heldout pass
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
1. v9.2.55 boundary cannot be reproduced；
2. true branch-delta tensor interface still not implemented；
3. implementation uses source-measured gap input；
4. formula proxy is promoted illegally；
5. K7d cost cannot be attributed；
6. K7d fusion does not reduce cost；
7. true branch-delta loses safe-good predictivity；
8. true branch-delta remains too expensive；
9. true branch-delta passes system but agreement fails；
10. exact-signal controller cannot simultaneously meet precision / coverage / bad-event；
11. online support remains too narrow；
12. fresh natural oracle support collapses；
13. leave-dataset-out fails；
14. leave-stratum-out fails；
15. paired replay remains control-equivalent；
16. shuffle controls pass, indicating overfit；
17. short-run task drops；
18. full run gives no macro / hard-stratum / geometry gain；
19. functional breaks system gate；
20. gains are explained by QuadraticFeatureMLP；
21. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：true branch-delta + exact-signal controller + LDO/LSO + paired replay pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation and external robustness。

## Case B：true branch-delta interface still not implemented

必须声明：

```text
v9.2.56 did not test the real blocker; true branch-delta tensor interface remains missing.
```

下一步必须工程实现 interface，不允许继续调 controller threshold。

## Case C：true branch-delta predicts but remains too expensive

必须声明：

```text
safe-good is observable through true branch-output displacement, but implementation remains outside system envelope.
```

下一步进入 lower-level CUDA/C++ optimization，不调 dataset。

## Case D：true branch-delta becomes cheap but loses signal

必须声明：

```text
current compression loses exact branch-delta information.
```

下一步回到 exact branch-delta compression or richer output-delta statistics。

## Case E：controller fails despite system-legal branch-delta

必须声明：

```text
value is observable and cheap, but accept/abstain support geometry is not deployable.
```

下一步修 support/risk/family controller，不修 base/attach。

## Case F：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

## Case G：oracle support collapses

必须声明：

```text
fresh natural replay no longer has enough safe-good support.
```

下一步回到 carrier/support mechanism。

---

# Part XII. 最终建议

v9.2.56 的一句话策略是：

$$
\boxed{
\text{不要再用 source-measured gap 或 formula proxy 代替 branch-delta；实现 true branch-delta tensor interface，并融合 K7d control-gain compute。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
CBD0 有没有 AUC；
CBD1 是否够快；
CBD4 是否编译成功；
C0/C6 threshold 是否差一点；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. 是否能从真实 functional / control branch delta 计算 logits？
2. 是否能杜绝 source-measured scalar gap input？
3. 是否能把 K7d control_gain_compute 融合到 kernel 内？
4. 是否能保留 CBD0 的 AUC=0.8884008136827201 与 agreement=1.0？
5. 是否能把 step ratio 从 2.863279885 降到 <=1.50？
6. branch-delta-primary controller 能否 heldout pass？
7. support 是否能扩展到 >=6 measured strata？
8. controller 能否 LDO/LSO？
9. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.56 的结果将给出清晰分叉：

```text
if true branch-delta + K7d fusion + controller + LDO/LSO + paired replay pass:
  strict PureKAN functional obtains local causal evidence.

if true interface still missing:
  implementation remains blocker; stop controller tuning.

if true branch-delta predicts but too expensive:
  lower-level CUDA/C++ optimization remains blocker.

if true branch-delta cheap but loses signal:
  exact branch-delta information cannot be compressed by current implementation.

if controller local pass but leave-out fails:
  no dataset tuning; broaden signal-stratum/family support.

if oracle support collapses:
  carrier/support stability is blocker.
```
