# DG-KAN v9.2.53 D0 Vectorized Controller-Prep 与 Real Fused Branch-Delta Kernel Closure 完整实验计划

> 本计划基于 v9.2.52 `Fused Delta-Prep 与 Selected-Logit Cascade Controller` 的真实复盘制定。  
> v9.2.52 的 terminal route 是：
>
> ```text
> route = R10-FusedDeltaPredictiveButTooExpensive
> base_candidate = LQ-t2-h256
> success_v9252_strict_purekan_functional = False
> success_v9252_full_functional = False
> success_v9252_external_ready = False
> ```
>
> v9.2.52 的关键事实是：
>
> ```text
> P1 L1 delta-prep attribution:
>   pass = 1
>   dominant subphase = D0-role score / controller scalar preparation
>   D0 ratio = 0.770145
>
> P2 fused delta-prep:
>   best = FDP6-HybridFusedDeltaPrep
>   correctness pass = 1
>   step ratio = 3.898403 > 1.50
>   system pass = 0
>   FDP5-FusedRealFunctionalDeltaPrep = not_implemented_real_fused_kernel
>
> P3 selected-logit branch delta:
>   best route-level reference = SLD0-FullLogitExactReference
>   AUC = 0.888401
>   agreement = 1.0
>   step ratio = 6.096370
>
>   SLD5-CachedControlSelectedDelta
>   step ratio = 1.356746
>   agreement = 0.448661
>   not official
>
> P4 candidate generator:
>   best = CG1-SelectedLogitRecall
>   recall = 0.576766
>   candidate bad-event = 0.251535
>
> P5 cascade controller:
>   best = C4-HybridExactBorderlineController
>   precision = 0.152439
>   coverage = 0.040675
>   bad-event = 0.256098
>   official eligible = 0
>
> P6 oracle support:
>   precision = 1.0
>   coverage = 0.121693
>   bad-event = 0.0
>   measured signal strata = 3
>   balanced diagnostic rows = 36
>   support measurement pass = 0
>
> current blocker:
>   selected_or_fused_delta_predictive_but_not_system_legal
> ```
>
> 本轮最重要的解释不是 “selected-logit 失败”，也不是 “functional update 不可行”。更准确地说：
>
> $$
> \boxed{
> \text{exact branch-delta / full-logit reference 能稳定识别 safe-good，但当前实现把大量时间花在 D0 controller scalar prep 和 branch-delta orchestration 上。}
> }
> $$
>
> 因此 v9.2.53 不应继续试一个新的 cheap score，也不应继续只做 source-row autopsy。  
> v9.2.53 必须把 system blocker 分成两类同时处理：
>
> $$
> \boxed{
> \text{D0 role/controller scalar preparation 的批量化与去同步}
> }
> \quad+\quad
> \boxed{
> \text{真正实现 real fused branch-delta kernel，而不是 diagnostic/not-implemented row}
> }.
> $$
>
> 这轮要做的是：**把 v9.2.52 的 exact safe-good observability 变成 system-legal online controller**。如果做不到，也要明确进入 CUDA/Triton kernelization stop-go，而不是继续在 Python/controller 层小修小补。

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
  choose candidate generator separately per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

所有 official controller 必须只依赖 train-stream / event-level / role-level / support-level / branch-delta features。MNIST / Fashion / KMNIST 只用于 failure slicing 与 LDO 评估。

---

# Part I. 对 v9.2.52 的独立判断

## 1. v9.2.52 没有达到目标

v9.2.52 没有达到 strict PureKAN functional success。原因不是 full-run 失败，而是更前面的 system 和 controller gate 没有闭合：

```text
fused_delta_system_pass = 0
selected_delta_system_pass = 0
candidate_generator_pass = 0
cascade_controller_pass = 0
support_measurement_pass = 0
LDO / LSO / paired replay = not_run
```

因此不能声明：

```text
strict PureKAN local causal evidence
full functional success
external-ready success
Beyond-MLP success
```

`SLD0-FullLogitExactReference` 的 AUC 和 agreement 很强，但它 step ratio `6.096370`；`FDP6-HybridFusedDeltaPrep` correctness pass，但 step ratio `3.898403`；这些都不能进入 official route。

## 2. v9.2.52 的真实进展

v9.2.52 有两个明确进展。

第一，成本瓶颈进一步定位。v9.2.51 发现 L1 RealFunctional delta preparation dominant，而 v9.2.52 继续拆解 L1，发现 dominant subphase 是：

```text
D0-role score / controller scalar preparation
ratio = 0.770145
```

这非常关键。它提示我们，所谓 “delta-prep expensive” 可能并不只是数学上的 functional delta 本身，而可能是：

```text
per-event scalar score construction
per-event controller decision
role score / risk score / support score computation
family reliability lookup
threshold scan
Python loop / .item() / host sync
branch pack / branch scale
logging and decision materialization
```

因此下一步如果直接写一个新 selected-logit score，仍可能绕不过 D0。

第二，exact branch-delta signal 继续强。`SLD0-FullLogitExactReference` AUC `0.888401`，agreement `1.0`，oracle precision `1.0`，coverage `0.121693`，bad-event `0.0`。这说明 safe-good events 没消失，exact branch-output displacement 仍然是目前最可靠的 online value signal。

## 3. 当前真正 blocker

当前 blocker 不是：

```text
base 不稳；
attach 污染；
carrier silent；
safe-good events 不存在；
exact gap-probe 没有 value；
oracle support 消失；
functional update 应该放弃。
```

当前 blocker 是：

$$
\boxed{
\text{exact value signal exists, but the path that computes and deploys it is not system-legal.}
}
$$

它由三部分组成。

### 3.1 D0 orchestration / scalar-prep blocker

D0 占 L1 的 `0.770145`。这说明 controller scalar preparation 可能存在严重 per-row overhead。需要确认它是 GPU math、CPU sync、Python loop、family lookup、logging、还是 branch pack 引起的。如果是 Python / sync / logging，继续写 Triton branch kernel并不能直接解决；必须先批量化 controller prep。

### 3.2 Fused kernel not implemented blocker

`FDP5-FusedRealFunctionalDeltaPrep` 被明确写为 `not_implemented_real_fused_kernel`。这说明 v9.2.52 还没有真正实现 real fused kernel，只是测了 hybrid/preallocation/diagnostic candidates。下一步必须把 “not implemented fused kernel” 变成真实 candidate，或者明确 route 为需要低层 CUDA/Triton 实现。

### 3.3 Cheap approximation loses exact value signal

低成本 candidates 仍丢失 signal 或 agreement：

```text
SLD5:
  step ratio = 1.356746
  agreement = 0.448661

BLD2 from v9.2.51:
  step ratio = 1.411116
  AUC = 0.534446
  agreement = 0.401389
```

这说明 “便宜但错” 不是出路。当前需要的是 **exact-like but amortized / fused / selected**，而不是纯 cheap surrogate。

## 4. 进度如何

进度慢，但不是无效。最近几轮的 blocker 逐步具体化：

```text
v9.2.48:
  online gap_probe 有强 AUC，但 probe overhead 爆炸。

v9.2.49:
  overhead 定位到 F7 metric computation。

v9.2.50:
  F7 内部 dominant subphase 定位到 M0-logit preparation。

v9.2.51:
  M0 内部定位到 L1 RealFunctional delta preparation；
  exact branch-delta AUC 强，但 step ratio 4.700044。

v9.2.52:
  L1 内部定位到 D0 controller scalar preparation；
  exact full-logit reference AUC 强，但 step ratio 6.096370；
  hybrid fused delta correctness pass，但 step ratio 3.898403。
```

这条线说明我们没有走偏，但下一步必须从 “诊断 + approximation matrix” 转向 “真实系统实现 + stop-go”。如果 v9.2.53 仍只是增加几个 cheap scores，而不是解决 D0 和 real fused kernel，就会继续慢。

## 5. 是否在正确道路上

是，但路线必须进一步收紧：

```text
online safe-good signal 已证明
exact branch-delta signal 已证明
oracle support 仍强
当前必须做 D0 vectorization + real fused branch-delta kernel
```

错误路线是：

```text
继续调 C4 / CG1 threshold；
继续只加 cheap recall feature；
把 SLD0 AUC 写成 functional success；
把 oracle support 写成 controller success；
按 dataset tuning；
回到 basis sweep；
提前研究 PureKANConv / PureKANFormer。
```

当前基函数 / base 不是主 blocker。LQ-t2-h256 仍是当前 base candidate；functional update 的主 blocker 是 online value-signal path 的 system legality。

---

# Part II. v9.2.53 总体目标

v9.2.53 的总体目标是：

$$
\boxed{
\text{用 D0 vectorized controller-prep 与 real fused branch-delta kernel，将 exact safe-good observability 转化为 system-legal online controller。}
}
$$

本轮要同时闭合六件事：

```text
1. 把 D0-role score / controller scalar preparation 拆成 CPU/GPU/sync/Python/family lookup 等更细 subphase；
2. 实现 vectorized controller-prep，消除 per-row scalar sync 与 Python loop；
3. 实现真实 real fused branch-delta candidate，而不是 not_implemented diagnostic；
4. 用 selected-logit / full-logit hybrid exact confirmation 保留 SLD0 的 safe-good signal；
5. 建立 coverage-preserving cascade controller；
6. 在同一 runner 中并行执行 support expansion、LDO/LSO scout、paired replay scout。
```

---

## 1. D0 vectorization success

v9.2.53 必须把 D0 拆成更细粒度，并实现批量化 controller-prep。

D0 subphases：

```text
D0a role scalar computation
D0b risk scalar computation
D0c support density / family reliability lookup
D0d candidate thresholding
D0e branch scale / branch pack
D0f tensor-to-host scalar extraction
D0g logging / hash / timestamp overhead
D0h Python loop / dispatch overhead
D0i device sync / stream wait
```

D0 attribution pass：

```text
unknown_fraction <= 0.10
dominant_d0_subphase_identified = 1
D0_subphase_sum_close_to_D0 within ±0.05
```

D0 vectorization pass：

$$
T_{D0,\text{vectorized}}\leq0.35T_{D0,\text{ref}}.
$$

and:

$$
Agreement_{\text{controller-prep}}\geq0.99.
$$

System contribution gate：

$$
StepRatio_{q90}\leq1.50
$$

or if full step not yet pass:

$$
StepRatio_{q90,\text{vectorized}}\leq0.70StepRatio_{q90,\text{ref}}.
$$

---

## 2. Real fused branch-delta kernel success

`FDP5-FusedRealFunctionalDeltaPrep` must become implemented. The implementation must compute branch-delta features for a batch of events / branches without repeated materialization.

Target form:

$$
\widehat z_{b,\mathcal{C}_s}
=
z_{0,\mathcal{C}_s}
+
\widehat{\Delta z}_{b,\mathcal{C}_s}.
$$

For exact-like candidate:

$$
Agreement_{\text{accept}}\geq0.90.
$$

Predictivity:

$$
AUC(\widehat S_{\text{gap}},Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(\widehat S_{\text{gap}},V_{\text{safe-grounded}})\geq0.35.
$$

System:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

If predictivity and agreement pass but system fail, route must be:

```text
R8-FusedBranchDeltaPredictiveButNeedsCustomKernel
```

not success.

---

## 3. Selected-logit exactness repair success

v9.2.52 showed selected/cached cheap variants can be system-light but low agreement. v9.2.53 must distinguish three roles:

```text
selected-logit for candidate recall
selected-logit for approximate confirmation
exact/full-logit only for borderline confirmation
```

Selected-logit diagnostic pass:

$$
AUC_{\text{safe-good}}\geq0.60,
$$

$$
Agreement_{\text{accept}}\geq0.80.
$$

Selected-logit official pass:

$$
AUC_{\text{safe-good}}\geq0.70,
$$

$$
Agreement_{\text{accept}}\geq0.90,
$$

$$
BadEventRate\leq0.05.
$$

If selected-logit has AUC but agreement below 0.80, it can only be used as weak recall feature.

---

## 4. Candidate generator success

Candidate generator no longer relies on cheap family reliability alone. It uses D0-vectorized candidate scores plus selected-logit weak signal.

Candidate generator pass:

$$
Recall_{\text{safe-good}}\geq0.80,
$$

$$
CandidateRate\leq0.35,
$$

$$
CandidateBadEventRate\leq0.20.
$$

If candidate recall fails while oracle support remains high, route must be:

```text
R11-CandidateDiscoveryFailOracleHigh
```

---

## 5. Cascade controller success

Official controller:

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

where:

$$
Candidate(e)=\mathbb{1}[S_{\text{candidate}}(e)\geq\tau_c],
$$

$$
FusedBranchDeltaConfirm(e)=\mathbb{1}[\widehat S_{\text{gap}}(e)\geq\tau_g],
$$

$$
RiskSafe(e)=\mathbb{1}[S_{\text{risk}}(e)\leq\rho],
$$

$$
SupportBalanced(e)
=
\mathbb{1}[Rel(family(e))\geq r_0]
\cdot
\mathbb{1}[Density(e)\geq d_0].
$$

Heldout pass:

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

System pass:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Support pass:

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

---

## 6. Support measurement success

v9.2.52 measured signal strata count was only `3` and balanced diagnostic rows only `36`。v9.2.53 must expand measurement materially.

Measurement pass:

```text
natural_real_event_count >= 12000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 6
measured_family_count >= 12
```

Official split rule:

```text
natural rows:
  official controller / leave-out / paired replay basis

balanced diagnostic rows:
  attribution / support analysis only
```

Balanced diagnostic rows must not be mixed into official heldout pass unless calibration / heldout / diagnostic roles are explicitly separated.

---

# Part III. 核心假设

## H1：v9.2.52 的主要 blocker 是 D0 scalar-prep / orchestration，不是 safe-good 不可观测

Evidence:

```text
D0 ratio = 0.770145
SLD0 AUC = 0.888401
SLD0 agreement = 1.0
oracle precision = 1.0
oracle coverage = 0.121693
oracle bad-event = 0.0
```

H1 成立标准：

```text
D0 subphase attribution identifies CPU/sync/Python/scalar/family lookup or branch-pack bottleneck
and D0 vectorization reduces D0 time by >= 65%
```

H1 失败标准：

```text
D0 is dominated by unavoidable GPU math that cannot be batched/fused
or D0 vectorization does not reduce total step time by >= 20%
```

---

## H2：real fused branch-delta kernel can preserve exact path signal

`SLD0` and `BLD1` exact paths have strong AUC and agreement. H2 assumes their information can be preserved with a fused implementation.

H2 成立标准：

at least one fused branch-delta candidate gets:

$$
AUC_{\text{safe-good}}\geq0.70
$$

or:

$$
Corr_{\text{safe-grounded}}\geq0.35
$$

and:

$$
Agreement_{\text{accept}}\geq0.90.
$$

H2 失败标准：

```text
all fused/selected delta candidates either:
  lose signal, AUC < 0.60
  or agreement < 0.80
```

---

## H3：selected-logit should not be final accept unless agreement is fixed

SLD5 was system-light but agreement only `0.448661`. H3 says selected-logit can be useful, but only if exactness is repaired.

H3 成立标准：

selected-logit exactness repair raises:

$$
Agreement_{\text{accept}}\geq0.80
$$

diagnostic, or:

$$
Agreement_{\text{accept}}\geq0.90
$$

official.

H3 失败标准：

```text
selected-logit variants stay agreement < 0.80
```

Then selected-logit remains candidate generator only.

---

## H4：candidate discovery remains a blocker unless selected-logit signal is added

CG1 recall was `0.576766`, below `0.80`, and candidate bad-event was `0.251535`, above `0.20`. H4 says cheap candidate discovery alone is insufficient.

H4 成立标准：

hybrid candidate generator achieves:

$$
Recall_{\text{safe-good}}\geq0.80,
$$

$$
CandidateRate\leq0.35,
$$

$$
CandidateBadEventRate\leq0.20.
$$

H4 失败标准：

```text
candidate recall < 0.60
or candidate bad-event > 0.30
or candidate rate approaches all-accept
```

---

## H5：oracle support still exists; no carrier reset unless oracle collapses

H5 成立标准：

fresh v9.2.53 natural rows:

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

If H5 fails, return to carrier/support design. If H5 holds but controller fails, continue legal/system/controller.

---

# Part IV. 并行执行设计

v9.2.53 runner must execute these lanes in parallel:

```text
Lane A:
  v9.2.52 boundary reproduction

Lane B:
  D0 controller scalar-prep micro-attribution:
    role scalar
    risk scalar
    support density / family reliability
    thresholding
    branch pack
    host sync
    logging
    Python dispatch

Lane C:
  vectorized controller-prep:
    batch event features
    device-side thresholding
    no per-row .item()
    delayed logging
    preallocated event buffers
    vectorized family / support ids

Lane D:
  real fused branch-delta kernel:
    implement FDP5 as real candidate
    fused functional delta-prep
    selected-logit branch delta
    multi-branch branch dimension

Lane E:
  selected-logit exactness repair:
    true/top/hard-negative/tail logits
    selected CE/margin/risk/gap
    borderline exact confirmation

Lane F:
  high-recall candidate generator:
    cheap + selected-logit + support density

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

Gate discipline:

```text
diagnostic rows may be measured early
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  vectorized / fused branch-delta predictivity pass
  branch-delta system gate pass
  cascade controller pass
  LDO/LSO pass
```

---

# Part V. Candidate designs

## 1. Base and carrier

Official base remains:

```text
R2-LQ-fanin-output-scale-confirmed
LQ-t2-h256
```

Carrier candidates:

```text
A0-current-v9252
A3-RoleWiseFT7ResetCarrier
A6-HybridRoleControlRiskCarrier
```

A1 risk-bounded carrier remains diagnostic unless it passes carrier-active gate:

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

---

## 2. D0 vectorization candidates

### D0V0：v9.2.52 reference

Reference only.

### D0V1：NoItemDeviceThreshold

Eliminate per-row `.item()` / CPU scalar extraction. All threshold decisions remain device-side until bulk logging.

### D0V2：VectorizedRoleRiskScalars

Compute role score, risk score, support score for all candidate events as tensors:

$$
S_{\text{role}},S_{\text{risk}},S_{\text{support}}\in\mathbb{R}^{N_{\text{events}}}.
$$

### D0V3：PackedFamilySupportLookup

Replace Python dict / tuple family lookup with tensor-coded family ids:

$$
family\_id(e)
=
hash(stratum,horizon,risk\_bucket,role\_bucket,attach,branch).
$$

### D0V4：PreallocatedEventBuffers

Preallocate:

```text
event_feature_buffer
role_score_buffer
risk_score_buffer
support_score_buffer
decision_buffer
branch_pack_buffer
```

### D0V5：DelayedBulkLogging

No per-row logging sync. Store on-device or pinned CPU batch summary; flush once per block.

### D0V6：D0VectorizedAll

Combines D0V1-D0V5.

---

## 3. Real fused branch-delta candidates

### FBD0：SLD0-FullLogitExactReference

Reference only. Predictive but too expensive.

### FBD1：FDP5-RealFusedFunctionalDeltaPrep

This is the previously not-implemented real fused kernel candidate. It must now be implemented or explicitly route to low-level kernelization failure.

Computes:

```text
role weights
functional scale
risk cap
support cap
delta buffer
branch scale
```

in one fused path.

### FBD2：FusedSelectedLogitDelta

Computes selected logits only:

```text
true class
top-1
top-2
hard negative
tail class
```

### FBD3：FusedMultiBranchSelectedDelta

Computes selected logits for multiple branches:

```text
RealFunctional
AdamWParallel
bestLR
NoOp
```

in a fused branch dimension.

### FBD4：HybridSelectedExactBorderline

Pipeline:

```text
selected-logit delta
→ if high confidence, accept/reject
→ if borderline, exact delta-view confirm
```

### FBD5：CachedFunctionalDeltaState

Compute $\Delta\theta_F$ once per step, reuse across branches and event decisions.

### FBD6：D0VectorizedFusedDelta

Combines D0V6 + FBD1/FBD3.

---

## 4. Selected-logit exactness repair candidates

### SLR1：TrueTopHardNegativeRepair

Use logits:

```text
true class
top-1
top-2
hard negative
```

### SLR2：TopKSelectedLogits

Use $k\in\{3,5,10\}$ selected logits. For 10-class datasets, $k=10$ equals full logit diagnostic and should reproduce exact.

### SLR3：CE-MarginRiskSelected

Approximate CE, margin, wrong-confidence, risk with selected logits.

### SLR4：BorderlineExactConfirm

Exact confirm only if selected decision margin:

$$
|S_{\text{selected-gap}}-\tau_g| < \epsilon_b.
$$

### SLR5：RiskAwareSelectedConfirm

Selected-logit decision is official only if risk score is comfortably safe:

$$
S_{\text{risk}}\leq \rho_{\text{safe}}.
$$

---

## 5. Candidate generator candidates

### CG0：v9.2.52 CG1 reference

Reference only.

### CG1：SelectedLogitRecallV2

Uses selected-logit gap as recall feature, not final accept.

### CG2：D0VectorizedRiskRoleRecall

Uses vectorized role/risk/support scores.

### CG3：FamilyDensityRecall

Uses family reliability and density but must not use dataset name.

Family definition:

$$
family(e)
=
(stratum,horizon,risk\_bucket,role\_bucket,attach\_type,branch\_bucket,probe\_bucket).
$$

### CG4：HybridRecallV2

$$
S_{\text{candidate}}
=
a_1S_{\text{selected-gap}}
+
a_2S_{\text{risk-safe}}
+
a_3S_{\text{role}}
+
a_4Rel(f)
+
a_5Density.
$$

Coefficient selection uses calibration split only.

---

## 6. Controller candidates

### C0：v9.2.52 reference

Reference only.

### C1：VectorizedCandidateFusedDeltaConfirm

$$
Accept(e)
=
CG(e)
\land
FBD(e)
\land
RiskSafe(e).
$$

### C2：FamilyBalancedFusedDelta

$$
Accept(e)
=
CG(e)
\land
FBD(e)
\land
RiskSafe(e)
\land
Rel(family(e))\geq r_0
\land
FamilyBalance(e).
$$

### C3：SelectedBorderlineExactController

$$
Accept(e)
=
HighConfidenceSelectedAccept(e)
\lor
[
BorderlineSelected(e)
\land
ExactFusedConfirm(e)
].
$$

### C4：D0VectorizedParetoController

Accept if event lies on Pareto frontier of:

```text
low D0 cost
positive fused branch-delta gap
low risk
high support density
family balance
system envelope
```

### C5：Oracle

Posthoc diagnostic only. Never official.

---

# Part VI. 实验阶段

## P0：v9.2.52 boundary reproduction

### 目标

确认 v9.2.52 boundary 稳定。

### 必须记录

```text
route
source_route_v9251
dominant_l1_subphase
d0_ratio
best_fused_delta_id
fused_delta_correctness_pass
fused_delta_step_ratio
best_selected_delta_id
selected_delta_auc
selected_delta_agreement
selected_delta_step_ratio
candidate_generator_recall
candidate_bad_event
cascade_precision
cascade_coverage
cascade_bad_event
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
route = R10-FusedDeltaPredictiveButTooExpensive
D0 dominant confirmed
oracle support pass = 1
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_d0_dominance_ladder.svg
p0_oracle_high_system_low.svg
```

---

## P1：D0 controller scalar-prep micro-attribution

### 目标

拆开 D0，确认瓶颈是 CPU sync、Python loop、family lookup、branch packing，还是 GPU math。

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
host_item_count
python_loop_count
family_lookup_count
branch_pack_MB
logging_bytes
unknown_fraction
```

Subphases：

```text
D0a-role_scalar_compute
D0b-risk_scalar_compute
D0c-support_density_family_lookup
D0d-thresholding_decision
D0e-branch_scale_pack
D0f-host_scalar_extraction
D0g-logging_hash_timestamp
D0h-python_dispatch_loop
D0i-device_sync_stream_wait
```

### 判断标准

P1 pass：

```text
unknown_fraction <= 0.10
dominant_d0_subphase_identified = 1
D0 subphase sum within ±0.05 of D0 total
```

### 可视化

```text
p1_d0_subphase_waterfall.svg
p1_host_sync_vs_gpu_math.svg
p1_python_loop_count_vs_time.svg
p1_family_lookup_cost.svg
```

---

## P2：D0 vectorized controller-prep

### 目标

把 D0 controller-prep 批量化，减少 per-row scalar overhead。

### 必须记录

```text
d0_vector_id
role_score_agreement
risk_score_agreement
support_score_agreement
decision_agreement
d0_time_ratio_vs_ref
d0_memory_ratio_vs_ref
host_item_count
python_loop_count
sync_count
step_ratio_q90
memory_ratio
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

D0 vectorization pass：

$$
T_{D0,\text{vectorized}}\leq0.35T_{D0,\text{ref}},
$$

$$
Agreement_{\text{decision}}\geq0.99.
$$

System diagnostic：

$$
StepRatio_{q90,\text{vectorized}}\leq0.70StepRatio_{q90,\text{ref}}.
$$

Official system still requires:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p2_d0_vectorization_cost_reduction.svg
p2_decision_agreement.svg
p2_d0_host_item_elimination.svg
```

---

## P3：Real fused branch-delta implementation

### 目标

把 `FDP5-FusedRealFunctionalDeltaPrep` 从 not_implemented 变成真实 measured candidate。

### 必须记录

```text
fused_branch_delta_id
implementation_status
uses_full_logits
uses_selected_logits
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
p3_fused_branch_delta_auc_cost_pareto.svg
p3_fused_branch_delta_agreement.svg
p3_fused_branch_delta_bad_event_curve.svg
p3_kernel_count_sync_count.svg
```

---

## P4：Selected-logit exactness repair

### 目标

判断 selected-logit path 能否从 weak diagnostic 变成 official confirm path，或者只能作为 candidate generator。

### 必须记录

```text
selected_repair_id
selected_class_count
classes_used
AUC_safe_good
corr_safe_grounded
agreement_exact_accept
bad_event_at_gate
precision_at_gate
coverage_at_gate
step_ratio_q90
memory_ratio
borderline_exact_call_rate
```

### 判断标准

Official selected pass：

$$
AUC_{\text{safe-good}}\geq0.70,
$$

$$
Agreement_{\text{accept}}\geq0.90,
$$

$$
BadEventRate\leq0.05,
$$

$$
StepRatio_{q90}\leq1.50.
$$

Diagnostic selected pass：

$$
AUC_{\text{safe-good}}\geq0.60,
$$

$$
Agreement_{\text{accept}}\geq0.80.
$$

### 可视化

```text
p4_selected_exactness_repair.svg
p4_selected_class_ablation.svg
p4_borderline_exact_tradeoff.svg
```

---

## P5：High-recall candidate generator

### 目标

修复 v9.2.52 candidate recall 不足与 candidate bad-event 偏高。

### 必须记录

```text
candidate_generator_id
features_used
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

Candidate pass：

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
p5_candidate_recall_candidate_rate.svg
p5_candidate_bad_event_curve.svg
p5_candidate_feature_ablation.svg
p5_candidate_family_coverage.svg
```

---

## P6：Coverage-preserving cascade controller

### 目标

用 vectorized candidate + fused/selected branch-delta confirm + risk/support balance 建立 official local controller。

### Split design

```text
calibration split:
  choose thresholds / coefficients

heldout natural split:
  official local controller gate

balanced diagnostic split:
  support/failure analysis only

leave-dataset-out:
  P8

leave-stratum-out:
  P8
```

### 必须记录

```text
controller_id
d0_vector_id
fused_branch_delta_id
selected_repair_id
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
p6_controller_precision_coverage_bad.svg
p6_controller_cost_vs_value.svg
p6_controller_family_coverage.svg
p6_cascade_stage_sankey.svg
p6_oracle_legal_gap.svg
```

---

## P7：Online support and stratum expansion

### 目标

扩大 natural / balanced support，避免 measured support 过窄。

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
d0_vector_id
fused_branch_delta_id
selected_repair_id
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
p7_signal_strata_coverage.svg
p7_family_support_heatmap.svg
p7_oracle_support_by_stratum.svg
p7_natural_vs_balanced_distribution.svg
```

---

## P8：Leave-dataset-out / leave-stratum-out

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
d0_vector_id
fused_branch_delta_id
selected_repair_id
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
p8_leave_dataset_out_matrix.svg
p8_leave_stratum_out_matrix.svg
p8_hidden_dataset_tuning_audit.svg
p8_leaveout_failure_modes.svg
```

---

## P9：Official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
controller_id = best P8 survivor
d0_vector_id = best P8 survivor
fused_branch_delta_id = best P8 survivor
selected_repair_id = best P8 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledD0Vector, ShuffledFusedDelta, ShuffledSelectedDelta,
           ShuffledCandidateRecall, ShuffledRiskScore, ShuffledBranchRatio,
           ShuffledControlGap, ShuffledValueScore, ShuffledSignalChannel,
           FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled,
           DatasetRouteShuffled, EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
controller_id
d0_vector_id
fused_branch_delta_id
selected_repair_id
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
ShuffledD0Vector = fail
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
p9_official_paired_replay_pareto.svg
p9_macro_beat_rate.svg
p9_signal_stratum_win_matrix.svg
p9_shuffle_control_matrix.svg
p9_system_gate_distribution.svg
```

---

## P10：Short-run scout

### 目标

如果 P9 pass，验证局部 paired replay 优势能否在连续训练中保持。

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
p10_short_run_task_mechanism_pareto.svg
p10_short_run_controls.svg
p10_event_timeline.svg
p10_ce_tail_margin_panel.svg
```

---

## P11：Full 10-seed validation

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

## P12：Robustness / strong baseline / external-ready

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
contract_audit_v9253.csv
p0_v9252_boundary_reproduction.csv
p1_d0_controller_scalar_prep_micro_attribution.csv
p2_d0_vectorized_controller_prep.csv
p3_real_fused_branch_delta_implementation.csv
p4_selected_logit_exactness_repair.csv
p5_high_recall_candidate_generator.csv
p6_coverage_preserving_cascade_controller.csv
p7_online_support_stratum_expansion.csv
p8_leave_dataset_and_stratum_out.csv
p9_official_paired_replay.csv
p10_short_run_functional_validation.csv
p11_full_10seed_functional_validation.csv
p12_robustness_external_ready.csv
d0_subphase_trace_v9253.csv
d0_vectorization_trace_v9253.csv
real_fused_branch_delta_trace_v9253.csv
selected_logit_exactness_trace_v9253.csv
candidate_generator_trace_v9253.csv
cascade_controller_trace_v9253.csv
support_density_trace_v9253.csv
leaveout_trace_v9253.csv
paired_replay_branch_trace_v9253.csv
system_fused_delta_overhead_trace_v9253.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9252_boundary_unstable
F3_dataset_tuning_detected
F4_d0_subphase_unattributed
F5_d0_vectorization_no_speedup
F6_d0_vectorization_breaks_decision
F7_real_fused_branch_delta_not_implemented
F8_fused_branch_delta_not_predictive
F9_fused_branch_delta_not_system_legal
F10_selected_logit_exactness_fail
F11_candidate_generator_low_recall
F12_candidate_generator_high_bad_event
F13_cascade_precision_fail
F14_cascade_coverage_fail
F15_cascade_bad_event_fail
F16_support_measurement_too_narrow
F17_oracle_support_collapse
F18_leave_dataset_out_fail
F19_leave_stratum_out_fail
F20_paired_replay_control_equivalent
F21_shuffle_control_pass
F22_functional_lr_equivalent
F23_short_run_task_drop
F24_full_run_no_macro_hard_stratum_gain
F25_strong_baseline_explains_gain
F26_robustness_fail
F27_external_not_ready
F28_fake_or_proxy_violation
F29_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-D0SubphaseAttributed:
  D0 controller scalar-prep cost is fully attributed.

R2-D0VectorizedControllerPrepPass:
  vectorized controller-prep reduces D0 cost and preserves decision agreement.

R3-RealFusedBranchDeltaImplemented:
  previously not-implemented real fused branch-delta candidate is now measured.

R4-FusedBranchDeltaPredictive:
  fused branch-delta predicts safe-good.

R5-FusedBranchDeltaSystemPass:
  fused branch-delta passes system envelope.

R6-SelectedLogitExactnessRepaired:
  selected-logit path reaches useful exact agreement.

R7-CandidateGeneratorPass:
  high-recall candidate generator finds enough safe-good candidates.

R8-CascadeControllerPass:
  coverage-preserving cascade passes heldout precision / coverage / bad-event / system gate.

R9-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R10-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R11-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R12-FusedBranchDeltaPredictiveButNeedsCustomKernel:
  fused/selected branch delta predicts safe-good but remains outside system envelope.

R13-SelectedDeltaCheapButUninformative:
  selected-logit path is cheap but loses exact gap-probe signal.

R14-CandidateDiscoveryFailOracleHigh:
  oracle support exists but candidate generator cannot find enough events.

R15-ControllerStillCoverageLimited:
  precision and safety are acceptable but coverage remains below 0.03.

R16-ControllerUnsafe:
  coverage exists but bad-event remains above 0.05.

R17-OracleSupportCollapse:
  fresh natural replay no longer has enough safe-good support.

R18-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R19-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R20-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9252_boundary_pass
dataset_tuning_detected
d0_subphase_attribution_pass
dominant_d0_subphase
d0_vectorization_pass
d0_time_ratio_vs_ref
d0_decision_agreement
real_fused_branch_delta_implemented
best_fused_branch_delta_id
fused_branch_delta_predictivity_pass
fused_branch_delta_system_pass
fused_branch_delta_auc
fused_branch_delta_corr
fused_branch_delta_accept_agreement
fused_branch_delta_step_ratio_q90
fused_branch_delta_memory_ratio
selected_logit_exactness_pass
best_selected_repair_id
selected_repair_agreement
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
success_v9253_strict_purekan_functional
success_v9253_full_functional
success_v9253_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 D0 controller scalar-prep micro-attribution
  P2 D0 vectorized controller-prep
  P3 real fused branch-delta implementation
  P4 selected-logit exactness repair
  P5 high-recall candidate generator

Batch 2:
  P6 coverage-preserving cascade controller
  P7 online support / stratum expansion
  P8 leave-dataset-out / leave-stratum-out
  P9 paired replay scout

Batch 3:
  official P9 paired replay
  P10 short-run if paired replay passes

Batch 4:
  P11 full 10-seed
  P12 robustness / strong baseline
```

Gate rule：

```text
P6/P9 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  vectorized / fused branch-delta predictivity pass
  branch-delta system gate pass
  cascade controller pass
  LDO/LSO pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.52 boundary reproduced
D0 subphase attribution completed
D0 vectorization measured
real fused branch-delta candidate implemented or explicitly failed
selected-logit exactness repair measured
candidate generator measured
support expansion measured
no fake/proxy/offload/loss/teacher violation
```

## Branch-delta system success

```text
Minimum diagnostic success
+
at least one vectorized/fused branch-delta path predicts safe-good
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
1. v9.2.52 boundary cannot be reproduced；
2. D0 cost cannot be attributed；
3. D0 vectorization cannot reduce cost；
4. real fused branch-delta cannot be implemented；
5. fused branch-delta loses safe-good predictivity；
6. all fused/selected branch-delta paths remain too expensive；
7. selected-logit path remains low agreement；
8. candidate generator cannot find enough safe-good events；
9. cascade controller cannot simultaneously meet precision / coverage / bad-event；
10. online support remains too narrow；
11. fresh natural oracle support collapses；
12. leave-dataset-out fails；
13. leave-stratum-out fails；
14. paired replay remains control-equivalent；
15. shuffle controls pass, indicating overfit；
16. short-run task drops；
17. full run gives no macro / hard-stratum / geometry gain；
18. functional breaks system gate；
19. gains are explained by QuadraticFeatureMLP；
20. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：D0 vectorization + fused branch-delta + controller + LDO/LSO + paired replay pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation and external robustness.

## Case B：D0 vectorization helps but fused delta still too expensive

必须声明：

```text
D0 orchestration was part of the blocker, but branch-delta kernelization remains system blocker.
```

下一步进入 CUDA/Triton fused branch-delta kernelization，不再调 threshold。

## Case C：fused branch-delta implemented but loses exact signal

必须声明：

```text
current fused approximation does not preserve exact gap-probe information.
```

下一步回到 exact branch-delta compression or richer output-delta statistics.

## Case D：selected-logit remains low agreement

必须声明：

```text
selected-logit path is useful at most for recall, not official accept.
```

不要把 selected-logit AUC 写成 functional success。

## Case E：candidate generator fails but oracle high

必须声明：

```text
good events exist, but legal candidate discovery cannot find them.
```

下一步设计 richer recall features; do not tune dataset.

## Case F：controller local pass but LDO/LSO fail

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

v9.2.53 的一句话策略是：

$$
\boxed{
\text{不要再堆 cheap scores；先把 D0 controller-prep 批量化，并真正实现 real fused branch-delta kernel。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
SLD0 有没有 AUC；
oracle support 是否存在；
C4 threshold 是否差一点；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. D0 role/controller scalar preparation 具体贵在哪里？
2. D0 是否能通过 vectorization / no-item / delayed logging / packed family lookup 降低到参考的 35% 以下？
3. FDP5 real fused kernel 是否能真正实现，而不是继续 not_implemented？
4. fused branch-delta 是否能保留 exact path 的 AUC 和 agreement？
5. selected-logit path 是否能修复 agreement，还是只能做 recall？
6. candidate generator 能否把 safe-good recall 拉到 >=0.80？
7. cascade controller 能否同时满足 precision / coverage / bad-event / system？
8. support 是否能扩展到 >=6 measured strata？
9. controller 能否 LDO/LSO？
10. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.53 的结果将给出清晰分叉：

```text
if vectorized D0 + real fused branch-delta + cascade + LDO/LSO + paired replay pass:
  strict PureKAN functional obtains local causal evidence.

if fused branch-delta predicts but too expensive:
  custom CUDA/Triton kernelization remains blocker.

if fused branch-delta becomes cheap but loses signal:
  exact gap-probe information cannot be compressed by current approximation.

if D0 vectorization fails:
  controller orchestration / support scoring is the actual system blocker.

if candidate discovery fails:
  legal recall / sufficient-statistic design remains blocker.

if controller local pass but leave-out fails:
  no dataset tuning; broaden signal-stratum/family support.

if oracle support collapses:
  carrier/support stability is blocker.
```
