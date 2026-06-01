# DG-KAN v9.2.68 True-Delta System Closure 与 Reference-Feasible Controller Promotion 完整实验计划

> 本计划基于 v9.2.67 `Borderline-Localized Bridge Repair 与 C0/C4 Pareto Calibration` 的真实执行结果制定。  
> v9.2.67 的 terminal route 是：
>
> ```text
> route = R17-ReferenceFeasibleButComputeFail
> base_candidate = LQ-t2-h256
> success_v9267_strict_purekan_functional = False
> success_v9267_full_functional = False
> success_v9267_external_ready = False
> ```
>
> v9.2.67 的关键事实是：
>
> ```text
> P1 C0/T2/C4/E2 autopsy:
>   pass = 1
>   T2 removed attribution = 1.0
>   E2 added attribution = 1.0
>   C0 false-positive attribution = 1.0
>   oracle miss attribution = 1.0
>   best backfillable mode = D3-overtrim-safe-useful
>   backfillable coverage = 0.004340
>
> P2 C0 fine trim:
>   best = T2e-MinimalDeletionConstrainedTrimV2
>   precision = 0.817241
>   coverage = 0.031966
>   bad-event = 0.020690
>   null-rate = 0.144828
>   precision LCB = 0.768711
>   bad-event UCB = 0.044396
>   C0 fine trim pass = 1
>
> P3 T2 coverage backfill:
>   best = B5-CoverageDeficitKnapsack
>   precision = 0.838028
>   coverage = 0.031305
>   bad-event = 0.024648
>   null-rate = 0.130282
>   precision LCB = 0.790716
>   bad-event UCB = 0.049995
>   T2 backfill pass = 1
>
> P4 C4/E2 filtered expansion:
>   best = X2-E2FilteredByLocalBadUCB
>   coverage = 0.050265
>   bad-event = 0.028509
>   bad-event UCB = 0.048161
>   Jaccard = 0.494135
>   diagnostic pass = 1
>   precision = 0.739035 < 0.75
>   null-rate = 0.214912 > 0.15
>   official pass = 0
>
> P5 constrained bridge fill-up:
>   best = C3-T2PlusBackfill
>   precision = 0.838028
>   coverage = 0.031305
>   bad-event = 0.024648
>   null-rate = 0.130282
>   precision LCB = 0.790716
>   bad-event UCB = 0.049995
>   bridge fill-up pass = 1
>
> P6 exact-reference deployable frontier v11:
>   best reference controller = C3-T2PlusBackfill
>   exact_reference_deployable = 1
>
> P7 true-delta compute v12:
>   best = TBD0-V9256CBD0Reference
>   AUC = 0.879072
>   bridge AUC = 0.811361
>   agreement = 1.0
>   step_ratio_q90 = 2.863280 > 1.50
>   true_delta_compute_pass = 0
>
> Downstream:
>   P8-P11 = not_run
>   reason = P7_true_delta_compute_failed
> ```
>
> v9.2.68 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.67 是路线性质的突破：decision geometry 首次过 reference deployable gate。}
> }
> $$
>
> 因此 v9.2.68 不应继续把主力放在：
>
> ```text
> 再设计一个新的 accept frontier；
> 再调 C0/T2/C4/E2 threshold；
> 再扩 support rows；
> 再重新定义 safe-useful label；
> 再用 oracle 查找新 controller；
> 按 dataset 单独调参。
> ```
>
> v9.2.68 的主线必须转为：
>
> $$
> \boxed{
> \text{把 C3-T2PlusBackfill reference controller 落成 system-legal true-delta controller。}
> }
> $$
>
> 换句话说，现在的主问题不是：
>
> ```text
> 是否存在 deployable exact-reference frontier？
> ```
>
> 这个问题 v9.2.67 已经回答：存在。
>
> 现在的主问题是：
>
> ```text
> 能否用真实 branch-delta / graph-free / no-proxy / MLP-comparable system path
> 复现 C3-T2PlusBackfill 的 accept decisions？
> ```

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

v9.2.68 的新增硬约束是：

```text
C3-T2PlusBackfill reference frontier 不允许被隐式改写。
如果为了系统实现修改 feature / threshold / event set，
必须重新经过 reference equivalence / agreement gate。
```

Official system controller 不得使用：

```text
source-measured gap
formula proxy
posthoc oracle label
validation/test metric at commit time
dataset_name branch
CPU offload
fake rows
duplicated balanced rows
```

允许使用：

```text
train-stream update/probe/candidate rows
true branch logits
true branch-delta tensors
C3/T2 backfill legal feature definitions
support/family reliability features
calibration-split thresholds
legal local risk / null / support statistics
```

---

# Part I. 对 v9.2.67 的独立判断

## 1. v9.2.67 没有达到最终目标

v9.2.67 没有 strict PureKAN functional success，原因是：

```text
exact_reference_deployable = 1
true_delta_compute_pass = 0
system_legal_controller = not_run
LDO / LSO = not_run
official paired replay = not_run
short-run / full-run = not_run
```

这意味着：

```text
不能声明 strict PureKAN local causal evidence；
不能声明 functional update 已经 official；
不能声明 paired replay 打赢 AdamWParallel / bestLR；
不能声明 external-ready；
不能声明 full functional success。
```

但这次失败和前几轮性质不同。过去 v9.2.58-v9.2.66 的主要失败是 reference decision geometry 不可部署；v9.2.67 第一次把这个门打开了。现在 blocker 明确转移到 system implementation。

## 2. v9.2.67 的真实进展

v9.2.67 的核心进展有四个。

第一，C0 broad seed 被证明可修。v9.2.66 里 C0 有 coverage，但 bad-event/confidence 不过。v9.2.67 的 `T2e-MinimalDeletionConstrainedTrimV2` 达到：

$$
Precision=0.817241,
$$

$$
Coverage=0.031966,
$$

$$
BadEvent=0.020690,
$$

$$
NullRate=0.144828,
$$

$$
PrecisionLCB=0.768711,
$$

$$
BadEventUCB=0.044396.
$$

这完整通过了 reference decision gate。说明 C0 broad seed 不是整体危险，只需要精细局部 trim。

第二，T2 safety core 能够回填 coverage。v9.2.66 的 T2 coverage 只有 `0.026124`，离 gate 差 `0.003876`。v9.2.67 的 `B5-CoverageDeficitKnapsack` 把 coverage 回到 `0.031305`，同时保持：

$$
Precision=0.838028,
$$

$$
BadEvent=0.024648,
$$

$$
NullRate=0.130282,
$$

$$
PrecisionLCB=0.790716,
$$

$$
BadEventUCB=0.049995.
$$

这说明 “safe core + constrained backfill” 是有效机制，而不是偶然 threshold。

第三，bridge fill-up 过 gate。P5 best `C3-T2PlusBackfill` 与 P3 backfill 指标一致，说明 final reference frontier 不再是 tiny clean slice，而是可部署 accept region：

$$
Coverage=0.031305 \in [0.03,0.15].
$$

第四，C4/E2 虽未 official，但有 diagnostic 价值。`X2-E2FilteredByLocalBadUCB` coverage `0.050265`、bad-event `0.028509`、bad UCB `0.048161`、Jaccard `0.494135`，说明 E2 仍可作为 high-recall candidate pool；只是 precision `0.739035` 与 null-rate `0.214912` 不能 official。

## 3. v9.2.67 的真实失败

v9.2.67 的真实失败只有一个主轴：

$$
\boxed{
\text{true-delta compute still too expensive。}
}
$$

当前 best compute row：

```text
TBD0-V9256CBD0Reference
AUC = 0.879072
bridge AUC = 0.811361
agreement = 1.0
step_ratio_q90 = 2.863280
memory_ratio previously around 0.969501
```

信号没有消失，agreement 也没有问题。问题是：

$$
StepRatio_{q90}=2.863280>1.50.
$$

距离 system gate：

$$
2.863280-1.50=1.363280.
$$

换成相对压缩目标：

$$
\frac{1.50}{2.863280}\approx0.524.
$$

也就是说，v9.2.68 至少要把 current true-delta path 的 q90 step cost 压掉约 `47.6%`，同时不能损失 accept agreement。

## 4. 问题的本质

当前本质不是 “functional update 失败”，也不是 “decision controller 失败”。当前本质是：

$$
\boxed{
\text{reference controller 已经可部署，但 official true-delta realization 没有 system-legal。}
}
$$

这把项目推进到了一个新阶段。

此前问题是：

```text
value / risk / null / support / oracle gap / bridge frontier 是否能定义可部署 accept region？
```

现在问题是：

```text
如何用真实 system path 便宜地计算这个 accept region？
```

这意味着下一步应该从 decision science 转向 system closure。更具体地说，v9.2.68 要分清三种成本来源：

```text
1. true branch-delta logits / branch replay 成本；
2. bridge feature compute 成本；
3. controller search / support-family aggregation 成本。
```

如果成本主要在 branch replay，就要做 fused / event-sparse / candidate-sparse branch-delta。  
如果成本主要在 bridge feature compute，就要做 cached family statistics / compact bridge score / local support table。  
如果成本主要在 frontier search，就要把 v9.2.67 的 search result freeze 成 fixed controller，不再在线搜索。

## 5. 是否还在正确道路上

是，而且 v9.2.67 是阶段性突破。当前最合理路线是：

```text
reference deployable frontier pass
→ true-delta compute closure
→ system-legal controller
→ LDO / LSO
→ official paired replay
→ short-run
→ full-run / robustness / strong baseline
```

错误路线是：

```text
继续设计新的 decision frontier；
继续调 C0/T2/C4/E2 threshold；
继续把 E2 diagnostic 当 official；
继续只追更高 AUC；
继续做 kernel-only 但不验证 accept agreement；
按 dataset 单独调 system path。
```

v9.2.68 应保留 C3-T2PlusBackfill 为 frozen reference target。任何系统优化都必须证明：

$$
Agreement(A_{\text{system}},A_{\text{reference}})\geq0.90.
$$

否则系统优化只是便宜 proxy，不是 official controller。

---

# Part II. v9.2.68 总体目标

v9.2.68 的总体目标是：

$$
\boxed{
\text{把 C3-T2PlusBackfill exact-reference controller 转化为 system-legal true-delta controller。}
}
$$

本轮必须并行回答七个问题：

```text
Q1:
  v9.2.67 boundary 是否稳定复现？

Q2:
  true-delta compute 的 residual cost 到底来自 branch logits、bridge features、support lookup 还是 frontier bookkeeping？

Q3:
  能否冻结 C3-T2PlusBackfill 的 reference search，把 online path 简化为 fixed accept rule？

Q4:
  能否用 event-sparse / candidate-sparse / two-stage cascade 只对少量 rows 做 expensive true-delta confirmation？

Q5:
  能否用 fused true-branch-delta kernel / compact bridge score 把 step_ratio_q90 压到 <=1.50？

Q6:
  system controller 是否能保持 reference precision / coverage / bad-event / null-rate / support balance？

Q7:
  如果 system controller 过，LDO/LSO 和 official paired replay 是否能打开？
```

v9.2.68 的核心 stop-go 是：

$$
\boxed{
\text{true-delta compute 是否能过 system envelope 且保持 C3-T2PlusBackfill accept agreement？}
}
$$

---

# Part III. 核心假设

## H1：v9.2.67 reference frontier 稳定可复现

H1 认为 `C3-T2PlusBackfill` 不是偶然 search artifact，而是稳定 reference frontier。

H1 成立标准：

复现 run 中：

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
NullRate_{\text{heldout}}\leq0.15,
$$

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05.
$$

且：

```text
best_reference_controller = C3-T2PlusBackfill
exact_reference_deployable = 1
fake/proxy/offload = 0
```

H1 失败标准：

C3-T2PlusBackfill 复现掉到：

```text
coverage < 0.03
bad-event UCB > 0.05
precision LCB < 0.75
```

若 H1 失败，不能进入 compute closure，必须回到 localized bridge frontier stability。

## H2：current step ratio 主要来自 expensive exact branch-delta confirmation，而不是 controller logic

H2 成立标准：

P1 residual attribution 显示：

```text
branch_delta_logits_or_replay_component_ratio >= 0.50
```

或：

```text
true_branch_delta_component_time dominates q90 step
```

H2 失败标准：

如果 cost 主要来自 support / bridge feature bookkeeping，则 v9.2.68 应优先做 feature table / lookup compaction，而不是 CUDA branch-delta kernel。

## H3：fixed C3-T2PlusBackfill controller 可以替代 online frontier search

v9.2.67 的 P5 已经确定 best controller。H3 认为 online path 不需要再搜索 trim/backfill/expansion thresholds，而只需执行 frozen rule。

H3 成立标准：

Frozen rule 与 reference search rule：

$$
Agreement\geq0.99
$$

on calibration split and:

$$
Agreement\geq0.95
$$

on heldout.

并且 frozen rule 的 feature compute time 低于 search rule：

$$
T_{\text{frozen}}\leq0.50T_{\text{search}}.
$$

H3 失败标准：

如果 frozen rule 在 heldout 上改变 accept composition 或 LCB/UCB，说明 v9.2.67 controller 仍依赖 search dynamics，不能 official freeze。

## H4：two-stage cascade 能把 expensive true-delta rows 降到可控比例

H4 认为不需要对所有 rows 做 full true branch-delta confirmation。可以先用 cheap legal prefilter 选出 candidate region，再对 candidates 做 exact true-delta confirmation。

Candidate rate：

$$
r_{\text{candidate}}=\frac{|A_{\text{prefilter}}|}{|E|}.
$$

H4 成立标准：

$$
Recall_{\text{reference-accepted}}\geq0.95,
$$

$$
CandidateRate\leq0.25,
$$

$$
BadEventRate_{\text{system}}\leq0.05,
$$

$$
Coverage_{\text{system}}\geq0.03.
$$

H4 失败标准：

如果 candidate rate 必须接近 1 才能保 recall，则 cascade 不能降成本。

## H5：fused / compact bridge-score compute 可以保持 signal while reducing step ratio

H5 成立标准：

至少一个 system candidate 满足：

$$
AUC_{\text{safe-good}}\geq0.70,
$$

$$
AUC_{\text{bridge-accept}}\geq0.70,
$$

$$
Agreement_{\text{reference-accept}}\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

H5 失败标准：

signal lost or:

$$
StepRatio_{q90}>1.50.
$$

## H6：如果 reference and compute both pass，则 LDO/LSO 是下一个真实 scientific gate

H6 成立标准：

system controller 在 leave-dataset-out / leave-stratum-out 中 task-safe，并至少在两个 heldout splits 上不输 AdamWParallel / bestLR beyond tolerance。

H6 失败标准：

system controller 只能在 pooled calibration 里过，leave-out 崩溃。此时不能 dataset-specific 调参，必须修 dataset-agnostic family/support reliability。

---

# Part IV. System architecture design

## 1. Reference controller freeze

v9.2.67 reference controller：

$$
A_{\text{ref}}=C3\text{-}T2PlusBackfill.
$$

其 official reference metrics：

$$
Precision=0.838028,
$$

$$
Coverage=0.031305,
$$

$$
BadEvent=0.024648,
$$

$$
NullRate=0.130282,
$$

$$
PrecisionLCB=0.790716,
$$

$$
BadEventUCB=0.049995.
$$

v9.2.68 将其作为 frozen target：

$$
Accept_{\text{ref}}(e)
=
\mathbb{1}[e\in A_{\text{C3-T2PlusBackfill}}].
$$

任何 system candidate 必须近似：

$$
Accept_{\text{sys}}(e)\approx Accept_{\text{ref}}(e).
$$

## 2. System controller decomposition

System path 不再执行 search，而是执行：

```text
Stage 0:
  cheap event eligibility / no-event filter

Stage 1:
  cheap legal prefilter:
    C0-compatible low-risk region
    T2-compatible safety core
    family/support basic reliability

Stage 2:
  exact true branch-delta only for candidates

Stage 3:
  frozen bridge score:
    C0 fine trim rule
    T2 backfill rule
    support-family balance rule

Stage 4:
  accept / abstain / reject
```

Mathematically：

$$
Accept_{\text{sys}}(e)
=
Prefilter(e)
\land
ExactConfirm(e)
\land
FrozenBridge(e).
$$

## 3. Event-sparse exact confirmation

Let $E$ be all possible events. Let $P(E)$ be prefilter survivors.

$$
P(E)=\{e:S_{\text{cheap}}(e)\geq\tau_p\}.
$$

Exact confirmation only runs on $P(E)$:

$$
ExactConfirm(e)
=
\mathbb{1}[S_{\text{true-delta}}(e)\geq\tau_d],
\quad e\in P(E).
$$

Cost model：

$$
T_{\text{sys}}
=
T_{\text{base}}
+
r_{\text{candidate}}T_{\text{exact-delta}}
+
T_{\text{cheap-prefilter}}
+
T_{\text{bridge-lookup}}.
$$

To pass system gate：

$$
\frac{T_{\text{sys}}}{T_{\text{MLP}}}\leq1.50.
$$

If current exact path has ratio `2.863280`, candidate sparsity alone requires roughly:

$$
r_{\text{candidate}}
\leq
\frac{1.50-1.00}{2.863280-1.00}
\approx0.268.
$$

So the practical cascade target is:

$$
CandidateRate\leq0.25.
$$

## 4. Bridge feature compaction

Bridge score should not recompute large tables online. Precompute calibration/family constants and use lookup tables:

```text
family_id -> safe-useful LCB
family_id -> bad UCB
family_id -> null UCB
family_id -> support density
event bucket -> C0 trim flag
event bucket -> T2 backfill priority
event bucket -> bridge accept threshold
```

Online bridge compute becomes:

$$
S_{\text{bridge}}(e)
=
a_1S_{\text{true-delta}}(e)
+
a_2LCB_{\text{family}}(e)
-
a_3UCB_{\text{bad-family}}(e)
-
a_4UCB_{\text{null-family}}(e)
+
a_5S_{\text{backfill-priority}}(e).
$$

Accept:

$$
Accept(e)=
\mathbb{1}[S_{\text{bridge}}(e)\geq\tau]
\land
\mathbb{1}[UCB_{\text{bad}}(e)\leq\rho_b]
\land
\mathbb{1}[UCB_{\text{null}}(e)\leq\rho_n].
$$

## 5. Kernel targets

System candidates must be grouped by what they optimize:

```text
K-family:
  true branch-delta kernel / branch logits path

F-family:
  frozen bridge feature table / lookup path

C-family:
  candidate cascade / event-sparse exact confirmation

H-family:
  hybrid: cascade + fused delta + compact bridge
```

---

# Part V. Candidate designs

## 1. Residual attribution candidates

### RA0：v9.2.67 TBD0 reference

Reference only.

### RA1：Subphase timer

Measure:

```text
branch logits construction
branch replay forward
delta tensor construction
bridge score compute
family/support lookup
LCB/UCB compute
threshold / accept bookkeeping
synchronization overhead
memory allocation overhead
```

### RA2：Kernel count / allocation audit

Record:

```text
kernel_count
sync_count
allocated_MB
reserved_MB
read_MB
write_MB
temporary_tensor_count
largest_temp_tensor_MB
```

### RA3：Candidate-rate cost simulator

Simulate step ratio under candidate rates:

$$
r\in\{0.05,0.10,0.15,0.20,0.25,0.30,0.50,1.00\}.
$$

## 2. Frozen controller candidates

### FR0：Search reference

v9.2.67 exact P5/P6 reference. Diagnostic only.

### FR1：FrozenC3Rule

Freeze C3-T2PlusBackfill thresholds / buckets / backfill flags.

### FR2：FrozenC3LookupTable

Same as FR1 but all family/support metrics served by compact lookup table.

### FR3：FrozenC3MinimalFeature

Ablate bridge features to minimal subset while maintaining agreement.

### FR4：FrozenC3QuantizedBuckets

Quantize risk/null/support buckets to reduce compute.

## 3. Cheap prefilter candidates

### PF0：No prefilter

Reference; exact confirmation for all rows.

### PF1：C0RegionPrefilter

Accept candidates if they fall into C0-compatible legal region.

### PF2：T2SafetyCorePrefilter

Accept candidates near T2 safety core.

### PF3：BackfillPriorityPrefilter

Accept candidates if they match D3-overtrim-safe-useful backfillable mode.

### PF4：UnionHighRecallPrefilter

$$
PF4 = PF1 \lor PF2 \lor PF3.
$$

### PF5：LearnedMonotoneCheapPrefilter

Calibration-only monotone score using legal features:

```text
control gap bucket
horizon bucket
family safe-useful LCB
bad UCB
null UCB
support density
role bucket
delta efficiency bucket
```

No dataset name.

## 4. Exact confirmation candidates

### EC0：TBD0 exact reference

Current exact path.

### EC1：EventSparseTBD0

Run TBD0 only on prefilter candidates.

### EC2：CachedBranchLogitsTBD

Cache branch logits for candidate families inside step.

### EC3：SelectedLogitExactBridge

Compute exact branch-delta only on logits needed for bridge accept decision.

Must prove:

$$
Agreement_{\text{accept}}\geq0.90.
$$

### EC4：FullLogitSmallC

Use smaller candidate count but full logits for selected branches.

### EC5：FusedBranchDeltaBridgeKernel

Triton/CUDA fused path:

```text
input logits
candidate branch delta
bridge score partials
accept logits / score
```

### EC6：TwoPassConfirm

First pass approximate selected-logit; second pass exact full-logit only for borderline rows.

## 5. Bridge system candidates

### BS0：ReferenceSearchAllRows

Diagnostic reference.

### BS1：FrozenAllRows

Frozen C3 rule but still all rows exact confirmation.

### BS2：FrozenPrefilterExact

Cheap prefilter + exact confirmation.

### BS3：FrozenPrefilterCached

Cheap prefilter + cached branch logits.

### BS4：FrozenPrefilterSelectedLogit

Cheap prefilter + selected-logit exact bridge.

### BS5：FrozenPrefilterTwoPass

Cheap prefilter + selected first pass + exact borderline second pass.

### BS6：FusedBridgeSystem

Fused branch-delta + compact bridge lookup.

### BS7：HybridBestSystem

Best combination from PF / EC / FR candidates.

---

# Part VI. 实验阶段

## P0：v9.2.67 boundary reproduction

### 目标

确认 reference frontier 真的稳定，避免在不稳定 controller 上做 system closure。

### 必须记录

```text
route
source_route_v9266
C0_fine_trim_pass
T2_backfill_pass
C4_filtered_expansion_pass
bridge_fillup_pass
exact_reference_deployable
reference_precision
reference_coverage
reference_bad_event
reference_null_rate
reference_precision_lcb
reference_bad_event_ucb
true_delta_compute_pass
true_delta_step_ratio_q90
true_delta_bridge_auc
true_delta_agreement
fake_proxy_count
cpu_offload_used
```

### 判断标准

P0 pass：

```text
route = R17-ReferenceFeasibleButComputeFail
exact_reference_deployable = 1
reference controller = C3-T2PlusBackfill
true_delta_compute_pass = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_reference_compute_gate_ladder.svg
p0_v9266_to_v9267_progress_ladder.svg
p0_C0_T2_C4_E2_to_C3_frontier.svg
```

---

## P1：true-delta residual attribution

### 目标

找到 `step_ratio_q90 = 2.863280` 的主要来源。

### 必须记录

```text
candidate_id
phase
subphase
time_ms_mean
time_ms_q90
time_ratio_vs_mlp
component_ratio
memory_MB
kernel_count
sync_count
temp_tensor_count
read_MB
write_MB
dominant_subphase
unknown_fraction
```

Subphases：

```text
S1-branch_logits_forward
S2-branch_delta_tensor
S3-control_gap_compute
S4-risk_null_support_lookup
S5-bridge_score_compute
S6-LCB_UCB_compute
S7-accept_bookkeeping
S8-sync_allocation_overhead
S9-other_unknown
```

### 判断标准

P1 pass：

```text
dominant_subphase identified
unknown_fraction <= 0.10
at least one removable/compressible subphase with component_ratio >= 0.25
```

### 可视化

```text
p1_true_delta_cost_stack.svg
p1_kernel_count_by_subphase.svg
p1_memory_temp_tensor_breakdown.svg
p1_step_ratio_waterfall.svg
```

---

## P2：frozen reference controller equivalence

### 目标

冻结 C3-T2PlusBackfill，证明 frozen rule 与 reference search 等价。

### 必须记录

```text
frozen_rule_id
reference_controller_id
features_used
thresholds
lookup_tables_used
agreement_cal
agreement_heldout
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
precision_lcb
bad_event_ucb
feature_compute_time_ms
search_compute_time_ms
time_reduction
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

Frozen controller pass：

$$
Agreement_{\text{heldout}}\geq0.95,
$$

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
NullRate_{\text{heldout}}\leq0.15.
$$

Compute diagnostic：

$$
T_{\text{frozen}}\leq0.50T_{\text{search}}.
$$

### 可视化

```text
p2_frozen_vs_reference_agreement.svg
p2_feature_ablation_agreement.svg
p2_frozen_compute_reduction.svg
p2_accept_set_overlap.svg
```

---

## P3：cheap prefilter / candidate cascade

### 目标

降低 exact true-delta confirmation 的 row count，同时保持 reference accept recall。

### 必须记录

```text
prefilter_id
features_used
candidate_rate
reference_accept_recall
reference_accept_precision
coverage_after_prefilter
bad_event_after_prefilter
null_rate_after_prefilter
candidate_bad_event
candidate_null_rate
prefilter_time_ms
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

Prefilter pass：

$$
Recall_{\text{reference-accepted}}\geq0.95,
$$

$$
CandidateRate\leq0.25.
$$

Safety diagnostic：

$$
BadEventRate_{\text{candidate}}\leq0.20,
$$

$$
NullRate_{\text{candidate}}\leq0.40.
$$

If no prefilter reaches candidate rate `<=0.25`, run candidate-rate simulation to determine exact compute target.

### 可视化

```text
p3_candidate_rate_recall_curve.svg
p3_prefilter_precision_coverage_bad.svg
p3_prefilter_feature_ablation.svg
p3_event_sparse_cost_projection.svg
```

---

## P4：event-sparse exact confirmation

### 目标

只在 P3 candidate rows 上运行 exact true-delta confirmation，测真实 step ratio。

### 必须记录

```text
exact_candidate_id
prefilter_id
candidate_rate
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
AUC_safe_good
AUC_bridge_accept
agreement_reference_accept
precision
coverage
bad_event
null_rate
precision_lcb
bad_event_ucb
step_ratio_q90
memory_ratio
kernel_count
sync_count
```

### 判断标准

Event-sparse exact pass：

$$
Agreement_{\text{reference-accept}}\geq0.90,
$$

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05,
$$

$$
NullRate\leq0.15,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Diagnostic pass：

$$
StepRatio_{q90}\leq2.00
$$

with:

$$
Agreement_{\text{reference-accept}}\geq0.90.
$$

### 可视化

```text
p4_event_sparse_step_ratio.svg
p4_candidate_rate_vs_step_ratio.svg
p4_exact_confirmation_agreement.svg
p4_system_frontier.svg
```

---

## P5：fused / compact bridge compute

### 目标

进一步压缩 feature and bridge compute，避免 event-sparse 仍超 system envelope。

### 必须记录

```text
system_candidate_id
prefilter_id
exact_candidate_id
bridge_feature_candidate_id
uses_fused_kernel
uses_compact_lookup
uses_quantized_bucket
uses_true_delta
agreement_reference_accept
AUC_bridge_accept
precision
coverage
bad_event
null_rate
precision_lcb
bad_event_ucb
step_ratio_q90
memory_ratio
dominant_subphase
read_MB
write_MB
kernel_count
```

### 判断标准

Fused bridge compute pass：

$$
Agreement_{\text{reference-accept}}\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Decision gate must remain:

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05,
$$

$$
NullRate\leq0.15.
$$

### 可视化

```text
p5_fused_bridge_cost_signal_pareto.svg
p5_subphase_before_after.svg
p5_lookup_compaction_effect.svg
p5_kernel_count_reduction.svg
```

---

## P6：system-legal exact-signal controller v1

### 目标

建立 official-eligible controller。只有 P2 frozen equivalence + P4/P5 system compute pass 同时成立时才能通过。

### 必须记录

```text
controller_id
prefilter_id
exact_candidate_id
bridge_system_candidate_id
thresholds
calibration_split_id
heldout_split_id
precision_cal
coverage_cal
bad_event_cal
null_rate_cal
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
precision_lcb
bad_event_ucb
accepted_strata_count
accepted_family_count
max_family_share
max_stratum_share
AUC_safe_good
AUC_bridge_accept
agreement_reference_accept
step_ratio_q90
memory_ratio
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
official_eligible
```

### 判断标准

System-legal controller pass：

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
NullRate_{\text{heldout}}\leq0.15,
$$

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05,
$$

$$
Agreement_{\text{reference-accept}}\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Support balance：

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

### 可视化

```text
p6_system_controller_frontier.svg
p6_reference_vs_system_accept_overlap.svg
p6_system_cost_vs_decision_quality.svg
p6_family_strata_balance.svg
```

---

## P7：Leave-dataset-out / leave-stratum-out

### 目标

证明 system controller 不是 dataset-specific 或 stratum-specific artifact。

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
prefilter_id
exact_candidate_id
bridge_system_candidate_id
threshold
precision
coverage
bad_event_rate
null_rate
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

At least $70\%$ held-out strata task-safe and:

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

## P8：Official paired replay scout

### 目标

在 official system controller 上验证 RealFunctional 是否有 local causal advantage。

### 设置

```text
base = R2 repaired base checkpoint
controller_id = best P7 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledTrueBranchDelta, ShuffledBridgeController,
           ShuffledPrefilter, ShuffledExactConfirm, ShuffledSupportStat,
           ShuffledControlGain, ShuffledCandidateGate, ShuffledBranchRatio,
           ShuffledSignalChannel, FunctionalChannelShuffled, TailMaskShuffled,
           RoleScoreShuffled, DatasetRouteShuffled, EventRouteShuffled,
           InvertedRoleMask
```

### 必须记录

```text
controller_id
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
null_rate
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

Shuffle controls must fail:

```text
ShuffledTrueBranchDelta = fail
ShuffledBridgeController = fail
ShuffledPrefilter = fail
ShuffledExactConfirm = fail
ShuffledSupportStat = fail
ShuffledControlGain = fail
ShuffledCandidateGate = fail
ShuffledBranchRatio = fail
ShuffledSignalChannel = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

System:

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

如果 P8 paired replay passes，验证 local advantage 能否进入 continuous training。

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
null_rate
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

Mechanism pass, at least one:

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

## P10：Full run / robustness / strong baseline

### 目标

只有 P9 pass 后打开。验证 functional advantage 不是局部 replay artifact。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
controls = AdamWOnly, AdamWParallel, bestLR, StrongLRGrid, QuadraticFeatureMLP,
           NoOp, Random, ShuffledTrueBranchDelta, ShuffledBridgeController
```

### 必须记录

```text
dataset
seed
candidate
final_acc
best_val_acc
test_acc
train_loss
val_loss
test_loss
ECE
NLL
CEp99
margin_p10
curvature
local_lipschitz
functional_event_count
coverage
bad_event_rate
null_rate
step_ratio_q90
memory_ratio
strong_baseline_beaten
robustness_pass
base_checkpoint_hash
```

### 判断标准

Full functional pass:

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005
$$

and at least one:

$$
Acc_{\text{functional}}>Acc_{\text{AdamWParallel}},
$$

$$
ECE_{\text{functional}}<ECE_{\text{AdamW}},
$$

$$
NLL_{\text{functional}}<NLL_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

Strong baseline pass:

```text
Functional not explained by QuadraticFeatureMLP
Functional not explained by LR grid
Functional not explained by shuffled controller
```

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9268.csv
p0_v9267_boundary_reproduction.csv
p1_true_delta_residual_attribution.csv
p2_frozen_reference_controller_equivalence.csv
p3_cheap_prefilter_candidate_cascade.csv
p4_event_sparse_exact_confirmation.csv
p5_fused_compact_bridge_compute.csv
p6_system_legal_exact_signal_controller_v1.csv
p7_leave_dataset_and_stratum_out.csv
p8_official_paired_replay_scout.csv
p9_short_run_functional_validation.csv
p10_full_run_robustness_strong_baseline.csv
true_delta_residual_trace_v9268.csv
frozen_controller_trace_v9268.csv
prefilter_cascade_trace_v9268.csv
event_sparse_exact_trace_v9268.csv
fused_bridge_compute_trace_v9268.csv
system_controller_trace_v9268.csv
leaveout_trace_v9268.csv
paired_replay_branch_trace_v9268.csv
short_run_trace_v9268.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy:

```text
F1_contract_violation
F2_v9267_boundary_unstable
F3_dataset_tuning_detected
F4_reference_controller_not_reproducible
F5_true_delta_residual_unattributed
F6_frozen_controller_agreement_fail
F7_frozen_controller_decision_gate_fail
F8_prefilter_recall_fail
F9_prefilter_candidate_rate_too_high
F10_event_sparse_exact_system_fail
F11_event_sparse_agreement_fail
F12_fused_bridge_compute_signal_lost
F13_fused_bridge_compute_system_fail
F14_system_controller_precision_fail
F15_system_controller_coverage_fail
F16_system_controller_bad_event_fail
F17_system_controller_null_rate_fail
F18_system_controller_lcb_ucb_fail
F19_true_delta_compute_still_expensive
F20_true_delta_signal_lost
F21_leave_dataset_out_fail
F22_leave_stratum_out_fail
F23_paired_replay_control_equivalent
F24_shuffle_control_pass
F25_functional_lr_equivalent
F26_short_run_task_drop
F27_full_run_no_macro_hard_stratum_gain
F28_strong_baseline_explains_gain
F29_robustness_fail
F30_external_not_ready
F31_fake_or_proxy_violation
F32_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.67 boundary reproduced.

R2-FrozenReferenceControllerPass:
  C3-T2PlusBackfill frozen rule matches reference search and preserves decision metrics.

R3-TrueDeltaResidualAttributed:
  cost subphase of true-delta compute is localized.

R4-PrefilterCascadePass:
  cheap prefilter reaches high recall and low candidate rate.

R5-EventSparseExactPass:
  exact true-delta confirmation becomes system-legal through candidate sparsity.

R6-FusedBridgeComputePass:
  fused / compact bridge system passes cost and signal gates.

R7-SystemLegalExactSignalControllerPass:
  system controller passes decision + compute gates.

R8-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R9-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R10-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R11-ShortRunFunctionalPass:
  short-run task-safe mechanism gain.

R12-FullFunctionalPass:
  full run task/geometry/system/control gates pass.

R13-FrozenControllerUnstable:
  v9.2.67 reference controller cannot be frozen/reproduced.

R14-PrefilterRecallCostTradeoffFail:
  candidate cascade cannot reduce exact rows without losing reference accept recall.

R15-EventSparseStillExpensive:
  candidate sparsity helps but step ratio remains >1.50.

R16-FusedComputeSignalLost:
  system optimization loses reference signal/agreement.

R17-ReferenceFeasibleButComputeFail:
  decision geometry viable but true-delta compute still too expensive.

R18-ComputePassButLeaveoutFail:
  system controller overfits calibration distribution.

R19-ComputePassButPairedReplayFail:
  controller is system-legal but not causally better than controls.

R20-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9267_boundary_pass
dataset_tuning_detected
reference_controller_id
reference_controller_reproduced
reference_precision
reference_coverage
reference_bad_event
reference_null_rate
reference_precision_lcb
reference_bad_event_ucb
frozen_controller_id
frozen_controller_pass
frozen_reference_agreement
true_delta_residual_attribution_pass
dominant_subphase
dominant_subphase_ratio
unknown_fraction
best_prefilter_id
prefilter_pass
candidate_rate
reference_accept_recall
best_exact_candidate_id
event_sparse_exact_pass
exact_agreement
exact_step_ratio_q90
exact_memory_ratio
best_fused_bridge_id
fused_bridge_compute_pass
fused_bridge_step_ratio_q90
fused_bridge_memory_ratio
best_system_controller_id
system_legal_controller_pass
controller_precision
controller_coverage
controller_bad_event
controller_null_rate
controller_precision_lcb
controller_bad_event_ucb
controller_reference_agreement
controller_step_ratio_q90
controller_memory_ratio
accepted_signal_strata_count
accepted_family_count
max_family_share
max_stratum_share
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
robustness_pass
strong_baseline_pass
external_ready
primary_blocker
next_required_implementation
success_v9268_strict_purekan_functional
success_v9268_full_functional
success_v9268_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 residual attribution
  P2 frozen controller equivalence
  P3 cheap prefilter cascade
  P4 event-sparse exact confirmation
  P5 fused / compact bridge compute

Batch 2:
  P6 system-legal controller
  P7 leave-dataset-out / leave-stratum-out
  P8 paired replay scout

Batch 3:
  official P8 paired replay
  P9 short-run if P8 passes

Batch 4:
  P10 full run / robustness / strong baseline only if P9 passes
```

Gate rule:

```text
P1-P5 can run in parallel.
P6 cannot pass unless:
  P0 pass
  P2 frozen equivalence pass
  P4 or P5 compute pass
P7/P8 diagnostic rows may be measured before all gates finish,
but official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  reference controller reproduced
  frozen controller pass
  true branch-delta legality pass
  true branch-delta predictivity pass
  true branch-delta agreement pass
  true branch-delta system pass
  system-legal controller pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.67 boundary reproduced
reference controller frozen or failure explained
true-delta residual cost attributed
prefilter cascade measured
event-sparse exact confirmation measured
fused / compact bridge compute measured
system controller measured
no fake/proxy/offload/loss/teacher violation
```

## Decision-system success

```text
Minimum diagnostic success
+
frozen reference controller pass
+
event-sparse or fused compute pass
+
system-legal exact-signal controller pass
```

## Local functional success

```text
Decision-system success
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
+
robustness / strong baseline pass
```

## Failure stop

```text
1. v9.2.67 boundary cannot be reproduced；
2. C3-T2PlusBackfill cannot be frozen；
3. frozen controller changes accept composition；
4. true-delta residual cost cannot be attributed；
5. no cheap prefilter reaches recall >=0.95；
6. candidate rate cannot go below 0.25；
7. event-sparse exact path still step_ratio_q90 >1.50；
8. fused bridge compute loses signal/agreement；
9. system controller cannot meet precision / coverage / bad-event / null-rate；
10. precision LCB below 0.75；
11. bad-event UCB above 0.05；
12. leave-dataset-out fails；
13. leave-stratum-out fails；
14. paired replay remains control-equivalent；
15. shuffle controls pass；
16. short-run task drops；
17. full run gives no macro / hard-stratum / geometry gain；
18. functional breaks system gate；
19. gains are explained by QuadraticFeatureMLP；
20. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：system controller pass + LDO/LSO + paired replay pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full run and robustness.

## Case B：reference stable but compute fail

必须声明：

```text
decision geometry is solved at reference level, but true-delta implementation remains system blocker.
```

下一步继续 kernel / event-sparse / fused branch-delta path，不调 decision frontier。

## Case C：frozen controller unstable

必须声明：

```text
v9.2.67 reference pass depended on search dynamics; controller is not yet stable enough for system promotion.
```

下一步回到 reference frontier freeze and stability.

## Case D：prefilter fails

必须声明：

```text
exact confirmation cannot be sparsified without losing reference accept recall.
```

下一步优先做 fused full exact path rather than cascade.

## Case E：compute pass but paired replay fail

必须声明：

```text
controller is system-legal but not causally superior to matched controls.
```

下一步回到 functional event/value target, not kernelization.

## Case F：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能通过 dataset-specific tuning 写成功。

---

# Part XII. 最终建议

v9.2.68 的一句话策略是：

$$
\boxed{
\text{reference frontier 已经过了；不要继续调 decision，集中火力把 true-delta controller 做到 system-legal。}
}
$$

当前最关键的问题不是：

```text
oracle 是否存在；
support 是否够；
C0/T2/C4/E2 是否能 bridge；
safe-useful target 是否可行；
null / bad / risk score 是否有信号；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. C3-T2PlusBackfill 能否稳定 frozen？
2. true-delta step_ratio_q90 = 2.863280 的主成本来自哪里？
3. candidate cascade 能否把 exact rows 降到 <=25%？
4. event-sparse exact confirmation 能否保持 agreement >=0.90？
5. fused bridge / compact lookup 能否把 step_ratio_q90 压到 <=1.50？
6. system controller 能否保持 reference precision / coverage / bad-event / null-rate？
7. system controller 能否 LDO/LSO？
8. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.68 的结果将给出清晰分叉：

```text
if system controller passes and paired replay passes:
  strict PureKAN functional route obtains local causal evidence.

if reference stable but compute still fails:
  kernelization / event-sparse true-delta remains primary blocker.

if compute passes but paired replay fails:
  functional event target still not causally superior.

if frozen reference unstable:
  return to reference controller stabilization.

if LDO/LSO fails:
  repair dataset-agnostic / stratum-agnostic support-family reliability, not dataset-specific tuning.
```
