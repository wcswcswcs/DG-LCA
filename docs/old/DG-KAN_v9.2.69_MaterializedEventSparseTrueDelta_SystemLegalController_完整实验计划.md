# DG-KAN v9.2.69 Materialized Event-Sparse True-Delta Kernel 与 System-Legal Controller Closure 完整实验计划

> 本计划基于 v9.2.68 `True-Delta System Closure 与 Reference-Feasible Controller Promotion` 的真实执行结果制定。  
> v9.2.68 的 terminal route 是：
>
> ```text
> route = R17-ReferenceFeasibleButComputeFail
> base_candidate = LQ-t2-h256
> success_v9268_strict_purekan_functional = False
> success_v9268_full_functional = False
> success_v9268_external_ready = False
> ```
>
> v9.2.68 的关键事实是：
>
> ```text
> P0:
>   v9.2.67 boundary reproduced
>   reference_controller_id = C3-T2PlusBackfill
>   exact_reference_deployable = 1
>   true_delta_compute_pass = 0
>   fake/proxy/offload = 0
>
> P1 true-delta residual attribution:
>   pass = 1
>   dominant_subphase = S1-branch_logits_forward
>   dominant_subphase_ratio = 0.396711
>   branch_delta_logits_or_replay_combined_ratio = 0.440701
>   unknown_fraction = 0.043181
>
> P2 frozen controller:
>   best = FR1-FrozenC3Rule
>   frozen_controller_pass = 1
>   heldout agreement = 1.0
>   precision = 0.838028
>   coverage = 0.031305
>   bad-event = 0.024648
>   null-rate = 0.130282
>
> P3 cheap prefilter:
>   best = PF5-LearnedMonotoneCheapPrefilter
>   prefilter_pass = 1
>   candidate_rate = 0.103051
>   reference_accept_recall = 1.0
>
> P4 event-sparse exact:
>   best = EC1-EventSparseTBD0CostProjection
>   agreement = 1.0
>   projected_step_ratio_q90 = 1.227012
>   materialized_system_path = 0
>   official pass = 0
>
> P5 fused / compact bridge:
>   best = BS2-FrozenPrefilterExactProjection
>   agreement = 1.0
>   projected_step_ratio_q90 = 1.227012
>   materialized_system_path = 0
>   official pass = 0
>
> P6 system controller:
>   system_legal_controller_pass = 0
>
> P7-P10:
>   not_run
>   reason = compute/system gate not passed
>
> primary_blocker:
>   true_delta_system_path_not_materialized
> ```
>
> v9.2.69 的核心判断是：
>
> $$
> \boxed{
> \text{reference decision geometry 已经解决；现在必须把 projected event-sparse true-delta 变成 materialized system path。}
> }
> $$
>
> 因此 v9.2.69 不应继续做：
>
> ```text
> 继续调 C3/T2/C4/E2 frontier；
> 继续重设 safe-useful / null / bad-event target；
> 继续增加 oracle / reference controller；
> 继续把 projected step ratio 当作 official；
> 继续用 source-measured gap 或 formula proxy 替代 true branch-delta；
> 按 dataset 单独调 prefilter 或 threshold。
> ```
>
> v9.2.69 的主线必须是：
>
> $$
> \boxed{
> \text{materialize PF5 + FrozenC3 + true branch-delta exact confirm，并用真实 step timing 关闭 system gate。}
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

v9.2.69 新增硬约束是：

```text
projection rows 不能 official；
projected_step_ratio 只能作为 diagnostic；
materialized_system_path 必须为 1；
must time actual executed kernels, not recompute cost from full trace;
must prove true branch-delta legality / agreement / system envelope on same candidate;
```

本轮 official system controller 不得使用：

```text
source_measured_gap
formula_proxy
posthoc oracle label
validation/test metric at commit time
dataset_name branch
CPU offload
fake rows
duplicated balanced rows
projected-only cost estimate
full-trace-derived sparse projection as pass
```

允许使用：

```text
train-stream update/probe/candidate rows
PF5 learned monotone cheap legal prefilter
FR1 frozen C3-T2PlusBackfill rule
true branch logits
true branch-delta tensors
compact support/family lookup
materialized event-sparse branch forward
materialized fused bridge score
calibration-split frozen thresholds
```

---

# Part I. 对 v9.2.68 的独立判断

## 1. v9.2.68 没有达到最终目标

v9.2.68 没有 strict PureKAN functional success。原因不是 reference controller 不可部署，而是 system path 还没有 materialize：

```text
reference_controller_reproduced = 1
frozen_controller_pass = 1
prefilter_pass = 1
event_sparse_exact_diagnostic_pass = 1
fused_bridge_diagnostic_pass = 1
event_sparse_exact_pass = 0
fused_bridge_compute_pass = 0
system_legal_controller_pass = 0
paired replay / short-run / full-run = not_run
```

因此不能声明：

```text
strict PureKAN local causal evidence
system-legal functional controller
official paired replay pass
short-run functional pass
full functional success
external-ready
```

尤其不能把：

```text
projected step ratio = 1.227012
candidate_rate = 0.103051
frozen agreement = 1.0
reference controller reproduced
```

写成 official system success。它们证明路径有希望，但不是已经实现。

## 2. v9.2.68 的真实进展

v9.2.68 有三个非常重要的正结果。

第一，`C3-T2PlusBackfill` 的 reference frontier 稳定复现，而且可以 frozen。Reference metrics 仍满足 gate：

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

这说明 v9.2.67 的 reference breakthrough 不是偶然 search artifact。

第二，cheap prefilter 真的有效。`PF5-LearnedMonotoneCheapPrefilter` 得到：

$$
CandidateRate=0.103051,
$$

$$
ReferenceAcceptRecall=1.0.
$$

如果这个 candidate sparsity 能 materialize，理论上足够把 exact confirmation 成本压进 system envelope。此前 v9.2.52-v9.2.56 的 candidate generator 问题是 recall 不够或 candidate bad-event 高；v9.2.68 的 PF5 不再是这个问题。

第三，cost projection 显示 system envelope 有希望。Event-sparse / fused path projected step ratio 为：

$$
StepRatio_{\text{projected}}=1.227012.
$$

这低于 official gate：

$$
1.227012<1.50.
$$

所以当前不是“数学上不可能快”，而是“还没有把这个低成本路径落成真实执行路径”。

## 3. v9.2.68 的真实失败

v9.2.68 的失败非常明确：

$$
\boxed{
\text{event-sparse / fused true-delta 仍是 projection，不是 materialized system path。}
}
$$

`EC1-EventSparseTBD0CostProjection` 与 `BS2-FrozenPrefilterExactProjection` 共同的问题是：

```text
agreement = 1.0
projected_step_ratio_q90 = 1.227012
materialized_system_path = 0
official pass = 0
```

这说明当前低 step ratio 来自 measured full-exact trace 的 sparse cost projection，而不是实际执行了：

```text
PF5 prefilter
candidate compaction
candidate-only true branch forward
candidate-only branch-delta
frozen bridge lookup
materialized accept decision
```

所以 v9.2.69 必须把 projection audit 改成 materialized audit。

## 4. 当前 blocker 的本质

当前 blocker 应写成：

$$
\boxed{
\text{true-delta system path not materialized, despite reference and cascade feasibility.}
}
$$

更具体地说：

```text
1. decision geometry 已经过 reference gate；
2. frozen controller 已经过 agreement gate；
3. cheap prefilter 已经把 candidate rate 降到 10.3%；
4. projection 显示 step ratio 可达 1.227；
5. 但 official path 仍未真实运行 candidate-sparse true branch-delta；
6. branch logits forward 是 dominant cost；
7. downstream 不能打开，因为 system_legal_controller_pass = 0。
```

因此 v9.2.69 的核心不是再找一个更聪明的 controller，而是回答：

```text
如何把 full-trace projection 变成 actual event-sparse execution？
```

## 5. 是否还在正确道路上

是，而且路线已经进入 system closure 阶段。

正确路线现在是：

```text
reference deployable frontier pass
→ frozen controller pass
→ cheap prefilter pass
→ materialized event-sparse true-delta
→ fused / compact bridge compute
→ system-legal controller
→ LDO / LSO
→ official paired replay
→ short-run / full-run
```

错误路线是：

```text
继续调 reference frontier；
继续回到 oracle support / safe-useful target；
继续设计 C0/T2/C4/E2 新 variant；
继续把 projection 当 official；
继续只做 selected-logit formula proxy；
继续按 dataset 调 prefilter。
```

---

# Part II. v9.2.69 总体目标

v9.2.69 的总体目标是：

$$
\boxed{
\text{把 PF5 + FrozenC3 + true branch-delta exact confirm materialize 成 system-legal controller。}
}
$$

本轮必须并行回答八个问题：

```text
Q1:
  v9.2.68 boundary 是否稳定复现？

Q2:
  projection 与 materialized path 的差距在哪里？
  是 candidate packing、branch forward、bridge lookup、还是 sync/allocation？

Q3:
  PF5 能否在真实执行中输出 candidate index / family bucket / event metadata，
  而不是只在 CSV 中给 candidate_rate？

Q4:
  candidate-only true branch forward 是否能保持 exact reference agreement？

Q5:
  materialized event-sparse true-delta 是否能把 step_ratio_q90 压到 <=1.50？

Q6:
  compact bridge lookup 是否能避免 P5/P6 CPU-heavy frontier 汇总进入 online path？

Q7:
  system-legal controller 是否能保持 reference precision / coverage / bad-event / null-rate / LCB-UCB？

Q8:
  system controller 一旦过线，是否能打开 LDO/LSO 与 official paired replay？
```

v9.2.69 的核心 stop-go 是：

$$
\boxed{
\text{materialized_system_path}=1
\land
StepRatio_{q90}\leq1.50
\land
Agreement_{\text{reference}}\geq0.90.
}
$$

---

# Part III. 核心假设

## H1：v9.2.68 的 projection optimism 可以转化为真实 materialized speedup

v9.2.68 projection：

$$
CandidateRate=0.103051,
$$

$$
ProjectedStepRatio=1.227012.
$$

H1 认为 actual event-sparse path 的 overhead 不会吃掉全部 sparsity gain。

H1 成立标准：

materialized path 达到：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05,
$$

$$
materialized\_system\_path=1,
$$

$$
projection\_used\_for\_pass=0.
$$

H1 失败标准：

materialized path 实测：

$$
StepRatio_{q90}>1.50
$$

or:

```text
materialized path 需要 fallback 到 full exact trace / projection / CPU summary
```

## H2：dominant cost 是 candidate branch logits forward，可以通过 candidate packing / branch batching 降低

P1 已显示：

$$
Ratio(S1\text{-}branch\_logits\_forward)=0.396711,
$$

且 branch logits/replay combined ratio 为：

$$
0.440701.
$$

H2 成立标准：

materialized candidate branch forward 后：

$$
Ratio(S1)\leq0.25
$$

or total:

$$
StepRatio_{q90}\leq1.50.
$$

H2 失败标准：

即使 candidate rate 只有 `0.103051`，branch forward ratio 仍 `>=0.35`，说明 candidate packing 没有真正减少 expensive work。

## H3：frozen C3 rule 可直接用于 online accept，不需要在线 frontier search

H3 已在 v9.2.68 基本成立，heldout agreement = `1.0`。v9.2.69 需要在 materialized path 中重新验证。

H3 成立标准：

$$
Agreement(A_{\text{materialized}}, A_{\text{frozen-ref}})\geq0.95
$$

and:

$$
Agreement(A_{\text{materialized}}, A_{\text{search-ref}})\geq0.90.
$$

H3 失败标准：

materialized feature layout 改变 accept set，导致 reference agreement `<0.90`。

## H4：PF5 不只是统计 prefilter，而是可执行 candidate selector

v9.2.68 证明 PF5 candidate rate / recall 好，但还没有证明它能作为 runtime candidate selector 落地。

H4 成立标准：

PF5 runtime selector 输出：

```text
candidate_indices
event_ids
family_ids
bucket_ids
candidate_rate
reference_accept_recall
```

并且：

$$
CandidateRate\leq0.25,
$$

$$
Recall_{\text{reference-accepted}}\geq0.95.
$$

H4 失败标准：

PF5 只能在 posthoc CSV 上计算，不能在 train-stream online path 中输出 candidate pack。

## H5：compact bridge lookup 能消除 CPU-heavy frontier summary

v9.2.68 后段 GPU sampling 出现 `0%, 326 MiB`，说明 frontier 汇总阶段可能 CPU-heavy。H5 认为 official online path 不应包含这种 summary search。

H5 成立标准：

online bridge path 只使用 frozen lookup tables：

```text
family_id -> safe_useful_lcb
family_id -> bad_ucb
family_id -> null_ucb
family_id -> support_density
bucket_id -> trim/backfill flag
bucket_id -> threshold
```

并且：

$$
T_{\text{bridge-lookup}}\leq0.10T_{\text{step}}.
$$

H5 失败标准：

bridge lookup / support aggregation 成为 new dominant cost，或仍依赖 CPU-heavy summary.

## H6：一旦 system controller 过线，LDO/LSO 与 paired replay 是真正科学门

H6 成立标准：

system controller pass 后，P7/P8/P9 可打开，并且在 leave-out / paired replay 中仍 task-safe。

H6 失败标准：

system controller 过 pooled heldout，但 LDO/LSO 或 paired replay fail。此时不能 dataset-specific tuning，只能修 dataset-agnostic support/family reliability 或 functional causal target。

---

# Part IV. Materialized system design

## 1. Frozen reference target

Reference target 固定为：

$$
A_{\text{ref}}=A_{\text{C3-T2PlusBackfill}}.
$$

Reference metrics：

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

Materialized system path 的目标不是重新搜索更好的 frontier，而是复现 $A_{\text{ref}}$：

$$
A_{\text{sys}}\approx A_{\text{ref}}.
$$

## 2. Runtime event-sparse pipeline

系统执行管线必须是：

```text
Stage 0:
  Build train-stream legal event table for current step.

Stage 1:
  PF5 runtime prefilter produces candidate indices.

Stage 2:
  Candidate pack:
    gather candidate rows
    gather candidate branch metadata
    gather family/bucket ids
    build contiguous candidate tensors

Stage 3:
  Candidate-only true branch forward:
    no full all-row branch replay
    no source gap proxy
    no formula proxy

Stage 4:
  Candidate-only true branch-delta tensor:
    compute exact logits / selected required logits according to frozen rule
    preserve agreement with TBD0 reference

Stage 5:
  Frozen bridge lookup:
    compact table lookup
    no online frontier search
    no CPU summary

Stage 6:
  Accept / abstain / reject
```

Formal accept:

$$
Accept_{\text{sys}}(e)
=
PF5(e)
\land
TrueDeltaConfirm(e)
\land
FrozenC3Bridge(e).
$$

## 3. Cost model

Let:

$$
r=\frac{|P_{\text{PF5}}|}{|E|}.
$$

v9.2.68 measured:

$$
r=0.103051.
$$

Approximate system cost:

$$
T_{\text{sys}}
=
T_{\text{base}}
+
T_{\text{prefilter}}
+
T_{\text{pack}}
+
rT_{\text{branch-forward}}
+
rT_{\text{delta}}
+
T_{\text{bridge-lookup}}
+
T_{\text{accept}}.
$$

Official pass:

$$
\frac{T_{\text{sys}}}{T_{\text{MLP}}}\leq1.50.
$$

Projection is not enough. v9.2.69 must record actual measured:

```text
cuda_event_time_ms
wallclock_time_ms
kernel_count
sync_count
allocation_count
peak_memory
candidate_pack_time
candidate_branch_forward_time
candidate_delta_time
bridge_lookup_time
accept_time
```

## 4. Materialization audit

Every candidate must explicitly record:

```text
materialized_system_path
projection_used
full_trace_projection_used
source_measured_gap_used
formula_proxy_used
cpu_offload_used
candidate_only_branch_forward_used
candidate_pack_materialized
frozen_bridge_lookup_used
online_frontier_search_used
```

Official pass requires:

```text
materialized_system_path = 1
projection_used = 0
full_trace_projection_used = 0
source_measured_gap_used = 0
formula_proxy_used = 0
cpu_offload_used = 0
online_frontier_search_used = 0
```

---

# Part V. Candidate designs

## 1. Materialization audit candidates

### MA0：v9.2.68 projection reference

Diagnostic only.

### MA1：MaterializedPF5SelectorSmoke

Implement PF5 as runtime selector, no true-delta yet. Measures candidate pack correctness.

### MA2：MaterializedPF5CandidatePack

PF5 + contiguous candidate tensors:

```text
candidate_event_id
candidate_row_index
candidate_family_id
candidate_bucket_id
candidate_branch_id
candidate_horizon
candidate_role
candidate_support_features
```

### MA3：Projection-vs-Materialized Delta Audit

Compare projected and actual candidate sets:

$$
Jaccard(P_{\text{projection}},P_{\text{materialized}}),
$$

$$
Recall(A_{\text{ref}}\mid P_{\text{materialized}}).
$$

## 2. Prefilter candidates

### PF0：No prefilter

Diagnostic baseline; all rows exact confirm.

### PF5：v9.2.68 LearnedMonotoneCheapPrefilter

Frozen PF5.

### PF5a：PF5 lookup-only implementation

Same logic, but all family/bucket quantities precomputed.

### PF5b：PF5 high-recall widened

Only diagnostic if PF5 materialized recall drops.

### PF5c：PF5 candidate-rate capped

Tie-breaker if materialized candidate rate exceeds expected.

Pass rule:

$$
Recall_{\text{ref}}\geq0.95,
$$

$$
CandidateRate\leq0.25.
$$

## 3. Candidate branch forward candidates

### BF0：Full-row TBD0 reference

All rows true branch forward. Diagnostic only.

### BF1：CandidatePackedBranchForward

Only candidate rows are fed through branch forward.

### BF2：CandidatePackedBranchForwardBatched

Pack candidates across events/families into a single batch tensor.

### BF3：CandidateFamilyGroupedForward

Group candidates by family/branch/horizon to reduce kernel launch overhead.

### BF4：CandidatePersistentBranchWorkspace

Preallocate candidate workspace; avoid per-step allocations.

### BF5：TritonCandidateGatherScatter

Fused gather/scatter for candidate pack and output writeback.

## 4. True branch-delta candidates

### TD0：TBD0 full exact reference

Diagnostic reference.

### TD1：CandidateOnlyTBD0

Run exact TBD0 only on packed candidates.

### TD2：CandidateOnlySelectedBridgeLogits

Compute only bridge-required logits, but must prove accept agreement:

$$
Agreement\geq0.90.
$$

### TD3：CandidateOnlyTwoPassBorderline

First pass selected bridge logits; second pass full logits for borderline candidates.

### TD4：FusedCandidateDeltaKernel

Fused candidate delta computation:

```text
candidate logits
control logits
delta vector
bridge partial score
```

### TD5：PersistentCandidateDeltaWorkspace

Same as TD1/TD4 but with preallocated workspace.

## 5. Bridge compute candidates

### BR0：Reference search bridge

Diagnostic only; not online.

### BR1：FrozenC3Lookup

Lookup table implementation of C3-T2PlusBackfill.

### BR2：FrozenC3QuantizedBucketLookup

Quantized risk/null/support buckets.

### BR3：FusedBridgeScoreKernel

Fused bridge score for candidate rows.

### BR4：BridgeLookupNoCPU

Audit that no CPU summary is used in online path.

## 6. System candidates

### SYS0：FullExactFrozenC3

All rows exact confirm + frozen C3. Diagnostic; expected too expensive.

### SYS1：PF5 + CandidateOnlyTBD0 + FrozenC3Lookup

Primary materialization candidate.

### SYS2：PF5 + CandidateFamilyGroupedForward + FrozenC3Lookup

Reduce branch forward overhead.

### SYS3：PF5 + CandidateOnlySelectedBridgeLogits + FrozenC3Lookup

Lower logit compute, must keep agreement.

### SYS4：PF5 + CandidateOnlyTwoPassBorderline + FrozenC3Lookup

Two-pass balance.

### SYS5：PF5 + FusedCandidateDeltaKernel + FusedBridgeScoreKernel

Target final fused path.

### SYS6：PF5 + PersistentCandidateDeltaWorkspace + FrozenC3Lookup

Target allocation/sync overhead.

### SYS7：HybridBestMaterializedSystem

Best combination from PF / BF / TD / BR.

---

# Part VI. 实验阶段

## P0：v9.2.68 boundary reproduction

### 目标

确认 v9.2.68 boundary 稳定，尤其是 reference frozen + PF5 pass + projection not materialized。

### 必须记录

```text
route
source_route_v9267
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
best_prefilter_id
candidate_rate
reference_accept_recall
best_exact_candidate_id
event_sparse_exact_diagnostic_pass
materialized_system_path
system_legal_controller_pass
primary_blocker
fake_proxy_count
cpu_offload_used
```

### 判断标准

P0 pass：

```text
route = R17-ReferenceFeasibleButComputeFail
reference_controller_id = C3-T2PlusBackfill
frozen_controller_pass = 1
prefilter_pass = 1
system_legal_controller_pass = 0
primary_blocker = true_delta_system_path_not_materialized
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9268_gate_ladder.svg
p0_reference_projection_materialization_boundary.svg
p0_step_ratio_reference_projection.svg
```

---

## P1：projection-to-materialization gap audit

### 目标

把 v9.2.68 的 projection 拆开，确认哪些信息目前只在 CSV/posthoc projection 中存在，哪些可以在线 materialize。

### 必须记录

```text
audit_id
component
projection_available
materialized_available
required_for_official
gap_type
implementation_file
runtime_input
runtime_output
uses_posthoc
uses_projection
uses_cpu_summary
materialization_blocker
```

Components：

```text
PF5 candidate selector
candidate indices
candidate family ids
candidate branch ids
candidate tensors
true branch logits
true branch-delta
frozen C3 lookup
bridge score
accept decision
timing measurement
```

### 判断标准

P1 pass：

```text
all official-required components mapped
no unknown materialization gap
implementation priority assigned for every missing component
```

### 可视化

```text
p1_projection_materialization_matrix.svg
p1_component_gap_sankey.svg
p1_official_required_component_ladder.svg
```

---

## P2：runtime PF5 selector and candidate pack

### 目标

把 PF5 从统计 prefilter 变成 runtime candidate selector。

### 必须记录

```text
prefilter_id
implementation_id
candidate_rate
reference_accept_recall
candidate_count
event_count
candidate_pack_time_ms
candidate_pack_memory_MB
candidate_indices_materialized
candidate_family_ids_materialized
candidate_branch_ids_materialized
candidate_tensor_contiguous
projection_used
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

PF5 materialization pass：

$$
CandidateRate\leq0.25,
$$

$$
Recall_{\text{reference-accepted}}\geq0.95,
$$

```text
candidate_indices_materialized = 1
candidate_tensor_contiguous = 1
projection_used = 0
posthoc_used_at_commit = 0
dataset_name_used = 0
```

### 可视化

```text
p2_candidate_rate_recall_curve.svg
p2_candidate_pack_time_breakdown.svg
p2_candidate_distribution_by_stratum_family.svg
```

---

## P3：candidate-only true branch forward

### 目标

只对 PF5 candidates 真实执行 branch forward，验证实际 branch logits cost 是否下降。

### 必须记录

```text
branch_forward_id
prefilter_id
candidate_rate
candidate_count
full_row_count
uses_candidate_only_forward
uses_full_row_forward
branch_forward_time_ms_mean
branch_forward_time_ms_q90
branch_forward_ratio_vs_full
kernel_count
sync_count
allocation_count
read_MB
write_MB
largest_temp_tensor_MB
logit_max_abs_diff_vs_full_reference
logit_rel_err
```

### 判断标准

Candidate branch forward pass：

```text
uses_candidate_only_forward = 1
uses_full_row_forward = 0
```

and:

$$
LogitMaxAbsDiff\leq10^{-6}
$$

or accept-level equivalent if exact ordering differs only by numerically irrelevant epsilon.

Cost diagnostic pass：

$$
T_{\text{candidate-forward}}\leq0.35T_{\text{full-forward}}.
$$

### 可视化

```text
p3_candidate_forward_cost.svg
p3_full_vs_candidate_forward_ratio.svg
p3_candidate_pack_kernel_count.svg
p3_logit_error_histogram.svg
```

---

## P4：materialized event-sparse true-delta exact confirmation

### 目标

真正实现 `PF5 + candidate-only true branch-delta + FrozenC3`，而不是 projection。

### 必须记录

```text
exact_candidate_id
prefilter_id
branch_forward_id
true_delta_id
candidate_rate
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
materialized_system_path
projection_used
full_trace_projection_used
agreement_reference_accept
AUC_safe_good
AUC_bridge_accept
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
dominant_subphase
```

### 判断标准

Materialized event-sparse exact pass：

```text
materialized_system_path = 1
projection_used = 0
full_trace_projection_used = 0
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
```

and:

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
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Diagnostic pass：

$$
Agreement_{\text{reference-accept}}\geq0.90
$$

and:

$$
StepRatio_{q90}\leq2.00.
$$

### 可视化

```text
p4_materialized_event_sparse_frontier.svg
p4_projection_vs_materialized_step_ratio.svg
p4_reference_agreement_confusion.svg
p4_subphase_cost_waterfall.svg
```

---

## P5：fused / compact bridge materialized compute

### 目标

把 frozen bridge lookup 和 candidate delta scoring 合并为低开销 path，避免 CPU-heavy frontier summary。

### 必须记录

```text
bridge_system_id
exact_candidate_id
bridge_compute_id
uses_fused_kernel
uses_compact_lookup
uses_quantized_bucket
uses_cpu_summary
online_frontier_search_used
materialized_system_path
agreement_reference_accept
AUC_bridge_accept
precision
coverage
bad_event
null_rate
precision_lcb
bad_event_ucb
bridge_lookup_time_ms
bridge_score_time_ms
step_ratio_q90
memory_ratio
kernel_count
sync_count
read_MB
write_MB
```

### 判断标准

Fused / compact bridge pass：

```text
materialized_system_path = 1
uses_cpu_summary = 0
online_frontier_search_used = 0
```

and:

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
p5_bridge_lookup_time_before_after.svg
p5_cpu_summary_elimination.svg
p5_kernel_count_reduction.svg
```

---

## P6：system-legal exact-signal controller v2

### 目标

只有 P2/P4/P5 的 materialized path 过 gate，P6 才能把 controller 标为 official eligible。

### 必须记录

```text
controller_id
prefilter_id
branch_forward_id
exact_candidate_id
bridge_system_id
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
candidate_rate
step_ratio_q90
memory_ratio
materialized_system_path
projection_used
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
official_eligible
```

### 判断标准

System-legal controller pass：

```text
official_eligible = 1
materialized_system_path = 1
projection_used = 0
```

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
p6_system_controller_decision_frontier.svg
p6_reference_vs_system_accept_overlap.svg
p6_system_cost_vs_quality.svg
p6_family_strata_balance.svg
```

---

## P7：leave-dataset-out / leave-stratum-out

### 目标

证明 materialized system controller 不是 pooled calibration artifact，也不是 dataset-specific route。

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
bridge_system_id
threshold
candidate_rate
reference_accept_recall
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
p7_dataset_tuning_audit.svg
p7_leaveout_failure_modes.svg
```

---

## P8：official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
controller_id = best P7 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledTrueBranchDelta, ShuffledPF5Prefilter,
           ShuffledCandidatePack, ShuffledExactConfirm,
           ShuffledFrozenBridge, ShuffledSupportStat,
           ShuffledControlGain, ShuffledCandidateGate,
           ShuffledBranchRatio, ShuffledSignalChannel,
           FunctionalChannelShuffled, TailMaskShuffled,
           RoleScoreShuffled, DatasetRouteShuffled,
           EventRouteShuffled, InvertedRoleMask
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
candidate_rate
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

Shuffle controls fail：

```text
ShuffledTrueBranchDelta = fail
ShuffledPF5Prefilter = fail
ShuffledCandidatePack = fail
ShuffledExactConfirm = fail
ShuffledFrozenBridge = fail
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

## P9：short-run scout

### 目标

如果 P8 paired replay pass，验证 local causal advantage 能否进入连续训练。

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
candidate_rate
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

## P10：full run / robustness / strong baseline

### 目标

只有 P9 pass 后打开。验证 functional advantage 不是 local replay artifact。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
controls = AdamWOnly, AdamWParallel, bestLR, StrongLRGrid, QuadraticFeatureMLP,
           NoOp, Random, ShuffledTrueBranchDelta, ShuffledPF5, ShuffledFrozenBridge
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
candidate_rate
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

Full functional pass：

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

Strong baseline pass：

```text
Functional not explained by QuadraticFeatureMLP
Functional not explained by LR grid
Functional not explained by shuffled controller
```

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9269.csv
p0_v9268_boundary_reproduction.csv
p1_projection_to_materialization_gap_audit.csv
p2_runtime_pf5_selector_candidate_pack.csv
p3_candidate_only_true_branch_forward.csv
p4_materialized_event_sparse_true_delta_exact_confirmation.csv
p5_fused_compact_bridge_materialized_compute.csv
p6_system_legal_exact_signal_controller_v2.csv
p7_leave_dataset_and_stratum_out.csv
p8_official_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_run_robustness_strong_baseline.csv
materialization_gap_trace_v9269.csv
pf5_candidate_pack_trace_v9269.csv
candidate_branch_forward_trace_v9269.csv
materialized_true_delta_trace_v9269.csv
fused_bridge_materialized_trace_v9269.csv
system_controller_trace_v9269.csv
leaveout_trace_v9269.csv
paired_replay_branch_trace_v9269.csv
short_run_trace_v9269.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9268_boundary_unstable
F3_dataset_tuning_detected
F4_reference_controller_not_reproducible
F5_projection_materialization_gap_unmapped
F6_pf5_runtime_selector_fail
F7_pf5_recall_fail
F8_candidate_rate_too_high
F9_candidate_pack_not_materialized
F10_candidate_branch_forward_too_expensive
F11_candidate_branch_forward_numerical_mismatch
F12_materialized_true_delta_agreement_fail
F13_materialized_true_delta_system_fail
F14_materialized_true_delta_memory_fail
F15_fused_bridge_uses_cpu_summary
F16_fused_bridge_signal_lost
F17_system_controller_precision_fail
F18_system_controller_coverage_fail
F19_system_controller_bad_event_fail
F20_system_controller_null_rate_fail
F21_system_controller_lcb_ucb_fail
F22_system_controller_projection_used
F23_true_delta_compute_still_expensive
F24_true_delta_signal_lost
F25_leave_dataset_out_fail
F26_leave_stratum_out_fail
F27_paired_replay_control_equivalent
F28_shuffle_control_pass
F29_functional_lr_equivalent
F30_short_run_task_drop
F31_full_run_no_macro_hard_stratum_gain
F32_strong_baseline_explains_gain
F33_robustness_fail
F34_external_not_ready
F35_fake_or_proxy_violation
F36_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.68 boundary reproduced.

R2-MaterializationGapMapped:
  projection-to-materialization gap is fully attributed.

R3-PF5RuntimeSelectorPass:
  PF5 materialized selector reaches candidate_rate / recall gate.

R4-CandidateBranchForwardPass:
  candidate-only branch forward is materialized and numerically correct.

R5-MaterializedEventSparseExactPass:
  event-sparse true-delta exact confirmation passes agreement and system gate.

R6-FusedBridgeMaterializedPass:
  fused / compact bridge compute passes without CPU summary or projection.

R7-SystemLegalExactSignalControllerPass:
  system controller passes decision + compute + materialization gates.

R8-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R9-LeaveStratumOutPass:
  controller generalizes across held-out strata.

R10-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R11-ShortRunFunctionalPass:
  short-run task-safe mechanism gain.

R12-FullFunctionalPass:
  full run task / geometry / system / control gates pass.

R13-ReferenceControllerUnstable:
  v9.2.67/68 reference controller cannot be reproduced or frozen.

R14-PF5NotMaterializable:
  PF5 works as statistics but not as runtime selector.

R15-PrefilterRecallCostTradeoffFail:
  candidate rate cannot be reduced while keeping recall.

R16-MaterializedPathStillExpensive:
  candidate sparsity materializes but step_ratio_q90 remains >1.50.

R17-FusedBridgeSignalLost:
  fused / compact bridge path loses agreement or decision quality.

R18-ReferenceFeasibleButComputeFail:
  decision geometry viable but true-delta system implementation remains blocker.

R19-ComputePassButLeaveoutFail:
  system controller overfits pooled calibration.

R20-ComputePassButPairedReplayFail:
  controller is system-legal but not causally superior to controls.

R21-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9268_boundary_pass
dataset_tuning_detected
reference_controller_id
reference_controller_reproduced
reference_precision
reference_coverage
reference_bad_event
reference_null_rate
reference_precision_lcb
reference_bad_event_ucb
materialization_gap_mapped
best_prefilter_id
pf5_runtime_selector_pass
candidate_rate
reference_accept_recall
candidate_pack_materialized
best_branch_forward_id
candidate_branch_forward_pass
candidate_forward_time_ratio
branch_logit_error_max
best_exact_candidate_id
materialized_event_sparse_exact_pass
materialized_system_path
projection_used
exact_agreement
exact_step_ratio_q90
exact_memory_ratio
best_fused_bridge_id
fused_bridge_materialized_pass
fused_bridge_step_ratio_q90
fused_bridge_memory_ratio
uses_cpu_summary
online_frontier_search_used
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
success_v9269_strict_purekan_functional
success_v9269_full_functional
success_v9269_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 projection-to-materialization gap audit
  P2 runtime PF5 selector / candidate pack
  P3 candidate-only branch forward
  P4 materialized event-sparse exact confirmation
  P5 fused / compact bridge materialized compute

Batch 2:
  P6 system-legal exact-signal controller
  P7 leave-dataset-out / leave-stratum-out
  P8 paired replay scout

Batch 3:
  official P8 paired replay
  P9 short-run if P8 passes

Batch 4:
  P10 full run / robustness / strong baseline only if P9 passes
```

Gate rule：

```text
P2/P3/P4/P5 can run in parallel after P1 component mapping.
P6 cannot pass unless:
  P0 pass
  P2 materialized selector pass
  P4 or P5 materialized compute pass
  no projection used
P7/P8 diagnostic rows may be measured before all gates finish,
but official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  reference controller reproduced
  frozen controller pass
  PF5 runtime selector pass
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
v9.2.68 boundary reproduced
projection-to-materialization gap mapped
PF5 materialized selector measured
candidate branch forward measured
materialized event-sparse true-delta measured
fused / compact bridge measured
system controller measured
no fake/proxy/offload/loss/teacher violation
```

## Materialization success

```text
Minimum diagnostic success
+
materialized_system_path = 1
+
projection_used = 0
+
candidate-only true branch forward materialized
+
frozen bridge lookup materialized
```

## System success

```text
Materialization success
+
agreement_reference_accept >= 0.90
+
step_ratio_q90 <= 1.50
+
memory_ratio <= 1.05
+
decision gates pass
```

## Local functional success

```text
System success
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
1. v9.2.68 boundary cannot be reproduced；
2. C3-T2PlusBackfill frozen controller becomes unstable；
3. projection/materialization gap cannot be mapped；
4. PF5 cannot materialize runtime candidate selector；
5. PF5 materialized candidate rate >0.25；
6. PF5 materialized reference recall <0.95；
7. candidate-only branch forward is slower than full forward after packing overhead；
8. materialized exact confirmation loses reference agreement；
9. materialized exact confirmation still step_ratio_q90 >1.50；
10. fused bridge compute uses CPU summary / online search；
11. fused bridge compute loses signal；
12. system controller cannot meet precision / coverage / bad-event / null-rate；
13. precision LCB below 0.75；
14. bad-event UCB above 0.05；
15. leave-dataset-out fails；
16. leave-stratum-out fails；
17. paired replay remains control-equivalent；
18. shuffle controls pass；
19. short-run task drops；
20. full run gives no macro / hard-stratum / geometry gain；
21. functional breaks system gate；
22. gains are explained by QuadraticFeatureMLP；
23. any teacher/loss/fake/proxy/offload/projection-as-pass violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：materialized system controller + LDO/LSO + paired replay pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full run and robustness。

## Case B：materialized system path pass but LDO/LSO fail

必须声明：

```text
system implementation is viable, but controller is not dataset-agnostic / stratum-agnostic enough.
```

不能按 dataset 调 threshold。下一步修 support/family reliability。

## Case C：materialized path remains expensive

必须声明：

```text
projection was optimistic; actual candidate-sparse true-delta still exceeds system envelope.
```

下一步继续 branch-forward kernel / candidate packing / fused delta kernel，不调 reference frontier。

## Case D：PF5 not materializable

必须声明：

```text
cheap prefilter was a statistical diagnostic, not an executable online selector.
```

下一步实现 executable cheap prefilter or redesign runtime candidate generator。

## Case E：fused compute loses signal

必须声明：

```text
low-cost implementation changes accept decisions and cannot replace exact true-delta.
```

下一步修 exactness / two-pass borderline confirmation。

## Case F：compute pass but paired replay fail

必须声明：

```text
controller is system-legal but not causally superior to matched controls.
```

下一步回到 functional event/value target，而不是继续 kernelization。

---

# Part XII. 最终建议

v9.2.69 的一句话策略是：

$$
\boxed{
\text{不要再用 projected cost 说服自己；把 PF5 + FrozenC3 + true-delta exact confirm 真正 materialize，并重测真实 step ratio。}
}
$$

当前最关键的问题不是：

```text
oracle 是否存在；
reference frontier 是否 deployable；
C0/T2/C4/E2 是否能 bridge；
safe-useful target 是否可行；
PF5 candidate rate 是否看起来好；
projected step ratio 是否 <=1.50；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. PF5 能否成为 runtime selector？
2. candidate pack 是否真实连续化？
3. candidate-only branch forward 是否真实减少 S1 cost？
4. true branch-delta exact confirmation 是否保持 agreement？
5. fused / compact bridge 是否完全摆脱 CPU summary 和 online search？
6. materialized step_ratio_q90 是否真的 <=1.50？
7. system controller 是否保持 reference precision / coverage / bad-event / null-rate？
8. LDO/LSO 是否通过？
9. official paired replay 是否打过 AdamWParallel / bestLR？
```

v9.2.69 的结果将给出清晰分叉：

```text
if materialized system path passes:
  open system-legal controller, LDO/LSO, paired replay.

if materialized system path still too expensive:
  projection was optimistic; continue kernelization/candidate packing/fused delta.

if PF5 cannot materialize:
  build executable candidate generator.

if compute passes but paired replay fails:
  system is legal, but functional causality not enough.

if LDO/LSO fails:
  repair dataset-agnostic support/family reliability, not dataset-specific tuning.
```
