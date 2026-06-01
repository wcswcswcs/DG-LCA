# DG-KAN v9.2.78 Native Bridge-Accept Numerical Closure 与 Bucketed Runtime Official Promotion 完整实验计划

> 本计划基于 v9.2.77 `Quantile-Tail BasisNorm 与 Static-Bucket Bridge Closure` 的真实执行结果制定。  
> v9.2.77 的 terminal route 是：
>
> ```text
> route = R17-BasisNormRuntimeStillInsufficient
> base_candidate = LQ-t2-h256
> success_v9277_strict_purekan_functional = False
> success_v9277_full_functional = False
> success_v9277_external_ready = False
> ```
>
> v9.2.77 的主线结果：
>
> ```text
> P0:
>   source route = R15-BasisNormDominantUnfixed
>   payload_binding_contract_pass = 1
>   candidate/update payload missing count = 0
>   system legal controller pass = 0
>
> P1 quantile-tail attribution:
>   quantile_tail_internal_attribution_pass = 1
>   quantile_tail_unknown_fraction = 0.001664
>   dominant = threshold_compute_time_ms
>   threshold_compute_time_ms = 0.220606
>   quantile_tail_total_time_ms = 0.578957
>
> P2 quantile-tail runtime:
>   quantile_tail_runtime_pass = 1
>   strategy = kthvalue_exact_tail_threshold
>   quantile_tail_time_before = 0.578957 ms
>   quantile_tail_time_after = 0.194465 ms
>   reduction = 0.664111
>   audit_agreement = 1.0
>   logits_error = 0.0
>   delta_error = 0.0
>   source gap / formula proxy / projection = 0
>
> P3 basis_norm integration:
>   basis_norm_time_before = 0.535139 ms
>   basis_norm_time_after = 0.348024 ms
>   reduction = 0.349657 < 0.50
>   basis_norm_bucketed = 0
>   basis_norm_fused_kernel_used = 0
>   basis_norm_runtime_pass = 0
>
> P4 static bucket / persistent workspace:
>   runtime_bucket_used = 1
>   persistent_workspace_used = 1
>   allocation_count_before = 2493
>   allocation_count_after = 1
>   basis_norm_bucketed = 0
>   W2_delta_bucketed = 0
>   bridge_score_bucketed = 0
>   kernel_count_reduction = 0.0
>   sync_count_reduction = 0.0
>   static_bucket_workspace_pass = 0
>
> P5 single-pass bridge:
>   quantile_tail_inside_kernel = 0
>   bridge_score_inside_kernel = 0
>   basis_norm_delta_bridge_single_pass = 0
>
> P6/P7:
>   materialized_runtime_path = 0
>   diagnostic_derived_from_measured_components = 1
>   official_eligible = 0
>   system_legal_controller_pass = 0
>   controller_step_ratio_q90 = 2.713296 > 1.50
> ```
>
> v9.2.77 的追加结果：
>
> ```text
> SP2 TorchCompile single-pass:
>   kth-tail + basis_norm + W2 delta + probe logits + bridge score + accept bit
>   数值正确
>   局部 q90 约降低 19.06%
>   但不是 static-bucket / workspace integrated runtime
>
> SP3 static-bucket persistent compiled single-pass:
>   basis_norm_bucketed = 1
>   W2_delta_bucketed = 1
>   bridge_score_bucketed = 1
>   persistent_workspace_used = 1
>   workspace memory = 0.256875 MB
>   eager q90 = 1.925641 ms
>   workspace q90 = 1.329119 ms
>   q90 reduction = 30.98%
>   logits/delta/bridge/tail/accept 数值对照通过
>   但 kernel_count 3750 -> 3750
>   sync_count 1250 -> 1250
>   avg_candidates_per_kernel = 0.6648
>   static_bucket_workspace_pass = 0
>
> SP4 / P15 CUDA graph:
>   CUDAGraph static-bucket replay failed
>   precomputed-tail delta/bridge isolation also failed capture
>   conclusion: PyTorch CUDAGraph / torch.compile 外壳不能承载 official bucketed single-pass runtime
>
> P16 native CUDA bucket kernel:
>   native CUDA bucket kernel 已执行
>   basis_norm_bucketed = 1
>   W2_delta_bucketed = 1
>   bridge_score_inside_kernel = 1
>   accept_bit_inside_kernel = 1
>   最佳 q90 reduction = 45.26%
>   bridge_score_error_max ≈ 1.19e-07
>   accept_disagreement_count = 10
>   deterministic / fixed-order float reduce 未能清零 disagreement
> ```
>
> v9.2.78 的核心判断是：
>
> $$
> \boxed{
> \text{当前 blocker 已从“没有 native bucket kernel”推进到“native bridge accept 的边界数值稳定性未闭合”。}
> }
> $$
>
> v9.2.77 证明了三个重要事实：
>
> ```text
> 1. quantile-tail 子项可以被 kthvalue exact-tail 大幅降低；
> 2. static bucket + persistent workspace + single-pass 图有真实局部收益；
> 3. native CUDA bucket kernel 方向是有效的，但 accept bit 在 median 边界附近发生数值翻转。
> ```
>
> 因此 v9.2.78 不应继续做泛泛的 torch.compile / CUDAGraph 尝试，也不应回到 controller threshold 调参。  
> 本轮必须围绕一个更本质的问题设计：
>
> $$
> \boxed{
> \text{如何让 native bucket kernel 的 bridge score / accept decision 与 frozen reference 形成稳定、一致、可 official 的 decision contract？}
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

v9.2.78 新增硬约束：

```text
不能通过放宽 local tolerance 把 accept_disagreement 写成 pass；
不能把 accept bit 移回 Python 再声明 bridge_score_inside_kernel pass；
不能使用 source-measured gap；
不能使用 formula proxy；
不能把 precomputed-tail isolation row 写成 full single-pass official；
不能按 dataset 选择不同 accept threshold / bucket route / tie policy；
不能只用 torch.compile 或 CUDAGraph 外壳反复尝试；
不能只报告 q90 reduction 而忽略 accept disagreement；
不能只在 P16 局部修 tolerance，必须重跑 decision/support/system gates。
```

允许使用：

```text
native CUDA / Triton bucket kernel
deterministic bridge-score reduction
stable rank / tie policy
borderline exact fallback
calibration-split frozen margin rule
compact bridge lookup inside kernel
post-step audit subset
dataset-agnostic family / horizon / bucket diagnostics
```

---

# Part I. 对 v9.2.77 的独立判断

## 1. v9.2.77 没有达到最终目标

v9.2.77 没有达到 strict PureKAN functional success。直接原因是：

```text
system_legal_controller_pass = 0
official_eligible = 0
controller_step_ratio_q90 = 2.713296 > 1.50
LDO / LSO = not_run
official paired replay = not_run
short-run / full-run = not_run
```

更深一层，不能 official 的原因不是 quantile-tail 局部优化失败，而是 integrated runtime 没形成 official path：

```text
P2 quantile-tail runtime pass = 1
P3 basis_norm runtime pass = 0
P4 static bucket workspace pass = 0
P5 single-pass bridge pass = 0
P6 integrated runtime pass = 0
P7 system controller pass = 0
```

追加 SP3 和 P16 说明方向没有错，但仍不能转正。SP3 只是 workspace-bound single-pass diagnostic，kernel/sync 没降；P16 native CUDA 已有真实 q90 收益，但 accept bit disagreement 未清零。一个 `accept_disagreement_count = 10` 的 native kernel，不能写成 official controller。

## 2. v9.2.77 的真实进展

v9.2.77 的进展不小，而且比 v9.2.76 更接近 system closure。

第一，quantile-tail 子项被真正打穿。v9.2.76 的 no-clone/logsumexp/top2 只把 basis_norm time 从 `0.797227` 降到 `0.581002`，降幅 `27.12%`；v9.2.77 的 kthvalue exact-tail 把 quantile-tail time 从 `0.578957` 降到 `0.194465`，降幅 `66.41%`，并保持 audit agreement = `1.0`。

第二，SP3 已经不是纯 accounting。它把 single-pass 输出真实写入 static bucket persistent workspace，并且 basis_norm/W2_delta/bridge_score 都 bucketed 到 workspace path。它的 q90 从 `1.925641 ms` 降到 `1.329119 ms`，这是实际局部 runtime 改善，不是 report-only projection。

第三，P16 native CUDA bucket kernel 已经执行，并取得了真实局部 runtime 改善。最佳 q90 reduction 达到 `45.26%`，并且已经把 basis_norm、W2_delta、bridge_score、accept_bit 放进 native kernel。这个结果把困难从“没有 native implementation”推进到“native decision numerical closure”。

## 3. v9.2.77 的真实失败

v9.2.77 的失败不应再写成“basis_norm 没优化”。更准确是：

$$
\boxed{
\text{native bucket runtime 已接近，但 bridge accept decision 在 median 边界附近不稳定。}
}
$$

P16 中 bridge_score_error_max 约为 $1.19\times10^{-7}$，这本身并不大；但 accept rule 是类似 `score > median` 的边界判定，少量 row 的 score 位于 median 边界附近，因此极小浮点差异就足以翻转 accept bit。fixed-order double reduce 和 fixed-order float reduce 都不能清零 disagreement，说明这不是简单 atomic 顺序问题，而是 decision contract 本身对边界数值过敏。

这类失败和之前的失败不同。之前我们主要缺 implementation；现在已经有 implementation，但 reference decision contract 对浮点边界太脆弱。下一步不能靠“调小误差”赌过去，而要明确 accept rule 的稳定合同。

## 4. 当前 blocker 的本质

当前 blocker 是：

$$
\boxed{
\text{reference accept rule is numerically fragile under native kernel realization.}
}
$$

更具体地说：

```text
1. quantile-tail exact runtime 已局部过；
2. static bucket workspace 已有局部可行 path；
3. native CUDA bucket kernel 已有局部 q90 reduction；
4. logits / delta / tail / bridge score 误差很小；
5. accept bit 仍有 disagreement；
6. disagreement 集中在 median-threshold 边界；
7. 所以 official blocker 是 native accept equivalence，不是 controller quality。
```

这也解释为什么不能按 dataset 调参。问题不是 MNIST / Fashion / KMNIST 某个数据集坏了，而是任何数据集都可能在 bridge score median 附近出现边界 row。解决方案必须是 dataset-agnostic 的 accept contract。

## 5. 是否在正确道路上

是，但路线必须切换到 “decision-contract + native-kernel co-design”。正确路线是：

```text
quantile-tail exact runtime pass
→ native bucket kernel pass
→ bridge score inside kernel
→ accept-bit numerical autopsy
→ stable accept contract
→ native bucket kernel v2
→ integrated system controller
→ LDO / LSO
→ paired replay
```

错误路线是：

```text
继续调 C3-T2PlusBackfill threshold；
继续调 dataset-specific threshold；
继续做 torch.compile / CUDAGraph 外壳；
继续只报 q90 reduction；
继续用 tolerance 放过 accept disagreement；
继续把 accept bit 移回 Python；
继续把 source gap / formula proxy 当 true-delta。
```

---

# Part II. v9.2.78 总体目标

v9.2.78 的总体目标是：

$$
\boxed{
\text{在保持 native bucket kernel runtime reduction 的前提下，关闭 bridge accept 数值一致性，并形成 official-eligible system controller。}
}
$$

强目标：

$$
StepRatio_{q90}\leq1.50,
$$

$$
AcceptDisagreement=0,
$$

$$
OfficialEligible=1.
$$

最低有效推进目标：

```text
native bridge accept disagreement autopsy pass = 1
stable accept contract implemented = 1
native bucket kernel v2 materialized = 1
accept disagreement reduced from 10 to 0 or to formally gated borderline fallback
system step ratio <= 2.00
```

本轮必须回答十一个问题：

```text
Q1:
  v9.2.77 boundary 是否稳定复现？

Q2:
  P16 的 10 个 accept disagreement 到底来自哪里？
  是 score error、median threshold error、rank tie、tail threshold、W2 delta、probe logits，还是 reduction ordering？

Q3:
  disagreement row 的 bridge score margin 与 median threshold 的距离是多少？
  是否全部处在 tiny margin band 内？

Q4:
  reference accept rule 是否定义了稳定 tie policy？
  若没有，是否应该冻结一个 deterministic rank/tie policy？

Q5:
  是否可以让 native kernel bit-exact 复刻 reference？
  如果不能，是否可用 borderline exact fallback 保持 official correctness？

Q6:
  borderline fallback 的比例是多少？
  fallback 后 step ratio 是否仍可 <=1.50？

Q7:
  stable accept contract 是否会改变 decision metrics？
  precision / coverage / bad-event / null-rate / LCB-UCB 是否仍过？

Q8:
  native bucket kernel v2 是否真正降低 kernel/sync/avg candidates per kernel，而不只是 q90 局部降低？

Q9:
  如果重新定义 accept rule，是否必须重新跑 calibration / heldout / support gates？

Q10:
  system controller pass 后，LDO/LSO 是否通过？

Q11:
  official paired replay 是否能打过 AdamWParallel / bestLR？
```

---

# Part III. 核心假设

## H1：P16 accept disagreement 是边界决策脆弱性，而不是 true-delta 数值崩坏

H1 成立标准：

```text
accept_disagreement_count > 0
bridge_score_error_max <= 1e-6
logits_error_max <= 5e-5
delta_error_max <= 1e-8
tail_disagreement_count = 0 or explainable
all disagreement rows have |score_ref - threshold_ref| <= eps_boundary
```

其中：

$$
eps_{\text{boundary}} = 10^{-6}
$$

作为初始诊断阈值，实际可根据 fp32/fp64 error distribution 预注册调整，但不能用它直接放宽 official pass。

H1 失败标准：

```text
disagreement rows have large score margin；
or logits/delta/tail mismatch；
or bridge score component mismatch dominates；
or native kernel uses wrong feature / wrong row binding。
```

若 H1 失败，下一步应修 kernel correctness，不应改 accept contract。

## H2：stable rank/tie accept contract 可以消除 native/reference disagreement

当前 median accept 过于依赖浮点阈值比较。H2 认为可以把 accept rule 改写为 deterministic rank/tie contract：

$$
Accept(e)=1
\iff
(rank(score(e), tie\_key(e)) \leq K)
$$

其中 tie key 必须 dataset-agnostic，例如：

```text
global_event_id
candidate_event_id
family_id
bucket_id
horizon
```

H2 成立标准：

```text
stable_tie_policy_defined = 1
same stable policy used by reference and native kernel = 1
accept_disagreement_count = 0
agreement_reference_accept >= 0.99
precision/coverage/bad/null gates pass
```

H2 失败标准：

rank/tie policy 仍无法复现 reference，或使 decision metrics drift below gates。

## H3：borderline exact fallback can preserve correctness at low cost

若完全 bit-exact native accept 很难，H3 认为可使用 two-pass borderline fallback：

```text
Pass 1:
  native fast bridge score and margin band

Pass 2:
  exact reference-compatible computation only for borderline rows
```

Borderline set：

$$
B = \{e: |score(e)-threshold| \leq \epsilon_{\text{border}}\}.
$$

H3 成立标准：

```text
borderline_rate <= 0.05
fallback_exact_agreement = 1.0
final_accept_disagreement_count = 0
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

Diagnostic lower bar：

```text
borderline_rate <= 0.10
step_ratio_q90 <= 2.00
```

H3 失败标准：

borderline rate 太高，fallback 把 system cost 拉回 >1.50，或 fallback 仍不能消除 disagreement。

## H4：native bucket kernel v2 能保持 v9.2.77 的 q90 reduction

H4 成立标准：

```text
native_cuda_bucket_kernel_used = 1
basis_norm_bucketed = 1
W2_delta_bucketed = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
kernel_count_reduction >= 0.50 or avg_candidates_per_kernel_after >= 8
sync_count_reduction >= 0.50
allocation_count_after <= 0.10 * allocation_count_before
q90_reduction >= 0.40
```

H4 失败标准：

correctness 修好后 q90 reduction 消失，或 kernel/sync 仍无下降，导致 system step ratio 仍 >1.50。

## H5：如果 stable accept requires rule change, the whole controller gate must rerun

H5 成立标准：

```text
accept_rule_changed = 1
calibration_split_rerun = 1
heldout_split_rerun = 1
precision/coverage/bad/null/LCB-UCB rerun = 1
support balance rerun = 1
leave-out not opened until P6 pass
```

H5 失败标准：

只在 P16 局部改 tie/tolerance 就宣布 official pass。

## H6：若 native stable accept pass 但 paired replay fail，blocker moves back to functional value target

H6 成立标准：

system controller legal and compute pass，但 official paired replay 不 beat AdamWParallel / bestLR。  
H6 失败标准：

system controller 还没过，就讨论 functional event/value target。当前还没到这一步。

---

# Part IV. Native accept contract

## 1. Reference bridge score decomposition

每个 candidate event $e$ 必须记录：

```text
event_id
candidate_id
family_id
bucket_id
horizon
score_ref
score_native
score_abs_err
score_rel_err
threshold_ref
threshold_native
rank_ref
rank_native
tie_key_ref
tie_key_native
accept_ref
accept_native
accept_disagreement
```

并分解 bridge score：

$$
score(e)
=
f(
tail(e),
basis\_norm(e),
W2\_delta(e),
probe\_logits(e),
bad\_ucb(e),
null\_ucb(e),
support\_lcb(e)
).
$$

必须记录各 component error：

```text
tail_threshold_error
tail_stat_error
basis_norm_error
W2_delta_error
probe_logit_error
bad_ucb_error
null_ucb_error
support_lcb_error
bridge_partial_error
```

## 2. Margin-band diagnosis

定义：

$$
margin(e)=score_{\text{ref}}(e)-threshold_{\text{ref}}.
$$

必须记录：

```text
margin_abs
margin_signed
margin_rank
is_borderline_eps_1e_8
is_borderline_eps_1e_7
is_borderline_eps_1e_6
is_borderline_eps_1e_5
```

可视化：

```text
accept disagreement rows on score-threshold margin histogram
score error vs margin scatter
rank error vs margin scatter
```

## 3. Stable tie policy

候选 tie policies：

```text
TIE0-current-reference:
  current score > median behavior

TIE1-score-rank-eventid:
  score rank primary, event_id secondary

TIE2-score-rank-family-bucket-eventid:
  score rank primary, family_id/bucket_id/event_id secondary

TIE3-fixed-K-top-score:
  accept exactly K rows using stable sort

TIE4-borderline-exact-fallback:
  native accept for non-borderline, exact fallback for borderline

TIE5-calibration-margin-rule:
  frozen calibration chooses threshold with minimum margin band, must rerun all gates
```

Official allowed paths:

```text
TIE0 only if accept disagreement = 0
TIE1/TIE2/TIE3 allowed only after reference/native both use same policy and full decision gates rerun
TIE4 allowed if exact fallback ratio and step ratio pass
TIE5 allowed only if full calibration/heldout/support gates rerun
```

Forbidden:

```text
tolerance-only accept
dataset-specific tie policy
posthoc choosing accept rule based on paired replay
moving accept back to Python while claiming inside-kernel pass
```

---

# Part V. Candidate designs

## 1. Accept-disagreement autopsy candidates

### AD0：P16 native CUDA reference

Current native bucket kernel with accept disagreement.

### AD1：MarginBandAutopsy

Compute score margin and error for all disagreement and non-disagreement rows.

### AD2：ComponentErrorAutopsy

Decompose score error into tail / basis_norm / delta / logits / bridge components.

### AD3：RankTieAutopsy

Compare reference rank and native rank under stable event-id tie.

### AD4：DatasetFamilyHorizonDiagnostic

Diagnostic only; report disagreement distribution by dataset/family/horizon/bucket.  
Cannot be used for dataset-specific tuning.

## 2. Stable accept candidates

### AC0：CurrentMedianThreshold

Current reference behavior. Official only if disagreement = 0.

### AC1：BitExactReferenceReduce

Try to reproduce exact reference ordering/dtype/reduction.

### AC2：StableRankEventIdTie

Stable top-K / median-rank selection with event-id tie.

### AC3：StableFamilyBucketEventTie

Stable tie key includes family/bucket/horizon/event id.

### AC4：BorderlineExactFallback

Native fast path + exact fallback for borderline rows.

### AC5：CalibrationMarginRule

Calibration chooses a threshold with minimum margin buffer; all decision gates rerun.

### AC6：DualThresholdAbstainBorderline

Accept high-confidence positives, abstain borderline; must preserve coverage >=0.03.

## 3. Native kernel candidates

### NK0：P16 native CUDA bucket kernel reference

Current path with q90 reduction but accept disagreement.

### NK1：NativeStableRankKernel

Native kernel outputs bridge score and rank/tie accept.

### NK2：NativeBorderlineFlagKernel

Native kernel outputs bridge score, accept bit, and borderline flag.

### NK3：NativeExactFallbackIntegratedKernel

Kernel writes non-borderline accept; fallback kernel recomputes borderline exact.

### NK4：NativeCompactBridgeLookupKernel

Move bad/null/support compact lookup inside kernel.

### NK5：NativeBatchMajorBucketKernel

Increase candidates per kernel and reduce launch/sync.

### NK6：TritonBatchMajorBucketKernel

Alternative if CUDA extension remains fragile.

## 4. System candidates

### SYS0：v9.2.77 route reference

Expected fail.

### SYS1：AD1 + AC1 + NK1

Bit-exact reference reduce path.

### SYS2：AD1 + AC2 + NK1

Stable rank/event-id tie path.

### SYS3：AD1 + AC4 + NK2/NK3

Borderline exact fallback path.

### SYS4：AC2 + NK5

Stable rank tie + batch-major native bucket.

### SYS5：AC4 + NK5

Borderline exact fallback + batch-major native bucket.

### SYS6：AC5 + NK5

Calibration margin rule + batch-major native bucket; requires full rerun.

### SYS7：HybridBestRuntimeV9278

Best official-eligible candidate from SYS1-SYS6.

---

# Part VI. 实验阶段

## P0：v9.2.77 boundary reproduction

### 目标

确认 v9.2.77 boundary 和 P16 native-kernel failure 稳定。

### 必须记录

```text
route
source_route_v9277
payload_binding_contract_pass
candidate_tensor_payload_missing_count
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
functional_update_payload_missing_count
quantile_tail_runtime_pass
basis_norm_runtime_pass
static_bucket_workspace_pass
single_pass_bridge_pass
native_cuda_bucket_kernel_used
native_cuda_bucket_kernel_q90_reduction
accept_disagreement_count
bridge_score_error_max
controller_step_ratio_q90
official_eligible
system_legal_controller_pass
fake_proxy_count
cpu_offload_used
```

### 判断标准

P0 pass：

```text
route = R17-BasisNormRuntimeStillInsufficient
payload binding pass = 1
quantile_tail runtime pass = 1
native kernel attempted/used = 1
system controller pass = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9277_boundary_ladder.svg
p0_route_progress_v9272_v9277.svg
p0_native_kernel_status_dashboard.svg
```

---

## P1：native accept disagreement autopsy

### 目标

定位 accept disagreement 的根因。P1 是 v9.2.78 最关键阶段：如果不知道 disagreement 来源，不能随意改 kernel 或 tie rule。

### 必须记录

```text
autopsy_candidate_id
event_id
candidate_id
dataset
family_id
bucket_id
horizon
score_ref
score_native
score_abs_err
score_rel_err
threshold_ref
threshold_native
threshold_abs_err
rank_ref
rank_native
tie_key_ref
tie_key_native
accept_ref
accept_native
accept_disagreement
margin_ref
margin_native
margin_abs
is_borderline_1e_8
is_borderline_1e_7
is_borderline_1e_6
is_borderline_1e_5
tail_error
basis_norm_error
W2_delta_error
probe_logits_error
bad_ucb_error
null_ucb_error
support_lcb_error
bridge_partial_error
```

### 判断标准

P1 pass：

```text
accept_disagreement_count measured
all disagreement rows have component decomposition
all disagreement rows have margin diagnostics
dominant disagreement mode identified
```

H1 pass：

```text
bridge_score_error_max <= 1e-6
large_margin_disagreement_count = 0
borderline_disagreement_fraction >= 0.90
```

If not, route to kernel correctness repair.

### 可视化

```text
p1_accept_disagreement_margin_hist.svg
p1_score_error_vs_margin.svg
p1_component_error_waterfall.svg
p1_disagreement_by_family_bucket_horizon.svg
```

---

## P2：stable accept contract implementation

### 目标

实现 reference/native 一致的 stable accept contract。不能只在 native kernel 内临时改 tolerance。

### 必须记录

```text
accept_contract_id
tie_policy
accept_rule_changed
reference_rule_updated
native_rule_updated
calibration_rerun_required
stable_sort_used
stable_tie_key
borderline_fallback_used
borderline_epsilon
borderline_count
borderline_rate
accept_disagreement_before
accept_disagreement_after
agreement_reference_accept
precision
coverage
bad_event
null_rate
precision_lcb
bad_event_ucb
```

### 判断标准

P2 pass：

```text
accept_disagreement_after = 0
or
borderline_fallback_used = 1 and final_accept_disagreement = 0
agreement_reference_accept >= 0.99
```

If rule changed：

```text
calibration_rerun_required = 1
heldout gates must be rerun before official promotion
```

### 可视化

```text
p2_accept_contract_comparison.svg
p2_borderline_rate_vs_epsilon.svg
p2_decision_metric_drift.svg
p2_accept_overlap_confusion.svg
```

---

## P3：native bucket kernel v2

### 目标

将 P2 accept contract 接入 native bucket kernel，并保持 runtime reduction。

### 必须记录

```text
native_kernel_id
accept_contract_id
native_cuda_bucket_kernel_used
triton_bucket_kernel_used
basis_norm_bucketed
W2_delta_bucketed
bridge_score_inside_kernel
accept_bit_inside_kernel
borderline_flag_inside_kernel
compact_bridge_lookup_inside_kernel
kernel_count_before
kernel_count_after
sync_count_before
sync_count_after
allocation_count_before
allocation_count_after
avg_candidates_per_kernel_before
avg_candidates_per_kernel_after
native_time_ms_mean
native_time_ms_q90
eager_time_ms_q90
q90_reduction
accept_disagreement_count
bridge_score_error_max
logits_error_max
delta_error_max
memory_ratio
step_ratio_q90
```

### 判断标准

P3 pass：

```text
native_cuda_bucket_kernel_used = 1 or triton_bucket_kernel_used = 1
basis_norm_bucketed = 1
W2_delta_bucketed = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
accept_disagreement_count = 0
q90_reduction >= 0.40
```

System diagnostic:

$$
StepRatio_{q90}\leq2.00.
$$

Full system candidate:

$$
StepRatio_{q90}\leq1.50.
$$

### 可视化

```text
p3_native_kernel_runtime_pareto.svg
p3_kernel_sync_allocation_reduction.svg
p3_accept_correctness_vs_runtime.svg
p3_candidates_per_kernel_histogram.svg
```

---

## P4：borderline exact fallback system

### 目标

如果 stable accept bit-exact path 不闭合，则实现 exact borderline fallback，并验证成本仍可接受。

### 必须记录

```text
fallback_candidate_id
native_kernel_id
borderline_epsilon
borderline_count
borderline_rate
exact_fallback_count
exact_fallback_time_ms_mean
exact_fallback_time_ms_q90
native_fast_path_time_ms_q90
combined_time_ms_q90
final_accept_disagreement_count
agreement_reference_accept
precision
coverage
bad_event
null_rate
step_ratio_q90
memory_ratio
```

### 判断标准

P4 pass：

```text
final_accept_disagreement_count = 0
borderline_rate <= 0.05
agreement_reference_accept >= 0.99
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

Diagnostic pass：

```text
borderline_rate <= 0.10
step_ratio_q90 <= 2.00
```

### 可视化

```text
p4_borderline_rate_cost_curve.svg
p4_fallback_time_distribution.svg
p4_final_accept_confusion.svg
```

---

## P5：batch-major native bucket runtime

### 目标

解决 SP3/P16 的结构性问题：avg candidates per kernel 仍低，kernel/sync 不降。

### 必须记录

```text
batch_major_candidate_id
bucket_strategy
bucket_sizes
candidate_count
effective_candidate_count
kernel_count_before
kernel_count_after
sync_count_before
sync_count_after
allocation_count_before
allocation_count_after
avg_candidates_per_kernel_before
avg_candidates_per_kernel_after
effective_candidates_per_launch
workspace_memory_MB
native_time_ms_q90
step_ratio_q90
accept_disagreement_count
agreement_reference_accept
```

### 判断标准

P5 pass：

```text
avg_candidates_per_kernel_after >= 8
or effective_candidates_per_launch >= 8
kernel_count_after <= 0.50 * kernel_count_before
sync_count_after <= 0.50 * sync_count_before
accept_disagreement_count = 0
```

System candidate:

$$
StepRatio_{q90}\leq1.50.
$$

### 可视化

```text
p5_bucket_occupancy.svg
p5_candidates_per_kernel_before_after.svg
p5_kernel_sync_vs_step_ratio.svg
p5_bucket_strategy_pareto.svg
```

---

## P6：system-legal exact-signal controller v10

### 目标

只有 P3/P4/P5 有 official-eligible native runtime survivor 时，P6 才能 official pass。

### 必须记录

```text
controller_id
system_candidate_id
accept_contract_id
native_kernel_id
fallback_candidate_id
bucket_strategy
prefilter_id
thresholds
calibration_split_id
heldout_split_id
event_count
candidate_count
accepted_count
candidate_rate
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
accepted_signal_strata_count
accepted_family_count
max_family_share
max_stratum_share
AUC_safe_good
AUC_bridge_accept
agreement_reference_accept
accept_disagreement_count
borderline_rate
step_ratio_q90
memory_ratio
kernel_count
sync_count
allocation_count
avg_candidates_per_kernel
candidate_tensor_payload_missing_count
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
functional_update_payload_missing_count
materialized_system_path
native_bucket_kernel_used
bridge_score_inside_kernel
accept_bit_inside_kernel
borderline_fallback_used
audit_only_cost_removal
diagnostic_derived_from_measured_components
projection_used
full_trace_projection_used
full_online_row_binding
full_online_payload_binding
full_online_update_payload_binding
source_measured_gap_used
formula_proxy_used
cpu_offload_used
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
official_eligible
system_legal_controller_pass
```

### 判断标准

P6 pass：

```text
official_eligible = 1
system_legal_controller_pass = 1
materialized_system_path = 1
native_bucket_kernel_used = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
accept_disagreement_count = 0
audit_only_cost_removal = 0
diagnostic_derived_from_measured_components = 0
full_online_row_binding = 1
full_online_payload_binding = 1
full_online_update_payload_binding = 1
projection_used = 0
full_trace_projection_used = 0
source_measured_gap_used = 0
formula_proxy_used = 0
cpu_offload_used = 0
dataset_name_used = 0
```

Decision gate：

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

System gate：

$$
Agreement_{\text{reference}}\geq0.90,
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
p6_system_controller_cost_quality_frontier.svg
p6_reference_vs_system_accept_overlap.svg
p6_accept_disagreement_zero_audit.svg
p6_step_ratio_progress_v9277_v9278.svg
p6_family_strata_balance.svg
```

---

## P7：leave-dataset-out / leave-stratum-out

### 目标

证明 system controller 不是 pooled calibration artifact，也不是 dataset-specific route。

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
system_candidate_id
accept_contract_id
native_kernel_id
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
branches =
  RealFunctional
  AdamWOnly
  AdamWParallel
  bestLR
  NoOp
  Random
  ShuffledTrueBranchDelta
  ShuffledPF5Prefilter
  ShuffledCandidatePayload
  ShuffledQuantileTailRuntime
  ShuffledBasisNormRuntime
  ShuffledNativeBucketKernel
  ShuffledBridgeScore
  ShuffledAcceptTiePolicy
  ShuffledAcceptBit
  ShuffledBorderlineFallback
  ShuffledFunctionalUpdatePayload
  ShuffledFrozenBridge
  ShuffledSupportStat
  ShuffledControlGain
  ShuffledCandidateGate
  ShuffledBranchRatio
  ShuffledSignalChannel
  FunctionalChannelShuffled
  TailMaskShuffled
  RoleScoreShuffled
  DatasetRouteShuffled
  EventRouteShuffled
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

Shuffle controls must fail：

```text
ShuffledTrueBranchDelta = fail
ShuffledPF5Prefilter = fail
ShuffledCandidatePayload = fail
ShuffledQuantileTailRuntime = fail
ShuffledBasisNormRuntime = fail
ShuffledNativeBucketKernel = fail
ShuffledBridgeScore = fail
ShuffledAcceptTiePolicy = fail
ShuffledAcceptBit = fail
ShuffledBorderlineFallback = fail
ShuffledFunctionalUpdatePayload = fail
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
controls =
  AdamWOnly
  AdamWParallel
  bestLR
  StrongLRGrid
  QuadraticFeatureMLP
  NoOp
  Random
  ShuffledTrueBranchDelta
  ShuffledPF5
  ShuffledNativeBucketKernel
  ShuffledBridgeScore
  ShuffledAcceptTiePolicy
  ShuffledBorderlineFallback
  ShuffledFunctionalUpdatePayload
  ShuffledFrozenBridge
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
Functional not explained by shuffled controller/kernel/payload
```

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9278.csv
p0_v9277_boundary_reproduction.csv
p1_native_accept_disagreement_autopsy.csv
p2_stable_accept_contract_implementation.csv
p3_native_bucket_kernel_v2.csv
p4_borderline_exact_fallback_system.csv
p5_batch_major_native_bucket_runtime.csv
p6_system_legal_exact_signal_controller_v10.csv
p7_leave_dataset_and_stratum_out.csv
p8_official_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_run_robustness_strong_baseline.csv

accept_disagreement_trace_v9278.csv
bridge_score_component_error_trace_v9278.csv
stable_accept_contract_trace_v9278.csv
native_bucket_kernel_v2_trace_v9278.csv
borderline_fallback_trace_v9278.csv
batch_major_bucket_trace_v9278.csv
system_controller_trace_v9278.csv
leaveout_trace_v9278.csv
paired_replay_branch_trace_v9278.csv
short_run_trace_v9278.csv

route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9277_boundary_unstable
F3_dataset_tuning_detected
F4_payload_binding_regression
F5_decision_metric_regression
F6_native_accept_autopsy_incomplete
F7_accept_disagreement_large_margin
F8_bridge_score_component_error_large
F9_tail_or_delta_mismatch
F10_stable_tie_policy_not_defined
F11_reference_native_policy_mismatch
F12_accept_disagreement_not_zero
F13_borderline_rate_too_high
F14_borderline_fallback_too_expensive
F15_native_kernel_correctness_fail
F16_native_kernel_runtime_regression
F17_kernel_sync_allocation_unfixed
F18_avg_candidates_per_kernel_too_low
F19_bridge_score_not_inside_kernel
F20_accept_bit_not_inside_kernel
F21_source_gap_or_formula_proxy_used
F22_system_controller_precision_fail
F23_system_controller_coverage_fail
F24_system_controller_bad_event_fail
F25_system_controller_null_rate_fail
F26_system_controller_lcb_ucb_fail
F27_system_controller_step_ratio_fail
F28_system_controller_memory_fail
F29_system_controller_projection_used
F30_leave_dataset_out_fail
F31_leave_stratum_out_fail
F32_paired_replay_control_equivalent
F33_shuffle_control_pass
F34_functional_lr_equivalent
F35_short_run_task_drop
F36_full_run_no_macro_hard_stratum_gain
F37_strong_baseline_explains_gain
F38_robustness_fail
F39_external_not_ready
F40_fake_or_proxy_violation
F41_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.77 boundary reproduced.

R2-NativeAcceptAutopsyPass:
  accept disagreement root cause localized.

R3-StableAcceptContractPass:
  stable accept contract implemented with zero final disagreement.

R4-NativeBucketKernelV2Pass:
  native bucket kernel preserves correctness and runtime reduction.

R5-BorderlineFallbackPass:
  exact fallback eliminates disagreement at acceptable cost.

R6-BatchMajorBucketRuntimePass:
  avg candidates per kernel / launch efficiency improves.

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
  full run task / geometry / system / control gates pass.

R13-PayloadBindingRegression:
  payload binding no longer reproduces.

R14-NativeAcceptDisagreementUnattributed:
  accept disagreement not localized.

R15-NativeKernelNumericalCorrectnessFail:
  disagreement caused by large component error or wrong row binding.

R16-StableAcceptContractFail:
  stable tie/rank/fallback cannot eliminate disagreement.

R17-NativeRuntimeStillTooFragmented:
  correctness fixed but kernel/sync/candidates-per-kernel remain poor.

R18-SystemStillTooExpensive:
  materialized native runtime correct but step_ratio_q90 remains >1.50.

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
v9277_boundary_pass
dataset_tuning_detected
reference_controller_id
payload_binding_contract_pass
candidate_tensor_payload_missing_count
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
functional_update_payload_missing_count

native_accept_autopsy_pass
accept_disagreement_count_before
accept_disagreement_root_cause
large_margin_disagreement_count
borderline_disagreement_fraction
bridge_score_error_max
tail_error_max
basis_norm_error_max
W2_delta_error_max
probe_logits_error_max

best_accept_contract_id
stable_accept_contract_pass
tie_policy
accept_rule_changed
borderline_fallback_used
borderline_rate
accept_disagreement_count_after
agreement_reference_accept

best_native_kernel_id
native_bucket_kernel_v2_pass
native_cuda_bucket_kernel_used
triton_bucket_kernel_used
basis_norm_bucketed
W2_delta_bucketed
bridge_score_inside_kernel
accept_bit_inside_kernel
kernel_count_reduction
sync_count_reduction
allocation_count_reduction
avg_candidates_per_kernel_after
q90_reduction

best_system_controller_id
system_legal_controller_pass
official_eligible
controller_precision
controller_coverage
controller_bad_event
controller_null_rate
controller_precision_lcb
controller_bad_event_ucb
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
success_v9278_strict_purekan_functional
success_v9278_full_functional
success_v9278_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 native accept disagreement autopsy
  P2 stable accept contract implementation
  P3 native bucket kernel v2
  P4 borderline exact fallback
  P5 batch-major native bucket runtime

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

Gate rule：

```text
P2/P3/P4/P5 can run in parallel after P1 autopsy.
P6 cannot pass unless:
  P0 pass
  P1 accept autopsy pass
  P2 stable accept contract pass or P4 borderline fallback pass
  P3 or P5 native runtime pass
  accept_disagreement_count_after = 0
  step_ratio_q90 <= 1.50
  materialized_system_path = 1
  native_bucket_kernel_used = 1
  bridge_score_inside_kernel = 1
  accept_bit_inside_kernel = 1
  audit_only_cost_removal = 0
  diagnostic_derived_from_measured_components = 0
  projection_used = 0
  source_measured_gap_used = 0
  formula_proxy_used = 0
  full_online_payload_binding = 1
  full_online_update_payload_binding = 1
```

P7/P8 diagnostic rows may be measured before all gates finish, but official status requires：

```text
base robust pass
attach equivalence pass
no-event preservation pass
carrier active
reference controller reproduced
frozen controller pass
PF5 runtime selector pass
payload binding pass
true branch-delta legality pass
true branch-delta predictivity pass
true branch-delta agreement pass
true branch-delta system pass
functional update payload pass
system-legal controller pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.77 boundary reproduced
native accept disagreement autopsy completed
stable accept contract measured
native bucket kernel v2 measured
borderline fallback measured
batch-major native bucket measured
system controller measured
no fake/proxy/offload/loss/teacher violation
```

## Accept closure success

```text
Minimum diagnostic success
+
accept_disagreement_count_after = 0
+
agreement_reference_accept >= 0.99
+
bridge_score_inside_kernel = 1
+
accept_bit_inside_kernel = 1
+
no source gap / formula proxy / projection
```

## Runtime materialization success

```text
Accept closure success
+
native_bucket_kernel_used = 1
+
materialized_runtime_path = 1
+
kernel/sync/allocation reduction measured
+
avg_candidates_per_kernel_after >= 8
+
q90_reduction >= 0.40
```

## System success

```text
Runtime materialization success
+
step_ratio_q90 <= 1.50
+
memory_ratio <= 1.05
+
decision gates pass
+
official_eligible = 1
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
1. v9.2.77 boundary cannot be reproduced；
2. payload binding regresses；
3. decision metrics regress；
4. accept disagreement autopsy incomplete；
5. disagreement rows have large safe margin；
6. bridge score component error is large；
7. native kernel row binding mismatch；
8. stable tie policy cannot eliminate disagreement；
9. borderline fallback rate too high；
10. fallback cost pushes step_ratio_q90 >1.50；
11. native kernel loses q90 reduction after correctness repair；
12. bridge score cannot stay inside kernel；
13. accept bit cannot stay inside kernel；
14. kernel/sync/allocation remains unchanged；
15. avg candidates per kernel remains <2；
16. low-cost path uses source gap / formula proxy；
17. system step_ratio_q90 remains >1.50；
18. memory ratio >1.05；
19. system controller fails precision / coverage / bad-event / null-rate；
20. precision LCB below 0.75；
21. bad-event UCB above 0.05；
22. leave-dataset-out fails；
23. leave-stratum-out fails；
24. paired replay remains control-equivalent；
25. shuffle controls pass；
26. short-run task drops；
27. full run gives no macro / hard-stratum / geometry gain；
28. functional breaks system gate；
29. gains are explained by QuadraticFeatureMLP；
30. any teacher/loss/fake/proxy/offload/projection-as-pass violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：native accept closure + system pass + LDO/P8 pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full run and robustness。

## Case B：accept disagreement comes from large component error

必须声明：

```text
native kernel correctness is not closed; current blocker is kernel component mismatch, not tie policy.
```

下一步修 tail/basis_norm/delta/bridge component error。

## Case C：accept disagreement only occurs near median boundary

必须声明：

```text
current accept rule is numerically fragile; stable rank/tie or exact borderline fallback is required.
```

下一步做 stable accept contract，不调 dataset threshold。

## Case D：stable accept closes but system still slow

必须声明：

```text
decision correctness is closed, but native runtime remains too fragmented or too expensive.
```

下一步修 batch-major bucket/kernel launch/sync。

## Case E：system pass but paired replay fail

必须声明：

```text
controller is system-legal but not causally superior to matched controls.
```

下一步回到 functional event/value target，而不是继续 kernelization。

## Case F：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能通过 dataset-specific tuning 写成功；下一步修 support/family reliability。

---

# Part XII. 最终建议

v9.2.78 的一句话策略是：

$$
\boxed{
\text{不要再泛泛优化 runtime；现在主攻 native bridge accept 的数值稳定性，把 q90 reduction 变成 official system pass。}
}
$$

当前最关键的问题不是：

```text
oracle 是否存在；
reference frontier 是否 deployable；
PF5 candidate rate 是否可行；
payload 是否 missing；
quantile-tail 是否可优化；
basis_norm 是否有局部 runtime；
static bucket 是否能写 workspace；
C3/T2/C4/E2 controller 是否要重调；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. P16 的 10 个 accept disagreement 是否全部处在 median 边界？
2. bridge_score_error_max 约 1e-7 为什么足以翻转 accept？
3. reference accept rule 是否有稳定 tie policy？
4. native kernel 能否用 stable rank/event-id tie 和 reference 完全一致？
5. borderline exact fallback rate 是否足够低？
6. fallback 后 step q90 是否仍 <=1.50？
7. native bucket kernel correctness 修好后 q90 reduction 是否仍 >=40%？
8. kernel/sync/allocation 和 avg candidates per kernel 是否真正改善？
9. system controller 是否 official eligible？
10. LDO/LSO 是否通过？
11. official paired replay 是否打过 AdamWParallel / bestLR？
```

v9.2.78 的结果将给出清晰分叉：

```text
if stable accept + native kernel passes system gate:
  open system-legal controller, LDO/LSO, paired replay.

if accept disagreement is large-margin:
  repair native kernel component correctness.

if disagreement is boundary-only:
  implement stable tie/rank or exact borderline fallback.

if correctness closes but step ratio remains >1.50:
  continue batch-major bucket / kernel launch / sync reduction.

if compute passes but paired replay fails:
  system is legal, but functional causal advantage is insufficient.

if LDO/LSO fails:
  repair dataset-agnostic support/family reliability, not dataset-specific tuning.
```
