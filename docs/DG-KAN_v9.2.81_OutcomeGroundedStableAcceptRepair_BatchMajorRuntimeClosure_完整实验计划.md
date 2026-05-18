# DG-KAN v9.2.81 Outcome-Grounded StableAccept Repair 与 Full-System BatchMajor Runtime Closure 完整实验计划

> 本计划基于 v9.2.80 `Full-Train-Stream StableAccept Materializer 与 Outcome-Label Rebuild` 的真实执行结果制定。  
> v9.2.80 的 terminal route 是：
>
> ```text
> route = R15-StableAcceptNativeQuantizationMismatch
> base_candidate = LQ-t2-h256
> success_v9280_strict_purekan_functional = False
> success_v9280_full_functional = False
> success_v9280_external_ready = False
> ```
>
> v9.2.80 的核心事实是：
>
> ```text
> Materializer:
>   event_count = 24192
>   candidate_count = 2876
>   stable_candidate_rows = 2876
>   stable_accept_full_row_materialization_present = 1
>   outcome_labels_present = 1
>   missing_primary_label_count = 0
>   label_join_mode = direct_same_run_event_id
>   candidate_missing_count = 0
>   duplicate_event_id_count = 0
>   accept_disagreement_count = 0
>   score_quantized_disagreement_count = 18
>   rank_disagreement_count = 2
>   secondary_outcome_delta_field_missing_count = 14380
>
> Calibration / heldout:
>   stable_accept_official_calibration_pass = 0
>   stable_accept_heldout_support_pass = 0
>   accepted_count_held_native = 468
>   precision_heldout = 0.5128205128205128
>   coverage_heldout = 0.051587301587301584
>   bad_event_heldout = 0.3034188034188034
>   null_rate_heldout = 0.11752136752136752
>   precision_lcb = 0.47816314913056096
>   bad_event_ucb = 0.33529565729782573
>   accepted_signal_strata_count = 21
>   accepted_family_count = 66
>
> Runtime:
>   native_bucket_kernel_used_in_p6 = 1
>   stable_accept_cuda_kernel_used = 1
>   basis_norm_bucketed = 1
>   W2_delta_bucketed = 1
>   bridge_score_inside_kernel = 1
>   accept_bit_inside_kernel = 1
>   new_full_system_step_ratio_measured = 1
>   old_step_ratio_reused_as_measurement = 0
>   native_time_ms_q90 = 0.6184950470924377
>   eager_time_ms_q90 = 1.0737529955804348
>   q90_reduction = 0.4239875933867825
>   controller_step_ratio_q90 = 2.213009156635521
>   kernel_count_before = 3750
>   kernel_count_after = 72576
>   sync_count_before = 1250
>   sync_count_after = 8064
>   avg_candidates_per_kernel_after = 0.03962742504409171
>   batch_major_runtime_pass = 0
>
> P6:
>   official_eligible = 0
>   system_legal_controller_pass = 0
>   reason = stable_accept_heldout_support_failed
>   primary_blocker = stable_accept_native_quantized_rank_mismatch
> ```
>
> v9.2.81 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.80 已经解决“不能评估”的问题；现在暴露出两个真实 blocker：decision region 不安全，以及 full-system runtime 仍碎片化。}
> }
> $$
>
> 因此 v9.2.81 不能继续只做 materializer，也不能继续只修 local native kernel。  
> 本轮必须围绕两个并行目标推进：
>
> ```text
> 1. 让 AC2Q2 / StableAccept 的 official accept region 在 same-run outcome labels 下重新变安全；
> 2. 让 native stable bucket kernel 在 full-system path 中保持 batch-major efficiency。
> ```
>
> 更直白地说，v9.2.80 的路线名强调 `native_quantized_rank_mismatch`，这个 blocker 是真实的；但独立看数据，更大的 blocker 是：
>
> $$
> Precision=0.5128,\quad BadEvent=0.3034,\quad StepRatio_{q90}=2.2130.
> $$
>
> 如果只修 18 个 quantized disagreement 与 2 个 rank disagreement，而不修 accepted region 的 bad-event 和 full-system kernel fragmentation，v9.2.81 仍然不会进入 official controller。

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

v9.2.81 的新增硬约束是：

```text
不能只修 quantized/rank mismatch 就声明 official；
不能沿用旧 C3/T2PlusBackfill metrics；
不能沿用 v9.2.78 local q90；
不能复用旧 source boundary step_ratio_q90；
不能按 dataset 调 StableAccept threshold / tie policy / risk gate；
不能用 secondary outcome delta 缺失的 row 做 paired replay official；
不能把 support count pass 写成 precision/bad-event pass；
不能把 native q90 local reduction 写成 full-system runtime pass；
不能把 batch-major runtime pass 写成 diagnostic-derived；
不能把 accept_disagreement_count = 0 单独写成 stable_accept_full_row_materialization_pass。
```

允许使用：

```text
AC2Q2 stable quantized score / event-id tie；
same-run full-row outcome labels；
same-run secondary outcome deltas；
native CUDA / Triton stable bucket kernel；
compact bridge lookup inside kernel；
dataset-agnostic risk/support filter；
borderline exact fallback；
calibration split frozen thresholds；
heldout rerun；
leave-dataset-out / leave-stratum-out after system pass；
post-step audit subset；
per-dataset diagnostics without dataset-specific dispatch。
```

---

# Part I. 对 v9.2.80 的独立判断

## 1. v9.2.80 没有达到目标

v9.2.80 没有 strict PureKAN functional success。失败不是因为缺少 materializer，而是因为 materializer 成功后真实暴露了两个 gate failure：

```text
stable_accept_heldout_support_pass = 0
system_legal_controller_pass = 0
official_eligible = 0
```

P6 的 official reason 是：

```text
stable_accept_heldout_support_failed
```

这说明 P7-P10 继续 not_run 是正确的。不能用 materializer pass、accept disagreement zero 或 local native q90 reduction 写 paired replay / short-run / full-run success。

## 2. v9.2.80 的真实进展

v9.2.80 的进展非常实在。v9.2.79 的 terminal blocker 是 full-row materialization missing；v9.2.80 已经把这个 blocker 推进掉：

```text
stable_candidate_rows = 2876
outcome_labels_present = 1
missing_primary_label_count = 0
label_join_mode = direct_same_run_event_id
candidate_missing_count = 0
duplicate_event_id_count = 0
```

这意味着现在不是 “无法评估 official gate”，而是可以真实评估 official gate。

第二，P6 native runtime 也不是旧 boundary：

```text
native_bucket_kernel_used_in_p6 = 1
new_full_system_step_ratio_measured = 1
old_step_ratio_reused_as_measurement = 0
```

这比 v9.2.79 有重大推进。v9.2.79 只能说 “local native good but P6 not wired”；v9.2.80 终于把 native kernel 接进了 full-system trace。

第三，P6 step ratio 从旧 source boundary `2.713296` 降到新测 `2.213009`。这说明 full-system native path 有真实收益：

$$
1-\frac{2.213009}{2.713296}\approx0.1844.
$$

也就是约 `18.4%` 的 full-system step ratio 改善。

## 3. v9.2.80 的真实失败

v9.2.80 的失败比 v9.2.79 更深，因为它不再是 “缺字段”，而是 “字段齐了以后 gate 真的失败”。

第一，stable accept full-row 仍不完全 bit-exact：

```text
accept_disagreement_count = 0
score_quantized_disagreement_count = 18
rank_disagreement_count = 2
```

这说明 final accept bit 目前一致，但 score quantization / rank contract 仍不一致。对于 official system 这不能忽略，因为 AC2Q2 的定义是 quantized score + stable rank + event tie。如果 quantized/rank contract 不一致，只看 accept bit 零差异会掩盖边界不稳定性。

第二，decision gate 明显失败：

```text
accepted_count_held_native = 468
precision_heldout = 0.5128205
bad_event_heldout = 0.3034188
```

按 468 个 accepted row 粗算：

$$
SafeGood \approx 468 \times 0.5128205 = 240,
$$

$$
BadEvent \approx 468 \times 0.3034188 = 142,
$$

$$
NullEvent \approx 468 \times 0.1175214 = 55.
$$

这不是小幅漂移。它说明 AC2Q2 stable accept 的 accepted region 在 same-run outcome labels 下混入了大量 bad events。旧 reference frontier 的 precision `0.838`、bad-event `0.0246` 没有在新 official table 上保住。

第三，runtime gate 仍失败：

```text
controller_step_ratio_q90 = 2.213009 > 1.50
```

而且 full-system kernel structure 反而严重碎片化：

```text
kernel_count_after = 72576
sync_count_after = 8064
avg_candidates_per_kernel_after = 0.0396
```

这说明 P6 虽然用了 native kernel，但没有保持 v9.2.78 local batch-major structure。当前 full-system path 可能在 event/candidate 层反复 launch kernel，导致 local native q90 reduction 无法转化成 system envelope pass。

## 4. 当前 blocker 的本质

当前 blocker 应写成：

$$
\boxed{
\text{StableAccept 已 full-row materialized，但 official decision region 与 full-system runtime 都没有闭合。}
}
$$

更具体地说：

```text
1. v9.2.79 的 missing materialization blocker 已推进；
2. AC2Q2 stable accept full-row fields 已落盘；
3. primary same-run outcome labels 已落盘；
4. native stable kernel 已进入 P6；
5. 但是 stable quantized/rank contract 仍不 bit-exact；
6. AC2Q2 accepted region 在 same-run outcome labels 下 precision 太低、bad-event 太高；
7. native runtime 在 full-system path 中 kernel/sync 爆炸，batch-major advantage 没保住；
8. 因此 official controller 仍不能打开。
```

这不是 dataset-specific failure。可以诊断不同 dataset、family、horizon、bucket 的 bad-event 和 kernel occupancy，但不能按 dataset 调 controller 或 runtime route。我们不是在这些数据集上打榜；我们是在验证一个 dataset-agnostic 的 functional controller/system 是否存在。

## 5. 是否在正确道路上

是。路线仍然正确，因为每轮都把 blocker 往更真实的位置推进：

```text
v9.2.78:
  stable accept local correctness + native local kernel pass。

v9.2.79:
  official promotion audit 发现 full-row materialization missing。

v9.2.80:
  full-row materializer 和 same-run primary labels 已落盘；
  native kernel 已接进 P6；
  official gates 从 blocked 变成 measured failure。
```

现在不应回头调 C3/T2PlusBackfill 或重设 safe-useful target。下一步要做的是：

```text
official row table 的 stable/rank contract 修复；
accepted bad-event 的因果分解；
dataset-agnostic risk/support repair；
full-system batch-major runtime rewiring。
```

---

# Part II. v9.2.81 总体目标

v9.2.81 的总体目标是：

$$
\boxed{
\text{在 full-row same-run outcome table 上，修复 AC2Q2/StableAccept 的 official decision gate，并让 native batch-major runtime 进入 } StepRatio_{q90}\leq1.50.
}
$$

强目标：

$$
OfficialEligible=1,
$$

$$
SystemLegalControllerPass=1,
$$

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
BadEventRate_{\text{heldout}}\leq0.05,
$$

$$
StepRatio_{q90}\leq1.50.
$$

最低有效推进目标：

```text
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
decision failure autopsy completed
accepted bad-event attribution fraction >= 0.90
at least one dataset-agnostic repair candidate measured
native full-system batch-major integration measured
step_ratio_q90 <= 2.00
```

本轮必须回答十二个问题：

```text
Q1:
  v9.2.80 boundary 是否稳定复现？

Q2:
  candidate_count 为什么从 v9.2.79 的 2493 变成 2876？
  是 full materializer 发现了更多合法 candidates，还是 candidate generator drift？

Q3:
  18 个 quantized disagreement 和 2 个 rank disagreement 的根因是什么？
  round mode、fp32/fp64、score scale、tie key、row order、native kernel output，还是 event_id mismatch？

Q4:
  accept_disagreement = 0 但 quantized/rank disagreement 非零是否会在 borderline row 上引发后续 instability？

Q5:
  precision 从旧 reference 的 0.838 跌到 0.513，是 outcome label definition 改变、candidate set 改变、stable accept region 改变，还是 risk/support filter 丢失？

Q6:
  142 个 bad accepted rows 属于哪些 failure mode？
  bad UCB under-estimated、support LCB unreliable、family instability、horizon tail、quantile-tail context、null/bad conflict，还是 outcome label bug？

Q7:
  secondary CEp99/margin/ECE/NLL/curvature fields 缺失是否影响 bad-event 定义和 paired replay readiness？

Q8:
  是否存在 dataset-agnostic risk/support repair，使 precision/bad-event 过线，同时 coverage 保持在 [0.03,0.15]？

Q9:
  native kernel 在 P6 中为什么从 local 216 kernels / 24 sync 变成 72576 kernels / 8064 sync？

Q10:
  full-system P6 是否可以恢复 batch-major grouping，使 avg_candidates_per_kernel_after >=8？

Q11:
  system pass 后 LDO/LSO 是否通过？

Q12:
  official paired replay 是否能打过 AdamWParallel / bestLR？
```

---

# Part III. 核心假设

## H1：StableAccept quantized/rank mismatch 是 contract bug，不是 accept-region 科学失败的主因

H1 认为 18 个 quantized disagreement 和 2 个 rank disagreement 是 native/reference bit-exact contract bug。它必须修，但它不解释 precision 0.513 / bad-event 0.303 的全部失败，因为 accept_disagreement_count 已经是 0。

H1 成立标准：

```text
quantized_rank_autopsy_pass = 1
score_quantized_disagreement_count_after = 0
rank_disagreement_count_after = 0
accept_disagreement_count_after = 0
decision metrics after bit-exact repair remain approximately same
```

若修复 quantized/rank 后 decision metrics 仍低：

```text
precision_heldout < 0.75
bad_event_heldout > 0.05
```

则说明真正 decision blocker 是 accept region / risk support，而不是 quantization.

H1 失败标准：

修复 quantized/rank 后 accepted set 大幅改变，并使 precision/bad-event 恢复到 gate 内。若如此，v9.2.81 的 route 应转为 `R2-QuantizedRankContractWasPrimary`.

## H2：same-run outcome labels 暴露了旧 AC2Q2 accepted region 的 bad-event contamination

H2 认为 v9.2.80 的 same-run labels 比旧 artifact join 更可信，因此 precision/bad-event 失败是真实的 accepted region 问题。

H2 成立标准：

```text
outcome_label_audit_pass = 1
missing_primary_label_count = 0
ambiguous_label_count = 0
same-run label consistency pass = 1
bad accepted rows have valid same-run event_id/candidate_id binding
```

H2 失败标准：

发现 outcome labels 定义错误、event_id mismatch、branch/horizon mismatch 或 safe_good/bad/null label 互斥关系不成立。若 H2 失败，不能修 controller，必须先修 label materializer。

## H3：bad-event contamination 可以通过 dataset-agnostic risk/support filter 修复

H3 认为 AC2Q2 stable accept 本身太宽，但可以加入原 C3/T2PlusBackfill 风格的 risk/support constraints，例如：

```text
bad_ucb <= threshold
null_ucb <= threshold
support_lcb >= threshold
family_reliability_lcb >= threshold
horizon_tail_risk <= threshold
stable_score_margin >= threshold
```

这些阈值只能从 calibration split 冻结，不能按 dataset 调。

H3 成立标准：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
BadEventRate_{\text{heldout}}\leq0.05,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
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

H3 失败标准：

所有 dataset-agnostic filters 要么 coverage <0.03，要么 precision/bad-event 不过。此时应承认 AC2Q2 stable accept region 与 same-run outcome target 不匹配，需要重新设计 accept score，不应按 dataset 调 threshold。

## H4：candidate set drift 是 decision failure 的可能来源

v9.2.79 candidate count 是 2493，v9.2.80 candidate count 是 2876。H4 认为 candidate set drift 可能引入了更多 high-risk candidates，导致 stable accept region 变差。

H4 成立标准：

```text
candidate_set_jaccard_old_new < 0.90
new_only_candidate_count > 0
new_only_bad_event_rate > old_candidate_bad_event_rate
new_only_accept_share is nontrivial
```

H4 失败标准：

old/new candidate set 基本一致，decision failure 不来自 candidate generation drift。

## H5：P6 runtime failure 来自 full-system wiring fragmentation，而不是 native kernel arithmetic

H5 认为 P6 中 kernel_count_after = 72576、sync_count_after = 8064、avg_candidates_per_kernel_after = 0.0396 说明 runtime 被 per-row/per-candidate launch 破坏，而不是 native kernel arithmetic 慢。

H5 成立标准：

```text
runtime_fragmentation_autopsy_pass = 1
kernel_launch_by_candidate_loop dominates
sync_by_event_loop dominates
avg_candidates_per_kernel_after < 1 before repair
batch-major rewiring raises avg_candidates_per_kernel_after >= 8
```

H5 失败标准：

即使 batch-major rewiring 后，arithmetic component 仍使 step_ratio_q90 >1.50。此时才转入 deeper kernel arithmetic optimization。

## H6：system pass 后，paired replay 才是 functional causal advantage 的问题

H6 成立标准：

```text
P8 system controller pass = 1
P9 LDO/LSO measured
P10 paired replay measured
paired replay fails vs AdamWParallel / bestLR
```

H6 失败标准：

P8 未过时提前讨论 functional event/value target。这会混淆 system legality 与 causal advantage。

---

# Part IV. Required data contracts

## 1. Stable quantized rank contract

每个 candidate row 必须记录：

```text
event_id
candidate_id
score_ref
score_native
score_abs_err
score_quantized_ref
score_quantized_native
quantization_mode_ref
quantization_mode_native
stable_rank_ref
stable_rank_native
tie_key_ref
tie_key_native
accept_ref
accept_native
accept_disagreement
score_quantized_disagreement
rank_disagreement
```

官方 pass：

```text
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
accept_disagreement_count = 0
```

## 2. Outcome label contract

每个 candidate row 必须记录：

```text
event_id
candidate_id
safe_good_label
bad_event_label
null_event_label
task_safe_label
useful_label
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
real_beats_adamwparallel
real_beats_bestlr
outcome_source
outcome_horizon
outcome_branch
```

Primary official gate 至少需要：

```text
safe_good_label
bad_event_label
null_event_label
```

Downstream paired replay readiness 需要 secondary fields：

```text
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
real_beats_adamwparallel
real_beats_bestlr
```

v9.2.81 必须把 secondary outcome delta missing count 从 `14380` 降到 `0`，或者明确写为 downstream_not_ready，并不能打开 P10 paired replay。

## 3. Decision controller contract

官方 controller 不得使用 dataset name。允许使用：

```text
stable score
quantized stable rank
bad_ucb
null_ucb
support_lcb
family_reliability_lcb
horizon_tail_risk
bucket_id
family_id
horizon
event_id tie key
```

禁止：

```text
if dataset == MNIST/Fashion/KMNIST branch
test metric
validation/test leakage at commit time
posthoc outcome label at commit time
source-measured gap
formula proxy
```

## 4. Full-system runtime contract

每个 timed step 必须记录：

```text
native_bucket_kernel_used_in_p6
basis_norm_bucketed
W2_delta_bucketed
bridge_score_inside_kernel
accept_bit_inside_kernel
stable_accept_inside_kernel
kernel_count_before
kernel_count_after
sync_count_before
sync_count_after
allocation_count_before
allocation_count_after
avg_candidates_per_kernel_before
avg_candidates_per_kernel_after
effective_candidates_per_launch
selector_time_ms
candidate_pack_time_ms
native_kernel_time_ms
bridge_score_time_ms
stable_accept_time_ms
update_payload_time_ms
payload_apply_time_ms
audit_outside_timed_time_ms
total_step_time_ms_q90
mlp_step_time_ms_q90
step_ratio_q90
memory_ratio
```

官方 pass：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

---

# Part V. Candidate designs

## 1. Quantized/rank repair candidates

### QR0：v9.2.80 reference

Expected fail:

```text
score_quantized_disagreement_count = 18
rank_disagreement_count = 2
accept_disagreement_count = 0
```

### QR1：IdenticalRoundModeInt64

Reference and native both use explicit integer quantization:

$$
score_q = \operatorname{int64}\left(\operatorname{round}(10^5 score)\right).
$$

All tie/rank uses `score_q` only.

### QR2：FloorOffsetQuantization

Use:

$$
score_q = \left\lfloor 10^5 score + 0.5 \right\rfloor
$$

with explicitly identical fp64 pre-cast before integer conversion.

### QR3：NativeEmitsScoreQOnly

Native kernel emits integer `score_q` directly; reference computes from the same integer path for audit.

### QR4：StableSortEventIdCanonical

For identical `score_q`, rank tie is:

```text
event_id ascending
candidate_id ascending
family_id ascending
bucket_id ascending
```

### QR5：BorderlineQuantizedAudit

For rows where `score_q_ref != score_q_native`, run exact fallback and record whether accept/rank changes.

## 2. Decision-region repair candidates

### DR0：AC2Q2StableAcceptCurrent

Current v9.2.80 accepted region. Expected fail.

### DR1：AC2Q2BitExactOnly

Only quantized/rank repair; no risk filter. Tests whether QR mismatch was primary.

### DR2：AC2Q2PlusBadUCBFilter

Accept only if stable accept and:

$$
BadUCB(e)\leq \tau_b.
$$

$\tau_b$ is chosen on calibration split only.

### DR3：AC2Q2PlusSupportLCBFilter

Accept only if:

$$
SupportLCB(e)\geq \tau_s.
$$

### DR4：AC2Q2RiskSupportPareto

Use both:

$$
BadUCB(e)\leq \tau_b,
$$

$$
SupportLCB(e)\geq \tau_s,
$$

$$
NullUCB(e)\leq \tau_n.
$$

### DR5：AC2Q2StableScoreMarginFilter

Reject borderline score-margin rows if they correlate with bad events:

$$
|score_q(e)-score_q^{threshold}|\geq m.
$$

### DR6：AC2Q2FamilyReliabilityLCB

Dataset-agnostic family reliability gate:

$$
FamilyLCB(family(e))\geq \tau_f.
$$

### DR7：AC2Q2HorizonTailRiskFilter

Reject high tail-risk horizon contexts:

$$
HorizonTailRisk(e)\leq \tau_h.
$$

### DR8：AC2Q2CompositeMonotoneRepair

Monotone repair score:

$$
S_{\text{repair}}(e)
=
a_1 score_q(e)
-
a_2 BadUCB(e)
-
a_3 NullUCB(e)
+
a_4 SupportLCB(e)
+
a_5 FamilyLCB(e).
$$

Weights are calibrated on calibration split under monotonic constraints, not by dataset.

### DR9：TwoStageAC2Q2ThenExactSafety

Stage 1 stable accept; stage 2 exact safety confirm for borderline/high-risk rows.

## 3. Outcome materialization candidates

### OUT0：PrimaryOnlyReference

Current v9.2.80 primary labels only. Not paired replay ready.

### OUT1：SameRunSecondaryDeltaMaterializer

Records CEp99/margin/ECE/NLL/curvature deltas for every candidate.

### OUT2：MatchedControlOutcomeMaterializer

Records RealFunctional / AdamWParallel / bestLR outcomes for every accepted/candidate row.

### OUT3：MultiHorizonOutcomeMaterializer

Records horizon-specific outcome labels, but official horizon is frozen before heldout.

### OUT4：OutcomeConsistencyAudit

Verifies label exclusivity:

```text
safe_good and bad_event cannot both be 1
bad_event and null_event can be analyzed but must match definition
missing secondary fields = 0 for downstream-ready rows
```

## 4. Runtime candidates

### RT0：v9.2.80 full-system native reference

Expected fail:

```text
kernel_count_after = 72576
sync_count_after = 8064
avg_candidates_per_kernel_after = 0.0396
step_ratio_q90 = 2.2130
```

### RT1：P6BatchMajorCandidateTable

Build full candidate table grouped by:

```text
step
bucket_id
family_id
horizon
kernel_shape
```

### RT2：P6PersistentWorkspaceNoPerCandidateSync

Keep all candidate, basis_norm, W2_delta, bridge, accept buffers persistent across grouped launches.

### RT3：P6SingleLaunchPerBucket

One kernel per bucket per step, no per-candidate launches.

### RT4：P6CompactBridgeLookupBatchMajor

Move compact bridge/risk/support lookup inside grouped kernel.

### RT5：P6AsyncAuditOutsideTimedPath

Post-step audit only; no sync or CSV in timed path.

### RT6：P6HybridBestBatchMajorNative

Best combination from RT1-RT5.

---

# Part VI. 实验阶段

## P0：v9.2.80 boundary reproduction

### 目标

确认 v9.2.80 boundary 稳定，并确保本轮不是在不可复现 artifact 上修。

### 必须记录

```text
route
source_route_v9280
candidate_count
event_count
stable_accept_full_row_materialization_present
outcome_labels_present
accept_disagreement_count
score_quantized_disagreement_count
rank_disagreement_count
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
controller_step_ratio_q90
native_bucket_kernel_used_in_p6
kernel_count_after
sync_count_after
avg_candidates_per_kernel_after
official_eligible
system_legal_controller_pass
fake_data_used
proxy_row_used
cpu_offload_used
```

### 判断标准

P0 pass:

```text
route = R15-StableAcceptNativeQuantizationMismatch
full materializer used = 1
outcome labels present = 1
official eligible = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9280_boundary_ladder.svg
p0_decision_runtime_failure_split.svg
p0_old_v9279_to_v9280_progress.svg
```

---

## P1：stable quantized/rank bit-exact repair

### 目标

修复 18 个 score quantized disagreement 和 2 个 rank disagreement。P1 不能直接宣称 decision success；它只关闭 official accept contract correctness。

### 必须记录

```text
quantized_repair_id
event_id
candidate_id
score_ref
score_native
score_abs_err
score_quantized_ref_before
score_quantized_native_before
score_quantized_ref_after
score_quantized_native_after
quantization_mode
rounding_mode
stable_rank_ref_before
stable_rank_native_before
stable_rank_ref_after
stable_rank_native_after
tie_key_ref
tie_key_native
accept_ref
accept_native
accept_disagreement
score_quantized_disagreement
rank_disagreement
```

### 判断标准

P1 pass:

```text
score_quantized_disagreement_count_after = 0
rank_disagreement_count_after = 0
accept_disagreement_count_after = 0
missing_stable_accept_fields = ""
```

### 可视化

```text
p1_quantized_disagreement_before_after.svg
p1_rank_disagreement_trace.svg
p1_score_error_vs_quantized_disagreement.svg
```

---

## P2：candidate-set and row-lifecycle continuity audit

### 目标

解释 candidate_count 从 `2493` 到 `2876` 的变化，并确认 candidate set drift 是否导致 bad-event contamination。

### 必须记录

```text
candidate_set_audit_id
old_candidate_count
new_candidate_count
candidate_jaccard
old_only_candidate_count
new_only_candidate_count
shared_candidate_count
old_only_accept_rate
new_only_accept_rate
shared_accept_rate
old_only_bad_event_rate
new_only_bad_event_rate
shared_bad_event_rate
old_only_precision
new_only_precision
shared_precision
event_id_mismatch_count
candidate_id_mismatch_count
payload_hash_mismatch_count
```

### 判断标准

P2 pass:

```text
candidate lifecycle audit completed
event_id_mismatch_count = 0
candidate_id_mismatch_count = 0
payload_hash_mismatch_count = 0
candidate_count_change_explained = 1
```

If candidate drift is primary:

```text
new_only_bad_event_rate >> shared_bad_event_rate
new_only_accept_share nontrivial
```

### 可视化

```text
p2_candidate_set_venn.svg
p2_old_new_bad_event_comparison.svg
p2_candidate_drift_by_family_horizon.svg
```

---

## P3：outcome label and secondary delta materialization

### 目标

确认 primary outcome labels 可靠，并补齐 secondary outcome deltas，避免 downstream readiness 再次 blocked。

### 必须记录

```text
outcome_materializer_id
event_id
candidate_id
safe_good_label
bad_event_label
null_event_label
task_safe_label
useful_label
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
real_beats_adamwparallel
real_beats_bestlr
outcome_horizon
outcome_branch
label_source
missing_primary_label_count
missing_secondary_delta_count
ambiguous_label_count
label_exclusivity_violation_count
```

### 判断标准

P3 primary pass:

```text
missing_primary_label_count = 0
ambiguous_label_count = 0
label_source = same_run_train_stream
label_exclusivity_violation_count = 0
```

P3 downstream-ready pass:

```text
missing_secondary_delta_count = 0
CEp99_delta present
margin_p10_delta present
ECE_delta present
NLL_delta present
curvature_delta present
real_beats_adamwparallel present
real_beats_bestlr present
```

### 可视化

```text
p3_outcome_label_balance.svg
p3_bad_null_safe_good_overlap.svg
p3_secondary_delta_missing_before_after.svg
p3_outcome_by_family_horizon.svg
```

---

## P4：decision failure autopsy

### 目标

解释为什么 heldout precision 只有 `0.5128`，bad-event 高达 `0.3034`。P4 是本轮科学核心。如果不知道 bad accepted rows 来自哪里，后续 repair 容易变成 blind tuning。

### 必须记录

```text
decision_autopsy_id
accepted_count
safe_good_count
bad_event_count
null_event_count
precision
coverage
bad_event_rate
null_rate
bad_accepted_event_id
bad_accepted_family_id
bad_accepted_bucket_id
bad_accepted_horizon
stable_score_q
stable_rank
bad_ucb
null_ucb
support_lcb
family_lcb
horizon_tail_risk
score_margin
candidate_origin_old_new
failure_mode
failure_mode_fraction
```

Failure modes:

```text
FMA1-quantized-rank-contract
FMA2-candidate-set-drift
FMA3-risk-ucb-underestimation
FMA4-support-lcb-overestimation
FMA5-null-bad-conflict
FMA6-family-instability
FMA7-horizon-tail-risk
FMA8-outcome-label-definition
FMA9-stable-score-miscalibration
FMA10-native-row-binding-mismatch
```

### 判断标准

P4 pass:

```text
bad accepted attribution fraction >= 0.90
precision failure attribution fraction >= 0.90
coverage/null side effects measured
```

### 可视化

```text
p4_bad_accepted_failure_modes.svg
p4_precision_bad_event_tradeoff.svg
p4_bad_event_by_family_horizon_bucket.svg
p4_stable_score_vs_bad_ucb.svg
```

---

## P5：dataset-agnostic StableAccept decision repair

### 目标

在不按 dataset 调参的前提下，修复 AC2Q2 accepted region。P5 不是要打榜，而是验证是否存在 dataset-agnostic legal repair，使 stable accept 在 same-run labels 下恢复 safe-good frontier。

### 必须记录

```text
repair_candidate_id
repair_type
calibration_split_id
heldout_split_id
thresholds
dataset_name_used
event_count
candidate_count
accepted_count_cal
accepted_count_heldout
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
accept_disagreement_count
score_quantized_disagreement_count
rank_disagreement_count
```

### 判断标准

P5 pass:

```text
dataset_name_used = 0
accept_disagreement_count = 0
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

Decision gate:

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

### 可视化

```text
p5_repair_candidate_frontier.svg
p5_calibration_to_heldout_drift.svg
p5_support_balance.svg
p5_old_ac2q2_vs_repaired_accept_overlap.svg
```

---

## P6：full-system runtime fragmentation autopsy

### 目标

解释为什么 P6 native path kernel/sync 爆炸：

```text
kernel_count_after = 72576
sync_count_after = 8064
avg_candidates_per_kernel_after = 0.0396
```

### 必须记录

```text
runtime_autopsy_id
step_id
dataset
family_id
bucket_id
horizon
candidate_count_in_bucket
kernel_launch_count
sync_count
allocation_count
effective_candidates_per_launch
selector_time_ms
candidate_pack_time_ms
native_kernel_time_ms
bridge_lookup_time_ms
stable_accept_time_ms
update_payload_time_ms
payload_apply_time_ms
audit_time_outside_timed
unknown_fraction
dominant_runtime_fragmentation_mode
```

Fragmentation modes:

```text
FR1-per-candidate-kernel-launch
FR2-per-event-sync
FR3-bucket-table-not-reused
FR4-persistent-workspace-not-reused
FR5-bridge-lookup-outside-batch
FR6-audit-sync-in-timed-path
FR7-update-payload-row-loop
FR8-candidate-pack-regather-loop
```

### 判断标准

P6 pass:

```text
unknown_fraction <= 0.05
dominant fragmentation mode identified
at least one removable fragmentation component ratio >= 0.20
```

### 可视化

```text
p6_kernel_launch_by_phase.svg
p6_sync_by_phase.svg
p6_candidates_per_kernel_hist.svg
p6_runtime_fragmentation_waterfall.svg
```

---

## P7：batch-major native runtime rewiring

### 目标

把 full-system runtime 从 per-row/per-candidate launch 改成 batch-major grouped execution。P7 不能只复用 v9.2.78 local kernel；必须在 P6 full train-stream path 中测。

### 必须记录

```text
runtime_candidate_id
bucket_strategy
runtime_bucket_used
persistent_workspace_used
basis_norm_bucketed
W2_delta_bucketed
bridge_score_inside_kernel
accept_bit_inside_kernel
stable_accept_inside_kernel
kernel_count_before
kernel_count_after
sync_count_before
sync_count_after
allocation_count_before
allocation_count_after
avg_candidates_per_kernel_before
avg_candidates_per_kernel_after
effective_candidates_per_launch
native_time_ms_q90
full_system_step_ratio_q90
memory_ratio
accept_disagreement_count
score_quantized_disagreement_count
rank_disagreement_count
```

### 判断标准

P7 pass:

```text
runtime_bucket_used = 1
persistent_workspace_used = 1
basis_norm_bucketed = 1
W2_delta_bucketed = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
stable_accept_inside_kernel = 1
accept_disagreement_count = 0
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
avg_candidates_per_kernel_after >= 8
kernel_count_after <= 0.50 * kernel_count_before
sync_count_after <= 0.50 * sync_count_before
allocation_count_after <= 0.10 * allocation_count_before
```

System diagnostic:

$$
StepRatio_{q90}\leq2.00.
$$

Full system:

$$
StepRatio_{q90}\leq1.50.
$$

### 可视化

```text
p7_batch_major_runtime_before_after.svg
p7_kernel_sync_allocation_reduction.svg
p7_step_ratio_by_bucket_strategy.svg
p7_native_runtime_pareto.svg
```

---

## P8：system-legal exact-signal controller v13

### 目标

组合 P1-P7 的 survivor，建立 official system controller。P8 是本轮能否真正推进到 LDO/LSO 的核心 gate。

### 必须记录

```text
controller_id
decision_repair_candidate_id
runtime_candidate_id
accept_contract_id
native_kernel_id
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
score_quantized_disagreement_count
rank_disagreement_count
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
stable_accept_full_row_materialization_present
outcome_labels_present
secondary_outcome_delta_fields_present
materialized_system_path
native_bucket_kernel_used
stable_accept_contract_used
bridge_score_inside_kernel
accept_bit_inside_kernel
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

P8 pass:

```text
official_eligible = 1
system_legal_controller_pass = 1
materialized_system_path = 1
stable_accept_full_row_materialization_present = 1
outcome_labels_present = 1
accept_disagreement_count = 0
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
native_bucket_kernel_used = 1
stable_accept_contract_used = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
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

Decision gate:

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

System gate:

$$
Agreement_{\text{reference}}\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Support balance:

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

### 可视化

```text
p8_system_controller_cost_quality_frontier.svg
p8_decision_repair_vs_runtime_pareto.svg
p8_stable_accept_overlap_matrix.svg
p8_family_strata_balance.svg
```

---

## P9：leave-dataset-out / leave-stratum-out

### 目标

只有 P8 pass 后打开。证明 repaired StableAccept controller 不是 pooled calibration artifact，也不是 dataset-specific route。

### 设置

Leave-dataset-out:

```text
calibrate on MNIST + Fashion, evaluate KMNIST
calibrate on MNIST + KMNIST, evaluate Fashion
calibrate on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out:

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

LDO pass:

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two held-out datasets:

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

LSO pass:

At least $70\%$ held-out strata task-safe and:

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

### 可视化

```text
p9_leave_dataset_out_matrix.svg
p9_leave_stratum_out_matrix.svg
p9_dataset_tuning_audit.svg
p9_leaveout_failure_modes.svg
```

---

## P10：official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。只有 P8/P9 pass 后才允许 official。

### 设置

```text
base = R2 repaired base checkpoint
controller_id = best P9 survivor
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
  ShuffledStableAcceptScore
  ShuffledStableAcceptRank
  ShuffledStableAcceptTiePolicy
  ShuffledOutcomeLabel
  ShuffledRiskSupportFilter
  ShuffledNativeBucketKernel
  ShuffledBridgeScore
  ShuffledAcceptBit
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

Paired replay pass:

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety:

$$
Acc_{\text{each slice,Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Shuffle controls must fail:

```text
ShuffledTrueBranchDelta = fail
ShuffledPF5Prefilter = fail
ShuffledCandidatePayload = fail
ShuffledStableAcceptScore = fail
ShuffledStableAcceptRank = fail
ShuffledStableAcceptTiePolicy = fail
ShuffledOutcomeLabel = fail
ShuffledRiskSupportFilter = fail
ShuffledNativeBucketKernel = fail
ShuffledBridgeScore = fail
ShuffledAcceptBit = fail
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

System:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p10_official_paired_replay_pareto.svg
p10_macro_beat_rate.svg
p10_signal_stratum_win_matrix.svg
p10_shuffle_control_matrix.svg
p10_system_gate_distribution.svg
```

---

## P11：short-run scout

### 目标

如果 P10 paired replay pass，验证 local causal advantage 能否进入连续训练。

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

Task safety:

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Control superiority:

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

## P12：full run / robustness / strong baseline

### 目标

只有 P11 pass 后打开。验证 functional advantage 不是 local replay artifact。

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
  ShuffledStableAcceptScore
  ShuffledStableAcceptTiePolicy
  ShuffledOutcomeLabel
  ShuffledRiskSupportFilter
  ShuffledNativeBucketKernel
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
Functional not explained by shuffled controller/kernel/payload
```

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9281.csv
p0_v9280_boundary_reproduction.csv
p1_stable_quantized_rank_bitexact_repair.csv
p2_candidate_set_row_lifecycle_continuity_audit.csv
p3_outcome_label_secondary_delta_materialization.csv
p4_decision_failure_autopsy.csv
p5_dataset_agnostic_stable_accept_decision_repair.csv
p6_full_system_runtime_fragmentation_autopsy.csv
p7_batch_major_native_runtime_rewiring.csv
p8_system_legal_exact_signal_controller_v13.csv
p9_leave_dataset_and_stratum_out.csv
p10_official_paired_replay.csv
p11_short_run_functional_validation.csv
p12_full_run_robustness_strong_baseline.csv

stable_quantized_rank_trace_v9281.csv
candidate_set_diff_trace_v9281.csv
outcome_secondary_delta_trace_v9281.csv
bad_accepted_failure_trace_v9281.csv
decision_repair_trace_v9281.csv
runtime_fragmentation_trace_v9281.csv
batch_major_runtime_trace_v9281.csv
system_controller_trace_v9281.csv
leaveout_trace_v9281.csv
paired_replay_branch_trace_v9281.csv
short_run_trace_v9281.csv

route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

Failure taxonomy:

```text
F1_contract_violation
F2_v9280_boundary_unstable
F3_dataset_tuning_detected
F4_payload_binding_regression
F5_stable_quantized_rank_repair_fail
F6_accept_disagreement_reappears
F7_candidate_set_drift_unexplained
F8_candidate_row_identity_mismatch
F9_outcome_label_primary_fail
F10_secondary_outcome_delta_missing
F11_bad_accepted_autopsy_incomplete
F12_no_dataset_agnostic_decision_repair
F13_decision_repair_precision_fail
F14_decision_repair_coverage_fail
F15_decision_repair_bad_event_fail
F16_decision_repair_null_rate_fail
F17_decision_repair_lcb_ucb_fail
F18_support_balance_fail
F19_runtime_fragmentation_autopsy_incomplete
F20_batch_major_runtime_not_integrated
F21_kernel_sync_reduction_lost
F22_avg_candidates_per_kernel_too_low
F23_full_system_step_ratio_fail
F24_memory_ratio_fail
F25_source_gap_or_formula_proxy_used
F26_projection_used
F27_system_controller_precision_fail
F28_system_controller_coverage_fail
F29_system_controller_bad_event_fail
F30_system_controller_null_rate_fail
F31_system_controller_lcb_ucb_fail
F32_leave_dataset_out_fail
F33_leave_stratum_out_fail
F34_paired_replay_control_equivalent
F35_shuffle_control_pass
F36_functional_lr_equivalent
F37_short_run_task_drop
F38_full_run_no_macro_hard_stratum_gain
F39_strong_baseline_explains_gain
F40_robustness_fail
F41_external_not_ready
F42_fake_or_proxy_violation
F43_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.80 boundary reproduced.

R2-StableQuantizedRankContractPass:
  score quantized / rank / accept fields are bit-exact.

R3-CandidateLifecycleExplained:
  candidate count drift and row identity are explained.

R4-OutcomeSecondaryDeltaMaterialized:
  primary and secondary outcome fields are materialized.

R5-DecisionFailureAttributed:
  bad accepted rows have >=0.90 attribution.

R6-StableAcceptDecisionRepairPass:
  dataset-agnostic repaired decision frontier passes heldout.

R7-RuntimeFragmentationAttributed:
  full-system kernel/sync fragmentation has actionable attribution.

R8-BatchMajorNativeRuntimePass:
  batch-major native runtime reaches step_ratio_q90 <=2.00 and improves kernel/sync.

R9-SystemLegalExactSignalControllerPass:
  system controller passes decision + compute gates.

R10-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R11-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R12-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R13-ShortRunFunctionalPass:
  short-run task-safe mechanism gain.

R14-FullFunctionalPass:
  full run task / geometry / system / control gates pass.

R15-PayloadBindingRegression:
  payload binding no longer reproduces.

R16-StableQuantizedRankStillMismatch:
  quantized/rank contract still inconsistent.

R17-OutcomeLabelContractFail:
  primary or secondary outcome labels missing/ambiguous.

R18-DecisionRegionUnsafe:
  repaired StableAccept cannot achieve precision/bad-event gates.

R19-FullSystemRuntimeStillFragmented:
  native kernel is wired but full-system runtime remains per-row/per-candidate fragmented.

R20-SystemStillTooExpensive:
  decision gates pass but step_ratio_q90 remains >1.50.

R21-ComputePassButLeaveoutFail:
  system controller overfits pooled calibration.

R22-ComputePassButPairedReplayFail:
  controller is system-legal but not causally superior to controls.

R23-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` must record:

```text
route
v9280_boundary_pass
dataset_tuning_detected
base_candidate
reference_controller_id
stable_controller_id
decision_repair_candidate_id
runtime_candidate_id

payload_binding_contract_pass
candidate_tensor_payload_missing_count
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
functional_update_payload_missing_count

stable_quantized_rank_repair_pass
score_quantized_disagreement_count
rank_disagreement_count
accept_disagreement_count

candidate_lifecycle_pass
old_candidate_count
new_candidate_count
candidate_jaccard
new_only_bad_event_rate
candidate_count_change_explained

outcome_label_primary_pass
outcome_secondary_delta_pass
missing_primary_label_count
missing_secondary_delta_count
ambiguous_label_count
label_source

decision_failure_autopsy_pass
primary_bad_accepted_failure_mode
bad_accepted_attribution_fraction
precision_failure_attribution_fraction

stable_accept_decision_repair_pass
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

runtime_fragmentation_autopsy_pass
dominant_runtime_fragmentation_mode
kernel_count_before
kernel_count_after
sync_count_before
sync_count_after
allocation_count_before
allocation_count_after
avg_candidates_per_kernel_after
effective_candidates_per_launch

batch_major_runtime_pass
full_system_step_ratio_q90
memory_ratio
new_full_system_step_ratio_measured
old_step_ratio_reused_as_measurement

system_legal_controller_pass
official_eligible

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
success_v9281_strict_purekan_functional
success_v9281_full_functional
success_v9281_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 stable quantized/rank bit-exact repair
  P2 candidate-set row-lifecycle audit
  P3 outcome secondary delta materialization
  P4 decision failure autopsy
  P6 runtime fragmentation autopsy

Batch 2:
  P5 dataset-agnostic decision repair
  P7 batch-major native runtime rewiring

Batch 3:
  P8 system-legal controller
  P9 leave-dataset-out / leave-stratum-out scout

Batch 4:
  official P9 LDO/LSO
  official P10 paired replay

Batch 5:
  P11 short-run if P10 passes
  P12 full run / robustness / strong baseline only if P11 passes
```

Gate rule:

```text
P5 cannot pass unless:
  P1 pass
  P3 primary outcome labels pass
  P4 decision failure autopsy pass

P7 cannot pass unless:
  P1 pass
  P6 runtime fragmentation autopsy pass

P8 cannot pass unless:
  P1 stable quantized/rank repair pass
  P3 primary outcome labels pass
  P5 decision repair pass
  P7 runtime pass
  step_ratio_q90 <= 1.50
  accept_disagreement_count = 0
  score_quantized_disagreement_count = 0
  rank_disagreement_count = 0
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

P9/P10 diagnostic rows may be measured before all gates finish,
but official status requires:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  stable accept controller calibrated
  PF5 runtime selector pass
  payload binding pass
  stable accept materialization pass
  outcome label materialization pass
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
v9.2.80 boundary reproduced
stable quantized/rank mismatch repaired or fully attributed
candidate-set drift explained
outcome label primary/secondary materialization measured
decision failure autopsy completed
runtime fragmentation autopsy completed
decision repair measured
batch-major runtime measured
system controller measured
no fake/proxy/offload/loss/teacher violation
```

## Stable contract success

```text
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
accept_disagreement_count = 0
missing_stable_accept_fields = none
```

## Decision success

```text
Stable contract success
+
primary outcome labels pass
+
heldout decision gates pass
+
support balance pass
+
dataset_name_used = 0
```

## Runtime success

```text
Stable contract success
+
native kernel used in P6
+
batch-major runtime pass
+
step_ratio_q90 <= 1.50
+
memory_ratio <= 1.05
```

## System success

```text
Decision success
+
Runtime success
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
1. v9.2.80 boundary cannot be reproduced；
2. payload binding regresses；
3. score quantized / rank mismatch cannot be repaired；
4. accept disagreement reappears；
5. candidate set drift cannot be explained；
6. primary outcome labels become missing/ambiguous；
7. secondary deltas remain missing and paired replay is requested；
8. bad accepted rows cannot be attributed；
9. no dataset-agnostic decision repair reaches precision/bad-event gates；
10. repair reaches decision gate only by dataset-specific threshold；
11. runtime fragmentation cannot be attributed；
12. batch-major runtime still launches per candidate；
13. avg candidates per kernel remains <2；
14. kernel/sync count remains worse than baseline；
15. step_ratio_q90 remains >1.50；
16. memory ratio >1.05；
17. source gap / formula proxy / projection used；
18. leave-dataset-out fails；
19. leave-stratum-out fails；
20. paired replay remains control-equivalent；
21. shuffle controls pass；
22. short-run task drops；
23. full run gives no macro / hard-stratum / geometry gain；
24. functional breaks system gate；
25. gains are explained by QuadraticFeatureMLP；
26. any teacher/loss/fake/proxy/offload/projection-as-pass violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：P8 system pass + P9/P10 pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full run and robustness。

## Case B：quantized/rank 修复后 decision 仍失败

必须声明：

```text
native stable accept contract is bit-exact, but AC2Q2 accepted region is not safe under same-run outcome labels.
```

下一步修 accept/risk/support decision，不再追 quantization。

## Case C：decision repair pass but runtime still slow

必须声明：

```text
controller is decision-legal, but full-system native runtime remains above envelope.
```

下一步基于 P6/P7 waterfall 修 batch-major launch/sync，而不是重调 controller。

## Case D：runtime pass but decision repair fails

必须声明：

```text
system path is efficient, but StableAccept decision region does not select safe-good events reliably.
```

下一步修 outcome-grounded accept score / risk-support filter，而不是继续 kernelization。

## Case E：P8 pass but LDO/LSO fail

必须声明：

```text
controller overfits pooled calibration or signal strata.
```

不能通过 dataset-specific tuning 写成功；下一步修 dataset-agnostic support/family reliability。

## Case F：P8/P9 pass but paired replay fail

必须声明：

```text
controller is system-legal but not causally superior to matched controls.
```

下一步回到 functional event/value target，而不是继续 runtime kernelization。

---

# Part XII. 最终建议

v9.2.81 的一句话策略是：

$$
\boxed{
\text{不要只修 18 个 quantized mismatch；要同时修 safe-good decision region 和 full-system batch-major runtime。}
}
$$

当前最关键的问题不是：

```text
full-row materializer 是否存在；
primary outcome label 是否存在；
native kernel 是否能接入 P6；
local native q90 是否有 reduction；
stable accept disagreement 是否为 0；
old artifact join 是否可用；
C3/T2/C4/E2 controller 是否要重调；
Fashion/KMNIST/MNIST 谁更好。
```

而是：

```text
1. 18 个 quantized disagreement 与 2 个 rank disagreement 的根因是什么？
2. candidate_count 从 2493 到 2876 是否改变了 accepted population？
3. 468 accepted rows 中为什么约 142 个是 bad-event？
4. bad accepted rows 是否来自 risk/support underestimation、candidate drift、label definition，还是 stable-score miscalibration？
5. 是否存在 dataset-agnostic risk/support repair，让 precision 从 0.513 回到 >=0.75，同时 bad-event 从 0.303 降到 <=0.05？
6. secondary outcome deltas 能否补齐到 downstream-ready？
7. P6 runtime 为什么出现 72576 kernel / 8064 sync？
8. avg_candidates_per_kernel 为什么只有 0.0396？
9. batch-major native runtime 能否把 step_ratio 从 2.213 降到 <=1.50？
10. system controller 是否 official eligible？
11. LDO/LSO 是否通过？
12. official paired replay 是否打过 AdamWParallel / bestLR？
```

v9.2.81 的结果将给出清晰分叉：

```text
if quantized/rank repaired and decision+runtime pass:
  open LDO/LSO and official paired replay.

if quantized/rank repaired but decision fails:
  AC2Q2 accepted region is unsafe; repair risk/support score.

if decision repaired but runtime fails:
  runtime fragmentation is primary; repair batch-major P6 wiring.

if runtime repaired but decision fails:
  stop kernel work; fix outcome-grounded accept score.

if P8 passes but P10 paired replay fails:
  system is legal, but functional causal advantage is insufficient.

if LDO/LSO fails:
  repair dataset-agnostic support/family reliability, not dataset-specific tuning.
```
