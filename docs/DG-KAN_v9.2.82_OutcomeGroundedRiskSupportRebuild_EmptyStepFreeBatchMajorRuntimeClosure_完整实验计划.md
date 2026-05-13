# DG-KAN v9.2.82 Outcome-Grounded Risk-Support Rebuild 与 Empty-Step-Free BatchMajor Runtime Closure 完整实验计划

> 本计划基于 v9.2.81 `Outcome-Grounded StableAccept Repair 与 Full-System BatchMajor Runtime Closure` 的真实执行结果制定。
> v9.2.81 的 terminal route 是：
>
> ```text
> route = R16-OutcomeGroundedStableAcceptRegionUnsafe
> base_candidate = LQ-t2-h256
> success_v9281_strict_purekan_functional = False
> success_v9281_full_functional = False
> success_v9281_external_ready = False
> ```
>
> v9.2.81 的关键事实是：
>
> ```text
> P1 stable quantized/rank repair:
>   score_quantized_disagreement_count_before = 18
>   rank_disagreement_count_before = 2
>   accept_disagreement_count_before = 0
>   score_quantized_disagreement_count_after = 0
>   rank_disagreement_count_after = 0
>   accept_disagreement_count_after = 0
>   fallback_row_count = 20
>   native_scoreq_emit_bitexact = 0
>
> P2 candidate lifecycle:
>   old_candidate_count = 2493
>   new_candidate_count = 2876
>   shared_candidate_count = 2123
>   old_only_candidate_count = 370
>   new_only_candidate_count = 753
>   candidate_jaccard = 0.6540357362908195
>   candidate_id_mismatch_count = 2120
>   payload_hash_mismatch_count = 468
>   candidate_lifecycle_audit_pass = 0
>
> P3 outcome labels:
>   missing_primary_label_count = 0
>   ambiguous_label_count = 0
>   safe_bad_exclusivity_violation_count = 0
>   outcome_label_primary_pass = 1
>   missing_secondary_delta_count = 14380
>   outcome_downstream_ready_pass = 0
>
> P4 decision autopsy:
>   heldout_accepted = 468
>   safe = 240
>   bad = 142
>   null = 57
>   bad_accepted_attribution_fraction = 1.0
>   dominant_failure_mode = FMA3-risk-ucb-underestimation
>   failure_counts = FMA3:70, FMA2:41, FMA5:20, FMA7:11
>
> P5 decision repair:
>   best_repair_candidate_id = DR2-BadLevelTailRiskFilter
>   precision_heldout = 0.63003663003663
>   coverage_heldout = 0.03009259259259259
>   bad_event_heldout = 0.18315018315018314
>   null_rate_heldout = 0.19413919413919414
>   decision_repair_pass = 0
>
> P6 runtime fragmentation:
>   step_count = 8064
>   active_step_count = 1412
>   zero_candidate_step_count = 6652
>   empty_step_kernel_fraction = 0.8249007936507936
>   dominant_runtime_fragmentation_mode = FR2-per-step-sync-empty-step-fixed-launch
>   kernel_count_after = 72576
>   sync_count_after = 8064
>   avg_candidates_per_kernel_after = 0.03962742504409171
>   runtime_fragmentation_autopsy_pass = 1
>
> P8 system boundary:
>   controller_id = C3Q2-v13-OutcomeGroundedStableAcceptRepair
>   decision_repair_candidate_id = DR2-BadLevelTailRiskFilter
>   runtime_candidate_id = RT0-v9280-full-system-native-reference
>   accept_contract_id = AC2Q2+QR5-borderline-exact-fallback
>   precision_heldout = 0.63003663003663
>   coverage_heldout = 0.03009259259259259
>   bad_event_heldout = 0.18315018315018314
>   null_rate_heldout = 0.19413919413919414
>   step_ratio_q90 = 2.213009156635521
>   secondary_outcome_delta_fields_present = 0
>   decision_gate_pass = 0
>   runtime_gate_pass = 0
>   official_eligible = 0
>   system_legal_controller_pass = 0
>   reason = decision_repair_failed_and_batch_major_runtime_not_materialized
> ```
>
> v9.2.82 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.81 已经不是“字段缺失”或“accept contract mismatch”问题；现在是真实的 decision-region failure 和 runtime-fragmentation failure。}
> }
> $$
>
> 因此 v9.2.82 不应该继续只修 `score_q`、`rank`、fallback row，也不应该继续只写 batch-major diagnostic estimate。
> 本轮必须并行推进两条主线：
>
> ```text
> Decision side:
>   outcome-grounded risk/support sufficient statistics rebuild。
>
> Runtime side:
>   empty-step-free active-step scheduler + measured batch-major native runtime。
> ```
>
> 如果 v9.2.82 仍无法找到 dataset-agnostic repair 让 precision / bad-event 过线，则应判定：
>
> ```text
> AC2Q2 / StableAccept accepted region is unsafe under same-run outcome labels.
> ```
>
> 这时不能再继续对 StableAccept 小修小补，应回到 outcome-grounded accept score 或 observable risk primitive 设计。

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

v9.2.82 新增硬约束：

```text
不能只修 QR5 fallback / score_q / rank 就声明 official；
不能使用 dataset name 调 threshold、risk filter、tie policy、bucket route；
不能使用 old artifact join label；
不能使用 event_family 聚合 label 代替 row-level outcome；
不能复用 v9.2.78 local q90；
不能复用 v9.2.80/v9.2.81 old step ratio 作为新 batch-major measurement；
不能把 RT1 diagnostic estimate 写成 measured runtime；
不能把 primary outcome label pass 写成 paired replay ready；
不能在 secondary deltas missing 时打开 official paired replay；
不能把 support count pass 写成 precision / bad-event pass；
不能用 source-measured gap / formula proxy / projection 进入 official。
```

允许使用：

```text
same-run primary and secondary outcome labels；
AC2Q2 stable accept as one feature, not sufficient condition；
dataset-agnostic risk/support sufficient statistics；
bad_ucb / null_ucb / support_lcb / family_lcb / horizon_tail_risk；
candidate-origin drift diagnostics；
monotone risk-support score；
two-stage stable-accept + exact safety confirmation；
active-step compaction；
empty-step skip；
batch-major grouped native CUDA/Triton runtime；
persistent workspace；
post-step audit subset。
```

---

# Part I. 对 v9.2.81 的独立判断

## 1. v9.2.81 没有达到目标

v9.2.81 没有达到 strict PureKAN functional success，也没有 system-legal controller success。直接原因是：

```text
official_eligible = 0
system_legal_controller_pass = 0
decision_gate_pass = 0
runtime_gate_pass = 0
step_ratio_q90 = 2.213009156635521
```

P9 leave-dataset-out / leave-stratum-out、P10 official paired replay、P11 short-run、P12 full-run 均被正确 gate-blocked。
这不是保守过头，而是当前 best row 同时没有过 decision gate 和 runtime gate。

## 2. v9.2.81 的真实进展

v9.2.81 有三项真实进展。

第一，stable quantized/rank contract 的表层 mismatch 被关闭。score quantized disagreement 从 `18` 到 `0`，rank disagreement 从 `2` 到 `0`，accept disagreement 保持 `0`。这说明 v9.2.80 的 `StableAcceptNativeQuantizationMismatch` 不再是主 blocker。

但这个修复是 QR5 borderline exact fallback，不是 native kernel 直接 emit bit-exact integer `score_q`。所以它只能解除 official audit mismatch，不能说明 native quantization kernel 已经根治。

第二，decision failure 不再是黑箱。P4 autopsy 显示 468 个 heldout accepted rows 中：

```text
safe = 240
bad = 142
null = 57
```

bad accepted attribution fraction = `1.0`，主失败模式是：

```text
FMA3-risk-ucb-underestimation = 70
FMA2-candidate-set-drift = 41
FMA5-null-bad-conflict = 20
FMA7-horizon-tail-risk = 11
```

这说明 bad-event contamination 不是随机噪声，而是风险估计和 candidate lifecycle 机制性失败。

第三，runtime failure 也不再是黑箱。P6 明确显示：

```text
8064 steps
1412 active steps
6652 zero-candidate steps
empty-step kernel fraction = 0.824901
kernel_count_after = 72576
sync_count_after = 8064
avg_candidates_per_kernel_after = 0.0396
```

也就是说，当前 full-system runtime 不是 native arithmetic 慢，而是大量 zero-candidate steps 仍然固定 launch/sync，导致 batch-major advantage 完全丢失。

## 3. v9.2.81 的真实失败

v9.2.81 的失败有两个主轴。

第一，decision repair 失败。最好的非 posthoc repair 是 `DR2-BadLevelTailRiskFilter`：

```text
precision = 0.630037
coverage = 0.030093
bad-event = 0.183150
null-rate = 0.194139
```

相比 v9.2.80 的 precision `0.5128`、bad-event `0.3034` 有进步，但离 official gate 仍很远：

$$
Precision: 0.6300 \rightarrow 0.75,
$$

$$
BadEvent: 0.1832 \rightarrow 0.05,
$$

$$
NullRate: 0.1941 \rightarrow 0.15.
$$

这个幅度不是修一两个 quantized mismatch 能解决的。

第二，runtime repair 没有 materialize。RT1 只是 `ActiveStepBatchMajorGroupingDiagnostic`，仍是 diagnostic estimate，不是 measured runtime。当前 measured runtime 仍是 RT0 reference：

$$
StepRatio_{q90}=2.2130>1.50.
$$

## 4. 当前 blocker 的本质

当前 blocker 应写成：

$$
\boxed{
\text{StableAccept 的 outcome-grounded accept region 不安全；P6 full-system runtime 被 empty-step fixed launch/sync 碎片化。}
}
$$

不是：

```text
materializer missing；
primary labels missing；
accept disagreement still nonzero；
quantized/rank mismatch still dominant；
native kernel never entered P6；
quantile-tail unknown；
basis_norm unknown；
selected-feature missing。
```

这些问题都已经推进过。现在的问题更深：

```text
1. StableAccept score/rank 不是 safe-good sufficient statistic；
2. risk_ucb 低估 bad-event；
3. candidate lifecycle drift 改变了 accepted population；
4. null/bad 混淆仍存在；
5. horizon-tail risk 没被 official accept region 吸收；
6. runtime 在 zero-candidate steps 上仍固定 launch/sync；
7. batch-major native runtime 未在 P6 measured path materialize。
```

---

# Part II. v9.2.82 总体目标

v9.2.82 的总体目标是：

$$
\boxed{
\text{把 StableAccept 从单一 accept region 改为 outcome-grounded risk-support controller，并把 runtime 改为 empty-step-free batch-major measured path。}
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
risk-support feature rebuild measured
bad accepted rows attribution stable across heldout
candidate lifecycle canonicalization completed or failure explicitly isolated
secondary outcome deltas materialized
empty-step fixed launch eliminated
batch-major runtime measured, not diagnostic
step_ratio_q90 <= 2.00
```

本轮必须回答十二个问题：

```text
Q1:
  v9.2.81 boundary 是否稳定复现？

Q2:
  QR5 fallback 清零 score_q/rank 后，decision failure 是否仍存在？
  如果仍存在，说明 quantization 不是主因。

Q3:
  142 个 bad accepted rows 是否稳定由 FMA3/FMA2/FMA5/FMA7 主导？

Q4:
  FMA3 risk_ucb_underestimation 的具体机制是什么？
  是 bad UCB 校准不足、risk feature 不够、family/horizon context 缺失，还是 outcome horizon mismatch？

Q5:
  candidate lifecycle drift 对 bad-event contamination 贡献多大？
  new-only candidates 是否显著更 bad？

Q6:
  null/bad conflict 能否通过 null UCB 与 bad UCB 解耦解决？

Q7:
  horizon-tail risk 是否必须进入 official risk-support score？

Q8:
  是否存在 dataset-agnostic monotone repair，使 precision/bad-event/coverage/null 全部过 gate？

Q9:
  secondary outcome deltas 缺失是否阻止 paired replay readiness？
  能否补齐到 0 missing？

Q10:
  P6 empty-step fixed launch 是否能被 active-step compaction 消除？

Q11:
  batch-major grouped runtime 能否把 avg candidates/kernel 从 0.0396 提到 >=8？

Q12:
  system pass 后 LDO/LSO 和 official paired replay 是否通过？
```

---

# Part III. 核心假设

## H1：QR5 修复不是主 blocker；真正 blocker 是 accepted region unsafe

H1 认为 quantized/rank mismatch 清零后，decision gate 仍会失败，因为 accept region 本身混入 bad events。

H1 成立标准：

```text
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
accept_disagreement_count = 0
precision_heldout < 0.75
or bad_event_heldout > 0.05
```

若 H1 成立，后续不能继续把主要精力放在 quantization/tie policy 上，而应修 outcome-grounded risk/support score。

H1 失败标准：

bit-exact native integer emit 替代 QR5 后，accepted region 大幅改变，并让 precision/bad-event 过 gate。若如此，route 应转为：

```text
R2-NativeQuantizationWasPrimary
```

## H2：bad-event contamination 由可观测的 risk/support/candidate drift 机制解释

H2 成立标准：

```text
bad_accepted_attribution_fraction >= 0.95
top failure modes explain >= 0.90 of bad accepted rows
FMA3/FMA2/FMA5/FMA7 attribution stable across calibration/heldout
```

H2 失败标准：

bad accepted rows 无法稳定归因，或 attribution 在 calibration/heldout 完全不同。若 H2 失败，不能继续设计 repair filter，应先修 outcome labeling 或 failure taxonomy。

## H3：dataset-agnostic risk/support sufficient statistics 可以恢复 safe-good frontier

H3 认为 StableAccept score 需要增加 risk/support constraints：

$$
Accept(e)=
StableAccept(e)
\land
RiskUCB(e)\leq \tau_b
\land
NullUCB(e)\leq \tau_n
\land
SupportLCB(e)\geq \tau_s
\land
FamilyLCB(e)\geq \tau_f
\land
HorizonTailRisk(e)\leq \tau_h.
$$

所有阈值只在 calibration split 上冻结，不允许 dataset-specific branch。

H3 成立标准：

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

H3 失败标准：

所有 dataset-agnostic candidates 要么 coverage <0.03，要么 bad-event >0.05，要么 precision <0.75。
若 H3 失败，必须停止 StableAccept patching，进入 outcome-grounded accept primitive redesign。

## H4：candidate lifecycle drift 是 decision failure 的重要但不一定唯一来源

H4 成立标准：

```text
candidate_jaccard < 0.90
new_only_bad_event_rate > shared_bad_event_rate
new_only_accept_share nontrivial
candidate_id_mismatch or payload_hash_mismatch contributes to bad accepted rows
```

H4 失败标准：

candidate lifecycle canonicalized 后 decision failure 仍基本不变。若如此，candidate drift 是需要修的契约问题，但不是 decision failure 主因。

## H5：runtime failure 主要来自 empty-step fixed launch/sync，而不是 native arithmetic

H5 成立标准：

```text
empty_step_kernel_fraction >= 0.80
zero_candidate_step_count / step_count >= 0.80
active-step compaction reduces kernel_count and sync_count by >= 50%
avg_candidates_per_kernel_after >= 8 after grouping
```

H5 失败标准：

empty-step skip 后 step ratio 仍 >1.50，且 waterfall 显示 arithmetic / update payload / bridge lookup dominates。若 H5 失败，才进入 deeper native arithmetic optimization。

## H6：如果 decision 和 runtime 过线但 paired replay fail，blocker 才回到 functional value target

H6 成立标准：

```text
P8 system controller pass = 1
P9 leave-out measured
P10 paired replay measured
P10 fails vs AdamWParallel / bestLR
```

H6 失败标准：

P8 未过时提前讨论 functional value target。这会混淆 system legality 与 causal advantage。

---

# Part IV. Required data contracts

## 1. Candidate lifecycle contract

每个 candidate row 必须记录：

```text
event_id
candidate_id
candidate_schema_version
payload_hash
stable_accept_contract_id
candidate_origin_tag
old_new_candidate_status
candidate_family_id
bucket_id
horizon
step
dataset
seed
```

官方 pass：

```text
event_id_mismatch_count = 0
candidate_id_mismatch_count = 0 or fully explained by schema migration
payload_hash_mismatch_count = 0 or isolated from official candidate set
candidate_count_change_explained = 1
```

## 2. Outcome contract

每个 candidate row 必须记录 primary labels：

```text
safe_good_label
bad_event_label
null_event_label
task_safe_label
useful_label
```

并补齐 secondary deltas：

```text
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
real_beats_adamwparallel
real_beats_bestlr
```

官方 downstream-ready pass：

```text
missing_primary_label_count = 0
ambiguous_label_count = 0
safe_bad_exclusivity_violation_count = 0
missing_secondary_delta_count = 0
```

## 3. Risk-support feature contract

每个 row 必须记录：

```text
stable_score_q
stable_rank
score_margin
bad_ucb
bad_lcb
null_ucb
null_lcb
support_lcb
support_count
family_lcb
family_support_count
horizon_tail_risk
candidate_origin_risk
risk_residual_score
bad_level_tail_risk
risk_calibration_bin
```

这些只能来自 train-stream / calibration split 可得信息，不得使用 heldout/test label at commit time。

## 4. Runtime contract

每个 timed step 必须记录：

```text
step_id
candidate_count_in_step
active_step
zero_candidate_step
kernel_count
sync_count
allocation_count
effective_candidates_per_launch
avg_candidates_per_kernel
selector_time_ms
candidate_pack_time_ms
native_kernel_time_ms
bridge_lookup_time_ms
stable_accept_time_ms
update_payload_time_ms
payload_apply_time_ms
audit_outside_timed_ms
total_step_time_ms
```

官方 runtime pass：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

---

# Part V. Candidate designs

## 1. Stable quantized/rank candidates

### QR0：v9.2.81 QR5 fallback reference

Expected diagnostic pass, not native integer emit pass.

### QR1：NativeEmitInt64ScoreQ

Native kernel directly emits `int64 score_q` using exactly the same quantization mode as reference.

### QR2：ReferenceUsesNativeScoreQ

Reference audit uses native-emitted `score_q` as canonical row field to avoid double-rounding drift.

### QR3：ExactBorderlineFallbackOnly

Fallback only for rows where:

$$
|score_q^{ref}-score_q^{native}|>0
$$

or rank tie boundary is detected.

### QR4：ScoreQCanonicalizationAudit

Records whether score_q mismatch affects decision metrics. This cannot be official by itself.

## 2. Candidate lifecycle candidates

### CL0：v9.2.81 candidate lifecycle reference

Expected fail.

### CL1：SchemaVersionedCandidateID

Candidate ID becomes:

$$
candidate\_id = hash(event\_id, family, horizon, bucket, payload\_hash, schema\_version).
$$

### CL2：ContentAddressedPayloadHash

Payload hash is based on true candidate tensor / branch logits / update payload content, not row order.

### CL3：OldNewCandidateDriftAttribution

Separates shared / old-only / new-only decision metrics.

### CL4：OfficialCandidateSetFreeze

Freeze candidate generator before decision repair; all repair candidates run on same candidate set.

## 3. Outcome candidates

### OUT0：v9.2.81 primary labels only

Expected downstream not-ready.

### OUT1：SecondaryDeltaMaterializer

Materialize CEp99/margin/ECE/NLL/curvature/real-beats for all candidate rows.

### OUT2：OutcomeConsistencyAuditV2

Check label exclusivity and horizon consistency.

### OUT3：ControlMatchedOutcomeMaterializer

For accepted rows and a matched sample of rejected rows, record RealFunctional / AdamWParallel / bestLR / NoOp / Random branches.

## 4. Decision-region repair candidates

### DR0：v9.2.81 DR2 reference

Expected fail.

### DR1：BitExactStableAcceptOnly

Only QR/native integer repair; no decision repair. Tests whether QR mismatch was primary.

### DR2：BadUCBRecalibrated

Rebuild bad UCB with calibration reliability correction:

$$
BadUCB'(e)=BadMean(e)+z_\alpha \sqrt{\frac{BadMean(e)(1-BadMean(e))}{n(e)+\epsilon}}+\lambda_{family}RiskResidual(e).
$$

### DR3：RiskResidualModel

Predict whether current BadUCB underestimated actual bad event:

$$
RiskResidual(e)=P(BadEvent=1 \mid BadUCB(e), family, horizon, score\_margin, candidate\_origin).
$$

### DR4：NullBadDecoupledGate

Use separate null and bad gates:

$$
BadUCB'(e)\leq\tau_b,
$$

$$
NullUCB'(e)\leq\tau_n.
$$

### DR5：FamilyHorizonReliabilityGate

Add:

$$
FamilyLCB(e)\geq\tau_f,
$$

$$
HorizonTailRisk(e)\leq\tau_h.
$$

### DR6：CandidateOriginDriftPenalty

Penalize new-only / drift-prone candidate origins:

$$
Score'(e)=Score(e)-\lambda_d DriftRisk(e).
$$

### DR7：RiskSupportMonotoneScore

Monotone composite:

$$
S(e)
=
a_1 StableScore_q(e)
-a_2 BadUCB'(e)
-a_3 NullUCB'(e)
+a_4 SupportLCB(e)
+a_5 FamilyLCB(e)
-a_6 HorizonTailRisk(e)
-a_7 DriftRisk(e).
$$

Constraints:

```text
a_i >= 0
weights selected only on calibration split
no dataset-specific branch
```

### DR8：TwoStageStableThenExactSafety

Stage 1 stable accept; Stage 2 exact safety confirmation for rows whose risk margin is small.

### DR9：ObservableRiskPrimitiveCandidate

If DR1-DR8 fail, extract a new observable risk primitive from same-run outcomes; this is not a patch to StableAccept but a route pivot.

## 5. Runtime candidates

### RT0：v9.2.81 full-system native reference

Expected fail:

```text
step_ratio_q90 = 2.213009
kernel_count_after = 72576
sync_count_after = 8064
avg_candidates_per_kernel_after = 0.0396
```

### RT1：ActiveStepCompaction

Skip all zero-candidate steps in controller kernel launch path.

### RT2：EmptyStepNoSync

Do not synchronize or launch fixed kernels for empty steps; record empty-step audit outside timed path.

### RT3：BatchMajorCandidateTableV2

Build a per-run candidate table grouped by:

```text
active_step
bucket_id
family_id
horizon
kernel_shape
```

### RT4：SingleLaunchPerActiveBucket

One native kernel per active bucket per step.

### RT5：PersistentWorkspaceReuseV2

Reuse candidate / basis_norm / W2_delta / bridge / accept buffers across active steps.

### RT6：CompactBridgeRiskLookupInsideKernel

Move bad/null/support/family/horizon lookup inside grouped kernel.

### RT7：HybridEmptyStepFreeBatchMajorNative

Best measured combination of RT1-RT6.

---

# Part VI. 实验阶段

## P0：v9.2.81 boundary reproduction

### 目标

确认 v9.2.81 boundary 稳定，不在不可复现 artifact 上继续。

### 必须记录

```text
route
source_route_v9281
candidate_count
event_count
score_quantized_disagreement_count
rank_disagreement_count
accept_disagreement_count
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
best_repair_candidate_id
controller_step_ratio_q90
kernel_count_after
sync_count_after
avg_candidates_per_kernel_after
empty_step_kernel_fraction
official_eligible
system_legal_controller_pass
fake_data_used
proxy_row_used
cpu_offload_used
```

### 判断标准

P0 pass：

```text
route = R16-OutcomeGroundedStableAcceptRegionUnsafe
full materializer = 1
primary outcome labels = 1
decision gate = 0
runtime gate = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9281_boundary_ladder.svg
p0_decision_runtime_failure_split.svg
p0_progress_v9280_to_v9281.svg
```

---

## P1：native score_q / rank contract closure

### 目标

确认 QR5 fallback 是否足够，或实现 native integer score_q bit-exact emit。P1 不能直接声明 official success；它只关闭 score/rank contract。

### 必须记录

```text
quantized_repair_id
native_scoreq_emit_bitexact
fallback_used
fallback_row_count
score_quantized_disagreement_count_before
score_quantized_disagreement_count_after
rank_disagreement_count_before
rank_disagreement_count_after
accept_disagreement_count_before
accept_disagreement_count_after
rounding_mode
score_scale
tie_key_policy
```

### 判断标准

P1 pass：

```text
score_quantized_disagreement_count_after = 0
rank_disagreement_count_after = 0
accept_disagreement_count_after = 0
```

Native stronger pass：

```text
native_scoreq_emit_bitexact = 1
fallback_row_count = 0
```

### 可视化

```text
p1_scoreq_rank_disagreement_before_after.svg
p1_fallback_rows_margin.svg
p1_native_vs_reference_scoreq.svg
```

---

## P2：candidate lifecycle canonicalization and drift audit

### 目标

解释 candidate set drift，并判断它对 bad-event contamination 的贡献。

### 必须记录

```text
old_candidate_count
new_candidate_count
shared_candidate_count
old_only_candidate_count
new_only_candidate_count
candidate_jaccard
candidate_id_mismatch_count
payload_hash_mismatch_count
old_only_precision
new_only_precision
shared_precision
old_only_bad_event_rate
new_only_bad_event_rate
shared_bad_event_rate
old_only_accept_share
new_only_accept_share
shared_accept_share
candidate_count_change_explained
```

### 判断标准

P2 pass：

```text
candidate_count_change_explained = 1
event_id_mismatch_count = 0
payload_hash_mismatch_count = 0 or isolated from official candidate set
candidate lifecycle failure contribution measured
```

### 可视化

```text
p2_candidate_set_venn.svg
p2_old_new_bad_event_rates.svg
p2_candidate_drift_by_family_horizon.svg
p2_payload_hash_mismatch_matrix.svg
```

---

## P3：secondary outcome delta materialization

### 目标

补齐 downstream readiness，不再让 paired replay/short-run 因 secondary deltas missing 被 blocked。

### 必须记录

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
outcome_horizon
outcome_branch
label_source
missing_primary_label_count
missing_secondary_delta_count
ambiguous_label_count
label_exclusivity_violation_count
```

### 判断标准

Primary pass：

```text
missing_primary_label_count = 0
ambiguous_label_count = 0
label_exclusivity_violation_count = 0
```

Downstream-ready pass：

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
p3_secondary_delta_missing_before_after.svg
p3_outcome_label_balance.svg
p3_safe_bad_null_overlap.svg
p3_metric_delta_distribution.svg
```

---

## P4：bad accepted rows risk/support autopsy v2

### 目标

对 v9.2.81 的 142 bad accepted rows 做更深层的 causal-style attribution。不能只写 FMA3 标签，必须知道 risk_ucb 为什么低估。

### 必须记录

```text
bad_accepted_event_id
bad_accepted_candidate_id
family_id
bucket_id
horizon
stable_score_q
stable_rank
score_margin
bad_ucb
bad_lcb
null_ucb
support_lcb
family_lcb
horizon_tail_risk
candidate_origin_tag
candidate_drift_status
outcome_horizon
failure_mode
failure_submode
failure_mode_fraction
risk_underestimate_amount
```

Failure submodes：

```text
FMA3a-risk_mean_underestimated
FMA3b-risk_uncertainty_too_narrow
FMA3c-family_context_missing
FMA3d-horizon_context_missing
FMA3e-candidate_origin_shift
FMA5a-null_bad_label_overlap
FMA5b-null_filter_too_weak
FMA7a-tail_threshold_miscalibrated
FMA7b-horizon_tail_context_missing
```

### 判断标准

P4 pass：

```text
bad_accepted_attribution_fraction >= 0.95
top failure modes explain >= 0.90 of bad accepted rows
risk/support repair features identified
```

### 可视化

```text
p4_bad_accepted_failure_modes.svg
p4_risk_underestimation_hist.svg
p4_bad_event_by_family_horizon_bucket.svg
p4_bad_ucb_vs_actual_bad_scatter.svg
```

---

## P5：outcome-grounded risk/support sufficient statistics factory

### 目标

构造新的 dataset-agnostic risk/support features。P5 不是 controller repair，而是 feature validity test。

### 必须记录

```text
feature_id
feature_family
AUC_bad_event
AUC_safe_good
corr_bad_event
corr_safe_good
calibration_ECE_bad
risk_bin_count
family_coverage
horizon_coverage
candidate_origin_coverage
uses_dataset_name
uses_outcome_at_commit
```

### 判断标准

P5 pass：

```text
at least one bad-risk feature AUC >= 0.75
or at least one safe-good feature AUC >= 0.75
uses_dataset_name = 0
uses_outcome_at_commit = 0
```

Stronger pass：

```text
bad-risk AUC >= 0.80
risk calibration ECE <= 0.05
```

### 可视化

```text
p5_risk_feature_auc.svg
p5_risk_calibration_curve.svg
p5_feature_family_pareto.svg
p5_safe_good_vs_bad_risk_scatter.svg
```

---

## P6：dataset-agnostic decision repair

### 目标

在 calibration split 上冻结 thresholds，在 heldout split 上验证是否恢复 official safe-good frontier。不能按 dataset 调参。

### 必须记录

```text
repair_candidate_id
repair_type
feature_set
thresholds
calibration_split_id
heldout_split_id
dataset_name_used
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

P6 pass：

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

### 可视化

```text
p6_decision_repair_frontier.svg
p6_precision_bad_event_coverage_pareto.svg
p6_calibration_to_heldout_drift.svg
p6_support_balance.svg
```

---

## P7：runtime empty-step elimination

### 目标

消除 P6 path 中的 zero-candidate fixed launch/sync。此阶段必须 measured，不得只估计。

### 必须记录

```text
runtime_candidate_id
step_count_before
active_step_count
zero_candidate_step_count
empty_step_kernel_fraction_before
empty_step_kernel_fraction_after
kernel_count_before
kernel_count_after
sync_count_before
sync_count_after
allocation_count_before
allocation_count_after
avg_candidates_per_kernel_before
avg_candidates_per_kernel_after
effective_candidates_per_launch
step_ratio_q90
memory_ratio
```

### 判断标准

P7 pass：

```text
empty_step_kernel_fraction_after <= 0.05
kernel_count_after <= 0.50 * kernel_count_before
sync_count_after <= 0.50 * sync_count_before
avg_candidates_per_kernel_after >= 2
```

Stronger pass：

```text
avg_candidates_per_kernel_after >= 8
step_ratio_q90 <= 2.00
```

### 可视化

```text
p7_empty_step_kernel_fraction_before_after.svg
p7_active_step_compaction.svg
p7_kernel_sync_reduction.svg
p7_candidates_per_kernel_hist.svg
```

---

## P8：measured batch-major native runtime

### 目标

把 active steps 内的 candidates 做 batch-major grouping，并在 P6 full-system measured path 中验证。

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
risk_support_lookup_inside_kernel
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

P8 pass：

```text
runtime_bucket_used = 1
persistent_workspace_used = 1
basis_norm_bucketed = 1
W2_delta_bucketed = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
stable_accept_inside_kernel = 1
risk_support_lookup_inside_kernel = 1
accept_disagreement_count = 0
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
avg_candidates_per_kernel_after >= 8
kernel_count_after <= 0.50 * kernel_count_before
sync_count_after <= 0.50 * sync_count_before
allocation_count_after <= 0.10 * allocation_count_before
```

System gate：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p8_batch_major_runtime_before_after.svg
p8_step_ratio_by_bucket_strategy.svg
p8_native_runtime_pareto.svg
p8_runtime_component_waterfall.svg
```

---

## P9：system-legal exact-signal controller v14

### 目标

组合 P6 decision survivor 和 P8 runtime survivor，建立 official system controller。

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
secondary_outcome_delta_fields_present
candidate_tensor_payload_missing_count
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
functional_update_payload_missing_count
stable_accept_full_row_materialization_present
outcome_labels_present
materialized_system_path
native_bucket_kernel_used
stable_accept_contract_used
risk_support_repair_used
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

P9 pass：

```text
official_eligible = 1
system_legal_controller_pass = 1
materialized_system_path = 1
stable_accept_full_row_materialization_present = 1
outcome_labels_present = 1
native_bucket_kernel_used = 1
risk_support_repair_used = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
accept_disagreement_count = 0
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
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
p9_system_controller_cost_quality_frontier.svg
p9_decision_repair_vs_runtime_pareto.svg
p9_family_strata_balance.svg
p9_official_gate_dashboard.svg
```

---

## P10：leave-dataset-out / leave-stratum-out

### 目标

只有 P9 pass 后 official 打开。证明 repaired controller 不是 pooled calibration artifact，也不是 dataset-specific route。

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
p10_leave_dataset_out_matrix.svg
p10_leave_stratum_out_matrix.svg
p10_dataset_tuning_audit.svg
p10_leaveout_failure_modes.svg
```

---

## P11：official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。只有 P9/P10 pass 后允许 official。

### 设置

```text
base = R2 repaired base checkpoint
controller_id = best P10 survivor
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
  ShuffledRiskSupportFilter
  ShuffledOutcomeLabel
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
ShuffledStableAcceptScore = fail
ShuffledStableAcceptRank = fail
ShuffledStableAcceptTiePolicy = fail
ShuffledRiskSupportFilter = fail
ShuffledOutcomeLabel = fail
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

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p11_official_paired_replay_pareto.svg
p11_macro_beat_rate.svg
p11_signal_stratum_win_matrix.svg
p11_shuffle_control_matrix.svg
p11_system_gate_distribution.svg
```

---

## P12：short-run / full-run / robustness

### 目标

只有 P11 pass 后打开。验证 local causal advantage 能否进入连续训练，并排除 LR / QuadraticFeatureMLP / shuffled-controller explanations。

### 设置

```text
short-run steps = 50,240,640
full-run seeds = 0..9
datasets = MNIST,Fashion-MNIST,KMNIST
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
  ShuffledRiskSupportFilter
  ShuffledOutcomeLabel
  ShuffledNativeBucketKernel
  ShuffledFunctionalUpdatePayload
  ShuffledFrozenBridge
```

### 必须记录

```text
dataset
seed
candidate
steps
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

and at least one：

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
contract_audit_v9282.csv
p0_v9281_boundary_reproduction.csv
p1_native_scoreq_rank_contract_closure.csv
p2_candidate_lifecycle_canonicalization_drift_audit.csv
p3_secondary_outcome_delta_materialization.csv
p4_bad_accepted_risk_support_autopsy_v2.csv
p5_outcome_grounded_risk_support_sufficient_statistics.csv
p6_dataset_agnostic_decision_repair.csv
p7_runtime_empty_step_elimination.csv
p8_measured_batch_major_native_runtime.csv
p9_system_legal_exact_signal_controller_v14.csv
p10_leave_dataset_and_stratum_out.csv
p11_official_paired_replay.csv
p12_short_full_robustness_strong_baseline.csv

scoreq_rank_contract_trace_v9282.csv
candidate_lifecycle_trace_v9282.csv
secondary_outcome_delta_trace_v9282.csv
bad_accepted_autopsy_trace_v9282.csv
risk_support_feature_trace_v9282.csv
decision_repair_trace_v9282.csv
empty_step_runtime_trace_v9282.csv
batch_major_native_runtime_trace_v9282.csv
system_controller_trace_v9282.csv
leaveout_trace_v9282.csv
paired_replay_branch_trace_v9282.csv
short_full_trace_v9282.csv

route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9281_boundary_unstable
F3_dataset_tuning_detected
F4_payload_binding_regression
F5_scoreq_rank_contract_fail
F6_accept_disagreement_reappears
F7_candidate_lifecycle_unexplained
F8_candidate_row_identity_mismatch
F9_secondary_outcome_delta_missing
F10_bad_accepted_autopsy_incomplete
F11_risk_support_feature_unpredictive
F12_no_dataset_agnostic_decision_repair
F13_decision_repair_precision_fail
F14_decision_repair_coverage_fail
F15_decision_repair_bad_event_fail
F16_decision_repair_null_rate_fail
F17_decision_repair_lcb_ucb_fail
F18_support_balance_fail
F19_empty_step_elimination_fail
F20_batch_major_runtime_not_materialized
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
  v9.2.81 boundary reproduced.

R2-ScoreQRankContractPass:
  score_q / rank / accept contract closed.

R3-CandidateLifecycleExplained:
  candidate drift explained or canonicalized.

R4-SecondaryOutcomeDeltaMaterialized:
  downstream outcome deltas complete.

R5-BadAcceptedAutopsyPass:
  bad accepted rows have stable attribution.

R6-RiskSupportFeaturePass:
  outcome-grounded risk/support sufficient statistics predictive.

R7-DecisionRepairPass:
  dataset-agnostic decision repair passes heldout gates.

R8-EmptyStepRuntimePass:
  zero-candidate fixed launch/sync eliminated.

R9-BatchMajorNativeRuntimePass:
  measured batch-major native runtime reaches system envelope.

R10-SystemLegalExactSignalControllerPass:
  system controller passes decision + compute gates.

R11-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R12-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R13-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R14-FullFunctionalPass:
  short/full run task / geometry / system / control gates pass.

R15-PayloadBindingRegression:
  payload binding no longer reproduces.

R16-StableAcceptDecisionRegionUnsafe:
  no dataset-agnostic repair reaches precision/bad-event gates.

R17-CandidateLifecycleStillUnstable:
  candidate schema drift cannot be canonicalized.

R18-RuntimeStillFragmented:
  empty-step / batch-major measured runtime not closed.

R19-SystemStillTooExpensive:
  decision gates pass but step_ratio_q90 remains >1.50.

R20-ComputePassButLeaveoutFail:
  system controller overfits pooled calibration.

R21-ComputePassButPairedReplayFail:
  controller is system-legal but not causally superior to controls.

R22-PivotToOutcomeGroundedPrimitive:
  StableAccept patching exhausted; redesign accept primitive.

R23-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` must record：

```text
route
v9281_boundary_pass
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

scoreq_rank_contract_pass
score_quantized_disagreement_count
rank_disagreement_count
accept_disagreement_count
native_scoreq_emit_bitexact
fallback_row_count

candidate_lifecycle_pass
old_candidate_count
new_candidate_count
candidate_jaccard
candidate_count_change_explained
candidate_lifecycle_primary_effect

outcome_primary_pass
outcome_secondary_delta_pass
missing_primary_label_count
missing_secondary_delta_count
ambiguous_label_count
label_source

bad_accepted_autopsy_pass
primary_bad_accepted_failure_mode
bad_accepted_attribution_fraction
precision_failure_attribution_fraction

risk_support_feature_pass
best_bad_risk_feature
best_bad_risk_auc
best_safe_good_feature
best_safe_good_auc

decision_repair_pass
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

empty_step_runtime_pass
empty_step_kernel_fraction_before
empty_step_kernel_fraction_after
zero_candidate_step_count
active_step_count

batch_major_runtime_pass
kernel_count_before
kernel_count_after
sync_count_before
sync_count_after
allocation_count_before
allocation_count_after
avg_candidates_per_kernel_after
effective_candidates_per_launch
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
success_v9282_strict_purekan_functional
success_v9282_full_functional
success_v9282_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 native score_q / rank contract closure
  P2 candidate lifecycle canonicalization
  P3 secondary outcome delta materialization
  P4 bad accepted autopsy v2
  P7 empty-step runtime elimination

Batch 2:
  P5 risk/support sufficient statistics
  P6 dataset-agnostic decision repair
  P8 measured batch-major native runtime

Batch 3:
  P9 system-legal controller
  P10 leave-dataset-out / leave-stratum-out scout

Batch 4:
  official P10 LDO/LSO
  official P11 paired replay

Batch 5:
  P12 short/full/robustness only if P11 passes
```

Gate rule：

```text
P6 cannot pass unless:
  P1 pass
  P3 primary outcome labels pass
  P4 bad accepted autopsy pass
  P5 risk/support feature pass

P8 cannot pass unless:
  P1 pass
  P7 empty-step runtime pass

P9 cannot pass unless:
  P6 decision repair pass
  P8 measured runtime pass
  step_ratio_q90 <= 1.50
  accept_disagreement_count = 0
  score_quantized_disagreement_count = 0
  rank_disagreement_count = 0
  materialized_system_path = 1
  native_bucket_kernel_used = 1
  bridge_score_inside_kernel = 1
  accept_bit_inside_kernel = 1
  risk_support_repair_used = 1
  audit_only_cost_removal = 0
  diagnostic_derived_from_measured_components = 0
  projection_used = 0
  source_measured_gap_used = 0
  formula_proxy_used = 0
  full_online_payload_binding = 1
  full_online_update_payload_binding = 1

P10/P11 diagnostic rows may be measured before all gates finish,
but official status requires:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  stable accept controller calibrated
  risk/support repair calibrated
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
v9.2.81 boundary reproduced
score_q/rank contract repaired or fully attributed
candidate lifecycle drift explained
secondary outcome deltas measured
bad accepted rows autopsy completed
risk/support feature factory measured
decision repair measured
empty-step runtime elimination measured
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
risk/support feature pass
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
empty-step runtime pass
+
batch-major runtime pass
+
native kernel used in P6
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
1. v9.2.81 boundary cannot be reproduced；
2. payload binding regresses；
3. score_q / rank mismatch reappears；
4. candidate lifecycle drift cannot be explained；
5. primary outcome labels become missing/ambiguous；
6. secondary deltas remain missing and paired replay is requested；
7. bad accepted rows cannot be attributed；
8. no predictive risk/support feature exists；
9. no dataset-agnostic decision repair reaches precision/bad-event gates；
10. repair reaches gates only by dataset-specific threshold；
11. empty-step launch/sync cannot be eliminated；
12. batch-major runtime remains diagnostic-only；
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

## Case A：P9 system pass + P10/P11 pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full run and robustness。

## Case B：score_q/rank 修复后 decision 仍失败

必须声明：

```text
StableAccept contract is bit-exact, but accepted region is unsafe under same-run outcome labels.
```

下一步修 outcome-grounded accept score / risk-support primitive，不再追 quantization。

## Case C：risk/support repair 失败

必须声明：

```text
AC2Q2 StableAccept cannot be made safe by dataset-agnostic risk/support patching.
```

下一步进入 new observable risk primitive 或 accept-score redesign，不按 dataset 调 threshold。

## Case D：decision repair pass but runtime still slow

必须声明：

```text
controller is decision-legal, but full-system native runtime remains above envelope.
```

下一步基于 P7/P8 waterfall 修 batch-major launch/sync，而不是重调 controller。

## Case E：runtime pass but decision repair fails

必须声明：

```text
system path is efficient, but StableAccept decision region does not select safe-good events reliably.
```

下一步修 outcome-grounded accept score，而不是继续 kernelization。

## Case F：P9 pass but LDO/LSO fail

必须声明：

```text
controller overfits pooled calibration or signal strata.
```

不能通过 dataset-specific tuning 写成功；下一步修 dataset-agnostic support/family reliability。

## Case G：P9/P10 pass but paired replay fail

必须声明：

```text
controller is system-legal but not causally superior to matched controls.
```

下一步回到 functional event/value target，而不是继续 runtime kernelization。

---

# Part XII. 最终建议

v9.2.82 的一句话策略是：

$$
\boxed{
\text{不要再围绕 StableAccept 做小补丁；用 same-run outcomes 重建 risk/support accept score，同时把 empty-step-free batch-major runtime 做成 measured path。}
}
$$

当前最关键的问题不是：

```text
full-row materializer 是否存在；
primary outcome label 是否存在；
accept_disagreement 是否为 0；
score_q/rank 是否已经 fallback 清零；
native kernel 是否进入 P6；
quantile-tail 是否可优化；
basis_norm 是否可局部降低；
PF5 candidate rate 是否可行；
C3/T2/C4/E2 controller 是否要重调；
Fashion/KMNIST/MNIST 谁更好。
```

而是：

```text
1. StableAccept accepted region 为什么仍然有 142 个 bad rows？
2. FMA3 risk_ucb_underestimation 为什么占主导？
3. FMA2 candidate-set drift 是否改变了 accepted population？
4. null/bad conflict 是否可以用 decoupled UCB 解决？
5. horizon-tail risk 是否必须进入 official score？
6. 是否存在 dataset-agnostic risk/support repair，让 precision >=0.75 且 bad-event <=0.05？
7. secondary outcome deltas 能否补齐到 paired replay ready？
8. P6 为什么在 6652 个 zero-candidate steps 上仍 launch/sync？
9. empty-step skip 能否把 kernel/sync 大幅降低？
10. avg_candidates_per_kernel 能否从 0.0396 拉到 >=8？
11. step_ratio_q90 能否从 2.213 降到 <=1.50？
12. system controller 是否 official eligible？
13. LDO/LSO 是否通过？
14. official paired replay 是否打过 AdamWParallel / bestLR？
```

v9.2.82 的结果将给出清晰分叉：

```text
if decision repair and runtime repair both pass:
  open LDO/LSO and official paired replay.

if score_q/rank bit-exact but decision fails:
  StableAccept accepted region unsafe; redesign outcome-grounded accept score.

if risk/support repair cannot reach gates:
  stop patching StableAccept; pivot to observable risk primitive.

if decision passes but runtime fails:
  continue empty-step-free batch-major runtime, not controller tuning.

if runtime passes but decision fails:
  stop kernel work; fix accept score.

if P9 passes but paired replay fails:
  system is legal, but functional causal advantage is insufficient.

if LDO/LSO fails:
  repair dataset-agnostic support/family reliability, not dataset-specific tuning.
```
