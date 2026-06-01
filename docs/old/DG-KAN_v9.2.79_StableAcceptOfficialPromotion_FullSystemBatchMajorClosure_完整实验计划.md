# DG-KAN v9.2.79 Stable-Accept Official Promotion 与 Full-System Batch-Major Runtime Closure 完整实验计划

> 本计划基于 v9.2.78 `Native Bridge-Accept Numerical Closure 与 Bucketed Runtime Official Promotion` 的真实执行结果制定。  
> v9.2.78 的 terminal route 是：
>
> ```text
> route = R18-StableAcceptLocalClosedButSystemNotOfficial
> base_candidate = LQ-t2-h256
> success_v9278_strict_purekan_functional = False
> success_v9278_full_functional = False
> success_v9278_external_ready = False
> ```
>
> v9.2.78 的关键事实是：
>
> ```text
> Native accept autopsy:
>   accept_disagreement_count_before = 12
>   large_margin_disagreement_count = 0
>   borderline_disagreement_fraction_1e_6 = 1.0
>   bridge_score_error_max = 5.960464477539063e-08
>   logits_error_max = 1.811981201171875e-05
>   delta_error_max = 3.3527612686157227e-08
>   tail_disagreement_count = 0
>   root_cause = median_boundary_fragility
>
> Stable accept contract:
>   best_accept_contract_id = AC2Q2-stable-quantized-1e5-event-tie
>   tie_policy = stable_rank_event_id_tie
>   accept_rule_changed = 1
>   calibration_rerun_required = 1
>   accept_disagreement_before = 12
>   accept_disagreement_after = 0
>   agreement_reference_accept = 1.0
>   reference_drift_vs_current_accept = 0
>   stable_accept_contract_pass = 1
>
> Native stable-accept kernel:
>   native_cuda_bucket_kernel_used = 1
>   stable_accept_cuda_kernel_used = 1
>   basis_norm_bucketed = 1
>   W2_delta_bucketed = 1
>   bridge_score_inside_kernel = 1
>   accept_bit_inside_kernel = 1
>   kernel_count_before = 3750
>   kernel_count_after = 216
>   sync_count_before = 1250
>   sync_count_after = 24
>   avg_candidates_per_kernel_after = 11.541666666666666
>   native_time_ms_q90 = 0.6102416664361954
>   eager_time_ms_q90 = 1.098200213164091
>   q90_reduction = 0.4443256711105608
>   native_bucket_kernel_v2_pass = 1
>
> Controller boundary:
>   controller_precision = 0.8380281690140845
>   controller_coverage = 0.03130511463844797
>   controller_bad_event = 0.02464788732394366
>   controller_null_rate = 0.13028169014084506
>   controller_step_ratio_q90 = 2.713295831053225
>   official_eligible = 0
>   system_legal_controller_pass = 0
>   primary_blocker = batch_major_or_full_system_step_ratio_not_closed
>   next_required_implementation = integrate_stable_accept_into_batch_major_system_runtime
> ```
>
> v9.2.79 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.78 已经局部解决 native accept correctness；下一步不是再找 signal，而是把 stable accept 作为 official controller rule 完整重跑并接入 full-system measured runtime。}
> }
> $$
>
> 这次不能再把问题泛化为 “basis_norm 太慢” 或 “native kernel 没实现”。数据已经说明：
>
> ```text
> quantile-tail 已有 exact runtime pass；
> native CUDA bucket kernel 已有 q90 reduction；
> accept disagreement 已被 stable rank/event-id tie 清零；
> 真正缺口是 official promotion：
>   1. stable accept 是 rule change，需要完整 calibration / heldout / support rerun；
>   2. P3 local native q90 不能直接替代 P6 full-system step ratio；
>   3. native stable-accept kernel 必须进入 end-to-end train-stream controller path；
>   4. P7/P8/P9/P10 仍因 P6 not official 被 gate。
> ```
>
> 因此 v9.2.79 的目标不是小修 kernel，也不是调 threshold，而是完成 **stable accept officialization + full-system runtime closure**。

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

v9.2.79 新增硬约束：

```text
stable accept rule change 必须触发 full calibration / heldout / support rerun；
不能只凭 P2 local stable accept pass 宣称 official；
不能只凭 P3 native q90 reduction 宣称 system pass；
不能复用旧 P6 step_ratio_q90 当作 native full-system measurement；
不能把 local native kernel timing 替代 end-to-end step timing；
不能把 accept bit 移回 Python 再声明 inside-kernel pass；
不能使用 source-measured gap；
不能使用 formula proxy；
不能按 dataset 选择不同 accept threshold / tie policy / bucket route；
不能只 report q90 reduction 而忽略 official decision/support gates；
不能打开 LDO/LSO 或 paired replay，除非 P6 system controller official。
```

允许使用：

```text
native CUDA / Triton bucket kernel
stable rank / event-id tie
stable quantized score key
borderline exact fallback
compact bridge lookup inside kernel
batch-major bucket runtime
persistent workspace
post-step audit subset
calibration split rerun
heldout split rerun
leave-dataset-out / leave-stratum-out after system pass
```

---

# Part I. 对 v9.2.78 的独立判断

## 1. v9.2.78 没有达到最终目标

v9.2.78 没有 strict PureKAN functional success。原因不是 signal 不存在，也不是 kernel 完全不可行，而是 system controller 仍没有 official：

```text
official_eligible = 0
system_legal_controller_pass = 0
controller_step_ratio_q90 = 2.713295831053225
P7-P10 = not_run
```

这次的 gate 是合理的。stable accept 的局部闭合是重大进展，但 stable accept 是 rule change；只要 rule 变了，就必须重新验证 calibration / heldout / support balance / decision gates。不能用旧 C3-T2PlusBackfill 的 decision metrics 直接盖章。

## 2. v9.2.78 的真实进展

v9.2.78 的进展很关键，主要是三件事。

第一，accept disagreement 的根因已定位。P1 autopsy 显示 disagreement 不是大 margin 错误，不是 tail/logits/delta 崩坏，而是 median boundary fragility：

```text
large_margin_disagreement_count = 0
borderline_disagreement_fraction_1e_6 = 1.0
bridge_score_error_max = 5.960464477539063e-08
tail_disagreement_count = 0
root_cause = median_boundary_fragility
```

这把问题从 “native kernel 可能错” 缩小到 “边界 accept contract 对浮点微小差异不稳定”。

第二，stable rank/event-id tie 已经局部闭合。`AC2Q2-stable-quantized-1e5-event-tie` 将 accept disagreement 从 `12` 降到 `0`，agreement vs current reference accept = `1.0`。这说明稳定 tie policy 是正确方向。

第三，native stable-accept kernel 已经有真实局部 runtime 优势。P3 中：

```text
kernel_count: 3750 -> 216
sync_count: 1250 -> 24
avg_candidates_per_kernel_after = 11.541666666666666
native q90 = 0.610242 ms
eager q90 = 1.098200 ms
q90 reduction = 44.43%
```

这说明 batch-major/native bucket 不是空想。v9.2.77 的 “native kernel 有 reduction 但 accept 不一致” 已被推进到 “accept 一致，kernel 局部 pass，但 full-system promotion 未闭合”。

## 3. v9.2.78 的真实失败

v9.2.78 的失败是 **promotion failure**，不是 **local native-kernel failure**。当前的悖论是：

```text
P3 local native kernel v2 pass = 1；
accept disagreement after stable tie = 0；
kernel/sync/avg candidates per kernel 明显改善；
但 P6 controller_step_ratio_q90 仍是 2.713296；
official_eligible = 0。
```

这说明 P6 还没有把 stable accept native kernel 作为完整 train-stream controller runtime 来重跑，或者 P6 仍在使用 source boundary / old runtime accounting。也就是说，当前不能证明 native path 不够快；只能证明 native path 尚未被提升成 end-to-end system path。

因此下一步最关键的不是继续局部优化 q90，而是问：

```text
P3 native stable-accept kernel 是否真正进入 P6 end-to-end controller step？
P6 的 step_ratio_q90 = 2.713296 是旧路径残留，还是新路径真实系统成本？
stable accept rule change 后，decision gates 是否仍保持？
batch-major runtime 在 full train-stream 是否保持 P3 的 kernel/sync 优势？
```

## 4. 当前 blocker 的本质

当前 blocker 应写成：

$$
\boxed{
\text{stable native accept is locally correct, but not yet promoted into a calibrated, heldout-tested, end-to-end measured system controller.}
}
$$

不是：

```text
quantile-tail 失败；
basis_norm 失败；
selected-feature 失败；
native CUDA kernel 不可行；
bridge score 无法 inside kernel；
accept bit 不能 inside kernel；
controller precision/coverage 没有信号；
dataset-specific failure。
```

当前真正缺的是：

```text
1. stable accept rule officialization；
2. calibration / heldout / support rerun；
3. full-system runtime measurement using the native stable kernel；
4. batch-major system path closure；
5. after P6 pass, LDO/LSO and official paired replay。
```

## 5. 是否还在正确道路上

是。v9.2.78 是正向推进。路线已经走到 system-legal controller 前最后几步：

```text
decision frontier stable
payload binding stable
quantile-tail runtime pass
native bucket kernel local pass
accept disagreement closed
remaining blocker = official promotion + full-system runtime
```

现在如果回去调 C3/T2/C4/E2、重设 safe-useful target、按 dataset 调参、或者换一个普通 basis，都是偏离问题本质。

---

# Part II. v9.2.79 总体目标

v9.2.79 的总体目标是：

$$
\boxed{
\text{将 AC2Q2 stable accept contract 与 native bucket kernel v2 提升为 official system controller，并验证 full-system step ratio 是否进入 }1.50\text{ envelope。}
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
StepRatio_{q90}\leq1.50,
$$

$$
AcceptDisagreement=0.
$$

最低有效推进目标：

```text
stable accept official calibration rerun completed；
heldout decision gates rerun completed；
support balance rerun completed；
native stable kernel used in full-system P6 path；
end-to-end native step timing measured；
P6 reason 不再是 old boundary / rule-change blocked。
```

v9.2.79 必须回答十个问题：

```text
Q1:
  v9.2.78 boundary 是否稳定复现？

Q2:
  AC2Q2 stable accept rule 作为 official controller rule 后，calibration split 指标是否保持？

Q3:
  heldout split 上 precision / coverage / bad-event / null-rate / LCB-UCB 是否仍满足 gate？

Q4:
  support balance 是否仍满足 accepted strata / family / max share gates？

Q5:
  native stable-accept kernel 是否真的进入 full P6 train-stream controller runtime？

Q6:
  P6 end-to-end step ratio 是否仍是 2.713296？
  如果是，它来自哪个 component？
  如果不是，是否进入 <=1.50？

Q7:
  P3 local native q90 reduction 是否能在 full train-stream 中保持？

Q8:
  batch-major kernel/sync reduction 是否能在 full P6 path 中保持？
  kernel_count_after / sync_count_after 是否仍为 216 / 24 级别？

Q9:
  如果 P6 pass，LDO/LSO 是否通过？

Q10:
  official paired replay 是否打过 AdamWParallel / bestLR？
```

---

# Part III. 核心假设

## H1：stable accept rule 官方化后，decision metrics 不会明显漂移

H1 成立标准：

```text
accept_rule = AC2Q2-stable-quantized-1e5-event-tie
calibration_rerun = 1
heldout_rerun = 1
accept_disagreement_count = 0
reference_drift_vs_current_accept <= 0.01
```

并且：

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

H1 失败标准：

```text
stable accept 改 rule 后 precision / bad-event / null-rate 任一 gate 失败；
or coverage 掉出 [0.03,0.15]；
or stable accept 与 current accept drift 大于 0.01。
```

## H2：P6 step ratio = 2.713296 是旧路径残留或未集成，而不是 native stable kernel 的真实 full-system下界

H2 成立标准：

P6 full-system timing 使用 native stable kernel 后：

```text
native_bucket_kernel_used_in_p6 = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
kernel_count_after <= 0.50 * kernel_count_before
sync_count_after <= 0.50 * sync_count_before
avg_candidates_per_kernel_after >= 8
```

且：

$$
StepRatio_{q90}<2.713296.
$$

Full success：

$$
StepRatio_{q90}\leq1.50.
$$

H2 失败标准：

native stable kernel 明确进入 P6，component attribution 完整，但 step ratio 仍 `>1.50`。此时才说明 full-system cost 仍未闭合。

## H3：batch-major native runtime 可以维持 P3 的局部收益

H3 成立标准：

```text
native_time_q90 <= 0.75 * eager_time_q90
q90_reduction >= 0.40
avg_candidates_per_kernel_after >= 8
kernel_count_after <= 216 * 1.25
sync_count_after <= 24 * 1.25
accept_disagreement_count = 0
```

H3 失败标准：

P3 local native q90 pass 不能复现在 full train-stream；kernel/sync 回到 3750/1250 量级；或者 avg candidates per kernel 低于 2。

## H4：stable accept rule change 需要重跑全部 downstream gate，不能只检查局部 accept agreement

H4 成立标准：

```text
calibration_split_rerun = 1
heldout_split_rerun = 1
support_balance_rerun = 1
leaveout_gated_until_p6_pass = 1
paired_replay_gated_until_p6_pass = 1
```

H4 失败标准：

P2/P3 局部 pass 后直接打开 P7/P8，或只用 old decision metrics 写 official。

## H5：如果 P6 pass 但 paired replay fail，blocker 才回到 functional value target

H5 成立标准：

```text
P6 system pass = 1
P7 LDO/LSO measured
P8 paired replay measured
P8 fails to beat AdamWParallel / bestLR
```

H5 失败标准：

P6 未过就讨论 functional value target。这会混淆 system closure 和 causal advantage。

---

# Part IV. Stable accept official controller contract

## 1. Official accept rule

v9.2.79 official candidate rule is:

```text
accept_contract_id = AC2Q2-stable-quantized-1e5-event-tie
score_quantization = 1e5
tie_policy = stable_rank_event_id_tie
dataset_name_used = 0
```

For each candidate event $e$:

$$
score_q(e)=round(10^5\cdot score(e)).
$$

Rank is determined by:

$$
rank(e)=stable\_sort(-score_q(e), event\_id(e)).
$$

The accept rule is:

$$
Accept(e)=1 \iff rank(e)\leq K,
$$

where $K$ is determined by the same frozen calibration process used for the official controller, not by test data.

This rule must be used identically in:

```text
reference implementation
native CUDA / Triton kernel
calibration split
heldout split
leave-dataset-out
leave-stratum-out
paired replay
short/full run
```

## 2. Required row fields

Each accepted or candidate row must record:

```text
event_id
candidate_id
dataset
seed
family_id
bucket_id
horizon
score_ref
score_native
score_quantized_ref
score_quantized_native
rank_ref
rank_native
tie_key_ref
tie_key_native
accept_ref
accept_native
accept_disagreement
calibration_split_id
heldout_split_id
controller_rule_version
native_kernel_version
```

## 3. Rule change handling

Because `accept_rule_changed = 1`, v9.2.79 must create a new official controller ID:

```text
controller_id = C3Q2-StableTieNativeBridge
```

It must not silently overwrite:

```text
C3-T2PlusBackfill
```

The old controller is a reference baseline. The new controller must pass all gates independently.

---

# Part V. Candidate designs

## 1. Controller candidates

### CTL0：Old C3-T2PlusBackfill reference

Reference only. It is useful for agreement, but not official for stable tie unless rerun.

### CTL1：C3Q2-StableTieReference

Reference Python/torch stable quantized rank/event-id tie.

### CTL2：C3Q2-StableTieNative

Native CUDA stable tie implementation.

### CTL3：C3Q2-StableTieNativeBatchMajor

Native CUDA stable tie with batch-major bucket runtime.

### CTL4：C3Q2-StableTieBorderlineFallback

Stable tie plus exact fallback for rare borderline rows if any disagreement reappears.

### CTL5：C3Q2-StableTieCompactBridge

Stable tie with compact bridge lookup inside kernel.

## 2. Runtime candidates

### RT0：v9.2.78 P3 local native kernel reference

Local native q90 reduction reference. Not enough for P6 official by itself.

### RT1：P6IntegratedNativeStableKernel

Same native kernel, but wired into end-to-end P6 controller.

### RT2：BatchMajorNativeStableKernel

Use batch-major buckets to maintain `avg_candidates_per_kernel_after >= 8`.

### RT3：PersistentWorkspaceNativeStableKernel

Keep candidate / basis_norm / W2_delta / bridge_score / accept workspace persistent across P6.

### RT4：CompactBridgeLookupNativeKernel

Move compact bridge lookup fully inside native kernel.

### RT5：NativeStableKernelWithExactAudit

Native timed path plus post-step exact audit subset.

### RT6：HybridBestP6Runtime

Best combination of RT1-RT5.

## 3. Numerical audit candidates

### NA0：Stable accept zero-disagreement reference

Reference audit.

### NA1：Quantized score error audit

Compare `score_q_ref` and `score_q_native`.

### NA2：Rank stability audit

Compare `rank_ref` and `rank_native`.

### NA3：Event-id tie audit

Verify tie key consistency.

### NA4：Borderline fallback stress audit

Stress rows within small score margin.

### NA5：Dataset/family/horizon diagnostic

Diagnostic only, not controller route.

---

# Part VI. 实验阶段

## P0：v9.2.78 boundary reproduction

### 目标

确认 v9.2.78 boundary 稳定。必须复现 stable accept local closure 与 P6 not official 状态。

### 必须记录

```text
route
source_route_v9278
payload_binding_contract_pass
quantile_tail_runtime_pass
native_accept_autopsy_pass
stable_accept_contract_pass
native_bucket_kernel_v2_pass
accept_disagreement_count_before
accept_disagreement_count_after
accept_rule_changed
calibration_rerun_required
native_cuda_bucket_kernel_used
basis_norm_bucketed
W2_delta_bucketed
bridge_score_inside_kernel
accept_bit_inside_kernel
native_time_ms_q90
eager_time_ms_q90
q90_reduction
kernel_count_before
kernel_count_after
sync_count_before
sync_count_after
avg_candidates_per_kernel_after
controller_step_ratio_q90
official_eligible
system_legal_controller_pass
fake_proxy_count
cpu_offload_used
```

### 判断标准

P0 pass：

```text
source route = R18-StableAcceptLocalClosedButSystemNotOfficial
accept_disagreement_count_after = 0
stable_accept_contract_pass = 1
native_bucket_kernel_v2_pass = 1
official_eligible = 0
system_legal_controller_pass = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9278_boundary_ladder.svg
p0_accept_disagreement_before_after.svg
p0_native_local_vs_system_gap.svg
```

---

## P1：stable accept official calibration rerun

### 目标

把 AC2Q2 stable accept 从 local repair 提升为 official calibration rule。P1 是本轮第一硬门：如果不重跑 calibration，不能 official。

### 必须记录

```text
controller_id
accept_contract_id
accept_rule_changed
calibration_rerun
calibration_split_id
threshold_or_topk_policy
score_quantization
tie_policy
event_count
candidate_count
accepted_count
candidate_rate
precision_cal
coverage_cal
bad_event_cal
null_rate_cal
precision_lcb_cal
bad_event_ucb_cal
support_cal
accepted_signal_strata_count_cal
accepted_family_count_cal
max_family_share_cal
max_stratum_share_cal
accept_disagreement_count
reference_drift_vs_old_accept
dataset_name_used
validation_used
test_used
```

### 判断标准

P1 pass：

```text
calibration_rerun = 1
accept_disagreement_count = 0
dataset_name_used = 0
validation_used = 0
test_used = 0
accepted_signal_strata_count_cal >= 5
accepted_family_count_cal >= 32
max_family_share_cal <= 0.50
max_stratum_share_cal <= 0.60
```

Decision pre-gate：

$$
Precision_{\text{cal}}\geq0.75,
$$

$$
Coverage_{\text{cal}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{cal}}\leq0.05,
$$

$$
NullRate_{\text{cal}}\leq0.15.
$$

### 可视化

```text
p1_calibration_frontier_stable_accept.svg
p1_old_vs_stable_accept_overlap.svg
p1_support_balance_calibration.svg
```

---

## P2：stable accept heldout and support rerun

### 目标

验证 stable accept rule 在 heldout 上仍满足 official decision gates。P2 不允许复用旧 heldout summary；必须重跑。

### 必须记录

```text
controller_id
accept_contract_id
heldout_split_id
event_count
candidate_count
accepted_count
candidate_rate
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
reference_accept_agreement
accept_disagreement_count
score_quantized_disagreement_count
rank_disagreement_count
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

P2 pass：

```text
accept_disagreement_count = 0
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
dataset_name_used = 0
posthoc_used_at_commit = 0
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
p2_heldout_decision_metrics.svg
p2_lcb_ucb_gate.svg
p2_rank_agreement_confusion.svg
p2_support_balance_heldout.svg
```

---

## P3：native stable kernel full-system integration audit

### 目标

证明 P3 local native kernel 真实进入 P6 train-stream full-system controller path，而不是只存在局部 timing。

### 必须记录

```text
runtime_candidate_id
native_kernel_id
controller_id
native_kernel_used_in_p6
basis_norm_bucketed
W2_delta_bucketed
bridge_score_inside_kernel
accept_bit_inside_kernel
compact_bridge_lookup_inside_kernel
stable_accept_inside_kernel
candidate_count
accepted_count
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
tail_disagreement_count
```

### 判断标准

P3 pass：

```text
native_kernel_used_in_p6 = 1
basis_norm_bucketed = 1
W2_delta_bucketed = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
stable_accept_inside_kernel = 1
accept_disagreement_count = 0
q90_reduction >= 0.40
avg_candidates_per_kernel_after >= 8
kernel_count_after <= 0.50 * kernel_count_before
sync_count_after <= 0.50 * sync_count_before
allocation_count_after <= 0.10 * allocation_count_before
```

### 可视化

```text
p3_p6_kernel_integration_ladder.svg
p3_kernel_sync_allocation_before_after.svg
p3_candidates_per_kernel_hist.svg
p3_native_vs_eager_q90.svg
```

---

## P4：full-system step attribution with native stable kernel

### 目标

解释 P6 step ratio 是否真正下降。P4 必须回答：旧 `2.713296` 是旧路径残留，还是 native path full-system 仍超线。

### 必须记录

```text
system_runtime_id
controller_id
runtime_candidate_id
selector_time_ms
candidate_pack_time_ms
quantile_tail_time_ms
basis_norm_time_ms
W2_delta_time_ms
probe_logits_time_ms
bridge_score_time_ms
stable_accept_time_ms
update_payload_time_ms
payload_apply_time_ms
kernel_launch_time_ms
sync_time_ms
allocation_time_ms
audit_outside_timed_time_ms
other_time_ms
unknown_fraction
total_step_time_ms_q50
total_step_time_ms_q90
mlp_step_time_ms_q90
step_ratio_q90
memory_ratio
```

### 判断标准

P4 pass：

```text
unknown_fraction <= 0.05
native stable kernel component present = 1
audit outside timed path = 1
```

System diagnostic pass：

$$
StepRatio_{q90}\leq2.00.
$$

Full system candidate:

$$
StepRatio_{q90}\leq1.50.
$$

### 可视化

```text
p4_full_system_cost_waterfall.svg
p4_old_vs_native_step_ratio.svg
p4_component_ratio_stacked_bar.svg
p4_runtime_q50_q90_distribution.svg
```

---

## P5：batch-major native runtime closure

### 目标

如果 P4 仍超线，P5 进一步优化 batch-major native runtime，目标是把 local native advantage 保留到 full-system path。

### 必须记录

```text
batch_major_candidate_id
bucket_strategy
bucket_sizes
event_count
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
full_system_step_ratio_q90
accept_disagreement_count
agreement_reference_accept
precision
coverage
bad_event
null_rate
```

### 判断标准

P5 pass：

```text
avg_candidates_per_kernel_after >= 8
or effective_candidates_per_launch >= 8
kernel_count_after <= 0.50 * kernel_count_before
sync_count_after <= 0.50 * sync_count_before
accept_disagreement_count = 0
agreement_reference_accept >= 0.99
```

System candidate：

$$
StepRatio_{q90}\leq1.50.
$$

### 可视化

```text
p5_bucket_occupancy.svg
p5_effective_candidates_per_launch.svg
p5_bucket_strategy_pareto.svg
p5_step_ratio_by_bucket.svg
```

---

## P6：system-legal exact-signal controller v11

### 目标

组合 P1-P5，建立 official system-legal controller。P6 是本轮核心成败点。

### 必须记录

```text
controller_id
system_candidate_id
accept_contract_id
native_kernel_id
runtime_candidate_id
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

P6 pass：

```text
official_eligible = 1
system_legal_controller_pass = 1
materialized_system_path = 1
native_bucket_kernel_used = 1
stable_accept_contract_used = 1
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
p6_system_controller_cost_quality_frontier.svg
p6_reference_vs_stable_accept_overlap.svg
p6_step_ratio_progress_v9271_to_v9279.svg
p6_family_strata_balance.svg
p6_accept_contract_audit_dashboard.svg
```

---

## P7：leave-dataset-out / leave-stratum-out

### 目标

只有 P6 pass 后打开。证明 stable accept native controller 不是 pooled calibration artifact，也不是 dataset-specific route。

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

验证 stable accept native controller 下的 RealFunctional 是否在 strong controls 下有局部因果优势。

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
  ShuffledStableAcceptTiePolicy
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
ShuffledQuantileTailRuntime = fail
ShuffledBasisNormRuntime = fail
ShuffledNativeBucketKernel = fail
ShuffledBridgeScore = fail
ShuffledStableAcceptTiePolicy = fail
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
  ShuffledStableAcceptTiePolicy
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
contract_audit_v9279.csv
p0_v9278_boundary_reproduction.csv
p1_stable_accept_official_calibration.csv
p2_stable_accept_heldout_support_rerun.csv
p3_native_stable_kernel_full_system_integration.csv
p4_full_system_step_attribution_native_stable.csv
p5_batch_major_native_runtime_closure.csv
p6_system_legal_exact_signal_controller_v11.csv
p7_leave_dataset_and_stratum_out.csv
p8_official_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_run_robustness_strong_baseline.csv

stable_accept_calibration_trace_v9279.csv
stable_accept_heldout_trace_v9279.csv
native_kernel_full_system_trace_v9279.csv
full_system_step_cost_trace_v9279.csv
batch_major_native_runtime_trace_v9279.csv
system_controller_trace_v9279.csv
leaveout_trace_v9279.csv
paired_replay_branch_trace_v9279.csv
short_run_trace_v9279.csv

route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9278_boundary_unstable
F3_dataset_tuning_detected
F4_payload_binding_regression
F5_decision_metric_regression
F6_stable_accept_calibration_fail
F7_stable_accept_heldout_fail
F8_support_balance_fail
F9_accept_disagreement_reappears
F10_score_quantized_disagreement
F11_rank_tie_disagreement
F12_native_kernel_not_used_in_p6
F13_bridge_score_not_inside_kernel
F14_accept_bit_not_inside_kernel
F15_native_runtime_q90_regression
F16_kernel_sync_reduction_lost
F17_avg_candidates_per_kernel_too_low
F18_full_system_step_attribution_incomplete
F19_full_system_step_ratio_fail
F20_memory_ratio_fail
F21_source_gap_or_formula_proxy_used
F22_projection_used
F23_system_controller_precision_fail
F24_system_controller_coverage_fail
F25_system_controller_bad_event_fail
F26_system_controller_null_rate_fail
F27_system_controller_lcb_ucb_fail
F28_leave_dataset_out_fail
F29_leave_stratum_out_fail
F30_paired_replay_control_equivalent
F31_shuffle_control_pass
F32_functional_lr_equivalent
F33_short_run_task_drop
F34_full_run_no_macro_hard_stratum_gain
F35_strong_baseline_explains_gain
F36_robustness_fail
F37_external_not_ready
F38_fake_or_proxy_violation
F39_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.78 boundary reproduced.

R2-StableAcceptCalibrationPass:
  AC2Q2 stable accept official calibration pass.

R3-StableAcceptHeldoutSupportPass:
  heldout decision gates and support balance pass.

R4-NativeStableKernelIntegrated:
  native stable kernel is used in full P6 system path.

R5-FullSystemNativeRuntimePass:
  full-system native runtime reaches step_ratio_q90 <= 2.00 with attribution.

R6-BatchMajorRuntimePass:
  batch-major native runtime keeps kernel/sync/candidates-per-kernel improvements.

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

R14-StableAcceptOfficializationFail:
  stable accept local rule fails calibration or heldout official gates.

R15-NativeKernelIntegrationFail:
  native stable kernel cannot be integrated into P6 full-system path.

R16-FullSystemRuntimeStillOldPath:
  P6 step ratio remains old source boundary because runtime integration did not occur.

R17-BatchMajorRuntimeRegression:
  local q90 pass is lost in full train-stream path.

R18-SystemStillTooExpensive:
  native stable path is correct and integrated but step_ratio_q90 remains >1.50.

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
v9278_boundary_pass
dataset_tuning_detected
reference_controller_id
stable_controller_id
accept_contract_id
accept_rule_changed
calibration_rerun
heldout_rerun
support_rerun

payload_binding_contract_pass
candidate_tensor_payload_missing_count
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
functional_update_payload_missing_count

stable_accept_calibration_pass
stable_accept_heldout_support_pass
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
reference_drift_vs_old_accept

native_kernel_integrated_pass
native_kernel_used_in_p6
native_bucket_kernel_used
stable_accept_cuda_kernel_used
basis_norm_bucketed
W2_delta_bucketed
bridge_score_inside_kernel
accept_bit_inside_kernel
kernel_count_reduction
sync_count_reduction
allocation_count_reduction
avg_candidates_per_kernel_after
native_time_ms_q90
eager_time_ms_q90
q90_reduction

full_system_step_attribution_pass
selector_time_ms
quantile_tail_time_ms
basis_norm_time_ms
W2_delta_time_ms
bridge_score_time_ms
stable_accept_time_ms
update_payload_time_ms
kernel_launch_time_ms
sync_time_ms
allocation_time_ms
unknown_fraction
controller_step_ratio_q90
controller_memory_ratio

best_system_controller_id
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
success_v9279_strict_purekan_functional
success_v9279_full_functional
success_v9279_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 stable accept official calibration rerun
  P2 stable accept heldout/support rerun
  P3 native stable kernel full-system integration audit
  P4 full-system step attribution

Batch 2:
  P5 batch-major native runtime closure
  P6 system-legal controller
  P7 leave-dataset-out / leave-stratum-out scout

Batch 3:
  official P7 LDO/LSO
  official P8 paired replay

Batch 4:
  P9 short-run if P8 passes
  P10 full run / robustness / strong baseline only if P9 passes
```

Gate rule：

```text
P1/P2/P3/P4 can run in parallel after P0.
P6 cannot pass unless:
  P1 stable accept calibration pass
  P2 stable accept heldout/support pass
  P3 native stable kernel integrated pass
  P4 full-system step attribution pass
  P5 batch-major runtime pass if P4 step ratio >1.50
  accept_disagreement_count = 0
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

P7/P8 diagnostic rows may be measured before all gates finish,
but official status requires:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  reference controller reproduced
  stable accept controller calibrated
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
v9.2.78 boundary reproduced
stable accept official calibration rerun completed
heldout/support rerun completed
native kernel full-system integration measured
full-system native step attribution measured
system controller measured
no fake/proxy/offload/loss/teacher violation
```

## Stable accept official success

```text
Minimum diagnostic success
+
accept_disagreement_count = 0
+
score_quantized_disagreement_count = 0
+
rank_disagreement_count = 0
+
calibration decision gates pass
+
heldout decision gates pass
+
support balance pass
```

## Runtime materialization success

```text
Stable accept official success
+
native_kernel_used_in_p6 = 1
+
bridge_score_inside_kernel = 1
+
accept_bit_inside_kernel = 1
+
q90_reduction >= 0.40
+
avg_candidates_per_kernel_after >= 8
+
kernel/sync reduction measured
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
1. v9.2.78 boundary cannot be reproduced；
2. payload binding regresses；
3. decision metrics regress under stable accept；
4. stable accept calibration fails；
5. heldout precision / coverage / bad-event / null-rate fails；
6. support balance fails；
7. accept disagreement reappears；
8. quantized score / rank / tie key mismatch；
9. native kernel does not enter P6 full-system path；
10. P6 step ratio remains old 2.713296 because runtime integration did not occur；
11. native q90 reduction disappears in full-system path；
12. bridge score cannot stay inside kernel；
13. accept bit cannot stay inside kernel；
14. kernel/sync/allocation remains old path；
15. avg candidates per kernel remains <2；
16. low-cost path uses source gap / formula proxy；
17. system step_ratio_q90 remains >1.50；
18. memory ratio >1.05；
19. leave-dataset-out fails；
20. leave-stratum-out fails；
21. paired replay remains control-equivalent；
22. shuffle controls pass；
23. short-run task drops；
24. full run gives no macro / hard-stratum / geometry gain；
25. functional breaks system gate；
26. gains are explained by QuadraticFeatureMLP；
27. any teacher/loss/fake/proxy/offload/projection-as-pass violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：P6 system pass + P7/P8 pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full run and robustness。

## Case B：stable accept calibration fails

必须声明：

```text
stable tie policy solved native disagreement locally, but changed the official accept region too much.
```

下一步修 accept contract，不调 dataset-specific threshold。

## Case C：decision gates pass but P6 runtime remains old path

必须声明：

```text
controller is decision-legal, but native stable kernel is not yet connected to full-system runtime.
```

下一步修 runtime integration，不做 paired replay。

## Case D：native kernel integrated but system still slow

必须声明：

```text
decision correctness is closed, but full-system runtime remains above envelope.
```

下一步基于 P4 waterfall 修 batch-major bucket / kernel launch / remaining basis_norm-delta components。

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

v9.2.79 的一句话策略是：

$$
\boxed{
\text{不要再局部修 native kernel；把 AC2Q2 stable accept 正式升格为 controller rule，重跑 calibration/heldout/support，并把 native stable kernel 接进 P6 full-system runtime。}
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
native CUDA kernel 是否有 q90 reduction；
accept disagreement 是否可以清零；
C3/T2/C4/E2 controller 是否要重调；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. AC2Q2 stable accept 作为 official controller rule 后，calibration 是否过？
2. heldout precision / coverage / bad-event / null-rate 是否仍过？
3. support balance 是否仍过？
4. native stable kernel 是否真的进入 P6 full-system path？
5. P6 的 2.713296 step ratio 是旧路径残留还是新路径真实成本？
6. P3 的 q90 reduction 是否能在 full train-stream 中保持？
7. kernel_count / sync_count 是否仍能维持 216 / 24 级别？
8. avg_candidates_per_kernel 是否稳定 >=8？
9. system controller 是否 official eligible？
10. LDO/LSO 是否通过？
11. official paired replay 是否打过 AdamWParallel / bestLR？
```

v9.2.79 的结果将给出清晰分叉：

```text
if stable accept + native runtime passes P6:
  open LDO/LSO and official paired replay.

if stable accept decision gates fail:
  repair accept contract or support reliability, not dataset-specific threshold.

if native kernel does not enter full system:
  fix P6 runtime wiring.

if native kernel enters full system but step ratio >1.50:
  use P4 waterfall to target batch-major / launch / remaining compute.

if compute passes but paired replay fails:
  system is legal, but functional causal advantage is insufficient.

if LDO/LSO fails:
  repair dataset-agnostic support/family reliability, not dataset-specific tuning.
```
