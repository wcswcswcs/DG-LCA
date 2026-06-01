# DG-KAN v9.3.1 Legal Action-Value Observability 与 Measured Event-Driven Runtime Closure 完整实验计划

> 本计划基于 v9.3.0 `Outcome-Action Primitive Reset 与 Event-Driven Runtime Closure` 的真实执行结果制定。  
> v9.3.0 的 terminal route 是：
>
> ```text
> route = R9-OGPFeatureFail
> base_candidate = LQ-t2-h256
> success_v9300_strict_purekan_functional = False
> success_v9300_full_functional = False
> success_v9300_external_ready = False
> ```
>
> v9.3.1 不继续抢救 StableAccept，也不把 OGP feature factory 当成新的补丁堆。  
> 本轮要回答一个更根本的问题：
>
> $$
> \boxed{
> \text{当前 functional action 中已经存在 oracle-safe frontier，但为什么 legal pre-commit observables 看不出来？}
> }
> $$
>
> 如果这个问题无法通过可观测 action-value / risk primitive 解决，就必须正式进入 action primitive redesign，而不是继续调阈值。

---

# 0. 执行摘要

v9.3.0 是一次有价值的失败。它没有让 strict PureKAN functional success 过线，也没有让 system-legal controller 过线，但它明确证伪了两个旧解释，并定位出一个更深的主问题。

## 0.1 v9.3.0 真实推进了什么

v9.3.0 的进展不是 performance pass，而是科学边界推进：

```text
1. P1 canonical candidate identity freeze 过：
   candidate/action rows = 2876；
   duplicate candidate/action/event id = 0；
   payload hash missing/mismatch = 0。

2. P3 oracle frontier 强通过：
   OR1-SafeGoodOracle accepted = 494；
   precision = 1.0；
   coverage = 0.054453；
   bad/null = 0；
   support balance = 1。

3. P3.5 primitive autopsy 指向：
   legal_observability_gap_not_oracle_population_absence。

4. StableAccept patch line 被进一步证伪：
   StableAccept-only 与 registered patches 均无法形成 official accepted region。
```

这说明：当前 candidate/action population 里面不是完全没有好 action。至少在 outcome label 视角下，safe-good rows 足够多，甚至超过 official coverage 下限。

## 0.2 v9.3.0 没有推进什么

v9.3.0 没有完成以下关键 gate：

```text
1. action lifecycle official：
   action_apply_error 未测，action_lifecycle_pass = 0。

2. secondary/control outcomes：
   missing_secondary_delta_count = 28760；
   matched_control_count_per_event = 0；
   official_sample_coverage = 0.0。

3. legal OGP feature：
   best feature = OGP-C1-CandidateScore；
   AUC_bad_event = 0.617012；
   AUC_safe_good = 0.756979；
   ECE_bad = 0.206504；
   feature_cost_recorded = 0；
   ogp_feature_pass = 0。

4. minimal controller：
   best selected controller heldout accepted = 0；
   coverage = 0.0；
   decision_pass = 0。

5. measured event-driven runtime：
   reference step_ratio_q90 = 2.213009；
   empty-step controller kernels = 59868；
   controller launches per active step q90 = 9.0；
   event_driven_runtime_pass = 0。
```

所以 v9.3.0 的正确解读不是“OGP 失败所以项目没戏”，也不是“oracle 过了所以快成功了”。正确解读是：

$$
\boxed{
\text{当前 action population 有 outcome 上界，但 legal action-value/risk observability 不足。}
}
$$

## 0.3 v9.3.1 的一句话目标

v9.3.1 的目标是：

$$
\boxed{
\text{从 action-conditioned legal observables 入手，重建可部署的 value/risk/null/support/cost primitive，并同时把 event-driven runtime 做成 measured path。}
}
$$

v9.3.1 不以 full functional success 为最低目标。最低有效推进是把 route 明确落在以下四类之一：

```text
R-A: LegalActionValueObservablePass
     legal observables 足够强，可以进入 controller。

R-B: LegalObservabilityFailButOraclePresent
     当前 action population 有 oracle frontier，但 commit 前观测不到；进入 action primitive redesign。

R-C: ActionPrimitiveOracleFail
     新 action primitive 也没有 oracle frontier；说明 candidate/action generation 本身不足。

R-D: RuntimeMeasuredFail
     decision path 有希望，但 measured online runtime 过不了 system envelope。
```

---

# 1. 独立数据判断

## 1.1 StableAccept 与旧 patch 已经不能继续作为主线

P0 复现的核心数据：

```text
heldout_denominator = 9072
StableAccept accepted/safe/bad/null = 468 / 240 / 142 / 57
precision = 0.5128205128205128
coverage = 0.051587301587301584
bad_event_rate = 0.3034188034188034
null_rate = 0.12179487179487179
precision_lcb = 0.46761511726824007
bad_event_ucb = 0.3465326908369387
```

official coverage 下限要求：

$$
N_{accept,min} = \lceil 0.03 \times 9072 \rceil = 273.
$$

在 $273$ 个 accepted rows 时，gate 要求：

$$
SafeGood \ge 205,
$$

$$
BadEvent \le 13,
$$

$$
NullEvent \le 40.
$$

StableAccept 当前接受 $468$ 个 row，其中 $142$ 个是 bad。要过 gate，bad-event 不是从 $142$ 降到 $100$，而是要降到约 $13$ 的量级，同时 coverage 不能掉到 $273$ 以下。

这说明：

```text
StableAccept 的错误不是边界噪声；
也不是 rank/scoreq 修好后会自然恢复；
而是 accepted region 的 safety statistic 错了。
```

因此 v9.3.1 不再做以下工作：

```text
不再调 StableAccept threshold；
不再围绕 QR5/fallback/tie policy 做主线；
不再把 DR2/DR7 当成需要继续抢救的 controller；
不再把 coverage=0 的 heldout rule 当成“接近成功”。
```

## 1.2 oracle 强通过，但这不是 deployable success

P3 oracle frontier：

```text
OR1-SafeGoodOracle:
  accepted_count = 494
  precision = 1.0
  coverage = 0.05445326278659612
  bad_event_rate = 0.0
  null_rate = 0.0
  accepted_family_count = 90
  accepted_signal_strata_count = 17
  accepted_action_family_count = 3
  support_balance_pass = 1
  oracle_weak_pass = 1
  oracle_strong_pass = 1
```

这有两个含义。

第一，当前 frozen candidate/action population 里存在足够多 safe-good rows。official 最低需要 $273$ 个 accepted rows，而 oracle 有 $494$ 个，余量为：

$$
494 - 273 = 221.
$$

换成 oracle-safe recall 要求，一个 legal controller 至少要抓住：

$$
\frac{273}{494} = 0.5526
$$

也就是约 $55.3\%$ 的 oracle-safe rows，同时把 bad rows 控制在 $13$ 个以内。

第二，oracle 使用 outcome labels，不能部署。它只能证明“候选总体不是空的”，不能证明“commit 前能识别”。因此 v9.3.1 的核心不是再证明 oracle，而是量化：

$$
Gap_{obs} = Frontier_{oracle} - Frontier_{legal}.
$$

如果 $Gap_{obs}$ 很大，就必须修 observability 或 action primitive，而不是继续 controller search。

## 1.3 当前 legal feature 的问题不是没有 safe signal，而是 bad-tail signal 太弱

P4 best feature：

```text
feature_id = OGP-C1-CandidateScore
feature_group = C_logit_state
AUC_bad_event = 0.6170115546218488
AUC_safe_good = 0.7569791472566192
PR_AUC_bad_event_lift = 1.2997707106024358
ECE_bad = 0.20650380143905552
feature_cost_recorded = 0
feature_cost_pass = 0
signal_pass = 0
weak_signal_pass = 1
```

这个结果很关键。它说明当前 legal feature 对 safe-good 有接近门槛的排序能力：

$$
AUC_{safe} = 0.756979 \approx 0.78.
$$

但对 bad-event 的识别能力很差：

$$
AUC_{bad} = 0.617012 \ll 0.80.
$$

而 decision gate 的核心不是“多接受一些 safe”，而是“在 coverage 下限附近几乎不能接受 bad”。因此 v9.3.1 不应该只优化 safe-good AUC。必须直接优化 bad-tail observability：

```text
bad-event tail recall；
bad-event UCB calibration；
top-k accepted bad count；
bad-event ECE；
bad-risk leaveout stability。
```

AUC 只是辅助指标。真正 relevant 的 frontier 指标是：

$$
BadCount@Accept273 \le 13,
$$

$$
SafeGoodCount@Accept273 \ge 205,
$$

$$
NullCount@Accept273 \le 40.
$$

## 1.4 secondary/control outcome 缺失使“value”还没有被真正定义

v9.3.0 P2：

```text
primary_label_rows = 2876
missing_primary_label_count = 0
ambiguous_label_count = 0
label_exclusivity_violation_count = 0
missing_secondary_delta_count_official_sample = 28760
secondary_complete_candidate_count = 0
official_sample_coverage = 0.0
matched_control_branch_present = 0
matched_control_count_per_event = 0
```

这说明 primary safe/bad/null labels 可以评估 decision gate，但还不能回答：

```text
这个 action 是否真的比 AdamWParallel 好？
它改善的是 CEp99、margin、ECE、NLL、curvature，还是只是不坏？
短 horizon 有用，长 horizon 是否伤害？
它是不是 NoOp / Null 的伪安全？
```

因此 v9.3.1 必须把 secondary/control outcome materializer 提到第一优先级。没有 secondary outcomes，所谓 $V(e)$ 只是弱定义；没有 matched controls，所谓 functional causal advantage 还没有被测量。

## 1.5 action lifecycle 仍未 official

P1 中 candidate identity 已经 freeze，但 action lifecycle 未通过：

```text
candidate_lifecycle_pass = 1
action_lifecycle_pass = 0
reason = action_apply_error_not_measured_in_source_artifacts
```

这说明我们现在有了 action identity，但还没有证明：

```text
action payload 被正确 apply；
manual update 后参数差异符合预期；
no-event step 与 base AdamW 等价；
action scale / direction 没有被 runtime 或 materializer 改写；
native/apply path 与 reference/apply path 数值一致。
```

如果 action apply 本身没有测，后续所有 action-value observability 都存在解释漏洞。因此 v9.3.1 必须先补 action apply measurement。

## 1.6 runtime 是 measured implementation 缺失，不是理论不可行

runtime 当前数据：

```text
step_count = 8064
active_step_count = 1412
zero_candidate_step_count = 6652
zero_candidate_fraction = 0.8249007936507936
candidate_count_per_active_step = 2.036827195467422
kernel_count = 72576
sync_count = 8064
empty_step_controller_kernel_count = 59868
controller_launches_per_active_step_q90 = 9.0
step_ratio_q90 = 2.213009156635521
```

拆开看：

$$
6652 \times 9 = 59868.
$$

也就是说，$82.49\%$ 的 controller kernels 出现在 zero-candidate steps 上。这个问题不是深层数学问题，而是 scheduler/path implementation 问题。

但也要注意：

$$
\frac{2876}{1412}=2.0368.
$$

所以 online sequential official mode 不能硬追：

$$
AvgCandidatesPerKernel \ge 8.
$$

这个目标只适合 offline/replay materializer。online official mode 的合理目标是：

```text
zero-candidate step: 0 controller launch, 0 controller sync；
active step: q90 <= 2 controller launches；
controller sync per active step q90 <= 1；
step_ratio_q90 <= 1.50。
```

---

# 2. 是否还在正确道路上

## 2.1 高层方向仍然正确

项目仍在正确道路上，因为 v9.3.0 没有粉饰失败，也没有把 oracle/diagnostic/runtime estimate 写成 official success。它做对了三件事：

```text
1. 彻底停止把 StableAccept 当 final controller；
2. 用 oracle frontier 证明 current population 不是完全无解；
3. 把 blocker 定位到 legal observability、secondary outcome、measured runtime。
```

这些都是第一性原理层面的推进。

## 2.2 但如果下一轮继续“找 feature + 调阈值”，就会偏离正路

v9.3.1 必须避免从：

```text
StableAccept patching
```

变成：

```text
OGP feature patching
```

当前应该承认：OGP-C1 这种 state/candidate score 对 safe 有一定信号，但对 bad tail 不够。继续堆更多 weak features，可能会得到 calibration pass / heldout collapse 的重复结果。

因此下一轮要把实验对象换成：

```text
action-conditioned legal observables
```

而不是普通 candidate-state features。

## 2.3 离目标还差多远

当前离不同层级目标的距离如下。

```text
离 candidate identity clean:
  已接近完成，P1 candidate lifecycle pass。

离 action lifecycle clean:
  还缺 action apply error、reference/native apply agreement、no-event equivalence。

离 legal decision controller:
  还差一个 major blocker：legal action-value/risk observability。

离 system runtime pass:
  还差一个 measured implementation blocker：empty-step-free event-driven runtime。

离 local functional causal evidence:
  还差 system pass + LDO/LSO + official paired replay。

离 full functional / external ready:
  还差 short/full run、strong baseline、robustness、sample efficiency、continual/anti-forgetting。
```

所以不能说“马上就到目标”。更准确的是：

$$
\boxed{
\text{项目已经越过 StableAccept 局部补丁阶段，但还没进入 causal validation；目前卡在 legal observability + measured runtime 的双主 blocker。}
}
$$

---

# 3. v9.3.1 总体目标

v9.3.1 的总体目标是：

$$
\boxed{
\text{构造并验证 action-conditioned legal value/risk observables，使 current oracle-safe action frontier 在 commit 前可识别。}
}
$$

同时必须完成 measured event-driven runtime：

$$
\boxed{
\text{把 zero-candidate controller launch/sync 从 measured online path 中真正移除。}
}
$$

v9.3.1 的强目标：

$$
SystemLegalControllerPass = 1,
$$

$$
Precision_{heldout} \ge 0.75,
$$

$$
BadEventRate_{heldout} \le 0.05,
$$

$$
Coverage_{heldout} \in [0.03, 0.15],
$$

$$
NullRate_{heldout} \le 0.15,
$$

$$
StepRatio_{q90} \le 1.50,
$$

$$
MemoryRatio \le 1.05.
$$

v9.3.1 的最低有效推进目标：

```text
1. action apply error measured；
2. secondary/control outcomes materialized for official sample；
3. oracle-vs-legal observability gap quantified；
4. at least one action-conditioned legal observable has strong bad-tail signal；
5. if no such observable exists, action primitive reset is triggered with evidence；
6. measured empty-step-free runtime implemented；
7. route cleanly distinguishes feature failure, action primitive failure, runtime failure, and causal failure。
```

---

# 4. 本轮明确不做什么

v9.3.1 不做：

```text
1. 不继续调 StableAccept threshold / tie / rank / fallback；
2. 不把 StableAccept 当 final controller；
3. 不用 dataset_name 做 threshold、feature、runtime route；
4. 不把 safe-good oracle 当 deployable controller；
5. 不把 diagnostic event-driven estimate 写成 measured runtime；
6. 不在 secondary/control outcomes 缺失时打开 official paired replay；
7. 不用 heldout outcome label at commit time；
8. 不用 validation/test metric at commit time；
9. 不把 feature AUC 单独当 decision pass；
10. 不接受 heldout coverage = 0；
11. 不为了保 precision 把 coverage 降到 gate 外；
12. 不为了过 bad-event 放宽 bad-event gate；
13. 不把 feature cost missing 的 feature 放入 official controller；
14. 不用 teacher / distillation / loss modification；
15. 不把 functional update 改成 CE auxiliary loss。
```

允许做：

```text
1. dataset-level diagnostics；
2. leave-dataset-out / leave-stratum-out；
3. action-family / horizon / bucket diagnostics；
4. calibration split 上冻结 threshold；
5. diagnostic paired replay scout，但不能反向影响 controller；
6. action primitive redesign，但必须保持 Clean PureKAN / manual update / standard CE / no teacher；
7. offline replay materializer batching，但不能作为 online official runtime。
```

---

# 5. 核心假设

## H1：v9.3.0 的主 blocker 是 legal observability gap，不是 oracle frontier absence

H1 认为当前 candidate/action population 有 enough safe-good rows，但 current legal features 无法识别它们。

H1 成立标准：

```text
OR1 oracle frontier pass；
oracle accepted_count >= 273；
oracle precision >= 0.90；
oracle bad_event <= 0.02；
oracle null_rate <= 0.10；

同时：

best legal score frontier 在 heldout 上无法满足：
safe_good_count@273 >= 205
bad_count@273 <= 13
null_count@273 <= 40。
```

H1 失败标准：

```text
新的 action lifecycle / secondary outcome materializer 发现 primary oracle label 定义有误；
或 oracle frontier 在 secondary/control outcomes 下不再成立。
```

若 H1 失败，必须回到 outcome label/action application audit，而不是继续 features。

## H2：current legal features 失败是因为它们不是 action-conditioned effect observables

H2 认为 OGP-C1 CandidateScore 能预测一部分 safe-good，但不能预测 bad tail，因为它主要是 state/candidate score，而不是 action effect estimate。

H2 成立标准：

```text
state-only features:
  AUC_bad_event < 0.70
  BadCount@Accept273 > 13
  ECE_bad > 0.10

action-conditioned features:
  AUC_bad_event >= 0.80
  BadCount@Accept273 <= 13
  ECE_bad <= 0.05
```

H2 失败标准：

```text
state-only features already become sufficient once secondary/action lifecycle is fixed。
```

## H3：secondary/control outcomes will reveal whether safe-good oracle is truly useful, not merely non-bad

H3 认为 primary safe-good labels may be necessary but not sufficient. We need $V(e)$ against controls.

H3 成立标准：

```text
secondary_complete_candidate_count >= official sample target；
matched_control_count_per_event >= 3；
real_beats_adamwparallel and real_beats_bestlr available；
value metrics distinguish useful vs null-safe rows。
```

H3 失败标准：

```text
secondary materializer too expensive or inconsistent；
or primary safe-good rows mostly fail against controls。
```

## H4：measured empty-step-free runtime can reduce controller overhead enough to reach system envelope

H4 成立标准：

```text
empty_step_controller_kernel_count = 0；
empty_step_controller_sync_count = 0；
controller_launches_per_active_step_q90 <= 2；
controller_syncs_per_active_step_q90 <= 1；
step_ratio_q90 <= 1.50；
memory_ratio <= 1.05；
no_event_preservation_pass = 1。
```

H4 失败标准：

```text
measured implementation still step_ratio_q90 > 1.50；
or no-event semantics / AdamW equivalence is broken。
```

## H5：如果 legal observability still fails, the correct pivot is action primitive redesign, not controller tuning

H5 成立标准：

```text
oracle frontier pass；
action apply measured；
secondary outcomes ready；
multiple legal action-conditioned observables fail bad-tail frontier；
coverage/precision/bad gates fail across cross-fit/LDO/LSO。
```

Then route:

```text
R-B-LegalObservabilityFailButOraclePresent
```

Next step:

```text
redesign action primitive to expose observable value/risk, not tune thresholds.
```

---

# 6. 数据合同

## 6.1 action lifecycle contract

Every action row must record:

```text
event_id
candidate_id
action_id
action_schema_version
primitive_id
payload_hash
payload_tensor_hash
apply_path
reference_apply_hash
native_apply_hash
theta_before_hash
theta_after_reference_hash
theta_after_native_hash
delta_theta_reference_norm
delta_theta_native_norm
apply_error_linf
apply_error_l2
apply_error_relative
action_scale
action_norm
action_role_entropy
action_layer_span
no_event_step_flag
base_adamw_equivalence_error
optimizer_state_equivalence_error
```

Pass:

```text
duplicate_action_id_count = 0
action_payload_missing_count = 0
apply_error_linf <= 1e-6 or registered tolerance
apply_error_relative <= 1e-5
base_adamw_equivalence_on_zero_candidate_steps = 1
event_scheduler_does_not_reorder_training_batches = 1
action_lifecycle_pass = 1
```

## 6.2 secondary/control outcome contract

Each measured candidate/action row must record:

```text
candidate_id
action_id
event_id
dataset
seed
step
family_id
horizon
bucket_id
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
local_lipschitz_delta
basis_usage_entropy_delta
functional_channel_entropy_delta
acc_delta
loss_auc_delta_step
loss_auc_delta_time
real_beats_adamwparallel
real_beats_bestlr
real_beats_noop
real_beats_random
safe_good_label
bad_event_label
null_event_label
task_safe_label
useful_label
outcome_runtime_ms
outcome_materializer_cost_ratio
```

Branches:

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledFunctionalPayload
ShuffledBranchDelta
ShuffledActionScore
ShuffledRiskScore
ShuffledSupportScore
```

Official sample coverage:

```text
all StableAccept accepted rows；
all OGP/controller selected rows；
oracle-safe stratified sample；
matched rejected rows；
family × horizon × bucket minimum sample count。
```

Pass:

```text
missing_primary_label_count = 0
missing_secondary_delta_count_official_sample = 0
matched_control_count_per_event >= 3
official_sample_coverage >= 0.30
or each family × horizon × bucket has n >= 20 where available
p2_cost_ratio <= registered budget
```

## 6.3 legal action-value observable contract

Each observable must record:

```text
observable_id
observable_group
uses_dataset_name
uses_outcome_at_commit
uses_future_step
uses_validation_or_test
uses_source_measured_gap
uses_formula_proxy_for_official
feature_compute_time_ms_q90
feature_memory_ratio
commit_time_available
materialized_online_path
missing_rate
AUC_bad_event
AUC_safe_good
AUC_null_event
PR_AUC_bad_lift
ECE_bad
Brier_bad
BadCount@Accept273
SafeGoodCount@Accept273
NullCount@Accept273
leave_seed_auc_drop
leave_dataset_auc_drop
leave_stratum_auc_drop
calibration_to_heldout_drift
```

Feature legality pass:

```text
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
uses_validation_or_test = 0
feature_compute_time_ms_q90 recorded
feature_memory_ratio recorded
commit_time_available = 1
materialized_online_path = 1
missing_rate <= 0.01
```

## 6.4 online runtime contract

Each step must record:

```text
step_id
dataset
seed
active_step_flag
zero_candidate_step_flag
candidate_count_in_step
accepted_count_in_step
controller_kernel_launch_count
controller_sync_count
allocation_count
controller_launches_per_active_step
controller_syncs_per_active_step
candidate_pack_time_ms
observable_compute_time_ms
risk_value_score_time_ms
accept_decision_time_ms
payload_materialize_time_ms
payload_apply_time_ms
audit_time_ms_outside_timed
base_train_step_time_ms
controller_extra_time_ms
total_step_time_ms
mlp_step_time_ms
step_ratio
peak_memory_mb
memory_ratio
no_event_preservation_error
optimizer_state_equivalence_error
```

Official online runtime pass:

```text
runtime_mode = online_sequential_official
empty_step_controller_kernel_count = 0
empty_step_controller_sync_count = 0
controller_launches_per_active_step_q90 <= 2
controller_syncs_per_active_step_q90 <= 1
allocation_count_per_active_step <= 0.05
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
no_event_preservation_pass = 1
base_adamw_equivalence_on_zero_candidate_steps = 1
```

---

# 7. v9.3.1 实验阶段

---

## P0：v9.3.0 boundary reanalysis and gap quantification

### 目标

不信任 route 名称，只从 landed metrics 重新计算三类 gap：

```text
decision gap；
observability gap；
runtime implementation gap。
```

### 假设

H1/P0：当前 blocker 是 legal observability + runtime measured implementation，而不是 candidate population absence。

### 必须记录

```text
candidate_count
action_count
event_count
heldout_denominator
StableAccept_accepted_count
StableAccept_safe_count
StableAccept_bad_count
StableAccept_null_count
StableAccept_precision
StableAccept_coverage
StableAccept_bad_event_rate
min_accepted_for_coverage
safe_needed_at_min_coverage
bad_allowed_at_min_coverage
null_allowed_at_min_coverage
oracle_accepted_count
oracle_precision
oracle_coverage
oracle_bad_event
oracle_null_rate
oracle_safe_recall_needed
best_legal_feature_id
best_legal_feature_auc_bad
best_legal_feature_auc_safe
best_legal_feature_ece_bad
best_legal_feature_cost_recorded
current_best_controller_coverage_heldout
secondary_missing_count
matched_control_count_per_event
step_count
active_step_count
zero_candidate_step_count
empty_step_controller_kernel_count
controller_launches_per_active_step_q90
step_ratio_q90
```

### 判断标准

P0 pass:

```text
v9.3.0 metrics reproduced；
oracle-safe recall needed computed；
BadCount@Accept273 target computed；
runtime lower-bound targets computed；
route inputs frozen。
```

### 可视化

```text
p0_decision_gate_gap_bar.svg
p0_oracle_vs_legal_frontier.svg
p0_feature_bad_vs_safe_auc.svg
p0_runtime_kernel_decomposition.svg
p0_progress_ladder_v9280_to_v9300.svg
```

---

## P1：action lifecycle apply-error closure

### 目标

把 action lifecycle 从 “identity exists” 推到 “apply measured and official”。

### 假设

H1/P1：action identity freeze 不等于 action lifecycle pass；必须证明 payload apply 和 no-event semantics 正确。

### 实验设计

新增 runner：

```text
experiments/run_v9310_action_lifecycle_apply_error_closure.py
```

对 frozen action table 执行：

```text
1. reference manual apply；
2. native/manual fast apply；
3. theta_before/theta_after hash；
4. action delta norm comparison；
5. optimizer state comparison；
6. zero-candidate step no-event preservation；
7. event scheduler batch-order audit。
```

### 必须记录

```text
action_id
candidate_id
event_id
apply_path
payload_hash
theta_before_hash
theta_after_reference_hash
theta_after_native_hash
delta_theta_reference_norm
delta_theta_native_norm
apply_error_linf
apply_error_l2
apply_error_relative
optimizer_m_error_linf
optimizer_v_error_linf
no_event_preservation_error
scheduler_batch_order_changed
action_apply_error_measured
```

### 判断标准

P1 pass:

```text
action_apply_error_measured = 1
apply_error_linf <= tolerance
apply_error_relative <= tolerance
optimizer_m_error_linf <= tolerance
optimizer_v_error_linf <= tolerance
no_event_preservation_pass = 1
scheduler_batch_order_changed = 0
action_lifecycle_pass = 1
```

### 可视化

```text
p1_apply_error_hist.svg
p1_reference_native_delta_norm_scatter.svg
p1_no_event_preservation_error.svg
p1_action_norm_by_family_horizon.svg
```

---

## P2：secondary/control outcome materializer v2

### 目标

补齐 $V(e)$ 的真实定义。P2 只 materialize，不做 controller，不用结果调 threshold。

### 假设

H3：没有 secondary/control outcomes，就不能区分 useful-safe、null-safe、short-horizon good、long-horizon bad。

### 实验设计

新增 runner：

```text
experiments/run_v9310_secondary_control_outcome_materializer_v2.py
```

采样策略：

```text
Tier A official-minimum:
  StableAccept accepted rows；
  current OGP selected rows；
  oracle-safe stratified sample；
  matched rejected rows。

Tier B stratified sample:
  family × horizon × bucket 每格 n >= 20；
  如果格子不足则全取并记录 insufficient support。

Tier C optional full:
  full 2876 candidate/action rows。
```

Branches:

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledFunctionalPayload
ShuffledBranchDelta
ShuffledActionScore
ShuffledRiskScore
ShuffledSupportScore
```

Horizons:

```text
20, 80, 240, 640
```

### 必须记录

```text
candidate_id
action_id
event_id
dataset
seed
step
family_id
horizon
bucket_id
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
local_lipschitz_delta
basis_usage_entropy_delta
functional_channel_entropy_delta
acc_delta
loss_auc_delta_step
loss_auc_delta_time
time_to_target_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_noop
real_beats_random
task_safe_label
useful_label
bad_event_label
null_event_label
safe_good_label
matched_control_group_id
outcome_runtime_ms
materializer_step_ratio_q90
```

### 判断标准

P2 pass:

```text
missing_secondary_delta_count_official_sample = 0
matched_control_count_per_event >= 3
official_sample_coverage >= 0.30
or family × horizon × bucket n >= 20 for all supported cells
outcome_materializer_cost_ratio <= registered budget
horizon_consistency_audit_pass = 1
branch_consistency_audit_pass = 1
```

### 可视化

```text
p2_secondary_missing_before_after.svg
p2_real_vs_control_paired_delta.svg
p2_value_bad_null_phase_diagram.svg
p2_horizon_consistency_heatmap.svg
p2_branch_outcome_pareto.svg
p2_materializer_cost_waterfall.svg
```

---

## P3：oracle frontier revalidation under secondary/control outcomes

### 目标

检查 v9.3.0 的 primary-label oracle 是否仍然是 useful frontier，而不只是 primary-safe frontier。

### 假设

H3/P3：OR1 safe-good rows 中应存在足够多 rows 在 secondary/control outcomes 下仍 useful。

### Oracle definitions

Primary oracle:

$$
OraclePrimary(e)=1
\iff SafeGood(e)=1 \land BadEvent(e)=0 \land NullEvent(e)=0.
$$

Value oracle:

$$
V(e)=
-w_1 CEp99_{\Delta}(e)
+w_2 MarginP10_{\Delta}(e)
-w_3 ECE_{\Delta}(e)
-w_4 NLL_{\Delta}(e)
-w_5 Curvature_{\Delta}(e)
+w_6 LossAUC_{\Delta}(e).
$$

Control-robust oracle:

$$
OracleControl(e)=1
\iff
OraclePrimary(e)=1
\land V(e)>0
\land RealBeatsAdamWParallel(e)=1
\land RealBeatsBestLR(e)=1.
$$

### 必须记录

```text
oracle_id
oracle_type
accepted_count
precision
coverage
bad_event_rate
null_rate
value_mean
value_lcb
CEp99_delta_mean
margin_p10_delta_mean
ECE_delta_mean
NLL_delta_mean
curvature_delta_mean
beats_adamwparallel_rate
beats_bestlr_rate
accepted_family_count
accepted_signal_strata_count
accepted_action_family_count
max_family_share
max_stratum_share
support_balance_pass
```

### 判断标准

P3 strong useful oracle pass:

```text
coverage >= 0.03
precision >= 0.90
bad_event_rate <= 0.02
null_rate <= 0.10
value_lcb > 0
beats_adamwparallel_rate >= 0.60
beats_bestlr_rate >= 0.60
support_balance_pass = 1
```

P3 weak useful oracle pass:

```text
coverage >= 0.03
precision >= 0.75
bad_event_rate <= 0.05
null_rate <= 0.15
value_mean > 0
```

P3 fail:

```text
primary oracle pass but useful/control oracle fail。
```

If P3 fail, route:

```text
R-C-ActionPrimitivePrimarySafeButNotUseful
```

Then action primitive redesign is required.

### 可视化

```text
p3_primary_vs_value_oracle_overlap.svg
p3_oracle_value_risk_pareto.svg
p3_control_beat_rate_by_horizon.svg
p3_oracle_support_balance.svg
p3_oracle_family_action_family_matrix.svg
```

---

## P4：oracle-to-legal observability gap analysis

### 目标

量化 legal features 为什么抓不住 oracle-safe frontier。P4 不训练新 controller，只做 gap decomposition。

### 假设

H2：current feature failure is dominated by bad-tail observability gap.

### 必须记录

```text
feature_id
feature_group
oracle_overlap_at_top273
safe_recall_at_top273
bad_count_at_top273
null_count_at_top273
safe_good_count_at_top273
precision_at_top273
bad_event_at_top273
feature_auc_bad
feature_auc_safe
feature_ece_bad
oracle_positive_score_distribution
bad_negative_score_distribution
overlap_error_type
```

Error types:

```text
GE1-safe_oracle_low_score
GE2-bad_event_high_score
GE3-null_event_high_score
GE4-family_support_gap
GE5-horizon_tail_gap
GE6-action_apply_uncertainty
GE7-secondary_value_disagreement
GE8-cost_unmeasured
```

### 判断标准

P4 pass:

```text
oracle-to-legal gap attribution fraction >= 0.90
bad-event high-score attribution fraction measured
missed-oracle attribution fraction measured
next observable target selected
```

### 可视化

```text
p4_oracle_legal_score_scatter.svg
p4_top273_confusion_bar.svg
p4_gap_error_sankey.svg
p4_feature_score_by_label_violin.svg
p4_bad_tail_high_score_examples.svg
```

---

## P5：action-conditioned legal observable factory

### 目标

构造真正与 action effect 相关的 legal observables。P5 不是大杂烩 feature pile；每组 observable 都必须有明确机制假设、成本记录和 ablation。

### 假设

H2：action-conditioned observables 比 state-only/candidate-only features 更能预测 bad-event tail。

### Observable groups

#### O1：manual gradient × action alignment

记录：

```text
grad_dot_action
cos_action_negative_grad
cos_action_adamw_delta
action_norm_over_adamw_norm
rolewise_grad_conflict_score
tail_sample_grad_conflict_score
```

线性化 CE 变化：

$$
\widehat{\Delta CE}_{lin}(e)=\nabla_{\theta}CE(\theta_t)^T\Delta\theta_{action}(e).
$$

Good sign:

$$
\widehat{\Delta CE}_{lin}(e)<0.
$$

#### O2：logit-space action effect

记录：

```text
logit_delta_norm
true_class_logit_delta
wrong_class_max_logit_delta
margin_delta_linearized
CE_delta_linearized
CEp99_delta_linearized
hard_tail_margin_delta_linearized
```

Tail risk:

$$
TailRisk_{logit}(e)=
a_1 WrongClassMaxDelta
-a_2 TrueClassDelta
-a_3 MarginDelta_{tail}
+a_4 CEp99Delta_{lin}.
$$

#### O3：curvature / trust-region observable

记录：

```text
action_norm
adamw_norm
curvature_proxy
gauss_newton_diag_action_energy
linearization_error_estimate
trust_region_ratio
```

Trust score:

$$
Trust(e)=
\frac{-\widehat{\Delta CE}_{lin}(e)}
{\epsilon + \widehat{\Delta CE}_{quad}^{+}(e)}.
$$

#### O4：payload geometry and role entropy

记录：

```text
payload_norm
payload_tail_norm
payload_role_entropy
payload_layer_entropy
payload_sparsity
functional_channel_entropy
rolewise_update_balance
orthogonal_component_ratio
```

Hypothesis:

```text
over-concentrated or high-tail payloads create bad-event tail。
```

#### O5：empirical-Bayes support/reliability observable

记录：

```text
family_count
horizon_count
action_family_count
feature_neighborhood_count
effective_sample_size
empirical_bayes_bad_mean
empirical_bayes_bad_ucb
empirical_bayes_value_lcb
shift_risk
```

Bad UCB:

$$
BadUCB_{EB}(e)=
\hat p_{bad}(e)
+
z_{\alpha}
\sqrt{
\frac{\hat p_{bad}(e)(1-\hat p_{bad}(e))}
{n_{eff}(e)+\epsilon}
}
+
\lambda_{shift}ShiftRisk(e).
$$

#### O6：null-risk observable

记录：

```text
predicted_value_abs
payload_norm
branch_delta_norm
value_lcb
null_ucb
no_op_similarity
```

Null UCB:

$$
NullUCB(e)=P(|V(e)|<\epsilon_V \mid x_e).
$$

#### O7：online cost observable

记录：

```text
observable_compute_time_ms_q90
payload_apply_time_ms_q90
controller_launch_cost_estimate
kernel_shape_id
active_step_flag
candidate_count_in_step
memory_ratio_contribution
```

Cost gate:

$$
C(e) \le C_{max}.
$$

### Minimality rule

Each candidate controller may use at most:

```text
5 observable groups
```

and must report:

```text
single-group performance；
leave-one-group-out ablation；
monotone sign consistency；
calibration-to-heldout drift；
feature cost。
```

No opaque score with 20 un-attributed features may enter official.

### 必须记录

```text
observable_id
observable_group
mechanism_hypothesis
uses_dataset_name
uses_outcome_at_commit
uses_future_step
feature_compute_time_ms_q90
feature_memory_ratio
commit_time_available
materialized_online_path
missing_rate
AUC_bad_event
AUC_safe_good
AUC_null_event
PR_AUC_bad_lift
ECE_bad
Brier_bad
SafeGoodCount@Accept273
BadCount@Accept273
NullCount@Accept273
leave_seed_auc_drop
leave_dataset_auc_drop
leave_stratum_auc_drop
monotone_sign_pass
cost_pass
```

### 判断标准

P5 observable pass:

```text
legality pass = 1
cost pass = 1
AUC_bad_event >= 0.80
ECE_bad <= 0.05
BadCount@Accept273 <= 13
SafeGoodCount@Accept273 >= 205
NullCount@Accept273 <= 40
leave_dataset_auc_drop <= 0.07
leave_stratum_auc_drop <= 0.10
```

P5 weak pass:

```text
AUC_bad_event >= 0.75
or SafeGoodCount@Accept273 >= 205
but bad/null/cost/leaveout not pass。
```

Weak pass不能进入 official controller，只能进入 observable redesign or action primitive reset。

### 可视化

```text
p5_observable_auc_bar.svg
p5_bad_tail_frontier_top273.svg
p5_calibration_curve_bad.svg
p5_value_risk_null_phase_diagram.svg
p5_feature_cost_vs_signal_pareto.svg
p5_ablation_heatmap.svg
p5_leaveout_drop_heatmap.svg
```

---

## P6：cross-fitted minimal legal action-value controller

### 目标

用 P5 survivor 构造 final controller candidate。所有阈值在 calibration folds 冻结，再在 heldout / LDO / LSO 上评估。

### Controller form

$$
Accept(e)=1
\iff
LCB(V(e))>0
\land
UCB(B(e))\le \tau_b
\land
UCB(N(e))\le \tau_n
\land
LCB(S(e))\ge \tau_s
\land
C(e)\le C_{max}.
$$

Monotone score candidate:

$$
S(e)=
a_1 V_{LCB}(e)
-a_2 Bad_{UCB}(e)
-a_3 Null_{UCB}(e)
+a_4 Support_{LCB}(e)
-a_5 Cost(e),
$$

with:

```text
a_i >= 0
feature groups <= 5
thresholds frozen
no dataset branch
```

### Splits

```text
K-fold seed split；
leave-dataset-out；
leave-stratum-out；
leave-action-family-out diagnostic；
leave-horizon-out diagnostic。
```

### 必须记录

```text
controller_id
observable_set
feature_group_count
monotone_constraints
thresholds
calibration_fold
heldout_fold
split_type
heldout_entity
dataset_name_used
accepted_count_cal
accepted_count_heldout
precision_cal
coverage_cal
bad_event_cal
null_rate_cal
value_mean_cal
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
value_mean_heldout
precision_lcb
bad_event_ucb
null_event_ucb
accepted_family_count
accepted_signal_strata_count
accepted_action_family_count
max_family_share
max_stratum_share
oracle_overlap
stableaccept_overlap
step_cost_estimate
feature_cost_pass
```

### 判断标准

P6 decision pass:

$$
Precision_{heldout}\ge0.75,
$$

$$
Coverage_{heldout}\in[0.03,0.15],
$$

$$
BadEventRate_{heldout}\le0.05,
$$

$$
NullRate_{heldout}\le0.15,
$$

$$
Precision_{LCB}\ge0.75,
$$

$$
BadEventRate_{UCB}\le0.05.
$$

Support pass:

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
accepted_action_family_count >= 2
max_family_share <= 0.50
max_stratum_share <= 0.60
coverage_heldout > 0 in every main fold
```

Cross-fit pass:

```text
bad-event gate pass in at least 2/3 seed folds；
macro decision score pass；
no dataset-specific branch。
```

### 可视化

```text
p6_controller_precision_bad_coverage_frontier.svg
p6_calibration_to_heldout_drift.svg
p6_crossfit_fold_matrix.svg
p6_observable_ablation_controller.svg
p6_oracle_overlap_vs_bad_rate.svg
p6_support_balance_sunburst.svg
```

---

## P7：decision failure autopsy v4

### 目标

如果 P6 未过，必须知道是 observable 不够、controller 形式不够、action primitive 不可观测，还是 secondary/control oracle 不稳定。

### Failure modes

```text
DF1-primary_oracle_absent
DF2-control_oracle_absent
DF3-legal_observable_bad_tail_gap
DF4-legal_observable_value_gap
DF5-null_risk_gap
DF6-support_collapse
DF7-heldout_coverage_zero
DF8-action_apply_uncertainty
DF9-feature_cost_infeasible
DF10-family_horizon_shift
DF11-dataset_leaveout_instability
DF12-controller_form_insufficient
DF13-action_primitive_unobservable
```

### 必须记录

```text
failed_controller_id
bad_accepted_count
null_accepted_count
missed_oracle_safe_count
coverage_lost_count
failure_mode
failure_submode
observable_values
oracle_primary_label
oracle_control_label
legal_score
support_stats
family_id
action_family_id
horizon
bucket_id
seed
dataset
```

### 判断标准

P7 pass:

```text
bad accepted attribution fraction >= 0.95
missed oracle-safe attribution fraction >= 0.90
coverage collapse attribution fraction >= 0.90
next route selected unambiguously
```

### 可视化

```text
p7_failure_sankey.svg
p7_missed_oracle_vs_bad_accepted.svg
p7_bad_tail_observability_gap.svg
p7_support_collapse_by_family_horizon.svg
```

---

## P8：action primitive redesign branch

### 目标

如果 P5/P6 证明 current action population has oracle frontier but remains legally unobservable，不能继续 controller tuning；必须产生更可观测的 action primitive。

### Trigger

P8 is triggered if:

```text
P3 useful/control oracle pass = 1
P5 observable pass = 0
P6 decision pass = 0
P7 primary failure = DF13-action_primitive_unobservable
```

### Candidate action primitives

#### AP0：current action reference

保留 current action primitive，作为 baseline。

#### AP1：gradient-aligned shrinkage action

目标：让 action direction 与 manual CE descent 对齐，降低 bad-tail。

$$
\Delta\theta_{AP1}
=
\alpha(e)
\cdot
Proj_{safe}
\left(
\Delta\theta_{func}
\right),
$$

where $\alpha(e)$ is legal and commit-time available.

记录：

```text
alignment_gain
bad_tail_reduction
oracle_frontier_change
legal_observable_auc_change
```

#### AP2：tail-safe margin repair action

目标：action 只作用于 hard-tail margin 改善方向。

$$
\Delta\theta_{AP2}
=
\alpha(e)
\cdot
\Delta\theta_{margin-tail}.
$$

记录：

```text
margin_p10_linearized_gain
wrong_conf_delta
tail_CE_delta
```

#### AP3：orthogonal residual action

目标：减少与 AdamW 主更新冲突，同时保留 functional channel。

$$
\Delta\theta_{AP3}
=
\Delta\theta_{func}
-
Proj_{\Delta\theta_{AdamW}}
(\Delta\theta_{func}).
$$

记录：

```text
orthogonal_component_ratio
cos_adamw_after_projection
value_change
bad_change
```

#### AP4：trust-region action scaling

目标：使用 curvature / linearization reliability 控制 action magnitude。

$$
\Delta\theta_{AP4}
=
\alpha_{trust}(e)\Delta\theta_{func},
$$

with:

$$
\alpha_{trust}(e)=
\min
\left(
1,
\frac{\rho}{\epsilon + \widehat{\Delta CE}_{quad}^{+}(e)}
\right).
$$

#### AP5：null-aware gated action

目标：降低 null/no-op accepted rows。

$$
\Delta\theta_{AP5}
=
\mathbb{1}[NullUCB(e)\le\tau_n]
\cdot
\Delta\theta_{func}.
$$

### For each AP candidate, run

```text
candidate/action generation；
action lifecycle apply-error；
primary labels；
secondary/control outcomes on sample；
oracle frontier；
legal observable frontier；
runtime cost smoke。
```

### 必须记录

```text
action_primitive_id
candidate_count
action_count
candidate_rate
oracle_primary_accepted_count
oracle_primary_precision
oracle_primary_coverage
oracle_primary_bad_event
oracle_control_accepted_count
oracle_control_value_lcb
best_legal_auc_bad
best_legal_auc_safe
BadCount@Accept273
SafeGoodCount@Accept273
NullCount@Accept273
action_apply_error
step_cost_estimate
memory_ratio_estimate
```

### 判断标准

P8 action primitive survivor:

```text
oracle weak/strong frontier pass；
legal observable pass improves over AP0；
BadCount@Accept273 <= 13；
SafeGoodCount@Accept273 >= 205；
action_apply_error pass；
runtime cost estimate not worse than registered budget。
```

If all AP candidates fail oracle frontier:

```text
R-C-ActionPrimitiveOracleFail
```

If oracle passes but legal observability still fails:

```text
R-B-LegalObservabilityFailButOraclePresent
```

### 可视化

```text
p8_action_primitive_oracle_frontier.svg
p8_action_primitive_legal_observable_frontier.svg
p8_action_apply_error_by_primitive.svg
p8_action_value_risk_pareto.svg
p8_ap0_vs_ap_survivor_overlap.svg
```

---

## P9：measured online event-driven runtime implementation

### 目标

把 runtime 从 diagnostic estimate 变成 measured official online path。

### Runtime candidates

#### RT0：v9.3.0 reference

```text
fixed per-step launch/sync
step_ratio_q90 = 2.213009
empty_step_controller_kernel_count = 59868
controller_launches_per_active_step_q90 = 9.0
```

#### RT1：measured zero-candidate skip

```text
zero-candidate step:
  no controller kernel launch；
  no controller sync；
  base AdamW path unchanged。
```

#### RT2：measured active-step fused score/accept

```text
active step:
  one fused kernel for feature lookup + score + accept mask。
```

#### RT3：measured two-kernel active apply

```text
kernel 1: score/accept；
kernel 2: payload materialize/apply。
```

#### RT4：persistent workspace

```text
preallocate candidate/action/observable/accept buffers；
no per-step allocation。
```

#### RT5：audit outside timed path

```text
post-step exact audit subset；
not in timed controller path；
audit cannot alter accept decision。
```

### 必须记录

```text
runtime_candidate_id
runtime_mode
step_count
active_step_count
zero_candidate_step_count
candidate_count
accepted_count
empty_step_controller_kernel_count
empty_step_controller_sync_count
controller_kernel_launch_count
controller_sync_count
controller_launches_per_active_step_mean
controller_launches_per_active_step_q90
controller_syncs_per_active_step_mean
controller_syncs_per_active_step_q90
allocation_count
allocation_count_per_active_step
candidate_pack_time_ms_q90
observable_compute_time_ms_q90
score_accept_time_ms_q90
payload_apply_time_ms_q90
audit_outside_timed_ms
total_step_time_ms_q50
total_step_time_ms_q90
active_step_time_ms_q90
empty_step_time_ms_q90
mlp_step_time_ms_q90
step_ratio_q90
active_step_ratio_q90
memory_ratio
no_event_preservation_error
base_adamw_equivalence_error
accept_disagreement_count
payload_apply_error_max
```

### 判断标准

P9 runtime pass:

```text
runtime_mode = online_sequential_official
empty_step_controller_kernel_count = 0
empty_step_controller_sync_count = 0
controller_launches_per_active_step_q90 <= 2
controller_syncs_per_active_step_q90 <= 1
allocation_count_per_active_step <= 0.05
accept_disagreement_count = 0
payload_apply_error_max <= tolerance
no_event_preservation_pass = 1
base_adamw_equivalence_on_zero_candidate_steps = 1
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

### 可视化

```text
p9_runtime_mode_comparison.svg
p9_empty_vs_active_step_timing.svg
p9_kernel_sync_reduction.svg
p9_launches_per_active_step_hist.svg
p9_runtime_waterfall.svg
p9_step_ratio_distribution.svg
p9_no_event_equivalence_error.svg
```

---

## P10：system integration gate

### 目标

组合 P6/P8 decision survivor 与 P9 runtime survivor。P10 是 v9.3.1 的 official system gate。

### 必须记录

```text
system_candidate_id
controller_id
action_primitive_id
runtime_candidate_id
observable_set
thresholds
candidate_count
action_count
event_count
accepted_count
candidate_rate
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
value_mean_heldout
precision_lcb
bad_event_ucb
accepted_family_count
accepted_signal_strata_count
accepted_action_family_count
max_family_share
max_stratum_share
step_ratio_q90
active_step_ratio_q90
memory_ratio
controller_launches_per_active_step_q90
empty_step_controller_kernel_count
accept_disagreement_count
payload_apply_error_max
action_lifecycle_pass
secondary_outcome_ready
materialized_system_path
diagnostic_derived_from_measured_components
projection_used
source_measured_gap_used
formula_proxy_used
dataset_name_used
official_eligible
system_legal_controller_pass
```

### 判断标准

P10 system pass:

```text
official_eligible = 1
system_legal_controller_pass = 1
action_lifecycle_pass = 1
secondary_outcome_ready = 1
materialized_system_path = 1
diagnostic_derived_from_measured_components = 0
projection_used = 0
source_measured_gap_used = 0
formula_proxy_used = 0
dataset_name_used = 0
```

Decision gates:

$$
Precision_{heldout}\ge0.75,
$$

$$
Coverage_{heldout}\in[0.03,0.15],
$$

$$
BadEventRate_{heldout}\le0.05,
$$

$$
NullRate_{heldout}\le0.15,
$$

$$
Precision_{LCB}\ge0.75,
$$

$$
BadEventRate_{UCB}\le0.05.
$$

Runtime gates:

$$
StepRatio_{q90}\le1.50,
$$

$$
MemoryRatio\le1.05.
$$

### 可视化

```text
p10_system_quality_cost_frontier.svg
p10_official_gate_dashboard.svg
p10_controller_runtime_pareto.svg
p10_failure_reason_matrix.svg
```

---

## P11：parallel diagnostic causal scout

### 目标

加快实验，但不让 diagnostic 反向污染 controller。P11 可以和 P5/P6/P9 并行准备。

### 隔离规则

```text
diagnostic paired replay cannot affect:
  feature selection；
  controller threshold；
  route decision before P10；
  official pass/fail。
```

`route_decision.json` must record:

```text
diagnostic_downstream_used_for_controller = 0
```

### 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
horizons = 20,80,240
controllers = top2 legal observable controllers + AP survivor + StableAccept negative control
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, shuffled controls
```

### 必须记录

```text
controller_id
action_primitive_id
dataset
seed
horizon
branch
accepted_count
coverage
bad_event_rate
null_rate
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
loss_auc_delta_step
loss_auc_delta_time
beats_adamwparallel
beats_bestlr
task_safe
step_ratio_q90
memory_ratio
status = diagnostic_not_official
```

### 判断标准

P11 promising diagnostic:

```text
RealFunctional beats AdamWParallel in >= 50% macro slices；
RealFunctional beats bestLR in >= 50% macro slices；
task_safe holds: Acc_real >= Acc_adamw - 0.005；
shuffled controls do not match RealFunctional。
```

### 可视化

```text
p11_diagnostic_macro_beat_rate.svg
p11_branch_delta_pareto.svg
p11_shuffle_control_matrix.svg
p11_task_safety_by_slice.svg
```

---

## P12：official leave-dataset-out / leave-stratum-out

### 目标

只有 P10 pass 后打开。证明 controller 不是 pooled calibration artifact，也不是 dataset-specific route。

### 设置

Leave-dataset-out:

```text
calibrate MNIST + Fashion, evaluate KMNIST
calibrate MNIST + KMNIST, evaluate Fashion
calibrate Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out:

```text
calibrate all but one signal stratum
evaluate held-out stratum
```

Leave-action-family-out diagnostic:

```text
calibrate all but one action family
evaluate held-out action family
```

### 必须记录

```text
split_type
heldout_entity
controller_id
action_primitive_id
runtime_candidate_id
candidate_rate
precision
coverage
bad_event_rate
null_rate
precision_lcb
bad_event_ucb
value_mean
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
dataset_name_used
support_balance_pass
```

### 判断标准

LDO pass:

```text
at least 2/3 held-out datasets pass decision gates；
no held-out dataset has bad_event_rate > 0.10；
coverage > 0 for all held-out datasets；
dataset_name_used = 0。
```

LSO pass:

```text
>=70% held-out strata task-safe；
macro bad_event_rate <= 0.05；
macro precision >= 0.75。
```

### 可视化

```text
p12_leave_dataset_out_matrix.svg
p12_leave_stratum_out_matrix.svg
p12_leave_action_family_out_diagnostic.svg
p12_dataset_diagnostic_no_tuning_audit.svg
```

---

## P13：official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。只有 P10/P12 pass 后 official 打开。

### 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches =
  RealFunctional
  AdamWOnly
  AdamWParallel
  bestLR
  NoOp
  Random
  ShuffledFunctionalPayload
  ShuffledBranchDelta
  ShuffledActionScore
  ShuffledOGPValueScore
  ShuffledOGPRiskScore
  ShuffledOGPSupportScore
  ShuffledOutcomeLabel
  ShuffledRuntimePath
  ShuffledActionPrimitive
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
action_primitive_id
dataset
seed
horizon
signal_stratum
event_family
action_family
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
loss_auc_delta_step
loss_auc_delta_time
time_to_target_delta
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
BeatRate_{macro,Real\ vs\ AdamWParallel}\ge0.60.
$$

$$
BeatRate_{macro,Real\ vs\ bestLR}\ge0.60.
$$

Task safety:

$$
Acc_{each\ slice,Real}\ge Acc_{AdamW}-0.005.
$$

Shuffle controls:

```text
No shuffled control may match RealFunctional macro beat pattern。
```

### 可视化

```text
p13_official_paired_replay_pareto.svg
p13_macro_beat_rate.svg
p13_signal_stratum_win_matrix.svg
p13_shuffle_control_matrix.svg
p13_system_gate_distribution.svg
```

---

## P14：short-run / full-run / continual / robustness

### 目标

只有 P13 pass 后打开。验证 local causal advantage 能否进入连续训练，并检查 sample efficiency 与 anti-forgetting。

### 设置

```text
short-run steps = 50, 240, 640
full-run seeds = 0..9
datasets = MNIST, Fashion-MNIST, KMNIST
continual sequence = MNIST -> Fashion-MNIST -> KMNIST and reverse diagnostic
controls =
  AdamWOnly
  AdamWParallel
  bestLR
  StrongLRGrid
  QuadraticFeatureMLP
  NoOp
  Random
  shuffled action/controller/runtime controls
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
ValLossAUC_step
ValLossAUC_time
steps_to_target
time_to_target
functional_event_count
candidate_rate
coverage
bad_event_rate
null_rate
step_ratio_q90
memory_ratio
continual_retained_accuracy
forgetting
backward_transfer
forward_transfer
old_task_CEp99_drift
old_task_margin_drift
strong_baseline_beaten
robustness_pass
base_checkpoint_hash
```

### 判断标准

Full functional pass:

$$
Acc_{functional}\ge Acc_{AdamW}-0.005
$$

and at least one:

$$
Acc_{functional}>Acc_{AdamWParallel},
$$

$$
ECE_{functional}<ECE_{AdamW},
$$

$$
NLL_{functional}<NLL_{AdamW},
$$

$$
Curvature_{functional}\le0.90Curvature_{AdamW},
$$

$$
ValLossAUC_{functional}<ValLossAUC_{AdamW}.
$$

Continual pass:

```text
retained_accuracy >= AdamW retained_accuracy - 0.005
forgetting <= AdamW forgetting
backward_transfer >= AdamW backward_transfer
old_task_CEp99_drift <= AdamW old_task_CEp99_drift
```

### 可视化

```text
p14_short_full_learning_curves.svg
p14_val_loss_auc_step_time.svg
p14_time_to_target.svg
p14_continual_retention_matrix.svg
p14_forgetting_bar.svg
p14_strong_baseline_comparison.svg
p14_robustness_frontier.svg
```

---

# 8. 并行执行计划

v9.3.1 必须并行，避免一轮只跑一个大 runner。

## Batch A：contract + materializer + runtime smoke

可以立即并行：

```text
A1: P0 v9.3.0 boundary reanalysis
A2: P1 action lifecycle apply-error closure
A3: P2 secondary/control outcome materializer smoke
A4: P9 measured zero-candidate skip runtime smoke
```

产物：

```text
p0_v9300_boundary_reanalysis.csv
p1_action_lifecycle_apply_error.csv
p2_secondary_control_outcome_smoke.csv
p9_runtime_empty_step_skip_smoke.csv
```

## Batch B：observability + useful oracle + measured runtime

P1/P2 smoke ready 后并行：

```text
B1: P3 useful/control oracle revalidation
B2: P4 oracle-to-legal gap analysis
B3: P5 action-conditioned observable factory
B4: P9 measured active-step fused runtime
```

## Batch C：controller + action primitive branch

P5 has survivor or fail evidence 后并行：

```text
C1: P6 cross-fitted minimal controller
C2: P7 decision failure autopsy v4
C3: P8 action primitive redesign branch if triggered
C4: P11 diagnostic paired replay isolated
```

## Batch D：official system

Only if P6/P8 decision survivor and P9 runtime pass:

```text
D1: P10 system integration
D2: P12 LDO/LSO official
D3: P13 official paired replay
```

## Batch E：longer runs

Only if P13 pass:

```text
E1: P14 short-run
E2: P14 full-run
E3: P14 continual / robustness / strong baseline
```

---

# 9. Required artifacts

```text
run_manifest.json
contract_audit_v9310.csv

p0_v9300_boundary_reanalysis.csv
p1_action_lifecycle_apply_error_closure.csv
action_apply_error_trace_v9310.csv
no_event_semantics_trace_v9310.csv

p2_secondary_control_outcome_materializer_v2.csv
secondary_control_outcome_trace_v9310.csv
matched_control_trace_v9310.csv
outcome_materializer_cost_trace_v9310.csv

p3_oracle_revalidation_secondary_control.csv
oracle_value_control_trace_v9310.csv

p4_oracle_legal_observability_gap.csv
observability_gap_trace_v9310.csv

p5_action_conditioned_observable_factory.csv
observable_feature_trace_v9310.csv
observable_cost_trace_v9310.csv
observable_ablation_trace_v9310.csv

p6_crossfitted_action_value_controller.csv
controller_frontier_trace_v9310.csv

p7_decision_failure_autopsy_v4.csv
decision_failure_trace_v9310.csv

p8_action_primitive_redesign_branch.csv
action_primitive_candidate_trace_v9310.csv

p9_measured_event_driven_runtime.csv
runtime_event_scheduler_trace_v9310.csv
runtime_empty_event_semantics_trace_v9310.csv
runtime_component_trace_v9310.csv

p10_system_integration_v9310.csv
system_controller_trace_v9310.csv

p11_diagnostic_causal_scout.csv
diagnostic_isolation_audit_v9310.csv

p12_leave_dataset_stratum_out.csv
leaveout_trace_v9310.csv

p13_official_paired_replay.csv
official_paired_replay_trace_v9310.csv

p14_short_full_continual_robustness.csv
short_full_continual_trace_v9310.csv

route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

---

# 10. Failure taxonomy

```text
F1_contract_violation
F2_dataset_tuning_detected
F3_teacher_or_loss_modification_detected
F4_candidate_identity_regression
F5_action_lifecycle_apply_error_fail
F6_no_event_semantics_fail
F7_secondary_outcome_missing
F8_matched_control_missing
F9_outcome_materializer_cost_fail
F10_primary_oracle_revalidation_fail
F11_control_oracle_absent
F12_legal_observability_gap
F13_action_conditioned_feature_unpredictive
F14_bad_tail_observable_fail
F15_feature_cost_infeasible
F16_controller_precision_fail
F17_controller_coverage_fail
F18_controller_bad_event_fail
F19_controller_null_rate_fail
F20_controller_lcb_ucb_fail
F21_support_balance_fail
F22_heldout_coverage_zero
F23_action_primitive_oracle_absent
F24_action_primitive_unobservable
F25_empty_step_runtime_fail
F26_empty_event_semantics_fail
F27_active_step_launch_count_fail
F28_step_ratio_fail
F29_memory_ratio_fail
F30_runtime_measured_path_missing
F31_diagnostic_promoted_to_official
F32_source_gap_or_formula_proxy_used
F33_projection_used_for_official
F34_leave_dataset_out_fail
F35_leave_stratum_out_fail
F36_paired_replay_control_equivalent
F37_shuffle_control_pass
F38_functional_lr_equivalent
F39_short_run_task_drop
F40_full_run_no_macro_or_hard_stratum_gain
F41_strong_baseline_explains_gain
F42_continual_forgetting_fail
F43_robustness_fail
F44_external_not_ready
F45_fake_or_proxy_violation
F46_artifact_missing
```

---

# 11. Route decision

```text
R1-BoundaryReanalyzed:
  v9.3.0 metrics reproduced and decision/observability/runtime gaps quantified.

R2-ActionLifecyclePass:
  action apply error and no-event semantics pass.

R3-SecondaryControlOutcomeReady:
  secondary deltas and matched controls materialized for official sample.

R4-UsefulOraclePass:
  oracle frontier remains useful under secondary/control outcomes.

R5-UsefulOracleFail:
  primary safe oracle is not useful/control-robust.

R6-ObservabilityGapQuantified:
  oracle-to-legal gap attribution pass.

R7-LegalActionObservablePass:
  action-conditioned legal observables pass bad-tail/value/cost gates.

R8-LegalActionObservableFail:
  legal observables fail despite oracle frontier.

R9-ActionValueControllerPass:
  cross-fitted minimal controller passes decision gates.

R10-ActionValueControllerFail:
  controller fails despite observable signal.

R11-ActionPrimitiveResetTriggered:
  current action primitive is judged legally unobservable.

R12-ActionPrimitiveSurvivorFound:
  redesigned action primitive improves oracle + legal observability.

R13-ActionPrimitiveOracleFail:
  redesigned action primitives fail oracle frontier.

R14-EventDrivenRuntimeMeasuredPass:
  measured online event-driven runtime reaches system envelope.

R15-EventDrivenRuntimeMeasuredFail:
  measured event-driven runtime remains outside envelope.

R16-SystemLegalControllerPass:
  decision + action lifecycle + secondary outcomes + runtime + contracts pass.

R17-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R18-LeaveStratumOutPass:
  controller generalizes across held-out strata.

R19-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R20-PairedReplayFail:
  system is legal but functional update is control-equivalent.

R21-ShortRunFunctionalPass:
  short-run task-safe mechanism gain.

R22-FullFunctionalPass:
  full run task / geometry / system / control / robustness gates pass.

R23-ContinualRobustnessPass:
  sample efficiency / anti-forgetting / robustness pass.

R24-ExternalReady:
  strict PureKAN functional route passes final external-ready gates.
```

`route_decision.json` must record:

```text
route
base_candidate
candidate_count
action_count
event_count

candidate_lifecycle_pass
action_lifecycle_pass
action_apply_error_linf
action_apply_error_relative
no_event_preservation_pass

secondary_outcome_ready
missing_secondary_delta_count
official_sample_coverage
matched_control_count_per_event

primary_oracle_pass
useful_oracle_pass
control_oracle_pass
oracle_precision
oracle_coverage
oracle_bad_event
oracle_null_rate
oracle_value_lcb
oracle_beats_adamwparallel_rate
oracle_beats_bestlr_rate

observability_gap_pass
best_observable_id
best_observable_auc_bad
best_observable_auc_safe
best_observable_ece_bad
best_observable_BadCountAt273
best_observable_SafeGoodCountAt273
best_observable_NullCountAt273
feature_cost_pass

controller_id
action_primitive_id
ogp_decision_pass
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
value_mean_heldout
precision_lcb
bad_event_ucb
support_balance_pass
stableaccept_patch_exhausted

runtime_candidate_id
runtime_mode
empty_step_controller_kernel_count
empty_step_controller_sync_count
controller_launches_per_active_step_q90
controller_syncs_per_active_step_q90
step_ratio_q90
memory_ratio
event_driven_runtime_pass

system_legal_controller_pass
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
continual_pass
robustness_pass
strong_baseline_pass
external_ready

primary_blocker
next_required_implementation
success_v9310_strict_purekan_functional
success_v9310_full_functional
success_v9310_external_ready

fake_data_used
proxy_row_used
cpu_offload_used
uses_dataset_name_for_controller
uses_loss_backward
uses_teacher
uses_loss_modification
diagnostic_downstream_used_for_controller
```

---

# 12. 停止条件

## Minimum diagnostic success

```text
P0 boundary reanalysis pass；
P1 action lifecycle apply-error measured；
P2 secondary/control materializer measured；
P3 useful/control oracle measured；
P4 observability gap measured；
P5 action-conditioned observables measured；
P6 controller measured；
P9 measured event-driven runtime measured；
no fake/proxy/offload/teacher/loss violation。
```

## Pivot to outcome/materializer fix

If:

```text
P1 action lifecycle fail
or P2 secondary/control outcome fail
```

then stop controller work. Fix action apply or outcome materializer first.

## Pivot to action primitive reset

If:

```text
P3 useful/control oracle pass
and P5 legal observable fail
and P7 identifies DF13-action_primitive_unobservable
```

then trigger P8 action primitive redesign.

## Pivot to candidate/action generation reset

If:

```text
P3 useful/control oracle fail
or P8 all action primitives oracle fail
```

then current functional event source is insufficient. Stop controller tuning.

## Pivot to controller redesign

If:

```text
P5 observable pass
but P6 controller fail
```

then fix support/cross-fitting/monotone controller form, not action primitive.

## Pivot to runtime redesign

If:

```text
P6 decision pass
but P9/P10 runtime fail
```

then stop controller tuning and focus on measured runtime path.

## Open official causal validation

Only if:

```text
P10 system pass
and P12 LDO/LSO pass
```

open P13 official paired replay.

## Full functional success

Only if:

```text
P13 paired replay pass
and P14 short/full/continual/robustness pass
```

declare full functional success.

---

# 13. 最终建议

v9.3.1 的一句话策略是：

$$
\boxed{
\text{把“有 oracle 好 action”推进为“commit 前能识别并低成本执行的 legal action-value controller”。}
}
$$

更具体地说：

```text
1. 先补 action apply error；否则 action lifecycle 不 official。
2. 先补 secondary/control outcomes；否则 value 不是完整定义。
3. 不再围绕 StableAccept 小修小补。
4. 不把 OGP feature factory 变成新补丁堆。
5. 用 oracle-to-legal gap 量化当前可观测性缺口。
6. 重点构造 action-conditioned observables，而不是 state-only candidate scores。
7. 用 BadCount@Accept273 / SafeGoodCount@Accept273 / NullCount@Accept273 替代单纯 AUC 崇拜。
8. observable 必须记录 compute cost，否则不能 official。
9. 如果 action-conditioned observables 仍失败，就进入 action primitive redesign。
10. runtime 必须 measured empty-step-free，不能再写 diagnostic pass。
11. dataset 只用于 diagnostics / leave-out，不能用于 controller branch。
12. system pass 后再打开 official paired replay。
```

v9.3.1 成功不一定等于项目终极成功，但它应该能决定下一条真正路线：

```text
A. legal observable 成立 -> controller/runtime/causal validation；
B. legal observable 不成立但 oracle 成立 -> action primitive redesign；
C. useful oracle 不成立 -> candidate/action generation reset；
D. decision 成立但 runtime 失败 -> measured event-driven runtime engineering。
```

这比继续调阈值更接近问题本质。
