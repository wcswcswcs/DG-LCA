# DG-KAN v9.3.2 Action Apply / Control Outcome Materialization 与 Event Runtime 实验计划

> 本计划基于 v9.3.1 `Legal Action-Value Observability 与 Measured Event Runtime Closure` 的真实执行结果制定。  
> v9.3.1 的核心价值不是系统成功，而是把 v9.3.0 之后最危险的幻觉暴露出来：当前 artifact 仍然不能支撑 action-value controller、control oracle、paired replay 或 measured event runtime。  
> v9.3.2 不再做“再审计一遍缺字段”的 audit-only 轮次，而是必须真正 materialize：action apply error、secondary/control outcomes、action-conditioned observables、以及 measured event-driven runtime。

---

# 0. 一句话结论

v9.3.1 的真实状态是：

```text
route = R1-BoundaryReanalyzed
base_candidate = LQ-t2-h256
success_v9310_strict_purekan_functional = False
success_v9310_full_functional = False
success_v9310_external_ready = False
```

这轮没有证明 controller 失败于新 OGP 设计，也没有证明 action primitive 没有价值；它证明的是：

$$
\boxed{
\text{当前落盘 artifact 还没有足够的信息来定义、验证或部署 action-value controller。}
}
$$

更具体地说，v9.3.1 复现了 v9.3.0 的 primary oracle frontier：

```text
candidate_count = 2876
action_count = 2876
event_count = 24192
primary_oracle_precision = 1.0
primary_oracle_coverage = 0.054453
primary_oracle_bad_event = 0.0
primary_oracle_null_rate = 0.0
```

但 action-value 所需的关键 materialization 仍然没有发生：

```text
action_apply_error_measured = 0
action_apply_error_missing_count = 2876
secondary_outcome_ready = 0
missing_secondary_delta_count = 28760
matched_control_count_per_event = 0
event_driven_runtime_pass = 0
```

因此 v9.3.2 的目标不是继续找一个新 threshold，也不是继续写 StableAccept / OGP repair，而是实现一个真正的 action-value 数据闭环：

$$
\boxed{
\text{Action Apply Materializer}
+
\text{Matched Control Outcome Materializer}
+
\text{Action-Conditioned Observable Controller}
+
\text{Measured Event Runtime}
}
$$

---

# 1. 对 v9.3.1 的独立判断

## 1.1 有进展，但不是性能进展

v9.3.1 有进展，进展类型是 **边界清理 / scientific hygiene**，不是 controller 或 runtime 的实质闭合。

它做对了三件事：

```text
1. 没有把 primary oracle frontier 写成 useful oracle frontier；
2. 没有把缺失的 action apply error 写成 action lifecycle pass；
3. 没有把 reference runtime 或 diagnostic runtime 写成 measured event-driven runtime pass。
```

这很重要，因为如果跳过这些审计，后面很容易出现虚假的 paired replay success 或 system pass。

但它没有完成 v9.3.1 名义上最关键的事情：

```text
1. 没有 materialize action apply error；
2. 没有 materialize secondary/control outcomes；
3. 没有 materialize event-driven runtime；
4. 没有形成 legal action-value observable；
5. 没有形成 deployable controller。
```

所以我对 v9.3.1 的定位是：

$$
\boxed{
\text{它把“不能推进的原因”定位清楚了，但没有向目标本身推进太多。}
}
$$

## 1.2 v9.3.1 最有价值的数据点

### 1.2.1 primary oracle frontier 仍然存在

v9.3.0 / v9.3.1 共同说明：在当前 2876 个 candidate/action population 中，按 primary safe-good label 看，确实有一组 oracle-safe rows：

```text
OR1-SafeGoodOracle accepted = 494
precision = 1.0
coverage = 0.054453
bad_event = 0.0
null_rate = 0.0
```

heldout denominator 为：

$$
N_{heldout}=9072.
$$

official coverage 下限为：

$$
N_{accept,min}=\lceil 0.03 \times 9072 \rceil=273.
$$

因此 oracle 的 494 accepted rows 明显高于 coverage 下限。这个事实很关键：

```text
当前 candidate/action population 不是 primary-safe frontier absent。
```

但它仍然只是 primary oracle，不是 useful/control oracle。它只说明这些 rows 在 safe/bad/null primary labels 上看起来好，不说明 RealFunctional 真正打过 AdamWParallel、bestLR、NoOp 或 shuffled controls。

### 1.2.2 best observable 已经接近 precision gate，但 bad-tail 失败

v9.3.1 记录了：

```text
best_observable_id = OGP-C1-CandidateScore
best_observable_SafeGoodCountAt273 = 210
best_observable_BadCountAt273 = 49
best_observable_NullCountAt273 = 16
best_observable_auc_safe = 0.756979
best_observable_auc_bad = 0.617012
```

在 accepted count = 273 时，official gate 需要：

```text
safe_good >= 205
bad_event <= 13
null <= 40
```

而 OGP-C1-CandidateScore 的 top-273 结果是：

```text
safe_good = 210  -> 过 precision 所需安全数量
null = 16       -> 过 null gate
bad = 49        -> 远超 bad gate
```

换成比例：

$$
Precision_{top273}=\frac{210}{273}=0.769.
$$

$$
BadRate_{top273}=\frac{49}{273}=0.179.
$$

$$
NullRate_{top273}=\frac{16}{273}=0.059.
$$

这说明当前 observable 的问题不是“完全找不到 safe rows”，而是 **bad-tail separation 太弱**：

$$
\boxed{
\text{它能找到足够多的 safe-good rows，但无法把 bad tail 压到 } 13 \text{ 个以内。}
}
$$

这个判断比报告里的 `observability_gap_pass = 1` 更具体。真正要攻克的是 bad-tail risk，不是再泛泛地说 feature 不够。

### 1.2.3 action-value 尚未定义

v9.3.1 的 route 里：

```text
useful_oracle_pass = 0
control_oracle_pass = 0
oracle_beats_adamwparallel_rate = ""
oracle_beats_bestlr_rate = ""
oracle_value_lcb = ""
```

这不是因为 oracle 一定没用，而是因为 matched controls 和 secondary outcomes 没有 materialize。

因此当前不能回答：

```text
1. safe-good rows 是否真的 useful？
2. RealFunctional 是否优于 AdamWParallel？
3. RealFunctional 是否优于 bestLR？
4. safe-good rows 是否只是 no-op / low-risk null rows？
5. functional payload 是否比 shuffled payload 有真实机制优势？
```

这就是 v9.3.1 的根本卡点。

### 1.2.4 runtime 仍不是新 runtime

v9.3.1 仍然是：

```text
runtime_candidate_id = RT0-v9300-reference-fixed-per-step
runtime_mode = online_sequential_official_reference
empty_step_controller_kernel_count = 59868
controller_launches_per_active_step_q90 = 9.0
step_ratio_q90 = 2.213009
```

这说明 v9.3.1 没有实现 measured event-driven runtime。它只确认了旧 runtime reference 仍然失败。

---

# 2. 当前真正卡在哪里

## 2.1 主卡点一：action apply lifecycle 没有 measured

当前记录是：

```text
action_apply_error_measured = 0
action_apply_error_missing_count = 2876
action_lifecycle_pass = 0
```

这里的 `action_apply_error_measured = 0` 不是说 action apply error 为 0，而是说 **没有测量**。

这会导致一个根本问题：我们甚至不能确认：

$$
\Delta\theta_{functional}^{logged}(e)
\equiv
\Delta\theta_{functional}^{applied}(e).
$$

如果这个等价不成立，后面所有 outcome labels、control comparison、risk/value feature 都会混乱。因为 controller 选的是 logged action，但训练中实际应用的可能不是同一个 action。

因此 v9.3.2 的第一个硬门槛必须是：

```text
每个 action row 都有可重放 payload tensor artifact；
每个 action row 都能 replay apply；
logged action 与 applied action 的误差可测；
误差必须低于 tolerance。
```

## 2.2 主卡点二：secondary/control outcomes 没有 materialize

当前记录是：

```text
secondary_outcome_ready = 0
missing_secondary_delta_count = 28760
matched_control_count_per_event = 0
```

这意味着目前只有 primary labels 能评估 decision gate，但没有足够信息定义：

$$
V(e), B(e), N(e), S(e), C(e).
$$

尤其缺少 control branches：

```text
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledFunctionalPayload
ShuffledBranchDelta
ShuffledOGPScore
```

没有这些 controls，primary oracle frontier 只能说明“这些 rows 不坏且非空”，不能说明 functional update 是真实增益。

## 2.3 主卡点三：bad-tail risk 不是均值风险问题

当前最好的 observable 在 top-273 上 safe 和 null 已过，但 bad 太多：

```text
bad = 49
allowed_bad <= 13
excess_bad = 36
```

所以 v9.3.2 不应该把主目标写成“提高 safe-good AUC 一点点”。更正确的目标是：

$$
\boxed{
\text{在保住至少 273 accepted rows 的同时，把 bad-tail 从 49 降到 } \le 13.
}
$$

这要求直接建模 bad-tail，而不是继续优化一个综合 score。

## 2.4 主卡点四：runtime 还在旧 reference path

v9.2.81 到 v9.3.1 都反复说明：

```text
step_count = 8064
active_step_count = 1412
zero_candidate_step_count = 6652
```

绝大多数 step 没有 candidate，却仍然被 fixed controller launch/sync 拖累。

这不是 kernel arithmetic 问题，而是 event scheduling 问题：

$$
\frac{zero\_candidate\_steps}{step\_count}=\frac{6652}{8064}=0.8249.
$$

v9.3.2 必须实现 measured event runtime；继续写 diagnostic estimate 没有意义。

## 2.5 流程问题：正在接近 audit treadmill

v9.3.0 已经发现：

```text
primary oracle frontier 存在；
legal observable 不足；
secondary/control missing；
runtime not materialized。
```

v9.3.1 又严格确认了：

```text
action apply missing；
secondary/control missing；
runtime missing。
```

如果 v9.3.2 仍然只是读取旧 artifacts 并再次确认 missing，那么项目会进入 audit treadmill：

$$
\boxed{
\text{每轮都更严格地证明自己缺东西，但不生成缺失的东西。}
}
$$

v9.3.2 必须改成 implementation-first：先生成 action/control/runtime artifacts，再做 controller。

---

# 3. 是否还在正确道路上

## 3.1 高层方向仍然正确

高层方向仍然正确，因为项目一直遵守了几个关键原则：

```text
1. 不用 teacher / distillation / loss modification；
2. 不按 dataset 调参；
3. 不把 diagnostic 写成 official；
4. 不在 system pass 前打开 official downstream；
5. 不把 primary oracle 当 functional success。
```

这保证了项目不是小数据集打榜，而是在验证一个更难的问题：

$$
\boxed{
\text{Clean PureKAN functional update 是否能以 MLP 级成本产生可控、可泛化、可因果验证的优势？}
}
$$

## 3.2 但具体执行路线需要从“计划审计”转为“artifact 生产”

v9.3.1 的路线如果继续重复，会偏离目标。正确的下一步不是：

```text
再做一个 observability audit；
再做一个 risk/support feature table；
再做一个 DR threshold search；
再做一个 runtime diagnostic estimate。
```

正确的下一步是：

```text
真实 replay/apply every action；
真实 materialize secondary/control outcomes；
真实测 event-driven runtime；
在这些真实数据上重做 action-conditioned controller。
```

换句话说，v9.3.2 的中心不是“找 controller”，而是 **让 action-value 这个对象第一次完整存在**。

---

# 4. 离终极目标还差多远

当前离终极目标还比较远。可以按 gate 分解：

## 4.1 离 system-legal local controller

还差至少四个 gate：

```text
1. action apply lifecycle pass；
2. secondary/control outcome ready；
3. bad-tail observability / decision pass；
4. measured event runtime pass。
```

这四个 gate 任何一个不过，都不能称为 system-legal controller。

## 4.2 离 strict PureKAN functional causal evidence

在 system controller 之后，还要过：

```text
1. leave-dataset-out；
2. leave-stratum-out；
3. official paired replay；
4. shuffle controls；
5. AdamWParallel / bestLR comparison。
```

目前还没到这个阶段，因为 matched controls 都还没有 materialize。

## 4.3 离 external-ready full functional success

还要进一步过：

```text
1. short-run continuous training；
2. full-run multi-seed robustness；
3. strong LR grid；
4. QuadraticFeatureMLP；
5. sample efficiency；
6. continual / anti-forgetting；
7. geometry / calibration / robustness advantages。
```

所以当前状态不能说“快成功了”。更准确是：

$$
\boxed{
\text{项目已经走出 StableAccept patching，但刚进入 action-value materialization 的门口。}
}
$$

---

# 5. v9.3.2 总体目标

v9.3.2 的总体目标是：

$$
\boxed{
\text{把 action-value 从报告概念变成可重放、可比较、可控制、可部署的真实数据对象。}
}
$$

具体目标分四条线并行推进：

```text
Line A: Action Apply Materialization
  确认每个 logged action 能被 replay/apply，且 applied tensor 与 logged payload 一致。

Line B: Secondary / Control Outcome Materialization
  为 official sample materialize RealFunctional 与 matched controls 的 multi-horizon outcomes。

Line C: Action-Conditioned Observability / Controller
  用 action geometry + pre-commit state + control-free legal features 建模 value / bad / null / support / cost。

Line D: Measured Event Runtime
  实现 zero-candidate skip、active-step fused controller、payload apply measured runtime。
```

v9.3.2 的强目标：

$$
ActionLifecyclePass=1,
$$

$$
SecondaryControlOutcomeReady=1,
$$

$$
ControlOraclePass=1,
$$

$$
OGPDecisionPass=1,
$$

$$
EventRuntimePass=1,
$$

$$
SystemLegalControllerPass=1.
$$

v9.3.2 的最低有效推进目标：

```text
1. action apply error 全量 materialize；
2. official sample 的 secondary/control outcomes materialize；
3. useful/control oracle frontier 被真实测量；
4. bad-tail observability gap 被定量到 top-k bad count；
5. event-driven runtime 被真实实现或明确失败；
6. 如果 controller 不过，能明确判断是 action primitive 不够、observable 不够、control value 不够，还是 runtime 不够。
```

---

# 6. v9.3.2 明确不做什么

本轮不做：

```text
1. 不继续抢救 StableAccept final controller；
2. 不继续只做 DR threshold repair；
3. 不按 MNIST / Fashion / KMNIST 单独调 threshold；
4. 不把 primary safe-good oracle 当 useful/control oracle；
5. 不在 action apply missing 时做 official controller；
6. 不在 matched controls missing 时做 paired replay claim；
7. 不把 diagnostic event-runtime estimate 写成 measured pass；
8. 不把 feature AUC pass 写成 decision pass；
9. 不把 heldout accepted count = 0 的高 precision 写成成功；
10. 不为了打榜改变 CE loss、sampler、teacher、distillation 或 class weight。
```

允许做：

```text
1. dataset-level diagnostic；
2. leave-dataset-out diagnostic；
3. per-dataset failure attribution；
4. parallel branch replay；
5. stratified official sample；
6. candidate/action primitive ablation；
7. diagnostic paired replay scout；
8. runtime microbenchmark with dummy/frozen accept masks；
9. offline materializer batching，只要不写成 online official runtime。
```

---

# 7. 核心假设

## H1：当前主要 blocker 是 action-value materialization missing，而不是 action population primary frontier absent

v9.3.0 / v9.3.1 已经显示 primary oracle frontier 存在。因此 H1 认为：当前不能推进 controller 的主因，是 action apply 与 secondary/control outcomes 没有 materialize。

H1 成立标准：

```text
primary_oracle_pass = 1
action_apply_error_missing_count > 0
secondary_outcome_ready = 0
matched_control_count_per_event = 0
```

v9.3.2 通过 H1 的标准：

```text
action_apply_error_missing_count = 0
secondary_outcome_ready = 1
matched_control_count_per_event >= 4
```

## H2：primary safe-good oracle 不等于 useful/control oracle

H2 认为：即使 OR1-SafeGoodOracle 通过，也可能只是安全、不坏、非空；它未必比 AdamWParallel 或 bestLR 更好。

定义 useful/control value：

$$
V_{ctrl}(e)=
\Delta M_{RealFunctional}(e)
-
\max_{c\in Controls}\Delta M_c(e).
$$

其中 $M$ 是综合 metric：

$$
\Delta M(e)=
-w_1 CEp99_{\Delta}(e)
+w_2 MarginP10_{\Delta}(e)
-w_3 ECE_{\Delta}(e)
-w_4 NLL_{\Delta}(e)
-w_5 Curvature_{\Delta}(e)
+w_6 Acc_{\Delta}(e).
$$

H2 成立标准：

```text
primary_oracle_pass = 1
but control_oracle_pass must be measured separately
```

H2 failure 情况：

```text
primary oracle rows all/mostly beat controls。
```

如果 H2 failure，说明当前 action population 比想象中更有价值，可以直接进入 OGP controller；否则需要 redesign action primitive。

## H3：bad-tail 是当前 observable 的主要缺口

H3 认为 OGP-C1-CandidateScore 在 top-273 已经能达到 safe/null 要求，但 bad-tail 不过。

H3 成立标准：

```text
Top273 safe_good >= 205
Top273 null <= 40
Top273 bad > 13
```

v9.3.1 已满足：

```text
safe_good = 210
null = 16
bad = 49
```

v9.3.2 的目标不是泛泛提高 AUC，而是：

```text
Top273 bad <= 13
且 safe_good >= 205
且 null <= 40
```

## H4：action-conditioned features 能比 state-only features 更好地区分 bad-tail

H4 认为当前 `CandidateScore` 主要是 state/support diagnostic，不够 action-conditioned。加入 action payload geometry、AdamW alignment、linearized descent、apply error、payload norm、control-free branch approximation 后，bad-tail 可分性会提升。

H4 成立标准：

```text
AUC_bad_event >= 0.80
Top273 bad <= 13
ECE_bad <= 0.05
leave_dataset_auc_drop <= 0.07
leave_stratum_auc_drop <= 0.10
```

H4 失败标准：

```text
action-conditioned features 仍然 bad AUC < 0.70
或 Top273 bad > 30
或 LDO/LSO collapse。
```

若 H4 失败，应优先重构 action primitive，而不是继续堆 features。

## H5：若 control oracle 不存在，当前 action primitive 不是可部署 functional update primitive

H5 认为：如果用真实 outcome/control labels 的 oracle 都无法在 coverage >= 0.03 下找到打过 controls 的 action subset，则 controller 无论多好都没有意义。

Control oracle pass：

```text
coverage >= 0.03
precision_control_useful >= 0.75
bad_event <= 0.05
null_rate <= 0.15
RealFunctional beats AdamWParallel rate >= 0.60
RealFunctional beats bestLR rate >= 0.60
support_balance_pass = 1
```

Control oracle fail：

```text
No oracle can satisfy coverage >=0.03 and useful/control gates。
```

若 H5 fail，应 route 到：

```text
R20-ActionPrimitiveInsufficient
```

而不是 controller repair。

## H6：measured event runtime 不是 optional；没有 runtime 就没有 official system

H6 认为：即使 decision 过线，runtime 不过也不能 official。

Event runtime pass：

```text
zero_candidate_controller_kernel_count = 0
zero_candidate_controller_sync_count = 0
controller_launches_per_active_step_q90 <= 2
controller_syncs_per_active_step_q90 <= 1
allocation_count_per_active_step <= 0.05
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
no_event_preservation_pass = 1
base_adamw_equivalence_on_zero_candidate_steps = 1
```

---

# 8. 数据合同

## 8.1 Action identity and apply contract

每个 action row 必须记录：

```text
action_id
candidate_id
event_id
action_schema_version
primitive_id
dataset
seed
step
batch_id
family_id
bucket_id
horizon
payload_hash
payload_tensor_hash
pre_state_hash
post_state_hash_expected
post_state_hash_applied
adamw_delta_hash
functional_delta_hash
applied_delta_hash
```

Action apply error：

$$
E_{linf}(e)=\|\Delta\theta_{functional}^{logged}(e)-\Delta\theta_{functional}^{applied}(e)\|_{\infty}.
$$

$$
E_{rel}(e)=
\frac{
\|\Delta\theta_{functional}^{logged}(e)-\Delta\theta_{functional}^{applied}(e)\|_2
}{
\|\Delta\theta_{functional}^{logged}(e)\|_2+\epsilon
}.
$$

Pass：

```text
action_apply_error_measured = 1
action_apply_error_missing_count = 0
max_action_apply_error_linf <= 1e-5
max_action_apply_error_relative <= 1e-4
payload_hash_missing_count = 0
payload_replay_success_count = action_count
```

如果 fp16/bf16 path 使用近似，需要额外记录：

```text
precision_mode
expected_tolerance
observed_tolerance
bitexact_required = 0/1
```

## 8.2 Secondary outcome contract

每个 measured branch row 必须记录：

```text
action_id
candidate_id
event_id
branch_id
horizon
dataset
seed
step
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
local_lipschitz_delta
acc_delta
loss_auc_delta
margin_auc_delta
time_to_target_delta
steps_to_target_delta
basis_usage_entropy_delta
functional_channel_entropy_delta
task_safe_label
useful_label
bad_event_label
null_event_label
safe_good_label
```

Branches：

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledFunctionalPayload
ShuffledBranchDelta
ShuffledOGPScore
ShuffledCandidatePayload
ShuffledActionDirection
ShuffledActionMagnitude
```

Pass：

```text
missing_secondary_delta_count_official_sample = 0
matched_control_count_per_event >= 4
branch_missing_count = 0 for official rows
horizon_missing_count = 0 for official rows
outcome_runtime_recorded = 1
```

## 8.3 Official sample contract

由于全量 2876 actions × branches × horizons 可能成本较高，v9.3.2 采用三层 sample，但必须明示 official 与 diagnostic 的边界。

### Tier 1：official-minimum sample

必须包含：

```text
1. StableAccept accepted rows；
2. v9.3.0 oracle-safe rows；
3. OGP-C1 top-k rows；
4. OGP-C1 bad-tail suspected rows；
5. matched rejected rows by family × horizon × bucket × seed；
6. 每个 dataset / seed / family / horizon 至少 n_min rows。
```

要求：

```text
official_sample_candidate_coverage >= 0.50
or all cells family × horizon × bucket have n >= 20
```

若低于该覆盖，只能 diagnostic，不能进入 P6 controller official。

### Tier 2：stratified full-risk sample

用于 bad-tail 建模：

```text
oversample bad-risk suspected bins；
oversample low-support bins；
oversample horizon-tail bins；
oversample candidate-origin drift bins。
```

### Tier 3：full candidate optional

如果成本允许，对全部 2876 actions materialize all branches/horizons。

## 8.4 OGP feature contract

所有 OGP features 必须 commit 前可得，且记录 legality：

```text
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
uses_validation_or_test = 0
uses_control_branch_at_commit = 0
source_measured_gap_used = 0
formula_proxy_used_for_official = 0
```

Feature groups：

### Group A：state / tail features

```text
batch_CE_mean
batch_CE_p90
batch_CE_p99
batch_margin_p10
batch_margin_p20
wrong_conf_p90
entropy_mean
hard_tail_fraction
```

### Group B：action geometry features

```text
payload_norm
payload_tail_norm
payload_role_entropy
functional_delta_norm
functional_delta_tail_norm
functional_norm_over_adamw_norm
action_sparsity
action_family_id
action_direction_family
```

### Group C：AdamW alignment features

```text
cos_functional_delta_adamw_delta
cos_functional_delta_negative_grad
cos_payload_adamw_update
projected_CE_descent_estimate
gradient_conflict_rate
rolewise_conflict_score
```

### Group D：linearized effect features

```text
linearized_CE_delta
linearized_margin_delta
linearized_tail_CE_delta
linearized_curvature_proxy
linearization_reliability_lcb
linearization_error_ucb
```

### Group E：support / empirical Bayes features

```text
neighbor_count
family_count
horizon_count
feature_bin_count
empirical_bayes_bad_ucb
empirical_bayes_null_ucb
empirical_bayes_value_lcb
support_effective_sample_size
shift_risk
```

### Group F：runtime cost features

```text
candidate_count_in_step
active_step_flag
kernel_shape_id
estimated_controller_launch_cost
estimated_payload_apply_cost
feature_compute_time_ms
payload_apply_time_ms
```

Minimality rule：

```text
每个 deployable controller 最多使用 5 个 feature groups；
必须报告 single-feature baseline；
必须报告 feature ablation；
必须报告 monotone sign consistency；
禁止 opaque 20-feature pile 直接转正。
```

---

# 9. 实验阶段

---

## P0：v9.3.1 boundary reproduction and no-audit-only guard

### 目标

复现 v9.3.1 boundary，并强制建立 v9.3.2 的 no-audit-only guard：如果本轮 runner 只读旧 artifact 而没有生成 action apply / control outcome / measured runtime 三类新 artifact，则 route 必须直接失败。

### 假设

H0：v9.3.1 已经足够说明缺什么；v9.3.2 不能再只是证明缺失。

### 必须记录

```text
source_route_v9310
candidate_count
action_count
event_count
primary_oracle_pass
primary_oracle_precision
primary_oracle_coverage
best_observable_id
best_observable_auc_safe
best_observable_auc_bad
best_observable_SafeGoodCountAt273
best_observable_BadCountAt273
best_observable_NullCountAt273
action_apply_error_measured_before
action_apply_error_missing_count_before
secondary_outcome_ready_before
missing_secondary_delta_count_before
matched_control_count_per_event_before
runtime_candidate_id_before
step_ratio_q90_before
empty_step_controller_kernel_count_before
```

新增 no-audit-only fields：

```text
new_action_apply_artifact_written
new_secondary_control_artifact_written
new_event_runtime_artifact_written
new_artifact_row_count
audit_only_round_detected
```

### 判断标准

P0 pass：

```text
v9.3.1 boundary reproduced；
no-audit-only guard installed；
new artifact generation planned and tracked。
```

Hard fail：

```text
audit_only_round_detected = 1
and no new materialization artifact written。
```

### Artifacts

```text
p0_v9310_boundary_reproduction.csv
p0_no_audit_only_guard.csv
```

### 可视化

```text
p0_v9310_gate_ladder.svg
p0_missing_materialization_dashboard.svg
p0_top273_safe_bad_null_bar.svg
```

---

## P1：Action Apply Materializer

### 目标

为全部 2876 actions materialize 可重放 action payload，并测量 logged action 与 applied action 的误差。P1 只回答 action lifecycle 是否真实闭合，不做 controller。

### 假设

H1：当前 action population 的 identity 可以 freeze，但 action apply 仍未被证明；必须先验证 action payload 是否真的可用。

### 实现

新增 runner：

```text
experiments/run_v9320_action_apply_materializer.py
```

执行逻辑：

```text
1. 读取 frozen_candidate_action_table_v9300/v9310；
2. 对每个 action 定位 pre-state checkpoint；
3. 读取或重建 logged functional payload；
4. 在 isolated replay context 中 apply functional payload；
5. 比较 logged delta、expected applied delta、actual applied delta；
6. 记录 action_apply_error_linf / relative / cosine；
7. 记录 AdamW base equivalence 与 no-event preservation；
8. 输出 action_apply_trace_v9320.csv。
```

### 必须记录

```text
action_id
candidate_id
event_id
primitive_id
dataset
seed
step
family_id
bucket_id
horizon
pre_state_hash
base_adamw_delta_hash
functional_payload_hash
logged_delta_hash
applied_delta_hash
post_state_hash_expected
post_state_hash_actual
action_apply_error_linf
action_apply_error_l2
action_apply_error_relative
cos_logged_applied
payload_norm
payload_l2
payload_linf
functional_norm_over_adamw_norm
no_event_preservation_error
base_adamw_equivalence_error
replay_success
```

### 判断标准

P1 pass：

```text
action_apply_error_measured = 1
action_apply_error_missing_count = 0
replay_success_count = action_count
max_action_apply_error_linf <= 1e-5
max_action_apply_error_relative <= 1e-4
min_cos_logged_applied >= 0.9999
no_event_preservation_pass = 1
base_adamw_equivalence_on_zero_candidate_steps = 1
```

P1 fail route：

```text
R2-ActionApplyMaterializationFail
```

### 可视化

```text
p1_action_apply_error_hist.svg
p1_logged_vs_applied_norm_scatter.svg
p1_cos_logged_applied_hist.svg
p1_payload_norm_by_family_horizon.svg
p1_no_event_preservation_error.svg
```

### 解释规则

如果 P1 fail，不允许进入 OGP controller。因为此时 action identity 与 effect 不可信。

---

## P2：Secondary / Matched Control Outcome Materializer

### 目标

为 official sample materialize RealFunctional 与 matched controls 的 multi-horizon outcomes。P2 是 v9.3.2 的核心数据生产阶段。

### 假设

H2：只有 matched controls materialize 后，才能区分 safe-good oracle 与 useful/control oracle。

### 实现

新增 runner：

```text
experiments/run_v9320_secondary_control_outcome_materializer.py
```

并行 shard：

```text
--shard-id k --num-shards K
```

推荐并行方式：

```text
Shard axis 1: dataset × seed
Shard axis 2: branch
Shard axis 3: horizon
Shard axis 4: candidate/action strata
```

Branches：

```text
B0-RealFunctional
B1-AdamWOnly
B2-AdamWParallel
B3-bestLR
B4-NoOp
B5-Random
B6-ShuffledFunctionalPayload
B7-ShuffledBranchDelta
B8-ShuffledCandidatePayload
B9-ShuffledActionDirection
B10-ShuffledActionMagnitude
B11-ShuffledOGPScoreDiagnostic
```

Horizons：

```text
horizon = 20, 80, 240
optional horizon = 640
```

### 必须记录

```text
action_id
candidate_id
event_id
branch_id
horizon
dataset
seed
step
family_id
bucket_id
pre_state_hash
branch_state_hash
branch_payload_hash
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
local_lipschitz_delta
acc_delta
train_loss_delta
val_loss_proxy_delta
loss_auc_delta
margin_auc_delta
basis_usage_entropy_delta
functional_channel_entropy_delta
time_to_target_delta
steps_to_target_delta
task_safe_label
useful_label
bad_event_label
null_event_label
safe_good_label
branch_runtime_ms
materializer_shard_id
```

### 判断标准

P2 official-ready pass：

```text
missing_secondary_delta_count_official_sample = 0
matched_control_count_per_event >= 4
branch_missing_count = 0
horizon_missing_count = 0
official_sample_candidate_coverage >= 0.50
or family_horizon_bucket_min_n >= 20
outcome_runtime_recorded = 1
```

P2 diagnostic-only：

```text
coverage < 0.50
or matched_control_count_per_event < 4
```

P2 fail route：

```text
R3-SecondaryControlOutcomeMaterializationFail
```

### 可视化

```text
p2_materialization_coverage_by_family_horizon.svg
p2_branch_delta_distribution.svg
p2_real_vs_adamwparallel_scatter.svg
p2_real_vs_bestlr_scatter.svg
p2_noop_random_shuffle_control_matrix.svg
p2_horizon_consistency_heatmap.svg
p2_missing_branch_horizon_dashboard.svg
```

---

## P3：Useful / Control Oracle Frontier

### 目标

在 P2 真实 outcomes 上判断：当前 action population 是否不仅 primary-safe，而且有 control-useful frontier。

### 假设

H5：如果 control oracle 都不过，当前 action primitive 没有可部署价值，应重构 action generator，而不是继续找 controller。

### Oracle definitions

Primary oracle：

$$
Oracle_{primary}(e)=1
\iff
SafeGood(e)=1
\land BadEvent(e)=0
\land NullEvent(e)=0.
$$

Control-useful oracle：

$$
Oracle_{control}(e)=1
\iff
Oracle_{primary}(e)=1
\land V_{ctrl}(e)>0
\land Beat_{AdamWParallel}(e)=1
\land Beat_{bestLR}(e)=1.
$$

Composite value：

$$
V(e)=
-w_1 CEp99_{\Delta}(e)
+w_2 MarginP10_{\Delta}(e)
-w_3 ECE_{\Delta}(e)
-w_4 NLL_{\Delta}(e)
-w_5 Curvature_{\Delta}(e)
+w_6 Acc_{\Delta}(e).
$$

Control value：

$$
V_{ctrl}(e)=V_{RealFunctional}(e)-\max(V_{AdamWParallel}(e),V_{bestLR}(e),V_{NoOp}(e)).
$$

### 必须记录

```text
oracle_id
coverage_target
accepted_count
precision_primary
precision_control_useful
coverage
bad_event_rate
null_rate
value_mean
value_lcb
control_value_mean
control_value_lcb
beats_adamwparallel_rate
beats_bestlr_rate
beats_noop_rate
beats_random_rate
shuffle_gap
accepted_family_count
accepted_signal_strata_count
accepted_action_family_count
max_family_share
max_stratum_share
support_balance_pass
```

### 判断标准

P3 primary oracle pass：

```text
coverage >= 0.03
precision_primary >= 0.90
bad_event_rate <= 0.02
null_rate <= 0.10
support_balance_pass = 1
```

P3 control oracle weak pass：

```text
coverage >= 0.03
precision_control_useful >= 0.65
bad_event_rate <= 0.05
null_rate <= 0.15
beats_adamwparallel_rate >= 0.50
beats_bestlr_rate >= 0.50
```

P3 control oracle strong pass：

```text
coverage >= 0.03
precision_control_useful >= 0.75
bad_event_rate <= 0.05
null_rate <= 0.15
beats_adamwparallel_rate >= 0.60
beats_bestlr_rate >= 0.60
value_lcb > 0
control_value_lcb > 0
support_balance_pass = 1
```

P3 fail route：

```text
R4-ControlOracleAbsent_ActionPrimitiveInsufficient
```

### 可视化

```text
p3_primary_vs_control_oracle_frontier.svg
p3_value_risk_null_pareto.svg
p3_real_vs_controls_beat_matrix.svg
p3_oracle_support_balance.svg
p3_shuffle_gap_distribution.svg
```

---

## P4：Bad-tail Observability and Action-Conditioned Feature Factory

### 目标

构造并评估 legal pre-commit action-conditioned features，重点不是泛泛 AUC，而是能否在 accepted count = 273 附近把 bad count 从 49 压到 13 以下。

### 假设

H4：action-conditioned features 能修复 CandidateScore 的 bad-tail blindness。

### Feature candidates

#### F1：Action-AdamW alignment

$$
F_{align}(e)=cos(\Delta\theta_{func}(e),\Delta\theta_{AdamW}(e)).
$$

#### F2：Projected CE descent

$$
F_{descent}(e)=\nabla_{\theta}CE(\theta_t)^T\Delta\theta_{func}(e).
$$

Good sign：

$$
F_{descent}(e)<0.
$$

#### F3：Tail conflict score

$$
F_{tail}(e)=
CEp99_t
-eta_1 MarginP10_t
+eta_2 TailDeltaConflict(e).
$$

#### F4：Payload norm risk

$$
F_{norm}(e)=
\frac{\|\Delta\theta_{func}(e)\|_2}{\|\Delta\theta_{AdamW}(e)\|_2+\epsilon}.
$$

#### F5：Empirical Bayes bad UCB

$$
BadUCB_{EB}(e)=
\hat p_{bad}(e)
+
 z_{\alpha}\sqrt{\frac{\hat p_{bad}(e)(1-\hat p_{bad}(e))}{n_{eff}(e)+\epsilon}}
+
\lambda_{shift}ShiftRisk(e).
$$

#### F6：Linearization reliability

$$
Rel(e)=1-UCB(|\Delta_{realized}^{audit}(e)-\Delta_{linearized}(e)|).
$$

#### F7：Cost-aware acceptability

$$
Cost(e)=FeatureCost(e)+ScoreCost(e)+PayloadApplyCost(e).
$$

### 必须记录

```text
feature_id
feature_group
uses_dataset_name
uses_outcome_at_commit
uses_control_branch_at_commit
uses_future_step
feature_missing_rate
feature_compute_time_ms_q50
feature_compute_time_ms_q90
feature_memory_ratio
AUC_bad_event
AUC_safe_good
AUC_null_event
PR_AUC_bad_lift
Brier_bad
ECE_bad
calibration_slope
Top273_safe_count
Top273_bad_count
Top273_null_count
Top273_precision
Top273_bad_rate
Top273_null_rate
leave_dataset_auc_drop
leave_stratum_auc_drop
monotone_sign_pass
```

### 判断标准

P4 feature pass：

```text
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_control_branch_at_commit = 0
feature_missing_rate <= 0.01
feature_compute_time_ms_q90 recorded
feature_memory_ratio recorded
```

Signal pass，至少满足一组：

```text
AUC_bad_event >= 0.80
or Top273_bad_count <= 13 with Top273_safe_count >= 205 and Top273_null_count <= 40
```

Calibration pass：

```text
ECE_bad <= 0.05
leave_dataset_auc_drop <= 0.07
leave_stratum_auc_drop <= 0.10
monotone_sign_pass = 1
```

P4 fail route：

```text
R5-ActionConditionedObservabilityFail
```

### 可视化

```text
p4_feature_auc_bar.svg
p4_top273_bad_tail_comparison.svg
p4_risk_calibration_curve.svg
p4_feature_cost_vs_signal_pareto.svg
p4_leaveout_auc_drop_heatmap.svg
p4_action_geometry_bad_tail_scatter.svg
p4_linearized_vs_realized_delta.svg
```

---

## P5：Action Primitive Insufficiency Autopsy

### 目标

如果 P3 control oracle fail 或 P4 observability fail，必须判断问题是 action primitive 没有产生可用 action，还是 observable 不够。P5 不是小修补丁，而是决定是否重构 functional update primitive。

### 触发条件

```text
P3 control oracle fail
or P4 action-conditioned observability fail
or P4 Top273_bad_count remains > 30
```

### Autopsy dimensions

记录：

```text
action_direction_family
payload_norm_bin
payload_alignment_bin
horizon
family_id
bucket_id
candidate_origin_tag
control_value_distribution
bad_event_distribution
null_event_distribution
action_apply_error_distribution
linearization_error_distribution
oracle_miss_reason
```

Failure reasons：

```text
APF1-action_too_small_null_dominant
APF2-action_too_large_bad_tail
APF3-action_adamw_conflict
APF4-action_tail_conflict
APF5-action_horizon_unstable
APF6-action_family_concentrated
APF7-action_payload_not_replayable
APF8-action_no_control_advantage
APF9-action_observable_hidden_state_missing
```

### Candidate primitive redesign scouts

这些 scout 不按 dataset 调参，只改变 action primitive 的第一性结构：

#### AP1：Gradient-aligned projection

$$
\Delta\theta_{func}'=
Proj_{\cos(\Delta\theta_{func},-g)\ge \rho}(\Delta\theta_{func}).
$$

#### AP2：Norm-controlled functional delta

$$
\Delta\theta_{func}'=
\alpha(e)\Delta\theta_{func},
$$

其中：

$$
\alpha(e)=\min\left(1,\frac{r\|\Delta\theta_{AdamW}\|_2}{\|\Delta\theta_{func}\|_2+\epsilon}\right).
$$

#### AP3：Tail-safe delta projection

Reject or project action if linearized tail CE risk is positive：

$$
\widehat{\Delta CE}_{tail}(e)>0.
$$

#### AP4：Horizon-stable action

Only keep actions whose predicted effects agree across short horizon proxies：

$$
sign(\widehat{V}_{20})=sign(\widehat{V}_{80})
$$

and tail-risk does not increase.

#### AP5：Null-resistant payload

Reject low payload/effect actions：

$$
\|\Delta\theta_{func}\|_2 < \tau_{payload}
\land
|\widehat{\Delta CE}_{lin}|<\epsilon.
$$

#### AP6：Control-aware candidate generator diagnostic

Use controls only offline to diagnose whether a new primitive population has useful oracle frontier. It cannot be used at commit time.

### 判断标准

P5 primitive sufficient：

```text
current primitive has control oracle weak/strong pass
and action-conditioned observability can reach Top273 bad <=13
```

P5 primitive insufficient：

```text
control oracle fail
or action family with useful control value coverage <0.03
or bad-tail cannot be reduced below 30 even with action-conditioned features
```

If insufficient：

```text
route = R20-ActionPrimitiveInsufficient
next_required_implementation = redesign_functional_action_generator
```

### 可视化

```text
p5_action_primitive_failure_sankey.svg
p5_action_value_by_direction_family.svg
p5_payload_norm_value_risk_phase.svg
p5_adamw_alignment_vs_control_value.svg
p5_horizon_stability_matrix.svg
p5_primitive_scout_oracle_frontier.svg
```

---

## P6：Cross-fitted Minimal Action-Value Controller

### 目标

只有 P3 control oracle pass 且 P4 feature pass 后，才训练/选择 legal controller。Controller 必须 minimal、monotone、cross-fitted、dataset-agnostic。

### Controller form

$$
Accept(e)=1
\iff
LCB(V(e))>0
\land
UCB(B(e))\le\tau_b
\land
UCB(N(e))\le\tau_n
\land
LCB(S(e))\ge\tau_s
\land
C(e)\le C_{max}.
$$

Monotone score：

$$
S_{OGP}(e)=
 a_1LCB(V(e))
-a_2UCB(B(e))
-a_3UCB(N(e))
+a_4LCB(S(e))
-a_5C(e),
$$

with：

$$
a_i\ge0.
$$

### Candidate controllers

```text
C0-StableAccept negative control
C1-CandidateScore baseline
C2-BadTailOnly gate
C3-ActionAlignmentRisk gate
C4-ValueRiskNullSupportCost gate
C5-TwoStageCheapThenExactLegalAudit
C6-MinimalMonotoneOGP
```

### Splits

```text
Seed folds:
  rotate calibration / heldout seeds if available。

Leave-dataset-out:
  train/calibrate on two datasets, evaluate third。

Leave-stratum-out:
  hold out signal stratum。

Leave-family-out diagnostic:
  hold out high-volume family。
```

### 必须记录

```text
controller_id
feature_set
feature_group_count
model_class
monotonic_constraints
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
control_value_mean_cal
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
value_mean_heldout
control_value_mean_heldout
beats_adamwparallel_rate
beats_bestlr_rate
precision_lcb
bad_event_ucb
null_event_ucb
value_lcb
control_value_lcb
accepted_family_count
accepted_signal_strata_count
max_family_share
max_stratum_share
Top273_safe_count
Top273_bad_count
Top273_null_count
feature_ablation_pass
single_feature_baseline_pass
monotone_sign_pass
```

### 判断标准

Decision pass：

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

Control value pass：

```text
beats_adamwparallel_rate >= 0.60
beats_bestlr_rate >= 0.60
control_value_lcb > 0
```

Support pass：

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

Cross-fit pass：

```text
coverage_heldout > 0 in every main fold
bad_event gate pass in at least 2/3 folds
no dataset-specific branch
macro decision score pass
```

P6 fail route：

```text
R6-ActionValueControllerFail
```

### 可视化

```text
p6_controller_precision_bad_coverage_frontier.svg
p6_control_value_frontier.svg
p6_crossfit_fold_matrix.svg
p6_calibration_to_heldout_drift.svg
p6_support_balance_sunburst.svg
p6_feature_ablation_waterfall.svg
p6_stableaccept_vs_ogp_overlap.svg
```

---

## P7：Measured Online Event Runtime

### 目标

实现并测量 online event-driven runtime。P7 不依赖 final controller；可以用 frozen accept masks / dummy masks / top-k masks 先验证 runtime semantics，但 official pass 必须在 real controller 上测。

### 假设

H6：runtime failure 主要来自 zero-candidate fixed launch/sync 与 active-step kernel fragmentation。

### Runtime candidates

```text
RT0-v9310-reference-fixed-per-step
RT1-zero-candidate-skip-measured
RT2-active-step-single-fused-controller
RT3-active-step-two-kernel-score-apply
RT4-persistent-workspace-event-scheduler
RT5-online-event-runtime-with-real-controller
RT6-offline-batch-major-materializer-runtime diagnostic only
```

### Empty-event semantics

必须证明：

```text
zero-candidate step 不运行 functional controller；
zero-candidate step 不改变 base AdamW update；
zero-candidate step 不改变 batch order；
zero-candidate step 不改变 optimizer state；
audit outside timed path；
```

### 必须记录

```text
runtime_candidate_id
runtime_mode
controller_id
accept_mask_source
step_count
active_step_count
zero_candidate_step_count
candidate_count
accepted_count
zero_candidate_controller_kernel_count
zero_candidate_controller_sync_count
controller_kernel_launch_count
controller_sync_count
controller_launches_per_active_step_mean
controller_launches_per_active_step_q90
controller_syncs_per_active_step_mean
controller_syncs_per_active_step_q90
allocation_count
allocation_count_per_active_step
candidate_pack_time_ms_q90
feature_compute_time_ms_q90
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
accept_disagreement_count
payload_apply_error_max
no_event_preservation_pass
base_adamw_equivalence_on_zero_candidate_steps
event_scheduler_does_not_reorder_batches
```

### 判断标准

P7 measured runtime pass：

```text
runtime_mode = online_sequential_official
zero_candidate_controller_kernel_count = 0
zero_candidate_controller_sync_count = 0
controller_launches_per_active_step_q90 <= 2
controller_syncs_per_active_step_q90 <= 1
allocation_count_per_active_step <= 0.05
accept_disagreement_count = 0
payload_apply_error_max <= tolerance
no_event_preservation_pass = 1
base_adamw_equivalence_on_zero_candidate_steps = 1
event_scheduler_does_not_reorder_batches = 1
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

Diagnostic runtime pass：

```text
zero_candidate_controller_kernel_count = 0
launch/sync reduction >= 80%
step_ratio_q90 <= 2.00
```

Diagnostic pass 不能写成 official。

### 可视化

```text
p7_runtime_mode_comparison.svg
p7_empty_vs_active_step_timing.svg
p7_kernel_sync_reduction.svg
p7_launches_per_active_step_hist.svg
p7_runtime_waterfall.svg
p7_step_ratio_q50_q90.svg
p7_no_event_semantics_dashboard.svg
p7_memory_ratio.svg
```

---

## P8：Controller + Runtime System Integration

### 目标

组合 P6 best controller 与 P7 measured runtime，建立 system legal controller。只有 decision、control value、runtime、contract 全部过，P8 才能 pass。

### 必须记录

```text
system_candidate_id
controller_id
runtime_candidate_id
action_primitive_id
feature_set
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
precision_lcb
bad_event_ucb
value_mean_heldout
value_lcb
control_value_mean_heldout
control_value_lcb
beats_adamwparallel_rate
beats_bestlr_rate
accepted_family_count
accepted_signal_strata_count
max_family_share
max_stratum_share
step_ratio_q90
active_step_ratio_q90
memory_ratio
controller_launches_per_active_step_q90
zero_candidate_controller_kernel_count
action_lifecycle_pass
secondary_control_outcome_ready
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

P8 pass：

```text
official_eligible = 1
system_legal_controller_pass = 1
action_lifecycle_pass = 1
secondary_control_outcome_ready = 1
materialized_system_path = 1
diagnostic_derived_from_measured_components = 0
projection_used = 0
source_measured_gap_used = 0
formula_proxy_used = 0
dataset_name_used = 0
```

Decision gates：

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
NullRate_{heldout}\le0.15.
$$

Control gates：

$$
BeatRate_{Real\ vs\ AdamWParallel}\ge0.60,
$$

$$
BeatRate_{Real\ vs\ bestLR}\ge0.60,
$$

$$
ControlValue_{LCB}>0.
$$

Runtime gates：

$$
StepRatio_{q90}\le1.50,
$$

$$
MemoryRatio\le1.05.
$$

### 可视化

```text
p8_system_quality_cost_frontier.svg
p8_official_gate_dashboard.svg
p8_controller_runtime_pareto.svg
p8_failure_reason_matrix.svg
p8_value_risk_cost_phase.svg
```

---

## P9：Leave-dataset-out / Leave-stratum-out

### 目标

只有 P8 pass 后 official 打开。证明 controller 不是 pooled calibration artifact，也不是 dataset-specific tuning。

### 设置

Leave-dataset-out：

```text
calibrate MNIST + Fashion, evaluate KMNIST
calibrate MNIST + KMNIST, evaluate Fashion
calibrate Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
calibrate all but one signal stratum
evaluate held-out stratum
```

Leave-family-out diagnostic：

```text
calibrate all but one high-volume family
evaluate held-out family
```

### 必须记录

```text
split_type
heldout_entity
controller_id
runtime_candidate_id
candidate_rate
precision
coverage
bad_event_rate
null_rate
precision_lcb
bad_event_ucb
value_mean
value_lcb
control_value_mean
control_value_lcb
beats_adamwparallel_rate
beats_bestlr_rate
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
step_ratio_q90
memory_ratio
dataset_name_used
support_balance_pass
```

### 判断标准

LDO decision pass：

```text
at least 2/3 held-out datasets pass decision gates
no held-out dataset has bad_event_rate > 0.10
coverage > 0 for all held-out datasets
dataset_name_used = 0
```

LDO control pass：

```text
at least 2/3 held-out datasets have beats_adamwparallel_rate >= 0.50
at least 2/3 held-out datasets have beats_bestlr_rate >= 0.50
```

LSO pass：

```text
>=70% held-out strata task-safe
macro bad_event_rate <= 0.05
macro precision >= 0.75
macro control_value_lcb > 0
```

### 可视化

```text
p9_leave_dataset_out_matrix.svg
p9_leave_stratum_out_matrix.svg
p9_leave_family_out_diagnostic.svg
p9_dataset_diagnostic_no_tuning_audit.svg
p9_control_value_leaveout_heatmap.svg
```

---

## P10：Diagnostic paired replay scout

### 目标

加快实验，但保持隔离。P10 可以在 P6/P7 之后并行跑，不影响 controller threshold / feature selection。

### 隔离规则

```text
diagnostic_paired_replay_used_for_controller = 0
diagnostic_paired_replay_used_for_threshold = 0
diagnostic_paired_replay_used_for_feature_selection = 0
```

### 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
horizons = 20,80,240
controllers = top2 OGP + StableAccept negative control
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, shuffled controls
```

### 必须记录

```text
controller_id
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
beats_adamwparallel
beats_bestlr
task_safe
step_ratio_q90
memory_ratio
status = diagnostic_not_official
```

### 判断标准

P10 scout promising：

```text
RealFunctional beats AdamWParallel in >= 50% macro slices
RealFunctional beats bestLR in >= 50% macro slices
task_safe holds: Acc_real >= Acc_adamw - 0.005
shuffled controls do not match RealFunctional
```

### 可视化

```text
p10_diagnostic_paired_replay_macro_beat.svg
p10_branch_delta_pareto.svg
p10_shuffle_control_matrix.svg
p10_task_safety_by_slice.svg
```

---

## P11：Official paired replay

### 前置条件

```text
P8 system pass = 1
P9 LDO/LSO pass = 1
secondary_control_outcome_ready = 1
runtime official pass = 1
```

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
  ShuffledCandidatePayload
  ShuffledActionDirection
  ShuffledActionMagnitude
  ShuffledOGPValueScore
  ShuffledOGPRiskScore
  ShuffledOGPSupportScore
  FunctionalChannelShuffled
  TailMaskShuffled
  RoleScoreShuffled
  DatasetRouteShuffled
  EventRouteShuffled
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
BeatRate_{macro,Real\ vs\ AdamWParallel}\ge0.60.
$$

$$
BeatRate_{macro,Real\ vs\ bestLR}\ge0.60.
$$

Task safety：

$$
Acc_{each\ slice,Real}\ge Acc_{AdamW}-0.005.
$$

Shuffle controls：

```text
所有 shuffled controls 不得达到 RealFunctional 的 macro beat pattern。
```

### 可视化

```text
p11_official_paired_replay_pareto.svg
p11_macro_beat_rate.svg
p11_signal_stratum_win_matrix.svg
p11_shuffle_control_matrix.svg
p11_system_gate_distribution.svg
```

---

## P12：Short-run / full-run / robustness / sample efficiency / continual

### 目标

只有 P11 pass 后打开。验证 local causal advantage 是否能进入连续训练，并检查 sample efficiency 与 anti-forgetting。

### 设置

```text
short-run steps = 50, 240, 640
full-run seeds = 0..9
datasets = MNIST, Fashion-MNIST, KMNIST
controls =
  AdamWOnly
  AdamWParallel
  bestLR
  StrongLRGrid
  QuadraticFeatureMLP
  NoOp
  Random
  ShuffledFunctionalPayload
  ShuffledOGPScore
  ShuffledRuntimePath
```

Continual setting：

```text
Task order examples:
  MNIST -> Fashion -> KMNIST
  Fashion -> KMNIST -> MNIST
  KMNIST -> MNIST -> Fashion
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
ValLossAUC_step
ValLossAUC_time
time_to_target
steps_to_target
sample_efficiency_gain
continual_task_order
retained_accuracy
forgetting
backward_transfer
forward_transfer
old_task_CEp99_drift
old_task_margin_drift
base_checkpoint_hash
```

### 判断标准

Full functional pass：

$$
Acc_{functional}\ge Acc_{AdamW}-0.005
$$

and at least one：

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
Curvature_{functional}\le0.90Curvature_{AdamW}.
$$

Sample efficiency pass：

```text
ValLossAUC_step improves over AdamWParallel
or time_to_target improves while step_ratio_q90 <= 1.50
```

Continual pass：

```text
retained_accuracy >= AdamW retained_accuracy - 0.005
forgetting <= AdamW forgetting
old_task_CEp99_drift <= AdamW old_task_CEp99_drift
```

### 可视化

```text
p12_short_full_learning_curves.svg
p12_val_loss_auc_step_time.svg
p12_time_to_target.svg
p12_robustness_strong_baseline_matrix.svg
p12_continual_forgetting_matrix.svg
p12_sample_efficiency_pareto.svg
```

---

# 10. 并行执行计划

v9.3.2 必须并行推进，避免又变成一条串行 audit。

## Batch A：必须立即并行启动

```text
A1: P0 boundary reproduction + no-audit-only guard
A2: P1 action apply materializer full 2876 actions
A3: P2 outcome materializer smoke, 5% stratified sample
A4: P7 runtime scheduler smoke with dummy/frozen accept mask
```

产物：

```text
p0_v9310_boundary_reproduction.csv
p1_action_apply_trace_v9320.csv
p2_secondary_control_outcome_smoke.csv
p7_event_runtime_smoke.csv
```

## Batch B：P1 pass 后

```text
B1: P2 official-minimum sample outcome materialization
B2: P7 zero-candidate skip measured runtime
B3: P4 action-conditioned feature computation without control leakage
B4: P5 primitive autopsy preliminary using P2 smoke
```

## Batch C：P2 official sample ready 后

```text
C1: P3 useful/control oracle frontier
C2: P4 bad-tail observability full analysis
C3: P6 cross-fitted minimal controller if P3/P4 pass
C4: P10 diagnostic paired replay scout, isolated
```

## Batch D：Decision/runtime integration

```text
D1: P7 real-controller measured runtime
D2: P8 system integration
D3: P9 LDO/LSO official if P8 pass
```

## Batch E：Causal and long-run

```text
E1: P11 official paired replay if P8/P9 pass
E2: P12 short-run if P11 pass
E3: P12 full-run / robustness / continual if short-run pass
```

---

# 11. Required artifacts

```text
run_manifest.json
contract_audit_v9320.csv
p0_v9310_boundary_reproduction.csv
p0_no_audit_only_guard.csv
p1_action_apply_materializer.csv
action_apply_trace_v9320.csv
action_payload_replay_manifest_v9320.json
p2_secondary_control_outcome_materializer.csv
secondary_control_outcome_trace_v9320.csv
matched_control_outcome_trace_v9320.csv
p3_useful_control_oracle_frontier.csv
control_oracle_frontier_trace_v9320.csv
p4_action_conditioned_feature_factory.csv
action_feature_trace_v9320.csv
bad_tail_observability_trace_v9320.csv
feature_cost_trace_v9320.csv
p5_action_primitive_insufficiency_autopsy.csv
action_primitive_autopsy_trace_v9320.csv
p6_crossfitted_action_value_controller.csv
controller_frontier_trace_v9320.csv
p7_measured_online_event_runtime.csv
runtime_event_scheduler_trace_v9320.csv
runtime_empty_event_semantics_trace_v9320.csv
runtime_component_trace_v9320.csv
p8_system_legal_controller_v9320.csv
system_controller_trace_v9320.csv
p9_leave_dataset_stratum_out.csv
leaveout_trace_v9320.csv
p10_diagnostic_paired_replay_scout.csv
diagnostic_paired_replay_trace_v9320.csv
p11_official_paired_replay.csv
official_paired_replay_trace_v9320.csv
p12_short_full_robustness_continual.csv
short_full_continual_trace_v9320.csv
route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

---

# 12. Failure taxonomy

```text
F1_contract_violation
F2_dataset_tuning_detected
F3_teacher_or_loss_modification_detected
F4_fake_or_proxy_violation
F5_audit_only_round_detected
F6_action_payload_missing
F7_action_apply_error_missing
F8_action_apply_error_too_large
F9_no_event_preservation_fail
F10_base_adamw_equivalence_fail
F11_secondary_outcome_missing
F12_matched_control_missing
F13_official_sample_coverage_too_low
F14_control_oracle_absent
F15_primary_oracle_only_no_control_value
F16_bad_tail_observability_fail
F17_action_conditioned_feature_unpredictive
F18_feature_cost_unmeasured
F19_feature_cost_too_high
F20_opaque_feature_pile_detected
F21_minimality_gate_fail
F22_no_dataset_agnostic_controller
F23_decision_precision_fail
F24_decision_coverage_fail
F25_decision_bad_event_fail
F26_decision_null_rate_fail
F27_control_value_fail
F28_support_balance_fail
F29_action_primitive_insufficient
F30_zero_candidate_runtime_fail
F31_empty_event_semantics_fail
F32_active_step_launch_count_fail
F33_runtime_measured_path_missing
F34_step_ratio_fail
F35_memory_ratio_fail
F36_diagnostic_promoted_to_official
F37_source_gap_or_formula_proxy_used
F38_projection_used_for_official
F39_leave_dataset_out_fail
F40_leave_stratum_out_fail
F41_paired_replay_control_equivalent
F42_shuffle_control_pass
F43_functional_lr_equivalent
F44_short_run_task_drop
F45_full_run_no_macro_or_hard_stratum_gain
F46_strong_baseline_explains_gain
F47_sample_efficiency_fail
F48_continual_forgetting_fail
F49_robustness_fail
F50_external_not_ready
```

---

# 13. Route decision

```text
R1-BoundaryReproduced_NoAuditGuardInstalled:
  v9.3.1 boundary reproduced and no-audit-only guard active.

R2-ActionApplyMaterializationFail:
  action payload replay/apply error cannot be measured or exceeds tolerance.

R3-ActionLifecyclePass:
  every action has replayable payload and measured apply error within tolerance.

R4-SecondaryControlOutcomeMaterializationFail:
  secondary/control outcome materializer missing or insufficient coverage.

R5-SecondaryControlOutcomeReady:
  official sample has matched controls and secondary outcomes.

R6-ControlOraclePass:
  current action population has useful/control oracle frontier.

R7-ControlOracleAbsent_ActionPrimitiveInsufficient:
  primary oracle exists but useful/control oracle absent.

R8-ActionConditionedObservabilityPass:
  legal action-conditioned features can separate bad-tail/value/null.

R9-ActionConditionedObservabilityFail:
  features cannot reduce bad-tail under coverage constraints.

R10-ActionValueControllerPass:
  cross-fitted minimal controller passes decision/control gates.

R11-ActionValueControllerFail:
  features or controller cannot produce heldout accepted region.

R12-MeasuredEventRuntimePass:
  online event runtime meets step/memory/semantics gates.

R13-MeasuredEventRuntimeFail:
  runtime remains too slow or diagnostic-only.

R14-SystemLegalControllerPass:
  action lifecycle + outcomes + controller + runtime + contracts pass.

R15-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R16-LeaveStratumOutPass:
  controller generalizes across held-out strata.

R17-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R18-PairedReplayFail:
  system legal but RealFunctional is control-equivalent.

R19-ShortRunFunctionalPass:
  short-run continuous training task-safe gain.

R20-FullFunctionalPass:
  full run task / geometry / calibration / robustness / system pass.

R21-ExternalReady:
  strict PureKAN functional route reaches external-ready standard.
```

`route_decision.json` 必须记录：

```text
route
base_candidate
action_primitive_id
candidate_count
action_count
event_count
primary_oracle_pass
control_oracle_pass
action_lifecycle_pass
action_apply_error_measured
action_apply_error_missing_count
action_apply_error_linf_max
action_apply_error_relative_max
secondary_control_outcome_ready
missing_secondary_delta_count
matched_control_count_per_event
official_sample_coverage
best_observable_id
best_observable_auc_bad
best_observable_auc_safe
best_observable_Top273_safe_count
best_observable_Top273_bad_count
best_observable_Top273_null_count
action_conditioned_feature_pass
best_action_feature_id
best_action_feature_auc_bad
best_action_feature_Top273_bad_count
feature_cost_pass
ogp_decision_pass
controller_id
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
precision_lcb
bad_event_ucb
value_lcb
control_value_lcb
beats_adamwparallel_rate
beats_bestlr_rate
support_balance_pass
runtime_candidate_id
runtime_mode
zero_candidate_controller_kernel_count
zero_candidate_controller_sync_count
controller_launches_per_active_step_q90
controller_syncs_per_active_step_q90
step_ratio_q90
memory_ratio
no_event_preservation_pass
base_adamw_equivalence_on_zero_candidate_steps
measured_event_runtime_pass
system_legal_controller_pass
leave_dataset_out_pass
leave_stratum_out_pass
diagnostic_paired_replay_used_for_controller
paired_replay_pass
short_run_pass
full_run_pass
sample_efficiency_pass
continual_pass
robustness_pass
strong_baseline_pass
external_ready
primary_blocker
next_required_implementation
success_v9320_strict_purekan_functional
success_v9320_full_functional
success_v9320_external_ready
fake_data_used
proxy_row_used
cpu_offload_used
uses_dataset_name_for_controller
uses_loss_backward
uses_teacher
uses_loss_modification
```

---

# 14. 停止条件

## Minimum diagnostic success

```text
P0 boundary reproduced；
P1 action apply materializer measured；
P2 secondary/control materializer measured；
P3 useful/control oracle measured；
P4 action-conditioned observability measured；
P7 measured event runtime measured；
no fake/proxy/offload/teacher/loss violation。
```

## Pivot to action materializer implementation

如果：

```text
action_apply_error_missing_count > 0
```

则停止 controller work，修 P1。

## Pivot to secondary/control materializer implementation

如果：

```text
matched_control_count_per_event < 4
or missing_secondary_delta_count_official_sample > 0
```

则停止 controller work，修 P2。

## Pivot to action primitive redesign

如果：

```text
primary_oracle_pass = 1
but control_oracle_pass = 0
```

或者：

```text
control oracle coverage < 0.03
```

则停止 controller repair，进入 action primitive redesign。

## Pivot to observable redesign

如果：

```text
control_oracle_pass = 1
but P4 action-conditioned observability fail
```

则说明 action population 有用，但 legal pre-commit observability 不够。下一步修 observable primitive，不调 dataset threshold。

## Pivot to controller redesign

如果：

```text
P4 feature pass
but P6 controller fail
```

则修 cross-fitting、support、monotone controller，不新增 opaque feature pile。

## Pivot to runtime redesign

如果：

```text
P6 controller pass
but P7 runtime fail
```

则停止 controller tuning，专注 event-driven runtime。

## Open official causal validation

只有当：

```text
P8 system pass
and P9 LDO/LSO pass
```

才 official 打开 P11 paired replay。

## Full functional success

只有当：

```text
P11 paired replay pass
and P12 short/full/robustness/sample-efficiency/continual pass
```

才能声明 full functional success。

---

# 15. 最终执行建议

v9.3.2 的一句话策略是：

$$
\boxed{
\text{别再审计缺失；直接生成 action apply、control outcome、event runtime 三个缺失闭环。}
}
$$

最关键的优先级：

```text
1. 先让每个 action 可重放、可 apply、可误差审计；
2. 再让 RealFunctional 与 controls 的 secondary outcomes 存在；
3. 再判断 primary oracle 是否升级为 control-useful oracle；
4. 再用 action-conditioned features 解决 bad-tail；
5. 再训练 minimal cross-fitted controller；
6. 同时实现 measured event-driven runtime；
7. 最后才进入 LDO/LSO、paired replay、short/full run。
```

如果 v9.3.2 做完后仍然卡住，应该能明确落入以下几类之一：

```text
A. action payload 不可重放；
B. primary-safe rows 没有 control value；
C. action 有 control value，但 commit 前 observability 不够；
D. observability 足够，但 controller/support 不稳；
E. controller 过线，但 runtime 不过；
F. system 过线，但 paired replay control-equivalent。
```

这才是下一轮应该给出的真正信息量。

