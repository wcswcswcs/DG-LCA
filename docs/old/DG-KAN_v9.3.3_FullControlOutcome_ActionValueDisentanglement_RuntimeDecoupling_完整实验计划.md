# DG-KAN v9.3.3 Full-Control Outcome Materialization / Action-Value Disentanglement / Runtime Decoupling 完整实验计划

> 本计划基于 v9.3.2 `Action Apply / Control Outcome Materialization 与 Event Runtime` 的真实执行结果制定。  
> v9.3.3 不再做 audit-only，不再把 partial immediate-probe controls 转正，也不再继续找单个 action feature 阈值。  
> 本轮目标是把 **action-value 这个对象真正 materialize 完整**，并把 **online runtime 与 offline outcome materializer runtime 分离测量**。

---

# 0. 执行摘要

v9.3.2 的 terminal route 是：

```text
route = R4-SecondaryControlOutcomeMaterializationFail
base_candidate = LQ-t2-h256
success_v9320_strict_purekan_functional = False
success_v9320_full_functional = False
success_v9320_external_ready = False
```

我对 v9.3.2 的独立判断是：

$$
\boxed{
\text{v9.3.2 有真实推进，但推进的是 action apply 物理闭合；不是 controller / runtime / causal success。}
}
$$

v9.3.2 做成了一件关键事情：它把 v9.3.1 中完全缺失的 action apply error 测量推进为全量闭合：

```text
action_count = 2876
action_apply_rows = 2876
action_apply_error_measured = 1
action_apply_error_missing_count = 0
replay_success_count = 2876
action_apply_error_linf_max = 7.450580596923828e-09
action_apply_error_relative_max = 1.1583176888604244e-05
action_apply_cosine_logged_applied_min = 0.9999999932719165
action_lifecycle_pass = 1
```

这说明当前 logged functional action 不是“纸面事件”，它确实可以被重新 apply，并且 logged/applied 数值误差极小。这是 v9.3.2 最大进展。

但 v9.3.2 没有完成 action-value controller 所需的核心材料：

```text
secondary/control probe sample action = 143
official_sample_coverage = 0.04972183588317107
matched_control_count_per_event_min = 2
required_matched_control_count = 4
branch_missing_count = 429
horizon_missing_count = 286
missing_secondary_delta_count = 1716
secondary_outcome_ready = 0
control_oracle_pass = 0
```

因此，v9.3.2 不能回答：

```text
RealFunctional 是否打过 AdamWParallel？
RealFunctional 是否打过 bestLR？
RealFunctional 是否只是 NoOp / safe but useless？
RealFunctional 是否在 horizon = 20 / 80 / 240 下稳定？
action-value 的 downside distribution 是什么？
```

v9.3.2 的 P3 primary oracle 仍然很强：

```text
OR1-PrimarySafeGoodOracleRetained:
  accepted_count = 494
  precision = 1.0
  coverage = 0.05445326278659612
  bad_event = 0.0
  null_rate = 0.0
  primary_oracle_pass = 1
```

但这只是 primary safe-good oracle，不是 control-positive oracle。现在不能把它解释成 functional action 有因果优势，因为 matched controls 与 secondary horizons 没有闭合。

v9.3.2 的 P4 action-conditioned feature 结果也很重要：

```text
best_feature_id = AVF1-PayloadNorm
best_feature_top273_safe_good = 155
best_feature_top273_bad_event = 20
best_feature_top273_null_event = 81
action_conditioned_feature_pass = 0
```

在 accepted count = 273 的 official coverage 下限处，decision gate 大致要求：

```text
safe_good >= 205
bad_event <= 13
null_event <= 40
```

因此 `AVF1-PayloadNorm` 的缺口是：

```text
safe_good deficit = 205 - 155 = 50
bad_event excess = 20 - 13 = 7
null_event excess = 81 - 40 = 41
```

这个结果不能理解成“PayloadNorm 差一点”。它说明 PayloadNorm 更像 **action magnitude / nontriviality signal** 或 **粗风险 signal**，但不是 value sufficient statistic。它能把 bad 从 v9.3.1 CandidateScore 的 high-bad 区域往下拉，但同时丢掉大量 safe-good，并引入大量 null。

v9.3.1 的 `OGP-C1-CandidateScore` 在 top273 上是：

```text
safe = 210
bad = 49
null = 16
```

它能找到足够 safe / non-null，但 bad-tail 太多。v9.3.2 的 `AVF1-PayloadNorm` 是：

```text
safe = 155
bad = 20
null = 81
```

它压低一部分 bad-tail，但严重损失 useful/safe 和 non-null。

这揭示了一个比“调阈值”更深的问题：

$$
\boxed{
\text{当前可观测量不是完全没信号，而是 value、bad-risk、null-risk 被不同弱信号分裂承载。}
}
$$

因此 v9.3.3 不应该继续找单个 best feature，也不应该把 CandidateScore 和 PayloadNorm 硬拼成一个 opaque score。正确方向是把 controller 明确拆成：

```text
Value head
Bad-risk head
Null-risk head
Support head
Cost head
```

形式上仍然是：

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

v9.3.2 的 runtime 也不能写成进展过线：

```text
runtime_candidate_id = RT2-ActionReplayEventSchedulerMeasuredPartial
runtime_measured = 1
zero_candidate_controller_kernel_count = 0
zero_candidate_controller_sync_count = 0
controller_launches_per_active_step_q90 = 3.0
controller_syncs_per_active_step_q90 = 1.0
step_ratio_q90 = 5.697758752709664
event_driven_runtime_pass = 0
```

这里的真实含义是：

```text
zero-candidate skip semantics 已经被 measured；
但 active-step path 仍没有 fused 到要求；
更严重的是 action replay / materializer path 与 online runtime 被混在一起，导致 step_ratio_q90 从 2.213 上升到 5.698。
```

v9.3.3 必须把 runtime 拆成两条：

```text
1. online_sequential_official_runtime:
   只测训练时真正需要的 cheap feature / score / accept / apply。
   不跑 matched controls，不跑 secondary outcome materializer，不写 heavy audit。

2. offline_outcome_materializer_runtime:
   专门为 matched controls / horizons / labels 服务。
   可以 batch-major、parallel、跨 branch/horizon 调度。
   它不作为 online system step_ratio official gate。
```

---

# 1. v9.3.2 独立结论

## 1.1 有进展吗？

有，而且是实质工程进展。v9.3.2 清掉了 v9.3.1 的一个硬 blocker：

```text
v9.3.1:
  action_apply_error_measured = 0
  action_apply_error_missing_count = 2876
  action_lifecycle_pass = 0

v9.3.2:
  action_apply_error_measured = 1
  action_apply_error_missing_count = 0
  replay_success_count = 2876 / 2876
  action_lifecycle_pass = 1
```

这说明 action primitive 的物理 apply 路径已经可以审计。之前我们只能说“有 action id / payload hash / primary label”，现在可以说“logged action 和 applied action 数值一致”。

这一步很重要，因为如果 action apply 不闭合，那么任何 action-value / paired replay / control oracle 都没有物理基础。

但这个进展有两个限制：

```text
1. payload_tensor_file_written = 0；
   当前是同轮 train-stream 重建 + 内存 replay，不是 durable tensor payload package。

2. action apply pass 只证明 action 可以被 apply；
   不证明 action 有 value，也不证明 action 优于 controls。
```

所以 v9.3.2 是：

$$
\boxed{
\text{Action lifecycle local closure}
}
$$

不是：

$$
\boxed{
\text{Functional controller closure}
}
$$

## 1.2 进度如何？

从 v9.3.0 到 v9.3.2，项目已经完成了这条转向：

```text
v9.3.0:
  StableAccept patch line exhausted；
  candidate/action population 有 primary oracle frontier；
  legal OGP feature/controller 找不到 frontier；
  secondary/control outcome 和 measured runtime 未闭合。

v9.3.1:
  严格审计 action apply / secondary-control / runtime；
  发现 action apply error、secondary/control outcome、measured event runtime 都缺失。

v9.3.2:
  action apply error 全量 materialize 并数值闭合；
  secondary/control 只做到 partial immediate-probe；
  measured scheduler trace 有 zero-candidate skip，但 runtime 不过。
```

这是一条正确的推进路线：从“StableAccept repair”转向“action-value materialization”。

但进度仍处在 **pre-controller** 阶段。现在还没有 official controller，因为 controller 所需的 $V/B/N/S/C$ 中，只有 action apply 的物理一致性闭合了；$V$ 和 control-relative value 仍然不存在。

我会把当前阶段标为：

```text
Stage = action_physics_closed_but_action_value_unmaterialized
```

## 1.3 当前卡在哪里？

当前最大 blocker 是：

$$
\boxed{
\text{full matched-control secondary outcomes and horizons are not materialized.}
}
$$

具体表现：

```text
matched_control_count_per_event_min = 2 < 4
branch_missing_count = 429
horizon_missing_count = 286
missing_secondary_delta_count = 1716
secondary_outcome_ready = 0
control_oracle_pass = 0
```

这不是“多补几个字段”的小问题。它决定了我们能不能定义真正的 action value：

$$
V_{ctrl}(e)
=
V_{RealFunctional}(e)
-
\max
\left(
V_{AdamWParallel}(e),
V_{bestLR}(e),
V_{NoOp}(e),
V_{Random}(e)
\right).
$$

如果 $V_{ctrl}(e)$ 不存在，那么：

```text
primary oracle 只是 safe-good oracle；
legal feature 只是 primary label diagnostic；
controller 只能选择“看起来安全”的 action；
paired replay 不能 official；
full functional success 不能判断。
```

第二个 blocker 是 action observability：

```text
best action feature = AVF1-PayloadNorm
top273 safe/bad/null = 155 / 20 / 81
action_conditioned_feature_pass = 0
```

这说明现有 action-conditioned 单特征不能作为 accept sufficient statistic。

第三个 blocker 是 runtime：

```text
zero-candidate controller kernel/sync = 0 / 0
controller_launches_per_active_step_q90 = 3.0
step_ratio_q90 = 5.697758752709664
```

zero-candidate skip 已经不再是主要 blocker。新的 runtime blocker 是：

```text
active-step path not fused enough；
action replay/materializer path too slow；
online runtime 与 offline materializer runtime 混在同一条 measured path 中。
```

## 1.4 发现了什么本质问题？

v9.3.2 发现的本质问题不是“feature 不够多”，而是 **action-value 对象仍然没有被完整定义**。

现在我们有：

```text
Action identity: yes
Action apply: yes
Primary label: yes
Primary oracle: yes
Partial immediate controls: yes
Full control-relative value: no
Full horizon robustness: no
Legal action-value observable: no
Online runtime closure: no
```

这意味着继续做 feature/threshold search 会重复过去的错误：

```text
StableAccept patching
  -> OGP feature patching
  -> Action feature patching
```

这些都不是第一性原理路线。第一性问题应该是：

$$
\boxed{
\text{当前 functional action 是否在 strong controls 下产生可选择的正 value？}
}
$$

如果答案未知，就不该先做 controller 转正。

## 1.5 是否在正确道路上？

高层方向仍然对。v9.3.2 没有把 partial controls 写成 official，没有把 primary oracle 写成 functional success，也没有把 measured scheduler trace 写成 runtime pass。这些 gate discipline 是正确的。

但 v9.3.3 必须避免两个错误方向：

```text
错误方向 1:
  继续扩展 AVF / OGP feature list，试图用更多特征弥补没有 full control outcome 的事实。

错误方向 2:
  继续在 action replay materializer path 上优化 step_ratio，然后把它当 online runtime。
```

正确方向是：

```text
1. materialize full matched controls and horizons；
2. 用 full outcomes 建立 control-positive oracle；
3. 再评估 legal action-value observability；
4. 同时独立实现 online runtime，不把 offline materializer cost 混进 online gate。
```

## 1.6 离目标还差多远？

距离可以分成四层。

### 距离 system-legal local controller

还差至少三个 hard gates：

```text
1. full secondary/control outcome ready；
2. action-value observable/controller pass；
3. online event-driven runtime pass。
```

v9.3.2 只完成了 action lifecycle pass。

### 距离 strict PureKAN functional causal evidence

还差：

```text
1. system-legal controller；
2. leave-dataset-out；
3. leave-stratum-out；
4. official paired replay；
5. RealFunctional beats AdamWParallel / bestLR / NoOp / Random / shuffled controls。
```

### 距离 full functional success

还差：

```text
short-run；
full-run；
strong LR grid；
QuadraticFeatureMLP；
robustness；
calibration / geometry / hard-stratum advantage；
sample efficiency；
continual / anti-forgetting。
```

### 距离 external ready

还差：

```text
durable payload tensor artifacts；
full reproducibility package；
no fragile finalization errors；
runtime official path；
cross-run artifact integrity；
external rerun without source-specific hidden state。
```

因此当前不能说“快成功了”。更准确是：

$$
\boxed{
\text{项目已经从 StableAccept patching 走出，但刚进入 action-value materialization 的核心阶段。}
}
$$

---

# 2. v9.3.3 总体目标

v9.3.3 的总体目标是：

$$
\boxed{
\text{把 action-value 从 partial probe 变成 full matched-control outcome table，并把 online runtime 与 offline materializer runtime 解耦。}
}
$$

本轮不追求直接 full functional success。v9.3.3 的强目标是：

```text
1. durable action payload artifact pass；
2. full matched-control outcome materializer pass；
3. control-positive oracle frontier measured；
4. action-value / bad-risk / null-risk legal observability measured；
5. cross-fitted minimal controller measured；
6. online event-driven runtime measured independently from outcome materializer；
7. system route 明确落在 action-value pass / action-value fail / runtime fail / primitive insufficient 中之一。
```

v9.3.3 的一句话策略：

$$
\boxed{
\text{不再补 AVF 单特征；先让 controls/horizons 完整存在，再用 disentangled } V/B/N/S/C \text{ controller 判断 action 是否可部署。}
}
$$

---

# 3. v9.3.3 不做什么

本轮明确不做：

```text
1. 不把 partial immediate-probe controls 转 official；
2. 不用 primary oracle 替代 control oracle；
3. 不继续找单个 best action feature 阈值；
4. 不把 CandidateScore + PayloadNorm 直接拼成 opaque score 后宣布 pass；
5. 不按 dataset 调 threshold / branch / horizon / runtime route；
6. 不用 test/validation metric at commit time；
7. 不用 teacher / distillation / auxiliary loss / loss modification；
8. 不把 offline outcome materializer runtime 写成 online training runtime；
9. 不把 diagnostic paired replay 结果反向用于 P5 controller 阈值；
10. 不在 full controls/horizons 缺失时打开 official paired replay；
11. 不把 durable payload missing 的 run 写成 external-ready。
```

允许做：

```text
1. dataset-level diagnostics；
2. family / horizon / bucket failure diagnosis；
3. leave-dataset-out / leave-stratum-out；
4. calibration split 上冻结阈值；
5. offline batch-major outcome materializer；
6. parallel branch/horizon outcome computation；
7. diagnostic paired replay scout，但必须隔离；
8. action-value oracle upper bound；
9. minimal monotone controller；
10. online runtime microbenchmark 与 offline materializer throughput 分开记录。
```

---

# 4. 核心假设

## H0：v9.3.2 action apply closure 是真实进展，但 external-ready 仍需 durable payload

H0 认为 v9.3.2 的 action apply replay 数值闭合可信，但由于没有 `.pt` / durable tensor payload package，仍不能 external-ready。

H0 成立标准：

```text
action_payload_tensor_file_written = 1
payload_shard_count >= 1
all action_id have payload_hash
replay_from_disk_success_count = 2876
replay_from_disk_error_linf_max <= 1e-6 or pre-registered tolerance
replay_from_disk_cosine_min >= 0.999999
```

H0 失败标准：

```text
disk replay 与 in-memory replay 误差明显不同；
payload hash 不稳定；
action_id 无法对应 payload tensor。
```

## H1：primary oracle frontier 不等于 useful/control oracle frontier

H1 认为 v9.3.0-v9.3.2 的 primary safe-good oracle 只说明 candidate/action population 中有安全行，不说明 functional action 优于 controls。

H1 成立标准：

```text
primary_oracle_pass = 1
but control_oracle_pass must be separately measured
```

若 control oracle fail，即使 primary oracle pass，也必须进入：

```text
R-ActionPrimitiveSafeButNotUseful
```

## H2：当前 feature failure 是 value/bad/null 信号分裂，不是单纯 feature absence

H2 认为 CandidateScore 和 PayloadNorm 各自携带不同信息：

```text
CandidateScore:
  safe / useful signal stronger
  bad-tail risk weak

PayloadNorm:
  bad-tail risk somewhat lower
  null-risk and useful-value weak
```

H2 成立标准：

```text
2D feature frontier improves over both single-feature frontiers
but opaque high-dimensional feature pile is not required。
```

H2 失败标准：

```text
2D / minimal monotone disentangled controller cannot improve any gate；
legal observability remains insufficient。
```

## H3：必须用 full controls/horizons 定义 $V_{ctrl}$

H3 认为 action-value 必须按 branches 和 horizons 定义：

$$
V_{ctrl}(e,h)
=
V_{RealFunctional}(e,h)
-
\max
\left(
V_{AdamWParallel}(e,h),
V_{bestLR}(e,h),
V_{NoOp}(e,h),
V_{Random}(e,h)
\right).
$$

H3 成立标准：

```text
for official candidate sample:
  branch_missing_count = 0
  horizon_missing_count = 0
  missing_secondary_delta_count = 0
  matched_control_count_per_event_min >= 4
  horizons = 20, 80, 240 complete
```

H3 失败标准：

```text
P2 又只做到 partial immediate-probe；
coverage < official threshold；
controls/horizons missing。
```

## H4：online runtime failure 不能用 offline materializer path 判断

H4 认为 v9.3.2 的 step_ratio_q90 = 5.697759 主要反映 action replay / materializer path，而非最终 online controller 的必要成本。

H4 成立标准：

```text
online_sequential_official_runtime measured separately；
offline_outcome_materializer_runtime measured separately；
online runtime excludes matched controls, secondary horizons, heavy CSV/hash audit；
online step_ratio_q90 <= 1.50 or failure attribution identifies true online component。
```

H4 失败标准：

```text
online path 和 offline materializer path 继续混在一起；
step_ratio 无法解释；
runtime_candidate_id 不区分 runtime_mode。
```

## H5：如果 control-positive oracle 都不存在，问题回到 action primitive，不是 controller

H5 成立标准：

```text
control_oracle_pass = 0
even with outcome labels and oracle access
```

此时停止 controller search，进入：

```text
candidate/action primitive redesign
functional update source redesign
horizon/value target redesign
```

## H6：dataset 只能用于诊断，不用于 tuning

H6 成立标准：

```text
uses_dataset_name_for_controller = 0
thresholds frozen on calibration folds
dataset diagnostics reported separately
leave-dataset-out evaluated
no dataset-specific branch / threshold / runtime route。
```

H6 失败标准：

```text
任何 controller 逻辑中出现 dataset branch；
任何 threshold 按 MNIST / FMNIST / KMNIST 单独选择。
```

---

# 5. v9.3.3 数据合同

## 5.1 Action payload durable artifact contract

每个 action 必须有可重放的 durable payload：

```text
action_id
candidate_id
event_id
payload_schema_version
payload_tensor_file
payload_tensor_offset
payload_tensor_shape
payload_tensor_dtype
payload_tensor_device_original
payload_hash
payload_norm
payload_l2_norm
payload_linf_norm
payload_role_entropy
functional_delta_norm
functional_delta_hash
logged_apply_hash
disk_replay_apply_hash
in_memory_replay_apply_hash
action_apply_error_linf
action_apply_error_relative
action_apply_cosine_logged_applied
```

Pass：

```text
payload_tensor_file_written = 1
payload_hash_missing_count = 0
payload_tensor_missing_count = 0
replay_from_disk_success_count = action_count
action_apply_error_missing_count = 0
action_apply_error_linf_max <= tolerance
action_apply_cosine_logged_applied_min >= 0.999999
```

## 5.2 Full matched-control outcome contract

Branches official minimum：

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
```

Diagnostic shuffled branches：

```text
ShuffledFunctionalPayload
ShuffledBranchDelta
ShuffledCandidateScore
ShuffledPayloadNorm
ShuffledOGPValueScore
ShuffledOGPRiskScore
```

Horizons official minimum：

```text
horizon = 20
horizon = 80
horizon = 240
```

Optional：

```text
horizon = 640
```

Each branch-horizon row must record：

```text
action_id
candidate_id
event_id
branch
horizon
dataset
seed
step
family_id
bucket_id
signal_stratum_id
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
local_lipschitz_delta
basis_usage_entropy_delta
functional_channel_entropy_delta
task_safe_label
useful_label
bad_event_label
null_event_label
safe_good_label
value_score
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
outcome_runtime_ms
outcome_source
```

Pass：

```text
branch_missing_count = 0
horizon_missing_count = 0
missing_secondary_delta_count = 0
matched_control_count_per_event_min >= 4
matched_control_count_per_event_mean >= 4
official_candidate_coverage >= 0.80
or full_candidate_coverage = 1 for all 2876 actions if feasible
```

If coverage is below threshold：

```text
status = diagnostic_only
controller_promotion_allowed = 0
```

## 5.3 Value definition contract

Define per-branch value as:

$$
V_b(e,h)
=
-w_{ce}\Delta CEp99_b(e,h)
+w_m\Delta MarginP10_b(e,h)
-w_{ece}\Delta ECE_b(e,h)
-w_{nll}\Delta NLL_b(e,h)
-w_c\Delta Curvature_b(e,h).
$$

Weights are pre-registered and not dataset-specific. If weights are uncertain, report each metric separately and use a Pareto oracle.

Control-relative value：

$$
V_{ctrl}(e,h)
=
V_{Real}(e,h)
-
\max
\left(
V_{AdamWParallel}(e,h),
V_{bestLR}(e,h),
V_{NoOp}(e,h),
V_{Random}(e,h)
\right).
$$

Horizon-robust value：

$$
V_{robust}(e)
=
\min_{h\in\{20,80,240\}}
V_{ctrl}(e,h).
$$

Bad-risk：

$$
B(e)
=
P(BadEvent_{Real}(e,h)=1 \text{ for any official } h).
$$

Null-risk：

$$
N(e)
=
P(NullEvent_{Real}(e,h)=1 \text{ for all official } h).
$$

Support：

$$
S(e)
=
n_{eff}(family,horizon,bucket,feature\_bin).
$$

Cost：

$$
C(e)
=
C_{feature}(e)+C_{score}(e)+C_{apply}(e).
$$

## 5.4 Legal action-value observable contract

Feature groups are limited to avoid feature pile:

```text
max_feature_groups_per_controller = 5
must_report_single_feature_baselines = 1
must_report_pairwise_feature_frontiers = 1
must_report_ablation = 1
must_report_monotone_sign_audit = 1
```

Allowed feature groups：

```text
G1 CandidateScore / StableAccept legacy:
  stable_score_q
  stable_rank
  score_margin
  candidate_score

G2 Payload geometry:
  payload_norm
  payload_linf_norm
  payload_role_entropy
  true_delta_norm
  functional_delta_norm

G3 Alignment / descent:
  cos_functional_delta_adamw_delta
  cos_functional_delta_negative_grad
  projected_CE_descent_estimate
  gradient_conflict_rate

G4 Tail state:
  CEp99_current
  margin_p10_current
  wrong_conf_p90
  hard_tail_fraction
  entropy_mean

G5 Support / reliability:
  support_count
  support_lcb
  family_bad_ucb
  horizon_tail_risk
  empirical_bayes_bad_ucb
  empirical_bayes_value_lcb

G6 Cost:
  feature_compute_time_ms
  candidate_count_in_step
  estimated_payload_apply_cost
  controller_launch_cost_estimate
```

Prohibited：

```text
dataset_name at commit
test / validation outcome at commit
future step information
heldout label at commit
source measured gap
formula proxy as official
posthoc safe_good label at commit
```

## 5.5 Runtime contract

Every runtime artifact must specify：

```text
runtime_mode
```

Allowed values：

```text
online_sequential_official_runtime
offline_outcome_materializer_runtime
diagnostic_runtime
```

Online runtime must record：

```text
step_id
active_step_flag
zero_candidate_step_flag
candidate_count_in_step
controller_kernel_launch_count
controller_sync_count
controller_launches_per_active_step
controller_syncs_per_active_step
feature_compute_time_ms
score_accept_time_ms
payload_apply_time_ms
audit_time_ms_outside_timed
csv_hash_time_ms_outside_timed
base_train_step_time_ms
controller_extra_time_ms
total_step_time_ms
mlp_step_time_ms
step_ratio
peak_memory_mb
memory_ratio
accept_disagreement_count
payload_apply_error_max
```

Online pass：

```text
runtime_mode = online_sequential_official_runtime
zero_candidate_controller_kernel_count = 0
zero_candidate_controller_sync_count = 0
controller_launches_per_active_step_q90 <= 2
controller_syncs_per_active_step_q90 <= 1
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
audit_outside_timed_path = 1
matched_controls_in_timed_online_path = 0
secondary_horizon_replay_in_timed_online_path = 0
```

Offline materializer must record throughput, not online step ratio：

```text
branch_horizon_rows_per_second
actions_per_second
gpu_utilization
materializer_wallclock
materializer_memory_peak
branch_horizon_completion_rate
```

---

# 6. v9.3.3 实验阶段

---

## P0：v9.3.2 boundary independent reanalysis

### 目标

复现 v9.3.2 的 route 和 gate gaps，计算本轮最小需要修复的差距。P0 不做新实验，只做独立审计。

### 假设

H0/P0：v9.3.2 的 action apply pass 可信；主 blocker 是 secondary/control outcome incomplete 和 runtime path conflation。

### 必须记录

```text
source_route_v9320
candidate_count
action_count
event_count
action_apply_error_measured
action_apply_error_missing_count
action_apply_error_linf_max
action_apply_error_relative_max
action_apply_cosine_logged_applied_min
payload_tensor_file_written
secondary_outcome_ready
outcome_sample_action_count
official_sample_coverage
matched_control_count_per_event_min
branch_missing_count
horizon_missing_count
missing_secondary_delta_count
primary_oracle_pass
primary_oracle_accepted_count
primary_oracle_precision
primary_oracle_coverage
control_oracle_pass
best_action_feature_id
best_action_feature_top273_safe_good
best_action_feature_top273_bad_event
best_action_feature_top273_null_event
safe_needed_at_273
bad_allowed_at_273
null_allowed_at_273
runtime_candidate_id
runtime_mode
zero_candidate_controller_kernel_count
controller_launches_per_active_step_q90
step_ratio_q90
event_driven_runtime_pass
```

### 判断标准

P0 pass：

```text
v9.3.2 route reproduced；
action apply pass reproduced；
control outcome missing reproduced；
feature top273 gap computed；
runtime measured failure reproduced。
```

### 可视化

```text
p0_v9320_gate_ladder.svg
p0_action_apply_error_distribution.svg
p0_top273_feature_gap_bar.svg
p0_primary_oracle_vs_control_oracle_boundary.svg
p0_runtime_online_vs_materializer_confusion.svg
```

---

## P1：durable action payload package and disk replay

### 目标

把 v9.3.2 的 in-memory replay closure 推进为 durable payload closure，使 action apply 可以跨进程、跨 run、跨机器审计。

### 假设

H0：in-memory replay closure 可以转化为 disk replay closure。

### 实现

新增：

```text
experiments/run_v9330_durable_action_payload_package.py
```

输出：

```text
action_payload_shards_v9330/
action_payload_manifest_v9330.json
action_payload_replay_from_disk_trace_v9330.csv
p1_durable_action_payload_package.csv
```

### 必须记录

```text
action_id
candidate_id
event_id
payload_shard_path
payload_tensor_offset
payload_tensor_shape
payload_tensor_dtype
payload_hash_expected
payload_hash_loaded
payload_hash_match
disk_replay_success
disk_replay_error_linf
disk_replay_error_relative
disk_replay_cosine_logged_applied
in_memory_vs_disk_error_linf
in_memory_vs_disk_cosine
```

### 判断标准

P1 pass：

```text
payload_tensor_file_written = 1
payload_hash_match_rate = 1.0
disk_replay_success_count = 2876
disk_replay_error_linf_max <= 1e-6 or justified tolerance
disk_replay_cosine_logged_applied_min >= 0.999999
payload_tensor_missing_count = 0
```

### 可视化

```text
p1_disk_replay_error_hist.svg
p1_payload_norm_distribution.svg
p1_payload_hash_audit.svg
p1_in_memory_vs_disk_replay_scatter.svg
```

---

## P2：full matched-control outcome materializer

### 目标

把 v9.3.2 的 partial immediate-probe controls 扩展为 full matched-control outcome table。P2 是 v9.3.3 的核心 blocking phase。没有 P2，就没有 action-value controller。

### 假设

H3：full branch/horizon controls 可以被 materialize，并且成本可通过 offline batch-major / parallel execution 控制。

### 实现

新增：

```text
experiments/run_v9330_full_matched_control_outcome_materializer.py
```

Execution layout：

```text
parallel axis 1: branch
parallel axis 2: horizon
parallel axis 3: dataset / seed
parallel axis 4: action shard
```

Official branches：

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
```

Official horizons：

```text
20
80
240
```

Diagnostic shuffled branches：

```text
ShuffledFunctionalPayload
ShuffledBranchDelta
ShuffledCandidateScore
ShuffledPayloadNorm
```

Materialization tiers：

```text
Tier Full:
  all 2876 actions × official branches × official horizons。

Tier Official-Minimum fallback:
  union of:
    primary oracle accepted rows；
    StableAccept accepted rows；
    top273 CandidateScore rows；
    top273 PayloadNorm rows；
    top273 2D frontier rows；
    matched rejected rows by family/horizon/bucket/seed；
    stratified random sample。
  official_candidate_coverage >= 0.80。

Tier Diagnostic-only:
  coverage < 0.80。
  Cannot enter P5 controller official.
```

### 必须记录

```text
action_id
candidate_id
event_id
branch
horizon
dataset
seed
step
family_id
bucket_id
signal_stratum_id
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
local_lipschitz_delta
basis_usage_entropy_delta
task_safe_label
useful_label
bad_event_label
null_event_label
safe_good_label
value_score
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
outcome_runtime_ms
branch_completed
horizon_completed
outcome_source
```

Summary fields：

```text
action_count
branch_horizon_row_count_expected
branch_horizon_row_count_actual
official_candidate_coverage
matched_control_count_per_event_min
matched_control_count_per_event_mean
branch_missing_count
horizon_missing_count
missing_secondary_delta_count
branch_completion_rate
horizon_completion_rate
materializer_wallclock_sec
materializer_rows_per_sec
offline_runtime_mode
fake_data_used
proxy_row_used
```

### 判断标准

P2 full pass：

```text
branch_missing_count = 0
horizon_missing_count = 0
missing_secondary_delta_count = 0
matched_control_count_per_event_min >= 4
official_candidate_coverage = 1.0
```

P2 official-minimum pass：

```text
branch_missing_count = 0 for official sample
horizon_missing_count = 0 for official sample
missing_secondary_delta_count = 0 for official sample
matched_control_count_per_event_min >= 4
official_candidate_coverage >= 0.80
family × horizon × bucket coverage sufficient
```

P2 diagnostic only：

```text
official_candidate_coverage < 0.80
or branch/horizon missing remains nonzero
```

### 可视化

```text
p2_branch_horizon_completion_heatmap.svg
p2_missing_secondary_delta_before_after.svg
p2_real_vs_controls_value_scatter.svg
p2_branch_value_distribution.svg
p2_horizon_value_stability.svg
p2_materializer_throughput_by_branch_horizon.svg
p2_sample_coverage_by_family_horizon_bucket.svg
```

---

## P3：control-positive oracle and impossibility test

### 目标

判断当前 action population 是否不仅有 primary safe-good frontier，而且有 control-positive frontier。如果 oracle 都打不过 controls，controller 再好也没意义。

### 假设

H1/H5：control oracle 是 action primitive 是否值得继续的必要条件。

### Oracle definitions

Primary oracle：

$$
OraclePrimary(e)=1
\iff
SafeGood_{Real}(e)=1
\land
BadEvent_{Real}(e)=0
\land
NullEvent_{Real}(e)=0.
$$

Control-positive oracle：

$$
OracleControl(e)=1
\iff
V_{ctrl}(e)>0
\land
BadEvent_{Real}(e)=0
\land
NullEvent_{Real}(e)=0.
$$

Horizon-robust oracle：

$$
OracleRobust(e)=1
\iff
\min_{h\in\{20,80,240\}}V_{ctrl}(e,h)>0
\land
\max_h BadEvent(e,h)=0.
$$

Pareto oracle：

```text
Select rows on Pareto frontier of:
  maximize V_ctrl
  minimize bad risk
  minimize null risk
  maximize support
  minimize cost
```

### Oracle candidates

```text
OR0-PrimarySafeGoodOracleRetained
OR1-ControlPositiveOracle
OR2-HorizonRobustControlOracle
OR3-ValueRiskParetoOracle
OR4-SupportBalancedControlOracle
OR5-NoOpDominanceOracle
OR6-AdamWParallelDominanceOracle
OR7-BestLRDominanceOracle
```

### 必须记录

```text
oracle_id
accepted_count
coverage
precision_primary
bad_event_rate
null_rate
value_mean
value_lcb
beats_adamwparallel_rate
beats_bestlr_rate
beats_noop_rate
beats_random_rate
horizon_robust_pass
accepted_family_count
accepted_signal_strata_count
max_family_share
max_stratum_share
support_balance_pass
oracle_uses_outcome_label = 1
```

### 判断标准

Weak control oracle pass：

```text
coverage >= 0.03
precision_primary >= 0.75
bad_event_rate <= 0.05
null_rate <= 0.15
beats_adamwparallel_rate >= 0.50
beats_bestlr_rate >= 0.50
support_balance_pass = 1
```

Strong control oracle pass：

```text
coverage >= 0.03
precision_primary >= 0.90
bad_event_rate <= 0.02
null_rate <= 0.10
beats_adamwparallel_rate >= 0.60
beats_bestlr_rate >= 0.60
horizon_robust_pass = 1
support_balance_pass = 1
```

If weak control oracle fail：

```text
route = R30-ActionPrimitiveControlFrontierAbsent
stop controller search
pivot to action primitive redesign
```

### 可视化

```text
p3_primary_vs_control_oracle_frontier.svg
p3_value_risk_pareto.svg
p3_real_vs_adamwparallel_value_scatter.svg
p3_real_vs_bestlr_value_scatter.svg
p3_horizon_robust_oracle_matrix.svg
p3_control_oracle_support_balance.svg
```

---

## P4：action-value observability and feature disentanglement

### 目标

不再找单个 best feature，而是评估 legal pre-commit observables 是否能分别预测 value、bad-risk、null-risk、support 和 cost。P4 不做 final controller，只做 observability。

### 假设

H2：CandidateScore 与 PayloadNorm 是互补弱信号；minimal disentangled observables 可能形成 frontier。

### Feature groups

```text
F0-CandidateScore
F1-PayloadNorm
F2-DeltaNorm
F3-FunctionalAdamWAlignment
F4-LinearizedCEDescent
F5-TailStateRisk
F6-EmpiricalBayesSupport
F7-HorizonTailRisk
F8-ActionOriginReliability
F9-CostEstimate
```

Minimal pair diagnostics：

```text
Pair-CandidateScore-PayloadNorm
Pair-CandidateScore-Alignment
Pair-CandidateScore-TailRisk
Pair-PayloadNorm-NullRisk
Pair-Alignment-Support
```

### 必须记录

For each feature / pair / minimal group：

```text
feature_id
feature_group_count
uses_dataset_name
uses_outcome_at_commit
uses_future_step
uses_validation_test
feature_missing_rate
feature_compute_time_ms_q90
feature_memory_ratio
AUC_value_positive
AUC_control_positive
AUC_bad_event
AUC_null_event
PR_AUC_bad_lift
Brier_bad
ECE_bad
calibration_slope
Top273_safe_good
Top273_bad_event
Top273_null_event
Top273_control_positive
Top273_beats_adamwparallel
Top273_beats_bestlr
leave_dataset_auc_drop
leave_stratum_auc_drop
monotone_sign_pass
```

### 判断标准

P4 weak observability pass：

```text
at least one legal feature group or 2D pair satisfies:
  AUC_control_positive >= 0.70
  or AUC_bad_event >= 0.75
  or Top273 gate gap improves over both CandidateScore and PayloadNorm baselines
feature_compute_time_ms_q90 recorded
uses_dataset_name = 0
uses_outcome_at_commit = 0
```

P4 strong observability pass：

```text
AUC_control_positive >= 0.78
AUC_bad_event >= 0.80
ECE_bad <= 0.05
Top273_safe_good >= 205
Top273_bad_event <= 13
Top273_null_event <= 40
Top273_beats_adamwparallel >= 0.60
Top273_beats_bestlr >= 0.60
leave_dataset_auc_drop <= 0.07
leave_stratum_auc_drop <= 0.10
feature_compute_time_ms_q90 within online budget
```

P4 fail：

```text
primary oracle pass but control/value observability weak；
no legal feature/pair can separate value/bad/null enough。
```

### 可视化

```text
p4_feature_auc_bar_value_bad_null.svg
p4_candidate_score_vs_payloadnorm_phase_diagram.svg
p4_2d_feature_frontier_top273.svg
p4_value_bad_null_feature_pareto.svg
p4_feature_calibration_curves.svg
p4_leaveout_auc_drop_heatmap.svg
p4_feature_cost_vs_signal_pareto.svg
```

---

## P5：cross-fitted minimal action-value controller

### 目标

在 P2/P3/P4 完成后，构建 calibration-frozen、dataset-agnostic、minimal monotone controller。P5 是 decision gate，不是 feature fishing。

### 假设

H2/H6：如果 legal observability 足够，minimal $V/B/N/S/C$ controller 可以在 heldout 和 leave-out 中过线。

### Controller candidates

#### C0：negative controls

```text
C0a-StableAcceptOnly
C0b-CandidateScoreOnly
C0c-PayloadNormOnly
C0d-RandomSupportMatched
```

#### C1：two-signal disentangled controller

```text
Value head: CandidateScore / linearized descent
Bad head: PayloadNorm / tail risk / EB bad UCB
Null head: payload delta norm / predicted margin gain
Support head: family-horizon support LCB
Cost head: online cost estimate
```

Accept：

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

#### C2：monotone score with hard safety gates

$$
S(e)
=
a_1LCB(V(e))
-a_2UCB(B(e))
-a_3UCB(N(e))
+a_4LCB(S(e))
-a_5C(e),
$$

with：

```text
a_i >= 0
feature groups <= 5
thresholds frozen on calibration split
```

Accept：

$$
Accept(e)=1
\iff
S(e)\ge\tau
\land
UCB(B(e))\le\tau_b
\land
UCB(N(e))\le\tau_n.
$$

#### C3：two-stage exact confirm

```text
Stage 1: cheap legal prefilter。
Stage 2: exact commit-time safety confirm for borderline rows only。
```

Stage 2 must not use future outcomes or validation/test.

### Splits

```text
Seed folds:
  rotate calibration and heldout seeds if available。

Leave-dataset-out:
  train/calibrate on two datasets, evaluate third。

Leave-stratum-out:
  hold out signal stratum。

Leave-family-out diagnostic:
  hold out high-volume action family。
```

### 必须记录

```text
controller_id
feature_set
feature_group_count
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
beats_adamwparallel_cal
beats_bestlr_cal
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
value_mean_heldout
value_lcb_heldout
beats_adamwparallel_heldout
beats_bestlr_heldout
precision_lcb
bad_event_ucb
null_event_ucb
accepted_family_count
accepted_signal_strata_count
max_family_share
max_stratum_share
oracle_overlap_primary
oracle_overlap_control
stableaccept_overlap
feature_compute_time_ms_q90
```

### Decision pass

Primary safety gates：

$$
Precision_{heldout}\ge0.75
$$

$$
Coverage_{heldout}\in[0.03,0.15]
$$

$$
BadEventRate_{heldout}\le0.05
$$

$$
NullRate_{heldout}\le0.15
$$

$$
Precision_{LCB}\ge0.75
$$

$$
BadEventRate_{UCB}\le0.05
$$

Control-value gates：

$$
BeatRate_{Real\ vs\ AdamWParallel}\ge0.50
$$

$$
BeatRate_{Real\ vs\ bestLR}\ge0.50
$$

$$
ValueLCB_{heldout}>0
$$

Support balance：

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

Cross-fit stability：

```text
coverage_heldout > 0 in every main fold
bad_event gate pass in at least 2/3 seed folds
no dataset-specific branch
leave-dataset catastrophic fail = 0
```

### 可视化

```text
p5_controller_precision_bad_coverage_frontier.svg
p5_control_value_frontier.svg
p5_calibration_to_heldout_drift.svg
p5_crossfit_fold_matrix.svg
p5_controller_ablation_waterfall.svg
p5_support_balance_sunburst.svg
p5_oracle_overlap_primary_vs_control.svg
```

---

## P6：online runtime / offline materializer runtime decoupling

### 目标

把 v9.3.2 的 `RT2-ActionReplayEventSchedulerMeasuredPartial` 拆开，分别测 online official runtime 和 offline materializer runtime。P6 必须回答：step_ratio = 5.697759 是 online 必需成本，还是 materializer/replay/audit 混入成本？

### 假设

H4：online path 可以比 v9.3.2 measured partial path 快很多；offline materializer 的成本不能计入 online system gate。

### Runtime candidates

#### RT0：v9.3.2 measured partial reference

```text
runtime_mode = mixed_action_replay_materializer
step_ratio_q90 = 5.697758752709664
controller_launches_per_active_step_q90 = 3.0
```

#### RT1：online no-control scheduler

```text
runtime_mode = online_sequential_official_runtime
matched_controls_in_timed_online_path = 0
secondary_horizon_replay_in_timed_online_path = 0
heavy_csv_hash_audit_in_timed_path = 0
```

#### RT2：online one-kernel score/accept

```text
feature compute + score + accept in one fused kernel
payload apply separate if needed
```

#### RT3：online two-kernel score/apply

```text
kernel 1: feature / value-risk-null score / accept
kernel 2: payload materialize/apply
```

#### RT4：persistent workspace

```text
no per-active-step allocation
preallocated feature/action buffers
```

#### RT5：offline outcome materializer batch-major

```text
runtime_mode = offline_outcome_materializer_runtime
branches/horizons grouped
throughput measured, not online step_ratio gate
```

### 必须记录

Online：

```text
runtime_candidate_id
runtime_mode
step_count
active_step_count
zero_candidate_step_count
zero_candidate_controller_kernel_count
zero_candidate_controller_sync_count
controller_launches_per_active_step_q90
controller_syncs_per_active_step_q90
allocation_count_per_active_step
feature_compute_time_ms_q90
score_accept_time_ms_q90
payload_apply_time_ms_q90
audit_outside_timed_ms
base_train_step_time_ms_q90
controller_extra_time_ms_q90
total_step_time_ms_q90
mlp_step_time_ms_q90
step_ratio_q90
memory_ratio
accept_disagreement_count
payload_apply_error_max
```

Offline：

```text
offline_runtime_candidate_id
branch_horizon_rows
branch_horizon_rows_per_sec
actions_per_sec
materializer_wallclock_sec
gpu_utilization
peak_memory_mb
branch_completion_rate
horizon_completion_rate
```

### 判断标准

Online runtime pass：

```text
runtime_mode = online_sequential_official_runtime
zero_candidate_controller_kernel_count = 0
zero_candidate_controller_sync_count = 0
controller_launches_per_active_step_q90 <= 2
controller_syncs_per_active_step_q90 <= 1
allocation_count_per_active_step <= 0.05
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
accept_disagreement_count = 0
payload_apply_error_max <= tolerance
```

Offline materializer pass：

```text
all official branch-horizon rows complete
materializer throughput recorded
no fake/proxy rows
runtime not promoted to online system pass
```

### 可视化

```text
p6_runtime_mode_separation.svg
p6_online_runtime_waterfall.svg
p6_offline_materializer_throughput.svg
p6_step_ratio_mixed_vs_online.svg
p6_active_step_launch_hist.svg
p6_zero_candidate_semantics_audit.svg
```

---

## P7：system controller integration

### 目标

只有 P5 decision pass 和 P6 online runtime pass 后，组合成 official system controller。P7 不允许用 P2/P3/P4/P6 的 partial success 单独转正。

### 必须记录

```text
system_candidate_id
controller_id
runtime_candidate_id
feature_set
thresholds
candidate_count
action_count
event_count
accepted_count
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
value_mean_heldout
value_lcb_heldout
beats_adamwparallel_heldout
beats_bestlr_heldout
precision_lcb
bad_event_ucb
accepted_family_count
accepted_signal_strata_count
max_family_share
max_stratum_share
step_ratio_q90
memory_ratio
controller_launches_per_active_step_q90
zero_candidate_controller_kernel_count
payload_binding_pass
action_lifecycle_pass
secondary_outcome_ready
control_oracle_pass
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

P7 pass：

```text
official_eligible = 1
system_legal_controller_pass = 1
action_lifecycle_pass = 1
secondary_outcome_ready = 1
control_oracle_pass = 1
decision_gate_pass = 1
runtime_gate_pass = 1
payload_binding_pass = 1
materialized_system_path = 1
diagnostic_derived_from_measured_components = 0
dataset_name_used = 0
projection_used = 0
source_measured_gap_used = 0
formula_proxy_used = 0
```

### 可视化

```text
p7_system_gate_dashboard.svg
p7_quality_cost_frontier.svg
p7_controller_runtime_pareto.svg
p7_failure_reason_matrix.svg
```

---

## P8：parallel diagnostic paired replay scout with isolation

### 目标

加快实验，但不污染 controller selection。P8 可与 P4/P5 并行准备，但结果不能影响 P5 阈值或 P7 route。

### 设置

```text
controllers = top2 diagnostic controllers + StableAccept negative control + NoFunctional
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
horizons = 20,80,240
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random, shuffled controls
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
diagnostic_used_for_controller = 0
```

### 判断标准

P8 promising：

```text
RealFunctional beats AdamWParallel in >= 50% macro slices
RealFunctional beats bestLR in >= 50% macro slices
task_safe holds
shuffled controls do not match RealFunctional
```

P8 cannot produce official success unless P7 already pass.

### 可视化

```text
p8_diagnostic_macro_beat_rate.svg
p8_branch_delta_pareto.svg
p8_shuffle_control_matrix.svg
p8_task_safety_by_slice.svg
p8_diagnostic_isolation_audit.svg
```

---

## P9：leave-dataset-out / leave-stratum-out official

### 目标

只有 P7 system pass 后 official 打开。证明 controller 不是 pooled calibration artifact，也不是 dataset-specific tuning。

### 设置

Leave-dataset-out：

```text
train/calibrate on MNIST + Fashion, evaluate KMNIST
train/calibrate on MNIST + KMNIST, evaluate Fashion
train/calibrate on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
train/calibrate on all but one signal stratum
evaluate held-out stratum
```

### 必须记录

```text
split_type
heldout_entity
controller_id
runtime_candidate_id
precision
coverage
bad_event_rate
null_rate
value_mean
value_lcb
beats_adamwparallel
beats_bestlr
task_safe
step_ratio_q90
memory_ratio
dataset_name_used
support_balance_pass
```

### 判断标准

LDO pass：

```text
at least 2/3 held-out datasets pass decision gates
no held-out dataset has bad_event_rate > 0.10
coverage > 0 for every held-out dataset
dataset_name_used = 0
```

LSO pass：

```text
>=70% held-out strata task-safe
macro bad_event_rate <= 0.05
macro precision >= 0.75
macro value_lcb > 0
```

### 可视化

```text
p9_leave_dataset_out_matrix.svg
p9_leave_stratum_out_matrix.svg
p9_dataset_diagnostic_no_tuning_audit.svg
p9_leaveout_failure_modes.svg
```

---

## P10：official paired replay

### 目标

验证 system-legal controller 选择的 RealFunctional 是否在 strong controls 下有局部因果优势。

### 前置条件

```text
P7 system pass = 1
P9 LDO/LSO pass = 1
secondary_outcome_ready = 1
online_runtime_pass = 1
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
  ShuffledCandidateScore
  ShuffledPayloadNorm
  ShuffledOGPValueScore
  ShuffledOGPRiskScore
  ShuffledOGPSupportScore
  ShuffledRuntimePath
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
BeatRate_{macro,Real\ vs\ AdamWParallel}\ge0.60
$$

$$
BeatRate_{macro,Real\ vs\ bestLR}\ge0.60
$$

Task safety：

$$
Acc_{each\ slice,Real}\ge Acc_{AdamW}-0.005
$$

Shuffle controls：

```text
all shuffled controls fail to reproduce RealFunctional macro beat pattern
```

### 可视化

```text
p10_official_paired_replay_pareto.svg
p10_macro_beat_rate.svg
p10_signal_stratum_win_matrix.svg
p10_shuffle_control_matrix.svg
p10_system_gate_distribution.svg
```

---

## P11：short-run / full-run / robustness / sample-efficiency / continual

### 目标

只有 P10 pass 后打开。验证 local causal advantage 能否进入连续训练，并排除 LR grid、QuadraticFeatureMLP、shuffled controller 等解释。

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
  shuffled controllers / shuffled payloads / shuffled runtime path
```

### 必须记录

Task：

```text
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
```

Sample efficiency：

```text
ValLossAUC_step
ValLossAUC_time
steps_to_target_acc
time_to_target_acc
data_seen_to_target
```

Continual / anti-forgetting：

```text
old_task_retained_accuracy
forgetting
backward_transfer
forward_transfer
old_task_CEp99_drift
old_task_margin_drift
```

### 判断标准

Full functional pass：

$$
Acc_{functional}\ge Acc_{AdamW}-0.005
$$

and at least one：

$$
Acc_{functional}>Acc_{AdamWParallel}
$$

$$
ECE_{functional}<ECE_{AdamW}
$$

$$
NLL_{functional}<NLL_{AdamW}
$$

$$
Curvature_{functional}\le0.90Curvature_{AdamW}
$$

Sample efficiency pass：

```text
ValLossAUC_step improves over AdamW by meaningful margin
or time_to_target improves without breaking step_ratio gate
```

Continual pass：

```text
forgetting <= AdamW forgetting
and retained accuracy improves or remains task-safe
```

Strong baseline pass：

```text
gain not explained by LR grid
gain not explained by QuadraticFeatureMLP
gain not explained by shuffled controller/kernel/payload
```

---

# 7. 并行执行计划

v9.3.3 必须并行推进，避免每轮只发现一个 missing blocker。

## Batch A：contract and payload

```text
A1: P0 v9.3.2 boundary reanalysis
A2: P1 durable payload package
A3: P6 online runtime no-control smoke
A4: P2 materializer branch/horizon dry-run on small shard
```

Exit：

```text
P1 disk replay pass
P6 online runtime mode separated
P2 branch/horizon schema validated
```

## Batch B：full outcomes and runtime split

```text
B1: P2 full matched-control outcome materializer
B2: P6 online RT1/RT2/RT3 measured
B3: P6 offline RT5 materializer throughput
B4: P4 feature extraction cost profiler
```

Exit：

```text
secondary_outcome_ready = 1 or diagnostic-only declared
online runtime pass/fail measured
```

## Batch C：oracle / observability / controller

```text
C1: P3 control-positive oracle
C2: P4 action-value observability
C3: P5 cross-fitted controller
C4: P8 diagnostic paired replay scout isolated
```

Exit：

```text
control oracle route known
feature observability route known
controller candidate selected or feature/primitive fail declared
```

## Batch D：official system

```text
D1: P7 system integration
D2: P9 LDO/LSO official
D3: P10 official paired replay
```

Only after P7 pass.

## Batch E：longer runs

```text
E1: P11 short-run
E2: P11 full-run
E3: sample efficiency / continual / robustness
```

Only after P10 pass.

---

# 8. Required artifacts

```text
run_manifest.json
contract_audit_v9330.csv
provenance_audit_v9330.csv
artifact_hashes.csv

p0_v9320_boundary_reanalysis.csv

p1_durable_action_payload_package.csv
action_payload_manifest_v9330.json
action_payload_shards_v9330/
action_payload_disk_replay_trace_v9330.csv

p2_full_matched_control_outcome_materializer.csv
full_control_outcome_table_v9330.csv
matched_control_outcome_trace_v9330.csv
branch_horizon_completion_trace_v9330.csv
offline_materializer_runtime_trace_v9330.csv

p3_control_positive_oracle.csv
control_oracle_frontier_trace_v9330.csv
oracle_support_balance_trace_v9330.csv

p4_action_value_observability.csv
action_value_feature_trace_v9330.csv
feature_pair_frontier_trace_v9330.csv
feature_cost_trace_v9330.csv
minimality_audit_trace_v9330.csv

p5_crossfitted_action_value_controller.csv
controller_frontier_trace_v9330.csv
controller_ablation_trace_v9330.csv
crossfit_leaveout_trace_v9330.csv

p6_runtime_decoupling.csv
online_runtime_trace_v9330.csv
offline_runtime_trace_v9330.csv
runtime_component_trace_v9330.csv
runtime_empty_event_semantics_trace_v9330.csv

p7_system_controller_v9330.csv
system_controller_trace_v9330.csv

p8_diagnostic_paired_replay_scout.csv
diagnostic_isolation_audit_v9330.csv

p9_leave_dataset_stratum_out.csv
leaveout_trace_v9330.csv

p10_official_paired_replay.csv
official_paired_replay_trace_v9330.csv

p11_short_full_sampleeff_continual_robustness.csv
short_full_trace_v9330.csv

route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

---

# 9. Failure taxonomy

```text
F1_contract_violation
F2_dataset_tuning_detected
F3_teacher_or_loss_modification_detected
F4_fake_or_proxy_violation
F5_payload_tensor_file_missing
F6_disk_replay_fail
F7_action_apply_error_regression
F8_secondary_control_outcome_incomplete
F9_branch_missing
F10_horizon_missing
F11_secondary_delta_missing
F12_matched_control_insufficient
F13_control_oracle_absent
F14_primary_oracle_control_oracle_conflict
F15_action_value_observability_gap
F16_bad_tail_unobservable
F17_null_risk_unobservable
F18_value_signal_unobservable
F19_feature_cost_unmeasured
F20_feature_pile_detected
F21_no_dataset_agnostic_controller
F22_controller_precision_fail
F23_controller_bad_event_fail
F24_controller_null_rate_fail
F25_controller_coverage_fail
F26_controller_value_fail
F27_support_balance_fail
F28_online_offline_runtime_conflated
F29_online_runtime_step_ratio_fail
F30_active_step_launch_count_fail
F31_zero_event_semantics_fail
F32_memory_ratio_fail
F33_offline_materializer_incomplete
F34_diagnostic_promoted_to_official
F35_leave_dataset_out_fail
F36_leave_stratum_out_fail
F37_paired_replay_control_equivalent
F38_shuffle_control_pass
F39_short_run_task_drop
F40_full_run_no_macro_or_hard_stratum_gain
F41_strong_baseline_explains_gain
F42_sample_efficiency_fail
F43_continual_forgetting_fail
F44_external_not_ready
```

---

# 10. Route decision

```text
R1-BoundaryReanalyzed:
  v9.3.2 metrics reproduced and gate gaps computed.

R2-DurableActionPayloadPass:
  action payload tensors are written and disk replay is bit/numerically stable.

R3-FullControlOutcomeReady:
  full matched controls and horizons are materialized.

R4-ControlOutcomeMaterializationFail:
  controls/horizons remain incomplete.

R5-ControlOraclePass:
  current action population has control-positive oracle frontier.

R6-ControlOracleFail:
  even oracle cannot find useful/control-positive safe frontier.

R7-ActionValueObservabilityPass:
  legal action-value / bad-risk / null-risk observables are predictive and affordable.

R8-ActionValueObservabilityFail:
  control oracle exists but legal pre-commit observability cannot identify it.

R9-ActionValueControllerPass:
  cross-fitted minimal controller passes decision and value gates.

R10-ActionValueControllerFail:
  features exist but no dataset-agnostic controller passes.

R11-OnlineRuntimePass:
  online event-driven runtime reaches step_ratio_q90 <= 1.50.

R12-OnlineRuntimeFail:
  online runtime remains above envelope.

R13-SystemLegalControllerPass:
  decision + value + runtime + contracts pass.

R14-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R15-LeaveStratumOutPass:
  controller generalizes across held-out strata.

R16-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R17-PairedReplayFail:
  system legal but functional update is control-equivalent.

R18-ShortFullFunctionalPass:
  short/full task-safe mechanism gain.

R19-SampleEfficiencyContinualPass:
  sample efficiency or anti-forgetting evidence passes.

R20-ExternalReady:
  strict PureKAN functional route passes external-ready gates.

R30-ActionPrimitiveControlFrontierAbsent:
  primary oracle exists but control-positive oracle absent; redesign action primitive.

R31-ObservablePrimitiveInsufficient:
  control oracle exists but no legal observable can identify it.
```

`route_decision.json` must record：

```text
route
base_candidate
candidate_count
action_count
event_count

action_lifecycle_pass
durable_payload_pass
payload_tensor_file_written
disk_replay_success_count
action_apply_error_linf_max
action_apply_cosine_logged_applied_min

secondary_outcome_ready
official_candidate_coverage
matched_control_count_per_event_min
branch_missing_count
horizon_missing_count
missing_secondary_delta_count

primary_oracle_pass
control_oracle_pass
oracle_coverage
oracle_precision
oracle_bad_event
oracle_null_rate
oracle_beats_adamwparallel_rate
oracle_beats_bestlr_rate

action_value_observability_pass
best_feature_group
best_feature_pair
best_auc_control_positive
best_auc_bad_event
best_top273_safe_good
best_top273_bad_event
best_top273_null_event
feature_cost_pass

ogp_controller_pass
controller_id
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
value_lcb_heldout
beats_adamwparallel_heldout
beats_bestlr_heldout
precision_lcb
bad_event_ucb
support_balance_pass

online_runtime_pass
runtime_candidate_id
runtime_mode
zero_candidate_controller_kernel_count
controller_launches_per_active_step_q90
step_ratio_q90
memory_ratio
online_offline_runtime_conflated

system_legal_controller_pass
official_eligible
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
sample_efficiency_pass
continual_pass
robustness_pass
strong_baseline_pass
external_ready

uses_dataset_name_for_controller
uses_loss_backward
uses_teacher
uses_loss_modification
fake_data_used
proxy_row_used
cpu_offload_used
diagnostic_promoted_to_official

primary_blocker
next_required_implementation
success_v9330_strict_purekan_functional
success_v9330_full_functional
success_v9330_external_ready
```

---

# 11. 停止条件

## Minimum diagnostic success

```text
P0 boundary reproduced；
P1 durable payload measured；
P2 full or official-minimum matched-control outcomes measured；
P3 control oracle measured；
P4 observability measured；
P5 controller measured；
P6 online/offline runtime separated and measured；
no fake/proxy/offload/teacher/loss violation。
```

## Stop: materializer incomplete

If：

```text
branch_missing_count > 0
or horizon_missing_count > 0
or missing_secondary_delta_count > 0
or matched_control_count_per_event_min < 4
```

Then：

```text
route = R4-ControlOutcomeMaterializationFail
do not run official controller / paired replay
next = full control materializer engineering
```

## Stop: control oracle absent

If：

```text
primary_oracle_pass = 1
but control_oracle_pass = 0
```

Then：

```text
route = R30-ActionPrimitiveControlFrontierAbsent
stop controller feature search
redesign action primitive / candidate generator / functional update source
```

## Stop: observability absent

If：

```text
control_oracle_pass = 1
but action_value_observability_pass = 0
```

Then：

```text
route = R31-ObservablePrimitiveInsufficient
do not pile more opaque features
redesign legal observable primitive
```

## Stop: controller fail

If：

```text
observability_pass = 1
but controller_pass = 0
```

Then：

```text
route = R10-ActionValueControllerFail
fix calibration/support/cross-fitting, not dataset-specific thresholds
```

## Stop: runtime fail

If：

```text
controller_pass = 1
but online_runtime_pass = 0
```

Then：

```text
route = R12-OnlineRuntimeFail
stop controller tuning
fix online runtime components
```

## Open official causal validation

Only if：

```text
system_legal_controller_pass = 1
and leave_dataset_out_pass = 1
and leave_stratum_out_pass = 1
```

Then open：

```text
P10 official paired replay
```

## Full success

Only if：

```text
paired_replay_pass = 1
and short/full functional pass = 1
and strong baseline pass = 1
and robustness/sample-efficiency/continual evidence recorded
```

Then declare：

```text
success_v9330_full_functional = 1
```

---

# 12. 最终建议

v9.3.3 的核心不是“再修 feature”，而是：

$$
\boxed{
\text{先让 RealFunctional 与 controls 在 branch/horizon 上完整可比，再判断是否存在 legal action-value controller。}
}
$$

最关键的实施顺序是：

```text
1. 把 v9.3.2 的 action apply pass 固化成 durable payload package；
2. 全量或 official-minimum materialize matched controls and horizons；
3. 用 controls 定义真正的 V_ctrl；
4. 先测 control-positive oracle；
5. 如果 control oracle 不存在，重构 action primitive；
6. 如果 control oracle 存在，再测 legal observability；
7. 用 CandidateScore/PayloadNorm 等信号做 disentangled V/B/N/S/C，而不是 opaque feature pile；
8. online runtime 与 offline materializer runtime 分开测；
9. system pass 前不打开 official paired replay；
10. 数据集只用于诊断和 leave-out，不用于 tuning。
```

本轮最希望看到的结果不是“直接 external-ready”，而是一个明确分叉：

```text
Case A:
  full controls ready + control oracle pass + observability pass + runtime pass
  -> 进入 system controller / LDO / official paired replay。

Case B:
  full controls ready + control oracle fail
  -> 说明 current action primitive safe but not useful，重构 action generator。

Case C:
  control oracle pass + observability fail
  -> 说明 action 有价值，但 legal observable primitive 不足，重构 observable。

Case D:
  controller pass + runtime fail
  -> 说明科学对象可行，系统工程仍需 online runtime closure。

Case E:
  runtime pass + controller fail
  -> 说明系统够快，但 action-value decision 不可部署。
```

这样 v9.3.3 才不会变成“v9.3.2 的补丁版”，而是第一次真正回答：

$$
\boxed{
\text{DG-KAN 的 functional action 是否有 control-positive value，且能否被 legal online controller 以 MLP-comparable runtime 选择？}
}
$$
