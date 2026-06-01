# DG-KAN v9.3.4 Control Outcome Materializer Scale-up / Action-Value Identifiability / Online Runtime Remeasure 完整实验计划

> 本计划基于 v9.3.3 `Full Control Outcome / Action-Value Disentanglement 与 Runtime Decoupling` 的真实执行结果制定。  
> v9.3.4 不再把重点放在 StableAccept repair，也不把 `PayloadNorm` / 单特征 AUC 当成下一轮主线。  
> 本轮核心目标是把 action-value 这个对象真正 materialize 出来，并在足够覆盖、足够吞吐、足够分层完整的 control outcome table 上判断：functional action 是否有可识别、可部署、可泛化的正价值。

---

# 0. 执行摘要

v9.3.3 的真实进展很清楚：

```text
1. durable action payload package 已经闭合；
2. disk replay hash / error / cosine 数值闭合；
3. control outcome materializer 进入真实 branch/horizon dry-run；
4. online runtime 与 offline materializer 已经分账；
5. 但 full control outcome materialization 仍没有完成，controller / oracle / paired replay 全部不能 official。
```

关键数据：

```text
action_count = 2876
expected_branch_horizon_rows = 51768
actual_branch_horizon_rows = 48
official_candidate_coverage = 0.0027816411682892906
matched_control_count_per_event_min = 5
branch_missing_count = 17208
horizon_missing_count = 51720
missing_secondary_delta_count = 192
secondary_outcome_ready = 0
control_oracle_pass = 0
action_value_observability_pass = 0
online_runtime_pass = 0
system_legal_controller_pass = 0
```

独立复算：

$$
51768 = 2876 \times 6 \times 3.
$$

这说明 v9.3.3 的 official branch-horizon design 实际上是：

```text
2876 actions
6 control branches
3 horizons
```

而实际完成：

$$
48 = 8 \times 6 \times 1.
$$

也就是说本轮只覆盖了 8 个 actions、6 个 branches、1 个 horizon。candidate/action 覆盖率是：

$$
\frac{8}{2876}=0.0027816411682892906.
$$

branch-horizon row 覆盖率是：

$$
\frac{48}{51768}=0.0009272137.
$$

因此 v9.3.3 的核心不是 controller 失败，而是 action-value label universe 还没有形成。现在没有足够数据去回答：

```text
RealFunctional 是否比 AdamWParallel / bestLR / NoOp / Random 更有价值；
哪些 action 是 control-positive；
哪些 feature 能在 commit 前识别 control-positive；
functional controller 是否有 causal advantage；
online runtime 是否在不混入 offline materializer 的情况下过线。
```

v9.3.4 的一句话目标是：

$$
\boxed{
\text{把 control outcome materializer 从 8-action dry-run 扩展为可审计、可并行、可覆盖 official universe 的 action-value label engine。}
}
$$

本轮不能把重点放在“再试一个 feature”或“再调一个 controller”。v9.3.4 的最低有效推进是：

```text
1. full branch/horizon outcome table 或 official-minimum stratified outcome table materialized；
2. missing branch / horizon / secondary delta 明确降到 gate 内；
3. control-positive oracle 可评估，而不是 dry-run diagnostic；
4. action-value features 在足够样本上评估，而不是 48 rows 上的 AUC=0.5；
5. online runtime 独立重新测量，不能继续写 online_runtime_not_remeasured_after_decoupling；
6. 若 materializer 仍无法 scale，要定位瓶颈，不继续 controller。
```

---

# 1. 对 v9.3.3 的独立判断

## 1.1 这次有进展，但不是 functional success

v9.3.3 真实推进的是 action lifecycle / reproducibility：

```text
payload_shard_count = 45
payload_tensor_file_written = 1
payload_hash_match_rate = 1.0
disk_replay_success_count = 2876 / 2876
disk_replay_error_linf_max = 0.0
disk_replay_error_relative_max = 0.0
disk_replay_cosine_logged_applied_min = 0.9999999999999968
durable_payload_pass = 1
action_lifecycle_pass = 1
```

这非常重要。v9.3.2 的 action apply closure 还是同轮 train-stream 重建与内存 replay；v9.3.3 把 payload 写成 durable sharded tensor package，并从磁盘 replay 到完全数值一致。也就是说，action 不再只是 ephemeral log artifact，而是可复现的物理对象。

但这不是 functional success，因为 functional success 的核心不是“action 能否被 replay”，而是：

$$
\boxed{
\text{action replay 后是否相对 matched controls 产生稳定、正向、低风险收益。}
}
$$

这个问题在 v9.3.3 仍然没有被回答。

## 1.2 当前最大 blocker 已经从 payload 转为 outcome materialization

v9.3.3 的 P2 只有：

```text
branch_horizon_row_count_actual = 48
branch_horizon_row_count_expected = 51768
```

这不是“覆盖差一点”。full row completion 是：

$$
RowCompletion = \frac{48}{51768}=0.0009272137.
$$

如果用 30% official-minimum stratified panel 作为下一轮最低可训练/可评估样本，那么至少需要：

$$
N_{rows,30\%}=0.30 \times 2876 \times 6 \times 3 = 15530.4.
$$

取整后约为：

```text
minimum_official_panel_rows ≈ 15534
```

当前只有 48 rows，距离 30% panel 仍差约：

$$
\frac{15534}{48}=323.625.
$$

距离 full completion 差约：

$$
\frac{51768}{48}=1078.5.
$$

所以 v9.3.4 不能继续做 8-action dry-run 的扩展版本。它必须变成一个 throughput / sharding / coverage / retry / audit 完整的 materializer scale-up 实验。

## 1.3 v9.3.3 的 `AUC=0.5` 不能证明 feature 没用

P4 写着：

```text
best_feature_group = F1-PayloadNorm
best_auc_control_positive = 0.5
action_value_observability_pass = 0
```

这个结果只能说明在当前 dry-run labels 下没有形成可用 signal，不能说明 PayloadNorm 或其他 action-conditioned features 本质无用。原因是 P2 只覆盖 8 个 actions、48 个 rows、1 个 horizon。这样的 label universe 太小，控制分支也不完整，control-positive oracle 本身都没有 ready。

正确解释是：

$$
\boxed{
\text{action-value observability 仍不可评估，而不是 observability 已被证伪。}
}
$$

因此 v9.3.4 不应基于 v9.3.3 的 `AUC=0.5` 去抛弃 payload geometry，也不应继续在 48 rows 上筛 feature。先扩展 control outcomes，再谈 feature。

## 1.4 runtime decoupling 有进步，但 online runtime 仍没有答案

v9.3.3 做对的一点是没有把 offline materializer 计入 online runtime：

```text
online_offline_runtime_conflated = 0
offline_materializer_runtime_recorded = 1
offline_branch_horizon_rows_per_sec = 0.45211140000559824
online_runtime_pass = 0
primary_blocker = online_runtime_not_remeasured_after_decoupling
```

这说明项目避免了一个重要错误：用昂贵的 label materializer 成本污染 online controller runtime。

但 online runtime 仍没有重新测量，所以不能说 system envelope 有进展。v9.3.4 必须单独做：

```text
online controller-only runtime measurement
no control materializer in timed path
no disk replay in timed path unless official online path requires it
no diagnostic-derived runtime pass
```

## 1.5 当前是否在正确道路上

高层方向仍然正确。v9.3.0 到 v9.3.3 的路线已经从 StableAccept patching 逐步推进到 action-value materialization：

```text
v9.3.0: 证明 candidate/action population 有 primary oracle frontier，但 legal features 找不到。
v9.3.1: 证明 action apply / secondary-control / runtime 都还缺。
v9.3.2: action apply error 全量测量闭合。
v9.3.3: durable payload disk replay 闭合。
```

这不是原地打转。真正的问题是：v9.3.3 只完成了 control outcome materializer 的 proof-of-life，没有完成 scale-up。

所以 v9.3.4 的方向必须是 implementation-first：

```text
full control outcome materializer scale-up
+ action-value label definition
+ control-positive oracle
+ legal observability
+ online runtime remeasure
```

不是：

```text
more PayloadNorm threshold
more OGP feature patch
more dry-run oracle
more diagnostic paired replay
```

---

# 2. 当前离终极目标有多远

DG-KAN 的终极目标仍然是：

$$
\text{Clean FullEdge PureKAN}
+
\text{Graph-Free Manual Training}
+
\text{Kernel-Native Efficiency}
+
\text{Task-Safe Functional Update}
+
\text{Outcome-Grounded Controller}
+
\text{External Fair Advantage}.
$$

按 gate 拆开看：

```text
已基本闭合：
  LQ-t2-h256 base anchor
  manual forward / backward / AdamW update contract
  primary same-run labels
  candidate/action identity freeze
  action apply error
  durable action payload package
  disk replay numerical closure

部分闭合：
  control outcome materializer proof-of-life
  online/offline runtime accounting separation

未闭合：
  full matched-control outcomes
  complete horizons 20/80/240
  secondary deltas
  control-positive oracle
  action-value feature observability
  cross-fitted value/risk/null controller
  online runtime remeasurement
  system-legal controller
  leave-dataset-out / leave-stratum-out
  official paired replay
  short/full run
  robustness / sample efficiency / continual anti-forgetting
```

所以现在离最终目标还比较远，但路线不是错了。更准确的阶段判断是：

$$
\boxed{
\text{项目刚完成 action 物理可复现性，正进入 action-value label universe 建设阶段。}
}
$$

下一阶段如果成功，会第一次让下面这个对象真实存在：

$$
V_{ctrl}(e,h)=V_{RealFunctional}(e,h)-\max_{b\in Controls}V_b(e,h).
$$

没有这个对象，controller、paired replay、full functional 都只是空谈。

---

# 3. v9.3.4 总体目标

v9.3.4 的总体目标是：

$$
\boxed{
\text{建立可扩展的 full control outcome materializer，并在足够覆盖的 label universe 上判断 action-value 是否可识别。}
}
$$

本轮强目标：

```text
full_control_outcome_ready = 1
control_oracle_ready = 1
action_value_observability_measured = 1
online_runtime_remeasured = 1
```

若再进一步，争取：

```text
action_value_observability_pass = 1
crossfitted_minimal_controller_pass = 1
online_runtime_pass = 1
system_legal_controller_pass = 1
```

但 v9.3.4 的最低有效推进不是 controller pass，而是：

```text
1. full or official-minimum control outcome table materialized；
2. branch_missing_count / horizon_missing_count / missing_secondary_delta_count 降到 official gate 内；
3. materializer throughput 从 dry-run 级别提升到可完成 full universe；
4. control-positive oracle 在非 dry-run 数据上可评估；
5. online controller runtime 独立重新测量；
6. 如果 action-value 不存在，能明确区分是 candidate primitive 问题、control definition 问题、horizon 问题，还是 feature observability 问题。
```

---

# 4. v9.3.4 明确不做什么

本轮不做：

```text
1. 不继续修 StableAccept final controller；
2. 不在 48 rows / 8 actions dry-run 上选 official feature；
3. 不把 PayloadNorm AUC=0.5 写成 feature 不可用的最终证据；
4. 不把 control oracle dry-run 写成 useful oracle；
5. 不把 durable payload pass 写成 functional success；
6. 不把 offline materializer throughput 写成 online runtime pass；
7. 不按 dataset 调 threshold、control branch、horizon 或 controller；
8. 不用 teacher、distillation、aux loss、label smoothing、focal loss、margin loss；
9. 不用 validation/test outcome at commit time；
10. 不在 full_control_outcome_ready 之前打开 official paired replay；
11. 不把 diagnostic paired replay scout 结果用于 P7 controller threshold；
12. 不为了赶进度把 8-action dry-run 转正。
```

允许做：

```text
1. dataset / seed / family / horizon 级诊断；
2. 并行 materializer worker；
3. shard-level retry / checkpoint caching / branch batching；
4. progressive widening；
5. official-minimum stratified panel；
6. full candidate materialization；
7. calibration-fold 内部 feature/model selection；
8. heldout outcome 只用于 evaluation，不用于 controller threshold selection；
9. offline materializer runtime 单独报告；
10. online controller runtime 单独报告。
```

---

# 5. 核心假设

## H0：durable payload package 已不是 blocker

H0 认为 v9.3.3 已经把 action payload 从 ephemeral log artifact 推进为 durable, disk-replayable, hash-verified object。v9.3.4 不应再在 payload package 上绕圈，除非出现 regression。

H0 成立标准：

```text
payload_tensor_file_written = 1
payload_hash_match_rate = 1.0
disk_replay_success_count = action_count
disk_replay_error_linf_max <= 1e-8
disk_replay_cosine_logged_applied_min >= 0.99999999
action_lifecycle_pass = 1
```

H0 失败标准：

```text
durable payload replay 在 v9.3.4 source artifacts 上回退；
任何 shard hash mismatch；
任何 action replay error 超过 tolerance。
```

若 H0 失败，停止 control outcome scale-up，先修 payload package。

## H1：control outcome materializer 可以通过 sharding / caching / branch batching scale

H1 认为 v9.3.3 的 `0.452 rows/sec` 不是不可避免下界，而是 dry-run implementation 没有充分利用 batching、checkpoint cache、branch/horizon vectorization 和 parallel shards。

H1 成立标准：

```text
throughput_rows_per_sec_total >= 2.0 minimum
throughput_rows_per_sec_total >= 5.0 strong
materializer_gpu_util_mean >= 0.40 if GPU path used
checkpoint_reload_fraction <= 0.10
rows_failed_retry_unresolved = 0
branch_horizon_row_count_actual >= official_minimum_rows
```

H1 失败标准：

```text
即使 sharding/caching/batching 后 throughput_rows_per_sec_total < 1.0；
full or official-minimum coverage 无法在预算内完成；
dominant cost 不是工程开销，而是真实 branch training horizon compute。
```

若 H1 失败，本轮 route 应转为：

```text
R2-ControlOutcomeMaterializerThroughputFail
```

下一步不是 controller，而是 materializer algorithm redesign。

## H2：control-positive oracle 在 full outcomes 上可能存在，也可能不存在

H2 不是假设 oracle 一定存在，而是要求 v9.3.4 首次真正测量它。

定义 action-value：

$$
V_{Real}(e,h)=
-w_1\Delta CEp99(e,h)
+w_2\Delta MarginP10(e,h)
-w_3\Delta ECE(e,h)
-w_4\Delta NLL(e,h)
-w_5\Delta Curvature(e,h).
$$

对照分支集合：

$$
Controls=\{AdamWParallel, AdamWOnly, bestLR, NoOp, Random\}.
$$

控制优势：

$$
V_{ctrl}(e,h)=V_{Real}(e,h)-\max_{b\in Controls}V_b(e,h).
$$

robust action value：

$$
V_{robust}(e)=\min_{h\in\{20,80,240\}}V_{ctrl}(e,h).
$$

Control-positive label：

$$
ControlPositive(e)=1
\iff
V_{robust}(e)>0
\land BadEvent(e)=0
\land NullEvent(e)=0
\land TaskSafe(e)=1.
$$

H2 成立标准：

```text
control_positive_oracle_coverage >= 0.03
control_positive_oracle_precision >= 0.75
control_positive_bad_event_rate <= 0.05
control_positive_null_rate <= 0.15
support_balance_pass = 1
```

Strong oracle：

```text
coverage >= 0.03
precision >= 0.90
bad_event_rate <= 0.02
null_rate <= 0.10
beats_adamwparallel_rate >= 0.60
beats_bestlr_rate >= 0.60
```

H2 失败标准：

```text
在 full or official-minimum control outcomes 上，即使用 oracle labels，也找不到 coverage >= 0.03 的 control-positive frontier。
```

若 H2 失败，下一步应回到 candidate/action primitive，而不是 controller。

## H3：action-value signals 是可分解的，不应再寻找单一 magic score

v9.3.1 和 v9.3.2 已经显示：一些特征能找 safe，一些能压 bad，一些保 non-null，但单特征不能同时过线。v9.3.4 认为 value、bad-risk、null-risk、support、cost 必须被分开估计。

目标 controller 形式：

$$
Accept(e)=1
\iff
LCB(V_{ctrl}(e))>0
\land UCB(Bad(e))\le\tau_b
\land UCB(Null(e))\le\tau_n
\land LCB(Support(e))\ge\tau_s
\land Cost(e)\le C_{max}.
$$

H3 成立标准：

```text
至少一个 legal feature group 对 V_ctrl AUC 或 PR-AUC 有稳定 lift；
至少一个 legal feature group 对 bad risk 有稳定 lift；
至少一个 legal feature group 对 null risk 有稳定 lift；
combined minimal controller feature groups <= 5；
monotone signs consistent across folds。
```

H3 失败标准：

```text
full outcome labels ready 后，legal pre-commit features 仍无法区分 V_ctrl / bad / null / support；
所有 feature 在 LDO/LSO 上崩；
best heldout controller coverage = 0 或 bad_event > gate。
```

## H4：online runtime 必须重新测量，而不是继承 v9.3.2 或 v9.3.3 diagnostic

H4 认为 v9.3.3 的 `online_offline_runtime_conflated = 0` 是 accounting progress，但不是 runtime pass。

H4 成立标准：

```text
runtime_mode = online_sequential_official_runtime
control_outcome_materializer_in_timed_path = 0
disk_payload_replay_in_timed_path = explicitly recorded
zero_candidate_controller_kernel_count = 0
controller_launches_per_active_step_q90 <= 2
controller_syncs_per_active_step_q90 <= 1
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

H4 失败标准：

```text
online runtime 未重新测量；
或测量后 step_ratio_q90 > 1.50；
或 timing path 混入 offline materializer / audit / label replay。
```

## H5：dataset 只能作为诊断维度，不能作为 controller 分支

H5 认为 MNIST / Fashion-MNIST / KMNIST 的差异可以帮助定位 support holes、horizon mismatch、family instability，但不能用于调 threshold 或切换 route。

H5 成立标准：

```text
uses_dataset_name_for_controller = 0
dataset-specific threshold count = 0
dataset-specific branch count = 0
leave-dataset-out reported
per-dataset diagnostics reported but not used in commit rule
```

H5 失败标准：

```text
任何 controller threshold / branch / feature selection 使用 dataset_name 作为 commit-time input。
```

---

# 6. 数据合同

## 6.1 Action payload contract

继承 v9.3.3 durable payload package。每个 action 必须记录：

```text
action_id
candidate_id
event_id
payload_package_id
payload_shard_id
payload_tensor_offset
payload_tensor_shape
payload_tensor_dtype
payload_hash
payload_norm
payload_l2_norm
payload_linf_norm
payload_role_entropy
payload_nonzero_fraction
payload_tensor_file_written
payload_hash_match
disk_replay_success
disk_replay_error_linf
disk_replay_error_relative
disk_replay_cosine_logged_applied
```

Pass：

```text
payload_tensor_file_written = 1
payload_hash_missing_count = 0
payload_hash_match_rate = 1.0
disk_replay_success_count = action_count
disk_replay_error_linf_max <= 1e-8
disk_replay_cosine_logged_applied_min >= 0.99999999
```

## 6.2 Control outcome row contract

每一行 branch-horizon outcome 必须唯一对应：

```text
control_outcome_row_id = hash(action_id, branch_id, horizon, base_checkpoint_hash, replay_seed, outcome_schema_version)
```

必须记录：

```text
control_outcome_row_id
action_id
candidate_id
event_id
dataset
seed
step
batch_id
family_id
bucket_id
horizon
branch_id
branch_family
base_checkpoint_hash
payload_hash
payload_shard_id
replay_seed
outcome_schema_version
branch_start_state_hash
branch_end_state_hash
branch_apply_success
branch_apply_error_linf
branch_runtime_ms
branch_step_count
branch_wallclock_start
branch_wallclock_end
```

Outcome metrics：

```text
CE_mean_before
CE_mean_after
CE_mean_delta
CEp90_delta
CEp99_delta
margin_p10_before
margin_p10_after
margin_p10_delta
ECE_before
ECE_after
ECE_delta
NLL_before
NLL_after
NLL_delta
acc_before
acc_after
acc_delta
curvature_before
curvature_after
curvature_delta
local_lipschitz_before
local_lipschitz_after
local_lipschitz_delta
basis_usage_entropy_delta
functional_channel_entropy
```

Primary labels：

```text
task_safe_label
useful_label
bad_event_label
null_event_label
safe_good_label
control_positive_label
```

Control comparison fields：

```text
V_real
V_branch
V_ctrl_vs_adamwparallel
V_ctrl_vs_bestlr
V_ctrl_vs_noop
V_ctrl_vs_random
V_ctrl_max_control_gap
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
```

Pass：

```text
duplicate_control_outcome_row_id_count = 0
branch_apply_success_rate = 1.0
missing_primary_label_count = 0
missing_secondary_delta_count = 0
branch_missing_count = 0 for full pass
horizon_missing_count = 0 for full pass
label_exclusivity_violation_count = 0
```

## 6.3 Official-minimum coverage contract

v9.3.4 支持两种 outcome readiness。

### Full outcome ready

```text
branch_horizon_row_count_actual = 51768
branch_missing_count = 0
horizon_missing_count = 0
missing_secondary_delta_count = 0
full_control_outcome_ready = 1
```

### Official-minimum panel ready

若 full completion 因成本未完成，允许 official-minimum panel 进入 oracle/feature diagnostic，但不能直接进入 final external-ready claim。

最低要求：

```text
official_candidate_coverage >= 0.30
branch_horizon_row_coverage >= 0.30
all branches present for sampled actions
all horizons 20/80/240 present for sampled actions
matched_control_count_per_event_min >= 5
missing_secondary_delta_count = 0
per_dataset_min_action_count >= 100
per_horizon_min_action_count >= 250
per_branch_min_row_count >= 250
family_horizon_bucket_min_count >= 5 where bucket is active
```

若只达到 official-minimum panel：

```text
control_oracle can be reported as panel_oracle；
action_value_observability can be reported as panel_observability；
final system_legal_controller cannot pass unless all accepted rows and matched rejected rows have complete outcomes。
```

## 6.4 Materializer runtime contract

每个 worker / shard 必须记录：

```text
worker_id
shard_id
dataset
seed
action_id_start
action_id_end
branch_group
horizon_group
row_count_expected
row_count_completed
row_count_failed
row_count_retried
rows_per_sec
checkpoint_load_time_ms
payload_load_time_ms
branch_apply_time_ms
horizon_rollout_time_ms
metric_compute_time_ms
csv_write_time_ms
tensor_write_time_ms
sync_time_ms
idle_time_ms
gpu_util_mean
gpu_memory_peak_mb
cpu_memory_peak_mb
io_read_mb
io_write_mb
```

Pass：

```text
unresolved_failed_rows = 0
shard_manifest_complete = 1
rows_per_sec_total >= 2.0 minimum
rows_per_sec_total >= 5.0 strong
checkpoint_load_fraction <= 0.10 strong
io_time_fraction <= 0.20 strong
```

## 6.5 Online runtime contract

单独记录 online controller runtime，不允许混入 offline materializer。

```text
runtime_mode
control_outcome_materializer_in_timed_path
offline_audit_in_timed_path
disk_payload_replay_in_timed_path
step_id
candidate_count_in_step
active_step_flag
zero_candidate_step_flag
controller_kernel_launch_count
controller_sync_count
allocation_count
candidate_pack_time_ms
feature_compute_time_ms
score_accept_time_ms
payload_apply_time_ms
durable_payload_lookup_time_ms
audit_outside_timed_ms
base_train_step_time_ms
controller_extra_time_ms
total_step_time_ms
mlp_step_time_ms
step_ratio
peak_memory_mb
memory_ratio
```

Pass：

```text
control_outcome_materializer_in_timed_path = 0
offline_audit_in_timed_path = 0
zero_candidate_controller_kernel_count = 0
controller_launches_per_active_step_q90 <= 2
controller_syncs_per_active_step_q90 <= 1
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

---

# 7. 实验阶段

---

## P0：v9.3.3 boundary independent reanalysis and budget computation

### 目标

重新计算 v9.3.3 的真实进展与缺口，不依赖 route 名称。P0 要输出 materializer scale-up 所需的绝对工作量、最小 official panel 工作量、throughput 预算和并行 worker 目标。

### 假设

H0/P0：v9.3.3 的主 blocker 是 `full_control_outcome_materialization_incomplete`，不是 payload replay、不是 control oracle 本身、不是 action-value feature 本身。

### 必须记录

```text
source_run_id
source_artifact_hash
action_count
candidate_count
event_count
payload_shard_count
payload_tensor_file_written
disk_replay_success_count
disk_replay_error_linf_max
disk_replay_cosine_logged_applied_min
expected_branch_horizon_rows
actual_branch_horizon_rows
full_row_completion_rate
official_candidate_coverage
actual_action_count_covered
branch_count_expected_per_action
horizon_count_expected_per_action
branch_missing_count
horizon_missing_count
missing_secondary_delta_count
matched_control_count_per_event_min
matched_control_count_per_event_mean
offline_rows_per_sec_current
estimated_full_materializer_wallclock_hours_current
estimated_30pct_panel_wallclock_hours_current
throughput_required_for_8h_full
throughput_required_for_4h_30pct_panel
```

### 判断标准

P0 pass：

```text
v9.3.3 route reproduced
payload pass reproduced
row completion rates computed
throughput budget computed
next_required_implementation = scalable_control_outcome_materializer
```

### 可视化

```text
p0_v9330_gate_ladder.svg
p0_control_outcome_completion_bar.svg
p0_branch_horizon_missing_heatmap.svg
p0_materializer_throughput_budget.svg
p0_expected_vs_actual_rows_logscale.svg
```

---

## P1：durable payload regression audit

### 目标

确认 v9.3.3 的 durable payload package 在 v9.3.4 source environment 下仍然全量 replay。P1 是短阶段；不能把 v9.3.4 变成 payload 反复验证实验。

### 假设

H0：durable payload package 已闭合，除非回归。

### 实现

读取 v9.3.3 artifacts：

```text
action_payload_manifest_v9330.json
action_payload_shards_v9330/
action_payload_disk_replay_trace_v9330.csv
```

对所有 2876 actions 执行 disk replay audit。可并行，但必须 deterministic。

### 必须记录

```text
action_id
payload_shard_id
payload_hash_expected
payload_hash_actual
hash_match
replay_success
disk_replay_error_linf
disk_replay_error_relative
disk_replay_cosine_logged_applied
replay_time_ms
```

### 判断标准

P1 pass：

```text
replay_success_count = 2876
payload_hash_match_rate = 1.0
disk_replay_error_linf_max <= 1e-8
disk_replay_cosine_logged_applied_min >= 0.99999999
action_lifecycle_pass = 1
```

### 可视化

```text
p1_disk_replay_error_hist.svg
p1_payload_shard_replay_time.svg
p1_payload_norm_distribution.svg
```

---

## P2：control outcome materializer architecture rebuild

### 目标

把 P2 从 dry-run runner 变成可扩展 materializer。P2 不追 controller，不追 AUC，只追 throughput、coverage、正确性。

### 假设

H1：当前 0.452 rows/sec 主要来自 implementation overhead，而不是不可避免的 branch horizon compute。

### 实现原则

Materializer 应按 action shard、branch group、horizon group 组织，而不是逐行 Python loop。

推荐 shard unit：

```text
shard_id = hash(dataset, seed, action_id_range, horizon_group)
```

每个 shard 内：

```text
1. 一次加载 base checkpoint；
2. 一次加载 action payload shard；
3. 同一 action 上连续运行 6 branches；
4. 同一 branch 上连续记录 20/80/240 horizons；
5. metrics 批量计算；
6. CSV/JSON/tensor 只在 shard end 写出；
7. failed rows 写 retry manifest，不 silently drop。
```

Branches：

```text
RealFunctional
AdamWParallel
AdamWOnly
bestLR
NoOp
Random
```

Horizons：

```text
20
80
240
```

### 必须记录

```text
materializer_architecture_id
worker_count
shard_count
shard_size_actions
branch_batching_enabled
horizon_checkpoint_reuse_enabled
checkpoint_cache_enabled
payload_shard_cache_enabled
metric_batch_compute_enabled
csv_write_batched
retry_manifest_enabled
expected_rows_per_shard
completed_rows_per_shard
failed_rows_per_shard
rows_per_sec_per_worker
rows_per_sec_total
checkpoint_load_fraction
payload_load_fraction
branch_compute_fraction
metric_compute_fraction
write_fraction
idle_fraction
```

### 判断标准

P2 architecture pass：

```text
retry_manifest_enabled = 1
shard_manifest_complete = 1
branch_batching_enabled = 1
checkpoint_cache_enabled = 1
unresolved_failed_rows = 0 in smoke
rows_per_sec_total >= 2.0 in scale smoke
```

Strong pass：

```text
rows_per_sec_total >= 5.0
checkpoint_load_fraction <= 0.10
write_fraction <= 0.20
```

### 可视化

```text
p2_materializer_runtime_waterfall.svg
p2_rows_per_sec_by_worker.svg
p2_shard_completion_gantt.svg
p2_cost_fraction_by_phase.svg
p2_retry_failure_reason.svg
```

---

## P3：scaled full control outcome materialization

### 目标

真实 materialize control outcomes。P3 是 v9.3.4 的主实验，不允许只跑 8 actions。P3 分 progressive panels，但最终必须明确是 full pass、official-minimum pass，还是 throughput fail。

### 假设

H1：scaled materializer 能在合理预算内完成 full 或 official-minimum coverage。

### Progressive panels

#### Panel A：scale smoke

```text
action_count = 64
branches = 6
horizons = 3
expected_rows = 64 * 6 * 3 = 1152
```

目的：验证 branch/horizon semantics、retry、throughput、secondary deltas。

#### Panel B：stratified official-minimum

```text
action_candidate_coverage >= 0.30
sampled_actions >= 864
branches = 6
horizons = 3
expected_rows >= 15552
```

采样方式必须 pre-registered，不能按 outcome 选：

```text
stratify by dataset, seed, family_id, bucket_id, horizon availability, stable_score_bin, payload_norm_bin, candidate_origin_tag
```

#### Panel C：full materialization

```text
action_count = 2876
branches = 6
horizons = 3
expected_rows = 51768
```

### 必须记录

```text
panel_id
panel_type
sample_rule_id
sample_rule_uses_outcome
sample_rule_uses_dataset_for_threshold
action_count_expected
action_count_completed
branch_horizon_row_count_expected
branch_horizon_row_count_actual
action_coverage
row_coverage
branch_completion_rate
horizon_completion_rate
secondary_delta_completion_rate
missing_branch_count
missing_horizon_count
missing_secondary_delta_count
matched_control_count_per_event_min
matched_control_count_per_event_mean
rows_per_sec_total
wallclock_hours
unresolved_failed_rows
retry_count
```

### 判断标准

P3 smoke pass：

```text
Panel A row_coverage = 1.0
missing_secondary_delta_count = 0
unresolved_failed_rows = 0
rows_per_sec_total >= 2.0
```

P3 official-minimum pass：

```text
Panel B action_coverage >= 0.30
Panel B row_coverage >= 0.30
all sampled actions have all 6 branches and all 3 horizons
missing_secondary_delta_count = 0
matched_control_count_per_event_min >= 5
per_dataset_min_action_count >= 100
per_horizon_min_action_count >= 250
unresolved_failed_rows = 0
```

P3 full pass：

```text
Panel C branch_horizon_row_count_actual = 51768
branch_missing_count = 0
horizon_missing_count = 0
missing_secondary_delta_count = 0
full_control_outcome_ready = 1
```

P3 fail：

```text
Panel A fails correctness；
or Panel B cannot reach official-minimum coverage within budget；
or unresolved_failed_rows > 0 without isolated reason。
```

### 可视化

```text
p3_panel_completion_ladder.svg
p3_branch_horizon_completion_heatmap.svg
p3_dataset_seed_family_coverage_heatmap.svg
p3_rows_per_sec_scaling_curve.svg
p3_missing_rows_by_reason.svg
p3_full_vs_panel_cost_projection.svg
```

---

## P4：outcome quality, consistency, and control semantics audit

### 目标

确认 P3 产生的 labels 能用于 control-positive oracle 和 feature learning。P4 不做 controller。

### 假设

H2/P4：如果 labels 不一致或 branch semantics 错，action-value 分析会产生假结论。

### 必须审计

```text
1. label exclusivity：safe_good / bad_event / null_event 是否满足定义；
2. branch identity：RealFunctional / controls 是否实际执行对应 update；
3. horizon consistency：20/80/240 是否同一 branch trajectory 的 checkpoints；
4. control fairness：bestLR / AdamWParallel 是否使用同等 compute/data horizon；
5. NoOp semantics：NoOp 是否只前进 training stream 而不额外 update；
6. Random semantics：Random 是否 matched norm / matched support；
7. branch state hash：start/end state 是否可复现；
8. metric monotonic sanity：极端 CEp99 / margin / acc delta 是否需要人工检查；
9. replay determinism：sampled rows rerun 是否数值一致。
```

### 必须记录

```text
quality_audit_id
rows_checked
label_exclusivity_violation_count
branch_identity_violation_count
horizon_state_hash_mismatch_count
control_branch_missing_count
control_branch_unfair_count
noop_semantics_violation_count
random_norm_match_error_max
metric_nan_count
metric_inf_count
metric_outlier_count
rerun_sample_count
rerun_metric_error_max
quality_audit_pass
```

### 判断标准

P4 pass：

```text
label_exclusivity_violation_count = 0
branch_identity_violation_count = 0
horizon_state_hash_mismatch_count = 0
control_branch_missing_count = 0 for official rows
metric_nan_count = 0
metric_inf_count = 0
rerun_metric_error_max <= tolerance
quality_audit_pass = 1
```

### 可视化

```text
p4_label_exclusivity_matrix.svg
p4_branch_metric_delta_distribution.svg
p4_horizon_consistency_trajectory.svg
p4_control_fairness_dashboard.svg
p4_rerun_error_hist.svg
```

---

## P5：control-positive oracle and action-value distribution

### 目标

首次在非 dry-run control outcomes 上判断：当前 action population 是否存在 control-positive frontier。如果 oracle 都不存在，后续不应继续 controller，而应回到 action primitive。

### 假设

H2：当前 action population 可能存在 control-positive frontier，但 v9.3.3 没有足够 outcome rows 评估。

### Oracle candidates

```text
OR0-PrimarySafeGoodOracle
OR1-ControlPositiveOracle
OR2-VctrlMaxOracle
OR3-HorizonRobustOracle
OR4-BadMinControlPositiveOracle
OR5-NullMinControlPositiveOracle
OR6-SupportBalancedControlPositiveOracle
OR7-ParetoValueRiskNullOracle
```

### 必须记录

```text
oracle_id
outcome_panel_id
candidate_universe
accepted_count
coverage
precision_primary
precision_control_positive
bad_event_rate
null_rate
V_ctrl_mean
V_ctrl_median
V_ctrl_lcb
V_robust_mean
beats_adamwparallel_rate
beats_bestlr_rate
beats_noop_rate
beats_random_rate
support_balance_pass
accepted_dataset_count
accepted_seed_count
accepted_family_count
accepted_signal_strata_count
max_dataset_share
max_family_share
max_stratum_share
```

### 判断标准

Weak control oracle pass：

```text
coverage >= 0.03
precision_control_positive >= 0.75
bad_event_rate <= 0.05
null_rate <= 0.15
beats_adamwparallel_rate >= 0.50
beats_bestlr_rate >= 0.50
support_balance_pass = 1
```

Strong control oracle pass：

```text
coverage >= 0.03
precision_control_positive >= 0.90
bad_event_rate <= 0.02
null_rate <= 0.10
beats_adamwparallel_rate >= 0.60
beats_bestlr_rate >= 0.60
V_ctrl_lcb > 0
support_balance_pass = 1
```

Oracle fail：

```text
No oracle reaches coverage >=0.03 with bad_event_rate <=0.05 and beats controls meaningfully。
```

若 P5 oracle fail，route：

```text
R5-ControlPositiveOracleAbsent
```

下一步重建 action primitive / candidate generation。

### 可视化

```text
p5_control_positive_oracle_frontier.svg
p5_vctrl_distribution_by_horizon.svg
p5_real_vs_controls_pareto.svg
p5_oracle_support_balance.svg
p5_primary_safe_vs_control_positive_overlap.svg
p5_horizon_robustness_heatmap.svg
```

---

## P6：action-value observability disentanglement

### 目标

在 P3/P4/P5 ready 后，评估 legal pre-commit features 是否能预测 value、bad、null、support，而不是继续找单个 magic score。

### 假设

H3：action-value observability 可能存在，但需要 value / bad / null / support 分开建模。

### Feature groups

#### F0：legacy / StableAccept features

```text
stable_score_q
stable_rank
score_margin
stable_accept_bit
```

#### F1：payload geometry

```text
payload_norm
payload_linf_norm
payload_role_entropy
payload_nonzero_fraction
payload_tail_selectivity
```

#### F2：functional-vs-AdamW alignment

```text
cos_functional_delta_adamw_delta
cos_functional_delta_negative_grad
functional_norm_over_adamw_norm
projected_CE_descent_estimate
gradient_conflict_rate
```

#### F3：state / hard-tail features

```text
batch_CE_mean
batch_CE_p90
batch_CE_p99
margin_p10
wrong_conf_p90
entropy_mean
hard_tail_fraction
```

#### F4：support / reliability features

```text
support_count
support_lcb
family_support_count
family_value_lcb
family_bad_ucb
horizon_tail_risk
candidate_origin_risk
empirical_bayes_value_lcb
empirical_bayes_bad_ucb
```

#### F5：cost features

```text
candidate_count_in_step
active_step_flag
estimated_feature_compute_cost
estimated_payload_apply_cost
kernel_shape_id
runtime_bucket_id
```

### Targets

```text
Y_value = V_ctrl > 0
Y_bad = bad_event_label
Y_null = null_event_label
Y_support = support_balance / effective sample quality
Y_control_positive = control_positive_label
```

### 必须记录

```text
feature_group
feature_id
uses_dataset_name
uses_outcome_at_commit
uses_future_step
feature_missing_rate
feature_compute_time_ms_q90
feature_memory_ratio
AUC_value_positive
AUC_control_positive
AUC_bad_event
AUC_null_event
PR_AUC_control_positive
PR_AUC_bad_lift
Brier_bad
ECE_bad
calibration_slope_bad
leave_dataset_auc_drop
leave_stratum_auc_drop
monotone_sign_consistency
single_feature_baseline_score
ablation_delta
```

### 判断标准

P6 feature diagnostic pass：

```text
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
feature_missing_rate <= 0.01
feature_compute_time_ms_q90 recorded
```

P6 observability pass：

```text
AUC_control_positive >= 0.75 or PR_AUC_control_positive lift >= 2.0
AUC_bad_event >= 0.80 or PR_AUC_bad_lift >= 2.0
AUC_null_event >= 0.75
ECE_bad <= 0.05
leave_dataset_auc_drop <= 0.07
leave_stratum_auc_drop <= 0.10
```

Minimality pass：

```text
selected_feature_groups <= 5
single-feature baselines reported
ablation reported
monotone signs stable across folds
no opaque 20-feature pile
```

### 可视化

```text
p6_feature_auc_pr_lift_bar.svg
p6_value_bad_null_disentanglement_matrix.svg
p6_risk_calibration_curve.svg
p6_feature_cost_vs_auc_pareto.svg
p6_leaveout_auc_drop_heatmap.svg
p6_feature_ablation_waterfall.svg
```

---

## P7：cross-fitted minimal action-value controller

### 目标

如果 P5 oracle exists 且 P6 observability 有信号，则训练/选择 minimal legal controller。P7 不能使用 dataset-specific branch，不能使用 heldout labels 选 threshold。

### Controller form

主 controller：

$$
Accept(e)=1
\iff
LCB(V_{ctrl}(e))>0
\land UCB(Bad(e))\le\tau_b
\land UCB(Null(e))\le\tau_n
\land LCB(Support(e))\ge\tau_s
\land C(e)\le C_{max}.
$$

Monotone score optional：

$$
S(e)=a_1LCB(V_{ctrl}(e))-a_2UCB(Bad(e))-a_3UCB(Null(e))+a_4LCB(Support(e))-a_5C(e),
$$

with：

```text
a_i >= 0
selected_feature_groups <= 5
thresholds frozen on calibration folds
```

### Splits

```text
Seed-fold cross fitting
Leave-dataset-out
Leave-stratum-out
Leave-family-out diagnostic
```

### 必须记录

```text
controller_id
feature_groups
model_class
monotone_constraints
thresholds
calibration_fold
heldout_fold
split_type
heldout_entity
dataset_name_used
accepted_count_cal
accepted_count_heldout
coverage_cal
coverage_heldout
precision_control_positive_cal
precision_control_positive_heldout
bad_event_cal
bad_event_heldout
null_rate_cal
null_rate_heldout
V_ctrl_mean_cal
V_ctrl_mean_heldout
V_ctrl_lcb_heldout
beats_adamwparallel_rate_heldout
beats_bestlr_rate_heldout
precision_lcb
bad_event_ucb
accepted_family_count
accepted_signal_strata_count
max_family_share
max_stratum_share
oracle_overlap
stableaccept_overlap
```

### 判断标准

P7 decision pass：

```text
coverage_heldout in [0.03, 0.15]
precision_control_positive_heldout >= 0.75
bad_event_heldout <= 0.05
null_rate_heldout <= 0.15
V_ctrl_lcb_heldout > 0
beats_adamwparallel_rate_heldout >= 0.50
beats_bestlr_rate_heldout >= 0.50
precision_lcb >= 0.75
bad_event_ucb <= 0.05
support_balance_pass = 1
dataset_name_used = 0
```

Cross-fit stability：

```text
coverage_heldout > 0 in every main fold
bad_event gate pass in at least 2/3 seed folds
LDO no heldout dataset bad_event_rate > 0.10
LSO macro precision >= 0.75
```

### 可视化

```text
p7_controller_value_risk_coverage_frontier.svg
p7_crossfit_fold_matrix.svg
p7_calibration_to_heldout_drift.svg
p7_controller_oracle_overlap.svg
p7_support_balance_sunburst.svg
p7_vctrl_bad_null_phase_diagram.svg
```

---

## P8：online controller runtime remeasurement

### 目标

独立重测 online runtime。P8 与 P3 materializer 分开，不允许 offline materializer 成本进入 online timed path。

### 假设

H4：online runtime 只有在 control outcome materializer 完全移出 timed path 后才有意义。

### Runtime candidates

```text
RT0-v9320-reference-measured-partial
RT1-v9330-runtime-decoupled-not-remeasured-reference
RT2-online-controller-only-durable-payload-lookup
RT3-online-controller-only-preloaded-payload
RT4-online-feature-score-accept-no-payload-apply
RT5-online-full-accept-and-payload-apply
```

RT2/RT3 区分很重要：如果 disk lookup 太慢，可以把 durable payload package 作为 external reproducibility artifact，但 online path 使用 preloaded payload pointers。两者都要记录，不得混淆。

### 必须记录

```text
runtime_candidate_id
runtime_mode
control_outcome_materializer_in_timed_path
disk_payload_lookup_in_timed_path
payload_preloaded
step_count
active_step_count
zero_candidate_step_count
candidate_count
accepted_count
zero_candidate_controller_kernel_count
zero_candidate_controller_sync_count
controller_launches_per_active_step_mean
controller_launches_per_active_step_q90
controller_syncs_per_active_step_q90
candidate_pack_time_ms_q90
feature_compute_time_ms_q90
score_accept_time_ms_q90
payload_lookup_time_ms_q90
payload_apply_time_ms_q90
audit_outside_timed_ms
base_train_step_time_ms_q90
total_step_time_ms_q90
mlp_step_time_ms_q90
step_ratio_q90
memory_ratio
accept_disagreement_count
payload_apply_error_linf_max
```

### 判断标准

P8 online runtime pass：

```text
runtime_mode = online_sequential_official_runtime
control_outcome_materializer_in_timed_path = 0
offline_audit_in_timed_path = 0
zero_candidate_controller_kernel_count = 0
zero_candidate_controller_sync_count = 0
controller_launches_per_active_step_q90 <= 2
controller_syncs_per_active_step_q90 <= 1
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
accept_disagreement_count = 0
payload_apply_error_linf_max <= tolerance
```

Diagnostic pass：

```text
step_ratio_q90 <= 2.00
launch/sync reduction >= 80%
```

### 可视化

```text
p8_online_runtime_mode_comparison.svg
p8_online_vs_offline_runtime_separation.svg
p8_step_ratio_q50_q90.svg
p8_active_empty_step_timing.svg
p8_payload_lookup_apply_waterfall.svg
p8_memory_ratio.svg
```

---

## P9：system controller integration

### 目标

只有 P3/P4/P5/P6/P7/P8 都满足对应 gate 时，组合 system controller。P9 不允许因单项 pass 写成 system pass。

### 必须记录

```text
system_candidate_id
controller_id
runtime_candidate_id
outcome_panel_id
full_control_outcome_ready
official_minimum_panel_ready
control_oracle_pass
action_value_observability_pass
decision_gate_pass
runtime_gate_pass
payload_binding_pass
action_lifecycle_pass
secondary_outcome_ready
candidate_count
action_count
accepted_count
coverage_heldout
precision_control_positive_heldout
bad_event_heldout
null_rate_heldout
V_ctrl_lcb_heldout
beats_adamwparallel_rate_heldout
beats_bestlr_rate_heldout
step_ratio_q90
memory_ratio
feature_compute_time_ms_q90
uses_dataset_name_for_controller
uses_loss_backward
uses_teacher
uses_loss_modification
projection_used
source_gap_used
formula_proxy_used
diagnostic_promoted_to_official
official_eligible
system_legal_controller_pass
```

### 判断标准

P9 system pass：

```text
official_eligible = 1
system_legal_controller_pass = 1
action_lifecycle_pass = 1
payload_binding_pass = 1
secondary_outcome_ready = 1
control_oracle_pass = 1
decision_gate_pass = 1
runtime_gate_pass = 1
uses_dataset_name_for_controller = 0
uses_loss_backward = 0
uses_teacher = 0
uses_loss_modification = 0
projection_used = 0
source_gap_used = 0
formula_proxy_used = 0
diagnostic_promoted_to_official = 0
```

Decision gates：

```text
coverage_heldout in [0.03, 0.15]
precision_control_positive_heldout >= 0.75
bad_event_heldout <= 0.05
null_rate_heldout <= 0.15
V_ctrl_lcb_heldout > 0
beats_adamwparallel_rate_heldout >= 0.50
beats_bestlr_rate_heldout >= 0.50
```

Runtime gates：

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

### 可视化

```text
p9_system_gate_dashboard.svg
p9_quality_cost_pareto.svg
p9_controller_runtime_frontier.svg
p9_failure_reason_matrix.svg
```

---

## P10：parallel diagnostic causal scouts

### 目标

加快实验，但不污染 controller selection。P10 可以在 P7 top candidates 出现后并行运行，但必须记录隔离标志。

### 设置

```text
controllers = top2 P7 candidates + StableAccept negative control + NoFunctional baseline
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random, ShuffledPayload, ShuffledScore
horizons = 20,80,240
seeds = diagnostic subset
```

### 必须记录

```text
diagnostic_id
controller_id
dataset
seed
horizon
branch
accepted_count
coverage
bad_event_rate
null_rate
V_ctrl_mean
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
beats_adamwparallel
beats_bestlr
task_safe
used_for_controller_selection
used_for_threshold_selection
status
```

### 判断标准

P10 scout promising：

```text
used_for_controller_selection = 0
used_for_threshold_selection = 0
RealFunctional beats AdamWParallel in >= 50% macro diagnostic slices
RealFunctional beats bestLR in >= 50% macro diagnostic slices
shuffled controls do not match RealFunctional pattern
```

P10 fail 不代表 route fail，除非 P9/P11 official 之后仍 fail。

### 可视化

```text
p10_diagnostic_branch_pareto.svg
p10_shuffle_control_matrix.svg
p10_macro_beat_rate_diagnostic.svg
p10_diagnostic_isolation_audit.svg
```

---

## P11：leave-dataset-out / leave-stratum-out official

### 目标

只有 P9 system pass 后打开 official LDO/LSO。验证 controller 不是 pooled calibration artifact，不是 dataset-specific tuning。

### 设置

Leave-dataset-out：

```text
train/calibrate MNIST + Fashion, evaluate KMNIST
train/calibrate MNIST + KMNIST, evaluate Fashion
train/calibrate Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
train/calibrate all but one signal stratum
evaluate held-out stratum
```

### 必须记录

```text
split_type
heldout_entity
controller_id
runtime_candidate_id
outcome_panel_id
accepted_count
coverage
precision_control_positive
bad_event_rate
null_rate
V_ctrl_lcb
beats_adamwparallel_rate
beats_bestlr_rate
step_ratio_q90
memory_ratio
dataset_name_used
support_balance_pass
```

### 判断标准

LDO pass：

```text
at least 2/3 held-out datasets pass decision gates
no held-out dataset bad_event_rate > 0.10
coverage > 0 for all held-out datasets
dataset_name_used = 0
```

LSO pass：

```text
>=70% held-out strata task-safe
macro bad_event_rate <= 0.05
macro precision_control_positive >= 0.75
macro V_ctrl_lcb > 0
```

### 可视化

```text
p11_leave_dataset_out_matrix.svg
p11_leave_stratum_out_matrix.svg
p11_dataset_diagnostic_no_tuning_audit.svg
p11_support_hole_heatmap.svg
```

---

## P12：official paired replay

### 目标

只有 P9 system pass 和 P11 leave-out pass 后打开。验证 RealFunctional 是否在 strong controls 下有 causal advantage。

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
  ShuffledOGPValueScore
  ShuffledOGPRiskScore
  ShuffledOGPSupportScore
  ShuffledCandidatePayload
  ShuffledEventRoute
  FunctionalChannelShuffled
  TailMaskShuffled
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
V_ctrl
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
p12_official_paired_replay_pareto.svg
p12_macro_beat_rate.svg
p12_signal_stratum_win_matrix.svg
p12_shuffle_control_matrix.svg
p12_system_gate_distribution.svg
```

---

## P13：short-run / full-run / sample efficiency / continual robustness

### 目标

只有 P12 pass 后打开。验证 local causal advantage 能否进入连续训练，并排除 LR grid / QuadraticFeatureMLP / shuffled controller explanations。

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
  ShuffledFunctionalPayload
  ShuffledOGPValueScore
  ShuffledOGPRiskScore
  ShuffledRuntimePath
```

Sample efficiency：

```text
train_fraction = 1%, 2%, 5%, 10%, 25%, 100%
time_budget = matched wallclock
step_budget = matched steps
```

Continual / anti-forgetting：

```text
Task sequence A -> B -> C
retained_accuracy_old_tasks
forgetting = best_old_acc - final_old_acc
backward_transfer
forward_transfer
old_task_CEp99_drift
old_task_margin_drift
```

### 必须记录

```text
dataset
seed
candidate
steps
train_fraction
wallclock_time
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
strong_baseline_beaten
robustness_pass
retained_accuracy
forgetting
backward_transfer
forward_transfer
base_checkpoint_hash
```

### 判断标准

Short/full functional pass：

```text
Acc_functional >= Acc_AdamW - 0.005
```

且至少一个成立：

```text
Acc_functional > Acc_AdamWParallel
ECE_functional < ECE_AdamW
NLL_functional < NLL_AdamW
Curvature_functional <= 0.90 * Curvature_AdamW
ValLossAUC_time_functional < ValLossAUC_time_AdamW
```

Sample efficiency pass：

```text
Functional reaches same target accuracy with <= 0.80 steps or <= 0.80 wallclock in at least two datasets；
或在 same data fraction 下 macro metric 显著优于 controls。
```

Continual pass：

```text
forgetting_functional <= 0.80 * forgetting_AdamW
retained_accuracy_functional >= retained_accuracy_AdamW
no old-task CEp99 explosion
```

### 可视化

```text
p13_short_full_learning_curves.svg
p13_val_loss_auc_step_time.svg
p13_time_to_target.svg
p13_sample_efficiency_curve.svg
p13_continual_forgetting_bar.svg
p13_robustness_strong_baseline_matrix.svg
```

---

# 8. 并行执行计划

v9.3.4 必须并行，不要一轮只跑一个 dry-run。

## Batch A：立即并行的基础设施

```text
A1: P0 boundary reanalysis and budget computation
A2: P1 durable payload regression audit
A3: P2 materializer architecture rebuild smoke
A4: P8 online runtime remeasurement smoke with no materializer
```

输出：

```text
p0_boundary_budget_v9340.csv
p1_payload_regression_audit_v9340.csv
p2_materializer_architecture_smoke_v9340.csv
p8_online_runtime_smoke_v9340.csv
```

## Batch B：materializer scale-up 与 runtime 并行

P2 smoke pass 后：

```text
B1: P3 Panel A scale smoke, 64 actions
B2: P3 Panel B stratified official-minimum, >=864 actions
B3: P8 online runtime full remeasure
B4: P4 outcome quality audit on completed shards
```

注意：P4 可 streaming audit，不必等 P3 full 完成。

## Batch C：oracle / feature / throughput fallback 并行

P3 Panel B 至少达到 30% coverage 后：

```text
C1: P5 control-positive oracle on official-minimum panel
C2: P6 action-value feature disentanglement
C3: P3 Panel C full materialization continues in background of current run execution, not asynchronous promise
C4: materializer bottleneck autopsy if throughput below target
```

这里的 “continues” 指同一执行计划中的并行 runner，不是承诺后台异步交付。

## Batch D：controller / system

只有 P5 oracle exists 且 P6 observability pass：

```text
D1: P7 cross-fitted minimal action-value controller
D2: P9 system integration with P8 runtime survivor
D3: P10 diagnostic causal scout isolated
```

## Batch E：official downstream

只有 P9 pass：

```text
E1: P11 leave-dataset-out / leave-stratum-out
E2: P12 official paired replay
E3: P13 short/full/sample-efficiency/continual only if P12 pass
```

---

# 9. Required artifacts

```text
run_manifest.json
contract_audit_v9340.csv
provenance_audit_v9340.csv
artifact_hashes.csv
route_decision.json
aggregate_decision.json
failure_table.csv

p0_boundary_budget_v9340.csv
p1_payload_regression_audit_v9340.csv
p2_materializer_architecture_rebuild.csv
p3_scaled_full_control_outcome_materializer.csv
full_control_outcome_table_v9340.csv
branch_horizon_completion_trace_v9340.csv
matched_control_outcome_trace_v9340.csv
materializer_worker_trace_v9340.csv
materializer_retry_manifest_v9340.csv
p4_outcome_quality_consistency_audit.csv
p5_control_positive_oracle.csv
control_positive_oracle_trace_v9340.csv
p6_action_value_observability_disentanglement.csv
action_value_feature_trace_v9340.csv
feature_cost_trace_v9340.csv
p7_crossfitted_action_value_controller.csv
controller_frontier_trace_v9340.csv
p8_online_runtime_remeasurement.csv
online_runtime_component_trace_v9340.csv
p9_system_controller_v9340.csv
system_controller_trace_v9340.csv
p10_diagnostic_causal_scout.csv
diagnostic_isolation_audit_v9340.csv
p11_leave_dataset_stratum_out.csv
leaveout_trace_v9340.csv
p12_official_paired_replay.csv
official_paired_replay_trace_v9340.csv
p13_short_full_sampleeff_continual_robustness.csv
short_full_trace_v9340.csv
figures/
```

---

# 10. Failure taxonomy

```text
F1_contract_violation
F2_dataset_tuning_detected
F3_teacher_or_loss_modification_detected
F4_payload_regression
F5_payload_hash_mismatch
F6_disk_replay_error_regression
F7_materializer_architecture_smoke_fail
F8_materializer_throughput_fail
F9_materializer_shard_retry_unresolved
F10_control_outcome_row_missing
F11_branch_missing
F12_horizon_missing
F13_secondary_delta_missing
F14_branch_identity_violation
F15_horizon_state_hash_mismatch
F16_control_branch_unfair
F17_label_exclusivity_violation
F18_control_positive_oracle_absent
F19_control_positive_oracle_panel_too_small
F20_action_value_observability_fail
F21_feature_cost_infeasible
F22_minimality_gate_fail
F23_no_dataset_agnostic_controller
F24_controller_precision_fail
F25_controller_coverage_fail
F26_controller_bad_event_fail
F27_controller_null_rate_fail
F28_controller_vctrl_lcb_fail
F29_controller_beats_controls_fail
F30_support_balance_fail
F31_online_runtime_not_remeasured
F32_online_offline_runtime_conflated
F33_zero_candidate_runtime_fail
F34_active_step_launch_count_fail
F35_step_ratio_fail
F36_memory_ratio_fail
F37_system_controller_not_official
F38_leave_dataset_out_fail
F39_leave_stratum_out_fail
F40_paired_replay_control_equivalent
F41_shuffle_control_pass
F42_short_run_task_drop
F43_full_run_no_macro_or_hard_stratum_gain
F44_sample_efficiency_fail
F45_continual_forgetting_fail
F46_strong_baseline_explains_gain
F47_external_not_ready
F48_fake_or_proxy_violation
F49_artifact_missing
```

---

# 11. Route decision

```text
R1-BoundaryAndPayloadConfirmed:
  v9.3.3 boundary reproduced and durable payload package remains closed.

R2-ControlOutcomeMaterializerThroughputFail:
  materializer cannot reach official-minimum coverage within throughput/cost budget.

R3-OfficialMinimumControlOutcomeReady:
  >=30% stratified panel complete with all branches/horizons/secondary deltas.

R4-FullControlOutcomeReady:
  all 51768 branch-horizon rows complete, branch/horizon/secondary missing = 0.

R5-ControlPositiveOracleAbsent:
  even oracle cannot find control-positive frontier at required coverage.

R6-ControlPositiveOraclePass:
  current action population has useful control-positive frontier.

R7-ActionValueObservabilityFail:
  legal features cannot predict value/risk/null/support sufficiently.

R8-ActionValueObservabilityPass:
  legal minimal feature groups show stable action-value observability.

R9-ActionValueControllerPass:
  cross-fitted dataset-agnostic controller passes value/risk/null gates.

R10-OnlineRuntimePass:
  online controller runtime passes step_ratio/memory gates.

R11-OnlineRuntimeFail:
  online runtime remains above envelope after remeasurement.

R12-SystemLegalControllerPass:
  decision + runtime + contracts pass.

R13-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R14-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R15-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R16-PairedReplayFail:
  system legal but functional update control-equivalent.

R17-ShortFullSampleEfficiencyPass:
  short/full/sample-efficiency/continual gates pass.

R18-CandidateActionPrimitiveResetRequired:
  control-positive oracle absent or action-value not robust across horizons.

R19-ExternalReady:
  strict PureKAN functional route passes final external-ready gates.
```

`route_decision.json` 必须记录：

```text
route
base_candidate
candidate_count
action_count
event_count
payload_pass
payload_shard_count
payload_hash_match_rate
disk_replay_success_count
disk_replay_error_linf_max
disk_replay_cosine_logged_applied_min
materializer_architecture_pass
outcome_panel_id
full_control_outcome_ready
official_minimum_panel_ready
expected_branch_horizon_rows
actual_branch_horizon_rows
official_candidate_coverage
branch_completion_rate
horizon_completion_rate
branch_missing_count
horizon_missing_count
missing_secondary_delta_count
rows_per_sec_total
wallclock_hours
quality_audit_pass
control_oracle_pass
control_oracle_coverage
control_oracle_precision
control_oracle_bad_event
control_oracle_null_rate
control_oracle_vctrl_lcb
beats_adamwparallel_rate
beats_bestlr_rate
action_value_observability_pass
best_feature_group
best_auc_control_positive
best_auc_bad
best_auc_null
feature_cost_pass
minimality_pass
controller_id
decision_gate_pass
coverage_heldout
precision_control_positive_heldout
bad_event_heldout
null_rate_heldout
V_ctrl_lcb_heldout
online_runtime_remeasured
online_offline_runtime_conflated
control_outcome_materializer_in_timed_path
runtime_candidate_id
step_ratio_q90
memory_ratio
online_runtime_pass
system_legal_controller_pass
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
sample_efficiency_pass
continual_pass
external_ready
primary_blocker
next_required_implementation
success_v9340_strict_purekan_functional
success_v9340_full_functional
success_v9340_external_ready
fake_data_used
proxy_row_used
cpu_offload_used
uses_dataset_name_for_controller
uses_loss_backward
uses_teacher
uses_loss_modification
projection_used
source_gap_used
formula_proxy_used
diagnostic_promoted_to_official
```

---

# 12. 停止条件

## Minimum diagnostic success

```text
P0 boundary/budget pass
P1 durable payload regression audit pass
P2 materializer architecture smoke pass
P3 at least Panel A scale smoke pass
P4 quality audit pass for completed rows
P8 online runtime remeasured
no fake/proxy/offload/teacher/loss violation
```

## Official-minimum materializer success

```text
P3 Panel B action_coverage >= 0.30
all sampled actions have 6 branches and 3 horizons
missing_secondary_delta_count = 0
matched_control_count_per_event_min >= 5
quality_audit_pass = 1
```

## Full materializer success

```text
branch_horizon_row_count_actual = 51768
branch_missing_count = 0
horizon_missing_count = 0
missing_secondary_delta_count = 0
quality_audit_pass = 1
full_control_outcome_ready = 1
```

## Pivot to materializer redesign

如果：

```text
P2 architecture smoke fails
or P3 Panel A correctness fails
or P3 Panel B cannot reach official-minimum coverage
or rows_per_sec_total < 1.0 after scale attempts
```

则停止 feature/controller work，进入 materializer redesign。

## Pivot to action primitive reset

如果：

```text
P3/P4 ready
but P5 control-positive oracle fail
```

则说明当前 action population 即使在 oracle 下也没有足够 control-positive frontier。下一步重建 candidate/action primitive，而不是 controller。

## Pivot to observability redesign

如果：

```text
P5 oracle pass
but P6 action-value observability fail
```

则说明好 action 存在，但 legal commit-time observables 仍不足。下一步重建 action-conditioned observable primitives。

## Pivot to controller redesign

如果：

```text
P6 observability pass
but P7 controller fail
```

则说明 feature 有信息，但 calibration/support/monotone controller 不稳。下一步修 cross-fitting、support confidence、threshold freezing。

## Pivot to runtime redesign

如果：

```text
P7 decision pass
but P8 online runtime fail
```

则停止 controller tuning，修 online sparse event runtime。

## Open official causal validation

只有当：

```text
P9 system pass
and P11 leave-out pass
```

才 official 打开 P12 paired replay。

## Full functional success

只有当：

```text
P12 paired replay pass
and P13 short/full/sample-efficiency/continual robustness pass
```

才能声明 full functional success。

---

# 13. 最终建议

v9.3.4 的一句话策略是：

$$
\boxed{
\text{不要再围绕 feature 或 controller 小修小补；先把 full control outcome label universe 做出来，并让 action-value 第一次可被严肃评估。}
}
$$

具体执行重点：

```text
1. 保留 v9.3.3 durable payload package 作为已闭合资产；
2. 将 P2 materializer 从 8-action dry-run 扩展到 official-minimum panel 或 full universe；
3. 并行化 branch/horizon materialization，记录 worker throughput 和 bottleneck；
4. 所有 sampled actions 必须完整跑 6 branches × 3 horizons，不能只跑 horizon=20；
5. 补齐 CEp99 / margin / ECE / NLL / curvature / real-vs-control deltas；
6. 用 full/panel outcomes 重新定义 control-positive oracle；
7. 若 oracle 不存在，立即 pivot action primitive；
8. 若 oracle 存在，再做 legal action-value observability；
9. controller 必须 minimal、monotone、cross-fitted、dataset-agnostic；
10. online runtime 必须单独重测，不混入 offline materializer；
11. diagnostics 可以并行，但不能影响 official controller thresholds；
12. 不在 system pass 前打开 official paired replay/full-run。
```

v9.3.3 之后最重要的判断是：

$$
\boxed{
\text{项目不是卡在 action 能不能 replay，而是卡在 action 的 matched-control value 还没有被足量、完整、可审计地 materialize。}
}
$$

v9.3.4 成功的标志不是“某个新 feature AUC 提升”，而是：

```text
full_control_outcome_ready 或 official_minimum_control_outcome_ready；
control-positive oracle 被真实评估；
action-value observability 在足够 label universe 上被评估；
online runtime 被重新测量；
route 不再停在 R4-ControlOutcomeMaterializationFail。
```
