# DG-KAN v9.3.2 Action Apply / Control Outcome Materialization 与 Event Runtime 实验复盘

> 本复盘记录 `DG-KAN_v9.3.2_ActionApplyControlOutcome_EventRuntime_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 partial control probe、primary oracle、action-conditioned feature diagnostic 或 measured scheduler trace 写成 official system pass。

## 0. 最新结论

```text
route = R4-SecondaryControlOutcomeMaterializationFail
base_candidate = LQ-t2-h256
success_v9320_strict_purekan_functional = False
success_v9320_full_functional = False
success_v9320_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9320_action_apply_control_outcome_event_runtime_first_20260513T223000Z/
```

核心结论：

1. P0 复现 v9.3.1 boundary：source route = `R1-BoundaryReanalyzed`，v9.3.1 中 action apply / secondary-control / event runtime 均未 official。
2. 本轮不再 audit-only：新落盘 action apply rows = `2876`，secondary/control probe sample action = `143`，runtime step rows = `8064`，`no_audit_only_guard_pass = 1`。
3. P1 action apply error 已全量测量：`action_apply_error_measured = 1`，missing count = `0`，replay success = `2876 / 2876`。
4. P1 error 数值闭合：max L∞ = `7.450580596923828e-09`，max relative = `1.1583176888604244e-05`，min logged/applied cosine = `0.9999999932719165`。
5. 注意：P1 payload 是同轮 train-stream 重建并内存 replay；没有单独 `.pt` tensor payload 文件，`payload_tensor_file_written = 0`。因此不能把本轮写成 external-ready replay package。
6. P2 只完成 partial immediate-probe controls：sample action = `143`，coverage = `0.04972183588317107`，matched controls min/mean = `2 / 2.0`，低于计划要求 `>=4`。
7. P2 不能 official：branch missing count = `429`，horizon missing count = `286`，missing secondary delta count = `1716`，`secondary_outcome_ready = 0`。
8. P3 primary oracle 仍存在：candidate/action heldout safe-good oracle accepted = `494`，precision = `1.0`，coverage = `0.05445326278659612`，bad/null = `0.0 / 0.0`；但 control oracle 被 P2 阻塞。
9. P4 action-conditioned feature diagnostic 未过：best feature = `AVF1-PayloadNorm`，top273 safe/bad/null = `155 / 20 / 81`，`action_conditioned_feature_pass = 0`。
10. P7 写入 measured scheduler trace：zero-candidate controller kernel/sync = `0 / 0`，但 active step launches q90 = `3.0`，step ratio q90 = `5.697758752709664`，所以 `event_driven_runtime_pass = 0`。
11. P8 system controller 未打开：`official_eligible = 0`，reason = `secondary_control_outcome_materialization_incomplete`。
12. 当前 blocker：`secondary_control_outcome_materialization_incomplete`；下一步必须 materialize full matched controls 与 horizons，而不是拿 immediate-probe sample 转正。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9320_action_apply_control_outcome_event_runtime.py` | v9.3.2 runner；重建 train stream，materialize action apply replay error，写 partial secondary/control probe、action-conditioned features、measured event scheduler trace 和 P8 gate |

代码检查：

```text
python -m py_compile experiments/run_v9320_action_apply_control_outcome_event_runtime.py
```

正式运行：

```bash
python experiments/run_v9320_action_apply_control_outcome_event_runtime.py \
  --out-dir results/real_rerun_20260506/v9320_action_apply_control_outcome_event_runtime_first_20260513T223000Z \
  --fresh --device auto --data-root data --seed 1314
```

说明：第一次正式运行在所有主要 CSV 写完后，于 no-fake audit 收尾处因 `audit_no_fake` 参数类型错误退出；随后只修正 audit/hash/manifest 收尾，不重算数据。该错误不影响已落盘 P1/P2/P7 测量值；最终 artifact 包含修正后的 provenance audit 与 hash。

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R4-SecondaryControlOutcomeMaterializationFail",
  "source_route_v9310": "R1-BoundaryReanalyzed",
  "candidate_count": 2876,
  "action_count": 2876,
  "event_count": 24192,
  "no_audit_only_guard_pass": 1,
  "action_apply_error_measured": 1,
  "action_apply_error_missing_count": 0,
  "action_apply_error_linf_max": 7.450580596923828e-09,
  "action_apply_error_relative_max": 1.1583176888604244e-05,
  "action_apply_cosine_logged_applied_min": 0.9999999932719165,
  "action_lifecycle_pass": 1,
  "secondary_outcome_ready": 0,
  "outcome_sample_action_count": 143,
  "official_sample_coverage": 0.04972183588317107,
  "matched_control_count_per_event_min": 2,
  "branch_missing_count": 429,
  "horizon_missing_count": 286,
  "missing_secondary_delta_count": 1716,
  "primary_oracle_pass": 1,
  "control_oracle_pass": 0,
  "best_action_feature_id": "AVF1-PayloadNorm",
  "best_action_feature_top273_safe_good": 155,
  "best_action_feature_top273_bad_event": 20,
  "action_conditioned_feature_pass": 0,
  "runtime_candidate_id": "RT2-ActionReplayEventSchedulerMeasuredPartial",
  "runtime_measured": 1,
  "zero_candidate_controller_kernel_count": 0,
  "controller_launches_per_active_step_q90": 3.0,
  "step_ratio_q90": 5.697758752709664,
  "event_driven_runtime_pass": 0,
  "official_eligible": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "secondary_control_outcome_materialization_incomplete"
}
```

判断：本轮真实推进了 action apply materialization；但 P2 secondary/control 不完整，所以 route 正确停在 `R4`。

## 3. P1 Action Apply

Artifacts：

```text
p1_action_apply_materializer.csv
action_apply_trace_v9320.csv
action_payload_replay_manifest_v9320.json
```

Summary：

```text
action_apply_materializer_id = AAP1-RecomputedFunctionalDeltaReplay
action_count = 2876
action_apply_rows = 2876
action_apply_error_measured = 1
action_apply_error_missing_count = 0
replay_success_count = 2876
action_apply_error_linf_max = 7.450580596923828e-09
action_apply_error_relative_max = 1.1583176888604244e-05
action_apply_cosine_logged_applied_min = 0.9999999932719165
payload_tensor_file_written = 0
payload_replay_manifest_written = 1
action_lifecycle_pass = 1
```

判断：v9.3.1 的 “action apply error completely missing” blocker 被真实推进为全量 measured rows。限制也必须写清楚：这不是 durable tensor payload package；payload tensors 是同轮从 train stream 重建并 replay 的，manifest 记录 hash/norm/error，而不是 `.pt` 文件。

## 4. P2 Secondary / Control Outcome

Artifacts：

```text
p2_secondary_control_outcome_materializer.csv
secondary_control_outcome_trace_v9320.csv
matched_control_outcome_trace_v9320.csv
```

Summary：

```text
secondary_control_materializer_id = SCM1-ImmediateProbePartialControls
candidate_action_count = 2876
outcome_sample_action_count = 143
official_sample_coverage = 0.04972183588317107
matched_control_count_per_event_min = 2
matched_control_count_per_event_mean = 2.0
required_matched_control_count = 4
branch_missing_count = 429
horizon_missing_count = 286
missing_secondary_delta_count = 1716
secondary_outcome_ready = 0
reason = partial_immediate_probe_only_controls_and_horizons_missing
```

判断：P2 没有过。它只 materialize 了 immediate-probe 的 `RealFunctional / AdamWOnly / NoOp` 局部对照；没有 `AdamWParallel / bestLR / Random / shuffled` controls，也没有 20/80/240 horizons。因此不能声明 useful/control oracle，也不能 paired replay。

## 5. P3/P4 Oracle 与 Action Features

P3：

```text
oracle_id = OR1-PrimarySafeGoodOracleRetained
accepted_count = 494
precision = 1.0
coverage = 0.05445326278659612
bad_event = 0.0
null_rate = 0.0
primary_oracle_pass = 1
control_oracle_pass = 0
reason = control_oracle_blocked_by_secondary_control_materializer
```

P4：

```text
action_conditioned_feature_count = 7
best_feature_id = AVF1-PayloadNorm
best_feature_top273_safe_good = 155
best_feature_top273_bad_event = 20
best_feature_top273_null_event = 81
action_conditioned_feature_pass = 0
```

判断：primary oracle 仍说明 candidate/action population 里有 safe-good frontier；但 action-conditioned feature 没找到 official bad-tail separation，且 control oracle 因 P2 blocked。

## 6. P7 Event Runtime

Artifacts：

```text
p7_measured_online_event_runtime.csv
runtime_event_scheduler_trace_v9320.csv
runtime_empty_event_semantics_trace_v9320.csv
runtime_component_trace_v9320.csv
```

Summary：

```text
runtime_candidate_id = RT2-ActionReplayEventSchedulerMeasuredPartial
step_count = 8064
active_step_count = 1412
zero_candidate_step_count = 6652
zero_candidate_controller_kernel_count = 0
zero_candidate_controller_sync_count = 0
controller_launches_per_active_step_q90 = 3.0
controller_syncs_per_active_step_q90 = 1.0
step_ratio_q90 = 5.697758752709664
runtime_measured = 1
diagnostic_derived_from_measured_components = 0
official_event_runtime = 0
event_driven_runtime_pass = 0
```

判断：zero-candidate skip semantics 在 measured scheduler trace 中成立，但 active step 没有 fused 到计划要求的 `<=2` launches，且 action replay materializer path 很慢，step ratio q90 达 `5.697759`。不能把它写成 event-driven runtime closure。

## 7. P8 / Downstream Boundary

P8：

```text
controller_id = not_selected_secondary_control_blocked
action_apply_pass = 1
secondary_control_outcome_ready = 0
control_oracle_pass = 0
action_conditioned_feature_pass = 0
event_driven_runtime_pass = 0
official_eligible = 0
system_legal_controller_pass = 0
reason = secondary_control_outcome_materialization_incomplete
```

P9-P12 均以 `not_run` 落盘，reason = `P8_system_controller_not_official`。没有把 P1 action apply pass、P2 partial probe、P3 primary oracle 或 P7 measured scheduler trace 写成 official downstream success。

## 8. No-fake Audit

```text
rows_checked = 28261
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

Contract audit：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
train_stream_probe = 1
action_apply_materializer_used = 1
action_lifecycle_pass = 1
secondary_control_outcome_ready = 0
primary_oracle_pass = 1
control_oracle_pass = 0
action_conditioned_feature_pass = 0
event_driven_runtime_pass = 0
system_legal_controller_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection/source_gap/formula_proxy = 0/0/0
diagnostic_promoted_to_official = 0
```

## 9. Hash

| artifact | SHA256 |
|---|---|
| plan | `9663806365fbaaa277d203dee4fb6a90df7889d03acbe059aa8516b41ae0c059` |
| runner | `5736ae865eacd808d568694aeb9520c19b295a686ca85a8a580b515715ad135a` |
| run manifest | `61ba9a4d2bcca9daff31124f820ba9bda006e2dfd80be07f3c5d69bac50e601d` |
| route | `47021d4c334c864a83a7a3b0a41a245a524d07aad935ce61acb79c3a2278e38f` |
| aggregate decision | `47021d4c334c864a83a7a3b0a41a245a524d07aad935ce61acb79c3a2278e38f` |
| contract audit | `fd04517192e4e13668e3c48e9dfd8c46d757a443c495f79c1c3745cbe945d3a9` |
| provenance audit | `7818ada73c6726891e4182c13eff9b0380de4df72e0e35b393f1831a284c5fe1` |
| P1 action apply | `564b75a538723d74452eb442aafb1b0ad710af221191dbd51cb14b914400caea` |
| action apply trace | `6e0caa4aa814a9b3f60878001ee698e1ceb4b3e20d752e9ec8160d6603ce97b6` |
| P2 secondary control | `2df5304140a2746d86f3bec0df5262e0d2b4137b6e3beee6f3467ac33207b7e9` |
| secondary control trace | `e32b68f6ad19b4aab154334a48b8f5167f5f607d376eec7da8cc26d7db518608` |
| P3 oracle | `70a28def7be0cd424c937d9b61b40eff948d77517ee796b9eefe1b8548be7dde` |
| P4 features | `6032d71316306c40f8b0052486a95d170e9aafe4aad89c9624a501095e1a5423` |
| P7 runtime | `2589e03c77fbb5ce43a01c484e12df9d03601f8b1c5ed57555631aaf23805f87` |
| P8 system | `332a387654b1b9f91f6fec1c4c010f6d71a4703a1f5124004303852b9b1004a5` |
| failure table | `7109c4fbdae71a43a72fb927658ce6e180303699e342fb2366560737acc31e88` |

## 10. 最终分析结论

v9.3.2 的真实推进是：

```text
v9.3.1:
  action apply error、secondary/control outcome、measured runtime 都是 missing/blocker。

v9.3.2:
  全量 action apply error 首次真实 materialize 并数值闭合；
  但 secondary/control 只做到 partial immediate-probe sample；
  useful/control oracle、action-value controller 和 event runtime 均不能 official。
```

机制判断：

1. H1 部分成立：action apply error missing blocker 被清掉，`2876 / 2876` replay success；但 durable tensor payload artifact 没写出，external-ready 仍不能打开。
2. H2 未闭合：primary safe-good oracle 仍存在，但 matched controls/horizons 不足，不能声明 useful/control oracle。
3. H3 未成立：action-conditioned feature diagnostic 没有形成 bad-tail official separation，top273 bad still `20 > 13`，safe only `155 < 205`。
4. H4 未成立：measured event scheduler 有 zero-candidate skip，但 active-step launch q90 = `3.0`，step ratio q90 = `5.697759`，不是 runtime closure。
5. H5 成立：没有把 partial control sample、primary oracle、feature diagnostic 或 measured scheduler trace 提升成 official controller/downstream。

最终一句话：

> v9.3.2 真实执行后停在 `R4-SecondaryControlOutcomeMaterializationFail`：action apply error 已全量闭合，这是本轮真实推进；但 full secondary/control outcomes 和 horizons 仍未 materialize，event runtime 也未闭合，strict PureKAN functional 仍不能转正。
