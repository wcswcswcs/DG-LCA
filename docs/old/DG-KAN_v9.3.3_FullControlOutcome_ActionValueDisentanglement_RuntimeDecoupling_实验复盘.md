# DG-KAN v9.3.3 Full Control Outcome / Action-Value Disentanglement 与 Runtime Decoupling 实验复盘

> 本复盘记录 `DG-KAN_v9.3.3_FullControlOutcome_ActionValueDisentanglement_RuntimeDecoupling_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 durable payload pass、control dry-run、action-value diagnostic 或 runtime decoupling diagnostic 写成 official system pass。

## 0. 最新结论

```text
route = R4-ControlOutcomeMaterializationFail
base_candidate = LQ-t2-h256
success_v9330_strict_purekan_functional = False
success_v9330_full_functional = False
success_v9330_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z/
```

核心结论：

1. P0 复现 v9.3.2 boundary：source route = `R4-SecondaryControlOutcomeMaterializationFail`，candidate/action/event = `2876 / 2876 / 24192`。
2. P1 durable action payload package 首次闭合：payload shard count = `45`，payload tensor file written = `1`，disk replay success = `2876 / 2876`。
3. P1 disk replay 数值闭合：payload hash match rate = `1.0`，disk replay L∞ max = `0.0`，relative max = `0.0`，cosine min = `0.9999999999999968`。
4. P2 只完成真实 dry-run control materialization：expected branch-horizon rows = `51768`，actual rows = `48`，official coverage = `0.0027816411682892906`。
5. P2 不过 official：matched control min = `5`，但 branch missing = `17208`，horizon missing = `51720`，missing secondary delta = `192`，secondary outcome ready = `0`。
6. P3 control-positive oracle 只能作为 dry-run diagnostic：accepted = `1`，coverage = `0.00011022927689594356`，control oracle pass = `0`，原因是 P2 incomplete。
7. P4 action-value observability 未闭合：best feature group = `F1-PayloadNorm`，best AUC control-positive = `0.5`，action value observability pass = `0`。
8. P6 已把 online runtime 与 offline materializer 区分开：online/offline conflated = `0`，offline rows/sec = `0.45211140000559824`；但 online runtime 未重新测量，online runtime pass = `0`。
9. P7 system controller 未打开：official eligible = `0`，system legal controller pass = `0`，reason = `full_control_outcome_materialization_incomplete`。
10. 当前 blocker：`full_control_outcome_materialization_incomplete`；下一步必须把 full control outcome materializer 扩展到 official coverage 与完整 horizons，而不是拿 8-action dry-run 转正。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9330_full_control_outcome_action_value_runtime_decoupling.py` | v9.3.3 runner；读取 v9.3.2 artifacts，写 durable payload package，执行 control dry-run、control oracle/action-value diagnostic、runtime decoupling 与 system gate |

代码检查：

```text
python -m py_compile experiments/run_v9330_full_control_outcome_action_value_runtime_decoupling.py
```

正式运行：

```bash
python experiments/run_v9330_full_control_outcome_action_value_runtime_decoupling.py \
  --out-dir results/real_rerun_20260506/v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z \
  --fresh --device auto --data-root data --seed 1314
```

运行结果：

```json
{
  "durable_payload_pass": 1,
  "route": "R4-ControlOutcomeMaterializationFail",
  "secondary_outcome_ready": 0
}
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R4-ControlOutcomeMaterializationFail",
  "source_route_v9320": "R4-SecondaryControlOutcomeMaterializationFail",
  "candidate_count": 2876,
  "action_count": 2876,
  "event_count": 24192,
  "durable_payload_pass": 1,
  "payload_tensor_file_written": 1,
  "disk_replay_success_count": 2876,
  "disk_replay_error_linf_max": 0.0,
  "disk_replay_cosine_logged_applied_min": 0.9999999999999968,
  "secondary_outcome_ready": 0,
  "official_candidate_coverage": 0.0027816411682892906,
  "matched_control_count_per_event_min": 5,
  "branch_missing_count": 17208,
  "horizon_missing_count": 51720,
  "missing_secondary_delta_count": 192,
  "control_oracle_pass": 0,
  "action_value_observability_pass": 0,
  "online_offline_runtime_conflated": 0,
  "online_runtime_pass": 0,
  "official_eligible": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "full_control_outcome_materialization_incomplete"
}
```

判断：v9.3.3 真实推进的是 P1 durable payload package；P2 仍没有达到 full control outcome official materialization，因此后续 oracle/controller/system 必须 gate-block。

## 3. P1 durable action payload package

Artifacts：

```text
p1_durable_action_payload_package.csv
action_payload_disk_replay_trace_v9330.csv
action_payload_manifest_v9330.json
action_payload_shards_v9330/
```

Summary：

```text
payload_package_id = DPP1-ShardedTorchPayloadDiskReplay
action_count = 2876
payload_shard_count = 45
payload_tensor_file_written = 1
payload_tensor_missing_count = 0
payload_hash_missing_count = 0
payload_hash_match_rate = 1.0
disk_replay_success_count = 2876
disk_replay_error_linf_max = 0.0
disk_replay_error_relative_max = 0.0
disk_replay_cosine_logged_applied_min = 0.9999999999999968
durable_payload_pass = 1
action_lifecycle_pass = 1
```

判断：v9.3.2 的限制“没有 durable tensor payload package”在本轮被真实推进。payload 从磁盘 shard replay，hash 与数值均闭合。

## 4. P2 full matched control outcome materializer

Artifacts：

```text
p2_full_matched_control_outcome_materializer.csv
full_control_outcome_table_v9330.csv
branch_horizon_completion_trace_v9330.csv
matched_control_outcome_trace_v9330.csv
```

Summary：

```text
materializer_id = FCM1-RealBranchHorizonDryRun
action_count = 2876
branch_horizon_row_count_expected = 51768
branch_horizon_row_count_actual = 48
official_candidate_coverage = 0.0027816411682892906
matched_control_count_per_event_min = 5
matched_control_count_per_event_mean = 5.0
branch_missing_count = 17208
horizon_missing_count = 51720
missing_secondary_delta_count = 192
branch_completion_rate = 0.0027816411682892906
horizon_completion_rate = 0.3333333333333333
secondary_outcome_ready = 0
control_oracle_ready = 0
reason = dry_run_coverage_or_horizon_or_secondary_fields_incomplete
```

判断：P2 执行了真实 branch/horizon dry-run，不是 fake/proxy；但只覆盖 `8 / 2876` actions 且只跑了 horizon `20`，没有覆盖完整 `20/80/240` horizons。因此不能写成 full matched control outcome materialization。

## 5. P3/P4 control oracle 与 action-value observability

P3：

```text
oracle_id = OR1-ControlPositiveDryRun
accepted_count = 1
coverage = 0.00011022927689594356
precision_primary = 1.0
bad_event_rate = 0.0
null_rate = 0.0
value_mean = 0.40119545347988605
control_oracle_pass = 0
reason = control_oracle_blocked_by_incomplete_p2_materializer
```

P4：

```text
action_value_observability_pass = 0
best_feature_group = F1-PayloadNorm
best_auc_control_positive = 0.5
feature_cost_pass = 1
reason = observability_official_blocked_by_incomplete_control_outcomes
```

判断：P3/P4 只能作为 dry-run diagnostic。control outcomes 不完整时，不能声明 useful/control oracle，也不能选 official action-value controller。

## 6. P6/P7 runtime 与 system boundary

P6：

```text
online_offline_runtime_conflated = 0
offline_materializer_runtime_recorded = 1
offline_branch_horizon_rows_per_sec = 0.45211140000559824
runtime_candidate_id = RT1-OnlineNoControlSeparatedFromOfflineMaterializer
runtime_mode = online_sequential_official_runtime
online_runtime_pass = 0
primary_blocker = online_runtime_not_remeasured_after_decoupling
```

P7：

```text
system_candidate_id = SYS-v9330-action-value-runtime-decoupled
controller_id = not_selected_control_outcome_blocked
action_lifecycle_pass = 1
secondary_outcome_ready = 0
control_oracle_pass = 0
decision_gate_pass = 0
runtime_gate_pass = 0
payload_binding_pass = 1
official_eligible = 0
system_legal_controller_pass = 0
reason = full_control_outcome_materialization_incomplete
```

判断：本轮没有把 offline materializer 计入 online runtime，也没有把 runtime decoupling diagnostic 写成 pass。P7 正确停在 P2 blocker。

## 7. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p8_diagnostic_paired_replay_scout.csv` | `P7_system_controller_not_official` |
| `p9_leave_dataset_stratum_out.csv` | same |
| `p10_official_paired_replay.csv` | same |
| `p11_short_full_sampleeff_continual_robustness.csv` | same |

没有把 durable payload pass、control dry-run、control oracle dry-run 或 action-value diagnostic 写成 downstream success。

## 8. No-fake audit

```text
rows_checked = 3034
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
durable_payload_pass = 1
action_lifecycle_pass = 1
secondary_outcome_ready = 0
control_oracle_pass = 0
action_value_observability_pass = 0
online_runtime_pass = 0
system_legal_controller_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection/source_gap/formula_proxy = 0/0/0
diagnostic_promoted_to_official = 0
```

## 9. Hash

| artifact | SHA256 |
|---|---|
| plan | `b9060983d304265501de60e70f084b32326205ac7d56d984c58df3f903a2277b` |
| runner | `da61a15e056cc2608717af6a15b725982671da31cee1c923aa8b1ae97f244911` |
| run manifest | `9d5294a3175447234c1e6aaad9e36e8afb3f252f6bbc7ca498e7863377bdbc9a` |
| route | `b2a189d7b1375b5a315ad7e03addb49a70e5031e9766f92adf88719ce699746e` |
| aggregate decision | `b2a189d7b1375b5a315ad7e03addb49a70e5031e9766f92adf88719ce699746e` |
| contract audit | `0fb764e6c6e7586f741bce81550260461bbb1d8339f47fd00bee8452453abf1c` |
| provenance audit | `a4274779fe4df3ee7501a3ed243111150bdc0e6c63c653210e66cc0133ef1f12` |
| P0 boundary | `f900e98f3578c1c453bdb9021c58dc4fd3b82f153be0723c7b4eae6cfa10f751` |
| P1 durable payload | `57f6f2792fa44379e8e9f52e7478a86b596aa55ead20436668556e002f613d60` |
| action payload trace | `4b112d130cf58273cbc7449cc1321fc0742cdd4e6c0cdea41e5a87d2c446f672` |
| P2 control outcome | `0c0408e9608c8484293b1de6160ec28e396fd4c0b412bb9995cf0af2f75fad3f` |
| full control table | `134141be33e5eb45d467edcf026118124d7cadf55f574eee1f1102fa906e36c0` |
| P3 control oracle | `7158dc6ed623d9b6d2d1b4e4eff86b1fd6958485b1af69c58aaf9da3b62b97ce` |
| P4 observability | `f704214c82c9d9c0b484e8e805dcf2a3daab7ec81afdcdb0d85b490086e97807` |
| P6 runtime decoupling | `ad63b7da21968c81a409a727d95dd35366fc13b5d39b7df21ca6553370a68b4f` |
| P7 system controller | `1b44ff7bded945ead4428e2ae50c7386532c7b8a723cde09f6aef213265f1277` |
| failure table | `e8f64df27f37f582931f0fb131efa35d34abf26dc9923c5f412b8134a8335538` |

## 10. 最终分析结论

v9.3.3 的真实推进是：

```text
v9.3.2:
  action apply error 已全量闭合；
  但 durable payload package 缺失，secondary/control outcomes 只做到 partial immediate probe。

v9.3.3:
  durable action payload package 与 disk replay 数值闭合；
  control outcome materializer 进入真实 branch/horizon dry-run；
  但 full official coverage、完整 horizons 和 secondary deltas 仍未 materialize。
```

机制判断：

1. H1 成立：durable payload package 已落盘，disk replay hash/error/cosine 全部闭合。
2. H2 未成立：full matched controls 未闭合；当前只有 `48 / 51768` branch-horizon rows。
3. H3 未打开：control oracle 受 P2 blocker 限制，只能 dry-run diagnostic。
4. H4 未成立：action-value feature 在当前 dry-run labels 下没有形成可用 signal，best AUC = `0.5`。
5. H5 部分成立：offline materializer 与 online runtime 已分账；但 online runtime 未重新测量，不能 runtime pass。
6. H6 未打开：P7 未 pass，因此 P8-P11 全部 gate-blocked。

最终一句话：

> v9.3.3 真实执行后停在 `R4-ControlOutcomeMaterializationFail`：durable action payload package 已经闭合，这是本轮真实推进；但 full control outcome materializer 仍只有低覆盖 dry-run，control oracle/action-value controller/system runtime 都不能转正，strict PureKAN functional 仍未成功。
