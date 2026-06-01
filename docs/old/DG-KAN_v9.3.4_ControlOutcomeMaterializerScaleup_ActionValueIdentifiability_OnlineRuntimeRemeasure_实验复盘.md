# DG-KAN v9.3.4 Control Outcome Materializer Scale-up / Action-Value Identifiability / Online Runtime Remeasure 实验复盘

> 本复盘记录 `DG-KAN_v9.3.4_ControlOutcomeMaterializerScaleup_ActionValueIdentifiability_OnlineRuntimeRemeasure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 official-minimum panel、low-coverage oracle 或 runtime smoke 写成 official system pass。

## 0. 最新结论

```text
route = R5-ControlPositiveOracleAbsent
base_candidate = LQ-t2-h256
success_v9340_strict_purekan_functional = False
success_v9340_full_functional = False
success_v9340_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9340_control_outcome_materializer_scaleup_action_value_runtime_remeasure_first_20260514T010000Z/
```

核心结论：

1. P0 复现 v9.3.3 boundary：source route = `R4-ControlOutcomeMaterializationFail`，v9.3.3 只有 `48 / 51768` branch-horizon rows，completion rate = `0.0009272137227630969`。
2. P1 durable payload regression audit 过 gate：audited action = `2876`，hash match rate = `1.0`，replay success = `2876`，L∞ max = `0.0`，cosine min = `0.9999999999999968`。
3. P2 materializer scale-up 不再是 8-action dry-run：本轮真实完成 `864` actions、`15552` branch-horizon rows，即 `30.04%` action/row universe。
4. P2/P3 architecture 与 panel 均过：rows/sec total = `22.08765790513577`，unresolved failed rows = `0`，branch/horizon/secondary missing 均为 `0`。
5. P3 official-minimum panel ready = `1`，但 full control outcome ready = `0`；完整 full universe 仍是 `2876 * 6 * 3 = 51768` rows。
6. P4 quality audit 过：rows checked = `15552`，label exclusivity / duplicate row id / branch identity / horizon hash / metric NaN/Inf violations 全部为 `0`。
7. P5 control-positive oracle 未过：accepted = `155`，coverage = `0.01708553791887125 < 0.03`；bad/null = `0.0 / 0.0`，beats all listed controls = `1.0`，但 coverage gate 不够。
8. P6 action-value observability 仍弱：best feature group = `F3-PayloadLoadTime`，AUC control-positive = `0.527427910974128`，observability pass = `0`。
9. P8 online runtime 已独立重测且未混入 offline materializer：step ratio q90 = `1.0266250613293726`，materializer in timed path = `0`；但只是 no-payload-apply smoke，无 official controller，runtime pass = `0`。
10. P9 system controller 未打开：official eligible = `0`，system legal controller pass = `0`，primary blocker = `control_positive_oracle_absent`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9340_control_outcome_materializer_scaleup_action_value_runtime_remeasure.py` | v9.3.4 runner；执行 payload regression audit、official-minimum control outcome panel、quality audit、control oracle、observability diagnostic、online runtime smoke 与 system gate |

代码检查：

```text
python -m py_compile experiments/run_v9340_control_outcome_materializer_scaleup_action_value_runtime_remeasure.py
```

正式运行：

```bash
python experiments/run_v9340_control_outcome_materializer_scaleup_action_value_runtime_remeasure.py \
  --out-dir results/real_rerun_20260506/v9340_control_outcome_materializer_scaleup_action_value_runtime_remeasure_first_20260514T010000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --architecture-smoke-actions 864 --panel-a-actions 864
```

运行结果：

```json
{
  "out_dir": "results/real_rerun_20260506/v9340_control_outcome_materializer_scaleup_action_value_runtime_remeasure_first_20260514T010000Z",
  "panel_a_rows": 15552,
  "route": "R5-ControlPositiveOracleAbsent",
  "rows_per_sec_total": 22.08765790513577
}
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R5-ControlPositiveOracleAbsent",
  "source_route_v9330": "R4-ControlOutcomeMaterializationFail",
  "payload_regression_pass": 1,
  "materializer_architecture_pass": 1,
  "architecture_strong_pass": 1,
  "panel_a_pass": 1,
  "official_minimum_panel_ready": 1,
  "full_control_outcome_ready": 0,
  "quality_audit_pass": 1,
  "control_oracle_pass": 0,
  "control_oracle_accepted_count": 155,
  "action_value_observability_pass": 0,
  "online_runtime_remeasured": 1,
  "online_runtime_pass": 0,
  "step_ratio_q90": 1.0266250613293726,
  "system_legal_controller_pass": 0,
  "primary_blocker": "control_positive_oracle_absent"
}
```

判断：v9.3.4 真实推进了 full-control outcome materializer 到 official-minimum panel，但 oracle coverage 不足，不能进入 controller/system promotion。

## 3. P0 boundary budget

Artifact：

```text
p0_boundary_budget_v9340.csv
```

Summary：

```text
expected_branch_horizon_rows = 51768
actual_branch_horizon_rows_v9330 = 48
full_row_completion_rate_v9330 = 0.0009272137227630969
offline_rows_per_sec_current_v9330 = 0.45211140000559824
estimated_full_materializer_wallclock_hours_current = 31.806320300310812
estimated_30pct_panel_wallclock_hours_current = 9.544107934342222
throughput_required_for_8h_full = 1.7975
throughput_required_for_4h_30pct_panel = 1.07875
```

判断：v9.3.3 的 blocker 被定量确认。按旧吞吐 full materializer 约需 `31.8h`，30% panel 约需 `9.54h`，必须做 architecture scale-up。

## 4. P1 payload regression audit

Artifacts：

```text
p1_payload_regression_audit_v9340.csv
payload_regression_trace_v9340.csv
```

Summary：

```text
payload_package_id = DPP1-ShardedTorchPayloadDiskReplay
action_count = 2876
audited_action_count = 2876
payload_hash_match_rate = 1.0
replay_success_count = 2876
disk_replay_error_linf_max = 0.0
disk_replay_error_relative_max = 0.0
disk_replay_cosine_logged_applied_min = 0.9999999999999968
action_lifecycle_pass = 1
payload_replay_wallclock_sec = 32.302248551975936
```

判断：H0 成立。durable payload package 没有回退。

## 5. P2/P3 materializer scale-up

Artifacts：

```text
p2_materializer_architecture_rebuild.csv
p3_scaled_full_control_outcome_materializer.csv
full_control_outcome_table_v9340.csv
branch_horizon_completion_trace_v9340.csv
matched_control_outcome_trace_v9340.csv
materializer_worker_trace_v9340.csv
materializer_retry_manifest_v9340.csv
```

P2 architecture summary：

```text
materializer_architecture_id = MAT2-CheckpointReuseSequentialSmoke
worker_count = 1
shard_size_actions = 864
branch_batching_enabled = 1
horizon_checkpoint_reuse_enabled = 1
checkpoint_cache_enabled = 1
payload_shard_cache_enabled = 1
retry_manifest_enabled = 1
expected_rows_per_shard = 15552
completed_rows_per_shard = 15552
failed_rows_per_shard = 0
rows_per_sec_total = 22.08765790513577
architecture_smoke_pass = 1
architecture_strong_pass = 1
```

P3 panel summary：

```text
panel_id = PANEL_A_SCALE_SMOKE
action_count_expected/completed = 864 / 864
branch_horizon_row_count_expected/actual = 15552 / 15552
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1.0
missing_branch_count = 0
missing_horizon_count = 0
missing_secondary_delta_count = 0
matched_control_count_per_event_min = 5
per_dataset_min_action_count = 288
per_horizon_min_action_count = 5184
per_branch_min_row_count = 2592
family_horizon_bucket_min_count = 6
official_minimum_panel_ready = 1
full_control_outcome_ready = 0
```

判断：H1 成立到 official-minimum panel。吞吐从 v9.3.3 的 `0.452 rows/sec` 提升到 `22.088 rows/sec`，并完成 30% panel；但 full `51768` rows 尚未完成，因此不能声明 full outcome ready。

## 6. P4 outcome quality audit

Artifact：

```text
p4_outcome_quality_consistency_audit.csv
```

Summary：

```text
rows_checked = 15552
label_exclusivity_violation_count = 0
duplicate_control_outcome_row_id_count = 0
branch_identity_violation_count = 0
horizon_state_hash_mismatch_count = 0
control_branch_missing_count = 0
metric_nan_count = 0
metric_inf_count = 0
quality_audit_pass = 1
```

判断：official-minimum panel 的 row quality 可以用于 oracle diagnostic；没有用 fake/proxy 补 secondary fields。

## 7. P5 control-positive oracle

Artifacts：

```text
p5_control_positive_oracle.csv
control_positive_oracle_trace_v9340.csv
```

Summary：

```text
oracle_id = OR1-ControlPositivePanelOracle
candidate_universe = official_minimum_panel
accepted_count = 155
coverage = 0.01708553791887125
precision_primary = 1.0
precision_control_positive = 1.0
bad_event_rate = 0.0
null_rate = 0.0
V_ctrl_mean = 0.40297435385084923
V_ctrl_median = 0.28818584233522415
beats_adamwparallel_rate = 1.0
beats_bestlr_rate = 1.0
beats_noop_rate = 1.0
beats_random_rate = 1.0
support_balance_pass = 1
control_oracle_pass = 0
reason = control_positive_oracle_coverage_or_support_gate_failed
```

判断：control-positive rows 质量很干净，但 coverage `0.0171` 低于计划 gate `0.03`。因此 oracle 不能转正，route 正确停在 `R5-ControlPositiveOracleAbsent`。

## 8. P6 action-value observability

Artifacts：

```text
p6_action_value_observability_disentanglement.csv
action_value_feature_trace_v9340.csv
feature_cost_trace_v9340.csv
```

Feature summary：

| feature | AUC_control_positive |
|---|---:|
| `F1-PayloadNorm` | `0.4637702634620514` |
| `F2-PayloadLinf` | `0.47698781969593695` |
| `F3-PayloadLoadTime` | `0.527427910974128` |

Summary：

```text
action_value_observability_measured = 1
action_value_observability_pass = 0
best_feature_group = F3-PayloadLoadTime
best_auc_control_positive = 0.527427910974128
feature_cost_pass = 1
```

判断：action-value observability 仍弱。这个结果不是最终证伪所有 legal features；它只说明当前三组轻量特征在 official-minimum panel 上不能识别 control-positive frontier。

## 9. P8 online runtime remeasurement

Artifacts：

```text
p8_online_runtime_remeasurement.csv
online_runtime_component_trace_v9340.csv
```

Summary：

```text
runtime_candidate_id = RT4-online-feature-score-accept-no-payload-apply-smoke
runtime_mode = online_sequential_official_runtime
control_outcome_materializer_in_timed_path = 0
offline_audit_in_timed_path = 0
disk_payload_lookup_in_timed_path = 0
payload_preloaded = 1
step_count = 96
active_step_count = 96
candidate_count = 199
zero_candidate_controller_kernel_count = 0
zero_candidate_controller_sync_count = 0
controller_launches_per_active_step_q90 = 0.0
controller_syncs_per_active_step_q90 = 0.0
feature_compute_time_ms_q90 = 0.007431022822856903
score_accept_time_ms_q90 = 0.0066300854086875916
base_train_step_time_ms_q90 = 0.7532583549618721
total_step_time_ms_q90 = 0.7747514173388481
step_ratio_q90 = 1.0266250613293726
memory_ratio = 1.0
online_runtime_remeasured = 1
online_runtime_pass = 0
reason = runtime_smoke_measured_but_no_official_controller_or_payload_apply
```

判断：H4 部分成立。online/offline 已分账，control materializer 不在 timed path，runtime smoke ratio 很低；但没有 official controller，也没有 payload apply path，因此不能写成 online runtime pass。

## 10. P9 / downstream boundary

P9：

```text
official_minimum_panel_ready = 1
control_oracle_pass = 0
action_value_observability_pass = 0
decision_gate_pass = 0
runtime_gate_pass = 0
payload_binding_pass = 1
action_lifecycle_pass = 1
secondary_outcome_ready = 1
official_eligible = 0
system_legal_controller_pass = 0
reason = control_positive_oracle_absent
```

Downstream artifact 均为 `not_run`：

| artifact | reason |
|---|---|
| `p7_crossfitted_action_value_controller.csv` | `P9_system_controller_not_official` |
| `p10_diagnostic_causal_scout.csv` | same |
| `p11_leave_dataset_stratum_out.csv` | same |
| `p12_official_paired_replay.csv` | same |
| `p13_short_full_sampleeff_continual_robustness.csv` | same |

没有把 official-minimum panel、clean but low-coverage oracle 或 runtime smoke 写成 official pass。

## 11. No-fake audit

```text
rows_checked = 36855
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
payload_regression_pass = 1
materializer_architecture_pass = 1
official_minimum_panel_ready = 1
full_control_outcome_ready = 0
quality_audit_pass = 1
control_oracle_pass = 0
action_value_observability_pass = 0
online_runtime_pass = 0
system_legal_controller_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
source_measured_gap/formula_proxy = 0/0
diagnostic_promoted_to_official = 0
```

## 12. Hash

| artifact | SHA256 |
|---|---|
| plan | `1b4a70df960221e1dfee7ad0bc72d01402cc9f8e74200099f189467f04385d50` |
| runner | `787fba013739fd335212d4c6a22813b3fc31d1dbaafbc3eb5545d3349b6d2267` |
| run manifest | `a2688012fcbae770ba1cacef493747e0be13651c7dfc4964082f428d02de46fa` |
| route | `0395770b7cb80bbfbc47751e520d9a2657c9bf8691fa2985efe478df90b0dd5d` |
| aggregate decision | `0395770b7cb80bbfbc47751e520d9a2657c9bf8691fa2985efe478df90b0dd5d` |
| contract audit | `6718c7281fce88d5c6f889ec20295fbc8587386f0133188676dcdd34d888be93` |
| provenance audit | `be3a16afa46026680af8e7e0548e9068a74c7141417ca5081a6dcf1070bda332` |
| P0 boundary budget | `7cced782bec84a731d7036a5504c2c3f00f0e2bc9540c54555653e0f298bdd22` |
| P1 payload regression | `41411e83102c8c0b95c301bb559df2292511afe625d9a3f867f61f74dff15730` |
| payload regression trace | `2bc0121fdce0a6fcb0daa1c1b89bda597f05dc73ecd9e3c7e7e27e96426cf9ba` |
| P2 architecture | `6dadc8d72be14c7d278905e5fc119dac8a2cf7aff513bbcaad4ac6042a2e2389` |
| P3 scaled materializer | `8b1ba6acdfdb19d0198ccaea96e90e11d19e1f49201fb133bf7171ff8f391558` |
| full control outcome table | `82621647c88575922a3df81ef9153372c34b59a4a7af851c234ac005b3da7761` |
| P4 quality audit | `bb0448841e52029e12c0797664b7e1683e469c290e7369d1675db542c3d9b1f0` |
| P5 control oracle | `d6fbee1e74deec21ef58f1411cd4885644e65285cdf76f55ad8bcc53a3a5a3fb` |
| P6 observability | `e4da925ba0c6c68a79623c625e40dcc06de8c59a963a903e230a9e82f9502a03` |
| P8 runtime | `712651668672acf756124662559961555f80e0f48b74b78a93d7d6c6604c1094` |
| P9 system | `eda271aa7e571923c6bd6e559e2a0f24e1a0306b3553c8ab03a479d5df0e1460` |
| failure table | `2ea197e024d70cf4f3f35d78005a481f6ec8c90ae650beec057b7396bdee34ed` |

## 13. 最终分析结论

v9.3.4 的真实推进是：

```text
v9.3.3:
  durable payload package 已闭合；
  control outcome 只有 8-action / 48-row dry-run。

v9.3.4:
  durable payload regression 继续闭合；
  control outcome materializer 扩展到 864-action / 15552-row official-minimum panel；
  panel quality audit 通过；
  online runtime smoke 独立重测且未混入 offline materializer；
  但 control-positive oracle coverage 不到 0.03，action-value observability 也未形成可用 signal。
```

机制判断：

1. H0 成立：payload package 仍稳定。
2. H1 成立：materializer scale-up 成功，吞吐 `22.09 rows/sec`，远超 `>=2.0` minimum 和 `>=5.0` strong gate。
3. H2 未成立：official-minimum panel 上 control-positive oracle only accepted `155`，coverage `0.0171`，未达 `0.03`。
4. H3 未成立：当前轻量 legal features 无法识别 control-positive；best AUC 只有 `0.5274`。
5. H4 部分成立：online runtime 已独立重测且 ratio `1.0266`，但没有 official controller/payload apply，不能 pass。
6. H5 成立：controller 没有使用 dataset_name，也没有 dataset-specific threshold。
7. H6 未打开：P9 未 pass，因此 P10-P13 全部 gate-blocked。

最终一句话：

> v9.3.4 真实执行后停在 `R5-ControlPositiveOracleAbsent`：full control outcome materializer 已从 48-row dry-run 推进到 15552-row official-minimum panel，但该 panel 中可用的 control-positive frontier coverage 只有 `0.0171`，不足以支撑 official controller；strict PureKAN functional 仍未成功。
