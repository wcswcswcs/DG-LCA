# DG-KAN v9.3.5 ControlPositive Frontier Completion / Legal Probe / ActionDensity Runtime Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.3.5_ControlPositiveFrontierCompletion_LegalProbe_ActionDensityRuntimeClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 full oracle pass、legal probe diagnostic 或 oracle-mask runtime upperbound 写成 official system pass。

## 0. 最新结论

```text
route = R8-StaticObservabilityFail
base_candidate = LQ-t2-h256
success_v9350_strict_purekan_functional = False
success_v9350_full_functional = False
success_v9350_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9350_control_positive_frontier_completion_legal_probe_runtime_first_20260514T023000Z/
```

核心结论：

1. P0 重新分析 v9.3.4 panel：raw coverage = `0.01708553791887125`，但 uniform scaled coverage = `0.0568726933503168`，class = `R5b-FullOracleUnresolved_panel_raw_below_scaled_above_gate`。
2. P1 full control outcome completion 真实完成：actions = `2876 / 2876`，branch-horizon rows = `51768 / 51768`，missing branch/horizon/secondary = `0 / 0 / 0`。
3. P1 本轮为了同轮一致性做 full rerun，`row_count_reused_from_v9340 = 0`，`row_count_newly_materialized = 51768`；rows/sec = `24.308406908953682`，wallclock = `2479.474467053078 sec`。
4. P1 quality pass：label exclusivity、duplicate row id、branch identity、horizon hash、metric NaN/Inf violation 均为 `0`。
5. P2 full weak control-positive oracle 强通过：accepted = `893`，coverage = `0.0984347442680776`，coverage LCB = `0.09247342078415305`，bad/null = `0.0 / 0.0`，V_ctrl LCB = `0.2833107634852314`，beats AdamWParallel/bestLR/NoOp/Random 全部 `1.0`。
6. P2 证明 v9.3.4 的 `ControlPositiveOracleAbsent` 过强：full universe 中 control-positive frontier 存在且 coverage 足够。
7. P3 panel representativeness 未过：panel raw coverage 与 full coverage 发生 route flip，weighted coverage error = `0.050333370020627724`，panel CP macro error = `0.09310090256697677`。
8. P4 static legal observability 未过：best static feature = `F4-StateNLL` / `C_state_tail`，AUC_CP = `0.6151712728688206`，PR lift = `1.2117767597947864`。
9. P5 legal pre-commit probe 已真实测量 `864` actions，但未过：best probe = `PR5-ProbeReliability`，AUC_CP = `0.5030687962043473`，PR lift = `1.080332409972299`，probe cost q90 = `2.4969042278826237 ms`。
10. P8 payload-apply runtime 是 oracle-mask diagnostic upperbound，不是 official controller runtime：accepted = `61`，payload apply error L∞ = `0.0`，cosine min = `0.9999999996481125`，step ratio q90 = `2.068736718507292 > 1.50`。
11. P9 system controller 未打开：`official_eligible = 0`，`system_legal_controller_pass = 0`，primary blocker = `legal_observability_gap`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9350_control_positive_frontier_completion_legal_probe_runtime.py` | v9.3.5 runner；完成 full control outcome universe，重算 full control-positive oracle、panel representativeness、static/probe observability、payload-apply runtime diagnostic 与 system gate |

代码检查：

```text
python -m py_compile experiments/run_v9350_control_positive_frontier_completion_legal_probe_runtime.py
```

正式运行：

```bash
python experiments/run_v9350_control_positive_frontier_completion_legal_probe_runtime.py \
  --out-dir results/real_rerun_20260506/v9350_control_positive_frontier_completion_legal_probe_runtime_first_20260514T023000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --complete-actions 2876 --probe-actions 864 --online-runtime-steps 192
```

运行结果：

```json
{
  "coverage_full": 0.0984347442680776,
  "full_rows": 51768,
  "out_dir": "results/real_rerun_20260506/v9350_control_positive_frontier_completion_legal_probe_runtime_first_20260514T023000Z",
  "route": "R8-StaticObservabilityFail"
}
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R8-StaticObservabilityFail",
  "full_control_outcome_ready": 1,
  "full_control_rows_expected": 51768,
  "full_control_rows_actual": 51768,
  "rows_per_sec_total": 24.308406908953682,
  "coverage_raw_v9340": 0.01708553791887125,
  "coverage_scaled_uniform_v9340": 0.0568726933503168,
  "coverage_full_v9350": 0.0984347442680776,
  "control_positive_oracle_pass": 1,
  "control_positive_oracle_accepted_count": 893,
  "precision_control_positive": 1.0,
  "bad_event_rate": 0.0,
  "null_rate": 0.0,
  "V_ctrl_mean": 0.30498170818479614,
  "V_ctrl_lcb": 0.2833107634852314,
  "static_observability_pass": 0,
  "best_static_feature_group": "C_state_tail",
  "best_static_auc_CP": 0.6151712728688206,
  "probe_observability_pass": 0,
  "best_probe_id": "PR5-ProbeReliability",
  "best_probe_auc_CP": 0.5030687962043473,
  "payload_apply_runtime_pass": 0,
  "step_ratio_q90": 2.068736718507292,
  "system_legal_controller_pass": 0,
  "primary_blocker": "legal_observability_gap"
}
```

判断：v9.3.5 成功把 v9.3.4 的 oracle uncertainty 消掉。现在不是 control-positive frontier 缺失，而是 legal commit-time observability 不足。

## 3. P0 v9.3.4 independent reanalysis

Artifact：

```text
p0_v9340_independent_reanalysis.csv
```

Summary：

```text
panel_action_count = 864
candidate_count_full = 2876
panel_fraction = 0.3004172461752434
control_positive_accepted_panel = 155
coverage_raw = 0.01708553791887125
accepted_scaled_uniform = 515.949074074074
coverage_scaled_uniform = 0.0568726933503168
coverage_scaled_uniform_CI = [0.04876195737099799, 0.06498342932963562]
raw_accepted_shortfall = 118
full_oracle_uncertainty_class = R5b-FullOracleUnresolved_panel_raw_below_scaled_above_gate
```

判断：P0 pass。v9.3.4 的 raw panel coverage 不能作为 full oracle absence 证据。

## 4. P1 full control outcome completion

Artifacts：

```text
p1_full_control_outcome_completion.csv
full_control_outcome_table_v9350.csv
branch_horizon_completion_trace_v9350.csv
materializer_worker_trace_v9350.csv
materializer_retry_manifest_v9350.csv
p1_full_control_outcome_quality_audit.csv
```

Summary：

```text
materializer_id = MAT3-FullUniverseCheckpointReuseSequential
action_count_expected/completed = 2876 / 2876
row_count_expected/actual = 51768 / 51768
row_count_reused_from_v9340 = 0
row_count_newly_materialized = 51768
row_count_failed/retried = 0 / 0
rows_per_sec_total = 24.308406908953682
wallclock_sec = 2479.474467053078
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1.0
missing_branch/horizon/secondary = 0 / 0 / 0
quality_audit_pass = 1
full_control_outcome_ready = 1
strong_pass = 1
```

判断：P1 pass。full universe 已真实 materialize；没有用 v9.3.4 panel rows 填充，也没有 proxy rows。

## 5. P2 full control-positive oracle

Artifacts：

```text
p2_full_control_positive_oracle.csv
control_positive_oracle_trace_v9350.csv
```

Full weak oracle：

```text
oracle_id = OR1-FullWeakControlPositiveOracle
accepted_count = 893
coverage = 0.0984347442680776
coverage_lcb = 0.09247342078415305
precision_primary = 1.0
precision_control_positive = 1.0
bad_event_rate = 0.0
null_rate = 0.0
V_ctrl_mean = 0.30498170818479614
V_ctrl_lcb = 0.2833107634852314
V_ctrl_p10 = 0.026876037940382957
beats_adamwparallel/bestlr/noop/random = 1.0 / 1.0 / 1.0 / 1.0
support_balance_pass = 1
full_oracle_strong_pass = 1
full_oracle_weak_pass = 1
```

Other oracle diagnostics：

```text
OR2-FullStrongControlPositiveOracle:
  accepted_count = 146
  coverage = 0.01609347442680776
  sparse_high_quality_frontier = 1

OR3-HorizonRobustControlPositiveOracle:
  accepted_count = 13
  coverage = 0.0014329805996472662
  support_balance_pass = 0
```

判断：P2 full weak oracle pass。AP0 current action primitive 确实产生足够多的 clean control-positive rows；full horizon-robust frontier 仍稀疏，但这不是本轮 route blocker。

## 6. P3 panel representativeness

Artifacts：

```text
p3_panel_representativeness_audit.csv
panel_full_comparison_trace_v9350.csv
```

Summary：

```text
coverage_raw_panel = 0.01708553791887125
coverage_full = 0.0984347442680776
weighted_coverage_estimate = 0.04810137424744988
weighted_coverage_error = 0.050333370020627724
panel_CP_rate_error_macro = 0.09310090256697677
max_stratum_coverage_gap = 0.6995827538247565
route_flip_panel_vs_full = 1
panel_representativeness_pass = 0
```

判断：P3 fail。v9.3.4 30% panel 可以作为 smoke panel，但不能作为 future official-minimum conclusion panel；它对 CP coverage 明显低估。

## 7. P4 static action-value observability

Artifacts：

```text
p4_static_action_value_observability.csv
static_feature_trace_v9350.csv
```

Feature summary：

| feature | group | AUC_CP |
|---|---|---:|
| `F0-v9340-PayloadNorm` | `A_static_payload` | `0.4113748133222408` |
| `F1-PayloadLinf` | `A_static_payload` | `0.41382102341390686` |
| `F2-StateTailCEp99` | `C_state_tail` | `0.6067577348832358` |
| `F3-StateMarginP10` | `C_state_tail` | `0.6108385679430719` |
| `F4-StateNLL` | `C_state_tail` | `0.6151712728688206` |
| `F5-FamilySupportCount` | `E_support_calibration` | `0.42173526994812455` |

Summary：

```text
best_static_feature_group = C_state_tail
best_static_feature_id = F4-StateNLL
best_static_auc_CP = 0.6151712728688206
best_static_pr_lift = 1.2117767597947864
static_observability_pass = 0
static_observability_weak_pass = 0
```

判断：P4 fail。full labels 证明 payload norm/linf 方向更弱，state-tail static features 有一点信号但不到 `0.65` weak pass。

## 8. P5 legal pre-commit probe

Artifacts：

```text
p5_legal_precommit_probe.csv
probe_feature_trace_v9350.csv
probe_cost_trace_v9350.csv
```

Summary：

```text
probe_action_count = 864
best_probe_id = PR5-ProbeReliability
best_probe_auc_CP = 0.5030687962043473
best_probe_pr_lift = 1.080332409972299
probe_cost_q90 = 2.4969042278826237 ms
probe_memory_ratio = 1.0
probe_observability_pass = 0
reason = legal_probe_below_observability_or_cost_gate
```

判断：P5 measured but failed。当前 probe 不仅识别能力接近随机，成本也远高于计划 `0.15 ms` q90 gate；不能作为 legal controller feature。

## 9. P8 payload-apply runtime

Artifacts：

```text
p8_online_payload_apply_runtime.csv
online_payload_runtime_trace_v9350.csv
runtime_component_trace_v9350.csv
```

Summary：

```text
runtime_candidate_id = RT5-oracle-mask-diagnostic-payload-upperbound
runtime_mode = online_sequential_diagnostic_upperbound
official_controller_used = 0
payload_apply_used = 1
accepted_count = 61
candidate_count = 409
step_count = 192
active_step_count = 192
zero_candidate_controller_kernel/sync = 0 / 0
feature_compute_time_ms_q90 = 0.006469897925853729
score_accept_time_ms_q90 = 0.003225170075893402
payload_apply_time_ms_q90 = 0.7623820565640926
base_train_step_time_ms_q90 = 0.8033746853470802
total_step_time_ms_q90 = 1.5542390756309032
step_ratio_q90 = 2.068736718507292
payload_apply_error_linf_max = 0.0
payload_apply_cosine_min = 0.9999999996481125
payload_apply_runtime_pass = 0
```

判断：P8 是真实 payload-apply diagnostic，不是 no-payload smoke；但它使用 oracle mask diagnostic，不是 official controller，而且 step ratio 超 `1.50`，所以不能 official。

## 10. P9 / downstream boundary

P9：

```text
full_control_outcome_ready = 1
control_positive_oracle_pass = 1
legal_observability_pass = 0
decision_gate_pass = 0
payload_apply_runtime_pass = 0
system_legal_controller_pass = 0
reason = legal_observability_gap
```

P11-P14 均以 `not_run` 落盘，reason = `P9_system_controller_not_official`。没有把 full oracle pass、probe diagnostic 或 oracle-mask runtime upperbound 写成 LDO、paired replay、short/full functional success。

## 11. No-fake audit

```text
rows_checked = 67417
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
full_control_outcome_ready = 1
control_positive_oracle_pass = 1
panel_representativeness_pass = 0
static_observability_pass = 0
probe_observability_pass = 0
decision_gate_pass = 0
payload_apply_runtime_pass = 0
system_legal_controller_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
uses_test_or_validation = 0
uses_future_outcome_for_features = 0
source_measured_gap/formula_proxy = 0/0
diagnostic_promoted_to_official = 0
```

Failure table：

```text
F10_panel_representativeness_fail = 1
F17_static_feature_unobservable = 1
F18_probe_feature_unobservable = 1
F21_no_dataset_agnostic_controller = 1
F28_payload_apply_runtime_fail = 1
F32_system_integration_fail = 1
primary_blocker = legal_observability_gap
```

## 12. Hash

| artifact | SHA256 |
|---|---|
| plan | `ba1562e5e85f41ad8461adc5e11349210050dbdde2805b3541a764cc287c9b90` |
| runner | `397bc7fb7470fff25cbd4b7a5ffe11fdfa76d861e8900a570447711d884c7f4a` |
| run manifest | `615e171f23c4cc9671c147a6a54b7a4cb1a37764709e4b457f2630581a8ec816` |
| route | `1fc8eebd1da5217e944025d8f6bc2b809f2781453e351574b6e487b01f1399e3` |
| aggregate decision | `1fc8eebd1da5217e944025d8f6bc2b809f2781453e351574b6e487b01f1399e3` |
| contract audit | `69eb9b75623500c8d045dd0c1b08f308ef455d8c4c8ad040599ff249ddba6f2e` |
| provenance audit | `83826324fe9aa8ed86d77ca959e84aec125136342e5ffb250ab16472174ed522` |
| P0 reanalysis | `9c786f98b7b6cee4ff5bb090765b7873674998a8e64639ebda7fec878776da72` |
| P1 full completion | `08faa500f5cb030dfdcc049c3c812f580c0d05b7039b153c3d7f2376a9431140` |
| full control outcome table | `d3c09b7079e988dfc9bae84072408057ea24483638f27ca5c606cc74dfc1f6e0` |
| P1 quality audit | `5c1861db036edc38dfe6e1c3b0b9a9da34b18128bf753012d5303500bb92d474` |
| P2 oracle | `162bdd0901381f63f7cf83192d36acfb9bdc44c3b6d44082baf342d82a251734` |
| P3 representativeness | `7ed54d5bd9e53b6f0c744fdc20ba2af78174394851169682e539ba57947c12be` |
| P4 static observability | `716d30f7de6302fa50b47e611b793be1f982b87d316132e427677b1733ef463d` |
| P5 legal probe | `84008442c329536a8482b456ca7683a2fe36f6684e5ca2a6fe4867d168d3d9ba` |
| P8 runtime | `3f17af97ba0e0981e397e16d2dd8878c1809a03c95d79a2a0476d1232c1eae03` |
| P9 system | `0ae908c1f993c33e0be6e51f20399b8160ae9bbb01aed209894e2b02eecee9ca` |
| failure table | `8f65733a54fcb3dcb16fca6900fb77d80044aeccf4dd28a2c851ec77b233c280` |

## 13. 最终分析结论

v9.3.5 的真实推进是：

```text
v9.3.4:
  只完成 864-action / 15552-row panel；
  panel raw coverage = 0.0171，被保守标成 ControlPositiveOracleAbsent。

v9.3.5:
  完成 2876-action / 51768-row full universe；
  full weak control-positive oracle coverage = 0.0984，证明 frontier 真实存在；
  但 v9.3.4 panel 不具代表性，static/probe legal observability 均失败；
  payload-apply runtime diagnostic 真实测量但超 envelope 且无 official controller。
```

机制判断：

1. H1 成立：v9.3.4 low raw coverage 是 panel artifact；full oracle strong/weak pass。
2. H2 不作为主 blocker：frontier 不只是 sparse high-value，full weak oracle coverage 已达 `0.0984`。
3. H3 成立：static payload/state/support features 不是 action-value sufficient statistic；best AUC 只有 `0.6152`。
4. H4 未成立：本轮 legal pre-commit probe 未提升 observability，best AUC `0.5031`，且 q90 cost `2.497 ms` 过高。
5. H5 未闭合：payload apply runtime 已真实测量，但 oracle-mask diagnostic step ratio `2.0687 > 1.50`，且没有 official controller。
6. H6 成立：没有 dataset_name controller/threshold，也没有 dataset-specific route。
7. P9 未打开：不能进入 official paired replay 或 short/full functional validation。

最终一句话：

> v9.3.5 真实执行后停在 `R8-StaticObservabilityFail`：full control-positive frontier 已被证明存在且足够 dense，v9.3.4 的 oracle absence 结论被纠正；但当前 legal static features 和 pre-commit probe 都无法识别这个 frontier，payload-apply runtime 也未进 envelope，strict PureKAN functional 仍未成功。
