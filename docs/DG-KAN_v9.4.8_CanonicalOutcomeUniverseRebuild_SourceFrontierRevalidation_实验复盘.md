# DG-KAN v9.4.8 Canonical Outcome Universe Rebuild / Source Frontier Revalidation 实验复盘

> 本复盘记录 `DG-KAN_v9.4.8_CanonicalOutcomeUniverseRebuild_SourceFrontierRevalidation_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把旧 v9.3.5 outcome table、oracle diagnostic frontier、Base-Acc Sentinel 或未打开的 generator/controller/runtime 写成 official system pass。

## 0. 最新结论

```text
route = R4-CanonicalSourceFrontierExistsLegalOpaque
base_candidate = LQ-t2-h256
success_v9480_strict_purekan_functional = False
success_v9480_full_functional = False
success_v9480_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_recovery_20260514T160000Z/
```

第一次 full run artifact 保留为失败边界：

```text
results/real_rerun_20260506/v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_first_20260514T150000Z/
```

核心结论：

1. 第一次 full rebuild 没有崩溃，但 `169` 个 action 触发 CUDA `OutOfMemoryError`，只完成 `48726 / 51768` rows，route = `R0-CanonicalOutcomeTableIncomplete`。
2. 随后执行 recovery run：复用第一次已真实落盘的 `2707` 个完整 action / `48726` rows，只对缺失 `169` actions 开启逐 action 清缓存重跑。
3. Recovery 后 P2 full canonical AP0 outcome universe 闭合：actions = `2876 / 2876`，rows = `51768 / 51768`。
4. Recovery 新 materialized rows = `3042`，正好补齐 `169 * 6 * 3`；retry manifest rows = `0`。
5. P2 quality audit 过：missing hash、duplicate row id、NaN/Inf、label exclusivity、old-table quarantine violation 全部为 `0`。
6. P0/P1 继续过：旧 v9.3.5 outcome table quarantine enforced = `1`，canonical replay preflight / no-transform equivalence = `1`。
7. P3 old-vs-canonical drift taxonomy 完成：comparison rows = `8628`，old/new label match rate = `0.5562123319425127`，V_ctrl abs diff max = `10.772544227307662`，旧表 quarantine 继续成立。
8. P4 canonical weak CP oracle 过：accepted = `1279`，coverage = `0.14823829392675011`，V_ctrl LCB = `0.33908824425448797`。
9. P4 canonical horizon-robust oracle 过：horizon robust action = `120`，coverage = `0.04172461752433936`。
10. P5 canonical source frontier diagnostic 过：best = `SRC-ORC-YRobust-K16`，h20 weak CP = `1.0`，h20 V_ctrl LCB = `0.8638057411703248`，h240 long-risk = `0.0`。
11. P6 legal observability 未过：best feature = `NegLongRiskPayloadNorm`，AUC_Yrobust = `0.5236967827769714`，TopK64 Yrobust precision = `0.046875`。
12. P11 Base-Acc Sentinel continuation complete，复用 v9.4.7 sentinel：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
13. P7-P10/P13-P14 均 gate-blocked：没有 generator preservation、certificate、source controller、selected runtime、paired replay 或 short/full validation。
14. 当前 primary blocker 已从 `canonical_outcome_table_incomplete` 推进为 `canonical_legal_observability_failed`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9480_canonical_outcome_universe_rebuild.py` | v9.4.8 runner；读取 v9.4.7/v9.3.5/v9.3.3 artifacts，执行旧表 quarantine、canonical replay preflight、canonical AP0 outcome rebuild、old-new drift taxonomy、canonical CP/source frontier、legal observability、Base-Acc Sentinel 与 system gate |

代码检查：

```text
python -m py_compile experiments/run_v9480_canonical_outcome_universe_rebuild.py
```

第一次正式运行：

```bash
python experiments/run_v9480_canonical_outcome_universe_rebuild.py \
  --out-dir results/real_rerun_20260506/v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_first_20260514T150000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --run-base-acc --sentinel-seeds 0,1,2,3,4,5,6,7,8,9
```

第一次运行结果：

```json
{
  "out_dir": "results/real_rerun_20260506/v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_first_20260514T150000Z",
  "route": "R0-CanonicalOutcomeTableIncomplete",
  "canonical_rows": 48726,
  "canonical_control_positive_oracle_pass": 1,
  "canonical_source_frontier_pass": 1
}
```

Recovery 运行：

```bash
python experiments/run_v9480_canonical_outcome_universe_rebuild.py \
  --out-dir results/real_rerun_20260506/v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_recovery_20260514T160000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --resume-from results/real_rerun_20260506/v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_first_20260514T150000Z \
  --clear-caches-each-action --shard-size-actions 16
```

Recovery 运行结果：

```json
{
  "out_dir": "results/real_rerun_20260506/v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_recovery_20260514T160000Z",
  "route": "R4-CanonicalSourceFrontierExistsLegalOpaque",
  "canonical_rows": 51768,
  "canonical_control_positive_oracle_pass": 1,
  "canonical_source_frontier_pass": 1
}
```

说明：recovery run 没有 CPU offload，没有 proxy rows；只是复用已完成 canonical rows，并对缺失 action 使用清缓存 replay 补齐。

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R4-CanonicalSourceFrontierExistsLegalOpaque",
  "source_route_v9470": "R6a-OldOutcomeTableQuarantineRequired",
  "canonical_replay_preflight_pass": 1,
  "old_table_quarantine_enforced": 1,
  "canonical_full_control_outcome_ready": 1,
  "canonical_row_count_expected": 51768,
  "canonical_row_count_actual": 51768,
  "quality_audit_pass": 1,
  "canonical_control_positive_oracle_pass": 1,
  "canonical_weak_CP_coverage": 0.14823829392675011,
  "canonical_horizon_robust_oracle_pass": 1,
  "canonical_horizon_robust_coverage": 0.04172461752433936,
  "canonical_source_frontier_pass": 1,
  "best_source_panel_id": "SRC-ORC-YRobust-K16",
  "best_source_h20_weak_CP": 1.0,
  "best_source_h240_long_risk": 0.0,
  "legal_observability_pass": 0,
  "best_legal_feature_id": "NegLongRiskPayloadNorm",
  "best_legal_AUC_Yrobust": 0.5236967827769714,
  "base_acc_sentinel_pass": "1",
  "base_acc_used_for_controller": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "canonical_legal_observability_failed"
}
```

判断：v9.4.8 真实闭合 canonical outcome universe，并证明 canonical AP0 CP/source frontier 存在；当前不是 table incomplete，也不是 old table 问题，而是 legal commit-time observability 看不见 canonical source frontier。

## 3. P0/P1 quarantine 与 preflight

Artifacts：

```text
p0_v9470_boundary_reproduction.csv
p0_old_table_quarantine_reader_audit.csv
p1_canonical_replay_preflight.csv
p1_no_transform_sentinel_grid.csv
```

Summary：

```text
source_route_v9470 = R6a-OldOutcomeTableQuarantineRequired
old_table_read_attempt_count = 3
official_old_table_read_blocked_count = 3
diagnostic_old_table_read_count = 3
quarantine_violation_count = 0
old_table_quarantine_enforced = 1

source_clone_replay_key_pass = 1
single_action_preflight_pass = 1
canonical_runner_semantics_pass = 1
side_by_side_no_transform_replay_pass = 1
no_transform_equivalence_pass = 1
branch/horizon/optimizer/batch_sequence hash match rate = 1.0 / 1.0 / 1.0 / 1.0
metric_abs_diff_max = 0.0
label_match_rate = 1.0
negative_control_divergence_present = 1
canonical_replay_preflight_pass = 1
```

判断：P0/P1 pass。旧表只作为 diagnostic read；canonical no-transform semantics 仍闭合。

## 4. P2 canonical AP0 outcome materializer

Artifacts：

```text
p2_canonical_ap0_full_control_outcome_materializer.csv
canonical_full_control_outcome_table_v9480.csv
canonical_branch_horizon_completion_trace_v9480.csv
canonical_materializer_worker_trace_v9480.csv
canonical_materializer_retry_manifest_v9480.csv
p2_quality_audit.csv
```

Summary：

```text
materializer_id = CANMAT-v9480-canonical-branch-name-invariant
outcome_table_version = canonical_v9480
runner_semantics_version = canonical_branch_name_invariant_v9470_or_later
action_count_expected/completed = 2876 / 2876
row_count_expected/actual = 51768 / 51768
row_count_reused_from_resume = 48726
action_count_reused_from_resume = 2707
row_count_newly_materialized = 3042
row_count_failed/retried/unresolved = 0 / 0 / 0
rows_per_sec_total = 223.23899323480558
wallclock_sec = 231.89497161703184
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1
missing_branch/horizon/secondary = 0 / 0 / 0
metric_nan/inf = 0 / 0
label_exclusivity_violation_count = 0
duplicate_outcome_row_id_count = 0
quality_audit_pass = 1
canonical_full_control_outcome_ready = 1
```

Branch/horizon 分布：

```text
branch_id:
  RealFunctional = 8628
  AdamWOnly = 8628
  AdamWParallel = 8628
  bestLR = 8628
  NoOp = 8628
  Random = 8628

horizon:
  20 = 17256
  80 = 17256
  240 = 17256
```

判断：P2 pass。第一次 run 的 OOM blocker 已经通过 recovery 补齐；没有 fake/proxy rows，retry manifest rows = `0`。

## 5. P3 old-new drift taxonomy

Artifact：

```text
p3_old_new_drift_taxonomy.csv
```

Summary：

```text
comparison_row_count = 8628
old_new_label_match_rate = 0.5562123319425127
old_new_V_ctrl_abs_diff_max = 10.772544227307662
old_new_V_ctrl_abs_diff_mean = 0.8638358256578607
D0_no_drift_count = 0
D8_old_runner_materializer_bug_count = 8628
attribution_fraction = 1.0
old_table_official_quarantine_remains = 1
old_new_drift_taxonomy_complete = 1
```

判断：P3 pass。旧 v9.3.5 table 与 canonical runner 的 drift 非常大，旧表 quarantine 继续成立。

## 6. P4 canonical control-positive oracle

Artifact：

```text
p4_canonical_control_positive_oracle.csv
```

Summary：

```text
oracle_id = OR1-CanonicalFullWeakControlPositiveOracle
action_count = 2876
row_count = 8628
accepted_count = 1279
coverage = 0.14823829392675011
precision_control_positive = 1.0
bad_event_rate = 0.0
null_rate = 0.0
V_ctrl_mean = 0.3599340002608349
V_ctrl_lcb = 0.33908824425448797
V_ctrl_p10 = 0.05089855450205505
h20/h80/h240 weak CP rate = 0.14673157162726008 / 0.14812239221140472 / 0.14986091794158554
strong_CP_rate = 0.09318497913769123
horizon_robust_action_count = 120
horizon_robust_coverage = 0.04172461752433936
long_risk_action_count = 2095
long_risk_rate = 0.728442280945758
support_balance_pass = 1
canonical_control_positive_oracle_pass = 1
canonical_horizon_robust_oracle_pass = 1
```

判断：P4 pass。canonical AP0 full universe 下 control-positive frontier 和 horizon-robust frontier 都存在；这纠正了 v9.4.7 old-table quarantine 后的未知状态。

## 7. P5 canonical source frontier

Artifact：

```text
p5_canonical_source_frontier.csv
```

Best panel：

```text
best_panel_id = SRC-ORC-YRobust-K16
selector_type = oracle_diagnostic
K = 16
h20_weak_CP = 1.0
h20_strong_CP = 1.0
h20_V_ctrl_lcb = 0.8638057411703248
h80_weak_CP = 0.1875
h80_V_ctrl_lcb = -1.53319783648607
h240_weak_CP = 0.75
h240_V_ctrl_lcb = 0.07955017988543481
h240_long_risk = 0.0
Y_robust_count = 16
Y_robust_rate = 1.0
support_balance_pass = 1
canonical_source_frontier_pass = 1
```

判断：P5 pass，但它是 oracle diagnostic source frontier，不是 legal selector/controller。它证明 canonical source frontier 存在，后续 blocker 转到 legal observability。

## 8. P6 canonical legal observability

Artifact：

```text
p6_canonical_legal_observability.csv
```

Summary：

```text
feature_count = 6
best_feature_id = NegLongRiskPayloadNorm
best_AUC_Yrobust = 0.5236967827769714
best_TopK64_Yrobust_precision = 0.046875
best_TopK64_h240_longrisk = 0.546875
legal_observability_pass = 0
legal_observability_strong_pass = 0
```

Feature diagnostics：

| feature | AUC_Yrobust | TopK64 Yrobust precision | TopK64 h240 longrisk |
|---|---:|---:|---:|
| PayloadNorm | 0.47630321722302854 | 0.03125 | 0.703125 |
| PayloadLinf | 0.48404088050314464 | 0.046875 | 0.640625 |
| Step | 0.5029738147073053 | 0.03125 | 0.8125 |
| FamilySupportCount | 0.49708817126269955 | 0.03125 | 0.828125 |
| CandidateId | 0.49588171262699565 | 0.03125 | 0.609375 |
| NegLongRiskPayloadNorm | 0.5236967827769714 | 0.046875 | 0.546875 |

判断：P6 未过。canonical frontier 虽然存在，但当前 legal commit-time features 仍几乎看不见 robust source。

## 9. P7-P10 / P12-P14 boundary

这些 artifact 均按 upstream gate 落盘：

| artifact | status / reason |
|---|---|
| `p7_canonical_generator_preservation.csv` | `not_run`, `canonical_legal_observability_failed` |
| `p8_effect_valid_certificate.csv` | `not_run`, `canonical_legal_observability_failed` |
| `p9_minimal_source_controller.csv` | `not_run`, `canonical_legal_observability_failed` |
| `p10_selected_runtime.csv` | `not_run`, `canonical_legal_observability_failed` |
| `p12_system_integration_gate.csv` | official eligible = `0` |
| `p13_leaveout_paired_replay_boundary.csv` | `P12_system_controller_not_official` |
| `p14_short_full_validation_boundary.csv` | `P13_paired_replay_not_open` |

判断：没有把 oracle source frontier diagnostic 写成 generator/certificate/controller/runtime/system pass。

## 10. P11 Base-Acc Sentinel

Artifacts：

```text
p11_base_acc_sentinel_continuation.csv
base_acc_training_trace_v9480.csv
```

Summary：

```text
sentinel_row_count = 120
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
model_count = 4
sentinel_complete = 1
mean_test_acc_LQ = 0.6537760416666667
mean_test_acc_MLP = 0.562890625
mean_test_acc_QuadraticFeatureMLP = 0.40234375
mean_test_acc_AdamWStrongLRGridMLP = 0.628515625
LQ_minus_MLP_mean_test_acc = 0.09088541666666672
LQ_minus_QuadraticFeatureMLP_mean_test_acc = 0.2514322916666667
LQ_minus_AdamWStrongLRGridMLP_mean_test_acc = 0.025260416666666674
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
base_acc_sentinel_pass = 1
base_acc_reused_from_v9470 = 1
```

判断：Base-Acc Sentinel 继续健康，但它没有用于 selector/generator/controller，也不是 functional success。

## 11. No-fake audit

```text
rows_checked = 63439
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = 1
no_proxy = 1
```

Contract audit：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
train_stream_probe = 1
canonical_replay_preflight_pass = 1
canonical_full_control_outcome_ready = 1
old_table_quarantine_enforced = 1
canonical_control_positive_oracle_pass = 1
canonical_source_frontier_pass = 1
legal_observability_pass = 0
generator_preservation_pass = 0
certificate_effect_valid_pass = 0
source_controller_pass = 0
selected_runtime_pass = 0
system_legal_controller_pass = 0
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R4-CanonicalSourceFrontierExistsLegalOpaque
F0_canonical_outcome_table_incomplete = 0
F1_old_table_quarantine_violation = 0
F2_canonical_ap0_frontier_absent = 0
F3_canonical_short_horizon_only_longrisk = 0
F4_legal_observability_fail = 1
F5_generator_revalidation_blocked = 1
F6_certificate_revalidation_blocked = 1
F7_system_not_official = 1
F8_base_acc_catastrophic = 0
primary_blocker = canonical_legal_observability_failed
```

## 12. Hash

| artifact | SHA256 |
|---|---|
| plan | `50f202fc01500eebf59cdcf8c0a747d7c0dbc61765aa01afd10467069ff1b18d` |
| runner | `a6e7862a36a6a0dee6b8940cd05adf201c0b834340aeed4f46644815dea7aafa` |
| run manifest | `30b90061020d70b1d0a01166cc3f49897afedfe1b0b61a8e434a8443c6143fbf` |
| route | `521c326038b1ed89c40d0da462d8cef0fecdc963e300f75059b3e26a2b852ce1` |
| aggregate decision | `521c326038b1ed89c40d0da462d8cef0fecdc963e300f75059b3e26a2b852ce1` |
| P0 boundary | `9501b9c4609df5a147a03db549396aa992ce281393ddb7c0cd92f5e6cb04bf04` |
| P0 quarantine | `12d5e3655b4536c37e8f6535c0c39c55dc8cf76fe2fe17269311791f09765558` |
| P1 preflight | `7f21cf2b04aff3737a170469d9e4196f7f9c2bff20089a453b847b5a9ddcad49` |
| P2 materializer | `d32b6946936502f76505b25302f75bfe173d6edbdb97fddb3a4983cdb5bc6033` |
| canonical outcome table | `5f57f443aba3c1ba7568059ca3b7628e80c1c88aae1819a6dc1d597f08bd9a13` |
| P2 quality audit | `03470b7a183da265376e38ad85ca422280baa305f04c4d01f775255fd39b31aa` |
| P3 drift | `6db65937baeccdfbe6f891811cf0d2f0098ee57066b026a4633afc113602dab6` |
| P4 oracle | `ee0fc4a74066a10dd2f6a7e91ef8b9c384284d3383cbe2fa6d5fd670603ec7ab` |
| P5 source frontier | `b05693aaebe3fedaed073fe5ba23c7d1cf63f950f732d8410b2528a7084c5f23` |
| P6 legal observability | `0e2a513ad6c629094140414f2105a2520e764284feab588567e232a744d08398` |
| P11 Base-Acc Sentinel | `f18e3fea78a7549ef155bcaaa2c145bb310a9290d2d206902a95efb1b47533bd` |
| P12 system | `a25aec7d93cd0ecd2c67cefd565113d762f22e5c5a19f141078c199e05d1f10f` |
| contract audit | `3a56ecaeaad41ec19b7d0bfeecbb26db2e817d79211040a71e0c37d41f380b68` |
| provenance audit | `890eb3c2b4c5d79fc0c6569e4a713bf8ab337386c0bf278c95da23a2fc8e93f2` |
| failure table | `11a8d44501bc0ca062fabde8bad567b3727b69626f75bdaf79b3f58ecb538195` |

## 13. 最终分析结论

v9.4.8 的真实推进是：

```text
v9.4.7:
  no-transform replay semantics 已闭合；
  但旧 v9.3.5 ORC-D-K16 oracle survivor 不被 canonical runner 复现；
  旧 outcome table 必须 quarantine。

v9.4.8:
  在 canonical repaired runner 下重建 AP0 outcome universe；
  第一次 full run 因尾部 CUDA OOM 停在 48726 rows；
  recovery run 复用已完成 rows，并重跑缺失 169 actions；
  最终 canonical full table 51768 / 51768 rows 闭合；
  canonical CP/source frontier 存在；
  但 legal commit-time features 仍无法识别 robust source。
```

机制判断：

1. H0 成立：旧表 quarantine 被执行，旧 v9.3.5 table 没有用于 official gate。
2. H1 成立：canonical no-transform runner preflight 继续闭合。
3. H2 成立：canonical AP0 full universe 已完成，`51768 / 51768` rows，quality audit pass。
4. H3 成立：canonical weak CP coverage = `0.1482`，horizon-robust coverage = `0.0417`，AP0 frontier 真实存在。
5. H4 成立但只能诊断：best oracle source panel `SRC-ORC-YRobust-K16` 的 h20 weak CP = `1.0`，h240 long-risk = `0.0`。
6. H5 未成立：legal observability 仍弱，best AUC_Yrobust 只有 `0.5237`，TopK64 Yrobust precision 只有 `0.046875`。
7. H6-H8 未打开：legal observability fail 后，generator/certificate/controller/runtime/system/downstream 全部 gate-blocked。
8. Base-Acc Sentinel 继续健康，但仍是 isolated diagnostic，没有进入 controller。

最终一句话：

> v9.4.8 真实执行后停在 `R4-CanonicalSourceFrontierExistsLegalOpaque`：OOM 导致的 canonical table incomplete 已通过 recovery run 补齐；canonical AP0 full universe 下 CP/source frontier 真实存在，但当前 legal commit-time features 看不见这个 robust source frontier，strict PureKAN functional 仍未成功。
