# DG-KAN v9.3.0 Outcome-Action Primitive Reset 与 Event-Driven Runtime Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.3.0_OutcomeActionPrimitiveReset_EventDrivenRuntimeClosure_最终完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 oracle label frontier、diagnostic empty-step estimate 或旧 full-system runtime 写成 official system pass。

## 0. 最新结论

```text
route = R9-OGPFeatureFail
base_candidate = LQ-t2-h256
success_v9300_strict_purekan_functional = False
success_v9300_full_functional = False
success_v9300_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9300_outcome_action_primitive_reset_event_driven_runtime_closure_first_20260513T233000Z/
```

核心结论：

1. P0 复现 v9.2.82 boundary：candidate count = `2876`，event count = `24192`，heldout denominator = `9072`。
2. StableAccept heldout 仍失败：accepted = `468`，safe = `240`，bad = `142`，null = `57`，precision = `0.512821`，bad-event = `0.303419`。
3. coverage 下限需要 accepted >= `273`；在该 count 下 gate 要求 safe >= `205`、bad <= `13`、null <= `40`。
4. P1 canonical candidate identity freeze 过：candidate/action rows = `2876`，duplicate candidate/action/event id 均为 `0`，payload hash missing = `0`。
5. P1 action lifecycle 不能转正：source artifact 没有 action apply error measurement，`action_lifecycle_pass = 0`。
6. P2 primary labels 可用：missing primary label = `0`，label exclusivity violation = `0`；但 secondary downstream 不 ready：missing secondary delta count = `28760`，matched control count per event = `0`。
7. P3 oracle frontier 强通过：`OR1-SafeGoodOracle` heldout accepted = `494`，precision = `1.0`，coverage = `0.054453`，bad/null = `0`，support balance = `1`。
8. P3 说明 current candidate/action population 并非 oracle weak frontier absent；问题不是“候选总体无解”，而是 legal pre-commit observability/controller 找不到这个 frontier。
9. P4 legal OGP feature 没过 strict signal/cost gate：best feature = `OGP-C1-CandidateScore`，bad AUC = `0.617012`，safe AUC = `0.756979`，feature cost q90/memory 未测，`ogp_feature_pass = 0`。
10. P5 minimal controller 未过：best selected 仍是 `DR7-_risk_residual_score-thr-0.715503-n1.0-s0.05`，heldout accepted = `0`，coverage = `0.0`，decision pass = `0`。
11. StableAccept patch line exhausted：`stableaccept_patch_exhausted = 1`。
12. P7 event-driven runtime 未 materialize：reference path step_ratio q90 = `2.213009`，empty-step controller kernels = `59868`，controller launches per active step q90 = `9.0`，event-driven runtime pass = `0`。
13. P8/P10/P11/P12 全部 gate-blocked；没有把 diagnostic scout、oracle、或 runtime estimate 写成 official。
14. 当前 primary blocker：`legal_ogp_feature_signal_or_cost_fail`；下一步需要重建 legal OGP observability / action value features，而不是继续调 StableAccept threshold。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9300_outcome_action_primitive_reset_event_driven_runtime_closure.py` | v9.3.0 runner；读取 v9.2.80/v9.2.81/v9.2.82 artifacts，执行 canonical candidate/action freeze、secondary outcome audit、oracle frontier、primitive autopsy、legal OGP feature/controller audit、event-driven runtime boundary 与 system gate |

代码检查：

```text
python -m py_compile experiments/run_v9300_outcome_action_primitive_reset_event_driven_runtime_closure.py
```

正式运行：

```bash
python experiments/run_v9300_outcome_action_primitive_reset_event_driven_runtime_closure.py \
  --out-dir results/real_rerun_20260506/v9300_outcome_action_primitive_reset_event_driven_runtime_closure_first_20260513T233000Z \
  --fresh --device auto --data-root data --seed 1314
```

输入 artifact：

```text
source_v9282 = results/real_rerun_20260506/v9282_outcome_grounded_risk_support_rebuild_empty_step_free_batch_major_runtime_closure_first_20260513T220000Z
source_v9281 = results/real_rerun_20260506/v9281_outcome_grounded_stable_accept_repair_batch_major_runtime_closure_first_20260513T210000Z
source_v9280 = results/real_rerun_20260506/v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_rerun_20260513T190000Z
source_v9272 = results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_async_basis_cuda_ext_20260513T110000Z
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R9-OGPFeatureFail",
  "candidate_lifecycle_pass": 1,
  "action_lifecycle_pass": 0,
  "candidate_count": 2876,
  "action_count": 2876,
  "secondary_outcome_ready": 0,
  "missing_secondary_delta_count": 28760,
  "official_sample_coverage": 0.0,
  "matched_control_count_per_event": 0,
  "oracle_weak_frontier_pass": 1,
  "oracle_strong_frontier_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.05445326278659612,
  "oracle_bad_event": 0.0,
  "oracle_null_rate": 0.0,
  "ogp_feature_pass": 0,
  "ogp_feature_weak_pass": 1,
  "best_feature_id": "OGP-C1-CandidateScore",
  "best_feature_auc_bad": 0.6170115546218488,
  "best_feature_auc_safe": 0.7569791472566192,
  "ogp_decision_pass": 0,
  "controller_id": "DR7-_risk_residual_score-thr-0.715503-n1.0-s0.05",
  "precision_heldout": 0.0,
  "coverage_heldout": 0.0,
  "stableaccept_patch_exhausted": 1,
  "empty_step_controller_kernel_count": 59868,
  "step_ratio_q90": 2.213009156635521,
  "event_driven_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "legal_ogp_feature_signal_or_cost_fail"
}
```

判断：v9.3.0 成功把问题从 StableAccept patching 重置到 action/OGP framing；但在当前真实 artifact 下，legal OGP feature 没形成 official-grade observability，secondary/control outcomes 与 measured event-driven runtime 也未闭合。

## 3. P0 boundary reanalysis

Artifacts：

```text
p0_boundary_reanalysis.csv
```

Summary：

```text
candidate_count = 2876
event_count = 24192
heldout_denominator = 9072
StableAccept accepted/safe/bad/null = 468 / 240 / 142 / 57
precision = 0.5128205128205128
coverage = 0.051587301587301584
bad_event_rate = 0.3034188034188034
null_rate = 0.12179487179487179
precision_lcb = 0.46761511726824007
bad_event_ucb = 0.3465326908369387
min_accepted_for_coverage = 273
safe_needed_at_min_coverage = 205
bad_allowed_at_min_coverage = 13
null_allowed_at_min_coverage = 40
step_count = 8064
active_step_count = 1412
zero_candidate_step_count = 6652
candidate_count_per_active_step = 2.036827195467422
kernel_count = 72576
sync_count = 8064
step_ratio_q90 = 2.213009156635521
```

判断：P0 pass。Quantized/rank mismatch 已不是主因；当前 failure 是 decision frontier、legal feature observability 和 sparse runtime path。

## 4. P1 canonical candidate/action lifecycle freeze

Artifacts：

```text
p1_candidate_action_lifecycle_freeze.csv
frozen_candidate_action_table_v9300.csv
candidate_action_lifecycle_trace_v9300.csv
```

Summary：

```text
candidate_count = 2876
action_count = 2876
duplicate_candidate_id_count = 0
duplicate_action_id_count = 0
duplicate_event_id_count = 0
payload_hash_missing_count = 0
payload_hash_mismatch_count = 0
action_payload_missing_count = 0
candidate_schema_version = v9300
action_schema_version = v9300
candidate_lifecycle_pass = 1
action_lifecycle_pass = 0
reason = action_apply_error_not_measured_in_source_artifacts
```

判断：candidate identity 可以冻结；action identity 也可生成，但 action apply error 没有在 source artifact 中真实测量，因此不能把 action lifecycle 写成 pass。

## 5. P2 secondary outcome / matched control materializer

Artifacts：

```text
p2_secondary_outcome_control_materializer.csv
secondary_outcome_trace_v9300.csv
matched_control_outcome_trace_v9300.csv
```

Summary：

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
p2_primary_pass = 1
p2_downstream_ready_pass = 0
p2_coverage_pass = 0
p2_cost_pass = 0
```

判断：primary decision gate 可以真实评估；但 ECE/NLL/curvature/local-lipschitz/entropy/real-beats/control branches 未 materialize，不能打开 official paired replay 或 downstream readiness。

## 6. P3 oracle weak/strong frontier

Artifacts：

```text
p3_oracle_frontier_weak_strong.csv
oracle_frontier_trace_v9300.csv
```

StableAccept baseline：

```text
OR0-StableAcceptCurrent:
  accepted_count = 468
  precision = 0.5128205128205128
  coverage = 0.051587301587301584
  bad_event_rate = 0.3034188034188034
  null_rate = 0.12179487179487179
  support_balance_pass = 0
  oracle_weak_pass = 0
```

Best oracle：

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

判断：H3 成立但方向很清楚：candidate/action population 有 oracle frontier；不是 weak frontier absent。因此不应先 pivot 到 candidate generator，而应修 legal OGP observability / action value features。

## 7. P3.5 primitive insufficiency autopsy

Artifacts：

```text
p35_candidate_action_primitive_autopsy.csv
primitive_autopsy_trace_v9300.csv
```

Summary：

```text
oracle_weak_frontier_pass = 1
oracle_strong_frontier_pass = 1
autopsy_attribution_fraction = 1.0
primary_primitive_blocker = legal_observability_gap_not_oracle_population_absence
next_primitive_redesign_target = legal_ogp_observable_or_action_value_features
primitive_autopsy_pass = 1
```

判断：P3.5 没有判 current primitive population 无解；它把 blocker 定位为 legal observability gap，而不是 oracle frontier absence。

## 8. P4 legal OGP feature factory

Artifacts：

```text
p4_ogp_feature_factory.csv
ogp_feature_trace_v9300.csv
feature_cost_trace_v9300.csv
minimality_audit_trace_v9300.csv
```

Best feature：

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

判断：P4 只有 weak signal，没过 strict feature gate。safe-good AUC 接近但低于 `0.78`，bad-event AUC 远低于 `0.80`，PR lift 和 ECE 也不够；更重要的是 feature compute q90 / memory ratio 未测，不能 official。

## 9. P5 cross-fitted minimal OGP controller

Artifacts：

```text
p5_crossfitted_minimal_ogp_controller.csv
controller_frontier_trace_v9300.csv
```

Reference：

```text
DR0-v9281-DR2-reference:
  accepted_count_heldout = 273
  precision_heldout = 0.63003663003663
  coverage_heldout = 0.03009259259259259
  bad_event_heldout = 0.18315018315018314
  null_rate_heldout = 0.19413919413919414
  decision_gate_pass = 0

DR1-BitExactStableAcceptOnly:
  accepted_count_heldout = 468
  precision_heldout = 0.5128205128205128
  bad_event_heldout = 0.3034188034188034
  decision_gate_pass = 0
```

Best selected：

```text
controller_id = DR7-_risk_residual_score-thr-0.715503-n1.0-s0.05
precision_heldout = 0.0
coverage_heldout = 0.0
bad_event_heldout = 0.0
null_rate_heldout = 0.0
precision_lcb = 0.0
bad_event_ucb = 1.0
decision_gate_pass = 0
```

判断：P5 未过。StableAccept 和 registered patches 在 canonical set 上仍失败；OGP minimal controller 没有形成 heldout accepted region。

## 10. P6 decision failure autopsy v3

Artifacts：

```text
p6_decision_failure_autopsy_v3.csv
decision_failure_trace_v9300.csv
```

Summary：

```text
failed_controller_id = DR7-_risk_residual_score-thr-0.715503-n1.0-s0.05
bad_accepted_count = 142
null_accepted_count = 57
missed_safe_good_count = 254
coverage_lost_count = 273
failure_mode = DF3-legal_feature_oracle_gap
bad_accepted_attribution_fraction = 1.0
missed_safe_good_attribution_fraction = 1.0
coverage_collapse_attribution_fraction = 1.0
decision_failure_autopsy_pass = 1
```

判断：P6 pass。失败不是 oracle 不存在，而是 legal feature/controller 无法识别 safe-good oracle frontier。

## 11. P7 event-driven runtime

Artifacts：

```text
p7_online_event_driven_runtime.csv
runtime_event_scheduler_trace_v9300.csv
runtime_empty_event_semantics_trace_v9300.csv
runtime_component_trace_v9300.csv
```

Measured reference：

```text
RT0-v9.2.82-reference-fixed-per-step:
  runtime_mode = online_sequential_official_reference
  step_count = 8064
  active_step_count = 1412
  zero_candidate_step_count = 6652
  empty_step_controller_kernel_count = 59868
  empty_step_controller_sync_count = 6652
  controller_kernel_launch_count = 72576
  controller_sync_count = 8064
  controller_launches_per_active_step_mean = 9.0
  controller_launches_per_active_step_q90 = 9.0
  controller_syncs_per_active_step_mean = 1.0
  step_ratio_q90 = 2.213009156635521
  runtime_measured = 1
  online_runtime_pass = 0
```

Diagnostic only：

```text
RT1-empty-step-skip-diagnostic-only:
  empty_step_controller_kernel_count = 0
  controller_launches_per_active_step_q90 = 2.0
  runtime_measured = 0
  diagnostic_derived_from_measured_components = 1
  online_runtime_pass = 0
```

判断：P7 未过。empty-step-free scheduler 没有 measured materialization，不能把 RT1 estimate 写成 runtime pass。

## 12. P8-P13 boundary

Artifacts：

```text
p8_system_legal_controller_v9300.csv
p9_diagnostic_paired_replay_scout.csv
p10_leave_dataset_stratum_out.csv
p11_official_paired_replay.csv
p12_short_full_continual_robustness.csv
p13_candidate_action_primitive_reset.csv
diagnostic_isolation_audit_v9300.csv
```

Boundary：

```text
system_legal_controller_pass = 0
official_eligible = 0
reason = ogp_feature_or_controller_failed_secondary_not_ready_event_runtime_not_materialized
diagnostic_paired_replay_status = not_run
diagnostic_downstream_used_for_controller = 0
leave_dataset_out / paired replay / short-full = not_run
P13 primitive reset = not_triggered, reason = oracle_weak_frontier_present
```

判断：没有打开 downstream。由于 oracle weak/strong 已存在，P13 不作为本轮主路线；下一步应优先修 legal OGP features 和 measured event-driven runtime。

## 13. No-fake audit

```text
rows_checked = 2876
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
candidate_lifecycle_pass = 1
action_lifecycle_pass = 0
secondary_outcome_ready = 0
oracle_weak_frontier_pass = 1
oracle_strong_frontier_pass = 1
primitive_autopsy_pass = 1
ogp_feature_pass = 0
ogp_decision_pass = 0
event_driven_runtime_pass = 0
system_legal_controller_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection/source_gap/formula_proxy = 0/0/0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
F5_action_lifecycle_freeze_fail = 1
F10_secondary_outcome_missing = 1
F12_matched_control_missing = 1
F18_ogp_feature_unpredictive = 1
F20_ogp_feature_cost_infeasible = 1
F22_no_dataset_agnostic_ogp_decision = 1
F30_empty_step_runtime_fail = 1
F31_empty_event_semantics_fail = 1
F35_step_ratio_fail = 1
F37_runtime_measured_path_missing = 1
```

## 14. Hash

| artifact | SHA256 |
|---|---|
| plan | `afe6c62185d35fd50eafdd2ae78c9ee793e41999d9f7082eec6c8c1d939d4f08` |
| runner | `e4c4ac13bdc1b85165e4ac4b1191fc65e123c7693f18c71510203975e4832d17` |
| run manifest | `4533902391f1a1c383e24a9679a2b9e536fcb67a4e7e798afd7b5d744e39d853` |
| contract audit | `3de0530fc1bede81daee80925be5598a94083fbdcb7261eb8c4a9b71bfa468fd` |
| P0 boundary | `b91915cf67dfe5b29329ca0db6a145857c99cc2a317efc0419dc4bbd04b13155` |
| P1 lifecycle | `3a7696a5e226a8c0fc6763c6a50fc90e0d285dfd4a209bdf65b912847177ab0e` |
| frozen candidate action | `5e08b9b00638ea077d322cf889b45c31294513f3fc706495680c9df2a3042311` |
| P2 secondary | `31ab6005a5c20406be1f0149c01b481ae7e66fbb1cae96beef9ffcb6acefc2d4` |
| P3 oracle | `139eecea04b30c6ae418afb993df18456a966a7cca0d5b7ac50c0aa768450572` |
| P35 autopsy | `78c0f03590b838e6a5653377736c37efb1f3c5fa78f1a34a8ff3897af16339e3` |
| P4 features | `de2f4272aba76f45792c65b03f62e58b786d2cfd45a4124283d2ed9e21c1ae14` |
| P5 controller | `f16d887f63e93b07bbf0f60be8b2b9f873a6ff95004413a8452a8e7818588a69` |
| P6 decision autopsy | `d373955be6d5b35f73aaf8a6cd862d1a8af6de2e616cb2fc2aa446d986d13917` |
| P7 runtime | `47817af64ac6deecb741b18cba7532c53887fc91d0904055147971963d5838cc` |
| P8 system | `a9400bf8dab1e828ad69c650b317af8122a83a2d5e5075fd42eec47cfe35300d` |
| route | `7ab4682a8cc9b29d99ac50a637621fe8e57c6d814ef3d762a12edf2a83b9eaee` |
| failure table | `12f9daaa8c063ad539bfa3bb4b295831bf3132650520489a44bd605d5a00c661` |
| provenance | `99f7e2b0578bf645346463fb5710b62b295fcfb4a62d856a3511e12ad073d8e0` |

## 15. 最终分析结论

v9.3.0 的真实推进是：

```text
v9.2.82:
  StableAccept patch / risk-support threshold route 仍失败；
  runtime 仍停在 fixed per-step launch/sync。

v9.3.0:
  将 candidate/action population canonical freeze；
  用 oracle weak/strong 证明当前 population 有 safe-good frontier；
  但 legal pre-commit OGP feature/controller 找不到这个 frontier；
  secondary matched-control outcomes 与 event-driven runtime 都没有 materialize。
```

机制判断：

1. H0 成立：StableAccept final-controller 路线已耗尽，StableAccept-only 与 registered patches 仍不过 decision gate。
2. H1 部分成立：candidate identity freeze 已过，但 action apply measurement 缺失，action lifecycle 不能 official。
3. H2 未闭合：primary labels 可用，但 secondary/control outcome 不 ready。
4. H3 成立：oracle weak/strong frontier 存在，`OR1-SafeGoodOracle` 强通过。
5. H4 成立：primitive autopsy 指向 `legal_observability_gap_not_oracle_population_absence`，不是 candidate population 无解。
6. H5/H6 未闭合：legal OGP feature 只有 weak signal，且 feature cost 未测。
7. H7 未闭合：minimal controller 未过，best selected heldout coverage = `0.0`。
8. H8/H9 未闭合：event-driven runtime 没有 measured implementation，empty-event semantics 也未证明。
9. H10 成立：diagnostic paired replay 未运行，也没有用于 controller/threshold/feature selection。
10. H11 未打开：P8/P10 未过，不能进入 official paired replay 或 full functional 判断。

最终一句话：

> v9.3.0 真实执行后停在 `R9-OGPFeatureFail`：当前 candidate/action population 在 oracle 下有强 frontier，但 legal OGP observability、secondary/control outcome materialization 与 measured event-driven runtime 都未闭合；strict PureKAN functional 仍未成功，下一步应重建 legal action-value/risk observables，并实现 empty-step-free measured runtime。
