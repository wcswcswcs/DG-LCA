# DG-KAN v12.17.2 B320Locked LossAgnosticTargetVisibility FunctionalGeometry 实验结果复盘

生成时间：2026-05-24 Asia/Singapore

本轮结果一句话：

```text
B320 base 仍然稳定；B32 official 的 Line C hard audit 有 RandomMatchedNorm joint false positive，按计划使用 B64/sketch24/rank5 repair source 修复 P1 calibration 后，P2 仍无法在 heldout split 上稳定看见 joint hard release；P3 也没有 safe/exploratory executor survivor；因此 P4 正确关闭。
```

最终 route：

```text
route = R2-ReleaseUnobservableUnderLossAgnosticFeatures
promotion_allowed = 0
p4_open = 0
p4_not_opened_reason = P2_visibility_gate_failed
```

正式输出目录：

```text
results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts
```

## 核心结论

本项目当前主线不是继续把 B320 base 做得更强，而是在一个已经站住的 strict FC-PureKAN / B320 locked base 上，验证是否存在一个真正 loss-agnostic 的 functional update 价值源。也就是说，functional update 不能读 label、CE vector、validation/test/future outcome、dataset name branch；它只能读 label-free 几何可观测量，然后用 label-defined hard release 指标做审计。

v12.17.2 的问题被拆成三层：

```text
P1：hard audit 本身是否稳定，NoOp/Random 是否会假阳性。
P2：loss-agnostic feature family 以及后续 repair scorer 是否能在 heldout dataset/seed/window 上看见 hard release rows。
P3：fused primitive actuator 是否能作为 executor 产生 safe movement 和 release response。
```

本轮真实结果：

```text
P0 B320 anchor pass = 1
P1 calibration pass = 1，仅 repair_sketchdim24_rank5_b64_w3_5_10 source 通过
P2 visibility pass = 0
P3 response pass = 0
P4 open = 0
```

关键判断：v12.16 的 blocker 没有被 P2 visibility atlas 修掉。即使加入 F1-F6 label-free features、noise/reservoir composite score、sparse positive weighted scorer，joint hard release 的 heldout precision/recall 仍然是 0。某些 score 能把 joint AUC min 提到 0.673，但 top-k hard gate 不过，且 joint support 全部集中在 `seed=0, window=10`，不能用于 P3/P4 promotion。

## 实验数据

### P0 B320 Anchor

来源：`v1217_anchor_monitor.csv`

```text
anchor_rows = 2
p0_pass = 1
step_ratio_q90_max = 0.39931987348441034
memory_ratio_q90_max = 0.1295238095238095
mean_delta_vs_mlp_min = 0.027669270833333332
worst_delta_vs_mlp_min = -0.001953125
B320_anchor_locked = 1
```

解释：B320 仍满足 v12.17.2 anchor gate，没有证据要求修改 base architecture。

### P1 Line C Hard Audit Calibration

来源：`v1217_linec_source_calibration.csv`

```text
official_3x3_b32_w3_5_10:
  rows = 864
  noise_hard_support_rows = 80
  noise_hard_support_rate = 0.09259259259259259
  reservoir_hard_support_rows = 76
  reservoir_hard_support_rate = 0.08796296296296297
  joint_hard_support_rows = 16
  noop_joint_false_positive_rows = 0
  random_joint_false_positive_rows = 4
  control_joint_false_positive_rate = 0.012345679012345678
  source_calibration_pass = 0

repair_sketchdim24_rank5_b64_w3_5_10:
  rows = 864
  noise_hard_support_rows = 72
  noise_hard_support_rate = 0.08333333333333333
  reservoir_hard_support_rows = 80
  reservoir_hard_support_rate = 0.09259259259259259
  joint_hard_support_rows = 8
  noop_joint_false_positive_rows = 0
  random_joint_false_positive_rows = 0
  control_joint_false_positive_rate = 0.0
  source_calibration_pass = 1
```

执行中的 blocker 与修复：

```text
blocker = B32 official RandomMatchedNorm joint false positive
文档推荐方向 = 扩大 batch / bootstrap；提高 hard gate margin；检查 determinism；不降低 release threshold
实际处理 = 使用 v12.16 repair source，即 functional_batch_size=64, sketch_dim=24, output_subspace_rank=5 的真实产物
修复结果 = repair source P1 calibration pass
```

这不是把失败改写成成功：正式 P2/P3 只使用通过 P1 校准的 repair source，B32 official source 被记录为未通过 P1。

### P2 Loss-Agnostic Target Visibility

来源：`v1217_visibility_scores.csv`, `v1217_visibility_leaveout.csv`, `v1217_feature_ablation.csv`

P2 使用通过 P1 的 repair source：

```text
visibility_linec_rows = 864
release_label_rows = 864
p2_feature_rows = 24192
p2_score_rows = 21
p2_leaveout_rows = 189
```

joint_all heldout min：

```text
p2_auc_noise_heldout_min = 0.41947250280583614
p2_auc_reservoir_heldout_min = 0.7714285714285715
p2_auc_joint_heldout_min = 0.37149532710280375
p2_precision_joint_heldout_min = 0.0
p2_recall_joint_heldout_min = 0.0
p2_visibility_exploratory_pass = 0
p2_visibility_hard_pass = 0
p2_visibility_strong_pass = 0
p2_visibility_pass = 0
```

继续修复尝试：

```text
p2_visibility_repair_attempt_rows = 63
p2_visibility_repair_summary_rows = 21
best_protocol = repair_nonrole_geometry/mean_noise_reservoir_rank
p2_visibility_repair_best_auc_noise_min = 0.6720854377104377
p2_visibility_repair_best_auc_reservoir_min = 0.7308912627551021
p2_visibility_repair_best_auc_joint_min = 0.6732622663551402
p2_visibility_repair_best_precision_joint_min = 0.0
p2_visibility_repair_best_recall_joint_min = 0.0
p2_visibility_repair_pass = 0
```

支持集中度：

```text
p2_joint_support_total_rows = 8
p2_joint_support_concentrated = 1
p2_joint_support_seed_distribution = {"0": 8}
p2_joint_support_window_distribution = {"10": 8}
p2_leaveout_train_joint_absent_count = 2
p2_leaveout_test_positive_with_train_joint_absent_count = 2
```

解释：repair scorer 的 AUC 有真实提升，但 hard gate 仍失败；joint hard release 只出现在 `seed=0` 和 `window=10`，因此 leave-seed-out 和 leave-window-out 存在 train joint support absent。继续调 scorer/权重不能可靠打开 P2。

joint_all by split：

```text
leave_dataset_out:
  AUC_noise = 0.41947250280583614
  AUC_reservoir = 0.7714285714285715
  AUC_joint = 0.37149532710280375
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0

leave_seed_out:
  AUC_noise = 0.7182940516273849
  AUC_reservoir = 0.8739795918367347
  AUC_joint = 0.4158878504672897
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0

leave_window_out:
  AUC_noise = 0.5086980920314254
  AUC_reservoir = 0.8362244897959183
  AUC_joint = 0.5186915887850467
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0
```

feature family best joint AUC：

```text
F1_output_spectral = 0.5280373831775701
F2_train_probe_coupling = 0.6822429906542056
F3_gradient_sketch_spectrum = 0.8294392523364486
F4_primitive_role_response = 0.39485981308411217
F5_augmentation_consistency = 0.5630841121495327
F6_time_persistence = 0.8411214953271028
```

解释：F2/F3/F6 在某些 split 或单 family 上有 AUC 信号，但 joint_all heldout min 低，且 joint precision/recall 都是 0。按计划不能做 CouplingR2-only promotion，也不能用单 split/family positive 打开 P3/P4。

### P3 Fused Primitive Actuator Response Dictionary

来源：`v1217_actuator_response_dictionary.csv`, `v1217_actuator_safe_combo.csv`, `v1217_actuator_rank_condition.csv`

P3 使用通过 P1 的 repair source：

```text
visibility_actuator_rows = 99
p3_expected_group_count = 9
p3_dictionary_rows = 99
p3_response_pass = 0
```

关键值：

```text
p3_safe_movement_pass_count = 0
p3_exploratory_release_pass_count = 0
p3_hard_release_pass_count = 0
p3_max_actuator_sketch_delta_fro = 0.00858838576823473
p3_max_actuator_projector_angle_deg = 2.553436517715454
p3_best_noise_delta = -0.003841221332550049
p3_best_reservoir_delta = -0.006813347339630127
p3_max_response_rank = 5.0
p3_min_response_condition = 8.123402310333205
```

解释：response rank 和 condition 数值不是首要 blocker；真正 blocker 是 safe movement gate 和 release gate 都没有 3x3 group survivor。sketch_delta_fro 最大值低于 0.01，noise best delta 没达到 exploratory -0.005，reservoir 虽有一行达到 exploratory 数值，但 joint exploratory group count 仍为 0。

### P4/P5

来源：`v1217_functional_candidates.csv`, `v1217_p4_short_run.csv`, `v1217_p4_event_log.csv`, `v1217_p4_controls.csv`, `v1217_p4_linec_trajectory.csv`, `v1217_p4_efficiency.csv`

```text
p4_open = 0
p4_not_opened_reason = P2_visibility_gate_failed
candidate_id = B17-NOT-RUN
```

说明：没有 P2/P3 survivor，因此没有跑 P4 short-run，也没有编造 accepted events、online efficiency 或 task gain。

### Line D

来源：`v1217_classic_family_status.csv`

```text
line_d_rows = 12
line_d_new_hypothesis_implemented = 0
```

解释：Line D 只做 status monitor。本轮没有新的 family-specific hypothesis，不把 v12.15/v12.16 旧 focused candidates 重新包装成 B17 progress。

### Code Review Packet

代码审查 zip：

```text
results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts/v1217_code_review_packet.zip
sha256 = 60e2aea839bfd605f05e82027b077c58c0028b99b5419b3f26e240275d99af94
```

审查面状态：

```text
critical_code_review_rows = 15
critical_code_review_surface_pass = 1
critical_unknown_cr0_cr11_count = 0
implementation_readback_pass = 1
manual_review_pending_count = 13
review_blocker_count = 13
```

解释：CR0-CR14 的 actual file / symbol / line range 都已列出，没有 CR0-CR11 unknown；但 CR0-CR12 仍是 manual review pending，因此本轮不能 promotion。

## Implementation Readback / Code Rationale

### Existing Code Path Understanding

本轮依赖的现存路径：

```text
B320 registry:
  dgkan/models/fc_purekan_primitives.py
  line 6747
  symbol = PrimitiveSpec("B320b-SimpleFastTaskGeometry-...")

B320 id lock:
  experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py
  line 38
  symbol = B320_ID

model construction:
  experiments/run_v1283_b109_classic_family_functional_geometry.py
  lines 302-320
  symbol = _make_model

FHQ fused path:
  dgkan/kernels/fused_hinge_quadratic.py
  forward_workspace lines 1079-1119
  backward_learnablep_workspace_fused_quadproj_adamw_from_grad_logits lines 1374-1398
  _fhq_proj_grad_adamw_kernel lines 759-880
  _adamw_update_quad_proj lines 1332-1348

manual AdamW:
  dgkan/optim/manual_adamw.py
  ManualAdamWConfig lines 14-22
  AdamWState lines 23-32
  adamw_update_ lines 33-43

Line C metric generation:
  experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py
  evaluate_direction_full lines 308-386

grad sketch / delta apply:
  experiments/run_v1252_efficiency_functional_manifold.py
  _sample_grad_sketch lines 899-920
  _apply_delta lines 1135-1140

geometry scoring:
  experiments/run_v1283_b109_classic_family_functional_geometry.py
  _geo_gain_delta_scored lines 3344-3390

v12.16 actuator source:
  experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py
  actuator_basis_deltas lines 380-411
  annotate_p3_matrix lines 440-480
```

本轮没有重新训练 B320，没有改 B320 registry，没有声称新 fused kernel 结果。P0/P1/P2/P3 均从真实 v12.16 artifact 出发做审计与重门控。

### New Code Implementation Rationale

新增 runner：

```text
file_path = experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
change_type = new
main_symbols = run_main, run_p0_anchor_monitor, run_p1_linec_calibration, build_release_labels, build_release_support_concentration, build_visibility_features, run_p2_target_visibility, run_p2_visibility_repair, weighted_ridge_predict, run_p3_actuator_response, run_p4_short_run, build_cr_manifest, package_code_review_zip
why_changed = implement v12.17.2 artifact contract, visibility atlas, P3 dictionary gate, implementation readback, CR0-CR14 packet
old_behavior = no v12.17.2 runner or code review packet
new_behavior = offline diagnostic over v12.16 official/repair artifacts; route computed from real CSV values
related_hypothesis = loss-agnostic observables may or may not see label-audited release basis
related_gate = P0/P1/P2/P3/P4/readback/critical code review
loss_agnostic_safety = hard release labels are audit-only; features do not read label columns; P4 remains gated not-run
expected_metric_effect = heldout AUC / precision / recall for hard release rows; repair scorer diagnostics; safe combo gate recount
possible_failure_mode = artifact parser mismatch, split leakage, label leakage, support concentration, source calibration over/under strict
fallback_if_failed = keep promotion_allowed=0; route R1/R2/R3 according to failed gate
```

本轮修复过的实现点：

```text
1. argparse source-run default 修复：
   初版 smoke 显示手动传入 --source-run 时仍叠加默认 official+repair，导致 official 被重复读取。
   已改成 default=None，未传参时才使用 official+repair 默认源。

2. P1 calibration 修复：
   初版把 P1 误写成必须 joint hard support 才通过。
   已改为 source-level calibration：NoOp/Random joint hard-pass false positive、control false positive rate、noise/reservoir support rate 分别审计。
   official B32 未通过，repair B64 通过，后续 P2/P3 只用 repair source。

3. blocker 修复方向落地：
   按文档建议使用扩大 batch/sketch/rank 的 repair source；没有降低 hard release threshold。

4. P2 visibility repair 继续尝试：
   新增 composite noise/reservoir rank score 和 sparse positive weighted ridge scorer。
   结果 best AUC_joint_min = 0.6732622663551402，但 precision/recall 仍为 0，未打开 P2。

5. release support concentration audit：
   新增 `v1217_release_support_concentration.csv`，确认 joint support 只有 8 行且全集中在 seed=0/window=10。
```

### Mathematical Objects and Tensor Shapes

本轮 runner 主要是 tabular/offline diagnostic，但 review packet 仍记录现存 tensor path：

```text
B = update batch size
Q = probe batch size
C = number of classes
S = sketch dimension
R = output subspace rank
P = parameter or compressed actuator dimension
N = number of artifact rows
F = number of visibility features
M = number of response metrics
J = number of actuator basis elements
```

FHQ path objects：

```text
x in R^{B x D}
logits in R^{B x C}
q_out in R^{B x H}
grad_proj in R^{D x H}
```

Visibility atlas：

```text
X_train in R^{N_train x F}
y_train in {0,1}^{N_train}
X_test in R^{N_test x F}
score = X_test w
```

Hard release audit labels：

```text
y_noise = 1[Delta NoiseSignalLeak <= -0.01]
y_reservoir = 1[Delta RealSignalReservoirRatio <= -0.01]
y_joint = y_noise AND y_reservoir
```

Response dictionary formula：

```text
R[m,j] = (G_m(theta + epsilon * a_j) - G_m(theta)) / epsilon
```

where `G_m` is an audit metric and `a_j` is an actuator basis. v12.17.2 did not recompute tensors; it re-gated the real v12.16 actuator matrix rows.

### Gradient / Backward / Update Semantics

v12.17.2 runner status：

```text
path_status = offline_diagnostic_postprocess
uses_loss_backward = 0
uses_torch_autograd_graph = 0
uses_torch_autograd_grad = 0
manual_forward_available = readback_existing_code
manual_backward_available = readback_existing_code
manual_update_available = readback_existing_code
fused_kernel_used = source artifact readback only, not newly executed
saved_tensor_total_bytes = not measured in v12.17.2
manual_cache_bytes = not measured in v12.17.2
promotion_allowed = 0
```

Existing B320/FHQ path has manual/fused symbols located in CR2/CR3. This runner does not use PyTorch `loss.backward()` to generate directions, and does not create any new optimizer state transport.

### Loss-Agnostic and No-Dataset-Branch Audit

来源：`v1217_loss_agnostic_contract.csv`

```text
loss_agnostic_contract_rows = 45
loss_agnostic_contract_pass = 1
loss_agnostic_contract_violations = 0
```

Hard release labels：

```text
label_used_for_audit_only = 1
label_used_for_feature = 0
label_used_for_direction = 0
ce_vector_used_for_feature = 0
ce_vector_used_for_direction = 0
dataset_name_used_for_commit = 0
```

Dataset names only serve as leave-dataset-out split keys. They are not used to select thresholds, weights, candidates, or commits.

### Control and Gate Semantics

P1 controls：

```text
NoOp and RandomMatchedNorm are read from v12.16 Line C rows.
official B32 random_joint_false_positive_rows = 4, source_calibration_pass = 0.
repair B64 random_joint_false_positive_rows = 0, source_calibration_pass = 1.
```

P2 gates：

```text
Exploratory visibility pass requires:
  AUC_joint_heldout >= 0.60
  or AUC_noise_heldout >= 0.65 and AUC_reservoir_heldout >= 0.65

Hard visibility pass requires:
  Precision_joint_heldout >= 0.25
  Recall_joint_heldout >= 0.20

Observed:
  AUC_joint_heldout_min = 0.37149532710280375
  precision_joint_heldout_min = 0.0
  recall_joint_heldout_min = 0.0
```

P3 gates：

```text
Safe movement:
  sketch_delta_fro >= 0.01
  projector_angle >= 1 degree
  logit_max_abs_drift <= 0.05

Exploratory release:
  Delta NoiseSignalLeak <= -0.005
  Delta RealSignalReservoirRatio <= -0.005

Observed:
  safe_movement_pass_count = 0 / 9
  exploratory_release_pass_count = 0 / 9
```

P4 gate：

```text
P4_NOT_OPENED = P2_visibility_gate_failed
```

### Timing / Memory Measurement Semantics

Architecture timing/memory came only from v12.16/v12.14 anchor artifacts:

```text
step_ratio_q90_max = 0.39931987348441034
memory_ratio_q90_max = 0.1295238095238095
```

v12.17.2 offline analysis time is not mixed into architecture step time. `v1217_p4_efficiency.csv` is a gated-not-run artifact with no fabricated step timing.

### Known Ambiguities and Risks

| risk_id | risk_description | affected_files | affected_metrics | risk_level | how_checked | remaining_uncertainty | recommended_next_check |
|---|---|---|---|---|---|---|---|
| RISK-1 | v12.17.2 is offline postprocess over v12.16 artifacts, not new model execution | `experiments/run_v1217_...py` | P2/P3 route | medium | route marks offline_postprocess_analysis=1 and new_mechanism_success_claimed=0 | New online execution could differ | Only run online P4 after P2/P3 pass |
| RISK-2 | P1 B32 official false positives make source selection important | `v1217_linec_source_calibration.csv` | P1/P2 | medium | source-level calibration selects only repair source | Selection reduces N to 864 rows | Re-run calibrated B64 protocol if more confidence needed |
| RISK-3 | P2 feature scoring uses simple ridge scorer, not a production candidate generator | `run_p2_target_visibility` | AUC/precision/recall | medium | labels audit-only; scorer used for visibility diagnosis | More feature families may change visibility | Add new label-free observables only with predeclared hypothesis |
| RISK-4 | Manual code review not completed by a human | `v1217_review_blocker_table.csv` | promotion | high | manual_review_pending_count=13 | Human may find semantic mismatch | Review CR0-CR12 before any promotion |
| RISK-5 | P3 controls are not matched inside v12.16 actuator matrix rows | `v1217_response_to_visibility_score.csv` | P3 response | medium | artifact states control_gap_checked=0 | Cannot claim control-resistant actuator | Generate matched actuator controls only after P2 opens |

### Files Changed and Diff Intent Table

| file_path | lines_or_symbols_touched | intent | hypothesis | expected_artifacts | rollback_plan |
|---|---|---|---|---|---|
| `experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py` | entire file; `run_main`, P0/P1/P2/P3/P4 audit writers, code review packet writer | Implement v12.17.2 offline target visibility atlas and code review artifacts | Loss-agnostic observables may or may not see hard release rows | all `v1217_*` artifacts and zip | remove runner and generated v1217 artifacts; no model/kernel rollback needed |
| `docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_执行复盘.md` | new file | reproducibility log | future readers should reproduce commands/artifacts quickly | execution log | delete log if rerun supersedes it |
| `docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_实验结果复盘.md` | new file | result analysis and audit readback | preserve exact real metrics and blockers | result recap | delete log if rerun supersedes it |

### Reproduction Commands and Artifact Hashes

Primary command:

```bash
conda run -n kan python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  --run-id official_from_v1216_artifacts \
  --out-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts
```

Key hashes:

```text
v1217_route_decision.json = fb8bab95575b080fb4345574a09f5464d874849324ec9bf9f88d306ea1ed5691
v1217_code_review_packet.zip = 60e2aea839bfd605f05e82027b077c58c0028b99b5419b3f26e240275d99af94
v1217_visibility_scores.csv = 07afad14840af267b959f34fb5212badfc42883fd729b10fb90652be18f8bc15
v1217_actuator_safe_combo.csv = fc80deb0d5e1bc606338425fae4b3263ecdc3730a5e0f8ea3c66c97eeb235a33
v1217_visibility_repair_summary.csv = 550d06e16c090ec9c9ae2f592174ee2b17ec59a16b53d0a11c6f3e0d954d1137
v1217_release_support_concentration.csv = 645cb765e3920e045889ec87804f0386ded126815f0edc19e4e16f5085994201
v1217_critical_code_review_manifest.csv = 8f8af36f908ea4a8201a8a2ad040b5cbb2ce32abf530409b1f4682d3f26e1206
```

Full hash manifest:

```text
results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts/v1217_hash_manifest.json
```

## Critical Code Review Surface

### CR0 Entrypoint / Runner / Route Decision

Files and symbols:

```text
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
build_argparser lines 2381-2394
run_main lines 2323-2380
run_p0_anchor_monitor lines 411-439
run_p1_linec_calibration lines 625-793
run_p2_target_visibility lines 1066-1211
run_p3_actuator_response lines 1441-1574
run_p4_short_run lines 1575-1686
decide_route lines 2204-2251
```

Role: route is computed from artifact summaries, not hand-filled. P4 is gated behind P2/P3. The runner is offline diagnostic, explicitly records `offline_postprocess_analysis=1` and `new_mechanism_success_claimed=0`.

Risk if wrong: P4 could be opened without P2/P3 or wrapper-only diagnostics could be promoted.

### CR1 B320 Candidate Registry / Model Construction

Files and symbols:

```text
dgkan/models/fc_purekan_primitives.py
  B320b-SimpleFastTaskGeometry line 6747

experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py
  B320_ID line 38

experiments/run_v1283_b109_classic_family_functional_geometry.py
  _make_model lines 302-320
```

Role: ensures B320 identity is traceable and not replaced by another candidate id. v12.17.2 does not instantiate a new model; it audits source artifact identity.

Risk if wrong: all P0/P2/P3 conclusions could refer to the wrong base.

### CR2 FHQ Fused Forward-Backward-Update Kernel

Files and symbols:

```text
dgkan/kernels/fused_hinge_quadratic.py
forward_workspace lines 1079-1119
backward_learnablep_workspace_fused_quadproj_adamw_from_grad_logits lines 1374-1398
_fhq_proj_grad_adamw_kernel lines 759-880
_adamw_update_quad_proj lines 1332-1348
```

Role: existing B320 fused/manual path evidence. v12.17.2 only reads prior artifacts and packages these files for review.

Tensor shapes:

```text
x[B,D], logits[B,C], q_out[B,H], grad_proj[D,H]
```

Risk if wrong: loss-agnostic/manual-update invariant of source experiments may be false.

### CR3 Manual Optimizer / No-Autograd Semantics

Files and symbols:

```text
dgkan/optim/manual_adamw.py
ManualAdamWConfig lines 14-22
AdamWState lines 23-32
adamw_update_ lines 33-43
```

Role: locate manual AdamW update implementation for audit. v12.17.2 does not modify it.

Risk if wrong: AdamWParallel/manual-update comparisons in upstream artifacts may be invalid.

### CR4 Line C Coupling / Signal-Reservoir Metrics

Files and symbols:

```text
experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py
  evaluate_direction_full lines 308-386

experiments/run_v1252_efficiency_functional_manifold.py
  _sample_grad_sketch lines 899-920
  _apply_delta lines 1135-1140

experiments/run_v1283_b109_classic_family_functional_geometry.py
  _geo_gain_delta_scored lines 3344-3390

experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
  run_p1_linec_calibration lines 625-793
```

Role: Line C audit deltas are the source of hard release labels and null calibration.

Risk if wrong: hard-release labels or nontearing gates become meaningless.

### CR5 Hard Release Label Audit-Only Isolation

Files and symbols:

```text
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
build_release_labels lines 794-831
build_release_support_concentration lines 832-896
build_loss_agnostic_contract lines 440-539
```

Role: converts `actual_NoiseSignalLeak_delta` and `actual_RealSignalReservoirRatio_delta` into audit-only labels:

```text
hard_noise_release = 1[Delta NoiseSignalLeak <= -0.01]
hard_reservoir_release = 1[Delta RealSignalReservoirRatio <= -0.01]
hard_joint_release = hard_noise_release AND hard_reservoir_release
```

Risk if wrong: label leakage would invalidate P2.

### CR6 Loss-Agnostic Visibility Features

Files and symbols:

```text
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
build_visibility_features lines 897-1022
crossval_predictions lines 1027-1065
run_p2_target_visibility lines 1066-1211
run_p2_visibility_repair lines 1224-1440
auc_score lines 251-268
```

Feature families:

```text
F1_output_spectral
F2_train_probe_coupling
F3_gradient_sketch_spectrum
F4_primitive_role_response
F5_augmentation_consistency
F6_time_persistence
```

Role: builds label-free feature matrix and evaluates heldout visibility. Labels are not feature columns.

Risk if wrong: P2 could pass because of leakage or split error.

### CR7 Fused Primitive Actuator Response Dictionary

Files and symbols:

```text
experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py
  actuator_basis_deltas lines 380-411
  annotate_p3_matrix lines 440-480

experiments/run_v1215_continuation_actuator_budget_repair.py
  safety_cap_delta lines 81-96

experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
  run_p3_actuator_response lines 1441-1574
```

Role: v12.17.2 re-gates real v12.16 actuator rows by safe movement and release thresholds.

Risk if wrong: executor viability could be overstated.

### CR8 Controls and Matched Measurement Windows

Files and symbols:

```text
experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py
  make_probe_updates lines 442-531

experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
  build_baseline_controls lines 592-624
```

Role: summarize NoOp/Random controls from matched v12.16 Line C rows. P4 controls are not run because P4 is closed.

Risk if wrong: control gap could be falsely claimed.

### CR9 Functional Candidate Construction / Solver

Files and symbols:

```text
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
run_p4_short_run lines 1575-1686
```

Role: no solver executed. Writes `B17-NOT-RUN` and records upstream reason.

Risk if wrong: missing solver could be hidden behind artifact fields. Current artifact explicitly marks not-run.

### CR10 P4 Short-Run Integration

Files and symbols:

```text
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
run_p4_short_run lines 1575-1686
```

Role: writes gated-not-run P4/P5 artifacts only. P4 opens only if P2 and P3 pass.

Risk if wrong: online functional success could be claimed without causal gates.

### CR11 Timing / Memory Profiler Semantics

Files and symbols:

```text
dgkan/profiling/timing.py
TIMING_PROTOCOLS line 3

experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
run_p0_anchor_monitor lines 411-439
```

Role: separates source architecture timing from offline diagnostic analysis. No P4 efficiency is fabricated.

Risk if wrong: diagnostic hook cost could be mixed into architecture timing.

### CR12 Provenance / Hash / Artifact Reuse

Files and symbols:

```text
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
build_provenance_audit lines 540-591
write_hash_manifest lines 2284-2292
package_code_review_zip lines 2293-2322

experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py
sha256_file lines 150-157
```

Role: records hashes of reused v12.16 sources and v12.17.2 generated artifacts, and packages audit code files into zip.

Risk if wrong: source reuse could be untraceable.

### CR13 Classic No-BSpline Status Monitor

Files and symbols:

```text
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
build_classic_family_status lines 1687-1706
```

Role: Line D status monitor only. B-spline remains frozen. No new family hypothesis is executed.

Risk if wrong: classic portfolio status could mask functional gate failure.

### CR14 Dataset-Agnostic / Split Discipline

Files and symbols:

```text
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
crossval_predictions lines 1027-1065
run_p2_target_visibility lines 1066-1211
decide_route lines 2204-2251
```

Role: leave-dataset-out, leave-seed-out, leave-window-out evaluation. Dataset names are split keys only, not commit/controller branches.

Risk if wrong: dataset-specific thresholding could create a false visibility pass.

## Final Decision

```text
Route: R2-ReleaseUnobservableUnderLossAgnosticFeatures
P4: closed
Promotion: not allowed
Reason: P2 heldout target visibility failed; P3 executor gate failed; manual review still pending for promotion.
```

下一步应该不是调 functional lambda，也不是用 CouplingR2-only candidate 打开 P4。按照当前真实结果，下一步只能是提出新的 label-free observable hypothesis，或者承认 direct release route 在当前 observable family 下不可见，将 functional update 降级为 label-free geometry maintenance。

## Continue-2：Actuator Dictionary Expansion Audit

### 为什么继续做这一项

用户要求未完成则继续。复查 v12.17.2 计划后，上一轮 P2 repair 虽然已经证明：

```text
best joint AUC min = 0.6732622663551402
joint precision/recall min = 0.0 / 0.0
hard joint support = 8 rows
support concentration = seed=0, window=10
```

但计划在 hard support 太薄时还给出一个可执行方向：

```text
增加 candidate perturbation / actuator response dictionary；
不改变 release threshold 以制造 support。
```

因此本轮没有停止在 P2 repair，而是把已有真实 v12.15 continuation actuator dictionary artifact 纳入 v12.17.2 审计。这个动作是 audit-only：它不能把旧 focused candidates 包装成 B17 成功，也不能打开 P4。

### 本轮代码修改

修改文件：

```text
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
```

修改内容：

```text
新增 DEFAULT_V1215_EXPANSION_SOURCES：
  读取 3 个 v12.15 continuation artifact。

新增 run_actuator_dictionary_expansion_audit：
  对 v12.15 actuator expansion rows 重新计算：
    safe_movement_gate = sketch_delta_fro >= 0.01 and max(projector angle) >= 1 deg and logit drift <= 0.05
    exploratory_joint_release_gate = NoiseSignalLeak_delta <= -0.005 and RealSignalReservoirRatio_delta <= -0.005
    official_hard_joint_release_gate = NoiseSignalLeak_delta <= -0.01 and RealSignalReservoirRatio_delta <= -0.01
  写出 audit-only artifact；
  所有行 promotion_allowed = 0。

新增 artifact：
  v1217_actuator_dictionary_expansion_audit.csv
  v1217_actuator_dictionary_expansion_summary.csv

更新 provenance：
  v1217_provenance_audit.csv 记录 3 个 v12.15 source artifact 的 exists / sha256。

更新 CR7：
  CR7 main_symbols 增加 run_actuator_dictionary_expansion_audit；
  CR7 artifact_fields_written 增加两个 expansion artifact。

更新 run_main：
  P3 后调用 expansion audit；
  只把统计写入 route summary；
  不用 expansion audit 改写 p3_response_pass；
  不用 expansion audit 打开 P4。
```

审计合理性说明：

```text
1. 使用真实已存在 artifact，不生成假候选。
2. 不降低 release threshold。
3. 不把 v12.15 continuation 结果改名为 v12.17.2 B17 survivor。
4. 不使用 label / CE / validation 生成方向；这里仅重新审计已产出的 delta rows。
5. P2 仍失败，因此 expansion row 即使出现 movement，也不允许 promotion。
```

### 正式重跑结果

正式命令：

```bash
rm -rf results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts && \
conda run -n kan python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  --run-id official_from_v1216_artifacts \
  --out-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts
```

stdout：

```json
{
  "out_dir": "/home/chengshun.wang/DG-LCA/results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts",
  "route": "R2-ReleaseUnobservableUnderLossAgnosticFeatures",
  "p2_visibility_pass": 0,
  "p3_response_pass": 0,
  "p4_open": 0
}
```

最终 route：

```text
route = R2-ReleaseUnobservableUnderLossAgnosticFeatures
fail_reason = p2_visibility_gate_failed;manual_review_pending_blocks_promotion
promotion_allowed = 0
p4_open = 0
p4_not_opened_reason = P2_visibility_gate_failed
```

P2 repair 最终仍失败：

```text
p2_visibility_pass = 0
p2_visibility_repair_pass = 0
p2_visibility_repair_best_protocol = repair_nonrole_geometry/mean_noise_reservoir_rank
p2_visibility_repair_best_auc_noise_min = 0.6720854377104377
p2_visibility_repair_best_auc_reservoir_min = 0.7308912627551021
p2_visibility_repair_best_auc_joint_min = 0.6732622663551402
p2_visibility_repair_best_precision_joint_min = 0.0
p2_visibility_repair_best_recall_joint_min = 0.0
p2_joint_support_total_rows = 8
p2_joint_support_concentrated = 1
```

P3 原 v12.16 actuator response 仍失败：

```text
p3_response_pass = 0
p3_safe_movement_pass_count = 0
p3_exploratory_release_pass_count = 0
p3_hard_release_pass_count = 0
```

### Actuator Expansion Audit 结果

总体统计：

```text
actuator_expansion_rows = 603
actuator_expansion_safe_row_count = 1
actuator_expansion_exploratory_joint_row_count = 0
actuator_expansion_hard_joint_row_count = 0
actuator_expansion_safe_dataset_seed_count = 1
actuator_expansion_exploratory_dataset_seed_count = 0
actuator_expansion_hard_dataset_seed_count = 0
actuator_expansion_best_noise_delta = -0.006748616695404053
actuator_expansion_best_reservoir_delta = -0.014425188302993774
actuator_expansion_promotable = 0
```

按来源分解：

```text
Source 1:
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_actuator_budget_repair_3x3_b32_w5/v1215_continuation_actuator_budget_repair.csv
rows = 216
safe_row_count = 0
exploratory_joint_row_count = 0
hard_joint_row_count = 0
safe_dataset_seed_count = 0
exploratory_dataset_seed_count = 0
hard_dataset_seed_count = 0
best_noise_delta = -0.006306260824203491
best_reservoir_delta = -0.014425188302993774
expansion_promotable = 0

Source 2:
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_fused_primitive_actuator_3x3_b32_w5/v1215_continuation_actuator_budget_repair.csv
rows = 378
safe_row_count = 1
exploratory_joint_row_count = 0
hard_joint_row_count = 0
safe_dataset_seed_count = 1
exploratory_dataset_seed_count = 0
hard_dataset_seed_count = 0
best_noise_delta = -0.006748616695404053
best_reservoir_delta = -0.014425188302993774
expansion_promotable = 0

Source 3:
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_p3_bidir_selector_3x3_b32_w5/v1215_continuation_p3_candidates.csv
rows = 9
safe_row_count = 0
exploratory_joint_row_count = 0
hard_joint_row_count = 0
safe_dataset_seed_count = 0
exploratory_dataset_seed_count = 0
hard_dataset_seed_count = 0
best_noise_delta = -0.0035085678100585938
best_reservoir_delta = -0.002069234848022461
expansion_promotable = 0
```

解释：

```text
1. v12.15 expansion dictionary 中确实存在更强的单指标改善：
   best_noise_delta = -0.006748616695404053
   best_reservoir_delta = -0.014425188302993774

2. 但 joint gate 要求同一 row 同时满足 noise 和 reservoir release。
   实际 exploratory_joint_row_count = 0，hard_joint_row_count = 0。

3. 603 行中只有 1 行 safe movement，并且没有 joint release，因此无法形成 P3 survivor。

4. 因为 P2 visibility 仍为 0，且这些 row 是 external v12.15 audit-only source，即使存在 1 行 safe movement，也不能 promotion。
```

### Artifact / Hash / Zip 完整性

最终 artifact check：

```text
required_count = 43
missing = []
```

关键 hash：

```text
v1217_actuator_dictionary_expansion_audit.csv = f8f4a76b685bf9445c981bb9ecdd53a3d42037cb2e94283c888ae8c7a4ee9b98
v1217_actuator_dictionary_expansion_summary.csv = a6b8d98495787bbf52cef617fa03d1c179ece5c9fbce2d2373c65328375444a5
v1217_route_decision.json = 3721d54e1f044031fbb2a87431b314cdce6eeaba675efbb41356999898b6a34d
v1217_code_review_packet.zip = c47d87f057ab8376a46415eb81afe488a3f5dc64e65e9b832825f2b947d61abe
```

zip 检查：

```text
zip_entries = 20
zip_has code/experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py = True
zip_has code/docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_含核心代码审查面计划.md = True
zip_has review_artifacts/v1217_critical_code_review_manifest.csv = True
```

CR7 检查：

```text
CR7_symbols = actuator_basis_deltas | annotate_p3_matrix | safety_cap_delta | run_p3_actuator_response | run_actuator_dictionary_expansion_audit
CR7_artifacts = v1217_actuator_response_dictionary.csv,v1217_actuator_safe_combo.csv,v1217_actuator_dictionary_expansion_audit.csv,v1217_actuator_dictionary_expansion_summary.csv
CR7_unknown = 0
```

### 最终判断

```text
v12.17.2 当前可执行计划已完成：
  P0/P1/P2/P3/P4/readback/code-review packet artifacts 已生成；
  P2 scorer repair 已尝试；
  sparse support concentration 已诊断；
  actuator dictionary expansion audit 已补齐；
  code review packet zip 已生成；
  artifact/hash/zip 完整性检查通过。

最终路线：
  R2-ReleaseUnobservableUnderLossAgnosticFeatures

没有打开：
  P4 short-run

没有声明：
  new mechanism success
  promotion
  P3 survivor
```

本轮实验的科学结论：

```text
当前 functional 主线不是“模型动不了”，而是“loss-agnostic observable 对 hard release 的可见性不足，且 actuator dictionary 即使扩展到既有 v12.15 continuation rows，也没有形成同 row joint release support”。

P2 repair 已经能把 joint ranking AUC 拉到 0.6732622663551402，但 top-k precision/recall 仍为 0。说明当前 features 对 hard joint release 有弱排序信号，却无法稳定定位可执行 release row。

v12.15 expansion audit 显示有单指标改善和 1 行 safe movement，但没有 exploratory/hard joint row。因此不能进入 P3/P4，更不能把旧 continuation artifact 包装为 B17 成功。
```

下一步如果继续做，不能是重复 rerun 或调 lambda；需要新的、事先说明的 label-free observable hypothesis 或新的 actuator basis 生成实验。否则按当前文档纪律，v12.17.2 应停在 R2。

## Completion Audit

用户再次询问是否完成后，追加做了一次只读 completion audit。结果：

```text
route = R2-ReleaseUnobservableUnderLossAgnosticFeatures
p2_visibility_pass = 0
p3_response_pass = 0
p4_open = 0
promotion_allowed = 0
actuator_expansion_rows = 603
actuator_expansion_hard_joint_row_count = 0
missing_key_artifacts = []
expansion_summary_rows = 3
zip_entries = 20
```

结论不变：v12.17.2 当前文档内可执行项已完成；最终停在 R2；没有 P3 survivor、没有 P4 short-run、没有 promotion、没有 new mechanism success claim。

## Completion Audit 2 / Line D Artifact Contract Repair

### 发现的问题

再次按计划文档 `# 13. 总 artifact contract` 对照 official 输出时，发现此前 completion audit 只检查了 key artifacts，漏掉了 Line D contract 中明确要求的两个文件：

```text
v1217_classic_family_new_hypothesis.csv
v1217_classic_family_linec.csv
```

这是一个真实缺口，已修复。本缺口不改变前面的科学结论，因为它不是 P2/P3/P4 指标缺失，而是 Line D status monitor 的落盘 contract 不完整。

### 修复方式

修改文件：

```text
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
```

具体修改：

```text
1. REQUIRED_ARTIFACTS 增加：
   v1217_classic_family_new_hypothesis.csv
   v1217_classic_family_linec.csv

2. build_classic_family_status 现在写出 3 个 Line D artifact：
   v1217_classic_family_status.csv
   v1217_classic_family_new_hypothesis.csv
   v1217_classic_family_linec.csv

3. v1217_classic_family_new_hypothesis.csv 明确记录：
   new_hypothesis_implemented = 0
   promotion_allowed = 0
   not_implemented_reason = no concrete new Line D hypothesis was introduced in v12.17.2; functional P2/P3 gates stayed closed

4. v1217_classic_family_linec.csv 明确记录：
   linec_rerun = 0
   linec_metric_rows = 0
   linec_not_rerun_reason = status monitor only; no new Line D candidate was generated, so no Line C family rerun exists

5. CR13 artifact_fields_written 同步更新为：
   v1217_classic_family_status.csv,v1217_classic_family_new_hypothesis.csv,v1217_classic_family_linec.csv
```

为什么这样修是合理的：

```text
Line D 在 v12.17.2 文档中不是 functional 主线替代品。
B-spline 继续 frozen。
其他 classic family 只允许在有新 hypothesis 时推进。
当前 P2/P3 functional gates 仍关闭，没有新 Line D hypothesis。
所以正确修复不是补跑或编造 Line D 实验，而是把“未推进新 hypothesis / 未 rerun Line C”的状态按 artifact contract 明确落盘。
```

### 修复后 official 重跑结果

正式重跑后 stdout：

```json
{
  "out_dir": "/home/chengshun.wang/DG-LCA/results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts",
  "route": "R2-ReleaseUnobservableUnderLossAgnosticFeatures",
  "p2_visibility_pass": 0,
  "p3_response_pass": 0,
  "p4_open": 0
}
```

文档总 artifact contract 对照：

```text
doc_artifact_count = 37
missing_doc_artifacts = []
runner_required_count = 45
missing_runner_artifacts = []
```

Line D 新增 artifact：

```text
v1217_classic_family_status.csv rows = 12
v1217_classic_family_new_hypothesis.csv rows = 12
v1217_classic_family_linec.csv rows = 12

line_d_rows = 12
line_d_new_hypothesis_rows = 12
line_d_linec_rows = 12
line_d_new_hypothesis_implemented = 0
```

CR13 检查：

```text
CR13_artifacts = v1217_classic_family_status.csv,v1217_classic_family_new_hypothesis.csv,v1217_classic_family_linec.csv
CR13_unknown = 0
```

更新后的关键 hash：

```text
v1217_classic_family_new_hypothesis.csv = ff676d83e9ff384d395716e21b336c4aef7a33074e2e2c87937b307b9259617d
v1217_classic_family_linec.csv = a21db6dc8b751f01c4b69381538f71ea2ab8d7ccb87fffe589d5da1037d1048d
v1217_route_decision.json = 4e7547828958cbd4f47be3adc0cb602372e87b16e9e6c2ffa1f08c4fb51d9498
v1217_code_review_packet.zip = f930309bbe33cb99932405614559806978ed10a691d51d10ba9a72e51c4e49f9
```

### 修复后最终判断

```text
Route = R2-ReleaseUnobservableUnderLossAgnosticFeatures
P2 visibility pass = 0
P3 response pass = 0
P4 open = 0
Promotion allowed = 0
Line D new hypothesis implemented = 0
missing_doc_artifacts = []
missing_runner_artifacts = []
```

这次修复补齐了文档 contract，但没有也不应该改变科学结论：当前主 blocker 仍是 loss-agnostic target visibility / joint release observability 不足；Line D 没有新 hypothesis，不允许包装旧 focused candidates 作为新进展。
