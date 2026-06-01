# DG-KAN v12.17.2 B320Locked LossAgnosticTargetVisibility FunctionalGeometry 执行复盘

生成时间：2026-05-24 Asia/Singapore

本轮目标：阅读并执行 `docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_含核心代码审查面计划.md`，在不伪造数据、不绕过 gate 的前提下，完成 v12.17.2 的 loss-agnostic target visibility / functional geometry 诊断，并生成执行日志、实验结果复盘日志和代码审查 zip 包。

## 环境

```bash
cd /home/chengshun.wang/DG-LCA
conda run -n kan ...
```

本轮所有 Python 检查与 runner 执行均使用 `conda env kan`。

## 新增文件

```text
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_执行复盘.md
docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_实验结果复盘.md
```

本轮没有修改 B320 model / kernel / optimizer 的实现文件；runner 只读取并审计既有 v12.16 真实 artifact，生成 v12.17.2 的可见性、门控、审计和 code review packet 产物。

## 输入计划与输入产物

计划文档：

```text
docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_含核心代码审查面计划.md
```

输入 v12.16 source runs：

```text
results/v12_16_b320locked_explicit_signal_reservoir_functional/official_3x3_b32_w3_5_10
results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10
```

正式输出目录：

```text
results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts
```

smoke 输出目录：

```text
results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/smoke_official_only
```

## 执行命令

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
```

结果：通过，无输出错误。

official-only smoke：

```bash
rm -rf results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/smoke_official_only
conda run -n kan python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  --run-id smoke_official_only \
  --out-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/smoke_official_only \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/official_3x3_b32_w3_5_10
```

smoke 结果：

```json
{
  "route": "R1-LineCCalibrationUnstable",
  "p2_visibility_pass": 0,
  "p3_response_pass": 0,
  "p4_open": 0
}
```

解释：B32 official source 在 P1 null control calibration 出现 RandomMatchedNorm joint hard-pass false positives，所以不能直接进入 promotion。该 blocker 按计划的修复方向使用 v12.16 repair source，即扩大 batch/sketch/rank 后的 `repair_sketchdim24_rank5_b64_w3_5_10`。

正式 official+repair：

```bash
rm -rf results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts
conda run -n kan python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  --run-id official_from_v1216_artifacts \
  --out-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts
```

正式结果：

```json
{
  "route": "R2-ReleaseUnobservableUnderLossAgnosticFeatures",
  "p2_visibility_pass": 0,
  "p3_response_pass": 0,
  "p4_open": 0
}
```

artifact 完整性检查：

```bash
conda run -n kan python -c "from pathlib import Path; req='v1217_route_decision.json v1217_anchor_monitor.csv v1217_loss_agnostic_contract.csv v1217_provenance_audit.csv v1217_implementation_readback_audit.csv v1217_code_path_map.json v1217_diff_intent_table.csv v1217_critical_code_review_manifest.csv v1217_core_symbol_map.json v1217_code_semantics_trace.csv v1217_manual_review_packet.md v1217_review_blocker_table.csv v1217_linec_null_distribution.csv v1217_linec_variance.csv v1217_linec_threshold_sensitivity.csv v1217_linec_hard_support.csv v1217_target_visibility_features.csv v1217_release_labels_audit_only.csv v1217_visibility_scores.csv v1217_visibility_leaveout.csv v1217_feature_ablation.csv v1217_actuator_response_dictionary.csv v1217_actuator_rank_condition.csv v1217_actuator_safe_combo.csv v1217_response_to_visibility_score.csv v1217_functional_candidates.csv v1217_functional_p3_audit.csv v1217_functional_controls.csv v1217_functional_candidate_selection_rule.json v1217_p4_short_run.csv v1217_p4_event_log.csv v1217_p4_controls.csv v1217_p4_linec_trajectory.csv v1217_p4_efficiency.csv v1217_classic_family_status.csv v1217_failure_table.csv v1217_hash_manifest.json v1217_code_review_packet.zip'.split(); out=Path('results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts'); missing=[x for x in req if not (out/x).exists()]; print('missing=', missing); print('count=', len(req), 'existing=', len(req)-len(missing))"
```

结果：

```text
missing= []
count= 38 existing= 38
```

代码审查 zip 内容检查：

```bash
conda run -n kan python -c "import zipfile; p='results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts/v1217_code_review_packet.zip'; z=zipfile.ZipFile(p); [print(f'{i.file_size:8d} {i.filename}') for i in z.infolist()]"
```

说明：系统中没有 `unzip` 命令，已改用 Python `zipfile` 验证 zip 内容。

## 关键产物

正式 route：

```text
results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts/v1217_route_decision.json
```

代码审查 zip：

```text
results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts/v1217_code_review_packet.zip
sha256 = 60e2aea839bfd605f05e82027b077c58c0028b99b5419b3f26e240275d99af94
```

zip 中包含代码文件：

```text
dgkan/kernels/fused_hinge_quadratic.py
dgkan/models/fc_purekan_primitives.py
dgkan/optim/manual_adamw.py
dgkan/profiling/timing.py
docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_含核心代码审查面计划.md
experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py
experiments/run_v1215_continuation_actuator_budget_repair.py
experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
experiments/run_v1252_efficiency_functional_manifold.py
experiments/run_v1283_b109_classic_family_functional_geometry.py
```

zip 中包含审查 artifact：

```text
v1217_critical_code_review_manifest.csv
v1217_core_symbol_map.json
v1217_code_semantics_trace.csv
v1217_manual_review_packet.md
v1217_review_blocker_table.csv
v1217_implementation_readback_audit.csv
v1217_code_path_map.json
v1217_diff_intent_table.csv
v1217_route_decision.json
```

## blocker 与修复尝试记录

### Blocker 1：official B32 P1 null false positive

smoke 中 B32 official-only route 为：

```text
R1-LineCCalibrationUnstable
```

正式 P1 source calibration：

```text
official_3x3_b32_w3_5_10:
  rows = 864
  noise_hard_support_rows = 80
  reservoir_hard_support_rows = 76
  joint_hard_support_rows = 16
  random_joint_false_positive_rows = 4
  control_joint_false_positive_rate = 0.012345679012345678
  source_calibration_pass = 0

repair_sketchdim24_rank5_b64_w3_5_10:
  rows = 864
  noise_hard_support_rows = 72
  reservoir_hard_support_rows = 80
  joint_hard_support_rows = 8
  random_joint_false_positive_rows = 0
  control_joint_false_positive_rate = 0.0
  source_calibration_pass = 1
```

按计划修复方向：使用已存在的 v12.16 repair source，即扩大 batch/sketch/rank 后的 B64/sketch24/rank5 产物进入 P2/P3；没有降低 release threshold。

### Blocker 2：P2 visibility 不过

正式 route：

```text
R2-ReleaseUnobservableUnderLossAgnosticFeatures
```

P2 使用通过 P1 校准的 repair source：

```text
visibility_linec_rows = 864
release_label_rows = 864
p2_feature_rows = 24192
p2_score_rows = 21
p2_leaveout_rows = 189
```

关键 heldout min：

```text
p2_auc_joint_heldout_min = 0.37149532710280375
p2_auc_noise_heldout_min = 0.41947250280583614
p2_auc_reservoir_heldout_min = 0.7714285714285715
p2_precision_joint_heldout_min = 0.0
p2_recall_joint_heldout_min = 0.0
p2_visibility_pass = 0
```

因此没有进入 P4。

### Blocker 3：P3 executor gate 不过

P3 使用通过 P1 校准的 repair actuator rows：

```text
visibility_actuator_rows = 99
p3_expected_group_count = 9
p3_safe_movement_pass_count = 0
p3_exploratory_release_pass_count = 0
p3_hard_release_pass_count = 0
p3_response_pass = 0
p3_max_actuator_sketch_delta_fro = 0.00858838576823473
p3_max_actuator_projector_angle_deg = 2.553436517715454
p3_best_noise_delta = -0.003841221332550049
p3_best_reservoir_delta = -0.006813347339630127
p3_max_response_rank = 5.0
p3_min_response_condition = 8.123402310333205
```

结论：rank/condition 不是首要 blocker；safe movement 和 exploratory release 都没有过 3x3 group gate。

## P4 状态

```text
p4_open = 0
p4_not_opened_reason = P2_visibility_gate_failed
```

P4 short-run、event log、controls、LineC trajectory 和 efficiency 均生成 gated-not-run artifact，没有编造在线训练结果。

## 审计说明

`v1217_critical_code_review_manifest.csv` 生成 CR0-CR14 共 15 行：

```text
critical_code_review_surface_pass = 1
critical_unknown_cr0_cr11_count = 0
manual_review_pending_count = 13
review_blocker_count = 13
```

含义：Codex 已列出 CR0-CR14 的文件、symbol、line range 和审查说明；CR0-CR12 仍是 manual review pending，所以本轮 diagnostic 可以成立，但不允许 promotion。

## 工作区提醒

执行 `git status --short` 时发现仓库已有大量与本轮无关的 modified/deleted/untracked 文件。本轮没有回滚这些文件，也没有把它们作为本轮修改处理。新增/修改与本轮相关的主要文件是 v12.17.2 runner 和两份复盘日志。

## 继续推进记录：P2 Visibility Repair

用户要求“没有完成请继续”，因此在基础 R2 结果后继续按文档推荐方向尝试 P2 修复。继续点如下：

```text
基础 P2 blocker:
  joint_all heldout AUC min = 0.37149532710280375
  joint precision/recall min = 0.0 / 0.0

文档推荐方向:
  如果 proxy 有信号但 hard audit 不改善，不调 functional lambda；
  回到 P2 visibility atlas 增加 feature family / normalization；
  如果 support 太薄，增加 actuator response dictionary，不降低 release threshold。
```

追加实现：

```text
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
  新增 weighted_ridge_predict
  新增 run_p2_visibility_repair
  新增 build_release_support_concentration
```

追加 artifact：

```text
v1217_visibility_repair_attempts.csv
v1217_visibility_repair_summary.csv
v1217_release_support_concentration.csv
```

追加 smoke：

```bash
rm -rf results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/smoke_repair_visibility
conda run -n kan python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  --run-id smoke_repair_visibility \
  --out-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/smoke_repair_visibility \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10
```

追加 smoke 结果：

```json
{
  "route": "R2-ReleaseUnobservableUnderLossAgnosticFeatures",
  "p2_visibility_pass": 0,
  "p3_response_pass": 0,
  "p4_open": 0
}
```

追加正式重跑：

```bash
rm -rf results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts
conda run -n kan python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  --run-id official_from_v1216_artifacts \
  --out-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts
```

追加正式结果：

```json
{
  "route": "R2-ReleaseUnobservableUnderLossAgnosticFeatures",
  "p2_visibility_pass": 0,
  "p3_response_pass": 0,
  "p4_open": 0
}
```

P2 repair 结果：

```text
p2_visibility_repair_attempt_rows = 63
p2_visibility_repair_summary_rows = 21
best_protocol = repair_nonrole_geometry/mean_noise_reservoir_rank
best_auc_noise_min = 0.6720854377104377
best_auc_reservoir_min = 0.7308912627551021
best_auc_joint_min = 0.6732622663551402
best_precision_joint_min = 0.0
best_recall_joint_min = 0.0
p2_visibility_repair_pass = 0
```

支持集中度结果：

```text
p2_joint_support_total_rows = 8
p2_joint_support_concentrated = 1
p2_joint_support_seed_distribution = {"0": 8}
p2_joint_support_window_distribution = {"10": 8}
p2_leaveout_train_joint_absent_count = 2
p2_leaveout_test_positive_with_train_joint_absent_count = 2
```

解释：repair composite score 能把 joint AUC min 从 `0.37149532710280375` 提高到 `0.6732622663551402`，说明 proxy/ranking 有部分信号；但 hard gate 需要 top-k joint precision/recall，结果仍为 `0.0/0.0`。更关键的是，repair source 的 joint hard release 只有 8 行，且全部集中在 `seed=0, window=10`，导致 leave-seed 和 leave-window 场景存在 train joint support absent。因此继续调 scorer/权重不能打开 P2/P4。

更新后的关键 hash：

```text
v1217_route_decision.json = fb8bab95575b080fb4345574a09f5464d874849324ec9bf9f88d306ea1ed5691
v1217_code_review_packet.zip = 60e2aea839bfd605f05e82027b077c58c0028b99b5419b3f26e240275d99af94
v1217_visibility_repair_summary.csv = 550d06e16c090ec9c9ae2f592174ee2b17ec59a16b53d0a11c6f3e0d954d1137
v1217_release_support_concentration.csv = 645cb765e3920e045889ec87804f0386ded126815f0edc19e4e16f5085994201
```

更新后的 artifact 完整性检查：

```bash
conda run -n kan python -c "from pathlib import Path; req='v1217_route_decision.json v1217_anchor_monitor.csv v1217_loss_agnostic_contract.csv v1217_provenance_audit.csv v1217_implementation_readback_audit.csv v1217_code_path_map.json v1217_diff_intent_table.csv v1217_critical_code_review_manifest.csv v1217_core_symbol_map.json v1217_code_semantics_trace.csv v1217_manual_review_packet.md v1217_review_blocker_table.csv v1217_linec_null_distribution.csv v1217_linec_variance.csv v1217_linec_threshold_sensitivity.csv v1217_linec_hard_support.csv v1217_target_visibility_features.csv v1217_release_labels_audit_only.csv v1217_release_support_concentration.csv v1217_visibility_scores.csv v1217_visibility_leaveout.csv v1217_feature_ablation.csv v1217_visibility_repair_attempts.csv v1217_visibility_repair_summary.csv v1217_actuator_response_dictionary.csv v1217_actuator_rank_condition.csv v1217_actuator_safe_combo.csv v1217_response_to_visibility_score.csv v1217_functional_candidates.csv v1217_functional_p3_audit.csv v1217_functional_controls.csv v1217_functional_candidate_selection_rule.json v1217_p4_short_run.csv v1217_p4_event_log.csv v1217_p4_controls.csv v1217_p4_linec_trajectory.csv v1217_p4_efficiency.csv v1217_classic_family_status.csv v1217_failure_table.csv v1217_hash_manifest.json v1217_code_review_packet.zip'.split(); out=Path('results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts'); missing=[x for x in req if not (out/x).exists()]; print('missing=', missing); print('count=', len(req), 'existing=', len(req)-len(missing))"
```

结果：

```text
missing= []
count= 41 existing= 41
```

## 2026-05-24 继续推进 2：Actuator Dictionary Expansion Audit

触发原因：用户再次要求“没有完成请继续”。复查计划文件后，确认 v12.17.2 在 P2 support 太薄时给出的下一步不是降低 release threshold，也不是调 functional lambda，而是“增加 candidate perturbation / actuator response dictionary”。上一轮只做了 P2 scorer repair 和 support concentration 诊断，还没有把既有 v12.15 actuator dictionary 扩展候选纳入审计，因此本轮继续补齐这一项。

本轮修改文件：

```text
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
```

代码修改摘要：

```text
1. 新增 DEFAULT_V1215_EXPANSION_SOURCES，纳入 3 个真实存在的 v12.15 continuation actuator artifact：
   - results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_actuator_budget_repair_3x3_b32_w5/v1215_continuation_actuator_budget_repair.csv
   - results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_fused_primitive_actuator_3x3_b32_w5/v1215_continuation_actuator_budget_repair.csv
   - results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_p3_bidir_selector_3x3_b32_w5/v1215_continuation_p3_candidates.csv

2. 新增 run_actuator_dictionary_expansion_audit：
   - 只读取真实 v12.15 artifact；
   - 重新计算 safe_movement_gate / exploratory_joint_release_gate / official_hard_joint_release_gate；
   - 写出 audit-only 表；
   - 所有行 promotion_allowed=0；
   - not_promotable_reason 明确写入 external_v1215_dictionary_expansion_audit_only / p2_visibility_failed。

3. REQUIRED_ARTIFACTS 新增：
   - v1217_actuator_dictionary_expansion_audit.csv
   - v1217_actuator_dictionary_expansion_summary.csv

4. build_provenance_audit 增加 v12.15 expansion source hash / exists 记录。

5. CR7 manifest 增加 run_actuator_dictionary_expansion_audit 和两个 expansion artifact。

6. write_readback_artifacts 的 code_path_map 增加 actuator_expansion_audit_sources。

7. run_main 在 P3 后调用 expansion audit，并把统计字段写入 route summary。该统计不参与 P3 pass，不打开 P4。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
```

结果：通过，退出码 0。

正式重跑命令：

```bash
rm -rf results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts && \
conda run -n kan python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  --run-id official_from_v1216_artifacts \
  --out-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts
```

正式重跑 stdout：

```json
{
  "out_dir": "/home/chengshun.wang/DG-LCA/results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts",
  "route": "R2-ReleaseUnobservableUnderLossAgnosticFeatures",
  "p2_visibility_pass": 0,
  "p3_response_pass": 0,
  "p4_open": 0
}
```

route 关键字段读取命令：

```bash
conda run -n kan python -c 'import json; from pathlib import Path; out=Path("results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts"); route=json.loads((out/"v1217_route_decision.json").read_text()); keys=["route","fail_reason","promotion_allowed","p2_visibility_pass","p2_visibility_repair_pass","p2_visibility_repair_best_protocol","p2_visibility_repair_best_auc_noise_min","p2_visibility_repair_best_auc_reservoir_min","p2_visibility_repair_best_auc_joint_min","p2_visibility_repair_best_precision_joint_min","p2_visibility_repair_best_recall_joint_min","p2_joint_support_total_rows","p2_joint_support_concentrated","p3_response_pass","p3_safe_movement_pass_count","p3_exploratory_release_pass_count","p3_hard_release_pass_count","actuator_expansion_rows","actuator_expansion_safe_row_count","actuator_expansion_exploratory_joint_row_count","actuator_expansion_hard_joint_row_count","actuator_expansion_safe_dataset_seed_count","actuator_expansion_exploratory_dataset_seed_count","actuator_expansion_hard_dataset_seed_count","actuator_expansion_best_noise_delta","actuator_expansion_best_reservoir_delta","actuator_expansion_promotable","p4_open","p4_not_opened_reason","code_review_packet_zip_sha256","hash_manifest_entries"]; [print(f"{k}={route.get(k)}") for k in keys]'
```

route 关键字段结果：

```text
route=R2-ReleaseUnobservableUnderLossAgnosticFeatures
fail_reason=p2_visibility_gate_failed;manual_review_pending_blocks_promotion
promotion_allowed=0
p2_visibility_pass=0
p2_visibility_repair_pass=0
p2_visibility_repair_best_protocol=repair_nonrole_geometry/mean_noise_reservoir_rank
p2_visibility_repair_best_auc_noise_min=0.6720854377104377
p2_visibility_repair_best_auc_reservoir_min=0.7308912627551021
p2_visibility_repair_best_auc_joint_min=0.6732622663551402
p2_visibility_repair_best_precision_joint_min=0.0
p2_visibility_repair_best_recall_joint_min=0.0
p2_joint_support_total_rows=8
p2_joint_support_concentrated=1
p3_response_pass=0
p3_safe_movement_pass_count=0
p3_exploratory_release_pass_count=0
p3_hard_release_pass_count=0
actuator_expansion_rows=603
actuator_expansion_safe_row_count=1
actuator_expansion_exploratory_joint_row_count=0
actuator_expansion_hard_joint_row_count=0
actuator_expansion_safe_dataset_seed_count=1
actuator_expansion_exploratory_dataset_seed_count=0
actuator_expansion_hard_dataset_seed_count=0
actuator_expansion_best_noise_delta=-0.006748616695404053
actuator_expansion_best_reservoir_delta=-0.014425188302993774
actuator_expansion_promotable=0
p4_open=0
p4_not_opened_reason=P2_visibility_gate_failed
code_review_packet_zip_sha256=c47d87f057ab8376a46415eb81afe488a3f5dc64e65e9b832825f2b947d61abe
hash_manifest_entries=44
```

expansion summary 读取命令：

```bash
conda run -n kan python -c 'import csv; from pathlib import Path; out=Path("results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts"); names=["v1217_actuator_dictionary_expansion_summary.csv","v1217_actuator_dictionary_expansion_audit.csv","v1217_provenance_audit.csv","v1217_critical_code_review_manifest.csv"]; 
for name in names:
    rows=list(csv.DictReader((out/name).open(newline="", encoding="utf-8"))); print(f"{name}: rows={len(rows)}");
    if name.endswith("summary.csv"):
        [print({k:r.get(k) for k in ["source_artifact","rows","safe_row_count","exploratory_joint_row_count","hard_joint_row_count","safe_dataset_seed_count","exploratory_dataset_seed_count","hard_dataset_seed_count","best_noise_delta","best_reservoir_delta","expansion_promotable"]}) for r in rows]'
```

expansion summary 结果：

```text
v1217_actuator_dictionary_expansion_summary.csv: rows=3
{'source_artifact': 'results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_actuator_budget_repair_3x3_b32_w5/v1215_continuation_actuator_budget_repair.csv', 'rows': '216', 'safe_row_count': '0', 'exploratory_joint_row_count': '0', 'hard_joint_row_count': '0', 'safe_dataset_seed_count': '0', 'exploratory_dataset_seed_count': '0', 'hard_dataset_seed_count': '0', 'best_noise_delta': '-0.006306260824203491', 'best_reservoir_delta': '-0.014425188302993774', 'expansion_promotable': '0'}
{'source_artifact': 'results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_fused_primitive_actuator_3x3_b32_w5/v1215_continuation_actuator_budget_repair.csv', 'rows': '378', 'safe_row_count': '1', 'exploratory_joint_row_count': '0', 'hard_joint_row_count': '0', 'safe_dataset_seed_count': '1', 'exploratory_dataset_seed_count': '0', 'hard_dataset_seed_count': '0', 'best_noise_delta': '-0.006748616695404053', 'best_reservoir_delta': '-0.014425188302993774', 'expansion_promotable': '0'}
{'source_artifact': 'results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_p3_bidir_selector_3x3_b32_w5/v1215_continuation_p3_candidates.csv', 'rows': '9', 'safe_row_count': '0', 'exploratory_joint_row_count': '0', 'hard_joint_row_count': '0', 'safe_dataset_seed_count': '0', 'exploratory_dataset_seed_count': '0', 'hard_dataset_seed_count': '0', 'best_noise_delta': '-0.0035085678100585938', 'best_reservoir_delta': '-0.002069234848022461', 'expansion_promotable': '0'}
v1217_actuator_dictionary_expansion_audit.csv: rows=603
v1217_provenance_audit.csv: rows=21
v1217_critical_code_review_manifest.csv: rows=15
```

一次 artifact check 命令失败记录：

```text
目的：动态 import runner 并读取 REQUIRED_ARTIFACTS。
失败原因：检查脚本使用 importlib.util.module_from_spec 后未把模块写入 sys.modules，dataclass 初始化报 AttributeError: 'NoneType' object has no attribute '__dict__'。
性质：检查脚本失败，不是 runner 失败；runner 已通过 py_compile 和正式重跑。
修复：检查命令中加入 sys.modules["v1217"]=mod 后重跑。
```

修正后的 artifact / hash / zip check 命令：

```bash
conda run -n kan python -c 'import json, zipfile, importlib.util, sys; from pathlib import Path; runner=Path("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py"); spec=importlib.util.spec_from_file_location("v1217", runner); mod=importlib.util.module_from_spec(spec); sys.modules["v1217"]=mod; spec.loader.exec_module(mod); out=Path("results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts"); missing=[name for name in mod.REQUIRED_ARTIFACTS if not (out/name).exists()]; print("required_count=", len(mod.REQUIRED_ARTIFACTS)); print("missing=", missing); manifest=json.loads((out/"v1217_hash_manifest.json").read_text());
for name in ["v1217_actuator_dictionary_expansion_audit.csv","v1217_actuator_dictionary_expansion_summary.csv","v1217_route_decision.json","v1217_code_review_packet.zip"]: print(name, "hash_manifest_has=", name in manifest, "sha256=", manifest.get(name));
with zipfile.ZipFile(out/"v1217_code_review_packet.zip") as zf: names=zf.namelist(); print("zip_entries=", len(names)); [print("zip_has", target, target in names) for target in ["code/experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py","code/docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_含核心代码审查面计划.md","review_artifacts/v1217_critical_code_review_manifest.csv"]]'
```

修正后的 artifact / hash / zip check 结果：

```text
required_count= 43
missing= []
v1217_actuator_dictionary_expansion_audit.csv hash_manifest_has= True sha256= f8f4a76b685bf9445c981bb9ecdd53a3d42037cb2e94283c888ae8c7a4ee9b98
v1217_actuator_dictionary_expansion_summary.csv hash_manifest_has= True sha256= a6b8d98495787bbf52cef617fa03d1c179ece5c9fbce2d2373c65328375444a5
v1217_route_decision.json hash_manifest_has= True sha256= 3721d54e1f044031fbb2a87431b314cdce6eeaba675efbb41356999898b6a34d
v1217_code_review_packet.zip hash_manifest_has= True sha256= c47d87f057ab8376a46415eb81afe488a3f5dc64e65e9b832825f2b947d61abe
zip_entries= 20
zip_has code/experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py True
zip_has code/docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_含核心代码审查面计划.md True
zip_has review_artifacts/v1217_critical_code_review_manifest.csv True
```

CR7 manifest check 命令：

```bash
conda run -n kan python -c 'import csv; from pathlib import Path; out=Path("results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts"); rows=list(csv.DictReader((out/"v1217_critical_code_review_manifest.csv").open(newline="", encoding="utf-8"))); cr7=[r for r in rows if r["review_id"]=="CR7"][0]; print("CR7_symbols=", cr7["main_symbols"]); print("CR7_artifacts=", cr7["artifact_fields_written"]); print("CR7_unknown=", cr7["unknown_or_not_inspected"]);'
```

CR7 manifest check 结果：

```text
CR7_symbols= actuator_basis_deltas | annotate_p3_matrix | safety_cap_delta | run_p3_actuator_response | run_actuator_dictionary_expansion_audit
CR7_artifacts= v1217_actuator_response_dictionary.csv,v1217_actuator_safe_combo.csv,v1217_actuator_dictionary_expansion_audit.csv,v1217_actuator_dictionary_expansion_summary.csv
CR7_unknown= 0
```

收尾校验：

```bash
conda run -n kan python -m py_compile experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
```

结果：通过，退出码 0。

```bash
git diff --check -- experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_执行复盘.md \
  docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_实验结果复盘.md
```

结果：无输出，退出码 0。

```bash
ps -u "$USER" -o pid,stat,etime,cmd | rg 'conda run -n kan python experiments/run_v1217|python experiments/run_v1217' | rg -v 'rg conda run|rg python experiments|/bin/bash -lc ps' || true
```

结果：无输出；没有遗留中的 v12.17.2 official runner 进程。

相关文件状态：

```text
?? docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_实验结果复盘.md
?? docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_执行复盘.md
?? experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
```

说明：上述 3 个文件是本轮新建/维护的 v12.17.2 runner 与双日志；未回滚或覆盖用户其他工作区改动。

## 2026-05-24 Completion Audit

用户再次询问是否完成。本节只核对现有产物和日志，不新增实验、不改变 route。

命令：

```bash
conda run -n kan python -c 'import json, csv, zipfile; from pathlib import Path; out=Path("results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts"); route=json.loads((out/"v1217_route_decision.json").read_text()); required=["v1217_route_decision.json","v1217_actuator_dictionary_expansion_audit.csv","v1217_actuator_dictionary_expansion_summary.csv","v1217_code_review_packet.zip","v1217_hash_manifest.json"]; missing=[x for x in required if not (out/x).exists()]; exp=list(csv.DictReader((out/"v1217_actuator_dictionary_expansion_summary.csv").open(newline="", encoding="utf-8"))); z=zipfile.ZipFile(out/"v1217_code_review_packet.zip"); print("route=", route.get("route")); print("p2_visibility_pass=", route.get("p2_visibility_pass")); print("p3_response_pass=", route.get("p3_response_pass")); print("p4_open=", route.get("p4_open")); print("promotion_allowed=", route.get("promotion_allowed")); print("actuator_expansion_rows=", route.get("actuator_expansion_rows")); print("actuator_expansion_hard_joint_row_count=", route.get("actuator_expansion_hard_joint_row_count")); print("missing_key_artifacts=", missing); print("expansion_summary_rows=", len(exp)); print("zip_entries=", len(z.namelist()));'
```

结果：

```text
route= R2-ReleaseUnobservableUnderLossAgnosticFeatures
p2_visibility_pass= 0
p3_response_pass= 0
p4_open= 0
promotion_allowed= 0
actuator_expansion_rows= 603
actuator_expansion_hard_joint_row_count= 0
missing_key_artifacts= []
expansion_summary_rows= 3
zip_entries= 20
```

completion audit 结论：v12.17.2 文档当前可执行计划已执行到收尾边界；没有 missing key artifacts；route 没有变化；没有新增成功声明。

## 2026-05-24 Completion Audit 2 / Line D Artifact Contract Repair

再次按用户要求复查“是否完成”时，重新对照计划文档 `# 13. 总 artifact contract`，发现上一轮 completion audit 只检查了 key artifacts，没有覆盖总 contract 中的 Line D 两个文件：

```text
v1217_classic_family_new_hypothesis.csv
v1217_classic_family_linec.csv
```

缺口性质：

```text
这是 artifact contract 缺口，不是实验成功/失败数据缺口。
Line D 在 v12.17.2 中仍是 status monitor only，不允许把旧 focused candidates 重新包装为新进展。
因此修复方向是补 not-run/status 审计表，而不是补假 Line D 实验。
```

修改文件：

```text
experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
```

修改内容：

```text
1. REQUIRED_ARTIFACTS 增加：
   - v1217_classic_family_new_hypothesis.csv
   - v1217_classic_family_linec.csv

2. build_classic_family_status 同时写出：
   - v1217_classic_family_status.csv
   - v1217_classic_family_new_hypothesis.csv
   - v1217_classic_family_linec.csv

3. new_hypothesis 表明确：
   - new_hypothesis_implemented = 0
   - promotion_allowed = 0
   - not_implemented_reason = no concrete new Line D hypothesis was introduced in v12.17.2; functional P2/P3 gates stayed closed

4. linec 表明确：
   - linec_rerun = 0
   - linec_metric_rows = 0
   - linec_not_rerun_reason = status monitor only; no new Line D candidate was generated, so no Line C family rerun exists

5. CR13 artifact_fields_written 增加两个 Line D artifact。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
```

结果：通过，退出码 0。

正式重跑：

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

总 contract 对照命令：

```bash
conda run -n kan python -c 'import re,json,sys,importlib.util,csv; from pathlib import Path; doc=Path("docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_含核心代码审查面计划.md").read_text(encoding="utf-8"); block=doc.split("# 13. 总 artifact contract",1)[1].split("# required sections",1)[0]; doc_artifacts=sorted(set(re.findall(r"v1217_[A-Za-z0-9_]+\\.(?:csv|json|md|zip)", block))); out=Path("results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts"); missing_doc=[x for x in doc_artifacts if not (out/x).exists()]; runner=Path("experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py"); spec=importlib.util.spec_from_file_location("v1217", runner); mod=importlib.util.module_from_spec(spec); sys.modules["v1217"]=mod; spec.loader.exec_module(mod); missing_runner=[x for x in mod.REQUIRED_ARTIFACTS if not (out/x).exists()]; route=json.loads((out/"v1217_route_decision.json").read_text()); print("doc_artifact_count=", len(doc_artifacts)); print("missing_doc_artifacts=", missing_doc); print("runner_required_count=", len(mod.REQUIRED_ARTIFACTS)); print("missing_runner_artifacts=", missing_runner); print("route=", route.get("route")); print("line_d_rows=", route.get("line_d_rows")); print("line_d_new_hypothesis_rows=", route.get("line_d_new_hypothesis_rows")); print("line_d_linec_rows=", route.get("line_d_linec_rows")); print("line_d_new_hypothesis_implemented=", route.get("line_d_new_hypothesis_implemented"));'
```

结果：

```text
doc_artifact_count= 37
missing_doc_artifacts= []
runner_required_count= 45
missing_runner_artifacts= []
route= R2-ReleaseUnobservableUnderLossAgnosticFeatures
line_d_rows= 12
line_d_new_hypothesis_rows= 12
line_d_linec_rows= 12
line_d_new_hypothesis_implemented= 0
```

Line D artifact 读取命令：

```bash
conda run -n kan python -c 'import csv,json; from pathlib import Path; out=Path("results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts");
for name in ["v1217_classic_family_status.csv","v1217_classic_family_new_hypothesis.csv","v1217_classic_family_linec.csv"]:
 rows=list(csv.DictReader((out/name).open(newline="", encoding="utf-8"))); print(name, "rows=", len(rows)); print(rows[0] if rows else {});
manifest=json.loads((out/"v1217_hash_manifest.json").read_text());
for name in ["v1217_classic_family_new_hypothesis.csv","v1217_classic_family_linec.csv","v1217_route_decision.json","v1217_code_review_packet.zip"]: print(name, manifest.get(name));'
```

结果摘要：

```text
v1217_classic_family_status.csv rows= 12
v1217_classic_family_new_hypothesis.csv rows= 12
v1217_classic_family_linec.csv rows= 12

v1217_classic_family_new_hypothesis.csv ff676d83e9ff384d395716e21b336c4aef7a33074e2e2c87937b307b9259617d
v1217_classic_family_linec.csv a21db6dc8b751f01c4b69381538f71ea2ab8d7ccb87fffe589d5da1037d1048d
v1217_route_decision.json 4e7547828958cbd4f47be3adc0cb602372e87b16e9e6c2ffa1f08c4fb51d9498
v1217_code_review_packet.zip f930309bbe33cb99932405614559806978ed10a691d51d10ba9a72e51c4e49f9
```

CR13 检查：

```bash
conda run -n kan python -c 'import csv; from pathlib import Path; out=Path("results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts"); rows=list(csv.DictReader((out/"v1217_critical_code_review_manifest.csv").open(newline="", encoding="utf-8"))); cr13=[r for r in rows if r["review_id"]=="CR13"][0]; print("CR13_artifacts=", cr13["artifact_fields_written"]); print("CR13_unknown=", cr13["unknown_or_not_inspected"]); print("CR13_summary=", cr13["codex_summary"]);'
```

结果：

```text
CR13_artifacts= v1217_classic_family_status.csv,v1217_classic_family_new_hypothesis.csv,v1217_classic_family_linec.csv
CR13_unknown= 0
CR13_summary= Line D remains status-monitor only; new-hypothesis and LineC tables explicitly record that no new classic family repair was executed.
```

official 输出目录当前文件数：

```text
47
```

本次 completion audit 2 结论：此前确有 Line D artifact contract 缺口；已按文档语义补齐为 not-run/status 审计表，并重跑 official。route 仍为 R2，没有新增 P3/P4/promotion 成功声明。

Completion audit 2 收尾检查：

```bash
git diff --check -- experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_执行复盘.md \
  docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_实验结果复盘.md
```

结果：无输出，退出码 0。

```bash
conda run -n kan python -c 'from pathlib import Path; p=Path("docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_实验结果复盘.md"); text=p.read_text(encoding="utf-8"); required=["Implementation Readback / Code Rationale","Existing Code Path Understanding","New Code Implementation Rationale","Mathematical Objects and Tensor Shapes","Gradient / Backward / Update Semantics","Loss-Agnostic and No-Dataset-Branch Audit","Control and Gate Semantics","Timing / Memory Measurement Semantics","Known Ambiguities and Risks","Files Changed and Diff Intent Table","Reproduction Commands and Artifact Hashes","Critical Code Review Surface"]+[f"CR{i}" for i in range(15)]; missing=[x for x in required if x not in text]; print("required_section_count=", len(required)); print("missing_sections=", missing);'
```

结果：

```text
required_section_count= 27
missing_sections= []
```

```bash
ps -u "$USER" -o pid,stat,etime,cmd | rg 'conda run -n kan python experiments/run_v1217|python experiments/run_v1217' | rg -v 'rg conda run|rg python experiments|/bin/bash -lc ps' || true
```

结果：无输出；没有遗留中的 v12.17.2 official runner 进程。

最终工作区相关文件状态：

```text
?? docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_实验结果复盘.md
?? docs/DG-KAN_v12.17.2_B320Locked_LossAgnosticTargetVisibility_FunctionalGeometry_执行复盘.md
?? experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py
```
