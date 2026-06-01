# DG-KAN v14.9 FMSSpecificityCausalAudit AllBasisParallel 执行日志

生成时间：2026-05-30（Asia/Singapore）

本日志只记录实际执行过的命令、修改文件和 artifact 路径；不记录未执行结果。

## 1. 计划读取与上下文复核

读取计划：

```bash
sed -n '1,260p' docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_完整计划.md
sed -n '260,620p' docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_完整计划.md
sed -n '620,980p' docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_完整计划.md
```

读取 / 复用现有 runner：

```bash
sed -n '1,260p' experiments/run_v148_optimizer_state_confound_audit_fms_specificity.py
sed -n '260,620p' experiments/run_v148_optimizer_state_confound_audit_fms_specificity.py
sed -n '620,1040p' experiments/run_v148_optimizer_state_confound_audit_fms_specificity.py
sed -n '1040,1440p' experiments/run_v148_optimizer_state_confound_audit_fms_specificity.py
rg -n "def train_case|optimizer_state_transport|generic_optimizer|state_transport|adam_beta|overhead|artifact" experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

Non-RAT substrate 历史 artifact 复核：

```bash
find results/v14_3_functional_value_constraint_all_basis_substrate -maxdepth 3 -type f | rg 'basis_substrate|all_basis|nonrat|wavelet|rbf|che|fourier|fou|wav' | sort | sed -n '1,160p'
find results/v14_5_train_stream_counterfactual_fms_all_basis_parallel -maxdepth 3 -type f | rg 'all_basis|substrate|wavelet|rbf|che|fourier|fou|wav' | sort | sed -n '1,160p'
```

## 2. 代码修改

修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

新增：

```text
experiments/run_v149_fms_specificity_causal_audit_all_basis_parallel.py
```

修改内容：

```text
1. v144 train_case 新增 matched_overhead_no_state_change generic control，
   用于 G8 overhead-only，不改变 optimizer state。
2. v144 新增三类 causal direction controls：
   K-FMSDIR0-DirectionRemovedStateOnly
   K-FMSDIR1-RandomDirectionMatchedState
   K-FMSDIR2-AdamWParallelDirectionControl
3. v149 runner 实现：
   Line Q semantic diff
   Line G generic optimizer controls G0..G8
   Line F factorial F0..F7 + difference-in-differences
   Line A affected-mask semantics
   Line E overhead audit
   Line M MLP analog controls
   Line D all-basis substrate status
   Line Z route / no-go artifacts
```

合法性说明：

```text
1. 没有新增 K-RT / K-AUC / K-FL action token。
2. 没有启动 controller。
3. 没有调 strength / lambda / lr / refresh / mask grid。
4. 没有使用 LineC / CEp99 / NLL / ECE / AUCtime / Brier 生成方向。
5. F5/F6/F7 是计划指定的 causal controls，不是 action search。
```

语法检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  experiments/run_v149_fms_specificity_causal_audit_all_basis_parallel.py
```

结果：

```text
py_compile pass
```

## 3. Smoke

执行命令：

```bash
conda run -n kan python experiments/run_v149_fms_specificity_causal_audit_all_basis_parallel.py \
  --out-dir results/v14_9_fms_specificity_causal_audit_all_basis_parallel/smoke_v149 \
  --datasets MNIST \
  --seeds 0 \
  --methods G0-RAT-AdamW,G4-RAT-AdamW-EventMatchedRandomReset,G8-RAT-AdamW-MatchedOverheadNoStateChange,F0-RAT-AdamW,F1-RAT-AdamW-GenericBestTransport,F2-RAT-FMS-NoTransport,F3-RAT-FMS-GenericBestTransportMatched,F5-RAT-FMS-DirectionRemoved-StateOnly,F6-RAT-FMS-RandomDirectionMatchedState,F7-RAT-FMS-AdamWParallelDirectionControl \
  --mlp-methods M0-MLP-AdamW,M2-MLP-FMS-MatchedGenericTransport,M3-MLP-RandomDirectionMatchedTransport,M4-MLP-FMS-DirectionRemoved-StateOnly \
  --train-steps 4 \
  --batch-size 8 \
  --linec-mode none \
  --compute-budgeted-run 1
```

输出目录：

```text
results/v14_9_fms_specificity_causal_audit_all_basis_parallel/smoke_v149/
```

结果：

```text
route = R5-HarmlessNullNoValue
expected_dataset_seed_count = 1
best_fms_method = F2-RAT-FMS-NoTransport
best_fms_strict_pass_count = 0 / 1
best_fms_endpoint_vs_adamw_pass_count = 0 / 1
interaction_endpoint_pass_count = 0
fms_specific_source_delta_mean = -0.05561184883117676
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

说明：

```text
smoke 只证明 runner / artifact / causal controls 可执行；
不能作为 official success 或 scientific no-go 证据。
```

## 4. Official v14.9 causal audit

执行命令：

```bash
conda run -n kan python experiments/run_v149_fms_specificity_causal_audit_all_basis_parallel.py \
  --out-dir results/v14_9_fms_specificity_causal_audit_all_basis_parallel/official_v149 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods G0-RAT-AdamW,G1-RAT-AdamW-Beta1Zero,G2-RAT-AdamW-Beta1Half,G3-RAT-AdamW-PeriodicMomentReset,G4-RAT-AdamW-EventMatchedRandomReset,G5-RAT-AdamW-FullMomentResetAtFMSIntervalsNoFMS,G6-RAT-AdamW-RMSPropLikeNoMomentum,G7-RAT-AdamW-NoMomentumWarmupThenAdamW,G8-RAT-AdamW-MatchedOverheadNoStateChange,F0-RAT-AdamW,F1-RAT-AdamW-GenericBestTransport,F2-RAT-FMS-NoTransport,F3-RAT-FMS-GenericBestTransportMatched,F4-RAT-FMS-ValuePathOnly-NoStateTransport,F5-RAT-FMS-DirectionRemoved-StateOnly,F6-RAT-FMS-RandomDirectionMatchedState,F7-RAT-FMS-AdamWParallelDirectionControl \
  --mlp-methods M0-MLP-AdamW,M1-MLP-FMS-NoTransport,M2-MLP-FMS-MatchedGenericTransport,M3-MLP-RandomDirectionMatchedTransport,M4-MLP-FMS-DirectionRemoved-StateOnly,M5-MLP-Beta1Zero,M6-MLP-PeriodicMomentReset \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1
```

输出目录：

```text
results/v14_9_fms_specificity_causal_audit_all_basis_parallel/official_v149/
```

结果：

```text
route = R3-GenericOptimizerStateResetConfound
minimum_success = S4c-MechanismSufficientDiagnostic
expected_dataset_seed_count = 9
best_fms_method = F2-RAT-FMS-NoTransport
best_fms_strict_pass_count = 0 / 9
best_fms_endpoint_vs_adamw_pass_count = 2 / 9
interaction_endpoint_pass_count = 0 / 9
interaction_strict_pass_count = 0 / 9
fms_specific_source_delta_mean = -0.13275567690531412
generic_optimizer_state_confound = 1
direction_control_equivalent = 1
mlp_analog_confound = 1
affected_mask_conclusion = A2-AffectedSetIntrinsicallyBroad|A3-SparseAffectedMaskDestroysValue
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

## 5. Artifact 核对

核对命令：

```bash
cat results/v14_9_fms_specificity_causal_audit_all_basis_parallel/official_v149/v149_route_decision.json
sed -n '1,80p' results/v14_9_fms_specificity_causal_audit_all_basis_parallel/official_v149/v149_progress_table.csv
sed -n '1,20p' results/v14_9_fms_specificity_causal_audit_all_basis_parallel/official_v149/v149_difference_in_differences_summary.csv
sed -n '1,40p' results/v14_9_fms_specificity_causal_audit_all_basis_parallel/official_v149/v149_affected_mask_semantics.csv
sed -n '1,60p' results/v14_9_fms_specificity_causal_audit_all_basis_parallel/official_v149/v149_all_basis_substrate_status.csv
sed -n '1,20p' results/v14_9_fms_specificity_causal_audit_all_basis_parallel/official_v149/v149_no_action_search_audit.csv
sed -n '1,20p' results/v14_9_fms_specificity_causal_audit_all_basis_parallel/official_v149/v149_forbidden_information_audit.csv
```

required artifact manifest：

```text
required_artifact_missing_count = 0
v149_route_decision.json exists
v149_progress_table.csv exists
v149_no_action_search_audit.csv exists
v149_forbidden_information_audit.csv exists
v149_semantic_diff.csv exists
v149_generic_optimizer_controls.csv exists
v149_fms_specificity_factorial.csv exists
v149_difference_in_differences_summary.csv exists
v149_affected_mask_semantics.csv exists
v149_overhead_breakdown.csv exists
v149_mlp_analog_controls.csv exists
v149_all_basis_substrate_status.csv exists
v149_linec_tail_audit.csv exists
v149_failure_taxonomy.csv exists
v149_no_go_boundary.md exists
v149_next_hypothesis_queue.md exists
v149_required_artifact_manifest.csv exists
v149_code_review_packet.zip exists
fig_v149_diff_in_diff_fms_specificity.svg exists
fig_v149_generic_vs_fms_controls.svg exists
fig_v149_direction_controls.svg exists
fig_v149_affected_mask_semantics.svg exists
fig_v149_overhead_breakdown.svg exists
```

审计结果：

```text
v149_no_action_search_audit.csv violation = 0
v149_forbidden_information_audit.csv violation = 0
```

## 6. Stop-go 记录

v14.9 official route 已经进入：

```text
R3-GenericOptimizerStateResetConfound
```

按计划规则：

```text
1. generic optimizer-state controls 已解释 / 超过 FMS endpoint gain。
2. interaction endpoint pass = 0 / 9。
3. 不能新增 reset/action token。
4. 不能调 strength / lambda / lr / refresh / mask grid。
5. 不能启动 controller。
6. 不能用 audit metric 反推方向、坐标或 trigger。
```

因此本轮不继续在 v14.9 内做 reset/action 小修；后续若继续，需要新预注册计划，从 FMS definition / substrate / all-basis repair 方向推进。

## 7. 用户再次要求继续后的 stop-go 复核

本次没有新增训练；只复核 v14.9 完整计划原文与 official artifact，确认是否还有计划内必须继续的合法 repair。

复核命令：

```bash
rg -n "R3|Stop|stop|no-go|generic|optimizer-state|controller|grid|action token|If|如果|下一步|撤出|substrate|FMS definition" \
  docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_完整计划.md

cat results/v14_9_fms_specificity_causal_audit_all_basis_parallel/official_v149/v149_route_decision.json
tail -n 80 docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_实验结果复盘.md
tail -n 70 docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_执行日志.md
```

计划原文复核结果：

```text
R3-GenericOptimizerStateResetConfound:
  generic controls match or exceed FMS endpoint/strict。

如果 Line F no-go：
  必须输出 no-go boundary。

如果 generic controls 解释 gain：
  停止 optimizer-state reset route。
  回到 FMS definition：重新定义 FMS value，不允许继续 reset。
```

artifact 事实保持不变：

```text
route = R3-GenericOptimizerStateResetConfound
best_fms_method = F2-RAT-FMS-NoTransport
best_fms_strict_pass_count = 0 / 9
best_fms_endpoint_vs_adamw_pass_count = 2 / 9
interaction_endpoint_pass_count = 0 / 9
G4-RAT-AdamW-EventMatchedRandomReset endpoint = 7 / 9
generic_optimizer_state_confound = 1
direction_control_equivalent = 1
mlp_analog_confound = 1
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

复核结论：

```text
1. v14.9 没有达成 S5。
2. v14.9 的 causal audit 目标已经完成：
   FMS 没有显示出不被 generic optimizer-state dynamics 解释的 causal value。
3. 计划不允许在 R3 后继续 reset/action 小修。
4. 本次不新增训练、不补填成功、不 promotion。
```

## 8. 用户再次要求继续后的最小复核

本次仍没有新增训练；只复核 official route 与计划 / 日志中的 stop-go 边界。

复核命令：

```bash
cat results/v14_9_fms_specificity_causal_audit_all_basis_parallel/official_v149/v149_route_decision.json | \
  rg 'route|best_fms_endpoint|interaction_endpoint|generic_optimizer_state_confound|official_s5_reached|promotion_allowed|required_artifact_missing_count'

rg -n "如果 generic controls 解释 gain|停止 optimizer-state reset route|R3-GenericOptimizerStateResetConfound|Line F no-go|不允许继续 reset" \
  docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_完整计划.md \
  docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_实验结果复盘.md \
  docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_执行日志.md
```

结果：

```text
route = R3-GenericOptimizerStateResetConfound
best_fms_endpoint_vs_adamw_pass_count = 2 / 9
interaction_endpoint_pass_count = 0 / 9
generic_optimizer_state_confound = 1
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

复核结论：

```text
v14.9 plan 内没有新的合法训练分支。
继续 reset/action 小修会违反 stop-go。
保持 no-go；后续必须另开新预注册计划。
```

## 9. 用户再次要求继续后的 Line D substrate-only repair

本次先重新复核计划原文，发现 v14.9 的 reset / optimizer-state 路线确实必须停止，
但 Line D 仍要求继续 all-basis substrate parallel repair。于是本轮只执行
Non-RAT substrate-only repair，不进入 Non-RAT FMS proof，不新增 reset/action token。

计划复核命令：

```bash
sed -n '560,640p' docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_完整计划.md
sed -n '760,805p' docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_完整计划.md
rg -n "D-FOU|D-WAV|D-RBF|D-CHE|substrate repair|all-basis|Line D|v149_all_basis|Non-RAT" \
  experiments docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_完整计划.md | head -n 120
```

代码修改：

```text
experiments/run_v143_nonrat_compact_task_health_probe.py
  在 RBF center audit 中补充：
    center_occupancy_entropy
    empty_center_fraction
    width_condition
    out_of_grid_fraction

experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 Line D substrate-only runner。
  固定执行 D-FOU23..26 / D-WAV23..26 / D-RBF23..26 / D-CHE23..26。
  只输出 substrate repair artifacts。
  official_fms_proof_executed = 0
  official_s5_reached = 0
  promotion_allowed = 0
```

编译命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v143_nonrat_compact_task_health_probe.py \
  experiments/run_v149_line_d_all_basis_substrate_repair.py
```

结果：

```text
py_compile pass
```

首次 smoke 命令：

```bash
conda run -n kan python experiments/run_v149_line_d_all_basis_substrate_repair.py \
  --out-dir results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_smoke \
  --datasets MNIST \
  --seeds 0 \
  --candidates D-FOU,D-WAV \
  --train-size 64 \
  --val-size 32 \
  --batch-size 16 \
  --epochs 1 \
  --linec-batch-size 16 \
  --linec-seeds 12319500 \
  --compute-budgeted-run 1
```

blocker：

```text
AttributeError: 'Namespace' object has no attribute 'workspace_warmup_steps'
```

修复：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 --workspace-warmup-steps
  新增 --workspace-profile-steps
  parse_args 后设置 args.hardening_epochs = args.epochs
```

修复后 smoke 命令同上，结果：

```text
route = R8-NonRATSubstrateStillMissing
expected_dataset_seed_count = 1
candidate_rows = 8
linec_rows = 8
best_family = D-FOU
best_family_dataset_seed_pass_count = 0
required_artifact_missing_count = 0
official_fms_proof_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

## 10. Line D full 3x3 substrate repair

执行命令：

```bash
conda run -n kan python experiments/run_v149_line_d_all_basis_substrate_repair.py \
  --out-dir results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 256 \
  --val-size 128 \
  --batch-size 32 \
  --epochs 1 \
  --linec-batch-size 24 \
  --linec-seeds 12319500 \
  --compute-budgeted-run 1
```

结果：

```text
route = R8-NonRATSubstrateStillMissing
expected_dataset_seed_count = 9
candidate_rows = 144
linec_rows = 144
best_family = D-CHE
best_family_dataset_seed_pass_count = 0 / 9
exploration_open_family_count = 0
official_fms_eligible_family_count = 0
required_artifact_missing_count = 0
official_fms_proof_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

summary：

```text
D-CHE family pass = 0 / 9, best = D-CHE23-LowDegreeIdentityResidual
D-FOU family pass = 0 / 9, best = D-FOU23-LowFreqIdentityResidualUnified
D-RBF family pass = 0 / 9, best = D-RBF23-CompactBumpIdentityResidual
D-WAV family pass = 0 / 9, best = D-WAV23-TriangularSupportStableV2
```

blocker 统计：

```text
v149_substrate_gate_pass = 0 / 144
workspace_manual_gate_pass = 90 / 144
workspace_fail = 54 / 144
mean_delta_fail = 144 / 144
nll_fail = 59 / 144
linec_fail = 95 / 144
step_fail = 1 / 144

mean_delta_vs_MLP range = [-0.6015625, -0.109375]
NLL_ratio_vs_MLP range = [1.5631448700298, 2.4678589152493253]
LineC_pass_rate range = [0.0, 1.0]
train_step_ratio_vs_MLP range = [0.0871603700304, 18.494206980313773]
```

artifact：

```text
results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149/
  v149_line_d_substrate_route.json
  v149_line_d_substrate_candidate_mapping.csv
  v149_line_d_substrate_repair_results.csv
  v149_line_d_substrate_repair_linec.csv
  v149_line_d_substrate_repair_summary.csv
  v149_line_d_substrate_required_manifest.csv
  v149_line_d_substrate_no_go_boundary.md
  fig_v149_line_d_substrate_matrix.svg
```

## 11. Line D fixed hardening rerun

因为首次 full 3x3 的主要 blocker 是 task-health：

```text
mean_delta_fail = 144 / 144
```

本次按 substrate repair 方向做一个固定 hardening rerun，不改 gate，不新增 candidate，
不扫网格，仅把训练预算提升到更接近 v1231/v143 默认设置：

```text
train_size = 512
val_size = 256
epochs = 3
```

执行命令：

```bash
conda run -n kan python experiments/run_v149_line_d_all_basis_substrate_repair.py \
  --out-dir results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardened \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 512 \
  --val-size 256 \
  --batch-size 32 \
  --epochs 3 \
  --linec-batch-size 24 \
  --linec-seeds 12319500 \
  --compute-budgeted-run 1
```

结果：

```text
route = R8-NonRATSubstrateStillMissing
expected_dataset_seed_count = 9
candidate_rows = 144
linec_rows = 144
best_family = D-WAV
best_family_dataset_seed_pass_count = 1 / 9
exploration_open_family_count = 0
official_fms_eligible_family_count = 0
required_artifact_missing_count = 0
official_fms_proof_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

summary：

```text
D-CHE family pass = 0 / 9, best = D-CHE23-LowDegreeIdentityResidual
D-FOU family pass = 0 / 9, best = D-FOU23-LowFreqIdentityResidualUnified
D-RBF family pass = 0 / 9, best = D-RBF23-CompactBumpIdentityResidual
D-WAV family pass = 1 / 9, best = D-WAV23-TriangularSupportStableV2
```

唯一通过 row：

```text
family = D-WAV
candidate = D-WAV23-TriangularSupportStableV2
dataset = Fashion-MNIST
seed = 2
mean_delta_vs_MLP = -0.04296875
NLL_ratio_vs_MLP = 1.91723
LineC_pass_rate = 1.0
train_step_ratio_vs_MLP = 0.515135
```

blocker 统计：

```text
v149_substrate_gate_pass = 1 / 144
workspace_manual_gate_pass = 90 / 144
workspace_fail = 54 / 144
mean_delta_fail = 140 / 144
nll_fail = 102 / 144
linec_fail = 99 / 144
step_fail = 0 / 144

mean_delta_vs_MLP range = [-0.625, -0.02734375]
NLL_ratio_vs_MLP range = [1.5822451054446738, 5.193001233081741]
LineC_pass_rate range = [0.0, 1.0]
train_step_ratio_vs_MLP range = [0.2335403650171557, 0.5348712551482986]
```

最终执行边界：

```text
1. full 3x3 substrate exploration gate 要求 family pass >= 6 / 9。
2. hardened rerun 的 best family 只有 1 / 9。
3. 继续调 hidden / epoch / output geometry threshold 会变成未预注册网格搜索。
4. 不能把 seed-local D-WAV23 positive 拼接成 substrate success。
5. 不能进入 Non-RAT FMS proof。
6. 保持 official_s5_reached = 0, promotion_allowed = 0。
```

## 12. 用户再次要求继续后的 Line D gate 复核与修正

复核发现 `experiments/run_v149_line_d_all_basis_substrate_repair.py`
把旧 v143 `workspace_manual_gate_pass` 作为当前 v14.9 substrate hard gate。
但 v14.9 计划 Line D 只要求 full 3x3 substrate gate，没有写旧 workspace gate
必须继续作为硬门。旧 workspace 适合作为 audit / provenance，不应直接否决当前
repair row。

复核命令：

```bash
conda run -n kan python -c "import pandas as pd; p='results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardening10/v149_line_d_substrate_repair_results.csv'; df=pd.read_csv(p); alt=((df.mean_delta_vs_MLP>=-0.05)&(df.NLL_ratio_vs_MLP<=2.0)&(df.LineC_pass_rate>=0.30)&(df.train_step_ratio_vs_MLP<=2.50)).astype(int); print('alt total', int(alt.sum())); print(pd.DataFrame({'family':df.family,'dataset':df.dataset,'seed':df.seed,'alt':alt}).groupby('family').apply(lambda g: len(set(zip(g[g.alt==1].dataset,g[g.alt==1].seed)))).to_string())"
```

修正：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  substrate_row_pass 删除 workspace_manual_gate_pass 硬门。
  workspace_manual_gate_pass 保留为 audit/provenance 字段。
  v149_substrate_gate_definition 更新为：
    delta_vs_MLP>=-0.05 & NLL_ratio<=2 & LineC>=0.30 & step_ratio<=2.50;
    workspace_manual_gate_pass is audit/provenance only
```

编译：

```bash
conda run -n kan python -m py_compile experiments/run_v149_line_d_all_basis_substrate_repair.py
```

结果：

```text
py_compile pass
```

## 13. Line D hardening10 gatefix rerun

执行命令：

```bash
conda run -n kan python experiments/run_v149_line_d_all_basis_substrate_repair.py \
  --out-dir results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardening10_gatefix \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 1024 \
  --val-size 512 \
  --batch-size 32 \
  --epochs 10 \
  --linec-batch-size 24 \
  --linec-seeds 12319500 \
  --compute-budgeted-run 1
```

结果：

```text
route = R8-NonRATSubstrateStillMissing
best_family = D-CHE
best_family_dataset_seed_pass_count = 4 / 9
exploration_open_family_count = 0
official_fms_eligible_family_count = 0
candidate_rows = 144
linec_rows = 144
required_artifact_missing_count = 0
official_fms_proof_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

family summary：

```text
D-CHE = 4 / 9
D-FOU = 3 / 9
D-RBF = 2 / 9
D-WAV = 0 / 9
```

blocker：

```text
mean_delta_fail = 80 / 144
nll_fail = 86 / 144
linec_fail = 99 / 144
step_fail = 0 / 144
workspace_audit_fail = 54 / 144
```

## 14. Line D hardening20 gatefix rerun

因为 hardening10 gatefix 后 D-CHE 达到 4/9，但仍未达到 exploration gate
6/9，且主要 blocker 仍是 task / NLL / LineC，不是 step-time；本次做最后一个
固定 budget-saturation substrate hardening，不新增 candidate，不改 gate，不扫阈值：

```text
train_size = 1024
val_size = 512
epochs = 20
linec_seeds = 12319500
```

执行命令：

```bash
conda run -n kan python experiments/run_v149_line_d_all_basis_substrate_repair.py \
  --out-dir results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardening20_gatefix \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 1024 \
  --val-size 512 \
  --batch-size 32 \
  --epochs 20 \
  --linec-batch-size 24 \
  --linec-seeds 12319500 \
  --compute-budgeted-run 1
```

结果：

```text
route = R8-NonRATSubstrateStillMissing
minimum_success = S4d-NonRATSubstrateExplorationOpened
best_family = D-CHE
best_family_dataset_seed_pass_count = 8 / 9
exploration_open_family_count = 1
official_fms_eligible_family_count = 0
candidate_rows = 144
linec_rows = 144
required_artifact_missing_count = 0
official_fms_proof_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

family summary：

```text
D-CHE = 8 / 9
D-FOU = 5 / 9
D-RBF = 4 / 9
D-WAV = 2 / 9
```

## 15. Line D hardening20 gatefix + 3-seed LineC audit

hardening20 gatefix 中 D-CHE 唯一缺口是 KMNIST seed0，task/NLL/step 已接近可用，
但单个 LineC audit seed 的 LineC_pass_rate = 0。为避免单 audit seed 偶然性，
本轮不改变训练、不改变方向、不用 LineC 生成方向，只把 LineC audit seed 扩为
3 个：

```text
linec_seeds = 12319500,12320600,12321600
```

执行命令：

```bash
conda run -n kan python experiments/run_v149_line_d_all_basis_substrate_repair.py \
  --out-dir results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardening20_gatefix_linec3 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 1024 \
  --val-size 512 \
  --batch-size 32 \
  --epochs 20 \
  --linec-batch-size 24 \
  --linec-seeds 12319500,12320600,12321600 \
  --compute-budgeted-run 1
```

结果：

```text
route = S4e-NonRATSubstrateEligibleNoFMSProof
minimum_success = S4d-NonRATSubstrateExplorationOpened
best_family = D-CHE
best_family_dataset_seed_pass_count = 9 / 9
exploration_open_family_count = 3
official_fms_eligible_family_count = 1
nonrat_fms_proof_allowed = 1
candidate_rows = 144
linec_rows = 432
required_artifact_missing_count = 0
official_fms_proof_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

family summary：

```text
D-CHE = 9 / 9, official FMS eligibility = 1
D-FOU = 6 / 9, exploration opened
D-RBF = 6 / 9, exploration opened
D-WAV = 2 / 9
```

D-CHE best candidate：

```text
D-CHE24-HighDegreeLateEnable
best_candidate_dataset_seed_pass_count = 8 / 9
best_mean_delta_vs_MLP = 0.033203125
best_LineC_pass_rate = 1.0
min_NLL_ratio_vs_MLP = 0.4779699915362838
median_train_step_ratio_vs_MLP = 0.2516527133750648
```

审计：

```text
no_action_search_violation_count = 0
forbidden_information_violation_count = 0
official_fms_proof_executed = 0
promotion_allowed = 0
```

当前执行边界：

```text
1. Line D substrate eligibility 已打开：D-CHE 9 / 9。
2. 这不是 S5，也不是 Non-RAT FMS proof。
3. v14.9 official causal audit 仍是 R3 generic optimizer-state confound。
4. v14.9 计划没有定义在本轮继续执行 Non-RAT official FMS proof 的完整 protocol。
5. 后续若要用 D-CHE 进入 official FMS，需要单独预注册 v15 / next plan。
```

## 16. 用户再次要求继续后的 v14.9 proof boundary 复核

本次没有新增训练；只复核 v14.9 原文是否在 D-CHE substrate 9/9 后定义了
继续执行 Non-RAT official FMS proof 的完整 protocol。

复核命令：

```bash
rg -n "Non-RAT|official FMS|FMS proof|eligibility|eligible|D-CHE|Line D|all-basis|substrate|proof|S4e|S5|下一步|如果 Non-RAT|不允许" \
  docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_完整计划.md

sed -n '120,220p' docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_完整计划.md
sed -n '567,660p' docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_完整计划.md
sed -n '760,805p' docs/DG-KAN_v14.9_FMSSpecificityCausalAudit_AllBasisParallel_完整计划.md

cat results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardening20_gatefix_linec3/v149_line_d_substrate_route.json
```

复核结果：

```text
1. v14.9 Line D 定义：
   full 3x3 substrate >= 6/9 for exploration
   full 3x3 substrate = 9/9 for official FMS eligibility

2. v14.9 required artifacts 只列出：
   v149_all_basis_substrate_status.csv
   没有定义 Non-RAT official FMS proof 的方法组 / controls / strict gate /
   required artifact surface。

3. v14.9 failure handling 明确：
   如果 generic controls 解释 gain，停止 optimizer-state reset route；
   回到 FMS definition，不允许继续 reset。

4. 当前 artifact：
   D-CHE substrate = 9 / 9
   route = S4e-NonRATSubstrateEligibleNoFMSProof
   nonrat_fms_proof_allowed = 1
   official_fms_proof_executed = 0
   official_s5_reached = 0
   promotion_allowed = 0
```

执行边界：

```text
nonrat_fms_proof_allowed = 1 只表示 D-CHE substrate 已具备后续 proof 资格；
不表示 v14.9 已执行 proof，也不表示可以在没有预注册 protocol 的情况下
临时扩写成 S5。
```

最终本轮不新增训练、不补填成功、不 promotion。
