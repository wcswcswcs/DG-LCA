# DG-KAN v12.28 LabelFreeBaseRecovery FunctionalReentry ClassicNoBSpline 执行日志

生成时间：2026-05-26（Asia/Singapore）

本日志只记录实际执行过的命令、文件与产物路径；不补写未执行命令。

## 1. 计划入口

计划文档：

```text
docs/DG-KAN_v12.28_LabelFreeBaseRecovery_FunctionalReentry_ClassicNoBSpline_完整计划.md
```

结果目录：

```text
results/v12_28_label_free_base_recovery_functional_reentry/official_v1228
```

## 2. 本轮代码修改记录

修改文件：

```text
dgkan/models/fc_purekan_primitives.py
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1224_classic_hardening.py
experiments/run_v1228_label_free_base_reentry.py
experiments/run_v1228_finalize_label_free_base_recovery.py
```

新增日志文件：

```text
docs/DG-KAN_v12.28_LabelFreeBaseRecovery_FunctionalReentry_ClassicNoBSpline_执行日志.md
docs/DG-KAN_v12.28_LabelFreeBaseRecovery_FunctionalReentry_ClassicNoBSpline_实验结果复盘.md
```

## 3. 语法检查与 registry/provenance audit

执行命令：

```bash
python -m py_compile dgkan/models/fc_purekan_primitives.py experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1224_classic_hardening.py experiments/run_v1228_label_free_base_reentry.py experiments/run_v1228_finalize_label_free_base_recovery.py
```

结果：

```text
py_compile pass
```

执行命令：

```bash
conda run -n kan python -c "from experiments.run_v1218_b320_label_free_ablation import ablation_specs; from experiments.run_v1226_label_free_only_bridge import forbidden_token_present; from experiments.run_v1228_label_free_base_reentry import v1228_candidates; ..."
```

结果摘要：

```text
missing_base_ids = []
A-F1a..A-F4d / A-R1..A-R5: uses_y_for_stats=0, forbidden=0
F28-S1..F28-S6: forbidden=0
```

## 4. Smoke 与 blocker 修复

首次 smoke 命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_28_label_free_base_recovery_functional_reentry/official_v1228 \
  --artifact-prefix v1228_label_free_frame_smoke \
  --run-id v1228_label_free_frame_smoke \
  --summary-stage V1228_LABEL_FREE_FRAME_SMOKE_SUMMARY \
  --route-stage V1228_LABEL_FREE_FRAME_SMOKE_ROUTE \
  --result-scope v1228_label_free_frame_smoke \
  --datasets MNIST --seeds 0 \
  --ablation-ids A-F1b-SRHTP-LearnableFrameWarmup,A-F1c-BlockOrthoP-LearnableFrameWarmup,A-F1d-RandomLowCoherenceP-LearnableFrameWarmup,A-F4c-OvercompleteSparseP-h192-LF \
  --train-size 64 --val-size 32 --test-size 32 \
  --epochs 1 --batch-size 32 \
  --measure-linec 1 --linec-batch-size 16 --linec-sketch-dim 4 \
  2>&1 | tee results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/logs/v1228_label_free_frame_smoke.log
```

失败：

```text
ValueError: learnable-P workspace requires a learnable quad_proj
```

定位命令：

```bash
for id in A-F1b-SRHTP-LearnableFrameWarmup A-F1c-BlockOrthoP-LearnableFrameWarmup A-F1d-RandomLowCoherenceP-LearnableFrameWarmup A-F4c-OvercompleteSparseP-h192-LF; do ...; done
```

定位结果：

```text
A-F1b/A-F1c/A-F1d pass
A-F4c fail
```

修复：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

将 A-F4c 的 init_variant 从：

```text
sparsek8p_pupdateevery4_directreadinit090_quadramp050
```

改为：

```text
sparsek8p_fixedp_directreadinit090_quadramp050
```

理由：sparseP 实现中 `quad_proj` 是 fixed sparse frame buffer，不能走 dense learnable-P workspace。

复核命令：

```bash
python -m py_compile experiments/run_v1218_b320_label_free_ablation.py
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --out-dir results/v12_28_label_free_base_recovery_functional_reentry/official_v1228 --artifact-prefix v1228_single_smoke_F4c_fixedp --run-id v1228_single_smoke_F4c_fixedp --summary-stage V1228_SINGLE_SMOKE_SUMMARY --route-stage V1228_SINGLE_SMOKE_ROUTE --result-scope v1228_single_smoke --datasets MNIST --seeds 0 --ablation-ids A-F4c-OvercompleteSparseP-h192-LF --train-size 64 --val-size 32 --test-size 32 --epochs 1 --batch-size 32 --measure-linec 0
```

结果：

```text
A-F4c pass
```

组合 smoke 复核命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_28_label_free_base_recovery_functional_reentry/official_v1228 \
  --artifact-prefix v1228_label_free_frame_smoke \
  --run-id v1228_label_free_frame_smoke \
  --summary-stage V1228_LABEL_FREE_FRAME_SMOKE_SUMMARY \
  --route-stage V1228_LABEL_FREE_FRAME_SMOKE_ROUTE \
  --result-scope v1228_label_free_frame_smoke \
  --datasets MNIST --seeds 0 \
  --ablation-ids A-F1b-SRHTP-LearnableFrameWarmup,A-F1c-BlockOrthoP-LearnableFrameWarmup,A-F1d-RandomLowCoherenceP-LearnableFrameWarmup,A-F4c-OvercompleteSparseP-h192-LF \
  --train-size 64 --val-size 32 --test-size 32 \
  --epochs 1 --batch-size 32 \
  --measure-linec 1 --linec-batch-size 16 --linec-sketch-dim 4 \
  2>&1 | tee results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/logs/v1228_label_free_frame_smoke_after_sparse_fixedp.log
```

结果：

```text
rows = 6
summary_rows = 6
best_label_free_smoke_candidate = A-F1b-SRHTP-LearnableFrameWarmup
```

## 5. Line A scout

执行命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_28_label_free_base_recovery_functional_reentry/official_v1228 \
  --artifact-prefix v1228_label_free_frame_scout \
  --run-id v1228_label_free_frame_scout \
  --summary-stage V1228_LABEL_FREE_FRAME_SCOUT_SUMMARY \
  --route-stage V1228_LABEL_FREE_FRAME_SCOUT_ROUTE \
  --result-scope v1228_label_free_frame_scout \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-F1a-OrthoP-LearnableFrameWarmup,A-F1b-SRHTP-LearnableFrameWarmup,A-F1c-BlockOrthoP-LearnableFrameWarmup,A-F1d-RandomLowCoherenceP-LearnableFrameWarmup,A-F2a-RoleBalancedFHQ-LF,A-F2b-QuadDirectBalanceRamp-LF,A-F2c-BranchGainFrozenEarly-LF,A-F2d-RoleEnergyEqualizedWarmup-LF,A-F3a-UpdateSpectrumFrame-LF,A-F3b-MomentumCovFrame-LF,A-F3c-GradientNormOnlyFrame-LF,A-F3d-AdamSecondMomentFrame-LF,A-F4a-OvercompleteSRHTP-h192-LF,A-F4b-OvercompleteBlockP-h192-LF,A-F4c-OvercompleteSparseP-h192-LF,A-F4d-OvercompleteThenPruneP-LF \
  --train-size 512 --val-size 256 --test-size 256 \
  --epochs 3 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 32 --linec-sketch-dim 8 \
  2>&1 | tee results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/logs/v1228_linea_frame_scout.log
```

结果：

```text
rows = 162
summary_rows = 18
best_label_free_candidate = A-F4d-OvercompleteThenPruneP-LF
```

## 6. Line A hardening top candidates

按 scout family winner、task/LineC/AUC near-miss 选择：

```text
A-F4d-OvercompleteThenPruneP-LF
A-F2c-BranchGainFrozenEarly-LF
A-F1d-RandomLowCoherenceP-LearnableFrameWarmup
A-F1b-SRHTP-LearnableFrameWarmup
A-F3b-MomentumCovFrame-LF
A-F3a-UpdateSpectrumFrame-LF
A-F2d-RoleEnergyEqualizedWarmup-LF
A-F4a-OvercompleteSRHTP-h192-LF
```

执行命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_28_label_free_base_recovery_functional_reentry/official_v1228 \
  --artifact-prefix v1228_label_free_frame_hardening \
  --run-id v1228_label_free_frame_hardening \
  --summary-stage V1228_LABEL_FREE_FRAME_HARDENING_SUMMARY \
  --route-stage V1228_LABEL_FREE_FRAME_HARDENING_ROUTE \
  --result-scope v1228_label_free_frame_hardening_top8 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-F4d-OvercompleteThenPruneP-LF,A-F2c-BranchGainFrozenEarly-LF,A-F1d-RandomLowCoherenceP-LearnableFrameWarmup,A-F1b-SRHTP-LearnableFrameWarmup,A-F3b-MomentumCovFrame-LF,A-F3a-UpdateSpectrumFrame-LF,A-F2d-RoleEnergyEqualizedWarmup-LF,A-F4a-OvercompleteSRHTP-h192-LF \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 8 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 64 --linec-sketch-dim 24 \
  2>&1 | tee results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/logs/v1228_linea_frame_hardening.log
```

结果：

```text
rows = 90
summary_rows = 10
best_label_free_candidate = A-F2c-BranchGainFrozenEarly-LF
```

## 7. Line A failure-specific repair

因 hardening 没有 near-anchor，执行 A-R1..A-R5：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_28_label_free_base_recovery_functional_reentry/official_v1228 \
  --artifact-prefix v1228_label_free_frame_repair \
  --run-id v1228_label_free_frame_repair \
  --summary-stage V1228_LABEL_FREE_FRAME_REPAIR_SUMMARY \
  --route-stage V1228_LABEL_FREE_FRAME_REPAIR_ROUTE \
  --result-scope v1228_label_free_frame_repair_depth3 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-R1-TaskGoodLineCQuadWarm,A-R2-LineCGoodTaskDirectWarm,A-R3-AUCTrajectoryRelease,A-R4-UpdateSpectrumWeakRefresh,A-R5-OvercompleteRoleBalance \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 8 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 64 --linec-sketch-dim 24 \
  2>&1 | tee results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/logs/v1228_linea_frame_repair.log
```

结果：

```text
rows = 63
summary_rows = 7
best_label_free_candidate = A-R2-LineCGoodTaskDirectWarm
```

## 8. Functional shadow re-entry

Line A near-anchor 为 0，因此本步只作为 shadow diagnostic，不允许 promotion。

执行命令：

```bash
conda run -n kan python experiments/run_v1228_label_free_base_reentry.py \
  --out-dir results/v12_28_label_free_base_recovery_functional_reentry/official_v1228 \
  --artifact-prefix v1228_functional_shadow \
  --base-candidates A-F2c-BranchGainFrozenEarly-LF,A-R2-LineCGoodTaskDirectWarm,A-F1b-SRHTP-LearnableFrameWarmup \
  --datasets KMNIST --seeds 0 \
  --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 \
  --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12280400,12281400,12282400 \
  --linec-seeds 12289500,12290600,12291600,12292600,12293600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --ref-mode bootstrap_mom --ref-seed-base 12282525 --train-seed-salt-override 39 \
  2>&1 | tee results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/logs/v1228_functional_shadow.log
```

结果：

```text
candidate_rows = 54
aggregate_rows = 18
any_exploration_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
promotion_allowed = 0
```

## 9. Classic Rational D39-D43 repair

执行命令：

```bash
conda run -n kan python experiments/run_v1224_classic_hardening.py \
  --out-dir results/v12_28_label_free_base_recovery_functional_reentry/official_v1228 \
  --artifact-prefix v1228_classic_rational_repair \
  --run-id v1228_classic_rational_repair_D39_D43 \
  --families D39-RationalGroupWorkspaceRecomputeV2,D40-RationalReadoutGradChunked,D41-RationalHiddenResLowMemV2,D42-RationalDenStateFP16PlusCheckpoint,D43-RationalPairNormLineCNoExtraMem \
  --datasets MNIST,Fashion-MNIST \
  --seeds 0 \
  --train-size 1024 --val-size 512 --epochs 8 --batch-size 128 \
  --linec-batch-size 64 --linec-sketch-dim 24 \
  2>&1 | tee results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/logs/v1228_classic_rational_repair.log
```

结果：

```text
rows = 10
summary_rows = 5
exploration_pass_rows = 0
```

## 10. Finalizer 与聚合修复

首次 finalizer 命令：

```bash
conda run -n kan python experiments/run_v1228_finalize_label_free_base_recovery.py \
  2>&1 | tee results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/logs/v1228_finalize_initial.log
```

首次发现聚合 bug：

```text
1. MLP reference summary 被误计入 Line A near-anchor。
2. functional variant / multisketch rows 被误计为 candidate rows。
3. required manifest 需要在 route/zip 写出后重算。
```

修复文件：

```text
experiments/run_v1228_finalize_label_free_base_recovery.py
```

复核命令：

```bash
python -m py_compile experiments/run_v1228_finalize_label_free_base_recovery.py && \
conda run -n kan python experiments/run_v1228_finalize_label_free_base_recovery.py \
  2>&1 | tee results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/logs/v1228_finalize_aggregate_fix.log
```

修复后结果：

```text
route = R1-LabelFreeBaseRecoveryMissing
line_a_near_anchor_pass_count = 0
functional_candidate_rows = 54
functional_aggregate_rows = 18
required_artifact_missing_count = 0
fallback_depth = 6
fallback_all_executed = 1
```

## 11. 日志写入后的最终打包复核

执行命令：

```bash
conda run -n kan python experiments/run_v1228_finalize_label_free_base_recovery.py \
  2>&1 | tee results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/logs/v1228_finalize_after_logs.log
```

最终 route 与 code packet sha 以：

```text
results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/v1228_route_decision.json
```

为准。

## 12. 用户再次追问后的 stop-contract 只读复核

复核命令：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/v1228_route_decision.json')
r=json.loads(p.read_text())
for k in ['route','minimum_success','official_success_reached','p4_pass','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','line_a_near_anchor_pass_count','line_a_official_pass_count','line_f_exploration_gate_pass','line_f_official_gate_pass','functional_candidate_rows','functional_aggregate_rows','precommit_value_source_rows','classic_meaningful_progress_count','code_review_packet_entries','code_review_packet_sha256']:
    print(f'{k}={r.get(k)}')
PY
```

复核结果：

```text
route = R1-LabelFreeBaseRecoveryMissing
minimum_success = Minimum Success E
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_a_official_pass_count = 0
line_f_exploration_gate_pass = 0
line_f_official_gate_pass = 0
functional_candidate_rows = 54
functional_aggregate_rows = 18
precommit_value_source_rows = 54
classic_meaningful_progress_count = 0
code_review_packet_entries = 69
```

本次仅做 stop-contract 复核；没有新增实验。原因记录在实验结果复盘第 12 节：当前已经不确定继续扩展同一 A-F/A-R srhtp / lowcoherence / blockframe / rolebalance / optframe token family 是否有有效机制，继续排列 token 会变成低价值网格搜索。

日志更新后最终打包命令：

```bash
conda run -n kan python experiments/run_v1228_finalize_label_free_base_recovery.py \
  2>&1 | tee results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/logs/v1228_finalize_after_stop_contract_recheck.log
```

## 13. 用户再次追问后的 stop-contract 只读复核 2

复核命令：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/v1228_route_decision.json')
r=json.loads(p.read_text())
for k in ['route','minimum_success','official_success_reached','p4_pass','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','line_a_near_anchor_pass_count','line_a_official_pass_count','line_f_exploration_gate_pass','line_f_official_gate_pass','functional_candidate_rows','functional_aggregate_rows','precommit_value_source_rows','classic_meaningful_progress_count','code_review_packet_entries','code_review_packet_sha256']:
    print(f'{k}={r.get(k)}')
PY
```

复核结果：

```text
route = R1-LabelFreeBaseRecoveryMissing
minimum_success = Minimum Success E
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_a_official_pass_count = 0
line_f_exploration_gate_pass = 0
line_f_official_gate_pass = 0
functional_candidate_rows = 54
functional_aggregate_rows = 18
precommit_value_source_rows = 54
classic_meaningful_progress_count = 0
code_review_packet_entries = 70
```

本次仍仅做 stop-contract 复核；没有新增实验。原因不变：当前已经不确定继续扩展同一 A-F/A-R srhtp / lowcoherence / blockframe / rolebalance / optframe token family 是否有有效机制，继续排列 token 会变成低价值网格搜索。

日志更新后最终打包命令：

```bash
conda run -n kan python experiments/run_v1228_finalize_label_free_base_recovery.py \
  2>&1 | tee results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/logs/v1228_finalize_after_stop_contract_recheck2.log
```

## 14. 用户再次追问后的 stop-contract 只读复核 3

复核命令：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/v1228_route_decision.json')
r = json.loads(p.read_text())
keys = ['route','minimum_success','official_success_reached','p4_pass','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','line_a_near_anchor_pass_count','line_f_exploration_gate_pass','functional_candidate_rows','functional_aggregate_rows','precommit_value_source_rows','code_review_packet_entries','code_review_packet_sha256']
for k in keys:
    print(f'{k}={r.get(k)}')
PY
```

复核结果：

```text
route = R1-LabelFreeBaseRecoveryMissing
minimum_success = Minimum Success E
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
functional_candidate_rows = 54
functional_aggregate_rows = 18
precommit_value_source_rows = 54
code_review_packet_entries = 71
code_review_packet_sha256 = b46e23cbbf46290dffd8d902780b9e734b2f5ccf6f37640a97a5b590c05db013
```

本次仍仅做 stop-contract 复核；没有新增实验。原因不变：当前已经不确定继续扩展同一 A-F/A-R srhtp / lowcoherence / blockframe / rolebalance / optframe token family 是否有有效机制，继续排列 token 会变成低价值网格搜索。

日志更新后最终打包命令：

```bash
conda run -n kan python experiments/run_v1228_finalize_label_free_base_recovery.py \
  2>&1 | tee results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/logs/v1228_finalize_after_stop_contract_recheck3.log
```

最终打包结果：

```text
route = R1-LabelFreeBaseRecoveryMissing
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
functional_candidate_rows = 54
functional_aggregate_rows = 18
precommit_value_source_rows = 54
classic_meaningful_progress_count = 0
code_review_packet_entries = 72
code_review_packet_sha256 = 以最终 v1228_route_decision.json 为准
```
