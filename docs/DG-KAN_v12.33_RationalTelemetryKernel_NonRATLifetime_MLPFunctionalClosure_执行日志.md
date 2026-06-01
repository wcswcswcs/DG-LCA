# DG-KAN v12.33 RationalTelemetryKernel NonRATLifetime MLPFunctionalClosure 执行日志

生成时间：2026-05-27（Asia/Singapore）

本日志记录实际执行的命令、修改文件、artifact 路径与复现要点；不补写未执行命令，不把 smoke/diagnostic 写成 official evidence。

## 1. 计划入口

计划文件：

```text
docs/DG-KAN_v12.33_RationalTelemetryKernel_NonRATLifetime_MLPFunctionalClosure_完整计划.md
```

目标：

```text
1. Rational internal telemetry kernel / task / AUC / LineC / tail closure。
2. Non-RAT exact fused kernel 后的 incremental memory lifetime repair。
3. MLP functional M-I closure/no-go。
```

## 2. 执行记录

### 2.1 初始审计

已读取并对齐：

```text
docs/DG-KAN_v12.33_RationalTelemetryKernel_NonRATLifetime_MLPFunctionalClosure_完整计划.md
dgkan/diagnostics/basis_workspace.py
dgkan/functional/mlp_functional.py
experiments/run_v1231_basis_kernel_workspace.py
experiments/run_v1231_rational_auc_hardening.py
experiments/run_v1232_finalize_rational_tail_kernel.py
experiments/run_v1230_mlp_functional.py
```

待执行命令会按实际运行顺序继续追加。

### 2.2 代码修改与语法检查

修改文件：

```text
dgkan/diagnostics/basis_workspace.py
dgkan/functional/mlp_functional.py
experiments/run_v1231_basis_kernel_workspace.py
experiments/run_v1231_rational_auc_hardening.py
experiments/run_v1233_finalize_rational_telemetry_lifetime.py
```

说明：

```text
1. 新增 v12.33 basis registry、Rational telemetry helper、Non-RAT lifetime helper。
2. 新增 M-I1..M-I4 loss-agnostic MLP functional source。
3. workspace/AUC runner 增加 --candidate-registry v1233。
4. 新增 v12.33 finalizer。
```

默认 `python` import torch 失败：

```text
ModuleNotFoundError: No module named 'torch'
```

因此后续实验与 import audit 使用项目既有环境：

```text
conda run -n kan python
```

语法检查：

```bash
conda run -n kan python -m py_compile \
  dgkan/diagnostics/basis_workspace.py \
  dgkan/functional/mlp_functional.py \
  experiments/run_v1231_basis_kernel_workspace.py \
  experiments/run_v1231_rational_auc_hardening.py \
  experiments/run_v1233_finalize_rational_telemetry_lifetime.py \
  experiments/run_v1230_mlp_functional.py
```

结果：

```text
py_compile pass
```

Import/provenance audit：

```bash
conda run -n kan python -c "from dgkan.diagnostics.basis_workspace import V1233_BASIS_CANDIDATES; from dgkan.functional.mlp_functional import SOURCE_CANDIDATES, CONTROL_IDS; import experiments.run_v1218_b320_label_free_ablation as linea; import experiments.run_v1226_label_free_only_bridge as lfbridge; ..."
```

结果：

```text
basis_candidates = 18
basis_exact_kernel_sum = 7
rational_candidates = 8
nonrat_candidates = 10
mi_candidates = 4
controls = 5
adyn = 5
adyn_uses_y_for_stats = 0
adyn_forbidden = 0
```

## 3. Smoke checks

### 3.1 Basis workspace smoke

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --candidate-registry v1233 --strict-v1232-gate \
  --run-id v1233_basis_smoke \
  --artifact-prefix v1233_basis_smoke \
  --out-dir results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/smoke \
  --device cuda:0 \
  --datasets MNIST --seeds 0 \
  --candidates D-RAT24-DenDerivativeTelemetryKernel,D-CHE12-LifetimeRecomputeBackward-K3 \
  --train-size 64 --val-size 32 --batch-size 32 \
  --hardening-epochs 1 \
  --workspace-warmup-steps 1 --workspace-profile-steps 1 \
  --linec-seeds 12319500 --linec-batch-size 16 \
  --no-download
```

结果：

```text
workspace_rows = 2
workspace_gate_pass_rows = 1
workspace_strong_gate_pass_rows = 1
hardening_rows = 2
hardening_executed_rows = 1
family_near_pass_rows = 0
```

### 3.2 Rational telemetry smoke

```bash
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --candidate-registry v1233 \
  --out-dir results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/smoke \
  --artifact-prefix v1233_rational_telemetry_smoke \
  --run-id v1233_rational_telemetry_smoke \
  --workspace-csv results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/smoke/v1233_basis_smoke_workspace_truth.csv \
  --device cuda:0 \
  --datasets MNIST --seeds 0 \
  --candidates D-RAT24-DenDerivativeTelemetryKernel \
  --train-size 64 --val-size 32 --batch-size 32 \
  --epochs 1 --lr 0.0015 \
  --linec-seeds 12319500 --linec-batch-size 16 \
  --no-download
```

结果：

```text
candidate_rows = 1
summary_rows = 1
trajectory_rows = 2
linec_rows = 1
candidate_auc_near_pass_v1233_rows = 0
```

### 3.3 MLP M-I functional smoke

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py \
  --out-dir results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/smoke \
  --artifact-prefix v1233_mlp_functional_smoke \
  --device cuda:0 \
  --datasets MNIST --seeds 0 --train-seed-bases 12330400 \
  --candidates M-I1-CloneProbeCovarianceUpperBound \
  --train-size 64 --val-size 32 \
  --epochs 1 --batch-size 32 --functional-batch 32 \
  --linec-batch-size 16 --linec-seeds 12309500,12310600 \
  --no-download
```

结果：

```text
candidate_rows = 1
control_rows = 6
linec_rows = 4
gate_rows = 1
any_exploration_pass = 0
any_official_pass = 0
```

### 3.4 A-DYN smoke

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/smoke \
  --artifact-prefix v1233_adyn_monitor_smoke \
  --run-id v1233_adyn_monitor_smoke \
  --summary-stage V1233_ADYN_MONITOR_SMOKE_SUMMARY \
  --route-stage V1233_ADYN_MONITOR_SMOKE_ROUTE \
  --result-scope v1233_adyn_monitor_smoke \
  --datasets MNIST --seeds 0 \
  --ablation-ids A-DYN1-LearnableSignalFrameWarmup,A-DYN2-EarlySelfPredictiveFrame \
  --train-size 64 --val-size 32 --test-size 32 \
  --epochs 1 --batch-size 32 \
  --measure-linec 1 --linec-batch-size 16 --linec-sketch-dim 4 \
  --no-download
```

结果：

```text
rows = 4
summary_rows = 4
label_free_ablation_available_count = 2
```

## 4. Official Line D workspace / exact / lifetime audit

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --candidate-registry v1233 --strict-v1232-gate \
  --run-id v1233_basis_workspace_official \
  --artifact-prefix v1233_basis \
  --out-dir results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233 \
  --device cuda:0 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --candidates D-RAT24-DenDerivativeTelemetryKernel,D-RAT25-DenDerivativeTelemetryRecomputeBackward,D-RAT26-TangentTrustRegionNoCE,D-RAT27-DenSlopeGuardNoCE,D-RAT28-GroupDiversityPreservingRational,D-RAT29-LineCStableTangentMix,D-RAT30-LowMemoryTelemetryStrong,D-RAT31-TelemetryAblationControl,D-CHE12-LifetimeRecomputeBackward-K3,D-CHE13-FusedReadoutGradNoMaterialize-K3,D-CHE14-OptimizerStateLifetimeReuse-K3,D-CHE15-FullStepNoMaterialize-K3,D-FOU12-LifetimeRecomputeBackward-K2,D-FOU13-FusedReadoutGradNoMaterialize-K2,D-FOU14-SincosSharedWorkspace-K2,D-FOU15-FullStepNoMaterialize-K2,D-RBF11-CompactExpressionRepair-Monitor,D-WAV10-HatWaveletLifetimeRepair-Monitor \
  --train-size 512 --val-size 256 --batch-size 128 \
  --hardening-epochs 3 \
  --workspace-warmup-steps 5 --workspace-profile-steps 20 \
  --linec-seeds 12319500,12320600,12321600 --linec-batch-size 32 \
  --no-download
```

结果：

```text
workspace_rows = 162
workspace_gate_pass_rows = 72
workspace_strong_gate_pass_rows = 27
hardening_rows = 162
hardening_executed_rows = 72
family_near_pass_rows = 0
```

## 5. Rational telemetry / AUC official triage

### 5.1 lr=0.0015

```bash
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --candidate-registry v1233 \
  --out-dir results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233 \
  --artifact-prefix v1233_rational_telemetry_lr15 \
  --run-id v1233_rational_telemetry_lr15 \
  --workspace-csv results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_basis_workspace_truth.csv \
  --device cuda:0 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --candidates D-RAT24-DenDerivativeTelemetryKernel,D-RAT25-DenDerivativeTelemetryRecomputeBackward,D-RAT26-TangentTrustRegionNoCE,D-RAT27-DenSlopeGuardNoCE,D-RAT28-GroupDiversityPreservingRational,D-RAT29-LineCStableTangentMix,D-RAT30-LowMemoryTelemetryStrong,D-RAT31-TelemetryAblationControl \
  --train-size 512 --val-size 256 --batch-size 128 \
  --epochs 3 --lr 0.0015 \
  --linec-seeds 12319500,12320600,12321600 --linec-batch-size 32 \
  --no-download
```

结果：

```text
candidate_rows = 72
summary_rows = 8
trajectory_rows = 243
linec_rows = 216
candidate_auc_near_pass_v1233_rows = 0
candidate_auc_official_pass_v1233_rows = 0
```

### 5.2 lr=0.002

同 5.1，仅修改：

```text
--artifact-prefix v1233_rational_telemetry_lr20
--run-id v1233_rational_telemetry_lr20
--lr 0.002
```

结果：

```text
candidate_rows = 72
summary_rows = 8
trajectory_rows = 243
linec_rows = 216
candidate_auc_near_pass_v1233_rows = 0
candidate_auc_official_pass_v1233_rows = 0
```

## 6. MLP M-I functional closure

执行三次，window=3/5/10。window=3 命令如下；window=5/10 仅修改 `--artifact-prefix` 与 `--epochs`：

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py \
  --out-dir results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233 \
  --artifact-prefix v1233_mlp_functional_w3 \
  --device cuda:0 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-seed-bases 12330400,12331400,12332400 \
  --candidates M-I1-CloneProbeCovarianceUpperBound,M-I2-PrecommitUnlabeledResponseStability,M-I3-ControlResidualizedMicroProbe,M-I4-ArchitectureNeutralSNRTransport \
  --train-size 512 --val-size 256 \
  --epochs 3 --batch-size 128 --functional-batch 128 \
  --linec-batch-size 32 \
  --linec-seeds 12309500,12310600,12311600,12312600,12313600 \
  --no-download
```

窗口结果：

```text
window=3: candidate_rows=108, control_rows=648, linec_rows=1080, any_exploration_pass=0, any_official_pass=0
window=5: candidate_rows=108, control_rows=648, linec_rows=1080, any_exploration_pass=0, any_official_pass=0
window=10: candidate_rows=108, control_rows=648, linec_rows=1080, any_exploration_pass=0, any_official_pass=0
```

合并命令：

```bash
conda run -n kan python -c "import csv, pathlib; out=pathlib.Path('results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233'); windows=['w3','w5','w10']; mapping={'candidates':'v1233_mlp_functional_candidates.csv','controls':'v1233_mlp_functional_controls.csv','linec':'v1233_mlp_functional_linec.csv'}; ..."
```

合并结果：

```text
v1233_mlp_functional_candidates.csv rows = 324
v1233_mlp_functional_controls.csv rows = 1944
v1233_mlp_functional_linec.csv rows = 3240
```

## 7. A-DYN monitor

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233 \
  --artifact-prefix v1233_adyn_monitor \
  --run-id v1233_adyn_monitor \
  --summary-stage V1233_ADYN_MONITOR_SUMMARY \
  --route-stage V1233_ADYN_MONITOR_ROUTE \
  --result-scope v1233_adyn_monitor \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --ablation-ids A-DYN1-LearnableSignalFrameWarmup,A-DYN2-EarlySelfPredictiveFrame,A-DYN3-OptimizerObservableFrameRefresh,A-DYN4-OvercompleteRankGuardFrame,A-DYN5-RoleEnergyBalancedFHQMonitor \
  --train-size 512 --val-size 256 --test-size 256 \
  --epochs 3 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 32 --linec-sketch-dim 8 \
  --no-download
```

结果：

```text
rows = 63
summary_rows = 7
label_free_ablation_available_count = 5
```

## 8. First finalizer pass

```bash
conda run -n kan python experiments/run_v1233_finalize_rational_telemetry_lifetime.py
```

结果：

```text
route = R2-RationalTailTaskLineCNotColocated
required_artifact_missing_count = 0
rational_auc_near_pass_count = 0
nonrat_s3_lifetime_count = 0
mlp_functional_aggregate_pass_rows = 0
```

## 9. Rational output-geometry fallback repair

触发原因：

```text
Rational telemetry 可用，但 CEp99/AUC/task/LineC 仍未同位；
den/rprime telemetry 本身未解释 tail，因此按计划尝试 label-free output entropy/logit geometry proxy。
```

新增 repair candidates：

```text
D-RAT32-LogitSpectrumTelemetryRepairNoCE
D-RAT33-LogitNormAnchorTelemetryRepairNoCE
```

代码修改：

```text
dgkan/diagnostics/basis_workspace.py
experiments/run_v1233_finalize_rational_telemetry_lifetime.py
```

语法检查：

```bash
conda run -n kan python -m py_compile dgkan/diagnostics/basis_workspace.py experiments/run_v1233_finalize_rational_telemetry_lifetime.py
```

结果：

```text
py_compile pass
```

Repair workspace：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --candidate-registry v1233 --strict-v1232-gate \
  --run-id v1233_rational_output_geometry_repair \
  --artifact-prefix v1233_rational_output_geometry_repair \
  --out-dir results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233 \
  --device cuda:0 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --candidates D-RAT32-LogitSpectrumTelemetryRepairNoCE,D-RAT33-LogitNormAnchorTelemetryRepairNoCE \
  --train-size 512 --val-size 256 --batch-size 128 \
  --hardening-epochs 3 \
  --workspace-warmup-steps 5 --workspace-profile-steps 20 \
  --linec-seeds 12319500,12320600,12321600 --linec-batch-size 32 \
  --no-download
```

结果：

```text
workspace_rows = 18
workspace_gate_pass_rows = 18
workspace_strong_gate_pass_rows = 0
hardening_executed_rows = 18
```

Repair AUC：

```bash
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --candidate-registry v1233 \
  --out-dir results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233 \
  --artifact-prefix v1233_rational_telemetry_repair \
  --run-id v1233_rational_telemetry_repair_lr15 \
  --workspace-csv results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_rational_output_geometry_repair_workspace_truth.csv \
  --device cuda:0 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --candidates D-RAT32-LogitSpectrumTelemetryRepairNoCE,D-RAT33-LogitNormAnchorTelemetryRepairNoCE \
  --train-size 512 --val-size 256 --batch-size 128 \
  --epochs 3 --lr 0.0015 \
  --linec-seeds 12319500,12320600,12321600 --linec-batch-size 32 \
  --no-download
```

结果：

```text
candidate_rows = 18
summary_rows = 2
trajectory_rows = 81
linec_rows = 54
candidate_auc_near_pass_v1233_rows = 0
```

## 10. Finalizer

```bash
conda run -n kan python experiments/run_v1233_finalize_rational_telemetry_lifetime.py
```

最终结果：

```text
route = R2-RationalTailTaskLineCNotColocated
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_rows = 180
basis_workspace_pass_count = 90
basis_workspace_strong_pass_count = 27
rational_workspace_pass_count = 90
rational_workspace_strong_pass_count = 27
rational_telemetry_available_rows = 162
rational_auc_near_pass_count = 0
nonrat_s3_lifetime_count = 0
mlp_functional_candidate_rows = 324
mlp_functional_control_rows = 1944
mlp_functional_linec_rows = 3240
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_sha256 = 302eadebcc7cab674a659ae64142f0b48eb25f3eb33eb4acc90d78d2894b4ed3
```

## 11. 用户再次追问后的 stop-contract 复核

复核命令：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_route_decision.json')
d = json.loads(p.read_text())
keys = [
    'route','minimum_success','official_success_reached','promotion_allowed',
    'final_stop_allowed','hard_compute_budget_exhausted','fallback_depth',
    'fallback_all_executed','required_artifact_missing_count',
    'basis_workspace_pass_count','basis_workspace_strong_pass_count',
    'rational_workspace_pass_count','rational_workspace_strong_pass_count',
    'rational_telemetry_available_rows','rational_auc_near_pass_count',
    'nonrat_s3_lifetime_count','mlp_functional_aggregate_pass_rows',
    'mlp_functional_no_go_current_family','line_a_near_anchor_pass_count',
    'provenance_violation_count',
]
for k in keys:
    print(f'{k}={d.get(k)}')
PY
```

结果：

```text
route = R2-RationalTailTaskLineCNotColocated
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 90
basis_workspace_strong_pass_count = 27
rational_workspace_pass_count = 90
rational_workspace_strong_pass_count = 27
rational_telemetry_available_rows = 162
rational_auc_near_pass_count = 0
nonrat_s3_lifetime_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

说明：

```text
本次只做 stop-contract 复核，没有启动新增训练。
v12.33 计划中的不允许 final stop 条件已经闭合：
1. Rational telemetry artifact 已生成。
2. denominator / derivative / tangent autopsy 已执行。
3. Non-RAT lifetime waterfall 已生成。
4. MLP matched-control closure 已完成。
5. hard_compute_budget_exhausted=1, fallback_all_executed=1, final_stop_allowed=1。
```

## 12. 用户再次追问后的 stop-contract 复核 2

复核命令：

```bash
python - <<'PY'
import json
from pathlib import Path
plan_bad = Path('docs/DG-docs/DG-KAN_v12.33_RationalTelemetryKernel_NonRATLifetime_MLPFunctionalClosure_完整计划.md')
plan = Path('docs/DG-KAN_v12.33_RationalTelemetryKernel_NonRATLifetime_MLPFunctionalClosure_完整计划.md')
route = Path('results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_route_decision.json')
print(f'given_plan_exists={plan_bad.exists()}')
print(f'actual_plan_exists={plan.exists()}')
print(f'route_exists={route.exists()}')
d = json.loads(route.read_text())
keys = [
    'route','minimum_success','official_success_reached','promotion_allowed',
    'final_stop_allowed','hard_compute_budget_exhausted','fallback_depth',
    'fallback_all_executed','required_artifact_missing_count',
    'basis_workspace_pass_count','basis_workspace_strong_pass_count',
    'rational_workspace_pass_count','rational_workspace_strong_pass_count',
    'rational_telemetry_available_rows','rational_auc_near_pass_count',
    'nonrat_s3_lifetime_count','mlp_functional_aggregate_pass_rows',
    'mlp_functional_no_go_current_family','line_a_near_anchor_pass_count',
    'provenance_violation_count',
]
for k in keys:
    print(f'{k}={d.get(k)}')
PY
```

结果：

```text
given_plan_exists = False
actual_plan_exists = True
route_exists = True
route = R2-RationalTailTaskLineCNotColocated
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 90
basis_workspace_strong_pass_count = 27
rational_workspace_pass_count = 90
rational_workspace_strong_pass_count = 27
rational_telemetry_available_rows = 162
rational_auc_near_pass_count = 0
nonrat_s3_lifetime_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

说明：

```text
本次只做 stop-contract 复核，没有启动新增训练。
用户给出的 docs/DG-docs/... 路径不存在；实际计划文件在 docs/ 下。
最终 route 与上一轮一致，允许 final stop。
```

## 13. 用户再次追问后的 stop-contract 复核 3

复核命令：

```bash
python - <<'PY'
import json
from pathlib import Path
bad = Path('docs/DG-docs/DG-KAN_v12.33_RationalTelemetryKernel_NonRATLifetime_MLPFunctionalClosure_完整计划.md')
plan = Path('docs/DG-KAN_v12.33_RationalTelemetryKernel_NonRATLifetime_MLPFunctionalClosure_完整计划.md')
route_path = Path('results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_route_decision.json')
print('given_plan_exists=', bad.exists())
print('actual_plan_exists=', plan.exists())
print('route_exists=', route_path.exists())
d = json.loads(route_path.read_text())
keys = [
    'route','minimum_success','official_success_reached','promotion_allowed',
    'final_stop_allowed','hard_compute_budget_exhausted','fallback_depth',
    'fallback_all_executed','required_artifact_missing_count',
    'rational_telemetry_available_rows','rational_auc_near_pass_count',
    'nonrat_s3_lifetime_count','mlp_functional_aggregate_pass_rows',
    'mlp_functional_no_go_current_family','line_a_near_anchor_pass_count',
    'provenance_violation_count',
]
for k in keys:
    print(f'{k}={d.get(k)}')
PY
```

结果：

```text
given_plan_exists = False
actual_plan_exists = True
route_exists = True
route = R2-RationalTailTaskLineCNotColocated
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
rational_telemetry_available_rows = 162
rational_auc_near_pass_count = 0
nonrat_s3_lifetime_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

说明：

```text
本次没有启动新增训练，没有新增 artifact 指标。
仅复核最终 route 与计划 stop 条件；结论与上一轮一致。
```

## 14. 用户再次追问后的 stop-contract 复核 4

复核命令：

```bash
python - <<'PY'
import json
from pathlib import Path
bad = Path('docs/DG-docs/DG-KAN_v12.33_RationalTelemetryKernel_NonRATLifetime_MLPFunctionalClosure_完整计划.md')
plan = Path('docs/DG-KAN_v12.33_RationalTelemetryKernel_NonRATLifetime_MLPFunctionalClosure_完整计划.md')
route_path = Path('results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_route_decision.json')
print('given_plan_exists=', bad.exists())
print('actual_plan_exists=', plan.exists())
print('route_exists=', route_path.exists())
d = json.loads(route_path.read_text())
for k in [
    'route','minimum_success','official_success_reached','promotion_allowed',
    'final_stop_allowed','hard_compute_budget_exhausted','fallback_depth',
    'fallback_all_executed','required_artifact_missing_count',
    'basis_workspace_pass_count','basis_workspace_strong_pass_count',
    'rational_telemetry_available_rows','rational_auc_near_pass_count',
    'nonrat_s3_lifetime_count','mlp_functional_aggregate_pass_rows',
    'mlp_functional_no_go_current_family','line_a_near_anchor_pass_count',
    'provenance_violation_count',
]:
    print(f'{k}={d.get(k)}')
PY
```

结果：

```text
given_plan_exists = False
actual_plan_exists = True
route_exists = True
route = R2-RationalTailTaskLineCNotColocated
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 90
basis_workspace_strong_pass_count = 27
rational_telemetry_available_rows = 162
rational_auc_near_pass_count = 0
nonrat_s3_lifetime_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

说明：

```text
本次没有启动新增训练，没有新增 artifact 指标，没有修改代码或 gate。
结论与上一轮一致。
```
