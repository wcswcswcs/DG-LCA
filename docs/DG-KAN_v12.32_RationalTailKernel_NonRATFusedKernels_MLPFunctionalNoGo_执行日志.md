# DG-KAN v12.32 RationalTailKernel NonRATFusedKernels MLPFunctionalNoGo 执行日志

生成时间：2026-05-27（Asia/Singapore）

本日志只记录实际执行过的命令、修改过的文件、生成的 artifact 与失败原因；不把未执行步骤写成已完成。

## 1. 计划读取与执行约束

读取计划：

```bash
sed -n '1,240p' docs/DG-KAN_v12.32_RationalTailKernel_NonRATFusedKernels_MLPFunctionalNoGo_完整计划.md
sed -n '320,560p' docs/DG-KAN_v12.32_RationalTailKernel_NonRATFusedKernels_MLPFunctionalNoGo_完整计划.md
sed -n '760,900p' docs/DG-KAN_v12.32_RationalTailKernel_NonRATFusedKernels_MLPFunctionalNoGo_完整计划.md
sed -n '980,1145p' docs/DG-KAN_v12.32_RationalTailKernel_NonRATFusedKernels_MLPFunctionalNoGo_完整计划.md
```

理解到的硬约束：

```text
1. strict FC-PureKAN / classic no-BSpline active basis。
2. no teacher / distillation / loss modification / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target。
5. CE/NLL/ECE/CEp99 只能作为审计和坏化约束。
6. blocker 后必须执行计划 fallback；smoke/diagnostic/skipped row 不能写成 promotion。
```

## 2. 初始代码审计命令

```bash
rg -n "D-RAT1[6-9]|D-RAT2[0-3]|M-H|v1232|RationalTail" -S docs experiments dgkan | head -200
ls docs | rg 'v12\.32|v12\.31'
git status --short
sed -n '1,260p' dgkan/diagnostics/basis_workspace.py
sed -n '260,620p' dgkan/diagnostics/basis_workspace.py
sed -n '1,240p' dgkan/functional/mlp_functional.py
sed -n '1,240p' experiments/run_v1231_basis_kernel_workspace.py
sed -n '240,520p' experiments/run_v1231_basis_kernel_workspace.py
sed -n '1,220p' experiments/run_v1231_rational_auc_hardening.py
sed -n '220,520p' experiments/run_v1231_rational_auc_hardening.py
sed -n '1,260p' experiments/run_v1230_mlp_functional.py
sed -n '260,520p' experiments/run_v1230_mlp_functional.py
rg -n "def manual_gradient_audit|manual_gradient_audit" dgkan/models/fc_purekan_primitives.py experiments | head -40
rg -n "B7(lp|lq|ma|mb|mg|mh|il|im|is|iw|kc|kf|ki|kj)|B3e-|B3f-|B4g-|B4n-|B4w-" dgkan/models/fc_purekan_primitives.py
```

审计发现：

```text
1. v12.31 workspace core 已在 dgkan/diagnostics/basis_workspace.py。
2. v12.31/v12.30 runner 已可复用为 v12.32 编排基础，但需要 v12.32 registry、严格 workspace gate、kernel manifest/exact audit。
3. Chebyshev/Fourier 的 Triton no-materialize kernel 已存在于 dgkan/kernels 与 PrimitiveSpec 中；v12.32 需要显式审计，而不是重新命名为新实现。
4. Rational D-RAT16..23 可映射到已有 label-free rational/output-geometry PrimitiveSpec 做 tail-stability probe，但必须注明不是所有计划字面机制都已有 exact kernel。
5. M-H1..M-H3 需要新增到 dgkan.functional.mlp_functional，runner 只负责调用。
```

## 3. 代码修改与 import/provenance smoke

修改文件：

```text
dgkan/diagnostics/basis_workspace.py
dgkan/functional/mlp_functional.py
experiments/run_v1231_basis_kernel_workspace.py
experiments/run_v1231_rational_auc_hardening.py
experiments/run_v1232_finalize_rational_tail_kernel.py
docs/DG-KAN_v12.32_RationalTailKernel_NonRATFusedKernels_MLPFunctionalNoGo_执行日志.md
docs/DG-KAN_v12.32_RationalTailKernel_NonRATFusedKernels_MLPFunctionalNoGo_实验结果复盘.md
```

修改内容：

```text
1. dgkan/diagnostics/basis_workspace.py
   - 新增 V1232_BASIS_CANDIDATES：D-RAT16..D-RAT23、D-CHE10..D-CHE11、D-FOU10..D-FOU11。
   - 新增 v12.32 strict workspace gate：raw<=1.05, incremental<=1.75, step<=1.75。
   - 新增 strong gate：raw<=1.05, incremental<=1.35, step<=1.40。
   - 新增 kernel_implementation_manifest_rows / exact_kernel_audit_rows / rational_tail_metrics。

2. dgkan/functional/mlp_functional.py
   - 新增 M-H1-UnlabeledMicroProbeResponsePredictor。
   - 新增 M-H2-RandomCotangentLowRankResponseController。
   - 新增 M-H3-ActivationSpectrumGuardWithMatchedControls。

3. experiments/run_v1231_basis_kernel_workspace.py
   - 增加 --candidate-registry v1232 与 --strict-v1232-gate。
   - v1232 registry 下输出 v1232_kernel_implementation_manifest.csv 与 v1232_exact_kernel_audit.csv。

4. experiments/run_v1231_rational_auc_hardening.py
   - 增加 --candidate-registry v1232。
   - v1232 registry 下写入 rational tail unlabeled audit metrics 与 v1232 near-pass 字段。

5. experiments/run_v1232_finalize_rational_tail_kernel.py
   - 新增 v12.32 finalizer，生成 route、manifest、fallback、provenance、forbidden audit、figures、no-go boundary、next queue、code packet。
```

语法检查：

```bash
conda run -n kan python -m py_compile \
  dgkan/diagnostics/basis_workspace.py \
  dgkan/functional/mlp_functional.py \
  experiments/run_v1231_basis_kernel_workspace.py \
  experiments/run_v1231_rational_auc_hardening.py \
  experiments/run_v1230_mlp_functional.py \
  experiments/run_v1232_finalize_rational_tail_kernel.py
```

结果：

```text
py_compile pass
```

Import/provenance smoke：

```bash
conda run -n kan python -c "from dgkan.diagnostics.basis_workspace import V1232_BASIS_CANDIDATES; from dgkan.functional.mlp_functional import SOURCE_CANDIDATES, CONTROL_IDS; from experiments.run_v1218_b320_label_free_ablation import ablation_specs; from experiments.run_v1226_label_free_only_bridge import forbidden_token_present; specs=ablation_specs(784,10); adyn=[(k,v) for k,v in specs.items() if k.startswith('A-DYN')]; print({'basis_candidates': len(V1232_BASIS_CANDIDATES), 'basis_exact_kernel_sum': sum(c.exact_kernel_implemented for c in V1232_BASIS_CANDIDATES.values()), 'mh_candidates': len([c for c in SOURCE_CANDIDATES if c.startswith('M-H')]), 'controls': len(CONTROL_IDS), 'adyn': len(adyn), 'adyn_uses_y_for_stats': sum(int(v.get('uses_y_for_stats',0)) for _,v in adyn), 'adyn_forbidden': sum(int(forbidden_token_present(k, getattr(v['spec'], 'candidate_id', ''), getattr(v['spec'], 'init_variant', ''))) for k,v in adyn)})"
```

结果：

```text
{'basis_candidates': 12, 'basis_exact_kernel_sum': 4, 'mh_candidates': 3, 'controls': 5, 'adyn': 5, 'adyn_uses_y_for_stats': 0, 'adyn_forbidden': 0}
```

## 4. Smoke checks

### 4.1 Basis workspace smoke

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --candidate-registry v1232 --strict-v1232-gate \
  --run-id v1232_basis_smoke \
  --artifact-prefix v1232_basis_smoke \
  --out-dir results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/smoke \
  --device cuda:0 \
  --datasets MNIST --seeds 0 \
  --candidates D-RAT16-DenP01FloorNoCE,D-CHE10-FusedRecurrenceNoMaterialize-K3 \
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
kernel_manifest_csv = results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/smoke/v1232_basis_smoke_kernel_implementation_manifest.csv
correctness_csv = results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/smoke/v1232_basis_smoke_exact_kernel_audit.csv
```

### 4.2 MLP M-H functional smoke

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py \
  --out-dir results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/smoke \
  --artifact-prefix v1232_mlp_functional_smoke \
  --device cuda:0 \
  --datasets MNIST --seeds 0 --train-seed-bases 12320400 \
  --candidates M-H1-UnlabeledMicroProbeResponsePredictor \
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

### 4.3 A-DYN monitor smoke

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/smoke \
  --artifact-prefix v1232_adyn_monitor_smoke \
  --run-id v1232_adyn_monitor_smoke \
  --summary-stage V1232_ADYN_MONITOR_SMOKE_SUMMARY \
  --route-stage V1232_ADYN_MONITOR_SMOKE_ROUTE \
  --result-scope v1232_adyn_monitor_smoke \
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
best_label_free_candidate = A-DYN1-LearnableSignalFrameWarmup
smoke_not_official = 1
```

## 5. Line D official basis workspace / exact kernel audit

执行命令：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --candidate-registry v1232 --strict-v1232-gate \
  --run-id v1232_basis_workspace \
  --artifact-prefix v1232_basis \
  --out-dir results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232 \
  --device cuda:0 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --candidates D-RAT16-DenP01FloorNoCE,D-RAT17-RPrimeCapNoCE,D-RAT18-TangentNormTrustNoCE,D-RAT19-DenSlopeJointStabilityNoCE,D-RAT20-LogitSpectrumClipStopGrad,D-RAT21-EntropyFloorUnlabeled,D-RAT22-TopGapFloorNoLabel,D-RAT23-LogitNormEMAAnchorNoCE,D-CHE10-FusedRecurrenceNoMaterialize-K3,D-CHE11-FusedRecurrenceNoMaterialize-K4,D-FOU10-FusedSincosLowFreqK2,D-FOU11-FusedSincosLowFreqK4SharedAmp \
  --train-size 512 --val-size 256 --batch-size 128 \
  --hardening-epochs 3 \
  --workspace-warmup-steps 5 --workspace-profile-steps 20 \
  --linec-seeds 12319500,12320600,12321600 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --no-download
```

结果：

```text
workspace_rows = 108
workspace_gate_pass_rows = 72
workspace_strong_gate_pass_rows = 0
hardening_rows = 108
hardening_executed_rows = 72
family_near_pass_rows = 0
```

Family workspace 摘要：

```text
D-RAT: rows=72, workspace_pass=72, strong_pass=0, min_raw=0.7334602829162132, min_incremental=1.5938538205980066, min_step=0.9099883499461409
D-CHE: rows=18, workspace_pass=0, strong_pass=0, min_raw=1.00025389916576, min_incremental=6.923588039867109, min_step=1.1696214070169357
D-FOU: rows=18, workspace_pass=0, strong_pass=0, min_raw=0.9418389553862895, min_incremental=3.37375415282392, min_step=0.9981457309774058
```

Exact kernel audit 摘要：

```text
D-CHE10 manual_gradcheck_pass=1, max_forward_abs_error=4.470348358154297e-08, max_grad_rel_error=1.9474374823857943e-07
D-CHE11 manual_gradcheck_pass=1, max_forward_abs_error=5.960464477539063e-08, max_grad_rel_error=2.2848143998999149e-07
D-FOU10 manual_gradcheck_pass=1, max_forward_abs_error=1.816079020500183e-08, max_grad_rel_error=3.6563899357133778e-06
D-FOU11 manual_gradcheck_pass=1, max_forward_abs_error=4.551839083433151e-08, max_grad_rel_error=3.1175153480944573e-07
```

解释：Non-RAT exact kernels 的 gradcheck/A4 expression smoke 通过，但 workspace gate 未过，因此不能达成 S3。

## 6. Line D-RAT Rational tail/AUC triage

### 6.1 lr=0.0015

执行命令：

```bash
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --candidate-registry v1232 \
  --out-dir results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232 \
  --artifact-prefix v1232_rational_tail_auc_lr15 \
  --run-id v1232_rational_tail_auc_lr15 \
  --device cuda:0 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --candidates D-RAT16-DenP01FloorNoCE,D-RAT17-RPrimeCapNoCE,D-RAT18-TangentNormTrustNoCE,D-RAT19-DenSlopeJointStabilityNoCE,D-RAT20-LogitSpectrumClipStopGrad,D-RAT21-EntropyFloorUnlabeled,D-RAT22-TopGapFloorNoLabel,D-RAT23-LogitNormEMAAnchorNoCE \
  --workspace-csv results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_basis_workspace_truth.csv \
  --train-size 512 --val-size 256 --batch-size 128 \
  --epochs 3 --lr 0.0015 --weight-decay 0.001 \
  --linec-seeds 12319500,12320600,12321600 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --no-download
```

结果：

```text
candidate_rows = 72
summary_rows = 8
trajectory_rows = 243
linec_rows = 216
candidate_auc_near_pass_v1232_rows = 0
```

最接近行：

```text
D-RAT18-TangentNormTrustNoCE:
mean_delta_vs_MLP = 0.016927083333333332
worst_delta_vs_MLP = -0.01953125
max_AUC_time_ratio_vs_MLP = 1.4682377071551969
max_CEp99_delta_vs_MLP = 2.6176023483276367
LineC_pass_count = 20/27
candidate_auc_near_pass_v1232 = 0
```

### 6.2 lr=0.002

执行命令：

```bash
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --candidate-registry v1232 \
  --out-dir results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232 \
  --artifact-prefix v1232_rational_tail_auc_lr20 \
  --run-id v1232_rational_tail_auc_lr20 \
  --device cuda:0 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --candidates D-RAT16-DenP01FloorNoCE,D-RAT17-RPrimeCapNoCE,D-RAT18-TangentNormTrustNoCE,D-RAT19-DenSlopeJointStabilityNoCE,D-RAT20-LogitSpectrumClipStopGrad,D-RAT21-EntropyFloorUnlabeled,D-RAT22-TopGapFloorNoLabel,D-RAT23-LogitNormEMAAnchorNoCE \
  --workspace-csv results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_basis_workspace_truth.csv \
  --train-size 512 --val-size 256 --batch-size 128 \
  --epochs 3 --lr 0.002 --weight-decay 0.001 \
  --linec-seeds 12319500,12320600,12321600 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --no-download
```

结果：

```text
candidate_rows = 72
summary_rows = 8
trajectory_rows = 243
linec_rows = 216
candidate_auc_near_pass_v1232_rows = 0
```

最接近行：

```text
D-RAT16-DenP01FloorNoCE:
mean_delta_vs_MLP = 0.013888888888888888
worst_delta_vs_MLP = -0.015625
max_AUC_time_ratio_vs_MLP = 1.9817887704723836
max_CEp99_delta_vs_MLP = 3.3433451652526855
LineC_pass_count = 27/27
candidate_auc_near_pass_v1232 = 0
```

## 7. Line M M-H MLP functional no-go

执行命令（window=3；window=5/10 仅 `--artifact-prefix` 与 `--epochs` 分别改为 w5/5、w10/10）：

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py \
  --out-dir results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232 \
  --artifact-prefix v1232_mlp_functional_w3 \
  --device cuda:0 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-seed-bases 12320400,12321400,12322400 \
  --candidates M-H1-UnlabeledMicroProbeResponsePredictor,M-H2-RandomCotangentLowRankResponseController,M-H3-ActivationSpectrumGuardWithMatchedControls \
  --train-size 512 --val-size 256 \
  --epochs 3 --batch-size 128 --functional-batch 128 \
  --linec-batch-size 32 \
  --linec-seeds 12309500,12310600,12311600,12312600,12313600 \
  --no-download
```

窗口结果：

```text
window=3: candidate_rows=81, control_rows=486, linec_rows=810, any_exploration_pass=0, any_official_pass=0
window=5: candidate_rows=81, control_rows=486, linec_rows=810, any_exploration_pass=0, any_official_pass=0
window=10: candidate_rows=81, control_rows=486, linec_rows=810, any_exploration_pass=0, any_official_pass=0
```

合并命令：

```bash
conda run -n kan python -c "import csv, pathlib; out=pathlib.Path('results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232'); kinds=['candidates','controls','linec']; ..."
```

合并结果：

```text
v1232_mlp_functional_candidates.csv rows = 243
v1232_mlp_functional_controls.csv rows = 1458
v1232_mlp_functional_linec.csv rows = 2430
```

Finalizer 后 v12.32 strict M-H gate：

```text
M-H aggregate pass rows = 0
best mean_source_vs_control = -0.0008680555555555555
max_LineC_pass_count = 5
MLPFunctionalNoGo_CurrentLossAgnosticObservableFamily = 1
```

## 8. Line A A-DYN monitor

执行命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232 \
  --artifact-prefix v1232_adyn_monitor \
  --run-id v1232_adyn_monitor \
  --summary-stage V1232_ADYN_MONITOR_SUMMARY \
  --route-stage V1232_ADYN_MONITOR_ROUTE \
  --result-scope v1232_adyn_monitor \
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
best_label_free_candidate = A-DYN1-LearnableSignalFrameWarmup
smoke_not_official = 1
```

## 9. Finalizer

执行命令：

```bash
conda run -n kan python experiments/run_v1232_finalize_rational_tail_kernel.py
```

## 13. 用户再次追问后的 route 复核 4

执行命令：

```bash
conda run -n kan python -c "import json; from pathlib import Path; d=json.loads(Path('results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_route_decision.json').read_text()); keys=['route','minimum_success','official_success_reached','p4_pass','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_workspace_pass_count','basis_workspace_strong_pass_count','rational_workspace_pass_count','rational_workspace_strong_pass_count','rational_auc_near_pass_count','nonrat_s3_true_kernel_count','mlp_functional_aggregate_pass_rows','mlp_functional_no_go_current_family','line_a_near_anchor_pass_count','provenance_violation_count','code_review_packet_entries','code_review_packet_sha256']; print('\n'.join(f'{k} = {d.get(k)}' for k in keys))"
```

结果：

```text
route = R1-RationalTailStabilityBlocked
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 72
basis_workspace_strong_pass_count = 0
rational_workspace_pass_count = 72
rational_workspace_strong_pass_count = 0
rational_auc_near_pass_count = 0
nonrat_s3_true_kernel_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_entries = 15
code_review_packet_sha256 = 以最终 v1232_route_decision.json 为准
```

说明：本次没有新增训练实验，只做 stop-contract route 复核。复核后重新执行 finalizer，使更新后的执行日志和复盘日志进入 code review packet。

执行命令：

```bash
conda run -n kan python experiments/run_v1232_finalize_rational_tail_kernel.py
```

## 12. 用户再次追问后的 route 复核 3

执行命令：

```bash
conda run -n kan python -c "import json; from pathlib import Path; d=json.loads(Path('results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_route_decision.json').read_text()); keys=['route','minimum_success','official_success_reached','p4_pass','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_workspace_pass_count','basis_workspace_strong_pass_count','rational_workspace_pass_count','rational_workspace_strong_pass_count','rational_auc_near_pass_count','nonrat_s3_true_kernel_count','mlp_functional_aggregate_pass_rows','mlp_functional_no_go_current_family','line_a_near_anchor_pass_count','provenance_violation_count','code_review_packet_entries','code_review_packet_sha256']; print('\n'.join(f'{k} = {d.get(k)}' for k in keys))"
```

结果：

```text
route = R1-RationalTailStabilityBlocked
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 72
basis_workspace_strong_pass_count = 0
rational_workspace_pass_count = 72
rational_workspace_strong_pass_count = 0
rational_auc_near_pass_count = 0
nonrat_s3_true_kernel_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_entries = 15
code_review_packet_sha256 = 以最终 v1232_route_decision.json 为准
```

说明：本次没有新增训练实验，只做 stop-contract route 复核。复核后重新执行 finalizer，使更新后的执行日志和复盘日志进入 code review packet。

执行命令：

```bash
conda run -n kan python experiments/run_v1232_finalize_rational_tail_kernel.py
```

## 11. 用户再次追问后的 route 复核 2

执行命令：

```bash
conda run -n kan python -c "import json; from pathlib import Path; d=json.loads(Path('results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_route_decision.json').read_text()); keys=['route','minimum_success','official_success_reached','p4_pass','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_workspace_pass_count','basis_workspace_strong_pass_count','rational_workspace_pass_count','rational_workspace_strong_pass_count','rational_auc_near_pass_count','nonrat_s3_true_kernel_count','mlp_functional_aggregate_pass_rows','mlp_functional_no_go_current_family','line_a_near_anchor_pass_count','provenance_violation_count','code_review_packet_entries','code_review_packet_sha256']; print('\n'.join(f'{k} = {d.get(k)}' for k in keys))"
```

结果：

```text
route = R1-RationalTailStabilityBlocked
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 72
basis_workspace_strong_pass_count = 0
rational_workspace_pass_count = 72
rational_workspace_strong_pass_count = 0
rational_auc_near_pass_count = 0
nonrat_s3_true_kernel_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_entries = 15
code_review_packet_sha256 = 以最终 v1232_route_decision.json 为准
```

说明：本次没有新增训练实验，只做 stop-contract route 复核。复核后重新执行 finalizer，使更新后的执行日志和复盘日志进入 code review packet。

执行命令：

```bash
conda run -n kan python experiments/run_v1232_finalize_rational_tail_kernel.py
```

结果：

```text
route = R1-RationalTailStabilityBlocked
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_rows = 108
basis_workspace_pass_count = 72
basis_workspace_strong_pass_count = 0
rational_workspace_pass_count = 72
rational_workspace_strong_pass_count = 0
rational_auc_near_pass_count = 0
nonrat_s3_true_kernel_count = 0
mlp_functional_candidate_rows = 243
mlp_functional_control_rows = 1458
mlp_functional_linec_rows = 2430
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_entries = 15
code_review_packet_sha256 = 以最终 v1232_route_decision.json 为准
```

## 10. 用户再次追问后的 route 复核

执行命令：

```bash
conda run -n kan python -c "import json; from pathlib import Path; d=json.loads(Path('results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_route_decision.json').read_text()); keys=['route','minimum_success','official_success_reached','p4_pass','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_workspace_pass_count','basis_workspace_strong_pass_count','rational_workspace_pass_count','rational_workspace_strong_pass_count','rational_auc_near_pass_count','nonrat_s3_true_kernel_count','mlp_functional_aggregate_pass_rows','mlp_functional_no_go_current_family','line_a_near_anchor_pass_count','provenance_violation_count','code_review_packet_entries','code_review_packet_sha256']; print('\n'.join(f'{k} = {d.get(k)}' for k in keys))"
```

结果：

```text
route = R1-RationalTailStabilityBlocked
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 72
basis_workspace_strong_pass_count = 0
rational_workspace_pass_count = 72
rational_workspace_strong_pass_count = 0
rational_auc_near_pass_count = 0
nonrat_s3_true_kernel_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_entries = 15
code_review_packet_sha256 = 以最终 v1232_route_decision.json 为准
```

说明：本次没有新增训练实验，只做 stop-contract route 复核。复核后重新执行 finalizer，使更新后的执行日志和复盘日志进入 code review packet。

执行命令：

```bash
conda run -n kan python experiments/run_v1232_finalize_rational_tail_kernel.py
```
