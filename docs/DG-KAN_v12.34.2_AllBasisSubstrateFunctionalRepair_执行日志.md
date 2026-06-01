# DG-KAN v12.34.2 AllBasisSubstrateFunctionalRepair 执行日志

生成时间：2026-05-27（Asia/Singapore）

本日志记录实际执行过的命令、修改文件与 artifact 路径；不补写未执行命令。

## 1. 计划文件

```text
docs/DG-KAN_v12.34.2_AllBasisSubstrateFunctionalRepair_完整计划.md
```

## 2. 代码修改记录

已修改/新增：

```text
dgkan/diagnostics/basis_workspace.py
dgkan/functional/mlp_functional.py
experiments/run_v1231_basis_kernel_workspace.py
experiments/run_v1231_rational_auc_hardening.py
experiments/run_v12342_basis_functional_repair.py
experiments/run_v12342_finalize_all_basis_substrate_functional.py
docs/DG-KAN_v12.34.2_AllBasisSubstrateFunctionalRepair_执行日志.md
docs/DG-KAN_v12.34.2_AllBasisSubstrateFunctionalRepair_实验结果复盘.md
```

修改意图：

```text
1. 增加 v12342 all-basis substrate candidate registry 与 workspace gate。
2. 增加 all-family label-free telemetry；Rational 使用 denominator/derivative telemetry，Non-RAT 使用 BasisKAN.basis_diagnostics + unlabeled response telemetry。
3. 扩展 workspace/AUC runner 支持 --candidate-registry v12342。
4. 增加 basis-specific functional repair P3 runner，比较 NoOp/Random/AdamWParallel/SNR controls。
5. 增加 M-J1..M-J3 MLP closure candidates。
6. 增加 v12342 finalizer 生成 required artifacts、figures、route、code packet。
```

## 3. 已执行命令

### 3.1 代码语法与 registry smoke

```bash
conda run -n kan python -m py_compile dgkan/diagnostics/basis_workspace.py dgkan/functional/mlp_functional.py experiments/run_v1231_basis_kernel_workspace.py experiments/run_v1231_rational_auc_hardening.py experiments/run_v12342_basis_functional_repair.py experiments/run_v12342_finalize_all_basis_substrate_functional.py
```

结果：

```text
py_compile_pass
```

```bash
conda run -n kan python -c "from dgkan.diagnostics.basis_workspace import V12342_BASIS_CANDIDATES; from dgkan.functional.mlp_functional import SOURCE_CANDIDATES; print('basis', len(V12342_BASIS_CANDIDATES)); print('families', sorted({c.family for c in V12342_BASIS_CANDIDATES.values()})); print('mj', [k for k in SOURCE_CANDIDATES if k.startswith('M-J')])"
```

结果：

```text
basis 44
families ['D-CHE', 'D-FOU', 'D-RAT', 'D-RBF', 'D-WAV']
mj ['M-J1-CloneProbeCovarianceTransportV2', 'M-J2-UnlabeledOptimizerObservableTransportV2', 'M-J3-ArchitectureNeutralSNRTransportV2']
```

### 3.2 Smoke: v12342 workspace registry

```bash
mkdir -p results/v12_34_2_all_basis_substrate_functional_repair/smoke_v12342 && conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py --out-dir results/v12_34_2_all_basis_substrate_functional_repair/smoke_v12342 --artifact-prefix v12342_smoke_basis --run-id v12342_smoke_basis --candidate-registry v12342 --candidates D-RAT24-DenDerivativeTelemetryKernel,D-CHE12-LifetimeRecomputeBackward-K3 --datasets MNIST --seeds 0 --train-size 128 --val-size 64 --batch-size 64 --hardening-epochs 1 --workspace-warmup-steps 1 --workspace-profile-steps 1 --linec-seeds 12349500 --linec-batch-size 16
```

结果：

```text
workspace_rows = 2
workspace_gate_pass_rows = 1
workspace_strong_gate_pass_rows = 1
hardening_executed_rows = 1
```

### 3.3 Smoke: v12342 AUC/telemetry runner

```bash
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py --out-dir results/v12_34_2_all_basis_substrate_functional_repair/smoke_v12342 --artifact-prefix v12342_smoke_auc --run-id v12342_smoke_auc --candidate-registry v12342 --candidates D-RAT24-DenDerivativeTelemetryKernel --workspace-csv results/v12_34_2_all_basis_substrate_functional_repair/smoke_v12342/v12342_smoke_basis_workspace_truth.csv --datasets MNIST --seeds 0 --train-size 128 --val-size 64 --batch-size 64 --epochs 1 --lr 0.0015 --linec-seeds 12349500 --linec-batch-size 16
```

结果：

```text
candidate_rows = 1
summary_rows = 1
linec_rows = 1
candidate_auc_near_pass_v1233_rows = 0
```

### 3.4 Smoke: basis-specific functional P3 runner

```bash
conda run -n kan python experiments/run_v12342_basis_functional_repair.py --out-dir results/v12_34_2_all_basis_substrate_functional_repair/smoke_v12342 --artifact-prefix v12342_smoke_basis_functional --workspace-csv results/v12_34_2_all_basis_substrate_functional_repair/smoke_v12342/v12342_smoke_basis_workspace_truth.csv --auc-summary-csv results/v12_34_2_all_basis_substrate_functional_repair/smoke_v12342/v12342_smoke_auc_summary.csv --datasets MNIST --seeds 0 --train-size 128 --val-size 64 --batch-size 64 --functional-batch 64 --window-epochs 1 --linec-seeds 12349500 --linec-batch-size 16 --functional-candidates B-RAT1-TangentTrustRegionNoCE
```

结果：

```text
p3_rows = 1
p3_executed_rows = 1
p3_pass_rows = 0
p4_rows = 1
p4_pass_rows = 0
```

### 3.5 Smoke: M-J MLP closure runner

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py --out-dir results/v12_34_2_all_basis_substrate_functional_repair/smoke_v12342 --artifact-prefix v12342_smoke_mlp_functional --candidates M-J1-CloneProbeCovarianceTransportV2 --datasets MNIST --seeds 0 --train-seed-bases 12340400 --train-size 128 --val-size 64 --epochs 1 --batch-size 64 --functional-batch 64 --linec-seeds 12349500 --linec-batch-size 16
```

结果：

```text
candidate_rows = 1
control_rows = 6
linec_rows = 2
gate_rows = 1
any_exploration_pass = 0
any_official_pass = 0
```

## 4. Official Line D-S: all-basis substrate / workspace map

执行命令：

```bash
OUT=results/v12_34_2_all_basis_substrate_functional_repair/official_v12342
mkdir -p "$OUT"
CANDS=$(conda run -n kan python -c "from dgkan.diagnostics.basis_workspace import V12342_BASIS_CANDIDATES; print(','.join(V12342_BASIS_CANDIDATES))")
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --out-dir "$OUT" \
  --artifact-prefix v12342_basis \
  --run-id v12342_basis_official \
  --candidate-registry v12342 \
  --candidates "$CANDS" \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 512 \
  --val-size 256 \
  --batch-size 128 \
  --hardening-epochs 3 \
  --workspace-warmup-steps 5 \
  --workspace-profile-steps 20 \
  --linec-seeds 12349500,12350600,12351600 \
  --linec-batch-size 32 \
  --lr 0.002 \
  --weight-decay 0.001
```

结果：

```text
workspace_rows = 396
workspace_gate_pass_rows = 137
workspace_strong_gate_pass_rows = 0
hardening_rows = 396
hardening_executed_rows = 137
family_near_pass_rows = 0
```

产物：

```text
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_basis_workspace_truth.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_basis_component_peak.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_basis_exact_kernel_audit.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_basis_kernel_implementation_manifest.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_basis_hardening.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_basis_linec.csv
```

## 5. Official Line D-T: all-basis AUC / telemetry

### 5.1 lr=0.0015 初次运行 blocker

执行命令：

```bash
OUT=results/v12_34_2_all_basis_substrate_functional_repair/official_v12342
CANDS=$(conda run -n kan python -c "from dgkan.diagnostics.basis_workspace import V12342_BASIS_CANDIDATES; print(','.join(V12342_BASIS_CANDIDATES))")
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --out-dir "$OUT" \
  --artifact-prefix v12342_all_basis_auc_lr15 \
  --run-id v12342_all_basis_auc_lr15 \
  --candidate-registry v12342 \
  --candidates "$CANDS" \
  --workspace-csv "$OUT/v12342_basis_workspace_truth.csv" \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 512 \
  --val-size 256 \
  --batch-size 128 \
  --epochs 3 \
  --lr 0.0015 \
  --weight-decay 0.001 \
  --linec-seeds 12349500,12350600,12351600 \
  --linec-batch-size 32
```

结果：

```text
blocked: IndexError in summary aggregation for all-skipped Non-RAT candidates.
```

修复：

```text
experiments/run_v1231_rational_auc_hardening.py
summary family field now uses executed[0] when present, otherwise rows[0].
All-skipped candidates are written as explicit skipped summary rows instead of crashing.
```

### 5.2 lr=0.0015 修复后重跑

执行命令同 5.1。

结果：

```text
candidate_rows = 396
summary_rows = 44
trajectory_rows = 438
linec_rows = 411
candidate_auc_near_pass_rows = 0
candidate_auc_near_pass_v1232_rows = 0
candidate_auc_near_pass_v1233_rows = 0
candidate_auc_official_pass_v1233_rows = 0
```

## 13. 用户再次追问后的计划文件 stop-contract 复核 3

执行命令：

```bash
rg -n "S5|P3|P4|fallback|final_stop|stop|不允许|允许|推荐|R2|R5|Substrate|M-J" \
  docs/DG-KAN_v12.34.2_AllBasisSubstrateFunctionalRepair_完整计划.md
```

关键输出：

```text
1046:S1-AnyFamilySubstratePass:
1049:S2-AnyFamilyFunctionalRepairP3:
1052:S3-AnyFamilyFunctionalRepairP4:
1058:S5-OfficialBasisFunctionalSuccess:
1064:R2-SubstrateButNoFunctionalRepair:
1073:R5-MLPFunctionalNoGo_BasisSpecificOnly:
1087:没有 telemetry 的 family 不允许进入 functional repair。
```

执行命令：

```bash
sed -n '1038,1092p' docs/DG-KAN_v12.34.2_AllBasisSubstrateFunctionalRepair_完整计划.md
```

计划复核要点：

```text
S1 = 至少一个 active basis 通过 Substrate Gate。
S2 = 至少一个 basis-specific functional repair 通过 P3。
S3 = 至少一个 basis-specific functional repair 通过 P4。
S5 = label-free strict FC-PureKAN base/substrate + functional repair 通过 official controls。
R2 = 有 substrate，但 no functional P3/P4。
R5 = MLP functional closure 失败，后续 functional 预算转向 basis-specific。
下一步优先级 = all-basis substrate map -> family telemetry -> basis-specific functional repair。
```

执行命令：

```bash
conda run -n kan python -c "import json, pathlib; p=pathlib.Path('results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_route_decision.json'); d=json.loads(p.read_text()); keys=['route','route_detail','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_workspace_rows','basis_workspace_pass_count','basis_workspace_strong_pass_count','substrate_gate_pass_count','substrate_near_pass_count','healthy_base_gate_pass_count','nonrat_s3_lifetime_count','basis_functional_p3_pass_count','basis_functional_p4_pass_count','mlp_functional_mj_pass_rows','mlp_functional_no_go_current_family_v2','provenance_violation_count']; [print(f'{k}={d.get(k)}') for k in keys]"
```

结果：

```text
route=R2-SubstrateButNoFunctionalRepair
route_detail=R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success=S1-AnyFamilySubstratePass
official_success_reached=0
promotion_allowed=0
final_stop_allowed=1
hard_compute_budget_exhausted=1
fallback_depth=7
fallback_all_executed=1
required_artifact_missing_count=0
basis_workspace_rows=504
basis_workspace_pass_count=172
basis_workspace_strong_pass_count=0
substrate_gate_pass_count=11
substrate_near_pass_count=15
healthy_base_gate_pass_count=0
nonrat_s3_lifetime_count=3
basis_functional_p3_pass_count=0
basis_functional_p4_pass_count=0
mlp_functional_mj_pass_rows=0
mlp_functional_no_go_current_family_v2=1
provenance_violation_count=0
```

本次没有新增训练实验、没有新增 artifact 指标、没有修改代码或 gate。计划中的 all-basis substrate map、family telemetry、basis-specific functional repair、Non-RAT lifetime fallback 与 focused P3 均已执行；当前 final stop 仍由 artifact 支持。

## 14. 用户再次追问后的 stop-contract 复核 4

执行命令：

```bash
conda run -n kan python -c "import json, pathlib; p=pathlib.Path('results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_route_decision.json'); d=json.loads(p.read_text()); keys=['route','route_detail','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_workspace_rows','basis_workspace_pass_count','basis_workspace_strong_pass_count','substrate_gate_pass_count','substrate_near_pass_count','healthy_base_gate_pass_count','nonrat_s3_lifetime_count','basis_functional_p3_pass_count','basis_functional_p4_pass_count','mlp_functional_mj_pass_rows','mlp_functional_no_go_current_family_v2','provenance_violation_count']; [print(f'{k}={d.get(k)}') for k in keys]"
```

结果：

```text
route=R2-SubstrateButNoFunctionalRepair
route_detail=R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success=S1-AnyFamilySubstratePass
official_success_reached=0
promotion_allowed=0
final_stop_allowed=1
hard_compute_budget_exhausted=1
fallback_depth=7
fallback_all_executed=1
required_artifact_missing_count=0
basis_workspace_rows=504
basis_workspace_pass_count=172
basis_workspace_strong_pass_count=0
substrate_gate_pass_count=11
substrate_near_pass_count=15
healthy_base_gate_pass_count=0
nonrat_s3_lifetime_count=3
basis_functional_p3_pass_count=0
basis_functional_p4_pass_count=0
mlp_functional_mj_pass_rows=0
mlp_functional_no_go_current_family_v2=1
provenance_violation_count=0
```

本次没有新增训练实验、没有新增 artifact 指标、没有修改代码或 gate。当前 final stop 仍由 artifact 支持。

## 16. 用户再次追问后的 stop-contract 复核 6

执行命令：

```bash
conda run -n kan python -c "import json, pathlib; d=json.loads(pathlib.Path('results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_route_decision.json').read_text()); keys=['route','route_detail','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_workspace_rows','basis_workspace_pass_count','substrate_gate_pass_count','substrate_near_pass_count','nonrat_s3_lifetime_count','basis_functional_p3_pass_count','basis_functional_p4_pass_count','mlp_functional_mj_pass_rows','mlp_functional_no_go_current_family_v2','provenance_violation_count']; print('\n'.join(f'{k}={d.get(k)}' for k in keys))"
```

结果：

```text
route=R2-SubstrateButNoFunctionalRepair
route_detail=R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success=S1-AnyFamilySubstratePass
official_success_reached=0
promotion_allowed=0
final_stop_allowed=1
hard_compute_budget_exhausted=1
fallback_depth=7
fallback_all_executed=1
required_artifact_missing_count=0
basis_workspace_rows=504
basis_workspace_pass_count=172
substrate_gate_pass_count=11
substrate_near_pass_count=15
nonrat_s3_lifetime_count=3
basis_functional_p3_pass_count=0
basis_functional_p4_pass_count=0
mlp_functional_mj_pass_rows=0
mlp_functional_no_go_current_family_v2=1
provenance_violation_count=0
```

本次没有新增训练实验、没有新增 artifact 指标、没有修改代码或 gate。当前 final stop 仍由 artifact 支持。

## 15. 用户再次追问后的 stop-contract 复核 5

执行命令：

```bash
rg -n "fallback|Fallback|修复|repair|如果|若|必须|不允许|允许|优先级|P3|P4|Substrate|telemetry|Non-RAT|AdamW|foreach|LineC|CouplingR2" \
  docs/DG-KAN_v12.34.2_AllBasisSubstrateFunctionalRepair_完整计划.md
```

用途：

```text
重新确认计划推荐路径、functional repair gate、kernel lifetime fallback 与 MLP closure 条件。
```

执行命令：

```bash
ls -lh results/v12_34_2_all_basis_substrate_functional_repair/official_v12342
```

用途：

```text
确认 v12.34.2 official artifact 目录包含 workspace、family telemetry、basis functional、kernel lifetime、MLP closure、figures、route 与 manifest 等产物。
```

执行命令：

```bash
conda run -n kan python -c "import json, pathlib; d=json.loads(pathlib.Path('results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_route_decision.json').read_text()); keys=['route','route_detail','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_workspace_rows','basis_workspace_pass_count','substrate_gate_pass_count','substrate_near_pass_count','nonrat_s3_lifetime_count','basis_functional_p3_pass_count','basis_functional_p4_pass_count','mlp_functional_mj_pass_rows','mlp_functional_no_go_current_family_v2','provenance_violation_count']; print('\n'.join(f'{k}={d.get(k)}' for k in keys))"
```

结果：

```text
route=R2-SubstrateButNoFunctionalRepair
route_detail=R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success=S1-AnyFamilySubstratePass
official_success_reached=0
promotion_allowed=0
final_stop_allowed=1
hard_compute_budget_exhausted=1
fallback_depth=7
fallback_all_executed=1
required_artifact_missing_count=0
basis_workspace_rows=504
basis_workspace_pass_count=172
substrate_gate_pass_count=11
substrate_near_pass_count=15
nonrat_s3_lifetime_count=3
basis_functional_p3_pass_count=0
basis_functional_p4_pass_count=0
mlp_functional_mj_pass_rows=0
mlp_functional_no_go_current_family_v2=1
provenance_violation_count=0
```

执行命令：

```bash
conda run -n kan python -c "import csv, pathlib; p=pathlib.Path('results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_required_artifact_manifest.csv'); rows=list(csv.DictReader(p.open())); print(f'manifest_rows={len(rows)}'); print(f'missing_rows={sum(1 for r in rows if str(r.get(\"exists\",\"\")).lower() not in [\"1\",\"true\",\"yes\"]) }')"
```

结果：

```text
manifest_rows=28
missing_rows=0
```

本次没有新增训练实验、没有新增 artifact 指标、没有修改代码或 gate。当前 final stop 仍由 artifact 支持。

## 12. 用户再次追问后的 stop-contract 复核 2

执行命令：

```bash
conda run -n kan python -c "import json, pathlib; p=pathlib.Path('results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_route_decision.json'); d=json.loads(p.read_text()); keys=['route','route_detail','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_workspace_rows','basis_workspace_pass_count','basis_workspace_strong_pass_count','substrate_gate_pass_count','substrate_near_pass_count','healthy_base_gate_pass_count','nonrat_s3_lifetime_count','basis_functional_p3_pass_count','basis_functional_p4_pass_count','mlp_functional_mj_pass_rows','mlp_functional_no_go_current_family_v2','provenance_violation_count']; [print(f'{k}={d.get(k)}') for k in keys]"
```

结果：

```text
route=R2-SubstrateButNoFunctionalRepair
route_detail=R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success=S1-AnyFamilySubstratePass
official_success_reached=0
promotion_allowed=0
final_stop_allowed=1
hard_compute_budget_exhausted=1
fallback_depth=7
fallback_all_executed=1
required_artifact_missing_count=0
basis_workspace_rows=504
basis_workspace_pass_count=172
basis_workspace_strong_pass_count=0
substrate_gate_pass_count=11
substrate_near_pass_count=15
healthy_base_gate_pass_count=0
nonrat_s3_lifetime_count=3
basis_functional_p3_pass_count=0
basis_functional_p4_pass_count=0
mlp_functional_mj_pass_rows=0
mlp_functional_no_go_current_family_v2=1
provenance_violation_count=0
```

本次没有新增训练实验、没有新增 artifact 指标、没有修改代码或 gate。当前 final stop 仍由 artifact 支持。

## 11. 用户再次追问后的 stop-contract 复核

执行命令：

```bash
conda run -n kan python -c "import json, pathlib; p=pathlib.Path('results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_route_decision.json'); d=json.loads(p.read_text()); keys=['route','route_detail','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_all_executed','required_artifact_missing_count','basis_workspace_rows','basis_workspace_pass_count','basis_workspace_strong_pass_count','substrate_gate_pass_count','substrate_near_pass_count','healthy_base_gate_pass_count','nonrat_s3_lifetime_count','basis_functional_p3_pass_count','basis_functional_p4_pass_count','mlp_functional_mj_pass_rows','provenance_violation_count']; [print(f'{k}={d.get(k)}') for k in keys]"
```

结果：

```text
route=R2-SubstrateButNoFunctionalRepair
route_detail=R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success=S1-AnyFamilySubstratePass
official_success_reached=0
promotion_allowed=0
final_stop_allowed=1
hard_compute_budget_exhausted=1
fallback_all_executed=1
required_artifact_missing_count=0
basis_workspace_rows=504
basis_workspace_pass_count=172
basis_workspace_strong_pass_count=0
substrate_gate_pass_count=11
substrate_near_pass_count=15
healthy_base_gate_pass_count=0
nonrat_s3_lifetime_count=3
basis_functional_p3_pass_count=0
basis_functional_p4_pass_count=0
mlp_functional_mj_pass_rows=0
provenance_violation_count=0
```

本次没有新增训练实验，也没有改动 gate。当前 final stop 仍由 artifact 支持。

## 6. Official Line B-F: Rational substrate basis-specific P3

选择理由：

```text
从 lr=0.0015 AUC summary 中选择 executed_rows=9 且 mean_delta 最靠前的完整执行候选：
D-RAT34-TrainingDenDerivativeStabilityNoCE
D-RAT28-GroupDiversityPreservingRational
D-RAT26-TangentTrustRegionNoCE
未选择 D-RAT38，因为它有 skipped row，不适合作为第一轮 P3 base。
```

执行命令：

```bash
OUT=results/v12_34_2_all_basis_substrate_functional_repair/official_v12342
conda run -n kan python experiments/run_v12342_basis_functional_repair.py \
  --out-dir "$OUT" \
  --artifact-prefix v12342_basis_functional \
  --workspace-csv "$OUT/v12342_basis_workspace_truth.csv" \
  --auc-summary-csv "$OUT/v12342_all_basis_auc_lr15_summary.csv" \
  --base-candidates D-RAT34-TrainingDenDerivativeStabilityNoCE,D-RAT28-GroupDiversityPreservingRational,D-RAT26-TangentTrustRegionNoCE \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 512 \
  --val-size 256 \
  --batch-size 128 \
  --functional-batch 128 \
  --window-epochs 1 \
  --linec-seeds 12349500,12350600,12351600 \
  --linec-batch-size 32 \
  --lr 0.0015 \
  --weight-decay 0.001
```

结果：

```text
p3_rows = 135
p3_executed_rows = 135
p3_pass_rows = 0
p4_rows = 135
p4_pass_rows = 0
```

## 7. Official Line M: M-J MLP closure

执行命令：

```bash
OUT=results/v12_34_2_all_basis_substrate_functional_repair/official_v12342
conda run -n kan python experiments/run_v1230_mlp_functional.py \
  --out-dir "$OUT" \
  --artifact-prefix v12342_mlp_functional \
  --candidates M-J1-CloneProbeCovarianceTransportV2,M-J2-UnlabeledOptimizerObservableTransportV2,M-J3-ArchitectureNeutralSNRTransportV2 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-seed-bases 12340400,12341400,12342400 \
  --train-size 512 \
  --val-size 256 \
  --epochs 3 \
  --batch-size 128 \
  --functional-batch 128 \
  --linec-seeds 12349500,12350600,12351600,12352600,12353600 \
  --linec-batch-size 32 \
  --lr 0.0015 \
  --weight-decay 0.001
```

结果：

```text
candidate_rows = 81
control_rows = 486
linec_rows = 810
gate_rows = 3
any_exploration_pass = 0
any_official_pass = 0
```

## 8. First finalizer pass

执行命令：

```bash
conda run -n kan python experiments/run_v12342_finalize_all_basis_substrate_functional.py
```

结果：

```text
route = R2-SubstrateButNoFunctionalRepair
minimum_success = S1-AnyFamilySubstratePass
official_success_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
basis_workspace_rows = 396
basis_workspace_pass_count = 137
basis_functional_p3_pass_count = 0
mlp_functional_mj_pass_rows = 0
```

## 9. Blocker-driven fallback: AdamW foreach-off Non-RAT lifetime repair

触发原因：

```text
v12342_kernel_lifetime_waterfall 显示 Non-RAT largest live component 多数为 optimizer_state；
计划 8.4 明确要求尝试 foreach disabled audit / optimizer-state lifetime repair。
```

代码修改：

```text
experiments/run_v1231_basis_kernel_workspace.py
1. 新增 --adamw-foreach auto|true|false。
2. 新增 make_adamw() helper。
3. workspace profile / MLP reference / basis hardening 使用相同 AdamW foreach 配置。

experiments/run_v1231_rational_auc_hardening.py
1. 新增 --adamw-foreach auto|true|false。
2. AUC/telemetry MLP reference 与 basis candidate 使用同样 make_adamw()。

experiments/run_v12342_basis_functional_repair.py
1. 新增 --adamw-foreach auto|true|false。
2. focused Non-RAT P3 使用 foreach-off AdamW。

experiments/run_v12342_finalize_all_basis_substrate_functional.py
1. 合并 v12342_nonrat_lifetime_foreachoff_* fallback artifacts。
2. 合并 focused Non-RAT P3/P4/LineC fallback artifacts。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1231_basis_kernel_workspace.py experiments/run_v1231_rational_auc_hardening.py
conda run -n kan python -m py_compile experiments/run_v12342_basis_functional_repair.py experiments/run_v12342_finalize_all_basis_substrate_functional.py
```

结果：

```text
py_compile pass
```

### 9.1 foreach-off workspace/lifetime repair

执行命令：

```bash
OUT=results/v12_34_2_all_basis_substrate_functional_repair/official_v12342
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --out-dir "$OUT" \
  --artifact-prefix v12342_nonrat_lifetime_foreachoff \
  --run-id v12342_nonrat_lifetime_foreachoff \
  --candidate-registry v12342 \
  --candidates D-CHE12-LifetimeRecomputeBackward-K3,D-CHE13-FusedReadoutGradNoMaterialize-K3,D-CHE15-FullStepNoMaterialize-K3,D-FOU12-LifetimeRecomputeBackward-K2,D-FOU13-FusedReadoutGradNoMaterialize-K2,D-FOU14-SincosSharedWorkspace-K2,D-FOU15-FullStepNoMaterialize-K2,D-FOU16-FrequencyBandDampingSubstrate,D-RBF11-CompactExpressionRepair-Monitor,D-RBF12-CenterOccupancyRebalanceSubstrate,D-WAV10-HatWaveletLifetimeRepair-Monitor,D-WAV11-ScaleEnergyBalanceSubstrate \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 512 \
  --val-size 256 \
  --batch-size 128 \
  --hardening-epochs 3 \
  --workspace-warmup-steps 5 \
  --workspace-profile-steps 20 \
  --linec-seeds 12349500,12350600,12351600 \
  --linec-batch-size 32 \
  --lr 0.002 \
  --weight-decay 0.001 \
  --adamw-foreach false
```

结果：

```text
workspace_rows = 108
workspace_gate_pass_rows = 35
workspace_strong_gate_pass_rows = 0
hardening_executed_rows = 35
family_near_pass_rows = 0
```

### 9.2 foreach-off AUC/telemetry repair

执行命令：

```bash
OUT=results/v12_34_2_all_basis_substrate_functional_repair/official_v12342
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --out-dir "$OUT" \
  --artifact-prefix v12342_all_basis_auc_repair \
  --run-id v12342_all_basis_auc_repair_foreachoff \
  --candidate-registry v12342 \
  --candidates D-CHE12-LifetimeRecomputeBackward-K3,D-CHE13-FusedReadoutGradNoMaterialize-K3,D-CHE15-FullStepNoMaterialize-K3,D-FOU12-LifetimeRecomputeBackward-K2,D-FOU13-FusedReadoutGradNoMaterialize-K2,D-FOU14-SincosSharedWorkspace-K2,D-FOU15-FullStepNoMaterialize-K2,D-FOU16-FrequencyBandDampingSubstrate,D-RBF11-CompactExpressionRepair-Monitor,D-RBF12-CenterOccupancyRebalanceSubstrate,D-WAV10-HatWaveletLifetimeRepair-Monitor,D-WAV11-ScaleEnergyBalanceSubstrate \
  --workspace-csv "$OUT/v12342_nonrat_lifetime_foreachoff_workspace_truth.csv" \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 512 \
  --val-size 256 \
  --batch-size 128 \
  --epochs 3 \
  --lr 0.002 \
  --weight-decay 0.001 \
  --linec-seeds 12349500,12350600,12351600 \
  --linec-batch-size 32 \
  --adamw-foreach false
```

结果：

```text
candidate_rows = 108
summary_rows = 12
trajectory_rows = 132
linec_rows = 105
candidate_auc_near_pass_rows = 0
candidate_auc_official_pass_v1233_rows = 0
```

### 9.3 focused Non-RAT Fourier P3 repair

执行命令：

```bash
OUT=results/v12_34_2_all_basis_substrate_functional_repair/official_v12342
conda run -n kan python experiments/run_v12342_basis_functional_repair.py \
  --out-dir "$OUT" \
  --artifact-prefix v12342_basis_functional_nonrat_foreachoff \
  --workspace-csv "$OUT/v12342_nonrat_lifetime_foreachoff_workspace_truth.csv" \
  --auc-summary-csv "$OUT/v12342_all_basis_auc_repair_summary.csv" \
  --base-candidates D-FOU12-LifetimeRecomputeBackward-K2,D-FOU14-SincosSharedWorkspace-K2,D-FOU16-FrequencyBandDampingSubstrate \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 512 \
  --val-size 256 \
  --batch-size 128 \
  --functional-batch 128 \
  --window-epochs 1 \
  --linec-seeds 12349500,12350600,12351600 \
  --linec-batch-size 32 \
  --lr 0.002 \
  --weight-decay 0.001 \
  --adamw-foreach false
```

结果：

```text
p3_rows = 135
p3_executed_rows = 60
p3_pass_rows = 0
p4_rows = 60
p4_pass_rows = 0
```

## 10. Finalizer after fallback

执行命令：

```bash
conda run -n kan python experiments/run_v12342_finalize_all_basis_substrate_functional.py
```

补充修复：

```text
计划 6.2 要求 fig_family_telemetry_vs_auc.svg 与 fig_family_telemetry_correlation_heatmap.svg。
finalizer 已补齐这两个 figure，并重新生成 route / required manifest / code packet。
```

最终结果：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success = S1-AnyFamilySubstratePass
official_success_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
fallback_all_executed = 1
hard_compute_budget_exhausted = 1
final_stop_allowed = 1
code_review_packet_entries = 30
code_review_packet_sha256 = 6d159cd0115d5405b3131ade41816478da627a13e7e3604935f77d78de806240
```

### 5.3 lr=0.002

执行命令：

```bash
OUT=results/v12_34_2_all_basis_substrate_functional_repair/official_v12342
CANDS=$(conda run -n kan python -c "from dgkan.diagnostics.basis_workspace import V12342_BASIS_CANDIDATES; print(','.join(V12342_BASIS_CANDIDATES))")
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --out-dir "$OUT" \
  --artifact-prefix v12342_all_basis_auc_lr20 \
  --run-id v12342_all_basis_auc_lr20 \
  --candidate-registry v12342 \
  --candidates "$CANDS" \
  --workspace-csv "$OUT/v12342_basis_workspace_truth.csv" \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 512 \
  --val-size 256 \
  --batch-size 128 \
  --epochs 3 \
  --lr 0.002 \
  --weight-decay 0.001 \
  --linec-seeds 12349500,12350600,12351600 \
  --linec-batch-size 32
```

结果：

```text
candidate_rows = 396
summary_rows = 44
trajectory_rows = 438
linec_rows = 411
candidate_auc_near_pass_rows = 0
candidate_auc_near_pass_v1232_rows = 0
candidate_auc_near_pass_v1233_rows = 0
candidate_auc_official_pass_v1233_rows = 0
```
