# DG-KAN v12.30 BasisFirst FunctionalAlsoMLP 执行日志

生成时间：2026-05-26（Asia/Singapore）

本日志只记录实际执行过的命令、文件与产物路径；不补写未执行命令。

## 1. 计划入口

计划文档：

```text
docs/DG-KAN_v12.30_BasisFirst_FunctionalAlsoMLP_完整计划.md
```

结果目录：

```text
results/v12_30_basis_first_functional_also_mlp/official_v1230
```

## 2. 当前待执行步骤

后续 sections 将按实际执行顺序写入：

```text
1. py_compile / provenance audit
2. smoke checks
3. Line D classic basis scout
4. Line D classic basis hardening
5. Line D family fallback repair
6. Line M MLP functional diagnostic
7. Line A FHQ dynamic monitor
8. finalizer / artifact packet
```

## 3. 语法检查与 provenance audit

执行命令：

```bash
python -m py_compile experiments/run_v1224_classic_hardening.py experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1230_mlp_functional.py experiments/run_v1230_finalize_basis_first.py \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_py_compile.log
```

结果：

```text
py_compile pass
```

第一次 heredoc 形式审计命令：

```bash
conda run -n kan python - <<'PY' 2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_registry_provenance_audit.log
...
PY
```

结果：

```text
conda run 未传入 heredoc stdin，日志为空；没有产生审计输出。
```

重跑命令：

```bash
conda run -n kan python -c "from experiments.run_v1218_b320_label_free_ablation import ablation_specs; from experiments.run_v1224_classic_hardening import FAMILY_SPECS; from experiments.run_v1226_label_free_only_bridge import forbidden_token_present; from experiments.run_v1230_mlp_functional import SOURCE_CANDIDATES, CONTROL_IDS; ..."
```

结果摘要：

```text
A_DYN_COUNT = 5
A-DYN1..A-DYN5 uses_y_for_stats=0
A-DYN1..A-DYN5 forbidden_token_present=0
CLASSIC_V1230_ALIAS_COUNT = 27
MLP_SOURCE_COUNT = 5
M-F1..M-F5 uses_label_for_direction=0 / uses_ce_vector_for_direction=0
C3-AdamWParallelDirection is control only, promotion_allowed=0
```

## 4. Smoke checks

Classic basis smoke command：

```bash
conda run -n kan python experiments/run_v1224_classic_hardening.py \
  --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 \
  --artifact-prefix v1230_classic_basis_smoke \
  --run-id v1230_classic_basis_smoke \
  --families D-CHE1-K3-degreeEnergyCap,D-RBF1-FastKANFixedCenterK4 \
  --datasets MNIST --seeds 0 \
  --train-size 64 --val-size 32 --epochs 1 --batch-size 32 \
  --linec-batch-size 16 --linec-sketch-dim 4 \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_classic_basis_smoke.log
```

结果：

```text
rows = 2
summary_rows = 2
exploration_pass_rows = 0
```

MLP functional smoke command：

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py \
  --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 \
  --artifact-prefix v1230_mlp_functional_smoke \
  --datasets MNIST --seeds 0 --train-seed-bases 12300400 \
  --candidates M-F1-ActivationCovTransport \
  --train-size 64 --val-size 32 --epochs 1 --batch-size 32 \
  --functional-batch 32 --linec-batch-size 16 --linec-sketch-dim 4 --linec-seeds 12309500 \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_mlp_functional_smoke.log
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

FHQ A-DYN monitor smoke command：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 \
  --artifact-prefix v1230_fhq_dynamic_monitor_smoke \
  --run-id v1230_fhq_dynamic_monitor_smoke \
  --summary-stage V1230_FHQ_DYNAMIC_MONITOR_SMOKE_SUMMARY \
  --route-stage V1230_FHQ_DYNAMIC_MONITOR_SMOKE_ROUTE \
  --result-scope v1230_fhq_dynamic_monitor_smoke \
  --datasets MNIST --seeds 0 \
  --ablation-ids A-DYN1-LearnableSignalFrameWarmup,A-DYN5-RoleEnergyBalancedFHQMonitor \
  --train-size 64 --val-size 32 --test-size 32 \
  --epochs 1 --batch-size 32 \
  --measure-linec 1 --linec-batch-size 16 --linec-sketch-dim 4 \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_fhq_dynamic_monitor_smoke.log
```

结果：

```text
rows = 4
summary_rows = 4
best_label_free_candidate = A-DYN1-LearnableSignalFrameWarmup
```

## 5. Line D classic basis scout

执行命令：

```bash
conda run -n kan python experiments/run_v1224_classic_hardening.py \
  --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 \
  --artifact-prefix v1230_classic_basis_scout \
  --run-id v1230_classic_basis_scout \
  --families D-RAT1-GroupWorkspaceRecomputeV3,D-RAT2-ReadoutGradChunkedV2,D-RAT3-DenStateFP16Checkpoint,D-RAT4-GroupRationalSharedDenom,D-RAT5-RationalLineCResidualMix,D-RAT6-RationalTaskTrajectoryWarmNoExtraMemV2,D-CHE1-K3-degreeEnergyCap,D-CHE2-K4-lateEnableHighDegree,D-CHE3-K4-roleWiseEnergyCap,D-CHE4-K3K4-lowDegreeResidual,D-CHE5-LineCEnergyMonitor,D-WAV1-HatLocalK4ScaleStable,D-WAV2-TriangleK4OccupancyBalanced,D-WAV3-HaarLiteLocalSupportDiagnostic,D-WAV4-ScaleDiversitySmallResidual,D-WAV5-LocalTailCoverageMonitor,D-RBF1-FastKANFixedCenterK4,D-RBF2-CompactLocalRBFKactive4,D-RBF3-CompactLocalRBFKactive8,D-RBF4-TrainStreamQuantileCenters,D-RBF5-TriangularBumpApprox,D-RBF6-IdentityCompactRBFResidual,D-FOU1-K2LowfreqIdentity,D-FOU2-K4LowfreqLinearResidual,D-FOU3-LearnedAmplitudeResidual,D-FOU4-LateEnableK4FromK2,D-FOU5-MultiScaleLowK \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 512 --val-size 256 --epochs 3 --batch-size 128 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_classic_basis_scout.log
```

结果：

```text
rows = 243
summary_rows = 27
exploration_pass_rows = 0
```

## 6. Line D classic basis hardening

选择依据：scout 后按 task mean、LineC、family coverage 选择：

```text
D-RAT4,D-RAT6,D-RAT3,D-RAT5,D-RAT1,D-WAV5,D-FOU5,D-CHE5,D-RBF6
```

执行命令：

```bash
conda run -n kan python experiments/run_v1224_classic_hardening.py \
  --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 \
  --artifact-prefix v1230_classic_basis_hardening \
  --run-id v1230_classic_basis_hardening \
  --families D-RAT4-GroupRationalSharedDenom,D-RAT6-RationalTaskTrajectoryWarmNoExtraMemV2,D-RAT3-DenStateFP16Checkpoint,D-RAT5-RationalLineCResidualMix,D-RAT1-GroupWorkspaceRecomputeV3,D-WAV5-LocalTailCoverageMonitor,D-FOU5-MultiScaleLowK,D-CHE5-LineCEnergyMonitor,D-RBF6-IdentityCompactRBFResidual \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 1024 --val-size 512 --epochs 8 --batch-size 128 \
  --linec-batch-size 64 --linec-sketch-dim 24 \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_classic_basis_hardening.log
```

结果：

```text
rows = 81
summary_rows = 9
exploration_pass_rows = 0
```

## 7. Line D family-specific fallback

原因：hardening 没有 FamilyNearPass；Rational scout 的 LineC 优势硬化后下降，非-Rational family 仍 task-blocked。按计划补跑 memory/task/geometry repair 代表候选。

执行命令：

```bash
conda run -n kan python experiments/run_v1224_classic_hardening.py \
  --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 \
  --artifact-prefix v1230_classic_basis_fallback \
  --run-id v1230_classic_basis_fallback \
  --families D-RAT2-ReadoutGradChunkedV2,D-CHE3-K4-roleWiseEnergyCap,D-CHE4-K3K4-lowDegreeResidual,D-WAV4-ScaleDiversitySmallResidual,D-RBF5-TriangularBumpApprox,D-FOU3-LearnedAmplitudeResidual,D-FOU4-LateEnableK4FromK2 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 1024 --val-size 512 --epochs 8 --batch-size 128 \
  --linec-batch-size 64 --linec-sketch-dim 24 \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_classic_basis_fallback.log
```

结果：

```text
rows = 63
summary_rows = 7
exploration_pass_rows = 0
```

## 8. Line M MLP functional diagnostic

计划要求 windows=3,5,10。初次运行后发现 `LineC_pass_count` 聚合 bug：candidate row 筛选 LineC rows 时漏掉 dataset，出现 `15/5` 这种不可能计数。修复文件：

```text
experiments/run_v1230_mlp_functional.py
```

修复内容：

```text
source_linec / noop_linec 过滤条件增加 dataset；
新增 window_epochs 字段便于三窗口合并审计。
```

验证命令：

```bash
python -m py_compile experiments/run_v1230_mlp_functional.py
```

结果：

```text
py_compile pass
```

fixed window=3 command：

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py \
  --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 \
  --artifact-prefix v1230_mlp_functional_w3_fixed \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-seed-bases 12300400,12301400,12302400 \
  --candidates M-F1-ActivationCovTransport,M-F2-LogitCovStabilization,M-F3-RandomCotangentJacobianSketch,M-F4-HiddenSpectralBalance,M-F5-GeometrySafeWeightAnchor \
  --train-size 512 --val-size 256 --epochs 3 --batch-size 128 \
  --functional-batch 128 --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12309500,12310600,12311600,12312600,12313600 \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_mlp_functional_w3_fixed.log
```

fixed window=5 command：

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py \
  --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 \
  --artifact-prefix v1230_mlp_functional_w5_fixed \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-seed-bases 12300400,12301400,12302400 \
  --candidates M-F1-ActivationCovTransport,M-F2-LogitCovStabilization,M-F3-RandomCotangentJacobianSketch,M-F4-HiddenSpectralBalance,M-F5-GeometrySafeWeightAnchor \
  --train-size 512 --val-size 256 --epochs 5 --batch-size 128 \
  --functional-batch 128 --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12309500,12310600,12311600,12312600,12313600 \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_mlp_functional_w5_fixed.log
```

fixed window=10 command：

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py \
  --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 \
  --artifact-prefix v1230_mlp_functional_w10_fixed \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-seed-bases 12300400,12301400,12302400 \
  --candidates M-F1-ActivationCovTransport,M-F2-LogitCovStabilization,M-F3-RandomCotangentJacobianSketch,M-F4-HiddenSpectralBalance,M-F5-GeometrySafeWeightAnchor \
  --train-size 512 --val-size 256 --epochs 10 --batch-size 128 \
  --functional-batch 128 --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12309500,12310600,12311600,12312600,12313600 \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_mlp_functional_w10_fixed.log
```

合并命令：

```bash
conda run -n kan python -c "import csv,json; from pathlib import Path; out=Path('results/v12_30_basis_first_functional_also_mlp/official_v1230'); ..."
```

合并结果：

```text
candidate_rows = 405
control_rows = 2430
linec_rows = 4050
gate_rows = 15
windows = 3,5,10
max_linec_pass_count = 5/5
any_exploration_pass = 0
any_official_pass = 0
```

## 9. Line A FHQ dynamic monitor

执行命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 \
  --artifact-prefix v1230_fhq_dynamic_monitor \
  --run-id v1230_fhq_dynamic_monitor \
  --summary-stage V1230_FHQ_DYNAMIC_MONITOR_SUMMARY \
  --route-stage V1230_FHQ_DYNAMIC_MONITOR_ROUTE \
  --result-scope v1230_fhq_dynamic_monitor \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --ablation-ids A-DYN1-LearnableSignalFrameWarmup,A-DYN2-EarlySelfPredictiveFrame,A-DYN3-OptimizerObservableFrameRefresh,A-DYN4-OvercompleteRankGuardFrame,A-DYN5-RoleEnergyBalancedFHQMonitor \
  --train-size 512 --val-size 256 --test-size 256 \
  --epochs 3 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 32 --linec-sketch-dim 8 \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_fhq_dynamic_monitor.log
```

结果：

```text
rows = 63
summary_rows = 7
best_label_free_candidate = A-DYN1-LearnableSignalFrameWarmup
```

## 10. Finalizer

首次执行命令：

```bash
conda run -n kan python experiments/run_v1230_finalize_basis_first.py \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_finalize_initial.log
```

首次结果：

```text
route = R1-BasisPortfolioNoProgress
required_artifact_missing_count = 2
```

修复文件：

```text
experiments/run_v1230_finalize_basis_first.py
```

修复内容：

```text
1. runner 输出 v1230_fhq_dynamic_monitor_ablation.csv，finalizer 额外写规范副本 v1230_fhq_dynamic_monitor.csv。
2. required manifest 在 manifest/zip 生成后重算，避免把即将生成的 v1230_required_artifact_manifest.csv 和 v1230_code_review_packet.zip 误报为缺失。
```

验证命令：

```bash
python -m py_compile experiments/run_v1230_finalize_basis_first.py && \
conda run -n kan python experiments/run_v1230_finalize_basis_first.py \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_finalize_manifest_fix.log
```

结果：

```text
route = R1-BasisPortfolioNoProgress
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
code_review_packet_entries = 92
code_review_packet_sha256 = 以最终 v1230_route_decision.json 为准
```

## 11. 日志写入后的最终打包复核

执行命令：

```bash
conda run -n kan python experiments/run_v1230_finalize_basis_first.py \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_finalize_after_logs_final.log
```

结果：

```text
route = R1-BasisPortfolioNoProgress
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
mlp_functional_candidate_rows = 405
mlp_functional_control_rows = 2430
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
code_review_packet_entries = 92
code_review_packet_sha256 = 以最终 v1230_route_decision.json 为准
```

## 12. 用户继续要求后的 memory accounting re-audit

继续原因：

```text
route = R1-BasisPortfolioNoProgress
basis_family_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
```

上一轮结论指出 classic runner 可能存在 memory accounting / basis workspace bottleneck，因此本轮先修复 memory ratio 统计口径并重跑 hardening/fallback，而不是继续扩展同一 alias 小网格。

修改文件：

```text
experiments/run_v1224_classic_hardening.py
```

修改内容：

```text
1. MLP reference 与 family run 均记录 memory baseline allocated bytes。
2. 新增 incremental peak memory bytes。
3. memory_ratio_vs_mlp 改为 family_incremental_peak_memory_bytes / mlp_incremental_peak_memory_bytes。
4. 原始 peak/raw ratio 保留为 memory_ratio_raw_vs_mlp。
5. gate 阈值不放宽，仍要求 memory_ratio_vs_mlp <= 1.0。
```

语法检查命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_classic_hardening.py experiments/run_v1230_finalize_basis_first.py
```

结果：

```text
py_compile pass
```

Hardening memory-fix 重跑命令：

```bash
conda run -n kan python experiments/run_v1224_classic_hardening.py \
  --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 \
  --artifact-prefix v1230_classic_basis_hardening \
  --run-id v1230_classic_basis_hardening_memoryfix \
  --families D-RAT4-GroupRationalSharedDenom,D-RAT6-RationalTaskTrajectoryWarmNoExtraMemV2,D-RAT3-DenStateFP16Checkpoint,D-RAT5-RationalLineCResidualMix,D-RAT1-GroupWorkspaceRecomputeV3,D-WAV5-LocalTailCoverageMonitor,D-FOU5-MultiScaleLowK,D-CHE5-LineCEnergyMonitor,D-RBF6-IdentityCompactRBFResidual \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 1024 --val-size 512 --epochs 8 --batch-size 128 \
  --linec-batch-size 64 --linec-sketch-dim 24
```

结果：

```text
rows = 81
summary_rows = 9
exploration_pass_rows = 0
```

Fallback memory-fix 重跑命令：

```bash
conda run -n kan python experiments/run_v1224_classic_hardening.py \
  --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 \
  --artifact-prefix v1230_classic_basis_fallback \
  --run-id v1230_classic_basis_fallback_memoryfix \
  --families D-RAT2-ReadoutGradChunkedV2,D-CHE3-K4-roleWiseEnergyCap,D-CHE4-K3K4-lowDegreeResidual,D-WAV4-ScaleDiversitySmallResidual,D-RBF5-TriangularBumpApprox,D-FOU3-LearnedAmplitudeResidual,D-FOU4-LateEnableK4FromK2 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 1024 --val-size 512 --epochs 8 --batch-size 128 \
  --linec-batch-size 64 --linec-sketch-dim 24
```

结果：

```text
rows = 63
summary_rows = 7
exploration_pass_rows = 0
```

memory accounting 复核结果：

```text
hardening max memory_ratio_vs_mlp ~= 9.967
hardening max memory_ratio_raw_vs_mlp ~= 2.102
fallback max memory_ratio_vs_mlp ~= 9.967
fallback max memory_ratio_raw_vs_mlp ~= 2.102
```

finalizer 复核命令：

```bash
conda run -n kan python experiments/run_v1230_finalize_basis_first.py
```

结果：

```text
route = R1-BasisPortfolioNoProgress
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
code_review_packet_sha256 = 以最终 v1230_route_decision.json 为准
```

## 13. 用户要求后的 core-to-dgkan refactor

继续原因：用户指出核心代码不应写在 runner 里，应沉淀到 `dgkan`。本节只记录实际完成的架构修复和 smoke，不改变任何既有实验 gate。

修改文件：

```text
dgkan/diagnostics/__init__.py
dgkan/diagnostics/classic_basis.py
dgkan/functional/mlp_functional.py
dgkan/training/eval.py
experiments/run_v1224_classic_hardening.py
experiments/run_v1230_mlp_functional.py
experiments/run_v1230_finalize_basis_first.py
```

修改内容：

```text
1. CLASSIC_FAMILY_SPECS、parse helpers、MLP reference training、incremental memory accounting 移到 dgkan/diagnostics/classic_basis.py。
2. MLP functional SOURCE_CANDIDATES / CONTROL_IDS / functional_objective / delta helpers 移到 dgkan/functional/mlp_functional.py。
3. classification_basic 移到 dgkan/training/eval.py。
4. run_v1224_classic_hardening.py 与 run_v1230_mlp_functional.py 只保留参数解析、数据加载、训练编排和 artifact 写出。
5. finalizer 的 code review manifest / implementation readback / packet code list 改为指向 dgkan 核心模块。
```

语法检查命令：

```bash
conda run -n kan python -m py_compile dgkan/diagnostics/__init__.py dgkan/diagnostics/classic_basis.py dgkan/functional/mlp_functional.py dgkan/training/eval.py experiments/run_v1224_classic_hardening.py experiments/run_v1230_mlp_functional.py experiments/run_v1230_finalize_basis_first.py
```

结果：

```text
py_compile pass
```

导入审计命令：

```bash
conda run -n kan python -c "from dgkan.diagnostics.classic_basis import CLASSIC_FAMILY_SPECS; from dgkan.functional.mlp_functional import SOURCE_CANDIDATES, CONTROL_IDS; import experiments.run_v1224_classic_hardening as d; import experiments.run_v1230_mlp_functional as m; ..."
```

结果：

```text
classic_specs = 64
runner_specs_same = True
mlp_sources = 5
controls = 5
runner_sources_same = True
```

Classic refactor smoke command：

```bash
conda run -n kan python experiments/run_v1224_classic_hardening.py   --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230   --artifact-prefix v1230_core_refactor_classic_smoke   --run-id v1230_core_refactor_classic_smoke   --families D-CHE1-K3-degreeEnergyCap   --datasets MNIST --seeds 0   --train-size 64 --val-size 32 --epochs 1 --batch-size 32   --linec-batch-size 16 --linec-sketch-dim 4 --no-download
```

结果：

```text
rows = 1
summary_rows = 1
exploration_pass_rows = 0
```

MLP functional refactor smoke command：

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py   --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230   --artifact-prefix v1230_core_refactor_mlp_smoke   --datasets MNIST --seeds 0 --train-seed-bases 12300400   --candidates M-F1-ActivationCovTransport   --train-size 64 --val-size 32 --epochs 1 --batch-size 32   --functional-batch 32 --linec-batch-size 16 --linec-sketch-dim 4   --linec-seeds 12309500 --no-download
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
Core refactor 后 finalizer 打包命令：

```bash
conda run -n kan python experiments/run_v1230_finalize_basis_first.py
```

结果：

```text
route = R1-BasisPortfolioNoProgress
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
code_review_packet_entries = 103
code_review_packet_sha256 = 以最终 v1230_route_decision.json 为准
```

## 14. 用户继续要求后的 M-F6 input-jitter functional value source repair

继续原因：

```text
route = R1-BasisPortfolioNoProgress
basis_family_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
promotion_allowed = 0
```

前一轮已经修复 memory accounting，并把核心代码沉淀到 `dgkan`。本轮不继续排列 M-F1..M-F5，而是新增一个不同机制的 MLP functional value source：无标签输入扰动一致性。

修改文件：

```text
dgkan/functional/mlp_functional.py
experiments/run_v1230_mlp_functional.py
```

新增候选：

```text
M-F6-InputJitterConsistency
```

机制说明：

```text
1. 对 train-stream x 加确定性小噪声并 clamp 到输入范围。
2. 最小化 jitter 前后 logits 与 hidden representation 的一致性误差。
3. 不读取 label / CE vector / validation / test / query batch / LineC target。
4. 使用既有 matched controls：TaskOnlyAdamW, NoOpMatchedOverhead, RandomMatchedNorm, AdamWParallelDirection(control only), SNROnlyAudit。
```

语法检查命令：

```bash
conda run -n kan python -m py_compile dgkan/functional/mlp_functional.py experiments/run_v1230_mlp_functional.py experiments/run_v1230_finalize_basis_first.py
```

结果：

```text
py_compile pass
```

Smoke command：

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 --artifact-prefix v1230_mf6_input_jitter_smoke --datasets MNIST --seeds 0 --train-seed-bases 12300400 --candidates M-F6-InputJitterConsistency --train-size 64 --val-size 32 --epochs 1 --batch-size 32 --functional-batch 32 --linec-batch-size 16 --linec-sketch-dim 4 --linec-seeds 12309500 --no-download
```

Smoke result：

```text
candidate_rows = 1
control_rows = 6
linec_rows = 2
gate_rows = 1
any_exploration_pass = 0
any_official_pass = 0
```

window=3 command：

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 --artifact-prefix v1230_mf6_input_jitter_w3 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-seed-bases 12300400,12301400,12302400 --candidates M-F6-InputJitterConsistency --train-size 512 --val-size 256 --epochs 3 --batch-size 128 --functional-batch 128 --linec-batch-size 32 --linec-sketch-dim 8 --linec-seeds 12309500,12310600,12311600,12312600,12313600 --no-download
```

window=3 result：

```text
candidate_rows = 27
control_rows = 162
linec_rows = 270
gate_rows = 1
any_exploration_pass = 0
any_official_pass = 0
```

window=5 command：

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 --artifact-prefix v1230_mf6_input_jitter_w5 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-seed-bases 12300400,12301400,12302400 --candidates M-F6-InputJitterConsistency --train-size 512 --val-size 256 --epochs 5 --batch-size 128 --functional-batch 128 --linec-batch-size 32 --linec-sketch-dim 8 --linec-seeds 12309500,12310600,12311600,12312600,12313600 --no-download
```

window=5 result：

```text
candidate_rows = 27
control_rows = 162
linec_rows = 270
gate_rows = 1
any_exploration_pass = 0
any_official_pass = 0
```

window=10 command：

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py --out-dir results/v12_30_basis_first_functional_also_mlp/official_v1230 --artifact-prefix v1230_mf6_input_jitter_w10 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-seed-bases 12300400,12301400,12302400 --candidates M-F6-InputJitterConsistency --train-size 512 --val-size 256 --epochs 10 --batch-size 128 --functional-batch 128 --linec-batch-size 32 --linec-sketch-dim 8 --linec-seeds 12309500,12310600,12311600,12312600,12313600 --no-download
```

window=10 result：

```text
candidate_rows = 27
control_rows = 162
linec_rows = 270
gate_rows = 1
any_exploration_pass = 0
any_official_pass = 0
```

合并结果：

```text
v1230_mlp_functional_candidates.csv rows = 486
v1230_mlp_functional_controls.csv rows = 2916
v1230_mlp_functional_linec.csv rows = 4860
v1230_mlp_functional_gate.csv rows = 18
```

M-F6 三窗口摘要：

| window | rows | mean_source_vs_noop | mean_source_vs_control | mean_CouplingR2_delta | max_LineC_pass_count | exploration_pass_rows |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 27 | 0.0 | -0.0013020833333333333 | -0.0007315950322315098 | 5 | 0 |
| 5 | 27 | -0.00028935185185185184 | -0.001591435185185185 | 0.0012489990826862628 | 5 | 0 |
| 10 | 27 | -0.00014467592592592592 | -0.001880787037037037 | 0.0013354520241659618 | 5 | 0 |

finalizer command：

```bash
conda run -n kan python experiments/run_v1230_finalize_basis_first.py
```

finalizer result：

```text
route = R1-BasisPortfolioNoProgress
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
mlp_functional_candidate_rows = 486
mlp_functional_control_rows = 2916
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
code_review_packet_sha256 = 以最终 v1230_route_decision.json 为准
```

## 15. 用户再次追问后的 stop-contract 复核

本次只读复核命令：

```bash
conda run -n kan python -c "import json; from pathlib import Path; p=Path("results/v12_30_basis_first_functional_also_mlp/official_v1230/v1230_route_decision.json"); r=json.loads(p.read_text()); ..."
```

复核结果：

```text
route = R1-BasisPortfolioNoProgress
minimum_success = Minimum Success B
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
basis_family_progress_count = 0
mlp_functional_candidate_rows = 486
mlp_functional_control_rows = 2916
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_sha256 = 以最终 v1230_route_decision.json 为准
```

本次没有新增实验。原因：M-F6 已经测试新的 input-jitter consistency value source，仍无法击败 matched controls；我已经不确定继续添加同类单项正则式 M-F objective 会形成有效机制。



## 16. 用户再次追问后的 stop-contract 复核 2

本次先尝试只读复核命令：

```bash
python -c "import json; from pathlib import Path; p=Path("results/v12_30_basis_first_functional_also_mlp/official_v1230/v1230_route_decision.json"); r=json.loads(p.read_text()); keys=["route","minimum_success","official_success_reached","p4_pass","promotion_allowed","final_stop_allowed","hard_compute_budget_exhausted","fallback_depth","fallback_all_executed","required_artifact_missing_count","basis_family_near_pass_count","basis_family_progress_count","mlp_functional_candidate_rows","mlp_functional_control_rows","mlp_functional_exploration_pass_rows","mlp_functional_official_pass_rows","line_a_near_anchor_pass_count","provenance_violation_count","code_review_packet_entries","code_review_packet_sha256"]; print("\n".join(f"{k}={r.get(k)}" for k in keys))"
```

结果：

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

改用已批准的 conda 前缀重跑只读复核：

```bash
conda run -n kan python -c "import json; from pathlib import Path; p=Path("results/v12_30_basis_first_functional_also_mlp/official_v1230/v1230_route_decision.json"); r=json.loads(p.read_text()); keys=["route","minimum_success","official_success_reached","p4_pass","promotion_allowed","final_stop_allowed","hard_compute_budget_exhausted","fallback_depth","fallback_all_executed","required_artifact_missing_count","basis_family_near_pass_count","basis_family_progress_count","mlp_functional_candidate_rows","mlp_functional_control_rows","mlp_functional_exploration_pass_rows","mlp_functional_official_pass_rows","line_a_near_anchor_pass_count","provenance_violation_count","code_review_packet_entries","code_review_packet_sha256"]; print("\n".join(f"{k}={r.get(k)}" for k in keys))"
```

复核结果：

```text
route = R1-BasisPortfolioNoProgress
minimum_success = Minimum Success B
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
basis_family_progress_count = 0
mlp_functional_candidate_rows = 486
mlp_functional_control_rows = 2916
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_entries = 123
code_review_packet_sha256 = 以最终 v1230_route_decision.json 为准
```

本次没有新增实验。原因：v12.30 已执行 fallback depth 6、M-F1..M-F6 matched controls、memory accounting re-audit 和 core-to-dgkan refactor；我已经不确定继续在当前 classic basis alias / MLP single-objective functional value-source family 上添加局部变体会形成有效机制。

日志更新后重新执行 finalizer：

```bash
conda run -n kan python experiments/run_v1230_finalize_basis_first.py \
  2>&1 | tee results/v12_30_basis_first_functional_also_mlp/official_v1230/logs/v1230_finalize_after_stop_contract_recheck2.log
```


补充：上面的带 `tee` finalizer 命令实际执行时失败：

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

为避免 shell 管道段再次触发 sandbox helper，改用无 `tee` finalizer 命令：

```bash
conda run -n kan python experiments/run_v1230_finalize_basis_first.py
```

该命令输出由本轮对话工具捕获；最终 route 与 code packet sha 以 `results/v12_30_basis_first_functional_also_mlp/official_v1230/v1230_route_decision.json` 为准。
