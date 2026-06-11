# DG-KAN v22.07 MetricDynamics FunctionalUpdate BasisEfficiency 4GPU 执行日志

生成时间：2026-06-06 21:34:53 +0800

记录原则：只记录真实命令、输入、输出、状态和 blocker；未执行项不写成完成。

## 2026-06-06 21:34:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_metric_dynamics_fu.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/debug_metric_smoke
```

- status: completed
- note: C0=0 C1_rows=0 C2_rows=0 C3_rows=0 route=C0-EstimatorBlocked

## 2026-06-06 21:35:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: started
- note: 4GPU queue launch

## 2026-06-06 21:35:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_efficiency_reconfirm.py --v2204-dir /home/chengshun.wang/DG-LCA/results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: rows=2 D-CHE=1 D-FOU=1 missing=backward_ratio_vs_mlp;functional_direction_ms;metric_solver_ms;no_materialize_complete

## 2026-06-06 21:35:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 21:35:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_drat_drbf_multibatch.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: rows=8 robust=1 near=8 telemetry_complete=0

## 2026-06-06 21:35:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_metric_dynamics_fu.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: C0=0 C1_rows=0 C2_rows=0 C3_rows=0 route=C0-EstimatorBlocked

## 2026-06-06 21:35:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_terminal_preservation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: entered_D0=0 decision=D0D1BlockedBeforeTerminalPreservation blocker=C0_source_estimator_predictor_gate;C1_target_contrast_gate;C2_metric_solver_gate;C3_source_formation_gate

## 2026-06-06 21:35:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_kan_source_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: decision=KANMappingNotEntered blocker=C0_source_estimator_predictor_gate;C1_target_contrast_gate;C2_metric_solver_gate;C3_source_formation_gate

## 2026-06-06 21:36:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 21:36:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C0-EstimatorBlocked promotion=0

## 2026-06-06 21:36:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-06 21:42:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_drat_drbf_multibatch.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: rows=8 robust=0 near=8 telemetry_complete=0 repair_rows=56 repair_micro_near=29

## 2026-06-06 21:44:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_drat_drbf_multibatch.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: rows=8 robust=0 near=8 telemetry_complete=0 repair_rows=56 repair_micro_near=29

## 2026-06-06 21:45:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 21:45:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C0-EstimatorBlocked promotion=0

## 2026-06-06 21:47:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 21:47:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C0-EstimatorBlocked promotion=0

## 2026-06-06 21:56:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_source_observability_repair.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: C0_repair=1 best=R3_E1_B3_control_gap p20=0.5 C1_repair_pass_rows=49 route=C1-TargetRepairOpened

## 2026-06-06 21:58:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_source_observability_repair.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: C0_repair=1 best=R3_E1_B3_control_gap p20=0.5 C1_repair_pass_rows=49 C2_repair_pass_rows=16 route=C2-TrueSolverRepairOpened-C3NotRun

## 2026-06-06 22:10:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_c3_repair_smoke.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: blocked
- note: manual SIGTERM after >210s: top9 x controls x h3200 recomputes metric solver every step; no C3 artifact emitted; retrying limited top2 h800 smoke

## 2026-06-06 22:12:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_c3_repair_smoke.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: rows=2 C3_smoke_pass=0 h800_pos=0 h3200_pos=0 route=C3-RepairSmokeBlocked

## 2026-06-06 22:14:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 22:14:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-RepairSmokeBlocked promotion=0

## 2026-06-06 22:14:49 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-06 22:16:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 22:16:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-RepairSmokeBlocked promotion=0

## 2026-06-06 22:16:14 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-06 22:17:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 22:17:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-RepairSmokeBlocked promotion=0

## 2026-06-06 22:17:25 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-06 22:22:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_c3_integration_repair.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: rows=8 early_pass=0 h800_pos=0 route=C3-IntegrationRepairBlocked

## 2026-06-06 22:27:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 22:27:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-IntegrationRepairBlocked promotion=0

## 2026-06-06 22:27:30 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-06 22:36:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_c3_semantic_target_repair.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: C1_pass=0 rows=8 early_pass=0 h800_pos=0 route=C3-SemanticTargetRepairBlocked

## 2026-06-06 22:38:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_c3_semantic_target_repair.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: C1_pass=0 rows=16 early_pass=0 h800_pos=0 route=C3-SemanticTargetRepairBlocked

## 2026-06-06 22:41:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 22:41:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-SemanticTargetRepairBlocked promotion=0

## 2026-06-06 22:41:30 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-06 22:46:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_source_theory_repair.py --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 12
```

- status: completed
- note: route=SourceTheoryRetainedObserverBlocked retained_pos=13 early_chain_pos=0 best=Q2_solver_feasible_retention auc=0.7200966572694322 p20=0.0

## 2026-06-06 22:49:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_source_theory_repair.py --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 12
```

- status: completed
- note: route=SourceTheoryModerateRetainedObserverSmokeOnly retained_pos=13 early_chain_pos=0 best=Q5_moderate_solver_feasible_retention smoke_auc=0.7950060410793395 smoke_p20=0.25

## 2026-06-06 22:51:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_c3_semantic_target_repair.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --selection-file results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_07_source_theory_top_candidates.csv
```

- status: completed
- note: C1_pass=0 rows=32 early_pass=7 h800_pos=9 route=C3-SemanticTargetRepairBlocked

## 2026-06-06 22:57:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_c3_semantic_target_repair.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 4 --steps 3200 --target-scale 16.0 --target-refresh 25 --selection-file results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_07_source_theory_top_candidates.csv
```

- status: completed
- note: C1_pass=0 rows=32 early_pass=7 h800_pos=9 route=C3-SemanticTargetRepairBlocked

## 2026-06-06 23:00:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 23:00:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-SemanticEarlyOpened-H3200Washout promotion=0

## 2026-06-06 23:00:52 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-06 23:08:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_c3_preservation_repair.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 3 --steps 3200 --preserve-start 800 --target-refresh 25
```

- status: completed
- note: rows=12 early_pass=2 h1600_pos=0 h3200_pos=0 route=C3-PreservationRepairBlocked

## 2026-06-06 23:15:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_c3_preservation_repair.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 3 --steps 3200 --preserve-start 800 --target-refresh 25
```

- status: completed
- note: rows=24 early_pass=5 h1600_pos=0 h3200_pos=0 route=C3-PreservationRepairBlocked

## 2026-06-06 23:24:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_c3_preservation_repair.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 3 --steps 3200 --preserve-start 800 --preserve-start-grid 400 --target-refresh 25
```

- status: completed
- note: rows=60 early_pass=9 h1600_pos=0 h3200_pos=0 route=C3-PreservationRepairBlocked

## 2026-06-06 23:28:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 23:28:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-PreservationRepairBlocked-H3200Washout promotion=0

## 2026-06-06 23:28:30 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-06 23:29:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 23:29:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-PreservationRepairBlocked-H3200Washout promotion=0

## 2026-06-06 23:29:59 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-06 23:39:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_retained_metric_solver_repair.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 2 --steps 3200 --solver-interval 100 --fu-lr 0.0001 --solver-grid M263-V2206MetricSolverT5G6LowNDSFU,M264-V2206MetricSolverT6G0DualMemoryFU,M269-V2206MetricSolverT11G0EarlyObservableFU,M270-V2206MetricSolverT12G0SoftCompensatedHiddenBlockFU
```

- status: completed
- note: rows=20 early=2 full=0 official_C2_rows=0 h1600_pos=0 h3200_pos=0 route=C3-RetainedMetricSolverBlocked

## 2026-06-06 23:46:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_stagefcp_hybrid_repair.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 2 --steps 3200 --semantic-refresh 25 --solver-interval 100 --fu-lr 0.0001 --solver-grid M264-V2206MetricSolverT6G0DualMemoryFU,M269-V2206MetricSolverT11G0EarlyObservableFU,M270-V2206MetricSolverT12G0SoftCompensatedHiddenBlockFU --switch-grid 400,800
```

- status: completed
- note: rows=12 early=3 full=0 official_C2_rows=0 h1600_pos=0 h3200_pos=0 route=C3-StageFCPHybridBlocked

## 2026-06-06 23:53:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_retained_metric_solver_repair.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 4 --steps 3200 --solver-interval 100 --fu-lr 0.0001 --solver-grid M263-V2206MetricSolverT5G6LowNDSFU,M264-V2206MetricSolverT6G0DualMemoryFU,M269-V2206MetricSolverT11G0EarlyObservableFU,M270-V2206MetricSolverT12G0SoftCompensatedHiddenBlockFU
```

- status: completed
- note: rows=38 early=5 full=0 official_C2_rows=0 h1600_pos=0 h3200_pos=0 route=C3-RetainedMetricSolverBlocked

## 2026-06-06 23:57:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 23:57:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-RetainedMetricSolverBlocked-H3200Washout promotion=0

## 2026-06-06 23:57:11 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-06 23:57:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 23:57:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-RetainedMetricSolverBlocked-H3200Washout promotion=0

## 2026-06-06 23:57:35 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-06 23:59:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 23:59:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-RetainedMetricSolverBlocked-H3200Washout promotion=0

## 2026-06-06 23:59:05 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-06 23:59:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 23:59:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-RetainedMetricSolverBlocked-H3200Washout promotion=0

## 2026-06-06 23:59:16 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-07 00:06:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_observer2_source_theory.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 3 --steps 3200 --target-refresh 25 --target-scale 1.0
```

- status: completed
- note: C1_pass=0 rows=15 early=0 full=0 h1600_pos=0 h3200_pos=0 route=C3-Observer2SourceTheoryBlocked

## 2026-06-07 00:09:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_observer2_source_theory.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 3 --steps 3200 --target-refresh 25 --target-scale 16.0
```

- status: completed
- note: C1_pass=0 rows=15 early=0 full=0 h1600_pos=0 h3200_pos=0 route=C3-Observer2SourceTheoryBlocked

## 2026-06-07 00:13:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-07 00:13:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-Observer2SourceTheoryBlocked promotion=0

## 2026-06-07 00:13:10 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-07 00:13:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-07 00:13:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-Observer2SourceTheoryBlocked promotion=0

## 2026-06-07 00:13:21 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-07 00:19:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_observer3_flow_theory.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 2 --steps 3200 --target-refresh 100 --target-scale 16.0 --fu-lr 0.0001
```

- status: completed
- note: C1_pass=0 rows=10 early=0 full=0 h1600_pos=0 h3200_pos=0 route=C3-Observer3FlowTheoryBlocked

## 2026-06-07 00:23:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-07 00:23:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-Observer3FlowTheoryBlocked promotion=0

## 2026-06-07 00:23:14 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-07 00:23:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-07 00:23:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-Observer3FlowTheoryBlocked promotion=0

## 2026-06-07 00:23:26 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-07 00:29:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_observer4_transfer_meta.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 2 --steps 3200 --target-refresh 50 --target-scale 16.0 --inner-lr 0.003 --fu-lr 0.0001
```

- status: completed
- note: C1_pass=0 rows=10 early=1 full=0 h1600_pos=0 h3200_pos=0 route=C3-Observer4TransferMetaBlocked

## 2026-06-07 00:32:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_observer4_transfer_meta.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 2 --steps 3200 --target-refresh 50 --target-scale 16.0 --inner-lr 0.03 --fu-lr 0.0001
```

- status: completed
- note: C1_pass=0 rows=10 early=1 full=0 h1600_pos=0 h3200_pos=0 route=C3-Observer4TransferMetaBlocked

## 2026-06-07 00:35:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-07 00:35:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-Observer4TransferMetaBlocked-H3200Washout promotion=0

## 2026-06-07 00:35:27 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-07 00:35:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-07 00:35:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-Observer4TransferMetaBlocked-H3200Washout promotion=0

## 2026-06-07 00:35:39 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-07 00:47:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_observer5_activation_manifold.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 2 --steps 3200 --target-refresh 50 --target-scale 16.0 --fu-lr 0.0001
```

- status: completed
- note: C1_pass=0 rows=10 early=1 full=0 h1600_pos=0 h3200_pos=0 route=C3-Observer5ActivationManifoldBlocked

## 2026-06-07 00:51:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_observer5_activation_manifold.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 2 --steps 3200 --target-refresh 50 --target-scale 16.0 --fu-lr 0.0001
```

- status: completed
- note: C1_pass=0 rows=16 early=1 full=0 h1600_pos=0 h3200_pos=0 route=C3-Observer5ActivationManifoldBlocked

## 2026-06-07 00:53:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-07 00:53:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-Observer5ActivationManifoldBlocked-H3200Washout promotion=0

## 2026-06-07 00:53:08 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-07 00:59:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_observer6_curvature_guard.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 2 --steps 3200 --target-refresh 100 --target-scale 16.0 --fu-lr 0.0001 --ema-beta 0.9
```

- status: completed
- note: C1_pass=0 rows=8 early=1 full=0 h1600_pos=0 h3200_pos=0 route=C3-Observer6CurvatureGuardBlocked

## 2026-06-07 01:03:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-07 01:03:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-Observer6CurvatureGuardBlocked-H3200Washout promotion=0

## 2026-06-07 01:03:31 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-07 01:11:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_observer7_control_nullspace.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 2 --steps 3200 --target-refresh 100 --target-scale 16.0 --fu-lr 0.0001 --flow-steps 4 --ema-beta 0.9
```

- status: completed
- note: C1_pass=0 rows=8 early=0 full=0 h1600_pos=0 h3200_pos=0 route=C3-Observer7ControlNullspaceBlocked

## 2026-06-07 01:17:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-07 01:17:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-Observer7ControlNullspaceBlocked promotion=0

## 2026-06-07 01:17:41 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-07 01:22:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_source_observer_nogo.py --out-dir results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-SourceObserverLocalNoGoBoundary local_no_go=1 observer_c1=0 observer_full=0 h1600_pos=0 h3200_pos=0

## 2026-06-07 01:24:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-07 01:24:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-SourceObserverLocalNoGoBoundary promotion=0

## 2026-06-07 01:24:44 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results

## 2026-06-07 01:25:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_s014_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-07 01:25:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_07_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: route=C3-SourceObserverLocalNoGoBoundary promotion=0

## 2026-06-07 01:26:01 +0800

```bash
zip artifact refresh /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07/v22_06_results_bundle.zip
```

- status: completed
- note: compat_bundle=v22_06_results_bundle.zip includes v22.07 code_review_packet and results
