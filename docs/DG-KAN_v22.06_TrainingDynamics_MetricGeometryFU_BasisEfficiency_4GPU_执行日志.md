# DG-KAN v22.06 Training-Dynamics MetricGeometryFU BasisEfficiency 4GPU 执行日志

生成时间：2026-06-06 15:30:43 +0800

记录原则：只记录真实命令、输入、输出、状态和 blocker；不把未执行内容写成结果。

## 2026-06-06 15:30:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=0 failed=metric_solver_tests;mechanism_contracts

## 2026-06-06 15:33:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 15:33:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_efficiency_reconfirm.py --v2204-dir /home/chengshun.wang/DG-LCA/results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=2 D-CHE=1 D-FOU=1

## 2026-06-06 15:34:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_drat_drbf_officialization.py --device cuda:2 --official-transition-batch-sizes 512 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=2 pass=0

## 2026-06-06 15:35:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_drat_drbf_officialization.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/drat_drbf_multibatch_probe
```

- status: completed
- note: rows=8 pass=0

## 2026-06-06 15:37:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_drat_drbf_officialization.py --device cuda:2 --official-transition-batch-sizes 512 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=2 pass=0

## 2026-06-06 15:38:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_drat_drbf_officialization.py --device cuda:2 --official-transition-batch-sizes 512 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=2 pass=1

## 2026-06-06 15:39:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_drat_drbf_officialization.py --device cuda:2 --official-transition-batch-sizes 512 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=2 pass=2

## 2026-06-06 15:47:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v2 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=3 route=F0-MetricNoEffect best=MLP-V2206-S1-T0G0-ReadoutSolver

## 2026-06-06 15:53:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 15:58:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v3 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=5 route=C3-ActuationOnly best=MLP-V2206-S1-T0G0-ReadoutSolver

## 2026-06-06 16:03:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 16:10:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v4 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=7 route=C3-ActuationOnly best=MLP-V2206-C4-T0G0-SourceStateCarry

## 2026-06-06 16:22:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v5 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=10 route=C3-ActuationOnly best=MLP-V2206-C4-T0G0-SourceStateCarry-lr50

## 2026-06-06 16:23:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_terminal_preservation.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: entered_C5=0 decision=C5BlockedBeforeTerminalPreservation

## 2026-06-06 16:23:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_kan_source_mapping.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: decision=KANMappingNotEntered blocker=MLP_h3200_source_missing

## 2026-06-06 16:26:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: blocked
- note: clean_unzip_returncode=1

## 2026-06-06 16:26:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: route=C3-ActuationOnly promotion=0

## 2026-06-06 16:28:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 16:28:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: route=C3-ActuationOnly promotion=0

## 2026-06-06 16:40:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 16:44:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v6_full --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=10 route=C3-ActuationOnly best=MLP-V2206-C4-T0G0-SourceStateCarry-lr50

## 2026-06-06 16:52:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v6_full_fresh --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=16 route=C3-ActuationOnly best=MLP-V2206-C4-T6G0-EarlyWarmCarry-stop1200-lr50

## 2026-06-06 16:58:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 17:04:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v7_hidden_block --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=20 route=C3-ActuationOnly best=MLP-V2206-C4-T7G0-SourceStateCarry

## 2026-06-06 17:09:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 17:17:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v8_adaptive_hidden_block --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=23 route=C3-ActuationOnly best=MLP-V2206-C4-T7G0-SourceStateCarry

## 2026-06-06 17:21:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=0 failed=mechanism_contracts

## 2026-06-06 17:22:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 17:31:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v9_c3_gated_hidden_block --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=26 route=C3-ActuationOnly best=MLP-V2206-C4-T7G0-SourceStateCarry

## 2026-06-06 17:37:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 17:38:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: route=C3-ActuationOnly promotion=0

## 2026-06-06 17:40:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 17:40:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: route=C3-ActuationOnly promotion=0

## 2026-06-06 17:42:02 +0800

```bash
python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: blocked
- note: environment error only; system python lacked torch (`ModuleNotFoundError: No module named 'torch'`). No scientific result was taken from this command. The same gate was rerun with `/home/chengshun.wang/miniconda3/envs/kan/bin/python` and passed.

## 2026-06-06 17:42:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 17:42:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: route=C3-ActuationOnly promotion=0

## 2026-06-06 17:53:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode mechanism_contracts --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 0 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 17:58:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v10_compensated_hidden_block --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=4 route=C3-ActuationOnly best=MLP-V2206-C4-T10G0-EarlyWarmCarry-stop800-lr50

## 2026-06-06 18:00:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode mechanism_contracts --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 0 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 18:04:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v10_compensated_hidden_block_fresh --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=4 route=C3-ActuationOnly best=MLP-V2206-C4-T10G0-EarlyWarmCarry-stop800-lr50

## 2026-06-06 18:09:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode mechanism_contracts --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 0 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 18:11:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v11_periodic_compensated_hidden_block --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=2 route=C3-ActuationOnly best=MLP-V2206-C4-T10G0-PeriodicCarry-alt200-lr50

## 2026-06-06 18:17:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode mechanism_contracts --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 0 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 18:19:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v12_sgd_bootstrap_compensated_hidden_block --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=2 route=C3-ActuationOnly best=MLP-V2206-C4-T10G0-SGDBootstrap800Carry-lr50

## 2026-06-06 18:25:10 +0800

```bash
python - <<'PY'  # merge v9/v10/v11/v12 source-retention CSV artifacts into metric_solver_source_smoke_combined_v9_to_v12_readback
```

- status: completed
- note: v21_01_mlp_source_retention_summary.csv=50; v21_01_source_retention_summary.csv=50; v21_01_source_retention_matrix.csv=150; v21_01_source_retention_raw_traces.csv=1350

## 2026-06-06 18:25:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v12_readback --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=34 route=C3-ActuationOnly best=MLP-V2206-C4-T7G0-SourceStateCarry

## 2026-06-06 18:26:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v12_readback --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=34 route=C3-ActuationOnly best=MLP-V2206-C4-T6G0-EarlyWarmCarry-stop1200-lr50

## 2026-06-06 18:27:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_06_metric_solver_fu.py experiments/run_v22_06_finalize.py
```

- status: completed
- note: best-row ordering and combined finalizer patch compile check passed

## 2026-06-06 18:27:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 18:27:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 18:27:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: route=C3-ActuationOnly promotion=0

## 2026-06-06 18:39:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode mechanism_contracts --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 0 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/s013_v13_transport_contract_check
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 18:42:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v13_optimizer_transport --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v13_optimizer_transport_analysis
```

- status: completed
- note: rows=2 route=C3-ActuationOnly best=MLP-V2206-C4-T10G0-OptTransport-lr150

## 2026-06-06 18:48:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v14_optimizer_transport_strength_sanity --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v14_optimizer_transport_strength_sanity_analysis
```

- status: completed
- note: rows=0 route=F0-MetricNoEffect best=

## 2026-06-06 18:49:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v14_optimizer_transport_strength_sanity --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v14_optimizer_transport_strength_sanity_analysis
```

- status: completed
- note: rows=2 route=C3-ActuationOnly best=MLP-V2206-C4-T10G0-OptTransport-lr50

## 2026-06-06 18:50:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python merge v22.06 v9-v14 metric solver source readback
```

- status: completed
- note: {'summary': 62, 'summary_alias': 62, 'matrix': 186, 'traces': 1674}

## 2026-06-06 18:51:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v14_readback --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=38 route=C3-ActuationOnly best=MLP-V2206-C4-T10G0-OptTransport-lr150

## 2026-06-06 18:51:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 18:52:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v17_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_06_metric_solver_fu.py experiments/run_v22_06_finalize.py
```

- status: completed
- note: v22.06 optimizer-state transport edits compile pass

## 2026-06-06 18:52:15 +0800

```bash
for shard in 0 1 2 3; do /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --shard-count 4 --shard-index $shard --device cuda:$shard --run-label v22_06_metric_solver_source_smoke_v13_optimizer_transport --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v13_optimizer_transport --spec-ids MLP-V2206-C4-T10G0-OptTransport-lr150,MLP-V2206-C4-T10G0-OptTransport-lr300,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead; done
```

- status: completed
- note: four shards completed returncode=0; v13 optimizer-state transport smoke

## 2026-06-06 18:52:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --shard-count 4 --merge-only --run-label v22_06_metric_solver_source_smoke_v13_optimizer_transport --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v13_optimizer_transport --spec-ids MLP-V2206-C4-T10G0-OptTransport-lr150,MLP-V2206-C4-T10G0-OptTransport-lr300,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v13 merge-only completed

## 2026-06-06 18:52:15 +0800

```bash
for shard in 0 1 2 3; do /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --shard-count 4 --shard-index $shard --device cuda:$shard --run-label v22_06_metric_solver_source_smoke_v14_optimizer_transport_strength_sanity --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v14_optimizer_transport_strength_sanity --spec-ids MLP-V2206-C4-T10G0-OptTransport-lr50,MLP-V2206-C4-T10G0-OptTransport-lr100,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead; done
```

- status: completed
- note: four shards completed returncode=0; v14 low-strength sanity smoke

## 2026-06-06 18:52:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --shard-count 4 --merge-only --run-label v22_06_metric_solver_source_smoke_v14_optimizer_transport_strength_sanity --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v14_optimizer_transport_strength_sanity --spec-ids MLP-V2206-C4-T10G0-OptTransport-lr50,MLP-V2206-C4-T10G0-OptTransport-lr100,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v14 merge-only completed

## 2026-06-06 18:52:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 18:52:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: route=C3-ActuationOnly promotion=0

## 2026-06-06 19:02:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode mechanism_contracts --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 0 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/s013_t11_contract_check
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 19:08:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v15_early_observable_source_channel --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v15_early_observable_source_channel_analysis
```

- status: completed
- note: rows=3 route=C3-ActuationOnly best=MLP-V2206-C4-T11G0-OptTransport-lr150

## 2026-06-06 19:15:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v15_early_observable_source_channel_fresh --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v15_early_observable_source_channel_fresh_analysis
```

- status: completed
- note: rows=3 route=C3-ActuationOnly best=MLP-V2206-C4-T11G0-OptTransport-lr150

## 2026-06-06 19:22:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode mechanism_contracts --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 0 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/s013_t11_i4_contract_check
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 19:25:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v16_train_split_control_gate --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v16_train_split_control_gate_analysis
```

- status: completed
- note: rows=1 route=C3-ActuationOnly best=MLP-V2206-C4-T11G0-ControlRelativeGate-stop800-lr50

## 2026-06-06 19:27:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v16_readback --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=42 route=C3-ActuationOnly best=MLP-V2206-C4-T10G0-OptTransport-lr150

## 2026-06-06 19:28:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 19:28:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: route=C3-ActuationOnly promotion=0

## 2026-06-06 19:46:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v17_adamw_compatible_c4 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v17_adamw_compatible_c4_analysis
```

- status: completed
- note: rows=2 route=C3-ActuationOnly best=MLP-V2206-C4-T10G0-AdamWBootstrap800OptTransport-lr150

## 2026-06-06 19:49:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 0 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/s013_after_adamw_c4_code_gate
```

- status: completed
- note: pass=0 failed=import_closure

## 2026-06-06 19:50:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v17_adamw_compatible_c4_fresh --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v17_adamw_compatible_c4_fresh_analysis
```

- status: completed
- note: rows=2 route=C3-ActuationOnly best=MLP-V2206-C4-T11G0-AdamWControlGate-stop800-lr50

## 2026-06-06 19:52:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v17_readback --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=44 route=C3-ActuationOnly best=MLP-V2206-C4-T10G0-OptTransport-lr150

## 2026-06-06 19:52:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_terminal_preservation.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: entered_C5=0 decision=C5BlockedBeforeTerminalPreservation

## 2026-06-06 19:52:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_kan_source_mapping.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: decision=KANMappingNotEntered blocker=MLP_h3200_source_missing

## 2026-06-06 19:53:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root . --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/s013_after_adamw_c4_code_gate_selfcontained
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 19:54:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 19:54:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: route=C3-ActuationOnly promotion=0

## 2026-06-06 20:04:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v18_early_warm_then_transport --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v18_early_warm_then_transport_analysis
```

- status: completed
- note: rows=0 route=F0-MetricNoEffect best=

## 2026-06-06 20:05:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v18_early_warm_then_transport --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v18_early_warm_then_transport_analysis
```

- status: completed
- note: rows=0 route=F0-MetricNoEffect best=

## 2026-06-06 20:09:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v18_early_warm_then_transport_fresh --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v18_early_warm_then_transport_fresh_analysis
```

- status: completed
- note: rows=2 route=C3-ActuationOnly best=MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150

## 2026-06-06 20:11:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v18_readback --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=46 route=C3-ActuationOnly best=MLP-V2206-C4-T10G0-OptTransport-lr150

## 2026-06-06 20:18:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode mechanism_contracts --source-root . --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/s013_t12_soft_compensation_contract_check
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 20:22:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v19_soft_compensated_hidden_block --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v19_soft_compensated_hidden_block_analysis
```

- status: completed
- note: rows=2 route=C3-ActuationOnly best=MLP-V2206-C4-T12G0-SourceStateCarry

## 2026-06-06 20:23:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_metric_solver_fu.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: rows=48 route=C3-ActuationOnly best=MLP-V2206-C4-T10G0-OptTransport-lr150

## 2026-06-06 20:24:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_terminal_preservation.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: entered_C5=0 decision=C5BlockedBeforeTerminalPreservation

## 2026-06-06 20:24:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_kan_source_mapping.py --source-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: decision=KANMappingNotEntered blocker=MLP_h3200_source_missing

## 2026-06-06 20:24:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root . --self-contained-import-check 1 --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/s013_after_t12_soft_compensation_code_gate_selfcontained
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 20:24:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 20:24:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: route=C3-ActuationOnly promotion=0

## 2026-06-06 20:29:10 +0800

```bash
for i in 0 1 2 3; do /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --shard-count 4 --shard-index $i --device cuda:$i --run-label v22_06_metric_solver_source_smoke_v18_early_warm_then_transport --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v18_early_warm_then_transport --spec-ids MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150,MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead; done
```

- status: completed
- note: v18 initial source-retention shards; registration/allowlist gap, rows were controls-only; not used as scientific evidence; artifact journal has exact shard timestamps

## 2026-06-06 20:29:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --merge-only --run-label v22_06_metric_solver_source_smoke_v18_early_warm_then_transport --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v18_early_warm_then_transport --spec-ids MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150,MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v18 initial merge; route rows=0 after summary, treated as registration blocker

## 2026-06-06 20:29:10 +0800

```bash
for i in 0 1 2 3; do /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --shard-count 4 --shard-index $i --device cuda:$i --run-label v22_06_metric_solver_source_smoke_v18_early_warm_then_transport_fresh --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v18_early_warm_then_transport_fresh --spec-ids MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150,MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead; done
```

- status: completed
- note: v18 fresh source-retention shards after allowlist fix; used as scientific evidence

## 2026-06-06 20:29:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --merge-only --run-label v22_06_metric_solver_source_smoke_v18_early_warm_then_transport_fresh --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v18_early_warm_then_transport_fresh --spec-ids MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150,MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v18 fresh merge; rows=18 grouped=6 in artifact-local v21_01_command_journal.csv

## 2026-06-06 20:29:10 +0800

```bash
for i in 0 1 2 3; do /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --shard-count 4 --shard-index $i --device cuda:$i --run-label v22_06_metric_solver_source_smoke_v19_soft_compensated_hidden_block --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v19_soft_compensated_hidden_block --spec-ids MLP-V2206-C4-T12G0-SourceStateCarry,MLP-V2206-C4-T12G0-OptTransport-lr150,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead; done
```

- status: completed
- note: v19 T12 soft-compensated hidden-block source-retention shards; used as scientific evidence

## 2026-06-06 20:29:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --merge-only --run-label v22_06_metric_solver_source_smoke_v19_soft_compensated_hidden_block --out-dir results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_v19_soft_compensated_hidden_block --spec-ids MLP-V2206-C4-T12G0-SourceStateCarry,MLP-V2206-C4-T12G0-OptTransport-lr150,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v19 merge; rows=18 grouped=6 in artifact-local v21_01_command_journal.csv

## 2026-06-06 20:29:10 +0800

```bash
unzip -t results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/v22_06_results_bundle.zip && unzip -t results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/v22_06_code_review_packet.zip
```

- status: completed
- note: zip integrity checks passed for results bundle and code review packet

## 2026-06-06 20:29:10 +0800

```bash
sha256sum results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/v22_06_results_bundle.zip results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/v22_06_code_review_packet.zip
```

- status: completed
- note: results=f11203e48461a8c3da159aa562e1639f041ffeb11c46e7bc51f3af412e379f38; packet=35fe6efb52d05e3475ed99eca7e826e6fddb200c00d347a74167da4a52a5fc63

## 2026-06-06 20:29:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 20:29:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: route=C3-ActuationOnly promotion=0

## 2026-06-06 20:38:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_s013_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 20:38:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_06_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06
```

- status: completed
- note: route=C3-ActuationOnly promotion=0
