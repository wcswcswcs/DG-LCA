# DG-KAN v22.10 ConstructiveRetainedSource FunctionalUpdate BasisEfficiency 4GPU 执行日志

生成时间：2026-06-07 12:44:08 +0800

记录原则：只记录真实命令、输入、输出、状态、blocker 与修复尝试；未执行项不写成完成。

## 2026-06-07 12:44:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 12:44:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 12:44:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 12:44:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=2 selected_attempt=fallback_lower_norm_0p02_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_source_atom_payload.pt

## 2026-06-07 12:44:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S3-VariationalSourceSolveNoGo pass_rows=0

## 2026-06-07 12:44:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-BlockedBeforeMetricDynamicsCommit pass_rows=0 blocker=S3_variational_source_solve_gate_failed

## 2026-06-07 12:44:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S5-BlockedBeforeHorizonSourceFormation c3=0 c4=0

## 2026-06-07 12:44:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: decision=KANMappingNotEntered blocker=MLP_C3_C4_source_formation_gate_failed

## 2026-06-07 12:44:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R3-VariationalSourceSolveNoGo artifacts=1339 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 12:44:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 12:45:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R3-VariationalSourceSolveNoGo artifacts=1343 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 12:46:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 12:46:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R3-VariationalSourceSolveNoGo artifacts=1406 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 12:48:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 12:48:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R3-VariationalSourceSolveNoGo artifacts=89 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 13:06:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=1 selected_attempt=initial_norm_0p08_split4_all payload=results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_source_atom_payload.pt

## 2026-06-07 13:06:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=4

## 2026-06-07 13:08:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-MetricDynamicsCommitNoGo pass_rows=0 blocker=projection_residual_Gf_gate;ActuationR2_gate;B2_transfer_gain_vs_random_gate;function_displacement_cos_gate

## 2026-06-07 13:11:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=1 blocker=

## 2026-06-07 13:14:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationNoGo c3=0 c4=0

## 2026-06-07 13:18:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationNoGo c3=0 c4=0

## 2026-06-07 13:24:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationPass c3=4 c4=4

## 2026-06-07 13:27:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANSourceChannelMismatchConfirmed blocker=KAN_specific_delta_or_source_horizon_gate_failed

## 2026-06-07 13:29:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANSourceChannelMismatchConfirmed blocker=KAN_specific_delta_or_source_horizon_gate_failed

## 2026-06-07 13:31:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 13:31:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 13:31:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 13:31:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=1 selected_attempt=initial_norm_0p08_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_source_atom_payload.pt

## 2026-06-07 13:31:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=4

## 2026-06-07 13:31:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=1 blocker=

## 2026-06-07 13:33:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationPass c3=4 c4=4

## 2026-06-07 13:33:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANSourceChannelMismatchConfirmed blocker=KAN_specific_delta_or_source_horizon_gate_failed

## 2026-06-07 13:33:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R7-KANSourceChannelMismatchConfirmed artifacts=90 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 13:33:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 13:33:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R7-KANSourceChannelMismatchConfirmed artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 13:33:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S7_finalize_after_queue_manifest.log

## 2026-06-07 13:44:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANRetainedSourceOpened blocker=

## 2026-06-07 13:45:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 13:45:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 13:45:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 13:45:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=1 selected_attempt=initial_norm_0p08_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_source_atom_payload.pt

## 2026-06-07 13:45:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=4

## 2026-06-07 13:45:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=1 blocker=

## 2026-06-07 13:47:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationPass c3=4 c4=4

## 2026-06-07 13:48:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANRetainedSourceOpened blocker=

## 2026-06-07 13:48:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 13:48:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 13:48:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 13:48:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S7_finalize_after_queue_manifest.log

## 2026-06-07 13:58:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 13:58:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,32,40,48,64,80,96,128 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=96

## 2026-06-07 14:02:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 14:02:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,32,40,48,64,80,96,128 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=128

## 2026-06-07 14:04:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 14:04:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 14:04:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=1 selected_attempt=initial_norm_0p08_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_source_atom_payload.pt

## 2026-06-07 14:04:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=4

## 2026-06-07 14:04:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=1 blocker=

## 2026-06-07 14:04:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 14:04:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,32,40,48,64,80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=128

## 2026-06-07 14:06:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationPass c3=4 c4=4

## 2026-06-07 14:06:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANRetainedSourceOpened blocker=

## 2026-06-07 14:06:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=95 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 14:06:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 14:06:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=95 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 14:06:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S7_finalize_after_queue_manifest.log

## 2026-06-07 14:07:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 14:07:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 14:07:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=1 selected_attempt=initial_norm_0p08_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_source_atom_payload.pt

## 2026-06-07 14:07:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=4

## 2026-06-07 14:07:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=1 blocker=

## 2026-06-07 14:08:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 14:08:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,32,40,48,64,80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=128

## 2026-06-07 14:10:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationPass c3=4 c4=4

## 2026-06-07 14:10:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANRetainedSourceOpened blocker=

## 2026-06-07 14:10:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=95 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 14:10:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 14:10:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=95 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 14:10:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S7_finalize_after_queue_manifest.log

## 2026-06-07 14:24:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 14:24:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,32,40,48,64,80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=128

## 2026-06-07 14:24:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockh-repair-hidden-grid 80,96,128 --official-blockh-grid 16,64,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=144

## 2026-06-07 14:38:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 14:38:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,32,40,48,64,80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=128

## 2026-06-07 14:38:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockh-repair-hidden-grid 80,96,128 --official-blockh-grid 16,64,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=144

## 2026-06-07 14:38:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-fastk2-repair-hidden-grid 80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal fastK2 repair scan]
```

- status: completed
- note: fastk2_scan_route=S1-D-RAT-D-RBFOfficialFastK2RepairBlocked rows=24

## 2026-06-07 14:45:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 14:45:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,32,40,48,64,80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=128

## 2026-06-07 14:45:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockh-repair-hidden-grid 80,96,128 --official-blockh-grid 16,64,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=144

## 2026-06-07 14:45:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-fastk2-repair-hidden-grid 80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal fastK2 repair scan]
```

- status: completed
- note: fastk2_scan_route=S1-D-RAT-D-RBFOfficialFastK2RepairBlocked rows=24

## 2026-06-07 14:45:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockb-repair-hidden-grid 80,96,128 --official-blockb-grid 16,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal blockB repair scan]
```

- status: completed
- note: blockb_scan_route=S1-D-RAT-D-RBFOfficialBlockBRepairBlocked rows=96

## 2026-06-07 14:46:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:2 --seed 2210
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 14:46:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 14:46:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=1 selected_attempt=initial_norm_0p08_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_source_atom_payload.pt

## 2026-06-07 14:46:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=4

## 2026-06-07 14:46:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=1 blocker=

## 2026-06-07 14:47:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 14:47:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,32,40,48,64,80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=128

## 2026-06-07 14:47:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockh-repair-hidden-grid 80,96,128 --official-blockh-grid 16,64,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=144

## 2026-06-07 14:47:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-fastk2-repair-hidden-grid 80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal fastK2 repair scan]
```

- status: completed
- note: fastk2_scan_route=S1-D-RAT-D-RBFOfficialFastK2RepairBlocked rows=24

## 2026-06-07 14:47:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockb-repair-hidden-grid 80,96,128 --official-blockb-grid 16,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal blockB repair scan]
```

- status: completed
- note: blockb_scan_route=S1-D-RAT-D-RBFOfficialBlockBRepairBlocked rows=96

## 2026-06-07 14:49:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationPass c3=4 c4=4

## 2026-06-07 14:49:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANRetainedSourceOpened blocker=

## 2026-06-07 14:49:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=107 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 14:49:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:2 --seed 2210
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 14:49:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=107 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 14:49:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S7_finalize_after_queue_manifest.log

## 2026-06-07 14:58:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128 --hidden 128 --train-size 512 --val-size 64 --profiler-repeats 1 --profiler-warmup 0 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/telemetry_sanity_v22_10
```

- status: blocked
- note: argument sanity attempt failed: v22_10 wrapper does not expose hidden/train-size/profiler flags; no scientific rows produced

## 2026-06-07 14:58:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_drat_drbf_repair.py --device cuda:2 --batch-sizes 128 --official-transition 1 --official-transition-batch-sizes 128 --hidden 128 --train-size 512 --val-size 64 --profiler-repeats 1 --profiler-warmup 0 --iters 1 --warmup 0 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/telemetry_sanity_v22_10
```

- status: completed
- note: telemetry sanity only: official transition rows wrote official_fused_runner_probe component fields for D-RAT/D-RBF; not used as final gate

## 2026-06-07 14:59:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:2 --seed 2210
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 14:59:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 14:59:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=1 selected_attempt=initial_norm_0p08_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_source_atom_payload.pt

## 2026-06-07 14:59:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=4

## 2026-06-07 14:59:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=1 blocker=

## 2026-06-07 15:00:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 15:00:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,32,40,48,64,80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=128

## 2026-06-07 15:00:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockh-repair-hidden-grid 80,96,128 --official-blockh-grid 16,64,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=144

## 2026-06-07 15:00:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-fastk2-repair-hidden-grid 80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal fastK2 repair scan]
```

- status: completed
- note: fastk2_scan_route=S1-D-RAT-D-RBFOfficialFastK2RepairBlocked rows=24

## 2026-06-07 15:00:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockb-repair-hidden-grid 80,96,128 --official-blockb-grid 16,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal blockB repair scan]
```

- status: completed
- note: blockb_scan_route=S1-D-RAT-D-RBFOfficialBlockBRepairBlocked rows=96

## 2026-06-07 15:01:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationPass c3=4 c4=4

## 2026-06-07 15:01:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANRetainedSourceOpened blocker=

## 2026-06-07 15:01:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=107 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 15:01:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:2 --seed 2210
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 15:01:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=107 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 15:01:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S7_finalize_after_queue_manifest.log

## 2026-06-07 15:11:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_singlelaunch_sanity
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_singlelaunch_sanity/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 15:11:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256 --official-repair-hidden-grid 48 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_singlelaunch_sanity [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=12

## 2026-06-07 15:11:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256 --official-blockh-repair-hidden-grid 80 --official-blockh-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_singlelaunch_sanity [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=8

## 2026-06-07 15:11:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256 --official-fastk2-repair-hidden-grid 80 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_singlelaunch_sanity [internal fastK2 repair scan]
```

- status: completed
- note: fastk2_scan_route=S1-D-RAT-D-RBFOfficialFastK2RepairBlocked rows=4

## 2026-06-07 15:11:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256 --official-blockb-repair-hidden-grid 80 --official-blockb-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_singlelaunch_sanity [internal blockB repair scan]
```

- status: completed
- note: blockb_scan_route=S1-D-RAT-D-RBFOfficialBlockBRepairBlocked rows=8

## 2026-06-07 15:13:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:2 --seed 2210
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 15:13:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 15:13:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=1 selected_attempt=initial_norm_0p08_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_source_atom_payload.pt

## 2026-06-07 15:13:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=4

## 2026-06-07 15:13:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=1 blocker=

## 2026-06-07 15:15:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationPass c3=4 c4=4

## 2026-06-07 15:15:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANRetainedSourceOpened blocker=

## 2026-06-07 15:22:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 15:22:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,32,40,48,64,80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=192

## 2026-06-07 15:22:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockh-repair-hidden-grid 24,32,40,48,64,80,96,128 --official-blockh-grid 16,64,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=384

## 2026-06-07 15:22:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-fastk2-repair-hidden-grid 24,32,40,48,64,80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal fastK2 repair scan]
```

- status: completed
- note: fastk2_scan_route=S1-D-RAT-D-RBFOfficialFastK2RepairBlocked rows=64

## 2026-06-07 15:22:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockb-repair-hidden-grid 24,32,40,48,64,80,96,128 --official-blockb-grid 16,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal blockB repair scan]
```

- status: completed
- note: blockb_scan_route=S1-D-RAT-D-RBFOfficialBlockBRepairBlocked rows=256

## 2026-06-07 15:22:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=150 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 15:22:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:2 --seed 2210
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 15:22:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=150 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 15:22:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S7_finalize_after_queue_manifest.log

## 2026-06-07 15:27:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lowhidden_sanity
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lowhidden_sanity/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 15:27:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 16,20 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lowhidden_sanity [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=48

## 2026-06-07 15:27:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockh-repair-hidden-grid 16,20 --official-blockh-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lowhidden_sanity [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=32

## 2026-06-07 15:27:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-fastk2-repair-hidden-grid 16,20 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lowhidden_sanity [internal fastK2 repair scan]
```

- status: completed
- note: fastk2_scan_route=S1-D-RAT-D-RBFOfficialFastK2RepairBlocked rows=16

## 2026-06-07 15:27:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockb-repair-hidden-grid 16,20 --official-blockb-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lowhidden_sanity [internal blockB repair scan]
```

- status: completed
- note: blockb_scan_route=S1-D-RAT-D-RBFOfficialBlockBRepairBlocked rows=32

## 2026-06-07 15:29:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=193 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 15:36:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_cefix_sanity
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_cefix_sanity/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 15:36:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_cefix_sanity [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=48

## 2026-06-07 15:36:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockh-repair-hidden-grid 24 --official-blockh-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_cefix_sanity [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=16

## 2026-06-07 15:36:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-fastk2-repair-hidden-grid 24 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_cefix_sanity [internal fastK2 repair scan]
```

- status: completed
- note: fastk2_scan_route=S1-D-RAT-D-RBFOfficialFastK2RepairBlocked rows=8

## 2026-06-07 15:36:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockb-repair-hidden-grid 24 --official-blockb-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_cefix_sanity [internal blockB repair scan]
```

- status: completed
- note: blockb_scan_route=S1-D-RAT-D-RBFOfficialBlockBRepairBlocked rows=16

## 2026-06-07 15:41:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_warps_sanity
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_warps_sanity/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 15:41:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_warps_sanity [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=96

## 2026-06-07 15:41:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockh-repair-hidden-grid 24 --official-blockh-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_warps_sanity [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=16

## 2026-06-07 15:41:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-fastk2-repair-hidden-grid 24 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_warps_sanity [internal fastK2 repair scan]
```

- status: completed
- note: fastk2_scan_route=S1-D-RAT-D-RBFOfficialFastK2RepairBlocked rows=8

## 2026-06-07 15:41:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockb-repair-hidden-grid 24 --official-blockb-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_warps_sanity [internal blockB repair scan]
```

- status: completed
- note: blockb_scan_route=S1-D-RAT-D-RBFOfficialBlockBRepairBlocked rows=16

## 2026-06-07 15:44:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_combo_sanity
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_combo_sanity/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 15:44:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_combo_sanity [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=128

## 2026-06-07 15:44:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockh-repair-hidden-grid 24 --official-blockh-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_combo_sanity [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=16

## 2026-06-07 15:44:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-fastk2-repair-hidden-grid 24 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_combo_sanity [internal fastK2 repair scan]
```

- status: completed
- note: fastk2_scan_route=S1-D-RAT-D-RBFOfficialFastK2RepairBlocked rows=8

## 2026-06-07 15:44:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockb-repair-hidden-grid 24 --official-blockb-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_combo_sanity [internal blockB repair scan]
```

- status: completed
- note: blockb_scan_route=S1-D-RAT-D-RBFOfficialBlockBRepairBlocked rows=16

## 2026-06-07 15:46:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,128 --official-blockh-repair-hidden-grid 24 --official-blockh-grid 16 --official-fastk2-repair-hidden-grid 24 --official-blockb-repair-hidden-grid 24 --official-blockb-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_cefix_sanity
```

- status: completed
- note: manual CE duplicate-cost repair focused sanity; candidate_groups=0; blocker remained forward_ratio

## 2026-06-07 15:46:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,128 --official-blockh-repair-hidden-grid 24 --official-blockh-grid 16 --official-fastk2-repair-hidden-grid 24 --official-blockb-repair-hidden-grid 24 --official-blockb-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_warps_sanity
```

- status: completed
- note: D-RAT warps1/warps2 launch-meta focused sanity; candidate_groups=0; warps worsened forward

## 2026-06-07 15:46:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,128 --official-blockh-repair-hidden-grid 24 --official-blockh-grid 16 --official-fastk2-repair-hidden-grid 24 --official-blockb-repair-hidden-grid 24 --official-blockb-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_combo_sanity
```

- status: completed
- note: D-RAT singlelaunch-blockH64/128 focused sanity; candidate_groups=0; combo worsened forward/step

## 2026-06-07 15:46:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:2 --seed 2210
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 15:46:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 15:46:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=1 selected_attempt=initial_norm_0p08_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_source_atom_payload.pt

## 2026-06-07 15:46:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=4

## 2026-06-07 15:46:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=1 blocker=

## 2026-06-07 15:48:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationPass c3=4 c4=4

## 2026-06-07 15:49:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANRetainedSourceOpened blocker=

## 2026-06-07 15:53:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 15:53:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 24,32,40,48,64,80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=512

## 2026-06-07 15:53:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockh-repair-hidden-grid 24,32,40,48,64,80,96,128 --official-blockh-grid 16,64,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=384

## 2026-06-07 15:53:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-fastk2-repair-hidden-grid 24,32,40,48,64,80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal fastK2 repair scan]
```

- status: completed
- note: fastk2_scan_route=S1-D-RAT-D-RBFOfficialFastK2RepairBlocked rows=64

## 2026-06-07 15:53:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockb-repair-hidden-grid 24,32,40,48,64,80,96,128 --official-blockb-grid 16,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal blockB repair scan]
```

- status: completed
- note: blockb_scan_route=S1-D-RAT-D-RBFOfficialBlockBRepairBlocked rows=256

## 2026-06-07 15:53:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=322 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 15:53:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:2 --seed 2210
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 15:53:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=322 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 15:53:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S7_finalize_after_queue_manifest.log

## 2026-06-07 16:03:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_rbf_singlelaunch_sanity
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_rbf_singlelaunch_sanity/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 16:03:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 32,40,48 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_rbf_singlelaunch_sanity [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=216

## 2026-06-07 16:03:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockh-repair-hidden-grid 32 --official-blockh-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_rbf_singlelaunch_sanity [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=16

## 2026-06-07 16:03:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-fastk2-repair-hidden-grid 32,40,48 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_rbf_singlelaunch_sanity [internal fastK2 repair scan]
```

- status: completed
- note: fastk2_scan_route=S1-D-RAT-D-RBFOfficialFastK2RepairBlocked rows=36

## 2026-06-07 16:03:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockb-repair-hidden-grid 32 --official-blockb-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_rbf_singlelaunch_sanity [internal blockB repair scan]
```

- status: completed
- note: blockb_scan_route=S1-D-RAT-D-RBFOfficialBlockBRepairBlocked rows=16

## 2026-06-07 16:10:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_low_hidden_sanity
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_low_hidden_sanity/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 16:10:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 8,12,16,20 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_low_hidden_sanity [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=288

## 2026-06-07 16:10:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockh-repair-hidden-grid 8,12,16,20 --official-blockh-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_low_hidden_sanity [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=64

## 2026-06-07 16:10:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-fastk2-repair-hidden-grid 8,12,16,20 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_low_hidden_sanity [internal fastK2 repair scan]
```

- status: completed
- note: fastk2_scan_route=S1-D-RAT-D-RBFOfficialFastK2RepairBlocked rows=48

## 2026-06-07 16:10:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockb-repair-hidden-grid 8,12,16,20 --official-blockb-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_low_hidden_sanity [internal blockB repair scan]
```

- status: completed
- note: blockb_scan_route=S1-D-RAT-D-RBFOfficialBlockBRepairBlocked rows=64

## 2026-06-07 16:15:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/kernels/fused_rbf.py experiments/run_v17_common.py experiments/run_v22_10_basis_efficiency_closure.py experiments/run_v22_10_finalize.py experiments/run_v22_10_common.py experiments/run_v22_10_s017_truth_gate.py
```

- status: completed
- note: manual syntax gate for v22.10 RBF singlelaunch + low-hidden formal scan edits

## 2026-06-07 16:15:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<PY [RBF singlelaunch logits/cache equivalence sanity on cuda:2]
```

- status: completed
- note: K2/K4 max_logit_abs=0 and max_h_abs=0 against old RBF two-launch path on 64-dim formal input shape

## 2026-06-07 16:15:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 32,40,48 --official-blockh-repair-hidden-grid 32 --official-blockh-grid 16 --official-fastk2-repair-hidden-grid 32,40,48 --official-blockb-repair-hidden-grid 32 --official-blockb-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_rbf_singlelaunch_sanity
```

- status: completed
- note: RBF singlelaunch focused scan: candidate_groups=0; forward_ratio remained blocker

## 2026-06-07 16:15:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 8,12,16,20 --official-blockh-repair-hidden-grid 8,12,16,20 --official-blockh-grid 16 --official-fastk2-repair-hidden-grid 8,12,16,20 --official-blockb-repair-hidden-grid 8,12,16,20 --official-blockb-grid 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_low_hidden_sanity
```

- status: completed
- note: low-hidden focused scan: candidate_groups=0 across candidate/blockH/fastK2/blockB

## 2026-06-07 16:15:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:2 --seed 2210
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 16:15:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 16:15:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=1 selected_attempt=initial_norm_0p08_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_source_atom_payload.pt

## 2026-06-07 16:15:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=4

## 2026-06-07 16:16:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=1 blocker=

## 2026-06-07 16:18:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationPass c3=4 c4=4

## 2026-06-07 16:18:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANRetainedSourceOpened blocker=

## 2026-06-07 16:22:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: inner_returncode=0 route=S1-BasisEfficiencyReconfirmed log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log

## 2026-06-07 16:22:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-repair-hidden-grid 8,12,16,20,24,32,40,48,64,80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal official candidate scan]
```

- status: completed
- note: candidate_scan_route=S1-D-RAT-D-RBFOfficialCandidateBlocked rows=864

## 2026-06-07 16:22:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockh-repair-hidden-grid 8,12,16,20,24,32,40,48,64,80,96,128 --official-blockh-grid 16,64,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal blockH repair scan]
```

- status: completed
- note: blockh_scan_route=S1-D-RAT-D-RBFOfficialBlockHRepairBlocked rows=576

## 2026-06-07 16:22:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-fastk2-repair-hidden-grid 8,12,16,20,24,32,40,48,64,80,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal fastK2 repair scan]
```

- status: completed
- note: fastk2_scan_route=S1-D-RAT-D-RBFOfficialFastK2RepairBlocked rows=144

## 2026-06-07 16:22:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --official-blockb-repair-hidden-grid 8,12,16,20,24,32,40,48,64,80,96,128 --official-blockb-grid 16,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 [internal blockB repair scan]
```

- status: completed
- note: blockb_scan_route=S1-D-RAT-D-RBFOfficialBlockBRepairBlocked rows=384

## 2026-06-07 16:22:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=408 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 16:22:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:2 --seed 2210
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 16:22:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=408 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 16:22:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S7_finalize_after_queue_manifest.log

## 2026-06-07 16:24:39 +0800

```bash
unzip -t results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip
```

- status: completed
- note: verification command completed: No errors detected in compressed data

## 2026-06-07 16:24:39 +0800

```bash
unzip -t results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip
```

- status: completed
- note: verification command completed: No errors detected in compressed data

## 2026-06-07 16:24:39 +0800

```bash
unzip -l results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip | rg "fused_rbf.py|fused_rational_k4.py|run_v17_common.py|run_v22_10_basis_efficiency_closure.py|run_v22_10_finalize.py|run_v22_10_common.py|run_v22_10_s017_truth_gate.py|efficiency_v17.py|fc_purekan_primitives.py"
```

- status: completed
- note: code packet contains all v22.10 efficiency-side modified source files

## 2026-06-07 16:25:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=408 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 16:44:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity
```

- status: completed
- note: route=S2-SourceAtomGenerationNoGo pass_rows=0 selected_attempt= payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity/v22_10_source_atom_payload.pt

## 2026-06-07 16:44:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity
```

- status: completed
- note: route=S3-BlockedBeforeVariationalSolve pass_rows=0

## 2026-06-07 16:44:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity
```

- status: completed
- note: route=S4-BlockedBeforeMetricDynamicsCommit pass_rows=0 blocker=S3_variational_source_solve_gate_failed

## 2026-06-07 16:44:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity --seed 2210
```

- status: completed
- note: route=S5-BlockedBeforeHorizonSourceFormation c3=0 c4=0

## 2026-06-07 16:44:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity --seed 2210
```

- status: completed
- note: decision=KANMappingNotEntered blocker=MLP_C3_C4_source_formation_gate_failed

## 2026-06-07 16:45:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity2
```

- status: completed
- note: route=S2-SourceAtomGenerationNoGo pass_rows=0 selected_attempt= payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity2/v22_10_source_atom_payload.pt

## 2026-06-07 16:45:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity2 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity2
```

- status: completed
- note: route=S3-BlockedBeforeVariationalSolve pass_rows=0

## 2026-06-07 16:45:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity2 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity2
```

- status: completed
- note: route=S4-BlockedBeforeMetricDynamicsCommit pass_rows=0 blocker=S3_variational_source_solve_gate_failed

## 2026-06-07 16:46:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity2 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity2 --seed 2210
```

- status: completed
- note: route=S5-BlockedBeforeHorizonSourceFormation c3=0 c4=0

## 2026-06-07 16:46:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity2 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity2 --seed 2210
```

- status: completed
- note: decision=KANMappingNotEntered blocker=MLP_C3_C4_source_formation_gate_failed

## 2026-06-07 16:47:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity3
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=2 selected_attempt=initial_norm_0p08_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity3/v22_10_source_atom_payload.pt

## 2026-06-07 16:47:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity3 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity3
```

- status: completed
- note: route=S3-VariationalSourceSolveNoGo pass_rows=0

## 2026-06-07 16:47:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity3 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity3
```

- status: completed
- note: route=S4-BlockedBeforeMetricDynamicsCommit pass_rows=0 blocker=S3_variational_source_solve_gate_failed

## 2026-06-07 16:47:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity3 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity3 --seed 2210
```

- status: completed
- note: route=S5-BlockedBeforeHorizonSourceFormation c3=0 c4=0

## 2026-06-07 16:47:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity3 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity3 --seed 2210
```

- status: completed
- note: decision=KANMappingNotEntered blocker=MLP_C3_C4_source_formation_gate_failed

## 2026-06-07 16:50:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity4
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=2 selected_attempt=initial_norm_0p08_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity4/v22_10_source_atom_payload.pt

## 2026-06-07 16:50:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity4 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity4
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=1

## 2026-06-07 16:50:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity4 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity4
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=1 blocker=

## 2026-06-07 16:52:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity4 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity4 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationNoGo c3=0 c4=0

## 2026-06-07 16:52:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity4 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity4 --seed 2210
```

- status: completed
- note: decision=KANMappingNotEntered blocker=MLP_C3_C4_source_formation_gate_failed

## 2026-06-07 16:59:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity4 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity5 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationPass c3=4 c4=4

## 2026-06-07 16:59:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity5 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lossagnostic_sanity5 --seed 2210
```

- status: completed
- note: decision=KANRetainedSourceOpened blocker=

## 2026-06-07 17:10:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 17:10:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=0 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 17:10:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=2 selected_attempt=initial_norm_0p08_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_source_atom_payload.pt

## 2026-06-07 17:10:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=1

## 2026-06-07 17:10:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=1 blocker=

## 2026-06-07 17:13:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationPass c3=4 c4=4

## 2026-06-07 17:13:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANRetainedSourceOpened blocker=

## 2026-06-07 17:33:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R0-CodePacketOrImplementationClosureFail artifacts=500 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 17:33:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: blocked
- note: queue_drained=0 blocked=1

## 2026-06-07 17:33:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R0-CodePacketOrImplementationClosureFail artifacts=500 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 17:33:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S7_finalize_after_queue_manifest.log

## 2026-06-07 17:35:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 17:35:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 17:35:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=2 selected_attempt=initial_norm_0p08_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_source_atom_payload.pt

## 2026-06-07 17:36:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=1

## 2026-06-07 17:36:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=1 blocker=

## 2026-06-07 17:46:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 17:46:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: blocked
- note: formal S1 stopped before CE-targeted trainpath; arbitrary-loss/upstream-gradient efficiency runner is not implemented

## 2026-06-07 17:46:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 17:46:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=2 selected_attempt=initial_norm_0p08_split4_all payload=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_source_atom_payload.pt

## 2026-06-07 17:46:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=1

## 2026-06-07 17:46:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=1 blocker=

## 2026-06-07 17:49:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationPass c3=4 c4=4

## 2026-06-07 17:49:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANRetainedSourceOpened blocker=

## 2026-06-07 17:49:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=86 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 17:49:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 17:49:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 17:49:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S7_finalize_after_queue_manifest.log

## 2026-06-07 17:52:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 17:53:16 +0800

```bash
rm -rf /tmp/v22_10_final_packet_check && mkdir -p /tmp/v22_10_final_packet_check && unzip -q results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip -d /tmp/v22_10_final_packet_check && /home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q /tmp/v22_10_final_packet_check/dgkan /tmp/v22_10_final_packet_check/experiments
```

- status: completed
- note: final v22_10_code_review_packet.zip clean unzip compileall_rc=0

## 2026-06-07 17:53:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 17:58:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1

## 2026-06-07 18:02:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: route=S5-HorizonSourceFormationPass c3=7 c4=7

## 2026-06-07 18:03:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210
```

- status: completed
- note: decision=KANRetainedSourceOpened blocker=

## 2026-06-07 18:03:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 18:04:17 +0800

```bash
rm -rf /tmp/v22_10_pid_packet_check && mkdir -p /tmp/v22_10_pid_packet_check && unzip -q results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip -d /tmp/v22_10_pid_packet_check && /home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q /tmp/v22_10_pid_packet_check/dgkan /tmp/v22_10_pid_packet_check/experiments
```

- status: completed
- note: post-PID v22_10_code_review_packet.zip clean unzip compileall_rc=0

## 2026-06-07 18:04:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 19:25:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip

## 2026-06-07 19:26:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_10_finalize.py
```

- status: completed
- note: expanded v22.10 recap generator syntax check returncode=0

## 2026-06-07 19:26:52 +0800

```bash
rm -rf /tmp/v22_10_rich_recap_packet_check && mkdir -p /tmp/v22_10_rich_recap_packet_check && unzip -q results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip -d /tmp/v22_10_rich_recap_packet_check && /home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q /tmp/v22_10_rich_recap_packet_check/dgkan /tmp/v22_10_rich_recap_packet_check/experiments
```

- status: completed
- note: rich-recap v22_10_code_review_packet.zip clean unzip compileall_rc=0

## 2026-06-07 19:27:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip
