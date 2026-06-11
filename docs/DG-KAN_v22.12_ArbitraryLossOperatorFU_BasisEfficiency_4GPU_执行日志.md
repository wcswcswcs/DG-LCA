# DG-KAN v22.12 ArbitraryLossOperatorFU BasisEfficiency 4GPU 执行日志

生成时间：2026-06-08 00:18:04 +0800

记录原则：只记录真实命令、文件、输入、输出、状态、blocker 与修复尝试；未执行项不写成完成。

## 2026-06-08 00:18:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_s019_truth_gate.py --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=S0.19-CodeSemanticTruthGatePass S0.19=1 blocker=

## 2026-06-08 00:18:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_operator_fu.py --seed 2212 --norm-scale 0.04 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: S2=S2-OperatorAtomNoGo S3=S3-BlockedBeforeOperatorSolve S4=S4-BlockedBeforeOperatorCommit blocker=S3_operator_variational_gate_failed

## 2026-06-08 00:20:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_operator_fu.py --seed 2212 --norm-scale 0.04 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: S2=S2-OperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricDynamicsCommitPass blocker=

## 2026-06-08 00:20:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_basis_efficiency.py --device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1 --seed 2212 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=S1-ArbitraryCotangentOperatorEfficiencyPass rows=360 fused_complete=0 blocker=

## 2026-06-08 00:21:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_basis_efficiency.py --device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1 --seed 2212 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=S1-ArbitraryCotangentOperatorEfficiencyPass rows=360 fused_complete=0 blocker=

## 2026-06-08 00:27:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_arbitrary_loss_horizon.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S5-TargetRetentionOnly_NotTaskUseful c3_adapters=0 c4_adapters=0 blocker=source_func_or_source_loss_or_control_gate_failed

## 2026-06-08 00:28:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_operator_fu.py --seed 2212 --norm-scale 0.08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: S2=S2-OperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricDynamicsCommitPass blocker=

## 2026-06-08 00:34:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_arbitrary_loss_horizon.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S5-TargetRetentionOnly_NotTaskUseful c3_adapters=0 c4_adapters=0 blocker=source_func_or_source_loss_or_control_gate_failed

## 2026-06-08 00:36:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S6-KANMappingNotEntered pass_rows=0 blocker=MLP_operator_C3_or_source_loss_gate_failed

## 2026-06-08 00:36:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212 --efficiency-device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-08 00:36:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_operator_fu.py --seed 2212 --norm-scale 0.08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: S2=S2-OperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricDynamicsCommitPass blocker=

## 2026-06-08 00:36:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_s019_truth_gate.py --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=S0.19-CodeSemanticTruthGatePass S0.19=1 blocker=

## 2026-06-08 00:36:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_basis_efficiency.py --device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1 --seed 2212 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=S1-ArbitraryCotangentOperatorEfficiencyPass rows=360 fused_complete=0 blocker=

## 2026-06-08 00:42:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_arbitrary_loss_horizon.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S5-TargetRetentionOnly_NotTaskUseful c3_adapters=0 c4_adapters=0 blocker=source_func_or_source_loss_or_control_gate_failed

## 2026-06-08 00:42:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S6-KANMappingNotEntered pass_rows=0 blocker=MLP_operator_C3_or_source_loss_gate_failed

## Final Artifact Summary
- final route: `R4-TargetRetentionOnly_NotTaskUseful`
- exploration_promotion_allowed: `0`
- official_promotion_allowed: `0`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`
- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-08 00:42:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=R4-TargetRetentionOnly_NotTaskUseful artifacts=54 bundle=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip

## 2026-06-08 00:42:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212 --efficiency-device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1
```

- status: completed
- note: queue_drained=1 blocked=0

## Final Artifact Summary
- final route: `R4-TargetRetentionOnly_NotTaskUseful`
- exploration_promotion_allowed: `0`
- official_promotion_allowed: `0`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`
- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-08 00:42:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=R4-TargetRetentionOnly_NotTaskUseful artifacts=59 bundle=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip

## 2026-06-08 00:42:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/logs/S7_finalize_after_queue_manifest.log

## 2026-06-08 00:46:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_operator_fu.py --seed 2212 --norm-scale 0.08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: S2=S2-OperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricDynamicsCommitPass blocker=

## 2026-06-08 00:52:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_arbitrary_loss_horizon.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S5-AdapterSpecificSourceOpened_NotUniversal c3_adapters=1 c4_adapters=0 blocker=source_func_or_source_loss_or_control_gate_failed

## 2026-06-08 01:03:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_arbitrary_loss_horizon.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S5-AdapterSpecificSourceOpened_NotUniversal c3_adapters=1 c4_adapters=1 blocker=source_func_or_source_loss_or_control_gate_failed

## 2026-06-08 01:12:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q dgkan experiments
```

- status: completed
- note: post row-mean preserving operator normalization repair compile check passed

## 2026-06-08 01:12:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_operator_fu.py --seed 2212 --norm-scale 0.08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: S2=S2-OperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricDynamicsCommitPass blocker=

## 2026-06-08 01:21:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_arbitrary_loss_horizon.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S5-AdapterSpecificSourceOpened_NotUniversal c3_adapters=1 c4_adapters=1 blocker=source_func_or_source_loss_or_control_gate_failed

## 2026-06-08 01:30:02 +0800

```bash
embedded diagnostic: compare O1/O4/mixed v22.12 operator targets over CE/ranking/MSE abbreviated horizons
```

- status: completed
- note: printed source_func/source_loss diagnostic; no gate files mutated

## 2026-06-08 01:35:56 +0800

```bash
embedded diagnostic: PID retention repair over O1/O4 v22.12 operator targets for CE/ranking/MSE abbreviated horizons
```

- status: completed
- note: printed PID repair diagnostic; no gate files mutated

## 2026-06-08 01:40:12 +0800

```bash
embedded diagnostic: stronger PID retention repair over O1/O1+O4 operator targets for CE/ranking/MSE abbreviated horizons
```

- status: completed
- note: printed stronger PID repair diagnostic; no gate files mutated

## 2026-06-08 01:46:38 +0800

```bash
embedded diagnostic: norm-scale scan for O1/O4 PID repair over CE/ranking/MSE abbreviated horizons
```

- status: completed
- note: printed norm-scale diagnostic; no gate files mutated

## 2026-06-08 01:55:16 +0800

```bash
embedded diagnostic: O4 scale/PID local scan for ranking+MSE C3 repair abbreviated horizons
```

- status: completed
- note: printed local scale/PID diagnostic; no gate files mutated

## 2026-06-08 02:05:22 +0800

```bash
embedded diagnostic: narrow O1 scale/lr/PID scan for ranking C3 repair abbreviated horizons
```

- status: completed
- note: printed O1 ranking narrow diagnostic; no gate files mutated

## 2026-06-08 02:08:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q dgkan experiments
```

- status: completed
- note: post repair diagnostics runner compile check passed

## 2026-06-08 02:09:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_repair_diagnostics.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: wrote v22_12_repair_diagnostic_summary.csv; diagnostic only, official route unchanged

## 2026-06-08 02:09:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212 --efficiency-device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-08 02:09:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_operator_fu.py --seed 2212 --norm-scale 0.08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: S2=S2-OperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricDynamicsCommitPass blocker=

## 2026-06-08 02:09:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_s019_truth_gate.py --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=S0.19-CodeSemanticTruthGatePass S0.19=1 blocker=

## 2026-06-08 02:10:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_basis_efficiency.py --device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1 --seed 2212 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=S1-ArbitraryCotangentOperatorEfficiencyPass rows=360 fused_complete=0 blocker=

## 2026-06-08 02:19:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_arbitrary_loss_horizon.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S5-AdapterSpecificSourceOpened_NotUniversal c3_adapters=1 c4_adapters=1 blocker=source_func_or_source_loss_or_control_gate_failed

## 2026-06-08 02:22:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S6-KANSourceExistsEfficiencyBlocked pass_rows=0 blocker=KANEfficiencyContractBlocked;KANTargetRetentionOnly;D-RAT/D-RBF PrimitiveKAN mapping not implemented in v22_12 runner

## Final Artifact Summary
- final route: `R3-AdapterSpecificSourceOpened_NotUniversal`
- exploration_promotion_allowed: `0`
- official_promotion_allowed: `0`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`
- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-08 02:22:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=R3-AdapterSpecificSourceOpened_NotUniversal artifacts=60 bundle=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip

## 2026-06-08 02:22:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212 --efficiency-device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1
```

- status: completed
- note: queue_drained=1 blocked=0

## Final Artifact Summary
- final route: `R3-AdapterSpecificSourceOpened_NotUniversal`
- exploration_promotion_allowed: `0`
- official_promotion_allowed: `0`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`
- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-08 02:22:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=R3-AdapterSpecificSourceOpened_NotUniversal artifacts=60 bundle=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip

## 2026-06-08 02:22:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/logs/S7_finalize_after_queue_manifest.log

## 2026-06-08 02:58:25 +0800

```bash
embedded diagnostic: non-random loss-interface consensus operator targets treating random/stable cotangents as controls
```

- status: interrupted
- note: manual stop after ~30min with no further stdout; partial stdout showed near misses but no confirmed C3/C4 beyond MSE, so convert bounded cases into v22_12 repair diagnostic runner

## 2026-06-08 03:05:39 +0800

```bash
embedded bounded diagnostic: ranking blend of supervised/nonrandom raw and smooth loss-interface consensus operators
```

- status: completed
- note: ranking-only bounded search; stdout captured in Codex session, official gate files unchanged

## 2026-06-08 03:13:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q dgkan experiments
```

- status: completed
- note: post-O10/source_loss_pid20 repair syntax check passed

## 2026-06-08 03:13:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_operator_fu.py --seed 2212 --norm-scale 0.08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: S2=S2-OperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricDynamicsCommitPass blocker=

## 2026-06-08 03:15:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q dgkan experiments
```

- status: completed
- note: post-norm-scale-0.16/O10 queue repair syntax check passed

## 2026-06-08 03:16:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_operator_fu.py --seed 2212 --norm-scale 0.16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: S2=S2-OperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricDynamicsCommitPass blocker=

## 2026-06-08 03:27:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_arbitrary_loss_horizon.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S5-ArbitraryLossOperatorHorizonPass c3_adapters=2 c4_adapters=1 blocker=

## 2026-06-08 03:29:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_repair_diagnostics.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: wrote v22_12_repair_diagnostic_summary.csv; diagnostic only, official route unchanged

## 2026-06-08 03:30:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212 --efficiency-device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-08 03:30:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_operator_fu.py --seed 2212 --norm-scale 0.16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: S2=S2-OperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricDynamicsCommitPass blocker=

## 2026-06-08 03:30:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_s019_truth_gate.py --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=S0.19-CodeSemanticTruthGatePass S0.19=1 blocker=

## 2026-06-08 03:30:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_basis_efficiency.py --device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1 --seed 2212 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=S1-ArbitraryCotangentOperatorEfficiencyPass rows=360 fused_complete=0 blocker=

## 2026-06-08 03:40:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_arbitrary_loss_horizon.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S5-ArbitraryLossOperatorHorizonPass c3_adapters=2 c4_adapters=1 blocker=

## 2026-06-08 03:42:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S6-KANSourceChannelMismatchConfirmed pass_rows=0 blocker=KANSourceChannelMismatchConfirmed;KANTargetRetentionOnly;D-RAT/D-RBF PrimitiveKAN mapping not implemented in v22_12 runner

## Final Artifact Summary
- final route: `R2-SourceOperatorNoGo`
- exploration_promotion_allowed: `0`
- official_promotion_allowed: `0`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`
- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-08 03:42:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=R2-SourceOperatorNoGo artifacts=60 bundle=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip

## 2026-06-08 03:42:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212 --efficiency-device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1
```

- status: completed
- note: queue_drained=1 blocked=0

## Final Artifact Summary
- final route: `R2-SourceOperatorNoGo`
- exploration_promotion_allowed: `0`
- official_promotion_allowed: `0`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`
- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-08 03:42:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=R2-SourceOperatorNoGo artifacts=60 bundle=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip

## 2026-06-08 03:42:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/logs/S7_finalize_after_queue_manifest.log

## 2026-06-08 03:47:18 +0800

```bash
embedded bounded diagnostic: KAN readout replay scale/interval repair scan for D-FOU/D-CHE after O10 functional pass
```

- status: completed
- note: stdout captured in Codex session; official gate files unchanged

## 2026-06-08 03:48:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q dgkan experiments
```

- status: completed
- note: post-K8 KAN repair scan syntax check passed

## 2026-06-08 03:49:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_s019_truth_gate.py --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=S0.19-CodeSemanticTruthGatePass S0.19=1 blocker=

## 2026-06-08 03:55:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S6-KANSourceChannelMismatchConfirmed pass_rows=0 blocker=KANSourceChannelMismatchConfirmed;KANTargetRetentionOnly;D-RAT/D-RBF PrimitiveKAN mapping not implemented in v22_12 runner

## Final Artifact Summary
- final route: `R2-SourceOperatorNoGo`
- exploration_promotion_allowed: `0`
- official_promotion_allowed: `0`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`
- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-08 03:55:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=R2-SourceOperatorNoGo artifacts=60 bundle=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip

## 2026-06-08 04:00:34 +0800

```bash
embedded bounded diagnostic: K9 KAN all/basis/readout target-gradient commit scan after K8 mismatch
```

- status: completed
- note: stdout captured in Codex session; official gate files unchanged

## 2026-06-08 04:05:43 +0800

```bash
embedded bounded diagnostic: KAN mapping on Delta-RankingAdapter C3 same-metric source after O10 functional pass
```

- status: completed
- note: stdout captured in Codex session; official gate files unchanged

## 2026-06-08 04:10:25 +0800

```bash
embedded bounded diagnostic: K10 KAN readout update-gain scan with matched-norm controls after K9 mismatch
```

- status: interrupted
- note: stopped oversized 56-row CPU scan after ~4min without stdout; replaced by smaller flushed candidate scan

## 2026-06-08 04:13:09 +0800

```bash
embedded bounded diagnostic: K10 small KAN readout gain candidates with matched-norm controls after K9 mismatch
```

- status: completed
- note: stdout captured in Codex session; official gate files unchanged

## 2026-06-08 04:15:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q dgkan experiments
```

- status: completed
- note: post-K10 official S6 rows syntax check passed

## 2026-06-08 04:15:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_s019_truth_gate.py --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=S0.19-CodeSemanticTruthGatePass S0.19=1 blocker=

## 2026-06-08 04:24:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S6-KANSourceChannelMismatchConfirmed pass_rows=0 blocker=KANSourceChannelMismatchConfirmed;KANTargetRetentionOnly;D-RAT/D-RBF PrimitiveKAN mapping not implemented in v22_12 runner

## Final Artifact Summary
- final route: `R2-SourceOperatorNoGo`
- exploration_promotion_allowed: `0`
- official_promotion_allowed: `0`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`
- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-08 04:24:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=R2-SourceOperatorNoGo artifacts=60 bundle=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip

## 2026-06-08 04:31:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q experiments/run_v22_12_kan_mapping.py experiments/run_v22_12_finalize.py
```

- status: completed
- note: post K11 basis-linearized KAN mapping repair compile check passed

## 2026-06-08 04:42:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S6-KANSourceChannelMismatchConfirmed pass_rows=0 blocker=KANSourceChannelMismatchConfirmed;KANTargetRetentionOnly;D-RAT/D-RBF PrimitiveKAN mapping not implemented in v22_12 runner

## Final Artifact Summary
- final route: `R2-SourceOperatorNoGo`
- exploration_promotion_allowed: `0`
- official_promotion_allowed: `0`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`
- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-08 04:43:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=R2-SourceOperatorNoGo artifacts=60 bundle=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip

## 2026-06-08 04:47:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q experiments/run_v22_12_kan_mapping.py experiments/run_v22_12_finalize.py
```

- status: completed
- note: post K12 D-FOU low-frequency init-variant KAN repair compile check passed

## 2026-06-08 04:59:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S6-KANSourceChannelMismatchConfirmed pass_rows=0 blocker=KANSourceChannelMismatchConfirmed;KANTargetRetentionOnly;D-RAT/D-RBF PrimitiveKAN mapping not implemented in v22_12 runner

## 2026-06-08 05:01:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q experiments/run_v22_12_kan_mapping.py experiments/run_v22_12_finalize.py
```

- status: completed
- note: post K13 D-FOU fan-scale source_loss boundary repair compile check passed

## 2026-06-08 05:15:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S6-DFOUReadoutSourceOpened_DCHEMismatch pass_rows=2 blocker=

## 2026-06-08 05:17:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q experiments/run_v22_12_kan_mapping.py experiments/run_v22_12_finalize.py
```

- status: completed
- note: post K13 route-blocker consistency repair compile check passed

## 2026-06-08 05:30:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S6-DFOUReadoutSourceOpened_DCHEMismatch pass_rows=2 blocker=D-CHE_source_channel_mismatch_after_DFOU_open

## Final Artifact Summary
- final route: `R6-DFOUReadoutSourceOpened_DCHEMismatch`
- exploration_promotion_allowed: `1`
- official_promotion_allowed: `0`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`
- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-08 05:31:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=R6-DFOUReadoutSourceOpened_DCHEMismatch artifacts=60 bundle=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip

## 2026-06-08 05:44:45 +0800

```bash
python - <<PY # K14 corrected-readout-layout diagnostic using system python
```

- status: failed
- note: ModuleNotFoundError: No module named torch; reran with kan env python

## 2026-06-08 05:44:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<PY # full K14 D-CHE corrected-readout-layout diagnostic grid
```

- status: killed
- note: diagnostic grid exceeded useful time budget; killed pids 386257/386253 without mutating gate files

## 2026-06-08 05:44:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<PY # shortened K14 D-CHE corrected-readout-layout diagnostic
```

- status: completed
- note: fan_scale_repair 25x0p08 diagnostic: source_func_h3200=6.6585355047136545 source_loss_h3200=0.00013247229526314186 source_loss_h4800=5.908433195145335e-05 delta_vs_MLP=5.289963683060225; candidate should be officialized as KANEfficiencyContractBlocked if carrier gate remains 0

## 2026-06-08 05:54:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: killed
- note: first K14 official S6 run exceeded useful time budget with identity/inputcross bracket rows; killed pid 390432 and reduced official K14 matrix to fan_scale_25x0p08 row, leaving brackets as diagnostics only

## 2026-06-08 05:54:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q experiments/run_v22_12_kan_mapping.py experiments/run_v22_12_finalize.py
```

- status: completed
- note: post K14 single-row route repair compile check passed

## 2026-06-08 06:02:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: killed
- note: second full S6 rerun with single K14 row still exceeded useful repeated-row recomputation budget; killed pid 395686 and moved K14 officialization to incremental K14-only runner that reuses existing S6 rows and computes only the new row

## 2026-06-08 06:02:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q experiments/run_v22_12_kan_mapping.py experiments/run_v22_12_kan_k14_repair.py experiments/run_v22_12_finalize.py
```

- status: completed
- note: post K14-only repair runner compile check passed

## 2026-06-08 06:02:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_k14_repair.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S6-DFOUReadoutSourceOpened_DCHEMismatch K14_decision=KANSourceChannelMismatchConfirmed K14_source_func_h3200=8.544725932180882 K14_source_loss_h3200=0.00013344238095669425 blocker=D-CHE_source_channel_mismatch_after_DFOU_open

## 2026-06-08 06:04:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q experiments/run_v22_12_kan_mapping.py experiments/run_v22_12_kan_k14_repair.py experiments/run_v22_12_finalize.py
```

- status: completed
- note: post K14 source-exploration/terminal-loss split compile check passed

## 2026-06-08 06:05:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_k14_repair.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212
```

- status: completed
- note: route=S6-DFOUSourceOpened_DCHESourceExistsEfficiencyBlocked K14_decision=KANEfficiencyContractBlocked K14_source_func_h3200=8.544725932180882 K14_source_loss_h3200=0.00013344238095669425 blocker=D-CHE_arbitrary_cotangent_efficiency_blocked_after_source_exists;D-CHE_source_loss_h4800_gate_after_source_exists

## Final Artifact Summary
- final route: `R7-KANSourceExistsEfficiencyBlocked`
- exploration_promotion_allowed: `1`
- official_promotion_allowed: `0`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`
- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-08 06:07:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=R7-KANSourceExistsEfficiencyBlocked artifacts=60 bundle=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip

## 2026-06-08 06:08:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_s019_truth_gate.py --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=S0.19-CodeSemanticTruthGatePass S0.19=1 blocker=

## Final Artifact Summary
- final route: `R7-KANSourceExistsEfficiencyBlocked`
- exploration_promotion_allowed: `1`
- official_promotion_allowed: `0`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`
- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-08 06:09:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=R7-KANSourceExistsEfficiencyBlocked artifacts=60 bundle=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip

## Final Artifact Summary
- final route: `R7-KANSourceExistsEfficiencyBlocked`
- exploration_promotion_allowed: `1`
- official_promotion_allowed: `0`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`
- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-08 11:30:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=R7-KANSourceExistsEfficiencyBlocked artifacts=60 bundle=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip

## 2026-06-08 11:30:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q experiments/run_v22_12_finalize.py
```

- status: completed
- note: post recap-analysis expansion compile check passed

## Final Artifact Summary
- final route: `R7-KANSourceExistsEfficiencyBlocked`
- exploration_promotion_allowed: `1`
- official_promotion_allowed: `0`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`
- key commands are also in `v22_12_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-08 11:30:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12
```

- status: completed
- note: route=R7-KANSourceExistsEfficiencyBlocked artifacts=60 bundle=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip
