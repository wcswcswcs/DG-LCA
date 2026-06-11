# DG-KAN v22.11 ArbitraryLoss ConstructiveFU BasisEfficiency 执行日志

生成时间：2026-06-07 20:57:24 +0800

记录原则：只记录真实命令、输入、输出、状态、blocker 与修复尝试；未执行项不写成完成。

## 2026-06-07 20:57:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --seed 2211 --efficiency-device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 20:57:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_source_atom_generation.py --seed 2211 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=S2-SourceAtomNoGo_v22.11 pass_rows=0 selected_attempt= blocker=loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;loss_agnostic_NDS_gate;loss_agnostic_B3_safety_gain_gate;corrupt_target_gap_gate;commutator_norm_ratio_gate;cotangent_same_direction_gate;loss_adapter_invariance_gate

## 2026-06-07 20:57:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=S3-BlockedBeforeVariationalSourceSolve pass_rows=0 blocker=S2_source_atom_gate_failed

## 2026-06-07 20:57:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_metric_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=S4-BlockedBeforeMetricDynamicsCommit pass_rows=0 blocker=S3_variational_source_solve_gate_failed

## 2026-06-07 20:57:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_s018_truth_gate.py --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=R0-CodeOrLossInterfaceGateFailed S0.18=0 blocker=clean_unzip_import_failed_or_not_requested

## 2026-06-07 20:57:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_arbitrary_loss_horizon.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --seed 2211
```

- status: completed
- note: route=S5-BlockedBeforeArbitraryLossHorizon c3=0 c4=0 blocker=S4_metric_dynamics_commit_gate_failed

## 2026-06-07 20:57:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_basis_efficiency.py --device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1 --seed 2211 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=S1-ArbitraryCotangentEfficiencyBlocked rows=120 blocker=ratio_or_fallback_kernel_gate_failed

## 2026-06-07 20:57:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --seed 2211
```

- status: completed
- note: route=S6-KANMappingNotEntered pass_rows=0 blocker=MLP_C3_C4_or_source_loss_gate_failed

## Final Artifact Summary

- final route: `R0-CodeOrLossInterfaceGateFailed`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip`
- key commands are also in `v22_11_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-07 20:57:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=R0-CodeOrLossInterfaceGateFailed artifacts=62 bundle=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip

## 2026-06-07 20:57:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --seed 2211 --efficiency-device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1
```

- status: completed
- note: queue_drained=1 blocked=0

## Final Artifact Summary

- final route: `R0-CodeOrLossInterfaceGateFailed`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip`
- key commands are also in `v22_11_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-07 20:57:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=R0-CodeOrLossInterfaceGateFailed artifacts=67 bundle=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip

## 2026-06-07 20:57:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/logs/S7_finalize_after_queue_manifest.log

## 2026-06-07 20:59:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --seed 2211 --efficiency-device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 20:59:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_source_atom_generation.py --seed 2211 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=S2-SourceAtomNoGo_v22.11 pass_rows=0 selected_attempt= blocker=loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;loss_agnostic_NDS_gate;loss_agnostic_B3_safety_gain_gate;corrupt_target_gap_gate;commutator_norm_ratio_gate;cotangent_same_direction_gate;loss_adapter_invariance_gate

## 2026-06-07 20:59:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=S3-BlockedBeforeVariationalSourceSolve pass_rows=0 blocker=S2_source_atom_gate_failed

## 2026-06-07 20:59:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_metric_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=S4-BlockedBeforeMetricDynamicsCommit pass_rows=0 blocker=S3_variational_source_solve_gate_failed

## 2026-06-07 20:59:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_s018_truth_gate.py --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=S0.18-CodeLossInterfaceTruthGatePass S0.18=1 blocker=

## 2026-06-07 20:59:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_arbitrary_loss_horizon.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --seed 2211
```

- status: completed
- note: route=S5-BlockedBeforeArbitraryLossHorizon c3=0 c4=0 blocker=S4_metric_dynamics_commit_gate_failed

## 2026-06-07 20:59:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_basis_efficiency.py --device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1 --seed 2211 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=S1-ArbitraryCotangentEfficiencyBlocked rows=120 blocker=ratio_or_fallback_kernel_gate_failed

## 2026-06-07 20:59:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --seed 2211
```

- status: completed
- note: route=S6-KANMappingNotEntered pass_rows=0 blocker=MLP_C3_C4_or_source_loss_gate_failed

## Final Artifact Summary

- final route: `R8-SourceOrHorizonWithEfficiencyBlocked`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip`
- key commands are also in `v22_11_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-07 20:59:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=R8-SourceOrHorizonWithEfficiencyBlocked artifacts=67 bundle=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip

## 2026-06-07 20:59:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --seed 2211 --efficiency-device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1
```

- status: completed
- note: queue_drained=1 blocked=0

## Final Artifact Summary

- final route: `R8-SourceOrHorizonWithEfficiencyBlocked`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip`
- key commands are also in `v22_11_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-07 20:59:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=R8-SourceOrHorizonWithEfficiencyBlocked artifacts=67 bundle=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip

## 2026-06-07 20:59:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/logs/S7_finalize_after_queue_manifest.log

## Final Artifact Summary

- final route: `R2-SourceAtomNoGoWithEfficiencyBlocked`
- results bundle: `results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip`
- code review packet: `results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip`
- key commands are also in `v22_11_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-07 21:00:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_finalize.py --out-dir results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=R2-SourceAtomNoGoWithEfficiencyBlocked artifacts=67 bundle=results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip code_review_packet=results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip

## 2026-06-07 21:02:01 +0800

```bash
rm -rf /tmp/v22_11_packet_check && mkdir -p /tmp/v22_11_packet_check && unzip -q results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip -d /tmp/v22_11_packet_check && /home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q /tmp/v22_11_packet_check/dgkan /tmp/v22_11_packet_check/experiments && cd /tmp/v22_11_packet_check && /home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY2'
import importlib
mods=['dgkan.fu.loss_interface','dgkan.fu.upstream_cotangent','dgkan.profiling.efficiency_v22_11','experiments.run_v22_11_full','experiments.run_v22_11_finalize']
for m in mods: importlib.import_module(m)
print('packet-clean-import-ok')
PY2
```

- status: completed
- note: final v22_11_code_review_packet.zip clean unzip compile/import ok; final route supersedes earlier interim finalize summaries

## Final Artifact Summary

- final route: `R2-SourceAtomNoGoWithEfficiencyBlocked`
- results bundle: `results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip`
- code review packet: `results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip`
- key commands are also in `v22_11_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-07 21:02:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_finalize.py --out-dir results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=R2-SourceAtomNoGoWithEfficiencyBlocked artifacts=67 bundle=results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip code_review_packet=results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip

## 2026-06-07 21:14:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_source_atom_generation.py --seed 2211 --out-dir /tmp/v22_11_s2_repair
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=3 selected_attempt=fallback_arbitrary_cotangent_invariant_0p04_split4_all blocker=

## 2026-06-07 21:14:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_variational_source_solve.py --source-dir /tmp/v22_11_s2_repair --out-dir /tmp/v22_11_s2_repair
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=4 blocker=

## 2026-06-07 21:16:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_basis_efficiency.py --device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1 --seed 2211 --out-dir /tmp/v22_11_s1_repair
```

- status: completed
- note: route=S1-ArbitraryCotangentEfficiencyPass rows=120 blocker=

## 2026-06-07 21:17:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_basis_efficiency.py --device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1 --seed 2211 --out-dir /tmp/v22_11_s1_repair2
```

- status: completed
- note: route=S1-ArbitraryCotangentEfficiencyPass rows=120 blocker=

## 2026-06-07 21:18:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_metric_commit.py --source-dir /tmp/v22_11_s2_repair --out-dir /tmp/v22_11_s2_repair
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=2 blocker=

## 2026-06-07 21:21:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_arbitrary_loss_horizon.py --source-dir /tmp/v22_11_s2_repair --out-dir /tmp/v22_11_s2_repair --seed 2211
```

- status: completed
- note: route=S5-ArbitraryLossHorizonNoGo c3=0 c4=0 blocker=source_func_or_source_loss_or_control_gate_failed

## 2026-06-07 21:22:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_kan_mapping.py --source-dir /tmp/v22_11_s2_repair --out-dir /tmp/v22_11_s2_repair --seed 2211
```

- status: completed
- note: route=S6-KANMappingNotEntered pass_rows=0 blocker=MLP_C3_C4_or_source_loss_gate_failed

## 2026-06-07 21:31:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_arbitrary_loss_horizon.py --source-dir /tmp/v22_11_s2_repair --out-dir /tmp/v22_11_s2_repair --seed 2211
```

- status: completed
- note: route=S5-ArbitraryLossHorizonNoGo c3=2 c4=0 blocker=source_func_or_source_loss_or_control_gate_failed

## 2026-06-07 21:31:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_kan_mapping.py --source-dir /tmp/v22_11_s2_repair --out-dir /tmp/v22_11_s2_repair --seed 2211
```

- status: completed
- note: route=S6-KANMappingNotEntered pass_rows=0 blocker=MLP_C3_C4_or_source_loss_gate_failed

## 2026-06-07 21:41:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_arbitrary_loss_horizon.py --source-dir /tmp/v22_11_s2_repair --out-dir /tmp/v22_11_s2_repair --seed 2211
```

- status: completed
- note: route=S5-ArbitraryLossHorizonPass c3=3 c4=1 blocker=

## 2026-06-07 21:42:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_kan_mapping.py --source-dir /tmp/v22_11_s2_repair --out-dir /tmp/v22_11_s2_repair --seed 2211
```

- status: completed
- note: route=S6-KANSourceChannelMismatchConfirmed pass_rows=0 blocker=KANTargetRetentionOnly

## 2026-06-07 21:50:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_kan_mapping.py --source-dir /tmp/v22_11_s2_repair --out-dir /tmp/v22_11_s2_repair --seed 2211
```

- status: completed
- note: route=S6-KANRetainedSourceOpened pass_rows=2 blocker=

## 2026-06-07 21:52:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --seed 2211 --efficiency-device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1
```

- status: started
- note: dynamic 4GPU queue launch

## 2026-06-07 21:52:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_source_atom_generation.py --seed 2211 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=S2-SourceAtomProgress pass_rows=3 selected_attempt=fallback_arbitrary_cotangent_invariant_0p04_split4_all blocker=

## 2026-06-07 21:52:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=S3-VariationalSourceSolvePass pass_rows=4 blocker=

## 2026-06-07 21:52:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_metric_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=S4-MetricDynamicsCommitPass pass_rows=2 blocker=

## 2026-06-07 21:52:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_s018_truth_gate.py --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=S0.18-CodeLossInterfaceTruthGatePass S0.18=1 blocker=

## 2026-06-07 21:52:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_basis_efficiency.py --device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1 --seed 2211 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=S1-ArbitraryCotangentEfficiencyPass rows=120 blocker=

## 2026-06-07 22:01:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_arbitrary_loss_horizon.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --seed 2211
```

- status: completed
- note: route=S5-ArbitraryLossHorizonPass c3=3 c4=1 blocker=

## 2026-06-07 22:03:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --seed 2211
```

- status: completed
- note: route=S6-KANRetainedSourceOpened pass_rows=2 blocker=

## Final Artifact Summary

- final route: `R9-KANArbitraryLossSourceOpened`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip`
- key commands are also in `v22_11_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-07 22:03:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=R9-KANArbitraryLossSourceOpened artifacts=69 bundle=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip

## 2026-06-07 22:03:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --seed 2211 --efficiency-device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1
```

- status: completed
- note: queue_drained=1 blocked=0

## Final Artifact Summary

- final route: `R9-KANArbitraryLossSourceOpened`
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip`
- key commands are also in `v22_11_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-07 22:03:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=R9-KANArbitraryLossSourceOpened artifacts=69 bundle=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip

## 2026-06-07 22:03:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/logs/S7_finalize_after_queue_manifest.log

## 2026-06-07 22:05:21 +0800

```bash
rm -rf /tmp/v22_11_final_packet_check && mkdir -p /tmp/v22_11_final_packet_check && unzip -q results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip -d /tmp/v22_11_final_packet_check && /home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q /tmp/v22_11_final_packet_check/dgkan /tmp/v22_11_final_packet_check/experiments && cd /tmp/v22_11_final_packet_check && /home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY2'
import importlib
mods = ['dgkan.fu.loss_interface','dgkan.fu.upstream_cotangent','dgkan.profiling.efficiency_v22_11','experiments.run_v22_11_source_atom_generation','experiments.run_v22_11_arbitrary_loss_horizon','experiments.run_v22_11_kan_mapping','experiments.run_v22_11_full','experiments.run_v22_11_finalize']
for mod in mods:
    importlib.import_module(mod)
print('v22_11_final_packet_clean_import_ok')
PY2
```

- status: completed
- note: final v22_11_code_review_packet.zip clean unzip compile/import ok; output=v22_11_final_packet_clean_import_ok

## Final Artifact Summary

- final route: `R9-KANArbitraryLossSourceOpened`
- results bundle: `results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip`
- code review packet: `results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip`
- key commands are also in `v22_11_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-07 22:05:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_finalize.py --out-dir results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=R9-KANArbitraryLossSourceOpened artifacts=69 bundle=results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip code_review_packet=results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip

## 2026-06-07 22:08:33 +0800

```bash
rm -rf /tmp/v22_11_final_packet_check_after_finalize && mkdir -p /tmp/v22_11_final_packet_check_after_finalize && unzip -q results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip -d /tmp/v22_11_final_packet_check_after_finalize && /home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q /tmp/v22_11_final_packet_check_after_finalize/dgkan /tmp/v22_11_final_packet_check_after_finalize/experiments && cd /tmp/v22_11_final_packet_check_after_finalize && /home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY2'
import importlib
mods = ['dgkan.fu.loss_interface','dgkan.fu.upstream_cotangent','dgkan.profiling.efficiency_v22_11','experiments.run_v22_11_full','experiments.run_v22_11_finalize']
for mod in mods:
    importlib.import_module(mod)
print('v22_11_final_packet_after_finalize_clean_import_ok')
PY2
```

- status: completed
- note: after-finalize v22_11_code_review_packet.zip clean unzip compile/import ok; output=v22_11_final_packet_after_finalize_clean_import_ok

## Final Artifact Summary

- final route: `R9-KANArbitraryLossSourceOpened`
- results bundle: `results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip`
- code review packet: `results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip`
- key commands are also in `v22_11_command_journal.csv`; per-task stdout/stderr logs are under `logs/`.

## 2026-06-07 22:08:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_finalize.py --out-dir results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11
```

- status: completed
- note: route=R9-KANArbitraryLossSourceOpened artifacts=69 bundle=results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip code_review_packet=results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip
