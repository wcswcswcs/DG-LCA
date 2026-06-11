# DG-KAN v22.09 RetainedSourceObservability FunctionalUpdate BasisEfficiency 4GPU 执行日志

生成时间：2026-06-07 08:46:56 +0800

记录原则：只记录真实命令、输入、输出、状态、blocker 与修复尝试；未执行项不写成完成。

## 2026-06-07 08:46:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: started
- note: 4GPU queue launch

## 2026-06-07 08:46:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_retained_source_observer.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 --top-k 20
```

- status: completed
- note: rows=204 families=3 pass_rows=0 official_early_positive=0 route=RetainedSourceObserverLocalNoGo_v22.09

## 2026-06-07 08:47:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=0 code_route=R0-CodePacketNotSelfContained missing_zip=0 clean_import=1

## 2026-06-07 08:47:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 08:47:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_metric_dynamics_solver.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: observer_pass=0 c2_rows=4 c2_pass_rows=0 c3_pass=0 route=RetainedSourceObserverLocalNoGo_v22.09

## 2026-06-07 08:47:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_terminal_preservation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: c3_pass=0 route=D0D1BlockedBeforeTerminalPreservation_v22.09

## 2026-06-07 08:47:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: c3_pass=0 c4_pass=0 route=KANMappingNotEntered

## 2026-06-07 08:47:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: route=R0-CodePacketNotSelfContained artifacts=655 bundle=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_code_review_packet.zip

## 2026-06-07 08:47:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 08:48:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 08:49:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: route=RetainedSourceObserverLocalNoGo_v22.09 artifacts=1382 bundle=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_code_review_packet.zip

## 2026-06-07 08:50:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 08:50:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 08:51:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: route=RetainedSourceObserverLocalNoGo_v22.09 artifacts=1451 bundle=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_code_review_packet.zip

## 2026-06-07 08:52:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 --drat-device cuda:2 --observer-source-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: started
- note: 4GPU queue launch

## 2026-06-07 08:52:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_retained_source_observer.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 --top-k 20
```

- status: completed
- note: rows=204 families=3 pass_rows=0 official_early_positive=0 route=RetainedSourceObserverLocalNoGo_v22.09

## 2026-06-07 08:52:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 08:52:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 08:52:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_metric_dynamics_solver.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: observer_pass=0 c2_rows=4 c2_pass_rows=0 c3_pass=0 route=RetainedSourceObserverLocalNoGo_v22.09

## 2026-06-07 08:52:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_terminal_preservation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: c3_pass=0 route=D0D1BlockedBeforeTerminalPreservation_v22.09

## 2026-06-07 08:52:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: c3_pass=0 c4_pass=0 route=KANMappingNotEntered

## 2026-06-07 08:52:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: route=RetainedSourceObserverLocalNoGo_v22.09 artifacts=1520 bundle=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_code_review_packet.zip

## 2026-06-07 08:52:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 --drat-device cuda:2 --observer-source-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 08:52:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: route=RetainedSourceObserverLocalNoGo_v22.09 artifacts=1520 bundle=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_code_review_packet.zip

## 2026-06-07 09:04:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 09:06:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_post_boundary_flow_replay_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:1 --max-probe-jobs 144 --fresh-top-k 2 --steps 3200 --local-replay-steps 4 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: probe_rows=144 measured=144 train_only_pass=126 fresh_pass=0 route=PostBoundaryBVFRBlocked

## 2026-06-07 09:13:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 09:15:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_post_optimizer_state_holonomy_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:1 --max-probe-jobs 144 --fresh-top-k 2 --steps 3200 --micro-horizon 6 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: probe_rows=144 measured=144 train_only_pass=9 fresh_pass=0 route=PostStateHolonomyBlocked

## 2026-06-07 09:18:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 09:19:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_post_boundary_flow_replay_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:1 --max-probe-jobs 144 --fresh-top-k 2 --steps 3200 --local-replay-steps 4 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: probe_rows=144 measured=144 train_only_pass=126 fresh_pass=0 route=PostBoundaryBVFRBlocked

## 2026-06-07 09:20:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_post_optimizer_state_holonomy_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:1 --max-probe-jobs 144 --fresh-top-k 2 --steps 3200 --micro-horizon 6 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: probe_rows=144 measured=144 train_only_pass=9 fresh_pass=0 route=PostStateHolonomyBlocked

## 2026-06-07 09:21:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: route=RetainedSourceObserverLocalNoGo_v22.09 artifacts=1771 bundle=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_code_review_packet.zip

## 2026-06-07 09:29:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 09:30:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_post_row_orthogonal_source_transport_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:1 --max-probe-jobs 144 --fresh-top-k 2 --steps 3200 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: probe_rows=144 measured=144 train_only_pass=120 fresh_pass=0 route=PostRowOrthogonalBlocked

## 2026-06-07 09:31:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: route=RetainedSourceObserverLocalNoGo_v22.09 artifacts=1866 bundle=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_code_review_packet.zip

## 2026-06-07 09:38:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 09:40:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_post_info_volume_signal_channel_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:1 --max-probe-jobs 144 --fresh-top-k 2 --steps 3200 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: probe_rows=144 measured=144 train_only_pass=126 fresh_pass=0 route=PostInfoVolumeBlocked

## 2026-06-07 09:40:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: route=RetainedSourceObserverLocalNoGo_v22.09 artifacts=1969 bundle=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_code_review_packet.zip

## 2026-06-07 09:50:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 09:52:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_post_fast_slow_memory_resonance_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:1 --max-probe-jobs 144 --fresh-top-k 2 --steps 3200 --memory-beta 0.92 --micro-steps 8 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: probe_rows=144 measured=144 train_only_pass=143 fresh_pass=0 route=PostFastSlowMemoryBlocked

## 2026-06-07 09:56:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 09:57:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_post_fast_slow_memory_damped_repair.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:1 --fresh-top-k 2 --steps 3200 --fu-lr 2.5e-05 --memory-beta 0.97 --refresh-interval 16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: selected=2 fresh_pass=0 route=PostFastSlowMemoryDampedRepairBlocked fu_lr=2.5e-05 beta=0.97 refresh=16

## 2026-06-07 09:58:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: route=RetainedSourceObserverLocalNoGo_v22.09 artifacts=2195 bundle=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_code_review_packet.zip

## 2026-06-07 10:04:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 10:05:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=1

## 2026-06-07 10:07:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 10:07:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 10:08:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 10:09:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 10:11:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: route=RetainedSourceObserverLocalNoGo_v22.09 artifacts=2545 bundle=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_code_review_packet.zip

## 2026-06-07 10:23:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 10:24:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_post_time_reversal_adjoint_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:1 --max-probe-jobs 144 --fresh-top-k 2 --steps 3200 --fu-lr 7.5e-05 --refresh-interval 8 --transport-mix 0.35 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: probe_rows=144 measured=144 train_only_pass=124 fresh_pass=0 route=PostTimeReversalAdjointBlocked

## 2026-06-07 10:29:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 10:30:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_post_time_reversal_adjoint_damped_repair.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:1 --fresh-top-k 2 --steps 3200 --fu-lr 1.5e-05 --refresh-interval 24 --transport-mix 0.15 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: selected=2 fresh_pass=0 route=PostTimeReversalAdjointDampedRepairBlocked fu_lr=1.5e-05 refresh=24 mix=0.15

## 2026-06-07 10:30:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: route=RetainedSourceObserverLocalNoGo_v22.09 artifacts=2805 bundle=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_code_review_packet.zip

## 2026-06-07 10:38:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 10:38:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:0 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 10:40:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:0 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=1

## 2026-06-07 10:42:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:0 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 10:48:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_drat_drbf_shape_localk_repair_scan.py --device cuda:0 --official-transition-batch-sizes 128,256,512,1024 --hidden-grid 64,96,128 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: shape_localK_route=D-RAT_D-RBFShapeLocalKRepairBlocked scan_rows=36 candidate_groups=0

## 2026-06-07 10:49:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: S0.16=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0

## 2026-06-07 10:50:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09
```

- status: completed
- note: route=RetainedSourceObserverLocalNoGo_v22.09 artifacts=3085 bundle=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_code_review_packet.zip

## 2026-06-07 12:44:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 13:31:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 13:45:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 13:57:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 14:01:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 14:04:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 14:07:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 14:21:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 14:32:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=1

## 2026-06-07 14:43:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 14:46:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 14:59:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 15:10:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_singlelaunch_sanity
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=28 smoke_deferred_rows=0

## 2026-06-07 15:13:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 15:25:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_lowhidden_sanity
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 15:36:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_cefix_sanity
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 15:40:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_warps_sanity
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 15:43:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_combo_sanity
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 15:46:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 16:02:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_rbf_singlelaunch_sanity
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 16:04:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_low_hidden_sanity
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 16:15:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 17:10:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2

## 2026-06-07 17:35:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10
```

- status: completed
- note: D-CHE=1 D-FOU=1 DRAT_DRBF_closed=0 repair_rows=56 smoke_deferred_rows=2
