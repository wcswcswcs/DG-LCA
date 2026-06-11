# DG-KAN v22.08 RetainedSourceDynamics FunctionalUpdate BasisEfficiency 4GPU 执行日志

生成时间：2026-06-07 02:16:29 +0800

记录原则：只记录真实命令、输入、输出、状态、blocker 与修复尝试；未执行项不写成完成。

## 2026-06-07 02:16:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: started
- note: 4GPU queue launch

## 2026-06-07 02:16:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_efficiency_reconfirm.py --v2204-dir /home/chengshun.wang/DG-LCA/results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: rows=2 D-CHE=1 D-FOU=1 missing=backward_ratio_vs_mlp;functional_direction_ms;metric_solver_ms

## 2026-06-07 02:16:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_retained_source_observer.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 20 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: families=5 pass=0 official_early_positive=0 route=RetainedSourceObserverLocalNoGo_v22.08

## 2026-06-07 02:16:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_s015_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: S0.15=0 failed=c2_recompute_tests

## 2026-06-07 02:16:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_drat_drbf_multibatch.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: rows=8 robust=0 near=8 telemetry_complete=0 repair_rows=56 repair_micro_near=17

## 2026-06-07 02:16:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_metric_dynamics_solver.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: observer_pass=0 c2_recompute_rows=4 c2_pass_rows=0 route=R-ObserverLocalNoGoBoundary

## 2026-06-07 02:16:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_terminal_preservation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: decision=D0D1BlockedBeforeTerminalPreservation blocker=retained_source_observer_gate_failed

## 2026-06-07 02:16:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_kan_source_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: decision=KANMappingNotEntered blocker=retained_source_observer_gate_failed

## 2026-06-07 02:16:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: route=R-CodeMetricMechanismInvalid artifacts=65 bundle=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_results_bundle.zip

## 2026-06-07 02:16:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 02:18:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: started
- note: 4GPU queue launch

## 2026-06-07 02:18:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_efficiency_reconfirm.py --v2204-dir /home/chengshun.wang/DG-LCA/results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: rows=2 D-CHE=1 D-FOU=1 missing=backward_ratio_vs_mlp;functional_direction_ms;metric_solver_ms

## 2026-06-07 02:18:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_retained_source_observer.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --top-k 20 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: families=5 pass=0 official_early_positive=0 route=RetainedSourceObserverLocalNoGo_v22.08

## 2026-06-07 02:18:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_s015_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: S0.15=1 failed=

## 2026-06-07 02:18:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_drat_drbf_multibatch.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: rows=8 robust=0 near=8 telemetry_complete=0 repair_rows=56 repair_micro_near=23

## 2026-06-07 02:18:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_metric_dynamics_solver.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: observer_pass=0 c2_recompute_rows=4 c2_pass_rows=0 route=R-ObserverLocalNoGoBoundary

## 2026-06-07 02:18:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_terminal_preservation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: decision=D0D1BlockedBeforeTerminalPreservation blocker=retained_source_observer_gate_failed

## 2026-06-07 02:18:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_kan_source_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: decision=KANMappingNotEntered blocker=retained_source_observer_gate_failed

## 2026-06-07 02:18:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: route=R-ObserverLocalNoGoBoundary artifacts=71 bundle=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_results_bundle.zip

## 2026-06-07 02:18:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: queue_drained=1 blocked=0

## 2026-06-07 02:22:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: route=R-ObserverLocalNoGoBoundary artifacts=71 bundle=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_results_bundle.zip

## 2026-06-07 02:36:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_post_nogo_source_state_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --fresh-top-k 2 --steps 3200 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: probe_rows=144 train_only_pass=144 fresh_C3_pass=0 route=PostNoGoSourceStateCertificateBlocked

## 2026-06-07 02:42:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_post_nogo_counterfactual_washout.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:2 --fresh-top-k 2 --steps 3200 --washout-steps 12 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: probe_rows=144 train_only_pass=65 fresh_C3_pass=0 route=PostNoGoCounterfactualWashoutBlocked

## 2026-06-07 02:45:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: route=R-ObserverLocalNoGoBoundary artifacts=89 bundle=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_code_review_packet.zip

## 2026-06-07 02:54:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_post_nogo_ntk_channel_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:1 --fresh-top-k 2 --steps 3200 --jacobian-samples 16 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: probe_rows=144 train_only_pass=0 fresh_C3_pass=0 route=PostNoGoNTKChannelBlocked

## 2026-06-07 02:58:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: route=R-ObserverLocalNoGoBoundary artifacts=97 bundle=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_code_review_packet.zip

## 2026-06-07 03:07:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_post_nogo_aug_tangent_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:0 --fresh-top-k 2 --steps 3200 --aug-noise 0.03 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: probe_rows=144 train_only_pass=13 fresh_C3_pass=0 route=PostNoGoAugTangentBlocked

## 2026-06-07 03:10:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: route=R-ObserverLocalNoGoBoundary artifacts=105 bundle=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_code_review_packet.zip

## 2026-06-07 03:12:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: route=R-ObserverLocalNoGoBoundary artifacts=105 bundle=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_code_review_packet.zip

## 2026-06-07 03:14:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: route=R-ObserverLocalNoGoBoundary artifacts=105 bundle=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_code_review_packet.zip

## 2026-06-07 03:24:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_post_nogo_bootstrap_basin_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:2 --fresh-top-k 2 --steps 3200 --bootstrap-steps 16 --bootstrap-replicas 3 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: probe_rows=144 train_only_pass=144 fresh_C3_pass=0 route=PostNoGoBootstrapBasinBlocked

## 2026-06-07 03:27:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: route=R-ObserverLocalNoGoBoundary artifacts=113 bundle=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_code_review_packet.zip

## 2026-06-07 03:39:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_post_nogo_crossfit_influence_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:3 --fresh-top-k 2 --steps 3200 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: probe_rows=144 train_only_pass=144 fresh_C3_pass=0 route=PostNoGoCrossFitInfluenceBlocked

## 2026-06-07 03:40:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: route=R-ObserverLocalNoGoBoundary artifacts=121 bundle=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_code_review_packet.zip

## 2026-06-07 03:52:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_post_nogo_train_flow_commutator_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:2 --fresh-top-k 2 --steps 3200 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: probe_rows=144 train_only_pass=48 fresh_C3_pass=1 route=PostNoGoTrainFlowCommutatorFreshC3Opened

## 2026-06-07 03:58:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_post_nogo_train_flow_commutator_certificate.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:2 --fresh-top-k 2 --steps 3200 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: probe_rows=144 train_only_pass=48 fresh_C3_pass=1 route=PostNoGoTrainFlowCommutatorFreshC3Opened

## 2026-06-07 04:00:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_train_flow_commutator_fresh_verify.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/metric_solver_source_smoke_combined_v9_to_v19_readback --device cuda:2 --verify-top-k 1 --repeats 3 --steps 3200 --refresh-interval 8 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: verify_candidates=1 robust_fresh_verified=0 route=TrainFlowCommutatorRepeatVerifyBlocked

## 2026-06-07 04:00:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: route=R-PostNoGoTrainFlowCommutatorFreshSmokeOnly artifacts=134 bundle=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_code_review_packet.zip

## 2026-06-07 04:02:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08
```

- status: completed
- note: route=R-PostNoGoTrainFlowCommutatorRepeatVerifyBlocked artifacts=134 bundle=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/v22_08_code_review_packet.zip
