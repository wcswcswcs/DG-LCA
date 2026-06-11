# DG-KAN v22.15 AFG + Source-Manifold Controller 执行日志

生成时间：2026-06-11 12:31:06 +0800

记录原则：只记录真实命令、文件、输入、输出、状态、blocker 与修复尝试；未执行项不写成完成。

## 2026-06-11 12:31:09 +0800 S0

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_s0_truth.py --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: 0
- status: blocked
- files: v22_15_code_truth_gate.csv; v22_15_semantic_firewall.csv; v22_15_adaptive_controller_unit_tests.csv
- note: route=R0-CodeOrSemanticGateFailed blocker=future

## 2026-06-11 12:32:20 +0800 S0

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_s0_truth.py --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: 0
- status: completed
- files: v22_15_code_truth_gate.csv; v22_15_semantic_firewall.csv; v22_15_adaptive_controller_unit_tests.csv
- note: route=S0-CodeSemanticControllerTruthGatePass blocker=

## 2026-06-11 12:33:11 +0800 C3

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_loss_geometry_operator.py --device cuda:0 --seeds 2215,2216,2217 --dim 32 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_loss_geometry_operator_matrix.csv; v22_15_adapter_renaming_tests.csv
- note: route=R4-PointwiseAdaptiveFUOpened_PairwiseNoGo ranking=0 controls_fail=1

## 2026-06-11 12:33:12 +0800 B-D-FOU

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_efficiency_controller_loop.py --device cuda:0 --carriers D-FOU --batch-sizes 128,256,512,1024 --hidden 64,128 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_adaptive_efficiency_matrix.csv; v22_15_controller_component_timing.csv
- note: route=R1-AdaptiveEfficiencyBlocked pass=0

## 2026-06-11 12:33:12 +0800 B-D-CHE

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_efficiency_controller_loop.py --device cuda:0 --carriers D-CHE --batch-sizes 128,256,512,1024 --hidden 64,128 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_adaptive_efficiency_matrix.csv; v22_15_controller_component_timing.csv
- note: route=R1-AdaptiveEfficiencyBlocked pass=0

## 2026-06-11 12:42:50 +0800 S0

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_s0_truth.py --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: 0
- status: completed
- files: v22_15_code_truth_gate.csv; v22_15_semantic_firewall.csv; v22_15_adaptive_controller_unit_tests.csv
- note: route=S0-CodeSemanticControllerTruthGatePass blocker=

## 2026-06-11 12:43:20 +0800 C5

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_kan_basis_controller.py --device cuda:0 --seeds 2215,2216,2217 --function-dim 48 --basis-dim 16 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_KAN_basis_controller_matrix.csv; v22_15_basis_projection_coverage.csv
- note: route=C5-KANBasisAdaptiveExplorationPass explore=1 official=1

## 2026-06-11 12:43:21 +0800 C6

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_source_manifold_controller.py --device cuda:0 --seeds 2215,2216,2217 --dim 48 --manifold-dims 4,8,16,32 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_source_manifold_controller_matrix.csv; v22_15_source_manifold_controls_matrix.csv; v22_15_KAN_basis_manifold_controller_matrix.csv
- note: route=C6-SourceManifoldExplorationPass mlp=1 kan=1 controls_fail=1

## 2026-06-11 12:43:22 +0800 B-D-CHE

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_efficiency_controller_loop.py --device cuda:0 --carriers D-CHE --batch-sizes 128,256,512,1024 --hidden 64,128 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_adaptive_efficiency_matrix.csv; v22_15_controller_component_timing.csv
- note: route=R1-AdaptiveEfficiencyBlocked pass=0

## 2026-06-11 12:44:06 +0800 B-D-FOU

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_efficiency_controller_loop.py --device cuda:0 --carriers D-FOU --batch-sizes 128,256,512,1024 --hidden 64,128 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_adaptive_efficiency_matrix.csv; v22_15_controller_component_timing.csv
- note: route=R1-AdaptiveEfficiencyBlocked pass=0

## 2026-06-11 12:47:17 +0800 S0

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_s0_truth.py --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: 0
- status: completed
- files: v22_15_code_truth_gate.csv; v22_15_semantic_firewall.csv; v22_15_adaptive_controller_unit_tests.csv
- note: route=S0-CodeSemanticControllerTruthGatePass blocker=

## 2026-06-11 12:47:58 +0800 C1-C2-C4

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_adaptive_mlp_lab.py --device cuda:0 --seeds 2215,2216,2217 --steps 6400 --dim 64 --exact-interval 25 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_mlp_adaptive_guidance_matrix.csv; v22_15_source_risk_prediction_matrix.csv; v22_15_source_state_release_matrix.csv
- note: route=R3-AdaptiveControllerNoGo c1=0 c2=1 c4=1

## 2026-06-11 12:51:15 +0800 S0

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_s0_truth.py --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: 0
- status: completed
- files: v22_15_code_truth_gate.csv; v22_15_semantic_firewall.csv; v22_15_adaptive_controller_unit_tests.csv
- note: route=S0-CodeSemanticControllerTruthGatePass blocker=

## 2026-06-11 12:51:33 +0800 C3

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_loss_geometry_operator.py --device cuda:0 --seeds 2215,2216,2217 --dim 32 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_loss_geometry_operator_matrix.csv; v22_15_adapter_renaming_tests.csv
- note: route=C3-LossGeometryOperatorPass ranking=1 controls_fail=1

## 2026-06-11 12:51:35 +0800 B-D-CHE

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_efficiency_controller_loop.py --device cuda:0 --carriers D-CHE --batch-sizes 128,256,512,1024 --hidden 64,128 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_adaptive_efficiency_matrix.csv; v22_15_controller_component_timing.csv
- note: route=R1-AdaptiveEfficiencyBlocked pass=0

## 2026-06-11 12:51:58 +0800 C1-C2-C4

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_adaptive_mlp_lab.py --device cuda:0 --seeds 2215,2216,2217 --steps 6400 --dim 64 --exact-interval 25 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_mlp_adaptive_guidance_matrix.csv; v22_15_source_risk_prediction_matrix.csv; v22_15_source_state_release_matrix.csv
- note: route=R3-AdaptiveControllerNoGo c1=0 c2=1 c4=1

## 2026-06-11 12:52:15 +0800 B-D-FOU

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_efficiency_controller_loop.py --device cuda:0 --carriers D-FOU --batch-sizes 128,256,512,1024 --hidden 64,128 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_adaptive_efficiency_matrix.csv; v22_15_controller_component_timing.csv
- note: route=R1-AdaptiveEfficiencyBlocked pass=0

## 2026-06-11 12:52:56 +0800 D-task-readback

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_task_readback.py --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: n/a
- status: completed
- files: v22_15_task_eval_matrix.csv
- note: route=TaskReadbackDeferredByMechanismGate mechanism_ready=0

## 2026-06-11 12:53:17 +0800 finalize

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_finalize.py --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: n/a
- status: completed
- files: v22_15_final_route.json; v22_15_artifact_index.csv; v22_15_results_bundle.zip; docs recap
- note: final_route=R1-AdaptiveEfficiencyBlocked

## 2026-06-11 12:54:08 +0800 C1-repair-note-1

```bash
CUDA_VISIBLE_DEVICES=1 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_adaptive_mlp_lab.py --device cuda:0 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0 (physical GPU1)
- status: interrupted
- files: no C1 matrix written by interrupted process
- note: PID 1234852 was terminated after ~5m34s CPU time because per-step 64x64 prox solve made the harness impractical; repaired by using identity-J closed-form prox and exact source-manifold solve every K steps.

## 2026-06-11 12:54:08 +0800 C1-repair-note-2

```bash
CUDA_VISIBLE_DEVICES=1 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_adaptive_mlp_lab.py --device cuda:0 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0 (physical GPU1)
- status: interrupted
- files: no C1 matrix written by interrupted process
- note: PID 1239848 was terminated after ~2m25s CPU time because Python per-step loop remained the bottleneck; repaired by segment dynamics every exact_interval=25 while preserving every-25-step and horizon logging.

## 2026-06-11 12:54:08 +0800 B-repair-note-merge

```bash
parallel D-CHE/D-FOU efficiency first run
```

- gpu: physical GPU0/GPU2
- status: superseded
- files: v22_15_adaptive_efficiency_matrix.csv rerun after merge fix
- note: Initial parallel efficiency runs could overwrite aggregate CSVs; fixed run_v22_15_efficiency_controller_loop.py to merge by carrier, filter blank carrier rows, and reran D-CHE then D-FOU.

## 2026-06-11 12:54:08 +0800 S0-repair-note

```bash
S0 semantic scanner repair
```

- gpu: n/a
- status: completed
- files: experiments/run_v22_15_s0_truth.py
- note: Initial S0 static firewall falsely counted Python __future__ import as forbidden future signal; scanner now ignores __future__ lines and clean S0 was rerun successfully.

## 2026-06-11 12:55:42 +0800 S0

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_s0_truth.py --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: 0
- status: completed
- files: v22_15_code_truth_gate.csv; v22_15_semantic_firewall.csv; v22_15_adaptive_controller_unit_tests.csv
- note: route=S0-CodeSemanticControllerTruthGatePass blocker=

## 2026-06-11 12:55:58 +0800 finalize

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_finalize.py --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: n/a
- status: completed
- files: v22_15_final_route.json; v22_15_artifact_index.csv; v22_15_results_bundle.zip; docs recap
- note: final_route=R1-AdaptiveEfficiencyBlocked

## 2026-06-11 15:06:54 +0800 S0

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_s0_truth.py --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: 0
- status: completed
- files: v22_15_code_truth_gate.csv; v22_15_semantic_firewall.csv; v22_15_adaptive_controller_unit_tests.csv
- note: route=S0-CodeSemanticControllerTruthGatePass blocker=

## 2026-06-11 15:07:39 +0800 B-D-CHE

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_efficiency_controller_loop.py --device cuda:0 --carriers D-CHE --batch-sizes 128,256,512,1024 --hidden 64,128 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_adaptive_efficiency_matrix.csv; v22_15_controller_component_timing.csv
- note: route=B-AdaptiveEfficiencyPass pass=1

## 2026-06-11 15:08:07 +0800 B-D-FOU

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_efficiency_controller_loop.py --device cuda:0 --carriers D-FOU --batch-sizes 128,256,512,1024 --hidden 64,128 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_adaptive_efficiency_matrix.csv; v22_15_controller_component_timing.csv
- note: route=B-AdaptiveEfficiencyPass pass=1

## 2026-06-11 15:09:08 +0800 C1-C2-C4

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_adaptive_mlp_lab.py --device cuda:0 --seeds 2215,2216,2217 --steps 6400 --dim 64 --exact-interval 25 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_mlp_adaptive_guidance_matrix.csv; v22_15_source_risk_prediction_matrix.csv; v22_15_source_state_release_matrix.csv
- note: route=C1-C2-C4-MechanismLabPass c1=1 c2=1 c4=1

## 2026-06-11 15:11:00 +0800 C1-C2-C4

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_adaptive_mlp_lab.py --device cuda:0 --seeds 2215,2216,2217 --steps 6400 --dim 64 --exact-interval 25 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_mlp_adaptive_guidance_matrix.csv; v22_15_source_risk_prediction_matrix.csv; v22_15_source_state_release_matrix.csv
- note: route=C1-C2-C4-MechanismLabPass c1=1 c2=1 c4=1

## 2026-06-11 15:11:15 +0800 D-task-readback

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_task_readback.py --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: n/a
- status: completed
- files: v22_15_task_eval_matrix.csv
- note: route=TaskReadbackPending_NoTaskMetricFabricated mechanism_ready=1

## 2026-06-11 15:17:57 +0800 D-task-readback

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_task_readback.py --device cuda:0 --datasets MNIST --seeds 0 --train-size 256 --test-size 128 --steps 2 --batch-size 64 --hidden 32 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_task_eval_matrix.csv; v22_15_convergence_speed_matrix.csv; v22_15_calibration_debt_matrix.csv
- note: route=TaskReadbackPartialOrBlocked mechanism_ready=1 rows=9

## 2026-06-11 15:18:14 +0800 D-task-readback

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_task_readback.py --device cuda:0 --datasets MNIST,FashionMNIST,KMNIST --seeds 0,1,2 --train-size 1024 --test-size 512 --steps 80 --batch-size 128 --hidden 64 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_task_eval_matrix.csv; v22_15_convergence_speed_matrix.csv; v22_15_calibration_debt_matrix.csv
- note: route=TaskReadbackPartialOrBlocked mechanism_ready=1 rows=81

## 2026-06-11 15:20:50 +0800 D-task-readback

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_task_readback.py --device cuda:0 --datasets MNIST,FashionMNIST,KMNIST --seeds 0,1,2 --train-size 1024 --test-size 512 --steps 80 --batch-size 128 --hidden 64 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_task_eval_matrix.csv; v22_15_convergence_speed_matrix.csv; v22_15_calibration_debt_matrix.csv
- note: route=TaskReadbackFunctionalValueOnly_MLPNotBeaten mechanism_ready=1 rows=81

## 2026-06-11 15:29:16 +0800 D-task-readback

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_task_readback.py --device cuda:0 --datasets MNIST,FashionMNIST,KMNIST --seeds 0,1,2 --train-size 1024 --test-size 512 --steps 320 --batch-size 128 --hidden 64 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_task_eval_matrix.csv; v22_15_convergence_speed_matrix.csv; v22_15_calibration_debt_matrix.csv
- note: route=TaskReadbackFunctionalValueOnly_MLPNotBeaten mechanism_ready=1 rows=81

## 2026-06-11 15:42:53 +0800 D-task-readback

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_task_readback.py --device cuda:0 --datasets MNIST,FashionMNIST,KMNIST --seeds 0,1,2 --train-size 1024 --test-size 512 --steps 320 --batch-size 128 --hidden 64 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_task_eval_matrix.csv; v22_15_convergence_speed_matrix.csv; v22_15_calibration_debt_matrix.csv
- note: route=TaskReadbackOfficialDGKANBeatsMLP mechanism_ready=1 rows=81

## 2026-06-11 15:44:55 +0800 S0

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_s0_truth.py --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: 0
- status: completed
- files: v22_15_code_truth_gate.csv; v22_15_semantic_firewall.csv; v22_15_adaptive_controller_unit_tests.csv
- note: route=S0-CodeSemanticControllerTruthGatePass blocker=

## 2026-06-11 15:45:11 +0800 finalize

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_finalize.py --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: n/a
- status: completed
- files: v22_15_final_route.json; v22_15_artifact_index.csv; v22_15_results_bundle.zip; docs recap
- note: final_route=R13-OfficialDGKANBeatsMLPReady

## 2026-06-11 15:51:50 +0800 D-task-readback

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_task_readback.py --device cuda:0 --datasets MNIST,FashionMNIST,KMNIST --seeds 0,1,2 --train-size 1024 --test-size 512 --steps 320 --batch-size 128 --hidden 64 --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: cuda:0
- status: completed
- files: v22_15_task_eval_matrix.csv; v22_15_convergence_speed_matrix.csv; v22_15_calibration_debt_matrix.csv
- note: route=TaskReadbackOfficialDGKANBeatsMLP mechanism_ready=1 rows=81

## 2026-06-11 15:52:59 +0800 S0

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_s0_truth.py --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: 0
- status: completed
- files: v22_15_code_truth_gate.csv; v22_15_semantic_firewall.csv; v22_15_adaptive_controller_unit_tests.csv
- note: route=S0-CodeSemanticControllerTruthGatePass blocker=

## 2026-06-11 15:53:00 +0800 finalize

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_15_finalize.py --out-dir results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15
```

- gpu: n/a
- status: completed
- files: v22_15_final_route.json; v22_15_artifact_index.csv; v22_15_results_bundle.zip; docs recap
- note: final_route=R13-OfficialDGKANBeatsMLPReady
