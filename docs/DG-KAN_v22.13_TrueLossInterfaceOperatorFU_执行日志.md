# DG-KAN v22.13 TrueLossInterfaceOperatorFU 执行日志

生成时间：2026-06-08 19:43:16 +0800

记录原则：只记录真实命令、文件、输入、输出、状态、blocker 与修复尝试；未执行项不写成完成。

## 2026-06-08 19:43:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_s0_truth_gate.py --check all --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: blocked
- note: route=R0-CodeTruthFailed S0=0 blocker=dgkan/fu/operator_atoms_v22_13.py:if not renaming["adapter;clean_unzip_import_failed

## 2026-06-08 19:48:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_s0_truth_gate.py --check all --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: route=S0-CodeSemanticLayoutTruthGatePass S0=1 blocker=

## 2026-06-08 19:48:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_semantic_tests.py --seed 2213 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: route=OperatorSemanticTestsPass semantic_pass=1

## 2026-06-08 19:49:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_construct.py --seed 2213 --norm-scale 0.16 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: S2=S2-RoleBlindOperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricCommitPass blocker=

## 2026-06-08 19:50:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_efficiency_native.py --device cuda:0 --batch-sizes 128,256,512 --hidden 64 --repeats 3 --warmup 1 --seed 2213 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: blocked
- note: route=R3-KernelNativeEfficiencyBlocked fused_complete=0 blocker=arbitrary_cotangent_fused_backward_contract_missing

## 2026-06-08 19:50:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_task_eval.py --device cuda:3 --datasets MNIST,FashionMNIST,KMNIST --seeds 0,1,2 --train-size 1024 --test-size 512 --steps 80 --batch-size 128 --hidden 64 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: route=TaskReadbackPartialOrBlocked rows=45 blocker=MNIST:0:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;MNIST:1:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;MNIST:2:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;FashionMNIST:0:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;FashionMNIST:1:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;FashionMNIST:2:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;KMNIST:0:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;KMNIST:1:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;KMNIST:2:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!

## 2026-06-08 19:52:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_task_eval.py --device cuda:3 --datasets MNIST,FashionMNIST,KMNIST --seeds 0,1,2 --train-size 1024 --test-size 512 --steps 80 --batch-size 128 --hidden 64 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: route=TaskReadbackCompleted rows=45 blocker=

## 2026-06-08 19:55:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_horizon.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213
```

- status: blocked
- note: route=S5-RoleBlindOperatorHorizonNoGo c3=1 c4=0 blocker=source_func_or_source_loss_or_control_gate_failed

## 2026-06-08 19:58:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_construct.py --seed 2213 --norm-scale 0.16 --prefer-attempt source_loss_boundary_only --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: S2=S2-RoleBlindOperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricCommitPass blocker=

## 2026-06-08 20:04:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_horizon.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213
```

- status: blocked
- note: route=S5-RoleBlindOperatorHorizonNoGo c3=1 c4=0 blocker=source_func_or_source_loss_or_control_gate_failed

## 2026-06-08 20:12:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_horizon.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213
```

- status: blocked
- note: route=S5-RoleBlindOperatorHorizonNoGo c3=0 c4=0 blocker=source_func_or_source_loss_or_control_gate_failed

## 2026-06-08 20:13:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_construct.py --seed 2213 --norm-scale 0.16 --prefer-attempt split_control_null_only --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: S2=S2-RoleBlindOperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricCommitPass blocker=

## 2026-06-08 20:19:06 +0800

```bash
kill 745263  # aborted CPU run_v22_13_operator_horizon.py after user required GPU execution
```

- status: blocked
- note: CPU operator horizon aborted; its partial/fixed CPU artifacts are diagnostic only and must not count toward final gate

## 2026-06-08 20:47:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_horizon.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1
```

- status: blocked
- note: route=S5-RoleBlindOperatorHorizonNoGo c3=0 c4=0 device=cuda:1 blocker=source_func_or_source_loss_or_control_gate_failed

## 2026-06-08 20:50:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_kan_corrected_mapping.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:2
```

- status: completed
- note: route=R7-DFOUCorrectedSourceOpened_DCHEBlocked pass_rows=0 device=cuda:2 blocker=KANEfficiencyContractBlocked

## 2026-06-08 20:54:24 +0800

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

- status: completed
- note: GPU evidence recorded to `results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_gpu_runtime_evidence.csv`; observed horizon PID 749994 on cuda:1 with 750 MiB and KAN PID 762411 on cuda:2 with 750 MiB while each process was running.

## 2026-06-08 20:55:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_finalize.py --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: route=R8-DCHESourceExists_EfficiencyOrSourceLossBlocked exploration=0 official=0 blocker=KANEfficiencyContractBlocked

## 2026-06-08 21:04:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_repair_sweep.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --operator-ids LIO2_SplitCoherentControlNull --norm-scales 0.08,0.16,0.32,0.64,1.28,2.56 --adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter --attempts role_blind_initial_commit_lr1,slow_source_state_fixed_800x0p006 --horizons 100,400,800
```

- status: blocked
- note: repair sweep script reused full horizon evaluator which expected h3200/h4800 and failed with KeyError: 3200; fixed by adding diagnostic-only preview evaluator for selected horizons.

## 2026-06-08 20:57:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_finalize.py --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: route=R8-DCHESourceExists_EfficiencyOrSourceLossBlocked exploration=0 official=0 blocker=KANEfficiencyContractBlocked

## 2026-06-08 21:12:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_repair_sweep.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --operator-ids LIO2_SplitCoherentControlNull --norm-scales 0.08,0.16,0.32,0.64,1.28,2.56 --adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter --attempts role_blind_initial_commit_lr1,slow_source_state_fixed_800x0p006 --horizons 100,400,800
```

- status: completed
- note: route=RepairSweepPreviewPassFound preview_pass_rows=2 best=LIO2_SplitCoherentControlNull;0.16;Delta-MSEAdapter;slow_source_state_fixed_800x0p006 device=cuda:1 diagnostic_only=1

## 2026-06-08 21:19:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_repair_sweep.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --operator-ids LIO1_LowNDSGreen,LIO3_KANLowBankSpectral,LIO6_SourceLossBoundary --norm-scales 0.32,0.64,1.28,2.56,5.12 --adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter --attempts slow_source_state_fixed_800x0p006 --horizons 100,400,800
```

- status: completed
- note: route=RepairSweepPreviewPassFound preview_pass_rows=7 best=LIO3_KANLowBankSpectral;0.32;Delta-MSEAdapter;slow_source_state_fixed_800x0p006 device=cuda:1 diagnostic_only=1

## 2026-06-08 21:26:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_repair_sweep.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --operator-ids LIO2_SplitCoherentControlNull,LIO3_KANLowBankSpectral,LIO6_SourceLossBoundary --norm-scales 3.84,5.12,6.40,7.68,10.24 --adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter --attempts slow_source_state_fixed_800x0p006 --horizons 100,400,800
```

- status: completed
- note: route=RepairSweepPreviewPassFound preview_pass_rows=19 best=LIO2_SplitCoherentControlNull;10.24;Delta-RankingAdapter;slow_source_state_fixed_800x0p006 device=cuda:1 diagnostic_only=1

## 2026-06-08 21:27:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_construct.py --seed 2213 --norm-scale 10.24 --prefer-attempt split_control_null_only --out-dir results/v22_13_true_loss_interface_operator_fu/repair_probe_norm10_24
```

- status: blocked
- note: S2=S2-RoleBlindOperatorAtomNoGo S3=S3-BlockedBeforeOperatorSolve S4=S4-BlockedBeforeOperatorCommit blocker=S3_gate_failed

## 2026-06-08 21:29:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_construct.py --seed 2213 --norm-scale 5.0 --prefer-attempt split_control_null_only --out-dir results/v22_13_true_loss_interface_operator_fu/repair_probe_norm5_0
```

- status: completed
- note: S2=S2-RoleBlindOperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricCommitPass blocker=

## 2026-06-08 21:30:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_construct.py --seed 2213 --norm-scale 5.0 --prefer-attempt low_nds_control_null_only --out-dir results/v22_13_true_loss_interface_operator_fu/repair_probe_lio7_norm5_0
```

- status: completed
- note: S2=S2-RoleBlindOperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricCommitPass blocker=

## 2026-06-08 21:32:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_repair_sweep.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --operator-ids LIO7_LowNDSControlNull --norm-scales 4.50,5.00,5.12 --adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter --attempts slow_source_state_fixed_800x0p006 --horizons 100,400,800
```

- status: completed
- note: route=RepairSweepPreviewPassFound preview_pass_rows=2 best=LIO7_LowNDSControlNull;5.12;Delta-LossCEAdapter;slow_source_state_fixed_800x0p006 device=cuda:1 diagnostic_only=1

## 2026-06-08 21:34:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_construct.py --seed 2213 --norm-scale 5.0 --prefer-attempt split_control_null_only --out-dir results/v22_13_true_loss_interface_operator_fu/repair_probe_rowaxis_preserve_norm5_0
```

- status: completed
- note: S2=S2-RoleBlindOperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricCommitPass blocker=

## 2026-06-08 21:37:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_repair_sweep.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --operator-ids LIO2_SplitCoherentControlNull,LIO7_LowNDSControlNull --norm-scales 4.50,5.00,5.12 --adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter --attempts slow_source_state_fixed_800x0p006 --horizons 100,400,800
```

- status: completed
- note: route=RepairSweepPreviewPassFound preview_pass_rows=10 best=LIO7_LowNDSControlNull;5.12;Delta-RankingAdapter;slow_source_state_fixed_800x0p006 device=cuda:1 diagnostic_only=1

## 2026-06-08 21:38:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_construct.py --seed 2213 --norm-scale 5.12 --prefer-attempt split_control_null_only --out-dir results/v22_13_true_loss_interface_operator_fu/repair_probe_rowaxis_preserve_norm5_12
```

- status: completed
- note: S2=S2-RoleBlindOperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricCommitPass blocker=

## 2026-06-08 21:38:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_s0_truth_gate.py --check all --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: route=S0-CodeSemanticLayoutTruthGatePass S0=1 blocker=

## 2026-06-08 21:39:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_construct.py --seed 2213 --norm-scale 5.12 --prefer-attempt split_control_null_only --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: S2=S2-RoleBlindOperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricCommitPass blocker=

## 2026-06-08 21:48:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_horizon.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --norm-scale 5.12 --attempts slow_source_state_fixed_800x0p006
```

- status: blocked
- note: route=S5-RoleBlindOperatorHorizonNoGo c3=1 c4=0 device=cuda:1 norm_scale=5.12 attempts=slow_source_state_fixed_800x0p006 blocker=source_func_or_source_loss_or_control_gate_failed

## 2026-06-08 21:50:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_finalize.py --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: route=R8-DCHESourceExists_EfficiencyOrSourceLossBlocked exploration=0 official=0 blocker=KANEfficiencyContractBlocked

## 2026-06-08 22:03:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_repair_sweep.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --operator-ids LIO2_SplitCoherentControlNull,LIO7_LowNDSControlNull --norm-scales 5.12 --adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter --attempts slow_source_anchor_w0p02_800x0p006,slow_source_anchor_w0p10_800x0p006,low_lr_source_anchor_w0p10_lr0p02 --horizons 100,400,800,1600
```

- status: completed
- note: route=RepairSweepPreviewPassFound preview_pass_rows=18 best=LIO7_LowNDSControlNull;5.12;Delta-RankingAdapter;low_lr_source_anchor_w0p10_lr0p02 device=cuda:1 diagnostic_only=1

## 2026-06-08 22:15:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_horizon.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --norm-scale 5.12 --attempts low_lr_source_anchor_w0p10_lr0p02
```

- status: completed
- note: route=S5-RoleBlindOperatorHorizonPass c3=4 c4=3 device=cuda:1 norm_scale=5.12 attempts=low_lr_source_anchor_w0p10_lr0p02 blocker=

## 2026-06-08 22:19:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_kan_corrected_mapping.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:2
```

- status: completed
- note: route=S6-KANCorrectedCarrierNoGo pass_rows=0 device=cuda:2 blocker=KANSourceChannelMismatchConfirmed

## 2026-06-08 22:23:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_finalize.py --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: route=R9-TrueLossInterfaceOperatorExplorationSuccess_KANPending exploration=1 official=1 blocker=KANSourceChannelMismatchConfirmed;arbitrary_cotangent_fused_backward_contract_missing

## 2026-06-08 22:24:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_finalize.py --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: route=R9-TrueLossInterfaceOperatorExplorationSuccess_KANPending exploration=1 official=1 blocker=KANSourceChannelMismatchConfirmed;arbitrary_cotangent_fused_backward_contract_missing

## 2026-06-08 22:34:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_efficiency_native.py --device cuda:0 --batch-sizes 32 --hidden 32 --repeats 1 --warmup 0 --seed 2213 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13_eff_smoke
```

- status: completed
- note: route=S1-NativeOfficialRowsPresent fused_complete=28 blocker=

## 2026-06-08 22:36:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_efficiency_native.py --device cuda:0 --batch-sizes 32 --hidden 32 --repeats 1 --warmup 1 --seed 2213 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13_eff_smoke2
```

- status: completed
- note: route=S1-NativeOfficialRowsPresent fused_complete=28 blocker=

## 2026-06-08 22:37:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_efficiency_native.py --device cuda:0 --batch-sizes 128,256,512 --hidden 64 --repeats 3 --warmup 1 --seed 2213 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: route=S1-NativeOfficialRowsPresent fused_complete=84 blocker=

## 2026-06-08 22:40:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_kan_corrected_mapping.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:2
```

- status: completed
- note: route=S6-KANCorrectedCarrierNoGo pass_rows=0 device=cuda:2 blocker=KANSourceChannelMismatchConfirmed

## 2026-06-08 22:48:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_kan_corrected_mapping.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:2
```

- status: completed
- note: route=S6-KANCorrectedCarrierNoGo pass_rows=0 device=cuda:2 blocker=KANSourceChannelMismatchConfirmed

## 2026-06-08 22:59:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_kan_corrected_mapping.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:2
```

- status: completed
- note: route=S6-KANCorrectedCarrierNoGo pass_rows=0 device=cuda:2 blocker=KANSourceChannelMismatchConfirmed

## 2026-06-08 23:01:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_finalize.py --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13
```

- status: completed
- note: route=R9-TrueLossInterfaceOperatorExplorationSuccess_KANPending exploration=1 official=1 blocker=KANSourceChannelMismatchConfirmed

## 2026-06-08 23:21:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_kan_corrected_mapping.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:2
```

- status: completed
- note: route=S6-KANCorrectedCarrierNoGo pass_rows=0 device=cuda:2 blocker=KANSourceChannelMismatchConfirmed
