# DG-KAN v15.8 DynamicsHarness DecoupledDecayRecovery AllBasis 执行日志

生成时间：2026-05-31（Asia/Singapore）

## 1. 文件

```text
plan = /home/chengshun.wang/DG-LCA/docs/DG-KAN_v15.8_DynamicsHarness_DecoupledDecayRecovery_AllBasis_完整计划.md
runner = experiments/run_v158_dynamics_harness_decoupled_decay_recovery_allbasis.py
out_dir = results/v15_8_dynamics_harness_decoupled_decay_recovery_allbasis/official_v158
line_d_out = results/v15_8_dynamics_harness_decoupled_decay_recovery_allbasis/line_d_v158_allbasis_substrate
```

## 2. py_compile

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v158_dynamics_harness_decoupled_decay_recovery_allbasis.py experiments/run_v149_line_d_all_basis_substrate_repair.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v158_h800_extension.py experiments/run_v158_dynamics_harness_decoupled_decay_recovery_allbasis.py
```

## 3. Line D substrate-only 执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v15_8_dynamics_harness_decoupled_decay_recovery_allbasis/line_d_v158_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --batch-size 32 --epochs 1 --candidates D-FOU77-LowFreqIdentityResidualV6,D-FOU78-BandwiseConsensusMetricV3,D-FOU79-PhaseStableBandMixV3,D-FOU80-NoMaterializeLifetimeV5,D-FOU81-HighFrequencyQuarantineV3,D-RBF75-ActiveCenterOccupancyV5,D-RBF76-WidthConditionGuardV5,D-RBF77-CompactBumpNoDenseV5,D-RBF78-GaussianLocalK4TaskHealthV3,D-RBF79-CenterSplitConsensusMetricV2,D-WAV65-TriangularSupportV6,D-WAV66-ScaleOccupancyV4,D-WAV67-SupportOverlapDampingV4,D-WAV68-LocalTailCoverageAuditV3
```

## 4. Official v15.8 GPU 首次训练指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v158_dynamics_harness_decoupled_decay_recovery_allbasis.py --out-dir results/v15_8_dynamics_harness_decoupled_decay_recovery_allbasis/official_v158 --line-d-out results/v15_8_dynamics_harness_decoupled_decay_recovery_allbasis/line_d_v158_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 0
```

## 5. Finalizer / manifest 修复重算指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v158_dynamics_harness_decoupled_decay_recovery_allbasis.py --out-dir results/v15_8_dynamics_harness_decoupled_decay_recovery_allbasis/official_v158 --line-d-out results/v15_8_dynamics_harness_decoupled_decay_recovery_allbasis/line_d_v158_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 1
```

## 6. 用户追问后的 H=800 extension 指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v158_h800_extension.py --out-dir results/v15_8_dynamics_harness_decoupled_decay_recovery_allbasis/official_v158 --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 256 --long-horizon-steps 800 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1
```

## 7. Line Z closure / manifest 重算指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v158_dynamics_harness_decoupled_decay_recovery_allbasis.py --out-dir results/v15_8_dynamics_harness_decoupled_decay_recovery_allbasis/official_v158 --line-d-out results/v15_8_dynamics_harness_decoupled_decay_recovery_allbasis/line_d_v158_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 1
```

## 8. 最终结果

```text
route = R6-AllBasisCarrierBlocked
minimum_success = S1-SourceSignalObservable
promotion_allowed = 0
line_t_gate_pass = 0
line_w_gate_pass = 0
line_h_gate_pass = 0
real_lite_pass_count = 0 / 9
required_artifact_missing_count = 0
line_z_no_go_boundary_rows = 8
line_z_next_hypothesis_queue_rows = 3
line_h800_gate_pass = 0
line_h800_real_lite_pass_count = 0 / 9
line_h800_source_vs_best_control_mean = -0.004856328169504802
line_h800_bad_event_fraction = 1.0
all commands used --device cuda:0; cpu_offload_used=0
```
