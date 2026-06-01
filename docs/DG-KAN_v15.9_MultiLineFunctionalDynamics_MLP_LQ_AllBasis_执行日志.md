# DG-KAN v15.9 MultiLineFunctionalDynamics MLP LQ AllBasis 执行日志

生成时间：2026-05-31（Asia/Singapore）

## 1. 文件

```text
plan = /home/chengshun.wang/DG-LCA/docs/DG-KAN_v15.9_MultiLineFunctionalDynamics_MLP_LQ_AllBasis_完整计划.md
runner = experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py
out_dir = results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159
line_e_out = results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate
```

## 2. py_compile

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py experiments/run_v149_line_d_all_basis_substrate_repair.py
```

## 3. Line E substrate-only 执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --batch-size 32 --epochs 1 --candidates D-FOU82-LowFreqIdentityResidualV5,D-FOU83-BandwiseSNRWarmupV5,D-FOU84-PhaseStableBandMixV5,D-FOU85-NoMaterializeLifetimeV5,D-FOU86-HighFrequencyQuarantineV5,D-RBF80-ActiveCenterOccupancyV5,D-RBF81-WidthConditionGuardV5,D-RBF82-CompactBumpNoDenseV5,D-RBF83-GaussianLocalK4TaskHealthV5,D-RBF84-IdentityResidualWidthWarmupV5,D-WAV69-TriangularSupportV5,D-WAV70-ScaleOccupancyV5,D-WAV71-SupportOverlapDampingV5,D-WAV72-LocalTailCoverageAuditV5
```

## 4. Official v15.9 GPU 执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py --out-dir results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159 --line-e-out results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --rational-steps 80 --lq-steps 80 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 0
```

四卡并行分片执行指令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py --out-dir results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159 --line-e-out results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --rational-steps 80 --lq-steps 80 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 1 --run-lines S
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py --out-dir results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159 --line-e-out results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --rational-steps 80 --lq-steps 80 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 0 --run-lines B
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py --out-dir results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159 --line-e-out results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate --device cuda:2 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --rational-steps 80 --lq-steps 80 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 0 --run-lines C
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py --out-dir results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159 --line-e-out results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate --device cuda:3 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --rational-steps 80 --lq-steps 80 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 0 --run-lines D
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py --out-dir results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159 --line-e-out results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --rational-steps 80 --lq-steps 80 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 0 --run-lines A --skip-h400 1 --artifact-suffix a0 --line-a-methods A0-D-CHE-AdamW,A1-D-CHE-G7R-PulseOnce-AdamWRecovery,A2-D-CHE-G7R-PulseEvery50-AdamWRecovery,A3-D-CHE-G7R-EarlyPulseOnly-AdamWRecovery,ACTRL1-D-CHE-RandomMatchedPulse-SameAdamWRecovery
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py --out-dir results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159 --line-e-out results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --rational-steps 80 --lq-steps 80 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 0 --run-lines A --skip-h400 1 --artifact-suffix a1 --line-a-methods A4-D-CHE-G7R-MidPulseOnly-AdamWRecovery,A5-D-CHE-G7R-LatePulseOnly-AdamWRecovery,A6-D-CHE-G7R-Pulse-GlobalDecoupledDecayRecovery,A7-D-CHE-G7R-Pulse-DegreeWiseDecayRecovery,ACTRL2-D-CHE-RandomMatchedPulse-SameDecayRecovery
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py --out-dir results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159 --line-e-out results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate --device cuda:2 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --rational-steps 80 --lq-steps 80 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 0 --run-lines A --skip-h400 1 --artifact-suffix a2 --line-a-methods A8-D-CHE-G7R-Pulse-HighDegreeExtraDecayRecovery,A9-D-CHE-G7R-Pulse-ReadoutBasisDecoupledDecayRecovery,A10-D-CHE-G7R-Pulse-MomentumEMARecovery,A11-D-CHE-G7R-Pulse-LRCooldownRecovery,ACTRL3-D-CHE-NoOpMatchedOverhead
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py --out-dir results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159 --line-e-out results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate --device cuda:3 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --rational-steps 80 --lq-steps 80 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 0 --run-lines A --skip-h400 1 --artifact-suffix a3 --line-a-methods A12-D-CHE-G7R-Pulse-LookaheadConsolidation,A13-D-CHE-G7R-Pulse-ScheduleFreeLongEMARecovery,ACTRL4-D-CHE-AdamWExtraStepsMatchedTime
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py --out-dir results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159 --line-e-out results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --rational-steps 80 --lq-steps 80 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 1 --run-lines MERGEA
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py --out-dir results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159 --line-e-out results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --rational-steps 80 --lq-steps 80 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 0 --run-lines A400
wait
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py --out-dir results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159 --line-e-out results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --rational-steps 80 --lq-steps 80 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 1 --run-lines E,finalize
```

## 5. Finalizer / route taxonomy / manifest 重算指令

用户再次追问后执行该命令重算 route / Line Z / manifest / docs；随后按最低合同补齐 D-CHE no-regression monitor artifact 并再次重算 manifest/contract/docs；不重跑训练指标。

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py --out-dir results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159 --line-e-out results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --rational-steps 80 --lq-steps 80 --long-horizon-steps 400 --batch-size 256 --split-count 4 --trace-interval 40 --linec-seeds 0 --real-linec 1 --reuse-if-present 1
```

## 6. 最终结果

```text
route = R-F5-AllFunctionalDynamicsNoGo
promotion_allowed = 0
line_a_gate_pass = 0
line_b_gate_pass = 0
line_c_reanchor_gate_pass = 0
line_d_rational_gate_pass = 0
line_e_allbasis_gate_pass = 0
required_artifact_missing_count = 0
training shards used cuda:0,cuda:1,cuda:2,cuda:3; cpu_offload_used=0
```

## 7. 用户质疑后复盘扩展

```bash
python - <<'PY'
# 从 official_v159 CSV artifacts 重新汇总并写入实验结果复盘的详细审计补充章节；不重跑训练。
PY
```

```text
追加内容 = artifact inventory + Line A/B full surface + h400 + Line C/D/E detail + failure taxonomy + budget/no-go/contract/deep coverage + D-CHE monitor summary
training_rerun = 0
metrics_fabricated = 0
```
