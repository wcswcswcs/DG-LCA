# DG-KAN v22.05 MetricFirst FunctionalUpdate BasisEfficiency 4GPU 执行日志

生成时间：2026-06-06 05:18:49 +0800

记录原则：只记录真实执行过的命令、输入文件、输出 artifact、状态和 blocker；不把未执行内容写成结果。

## 2026-06-06 05:18:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root . --self-contained-import-check 1
```

- status: completed
- note: pass=0 failed=mechanism_contracts

## 2026-06-06 05:19:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root . --self-contained-import-check 1
```

- status: completed
- note: pass=0 failed=mechanism_contracts

## 2026-06-06 05:20:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_efficiency_reconfirm.py --v2204-dir /home/chengshun.wang/DG-LCA/results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
```

- status: completed
- note: rows=2 D-CHE=1 D-FOU=1

## 2026-06-06 05:20:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_drat_drbf_repair.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --device cuda:2 --batch-sizes 512 --hidden 128 --iters 24 --warmup 6
```

- status: completed
- note: rows=14 near=9 official=0

## 2026-06-06 05:30:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_first_mlp_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
```

- status: completed
- note: metric_rows=8 best=MLP-MF-G0-L2 route=F0-MetricNoEffect

## 2026-06-06 05:36:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root . --self-contained-import-check 1
```

- status: completed
- note: pass=0 failed=mechanism_contracts

## 2026-06-06 05:48:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_source_memory_repair_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
```

- status: completed
- note: metric_rows=8 best=MLP-MFSM-G0-L2 route=F0-MetricNoEffect

## 2026-06-06 05:53:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_source_memory_repair_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
```

- status: completed
- note: metric_rows=8 best=MLP-MFSM-G0-L2 route=F0-MetricNoEffect

## 2026-06-06 05:53:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
```

- status: completed
- note: rows=16 dominant=EarlySourceFormationFailed;ControlEquivalent;MetricNoEffect best=MLP-MFSM-G5-FisherRKHS

## 2026-06-06 05:53:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_kan_source_mapping.py --source-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
```

- status: completed
- note: rows=16 decision=KANSourceChannelMismatch

## 2026-06-06 05:54:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_queue.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --samples 4 --interval-sec 5.0
```

- status: completed
- note: samples=16 idle_violation=0

## 2026-06-06 05:55:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_queue.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --samples 4 --interval-sec 5.0
```

- status: completed
- note: samples=16 idle_violation=0

## 2026-06-06 05:55:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
```

- status: blocked
- note: clean_unzip_returncode=1

## 2026-06-06 05:56:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
```

- status: blocked
- note: clean_unzip_returncode=1

## 2026-06-06 05:56:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_finalize.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --run-clean-self-test 1
```

- status: completed
- note: route=R0-CodeMetricInvalid packet=v22_05_code_review_packet.zip bundle=v22_05_results_bundle.zip

## 2026-06-06 05:57:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 05:57:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --run-clean-self-test 1
```

- status: completed
- note: route=R3-DRATDRBFRepairBlocked packet=v22_05_code_review_packet.zip bundle=v22_05_results_bundle.zip

## Reproduction Details

- workspace: `/home/chengshun.wang/DG-LCA`
- python: `/home/chengshun.wang/miniconda3/envs/kan/bin/python`
- official out-dir: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05`
- v22.04 baseline/readback dir: `results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04`

### Metric-Only Full Matrix

The metric-only full matrix used:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py \
  --scope mlp \
  --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_first_mlp_full \
  --run-label v22_05_metric_first_mlp_full \
  --spec-ids MLP-MF-G0-L2,MLP-MF-G1-DiagFisher,MLP-MF-G2-PopRiskDiag,MLP-MF-G3-SobolevH1,MLP-MF-G4-RKHSKNN,MLP-MF-G5-FisherRKHS,MLP-MF-G6-LowNDS,MLP-MF-G8-Ensemble,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --steps 6400 \
  --shard-count 4 \
  --shard-index {0,1,2,3} \
  --device cuda:{0,1,2,3}
```

- shard output pattern: `metric_first_mlp_full/v21_01_source_retention_matrix_v22_05_metric_first_mlp_full_fc*.csv`
- shard rows: 27 rows per shard, four shards
- merge command:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py \
  --scope mlp \
  --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_first_mlp_full \
  --run-label v22_05_metric_first_mlp_full \
  --merge-only
```

- summary command:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py \
  --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 \
  --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_first_mlp_full \
  --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04
```

- result: `F0-MetricNoEffect`; best metric-only row was `MLP-MF-G0-L2`, h4800 negative, row_h4800_positive_count=0.

### Source-Memory Repair Full Matrix

After metric-only early-source failure, source-memory metric repair was implemented as M228-M235 and run with:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py \
  --scope mlp \
  --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_source_memory_repair_full \
  --run-label v22_05_metric_source_memory_repair_full \
  --spec-ids MLP-MFSM-G0-L2,MLP-MFSM-G1-DiagFisher,MLP-MFSM-G2-PopRiskDiag,MLP-MFSM-G3-SobolevH1,MLP-MFSM-G4-RKHSKNN,MLP-MFSM-G5-FisherRKHS,MLP-MFSM-G6-LowNDS,MLP-MFSM-G8-Ensemble,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --steps 6400 \
  --shard-count 4 \
  --shard-index {0,1,2,3} \
  --device cuda:{0,1,2,3}
```

- shard output pattern: `metric_source_memory_repair_full/v21_01_source_retention_matrix_v22_05_metric_source_memory_repair_full_fc*.csv`
- shard rows: 27 rows per shard, four shards
- merge command:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py \
  --scope mlp \
  --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_source_memory_repair_full \
  --run-label v22_05_metric_source_memory_repair_full \
  --merge-only
```

- official repair summary command:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py \
  --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 \
  --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_source_memory_repair_full \
  --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04
```

- result: `F0-MetricNoEffect`; best official repair row was `MLP-MFSM-G0-L2` in route summary, while autopsy best h4800 was `MLP-MFSM-G5-FisherRKHS`; all h100/h800/h3200/h4800 means were negative.

### Blocker Fixes During Execution

- mechanism contract initially failed due undeclared semantic alias; fixed schema/semantic groups and differentiated metric mechanisms, then clean S0.12 passed.
- metric-only MLP-MF-G0..G8 failed to form early source; implemented M228-M235 source-memory metric repair and reran full 4-shard matrix.
- queue runner initially only wrote `v22_05_gpu_utilization_timeline.csv`; added plan-compatible `gpu_utilization_timeline.csv` with identical rows.
- finalizer clean self-test initially failed because `--source-root` was relative after cwd changed to clean source; fixed finalizer to resolve `out-dir`/`source-root` to absolute paths, then clean unzip self-test passed.

## 2026-06-06 05:59:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA/results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
```

- status: completed
- note: clean_unzip_returncode=0

## 2026-06-06 06:00:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --run-clean-self-test 1
```

- status: completed
- note: route=R3-DRATDRBFRepairBlocked packet=v22_05_code_review_packet.zip bundle=v22_05_results_bundle.zip

## 2026-06-06 06:12:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root . --self-contained-import-check 1
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 06:14:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_smoke_summary
```

- status: completed
- note: metric_rows=4 best=MLP-MFT-T0-LossCotangent route=F0-MetricNoEffect

## 2026-06-06 06:18:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_full_summary
```

- status: completed
- note: metric_rows=4 best=MLP-MFT-T0-LossCotangent route=F0-MetricNoEffect

## 2026-06-06 06:19:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_full_summary
```

- status: completed
- note: rows=4 dominant=EarlySourceFormationFailed;ControlEquivalent;MetricNoEffect best=MLP-MFT-T5-SourceProjectedB3Null

## 2026-06-06 06:27:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_full_summary
```

- status: completed
- note: rows=4 dominant=EarlySourceFormationFailed;ControlEquivalent;MetricNoEffect best=MLP-MFTW-T5-SourceProjectedB3Null-stop800

## 2026-06-06 06:27:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_full_summary
```

- status: completed
- note: metric_rows=4 best=MLP-MFTW-T3-LowDegreeReadout-stop800 route=F0-MetricNoEffect

## 2026-06-06 06:36:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_full_summary
```

- status: completed
- note: rows=4 dominant=EarlySourceFormationFailed;ControlEquivalent;MetricNoEffect best=MLP-MFTC-T5-stop800-postM218

## 2026-06-06 06:36:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_full_summary
```

- status: completed
- note: metric_rows=4 best=MLP-MFTC-T3-stop800-postM181 route=F0-MetricNoEffect

## 2026-06-06 06:37:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root . --self-contained-import-check 1
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 06:38:35 +0800 继续推进 / metric-target 修复审计记录

本轮目标：检查 v22.05 是否达成；结论为未达成，因此按计划中的 metric-first 修复方向继续推进。所有实验数据均来自 fresh runner artifact；未把 smoke 或 negative result 写成 promotion。

### 代码修改

- `dgkan/fu/mechanisms.py`
  - 新增 `M236-MetricTargetLossCotangentFU`、`M237-MetricTargetLowDegreeReadoutFU`、`M238-MetricTargetB1TransferFU`、`M239-MetricTargetSourceProjectedB3NullFU`。
  - 新增 `_metric_target_readout_update(...)`，复用已有 `_exact_readout_function_space_actuation_update(...)`，只把 train-only loss/source-channel target 解成 readout/function-space update，并记录 `metric_target_*` diagnostics；不生成实验数值。
- `experiments/run_v17_common.py`
  - 将 M236-M239 接入 readout-target alt-period 分支：大多数 step 仍执行 SGD，只在 `alt_period` 到点提交 FU。
  - 新增 `source_stop_steps` 调度，用于 early-window 修复；新增 `post_stop_mechanism/post_stop_fu_lr` 调度，用于 metric-target 后接 terminal anchor 的组合修复。
- `experiments/run_v21_01_source_retention.py`
  - 注册 `MLP-MFT-*`、`MLP-MFTW-*`、`MLP-MFTC-*` continuation spec，并把 `source_stop_steps/post_stop_mechanism/post_stop_fu_lr` 从 spec 传入 runner。
- `experiments/run_v22_05_metric_first_fu.py`、`experiments/run_v22_05_metric_autopsy.py`
  - 扩展 metric ID 白名单，使 continuation 结果能进入 summary/autopsy。
- `experiments/run_v22_05_s012_truth_gate.py`
  - S0 semantic audit 的 tiny model 改为带 `w2` 和 `frozen_readout_features` 的 readout model，避免 exact-readout 机制退化成普通梯度后产生假性 alias。

### 编译 / Truth Gate

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_05_metric_first_fu.py experiments/run_v22_05_metric_autopsy.py experiments/run_v22_05_s012_truth_gate.py
```

- status: completed
- note: py_compile 0

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root . --self-contained-import-check 1 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
```

- status: completed
- note: `v22_05_code_route_decision.json` pass=1；mechanism_contracts=22/22；alias_undeclared=0

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root . --self-contained-import-check 1 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
```

- status: completed
- note: pass=1；required_source_files=31/31；metric_tests=19/19；kernel_gradcheck=4；profiler_phase_tests=3/3

### Metric Target Repair

Smoke:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_smoke --run-label v22_05_metric_target_repair_smoke --spec-ids MLP-MFT-T0-LossCotangent,MLP-MFT-T3-LowDegreeReadout,MLP-MFT-T4-B1Transfer,MLP-MFT-T5-SourceProjectedB3Null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST --seeds 0 --steps 1600 --shard-count 1 --shard-index 0 --device cuda:0
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_smoke --run-label v22_05_metric_target_repair_smoke --merge-only
```

- status: completed
- note: rows=8 grouped=8；T3/T4/T5 在 MNIST seed0 h800/h1600 出现 early positive，但 smoke 不作为 promotion。

Full 4GPU:

```bash
# shard_index=0,1,2,3 分别绑定 cuda:0,cuda:1,cuda:2,cuda:3
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_full --run-label v22_05_metric_target_repair_full --spec-ids MLP-MFT-T0-LossCotangent,MLP-MFT-T3-LowDegreeReadout,MLP-MFT-T4-B1Transfer,MLP-MFT-T5-SourceProjectedB3Null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --steps 6400 --shard-count 4 --shard-index {0..3} --device cuda:{0..3}
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_full --run-label v22_05_metric_target_repair_full --merge-only
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_full_summary --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --label "metric-target repair full"
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_full_summary --attempt metric_target_repair:results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_full
```

- status: completed
- note: shard command journal: `results/.../metric_target_repair_full/v21_01_command_journal.csv`；summary route `F0-MetricNoEffect`。
- correction: first attempted `run_v22_05_metric_autopsy.py --source-root ...` failed because script has no `--source-root`; rerun with supported `--attempt label:path`.

### Metric Target Early Window

Smoke:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_smoke --run-label v22_05_metric_target_window_smoke --spec-ids MLP-MFTW-T3-LowDegreeReadout-stop800,MLP-MFTW-T4-B1Transfer-stop800,MLP-MFTW-T5-SourceProjectedB3Null-stop800,MLP-MFTW-T5-SourceProjectedB3Null-stop1600,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST --seeds 0 --steps 1600 --shard-count 1 --shard-index 0 --device cuda:0
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_smoke --run-label v22_05_metric_target_window_smoke --merge-only
```

- status: completed
- note: window smoke 保住了部分 h1600 positive；继续 full。

Full 4GPU:

```bash
# shard_index=0,1,2,3 分别绑定 cuda:0,cuda:1,cuda:2,cuda:3
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_full --run-label v22_05_metric_target_window_full --spec-ids MLP-MFTW-T3-LowDegreeReadout-stop800,MLP-MFTW-T4-B1Transfer-stop800,MLP-MFTW-T5-SourceProjectedB3Null-stop800,MLP-MFTW-T5-SourceProjectedB3Null-stop1600,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --steps 6400 --shard-count 4 --shard-index {0..3} --device cuda:{0..3}
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_full --run-label v22_05_metric_target_window_full --merge-only
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_full_summary --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --label "metric-target window full"
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_full_summary --attempt metric_target_window:results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_full
```

- status: completed
- note: best h4800 still negative；route `F0-MetricNoEffect`。

### Metric Target + Post-Stop Terminal Anchor

Smoke:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_smoke --run-label v22_05_metric_target_composite_smoke --spec-ids MLP-MFTC-T3-stop800-postM181,MLP-MFTC-T5-stop800-postM181,MLP-MFTC-T5-stop800-postM218,MLP-MFTC-T5-stop1600-postM181,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST --seeds 0 --steps 1600 --shard-count 1 --shard-index 0 --device cuda:0
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_smoke --run-label v22_05_metric_target_composite_smoke --merge-only
```

- status: completed
- note: post-stop 只影响 h3200/h4800，因此仍跑 full。

Full 4GPU:

```bash
# shard_index=0,1,2,3 分别绑定 cuda:0,cuda:1,cuda:2,cuda:3
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_full --run-label v22_05_metric_target_composite_full --spec-ids MLP-MFTC-T3-stop800-postM181,MLP-MFTC-T5-stop800-postM181,MLP-MFTC-T5-stop800-postM218,MLP-MFTC-T5-stop1600-postM181,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --steps 6400 --shard-count 4 --shard-index {0..3} --device cuda:{0..3}
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_full --run-label v22_05_metric_target_composite_full --merge-only
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_full_summary --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --label "metric-target composite full"
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_full_summary --attempt metric_target_composite:results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_full
```

- status: completed
- note: best h4800 still negative；route `F0-MetricNoEffect`。

### 关键 artifact

- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_full_summary/v22_05_metric_first_mlp_summary.csv`
- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_full_summary/v22_05_metric_first_mlp_summary.csv`
- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_full_summary/v22_05_metric_first_mlp_summary.csv`
- 对应 autopsy verdict: `*/v22_05_metric_failure_verdict.json`
- 对应逐 shard journal: `metric_target_*_full/v21_01_command_journal.csv`

## 2026-06-06 07:03:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root . --self-contained-import-check 1
```

- status: completed
- note: pass=0 failed=mechanism_contracts

## 2026-06-06 07:22:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_analysis
```

- status: completed
- note: rows=3 dominant=EarlySourceFormationFailed;ControlEquivalent;MetricNoEffect best=MLP-HC2-L2Projection

## 2026-06-06 07:22:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_analysis
```

- status: completed
- note: metric_rows=3 best=MLP-HC2-HalfProjection route=F0-MetricNoEffect

## 2026-06-06 07:29:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_analysis
```

- status: completed
- note: rows=3 dominant=EarlySourceFormationFailed;ControlEquivalent;MetricNoEffect best=MLP-HC2-L2Projection

## 2026-06-06 07:29:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_analysis
```

- status: completed
- note: metric_rows=3 best=MLP-HC2-HalfProjection route=F0-MetricNoEffect

## 2026-06-06 07:56:20 +0800 H-C2 stable-seed continuation

### Code / Gate 修复

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_05_s012_truth_gate.py experiments/run_v22_05_metric_first_fu.py experiments/run_v22_05_metric_autopsy.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root . --self-contained-import-check 1 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
```

- status: completed
- note: 修复 `run_v21_01_source_retention.py` 的 `train_seed=210100+job_index` 队列依赖；改为基于 `scope/carrier/variant/dataset/seed/v21_id/mechanism` 的稳定 sha256 seed，并写入 matrix/trace。
- note: 新增 H-C2 机制 M240-M244；M240-M244 的 stateless `make_update` audit fallback 与 M220-G0-L2 同向，已在 mechanism contract 中声明 alias group，避免假装 non-collapse。当前 mechanism_contracts route pass=1。
- note: 修复 `run_v22_05_s012_truth_gate.py`：CSV 继续保留历史失败行，但 `v22_05_code_route_decision.json` 只反映当前 invocation，避免已修复后仍被旧行判 fail。

### Seed-invariance smoke

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/seed_invariance_d1b_only_smoke --run-label v22_05_seed_invariance_d1b_only_smoke --spec-ids MLP-D1b-roworth-hidden-readout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST --seeds 0 --steps 100 --shard-count 1 --shard-index 0 --device cuda:0
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/seed_invariance_d1b_hc2_context_smoke --run-label v22_05_seed_invariance_d1b_hc2_context_smoke --spec-ids MLP-HC2-NoProjection,MLP-HC2-L2Projection,MLP-HC2-HalfProjection,MLP-D1b-roworth-hidden-readout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST --seeds 0 --steps 100 --shard-count 1 --shard-index 0 --device cuda:0
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/seed_invariance_d1b_only_smoke --run-label v22_05_seed_invariance_d1b_only_smoke --merge-only
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/seed_invariance_d1b_hc2_context_smoke --run-label v22_05_seed_invariance_d1b_hc2_context_smoke --merge-only
```

- status: completed
- evidence: D1b-only 与 H-C2 context 的 `MLP-D1b-roworth-hidden-readout` 均为 `train_seed=1186690`，h100 均为 `-0.5388312339782715`。

### H-C2 h3200 projection stable-seed full

```bash
# shard_index=0,1,2,3 分别绑定 cuda:0,cuda:1,cuda:2,cuda:3
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_full --run-label v22_05_hc2_h3200_projection_stableseed_full --spec-ids MLP-HC2-NoProjection,MLP-HC2-L2Projection,MLP-HC2-HalfProjection,MLP-D1b-roworth-hidden-readout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --steps 6400 --shard-count 4 --shard-index {0..3} --device cuda:{0..3}
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_full --run-label v22_05_hc2_h3200_projection_stableseed_full --merge-only
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_analysis
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_analysis --attempt hc2_h3200_projection_stableseed:results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_full
```

- status: completed
- artifact: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_analysis/`
- route: `F0-MetricNoEffect`; productive rows=0。

### H-C2 h4000 recompute projection stable-seed full

```bash
# shard_index=0,1,2,3 分别绑定 cuda:0,cuda:1,cuda:2,cuda:3
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_full --run-label v22_05_hc2_h4000_recompute_projection_stableseed_full --spec-ids MLP-HC2-NoProjection,MLP-HC2-H4000L2Projection,MLP-HC2-H4000HalfProjection,MLP-D1b-roworth-hidden-readout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --steps 6400 --shard-count 4 --shard-index {0..3} --device cuda:{0..3}
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_full --run-label v22_05_hc2_h4000_recompute_projection_stableseed_full --merge-only
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_analysis
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_analysis --attempt hc2_h4000_recompute_projection_stableseed:results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_full
```

- status: completed
- artifact: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_analysis/`
- route: `F0-MetricNoEffect`; productive rows=0。

## 2026-06-06 07:38:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_analysis
```

- status: completed
- note: rows=3 dominant=MetricNoEffect best=MLP-HC2-NoProjection

## 2026-06-06 07:38:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_analysis
```

- status: completed
- note: metric_rows=3 best=MLP-HC2-HalfProjection route=F0-MetricNoEffect

## 2026-06-06 07:45:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root . --self-contained-import-check 1
```

- status: completed
- note: pass=0 failed=mechanism_contracts;mechanism_contracts

## 2026-06-06 07:46:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root . --self-contained-import-check 1
```

- status: completed
- note: pass=0 failed=mechanism_contracts;mechanism_contracts;mechanism_contracts

## 2026-06-06 07:48:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root . --self-contained-import-check 1
```

- status: completed
- note: pass=0 failed=mechanism_contracts;mechanism_contracts;mechanism_contracts

## 2026-06-06 07:49:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root . --self-contained-import-check 1
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 07:54:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_analysis
```

- status: completed
- note: rows=3 dominant=MetricNoEffect best=MLP-HC2-NoProjection

## 2026-06-06 07:54:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_analysis
```

- status: completed
- note: metric_rows=3 best=MLP-HC2-H4000HalfProjection route=F0-MetricNoEffect

## 2026-06-06 07:55:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_analysis
```

- status: completed
- note: rows=3 dominant=MetricNoEffect best=MLP-HC2-NoProjection

## 2026-06-06 07:55:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_analysis
```

- status: completed
- note: rows=3 dominant=MetricNoEffect best=MLP-HC2-NoProjection

## 2026-06-06 08:08:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root . --self-contained-import-check 1
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 08:09:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_smoke_summary
```

- status: completed
- note: metric_rows=0 best= route=F0-MetricNoEffect

## 2026-06-06 08:09:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_smoke_summary
```

- status: completed
- note: metric_rows=5 best=MLP-HC3-T1-SplitConsensus route=F0-MetricNoEffect

## 2026-06-06 08:09:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_smoke_summary
```

- status: completed
- note: rows=5 dominant=EarlySourceFormationFailed;ControlEquivalent;MetricNoEffect best=MLP-HC3-T1-SplitConsensus

## 2026-06-06 08:15:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full_summary
```

- status: completed
- note: metric_rows=5 best=MLP-HC3-T1-SplitConsensus route=F0-MetricNoEffect

## 2026-06-06 08:15:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full_summary
```

- status: completed
- note: rows=5 dominant=EarlySourceFormationFailed;ControlEquivalent;MetricNoEffect best=MLP-HC3-T6-DiffeomorphicNoFold

## 2026-06-06 08:16:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_analysis
```

- status: completed
- note: metric_rows=3 best=MLP-HC2-NoProjection route=F0-MetricNoEffect

## 2026-06-06 08:16:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_analysis
```

- status: completed
- note: metric_rows=3 best=MLP-HC2-NoProjection route=F0-MetricNoEffect

## 2026-06-06 08:16:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full_summary
```

- status: completed
- note: metric_rows=5 best=MLP-HC3-T6-DiffeomorphicNoFold route=F0-MetricNoEffect

## 2026-06-06 08:17:41 +0800 H-C3 supplemental metric-target continuation

### Code / Gate 修复

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_05_s012_truth_gate.py experiments/run_v22_05_metric_first_fu.py experiments/run_v22_05_metric_autopsy.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root . --self-contained-import-check 1 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
```

- status: completed
- note: 新增 H-C3 supplemental metric-target 机制 M245-M249：T1 SplitConsensus、T2 PopRiskSNR、T6 DiffeomorphicNoFold、T7 RandomMatched、T8 SignFlipped。
- note: 修复 `run_v22_05_metric_first_fu.py` 的 best-row 选择逻辑：当 R4800/3200 不可评估时按 h4800/h3200 选择 best，避免把第一行误记为 best。当前 mechanism_contracts route pass=1，value=`32/32;alias_undeclared=0`。

### H-C3 supplemental smoke

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_smoke --run-label v22_05_hc3_metric_target_supplemental_smoke --spec-ids MLP-HC3-T1-SplitConsensus,MLP-HC3-T2-PopRiskSNR,MLP-HC3-T6-DiffeomorphicNoFold,MLP-HC3-T7-RandomMatched,MLP-HC3-T8-SignFlipped,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST --seeds 0 --steps 400 --shard-count 1 --shard-index 0 --device cuda:0
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_smoke --run-label v22_05_hc3_metric_target_supplemental_smoke --merge-only
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_smoke_summary
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_smoke_summary --attempt hc3_supplemental_smoke:results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_smoke
```

- status: completed
- note: smoke rows=5；新 target path 产生 Actuation/metric diagnostics，未出现 runtime blocker。400-step smoke 不用于 terminal promotion。

### H-C3 supplemental 4GPU full

```bash
# shard_index=0,1,2,3 分别绑定 cuda:0,cuda:1,cuda:2,cuda:3
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full --run-label v22_05_hc3_metric_target_supplemental_full --spec-ids MLP-HC3-T1-SplitConsensus,MLP-HC3-T2-PopRiskSNR,MLP-HC3-T6-DiffeomorphicNoFold,MLP-HC3-T7-RandomMatched,MLP-HC3-T8-SignFlipped,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --steps 6400 --shard-count 4 --shard-index {0..3} --device cuda:{0..3}
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full --run-label v22_05_hc3_metric_target_supplemental_full --merge-only
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full_summary
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full_summary --attempt hc3_supplemental_full:results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full
```

- status: completed
- artifact: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full_summary/`
- route: `F0-MetricNoEffect`; metric_rows=5; productive rows=0; best by h4800=`MLP-HC3-T6-DiffeomorphicNoFold`, h4800=-0.3521238896581862。

### Summary rerun after best-selection fix

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_analysis
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_analysis
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full_summary
```

- status: completed
- note: H-C2 h3200/h4000 route best 均修正为 `MLP-HC2-NoProjection`；H-C3 supplemental route best 修正为 `MLP-HC3-T6-DiffeomorphicNoFold`。

## 2026-06-06 08:34:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_drat_drbf_repair.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --device cuda:0 --batch-sizes 512 --hidden 128 --iters 24 --warmup 6
```

- status: completed
- note: rows=16 near=10 official=2

## 2026-06-06 08:36:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root . --self-contained-import-check 1
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 08:36:59 +0800 D-RBF official fused transition repair

### Wiring / compile

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v21_common.py experiments/run_v21_efficiency_officialization.py experiments/run_v17_common.py dgkan/kernels/fused_rbf.py dgkan/models/fc_purekan_primitives.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from experiments.run_v21_common import to_v19_repair_variant
from experiments.run_v17_common import v19_basis_repair_config
variant = to_v19_repair_variant('D-RBF','RBF21-compact-local-k4-smoke')
print(variant)
print(v19_basis_repair_config('D-RBF', variant))
PY
```

- status: completed
- note: mapping self-check returned `RBF22.03-R1-compact-local-k4-no-dense` and `(4, 0, 'rbf_k4_triton_l3_matmul')`.

### Pre-wiring smoke showing old path

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/drbf_official_transition_smoke --families D-RBF --batches 32 --train-size 256 --val-size 64 --input-size 8 --hidden 24 --param-budget 12000 --profiler-repeats 2 --profiler-warmup 1 --shard-count 1 --shard-index 0 --device cuda:0
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/drbf_official_transition_smoke --merge-only
```

- status: completed
- note: pre-wiring row used `compact_rbf_stream_recompute`, `official_fused_kernel_complete=0`; this exposed the missing D-RBF variant mapping and is kept as a negative wiring artifact.

### Official path smoke and block tuning

```bash
rm -rf results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/drbf_official_transition_smoke2
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/drbf_official_transition_smoke2 --families D-RBF --batches 32 --train-size 256 --val-size 64 --input-size 8 --hidden 24 --param-budget 12000 --profiler-repeats 2 --profiler-warmup 1 --shard-count 1 --shard-index 0 --device cuda:0
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/drbf_official_transition_smoke2 --merge-only
```

- status: completed
- note: smoke2 reached `rbf_k4_triton_l3_matmul`, `manual_correctness_pass=1`, `official_fused_kernel_complete=1`, but small-batch forward ratio was 3.8689944788355364.

```bash
rm -rf results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/drbf_official_transition_full
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/drbf_official_transition_full --families D-RBF --batches 512,2048 --train-size 2048 --val-size 128 --input-size 8 --hidden 128 --param-budget 12000 --profiler-repeats 3 --profiler-warmup 1 --shard-count 1 --shard-index 0 --device cuda:0
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/drbf_official_transition_full --merge-only
```

- status: completed
- note: pre-tune official path passed correctness but failed forward ratio: batch512 forward=4.366619876405829, batch2048 forward=2.8508475673038873.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/kernels/fused_rbf.py dgkan/models/fc_purekan_primitives.py experiments/run_v21_common.py experiments/run_v21_efficiency_officialization.py
rm -rf results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/drbf_official_transition_blocktune
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/drbf_official_transition_blocktune --families D-RBF --batches 512,2048 --train-size 2048 --val-size 128 --input-size 8 --hidden 128 --param-budget 12000 --profiler-repeats 3 --profiler-warmup 1 --shard-count 1 --shard-index 0 --device cuda:0
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/drbf_official_transition_blocktune --merge-only
```

- status: completed
- note: adaptive `fused_rbf.forward_matmul` block sizing improved batch2048 forward ratio from 2.8508475673038873 to 1.6483218841013845 in the standalone transition artifact.

### v22.05 route rerun

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_05_drat_drbf_repair.py dgkan/kernels/fused_rbf.py experiments/run_v21_common.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_drat_drbf_repair.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --device cuda:0 --batch-sizes 512 --hidden 128 --iters 24 --warmup 6 --official-transition 1 --official-transition-batch-sizes 512,2048 --train-size 2048 --val-size 128 --input-size 8 --profiler-repeats 3 --profiler-warmup 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root . --self-contained-import-check 1 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from pathlib import Path
from experiments.run_v22_05_finalize import _route
from experiments.run_v22_05_common import read_rows, read_json, write_json, write_rows
out=Path('results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05')
route=_route(read_rows(out/'v22_05_code_truth_gate.csv'), read_json(out/'v22_05_efficiency_route.json'), read_json(out/'v22_05_drat_drbf_repair_route.json'), read_json(out/'v22_05_metric_first_route.json'), read_json(out/'v22_05_kan_source_mapping_route.json'), read_rows(out/'v22_05_idle_violation.csv'))
write_json(out/'v22_05_final_route.json', route)
write_rows(out/'v22_05_final_route.csv', [route])
print(route)
PY
```

- status: completed
- artifact: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/v22_05_drbf_official_transition.csv`
- artifact: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/v22_05_drat_drbf_repair_route.json`
- note: D-RBF route is now `OfficialRepairOpened` with 2 official E1 rows; D-RAT remains `MicroNearE1OfficialFusedBlocked` with `official_fused_missing`.
- note: final route remains `R3-DRATDRBFRepairBlocked`, `promotion_allowed=0`, because D-RAT official fused is still missing and functional route remains `F0-MetricNoEffect`.

## 2026-06-06 08:50:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_drat_drbf_repair.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --device cuda:0 --batch-sizes 512 --hidden 128 --iters 24 --warmup 6
```

- status: completed
- note: rows=18 near=12 official=1

## 2026-06-06 08:52:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_drat_drbf_repair.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --device cuda:0 --batch-sizes 512 --hidden 128 --iters 24 --warmup 6
```

- status: completed
- note: rows=18 near=13 official=2

## 2026-06-06 08:53:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_drat_drbf_repair.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --device cuda:0 --batch-sizes 512 --hidden 128 --iters 24 --warmup 6
```

- status: completed
- note: rows=18 near=10 official=2

## 2026-06-06 08:54:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root . --self-contained-import-check 1
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 09:03:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root . --self-contained-import-check 1
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 09:04:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_smoke_summary
```

- status: completed
- note: metric_rows=2 best=MLP-HC4-T9-DebtCalibratedSource route=F0-MetricNoEffect

## 2026-06-06 09:04:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_smoke_summary
```

- status: completed
- note: rows=0 dominant=MetricNoEffect best=

## 2026-06-06 09:09:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_smoke_summary
```

- status: completed
- note: metric_rows=2 best=MLP-HC4-T9-DebtCalibratedSource route=F0-MetricNoEffect

## 2026-06-06 09:09:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_smoke_summary
```

- status: completed
- note: rows=0 dominant=MetricNoEffect best=

## 2026-06-06 09:12:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_alt10_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_alt10_smoke_summary
```

- status: completed
- note: metric_rows=2 best=MLP-HC4-T9-earlyburst-alt10 route=F0-MetricNoEffect

## 2026-06-06 09:12:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_alt10_smoke_summary
```

- status: completed
- note: rows=0 dominant=MetricNoEffect best=

## 2026-06-06 09:16:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_alt10_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_alt10_full_summary
```

- status: completed
- note: metric_rows=2 best=MLP-HC4-T9-earlyburst-alt10-stop1600-postM181 route=F0-MetricNoEffect

## 2026-06-06 09:16:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_alt10_full_summary
```

- status: completed
- note: rows=0 dominant=MetricNoEffect best=

## 2026-06-06 09:19:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_earlystop_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_earlystop_smoke_summary
```

- status: completed
- note: metric_rows=2 best=MLP-HC4-T9-earlyburst-alt10-stop400-postM181 route=F0-MetricNoEffect

## 2026-06-06 09:19:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_earlystop_smoke_summary
```

- status: completed
- note: rows=0 dominant=MetricNoEffect best=

## 2026-06-06 09:23:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_earlystop_full --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_earlystop_full_summary
```

- status: completed
- note: metric_rows=2 best=MLP-HC4-T9-earlyburst-alt10-stop400-postM181 route=F0-MetricNoEffect

## 2026-06-06 09:23:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_earlystop_full_summary
```

- status: completed
- note: rows=0 dominant=MetricNoEffect best=

## 2026-06-06 09:26:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_postanchor_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_postanchor_smoke_summary
```

- status: completed
- note: metric_rows=2 best=MLP-HC4-T9-earlyburst-alt10-stop400-postM181-lr2 route=F0-MetricNoEffect

## 2026-06-06 09:26:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc4_debt_calibrated_source_postanchor_smoke_summary
```

- status: completed
- note: rows=0 dominant=MetricNoEffect best=

## 2026-06-06 09:36:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc4_t9_poststop_slowstate_wired_smoke --baseline-dir /home/chengshun.wang/DG-LCA/results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc4_t9_poststop_slowstate_wired_smoke
```

- status: completed
- note: metric_rows=2 best=MLP-HC4-T9-earlyburst-alt10-stop400-postM181 route=F0-MetricNoEffect

## 2026-06-06 09:39:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc4_t9_poststop_slowstate_wired_controls_smoke --baseline-dir /home/chengshun.wang/DG-LCA/results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc4_t9_poststop_slowstate_wired_controls_smoke
```

- status: completed
- note: metric_rows=2 best=MLP-HC4-T9-earlyburst-alt10-stop400-postM181 route=F0-MetricNoEffect

## 2026-06-06 09:39:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc4_t9_poststop_slowstate_wired_controls_smoke --baseline-dir /home/chengshun.wang/DG-LCA/results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc4_t9_poststop_slowstate_wired_controls_smoke
```

- status: completed
- note: metric_rows=2 best=MLP-HC4-T9-earlyburst-alt10-stop400-postM181 route=F0-MetricNoEffect

## 2026-06-06 09:41:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc4_t9_poststop_slowstate_wired_3ds_smoke --baseline-dir /home/chengshun.wang/DG-LCA/results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc4_t9_poststop_slowstate_wired_3ds_smoke
```

- status: completed
- note: metric_rows=2 best=MLP-HC4-T9-earlyburst-alt10-stop400-postM181 route=F0-MetricNoEffect

## 2026-06-06 09:54:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc5_t10_observable_mid_debt_3ds_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc5_t10_observable_mid_debt_3ds_smoke_summary
```

- status: completed
- note: metric_rows=0 best= route=F0-MetricNoEffect

## 2026-06-06 09:54:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc5_t10_observable_mid_debt_3ds_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc5_t10_observable_mid_debt_3ds_smoke_summary
```

- status: completed
- note: metric_rows=4 best=MLP-HC5-T10-earlyburst-alt10-stop400-postM218 route=F0-MetricNoEffect

## 2026-06-06 09:59:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc5_t10_stoponly_3ds_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc5_t10_stoponly_3ds_smoke_summary
```

- status: completed
- note: metric_rows=3 best=MLP-HC5-T10-earlyburst-alt10-stop800 route=F0-MetricNoEffect

## 2026-06-06 10:11:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 0
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 10:11:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 0
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 10:13:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_3ds_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_3ds_smoke_summary
```

- status: completed
- note: metric_rows=0 best= route=F0-MetricNoEffect

## 2026-06-06 10:13:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_3ds_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_3ds_smoke_summary
```

- status: completed
- note: metric_rows=2 best=MLP-HC6-T10M181Bridge-stop400 route=F0-MetricNoEffect

## 2026-06-06 10:20:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_lowlr_3ds_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_lowlr_3ds_smoke_summary
```

- status: completed
- note: metric_rows=0 best= route=F0-MetricNoEffect

## 2026-06-06 10:20:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_lowlr_3ds_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_lowlr_3ds_smoke_summary
```

- status: completed
- note: metric_rows=3 best=MLP-HC6-T10M181Bridge-stop1600-lr025 route=F0-MetricNoEffect

## 2026-06-06 10:23:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_3ds_smoke_summary
```

- status: completed
- note: rows=0 dominant=MetricNoEffect best=

## 2026-06-06 10:23:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_lowlr_3ds_smoke_summary
```

- status: completed
- note: rows=0 dominant=MetricNoEffect best=

## 2026-06-06 10:24:22 +0800 Continuation Reproduction Notes

本节补录本轮关键手工命令，runner 自动写入的 command journal 只保留了缩略 command；这里保留完整 `spec-ids` / artifact 路径，方便复现。

### Contract / Compile

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_05_metric_first_fu.py experiments/run_v22_05_s012_truth_gate.py
```

- status: completed
- note: no stdout; py_compile pass.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --mode mechanism-contracts
```

- status: completed-noop
- note: mode 名写成 `mechanism-contracts`，runner 识别的是 `mechanism_contracts`；未写出 rows，作为命令错误保留。

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --mode mechanism_contracts
```

- status: completed
- note: `v22_05_code_truth_gate.csv` 最新 mechanism_contracts=`35/35;alias_undeclared=0`；M252 contract row 已落盘。

### M252 Preheated Bridge Smoke

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_3ds_smoke --run-label v22_05_hc6_t10_m181_bridge_3ds_smoke --spec-ids CTRL-AdamW,CTRL-SGD,CTRL-NoOpMatchedOverhead,CTRL-RandomMatchedNorm,MLP-HC6-T10M181Bridge-stop400,MLP-HC6-T10M181Bridge-stop800 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --shard-count 1 --shard-index 0 --device cuda:0
```

- status: completed
- artifact: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_3ds_smoke`

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_3ds_smoke --run-label v22_05_hc6_t10_m181_bridge_3ds_smoke --spec-ids CTRL-AdamW,CTRL-SGD,CTRL-NoOpMatchedOverhead,CTRL-RandomMatchedNorm,MLP-HC6-T10M181Bridge-stop400,MLP-HC6-T10M181Bridge-stop800 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --shard-count 1 --shard-index 0 --device cuda:0 --merge-only
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_3ds_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_3ds_smoke_summary
```

- status: completed
- note: metric_rows=2; best=`MLP-HC6-T10M181Bridge-stop400`; route=`F0-MetricNoEffect`.

### M252 Low Post-LR / Late Handoff Smoke

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v21_01_source_retention.py experiments/run_v22_05_metric_first_fu.py
```

- status: completed
- note: no stdout; spec/summary mapping compile pass.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from experiments.run_v21_01_source_retention import _mlp_specs
PY
```

- status: failed
- note: `_mlp_specs` is not an exported helper; this was only a local self-check attempt, not an experiment runner.

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_lowlr_3ds_smoke --run-label v22_05_hc6_t10_m181_bridge_lowlr_3ds_smoke --spec-ids CTRL-AdamW,CTRL-SGD,CTRL-NoOpMatchedOverhead,CTRL-RandomMatchedNorm,MLP-HC6-T10M181Bridge-stop800-lr025,MLP-HC6-T10M181Bridge-stop1200-lr025,MLP-HC6-T10M181Bridge-stop1600-lr025 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --shard-count 1 --shard-index 0 --device cuda:0
```

- status: completed
- artifact: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_lowlr_3ds_smoke`

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_lowlr_3ds_smoke --run-label v22_05_hc6_t10_m181_bridge_lowlr_3ds_smoke --spec-ids CTRL-AdamW,CTRL-SGD,CTRL-NoOpMatchedOverhead,CTRL-RandomMatchedNorm,MLP-HC6-T10M181Bridge-stop800-lr025,MLP-HC6-T10M181Bridge-stop1200-lr025,MLP-HC6-T10M181Bridge-stop1600-lr025 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --shard-count 1 --shard-index 0 --device cuda:0 --merge-only
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_lowlr_3ds_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_lowlr_3ds_smoke_summary
```

- status: completed
- note: metric_rows=3; best=`MLP-HC6-T10M181Bridge-stop1600-lr025`; route=`F0-MetricNoEffect`.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_3ds_smoke_summary
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_autopsy.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc6_t10_m181_bridge_lowlr_3ds_smoke_summary
```

- status: completed
- note: both produced rows=0/dominant=MetricNoEffect because this autopsy runner only ingested the root attempt layout, not these continuation summary dirs; not used as scientific evidence.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_finalize.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04
```

- status: failed
- note: finalizer does not accept `--baseline-dir`; no output artifact changed.

## 2026-06-06 10:38:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode mechanism_contracts --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 0
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 10:43:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_analysis
```

- status: completed
- note: metric_rows=3 best=MLP-HC7-T11-GradObservableMidDebt-stop1600 route=F0-MetricNoEffect

## 2026-06-06 10:47:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_analysis
```

- status: completed
- note: metric_rows=3 best=MLP-HC7-T11-GradObservableMidDebt-stop800 route=F0-MetricNoEffect

## 2026-06-06 10:54:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_analysis
```

- status: completed
- note: metric_rows=3 best=MLP-HC7-T11-GradObservableMidDebt-stop800 route=F0-MetricNoEffect

## 2026-06-06 10:57:29 +0800 Continuation: HC7/T11 GradientObservableMidDebt

### Goal / Plan Check

- Checked `docs/DG-KAN_v22.05_MetricFirst_FunctionalUpdate_BasisEfficiency_4GPU_完整计划.md` and the current root artifact `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/v22_05_final_route.json`.
- Goal is still not achieved: D-RAT/D-RBF official fused blocker is closed, but root `functional_route=F0-MetricNoEffect`, `promotion_allowed=0`, and `KAN_mapping_decision=KANSourceChannelMismatch`.
- Per the plan's repair direction after MetricNoEffect, implemented one new train-only metric-target family instead of reusing validation/test/future information.

### Code / Wiring Changes

- `dgkan/fu/mechanisms.py`: added `M253-MetricTargetGradientObservableMidDebtFU` / `T11-GradientObservableMidDebtSource`.
  - It starts from the T10 observable-mid-debt source target.
  - It projects the target update onto the current train-batch loss-gradient halfspace on the nonzero target support.
  - It records `target_gradient_conflict_cos_before`, `target_gradient_conflict_cos_after`, `target_gradient_conflict_removed_fraction`, and `source_observability_gate_accept`.
- `experiments/run_v21_01_source_retention.py`: registered HC7/T11 specs:
  - `MLP-HC7-T11-GradObservableMidDebt-stop400`
  - `MLP-HC7-T11-GradObservableMidDebt-stop800`
  - `MLP-HC7-T11-GradObservableMidDebt-stop1600`
  - Added these IDs to the `--scope mlp` allowlist after the first no-row smoke exposed the missing wiring.
- `experiments/run_v22_05_metric_first_fu.py`: registered T11 metric names and summary diagnostic fields.
- `experiments/run_v22_05_s012_truth_gate.py`: added M253 to the expected v22.05 mechanism contract list.
- `experiments/run_v17_common.py`: added T11 diagnostic keys to the trace row whitelist so raw traces preserve the halfspace diagnostics.

### Compile / Contract Gate

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_05_metric_first_fu.py experiments/run_v22_05_s012_truth_gate.py
```

- status: completed
- note: no stdout; py_compile pass.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05 --mode mechanism_contracts
```

- status: completed
- note: `mechanism_contracts,1,v22.05 schema+semantic,36/36;alias_undeclared=0,`
- M253 contract row: `M253-MetricTargetGradientObservableMidDebtFU,function_metric_target,metric_target_T11_gradient_observable_mid_debt_source,...,leakage_free=1,mechanism_family=T11-GradientObservableMidDebtSource`

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import torch
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
PY
```

- status: completed
- output: `cuda_available True`, `device_count 4`

### Runner Wiring Checks / Non-scientific Attempts

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC7-T11-GradObservableMidDebt-stop400,MLP-HC7-T11-GradObservableMidDebt-stop800,MLP-HC7-T11-GradObservableMidDebt-stop1600 --run-label v22_05_hc7_t11_gradient_observable_smoke --shard-count 1 --shard-index 0
```

- status: completed but no scientific rows
- note: first HC7/T11 smoke produced only a header because the new specs were not yet included in the `--scope mlp` selected-spec allowlist. Fixed the allowlist before any result was used.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from experiments.run_v21_01_source_retention import parser, selected_specs
args = parser().parse_args(['--scope','mlp','--spec-ids','MLP-HC7-T11-GradObservableMidDebt-stop400,MLP-HC7-T11-GradObservableMidDebt-stop800,MLP-HC7-T11-GradObservableMidDebt-stop1600'])
print([s['v21_id'] for s in selected_specs(args)])
PY
```

- status: completed
- output: `['MLP-HC7-T11-GradObservableMidDebt-stop400', 'MLP-HC7-T11-GradObservableMidDebt-stop800', 'MLP-HC7-T11-GradObservableMidDebt-stop1600']`

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC7-T11-GradObservableMidDebt-stop400,MLP-HC7-T11-GradObservableMidDebt-stop800,MLP-HC7-T11-GradObservableMidDebt-stop1600 --run-label v22_05_hc7_t11_gradient_observable_smoke --shard-count 1 --shard-index 0
```

- status: completed but not used as final scientific evidence
- note: T11-only run lacked matched controls, so source-retention/control-equivalence classification was incomplete. Superseded by the controls-included smoke below.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_analysis --label v22.05 HC7 T11 gradient-observable smoke
```

- status: failed
- note: shell argument quoting error; unquoted label words were parsed as extra arguments. No scientific result was taken from this failed command.

### Final HC7/T11 Smoke With Matched Controls

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC7-T11-GradObservableMidDebt-stop400,MLP-HC7-T11-GradObservableMidDebt-stop800,MLP-HC7-T11-GradObservableMidDebt-stop1600,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v22_05_hc7_t11_gradient_observable_smoke --shard-count 1 --shard-index 0
```

- status: completed
- artifact dir: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke`
- note: rerun after trace whitelist fix; this is the final HC7/T11 smoke used for analysis.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke --scope mlp --run-label v22_05_hc7_t11_gradient_observable_smoke --merge-only
```

- status: completed

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_analysis --label 'v22.05 HC7 T11 gradient-observable smoke'
```

- status: completed
- artifact dir: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_analysis`
- result: `metric_rows=3`, best=`MLP-HC7-T11-GradObservableMidDebt-stop800`, `productive_metric_rows=0`, `functional_route=F0-MetricNoEffect`
- decision: smoke failed the h3200/productive route gate, so no 4GPU full was launched.

### Artifact Size / Hash Check

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import hashlib
from pathlib import Path
paths = [
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke/v21_01_source_retention_matrix.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke/v21_01_source_retention_raw_traces.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_analysis/v22_05_metric_first_mlp_summary.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_analysis/v22_05_metric_first_route.json',
]
for s in paths:
    p = Path(s)
    data = p.read_bytes()
    print(s, 'exists', p.exists(), 'size', p.stat().st_size, 'lines', data.count(b'\n'), 'sha256', hashlib.sha256(data).hexdigest())
PY
```

- status: completed
- output:
  - `v21_01_source_retention_matrix.csv` size=92801 lines=22 sha256=`46b23945404af80aa64f89d4698682d9413bae82529588e83ebfced0e14f8b76`
  - `v21_01_source_retention_raw_traces.csv` size=186583 lines=190 sha256=`4a3a7be845e7cc40ddcddfb0fa237e3669c52ec05c4c902328c555adcd3cecce`
  - `v22_05_metric_first_mlp_summary.csv` size=3505 lines=4 sha256=`23a1433a451731f36a803ce69ab23436f88951e3ae46ece0ff83c221dd0fef65`
  - `v22_05_metric_first_route.json` size=472 lines=13 sha256=`3e367922e8776cd0dd8d7c78fe4b4ae5a2931dc403609a9bee7dc0ae28564dbd`

## 2026-06-06 11:10:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 0
```

- status: completed
- note: pass=0 failed=import_closure

## 2026-06-06 11:12:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 11:21:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_analysis
```

- status: completed
- note: metric_rows=3 best=MLP-HC8-T12-PreH3200SourceAntiWashout-stop1200 route=F0-MetricNoEffect

## 2026-06-06 11:34:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_residual_strength_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_residual_strength_analysis
```

- status: completed
- note: metric_rows=4 best=MLP-HC8-T12-PreH3200SourceAntiWashout-stop1600-lr200 route=F0-MetricNoEffect

## 2026-06-06 11:44:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_longwindow_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_longwindow_analysis
```

- status: completed
- note: metric_rows=3 best=MLP-HC8-T12-PreH3200SourceAntiWashout-stop2400-lr200 route=F0-MetricNoEffect

## 2026-06-06 11:45:31 +0800 Continuation: HC8/T12 Source-Channel Anti-Washout

### Code / Contract

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_05_metric_first_fu.py experiments/run_v22_05_s012_truth_gate.py
```

- status: completed
- note: no stdout; syntax check passed after adding M254/T12 and HC8 specs.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t12_contract_check --self-contained-import-check 0
```

- status: completed but not final audit evidence
- result: `mechanism_contracts=37/37;alias_undeclared=0`, `import_closure=0`
- note: import command itself returned 0, but the audit pass flag requires `--self-contained-import-check 1`; superseded by the next command.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t12_contract_check_selfcontained --self-contained-import-check 1
```

- status: completed
- result: `pass=1`, `mechanism_contracts=37/37;alias_undeclared=0`, `import_closure=1`

### HC8/T12 Low Residual Smoke

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC8-T12-PreH3200SourceAntiWashout-stop800,MLP-HC8-T12-PreH3200SourceAntiWashout-stop1200,MLP-HC8-T12-PreH3200SourceAntiWashout-stop1600,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v22_05_hc8_t12_pre_h3200_source_antiwashout_smoke --shard-count 1 --shard-index 0
```

- status: completed
- result: rows=21 traces=189

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_smoke --scope mlp --run-label v22_05_hc8_t12_pre_h3200_source_antiwashout_smoke --merge-only
```

- status: completed
- result: rows=21 grouped=7

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_analysis --label 'v22.05 HC8 T12 pre-H3200 source-channel anti-washout smoke'
```

- status: completed
- result: `metric_rows=3`, best=`MLP-HC8-T12-PreH3200SourceAntiWashout-stop1200`, route=`F0-MetricNoEffect`

### HC8/T12 Residual Strength Smoke

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v21_01_source_retention.py experiments/run_v22_05_metric_first_fu.py
```

- status: completed
- note: syntax check passed after adding `lr100/lr200` residual-strength specs.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from experiments.run_v21_01_source_retention import build_jobs
PY
```

- status: failed
- note: auxiliary dry-check only; `build_jobs` is not exported by the script. No scientific result was taken from this failed command.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_residual_strength_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC8-T12-PreH3200SourceAntiWashout-stop800-lr100,MLP-HC8-T12-PreH3200SourceAntiWashout-stop1200-lr100,MLP-HC8-T12-PreH3200SourceAntiWashout-stop1600-lr100,MLP-HC8-T12-PreH3200SourceAntiWashout-stop1600-lr200,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v22_05_hc8_t12_pre_h3200_source_antiwashout_residual_strength_smoke --shard-count 1 --shard-index 0
```

- status: completed
- result: rows=24 traces=216

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_residual_strength_smoke --scope mlp --run-label v22_05_hc8_t12_pre_h3200_source_antiwashout_residual_strength_smoke --merge-only
```

- status: completed
- result: rows=24 grouped=8

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_residual_strength_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_residual_strength_analysis --label 'v22.05 HC8 T12 pre-H3200 source-channel anti-washout residual strength smoke'
```

- status: completed
- result: `metric_rows=4`, best=`MLP-HC8-T12-PreH3200SourceAntiWashout-stop1600-lr200`, route=`F0-MetricNoEffect`

### HC8/T12 Long-Window Smoke

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v21_01_source_retention.py experiments/run_v22_05_metric_first_fu.py
```

- status: completed
- note: syntax check passed after adding stop2400/stop3200 specs.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_longwindow_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC8-T12-PreH3200SourceAntiWashout-stop2400-lr100,MLP-HC8-T12-PreH3200SourceAntiWashout-stop2400-lr200,MLP-HC8-T12-PreH3200SourceAntiWashout-stop3200-lr200,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v22_05_hc8_t12_pre_h3200_source_antiwashout_longwindow_smoke --shard-count 1 --shard-index 0
```

- status: completed
- result: rows=21 traces=189

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_longwindow_smoke --scope mlp --run-label v22_05_hc8_t12_pre_h3200_source_antiwashout_longwindow_smoke --merge-only
```

- status: completed
- result: rows=21 grouped=7

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_longwindow_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_longwindow_analysis --label 'v22.05 HC8 T12 pre-H3200 source-channel anti-washout long-window smoke'
```

- status: completed
- result: `metric_rows=3`, best=`MLP-HC8-T12-PreH3200SourceAntiWashout-stop2400-lr200`, route=`F0-MetricNoEffect`

### Artifact Hash Check

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import hashlib
from pathlib import Path
paths = [
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t12_contract_check_selfcontained/v22_05_code_truth_gate.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_analysis/v22_05_metric_first_mlp_summary.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_analysis/v22_05_metric_first_route.json',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_residual_strength_analysis/v22_05_metric_first_mlp_summary.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_residual_strength_analysis/v22_05_metric_first_route.json',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_longwindow_analysis/v22_05_metric_first_mlp_summary.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_longwindow_analysis/v22_05_metric_first_route.json',
]
for s in paths:
    p=Path(s); data=p.read_bytes(); lines=data.count(b'\n')
    print(f"{s}|{p.exists()}|{p.stat().st_size}|{lines}|{hashlib.sha256(data).hexdigest()}")
PY
```

- status: completed
- note: first hash helper attempt had an f-string backslash syntax error and produced no result; the command above is the corrected artifact hash check.

## 2026-06-06 11:59:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 12:09:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc9_t13_risk_weighted_split_consensus_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc9_t13_risk_weighted_split_consensus_analysis
```

- status: completed
- note: metric_rows=4 best=MLP-HC9-T13-RiskWeightedSplitConsensus-stop2400-lr100 route=F0-MetricNoEffect

## 2026-06-06 12:14:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 12:22:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc10_t14_view_consistent_source_carry_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc10_t14_view_consistent_source_carry_analysis
```

- status: completed
- note: metric_rows=3 best=MLP-HC10-T14-ViewConsistentSourceCarry-stop1600-lr100 route=F0-MetricNoEffect

## 2026-06-06 12:28:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 12:37:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_continuous_source_carry_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_continuous_source_carry_analysis
```

- status: completed
- note: metric_rows=3 best=MLP-HC11-T15-ContinuousSourceCarry-stop2400-lr300 route=F0-MetricNoEffect

## 2026-06-06 12:45:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_strength_window_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_strength_window_analysis
```

- status: completed
- note: metric_rows=2 best=MLP-HC11-T15-ContinuousSourceCarry-stop3200-lr300 route=F0-MetricNoEffect

## 2026-06-06 12:48:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 12:54:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc12_t16_noise_orthogonal_source_carry_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc12_t16_noise_orthogonal_source_carry_analysis
```

- status: completed
- note: metric_rows=2 best=MLP-HC12-T16-NoiseOrthogonalCarry-stop2400-lr500 route=F0-MetricNoEffect

## 2026-06-06 12:56:32 +0800 补录：HC9/T13 -> HC12/T16 完整执行命令

### HC9/T13 Risk-Weighted Split-Consensus

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_05_metric_first_fu.py experiments/run_v22_05_s012_truth_gate.py
```

- status: completed
- result: py_compile pass

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t13_contract_check_selfcontained --self-contained-import-check 1
```

- status: completed
- result: `mechanism_contracts=38/38;alias_undeclared=0`

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc9_t13_risk_weighted_split_consensus_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC9-T13-RiskWeightedSplitConsensus-stop800-lr100,MLP-HC9-T13-RiskWeightedSplitConsensus-stop1200-lr100,MLP-HC9-T13-RiskWeightedSplitConsensus-stop1600-lr100,MLP-HC9-T13-RiskWeightedSplitConsensus-stop2400-lr100,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v22_05_hc9_t13_risk_weighted_split_consensus_smoke --shard-count 1 --shard-index 0
```

- status: completed
- result: shard rows=24 data, traces=216

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc9_t13_risk_weighted_split_consensus_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC9-T13-RiskWeightedSplitConsensus-stop800-lr100,MLP-HC9-T13-RiskWeightedSplitConsensus-stop1200-lr100,MLP-HC9-T13-RiskWeightedSplitConsensus-stop1600-lr100,MLP-HC9-T13-RiskWeightedSplitConsensus-stop2400-lr100,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v22_05_hc9_t13_risk_weighted_split_consensus_smoke --shard-count 1 --shard-index 0 --merge-only
```

- status: completed

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc9_t13_risk_weighted_split_consensus_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc9_t13_risk_weighted_split_consensus_analysis --label 'v22.05 HC9 T13 risk-weighted split-consensus smoke'
```

- status: completed
- result: `metric_rows=4`, best=`MLP-HC9-T13-RiskWeightedSplitConsensus-stop2400-lr100`, route=`F0-MetricNoEffect`

### HC10/T14 View-Consistent Source-Carry

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_05_metric_first_fu.py experiments/run_v22_05_s012_truth_gate.py
```

- status: completed
- result: py_compile pass

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t14_contract_check_selfcontained --self-contained-import-check 1
```

- status: completed
- result: `mechanism_contracts=39/39;alias_undeclared=0`

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc10_t14_view_consistent_source_carry_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC10-T14-ViewConsistentSourceCarry-stop800-lr100,MLP-HC10-T14-ViewConsistentSourceCarry-stop1600-lr100,MLP-HC10-T14-ViewConsistentSourceCarry-stop2400-lr100,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v22_05_hc10_t14_view_consistent_source_carry_smoke --shard-count 1 --shard-index 0
```

- status: completed
- result: shard rows=21 data, traces=189

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc10_t14_view_consistent_source_carry_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC10-T14-ViewConsistentSourceCarry-stop800-lr100,MLP-HC10-T14-ViewConsistentSourceCarry-stop1600-lr100,MLP-HC10-T14-ViewConsistentSourceCarry-stop2400-lr100,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v22_05_hc10_t14_view_consistent_source_carry_smoke --shard-count 1 --shard-index 0 --merge-only
```

- status: completed

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc10_t14_view_consistent_source_carry_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc10_t14_view_consistent_source_carry_analysis --label 'v22.05 HC10 T14 view-consistent source-carry smoke'
```

- status: completed
- result: `metric_rows=3`, best=`MLP-HC10-T14-ViewConsistentSourceCarry-stop1600-lr100`, route=`F0-MetricNoEffect`

### HC11/T15 Continuous Source-Carry

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_05_metric_first_fu.py experiments/run_v22_05_s012_truth_gate.py
```

- status: completed
- result: py_compile pass

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t15_contract_check_selfcontained --self-contained-import-check 1
```

- status: completed
- result: `mechanism_contracts=40/40;alias_undeclared=0`

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_continuous_source_carry_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC11-T15-ContinuousSourceCarry-stop1600-lr200,MLP-HC11-T15-ContinuousSourceCarry-stop2400-lr200,MLP-HC11-T15-ContinuousSourceCarry-stop2400-lr300,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v22_05_hc11_t15_continuous_source_carry_smoke --shard-count 1 --shard-index 0
```

- status: completed
- result: shard rows=21 data, traces=189

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_continuous_source_carry_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC11-T15-ContinuousSourceCarry-stop1600-lr200,MLP-HC11-T15-ContinuousSourceCarry-stop2400-lr200,MLP-HC11-T15-ContinuousSourceCarry-stop2400-lr300,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v22_05_hc11_t15_continuous_source_carry_smoke --shard-count 1 --shard-index 0 --merge-only
```

- status: completed

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_continuous_source_carry_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_continuous_source_carry_analysis --label 'v22.05 HC11 T15 continuous source-carry smoke'
```

- status: completed
- result: `metric_rows=3`, best=`MLP-HC11-T15-ContinuousSourceCarry-stop2400-lr300`, route=`F0-MetricNoEffect`

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v21_01_source_retention.py experiments/run_v22_05_metric_first_fu.py
```

- status: completed
- result: py_compile pass for added strength/window specs

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_strength_window_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC11-T15-ContinuousSourceCarry-stop2400-lr500,MLP-HC11-T15-ContinuousSourceCarry-stop3200-lr300,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v22_05_hc11_t15_strength_window_smoke --shard-count 1 --shard-index 0
```

- status: completed
- result: shard rows=18 data, traces=162

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_strength_window_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC11-T15-ContinuousSourceCarry-stop2400-lr500,MLP-HC11-T15-ContinuousSourceCarry-stop3200-lr300,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v22_05_hc11_t15_strength_window_smoke --shard-count 1 --shard-index 0 --merge-only
```

- status: completed

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_strength_window_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_strength_window_analysis --label 'v22.05 HC11 T15 continuous source-carry strength/window smoke'
```

- status: completed
- result: `metric_rows=2`, best=`MLP-HC11-T15-ContinuousSourceCarry-stop3200-lr300`, route=`F0-MetricNoEffect`

### HC12/T16 Noise-Orthogonal Source-Carry

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_05_metric_first_fu.py experiments/run_v22_05_s012_truth_gate.py
```

- status: completed
- result: py_compile pass

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_s012_truth_gate.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t16_contract_check_selfcontained --self-contained-import-check 1
```

- status: completed
- result: `mechanism_contracts=41/41;alias_undeclared=0`

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc12_t16_noise_orthogonal_source_carry_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC12-T16-NoiseOrthogonalCarry-stop2400-lr500,MLP-HC12-T16-NoiseOrthogonalCarry-stop3200-lr300,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v22_05_hc12_t16_noise_orthogonal_source_carry_smoke --shard-count 1 --shard-index 0
```

- status: completed
- result: shard rows=18 data, traces=162

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc12_t16_noise_orthogonal_source_carry_smoke --scope mlp --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --steps 3200 --spec-ids MLP-HC12-T16-NoiseOrthogonalCarry-stop2400-lr500,MLP-HC12-T16-NoiseOrthogonalCarry-stop3200-lr300,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v22_05_hc12_t16_noise_orthogonal_source_carry_smoke --shard-count 1 --shard-index 0 --merge-only
```

- status: completed

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_metric_first_fu.py --source-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc12_t16_noise_orthogonal_source_carry_smoke --baseline-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --out-dir results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc12_t16_noise_orthogonal_source_carry_analysis --label 'v22.05 HC12 T16 noise-orthogonal source-carry smoke'
```

- status: completed
- result: `metric_rows=2`, best=`MLP-HC12-T16-NoiseOrthogonalCarry-stop2400-lr500`, route=`F0-MetricNoEffect`

### Artifact Hash Check

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import hashlib
from pathlib import Path
paths = [
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t13_contract_check_selfcontained/v22_05_code_truth_gate.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc9_t13_risk_weighted_split_consensus_analysis/v22_05_metric_first_mlp_summary.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc9_t13_risk_weighted_split_consensus_analysis/v22_05_metric_first_route.json',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t14_contract_check_selfcontained/v22_05_code_truth_gate.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc10_t14_view_consistent_source_carry_analysis/v22_05_metric_first_mlp_summary.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc10_t14_view_consistent_source_carry_analysis/v22_05_metric_first_route.json',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t15_contract_check_selfcontained/v22_05_code_truth_gate.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_continuous_source_carry_analysis/v22_05_metric_first_mlp_summary.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_continuous_source_carry_analysis/v22_05_metric_first_route.json',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_strength_window_analysis/v22_05_metric_first_mlp_summary.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_strength_window_analysis/v22_05_metric_first_route.json',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t16_contract_check_selfcontained/v22_05_code_truth_gate.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc12_t16_noise_orthogonal_source_carry_analysis/v22_05_metric_first_mlp_summary.csv',
 'results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc12_t16_noise_orthogonal_source_carry_analysis/v22_05_metric_first_route.json',
]
for s in paths:
    p = Path(s)
    data = p.read_bytes()
    print(f"{s}|1|{p.stat().st_size}|{data.count(bytes([10]))}|{hashlib.sha256(data).hexdigest()}")
PY
```

- status: completed

## 2026-06-07 14:57:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_05_drat_drbf_repair.py --out-dir results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/telemetry_sanity_v22_10 --device cuda:2 --batch-sizes 128 --hidden 128 --iters 1 --warmup 0
```

- status: completed
- note: rows=16 near=8 official=0
