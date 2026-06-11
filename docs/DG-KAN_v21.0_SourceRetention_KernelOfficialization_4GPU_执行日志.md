# DG-KAN v21.0 Source Retention + Kernel Officialization 执行日志

生成时间：2026-06-03 22:27:24 +0800

## 命令日志

### 2026-06-03 16:19:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py --check all --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21 --device cuda:0
```
- status: started
- note: 

### 2026-06-03 16:19:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py --check all
```
- status: completed
- note: S0_5_preflight_pass=1

### 2026-06-03 16:20:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 1 --shard-count 4 --device cuda:1
```
- status: started
- note: jobs=10

### 2026-06-03 16:20:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 2 --shard-count 4 --device cuda:2
```
- status: started
- note: jobs=10

### 2026-06-03 16:20:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 3 --shard-count 4 --device cuda:3
```
- status: started
- note: jobs=10

### 2026-06-03 16:20:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0 --shard-count 4 --device cuda:0
```
- status: started
- note: jobs=10

### 2026-06-03 16:20:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 3
```
- status: completed
- note: rows=10

### 2026-06-03 16:20:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 2
```
- status: completed
- note: rows=10

### 2026-06-03 16:20:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 1
```
- status: completed
- note: rows=10

### 2026-06-03 16:20:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0
```
- status: completed
- note: rows=10

### 2026-06-03 16:20:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --merge-only
```
- status: completed
- note: rows=40 waterfall=480 grad=40

### 2026-06-03 16:21:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --merge-only
```
- status: completed
- note: rows=40 waterfall=480 grad=40

### 2026-06-03 16:22:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 0 --device cuda:0
```
- status: started
- note: jobs=25

### 2026-06-03 16:22:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 1 --device cuda:1
```
- status: started
- note: jobs=25

### 2026-06-03 16:22:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 3 --device cuda:3
```
- status: started
- note: jobs=24

### 2026-06-03 16:22:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 2 --device cuda:2
```
- status: started
- note: jobs=25

### 2026-06-03 16:26:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 3
```
- status: completed
- note: rows=24 traces=240

### 2026-06-03 16:26:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 0
```
- status: completed
- note: rows=25 traces=250

### 2026-06-03 16:26:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 1
```
- status: completed
- note: rows=25 traces=250

### 2026-06-03 16:26:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 2
```
- status: completed
- note: rows=25 traces=250

### 2026-06-03 16:26:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --merge-only
```
- status: completed
- note: rows=99 grouped=11

### 2026-06-03 16:28:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_kan_source_channel_writer.py --scope kan_writer --shard-index 0..3
```
- status: blocked:ModuleNotFoundError
- note: wrapper lacked repo-root sys.path guard; fixed in run_v21_kan_source_channel_writer.py and py_compile rerun

### 2026-06-03 16:28:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 0 --device cuda:0
```
- status: started
- note: jobs=32

### 2026-06-03 16:28:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 3 --device cuda:3
```
- status: started
- note: jobs=31

### 2026-06-03 16:28:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 1 --device cuda:1
```
- status: started
- note: jobs=32

### 2026-06-03 16:28:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 2 --device cuda:2
```
- status: started
- note: jobs=31

### 2026-06-03 16:38:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 3
```
- status: completed
- note: rows=31 traces=310

### 2026-06-03 16:38:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 0
```
- status: completed
- note: rows=32 traces=320

### 2026-06-03 16:38:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 2
```
- status: completed
- note: rows=31 traces=310

### 2026-06-03 16:39:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 1
```
- status: completed
- note: rows=32 traces=320

### 2026-06-03 16:39:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --merge-only
```
- status: completed
- note: rows=126 grouped=14

### 2026-06-03 16:39:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_function_space_target_reset.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: started
- note: 

### 2026-06-03 16:39:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_function_space_target_reset.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: completed
- note: rows=2

### 2026-06-03 16:39:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_finalize.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: started
- note: 

### 2026-06-03 16:39:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_finalize.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: completed
- note: route=R3-WeakMLPSourceOnly-H3200Washout promotion_allowed=0

### 2026-06-03 16:40:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_finalize.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: started
- note: 

### 2026-06-03 16:40:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_finalize.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: completed
- note: route=R3-WeakMLPSourceOnly-H3200Washout promotion_allowed=0

### 2026-06-03 16:52:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_optimizer_overwrite_diagnostic.py --scope mlp_source --device cuda:0
```
- status: started
- note: jobs=81

### 2026-06-03 16:52:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_optimizer_overwrite_diagnostic.py --scope mlp_source
```
- status: completed
- note: rows=81 grouped=9

### 2026-06-03 16:55:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 0 --device cuda:0
```
- status: started
- note: jobs=3

### 2026-06-03 16:55:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 3 --device cuda:3
```
- status: started
- note: jobs=2

### 2026-06-03 16:55:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 1 --device cuda:1
```
- status: started
- note: jobs=2

### 2026-06-03 16:55:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 2 --device cuda:2
```
- status: started
- note: jobs=2

### 2026-06-03 16:55:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 3
```
- status: completed
- note: rows=2 traces=20

### 2026-06-03 16:55:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 2
```
- status: completed
- note: rows=2 traces=20

### 2026-06-03 16:55:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 1
```
- status: completed
- note: rows=2 traces=20

### 2026-06-03 16:55:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 0
```
- status: completed
- note: rows=3 traces=30

### 2026-06-03 16:55:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --merge-only
```
- status: completed
- note: rows=108 grouped=12

### 2026-06-03 17:01:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 0 --device cuda:0
```
- status: started
- note: jobs=3

### 2026-06-03 17:01:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 1 --device cuda:1
```
- status: started
- note: jobs=2

### 2026-06-03 17:01:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 2 --device cuda:2
```
- status: started
- note: jobs=2

### 2026-06-03 17:01:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 3 --device cuda:3
```
- status: started
- note: jobs=2

### 2026-06-03 17:01:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 2
```
- status: completed
- note: rows=2 traces=20

### 2026-06-03 17:01:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 1
```
- status: completed
- note: rows=2 traces=20

### 2026-06-03 17:01:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 3
```
- status: completed
- note: rows=2 traces=20

### 2026-06-03 17:01:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --shard-index 0
```
- status: completed
- note: rows=3 traces=30

### 2026-06-03 17:01:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope mlp_source --merge-only
```
- status: completed
- note: rows=117 grouped=13

### 2026-06-03 17:02:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py --check all --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21 --device cuda:0
```
- status: started
- note: 

### 2026-06-03 17:02:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py --check all
```
- status: completed
- note: S0_5_preflight_pass=1

### 2026-06-03 17:03:34 +0800

```bash
cp smoke_m47_bridge and smoke_m48_dualtime summaries/matrices into official_v21
```
- status: completed
- note: files: v21_m47_bridge_smoke_summary/matrix.csv; v21_m48_dualtime_smoke_summary/matrix.csv

### 2026-06-03 17:04:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_finalize.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: started
- note: 

### 2026-06-03 17:04:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_finalize.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: completed
- note: route=R3-WeakMLPSourceOnly-H3200Washout promotion_allowed=0

### 2026-06-03 17:12:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_function_space_target_contrast.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21 --device cuda:0
```
- status: started
- note: rows_planned=108

### 2026-06-03 17:12:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_function_space_target_contrast.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: completed
- note: rows=108 summary=12

### 2026-06-03 17:24:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 0 --device cuda:0
```
- status: started
- note: jobs=18

### 2026-06-03 17:27:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 0
```
- status: completed
- note: rows=18 traces=162

### 2026-06-03 17:27:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --merge-only
```
- status: completed
- note: rows=144 grouped=24

### 2026-06-03 17:28:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 0 --device cuda:0
```
- status: started
- note: jobs=36

### 2026-06-03 17:28:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 3 --device cuda:3
```
- status: started
- note: jobs=36

### 2026-06-03 17:28:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 2 --device cuda:2
```
- status: started
- note: jobs=36

### 2026-06-03 17:28:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 1 --device cuda:1
```
- status: started
- note: jobs=36

### 2026-06-03 17:36:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 2
```
- status: completed
- note: rows=36 traces=360

### 2026-06-03 17:36:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 3
```
- status: completed
- note: rows=36 traces=360

### 2026-06-03 17:36:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 0
```
- status: completed
- note: rows=36 traces=360

### 2026-06-03 17:36:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 1
```
- status: completed
- note: rows=36 traces=360

### 2026-06-03 17:37:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --merge-only
```
- status: completed
- note: rows=288 grouped=24

### 2026-06-03 17:38:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --merge-only
```
- status: completed
- note: rows=288 grouped=22

### 2026-06-03 17:39:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py --check all --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21 --device cuda:0
```
- status: started
- note: 

### 2026-06-03 17:39:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py --check all
```
- status: completed
- note: S0_5_preflight_pass=1

### 2026-06-03 17:39:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_finalize.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: started
- note: 

### 2026-06-03 17:39:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_finalize.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: completed
- note: route=R3-WeakMLPSourceOnly-H3200Washout promotion_allowed=0

### 2026-06-03 20:03:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_source_retention_estimator_audit.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: started
- note: 

### 2026-06-03 20:03:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_source_retention_estimator_audit.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: completed
- note: rows=387 passing_predictors=1 decision=EstimatorUsableForNextDirection

### 2026-06-03 21:15:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_source_retention_estimator_audit.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: started
- note: 

### 2026-06-03 21:15:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_source_retention_estimator_audit.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: completed
- note: rows=387 passing_predictors=1 decision=EstimatorUsableForNextDirection

### 2026-06-03 21:18:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_source_retention_estimator_audit.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: started
- note: 

### 2026-06-03 21:18:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_source_retention_estimator_audit.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: completed
- note: rows=387 passing_predictors=1 decision=OnlyEarlySourceReadbackPredictsRetention_NoTrainOnlyDirectionSelector

### 2026-06-03 21:22:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_source_retention_estimator_audit.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: started
- note: 

### 2026-06-03 21:22:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_source_retention_estimator_audit.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: completed
- note: rows=387 stratification_rows=47 passing_predictors=1 train_only_passing=0 decision=OnlyEarlySourceReadbackPredictsRetention_NoTrainOnlyDirectionSelector

### 2026-06-03 21:38:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 3 --device cuda:3
```
- status: started
- note: jobs=15

### 2026-06-03 21:38:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 0 --device cuda:0
```
- status: started
- note: jobs=16

### 2026-06-03 21:38:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 2 --device cuda:2
```
- status: started
- note: jobs=16

### 2026-06-03 21:38:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 1 --device cuda:1
```
- status: started
- note: jobs=16

### 2026-06-03 21:41:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 3
```
- status: completed
- note: rows=15 traces=150

### 2026-06-03 21:41:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 2
```
- status: completed
- note: rows=16 traces=160

### 2026-06-03 21:41:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 0
```
- status: completed
- note: rows=16 traces=160

### 2026-06-03 21:41:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 1
```
- status: completed
- note: rows=16 traces=160

### 2026-06-03 21:42:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --merge-only
```
- status: completed
- note: rows=351 grouped=25

### 2026-06-03 21:45:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 3 --device cuda:3
```
- status: started
- note: jobs=13

### 2026-06-03 21:45:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 1 --device cuda:1
```
- status: started
- note: jobs=14

### 2026-06-03 21:45:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 2 --device cuda:2
```
- status: started
- note: jobs=13

### 2026-06-03 21:45:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 0 --device cuda:0
```
- status: started
- note: jobs=14

### 2026-06-03 21:48:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 3
```
- status: completed
- note: rows=13 traces=130

### 2026-06-03 21:48:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 2
```
- status: completed
- note: rows=13 traces=130

### 2026-06-03 21:48:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 1
```
- status: completed
- note: rows=14 traces=140

### 2026-06-03 21:48:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 0
```
- status: completed
- note: rows=14 traces=140

### 2026-06-03 21:48:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --merge-only
```
- status: completed
- note: rows=405 grouped=27

### 2026-06-03 22:20:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 3 --device cuda:3
```
- status: started
- note: jobs=13

### 2026-06-03 22:20:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 0 --device cuda:0
```
- status: started
- note: jobs=14

### 2026-06-03 22:20:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 1 --device cuda:1
```
- status: started
- note: jobs=14

### 2026-06-03 22:20:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 2 --device cuda:2
```
- status: started
- note: jobs=13

### 2026-06-03 22:23:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 3
```
- status: completed
- note: rows=13 traces=130

### 2026-06-03 22:23:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 1
```
- status: completed
- note: rows=14 traces=140

### 2026-06-03 22:23:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 2
```
- status: completed
- note: rows=13 traces=130

### 2026-06-03 22:23:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --shard-index 0
```
- status: completed
- note: rows=14 traces=140

### 2026-06-03 22:23:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_mlp_source_dynamics.py --scope kan_writer --merge-only
```
- status: completed
- note: rows=459 grouped=29

### 2026-06-03 22:26:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_source_retention_estimator_audit.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: started
- note: 

### 2026-06-03 22:26:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_source_retention_estimator_audit.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: completed
- note: rows=558 stratification_rows=54 passing_predictors=1 train_only_passing=0 decision=OnlyEarlySourceReadbackPredictsRetention_NoTrainOnlyDirectionSelector

### 2026-06-03 22:26:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py --check all --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21 --device cuda:0
```
- status: started
- note: 

### 2026-06-03 22:26:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py --check all
```
- status: completed
- note: S0_5_preflight_pass=1

### 2026-06-03 22:27:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_finalize.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```
- status: started
- note: 

## 关键文件

- plan: `/home/chengshun.wang/DG-LCA/docs/DG-KAN_v21.0_SourceRetention_KernelOfficialization_4GPU_完整计划.md`
- result_dir: `results/v21_0_source_retention_kernel_officialization_4gpu/official_v21`
- route: `results/v21_0_source_retention_kernel_officialization_4gpu/official_v21/v21_route_decision.json`
- packet: `results/v21_0_source_retention_kernel_officialization_4gpu/official_v21/v21_code_review_packet.zip`


## 2026-06-03 22:27:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_finalize.py --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```

- status: completed
- note: route=R3-WeakMLPSourceOnly-H3200Washout promotion_allowed=0


## 2026-06-03 22:35 F5-F8 复现补充

### 代码/配置修改

- `experiments/run_v21_source_retention_estimator_audit.py`：新增 F5 source-retention estimator 离线审计脚本。
- `experiments/run_v21_common.py`：新增 F6/F7/F8 KSW2 specs，并把 F5 audit 脚本与 artifacts 加入 packet。
- `experiments/run_v21_s05_truth_gate.py`：加入 F5 audit runner import/compile check。
- `experiments/run_v21_finalize.py`：加入 F5 rows/summary/decision/stratification required artifacts 与复盘输出。

### 关键复现命令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  experiments/run_v21_common.py \
  experiments/run_v21_kan_source_channel_writer.py \
  experiments/run_v21_source_retention_estimator_audit.py \
  experiments/run_v21_s05_truth_gate.py \
  experiments/run_v21_finalize.py
```

F6 KSW2 density full run（实际按 shard-index 0/1/2/3 对应 cuda:0/1/2/3 并行执行）：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_kan_source_channel_writer.py \
  --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21 \
  --device cuda:${SHARD} --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R4-k3-gradbuf-triton \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --steps 4800 \
  --shard-count 4 --shard-index ${SHARD} --run-label f6ksw2density \
  --spec-ids F6-KSW2-density-smallstep-alt50,F6-KSW2-density-alt100,F6-KSW2-density-smallstep-alt100,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

F7 KSW2 warmup full run：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_kan_source_channel_writer.py \
  --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21 \
  --device cuda:${SHARD} --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R4-k3-gradbuf-triton \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --steps 4800 \
  --shard-count 4 --shard-index ${SHARD} --run-label f7ksw2warm \
  --spec-ids F7-KSW2-warm400-smallstep-alt50,F7-KSW2-warm800-smallstep-alt50,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

F8 KSW2 early-boost full run：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_kan_source_channel_writer.py \
  --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21 \
  --device cuda:${SHARD} --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R4-k3-gradbuf-triton \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --steps 4800 \
  --shard-count 4 --shard-index ${SHARD} --run-label f8ksw2earlyboost \
  --spec-ids F8-KSW2-earlyboost-alt25,F8-KSW2-earlyboost-highstep-alt50,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

Merge / audit / truth gate / finalize：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_kan_source_channel_writer.py \
  --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21 \
  --scope kan_writer --merge-only

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_source_retention_estimator_audit.py \
  --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py \
  --check all --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21 --device cuda:0

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_finalize.py \
  --out-dir results/v21_0_source_retention_kernel_officialization_4gpu/official_v21
```

### 关键 artifact

- `results/v21_0_source_retention_kernel_officialization_4gpu/official_v21/v21_kan_source_writer_matrix.csv`
- `results/v21_0_source_retention_kernel_officialization_4gpu/official_v21/v21_kan_source_writer_raw_matrix.csv`
- `results/v21_0_source_retention_kernel_officialization_4gpu/official_v21/v21_f5_source_retention_estimator_summary.csv`
- `results/v21_0_source_retention_kernel_officialization_4gpu/official_v21/v21_f5_source_retention_estimator_decision.csv`
- `results/v21_0_source_retention_kernel_officialization_4gpu/official_v21/v21_f5_early_source_stratification.csv`
- `results/v21_0_source_retention_kernel_officialization_4gpu/official_v21/v21_route_decision.json`

## 2026-06-04 00:39:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0 --shard-count 4 --device cuda:0
```

- status: started
- note: jobs=5

## 2026-06-04 00:39:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 3 --shard-count 4 --device cuda:3
```

- status: started
- note: jobs=4

## 2026-06-04 00:39:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 1 --shard-count 4 --device cuda:1
```

- status: started
- note: jobs=5

## 2026-06-04 00:39:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 2 --shard-count 4 --device cuda:2
```

- status: started
- note: jobs=4

## 2026-06-04 00:39:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 2
```

- status: completed
- note: rows=4

## 2026-06-04 00:39:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 3
```

- status: completed
- note: rows=4

## 2026-06-04 00:39:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0
```

- status: completed
- note: rows=5

## 2026-06-04 00:39:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 1
```

- status: completed
- note: rows=5

## 2026-06-04 00:43:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --merge-only
```

- status: completed
- note: rows=18 waterfall=216 grad=18

## 2026-06-04 00:45:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 1 --shard-count 4 --device cuda:1
```

- status: started
- note: jobs=2

## 2026-06-04 00:45:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0 --shard-count 4 --device cuda:0
```

- status: started
- note: jobs=2

## 2026-06-04 00:45:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 2 --shard-count 4 --device cuda:2
```

- status: started
- note: jobs=1

## 2026-06-04 00:45:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 3 --shard-count 4 --device cuda:3
```

- status: started
- note: jobs=1

## 2026-06-04 00:45:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 2
```

- status: completed
- note: rows=1

## 2026-06-04 00:45:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 3
```

- status: completed
- note: rows=1

## 2026-06-04 00:45:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 1
```

- status: completed
- note: rows=2

## 2026-06-04 00:45:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0
```

- status: completed
- note: rows=2

## 2026-06-04 00:45:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --merge-only
```

- status: completed
- note: rows=6 waterfall=72 grad=6

## 2026-06-04 02:28:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 1 --shard-count 4 --device cuda:1
```

- status: started
- note: jobs=2

## 2026-06-04 02:28:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0 --shard-count 4 --device cuda:0
```

- status: started
- note: jobs=3

## 2026-06-04 02:28:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 2 --shard-count 4 --device cuda:2
```

- status: started
- note: jobs=2

## 2026-06-04 02:28:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 3 --shard-count 4 --device cuda:3
```

- status: started
- note: jobs=2

## 2026-06-04 02:28:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 3
```

- status: completed
- note: rows=2

## 2026-06-04 02:29:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 1
```

- status: completed
- note: rows=2

## 2026-06-04 02:29:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0
```

- status: completed
- note: rows=3

## 2026-06-04 02:29:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 2
```

- status: completed
- note: rows=2

## 2026-06-04 02:29:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --merge-only
```

- status: completed
- note: rows=9 waterfall=108 grad=9

## 2026-06-04 19:26:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0 --shard-count 1 --device cuda:3
```

- status: started
- note: jobs=4

## 2026-06-04 19:26:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0
```

- status: completed
- note: rows=4

## 2026-06-04 19:26:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --merge-only
```

- status: completed
- note: rows=4 waterfall=48 grad=4

## 2026-06-05 04:55:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py --check all --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02 --device cuda:0
```

- status: started

## 2026-06-05 04:55:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py --check all
```

- status: completed
- note: S0_5_preflight_pass=1

## 2026-06-05 05:13:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py --check all --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02 --device cuda:0
```

- status: started

## 2026-06-05 05:13:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py --check all
```

- status: completed
- note: S0_5_preflight_pass=1

## 2026-06-05 05:24:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py --check all --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02 --device cuda:0
```

- status: started

## 2026-06-05 05:24:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_s05_truth_gate.py --check all
```

- status: completed
- note: S0_5_preflight_pass=1

## 2026-06-06 08:26:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0 --shard-count 1 --device cuda:0
```

- status: started
- note: jobs=1

## 2026-06-06 08:26:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0
```

- status: completed
- note: rows=1

## 2026-06-06 08:27:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --merge-only
```

- status: completed
- note: rows=1 waterfall=12 grad=1

## 2026-06-06 08:27:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0 --shard-count 1 --device cuda:0
```

- status: started
- note: jobs=1

## 2026-06-06 08:28:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0
```

- status: completed
- note: rows=1

## 2026-06-06 08:28:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --merge-only
```

- status: completed
- note: rows=1 waterfall=12 grad=1

## 2026-06-06 08:29:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0 --shard-count 1 --device cuda:0
```

- status: started
- note: jobs=2

## 2026-06-06 08:29:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0
```

- status: completed
- note: rows=2

## 2026-06-06 08:30:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --merge-only
```

- status: completed
- note: rows=2 waterfall=24 grad=2

## 2026-06-06 08:31:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0 --shard-count 1 --device cuda:0
```

- status: started
- note: jobs=2

## 2026-06-06 08:31:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0
```

- status: completed
- note: rows=2

## 2026-06-06 08:31:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --merge-only
```

- status: completed
- note: rows=2 waterfall=24 grad=2
