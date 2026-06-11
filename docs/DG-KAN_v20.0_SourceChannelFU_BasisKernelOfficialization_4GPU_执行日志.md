# DG-KAN v20.0 Source-Channel FU + Basis Kernel Officialization 执行日志

生成时间：2026-06-03 12:40:01 +0800

## 命令日志

### 2026-06-03 11:47:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_s04_truth_gate.py --check import_closure --out-dir results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20 --device cuda:0
```
- status: started
- note: 

### 2026-06-03 11:47:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_s04_truth_gate.py --check import_closure
```
- status: completed
- note: S0_4_preflight_pass=1

### 2026-06-03 11:47:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_s04_truth_gate.py --check linec_golden --out-dir results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20 --device cuda:0
```
- status: started
- note: 

### 2026-06-03 11:47:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_s04_truth_gate.py --check linec_golden
```
- status: completed
- note: S0_4_preflight_pass=1

### 2026-06-03 11:47:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_s04_truth_gate.py --check debt_route --out-dir results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20 --device cuda:0
```
- status: started
- note: 

### 2026-06-03 11:47:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_s04_truth_gate.py --check debt_route
```
- status: completed
- note: S0_4_preflight_pass=1

### 2026-06-03 11:47:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_s04_truth_gate.py --check mechanism_semantics --out-dir results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20 --device cuda:0
```
- status: started
- note: 

### 2026-06-03 11:47:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_s04_truth_gate.py --check mechanism_semantics
```
- status: completed
- note: S0_4_preflight_pass=1

### 2026-06-03 11:47:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_s04_truth_gate.py --check efficiency_profiler --out-dir results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20 --device cuda:0
```
- status: started
- note: 

### 2026-06-03 11:47:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_s04_truth_gate.py --check efficiency_profiler
```
- status: completed
- note: S0_4_preflight_pass=1

### 2026-06-03 11:47:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_s04_truth_gate.py --check kernel_gradcheck --out-dir results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20 --device cuda:0
```
- status: started
- note: 

### 2026-06-03 11:47:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_s04_truth_gate.py --check kernel_gradcheck
```
- status: completed
- note: S0_4_preflight_pass=1

### 2026-06-03 11:48:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_basis_kernel_officialization.py --shard-index 2 --shard-count 4 --device cuda:2
```
- status: started
- note: jobs=9

### 2026-06-03 11:48:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_basis_kernel_officialization.py --shard-index 0 --shard-count 4 --device cuda:0
```
- status: started
- note: jobs=9

### 2026-06-03 11:48:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_basis_kernel_officialization.py --shard-index 3 --shard-count 4 --device cuda:3
```
- status: started
- note: jobs=9

### 2026-06-03 11:48:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_basis_kernel_officialization.py --shard-index 1 --shard-count 4 --device cuda:1
```
- status: started
- note: jobs=9

### 2026-06-03 11:48:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_basis_kernel_officialization.py --shard-index 0
```
- status: completed
- note: rows=9

### 2026-06-03 11:48:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_basis_kernel_officialization.py --shard-index 3
```
- status: completed
- note: rows=9

### 2026-06-03 11:48:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_basis_kernel_officialization.py --shard-index 1
```
- status: completed
- note: rows=9

### 2026-06-03 11:48:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_basis_kernel_officialization.py --shard-index 2
```
- status: completed
- note: rows=9

### 2026-06-03 11:56:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_basis_kernel_officialization.py --merge-only
```
- status: completed
- note: rows=36 waterfall=432 grad=36

### 2026-06-03 11:59:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_basis_kernel_officialization.py --merge-only
```
- status: completed
- note: rows=36 waterfall=432 grad=36

### 2026-06-03 12:00:35 +0800

```bash
touch v20_efficiency_gpu_monitor.stop
```
- status: completed
- note: released GPU monitor deadlock after four efficiency shards had written b0-b3 tables; merge-only then completed

### 2026-06-03 12:00:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_basis_kernel_officialization.py --merge-only
```
- status: completed
- note: recomputed S1 gate after audit_cost_separated training-loop overhead fix

### 2026-06-03 12:01:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2 --device cuda:2
```
- status: started
- note: jobs=22

### 2026-06-03 12:01:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1 --device cuda:1
```
- status: started
- note: jobs=23

### 2026-06-03 12:01:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3 --device cuda:3
```
- status: started
- note: jobs=22

### 2026-06-03 12:01:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```
- status: started
- note: jobs=23

### 2026-06-03 12:03:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3
```
- status: completed
- note: rows=22 traces=220

### 2026-06-03 12:03:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1
```
- status: completed
- note: rows=23 traces=230

### 2026-06-03 12:03:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```
- status: completed
- note: rows=23 traces=230

### 2026-06-03 12:03:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2
```
- status: completed
- note: rows=22 traces=220

### 2026-06-03 12:04:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```
- status: completed
- note: rows=90 grouped=10

### 2026-06-03 12:10:07 +0800

```bash
experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --init-seed-offset {100000,200000,300000} --spec-ids F1-MLP-M16-replay-exact,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```
- status: completed
- note: x3 independent offset rerun stored in mlp_m16_offsets/ and v20_mlp_m16_independent_offsets.csv

### 2026-06-03 12:11:53 +0800

```bash
experiments/run_v20_kan_source_channel_writer.py
```
- status: blocked_then_fixed
- note: initial direct wrapper failed with ModuleNotFoundError: experiments; added repo-root sys.path guard and py_compile passed

### 2026-06-03 12:12:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope kan_writer --shard-index 1 --device cuda:1
```
- status: started
- note: jobs=41

### 2026-06-03 12:12:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope kan_writer --shard-index 3 --device cuda:3
```
- status: started
- note: jobs=40

### 2026-06-03 12:12:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope kan_writer --shard-index 2 --device cuda:2
```
- status: started
- note: jobs=40

### 2026-06-03 12:12:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope kan_writer --shard-index 0 --device cuda:0
```
- status: started
- note: jobs=41

### 2026-06-03 12:35:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope kan_writer --shard-index 3
```
- status: completed
- note: rows=40 traces=400

### 2026-06-03 12:35:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope kan_writer --shard-index 2
```
- status: completed
- note: rows=40 traces=400

### 2026-06-03 12:38:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope kan_writer --shard-index 1
```
- status: completed
- note: rows=41 traces=410

### 2026-06-03 12:38:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope kan_writer --shard-index 0
```
- status: completed
- note: rows=41 traces=410

### 2026-06-03 12:38:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope kan_writer --merge-only
```
- status: completed
- note: rows=162 grouped=18

### 2026-06-03 12:39:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_function_space_actuation.py --out-dir results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20
```
- status: started
- note: 

### 2026-06-03 12:39:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_function_space_actuation.py
```
- status: completed
- note: actuation_rows=1191 summary=16

### 2026-06-03 12:40:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_merge_finalize.py --out-dir results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20
```
- status: started
- note: 

## 关键文件

- plan: `/home/chengshun.wang/DG-LCA/docs/DG-KAN_v20.0_SourceChannelFU_BasisKernelOfficialization_4GPU_完整计划.md`
- result_dir: `results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20`
- route: `results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/v20_route_decision.json`
- packet: `results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/v20_code_review_packet.zip`


## 2026-06-03 12:40:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_merge_finalize.py --out-dir results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20
```

- status: completed
- note: route=R-A1-MLPGenericSourceNotReproducible promotion_allowed=0

## 2026-06-03 12:45:00 +0800 审计补充

### 关键代码修改

```text
dgkan/profiling/efficiency_v20.py
  - audit_cost_separated=1 时，S1 使用 audit_overhead_ratio_training_loop=0；
  - 原始 audit readback 保留为 audit_overhead_ratio_raw。

experiments/run_v20_basis_kernel_officialization.py
  - merge-only 重新计算 v20_officialish_gate。

experiments/run_v20_kan_source_channel_writer.py
  - 添加 repo-root sys.path guard，修复直接运行 wrapper 的 ModuleNotFoundError。
```

验证：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/profiling/efficiency_v20.py \
  experiments/run_v20_basis_kernel_officialization.py \
  experiments/run_v20_kan_source_channel_writer.py
```

### 4GPU efficiency wrapper

执行目标：D-CHE/D-FOU/LQ/RAT/RBF/WAV basis officialization profiling，batch 8/32/128/256，4GPU shards。

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20
SNAP=$OUT/v20_efficiency_gpu_runtime_snapshots.csv
STOP=$OUT/v20_efficiency_gpu_monitor.stop

for IDX in 0 1 2 3; do
  $KAN experiments/run_v20_basis_kernel_officialization.py \
    --out-dir "$OUT" --device cuda:$IDX --data-root data \
    --shard-count 4 --shard-index $IDX \
    --batches 8,32,128,256 --profiler-repeats 2 --profiler-warmup 1 &
done
```

执行 blocker/fix：

```text
四个 shard 在 2026-06-03 11:48:26 +0800 已写完 b0-b3 CSV。
原 wrapper 的 wait 同时等待 monitor，monitor 等 STOP 文件，形成死锁。
手动执行 touch v20_efficiency_gpu_monitor.stop 释放 monitor。
随后运行 merge-only，生成 v20_efficiency_truth_table.csv / waterfall / gradcheck。
```

gate 修复后重新聚合：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python \
  experiments/run_v20_basis_kernel_officialization.py \
  --out-dir results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20 \
  --merge-only
```

### MLP M16 independent offsets

执行目标：计划要求 M16 replay 失败时 x3 independent offsets。

```bash
for OFF in 100000 200000 300000; do
  OUT=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/mlp_m16_offsets/off$OFF
  for IDX in 0 1 2 3; do
    /home/chengshun.wang/miniconda3/envs/kan/bin/python \
      experiments/run_v20_mlp_retained_source_anatomy.py \
      --out-dir "$OUT" --scope mlp_anatomy --device cuda:$IDX --data-root data \
      --shard-count 4 --shard-index $IDX --steps 4800 \
      --run-label v20mlp_off$OFF --init-seed-offset $OFF \
      --spec-ids F1-MLP-M16-replay-exact,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead &
  done
  wait
  /home/chengshun.wang/miniconda3/envs/kan/bin/python \
    experiments/run_v20_mlp_retained_source_anatomy.py \
    --out-dir "$OUT" --scope mlp_anatomy --merge-only
done
```

汇总文件：

```text
results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/v20_mlp_m16_independent_offsets.csv
```

### KAN source writer full run

首次直接运行 wrapper 失败：

```text
ModuleNotFoundError: No module named 'experiments'
```

修复 `experiments/run_v20_kan_source_channel_writer.py` 后重跑：

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20

for IDX in 0 1 2 3; do
  $KAN experiments/run_v20_kan_source_channel_writer.py \
    --out-dir "$OUT" --scope kan_writer --device cuda:$IDX --data-root data \
    --carriers D-CHE,D-FOU \
    --basis-repair-variant CHE-R2-low-degree-k3-triton \
    --basis-repair-variant-fou FOU-R4-k4-triton-no-materialize \
    --shard-count 4 --shard-index $IDX --steps 4800 --run-label v20kan &
done
wait

$KAN experiments/run_v20_kan_source_channel_writer.py \
  --out-dir "$OUT" --scope kan_writer --merge-only
```

产物：

```text
v20_kan_source_writer_raw_matrix.csv rows=162
v20_kan_source_writer_matrix.csv grouped=18
v20_kan_source_writer_traces.csv rows=1620
v20_kan_source_writer_gpu_runtime_snapshots.csv rows=1233
```

运行备注：`M13-LowRankMatrixBlockFU` 期间 PyTorch 报一次 SVD fallback warning；run 未失败，rows 已落表。

### finalizer / artifact check

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python \
  experiments/run_v20_function_space_actuation.py \
  --out-dir results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20

/home/chengshun.wang/miniconda3/envs/kan/bin/python \
  experiments/run_v20_merge_finalize.py \
  --out-dir results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20
```

最终校验：

```text
v20_required_artifact_manifest.csv rows=30 missing=0
v20_code_review_packet.zip size=461K
v20_results_bundle.zip size=2.0M
v20_route_decision.json route=R-A1-MLPGenericSourceNotReproducible
promotion_allowed=0
```

## 2026-06-03 12:51:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=6

## 2026-06-03 12:52:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=6 traces=42

## 2026-06-03 12:52:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=6 grouped=6

## 2026-06-03 13:20 Case A continuation / repair audit

### 代码修改与语法检查

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v20_common.py \
  experiments/run_v20_mlp_retained_source_anatomy.py
```

- status: completed
- note: syntax gate passed after Case A changes

修改文件：

```text
dgkan/fu/mechanisms.py
experiments/run_v17_common.py
experiments/run_v20_common.py
```

修改摘要：

- 给 `M14-SourceChannelPopRiskSlowFU` / `M8-PopRiskSNRFU` 增加 PopRisk/SNR diagnostics。
- 修复 `M11/M12/M14` 这类 slow-state writer 在 `train_one()` 默认分支没有传入 `slow_state` 的 blocker；新增 `M33/M34/M35` 也走同一 slow-state readback。
- 新增 `M33-PopRiskMatrixBlockSlowFU`：PopRisk/SNR slow-state + matrix-block projection。
- 新增 `M34-ExactPopRiskSlowFU`：micro-split K=32 + exact/unbiased variance PopRisk slow-state。
- 新增 `M35-ExactPopRiskMatrixBlockSlowFU`：K=32 exact variance + matrix-block projection。
- 新增 `M36-MomentumWarmReadoutBlockFU`：M2-style momentum warmup 后切到 readout-block pulse，用于修复 F5b h1600-only positive 断链。
- `experiments/run_v20_common.py` 新增 Case A F3/F5 repair specs，全部可用 `--spec-ids` 精确复现。

### Case A K16/block full 4GPU run

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu
ROOTOUT=$BASE/official_v20
OUT=$ROOTOUT/mlp_case_a_source_discovery
SNAP=$ROOTOUT/v20_case_a_mlp_source_discovery_gpu_runtime_snapshots.csv
STOP=$ROOTOUT/v20_case_a_mlp_source_discovery_gpu_monitor.stop
SPEC_IDS=F3a-MLP-PopRiskSlowState-fu0p0005,F3a2-MLP-PopRiskSlowState-fu0p0001,F3b-MLP-PopRiskMatrixBlock-fu0p0005,F3b2-MLP-PopRiskMatrixBlock-fu0p0001,F5a-MLP-LowRankMatrixBlock-fu0p0005,F5a2-MLP-ExplicitMatrixBlock-fu0p0005,F5b-MLP-OutputReadoutBlock-fu0p0001,F1d-random-same-norm-source-state,F1f-AdamW-primary-only,CTRL-NoOpMatchedOverhead,CTRL-SGD

for IDX in 0 1 2 3; do
  $KAN experiments/run_v20_mlp_retained_source_anatomy.py \
    --out-dir "$OUT" --scope mlp_anatomy --device cuda:$IDX --data-root data \
    --shard-count 4 --shard-index $IDX --steps 4800 \
    --run-label v20casea --spec-ids "$SPEC_IDS" &
done
wait

$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" --scope mlp_anatomy --merge-only
```

- status: completed
- shard rows: 25/25/25/24
- merged rows/grouped/traces: 99/11/990
- GPU snapshots: `v20_case_a_mlp_source_discovery_gpu_runtime_snapshots.csv` rows=800, max_mem=759 MB, max_util=35%
- canonical artifacts:
  - `v20_case_a_mlp_source_discovery_summary.csv`
  - `v20_case_a_mlp_source_discovery_matrix.csv`
  - `v20_case_a_mlp_source_discovery_raw_matrix.csv`
  - `v20_case_a_mlp_source_discovery_traces.csv`

### Case A exact-K32 PopRisk repair smoke

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/mlp_case_a_exactk32_smoke

$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" --scope mlp_anatomy --device cuda:0 --data-root data \
  --datasets MNIST --seeds 0 --steps 1600 --shard-count 1 --shard-index 0 \
  --run-label v20casea_exactk32_smoke \
  --spec-ids F3r1-MLP-ExactPopRiskK32-fu0p0005,F3r3-MLP-ExactPopRiskK32Block-fu0p0005,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" --scope mlp_anatomy --merge-only
```

- status: completed
- rows/grouped: 6/6
- canonical artifacts:
  - `v20_case_a_exactk32_smoke_summary.csv`
  - `v20_case_a_exactk32_smoke_matrix.csv`
  - `v20_case_a_exactk32_smoke_traces.csv`

### Case A M36 momentum-warm readout repair smoke

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/mlp_case_a_m36_smoke

$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" --scope mlp_anatomy --device cuda:0 --data-root data \
  --datasets MNIST --seeds 0 --steps 1600 --shard-count 1 --shard-index 0 \
  --run-label v20casea_m36_smoke \
  --spec-ids F5r1-MLP-MomentumWarmReadoutBlock-alt50-fu0p0001,F5r2-MLP-MomentumWarmReadoutBlock-alt100-fu0p00005,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" --scope mlp_anatomy --merge-only
```

- status: completed
- rows/grouped: 6/6
- canonical artifacts:
  - `v20_case_a_m36_smoke_summary.csv`
  - `v20_case_a_m36_smoke_matrix.csv`
  - `v20_case_a_m36_smoke_traces.csv`

### route / artifact consolidation

```bash
python - <<'PY'
# consolidate Case A canonical artifacts, compute PopRisk Spearman,
# update v20_route_decision.json, append manifest rows, and build
# v20_case_a_continuation_packet.zip
PY
```

- status: completed
- route: `R-A1-MLPGenericSourceNotReproducible-CaseAContinuationNoRetainedSource`
- decision artifact: `v20_case_a_continuation_decision.csv`
- rollup artifact: `v20_case_a_continuation_rollup.csv`
- packet: `v20_case_a_continuation_packet.zip` size=265261 bytes before docs append

## 2026-06-03 12:52:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=25

## 2026-06-03 12:52:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1 --device cuda:1
```

- status: started
- note: jobs=25

## 2026-06-03 12:52:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3 --device cuda:3
```

- status: started
- note: jobs=24

## 2026-06-03 12:52:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2 --device cuda:2
```

- status: started
- note: jobs=25

## 2026-06-03 13:07:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=25 traces=250

## 2026-06-03 13:09:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3
```

- status: completed
- note: rows=24 traces=240

## 2026-06-03 13:09:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1
```

- status: completed
- note: rows=25 traces=250

## 2026-06-03 13:09:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2
```

- status: completed
- note: rows=25 traces=250

## 2026-06-03 13:09:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=99 grouped=11

## 2026-06-03 13:15:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=6

## 2026-06-03 13:17:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=6 traces=42

## 2026-06-03 13:17:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=6 grouped=6

## 2026-06-03 13:19:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=6

## 2026-06-03 13:19:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=6 traces=42

## 2026-06-03 13:19:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=6 grouped=6

## 2026-06-03 13:31-13:55 +0800 F7 Split-Fisher source-observability continuation

### 代码修改 / 编译

本轮是在 Case A/F3-F5 失败后继续推进的新 source theory，不是继续调 M14/M33/M36：

- 修改 `dgkan/fu/mechanisms.py`
  - 新增 `M37-SplitFisherAgreementSlowFU`
  - 新增 `M38-AdamWSplitFisherAgreementResidualFU`
  - 新增 `M39-MomentumWarmSplitFisherFU`
  - 增加 mechanism semantic contract rows。
- 修改 `experiments/run_v17_common.py`
  - 新增 train-split Fisher/diagonal-whitened agreement source-state 分支。
  - source 只使用 train split A/B 梯度一致性；corrupted-label train batch 只做坏方向 rejection。
  - trace 增加 `source_state_balance_mean`、`source_state_current_cos`、`source_state_whitened_norm`。
  - 修复 M39 warmup 语义：warmup 期间每 step 执行 M2-style update，而不是只在 `alt_period` step 执行。
  - 修复 M39/M16/M36 momentum warmup 语义：primary optimizer 初始化为 `sgd-momentum`。
- 修改 `experiments/run_v20_common.py`
  - 新增 F7/M37-M39 specs。

编译命令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v20_common.py \
  experiments/run_v20_mlp_retained_source_anatomy.py
```

- status: passed

### F7 M37/M38 initial smoke

```bash
OUT=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/mlp_case_a_f7_fisher_smoke
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python

$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" \
  --scope mlp_anatomy \
  --device cuda:0 \
  --data-root data \
  --carriers MLP \
  --datasets MNIST \
  --seeds 0 \
  --steps 1600 \
  --shard-count 1 \
  --shard-index 0 \
  --run-label v20casea_f7_smoke \
  --spec-ids F7a-MLP-SplitFisherAgreement-alt50-fu0p0001,F7b-MLP-SplitFisherAgreement-alt100-fu0p00005,F7c-MLP-AdamWSplitFisherResidual-alt100-fu0p00005,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" \
  --scope mlp_anatomy \
  --merge-only
```

- status: completed
- rows/grouped/traces: 7/7/49
- GPU snapshots: 16 rows, max_mem=693 MB, max_util=9%
- decision: no full escalation; M37 was h1600-only positive, h800 negative.

### M39 sparse-warmup bug smoke/full

```bash
OUT=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/mlp_case_a_f7_m39_smoke
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python

$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" --scope mlp_anatomy --device cuda:0 \
  --data-root data --carriers MLP --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label v20casea_f7_m39_smoke \
  --spec-ids F7r1-MLP-MomentumWarmSplitFisher-warm400-alt50-fu0p0001,F7r2-MLP-MomentumWarmSplitFisher-warm800-alt50-fu0p0001,F7b-MLP-SplitFisherAgreement-alt100-fu0p00005,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

$KAN experiments/run_v20_mlp_retained_source_anatomy.py --out-dir "$OUT" --scope mlp_anatomy --merge-only
```

- status: completed
- note: warm800 smoke h800/h1600 was positive, but subsequent audit found M39 warmup was incorrectly sparse because it was guarded by `alt_period`.

Bugged full run was executed before the semantic issue was found:

```bash
OUT=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/mlp_case_a_f7_m39_full
SPEC_IDS=F7r2-MLP-MomentumWarmSplitFisher-warm800-alt50-fu0p0001,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$SHARD $KAN experiments/run_v20_mlp_retained_source_anatomy.py \
    --out-dir "$OUT" --scope mlp_anatomy --device cuda:0 \
    --data-root data --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST \
    --seeds 0,1,2 --steps 4800 --shard-count 4 --shard-index $SHARD \
    --run-label v20casea_f7_m39_full --spec-ids "$SPEC_IDS" &
done
wait
$KAN experiments/run_v20_mlp_retained_source_anatomy.py --out-dir "$OUT" --scope mlp_anatomy --merge-only
```

- status: completed but not used for final decision
- reason: M39 warmup semantic bug; artifact retained for audit only.

### M39 fixed warmup and momentum-primary repair smokes

After fixing warmup to run every step:

```bash
OUT=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/mlp_case_a_f7_m39_fixed_smoke
$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" --scope mlp_anatomy --device cuda:0 \
  --data-root data --carriers MLP --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label v20casea_f7_m39_fixed_smoke \
  --spec-ids F7r2-MLP-MomentumWarmSplitFisher-warm800-alt50-fu0p0001,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
$KAN experiments/run_v20_mlp_retained_source_anatomy.py --out-dir "$OUT" --scope mlp_anatomy --merge-only
```

- status: completed
- result: h800=-0.1283029317855835, h1600=-0.12300193309783936; no escalation.

After fixing warmup primary optimizer to `sgd-momentum`:

```bash
OUT=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/mlp_case_a_f7_m39_momentum_smoke
$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" --scope mlp_anatomy --device cuda:0 \
  --data-root data --carriers MLP --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label v20casea_f7_m39_momentum_smoke \
  --spec-ids F7r2-MLP-MomentumWarmSplitFisher-warm800-alt50-fu0p0001,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
$KAN experiments/run_v20_mlp_retained_source_anatomy.py --out-dir "$OUT" --scope mlp_anatomy --merge-only
```

- status: completed
- result: h800=0.19289636611938477, h1600=-0.39523887634277344; transition mismatch remained.

### M39 transition smoke and full 9-row

```bash
OUT=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/mlp_case_a_f7_m39_transition_smoke
$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" --scope mlp_anatomy --device cuda:0 \
  --data-root data --carriers MLP --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label v20casea_f7_m39_transition_smoke \
  --spec-ids F7r1-MLP-MomentumWarmSplitFisher-warm400-alt50-fu0p0001,F7r2-MLP-MomentumWarmSplitFisher-warm800-alt50-fu0p0001,F7r3-MLP-MomentumWarmSplitFisher-warm1200-alt50-fu0p0001,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
$KAN experiments/run_v20_mlp_retained_source_anatomy.py --out-dir "$OUT" --scope mlp_anatomy --merge-only
```

- status: completed
- decision: warm800 and warm1200 h800/h1600 were positive on MNIST seed0, so they were escalated.

```bash
OUT=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/mlp_case_a_f7_m39_transition_full
SPEC_IDS=F7r2-MLP-MomentumWarmSplitFisher-warm800-alt50-fu0p0001,F7r3-MLP-MomentumWarmSplitFisher-warm1200-alt50-fu0p0001,F1b-remove-alternation,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$SHARD $KAN experiments/run_v20_mlp_retained_source_anatomy.py \
    --out-dir "$OUT" --scope mlp_anatomy --device cuda:0 \
    --data-root data --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST \
    --seeds 0,1,2 --steps 4800 --shard-count 4 --shard-index $SHARD \
    --run-label v20casea_f7_m39_transition_full --spec-ids "$SPEC_IDS" &
done
wait
$KAN experiments/run_v20_mlp_retained_source_anatomy.py --out-dir "$OUT" --scope mlp_anatomy --merge-only
```

- status: completed
- rows/grouped/traces: 63/7/630
- blocked rows: 0
- GPU snapshots: 92 rows, max_mem=691 MB, max_util=10%
- decision: no retained candidate; final route updated to `R-A1-MLPGenericSourceNotReproducible-CaseAContinuationNoRetainedSource-F7SplitFisherNoRetained`.

### artifact consolidation

```bash
python - <<'PY'
# copy F7 smoke/full outputs to canonical official_v20 artifacts,
# write v20_case_a_f7_split_fisher_decision.csv/md,
# update v20_route_decision.json and v20_required_artifact_manifest.csv
PY
```

- status: completed
- canonical decision artifact: `v20_case_a_f7_split_fisher_decision.csv`
- canonical rollup artifact: `v20_case_a_f7_split_fisher_rollup.csv`

## 2026-06-03 13:31:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=7

## 2026-06-03 13:32:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=7 traces=49

## 2026-06-03 13:32:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-03 13:35:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=7

## 2026-06-03 13:35:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=7 traces=49

## 2026-06-03 13:36:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-03 13:37:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3 --device cuda:0
```

- status: started
- note: jobs=11

## 2026-06-03 13:37:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=12

## 2026-06-03 13:37:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1 --device cuda:0
```

- status: started
- note: jobs=11

## 2026-06-03 13:37:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2 --device cuda:0
```

- status: started
- note: jobs=11

## 2026-06-03 13:38:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-03 13:38:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-03 13:38:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-03 13:38:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-03 13:38:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-03 13:40:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=5

## 2026-06-03 13:40:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=5 traces=35

## 2026-06-03 13:41:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-03 13:42:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=5

## 2026-06-03 13:43:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=5 traces=35

## 2026-06-03 13:43:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-03 13:44:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=7

## 2026-06-03 13:44:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=7 traces=49

## 2026-06-03 13:45:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-03 13:45:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3 --device cuda:0
```

- status: started
- note: jobs=15

## 2026-06-03 13:45:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1 --device cuda:0
```

- status: started
- note: jobs=16

## 2026-06-03 13:45:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2 --device cuda:0
```

- status: started
- note: jobs=16

## 2026-06-03 13:45:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=16

## 2026-06-03 13:47:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3
```

- status: completed
- note: rows=15 traces=150

## 2026-06-03 13:47:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2
```

- status: completed
- note: rows=16 traces=160

## 2026-06-03 13:47:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1
```

- status: completed
- note: rows=16 traces=160

## 2026-06-03 13:47:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=16 traces=160

## 2026-06-03 13:47:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-03 13:57:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v20_common.py \
  experiments/run_v20_mlp_retained_source_anatomy.py
```

- status: completed
- note: py_compile passed

## 2026-06-03 13:57:02 +0800

```bash
python - <<'PY'
from pathlib import Path
import csv, json, zipfile

base = Path('results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20')
packet = base / 'v20_case_a_f7_split_fisher_packet.zip'
manifest = base / 'v20_required_artifact_manifest.csv'
rows = list(csv.DictReader(manifest.open()))
missing = []
for row in rows:
    name = row.get('artifact') or row.get('path') or row.get('file') or row.get('artifact_name')
    required = (row.get('required') or '1').strip()
    if name and required not in {'0', 'false', 'False'} and not (base / name).exists() and not Path(name).exists():
        missing.append(name)
route = json.loads((base / 'v20_route_decision.json').read_text())
with zipfile.ZipFile(packet) as zf:
    names = zf.namelist()
print('packet_exists', packet.exists())
print('packet_size', packet.stat().st_size if packet.exists() else '')
print('packet_files', len(names))
print('contains_route', 'official_v20/v20_route_decision.json' in names)
print('manifest_rows', len(rows))
print('required_missing', len(missing))
print('route', route.get('route'))
print('promotion_allowed', route.get('promotion_allowed'))
print('official_success_reached', route.get('official_success_reached'))
print('f7_retained', route.get('f7_m39_transition_full_retained_candidates'))
PY
```

- status: completed
- note: packet_exists=True; packet_size=224144 before final log append; packet_files=22; contains_route=True; manifest_rows=64; required_missing=0; route=R-A1-MLPGenericSourceNotReproducible-CaseAContinuationNoRetainedSource-F7SplitFisherNoRetained; promotion_allowed=0; official_success_reached=0; f7_retained=0

## 2026-06-03 13:57:02 +0800

```bash
ps -ef | rg 'run_v20_|v20_case_a_f7|gpu_monitor|nvidia-smi' || true
```

- status: completed
- note: no active v20 training or gpu monitor process found; output only contained the transient nvidia-smi query and the rg/check command itself

## 2026-06-03 13:57:02 +0800

```bash
python - <<'PY'
from pathlib import Path
import zipfile

root = Path('.')
base = Path('results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20')
packet = base / 'v20_case_a_f7_split_fisher_packet.zip'
members = [
    'dgkan/fu/mechanisms.py',
    'experiments/run_v17_common.py',
    'experiments/run_v20_common.py',
    'experiments/run_v20_mlp_retained_source_anatomy.py',
    'docs/DG-KAN_v20.0_SourceChannelFU_BasisKernelOfficialization_4GPU_完整计划.md',
    'docs/DG-KAN_v20.0_SourceChannelFU_BasisKernelOfficialization_4GPU_执行日志.md',
    'docs/DG-KAN_v20.0_SourceChannelFU_BasisKernelOfficialization_4GPU_实验结果复盘.md',
]
artifact_names = [
    'v20_route_decision.json',
    'v20_required_artifact_manifest.csv',
    'v20_case_a_f7_split_fisher_decision.csv',
    'v20_case_a_f7_split_fisher_rollup.csv',
    'v20_case_a_f7_split_fisher_decision.md',
    'v20_case_a_f7_fisher_smoke_summary.csv',
    'v20_case_a_f7_fisher_smoke_matrix.csv',
    'v20_case_a_f7_fisher_smoke_traces.csv',
    'v20_case_a_f7_m39_transition_smoke_summary.csv',
    'v20_case_a_f7_m39_transition_smoke_matrix.csv',
    'v20_case_a_f7_m39_transition_smoke_traces.csv',
    'v20_case_a_f7_m39_transition_full_summary.csv',
    'v20_case_a_f7_m39_transition_full_matrix.csv',
    'v20_case_a_f7_m39_transition_full_traces.csv',
    'v20_case_a_f7_m39_transition_full_gpu_runtime_snapshots.csv',
]
with zipfile.ZipFile(packet, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for member in members:
        p = root / member
        if p.exists():
            zf.write(p, member)
    for name in artifact_names:
        p = base / name
        if p.exists():
            zf.write(p, f'official_v20/{name}')
print(packet, packet.stat().st_size)
PY
```

- status: completed
- note: refreshed v20_case_a_f7_split_fisher_packet.zip after final doc append; packet contains source files, plan, execution log, recap log, route, manifest, and F7 artifacts

## 2026-06-03 14:07:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=7

## 2026-06-03 14:07:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=7 traces=49

## 2026-06-03 14:07:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-03 14:08:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2 --device cuda:2
```

- status: started
- note: jobs=16

## 2026-06-03 14:08:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=16

## 2026-06-03 14:08:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3 --device cuda:3
```

- status: started
- note: jobs=15

## 2026-06-03 14:08:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1 --device cuda:1
```

- status: started
- note: jobs=16

## 2026-06-03 14:10:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1
```

- status: completed
- note: rows=16 traces=160

## 2026-06-03 14:10:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2
```

- status: completed
- note: rows=16 traces=160

## 2026-06-03 14:10:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3
```

- status: completed
- note: rows=15 traces=150

## 2026-06-03 14:10:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=16 traces=160

## 2026-06-03 14:10:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-03 14:16:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=9

## 2026-06-03 14:17:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=9 traces=63

## 2026-06-03 14:17:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=9 grouped=9

## 2026-06-03 14:18:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2 --device cuda:2
```

- status: started
- note: jobs=13

## 2026-06-03 14:18:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3 --device cuda:3
```

- status: started
- note: jobs=13

## 2026-06-03 14:18:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1 --device cuda:1
```

- status: started
- note: jobs=14

## 2026-06-03 14:18:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=14

## 2026-06-03 14:19:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3
```

- status: completed
- note: rows=13 traces=130

## 2026-06-03 14:19:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2
```

- status: completed
- note: rows=13 traces=130

## 2026-06-03 14:19:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1
```

- status: completed
- note: rows=14 traces=140

## 2026-06-03 14:19:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=14 traces=140

## 2026-06-03 14:20:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=54 grouped=6

## 2026-06-03 14:34:25 +0800 F8-F10 Case A washout repair continuation

### compile / spec registration

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v20_common.py \
  experiments/run_v20_mlp_retained_source_anatomy.py
```

- status: completed
- note: M40/M41/M42/M43 specs registered and py_compile passed.

### F8 M40 anti-washout smoke

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20
OUT=$BASE/mlp_case_a_f8_antiwashout_smoke
$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" --scope mlp_anatomy \
  --device cuda:0 --data-root data \
  --carriers MLP --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label v20casea_f8_smoke \
  --spec-ids F8a-MLP-MomentumWarmAntiWashout-warm800,F8b-MLP-MomentumWarmAntiWashout-warm1200,F1b-remove-alternation,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
$KAN experiments/run_v20_mlp_retained_source_anatomy.py --out-dir "$OUT" --scope mlp_anatomy --merge-only
```

- status: completed
- result: F8a/F8b MNIST seed0 h800/h1600 both positive, upgraded to full.
- GPU snapshots: `mlp_case_a_f8_antiwashout_smoke/v20_case_a_f8_antiwashout_smoke_gpu_runtime_snapshots.csv`, rows=20, max_mem=693 MB, max_util=14%.

### F8 M40 anti-washout full

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20
OUT=$BASE/mlp_case_a_f8_antiwashout_full
for SHARD in 0 1 2 3; do
  DEV="cuda:$SHARD"
  $KAN experiments/run_v20_mlp_retained_source_anatomy.py \
    --out-dir "$OUT" --scope mlp_anatomy \
    --device "$DEV" --data-root data \
    --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --steps 4800 \
    --shard-count 4 --shard-index "$SHARD" --run-label v20casea_f8_full \
    --spec-ids F8a-MLP-MomentumWarmAntiWashout-warm800,F8b-MLP-MomentumWarmAntiWashout-warm1200,F1b-remove-alternation,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead &
done
wait
$KAN experiments/run_v20_mlp_retained_source_anatomy.py --out-dir "$OUT" --scope mlp_anatomy --merge-only
```

- status: completed
- result: F8b best h800/h1600/h3200/h4800 = `0.33461103174421525 / -0.043901529577043324 / -0.30157148838043213 / -0.34824442863464355`, retained=0.

### F8 M41 source-anchor smoke/full

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20
OUT=$BASE/mlp_case_a_f8_source_anchor_smoke
$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" --scope mlp_anatomy \
  --device cuda:0 --data-root data \
  --carriers MLP --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label v20casea_f8_anchor_smoke \
  --spec-ids F8c-MLP-MomentumWarmSourceAnchor-warm800,F8d-MLP-MomentumWarmSourceAnchor-warm1200,F8a-MLP-MomentumWarmAntiWashout-warm800,F8b-MLP-MomentumWarmAntiWashout-warm1200,F1b-remove-alternation,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
$KAN experiments/run_v20_mlp_retained_source_anatomy.py --out-dir "$OUT" --scope mlp_anatomy --merge-only

OUT=$BASE/mlp_case_a_f8_source_anchor_full
for SHARD in 0 1 2 3; do
  DEV="cuda:$SHARD"
  $KAN experiments/run_v20_mlp_retained_source_anatomy.py \
    --out-dir "$OUT" --scope mlp_anatomy \
    --device "$DEV" --data-root data \
    --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --steps 4800 \
    --shard-count 4 --shard-index "$SHARD" --run-label v20casea_f8_anchor_full \
    --spec-ids F8d-MLP-MomentumWarmSourceAnchor-warm1200,F1b-remove-alternation,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead &
done
wait
$KAN experiments/run_v20_mlp_retained_source_anatomy.py --out-dir "$OUT" --scope mlp_anatomy --merge-only
```

- status: completed
- result: F8d full h800/h1600/h3200/h4800 = `0.24928910202450222 / -0.19618584050072563 / -0.4315970738728841 / -0.39400559001498753`, retained=0.

### F9 M42 post-warmup hold smoke/full

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20
OUT=$BASE/mlp_case_a_f9_hold_smoke
$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" --scope mlp_anatomy \
  --device cuda:0 --data-root data \
  --carriers MLP --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label v20casea_f9_hold_smoke \
  --spec-ids F9a-MLP-MomentumWarmHold-warm800,F9b-MLP-MomentumWarmHold-warm1200,F8b-MLP-MomentumWarmAntiWashout-warm1200,F1b-remove-alternation,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
$KAN experiments/run_v20_mlp_retained_source_anatomy.py --out-dir "$OUT" --scope mlp_anatomy --merge-only

OUT=$BASE/mlp_case_a_f9_hold_full
for SHARD in 0 1 2 3; do
  DEV="cuda:$SHARD"
  $KAN experiments/run_v20_mlp_retained_source_anatomy.py \
    --out-dir "$OUT" --scope mlp_anatomy \
    --device "$DEV" --data-root data \
    --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --steps 4800 \
    --shard-count 4 --shard-index "$SHARD" --run-label v20casea_f9_hold_full \
    --spec-ids F9a-MLP-MomentumWarmHold-warm800,F1b-remove-alternation,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead &
done
wait
$KAN experiments/run_v20_mlp_retained_source_anatomy.py --out-dir "$OUT" --scope mlp_anatomy --merge-only
```

- status: completed
- result: F9a full h800/h1600/h3200/h4800 = `0.2418862051433987 / 0.02575813399420844 / -0.1102049085828993 / -0.07841628127627903`, retained=0.

### F10 M43 cyclic refresh-hold smoke/full

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20
OUT=$BASE/mlp_case_a_f10_cycle_hold_smoke
$KAN experiments/run_v20_mlp_retained_source_anatomy.py \
  --out-dir "$OUT" --scope mlp_anatomy \
  --device cuda:0 --data-root data \
  --carriers MLP --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label v20casea_f10_cycle_hold_smoke \
  --spec-ids F10a-MLP-MomentumCycleHold-warm800,F9a-MLP-MomentumWarmHold-warm800,F1b-remove-alternation,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
$KAN experiments/run_v20_mlp_retained_source_anatomy.py --out-dir "$OUT" --scope mlp_anatomy --merge-only

OUT=$BASE/mlp_case_a_f10_cycle_hold_full
for SHARD in 0 1 2 3; do
  DEV="cuda:$SHARD"
  $KAN experiments/run_v20_mlp_retained_source_anatomy.py \
    --out-dir "$OUT" --scope mlp_anatomy \
    --device "$DEV" --data-root data \
    --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --steps 4800 \
    --shard-count 4 --shard-index "$SHARD" --run-label v20casea_f10_cycle_hold_full \
    --spec-ids F10a-MLP-MomentumCycleHold-warm800,F1b-remove-alternation,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead &
done
wait
$KAN experiments/run_v20_mlp_retained_source_anatomy.py --out-dir "$OUT" --scope mlp_anatomy --merge-only
```

- status: completed
- result: F10a full h800/h1600/h3200/h4800 = `0.2562202347649468 / 0.040092163615756564 / -0.31356167793273926 / -0.358368886841668`, retained=0.

### artifact consolidation

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
# Copy F8-F10 smoke/full CSVs to official_v20 root, write
# v20_case_a_f8_f10_washout_repair_{rollup,diagnostics,decision}.csv/md,
# update v20_route_decision.json and v20_required_artifact_manifest.csv,
# and build v20_case_a_f8_f10_washout_repair_packet.zip.
PY
```

- status: completed
- rollup rows: 5
- diagnostics rows: 4
- manifest rows: 100
- required missing count: 0
- packet: `v20_case_a_f8_f10_washout_repair_packet.zip` size=460937 bytes
- route: `R-A1-MLPGenericSourceNotReproducible-CaseAContinuationNoRetainedSource-F7F10NoRetained`
- promotion_allowed: 0

### final verification

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v20_common.py \
  experiments/run_v20_mlp_retained_source_anatomy.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
# read v20_required_artifact_manifest.csv, v20_route_decision.json,
# F8-F10 rollup/diagnostics/decision rows, packet size, bundle size
PY

ps -eo pid,ppid,stat,cmd | rg 'run_v20|v20_.*gpu|nvidia-smi' || true
```

- status: completed
- py_compile: pass
- `v20_case_a_f8_f10_washout_repair_rollup.csv` rows=5
- `v20_case_a_f8_f10_washout_repair_diagnostics.csv` rows=4
- `v20_case_a_f8_f10_washout_repair_decision.csv` rows=1
- manifest rows=100, required missing=0
- route=`R-A1-MLPGenericSourceNotReproducible-CaseAContinuationNoRetainedSource-F7F10NoRetained`
- promotion_allowed=0, official_success_reached=0
- packet/bundle: refreshed; exact size read back by final verification command.
- process check: no continuing v20 training or GPU monitor; `ps` matched only the verification command itself.

## 2026-06-03 14:23:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=8

## 2026-06-03 14:23:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=8 traces=56

## 2026-06-03 14:23:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=8 grouped=8

## 2026-06-03 14:24:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=14

## 2026-06-03 14:24:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2 --device cuda:2
```

- status: started
- note: jobs=13

## 2026-06-03 14:24:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1 --device cuda:1
```

- status: started
- note: jobs=14

## 2026-06-03 14:24:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3 --device cuda:3
```

- status: started
- note: jobs=13

## 2026-06-03 14:25:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2
```

- status: completed
- note: rows=13 traces=130

## 2026-06-03 14:25:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3
```

- status: completed
- note: rows=13 traces=130

## 2026-06-03 14:25:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1
```

- status: completed
- note: rows=14 traces=140

## 2026-06-03 14:25:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=14 traces=140

## 2026-06-03 14:25:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=54 grouped=6

## 2026-06-03 14:28:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=7

## 2026-06-03 14:28:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=7 traces=49

## 2026-06-03 14:28:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-03 14:29:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0 --device cuda:0
```

- status: started
- note: jobs=14

## 2026-06-03 14:29:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3 --device cuda:3
```

- status: started
- note: jobs=13

## 2026-06-03 14:29:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2 --device cuda:2
```

- status: started
- note: jobs=13

## 2026-06-03 14:29:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1 --device cuda:1
```

- status: started
- note: jobs=14

## 2026-06-03 14:31:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 2
```

- status: completed
- note: rows=13 traces=130

## 2026-06-03 14:31:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 3
```

- status: completed
- note: rows=13 traces=130

## 2026-06-03 14:31:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 1
```

- status: completed
- note: rows=14 traces=140

## 2026-06-03 14:31:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --shard-index 0
```

- status: completed
- note: rows=14 traces=140

## 2026-06-03 14:31:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v20_mlp_retained_source_anatomy.py --scope mlp_anatomy --merge-only
```

- status: completed
- note: rows=54 grouped=6
