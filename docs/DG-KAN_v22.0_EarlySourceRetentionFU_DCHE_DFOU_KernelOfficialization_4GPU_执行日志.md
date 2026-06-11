# DG-KAN v22.0 EarlySourceRetentionFU DCHE/DFOU KernelOfficialization 4GPU 执行日志

生成时间：2026-06-04 10:03:34 +0800

记录原则：只记录真实执行过的命令、文件与状态；不把未执行内容写成结果。

## 2026-06-04 10:03:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/source_chain.py dgkan/profiling/efficiency_v22.py experiments/run_v22_common.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_efficiency_officialization.py experiments/run_v22_mlp_source_lab.py experiments/run_v22_kan_source_writer.py experiments/run_v22_function_space_target_reset.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: new v22 runner py_compile passed

## 2026-06-04 10:04:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --check all
```

- status: failed
- note: ImportError: run_linec_channel_golden_tests imported from wrong module; fixed run_v22_s07_truth_gate.py

## 2026-06-04 10:04:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 10:04:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 10:05:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --families D-CHE,D-FOU --batches 128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 3
```

- status: started

## 2026-06-04 10:05:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --families D-CHE,D-FOU --batches 128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 2
```

- status: started
- note: delegates to v21 efficiency profiler and applies v22 gates
- note: delegates to v21 efficiency profiler and applies v22 gates

## 2026-06-04 10:05:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --families D-CHE,D-FOU --batches 128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 0
```

- status: started
- note: delegates to v21 efficiency profiler and applies v22 gates

## 2026-06-04 10:05:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --families D-CHE,D-FOU --batches 128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 1
```

- status: started
- note: delegates to v21 efficiency profiler and applies v22 gates

## 2026-06-04 10:05:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0 --shard-count 4 --device cuda:0
```

- status: started
- note: jobs=3

## 2026-06-04 10:05:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 1 --shard-count 4 --device cuda:1
```

- status: started
- note: jobs=3

## 2026-06-04 10:05:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 3 --shard-count 4 --device cuda:3
```

- status: started
- note: jobs=3

## 2026-06-04 10:05:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 2 --shard-count 4 --device cuda:2
```

- status: started
- note: jobs=3

## 2026-06-04 10:05:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0
```

- status: completed
- note: rows=3

## 2026-06-04 10:05:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --families D-CHE,D-FOU --batches 128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 0
```

- status: completed
- note: raw v21 efficiency artifacts generated in v22 official dir

## 2026-06-04 10:05:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 1
```

- status: completed
- note: rows=3

## 2026-06-04 10:05:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --families D-CHE,D-FOU --batches 128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 1
```

- status: completed
- note: raw v21 efficiency artifacts generated in v22 official dir

## 2026-06-04 10:05:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 3
```

- status: completed
- note: rows=3

## 2026-06-04 10:05:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --families D-CHE,D-FOU --batches 128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 3
```

- status: completed
- note: raw v21 efficiency artifacts generated in v22 official dir

## 2026-06-04 10:05:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 2
```

- status: completed
- note: rows=3

## 2026-06-04 10:05:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --families D-CHE,D-FOU --batches 128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 2
```

- status: completed
- note: raw v21 efficiency artifacts generated in v22 official dir

## 2026-06-04 10:05:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --families D-CHE,D-FOU --batches 128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 1 --shard-index 0 --merge-only
```

- status: started
- note: delegates to v21 efficiency profiler and applies v22 gates

## 2026-06-04 10:05:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --merge-only
```

- status: completed
- note: rows=12 waterfall=144 grad=12

## 2026-06-04 10:05:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --families D-CHE,D-FOU --batches 128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 1 --shard-index 0 --merge-only
```

- status: completed
- note: raw v21 efficiency artifacts generated in v22 official dir

## 2026-06-04 10:05:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --merge-only
```

- status: completed
- note: v22 efficiency rows=12

## 2026-06-04 10:06:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --families D-CHE,D-FOU --batches 8,32,128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 1
```

- status: started
- note: delegates to v21 efficiency profiler and applies v22 gates

## 2026-06-04 10:06:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --families D-CHE,D-FOU --batches 8,32,128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 0
```

- status: started
- note: delegates to v21 efficiency profiler and applies v22 gates

## 2026-06-04 10:06:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --families D-CHE,D-FOU --batches 8,32,128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 3
```

- status: started
- note: delegates to v21 efficiency profiler and applies v22 gates

## 2026-06-04 10:06:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --families D-CHE,D-FOU --batches 8,32,128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 2
```

- status: started
- note: delegates to v21 efficiency profiler and applies v22 gates

## 2026-06-04 10:06:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 1 --shard-count 4 --device cuda:1
```

- status: started
- note: jobs=6

## 2026-06-04 10:06:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 3 --shard-count 4 --device cuda:3
```

- status: started
- note: jobs=6

## 2026-06-04 10:06:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0 --shard-count 4 --device cuda:0
```

- status: started
- note: jobs=6

## 2026-06-04 10:06:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 2 --shard-count 4 --device cuda:2
```

- status: started
- note: jobs=6

## 2026-06-04 10:06:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 3
```

- status: completed
- note: rows=6

## 2026-06-04 10:06:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --families D-CHE,D-FOU --batches 8,32,128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 3
```

- status: completed
- note: raw v21 efficiency artifacts generated in v22 official dir

## 2026-06-04 10:06:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 2
```

- status: completed
- note: rows=6

## 2026-06-04 10:06:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --families D-CHE,D-FOU --batches 8,32,128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 2
```

- status: completed
- note: raw v21 efficiency artifacts generated in v22 official dir

## 2026-06-04 10:06:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 0
```

- status: completed
- note: rows=6

## 2026-06-04 10:06:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --families D-CHE,D-FOU --batches 8,32,128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 0
```

- status: completed
- note: raw v21 efficiency artifacts generated in v22 official dir

## 2026-06-04 10:06:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --shard-index 1
```

- status: completed
- note: rows=6

## 2026-06-04 10:06:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --families D-CHE,D-FOU --batches 8,32,128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 4 --shard-index 1
```

- status: completed
- note: raw v21 efficiency artifacts generated in v22 official dir

## 2026-06-04 10:06:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --families D-CHE,D-FOU --batches 8,32,128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 1 --shard-index 0 --merge-only
```

- status: started
- note: delegates to v21 efficiency profiler and applies v22 gates

## 2026-06-04 10:06:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --merge-only
```

- status: completed
- note: rows=24 waterfall=288 grad=24

## 2026-06-04 10:06:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --families D-CHE,D-FOU --batches 8,32,128,256 --train-size 256 --val-size 64 --profiler-repeats 2 --profiler-warmup 1 --shard-count 1 --shard-index 0 --merge-only
```

- status: completed
- note: raw v21 efficiency artifacts generated in v22 official dir

## 2026-06-04 10:06:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_efficiency_officialization.py --merge-only
```

- status: completed
- note: v22 efficiency rows=24

## 2026-06-04 10:08:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c2c3_target_source_smoke --shard-count 4 --shard-index 1 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:08:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c2c3_target_source_smoke --shard-count 4 --shard-index 3 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:08:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c2c3_target_source_smoke --shard-count 4 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:08:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c2c3_target_source_smoke --shard-count 4 --shard-index 2 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:08:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:2 --steps 1600
```

- status: started
- note: jobs=63

## 2026-06-04 10:08:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:1 --steps 1600
```

- status: started
- note: jobs=63

## 2026-06-04 10:08:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=63

## 2026-06-04 10:08:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:3 --steps 1600
```

- status: started
- note: jobs=63

## 2026-06-04 10:13:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=63 traces=441

## 2026-06-04 10:13:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c2c3_target_source_smoke --shard-count 4 --shard-index 3 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:13:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=63 traces=441

## 2026-06-04 10:13:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c2c3_target_source_smoke --shard-count 4 --shard-index 2 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:13:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=63 traces=441

## 2026-06-04 10:13:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c2c3_target_source_smoke --shard-count 4 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:13:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=63 traces=441

## 2026-06-04 10:13:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c2c3_target_source_smoke --shard-count 4 --shard-index 1 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:14:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c2c3_target_source_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:14:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=252 grouped=28

## 2026-06-04 10:14:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c2c3_target_source_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:14:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=28 decision=NoEarlySourceChain

## 2026-06-04 10:14:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c3_kan_writer_smoke --shard-count 4 --shard-index 0 --spec-ids KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=kan; specs=KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:14:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c3_kan_writer_smoke --shard-count 4 --shard-index 2 --spec-ids KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=kan; specs=KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:14:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c3_kan_writer_smoke --shard-count 4 --shard-index 3 --spec-ids KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=kan; specs=KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:14:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c3_kan_writer_smoke --shard-count 4 --shard-index 1 --spec-ids KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=kan; specs=KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:15:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=32

## 2026-06-04 10:15:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2 --device cuda:2 --steps 1600
```

- status: started
- note: jobs=31

## 2026-06-04 10:15:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3 --device cuda:3 --steps 1600
```

- status: started
- note: jobs=31

## 2026-06-04 10:15:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1 --device cuda:1 --steps 1600
```

- status: started
- note: jobs=32

## 2026-06-04 10:17:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2
```

- status: completed
- note: rows=31 traces=217

## 2026-06-04 10:17:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c3_kan_writer_smoke --shard-count 4 --shard-index 2 --spec-ids KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:17:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3
```

- status: completed
- note: rows=31 traces=217

## 2026-06-04 10:17:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c3_kan_writer_smoke --shard-count 4 --shard-index 3 --spec-ids KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:17:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=32 traces=224

## 2026-06-04 10:17:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c3_kan_writer_smoke --shard-count 4 --shard-index 0 --spec-ids KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:17:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1
```

- status: completed
- note: rows=32 traces=224

## 2026-06-04 10:17:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c3_kan_writer_smoke --shard-count 4 --shard-index 1 --spec-ids KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:18:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c3_kan_writer_smoke --shard-count 1 --shard-index 0 --spec-ids KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=kan; specs=KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:18:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=378 grouped=34

## 2026-06-04 10:18:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c3_kan_writer_smoke --shard-count 1 --shard-index 0 --spec-ids KSW2-lowdegree-lowfreq-source-bank,F6-KSW2-density-smallstep-alt50,F8-KSW2-earlyboost-alt25,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:18:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=34 decision=NoEarlySourceChain

## 2026-06-04 10:19:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c1_mlp_lab_smoke --shard-count 4 --shard-index 3 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:19:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c1_mlp_lab_smoke --shard-count 4 --shard-index 1 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:19:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c1_mlp_lab_smoke --shard-count 4 --shard-index 2 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:19:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 1600
```

- status: started
- note: jobs=18

## 2026-06-04 10:19:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 1600
```

- status: started
- note: jobs=18

## 2026-06-04 10:19:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 1600
```

- status: started
- note: jobs=18

## 2026-06-04 10:19:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=18 traces=126

## 2026-06-04 10:19:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c1_mlp_lab_smoke --shard-count 4 --shard-index 1 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:19:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=18 traces=126

## 2026-06-04 10:19:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c1_mlp_lab_smoke --shard-count 4 --shard-index 2 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:20:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=18 traces=126

## 2026-06-04 10:20:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c1_mlp_lab_smoke --shard-count 4 --shard-index 3 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:21:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --v21-scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --steps 1600 --train-size 512 --val-size 256 --batch-size 64 --run-label v2200_c1_mlp_lab_smoke --shard-count 4 --shard-index 0
```

- status: failed
- note: command journal concurrent read/write produced CSV None field; fixed append_exec helpers and rerunning shard0

## 2026-06-04 10:21:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c1_mlp_lab_smoke --shard-count 4 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:21:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=18

## 2026-06-04 10:22:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=126

## 2026-06-04 10:22:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c1_mlp_lab_smoke --shard-count 4 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:22:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c1_mlp_lab_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:22:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=450 grouped=42

## 2026-06-04 10:22:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_c1_mlp_lab_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:22:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=42 decision=NoEarlySourceChain

## 2026-06-04 10:25:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 10:25:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=SourceObservabilityNoGo train_only_pass=0

## 2026-06-04 10:25:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 10:25:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 10:26:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: started

## 2026-06-04 10:26:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: completed
- note: route=R2-EarlySourceChainNotOpened-SourceObservabilityNoGo promotion_allowed=0

## 2026-06-04 10:43:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py experiments/run_v22_common.py experiments/run_v22_f40_phase_reset_summary.py
```

- status: completed
- note: F40 phase-reset code py_compile passed

## 2026-06-04 10:44:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_smoke --shard-count 4 --shard-index 2 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:44:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_smoke --shard-count 4 --shard-index 1 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:44:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_smoke --shard-count 4 --shard-index 3 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:44:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_smoke --shard-count 4 --shard-index 0 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:44:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 1600
```

- status: started
- note: jobs=11

## 2026-06-04 10:44:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 1600
```

- status: started
- note: jobs=11

## 2026-06-04 10:44:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 1600
```

- status: started
- note: jobs=11

## 2026-06-04 10:44:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=12

## 2026-06-04 10:45:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=77

## 2026-06-04 10:45:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_smoke --shard-count 4 --shard-index 1 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:45:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=77

## 2026-06-04 10:45:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_smoke --shard-count 4 --shard-index 3 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:45:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=77

## 2026-06-04 10:45:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_smoke --shard-count 4 --shard-index 2 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:45:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=84

## 2026-06-04 10:45:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_smoke --shard-count 4 --shard-index 0 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:45:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_
```

- status: started

## 2026-06-04 10:45:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_
```

- status: completed
- note: decision=F40RowsMissing rows=0 early=0

## 2026-06-04 10:45:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:45:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=495 grouped=43

## 2026-06-04 10:45:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:45:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=43 decision=EarlyChainButNoContinuousRetention

## 2026-06-04 10:46:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_
```

- status: started

## 2026-06-04 10:46:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_
```

- status: completed
- note: decision=F40EarlyChainNeedsFullH3200Validation rows=45 early=1

## 2026-06-04 10:48:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:48:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:48:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:48:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:48:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 10:48:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 10:48:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 10:48:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 10:49:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 10:49:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:49:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 10:49:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:49:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 10:49:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:49:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 10:49:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F40-adamw-boundary-to-momentum-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:50:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_phase_reset_full_h4800
```

- status: started

## 2026-06-04 10:50:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_phase_reset_full_h4800
```

- status: completed
- note: decision=F40RowsMissing rows=0 early=0

## 2026-06-04 10:50:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 10:50:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=540 grouped=43

## 2026-06-04 10:50:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f40_phase_reset_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 10:50:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=43 decision=EarlyChainButNoContinuousRetention

## 2026-06-04 10:50:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_phase_reset_full_h4800
```

- status: started

## 2026-06-04 10:50:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_phase_reset_full_h4800
```

- status: completed
- note: decision=F40EarlyChainNeedsFullH3200Validation rows=45 early=1

## 2026-06-04 10:59:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_f40_phase_reset_summary.py experiments/run_v22_merge_finalize.py experiments/run_v22_common.py
```

- status: completed
- note: F41 wiring compile passed

## 2026-06-04 11:00:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/dgkan_v21/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f41_phase_reset_dual_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:00:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/dgkan_v21/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f41_phase_reset_dual_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started

## 2026-06-04 11:00:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/dgkan_v21/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f41_phase_reset_dual_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:00:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/dgkan_v21/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f41_phase_reset_dual_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:00:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:00:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:00:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 11:00:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:01:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:01:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/dgkan_v21/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f41_phase_reset_dual_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:02:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:02:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/dgkan_v21/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f41_phase_reset_dual_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:02:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:02:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/dgkan_v21/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f41_phase_reset_dual_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:02:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 11:02:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/dgkan_v21/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f41_phase_reset_dual_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:02:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/dgkan_v21/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f41_phase_reset_dual_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:02:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-04 11:02:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/dgkan_v21/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f41_phase_reset_dual_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:02:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=5 decision=EarlyChainButNoContinuousRetention

## 2026-06-04 11:02:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f41_phase_reset_dual_full_h4800 --candidate-prefix MLP-F41 --artifact-prefix v22_f41_phase_reset --phase-label F41
```

- status: started

## 2026-06-04 11:02:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f41_phase_reset_dual_full_h4800 --candidate-prefix MLP-F41 --artifact-prefix v22_f41_phase_reset --phase-label F41
```

- status: completed
- note: decision=F41EarlyChainNoContinuousRetentionH3200 rows=45 early=1

## 2026-06-04 11:03:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_phase_reset_full_h4800 --candidate-prefix MLP-F40 --artifact-prefix v22_f40_phase_reset --phase-label F40
```

- status: started

## 2026-06-04 11:03:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_phase_reset_full_h4800 --candidate-prefix MLP-F40 --artifact-prefix v22_f40_phase_reset --phase-label F40
```

- status: completed
- note: decision=F40RowsMissing rows=0 early=0

## 2026-06-04 11:05:57 +0800

```bash
cp results/dgkan_v21/official_v22/v21_01_source_retention_matrix_v2200_f41_phase_reset_dual_full_h4800_fc*.csv results/dgkan_v21/official_v22/v21_01_source_retention_traces_v2200_f41_phase_reset_dual_full_h4800_fc*.csv results/dgkan_v21/official_v22/v22_f41_phase_reset_*.csv results/dgkan_v21/official_v22/v22_f41_phase_reset_*.json results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22/
```

- status: completed
- note: consolidated F41 artifacts into canonical v22 official dir

## 2026-06-04 11:06:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label cumulative_after_f41 --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:06:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=585 grouped=44

## 2026-06-04 11:06:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label cumulative_after_f41 --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:06:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=44 decision=EarlyChainButNoContinuousRetention

## 2026-06-04 11:07:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f41_phase_reset_dual_full_h4800 --candidate-prefix MLP-F41 --artifact-prefix v22_f41_phase_reset --phase-label F41
```

- status: started

## 2026-06-04 11:07:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_phase_reset_full_h4800 --candidate-prefix MLP-F40 --artifact-prefix v22_f40_phase_reset --phase-label F40
```

- status: started

## 2026-06-04 11:07:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f41_phase_reset_dual_full_h4800 --candidate-prefix MLP-F41 --artifact-prefix v22_f41_phase_reset --phase-label F41
```

- status: completed
- note: decision=F41PhaseResetNoEarlyChain rows=45 early=0

## 2026-06-04 11:07:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_phase_reset_full_h4800 --candidate-prefix MLP-F40 --artifact-prefix v22_f40_phase_reset --phase-label F40
```

- status: completed
- note: decision=F40EarlyChainNoContinuousRetentionH3200 rows=45 early=1

## 2026-06-04 11:09:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label cumulative_after_f41_runlabel_controlkey --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:09:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=585 grouped=44

## 2026-06-04 11:09:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label cumulative_after_f41_runlabel_controlkey --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:09:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=44 decision=EarlyChainButNoContinuousRetention

## 2026-06-04 11:09:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_phase_reset_full_h4800 --candidate-prefix MLP-F40 --artifact-prefix v22_f40_phase_reset --phase-label F40
```

- status: started

## 2026-06-04 11:09:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f41_phase_reset_dual_full_h4800 --candidate-prefix MLP-F41 --artifact-prefix v22_f41_phase_reset --phase-label F41
```

- status: started

## 2026-06-04 11:09:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_phase_reset_full_h4800 --candidate-prefix MLP-F40 --artifact-prefix v22_f40_phase_reset --phase-label F40
```

- status: completed
- note: decision=F40EarlyChainNoContinuousRetentionH3200 rows=45 early=1

## 2026-06-04 11:09:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_f40_phase_reset_summary.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: run_label control-attribution key fix compile passed

## 2026-06-04 11:09:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f41_phase_reset_dual_full_h4800 --candidate-prefix MLP-F41 --artifact-prefix v22_f41_phase_reset --phase-label F41
```

- status: completed
- note: decision=F41EarlyChainNoContinuousRetentionH3200 rows=45 early=1

## 2026-06-04 11:11:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label cumulative_after_f41_ratio_fields --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:11:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=585 grouped=44

## 2026-06-04 11:11:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label cumulative_after_f41_ratio_fields --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:11:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=44 decision=EarlyChainButNoContinuousRetention

## 2026-06-04 11:12:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_phase_reset_full_h4800 --candidate-prefix MLP-F40 --artifact-prefix v22_f40_phase_reset --phase-label F40
```

- status: started

## 2026-06-04 11:12:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f41_phase_reset_dual_full_h4800 --candidate-prefix MLP-F41 --artifact-prefix v22_f41_phase_reset --phase-label F41
```

- status: started

## 2026-06-04 11:12:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f40_phase_reset_full_h4800 --candidate-prefix MLP-F40 --artifact-prefix v22_f40_phase_reset --phase-label F40
```

- status: completed
- note: decision=F40EarlyChainNoContinuousRetentionH3200 rows=45 early=1

## 2026-06-04 11:12:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f41_phase_reset_dual_full_h4800 --candidate-prefix MLP-F41 --artifact-prefix v22_f41_phase_reset --phase-label F41
```

- status: completed
- note: decision=F41EarlyChainNoContinuousRetentionRatio rows=45 early=1

## 2026-06-04 11:12:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_source_chain_dynamics.py experiments/run_v22_f40_phase_reset_summary.py experiments/run_v22_merge_finalize.py experiments/run_v21_01_source_retention.py
```

- status: completed
- note: retention-ratio audit fields compile passed

## 2026-06-04 11:15:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_f40_phase_reset_summary.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F42 SGD-floor mechanism compile passed

## 2026-06-04 11:16:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f42_sgd_floor_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:16:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f42_sgd_floor_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:16:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f42_sgd_floor_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:16:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f42_sgd_floor_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:16:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:16:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:16:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:16:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 11:17:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:17:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f42_sgd_floor_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:17:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:17:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f42_sgd_floor_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:17:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:17:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f42_sgd_floor_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:17:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 11:17:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f42_sgd_floor_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:18:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label cumulative_after_f42 --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:18:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=630 grouped=45

## 2026-06-04 11:18:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label cumulative_after_f42 --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:18:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=45 decision=EarlyChainButNoContinuousRetention

## 2026-06-04 11:18:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f42_sgd_floor_full_h4800 --candidate-prefix MLP-F42 --artifact-prefix v22_f42_phase_reset --phase-label F42
```

- status: started

## 2026-06-04 11:18:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f42_sgd_floor_full_h4800 --candidate-prefix MLP-F42 --artifact-prefix v22_f42_phase_reset --phase-label F42
```

- status: completed
- note: decision=F42EarlyChainNoContinuousRetentionH3200 rows=45 early=1

## 2026-06-04 11:21:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_f40_phase_reset_summary.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F43 late-SGD-floor mechanism compile passed

## 2026-06-04 11:22:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f43_late_sgd_floor_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:22:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f43_late_sgd_floor_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:22:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f43_late_sgd_floor_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:22:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f43_late_sgd_floor_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:22:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 11:22:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:22:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:22:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:23:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:23:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f43_late_sgd_floor_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:23:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:23:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f43_late_sgd_floor_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:23:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:23:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f43_late_sgd_floor_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:23:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 11:23:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f43_late_sgd_floor_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:24:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label cumulative_after_f43 --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:24:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=675 grouped=46

## 2026-06-04 11:24:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label cumulative_after_f43 --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:24:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=46 decision=EarlyChainButNoContinuousRetention

## 2026-06-04 11:24:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f43_late_sgd_floor_full_h4800 --candidate-prefix MLP-F43 --artifact-prefix v22_f43_phase_reset --phase-label F43
```

- status: started

## 2026-06-04 11:24:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f43_late_sgd_floor_full_h4800 --candidate-prefix MLP-F43 --artifact-prefix v22_f43_phase_reset --phase-label F43
```

- status: completed
- note: decision=F43EarlyChainNoContinuousRetentionRatio rows=45 early=1

## 2026-06-04 11:27:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_f40_phase_reset_summary.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F44 tiny-late-floor mechanism compile passed

## 2026-06-04 11:28:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f44_tiny_late_floor_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:28:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f44_tiny_late_floor_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:28:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f44_tiny_late_floor_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:28:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f44_tiny_late_floor_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:28:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:28:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:28:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:28:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 11:29:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:29:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f44_tiny_late_floor_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:29:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:29:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f44_tiny_late_floor_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:29:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:29:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f44_tiny_late_floor_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:29:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 11:29:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f44_tiny_late_floor_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:30:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label cumulative_after_f44 --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:30:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=720 grouped=47

## 2026-06-04 11:30:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label cumulative_after_f44 --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:30:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=47 decision=EarlyChainButNoContinuousRetention

## 2026-06-04 11:30:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f44_tiny_late_floor_full_h4800 --candidate-prefix MLP-F44 --artifact-prefix v22_f44_phase_reset --phase-label F44
```

- status: started

## 2026-06-04 11:30:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f44_tiny_late_floor_full_h4800 --candidate-prefix MLP-F44 --artifact-prefix v22_f44_phase_reset --phase-label F44
```

- status: completed
- note: decision=F44EarlyChainNoContinuousRetentionRatio rows=45 early=1

## 2026-06-04 11:33:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_f40_phase_reset_summary.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F45 reinforced tiny-floor mechanism compile passed

## 2026-06-04 11:34:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f45_reinforced_tiny_floor_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:34:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f45_reinforced_tiny_floor_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:34:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f45_reinforced_tiny_floor_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:34:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f45_reinforced_tiny_floor_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:34:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:34:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 11:34:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:34:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 11:35:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:35:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f45_reinforced_tiny_floor_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:35:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:35:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f45_reinforced_tiny_floor_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:35:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 11:35:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f45_reinforced_tiny_floor_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:35:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 11:35:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f45_reinforced_tiny_floor_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-AdamW,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:36:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label cumulative_after_f45 --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:36:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=765 grouped=48

## 2026-06-04 11:36:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label cumulative_after_f45 --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 11:36:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=48 decision=EarlyChainButNoContinuousRetention

## 2026-06-04 11:36:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f45_reinforced_tiny_floor_full_h4800 --candidate-prefix MLP-F45 --artifact-prefix v22_f45_phase_reset --phase-label F45
```

- status: started

## 2026-06-04 11:36:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f45_reinforced_tiny_floor_full_h4800 --candidate-prefix MLP-F45 --artifact-prefix v22_f45_phase_reset --phase-label F45
```

- status: completed
- note: decision=F45EarlyChainNoContinuousRetentionRatio rows=45 early=1

## 2026-06-04 11:37:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 11:37:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=3

## 2026-06-04 11:39:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 11:39:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=3

## 2026-06-04 11:39:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_source_observability_audit.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: C4 early-vs-continuous selector audit fields compile passed

## 2026-06-04 11:48:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f46_train_loss_selector_holdout.py --artifact-prefix v22_f46_train_loss_selector
```

- status: started

## 2026-06-04 11:48:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f46_train_loss_selector_holdout.py --artifact-prefix v22_f46_train_loss_selector
```

- status: completed
- note: decision=F46TrainLossSelectorHoldsOutAsRepairHypothesis best=train_loss_h100 holdout_h4800=1

## 2026-06-04 11:50:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f46_train_loss_selector_holdout.py --artifact-prefix v22_f46_train_loss_selector
```

- status: started

## 2026-06-04 11:50:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f46_train_loss_selector_holdout.py --artifact-prefix v22_f46_train_loss_selector
```

- status: completed
- note: decision=F46TrainLossSelectorHoldsOutAsRepairHypothesis best=train_loss_h400 holdout_h4800=1

## 2026-06-04 11:58:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py experiments/run_v22_f46_train_loss_selector_holdout.py
```

- status: completed
- note: F47 train-loss gated mechanism compile passed

## 2026-06-04 11:58:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f47_trainloss_gated_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F47-trainloss-gated-dual-timescale-tiny-late-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: failed
- note: argparse rejected old --scope parameter; rerun with --v21-scope mlp

## 2026-06-04 11:59:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f47_trainloss_gated_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F47-trainloss-gated-dual-timescale-tiny-late-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F47-trainloss-gated-dual-timescale-tiny-late-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 11:59:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=45

## 2026-06-04 12:03:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=45 traces=450

## 2026-06-04 12:03:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f47_trainloss_gated_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F47-trainloss-gated-dual-timescale-tiny-late-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 12:04:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 12:04:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=810 grouped=49

## 2026-06-04 12:04:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 12:04:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=49 decision=EarlyChainButNoContinuousRetention

## 2026-06-04 12:05:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f47_trainloss_gated_full_h4800 --candidate-prefix MLP-F47 --artifact-prefix v22_f47_trainloss_gated --phase-label F47
```

- status: started

## 2026-06-04 12:05:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f47_trainloss_gated_full_h4800 --candidate-prefix MLP-F47 --artifact-prefix v22_f47_trainloss_gated --phase-label F47
```

- status: completed
- note: decision=F47EarlyChainNoContinuousRetentionRatio rows=45 early=1

## 2026-06-04 12:14:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py
```

- status: started
- note: F48 hold-fallback repair compile check

## 2026-06-04 12:15:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F48 hold-fallback repair compile check passed

## 2026-06-04 12:15:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f48_trainloss_holdfallback_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 12:15:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=45

## 2026-06-04 12:20:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=45 traces=450

## 2026-06-04 12:20:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f48_trainloss_holdfallback_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 12:20:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,MLP-F47-trainloss-gated-dual-timescale-tiny-late-source,MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,MLP-F47-trainloss-gated-dual-timescale-tiny-late-source,MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 12:20:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=855 grouped=50

## 2026-06-04 12:20:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,MLP-F47-trainloss-gated-dual-timescale-tiny-late-source,MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 12:20:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=50 decision=EarlyChainButNoContinuousRetention

## 2026-06-04 12:21:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f48_trainloss_holdfallback_full_h4800 --candidate-prefix MLP-F48 --artifact-prefix v22_f48_trainloss_holdfallback --phase-label F48
```

- status: started

## 2026-06-04 12:21:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f48_trainloss_holdfallback_full_h4800 --candidate-prefix MLP-F48 --artifact-prefix v22_f48_trainloss_holdfallback --phase-label F48
```

- status: completed
- note: decision=F48EarlyChainNoContinuousRetentionRatio rows=45 early=1

## 2026-06-04 12:25:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py
```

- status: started
- note: F49 tiny-late fallback repair compile check

## 2026-06-04 12:26:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F49 tiny-late fallback repair compile check passed

## 2026-06-04 12:26:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f49_trainloss_tinyfallback_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 12:26:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=45

## 2026-06-04 12:30:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=45 traces=450

## 2026-06-04 12:30:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f49_trainloss_tinyfallback_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 12:31:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,MLP-F47-trainloss-gated-dual-timescale-tiny-late-source,MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source,MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,MLP-F47-trainloss-gated-dual-timescale-tiny-late-source,MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source,MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 12:31:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=900 grouped=51

## 2026-06-04 12:31:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,MLP-F47-trainloss-gated-dual-timescale-tiny-late-source,MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source,MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 12:31:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=51 decision=ContinuousH3200ButNoH4800

## 2026-06-04 12:31:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f49_trainloss_tinyfallback_full_h4800 --candidate-prefix MLP-F49 --artifact-prefix v22_f49_trainloss_tinyfallback --phase-label F49
```

- status: started

## 2026-06-04 12:31:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f49_trainloss_tinyfallback_full_h4800 --candidate-prefix MLP-F49 --artifact-prefix v22_f49_trainloss_tinyfallback --phase-label F49
```

- status: completed
- note: decision=F49ContinuousH3200ButH4800Failed rows=45 early=1

## 2026-06-04 12:35:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py
```

- status: started
- note: F50 boosted tiny-late fallback compile check

## 2026-06-04 12:36:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F50 boosted tiny-late fallback compile check passed

## 2026-06-04 12:36:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f50_trainloss_boosted_tinyfallback_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F50-trainloss-gated-dual-timescale-boosted-tiny-late-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F50-trainloss-gated-dual-timescale-boosted-tiny-late-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 12:36:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=45

## 2026-06-04 12:41:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=45 traces=450

## 2026-06-04 12:41:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f50_trainloss_boosted_tinyfallback_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F50-trainloss-gated-dual-timescale-boosted-tiny-late-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 12:41:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,MLP-F47-trainloss-gated-dual-timescale-tiny-late-source,MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source,MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source,MLP-F50-trainloss-gated-dual-timescale-boosted-tiny-late-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,MLP-F47-trainloss-gated-dual-timescale-tiny-late-source,MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source,MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source,MLP-F50-trainloss-gated-dual-timescale-boosted-tiny-late-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 12:41:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=945 grouped=52

## 2026-06-04 12:41:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F23-source-vs-sgd-lookahead-gate,MLP-F40-adamw-boundary-to-momentum-source,MLP-F41-adamw-boundary-to-dual-timescale-source,MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source,MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source,MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source,MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source,MLP-F47-trainloss-gated-dual-timescale-tiny-late-source,MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source,MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source,MLP-F50-trainloss-gated-dual-timescale-boosted-tiny-late-fallback-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 12:41:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=52 decision=ContinuousH3200ButNoH4800

## 2026-06-04 12:41:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f50_trainloss_boosted_tinyfallback_full_h4800 --candidate-prefix MLP-F50 --artifact-prefix v22_f50_trainloss_boosted_tinyfallback --phase-label F50
```

- status: started

## 2026-06-04 12:41:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f50_trainloss_boosted_tinyfallback_full_h4800 --candidate-prefix MLP-F50 --artifact-prefix v22_f50_trainloss_boosted_tinyfallback --phase-label F50
```

- status: completed
- note: decision=F50EarlyChainNoContinuousRetentionRatio rows=45 early=1

## 2026-06-04 12:42:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 12:42:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=2

## 2026-06-04 12:43:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 12:43:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 12:43:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: started

## 2026-06-04 12:43:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: completed
- note: route=R4-ContinuousH3200NoH4800-F50BoostedTinyFallbackH4800Failed promotion_allowed=0

## 2026-06-04 12:44:30 +0800

```bash
pgrep -af 'run_v22_source_chain_dynamics|run_v21_01_source_retention|run_v22_efficiency|nvidia-smi'
```

- status: started
- note: final persistent process check

## 2026-06-04 12:44:59 +0800

```bash
pgrep -af 'run_v22_source_chain_dynamics|run_v21_01_source_retention|run_v22_efficiency|nvidia-smi'
```

- status: completed
- note: no persistent v22/v21.01 training or GPU monitor process matched

## 2026-06-04 12:45:17 +0800

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
```

- status: started
- note: final GPU idle check

## 2026-06-04 12:45:48 +0800

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
```

- status: completed
- note: GPU rows: 0,0,0; 1,0,0; 2,0,0; 3,0,0

## 2026-06-04 12:46:36 +0800

```bash
python -c from experiments.run_v22_common import build_packet
```

- status: started
- note: rebuild packet/bundle after final audit boundary append

## 2026-06-04 12:46:37 +0800

```bash
python -c from experiments.run_v22_common import build_packet
```

- status: completed
- note: packet/bundle rebuilt after final recap update

## 2026-06-04 12:56:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f51_late_hold_recovery_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F51-trainloss-late-hold-recovery-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F51-trainloss-late-hold-recovery-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 12:56:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=45

## 2026-06-04 13:01:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=45 traces=450

## 2026-06-04 13:01:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f51_late_hold_recovery_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F51-trainloss-late-hold-recovery-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 13:01:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f51_late_hold_recovery_full_h4800 --candidate-prefix MLP-F51 --artifact-prefix v22_f51_late_hold_recovery --phase-label F51
```

- status: started

## 2026-06-04 13:01:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f51_late_hold_recovery_full_h4800 --candidate-prefix MLP-F51 --artifact-prefix v22_f51_late_hold_recovery --phase-label F51
```

- status: completed
- note: decision=F51RowsMissing rows=0 early=0

## 2026-06-04 13:01:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 13:02:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=990 grouped=53

## 2026-06-04 13:02:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 13:02:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=53 decision=ContinuousH3200ButNoH4800

## 2026-06-04 13:02:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f51_late_hold_recovery_full_h4800 --candidate-prefix MLP-F51 --artifact-prefix v22_f51_late_hold_recovery --phase-label F51
```

- status: started

## 2026-06-04 13:02:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f51_late_hold_recovery_full_h4800 --candidate-prefix MLP-F51 --artifact-prefix v22_f51_late_hold_recovery --phase-label F51
```

- status: completed
- note: decision=F51ContinuousH3200ButH4800Failed rows=45 early=1

## 2026-06-04 13:05:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f52_late_lookahead_floor_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F52-trainloss-late-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F52-trainloss-late-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 13:05:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=45

## 2026-06-04 13:10:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=45 traces=450

## 2026-06-04 13:10:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f52_late_lookahead_floor_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F52-trainloss-late-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 13:10:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 13:10:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1035 grouped=54

## 2026-06-04 13:10:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 13:10:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=54 decision=ContinuousH3200ButNoH4800

## 2026-06-04 13:10:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f52_late_lookahead_floor_full_h4800 --candidate-prefix MLP-F52 --artifact-prefix v22_f52_late_lookahead_floor --phase-label F52
```

- status: started

## 2026-06-04 13:10:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f52_late_lookahead_floor_full_h4800 --candidate-prefix MLP-F52 --artifact-prefix v22_f52_late_lookahead_floor --phase-label F52
```

- status: completed
- note: decision=F52ContinuousH3200ButH4800Failed rows=45 early=1

## 2026-06-04 13:13:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f53_terminal_lookahead_floor_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F53-trainloss-terminal-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F53-trainloss-terminal-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 13:13:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=45

## 2026-06-04 13:18:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=45 traces=450

## 2026-06-04 13:18:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f53_terminal_lookahead_floor_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F53-trainloss-terminal-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 13:18:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 13:18:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1080 grouped=55

## 2026-06-04 13:18:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 13:18:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=55 decision=ContinuousH3200ButNoH4800

## 2026-06-04 13:18:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f53_terminal_lookahead_floor_full_h4800 --candidate-prefix MLP-F53 --artifact-prefix v22_f53_terminal_lookahead_floor --phase-label F53
```

- status: started

## 2026-06-04 13:18:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f53_terminal_lookahead_floor_full_h4800 --candidate-prefix MLP-F53 --artifact-prefix v22_f53_terminal_lookahead_floor --phase-label F53
```

- status: completed
- note: decision=F53ContinuousH3200ButH4800Failed rows=45 early=1

## 2026-06-04 13:21:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f54_early_terminal_lookahead_floor_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F54-trainloss-early-terminal-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F54-trainloss-early-terminal-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 13:21:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=45

## 2026-06-04 13:26:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=45 traces=450

## 2026-06-04 13:26:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f54_early_terminal_lookahead_floor_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F54-trainloss-early-terminal-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 13:26:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 13:26:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1125 grouped=56

## 2026-06-04 13:26:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 13:26:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=56 decision=ContinuousH3200ButNoH4800

## 2026-06-04 13:26:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f54_early_terminal_lookahead_floor_full_h4800 --candidate-prefix MLP-F54 --artifact-prefix v22_f54_early_terminal_lookahead_floor --phase-label F54
```

- status: started

## 2026-06-04 13:26:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f54_early_terminal_lookahead_floor_full_h4800 --candidate-prefix MLP-F54 --artifact-prefix v22_f54_early_terminal_lookahead_floor --phase-label F54
```

- status: completed
- note: decision=F54ContinuousH3200ButH4800Failed rows=45 early=1

## 2026-06-04 13:29:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_f40_phase_reset_summary.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F51-F54 mechanism compile check passed

## 2026-06-04 13:29:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 13:29:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 13:29:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: started

## 2026-06-04 13:29:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: completed
- note: route=R4-ContinuousH3200NoH4800-F53TerminalNearMiss-F54EarlyTerminalRegressed promotion_allowed=0

## 2026-06-04 13:39:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 13:39:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=2 legal_pass=2

## 2026-06-04 14:20:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_f40_phase_reset_summary.py
```

- status: completed
- note: F59 parameter-EMA reentry mechanism compile check passed

## 2026-06-04 13:40:18 +0800

```bash
python experiments/run_v22_s07_truth_gate.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22
```

- status: failed
- note: bare python environment lacked torch; rerun with project PYTHON interpreter

## 2026-06-04 13:40:18 +0800

```bash
python -m py_compile experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_observability_audit.py experiments/run_v22_merge_finalize.py experiments/run_v22_f40_phase_reset_summary.py
```

- status: completed
- note: syntax check passed under bare python; repeated with project interpreter

## 2026-06-04 13:40:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 13:40:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 13:43:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 13:43:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1179 grouped=58

## 2026-06-04 13:43:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 13:43:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=58 decision=ContinuousH3200ButNoH4800

## 2026-06-04 13:44:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f55_f56_split_fisher_source_reset_full_h4800 --candidate-prefix MLP-F56 --artifact-prefix v22_f56_momentum_warm_split_fisher_source --phase-label F56
```

- status: started

## 2026-06-04 13:44:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f55_f56_split_fisher_source_reset_full_h4800 --candidate-prefix MLP-F55 --artifact-prefix v22_f55_split_fisher_source_reset --phase-label F55
```

- status: started

## 2026-06-04 13:44:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 13:44:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f55_f56_split_fisher_source_reset_full_h4800 --candidate-prefix MLP-F55 --artifact-prefix v22_f55_split_fisher_source_reset --phase-label F55
```

- status: completed
- note: decision=F55PhaseResetNoEarlyChain rows=54 early=0

## 2026-06-04 13:44:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f55_f56_split_fisher_source_reset_full_h4800 --candidate-prefix MLP-F56 --artifact-prefix v22_f56_momentum_warm_split_fisher_source --phase-label F56
```

- status: completed
- note: decision=F56PhaseResetNoEarlyChain rows=54 early=0

## 2026-06-04 13:44:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=2 legal_pass=2

## 2026-06-04 13:44:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 13:44:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 13:45:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: started

## 2026-06-04 13:45:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: completed
- note: route=R4-ContinuousH3200NoH4800-F53TerminalNearMiss-F55F56SplitFisherResetFailed promotion_allowed=0

## 2026-06-04 13:46:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_observability_audit.py experiments/run_v22_merge_finalize.py experiments/run_v22_f40_phase_reset_summary.py
```

- status: completed
- note: F55/F56 wiring py_compile passed with project interpreter

## 2026-06-04 13:46:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: started

## 2026-06-04 13:46:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: completed
- note: route=R4-ContinuousH3200NoH4800-F53TerminalNearMiss-F55F56SplitFisherResetFailed promotion_allowed=0

## 2026-06-04 13:58:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 13:58:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 13:59:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f57_f58_source_theory_reset_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 13:59:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f57_f58_source_theory_reset_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 13:59:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f57_f58_source_theory_reset_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 13:59:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f57_f58_source_theory_reset_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 13:59:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=14

## 2026-06-04 13:59:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=14

## 2026-06-04 13:59:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=13

## 2026-06-04 13:59:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=13

## 2026-06-04 14:01:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=13 traces=130

## 2026-06-04 14:01:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f57_f58_source_theory_reset_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:01:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=14 traces=140

## 2026-06-04 14:01:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f57_f58_source_theory_reset_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:01:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=13 traces=130

## 2026-06-04 14:01:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f57_f58_source_theory_reset_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:01:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=14 traces=140

## 2026-06-04 14:01:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f57_f58_source_theory_reset_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:02:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f57_f58_source_theory_reset_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:02:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=1233 grouped=60

## 2026-06-04 14:02:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f57_f58_source_theory_reset_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F57-adamw-boundary-dual-timescale-antiwashout-source,MLP-F58-adamw-boundary-dual-timescale-source-anchor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:02:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=60 decision=ContinuousH3200ButNoH4800

## 2026-06-04 14:02:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f57_f58_source_theory_reset_full_h4800 --candidate-prefix MLP-F57 --artifact-prefix v22_f57_antiwashout_source_reset --phase-label F57
```

- status: started

## 2026-06-04 14:02:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f57_f58_source_theory_reset_full_h4800 --candidate-prefix MLP-F57 --artifact-prefix v22_f57_antiwashout_source_reset --phase-label F57
```

- status: completed
- note: decision=F57EarlyChainNoContinuousRetentionH3200 rows=54 early=1

## 2026-06-04 14:02:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f57_f58_source_theory_reset_full_h4800 --candidate-prefix MLP-F58 --artifact-prefix v22_f58_source_anchor_reset --phase-label F58
```

- status: started

## 2026-06-04 14:02:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f57_f58_source_theory_reset_full_h4800 --candidate-prefix MLP-F58 --artifact-prefix v22_f58_source_anchor_reset --phase-label F58
```

- status: completed
- note: decision=F58PhaseResetNoEarlyChain rows=54 early=0

## 2026-06-04 14:07:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 14:07:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=2 legal_pass=2

## 2026-06-04 14:15:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 14:15:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 14:16:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f59_param_ema_reentry_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:16:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f59_param_ema_reentry_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:16:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f59_param_ema_reentry_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:16:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f59_param_ema_reentry_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:16:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 14:16:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 14:16:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 14:16:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 14:18:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 14:18:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f59_param_ema_reentry_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:18:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 14:18:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f59_param_ema_reentry_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:18:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 14:18:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f59_param_ema_reentry_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:18:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 14:18:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f59_param_ema_reentry_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:18:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f59_param_ema_reentry_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:18:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=1278 grouped=61

## 2026-06-04 14:18:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f59_param_ema_reentry_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:18:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=61 decision=ContinuousH3200ButNoH4800

## 2026-06-04 14:19:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f59_param_ema_reentry_full_h4800 --candidate-prefix MLP-F59 --artifact-prefix v22_f59_param_ema_reentry_reset --phase-label F59
```

- status: started

## 2026-06-04 14:19:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f59_param_ema_reentry_full_h4800 --candidate-prefix MLP-F59 --artifact-prefix v22_f59_param_ema_reentry_reset --phase-label F59
```

- status: completed
- note: decision=F59EarlyChainNoContinuousRetentionRatio rows=45 early=1

## 2026-06-04 14:19:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 14:19:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=2 legal_pass=2

## 2026-06-04 14:20:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: started

## 2026-06-04 14:20:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: completed
- note: route=R4-ContinuousH3200NoH4800-F57F58F59SourceTheoryResetFailed promotion_allowed=0

## 2026-06-04 14:30:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 14:30:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 14:31:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f60_readout_channel_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:31:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f60_readout_channel_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:31:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f60_readout_channel_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:31:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f60_readout_channel_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:31:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 14:31:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 14:31:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 14:31:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 14:32:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 14:32:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f60_readout_channel_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:32:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 14:32:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f60_readout_channel_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:32:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 14:32:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f60_readout_channel_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:32:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 14:32:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f60_readout_channel_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:33:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f60_readout_channel_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:33:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=1323 grouped=62

## 2026-06-04 14:33:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f60_readout_channel_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F60-adamw-boundary-dual-timescale-readout-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:33:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=62 decision=ContinuousH3200ButNoH4800

## 2026-06-04 14:33:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f60_readout_channel_full_h4800 --candidate-prefix MLP-F60 --artifact-prefix v22_f60_readout_channel_reset --phase-label F60
```

- status: started

## 2026-06-04 14:33:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f60_readout_channel_full_h4800 --candidate-prefix MLP-F60 --artifact-prefix v22_f60_readout_channel_reset --phase-label F60
```

- status: completed
- note: decision=F60EarlyChainNoContinuousRetentionH3200 rows=45 early=1

## 2026-06-04 14:43:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_f40_phase_reset_summary.py
```

- status: completed
- note: F61/M105 hidden-matrix channel source-theory reset static compile passed

## 2026-06-04 14:44:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 14:44:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 14:45:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f61_hidden_matrix_channel_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:45:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f61_hidden_matrix_channel_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:45:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f61_hidden_matrix_channel_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:45:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f61_hidden_matrix_channel_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:45:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 14:45:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 14:45:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 14:45:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 14:46:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 14:46:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f61_hidden_matrix_channel_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:46:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 14:46:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f61_hidden_matrix_channel_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:47:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 14:47:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f61_hidden_matrix_channel_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:47:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 14:47:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f61_hidden_matrix_channel_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:47:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f61_hidden_matrix_channel_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:47:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=1368 grouped=63

## 2026-06-04 14:47:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f61_hidden_matrix_channel_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:47:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=63 decision=ContinuousH3200ButNoH4800

## 2026-06-04 14:47:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f61_hidden_matrix_channel_full_h4800 --candidate-prefix MLP-F61 --artifact-prefix v22_f61_hidden_matrix_channel_reset --phase-label F61
```

- status: started

## 2026-06-04 14:47:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f61_hidden_matrix_channel_full_h4800 --candidate-prefix MLP-F61 --artifact-prefix v22_f61_hidden_matrix_channel_reset --phase-label F61
```

- status: completed
- note: decision=F61EarlyChainNoContinuousRetentionRatio rows=45 early=1

## 2026-06-04 14:48:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 14:48:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=2 legal_pass=2

## 2026-06-04 14:55:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_f40_phase_reset_summary.py
```

- status: completed
- note: F62/M106 terminal projected lookahead floor static compile passed

## 2026-06-04 14:55:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 14:56:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 14:57:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 14:57:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 14:58:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_s07_truth_gate.py dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py experiments/run_v22_f40_phase_reset_summary.py
```

- status: completed
- note: F62/M106 truth gate contract list compile passed; S0.7 mechanism_contracts=21/21

## 2026-06-04 14:58:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f62_terminal_projected_lookahead_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:58:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f62_terminal_projected_lookahead_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:58:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f62_terminal_projected_lookahead_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:58:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f62_terminal_projected_lookahead_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 14:58:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 14:58:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 14:58:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 14:58:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 14:59:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 14:59:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f62_terminal_projected_lookahead_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:59:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 14:59:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f62_terminal_projected_lookahead_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 14:59:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 14:59:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f62_terminal_projected_lookahead_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:00:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 15:00:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f62_terminal_projected_lookahead_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:00:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f62_terminal_projected_lookahead_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:00:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=1413 grouped=64

## 2026-06-04 15:00:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f62_terminal_projected_lookahead_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F62-trainloss-terminal-projected-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:00:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=64 decision=ContinuousH3200ButNoH4800

## 2026-06-04 15:00:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f62_terminal_projected_lookahead_full_h4800 --candidate-prefix MLP-F62 --artifact-prefix v22_f62_terminal_projected_lookahead_floor --phase-label F62
```

- status: started

## 2026-06-04 15:00:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f62_terminal_projected_lookahead_full_h4800 --candidate-prefix MLP-F62 --artifact-prefix v22_f62_terminal_projected_lookahead_floor --phase-label F62
```

- status: completed
- note: decision=F62EarlyChainNoContinuousRetentionRatio rows=45 early=1

## 2026-06-04 15:01:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 15:01:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=2 legal_pass=2

## 2026-06-04 15:01:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: started

## 2026-06-04 15:01:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: completed
- note: route=R4-ContinuousH3200NoH4800-F57F58F59F60F61F62SourceTheoryResetFailed promotion_allowed=0

## 2026-06-04 15:08:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_f40_phase_reset_summary.py
```

- status: completed
- note: F63/M107 terminal consensus lookahead floor static compile passed

## 2026-06-04 15:08:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 15:08:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 15:09:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f63_terminal_consensus_lookahead_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:09:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f63_terminal_consensus_lookahead_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:09:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f63_terminal_consensus_lookahead_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:09:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f63_terminal_consensus_lookahead_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:09:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 15:09:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 15:09:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 15:09:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 15:10:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 15:10:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f63_terminal_consensus_lookahead_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:10:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 15:10:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f63_terminal_consensus_lookahead_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:10:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 15:10:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f63_terminal_consensus_lookahead_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:10:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 15:10:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f63_terminal_consensus_lookahead_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:11:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f63_terminal_consensus_lookahead_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:11:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=1458 grouped=65

## 2026-06-04 15:11:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f63_terminal_consensus_lookahead_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F63-trainloss-terminal-consensus-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:11:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=65 decision=ContinuousH3200ButNoH4800

## 2026-06-04 15:11:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f63_terminal_consensus_lookahead_full_h4800 --candidate-prefix MLP-F63 --artifact-prefix v22_f63_terminal_consensus_lookahead_floor --phase-label F63
```

- status: started

## 2026-06-04 15:11:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f63_terminal_consensus_lookahead_full_h4800 --candidate-prefix MLP-F63 --artifact-prefix v22_f63_terminal_consensus_lookahead_floor --phase-label F63
```

- status: completed
- note: decision=F63ContinuousH3200ButH4800Failed rows=45 early=1

## 2026-06-04 15:20:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_f40_phase_reset_summary.py
```

- status: completed
- note: F64/M108 terminal selector lookahead floor static compile passed

## 2026-06-04 15:21:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 15:21:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 15:22:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f64_terminal_selector_lookahead_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:22:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f64_terminal_selector_lookahead_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:22:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f64_terminal_selector_lookahead_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:22:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f64_terminal_selector_lookahead_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:22:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 15:22:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 15:22:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 15:22:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 15:23:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 15:23:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f64_terminal_selector_lookahead_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:23:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 15:23:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f64_terminal_selector_lookahead_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:23:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 15:23:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f64_terminal_selector_lookahead_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:23:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 15:23:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f64_terminal_selector_lookahead_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:24:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f64_terminal_selector_lookahead_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:24:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=1503 grouped=66

## 2026-06-04 15:24:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f64_terminal_selector_lookahead_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F64-trainloss-terminal-selector-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:24:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=66 decision=ContinuousH3200ButNoH4800

## 2026-06-04 15:24:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f64_terminal_selector_lookahead_full_h4800 --candidate-prefix MLP-F64 --artifact-prefix v22_f64_terminal_selector_lookahead_floor --phase-label F64
```

- status: started

## 2026-06-04 15:24:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f64_terminal_selector_lookahead_full_h4800 --candidate-prefix MLP-F64 --artifact-prefix v22_f64_terminal_selector_lookahead_floor --phase-label F64
```

- status: completed
- note: decision=F64ContinuousH3200ButH4800Failed rows=45 early=1

## 2026-06-04 15:30:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_merge_finalize.py experiments/run_v22_s07_truth_gate.py
```

- status: completed
- note: F65 AdamW split-Fisher residual spec/finalizer static compile passed

## 2026-06-04 15:30:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 15:30:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 15:31:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_s07_truth_gate.py
```

- status: completed
- note: F65/M38 S0.7 wanted contract patch compile passed

## 2026-06-04 15:32:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 15:32:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 15:33:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f65_adamw_split_fisher_residual_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:33:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f65_adamw_split_fisher_residual_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:33:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f65_adamw_split_fisher_residual_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:33:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f65_adamw_split_fisher_residual_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:33:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 15:33:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 15:33:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 15:33:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 15:34:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 15:34:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f65_adamw_split_fisher_residual_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:34:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 15:34:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f65_adamw_split_fisher_residual_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:34:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 15:34:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f65_adamw_split_fisher_residual_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:34:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 15:34:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f65_adamw_split_fisher_residual_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:34:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f65_adamw_split_fisher_residual_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:35:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=1548 grouped=67

## 2026-06-04 15:35:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f65_adamw_split_fisher_residual_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F65-adamw-split-fisher-residual-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:35:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=67 decision=ContinuousH3200ButNoH4800

## 2026-06-04 15:35:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f65_adamw_split_fisher_residual_full_h4800 --candidate-prefix MLP-F65 --artifact-prefix v22_f65_adamw_split_fisher_residual --phase-label F65
```

- status: started

## 2026-06-04 15:35:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f65_adamw_split_fisher_residual_full_h4800 --candidate-prefix MLP-F65 --artifact-prefix v22_f65_adamw_split_fisher_residual --phase-label F65
```

- status: completed
- note: decision=F65EarlyChainNoContinuousRetentionH3200 rows=45 early=1

## 2026-06-04 15:35:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 15:35:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=2 legal_pass=2

## 2026-06-04 15:36:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: started

## 2026-06-04 15:36:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: completed
- note: route=R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65SourceTheoryResetFailed promotion_allowed=0

## 2026-06-04 15:42:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_merge_finalize.py experiments/run_v22_f40_phase_reset_summary.py
```

- status: completed
- note: F66 KAN readout-only writer scope/finalizer static compile passed

## 2026-06-04 15:43:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 15:43:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 15:44:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f66_kan_readout_commit_full_h4800 --shard-count 4 --shard-index 0 --spec-ids KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=kan; specs=KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:44:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f66_kan_readout_commit_full_h4800 --shard-count 4 --shard-index 1 --spec-ids KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=kan; specs=KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:44:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f66_kan_readout_commit_full_h4800 --shard-count 4 --shard-index 2 --spec-ids KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=kan; specs=KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:44:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f66_kan_readout_commit_full_h4800 --shard-count 4 --shard-index 3 --spec-ids KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=kan; specs=KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:44:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-04 15:44:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-04 15:44:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-04 15:44:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-04 15:49:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=27 traces=270

## 2026-06-04 15:49:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f66_kan_readout_commit_full_h4800 --shard-count 4 --shard-index 0 --spec-ids KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:50:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2
```

- status: completed
- note: rows=27 traces=270

## 2026-06-04 15:50:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f66_kan_readout_commit_full_h4800 --shard-count 4 --shard-index 2 --spec-ids KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:50:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1
```

- status: completed
- note: rows=27 traces=270

## 2026-06-04 15:50:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f66_kan_readout_commit_full_h4800 --shard-count 4 --shard-index 1 --spec-ids KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:50:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3
```

- status: completed
- note: rows=27 traces=270

## 2026-06-04 15:50:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f66_kan_readout_commit_full_h4800 --shard-count 4 --shard-index 3 --spec-ids KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:51:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f66_kan_readout_commit_full_h4800 --shard-count 1 --shard-index 0 --spec-ids KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=kan; specs=KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:51:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=1656 grouped=69

## 2026-06-04 15:51:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope kan --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f66_kan_readout_commit_full_h4800 --shard-count 1 --shard-index 0 --spec-ids KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 15:51:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=69 decision=ContinuousH3200ButNoH4800

## 2026-06-04 15:51:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f66_kan_readout_commit_full_h4800 --candidate-prefix KSW1 --artifact-prefix v22_f66_kan_readout_commit --phase-label F66
```

- status: started

## 2026-06-04 15:51:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f66_kan_readout_commit_full_h4800 --candidate-prefix KSW1 --artifact-prefix v22_f66_kan_readout_commit --phase-label F66
```

- status: completed
- note: decision=F66PhaseResetNoEarlyChain rows=108 early=0

## 2026-06-04 15:52:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 15:52:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=2 legal_pass=2

## 2026-06-04 15:52:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: started

## 2026-06-04 15:52:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: completed
- note: route=R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65SourceTheoryResetFailed promotion_allowed=0

## 2026-06-04 15:54:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F67 KAN lowbank B3-null finalizer static compile passed

## 2026-06-04 15:54:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f67_kan_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 3 --spec-ids F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:54:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f67_kan_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 1 --spec-ids F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:54:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f67_kan_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 0 --spec-ids F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:54:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f67_kan_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 2 --spec-ids F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 15:54:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-04 15:54:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-04 15:54:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-04 15:54:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-04 16:01:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=27 traces=270

## 2026-06-04 16:01:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f67_kan_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 0 --spec-ids F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:01:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=27 traces=270

## 2026-06-04 16:01:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f67_kan_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 2 --spec-ids F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:01:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=27 traces=270

## 2026-06-04 16:01:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f67_kan_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 3 --spec-ids F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:01:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=27 traces=270

## 2026-06-04 16:01:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f67_kan_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 1 --spec-ids F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:01:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f67_kan_lowbank_b3_null_full_h4800 --shard-count 1 --shard-index 0 --spec-ids F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:01:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1764 grouped=73

## 2026-06-04 16:01:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f67_kan_lowbank_b3_null_full_h4800 --shard-count 1 --shard-index 0 --spec-ids F36-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:01:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=73 decision=ContinuousH3200ButNoH4800

## 2026-06-04 16:02:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f67_kan_lowbank_b3_null_full_h4800 --candidate-prefix F38 --artifact-prefix v22_f67_kan_lowbank_b3_null --phase-label F67
```

- status: started

## 2026-06-04 16:02:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f67_kan_lowbank_b3_null_full_h4800 --candidate-prefix F38 --artifact-prefix v22_f67_kan_lowbank_b3_null --phase-label F67
```

- status: completed
- note: decision=F67PhaseResetNoEarlyChain rows=108 early=0

## 2026-06-04 16:03:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 16:03:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=2 legal_pass=2

## 2026-06-04 16:03:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: started

## 2026-06-04 16:03:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: completed
- note: route=R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65SourceTheoryResetFailed promotion_allowed=0

## 2026-06-04 16:13:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F68 code registration compile passed

## 2026-06-04 16:13:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check mechanism_contracts --device cuda:0
```

- status: started

## 2026-06-04 16:13:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check mechanism_contracts
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 16:13:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check update_semantics --device cuda:0
```

- status: started

## 2026-06-04 16:13:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check update_semantics
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 16:14:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/dg_lca_v22_f68_smoke --device cuda:0 --data-root data --scope target --carriers D-CHE --datasets MNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 120 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f68_adamw_boundary_lowbank_b3_null_smoke --shard-count 1 --shard-index 0 --spec-ids F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:14:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 120
```

- status: started
- note: jobs=5

## 2026-06-04 16:15:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=5 traces=20

## 2026-06-04 16:15:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/dg_lca_v22_f68_smoke --device cuda:0 --data-root data --scope target --carriers D-CHE --datasets MNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 120 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f68_adamw_boundary_lowbank_b3_null_smoke --shard-count 1 --shard-index 0 --spec-ids F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:15:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f68_adamw_boundary_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 3 --spec-ids F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:15:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f68_adamw_boundary_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 2 --spec-ids F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:15:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f68_adamw_boundary_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 0 --spec-ids F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:15:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f68_adamw_boundary_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 1 --spec-ids F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:15:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=23

## 2026-06-04 16:15:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=23

## 2026-06-04 16:15:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=22

## 2026-06-04 16:15:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=22

## 2026-06-04 16:21:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=22 traces=220

## 2026-06-04 16:21:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f68_adamw_boundary_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 2 --spec-ids F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:21:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=22 traces=220

## 2026-06-04 16:21:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f68_adamw_boundary_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 3 --spec-ids F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:21:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=23 traces=230

## 2026-06-04 16:21:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f68_adamw_boundary_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 0 --spec-ids F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:21:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=23 traces=230

## 2026-06-04 16:21:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f68_adamw_boundary_lowbank_b3_null_full_h4800 --shard-count 4 --shard-index 1 --spec-ids F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:21:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:21:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1854 grouped=75

## 2026-06-04 16:21:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:21:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=75 decision=ContinuousH3200ButNoH4800

## 2026-06-04 16:22:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f68_adamw_boundary_lowbank_b3_null_full_h4800 --candidate-prefix F68 --artifact-prefix v22_f68_adamw_boundary_lowbank_b3_null --phase-label F68
```

- status: started

## 2026-06-04 16:22:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f68_adamw_boundary_lowbank_b3_null_full_h4800 --candidate-prefix F68 --artifact-prefix v22_f68_adamw_boundary_lowbank_b3_null --phase-label F68
```

- status: completed
- note: decision=F68PhaseResetNoEarlyChain rows=90 early=0

## 2026-06-04 16:28:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F69 code registration compile passed

## 2026-06-04 16:28:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check update_semantics --device cuda:0
```

- status: started

## 2026-06-04 16:28:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check mechanism_contracts --device cuda:0
```

- status: started

## 2026-06-04 16:28:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check mechanism_contracts
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 16:28:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check update_semantics
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 16:29:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/dg_lca_v22_f69_smoke --device cuda:0 --data-root data --scope target --carriers D-FOU --datasets MNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 120 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f69_adamw_lowbank_anchor_antiwashout_smoke --shard-count 1 --shard-index 0 --spec-ids F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:29:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 120
```

- status: started
- note: jobs=5

## 2026-06-04 16:29:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=5 traces=20

## 2026-06-04 16:29:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/dg_lca_v22_f69_smoke --device cuda:0 --data-root data --scope target --carriers D-FOU --datasets MNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 120 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f69_adamw_lowbank_anchor_antiwashout_smoke --shard-count 1 --shard-index 0 --spec-ids F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:30:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f69_adamw_lowbank_anchor_antiwashout_full_h4800 --shard-count 4 --shard-index 0 --spec-ids F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:30:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f69_adamw_lowbank_anchor_antiwashout_full_h4800 --shard-count 4 --shard-index 2 --spec-ids F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:30:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f69_adamw_lowbank_anchor_antiwashout_full_h4800 --shard-count 4 --shard-index 3 --spec-ids F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:30:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f69_adamw_lowbank_anchor_antiwashout_full_h4800 --shard-count 4 --shard-index 1 --spec-ids F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:30:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=23

## 2026-06-04 16:30:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=23

## 2026-06-04 16:30:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=22

## 2026-06-04 16:30:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=22

## 2026-06-04 16:35:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=22 traces=220

## 2026-06-04 16:35:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f69_adamw_lowbank_anchor_antiwashout_full_h4800 --shard-count 4 --shard-index 2 --spec-ids F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:35:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=22 traces=220

## 2026-06-04 16:35:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f69_adamw_lowbank_anchor_antiwashout_full_h4800 --shard-count 4 --shard-index 3 --spec-ids F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:35:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=23 traces=230

## 2026-06-04 16:35:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f69_adamw_lowbank_anchor_antiwashout_full_h4800 --shard-count 4 --shard-index 0 --spec-ids F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:36:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=23 traces=230

## 2026-06-04 16:36:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f69_adamw_lowbank_anchor_antiwashout_full_h4800 --shard-count 4 --shard-index 1 --spec-ids F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:36:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:36:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1944 grouped=77

## 2026-06-04 16:36:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:36:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=77 decision=ContinuousH3200ButNoH4800

## 2026-06-04 16:36:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f69_adamw_lowbank_anchor_antiwashout_full_h4800 --candidate-prefix F69 --artifact-prefix v22_f69_adamw_lowbank_anchor_antiwashout --phase-label F69
```

- status: started

## 2026-06-04 16:36:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f69_adamw_lowbank_anchor_antiwashout_full_h4800 --candidate-prefix F69 --artifact-prefix v22_f69_adamw_lowbank_anchor_antiwashout --phase-label F69
```

- status: completed
- note: decision=F69EarlyChainNoContinuousRetentionH3200 rows=90 early=1

## 2026-06-04 16:37:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 16:37:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 16:38:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 16:38:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=2 legal_pass=2

## 2026-06-04 16:38:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: started

## 2026-06-04 16:38:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: completed
- note: route=R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65F66F67F68F69SourceTheoryResetFailed promotion_allowed=0

## 2026-06-04 16:46:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F70 code registration compile passed

## 2026-06-04 16:46:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check update_semantics --device cuda:0
```

- status: started

## 2026-06-04 16:46:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check mechanism_contracts --device cuda:0
```

- status: started

## 2026-06-04 16:46:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check mechanism_contracts
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 16:46:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check update_semantics
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 16:47:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/dg_lca_v22_f70_smoke --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 120 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f70_terminal_positive_lookahead_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:47:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 120
```

- status: started
- note: jobs=5

## 2026-06-04 16:47:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=20

## 2026-06-04 16:47:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/dg_lca_v22_f70_smoke --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 120 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f70_terminal_positive_lookahead_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:47:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f70_terminal_positive_lookahead_floor_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:47:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f70_terminal_positive_lookahead_floor_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:47:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f70_terminal_positive_lookahead_floor_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:47:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f70_terminal_positive_lookahead_floor_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:47:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 16:47:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 16:47:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 16:47:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 16:48:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 16:48:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f70_terminal_positive_lookahead_floor_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:49:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 16:49:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f70_terminal_positive_lookahead_floor_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:49:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 16:49:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f70_terminal_positive_lookahead_floor_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:49:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 16:49:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f70_terminal_positive_lookahead_floor_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F70-trainloss-terminal-positive-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:49:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:49:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1989 grouped=78

## 2026-06-04 16:49:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:49:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=78 decision=ContinuousH3200ButNoH4800

## 2026-06-04 16:49:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f70_terminal_positive_lookahead_floor_full_h4800 --candidate-prefix MLP-F70 --artifact-prefix v22_f70_terminal_positive_lookahead_floor --phase-label F70
```

- status: started

## 2026-06-04 16:49:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f70_terminal_positive_lookahead_floor_full_h4800 --candidate-prefix MLP-F70 --artifact-prefix v22_f70_terminal_positive_lookahead_floor --phase-label F70
```

- status: completed
- note: decision=F70ContinuousH3200ButH4800Failed rows=45 early=1

## 2026-06-04 16:56:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F71 code registration compile passed

## 2026-06-04 16:56:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check update_semantics --device cuda:0
```

- status: started

## 2026-06-04 16:56:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check update_semantics
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 16:56:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check mechanism_contracts --device cuda:0
```

- status: started

## 2026-06-04 16:56:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check mechanism_contracts
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 16:58:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/dg_lca_v22_f71_terminal_smoke --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST --seeds 0 --train-size 128 --val-size 128 --batch-size 64 --steps 4050 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f71_terminal_checkpoint_reentry_terminal_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:58:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4050
```

- status: started
- note: jobs=5

## 2026-06-04 16:58:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=45

## 2026-06-04 16:58:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/dg_lca_v22_f71_terminal_smoke --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST --seeds 0 --train-size 128 --val-size 128 --batch-size 64 --steps 4050 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f71_terminal_checkpoint_reentry_terminal_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 16:58:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f71_terminal_checkpoint_reentry_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:58:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f71_terminal_checkpoint_reentry_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:58:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f71_terminal_checkpoint_reentry_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:58:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f71_terminal_checkpoint_reentry_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 16:58:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 16:58:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 16:58:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 16:58:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 17:00:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 17:00:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f71_terminal_checkpoint_reentry_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:00:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 17:00:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f71_terminal_checkpoint_reentry_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:00:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 17:00:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f71_terminal_checkpoint_reentry_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:00:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 17:00:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f71_terminal_checkpoint_reentry_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F71-trainloss-terminal-h3200-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:00:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:00:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=2034 grouped=79

## 2026-06-04 17:00:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:00:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=79 decision=ContinuousH3200ButNoH4800

## 2026-06-04 17:01:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f71_terminal_checkpoint_reentry_full_h4800 --candidate-prefix MLP-F71 --artifact-prefix v22_f71_terminal_checkpoint_reentry --phase-label F71
```

- status: started

## 2026-06-04 17:01:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f71_terminal_checkpoint_reentry_full_h4800 --candidate-prefix MLP-F71 --artifact-prefix v22_f71_terminal_checkpoint_reentry --phase-label F71
```

- status: completed
- note: decision=F71ContinuousH3200ButH4800Failed rows=45 early=1

## 2026-06-04 17:04:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F71 row-level localization recap table compile passed

## 2026-06-04 17:04:07 +0800

```bash
python - <<'PY'  # inspect F53/F70/F71 source_chain_matrix row-level source_h* by dataset/seed
```

- status: completed
- note: F71 localization: MNIST h4800 positive; Fashion-MNIST/KMNIST terminal h4800 negative across F53/F70/F71; F71 aggregate h4800=-0.016645153363545735

## 2026-06-04 17:04:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 17:04:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 17:04:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 17:04:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=2 legal_pass=2

## 2026-06-04 17:04:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: started

## 2026-06-04 17:04:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: completed
- note: route=R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65F66F67F68F69F70F71SourceTheoryResetFailed promotion_allowed=0

## 2026-06-04 17:14:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F72 hard-split source code registration compile passed

## 2026-06-04 17:14:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check update_semantics --device cuda:0
```

- status: started

## 2026-06-04 17:14:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check mechanism_contracts --device cuda:0
```

- status: started

## 2026-06-04 17:14:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check mechanism_contracts
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 17:14:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check update_semantics
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 17:15:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/dg_lca_v22_f72_terminal_smoke --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST --seeds 0 --train-size 128 --val-size 128 --batch-size 64 --steps 4050 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f72_terminal_hard_split_source_terminal_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:15:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4050
```

- status: started
- note: jobs=5

## 2026-06-04 17:15:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=45

## 2026-06-04 17:15:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/dg_lca_v22_f72_terminal_smoke --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST --seeds 0 --train-size 128 --val-size 128 --batch-size 64 --steps 4050 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f72_terminal_hard_split_source_terminal_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:16:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f72_terminal_hard_split_source_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:16:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f72_terminal_hard_split_source_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:16:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f72_terminal_hard_split_source_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:16:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f72_terminal_hard_split_source_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:16:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 17:16:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 17:16:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 17:16:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 17:18:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 17:18:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f72_terminal_hard_split_source_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:18:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 17:18:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f72_terminal_hard_split_source_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:18:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 17:18:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f72_terminal_hard_split_source_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:18:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 17:18:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f72_terminal_hard_split_source_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F72-trainloss-terminal-hard-split-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:18:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:18:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=2079 grouped=80

## 2026-06-04 17:18:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:18:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=80 decision=ContinuousH3200ButNoH4800

## 2026-06-04 17:18:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f72_terminal_hard_split_source_full_h4800 --candidate-prefix MLP-F72 --artifact-prefix v22_f72_terminal_hard_split_source --phase-label F72
```

- status: started

## 2026-06-04 17:18:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f72_terminal_hard_split_source_full_h4800 --candidate-prefix MLP-F72 --artifact-prefix v22_f72_terminal_hard_split_source --phase-label F72
```

- status: completed
- note: decision=F72ContinuousH3200ButH4800Failed rows=45 early=1

## 2026-06-04 17:26:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F73 AdamW-lookahead code registration compile passed

## 2026-06-04 17:26:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check update_semantics --device cuda:0
```

- status: started

## 2026-06-04 17:26:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check mechanism_contracts --device cuda:0
```

- status: started

## 2026-06-04 17:26:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check mechanism_contracts
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 17:26:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check update_semantics
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 17:27:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/dg_lca_v22_f73_terminal_smoke --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST --seeds 0 --train-size 128 --val-size 128 --batch-size 64 --steps 4050 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f73_terminal_adamw_lookahead_terminal_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:27:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4050
```

- status: started
- note: jobs=5

## 2026-06-04 17:27:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=45

## 2026-06-04 17:27:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/dg_lca_v22_f73_terminal_smoke --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST --seeds 0 --train-size 128 --val-size 128 --batch-size 64 --steps 4050 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f73_terminal_adamw_lookahead_terminal_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:28:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f73_terminal_adamw_lookahead_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:28:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f73_terminal_adamw_lookahead_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:28:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f73_terminal_adamw_lookahead_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:28:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f73_terminal_adamw_lookahead_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:28:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 17:28:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 17:28:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 17:28:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 17:29:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 17:29:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f73_terminal_adamw_lookahead_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:29:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 17:29:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f73_terminal_adamw_lookahead_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:29:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 17:29:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f73_terminal_adamw_lookahead_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:29:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 17:29:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f73_terminal_adamw_lookahead_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F73-trainloss-terminal-adamw-lookahead,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:30:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:30:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=2124 grouped=81

## 2026-06-04 17:30:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:30:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=81 decision=ContinuousH3200ButNoH4800

## 2026-06-04 17:30:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f73_terminal_adamw_lookahead_full_h4800 --candidate-prefix MLP-F73 --artifact-prefix v22_f73_terminal_adamw_lookahead --phase-label F73
```

- status: started

## 2026-06-04 17:30:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f73_terminal_adamw_lookahead_full_h4800 --candidate-prefix MLP-F73 --artifact-prefix v22_f73_terminal_adamw_lookahead --phase-label F73
```

- status: completed
- note: decision=F73ContinuousH3200ButH4800Failed rows=45 early=1

## 2026-06-04 17:36:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_source_chain_dynamics.py experiments/run_v22_s07_truth_gate.py experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F74 terminal optimizer selector code registration compile passed

## 2026-06-04 17:36:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check mechanism_contracts --device cuda:0
```

- status: started

## 2026-06-04 17:36:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check mechanism_contracts
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 17:36:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check update_semantics --device cuda:0
```

- status: started

## 2026-06-04 17:36:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check update_semantics
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 17:37:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/dg_lca_v22_f74_terminal_smoke --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST --seeds 0 --train-size 128 --val-size 128 --batch-size 64 --steps 4050 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f74_terminal_optimizer_selector_terminal_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:37:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4050
```

- status: started
- note: jobs=5

## 2026-06-04 17:37:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=45

## 2026-06-04 17:37:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/dg_lca_v22_f74_terminal_smoke --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST --seeds 0 --train-size 128 --val-size 128 --batch-size 64 --steps 4050 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f74_terminal_optimizer_selector_terminal_smoke --shard-count 1 --shard-index 0 --spec-ids MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:38:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f74_terminal_optimizer_selector_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:38:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f74_terminal_optimizer_selector_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:38:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f74_terminal_optimizer_selector_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:38:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f74_terminal_optimizer_selector_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: delegates to v21.01 source runner; scope=mlp; specs=MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:38:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 17:38:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 17:38:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 17:38:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 17:39:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 17:39:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:2 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f74_terminal_optimizer_selector_full_h4800 --shard-count 4 --shard-index 2 --spec-ids MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:39:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 17:39:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:1 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f74_terminal_optimizer_selector_full_h4800 --shard-count 4 --shard-index 1 --spec-ids MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:39:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 17:39:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:3 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f74_terminal_optimizer_selector_full_h4800 --shard-count 4 --shard-index 3 --spec-ids MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:39:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 17:39:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_f74_terminal_optimizer_selector_full_h4800 --shard-count 4 --shard-index 0 --spec-ids MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:40:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: started
- note: delegates to v21.01 source runner; scope=target; specs=F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

## 2026-06-04 17:40:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=2169 grouped=82

## 2026-06-04 17:40:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22 --device cuda:0 --data-root data --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2200_source_chain_smoke --shard-count 1 --shard-index 0 --spec-ids F3-T1-loss-cotangent-target,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F10-T7-b1-cross-split-consensus-transfer,F25-loss-warm-to-b1-consensus-migration,F30-gain-gated-loss-warm-b1-consensus,F33-loss-warm-to-b1-consensus-b3-null,F35-loss-warm-to-view-consistent-loss,F37-loss-warm-to-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F68-adamw-boundary-to-gated-lowbank-b3-null,F69-adamw-boundary-lowbank-anchor-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --merge-only
```

- status: completed
- note: v21.01 raw/source-retention artifacts generated in v22 official dir

## 2026-06-04 17:40:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_chain_dynamics.py --merge-only
```

- status: completed
- note: v22 source groups=82 decision=ContinuousH3200ButNoH4800

## 2026-06-04 17:40:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f74_terminal_optimizer_selector_full_h4800 --candidate-prefix MLP-F74 --artifact-prefix v22_f74_terminal_optimizer_selector --phase-label F74
```

- status: started

## 2026-06-04 17:40:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_f40_phase_reset_summary.py --run-prefix v2200_f74_terminal_optimizer_selector_full_h4800 --candidate-prefix MLP-F74 --artifact-prefix v22_f74_terminal_optimizer_selector --phase-label F74
```

- status: completed
- note: decision=F74ContinuousH3200ButH4800Failed rows=45 early=1

## 2026-06-04 17:41:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_merge_finalize.py
```

- status: completed
- note: F74 recap row-level table compile passed

## 2026-06-04 17:41:41 +0800

```bash
python - <<'PY'  # inspect F53/F72/F73/F74 source_chain_matrix row-level source_h* by dataset/seed
```

- status: completed
- note: F74 localization: raw-vs-AdamW selector still h4800 failed; F53 remains best aggregate h4800=-0.003963543309105767; F74 h4800=-0.011182712184058296

## 2026-06-04 17:41:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all --device cuda:0
```

- status: started

## 2026-06-04 17:42:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_s07_truth_gate.py --check all
```

- status: completed
- note: S0.7_pass=1 checks=6

## 2026-06-04 17:42:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: started

## 2026-06-04 17:42:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_source_observability_audit.py
```

- status: completed
- note: decision=TrainOnlySelectorCandidateNeedsHeldOut train_only_pass=2 legal_pass=2

## 2026-06-04 17:42:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: started

## 2026-06-04 17:42:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_merge_finalize.py
```

- status: completed
- note: route=R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65F66F67F68F69F70F71F72F73F74SourceTheoryResetFailed promotion_allowed=0
