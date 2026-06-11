# DG-KAN v21.01 SourceRetentionFU KernelOfficialization 4GPU 执行日志

生成时间：2026-06-04 00:38:36 +0800

本日志只记录真实执行命令、状态和 blocker；未运行项不写成完成。

## 2026-06-04 00:38:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 00:38:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 00:39:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --families D-CHE,D-FOU --batches 32,128,256 --shard-count 4 --shard-index <0..3> --device cuda:<0..3>
```

- status: started
- note: v21.01 Part B fresh full-loop efficiency rerun; v21 runner reused, outputs aliased by finalizer

## 2026-06-04 00:43:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --families D-CHE,D-FOU --batches 32,128,256 --merge-only
```

- status: completed
- note: rows=18

## 2026-06-04 00:45:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/fallback_efficiency_repeat --families D-CHE,D-FOU --batches 256 --profiler-repeats 7 --profiler-warmup 3 --shard-count 4
```

- status: started
- note: CHE-FB/FOU-FB repeat-warmup measurement-noise fallback

## 2026-06-04 00:45:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/fallback_efficiency_repeat --families D-CHE,D-FOU --batches 256 --profiler-repeats 7 --profiler-warmup 3 --merge-only
```

- status: completed
- note: fallback_rows=6

## 2026-06-04 00:46:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-04 00:46:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-04 00:46:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-04 00:46:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-04 00:49:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=198

## 2026-06-04 00:49:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=18 traces=198

## 2026-06-04 00:49:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=18 traces=198

## 2026-06-04 00:50:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=18 traces=198

## 2026-06-04 00:50:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=72 grouped=8

## 2026-06-04 00:52:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 00:52:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 00:52:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 00:52:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 00:53:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 00:53:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 00:54:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 00:55:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 00:55:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=108 grouped=12

## 2026-06-04 01:00 +0800 MLP-F12 schedule-free source-iterate fresh h6400 补跑

原因：`MLP-F12-schedule-free-source-iterate` 已在 v21.01 plan 的 MLP fallback scope 中补入，但上一轮 merge 前该 spec 尚未进入 `--scope mlp` 的 selected ids。为避免 summary 漏掉 schedule-free 分支，单独补跑 3 datasets x 3 seeds 的 fresh 9-row，并重新 merge 全部 source-retention CSV。

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SPECS=MLP-F12-schedule-free-source-iterate

for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx $KAN experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" \
    --scope mlp \
    --device cuda:0 \
    --data-root data \
    --steps 6400 \
    --spec-ids "$SPECS" \
    --run-label v2101_mlp_f12_h6400 \
    --shard-count 4 \
    --shard-index $idx \
    > "$OUT/v21_01_mlp_f12_h6400_shard_${idx}.log" 2>&1 &
done
wait

$KAN experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" \
  --scope mlp \
  --merge-only \
  --run-label v2101_mlp_f12_h6400 \
  > "$OUT/v21_01_mlp_f12_h6400_merge.log" 2>&1
```

## 2026-06-04 01:03 +0800 KAN KSW2/repair h6400 fresh full matrix

目的：按 v21.01 计划继续 KAN 侧 source-retention，重点检查 KSW2 low-degree/low-frequency source-bank、density repair、warmup repair、early boost 与 matched controls 在 h100/h400/h800/h1600/h2400/h3200/h4800/h6400 的完整 retained chain。GPU monitor 用 stop file 在 shard 完成后显式关闭，避免把 monitor 放进同一个 `wait` 导致死锁。

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SNAP=$OUT/v21_01_kan_h6400_gpu_runtime_snapshots.csv
STOP=$OUT/v21_01_kan_h6400_gpu_monitor.stop
rm -f "$STOP"
printf 'timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n' > "$SNAP"
(
  while [ ! -f "$STOP" ]; do
    ts=$(date '+%Y-%m-%d %H:%M:%S %z')
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      while IFS=, read -r idx mem util; do
        idx=${idx// /}; mem=${mem// /}; util=${util// /}
        printf '%s,%s,%s,%s\n' "$ts" "$idx" "$mem" "$util" >> "$SNAP"
      done
    sleep 5
  done
) &
MON=$!

for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx $KAN experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" \
    --scope kan \
    --device cuda:0 \
    --data-root data \
    --steps 6400 \
    --run-label v2101_kan_h6400 \
    --shard-count 4 \
    --shard-index $idx \
    > "$OUT/v21_01_kan_h6400_shard_${idx}.log" 2>&1 &
  PIDS[$idx]=$!
done

STATUS=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then STATUS=1; fi
done
touch "$STOP"
wait "$MON" || true
if [ "$STATUS" -ne 0 ]; then exit "$STATUS"; fi

$KAN experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" \
  --scope kan \
  --merge-only \
  --run-label v2101_kan_h6400 \
  > "$OUT/v21_01_kan_h6400_merge.log" 2>&1
```

## 2026-06-04 01:18 +0800 D-FOU KSW2 productive candidate independent confirmation

原因：KAN h6400 full matrix 中 `D-FOU / FOU-R4-k4-triton-no-materialize / KSW2-lowdegree-lowfreq-source-bank` 出现 h800/h1600/h3200/h4800/h6400 grouped positive，并被标记为 productive candidate。按 v21.01 计划 13.6，不能直接 promotion，必须做 independent init offsets + matched controls。本轮先修复 runner：source/control matching key 加入 `init_seed_offset`，避免 independent rerun 和 offset0 controls 混比。

修改：

- `experiments/run_v21_01_source_retention.py`：`enrich_sources()` 的 best-control key 从 `(carrier, variant, dataset, seed, step)` 改为 `(carrier, variant, dataset, seed, init_seed_offset, step)`。

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  experiments/run_v21_01_source_retention.py \
  experiments/run_v21_01_finalize.py

KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SPECS=KSW2-lowdegree-lowfreq-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
SNAP=$OUT/v21_01_dfou_ksw2_independent_gpu_runtime_snapshots.csv
STOP=$OUT/v21_01_dfou_ksw2_independent_gpu_monitor.stop
rm -f "$STOP"
printf 'timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n' > "$SNAP"
(
  while [ ! -f "$STOP" ]; do
    ts=$(date '+%Y-%m-%d %H:%M:%S %z')
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      while IFS=, read -r idx mem util; do
        idx=${idx// /}; mem=${mem// /}; util=${util// /}
        printf '%s,%s,%s,%s\n' "$ts" "$idx" "$mem" "$util" >> "$SNAP"
      done
    sleep 5
  done
) &
MON=$!

for offset in 100000 200000 300000; do
  for idx in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$idx $KAN experiments/run_v21_01_source_retention.py \
      --out-dir "$OUT" \
      --scope kan \
      --carriers D-FOU \
      --device cuda:0 \
      --data-root data \
      --steps 6400 \
      --spec-ids "$SPECS" \
      --init-seed-offset "$offset" \
      --run-label "v2101_dfou_ksw2_independent_${offset}" \
      --shard-count 4 \
      --shard-index $idx \
      > "$OUT/v21_01_dfou_ksw2_independent_${offset}_shard_${idx}.log" 2>&1 &
    PIDS[$idx]=$!
  done
  STATUS=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then STATUS=1; fi
  done
  if [ "$STATUS" -ne 0 ]; then
    touch "$STOP"
    wait "$MON" || true
    exit "$STATUS"
  fi
done

touch "$STOP"
wait "$MON" || true

$KAN experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" \
  --scope kan \
  --merge-only \
  --run-label v2101_dfou_ksw2_independent \
  > "$OUT/v21_01_dfou_ksw2_independent_merge.log" 2>&1

$KAN experiments/run_v21_01_independent_confirmation.py \
  --out-dir "$OUT" \
  --carrier D-FOU \
  --variant FOU-R4-k4-triton-no-materialize \
  --v21-id KSW2-lowdegree-lowfreq-source-bank \
  --prefix v21_01_dfou_ksw2_independent_confirmation
```

## 2026-06-04 01:35 +0800 Function-space target source writer h6400 full matrix

目的：按 v21.01 计划继续 F3 function-space target 路线，覆盖 loss-cotangent target、low-rank loss target、random matched target、sign-flipped target、corrupted-label target 与 matched controls。若 loss target 不能稳定强过 controls 或不能形成 retained chain，则写成 target-to-retention blocker，不写 promotion。

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SNAP=$OUT/v21_01_target_h6400_gpu_runtime_snapshots.csv
STOP=$OUT/v21_01_target_h6400_gpu_monitor.stop
rm -f "$STOP"
printf 'timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n' > "$SNAP"
(
  while [ ! -f "$STOP" ]; do
    ts=$(date '+%Y-%m-%d %H:%M:%S %z')
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      while IFS=, read -r idx mem util; do
        idx=${idx// /}; mem=${mem// /}; util=${util// /}
        printf '%s,%s,%s,%s\n' "$ts" "$idx" "$mem" "$util" >> "$SNAP"
      done
    sleep 5
  done
) &
MON=$!

for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx $KAN experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" \
    --scope target \
    --device cuda:0 \
    --data-root data \
    --steps 6400 \
    --run-label v2101_target_h6400 \
    --shard-count 4 \
    --shard-index $idx \
    > "$OUT/v21_01_target_h6400_shard_${idx}.log" 2>&1 &
  PIDS[$idx]=$!
done

STATUS=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then STATUS=1; fi
done
touch "$STOP"
wait "$MON" || true
if [ "$STATUS" -ne 0 ]; then exit "$STATUS"; fi

$KAN experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" \
  --scope target \
  --merge-only \
  --run-label v2101_target_h6400 \
  > "$OUT/v21_01_target_h6400_merge.log" 2>&1
```

## 2026-06-04 01:01:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=3

## 2026-06-04 01:01:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=2

## 2026-06-04 01:01:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=2

## 2026-06-04 01:01:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=2

## 2026-06-04 01:01:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=2 traces=22

## 2026-06-04 01:01:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=2 traces=22

## 2026-06-04 01:01:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=2 traces=22

## 2026-06-04 01:01:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=3 traces=33

## 2026-06-04 01:01:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=117 grouped=13

## 2026-06-04 01:04:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=40

## 2026-06-04 01:04:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=40

## 2026-06-04 01:04:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=41

## 2026-06-04 01:04:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=41

## 2026-06-04 01:16:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2
```

- status: completed
- note: rows=40 traces=440

## 2026-06-04 01:16:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3
```

- status: completed
- note: rows=40 traces=440

## 2026-06-04 01:16:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=41 traces=451

## 2026-06-04 01:17:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1
```

- status: completed
- note: rows=41 traces=451

## 2026-06-04 01:17:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=279 grouped=31

## 2026-06-04 01:20:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=12

## 2026-06-04 01:20:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-04 01:20:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-04 01:20:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-04 01:23:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-04 01:23:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-04 01:23:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-04 01:24:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-04 01:24:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-04 01:24:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=12

## 2026-06-04 01:24:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-04 01:24:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-04 01:28:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-04 01:28:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-04 01:28:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-04 01:28:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-04 01:28:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-04 01:28:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-04 01:28:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=12

## 2026-06-04 01:28:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-04 01:32:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-04 01:32:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-04 01:32:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-04 01:32:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-04 01:32:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=414 grouped=31

## 2026-06-04 01:34:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_independent_confirmation.py --carrier D-FOU --v21-id KSW2-lowdegree-lowfreq-source-bank
```

- status: started

## 2026-06-04 01:34:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_independent_confirmation.py --carrier D-FOU --v21-id KSW2-lowdegree-lowfreq-source-bank
```

- status: completed
- note: {'carrier': 'D-FOU', 'basis_repair_variant': 'FOU-R4-k4-triton-no-materialize', 'v21_id': 'KSW2-lowdegree-lowfreq-source-bank', 'offset_groups': 3, 'reproduced_h3200_groups': 0, 'reproduced_h6400_groups': 0, 'independent_confirmation_pass': 0, 'decision': 'independent_confirmation_failed'}

## 2026-06-04 01:34:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_independent_confirmation.py --carrier D-FOU --v21-id KSW2-lowdegree-lowfreq-source-bank
```

- status: started

## 2026-06-04 01:34:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_independent_confirmation.py --carrier D-FOU --v21-id KSW2-lowdegree-lowfreq-source-bank
```

- status: completed
- note: {'carrier': 'D-FOU', 'basis_repair_variant': 'FOU-R4-k4-triton-no-materialize', 'v21_id': 'KSW2-lowdegree-lowfreq-source-bank', 'offset_groups': 3, 'reproduced_h3200_groups': 0, 'reproduced_h6400_groups': 0, 'independent_confirmation_pass': 0, 'decision': 'independent_confirmation_failed'}

## 2026-06-04 01:35:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=40

## 2026-06-04 01:35:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=40

## 2026-06-04 01:35:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=41

## 2026-06-04 01:35:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=41

## 2026-06-04 01:48:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=40 traces=440

## 2026-06-04 01:48:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=40 traces=440

## 2026-06-04 01:48:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=41 traces=451

## 2026-06-04 01:48:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=41 traces=451

## 2026-06-04 01:48:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=576 grouped=41

## 2026-06-04 01:52:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_finalize.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: started

## 2026-06-04 01:52:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_finalize.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: {"route": "R2-LateReboundNoContinuousRetention", "route_detail": "S0_6_pass=1, D-CHE_E1_rows=2, D-FOU_E1_rows=3, D-CHE_S1_rows=0, D-FOU_S1_rows=2, MLP_h3200=0, MLP_h4800=0, KAN_h3200=0, KAN_h4800=0, independent_confirmation_pass=0, late_rebound_groups=17", "CodeRoute": "S0_6-CodeMetricMechanismPassed", "EfficiencyRoute": "D-CHE_E1=2;D-FOU_E1=3;D-CHE_S1=0;D-FOU_S1=2", "FunctionalRoute": "MLP_h3200=0;KAN_h3200=0;KAN_h4800=0;independent=0;late_rebound=17", "promotion_allowed": 0, "S0_6_preflight_pass": 1, "D_CHE_E1_rows": 2, "D_FOU_E1_rows": 3, "D_CHE_S1_rows": 0, "D_FOU_S1_rows": 2, "source_group_rows": 41, "efficiency_rows": 24, "efficiency_fallback_rows": 6, "independent_confirmation_pass": 0, "independent_confirmation_decision": "independent_confirmation_failed", "required_artifact_missing_count": 0}

## 2026-06-04 02:01:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 02:01:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 02:02:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=27

## 2026-06-04 02:02:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=27

## 2026-06-04 02:02:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=27

## 2026-06-04 02:02:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=27

## 2026-06-04 02:11:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=27 traces=297

## 2026-06-04 02:11:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=27 traces=297

## 2026-06-04 02:11:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=27 traces=297

## 2026-06-04 02:11:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=27 traces=297

## 2026-06-04 02:11:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=684 grouped=53

## 2026-06-04 02:15:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 02:15:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 02:16:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-04 02:16:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-04 02:16:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-04 02:16:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-04 02:22:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=18 traces=198

## 2026-06-04 02:22:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=18 traces=198

## 2026-06-04 02:22:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=18 traces=198

## 2026-06-04 02:22:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=18 traces=198

## 2026-06-04 02:22:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=756 grouped=61

## 2026-06-04 02:28:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/fallback_efficiency_repeat2 --families D-CHE --batches 128,256,512 --train-size 1024 --profiler-repeats 11 --profiler-warmup 5 --shard-count 4 --device cuda:0,cuda:1,cuda:2,cuda:3
```

- status: started
- note: D-CHE S1 closure repeat2; higher warmup/repeats and batch 128/256/512

## 2026-06-04 02:29:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/fallback_efficiency_repeat2 --families D-CHE --batches 128,256,512 --train-size 1024 --profiler-repeats 11 --profiler-warmup 5 --shard-count 4 --merge-only
```

- status: completed
- note: D-CHE repeat2 rows=9 E1=6 S1=6 best_forward=0.9440247951264752

## 2026-06-04 02:30:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python <inline combine v21.01 fallback repeat + repeat2 efficiency truth tables>
```

- status: completed
- note: combined fallback rows=15 summary=[{'carrier': 'D-CHE', 'rows': 12, 'exploration_pass_rows': 8, 'official_like_pass_rows': 6, 'best_forward_ratio': 0.9440247951264752, 'best_step_ratio': 0.4269233776176479}, {'carrier': 'D-FOU', 'rows': 3, 'exploration_pass_rows': 3, 'official_like_pass_rows': 2, 'best_forward_ratio': 1.111236890399642, 'best_step_ratio': 0.4717201382550595}]

## 2026-06-04 02:31:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 02:31:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 02:31:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_finalize.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: started

## 2026-06-04 02:31:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_finalize.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: {"route": "R2-LateReboundNoContinuousRetention", "route_detail": "S0_6_pass=1, D-CHE_E1_rows=8, D-FOU_E1_rows=3, D-CHE_S1_rows=6, D-FOU_S1_rows=2, MLP_h3200=0, MLP_h4800=0, KAN_h3200=0, KAN_h4800=0, independent_confirmation_pass=0, late_rebound_groups=26", "CodeRoute": "S0_6-CodeMetricMechanismPassed", "EfficiencyRoute": "D-CHE_E1=8;D-FOU_E1=3;D-CHE_S1=6;D-FOU_S1=2", "FunctionalRoute": "MLP_h3200=0;KAN_h3200=0;KAN_h4800=0;independent=0;late_rebound=26", "promotion_allowed": 0, "S0_6_preflight_pass": 1, "D_CHE_E1_rows": 8, "D_FOU_E1_rows": 3, "D_CHE_S1_rows": 6, "D_FOU_S1_rows": 2, "source_group_rows": 61, "efficiency_rows": 33, "efficiency_fallback_rows": 15, "independent_confirmation_pass": 0, "independent_confirmation_decision": "independent_confirmation_failed", "required_artifact_missing_count": 0}

## 2026-06-04 02:32:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python <inline write v21.01 F9/F10/F11 continuation audit artifacts>
```

- status: completed
- note: F9/F10 grouped=20 row_rollup=10 retained_examples=11

## 2026-06-04 02:36 +0800 F9/F10 exact reproducibility supplement

说明：F9/F10 shard runner 的自动日志只记录了 scope/shard，下面补充 exact spec list 与完整复现命令。所有数据已落在 `results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/`，复盘补充 artifact 已生成。

F9 target-family full rerun:

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SPECS=F9-T2-cross-split-consensus-target,F9-T3-lowdegree-readout-target,F9-T4-weak-stable-target,F9-T5-b1-readout-transfer-target,F9-T6-reservoir-excluding-consensus-target,F9-TCTRL-stable-random-target

for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx $KAN experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" \
    --scope target \
    --device cuda:0 \
    --data-root data \
    --steps 6400 \
    --spec-ids "$SPECS" \
    --run-label v2101_f9_target_family_full \
    --shard-count 4 \
    --shard-index $idx \
    > "$OUT/v21_01_f9_target_family_full_shard_${idx}.log" 2>&1 &
done
wait

$KAN experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" \
  --scope target \
  --merge-only \
  --run-label v2101_f9_target_family_full \
  > "$OUT/v21_01_f9_target_family_full_merge.log" 2>&1
```

F10 B1-consensus transfer full rerun:

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SPECS=F10-T7-b1-cross-split-consensus-transfer,F10-T8-stable-b1-consensus-transfer,F10-T9-b1-weak-stable-transfer,F10-T10-lowdegree-b1-consensus-transfer

for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx $KAN experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" \
    --scope target \
    --device cuda:0 \
    --data-root data \
    --steps 6400 \
    --spec-ids "$SPECS" \
    --run-label v2101_f10_b1_consensus_full \
    --shard-count 4 \
    --shard-index $idx \
    > "$OUT/v21_01_f10_b1_consensus_full_shard_${idx}.log" 2>&1 &
done
wait

$KAN experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" \
  --scope target \
  --merge-only \
  --run-label v2101_f10_b1_consensus_full \
  > "$OUT/v21_01_f10_b1_consensus_full_merge.log" 2>&1
```

F11 D-CHE efficiency repeat2 was run and logged above with `--out-dir .../fallback_efficiency_repeat2 --families D-CHE --batches 128,256,512 --train-size 1024 --profiler-repeats 11 --profiler-warmup 5 --shard-count 4`.

## 2026-06-04 02:36:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python <inline rebuild v21.01 packet after F9/F10/F11 recap append>
```

- status: completed
- note: packet=7025409 bundle=10149953

## 2026-06-04 02:36:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python <inline final v21.01 verification + ps process check>
```

- status: completed
- note: route=R2-LateReboundNoContinuousRetention promotion_allowed=0 D-CHE_S1=6 D-FOU_S1=2 source_group_rows=61 missing=0 packet=7025409 bundle=10149953; process_check=no persistent v21.01 training/monitor process

## 2026-06-04 02:39:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --carriers D-FOU --spec-ids F9-T2-cross-split-consensus-target,F9-T5-b1-readout-transfer-target,F10-T7-b1-cross-split-consensus-transfer,F9-TCTRL-stable-random-target --init-seed-offset {100000,200000,300000} --run-label v2101_f12_late_rebound_independent_<offset> --shard-count 4 --steps 6400
```

- status: started
- note: F12 late-rebound independent classification: D-FOU strongest F9/F10 targets plus stable-random control, x3 offsets

## 2026-06-04 02:39:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 02:39:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 02:39:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 02:39:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 02:42:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 02:42:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 02:42:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 02:42:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 02:42:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --carriers D-FOU --run-label v2101_f12_late_rebound_independent_100000
```

- status: completed
- note: rows=36 expected; shards=4

## 2026-06-04 02:42:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 02:42:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 02:42:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 02:42:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 02:46:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 02:46:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 02:46:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 02:46:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 02:46:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --carriers D-FOU --run-label v2101_f12_late_rebound_independent_200000
```

- status: completed
- note: rows=36 expected; shards=4

## 2026-06-04 02:46:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 02:46:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 02:46:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 02:46:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=9

## 2026-06-04 02:49:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 02:49:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 02:49:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 02:49:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=9 traces=99

## 2026-06-04 02:49:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --carriers D-FOU --run-label v2101_f12_late_rebound_independent_300000
```

- status: completed
- note: rows=36 expected; shards=4

## 2026-06-04 02:49:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=864 grouped=61

## 2026-06-04 02:49:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only --run-label v2101_f12_late_rebound_independent
```

- status: completed
- note: merged rows=864 grouped=61 F12_groups=4

## 2026-06-04 02:51:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python <inline summarize F12 late rebound independent rerun>
```

- status: completed
- note: independent_rows=108 productive_groups=0 late_groups=3 retained_examples=20 control_h6400_late=1

## 2026-06-04 02:51:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_finalize.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: started

## 2026-06-04 02:51:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_finalize.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: {"route": "R2-LateReboundNoContinuousRetention", "route_detail": "S0_6_pass=1, D-CHE_E1_rows=8, D-FOU_E1_rows=3, D-CHE_S1_rows=6, D-FOU_S1_rows=2, MLP_h3200=0, MLP_h4800=0, KAN_h3200=0, KAN_h4800=0, independent_confirmation_pass=0, late_rebound_groups=26", "CodeRoute": "S0_6-CodeMetricMechanismPassed", "EfficiencyRoute": "D-CHE_E1=8;D-FOU_E1=3;D-CHE_S1=6;D-FOU_S1=2", "FunctionalRoute": "MLP_h3200=0;KAN_h3200=0;KAN_h4800=0;independent=0;late_rebound=26", "promotion_allowed": 0, "S0_6_preflight_pass": 1, "D_CHE_E1_rows": 8, "D_FOU_E1_rows": 3, "D_CHE_S1_rows": 6, "D_FOU_S1_rows": 2, "source_group_rows": 61, "efficiency_rows": 33, "efficiency_fallback_rows": 15, "independent_confirmation_pass": 0, "independent_confirmation_decision": "independent_confirmation_failed", "required_artifact_missing_count": 0}

## 2026-06-04 02:52 +0800 F12 exact reproducibility supplement

目的：按计划 13.6 对 F9/F10 strongest D-FOU late-rebound specs 做 x3 independent rerun，并加入 stable-random matched target control。自动日志已记录 shard start/complete；下面补充完整复现命令。

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SPECS=F9-T2-cross-split-consensus-target,F9-T5-b1-readout-transfer-target,F10-T7-b1-cross-split-consensus-transfer,F9-TCTRL-stable-random-target

for offset in 100000 200000 300000; do
  for idx in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$idx $KAN experiments/run_v21_01_source_retention.py \
      --out-dir "$OUT" \
      --scope target \
      --carriers D-FOU \
      --device cuda:0 \
      --data-root data \
      --steps 6400 \
      --spec-ids "$SPECS" \
      --init-seed-offset "$offset" \
      --run-label "v2101_f12_late_rebound_independent_${offset}" \
      --shard-count 4 \
      --shard-index $idx \
      > "$OUT/v21_01_f12_late_rebound_independent_${offset}_shard_${idx}.log" 2>&1 &
  done
  wait
done

$KAN experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" \
  --scope target \
  --merge-only \
  --run-label v2101_f12_late_rebound_independent \
  > "$OUT/v21_01_f12_late_rebound_independent_merge.log" 2>&1
```

F12 summary artifacts:

- `v21_01_f12_late_rebound_independent_per_offset.csv`
- `v21_01_f12_late_rebound_independent_aggregate.csv`
- `v21_01_f12_late_rebound_independent_retained_row_examples.csv`
- `v21_01_f12_late_rebound_independent_decision.csv`

## 2026-06-04 02:53:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python <inline rebuild v21.01 packet after F12 recap append>
```

- status: completed
- note: packet=7318873 bundle=11017569

## 2026-06-04 03:00:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f13_late_rebound_anatomy.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=108 delayed_groups=3 continuous_groups=0 control_h6400_positive_rows=14 train_only_predictor_pass=0 packet=5417047 bundle=9114146

## 2026-06-04 03:01:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f13_late_rebound_anatomy.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=108 delayed_groups=3 continuous_groups=0 control_h6400_positive_rows=14 train_only_predictor_pass=0 packet=6112756 bundle=9816124

## 2026-06-04 03:02:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python <inline final F13 artifact/manifest/packet verification>
ps -eo pid,ppid,stat,etime,cmd | rg 'run_v21_01|run_v21_efficiency|nvidia-smi|gpu_monitor' || true
```

- status: completed
- note: source_matrix_rows=864 source_group_rows=61 F12_aggregate_rows=4 F13_summary_rows=4 F13_row_vs_control_rows=81 F13_stratification_rows=18 F13_predictor_rows=8 F13_decision_rows=1 required_manifest_rows=26 required_missing=0 packet_refreshed=1 bundle_refreshed=1 packet_contains_F12_F13=1 route=R2-LateReboundNoContinuousRetention-F13DelayedMigrationControlDrift promotion_allowed=0; process_check_only_matched_verification_commands

## 2026-06-04 03:04:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python <inline rebuild v21.01 packet/bundle from v21_01_required_artifact_manifest.csv after final log/recap append>
```

- status: completed
- note: packet_refreshed=1 bundle_refreshed=1; exact byte sizes intentionally left to file stat to avoid self-referential zip-size churn

## 2026-06-04 03:13:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f14_early_source_selector_audit.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: selector_pass=0 near_candidates=1 best=F3-T1-loss-cotangent-target f15_recommended=1

## 2026-06-04 03:14:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-04 03:14:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-04 03:14:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-04 03:14:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-04 03:19:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=13 traces=143

## 2026-06-04 03:19:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=13 traces=143

## 2026-06-04 03:19:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=14 traces=154

## 2026-06-04 03:19:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=14 traces=154

## 2026-06-04 03:19:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-04 03:19:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-04 03:19:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-04 03:19:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-04 03:24:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=13 traces=143

## 2026-06-04 03:24:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=13 traces=143

## 2026-06-04 03:25:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=14 traces=154

## 2026-06-04 03:25:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=14 traces=154

## 2026-06-04 03:25:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-04 03:25:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-04 03:25:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-04 03:25:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-04 03:29:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=13 traces=143

## 2026-06-04 03:29:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=13 traces=143

## 2026-06-04 03:30:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=14 traces=154

## 2026-06-04 03:30:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=14 traces=154

## 2026-06-04 03:30:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1026 grouped=61

## 2026-06-04 03:34:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f15_loss_target_independent_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=162 blocked=0 candidate_h800=-0.08055646772737857 continuous_groups=0 promotion=0

## 2026-06-04 03:35:49 +0800

F15 independent confirmation exact reproduction command. This is the expanded command that produced the shard rows merged at 03:30:18.

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SPECS=F3-T1-loss-cotangent-target,F3-T1c-lowrank-loss-target,F3-T5-random-matched-target,F3-T6-sign-flipped-target,F3-T7-corrupted-label-target,F9-TCTRL-stable-random-target
for offset in 100000 200000 300000; do
  for idx in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$idx $KAN experiments/run_v21_01_source_retention.py \
      --out-dir "$OUT" \
      --scope target \
      --carriers D-FOU \
      --device cuda:0 \
      --data-root data \
      --steps 6400 \
      --spec-ids "$SPECS" \
      --init-seed-offset "$offset" \
      --run-label "v2101_f15_loss_target_independent_${offset}" \
      --shard-count 4 \
      --shard-index $idx \
      > "$OUT/v21_01_f15_loss_target_independent_${offset}_shard_${idx}.log" 2>&1 &
  done
  wait
done
$KAN experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" \
  --scope target \
  --merge-only \
  --run-label v2101_f15_loss_target_independent \
  > "$OUT/v21_01_f15_loss_target_independent_merge.log" 2>&1
```

- status: completed
- note: offsets=100000/200000/300000; carriers=D-FOU; specs=6; total_rows=162; blocked_rows=0; merged_matrix_rows=1026

## 2026-06-04 03:38:10 +0800

Final verification and packet refresh. The packet/bundle refresh command is rerun once after this log entry is written so the final archives include the latest execution log and recap.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import json
from pathlib import Path
from experiments.run_v21_01_common import build_packet, read_rows
out=Path('results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01')
manifest=read_rows(out/'v21_01_required_artifact_manifest.csv')
build_packet(out, [r.get('artifact','') for r in manifest if r.get('artifact')])
missing=[r['artifact'] for r in manifest if str(r.get('required','1'))!='0' and not (out/r['artifact']).exists()]
route=json.loads((out/'v21_01_route_decision.json').read_text())
print('manifest_rows', len(manifest))
print('missing_count', len(missing))
print('route', route.get('route'))
print('promotion_allowed', route.get('promotion_allowed'))
print('f15_rows', route.get('f15_rows'))
print('f15_blocked_rows', route.get('f15_blocked_rows'))
PY
ps -eo pid,ppid,stat,etime,cmd | rg 'run_v21_01|v21_01|gpu_monitor|nvidia-smi' | rg -v 'rg|ps -eo' || true
```

- status: completed
- note: manifest_rows=33; required_missing=0; route=R2-LateReboundNoContinuousRetention-F15LossTargetNoEarlyClosure; promotion_allowed=0; f15_rows=162; f15_blocked_rows=0; residual_training_or_monitor_processes=0

## 2026-06-04 03:46:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f16_source_target_theory_audit.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: target_rows=657 continuous_groups=0 legal_predictor_pass=1 closed=0 promotion=0

## 2026-06-04 03:50:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f16_source_target_theory_audit.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: target_rows=657 continuous_groups=0 legal_predictor_pass=1 closed=0 promotion=0

## 2026-06-04 03:50:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f17_train_loss_selector_holdout.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: threshold=1.68036288022995 holdout_selected=40 holdout_h800=-0.08922969549894333 actionable=0 promotion=0

## 2026-06-04 03:51:24 +0800

Final verification after F16/F17. Packet/bundle are refreshed again after this entry so the final archives include the newest logs and recap.

```bash
python - <<'PY'
import csv,json
from pathlib import Path
out=Path('results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01')
manifest=list(csv.DictReader((out/'v21_01_required_artifact_manifest.csv').open()))
missing=[r['artifact'] for r in manifest if str(r.get('required','1'))!='0' and not (out/r['artifact']).exists()]
print('manifest_rows',len(manifest),'missing_count',len(missing))
route=json.loads((out/'v21_01_route_decision.json').read_text())
for k in ['route','promotion_allowed','f16_legal_predictor_pass','f17_selector_actionable','new_source_target_theory_required']:
 print(k,route.get(k))
for name in ['v21_01_source_retention_matrix.csv','v21_01_f16_route_closure_decision.csv','v21_01_f17_train_loss_selector_decision.csv']:
 with (out/name).open() as f: print(name,sum(1 for _ in f)-1)
PY
ps -eo pid,ppid,stat,etime,cmd | rg 'run_v21_01|v21_01|gpu_monitor|nvidia-smi' | rg -v 'rg|ps -eo' || true
```

- status: completed
- note: manifest_rows=41; required_missing=0; route=R2-LateReboundNoContinuousRetention-F17TrainLossSelectorNotActionable; promotion_allowed=0; f16_legal_predictor_pass=1; f17_selector_actionable=0; residual_training_or_monitor_processes=0

## 2026-06-04 04:03:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=23

## 2026-06-04 04:03:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=22

## 2026-06-04 04:03:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=22

## 2026-06-04 04:03:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=23

## 2026-06-04 04:05:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=22 traces=154

## 2026-06-04 04:05:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=22 traces=154

## 2026-06-04 04:05:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=23 traces=161

## 2026-06-04 04:05:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=23 traces=161

## 2026-06-04 04:05:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1116 grouped=63

## 2026-06-04 04:05:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f18_contrastive_target_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=90 blocked=0 candidate_early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 04:09:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=22

## 2026-06-04 04:09:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=23

## 2026-06-04 04:09:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=22

## 2026-06-04 04:09:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=23

## 2026-06-04 04:12:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=22 traces=154

## 2026-06-04 04:12:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=22 traces=154

## 2026-06-04 04:12:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=23 traces=161

## 2026-06-04 04:12:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=23 traces=161

## 2026-06-04 04:12:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1206 grouped=65

## 2026-06-04 04:12:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f19_margin_target_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=90 blocked=0 candidate_early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 04:12:59 +0800

F18/F19 smoke exact reproduction commands. These are the expanded commands that produced the shard rows merged above.

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01

SPECS=F18-T11-contrastive-loss-random-orthogonal,F18-T12-b1-contrastive-loss-random-orthogonal,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F3-T6-sign-flipped-target,F3-T7-corrupted-label-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx $KAN experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" \
    --scope target \
    --carriers D-FOU \
    --device cuda:0 \
    --data-root data \
    --steps 1600 \
    --spec-ids "$SPECS" \
    --run-label v2101_f18_contrastive_target_smoke \
    --shard-count 4 \
    --shard-index $idx \
    > "$OUT/v21_01_f18_contrastive_target_smoke_shard_${idx}.log" 2>&1 &
done
wait
$KAN experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" \
  --scope target \
  --merge-only \
  --run-label v2101_f18_contrastive_target_smoke \
  > "$OUT/v21_01_f18_contrastive_target_smoke_merge.log" 2>&1
$KAN experiments/run_v21_01_f18_contrastive_target_summary.py --out-dir "$OUT"

SPECS=F19-T13-topwrong-margin-target,F19-T14-b1-topwrong-margin-transfer,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,F3-T6-sign-flipped-target,F3-T7-corrupted-label-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx $KAN experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" \
    --scope target \
    --carriers D-FOU \
    --device cuda:0 \
    --data-root data \
    --steps 1600 \
    --spec-ids "$SPECS" \
    --run-label v2101_f19_margin_target_smoke \
    --shard-count 4 \
    --shard-index $idx \
    > "$OUT/v21_01_f19_margin_target_smoke_shard_${idx}.log" 2>&1 &
done
wait
$KAN experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" \
  --scope target \
  --merge-only \
  --run-label v2101_f19_margin_target_smoke \
  > "$OUT/v21_01_f19_margin_target_smoke_merge.log" 2>&1
$KAN experiments/run_v21_01_f19_margin_target_summary.py --out-dir "$OUT"
```

- status: completed
- note: F18 rows=90 blocked=0 full_recommended=0; F19 rows=90 blocked=0 full_recommended=0

## 2026-06-04 04:14:23 +0800

Final verification after F18/F19. Packet/bundle are refreshed after the F18/F19 exact command block is written so the final archives include the newest execution log and recap.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv, json
from pathlib import Path
from experiments.run_v21_01_common import build_packet, read_rows
out=Path('results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01')
manifest=read_rows(out/'v21_01_required_artifact_manifest.csv')
build_packet(out, [r.get('artifact','') for r in manifest if r.get('artifact')])
missing=[r['artifact'] for r in manifest if str(r.get('required','1'))!='0' and not (out/r['artifact']).exists()]
route=json.loads((out/'v21_01_route_decision.json').read_text())
print('manifest_rows', len(manifest))
print('missing_count', len(missing))
print('route', route.get('route'))
print('promotion_allowed', route.get('promotion_allowed'))
for name in ['v21_01_source_retention_matrix.csv','v21_01_f18_contrastive_target_decision.csv','v21_01_f19_margin_target_decision.csv']:
    with (out/name).open() as f:
        print(name, sum(1 for _ in f)-1)
for name in ['v21_01_code_review_packet.zip','v21_01_results_bundle.zip']:
    p=out/name
    print(name, p.stat().st_size if p.exists() else 'missing')
PY
ps -eo pid,ppid,stat,etime,cmd | rg 'run_v21_01|v21_01|gpu_monitor|nvidia-smi' | rg -v 'rg|ps -eo' || true
```

- status: completed
- note: manifest_rows=47; required_missing=0; route=R2-LateReboundNoContinuousRetention-F19MarginTargetSmokeNoEarlySignal; promotion_allowed=0; source_retention_rows=1206; F18_decision_rows=1; F19_decision_rows=1; packet_refreshed=1; bundle_refreshed=1; residual_training_or_monitor_processes=0

## 2026-06-04 04:31:36 +0800

F20 margin split-consensus source-channel estimator exact reproduction commands. This continuation follows the v21.01 plan branch after F18/F19 target semantics failed: stop function-space target/actuation tuning and test a new train-only source-channel estimator.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v21_common.py \
  experiments/run_v21_01_source_retention.py \
  experiments/run_v21_01_s06_truth_gate.py \
  experiments/run_v21_01_f20_margin_source_summary.py
```

- status: completed
- note: py_compile passed for M68/F20 implementation and summary runner.

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
mkdir -p "$OUT"
SPECS=MLP-F20-margin-split-consensus-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx "$KAN" experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" --scope mlp --carriers MLP --device cuda:0 --data-root data --steps 1600 \
    --spec-ids "$SPECS" --run-label v2101_f20_margin_source_smoke_mlp --shard-count 4 --shard-index $idx \
    > "$OUT/v21_01_f20_margin_source_smoke_mlp_shard_${idx}.log" 2>&1 &
done
wait
```

- status: completed
- note: MLP F20 smoke completed, 45 matrix rows after merge scope.

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SPECS=KSW4-margin-split-consensus-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx "$KAN" experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" --scope kan --carriers D-FOU --device cuda:0 --data-root data --steps 1600 \
    --spec-ids "$SPECS" --run-label v2101_f20_margin_source_smoke_kan --shard-count 4 --shard-index $idx \
    > "$OUT/v21_01_f20_margin_source_smoke_kan_shard_${idx}.log" 2>&1 &
done
wait
```

- status: completed
- note: D-FOU F20 smoke completed, 45 matrix rows after merge scope.

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
"$KAN" experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" --scope kan --merge-only --run-label v2101_f20_margin_source_smoke \
  > "$OUT/v21_01_f20_margin_source_smoke_merge.log" 2>&1
"$KAN" experiments/run_v21_01_f20_margin_source_summary.py --out-dir "$OUT" \
  > "$OUT/v21_01_f20_margin_source_summary.log" 2>&1
```

- status: completed
- note: F20 rows=90; blocked=0; candidate early-chain groups=0; candidate productive h3200 groups=0; full_recommended=0; route=R2-LateReboundNoContinuousRetention-F20MarginSourceSmokeNoEarlySignal.

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py \
  --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 \
  --check import_closure \
  > results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_f20_s06_import_closure.log 2>&1
```

- status: completed
- note: S0.6 import closure/compileall passed after F20; missing source files=0; import errors=0.

## 2026-06-04 04:33:11 +0800

Final verification after F20. Packet/bundle are refreshed after this block is written so the final archives include the newest execution log and recap.

```bash
python - <<'PY'
from pathlib import Path
from experiments.run_v21_01_common import build_packet, read_json, read_rows

out = Path("results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01")
manifest = read_rows(out / "v21_01_required_artifact_manifest.csv")
missing = [r for r in manifest if str(r.get("required")) == "1" and str(r.get("exists")) != "1"]
route = read_json(out / "v21_01_route_decision.json")
f20 = read_rows(out / "v21_01_f20_margin_source_decision.csv")
source_rows = read_rows(out / "v21_01_source_retention_matrix.csv")
build_packet(out, [r.get("artifact", "") for r in manifest if r.get("artifact")])
print({
    "manifest_rows": len(manifest),
    "required_missing": len(missing),
    "route": route.get("route"),
    "promotion_allowed": route.get("promotion_allowed"),
    "source_retention_rows": len(source_rows),
    "f20_decision": f20,
})
PY
ps -eo pid,ppid,stat,cmd | rg 'run_v21_01_source_retention|gpu_monitor|nvidia-smi' || true
```

- status: completed
- note: manifest_rows=50; required_missing=0; route=R2-LateReboundNoContinuousRetention-F20MarginSourceSmokeNoEarlySignal; promotion_allowed=0; source_retention_rows=1296; F20_rows=90; F20_full_recommended=0; S0.6_import_closure_pass=1; residual_training_or_monitor_processes=0; packet_refreshed=1; bundle_refreshed=1.

## 2026-06-04 04:28:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=11

## 2026-06-04 04:28:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=12

## 2026-06-04 04:28:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=11

## 2026-06-04 04:28:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=11

## 2026-06-04 04:28:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=77

## 2026-06-04 04:28:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=77

## 2026-06-04 04:28:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=77

## 2026-06-04 04:28:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=84

## 2026-06-04 04:28:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=12

## 2026-06-04 04:28:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=11

## 2026-06-04 04:28:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=11

## 2026-06-04 04:28:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=11

## 2026-06-04 04:29:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3
```

- status: completed
- note: rows=11 traces=77

## 2026-06-04 04:29:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1
```

- status: completed
- note: rows=11 traces=77

## 2026-06-04 04:29:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2
```

- status: completed
- note: rows=11 traces=77

## 2026-06-04 04:29:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=12 traces=84

## 2026-06-04 04:30:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=1296 grouped=67

## 2026-06-04 04:30:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f20_margin_source_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=90 blocked=0 candidate_early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 04:31:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check import_closure --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 04:31:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check import_closure
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 04:40:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=20

## 2026-06-04 04:40:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=20

## 2026-06-04 04:40:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=20

## 2026-06-04 04:40:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=21

## 2026-06-04 04:44:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=20 traces=220

## 2026-06-04 04:44:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=20 traces=220

## 2026-06-04 04:44:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=20 traces=220

## 2026-06-04 04:44:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=21 traces=231

## 2026-06-04 04:44:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=1377 grouped=68

## 2026-06-04 04:44:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f21_optimizer_dynamics_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=81 blocked=0 productive_h3200=0 productive_h4800=0 adamw_washout_supported=0 promotion=0

## 2026-06-04 04:54:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v21_01_s06_truth_gate.py experiments/run_v21_01_f22_rotated_cautious_summary.py
```

- status: completed
- note: F22 implementation compile passed; no experiment data generated by this command.

## 2026-06-04 04:54:46 +0800

```bash
set -euo pipefail
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
mkdir -p "$OUT"
SPECS=MLP-F22-rotated-cautious-matrix-source,MLP-F1-M2-strong-source,MLP-F7-M2-source-with-matrix-block-retention,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx "$KAN" experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" --scope mlp --carriers MLP --device cuda:0 --data-root data --steps 1600 \
    --spec-ids "$SPECS" --run-label v2101_f22_rotated_cautious_smoke --shard-count 4 --shard-index $idx \
    > "$OUT/v21_01_f22_rotated_cautious_smoke_shard_${idx}.log" 2>&1 &
done
wait
```

- status: completed
- note: F22 smoke shards completed; rows expected=54.

## 2026-06-04 04:54:46 +0800

```bash
set -euo pipefail
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
"$KAN" experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --merge-only --run-label v2101_f22_rotated_cautious_smoke > "$OUT/v21_01_f22_rotated_cautious_smoke_merge.log" 2>&1
"$KAN" experiments/run_v21_01_f22_rotated_cautious_summary.py --out-dir "$OUT" > "$OUT/v21_01_f22_rotated_cautious_summary_smoke.log" 2>&1
```

- status: completed
- note: F22 rows=54 blocked=0 candidate_early_chain=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 04:53:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=14

## 2026-06-04 04:53:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=13

## 2026-06-04 04:53:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=13

## 2026-06-04 04:53:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=14

## 2026-06-04 04:53:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=14 traces=98

## 2026-06-04 04:54:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=13 traces=91

## 2026-06-04 04:54:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=14 traces=98

## 2026-06-04 04:54:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=13 traces=91

## 2026-06-04 04:54:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=1431 grouped=69

## 2026-06-04 04:54:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f22_rotated_cautious_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=54 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 05:00:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=13

## 2026-06-04 05:00:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=14

## 2026-06-04 05:00:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=14

## 2026-06-04 05:00:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=13

## 2026-06-04 05:00:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=13 traces=91

## 2026-06-04 05:00:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=14 traces=98

## 2026-06-04 05:00:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=14 traces=98

## 2026-06-04 05:01:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=13 traces=91

## 2026-06-04 05:01:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=1485 grouped=70

## 2026-06-04 05:01:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f23_lookahead_gate_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=54 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 05:01:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check import_closure --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 05:01:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check import_closure
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 05:02:15 +0800

```bash
set -euo pipefail
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
mkdir -p "$OUT"
SPECS=MLP-F21-adamw-primary-fu-residual,MLP-F1-M2-strong-source,MLP-F2-M15-weak-stable,MLP-F11-dual-timescale-retention-warm1200,MLP-F12-schedule-free-source-iterate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx "$KAN" experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" --scope mlp --carriers MLP --device cuda:0 --data-root data --steps 6400 \
    --spec-ids "$SPECS" --run-label v2101_f21_optimizer_dynamics_full --shard-count 4 --shard-index $idx \
    > "$OUT/v21_01_f21_optimizer_dynamics_full_shard_${idx}.log" 2>&1 &
done
wait
"$KAN" experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --merge-only --run-label v2101_f21_optimizer_dynamics_full > "$OUT/v21_01_f21_optimizer_dynamics_full_merge.log" 2>&1
"$KAN" experiments/run_v21_01_f21_optimizer_dynamics_summary.py --out-dir "$OUT" > "$OUT/v21_01_f21_optimizer_dynamics_summary.log" 2>&1
```

- status: completed
- note: F21 exact reproduction command block; actual run completed earlier at 04:44 with rows=81 blocked=0 productive_h3200=0.

## 2026-06-04 05:02:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v21_01_s06_truth_gate.py experiments/run_v21_01_f22_rotated_cautious_summary.py experiments/run_v21_01_f23_lookahead_gate_summary.py
```

- status: completed
- note: F23 implementation compile passed; no experiment data generated by this command.

## 2026-06-04 05:02:15 +0800

```bash
set -euo pipefail
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
mkdir -p "$OUT"
SPECS=MLP-F23-source-vs-sgd-lookahead-gate,MLP-F22-rotated-cautious-matrix-source,MLP-F1-M2-strong-source,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx "$KAN" experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" --scope mlp --carriers MLP --device cuda:0 --data-root data --steps 1600 \
    --spec-ids "$SPECS" --run-label v2101_f23_lookahead_gate_smoke --shard-count 4 --shard-index $idx \
    > "$OUT/v21_01_f23_lookahead_gate_smoke_shard_${idx}.log" 2>&1 &
done
wait
"$KAN" experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --merge-only --run-label v2101_f23_lookahead_gate_smoke > "$OUT/v21_01_f23_lookahead_gate_smoke_merge.log" 2>&1
"$KAN" experiments/run_v21_01_f23_lookahead_gate_summary.py --out-dir "$OUT" > "$OUT/v21_01_f23_lookahead_gate_summary_smoke.log" 2>&1
```

- status: completed
- note: F23 rows=54 blocked=0 candidate_early_chain=0 productive_h3200=0 full_recommended=0 promotion=0.

## 2026-06-04 05:02:15 +0800

```bash
set -euo pipefail
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
"$KAN" experiments/run_v21_01_s06_truth_gate.py --check import_closure --out-dir "$OUT" --device cuda:0 > "$OUT/v21_01_s06_import_closure_final.log" 2>&1
cat "$OUT/v21_01_route_decision.json"
awk -F, 'NR>1 && $3==1 && $2!=1 {missing++} END {print missing+0}' "$OUT/v21_01_required_artifact_manifest.csv"
awk 'END {print NR-1}' "$OUT/v21_01_required_artifact_manifest.csv"
ls -lh "$OUT"/v21_01_code_review_packet.zip "$OUT"/v21_01_results_bundle.zip
ps -u "$USER" -o pid,stat,etime,cmd | rg 'run_v21_01|v2101|gpu_monitor|nvidia-smi' || true
```

- status: completed
- note: final route=R2-LateReboundNoContinuousRetention-F23LookaheadGateSmokeNoEarlyChain; promotion_allowed=0; required missing=0; manifest rows=59; packet=5.2M; bundle=11M; residual process check only matched the check command/rg itself.

## 2026-06-04 05:17:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=27

## 2026-06-04 05:17:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=27

## 2026-06-04 05:17:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=27

## 2026-06-04 05:17:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=27

## 2026-06-04 05:19:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=27 traces=189

## 2026-06-04 05:19:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=27 traces=189

## 2026-06-04 05:24:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=27 traces=189

## 2026-06-04 05:24:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=27 traces=189

## 2026-06-04 05:24:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1593 grouped=72

## 2026-06-04 05:24:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f24_b1_lookahead_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=108 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 05:28:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=27

## 2026-06-04 05:28:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=27

## 2026-06-04 05:28:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=27

## 2026-06-04 05:28:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=27

## 2026-06-04 05:30:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=27 traces=189

## 2026-06-04 05:30:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=27 traces=189

## 2026-06-04 05:34:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=27 traces=189

## 2026-06-04 05:34:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=27 traces=189

## 2026-06-04 05:34:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1593 grouped=72

## 2026-06-04 05:34:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f24_b1_lookahead_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=108 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 05:36:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=27

## 2026-06-04 05:36:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=27

## 2026-06-04 05:36:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=27

## 2026-06-04 05:36:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=27

## 2026-06-04 05:38:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=27 traces=189

## 2026-06-04 05:38:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=27 traces=189

## 2026-06-04 05:43:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=27 traces=189

## 2026-06-04 05:43:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=27 traces=189

## 2026-06-04 05:43:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1593 grouped=72

## 2026-06-04 05:43:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f24_b1_lookahead_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=108 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 05:49:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=32

## 2026-06-04 05:49:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=31

## 2026-06-04 05:49:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=32

## 2026-06-04 05:49:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=31

## 2026-06-04 05:52:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=31 traces=217

## 2026-06-04 05:52:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=31 traces=217

## 2026-06-04 05:52:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=32 traces=224

## 2026-06-04 05:52:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=32 traces=224

## 2026-06-04 05:52:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1719 grouped=74

## 2026-06-04 05:52:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f25_migration_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=126 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 05:53:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check mechanism_contracts --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 05:53:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check import_closure --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 05:53:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check mechanism_contracts
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 05:53:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check import_closure
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 05:55:00 +0800 手工审计补充：F24/F25 精确复现命令

### 编译检查

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v21_common.py \
  experiments/run_v21_01_source_retention.py \
  experiments/run_v21_01_s06_truth_gate.py \
  experiments/run_v21_01_f24_b1_lookahead_summary.py \
  experiments/run_v21_01_f25_migration_summary.py
```

- status: completed
- note: py_compile 通过。

### F24 final smoke command

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SPECS=F24-B1-consensus-lookahead-gated-transfer,F10-T7-b1-cross-split-consensus-transfer,KSW2-lowdegree-lowfreq-source-bank,F3-T5-random-matched-target,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx "$KAN" experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" \
    --scope target \
    --carriers D-CHE,D-FOU \
    --device cuda:0 \
    --data-root data \
    --steps 1600 \
    --spec-ids "$SPECS" \
    --run-label v2101_f24_b1_lookahead_smoke \
    --shard-count 4 \
    --shard-index "$idx" \
    > "$OUT/v21_01_f24_b1_lookahead_smoke_targetonly_shard_${idx}.log" 2>&1 &
done
wait
"$KAN" experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" \
  --scope target \
  --merge-only \
  --run-label v2101_f24_b1_lookahead_smoke \
  > "$OUT/v21_01_f24_b1_lookahead_smoke_targetonly_merge.log" 2>&1
"$KAN" experiments/run_v21_01_f24_b1_lookahead_summary.py \
  --out-dir "$OUT" \
  > "$OUT/v21_01_f24_b1_lookahead_summary_targetonly_smoke.log" 2>&1
```

- status: completed
- note: F24 rows=108, blocked=0, early-chain groups=0, full escalation=0。
- audit note: F24 first smoke exposed missing diagnostics after gate accept; second smoke exposed wrong accept semantics (`SGD + target residual`). Final smoke above used target-only commit on accept and SGD fallback on reject.

### F25 smoke command

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SPECS=F25-loss-warm-to-b1-consensus-migration,F3-T1-loss-cotangent-target,F10-T7-b1-cross-split-consensus-transfer,F3-T5-random-matched-target,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx "$KAN" experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" \
    --scope target \
    --carriers D-CHE,D-FOU \
    --device cuda:0 \
    --data-root data \
    --steps 1600 \
    --spec-ids "$SPECS" \
    --run-label v2101_f25_migration_smoke \
    --shard-count 4 \
    --shard-index "$idx" \
    > "$OUT/v21_01_f25_migration_smoke_shard_${idx}.log" 2>&1 &
done
wait
"$KAN" experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" \
  --scope target \
  --merge-only \
  --run-label v2101_f25_migration_smoke \
  > "$OUT/v21_01_f25_migration_smoke_merge.log" 2>&1
"$KAN" experiments/run_v21_01_f25_migration_summary.py \
  --out-dir "$OUT" \
  > "$OUT/v21_01_f25_migration_summary_smoke.log" 2>&1
```

- status: completed
- note: F25 rows=126, blocked=0, early-chain groups=0, full escalation=0。

### 最终审计命令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py \
  --check import_closure \
  --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 \
  --device cuda:0 \
  > results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_s06_import_closure_after_f25.log 2>&1

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py \
  --check mechanism_contracts \
  --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 \
  --device cuda:0 \
  > results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_s06_mechanism_contracts_after_f25.log 2>&1

awk -F, 'NR>1 && $3==1 && $2!=1 {missing++} END {print missing+0}' \
  results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_required_artifact_manifest.csv
awk 'END {print NR-1}' \
  results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_required_artifact_manifest.csv
ls -lh \
  results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_code_review_packet.zip \
  results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_results_bundle.zip
ps -u "$USER" -o pid,stat,etime,cmd | rg 'run_v21_01|v2101|gpu_monitor|nvidia-smi' || true
```

- status: completed
- note: S0.6 import_closure=1, mechanism_contracts=1, required missing=0, manifest rows=65, no persistent v21.01 training process.

### 最终 packet/bundle 重建

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from pathlib import Path
from experiments.run_v21_01_common import build_packet, read_rows, int_flag
out = Path('results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01')
manifest = read_rows(out / 'v21_01_required_artifact_manifest.csv')
required = [str(r.get('artifact')) for r in manifest if int_flag(r.get('required', 1))]
build_packet(out, required)
print('rebuilt', len(required))
PY
awk -F, 'NR>1 && $3==1 && $2!=1 {missing++} END {print missing+0}' \
  results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_required_artifact_manifest.csv
awk 'END {print NR-1}' \
  results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_required_artifact_manifest.csv
ls -lh \
  results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_code_review_packet.zip \
  results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_results_bundle.zip
ps -u "$USER" -o pid,stat,etime,cmd | rg 'run_v21_01|v2101|gpu_monitor|nvidia-smi' || true
```

- status: completed
- note: rebuilt required artifacts=65; required missing=0; manifest rows=65; final packet=6.8M; final bundle=14M; process check only matched the check command/rg itself.

## 2026-06-04 06:11:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check mechanism_contracts --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 06:11:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check import_closure --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 06:11:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check mechanism_contracts
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 06:11:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check import_closure
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 06:11:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=40

## 2026-06-04 06:11:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=41

## 2026-06-04 06:11:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=41

## 2026-06-04 06:12:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=40

## 2026-06-04 06:16:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=40 traces=280

## 2026-06-04 06:16:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=41 traces=287

## 2026-06-04 06:17:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=40 traces=280

## 2026-06-04 06:17:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=41 traces=287

## 2026-06-04 06:17:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=1881 grouped=78

## 2026-06-04 06:17:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f26_easy_consensus_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=162 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 06:23:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check mechanism_contracts --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 06:23:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check mechanism_contracts
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 06:24:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 06:24:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 06:24:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 06:24:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 06:27:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 06:27:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 06:31:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 06:31:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 06:31:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=2061 grouped=82

## 2026-06-04 06:31:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f26_easy_consensus_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=342 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 06:34:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check mechanism_contracts --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 06:34:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check mechanism_contracts
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 06:35:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 06:35:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 06:35:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 06:35:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 06:38:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 06:38:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 06:38:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 06:38:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 06:39:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=2241 grouped=86

## 2026-06-04 06:39:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f26_easy_consensus_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=522 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 06:41:37 +0800 追加：F26-F31 手工复现命令与最终审计

### F26/F27 前置 code check

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v21_common.py \
  experiments/run_v21_01_source_retention.py \
  experiments/run_v21_01_s06_truth_gate.py \
  experiments/run_v21_01_f26_easy_consensus_summary.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py \
  --check import_closure \
  --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 \
  --device cuda:0

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py \
  --check mechanism_contracts \
  --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 \
  --device cuda:0
```

- status: completed
- note: py_compile pass; import_closure=1; mechanism_contracts pass

### F26/F27 easy-consensus smoke

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SPECS=F26-easy-b1-consensus-transfer,F27-loss-warm-to-easy-b1-consensus-migration,F25-loss-warm-to-b1-consensus-migration,F10-T7-b1-cross-split-consensus-transfer,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx "$KAN" experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" --scope target --carriers D-CHE,D-FOU --device cuda:0 \
    --data-root data --steps 1600 --spec-ids "$SPECS" \
    --run-label v2101_f26_easy_consensus_smoke \
    --shard-count 4 --shard-index "$idx" \
    > "$OUT/v21_01_f26_easy_consensus_smoke_shard_${idx}.log" 2>&1 &
done
wait
"$KAN" experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" --scope target --merge-only \
  --run-label v2101_f26_easy_consensus_smoke \
  > "$OUT/v21_01_f26_easy_consensus_smoke_merge.log" 2>&1
"$KAN" experiments/run_v21_01_f26_easy_consensus_summary.py \
  --out-dir "$OUT" \
  > "$OUT/v21_01_f26_easy_consensus_summary_smoke.log" 2>&1
```

- status: completed
- note: rows=162 blocked=0 early_chain=0 productive_h3200=0 full_recommended=0
- artifacts: `v21_01_f26_easy_consensus_summary.csv`, `v21_01_f26_easy_consensus_decision.csv`, `v21_01_f26_easy_consensus_examples.csv`

### F28/F29 blended target smoke

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SPECS=F28-loss-easy-b1-consensus-blend,F29-loss-warm-to-loss-easy-b1-consensus-blend,F27-loss-warm-to-easy-b1-consensus-migration,F25-loss-warm-to-b1-consensus-migration,F10-T7-b1-cross-split-consensus-transfer,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx "$KAN" experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" --scope target --carriers D-CHE,D-FOU --device cuda:0 \
    --data-root data --steps 1600 --spec-ids "$SPECS" \
    --run-label v2101_f26_f28_blend_smoke \
    --shard-count 4 --shard-index "$idx" \
    > "$OUT/v21_01_f28_blend_smoke_shard_${idx}.log" 2>&1 &
done
wait
"$KAN" experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" --scope target --merge-only \
  --run-label v2101_f26_f28_blend_smoke \
  > "$OUT/v21_01_f28_blend_smoke_merge.log" 2>&1
"$KAN" experiments/run_v21_01_f26_easy_consensus_summary.py \
  --out-dir "$OUT" \
  > "$OUT/v21_01_f28_blend_summary_smoke.log" 2>&1
```

- status: completed
- note: cumulative rows=342 blocked=0 early_chain=0 productive_h3200=0 full_recommended=0

### F30/F31 train-split gain-gated migration smoke

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SPECS=F30-gain-gated-loss-warm-b1-consensus,F31-gain-gated-loss-warm-blend-consensus,F29-loss-warm-to-loss-easy-b1-consensus-blend,F25-loss-warm-to-b1-consensus-migration,F10-T7-b1-cross-split-consensus-transfer,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx "$KAN" experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" --scope target --carriers D-CHE,D-FOU --device cuda:0 \
    --data-root data --steps 1600 --spec-ids "$SPECS" \
    --run-label v2101_f26_f30_gated_smoke \
    --shard-count 4 --shard-index "$idx" \
    > "$OUT/v21_01_f30_gated_smoke_shard_${idx}.log" 2>&1 &
done
wait
"$KAN" experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" --scope target --merge-only \
  --run-label v2101_f26_f30_gated_smoke \
  > "$OUT/v21_01_f30_gated_smoke_merge.log" 2>&1
"$KAN" experiments/run_v21_01_f26_easy_consensus_summary.py \
  --out-dir "$OUT" \
  > "$OUT/v21_01_f30_gated_summary_smoke.log" 2>&1
```

- status: completed
- note: cumulative rows=522 blocked=0 early_chain=0 productive_h3200=0 full_recommended=0 promotion=0

### 最终审计

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py \
  --check import_closure \
  --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 \
  --device cuda:0

ls -lh \
  results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_code_review_packet.zip \
  results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_results_bundle.zip

python - <<'PY'
import csv, json
from pathlib import Path
out=Path('results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01')
route=json.loads((out/'v21_01_route_decision.json').read_text())
manifest=list(csv.DictReader((out/'v21_01_required_artifact_manifest.csv').open()))
truth=list(csv.DictReader((out/'v21_01_code_truth_gate.csv').open()))
print(route.get('route'), route.get('promotion_allowed'))
print(len(manifest), sum(1 for r in manifest if str(r.get('exists','')) in {'0','False','false',''}))
print([(r.get('check'), r.get('pass'), r.get('value')) for r in truth if r.get('check') in {'import_closure','mechanism_contracts'}])
PY

nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
```

- status: completed
- note: route=`R2-LateReboundNoContinuousRetention-F26EasyConsensusSmokeNoEarlyChain`; promotion_allowed=0; import_closure=1; mechanism_contracts=13/13; manifest rows=68 missing=0; packet=7.3M; bundle=16M; GPU util/mem all 0.

### 文档补写后重新打包

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv
from pathlib import Path
from experiments.run_v21_01_common import build_packet
out = Path('results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01')
manifest = out / 'v21_01_required_artifact_manifest.csv'
required = [r['artifact'] for r in csv.DictReader(manifest.open()) if r.get('artifact')]
build_packet(out, required)
print('rebuilt', len(required))
PY
ls -lh \
  results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_code_review_packet.zip \
  results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_results_bundle.zip
```

- status: completed
- note: rebuilt required artifacts=68; packet=7.3M; bundle=16M

## 2026-06-04 06:41:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check import_closure --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 06:41:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check import_closure
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 06:54:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check mechanism_contracts --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 06:54:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check import_closure --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 06:54:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check mechanism_contracts
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 06:54:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check import_closure
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 06:56:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 06:56:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 06:56:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 06:56:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 06:59:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 06:59:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:02:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:03:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:03:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=2421 grouped=90

## 2026-06-04 07:03:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f26_easy_consensus_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=702 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 07:06:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 07:06:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 07:06:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 07:06:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 07:09:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:09:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:13:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:13:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:13:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=2421 grouped=90

## 2026-06-04 07:13:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f26_easy_consensus_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=702 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 07:15:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f26_easy_consensus_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=702 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 07:15:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check import_closure --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 07:15:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check mechanism_contracts --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 07:15:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check mechanism_contracts
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 07:15:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check import_closure
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 07:16:38 +0800 追加：F32/F33 B3-null source-channel repair 手工复现命令与最终审计

### 环境 / 前置说明

- 继续使用历史实验环境：`/home/chengshun.wang/miniconda3/envs/kan/bin/python`。
- 默认 `python` 是 `/home/chengshun.wang/miniconda3/bin/python`，没有 torch；一次用默认 `python` 跑 S0.6 触发 `ModuleNotFoundError: No module named 'torch'`，未产生实验数据。
- S0.6 脚本没有 `--no-log` 参数；一次误带该参数只产生 argparse error，未产生实验数据。

### F32/F33 修改后 code check

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v21_common.py \
  experiments/run_v21_01_source_retention.py \
  experiments/run_v21_01_s06_truth_gate.py \
  experiments/run_v21_01_f26_easy_consensus_summary.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py \
  --check mechanism_contracts \
  --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 \
  --device cuda:0

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py \
  --check import_closure \
  --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 \
  --device cuda:0
```

- status: completed
- note: py_compile pass; import_closure=1; mechanism_contracts=15/15

### F32/F33 B3-null smoke

第一次 F32/F33 smoke 完成后发现新增 `target_b3_null_rows` 没有进入 horizon matrix。随后补充 `experiments/run_v17_common.py` trace 字段和 `experiments/run_v21_01_source_retention.py` enrichment 字段，并用同一 run label 重跑以下命令覆盖 shard CSV。最终复盘数据来自这次重跑。

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
SPECS=F32-b1-consensus-b3-null-transfer,F33-loss-warm-to-b1-consensus-b3-null,F30-gain-gated-loss-warm-b1-consensus,F25-loss-warm-to-b1-consensus-migration,F10-T7-b1-cross-split-consensus-transfer,F3-T1-loss-cotangent-target,F3-T5-random-matched-target,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead

for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx "$KAN" experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" --scope target --carriers D-CHE,D-FOU --device cuda:0 \
    --data-root data --steps 1600 --spec-ids "$SPECS" \
    --run-label v2101_f26_f32_b3null_smoke \
    --shard-count 4 --shard-index "$idx" \
    > "$OUT/v21_01_f32_b3null_smoke_shard_${idx}.log" 2>&1 &
done
wait

"$KAN" experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" --scope target --merge-only \
  --run-label v2101_f26_f32_b3null_smoke \
  > "$OUT/v21_01_f32_b3null_smoke_merge.log" 2>&1

"$KAN" experiments/run_v21_01_f26_easy_consensus_summary.py \
  --out-dir "$OUT" \
  > "$OUT/v21_01_f32_b3null_summary_smoke.log" 2>&1
```

- status: completed
- note: cumulative F26-F33 rows=702; blocked=0; candidate early-chain groups=0; productive h3200 groups=0; full_recommended=0; promotion_allowed=0
- artifacts: `v21_01_source_retention_matrix_v2101_f26_f32_b3null_smoke_fc0.csv` ... `fc3.csv`; `v21_01_source_retention_traces_v2101_f26_f32_b3null_smoke_fc0.csv` ... `fc3.csv`; `v21_01_f26_easy_consensus_summary.csv`; `v21_01_f26_easy_consensus_decision.csv`; `v21_01_f26_easy_consensus_examples.csv`

### F32/F33 key result readback

```bash
python - <<'PY'
import csv,json
from pathlib import Path
out=Path('results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01')
route=json.loads((out/'v21_01_route_decision.json').read_text())
print(route.get('route'), route.get('promotion_allowed'))
summary=list(csv.DictReader((out/'v21_01_f26_easy_consensus_summary.csv').open()))
for r in summary:
    if r.get('run_label')=='v2101_f26_f32_b3null_smoke' and r.get('v21_id') in {
        'F32-b1-consensus-b3-null-transfer',
        'F33-loss-warm-to-b1-consensus-b3-null',
        'F30-gain-gated-loss-warm-b1-consensus',
        'F25-loss-warm-to-b1-consensus-migration',
    }:
        print(r.get('carrier'), r.get('v21_id'), r.get('source_h800_mean'), r.get('source_h1600_mean'), r.get('early_chain_group'), r.get('target_b3_null_rows_h800_mean'), r.get('target_b3_null_rows_h1600_mean'))
PY
```

- status: completed
- note: route=`R2-LateReboundNoContinuousRetention-F26EasyConsensusSmokeNoEarlyChain`; promotion_allowed=0; F32/F33 no early-chain; no full escalation

### 最终审计

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py \
  --check import_closure \
  --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 \
  --device cuda:0

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py \
  --check mechanism_contracts \
  --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 \
  --device cuda:0

python - <<'PY'
import csv, json
from pathlib import Path
out=Path('results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01')
route=json.loads((out/'v21_01_route_decision.json').read_text())
manifest=list(csv.DictReader((out/'v21_01_required_artifact_manifest.csv').open()))
truth=list(csv.DictReader((out/'v21_01_code_truth_gate.csv').open()))
print(route.get('route'), route.get('promotion_allowed'))
print(len(manifest), sum(1 for r in manifest if str(r.get('exists','')) in {'0','False','false',''}))
print([(r.get('check'), r.get('pass'), r.get('value')) for r in truth if r.get('check') in {'import_closure','mechanism_contracts'}])
PY

nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
```

- status: completed
- note: route=`R2-LateReboundNoContinuousRetention-F26EasyConsensusSmokeNoEarlyChain`; promotion_allowed=0; manifest rows=68 missing=0; import_closure=1; mechanism_contracts=15/15; GPU util/mem all 0/0

### 文档补写后重新打包

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv
from pathlib import Path
from experiments.run_v21_01_common import build_packet
out = Path('results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01')
manifest = out / 'v21_01_required_artifact_manifest.csv'
required = [r['artifact'] for r in csv.DictReader(manifest.open()) if r.get('artifact')]
build_packet(out, required)
print('rebuilt', len(required))
PY
ls -lh \
  results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_code_review_packet.zip \
  results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01/v21_01_results_bundle.zip
```

- status: completed
- note: rebuilt required artifacts=68; packet=7.5M; bundle=17M

## 2026-06-04 07:29:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 07:29:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 07:30:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 07:30:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 07:30:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 07:30:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 07:34:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:34:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:37:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:37:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:37:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=2601 grouped=94

## 2026-06-04 07:38:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f34_view_consistency_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=180 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 07:40:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 07:40:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 07:41:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 07:41:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 07:41:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 07:41:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=45

## 2026-06-04 07:44:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:44:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:48:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:48:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=45 traces=315

## 2026-06-04 07:48:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=2601 grouped=94

## 2026-06-04 07:49:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f34_view_consistency_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=180 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 07:50:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f34_view_consistency_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=180 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 07:51:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v21_01_s06_truth_gate.py experiments/run_v21_01_f34_view_consistency_summary.py && /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: completed
- note: F34/F35 implementation compile + S0.6 rerun passed; S0_6_preflight_pass=1; mechanism_contracts=17/17; M81 update semantics pass=1

## 2026-06-04 07:51:01 +0800

```bash
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01; SPECS='F34-view-consistent-loss-b3-null,F35-loss-warm-to-view-consistent-loss,F3-T1-loss-cotangent-target,F25-loss-warm-to-b1-consensus-migration,F33-loss-warm-to-b1-consensus-b3-null,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead'; for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --out-dir "$OUT" --run-label v2101_f34_smoke --spec-ids "$SPECS" --steps 1600 --shard-count 4 --shard-index $i --device cuda:0 > "$OUT/v21_01_f34_smoke_shard_${i}.log" 2>&1 & done; wait; /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --out-dir "$OUT" --run-label v2101_f34_smoke --spec-ids "$SPECS" --steps 1600 --merge-only --device cuda:0
```

- status: completed
- note: F34/F35 smoke rerun completed after trace diagnostic whitelist fix; rows=180 blocked=0; candidate early-chain groups=0; full escalation recommended=0

## 2026-06-04 08:03:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 08:03:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 08:04:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=54

## 2026-06-04 08:04:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=54

## 2026-06-04 08:04:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=54

## 2026-06-04 08:04:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=54

## 2026-06-04 08:09:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=54 traces=378

## 2026-06-04 08:09:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=54 traces=378

## 2026-06-04 08:15:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=54 traces=378

## 2026-06-04 08:16:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=54 traces=378

## 2026-06-04 08:16:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=2817 grouped=98

## 2026-06-04 08:17:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f36_lowbank_source_channel_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=216 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 08:19:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f36_lowbank_source_channel_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=216 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 08:24:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: started

## 2026-06-04 08:24:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all
```

- status: completed
- note: S0_6_preflight_pass=1

## 2026-06-04 08:25:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=63

## 2026-06-04 08:25:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=63

## 2026-06-04 08:25:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=63

## 2026-06-04 08:25:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=63

## 2026-06-04 08:30:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=63 traces=441

## 2026-06-04 08:31:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=63 traces=441

## 2026-06-04 08:34:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=63 traces=441

## 2026-06-04 08:34:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=63 traces=441

## 2026-06-04 08:34:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=3069 grouped=102

## 2026-06-04 08:35:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f36_lowbank_source_channel_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=252 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 08:36:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f36_lowbank_source_channel_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01
```

- status: completed
- note: rows=252 blocked=0 early=0 productive_h3200=0 full_recommended=0 promotion=0

## 2026-06-04 08:37:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v21_01_s06_truth_gate.py experiments/run_v21_01_f36_lowbank_source_channel_summary.py && /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_s06_truth_gate.py --check all --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --device cuda:0
```

- status: completed
- note: F36-F39 implementation compile + S0.6 rerun passed; S0_6_preflight_pass=1; mechanism_contracts=21/21; M83/M85 update semantics pass=1

## 2026-06-04 08:37:27 +0800

```bash
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01; SPECS='F36-lowbank-loss-b3-null,F37-loss-warm-to-lowbank-loss-b3-null,F34-view-consistent-loss-b3-null,F35-loss-warm-to-view-consistent-loss,F3-T1-loss-cotangent-target,F25-loss-warm-to-b1-consensus-migration,F33-loss-warm-to-b1-consensus-b3-null,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead'; for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --out-dir "$OUT" --run-label v2101_f36_smoke --spec-ids "$SPECS" --steps 1600 --shard-count 4 --shard-index $i --device cuda:0 > "$OUT/v21_01_f36_smoke_shard_${i}.log" 2>&1 & done; wait; /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --out-dir "$OUT" --run-label v2101_f36_smoke --spec-ids "$SPECS" --steps 1600 --merge-only --device cuda:0
```

- status: completed
- note: initial F36/F37 smoke rows=216 blocked=0, but audit found M83 direct writer was not yet in target-mechanism train-loop branch; old v2101_f36_smoke is not used for final F36 family decision

## 2026-06-04 08:37:27 +0800

```bash
OUT=results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01; SPECS='F36-lowbank-loss-b3-null,F37-loss-warm-to-lowbank-loss-b3-null,F38-gain-gated-lowbank-loss-b3-null,F39-loss-warm-to-gated-lowbank-loss-b3-null,F34-view-consistent-loss-b3-null,F35-loss-warm-to-view-consistent-loss,F3-T1-loss-cotangent-target,F25-loss-warm-to-b1-consensus-migration,F33-loss-warm-to-b1-consensus-b3-null,F3-T5-random-matched-target,F9-TCTRL-stable-random-target,CTRL-SGD,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead'; for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --out-dir "$OUT" --run-label v2101_f36_fix_smoke --spec-ids "$SPECS" --steps 1600 --shard-count 4 --shard-index $i --device cuda:0 > "$OUT/v21_01_f36_fix_smoke_shard_${i}.log" 2>&1 & done; wait; /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --out-dir "$OUT" --run-label v2101_f36_fix_smoke --spec-ids "$SPECS" --steps 1600 --merge-only --device cuda:0
```

- status: completed
- note: fresh F36-F39 smoke rows=252 blocked=0; candidate early-chain groups=0; full escalation recommended=0

## 2026-06-04 08:37:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_f36_lowbank_source_channel_summary.py --out-dir results/v21_01_source_retention_fu_kernel_officialization_4gpu/official_v21_01 --run-prefix v2101_f36_fix_
```

- status: completed
- note: F36-F39 recap/decision refreshed from v2101_f36_fix_ only; route remains promotion_allowed=0

## 2026-06-04 12:04:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=810 grouped=49

## 2026-06-04 13:41:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=13

## 2026-06-04 13:41:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=14

## 2026-06-04 13:41:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=14

## 2026-06-04 13:41:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=13

## 2026-06-04 13:42:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=14 traces=140

## 2026-06-04 13:42:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=13 traces=130

## 2026-06-04 13:42:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=13 traces=130

## 2026-06-04 13:42:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=14 traces=140

## 2026-06-04 13:43:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=1179 grouped=58

## 2026-06-04 19:42:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cpu --steps 120
```

- status: started
- note: jobs=5

## 2026-06-04 19:42:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=20

## 2026-06-04 19:43:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-04 19:47:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 19:47:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-04 19:47:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 19:47:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 19:59:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=165

## 2026-06-04 19:59:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 19:59:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 20:02:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 20:02:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-04 20:18:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cpu --steps 120
```

- status: started
- note: jobs=7

## 2026-06-04 20:18:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=28

## 2026-06-04 20:19:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-04 20:21:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 20:21:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 20:21:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-04 20:21:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 20:23:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=165

## 2026-06-04 20:24:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 20:24:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 20:24:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 20:25:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-04 20:33:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cpu --steps 120
```

- status: started
- note: jobs=7

## 2026-06-04 20:33:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=28

## 2026-06-04 20:34:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-04 20:35:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 20:35:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 20:35:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-04 20:35:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 20:37:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=165

## 2026-06-04 20:38:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 20:38:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 20:38:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 20:40:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-04 20:49:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cpu --steps 120
```

- status: started
- note: jobs=7

## 2026-06-04 20:49:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=28

## 2026-06-04 20:49:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-04 20:50:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 20:50:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-04 20:50:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 20:50:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 20:53:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=165

## 2026-06-04 20:53:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 20:53:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 20:53:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 20:54:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-04 20:56:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 20:56:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 20:56:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-04 20:56:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-04 20:59:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=165

## 2026-06-04 20:59:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 20:59:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 20:59:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=176

## 2026-06-04 21:00:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-04 21:14:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=45

## 2026-06-04 21:19:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=45 traces=450

## 2026-06-04 21:19:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-04 21:22:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=15

## 2026-06-04 21:22:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-04 21:22:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-04 21:22:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-04 21:24:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=150

## 2026-06-04 21:24:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=160

## 2026-06-04 21:24:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=160

## 2026-06-04 21:24:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=160

## 2026-06-04 21:25:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-04 21:27:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=23

## 2026-06-04 21:27:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=22

## 2026-06-04 21:27:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=23

## 2026-06-04 21:27:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=22

## 2026-06-04 21:30:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=22 traces=220

## 2026-06-04 21:30:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=22 traces=220

## 2026-06-04 21:30:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=23 traces=230

## 2026-06-04 21:30:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=23 traces=230

## 2026-06-04 21:31:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=90 grouped=10

## 2026-06-04 21:36:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-04 21:36:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-04 21:36:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=15

## 2026-06-04 21:36:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-04 21:45:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=150

## 2026-06-04 21:45:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=160

## 2026-06-04 21:45:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=160

## 2026-06-04 21:47:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=160

## 2026-06-04 21:49:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-04 22:02:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 22:02:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-04 22:02:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 22:02:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-04 22:04:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 22:04:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 22:04:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=110

## 2026-06-04 22:04:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=120

## 2026-06-04 22:05:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-04 23:59:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=21

## 2026-06-04 23:59:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=20

## 2026-06-04 23:59:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=20

## 2026-06-04 23:59:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=20

## 2026-06-05 00:01:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=20 traces=220

## 2026-06-05 00:01:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=20 traces=220

## 2026-06-05 00:01:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=20 traces=220

## 2026-06-05 00:01:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=21 traces=231

## 2026-06-05 00:02:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=81 grouped=9

## 2026-06-05 00:07:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:07:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:07:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 00:07:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:08:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:08:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:08:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:08:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:08:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:08:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:09:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 00:09:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 00:10:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:10:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:10:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:10:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:10:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:10:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:10:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 00:10:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 00:11:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:11:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:11:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:11:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:11:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:11:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:12:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 00:12:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 00:12:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:12:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:12:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:12:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:12:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:12:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:13:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 00:13:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 00:13:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:13:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:14:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:14:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 00:15:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=225 grouped=9

## 2026-06-05 00:33:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:33:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:33:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 00:33:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:34:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:34:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:34:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:34:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 00:34:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 00:34:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:34:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:34:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:35:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:35:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:36:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:36:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 00:36:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:36:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:36:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:36:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 00:37:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:37:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:37:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:37:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 00:37:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=135 grouped=7

## 2026-06-05 00:42:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:42:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:42:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 00:42:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:43:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:43:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:43:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 00:43:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:43:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:43:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:43:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 00:43:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:45:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:45:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:45:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:45:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 00:45:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 00:45:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:45:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:45:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:46:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:46:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:46:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:46:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 00:46:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=135 grouped=7

## 2026-06-05 00:51:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:51:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 00:51:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:51:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:52:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:53:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:53:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:53:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 00:53:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:53:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:53:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 00:53:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 00:54:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:54:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:54:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 00:54:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 00:54:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=90 grouped=6

## 2026-06-05 01:19:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cpu --steps 20
```

- status: started
- note: jobs=3

## 2026-06-05 01:19:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=3 traces=9

## 2026-06-05 01:19:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=3 grouped=3

## 2026-06-05 01:20:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 01:20:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 01:20:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 01:20:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 01:21:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=99

## 2026-06-05 01:21:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=99

## 2026-06-05 01:21:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=99

## 2026-06-05 01:21:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=99

## 2026-06-05 01:21:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-05 01:24:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cpu --steps 20
```

- status: started
- note: jobs=3

## 2026-06-05 01:24:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=3 traces=9

## 2026-06-05 01:24:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=3 grouped=3

## 2026-06-05 01:25:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 01:25:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 01:25:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 01:25:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 01:26:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 01:26:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 01:26:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 01:27:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 01:27:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-05 01:33:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cpu --steps 20
```

- status: started
- note: jobs=5

## 2026-06-05 01:33:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=5 traces=15

## 2026-06-05 01:33:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-05 01:34:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=32

## 2026-06-05 01:34:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=31

## 2026-06-05 01:34:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=31

## 2026-06-05 01:34:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=32

## 2026-06-05 01:46:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2
```

- status: completed
- note: rows=31 traces=341

## 2026-06-05 01:46:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3
```

- status: completed
- note: rows=31 traces=341

## 2026-06-05 01:46:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1
```

- status: completed
- note: rows=32 traces=352

## 2026-06-05 01:47:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=32 traces=352

## 2026-06-05 01:48:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=126 grouped=14

## 2026-06-05 01:50:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cpu --steps 20
```

- status: started
- note: jobs=4

## 2026-06-05 01:50:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=4 traces=12

## 2026-06-05 01:50:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=4 grouped=4

## 2026-06-05 01:51:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-05 01:51:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-05 01:51:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-05 01:51:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-05 01:58:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1
```

- status: completed
- note: rows=27 traces=297

## 2026-06-05 01:58:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2
```

- status: completed
- note: rows=27 traces=297

## 2026-06-05 01:58:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=27 traces=297

## 2026-06-05 01:58:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3
```

- status: completed
- note: rows=27 traces=297

## 2026-06-05 01:59:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=108 grouped=12

## 2026-06-05 02:01:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cpu --steps 20
```

- status: started
- note: jobs=4

## 2026-06-05 02:01:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=4 traces=12

## 2026-06-05 02:01:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=4 grouped=4

## 2026-06-05 02:02:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-05 02:02:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-05 02:02:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-05 02:02:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=27

## 2026-06-05 02:08:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2
```

- status: completed
- note: rows=27 traces=297

## 2026-06-05 02:08:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3
```

- status: completed
- note: rows=27 traces=297

## 2026-06-05 02:08:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=27 traces=297

## 2026-06-05 02:08:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1
```

- status: completed
- note: rows=27 traces=297

## 2026-06-05 02:09:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=108 grouped=12

## 2026-06-05 02:27:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 120
```

- status: started
- note: jobs=10

## 2026-06-05 02:28:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=10 traces=40

## 2026-06-05 02:28:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=10 grouped=10

## 2026-06-05 02:29:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=32

## 2026-06-05 02:29:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=31

## 2026-06-05 02:29:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=31

## 2026-06-05 02:29:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=32

## 2026-06-05 02:53:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 2
```

- status: completed
- note: rows=31 traces=341

## 2026-06-05 02:55:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 3
```

- status: completed
- note: rows=31 traces=341

## 2026-06-05 02:55:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 1
```

- status: completed
- note: rows=32 traces=352

## 2026-06-05 02:56:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=32 traces=352

## 2026-06-05 02:56:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=126 grouped=14

## 2026-06-05 02:59:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=36

## 2026-06-05 03:05:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=36 traces=216

## 2026-06-05 03:06:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=36 grouped=12

## 2026-06-05 03:09:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=36

## 2026-06-05 03:10:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=36 traces=216

## 2026-06-05 03:10:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=36 grouped=12

## 2026-06-05 03:13:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=30

## 2026-06-05 03:14:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=30 traces=180

## 2026-06-05 03:15:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=30 grouped=10

## 2026-06-05 03:15:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=36

## 2026-06-05 03:18:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=36 traces=216

## 2026-06-05 03:18:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=36 grouped=12

## 2026-06-05 03:19:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=30

## 2026-06-05 03:21:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --shard-index 0
```

- status: completed
- note: rows=30 traces=180

## 2026-06-05 03:21:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope target --merge-only
```

- status: completed
- note: rows=30 grouped=10

## 2026-06-05 03:37:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=36

## 2026-06-05 03:39:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=36 traces=252

## 2026-06-05 03:39:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=36 grouped=12

## 2026-06-05 03:42:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=18

## 2026-06-05 03:42:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=18

## 2026-06-05 03:42:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=18

## 2026-06-05 03:42:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=18

## 2026-06-05 03:45:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=198

## 2026-06-05 03:45:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=18 traces=198

## 2026-06-05 03:45:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=18 traces=198

## 2026-06-05 03:45:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=18 traces=198

## 2026-06-05 03:45:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=72 grouped=8

## 2026-06-05 03:49:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=12

## 2026-06-05 03:49:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=72

## 2026-06-05 03:49:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=12 grouped=4

## 2026-06-05 03:51:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=24

## 2026-06-05 03:52:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=24 traces=144

## 2026-06-05 03:52:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=24 grouped=8

## 2026-06-05 03:53:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=14

## 2026-06-05 03:53:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=14

## 2026-06-05 03:53:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=13

## 2026-06-05 03:53:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=13

## 2026-06-05 03:55:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=13 traces=143

## 2026-06-05 03:55:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=13 traces=143

## 2026-06-05 03:55:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=14 traces=154

## 2026-06-05 03:55:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=14 traces=154

## 2026-06-05 03:56:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=54 grouped=6

## 2026-06-05 03:58:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=24

## 2026-06-05 03:58:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=24 traces=144

## 2026-06-05 03:58:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=24 grouped=8

## 2026-06-05 04:00:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:00:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:00:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:00:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 04:01:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:01:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:01:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:01:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 04:02:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-05 04:04:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=21

## 2026-06-05 04:05:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=21 traces=126

## 2026-06-05 04:05:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=21 grouped=7

## 2026-06-05 04:05:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 04:05:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:05:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:05:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:07:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:07:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:07:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:07:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 04:07:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-05 04:11:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=15

## 2026-06-05 04:11:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=15 traces=90

## 2026-06-05 04:12:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=15 grouped=5

## 2026-06-05 04:12:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:12:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:12:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:12:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 04:14:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:14:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:14:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:14:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 04:14:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-05 04:18:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=15

## 2026-06-05 04:18:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=15 traces=90

## 2026-06-05 04:18:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=15 grouped=5

## 2026-06-05 04:19:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:19:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:19:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 04:19:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:20:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:20:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:20:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:20:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 04:21:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-05 04:25:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=15

## 2026-06-05 04:25:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=15 traces=90

## 2026-06-05 04:25:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=15 grouped=5

## 2026-06-05 04:26:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:26:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 04:26:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:26:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 04:27:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:27:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:27:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 04:27:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 04:28:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-05 04:57:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=15

## 2026-06-05 04:57:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=15 traces=90

## 2026-06-05 04:58:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=15 grouped=5

## 2026-06-05 05:00:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 05:00:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 05:00:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 05:00:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 05:01:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 05:01:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 05:01:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 05:01:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 05:01:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-05 05:03:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 05:03:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 05:03:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 05:03:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 05:04:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 05:04:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 05:04:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 05:04:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 05:04:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-05 05:05:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 05:05:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 05:05:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 05:05:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 05:07:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 05:07:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 05:07:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 05:07:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 05:07:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-05 05:14:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=9

## 2026-06-05 05:14:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=9 traces=54

## 2026-06-05 05:14:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=9 grouped=3

## 2026-06-05 05:15:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 05:15:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=12

## 2026-06-05 05:15:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 05:15:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=11

## 2026-06-05 05:16:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 05:16:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 05:16:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=121

## 2026-06-05 05:16:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=132

## 2026-06-05 05:17:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-05 05:38:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=15

## 2026-06-05 05:38:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=15 traces=90

## 2026-06-05 05:39:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=15 grouped=5

## 2026-06-05 05:40:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=15

## 2026-06-05 05:40:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=15 traces=105

## 2026-06-05 05:41:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=15 grouped=5

## 2026-06-05 05:46:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=15

## 2026-06-05 05:47:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=15 traces=105

## 2026-06-05 05:47:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=15 grouped=5

## 2026-06-05 05:52:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=15

## 2026-06-05 05:52:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 05:52:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 05:52:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 05:54:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=165

## 2026-06-05 05:55:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 05:55:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 05:55:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 05:55:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 06:23:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4050
```

- status: started
- note: jobs=5

## 2026-06-05 06:24:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=50

## 2026-06-05 06:24:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-05 06:25:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 06:25:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 06:25:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=15

## 2026-06-05 06:25:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 06:27:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=165

## 2026-06-05 06:27:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 06:27:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 06:27:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 06:28:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 06:48:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4050
```

- status: started
- note: jobs=5

## 2026-06-05 06:49:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=50

## 2026-06-05 06:49:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-05 06:51:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4050
```

- status: started
- note: jobs=5

## 2026-06-05 06:51:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=50

## 2026-06-05 06:52:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-05 06:56:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4050
```

- status: started
- note: jobs=5

## 2026-06-05 06:56:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=50

## 2026-06-05 06:57:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-05 06:58:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 06:58:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 06:58:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=15

## 2026-06-05 06:58:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 07:00:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=165

## 2026-06-05 07:00:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 07:00:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 07:00:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 07:01:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 07:24:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4050
```

- status: started
- note: jobs=5

## 2026-06-05 07:24:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=50

## 2026-06-05 07:24:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-05 07:27:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4050
```

- status: started
- note: jobs=5

## 2026-06-05 07:27:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=50

## 2026-06-05 07:27:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-05 07:28:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 07:28:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=15

## 2026-06-05 07:28:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 07:28:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 07:30:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=165

## 2026-06-05 07:30:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 07:30:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 07:31:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 07:31:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 07:48:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4050
```

- status: started
- note: jobs=2

## 2026-06-05 07:48:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=2 traces=20

## 2026-06-05 07:49:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=2 grouped=2

## 2026-06-05 07:50:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4050
```

- status: started
- note: jobs=5

## 2026-06-05 07:50:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=50

## 2026-06-05 07:50:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-05 07:51:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 07:51:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 07:51:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=15

## 2026-06-05 07:51:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 07:53:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=165

## 2026-06-05 07:53:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 07:53:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 07:53:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 07:53:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 08:11:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4050
```

- status: started
- note: jobs=5

## 2026-06-05 08:11:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=50

## 2026-06-05 08:11:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-05 08:12:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 08:12:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 08:12:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 08:12:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=15

## 2026-06-05 08:14:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=165

## 2026-06-05 08:14:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 08:14:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 08:14:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 08:14:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 08:37:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4050
```

- status: started
- note: jobs=5

## 2026-06-05 08:37:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=50

## 2026-06-05 08:37:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-05 08:42:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 08:42:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 4800
```

- status: started
- note: jobs=15

## 2026-06-05 08:42:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 08:42:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 4800
```

- status: started
- note: jobs=16

## 2026-06-05 08:44:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=165

## 2026-06-05 08:44:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 08:44:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 08:44:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=176

## 2026-06-05 08:44:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 16:58:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=7

## 2026-06-05 16:58:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=49

## 2026-06-05 16:58:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 16:59:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 16:59:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 16:59:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-05 16:59:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 17:02:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-05 17:02:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 17:02:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 17:02:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 17:09:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 17:27:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=7

## 2026-06-05 17:28:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=49

## 2026-06-05 17:28:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 17:29:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4200
```

- status: started
- note: jobs=7

## 2026-06-05 17:30:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=70

## 2026-06-05 17:30:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 17:32:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 17:32:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 17:32:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 17:32:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-05 17:35:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-05 17:35:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 17:35:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 17:35:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 17:36:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 18:06:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4200
```

- status: started
- note: jobs=2

## 2026-06-05 18:06:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=2 traces=20

## 2026-06-05 18:07:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=2 grouped=2

## 2026-06-05 18:09:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 18:09:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-05 18:09:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 18:09:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 18:12:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-05 18:12:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 18:12:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 18:12:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 18:12:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 18:21:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4200
```

- status: started
- note: jobs=2

## 2026-06-05 18:21:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=2 traces=20

## 2026-06-05 18:22:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=2 grouped=2

## 2026-06-05 18:23:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4200
```

- status: started
- note: jobs=7

## 2026-06-05 18:23:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=70

## 2026-06-05 18:24:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 18:25:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-05 18:25:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 18:25:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 18:25:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 18:28:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-05 18:28:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 18:28:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 18:28:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 18:29:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 18:47:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4200
```

- status: started
- note: jobs=7

## 2026-06-05 18:48:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=70

## 2026-06-05 18:49:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 18:51:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 18:51:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-05 18:51:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 18:51:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 18:53:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-05 18:54:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 18:54:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 18:54:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 18:54:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 19:10:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4200
```

- status: started
- note: jobs=7

## 2026-06-05 19:10:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=70

## 2026-06-05 19:11:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 19:12:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 19:12:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-05 19:12:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 19:12:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 19:15:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-05 19:15:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 19:15:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 19:15:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 19:15:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 19:25:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4200
```

- status: started
- note: jobs=7

## 2026-06-05 19:25:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=70

## 2026-06-05 19:26:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 19:27:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-05 19:27:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 19:27:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 19:27:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 19:30:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-05 19:30:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 19:30:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 19:30:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 19:30:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope f0 --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 19:56:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=14

## 2026-06-05 19:57:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=14 traces=98

## 2026-06-05 19:58:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=14 grouped=14

## 2026-06-05 20:16:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=4

## 2026-06-05 20:16:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=4 traces=28

## 2026-06-05 20:17:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=4

## 2026-06-05 20:17:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=4 traces=28

## 2026-06-05 20:17:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=8 grouped=8

## 2026-06-05 20:19:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=7

## 2026-06-05 20:20:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=7 traces=49

## 2026-06-05 20:20:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=7

## 2026-06-05 20:21:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=7 traces=49

## 2026-06-05 20:21:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=14 grouped=14

## 2026-06-05 20:37:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4200
```

- status: started
- note: jobs=7

## 2026-06-05 20:38:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=70

## 2026-06-05 20:39:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 20:41:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 20:41:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 20:41:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-05 20:41:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 20:43:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-05 20:44:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 20:44:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 20:44:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 20:44:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 20:59:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4200
```

- status: started
- note: jobs=7

## 2026-06-05 20:59:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=70

## 2026-06-05 21:00:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 21:01:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 21:01:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 21:01:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-05 21:01:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 21:03:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-05 21:04:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 21:04:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 21:04:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 21:04:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 21:39:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=7

## 2026-06-05 21:40:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=77

## 2026-06-05 21:40:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 21:42:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 21:42:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 21:42:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-05 21:42:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 21:45:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-05 21:45:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 21:45:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 21:45:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 21:45:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 21:57:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=7

## 2026-06-05 21:58:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=77

## 2026-06-05 21:58:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 21:59:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 21:59:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-05 21:59:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 21:59:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 22:02:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-05 22:02:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 22:02:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 22:02:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 22:03:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 22:29:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=7

## 2026-06-05 22:29:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=77

## 2026-06-05 22:30:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 22:34:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=7

## 2026-06-05 22:34:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=77

## 2026-06-05 22:35:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 22:36:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 22:36:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 22:36:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 22:36:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-05 22:39:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-05 22:39:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 22:39:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 22:39:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 22:39:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 23:03:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=7

## 2026-06-05 23:04:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=77

## 2026-06-05 23:04:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 23:07:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 23:07:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 23:07:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-05 23:07:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 23:10:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 23:10:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-05 23:10:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 23:10:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 23:10:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-05 23:32:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=7

## 2026-06-05 23:33:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=77

## 2026-06-05 23:33:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 23:37:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=7

## 2026-06-05 23:37:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=77

## 2026-06-05 23:38:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 23:41:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=7

## 2026-06-05 23:41:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=77

## 2026-06-05 23:42:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-05 23:43:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 23:43:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 23:43:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-05 23:43:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-05 23:46:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-05 23:46:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 23:46:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 23:46:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-05 23:47:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-06 00:12:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=4

## 2026-06-06 00:12:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=4 traces=44

## 2026-06-06 00:13:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=4 grouped=4

## 2026-06-06 00:15:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 4800
```

- status: started
- note: jobs=7

## 2026-06-06 00:16:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=77

## 2026-06-06 00:16:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-06 00:17:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-06 00:17:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-06 00:17:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=15

## 2026-06-06 00:17:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=16

## 2026-06-06 00:20:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=180

## 2026-06-06 00:20:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=16 traces=192

## 2026-06-06 00:20:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=16 traces=192

## 2026-06-06 00:20:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=16 traces=192

## 2026-06-06 00:20:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=63 grouped=7

## 2026-06-06 02:55:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=25

## 2026-06-06 02:55:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=25

## 2026-06-06 02:55:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=25

## 2026-06-06 02:55:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=24

## 2026-06-06 03:00:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=25 traces=300

## 2026-06-06 03:00:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=24 traces=288

## 2026-06-06 03:00:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=25 traces=300

## 2026-06-06 03:00:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=25 traces=300

## 2026-06-06 03:01:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=99 grouped=11

## 2026-06-06 03:03:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=12

## 2026-06-06 03:03:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-06 03:03:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-06 03:03:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-06 03:04:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=132

## 2026-06-06 03:05:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=132

## 2026-06-06 03:05:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=132

## 2026-06-06 03:05:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=144

## 2026-06-06 03:05:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-06 03:07:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-06 03:07:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=12

## 2026-06-06 03:07:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-06 03:07:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=11

## 2026-06-06 03:08:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=132

## 2026-06-06 03:08:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=11 traces=132

## 2026-06-06 03:08:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=11 traces=132

## 2026-06-06 03:09:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=144

## 2026-06-06 03:09:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=45 grouped=5

## 2026-06-06 03:21:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-06 03:21:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-06 03:21:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-06 03:21:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-06 03:22:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=13 traces=156

## 2026-06-06 03:23:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=13 traces=156

## 2026-06-06 03:23:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=14 traces=168

## 2026-06-06 03:23:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=14 traces=168

## 2026-06-06 03:23:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=54 grouped=6

## 2026-06-06 03:28:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-06 03:28:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-06 03:28:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-06 03:28:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-06 03:29:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=13 traces=156

## 2026-06-06 03:29:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=13 traces=156

## 2026-06-06 03:30:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=14 traces=168

## 2026-06-06 03:30:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=14 traces=168

## 2026-06-06 03:30:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=54 grouped=6

## 2026-06-06 03:34:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=2

## 2026-06-06 03:34:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=2 traces=12

## 2026-06-06 03:34:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=2 grouped=2

## 2026-06-06 03:36:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-06 03:36:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-06 03:36:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-06 03:36:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-06 03:37:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=13 traces=156

## 2026-06-06 03:38:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=13 traces=156

## 2026-06-06 03:38:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=14 traces=168

## 2026-06-06 03:38:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=14 traces=168

## 2026-06-06 03:38:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=54 grouped=6

## 2026-06-06 03:40:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 800
```

- status: started
- note: jobs=2

## 2026-06-06 03:40:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=2 traces=12

## 2026-06-06 03:41:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=2 grouped=2

## 2026-06-06 03:42:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-06 03:42:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-06 03:42:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-06 03:42:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-06 03:45:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=13 traces=156

## 2026-06-06 03:45:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=13 traces=156

## 2026-06-06 03:45:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=14 traces=168

## 2026-06-06 03:45:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=14 traces=168

## 2026-06-06 03:45:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=54 grouped=6

## 2026-06-06 03:49:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=36

## 2026-06-06 03:49:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=36

## 2026-06-06 03:49:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=36

## 2026-06-06 03:49:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=36

## 2026-06-06 04:00:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 0
```

- status: completed
- note: rows=36 traces=432

## 2026-06-06 04:00:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 1
```

- status: completed
- note: rows=36 traces=432

## 2026-06-06 04:01:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 3
```

- status: completed
- note: rows=36 traces=432

## 2026-06-06 04:01:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --shard-index 2
```

- status: completed
- note: rows=36 traces=432

## 2026-06-06 04:01:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope kan --merge-only
```

- status: completed
- note: rows=144 grouped=16

## 2026-06-06 05:21:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=27

## 2026-06-06 05:21:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=27

## 2026-06-06 05:21:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=27

## 2026-06-06 05:21:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=27

## 2026-06-06 05:28:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=27 traces=324

## 2026-06-06 05:28:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=27 traces=324

## 2026-06-06 05:29:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=27 traces=324

## 2026-06-06 05:30:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=27 traces=324

## 2026-06-06 05:30:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=108 grouped=12

## 2026-06-06 05:37:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=27

## 2026-06-06 05:37:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=27

## 2026-06-06 05:37:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=27

## 2026-06-06 05:37:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=27

## 2026-06-06 05:44:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=27 traces=324

## 2026-06-06 05:44:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=27 traces=324

## 2026-06-06 05:45:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=27 traces=324

## 2026-06-06 05:45:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=27 traces=324

## 2026-06-06 05:47:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=108 grouped=12

## 2026-06-06 06:13:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=8

## 2026-06-06 06:13:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=8 traces=56

## 2026-06-06 06:14:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=8 grouped=8

## 2026-06-06 06:15:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 06:15:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 06:15:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 06:15:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 06:18:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 06:18:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 06:18:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 06:18:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 06:18:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=72 grouped=8

## 2026-06-06 06:22:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=8

## 2026-06-06 06:23:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=8 traces=56

## 2026-06-06 06:23:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=8 grouped=8

## 2026-06-06 06:23:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 06:23:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 06:23:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 06:23:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 06:26:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 06:26:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 06:26:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 06:26:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 06:27:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=72 grouped=8

## 2026-06-06 06:31:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=8

## 2026-06-06 06:32:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=8 traces=56

## 2026-06-06 06:32:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=8 grouped=8

## 2026-06-06 06:33:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 06:33:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 06:33:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 06:33:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 06:35:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 06:36:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 06:36:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 06:36:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 06:36:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=72 grouped=8

## 2026-06-06 07:04:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=7

## 2026-06-06 07:05:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=63

## 2026-06-06 07:06:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-06 07:08:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=7

## 2026-06-06 07:09:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=63

## 2026-06-06 07:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-06 07:10:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=7

## 2026-06-06 07:11:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=63

## 2026-06-06 07:11:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-06 07:13:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=7

## 2026-06-06 07:13:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=7 traces=63

## 2026-06-06 07:14:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=7 grouped=7

## 2026-06-06 07:16:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 07:16:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 07:17:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-06 07:18:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 07:18:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 07:18:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 07:18:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 07:21:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 07:21:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 07:21:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 07:21:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 07:21:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=72 grouped=8

## 2026-06-06 07:33:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 100
```

- status: started
- note: jobs=8

## 2026-06-06 07:33:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 100
```

- status: started
- note: jobs=5

## 2026-06-06 07:33:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=20

## 2026-06-06 07:33:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=8 traces=32

## 2026-06-06 07:33:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=5 grouped=5

## 2026-06-06 07:33:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=8 grouped=8

## 2026-06-06 07:34:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 07:34:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 07:34:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 07:34:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 07:38:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 07:38:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 07:38:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 07:38:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 07:38:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=72 grouped=8

## 2026-06-06 07:50:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 07:50:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 07:50:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 07:50:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=18

## 2026-06-06 07:53:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 07:53:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 07:54:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 07:54:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=18 traces=216

## 2026-06-06 07:54:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=72 grouped=8

## 2026-06-06 08:09:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 400
```

- status: started
- note: jobs=9

## 2026-06-06 08:09:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=9 traces=45

## 2026-06-06 08:09:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=9 grouped=9

## 2026-06-06 08:10:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=21

## 2026-06-06 08:10:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=20

## 2026-06-06 08:10:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=20

## 2026-06-06 08:10:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=20

## 2026-06-06 08:14:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=20 traces=240

## 2026-06-06 08:14:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=20 traces=240

## 2026-06-06 08:14:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=20 traces=240

## 2026-06-06 08:14:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=21 traces=252

## 2026-06-06 08:14:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=81 grouped=9

## 2026-06-06 09:04:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=6

## 2026-06-06 09:04:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=6 traces=42

## 2026-06-06 09:04:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=6 grouped=6

## 2026-06-06 09:09:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=6

## 2026-06-06 09:09:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=6 traces=42

## 2026-06-06 09:09:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=6 grouped=6

## 2026-06-06 09:12:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=6

## 2026-06-06 09:12:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=6 traces=42

## 2026-06-06 09:12:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=6 grouped=6

## 2026-06-06 09:13:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-06 09:13:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-06 09:13:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-06 09:13:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-06 09:15:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=13 traces=156

## 2026-06-06 09:15:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=14 traces=168

## 2026-06-06 09:16:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=13 traces=156

## 2026-06-06 09:16:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=14 traces=168

## 2026-06-06 09:16:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=54 grouped=6

## 2026-06-06 09:19:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=6

## 2026-06-06 09:19:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=6 traces=42

## 2026-06-06 09:19:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=6 grouped=6

## 2026-06-06 09:20:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-06 09:20:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-06 09:20:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 6400
```

- status: started
- note: jobs=14

## 2026-06-06 09:20:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 6400
```

- status: started
- note: jobs=13

## 2026-06-06 09:22:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=14 traces=168

## 2026-06-06 09:22:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=13 traces=156

## 2026-06-06 09:22:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=13 traces=156

## 2026-06-06 09:22:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=14 traces=168

## 2026-06-06 09:23:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=54 grouped=6

## 2026-06-06 09:25:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=18

## 2026-06-06 09:26:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=162

## 2026-06-06 09:26:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=18 grouped=6

## 2026-06-06 09:32:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=2

## 2026-06-06 09:32:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=2 traces=14

## 2026-06-06 09:32:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=2 grouped=2

## 2026-06-06 09:35:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=2

## 2026-06-06 09:35:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=2 traces=14

## 2026-06-06 09:36:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=2 grouped=2

## 2026-06-06 09:38:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 1600
```

- status: started
- note: jobs=6

## 2026-06-06 09:38:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=6 traces=42

## 2026-06-06 09:39:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=6 grouped=6

## 2026-06-06 09:40:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=18

## 2026-06-06 09:40:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=162

## 2026-06-06 09:41:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=18 grouped=6

## 2026-06-06 09:52:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=24

## 2026-06-06 09:53:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=24 traces=216

## 2026-06-06 09:54:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=24 grouped=8

## 2026-06-06 09:57:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=21

## 2026-06-06 09:58:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=21 traces=189

## 2026-06-06 09:59:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=21 grouped=7

## 2026-06-06 10:12:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=18

## 2026-06-06 10:13:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=162

## 2026-06-06 10:13:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=18 grouped=6

## 2026-06-06 10:18:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=21

## 2026-06-06 10:19:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=21 traces=189

## 2026-06-06 10:20:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=21 grouped=7

## 2026-06-06 10:40:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=0

## 2026-06-06 10:40:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=0 traces=0

## 2026-06-06 10:42:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=9

## 2026-06-06 10:43:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=9 traces=81

## 2026-06-06 10:43:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=9 grouped=3

## 2026-06-06 10:44:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=21

## 2026-06-06 10:46:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=21 traces=189

## 2026-06-06 10:46:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=21 grouped=7

## 2026-06-06 10:52:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=21

## 2026-06-06 10:53:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=21 traces=189

## 2026-06-06 10:53:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=21 grouped=7

## 2026-06-06 11:13:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=21

## 2026-06-06 11:20:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=21 traces=189

## 2026-06-06 11:20:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=21 grouped=7

## 2026-06-06 11:25:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=24

## 2026-06-06 11:33:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=24 traces=216

## 2026-06-06 11:34:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=24 grouped=8

## 2026-06-06 11:37:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=21

## 2026-06-06 11:44:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=21 traces=189

## 2026-06-06 11:44:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=21 grouped=7

## 2026-06-06 12:00:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=24

## 2026-06-06 12:08:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=24 traces=216

## 2026-06-06 12:08:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=24 grouped=8

## 2026-06-06 12:15:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=21

## 2026-06-06 12:21:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=21 traces=189

## 2026-06-06 12:22:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=21 grouped=7

## 2026-06-06 12:29:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=21

## 2026-06-06 12:36:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=21 traces=189

## 2026-06-06 12:36:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=21 grouped=7

## 2026-06-06 12:39:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=18

## 2026-06-06 12:44:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=162

## 2026-06-06 12:44:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=18 grouped=6

## 2026-06-06 12:49:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=18

## 2026-06-06 12:53:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=162

## 2026-06-06 12:54:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=18 grouped=6

## 2026-06-06 15:41:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=12

## 2026-06-06 15:41:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=12 traces=108

## 2026-06-06 15:42:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=12 grouped=4

## 2026-06-06 15:44:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=21

## 2026-06-06 15:47:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=21 traces=189

## 2026-06-06 15:47:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=21 grouped=7

## 2026-06-06 15:53:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=27

## 2026-06-06 15:58:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=27 traces=243

## 2026-06-06 15:58:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=27 grouped=9

## 2026-06-06 16:03:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=33

## 2026-06-06 16:09:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=33 traces=297

## 2026-06-06 16:10:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=33 grouped=11

## 2026-06-06 16:13:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=42

## 2026-06-06 16:22:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=42 traces=378

## 2026-06-06 16:22:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=42 grouped=14

## 2026-06-06 16:41:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=11

## 2026-06-06 16:41:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=10

## 2026-06-06 16:41:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=11

## 2026-06-06 16:41:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=10

## 2026-06-06 16:44:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=10 traces=90

## 2026-06-06 16:44:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=10 traces=90

## 2026-06-06 16:44:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=11 traces=99

## 2026-06-06 16:44:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=11 traces=99

## 2026-06-06 16:44:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=42 grouped=14

## 2026-06-06 16:47:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=15

## 2026-06-06 16:47:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=15

## 2026-06-06 16:47:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=15

## 2026-06-06 16:47:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=15

## 2026-06-06 16:51:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=15 traces=135

## 2026-06-06 16:51:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=15 traces=135

## 2026-06-06 16:51:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=15 traces=135

## 2026-06-06 16:51:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=15 traces=135

## 2026-06-06 16:52:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=60 grouped=20

## 2026-06-06 16:59:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=18

## 2026-06-06 16:59:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=18

## 2026-06-06 16:59:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=18

## 2026-06-06 16:59:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=18

## 2026-06-06 17:04:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=18 traces=162

## 2026-06-06 17:04:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=18 traces=162

## 2026-06-06 17:04:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=18 traces=162

## 2026-06-06 17:04:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=18 traces=162

## 2026-06-06 17:04:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=72 grouped=24

## 2026-06-06 17:11:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=20

## 2026-06-06 17:11:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=20

## 2026-06-06 17:11:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=21

## 2026-06-06 17:11:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=20

## 2026-06-06 17:16:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=20 traces=180

## 2026-06-06 17:16:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=20 traces=180

## 2026-06-06 17:16:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=20 traces=180

## 2026-06-06 17:17:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=21 traces=189

## 2026-06-06 17:17:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=81 grouped=27

## 2026-06-06 17:24:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=22

## 2026-06-06 17:24:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=23

## 2026-06-06 17:24:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=23

## 2026-06-06 17:24:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=22

## 2026-06-06 17:30:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=22 traces=198

## 2026-06-06 17:30:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=22 traces=198

## 2026-06-06 17:30:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=23 traces=207

## 2026-06-06 17:31:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=23 traces=207

## 2026-06-06 17:31:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=90 grouped=30

## 2026-06-06 17:55:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=6

## 2026-06-06 17:55:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=6

## 2026-06-06 17:55:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=6

## 2026-06-06 17:55:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=6

## 2026-06-06 17:57:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=6 traces=54

## 2026-06-06 17:57:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=6 traces=54

## 2026-06-06 17:57:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=6 traces=54

## 2026-06-06 17:57:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=6 traces=54

## 2026-06-06 17:58:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=24 grouped=8

## 2026-06-06 18:01:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=6

## 2026-06-06 18:01:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=6

## 2026-06-06 18:01:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=6

## 2026-06-06 18:01:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=6

## 2026-06-06 18:03:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=6 traces=54

## 2026-06-06 18:03:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=6 traces=54

## 2026-06-06 18:03:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=6 traces=54

## 2026-06-06 18:03:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=6 traces=54

## 2026-06-06 18:04:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=24 grouped=8

## 2026-06-06 18:10:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 18:10:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 18:10:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 18:10:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 18:11:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 18:11:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 18:11:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 18:11:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 18:11:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=18 grouped=6

## 2026-06-06 18:17:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 18:17:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 18:17:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 18:17:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 18:18:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 18:18:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 18:19:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 18:19:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 18:19:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=18 grouped=6

## 2026-06-06 18:40:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 18:40:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 18:40:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 18:40:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 18:41:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 18:41:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 18:42:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 18:42:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 18:42:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=18 grouped=6

## 2026-06-06 18:46:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 18:46:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 18:46:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 18:46:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 18:47:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 18:47:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 18:48:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 18:48:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 18:48:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=18 grouped=6

## 2026-06-06 19:05:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 19:05:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 19:05:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=6

## 2026-06-06 19:05:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 19:07:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 19:07:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 19:07:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 19:08:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=6 traces=54

## 2026-06-06 19:08:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=21 grouped=7

## 2026-06-06 19:11:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 19:11:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 19:11:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=6

## 2026-06-06 19:11:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 19:13:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 19:13:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 19:13:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 19:14:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=6 traces=54

## 2026-06-06 19:14:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=21 grouped=7

## 2026-06-06 19:23:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 19:23:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=3

## 2026-06-06 19:23:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 19:23:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 19:23:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=3 traces=27

## 2026-06-06 19:24:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 19:24:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 19:24:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 19:25:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=15 grouped=5

## 2026-06-06 19:44:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=2

## 2026-06-06 19:44:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=2

## 2026-06-06 19:44:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=1

## 2026-06-06 19:44:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=1

## 2026-06-06 19:45:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=1 traces=9

## 2026-06-06 19:45:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=1 traces=9

## 2026-06-06 19:45:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=2 traces=18

## 2026-06-06 19:45:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=2 traces=18

## 2026-06-06 19:46:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=6 grouped=2

## 2026-06-06 19:47:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 19:47:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 19:47:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 19:47:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 19:48:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 19:48:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 19:49:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 19:49:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 19:49:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=18 grouped=6

## 2026-06-06 20:04:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=3

## 2026-06-06 20:04:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=3

## 2026-06-06 20:04:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=3

## 2026-06-06 20:04:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=3

## 2026-06-06 20:04:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=3 traces=27

## 2026-06-06 20:04:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=3 traces=27

## 2026-06-06 20:04:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=3 traces=27

## 2026-06-06 20:04:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=3 traces=27

## 2026-06-06 20:04:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=12 grouped=4

## 2026-06-06 20:07:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 20:07:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 20:07:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 20:07:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 20:08:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 20:08:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 20:09:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 20:09:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 20:09:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=18 grouped=6

## 2026-06-06 20:19:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2 --device cuda:2 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 20:19:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3 --device cuda:3 --steps 3200
```

- status: started
- note: jobs=4

## 2026-06-06 20:19:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1 --device cuda:1 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 20:19:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0 --device cuda:0 --steps 3200
```

- status: started
- note: jobs=5

## 2026-06-06 20:20:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 2
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 20:20:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 3
```

- status: completed
- note: rows=4 traces=36

## 2026-06-06 20:21:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 0
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 20:21:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --shard-index 1
```

- status: completed
- note: rows=5 traces=45

## 2026-06-06 20:22:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --scope mlp --merge-only
```

- status: completed
- note: rows=18 grouped=6
