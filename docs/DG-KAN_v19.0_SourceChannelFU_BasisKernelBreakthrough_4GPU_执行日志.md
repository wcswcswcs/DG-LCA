# DG-KAN v19.0 Source-Channel FU + Basis Kernel Breakthrough 执行日志

生成时间：2026-06-03 03:40:37 +08
计划文件：/home/chengshun.wang/DG-LCA/docs/DG-KAN_v19.0_SourceChannelFU_BasisKernelBreakthrough_4GPU_完整计划.md
结果目录：/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19
Python：/home/chengshun.wang/miniconda3/envs/kan/bin/python

## 实际执行命令
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_s03_truth_gate.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:0 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_source_channel_fu_matrix.py --stage mechanism-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 0 --device cuda:0 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_source_channel_fu_matrix.py --stage mechanism-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 1 --device cuda:1 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_source_channel_fu_matrix.py --stage mechanism-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 2 --device cuda:2 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_source_channel_fu_matrix.py --stage mechanism-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 3 --device cuda:3 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_basis_kernel_breakthrough.py --stage efficiency-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 0 --device cuda:0 --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_basis_kernel_breakthrough.py --stage efficiency-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 1 --device cuda:1 --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_basis_kernel_breakthrough.py --stage efficiency-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 2 --device cuda:2 --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_basis_kernel_breakthrough.py --stage efficiency-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 3 --device cuda:3 --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_source_channel_fu_matrix.py --stage horizon-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 0 --device cuda:0 --data-root data --horizon-all-rows 1 --horizon-steps 4800 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_source_channel_fu_matrix.py --stage horizon-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 1 --device cuda:1 --data-root data --horizon-all-rows 1 --horizon-steps 4800 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_source_channel_fu_matrix.py --stage horizon-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 2 --device cuda:2 --data-root data --horizon-all-rows 1 --horizon-steps 4800 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_source_channel_fu_matrix.py --stage horizon-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 3 --device cuda:3 --data-root data --horizon-all-rows 1 --horizon-steps 4800 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_late_rebound_rerun.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 0 --device cuda:0 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_late_rebound_rerun.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 1 --device cuda:1 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_late_rebound_rerun.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 2 --device cuda:2 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_late_rebound_rerun.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 3 --device cuda:3 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_merge_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --data-root data --horizon-all-rows 1 --horizon-steps 4800`

## 关键文件
- route: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_route_decision.json`
- S0 route: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v17_s0_route_decision.json`
- LineC-channel golden: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/linec_channel_golden_tests.csv`
- mechanism contract: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_mechanism_semantic_contract.csv`
- functional raw horizon matrix: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_raw_horizon_matrix.csv`
- source retention matrix: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_source_retention_matrix.csv`
- debt accounting matrix: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_debt_accounting_matrix.csv`
- debt metric availability: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_debt_metric_availability.csv`
- efficiency blocker table: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_efficiency_blocker_table.csv`
- efficiency waterfall: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_efficiency_waterfall.csv`
- kernel repair matrix: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_kernel_repair_matrix.csv`
- late rebound rerun matrix: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_late_rebound_rerun_matrix.csv`
- code packet: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_code_review_packet.zip`
- result bundle: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_results_bundle.zip`
- command journal: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/logs/v17_command_journal.md`

## 运行统计
- mechanism rows: 1197
- efficiency rows: 76
- kernel correctness rows: 6
- GPU snapshot rows: 4756
- max GPU memory used MB: 759.0
- max GPU util percent: 48.0

### Assignment Manifest Runtime

| gpu | runtime_sec_sum |
|---|---:|
| cuda:0 | 209.84 |
| cuda:1 | 210.30 |
| cuda:2 | 209.34 |
| cuda:3 | 204.97 |

### Horizon Shard Runtime

| shard | rows | runtime_sum_sec |
|---|---|---|
| h0 | 300 | 8013.44 |
| h1 | 299 | 8274.26 |
| h2 | 299 | 8586.17 |
| h3 | 299 | 8384.68 |

### Late Rebound Rerun Runtime

| shard | rows | runtime_sum_sec |
|---|---|---|
| r0 | 41 | 394.11 |
| r1 | 41 | 397.62 |
| r2 | 40 | 378.08 |
| r3 | 40 | 388.56 |

## Repro Notes
- v19 使用 `--horizon-all-rows 1 --horizon-steps 4800`，h100/h400/h800/h1600/h2400/h3200/h4800 均从 fresh trace/readback 汇总。
- LineC-fast 和 LineC-channel 均为 audit/readback；direction source 仍只来自 train-stream loss/gradient/update state。
- Debt recovery 对 CEp99/NLL/ECE/Brier/LineC-fast/LineC-channel/AUCtime 逐项记录，缺失项写 EvidenceIncomplete，不写 0 或成功。
- Basis repair rows 使用 `repair_variant` 区分 family-specific attempt；未达到 official/no-materialize gate 时 `promotion_allowed=0`。

## Continuation 2026-06-03：按计划修复 D-FOU/D-CHE Basis Forward Blocker

触发原因：`v19_route_decision.json` 在 2026-06-03 03:30:31 +08 显示 `D-FOU_pass=0`、`D-CHE_pass=0`，总 route 仍被 `D-FOU_efficiency_blocked;D-CHE_efficiency_blocked;MLP_no_retained_source` 卡住。按计划 Case F 推荐方向继续尝试 D-FOU low-frequency active bank / D-CHE recurrence no-materialize kernel repair。

### 代码修改

- `experiments/run_v17_common.py`
  - 新增 `v19_basis_repair_config()`，把 v19 repair labels 映射到已有真实 train-stream kernel path：`fourier_k2/k3/k4_triton_l3_matmul`、`cheby_k3/k4_triton_l3_matmul`、`cheby_k3_triton_l3_gradbuf`。
  - 扩充 v19 efficiency jobs：D-FOU 增加 k2/k3/k4 Triton/no-materialize variants；D-CHE 增加 k3/k4/gradbuf Triton variants。
  - efficiency row 新增 `manual_kernel_variant`、`manual_correctness_pass`、`manual_grad_relerr_max`、`manual_grad_cos_min`、`manual_output_max_abs_error`；若 manual correctness 不通过，finalizer 写 `ManualCorrectnessBlocked`，不允许 pass。
- `dgkan/profiling/efficiency_v17.py`
  - 新增 `use_manual_ce` profiling path；v19 efficiency shard 使用 `manual_ce_forward_cache()` / `manual_ce_backward_from_cache()` 计量 train-stream CE forward/backward，而不是只测普通 `model(x)` autograd path。

### 环境 / Smoke / Backup

- 默认 `python` smoke 失败，原因：当前 shell `/home/chengshun.wang/miniconda3/bin/python` 没有 `torch`，报 `ModuleNotFoundError: No module named 'torch'`。该命令不是实验数据，仅作为环境 blocker 记录。
- 正确解释器：`/home/chengshun.wang/miniconda3/envs/kan/bin/python`。
- 旧 efficiency/route 备份目录：`/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/pre_continuation_efficiency_20260603/`。
- smoke 目录：
  - `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_manual_kernel_repair/`：warmup=0，仅验证路径可执行，不作为 ratio 结论。
  - `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_manual_kernel_repair_warm/`：warmup=2/repeats=3，验证 D-FOU/D-CHE Triton path correctness/pass。

### Continuation 实际执行命令

- `/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/profiling/efficiency_v17.py experiments/run_v17_common.py`
- `mkdir -p results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/pre_continuation_efficiency_20260603 && cp results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v17_efficiency_truth_table*.csv results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_efficiency_*.csv results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_kernel_repair_matrix.csv results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_route_decision.json results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/pre_continuation_efficiency_20260603/`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_basis_kernel_breakthrough.py --stage efficiency-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 0 --device cuda:0 --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_basis_kernel_breakthrough.py --stage efficiency-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 1 --device cuda:1 --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_basis_kernel_breakthrough.py --stage efficiency-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 2 --device cuda:2 --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_basis_kernel_breakthrough.py --stage efficiency-shard --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --shard-count 4 --shard-index 3 --device cuda:3 --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_merge_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --data-root data --horizon-all-rows 1 --horizon-steps 4800`

### Continuation 输出文件

- GPU continuation snapshots：`results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_continuation_gpu_runtime_snapshots.csv`
- 更新后 route：`results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_route_decision.json`
- 更新后 efficiency truth table：`results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_efficiency_truth_table.csv`
- 更新后 kernel repair matrix：`results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_kernel_repair_matrix.csv`

## Continuation 2026-06-03：MLP Source Retention / Case B Functional Follow-up

触发原因：basis efficiency 已从 blocker 推进到 `E2-D-FOU-Pass;E2-D-CHE-Pass`，但 route 仍 `promotion_allowed=0`，剩余 blocker 为 `MLP_no_retained_source`。按计划 Case B / FU-H1 / FU-H2，继续尝试 FU scale、sparse alternating pulse、slow-state、dual-memory、schedule-free、PopRisk 和 train-split signal-channel filter。

### 代码修改

- 新增 `experiments/run_v19_functional_continuation.py`：MLP-only targeted continuation runner；输出 raw matrix、trace、summary；支持 `--run-label`、`--spec-ids`、`--merge-only`，避免重跑全 1197-row 矩阵。
- 修改 `dgkan/fu/mechanisms.py`：新增 `M15-LineCFilteredAlternatingFU`，注册到 mechanism contract；update tensor 使用 alternating FU pulse。
- 修改 `experiments/run_v17_common.py`：`train_one()` 新增 M15 train-split filter branch；只用 train split A/B 的临时 CE improvement 与 LineC leak 判断 pulse 是否 commit，拒绝时回退 SGD step；trace 记录 `LineC_filter_trials/accepts/accept_rate/last_*`。

### Functional continuation 实际执行命令

- `/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v19_functional_continuation.py`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:0 --data-root data --shard-count 4 --shard-index 0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:1 --data-root data --shard-count 4 --shard-index 1`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:2 --data-root data --shard-count 4 --shard-index 2`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:3 --data-root data --shard-count 4 --shard-index 3`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:{0,1,2,3} --data-root data --shard-count 4 --shard-index {0,1,2,3} --run-label r2 --spec-ids M5-alt100-fu0p0005,M5-alt100-fu0p00025,M5-alt200-fu0p0005,M2-fu0p0001,M2-fu0p00005`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:{0,1,2,3} --data-root data --shard-count 4 --shard-index {0,1,2,3} --run-label r3 --spec-ids M15-linec-alt50-fu0p0005,M15-linec-alt100-fu0p0005`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19`

注：上面 `{0,1,2,3}` 表示同一模板对四张卡分别执行，实际 shard commands 已在 shell session 中分别提交。

### Functional continuation 输出文件

- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_matrix.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_traces.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_summary.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_gpu_runtime_snapshots.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r2_gpu_runtime_snapshots.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r3_gpu_runtime_snapshots.csv`

### Functional continuation 结果摘要

- measured rows: 162/162
- retained candidates: 0/15
- best grouped h4800: `M15-linec-alt50-fu0p0005`，h4800 mean source = 0.12667612234751383，但 h800 mean source = -0.0853961706161499，不能算 retained。
- `M2-fu0p0001` h800 mean source = 0.2674288551012675，但 h1600 mean source = -0.14676260948181152，仍 washout。
- `M15-linec-alt50-fu0p0005` accept rate mean = 0.2696759259259259；`M15-linec-alt100-fu0p0005` accept rate mean = 0.21296296296296294。

## Continuation 2026-06-03：M16 Two-Phase Writer / Case B -> Case C Follow-up

触发原因：r1-r3 functional continuation 仍没有 retained candidate。按计划 Case B / FU-H2，继续尝试 slow-state/source retention 方向。M16 设计为两阶段 writer：前 800/1200 step 使用 M2 momentum+FU warmup 建立 h800 source，warmup 后改用 LineC-filtered sparse alternating pulse，尝试保住 h1600/h3200/h4800。r4 找到 MLP generic retained source 后，按 Case C 对 D-CHE/D-FOU 做 same-carrier controls 检查。

### 代码修改

- `dgkan/fu/mechanisms.py`
  - 新增 `M16-TwoPhaseMomentumThenLineCFU`，注册到 `MECHANISMS` 和 semantic contract。
  - M16 update tensor 标记为 `trajectory_boundary` / `train_stream_two_phase_momentum_then_linec_fu`。
- `experiments/run_v17_common.py`
  - `train_one()` 新增 M16 branch：`source_warmup_steps` 之前执行 M2 momentum+FU；之后每 `alt_period` step 做 M5 alternating pulse 的 train split A/B 临时 CE improvement + LineC leak filter，accept 才 commit，reject 回退 SGD。
- `experiments/run_v19_functional_continuation.py`
  - 新增 `--source-warmup-steps`、M16 spec sweep、`--carriers`。
  - `enrich_sources()` 改为 same-carrier controls key：`carrier,dataset,seed,horizon`。
  - `summarize()` 改为 `carrier + continuation_id` 分组。
  - 修复 `--spec-ids` filter：同时匹配 `continuation_id` 和 `mechanism`，避免 `CTRL-AdamW` 这类 control mechanism 名无法匹配。

### Smoke / Compile

- `/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v19_functional_continuation.py`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m16 --device cuda:0 --data-root data --datasets MNIST --seeds 0 --steps 20 --shard-count 1 --shard-index 0 --run-label m16smoke --spec-ids M16-warm800-alt100-fu0p0001`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m16`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m16_dche --device cuda:0 --data-root data --carriers D-CHE --datasets MNIST --seeds 0 --steps 20 --shard-count 1 --shard-index 0 --run-label m16dchesmoke --spec-ids M16-warm800-alt50-fu0p0001,CTRL-SGD`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m16_dche`

### r4 MLP M16 正式执行命令

GPU monitor：

- `bash -lc 'out=/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r4_gpu_runtime_snapshots.csv; stop=/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r4_gpu_monitor.stop; rm -f "$stop"; printf "timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n" > "$out"; while [ ! -f "$stop" ]; do ts=$(date +"%Y-%m-%d %H:%M:%S %z"); nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r idx mem util; do idx=${idx// /}; mem=${mem// /}; util=${util// /}; printf "%s,%s,%s,%s\n" "$ts" "$idx" "$mem" "$util" >> "$out"; done; sleep 5; done'`

Shard template（对 `N=0,1,2,3` 分别执行，`device=cuda:N`、`shard-index=N`）：

- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:N --data-root data --shard-count 4 --shard-index N --run-label r4 --spec-ids M16-warm800-alt50-fu0p0001,M16-warm800-alt100-fu0p0001,M16-warm1200-alt100-fu0p0001,M16-warm800-alt100-fu0p00005`
- `touch /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r4_gpu_monitor.stop`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19`

注：第一次启动 r4 monitor 的 shell quoting 写错，命令返回 `unexpected EOF while looking for matching '"'`，未进入实验计量；随后用上面的 monitor loop 重跑并真实写入 r4 snapshots。

### r5/r6 D-CHE/D-FOU same-carrier check

r5 先执行 D-CHE/D-FOU M16：

- `bash -lc 'out=/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r5_gpu_runtime_snapshots.csv; stop=/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r5_gpu_monitor.stop; rm -f "$stop"; printf "timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n" > "$out"; while [ ! -f "$stop" ]; do ts=$(date +"%Y-%m-%d %H:%M:%S %z"); nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r idx mem util; do idx=${idx// /}; mem=${mem// /}; util=${util// /}; printf "%s,%s,%s,%s\n" "$ts" "$idx" "$mem" "$util" >> "$out"; done; sleep 5; done'`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:N --data-root data --carriers D-CHE,D-FOU --shard-count 4 --shard-index N --run-label r5 --spec-ids CTRL-AdamW,CTRL-SGD,CTRL-RecoveryOnly,CTRL-NoOpMatchedOverhead,CTRL-RandomMatchedNorm,M16-warm800-alt50-fu0p0001`
- `touch /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r5_gpu_monitor.stop`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19`

r5 blocker：由于 controls 的 `continuation_id` 为 `CTRL-CTRL-AdamW` 等，而命令按 `mechanism=CTRL-AdamW` 写 `--spec-ids`，旧 filter 只匹配 continuation_id，导致 r5 实际只跑了 18 条 M16 rows，controls 未跑，D-CHE/D-FOU source 暂时 blank。修复 filter 后补跑 r6 controls。

r6 controls：

- `bash -lc 'out=/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r6_gpu_runtime_snapshots.csv; stop=/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r6_gpu_monitor.stop; rm -f "$stop"; printf "timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n" > "$out"; while [ ! -f "$stop" ]; do ts=$(date +"%Y-%m-%d %H:%M:%S %z"); nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r idx mem util; do idx=${idx// /}; mem=${mem// /}; util=${util// /}; printf "%s,%s,%s,%s\n" "$ts" "$idx" "$mem" "$util" >> "$out"; done; sleep 5; done'`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:N --data-root data --carriers D-CHE,D-FOU --shard-count 4 --shard-index N --run-label r6 --spec-ids CTRL-AdamW,CTRL-SGD,CTRL-RecoveryOnly,CTRL-NoOpMatchedOverhead,CTRL-RandomMatchedNorm`
- `touch /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r6_gpu_monitor.stop`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19`

### r4-r6 输出文件 / 结果摘要

- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_matrix.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_summary.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_traces.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r4_gpu_runtime_snapshots.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r5_gpu_runtime_snapshots.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r6_gpu_runtime_snapshots.csv`
- r4 rows=36, runtime_sum_sec=211.611, max_mem=693 MB, max_util=22。
- r5 rows=18, runtime_sum_sec=174.554, max_mem=693 MB, max_util=10。
- r6 rows=90, runtime_sum_sec=857.819, max_mem=693 MB, max_util=15。
- `MLP/M16-warm800-alt50-fu0p0001` retained=1：h800=0.01937996016608344, h1600=0.0803578429751926, h3200=0.07967897256215413, h4800=0.07385419474707709。
- `D-CHE/M16-warm800-alt50-fu0p0001` retained=0：h800=-0.8855499625205994, h1600=-0.77398638592826, h3200=-0.6744108001391093, h4800=-0.5914939045906067。
- `D-FOU/M16-warm800-alt50-fu0p0001` retained=0：h800=-0.9329363306363424, h1600=-0.8675869637065463, h3200=-0.7608865963088142, h4800=-0.6862555742263794。
- 更新后 route：`R-MLPGenericRetained-KANCarrierBlocked`，`promotion_allowed=0`。

### Continuation 审计包

- `mkdir -p results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_continuation_code_review_packet`
- `cp --parents dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v19_functional_continuation.py docs/DG-KAN_v19.0_SourceChannelFU_BasisKernelBreakthrough_4GPU_执行日志.md docs/DG-KAN_v19.0_SourceChannelFU_BasisKernelBreakthrough_4GPU_实验结果复盘.md results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_route_decision.json results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_next_hypothesis_queue.csv results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_next_hypothesis_queue.md results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_continuation_code_review_packet/`
- `bash -lc 'cd results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_continuation_code_review_packet && find . -type f -print | sort | while read -r p; do sha256sum "$p"; done > packet_sha256_manifest.txt'`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python -m zipfile -c results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_continuation_code_review_packet.zip results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_continuation_code_review_packet`
- packet dir: `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_continuation_code_review_packet/`
- packet zip: `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_continuation_code_review_packet.zip`

## Continuation 2026-06-03：Case C Carrier Block Attempts r7-r10

触发原因：r4-r6 找到 `MLP/M16-warm800-alt50-fu0p0001` retained source，但 D-CHE/D-FOU same-carrier M16 为强负。按 v19 计划 Case C / FU-H3，继续尝试 KAN carrier actuator、repaired basis variant、matrix/block FU、split-consensus source estimator。所有结果只按真实 CSV 写入，不把 MLP generic source 写成 KAN breakthrough。

### 代码修改

- `dgkan/fu/mechanisms.py`
  - 新增 `M17-ReadoutCarrierTwoPhaseLineCFU`：只作用 readout/w2/classifier carrier，避免全参数 trajectory pulse 直接打散 KAN basis。
  - 新增 `M18-CarrierLowRankBlockFU`：对 `w1/w2/readout` 参数块做 rank-4 SVD low-rank block update；`w1` carrier half-scale，`w2/readout` full-scale。
  - 新增 `CTRL-RandomSameRankBlock`：同 rank 随机 block control，作为 H3 matched control。
  - 新增 `M19-SplitConsensusLowRankBlockFU`：把当前 train batch 切成两个 split，只提交 split sign-agreement 的 rank-4 low-rank carrier block；不读取 validation/test/future。
- `experiments/run_v17_common.py`
  - `train_one()` 将 M17 纳入 two-phase branch：warmup 和 pulse 均使用 readout-carrier update，并继续使用 train split LineC filter。
- `experiments/run_v19_functional_continuation.py`
  - 支持 `--basis-repair-variant`，summary 按 `carrier + basis_repair_variant + continuation_id` 分组。
  - controls key 改为 `carrier,basis_repair_variant,dataset,seed,horizon`，防止 R0 controls 被误用于 repaired variants。
  - continuation specs 新增 M17/M18/M19 和 `CTRL-RandomSameRankBlock`。

### Compile / Smoke

- `/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v19_functional_continuation.py`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m17_dche --device cuda:0 --data-root data --carriers D-CHE --datasets MNIST --seeds 0 --steps 20 --shard-count 1 --shard-index 0 --run-label m17dchesmoke --spec-ids M17-readout-warm800-alt50-fu0p0001,CTRL-SGD`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m17_dche`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_repaired_dche --device cuda:0 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --datasets MNIST --seeds 0 --steps 20 --shard-count 1 --shard-index 0 --run-label repairedsmoke --spec-ids M17-readout-warm800-alt50-fu0p0001,CTRL-SGD`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_repaired_dche`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m18_dche --device cuda:0 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --datasets MNIST --seeds 0 --steps 20 --shard-count 1 --shard-index 0 --run-label m18dchesmoke --spec-ids M18-lowrank-r4-fu0p0001,CTRL-RandomSameRankBlock,CTRL-SGD`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m18_dche`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m19_dche --device cuda:0 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --datasets MNIST --seeds 0 --steps 20 --shard-count 1 --shard-index 0 --run-label m19dchesmoke --spec-ids M19-split-lowrank-r4-fu0p0001,CTRL-RandomSameRankBlock,CTRL-SGD`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m19_dche`

### r7 R0-current M17 Readout Carrier

GPU monitor：

- `bash -lc 'out=/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r7_gpu_runtime_snapshots.csv; stop=/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r7_gpu_monitor.stop; rm -f "$stop"; printf "timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n" > "$out"; while [ ! -f "$stop" ]; do ts=$(date +"%Y-%m-%d %H:%M:%S %z"); nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r idx mem util; do idx=${idx// /}; mem=${mem// /}; util=${util// /}; printf "%s,%s,%s,%s\n" "$ts" "$idx" "$mem" "$util" >> "$out"; done; sleep 5; done'`

Shard template（`N=0,1,2,3` 分别执行）：

- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:N --data-root data --carriers D-CHE,D-FOU --shard-count 4 --shard-index N --run-label r7 --spec-ids M17-readout-warm800-alt50-fu0p0001,M17-readout-warm800-alt100-fu0p0001,M17-readout-warm800-alt50-fu0p00005,M17-readout-warm1200-alt50-fu0p0001`
- `touch /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r7_gpu_monitor.stop`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19`

### r8 Repaired Variant M16/M17 Carrier Check

GPU monitor：

- `bash -lc 'out=/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r8_gpu_runtime_snapshots.csv; stop=/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r8_gpu_monitor.stop; rm -f "$stop"; printf "timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n" > "$out"; while [ ! -f "$stop" ]; do ts=$(date +"%Y-%m-%d %H:%M:%S %z"); nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r idx mem util; do idx=${idx// /}; mem=${mem// /}; util=${util// /}; printf "%s,%s,%s,%s\n" "$ts" "$idx" "$mem" "$util" >> "$out"; done; sleep 5; done'`

D-CHE shards：

- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:0 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --shard-count 2 --shard-index 0 --run-label r8che --spec-ids CTRL-AdamW,CTRL-SGD,CTRL-RecoveryOnly,CTRL-NoOpMatchedOverhead,CTRL-RandomMatchedNorm,M16-warm800-alt50-fu0p0001,M17-readout-warm800-alt50-fu0p0001,M17-readout-warm800-alt50-fu0p00005`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:1 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --shard-count 2 --shard-index 1 --run-label r8che --spec-ids CTRL-AdamW,CTRL-SGD,CTRL-RecoveryOnly,CTRL-NoOpMatchedOverhead,CTRL-RandomMatchedNorm,M16-warm800-alt50-fu0p0001,M17-readout-warm800-alt50-fu0p0001,M17-readout-warm800-alt50-fu0p00005`

D-FOU shards：

- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:2 --data-root data --carriers D-FOU --basis-repair-variant FOU-R2-low-frequency-k2-stream --shard-count 2 --shard-index 0 --run-label r8fou --spec-ids CTRL-AdamW,CTRL-SGD,CTRL-RecoveryOnly,CTRL-NoOpMatchedOverhead,CTRL-RandomMatchedNorm,M16-warm800-alt50-fu0p0001,M17-readout-warm800-alt50-fu0p0001,M17-readout-warm800-alt50-fu0p00005`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:3 --data-root data --carriers D-FOU --basis-repair-variant FOU-R2-low-frequency-k2-stream --shard-count 2 --shard-index 1 --run-label r8fou --spec-ids CTRL-AdamW,CTRL-SGD,CTRL-RecoveryOnly,CTRL-NoOpMatchedOverhead,CTRL-RandomMatchedNorm,M16-warm800-alt50-fu0p0001,M17-readout-warm800-alt50-fu0p0001,M17-readout-warm800-alt50-fu0p00005`

- `touch /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r8_gpu_monitor.stop`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19`

### r9/r10 H3 Low-Rank Block Attempts

r9 M18 + same-rank control：

- `bash -lc 'out=/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r9_gpu_runtime_snapshots.csv; stop=/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r9_gpu_monitor.stop; rm -f "$stop"; printf "timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n" > "$out"; while [ ! -f "$stop" ]; do ts=$(date +"%Y-%m-%d %H:%M:%S %z"); nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r idx mem util; do idx=${idx// /}; mem=${mem// /}; util=${util// /}; printf "%s,%s,%s,%s\n" "$ts" "$idx" "$mem" "$util" >> "$out"; done; sleep 5; done'`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:0 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --shard-count 2 --shard-index 0 --run-label r9che --spec-ids CTRL-RandomSameRankBlock,M18-lowrank-r4-fu0p0001,M18-lowrank-r4-fu0p00005,M18-lowrank-r4-fu0p00025`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:1 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --shard-count 2 --shard-index 1 --run-label r9che --spec-ids CTRL-RandomSameRankBlock,M18-lowrank-r4-fu0p0001,M18-lowrank-r4-fu0p00005,M18-lowrank-r4-fu0p00025`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:2 --data-root data --carriers D-FOU --basis-repair-variant FOU-R2-low-frequency-k2-stream --shard-count 2 --shard-index 0 --run-label r9fou --spec-ids CTRL-RandomSameRankBlock,M18-lowrank-r4-fu0p0001,M18-lowrank-r4-fu0p00005,M18-lowrank-r4-fu0p00025`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:3 --data-root data --carriers D-FOU --basis-repair-variant FOU-R2-low-frequency-k2-stream --shard-count 2 --shard-index 1 --run-label r9fou --spec-ids CTRL-RandomSameRankBlock,M18-lowrank-r4-fu0p0001,M18-lowrank-r4-fu0p00005,M18-lowrank-r4-fu0p00025`
- `touch /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r9_gpu_monitor.stop`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19`

r10 M19 split-consensus low-rank：

- `bash -lc 'out=/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r10_gpu_runtime_snapshots.csv; stop=/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r10_gpu_monitor.stop; rm -f "$stop"; printf "timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n" > "$out"; while [ ! -f "$stop" ]; do ts=$(date +"%Y-%m-%d %H:%M:%S %z"); nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r idx mem util; do idx=${idx// /}; mem=${mem// /}; util=${util// /}; printf "%s,%s,%s,%s\n" "$ts" "$idx" "$mem" "$util" >> "$out"; done; sleep 5; done'`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:0 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --shard-count 2 --shard-index 0 --run-label r10che --spec-ids M19-split-lowrank-r4-fu0p0001,M19-split-lowrank-r4-fu0p00005,M19-split-lowrank-r4-fu0p00025`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:1 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --shard-count 2 --shard-index 1 --run-label r10che --spec-ids M19-split-lowrank-r4-fu0p0001,M19-split-lowrank-r4-fu0p00005,M19-split-lowrank-r4-fu0p00025`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:2 --data-root data --carriers D-FOU --basis-repair-variant FOU-R2-low-frequency-k2-stream --shard-count 2 --shard-index 0 --run-label r10fou --spec-ids M19-split-lowrank-r4-fu0p0001,M19-split-lowrank-r4-fu0p00005,M19-split-lowrank-r4-fu0p00025`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19 --device cuda:3 --data-root data --carriers D-FOU --basis-repair-variant FOU-R2-low-frequency-k2-stream --shard-count 2 --shard-index 1 --run-label r10fou --spec-ids M19-split-lowrank-r4-fu0p0001,M19-split-lowrank-r4-fu0p00005,M19-split-lowrank-r4-fu0p00025`
- `touch /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_r10_gpu_monitor.stop`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir /home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19`

### r7-r10 输出 / Runtime

- final continuation matrix: `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_matrix.csv`
- final continuation summary: `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_summary.csv`
- final continuation traces: `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_traces.csv`
- matrix rows: 666
- summary grouped rows: 47
- trace rows: 6660

| round | rows | runtime_sum_sec | gpu_snapshot_rows | max_mem_mb | max_util |
|---|---:|---:|---:|---:|---:|
| r7 M17 R0-current | 72 | 685.542 | 168 | 693.0 | 10.0 |
| r8 repaired M16/M17 | 144 | 1577.922 | 364 | 693.0 | 20.0 |
| r9 repaired M18 + same-rank control | 72 | 1399.141 | 360 | 759.0 | 54.0 |
| r10 repaired M19 split-consensus | 54 | 2243.965 | 508 | 759.0 | 36.0 |

### r7-r10 结果摘要

- retained_total: 1/47，仍然只来自 `MLP/M16-warm800-alt50-fu0p0001`。
- non-MLP retained candidates: 0。
- KAN M17 R0-current retained candidates: 0。
- KAN repaired M16/M17 retained candidates: 0。
- KAN repaired M18 retained candidates: 0。
- KAN repaired M19 retained candidates: 0。
- route 仍为 `R-MLPGenericRetained-KANCarrierBlocked`，`promotion_allowed=0`。

下一步备注：我没有继续硬写 H4 operator-level mechanism，因为真实 H4 需要 function-space projection / ActuationR2 solver；当前若只写 `0.5 * gradient` 或低秩壳子，会违反计划中“不允许 mechanism 名字大于实现”的约束。

### Bundle / Packet Refresh

- `cp --parents dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v19_functional_continuation.py docs/DG-KAN_v19.0_SourceChannelFU_BasisKernelBreakthrough_4GPU_执行日志.md docs/DG-KAN_v19.0_SourceChannelFU_BasisKernelBreakthrough_4GPU_实验结果复盘.md results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_route_decision.json results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_next_hypothesis_queue.csv results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_next_hypothesis_queue.md results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_continuation_code_review_packet/`
- `bash -lc 'cd results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_continuation_code_review_packet && find . -type f ! -name packet_sha256_manifest.txt -print | sort | while read -r p; do sha256sum "$p"; done > packet_sha256_manifest.txt'`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python -m zipfile -c results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_continuation_code_review_packet.zip results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_continuation_code_review_packet`
- `zip -r -q /tmp/v19_results_bundle_new.zip . -x ./v19_results_bundle.zip` 失败，原因：系统未安装 `zip`。
- results bundle fallback command:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from pathlib import Path
import zipfile
base = Path('results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19')
out = base / 'v19_results_bundle.zip'
tmp = Path('/tmp/v19_results_bundle_new.zip')
with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(base.rglob('*')):
        if not p.is_file():
            continue
        if p.resolve() == out.resolve():
            continue
        zf.write(p, p.relative_to(base))
tmp.replace(out)
print(out, out.stat().st_size)
PY
```
- refreshed bundle: `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_results_bundle.zip`，大小 23M，更新时间 2026-06-03 05:18 +08。

## Continuation 2026-06-03：H4 Function-Space Actuation Solver Diagnostic

触发原因：r7-r10 已证明 KAN carrier block 不是 M16/M17/M18/M19 小修能解决。按计划 FU-H4，继续实现真实 function-space projection / actuation solver；若 `ActuationR2 < 0.20`，计划要求不跑 long horizon，先修 solver。

### 代码修改

- `dgkan/fu/core.py`
  - `UpdateTensor` 新增可选 `diagnostics` 字段，用于把 actuation solver 的真实 readback 写入 trace。
- `dgkan/fu/mechanisms.py`
  - 新增 `M20-TrainSplitFunctionSpaceActuationFU`。
  - M20 使用当前 train batch 切分 B1/B2/B3，不读取 validation/test/future。
  - M20 构造候选参数子空间 `U`，有限差分测 `J_U` 对 B1/B2 logits 的作用，解 ridge least-squares，再用 B3 做 safety readback。
  - 修复轮次：
    - v1：初始 finite-diff + normalized solved update。
    - v2：按真实 commit scale `1e-4` 解 target，并移除强制归一化，保留 norm cap。
    - v3：加入 readout-class directions。
    - v4：加入 cap sweep / line-search，按 B1/B2 gain、B3 safety、ActuationR2 选 scale。
    - v5：加入 readout top-entry directions。
    - v6：cap sweep 扩到 30000，同时保留 B3 safety penalty。
- `experiments/run_v17_common.py`
  - trace/final row 新增 `ActuationR2`、`ActuationCosine`、`B1_gain`、`B2_transfer_gain`、`B3_safety_gain`、`projection_residual_norm`、`function_displacement_norm`、`operator_rank`、`operator_cap_ratio`、`operator_clamp_ratio`、`operator_commit_scale`、`operator_status`。
- `experiments/run_v19_functional_continuation.py`
  - specs 新增 `M20-h4-actuation-r4-fu0p0001`、`M20-h4-actuation-r4-fu0p00005`、`M20-h4-actuation-r4-fu0p00025`。

### Compile / Smoke Commands

- `/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/core.py dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v19_functional_continuation.py`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m20_dche --device cuda:0 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --datasets MNIST --seeds 0 --steps 20 --shard-count 1 --shard-index 0 --run-label m20dchesmoke --spec-ids M20-h4-actuation-r4-fu0p0001,CTRL-RandomSameRankBlock,CTRL-SGD`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m20_dche`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m20_dche_v2 --device cuda:0 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --datasets MNIST --seeds 0 --steps 20 --shard-count 1 --shard-index 0 --run-label m20dchesmokev2 --spec-ids M20-h4-actuation-r4-fu0p0001,CTRL-RandomSameRankBlock,CTRL-SGD`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m20_dche_v2`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m20_dche_v3 --device cuda:0 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --datasets MNIST --seeds 0 --steps 20 --shard-count 1 --shard-index 0 --run-label m20dchesmokev3 --spec-ids M20-h4-actuation-r4-fu0p0001,CTRL-RandomSameRankBlock,CTRL-SGD`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m20_dche_v3`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m20_dche_v4 --device cuda:0 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --datasets MNIST --seeds 0 --steps 20 --shard-count 1 --shard-index 0 --run-label m20dchesmokev4 --spec-ids M20-h4-actuation-r4-fu0p0001,CTRL-RandomSameRankBlock,CTRL-SGD`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m20_dche_v4`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m20_dche_v5 --device cuda:0 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --datasets MNIST --seeds 0 --steps 20 --shard-count 1 --shard-index 0 --run-label m20dchesmokev5 --spec-ids M20-h4-actuation-r4-fu0p0001,CTRL-RandomSameRankBlock,CTRL-SGD`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m20_dche_v5`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m20_dche_v6 --device cuda:0 --data-root data --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton --datasets MNIST --seeds 0 --steps 20 --shard-count 1 --shard-index 0 --run-label m20dchesmokev6 --spec-ids M20-h4-actuation-r4-fu0p0001,CTRL-RandomSameRankBlock,CTRL-SGD`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v19_functional_continuation.py --merge-only --out-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/smoke_functional_m20_dche_v6`
- aggregate smoke traces:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv
from pathlib import Path
base = Path('results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu')
off = base / 'official_v19'
rows = []
for label, rel in [
    ('v1_initial_fd_normed', 'smoke_functional_m20_dche'),
    ('v2_commit_scale_cap100', 'smoke_functional_m20_dche_v2'),
    ('v3_class_readout_basis', 'smoke_functional_m20_dche_v3'),
    ('v4_cap_sweep_3000', 'smoke_functional_m20_dche_v4'),
    ('v5_top_entry_basis', 'smoke_functional_m20_dche_v5'),
    ('v6_cap_sweep_30000', 'smoke_functional_m20_dche_v6'),
]:
    p = base / rel / 'v19_functional_continuation_traces.csv'
    if not p.exists():
        continue
    with p.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('mechanism') == 'M20-TrainSplitFunctionSpaceActuationFU' and r.get('ActuationR2'):
                rows.append({
                    'repair_round': label,
                    'step': r.get('step'),
                    'carrier': r.get('carrier'),
                    'basis_repair_variant': r.get('basis_repair_variant'),
                    'ActuationR2': r.get('ActuationR2'),
                    'ActuationCosine': r.get('ActuationCosine'),
                    'B1_gain': r.get('B1_gain'),
                    'B2_transfer_gain': r.get('B2_transfer_gain'),
                    'B3_safety_gain': r.get('B3_safety_gain'),
                    'projection_residual_norm': r.get('projection_residual_norm'),
                    'function_displacement_norm': r.get('function_displacement_norm'),
                    'operator_rank': r.get('operator_rank'),
                    'operator_cap_ratio': r.get('operator_cap_ratio'),
                    'operator_clamp_ratio': r.get('operator_clamp_ratio'),
                    'operator_parameter_norm': r.get('operator_parameter_norm'),
                    'operator_status': r.get('operator_status'),
                })
out = off / 'v19_h4_actuation_solver_smoke.csv'
with out.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['repair_round'])
    w.writeheader()
    w.writerows(rows)
print(out, len(rows))
PY
```

### H4 Diagnostic Result

- official diagnostic artifact: `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h4_actuation_solver_smoke.csv`
- smoke diagnostic rows: 12
- best ActuationR2: 0.09360003471374512
- best ActuationCosine: 0.30666324496269226
- best B1_gain: 0.07385778427124023
- best B2_transfer_gain: 0.07891631126403809
- best B3_safety_gain: 0.03797030448913574
- best operator_rank: 36
- best operator_cap_ratio: 30000.0
- conclusion: ActuationR2 remains below 0.20, so long-horizon H4 run was not executed per plan.

## 2026-06-03 H4/M21-M26 Continuation

### Why Continued

- M20 的 train-split actuation solver 没有达到计划里 `ActuationR2 >= 0.20` 的继续门槛。
- 但 v19 目标仍未达成，且 blocker 已经缩小到 KAN carrier 的 function-space actuation/retention；因此继续按计划推荐方向尝试更强的 readout-space solver、cap/schedule、warmup、AdamW-coupled 和 gated repair。
- 本段所有新增结果来自 fresh run，不把 smoke row 当 official retained source。

### Code Changes Before Running

- `dgkan/fu/core.py`
  - `UpdateTensor` 新增 `diagnostics` 字段，用于把 function-space solver 的 ActuationR2/B1/B2/B3/gate 证据写入 trace。
- `dgkan/fu/mechanisms.py`
  - 新增 `M21-ExactReadoutFunctionSpaceActuationFU`。
  - 新增 `M22-ExactReadoutHighCapScheduledFU`。
  - 新增 `M23-ExactReadoutUltraCapScheduledFU`。
  - 新增 `M24-WarmupExactReadoutUltraCapFU`。
  - 新增 `M25-AdamWExactReadoutUltraCapFU`。
  - 新增 `M26-GatedAdamWExactReadoutFU`。
  - 所有 M21-M26 的 direction/source 均来自 train batch/split；没有读取 validation/test/future label。
- `experiments/run_v17_common.py`
  - 修复 M20/M21/M22/M23 之前忽略 `alt_period` 的 blocker，使 scheduled operator 只在 pulse step commit。
  - 新增 M24 warmup branch、M25 AdamW+exact-readout branch、M26 gated branch。
  - trace/final row 新增 `ActuationR2`、`ActuationCosine`、`B1_gain`、`B2_transfer_gain`、`B3_safety_gain`、`function_displacement_norm`、`operator_gate_accept` 等审计字段。
- `experiments/run_v19_functional_continuation.py`
  - 新增 M21-M26 continuation specs。
  - merge-only 汇总后更新 official summary、route decision、next hypothesis queue。

### Compile Command

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/core.py \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v19_functional_continuation.py
```

### Smoke Commands

通用说明：每个 smoke 后均执行一次 `--merge-only`，输出写在对应 `results/.../smoke_*` 目录。

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir $BASE/smoke_functional_m21_dche \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 20 \
  --shard-count 1 --shard-index 0 --run-label m21dchesmoke \
  --spec-ids M21-exact-readout-fu0p0001,CTRL-RandomSameRankBlock,CTRL-SGD

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir $BASE/smoke_functional_m21_dche_scheduled \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 20 \
  --shard-count 1 --shard-index 0 --run-label m21dchescheduled \
  --spec-ids M21-exact-readout-fu0p0001,CTRL-RandomSameRankBlock,CTRL-SGD

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir $BASE/smoke_functional_m22_dche \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 20 \
  --shard-count 1 --shard-index 0 --run-label m22dchesmoke \
  --spec-ids M22-exact-readout-highcap-fu0p0001,CTRL-RandomSameRankBlock,CTRL-SGD

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir $BASE/smoke_functional_m23_dche \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 20 \
  --shard-count 1 --shard-index 0 --run-label m23dchesmoke \
  --spec-ids M23-exact-readout-ultracap-fu0p0001,CTRL-RandomSameRankBlock,CTRL-SGD

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir $BASE/smoke_functional_alt50_grid_dche \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 800 \
  --shard-count 1 --shard-index 0 --run-label alt50griddche \
  --spec-ids M22-exact-readout-highcap-alt50-fu0p0001,M23-exact-readout-ultracap-alt50-fu0p0001,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir $BASE/smoke_functional_m24_dche \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label m24dchesmoke \
  --spec-ids M24-warm400-ultracap-alt50-fu0p0001,M24-warm800-ultracap-alt50-fu0p0001,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir $BASE/smoke_functional_m25_dche \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label m25dchesmoke \
  --spec-ids M25-adamw-ultracap-alt50-fu0p0001,M25-adamw-ultracap-alt100-fu0p0001,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir $BASE/smoke_functional_m26_dche \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label m26dchesmoke \
  --spec-ids M25-adamw-ultracap-alt100-fu0p0001,M26-gated-adamw-ultracap-alt100-fu0p0001,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock
```

### Smoke Results

| smoke | key result | decision |
|---|---|---|
| M21 unscheduled | step20 ActuationR2=0.44115543365478516 | exposed scheduling blocker: M21 was committing every step |
| M21 scheduled | step20 ActuationR2=0.10382378101348877 | below 0.20, no long run from this exact config |
| M22 high-cap | step20 ActuationR2=0.3892034888267517 | strong enough for follow-up |
| M23 ultra-cap | step20 ActuationR2=0.856488049030304 | strong enough for follow-up |
| M22 alt50 h800 | D-CHE source_h800=0.06744098663330078 | smoke positive, run full 9-row |
| M23 alt50 h800 | D-CHE source_h800=0.03887653350830078 | smoke positive, run full 9-row |
| M24 warm400 | h800=-0.09970617294311523, h1600=0.22084546089172363 | h800 negative, no promotion |
| M24 warm800 | h800=-0.9483625888824463, h1600=0.1281043291091919 | h800 negative, no promotion |
| M25 alt100 | h800=0.07347774505615234, h1600=0.14383041858673096 | AdamW-coupled smoke positive, run full 9-row |
| M26 gated alt100 | h800=0.11826753616333008, h1600=0.20043909549713135 | gated smoke positive, run full 9-row |

### 4GPU Long-Run Command Pattern

实际 official rows 通过 4GPU shard 写入 `official_v19`，command journal 记录在：

- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/logs/v17_command_journal.md`

复现模式如下；每个 round 对 D-CHE 与 D-FOU 各跑 2 个 shard，并在四张 GPU 上并行 drain。每轮结束后执行 merge-only。

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19

# Example shard command. Replace carrier/variant/spec/run_label/shard/device for r11-r15.
$KAN experiments/run_v19_functional_continuation.py \
  --out-dir $OUT \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST,KMNIST,Fashion-MNIST --seeds 0,1,2 --steps 4800 \
  --shard-count 2 --shard-index 0 --run-label r13che_fc0 \
  --spec-ids M22-exact-readout-highcap-alt50-fu0p0001,M23-exact-readout-ultracap-alt50-fu0p0001

$KAN experiments/run_v19_functional_continuation.py --merge-only --out-dir $OUT
```

Round mapping:

| round | carriers | repair variants | spec ids |
|---|---|---|---|
| r11 | D-CHE, D-FOU | CHE-R2-low-degree-k3-triton / FOU-R2-low-frequency-k2-stream | M21-exact-readout-fu0p0001 |
| r12 | D-CHE, D-FOU | CHE-R2-low-degree-k3-triton / FOU-R2-low-frequency-k2-stream | M23-exact-readout-ultracap-fu0p0001 |
| r13 | D-CHE, D-FOU | CHE-R2-low-degree-k3-triton / FOU-R2-low-frequency-k2-stream | M22-exact-readout-highcap-alt50-fu0p0001,M23-exact-readout-ultracap-alt50-fu0p0001 |
| r14 | D-CHE, D-FOU | CHE-R2-low-degree-k3-triton / FOU-R2-low-frequency-k2-stream | M25-adamw-ultracap-alt100-fu0p0001 |
| r15 | D-CHE, D-FOU | CHE-R2-low-degree-k3-triton / FOU-R2-low-frequency-k2-stream | M26-gated-adamw-ultracap-alt100-fu0p0001 |

### 4GPU Runtime Evidence

| round | rows added | gpu_snapshot_rows | max_mem_mb | max_util_percent |
|---|---:|---:|---:|---:|
| r11 M21 | 18 | 284 | 759.0 | 27.0 |
| r12 M23 | 18 | 76 | 759.0 | 21.0 |
| r13 M22/M23 alt50 | 36 | 100 | 759.0 | 23.0 |
| r14 M25 AdamW | 18 | 68 | 759.0 | 17.0 |
| r15 M26 gated | 18 | 68 | 759.0 | 19.0 |

### Long-Run Results

| round | carrier | spec | h800 mean | h1600 mean | h3200 mean | h4800 mean | retained |
|---|---|---|---:|---:|---:|---:|---:|
| r11 | D-CHE | M21 | -1.7059106760554843 | -1.9571839769681294 | -2.190653827455309 | -2.323306428061591 | 0 |
| r11 | D-FOU | M21 | -2.8070288366741605 | -3.2429449359575906 | -3.6243044667773776 | -3.8307798902193704 | 0 |
| r12 | D-CHE | M23 | -0.3645904196633233 | -0.37781767050425213 | -0.367608024014367 | -0.35427424642774796 | 0 |
| r12 | D-FOU | M23 | -0.7186367909113566 | -0.7717924184269376 | -0.7617345386081271 | -0.7259222070376078 | 0 |
| r13 | D-CHE | M22 alt50 | -0.1703893542289734 | -0.06318890386157566 | -0.028217554092407227 | -0.005586928791469998 | 0 |
| r13 | D-CHE | M23 alt50 | -0.07936050494511922 | 0.0037816431787278918 | 0.03331817520989312 | 0.05557106600867377 | 0 |
| r13 | D-FOU | M22 alt50 | -0.22767482863532174 | -0.13538656632105509 | -0.0811925729115804 | -0.030357791317833796 | 0 |
| r13 | D-FOU | M23 alt50 | -0.22098304828008017 | -0.22261044051912096 | -0.19528800911373562 | -0.15801193979051378 | 0 |
| r14 | D-CHE | M25 alt100 | -0.03924524121814304 | -0.0040361351437038845 | 0.0047302643458048505 | -0.013945202032725016 | 0 |
| r14 | D-FOU | M25 alt100 | -0.10326227876875135 | -0.11500602960586548 | -0.10978075530793932 | -0.11714468399683635 | 0 |
| r15 | D-CHE | M26 gated alt100 | -0.06443260113398235 | -0.028240985340542264 | -0.016286081737942167 | -0.03339660167694092 | 0 |
| r15 | D-FOU | M26 gated alt100 | -0.07220927874247234 | -0.06470630566279094 | -0.05711548858218723 | -0.06803824504216512 | 0 |

### Actuation / Gate Aggregate

- official artifact: `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h4_actuation_solver_extended.csv`
- rows: 840
- overall ActuationR2 min/max/mean: 0.014818370342254639 / 0.9999852180480957 / 0.8282578299442928
- M26 gate accept: 41/130
- route after continuation: `R-MLPGenericRetained-KANCarrierBlocked`
- promotion_allowed: 0

### Post-Continuation Artifacts

- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_functional_continuation_summary.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h4_actuation_solver_extended.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_route_decision.json`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_next_hypothesis_queue.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_next_hypothesis_queue.md`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_continuation_code_review_packet.zip`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_results_bundle.zip`

Packaging blocker/fix:

- blocker: local `zip` command was unavailable (`/bin/bash: zip: command not found`).
- fix: regenerated `v19_continuation_code_review_packet.zip` and refreshed `v19_results_bundle.zip` with Python stdlib `zipfile`; packet manifest and sha256 manifest are present in `v19_continuation_code_review_packet/`.

### Continuation Conclusion

- 目标没有达成。
- M21-M26 后 KAN repaired retained candidates 仍为 0。
- 唯一 retained candidate 仍是 MLP/M16 generic dynamics：h800=0.01937996016608344, h1600=0.0803578429751926, h3200=0.07967897256215413, h4800=0.07385419474707709。
- D-CHE/M23-alt50 出现 h1600/h3200/h4800 late positive，但 h800 mean=-0.07936050494511922，因此不能按计划定义为 retained source。
- M25/M26 的 AdamW-coupled/gated variants 在 smoke 中有正 source，但 full 9-row grouped mean 为负，不能 promotion。

## 2026-06-03 H5/M27-M28 Generalization Gate Continuation

### Why Continued Again

- M21-M26 后目标仍未达成，且 next queue 仍有 P0：`Generalization gate for source-channel actuator` 与 `AdamW-coupling audit before promotion`。
- 继续推进而不是停在 M26：新增更强 train-only generalization gate，验证 smoke positive 是否能跨 dataset/seed 留存。

### Code Changes

- `dgkan/fu/mechanisms.py`
  - 新增 `M27-MultiBatchGeneralizationGatedReadoutFU`：SGD primary + exact-readout residual + train-only multibatch/shuffled-label gate。
  - 新增 `M28-AdamWMultiBatchGatedReadoutFU`：AdamW primary + exact-readout residual + 同一 train-only gate。
- `experiments/run_v17_common.py`
  - M27/M28 pulse commit 前临时应用 update，读取 `channel_a`、`channel_b` 和 corrupted-label train audit loss。
  - gate 条件：ActuationR2/B2/B3 通过、channel_a/channel_b gain 为正、corrupted-label gain 不超过 signal allowance。
  - trace 新增 `generalization_gain_a`、`generalization_gain_b`、`shuffled_label_gain`、`generalization_signal_gain`、`generalization_gate_accept`。
- `experiments/run_v19_functional_continuation.py`
  - 新增 M27 alt50/alt100/alt200 与 M28 alt100/alt200 specs。

### Compile Command

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v19_functional_continuation.py
```

### Smoke Commands

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir $BASE/smoke_functional_m27_dche \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label m27dchesmoke \
  --spec-ids M27-multibatch-gated-alt50-fu0p0001,M27-multibatch-gated-alt100-fu0p0001,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir $BASE/smoke_functional_m27b_dche \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label m27bdchesmoke \
  --spec-ids M27-multibatch-gated-alt100-fu0p00005,M27-multibatch-gated-alt200-fu0p0001,M27-multibatch-gated-alt200-fu0p00005,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir $BASE/smoke_functional_m28_dche \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label m28dchesmoke \
  --spec-ids M28-adamw-multibatch-gated-alt100-fu0p0001,M28-adamw-multibatch-gated-alt100-fu0p00005,M28-adamw-multibatch-gated-alt200-fu0p0001,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock
```

### Smoke Results

| scope | spec | h800 | h1600 | gate/readback |
|---|---|---:|---:|---|
| M27 smoke | alt50 fu1e-4 | -0.06644022464752197 | -0.009639501571655273 | gate worked, source negative |
| M27 smoke | alt100 fu1e-4 | -0.0613330602645874 | 0.2805297374725342 | delayed positive only |
| M27b smoke | alt100 fu5e-5 | -0.23987281322479248 | 0.1628739833831787 | delayed positive only |
| M27b smoke | alt200 fu1e-4 | -0.3123122453689575 | 0.20300543308258057 | delayed positive only |
| M27b smoke | alt200 fu5e-5 | -0.6079999208450317 | -0.047327518463134766 | negative |
| M28 smoke | alt100 fu1e-4 | -0.019653797149658203 | 0.023518681526184082 | near miss h800 |
| M28 smoke | alt100 fu5e-5 | 0.1523970365524292 | 0.19489693641662598 | positive, run full 9-row |
| M28 smoke | alt200 fu1e-4 | 0.11948752403259277 | 0.160567045211792 | positive, run full 9-row |

### r16 Full 4GPU Command

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
OUT=results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19

for IDX in 0 1 2 3; do
  $KAN experiments/run_v19_functional_continuation.py \
    --out-dir $OUT \
    --device cuda:$IDX --data-root data \
    --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
    --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --steps 4800 \
    --shard-count 4 --shard-index $IDX --run-label r16m28dche \
    --spec-ids M28-adamw-multibatch-gated-alt100-fu0p00005,M28-adamw-multibatch-gated-alt200-fu0p0001,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock &
done
wait

$KAN experiments/run_v19_functional_continuation.py --merge-only --out-dir $OUT
```

Runtime note:

- I also ran an `nvidia-smi` CSV monitor for r16.
- blocker in wrapper command: a bare `wait` waited on the monitor loop too, so the shell did not advance to merge after training finished.
- fix: touched `v19_functional_continuation_r16_gpu_monitor.stop` externally after confirming shard files were written and no Python training process remained; then merge completed.
- This affected command orchestration only, not the already-written shard results.

### r16 Full Results

| spec | rows | h800 mean | h1600 mean | h3200 mean | h4800 mean | retained |
|---|---:|---:|---:|---:|---:|---:|
| M28 alt100 fu5e-5 | 9 | -0.06881865527894762 | -0.06802056895362006 | -0.08006360133488973 | -0.12355671326319377 | 0 |
| M28 alt200 fu1e-4 | 9 | -0.04596583048502604 | -0.03709010283152262 | -0.04783064126968384 | -0.07944080564710829 | 0 |

Per-row evidence showed individual positives, but grouped mean failed:

- r16 M28 mechanism rows: 18
- M28 diagnostic trace rows: 117
- M28 ActuationR2 min/max/mean: 0.8826937079429626 / 0.9999926686286926 / 0.9973855691078382
- M28 generalization gate accepts: 31/117
- r16 GPU snapshot rows: 2024
- r16 max memory used MB: 759.0
- r16 max GPU util percent: 18.0

### H5 Artifacts

- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h5_generalization_gate_m27_m28.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h4_actuation_solver_extended.csv` updated to 927 diagnostic rows.
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_route_decision.json` updated to matrix rows 819 and grouped probes 61.
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_next_hypothesis_queue.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_next_hypothesis_queue.md`

### H5 Conclusion

- 目标仍未达成。
- M28 证明 AdamW + multibatch/shuffled-label gate 可以产生 single-seed smoke positive，但 full grouped D-CHE source 仍为负。
- KAN repaired M20-M28 retained candidates: 0。
- 当前 blocker 已进一步缩小为 dataset/seed heterogeneity 与 source writer 设计问题；继续调 cap/alt-period/readout gate 已经不像有效方向。

## 2026-06-03 H6 Dataset/Seed Anatomy + FU-H2/H3 Smoke

### Why Continued

- v19 目标仍未达成。
- H5 后明确 next queue：先解释 dataset/seed heterogeneity，再验证计划 FU-H2/FU-H3 的 source-state writer / matrix-block route。
- 我没有继续调 readout cap/alt-period；改为跑计划中的 slow-state / PopRisk / matrix-block smoke。

### Code / Artifact Changes

- `experiments/run_v19_functional_continuation.py`
  - 新增 specs：
    - `M6-slowstate-fu0p0005`
    - `M6-slowstate-fu0p0001`
    - `M13-lowrank-r4-fu0p0005`
    - `M13-lowrank-r4-fu0p0001`
- 新增 official artifacts：
  - `v19_h6_dataset_seed_heterogeneity_rows.csv`
  - `v19_h6_dataset_seed_heterogeneity_summary.csv`
  - `v19_h6_source_state_writer_smoke.csv`
- artifact blocker/fix：
  - blocker: first heterogeneity CSV writer used the first row's fields, but grouped summaries used different keys (`dataset` vs `seed`), causing `ValueError: dict contains fields not in fieldnames: 'seed'`。
  - fix: rewrote writer to collect union fieldnames across all rows before writing.

### Heterogeneity Artifact Command

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
# Reads v19_functional_continuation_matrix.csv;
# extracts D-CHE M23/M25/M26/M28 row-level h800/h1600/h3200/h4800 source;
# writes v19_h6_dataset_seed_heterogeneity_rows.csv and summary.csv.
PY
```

Result:

- row-level heterogeneity rows: 45
- grouped heterogeneity summary rows: 35

### FU-H2/H3 Smoke Command

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu
OUT=$BASE/smoke_functional_h2_h3_dche

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir $OUT \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label h2h3dchesmoke \
  --spec-ids M6-slowstate-fu0p0005,M6-slowstate-fu0p0001,M11-dual-fu0p0005,M12-schedulefree-fu0p0005,M14-popriskslow-fu0p0005,M13-lowrank-r4-fu0p0005,M13-lowrank-r4-fu0p0001,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock

$KAN experiments/run_v19_functional_continuation.py --merge-only --out-dir $OUT
```

### FU-H2/H3 Smoke Result

| spec | h800 | h1600 | decision |
|---|---:|---:|---|
| M6-slowstate-fu0p0005 | -0.9580533504486084 | -0.8114761114120483 | negative smoke |
| M6-slowstate-fu0p0001 | -0.9498236179351807 | -0.8082684278488159 | negative smoke |
| M11-dual-fu0p0005 | -0.9413673877716064 | -0.8014754056930542 | negative smoke |
| M12-schedulefree-fu0p0005 | -0.9617340564727783 | -0.8188821077346802 | negative smoke |
| M14-popriskslow-fu0p0005 | -0.971245288848877 | -0.8306988477706909 | negative smoke |
| M13-lowrank-r4-fu0p0005 | -0.9408831596374512 | -0.8024903535842896 | negative smoke |
| M13-lowrank-r4-fu0p0001 | -0.9723532199859619 | -0.8309644460678101 | negative smoke |

GPU evidence:

- smoke GPU snapshot rows: 340
- max memory used MB: 759.0
- max GPU util percent: 51.0

### H6 Conclusion

- H2/H3 existing slow-state / PopRisk / matrix-block route did not pass even the D-CHE MNIST seed0 smoke.
- Therefore it was not escalated to full 9-row.
- route updated to include `H2H3SmokeNegative`; promotion remains 0.

## 2026-06-03 H7 Failure Anatomy / No-Actionable-Same-Family Decision

### Why Continued

- v19 目标仍未达成：`promotion_allowed=0`，KAN grouped retained candidates 仍为 0。
- H6 后没有直接再跑同类 full 4GPU sweep；先按计划第 13/14 节把现有 H1/H2/H3/H4 continuation evidence 做成 failure anatomy，判断是否还有可审计的同类修复方向。
- 这一步没有生成新的实验数据行，也没有伪造结果；它只读取 already measured official artifacts 并更新 route/queue/docs。

### H7 Artifact Command

```bash
python - <<'PY'
# Reads:
#   v19_functional_continuation_summary.csv
#   v19_h6_dataset_seed_heterogeneity_summary.csv
#   v19_h6_source_state_writer_smoke.csv
# Writes:
#   v19_h7_failure_anatomy_decision.csv
#   v19_h7_failure_anatomy_decision.md
# Updates:
#   v19_route_decision.json
#   v19_next_hypothesis_queue.csv
#   v19_next_hypothesis_queue.md
PY
```

Actual output:

- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h7_failure_anatomy_decision.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h7_failure_anatomy_decision.md`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_next_hypothesis_queue.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_next_hypothesis_queue.md`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_route_decision.json`

### H7 Decision Summary

| scope | rows | decision | continue same family |
|---|---:|---|---:|
| MLP retained functional source | 9 | generic MLP-only dynamics evidence | 0 |
| KAN repaired M20-M28 grouped continuation | 14 | same readout actuator family not promising | 0 |
| KAN late-positive rows | 1 | late positive, not retention | 0 |
| M28 smoke-positive full-negative | 2 | train-only gate not sufficient for grouped retention | 0 |
| Dataset/seed heterogeneity | 35 | blocks dataset/seed-agnostic promotion | 0 |
| FU-H2/H3 writer smoke | 7 | existing writer family negative smoke | 0 |
| H7 final decision | 61 | conceptual blocker, not more same-family runs | 0 |

Updated route fields:

- `FunctionalRoute=S3-MLPGenericRetainedSource;KANCarrierLatePositiveButNotRetained;M28SmokePositiveFullGroupedNegative;H2H3SmokeNegative;H7NoActionableSameFamilyRepair`
- `h7_failure_anatomy_rows=7`
- `h7_continue_same_family_recommended=0`
- `h7_new_theory_required=1`
- `h7_uncertainty=no_concrete_nonduplicative_source_state_mechanism_after_H1_H2_H3_H4_failures`
- `promotion_allowed=0`

### H7 Conclusion

- 目标没有达成。
- 当前没有可诚实继续执行的同类 repair：readout/actuation/gated family 已经 full grouped fail；现有 slow-state/PopRisk/matrix-block writer smoke fail。
- 下一步若继续，必须先提出新的 source-state theory / toy correctness / train-only observability gate。否则继续同类 4GPU full run 只是重复失败，不符合计划“停止换名字原地打转”的要求。

## 2026-06-03 H8 M29 Consensus Source-State New-Theory Smoke

### Why Continued

- v19 目标仍未达成：`promotion_allowed=0`。
- H7 明确判断同类 readout/gate/slow-state/matrix-block repair 不再有可审计的继续方向；推荐方向是提出新的 source-state theory。
- H8 因此实现并测试一个新机制 `M29-ConsensusSourceStateFU`：cross-batch sign consensus + corrupted-label projection + slow-state EMA + train-only commit gate。
- 这不是 full 4GPU official run，而是 D-CHE/MNIST seed0 smoke。若 smoke 失败，不升级 full 9-row，避免把明显负方向继续烧算力。

### Code Changes

- `dgkan/fu/mechanisms.py`
  - 新增 `M29-ConsensusSourceStateFU`。
  - 新增 semantic contract：`source_space=slow_state`，`optimizer_primary=SGD+train_consensus_gate`。
- `experiments/run_v17_common.py`
  - import `normalized_like`。
  - `eval_pack` 增加 `source_state_*` diagnostics 字段。
  - train loop 增加 M29 分支：两个 train batches 做 sign-consensus，corrupted-label gradient 做 safety projection，slow-state EMA 后只在 gate 通过时 commit，否则 fallback SGD。
- `experiments/run_v19_functional_continuation.py`
  - 新增 specs：
    - `M29-consensus-alt50-fu0p0001`
    - `M29-consensus-alt100-fu0p0001`
    - `M29-consensus-alt100-fu0p00005`

### Compile Check

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v19_functional_continuation.py
```

Result: passed.

### M29 Tiny Path Check

Purpose: verify the new mechanism can be selected and the continuation runner can produce rows before running the 1600-step smoke. Because `steps=20` is below `alt_period`, this is only a path check and is not used as scientific evidence.

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu
OUT=$BASE/smoke_functional_m29_dche_tiny

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir $OUT \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 20 \
  --shard-count 1 --shard-index 0 --run-label m29tiny \
  --spec-ids M29-consensus-alt50-fu0p0001,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock

$KAN experiments/run_v19_functional_continuation.py --merge-only --out-dir $OUT
```

Result: 4 measured rows, 12 trace rows; no h800/h1600 evidence because steps were intentionally tiny.

### M29 1600-step Smoke Command

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu
OUT=$BASE/smoke_functional_m29_dche
SNAP=$OUT/v19_m29_smoke_gpu_runtime_snapshots.csv
STOP=$OUT/v19_m29_smoke_gpu_monitor.stop

mkdir -p "$OUT"
rm -f "$STOP"
printf 'timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n' > "$SNAP"
(
  while [ ! -f "$STOP" ]; do
    ts=$(date '+%Y-%m-%d %H:%M:%S %z')
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r idx mem util; do
      idx=${idx// /}; mem=${mem// /}; util=${util// /}
      printf '%s,%s,%s,%s\n' "$ts" "$idx" "$mem" "$util" >> "$SNAP"
    done
    sleep 5
  done
) &
MON=$!

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir "$OUT" \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label m29dchesmoke \
  --spec-ids M29-consensus-alt50-fu0p0001,M29-consensus-alt100-fu0p0001,M29-consensus-alt100-fu0p00005,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock
STATUS=$?
touch "$STOP"
wait "$MON" || true
if [ "$STATUS" -ne 0 ]; then exit "$STATUS"; fi

$KAN experiments/run_v19_functional_continuation.py --merge-only --out-dir "$OUT"
```

Runner journal:

- `2026-06-03 07:00:41 +08 v19 functional continuation shard 0/1 device=cuda:0 jobs=6`
- `2026-06-03 07:01:06 +08 v19 functional continuation merge rows=6 traces=42`

### M29 Smoke Result

| spec | h800 | h1600 | retained | decision |
|---|---:|---:|---:|---|
| M29-consensus-alt100-fu0p00005 | -0.9577474594116211 | -0.8420370817184448 | 0 | negative_smoke_no_full_escalation |
| M29-consensus-alt100-fu0p0001 | -0.9486420154571533 | -0.8309754133224487 | 0 | negative_smoke_no_full_escalation |
| M29-consensus-alt50-fu0p0001 | -0.9486403465270996 | -0.8269315958023071 | 0 | negative_smoke_no_full_escalation |

Gate diagnostics:

| spec | diagnostic rows | gate accepts/trials | density mean | signal max | corrupt max |
|---|---:|---:|---:|---:|---:|
| M29-consensus-alt100-fu0p00005 | 4 | 0/4 | 0.6166009902954102 | 2.51084566116333e-06 | -6.407499313354492e-07 |
| M29-consensus-alt100-fu0p0001 | 4 | 0/4 | 0.6438626348972321 | 8.977949619293213e-06 | 1.862645149230957e-08 |
| M29-consensus-alt50-fu0p0001 | 4 | 0/4 | 0.5564001724123955 | 5.885958671569824e-06 | -1.1362135410308838e-06 |

GPU evidence:

- `v19_m29_smoke_gpu_runtime_snapshots.csv` rows: 20
- max memory used MB: 693.0
- max GPU util percent: 17.0

### Official Artifact Finalization

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
# Reads:
#   smoke_functional_m29_dche/v19_functional_continuation_summary.csv
#   smoke_functional_m29_dche/v19_functional_continuation_matrix.csv
#   smoke_functional_m29_dche/v19_functional_continuation_traces.csv
#   smoke_functional_m29_dche/v19_m29_smoke_gpu_runtime_snapshots.csv
# Writes:
#   official_v19/v19_h8_m29_consensus_source_state_smoke.csv
#   official_v19/v19_h8_m29_consensus_source_state_diagnostics.csv
#   official_v19/v19_h8_m29_gpu_summary.csv
#   official_v19/v19_h8_m29_raw_matrix.csv
#   official_v19/v19_h8_m29_raw_traces.csv
#   official_v19/v19_h8_m29_gpu_runtime_snapshots.csv
#   official_v19/v19_h8_m29_consensus_source_state_decision.md
# Updates:
#   official_v19/v19_route_decision.json
#   official_v19/v19_next_hypothesis_queue.csv
#   official_v19/v19_next_hypothesis_queue.md
PY
```

Packaging blocker/fix:

```bash
# blocker: zip command was not installed
cd results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19
zip -qr v19_results_bundle.zip .
# /bin/bash: zip: command not found

# fix: use Python zipfile to generate the same bundle artifact
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import pathlib, zipfile
base = pathlib.Path('results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19')
out = base / 'v19_results_bundle.zip'
if out.exists():
    out.unlink()
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(base.rglob('*')):
        if p == out or p.is_dir():
            continue
        z.write(p, p.relative_to(base).as_posix())
print(out)
print(out.stat().st_size)
PY
```

Result: `v19_results_bundle.zip` generated, size 24M.

Official route updates:

- `FunctionalRoute=S3-MLPGenericRetainedSource;KANCarrierLatePositiveButNotRetained;M28SmokePositiveFullGroupedNegative;H2H3SmokeNegative;H7NoActionableSameFamilyRepair;H8M29ConsensusSourceStateSmokeNegative`
- `h8_m29_smoke_rows=3`
- `h8_m29_matrix_rows=3`
- `h8_m29_gate_accepts=0`
- `h8_m29_gate_trials=12`
- `h8_m29_max_h800_source=-0.9486403465270996`
- `h8_m29_max_h1600_source=-0.8269315958023071`
- `h8_m29_full_escalation=0`
- `promotion_allowed=0`

### H8 Conclusion

- v19 目标仍未达成。
- M29 是按 H7 推荐方向实现的新 source-state theory smoke，不是同类 readout gate 换名。
- 结果为负：h800/h1600 全负，retained=0，gate accepts=0/12。
- 因为 smoke 未过，未升级 full 9-row，避免把负结果冒充成继续推进。
- 当前下一步不再是重跑现有机制，而是需要新的 toy-correctness/source-observability 理论；在没有新理论前，继续 full 4GPU sweep 不可审计。

### H8 Final Validation

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v19_functional_continuation.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path('results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_route_decision.json')
r = json.loads(p.read_text())
for k in ['route','FunctionalRoute','promotion_allowed','h8_m29_smoke_rows','h8_m29_gate_accepts','h8_m29_gate_trials','h8_m29_max_h800_source','h8_m29_max_h1600_source','h8_m29_full_escalation']:
    print(k, '=', r.get(k))
PY

/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import zipfile, pathlib
zpath = pathlib.Path('results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_results_bundle.zip')
need = [
    'v19_h8_m29_consensus_source_state_smoke.csv',
    'v19_h8_m29_consensus_source_state_diagnostics.csv',
    'v19_h8_m29_gpu_summary.csv',
    'v19_h8_m29_raw_matrix.csv',
    'v19_h8_m29_raw_traces.csv',
    'v19_h8_m29_consensus_source_state_decision.md',
    'v19_route_decision.json',
    'v19_next_hypothesis_queue.csv',
]
with zipfile.ZipFile(zpath) as z:
    names = set(z.namelist())
print('bundle_entries', len(names))
for n in need:
    print(n, 'present=' + str(n in names))
PY

wc -l \
  results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h8_m29_consensus_source_state_smoke.csv \
  results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h8_m29_consensus_source_state_diagnostics.csv \
  results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h8_m29_gpu_summary.csv

ps -eo pid,args | rg 'python .*run_v19|nvidia-smi --query-gpu' | rg -v 'rg|bash -lc' || true
```

Validation result:

- py_compile: passed.
- route: `promotion_allowed=0`, `h8_m29_smoke_rows=3`, `h8_m29_gate_accepts=0`, `h8_m29_gate_trials=12`, `h8_m29_full_escalation=0`。
- bundle entries: 1022; all H8 required bundle files present.
- H8 CSV line counts: smoke=4 including header, diagnostics=4 including header, gpu_summary=2 including header.
- residual `run_v19` / `nvidia-smi --query-gpu` process: none.

## 2026-06-03 H9 M30 Low-Threshold Consensus Gate Repair Smoke

### Why Continued

- H8 M29 failed, but its diagnostic showed a possible gate-threshold blocker: max signal gain was `8.977949619293213e-06`, slightly below the `1e-5` gate threshold, so gate accepts were 0/12.
- H9 tests a specific repair, not a promotion claim: lower the global signal threshold and faster slow-state EMA while keeping corrupted-label safety.
- If commits happen but source remains negative, then “M29 was only over-gated” is not a sufficient explanation.

### Code Changes

- `dgkan/fu/mechanisms.py`
  - Added `M30-LowThresholdConsensusSourceStateFU`.
  - Added mechanism contract row: `slow_state`, `SGD+low_threshold_consensus_gate`。
- `experiments/run_v17_common.py`
  - M29/M30 share the consensus source-state branch.
  - M30 uses `signal_threshold=1e-7` and `ema_beta=0.90`; M29 remains `1e-5` and `0.97`。
  - Added trace fields `source_state_gate_threshold` and `source_state_ema_beta`。
- `experiments/run_v19_functional_continuation.py`
  - Added:
    - `M30-lowthreshold-consensus-alt50-fu0p0001`
    - `M30-lowthreshold-consensus-alt100-fu0p0001`
    - `M30-lowthreshold-consensus-alt100-fu0p00005`

### Compile Check

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v19_functional_continuation.py
```

Result: passed.

### H9 Smoke Command

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu
OUT=$BASE/smoke_functional_m30_dche
SNAP=$OUT/v19_m30_smoke_gpu_runtime_snapshots.csv
STOP=$OUT/v19_m30_smoke_gpu_monitor.stop

mkdir -p "$OUT"
rm -f "$STOP"
printf 'timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n' > "$SNAP"
(
  while [ ! -f "$STOP" ]; do
    ts=$(date '+%Y-%m-%d %H:%M:%S %z')
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r idx mem util; do
      idx=${idx// /}; mem=${mem// /}; util=${util// /}
      printf '%s,%s,%s,%s\n' "$ts" "$idx" "$mem" "$util" >> "$SNAP"
    done
    sleep 5
  done
) &
MON=$!

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir "$OUT" \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label m30dchesmoke \
  --spec-ids M30-lowthreshold-consensus-alt50-fu0p0001,M30-lowthreshold-consensus-alt100-fu0p0001,M30-lowthreshold-consensus-alt100-fu0p00005,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock
STATUS=$?
touch "$STOP"
wait "$MON" || true
if [ "$STATUS" -ne 0 ]; then exit "$STATUS"; fi

$KAN experiments/run_v19_functional_continuation.py --merge-only --out-dir "$OUT"
```

Runner journal:

- `2026-06-03 07:12:01 +08 v19 functional continuation shard 0/1 device=cuda:0 jobs=6`
- `2026-06-03 07:12:26 +08 v19 functional continuation merge rows=6 traces=42`

### H9 Smoke Result

| spec | h800 | h1600 | retained | decision |
|---|---:|---:|---:|---|
| M30-lowthreshold-consensus-alt100-fu0p00005 | -0.95654296875 | -0.8404527902603149 | 0 | negative_smoke_no_full_escalation |
| M30-lowthreshold-consensus-alt100-fu0p0001 | -0.9484755992889404 | -0.8305522203445435 | 0 | negative_smoke_no_full_escalation |
| M30-lowthreshold-consensus-alt50-fu0p0001 | -0.9478328227996826 | -0.8264065980911255 | 0 | negative_smoke_no_full_escalation |

Gate diagnostics:

| spec | diagnostic rows | gate accepts/trials | density mean | signal max | corrupt max |
|---|---:|---:|---:|---:|---:|
| M30-lowthreshold-consensus-alt100-fu0p00005 | 4 | 4/4 | 0.6165540516376495 | 2.771615982055664e-06 | -7.897615432739258e-07 |
| M30-lowthreshold-consensus-alt100-fu0p0001 | 4 | 4/4 | 0.6396396607160568 | 6.634742021560669e-06 | 1.1175870895385742e-08 |
| M30-lowthreshold-consensus-alt50-fu0p0001 | 4 | 4/4 | 0.5566816851496696 | 1.2181699275970459e-05 | -2.3618340492248535e-06 |

GPU evidence:

- `v19_m30_smoke_gpu_runtime_snapshots.csv` rows: 20
- max memory used MB: 693.0
- max GPU util percent: 17.0

### Official Artifact Finalization

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
# Reads smoke_functional_m30_dche summary/matrix/traces/GPU snapshots.
# Writes:
#   official_v19/v19_h9_m30_lowthreshold_consensus_smoke.csv
#   official_v19/v19_h9_m30_lowthreshold_consensus_diagnostics.csv
#   official_v19/v19_h9_m30_gpu_summary.csv
#   official_v19/v19_h9_m30_raw_matrix.csv
#   official_v19/v19_h9_m30_raw_traces.csv
#   official_v19/v19_h9_m30_gpu_runtime_snapshots.csv
#   official_v19/v19_h9_m30_lowthreshold_consensus_decision.md
# Updates:
#   official_v19/v19_route_decision.json
#   official_v19/v19_next_hypothesis_queue.csv
#   official_v19/v19_next_hypothesis_queue.md
PY
```

Official route updates:

- `FunctionalRoute=S3-MLPGenericRetainedSource;KANCarrierLatePositiveButNotRetained;M28SmokePositiveFullGroupedNegative;H2H3SmokeNegative;H7NoActionableSameFamilyRepair;H8M29ConsensusSourceStateSmokeNegative;H9M30LowThresholdConsensusSmokeNegative`
- `h9_m30_smoke_rows=3`
- `h9_m30_matrix_rows=3`
- `h9_m30_gate_accepts=12`
- `h9_m30_gate_trials=12`
- `h9_m30_max_h800_source=-0.9478328227996826`
- `h9_m30_max_h1600_source=-0.8264065980911255`
- `h9_m30_full_escalation=0`
- `promotion_allowed=0`

### H9 Conclusion

- H9 repaired H8's zero-commit issue: M30 gate accepted 12/12 diagnostic opportunities.
- Source still failed: all h800/h1600 rows were negative and retained=0.
- Therefore M29/M30 consensus gate tuning is not an actionable path to promotion.
- No full 9-row escalation was run because the smoke did not pass.

### H9 Final Validation

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v19_functional_continuation.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path('results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_route_decision.json')
r = json.loads(p.read_text())
for k in ['route','FunctionalRoute','promotion_allowed','h9_m30_smoke_rows','h9_m30_gate_accepts','h9_m30_gate_trials','h9_m30_max_h800_source','h9_m30_max_h1600_source','h9_m30_full_escalation']:
    print(k, '=', r.get(k))
PY

/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import zipfile, pathlib
zpath = pathlib.Path('results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_results_bundle.zip')
need = [
    'v19_h9_m30_lowthreshold_consensus_smoke.csv',
    'v19_h9_m30_lowthreshold_consensus_diagnostics.csv',
    'v19_h9_m30_gpu_summary.csv',
    'v19_h9_m30_raw_matrix.csv',
    'v19_h9_m30_raw_traces.csv',
    'v19_h9_m30_lowthreshold_consensus_decision.md',
    'v19_route_decision.json',
    'v19_next_hypothesis_queue.csv',
]
with zipfile.ZipFile(zpath) as z:
    names = set(z.namelist())
print('bundle_entries', len(names))
for n in need:
    print(n, 'present=' + str(n in names))
PY

wc -l \
  results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h9_m30_lowthreshold_consensus_smoke.csv \
  results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h9_m30_lowthreshold_consensus_diagnostics.csv \
  results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h9_m30_gpu_summary.csv

ps -eo pid,args | rg 'python .*run_v19|nvidia-smi --query-gpu' | rg -v 'rg|bash -lc' || true
```

Validation result:

- py_compile: passed.
- route: `promotion_allowed=0`, `h9_m30_smoke_rows=3`, `h9_m30_gate_accepts=12`, `h9_m30_gate_trials=12`, `h9_m30_full_escalation=0`。
- bundle entries: 1029; all H9 required bundle files present.
- H9 CSV line counts: smoke=4 including header, diagnostics=4 including header, gpu_summary=2 including header.
- residual `run_v19` / `nvidia-smi --query-gpu` process: none.

## H10 2026-06-03：M31 AdamW Low-Threshold Consensus Residual

触发原因：H9 M30 修复了 zero-commit，但 D-CHE/MNIST seed0 smoke 仍全负。剩余可检验解释是 M29/M30 离开 AdamW primary training 后破坏 carrier dynamics，因此 H10 新增 AdamW primary + consensus residual 的 M31。

### 代码修改

- `dgkan/fu/mechanisms.py`
  - Added `M31-AdamWLowThresholdConsensusResidualFU`.
  - Added mechanism contract row: `slow_state`, `AdamW+low_threshold_consensus_residual`。
- `experiments/run_v17_common.py`
  - M29/M30/M31 share consensus readback/diagnostic branch.
  - M31 uses AdamW primary optimizer, then applies low-threshold consensus FU residual when gate accepts.
  - M31 keeps `signal_threshold=1e-7` and `ema_beta=0.90`。
- `experiments/run_v19_functional_continuation.py`
  - Added:
    - `M31-adamw-consensus-alt50-fu0p0001`
    - `M31-adamw-consensus-alt100-fu0p0001`
    - `M31-adamw-consensus-alt100-fu0p00005`

### Compile Check

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v19_functional_continuation.py
```

Result: passed.

### H10 Smoke Command

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu
OUT=$BASE/smoke_functional_m31_dche
SNAP=$OUT/v19_m31_smoke_gpu_runtime_snapshots.csv
STOP=$OUT/v19_m31_smoke_gpu_monitor.stop

mkdir -p "$OUT"
rm -f "$STOP"
printf 'timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n' > "$SNAP"
(
  while [ ! -f "$STOP" ]; do
    ts=$(date '+%Y-%m-%d %H:%M:%S %z')
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r idx mem util; do
      idx=${idx// /}; mem=${mem// /}; util=${util// /}
      printf '%s,%s,%s,%s\n' "$ts" "$idx" "$mem" "$util" >> "$SNAP"
    done
    sleep 5
  done
) &
MON=$!

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir "$OUT" \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label m31dchesmoke \
  --spec-ids M31-adamw-consensus-alt50-fu0p0001,M31-adamw-consensus-alt100-fu0p0001,M31-adamw-consensus-alt100-fu0p00005,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock
STATUS=$?
touch "$STOP"
wait "$MON" || true
if [ "$STATUS" -ne 0 ]; then exit "$STATUS"; fi

$KAN experiments/run_v19_functional_continuation.py --merge-only --out-dir "$OUT"
```

Smoke runner journal:

- `2026-06-03 07:18:53 +08 v19 functional continuation shard 0/1 device=cuda:0 jobs=6`
- `2026-06-03 07:19:16 +08 v19 functional continuation merge rows=6 traces=42`

Smoke result:

| spec | h800 | h1600 | decision |
|---|---:|---:|---|
| M31-adamw-consensus-alt100-fu0p00005 | 0.06744861602783203 | 0.0416184663772583 | upgrade_to_full_9row |
| M31-adamw-consensus-alt100-fu0p0001 | 0.09428870677947998 | 0.10595178604125977 | upgrade_to_full_9row |
| M31-adamw-consensus-alt50-fu0p0001 | 0.06738388538360596 | 0.10177969932556152 | upgrade_to_full_9row |

### H10 Full 4GPU Command

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu
OUT=$BASE/full_functional_m31_dche
SNAP=$OUT/v19_m31_full_gpu_runtime_snapshots.csv
STOP=$OUT/v19_m31_full_gpu_monitor.stop

mkdir -p "$OUT"
rm -f "$STOP"
printf 'timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n' > "$SNAP"
(
  while [ ! -f "$STOP" ]; do
    ts=$(date '+%Y-%m-%d %H:%M:%S %z')
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r idx mem util; do
      idx=${idx// /}; mem=${mem// /}; util=${util// /}
      printf '%s,%s,%s,%s\n' "$ts" "$idx" "$mem" "$util" >> "$SNAP"
    done
    sleep 10
  done
) &
MON=$!

PIDS=()
for SHARD in 0 1 2 3; do
  CUDA_DEVICE="cuda:$SHARD"
  $KAN experiments/run_v19_functional_continuation.py \
    --out-dir "$OUT" \
    --device "$CUDA_DEVICE" --data-root data \
    --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
    --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --steps 4800 \
    --shard-count 4 --shard-index "$SHARD" --run-label m31full \
    --spec-ids M31-adamw-consensus-alt50-fu0p0001,M31-adamw-consensus-alt100-fu0p0001,M31-adamw-consensus-alt100-fu0p00005,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock > "$OUT/shard_${SHARD}.stdout.log" 2> "$OUT/shard_${SHARD}.stderr.log" &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    STATUS=1
  fi
done
touch "$STOP"
wait "$MON" || true
if [ "$STATUS" -ne 0 ]; then
  echo "one or more shards failed" >&2
  for SHARD in 0 1 2 3; do
    echo "--- shard $SHARD stderr ---" >&2
    tail -n 80 "$OUT/shard_${SHARD}.stderr.log" >&2 || true
  done
  exit "$STATUS"
fi

$KAN experiments/run_v19_functional_continuation.py --merge-only --out-dir "$OUT"
```

Full runner journal:

- `2026-06-03 07:20:16 +08 v19 functional continuation shard 0/4 device=cuda:0 jobs=14`
- `2026-06-03 07:20:16 +08 v19 functional continuation shard 1/4 device=cuda:1 jobs=14`
- `2026-06-03 07:20:16 +08 v19 functional continuation shard 2/4 device=cuda:2 jobs=13`
- `2026-06-03 07:20:16 +08 v19 functional continuation shard 3/4 device=cuda:3 jobs=13`
- `2026-06-03 07:23:26 +08 v19 functional continuation merge rows=54 traces=540`

Full result:

| spec | rows | h800 mean | h1600 mean | h3200 mean | h4800 mean | retained |
|---|---:|---:|---:|---:|---:|---:|
| M31-adamw-consensus-alt100-fu0p00005 | 9 | 0.016972078217400446 | 0.022935476568010118 | 0.02758270502090454 | -0.007561306158701579 | 1 |
| M31-adamw-consensus-alt100-fu0p0001 | 9 | 0.016534394688076444 | 0.02917080455356174 | 0.02548849582672119 | -0.016299128532409668 | 1 |
| M31-adamw-consensus-alt50-fu0p0001 | 9 | -0.018990496794382732 | -0.027270734310150146 | -0.042707622051239014 | -0.0903786751959059 | 0 |

GPU evidence:

- `v19_m31_full_gpu_runtime_snapshots.csv` rows: 76
- max memory used MB: 693.0
- max GPU util percent: 15.0

### Official Artifact Finalization

```bash
python - <<'PY'
# Standard-library csv/json/shutil/zipfile script.
# Reads:
#   smoke_functional_m31_dche/v19_functional_continuation_summary.csv
#   full_functional_m31_dche/v19_functional_continuation_summary.csv
#   full_functional_m31_dche/v19_functional_continuation_matrix.csv
#   full_functional_m31_dche/v19_functional_continuation_traces.csv
# Writes:
#   official_v19/v19_h10_m31_smoke_summary.csv
#   official_v19/v19_h10_m31_full_summary.csv
#   official_v19/v19_h10_m31_full_matrix.csv
#   official_v19/v19_h10_m31_full_traces.csv
#   official_v19/v19_h10_m31_full_row_examples.csv
#   official_v19/v19_h10_m31_diagnostics.csv
#   official_v19/v19_h10_m31_gpu_summary.csv
#   official_v19/v19_h10_m31_adamw_consensus_decision.md
# Updates:
#   official_v19/v19_route_decision.json
#   official_v19/v19_next_hypothesis_queue.csv
#   official_v19/v19_next_hypothesis_queue.md
# Refreshes:
#   official_v19/v19_results_bundle.zip
PY
```

Official route updates:

- `route=R-MLPGenericRetained-KANCarrierPartialRetainedH3200ButH4800Washout`
- `FunctionalRoute=S3-MLPGenericRetainedSource;KANCarrierPartialRetainedToH3200;H4800Washout;H10M31AdamWConsensusPositivePartial`
- `h10_m31_smoke_summary_rows=3`
- `h10_m31_smoke_positive_rows=3`
- `h10_m31_full_rows=54`
- `h10_m31_full_m31_rows=27`
- `h10_m31_retained_candidates=2`
- `h10_m31_best_h800_mean=0.016972078217400446`
- `h10_m31_best_h1600_mean=0.02917080455356174`
- `h10_m31_best_h3200_mean=0.02758270502090454`
- `h10_m31_best_h4800_mean=-0.007561306158701579`
- `h10_m31_gate_accepts=55`
- `h10_m31_gate_trials=189`
- `h10_m31_h4800_washout=1`
- `promotion_allowed=0`

### H10 Conclusion

- H10 produced the first D-CHE repaired KAN carrier grouped retained source through h3200: 2/3 M31 specs retained.
- It is not promotion: h4800 grouped mean is negative for both retained specs, S4/S5 were not run, and pass count is not 9/9.
- Next queue now starts with H11 independent confirmation and h4800 washout analysis.

## H11 2026-06-03：M31 Independent Confirmation

触发原因：H10 只证明 offset0 下 M31 有 partial positive，不能排除初始化/采样轨迹偶然性。H11 增加 `--init-seed-offset`，用 3 个 independent offsets 验证 H10 的 h3200 retained source 是否复现。

### 代码修改

- `experiments/run_v19_functional_continuation.py`
  - Added `--init-seed-offset`.
  - Added `init_seed_offset` field into job rows.
  - `carrier_model()` seed and `train_one()` seed now add `init_seed_offset`; default is 0, so historical runs are unchanged.

### Compile Check

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  experiments/run_v19_functional_continuation.py \
  experiments/run_v17_common.py \
  dgkan/fu/mechanisms.py
```

Result: passed.

### H11 4GPU Command

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu
OUT=$BASE/h11_m31_independent_confirm_dche
SNAP=$OUT/v19_h11_m31_gpu_runtime_snapshots.csv
STOP=$OUT/v19_h11_m31_gpu_monitor.stop

mkdir -p "$OUT"
rm -f "$STOP"
printf 'timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n' > "$SNAP"
(
  while [ ! -f "$STOP" ]; do
    ts=$(date '+%Y-%m-%d %H:%M:%S %z')
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r idx mem util; do
      idx=${idx// /}; mem=${mem// /}; util=${util// /}
      printf '%s,%s,%s,%s\n' "$ts" "$idx" "$mem" "$util" >> "$SNAP"
    done
    sleep 10
  done
) &
MON=$!

STATUS=0
for REP in 1 2 3; do
  OFFSET=$((REP * 100000))
  PIDS=()
  for SHARD in 0 1 2 3; do
    CUDA_DEVICE="cuda:$SHARD"
    $KAN experiments/run_v19_functional_continuation.py \
      --out-dir "$OUT" \
      --device "$CUDA_DEVICE" --data-root data \
      --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
      --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --steps 4800 \
      --init-seed-offset "$OFFSET" \
      --shard-count 4 --shard-index "$SHARD" --run-label "h11r${REP}" \
      --spec-ids M31-adamw-consensus-alt100-fu0p0001,M31-adamw-consensus-alt100-fu0p00005,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock > "$OUT/shard_${REP}_${SHARD}.stdout.log" 2> "$OUT/shard_${REP}_${SHARD}.stderr.log" &
    PIDS+=("$!")
  done
  for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
      STATUS=1
    fi
  done
done
touch "$STOP"
wait "$MON" || true
if [ "$STATUS" -ne 0 ]; then
  echo "one or more H11 shards failed" >&2
  for REP in 1 2 3; do
    for SHARD in 0 1 2 3; do
      echo "--- rep $REP shard $SHARD stderr ---" >&2
      tail -n 80 "$OUT/shard_${REP}_${SHARD}.stderr.log" >&2 || true
    done
  done
  exit "$STATUS"
fi
$KAN experiments/run_v19_functional_continuation.py --merge-only --out-dir "$OUT"
```

Runtime evidence:

- rep1/rep2/rep3 all completed; all shard stderr logs were 0 bytes.
- `v19_functional_continuation_matrix.csv` rows: 135.
- `v19_functional_continuation_traces.csv` rows: 1350.
- GPU snapshots: 180 rows.
- max memory used MB: 693.0.
- max GPU util percent: 20.0.

### H11 Result

| spec | rows | h800 mean | h1600 mean | h3200 mean | h4800 mean | retained |
|---|---:|---:|---:|---:|---:|---:|
| M31-adamw-consensus-alt100-fu0p00005 | 27 | -0.06243820322884454 | -0.10373381994388721 | -0.1303710142771403 | -0.16738004154629177 | 0 |
| M31-adamw-consensus-alt100-fu0p0001 | 27 | -0.06181360836382265 | -0.09745090979116934 | -0.12961135970221627 | -0.16300636309164543 | 0 |

Per-rep/spec retained-to-h3200 count: 0/6.

### Official Artifact Finalization

```bash
python - <<'PY'
# Standard-library csv/json/shutil/zipfile script.
# Copies H11 matrix/traces/summary/GPU snapshots into official_v19.
# Writes:
#   v19_h11_m31_independent_summary.csv
#   v19_h11_m31_independent_matrix.csv
#   v19_h11_m31_independent_traces.csv
#   v19_h11_m31_independent_per_rep_summary.csv
#   v19_h11_m31_independent_diagnostics.csv
#   v19_h11_m31_gpu_summary.csv
#   v19_h11_m31_independent_confirmation_decision.md
# Updates:
#   v19_route_decision.json
#   v19_next_hypothesis_queue.csv/md
# Refreshes:
#   v19_results_bundle.zip
PY
```

Official route updates:

- `route=R-MLPGenericRetained-KANCarrierH10PositiveUnconfirmedByH11`
- `FunctionalRoute=S3-MLPGenericRetainedSource;H10M31PartialPositive;H11IndependentConfirmationFailed;KANCarrierNotRobust`
- `h11_m31_independent_matrix_rows=135`
- `h11_m31_independent_m31_rows=54`
- `h11_reproduced_retained_to_h3200=0`
- `h11_best_h800_mean=-0.06181360836382265`
- `h11_best_h1600_mean=-0.09745090979116934`
- `h11_best_h3200_mean=-0.12961135970221627`
- `h11_best_h4800_mean=-0.16300636309164543`
- `h11_m31_gate_accepts=109`
- `h11_m31_gate_trials=378`
- `promotion_allowed=0`

### H11 Conclusion

H11 did not confirm H10. The H10 partial positive is now classified as fragile/unconfirmed. Next actionable fix is H12: recompute consensus residual after AdamW step instead of computing it before AdamW and applying it after AdamW.

## H12 2026-06-03：M32 Post-AdamW Consensus Residual Smoke

触发原因：H11 failed to reproduce H10. Remaining concrete implementation hypothesis: M31 computes consensus residual before AdamW and applies it after AdamW, so the residual may be stale. H12 recomputes consensus residual after AdamW.

### 代码修改

- `dgkan/fu/mechanisms.py`
  - Added `M32-PostAdamWConsensusResidualFU`.
  - Added mechanism contract row: `slow_state`, `AdamW+post_step_consensus_residual`。
- `experiments/run_v17_common.py`
  - Added M32 to consensus source-state branch.
  - M32 commit order: `opt_adam.step()` first, then recompute channel consensus/corrupt projection/gate on post-AdamW parameters, then apply residual only if gate accepts.
- `experiments/run_v19_functional_continuation.py`
  - Added:
    - `M32-postadamw-consensus-alt50-fu0p0001`
    - `M32-postadamw-consensus-alt100-fu0p0001`
    - `M32-postadamw-consensus-alt100-fu0p00005`

### Compile Check

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v19_functional_continuation.py
```

Result: passed.

### H12 Smoke Command

```bash
KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu
OUT=$BASE/h12_m32_postadamw_smoke_dche
SNAP=$OUT/v19_h12_m32_smoke_gpu_runtime_snapshots.csv
STOP=$OUT/v19_h12_m32_smoke_gpu_monitor.stop

mkdir -p "$OUT"
rm -f "$STOP"
printf 'timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n' > "$SNAP"
(
  while [ ! -f "$STOP" ]; do
    ts=$(date '+%Y-%m-%d %H:%M:%S %z')
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r idx mem util; do
      idx=${idx// /}; mem=${mem// /}; util=${util// /}
      printf '%s,%s,%s,%s\n' "$ts" "$idx" "$mem" "$util" >> "$SNAP"
    done
    sleep 5
  done
) &
MON=$!

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir "$OUT" \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 1600 \
  --init-seed-offset 100000 \
  --shard-count 1 --shard-index 0 --run-label h12m32smoke \
  --spec-ids M32-postadamw-consensus-alt50-fu0p0001,M32-postadamw-consensus-alt100-fu0p0001,M32-postadamw-consensus-alt100-fu0p00005,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock
STATUS=$?
touch "$STOP"
wait "$MON" || true
if [ "$STATUS" -ne 0 ]; then exit "$STATUS"; fi

$KAN experiments/run_v19_functional_continuation.py --merge-only --out-dir "$OUT"
```

### H12 Smoke Result

| spec | h800 | h1600 | retained |
|---|---:|---:|---:|
| M32-postadamw-consensus-alt100-fu0p00005 | -0.12051677703857422 | -0.15167248249053955 | 0 |
| M32-postadamw-consensus-alt100-fu0p0001 | -0.10039818286895752 | -0.151373028755188 | 0 |
| M32-postadamw-consensus-alt50-fu0p0001 | -0.08437776565551758 | -0.12100136280059814 | 0 |

Gate diagnostics:

- trace rows: 42
- diagnostic rows: 12
- gate accepts/trials: 6/12
- max memory used MB: 693.0
- max GPU util percent: 16.0

### Official Artifact Finalization

```bash
python - <<'PY'
# Standard-library csv/json/shutil/zipfile script.
# Copies H12 matrix/traces/summary/GPU snapshots into official_v19.
# Writes:
#   v19_h12_m32_postadamw_smoke_summary.csv
#   v19_h12_m32_postadamw_smoke_matrix.csv
#   v19_h12_m32_postadamw_smoke_traces.csv
#   v19_h12_m32_postadamw_smoke_diagnostics.csv
#   v19_h12_m32_gpu_summary.csv
#   v19_h12_m32_postadamw_smoke_decision.md
# Updates:
#   v19_route_decision.json
#   v19_next_hypothesis_queue.csv/md
# Refreshes:
#   v19_results_bundle.zip
PY
```

Official route updates:

- `route=R-MLPGenericRetained-KANCarrierSourceStateFamilyNotRobust`
- `FunctionalRoute=S3-MLPGenericRetainedSource;H10M31PartialPositive;H11IndependentConfirmationFailed;H12M32PostAdamWSmokeNegative;NoActionableSourceStateFamilyRepair`
- `h12_m32_smoke_matrix_rows=6`
- `h12_m32_smoke_positive_rows=0`
- `h12_m32_best_h800_mean=-0.08437776565551758`
- `h12_m32_best_h1600_mean=-0.12100136280059814`
- `h12_m32_gate_accepts=6`
- `h12_m32_gate_trials=12`
- `h12_m32_full_escalation=0`
- `continue_same_family_recommended=0`
- `new_source_theory_required=1`
- `promotion_allowed=0`

### H12 Conclusion

H12 negative smoke closes the M31/M32 same-family repair path for now. Continuing by changing only threshold/alt_period/lr would be unsupported name churn. New source-observability toy correctness is required before another 4GPU mechanism sweep.

## H13 2026-06-03：H10/H11 Offline Anatomy

### H13 Trigger

H12 已经否定 post-AdamW stale residual 这个实现修复，但 H10 仍留下两个 h3200 positive grouped specs。为了判断这是控制组/字段计算问题，还是 independent init 下真实不复现，H13 不重跑训练，只读取 H10/H11 已有 matrix/trace 做离线解剖。

### H13 Input Files

- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h10_m31_full_matrix.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h10_m31_full_traces.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h11_m31_independent_matrix.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h11_m31_independent_traces.csv`
- `results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_route_decision.json`

### Validation Before H13

```bash
pwd && git status --short && \
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v19_functional_continuation.py

python - <<'PY'
import json, pathlib
p=pathlib.Path('results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_route_decision.json')
d=json.loads(p.read_text())
for k in ['route','FunctionalRoute','promotion_allowed','official_success_reached','h12_m32_smoke_positive_rows','new_source_theory_required','conceptual_uncertainty']:
    print(f'{k}={d.get(k)}')
PY

wc -l \
  results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h10_m31_full_summary.csv \
  results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h11_m31_independent_summary.csv \
  results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h12_m32_postadamw_smoke_summary.csv \
  docs/DG-KAN_v19.0_SourceChannelFU_BasisKernelBreakthrough_4GPU_执行日志.md \
  docs/DG-KAN_v19.0_SourceChannelFU_BasisKernelBreakthrough_4GPU_实验结果复盘.md

ps -eo pid,cmd | rg 'run_v19|nvidia-smi|gpu_runtime' || true
```

Validation result:

- py_compile passed.
- current route before H13: `R-MLPGenericRetained-KANCarrierSourceStateFamilyNotRobust`
- `promotion_allowed=0`
- `official_success_reached=0`
- `h12_m32_smoke_positive_rows=0`
- no persistent `run_v19`/monitor process after the check; the `ps` output only saw the validation command itself.

### H13 Offline Analysis Command

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  experiments/run_v19_h13_h10_h11_anatomy.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python \
  experiments/run_v19_h13_h10_h11_anatomy.py \
  --official-dir results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19
```

Script behavior:

- reads H10/H11 matrix and trace files;
- groups M31 rows by phase/spec;
- computes mean `source_h800/h1600/h3200/h4800`, positive counts, M31 own val loss, best control val loss per run-label/dataset/seed, computed source, and source-state gate diagnostics;
- writes H13 anatomy CSV/MD artifacts;
- updates `v19_route_decision.json` and `v19_required_artifact_manifest.csv`;
- refreshes `v19_results_bundle.zip`.

### H13 Manifest Blocker and Fix

First H13 run failed while appending `v19_required_artifact_manifest.csv`:

```text
ValueError: dict contains fields not in fieldnames: 'note'
```

Cause: legacy manifest fieldnames did not include `note`, while H13 artifact append rows included that field.

Fix applied inside the H13 generation command:

- read existing manifest fieldnames;
- add required fields by union: `artifact,path,exists,size_bytes,note`;
- preserve existing rows and set missing columns to blank;
- append/update H13 artifact rows.

This changed only manifest writing compatibility. It did not alter H10/H11 experimental rows or computed metrics.

### H13 Outputs

```text
wrote results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h13_h10_h11_anatomy_summary.csv
wrote results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h13_h10_h11_controls_vs_m31.csv
wrote results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h13_h10_h11_gate_diagnostics.csv
wrote results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h13_h10_h11_phase_rollup.csv
wrote results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h13_h10_h11_anatomy_decision.md
updated results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_route_decision.json
bundle_size 24768211
h10_pos_h3200 2 h11_pos_h3200 0 h10_h4800 0 h11_h4800 0
```

After saving the reproducible H13 script and rerunning it, the final bundle size is:

```text
bundle_size 24768261
```

### H13 Key Result

| phase | spec | h800 | h1600 | h3200 | h4800 | retained_to_h3200 |
|---|---|---:|---:|---:|---:|---:|
| H10-offset0 | M31 alt100 fu5e-5 | 0.016972078217400446 | 0.022935476568010118 | 0.02758270502090454 | -0.007561306158701579 | 1 |
| H10-offset0 | M31 alt100 fu1e-4 | 0.016534394688076444 | 0.02917080455356174 | 0.02548849582672119 | -0.016299128532409668 | 1 |
| H11-independent | M31 alt100 fu5e-5 | -0.06243820322884454 | -0.10373381994388721 | -0.1303710142771403 | -0.16738004154629177 | 0 |
| H11-independent | M31 alt100 fu1e-4 | -0.06181360836382265 | -0.09745090979116934 | -0.12961135970221627 | -0.16300636309164543 | 0 |

H10/H11 delta:

| spec | h11-h10 h800 | h11-h10 h1600 | h11-h10 h3200 | h11-h10 h4800 |
|---|---:|---:|---:|---:|
| M31 alt100 fu5e-5 | -0.07941028144624498 | -0.12666929651189732 | -0.15795371929804483 | -0.1598187353875902 |
| M31 alt100 fu1e-4 | -0.0783480030518991 | -0.12662171434473107 | -0.15509985552893746 | -0.14670723455923576 |

Gate diagnostics:

- H10 M31 alt100 fu5e-5: 17/63 accepts, accept_rate=0.2698412698412698
- H10 M31 alt100 fu1e-4: 19/63 accepts, accept_rate=0.30158730158730157
- H11 independent M31 rows have comparable accept_rate around 0.2857142857142857 in each init offset.

Interpretation: H11 failed despite a nonzero/comparable gate accept rate, so the immediate blocker is not simply “gate too strict”. Accepted M31 residuals do not produce robust KAN-carrier source under independent init offsets.

### H13 Official Route Updates

- `FunctionalRoute=S3-MLPGenericRetainedSource;H10M31PartialPositive;H11IndependentConfirmationFailed;H12M32PostAdamWSmokeNegative;H13H10H11AnatomyNoRepro;NoActionableSourceStateFamilyRepair`
- `h13_anatomy_completed=1`
- `h13_h10_h3200_positive_spec_count=2`
- `h13_h11_h3200_positive_spec_count=0`
- `h13_h10_h4800_positive_spec_count=0`
- `h13_h11_h4800_positive_spec_count=0`
- `h13_actionable_same_family_repair=0`
- `continue_same_family_recommended=0`
- `new_source_theory_required=1`
- `conceptual_uncertainty=1`
- `promotion_allowed=0`

### H13 Conclusion

H13 confirms the H10 M31 partial positive is fragile and non-reproduced. Same-family M31/M32 parameter sweeps are no longer recommended. The next valid step is not another 4GPU sweep with renamed thresholds, but a new source-observability/toy-correctness theory implemented as a new mechanism family and tested first on a small correctness probe.

## Final Validation 2026-06-03 10:06:49 +08

### Commands

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v19_functional_continuation.py \
  experiments/run_v19_h13_h10_h11_anatomy.py

python - <<'PY'
import json
from pathlib import Path
p=Path('results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_route_decision.json')
d=json.loads(p.read_text())
for k in ['route','route_detail','FunctionalRoute','h13_anatomy_completed','h13_h10_h3200_positive_spec_count','h13_h11_h3200_positive_spec_count','h13_h10_h4800_positive_spec_count','h13_h11_h4800_positive_spec_count','h13_actionable_same_family_repair','promotion_allowed','official_success_reached','new_source_theory_required']:
    print(f'{k}={d.get(k)}')
PY

wc -l \
  results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h13_h10_h11_anatomy_summary.csv \
  results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h13_h10_h11_controls_vs_m31.csv \
  results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h13_h10_h11_gate_diagnostics.csv \
  results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h13_h10_h11_phase_rollup.csv \
  docs/DG-KAN_v19.0_SourceChannelFU_BasisKernelBreakthrough_4GPU_执行日志.md \
  docs/DG-KAN_v19.0_SourceChannelFU_BasisKernelBreakthrough_4GPU_实验结果复盘.md

python - <<'PY'
import zipfile
from pathlib import Path
p=Path('results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_results_bundle.zip')
need=['v19_h13_h10_h11_anatomy_summary.csv','v19_h13_h10_h11_controls_vs_m31.csv','v19_h13_h10_h11_gate_diagnostics.csv','v19_h13_h10_h11_phase_rollup.csv','v19_h13_h10_h11_anatomy_decision.md','v19_route_decision.json']
with zipfile.ZipFile(p) as z:
    names=set(z.namelist())
    print('bundle_size', p.stat().st_size)
    for n in need:
        print(n, 'present' if n in names else 'MISSING')
PY

ps -eo pid,cmd | rg 'run_v19|nvidia-smi|gpu_runtime' || true
```

### Outputs / Status

- py_compile passed for modified v19/H13 code paths.
- route: `R-MLPGenericRetained-KANCarrierSourceStateFamilyNotRobust`
- FunctionalRoute: `S3-MLPGenericRetainedSource;H10M31PartialPositive;H11IndependentConfirmationFailed;H12M32PostAdamWSmokeNegative;H13H10H11AnatomyNoRepro;NoActionableSourceStateFamilyRepair`
- `h13_anatomy_completed=1`
- `h13_h10_h3200_positive_spec_count=2`
- `h13_h11_h3200_positive_spec_count=0`
- `h13_h10_h4800_positive_spec_count=0`
- `h13_h11_h4800_positive_spec_count=0`
- `h13_actionable_same_family_repair=0`
- `promotion_allowed=0`
- `official_success_reached=0`
- `new_source_theory_required=1`
- H13 artifact line counts: summary 6, controls-vs-M31 82, gate diagnostics 10, phase rollup 4.
- `v19_results_bundle.zip` size: 24768261 bytes.
- Bundle contains all H13 official artifacts and `v19_route_decision.json`.
- No persistent v19 training or GPU monitor process remained after validation; `ps` only matched the validation command itself.
