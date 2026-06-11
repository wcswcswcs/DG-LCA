# DG-KAN v22.01 EarlySourceRetention TerminalCollapse KernelOfficialization DRAT/DRBF 4GPU 执行日志

生成时间：2026-06-04 19:23:32 +0800

记录原则：只记录真实执行过的命令、文件、状态和 blocker；不把未执行内容写成结果。

## 2026-06-04 19:23:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_s08_truth_gate.py --check all
```

- status: completed
- note: checks=import_closure,source_chain,linec,debt,mechanisms,kernel_status

## 2026-06-04 19:24:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_efficiency_officialization.py --source-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22
```

- status: completed
- note: rechecked_rows=24; source=results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22

## 2026-06-04 19:24:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_terminal_collapse_autopsy.py --source-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22
```

- status: completed
- note: reaggregated_groups=82 terminal_groups=12 decision=TerminalCollapseAutopsyRequired

## 2026-06-04 19:25:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_mlp_source_lab.py --source-dir results/v22_0_early_source_retention_fu_dche_dfou_kernel_officialization_4gpu/official_v22
```

- status: completed
- note: mlp_rows=38 decision=MLPStillTerminalCollapse

## 2026-06-04 19:25:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_function_space_target_reset.py
```

- status: completed
- note: target_rows=18 decision=TargetObservabilityNoRetention

## 2026-06-04 19:25:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_kan_source_writer.py
```

- status: completed
- note: kan_rows=12 decision=KANWriterNoEarlyContinuousSource

## 2026-06-04 19:25:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --device cuda:3 --families D-RAT,D-RBF --batches 8,32 --train-size 128 --val-size 64 --profiler-repeats 1 --profiler-warmup 1 --shard-count 1 --shard-index 0
```

- status: started
- note: D-RAT/D-RBF active repair full-loop profile

## 2026-06-04 19:26:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --device cuda:3 --families D-RAT,D-RBF --batches 8,32 --train-size 128 --val-size 64 --profiler-repeats 1 --profiler-warmup 1 --shard-count 1 --shard-index 0
```

- status: exit=0
- note: profile subprocess finished

## 2026-06-04 19:26:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --device cuda:3 --families D-RAT,D-RBF --batches 8,32 --train-size 128 --val-size 64 --profiler-repeats 1 --profiler-warmup 1 --shard-count 1 --shard-index 0 --merge-only
```

- status: started
- note: merge D-RAT/D-RBF profile shards

## 2026-06-04 19:26:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_efficiency_officialization.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --device cuda:3 --families D-RAT,D-RBF --batches 8,32 --train-size 128 --val-size 64 --profiler-repeats 1 --profiler-warmup 1 --shard-count 1 --shard-index 0 --merge-only
```

- status: exit=0
- note: merge subprocess finished

## 2026-06-04 19:26:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_drat_drbf_active_repair.py --device cuda:3 --batches 8,32 --run-profile
```

- status: completed
- note: component_rows=2

## 2026-06-04 19:27:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_controls_and_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800-SourceTargetTheoryInsufficient packet=v22_01_code_review_packet.zip bundle=v22_01_results_bundle.zip

## 2026-06-04 19:46:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_continue_terminal_repair.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_smoke --skip-packet
```

- status: completed
- note: continuation summary rows=5 decision=ContinuationNoH4800SourceChain

## 2026-06-04 20:03:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_continue_terminal_repair.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_full
```

- status: completed
- note: continuation summary rows=63 decision=ContinuationNoH4800SourceChain

## 2026-06-04 20:04:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py
```

- status: completed
- note: F75-F77 mechanism/spec edits syntax check

## 2026-06-04 20:04:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from dgkan.fu.mechanisms import MECHANISMS, mechanism_contract_rows
wanted = ['M116-DatasetInvariantPopRiskSlowFU', 'M117-DatasetInvariantReadoutConsensusFU', 'M118-SourceConservingOptimizerOnlyFU']
rows = {r['mechanism']: r for r in mechanism_contract_rows()}
print('wanted_present', all(x in MECHANISMS and x in rows for x in wanted))
for x in wanted:
    print(rows.get(x))
PY
```

- status: completed
- note: mechanism contract check returned wanted_present True

## 2026-06-04 20:04:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_smoke --scope mlp --device cpu --datasets MNIST --seeds 0 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW --run-label v2201_cont_f75_f77_smoke --steps 120 --shard-count 1 --shard-index 0
```

- status: completed
- note: short smoke; generated shard files only

## 2026-06-04 20:04:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_smoke --scope mlp --device cpu --datasets MNIST --seeds 0 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW --run-label v2201_cont_f75_f77_smoke --steps 120 --merge-only
```

- status: completed
- note: smoke merge-only

## 2026-06-04 20:04:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_01_common.py experiments/run_v22_01_continue_terminal_repair.py
```

- status: completed
- note: post-summary-script syntax check

## 2026-06-04 20:04:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_full --scope mlp --device cuda:0 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f75_f77_full --steps 6400 --shard-count 4 --shard-index 0
```

- status: completed
- note: full continuation shard 0/4 on cuda:0

## 2026-06-04 20:04:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_full --scope mlp --device cuda:1 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f75_f77_full --steps 6400 --shard-count 4 --shard-index 1
```

- status: completed
- note: full continuation shard 1/4 on cuda:1

## 2026-06-04 20:04:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_full --scope mlp --device cuda:2 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f75_f77_full --steps 6400 --shard-count 4 --shard-index 2
```

- status: completed
- note: full continuation shard 2/4 on cuda:2

## 2026-06-04 20:04:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_full --scope mlp --device cuda:3 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f75_f77_full --steps 6400 --shard-count 4 --shard-index 3
```

- status: completed
- note: full continuation shard 3/4 on cuda:3

## 2026-06-04 20:04:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_full --scope mlp --device cuda:0 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f75_f77_full --steps 6400 --merge-only
```

- status: completed
- note: full continuation merge-only; matrix rows=63 including controls

## 2026-06-04 20:05:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_s08_truth_gate.py --check all
```

- status: completed
- note: checks=import_closure,source_chain,linec,debt,mechanisms,kernel_status

## 2026-06-04 20:06:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_s08_truth_gate.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01
```

- status: completed
- note: post-continuation S0.8 truth gate rerun exit=0

## 2026-06-04 20:07:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from experiments.run_v22_01_common import build_packet, ensure_out
out = ensure_out('results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01')
build_packet(out)
PY
```

- status: completed
- note: post-continuation packet rebuild packet_sha256=91bda5284e73c2271b5a04bd9e9802b4d9f349935c0f175aef8a020f58ce747c bundle_sha256=60f282c9bc59409f6e0fb7212a0047d5a5850257801517b053e906955c6eacd1

## 2026-06-04 20:18:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_01_continue_terminal_repair.py
```

- status: completed
- note: F78-F80 implementation compile check passed

## 2026-06-04 20:18:41 +0800

```bash
python - <<'PY' ... mechanism_contract_rows F78-F80 wiring check ... PY
```

- status: completed
- note: F78-F80 mechanisms present in manifest and semantic contracts

## 2026-06-04 20:19:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f80_smoke --scope mlp --device cpu --datasets MNIST --seeds 0 --steps 120 --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f78_f80_smoke --shard-count 1 --shard-index 0
```

- status: completed
- note: F78-F80 CPU smoke completed without runtime error; not used as promotion evidence

## 2026-06-04 20:19:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f80_smoke --scope mlp --merge-only
```

- status: completed
- note: F78-F80 CPU smoke merge-only command recorded

## 2026-06-04 20:20:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f80_full --scope mlp --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f78_f80_full --steps 6400 --shard-count 4 --device cuda:0 --shard-index 0
```

- status: started
- note: F78-F80 full source-retention shard launched

## 2026-06-04 20:20:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f80_full --scope mlp --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f78_f80_full --steps 6400 --shard-count 4 --device cuda:1 --shard-index 1
```

- status: started
- note: F78-F80 full source-retention shard launched

## 2026-06-04 20:20:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f80_full --scope mlp --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f78_f80_full --steps 6400 --shard-count 4 --device cuda:2 --shard-index 2
```

- status: started
- note: F78-F80 full source-retention shard launched

## 2026-06-04 20:20:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f80_full --scope mlp --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f78_f80_full --steps 6400 --shard-count 4 --device cuda:3 --shard-index 3
```

- status: started
- note: F78-F80 full source-retention shard launched

## 2026-06-04 20:25:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f80_full --scope mlp --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f78_f80_full --steps 6400 --shard-count 4 --device cuda:0 --shard-index 0
```

- status: completed
- note: F78-F80 full source-retention shard completed

## 2026-06-04 20:25:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f80_full --scope mlp --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f78_f80_full --steps 6400 --shard-count 4 --device cuda:1 --shard-index 1
```

- status: completed
- note: F78-F80 full source-retention shard completed

## 2026-06-04 20:25:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f80_full --scope mlp --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f78_f80_full --steps 6400 --shard-count 4 --device cuda:2 --shard-index 2
```

- status: completed
- note: F78-F80 full source-retention shard completed

## 2026-06-04 20:25:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f80_full --scope mlp --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f78_f80_full --steps 6400 --shard-count 4 --device cuda:3 --shard-index 3
```

- status: completed
- note: F78-F80 full source-retention shard completed

## 2026-06-04 20:25:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f80_full --scope mlp --merge-only
```

- status: completed
- note: F78-F80 full shard merge completed

## 2026-06-04 20:25:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_continue_terminal_repair.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f80_full --tag f78_f80 --new-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate
```

- status: completed
- note: continuation summary rows=63 decision=ContinuationNoH4800SourceChain

## 2026-06-04 20:33:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_01_continue_terminal_repair.py
```

- status: completed
- note: F81-F83 implementation compile check passed

## 2026-06-04 20:33:56 +0800

```bash
python - <<'PY' ... mechanism_contract_rows F81-F83 wiring check ... PY
```

- status: completed
- note: F81-F83 mechanisms present in manifest and semantic contracts

## 2026-06-04 20:34:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f81_f83_smoke --scope mlp --device cpu --datasets MNIST --seeds 0 --steps 120 --spec-ids MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f81_f83_smoke --shard-count 1 --shard-index 0
```

- status: completed
- note: F81-F83 CPU smoke completed without runtime error; not used as promotion evidence

## 2026-06-04 20:34:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f81_f83_smoke --scope mlp --merge-only
```

- status: completed
- note: F81-F83 CPU smoke merge completed

## 2026-06-04 20:34:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f81_f83_full --scope mlp --spec-ids MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f81_f83_full --steps 6400 --shard-count 4 --device cuda:0 --shard-index 0
```

- status: started
- note: F81-F83 full source-retention shard launched

## 2026-06-04 20:34:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f81_f83_full --scope mlp --spec-ids MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f81_f83_full --steps 6400 --shard-count 4 --device cuda:1 --shard-index 1
```

- status: started
- note: F81-F83 full source-retention shard launched

## 2026-06-04 20:34:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f81_f83_full --scope mlp --spec-ids MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f81_f83_full --steps 6400 --shard-count 4 --device cuda:2 --shard-index 2
```

- status: started
- note: F81-F83 full source-retention shard launched

## 2026-06-04 20:34:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f81_f83_full --scope mlp --spec-ids MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f81_f83_full --steps 6400 --shard-count 4 --device cuda:3 --shard-index 3
```

- status: started
- note: F81-F83 full source-retention shard launched

## 2026-06-04 20:40:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f81_f83_full --scope mlp --spec-ids MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f81_f83_full --steps 6400 --shard-count 4 --device cuda:0 --shard-index 0
```

- status: completed
- note: F81-F83 full source-retention shard completed; measured rows=16

## 2026-06-04 20:40:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f81_f83_full --scope mlp --spec-ids MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f81_f83_full --steps 6400 --shard-count 4 --device cuda:1 --shard-index 1
```

- status: completed
- note: F81-F83 full source-retention shard completed; measured rows=16

## 2026-06-04 20:40:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f81_f83_full --scope mlp --spec-ids MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f81_f83_full --steps 6400 --shard-count 4 --device cuda:2 --shard-index 2
```

- status: completed
- note: F81-F83 full source-retention shard completed; measured rows=16

## 2026-06-04 20:40:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f81_f83_full --scope mlp --spec-ids MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f81_f83_full --steps 6400 --shard-count 4 --device cuda:3 --shard-index 3
```

- status: completed
- note: F81-F83 full source-retention shard completed; measured rows=15

## 2026-06-04 20:40:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f81_f83_full --scope mlp --merge-only
```

- status: completed
- note: F81-F83 full source-retention merge completed

## 2026-06-04 20:41:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_continue_terminal_repair.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f81_f83_full --tag f81_f83 --new-ids MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout
```

- status: completed
- note: continuation summary rows=63 decision=ContinuationNoH4800SourceChain

## 2026-06-04 20:49:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_01_continue_terminal_repair.py
```

- status: completed
- note: F84-F86 implementation compile check passed

## 2026-06-04 20:49:10 +0800

```bash
python - <<'PY' ... mechanism_contract_rows F84-F86 wiring check ... PY
```

- status: completed
- note: F84-F86 mechanisms present in manifest and semantic contracts

## 2026-06-04 20:50:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_smoke --scope mlp --device cpu --datasets MNIST --seeds 0 --steps 120 --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f84_f86_smoke --shard-count 1 --shard-index 0
```

- status: completed
- note: F84-F86 CPU smoke completed without runtime error; not used as promotion evidence

## 2026-06-04 20:50:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_smoke --scope mlp --merge-only
```

- status: completed
- note: F84-F86 CPU smoke merge completed; rows=7 statuses=measured

## 2026-06-04 20:51:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_full --scope mlp --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f84_f86_full --steps 6400 --shard-count 4 --device cuda:0 --shard-index 0
```

- status: started
- note: F84-F86 full source-retention shard launched

## 2026-06-04 20:51:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_full --scope mlp --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f84_f86_full --steps 6400 --shard-count 4 --device cuda:1 --shard-index 1
```

- status: started
- note: F84-F86 full source-retention shard launched

## 2026-06-04 20:51:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_full --scope mlp --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f84_f86_full --steps 6400 --shard-count 4 --device cuda:2 --shard-index 2
```

- status: started
- note: F84-F86 full source-retention shard launched

## 2026-06-04 20:51:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_full --scope mlp --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f84_f86_full --steps 6400 --shard-count 4 --device cuda:3 --shard-index 3
```

- status: started
- note: F84-F86 full source-retention shard launched

## 2026-06-04 20:54:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_full --scope mlp --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f84_f86_full --steps 6400 --shard-count 4 --device cuda:0 --shard-index 0
```

- status: completed
- note: F84-F86 full source-retention shard completed; measured rows=16

## 2026-06-04 20:54:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_full --scope mlp --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f84_f86_full --steps 6400 --shard-count 4 --device cuda:1 --shard-index 1
```

- status: completed
- note: F84-F86 full source-retention shard completed; measured rows=16

## 2026-06-04 20:54:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_full --scope mlp --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f84_f86_full --steps 6400 --shard-count 4 --device cuda:2 --shard-index 2
```

- status: completed
- note: F84-F86 full source-retention shard completed; measured rows=16

## 2026-06-04 20:54:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_full --scope mlp --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f84_f86_full --steps 6400 --shard-count 4 --device cuda:3 --shard-index 3
```

- status: completed
- note: F84-F86 full source-retention shard completed; measured rows=15

## 2026-06-04 20:54:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_full --scope mlp --merge-only
```

- status: completed
- note: F84-F86 full source-retention merge completed

## 2026-06-04 20:54:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_continue_terminal_repair.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_full --tag f84_f86 --new-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source
```

- status: completed
- note: continuation summary rows=63 decision=ContinuationNoH4800SourceChain

## 2026-06-04 20:55:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_01_continue_terminal_repair.py
```

- status: completed
- note: F53/F70/F74 replay summary tag compile check passed

## 2026-06-04 20:57:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_f70_f74_replay_full --scope mlp --spec-ids MLP-F53-trainloss-terminal-lookahead-floor-source,MLP-F70-trainloss-terminal-positive-lookahead-floor-source,MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f53_f70_f74_replay_full --steps 6400 --shard-count 4 --device cuda:0 --shard-index 0
```

- status: started
- note: F53/F70/F74 fresh replay shard launched

## 2026-06-04 20:57:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_f70_f74_replay_full --scope mlp --spec-ids MLP-F53-trainloss-terminal-lookahead-floor-source,MLP-F70-trainloss-terminal-positive-lookahead-floor-source,MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f53_f70_f74_replay_full --steps 6400 --shard-count 4 --device cuda:1 --shard-index 1
```

- status: started
- note: F53/F70/F74 fresh replay shard launched

## 2026-06-04 20:57:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_f70_f74_replay_full --scope mlp --spec-ids MLP-F53-trainloss-terminal-lookahead-floor-source,MLP-F70-trainloss-terminal-positive-lookahead-floor-source,MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f53_f70_f74_replay_full --steps 6400 --shard-count 4 --device cuda:2 --shard-index 2
```

- status: started
- note: F53/F70/F74 fresh replay shard launched

## 2026-06-04 20:57:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_f70_f74_replay_full --scope mlp --spec-ids MLP-F53-trainloss-terminal-lookahead-floor-source,MLP-F70-trainloss-terminal-positive-lookahead-floor-source,MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f53_f70_f74_replay_full --steps 6400 --shard-count 4 --device cuda:3 --shard-index 3
```

- status: started
- note: F53/F70/F74 fresh replay shard launched

## 2026-06-04 21:00:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_f70_f74_replay_full --scope mlp --spec-ids MLP-F53-trainloss-terminal-lookahead-floor-source,MLP-F70-trainloss-terminal-positive-lookahead-floor-source,MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f53_f70_f74_replay_full --steps 6400 --shard-count 4 --device cuda:0 --shard-index 0
```

- status: completed
- note: F53/F70/F74 fresh replay shard completed; measured rows=16

## 2026-06-04 21:00:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_f70_f74_replay_full --scope mlp --spec-ids MLP-F53-trainloss-terminal-lookahead-floor-source,MLP-F70-trainloss-terminal-positive-lookahead-floor-source,MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f53_f70_f74_replay_full --steps 6400 --shard-count 4 --device cuda:1 --shard-index 1
```

- status: completed
- note: F53/F70/F74 fresh replay shard completed; measured rows=16

## 2026-06-04 21:00:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_f70_f74_replay_full --scope mlp --spec-ids MLP-F53-trainloss-terminal-lookahead-floor-source,MLP-F70-trainloss-terminal-positive-lookahead-floor-source,MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f53_f70_f74_replay_full --steps 6400 --shard-count 4 --device cuda:2 --shard-index 2
```

- status: completed
- note: F53/F70/F74 fresh replay shard completed; measured rows=16

## 2026-06-04 21:00:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_f70_f74_replay_full --scope mlp --spec-ids MLP-F53-trainloss-terminal-lookahead-floor-source,MLP-F70-trainloss-terminal-positive-lookahead-floor-source,MLP-F74-trainloss-terminal-optimizer-selector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2201_cont_f53_f70_f74_replay_full --steps 6400 --shard-count 4 --device cuda:3 --shard-index 3
```

- status: completed
- note: F53/F70/F74 fresh replay shard completed; measured rows=15

## 2026-06-04 21:00:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_f70_f74_replay_full --scope mlp --merge-only
```

- status: completed
- note: F53/F70/F74 fresh replay merge completed

## 2026-06-04 21:00:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_continue_terminal_repair.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_f70_f74_replay_full --tag f53_f70_f74_replay --new-ids MLP-F53-trainloss-terminal-lookahead-floor-source,MLP-F70-trainloss-terminal-positive-lookahead-floor-source,MLP-F74-trainloss-terminal-optimizer-selector
```

- status: completed
- note: continuation summary rows=63 decision=ContinuationNoH4800SourceChain

## 2026-06-04 21:01:58 +0800

```bash
python - <<'PY' ... build v22_01_f53_f70_f74_fresh_replay_delta.csv and append recap delta audit ... PY
```

- status: completed
- note: F53/F70/F74 old-vs-fresh replay delta audit appended to recap

## 2026-06-04 21:02:22 +0800

```bash
python - <<'PY' ... rebuild v22.01 packet/bundle after fresh replay delta audit ... PY
```

- status: completed
- note: post-replay packet rebuilt code_sha=97c913f9962c21e568277a5794a3f5acbefcbf933bfa26eac5566b72ef493606 bundle_sha=f31bca1722cd7bdb6c735a3f8ceced37a7a1235ba05c9071398c64c25075c1ea

## 2026-06-04 21:08:44 +0800

```bash
python - <<'PY' ... F53/F70/F74 old-vs-fresh job_index provenance audit ... PY
```

- status: completed
- note: audit_rows=135 paired_candidate_rows=0 shape_rows=15

## 2026-06-04 21:10:38 +0800

```bash
python - <<'PY' ... corrected F53/F70/F74 old-vs-fresh job_index provenance audit ... PY
```

- status: completed
- note: audit_rows=198 paired_candidate_rows=27 same_train_seed_pairs=1

## 2026-06-04 21:11:29 +0800

```bash
python - <<'PY' ... F53 MNIST seed0 raw val/control delta ... PY
```

- status: completed
- note: rows=10 old_label=v2200_f53_terminal_lookahead_floor_full_h4800 fresh_label=v2201_cont_f53_f70_f74_replay_full

## 2026-06-04 21:14:26 +0800

```bash
python - <<'PY' ... F53 replay parameter-drift audit from command journals ... PY
```

- status: completed
- note: old F53 used train_size=512 val_size=256 batch_size=64 steps=4800 single-candidate run; prior v22.01 combined replay used defaults

## 2026-06-04 21:14:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_oldshape_replay_full --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f53_oldshape_replay_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F53-trainloss-terminal-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: started
- note: F53 old-shape single-candidate replay; matches old v22 train/val/batch/steps/spec shape

## 2026-06-04 21:19:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_oldshape_replay_full --device cuda:0 --data-root data --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f53_oldshape_replay_full_h4800 --shard-count 1 --shard-index 0 --spec-ids MLP-F53-trainloss-terminal-lookahead-floor-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
```

- status: completed
- note: F53 old-shape replay shard completed

## 2026-06-04 21:19:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_oldshape_replay_full --scope mlp --merge-only
```

- status: started
- note: merge F53 old-shape replay source-retention matrix

## 2026-06-04 21:19:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_oldshape_replay_full --scope mlp --merge-only
```

- status: completed
- note: F53 old-shape replay merge completed

## 2026-06-04 21:21:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f84_f86_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:0 --shard-index 0
```

- status: started
- note: F84-F86 old-shape 4GPU terminal-hold repair shard launched

## 2026-06-04 21:21:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f84_f86_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:1 --shard-index 1
```

- status: started
- note: F84-F86 old-shape 4GPU terminal-hold repair shard launched

## 2026-06-04 21:21:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f84_f86_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:2 --shard-index 2
```

- status: started
- note: F84-F86 old-shape 4GPU terminal-hold repair shard launched

## 2026-06-04 21:21:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f84_f86_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:3 --shard-index 3
```

- status: started
- note: F84-F86 old-shape 4GPU terminal-hold repair shard launched

## 2026-06-04 21:24:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f84_f86_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:0 --shard-index 0
```

- status: completed
- note: F84-F86 old-shape shard completed; rows=16 traces=160

## 2026-06-04 21:24:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f84_f86_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:1 --shard-index 1
```

- status: completed
- note: F84-F86 old-shape shard completed; rows=16 traces=160

## 2026-06-04 21:24:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f84_f86_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:2 --shard-index 2
```

- status: completed
- note: F84-F86 old-shape shard completed; rows=16 traces=160

## 2026-06-04 21:24:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f84_f86_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F84-h2400-terminal-hold-source,MLP-F85-h2800-terminal-hold-source,MLP-F86-h2400-debt-bailout-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:3 --shard-index 3
```

- status: completed
- note: F84-F86 old-shape shard completed; rows=15 traces=150

## 2026-06-04 21:25:04 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_oldshape_full --scope mlp --merge-only
```

- status: started
- note: merge F84-F86 old-shape replay source-retention matrix

## 2026-06-04 21:25:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_oldshape_full --scope mlp --merge-only
```

- status: completed
- note: F84-F86 old-shape merge completed

## 2026-06-04 21:26:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f83_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f78_f83_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:0 --shard-index 0
```

- status: started
- note: F78-F83 old-shape 4GPU terminal-route repair shard launched

## 2026-06-04 21:26:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f83_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f78_f83_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:1 --shard-index 1
```

- status: started
- note: F78-F83 old-shape 4GPU terminal-route repair shard launched

## 2026-06-04 21:26:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f83_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f78_f83_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:2 --shard-index 2
```

- status: started
- note: F78-F83 old-shape 4GPU terminal-route repair shard launched

## 2026-06-04 21:26:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f83_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f78_f83_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:3 --shard-index 3
```

- status: started
- note: F78-F83 old-shape 4GPU terminal-route repair shard launched

## 2026-06-04 21:31:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f83_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f78_f83_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:0 --shard-index 0
```

- status: completed
- note: F78-F83 old-shape shard completed; rows=23 traces=230

## 2026-06-04 21:31:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f83_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f78_f83_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:1 --shard-index 1
```

- status: completed
- note: F78-F83 old-shape shard completed; rows=23 traces=230

## 2026-06-04 21:31:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f83_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f78_f83_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:2 --shard-index 2
```

- status: completed
- note: F78-F83 old-shape shard completed; rows=22 traces=220

## 2026-06-04 21:31:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f83_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f78_f83_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F78-terminal-source-conserving-route,MLP-F79-trainloss-risk-profile-route,MLP-F80-debt-aware-source-gate,MLP-F81-ungated-warm-terminal-source-route,MLP-F82-ungated-warm-risk-profile-route,MLP-F83-ungated-warm-debt-raw-bailout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:3 --shard-index 3
```

- status: completed
- note: F78-F83 old-shape shard completed; rows=22 traces=220

## 2026-06-04 21:31:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f83_oldshape_full --scope mlp --merge-only
```

- status: started
- note: merge F78-F83 old-shape replay source-retention matrix

## 2026-06-04 21:31:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f83_oldshape_full --scope mlp --merge-only
```

- status: completed
- note: F78-F83 old-shape merge completed

## 2026-06-04 21:36:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f75_f77_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:0 --shard-index 0
```

- status: started
- note: F75-F77 old-shape 4GPU invariant/optimizer repair shard launched

## 2026-06-04 21:36:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f75_f77_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:1 --shard-index 1
```

- status: started
- note: F75-F77 old-shape 4GPU invariant/optimizer repair shard launched

## 2026-06-04 21:36:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f75_f77_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:2 --shard-index 2
```

- status: started
- note: F75-F77 old-shape 4GPU invariant/optimizer repair shard launched

## 2026-06-04 21:36:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f75_f77_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:3 --shard-index 3
```

- status: started
- note: F75-F77 old-shape 4GPU invariant/optimizer repair shard launched

## 2026-06-04 21:48:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f75_f77_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:0 --shard-index 0
```

- status: completed
- note: F75-F77 old-shape shard completed; rows=16 traces=160

## 2026-06-04 21:48:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f75_f77_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:1 --shard-index 1
```

- status: completed
- note: F75-F77 old-shape shard completed; rows=16 traces=160

## 2026-06-04 21:48:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f75_f77_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:2 --shard-index 2
```

- status: completed
- note: F75-F77 old-shape shard completed; rows=16 traces=160

## 2026-06-04 21:48:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f75_f77_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:3 --shard-index 3
```

- status: completed
- note: F75-F77 old-shape shard completed; rows=15 traces=150

## 2026-06-04 21:49:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_oldshape_full --scope mlp --merge-only
```

- status: started
- note: merge F75-F77 old-shape replay source-retention matrix

## 2026-06-04 21:49:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_oldshape_full --scope mlp --merge-only
```

- status: completed
- note: F75-F77 old-shape merge completed

## 2026-06-04 21:50:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_continue_terminal_repair.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_oldshape_full --tag f75_f77_oldshape --title "F75-F77 old-shape invariant/optimizer replay" --new-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only --skip-packet
```

- status: started
- note: summarize F75-F77 old-shape continuation and append recap

## 2026-06-04 21:50:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_continue_terminal_repair.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_oldshape_full --tag f75_f77_oldshape --title "F75-F77 old-shape invariant/optimizer replay" --new-ids MLP-F75-dataset-invariant-poprisk-slow-source,MLP-F76-dataset-invariant-readout-consensus-source,MLP-F77-source-conserving-optimizer-only --skip-packet
```

- status: completed
- note: F75-F77 old-shape continuation summary completed

## 2026-06-04 21:53:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_oldshape_continuation_audit.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01
```

- status: started
- note: aggregate F53/F75-F86 old-shape continuation audit and append recap

## 2026-06-04 21:53:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_oldshape_continuation_audit.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01
```

- status: failed
- note: oldshape continuation audit import failed: repo root missing from sys.path; patching script import guard

## 2026-06-04 21:54:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_oldshape_continuation_audit.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01
```

- status: started
- note: retry aggregate oldshape continuation audit after import guard patch

## 2026-06-04 21:54:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_oldshape_continuation_audit.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01
```

- status: completed
- note: oldshape continuation aggregate audit completed; route/candidate/dataset CSVs written and recap appended

## 2026-06-04 22:02:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f87_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f87_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F87-terminal-raw-then-source-guard,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:0 --shard-index 0
```

- status: started
- note: F87 old-shape 4GPU terminal raw-then-source-guard shard launched

## 2026-06-04 22:02:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f87_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f87_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F87-terminal-raw-then-source-guard,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:1 --shard-index 1
```

- status: started
- note: F87 old-shape 4GPU terminal raw-then-source-guard shard launched

## 2026-06-04 22:02:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f87_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f87_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F87-terminal-raw-then-source-guard,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:2 --shard-index 2
```

- status: started
- note: F87 old-shape 4GPU terminal raw-then-source-guard shard launched

## 2026-06-04 22:02:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f87_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f87_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F87-terminal-raw-then-source-guard,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:3 --shard-index 3
```

- status: started
- note: F87 old-shape 4GPU terminal raw-then-source-guard shard launched

## 2026-06-04 22:05:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f87_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f87_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F87-terminal-raw-then-source-guard,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:0 --shard-index 0
```

- status: completed
- note: F87 old-shape shard completed; rows=12 traces=120

## 2026-06-04 22:05:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f87_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f87_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F87-terminal-raw-then-source-guard,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:1 --shard-index 1
```

- status: completed
- note: F87 old-shape shard completed; rows=11 traces=110

## 2026-06-04 22:05:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f87_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f87_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F87-terminal-raw-then-source-guard,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:2 --shard-index 2
```

- status: completed
- note: F87 old-shape shard completed; rows=11 traces=110

## 2026-06-04 22:05:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f87_oldshape_full --scope mlp --carriers MLP --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --lr 0.003 --fu-lr 0.0001 --alt-period 50 --init-seed-offset 0 --run-label v2201_f87_oldshape_full_h4800 --shard-count 4 --spec-ids MLP-F87-terminal-raw-then-source-guard,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --device cuda:3 --shard-index 3
```

- status: completed
- note: F87 old-shape shard completed; rows=11 traces=110

## 2026-06-04 22:05:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f87_oldshape_full --scope mlp --merge-only
```

- status: started
- note: merge F87 old-shape replay source-retention matrix

## 2026-06-04 22:06:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f87_oldshape_full --scope mlp --merge-only
```

- status: completed
- note: F87 old-shape merge completed

## 2026-06-04 22:06:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_continue_terminal_repair.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f87_oldshape_full --tag f87_oldshape --title "F87 old-shape terminal raw-then-source-guard repair" --new-ids MLP-F87-terminal-raw-then-source-guard --skip-packet
```

- status: started
- note: summarize F87 old-shape continuation and append recap

## 2026-06-04 22:07:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_continue_terminal_repair.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f87_oldshape_full --tag f87_oldshape --title "F87 old-shape terminal raw-then-source-guard repair" --new-ids MLP-F87-terminal-raw-then-source-guard --skip-packet
```

- status: completed
- note: F87 old-shape continuation summary completed

## 2026-06-04 22:08:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_oldshape_continuation_audit.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01
```

- status: started
- note: rerun aggregate oldshape continuation audit including F87

## 2026-06-04 22:09:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_01_oldshape_continuation_audit.py --out-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01
```

- status: completed
- note: oldshape continuation aggregate audit including F87 completed

## 2026-06-04 22:10:33 +0800

```bash
python - <<'PY'  # append v22.01 post-F87 closure note
from experiments.run_v22_01_common import append_text, V2201_RECAP_DOC
...
PY
```

- status: started
- note: append post-F87 closure note to recap

## 2026-06-04 22:10:33 +0800

```bash
python - <<'PY'  # append v22.01 post-F87 closure note
from experiments.run_v22_01_common import append_text, V2201_RECAP_DOC
...
PY
```

- status: completed
- note: post-F87 closure note appended to recap

## 2026-06-04 22:10:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'  # build v22.01 packet/bundle with F75-F87 old-shape continuations
from experiments.run_v22_01_common import build_packet
build_packet(...)
PY
```

- status: started
- note: build final v22.01 packet and results bundle after F87

## 2026-06-04 22:11:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'  # build v22.01 packet/bundle with F75-F87 old-shape continuations
from experiments.run_v22_01_common import build_packet
build_packet(...)
PY
```

- status: completed
- note: packet=v22_01_code_review_packet.zip sha256=e14906673b699317b2c41f2b791172b385d4b04ca9b05986986df954eb9983c1; bundle=v22_01_results_bundle.zip sha256=f5122e2fb830c0f04f0a08e77dd1ee38c8deb6e819378c08d6803b83c5e440b9
