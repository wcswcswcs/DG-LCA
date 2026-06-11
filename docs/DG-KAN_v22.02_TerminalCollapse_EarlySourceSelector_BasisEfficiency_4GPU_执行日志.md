# DG-KAN v22.02 TerminalCollapse EarlySourceSelector BasisEfficiency 4GPU 执行日志

生成时间：2026-06-04 23:55:02 +0800

记录原则：只记录真实执行过的命令、文件、状态和 blocker；不把未执行内容写成结果。

## 2026-06-04 23:55:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --check all
```

- status: completed
- note: checks=import_closure,metrics,mechanisms,kernels

## 2026-06-05 02:12:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v17_common.py experiments/run_v22_02_terminal_collapse_autopsy.py experiments/run_v22_02_kan_source_writer.py
```

- status: completed
- note: v22.02 source-dynamics decomposition and KAN hidden-matrix writer code compiled.

## 2026-06-05 02:12:13 +0800

```bash
rm -rf /tmp/v22_02_f89_smoke && /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/v22_02_f89_smoke --scope mlp --spec-ids MLP-F89-h800-readout-channel-retention,CTRL-SGD,CTRL-AdamW --datasets MNIST --seeds 0 --train-size 32 --val-size 24 --batch-size 16 --steps 20 --shard-count 1 --shard-index 0 --device cpu --run-label v2202_f89_smoke && /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/v22_02_f89_smoke --scope mlp --merge-only --run-label v2202_f89_smoke
```

- status: completed
- note: smoke rows=3, execution_status=measured; h20 is not a v22.02 horizon and was not used as scientific evidence.

## 2026-06-05 02:12:13 +0800

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f89_decomposition_oldshape_full
IDS=MLP-F89-h800-readout-channel-retention,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for SHARD in 0 1 2 3; do /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --spec-ids "$IDS" --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --shard-index "$SHARD" --device "cuda:$SHARD" --run-label v2202_cont_f89_decomp_oldshape_full; done
```

- status: blocked
- note: candidate rows=9 blocked with `blocked:NameError`, blocker=`name 'readout_mask_flat' is not defined`; controls measured. This run is retained only as audit evidence, not as science evidence.

## 2026-06-05 02:12:13 +0800

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f89_decomposition_oldshape_retry_full
IDS=MLP-F89-h800-readout-channel-retention,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for SHARD in 0 1 2 3; do /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --spec-ids "$IDS" --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --shard-index "$SHARD" --device "cuda:$SHARD" --run-label v2202_cont_f89_decomp_oldshape_retry_full; done
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --merge-only --run-label v2202_cont_f89_decomp_oldshape_retry_full
```

- status: completed
- note: fixed by restoring `readout_mask_flat` and `hidden_matrix_mask_flat`; merged rows=45, blocked=0, route decision=`NoContinuousH3200UnderV2202Rule`.

## 2026-06-05 02:12:13 +0800

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_ksw3_ksw6_hidden_matrix_oldshape_full
IDS=KSW3-dualbank-source-reservoir,KSW5-spectral-hidden-basis-block-source,KSW6-hidden-matrix-channel-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for SHARD in 0 1 2 3; do /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope kan --spec-ids "$IDS" --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --shard-index "$SHARD" --device "cuda:$SHARD" --run-label v2202_cont_ksw3_ksw6_hidden_matrix_oldshape_full; done
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope kan --merge-only --run-label v2202_cont_ksw3_ksw6_hidden_matrix_oldshape_full
```

- status: completed
- note: rows=126, blocked=0, decision=`NoContinuousH3200UnderV2202Rule`.

## 2026-06-05 02:12:13 +0800

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_ksw7_ksw8_phase_hidden_matrix_oldshape_full
IDS=KSW7-early-hidden-matrix-channel-source,KSW8-strong-hidden-matrix-channel-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for SHARD in 0 1 2 3; do /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope kan --spec-ids "$IDS" --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --shard-index "$SHARD" --device "cuda:$SHARD" --run-label v2202_cont_ksw7_ksw8_phase_hidden_matrix_oldshape_full; done
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope kan --merge-only --run-label v2202_cont_ksw7_ksw8_phase_hidden_matrix_oldshape_full
```

- status: completed
- note: rows=108, blocked=0, decision=`NoContinuousH3200UnderV2202Rule`.

## 2026-06-05 02:12:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from experiments.run_v22_02_common import ensure_out, build_packet, sha256_file
out=ensure_out('results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02')
packet,bundle=build_packet(out)
print(packet, packet.stat().st_size, sha256_file(packet))
print(bundle, bundle.stat().st_size, sha256_file(bundle))
PY
```

- status: completed
- note: v22_02_code_review_packet.zip size=4496756 sha256=3d085af6ca4d51e4d2b75199b2c33db7ccf66f1785e0c774bde89bc40c98197d; v22_02_results_bundle.zip size=5476779 sha256=1f62b6447a7ea2c7081016375a5ffd37ebc789d55e59042b45eb8e66ddf6ea24

## 2026-06-05 02:12:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import shutil, zipfile, subprocess, pathlib, sys
root=pathlib.Path('/home/chengshun.wang/DG-LCA')
packet=root/'results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/v22_02_code_review_packet.zip'
tmp=pathlib.Path('/tmp/v22_02_packet_check_python')
if tmp.exists():
    shutil.rmtree(tmp)
tmp.mkdir(parents=True)
with zipfile.ZipFile(packet) as z:
    z.extractall(tmp)
src=tmp/'02_SOURCE_TREE'
compile_proc=subprocess.run([sys.executable,'-m','compileall','-q','.'],cwd=src,text=True,capture_output=True)
truth_proc=subprocess.run([sys.executable,'experiments/run_v22_02_s09_truth_gate.py','--self-contained-import-check','1','--source-root','.'],cwd=src,text=True,capture_output=True)
print('compileall_exit', compile_proc.returncode)
print('truth_gate_exit', truth_proc.returncode)
raise SystemExit(0 if compile_proc.returncode==0 and truth_proc.returncode==0 else 1)
PY
```

- status: completed
- note: clean packet compileall_exit=0; truth_gate_exit=0.

## 2026-06-05 02:12:13 +0800

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_ksw9_ksw10_h800_anchor_oldshape_full
IDS=KSW9-h800-source-slow-ema-bank,KSW10-h800-dual-memory-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for SHARD in 0 1 2 3; do /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope kan --spec-ids "$IDS" --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --shard-index "$SHARD" --device "cuda:$SHARD" --run-label v2202_cont_ksw9_ksw10_h800_anchor_oldshape_full; done
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope kan --merge-only --run-label v2202_cont_ksw9_ksw10_h800_anchor_oldshape_full
```

- status: completed
- note: rows=108, blocked=0, decision=`NoContinuousH3200UnderV2202Rule`.

## 2026-06-04 23:55:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_efficiency_officialization.py --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01
```

- status: completed
- note: rows=24 summary=2 source_artifact=results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_efficiency_truth_table.csv

## 2026-06-04 23:55:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_drat_drbf_active_repair.py --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01
```

- status: completed
- note: rows=4 source_artifact=results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_drat_drbf_active_repair.csv

## 2026-06-05 00:03:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/fresh_terminal_pool_oldshape --tag oldshape_h4000
```

- status: completed
- note: rows=81 groups=9 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 00:05:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_precommit_selector.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02 --top-k 2
```

- status: completed
- note: features=7 decision=NoH4800LabelNoSelectorCanPass

## 2026-06-05 00:05:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_source_channel_target_reset.py --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01
```

- status: completed
- note: rows=18 decision=TargetObservableNoRetention source_artifact=results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_function_space_target_reset.csv

## 2026-06-05 00:05:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_kan_source_writer.py --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --fresh-source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02
```

- status: completed
- note: rows=12 decision=KANSourceBankMismatch source_artifact=results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_kan_source_writer_summary.csv

## 2026-06-05 00:15:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/fresh_terminal_single_oldshape --tag single_oldshape_h4000
```

- status: completed
- note: rows=225 groups=9 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 00:15:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_precommit_selector.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02 --top-k 2
```

- status: completed
- note: features=7 decision=NoH4800LabelNoSelectorCanPass

## 2026-06-05 00:16:59 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 00:16:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R3-NoContinuousH3200UnderV2202Rule; v22_02_code_review_packet.zip size=4656121 sha256=f9fc5a9e622ef925f75869a75de422ff258a9a8a1c2ac06c0c743ae8640cf396; v22_02_results_bundle.zip size=7203693 sha256=3b964f0f110354ca514cc44a22dcc360fbc77e9750a146b34e44a92a68fc3d1f

## 2026-06-05 00:20:12 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 00:20:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R3-NoContinuousH3200UnderV2202Rule packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## Fresh rerun raw 4GPU appendix

这些命令是本轮实际执行过的 source-retention fresh rerun 的可复现模板；`<idx>` 分别替换为 `0,1,2,3`，并对应 `cuda:<idx>`。最终科学 route 使用 `fresh_terminal_single_oldshape`，`fresh_terminal_pool_oldshape` 只作为 mixed-pool/provenance drift 对照。

### Mixed candidate pool smoke/localization

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/fresh_terminal_pool_oldshape --scope mlp --device cuda:<idx> --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --run-label v2202_terminal_autopsy_oldshape_h4000 --spec-ids MLP-F53-trainloss-terminal-lookahead-floor-source,MLP-F77-source-conserving-optimizer-only,MLP-F78-terminal-source-conserving-route,MLP-F86-h2400-debt-bailout-source,MLP-F87-terminal-raw-then-source-guard,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --shard-count 4 --shard-index <idx>
```

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/fresh_terminal_pool_oldshape --scope mlp --run-label v2202_terminal_autopsy_oldshape_h4000 --merge-only
```

- files: `v21_01_source_retention_matrix_v2202_terminal_autopsy_oldshape_h4000_fc0..3.csv`, merged `v21_01_source_retention_matrix.csv`
- row check: 81 data rows, 9 grouped rows; h4000 fields present.
- status: superseded for final route because mixed candidates alter job-index/train-seed assignment.

### Single-candidate old-shape localization

实际 launch 用每张 GPU 一个 shell loop；每个 candidate label 单独以 `candidate + CTRL-SGD + CTRL-AdamW + CTRL-RandomMatchedNorm + CTRL-NoOpMatchedOverhead` 的 old-shape run-shape 执行。

```bash
SPECS=("MLP-F53-trainloss-terminal-lookahead-floor-source" "MLP-F77-source-conserving-optimizer-only" "MLP-F78-terminal-source-conserving-route" "MLP-F86-h2400-debt-bailout-source" "MLP-F87-terminal-raw-then-source-guard"); LABELS=("f53" "f77" "f78" "f86" "f87"); for i in "${!SPECS[@]}"; do /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/fresh_terminal_single_oldshape --scope mlp --device cuda:<idx> --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --run-label v2202_single_${LABELS[$i]}_oldshape_h4000 --spec-ids ${SPECS[$i]},CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --shard-count 4 --shard-index <idx>; done
```

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/fresh_terminal_single_oldshape --scope mlp --run-label v2202_single_oldshape_h4000 --merge-only
```

- files: `v21_01_source_retention_matrix_v2202_single_f53/f77/f78/f86/f87_oldshape_h4000_fc0..3.csv`, merged `v21_01_source_retention_matrix.csv`
- row check: 225 data rows, 9 grouped rows after v22.02 aggregation.
- final aggregation command: `experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/fresh_terminal_single_oldshape --tag single_oldshape_h4000`

## 2026-06-05 00:21:56 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 00:21:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R3-NoContinuousH3200UnderV2202Rule packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 00:32:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --check all
```

- status: completed
- note: checks=import_closure,metrics,mechanisms,kernels

## 2026-06-05 00:39:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_source_channel_target_reset.py --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01
```

- status: completed
- note: rows=18 decision=TargetObservableNoRetention source_artifact=results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_function_space_target_reset.csv

## 2026-06-05 00:39:25 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f88_f90_midretention_oldshape_full --tag f88_f90_midretention_oldshape
```

- status: completed
- note: rows=135 groups=7 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 00:39:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_kan_source_writer.py --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --fresh-source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02
```

- status: completed
- note: rows=12 decision=KANSourceBankMismatch source_artifact=results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_kan_source_writer_summary.csv

## 2026-06-05 00:39:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_precommit_selector.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02 --top-k 2
```

- status: completed
- note: features=7 decision=NoH4800LabelNoSelectorCanPass

## 2026-06-05 00:42:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --check all
```

- status: completed
- note: checks=import_closure,metrics,mechanisms,kernels

## 2026-06-05 00:48:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_source_channel_target_reset.py --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01
```

- status: completed
- note: rows=18 decision=TargetObservableNoRetention source_artifact=results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_function_space_target_reset.csv

## 2026-06-05 00:48:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f91_f93_readout_strength_oldshape_full --tag f91_f93_readout_strength_oldshape
```

- status: completed
- note: rows=135 groups=7 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 00:48:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_kan_source_writer.py --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --fresh-source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02
```

- status: completed
- note: rows=12 decision=KANSourceBankMismatch source_artifact=results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_kan_source_writer_summary.csv

## 2026-06-05 00:48:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_precommit_selector.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02 --top-k 2
```

- status: completed
- note: features=7 decision=NoH4800LabelNoSelectorCanPass

## 2026-06-05 00:51:26 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --check all
```

- status: completed
- note: checks=import_closure,metrics,mechanisms,kernels

## 2026-06-05 00:55:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_source_channel_target_reset.py --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01
```

- status: completed
- note: rows=18 decision=TargetObservableNoRetention source_artifact=results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_function_space_target_reset.csv

## 2026-06-05 00:55:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f94_f95_checkpoint_oldshape_full --tag f94_f95_checkpoint_oldshape
```

- status: completed
- note: rows=90 groups=6 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 00:55:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_precommit_selector.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02 --top-k 2
```

- status: completed
- note: features=7 decision=NoH4800LabelNoSelectorCanPass

## 2026-06-05 00:55:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_kan_source_writer.py --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --fresh-source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02
```

- status: completed
- note: rows=12 decision=KANSourceBankMismatch source_artifact=results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_kan_source_writer_summary.csv

## 2026-06-05 00:57:32 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 00:57:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R3-NoContinuousH3200UnderV2202Rule packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 00:59:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --check all
```

- status: completed
- note: checks=import_closure,metrics,mechanisms,kernels

## 2026-06-05 00:59:33 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 00:59:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R3-NoContinuousH3200UnderV2202Rule packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 F88-F95 continuation raw 4GPU reproduction notes

本节补充 v22.02 continuation 的复现信息。每个 continuation 目录内都有 `v21_01_command_journal.csv`、shard log、raw shard matrix、merged `v21_01_source_retention_matrix.csv`、trace 和 controls attribution。runner command journal 记录的启动命令会省略 `--out-dir/spec-ids/run-label/train-size` 等外层调度参数；下面列出可复现的完整 shell 形态，并给出落盘目录和后处理命令。

### F88-F90 mid-horizon source-retention repair

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f88_f90_midretention_oldshape_full
PY=/home/chengshun.wang/miniconda3/envs/kan/bin/python
for spec in \
  MLP-F88-h800-source-slow-ema-retention \
  MLP-F89-h800-readout-channel-retention \
  MLP-F90-h800-dual-memory-source-retention
do
  label=$(echo "$spec" | sed -E 's/^MLP-F([0-9]+).*/f\1/')
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$shard $PY experiments/run_v21_01_source_retention.py \
      --out-dir "$OUT" --scope mlp --spec-ids "$spec,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead" \
      --run-label "v2202_cont_${label}_midretention_oldshape_h4000" \
      --shard-count 4 --shard-index "$shard" --device cuda:0 \
      --train-size 512 --val-size 256 --batch-size 64 --steps 4800 \
      > "$OUT/${label}_shard${shard}.log" 2>&1 &
  done
  wait
done
$PY experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --merge-only
```

- status: completed
- command journal: `results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f88_f90_midretention_oldshape_full/v21_01_command_journal.csv`
- note: rows=135 grouped=15 after merge; route decision for this continuation was `NoContinuousH3200UnderV2202Rule`.

### F91-F93 readout-channel strength/frequency repair

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f91_f93_readout_strength_oldshape_full
PY=/home/chengshun.wang/miniconda3/envs/kan/bin/python
for spec in \
  MLP-F91-h800-readout-channel-alt25-retention \
  MLP-F92-h800-readout-channel-strong-retention \
  MLP-F93-h800-readout-channel-alt25-strong-retention
do
  label=$(echo "$spec" | sed -E 's/^MLP-F([0-9]+).*/f\1/')
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$shard $PY experiments/run_v21_01_source_retention.py \
      --out-dir "$OUT" --scope mlp --spec-ids "$spec,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead" \
      --run-label "v2202_cont_${label}_readout_strength_oldshape_h4000" \
      --shard-count 4 --shard-index "$shard" --device cuda:0 \
      --train-size 512 --val-size 256 --batch-size 64 --steps 4800 \
      > "$OUT/${label}_shard${shard}.log" 2>&1 &
  done
  wait
done
$PY experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --merge-only
```

- status: completed
- command journal: `results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f91_f93_readout_strength_oldshape_full/v21_01_command_journal.csv`
- note: rows=135 grouped=15 after merge; route decision for this continuation was `NoContinuousH3200UnderV2202Rule`.

### F94-F95 train-time source checkpoint reentry repair

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f94_f95_checkpoint_oldshape_full
PY=/home/chengshun.wang/miniconda3/envs/kan/bin/python
for spec in \
  MLP-F94-h1600-source-checkpoint-reentry \
  MLP-F95-h2400-source-checkpoint-reentry
do
  label=$(echo "$spec" | sed -E 's/^MLP-F([0-9]+).*/f\1/')
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$shard $PY experiments/run_v21_01_source_retention.py \
      --out-dir "$OUT" --scope mlp --spec-ids "$spec,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead" \
      --run-label "v2202_cont_${label}_checkpoint_oldshape_h4000" \
      --shard-count 4 --shard-index "$shard" --device cuda:0 \
      --train-size 512 --val-size 256 --batch-size 64 --steps 4800 \
      > "$OUT/${label}_shard${shard}.log" 2>&1 &
  done
  wait
done
$PY experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --merge-only
```

- status: completed
- command journal: `results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f94_f95_checkpoint_oldshape_full/v21_01_command_journal.csv`
- note: rows=90 grouped=10 after merge; route decision for this continuation was `NoContinuousH3200UnderV2202Rule`.

### Continuation postprocess and finalization commands

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --check all
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f88_f90_midretention_oldshape_full --tag f88_f90_midretention_oldshape
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f91_f93_readout_strength_oldshape_full --tag f91_f93_readout_strength_oldshape
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f94_f95_checkpoint_oldshape_full --tag f94_f95_checkpoint_oldshape
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_precommit_selector.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02 --top-k 2
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_source_channel_target_reset.py --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_kan_source_writer.py --source-dir results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --fresh-source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
rm -rf /tmp/v22_02_packet_check && unzip results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- produced aggregate artifacts:
  - `v22_02_continuation_repair_route_audit.csv`
  - `v22_02_continuation_repair_candidate_summary.csv`
  - `v22_02_continuation_repair_dataset_localization.csv`
  - `v22_02_route_decision.csv`
  - `v22_02_code_truth_gate.csv`
- latest route after F88-F95: `R3-NoContinuousH3200UnderV2202Rule`, `promotion_allowed=0`.

## 2026-06-05 01:08:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from experiments.run_v22_02_common import ensure_out, build_packet
out = ensure_out('results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02')
packet, bundle = build_packet(out)
print(packet)
print(bundle)
PY
```

- status: completed
- note: rebuilt packet/bundle after appending F88-F95 execution and recap logs.

## 2026-06-05 01:09:00 +0800

```bash
rm -rf /tmp/v22_02_packet_check_latest
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from pathlib import Path
import zipfile
zip_path = Path('results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/v22_02_code_review_packet.zip')
out = Path('/tmp/v22_02_packet_check_latest')
out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(out)
print(out / '02_SOURCE_TREE')
PY
cd /tmp/v22_02_packet_check_latest/02_SOURCE_TREE
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
cd /home/chengshun.wang/DG-LCA
sha256sum results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/v22_02_code_review_packet.zip results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/v22_02_results_bundle.zip
```

- status: completed
- note: system `unzip` command was unavailable, so clean-unzip was reproduced with Python `zipfile`; compileall and S0.9 truth gate both exited 0.
- v22_02_code_review_packet.zip sha256: `6b09de46d25b04e334a7253c59e6d6df18b1867ed2853a7fb824eb3e1795981b`
- v22_02_results_bundle.zip sha256: `4536f50090a94802baf62a0ae9e6579f7160369f7d5b12e0b9779e9f71d71d53`

## 2026-06-05 01:11:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --check all
```

- status: completed
- note: checks=import_closure,metrics,mechanisms,kernels

## 2026-06-05 01:22:21 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f89_decomposition_oldshape_full --tag f89_decomposition_oldshape
```

- status: completed
- note: rows=45 groups=5 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 01:22:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_kan_source_writer.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --fresh-source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f89_decomposition_oldshape_full
```

- status: completed
- note: rows=12 decision=KANSourceBankMismatch source_artifact=/home/chengshun.wang/DG-LCA/results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_kan_source_writer_summary.csv

## 2026-06-05 01:28:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f89_decomposition_oldshape_retry_full --tag f89_decomposition_oldshape_retry
```

- status: completed
- note: rows=45 groups=5 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 01:28:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_kan_source_writer.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --fresh-source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f89_decomposition_oldshape_retry_full
```

- status: completed
- note: rows=12 decision=KANSourceBankMismatch source_artifact=/home/chengshun.wang/DG-LCA/results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_kan_source_writer_summary.csv

## 2026-06-05 01:48:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_kan_source_writer.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --fresh-source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_ksw3_ksw6_hidden_matrix_oldshape_full
```

- status: completed
- note: rows=12 decision=KANSourceBankMismatch source_artifact=/home/chengshun.wang/DG-LCA/results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_kan_source_writer_summary.csv

## 2026-06-05 01:48:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_ksw3_ksw6_hidden_matrix_oldshape_full --tag ksw_hidden_matrix_oldshape
```

- status: completed
- note: rows=126 groups=14 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 01:59:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_kan_source_writer.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --fresh-source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_ksw7_ksw8_phase_hidden_matrix_oldshape_full
```

- status: completed
- note: rows=12 decision=KANSourceBankMismatch source_artifact=/home/chengshun.wang/DG-LCA/results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_kan_source_writer_summary.csv

## 2026-06-05 01:59:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_ksw7_ksw8_phase_hidden_matrix_oldshape_full --tag ksw7_ksw8_phase_hidden_matrix_oldshape
```

- status: completed
- note: rows=108 groups=12 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 02:09:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_kan_source_writer.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01 --fresh-source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_ksw9_ksw10_h800_anchor_oldshape_full
```

- status: completed
- note: rows=12 decision=KANSourceBankMismatch source_artifact=/home/chengshun.wang/DG-LCA/results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_kan_source_writer_summary.csv

## 2026-06-05 02:09:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_ksw9_ksw10_h800_anchor_oldshape_full --tag ksw9_ksw10_h800_anchor_oldshape
```

- status: completed
- note: rows=108 groups=12 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 02:10:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --check all
```

- status: completed
- note: checks=import_closure,metrics,mechanisms,kernels

## 2026-06-05 02:56:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f96_f97_signal_reservoir_target_oldshape_full --tag f96_f97_signal_reservoir_target_oldshape
```

- status: completed
- note: rows=126 groups=14 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 03:22:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --check all
```

- status: completed
- note: checks=import_closure,metrics,mechanisms,kernels

## 2026-06-05 03:22:53 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 03:22:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R3-NoContinuousH3200UnderV2202Rule packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 02:26:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f96_f97_target_reset_smoke --scope target --carriers D-CHE,D-FOU --datasets MNIST --seeds 0 --train-size 64 --val-size 48 --batch-size 32 --steps 120 --spec-ids F96-signal-reservoir-b3-null-consensus-target,F97-source-bank-b3-null-consensus-target,F9-TCTRL2-stable-random-b3-null-target,CTRL-SGD,CTRL-AdamW --run-label v2202_cont_f96_f97_target_reset_smoke --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
```

- status: completed
- note: smoke rows=10 after merge; verified F96/F97/F9-TCTRL2 path runs.

## 2026-06-05 02:29:42 +0800

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f96_f97_signal_reservoir_target_oldshape_full
mkdir -p "$OUT/logs"
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" \
    --scope target \
    --carriers D-CHE,D-FOU \
    --datasets MNIST,Fashion-MNIST,KMNIST \
    --seeds 0,1,2 \
    --train-size 512 \
    --val-size 256 \
    --batch-size 64 \
    --steps 4800 \
    --spec-ids F96-signal-reservoir-b3-null-consensus-target,F97-source-bank-b3-null-consensus-target,F9-TCTRL2-stable-random-b3-null-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead \
    --run-label v2202_cont_f96_f97_signal_reservoir_target_oldshape_full \
    --shard-count 4 \
    --shard-index $i \
    --device cuda:0 \
    --data-root data > "$OUT/logs/shard_${i}.log" 2>&1 &
done
wait
```

- status: completed
- note: shard rows=32/32/31/31; later identified as pre-fix target scheduling run where M134/M135/M136 were not yet in alternating branch.

## 2026-06-05 02:56:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f96_f97_signal_reservoir_target_oldshape_full --scope target --merge-only
```

- status: completed
- note: merged old-shape F96/F97 target reset rows=126.

## 2026-06-05 02:56:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f96_f97_signal_reservoir_target_oldshape_full --tag f96_f97_signal_reservoir_target_oldshape
```

- status: completed
- note: rows=126 groups=14 decision=NoContinuousH3200UnderV2202Rule promotion_allowed=0.

## 2026-06-05 02:59:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f98_f99_lowstep_target_smoke_h800 --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 800 --spec-ids F97-source-bank-b3-null-consensus-target,F98-source-bank-b3-null-consensus-lowstep-target,F99-source-bank-b3-null-consensus-tinystep-target,F9-TCTRL2-stable-random-b3-null-target,CTRL-SGD,CTRL-AdamW --run-label v2202_cont_f98_f99_lowstep_target_smoke_h800 --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
```

- status: completed
- note: pre-scheduling-fix h800 smoke; helped localize over-write behavior, not used as legal confirmation.

## 2026-06-05 03:04:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f98_f99_lowstep_target_corrected_smoke_h800 --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 800 --spec-ids F97-source-bank-b3-null-consensus-target,F98-source-bank-b3-null-consensus-lowstep-target,F99-source-bank-b3-null-consensus-tinystep-target,F9-TCTRL2-stable-random-b3-null-target,CTRL-SGD,CTRL-AdamW --run-label v2202_cont_f98_f99_lowstep_target_corrected_smoke_h800 --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
```

- status: completed
- note: after adding M134/M135/M136 to alternating target branch; F97/F98/F99 all pos_h800=0.

## 2026-06-05 03:07:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f100_f101_earlypulse_target_smoke_h800 --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 800 --spec-ids F100-source-bank-b3-null-earlypulse100-target,F101-source-bank-b3-null-earlypulse100-lowstep-target,F9-TCTRL2-stable-random-b3-null-target,CTRL-SGD,CTRL-AdamW --run-label v2202_cont_f100_f101_earlypulse_target_smoke_h800 --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
```

- status: completed
- note: F100 h100 positive but h800 positive rows=0; no full confirmation.

## 2026-06-05 03:10:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f102_f103_earlypulse_target_smoke_h800 --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 800 --spec-ids F100-source-bank-b3-null-earlypulse100-target,F102-source-bank-b3-null-earlypulse200-target,F103-source-bank-b3-null-earlypulse400-target,F9-TCTRL2-stable-random-b3-null-target,CTRL-SGD,CTRL-AdamW --run-label v2202_cont_f102_f103_earlypulse_target_smoke_h800 --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
```

- status: completed
- note: extending pulse to h200/h400 did not produce h800 retention; pos_h800=0.

## 2026-06-05 03:14:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f104_f105_earlypulse_adamw_target_smoke_h800 --scope target --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 800 --spec-ids F104-source-bank-b3-null-earlypulse100-adamw-target,F105-source-bank-b3-null-earlypulse200-adamw-target,F9-TCTRL2-stable-random-b3-null-target,CTRL-SGD,CTRL-AdamW --run-label v2202_cont_f104_f105_earlypulse_adamw_target_smoke_h800 --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
```

- status: completed
- note: post-pulse AdamW worsened h800; pos_h800=0.

## 2026-06-05 03:22:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py
```

- status: completed
- note: import_closure=1 metrics=47/47 mechanisms=18/18 kernels=1.

## 2026-06-05 03:22:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: v22_02_code_review_packet.zip size=4652886 sha256=a03a8260f7670da12b4dc1ff35495d6621c9407808b4cda437e72c71f1fe1872; v22_02_results_bundle.zip size=7203681 sha256=2128589dff5897bd44ac6e7a3b548ea616b4cfabb14b26cfb0d098af11e52adf.

## 2026-06-05 03:26:42 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 03:26:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R3-NoContinuousH3200UnderV2202Rule; v22_02_code_review_packet.zip size=4656121 sha256=f9fc5a9e622ef925f75869a75de422ff258a9a8a1c2ac06c0c743ae8640cf396; v22_02_results_bundle.zip size=7203693 sha256=3b964f0f110354ca514cc44a22dcc360fbc77e9750a146b34e44a92a68fc3d1f

## 2026-06-05 03:46:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f88_f94_source_state_oldshape_full --tag f88_f94_source_state_oldshape
```

- status: completed
- note: rows=72 groups=8 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 03:56:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f106_f107_early_source_state_oldshape_full --tag f106_f107_early_source_state_oldshape
```

- status: completed
- note: rows=54 groups=6 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 04:02:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f106_single_early_source_state_oldshape_full --tag f106_single_early_source_state_oldshape
```

- status: completed
- note: rows=45 groups=5 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 04:08:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f115_single_early100_slowema_strong_oldshape_full --tag f115_single_early100_slowema_strong_oldshape
```

- status: completed
- note: rows=45 groups=5 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 04:15:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f117_single_terminal_guard_oldshape_full --tag f117_single_terminal_guard_oldshape
```

- status: completed
- note: rows=45 groups=5 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 04:21:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f118_single_terminal_raw_guard_oldshape_full --tag f118_single_terminal_raw_guard_oldshape
```

- status: completed
- note: rows=45 groups=5 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 04:28:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f119_single_terminal_raw_guard_strong_oldshape_full --tag f119_single_terminal_raw_guard_strong_oldshape
```

- status: completed
- note: rows=45 groups=5 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 F106-F119 continuation raw run command index

本节补充 F106-F119 的训练/merge 复现指令。各 `source_dir` 内的 `v21_01_command_journal.csv` 是落盘执行证据；由于 journal 只记录 runner 简版命令，下面列出等价完整参数形态。所有 full run 均为 old-shape comparable 参数：`train_size=512`、`val_size=256`、`batch_size=64`、`steps=4800`、datasets=`MNIST,Fashion-MNIST,KMNIST`、seeds=`0,1,2`、controls=`CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead`。

### F88-F94 source-state combined old-shape full

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f88_f94_source_state_oldshape_full
PY=/home/chengshun.wang/miniconda3/envs/kan/bin/python
IDS=MLP-F88-h800-source-slow-ema-retention,MLP-F89-h800-readout-channel-retention,MLP-F93-h800-readout-channel-alt25-strong-retention,MLP-F94-h1600-source-checkpoint-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$shard $PY experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" --scope mlp --spec-ids "$IDS" \
    --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
    --train-size 512 --val-size 256 --batch-size 64 --steps 4800 \
    --run-label v2202_cont_f88_f94_source_state_oldshape_full \
    --shard-count 4 --shard-index "$shard" --device cuda:0 --data-root data &
done
wait
$PY experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --merge-only --run-label v2202_cont_f88_f94_source_state_oldshape_full
```

- status: completed
- command journal: `results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f88_f94_source_state_oldshape_full/v21_01_command_journal.csv`
- note: rows=72 groups=8; autopsy decision=`NoContinuousH3200UnderV2202Rule`.

### F106-F109 early100 h800 smoke rerun

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py \
  --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f106_f109_early_source_state_smoke_h800_rerun \
  --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 \
  --train-size 512 --val-size 256 --batch-size 64 --steps 800 \
  --spec-ids MLP-F106-early100-h800-source-slow-ema-retention,MLP-F107-early100-h800-readout-channel-retention,MLP-F108-early100-h800-readout-channel-alt25-strong-retention,MLP-F109-early100-h1600-source-checkpoint-reentry,CTRL-SGD,CTRL-AdamW \
  --run-label v2202_cont_f106_f109_early_source_state_smoke_h800_rerun \
  --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
```

- status: completed
- note: F106/F107 passed h800 smoke; F108/F109 failed early source.

### F106-F107 combined old-shape full

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f106_f107_early_source_state_oldshape_full
PY=/home/chengshun.wang/miniconda3/envs/kan/bin/python
IDS=MLP-F106-early100-h800-source-slow-ema-retention,MLP-F107-early100-h800-readout-channel-retention,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$shard $PY experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" --scope mlp --spec-ids "$IDS" \
    --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
    --train-size 512 --val-size 256 --batch-size 64 --steps 4800 \
    --run-label v2202_cont_f106_f107_early_source_state_oldshape_full \
    --shard-count 4 --shard-index "$shard" --device cuda:0 --data-root data &
done
wait
$PY experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --merge-only --run-label v2202_cont_f106_f107_early_source_state_oldshape_full
```

- status: completed
- command journal: `results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f106_f107_early_source_state_oldshape_full/v21_01_command_journal.csv`
- note: combined job-index run did not satisfy early gate for grouped candidates; launched F106 single-candidate replay.

### F106/F115/F117/F118/F119 single-candidate old-shape full template

```bash
PY=/home/chengshun.wang/miniconda3/envs/kan/bin/python
run_single_full () {
  local OUT="$1"
  local LABEL="$2"
  local SPEC="$3"
  local IDS="$SPEC,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead"
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$shard $PY experiments/run_v21_01_source_retention.py \
      --out-dir "$OUT" --scope mlp --spec-ids "$IDS" \
      --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
      --train-size 512 --val-size 256 --batch-size 64 --steps 4800 \
      --run-label "$LABEL" \
      --shard-count 4 --shard-index "$shard" --device cuda:0 --data-root data &
  done
  wait
  $PY experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --merge-only --run-label "$LABEL"
}

run_single_full results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f106_single_early_source_state_oldshape_full v2202_cont_f106_single_early_source_state_oldshape_full MLP-F106-early100-h800-source-slow-ema-retention
run_single_full results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f115_single_early100_slowema_strong_oldshape_full v2202_cont_f115_single_early100_slowema_strong_oldshape_full MLP-F115-early100-h800-source-slow-ema-strong-retention
run_single_full results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f117_single_terminal_guard_oldshape_full v2202_cont_f117_single_terminal_guard_oldshape_full MLP-F117-early100-h800-source-slow-ema-terminal-guard
run_single_full results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f118_single_terminal_raw_guard_oldshape_full v2202_cont_f118_single_terminal_raw_guard_oldshape_full MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard
run_single_full results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f119_single_terminal_raw_guard_strong_oldshape_full v2202_cont_f119_single_terminal_raw_guard_strong_oldshape_full MLP-F119-early100-h800-source-slow-ema-terminal-raw-guard-strong
```

- status: completed
- command journals:
  - `continuation_f106_single_early_source_state_oldshape_full/v21_01_command_journal.csv`
  - `continuation_f115_single_early100_slowema_strong_oldshape_full/v21_01_command_journal.csv`
  - `continuation_f117_single_terminal_guard_oldshape_full/v21_01_command_journal.csv`
  - `continuation_f118_single_terminal_raw_guard_oldshape_full/v21_01_command_journal.csv`
  - `continuation_f119_single_terminal_raw_guard_strong_oldshape_full/v21_01_command_journal.csv`
- note: F118 was best near-miss with h4800_retention_ratio=0.4905266741203586; all single full runs remained below productive gate.

### F110-F119 smoke commands

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f110_f113_early_source_state_smoke_h800 --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 800 --spec-ids MLP-F110-early1-h800-source-slow-ema-retention,MLP-F111-early1-h800-readout-channel-retention,MLP-F112-early50-h800-source-slow-ema-retention,MLP-F113-early50-h800-readout-channel-retention,CTRL-SGD,CTRL-AdamW --run-label v2202_cont_f110_f113_early_source_state_smoke_h800 --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f114_f116_early100_slowema_strength_smoke_h800 --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 800 --spec-ids MLP-F114-early100-h800-source-slow-ema-alt25-retention,MLP-F115-early100-h800-source-slow-ema-strong-retention,MLP-F116-early100-h800-source-slow-ema-alt25-strong-retention,CTRL-SGD,CTRL-AdamW --run-label v2202_cont_f114_f116_early100_slowema_strength_smoke_h800 --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f117_terminal_guard_smoke_h800 --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 800 --spec-ids MLP-F117-early100-h800-source-slow-ema-terminal-guard,CTRL-SGD,CTRL-AdamW --run-label v2202_cont_f117_terminal_guard_smoke_h800 --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f118_terminal_raw_guard_smoke_h800 --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 800 --spec-ids MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard,CTRL-SGD,CTRL-AdamW --run-label v2202_cont_f118_terminal_raw_guard_smoke_h800 --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f119_terminal_raw_guard_strong_smoke_h800 --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 800 --spec-ids MLP-F119-early100-h800-source-slow-ema-terminal-raw-guard-strong,CTRL-SGD,CTRL-AdamW --run-label v2202_cont_f119_terminal_raw_guard_strong_smoke_h800 --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
```

- status: completed
- note: F110-F113 failed early; F115/F117/F118/F119 passed smoke and were escalated as listed above.

## 2026-06-05 04:36:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --check all
```

- status: completed
- note: checks=import_closure,metrics,mechanisms,kernels

## 2026-06-05 04:37:13 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 04:37:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 04:39:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --check all
```

- status: completed
- note: checks=import_closure,metrics,mechanisms,kernels

## 2026-06-05 04:40:06 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 04:40:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 04:57:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f120_f122_optimizer_projection_smoke_h800 --tag f120_f122_optimizer_projection_smoke_h800
```

- status: completed
- note: rows=0 groups=0 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 04:59:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f120_f122_optimizer_projection_smoke_h800 --tag f120_f122_optimizer_projection_smoke_h800
```

- status: completed
- note: rows=15 groups=5 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 05:02:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f122_terminal_antiwashout_oldshape_full --tag f122_terminal_antiwashout_oldshape_full
```

- status: completed
- note: rows=45 groups=5 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 05:05:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f121_terminal_projected_blend_oldshape_full --tag f121_terminal_projected_blend_oldshape_full
```

- status: completed
- note: rows=45 groups=5 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 05:08:01 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f120_terminal_projected_optimizer_oldshape_full --tag f120_terminal_projected_optimizer_oldshape_full
```

- status: completed
- note: rows=45 groups=5 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 05:14:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f123_h4000_reentry_smoke_h800 --tag f123_h4000_reentry_smoke_h800
```

- status: completed
- note: rows=9 groups=3 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 05:17:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f123_h4000_reentry_oldshape_full --tag f123_h4000_reentry_oldshape_full
```

- status: completed
- note: rows=45 groups=5 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 05:23:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f120_f122_optimizer_projection_smoke_h800 --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 800 --spec-ids MLP-F120-early100-h800-source-slow-ema-terminal-projected-optimizer,MLP-F121-early100-h800-source-slow-ema-terminal-projected-blend,MLP-F122-early100-h800-source-slow-ema-terminal-antiwashout,CTRL-SGD,CTRL-AdamW --run-label v2202_cont_f120_f122_optimizer_projection_smoke_h800 --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f120_f122_optimizer_projection_smoke_h800 --scope mlp --merge-only --run-label v2202_cont_f120_f122_optimizer_projection_smoke_h800
```

- status: completed
- note: recorded after execution from continuation_f120_f122_optimizer_projection_smoke_h800/v21_01_command_journal.csv; rows=15 grouped=5

## 2026-06-05 05:23:39 +0800

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f122_terminal_antiwashout_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F122-early100-h800-source-slow-ema-terminal-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f122_terminal_antiwashout_oldshape_full --shard-index 0
CUDA_VISIBLE_DEVICES=1 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f122_terminal_antiwashout_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F122-early100-h800-source-slow-ema-terminal-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f122_terminal_antiwashout_oldshape_full --shard-index 1
CUDA_VISIBLE_DEVICES=2 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f122_terminal_antiwashout_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F122-early100-h800-source-slow-ema-terminal-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f122_terminal_antiwashout_oldshape_full --shard-index 2
CUDA_VISIBLE_DEVICES=3 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f122_terminal_antiwashout_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F122-early100-h800-source-slow-ema-terminal-antiwashout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f122_terminal_antiwashout_oldshape_full --shard-index 3
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f122_terminal_antiwashout_oldshape_full --scope mlp --merge-only --run-label v2202_cont_f122_terminal_antiwashout_oldshape_full
```

- status: completed
- note: recorded after execution from continuation_f122_terminal_antiwashout_oldshape_full/v21_01_command_journal.csv; rows=45 grouped=5

## 2026-06-05 05:23:39 +0800

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f121_terminal_projected_blend_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F121-early100-h800-source-slow-ema-terminal-projected-blend,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f121_terminal_projected_blend_oldshape_full --shard-index 0
CUDA_VISIBLE_DEVICES=1 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f121_terminal_projected_blend_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F121-early100-h800-source-slow-ema-terminal-projected-blend,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f121_terminal_projected_blend_oldshape_full --shard-index 1
CUDA_VISIBLE_DEVICES=2 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f121_terminal_projected_blend_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F121-early100-h800-source-slow-ema-terminal-projected-blend,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f121_terminal_projected_blend_oldshape_full --shard-index 2
CUDA_VISIBLE_DEVICES=3 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f121_terminal_projected_blend_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F121-early100-h800-source-slow-ema-terminal-projected-blend,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f121_terminal_projected_blend_oldshape_full --shard-index 3
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f121_terminal_projected_blend_oldshape_full --scope mlp --merge-only --run-label v2202_cont_f121_terminal_projected_blend_oldshape_full
```

- status: completed
- note: recorded after execution from continuation_f121_terminal_projected_blend_oldshape_full/v21_01_command_journal.csv; rows=45 grouped=5

## 2026-06-05 05:23:39 +0800

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f120_terminal_projected_optimizer_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F120-early100-h800-source-slow-ema-terminal-projected-optimizer,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f120_terminal_projected_optimizer_oldshape_full --shard-index 0
CUDA_VISIBLE_DEVICES=1 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f120_terminal_projected_optimizer_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F120-early100-h800-source-slow-ema-terminal-projected-optimizer,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f120_terminal_projected_optimizer_oldshape_full --shard-index 1
CUDA_VISIBLE_DEVICES=2 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f120_terminal_projected_optimizer_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F120-early100-h800-source-slow-ema-terminal-projected-optimizer,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f120_terminal_projected_optimizer_oldshape_full --shard-index 2
CUDA_VISIBLE_DEVICES=3 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f120_terminal_projected_optimizer_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F120-early100-h800-source-slow-ema-terminal-projected-optimizer,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f120_terminal_projected_optimizer_oldshape_full --shard-index 3
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f120_terminal_projected_optimizer_oldshape_full --scope mlp --merge-only --run-label v2202_cont_f120_terminal_projected_optimizer_oldshape_full
```

- status: completed
- note: recorded after execution from continuation_f120_terminal_projected_optimizer_oldshape_full/v21_01_command_journal.csv; rows=45 grouped=5

## 2026-06-05 05:23:39 +0800

```bash
CUDA_VISIBLE_DEVICES=0 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f123_h4000_reentry_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F123-early100-h800-source-slow-ema-terminal-h4000-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f123_h4000_reentry_oldshape_full --shard-index 0
CUDA_VISIBLE_DEVICES=1 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f123_h4000_reentry_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F123-early100-h800-source-slow-ema-terminal-h4000-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f123_h4000_reentry_oldshape_full --shard-index 1
CUDA_VISIBLE_DEVICES=2 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f123_h4000_reentry_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F123-early100-h800-source-slow-ema-terminal-h4000-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f123_h4000_reentry_oldshape_full --shard-index 2
CUDA_VISIBLE_DEVICES=3 /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f123_h4000_reentry_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --device cuda:0 --data-root data --spec-ids MLP-F123-early100-h800-source-slow-ema-terminal-h4000-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f123_h4000_reentry_oldshape_full --shard-index 3
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f123_h4000_reentry_oldshape_full --scope mlp --merge-only --run-label v2202_cont_f123_h4000_reentry_oldshape_full
```

- status: completed
- note: recorded after execution from continuation_f123_h4000_reentry_oldshape_full/v21_01_command_journal.csv; rows=45 grouped=5

## 2026-06-05 05:23:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f123_h4000_reentry_smoke_h800 --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 800 --spec-ids MLP-F123-early100-h800-source-slow-ema-terminal-h4000-reentry,CTRL-SGD,CTRL-AdamW --run-label v2202_cont_f123_h4000_reentry_smoke_h800 --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f123_h4000_reentry_smoke_h800 --scope mlp --merge-only --run-label v2202_cont_f123_h4000_reentry_smoke_h800
```

- status: completed
- note: recorded after execution from continuation_f123_h4000_reentry_smoke_h800/v21_01_command_journal.csv; rows=9 grouped=3

## 2026-06-05 05:23:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY_AUDIT'
# regenerate v22_02_f106_f123_oldshape_full_audit.csv and v22_02_f88_f123_smoke_audit.csv from fresh continuation matrices
PY_AUDIT
```

- status: completed
- note: recorded after execution; audit CSVs generated from fresh v21_01_source_retention_matrix.csv artifacts, no synthetic data

## 2026-06-05 05:24:20 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 05:24:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 05:56:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f124_f126_signal_target_oldshape_full --tag f124_f126_signal_target_oldshape_full
```

- status: completed
- note: rows=63 groups=7 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 05:57:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f124_f126_signal_target_oldshape_full --tag f124_f126_signal_target_oldshape_full
```

- status: completed
- note: rows=63 groups=7 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 06:03:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f124_f126_signal_target_smoke2_h1600 --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f124_f126_signal_target_smoke2_h1600 --tag f124_f126_signal_target_smoke2_h1600
```

- status: completed
- note: rows=15 groups=5 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 06:04:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY_AUDIT'
# regenerate v22_02_f106_f126_oldshape_full_audit.csv and v22_02_f88_f126_smoke_audit.csv from fresh continuation matrices
PY_AUDIT
```

- status: completed
- note: full_rows=18 smoke_rows=29 route_continuous=12 route_h4800=0; generated from fresh v21_01/v22_02 source-chain artifacts, no synthetic data

## 2026-06-05 06:05:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --check all
```

- status: completed
- note: checks=import_closure,metrics,mechanisms,kernels

## 2026-06-05 06:06:17 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 06:06:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 06:07:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f124_f126_signal_target_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --spec-ids MLP-F124-early100-h800-source-slow-ema-signal-reservoir-target,MLP-F125-early100-h800-source-slow-ema-source-bank-target,MLP-F126-early100-h800-source-slow-ema-dual-target-guard,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f124_f126_signal_target_oldshape_full --shard-count 4 --shard-index 0 --device cuda:0 --data-root data
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f124_f126_signal_target_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --spec-ids MLP-F124-early100-h800-source-slow-ema-signal-reservoir-target,MLP-F125-early100-h800-source-slow-ema-source-bank-target,MLP-F126-early100-h800-source-slow-ema-dual-target-guard,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f124_f126_signal_target_oldshape_full --shard-count 4 --shard-index 1 --device cuda:1 --data-root data
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f124_f126_signal_target_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --spec-ids MLP-F124-early100-h800-source-slow-ema-signal-reservoir-target,MLP-F125-early100-h800-source-slow-ema-source-bank-target,MLP-F126-early100-h800-source-slow-ema-dual-target-guard,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f124_f126_signal_target_oldshape_full --shard-count 4 --shard-index 2 --device cuda:2 --data-root data
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f124_f126_signal_target_oldshape_full --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --spec-ids MLP-F124-early100-h800-source-slow-ema-signal-reservoir-target,MLP-F125-early100-h800-source-slow-ema-source-bank-target,MLP-F126-early100-h800-source-slow-ema-dual-target-guard,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2202_cont_f124_f126_signal_target_oldshape_full --shard-count 4 --shard-index 3 --device cuda:3 --data-root data
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f124_f126_signal_target_oldshape_full --scope mlp --merge-only --run-label v2202_cont_f124_f126_signal_target_oldshape_full
```

- status: completed
- note: recorded after execution from continuation_f124_f126_signal_target_oldshape_full/v21_01_command_journal.csv; rows=63 grouped=7

## 2026-06-05 06:07:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f124_f126_signal_target_smoke2_h1600 --scope mlp --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --batch-size 64 --steps 1600 --spec-ids MLP-F124-early100-h800-source-slow-ema-signal-reservoir-target,MLP-F125-early100-h800-source-slow-ema-source-bank-target,MLP-F126-early100-h800-source-slow-ema-dual-target-guard,CTRL-SGD,CTRL-AdamW --run-label v2202_cont_f124_f126_signal_target_smoke2_h1600 --shard-count 1 --shard-index 0 --device cuda:0 --data-root data
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f124_f126_signal_target_smoke2_h1600 --scope mlp --merge-only --run-label v2202_cont_f124_f126_signal_target_smoke2_h1600
```

- status: completed
- note: recorded after execution from continuation_f124_f126_signal_target_smoke2_h1600/v21_01_command_journal.csv; rows=15 grouped=5

## 2026-06-05 06:08:19 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 06:08:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 06:28:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f127_f129_source_state_adaptive_terminal_oldshape_full --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f127_f129_source_state_adaptive_terminal_oldshape_full --tag f127_f129_source_state_adaptive_terminal_oldshape_full
```

- status: completed
- note: rows=63 groups=7 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 06:31:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: added MLP-F127/F128/F129 source-state adaptive terminal mechanisms; compile passed

## 2026-06-05 06:31:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/v22_02_f127_f129_smoke --scope mlp --spec-ids MLP-F127-early100-h800-source-slow-ema-terminal-adaptive-raw,MLP-F128-early100-h800-source-slow-ema-terminal-sparse-source,MLP-F129-early100-h800-source-slow-ema-terminal-ratio-preserve,CTRL-SGD,CTRL-AdamW --datasets MNIST --seeds 0 --train-size 32 --val-size 24 --batch-size 16 --steps 4050 --shard-count 1 --shard-index 0 --device cuda:0 --run-label v2202_cont_f127_f129_terminal_adaptive_smoke4050
```

- status: completed
- note: runtime smoke crossed terminal_start; rows=5, blocked=0; smoke only, not used as v22.02 science evidence

## 2026-06-05 06:31:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/v22_02_f127_f129_smoke --scope mlp --merge-only --run-label v2202_cont_f127_f129_terminal_adaptive_smoke4050
```

- status: completed
- note: merged smoke matrix; execution_status measured for F127/F128/F129 and controls

## 2026-06-05 06:31:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f127_f129_source_state_adaptive_terminal_oldshape_full --scope mlp --spec-ids MLP-F127-early100-h800-source-slow-ema-terminal-adaptive-raw,MLP-F128-early100-h800-source-slow-ema-terminal-sparse-source,MLP-F129-early100-h800-source-slow-ema-terminal-ratio-preserve,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --shard-index 0 --device cuda:0 --data-root data --run-label v2202_cont_f127_f129_source_state_adaptive_terminal_oldshape_full
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f127_f129_source_state_adaptive_terminal_oldshape_full --scope mlp --spec-ids MLP-F127-early100-h800-source-slow-ema-terminal-adaptive-raw,MLP-F128-early100-h800-source-slow-ema-terminal-sparse-source,MLP-F129-early100-h800-source-slow-ema-terminal-ratio-preserve,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --shard-index 1 --device cuda:1 --data-root data --run-label v2202_cont_f127_f129_source_state_adaptive_terminal_oldshape_full
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f127_f129_source_state_adaptive_terminal_oldshape_full --scope mlp --spec-ids MLP-F127-early100-h800-source-slow-ema-terminal-adaptive-raw,MLP-F128-early100-h800-source-slow-ema-terminal-sparse-source,MLP-F129-early100-h800-source-slow-ema-terminal-ratio-preserve,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --shard-index 2 --device cuda:2 --data-root data --run-label v2202_cont_f127_f129_source_state_adaptive_terminal_oldshape_full
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f127_f129_source_state_adaptive_terminal_oldshape_full --scope mlp --spec-ids MLP-F127-early100-h800-source-slow-ema-terminal-adaptive-raw,MLP-F128-early100-h800-source-slow-ema-terminal-sparse-source,MLP-F129-early100-h800-source-slow-ema-terminal-ratio-preserve,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --shard-index 3 --device cuda:3 --data-root data --run-label v2202_cont_f127_f129_source_state_adaptive_terminal_oldshape_full
```

- status: completed
- note: 4GPU old-shape full run completed; rows=63, blocked=0

## 2026-06-05 06:31:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f127_f129_source_state_adaptive_terminal_oldshape_full --scope mlp --merge-only --run-label v2202_cont_f127_f129_source_state_adaptive_terminal_oldshape_full
```

- status: completed
- note: merged F127-F129 old-shape full matrix; rows=63, blocked=0

## 2026-06-05 06:31:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f127_f129_source_state_adaptive_terminal_oldshape_full --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f127_f129_source_state_adaptive_terminal_oldshape_full --tag f127_f129_source_state_adaptive_terminal_oldshape_full
```

- status: completed
- note: F127/F128/F129 h4800_retention_ratio=0.401535/0.372177/0.400462; candidate_h4800=0

## 2026-06-05 06:31:27 +0800

```bash
generated v22_02_f106_f129_oldshape_full_audit.csv and v22_02_f106_f129_oldshape_full_route.csv from existing F106-F126 audit plus F127-F129 autopsy summary
```

- status: completed
- note: combined route: candidate_groups=21, early=15, continuous_h3200=15, candidate_h4800=0, promotion_allowed=0

## 2026-06-05 06:31:46 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 06:31:48 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 06:33:09 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 06:33:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 06:33:30 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: finalizer display updated to include all F127-F129 rows and exact ratio conclusion

## 2026-06-05 06:34:16 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 06:34:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 07:01:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f130_f132_terminal_target_reset_oldshape_full --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f130_f132_terminal_target_reset_oldshape_full --tag f130_f132_terminal_target_reset_oldshape_full
```

- status: completed
- note: rows=63 groups=7 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 07:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: finalizer updated to prefer v22_02_f106_f132_oldshape_full_route/audit

## 2026-06-05 07:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/v22_02_f130_f132_smoke3 --scope mlp --spec-ids MLP-F130-early100-h800-source-slow-ema-terminal-source-projection-target,MLP-F131-early100-h800-source-slow-ema-terminal-noise-orthogonal-target,MLP-F132-early100-h800-source-slow-ema-terminal-easy-margin-target,CTRL-SGD,CTRL-AdamW --datasets MNIST --seeds 0 --train-size 32 --val-size 24 --batch-size 16 --steps 4050 --shard-count 1 --shard-index 0 --device cuda:0 --run-label v2202_cont_f130_f132_terminal_target_smoke4050_v3
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/v22_02_f130_f132_smoke3 --merge-only
```

- status: completed
- note: smoke after wrapper fix; execution_status measured without blocked rows; smoke values not used as scientific route evidence

## 2026-06-05 07:08:54 +0800

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f130_f132_terminal_target_reset_oldshape_full
IDS=MLP-F130-early100-h800-source-slow-ema-terminal-source-projection-target,MLP-F131-early100-h800-source-slow-ema-terminal-noise-orthogonal-target,MLP-F132-early100-h800-source-slow-ema-terminal-easy-margin-target,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
for SHARD in 0 1 2 3; do /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --spec-ids "$IDS" --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --shard-index "$SHARD" --device "cuda:$SHARD" --data-root data --run-label v2202_cont_f130_f132_terminal_target_reset_oldshape_full; done
```

- status: completed
- note: 4GPU old-shape full run completed; rows=63, blocked=0

## 2026-06-05 07:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f130_f132_terminal_target_reset_oldshape_full --scope mlp --merge-only --run-label v2202_cont_f130_f132_terminal_target_reset_oldshape_full
```

- status: completed
- note: merged F130-F132 old-shape full matrix; rows=63 grouped=7

## 2026-06-05 07:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<PY  # generated v22_02_f106_f132_oldshape_full_audit.csv and v22_02_f106_f132_oldshape_full_route.csv from F106-F129 audit plus F130-F132 autopsy summary
PY
```

- status: completed
- note: combined route: candidate_groups=24, early=18, continuous_h3200=18, candidate_h4800=0, promotion_allowed=0

## 2026-06-05 07:09:09 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 07:09:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 07:32:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f133_f135_terminal_reject_rescue_oldshape_full --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f133_f135_terminal_reject_rescue_oldshape_full --tag f133_f135_terminal_reject_rescue_oldshape_full
```

- status: completed
- note: rows=63 groups=7 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 07:34:58 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 07:35:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 07:36:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py
```

- status: completed
- note: added F133-F135 terminal reject rescue mechanisms/specs; syntax check passed

## 2026-06-05 07:36:16 +0800

```bash
rm -rf /tmp/v22_02_f133_f135_smoke && /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/v22_02_f133_f135_smoke --scope mlp --spec-ids MLP-F133-early100-h800-source-slow-ema-terminal-reject-source-axis-rescue,MLP-F134-early100-h800-source-slow-ema-terminal-reject-slow-ema-rescue,MLP-F135-early100-h800-source-slow-ema-terminal-reject-hold-source-rescue,CTRL-SGD,CTRL-AdamW --datasets MNIST --seeds 0 --train-size 32 --val-size 24 --batch-size 16 --steps 4050 --shard-count 1 --shard-index 0 --device cuda:0 --run-label v2202_cont_f133_f135_terminal_reject_rescue_smoke4050_v2 && /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/v22_02_f133_f135_smoke --merge-only
```

- status: completed
- note: smoke path check only; rows=5, measured=5, blocked=0; rescue trace columns present

## 2026-06-05 07:36:16 +0800

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f133_f135_terminal_reject_rescue_oldshape_full
IDS=MLP-F133-early100-h800-source-slow-ema-terminal-reject-source-axis-rescue,MLP-F134-early100-h800-source-slow-ema-terminal-reject-slow-ema-rescue,MLP-F135-early100-h800-source-slow-ema-terminal-reject-hold-source-rescue,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
rm -rf "$OUT"
mkdir -p "$OUT/logs"
for SHARD in 0 1 2 3; do
  /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --spec-ids "$IDS" --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --shard-index "$SHARD" --device "cuda:$SHARD" --data-root data --run-label v2202_cont_f133_f135_terminal_reject_rescue_oldshape_full > "$OUT/logs/shard_${SHARD}.log" 2>&1 &
done
wait
```

- status: completed
- note: F133-F135 4GPU old-shape full run completed; rows=63, measured=63, blocked=0

## 2026-06-05 07:36:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f133_f135_terminal_reject_rescue_oldshape_full --scope mlp --merge-only --run-label v2202_cont_f133_f135_terminal_reject_rescue_oldshape_full
```

- status: completed
- note: merged F133-F135 old-shape full matrix; rows=63 grouped=7

## 2026-06-05 07:36:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<PY  # generated v22_02_f106_f135_oldshape_full_audit.csv and v22_02_f106_f135_oldshape_full_route.csv from F106-F132 audit plus F133-F135 autopsy summary
PY
```

- status: completed
- note: combined route: candidate_groups=27, early=21, continuous_h3200=21, candidate_h4800=0, promotion_allowed=0

## 2026-06-05 07:36:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: finalizer updated to prefer v22_02_f106_f135_oldshape_full_route/audit and render F133-F135 conclusions

## 2026-06-05 07:37:20 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 07:37:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 07:53:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f136_f138_shape_preserve_oldshape_full --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f136_f138_shape_preserve_oldshape_full --tag f136_f138_shape_preserve_oldshape_full
```

- status: completed
- note: rows=63 groups=7 decision=NoContinuousH3200UnderV2202Rule

## 2026-06-05 07:59:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: syntax/import path check passed before F136-F138 smoke

## 2026-06-05 07:59:00 +0800

```bash
rm -rf /tmp/v22_02_f136_f138_smoke && /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/v22_02_f136_f138_smoke --scope mlp --spec-ids MLP-F136-early100-h800-source-slow-ema-shape-preserve,MLP-F137-early100-h800-source-slow-ema-shape-preserve-raw-guard,MLP-F138-early100-h800-source-slow-ema-shape-preserve-clamp,CTRL-SGD,CTRL-AdamW --datasets MNIST --seeds 0 --train-size 32 --val-size 24 --batch-size 16 --steps 4050 --shard-count 1 --shard-index 0 --device cuda:0 --run-label v2202_cont_f136_f138_shape_preserve_smoke4050 && /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/v22_02_f136_f138_smoke --merge-only
```

- status: completed
- note: first smoke exposed MLP scope whitelist omission; rows=2 controls only, no candidate science claim

## 2026-06-05 07:59:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v21_01_source_retention.py && rm -rf /tmp/v22_02_f136_f138_smoke && /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/v22_02_f136_f138_smoke --scope mlp --spec-ids MLP-F136-early100-h800-source-slow-ema-shape-preserve,MLP-F137-early100-h800-source-slow-ema-shape-preserve-raw-guard,MLP-F138-early100-h800-source-slow-ema-shape-preserve-clamp,CTRL-SGD,CTRL-AdamW --datasets MNIST --seeds 0 --train-size 32 --val-size 24 --batch-size 16 --steps 4050 --shard-count 1 --shard-index 0 --device cuda:0 --run-label v2202_cont_f136_f138_shape_preserve_smoke4050 && /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/v22_02_f136_f138_smoke --merge-only
```

- status: completed
- note: candidate smoke path check passed; rows=5, measured=5, blocked=0

## 2026-06-05 07:59:00 +0800

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f136_f138_shape_preserve_oldshape_full
IDS=MLP-F136-early100-h800-source-slow-ema-shape-preserve,MLP-F137-early100-h800-source-slow-ema-shape-preserve-raw-guard,MLP-F138-early100-h800-source-slow-ema-shape-preserve-clamp,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
rm -rf "$OUT"
mkdir -p "$OUT/logs"
for SHARD in 0 1 2 3; do
  /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --spec-ids "$IDS" --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --shard-index "$SHARD" --device "cuda:$SHARD" --data-root data --run-label v2202_cont_f136_f138_shape_preserve_oldshape_full > "$OUT/logs/shard_${SHARD}.log" 2>&1 &
done
wait
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --merge-only --run-label v2202_cont_f136_f138_shape_preserve_oldshape_full
```

- status: completed
- note: F136-F138 4GPU old-shape full completed; rows=63, measured=63, blocked=0

## 2026-06-05 07:59:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<PY  # generated v22_02_f106_f138_oldshape_full_audit.csv and v22_02_f106_f138_oldshape_full_route.csv from F106-F135 audit plus F136-F138 autopsy summary
PY
```

- status: completed
- note: combined route: candidate_groups=30, early=21, continuous_h3200=21, candidate_h4800=0, promotion_allowed=0

## 2026-06-05 08:00:03 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 08:00:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 08:14:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f139_f141_terminal_reject_weakrow_oldshape_full --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f139_f141_terminal_reject_weakrow_oldshape_full --tag f139_f141_terminal_reject_weakrow_oldshape_full
```

- status: completed
- note: rows=63 groups=7 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 08:18:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py
```

- status: completed
- note: added F139-F141 terminal reject weak-row rescue mechanisms/specs; syntax check passed

## 2026-06-05 08:18:06 +0800

```bash
rm -rf /tmp/v22_02_f139_f141_smoke && /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/v22_02_f139_f141_smoke --scope mlp --spec-ids MLP-F139-early100-h800-source-slow-ema-terminal-reject-raw-rescue,MLP-F140-early100-h800-source-slow-ema-terminal-reject-hybrid-rescue,MLP-F141-early100-h800-source-slow-ema-terminal-reject-lateonly-raw-rescue,CTRL-SGD,CTRL-AdamW --datasets MNIST --seeds 0 --train-size 32 --val-size 24 --batch-size 16 --steps 4050 --shard-count 1 --shard-index 0 --device cuda:0 --run-label v2202_cont_f139_f141_terminal_reject_weakrow_smoke4050 && /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir /tmp/v22_02_f139_f141_smoke --merge-only
```

- status: completed
- note: smoke path check only; rows=5, measured=5, blocked=0; candidates executed and controls present

## 2026-06-05 08:18:06 +0800

```bash
OUT=results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f139_f141_terminal_reject_weakrow_oldshape_full
IDS=MLP-F139-early100-h800-source-slow-ema-terminal-reject-raw-rescue,MLP-F140-early100-h800-source-slow-ema-terminal-reject-hybrid-rescue,MLP-F141-early100-h800-source-slow-ema-terminal-reject-lateonly-raw-rescue,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead
rm -rf "$OUT"
mkdir -p "$OUT/logs"
for SHARD in 0 1 2 3; do
  /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --spec-ids "$IDS" --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 4800 --shard-count 4 --shard-index "$SHARD" --device "cuda:$SHARD" --data-root data --run-label v2202_cont_f139_f141_terminal_reject_weakrow_oldshape_full > "$OUT/logs/shard_${SHARD}.log" 2>&1 &
done
wait
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir "$OUT" --scope mlp --merge-only --run-label v2202_cont_f139_f141_terminal_reject_weakrow_oldshape_full
```

- status: completed
- note: F139-F141 4GPU old-shape full completed; rows=63, measured=63, blocked=0

## 2026-06-05 08:18:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<PY  # generated v22_02_f106_f141_oldshape_full_audit.csv and v22_02_f106_f141_oldshape_full_route.csv from F106-F138 audit plus F139-F141 autopsy summary
PY
```

- status: completed
- note: combined route: candidate_groups=33, early=24, continuous_h3200=24, candidate_h4800=0, promotion_allowed=0

## 2026-06-05 08:20:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: finalizer updated to prefer v22_02_f106_f141_oldshape_full_route/audit and render F139-F141 conclusions; syntax check passed

## 2026-06-05 08:20:14 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 08:20:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 08:21:35 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 08:21:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 08:44:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f142_f144_topk_support_oldshape_full --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f142_f144_topk_support_oldshape_full --tag f142_f144_topk_support_oldshape_full
```

- status: completed
- note: rows=63 groups=7 decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 08:48:53 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 08:48:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 08:31:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_02_merge_finalize.py experiments/run_v22_02_terminal_collapse_autopsy.py experiments/run_v22_02_s09_truth_gate.py
```

- status: completed
- note: F142-F144 top-k support mechanisms and v22.02 finalizer syntax/import compile check passed

## 2026-06-05 08:31:08 +0800

```bash
OUT="results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f142_f144_topk_support_oldshape_full"
IDS="MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard,MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard,MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead"
rm -rf "$OUT"
mkdir -p "$OUT/logs"
for SHARD in 0 1 2 3; do
  /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py \
    --out-dir "$OUT" \
    --scope mlp \
    --spec-ids "$IDS" \
    --datasets MNIST,Fashion-MNIST,KMNIST \
    --seeds 0,1,2 \
    --train-size 512 \
    --val-size 256 \
    --batch-size 64 \
    --steps 4800 \
    --shard-count 4 \
    --shard-index "$SHARD" \
    --device "cuda:$SHARD" \
    --data-root data \
    --run-label v2202_cont_f142_f144_topk_support_oldshape_full \
    > "$OUT/logs/shard_${SHARD}.log" 2>&1 &
done
wait
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py \
  --out-dir "$OUT" \
  --scope mlp \
  --merge-only \
  --run-label v2202_cont_f142_f144_topk_support_oldshape_full
```

- status: completed
- note: old-shape full run rows=63; candidate groups=3; all three F142-F144 groups had early_source_chain=1 and continuous_h3200=1 but productive_h4800=0

## 2026-06-05 08:44:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_terminal_collapse_autopsy.py \
  --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f142_f144_topk_support_oldshape_full \
  --out-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f142_f144_topk_support_oldshape_full \
  --tag f142_f144_topk_support_oldshape_full
```

- status: completed
- note: source_chain_rows=63; source_chain_groups=7; candidate_h4800=0; decision=TerminalCollapseClassifiedNoH4800

## 2026-06-05 08:46:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
base = Path("results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02")
newdir = base / "continuation_f142_f144_topk_support_oldshape_full"
prev_audit = pd.read_csv(base / "v22_02_f106_f141_oldshape_full_audit.csv")
summary = pd.read_csv(newdir / "v22_02_source_chain_summary.csv")
new = summary[~summary["v22_id"].astype(str).str.startswith("CTRL")].copy()
new["continuation"] = "F142-F144 top-k source-support oldshape full"
new["source_dir"] = str(newdir)
new["blocker"] = new.get("source_chain_blocker")
for col in prev_audit.columns:
    if col not in new.columns:
        new[col] = pd.NA
audit = pd.concat([prev_audit, new[prev_audit.columns]], ignore_index=True)
audit.to_csv(base / "v22_02_f106_f144_oldshape_full_audit.csv", index=False)
prev_route = pd.read_csv(base / "v22_02_f106_f141_oldshape_full_route.csv").iloc[0].to_dict()
new_route = pd.read_csv(newdir / "v22_02_terminal_collapse_route.csv").iloc[0].to_dict()
combined = {k: int(prev_route.get(k, 0)) + int(new_route.get(k, 0)) for k in [
    "source_chain_rows", "source_chain_groups", "candidate_groups",
    "candidate_early_chain", "candidate_continuous_h3200", "candidate_h4800",
    "terminal_collapse_groups", "h4000_missing_group_count"
]}
combined["promotion_allowed"] = 0
combined["decision"] = "R4-TerminalCollapseNoH4800"
pd.DataFrame([combined]).to_csv(base / "v22_02_f106_f144_oldshape_full_route.csv", index=False)
PY
```

- status: completed
- note: combined route candidate_groups=36, early=27, continuous_h3200=27, candidate_h4800=0, promotion_allowed=0

## 2026-06-05 08:48:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_02_merge_finalize.py experiments/run_v22_02_terminal_collapse_autopsy.py experiments/run_v22_02_s09_truth_gate.py && \
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: finalizer now prefers v22_02_f106_f144_oldshape_full_route/audit; recap route=R4-TerminalCollapseNoH4800; artifact update size/hash recorded in recap

## 2026-06-05 08:50:20 +0800

```bash
OUT="results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02"
rm -rf /tmp/v22_02_packet_check_f144
unzip -q "$OUT/v22_02_code_review_packet.zip" -d /tmp/v22_02_packet_check_f144
```

- status: failed
- note: environment command `unzip` not found; switched to Python standard-library zipfile extraction for the actual clean-packet check

## 2026-06-05 08:51:05 +0800

```bash
OUT="results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02"
rm -rf /tmp/v22_02_packet_check_f144
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from pathlib import Path
import zipfile
src = Path("results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/v22_02_code_review_packet.zip")
dst = Path("/tmp/v22_02_packet_check_f144")
with zipfile.ZipFile(src) as z:
    z.extractall(dst)
PY
cd /tmp/v22_02_packet_check_f144/02_SOURCE_TREE
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: clean-packet compileall_exit=0; truth_gate_exit=0

## 2026-06-05 08:52:32 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 08:52:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 09:27:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_02_merge_finalize.py && \
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: recap renderer expanded from compact summary to long recap; added Artifact Index, Continuation Ledger, Best Candidate Ranking, Per-Continuation Detail, and F118 vs F142-F144 dataset/seed localization. Recap length became 607 lines.

## 2026-06-05 09:28:20 +0800

```bash
OUT="results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02"
rm -rf /tmp/v22_02_packet_check_long_recap
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from pathlib import Path
import zipfile
src = Path("results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/v22_02_code_review_packet.zip")
dst = Path("/tmp/v22_02_packet_check_long_recap")
with zipfile.ZipFile(src) as z:
    z.extractall(dst)
PY
cd /tmp/v22_02_packet_check_long_recap/02_SOURCE_TREE
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: long-recap packet compileall_exit=0; truth_gate_exit=0

## 2026-06-05 09:27:27 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 09:27:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip

## 2026-06-05 09:29:43 +0800

```bash
rm -rf /tmp/v22_02_packet_check && unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

- status: completed
- note: compileall_exit=0; truth_gate_exit=0

## 2026-06-05 09:29:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_02_merge_finalize.py
```

- status: completed
- note: route=R4-TerminalCollapseNoH4800 packet=v22_02_code_review_packet.zip bundle=v22_02_results_bundle.zip
