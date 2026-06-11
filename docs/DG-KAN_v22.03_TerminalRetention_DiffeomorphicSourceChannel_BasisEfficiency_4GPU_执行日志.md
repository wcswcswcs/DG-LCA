# DG-KAN v22.03 TerminalRetention DiffeomorphicSourceChannel BasisEfficiency 4GPU 执行日志

生成时间：2026-06-05 16:58:41 +0800

记录原则：只记录真实执行过的命令、文件、状态和 blocker；不把未执行内容写成结果。

## 2026-06-05 16:58:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f145_f147_source_preserve_smoke_h1600 --scope mlp --spec-ids MLP-F145,MLP-F146,MLP-F147,controls --run-label v2203_cont_f145_f147_source_preserve_smoke_h1600 --datasets MNIST --seeds 0 --train-size 128 --val-size 64 --batch-size 32 --steps 1600 --shard-count 1 --device cuda:0
```

- status: completed
- note: smoke syntax/runtime check for F145-F147; merge follows

## 2026-06-05 16:58:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir .../continuation_f145_f147_source_preserve_smoke_h1600 --scope mlp --merge-only
```

- status: completed
- note: merged smoke source-retention matrix

## 2026-06-05 17:00:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f145_f147_source_preserve_oldshape_full --scope mlp --spec-ids MLP-F145,MLP-F146,MLP-F147,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f145_f147_source_preserve_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: started
- note: v22.03 C1 F145-F147 source-preservation old-shape full shard

## 2026-06-05 17:00:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f145_f147_source_preserve_oldshape_full --scope mlp --spec-ids MLP-F145,MLP-F146,MLP-F147,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f145_f147_source_preserve_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: started
- note: v22.03 C1 F145-F147 source-preservation old-shape full shard

## 2026-06-05 17:00:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f145_f147_source_preserve_oldshape_full --scope mlp --spec-ids MLP-F145,MLP-F146,MLP-F147,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f145_f147_source_preserve_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: started
- note: v22.03 C1 F145-F147 source-preservation old-shape full shard

## 2026-06-05 17:00:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f145_f147_source_preserve_oldshape_full --scope mlp --spec-ids MLP-F145,MLP-F146,MLP-F147,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f145_f147_source_preserve_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: started
- note: v22.03 C1 F145-F147 source-preservation old-shape full shard

## 2026-06-05 17:09:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f145_f147_source_preserve_oldshape_full --scope mlp --spec-ids MLP-F145,MLP-F146,MLP-F147,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f145_f147_source_preserve_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: completed
- note: v22.03 C1 shard completed with process exit 0; source_retention_rows=16

## 2026-06-05 17:09:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f145_f147_source_preserve_oldshape_full --scope mlp --spec-ids MLP-F145,MLP-F146,MLP-F147,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f145_f147_source_preserve_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: completed
- note: v22.03 C1 shard completed with process exit 0; source_retention_rows=16

## 2026-06-05 17:09:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f145_f147_source_preserve_oldshape_full --scope mlp --spec-ids MLP-F145,MLP-F146,MLP-F147,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f145_f147_source_preserve_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: completed
- note: v22.03 C1 shard completed with process exit 0; source_retention_rows=16

## 2026-06-05 17:09:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f145_f147_source_preserve_oldshape_full --scope mlp --spec-ids MLP-F145,MLP-F146,MLP-F147,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f145_f147_source_preserve_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: completed
- note: v22.03 C1 shard completed with process exit 0; source_retention_rows=15

## 2026-06-05 17:09:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f145_f147_source_preserve_oldshape_full --scope mlp --merge-only
```

- status: completed
- note: merged 4GPU F145-F147 old-shape full matrix and trace artifacts

## 2026-06-05 17:10:53 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py dgkan/fu/mechanisms.py dgkan/fu/terminal_retention.py dgkan/fu/diffeomorphic_target.py experiments/run_v22_03_common.py experiments/run_v22_03_source_preservation.py experiments/run_v22_03_terminal_erosion_autopsy.py experiments/run_v22_03_code_truth_gate.py experiments/run_v22_03_efficiency_full_loop.py experiments/run_v22_03_drat_drbf_repair.py experiments/run_v22_03_kan_source_channel_writer.py experiments/run_v22_03_finalize.py
```

- status: completed
- note: manual syntax check before S0.10 truth gate; exit 0

## 2026-06-05 17:10:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 17:11:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_efficiency_full_loop.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02
```

- status: completed
- note: rows=2

## 2026-06-05 17:11:38 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_drat_drbf_repair.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02
```

- status: completed
- note: readback only; no Near-E1 promotion claimed

## 2026-06-05 17:11:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_kan_source_channel_writer.py --source-dir results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02
```

- status: completed
- note: rows=12 readback only

## 2026-06-05 17:12:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_terminal_erosion_autopsy.py --baseline-audit results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/v22_02_f106_f144_oldshape_full_audit.csv
```

- status: completed
- note: groups=27 decision=TerminalErosionExplained

## 2026-06-05 17:12:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f145_f147_source_preserve_oldshape_full --tag f145_f147_source_preserve
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 17:28:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py dgkan/fu/mechanisms.py
```

- status: completed
- note: post-C2 patch syntax check; exit 0

## 2026-06-05 17:28:37 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_smoke_h1600 --scope mlp --spec-ids MLP-F148,MLP-F149,MLP-F150,controls --run-label v2203_cont_f148_f150_diffeomorphic_target_smoke_h1600 --datasets MNIST --seeds 0 --train-size 128 --val-size 64 --batch-size 32 --steps 1600 --shard-count 1 --device cuda:0
```

- status: completed
- note: C2 F148-F150 smoke syntax/runtime check; not promotion evidence

## 2026-06-05 17:29:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_smoke_h1600 --scope mlp --merge-only
```

- status: completed
- note: merged C2 F148-F150 smoke matrix and trace artifacts

## 2026-06-05 17:30:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_smoke_h4200 --scope mlp --spec-ids MLP-F148,MLP-F149,MLP-F150,controls --run-label v2203_cont_f148_f150_diffeomorphic_target_smoke_h4200 --datasets MNIST --seeds 0 --train-size 128 --val-size 64 --batch-size 32 --steps 4200 --shard-count 1 --device cuda:0
```

- status: completed
- note: C2 F148-F150 h4200 smoke to trigger terminal target gate; not promotion evidence

## 2026-06-05 17:31:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_smoke_h4200 --scope mlp --merge-only
```

- status: completed
- note: merged C2 F148-F150 h4200 smoke matrix and trace artifacts

## 2026-06-05 17:31:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_oldshape_full --scope mlp --spec-ids MLP-F148,MLP-F149,MLP-F150,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f148_f150_diffeomorphic_target_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: started
- note: v22.03 C2 F148-F150 diffeomorphic target old-shape full shard

## 2026-06-05 17:31:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_oldshape_full --scope mlp --spec-ids MLP-F148,MLP-F149,MLP-F150,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f148_f150_diffeomorphic_target_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: started
- note: v22.03 C2 F148-F150 diffeomorphic target old-shape full shard

## 2026-06-05 17:31:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_oldshape_full --scope mlp --spec-ids MLP-F148,MLP-F149,MLP-F150,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f148_f150_diffeomorphic_target_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: started
- note: v22.03 C2 F148-F150 diffeomorphic target old-shape full shard

## 2026-06-05 17:31:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_oldshape_full --scope mlp --spec-ids MLP-F148,MLP-F149,MLP-F150,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f148_f150_diffeomorphic_target_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: started
- note: v22.03 C2 F148-F150 diffeomorphic target old-shape full shard

## 2026-06-05 17:35:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_oldshape_full --scope mlp --spec-ids MLP-F148,MLP-F149,MLP-F150,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f148_f150_diffeomorphic_target_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: completed
- note: v22.03 C2 shard completed with process exit 0; source_retention_rows=16

## 2026-06-05 17:35:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_oldshape_full --scope mlp --spec-ids MLP-F148,MLP-F149,MLP-F150,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f148_f150_diffeomorphic_target_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: completed
- note: v22.03 C2 shard completed with process exit 0; source_retention_rows=16

## 2026-06-05 17:35:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_oldshape_full --scope mlp --spec-ids MLP-F148,MLP-F149,MLP-F150,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f148_f150_diffeomorphic_target_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: completed
- note: v22.03 C2 shard completed with process exit 0; source_retention_rows=16

## 2026-06-05 17:35:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_oldshape_full --scope mlp --spec-ids MLP-F148,MLP-F149,MLP-F150,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f148_f150_diffeomorphic_target_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: completed
- note: v22.03 C2 shard completed with process exit 0; source_retention_rows=15

## 2026-06-05 17:36:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_oldshape_full --scope mlp --merge-only
```

- status: completed
- note: merged C2 F148-F150 old-shape full matrix and trace artifacts

## 2026-06-05 17:36:51 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_oldshape_full --tag f148_f150_diffeomorphic_target
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 17:44:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py dgkan/fu/mechanisms.py dgkan/fu/terminal_retention.py dgkan/fu/diffeomorphic_target.py experiments/run_v22_03_common.py experiments/run_v22_03_source_preservation.py experiments/run_v22_03_terminal_erosion_autopsy.py experiments/run_v22_03_code_truth_gate.py experiments/run_v22_03_efficiency_full_loop.py experiments/run_v22_03_drat_drbf_repair.py experiments/run_v22_03_kan_source_channel_writer.py experiments/run_v22_03_finalize.py
```

- status: completed
- note: post-C2 finalizer/code-gate py_compile passed

## 2026-06-05 17:44:57 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=0

## 2026-06-05 17:47:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_03_code_truth_gate.py experiments/run_v22_03_finalize.py
```

- status: completed
- note: fixed v22.03 M170/M171 expected mechanism names and finalizer wording

## 2026-06-05 17:47:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 17:48:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 17:48:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 17:49:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_03_finalize.py
```

- status: completed
- note: corrected C2 target-gate analysis wording: F149 has 1/54 accepted target attempt

## 2026-06-05 17:49:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 17:49:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 17:51:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_03_finalize.py
```

- status: completed
- note: finalizer now appends its command before final packet rebuild and recap

## 2026-06-05 17:51:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 17:51:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 18:05:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 18:07:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f151_f153_terminal_preserve_repair_smoke --scope mlp --spec-ids MLP-F151-early100-h800-source-slow-ema-terminal-source-preserve-very-strong,MLP-F152-early100-h800-source-slow-ema-h3600-terminal-source-preserve,MLP-F153-early100-h800-source-slow-ema-terminal-nora-row-orthogonal-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f151_f153_terminal_preserve_repair_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 4200 --device cuda:0
```

- status: completed
- note: F151-F153 smoke rows: MNIST seed0 steps=4200

## 2026-06-05 18:08:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f151_f153_terminal_preserve_repair_smoke --merge-only
```

- status: completed
- note: merged F151-F153 smoke shard outputs

## 2026-06-05 18:12:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f151_f153_terminal_preserve_repair_oldshape_full --scope mlp --spec-ids MLP-F151-early100-h800-source-slow-ema-terminal-source-preserve-very-strong,MLP-F152-early100-h800-source-slow-ema-h3600-terminal-source-preserve,MLP-F153-early100-h800-source-slow-ema-terminal-nora-row-orthogonal-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f151_f153_terminal_preserve_repair_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: completed
- note: F151-F153 old-shape full shard 0/4 completed

## 2026-06-05 18:12:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f151_f153_terminal_preserve_repair_oldshape_full --scope mlp --spec-ids MLP-F151-early100-h800-source-slow-ema-terminal-source-preserve-very-strong,MLP-F152-early100-h800-source-slow-ema-h3600-terminal-source-preserve,MLP-F153-early100-h800-source-slow-ema-terminal-nora-row-orthogonal-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f151_f153_terminal_preserve_repair_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: completed
- note: F151-F153 old-shape full shard 1/4 completed

## 2026-06-05 18:12:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f151_f153_terminal_preserve_repair_oldshape_full --scope mlp --spec-ids MLP-F151-early100-h800-source-slow-ema-terminal-source-preserve-very-strong,MLP-F152-early100-h800-source-slow-ema-h3600-terminal-source-preserve,MLP-F153-early100-h800-source-slow-ema-terminal-nora-row-orthogonal-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f151_f153_terminal_preserve_repair_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: completed
- note: F151-F153 old-shape full shard 2/4 completed

## 2026-06-05 18:12:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f151_f153_terminal_preserve_repair_oldshape_full --scope mlp --spec-ids MLP-F151-early100-h800-source-slow-ema-terminal-source-preserve-very-strong,MLP-F152-early100-h800-source-slow-ema-h3600-terminal-source-preserve,MLP-F153-early100-h800-source-slow-ema-terminal-nora-row-orthogonal-source,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f151_f153_terminal_preserve_repair_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: completed
- note: F151-F153 old-shape full shard 3/4 completed

## 2026-06-05 18:13:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f151_f153_terminal_preserve_repair_oldshape_full --merge-only
```

- status: completed
- note: merged F151-F153 old-shape full shard outputs

## 2026-06-05 18:13:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f151_f153_terminal_preserve_repair_oldshape_full --tag f151_f153_terminal_preserve_repair
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 18:21:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 18:22:05 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f154_f156_terminal_debt_lownds_dualmem_smoke --scope mlp --spec-ids MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve,MLP-F155-early100-h800-source-slow-ema-terminal-low-nds-matrix-block,MLP-F156-early100-h800-source-slow-ema-terminal-dual-memory-preserve,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f154_f156_terminal_debt_lownds_dualmem_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 4200 --device cuda:0
```

- status: completed
- note: F154-F156 smoke rows: MNIST seed0 steps=4200

## 2026-06-05 18:22:06 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f154_f156_terminal_debt_lownds_dualmem_smoke --merge-only
```

- status: completed
- note: merged F154-F156 smoke shard outputs

## 2026-06-05 18:24:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f154_f156_terminal_debt_lownds_dualmem_smoke_all --scope mlp --spec-ids MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve,MLP-F155-early100-h800-source-slow-ema-terminal-low-nds-matrix-block,MLP-F156-early100-h800-source-slow-ema-terminal-dual-memory-preserve,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f154_f156_terminal_debt_lownds_dualmem_smoke_all --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 4200 --shard-count 1 --shard-index 0 --device cuda:0
```

- status: completed
- note: F154-F156 all-spec smoke completed

## 2026-06-05 18:29:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f154_f156_terminal_debt_lownds_dualmem_oldshape_full --scope mlp --spec-ids MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve,MLP-F155-early100-h800-source-slow-ema-terminal-low-nds-matrix-block,MLP-F156-early100-h800-source-slow-ema-terminal-dual-memory-preserve,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f154_f156_terminal_debt_lownds_dualmem_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: completed
- note: F154-F156 old-shape full shard 0/4 completed

## 2026-06-05 18:29:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f154_f156_terminal_debt_lownds_dualmem_oldshape_full --scope mlp --spec-ids MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve,MLP-F155-early100-h800-source-slow-ema-terminal-low-nds-matrix-block,MLP-F156-early100-h800-source-slow-ema-terminal-dual-memory-preserve,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f154_f156_terminal_debt_lownds_dualmem_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: completed
- note: F154-F156 old-shape full shard 1/4 completed

## 2026-06-05 18:29:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f154_f156_terminal_debt_lownds_dualmem_oldshape_full --scope mlp --spec-ids MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve,MLP-F155-early100-h800-source-slow-ema-terminal-low-nds-matrix-block,MLP-F156-early100-h800-source-slow-ema-terminal-dual-memory-preserve,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f154_f156_terminal_debt_lownds_dualmem_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: completed
- note: F154-F156 old-shape full shard 2/4 completed

## 2026-06-05 18:29:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f154_f156_terminal_debt_lownds_dualmem_oldshape_full --scope mlp --spec-ids MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve,MLP-F155-early100-h800-source-slow-ema-terminal-low-nds-matrix-block,MLP-F156-early100-h800-source-slow-ema-terminal-dual-memory-preserve,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f154_f156_terminal_debt_lownds_dualmem_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: completed
- note: F154-F156 old-shape full shard 3/4 completed

## 2026-06-05 18:29:32 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f154_f156_terminal_debt_lownds_dualmem_oldshape_full --merge-only
```

- status: completed
- note: merged F154-F156 old-shape full shards into source retention matrices

## 2026-06-05 18:29:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f154_f156_terminal_debt_lownds_dualmem_oldshape_full --tag f154_f156_terminal_debt_lownds_dualmem
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 18:30:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 18:30:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 18:46:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 18:47:12 +0800

```bash
python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_03_code_truth_gate.py experiments/run_v22_03_finalize.py
```

- status: completed
- note: local syntax check after adding F157-F159

## 2026-06-05 18:48:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f157_f159_signal_estimator_smoke --scope mlp --spec-ids MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor,MLP-F158-early100-h800-source-slow-ema-split-consensus-estimator,MLP-F159-early100-h800-source-slow-ema-signal-reservoir-transport,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f157_f159_signal_estimator_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 4200 --shard-count 1 --shard-index 0 --device cuda:0
```

- status: completed
- note: F157-F159 smoke completed; implementation reached terminal estimator branch

## 2026-06-05 18:49:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f157_f159_signal_estimator_smoke --merge-only
```

- status: completed
- note: merged F157-F159 smoke shards

## 2026-06-05 18:51:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f157_f159_signal_estimator_oldshape_full --scope mlp --spec-ids MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor,MLP-F158-early100-h800-source-slow-ema-split-consensus-estimator,MLP-F159-early100-h800-source-slow-ema-signal-reservoir-transport,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f157_f159_signal_estimator_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: started
- note: F157-F159 old-shape full shard 0/4 started

## 2026-06-05 18:51:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f157_f159_signal_estimator_oldshape_full --scope mlp --spec-ids MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor,MLP-F158-early100-h800-source-slow-ema-split-consensus-estimator,MLP-F159-early100-h800-source-slow-ema-signal-reservoir-transport,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f157_f159_signal_estimator_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: started
- note: F157-F159 old-shape full shard 1/4 started

## 2026-06-05 18:51:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f157_f159_signal_estimator_oldshape_full --scope mlp --spec-ids MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor,MLP-F158-early100-h800-source-slow-ema-split-consensus-estimator,MLP-F159-early100-h800-source-slow-ema-signal-reservoir-transport,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f157_f159_signal_estimator_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: started
- note: F157-F159 old-shape full shard 2/4 started

## 2026-06-05 18:51:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f157_f159_signal_estimator_oldshape_full --scope mlp --spec-ids MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor,MLP-F158-early100-h800-source-slow-ema-split-consensus-estimator,MLP-F159-early100-h800-source-slow-ema-signal-reservoir-transport,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f157_f159_signal_estimator_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: started
- note: F157-F159 old-shape full shard 3/4 started

## 2026-06-05 18:54:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f157_f159_signal_estimator_oldshape_full --scope mlp --spec-ids MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor,MLP-F158-early100-h800-source-slow-ema-split-consensus-estimator,MLP-F159-early100-h800-source-slow-ema-signal-reservoir-transport,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f157_f159_signal_estimator_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: completed
- note: F157-F159 old-shape full shard 0/4 completed

## 2026-06-05 18:54:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f157_f159_signal_estimator_oldshape_full --scope mlp --spec-ids MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor,MLP-F158-early100-h800-source-slow-ema-split-consensus-estimator,MLP-F159-early100-h800-source-slow-ema-signal-reservoir-transport,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f157_f159_signal_estimator_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: completed
- note: F157-F159 old-shape full shard 1/4 completed

## 2026-06-05 18:54:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f157_f159_signal_estimator_oldshape_full --scope mlp --spec-ids MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor,MLP-F158-early100-h800-source-slow-ema-split-consensus-estimator,MLP-F159-early100-h800-source-slow-ema-signal-reservoir-transport,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f157_f159_signal_estimator_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: completed
- note: F157-F159 old-shape full shard 2/4 completed

## 2026-06-05 18:54:19 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f157_f159_signal_estimator_oldshape_full --scope mlp --spec-ids MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor,MLP-F158-early100-h800-source-slow-ema-split-consensus-estimator,MLP-F159-early100-h800-source-slow-ema-signal-reservoir-transport,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f157_f159_signal_estimator_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: completed
- note: F157-F159 old-shape full shard 3/4 completed

## 2026-06-05 18:54:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f157_f159_signal_estimator_oldshape_full --merge-only
```

- status: completed
- note: merged F157-F159 old-shape full shards into source retention matrices

## 2026-06-05 18:55:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f157_f159_signal_estimator_oldshape_full --tag f157_f159_signal_estimator
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 18:56:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 18:56:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 19:09:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 19:16:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f160_f162_post_h4000_source_floor_oldshape_full --tag f160_f162_post_h4000_source_floor
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 19:17:22 +0800

```bash
python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_03_code_truth_gate.py experiments/run_v22_03_finalize.py
```

- status: completed
- note: local syntax check after adding F160-F162

## 2026-06-05 19:17:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f160_f162_post_h4000_source_floor_smoke --scope mlp --spec-ids MLP-F160-early100-h800-source-slow-ema-terminal-source-floor,MLP-F161-early100-h800-source-slow-ema-terminal-h4000-anchor-floor,MLP-F162-early100-h800-source-slow-ema-terminal-decay-aware-floor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f160_f162_post_h4000_source_floor_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 4200 --shard-count 1 --shard-index 0 --device cuda:0
```

- status: completed
- note: F160-F162 smoke completed; source-floor helper reached anchor=1 path

## 2026-06-05 19:17:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f160_f162_post_h4000_source_floor_smoke --merge-only
```

- status: completed
- note: merged F160-F162 smoke shards

## 2026-06-05 19:17:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f160_f162_post_h4000_source_floor_oldshape_full --scope mlp --spec-ids MLP-F160-early100-h800-source-slow-ema-terminal-source-floor,MLP-F161-early100-h800-source-slow-ema-terminal-h4000-anchor-floor,MLP-F162-early100-h800-source-slow-ema-terminal-decay-aware-floor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f160_f162_post_h4000_source_floor_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: completed
- note: F160-F162 old-shape full shard 0/4 completed

## 2026-06-05 19:17:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f160_f162_post_h4000_source_floor_oldshape_full --scope mlp --spec-ids MLP-F160-early100-h800-source-slow-ema-terminal-source-floor,MLP-F161-early100-h800-source-slow-ema-terminal-h4000-anchor-floor,MLP-F162-early100-h800-source-slow-ema-terminal-decay-aware-floor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f160_f162_post_h4000_source_floor_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: completed
- note: F160-F162 old-shape full shard 1/4 completed

## 2026-06-05 19:17:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f160_f162_post_h4000_source_floor_oldshape_full --scope mlp --spec-ids MLP-F160-early100-h800-source-slow-ema-terminal-source-floor,MLP-F161-early100-h800-source-slow-ema-terminal-h4000-anchor-floor,MLP-F162-early100-h800-source-slow-ema-terminal-decay-aware-floor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f160_f162_post_h4000_source_floor_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: completed
- note: F160-F162 old-shape full shard 2/4 completed

## 2026-06-05 19:17:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f160_f162_post_h4000_source_floor_oldshape_full --scope mlp --spec-ids MLP-F160-early100-h800-source-slow-ema-terminal-source-floor,MLP-F161-early100-h800-source-slow-ema-terminal-h4000-anchor-floor,MLP-F162-early100-h800-source-slow-ema-terminal-decay-aware-floor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f160_f162_post_h4000_source_floor_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: completed
- note: F160-F162 old-shape full shard 3/4 completed

## 2026-06-05 19:17:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f160_f162_post_h4000_source_floor_oldshape_full --merge-only
```

- status: completed
- note: merged F160-F162 old-shape full shards into source retention matrices

## 2026-06-05 19:17:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 19:17:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 19:24:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 19:30:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f163_f165_raw_guard_post_h4000_floor_oldshape_full --tag f163_f165_raw_guard_post_h4000_floor
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 19:31:40 +0800

```bash
python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_03_code_truth_gate.py experiments/run_v22_03_finalize.py
```

- status: completed
- note: local syntax check after adding F163-F165

## 2026-06-05 19:31:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f163_f165_raw_guard_post_h4000_floor_smoke --scope mlp --spec-ids MLP-F163-early100-h800-source-slow-ema-terminal-raw-guard-source-floor,MLP-F164-early100-h800-source-slow-ema-terminal-raw-guard-h4000-anchor-floor,MLP-F165-early100-h800-source-slow-ema-terminal-raw-guard-decay-aware-floor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f163_f165_raw_guard_post_h4000_floor_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 4200 --shard-count 1 --shard-index 0 --device cuda:0
```

- status: completed
- note: F163-F165 smoke completed; raw-guard floor helper reached anchor=1 path

## 2026-06-05 19:31:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f163_f165_raw_guard_post_h4000_floor_smoke --merge-only
```

- status: completed
- note: merged F163-F165 smoke shards

## 2026-06-05 19:31:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f163_f165_raw_guard_post_h4000_floor_oldshape_full --scope mlp --spec-ids MLP-F163-early100-h800-source-slow-ema-terminal-raw-guard-source-floor,MLP-F164-early100-h800-source-slow-ema-terminal-raw-guard-h4000-anchor-floor,MLP-F165-early100-h800-source-slow-ema-terminal-raw-guard-decay-aware-floor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f163_f165_raw_guard_post_h4000_floor_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: completed
- note: F163-F165 old-shape full shard 0/4 completed

## 2026-06-05 19:31:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f163_f165_raw_guard_post_h4000_floor_oldshape_full --scope mlp --spec-ids MLP-F163-early100-h800-source-slow-ema-terminal-raw-guard-source-floor,MLP-F164-early100-h800-source-slow-ema-terminal-raw-guard-h4000-anchor-floor,MLP-F165-early100-h800-source-slow-ema-terminal-raw-guard-decay-aware-floor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f163_f165_raw_guard_post_h4000_floor_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: completed
- note: F163-F165 old-shape full shard 1/4 completed

## 2026-06-05 19:31:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f163_f165_raw_guard_post_h4000_floor_oldshape_full --scope mlp --spec-ids MLP-F163-early100-h800-source-slow-ema-terminal-raw-guard-source-floor,MLP-F164-early100-h800-source-slow-ema-terminal-raw-guard-h4000-anchor-floor,MLP-F165-early100-h800-source-slow-ema-terminal-raw-guard-decay-aware-floor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f163_f165_raw_guard_post_h4000_floor_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: completed
- note: F163-F165 old-shape full shard 2/4 completed

## 2026-06-05 19:31:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f163_f165_raw_guard_post_h4000_floor_oldshape_full --scope mlp --spec-ids MLP-F163-early100-h800-source-slow-ema-terminal-raw-guard-source-floor,MLP-F164-early100-h800-source-slow-ema-terminal-raw-guard-h4000-anchor-floor,MLP-F165-early100-h800-source-slow-ema-terminal-raw-guard-decay-aware-floor,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f163_f165_raw_guard_post_h4000_floor_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: completed
- note: F163-F165 old-shape full shard 3/4 completed

## 2026-06-05 19:31:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f163_f165_raw_guard_post_h4000_floor_oldshape_full --merge-only
```

- status: completed
- note: merged F163-F165 old-shape full shards into source retention matrices

## 2026-06-05 19:32:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 19:32:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 19:38:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_drat_drbf_repair.py --device cuda:0 --batch-sizes 512,2048 --hidden 128 --iters 20 --warmup 4
```

- status: completed
- note: active micro-kernel repair rows=28 device=cuda:0

## 2026-06-05 19:39:00 +0800

```bash
python -m py_compile experiments/run_v22_03_drat_drbf_repair.py
```

- status: completed
- note: syntax check after replacing readback-only D-RAT/D-RBF repair with active micro-kernel benchmark

## 2026-06-05 19:40:46 +0800

```bash
python -m py_compile experiments/run_v22_03_finalize.py experiments/run_v22_03_drat_drbf_repair.py
```

- status: completed
- note: syntax check after finalizer active-repair recap update

## 2026-06-05 19:41:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 19:41:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 19:42:56 +0800

```bash
python -m py_compile experiments/run_v22_03_drat_drbf_repair.py
```

- status: completed
- note: syntax check after adding --out-dir to active repair command journal entry

## 2026-06-05 19:42:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_drat_drbf_repair.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03 --device cuda:0 --batch-sizes 512,2048 --hidden 128 --iters 20 --warmup 4
```

- status: completed
- note: complete reproduction command for active micro-kernel repair run; artifact rows already generated at 2026-06-05 19:38:18 +0800

## 2026-06-05 19:43:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 19:43:16 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 19:47:32 +0800

```bash
python -m py_compile experiments/run_v22_03_drat_drbf_runner_integration.py
```

- status: completed
- note: syntax check for limited same-kernel D-RAT/D-RBF runner integration

## 2026-06-05 19:47:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_drat_drbf_runner_integration.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03 --device cuda:0 --batch-size 512 --hidden 128 --classes 10 --steps 24 --warmup 4 --rat-variant RAT22.03-R0-current-reference --rbf-variant RBF22.03-R4-exp-approx-trainpath
```

- status: completed
- note: limited same-kernel runner integration rows=2 device=cuda:0

## 2026-06-05 19:51:56 +0800

```bash
python -m py_compile experiments/run_v22_03_code_truth_gate.py experiments/run_v22_03_finalize.py experiments/run_v22_03_drat_drbf_runner_integration.py experiments/run_v22_03_drat_drbf_repair.py
```

- status: completed
- note: syntax check after runner integration finalizer/truth-gate update

## 2026-06-05 19:52:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 19:52:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 19:52:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 20:05:30 +0800

```bash
python -m py_compile experiments/run_v22_03_drat_drbf_limited_smoke_summary.py experiments/run_v22_03_code_truth_gate.py experiments/run_v22_03_finalize.py
```

- status: completed
- note: syntax check after adding D-RAT/D-RBF limited smoke summary artifacts

## 2026-06-05 20:05:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_drat_drbf_limited_smoke_summary.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03 --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_drat_drbf_limited_smoke
```

- status: completed
- note: decision=ExistingCarrierSmokeNoH800Source carrier_rows=14

## 2026-06-05 20:06:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_drat_drbf_limited_smoke --scope kan --carriers D-RAT,D-RBF --spec-ids KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,KSW9-h800-source-slow-ema-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_drat_drbf_limited_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 1600 --shard-count 1 --shard-index 0 --device cuda:0 --basis-repair-variant RAT22.03-R0-current-reference
```

- status: completed
- note: D-RAT/D-RBF existing-carrier limited smoke completed; rows=14 MNIST seed0 h1600

## 2026-06-05 20:06:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_drat_drbf_limited_smoke --scope kan --merge-only
```

- status: completed
- note: merged D-RAT/D-RBF existing-carrier limited smoke rows

## 2026-06-05 20:06:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 20:07:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 20:07:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 20:16:04 +0800

```bash
python -m py_compile experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v22_03_drat_drbf_limited_smoke_summary.py experiments/run_v22_03_finalize.py experiments/run_v22_03_code_truth_gate.py
```

- status: completed
- note: syntax check after adding C4 RAT-FU/RBF-FU smoke aliases and finalizer sections

## 2026-06-05 20:17:46 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_drat_drbf_limited_smoke_summary.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03 --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_drat_drbf_c4_smoke --tag drat_drbf_c4_smoke
```

- status: completed
- note: decision=ExistingCarrierSmokeNoH800Source carrier_rows=8

## 2026-06-05 20:19:31 +0800

```bash
python -m py_compile experiments/run_v21_01_source_retention.py experiments/run_v21_common.py experiments/run_v17_common.py experiments/run_v22_03_drat_drbf_limited_smoke_summary.py experiments/run_v22_03_finalize.py experiments/run_v22_03_code_truth_gate.py
```

- status: completed
- note: syntax check after adding C4 alias allowlist to source-retention runner

## 2026-06-05 20:21:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_drat_drbf_limited_smoke_summary.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03 --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_drat_drbf_c4_named_smoke --tag drat_drbf_c4_smoke
```

- status: completed
- note: decision=ExistingCarrierSmokeNoH800Source carrier_rows=14

## 2026-06-05 20:22:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_drat_drbf_c4_smoke --scope kan --carriers D-RAT --spec-ids RAT-FU1-readout-only-rational-source,RAT-FU2-denominator-safe-low-degree-source,RAT-FU3-numerator-only-source-writer,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_drat_c4_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 1600 --shard-count 1 --shard-index 0 --device cuda:0 --basis-repair-variant RAT22.03-R6-low-degree-rational-readout-only
```

- status: completed
- note: C4 alias smoke attempt before kan allowlist fix; candidate specs were filtered out, kept as implementation diagnostic

## 2026-06-05 20:22:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_drat_drbf_c4_smoke --scope kan --carriers D-RBF --spec-ids RBF-FU1-readout-only-local-support-source,RBF-FU2-active-center-low-k-source,RBF-FU3-compact-local-source-writer,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_rbf_c4_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 1600 --shard-count 1 --shard-index 0 --device cuda:0 --basis-repair-variant RBF22.03-R1-compact-local-k4-no-dense
```

- status: completed
- note: C4 alias smoke attempt before kan allowlist fix; candidate specs were filtered out, kept as implementation diagnostic

## 2026-06-05 20:22:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_drat_drbf_c4_smoke --scope kan --merge-only
```

- status: completed
- note: merged allowlist-miss C4 smoke diagnostic

## 2026-06-05 20:22:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_drat_drbf_c4_named_smoke --scope kan --carriers D-RAT --spec-ids RAT-FU1-readout-only-rational-source,RAT-FU2-denominator-safe-low-degree-source,RAT-FU3-numerator-only-source-writer,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_drat_c4_named_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 1600 --shard-count 1 --shard-index 0 --device cuda:0 --basis-repair-variant RAT22.03-R6-low-degree-rational-readout-only
```

- status: completed
- note: D-RAT C4 named FU smoke completed with 3 candidates plus controls

## 2026-06-05 20:22:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_drat_drbf_c4_named_smoke --scope kan --carriers D-RBF --spec-ids RBF-FU1-readout-only-local-support-source,RBF-FU2-active-center-low-k-source,RBF-FU3-compact-local-source-writer,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_rbf_c4_named_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 1600 --shard-count 1 --shard-index 0 --device cuda:0 --basis-repair-variant RBF22.03-R1-compact-local-k4-no-dense
```

- status: completed
- note: D-RBF C4 named FU smoke completed with 3 candidates plus controls

## 2026-06-05 20:22:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_drat_drbf_c4_named_smoke --scope kan --merge-only
```

- status: completed
- note: merged D-RAT/D-RBF C4 named FU smoke rows

## 2026-06-05 20:23:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 20:23:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 20:23:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 20:39:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f166_f168_anti_erosion_smoke --tag f166_f168_anti_erosion_smoke
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 20:44:29 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f166_f168_anti_erosion_oldshape_full --tag f166_f168_anti_erosion
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 21:00:12 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f169_f171_h3200_anchor_transport_smoke --tag f169_f171_h3200_anchor_transport_smoke
```

- status: completed
- note: candidate_groups=0 h4800=0

## 2026-06-05 21:04:55 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f169_f171_h3200_anchor_transport_oldshape_full --tag f169_f171_h3200_anchor_transport
```

- status: completed
- note: candidate_groups=0 h4800=0

## 2026-06-05 21:05:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f169_f171_h3200_anchor_transport_oldshape_full --tag f169_f171_h3200_anchor_transport
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 21:08:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f169_f171_h3200_anchor_transport_oldshape_full --tag f169_f171_h3200_anchor_transport
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 21:09:35 +0800

```bash
python -m compileall -q dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_03_finalize.py
```

- status: completed
- note: static compile after F166-F168 mechanism/finalizer edits

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f166_f168_anti_erosion_smoke --scope mlp --spec-ids MLP-F166-early100-h800-source-slow-ema-terminal-anti-erosion-orthogonal,MLP-F167-early100-h800-source-slow-ema-terminal-source-reflection-guard,MLP-F168-early100-h800-source-slow-ema-terminal-h4000-transport-corrector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f166_f168_anti_erosion_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 4200 --shard-count 1 --shard-index 0 --device cuda:0
```

- status: completed
- note: F166-F168 anti-erosion smoke

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f166_f168_anti_erosion_smoke --scope mlp --merge-only
```

- status: completed
- note: merge F166-F168 smoke

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03 --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f166_f168_anti_erosion_smoke --tag f166_f168_anti_erosion_smoke --label "F166-F168 anti-erosion smoke"
```

- status: completed
- note: summarize F166-F168 smoke

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f166_f168_anti_erosion_oldshape_full --scope mlp --spec-ids MLP-F166-early100-h800-source-slow-ema-terminal-anti-erosion-orthogonal,MLP-F167-early100-h800-source-slow-ema-terminal-source-reflection-guard,MLP-F168-early100-h800-source-slow-ema-terminal-h4000-transport-corrector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f166_f168_anti_erosion_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: completed
- note: F166-F168 4GPU old-shape full shard 0

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f166_f168_anti_erosion_oldshape_full --scope mlp --spec-ids MLP-F166-early100-h800-source-slow-ema-terminal-anti-erosion-orthogonal,MLP-F167-early100-h800-source-slow-ema-terminal-source-reflection-guard,MLP-F168-early100-h800-source-slow-ema-terminal-h4000-transport-corrector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f166_f168_anti_erosion_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: completed
- note: F166-F168 4GPU old-shape full shard 1

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f166_f168_anti_erosion_oldshape_full --scope mlp --spec-ids MLP-F166-early100-h800-source-slow-ema-terminal-anti-erosion-orthogonal,MLP-F167-early100-h800-source-slow-ema-terminal-source-reflection-guard,MLP-F168-early100-h800-source-slow-ema-terminal-h4000-transport-corrector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f166_f168_anti_erosion_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: completed
- note: F166-F168 4GPU old-shape full shard 2

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f166_f168_anti_erosion_oldshape_full --scope mlp --spec-ids MLP-F166-early100-h800-source-slow-ema-terminal-anti-erosion-orthogonal,MLP-F167-early100-h800-source-slow-ema-terminal-source-reflection-guard,MLP-F168-early100-h800-source-slow-ema-terminal-h4000-transport-corrector,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f166_f168_anti_erosion_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: completed
- note: F166-F168 4GPU old-shape full shard 3

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f166_f168_anti_erosion_oldshape_full --scope mlp --merge-only
```

- status: completed
- note: merge F166-F168 full

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03 --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f166_f168_anti_erosion_oldshape_full --tag f166_f168_anti_erosion --label "F166-F168 anti-erosion orthogonal/transport oldshape full"
```

- status: completed
- note: summarize F166-F168 full

## 2026-06-05 21:09:35 +0800

```bash
python -m compileall -q experiments/run_v22_03_source_preservation.py experiments/run_v22_03_finalize.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py dgkan/fu/mechanisms.py
```

- status: completed
- note: static compile after F169-F171 h3200-anchor edits

## 2026-06-05 21:09:35 +0800

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

- status: completed
- note: pre-launch GPU availability check; all four GPUs idle

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f169_f171_h3200_anchor_transport_smoke --scope mlp --spec-ids MLP-F169-early100-h800-source-slow-ema-terminal-h3200-anchor-transport,MLP-F170-early100-h800-source-slow-ema-terminal-raw-guard-h3200-anchor-transport,MLP-F171-early100-h800-source-slow-ema-terminal-h3200-ratio-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f169_f171_h3200_anchor_transport_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 4200 --shard-count 1 --shard-index 0 --device cuda:0
```

- status: completed
- note: F169-F171 h3200-anchor transport smoke

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f169_f171_h3200_anchor_transport_smoke --scope mlp --merge-only
```

- status: completed
- note: merge F169-F171 smoke

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03 --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f169_f171_h3200_anchor_transport_smoke --tag f169_f171_h3200_anchor_transport_smoke --label "F169-F171 h3200-anchor transport smoke"
```

- status: completed
- note: summarize F169-F171 smoke

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f169_f171_h3200_anchor_transport_oldshape_full --scope mlp --spec-ids MLP-F169-early100-h800-source-slow-ema-terminal-h3200-anchor-transport,MLP-F170-early100-h800-source-slow-ema-terminal-raw-guard-h3200-anchor-transport,MLP-F171-early100-h800-source-slow-ema-terminal-h3200-ratio-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f169_f171_h3200_anchor_transport_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: completed
- note: F169-F171 4GPU old-shape full shard 0

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f169_f171_h3200_anchor_transport_oldshape_full --scope mlp --spec-ids MLP-F169-early100-h800-source-slow-ema-terminal-h3200-anchor-transport,MLP-F170-early100-h800-source-slow-ema-terminal-raw-guard-h3200-anchor-transport,MLP-F171-early100-h800-source-slow-ema-terminal-h3200-ratio-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f169_f171_h3200_anchor_transport_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: completed
- note: F169-F171 4GPU old-shape full shard 1

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f169_f171_h3200_anchor_transport_oldshape_full --scope mlp --spec-ids MLP-F169-early100-h800-source-slow-ema-terminal-h3200-anchor-transport,MLP-F170-early100-h800-source-slow-ema-terminal-raw-guard-h3200-anchor-transport,MLP-F171-early100-h800-source-slow-ema-terminal-h3200-ratio-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f169_f171_h3200_anchor_transport_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: completed
- note: F169-F171 4GPU old-shape full shard 2

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f169_f171_h3200_anchor_transport_oldshape_full --scope mlp --spec-ids MLP-F169-early100-h800-source-slow-ema-terminal-h3200-anchor-transport,MLP-F170-early100-h800-source-slow-ema-terminal-raw-guard-h3200-anchor-transport,MLP-F171-early100-h800-source-slow-ema-terminal-h3200-ratio-reentry,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f169_f171_h3200_anchor_transport_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: completed
- note: F169-F171 4GPU old-shape full shard 3

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f169_f171_h3200_anchor_transport_oldshape_full --scope mlp --merge-only
```

- status: completed
- note: merge F169-F171 full

## 2026-06-05 21:09:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03 --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f169_f171_h3200_anchor_transport_oldshape_full --tag f169_f171_h3200_anchor_transport --label "F169-F171 h3200-anchor transport oldshape full"
```

- status: completed
- note: summarize F169-F171 full; candidate_h4800=0 best_ratio=0.41673011176809377

## 2026-06-05 21:11:35 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 21:11:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 21:11:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 21:17:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f145_f147_source_preserve_oldshape_full --tag f145_f147_source_preserve
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 21:17:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f148_f150_diffeomorphic_target_oldshape_full --tag f148_f150_diffeomorphic_target
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 21:17:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f151_f153_terminal_preserve_repair_oldshape_full --tag f151_f153_terminal_preserve_repair
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 21:17:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f154_f156_terminal_debt_lownds_dualmem_oldshape_full --tag f154_f156_terminal_debt_lownds_dualmem
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 21:17:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f157_f159_signal_estimator_oldshape_full --tag f157_f159_signal_estimator
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 21:17:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f160_f162_post_h4000_source_floor_oldshape_full --tag f160_f162_post_h4000_source_floor
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 21:17:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f163_f165_raw_guard_post_h4000_floor_oldshape_full --tag f163_f165_raw_guard_post_h4000_floor
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 21:17:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f166_f168_anti_erosion_oldshape_full --tag f166_f168_anti_erosion
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 21:17:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f169_f171_h3200_anchor_transport_oldshape_full --tag f169_f171_h3200_anchor_transport
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 21:17:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 21:17:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 21:41:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f172_f174_progress_carry_smoke --tag f172_f174_progress_carry_smoke
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 21:46:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f172_f174_progress_carry_oldshape_full --tag f172_f174_progress_carry
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 21:59:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f175_f177_gentle_progress_smoke --tag f175_f177_gentle_progress_smoke
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 22:03:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f175_f177_gentle_progress_oldshape_full --tag f175_f177_gentle_progress
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 22:06:33 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 22:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f172_f174_progress_carry_smoke --scope mlp --spec-ids MLP-F172-early100-h800-source-slow-ema-terminal-h3200-progress-carry,MLP-F173-early100-h800-source-slow-ema-terminal-h4000-progress-carry,MLP-F174-early100-h800-source-slow-ema-terminal-h3200-source-progress-blend,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f172_f174_progress_carry_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 4800 --shard-count 1 --shard-index 0 --device cuda:0
```

- status: completed
- note: manual backfill: completed before finalization

## 2026-06-05 22:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f172_f174_progress_carry_smoke --scope mlp --merge-only
```

- status: completed
- note: manual backfill: completed before finalization

## 2026-06-05 22:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f172_f174_progress_carry_oldshape_full --scope mlp --spec-ids MLP-F172-early100-h800-source-slow-ema-terminal-h3200-progress-carry,MLP-F173-early100-h800-source-slow-ema-terminal-h4000-progress-carry,MLP-F174-early100-h800-source-slow-ema-terminal-h3200-source-progress-blend,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f172_f174_progress_carry_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: completed
- note: manual backfill: completed before finalization

## 2026-06-05 22:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f172_f174_progress_carry_oldshape_full --scope mlp --spec-ids MLP-F172-early100-h800-source-slow-ema-terminal-h3200-progress-carry,MLP-F173-early100-h800-source-slow-ema-terminal-h4000-progress-carry,MLP-F174-early100-h800-source-slow-ema-terminal-h3200-source-progress-blend,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f172_f174_progress_carry_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: completed
- note: manual backfill: completed before finalization

## 2026-06-05 22:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f172_f174_progress_carry_oldshape_full --scope mlp --spec-ids MLP-F172-early100-h800-source-slow-ema-terminal-h3200-progress-carry,MLP-F173-early100-h800-source-slow-ema-terminal-h4000-progress-carry,MLP-F174-early100-h800-source-slow-ema-terminal-h3200-source-progress-blend,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f172_f174_progress_carry_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: completed
- note: manual backfill: completed before finalization

## 2026-06-05 22:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f172_f174_progress_carry_oldshape_full --scope mlp --spec-ids MLP-F172-early100-h800-source-slow-ema-terminal-h3200-progress-carry,MLP-F173-early100-h800-source-slow-ema-terminal-h4000-progress-carry,MLP-F174-early100-h800-source-slow-ema-terminal-h3200-source-progress-blend,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f172_f174_progress_carry_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: completed
- note: manual backfill: completed before finalization

## 2026-06-05 22:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f172_f174_progress_carry_oldshape_full --scope mlp --merge-only
```

- status: completed
- note: manual backfill: completed before finalization

## 2026-06-05 22:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f175_f177_gentle_progress_smoke --scope mlp --spec-ids MLP-F175-early100-h800-source-slow-ema-terminal-raw-guard-h3600-gentle-progress,MLP-F176-early100-h800-source-slow-ema-terminal-raw-guard-h4000-gentle-progress,MLP-F177-early100-h800-source-slow-ema-terminal-raw-guard-h4400-projected-progress,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f175_f177_gentle_progress_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 4800 --shard-count 1 --shard-index 0 --device cuda:0
```

- status: completed
- note: manual backfill: completed before finalization

## 2026-06-05 22:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f175_f177_gentle_progress_smoke --scope mlp --merge-only
```

- status: completed
- note: manual backfill: completed before finalization

## 2026-06-05 22:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f175_f177_gentle_progress_oldshape_full --scope mlp --spec-ids MLP-F175-early100-h800-source-slow-ema-terminal-raw-guard-h3600-gentle-progress,MLP-F176-early100-h800-source-slow-ema-terminal-raw-guard-h4000-gentle-progress,MLP-F177-early100-h800-source-slow-ema-terminal-raw-guard-h4400-projected-progress,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f175_f177_gentle_progress_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: completed
- note: manual backfill: completed before finalization

## 2026-06-05 22:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f175_f177_gentle_progress_oldshape_full --scope mlp --spec-ids MLP-F175-early100-h800-source-slow-ema-terminal-raw-guard-h3600-gentle-progress,MLP-F176-early100-h800-source-slow-ema-terminal-raw-guard-h4000-gentle-progress,MLP-F177-early100-h800-source-slow-ema-terminal-raw-guard-h4400-projected-progress,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f175_f177_gentle_progress_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: completed
- note: manual backfill: completed before finalization

## 2026-06-05 22:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f175_f177_gentle_progress_oldshape_full --scope mlp --spec-ids MLP-F175-early100-h800-source-slow-ema-terminal-raw-guard-h3600-gentle-progress,MLP-F176-early100-h800-source-slow-ema-terminal-raw-guard-h4000-gentle-progress,MLP-F177-early100-h800-source-slow-ema-terminal-raw-guard-h4400-projected-progress,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f175_f177_gentle_progress_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: completed
- note: manual backfill: completed before finalization

## 2026-06-05 22:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f175_f177_gentle_progress_oldshape_full --scope mlp --spec-ids MLP-F175-early100-h800-source-slow-ema-terminal-raw-guard-h3600-gentle-progress,MLP-F176-early100-h800-source-slow-ema-terminal-raw-guard-h4000-gentle-progress,MLP-F177-early100-h800-source-slow-ema-terminal-raw-guard-h4400-projected-progress,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f175_f177_gentle_progress_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: completed
- note: manual backfill: completed before finalization

## 2026-06-05 22:08:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f175_f177_gentle_progress_oldshape_full --scope mlp --merge-only
```

- status: completed
- note: manual backfill: completed before finalization

## 2026-06-05 22:09:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 22:09:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 22:13:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 22:13:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 22:30:22 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f178_f180_control_relative_catchup_smoke --tag f178_f180_control_relative_catchup_smoke
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 22:35:18 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f178_f180_control_relative_catchup_smoke --tag f178_f180_control_relative_catchup_smoke
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 22:40:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f178_f180_control_relative_catchup_oldshape_full --tag f178_f180_control_relative_catchup
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 22:41:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_03_code_truth_gate.py experiments/run_v22_03_finalize.py
```

- status: completed
- note: F178-F180 mechanism/finalizer syntax check

## 2026-06-05 22:41:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f178_f180_control_relative_catchup_smoke --scope mlp --spec-ids MLP-F178-early100-h800-source-slow-ema-terminal-control-relative-sgd-catchup,MLP-F179-early100-h800-source-slow-ema-terminal-control-relative-adamw-catchup,MLP-F180-early100-h800-source-slow-ema-terminal-control-relative-source-balanced-catchup,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f178_f180_control_relative_catchup_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 4800 --shard-count 1 --shard-index 0 --device cuda:0
```

- status: completed
- note: initial smoke exposed missing source-retention wiring; preserved as *_pre_wiring_fix and superseded

## 2026-06-05 22:41:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f178_f180_control_relative_catchup_smoke --scope mlp --merge-only
```

- status: completed
- note: initial F178-F180 smoke merge before wiring fix

## 2026-06-05 22:41:59 +0800

```bash
mv continuation_f178_f180_control_relative_catchup_smoke to continuation_f178_f180_control_relative_catchup_smoke_pre_wiring_fix and move smoke route/audit files
```

- status: completed
- note: kept failed wiring smoke provenance separate from fixed smoke

## 2026-06-05 22:41:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f178_f180_control_relative_catchup_smoke --scope mlp --spec-ids MLP-F178-early100-h800-source-slow-ema-terminal-control-relative-sgd-catchup,MLP-F179-early100-h800-source-slow-ema-terminal-control-relative-adamw-catchup,MLP-F180-early100-h800-source-slow-ema-terminal-control-relative-source-balanced-catchup,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f178_f180_control_relative_catchup_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 4800 --shard-count 1 --shard-index 0 --device cuda:0
```

- status: completed
- note: fixed F178-F180 smoke after adding M206-M208 to source-retention mechanism sets

## 2026-06-05 22:41:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f178_f180_control_relative_catchup_smoke --scope mlp --merge-only
```

- status: completed
- note: fixed F178-F180 smoke merge

## 2026-06-05 22:41:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03 --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f178_f180_control_relative_catchup_smoke --tag f178_f180_control_relative_catchup_smoke --label "F178-F180 control-relative terminal catch-up smoke"
```

- status: completed
- note: fixed smoke: candidate_early_chain=2 candidate_h4800=0 best_ratio=0.39633994995869104

## 2026-06-05 22:41:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f178_f180_control_relative_catchup_oldshape_full --scope mlp --spec-ids MLP-F178-early100-h800-source-slow-ema-terminal-control-relative-sgd-catchup,MLP-F179-early100-h800-source-slow-ema-terminal-control-relative-adamw-catchup,MLP-F180-early100-h800-source-slow-ema-terminal-control-relative-source-balanced-catchup,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f178_f180_control_relative_catchup_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: completed
- note: F178-F180 oldshape full shard 0/4

## 2026-06-05 22:41:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f178_f180_control_relative_catchup_oldshape_full --scope mlp --spec-ids MLP-F178-early100-h800-source-slow-ema-terminal-control-relative-sgd-catchup,MLP-F179-early100-h800-source-slow-ema-terminal-control-relative-adamw-catchup,MLP-F180-early100-h800-source-slow-ema-terminal-control-relative-source-balanced-catchup,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f178_f180_control_relative_catchup_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: completed
- note: F178-F180 oldshape full shard 1/4

## 2026-06-05 22:41:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f178_f180_control_relative_catchup_oldshape_full --scope mlp --spec-ids MLP-F178-early100-h800-source-slow-ema-terminal-control-relative-sgd-catchup,MLP-F179-early100-h800-source-slow-ema-terminal-control-relative-adamw-catchup,MLP-F180-early100-h800-source-slow-ema-terminal-control-relative-source-balanced-catchup,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f178_f180_control_relative_catchup_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: completed
- note: F178-F180 oldshape full shard 2/4

## 2026-06-05 22:41:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f178_f180_control_relative_catchup_oldshape_full --scope mlp --spec-ids MLP-F178-early100-h800-source-slow-ema-terminal-control-relative-sgd-catchup,MLP-F179-early100-h800-source-slow-ema-terminal-control-relative-adamw-catchup,MLP-F180-early100-h800-source-slow-ema-terminal-control-relative-source-balanced-catchup,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f178_f180_control_relative_catchup_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: completed
- note: F178-F180 oldshape full shard 3/4

## 2026-06-05 22:41:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f178_f180_control_relative_catchup_oldshape_full --scope mlp --merge-only
```

- status: completed
- note: F178-F180 oldshape full merge

## 2026-06-05 22:41:59 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03 --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f178_f180_control_relative_catchup_oldshape_full --tag f178_f180_control_relative_catchup --label "F178-F180 control-relative terminal catch-up oldshape full"
```

- status: completed
- note: F178-F180 full: candidate_h4800=0 best=F178 ratio=0.4097839293516009

## 2026-06-05 22:42:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 22:42:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 22:42:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 23:02:08 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 23:04:31 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f181_f183_trajectory_adaptive_smoke --tag f181_f183_trajectory_adaptive_smoke
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 23:11:14 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f181_f183_trajectory_adaptive_oldshape_full --tag f181_f183_trajectory_adaptive
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 23:31:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 23:33:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f184_f186_minimal_transport_smoke --tag f184_f186_minimal_transport_smoke
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 23:36:32 +0800

```bash
mv F184-F186 minimal_transport_smoke to *_pre_front_wiring_fix and move smoke route/audit files
```

- status: completed
- note: initial F184-F186 smoke exposed missing early active-source wiring; preserved and superseded

## 2026-06-05 23:36:54 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_03_code_truth_gate.py experiments/run_v22_03_finalize.py
```

- status: completed
- note: F184-F186 early active-source wiring fix syntax check

## 2026-06-05 23:36:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 23:38:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f184_f186_minimal_transport_smoke --tag f184_f186_minimal_transport_smoke
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 23:40:22 +0800

```bash
mv F184-F186 minimal_transport_smoke to *_pre_runner_wiring_fix and move smoke route/audit files
```

- status: completed
- note: second F184-F186 smoke showed M212-M214 missing from v17 source-retention runner allowlist; preserved and superseded

## 2026-06-05 23:40:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_03_code_truth_gate.py experiments/run_v22_03_finalize.py
```

- status: completed
- note: F184-F186 v17 source-retention runner allowlist fix syntax check

## 2026-06-05 23:40:43 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 23:42:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f184_f186_minimal_transport_smoke --tag f184_f186_minimal_transport_smoke
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 23:47:03 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f184_f186_minimal_transport_oldshape_full --tag f184_f186_minimal_transport
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-05 23:50:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f184_f186_minimal_transport_smoke --scope mlp --spec-ids MLP-F184-early100-h800-source-slow-ema-terminal-anti-source-clip,MLP-F185-early100-h800-source-slow-ema-terminal-debt-aware-hold,MLP-F186-early100-h800-source-slow-ema-terminal-anchor-flow-tiny,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f184_f186_minimal_transport_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 4800 --shard-count 1 --shard-index 0 --device cuda:0
```

- status: completed
- note: fixed F184-F186 smoke after M212-M214 runner allowlist and active-source wiring; best=F184 ratio=0.47235437280356474 h4800=0

## 2026-06-05 23:50:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f184_f186_minimal_transport_smoke --scope mlp --merge-only
```

- status: completed
- note: fixed F184-F186 smoke merge

## 2026-06-05 23:50:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f184_f186_minimal_transport_smoke --tag f184_f186_minimal_transport_smoke --label "F184-F186 minimal terminal transport smoke"
```

- status: completed
- note: fixed smoke preservation: candidate_early_chain=2 candidate_h4800=0

## 2026-06-05 23:50:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f184_f186_minimal_transport_oldshape_full --scope mlp --spec-ids MLP-F184-early100-h800-source-slow-ema-terminal-anti-source-clip,MLP-F185-early100-h800-source-slow-ema-terminal-debt-aware-hold,MLP-F186-early100-h800-source-slow-ema-terminal-anchor-flow-tiny,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f184_f186_minimal_transport_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
```

- status: completed
- note: F184-F186 oldshape full shard 0/4

## 2026-06-05 23:50:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f184_f186_minimal_transport_oldshape_full --scope mlp --spec-ids MLP-F184-early100-h800-source-slow-ema-terminal-anti-source-clip,MLP-F185-early100-h800-source-slow-ema-terminal-debt-aware-hold,MLP-F186-early100-h800-source-slow-ema-terminal-anchor-flow-tiny,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f184_f186_minimal_transport_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
```

- status: completed
- note: F184-F186 oldshape full shard 1/4

## 2026-06-05 23:50:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f184_f186_minimal_transport_oldshape_full --scope mlp --spec-ids MLP-F184-early100-h800-source-slow-ema-terminal-anti-source-clip,MLP-F185-early100-h800-source-slow-ema-terminal-debt-aware-hold,MLP-F186-early100-h800-source-slow-ema-terminal-anchor-flow-tiny,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f184_f186_minimal_transport_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
```

- status: completed
- note: F184-F186 oldshape full shard 2/4

## 2026-06-05 23:50:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f184_f186_minimal_transport_oldshape_full --scope mlp --spec-ids MLP-F184-early100-h800-source-slow-ema-terminal-anti-source-clip,MLP-F185-early100-h800-source-slow-ema-terminal-debt-aware-hold,MLP-F186-early100-h800-source-slow-ema-terminal-anchor-flow-tiny,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f184_f186_minimal_transport_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: completed
- note: F184-F186 oldshape full shard 3/4

## 2026-06-05 23:50:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f184_f186_minimal_transport_oldshape_full --scope mlp --merge-only
```

- status: completed
- note: F184-F186 oldshape full merge

## 2026-06-05 23:50:20 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f184_f186_minimal_transport_oldshape_full --tag f184_f186_minimal_transport --label "F184-F186 minimal terminal transport oldshape full"
```

- status: completed
- note: F184-F186 full: candidate_h4800=0 best=F186 ratio=0.36831442816645493

## 2026-06-05 23:50:42 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_03_code_truth_gate.py experiments/run_v22_03_finalize.py
```

- status: completed
- note: final F184-F186/finalizer syntax check before packet

## 2026-06-05 23:50:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-05 23:50:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 23:50:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-05 23:51:51 +0800

```bash
rename nested F184-F186 smoke provenance artifacts from *_pre_runner_wiring_fix_pre_front_wiring_fix_* to *_pre_front_wiring_fix_*
```

- status: completed
- note: artifact naming cleanup only; file contents unchanged

## 2026-06-05 23:52:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-05 23:52:11 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-06 00:11:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_code_truth_gate.py --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1

## 2026-06-06 00:12:49 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f187_f189_accept_memory_smoke --tag f187_f189_accept_memory_smoke
```

- status: completed
- note: candidate_groups=0 h4800=0

## 2026-06-06 00:13:52 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f187_f189_accept_memory_smoke --tag f187_f189_accept_memory_smoke
```

- status: completed
- note: candidate_groups=0 h4800=0

## 2026-06-06 00:14:15 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f187_f189_accept_memory_smoke --tag f187_f189_accept_memory_smoke
```

- status: completed
- note: candidate_groups=0 h4800=0

## 2026-06-06 00:16:36 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f187_f189_accept_memory_smoke --tag f187_f189_accept_memory_smoke
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-06 00:20:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_source_preservation.py --source-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f187_f189_accept_memory_oldshape_full --tag f187_f189_accept_memory
```

- status: completed
- note: candidate_groups=3 h4800=0

## 2026-06-06 00:21:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-06 00:21:10 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-06 00:23:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/fu/mechanisms.py experiments/run_v17_common.py experiments/run_v21_common.py experiments/run_v21_01_source_retention.py experiments/run_v22_03_code_truth_gate.py experiments/run_v22_03_finalize.py
```

- status: completed
- note: F187-F189 implementation syntax gate

## 2026-06-06 00:23:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f187_f189_accept_memory_smoke --scope mlp --spec-ids MLP-F187-early100-h800-source-slow-ema-terminal-accept-memory,MLP-F188-early100-h800-source-slow-ema-terminal-accept-memory-debt,MLP-F189-early100-h800-source-slow-ema-terminal-source-progress-memory,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f187_f189_accept_memory_smoke --datasets MNIST --seeds 0 --train-size 256 --val-size 128 --batch-size 64 --steps 4800 --shard-count 1 --shard-index 0 --device cuda:0
```

- status: completed
- note: first smoke before mlp scope list fix produced control-only rows; rerun after fixing `experiments/run_v21_01_source_retention.py` produced candidate_groups=3 h4800=0

## 2026-06-06 00:23:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f187_f189_accept_memory_smoke --scope mlp --merge-only
```

- status: completed
- note: smoke canonical matrix merge

## 2026-06-06 00:23:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f187_f189_accept_memory_oldshape_full --scope mlp --spec-ids MLP-F187-early100-h800-source-slow-ema-terminal-accept-memory,MLP-F188-early100-h800-source-slow-ema-terminal-accept-memory-debt,MLP-F189-early100-h800-source-slow-ema-terminal-source-progress-memory,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f187_f189_accept_memory_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 0 --device cuda:0
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f187_f189_accept_memory_oldshape_full --scope mlp --spec-ids MLP-F187-early100-h800-source-slow-ema-terminal-accept-memory,MLP-F188-early100-h800-source-slow-ema-terminal-accept-memory-debt,MLP-F189-early100-h800-source-slow-ema-terminal-source-progress-memory,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f187_f189_accept_memory_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 1 --device cuda:1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f187_f189_accept_memory_oldshape_full --scope mlp --spec-ids MLP-F187-early100-h800-source-slow-ema-terminal-accept-memory,MLP-F188-early100-h800-source-slow-ema-terminal-accept-memory-debt,MLP-F189-early100-h800-source-slow-ema-terminal-source-progress-memory,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f187_f189_accept_memory_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 2 --device cuda:2
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f187_f189_accept_memory_oldshape_full --scope mlp --spec-ids MLP-F187-early100-h800-source-slow-ema-terminal-accept-memory,MLP-F188-early100-h800-source-slow-ema-terminal-accept-memory-debt,MLP-F189-early100-h800-source-slow-ema-terminal-source-progress-memory,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --run-label v2203_cont_f187_f189_accept_memory_oldshape_full --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index 3 --device cuda:3
```

- status: completed
- note: four parallel shard commands; rows=63 after merge

## 2026-06-06 00:23:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/continuation_f187_f189_accept_memory_oldshape_full --scope mlp --merge-only
```

- status: completed
- note: full canonical matrix merge

## 2026-06-06 00:24:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-06 00:24:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained

## 2026-06-06 00:25:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q .  # inside extracted v22_03_code_review_packet.zip
```

- status: completed
- note: clean unzip self-test exit=0

## 2026-06-06 00:25:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_03_finalize.py --out-dir results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03
```

- status: completed
- note: route=R3-MLPTerminalErosionExplained
