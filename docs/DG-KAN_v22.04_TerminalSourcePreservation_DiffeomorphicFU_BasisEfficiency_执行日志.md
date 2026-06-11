# DG-KAN v22.04 TerminalSourcePreservation DiffeomorphicFU BasisEfficiency 执行日志

生成时间：2026-06-06 02:52:07 +0800

记录原则：只记录真实执行过的命令、输入文件、输出 artifact、状态和 blocker；不把未执行内容写成结果。

## 2026-06-06 02:52:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_s011_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=0 failed=kernel_gradcheck

## 2026-06-06 02:53:00 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_s011_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 02:53:24 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_efficiency_officialization.py --v2202-dir /home/chengshun.wang/DG-LCA/results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02 --v2203-dir /home/chengshun.wang/DG-LCA/results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03 --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04
```

- status: completed
- note: summary_rows=2 D-CHE=1 D-FOU=1

## 2026-06-06 02:53:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_drat_drbf_officialization.py --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --device cuda:0 --batch-sizes 128,512 --hidden 128 --iters 30 --warmup 6
```

- status: completed
- note: active micro-kernel rows=28; official fused promotion remains blocked

## 2026-06-06 03:01:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_source_preserving_fu.py --source-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full --tag d1_source_preserving_fu --label 'D1 source-preserving FU oldshape full' --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04
```

- status: completed
- note: candidate_groups=7 productive_h4800=0 best=MLP-D1b-roworth-hidden-readout

## 2026-06-06 03:05:40 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_source_preserving_fu.py --source-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1b_single_roworth_oldshape_full --tag d1b_single_roworth --label 'D1b single row-orthogonal oldshape full' --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04
```

- status: completed
- note: candidate_groups=1 productive_h4800=0 best=MLP-D1b-roworth-hidden-readout

## 2026-06-06 03:09:27 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_source_preserving_fu.py --source-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_single_spp_lambda025_oldshape_full --tag d1a_single_spp_lambda025 --label 'D1a single SPP lambda025 oldshape full' --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04
```

- status: completed
- note: candidate_groups=1 productive_h4800=0 best=MLP-D1a-SPP-lambda025

## 2026-06-06 03:17:17 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_s011_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 03:23:23 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_source_preserving_fu.py --source-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full --tag d1a_lambda_strength --label 'D1a lambda strength oldshape full' --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04
```

- status: completed
- note: candidate_groups=0 productive_h4800=0 best=

## 2026-06-06 03:23:45 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_source_preserving_fu.py --source-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full --tag d1a_lambda_strength --label 'D1a lambda strength oldshape full' --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04
```

- status: completed
- note: candidate_groups=2 productive_h4800=0 best=MLP-D1a-SPP-lambda050

## 2026-06-06 03:27:13 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_s011_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 03:30:41 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_source_preserving_fu.py --source-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full --tag d1a_lambda_strength_wired --label 'D1a lambda strength wired oldshape full' --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04
```

- status: completed
- note: candidate_groups=2 productive_h4800=0 best=MLP-D1a-SPP-lambda050

## 2026-06-06 03:33:47 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_s011_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 03:38:44 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_source_preserving_fu.py --source-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full --tag d1a_lambda_strength_wired2 --label 'D1a lambda strength wired2 oldshape full' --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04
```

- status: completed
- note: candidate_groups=2 productive_h4800=0 best=MLP-D1a-SPP-lambda050

## 2026-06-06 03:40:34 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_s011_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 03:45:56 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_source_preserving_fu.py --source-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full --tag d1a_lambda_strength_wired3 --label 'D1a lambda strength wired3 oldshape full' --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04
```

- status: completed
- note: candidate_groups=2 productive_h4800=0 best=MLP-D1a-SPP-lambda050

## 2026-06-06 03:47:50 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_s011_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 03:48:02 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_terminal_erosion_autopsy.py --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --include-v2203 1
```

- status: completed
- note: groups=56 explained_fraction=1.0

## 2026-06-06 04:02:07 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_kan_source_mapping.py --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04 --source-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d2_kan_source_mapping_oldshape_full
```

- status: completed
- note: fresh=1 candidate_groups=8 mapped=0

## 2026-06-06 04:04:22 +0800

```bash
CUDA_VISIBLE_DEVICES={SHARD} /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full --scope mlp --run-label v22_04_d1_source_preserving_fu_oldshape_full --spec-ids MLP-D1a-SPP-lambda025,MLP-D1a-SPP-readout-only,MLP-D1b-roworth-hidden-readout,MLP-D1c-lowNDS-with-source-preservation,MLP-D1d-dualmem-longonly-terminal,MLP-D1e-combined-debt-aware,MLP-D1f-info-volume-plus-roworth,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index {SHARD} --device cuda:0
```

- status: completed-template
- note: executed for SHARD=0,1,2,3; D1 mixed full

## 2026-06-06 04:04:22 +0800

```bash
CUDA_VISIBLE_DEVICES={SHARD} /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1b_single_roworth_oldshape_full --scope mlp --run-label v22_04_d1b_single_roworth_oldshape_full --spec-ids MLP-D1b-roworth-hidden-readout,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index {SHARD} --device cuda:0
```

- status: completed-template
- note: executed for SHARD=0,1,2,3; D1b single full

## 2026-06-06 04:04:22 +0800

```bash
CUDA_VISIBLE_DEVICES={SHARD} /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_single_spp_lambda025_oldshape_full --scope mlp --run-label v22_04_d1a_single_spp_lambda025_oldshape_full --spec-ids MLP-D1a-SPP-lambda025,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index {SHARD} --device cuda:0
```

- status: completed-template
- note: executed for SHARD=0,1,2,3; D1a lambda025 single full

## 2026-06-06 04:04:22 +0800

```bash
CUDA_VISIBLE_DEVICES={SHARD} /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full --scope mlp --run-label v22_04_d1a_lambda_strength_oldshape_full --spec-ids MLP-D1a-SPP-lambda050,MLP-D1a-SPP-lambda100,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index {SHARD} --device cuda:0
```

- status: completed-template
- note: executed for SHARD=0,1,2,3; pre-wiring lambda strength full, retained as blocker evidence

## 2026-06-06 04:04:22 +0800

```bash
CUDA_VISIBLE_DEVICES={SHARD} /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full --scope mlp --run-label v22_04_d1a_lambda_strength_wired3_oldshape_full --spec-ids MLP-D1a-SPP-lambda050,MLP-D1a-SPP-lambda100,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index {SHARD} --device cuda:0
```

- status: completed-template
- note: executed for SHARD=0,1,2,3; post-wiring official lambda strength full

## 2026-06-06 04:04:22 +0800

```bash
CUDA_VISIBLE_DEVICES={SHARD} /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v21_01_source_retention.py --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d2_kan_source_mapping_oldshape_full --scope kan --run-label v22_04_d2_kan_source_mapping_oldshape_full --spec-ids KSW1-basis-estimate-readout-commit,KSW2-lowdegree-lowfreq-source-bank,KSW9-h800-source-slow-ema-bank,KSW10-h800-dual-memory-source-bank,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead --carriers D-CHE,D-FOU --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --batch-size 64 --steps 6400 --shard-count 4 --shard-index {SHARD} --device cuda:0
```

- status: completed-template
- note: executed for SHARD=0,1,2,3; fresh D2 KAN source mapping full

## 2026-06-06 04:04:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_s011_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 04:05:39 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_s011_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 04:06:09 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_merge_finalize.py --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04
```

- status: completed
- note: route=R3-DRATDRBFRepairBlocked packet=v22_04_code_review_packet.zip:f403c3f7810e90b4fc61cdf3125fd20a6b8716c9edb937cc4f9006c6a89a1ecf bundle=v22_04_results_bundle.zip:c60ab7a3a1cd3b1a110c33e3037958f0144dc21705a655cd89278819a2ef90c8

## 2026-06-06 04:07:28 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_s011_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA
```

- status: completed
- note: pass=1 failed=

## 2026-06-06 04:07:58 +0800

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_04_merge_finalize.py --out-dir results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04
```

- status: completed
- note: route=R3-DRATDRBFRepairBlocked packet=v22_04_code_review_packet.zip:965620b9b595966cdcd1edff8c987f1f2ae77862a3982b5e9828f246f9b5eb7e bundle=v22_04_results_bundle.zip:c878343894d5783cf22c174c15aaff73d73e7391344876df8973902eba208ac8
