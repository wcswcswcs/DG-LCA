# DG-KAN v13.10 CoverFormingSubstrateArchitectureReset 执行日志

生成时间：2026-05-28（Asia/Singapore）

本日志只记录实际执行过的命令、输出目录和复现要点。

## 1. 计划与代码读取

```bash
sed -n '1,260p' docs/DG-KAN_v13.10_CoverFormingSubstrateArchitectureReset_完整计划.md
rg --files | rg 'v13\.10|v1310|CoverForming|cover_forming'
git status --short
rg -n "route|Route|R[0-9]|S3|S4|S5|K14|K15|K16|K17|K18|Case|artifact|manifest|stop|Stop|停止|执行|命令|runner|Non-RAT|promotion|gate|修复|fallback|official|smoke" docs/DG-KAN_v13.10_CoverFormingSubstrateArchitectureReset_完整计划.md
sed -n '260,620p' docs/DG-KAN_v13.10_CoverFormingSubstrateArchitectureReset_完整计划.md
sed -n '620,980p' docs/DG-KAN_v13.10_CoverFormingSubstrateArchitectureReset_完整计划.md
sed -n '980,1225p' docs/DG-KAN_v13.10_CoverFormingSubstrateArchitectureReset_完整计划.md
```

相关代码读取：

```bash
sed -n '1,240p' experiments/run_v139_signal_to_cover_functional_substrate_architecture.py
rg -n "K_METHOD_ALIASES|run_k1_lift|run_training_case|cover_purity|rational-candidate|choose_primary_rational|write_required|route_decision|argparse" experiments/run_v139_signal_to_cover_functional_substrate_architecture.py
sed -n '560,820p' experiments/run_v137_boundary_conditioned_poprisk_training.py
sed -n '820,1040p' experiments/run_v137_boundary_conditioned_poprisk_training.py
sed -n '180,420p' experiments/run_v137_boundary_conditioned_poprisk_training.py
conda run -n kan python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.__version__)"
```

GPU 检查结果：

```text
True 4 2.11.0+cu128
```

## 2. 代码修改与语法检查

新增 runner：

```text
experiments/run_v1310_cover_forming_substrate_architecture_reset.py
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1310_cover_forming_substrate_architecture_reset.py
```

结果：

```text
py_compile pass
```

## 3. smoke

```bash
conda run -n kan python experiments/run_v1310_cover_forming_substrate_architecture_reset.py \
  --out-dir results/v13_10_cover_forming_substrate_architecture_reset/smoke_v1310 \
  --architectures A-RCF1-MultiBandRationalCover,A-RCF6-OvercompleteSparseCoverBank \
  --synthetic-tasks X1 \
  --synthetic-seeds 0 \
  --k-losses CE \
  --k-methods K0-RAT-AdamW,K14-RCF-SNRClusterTraining,K18-RCF-OvercompleteSparseCover \
  --train-steps 4 \
  --batch-size 16 \
  --log-interval 2 \
  --synthetic-train-size 32 \
  --synthetic-val-size 16 \
  --datasets MNIST \
  --mlp-seeds 0 \
  --mlp-seed-threshold 1 \
  --real-train-size 64 \
  --real-val-size 32 \
  --real-test-size 32 \
  --real-epochs 1
```

第一次 smoke 发现 manifest 输出顺序问题：`v1310_route_decision.json` 在 manifest 检查后才写入，导致 `required_artifact_missing_count=1`。修复后重跑同一命令，结果：

```text
route = R1-NoCoverSubstrate
cover_substrate_pass_count = 0
cover_purity_median = 0.0660642609000206
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
required_artifact_missing_count = 0
```

## 4. all-A-RCF scout / compute-budgeted official_v1310

```bash
conda run -n kan python experiments/run_v1310_cover_forming_substrate_architecture_reset.py \
  --out-dir results/v13_10_cover_forming_substrate_architecture_reset/official_v1310 \
  --architectures A-RCF1-MultiBandRationalCover,A-RCF2-SNRClusterCoverWarmup,A-RCF3-PersistentCoverMemory,A-RCF4-SignalReservoirSplitGroups,A-RCF5-ReadoutBasisDecoupledCover,A-RCF6-OvercompleteSparseCoverBank \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --k-losses CE,Brier \
  --k-methods K0-RAT-AdamW,K14-RCF-SNRClusterTraining,K15-RCF-PersistentCoverMemory,K16-RCF-SignalReservoirSplit,K17-RCF-ReadoutBasisDecoupled,K18-RCF-OvercompleteSparseCover \
  --train-steps 60 \
  --batch-size 32 \
  --log-interval 20 \
  --synthetic-train-size 96 \
  --synthetic-val-size 48 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --mlp-seeds 0,1 \
  --mlp-seed-threshold 2 \
  --real-train-size 128 \
  --real-val-size 64 \
  --real-test-size 64 \
  --real-epochs 1
```

输出目录：

```text
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310
```

## 5. top-2 A-RCF hardening

依据 `official_v1310/v1310_rational_cover_substrate.csv` 与 `v1310_signal_to_cover_summary.csv`，选择：

```text
A-RCF1-MultiBandRationalCover
A-RCF6-OvercompleteSparseCoverBank
```

执行：

```bash
conda run -n kan python experiments/run_v1310_cover_forming_substrate_architecture_reset.py \
  --out-dir results/v13_10_cover_forming_substrate_architecture_reset/hardening_v1310_top2_arcf1_arcf6 \
  --architectures A-RCF1-MultiBandRationalCover,A-RCF6-OvercompleteSparseCoverBank \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0,1 \
  --k-losses CE,Brier \
  --k-methods K0-RAT-AdamW,K14-RCF-SNRClusterTraining,K15-RCF-PersistentCoverMemory,K16-RCF-SignalReservoirSplit,K17-RCF-ReadoutBasisDecoupled,K18-RCF-OvercompleteSparseCover \
  --train-steps 100 \
  --batch-size 32 \
  --log-interval 25 \
  --synthetic-train-size 96 \
  --synthetic-val-size 48 \
  --datasets MNIST \
  --mlp-seeds 0 \
  --mlp-seed-threshold 1 \
  --real-train-size 128 \
  --real-val-size 64 \
  --real-test-size 64 \
  --real-epochs 1
```

输出目录：

```text
results/v13_10_cover_forming_substrate_architecture_reset/hardening_v1310_top2_arcf1_arcf6
```

## 6. cover-fail fallback：A-RCF2/A-RCF3/A-RCF6

触发原因：all-A-RCF scout 与 top-2 hardening 均为 retention pass but cover purity fail。按计划第 9.6 节补跑 A-RCF2/A-RCF3/A-RCF6。

```bash
conda run -n kan python experiments/run_v1310_cover_forming_substrate_architecture_reset.py \
  --out-dir results/v13_10_cover_forming_substrate_architecture_reset/fallback_v1310_arcf2_arcf3_arcf6_coverfail \
  --architectures A-RCF2-SNRClusterCoverWarmup,A-RCF3-PersistentCoverMemory,A-RCF6-OvercompleteSparseCoverBank \
  --synthetic-tasks X5,X6,X7 \
  --synthetic-seeds 0,1 \
  --k-losses CE,Brier \
  --k-methods K0-RAT-AdamW,K15-RCF-PersistentCoverMemory,K18-RCF-OvercompleteSparseCover \
  --train-steps 80 \
  --batch-size 32 \
  --log-interval 20 \
  --synthetic-train-size 96 \
  --synthetic-val-size 48 \
  --datasets MNIST \
  --mlp-seeds 0 \
  --mlp-seed-threshold 1 \
  --real-train-size 128 \
  --real-val-size 64 \
  --real-test-size 64 \
  --real-epochs 1
```

输出目录：

```text
results/v13_10_cover_forming_substrate_architecture_reset/fallback_v1310_arcf2_arcf3_arcf6_coverfail
```

## 7. 结果解析与进程复核

```bash
python - <<'PY'
import csv,json,os,collections
runs=['smoke_v1310','official_v1310','hardening_v1310_top2_arcf1_arcf6','fallback_v1310_arcf2_arcf3_arcf6_coverfail']
base='results/v13_10_cover_forming_substrate_architecture_reset'
for run in runs:
    p=f'{base}/{run}/v1310_route_decision.json'
    if os.path.exists(p):
        r=json.load(open(p))
        print(run, r['route'], r['cover_substrate_pass_count'], r['cover_purity_median'], r['kan_s3_task_pass_count'], r['kan_s4_task_pass_count'])
PY

pgrep -af 'run_v1310|run_v139|run_v136' || true
ps -ef | rg 'run_v1310|run_v139|run_v136' | rg -v 'rg|pgrep' || true
```

最终进程复核：

```text
`ps -ef | rg ... | rg -v ...` 无输出；
无 v13.10 / v13.9 / v13.6 实验进程残留。
```

## 8. 关键 route 摘要

官方 all-A-RCF scout：

```text
out_dir = results/v13_10_cover_forming_substrate_architecture_reset/official_v1310
route = R1-NoCoverSubstrate
minimum_success = S0-ArchitectureScoutExecuted
promotion_allowed = 0
official_success_reached = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
is_new_substrate_architecture = 1
is_k_token_only_extension = 0
cover_substrate_pass_count = 0
snr_transfer_median_retention_group = 1.0
snr_transfer_median_cos_group_vs_param = 0.786395251750946
cover_purity_median = 0.03301357105374336
cover_churn_mean = 0.08117913581481596
kan_s3_task_pass_count = 2
kan_s4_task_pass_count = 0
nonrat_vertical_pass_count = 0
mlp_generic_dataset_pass_count = 0 / 3
```

top-2 A-RCF hardening：

```text
out_dir = results/v13_10_cover_forming_substrate_architecture_reset/hardening_v1310_top2_arcf1_arcf6
route = R1-NoCoverSubstrate
cover_substrate_pass_count = 0
cover_purity_median = 0.037662800401449203
cover_churn_mean = 0.10255101824901541
kan_s3_task_pass_count = 1
kan_s4_task_pass_count = 0
required_artifact_missing_count = 0
```

cover-fail fallback：

```text
out_dir = results/v13_10_cover_forming_substrate_architecture_reset/fallback_v1310_arcf2_arcf3_arcf6_coverfail
route = R1-NoCoverSubstrate
cover_substrate_pass_count = 0
cover_purity_median = 0.029089346528053284
cover_churn_mean = 0.0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
required_artifact_missing_count = 0
```
