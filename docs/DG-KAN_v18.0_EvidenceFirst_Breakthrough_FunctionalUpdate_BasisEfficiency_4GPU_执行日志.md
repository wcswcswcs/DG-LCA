# DG-KAN v18.0 Evidence-First 执行日志

生成时间：2026-06-02 22:12:33 +08
计划文件：/home/chengshun.wang/DG-LCA/docs/DG-KAN_v18.0_EvidenceFirst_Breakthrough_FunctionalUpdate_BasisEfficiency_4GPU_完整计划.md
结果目录：results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18
Python：/home/chengshun.wang/miniconda3/envs/kan/bin/python

## 实际执行命令
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v18_code_gate.py --out-dir results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18 --device cuda:0 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v18_functional_update_breakthrough.py --stage mechanism-shard --out-dir results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18 --shard-count 4 --shard-index 0 --device cuda:0 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v18_functional_update_breakthrough.py --stage mechanism-shard --out-dir results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18 --shard-count 4 --shard-index 1 --device cuda:1 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v18_functional_update_breakthrough.py --stage mechanism-shard --out-dir results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18 --shard-count 4 --shard-index 2 --device cuda:2 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v18_functional_update_breakthrough.py --stage mechanism-shard --out-dir results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18 --shard-count 4 --shard-index 3 --device cuda:3 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v18_basis_efficiency_breakthrough.py --stage efficiency-shard --out-dir results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18 --shard-count 4 --shard-index 0 --device cuda:0 --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v18_basis_efficiency_breakthrough.py --stage efficiency-shard --out-dir results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18 --shard-count 4 --shard-index 1 --device cuda:1 --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v18_basis_efficiency_breakthrough.py --stage efficiency-shard --out-dir results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18 --shard-count 4 --shard-index 2 --device cuda:2 --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v18_basis_efficiency_breakthrough.py --stage efficiency-shard --out-dir results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18 --shard-count 4 --shard-index 3 --device cuda:3 --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v18_functional_update_breakthrough.py --stage horizon-shard --out-dir results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18 --shard-count 4 --shard-index 0 --device cuda:0 --data-root data --horizon-all-rows 1 --horizon-steps 3200 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v18_functional_update_breakthrough.py --stage horizon-shard --out-dir results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18 --shard-count 4 --shard-index 1 --device cuda:1 --data-root data --horizon-all-rows 1 --horizon-steps 3200 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v18_functional_update_breakthrough.py --stage horizon-shard --out-dir results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18 --shard-count 4 --shard-index 2 --device cuda:2 --data-root data --horizon-all-rows 1 --horizon-steps 3200 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v18_functional_update_breakthrough.py --stage horizon-shard --out-dir results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18 --shard-count 4 --shard-index 3 --device cuda:3 --data-root data --horizon-all-rows 1 --horizon-steps 3200 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v18_merge_finalize.py --out-dir results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18 --data-root data --horizon-all-rows 1 --horizon-steps 3200`

## 关键文件
- route: `results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18/v18_route_decision.json`
- code packet: `results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18/v18_code_review_packet.zip`
- functional raw horizon matrix: `results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18/v18_functional_raw_horizon_matrix.csv`
- source retention matrix: `results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18/v18_source_retention_matrix.csv`
- debt accounting matrix: `results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18/v18_debt_accounting_matrix.csv`
- efficiency truth table: `results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18/v18_efficiency_truth_table.csv`
- efficiency blocker table: `results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18/v18_efficiency_blocker_table.csv`
- kernel repair matrix: `results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18/v18_kernel_repair_matrix.csv`
- queue drain: `results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18/v18_gpu_queue_drain_report.csv`
- command journal: `results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18/logs/v17_command_journal.md`

## 运行统计
- functional rows: 945
- efficiency rows: 28
- kernel rows: 6
- GPU snapshot rows: 948
- max GPU memory used MB: 693.0
- max GPU util percent: 22.0

| gpu | runtime_sec_sum |
|---|---:|
| cuda:0 | 117.40 |
| cuda:1 | 116.02 |
| cuda:2 | 124.32 |
| cuda:3 | 121.10 |

## Repro Notes
- `--horizon-all-rows 1 --horizon-steps 3200` 表示对已测 functional rows 逐行做 fresh horizon readback。
- v18 efficiency 使用 batch 8/32/128/256；threshold 按计划写入 `v18_efficiency_blocker_table.csv`。
- 当前 debt matrix 只从真实 trace 计算 CEp99 tail 和 val_loss AUC；LineC-channel/ECE/Brier debt 未真实测量时保留空值并标注 blocker。
- 当前 kernel repair matrix 对未实现 R1+ 变体逐条写 deferred，不冒充 official fused CUDA/C++。
