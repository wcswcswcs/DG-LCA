# DG-KAN v17 执行日志

生成时间：2026-06-02 16:14:37 +08
计划文件：/home/chengshun.wang/DG-LCA/docs/DG-KAN_v17_CodeAudit_AdamWFreeFU_BasisKernelEfficiency_4GPU完整计划.md
结果目录：results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17
Python：/home/chengshun.wang/miniconda3/envs/kan/bin/python

## 实际执行入口
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v170_code_audit_and_semantic_tests.py --out-dir results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17 --device cuda:0 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v173_full_carrier_mechanism_4gpu.py --stage mechanism-shard --out-dir results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17 --shard-count 4 --shard-index 0 --device cuda:0 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v173_full_carrier_mechanism_4gpu.py --stage mechanism-shard --out-dir results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17 --shard-count 4 --shard-index 1 --device cuda:1 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v173_full_carrier_mechanism_4gpu.py --stage mechanism-shard --out-dir results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17 --shard-count 4 --shard-index 2 --device cuda:2 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v173_full_carrier_mechanism_4gpu.py --stage mechanism-shard --out-dir results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17 --shard-count 4 --shard-index 3 --device cuda:3 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v172_basis_kernel_efficiency_breakthrough.py --out-dir results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17 --stage efficiency-shard --shard-count 4 --shard-index 0 --device cuda:0 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v172_basis_kernel_efficiency_breakthrough.py --out-dir results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17 --stage efficiency-shard --shard-count 4 --shard-index 1 --device cuda:1 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v172_basis_kernel_efficiency_breakthrough.py --out-dir results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17 --stage efficiency-shard --shard-count 4 --shard-index 2 --device cuda:2 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v172_basis_kernel_efficiency_breakthrough.py --out-dir results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17 --stage efficiency-shard --shard-count 4 --shard-index 3 --device cuda:3 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v173_full_carrier_mechanism_4gpu.py --stage horizon-shard --out-dir results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17 --shard-count 4 --shard-index 0 --device cuda:0 --data-root data --horizon-all-rows 1`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v173_full_carrier_mechanism_4gpu.py --stage horizon-shard --out-dir results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17 --shard-count 4 --shard-index 1 --device cuda:1 --data-root data --horizon-all-rows 1`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v173_full_carrier_mechanism_4gpu.py --stage horizon-shard --out-dir results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17 --shard-count 4 --shard-index 2 --device cuda:2 --data-root data --horizon-all-rows 1`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v173_full_carrier_mechanism_4gpu.py --stage horizon-shard --out-dir results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17 --shard-count 4 --shard-index 3 --device cuda:3 --data-root data --horizon-all-rows 1`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_finalize.py --out-dir results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17 --data-root data --horizon-all-rows 1`

## GPU 运行快照

| timestamp | gpu | util | memory | process_note |
|---|---:|---:|---:|---|
| 2026-06-02 15:51:57 +08 | 0 | 6 % | 693 MiB | horizon-shard active elapsed=07:47 pid=817700 |
| 2026-06-02 15:51:57 +08 | 1 | 6 % | 693 MiB | horizon-shard active elapsed=07:47 pid=817704 |
| 2026-06-02 15:51:57 +08 | 2 | 3 % | 693 MiB | horizon-shard active elapsed=07:47 pid=817707 |
| 2026-06-02 15:51:57 +08 | 3 | 9 % | 693 MiB | horizon-shard active elapsed=07:47 pid=817705 |
| 2026-06-02 15:55:47 +08 | 0 | 10 % | 693 MiB | horizon-shard active elapsed=11:37 pid=817700 |
| 2026-06-02 15:55:47 +08 | 1 | 6 % | 693 MiB | horizon-shard active elapsed=11:37 pid=817704 |
| 2026-06-02 15:55:47 +08 | 2 | 13 % | 693 MiB | horizon-shard active elapsed=11:37 pid=817707 |
| 2026-06-02 15:55:47 +08 | 3 | 16 % | 693 MiB | horizon-shard active elapsed=11:37 pid=817705 |
| 2026-06-02 16:00:03 +08 | 0 | 7 % | 693 MiB | horizon-shard active |
| 2026-06-02 16:00:03 +08 | 1 | 7 % | 693 MiB | horizon-shard active |
| 2026-06-02 16:00:03 +08 | 2 | 9 % | 693 MiB | horizon-shard active |
| 2026-06-02 16:00:03 +08 | 3 | 6 % | 693 MiB | horizon-shard active |
| 2026-06-02 16:08:36 +08 | 0 | 9 % | 693 MiB | horizon-shard active |
| 2026-06-02 16:08:36 +08 | 1 | 19 % | 693 MiB | horizon-shard active |
| 2026-06-02 16:08:36 +08 | 2 | 10 % | 693 MiB | horizon-shard active |
| 2026-06-02 16:08:36 +08 | 3 | 11 % | 693 MiB | horizon-shard active |
| 2026-06-02 16:12:04 +08 | 0 | 0 % | 0 MiB | horizon-shards complete |
| 2026-06-02 16:12:04 +08 | 1 | 1 % | 0 MiB | horizon-shards complete |
| 2026-06-02 16:12:04 +08 | 2 | 0 % | 0 MiB | horizon-shards complete |
| 2026-06-02 16:12:04 +08 | 3 | 0 % | 0 MiB | horizon-shards complete |

## 关键产物
- route: `results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17/v17_route_decision.json`
- mechanism matrix: `results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17/v17_functional_mechanism_matrix.csv`
- efficiency truth table: `results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17/v17_efficiency_truth_table.csv`
- kernel correctness: `results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17/v17_kernel_correctness.csv`
- horizon extension: `results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17/v17_horizon_extension.csv`
- command journal: `results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/corrective_v17/logs/v17_command_journal.md`

## 复现注意
- 方向来源只使用 train batch loss/gradient；LineC/tail/AUC/calibration 只作为 readback/audit。
- v17 runner 不 import v13-v16 runner；旧 artifact 只作为历史背景，不作为本次 fresh row。
- corrective_v17 使用 `--horizon-all-rows 1` 对 full 945 mechanism rows 执行 h800/h1600 fresh extension；若复现时省略该参数，将退化为 top-candidate extension。
- 本次没有把 family-specific torch repair 升级成 official fused CUDA/C++ kernel，也没有执行 S4/S5 real-transfer official gate；这些必须作为未完成项记录，不能写成 official success。
