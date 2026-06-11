# DG-KAN v17.0.1 执行日志

生成时间：2026-06-02 17:09:10 +08
计划文件：/home/chengshun.wang/DG-LCA/docs/DG-KAN_v17.0.1_完整计划_CodeAudit_AdamWFreeFU_BasisEfficiency_4GPU.md
结果目录：results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701
Python：/home/chengshun.wang/miniconda3/envs/kan/bin/python

## 实际执行入口
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_code_correctness_audit.py --out-dir results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701 --device cuda:0 --data-root data`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_4gpu_scheduler.py --stage mechanism-shard --out-dir results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701 --shard-count 4 --shard-index 0 --device cuda:0 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_4gpu_scheduler.py --stage mechanism-shard --out-dir results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701 --shard-count 4 --shard-index 1 --device cuda:1 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_4gpu_scheduler.py --stage mechanism-shard --out-dir results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701 --shard-count 4 --shard-index 2 --device cuda:2 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_4gpu_scheduler.py --stage mechanism-shard --out-dir results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701 --shard-count 4 --shard-index 3 --device cuda:3 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_basis_kernel_efficiency_matrix.py --out-dir results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701 --stage efficiency-shard --shard-count 4 --shard-index 0 --device cuda:0 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_basis_kernel_efficiency_matrix.py --out-dir results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701 --stage efficiency-shard --shard-count 4 --shard-index 1 --device cuda:1 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_basis_kernel_efficiency_matrix.py --out-dir results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701 --stage efficiency-shard --shard-count 4 --shard-index 2 --device cuda:2 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_basis_kernel_efficiency_matrix.py --out-dir results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701 --stage efficiency-shard --shard-count 4 --shard-index 3 --device cuda:3 --data-root data --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_4gpu_scheduler.py --stage horizon-shard --out-dir results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701 --shard-count 4 --shard-index 0 --device cuda:0 --data-root data --horizon-all-rows 1 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_4gpu_scheduler.py --stage horizon-shard --out-dir results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701 --shard-count 4 --shard-index 1 --device cuda:1 --data-root data --horizon-all-rows 1 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_4gpu_scheduler.py --stage horizon-shard --out-dir results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701 --shard-count 4 --shard-index 2 --device cuda:2 --data-root data --horizon-all-rows 1 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_4gpu_scheduler.py --stage horizon-shard --out-dir results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701 --shard-count 4 --shard-index 3 --device cuda:3 --data-root data --horizon-all-rows 1 --fail-on-missing-s0`
- `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v17_finalize.py --out-dir results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701 --data-root data --horizon-all-rows 1`

## GPU 运行快照

| timestamp | gpu | util | memory | process_note |
|---|---:|---:|---:|---|
| 2026-06-02 16:35:41 +08 | 0 | 8 % | 693 MiB | mechanism-shard active elapsed=00:09 pid=841484 |
| 2026-06-02 16:35:41 +08 | 1 | 11 % | 693 MiB | mechanism-shard active elapsed=00:09 pid=841490 |
| 2026-06-02 16:35:41 +08 | 2 | 4 % | 693 MiB | mechanism-shard active elapsed=00:09 pid=841493 |
| 2026-06-02 16:35:41 +08 | 3 | 6 % | 693 MiB | mechanism-shard active elapsed=00:09 pid=841491 |
| 2026-06-02 16:37:43 +08 | 0 | 0 % | 0 MiB | mechanism-shards complete |
| 2026-06-02 16:37:43 +08 | 1 | 0 % | 0 MiB | mechanism-shards complete |
| 2026-06-02 16:37:43 +08 | 2 | 0 % | 0 MiB | mechanism-shards complete |
| 2026-06-02 16:37:43 +08 | 3 | 0 % | 0 MiB | mechanism-shards complete |
| 2026-06-02 16:38:25 +08 | 0 | 0 % | 0 MiB | efficiency-shards completed before snapshot |
| 2026-06-02 16:38:25 +08 | 1 | 0 % | 0 MiB | efficiency-shards completed before snapshot |
| 2026-06-02 16:38:25 +08 | 2 | 0 % | 0 MiB | efficiency-shards completed before snapshot |
| 2026-06-02 16:38:25 +08 | 3 | 0 % | 0 MiB | efficiency-shards completed before snapshot |
| 2026-06-02 16:39:12 +08 | 0 | 8 % | 693 MiB | horizon-shard active elapsed=00:10 pid=844002 |
| 2026-06-02 16:39:12 +08 | 1 | 8 % | 693 MiB | horizon-shard active elapsed=00:10 pid=844003 |
| 2026-06-02 16:39:12 +08 | 2 | 8 % | 693 MiB | horizon-shard active elapsed=00:10 pid=844010 |
| 2026-06-02 16:39:12 +08 | 3 | 4 % | 693 MiB | horizon-shard active elapsed=00:10 pid=844011 |
| 2026-06-02 16:43:30 +08 | 0 | 7 % | 693 MiB | horizon-shard active elapsed=04:29 pid=844002 |
| 2026-06-02 16:43:30 +08 | 1 | 11 % | 693 MiB | horizon-shard active elapsed=04:29 pid=844003 |
| 2026-06-02 16:43:30 +08 | 2 | 6 % | 693 MiB | horizon-shard active elapsed=04:29 pid=844010 |
| 2026-06-02 16:43:30 +08 | 3 | 8 % | 693 MiB | horizon-shard active elapsed=04:29 pid=844011 |
| 2026-06-02 16:49:51 +08 | 0 | 10 % | 693 MiB | horizon-shard active |
| 2026-06-02 16:49:51 +08 | 1 | 8 % | 693 MiB | horizon-shard active |
| 2026-06-02 16:49:51 +08 | 2 | 6 % | 693 MiB | horizon-shard active |
| 2026-06-02 16:49:51 +08 | 3 | 4 % | 693 MiB | horizon-shard active |
| 2026-06-02 16:54:14 +08 | 0 | 7 % | 693 MiB | horizon-shard active elapsed=15:12 pid=844002 |
| 2026-06-02 16:54:14 +08 | 1 | 14 % | 693 MiB | horizon-shard active elapsed=15:12 pid=844003 |
| 2026-06-02 16:54:14 +08 | 2 | 13 % | 693 MiB | horizon-shard active elapsed=15:12 pid=844010 |
| 2026-06-02 16:54:14 +08 | 3 | 10 % | 693 MiB | horizon-shard active elapsed=15:12 pid=844011 |
| 2026-06-02 17:00:38 +08 | 0 | 13 % | 693 MiB | horizon-shard active |
| 2026-06-02 17:00:38 +08 | 1 | 7 % | 693 MiB | horizon-shard active |
| 2026-06-02 17:00:38 +08 | 2 | 11 % | 693 MiB | horizon-shard active |
| 2026-06-02 17:00:38 +08 | 3 | 8 % | 693 MiB | horizon-shard active |
| 2026-06-02 17:06:38 +08 | 0 | 0 % | 0 MiB | horizon-shards complete |
| 2026-06-02 17:06:38 +08 | 1 | 0 % | 0 MiB | horizon-shards complete |
| 2026-06-02 17:06:38 +08 | 2 | 0 % | 0 MiB | horizon-shards complete |
| 2026-06-02 17:06:38 +08 | 3 | 0 % | 0 MiB | horizon-shards complete |

## 关键产物
- route: `results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701/v17_route_decision.json`
- mechanism matrix: `results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701/v17_functional_mechanism_matrix.csv`
- efficiency truth table: `results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701/v17_efficiency_truth_table.csv`
- kernel correctness: `results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701/v17_kernel_correctness.csv`
- horizon extension: `results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701/v17_horizon_extension.csv`
- code review packet: `results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701/v17_code_review_packet.zip`
- packet manifest: `results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701/v17_code_review_packet/packet_manifest.csv`
- command journal: `results/v17_0_1_code_audit_adamw_free_fu_basis_efficiency_4gpu/official_v1701/logs/v17_command_journal.md`

## 复现注意
- 方向来源只使用 train batch loss/gradient；LineC/tail/AUC/calibration 只作为 readback/audit。
- v17 runner 不 import v13-v16 runner；旧 artifact 只作为历史背景，不作为本次 fresh row。
- 本次使用 `--horizon-all-rows 1` 对 full 945 mechanism rows 执行 h800/h1600 fresh extension；若复现时省略该参数，将退化为 top-candidate extension。
- 本次没有把 family-specific torch repair 升级成 official fused CUDA/C++ kernel，也没有执行 S4/S5 real-transfer official gate；这些必须作为未完成项记录，不能写成 official success。
