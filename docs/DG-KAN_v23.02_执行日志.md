# DG-KAN v23.02 PredictiveTrustFunctionalEdgePopulationFlow 执行日志

创建时间：2026-07-02 23:53:51 +0800

- 计划文档：`docs/DG-KAN_v23.02_PredictiveTrustFunctionalEdgePopulationFlow_完整详尽实验计划.md`
- runner：`experiments/run_v23_02_predictive_trust_functional_population_flow.py`
- 输出目录：`results/v23_02`
- 记录原则：只记录真实命令、真实 artifact、真实错误；缺失写 missing，不补造。


## 2026-07-02 23:53:51 +0800 part-b failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-b --device cuda:0`

Files: results/v23_02/part_b_history_boundary_lock.csv; results/v23_02/part_b_next_actions_for_codex.json


## 2026-07-02 23:53:52 +0800 part-a failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-a --device cuda:0`

Files: results/v23_02/part_a_identity_control_readiness_matrix.csv; results/v23_02/part_a_next_actions_for_codex.json


## 2026-07-02 23:56:24 +0800 part-b passed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-b --device cuda:0`

Files: results/v23_02/part_b_history_boundary_lock.csv; results/v23_02/part_b_next_actions_for_codex.json


## 2026-07-02 23:56:25 +0800 part-a passed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-a --device cuda:0`

Files: results/v23_02/part_a_identity_control_readiness_matrix.csv; results/v23_02/part_a_next_actions_for_codex.json


## 2026-07-02 23:57:48 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 2 --shard-index 0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0 --part-c-seed-count 1 --part-c-gate-floors 0.3 --train-steps 20 --synthetic-train-size 64 --synthetic-guard-size 64 --population-grad-examples 2 --debt-grad-examples 2 --stat-warmup-steps 5`

Files: results/v23_02_runtime_smoke/part_c_taskwise_c2_matrix_shard0_of_2.csv

Note: rows=2


## 2026-07-02 23:57:48 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 2 --shard-index 1 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0 --part-c-seed-count 1 --part-c-gate-floors 0.3 --train-steps 20 --synthetic-train-size 64 --synthetic-guard-size 64 --population-grad-examples 2 --debt-grad-examples 2 --stat-warmup-steps 5`

Files: results/v23_02_runtime_smoke/part_c_taskwise_c2_matrix_shard1_of_2.csv

Note: rows=2


## 2026-07-02 23:58:02 +0800 part-c-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c-merge --device cpu --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0 --part-c-seed-count 1 --part-c-gate-floors 0.3`

Files: results/v23_02_runtime_smoke/part_c_taskwise_c2_matrix.csv; results/v23_02_runtime_smoke/part_c_next_actions_for_codex.json


## 2026-07-03 00:08:10 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 2 --shard-index 0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C0_AdamW_control,C1_FunctionalGram_AdamW_s0_no_Qpop,C2_FunctionalGram_DiagonalSNR_s0,C3_FunctionalGram_BlockSNR_layer_s0,C4_FunctionalGram_BlockSNR_degree_s0,C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0,C8_FunctionalGram_SameComputeNoOp_s0,C9_FunctionalGram_s1_derivative_diagnostic --part-c-seed-count 5 --part-c-gate-floors 0.3,0.4 --train-steps 200 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20`

Files: results/v23_02/part_c_taskwise_c2_matrix_shard0_of_2.csv

Note: rows=90


## 2026-07-03 00:08:11 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 2 --shard-index 1 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C0_AdamW_control,C1_FunctionalGram_AdamW_s0_no_Qpop,C2_FunctionalGram_DiagonalSNR_s0,C3_FunctionalGram_BlockSNR_layer_s0,C4_FunctionalGram_BlockSNR_degree_s0,C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0,C8_FunctionalGram_SameComputeNoOp_s0,C9_FunctionalGram_s1_derivative_diagnostic --part-c-seed-count 5 --part-c-gate-floors 0.3,0.4 --train-steps 200 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20`

Files: results/v23_02/part_c_taskwise_c2_matrix_shard1_of_2.csv

Note: rows=90


## 2026-07-03 00:08:54 +0800 part-c-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c-merge --device cpu --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C0_AdamW_control,C1_FunctionalGram_AdamW_s0_no_Qpop,C2_FunctionalGram_DiagonalSNR_s0,C3_FunctionalGram_BlockSNR_layer_s0,C4_FunctionalGram_BlockSNR_degree_s0,C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0,C8_FunctionalGram_SameComputeNoOp_s0,C9_FunctionalGram_s1_derivative_diagnostic --part-c-seed-count 5 --part-c-gate-floors 0.3,0.4 --train-steps 200 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20`

Files: results/v23_02/part_c_taskwise_c2_matrix.csv; results/v23_02/part_c_next_actions_for_codex.json


## 2026-07-03 00:11:21 +0800 part-c-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c-merge --device cpu --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C0_AdamW_control,C1_FunctionalGram_AdamW_s0_no_Qpop,C2_FunctionalGram_DiagonalSNR_s0,C3_FunctionalGram_BlockSNR_layer_s0,C4_FunctionalGram_BlockSNR_degree_s0,C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0,C8_FunctionalGram_SameComputeNoOp_s0,C9_FunctionalGram_s1_derivative_diagnostic --part-c-seed-count 5 --part-c-gate-floors 0.3,0.4 --train-steps 200 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20`

Files: results/v23_02/part_c_taskwise_c2_matrix.csv; results/v23_02/part_c_taskwise_c2_task_summary.csv; results/v23_02/part_c_taskwise_c2_group_summary.csv; results/v23_02/part_c_next_actions_for_codex.json


## 2026-07-03 00:26:34 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 2 --shard-index 0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C3_FunctionalGram_BlockSNR_layer_s0,C4_FunctionalGram_BlockSNR_degree_s0,C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0 --part-c-seed-count 5 --part-c-gate-floors 0.32,0.35,0.37 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20`

Files: results/v23_02_repair_c_floor032035037_steps240/part_c_taskwise_c2_matrix_shard0_of_2.csv

Note: rows=60


## 2026-07-03 00:26:38 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 2 --shard-index 1 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C3_FunctionalGram_BlockSNR_layer_s0,C4_FunctionalGram_BlockSNR_degree_s0,C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0 --part-c-seed-count 5 --part-c-gate-floors 0.32,0.35,0.37 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20`

Files: results/v23_02_repair_c_floor032035037_steps240/part_c_taskwise_c2_matrix_shard1_of_2.csv

Note: rows=60


## 2026-07-03 00:27:00 +0800 part-c-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c-merge --device cpu --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C3_FunctionalGram_BlockSNR_layer_s0,C4_FunctionalGram_BlockSNR_degree_s0,C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0 --part-c-seed-count 5 --part-c-gate-floors 0.32,0.35,0.37 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20`

Files: results/v23_02_repair_c_floor032035037_steps240/part_c_taskwise_c2_matrix.csv; results/v23_02_repair_c_floor032035037_steps240/part_c_taskwise_c2_task_summary.csv; results/v23_02_repair_c_floor032035037_steps240/part_c_taskwise_c2_group_summary.csv; results/v23_02_repair_c_floor032035037_steps240/part_c_next_actions_for_codex.json


## 2026-07-03 00:29:20 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 1 --shard-index 0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25,C7_FunctionalGram_RandomMatchedGate_s0 --part-c-seed-count 1 --part-c-gate-floors 0.3 --train-steps 20 --synthetic-train-size 64 --synthetic-guard-size 64 --population-grad-examples 2 --debt-grad-examples 2 --stat-warmup-steps 5`

Files: results/v23_02_sobolev_smoke/part_c_taskwise_c2_matrix_shard0_of_1.csv

Note: rows=4


## 2026-07-03 00:29:43 +0800 part-c-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c-merge --device cpu --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25,C7_FunctionalGram_RandomMatchedGate_s0 --part-c-seed-count 1 --part-c-gate-floors 0.3`

Files: results/v23_02_sobolev_smoke/part_c_taskwise_c2_matrix.csv; results/v23_02_sobolev_smoke/part_c_taskwise_c2_task_summary.csv; results/v23_02_sobolev_smoke/part_c_taskwise_c2_group_summary.csv; results/v23_02_sobolev_smoke/part_c_next_actions_for_codex.json


## 2026-07-03 01:02:51 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 2 --shard-index 0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25,C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5,C12_FunctionalSobolev_DegreeEdgebankSNR_s1,C13_FunctionalSobolev_RandomMatchedGate_s0p25,C14_FunctionalSobolev_RandomMatchedGate_s0p5,C15_FunctionalSobolev_RandomMatchedGate_s1 --part-c-seed-count 5 --part-c-gate-floors 0.3,0.35,0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20`

Files: results/v23_02_repair_c_sobolev_qpop_steps240/part_c_taskwise_c2_matrix_shard0_of_2.csv

Note: rows=90


## 2026-07-03 01:02:51 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 2 --shard-index 1 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25,C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5,C12_FunctionalSobolev_DegreeEdgebankSNR_s1,C13_FunctionalSobolev_RandomMatchedGate_s0p25,C14_FunctionalSobolev_RandomMatchedGate_s0p5,C15_FunctionalSobolev_RandomMatchedGate_s1 --part-c-seed-count 5 --part-c-gate-floors 0.3,0.35,0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20`

Files: results/v23_02_repair_c_sobolev_qpop_steps240/part_c_taskwise_c2_matrix_shard1_of_2.csv

Note: rows=90


## 2026-07-03 01:03:34 +0800 part-c-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c-merge --device cpu --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-schemes C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25,C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5,C12_FunctionalSobolev_DegreeEdgebankSNR_s1,C13_FunctionalSobolev_RandomMatchedGate_s0p25,C14_FunctionalSobolev_RandomMatchedGate_s0p5,C15_FunctionalSobolev_RandomMatchedGate_s1 --part-c-seed-count 5 --part-c-gate-floors 0.3,0.35,0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20`

Files: results/v23_02_repair_c_sobolev_qpop_steps240/part_c_taskwise_c2_matrix.csv; results/v23_02_repair_c_sobolev_qpop_steps240/part_c_taskwise_c2_task_summary.csv; results/v23_02_repair_c_sobolev_qpop_steps240/part_c_taskwise_c2_group_summary.csv; results/v23_02_repair_c_sobolev_qpop_steps240/part_c_next_actions_for_codex.json


## 2026-07-03 01:10:22 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control --part-f-parts F3,F5 --part-f-c2-seed-count 1 --part-f-seed-count 1 --train-steps 20 --synthetic-train-size 64 --synthetic-guard-size 64 --population-grad-examples 2 --debt-grad-examples 2 --stat-warmup-steps 5 --finite-step-guard-examples 64`

Files: results/v23_02_direct_f_smoke/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=9; part_c_role=sanity


## 2026-07-03 01:10:41 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control --part-f-parts F3,F5 --part-f-c2-seed-count 1 --part-f-seed-count 1 --train-steps 20 --finite-step-guard-examples 64`

Files: results/v23_02_direct_f_smoke/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_direct_f_smoke/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_direct_f_smoke/part_f_next_actions_for_codex.json


## 2026-07-03 02:07:08 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 2 --shard-index 1 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0,E7_FunctionalGram_DegreeEdgebankSNR_s0p25,E7_FunctionalGram_DegreeEdgebankSNR_s0p5 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control,F7_cadence_finite_step_trust --part-f-parts F3,F5 --part-f-c2-seed-count 5 --part-f-seed-count 5 --train-steps 160 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20 --finite-step-guard-examples 128`

Files: results/v23_02_direct_f_diag5/part_f_predictive_trust_positive_control_matrix_shard1_of_2.csv

Note: rows=90; part_c_role=sanity


## 2026-07-03 02:09:03 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 2 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0,E7_FunctionalGram_DegreeEdgebankSNR_s0p25,E7_FunctionalGram_DegreeEdgebankSNR_s0p5 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control,F7_cadence_finite_step_trust --part-f-parts F3,F5 --part-f-c2-seed-count 5 --part-f-seed-count 5 --train-steps 160 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20 --finite-step-guard-examples 128`

Files: results/v23_02_direct_f_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_2.csv

Note: rows=90; part_c_role=sanity


## 2026-07-03 02:09:26 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0,E7_FunctionalGram_DegreeEdgebankSNR_s0p25,E7_FunctionalGram_DegreeEdgebankSNR_s0p5 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control,F7_cadence_finite_step_trust --part-f-parts F3,F5 --part-f-c2-seed-count 5 --part-f-seed-count 5 --train-steps 160 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20 --finite-step-guard-examples 128`

Files: results/v23_02_direct_f_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_direct_f_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_direct_f_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 02:24:45 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 2 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0,E7_FunctionalGram_DegreeEdgebankSNR_s0p5 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F3_finite_step_trust_tail_only,F4_finite_step_trust_tail_margin --part-f-parts F3,F5 --part-f-c2-seed-count 5 --part-f-seed-count 5 --train-steps 160 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20 --finite-step-guard-examples 128`

Files: results/v23_02_direct_f_repair_tail/part_f_predictive_trust_positive_control_matrix_shard0_of_2.csv

Note: rows=30; part_c_role=sanity


## 2026-07-03 02:24:57 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 2 --shard-index 1 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0,E7_FunctionalGram_DegreeEdgebankSNR_s0p5 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F3_finite_step_trust_tail_only,F4_finite_step_trust_tail_margin --part-f-parts F3,F5 --part-f-c2-seed-count 5 --part-f-seed-count 5 --train-steps 160 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20 --finite-step-guard-examples 128`

Files: results/v23_02_direct_f_repair_tail/part_f_predictive_trust_positive_control_matrix_shard1_of_2.csv

Note: rows=30; part_c_role=sanity


## 2026-07-03 02:25:19 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0,E7_FunctionalGram_DegreeEdgebankSNR_s0p5 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F3_finite_step_trust_tail_only,F4_finite_step_trust_tail_margin --part-f-parts F3,F5 --part-f-c2-seed-count 5 --part-f-seed-count 5 --train-steps 160 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --debt-grad-examples 4 --stat-warmup-steps 20 --finite-step-guard-examples 128`

Files: results/v23_02_direct_f_repair_tail/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_direct_f_repair_tail/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_direct_f_repair_tail/part_f_next_actions_for_codex.json


## 2026-07-03 02:28:43 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F8_predictive_scaled_cadence_scale0125,F8_predictive_scaled_cadence_scale025,F8_predictive_scaled_cadence_scale05 --part-f-parts F5 --part-f-c2-seed-count 1 --part-f-seed-count 1 --train-steps 20 --synthetic-train-size 64 --synthetic-guard-size 64 --population-grad-examples 2 --debt-grad-examples 2 --stat-warmup-steps 5 --finite-step-guard-examples 64`

Files: results/v23_02_predictive_scaled_smoke/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=3; part_c_role=sanity


## 2026-07-03 02:29:06 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F8_predictive_scaled_cadence_scale0125,F8_predictive_scaled_cadence_scale025,F8_predictive_scaled_cadence_scale05 --part-f-parts F5 --part-f-c2-seed-count 1 --part-f-seed-count 1 --train-steps 20 --finite-step-guard-examples 64`

Files: results/v23_02_predictive_scaled_smoke/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_predictive_scaled_smoke/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_predictive_scaled_smoke/part_f_next_actions_for_codex.json


## 2026-07-03 02:30:28 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F8_predictive_scaled_cadence_scale0125,F8_predictive_scaled_cadence_scale025,F8_predictive_scaled_cadence_scale05 --part-f-parts F5 --part-f-c2-seed-count 1 --part-f-seed-count 1 --train-steps 20 --synthetic-train-size 64 --synthetic-guard-size 64 --population-grad-examples 2 --debt-grad-examples 2 --stat-warmup-steps 5 --finite-step-guard-examples 64`

Files: results/v23_02_predictive_scaled_smoke/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=3; part_c_role=sanity


## 2026-07-03 02:30:53 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F8_predictive_scaled_cadence_scale0125,F8_predictive_scaled_cadence_scale025,F8_predictive_scaled_cadence_scale05 --part-f-parts F5 --part-f-c2-seed-count 1 --part-f-seed-count 1 --train-steps 20 --finite-step-guard-examples 64`

Files: results/v23_02_predictive_scaled_smoke/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_predictive_scaled_smoke/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_predictive_scaled_smoke/part_f_next_actions_for_codex.json


## 2026-07-03 02:31:38 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F8_predictive_scaled_cadence_scale0125,F8_predictive_scaled_cadence_scale025 --part-f-parts F5 --part-f-seed-count 1 --train-steps 40 --synthetic-train-size 64 --synthetic-guard-size 64 --population-grad-examples 2 --debt-grad-examples 2 --stat-warmup-steps 5 --finite-step-guard-examples 64 --finite-step-cadence 2`

Files: results/v23_02_predictive_scaled_cadence2_smoke/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=2; part_c_role=sanity


## 2026-07-03 02:31:39 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F8_predictive_scaled_cadence_scale0125,F8_predictive_scaled_cadence_scale025 --part-f-parts F5 --part-f-seed-count 1 --train-steps 40 --finite-step-cadence 2`

Files: results/v23_02_predictive_scaled_cadence2_smoke/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_predictive_scaled_cadence2_smoke/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_predictive_scaled_cadence2_smoke/part_f_next_actions_for_codex.json


## 2026-07-03 02:33:10 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F9_predictive_scaled_cadence_veto_scale0125,F9_predictive_scaled_cadence_veto_scale025 --part-f-parts F5 --part-f-seed-count 1 --train-steps 40 --synthetic-train-size 64 --synthetic-guard-size 64 --population-grad-examples 2 --debt-grad-examples 4 --stat-warmup-steps 5 --finite-step-guard-examples 64 --finite-step-cadence 2`

Files: results/v23_02_predictive_scaled_veto_smoke/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=2; part_c_role=sanity


## 2026-07-03 02:33:12 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F9_predictive_scaled_cadence_veto_scale0125,F9_predictive_scaled_cadence_veto_scale025 --part-f-parts F5 --part-f-seed-count 1 --train-steps 40 --finite-step-cadence 2`

Files: results/v23_02_predictive_scaled_veto_smoke/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_predictive_scaled_veto_smoke/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_predictive_scaled_veto_smoke/part_f_next_actions_for_codex.json


## 2026-07-03 02:34:14 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 1 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75`

Files: results/v23_02_exact_densegrid_smoke/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=1; part_c_role=sanity


## 2026-07-03 02:34:16 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 1 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75`

Files: results/v23_02_exact_densegrid_smoke/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_exact_densegrid_smoke/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_exact_densegrid_smoke/part_f_next_actions_for_codex.json


## 2026-07-03 02:41:19 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F10_predictive_slack_scale005,F10_predictive_slack_scale00625,F10_predictive_slack_scale0075,F10_predictive_slack_scale01,F11_predictive_slack_veto_scale0075,F11_predictive_slack_veto_scale01 --part-f-parts F5 --part-f-seed-count 1 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-cadence 999`

Files: results/v23_02_predictive_slack_smoke/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=6; part_c_role=sanity


## 2026-07-03 02:41:20 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F10_predictive_slack_scale005,F10_predictive_slack_scale00625,F10_predictive_slack_scale0075,F10_predictive_slack_scale01,F11_predictive_slack_veto_scale0075,F11_predictive_slack_veto_scale01 --part-f-parts F5 --part-f-seed-count 1 --train-steps 80 --finite-step-cadence 999`

Files: results/v23_02_predictive_slack_smoke/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_predictive_slack_smoke/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_predictive_slack_smoke/part_f_next_actions_for_codex.json


## 2026-07-03 02:42:49 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 1 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 4 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.006`

Files: results/v23_02_edge_lr_smoke/part_f_predictive_trust_positive_control_matrix_shard1_of_4.csv

Note: rows=1; part_c_role=sanity


## 2026-07-03 02:42:49 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 3 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 4 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.002`

Files: results/v23_02_edge_lr_smoke/part_f_predictive_trust_positive_control_matrix_shard3_of_4.csv

Note: rows=1; part_c_role=sanity


## 2026-07-03 02:43:03 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 2 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 4 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_edge_lr_smoke/part_f_predictive_trust_positive_control_matrix_shard2_of_4.csv

Note: rows=1; part_c_role=sanity


## 2026-07-03 02:43:16 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 4 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.01`

Files: results/v23_02_edge_lr_smoke/part_f_predictive_trust_positive_control_matrix_shard0_of_4.csv

Note: rows=1; part_c_role=sanity


## 2026-07-03 02:44:09 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 3 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_edge_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard3_of_4.csv

Note: rows=1; part_c_role=sanity


## 2026-07-03 02:44:09 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 1 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_edge_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard1_of_4.csv

Note: rows=1; part_c_role=sanity


## 2026-07-03 02:44:22 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 2 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_edge_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard2_of_4.csv

Note: rows=1; part_c_role=sanity


## 2026-07-03 02:44:26 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_edge_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_4.csv

Note: rows=2; part_c_role=sanity


## 2026-07-03 02:44:27 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_edge_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_edge_lr004_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_edge_lr004_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 02:50:48 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F12_proactive_corrected_veto_finite_step_all,F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 1 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 8 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_proactive_veto_smoke/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=2; part_c_role=sanity


## 2026-07-03 02:50:50 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F12_proactive_corrected_veto_finite_step_all,F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 1 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_proactive_veto_smoke/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_proactive_veto_smoke/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_proactive_veto_smoke/part_f_next_actions_for_codex.json


## 2026-07-03 02:55:13 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F13_debtreg0005_finite_step_all,F13_debtreg001_finite_step_all,F13_debtreg002_finite_step_all,F13_debtreg005_finite_step_all,F13_debtreg01_finite_step_all,F13_debtreg001_margin_only_finite_step_all,F13_debtreg005_margin_only_finite_step_all,F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 1 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_debtreg_smoke/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=8; part_c_role=sanity


## 2026-07-03 02:55:15 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F13_debtreg0005_finite_step_all,F13_debtreg001_finite_step_all,F13_debtreg002_finite_step_all,F13_debtreg005_finite_step_all,F13_debtreg01_finite_step_all,F13_debtreg001_margin_only_finite_step_all,F13_debtreg005_margin_only_finite_step_all,F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 1 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_debtreg_smoke/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_debtreg_smoke/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_debtreg_smoke/part_f_next_actions_for_codex.json


## 2026-07-03 02:56:28 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 2 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F13_debtreg002_finite_step_all,F13_debtreg001_margin_only_finite_step_all,F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_debtreg002_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard2_of_4.csv

Note: rows=4; part_c_role=sanity


## 2026-07-03 02:56:38 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 3 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F13_debtreg002_finite_step_all,F13_debtreg001_margin_only_finite_step_all,F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_debtreg002_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard3_of_4.csv

Note: rows=3; part_c_role=sanity


## 2026-07-03 02:56:44 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 1 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F13_debtreg002_finite_step_all,F13_debtreg001_margin_only_finite_step_all,F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_debtreg002_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard1_of_4.csv

Note: rows=4; part_c_role=sanity


## 2026-07-03 02:56:55 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F13_debtreg002_finite_step_all,F13_debtreg001_margin_only_finite_step_all,F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_debtreg002_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_4.csv

Note: rows=4; part_c_role=sanity


## 2026-07-03 02:56:56 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F13_debtreg002_finite_step_all,F13_debtreg001_margin_only_finite_step_all,F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004`

Files: results/v23_02_debtreg002_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_debtreg002_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_debtreg002_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 02:58:09 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_marginreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 1.0 --debt-regularization-components margin10`

Files: results/v23_02_marginreg_w1p0_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 02:58:09 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_marginreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components margin10`

Files: results/v23_02_marginreg_w0p1_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 02:58:09 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_marginreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 3.0 --debt-regularization-components margin10`

Files: results/v23_02_marginreg_w3p0_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 02:58:10 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_marginreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.3 --debt-regularization-components margin10`

Files: results/v23_02_marginreg_w0p3_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 02:58:11 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_marginreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 1.0 --debt-regularization-components margin10`

Files: results/v23_02_marginreg_w1p0_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_marginreg_w1p0_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_marginreg_w1p0_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 02:58:11 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_marginreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components margin10`

Files: results/v23_02_marginreg_w0p1_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_marginreg_w0p1_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_marginreg_w0p1_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 02:58:11 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_marginreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 3.0 --debt-regularization-components margin10`

Files: results/v23_02_marginreg_w3p0_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_marginreg_w3p0_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_marginreg_w3p0_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 02:58:12 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_marginreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.3 --debt-regularization-components margin10`

Files: results/v23_02_marginreg_w0p3_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_marginreg_w0p3_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_marginreg_w0p3_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:01:30 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F2_finite_step_trust_all,F7_cadence_finite_step_trust,F8_predictive_scaled_cadence_scale025 --part-f-parts F5 --part-f-seed-count 1 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.02 --finite-step-cadence 5`

Files: results/v23_02_margin_signfix_smoke/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=3; part_c_role=sanity


## 2026-07-03 03:01:32 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F2_finite_step_trust_all,F7_cadence_finite_step_trust,F8_predictive_scaled_cadence_scale025 --part-f-parts F5 --part-f-seed-count 1 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.02 --finite-step-cadence 5`

Files: results/v23_02_margin_signfix_smoke/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_margin_signfix_smoke/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_margin_signfix_smoke/part_f_next_actions_for_codex.json


## 2026-07-03 03:02:18 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_tailreg_finite_step_all --part-f-parts F5 --part-f-seed-count 4 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.02 --debt-regularization-weight 0.02 --debt-regularization-components tail95,tail99`

Files: results/v23_02_tailreg_grid_smoke/part_f_predictive_trust_positive_control_matrix_shard0_of_4.csv

Note: rows=1; part_c_role=sanity


## 2026-07-03 03:02:18 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 2 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_tailreg_finite_step_all --part-f-parts F5 --part-f-seed-count 4 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.02 --debt-regularization-components tail95,tail99`

Files: results/v23_02_tailreg_grid_smoke/part_f_predictive_trust_positive_control_matrix_shard2_of_4.csv

Note: rows=1; part_c_role=sanity


## 2026-07-03 03:02:36 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 3 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_tailreg_finite_step_all --part-f-parts F5 --part-f-seed-count 4 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.10 --debt-regularization-components tail95,tail99`

Files: results/v23_02_tailreg_grid_smoke/part_f_predictive_trust_positive_control_matrix_shard3_of_4.csv

Note: rows=1; part_c_role=sanity


## 2026-07-03 03:02:47 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 1 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_tailreg_finite_step_all --part-f-parts F5 --part-f-seed-count 4 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.02 --debt-regularization-weight 0.10 --debt-regularization-components tail95,tail99`

Files: results/v23_02_tailreg_grid_smoke/part_f_predictive_trust_positive_control_matrix_shard1_of_4.csv

Note: rows=1; part_c_role=sanity


## 2026-07-03 03:03:56 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 3 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_tailreg_finite_step_all,F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components tail95,tail99`

Files: results/v23_02_tailreg_w0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard3_of_4.csv

Note: rows=2; part_c_role=sanity


## 2026-07-03 03:04:02 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_tailreg_finite_step_all,F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components tail95,tail99`

Files: results/v23_02_tailreg_w0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_4.csv

Note: rows=3; part_c_role=sanity


## 2026-07-03 03:04:06 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 2 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_tailreg_finite_step_all,F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components tail95,tail99`

Files: results/v23_02_tailreg_w0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard2_of_4.csv

Note: rows=2; part_c_role=sanity


## 2026-07-03 03:04:09 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 1 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_tailreg_finite_step_all,F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components tail95,tail99`

Files: results/v23_02_tailreg_w0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard1_of_4.csv

Note: rows=3; part_c_role=sanity


## 2026-07-03 03:04:11 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F14_cli_tailreg_finite_step_all,F2_finite_step_trust_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components tail95,tail99`

Files: results/v23_02_tailreg_w0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_tailreg_w0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_tailreg_w0p1_lr004_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:06:12 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_debtreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.3 --debt-regularization-components Brier,ECE,tail95,tail99`

Files: results/v23_02_btail_w0p3_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:06:13 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_debtreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components ECE,tail95,tail99`

Files: results/v23_02_etail_w0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:06:13 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_debtreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.3 --debt-regularization-components ECE,tail95,tail99`

Files: results/v23_02_etail_w0p3_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:06:14 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_debtreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components Brier,ECE,tail95,tail99`

Files: results/v23_02_btail_w0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:06:14 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_debtreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.3 --debt-regularization-components Brier,ECE,tail95,tail99`

Files: results/v23_02_btail_w0p3_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_btail_w0p3_lr004_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_btail_w0p3_lr004_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:06:15 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_debtreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components ECE,tail95,tail99`

Files: results/v23_02_etail_w0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_etail_w0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_etail_w0p1_lr004_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:06:15 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_debtreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.3 --debt-regularization-components ECE,tail95,tail99`

Files: results/v23_02_etail_w0p3_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_etail_w0p3_lr004_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_etail_w0p3_lr004_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:06:16 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_debtreg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components Brier,ECE,tail95,tail99`

Files: results/v23_02_btail_w0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_btail_w0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_btail_w0p1_lr004_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:08:39 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.3 --debt-regularization-components ECE,tail95,tail99`

Files: results/v23_02_etail_cliw0p3_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:08:41 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components ECE,tail95,tail99`

Files: results/v23_02_etail_cliw0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:08:41 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.3 --debt-regularization-components ECE,tail95,tail99`

Files: results/v23_02_etail_cliw0p3_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_etail_cliw0p3_lr004_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_etail_cliw0p3_lr004_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:08:42 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components ECE,tail95,tail99`

Files: results/v23_02_etail_cliw0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_etail_cliw0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_etail_cliw0p1_lr004_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:08:43 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components Brier,ECE,tail95,tail99`

Files: results/v23_02_btail_cliw0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:08:45 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.1 --debt-regularization-components Brier,ECE,tail95,tail99`

Files: results/v23_02_btail_cliw0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_btail_cliw0p1_lr004_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_btail_cliw0p1_lr004_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:08:54 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.3 --debt-regularization-components Brier,ECE,tail95,tail99`

Files: results/v23_02_btail_cliw0p3_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:08:56 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.004 --debt-regularization-weight 0.3 --debt-regularization-components Brier,ECE,tail95,tail99`

Files: results/v23_02_btail_cliw0p3_lr004_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_btail_cliw0p3_lr004_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_btail_cliw0p3_lr004_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:11:00 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.001 --debt-regularization-weight 0.3 --debt-regularization-components ECE,tail95,tail99`

Files: results/v23_02_etail_cliw0p3_lr0p001_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:11:02 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.001 --debt-regularization-weight 0.3 --debt-regularization-components Brier,ECE,tail95,tail99`

Files: results/v23_02_btail_cliw0p3_lr0p001_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:11:02 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.001 --debt-regularization-weight 0.3 --debt-regularization-components ECE,tail95,tail99`

Files: results/v23_02_etail_cliw0p3_lr0p001_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_etail_cliw0p3_lr0p001_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_etail_cliw0p3_lr0p001_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:11:03 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.001 --debt-regularization-weight 0.3 --debt-regularization-components Brier,ECE,tail95,tail99`

Files: results/v23_02_btail_cliw0p3_lr0p001_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_btail_cliw0p3_lr0p001_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_btail_cliw0p3_lr0p001_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:11:04 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.002 --debt-regularization-weight 0.3 --debt-regularization-components ECE,tail95,tail99`

Files: results/v23_02_etail_cliw0p3_lr0p002_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:11:05 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.002 --debt-regularization-weight 0.3 --debt-regularization-components ECE,tail95,tail99`

Files: results/v23_02_etail_cliw0p3_lr0p002_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_etail_cliw0p3_lr0p002_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_etail_cliw0p3_lr0p002_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:11:31 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.002 --debt-regularization-weight 0.3 --debt-regularization-components Brier,ECE,tail95,tail99`

Files: results/v23_02_btail_cliw0p3_lr0p002_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:11:33 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_cli_reg_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.002 --debt-regularization-weight 0.3 --debt-regularization-components Brier,ECE,tail95,tail99`

Files: results/v23_02_btail_cliw0p3_lr0p002_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_btail_cliw0p3_lr0p002_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_btail_cliw0p3_lr0p002_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:17:06 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_direct_f_signfix_etailreg_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_4.csv

Note: rows=15; part_c_role=sanity


## 2026-07-03 03:17:25 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 2 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_direct_f_signfix_etailreg_diag5/part_f_predictive_trust_positive_control_matrix_shard2_of_4.csv

Note: rows=15; part_c_role=sanity


## 2026-07-03 03:18:22 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 1 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_direct_f_signfix_etailreg_diag5/part_f_predictive_trust_positive_control_matrix_shard1_of_4.csv

Note: rows=15; part_c_role=sanity


## 2026-07-03 03:18:51 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 3 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_direct_f_signfix_etailreg_diag5/part_f_predictive_trust_positive_control_matrix_shard3_of_4.csv

Note: rows=15; part_c_role=sanity


## 2026-07-03 03:18:53 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_direct_f_signfix_etailreg_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_direct_f_signfix_etailreg_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_direct_f_signfix_etailreg_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:33:21 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_direct_f_signfix_etailreg_seed15/part_f_predictive_trust_positive_control_matrix_shard0_of_4.csv

Note: rows=45; part_c_role=sanity


## 2026-07-03 03:35:08 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 2 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_direct_f_signfix_etailreg_seed15/part_f_predictive_trust_positive_control_matrix_shard2_of_4.csv

Note: rows=45; part_c_role=sanity


## 2026-07-03 03:35:44 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 1 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_direct_f_signfix_etailreg_seed15/part_f_predictive_trust_positive_control_matrix_shard1_of_4.csv

Note: rows=45; part_c_role=sanity


## 2026-07-03 03:35:48 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 3 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_direct_f_signfix_etailreg_seed15/part_f_predictive_trust_positive_control_matrix_shard3_of_4.csv

Note: rows=45; part_c_role=sanity


## 2026-07-03 03:35:50 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F2_finite_step_trust_all,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_direct_f_signfix_etailreg_seed15/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_direct_f_signfix_etailreg_seed15/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_direct_f_signfix_etailreg_seed15/part_f_next_actions_for_codex.json


## 2026-07-03 03:36:50 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.0001`

Files: results/v23_02_etail_lr0p0001_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:36:52 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.0001`

Files: results/v23_02_etail_lr0p0001_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_etail_lr0p0001_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_etail_lr0p0001_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:37:04 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00025`

Files: results/v23_02_etail_lr0p00025_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:37:06 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00025`

Files: results/v23_02_etail_lr0p00025_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_etail_lr0p00025_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_etail_lr0p00025_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:37:16 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.0005`

Files: results/v23_02_etail_lr0p0005_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:37:18 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.0005`

Files: results/v23_02_etail_lr0p0005_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_etail_lr0p0005_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_etail_lr0p0005_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:37:30 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00075`

Files: results/v23_02_etail_lr0p00075_f5_diag5/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=5; part_c_role=sanity


## 2026-07-03 03:37:31 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 5 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00075`

Files: results/v23_02_etail_lr0p00075_f5_diag5/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_etail_lr0p00075_f5_diag5/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_etail_lr0p00075_f5_diag5/part_f_next_actions_for_codex.json


## 2026-07-03 03:49:29 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00075`

Files: results/v23_02_direct_f_signfix_etailreg_lr00075_seed15/part_f_predictive_trust_positive_control_matrix_shard0_of_4.csv

Note: rows=34; part_c_role=sanity


## 2026-07-03 03:49:43 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 3 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00075`

Files: results/v23_02_direct_f_signfix_etailreg_lr00075_seed15/part_f_predictive_trust_positive_control_matrix_shard3_of_4.csv

Note: rows=33; part_c_role=sanity


## 2026-07-03 03:50:56 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 2 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00075`

Files: results/v23_02_direct_f_signfix_etailreg_lr00075_seed15/part_f_predictive_trust_positive_control_matrix_shard2_of_4.csv

Note: rows=34; part_c_role=sanity


## 2026-07-03 03:50:57 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 1 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00075`

Files: results/v23_02_direct_f_signfix_etailreg_lr00075_seed15/part_f_predictive_trust_positive_control_matrix_shard1_of_4.csv

Note: rows=34; part_c_role=sanity


## 2026-07-03 03:50:59 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00075`

Files: results/v23_02_direct_f_signfix_etailreg_lr00075_seed15/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_direct_f_signfix_etailreg_lr00075_seed15/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_direct_f_signfix_etailreg_lr00075_seed15/part_f_next_actions_for_codex.json


## 2026-07-03 03:55:50 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 2 --shard-index 1 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.0001`

Files: results/v23_02_f5only_lr0p0001_seed15/part_f_predictive_trust_positive_control_matrix_shard1_of_2.csv

Note: rows=22; part_c_role=sanity


## 2026-07-03 03:56:08 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 2 --shard-index 1 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00025`

Files: results/v23_02_f5only_lr0p00025_seed15/part_f_predictive_trust_positive_control_matrix_shard1_of_2.csv

Note: rows=22; part_c_role=sanity


## 2026-07-03 03:56:14 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 2 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.0001`

Files: results/v23_02_f5only_lr0p0001_seed15/part_f_predictive_trust_positive_control_matrix_shard0_of_2.csv

Note: rows=23; part_c_role=sanity


## 2026-07-03 03:57:18 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 2 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00025`

Files: results/v23_02_f5only_lr0p00025_seed15/part_f_predictive_trust_positive_control_matrix_shard0_of_2.csv

Note: rows=23; part_c_role=sanity


## 2026-07-03 03:57:20 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 15 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00025`

Files: results/v23_02_f5only_lr0p00025_seed15/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_f5only_lr0p00025_seed15/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_f5only_lr0p00025_seed15/part_f_next_actions_for_codex.json


## 2026-07-03 03:57:21 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 15 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.0001`

Files: results/v23_02_f5only_lr0p0001_seed15/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_f5only_lr0p0001_seed15/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_f5only_lr0p0001_seed15/part_f_next_actions_for_codex.json


## 2026-07-03 04:01:25 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 1 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 20 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_etailreg0p3_tries20_f5_seed15/part_f_predictive_trust_positive_control_matrix_shard0_of_1.csv

Note: rows=15; part_c_role=sanity


## 2026-07-03 04:01:41 +0800 part-f-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-safety-schemes F15_etailreg0p3_finite_step_all --part-f-parts F5 --part-f-seed-count 15 --train-steps 80 --finite-step-tries 20 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_etailreg0p3_tries20_f5_seed15/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_etailreg0p3_tries20_f5_seed15/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_etailreg0p3_tries20_f5_seed15/part_f_next_actions_for_codex.json


## 2026-07-03 04:09:18 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 3 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00025`

Files: results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_f_predictive_trust_positive_control_matrix_shard3_of_4.csv

Note: rows=33; part_c_role=sanity


## 2026-07-03 04:09:42 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00025`

Files: results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_f_predictive_trust_positive_control_matrix_shard0_of_4.csv

Note: rows=34; part_c_role=sanity


## 2026-07-03 04:10:13 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 1 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00025`

Files: results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_f_predictive_trust_positive_control_matrix_shard1_of_4.csv

Note: rows=34; part_c_role=sanity


## 2026-07-03 04:11:03 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 2 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00025`

Files: results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_f_predictive_trust_positive_control_matrix_shard2_of_4.csv

Note: rows=34; part_c_role=sanity


## 2026-07-03 04:11:05 +0800 part-f-merge passed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00025`

Files: results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_f_next_actions_for_codex.json


## 2026-07-03 04:11:37 +0800 part-g failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-g --device cpu`

Files: results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_g_c2_f5_positive_control_summary.json; results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_g_next_actions_for_codex.json


## 2026-07-03 04:11:38 +0800 finalize done

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode finalize --device cpu`

Files: results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/final_route.json; results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/reproduction_manifest.md; results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/stale_artifact_audit.json
## 2026-07-03 04:12:00 +0800 manual audit note: margin10 sign fix and final direct-F verification

Context:

- While analyzing F5 heavy rejection, `margin10` was traced back to `experiments/run_v22_88_edge_state_flow_mpfu.py::metrics_for_logits`, where `margin10` is the 10% quantile of top1-top2 probability margin and higher is better.
- Pre-fix v23.00/v23.02 guard incorrectly treated `margin10_delta <= no_debt_budget` as safe, which rejected positive margin improvement and counted it like Brier/ECE/tail debt.
- Code fix:
  - `experiments/run_v23_00r_curve_geometry_population_flow.py`: added `debt_component_ok()` and `debt_component_slack()`, using `margin10_delta >= -budget` for safety and slack.
  - `experiments/run_v23_00r_curve_geometry_population_flow.py`: final `F5_no_debt` now uses the same signed helper.
  - `experiments/run_v23_00r_curve_geometry_population_flow.py` and `experiments/run_v23_02_predictive_trust_functional_population_flow.py`: merge-level `component_non_positive_rows` now treats margin10 with the correct sign.
  - `experiments/run_v23_00r_curve_geometry_population_flow.py`: result rows now record `edge_lr`, `adamw_lr`, `no_debt_budget`, `debt_veto_conflict_mode`, `debt_regularization_weight`, and `debt_regularization_components`.
  - `dgkan/fu/debt_conflict_veto.py` and `dgkan/optim/edge_sobolev_population_flow.py`: added optional corrected descent conflict mode for new proactive-veto diagnostics; legacy mode remains available.
  - `experiments/run_v23_00r_curve_geometry_population_flow.py`: added train-only proactive debt regularization hook.
  - `experiments/run_v23_02_predictive_trust_functional_population_flow.py`: added safety-specific parsing for `F15_etailreg0p3_finite_step_all`.

Key verification commands:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/debt_conflict_veto.py \
  dgkan/optim/edge_sobolev_population_flow.py \
  experiments/run_v23_00r_curve_geometry_population_flow.py \
  experiments/run_v23_02_predictive_trust_functional_population_flow.py
```

Final selected direct-F run:

```bash
rm -rf results/v23_02_direct_f_signfix_etailreg_lr00025_seed15
mkdir -p results/v23_02_direct_f_signfix_etailreg_lr00025_seed15
cp results/v23_02/part_c_taskwise_c2_summary.json \
  results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_c_metric_sanity_summary.json

for idx in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$idx V2302_OUT_ROOT=results/v23_02_direct_f_signfix_etailreg_lr00025_seed15 \
  /home/chengshun.wang/miniconda3/envs/kan/bin/python \
  experiments/run_v23_02_predictive_trust_functional_population_flow.py \
  --mode part-f --device cuda:0 --shard-count 4 --shard-index $idx \
  --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 \
  --part-f-basis dche_k9 --part-f-depths depth3 \
  --part-f-tasks local_patch_interaction,rotation_sensitive \
  --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all \
  --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 \
  --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 \
  --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 \
  --finite-step-guard-examples 128 --finite-step-tries 12 --finite-step-shrink 0.75 \
  --edge-lr 0.00025 \
  > results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/shard_${idx}.log 2>&1 &
done
wait

V2302_OUT_ROOT=results/v23_02_direct_f_signfix_etailreg_lr00025_seed15 \
/home/chengshun.wang/miniconda3/envs/kan/bin/python \
experiments/run_v23_02_predictive_trust_functional_population_flow.py \
--mode part-f-merge --device cpu \
--part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 \
--part-f-basis dche_k9 --part-f-depths depth3 \
--part-f-tasks local_patch_interaction,rotation_sensitive \
--part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all \
--part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 \
--train-steps 80 --finite-step-tries 12 --finite-step-shrink 0.75 --edge-lr 0.00025
```

Final route commands:

```bash
cp results/v23_02/part_a_identity_control_readiness_summary.json \
  results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_a_identity_control_readiness_summary.json
cp results/v23_02/part_b_history_boundary_lock_summary.json \
  results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_b_history_boundary_lock_summary.json
cp results/v23_02/part_c_taskwise_c2_summary.json \
  results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_c_taskwise_c2_summary.json

V2302_OUT_ROOT=results/v23_02_direct_f_signfix_etailreg_lr00025_seed15 \
/home/chengshun.wang/miniconda3/envs/kan/bin/python \
experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-g --device cpu

V2302_OUT_ROOT=results/v23_02_direct_f_signfix_etailreg_lr00025_seed15 \
/home/chengshun.wang/miniconda3/envs/kan/bin/python \
experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode finalize --device cpu
```

Final key files:

- `results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_f_predictive_trust_positive_control_matrix.csv`
- `results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_f_predictive_trust_positive_control_group_summary.csv`
- `results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_f_predictive_trust_positive_control_summary.json`
- `results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_g_c2_f5_positive_control_summary.json`
- `results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/final_route.json`


## 2026-07-03 04:14:51 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 3 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 20 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_etailreg0p3_tries20_seed15/part_f_predictive_trust_positive_control_matrix_shard3_of_4.csv

Note: rows=33; part_c_role=sanity


## 2026-07-03 04:15:27 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 0 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 20 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_etailreg0p3_tries20_seed15/part_f_predictive_trust_positive_control_matrix_shard0_of_4.csv

Note: rows=34; part_c_role=sanity


## 2026-07-03 04:16:46 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 1 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 20 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_etailreg0p3_tries20_seed15/part_f_predictive_trust_positive_control_matrix_shard1_of_4.csv

Note: rows=34; part_c_role=sanity


## 2026-07-03 04:17:28 +0800 part-f shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f --device cuda:0 --shard-count 4 --shard-index 2 --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --synthetic-train-size 128 --synthetic-guard-size 128 --population-grad-examples 4 --debt-grad-examples 4 --stat-warmup-steps 10 --finite-step-guard-examples 128 --finite-step-tries 20 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_etailreg0p3_tries20_seed15/part_f_predictive_trust_positive_control_matrix_shard2_of_4.csv

Note: rows=34; part_c_role=sanity


## 2026-07-03 04:17:29 +0800 part-f-merge passed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-f-merge --device cpu --part-f-base-schemes E5_FunctionalGram_DegreeEdgebankSNR_s0 --part-f-basis dche_k9 --part-f-depths depth3 --part-f-tasks local_patch_interaction,rotation_sensitive --part-f-safety-schemes F0_no_safety_control,F6_random_veto_matched_control,F15_etailreg0p3_finite_step_all --part-f-parts F3,F5 --part-f-c2-seed-count 15 --part-f-seed-count 15 --train-steps 80 --finite-step-tries 20 --finite-step-shrink 0.75 --edge-lr 0.001`

Files: results/v23_02_etailreg0p3_tries20_seed15/part_f_predictive_trust_positive_control_matrix.csv; results/v23_02_etailreg0p3_tries20_seed15/part_f_predictive_trust_positive_control_group_summary.csv; results/v23_02_etailreg0p3_tries20_seed15/part_f_next_actions_for_codex.json


## 2026-07-03 04:18:28 +0800 part-g failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-g --device cpu`

Files: results/v23_02_etailreg0p3_tries20_seed15/part_g_c2_f5_positive_control_summary.json; results/v23_02_etailreg0p3_tries20_seed15/part_g_next_actions_for_codex.json


## 2026-07-03 04:18:30 +0800 finalize done

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode finalize --device cpu`

Files: results/v23_02_etailreg0p3_tries20_seed15/final_route.json; results/v23_02_etailreg0p3_tries20_seed15/reproduction_manifest.md; results/v23_02_etailreg0p3_tries20_seed15/stale_artifact_audit.json


## 2026-07-03 04:19:42 +0800 manual audit note: tries20 diagnostic closed

Additional diagnostic root:

- `results/v23_02_etailreg0p3_tries20_seed15`

Purpose:

- Re-run Direct Part F after the `margin10` sign fix with `finite_step_tries=20` and `edge_lr=0.001`.
- This is an additional boundary diagnostic, not the selected main result. The selected main result remains `results/v23_02_direct_f_signfix_etailreg_lr00025_seed15` because it has a less conservative trust retry profile.

Key files inspected:

- `results/v23_02_etailreg0p3_tries20_seed15/part_f_predictive_trust_positive_control_group_summary.csv`
- `results/v23_02_etailreg0p3_tries20_seed15/part_f_predictive_trust_positive_control_summary.json`
- `results/v23_02_etailreg0p3_tries20_seed15/part_g_c2_f5_positive_control_summary.json`
- `results/v23_02_etailreg0p3_tries20_seed15/final_route.json`

Observed F15 diagnostic group:

```json
{
  "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0",
  "basis_key": "dche_k9",
  "depth": "depth3",
  "safety_type": "F15_etailreg0p3_finite_step_all",
  "rows": 45,
  "F5_no_debt_count": 15,
  "component_non_positive_rows": 15,
  "C2_coverage_retention_vs_F0": 2.117805482167732,
  "finite_step_accept_rate_median": 0.375,
  "finite_step_scale_mean_median": 0.2553262910323156,
  "finite_step_skip_count_median": 50.0,
  "random_veto_matched_gap": 13,
  "overhead_ratio_vs_F0": 0.909128721336865,
  "official_pass": 1
}
```

Final status:

- Part F route: `F_DirectC2F5PositiveControlPass`
- Part G route: `G_DirectFPassButPartCSanityFailed`
- Final route: `DirectFPassButPartCSanityFailed`
- `promotion_allowed=0`

Interpretation:

- More finite-step tries can still pass Direct F after the sign fix, and in this diagnostic improves random-veto gap to 13.
- It also makes the accepted update profile more conservative than the selected main run (`accept_rate_median=0.375`, `scale_mean_median=0.2553262910323156`, `skip_count_median=50.0`).
- This diagnostic does not change the final conclusion because Part C official taskwise C2 remains failed.


## 2026-07-03 04:28:44 +0800 code-save audit note: checker-patch edge gate support

During final git-save audit, two remaining core-code diffs were found after the first commit and were classified as experiment/repair support code rather than generated artifacts.

Files:

- `dgkan/optim/edge_sobolev_population_flow.py`
- `experiments/run_v23_00r_curve_geometry_population_flow.py`
- `experiments/run_v23_02_predictive_trust_functional_population_flow.py`

Code changes saved:

- Added `gate_input_side` and `gate_patch_size` optimizer parameters.
- Added checker/local-patch edge-function block construction for first-layer KAN edge parameters.
- Added block-level random-matched permutation so checker/edgebank gates have matched random controls at the same block granularity.
- Added scheme parsing and v23.02 Part C mappings for `C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0` and `C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0`.

Verification command:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/optim/edge_sobolev_population_flow.py \
  experiments/run_v23_00r_curve_geometry_population_flow.py \
  experiments/run_v23_02_predictive_trust_functional_population_flow.py
```

Verification result:

- Exit code 0.
- No new experiment metrics were produced by this audit save step.
- The selected final result remains `results/v23_02_direct_f_signfix_etailreg_lr00025_seed15`.

## 2026-07-03 04:36:48 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 2 --part-c-schemes C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0,C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0,C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction --part-c-seed-count 5 --part-c-gate-floors 0.3,0.35,0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_checker_patch_c_local_seed5/part_c_taskwise_c2_matrix_shard2_of_4.csv

Note: rows=15


## 2026-07-03 04:36:59 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 1 --part-c-schemes C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0,C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0,C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction --part-c-seed-count 5 --part-c-gate-floors 0.3,0.35,0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_checker_patch_c_local_seed5/part_c_taskwise_c2_matrix_shard1_of_4.csv

Note: rows=15


## 2026-07-03 04:37:19 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 3 --part-c-schemes C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0,C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0,C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction --part-c-seed-count 5 --part-c-gate-floors 0.3,0.35,0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_checker_patch_c_local_seed5/part_c_taskwise_c2_matrix_shard3_of_4.csv

Note: rows=15


## 2026-07-03 04:37:24 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 0 --part-c-schemes C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0,C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0,C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction --part-c-seed-count 5 --part-c-gate-floors 0.3,0.35,0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_checker_patch_c_local_seed5/part_c_taskwise_c2_matrix_shard0_of_4.csv

Note: rows=15


## 2026-07-03 04:37:50 +0800 part-c-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c-merge --device cpu --part-c-schemes C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0,C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0,C5_FunctionalGram_BlockSNR_edgebank_s0,C7_FunctionalGram_RandomMatchedGate_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction --part-c-seed-count 5 --part-c-gate-floors 0.3,0.35,0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_checker_patch_c_local_seed5/part_c_taskwise_c2_matrix.csv; results/v23_02_checker_patch_c_local_seed5/part_c_taskwise_c2_task_summary.csv; results/v23_02_checker_patch_c_local_seed5/part_c_taskwise_c2_group_summary.csv; results/v23_02_checker_patch_c_local_seed5/part_c_next_actions_for_codex.json


## 2026-07-03 04:39:40 +0800 manual audit note: checker-patch Part C result interpretation

Additional root:

- `results/v23_02_checker_patch_c_local_seed5`

Important context:

- This run tested only `local_patch_interaction`, not the full two-task official Part C set.
- The run was started with 4 shards but without per-shard `CUDA_VISIBLE_DEVICES`, so all shards used visible `cuda:0`. GPUs 1/2/3 were not used by this specific diagnostic.
- No fake rows were reported (`used_fake_data_rows=0`).

Key observations:

- `C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0`, `gate_floor=0.4`: `taskwise_all_pass=1`, `C2_accuracy_improvement_median=0.53125`, `C2_coverage_improvement_median=0.10959683824330568`, `random_gap=0.034741508337901905`, `official_pass=0`.
- `C16` at floors 0.3 and 0.35 had positive coverage but negative random gaps, so they failed.
- `C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0` also had taskwise pass rows versus `C7_FunctionalGram_RandomMatchedGate_s0`, which means the checker-patch random-matched control itself can improve the local task and cannot be treated as proof of non-random structure.

Conclusion:

- This diagnostic does not change the official route: Part C remains `C_C2FormationTaskwiseFailed`.
- It does provide a concrete signal that local-patch edge blocks can move the intended task, but the random-control separation is still too weak for promotion.

## 2026-07-03 04:42:37 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 2 --shard-index 0 --part-c-schemes C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0,C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_checker_patch_c_both_seed5_floor04/part_c_taskwise_c2_matrix_shard0_of_2.csv

Note: rows=10


## 2026-07-03 04:42:37 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 2 --shard-index 1 --part-c-schemes C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0,C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_checker_patch_c_both_seed5_floor04/part_c_taskwise_c2_matrix_shard1_of_2.csv

Note: rows=10


## 2026-07-03 04:43:10 +0800 part-c-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c-merge --device cpu --part-c-schemes C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0,C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_checker_patch_c_both_seed5_floor04/part_c_taskwise_c2_matrix.csv; results/v23_02_checker_patch_c_both_seed5_floor04/part_c_taskwise_c2_task_summary.csv; results/v23_02_checker_patch_c_both_seed5_floor04/part_c_taskwise_c2_group_summary.csv; results/v23_02_checker_patch_c_both_seed5_floor04/part_c_next_actions_for_codex.json


## 2026-07-03 04:44:08 +0800 manual audit note: checker-patch full two-task Part C result interpretation

Additional root:

- `results/v23_02_checker_patch_c_both_seed5_floor04`

Important context:

- This run tested both `local_patch_interaction` and `rotation_sensitive`, with `gate_floor=0.4`.
- The run used two shards but did not set per-shard `CUDA_VISIBLE_DEVICES`; the observed processes used visible `cuda:0`.
- No fake rows were reported (`used_fake_data_rows=0`).

Key observations:

- `C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0`: group `C2_accuracy_improvement_median=0.546875`, `C2_coverage_improvement_median=0.22123066616404685`, `random_gap=0.014359392538608517`, `taskwise_all_pass=0`, `official_pass=0`.
- C16 task details:
  - `local_patch_interaction`: `taskwise_pass=1`, `C2_coverage_improvement_median=0.10959683824330568`, `random_gap=0.05742679873583256`.
  - `rotation_sensitive`: `taskwise_pass=0`, `C2_coverage_improvement_median=0.32753474580385955`, `random_gap=-0.1696422048189561`.
- `C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0`: `taskwise_all_pass=1`, `C2_coverage_improvement_median=0.20687127362543833`, `random_gap=0.20687127362543833`, `official_pass=0` because it is the matched random control family.

Conclusion:

- Full two-task Part C remains failed with route `C_C2FormationTaskwiseFailed`.
- The checker/local-patch block has useful task signal, but the matched random control is too strong and the structured C16 gate does not separate on `rotation_sensitive`.
- This reinforces the next repair direction: redesign the random-matched control and/or add a rotation-sensitive structured edge-function block before another full Part C attempt.

## 2026-07-03 04:51:07 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 1 --part-c-schemes C18_FunctionalGram_VisualPatternDegreeEdgebankSNR_s0,C19_FunctionalGram_VisualPatternDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_visual_pattern_c_both_seed5_floor04/part_c_taskwise_c2_matrix_shard1_of_4.csv

Note: rows=5


## 2026-07-03 04:51:10 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 0 --part-c-schemes C18_FunctionalGram_VisualPatternDegreeEdgebankSNR_s0,C19_FunctionalGram_VisualPatternDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_visual_pattern_c_both_seed5_floor04/part_c_taskwise_c2_matrix_shard0_of_4.csv

Note: rows=5


## 2026-07-03 04:51:10 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 2 --part-c-schemes C18_FunctionalGram_VisualPatternDegreeEdgebankSNR_s0,C19_FunctionalGram_VisualPatternDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_visual_pattern_c_both_seed5_floor04/part_c_taskwise_c2_matrix_shard2_of_4.csv

Note: rows=5


## 2026-07-03 04:51:10 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 3 --part-c-schemes C18_FunctionalGram_VisualPatternDegreeEdgebankSNR_s0,C19_FunctionalGram_VisualPatternDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_visual_pattern_c_both_seed5_floor04/part_c_taskwise_c2_matrix_shard3_of_4.csv

Note: rows=5


## 2026-07-03 04:51:12 +0800 part-c-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c-merge --device cpu --part-c-schemes C18_FunctionalGram_VisualPatternDegreeEdgebankSNR_s0,C19_FunctionalGram_VisualPatternDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_visual_pattern_c_both_seed5_floor04/part_c_taskwise_c2_matrix.csv; results/v23_02_visual_pattern_c_both_seed5_floor04/part_c_taskwise_c2_task_summary.csv; results/v23_02_visual_pattern_c_both_seed5_floor04/part_c_taskwise_c2_group_summary.csv; results/v23_02_visual_pattern_c_both_seed5_floor04/part_c_next_actions_for_codex.json


## 2026-07-03 04:51:55 +0800 manual audit note: visual-pattern Part C result interpretation

Additional root:

- `results/v23_02_visual_pattern_c_both_seed5_floor04`

Important execution note:

- The intended run used 4 shards with explicit `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- A conflicting automatically-started 2-shard C18/C19 run began writing to the same result root. PIDs `57875` and `57879` were stopped before they wrote shard CSVs, to prevent merge/artifact overwrite. The recorded result is from the 4-shard run (`part_c_taskwise_c2_matrix_shard0_of_4.csv` through `shard3_of_4.csv`).

Key observations:

- `C18_FunctionalGram_VisualPatternDegreeEdgebankSNR_s0`: group `C2_accuracy_improvement_median=0.51953125`, `C2_coverage_improvement_median=0.2034493596147513`, `random_gap=-0.0010013098562922096`, `taskwise_all_pass=0`, `official_pass=0`.
- C18 task details:
  - `local_patch_interaction`: `taskwise_pass=0`, `C2_coverage_improvement_median=-0.05120259671821259`, `random_gap=-0.0539369899634039`.
  - `rotation_sensitive`: `taskwise_pass=0`, `C2_coverage_improvement_median=0.3508070065290667`, `random_gap=-0.017270728771109134`.
- `C19_FunctionalGram_VisualPatternDegreeRandomMatched_s0`: `taskwise_all_pass=0`, `C2_coverage_improvement_median=0.2044506694710435`, `random_gap=0.2044506694710435`, `official_pass=0`; it passed rotation but not local patch.

Conclusion:

- Visual-pattern edge blocks did not repair Part C.
- The rotation raw coverage signal improved, but random-gap separation stayed negative for C18; local patch degraded.
- This result supports the same blocker diagnosis: matched random controls remain too strong and official Part C remains failed.

## 2026-07-03 04:58:28 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 0 --part-c-schemes C20_FunctionalGram_CornerCheckerHybridDegreeEdgebankSNR_s0,C21_FunctionalGram_CornerCheckerHybridDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_corner_checker_hybrid_c_both_seed5_floor04/part_c_taskwise_c2_matrix_shard0_of_4.csv

Note: rows=5


## 2026-07-03 04:58:28 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 1 --part-c-schemes C20_FunctionalGram_CornerCheckerHybridDegreeEdgebankSNR_s0,C21_FunctionalGram_CornerCheckerHybridDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_corner_checker_hybrid_c_both_seed5_floor04/part_c_taskwise_c2_matrix_shard1_of_4.csv

Note: rows=5


## 2026-07-03 04:58:47 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 3 --part-c-schemes C20_FunctionalGram_CornerCheckerHybridDegreeEdgebankSNR_s0,C21_FunctionalGram_CornerCheckerHybridDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_corner_checker_hybrid_c_both_seed5_floor04/part_c_taskwise_c2_matrix_shard3_of_4.csv

Note: rows=5


## 2026-07-03 04:58:47 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 2 --part-c-schemes C20_FunctionalGram_CornerCheckerHybridDegreeEdgebankSNR_s0,C21_FunctionalGram_CornerCheckerHybridDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_corner_checker_hybrid_c_both_seed5_floor04/part_c_taskwise_c2_matrix_shard2_of_4.csv

Note: rows=5


## 2026-07-03 04:59:19 +0800 part-c-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c-merge --device cpu --part-c-schemes C20_FunctionalGram_CornerCheckerHybridDegreeEdgebankSNR_s0,C21_FunctionalGram_CornerCheckerHybridDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_corner_checker_hybrid_c_both_seed5_floor04/part_c_taskwise_c2_matrix.csv; results/v23_02_corner_checker_hybrid_c_both_seed5_floor04/part_c_taskwise_c2_task_summary.csv; results/v23_02_corner_checker_hybrid_c_both_seed5_floor04/part_c_taskwise_c2_group_summary.csv; results/v23_02_corner_checker_hybrid_c_both_seed5_floor04/part_c_next_actions_for_codex.json


## 2026-07-03 05:00:59 +0800 manual audit note: corner-checker hybrid Part C result interpretation

Additional root:

- `results/v23_02_corner_checker_hybrid_c_both_seed5_floor04`

Key observations:

- `C20_FunctionalGram_CornerCheckerHybridDegreeEdgebankSNR_s0`: group `C2_accuracy_improvement_median=0.51171875`, `C2_coverage_improvement_median=0.16042050320675116`, `random_gap=0.01120207260510142`, `taskwise_all_pass=1`, `official_pass=0`.
- C20 task details:
  - `local_patch_interaction`: `taskwise_pass=1`, `C2_coverage_improvement_median=0.06363091863386217`, `random_gap=0.10580588797165547`.
  - `rotation_sensitive`: `taskwise_pass=1`, `C2_coverage_improvement_median=0.29832908615935594`, `random_gap=0.03422732156468555`.
- `C21_FunctionalGram_CornerCheckerHybridDegreeRandomMatched_s0`: `taskwise_all_pass=0`, `C2_coverage_improvement_median=0.14921843060164974`, `random_gap=0.14921843060164974`, `official_pass=0`.

Conclusion:

- This is the strongest Part C repair signal so far because C20 passes both tasks taskwise.
- Official Part C still fails because the aggregate random-gap separation is far below the official threshold.
- The next blocker is no longer raw task coverage; it is matched-random separation and/or the official gap threshold under this family.

## 2026-07-03 05:10:56 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 1 --part-c-schemes C20_FunctionalGram_CornerCheckerHybridDegreeEdgebankSNR_s0,C21_FunctionalGram_CornerCheckerHybridDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.3,0.35,0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_corner_checker_hybrid_c_both_seed5_floors/part_c_taskwise_c2_matrix_shard1_of_4.csv

Note: rows=15


## 2026-07-03 05:11:00 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 0 --part-c-schemes C20_FunctionalGram_CornerCheckerHybridDegreeEdgebankSNR_s0,C21_FunctionalGram_CornerCheckerHybridDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.3,0.35,0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_corner_checker_hybrid_c_both_seed5_floors/part_c_taskwise_c2_matrix_shard0_of_4.csv

Note: rows=15


## 2026-07-03 05:11:13 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 2 --part-c-schemes C20_FunctionalGram_CornerCheckerHybridDegreeEdgebankSNR_s0,C21_FunctionalGram_CornerCheckerHybridDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.3,0.35,0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_corner_checker_hybrid_c_both_seed5_floors/part_c_taskwise_c2_matrix_shard2_of_4.csv

Note: rows=15


## 2026-07-03 05:11:14 +0800 part-c shard-written

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c --device cuda:0 --shard-count 4 --shard-index 3 --part-c-schemes C20_FunctionalGram_CornerCheckerHybridDegreeEdgebankSNR_s0,C21_FunctionalGram_CornerCheckerHybridDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.3,0.35,0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_corner_checker_hybrid_c_both_seed5_floors/part_c_taskwise_c2_matrix_shard3_of_4.csv

Note: rows=15


## 2026-07-03 05:11:49 +0800 part-c-merge failed

Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_02_predictive_trust_functional_population_flow.py --mode part-c-merge --device cpu --part-c-schemes C20_FunctionalGram_CornerCheckerHybridDegreeEdgebankSNR_s0,C21_FunctionalGram_CornerCheckerHybridDegreeRandomMatched_s0 --part-c-basis dche_k9 --part-c-depths depth3 --part-c-tasks local_patch_interaction,rotation_sensitive --part-c-seed-count 5 --part-c-gate-floors 0.3,0.35,0.4 --train-steps 240 --synthetic-train-size 192 --synthetic-guard-size 128 --population-grad-examples 8 --stat-warmup-steps 20`

Files: results/v23_02_corner_checker_hybrid_c_both_seed5_floors/part_c_taskwise_c2_matrix.csv; results/v23_02_corner_checker_hybrid_c_both_seed5_floors/part_c_taskwise_c2_task_summary.csv; results/v23_02_corner_checker_hybrid_c_both_seed5_floors/part_c_taskwise_c2_group_summary.csv; results/v23_02_corner_checker_hybrid_c_both_seed5_floors/part_c_next_actions_for_codex.json


## 2026-07-03 05:12:29 +0800 manual audit note: corner-checker hybrid floor grid interpretation

Additional root:

- `results/v23_02_corner_checker_hybrid_c_both_seed5_floors`

Key observations:

- `C20` floor 0.3: `taskwise_all_pass=0`, `C2_coverage_improvement_median=0.17556355401484325`, `random_gap=-0.02962311116334604`, `official_pass=0`.
- `C20` floor 0.35: `taskwise_all_pass=0`, `C2_coverage_improvement_median=0.18068588554024245`, `random_gap=-0.02292403803903653`, `official_pass=0`.
- `C20` floor 0.4: `taskwise_all_pass=1`, `C2_coverage_improvement_median=0.16042050320675116`, `random_gap=-0.0013607958203465387`, `official_pass=0`.
- Task details at C20 floor 0.4:
  - `local_patch_interaction`: `taskwise_pass=1`, `C2_coverage_improvement_median=0.06363091863386217`, `random_gap=0.12672323535480245`.
  - `rotation_sensitive`: `taskwise_pass=1`, `C2_coverage_improvement_median=0.29832908615935594`, `random_gap=0.03194779217301402`.

Conclusion:

- The floor grid confirms floor 0.4 is the only tested C20 floor that passes both tasks taskwise.
- The official Part C gate still fails because aggregate random-gap separation remains near zero or negative.
- Raising/lowering the gate floor within 0.3/0.35/0.4 is not sufficient to fix official Part C.
