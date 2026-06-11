# DG-KAN v16.4.1 FunctionalUpdate EfficiencyBreakthrough 4GPU 执行日志

生成时间：2026-06-02 03:52:48（Asia/Singapore）

## 1. 文件 / 输出目录

- plan: /home/chengshun.wang/DG-LCA/docs/DG-KAN_v16.4.1_FunctionalUpdate_EfficiencyBreakthrough_4GPU_完整计划.md
- runner: /home/chengshun.wang/DG-LCA/experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py
- v16.3 source artifacts: results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163
- output dir: results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641

## 2. Repro commands

Compile:
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py
```

Efficiency shards:
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py --out-dir results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641 --v163-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --run-lines EFF --device cuda:0 --efficiency-repeats 1 --efficiency-warmup 0 --efficiency-train-size 128 --hidden 16 --lr 0.003 --weight-decay 0.001 --smoke-steps 20 --smoke-train-size 96 --efficiency-shard-index 0 --efficiency-shard-count 4
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py --out-dir results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641 --v163-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --run-lines EFF --device cuda:1 --efficiency-repeats 1 --efficiency-warmup 0 --efficiency-train-size 128 --hidden 16 --lr 0.003 --weight-decay 0.001 --smoke-steps 20 --smoke-train-size 96 --efficiency-shard-index 1 --efficiency-shard-count 4
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py --out-dir results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641 --v163-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --run-lines EFF --device cuda:2 --efficiency-repeats 1 --efficiency-warmup 0 --efficiency-train-size 128 --hidden 16 --lr 0.003 --weight-decay 0.001 --smoke-steps 20 --smoke-train-size 96 --efficiency-shard-index 2 --efficiency-shard-count 4
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py --out-dir results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641 --v163-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --run-lines EFF --device cuda:3 --efficiency-repeats 1 --efficiency-warmup 0 --efficiency-train-size 128 --hidden 16 --lr 0.003 --weight-decay 0.001 --smoke-steps 20 --smoke-train-size 96 --efficiency-shard-index 3 --efficiency-shard-count 4
```

Functional smoke shards:
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py --out-dir results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641 --v163-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --run-lines SMOKE --device cuda:0 --efficiency-repeats 1 --efficiency-warmup 0 --efficiency-train-size 128 --hidden 16 --lr 0.003 --weight-decay 0.001 --smoke-steps 20 --smoke-train-size 96 --smoke-shard-index 0 --smoke-shard-count 4
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py --out-dir results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641 --v163-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --run-lines SMOKE --device cuda:1 --efficiency-repeats 1 --efficiency-warmup 0 --efficiency-train-size 128 --hidden 16 --lr 0.003 --weight-decay 0.001 --smoke-steps 20 --smoke-train-size 96 --smoke-shard-index 1 --smoke-shard-count 4
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py --out-dir results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641 --v163-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --run-lines SMOKE --device cuda:2 --efficiency-repeats 1 --efficiency-warmup 0 --efficiency-train-size 128 --hidden 16 --lr 0.003 --weight-decay 0.001 --smoke-steps 20 --smoke-train-size 96 --smoke-shard-index 2 --smoke-shard-count 4
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py --out-dir results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641 --v163-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --run-lines SMOKE --device cuda:3 --efficiency-repeats 1 --efficiency-warmup 0 --efficiency-train-size 128 --hidden 16 --lr 0.003 --weight-decay 0.001 --smoke-steps 20 --smoke-train-size 96 --smoke-shard-index 3 --smoke-shard-count 4
```

Import / merge / finalize:
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py --out-dir results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641 --v163-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --run-lines IMPORT_MATRIX,MERGE_EFF,MERGE_SMOKE --device cuda:0 --efficiency-repeats 1 --efficiency-warmup 0 --efficiency-train-size 128 --hidden 16 --lr 0.003 --weight-decay 0.001 --smoke-steps 20 --smoke-train-size 96
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py --out-dir results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641 --v163-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --run-lines FINALIZE --device cuda:0 --efficiency-repeats 1 --efficiency-warmup 0 --efficiency-train-size 128 --hidden 16 --lr 0.003 --weight-decay 0.001 --smoke-steps 20 --smoke-train-size 96
```

## 3. Runnable queue / GPU assignment

| queue | gpu | run | artifact | rows | status |
| --- | --- | --- | --- | --- | --- |
| EFF-0 | cuda:0 | EFF | v1641_efficiency_truth_table_e0.csv | 30 | completed |
| EFF-1 | cuda:1 | EFF | v1641_efficiency_truth_table_e1.csv | 30 | completed |
| EFF-2 | cuda:2 | EFF | v1641_efficiency_truth_table_e2.csv | 27 | completed |
| EFF-3 | cuda:3 | EFF | v1641_efficiency_truth_table_e3.csv | 27 | completed |
| SMOKE-0 | cuda:0 | SMOKE | v1641_functional_smoke_s0.csv | 12 | completed |
| SMOKE-1 | cuda:1 | SMOKE | v1641_functional_smoke_s1.csv | 12 | completed |
| SMOKE-2 | cuda:2 | SMOKE | v1641_functional_smoke_s2.csv | 12 | completed |
| SMOKE-3 | cuda:3 | SMOKE | v1641_functional_smoke_s3.csv | 12 | completed |
| IMPORT | cuda:0 | IMPORT_MATRIX | v1641_line_a_dche_results.csv | 567 | completed |
| MERGE-EFF | cuda:0 | MERGE_EFF | v1641_efficiency_truth_table.csv | 114 | completed |
| MERGE-SMOKE | cuda:0 | MERGE_SMOKE | v1641_functional_smoke.csv | 48 | completed |
| FINALIZE | cuda:0 | FINALIZE | v1641_route_decision.json |  | completed |

## 4. Artifact inventory

| artifact | exists | bytes | missing |
| --- | --- | --- | --- |
| v1641_route_decision.json | 1 | 1186 | 0 |
| v1641_efficiency_truth_table.csv | 1 | 124156 | 0 |
| v1641_efficiency_phase_breakdown.csv | 1 | 98929 | 0 |
| v1641_memory_phase_breakdown.csv | 1 | 64914 | 0 |
| v1641_basis_efficiency_blocker_taxonomy.csv | 1 | 22848 | 0 |
| v1641_same_param_mlp_manifest.csv | 1 | 8972 | 0 |
| v1641_basis_efficiency_repair_attempts.csv | 1 | 25027 | 0 |
| v1641_functional_smoke.csv | 1 | 13032 | 0 |
| v1641_line_a_dche_results.csv | 1 | 1369280 | 0 |
| v1641_line_b_mlp_results.csv | 1 | 1360008 | 0 |
| v1641_line_c_lq_reanchor.csv | 1 | 29339 | 0 |
| v1641_line_d_rational_monitor.csv | 1 | 51983 | 0 |
| v1641_line_f_allbasis_substrate.csv | 1 | 209518 | 0 |
| v1641_line_m_attribution.csv | 1 | 22648 | 0 |
| v1641_h800_summary.csv | 1 | 5654 | 0 |
| v1641_h1600_summary.csv | 1 | 86449 | 0 |
| v1641_direction_provenance.csv | 1 | 11203994 | 0 |
| v1641_runnable_queue.csv | 1 | 7303 | 0 |
| v1641_gpu_assignment_manifest.csv | 1 | 7303 | 0 |
| v1641_gpu_utilization_dashboard.csv | 1 | 383 | 0 |
| v1641_idle_violation.csv | 1 | 137 | 0 |
| v1641_queue_drain_report.csv | 1 | 85 | 0 |
| v1641_deferred_items.csv | 1 | 559 | 0 |
| v1641_required_artifact_manifest.csv | 1 | 2327 | 0 |
| v1641_forbidden_information_audit.csv | 1 | 313 | 0 |
| v1641_no_action_search_audit.csv | 1 | 138 | 0 |
| v1641_failure_taxonomy.csv | 1 | 17608 | 0 |
| v1641_no_go_boundary.csv | 1 | 693 | 0 |
| v1641_code_review_packet.csv | 1 | 1094 | 0 |
| v1641_code_review_packet.zip | 1 | 3115238 | 0 |
| v1641_implementation_readback.md | 1 | 1223 | 0 |
| source_vs_control_curve_h1_h20_h100_h800_h1600.svg | 1 | 5896 | 0 |
| source_retention_curve.svg | 1 | 3470 | 0 |
| tail_debt_recovery_curve.svg | 1 | 3437 | 0 |
| linec_debt_recovery_curve.svg | 1 | 3448 | 0 |
| auc_debt_curve.svg | 1 | 3462 | 0 |
| carrier_mechanism_heatmap.svg | 1 | 3380 | 0 |
| kan_vs_mlp_attribution_matrix.svg | 1 | 6062 | 0 |
| matched_control_explainability_heatmap.svg | 1 | 3344 | 0 |
| efficiency_forward_ratio_by_basis.svg | 1 | 5695 | 0 |
| efficiency_backward_ratio_by_basis.svg | 1 | 5680 | 0 |
| efficiency_update_ratio_by_basis.svg | 1 | 5654 | 0 |
| efficiency_memory_ratio_by_basis.svg | 1 | 5699 | 0 |
| phase_time_stacked_bar_by_basis.svg | 1 | 5725 | 0 |
| memory_phase_stacked_bar_by_basis.svg | 1 | 5721 | 0 |
| basis_efficiency_pareto_step_vs_memory.svg | 1 | 5726 | 0 |
| functional_overhead_vs_source_scatter.svg | 1 | 5775 | 0 |
| gpu_utilization_timeline.svg | 1 | 2293 | 0 |
| queue_depth_over_time.svg | 1 | 2309 | 0 |
| idle_violation_timeline.svg | 1 | 439 | 0 |
| job_completion_gantt.svg | 1 | 2302 | 0 |

## 5. Final route snapshot

```text
route = R4-FunctionalSourceNotRetained
line_p_efficiency_truth_table_complete = 1
functional_smoke_complete = 1
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
promotion_allowed = 0
```

## 6. 审计备注

```text
1. 所有 efficiency timing 来自 actual CUDA phase profiler；fake_or_proxy_timing=0。
2. D-CHE/MLP full matrix 是 v16.3 real artifact readback，rows 写 reuse_readback=1；这不是 fresh v16.4.1 training。
3. Smoke 使用 synthetic train-stream gradient，只做 low-budget signal check；smoke_only=1。
4. LineC/tail/AUC/calibration 未用于方向源；Rational 未启 reset/controller/action。
5. Queue artifacts 记录实际分片命令和 artifact presence；required manifest 决定闭合。
```
