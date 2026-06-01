# DG-KAN v15.3 SplitTransferOperatorFU AllBasisAcceleration 执行日志

生成时间：2026-05-31（Asia/Singapore）

## 1. 计划文件
```text
/home/chengshun.wang/DG-LCA/docs/DG-KAN_v15.03_SplitTransferOperatorFU_AllBasisAcceleration_完整计划.md
```

## 2. 修改文件
```text
experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py
experiments/run_v149_line_d_all_basis_substrate_repair.py
```

## 3. 编译检查
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py experiments/run_v149_line_d_all_basis_substrate_repair.py
```

结果：通过。

## 4. Line D substrate-only 执行指令
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --epochs 1 --linec-seeds 0,1,2 --candidates D-FOU52-LowFreqIdentityResidualV5,D-FOU53-BandwiseSNRWarmupV3,D-FOU54-PhaseStableBandMixV3,D-FOU55-NoMaterializeLifetimeV3,D-FOU56-HighFrequencyQuarantineV2,D-RBF50-CompactBumpIdentityResidualV4,D-RBF51-ActiveCenterOccupancyRepairV3,D-RBF52-WidthConditionGuardV3,D-RBF53-GaussianLocalK4NoDenseV2,D-RBF54-CenterSNRWarmupV2,D-WAV45-TriangularSupportV5,D-WAV46-ScaleOccupancyHardeningV3,D-WAV47-SupportOverlapDampingV3,D-WAV48-LocalTailCoverageAuditV2
```

## 5. Official v15.3 执行指令
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153 --line-d-out results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 64 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 0 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment
```

## 5.1 GPU 使用约束
```text
formal_device = cuda:0
official_full_rerun_device = cuda:0
line_d_substrate_device = cuda:0
此前发现过 CPU 误用风险；CPU full rerun 已中断，正式 artifact 以 cuda:0 full rerun 为准。
后续实验默认先检查 CUDA/GPU 空闲情况并使用 GPU；只有 GPU 不可用/OOM/计划明确要求 CPU 时才能改用 CPU，并必须在执行日志写明原因。
```

## 6. Blocker 与修复记录
```text
1. blocker: default --dche-candidate = D-CHE6-PhaseScheduleDegreeFMS，底层候选表无法解析。
   fix: 改为 v1410.DEFAULT_D_CHE_CANDIDATE = D-CHE17-HighDegreeLateEnableSubstrate。
   result: D-CHE model construction 通过。
2. blocker: MLP control model construction 缺少 mlp_hidden。
   fix: case_args.mlp_hidden = args.hidden。
   result: MLP/generic control construction 通过。
3. coverage gap: Line T result table 未显式记录 step/subspace_type/solver_type/solve_time_ms。
   fix: 写入最终 step、subspace_type、solver_type、metric_choice，并记录 solve_time_ms。
   result: 重新执行 full official command 后 result/provenance 均包含这些字段。
4. coverage gap: Line D v15.3 telemetry 字段名与 v149 substrate artifact 未完全对齐。
   fix: 添加 plan-facing aliases；原始 artifact 没有的 width_p01/width_p99/active_center_snr 写 availability/source，不补假值。
   result: Line D gate 使用实际 step_ratio/memory_ratio 字段重新计算。
5. coverage gap: finalizer 在首次空目录运行时可能先写 contract、后写 gate recompute artifact。
   fix: 调整 run() 顺序，先生成 v153_plan_coverage_recheck.csv / v153_gate_route_recompute.csv，再写 execution contract。
   result: 首次运行与 reuse finalizer 都可保持 contract_unclosed_rows = 0。
6. 上述修复后重新执行 full official command；训练指标与 route 来自最终成功落盘 artifact。
7. coverage gap: 计划 6.3 的 M0/M1/M2 metric choice factor 初版只执行 M0。
   fix: 接入 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment；
        M1/M2 只使用当前 train-stream optimizer state / parameter group second moment。
   result: 重跑 full official command，v153_metric_choice_coverage.csv 显示 T/M metric factor complete。
8. coverage gap: Line X transfer audit 的 reservoir/noise readback 字段为空，T-FB6 未在 fallback CSV 中显式成行。
   fix: 用已落盘 B1/B2 loss delta 派生 audit-only readback，并补齐 T-FB6 行。
   result: reuse-if-present finalizer 重写 artifact；不新增训练、不改变 route。
9. route semantics: Line X route 名需要区分 audit signal 和 promotion，旧 R-X-TransferSupported 容易误读。
   fix: 显式写入 v15.02.1 baseline/reduced flag 与 line_x_gate_pass。
   result: route decision 更精确；final route 仍由 T/D/M gates 判定，promotion_allowed=0 不变。
10. denominator audit: metric-choice factor 将 Line X 3x3 audit 展开成 135 raw rows，raw count 不可直接对比 v15.02.1 10-row baseline。
    fix: 新增 v153_line_x_denominator_audit.csv；Line X gate 使用 dataset-seed 3x3 口径，raw count/fraction 只作审计。
    result: Line X audit gate 口径对齐计划；final route 仍由 S2/S3/S4/S5 与 Line D/T exhaustion 判定。
11. objective implementation: 非 Adam ST-FU methods 误实现为 AdamW + alpha * U direction。
    fix: 按计划 objective f + J U alpha 修正为纯 alpha * U direction；T0/M0 保持 AdamW baseline。
    result: 重新执行 full official command，所有 route/gate 以修正后 artifact 为准。
12. scale sanity: objective 修正后，T1-T5 在固定 alpha grid 上全部撞到 0.20 上界。
    fix: 在 T-FB4 fallback artifact 中补齐 alpha_at_grid_max_fraction / scale_grid_boundary_hit。
    result: 仅增强审计；不扩 alpha grid，不新增 method/control/action search。
13. gate recompute precedence: R1/R2/R6 可同时为真，旧 per-route equality check 会把被 R1 shadow 的 R2/R6 记为 inconsistent。
    fix: 按 route precedence 先重算 expected_route，再统一比较 final route。
    result: gate_route_inconsistent_rows 回到 0。
14. commit scale: 计划写 Δθ = Uα，旧实现对 ST-FU 提交时仍乘 args.lr。
    fix: 非 Adam ST-FU methods 使用 commit_lr_multiplier=1.0，T0/M0 AdamW baseline 保持 args.lr。
    result: 重新执行 full official command，所有 route/gate 以修正后 artifact 为准。
15. U normalization unit: commit-scale 修正后继续复核 actual_step_over_adam_step_norm，发现 U 被归一化到 raw Adam direction。
    fix: 非 Adam ST-FU methods 使用 adam * lr 作为 basis norm target，使 U 落在实际 AdamW step 单位。
    result: 重新执行 full official command；final route 以修正后 artifact 为准，promotion_allowed=0 不变。
    result: final route 更精确；promotion_allowed=0 不变。
16. finalizer blocker: R4 route 初版引用 x_gate 早于赋值。
    fix: 将 x_gate 赋值移到 r4 判断之前。
    result: reuse finalizer 通过；训练 artifact 不变。
17. runtime implementation: R4 blocker 指向 step_time，复核发现 trial evaluation 有重复 clone/restore 与重复 true B2 forward。
    fix: 每步单次 saved-params restore；非 B2Shuffled objective 复用 B2 objective loss。
    result: 用 cuda:0 重新执行 full official command；不改变 ST-FU objective/subspace/alpha grid/direction source。
18. runtime implementation: 每步重复计算 full batch gradient，而 B1/B2 gradients 已足够合成同一 mean gradient。
    fix: 使用 grad = (n1*grad_B1+n2*grad_B2)/(n1+n2)。
    result: 用 cuda:0 重新执行 full official command；不改变 direction source 或 gate。
```

## 7. 输出目录
```text
official_out = results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153
line_d_out = results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate
```

## 8. 最终核验
```text
route = R1-STFUStillLocalPositiveNoTransfer
minimum_success = S1-SplitTransferOperatorExecuted
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 9. 用户再次追问后的 Line X audit readback finalizer
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py experiments/run_v149_line_d_all_basis_substrate_repair.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153 --line-d-out results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 64 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment
```

复核结果：
```text
line_x_audit_rows = 135
line_x_blank_readback_rows = 0
line_t_fallback_rows = 30
plan_coverage_unclosed_rows = 0
gate_route_inconsistent_rows = 0
route = R1-STFUStillLocalPositiveNoTransfer
promotion_allowed = 0
```

## 10. 用户再次追问后的 Line X route 语义 finalizer
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153 --line-d-out results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 64 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment
```

复核结果：
```text
line_x_gate_count_basis = dataset_seed_3x3_after_metric_choice_expansion
transfer_supported_dataset_seed_count = 0
local_positive_all_dataset_seed_count = 9
transfer_supported_raw_count = 0
local_positive_no_transfer_raw_count = 135
line_x_gate_pass = 0
line_x_route = R-X-SplitTransferStillNoTransfer
line_x_local_positive_baseline_count = 10
line_x_local_positive_reduced_vs_v1521 = 0
gate_route_inconsistent_rows = 0
route = R1-STFUStillLocalPositiveNoTransfer
promotion_allowed = 0
```

## 11. 用户再次追问后的 Line T objective full rerun
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py experiments/run_v149_line_d_all_basis_substrate_repair.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153 --line-d-out results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 64 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 0 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153 --line-d-out results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 64 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment
```

复核结果：
```text
line_t_real_lite_pass_count = 0 / 9
line_t_source_vs_best_control_mean = -0.46042725774976945
line_t_control_equivalent_fraction = 1.0
line_t_bad_event_fraction = 1.0
line_t_best_step_time_overhead_fraction = 0.8888888888888888
line_t_best_memory_overhead_fraction = 0.0
line_t_best_tail_fail_fraction = 0.2222222222222222
line_t_best_linec_fail_fraction = 1.0
line_x_gate_pass = 0
line_x_route = R-X-SplitTransferStillNoTransfer
transfer_supported_dataset_seed_count = 0 / 9
local_positive_all_dataset_seed_count = 9 / 9
generic_stfu_explains = 1
kan_specific_pass_count = 0
line_t_fb4_scale_boundary_hit_rows = 5 / 5
route = R1-STFUStillLocalPositiveNoTransfer
promotion_allowed = 0
```

## 12. 用户再次追问后的 ST-FU commit scale full rerun
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py experiments/run_v149_line_d_all_basis_substrate_repair.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153 --line-d-out results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 64 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 0 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153 --line-d-out results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 64 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment
```

复核结果：
```text
line_t_real_lite_pass_count = 0 / 9
line_t_source_vs_best_control_mean = -0.46042725774976945
line_t_control_equivalent_fraction = 1.0
line_t_bad_event_fraction = 1.0
line_x_gate_pass = 0
line_x_route = R-X-SplitTransferStillNoTransfer
transfer_supported_dataset_seed_count = 0 / 9
local_positive_all_dataset_seed_count = 9 / 9
generic_stfu_explains = 1
kan_specific_pass_count = 0
line_t_fb4_scale_boundary_hit_rows = 5 / 5
route = R1-STFUStillLocalPositiveNoTransfer
promotion_allowed = 0
```

## 13. 用户再次追问后的 R4 runtime optimization full rerun
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py experiments/run_v149_line_d_all_basis_substrate_repair.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153 --line-d-out results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 64 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 0 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153 --line-d-out results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 64 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment
```

复核结果：
```text
line_t_real_lite_pass_count = 0 / 9
line_t_source_vs_best_control_mean = -0.46042725774976945
line_t_control_equivalent_fraction = 1.0
line_t_bad_event_fraction = 1.0
line_t_best_step_time_overhead_fraction = 0.8888888888888888
line_t_best_tail_fail_fraction = 0.2222222222222222
line_t_best_linec_fail_fraction = 1.0
line_t_fb5_candidate_eval_count_median = 4.0
line_t_fb5_duplicate_true_b2_eval_count_median = 0.0
line_t_fb5_saved_restore_rows = 5 / 5
line_t_fb5_full_grad_from_split_rows = 5 / 5
line_x_gate_pass = 0
generic_stfu_explains = 1
route = R1-STFUStillLocalPositiveNoTransfer
promotion_allowed = 0
```

## 14. 用户再次追问后的 R4 full-gradient backward optimization full rerun
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py experiments/run_v149_line_d_all_basis_substrate_repair.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153 --line-d-out results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 64 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 0 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153 --line-d-out results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 64 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment
```

复核结果：
```text
line_t_real_lite_pass_count = 0 / 9
line_t_source_vs_best_control_mean = -0.46042725774976945
line_t_control_equivalent_fraction = 1.0
line_t_bad_event_fraction = 1.0
line_t_best_step_time_overhead_fraction = 0.8888888888888888
line_t_best_tail_fail_fraction = 0.2222222222222222
line_t_best_linec_fail_fraction = 1.0
line_t_fb5_full_grad_from_split_rows = 5 / 5
line_x_gate_pass = 0
generic_stfu_explains = 1
route = R1-STFUStillLocalPositiveNoTransfer
promotion_allowed = 0
```

## 15. 最终可继续性复核 finalizer
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153 --line-d-out results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 64 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment
```

复核结果：
```text
route = R1-STFUStillLocalPositiveNoTransfer
minimum_success = S1-SplitTransferOperatorExecuted
real_lite_pass_count = 0 / 9
line_x_gate_pass = 0
generic_stfu_explains = 1
line_d_best_non_dche_dataset_seed_pass_count = 0 / 9
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
promotion_allowed = 0
remaining_legal_v153_branch = 0
```

## 16. U normalization unit bug 修复与 GPU full rerun

本次继续推进是因为复核发现上一轮 `actual_step_over_adam_step_norm≈19.44` 不是单纯理论 blocker，而是实现单位不一致：

```text
计划第 2.2 / 6.3 写的是：
  U0 = AdamW update direction span
  Delta theta = U alpha

旧实现：
  candidate_updates 先把所有 basis norm-match 到 raw Adam direction。
  非 Adam ST-FU commit_lr_multiplier = 1.0。
  因此 alpha=0.025 实际约等于 25x AdamW step。

修复：
  非 Adam ST-FU / MLP-STFU methods 使用 adam * lr 作为 basis norm target。
  T0-D-CHE-AdamW / M0-MLP-AdamW baseline 仍保持 raw adam + commit lr。
  J_B2_U_norm 改为用归一化后的 U 计算。
  same_norm_random_gap 改为比较 actual step norm。
```

执行命令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153 --line-d-out results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 64 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 0 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py --out-dir results/v15_03_split_transfer_operator_fu_allbasis_acceleration/official_v153 --line-d-out results/v15_03_split_transfer_operator_fu_allbasis_acceleration/line_d_v153_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 64 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1 --metric-choices M0-identity,M1-AdamVDiag,M2-DegreeRoleSecondMoment
```

artifact readback：

```text
route = R1-STFUStillLocalPositiveNoTransfer
minimum_success = S1-SplitTransferOperatorExecuted
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = -0.46042725774976945
control_equivalent_fraction = 1.0
bad_event_fraction = 1.0
best_tail_fail_fraction = 0.2222222222222222
best_linec_fail_fraction = 1.0
best_step_time_overhead_fraction = 0.8888888888888888
line_x_gate_pass = 0
generic_stfu_explains = 1
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

best method readback：

```text
method = T1-D-CHE-STFU-AdamSubspace [M2-DegreeRoleSecondMoment]
rows = 9
real_lite_pass = 0 / 9
strict_pass = 0 / 9
source_vs_best_control_mean = -0.46042725774976945
actual_step_over_adam_step_norm_mean = 0.20000001284909386
CEp99_delta_mean = -0.1379088560740153
step_time_ratio_mean = 1.634300476466451
tail_fail_from_fail_reason = 2 / 9
linec_fail_from_fail_reason = 9 / 9
```

audit：

```text
v153_required_artifact_manifest.csv rows = 31, missing_sum = 0
v153_forbidden_information_audit.csv rows = 7, violation_sum = 0
v153_no_action_search_audit.csv rows = 7, violation_sum = 0
v153_gate_route_recompute.csv rows = 10, inconsistent_rows = 0
v153_execution_contract_coverage_audit.csv rows = 11, unclosed_rows = 0
```

判断：

```text
U 单位修复后，previous R4 / 9-of-9 real-lite signal 被证伪：
  actual step scale 从约 19.44x AdamW step 回到 0.20x；
  tail 明显改善；
  但 source_vs_best_control 变为负数，Line X transfer_supported 变为 0/9。

因此 v15.3 的最终科学结论不是“有 transfer value 但 overhead blocker”，
而是修复实现 bug 后回到：
  R1-STFUStillLocalPositiveNoTransfer。

该修复不新增 T6/T7，不扩 alpha grid，不使用 audit metric 反推方向，
不新增 controller/action/reset，也没有 CPU offload。
```
