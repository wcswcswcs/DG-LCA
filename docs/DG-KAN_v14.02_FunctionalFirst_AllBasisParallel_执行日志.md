# DG-KAN v14.2 FunctionalFirst AllBasisParallel 执行日志

生成时间：2026-05-29（Asia/Singapore）

本日志只记录本轮实际执行过的命令、输出目录、blocker 与修复动作；不补写未执行命令。

## 1. 计划文件

```text
docs/DG-KAN_v14.2_FunctionalFirst_AllBasisParallel_完整计划.md
```

## 2. 新增 / 修改文件

```text
experiments/run_v142_functional_first_all_basis_parallel.py
docs/DG-KAN_v14.2_FunctionalFirst_AllBasisParallel_执行日志.md
docs/DG-KAN_v14.2_FunctionalFirst_AllBasisParallel_实验结果复盘.md
```

## 3. 初始实现与语法检查

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v142_functional_first_all_basis_parallel.py
```

结果：

```text
py_compile pass
```

## 4. Smoke blocker 1：import path

初始 smoke 命令：

```bash
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/smoke_v142 --synthetic-tasks X1 --synthetic-seeds 0 --loss-interfaces CE --methods F0-AdamWParallel,F3-LayerFMS,FCTRL-RandomMatchedNorm --train-steps 4 --batch-size 8 --synthetic-train-size 32 --synthetic-val-size 24 --mlp-hidden 16 --trace-interval 2 --skip-nonrat-fms --compute-budgeted-run 1
```

失败：

```text
ModuleNotFoundError: No module named 'dgkan'
```

修复：

```text
在 experiments/run_v142_functional_first_all_basis_parallel.py 开头加入 repo root sys.path。
```

复查：

```bash
conda run -n kan python -m py_compile experiments/run_v142_functional_first_all_basis_parallel.py
```

结果：

```text
py_compile pass
```

## 5. Smoke blocker 2：synthetic_dim

重跑 smoke 时 D-RAT30 在 synthetic_dim=12 下 forward shape mismatch：

```text
RuntimeError: The size of tensor a (78) must match the size of tensor b (136)
```

修复：

```text
将 runner 默认 synthetic_dim 改为 16；后续命令显式使用 --synthetic-dim 16。
```

## 6. Smoke pass

执行：

```bash
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/smoke_v142 --synthetic-tasks X1 --synthetic-seeds 0 --loss-interfaces CE --methods F0-AdamWParallel,F3-LayerFMS,FCTRL-RandomMatchedNorm --train-steps 4 --batch-size 8 --synthetic-train-size 32 --synthetic-val-size 24 --synthetic-dim 16 --mlp-hidden 16 --trace-interval 2 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R4-FMSNoGoCurrentDefinition
required_artifact_missing_count = 0
basis_substrate_pass_count = 1
healthy_base_pass_count = 1
mlp_fms_task_pass_count = 0
rational_fms_task_pass_count = 0
promotion_allowed = 0
```

输出目录：

```text
results/v14_2_functional_first_all_basis_parallel/smoke_v142/
```

## 7. Official compute-budgeted run

执行：

```bash
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/official_v142 --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,F1-PriorSNRReference,F2-ParameterFMS,F3-LayerFMS,F4-RoleFMS,F5-BasisGroupFMS,F7-PhaseScheduleFMS,FCTRL-RandomMatchedNorm --train-steps 20 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 10 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R4-FMSNoGoCurrentDefinition
required_artifact_missing_count = 0
basis_substrate_pass_count = 1
healthy_base_pass_count = 1
mlp_fms_task_pass_count = 0
rational_fms_task_pass_count = 0
promotion_allowed = 0
```

输出目录：

```text
results/v14_2_functional_first_all_basis_parallel/official_v142/
```

## 8. 修复：amortized FMS update

触发原因：

```text
official 中多行 source/AUC 有局部改善，但 FMS per-example gradient overhead 超过 gate，synthetic pass = 0。
```

代码修改：

```text
experiments/run_v142_functional_first_all_basis_parallel.py
  新增 --fms-update-interval。
  per-example FMS state 周期性刷新；非刷新 step 沿用 persistent state 缩放普通 train gradient。
  update trace 写入 fms_update_interval / amortized_fms_update。
```

## 9. Repair 1：beta=0.99, interval=20

执行：

```bash
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/repair_v142_amortized_beta099 --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,F2-ParameterFMS,F3-LayerFMS,F7-PhaseScheduleFMS,FCTRL-RandomMatchedNorm --train-steps 20 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 10 --fms-beta 0.99 --fms-update-interval 20 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R4-FMSNoGoCurrentDefinition
mlp_fms_task_pass_count = 4
rational_fms_task_pass_count = 0
promotion_allowed = 0
```

## 10. Repair 2：beta=0.99, interval=40

执行：

```bash
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/repair_v142_amortized40_beta099 --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,F2-ParameterFMS,F3-LayerFMS,F7-PhaseScheduleFMS,FCTRL-RandomMatchedNorm --train-steps 40 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 20 --fms-beta 0.99 --fms-update-interval 40 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
mlp_fms_task_pass_count = 5
rational_fms_task_pass_count = 1
promotion_allowed = 0
```

## 11. 修复：Rational candidate override

触发原因：

```text
MLP-FMS 已达 5/7，但 Rational 只有 1/7；按计划进入 Rational blocker repair。
```

代码修改：

```text
experiments/run_v142_functional_first_all_basis_parallel.py
  新增 --rational-candidate。
  substrate map 与 FMS training 可使用指定 D-RAT candidate。
```

## 12. Rational focused probes

D-RAT26：

```bash
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/repair_v142_rat26_amortized40 --rational-candidate D-RAT26-TangentTrustRegionNoCE --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,F2-ParameterFMS,F3-LayerFMS,F7-PhaseScheduleFMS,FCTRL-RandomMatchedNorm --train-steps 40 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 20 --fms-beta 0.99 --fms-update-interval 40 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
mlp_fms_task_pass_count = 5
rational_fms_task_pass_count = 1
```

D-RAT28：

```bash
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_amortized40 --rational-candidate D-RAT28-GroupDiversityPreservingRational --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,F2-ParameterFMS,F3-LayerFMS,F7-PhaseScheduleFMS,FCTRL-RandomMatchedNorm --train-steps 40 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 20 --fms-beta 0.99 --fms-update-interval 40 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
mlp_fms_task_pass_count = 5
rational_fms_task_pass_count = 2
```

D-RAT35：

```bash
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/repair_v142_rat35_amortized40 --rational-candidate D-RAT35-ReadoutRationalDecoupleNoCE --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,F2-ParameterFMS,F3-LayerFMS,F7-PhaseScheduleFMS,FCTRL-RandomMatchedNorm --train-steps 40 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 20 --fms-beta 0.99 --fms-update-interval 40 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
mlp_fms_task_pass_count = 5
rational_fms_task_pass_count = 1
```

D-RAT39：

```bash
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/repair_v142_rat39_amortized40 --rational-candidate D-RAT39-DenDerivativeSubstrateRepairNoCE --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,F2-ParameterFMS,F3-LayerFMS,F7-PhaseScheduleFMS,FCTRL-RandomMatchedNorm --train-steps 40 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 20 --fms-beta 0.99 --fms-update-interval 40 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
mlp_fms_task_pass_count = 5
rational_fms_task_pass_count = 0
```

## 13. Reduced-plasticity repair

触发原因：

```text
D-RAT28 有最好的 rational_fms_task_pass_count = 2，但许多 strong-source rows 被 LineC / tail gate 拒绝；
按计划尝试 reduced plasticity。
```

执行：

```bash
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_lowplasticity --rational-candidate D-RAT28-GroupDiversityPreservingRational --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,F2-ParameterFMS,F3-LayerFMS,F7-PhaseScheduleFMS,FCTRL-RandomMatchedNorm --train-steps 40 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 20 --fms-beta 0.99 --fms-strength 0.10 --fms-update-interval 40 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
mlp_fms_task_pass_count = 5
rational_fms_task_pass_count = 0
promotion_allowed = 0
```

## 14. Rational R6/R7/R8 safety repair

触发原因：

```text
计划第 5.4 / 13.2 / 13.5 节要求继续检查：
R6 denominator/slope safety；
R7 delayed readout-basis coupling；
R8 phase schedule。
```

代码修改：

```text
experiments/run_v142_functional_first_all_basis_parallel.py
  新增 R6-RationalDenominatorSafetyFMS。
  新增 R7-RationalDelayedReadoutBasisFMS。
  新增 R8-RationalPhaseScheduleFMS。
  R6/R7/R8 只使用 train-stream gradient 与 model-state safety scale；
  不使用 LineC / CEp99 / NLL / ECE 生成方向。
```

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v142_functional_first_all_basis_parallel.py
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_r6r7r8 --rational-candidate D-RAT28-GroupDiversityPreservingRational --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,R6-RationalDenominatorSafetyFMS,R7-RationalDelayedReadoutBasisFMS,R8-RationalPhaseScheduleFMS,FCTRL-RandomMatchedNorm --train-steps 40 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 20 --fms-beta 0.99 --fms-update-interval 40 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R4-FMSNoGoCurrentDefinition
mlp_fms_task_pass_count = 3
rational_fms_task_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

## 15. All-basis candidate remap

触发原因：

```text
早期 v14.2 candidate map 对 D-CHE / D-FOU / D-WAV 使用了 v13.10 名称，
在 v12.35 source substrate map 中没有命中 source row。
为避免把 source missing 误写成真实 substrate 结论，改用 v12.35 中实际存在的候选。
```

代码修改：

```text
D-CHE: D-CHE12-LifetimeRecomputeBackward-K3
D-FOU: D-FOU14-SincosSharedWorkspace-K2
D-WAV: D-WAV11-ScaleEnergyBalanceSubstrate
```

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v142_functional_first_all_basis_parallel.py
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/official_v142_remap --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,F1-PriorSNRReference,F2-ParameterFMS,F3-LayerFMS,F4-RoleFMS,F5-BasisGroupFMS,F7-PhaseScheduleFMS,FCTRL-RandomMatchedNorm --train-steps 20 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 10 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R4-FMSNoGoCurrentDefinition
basis_substrate_pass_count = 1
healthy_base_pass_count = 1
mlp_fms_task_pass_count = 0
rational_fms_task_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

## 16. 全 v12.35 substrate sweep

触发原因：

```text
为了排除“只选错 Non-RAT candidate”的审计风险，对 v12.35 全部 44 个 substrate rows 按 v14.2 gate 重判。
该步骤不新增训练，不生成 functional direction，不允许 promotion。
```

新增文件：

```text
experiments/analyze_v142_all_basis_substrate_sweep.py
```

执行：

```bash
python experiments/analyze_v142_all_basis_substrate_sweep.py
```

结果：

```text
source_rows = 44
v142_pass_rows = 11
families_with_pass = D-RAT
nonrat_families_with_pass = []
new_training_executed = 0
promotion_allowed = 0
```

输出目录：

```text
results/v14_2_functional_first_all_basis_parallel/substrate_sweep_v142/
```

## 17. Longer amortized Rational phase probe

触发原因：

```text
Rational strong-source rows 仍被 LineC / tail / overhead 拒绝；
为排除“FMS 刷新仍太频繁 / phase coupling 太早”，加长 train_steps 与 fms_update_interval。
```

执行：

```bash
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_interval80_phase --rational-candidate D-RAT28-GroupDiversityPreservingRational --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,F2-ParameterFMS,F3-LayerFMS,F7-PhaseScheduleFMS,R7-RationalDelayedReadoutBasisFMS,R8-RationalPhaseScheduleFMS,FCTRL-RandomMatchedNorm --train-steps 80 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 40 --fms-beta 0.99 --fms-update-interval 80 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
mlp_fms_task_pass_count = 6
rational_fms_task_pass_count = 2
required_artifact_missing_count = 0
promotion_allowed = 0
```

## 18. R9 Rational role-separated FMS

触发原因：

```text
计划第 13.2 节建议 numerator/denominator/readout update separation。
R9 固定降低 denominator/readout plasticity，保留 numerator/basis 更新；
不使用 LineC / CEp99 / NLL / ECE 生成方向。
```

代码修改：

```text
experiments/run_v142_functional_first_all_basis_parallel.py
  新增 R9-RationalRoleSeparatedFMS。
```

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v142_functional_first_all_basis_parallel.py
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_r9_rolesep --rational-candidate D-RAT28-GroupDiversityPreservingRational --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,R9-RationalRoleSeparatedFMS,FCTRL-RandomMatchedNorm --train-steps 80 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 40 --fms-beta 0.99 --fms-update-interval 80 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
mlp_fms_task_pass_count = 6
rational_fms_task_pass_count = 1
required_artifact_missing_count = 0
promotion_allowed = 0
```

## 19. Rational rejection / role signal mass audit

触发原因：

```text
计划第 13.2 节要求在 MLP-FMS positive 但 Rational-FMS 不过时，
不能直接返回 architecture search，必须做 Rational FMS rejection audit 与 role signal mass audit。
```

新增脚本：

```text
experiments/analyze_v142_rational_rejection_audit.py
```

执行：

```bash
conda run -n kan python -m py_compile experiments/analyze_v142_rational_rejection_audit.py
conda run -n kan python experiments/analyze_v142_rational_rejection_audit.py
```

读取 artifact：

```text
results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_amortized40/
results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_interval80_phase/
results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_r6r7r8/
results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_r9_rolesep/
```

输出：

```text
results/v14_2_functional_first_all_basis_parallel/rational_rejection_audit_v142/v142_rational_rejection_route.json
results/v14_2_functional_first_all_basis_parallel/rational_rejection_audit_v142/v142_rational_rejection_audit.csv
results/v14_2_functional_first_all_basis_parallel/rational_rejection_audit_v142/v142_rational_rejection_summary.csv
results/v14_2_functional_first_all_basis_parallel/rational_rejection_audit_v142/v142_rational_rejection_reason_counts.csv
results/v14_2_functional_first_all_basis_parallel/rational_rejection_audit_v142/v142_role_signal_mass_audit.csv
```

初始结果：

```text
diagnostic_route = D2-RationalLineCTailOverheadRejectedLocalSource
official_route_unchanged = R4-FMSNoGoCurrentDefinition
best_repair_route_observed = R3-GenericFunctionalOnlyKANSpecificNotEstablished
uses_existing_artifacts_only = 1
new_training_executed = 0
promotion_allowed = 0
```

## 20. R6 denominator loosen/tighten diagnostic

触发原因：

```text
计划第 13.2 节要求 denominator safety loosen/tighten diagnostic。
此前 R6 只有 standard clamp；本次新增 loose/tight 两个合法 train-stream diagnostic。
```

代码修改：

```text
experiments/run_v142_functional_first_all_basis_parallel.py
  新增 R6L-RationalDenominatorLooseFMS
  新增 R6T-RationalDenominatorTightFMS
  rational_denominator_safety_mode 增加 loose / tight 审计字段
```

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v142_functional_first_all_basis_parallel.py
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_r6_loose_tight --rational-candidate D-RAT28-GroupDiversityPreservingRational --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,R6L-RationalDenominatorLooseFMS,R6T-RationalDenominatorTightFMS,FCTRL-RandomMatchedNorm --train-steps 80 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 40 --fms-beta 0.99 --fms-update-interval 80 --skip-nonrat-fms --compute-budgeted-run 1
conda run -n kan python experiments/analyze_v142_rational_rejection_audit.py
```

结果：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
minimum_success = S2-GenericFMSPositive
mlp_fms_task_pass_count = 5
rational_fms_task_pass_count = 2
basis_substrate_pass_count = 1
healthy_base_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
```

审计摘要：

```text
R6L task_pass_count = 2
R6L median_source_vs_best_control = -0.022265374660491943
R6L max_source_vs_best_control = 1.5711631774902344
R6L source_positive_rows = 7
R6T task_pass_count = 0
R6T median_source_vs_best_control = -0.09048923850059509
R6T max_source_vs_best_control = 1.694591999053955
R6T source_positive_rows = 7
```

判断：

```text
loose/tight denominator safety 没有超过既有 best Rational 2/7；
局部 positive rows 仍主要被 LineC / tail / overhead 拒绝；
promotion_allowed 仍为 0。
```

## 21. Group granularity g8/g16 availability audit

触发原因：

```text
计划第 13.2 节要求 group granularity g8/g16。
当前 v14.2 runner 只能从 v12.35 substrate registry 选择 candidate；
因此先审计当前合法 substrate map 是否已有 RAT-A/B/C/D 对应项。
```

新增脚本：

```text
experiments/analyze_v142_group_granularity_availability.py
```

执行：

```bash
conda run -n kan python -m py_compile experiments/analyze_v142_group_granularity_availability.py
conda run -n kan python experiments/analyze_v142_group_granularity_availability.py
```

输出：

```text
results/v14_2_functional_first_all_basis_parallel/group_granularity_availability_v142/v142_group_granularity_route.json
results/v14_2_functional_first_all_basis_parallel/group_granularity_availability_v142/v142_group_granularity_candidate_inventory.csv
results/v14_2_functional_first_all_basis_parallel/group_granularity_availability_v142/v142_group_granularity_variant_coverage.csv
results/v14_2_functional_first_all_basis_parallel/group_granularity_availability_v142/v142_group_granularity_existing_fms_summary.csv
```

结果：

```text
diagnostic_route = D3-GroupGranularityG8UnavailableG16TritonCoveredNoKANSpecificFMS
official_route_unchanged = R4-FMSNoGoCurrentDefinition
best_repair_route_observed = R3-GenericFunctionalOnlyKANSpecificNotEstablished
uses_existing_artifacts_only = 1
new_training_executed = 0
promotion_allowed = 0
g8_available_in_v1235_substrate_map = 0
g16_triton_candidate_count = 17
g16_triton_v142_substrate_pass_count = 11
best_existing_g16_fms_task_pass_count = 2
```

计划 RAT-A/B/C/D 覆盖情况：

```text
RAT-A GroupRational-Horner-g8: unavailable in current v12.35 substrate map
RAT-B GroupRational-Horner-g16: unavailable in current v12.35 substrate map
RAT-C GroupRational-TritonEval-g8: unavailable in current v12.35 substrate map
RAT-D GroupRational-TritonEval-g16: 17 candidates available, 11 substrate-pass, best existing FMS = 2 tasks
```

判断：

```text
g16 TritonEval 已被现有 D-RAT candidate 覆盖但没有打开 KAN-specific FMS；
g8 与 Horner exact variants 不是当前 v14.2 substrate map 中的合法可选项。
在当前 runner 中强行补 g8/Horner 会变成新 substrate/base map 工作，不能写成 v14.2 official 修复。
```

## 22. R10 train-stream agreement-gated Rational FMS

触发原因：

```text
计划第 13.5 节要求当 FMS 改善 task 但 LineC / tail 坏时，
先做 safety projection rejection / leak source / reduced-plasticity phase schedule；
不能用 LineC / CEp99 / NLL / ECE 作为 direction target。
R6L/R6T 后仍有 source-positive rows 被 LineC/tail/overhead 拒绝，
因此补一个只使用 train-stream per-example gradient agreement 的 R10 diagnostic。
```

代码修改：

```text
experiments/run_v142_functional_first_all_basis_parallel.py
  新增 R10-RationalAgreementGatedFMS
  group_utilities() 增加 mean_gradient_agreement / low_agreement_fraction telemetry
  R10 只按 train-batch gradient agreement 调整 utility 与 role scale
  新增 rational_train_stream_agreement_gated 审计字段
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v142_functional_first_all_basis_parallel.py
```

结果：

```text
py_compile pass
```

R10 focused probe：

```bash
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_r10_agreement --rational-candidate D-RAT28-GroupDiversityPreservingRational --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,R10-RationalAgreementGatedFMS,FCTRL-RandomMatchedNorm --train-steps 80 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 40 --fms-beta 0.99 --fms-update-interval 80 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R4-FMSNoGoCurrentDefinition
minimum_success = S1-SubstrateMapWithFMSExecuted
mlp_fms_task_pass_count = 2
rational_fms_task_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

R10 longer amortized probe：

```bash
conda run -n kan python experiments/run_v142_functional_first_all_basis_parallel.py --out-dir results/v14_2_functional_first_all_basis_parallel/repair_v142_rat28_r10_agreement160 --rational-candidate D-RAT28-GroupDiversityPreservingRational --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier --methods F0-AdamWParallel,R10-RationalAgreementGatedFMS,FCTRL-RandomMatchedNorm --train-steps 160 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --mlp-hidden 24 --trace-interval 80 --fms-beta 0.99 --fms-update-interval 160 --skip-nonrat-fms --compute-budgeted-run 1
```

结果：

```text
route = R4-FMSNoGoCurrentDefinition
minimum_success = S1-SubstrateMapWithFMSExecuted
mlp_fms_task_pass_count = 4
rational_fms_task_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

R10 后重跑 rejection audit：

```bash
conda run -n kan python -m py_compile experiments/analyze_v142_rational_rejection_audit.py experiments/run_v142_functional_first_all_basis_parallel.py
conda run -n kan python experiments/analyze_v142_rational_rejection_audit.py
```

更新后结果：

```text
total_rational_rows_audited = 420
synthetic_gate_pass_rows = 7
source_positive_rejected_rows = 67
linec_rejection_rows = 173
tail_rejection_rows = 146
step_time_rejection_rows = 98
```

R10 摘要：

```text
R10 interval80 task_pass_count = 0
R10 interval80 median_source_vs_best_control = -0.4787488579750061
R10 interval80 max_source_vs_best_control = 1.3013577461242676
R10 interval80 source_positive_rows = 5
R10 interval160 task_pass_count = 0
R10 interval160 median_source_vs_best_control = -0.05639076232910156
R10 interval160 max_source_vs_best_control = 3.5262813568115234
R10 interval160 source_positive_rows = 6
```

判断：

```text
R10 train-stream agreement gating 不能打开 Rational gate；
更长摊销也不能把 source-positive row 转成 task-family pass；
主要拒绝仍来自 LineC / source / AUCtime / tail / step_time 组合。
```

## 23. Non-RAT cross-plan substrate repair audit

触发原因：

```text
用户再次要求未达成则继续。
计划第 13.4 节规定：某 basis substrate 不过时，不能跑该 basis 的 FMS official proof，
只能跑 substrate repair / telemetry diagnostic / minimal smoke。
v14.2 official all-basis sweep 中 Non-RAT families 没有 substrate gate pass；
因此复核 v12.34.2 foreach-off Non-RAT substrate repair artifacts 是否能在 v14.2 strict gate 下打开 substrate。
```

新增脚本：

```text
experiments/analyze_v142_nonrat_crossplan_substrate_repair.py
```

脚本说明：

```text
1. 只读取既有 v12.34.2 foreach-off Non-RAT repair artifacts。
2. 不运行训练，不实例化新模型，不生成 functional direction。
3. 用 v14.2 strict substrate gate 重新检查 raw memory / incremental memory / step / exact no-materialize。
4. 输出 family summary / functional summary / route。
5. promotion_allowed 固定为 0。
```

读取的 source artifacts：

```text
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_nonrat_lifetime_foreachoff_workspace_truth.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_nonrat_lifetime_foreachoff_hardening.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_basis_functional_nonrat_foreachoff_p3.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_basis_functional_nonrat_foreachoff_p4.csv
```

执行：

```bash
conda run -n kan python -m py_compile experiments/analyze_v142_nonrat_crossplan_substrate_repair.py
conda run -n kan python experiments/analyze_v142_nonrat_crossplan_substrate_repair.py
```

输出：

```text
results/v14_2_functional_first_all_basis_parallel/nonrat_crossplan_substrate_repair_v142/v142_nonrat_crossplan_route.json
results/v14_2_functional_first_all_basis_parallel/nonrat_crossplan_substrate_repair_v142/v142_nonrat_crossplan_workspace_repair.csv
results/v14_2_functional_first_all_basis_parallel/nonrat_crossplan_substrate_repair_v142/v142_nonrat_crossplan_family_summary.csv
results/v14_2_functional_first_all_basis_parallel/nonrat_crossplan_substrate_repair_v142/v142_nonrat_crossplan_functional_summary.csv
```

route 结果：

```text
diagnostic_route = D4-NonRATCrossPlanRepairStillFailsV142SubstrateGate
official_route_unchanged = R4-FMSNoGoCurrentDefinition
best_repair_route_observed = R3-GenericFunctionalOnlyKANSpecificNotEstablished
uses_existing_artifacts_only = 1
new_training_executed = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
source_v12342_workspace_rows = 108
source_v12342_workspace_gate_pass_rows = 35
source_v12342_workspace_strong_gate_pass_rows = 0
v142_strict_workspace_pass_rows = 0
nonrat_families_with_v142_strict_workspace_pass = []
source_p3_rows = 135
source_p3_executed_rows = 60
source_p3_pass_rows = 0
source_p4_rows = 60
source_p4_executed_rows = 0
source_p4_pass_rows = 0
```

family summary：

| family | workspace rows | v12.34.2 workspace pass | v14.2 strict pass | exact no-materialize rows | best raw ratio | best incremental ratio | best step ratio | P3 executed | P3 pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-CHE | 27 | 0 | 0 | 27 | 0.9503426374728398 | 4.229325215626585 | 1.0121619112102047 | 0 | 0 |
| D-FOU | 45 | 12 | 0 | 45 | 0.8871469162627444 | 2.311009639776763 | 0.9094278679292677 | 60 | 0 |
| D-RBF | 18 | 18 | 0 | 0 | 0.9184355674410831 | 2.165905631659056 | 1.2743499745399467 | 0 | 0 |
| D-WAV | 18 | 5 | 0 | 0 | 0.8873474845395287 | 2.165905631659056 | 1.3325370259191072 | 0 | 0 |

判断：

```text
1. v12.34.2 中有 35 个 foreach-off workspace pass rows，但按 v14.2 strict gate 重新检查后 strict pass = 0。
2. D-FOU / D-RBF / D-WAV 的 best raw memory 与 step 看起来接近，但 incremental_memory_ratio 仍高于 1.75。
3. D-CHE 有 exact no-materialize rows，但 incremental_memory_ratio 仍最低 4.2293，不能进入 v14.2 substrate。
4. D-FOU 有 P3 functional repair 执行 60 rows，但 P3 pass = 0，不能写成 functional proof。
5. Non-RAT 仍不能进入 v14.2 official FMS proof，不允许 promotion。
```
