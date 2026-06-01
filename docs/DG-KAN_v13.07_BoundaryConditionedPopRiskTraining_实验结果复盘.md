# DG-KAN v13.7 BoundaryConditionedPopRiskTraining 实验结果复盘

生成时间：2026-05-28（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 small-budget diagnostic / fallback row 写成 promotion。

## 1. 计划理解

v13.7 的核心目标是把 v13.6 的 one-shot PopRiskSNR 改成：

```text
boundary-conditioned population-risk training process
```

即：

```text
Population-risk signal selection + Basis-cover boundary schedule
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 label-informed initialization。
3. 不使用 teacher / distillation / loss modification / sampler / class weight / dataset-name branch。
4. validation/test/future/query batch 不能生成 direction。
5. CEp99/NLL/ECE/LineC hard target 只能作为 audit/gate。
6. functional direction 必须来自 train-stream generic loss-interface per-example gradients。
7. 不能把 MLP generic optimizer positive 写成 KAN-specific success。
```

v13.7 的关键 stop rule：

```text
If S1 pass but S2/S3 both fail:
  record PopRiskSNRNoGo_CurrentImplementation.

If S2 pass but S3 fail:
  functional is generic but not KAN-specific;
  continue MLP optimizer line separately.
```

## 2. 本轮代码修改

新增文件：

```text
experiments/run_v137_boundary_conditioned_poprisk_training.py
```

核心实现：

```text
1. 多步训练 loop，不再是一轮 one-shot perturbation。
2. SNRState 持久化 mu/var EMA。
3. 每步按 train-stream batch 计算 per-example gradient。
4. SNR gate 写入真实 named parameters，并更新 AdamW optimizer state。
5. phase schedule:
   - plasticity-open
   - cover-alignment
   - fixed-point-consolidation
6. cover policy:
   - CoverWeak
   - CoverPhaseSchedule
   - CoverConsolidateOnly
   - CoverNoPlasticity diagnostic
7. 输出 S1 sanity、MLP continuous SNR、Rational continuous SNR、basis-cover schedule、Non-RAT repair scout、controls、finalizer artifacts。
```

追加 Case B 审计修复：

```text
1. 新增 rational_boundary_terms(model)。
2. 新增 rational_rejection_audit(rat_rows)。
3. 记录 den_p01 / den_p99 / r_prime_p99 / r_double_prime_p99 / group_diversity。
4. 新增 v137_rational_rejection_audit.csv。
```

合法性说明：

```text
1. 方向不使用 validation/test/future/query batch。
2. LineC/CEp99/NLL/ECE 不生成方向。
3. 所有 functional update 都写真实 model parameters。
4. readout_feature_proxy_only = 0。
5. feature_table_proxy_only = 0。
```

## 3. Smoke

执行规模：

```text
task = X1
seed = 0
train_steps = 6
loss = CE
methods = MLP-AdamW, MLP-AdamW-SNRHard, RAT-AdamW, RAT-BasisSNR-CoverPhaseSchedule
```

结果：

```text
route = R4-PopRiskSNRNoGo_CurrentImplementation
minimum_success = S1-SNRImplementationSanity
s1_implementation_sanity_pass = 1
training_summary_rows = 4
required_artifact_missing_count = 0
```

解释：smoke 证明 runner / persistent SNR / schedule / manifest 可执行，不代表 promotion。

## 4. Official v13.7 small-budget

执行规模：

```text
synthetic_tasks = X1..X7
seeds = 0,1
loss_interfaces = CE,Brier
train_steps = 16
batch_size = 16
methods:
  MLP-AdamW
  MLP-AdamW-SNRHard
  MLP-AdamW-SNRSoft
  MLP-AdamW-SNREMA
  MLP-AdamW-SNRRoleNorm
  RAT-AdamW
  RAT-ParameterSNRHard
  RAT-ParameterSNRSoft
  RAT-ParameterSNREMA
  RAT-GroupSNR
  RAT-GroupSNREMA
  RAT-BasisSNR
  RAT-BasisSNR-CoverWeak
  RAT-BasisSNR-CoverPhaseSchedule
  RAT-BasisSNR-CoverConsolidateOnly
  RAT-BasisSNR-CoverNoPlasticity
```

注意：这是当前回合可闭环的 small-budget 执行，不是计划中的 200-step / 3-seed full budget。不能据此 promotion。

最终 route：

```text
route = R5-GenericSNRPositiveKANSpecificFailed
minimum_success = S2-MLPGenericSNRPositive
official_success_reached = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
final_stop_allowed = 1
s1_implementation_sanity_pass = 1
training_summary_rows = 448
mlp_s2_pass_rows = 21
mlp_s2_task_pass_count = 6
rat_parameter_pass_rows = 5
rat_basis_s3_pass_rows = 10
rat_basis_s3_task_pass_count = 0
cover_s4_pass_rows = 8
cover_s4_task_pass_count = 0
nonrat_substrate_health_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
code_review_packet_sha256 = ef36f124736f9f64f9cb87d261f5d744824f094cb934658bfa439f09ba5c466d
```

S1 sanity：

```text
per_example_gradient_mean_matches_batch_autograd:
  pass = 1
  value = 1.5971319555774244e-07

ab_formula_tiny_linear_signal_dim_ranked_first:
  pass = 1
  value = 1.0078780837357044

noisy_label_toy_snr_not_catastrophic:
  pass = 1
  value = -0.5836887359619141
```

Synthetic task-family summary：

| task | MLP task pass | RAT parameter task pass | RAT basis task pass | cover task pass |
|---|---:|---:|---:|---:|
| X1 | 1 | 0 | 0 | 0 |
| X2 | 1 | 0 | 0 | 0 |
| X3 | 0 | 0 | 0 | 0 |
| X4 | 1 | 0 | 0 | 0 |
| X5 | 1 | 0 | 0 | 0 |
| X6 | 1 | 0 | 0 | 0 |
| X7 | 1 | 0 | 0 | 0 |

解释：

```text
1. v13.7 证明 continuous SNR 与 v13.6 one-shot 不同：MLP-SNR 在 6/7 synthetic family 上出现 generic positive。
2. 但 Rational/KAN basis-SNR 没有达到 task-level S3。
3. cover rows 有局部 pass，但 cover task pass = 0/7。
4. 因 S3/S4 未达成，不允许 real short-run，也不允许 KAN-specific claim。
```

## 5. MLP generic SNR positive

official_v137 中 MLP pass rows：

| method | pass rows | best task | best seed | best loss | best source_vs_adamw | CEp99 delta | ECE delta |
|---|---:|---|---:|---|---:|---:|---:|
| MLP-AdamW-SNRHard | 7 | X4 | 0 | CE | 0.2302527999296431 | 0.11666297912597656 | -0.10956931114196777 |
| MLP-AdamW-SNRSoft | 12 | X5 | 0 | CE | 0.19852748354705907 | -1.0327057838439941 | -0.05878755450248718 |
| MLP-AdamW-SNREMA | 2 | X2 | 0 | CE | 0.21598624250727294 | 0.619532585144043 | 0.0038560032844543457 |
| MLP-AdamW-SNRRoleNorm | 0 | X2 | 0 | Brier | 0.14806906296449796 | 0.2962207794189453 | -0.053937554359436035 |

判断：

```text
MLP-SNR 是本轮真实进展；
但这说明当前 positive 是 generic optimizer signal，不是 KAN-specific。
```

## 6. Rational / KAN no-go

official_v137 中 Rational pass rows：

| method | pass rows | best task | best seed | best loss | best source_vs_adamw | blocker |
|---|---:|---|---:|---|---:|---|
| RAT-ParameterSNRHard | 2 | X7 | 0 | CE | 0.23004331934140776 | task-level seeds insufficient / noise |
| RAT-ParameterSNRSoft | 3 | X3 | 0 | CE | 0.3153950752483805 | task-level seeds insufficient |
| RAT-GroupSNR | 0 | X7 | 1 | CE | 0.3334038405603099 | gate not met |
| RAT-GroupSNREMA | 0 | X7 | 1 | CE | 0.3334511867601011 | gate not met |
| RAT-BasisSNR | 2 | X7 | 1 | CE | 0.3342707874571993 | task-level seeds insufficient |
| RAT-BasisSNR-CoverWeak | 2 | X7 | 1 | CE | 0.3336563244698567 | task-level seeds insufficient |
| RAT-BasisSNR-CoverPhaseSchedule | 2 | X7 | 1 | CE | 0.3332349632549556 | task-level seeds insufficient |
| RAT-BasisSNR-CoverConsolidateOnly | 2 | X7 | 1 | CE | 0.33346269223654224 | task-level seeds insufficient |
| RAT-BasisSNR-CoverNoPlasticity | 2 | X7 | 1 | CE | 0.3338217549743301 | task-level seeds insufficient |

解释：

```text
Rational 有强局部 source improvements，尤其 X7 seed=1 CE；
但 official gate 要求 task-level >=2 seeds 或 coverage，不允许把局部 row 写成 S3。
```

## 7. Non-RAT substrate repair scout

official_v137 执行：

```text
CHE-SNR4 / CHE-SNR5
FOU-SNR4 / FOU-SNR5
RBF-SNR4 / RBF-SNR5
WAV-SNR4 / WAV-SNR5
```

结果：

```text
nonrat_substrate_health_pass_count = 0
functional_proof_allowed = 0 for all Non-RAT rows
```

主要状态：

```text
D-CHE: WorkspaceOnly_NotFunctionalSubstrate
D-FOU: WorkspaceOnly_NotFunctionalSubstrate
D-RBF: WorkspaceOnly_NotFunctionalSubstrate
D-WAV: WorkspaceOnly_NotFunctionalSubstrate
```

解释：Non-RAT 仍不能进入 functional proof。

## 8. Case B fallback：Rational cover / group audit

触发原因：

```text
official route = R5-GenericSNRPositiveKANSpecificFailed
```

按计划 Case B 继续尝试：

```text
1. reduce basis cover guard strength;
2. compare parameter-SNR vs group-SNR;
3. inspect den/r'/r''/group diversity rejection;
4. run Rational no-cover SNR.
```

执行规模：

```text
tasks = X1..X7
seeds = 0,1
loss = CE
train_steps = 16
methods:
  RAT-AdamW
  RAT-ParameterSNRHard
  RAT-ParameterSNRSoft
  RAT-GroupSNR
  RAT-GroupSNREMA
  RAT-BasisSNR
  RAT-BasisSNR-CoverWeak
  RAT-BasisSNR-CoverNoPlasticity
  RAT-BasisSNR-CoverPhaseSchedule
```

结果：

```text
route = R4-PopRiskSNRNoGo_CurrentImplementation
minimum_success = S1-SNRImplementationSanity
rat_parameter_pass_rows = 2
rat_basis_s3_pass_rows = 4
rat_basis_s3_task_pass_count = 0
cover_s4_pass_rows = 3
cover_s4_task_pass_count = 0
required_artifact_missing_count = 0
```

Case B 判断：

```text
1. CoverWeak / CoverNoPlasticity / CoverPhaseSchedule 都没有打开 task-level S3。
2. No-cover 也只在 X7 seed=1 局部 pass，不能说明 cover boundary 是主要 blocker。
3. GroupSNR / GroupSNREMA pass rows = 0，而 ParameterSNRSoft 有 2 rows；group telemetry 当前弱于 parameter-level。
4. 但 ParameterSNR 也未达 task-level pass，因此不能转为 KAN-specific success。
```

Rational rejection audit：

```text
audit_rows = 378
rejection_reason none = 358
rejection_reason cover_boundary_rejected = 20
den_p01_min = 0.012837760150432587
r_prime_p99_max = 0.5169287323951721
r_double_prime_p99_max = 0.3387432396411896
group_diversity_min = 0.6490468978881836
```

解释：

```text
1. den/r'/r''/group diversity audit 没有显示明显 boundary catastrophe。
2. cover rejection 占比约 5.3%，不是主导失败。
3. 更可能的 blocker 是 Rational basis/group telemetry 没有形成跨 seed/task 稳定优势。
```

## 9. Required artifacts

official_v137 required manifest：

```text
manifest_rows = 33
missing_required = 0
```

fallback_case_b required manifest：

```text
manifest_rows = 34
missing_required = 0
```

主要 artifact：

```text
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_route_decision.json
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_implementation_readback.csv
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_snr_sanity_checks.csv
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_mlp_snr_training.csv
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_rational_snr_training.csv
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_basis_cover_schedule.csv
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_nonrat_substrate_repair.csv
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_training_controls.csv
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_synthetic_family_summary.csv
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_linec_audit.csv
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_forbidden_information_audit.csv
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_real_triage_if_opened.csv
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_failure_table.csv
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_progress_table.csv
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_no_go_boundary.md
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_next_hypothesis_queue.md
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_code_review_packet.zip
results/v13_7_boundary_conditioned_poprisk_training/fallback_case_b_rational_cover_audit_v137/v137_rational_rejection_audit.csv
```

## 10. 最终科学结论

v13.7 没有达成 S3/S4/S5；不允许 promotion。最终合法判断：

```text
official_v137 route = R5-GenericSNRPositiveKANSpecificFailed
minimum_success = S2-MLPGenericSNRPositive
promotion_allowed = 0
real_short_run_open_allowed = 0
```

已闭合事实：

```text
1. S1 implementation sanity 通过。
2. continuous training route 和 v13.6 one-shot 不同；MLP-SNR 在 small-budget 下达到 6/7 task pass。
3. Rational/KAN basis-SNR 没有达到 S3。
4. cover-boundary schedule 没有达到 S4。
5. Case B fallback 已执行，no-cover/weak-cover/group/parameter 对照没有打开 task-level S3。
6. Non-RAT substrate repair scout 仍为 0 pass。
7. required artifacts 缺失为 0，forbidden audit 为 0。
```

no-go boundary：

```text
1. 不能把 MLP generic SNR positive 写成 KAN-specific。
2. 不能把 X7 局部 Rational row 写成 S3。
3. 不能进入 real short-run，因为 S4 未打开。
4. 下一步应沿 generic MLP optimizer line 或重新设计 KAN basis/group telemetry，而不是继续 cover 小修。
```

## 11. plan-scale synthetic official 复核与最终结论更新

复核计划后发现上一节 `official_v137` 是 small-budget run，不能替代计划中 synthetic official 的：

```text
steps = 200
seeds = 0,1,2
```

因此补跑 plan-scale synthetic official：

```text
out_dir = results/v13_7_boundary_conditioned_poprisk_training/official_v137_plan_scale
tasks = X1..X7
seeds = 0,1,2
loss_interfaces = CE,Brier
train_size = 96
val_size = 48
train_steps = 200
batch_size = 16
methods = runner default MLP methods + runner default Rational methods
training_summary_rows = 672
```

最终 route：

```text
route = R5-GenericSNRPositiveKANSpecificFailed
minimum_success = S2-MLPGenericSNRPositive
official_success_reached = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
```

关键计数：

```text
s1_implementation_sanity_pass = 1
mlp_s2_pass_rows = 54
mlp_s2_task_pass_count = 7
rat_parameter_pass_rows = 10
rat_basis_s3_pass_rows = 30
rat_basis_s3_task_pass_count = 1
cover_s4_pass_rows = 24
cover_s4_task_pass_count = 1
nonrat_substrate_health_pass_count = 0
code_review_packet_sha256 = afe1446cb8e583bb0161d66512e38ed2665a05abd67045f70b9d1def99adb1a6
```

S1 sanity：

```text
per_example_gradient_mean_matches_batch_autograd pass = 1, value = 1.5971319555774244e-07
ab_formula_tiny_linear_signal_dim_ranked_first pass = 1, value = 1.0078780837357044
noisy_label_toy_snr_not_catastrophic pass = 1, value = -0.5836887359619141
```

Task-level pass：

| task | MLP S2 | RAT parameter | RAT basis S3 | Cover S4 |
|---|---:|---:|---:|---:|
| X1 | 1 | 1 | 1 | 1 |
| X2 | 1 | 0 | 0 | 0 |
| X3 | 1 | 0 | 0 | 0 |
| X4 | 1 | 0 | 0 | 0 |
| X5 | 1 | 0 | 0 | 0 |
| X6 | 1 | 0 | 0 | 0 |
| X7 | 1 | 1 | 0 | 0 |

Method-level pass rows：

| method | pass type | pass rows |
|---|---|---:|
| MLP-AdamW-SNRSoft | MLP S2 | 12 |
| MLP-AdamW-SNREMA | MLP S2 | 28 |
| MLP-AdamW-SNRHard | MLP S2 | 14 |
| RAT-ParameterSNREMA | RAT parameter | 2 |
| RAT-ParameterSNRHard | RAT parameter | 3 |
| RAT-ParameterSNRSoft | RAT parameter | 5 |
| RAT-BasisSNR | RAT basis S3 | 6 |
| RAT-BasisSNR-CoverWeak | RAT basis S3 | 6 |
| RAT-BasisSNR-CoverPhaseSchedule | RAT basis S3 | 6 |
| RAT-BasisSNR-CoverConsolidateOnly | RAT basis S3 | 6 |
| RAT-BasisSNR-CoverNoPlasticity | RAT basis S3 | 6 |
| RAT-BasisSNR-CoverWeak | Cover S4 | 6 |
| RAT-BasisSNR-CoverPhaseSchedule | Cover S4 | 6 |
| RAT-BasisSNR-CoverConsolidateOnly | Cover S4 | 6 |
| RAT-BasisSNR-CoverNoPlasticity | Cover S4 | 6 |

Rational rejection audit：

```text
audit_rows = 2310
rejection_reason none = 2131
rejection_reason cover_boundary_rejected = 179
den_p01_min = 0.012837760150432587
r_prime_p99_max = 0.5169287919998169
r_double_prime_p99_max = 0.33874326944351196
group_diversity_min = 0.6489378213882446
```

解释：

```text
1. MLP-SNR 在 plan-scale synthetic 上 7/7 task pass，S2 成立。
2. 这说明 population-risk SNR 作为 generic optimizer/training-process signal 是 positive。
3. RAT basis-SNR 只有 X1 达到 task-level pass，未达 S3 的 synthetic >=5/7。
4. Cover-boundary schedule 也只有 X1 达到 task-level pass，未达 S4。
5. Non-RAT repair scout 全部仍是 WorkspaceOnly_NotFunctionalSubstrate，不能进入 functional proof。
6. Rational rejection audit 没显示 den/r'/r'' 数值 catastrophe；cover_boundary_rejected = 179/2310，不足以解释 task-level failure。
```

最终科学结论更新：

```text
v13.7 达成 S1 和 S2；
v13.7 没有达成 S3/S4/S5；
不允许 promotion；
不允许 real short-run；
合法 route = R5-GenericSNRPositiveKANSpecificFailed。
```

no-go boundary 更新：

```text
1. v13.7 不是 PopRiskSNR 原理全失败：MLP generic SNR 在 plan-scale synthetic 上成功。
2. 当前失败是 KAN/Rational basis/group telemetry 没有把 generic SNR positive 转化为 KAN-specific basis advantage。
3. 不能把 MLP generic positive 写成 KAN functional promotion。
4. 不能把 X1-only RAT basis/cover success 写成 S3/S4。
5. 下一步应拆出 generic MLP optimizer line，或重新设计 KAN basis/group telemetry 与 substrate/base architecture；继续 cover 小修或阈值网格不是计划推荐方向。
```

## 12. 用户再次追问后的继续推进：Group telemetry soft repair

用户再次要求“未达成则继续”。本次复核 plan-scale route 后，结论仍是：

```text
route = R5-GenericSNRPositiveKANSpecificFailed
minimum_success = S2-MLPGenericSNRPositive
mlp_s2_task_pass_count = 7
rat_basis_s3_task_pass_count = 1
cover_s4_task_pass_count = 1
promotion_allowed = 0
real_short_run_open_allowed = 0
```

计划 Stop / go rules 对照：

```text
If S2 pass but S3 fail:
  functional is generic but not KAN-specific; continue MLP optimizer line separately.
```

`v137_next_hypothesis_queue.md` 同时写明：

```text
If R5: generic MLP SNR works but KAN does not; debug basis telemetry and Rational substrate interaction.
```

因此本次没有继续 cover threshold/grid search，也没有打开 real short-run，而是追加 Case B focused repair：测试 group telemetry 是否只是 hard gate 过硬。runner 已经支持 method 名中包含 `GroupSNR` + `Soft` 的 soft group gate，所以本次无需修改 promotion gate，只补跑：

```text
RAT-GroupSNRSoft
RAT-GroupSNREMASoft
```

执行规模：

```text
out_dir = results/v13_7_boundary_conditioned_poprisk_training/fallback_case_b_group_soft_repair_v137
tasks = X1..X7
seeds = 0,1,2
loss_interface = CE
train_steps = 200
methods:
  RAT-AdamW
  RAT-ParameterSNRSoft
  RAT-GroupSNR
  RAT-GroupSNRSoft
  RAT-GroupSNREMA
  RAT-GroupSNREMASoft
  RAT-BasisSNR
```

结果：

```text
route = R4-PopRiskSNRNoGo_CurrentImplementation
minimum_success = S1-SNRImplementationSanity
training_summary_rows = 147
rat_parameter_pass_rows = 0
rat_basis_s3_pass_rows = 0
rat_basis_s3_task_pass_count = 0
cover_s4_pass_rows = 0
cover_s4_task_pass_count = 0
required_artifact_missing_count = 0
code_review_packet_sha256 = 2b3e79b4260337e9bee448ca57fdfef409ac84499e1fbddfcbb366255feb126f
```

Method-level pass rows：

| method | pass rows |
|---|---:|
| RAT-ParameterSNRSoft | 0 |
| RAT-GroupSNR | 0 |
| RAT-GroupSNRSoft | 0 |
| RAT-GroupSNREMA | 0 |
| RAT-GroupSNREMASoft | 0 |
| RAT-BasisSNR | 0 |

最接近但不能 pass 的 rows：

| task | seed | method | source_vs_adamw | CEp99 delta | NoiseSignalLeak delta | Reservoir delta | pass |
|---|---:|---|---:|---:|---:|---:|---:|
| X5 | 0 | RAT-GroupSNREMA | 0.4459215658243332 | -3.704286575317383 | -0.00009906291961669922 | -5.656522274017334 | 0 |
| X5 | 0 | RAT-GroupSNREMASoft | 0.4457548908265303 | -3.702035427093506 | 0.00022870302200317383 | -5.657006502151489 | 0 |
| X5 | 0 | RAT-GroupSNRSoft | 0.4447917404111912 | -3.6889991760253906 | 0.002164483070373535 | -5.656068325042725 | 0 |

Rational rejection audit：

```text
audit_rows = 735
rejection_reason none = 717
rejection_reason cover_boundary_rejected = 18
den_p01_min = 0.012837760150432587
r_prime_p99_max = 0.5169287323951721
r_double_prime_p99_max = 0.3387432396411896
group_diversity_min = 0.6501316428184509
```

判断：

```text
1. GroupSNRSoft / GroupSNREMASoft 没有打开 S3。
2. 失败不是单纯 group hard gate 过硬。
3. CE-only focused run 中 parameter/group/basis pass rows 均为 0，说明 KAN-specific telemetry 在这个 slice 更不稳定。
4. v13.7 的合法 final route 仍以 plan-scale official 为准：
   R5-GenericSNRPositiveKANSpecificFailed。
5. 当前计划内不应继续 cover/threshold 小修；下一步应拆出 generic MLP optimizer line，或重新设计 KAN basis/group telemetry 与 substrate/base architecture。
```

## 13. 用户再次追问后的继续推进：MLP generic real triage line

用户再次要求“未达成则继续”。本次复核 v13.7 计划 stop/go rule 后，当前 KAN functional route 仍为：

```text
R5-GenericSNRPositiveKANSpecificFailed
```

计划允许的继续方向是：

```text
If S2 pass but S3 fail:
  functional is generic but not KAN-specific; continue MLP optimizer line separately.
```

因此本次没有继续 KAN cover threshold / small grid，也没有打开 KAN real short-run，而是新增 MLP-only real triage runner，用来检查 synthetic 上成立的 MLP generic SNR 是否在真实数据短训 triage 中仍有可见优势。

新增文件：

```text
experiments/run_v137_mlp_snr_real_triage.py
```

主要修改：

```text
1. 新增 MLP-only real triage，不允许写成 KAN promotion。
2. 使用 MLPBaseline 和 train-stream generic loss-interface cotangent。
3. 为 MLPBaseline 增加解析 per-example gradient，避免逐样本 autograd 太慢。
4. 支持 MLP-AdamW、MLP-AdamW-SNRHard、MLP-AdamW-SNRSoft、MLP-AdamW-SNREMA、MLP-AdamW-SNRRoleNorm。
5. validation/test/CEp99/NLL/ECE 只作为 audit/gate，不生成方向。
6. dataset name 只用于数据加载，不进入 optimizer direction。
7. 输出 route、training、summary、family_summary、forbidden_audit、required_manifest。
```

实现 blocker 与修复：

```text
blocker 1:
  relative out_dir 无法 relative_to(ROOT)，导致 smoke 直接报 ValueError。

fix:
  main() 中把相对 out_dir 解析到 ROOT 下；
  write_manifest() 中使用 path.resolve().relative_to(ROOT)，失败时保留绝对路径。

blocker 2:
  route 在 final manifest 刷新前写入，导致 smoke route 中 missing_required_count = 2。

fix:
  先写 route，再刷新 manifest，再用最终 missing_required_count 重写 route。
```

语法检查：

```text
conda run -n kan python -m py_compile experiments/run_v137_mlp_snr_real_triage.py
py_compile pass
```

smoke 结果：

```text
route = R5-MLPGenericRealTriageNotConfirmed
mlp_real_triage_rows = 2
mlp_real_triage_dataset_count = 1
mlp_real_triage_dataset_pass_count = 0
failure_rows = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
```

official real triage 执行规模：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods = MLP-AdamW,MLP-AdamW-SNRHard,MLP-AdamW-SNRSoft,MLP-AdamW-SNREMA,MLP-AdamW-SNRRoleNorm
train_size = 1024
val_size = 512
test_size = 512
epochs = 3
batch_size = 64
hidden_dim = 160
loss_interface = CE
```

official real triage route：

```text
route = R5-MLPGenericRealTriageNotConfirmed
minimum_success = S2-MLPGenericSNRPositive
mlp_real_triage_rows = 45
mlp_real_triage_dataset_count = 3
mlp_real_triage_dataset_pass_count = 0
failure_rows = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
kan_promotion_allowed = 0
real_short_run_open_allowed_for_kan = 0
```

dataset summary：

| dataset | seed pass count | dataset pass |
|---|---:|---:|
| Fashion-MNIST | 0 | 0 |
| KMNIST | 0 | 0 |
| MNIST | 0 | 0 |

最接近但不能 pass 的 rows：

| dataset | seed | method | source_vs_adamw | AUC time ratio | CEp99 delta | ECE delta | real triage pass |
|---|---:|---|---:|---:|---:|---:|---:|
| MNIST | 0 | MLP-AdamW-SNRSoft | 0.012638111652964246 | 0.9761460832315737 | 0.5351014137268066 | 0.010010592639446259 | 0 |
| Fashion-MNIST | 2 | MLP-AdamW-SNRSoft | 0.0008702662863998567 | 0.9988081263623956 | 0.3193988800048828 | -0.01061207801103592 | 0 |
| KMNIST | 2 | MLP-AdamW-SNRSoft | -0.026049696772194597 | 1.0288090300035826 | 0.2673001289367676 | 0.011949315667152405 | 0 |

判断：

```text
1. MLP generic SNR 在 synthetic plan-scale 上成立，但 real triage 未确认。
2. MNIST / Fashion-MNIST 有局部 source/AUC 改善，但 CEp99 坏化，不能写成 pass。
3. KMNIST 没有 source/AUC 改善。
4. 该分支是 MLP-only optimizer triage，不改变 v13.7 的 KAN functional route。
5. KAN promotion_allowed 仍为 0；KAN real_short_run_open_allowed 仍为 0。
```

最终科学结论更新：

```text
v13.7 达成 S1 和 synthetic S2；
v13.7 没有达成 KAN-specific S3/S4/S5；
MLP real triage 没有确认 synthetic S2 能直接迁移到真实数据短训；
不允许 KAN promotion；
不允许 KAN real short-run；
合法 KAN route 仍为 R5-GenericSNRPositiveKANSpecificFailed。
```

## 14. 用户再次追问后的继续推进：MLP Blend trust repair

用户再次要求“未达成则继续”。本次重新对照 v13.7 stop/go rule：

```text
If S2 pass but S3 fail:
  functional is generic but not KAN-specific; continue MLP optimizer line separately.
```

因此继续推进 MLP-only optimizer line。上一轮 real triage blocker 是：

```text
1. MNIST / Fashion-MNIST 有局部 source/AUC 改善，但 CEp99 坏化。
2. KMNIST 没有稳定 source/AUC 改善。
```

本次修复思路：

```text
用 train-stream trust/blend 约束 SNR update：
direction = alpha * SNR_filtered_train_batch_gradient
          + (1-alpha) * AdamW_train_batch_gradient

新增方法语义：
  Blend25 / Blend50 / Blend75

该方法不读取 CEp99/NLL/ECE/LineC/validation/test/future 来生成方向。
```

代码修改：

```text
experiments/run_v137_mlp_snr_real_triage.py
  train_one() 增加 Blend25/Blend50/Blend75 解析；
  SNR gate 后混合同 batch AdamW gradient；
  training rows 记录 snr_blend_alpha；
  blend 后重新记录 cos_snr_adamw / removed_update_norm_fraction；
  removed_update_norm_fraction clamp 到 [0,1]。
```

语法检查：

```text
conda run -n kan python -m py_compile experiments/run_v137_mlp_snr_real_triage.py experiments/run_v137_boundary_conditioned_poprisk_training.py
py_compile pass
```

Blend smoke：

```text
route = R5-MLPGenericRealTriagePass
mlp_real_triage_rows = 2
mlp_real_triage_dataset_count = 1
mlp_real_triage_dataset_pass_count = 1
failure_rows = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
```

解释：

```text
smoke 只证明 Blend 方法可执行，并不能写成 official success。
```

### Blend25 focused repair

结果：

```text
route = R5-MLPGenericRealTriageNotConfirmed
mlp_real_triage_rows = 36
mlp_real_triage_dataset_count = 3
mlp_real_triage_dataset_pass_count = 1
failure_rows = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
```

dataset summary：

| dataset | seed pass count | dataset pass |
|---|---:|---:|
| Fashion-MNIST | 2 | 1 |
| KMNIST | 1 | 0 |
| MNIST | 1 | 0 |

### Blend50/75 focused repair

结果：

```text
route = R5-MLPGenericRealTriageNotConfirmed
mlp_real_triage_rows = 45
mlp_real_triage_dataset_count = 3
mlp_real_triage_dataset_pass_count = 2
failure_rows = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
```

dataset summary：

| dataset | seed pass count | dataset pass |
|---|---:|---:|
| Fashion-MNIST | 1 | 0 |
| KMNIST | 2 | 1 |
| MNIST | 3 | 1 |

### Combined Blend official repair

执行规模：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods =
  MLP-AdamW
  MLP-AdamW-SNRSoft-Blend25
  MLP-AdamW-SNREMA-Blend25
  MLP-AdamW-SNRRoleNorm-Blend25
  MLP-AdamW-SNREMA-Blend50
  MLP-AdamW-SNRRoleNorm-Blend50
  MLP-AdamW-SNREMA-Blend75
  MLP-AdamW-SNRRoleNorm-Blend75
train_size = 1024
val_size = 512
test_size = 512
epochs = 3
batch_size = 64
hidden_dim = 160
loss_interface = CE
```

最终 route：

```text
route = R5-MLPGenericRealTriagePass
minimum_success = S2-MLPGenericSNRPositive
mlp_real_triage_rows = 72
mlp_real_triage_dataset_count = 3
mlp_real_triage_dataset_pass_count = 3
failure_rows = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
kan_promotion_allowed = 0
real_short_run_open_allowed_for_kan = 0
```

family summary：

| dataset | seed pass count | dataset pass | passing seeds |
|---|---:|---:|---|
| Fashion-MNIST | 2 | 1 | 0;2 |
| KMNIST | 3 | 1 | 0;1;2 |
| MNIST | 2 | 1 | 0;2 |

代表性 pass rows：

| dataset | seed | method | source_vs_adamw | AUC time ratio | CEp99 delta | ECE delta |
|---|---:|---|---:|---:|---:|---:|
| MNIST | 0 | MLP-AdamW-SNREMA-Blend50 | 0.05402895145299752 | 0.8981513286015965 | -0.04118680953979492 | -0.005311731249094009 |
| Fashion-MNIST | 2 | MLP-AdamW-SNRRoleNorm-Blend25 | 0.01777078028907797 | 0.9756575492560549 | -0.6446499824523926 | 0.009411342442035675 |
| KMNIST | 1 | MLP-AdamW-SNREMA-Blend50 | 0.028674350792256886 | 0.9631843894548472 | -0.8646512031555176 | -0.00255429744720459 |

判断：

```text
1. MLP generic optimizer line 已经通过 real triage。
2. 该结果仍然是 MLP-only continuation，不能写成 KAN functional promotion。
3. KAN plan-scale route 仍是 R5-GenericSNRPositiveKANSpecificFailed。
4. v13.7 的科学结论更清楚：
   PopRisk-SNR / blend optimizer 作为 generic MLP training signal 有价值；
   当前 KAN Rational basis/group telemetry 未能承接这个 generic optimizer value。
5. 不允许 KAN promotion；不允许 KAN real short-run。
```

最终科学结论更新：

```text
v13.7 达成 S1；
v13.7 达成 S2，并且 S2 现在同时有 synthetic MLP pass 与 MLP real triage pass；
v13.7 没有达成 KAN-specific S3/S4/S5；
合法 KAN route 仍为 R5-GenericSNRPositiveKANSpecificFailed；
MLP-only continuation route = R5-MLPGenericRealTriagePass；
promotion_allowed = 0 for KAN；
real_short_run_open_allowed_for_kan = 0。
```

## 15. 用户再次追问后的继续推进：MLP Blend 5/10-seed confirmation

用户再次要求“未达成则继续”。本次复核计划：

```text
If S5 pass:
  begin 5-seed / 10-seed confirmation.
```

边界说明：

```text
KAN 没有 S5；
MLP-only combined Blend 已经通过 3x3 real triage；
因此本次只对 MLP optimizer line 做 5/10-seed confirmation，不能写成 KAN promotion。
```

代码审计修复：

```text
experiments/run_v137_mlp_snr_real_triage.py
  family_summary() 增加 seed_threshold_override；
  CLI 增加 --dataset-pass-seed-threshold。
```

修复原因：

```text
原 3-seed triage 阈值为 2；
5/10-seed confirmation 不能继续使用 2-seed 阈值，否则会过松。
本次 5-seed 使用 threshold=3，10-seed 使用 threshold=6。
```

语法检查：

```text
conda run -n kan python -m py_compile experiments/run_v137_mlp_snr_real_triage.py
py_compile pass
```

### 5-seed confirmation

结果：

```text
route = R5-MLPGenericRealTriagePass
mlp_real_triage_rows = 120
mlp_real_triage_dataset_count = 3
mlp_real_triage_dataset_pass_count = 3
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
kan_promotion_allowed = 0
real_short_run_open_allowed_for_kan = 0
```

family summary：

| dataset | seed pass count | threshold | dataset pass | passing seeds |
|---|---:|---:|---:|---|
| Fashion-MNIST | 3 | 3 | 1 | 0;2;3 |
| KMNIST | 5 | 3 | 1 | 0;1;2;3;4 |
| MNIST | 3 | 3 | 1 | 0;2;3 |

### 10-seed confirmation

结果：

```text
route = R5-MLPGenericRealTriageNotConfirmed
mlp_real_triage_rows = 240
mlp_real_triage_dataset_count = 3
mlp_real_triage_dataset_pass_count = 1
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
kan_promotion_allowed = 0
real_short_run_open_allowed_for_kan = 0
```

family summary：

| dataset | seed pass count | threshold | dataset pass | passing seeds |
|---|---:|---:|---:|---|
| Fashion-MNIST | 3 | 6 | 0 | 0;2;5 |
| KMNIST | 9 | 6 | 1 | 0;1;2;3;4;6;7;8;9 |
| MNIST | 5 | 6 | 0 | 0;2;3;8;9 |

10-seed best pass rows：

| dataset | seed | method | source_vs_adamw | AUC time ratio | CEp99 delta | ECE delta |
|---|---:|---|---:|---:|---:|---:|
| Fashion-MNIST | 2 | MLP-AdamW-SNRRoleNorm-Blend25 | 0.01436830614479312 | 0.9802654287435487 | -0.6446499824523926 | 0.009411342442035675 |
| KMNIST | 7 | MLP-AdamW-SNRRoleNorm-Blend50 | 0.055924443389259615 | 0.9325213872409402 | -1.1899166107177734 | -0.007198957726359367 |
| MNIST | 0 | MLP-AdamW-SNREMA-Blend25 | 0.051681790817533435 | 0.9024964325518074 | -0.14273452758789062 | -0.00011614710092544556 |

10-seed failure pattern：

```text
1. KMNIST robust: 9/10 seeds pass。
2. MNIST borderline: 5/10 seeds pass，未达到 6/10 confirmation threshold。
3. Fashion-MNIST weak: 3/10 seeds pass。
4. 多个 non-pass row 有 positive source/AUC，但 CEp99_delta > 0.05，不能写成 pass。
```

最终判断更新：

```text
1. v13.7 KAN 目标仍未达成：
   route = R5-GenericSNRPositiveKANSpecificFailed。
2. MLP-only Blend line 达到 5-seed confirmation。
3. MLP-only Blend line 没有达到 10-seed confirmation。
4. 不能把 5-seed MLP pass 写成 final generic optimizer success。
5. 不能把 MLP-only result 写成 KAN promotion。
6. KAN promotion_allowed = 0。
7. KAN real_short_run_open_allowed = 0。
```
