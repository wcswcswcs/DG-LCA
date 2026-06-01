# DG-KAN v13.7 BoundaryConditionedPopRiskTraining 执行日志

生成时间：2026-05-28（Asia/Singapore）

本日志记录实际执行过的命令、artifact 路径和复现信息；不补填未执行命令。

## 1. 计划读取与实现入口

计划文件：

```text
docs/DG-KAN_v13.7_BoundaryConditionedPopRiskTraining_完整计划.md
```

新增 runner：

```text
experiments/run_v137_boundary_conditioned_poprisk_training.py
```

实现要点：

```text
1. 从 v13.6 复用 synthetic_data / eval_metrics / loss-interface cotangent / per-example gradient / cover_debt。
2. 新增 persistent SNRState，在多步训练 loop 中持续更新 mu/var EMA。
3. 新增 phase_for_step：plasticity-open / cover-alignment / fixed-point-consolidation。
4. 新增 cover policy：CoverWeak / CoverPhaseSchedule / CoverConsolidateOnly / CoverNoPlasticity。
5. functional direction 只来自 train-stream loss-interface per-example gradients。
6. validation metrics / LineC / CEp99 / NLL / ECE 只在 step 后审计，不参与方向生成。
7. 输出 implementation readback、sanity checks、MLP/RAT continuous training、Non-RAT repair scout、controls、route、manifest、code packet。
```

## 2. 语法检查

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v137_boundary_conditioned_poprisk_training.py
```

结果：

```text
py_compile pass
```

## 3. Smoke

执行：

```bash
conda run -n kan python experiments/run_v137_boundary_conditioned_poprisk_training.py \
  --out-dir results/v13_7_boundary_conditioned_poprisk_training/smoke_v137 \
  --synthetic-tasks X1 \
  --synthetic-seeds 0 \
  --synthetic-train-size 32 \
  --synthetic-val-size 16 \
  --train-steps 6 \
  --batch-size 8 \
  --log-interval 3 \
  --loss-interfaces CE \
  --mlp-methods MLP-AdamW,MLP-AdamW-SNRHard \
  --rational-methods RAT-AdamW,RAT-BasisSNR-CoverPhaseSchedule \
  --mlp-hidden 64
```

结果：

```text
route = R4-PopRiskSNRNoGo_CurrentImplementation
minimum_success = S1-SNRImplementationSanity
s1_implementation_sanity_pass = 1
training_summary_rows = 4
mlp_s2_pass_rows = 0
rat_basis_s3_pass_rows = 0
cover_s4_pass_rows = 0
required_artifact_missing_count = 0
code_review_packet_sha256 = e4d7f2e19f824dcdf2f5709cea6655fc0806f5fde1c405f693633ba2776e53ed
```

解释：smoke 只证明 runner / persistent SNR state / artifact surface 可执行，不能 promotion。

## 4. official_v137 small-budget run

注意：本次 official_v137 是当前回合可闭环的 small-budget 执行；不是计划里 synthetic 200 steps / seeds 0,1,2 的 full budget。不可把它写成 S5 或最终 promotion。

执行：

```bash
conda run -n kan python experiments/run_v137_boundary_conditioned_poprisk_training.py \
  --out-dir results/v13_7_boundary_conditioned_poprisk_training/official_v137 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0,1 \
  --synthetic-train-size 64 \
  --synthetic-val-size 32 \
  --train-steps 16 \
  --batch-size 16 \
  --log-interval 8 \
  --loss-interfaces CE,Brier \
  --mlp-hidden 160
```

结果：

```text
route = R5-GenericSNRPositiveKANSpecificFailed
minimum_success = S2-MLPGenericSNRPositive
official_success_reached = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
s1_implementation_sanity_pass = 1
synthetic_task_count = 7
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

Sanity checks：

```text
per_example_gradient_mean_matches_batch_autograd = pass, value = 1.5971319555774244e-07
ab_formula_tiny_linear_signal_dim_ranked_first = pass, value = 1.0078780837357044
noisy_label_toy_snr_not_catastrophic = pass, value = -0.5836887359619141
```

Family summary：

```text
MLP task pass = 6/7
RAT parameter task pass = 0/7
RAT basis task pass = 0/7
Cover task pass = 0/7
```

主要 method pass rows：

```text
MLP-AdamW-SNRHard pass rows = 7
MLP-AdamW-SNRSoft pass rows = 12
MLP-AdamW-SNREMA pass rows = 2
MLP-AdamW-SNRRoleNorm pass rows = 0
RAT-ParameterSNRHard pass rows = 2
RAT-ParameterSNRSoft pass rows = 3
RAT-BasisSNR pass rows = 2
RAT-BasisSNR-CoverWeak pass rows = 2
RAT-BasisSNR-CoverPhaseSchedule pass rows = 2
RAT-BasisSNR-CoverConsolidateOnly pass rows = 2
RAT-BasisSNR-CoverNoPlasticity pass rows = 2
```

判断：当前 small-budget 下 MLP continuous SNR 触发 S2，但 KAN/Rational S3 未打开；按计划进入 Case B。

## 5. Case B 代码补充：Rational rejection audit

触发原因：计划 Case B 要求在 MLP-SNR pass、RAT-SNR fail 时检查：

```text
1. reduce basis cover guard strength;
2. compare parameter-SNR vs group-SNR;
3. inspect den/r'/r''/group diversity rejection;
4. run Rational no-cover SNR;
5. if no-cover SNR works, cover boundary is too strong;
6. if parameter-SNR works but group-SNR fails, group telemetry wrong.
```

代码修改：

```text
experiments/run_v137_boundary_conditioned_poprisk_training.py
  新增 rational_boundary_terms(model)。
  新增 rational_rejection_audit(rat_rows)。
  训练日志增加 den_p01 / den_p99 / r_prime_p99 / r_double_prime_p99 / group_diversity。
  新增 artifact: v137_rational_rejection_audit.csv。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v137_boundary_conditioned_poprisk_training.py
```

结果：

```text
py_compile pass
```

## 6. Case B focused fallback

执行：

```bash
conda run -n kan python experiments/run_v137_boundary_conditioned_poprisk_training.py \
  --out-dir results/v13_7_boundary_conditioned_poprisk_training/fallback_case_b_rational_cover_audit_v137 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0,1 \
  --synthetic-train-size 64 \
  --synthetic-val-size 32 \
  --train-steps 16 \
  --batch-size 16 \
  --log-interval 8 \
  --loss-interfaces CE \
  --mlp-methods MLP-AdamW \
  --rational-methods RAT-AdamW,RAT-ParameterSNRHard,RAT-ParameterSNRSoft,RAT-GroupSNR,RAT-GroupSNREMA,RAT-BasisSNR,RAT-BasisSNR-CoverWeak,RAT-BasisSNR-CoverNoPlasticity,RAT-BasisSNR-CoverPhaseSchedule \
  --mlp-hidden 160
```

结果：

```text
route = R4-PopRiskSNRNoGo_CurrentImplementation
minimum_success = S1-SNRImplementationSanity
training_summary_rows = 140
rat_parameter_pass_rows = 2
rat_basis_s3_pass_rows = 4
rat_basis_s3_task_pass_count = 0
cover_s4_pass_rows = 3
cover_s4_task_pass_count = 0
required_artifact_missing_count = 0
code_review_packet_sha256 = f16203ada5fb196f9abce4929dd761f43875c3b52616812383150baea9702e2e
```

Case B method comparison：

```text
RAT-ParameterSNRSoft pass rows = 2
RAT-ParameterSNRHard pass rows = 0
RAT-GroupSNR pass rows = 0
RAT-GroupSNREMA pass rows = 0
RAT-BasisSNR pass rows = 1
RAT-BasisSNR-CoverWeak pass rows = 1
RAT-BasisSNR-CoverNoPlasticity pass rows = 1
RAT-BasisSNR-CoverPhaseSchedule pass rows = 1
task-level S3/S4 pass = 0/7
```

Rational rejection audit：

```text
audit rows = 378
rejection_reason none = 358
rejection_reason cover_boundary_rejected = 20
den_p01_min = 0.012837760150432587
r_prime_p99_max = 0.5169287323951721
r_double_prime_p99_max = 0.3387432396411896
group_diversity_min ≈ 0.6490468978881836
```

判断：

```text
1. No-cover / CoverWeak / CoverPhaseSchedule 都没有打开 task-level S3。
2. cover_boundary_rejected 只有 20/378；不是 cover boundary 过强导致的主要失败。
3. ParameterSNRSoft 有局部 pass，GroupSNR/GroupSNREMA 为 0，说明 group telemetry / grouping 可能比 parameter-level 更弱。
4. 但 parameter-level 也只有 2 rows，未达 task-level pass，不足以支持 KAN-specific claim。
5. v13.7 仍不允许 promotion，不允许 real short-run。
```

## 7. Required artifacts

official_v137：

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
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_required_manifest.csv
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_no_go_boundary.md
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_next_hypothesis_queue.md
results/v13_7_boundary_conditioned_poprisk_training/official_v137/v137_code_review_packet.zip
```

Manifest：

```text
official_v137 manifest_rows = 33
official_v137 missing_required = 0
fallback_case_b_rational_cover_audit_v137 manifest_rows = 34
fallback_case_b_rational_cover_audit_v137 missing_required = 0
```

## 8. plan-scale synthetic official run

复核计划后发现 `official_v137` 是 small-budget run，不能替代计划中 synthetic `steps=200, seeds=0,1,2` 的 official 证据。因此补跑 plan-scale synthetic official。

执行：

```bash
conda run -n kan python experiments/run_v137_boundary_conditioned_poprisk_training.py \
  --out-dir results/v13_7_boundary_conditioned_poprisk_training/official_v137_plan_scale \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0,1,2 \
  --synthetic-train-size 96 \
  --synthetic-val-size 48 \
  --train-steps 200 \
  --batch-size 16 \
  --log-interval 50 \
  --loss-interfaces CE,Brier \
  --mlp-hidden 160
```

执行说明：

```text
synthetic tasks = X1..X7
seeds = 0,1,2
loss interfaces = CE,Brier
methods = runner default MLP methods + runner default Rational methods
train_steps = 200
training_summary_rows = 672
terminal observed runtime ≈ 43 minutes
```

结果：

```text
route = R5-GenericSNRPositiveKANSpecificFailed
minimum_success = S2-MLPGenericSNRPositive
official_success_reached = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
final_stop_allowed = 1
s1_implementation_sanity_pass = 1
mlp_s2_pass_rows = 54
mlp_s2_task_pass_count = 7
rat_parameter_pass_rows = 10
rat_basis_s3_pass_rows = 30
rat_basis_s3_task_pass_count = 1
cover_s4_pass_rows = 24
cover_s4_task_pass_count = 1
nonrat_substrate_health_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
code_review_packet_sha256 = afe1446cb8e583bb0161d66512e38ed2665a05abd67045f70b9d1def99adb1a6
```

S1 sanity：

```text
per_example_gradient_mean_matches_batch_autograd pass = 1, value = 1.5971319555774244e-07
ab_formula_tiny_linear_signal_dim_ranked_first pass = 1, value = 1.0078780837357044
noisy_label_toy_snr_not_catastrophic pass = 1, value = -0.5836887359619141
```

Method pass rows：

```text
MLP-AdamW-SNRSoft mlp_s2_pass_rows = 12
MLP-AdamW-SNREMA mlp_s2_pass_rows = 28
MLP-AdamW-SNRHard mlp_s2_pass_rows = 14

RAT-ParameterSNREMA rat_param_pass_rows = 2
RAT-ParameterSNRHard rat_param_pass_rows = 3
RAT-ParameterSNRSoft rat_param_pass_rows = 5

RAT-BasisSNR rat_basis_s3_pass_rows = 6
RAT-BasisSNR-CoverWeak rat_basis_s3_pass_rows = 6
RAT-BasisSNR-CoverPhaseSchedule rat_basis_s3_pass_rows = 6
RAT-BasisSNR-CoverConsolidateOnly rat_basis_s3_pass_rows = 6
RAT-BasisSNR-CoverNoPlasticity rat_basis_s3_pass_rows = 6

RAT-BasisSNR-CoverWeak cover_s4_pass_rows = 6
RAT-BasisSNR-CoverPhaseSchedule cover_s4_pass_rows = 6
RAT-BasisSNR-CoverConsolidateOnly cover_s4_pass_rows = 6
RAT-BasisSNR-CoverNoPlasticity cover_s4_pass_rows = 6
```

Task-level summary：

```text
X1: MLP pass = 1, RAT parameter pass = 1, RAT basis pass = 1, cover pass = 1
X2: MLP pass = 1, RAT parameter pass = 0, RAT basis pass = 0, cover pass = 0
X3: MLP pass = 1, RAT parameter pass = 0, RAT basis pass = 0, cover pass = 0
X4: MLP pass = 1, RAT parameter pass = 0, RAT basis pass = 0, cover pass = 0
X5: MLP pass = 1, RAT parameter pass = 0, RAT basis pass = 0, cover pass = 0
X6: MLP pass = 1, RAT parameter pass = 0, RAT basis pass = 0, cover pass = 0
X7: MLP pass = 1, RAT parameter pass = 1, RAT basis pass = 0, cover pass = 0
```

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

Non-RAT：

```text
v137_nonrat_substrate_repair.csv rows = 8
status = WorkspaceOnly_NotFunctionalSubstrate for all 8 rows
substrate_health_pass = 0
```

Manifest：

```text
official_v137_plan_scale manifest_rows = 34
official_v137_plan_scale missing_required = 0
```

判断：

```text
1. v13.7 plan-scale synthetic run 证明 MLP continuous SNR 是 generic positive：MLP task pass = 7/7。
2. Rational parameter-SNR 只有 X1 / X7 task-level pass；RAT basis-SNR 和 cover-boundary schedule 只有 X1 task-level pass。
3. 因 RAT basis-SNR task pass = 1/7，S3 未达成。
4. 因 cover task pass = 1/7，S4 未达成。
5. 因 S4 未达成，real short-run 不允许打开。
6. 合法 final route 仍为 R5-GenericSNRPositiveKANSpecificFailed，不允许 promotion。
```

收尾语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v137_boundary_conditioned_poprisk_training.py
```

结果：

```text
py_compile pass
```

## 9. 用户追问后继续：Case B group telemetry soft repair

用户再次要求确认 v13.7 是否达成目标，若未达成则继续。复核 plan-scale route 为 `R5-GenericSNRPositiveKANSpecificFailed`，计划要求这个分支不要继续 cover threshold/grid search，而是 debug basis telemetry / Rational substrate interaction。本次测试 group telemetry 是否只是 hard gate 过硬；runner 已支持 method 名中包含 `GroupSNR` + `Soft` 的 soft group gate，因此没有修改 gate / promotion 规则。

执行：

```bash
conda run -n kan python experiments/run_v137_boundary_conditioned_poprisk_training.py \
  --out-dir results/v13_7_boundary_conditioned_poprisk_training/fallback_case_b_group_soft_repair_v137 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0,1,2 \
  --synthetic-train-size 96 \
  --synthetic-val-size 48 \
  --train-steps 200 \
  --batch-size 16 \
  --log-interval 50 \
  --loss-interfaces CE \
  --mlp-methods "" \
  --rational-methods RAT-AdamW,RAT-ParameterSNRSoft,RAT-GroupSNR,RAT-GroupSNRSoft,RAT-GroupSNREMA,RAT-GroupSNREMASoft,RAT-BasisSNR \
  --mlp-hidden 160
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

Method comparison：

```text
RAT-ParameterSNRSoft pass rows = 0
RAT-GroupSNR pass rows = 0
RAT-GroupSNRSoft pass rows = 0
RAT-GroupSNREMA pass rows = 0
RAT-GroupSNREMASoft pass rows = 0
RAT-BasisSNR pass rows = 0
```

最接近 rows：

```text
X5 seed=0 RAT-GroupSNREMA:
  source_vs_adamw = 0.4459215658243332
  CEp99_delta = -3.704286575317383
  NoiseSignalLeak_delta = -9.906291961669922e-05
  ReservoirRatio_delta = -5.656522274017334
  pass = 0

X5 seed=0 RAT-GroupSNREMASoft:
  source_vs_adamw = 0.4457548908265303
  CEp99_delta = -3.702035427093506
  NoiseSignalLeak_delta = 0.00022870302200317383
  ReservoirRatio_delta = -5.657006502151489
  pass = 0

X5 seed=0 RAT-GroupSNRSoft:
  source_vs_adamw = 0.4447917404111912
  CEp99_delta = -3.6889991760253906
  NoiseSignalLeak_delta = 0.002164483070373535
  ReservoirRatio_delta = -5.656068325042725
  pass = 0
```

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
4. 不允许把 MLP generic SNR positive 写成 KAN promotion。
5. v13.7 最终 route 仍以 plan-scale official 为准：R5-GenericSNRPositiveKANSpecificFailed。
```

## 10. 用户再次追问后继续：MLP generic real triage line

再次复核 v13.7 stop/go：

```text
If S2 pass but S3 fail:
  functional is generic but not KAN-specific; continue MLP optimizer line separately.
```

因此继续 MLP-only real triage。注意：这不是 KAN promotion，也不打开 KAN real short-run。

新增文件：

```text
experiments/run_v137_mlp_snr_real_triage.py
```

实现要点：

```text
1. 只跑 MLPBaseline。
2. 使用 train-stream generic loss-interface cotangent。
3. 为 MLPBaseline 写解析 per-example gradient，加速 real triage。
4. validation/test/LineC/CEp99/NLL/ECE 只做 audit/gate，不生成方向。
5. dataset name 只选择 data loader，optimizer rule 共享，不作为 controller branch。
6. 输出 route / training / summary / forbidden audit / required manifest。
```

第一次 smoke blocker：

```text
ValueError: relative out_dir cannot relative_to(ROOT)
```

修复：

```text
write_manifest() 对 path.resolve().relative_to(ROOT) 做 fallback；
main() 将相对 out_dir 转为 ROOT / out_dir。
```

第二次 smoke blocker：

```text
required_artifact_missing_count = 2
```

原因：

```text
route 在 manifest 最终刷新前写入。
```

修复：

```text
先写 route，再刷新 manifest，再把 final missing count 写回 route。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v137_mlp_snr_real_triage.py
```

结果：

```text
py_compile pass
```

smoke：

```bash
conda run -n kan python experiments/run_v137_mlp_snr_real_triage.py \
  --out-dir results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_smoke_v137 \
  --datasets MNIST \
  --seeds 0 \
  --methods MLP-AdamW,MLP-AdamW-SNREMA \
  --train-size 128 \
  --val-size 64 \
  --test-size 64 \
  --epochs 1 \
  --batch-size 32 \
  --log-interval 2 \
  --hidden-dim 64 \
  --no-download
```

smoke result：

```text
route = R5-MLPGenericRealTriageNotConfirmed
mlp_real_triage_rows = 2
mlp_real_triage_dataset_count = 1
mlp_real_triage_dataset_pass_count = 0
failure_rows = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
```

plan-scale MLP real triage：

```bash
conda run -n kan python experiments/run_v137_mlp_snr_real_triage.py \
  --out-dir results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_v137 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods MLP-AdamW,MLP-AdamW-SNRHard,MLP-AdamW-SNRSoft,MLP-AdamW-SNREMA,MLP-AdamW-SNRRoleNorm \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --epochs 3 \
  --batch-size 64 \
  --log-interval 16 \
  --hidden-dim 160 \
  --loss-interface CE \
  --no-download
```

结果：

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

```text
Fashion-MNIST: seed_pass_count = 0, dataset_pass = 0
KMNIST: seed_pass_count = 0, dataset_pass = 0
MNIST: seed_pass_count = 0, dataset_pass = 0
```

最接近 rows：

```text
MNIST seed=0 MLP-AdamW-SNRSoft:
  source_vs_adamw = 0.012638111652964246
  AUC_time_ratio_vs_adamw = 0.9761460832315737
  CEp99_delta = 0.5351014137268066
  ECE_delta = 0.010010592639446259
  real_triage_pass = 0

Fashion-MNIST seed=2 MLP-AdamW-SNRSoft:
  source_vs_adamw = 0.0008702662863998567
  AUC_time_ratio_vs_adamw = 0.9988081263623956
  CEp99_delta = 0.3193988800048828
  ECE_delta = -0.01061207801103592
  real_triage_pass = 0

KMNIST seed=2 MLP-AdamW-SNRSoft:
  source_vs_adamw = -0.026049696772194597
  AUC_time_ratio_vs_adamw = 1.0288090300035826
  CEp99_delta = 0.2673001289367676
  ECE_delta = 0.011949315667152405
  real_triage_pass = 0
```

判断：

```text
1. MLP generic SNR 在 synthetic plan-scale 上成立，但 real triage 未确认。
2. MNIST / Fashion-MNIST 的局部 AUC/source 改善被 CEp99 坏化挡住。
3. KMNIST 没有 AUC/source 改善。
4. MLP-only continuation 不能改变 v13.7 KAN route。
5. v13.7 最终仍不允许 KAN promotion / KAN real short-run。
```

## 11. 收尾一致性检查

runner 语法检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v137_mlp_snr_real_triage.py \
  experiments/run_v137_boundary_conditioned_poprisk_training.py
```

结果：

```text
py_compile pass
```

route artifact 复核：

```bash
jq . results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_v137/v137_mlp_real_triage_route.json
```

关键结果：

```text
route = R5-MLPGenericRealTriageNotConfirmed
minimum_success = S2-MLPGenericSNRPositive
mlp_real_triage_dataset_count = 3
mlp_real_triage_dataset_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
kan_promotion_allowed = 0
real_short_run_open_allowed_for_kan = 0
```

日志关键字复核：

```bash
rg -n "MLP generic real triage|R5-MLPGenericRealTriageNotConfirmed|experiments/run_v137_mlp_snr_real_triage.py|mlp_real_triage_v137|R5-GenericSNRPositiveKANSpecificFailed" \
  docs/DG-KAN_v13.7_BoundaryConditionedPopRiskTraining_执行日志.md \
  docs/DG-KAN_v13.7_BoundaryConditionedPopRiskTraining_实验结果复盘.md
```

结果：

```text
执行日志与实验结果复盘日志均包含 v13.7 KAN final route、MLP real triage route、新 runner 文件名与 official out_dir。
```

dataset family summary 复核：

```bash
sed -n '1,12p' results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_v137/v137_mlp_real_triage_family_summary.csv
```

结果：

```text
Fashion-MNIST seed_pass_count = 0, dataset_pass = 0
KMNIST seed_pass_count = 0, dataset_pass = 0
MNIST seed_pass_count = 0, dataset_pass = 0
```

## 12. 用户再次追问后继续：MLP Blend trust repair

用户再次要求确认 v13.7 是否达成目标，若未达成则继续。本次重新读取计划 stop/go rule：

```text
If S2 pass but S3 fail:
  functional is generic but not KAN-specific; continue MLP optimizer line separately.
```

当前 KAN plan-scale route 仍为：

```text
R5-GenericSNRPositiveKANSpecificFailed
```

上一轮 MLP real triage 的主要 blocker：

```text
MNIST / Fashion-MNIST 有局部 source/AUC 改善，但 CEp99 坏化；
KMNIST 没有稳定 source/AUC 改善。
```

本次修复思路：

```text
在 MLP-only runner 中新增 train-stream trust/blend 方法：
  MLP-AdamW-*-Blend25
  MLP-AdamW-*-Blend50
  MLP-AdamW-*-Blend75

direction = alpha * SNR_filtered_train_batch_gradient
          + (1-alpha) * AdamW_train_batch_gradient

合法性：
  只使用同一 train batch 的 generic loss-interface per-example gradient；
  不读取 validation/test/future/LineC/CEp99/NLL/ECE 生成方向；
  CEp99/ECE 仍只作为 audit/gate；
  该分支是 MLP-only，不允许 KAN promotion。
```

代码修改：

```text
experiments/run_v137_mlp_snr_real_triage.py
  1. 在 train_one() 中识别 Blend25 / Blend50 / Blend75。
  2. SNR gate 后按 alpha 混合 SNR-filtered grad 与 batch mean grad。
  3. training rows 新增 snr_blend_alpha。
  4. blend 后重新记录 cos_snr_adamw 与 removed_update_norm_fraction。
  5. removed_update_norm_fraction clamp 到 [0,1]，避免审计字段越界。
```

语法检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v137_mlp_snr_real_triage.py \
  experiments/run_v137_boundary_conditioned_poprisk_training.py
```

结果：

```text
py_compile pass
```

Blend smoke：

```bash
rm -rf results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_blend_smoke_v137 && \
conda run -n kan python experiments/run_v137_mlp_snr_real_triage.py \
  --out-dir results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_blend_smoke_v137 \
  --datasets MNIST \
  --seeds 0 \
  --methods MLP-AdamW,MLP-AdamW-SNRSoft-Blend25 \
  --train-size 128 \
  --val-size 64 \
  --test-size 64 \
  --epochs 1 \
  --batch-size 32 \
  --log-interval 2 \
  --hidden-dim 64 \
  --no-download
```

smoke 结果：

```text
route = R5-MLPGenericRealTriagePass
mlp_real_triage_rows = 2
mlp_real_triage_dataset_count = 1
mlp_real_triage_dataset_pass_count = 1
failure_rows = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
```

说明：

```text
smoke 只证明 Blend 方法可执行并能在 tiny MNIST slice 打开 gate；
不能写成 official MLP line success，更不能写成 KAN promotion。
```

Blend25 focused run：

```bash
rm -rf results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_blend25_v137 && \
conda run -n kan python experiments/run_v137_mlp_snr_real_triage.py \
  --out-dir results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_blend25_v137 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods MLP-AdamW,MLP-AdamW-SNRSoft-Blend25,MLP-AdamW-SNREMA-Blend25,MLP-AdamW-SNRRoleNorm-Blend25 \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --epochs 3 \
  --batch-size 64 \
  --log-interval 16 \
  --hidden-dim 160 \
  --loss-interface CE \
  --no-download
```

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

```text
Fashion-MNIST seed_pass_count = 2, dataset_pass = 1
KMNIST seed_pass_count = 1, dataset_pass = 0
MNIST seed_pass_count = 1, dataset_pass = 0
```

Blend50/75 focused run：

```bash
rm -rf results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_blend5075_v137 && \
conda run -n kan python experiments/run_v137_mlp_snr_real_triage.py \
  --out-dir results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_blend5075_v137 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods MLP-AdamW,MLP-AdamW-SNREMA-Blend50,MLP-AdamW-SNRRoleNorm-Blend50,MLP-AdamW-SNREMA-Blend75,MLP-AdamW-SNRRoleNorm-Blend75 \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --epochs 3 \
  --batch-size 64 \
  --log-interval 16 \
  --hidden-dim 160 \
  --loss-interface CE \
  --no-download
```

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

```text
Fashion-MNIST seed_pass_count = 1, dataset_pass = 0
KMNIST seed_pass_count = 2, dataset_pass = 1
MNIST seed_pass_count = 3, dataset_pass = 1
```

combined Blend official repair：

```bash
rm -rf results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_blend_combined_v137 && \
conda run -n kan python experiments/run_v137_mlp_snr_real_triage.py \
  --out-dir results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_blend_combined_v137 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods MLP-AdamW,MLP-AdamW-SNRSoft-Blend25,MLP-AdamW-SNREMA-Blend25,MLP-AdamW-SNRRoleNorm-Blend25,MLP-AdamW-SNREMA-Blend50,MLP-AdamW-SNRRoleNorm-Blend50,MLP-AdamW-SNREMA-Blend75,MLP-AdamW-SNRRoleNorm-Blend75 \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --epochs 3 \
  --batch-size 64 \
  --log-interval 16 \
  --hidden-dim 160 \
  --loss-interface CE \
  --no-download
```

结果：

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

```text
Fashion-MNIST seed_pass_count = 2, dataset_pass = 1, passing_seeds = 0;2
KMNIST seed_pass_count = 3, dataset_pass = 1, passing_seeds = 0;1;2
MNIST seed_pass_count = 2, dataset_pass = 1, passing_seeds = 0;2
```

代表性 pass rows：

```text
MNIST seed=0 MLP-AdamW-SNREMA-Blend50:
  source_vs_adamw = 0.05402895145299752
  AUC_time_ratio_vs_adamw = 0.8981513286015965
  CEp99_delta = -0.04118680953979492
  ECE_delta = -0.005311731249094009

Fashion-MNIST seed=2 MLP-AdamW-SNRRoleNorm-Blend25:
  source_vs_adamw = 0.01777078028907797
  AUC_time_ratio_vs_adamw = 0.9756575492560549
  CEp99_delta = -0.6446499824523926
  ECE_delta = 0.009411342442035675

KMNIST seed=1 MLP-AdamW-SNREMA-Blend50:
  source_vs_adamw = 0.028674350792256886
  AUC_time_ratio_vs_adamw = 0.9631843894548472
  CEp99_delta = -0.8646512031555176
  ECE_delta = -0.00255429744720459
```

判断：

```text
1. MLP generic optimizer line 在 real triage 上被 Blend repair 打开。
2. 该结果属于 MLP-only continuation，不是 KAN functional promotion。
3. KAN plan-scale route 仍是 R5-GenericSNRPositiveKANSpecificFailed。
4. v13.7 现在更明确地证明：generic PopRisk-SNR/blend optimizer 有价值，但 KAN basis/group telemetry 未能承接该价值。
5. 下一步不应继续 KAN cover/threshold 小修；应新开 KAN telemetry/substrate redesign，或把 MLP optimizer line 独立确认到更多 seeds/预算。
```

## 13. 用户再次追问后继续：MLP Blend 5/10-seed confirmation

用户再次要求“未达成则继续”。本次复核 v13.7 stop/go rule：

```text
If S5 pass:
  begin 5-seed / 10-seed confirmation.
```

严格边界：

```text
1. KAN 没有 S5，KAN route 仍是 R5-GenericSNRPositiveKANSpecificFailed。
2. MLP-only combined Blend 已经通过 3x3 real triage，因此只对 MLP optimizer line 做 5/10-seed confirmation。
3. 该确认不允许写成 KAN promotion。
```

代码审计修复：

```text
experiments/run_v137_mlp_snr_real_triage.py
  family_summary() 增加 seed_threshold_override；
  CLI 增加 --dataset-pass-seed-threshold；
  5-seed confirmation 使用 threshold=3；
  10-seed confirmation 使用 threshold=6。
```

原因：

```text
旧 family_summary 对任意 seed 数使用 min(2, len(seeds))，适合 3-seed triage；
如果直接用于 5/10-seed confirmation，会把确认阈值写得过松。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v137_mlp_snr_real_triage.py
```

结果：

```text
py_compile pass
```

5-seed confirmation：

```bash
rm -rf results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_blend_confirm5_v137 && \
conda run -n kan python experiments/run_v137_mlp_snr_real_triage.py \
  --out-dir results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_blend_confirm5_v137 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2,3,4 \
  --methods MLP-AdamW,MLP-AdamW-SNRSoft-Blend25,MLP-AdamW-SNREMA-Blend25,MLP-AdamW-SNRRoleNorm-Blend25,MLP-AdamW-SNREMA-Blend50,MLP-AdamW-SNRRoleNorm-Blend50,MLP-AdamW-SNREMA-Blend75,MLP-AdamW-SNRRoleNorm-Blend75 \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --epochs 3 \
  --batch-size 64 \
  --log-interval 16 \
  --hidden-dim 160 \
  --loss-interface CE \
  --dataset-pass-seed-threshold 3 \
  --no-download
```

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

5-seed family summary：

```text
Fashion-MNIST seed_pass_count = 3, dataset_pass = 1, passing_seeds = 0;2;3
KMNIST seed_pass_count = 5, dataset_pass = 1, passing_seeds = 0;1;2;3;4
MNIST seed_pass_count = 3, dataset_pass = 1, passing_seeds = 0;2;3
```

10-seed confirmation：

```bash
rm -rf results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_blend_confirm10_v137 && \
conda run -n kan python experiments/run_v137_mlp_snr_real_triage.py \
  --out-dir results/v13_7_boundary_conditioned_poprisk_training/mlp_real_triage_blend_confirm10_v137 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --methods MLP-AdamW,MLP-AdamW-SNRSoft-Blend25,MLP-AdamW-SNREMA-Blend25,MLP-AdamW-SNRRoleNorm-Blend25,MLP-AdamW-SNREMA-Blend50,MLP-AdamW-SNRRoleNorm-Blend50,MLP-AdamW-SNREMA-Blend75,MLP-AdamW-SNRRoleNorm-Blend75 \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --epochs 3 \
  --batch-size 64 \
  --log-interval 16 \
  --hidden-dim 160 \
  --loss-interface CE \
  --dataset-pass-seed-threshold 6 \
  --no-download
```

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

10-seed family summary：

```text
Fashion-MNIST seed_pass_count = 3, dataset_pass = 0, passing_seeds = 0;2;5
KMNIST seed_pass_count = 9, dataset_pass = 1, passing_seeds = 0;1;2;3;4;6;7;8;9
MNIST seed_pass_count = 5, dataset_pass = 0, passing_seeds = 0;2;3;8;9
```

10-seed best pass rows：

```text
Fashion-MNIST seed=2 MLP-AdamW-SNRRoleNorm-Blend25:
  source_vs_adamw = 0.01436830614479312
  AUC_time_ratio_vs_adamw = 0.9802654287435487
  CEp99_delta = -0.6446499824523926
  ECE_delta = 0.009411342442035675

KMNIST seed=7 MLP-AdamW-SNRRoleNorm-Blend50:
  source_vs_adamw = 0.055924443389259615
  AUC_time_ratio_vs_adamw = 0.9325213872409402
  CEp99_delta = -1.1899166107177734
  ECE_delta = -0.007198957726359367

MNIST seed=0 MLP-AdamW-SNREMA-Blend25:
  source_vs_adamw = 0.051681790817533435
  AUC_time_ratio_vs_adamw = 0.9024964325518074
  CEp99_delta = -0.14273452758789062
  ECE_delta = -0.00011614710092544556
```

10-seed failure pattern：

```text
1. KMNIST robust: 9/10 seeds pass。
2. MNIST borderline: 5/10 seeds pass，未达到 6/10 confirmation 阈值。
3. Fashion-MNIST weak: 3/10 seeds pass。
4. 许多 non-pass row 仍有 positive source/AUC，但 CEp99_delta 超过 0.05，不能写成 pass。
```

判断：

```text
1. MLP Blend optimizer line 达到 5-seed confirmation，但没有达到 10-seed confirmation。
2. 10-seed 结果不允许写成 fully confirmed generic optimizer success。
3. KAN route 不变：R5-GenericSNRPositiveKANSpecificFailed。
4. KAN promotion_allowed 仍为 0。
5. 当前 v13.7 内继续做 KAN cover/threshold 小修不符合计划；MLP line 若继续，需要新的 tail-safe optimizer confirmation 计划，而不是把 5-seed pass 编造成最终成功。
```
