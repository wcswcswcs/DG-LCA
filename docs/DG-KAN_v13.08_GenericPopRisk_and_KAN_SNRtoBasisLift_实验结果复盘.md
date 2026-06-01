# DG-KAN v13.8 GenericPopRisk and KAN SNRtoBasisLift 实验结果复盘

生成时间：2026-05-28（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功，不把 MLP-only positive、smoke、budgeted fallback 或 targeted repair 写成 KAN promotion。

## 1. 计划理解

v13.8 的核心不是继续调 SNR tau / cover threshold 小网格，而是分线回答：

```text
Line G: MLP PopRisk-SNR 是否能成为 10-seed 稳定 generic optimizer。
Line K0: parameter-level SNR signal 在 KAN basis/group lift 中是否丢失。
Line K1: KAN 是否能把 generic SNR signal lift 到 explicit basis cover 并达到 KAN-specific S3/S4。
Line D: Non-RAT 只做 substrate-health，不通过 substrate gate 不进入 functional proof。
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline。
2. 不使用 label-informed initialization。
3. 不使用 teacher / distillation / sampler / class weight / dataset-name branch。
4. functional direction 只使用 current train batch 与 generic loss interface。
5. validation/test/future/query batch、CEp99/NLL/ECE/LineC 只能 audit/gate，不能生成 direction。
6. MLP generic optimizer 不能写成 KAN-specific promotion。
```

## 2. 本轮代码修改

新增：

```text
experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py
```

修改：

```text
experiments/run_v137_boundary_conditioned_poprisk_training.py
```

核心内容：

```text
1. v13.8 runner 输出 Line G / K0 / K1 / D / audit / finalizer artifacts。
2. K1 writeback trace 独立为 v138_rat_snr_lift_writeback_trace.csv，不再混入 failure table。
3. 显式实现 K1-K7 method surface：
   K1 ParamSNR-only
   K2 ParamSNR + MinCoverGuard
   K3 RoleWiseSNRLift
   K4 DynamicSNRClusterLift
   K5 LowRankSNRCorrector
   K6 ParamSNR then BasisConsolidation
   K7 ParamSNR BlendAdamW then CoverPhase
4. v13.7 shared helper 增加 train-stream Blend、rank-1 low-rank corrector、phase cover policies。
5. route 写入 compute_budgeted_run 与实际 real/synthetic budget，避免把预算版伪装成 200-step full plan。
```

合法性说明：

```text
1. G6/G7/G8/G9 只使用 current train-stream loss/logits/per-example gradient statistics。
2. K1-K7 方向只来自 current train batch per-example gradients 与 generic loss interface。
3. K5 low-rank corrector 使用当前 batch gradient covariance，不使用 validation/test/future。
4. K7 blend 使用同 batch AdamW gradient，不使用 audit target。
5. CEp99/NLL/ECE/LineC 只进入 pass gate 与 failure explanation。
```

## 3. 语法与 smoke

语法检查：

```text
py_compile pass
```

smoke：

```text
smoke_v138 route = R1-GenericSNROptimizerNoGo
smoke_v138 required_artifact_missing_count = 0
smoke_v138_k1fix route = R1-GenericSNROptimizerNoGo
smoke_v138_k1fix required_artifact_missing_count = 0
```

解释：smoke 只证明 runner / artifact / K1-K7 surface 可执行，不能 promotion。

## 4. prepatch full official blocker

第一次 full official 尝试运行约 35 分钟后中断，没有写出 route artifact。

中断原因：

```text
1. 实现复核发现 K1 仍是 v13.7 method surface，未显式覆盖计划 K1-K7。
2. runner 末尾统一写 artifact，长跑期间不可审计。
```

处理：

```text
1. 中断结果不计入实验结论。
2. 修复 K1-K7 surface 与 writeback/failure table 边界。
3. 重新运行 smoke、official budgeted、targeted K1 repair、MLP full-size repair。
```

## 5. official_v138 结果

执行规模：

```text
MLP datasets = MNIST,Fashion-MNIST,KMNIST
MLP seeds = 0..9
MLP methods = AdamW + G6/G7/G8/G9 trust repairs
real_train_size = 512
real_val_size = 256
real_test_size = 256
real_epochs = 2
KAN tasks = X1..X7
KAN seeds = 0,1,2
loss = CE,Brier
KAN methods = RAT-AdamW + K1..K7
KAN train_steps = 24
batch_size = 16
compute_budgeted_run = 1
```

最终 route：

```text
route = R1-GenericSNROptimizerNoGo
minimum_success = S1-ImplementationReadback
official_success_reached = 0
promotion_allowed = 0
kan_real_short_run_open_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
failure_rows = 0
```

关键计数：

```text
mlp_generic_10seed_dataset_pass_count = 0 / 3
mlp_generic_10seed_confirmed = 0
snr_transfer_gate_pass = 1
snr_transfer_median_retention_group = 0.8716974258422852
snr_transfer_median_cos_group_vs_param = 0.7423757314682007
kan_s3_task_pass_count = 3 / 7
kan_s4_task_pass_count = 1 / 7
nonrat_substrate_health_pass_count = 0
```

MLP 10-seed：

| dataset | seed_pass_count | threshold | dataset_pass | passing_seeds |
|---|---:|---:|---:|---|
| Fashion-MNIST | 3 | 6 | 0 | 0;5;6 |
| KMNIST | 3 | 6 | 0 | 3;5;8 |
| MNIST | 2 | 6 | 0 | 3;8 |

判断：

```text
G6/G7/G8/G9 budgeted official 没有确认 generic optimizer。
因此 official route 合法为 R1。
```

## 6. K0 SNR transfer audit

official_v138：

```text
transfer_rows = 42
signal_retention_group min = 3.4100019317649055e-07
signal_retention_group mean = 0.692766571486746
signal_retention_group max = 1.0996593388122005
cos_group_vs_param min = 0.0
cos_group_vs_param mean = 0.6189879111590839
cos_group_vs_param max = 0.9973965883255005
median retention = 0.8716974258422852
median cosine = 0.7423757314682007
transfer_gate_pass = 1
```

role signal mass mean：

| role | mean_signal_mass | rows |
|---|---:|---:|
| bias | 0.5254325547257437 | 42 |
| hidden_weight | 0.5767148278026946 | 42 |
| rational_denominator | 0.14310711095837317 | 42 |
| rational_numerator | 0.3836324987275917 | 42 |

判断：

```text
K0 不支持 R3-SNRSignalLostInBasisLift。
median retention/cosine 过 gate，说明当前 blocker 不是 group/basis lift 完全吞掉 parameter-SNR signal。
```

## 7. K1 SNR-to-Basis Lift

official_v138 KAN task-level：

```text
kan_s3_task_pass_count = 3 / 7
kan_s4_task_pass_count = 1 / 7
```

有 S3 row 的 task：

```text
X1: 3 rows
X2: 1 row
X3: 2 rows, 但集中在同一 seed/loss，未形成 task pass
X4: 4 rows
X5: 1 row
X6: 0 rows
X7: 3 rows
```

按 v13.8 task-family gate，真正 task pass 为：

```text
X1, X4, X7
```

最强但不能 task-level promotion 的 blocker rows：

| task | seed | loss | method | source_vs_best | AUC_time_ratio | CEp99_delta | CouplingR2_delta | NoiseSignalLeak_delta | Reservoir_delta | pass_s3 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| X2 | 2 | CE | K6 | 0.31786029465464716 | 0.8220510072160016 | -0.055877685546875 | -1.0975875854492188 | -0.02442115545272827 | 0.32860231399536133 | 0 |
| X4 | 0 | Brier | K2 | 0.2612746689543842 | 0.7763543058722017 | 1.2567858695983887 | 1.4931411743164062 | -0.049606382846832275 | 0.01589202880859375 | 0 |
| X1 | 0 | CE | K4 | 0.25389892151742477 | 0.8519370408924101 | -0.27141904830932617 | 1.4569482803344727 | 0.020583629608154297 | -0.11744141578674316 | 0 |
| X4 | 0 | Brier | K4 | 0.24443381885415927 | 0.7907697240428188 | -0.16669702529907227 | 7.222243309020996 | -0.03605175018310547 | -0.7640347480773926 | 1 |
| X7 | 1 | Brier | K2 | 0.20647748905923957 | 0.7792037416796643 | -2.395874500274658 | 9.99658203125 | -0.06747788190841675 | -1.1432387828826904 | 1 |

判断：

```text
K1-K7 有局部强 source，但不能过 >=5/7 task-family gate。
主要 blocker 不是 source 完全没有，而是 LineC / reservoir / tail 条件不稳定，且 X2/X3/X5/X6 覆盖不够。
```

## 8. K1 targeted repair

执行：

```text
tasks = X2,X3,X5,X6
methods = RAT-AdamW,K2,K4,K5,K6,K7
train_steps = 80
batch_size = 16
```

结果：

```text
route = R1-GenericSNROptimizerNoGo
kan_s3_task_pass_count = 1 / 4
kan_s4_task_pass_count = 0 / 4
required_artifact_missing_count = 0
```

唯一形成 task pass：

```text
X2
```

判断：

```text
延长 K2/K4/K5/K6/K7 没有把 KAN coverage 从 official 的 3/7 推过 5/7；
targeted repair 反而只在 X2 上形成 task pass。
不能写成 KAN-specific S3。
```

## 9. MLP full-size 10-seed repair

由于 official_v138 是 compute-budgeted run，本轮补跑 MLP full-size repair：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
methods = AdamW + TrainLossQuantileTrust + ActiveFractionSchedule + PerExampleGradientClip
real_train_size = 1024
real_val_size = 512
real_test_size = 512
real_epochs = 3
KAN side = X1 / seed0 / RAT-AdamW / 1 step placeholder
```

结果：

```text
route = R1-GenericSNROptimizerNoGo
mlp_generic_10seed_dataset_pass_count = 0 / 3
required_artifact_missing_count = 0
```

MLP full-size 10-seed：

| dataset | seed_pass_count | threshold | dataset_pass | passing_seeds |
|---|---:|---:|---:|---|
| Fashion-MNIST | 0 | 6 | 0 | |
| KMNIST | 2 | 6 | 0 | 0;2 |
| MNIST | 2 | 6 | 0 | 0;1 |

判断：

```text
full-size G6/G8/G9 repair 仍没有确认 MLP generic optimizer。
这闭合了 v13.8 Line G 的主要 blocker；不能再把 v13.7 5-seed positive 扩写成 10-seed generic optimizer success。
```

## 10. Non-RAT substrate health

official_v138：

```text
nonrat rows = 8
substrate_health_pass_count = 0
functional_proof_allowed = 0 for all rows
```

代表性 rows：

| family | candidate | workspace_raw_ratio | workspace_incremental_ratio | step_ratio | LineC_pass_rate | substrate_health_pass |
|---|---|---:|---:|---:|---:|---:|
| D-CHE | CHE-SNR5-degreeLateEnable | 1.4106476848903848 | 13.232558139534884 | 1.878934745131025 | 0.0 | 0 |
| D-FOU | FOU-SNR4-lowFreqIdentityResidual | 1.457700354771218 | 17.039867109634553 | 2.321841552067897 | 0.0 | 0 |
| D-RBF | RBF-SNR4-compactOccupancyRepair | 1.2043800600382062 | 3.8596345514950166 | 2.182799341342765 | 0.0 | 0 |
| D-WAV | WAV-SNR4-hatScaleBalanced | 1.1813654143545893 | 4.25 | 1.9689910476591348 | 0.0 | 0 |

判断：

```text
Non-RAT 仍是 WorkspaceOnly_NotFunctionalSubstrate；
不能进入 functional proof，也不能写 family failure/promotion。
```

## 11. Required artifacts

official_v138：

```text
manifest_rows = 31
missing_required_rows = 0
```

targeted_k1_repair_v138：

```text
manifest_rows = 31
missing_required_rows = 0
```

mlp_full10seed_repair_v138：

```text
required_artifact_missing_count = 0
```

主要 artifact：

```text
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_route_decision.json
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_mlp_generic_optimizer_10seed.csv
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_snr_transfer_audit.csv
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_rat_snr_lift_summary.csv
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_nonrat_substrate_health.csv
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_no_go_boundary.md
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/targeted_k1_repair_v138/v138_route_decision.json
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/mlp_full10seed_repair_v138/v138_route_decision.json
```

## 12. 最终科学结论

v13.8 没有达成目标；最终合法结论：

```text
route = R1-GenericSNROptimizerNoGo
minimum_success = S1-ImplementationReadback
promotion_allowed = 0
official_success_reached = 0
kan_real_short_run_open_allowed = 0
```

已闭合事实：

```text
1. MLP generic optimizer 10-seed 未确认：
   budgeted official = 0/3 dataset pass；
   full-size MLP repair = 0/3 dataset pass。
2. K0 signal transfer gate 通过：
   median retention = 0.8716974258422852；
   median cosine = 0.7423757314682007。
3. K1-K7 official only reached KAN S3 3/7, S4 1/7。
4. K1 targeted longer repair only reached 1/4 task pass。
5. Non-RAT substrate-health pass = 0。
6. required artifacts missing = 0，forbidden/provenance violation = 0。
7. MLP-only rows 没有被写成 KAN promotion。
```

no-go boundary：

```text
1. v13.8 未能把 v13.7 的 MLP 5-seed positive 扩展成 10-seed generic optimizer claim。
2. K0 不支持“basis/group lift 完全丢失 signal”的解释；signal transfer median gate 是通过的。
3. KAN blocker 变成：有局部 source，但 LineC/reservoir/tail 稳定性与 task-family coverage 不够。
4. K1-K7 与 targeted longer repair 都没有达到 >=5/7 synthetic task-family S3。
5. Non-RAT 仍不能进入 functional proof。
```

最终判断：

```text
v13.8 未达成 S3/S4/S5；
不允许 promotion；
不允许 KAN real short-run；
当前应停止 MLP SNR 小修，并把 KAN 方向转向更强的 basis/substrate architecture 或新的 signal-to-cover mechanism。
```

## 13. 用户再次追问后的 stop-contract 复核与 Non-RAT exact vertical slice

用户再次要求确认 v13.8 是否达成目标，若未达成则继续。本次重新读取 v13.8 route 与计划 stop/go 条件后，结论仍是：

```text
v13.8 未达成 S3/S4/S5；
promotion_allowed = 0；
kan_real_short_run_open_allowed = 0；
official route = R1-GenericSNROptimizerNoGo。
```

复核发现一个计划覆盖边界：

```text
v13.8 第 8.3 节要求每轮至少执行：
1. Rational SNR-to-basis lift main run；
2. Two Non-RAT families substrate-health repair scout；
3. One Non-RAT family exact lifetime / task-health vertical slice。
```

前两项已由 official_v138 覆盖；第三项未单独记录。因此补跑：

```text
D-FOU20-LowFreqIdentityResidualHealthSubstrate
dataset = MNIST
seed = 0
train_size = 64
val_size = 32
workspace_warmup_steps = 2
workspace_profile_steps = 5
hardening_epochs = 1
```

结果：

```text
workspace_rows = 1
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_rows = 1
hardening_executed_rows = 0
linec_executed = 0
skip_reason = workspace_gate_fail
raw_memory_ratio_vs_mlp = 1.1053239288134742
incremental_memory_ratio_vs_mlp = 2.9649122807017543
step_ratio_vs_mlp = 1.5105504843235176
```

解释：

```text
1. D-FOU20 raw memory 与 step ratio 接近 gate，但 incremental memory ratio = 2.9649 > v13.8 substrate-health gate 的 2.00。
2. workspace gate fail 后，hardening 与 LineC 按规则 skip。
3. 该 row 不能写成 task-health pass，也不能进入 functional proof。
4. Non-RAT substrate_health_pass_count 仍为 0。
```

最终判断不变：

```text
v13.8 没有达成目标；
Line G: MLP 10-seed generic optimizer 未确认，full-size repair 仍 0/3 dataset pass。
Line K0: SNR transfer gate 通过，因此不是 R3。
Line K1: K1-K7 + targeted longer repair 未达到 >=5/7 KAN S3。
Line D: Non-RAT scout 与 exact vertical slice 均未打开 substrate-health gate。
不允许 promotion；
不允许 KAN real short-run；
当前合法 stop 是停止 MLP SNR 小修，并转向新的 KAN basis/substrate architecture 或 signal-to-cover mechanism。
```

## 14. 用户再次追问后的 route metadata 审计修复

用户再次要求确认 v13.8 是否达成目标，若未达成则继续。本次重新读取计划 stop/go 条件与 official route 后，结论仍不变：

```text
route = R1-GenericSNROptimizerNoGo
official_success_reached = 0
promotion_allowed = 0
kan_real_short_run_open_allowed = 0
required_artifact_missing_count = 0
```

本次发现一个审计字段缺口：

```text
official_v138/v138_route_decision.json 没有显式写出 final_stop_allowed。
```

修复内容：

```text
1. 修改 experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py。
2. build_route() 对 R1/R2/R3 且 artifact/provenance 无缺失的 no-go route 写入 final_stop_allowed。
3. 刷新当前 official_v138 route metadata，补入 final_stop_allowed = 1。
4. 重新生成 official_v138 code review packet，并刷新 route 中的 code_review_packet_sha256。
5. 未修改任何 gate、未新增或改写实验 CSV 指标、未把 diagnostic 写成 promotion。
```

校验结果：

```text
py_compile pass
route JSON parse pass
route = R1-GenericSNROptimizerNoGo
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
kan_real_short_run_open_allowed = 0
code_review_packet_sha256 = a27ad37b9ef8e10dcb657cd3fe0ca3e5844d38979959843905cf719e31d2e5e1
```

最终判断仍是：

```text
v13.8 没有达成 S3/S4/S5；
Line G 已按计划执行 trust/blend/full-size repair，MLP generic 10-seed 仍未确认；
Line K1 已执行 K1-K7 与 targeted longer repair，KAN S3 仍不足 5/7；
Line D 已执行 Non-RAT scout 与 exact vertical slice，substrate-health 仍为 0；
不允许 promotion；
不允许 KAN real short-run；
允许 final stop。
```

## 15. 用户再次追问后的 KAN 200-step 与 K4F freeze-cluster fallback

用户再次要求确认 v13.8 是否达成目标，若未达成则继续。本次复核发现：official_v138 明确标记为 `compute_budgeted_run = 1`，KAN 部分只有 24 steps；虽然 MLP full-size repair 已补跑，但 KAN basis line 还没有按计划步数做完整复核。

因此继续补跑 KAN full-step repair。该 run 的 MLP 侧只放最小 placeholder，因为 v13.8 第 14.1 节已经要求 MLP 10-seed 失败后停止扩展 MLP SNR 小修；本 run 只用于检查 K1-K7 在 200-step KAN synthetic line 上是否能打开 S3。

执行规模：

```text
out_dir = results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/kan_full200_repair_v138
synthetic_tasks = X1..X7
synthetic_seeds = 0,1,2
losses = CE,Brier
k_methods = RAT-AdamW,K1,K2,K3,K4,K5,K6,K7
train_steps = 200
batch_size = 64
MLP side = MNIST seed0 AdamW placeholder only
```

结果：

```text
route = R1-GenericSNROptimizerNoGo
compute_budgeted_run = 1
synthetic_train_steps = 200
snr_transfer_gate_pass = 1
snr_transfer_median_retention_group = 0.9576625823974609
snr_transfer_median_cos_group_vs_param = 0.8467055559158325
kan_s3_task_pass_count = 4 / 7
kan_s4_task_pass_count = 2 / 7
nonrat_substrate_health_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
kan_real_short_run_open_allowed = 0
```

200-step task-family 结果：

| task | S3 task pass | S4 task pass | s3 rows | s3 seeds | s3 losses | best method | best source_vs_best | best row pass |
|---|---:|---:|---:|---|---|---|---:|---:|
| X1 | 0 | 0 | 1 | 2 | Brier | K4 | 0.452417848159653 | 0 |
| X2 | 1 | 1 | 4 | 0,2 | Brier,CE | K4 | 0.19507057605075295 | 0 |
| X3 | 1 | 0 | 4 | 1,2 | Brier | K4 | 0.233382930391569 | 0 |
| X4 | 0 | 0 | 2 | 0 | Brier | K4 | 0.40966643083990806 | 1 |
| X5 | 1 | 1 | 4 | 0,1,2 | Brier,CE | K1 | 0.07280655955682447 | 0 |
| X6 | 1 | 0 | 3 | 1,2 | CE | K6 | 0.06266661359136289 | 1 |
| X7 | 0 | 0 | 0 | - | - | K4 | 0.3064354103369661 | 0 |

解释：

```text
1. 200-step KAN line 比 24-step official 更强，从 3/7 提升到 4/7。
2. 但仍低于 v13.8 S3 gate 的 >=5/7。
3. X1/X7 的高 source rows 主要卡在 NoiseSignalLeak / LineC gate。
4. X4 有单 seed Brier pass，但没有达到 task-family pass 的 >=2 seeds 或 >=2 loss interfaces。
5. 因 kan_s3_task_pass_count = 4/7，不允许 KAN real short-run。
```

随后检查计划 K1-C：

```text
Case K1-C: dynamic cluster unstable across seeds
  Freeze clusters after warmup; compare cluster drift.
```

本轮新增 K4F fallback：

```text
1. experiments/run_v137_boundary_conditioned_poprisk_training.py
   - SNRState 增加 frozen_gate / frozen_group_scores。
   - snr_gate() 对包含 FreezeCluster 的方法，在 plasticity-open 后冻结 train-stream group gate。
   - 不使用 CEp99/NLL/ECE/LineC 生成方向。
2. experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py
   - 新增 K4F-RAT-DynamicSNRClusterFreeze alias。
```

K4F targeted run：

```text
out_dir = results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/k4_freeze_targeted_v138
tasks = X1,X4,X7
seeds = 0,1,2
losses = CE,Brier
methods = RAT-AdamW,K4,K4F
train_steps = 200
```

结果：

```text
route = R1-GenericSNROptimizerNoGo
kan_s3_task_pass_count = 0 / 3
kan_s4_task_pass_count = 0 / 3
snr_transfer_gate_pass = 1
required_artifact_missing_count = 0
promotion_allowed = 0
```

K4F 对比：

| task | method | task pass | s3 rows | s3 seeds | s3 losses | best source_vs_best | best linec | best pass |
|---|---|---:|---:|---|---|---:|---:|---:|
| X1 | K4 | 0 | 1 | 2 | Brier | 0.4532723129555286 | 0 | 0 |
| X1 | K4F | 0 | 1 | 2 | Brier | 0.45247159472554344 | 0 | 0 |
| X4 | K4 | 0 | 1 | 0 | Brier | 0.4091217412041853 | 1 | 1 |
| X4 | K4F | 0 | 1 | 0 | Brier | 0.4098742882429317 | 1 | 1 |
| X7 | K4 | 0 | 0 | - | - | 0.30647107199183976 | 0 | 0 |
| X7 | K4F | 0 | 0 | - | - | 0.3065164241870677 | 0 | 0 |

解释：

```text
1. Freeze-cluster 没有把 X1/X4/X7 任意一个 family 推到 task pass。
2. X1/X7 仍主要是 NoiseSignalLeak / LineC blocker。
3. X4 仍只有 seed0 Brier 单点 pass，不满足 family gate。
4. K1-C fallback 已尝试但失败。
```

最终判断：

```text
v13.8 仍未达成 S3/S4/S5；
MLP full-size repair 未确认 generic optimizer；
KAN 200-step K1-K7 只达到 4/7；
K4F freeze-cluster targeted fallback 0/3；
Non-RAT scout 与 exact vertical slice 均未打开 substrate-health gate；
不允许 promotion；
不允许 KAN real short-run；
允许 final stop。
```

## 16. 用户再次追问后的 Non-RAT RBF/WAV exact substrate repair

用户再次要求“未达成则继续”。本次按 v13.8 Line D 的推荐方向继续检查 Non-RAT substrate-health blocker：从 200-step run 的 scout 结果中选择最接近 workspace gate 的 RBF/WAV task-health candidates，并运行 v1235 exact substrate workspace audit。

执行内容：

```text
D-RBF17-CompactCapacityK4HealthSubstrate
D-WAV16-SupportStableHatHealthSubstrate
dataset = MNIST
seed = 0
train_size = 64
val_size = 32
workspace_warmup_steps = 2
workspace_profile_steps = 5
hardening_epochs = 1
```

结果：

```text
workspace_rows = 2
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_rows = 2
hardening_executed_rows = 0
family_near_pass_rows = 0
promotion_allowed = 0
```

逐 candidate：

| candidate | raw memory ratio | incremental memory ratio | step ratio | workspace gate | exact kernel | materializes basis/derivative/readout grad | hardening | LineC | skip reason |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| D-RBF17-CompactCapacityK4HealthSubstrate | 1.2209141795727805 | 2.5254803675856308 | 1.2397470757373803 | 0 | 0 | 1/1/1 | 0 | 0 | workspace_gate_fail |
| D-WAV16-SupportStableHatHealthSubstrate | 1.1849946032781948 | 2.5254803675856308 | 1.2997384483553653 | 0 | 0 | 1/1/1 | 0 | 0 | workspace_gate_fail |

解释：

```text
1. RBF17/WAV16 比 CHE/FOU 更接近 raw memory gate，但 incremental memory ratio 仍为 2.52548，高于 S1 workspace gate 的 2.00。
2. exact_kernel_implemented = 0，且 basis / derivative / readout_grad tensor 仍被 materialize；这说明当前候选还不是计划希望的真正 exact/non-materialized repair。
3. hardening 与 LineC 被 workspace gate fail 阻断，不能写成 task-health pass。
4. Non-RAT substrate-health pass 仍为 0。
```

代码修改：

```text
本轮没有修改代码；只执行已有 v1235 candidate registry 中的 RBF17/WAV16 exact substrate audit。
```

最终判断仍是：

```text
v13.8 没有达成目标；
MLP generic optimizer full-size repair 未通过；
KAN 200-step K1-K7 最高只到 S3 4/7、S4 2/7；
K4F freeze-cluster fallback 未打开 X1/X4/X7；
Non-RAT FOU20、RBF17、WAV16 exact substrate repair 均未打开 workspace gate；
不允许 promotion；
不允许 KAN real short-run；
允许 final stop。
```

## 17. 用户再次追问后的 MLP G7 full-size 补跑

用户再次要求“未达成则继续”。本次重新对照 v13.8 Line G 失败修复链：

```text
Case G-A: source_vs_adamw positive but CEp99 fail
  Try TrainLossQuantileTrust -> LogitNormTrust -> PerExampleGradientClip。

Case G-B: AUC_time fail but source positive
  Try active fraction schedule and smaller blend alpha。

Case G-D: 10-seed fails after G6-G9
  Record GenericPopRiskSNRRealNoGo_v13_8; stop MLP optimizer expansion。
```

复核发现此前 full-size 10-seed repair 已覆盖：

```text
G6 TrainLossQuantileTrust
G8 ActiveFractionSchedule
G9 PerExampleGradientClip
```

但 G7 LogitNormTrust 只在 official budgeted run 覆盖，未在 full-size 10-seed repair 中单独覆盖。因此本次补跑 full-size G7：

```text
out_dir = results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/mlp_g7_logitnorm_full10seed_v138
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
methods = MLP-AdamW, MLP-AdamW-SNRRoleNorm-Blend50-LogitNormTrust
real_train_size = 1024
real_val_size = 512
real_test_size = 512
real_epochs = 3
```

结果：

```text
route = R1-GenericSNROptimizerNoGo
mlp_generic_10seed_confirmed = 0
mlp_generic_10seed_dataset_pass_count = 0 / 3
required_artifact_missing_count = 0
promotion_allowed = 0
```

10-seed gate：

| dataset | seed pass count | threshold | dataset pass | passing seeds |
|---|---:|---:|---:|---|
| Fashion-MNIST | 0 | 6 | 0 | - |
| KMNIST | 0 | 6 | 0 | - |
| MNIST | 0 | 6 | 0 | - |

最接近 rows：

| dataset | seed | source_vs_adamw | AUC_time_ratio | CEp99_delta | NLL_delta | ECE_delta | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | 0.05841256490086766 | 0.9135080659086646 | -0.31749391555786133 | 0.05088728666305542 | 0.008245181292295456 | 0 |
| Fashion-MNIST | 7 | -0.03371183383521881 | 1.0481259675329708 | 0.3629751205444336 | 0.013575732707977295 | 0.014368921518325806 | 0 |
| KMNIST | 9 | -0.03697864633723191 | 1.046387831223096 | -2.0538525581359863 | -0.056354641914367676 | -0.008521147072315216 | 0 |

解释：

```text
1. G7 LogitNormTrust 在 full-size 10-seed 下没有打开任何 dataset。
2. MNIST seed0 有 source/AUC 改善，但 NLL_delta = 0.050887 > 0.02，因此不能 pass。
3. Fashion-MNIST 和 KMNIST 最接近 rows 仍为 source_vs_adamw < 0 且 AUC_time_ratio > 1。
4. 至此 v13.8 计划内 G6/G7/G8/G9 full-size repair 均已执行或补齐，仍没有 MLP generic 10-seed confirmation。
5. 这只能写成 GenericPopRiskSNRRealNoGo，不能写成 generic optimizer success，更不能写成 KAN promotion。
```

最终判断仍是：

```text
v13.8 没有达成目标；
MLP full-size 10-seed G6/G7/G8/G9 全部未确认 generic optimizer；
KAN 200-step K1-K7 最高只到 S3 4/7、S4 2/7；
K4F freeze-cluster fallback 未打开 X1/X4/X7；
Non-RAT FOU20、RBF17、WAV16 exact substrate repair 均未打开 workspace gate；
不允许 promotion；
不允许 KAN real short-run；
允许 final stop。
```
