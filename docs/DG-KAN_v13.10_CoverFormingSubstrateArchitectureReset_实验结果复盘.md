# DG-KAN v13.10 CoverFormingSubstrateArchitectureReset 实验结果复盘

生成时间：2026-05-28（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 smoke / compute-budgeted scout / local S3 row / Non-RAT substrate audit 写成 promotion。

## 1. 计划理解

v13.10 的目标不是继续做 K8-K13 token 搜索，而是执行 cover-forming substrate architecture reset：

```text
population-risk SNR signal 已经能保留；
本轮要验证新的 Rational cover-forming substrate architecture 是否能形成 stable basis cover，
再进入 KAN-specific S3/S4/S5。
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 teacher / distillation / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. functional direction 只能使用 train-stream gradient / parameter / group / basis telemetry。
5. LineC / CEp99 / NLL / ECE 只能作为 audit / gate，不能生成方向。
6. MLP row 只能是 generic optimizer control，不能写成 KAN promotion。
7. Non-RAT 必须先过 exact no-materialize substrate-health gate，不能以 scout / proxy alias 进入 functional proof。
8. A-RCF 必须是真 substrate/base architecture reset，不能只是 K-token alias。
```

计划 stop-contract 要求：

```text
不能在 R2 后直接停止；必须覆盖 A-RCF1..A-RCF6 scout、top-2 A-RCF hardening、K14-K18 training、Fourier/Cheb Non-RAT vertical slice、MLP monitor、no-go boundary 与 next queue。
```

## 2. 本轮代码修改

新增文件：

```text
experiments/run_v1310_cover_forming_substrate_architecture_reset.py
```

主要修改：

```text
1. 新增 v13.10 runner 与 artifact surface。
2. 实现 A-RCF1..A-RCF6 substrate architecture reset：
   A-RCF1-MultiBandRationalCover
   A-RCF2-SNRClusterCoverWarmup
   A-RCF3-PersistentCoverMemory
   A-RCF4-SignalReservoirSplitGroups
   A-RCF5-ReadoutBasisDecoupledCover
   A-RCF6-OvercompleteSparseCoverBank
3. A-RCF 使用真实参数初始化 / group reset / overcomplete h40 primitive，不只是 K-token alias。
4. 新增 K14-K18 方法语义，且只在 A-RCF substrate 上运行。
5. 输出 architecture diff manifest、substrate architecture readback、rational cover substrate、cover telemetry、LineC audit、Non-RAT vertical slice、MLP monitor、required manifest、no-go boundary、next queue 与 figures。
6. Non-RAT vertical slice 对没有 exact no-materialize prior 的候选 fail-closed，不写成 functional substrate。
7. route decision 明确区分 cover substrate pass、S3/S4 family pass、Non-RAT vertical pass、MLP control。
```

合法性说明：

```text
1. K14-K18 direction 不使用 validation/test/future/query batch。
2. K14-K18 direction 不使用 LineC / CEp99 / NLL / ECE 生成方向。
3. architecture_diff_manifest 记录 is_new_substrate_architecture=1。
4. route 记录 is_k_token_only_extension=0。
5. MLP 只作为 control monitor。
6. 不改变 promotion gate。
```

## 3. 初始语法检查

执行：

```text
conda run -n kan python -m py_compile experiments/run_v1310_cover_forming_substrate_architecture_reset.py
```

结果：

```text
py_compile pass
```

## 4. Smoke

执行规模：

```text
architectures = A-RCF1,A-RCF6
synthetic_tasks = X1
synthetic_seeds = 0
loss = CE
k_methods = K0,K14,K18
train_steps = 4
batch_size = 16
```

第一次 smoke 发现 artifact 顺序 blocker：

```text
v1310_route_decision.json 在 manifest 检查后才写入，
导致 required_artifact_missing_count = 1。
```

修复：

```text
experiments/run_v1310_cover_forming_substrate_architecture_reset.py
  在 required manifest 检查前先写 route_decision；
  manifest 回填 missing count 后再重写 route_decision。
```

重跑 smoke 结果：

```text
route = R1-NoCoverSubstrate
minimum_success = S0-ArchitectureScoutExecuted
promotion_allowed = 0
official_success_reached = 0
required_artifact_missing_count = 0
cover_substrate_pass_count = 0
snr_transfer_median_retention_group = 1.0
snr_transfer_median_cos_group_vs_param = 0.7617124319076538
cover_purity_median = 0.0660642609000206
cover_churn_mean = 0.0
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
```

解释：

```text
smoke 只证明 runner、artifact、A-RCF/K14/K18 surface 可执行；
不能 promotion。
```

## 5. Official all-A-RCF scout

执行规模：

```text
architectures = A-RCF1..A-RCF6
synthetic_tasks = X1..X7
synthetic_seeds = 0
loss_interfaces = CE,Brier
k_methods = K0,K14,K15,K16,K17,K18
train_steps = 60
batch_size = 32
datasets = MNIST,Fashion-MNIST,KMNIST
mlp_seeds = 0,1
compute_budgeted_run = 1
```

最终 route：

```text
route = R1-NoCoverSubstrate
minimum_success = S0-ArchitectureScoutExecuted
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
is_new_substrate_architecture = 1
is_k_token_only_extension = 0
cover_substrate_pass_count = 0
snr_transfer_median_retention_group = 1.0
snr_transfer_median_cos_group_vs_param = 0.786395251750946
cover_purity_median = 0.03301357105374336
cover_churn_mean = 0.08117913581481596
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 2
kan_s4_task_pass_count = 0
nonrat_vertical_pass_count = 0
mlp_generic_dataset_pass_count = 0 / 3
```

A-RCF substrate scout：

| architecture | mapped candidate | gate | retention | cos | cover purity | cover churn | mean source delta | AUC ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A-RCF1 | D-RAT25 | 0 | 1.0 | 0.8632747530937195 | 0.03470376133918762 | 0.09523809239977882 | -0.15731786602123096 | 1.0363283157348633 |
| A-RCF2 | D-RAT27 | 0 | 1.0 | 0.6745054721832275 | 0.031543511897325516 | 0.03869047406173888 | -0.21806989759237233 | 1.104325532913208 |
| A-RCF3 | D-RAT25 | 0 | 1.0 | 0.7399504780769348 | 0.032653599977493286 | 0.14484126440116338 | -0.20458610186994297 | 1.0786938667297363 |
| A-RCF4 | D-RAT28 | 0 | 1.0 | 0.7420682907104492 | 0.036946311593055725 | 0.10317459986323402 | -0.20871549393020591 | 1.0895075798034668 |
| A-RCF5 | D-RAT35 | 0 | 1.0 | 0.8013214468955994 | 0.03570229560136795 | 0.07837301350775219 | -0.26594156227916754 | 1.15559983253479 |
| A-RCF6 | D-RAT-OVERCOMPLETE-H40 | 0 | 1.0 | 0.9786343574523926 | 0.041100457310676575 | 0.10793650647004445 | -0.008625071730056444 | 1.0065228939056396 |

KAN synthetic：

```text
v1310_signal_to_cover_summary rows = 504
pass_s3 rows = 26
pass_s4 rows = 0
S3 rows by task = X4:22, X7:4
S3 task family pass count = 2 / 7
S4 task family pass count = 0 / 7
```

最接近但不能写成 S3/S4 family success 的 rows：

| arch | task | seed | loss | method | source_vs_adamw | AUC ratio | NLL delta | ECE delta | Noise delta | Reservoir delta | cover purity | S3 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A-RCF4 | X4 | 0 | Brier | K17 | 0.23820662403222692 | 0.7779236222810887 | -0.40993523597717285 | -0.14754700660705566 | -0.023349523544311523 | 2.444263458251953 | 0.04157925397157669 | 0 |
| A-RCF3 | X3 | 0 | Brier | K15 | 0.22069927229093889 | 0.7864077942207431 | -1.4021024703979492 | -0.1852792501449585 | 0.007045567035675049 | -2.7879445552825928 | 0.042423419654369354 | 0 |
| A-RCF3 | X3 | 0 | Brier | K16 | 0.21983875738012537 | 0.7872405984071762 | -1.3969378471374512 | -0.1837165653705597 | 0.006119787693023682 | -2.7835845947265625 | 0.030586345121264458 | 0 |
| A-RCF3 | X3 | 0 | Brier | K18 | 0.21960501376360253 | 0.7874668148967988 | -1.3968907594680786 | -0.18405979871749878 | 0.006203413009643555 | -2.7839596271514893 | 0.10183577239513397 | 0 |

判断：

```text
1. A-RCF1..A-RCF6 均没有打开 cover substrate gate。
2. cover purity 仍只有 0.0330 median，低于计划 cover gate。
3. 局部 rows 有 source/AUC improvement，但 S3 family coverage 只有 2/7。
4. S4 row = 0，real short-run 不允许打开。
```

## 6. top-2 A-RCF hardening

选择依据：

```text
A-RCF1：official 中局部 S3 row 较多。
A-RCF6：cover purity / mean source delta 相对最好，并且是真 overcomplete h40 substrate primitive。
```

执行规模：

```text
architectures = A-RCF1,A-RCF6
synthetic_tasks = X1..X7
synthetic_seeds = 0,1
loss_interfaces = CE,Brier
k_methods = K0,K14,K15,K16,K17,K18
train_steps = 100
batch_size = 32
compute_budgeted_run = 1
```

结果：

```text
route = R1-NoCoverSubstrate
official_success_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
cover_substrate_pass_count = 0
snr_transfer_median_retention_group = 1.0
snr_transfer_median_cos_group_vs_param = 0.90666663646698
cover_purity_median = 0.037662800401449203
cover_churn_mean = 0.10255101824901541
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 1
kan_s4_task_pass_count = 0
```

逐行：

```text
summary rows = 336
pass_s3 rows = 11
pass_s4 rows = 0
S3 rows by task = X4:11
```

最接近 rows：

| arch | task | seed | loss | method | source_vs_adamw | AUC ratio | Noise delta | Reservoir delta | cover purity | S3 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| A-RCF1 | X4 | 0 | Brier | K16 | 0.19285168026623745 | 0.8247803771129473 | -0.08986139297485352 | -3.7553672790527344 | 0.024080809205770493 | 1 |
| A-RCF1 | X4 | 0 | Brier | K15 | 0.19222067812023558 | 0.8253536879490817 | -0.08942538499832153 | -3.7536203861236572 | 0.07203434407711029 | 1 |
| A-RCF1 | X4 | 0 | Brier | K18 | 0.1921442270838204 | 0.8254231492146071 | -0.08993619680404663 | -3.756009340286255 | 0.023822633549571037 | 1 |
| A-RCF1 | X4 | 0 | Brier | K14 | 0.1909078416078085 | 0.8265464943498466 | -0.08963924646377563 | -3.7545905113220215 | 0.0447942353785038 | 1 |

判断：

```text
top-2 hardening 没有扩大 family coverage；
局部成功集中在 X4，不能写成 S3 >=5/7；
cover purity 仍远低于 0.20。
```

## 7. cover-fail fallback：A-RCF2/A-RCF3/A-RCF6

触发原因：

```text
official 与 top-2 hardening 都是 retention pass but cover substrate fail。
按计划第 9.6 节补跑 A-RCF2/A-RCF3/A-RCF6。
```

执行规模：

```text
architectures = A-RCF2,A-RCF3,A-RCF6
synthetic_tasks = X5,X6,X7
synthetic_seeds = 0,1
loss_interfaces = CE,Brier
k_methods = K0,K15,K18
train_steps = 80
batch_size = 32
compute_budgeted_run = 1
```

结果：

```text
route = R1-NoCoverSubstrate
promotion_allowed = 0
official_success_reached = 0
required_artifact_missing_count = 0
cover_substrate_pass_count = 0
snr_transfer_median_retention_group = 1.0
snr_transfer_median_cos_group_vs_param = 0.7669346332550049
cover_purity_median = 0.029089346528053284
cover_churn_mean = 0.0
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
```

逐行：

```text
summary rows = 108
pass_s3 rows = 0
pass_s4 rows = 0
```

最接近 rows：

| arch | task | seed | loss | method | source_vs_adamw | AUC ratio | NLL delta | ECE delta | Noise delta | Reservoir delta | cover purity | S3 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A-RCF3 | X7 | 0 | Brier | K15 | 0.037935530007538465 | 0.9590730590865639 | -0.6425219774246216 | -0.13214942812919617 | 0.023145079612731934 | -3.1960136890411377 | 0.020426275208592415 | 0 |
| A-RCF3 | X7 | 0 | Brier | K18 | 0.03787730759205543 | 0.9591358726378136 | -0.6406513452529907 | -0.10882076621055603 | 0.02211672067642212 | -3.187980890274048 | 0.03882602974772453 | 0 |
| A-RCF2 | X7 | 0 | Brier | K15 | 0.008107992174341994 | 0.9908940931171882 | -0.5157152414321899 | -0.08692371845245361 | 0.06374067068099976 | -3.441843271255493 | 0.0515771359205246 | 0 |

判断：

```text
A-RCF2/A-RCF3/A-RCF6 fallback 没有打开 cover substrate；
NoiseSignalLeak 坏化仍是局部 blocker；
S3/S4 均为 0。
```

## 8. Non-RAT vertical slice 与 MLP control

Non-RAT official：

```text
rows = 10
nonrat_vertical_pass_count = 0
promotion_allowed = 0
```

说明：

```text
1. Fourier / RBF / WAV 等没有可用 exact no-materialize prior 时 fail-closed，workspace ratio 记录为 9.0。
2. Chebyshev mapped prior 存在，但 workspace_incremental_ratio 最高到 13.232558139534884，仍 fail。
3. LineC / hardening 未因 workspace fail 被写成 pass。
4. Non-RAT 仍不能进入 functional proof。
```

MLP control：

```text
mlp_generic_dataset_count = 3
mlp_generic_dataset_pass_count = 0
mlp_generic_confirmed = 0
```

解释：

```text
MLP row 只是 generic optimizer monitor；
本轮 MLP control 也没有形成 3/3 dataset confirmed。
```

## 9. Required artifacts

official_v1310 required manifest：

```text
manifest_rows = 27
missing_required_rows = 0
```

主要产物：

```text
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_route_decision.json
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_required_manifest.csv
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_code_review_manifest.csv
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_architecture_diff_manifest.csv
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_forbidden_information_audit.csv
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_substrate_architecture_readback.md
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_rational_cover_substrate.csv
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_rational_cover_telemetry.csv
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_signal_to_cover_training.csv
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_signal_to_cover_summary.csv
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_signal_to_cover_linec.csv
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_nonrat_substrate_vertical_slice.csv
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_mlp_control_monitor.csv
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_linec_audit.csv
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_failure_table.csv
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_no_go_boundary.md
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_next_hypothesis_queue.md
results/v13_10_cover_forming_substrate_architecture_reset/official_v1310/v1310_code_review_packet.zip
```

repair artifacts：

```text
results/v13_10_cover_forming_substrate_architecture_reset/smoke_v1310/
results/v13_10_cover_forming_substrate_architecture_reset/hardening_v1310_top2_arcf1_arcf6/
results/v13_10_cover_forming_substrate_architecture_reset/fallback_v1310_arcf2_arcf3_arcf6_coverfail/
```

## 10. 最终科学结论

v13.10 没有达成 S1/S2/S3/S4/S5；最终合法 official route：

```text
R1-NoCoverSubstrate
minimum_success = S0-ArchitectureScoutExecuted
promotion_allowed = 0
official_success_reached = 0
kan_real_short_run_open_allowed = 0
```

已闭合事实：

```text
1. A-RCF1..A-RCF6 均已执行，但 cover_substrate_pass_count = 0。
2. 这不是 K-token-only extension：is_new_substrate_architecture=1, is_k_token_only_extension=0。
3. signal retention/cosine 可见，但 stable cover 没形成：official cover_purity_median = 0.0330。
4. official K14-K18 summary rows = 504，S3 family pass = 2/7，S4 rows = 0。
5. top-2 A-RCF hardening 只得到 S3 family pass = 1/7，S4 = 0。
6. A-RCF2/A-RCF3/A-RCF6 cover-fail fallback 仍为 S3/S4 = 0。
7. Non-RAT vertical slice pass = 0，不能进入 functional proof。
8. MLP control pass = 0/3 datasets，且不能写成 KAN promotion。
9. required artifacts 缺失为 0，forbidden information violation 为 0。
```

no-go boundary：

```text
1. v13.10 证明只做 architecture reset seed / group permutation / overcomplete h40 bank 仍没有形成 stable basis cover。
2. 当前 blocker 仍是 cover substrate gate：cover purity 极低，family coverage 不足。
3. 局部 source/AUC positive rows 不能覆盖 task family，也不能打开 S4。
4. Non-RAT 仍卡在 exact workspace / vertical-slice substrate gate。
5. 继续在当前 runner 内排列 K14-K18 或微调 A-RCF2/A-RCF3/A-RCF6 小参数价值低；
   下一步需要更根本的 cover-forming base primitive 或重新定义 cover objective，但不能把 scout row 编造成 promotion。
```

最终判断：

```text
v13.10 未达成目标；
不允许 promotion；
不允许 real short-run；
允许 final stop，原因是计划内 A-RCF1..A-RCF6 scout、top-2 hardening、cover-fail fallback、Non-RAT vertical slice、MLP monitor、no-go boundary 与 next queue 均已执行且未打开 gate。
```
