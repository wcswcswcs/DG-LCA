# DG-KAN v18.0 Evidence-First 实验结果复盘

生成时间：2026-06-02 22:12:33 +08

## Route

- route: `R1-S0_2DebtMetricsIncomplete`
- route_detail: v18 executed fresh matrices but evidence-complete promotion blocked: debt_complete=0, non_mlp_efficiency_pass_rows=0, h800_candidates=0, h1600_candidates=0, late_rebound_candidates=1, deferred_kernel_repair_variants=29
- CodeRoute: S0_2-CodeCorrectnessPassedDebtReadbackIncomplete
- EfficiencyRoute: R-EfficiencyForwardBlocked-D-CHE-D-FOU-D-RAT-D-RBF-D-WAV-LQ
- FunctionalRoute: S3b-LateReboundNeedsIndependentRerun
- S0_2_preflight_pass: 1
- S0_2_evidence_complete: 0
- debt_metrics_complete: 0
- measured functional rows: 945
- measured efficiency rows: 28
- measured debt rows: 945
- required artifact missing count: 0
- promotion_allowed: 0

## 覆盖边界

- 已完成：S0 import/LineC/update/retention/route/kernel correctness 预检；fresh functional matrix；fresh h400/h800/h1600/h3200 horizon matrix；batch 8/32/128/256 efficiency census；v18 packet、queue、failure taxonomy、两份日志。
- 已真实修复/补齐：新增 v18 wrappers；新增 v18 final artifacts；新增 v18 code packet 结构；新增 CEp99 tail debt 与 val_loss AUC readback；新增 v18 kernel repair matrix，把未实现 repair variants 显式写成 deferred。
- 未完成/不可冒充：LineC-channel/ECE/Brier debt 未进入 horizon trace；D-FOU/D-CHE/LQ/D-RAT/D-RBF/D-WAV R1+ official fused repair 未实现；S4/S5 real-transfer official gate 未执行；因此不能写 breakthrough/promotion-ready。

## Part A：S0.2 Metric / Implementation Truth Gate

- compile/import/LineC/update/kernel preflight route: S0_2-CodeCorrectnessPassedDebtReadbackIncomplete
- debt rows incomplete: 945/945
- deferred kernel repair variants: 29
- 结论：代码正确性预检可以支撑继续跑 fresh rows；但 S0.2 evidence-complete gate 未过，因为计划要求的 LineC-channel/ECE/Brier debt 还没有真实 readback。

| item | value | evidence |
|---|---:|---|
| functional raw horizon rows | 4725 | `v18_functional_raw_horizon_matrix.csv` |
| source grouped rows | 105 | `v18_source_retention_matrix.csv` |
| debt rows | 945 | `v18_debt_accounting_matrix.csv` |
| required manifest missing | 0 | `v18_required_artifact_manifest.csv` |
| packet manifest rows | 586 | `v18_code_review_packet/packet_manifest.csv` |

## Part B：Basis Kernel Efficiency

- EfficiencyRoute: R-EfficiencyForwardBlocked-D-CHE-D-FOU-D-RAT-D-RBF-D-WAV-LQ
- v18 gate: forward/backward/step <= 1.75，memory <= 1.10，audit overhead <= 0.20；official gate 还要求 official fused kernel complete。

| blocker_class | rows |
|---|---:|
| ForwardBlocked | 5 |
| ForwardBlocked;BackwardBlocked;StepBlocked | 14 |
| ForwardBlocked;BackwardBlocked;StepBlocked;AuditOverheadBlocked | 3 |
| ForwardBlocked;StepBlocked | 2 |
| pass | 4 |

| carrier | batch | forward | backward | step | memory | audit | blocker |
|---|---:|---:|---:|---:|---:|---:|---|
| MLP | 8 | 1.3941870479120224 | 1.2709682553880395 | 1.2771627922042093 | 1.0 | 0.04655009762092047 | pass |
| D-CHE | 8 | 5.961686539695522 | 2.492918599594155 | 2.6020825094184654 | 1.001449318228874 | 0.10984842821365827 | ForwardBlocked;BackwardBlocked;StepBlocked |
| LQ | 8 | 6.851857201234377 | 1.6032491585627882 | 1.7941203054896546 | 1.001449318228874 | 0.14931245356401646 | ForwardBlocked;StepBlocked |
| D-RAT | 8 | 8.039730325644506 | 2.539719939958919 | 2.733207600394006 | 1.001449318228874 | 0.15169077325027486 | ForwardBlocked;BackwardBlocked;StepBlocked |
| D-FOU | 8 | 4.849268358642422 | 3.444300514235545 | 3.406603527737026 | 1.001449318228874 | 0.1622139275549736 | ForwardBlocked;BackwardBlocked;StepBlocked |
| D-RBF | 8 | 9.936802725438785 | 5.223669412662389 | 5.347146373408126 | 1.0021000325357154 | 0.1883218710250174 | ForwardBlocked;BackwardBlocked;StepBlocked |
| D-WAV | 8 | 16.549356194318374 | 2.5104997444685555 | 2.9494866933488897 | 1.0021000325357154 | 0.26722890035440233 | ForwardBlocked;BackwardBlocked;StepBlocked;AuditOverheadBlocked |
| MLP | 32 | 1.351281730150688 | 0.6944614369263146 | 0.740727865570203 | 1.0 | 0.06110482824245247 | pass |
| D-CHE | 32 | 4.174640998909425 | 2.817136138099403 | 2.8239164818300457 | 1.002246327550025 | 0.14073223053106643 | ForwardBlocked;BackwardBlocked;StepBlocked |
| LQ | 32 | 6.225616757926292 | 1.7305874770192826 | 1.9740789355875743 | 1.002246327550025 | 0.17976828421581698 | ForwardBlocked;StepBlocked |
| D-RAT | 32 | 8.435890168184816 | 3.1862632638930752 | 3.458639470199161 | 1.0032808205006947 | 0.15492435073840197 | ForwardBlocked;BackwardBlocked;StepBlocked |
| D-FOU | 32 | 5.29448666108242 | 1.335545891579323 | 1.508038764857438 | 1.0037537315638578 | 0.15938780319956258 | ForwardBlocked |
| D-RBF | 32 | 10.348887751344847 | 2.693882295552414 | 3.0727306571964985 | 1.0022167706085774 | 0.19237381508260407 | ForwardBlocked;BackwardBlocked;StepBlocked |
| D-WAV | 32 | 16.40125629958284 | 2.4277294559352622 | 3.0407810436714504 | 1.0036355037980669 | 0.22096859164909383 | ForwardBlocked;BackwardBlocked;StepBlocked;AuditOverheadBlocked |
| MLP | 128 | 1.2294120377829418 | 0.710463595343856 | 0.7512134551083365 | 1.0 | 0.054738767043878526 | pass |
| D-CHE | 128 | 4.52901468973055 | 1.513196910108985 | 1.6259933846559804 | 1.008511637462945 | 0.12457307285468956 | ForwardBlocked |
| LQ | 128 | 5.044375113966483 | 1.792119918697663 | 1.9681260729931758 | 1.008511637462945 | 0.16454078016817206 | ForwardBlocked;BackwardBlocked;StepBlocked |
| D-RAT | 128 | 6.481053201493175 | 1.7748429568097293 | 1.9608299210175286 | 1.0141469284728948 | 0.14966167075375117 | ForwardBlocked;BackwardBlocked;StepBlocked |
| D-FOU | 128 | 4.870098095588573 | 1.3891795150214714 | 1.5787065989675404 | 1.0160253588095447 | 0.16791696847821133 | ForwardBlocked |
| D-RBF | 128 | 10.099862396616405 | 2.1114915769464693 | 2.408909384603444 | 1.0087464412550262 | 0.17268094374993098 | ForwardBlocked;BackwardBlocked;StepBlocked |
| D-WAV | 128 | 14.599220543045414 | 3.4027786864583334 | 3.9059462899983766 | 1.0150861436412197 | 0.19262354917289948 | ForwardBlocked;BackwardBlocked;StepBlocked |
| MLP | 256 | 1.3311886470644068 | 0.7022030229129465 | 0.7531624652068186 | 1.0 | 0.05936488183373069 | pass |
| D-CHE | 256 | 4.707954273235247 | 1.5562985431631597 | 1.6708319291116045 | 1.01687874034048 | 0.11731175723982615 | ForwardBlocked |
| LQ | 256 | 5.92058650078282 | 1.2395537010443969 | 1.4374430009930554 | 1.01687874034048 | 0.16756090289612194 | ForwardBlocked |
| D-RAT | 256 | 6.742697609096984 | 1.828537046813538 | 2.02921535926708 | 1.0280343966068213 | 0.1468893192221357 | ForwardBlocked;BackwardBlocked;StepBlocked |
| D-FOU | 256 | 3.882617424921783 | 1.8231734785899956 | 1.9315582763793135 | 1.0317529486956016 | 0.17467837680584428 | ForwardBlocked;BackwardBlocked;StepBlocked |
| D-RBF | 256 | 10.118282775147582 | 1.9766380051823282 | 2.293722107208428 | 1.017227354598803 | 0.18146717957141928 | ForwardBlocked;BackwardBlocked;StepBlocked |
| D-WAV | 256 | 13.034003220420814 | 2.3513982654143013 | 2.739155808812238 | 1.029777467898437 | 0.2154507347265149 | ForwardBlocked;BackwardBlocked;StepBlocked;AuditOverheadBlocked |

### Kernel Repair Matrix Summary

| family | variants | implemented_current | deferred | official_fused_complete |
|---|---:|---:|---:|---:|
| D-CHE | 6 | 1 | 5 | 0 |
| D-FOU | 7 | 1 | 6 | 0 |
| D-RAT | 6 | 1 | 5 | 0 |
| D-RBF | 6 | 1 | 5 | 0 |
| D-WAV | 4 | 1 | 3 | 0 |
| LQ | 6 | 1 | 5 | 0 |

## Part C：Functional Update

- FunctionalRoute: S3b-LateReboundNeedsIndependentRerun
- h800 candidates: 0
- h1600 retained candidates: 0
- late rebound candidates: 1

### Source / Retention Evidence

| horizon | carrier | mechanism | mean_source | pass_count | retention |
|---|---|---|---:|---:|---:|
| h100 | D-RBF | M1-AdamWPrimaryFUResidual | 0.0027029779222276476 | 6 |  |
| h100 | D-CHE | M1-AdamWPrimaryFUResidual | 0.0006251732508341471 | 3 |  |
| h100 | LQ | M1-AdamWPrimaryFUResidual | -0.005612346861097548 | 2 |  |
| h100 | D-WAV | M1-AdamWPrimaryFUResidual | -0.006021446651882595 | 4 |  |
| h100 | D-RAT | M1-AdamWPrimaryFUResidual | -0.006166113747490777 | 3 |  |
| h800 | MLP | M2-SGDMomentumPrimaryFU | 0.1751598914464315 | 7 |  |
| h800 | D-CHE | M1-AdamWPrimaryFUResidual | -0.002686844931708442 | 5 | 0.0 |
| h800 | D-RBF | CTRL-RecoveryOnly | -0.007917450533972846 | 0 |  |
| h800 | D-WAV | CTRL-AdamW | -0.009836514790852865 | 0 |  |
| h800 | D-CHE | CTRL-AdamW | -0.012133015526665581 | 0 |  |
| h1600 | D-CHE | M1-AdamWPrimaryFUResidual | 0.0032123857074313695 | 7 |  |
| h1600 | D-RBF | M1-AdamWPrimaryFUResidual | -0.00665075249142117 | 4 |  |
| h1600 | D-WAV | CTRL-RecoveryOnly | -0.010287218623691134 | 0 |  |
| h1600 | LQ | CTRL-RecoveryOnly | -0.011402944723765055 | 0 |  |
| h1600 | LQ | M1-AdamWPrimaryFUResidual | -0.012305027908749051 | 4 |  |
| h3200 | D-CHE | M2-SGDMomentumPrimaryFU | 0.09768134355545044 | 5 |  |
| h3200 | MLP | CTRL-SGD | 0.0 | 0 |  |
| h3200 | MLP | M5-AlternatingFUGradient | -0.009376459651523165 | 3 |  |
| h3200 | D-RBF | M1-AdamWPrimaryFUResidual | -0.023841241995493572 | 3 |  |
| h3200 | D-RBF | CTRL-AdamW | -0.02936577796936035 | 0 |  |

### Control / KAN-vs-MLP Attribution

| metric | value |
|---|---:|
| control rows explaining positive | 102/105 |
| KAN advantage h1600 rows | 20/105 |
| KAN advantage h3200 rows | 37/105 |

### Debt Evidence

- CEp99 tail debt 和 val_loss AUC 是从真实 train/horizon trace 计算的。
- LineC-channel/ECE/Brier debt 字段保持空值并标记 `not_measured`，所以本轮不能写 debt recovery success。

| carrier | mechanism | dataset | seed | tail_peak | tail_final | recovery | status |
|---|---|---|---:|---:|---:|---:|---|
| D-CHE | CTRL-AdamW | Fashion-MNIST | 0 | 4.991177558898926 | 4.991177558898926 | 0.0 | tail_CEp99_and_val_loss_AUC_measured;LineC_channel_ECE_Brier_debt_not_measured |
| D-CHE | CTRL-AdamW | Fashion-MNIST | 1 | 2.730604410171509 | 2.730604410171509 | 0.0 | tail_CEp99_and_val_loss_AUC_measured;LineC_channel_ECE_Brier_debt_not_measured |
| D-CHE | CTRL-AdamW | Fashion-MNIST | 2 | 5.829788446426392 | 5.829788446426392 | 0.0 | tail_CEp99_and_val_loss_AUC_measured;LineC_channel_ECE_Brier_debt_not_measured |
| D-CHE | CTRL-AdamW | KMNIST | 0 | 6.013531684875488 | 6.013531684875488 | 0.0 | tail_CEp99_and_val_loss_AUC_measured;LineC_channel_ECE_Brier_debt_not_measured |
| D-CHE | CTRL-AdamW | KMNIST | 1 | 4.395620107650757 | 4.395620107650757 | 0.0 | tail_CEp99_and_val_loss_AUC_measured;LineC_channel_ECE_Brier_debt_not_measured |
| D-CHE | CTRL-AdamW | KMNIST | 2 | 3.592651605606079 | 3.592651605606079 | 0.0 | tail_CEp99_and_val_loss_AUC_measured;LineC_channel_ECE_Brier_debt_not_measured |
| D-CHE | CTRL-AdamW | MNIST | 0 | 3.6166694164276123 | 3.6166694164276123 | 0.0 | tail_CEp99_and_val_loss_AUC_measured;LineC_channel_ECE_Brier_debt_not_measured |
| D-CHE | CTRL-AdamW | MNIST | 1 | 4.9320714473724365 | 4.9320714473724365 | 0.0 | tail_CEp99_and_val_loss_AUC_measured;LineC_channel_ECE_Brier_debt_not_measured |
| D-CHE | CTRL-AdamW | MNIST | 2 | 4.711791038513184 | 4.711791038513184 | 0.0 | tail_CEp99_and_val_loss_AUC_measured;LineC_channel_ECE_Brier_debt_not_measured |
| D-CHE | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 0 | 0.0 | 0.0 | 1.0 | tail_CEp99_and_val_loss_AUC_measured;LineC_channel_ECE_Brier_debt_not_measured |
| D-CHE | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 1 | 0.0 | 0.0 | 1.0 | tail_CEp99_and_val_loss_AUC_measured;LineC_channel_ECE_Brier_debt_not_measured |
| D-CHE | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 2 | 0.0 | 0.0 | 1.0 | tail_CEp99_and_val_loss_AUC_measured;LineC_channel_ECE_Brier_debt_not_measured |

## 4GPU / Queue 证据

- GPU snapshot rows: 948
- max memory used MB: 693.0
- max util percent: 22.0
- queue drain report: `results/v18_0_evidencefirst_breakthrough_functional_update_basis_efficiency_4gpu/official_v18/v18_gpu_queue_drain_report.csv`

## 修改记录（便于审计）

- 新增 v18 runner wrappers：`experiments/run_v18_code_gate.py`、`run_v18_functional_update_breakthrough.py`、`run_v18_basis_efficiency_breakthrough.py`、`run_v18_merge_finalize.py`。
- 修改 `experiments/run_v17_common.py`：增加 v18 doc paths、v18 final artifacts、v18 packet、v18 debt matrix、v18 route decision、v18 required manifest。
- 新增 v18 debt accounting：从真实 trace 计算 CEp99 tail delta/recovery 和 val_loss AUC；未真实测量的 LineC-channel/ECE/Brier debt 明确标注 incomplete。
- 新增 v18 kernel repair matrix：R0-current measured，R1+ 未实现逐项 deferred；`official_fused_kernel_complete=0`。

## 分析 / Insight / 结论

- 本轮如果出现 h3200 转正但 h1600 非正，只能叫 late rebound；不能算 retention，也不能进入 promotion。
- 当前最硬 blocker 不是显存，而是 basis forward/backward compute 和 official fused repair 缺失；memory ratio 接近阈值内不等于效率 gate 通过。
- AdamW overwrite diagnostic 仍是判断 FU 被 optimizer writeback 洗掉的关键证据，但必须和 AdamW-free / slow-state rows 的跨 horizon retention 一起看。
- v18 的 evidence-first 结论是：fresh rows 已跑、证据包齐备，但 S0.2 evidence-complete / efficiency official / functional retained source 均未满足；不能写 breakthrough。
