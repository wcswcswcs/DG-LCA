# DG-KAN v19.0 Source-Channel FU + Basis Kernel Breakthrough 实验结果复盘

生成时间：2026-06-03 03:40:37 +08
复盘追加时间：2026-06-03 04:24:03 +08（r4-r6 continuation fresh rows）
复盘追加时间：2026-06-03 05:12:34 +08（r7-r10 carrier/block continuation fresh rows）
复盘追加时间：2026-06-03 05:46:00 +08（H4 function-space actuation solver diagnostic）
复盘追加时间：2026-06-03 06:54:39 +08（H7 failure anatomy / no-actionable-same-family decision）
复盘追加时间：2026-06-03 07:04:03 +08（H8 M29 consensus source-state new-theory smoke）
复盘追加时间：2026-06-03 07:13:48 +08（H9 M30 low-threshold consensus gate repair smoke）
复盘追加时间：2026-06-03 07:28:52 +08（H10 M31 AdamW consensus residual full 9-row rerun）
复盘追加时间：2026-06-03 07:42:00 +08（H11 M31 independent confirmation failed）
复盘追加时间：2026-06-03 07:47:00 +08（H12 M32 post-AdamW recompute smoke negative）
复盘追加时间：2026-06-03 10:06:49 +08（H13 H10/H11 offline anatomy）

## Route

- route: `R-MLPGenericRetained-KANCarrierSourceStateFamilyNotRobust`
- route_detail: H10 M31 partial positive was not independently confirmed by H11; H12 post-AdamW recompute M32 smoke was negative; H13 offline anatomy confirmed H10 h3200 positives wash out by h4800 and do not reproduce under independent init offsets. No same-family M31/M32 repair remains actionable without a new source-observability theory.
- CodeRoute: S0_3-CodeMetricMechanismPassed
- EfficiencyRoute: E2-D-FOU-Pass;E2-D-CHE-Pass
- FunctionalRoute: S3-MLPGenericRetainedSource;H10M31PartialPositive;H11IndependentConfirmationFailed;H12M32PostAdamWSmokeNegative;H13H10H11AnatomyNoRepro;NoActionableSourceStateFamilyRepair
- S0_3_preflight_pass: 1
- S0_3_evidence_complete: 1
- LineC_fast_golden_pass: 1
- LineC_channel_golden_pass: 1
- debt_metrics_complete: 1
- measured functional rows: 1197
- measured efficiency rows: 76
- measured debt rows: 1197
- late rebound rerun rows: 162
- late rebound M2 rerun rows: 27
- late rebound M2 positive rerun rows: 25
- functional continuation rows: 819
- functional continuation grouped probes: 61
- functional continuation retained candidates: 1
- MLP retained source count: 1
- KAN M16 same-carrier retained candidates: 0
- KAN M17 same-carrier retained candidates: 0
- KAN repaired carrier retained candidates: 0
- H4 actuation smoke rows: 12
- H4 best ActuationR2: 0.09360003471374512
- H4 long horizon executed: 0
- H4 extended actuation rows: 927
- H5 M28 retained candidates: 0
- H6 source-state positive smoke rows: 0
- H7 failure anatomy rows: 7
- H7 continue same-family recommended: 0
- H7 new theory required: 1
- H8 M29 smoke rows: 3
- H8 M29 matrix rows: 3
- H8 M29 gate accepts/trials: 0/12
- H8 M29 max h800 source: -0.9486403465270996
- H8 M29 max h1600 source: -0.8269315958023071
- H8 full escalation: 0
- H9 M30 smoke rows: 3
- H9 M30 matrix rows: 3
- H9 M30 gate accepts/trials: 12/12
- H9 M30 max h800 source: -0.9478328227996826
- H9 M30 max h1600 source: -0.8264065980911255
- H9 full escalation: 0
- H10 M31 smoke summary rows: 3
- H10 M31 smoke positive rows: 3
- H10 M31 full matrix rows: 54
- H10 M31 full M31 rows: 27
- H10 M31 full summary rows: 3
- H10 M31 retained candidates through h3200: 2
- H10 M31 best h800/h1600/h3200/h4800 grouped mean: 0.016972078217400446 / 0.02917080455356174 / 0.02758270502090454 / -0.007561306158701579
- H10 M31 gate accepts/trials: 55/189
- H10 M31 h4800 washout: 1
- H10 M31 GPU snapshot rows: 76
- H10 M31 max memory MB / util percent: 693.0 / 15.0
- H11 M31 independent matrix rows: 135
- H11 M31 independent M31 rows: 54
- H11 M31 independent per-rep/spec rows: 6
- H11 independent rerun count: 3
- H11 reproduced retained-to-h3200 rep/spec count: 0/6
- H11 h4800 positive rep/spec count: 0/6
- H11 best h800/h1600/h3200/h4800 aggregate mean: -0.06181360836382265 / -0.09745090979116934 / -0.12961135970221627 / -0.16300636309164543
- H11 M31 gate accepts/trials: 109/378
- H11 next fix: H12 post-AdamW recompute consensus residual
- H12 M32 smoke matrix rows: 6
- H12 M32 smoke summary rows: 3
- H12 M32 smoke positive rows: 0
- H12 M32 best h800/h1600 mean: -0.08437776565551758 / -0.12100136280059814
- H12 M32 gate accepts/trials: 6/12
- H12 full escalation: 0
- H12 continue same-family recommended: 0
- H12 new source theory required: 1
- H13 anatomy completed: 1
- H13 H10/H11 h3200 positive spec count: 2 / 0
- H13 H10/H11 h4800 positive spec count: 0 / 0
- H13 actionable same-family repair: 0
- required artifact missing count: 0
- promotion_allowed: 0

## 覆盖边界

- 已完成项只按 artifact 真实存在和 measured rows 记录；未执行、未通过、未实现项不会写成成功。
- v19 新增/修复：LineC-channel golden、S0.3 debt formula edge cases、mechanism semantic contract、LineC-fast/channel horizon trace、CEp99/NLL/ECE/Brier/AUCtime debt matrix、basis repair variant matrix、D-CHE late rebound independent rerun入口、v19 packet/bundle。
- continuation 追加修复/审计：D-FOU/D-CHE E2 basis efficiency repair；M15 LineC-filtered pulse；M16 two-phase momentum warmup + LineC-filtered sparse writer；M17 readout-carrier two-phase writer；M18 low-rank carrier block FU；M19 split-consensus low-rank block FU；M20 train-split function-space actuation solver；M21-M28 exact-readout/high-cap/AdamW/multi-batch gated actuator；M29 cross-batch consensus source-state writer；M30 low-threshold consensus gate repair；M31 AdamW low-threshold consensus residual；H6 dataset/seed heterogeneity；H7 failure anatomy decision；continuation runner 支持 same-carrier 和 basis-repair-variant source/control 分组。
- 不可冒充项：若 `debt_metrics_complete=0`、D-FOU/D-CHE efficiency gate 未过、late rebound rerun 未复现、或 official fused/no-materialize kernel 未完成，则不能写 breakthrough/promotion-ready。

## Part A：S0.3 Code / Metric / Mechanism Truth Gate

- s0_pass: 1
- linec_fast golden: 9/9
- linec_channel golden: 8/8
- mechanism semantic contract pass: 1
- required artifact missing count: 0

### LineC-channel Golden

| golden | pass | valid | CouplingR2 | NoiseLeak | route |
|---|---|---|---|---|---|
| C1-NoOpNull | 1 | 1 | 0.0 | 0.0 | R-LineCMeasured |
| C2-KnownTransferPositive | 1 | 1 | 0.995116607471406 | 0.0 | R-LineCMeasured |
| C3-NoiseLeakPositive | 1 | 1 | 0.0 | 0.23868066987062508 | R-LineCMeasured |
| C4-ReservoirOnlyPositive | 1 | 1 | 0.0 | 0.0010028782777381706 | R-LineCMeasured |
| C5-BatchPermutationMismatch | 1 | 1 | 0.0 | 0.0 | R-LineCMeasured |
| C6-ExceptionMeasurementInvalid | 1 | 0 |  |  | R0-LineCMeasurementInvalid |
| C7-ScaleInvariance | 1 | 1 | 0.9951164885535491 | 0.0 | R-LineCMeasured |
| C8-LongHorizonRecoverySynthetic | 1 | 1 | 1.0 | 0.0 | R-LineCMeasured |

### Debt Metric Availability

| metric | available_rows | total_rows | complete |
|---|---|---|---|
| CEp99 | 1197 | 1197 | 1 |
| NLL | 1197 | 1197 | 1 |
| ECE | 1197 | 1197 | 1 |
| Brier | 1197 | 1197 | 1 |
| LineC_fast_loss | 1197 | 1197 | 1 |
| LineC_channel_loss | 1197 | 1197 | 1 |
| AUCtime | 1197 | 1197 | 1 |

### Mechanism Semantic Contract

| mechanism | source_space | optimizer_primary | slow | matrix | poprisk | prototype |
|---|---|---|---|---|---|---|
| CTRL-AdamW | parameter | AdamW | 0 | 0 | 0 | 0 |
| CTRL-SGD | parameter | SGD | 0 | 0 | 0 | 0 |
| CTRL-NoOpMatchedOverhead | parameter | none | 0 | 0 | 0 | 0 |
| CTRL-RandomMatchedNorm | parameter | none | 0 | 0 | 0 | 0 |
| CTRL-RecoveryOnly | parameter | AdamW | 0 | 0 | 0 | 0 |
| M1-AdamWPrimaryFUResidual | parameter | AdamW | 0 | 0 | 0 | 0 |
| M2-SGDMomentumPrimaryFU | parameter | Momentum | 0 | 0 | 0 | 0 |
| M3-FUPrimary | function | none | 0 | 0 | 0 | 1 |
| M4-FUOnlyKeyParams | basis-channel | none | 0 | 0 | 0 | 1 |
| M5-AlternatingFUGradient | trajectory | SGD | 0 | 0 | 0 | 1 |
| M6-SlowStateFU | slow_state | none | 1 | 0 | 0 | 1 |
| M7-MatrixBlockFU | block | none | 0 | 1 | 0 | 1 |
| M8-PopRiskSNRFU | basis-channel | none | 0 | 0 | 1 | 0 |
| M9-FunctionSpaceOperatorFU | function | none | 0 | 0 | 0 | 1 |
| M10-RolePartitionOptimizer | role_partition | role_partition | 0 | 0 | 0 | 1 |
| M11-DualMemorySlowStateFU | slow_state | none | 1 | 0 | 0 | 0 |
| M12-ScheduleFreeAveragedFU | slow_state | schedule_free | 1 | 0 | 0 | 0 |
| M13-LowRankMatrixBlockFU | block | none | 0 | 1 | 0 | 0 |
| M14-SourceChannelPopRiskSlowFU | basis-channel | none | 1 | 0 | 1 | 0 |

## Part B：Basis Kernel Efficiency

- D-FOU efficiency pass: 1
- D-CHE efficiency pass: 1
- Gate 解释：forward/step/memory 等 ratio 达标只是 exploration 条件；official success 还需要 official fused/no-materialize kernel complete。

### Blocker Counts

| blocker_class | rows |
|---|---:|
| FamilyPass | 23 |
| KernelForwardBlocked | 23 |
| KernelForwardBlocked;KernelBackwardBlocked;StepBlocked | 16 |
| KernelForwardBlocked;StepBlocked | 14 |

### D-FOU / D-CHE Repair Rows

| family | variant | batch | forward | step | memory | gate | blocker |
|---|---|---|---|---|---|---|---|
| D-CHE | CHE-R0-current | 8 | 7.061573237795371 | 0.7794184994652952 | 1.0026028572273655 | 0 | KernelForwardBlocked |
| D-CHE | CHE-R1-k3-triton-no-materialize | 8 | 1.4910767749223734 | 0.5027003384302209 | 1.0002670940170941 | 1 | FamilyPass |
| D-CHE | CHE-R2-low-degree-k3-triton | 8 | 1.2037623055466475 | 0.4712448741097888 | 1.0002670940170941 | 1 | FamilyPass |
| D-CHE | CHE-R3-k4-triton-no-materialize | 8 | 1.2675542421650208 | 0.701084054039161 | 1.0002366233843059 | 1 | FamilyPass |
| D-CHE | CHE-R4-k3-gradbuf-triton | 8 | 1.1815494958186243 | 0.50746797765961 | 1.0002667615152054 | 1 | FamilyPass |
| D-FOU | FOU-R0-current | 8 | 6.307724101043009 | 1.175964611030166 | 1.0026028572273655 | 0 | KernelForwardBlocked |
| D-FOU | FOU-R1-k2-triton-no-materialize | 8 | 1.4091936189936656 | 0.579003447013354 | 1.0002679847546452 | 1 | FamilyPass |
| D-FOU | FOU-R2-low-frequency-k2-stream | 8 | 1.2268810476627063 | 0.5507452893885865 | 1.0002679847546452 | 1 | FamilyPass |
| D-FOU | FOU-R3-k3-triton-no-materialize | 8 | 1.3534255749882036 | 0.45557572119550016 | 1.0002670940170941 | 1 | FamilyPass |
| D-FOU | FOU-R4-k4-triton-no-materialize | 8 | 1.420512328207161 | 0.4082961330186338 | 1.0002366233843059 | 1 | FamilyPass |
| D-CHE | CHE-R0-current | 32 | 5.754466493184759 | 0.9191251923885475 | 1.005024680046109 | 0 | KernelForwardBlocked |
| D-CHE | CHE-R1-k3-triton-no-materialize | 32 | 2.1807512419022164 | 0.5447418515209199 | 0.99997034225043 | 0 | KernelForwardBlocked |
| D-CHE | CHE-R2-low-degree-k3-triton | 32 | 1.1208686647437704 | 0.5801971582512254 | 0.99997034225043 | 1 | FamilyPass |
| D-CHE | CHE-R3-k4-triton-no-materialize | 32 | 1.4989063664706106 | 0.5084673263958256 | 0.9998817722342092 | 1 | FamilyPass |
| D-CHE | CHE-R4-k3-gradbuf-triton | 32 | 0.9518846020846713 | 0.4486547674160997 | 0.9999703791469194 | 1 | FamilyPass |
| D-FOU | FOU-R0-current | 32 | 8.115950959206478 | 1.8643572546755862 | 1.005024680046109 | 0 | KernelForwardBlocked;StepBlocked |
| D-FOU | FOU-R1-k2-triton-no-materialize | 32 | 1.6465444129293156 | 0.5651003220543603 | 1.0000595184953724 | 1 | FamilyPass |
| D-FOU | FOU-R2-low-frequency-k2-stream | 32 | 1.0071820428525156 | 0.4435730513786116 | 1.0000595184953724 | 1 | FamilyPass |
| D-FOU | FOU-R3-k3-triton-no-materialize | 32 | 6.707518778353454 | 1.2443310473550497 | 0.99997034225043 | 0 | KernelForwardBlocked |
| D-FOU | FOU-R4-k4-triton-no-materialize | 32 | 6.83045159225156 | 1.359393063918053 | 0.9998817722342092 | 0 | KernelForwardBlocked |
| D-CHE | CHE-R0-current | 128 | 5.5851226748343175 | 0.8606085833907599 | 1.0129729095124886 | 0 | KernelForwardBlocked |
| D-CHE | CHE-R1-k3-triton-no-materialize | 128 | 2.234179167313931 | 2.5287532076654724 | 0.9962855963681386 | 0 | KernelForwardBlocked;KernelBackwardBlocked;StepBlocked |
| D-CHE | CHE-R2-low-degree-k3-triton | 128 | 1.0607762995169618 | 1.0206894109397742 | 0.9962855963681386 | 1 | FamilyPass |
| D-CHE | CHE-R3-k4-triton-no-materialize | 128 | 2.1513174480698565 | 2.642820090492077 | 0.9952158727363447 | 0 | KernelForwardBlocked;KernelBackwardBlocked;StepBlocked |

### Efficiency Best Rows

| carrier | variant | batch | forward | backward | step | memory | gate |
|---|---|---|---|---|---|---|---|
| D-FOU | FOU-R4-k4-triton-no-materialize | 8 | 1.420512328207161 | 0.3714097706216794 | 0.4082961330186338 | 1.0002366233843059 | 1 |
| D-CHE | CHE-R2-low-degree-k3-triton | 256 | 0.9409208491986851 | 0.39200262360803795 | 0.4262586849985044 | 0.9913216258072057 | 1 |
| D-FOU | FOU-R2-low-frequency-k2-stream | 32 | 1.0071820428525156 | 0.41309484856230955 | 0.4435730513786116 | 1.0000595184953724 | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | 32 | 0.9518846020846713 | 0.41546239043721844 | 0.4486547674160997 | 0.9999703791469194 | 1 |
| D-FOU | FOU-R3-k3-triton-no-materialize | 8 | 1.3534255749882036 | 0.4167257406461612 | 0.45557572119550016 | 1.0002670940170941 | 1 |
| D-FOU | FOU-R2-low-frequency-k2-stream | 128 | 0.9398309392032267 | 0.4346114590537627 | 0.4649395463352749 | 0.9976601605307586 | 1 |
| D-CHE | CHE-R2-low-degree-k3-triton | 8 | 1.2037623055466475 | 0.4355394935418255 | 0.4712448741097888 | 1.0002670940170941 | 1 |
| D-FOU | FOU-R2-low-frequency-k2-stream | 256 | 0.9804793740865773 | 0.4538018982055999 | 0.4863937459735541 | 0.994412422067992 | 1 |
| D-CHE | CHE-R1-k3-triton-no-materialize | 8 | 1.4910767749223734 | 0.4601890076472133 | 0.5027003384302209 | 1.0002670940170941 | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | 8 | 1.1815494958186243 | 0.4625800081668563 | 0.50746797765961 | 1.0002667615152054 | 1 |
| D-CHE | CHE-R3-k4-triton-no-materialize | 32 | 1.4989063664706106 | 0.46447142337017505 | 0.5084673263958256 | 0.9998817722342092 | 1 |
| D-CHE | CHE-R1-k3-triton-no-materialize | 32 | 2.1807512419022164 | 0.48392649624031675 | 0.5447418515209199 | 0.99997034225043 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | 256 | 1.119140783336666 | 0.5039433815877659 | 0.5470264340585554 | 0.9913322632423756 | 1 |
| D-FOU | FOU-R2-low-frequency-k2-stream | 8 | 1.2268810476627063 | 0.5168352574476216 | 0.5507452893885865 | 1.0002679847546452 | 1 |
| D-FOU | FOU-R1-k2-triton-no-materialize | 32 | 1.6465444129293156 | 0.5093214828039299 | 0.5651003220543603 | 1.0000595184953724 | 1 |
| D-FOU | FOU-R1-k2-triton-no-materialize | 8 | 1.4091936189936656 | 0.5340367305159366 | 0.579003447013354 | 1.0002679847546452 | 1 |

### Continuation Repair Evidence 2026-06-03

本节记录一次按计划 Case F 继续推进的真实修复。Continuation 前的 route 备份位于 `pre_continuation_efficiency_20260603/`，当时 `D-FOU_pass=0`、`D-CHE_pass=0`，promotion blockers 为 `D-FOU_efficiency_blocked;D-CHE_efficiency_blocked;MLP_no_retained_source`。修复后重新跑完整 4GPU targeted efficiency shard，并再次 finalize。

| phase | efficiency rows | FamilyPass rows | D-FOU pass | D-CHE pass | route efficiency |
|---|---:|---:|---:|---:|---|
| before continuation | 60 | 4 | 0/12 | 0/12 | R-D-FOUForwardBlocked;R-D-CHEForwardBlocked |
| after continuation | 76 | 23 | 8/20 | 11/20 | E2-D-FOU-Pass;E2-D-CHE-Pass |

修复内容审计：

- `experiments/run_v17_common.py`：新增 `v19_basis_repair_config()`，把 D-FOU/D-CHE repair labels 接到已有真实 manual train-stream kernels，而不是只改标签；新增 k2/k3/k4 D-FOU Triton variants 和 k3/k4/gradbuf D-CHE Triton variants。
- `dgkan/profiling/efficiency_v17.py`：新增 `use_manual_ce` profiling path，v19 efficiency 使用 `manual_ce_forward_cache()` / `manual_ce_backward_from_cache()` 计量 train-stream CE path。
- 所有 pass rows 均要求 `manual_correctness_pass=1`，否则写 `ManualCorrectnessBlocked`，不允许把错误 kernel 计入 pass。

关键 pass 证据：

| family | variant | batch | manual kernel | correctness | forward | backward | step | memory |
|---|---|---:|---|---:|---:|---:|---:|---:|
| D-FOU | FOU-R2-low-frequency-k2-stream | 32 | fourier_k2_triton_l3_matmul | 1 | 1.0071820428525156 | 0.41309484856230955 | 0.4435730513786116 | 1.0000595184953724 |
| D-FOU | FOU-R2-low-frequency-k2-stream | 128 | fourier_k2_triton_l3_matmul | 1 | 0.9398309392032267 | 0.4346114590537627 | 0.4649395463352749 | 0.9976601605307586 |
| D-FOU | FOU-R4-k4-triton-no-materialize | 8 | fourier_k4_triton_l3_matmul | 1 | 1.420512328207161 | 0.3714097706216794 | 0.4082961330186338 | 1.0002366233843059 |
| D-CHE | CHE-R2-low-degree-k3-triton | 256 | cheby_k3_triton_l3_matmul | 1 | 0.9409208491986851 | 0.39200262360803795 | 0.4262586849985044 | 0.9913216258072057 |
| D-CHE | CHE-R4-k3-gradbuf-triton | 32 | cheby_k3_triton_l3_gradbuf | 1 | 0.9518846020846713 | 0.41546239043721844 | 0.4486547674160997 | 0.9999703791469194 |
| D-CHE | CHE-R3-k4-triton-no-materialize | 32 | cheby_k4_triton_l3_matmul | 1 | 1.4989063664706106 | 0.46447142337017505 | 0.5084673263958256 | 0.9998817722342092 |

Continuation GPU 证据：

- `v19_continuation_gpu_runtime_snapshots.csv` rows: 104
- max memory used MB: 695.0
- max GPU util percent: 10.0
- nonzero memory/util snapshot rows: 36

解释：basis efficiency blocker 在 D-FOU/D-CHE 上已被实质推进到 exploration pass；这不是 promotion-ready。当时的中间 route 仍然被 functional retention 卡住；后续 r4-r6 已把 functional route 更新为 `S3-MLPGenericRetainedSource;KANCarrierNotRetained`。

## Part C：Functional Update

- FunctionalRoute: S3-MLPGenericRetainedSource;KANCarrierNotRetained
- MLP retained source count: 1
- KAN M16 same-carrier retained candidates: 0
- late rebound candidate count: 2
- late rebound M2 positive rerun rows: 25

### Source / Retention Evidence Chain

| horizon | carrier | mechanism | 9-row mean source | pass_count | retention |
|---|---|---|---:|---:|---:|
| h100 | D-RBF | M1-AdamWPrimaryFUResidual | 0.0027029779222276476 | 6 |  |
| h100 | D-CHE | M1-AdamWPrimaryFUResidual | 0.0006251732508341471 | 3 |  |
| h100 | LQ | M1-AdamWPrimaryFUResidual | -0.005612346861097548 | 2 |  |
| h100 | D-WAV | M1-AdamWPrimaryFUResidual | -0.006021446651882595 | 4 |  |
| h100 | D-RAT | M1-AdamWPrimaryFUResidual | -0.006166113747490777 | 3 |  |
| h100 | D-FOU | M1-AdamWPrimaryFUResidual | -0.0066297319200303816 | 3 |  |
| h800 | MLP | M2-SGDMomentumPrimaryFU | 0.1751598914464315 | 7 |  |
| h800 | D-CHE | M1-AdamWPrimaryFUResidual | -0.002686844931708442 | 5 | 0.0 |
| h800 | D-RBF | CTRL-RecoveryOnly | -0.007917450533972846 | 0 |  |
| h800 | D-WAV | CTRL-AdamW | -0.009836514790852865 | 0 |  |
| h800 | D-CHE | CTRL-AdamW | -0.012133015526665581 | 0 |  |
| h800 | LQ | CTRL-RecoveryOnly | -0.013121240668826632 | 0 |  |
| h1600 | D-CHE | M1-AdamWPrimaryFUResidual | 0.0032123857074313695 | 7 |  |
| h1600 | D-RBF | M1-AdamWPrimaryFUResidual | -0.00665075249142117 | 4 |  |
| h1600 | D-WAV | CTRL-RecoveryOnly | -0.010287218623691134 | 0 |  |
| h1600 | LQ | CTRL-RecoveryOnly | -0.011402944723765055 | 0 |  |
| h1600 | LQ | M1-AdamWPrimaryFUResidual | -0.012305027908749051 | 4 |  |
| h1600 | D-WAV | CTRL-AdamW | -0.013830310768551297 | 0 |  |
| h3200 | D-CHE | M2-SGDMomentumPrimaryFU | 0.09768134355545044 | 5 |  |
| h3200 | MLP | CTRL-SGD | 0.0 | 0 |  |
| h3200 | MLP | M5-AlternatingFUGradient | -0.009376459651523165 | 3 |  |
| h3200 | D-RBF | M1-AdamWPrimaryFUResidual | -0.023841241995493572 | 3 |  |
| h3200 | D-RBF | CTRL-AdamW | -0.02936577796936035 | 0 |  |
| h3200 | D-WAV | M1-AdamWPrimaryFUResidual | -0.03233219517601861 | 4 |  |
| h4800 | D-CHE | M2-SGDMomentumPrimaryFU | 0.3874351978302002 | 8 | 3.9663172488026452 |
| h4800 | MLP | M5-AlternatingFUGradient | 0.029943002594841853 | 9 |  |
| h4800 | MLP | CTRL-SGD | 0.0 | 0 |  |
| h4800 | MLP | M8-PopRiskSNRFU | -0.005995114644368489 | 3 |  |
| h4800 | D-RBF | CTRL-AdamW | -0.03230463796191745 | 0 |  |
| h4800 | D-RBF | M1-AdamWPrimaryFUResidual | -0.03339049551222059 | 4 |  |

### Late Rebound Independent Rerun

| rerun | dataset | seed | mechanism | source_h3200 | source_h4800 | status |
|---|---|---|---|---|---|---|
| 0 | MNIST | 0 | M2-SGDMomentumPrimaryFU | 0.009395837783813477 | 0.4434763193130493 | measured |
| 0 | MNIST | 0 | CTRL-RandomMatchedNorm | -0.4311414957046509 | -0.28820693492889404 | measured |
| 0 | MNIST | 1 | CTRL-SGD | -1.1693352460861206 | -1.1028331518173218 | measured |
| 0 | MNIST | 2 | M2-SGDMomentumPrimaryFU | -0.48075318336486816 | -0.16143417358398438 | measured |
| 0 | MNIST | 2 | CTRL-RandomMatchedNorm | -1.227831482887268 | -1.1724306344985962 | measured |
| 0 | Fashion-MNIST | 0 | CTRL-SGD | -0.7734733819961548 | -0.6137350797653198 | measured |
| 0 | Fashion-MNIST | 1 | M2-SGDMomentumPrimaryFU | 0.2445605993270874 | 0.5340249538421631 | measured |
| 0 | Fashion-MNIST | 1 | CTRL-RandomMatchedNorm | -0.5918283462524414 | -0.45268523693084717 | measured |
| 0 | Fashion-MNIST | 2 | CTRL-SGD | -0.654494047164917 | -0.48621630668640137 | measured |
| 0 | KMNIST | 0 | M2-SGDMomentumPrimaryFU | 0.052178382873535156 | 0.37123847007751465 | measured |
| 0 | KMNIST | 0 | CTRL-RandomMatchedNorm | -0.483925461769104 | -0.37487661838531494 | measured |
| 0 | KMNIST | 1 | CTRL-SGD | -0.4022338390350342 | -0.26830124855041504 | measured |
| 0 | KMNIST | 2 | M2-SGDMomentumPrimaryFU | 0.22793900966644287 | 0.5139745473861694 | measured |
| 0 | KMNIST | 2 | CTRL-RandomMatchedNorm | -0.2658047676086426 | -0.14696860313415527 | measured |
| 1 | MNIST | 0 | CTRL-SGD | -0.3679847717285156 | -0.2237241268157959 | measured |
| 1 | MNIST | 1 | M2-SGDMomentumPrimaryFU | -0.37555456161499023 | 0.026356935501098633 | measured |
| 1 | MNIST | 1 | CTRL-RandomMatchedNorm | -1.103416085243225 | -1.0260220766067505 | measured |
| 1 | MNIST | 2 | CTRL-SGD | -0.3356497287750244 | -0.2101755142211914 | measured |

### Functional Continuation：MLP Source Retention Case B

Basis efficiency 修复后，剩余 blocker 是 `MLP_no_retained_source`。按计划 Case B / FU-H1 / FU-H2，本次追加 MLP-only targeted continuation：FU scale sweep、sparse alternating pulse、dual-memory slow-state、schedule-free、PopRisk slow-state，以及 train-split LineC-filtered alternating pulse（M15）。这不是全矩阵替代，而是对剩余 blocker 的定向修复尝试。

| item | value |
|---|---:|
| continuation measured rows | 162/162 |
| continuation grouped probes | 15 |
| retained candidate count | 0 |
| round1 GPU rows / max util | 940 / 19.0 |
| round2 GPU rows / max util | 224 / 10.0 |
| round3 GPU rows / max util | 124 / 21.0 |

| continuation_id | mechanism | h800_mean | h1600_mean | h3200_mean | h4800_mean | retention_h1600/h800 | retention_h3200/h1600 | retained |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M15-linec-alt50-fu0p0005 | M15-LineCFilteredAlternatingFU | -0.0853961706161499 | 0.009985950258043077 | 0.09047187036938137 | 0.12667612234751383 |  | 9.05991598511236 | 0 |
| M5-alt200-fu0p0005 | M5-AlternatingFUGradient | -0.0819930632909139 | 0.03840446472167969 | 0.0987200074725681 | 0.11526542901992798 |  | 2.570534654967856 | 0 |
| M15-linec-alt100-fu0p0005 | M15-LineCFilteredAlternatingFU | -0.047544280687967934 | 0.07448791133032905 | 0.08008613851335314 | 0.054555667771233454 |  | 1.0751561841786357 | 0 |
| M2-fu0p0001 | M2-SGDMomentumPrimaryFU | 0.2674288551012675 | -0.14676260948181152 | -0.37322290738423664 | -0.3330954180823432 | 0.0 |  | 0 |
| M2-fu0p00005 | M2-SGDMomentumPrimaryFU | 0.17987990379333496 | -0.23004455036587185 | -0.47210097312927246 | -0.4277762042151557 | 0.0 |  | 0 |

M15 filter readback:

| continuation_id | final rows | filter trials per row | mean accept rate |
|---|---:|---|---:|
| M15-linec-alt50-fu0p0005 | 9 | 96 | 0.2696759259259259 |
| M15-linec-alt100-fu0p0005 | 9 | 48 | 0.21296296296296294 |

解释：

- M2 降低 FU scale 后 h800 grouped source 可以为正，但 h1600/h3200/h4800 仍 washout，因此不能支持 AdamW-washout-only 解释。
- 稀疏 M5 与 M15 LineC filter 能让 h1600/h3200/h4800 grouped source 转正，但 h800 grouped source 仍为负；这属于 phase mismatch / delayed migration，不满足 S3 retained source 公式。
- 截至 r3，v19 functional blocker 没有被修复；这触发了后续 r4 的 M16 two-phase writer 和 r5/r6 的 same-carrier KAN check。r3 的 delayed positives 不写成 promotion。

### Functional Continuation r4-r6：M16 Two-Phase Writer + Case C KAN Check

r4 按计划 Case B / FU-H2 尝试 M16：先用 M2-style momentum+FU warmup 保住 h800 source，再切到 LineC-filtered sparse pulse 尝试 h1600/h3200/h4800 retention。r5/r6 按计划 Case C 对 D-CHE/D-FOU 跑 same-carrier M16 与 matched controls，避免把 MLP generic source 写成 KAN-specific。

| item | value |
|---|---:|
| combined continuation matrix rows | 324 |
| combined grouped probe rows | 21 |
| retained candidate count | 1 |
| r4 MLP M16 rows / runtime_sum_sec | 36 / 211.611 |
| r5 D-CHE/D-FOU M16 rows / runtime_sum_sec | 18 / 174.554 |
| r6 D-CHE/D-FOU controls rows / runtime_sum_sec | 90 / 857.819 |
| r4 GPU rows / max util | 72 / 22.0 |
| r5 GPU rows / max util | 68 / 10.0 |
| r6 GPU rows / max util | 200 / 15.0 |

| carrier | continuation_id | h800_mean | h800_pass | h1600_mean | h1600_pass | h3200_mean | h3200_pass | h4800_mean | h4800_pass | retained |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | M16-warm800-alt50-fu0p0001 | 0.01937996016608344 | 5 | 0.0803578429751926 | 6 | 0.07967897256215413 | 6 | 0.07385419474707709 | 6 | 1 |
| MLP | M16-warm800-alt100-fu0p00005 | -0.09289987881978352 | 2 | 0.03803565767076281 | 5 | 0.11519361866845025 | 5 | 0.14795619249343872 | 6 | 0 |
| D-CHE | M16-warm800-alt50-fu0p0001 | -0.8855499625205994 | 0 | -0.77398638592826 | 0 | -0.6744108001391093 | 0 | -0.5914939045906067 | 1 | 0 |
| D-FOU | M16-warm800-alt50-fu0p0001 | -0.9329363306363424 | 0 | -0.8675869637065463 | 0 | -0.7608865963088142 | 0 | -0.6862555742263794 | 0 | 0 |

M16 retained row 的 retention chain：

| carrier | continuation_id | retention_h1600/h800 | retention_h3200/h1600 | retention_h4800/h3200 |
|---|---|---:|---:|---:|
| MLP | M16-warm800-alt50-fu0p0001 | 4.14644004871721 | 0.9915519084646405 | 0.9268969261553495 |

M16 r4 per-row 证据显示这是 grouped generic MLP source，不是单行 max：9 个 dataset/seed rows 中 h800 pass 5 个、h1600 pass 6 个、h3200 pass 6 个、h4800 pass 6 个。r5 初跑时 controls 没有按 mechanism 名匹配到 spec，D-CHE/D-FOU source 暂时 blank；已修复 `selected_specs()`，r6 真实补跑 90 条 D-CHE/D-FOU same-carrier controls 后，D-CHE/D-FOU M16 source 均为强负，不能写 KAN-specific。

结论：MLP source washout 这个 blocker 已被 M16 partially solved，形成 `MLPGenericRetainedSource`。但 v19 总目标没有达成，因为同一 M16 writer 在 D-CHE/D-FOU carrier 上没有 retained source，S4/S5 未执行，E2 efficiency 也还不是 E3 official/no-materialize gate。

### Functional Continuation r7-r10：KAN Carrier / H3 Block Repair Attempts

r4-r6 之后的关键问题是：MLP 的 M16 retained source 能否转移到 KAN carrier。按计划 Case C 与 FU-H3，继续尝试 readout carrier、repaired basis variant、low-rank block FU、split-consensus source estimator。所有 r7-r10 rows 都是 fresh measured rows。

| item | value |
|---|---:|
| final continuation matrix rows | 666 |
| final continuation grouped rows | 47 |
| final continuation trace rows | 6660 |
| retained candidate count | 1 |
| non-MLP retained candidates | 0 |
| r7 M17 R0 rows / runtime_sum_sec | 72 / 685.542 |
| r8 repaired M16/M17 rows / runtime_sum_sec | 144 / 1577.922 |
| r9 repaired M18 rows+same-rank control / runtime_sum_sec | 72 / 1399.141 |
| r10 repaired M19 rows / runtime_sum_sec | 54 / 2243.965 |

#### r7：M17 Readout-Carrier R0-current

| carrier | continuation_id | h800_mean | h1600_mean | h3200_mean | h4800_mean | retained |
|---|---|---:|---:|---:|---:|---:|
| D-CHE | M17-readout-warm800-alt50-fu0p00005 | -0.8798607918951247 | -0.7707849277390374 | -0.6736067467265658 | -0.5923082100020515 | 0 |
| D-CHE | M17-readout-warm800-alt50-fu0p0001 | -0.8861113588015238 | -0.7754378782378303 | -0.6779623892572191 | -0.5985392795668708 | 0 |
| D-FOU | M17-readout-warm1200-alt50-fu0p0001 | -0.9278650614950392 | -0.8628456658787198 | -0.7565849158498976 | -0.6821578741073608 | 0 |
| D-FOU | M17-readout-warm800-alt50-fu0p00005 | -0.9287947614987692 | -0.863005088435279 | -0.7552309632301331 | -0.6789163880878024 | 0 |

解释：readout-only carrier actuator 没有解决 D-CHE/D-FOU 不承载的问题；所有 grouped source 仍为强负。

#### r8：Repaired Basis Variant 上的 M16/M17

| carrier | basis_repair_variant | continuation_id | h800_mean | h1600_mean | h3200_mean | h4800_mean | retained |
|---|---|---|---:|---:|---:|---:|---:|
| D-CHE | CHE-R2-low-degree-k3-triton | M16-warm800-alt50-fu0p0001 | -1.0616207718849182 | -0.9420253104633756 | -0.8146882189644707 | -0.7346266110738119 | 0 |
| D-CHE | CHE-R2-low-degree-k3-triton | M17-readout-warm800-alt50-fu0p00005 | -1.0636576877699957 | -0.9427765938970778 | -0.8138382567299737 | -0.7345359060499403 | 0 |
| D-CHE | CHE-R2-low-degree-k3-triton | M17-readout-warm800-alt50-fu0p0001 | -1.0580391950077481 | -0.939201639758216 | -0.8131394783655802 | -0.7336875597635905 | 0 |
| D-FOU | FOU-R2-low-frequency-k2-stream | M16-warm800-alt50-fu0p0001 | -1.0181440247429743 | -0.9058648016717699 | -0.7325711912579007 | -0.6037499838405185 | 0 |
| D-FOU | FOU-R2-low-frequency-k2-stream | M17-readout-warm800-alt50-fu0p00005 | -1.0166224903530545 | -0.9038515686988831 | -0.7292676899168227 | -0.5987619757652283 | 0 |
| D-FOU | FOU-R2-low-frequency-k2-stream | M17-readout-warm800-alt50-fu0p0001 | -1.0176472663879395 | -0.9052479598257277 | -0.7315973308351305 | -0.6022730337248908 | 0 |

解释：把 M16/M17 接到 E2 repaired basis variant 后，source 没有改善，反而仍显著落后 same-variant controls。因此不能说“只是 R0 basis carrier 太慢导致 functional negative”。

#### r9：M18 Low-Rank Carrier Block FU + Same-Rank Control

| carrier | basis_repair_variant | continuation_id | h800_mean | h1600_mean | h3200_mean | h4800_mean | retained |
|---|---|---|---:|---:|---:|---:|---:|
| D-CHE | CHE-R2-low-degree-k3-triton | M18-lowrank-r4-fu0p00005 | -1.0577672123908997 | -0.9210190176963806 | -0.7669605811436971 | -0.6732859876420763 | 0 |
| D-CHE | CHE-R2-low-degree-k3-triton | M18-lowrank-r4-fu0p0001 | -1.0578158232900832 | -0.9212367468410068 | -0.7676091856426663 | -0.674410793516371 | 0 |
| D-CHE | CHE-R2-low-degree-k3-triton | M18-lowrank-r4-fu0p00025 | -1.0474738346205816 | -0.9106764992078146 | -0.7573481533262465 | -0.6651304297977023 | 0 |
| D-FOU | FOU-R2-low-frequency-k2-stream | M18-lowrank-r4-fu0p00005 | -1.0207952128516302 | -0.9114851752916971 | -0.7443431086010404 | -0.6224512192938063 | 0 |
| D-FOU | FOU-R2-low-frequency-k2-stream | M18-lowrank-r4-fu0p0001 | -1.0193469524383545 | -0.9099971784485711 | -0.7427757713529799 | -0.6208034555117289 | 0 |
| D-FOU | FOU-R2-low-frequency-k2-stream | M18-lowrank-r4-fu0p00025 | -1.0210011535220676 | -0.9114890164799161 | -0.7439452409744263 | -0.6216527819633484 | 0 |

Same-rank control readback：`CTRL-RandomSameRankBlock` r9 rows=18, mean final val loss=2.311803；`M18-CarrierLowRankBlockFU` r9 rows=54, mean final val loss=2.309915。这个 mean final loss 小差异没有形成 source retention；不能替代 source/control gate。

#### r10：M19 Split-Consensus Low-Rank Carrier Block

| carrier | basis_repair_variant | continuation_id | h800_mean | h1600_mean | h3200_mean | h4800_mean | retained |
|---|---|---|---:|---:|---:|---:|---:|
| D-CHE | CHE-R2-low-degree-k3-triton | M19-split-lowrank-r4-fu0p00005 | -1.0515826344490051 | -0.9150750835736593 | -0.7614943848715888 | -0.6682206524742974 | 0 |
| D-CHE | CHE-R2-low-degree-k3-triton | M19-split-lowrank-r4-fu0p0001 | -1.0672275291548834 | -0.9302558832698398 | -0.7758191823959351 | -0.6818773216671414 | 0 |
| D-CHE | CHE-R2-low-degree-k3-triton | M19-split-lowrank-r4-fu0p00025 | -1.0461820562680562 | -0.9094735185305277 | -0.7561592393451266 | -0.6638793415493436 | 0 |
| D-FOU | FOU-R2-low-frequency-k2-stream | M19-split-lowrank-r4-fu0p00005 | -1.0198686122894287 | -0.9105627338091532 | -0.743428905804952 | -0.621544725365109 | 0 |
| D-FOU | FOU-R2-low-frequency-k2-stream | M19-split-lowrank-r4-fu0p0001 | -1.017775085237291 | -0.9084170990520053 | -0.7411789496739706 | -0.6191913750436571 | 0 |
| D-FOU | FOU-R2-low-frequency-k2-stream | M19-split-lowrank-r4-fu0p00025 | -1.020502143436008 | -0.9109980331526862 | -0.7434708144929674 | -0.6211906472841898 | 0 |

解释：split-consensus 作为 source estimator 也没有让 KAN repaired carrier 承载 MLP M16 那种 retained source。至此，已经真实尝试过 M16 trajectory writer、M17 readout carrier、M18 matrix/block FU、M19 split-consensus block FU。下一步不能再靠“新增 token + 小变体”推进，必须进入计划 H4 的真实 function-space projection / actuation solver，并首先测 ActuationR2；下面的 M20 诊断正是这一修复尝试。

### H4 Function-Space Actuation Solver Diagnostic

按计划 FU-H4，本次继续实现真实 train-split function-space actuation solver `M20-TrainSplitFunctionSpaceActuationFU`。它使用当前 train batch 切分 B1/B2/B3，构造候选参数子空间 `U`，有限差分测 `J_U` 对 B1/B2 logits 的作用，解 ridge least-squares，再用 B3 做 safety readback。方向源不读取 validation/test/future。

计划要求：如果 `ActuationR2 < 0.20`，不跑 long horizon，先修 basis-channel U 或 projection solver。本轮正是这个情况。

| repair_round | step | ActuationR2 | ActuationCosine | B1_gain | B2_transfer_gain | B3_safety_gain | operator_rank | cap_ratio | clamp_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1_initial_fd_normed | 1 | 0.0002904534339904785 | 0.28889673948287964 | 0.00012040138244628906 | 0.00014448165893554688 | 6.532669067382812e-05 | 4 |  |  |
| v2_commit_scale_cap100 | 1 | 0.0028815865516662598 | 0.2888900637626648 | 0.0012021064758300781 | 0.0014448165893554688 | 0.0006506443023681641 | 4 |  | 0.017415949314783342 |
| v3_class_readout_basis | 1 | 0.0010082721710205078 | 0.3072971701622009 | 0.0004444122314453125 | 0.0004787445068359375 | 0.0002353191375732422 | 12 |  | 0.005682770936224506 |
| v4_cap_sweep_3000 | 1 | 0.02784097194671631 | 0.3071233034133911 | 0.013219118118286133 | 0.014225244522094727 | 0.006964445114135742 | 12 | 3000.0 | 0.1704831204766915 |
| v5_top_entry_basis | 1 | 0.02713632583618164 | 0.3079710304737091 | 0.012877702713012695 | 0.013801336288452148 | 0.006781339645385742 | 36 | 3000.0 | 0.1653862278801742 |
| v6_cap_sweep_30000 | 1 | 0.09360003471374512 | 0.30666324496269226 | 0.07385778427124023 | 0.07891631126403809 | 0.03797030448913574 | 36 | 30000.0 | 1.0 |

证据文件：`results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_h4_actuation_solver_smoke.csv`。

解释：

- H4 solver 不是空壳：B1/B2/B3 gain 都是真实 train split readback，best row 的 B1/B2/B3 均为正。
- 但 best ActuationR2 只有 0.09360003471374512，低于计划阈值 0.20；projection_residual_norm 仍约 0.952，说明 function-space displacement 没有足够拟合 target displacement。
- 因此没有执行 H4 long horizon。这不是放弃，而是按计划的 blocker 处理：先修 exact basis-channel JVP/VJP 或 conjugate-gradient projection，再进入 h800/h1600/h3200。

### Control / KAN-vs-MLP Attribution

以下 attribution 来自原 v19 full horizon/source matrix；r4-r6 continuation 的 same-carrier M16 attribution 已在上一节单独列出，不能混用行数。

| metric | value |
|---|---:|
| control rows explaining positive | 130/133 |
| KAN advantage h1600 rows | 20/133 |
| KAN advantage h3200 rows | 37/133 |
| KAN advantage h4800 rows | 0/133 |

### Debt Evidence

- incomplete debt rows: 0/1197
- peak=0 时 recovery 保持空值/undefined；缺失 metric 写 EvidenceIncomplete；这些都不能解释为成功。

| carrier | mechanism | dataset | seed | tail_peak | NLL_peak | ECE_peak | LineC_channel | status |
|---|---|---|---|---|---|---|---|---|
| D-CHE | CTRL-AdamW | Fashion-MNIST | 0 | 5.894970417022705 | 0.0475081205368042 | 0.1971922293305397 | 1.0 | measured |
| D-CHE | CTRL-AdamW | Fashion-MNIST | 1 | 3.239367961883545 | 0.015760421752929688 | 0.1750531792640686 | 0.8886887024118776 | measured |
| D-CHE | CTRL-AdamW | Fashion-MNIST | 2 | 6.679617166519165 | 0.5009264945983887 | 0.38954174518585205 | 0.22697845540023753 | measured |
| D-CHE | CTRL-AdamW | KMNIST | 0 | 7.060066223144531 | 0.5938320159912109 | 0.36451514065265656 | 0.9446410037454099 | measured |
| D-CHE | CTRL-AdamW | KMNIST | 1 | 5.113059759140015 | 0.037203311920166016 | 0.28100302070379257 | 1.0 | measured |
| D-CHE | CTRL-AdamW | KMNIST | 2 | 4.215216875076294 | 0.03197193145751953 | 0.2958635091781616 | 0.9815584772852906 | measured |
| D-CHE | CTRL-AdamW | MNIST | 0 | 4.1684534549713135 | 0.05719280242919922 | 0.27480920404195786 | 7.044032024339231e-12 | measured |
| D-CHE | CTRL-AdamW | MNIST | 1 | 5.513898134231567 | 0.12456631660461426 | 0.30985260009765625 | 0.9998067272416907 | measured |
| D-CHE | CTRL-AdamW | MNIST | 2 | 5.235038757324219 | 0.05655860900878906 | 0.22834034264087677 | 4.457656466172466e-12 | measured |
| D-CHE | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 0 | 0.19170331954956055 | 1.4691972732543945 | 0.028568662703037262 | 0.9337871775979837 | measured |
| D-CHE | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 1 | 0.11965751647949219 | 1.5769178867340088 | 0.061725493520498276 | 0.8886887024118776 | measured |
| D-CHE | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 2 | 0.060494184494018555 | 0.7497104406356812 | 0.0807995107024908 | 0.3853511341990664 | measured |
| D-CHE | CTRL-NoOpMatchedOverhead | KMNIST | 0 | 2.4080276489257812e-05 | 0.4305349588394165 | 0.07555469498038292 | 0.950534766101401 | measured |
| D-CHE | CTRL-NoOpMatchedOverhead | KMNIST | 1 | 0.04253244400024414 | 0.7363955974578857 | 6.36950135231018e-05 | 0.15889669264271977 | measured |
| D-CHE | CTRL-NoOpMatchedOverhead | KMNIST | 2 | 0.00013113021850585938 | 0.7998999357223511 | 0.0007214993238449097 | 0.9815584772852906 | measured |
| D-CHE | CTRL-NoOpMatchedOverhead | MNIST | 0 | 2.4557113647460938e-05 | 1.0071287155151367 | 0.013421513140201569 | 6.6519012520416254e-12 | measured |

## 4GPU / Queue 证据

- 下列总 GPU snapshot 为 v19 base/efficiency/horizon/late-rerun 主流程；r4-r6 continuation GPU 证据单独列在后面的 Functional Continuation Runtime。
- GPU snapshot rows: 4756
- max memory used MB: 759.0
- max util percent: 48.0
- queue drain report: `/home/chengshun.wang/DG-LCA/results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_queue_drain_report.csv`

### Horizon Shard Runtime

| shard | rows | runtime_sum_sec |
|---|---|---|
| h0 | 300 | 8013.44 |
| h1 | 299 | 8274.26 |
| h2 | 299 | 8586.17 |
| h3 | 299 | 8384.68 |

### Late Rebound Rerun Runtime

| shard | rows | runtime_sum_sec |
|---|---|---|
| r0 | 41 | 394.11 |
| r1 | 41 | 397.62 |
| r2 | 40 | 378.08 |
| r3 | 40 | 388.56 |

### Functional Continuation Runtime

| round | rows | runtime_sum_sec | gpu_snapshot_rows | max_mem_mb | max_util |
|---|---:|---:|---:|---:|---:|
| r4 MLP M16 | 36 | 211.611 | 72 | 693.0 | 22.0 |
| r5 D-CHE/D-FOU M16 | 18 | 174.554 | 68 | 693.0 | 10.0 |
| r6 D-CHE/D-FOU controls | 90 | 857.819 | 200 | 693.0 | 15.0 |
| r7 D-CHE/D-FOU M17 R0 | 72 | 685.542 | 168 | 693.0 | 10.0 |
| r8 repaired M16/M17 | 144 | 1577.922 | 364 | 693.0 | 20.0 |
| r9 repaired M18 + same-rank control | 72 | 1399.141 | 360 | 759.0 | 54.0 |
| r10 repaired M19 split-consensus | 54 | 2243.965 | 508 | 759.0 | 36.0 |

## 修改记录（便于审计）

- 修改 `dgkan/metrics/linec.py`：新增 LineC-channel train-split trajectory audit 与 golden tests；修复 C7 scale invariance 为通道改善向量缩放不变性测试。
- 修改 `dgkan/fu/mechanisms.py`：新增 M11 dual-memory slow-state、M12 schedule-free averaged FU、M13 low-rank matrix-block FU、M14 source-channel PopRisk/SNR slow FU，并新增 mechanism semantic contract rows。
- 修改 `dgkan/profiling/efficiency_v17.py`：拆分 SGD/AdamW/manual-FU update ms、basis eval/readout/backward/update waterfall 字段。
- 修改 `experiments/run_v17_common.py`：新增 v19 S0.3、h2400/h4800 trace/readback、CEp99/NLL/ECE/Brier/LineC-fast/LineC-channel/AUCtime debt、basis repair variants、v19 artifact/packet/bundle/docs/finalizer。
- 新增 v19 wrappers：`run_v19_s03_truth_gate.py`、`run_v19_source_channel_fu_matrix.py`、`run_v19_basis_kernel_breakthrough.py`、`run_v19_late_rebound_rerun.py`、`run_v19_merge_finalize.py`。
- Continuation 修复 `dgkan/profiling/efficiency_v17.py`：新增 manual CE train-stream profiling，v19 efficiency rows 使用 `manual_ce_forward_cache()` / `manual_ce_backward_from_cache()` 计量 D-FOU/D-CHE fused path。
- Continuation 修复 `experiments/run_v17_common.py`：`v19_basis_repair_config()` 将 repair variants 接到 `fourier_k2/k3/k4_triton_l3_matmul`、`cheby_k3/k4_triton_l3_matmul`、`cheby_k3_triton_l3_gradbuf`；同时新增 `manual_correctness_pass` guard，防止错误 kernel 被记为 efficiency pass。
- Functional continuation 新增 `experiments/run_v19_functional_continuation.py`：MLP-only FU scale / sparse pulse / slow-state / signal-channel filter runner，输出 continuation matrix/traces/summary。
- Functional continuation 修改 `dgkan/fu/mechanisms.py` 与 `experiments/run_v17_common.py`：新增 `M15-LineCFilteredAlternatingFU`，用 train split A/B 的临时 CE improvement 与 LineC leak 过滤 pulse commit，trace 记录 filter accept 证据。
- Functional continuation r4 修改 `dgkan/fu/mechanisms.py` 与 `experiments/run_v17_common.py`：新增 `M16-TwoPhaseMomentumThenLineCFU`，先用 M2 momentum+FU warmup，再用 LineC-filtered alternating pulse 做 sparse retention writer。
- Functional continuation r4-r6 修改 `experiments/run_v19_functional_continuation.py`：新增 `--source-warmup-steps`、M16 specs、`--carriers`、same-carrier control source 计算；修复 `--spec-ids` 过滤，让 control mechanism 名如 `CTRL-AdamW` 能匹配，避免 controls 漏跑导致 source blank。
- Functional continuation r7-r10 修改 `dgkan/fu/mechanisms.py`：新增 `M17-ReadoutCarrierTwoPhaseLineCFU`、`M18-CarrierLowRankBlockFU`、`M19-SplitConsensusLowRankBlockFU` 与 `CTRL-RandomSameRankBlock`；M19 只用 train-batch split consensus，不读 validation/test/future。
- Functional continuation r7-r10 修改 `experiments/run_v19_functional_continuation.py`：新增 `--basis-repair-variant`，summary 改为 `carrier+basis_repair_variant+continuation_id` 分组，controls baseline 也包含 `basis_repair_variant`，防止 repaired variant 与 R0 controls 混用。
- H4 continuation 修改 `dgkan/fu/core.py`：`UpdateTensor` 增加可选 diagnostics，用于审计 function-space solver。
- H4 continuation 修改 `dgkan/fu/mechanisms.py`：新增 `M20-TrainSplitFunctionSpaceActuationFU`，并逐步修复 commit-scale solve、readout-class U、cap sweep、top-entry U、30000 cap sweep；所有方向只来自 train split。
- H4 continuation 修改 `experiments/run_v17_common.py`：trace/final row 写入 ActuationR2、ActuationCosine、B1/B2/B3 gain、projection residual、operator rank/cap/clamp/commit scale/status。
- H4 continuation 新增 official diagnostic artifact：`v19_h4_actuation_solver_smoke.csv`；由于 best ActuationR2=0.09360003471374512 < 0.20，未执行 long horizon。
- 新增 continuation 审计包：`results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19/v19_continuation_code_review_packet.zip`，包含本轮 M16-M20/r4-r10/H4 smoke 相关代码、两份日志、route、next queue 与 sha256 manifest。

## 分析 / Insight / 结论

- v19 的第一原则是 evidence-first：只要 S0.3 debt/LineC/mechanism contract 不完整，就不能把后续 no-go 写成科学结论。
- MLP source 若跨 horizon 留存，按 generic training dynamics insight 写；只有 KAN row 本身 source 为正、retention 为正、controls 不能解释、debt recovery 完整，才可写 KAN-specific。
- D-CHE/M2 h3200 late rebound 必须通过独立 rerun 才能升级为候选；前一 horizon 非正时不能把后续转正写成 retained source。
- Basis efficiency 经过 continuation 后，D-FOU/D-CHE 从 blocked 变为 E2 pass：D-FOU 8/20 pass，D-CHE 11/20 pass；这说明计划推荐的 low-frequency/recurrent/no-materialize train-stream repair 方向有效。
- 但 basis efficiency pass 不等于 v19 总目标达成；r4 之后 `MLP retained source count=1`，但它是 generic MLP dynamics，不是 KAN-specific。D-CHE/D-FOU same-carrier M16 source 均为负，因此 route 仍不能 promotion。
- Functional continuation 进一步说明 blocker 的形态：M2 小 FU scale 可以 h800 positive 但 h1600 washout；M5/M15 稀疏或 LineC-filtered pulse 可以 h1600-h4800 positive 但 h800 negative；M16 two-phase writer 能把 MLP h800 和后续 horizons 接起来。当前可写结论是“MLP generic retained source 存在”，不是 KAN breakthrough。
- r5 的 controls 漏跑是一个实际 execution blocker：`--spec-ids CTRL-AdamW` 没匹配 `continuation_id=CTRL-CTRL-AdamW`。已修复为同时匹配 mechanism，并用 r6 90 条 controls 补齐 same-carrier source 证据。
- r7-r10 的结果进一步缩小 blocker：KAN carrier 不只是缺 readout-only actuator，也不是简单 E2 repaired variant、rank-4 block update 或 split-consensus block source 可以解决。后续需要真实 H4 function-space projection/actuation solver，先测 ActuationR2，而不是继续新增同类机制名。
- H4 M20 的诊断显示局部 train split CE gain 是存在的，但 ActuationR2 不足，说明问题已经从“有没有局部下降方向”推进到“basis-channel projection 能否精确实现目标 function displacement”。这是一个更窄的工程/science blocker。
- 显存低不等于 gate 通过，forward/basis-eval/step ratio 与 manual correctness guard 才是当前 basis repair 的核心证据链。
- `promotion_allowed=0` 是默认审计保护：只有 official S4/S5、debt complete、functional retained source、official/no-materialize efficiency gate 同时真实通过才可翻转。

## 2026-06-03 H4/M21-M26 追加复盘

### 是否达成 v19 目标

没有达成。

- 当前 route: `R-MLPGenericRetained-KANCarrierBlocked`
- FunctionalRoute: `S3-MLPGenericRetainedSource;KANCarrierLatePositiveButNotRetained`
- promotion_allowed: 0
- functional_continuation_rows: 774
- functional_continuation_grouped_probes: 59
- functional_continuation_retained_candidates: 1
- kan_repaired_m20_m26_retained_candidates: 0

唯一 retained candidate 仍是 MLP generic dynamics：

| carrier | mechanism/spec | h800 | h1600 | h3200 | h4800 |
|---|---|---:|---:|---:|---:|
| MLP | M16-warm800-alt50-fu0p0001 | 0.01937996016608344 | 0.0803578429751926 | 0.07967897256215413 | 0.07385419474707709 |

这不能写成 KAN breakthrough，因为 carrier 是 MLP，不是 D-CHE/D-FOU/D-RBF/D-RAT/D-WAV/LQ。

### 新增修复与审计边界

本次不是只整理旧表，而是继续按计划方向修改并运行：

- 新增 `M21-ExactReadoutFunctionSpaceActuationFU`：直接在 frozen readout feature space 解 ridge least-squares function displacement。
- 新增 `M22-ExactReadoutHighCapScheduledFU`：提高 cap，按 schedule sparse commit。
- 新增 `M23-ExactReadoutUltraCapScheduledFU`：进一步提高 cap，验证 projection 能否接近 exact actuation。
- 新增 `M24-WarmupExactReadoutUltraCapFU`：先 SGD warmup，再 exact-readout pulse。
- 新增 `M25-AdamWExactReadoutUltraCapFU`：AdamW 主训练 + exact-readout residual。
- 新增 `M26-GatedAdamWExactReadoutFU`：只有 ActuationR2/B2/B3/displacement gate 同时通过才 commit residual。
- 修复 M20-M23 `alt_period` 没被实际使用的 blocker；之前 unscheduled M21 smoke 发现 operator 每步 commit，这会把 sparse writer 的计划语义冲掉。
- trace 现在写入 `ActuationR2`、`ActuationCosine`、`B1_gain`、`B2_transfer_gain`、`B3_safety_gain`、`operator_gate_accept`，便于审计为什么某个 pulse 被接受/拒绝。

所有新增方向仍只使用 train split/batch 信息；没有读取 validation/test/future labels。

### Actuation Solver 证据

M20 原始 H4 solver 最佳 ActuationR2 只有 0.09360003471374512；M21-M26 后，projection/actuation 本身被显著修好：

| mechanism | rows | ActuationR2 min | ActuationR2 max | ActuationR2 mean | gate |
|---|---:|---:|---:|---:|---|
| M21 | 165 | 0.014818370342254639 | 0.9659602046012878 | 0.6294926368828976 | n/a |
| M22 | 130 | 0.08726805448532104 | 0.9796541929244995 | 0.7868888680751507 | n/a |
| M23 | 274 | 0.2956928014755249 | 0.9839740991592407 | 0.8283174161928414 | n/a |
| M24 | 3 | 0.7349652051925659 | 0.8385844826698303 | 0.80373615026474 | n/a |
| M25 | 138 | 0.5108537673950195 | 0.9999836683273315 | 0.97268272396447 | n/a |
| M26 | 130 | 0.43791741132736206 | 0.9999852180480957 | 0.969033406330989 | 41/130 accepted |

关键 insight：ActuationR2 被修到接近 1，不等于 source retention 成功。projection 能够实现 train-batch/readout-space displacement，但 grouped real-data horizon 上仍没有让 KAN carrier h800/h1600/h3200/h4800 全链路转正。

### Smoke 到 Full Run 的证据链

| probe | smoke result | full-run decision |
|---|---|---|
| M22 high-cap | step20 ActuationR2=0.3892034888267517 | 进入 full follow-up |
| M23 ultra-cap | step20 ActuationR2=0.856488049030304 | 进入 full follow-up |
| M22 alt50 | D-CHE MNIST seed0 h800=0.06744098663330078 | full 9-row h800 均值转负 |
| M23 alt50 | D-CHE MNIST seed0 h800=0.03887653350830078 | full 9-row 只形成 late positive |
| M24 warm400 | h800=-0.09970617294311523, h1600=0.22084546089172363 | h800 负，不进入 retained |
| M25 alt100 | h800=0.07347774505615234, h1600=0.14383041858673096 | full 9-row h800/h1600 均值转负 |
| M26 gated | h800=0.11826753616333008, h1600=0.20043909549713135 | full 9-row h800/h1600 均值转负 |

这说明单个 MNIST seed0 smoke positive 不是 stable source；必须看 3 dataset x 3 seed 的 grouped mean 与 retention。

### Full 9-row Results

| carrier | spec | h800 mean | h1600 mean | h3200 mean | h4800 mean | retained |
|---|---|---:|---:|---:|---:|---:|
| D-CHE | M21 | -1.7059106760554843 | -1.9571839769681294 | -2.190653827455309 | -2.323306428061591 | 0 |
| D-FOU | M21 | -2.8070288366741605 | -3.2429449359575906 | -3.6243044667773776 | -3.8307798902193704 | 0 |
| D-CHE | M23 | -0.3645904196633233 | -0.37781767050425213 | -0.367608024014367 | -0.35427424642774796 | 0 |
| D-FOU | M23 | -0.7186367909113566 | -0.7717924184269376 | -0.7617345386081271 | -0.7259222070376078 | 0 |
| D-CHE | M22 alt50 | -0.1703893542289734 | -0.06318890386157566 | -0.028217554092407227 | -0.005586928791469998 | 0 |
| D-CHE | M23 alt50 | -0.07936050494511922 | 0.0037816431787278918 | 0.03331817520989312 | 0.05557106600867377 | 0 |
| D-FOU | M22 alt50 | -0.22767482863532174 | -0.13538656632105509 | -0.0811925729115804 | -0.030357791317833796 | 0 |
| D-FOU | M23 alt50 | -0.22098304828008017 | -0.22261044051912096 | -0.19528800911373562 | -0.15801193979051378 | 0 |
| D-CHE | M25 alt100 | -0.03924524121814304 | -0.0040361351437038845 | 0.0047302643458048505 | -0.013945202032725016 | 0 |
| D-FOU | M25 alt100 | -0.10326227876875135 | -0.11500602960586548 | -0.10978075530793932 | -0.11714468399683635 | 0 |
| D-CHE | M26 gated alt100 | -0.06443260113398235 | -0.028240985340542264 | -0.016286081737942167 | -0.03339660167694092 | 0 |
| D-FOU | M26 gated alt100 | -0.07220927874247234 | -0.06470630566279094 | -0.05711548858218723 | -0.06803824504216512 | 0 |

D-CHE/M23-alt50 是最接近有用线索的 row：h1600/h3200/h4800 为正，但 h800 mean=-0.07936050494511922。按当前 retention 定义，前一 horizon/早期 h800 非正时不能写成 retained source，只能写 late positive。

### 为什么仍然失败

- **Actuation repair 成功但 retention 失败**：M23/M25/M26 的 ActuationR2 很高，说明 readout-space projection 能实现局部 target displacement；但这种 displacement 在 real horizon 上没有稳定变成 source-vs-control retained gain。
- **single-seed smoke 不可靠**：M22/M23/M25/M26 在 D-CHE MNIST seed0 smoke 有正值，full 9-row grouped mean 转负，说明 dataset/seed heterogeneity 是硬 blocker。
- **AdamW coupling 不是充分解**：M25/M26 把 AdamW 主训练与 exact-readout residual 结合后，smoke 更强，但 full grouped KAN h800 仍负；不能把 AdamW-coupled positive smoke 解释为 FU breakthrough。
- **gate 有审计价值但没解决 grouped source**：M26 gate accept 41/130，能过滤部分低安全/低 transfer pulse，但 full grouped retention 仍为 0。
- **D-CHE/M23-alt50 是下一轮线索，不是当前成功**：late positive 说明 ultra-cap alt50 可能在更晚 horizon 产生恢复，但缺少 h800 retained continuity，不能 promotion。

### 4GPU 证据

| round | rows added | gpu_snapshot_rows | max_mem_mb | max_util_percent |
|---|---:|---:|---:|---:|
| r11 M21 | 18 | 284 | 759.0 | 27.0 |
| r12 M23 | 18 | 76 | 759.0 | 21.0 |
| r13 M22/M23 alt50 | 36 | 100 | 759.0 | 23.0 |
| r14 M25 AdamW | 18 | 68 | 759.0 | 17.0 |
| r15 M26 gated | 18 | 68 | 759.0 | 19.0 |

### 追加修改记录

- `dgkan/fu/core.py`：新增 UpdateTensor diagnostics，便于把 function-space solver 证据链写入 trace。
- `dgkan/fu/mechanisms.py`：新增 M21-M26；新增 exact-readout ridge solver；新增 cap/schedule/warmup/AdamW/gate variants。
- `experiments/run_v17_common.py`：修复 scheduled operator alt_period blocker；新增 M24-M26 execution branch；扩展 trace/final diagnostics 字段。
- `experiments/run_v19_functional_continuation.py`：新增 M21-M26 specs，并把 continuation summary/route/next queue 更新到 official artifacts。
- `v19_route_decision.json`：更新为 `R-MLPGenericRetained-KANCarrierBlocked`，并记录 M20-M26 KAN retained count 为 0。
- `v19_next_hypothesis_queue.csv/md`：下一步改成 generalization gate、AdamW coupling audit、dataset/seed heterogeneity audit，而不是继续盲目增加同类 mechanism 名。

### 下一步队列

| priority | hypothesis | reason |
|---|---|---|
| P0 | Generalization gate for M23-alt50 | 它有 late positive，但 h800 负；需要判定是 delayed recovery 还是 overfit/rebound |
| P0 | AdamW coupling audit for M25/M26 | smoke positive 但 full grouped negative；需要拆掉 optimizer writeback 对 source 的解释 |
| P1 | Dataset/seed heterogeneity audit | seed0 MNIST positive 与 full 9-row negative 的差异是当前最硬证据 |

### 追加结论

v19 的 basis efficiency blocker 已经明显改善，M20-M26 也把 function-space actuation 从低 R2 推到接近 exact projection；但 KAN carrier 的 retained source 仍没有出现。当前可信结论是：**有 MLP generic retained source，有 KAN late-positive/actuation 线索，没有 KAN-specific breakthrough**。因此本轮不能写目标达成，也不能把 promotion_allowed 改为 1。

## 2026-06-03 H5/M27-M28 追加复盘

### 为什么继续到 M27/M28

H4 之后仍有明确 P0：M21-M26 的 ActuationR2 已经很高，但 grouped KAN h800 仍为负，说明 B3 train split 不足以预测真正 held-out/grouped source。因此继续实现 stronger train-only gate：

- M27：SGD primary + exact-readout residual + multi-batch train gate + corrupted-label rejection。
- M28：AdamW primary + exact-readout residual + 同一 multi-batch train gate，用于审计 AdamW-coupled smoke positive 是否能留存。

gate 没有读取 validation/test/future label；它只用固定 train channel_a、train channel_b 与 corrupted-label train audit。

### H5 Smoke Evidence

| mechanism | spec | D-CHE MNIST seed0 h800 | h1600 | interpretation |
|---|---|---:|---:|---|
| M27 | alt50 fu1e-4 | -0.06644022464752197 | -0.009639501571655273 | negative |
| M27 | alt100 fu1e-4 | -0.0613330602645874 | 0.2805297374725342 | delayed positive only |
| M27 | alt100 fu5e-5 | -0.23987281322479248 | 0.1628739833831787 | delayed positive only |
| M27 | alt200 fu1e-4 | -0.3123122453689575 | 0.20300543308258057 | delayed positive only |
| M27 | alt200 fu5e-5 | -0.6079999208450317 | -0.047327518463134766 | negative |
| M28 | alt100 fu1e-4 | -0.019653797149658203 | 0.023518681526184082 | near miss h800 |
| M28 | alt100 fu5e-5 | 0.1523970365524292 | 0.19489693641662598 | smoke positive |
| M28 | alt200 fu1e-4 | 0.11948752403259277 | 0.160567045211792 | smoke positive |

M28 的两个 smoke positive 触发 full 9-row D-CHE rerun。这里没有把 smoke row 当成功。

### H5 Full 9-row Evidence

| carrier | mechanism/spec | rows | h800 mean | h1600 mean | h3200 mean | h4800 mean | retained |
|---|---|---:|---:|---:|---:|---:|---:|
| D-CHE | M28 alt100 fu5e-5 | 9 | -0.06881865527894762 | -0.06802056895362006 | -0.08006360133488973 | -0.12355671326319377 | 0 |
| D-CHE | M28 alt200 fu1e-4 | 9 | -0.04596583048502604 | -0.03709010283152262 | -0.04783064126968384 | -0.07944080564710829 | 0 |

Individual rows did contain positives, which is exactly why grouped evidence is necessary:

- M28 alt200 MNIST seed0: h800=0.040940284729003906, h1600=0.10047388076782227, h3200=0.14155685901641846, h4800=0.18502318859100342。
- M28 alt200 Fashion-MNIST seed0: h800=0.03186720609664917, h1600=0.03759700059890747, h3200=0.03639566898345947, h4800=0.028116703033447266。
- M28 alt100 fu5e-5 KMNIST seed1: h800=0.0420377254486084, h1600=0.027187466621398926, h3200=0.0063517093658447266, h4800=-0.03408551216125488。

但 3 datasets x 3 seeds 的 mean 仍为负，因此不能 promotion。

### H5 Gate / Actuation Evidence

- M28 diagnostic trace rows: 117
- M28 ActuationR2 min/max/mean: 0.8826937079429626 / 0.9999926686286926 / 0.9973855691078382
- M28 generalization gate accepts: 31/117
- r16 GPU snapshot rows: 2024
- r16 max memory used MB: 759.0
- r16 max GPU util percent: 18.0

这说明 gate 本身不是没工作；问题是 gate 通过的局部 update 仍不能跨 dataset/seed 形成稳定 retained source。

### H5 修改记录

- `dgkan/fu/mechanisms.py`：新增 M27/M28 和 semantic contract rows。
- `experiments/run_v17_common.py`：新增 train-only multi-batch/corrupted-label gate；trace 增加 generalization gate 字段。
- `experiments/run_v19_functional_continuation.py`：新增 M27/M28 specs。
- `v19_h5_generalization_gate_m27_m28.csv`：新增 smoke/full 证据表。
- `v19_route_decision.json`：更新 functional_continuation_rows=819、grouped_probes=61、h5_m28_retained_candidates=0。
- `v19_next_hypothesis_queue.csv/md`：更新为 dataset/seed heterogeneity、AdamW-coupled source audit、需要新 source writer 三个方向。

### H5 结论与当前不确定性

M28 是一个关键反证：**AdamW + exact-readout + multi-batch/shuffled-label gate 可以在单个 D-CHE/MNIST seed0 上很漂亮，但 full grouped source 仍失败。** 这把 blocker 从 “projection 不够准 / gate 不够强” 推到了 “dataset/seed heterogeneity 与 source writer 设计本身”。继续微调 cap、alt_period、readout-only gate 的边际价值已经很低；下一步需要新的 source-state 设计或先做 per-dataset failure anatomy，而不是继续给同一类 readout actuator 换名字。

因此，当前我不能诚实地说 v19 目标达成；也不能继续假装有明确的同类修复方向可以马上推进。promotion_allowed 仍必须是 0。

## 2026-06-03 H6 Dataset/Seed Anatomy + FU-H2/H3 追加复盘

### H6 目的

H5 已经证明：readout actuator、high ActuationR2、multi-batch gate、AdamW coupling 都不能把 D-CHE grouped source 做成 retained source。因此 H6 不再调 readout gate，而是按计划检查：

- FU-H2：slow-state retention writer 是否能保住 source。
- FU-H3：matrix/block space update 是否比逐参数/readout actuator 更稳。
- dataset/seed heterogeneity 是否解释了 smoke positive 和 grouped negative 的冲突。

### Dataset/Seed Anatomy

新增 artifacts：

- `v19_h6_dataset_seed_heterogeneity_rows.csv`
- `v19_h6_dataset_seed_heterogeneity_summary.csv`

覆盖：

- row-level heterogeneity rows: 45
- grouped summary rows: 35
- focus mechanisms: D-CHE M23/M25/M26/M28。

关键观察：

- M23/M25/M26/M28 都存在 individual positive row，但没有 grouped retained row。
- M28 的 full grouped mean 为负，不是因为完全没有局部 source，而是因为 dataset/seed 间符号和幅度不稳定。
- 这进一步支持 “single-seed smoke positive 不可 promotion” 的审计原则。

### FU-H2/H3 Smoke Evidence

H6 在 D-CHE repaired carrier、MNIST seed0 上跑了计划中的 slow-state / PopRisk / matrix-block writer smoke：

| mechanism/spec | source_h800 | source_h1600 | conclusion |
|---|---:|---:|---|
| M6-slowstate-fu0p0005 | -0.9580533504486084 | -0.8114761114120483 | fail |
| M6-slowstate-fu0p0001 | -0.9498236179351807 | -0.8082684278488159 | fail |
| M11-dual-fu0p0005 | -0.9413673877716064 | -0.8014754056930542 | fail |
| M12-schedulefree-fu0p0005 | -0.9617340564727783 | -0.8188821077346802 | fail |
| M14-popriskslow-fu0p0005 | -0.971245288848877 | -0.8306988477706909 | fail |
| M13-lowrank-r4-fu0p0005 | -0.9408831596374512 | -0.8024903535842896 | fail |
| M13-lowrank-r4-fu0p0001 | -0.9723532199859619 | -0.8309644460678101 | fail |

GPU evidence:

- GPU snapshot rows: 340
- max memory used MB: 759.0
- max GPU util percent: 51.0

Because none of these passed even a D-CHE/MNIST seed0 smoke, H6 did not escalate them to full 9-row. This avoids spending 4GPU time on a route whose first diagnostic row is already strongly negative.

### H6 修改记录

- `experiments/run_v19_functional_continuation.py`：新增 M6 slow-state 和 M13 low-rank matrix-block continuation specs。
- 新增 `v19_h6_source_state_writer_smoke.csv`，记录 H2/H3 smoke 结果。
- 新增 `v19_h6_dataset_seed_heterogeneity_rows.csv` / `summary.csv`，记录 individual-positive vs grouped-negative 证据。
- 修复 heterogeneity writer 字段集合 bug：使用 union fieldnames，避免不同 grouping row 字段不一致导致 CSV writer failure。
- `v19_route_decision.json`：新增 `H2H3SmokeNegative`，并记录 `h6_source_state_positive_smoke_rows=0`。

### H6 结论

H6 关闭了一个重要但不令人开心的分支：**已有 slow-state / dual-memory / schedule-free / PopRisk / low-rank block writer 没有在 D-CHE repaired carrier 上表现出可继续的 smoke signal。**

这并不证明所有 source-state writer 都不可能；它只证明当前实现的 H2/H3 writer 不够。结合 H5，可以更精确地说：

- readout/function-space actuator 已经能局部 actuation，但不能 grouped retained。
- AdamW-coupled gate 能做 single-seed positive，但不能 grouped retained。
- 当前 slow-state/PopRisk/matrix-block writer 连 D-CHE MNIST seed0 smoke 都失败。
- 因此 v19 目标仍未达成，promotion_allowed 必须保持 0。

当前剩余的不确定性已经不是“再跑哪个现有机制名”，而是需要新的 source-state 设计或更深入的 per-dataset failure anatomy；在没有新机制假设前，继续烧同类 full runs 不会诚实地推进结论。

## 2026-06-03 H7 Failure Anatomy / Same-Family Stop Decision

### H7 为什么不是继续重跑

用户要求“没有达成就继续”，所以 H6 后继续推进了一步；但这一步不是再发起同类 4GPU full sweep，而是把“是否还有可审计的同类修复方向”做成 artifact。理由是：

- H4-M20/M21-M26：readout/function-space actuation 能做到高 ActuationR2，但 grouped KAN retained source 为 0。
- H5-M27/M28：multi-batch / corrupted-label train-only gate 有 single-seed smoke positive，但 full 9-row grouped mean 为负。
- H6-H2/H3：已有 slow-state / PopRisk / matrix-block writer 在 D-CHE MNIST seed0 smoke 上全负。
- 计划第 13 节明确说 Case C 应优先修 carrier/block/kernel；这些已做过 E2 与 repaired carrier source 检查。继续给同类 readout gate 改 cap/alt-period 已经没有新的因果假设。

新增 artifact：

- `v19_h7_failure_anatomy_decision.csv`
- `v19_h7_failure_anatomy_decision.md`
- 更新 `v19_next_hypothesis_queue.csv/md`
- 更新 `v19_route_decision.json`

### H7 Decision Table

| scope | rows | decision | continue same family | key evidence |
|---|---:|---|---:|---|
| MLP retained functional source | 9 | functional dynamics possible but generic MLP only | 0 | M16 h800=0.01937996016608344, h1600=0.0803578429751926, h3200=0.07967897256215413, h4800=0.07385419474707709 |
| KAN repaired M20-M28 grouped continuation | 14 | same readout actuator family not promising | 0 | retained=0; best h800 D-CHE/M28=-0.04596583048502604; best h4800 D-CHE/M23=0.01684041155709161 |
| KAN late-positive rows | 1 | late positive is not retention | 0 | D-CHE/M23 h800=-0.09618967771530151, h4800=0.01684041155709161 |
| M28 smoke-positive full-negative | 2 | train-only gate not sufficient for grouped retention | 0 | best full M28 h800=-0.04596583048502604, h1600=-0.03709010283152262, retained=0 |
| Dataset/seed heterogeneity | 35 | heterogeneity blocks dataset/seed-agnostic promotion | 0 | subgroups with retained rows=18, subgroups with late positive=20, but grouped retained KAN=0 |
| FU-H2/H3 writer smoke | 7 | existing writer family negative smoke | 0 | M6/M11/M12/M13/M14 non-control smoke positive rows=0 |
| H7 final decision | 61 | blocker is conceptual, not more same-family runs | 0 | KAN retained=0, MLP retained=1, promotion_allowed=0 |

### H7 结论

v19 目标仍未达成，且不能诚实地通过继续同类 sweep 来推进。当前可信结论是：

- MLP 上存在 grouped retained functional source，所以“functional dynamics 完全不可能”不成立。
- KAN repaired carrier 上没有 grouped retained source；H4/H5 的 high ActuationR2 和 train-only gate 都没转化成 full grouped retention。
- H6 已验证的 slow-state / source-state / matrix-block writer 没有 smoke signal。
- 因此当前 blocker 已经从可执行参数搜索变成 source-state 理论设计问题。下一步若继续，必须先提出新的 train-only source observability / writer theory，并通过 toy correctness 与 smoke gate；否则继续 4GPU full runs 只是在同一失败族里耗算力。

这不是 promotion，也不是 scientific no-go for all possible source-state writers。它是 v19 计划下对现有 H1/H2/H3/H4 实现族的诚实边界：`promotion_allowed=0`，`h7_continue_same_family_recommended=0`，`h7_new_theory_required=1`。

## 2026-06-03 H8 M29 Consensus Source-State 新理论 Smoke

### H8 为什么继续

H7 的结论不是“可以停止并声明完成”，而是“同类 readout/gate/slow-state/matrix-block repair 没有可继续的因果方向，需要新的 source-state theory”。因此 H8 没有再调 M20-M28 的 cap、alt_period 或同类 gate，而是实现一个新的 train-stream cross-batch consensus source-state writer：

- 用同一 train stream 的两个 batch 梯度做 sign-consensus，只保留跨 batch 方向一致的分量。
- 用 corrupted-label batch 作为坏 source proxy，把与 corrupted gradient 同向的正投影扣掉。
- 用 slow-state EMA 积累 consensus vector，但 `one_step_descent_claim=0`，不把它冒充为标准 train-gradient descent。
- 只在 train-only signal gain 为正、corrupt gain 很小、corrupt cosine 不大、consensus density 足够时 commit；否则回退到 SGD baseline step。

这一步是新 source-state 假设 smoke，不是 official success gate。

### H8 修改记录

- `dgkan/fu/mechanisms.py`
  - 新增 `M29-ConsensusSourceStateFU` 与 semantic contract：`source_space=slow_state`，`optimizer_primary=SGD+train_consensus_gate`。
  - `make_update()` 增加 M29 分支，返回 momentum-like slow-state update tensor。
- `experiments/run_v17_common.py`
  - train loop 增加 M29 专用分支：cross-batch consensus、corrupted-label projection、slow-state EMA、gate diagnostics、fallback SGD。
  - trace / final row 增加 `source_state_consensus_density`、`source_state_corrupt_cos`、`source_state_signal_gain`、`source_state_corrupt_gain`、`source_state_gate_accept`、`source_state_norm`。
- `experiments/run_v19_functional_continuation.py`
  - 新增三组 M29 specs：`M29-consensus-alt50-fu0p0001`、`M29-consensus-alt100-fu0p0001`、`M29-consensus-alt100-fu0p00005`。
- 新增 official artifacts：
  - `v19_h8_m29_consensus_source_state_smoke.csv`
  - `v19_h8_m29_consensus_source_state_diagnostics.csv`
  - `v19_h8_m29_gpu_summary.csv`
  - `v19_h8_m29_raw_matrix.csv`
  - `v19_h8_m29_raw_traces.csv`
  - `v19_h8_m29_gpu_runtime_snapshots.csv`
  - `v19_h8_m29_consensus_source_state_decision.md`
- packaging blocker/fix：系统无 `zip` 命令，`zip -qr` 失败；改用 Python `zipfile` 重新生成 `v19_results_bundle.zip`。

### H8 Smoke Command

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile \
  dgkan/fu/mechanisms.py \
  experiments/run_v17_common.py \
  experiments/run_v19_functional_continuation.py

KAN=/home/chengshun.wang/miniconda3/envs/kan/bin/python
BASE=results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu
OUT=$BASE/smoke_functional_m29_dche
SNAP=$OUT/v19_m29_smoke_gpu_runtime_snapshots.csv
STOP=$OUT/v19_m29_smoke_gpu_monitor.stop

mkdir -p "$OUT"
rm -f "$STOP"
printf 'timestamp,gpu_index,memory_used_mb,utilization_gpu_percent\n' > "$SNAP"
(
  while [ ! -f "$STOP" ]; do
    ts=$(date '+%Y-%m-%d %H:%M:%S %z')
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r idx mem util; do
      idx=${idx// /}; mem=${mem// /}; util=${util// /}
      printf '%s,%s,%s,%s\n' "$ts" "$idx" "$mem" "$util" >> "$SNAP"
    done
    sleep 5
  done
) &
MON=$!

$KAN experiments/run_v19_functional_continuation.py \
  --out-dir "$OUT" \
  --device cuda:0 --data-root data \
  --carriers D-CHE --basis-repair-variant CHE-R2-low-degree-k3-triton \
  --datasets MNIST --seeds 0 --steps 1600 \
  --shard-count 1 --shard-index 0 --run-label m29dchesmoke \
  --spec-ids M29-consensus-alt50-fu0p0001,M29-consensus-alt100-fu0p0001,M29-consensus-alt100-fu0p00005,CTRL-AdamW,CTRL-SGD,CTRL-RandomSameRankBlock
STATUS=$?
touch "$STOP"
wait "$MON" || true
if [ "$STATUS" -ne 0 ]; then exit "$STATUS"; fi

$KAN experiments/run_v19_functional_continuation.py --merge-only --out-dir "$OUT"
```

### H8 Smoke Result

| carrier | variant | spec | rows | h800 source | h1600 source | retained | decision |
|---|---|---|---:|---:|---:|---:|---|
| D-CHE | CHE-R2-low-degree-k3-triton | M29-consensus-alt100-fu0p00005 | 1 | -0.9577474594116211 | -0.8420370817184448 | 0 | negative_smoke_no_full_escalation |
| D-CHE | CHE-R2-low-degree-k3-triton | M29-consensus-alt100-fu0p0001 | 1 | -0.9486420154571533 | -0.8309754133224487 | 0 | negative_smoke_no_full_escalation |
| D-CHE | CHE-R2-low-degree-k3-triton | M29-consensus-alt50-fu0p0001 | 1 | -0.9486403465270996 | -0.8269315958023071 | 0 | negative_smoke_no_full_escalation |

M29 gate diagnostics:

| spec | trace rows | diagnostic rows | gate accepts/trials | density mean | signal gain max | corrupt gain max | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| M29-consensus-alt100-fu0p00005 | 7 | 4 | 0/4 | 0.6166009902954102 | 2.51084566116333e-06 | -6.407499313354492e-07 | gate_rejected_all_commits |
| M29-consensus-alt100-fu0p0001 | 7 | 4 | 0/4 | 0.6438626348972321 | 8.977949619293213e-06 | 1.862645149230957e-08 | gate_rejected_all_commits |
| M29-consensus-alt50-fu0p0001 | 7 | 4 | 0/4 | 0.5564001724123955 | 5.885958671569824e-06 | -1.1362135410308838e-06 | gate_rejected_all_commits |

GPU evidence:

- GPU snapshot rows: 20
- max memory used MB: 693.0
- max GPU util percent: 17.0
- nonzero memory rows: 16
- nonzero util rows: 4

### H8 结论

M29 没有达成目标，也没有给出值得 full 9-row escalation 的 smoke signal：

- 三个 M29 specs 的 h800/h1600 source 全为显著负值。
- retained candidates = 0。
- gate accepts = 0/12；最高 train-only signal gain 只有 `8.977949619293213e-06`，未达到 commit 条件。
- 因为 D-CHE/MNIST seed0 smoke 已经失败，按计划不能把它升级为 full 4GPU grouped run，更不能写成突破。

H8 后的可信边界是：v19 目标仍未达成，`promotion_allowed=0`。我已经按 H7 推荐方向尝试了一个新的 source-state theory，但 M29 也失败；当前不确定性已经上升到“需要重新提出新的 toy-correctness/source-observability 理论”，而不是继续把现有机制换名重跑。

## 2026-06-03 H9 M30 Low-Threshold Consensus Gate Repair Smoke

### H9 为什么继续

H8 的 M29 结果有一个可疑点：不是 consensus 向量完全不可算，而是 gate 阈值过严导致 0/12 commit。M29 的最高 train-only signal gain 为 `8.977949619293213e-06`，只比 `1e-5` threshold 低一点。为了避免把“gate 过严的假阴性”写成理论失败，H9 做了一个明确的 gate repair smoke：

- 新增 M30，保留 M29 的 cross-batch consensus + corrupted-label safety。
- 将 signal threshold 从 `1e-5` 降到 `1e-7`。
- 将 slow-state EMA 从 0.97/0.03 改成 0.90/0.10，让 source state 更快响应。
- 仍然只使用 train stream，不读取 validation/test/future/query。
- 若 smoke 仍负，则不升级 full 9-row。

### H9 修改记录

- `dgkan/fu/mechanisms.py`
  - 新增 `M30-LowThresholdConsensusSourceStateFU`。
  - mechanism contract: `source_space=slow_state`，`optimizer_primary=SGD+low_threshold_consensus_gate`。
- `experiments/run_v17_common.py`
  - M29/M30 共用 consensus branch，但 M30 使用 `signal_threshold=1e-7`、`ema_beta=0.90`。
  - trace 增加 `source_state_gate_threshold`、`source_state_ema_beta`。
- `experiments/run_v19_functional_continuation.py`
  - 新增 `M30-lowthreshold-consensus-alt50-fu0p0001`、`M30-lowthreshold-consensus-alt100-fu0p0001`、`M30-lowthreshold-consensus-alt100-fu0p00005`。
- 新增 official artifacts：
  - `v19_h9_m30_lowthreshold_consensus_smoke.csv`
  - `v19_h9_m30_lowthreshold_consensus_diagnostics.csv`
  - `v19_h9_m30_gpu_summary.csv`
  - `v19_h9_m30_raw_matrix.csv`
  - `v19_h9_m30_raw_traces.csv`
  - `v19_h9_m30_gpu_runtime_snapshots.csv`
  - `v19_h9_m30_lowthreshold_consensus_decision.md`

### H9 Smoke Result

| carrier | variant | spec | rows | h800 source | h1600 source | retained | decision |
|---|---|---|---:|---:|---:|---:|---|
| D-CHE | CHE-R2-low-degree-k3-triton | M30-lowthreshold-consensus-alt100-fu0p00005 | 1 | -0.95654296875 | -0.8404527902603149 | 0 | negative_smoke_no_full_escalation |
| D-CHE | CHE-R2-low-degree-k3-triton | M30-lowthreshold-consensus-alt100-fu0p0001 | 1 | -0.9484755992889404 | -0.8305522203445435 | 0 | negative_smoke_no_full_escalation |
| D-CHE | CHE-R2-low-degree-k3-triton | M30-lowthreshold-consensus-alt50-fu0p0001 | 1 | -0.9478328227996826 | -0.8264065980911255 | 0 | negative_smoke_no_full_escalation |

M30 gate diagnostics:

| spec | trace rows | diagnostic rows | gate accepts/trials | density mean | signal gain max | corrupt gain max | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| M30-lowthreshold-consensus-alt100-fu0p00005 | 7 | 4 | 4/4 | 0.6165540516376495 | 2.771615982055664e-06 | -7.897615432739258e-07 | gate_committed_but_source_negative |
| M30-lowthreshold-consensus-alt100-fu0p0001 | 7 | 4 | 4/4 | 0.6396396607160568 | 6.634742021560669e-06 | 1.1175870895385742e-08 | gate_committed_but_source_negative |
| M30-lowthreshold-consensus-alt50-fu0p0001 | 7 | 4 | 4/4 | 0.5566816851496696 | 1.2181699275970459e-05 | -2.3618340492248535e-06 | gate_committed_but_source_negative |

GPU evidence:

- GPU snapshot rows: 20
- max memory used MB: 693.0
- max GPU util percent: 17.0
- nonzero memory rows: 16
- nonzero util rows: 4

### H9 结论

H9 修复了 H8 的“zero commit”问题，但没有修复 source：

- M30 gate accepts = 12/12，说明低阈值 gate 确实让 consensus source-state 进入训练轨迹。
- 三个 M30 specs 的 h800/h1600 source 仍全为负。
- retained candidates = 0。
- 因此 “M29 只是 gate 太严导致假阴性” 这个解释被当前 smoke 否定。

这一步后，继续在 M29/M30 consensus family 里调阈值、EMA 或 alt_period 已经没有新的因果假设。v19 目标仍未达成，`promotion_allowed=0`；下一步若继续，必须是不同的 toy-correctness/source-observability 机制，而不是再调同一 consensus gate。

## 2026-06-03 H10 M31 AdamW Low-Threshold Consensus Residual Full Rerun

### H10 为什么继续

H9 的 M30 修复了 gate，但结果仍全负。对比 MLP retained source 和 D-CHE/D-FOU 负数结果后，剩余一个可检验解释是：M29/M30 用 SGD carrier 训练，而 v19 早期的正线索主要出现在 AdamW carrier 或 AdamW-coupled dynamics；如果完全离开 AdamW 主优化器，consensus residual 可能不是在修复 source，而是在破坏 carrier 训练。

H10 因此不是继续调 M30 阈值，而是尝试一个新的、可审计的机制：

- 新增 `M31-AdamWLowThresholdConsensusResidualFU`。
- AdamW 作为 primary optimizer 先正常更新。
- low-threshold consensus source-state 只作为 residual FU 叠加，不覆盖 AdamW 主训练。
- 仍使用 train-stream signal/corrupt split，不读 validation/test/future/query。
- 先跑 D-CHE/MNIST seed0 smoke；smoke h800/h1600 三个 spec 全正后，才升级为 D-CHE x 3 datasets x 3 seeds 的 4GPU full 9-row grouped rerun。

### H10 修改记录

- `dgkan/fu/mechanisms.py`
  - 新增 `M31-AdamWLowThresholdConsensusResidualFU`。
  - mechanism contract: `source_space=slow_state`，`optimizer_primary=AdamW+low_threshold_consensus_residual`。
- `experiments/run_v17_common.py`
  - M29/M30/M31 共用 consensus readback 分支。
  - M31 使用 `signal_threshold=1e-7`、`ema_beta=0.90`，但训练 step 改为 AdamW primary + FU residual。
  - trace 保留 `source_state_gate_accept`、`source_state_signal_gain`、`source_state_corrupt_gain`、`source_state_consensus_density`。
- `experiments/run_v19_functional_continuation.py`
  - 新增 `M31-adamw-consensus-alt50-fu0p0001`。
  - 新增 `M31-adamw-consensus-alt100-fu0p0001`。
  - 新增 `M31-adamw-consensus-alt100-fu0p00005`。
- 新增 official artifacts：
  - `v19_h10_m31_smoke_summary.csv`
  - `v19_h10_m31_smoke_matrix.csv`
  - `v19_h10_m31_smoke_traces.csv`
  - `v19_h10_m31_smoke_gpu_runtime_snapshots.csv`
  - `v19_h10_m31_full_summary.csv`
  - `v19_h10_m31_full_matrix.csv`
  - `v19_h10_m31_full_traces.csv`
  - `v19_h10_m31_full_row_examples.csv`
  - `v19_h10_m31_diagnostics.csv`
  - `v19_h10_m31_gpu_summary.csv`
  - `v19_h10_m31_adamw_consensus_decision.md`

### H10 Smoke Result

| carrier | variant | spec | rows | h800 source | h1600 source | smoke decision |
|---|---|---|---:|---:|---:|---|
| D-CHE | CHE-R2-low-degree-k3-triton | M31-adamw-consensus-alt100-fu0p00005 | 1 | 0.06744861602783203 | 0.0416184663772583 | upgrade_to_full_9row |
| D-CHE | CHE-R2-low-degree-k3-triton | M31-adamw-consensus-alt100-fu0p0001 | 1 | 0.09428870677947998 | 0.10595178604125977 | upgrade_to_full_9row |
| D-CHE | CHE-R2-low-degree-k3-triton | M31-adamw-consensus-alt50-fu0p0001 | 1 | 0.06738388538360596 | 0.10177969932556152 | upgrade_to_full_9row |

解释：M31 smoke 与 H9 M30 的 D-CHE/MNIST seed0 同 carrier / same repaired basis 对比，M30 全负，M31 全正。这说明前一个 blocker 不只是 gate 阈值，而是 SGD-only consensus writer 与 carrier training dynamics 不匹配。

### H10 Full 9-Row Grouped Result

| carrier | variant | spec | rows | h800 mean | h800 pass | h1600 mean | h1600 pass | h3200 mean | h3200 pass | h4800 mean | h4800 pass | retained |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-CHE | CHE-R2-low-degree-k3-triton | M31-adamw-consensus-alt100-fu0p00005 | 9 | 0.016972078217400446 | 6 | 0.022935476568010118 | 6 | 0.02758270502090454 | 5 | -0.007561306158701579 | 5 | 1 |
| D-CHE | CHE-R2-low-degree-k3-triton | M31-adamw-consensus-alt100-fu0p0001 | 9 | 0.016534394688076444 | 5 | 0.02917080455356174 | 7 | 0.02548849582672119 | 6 | -0.016299128532409668 | 6 | 1 |
| D-CHE | CHE-R2-low-degree-k3-triton | M31-adamw-consensus-alt50-fu0p0001 | 9 | -0.018990496794382732 | 3 | -0.027270734310150146 | 3 | -0.042707622051239014 | 2 | -0.0903786751959059 | 2 | 0 |

关键结论：

- H10 首次在 repaired KAN carrier 上得到 grouped retained candidate：2/3 M31 specs 在 h800/h1600/h3200 都为正。
- 这不是单 row positive；每个 spec 都是 D-CHE x MNIST/Fashion-MNIST/KMNIST x seed 0/1/2 的 9-row grouped mean。
- 但两个 retained specs 的 h4800 grouped mean 都转负：`-0.007561306158701579` 和 `-0.016299128532409668`。
- 因此 H10 是 partial breakthrough signal，不是 v19 promotion。

### Best Spec Row-Level Evidence

`M31-adamw-consensus-alt100-fu0p00005` 的 9-row source 轨迹如下。它解释了为什么 grouped mean 可以到 h3200 为正，但 h4800 被 outlier/washout 拉负。

| dataset | seed | h800 | h1600 | h3200 | h4800 |
|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | 0 | 0.03536081314086914 | 0.08109056949615479 | 0.11488997936248779 | 0.145737886428833 |
| Fashion-MNIST | 1 | -0.03888142108917236 | -0.06273621320724487 | -0.0387762188911438 | -0.0067809224128723145 |
| Fashion-MNIST | 2 | 0.07387173175811768 | 0.08999323844909668 | -0.022881031036376953 | -0.2977299690246582 |
| KMNIST | 0 | 0.07134735584259033 | 0.14491724967956543 | 0.2142629623413086 | 0.12284088134765625 |
| KMNIST | 1 | -0.10045933723449707 | -0.12731778621673584 | -0.13591575622558594 | -0.1213834285736084 |
| KMNIST | 2 | 0.06992495059967041 | 0.08472955226898193 | 0.1063307523727417 | 0.10700869560241699 |
| MNIST | 0 | 0.06744861602783203 | 0.0416184663772583 | 0.045124053955078125 | 0.03417015075683594 |
| MNIST | 1 | 0.04818367958068848 | 0.06658005714416504 | 0.09195780754089355 | 0.08464980125427246 |
| MNIST | 2 | -0.07404768466949463 | -0.11245584487915039 | -0.1267482042312622 | -0.13656485080718994 |

证据链解释：

- 6/9 rows 在 h800/h1600 为正。
- 5/9 rows 在 h3200/h4800 为正。
- h4800 grouped mean 转负主要由 Fashion-MNIST seed2 的 `-0.2977299690246582`、MNIST seed2 的 `-0.13656485080718994`、KMNIST seed1 的 `-0.1213834285736084` 拉低。
- 这支持 “KAN carrier partial retained source exists, but seed/dataset instability and long-horizon washout remain” 的 route，而不是 promotion route。

### M31 Gate Diagnostics

| spec | trace rows | diagnostic rows | gate accepts/trials | density mean | signal gain mean | corrupt gain mean | diagnosis |
|---|---:|---:|---:|---:|---:|---:|---|
| M31-adamw-consensus-alt100-fu0p00005 | 90 | 63 | 17/63 | 0.327497147141941 | 2.0202216027038438e-07 | -3.8157616342817035e-07 | retained_to_h3200_h4800_washout |
| M31-adamw-consensus-alt100-fu0p0001 | 90 | 63 | 19/63 | 0.32893012771530755 | 4.287631738753546e-07 | -8.258318144177633e-07 | retained_to_h3200_h4800_washout |
| M31-adamw-consensus-alt50-fu0p0001 | 90 | 63 | 19/63 | 0.33461140380019233 | 4.007167825918822e-07 | -8.911722236209446e-07 | negative_grouped |

Overall trace diagnostics:

- trace rows: 540
- M31 diagnostic rows: 189
- gate accepts/trials: 55/189
- consensus density range/mean: 0.17286036908626556 / 0.6392642855644226 / 0.33034622621914694
- signal gain range/mean: -3.6670826375484467e-09 / 3.0994415283203125e-06 / 3.4383403891254037e-07
- corrupt gain range/mean: -5.766749382019043e-06 / 2.9802322387695312e-08 / -6.995267338222927e-07

解释：M31 的 corrupted-label projection 平均为负，说明 safety gate 没有明显把 corrupted direction 当 source；但 gate 接受率只有 55/189，且 h4800 washout 仍在。这提示下一步要做 independent confirmation 和 long-horizon washout 定位，而不是直接 S5。

### GPU / Runtime Evidence

- Full H10 GPU snapshot rows: 76
- max memory used MB: 693.0
- max GPU util percent: 15.0
- nonzero memory rows: 72
- nonzero util rows: 68
- shard journal:
  - `2026-06-03 07:20:16 +08 v19 functional continuation shard 0/4 device=cuda:0 jobs=14`
  - `2026-06-03 07:20:16 +08 v19 functional continuation shard 1/4 device=cuda:1 jobs=14`
  - `2026-06-03 07:20:16 +08 v19 functional continuation shard 2/4 device=cuda:2 jobs=13`
  - `2026-06-03 07:20:16 +08 v19 functional continuation shard 3/4 device=cuda:3 jobs=13`
  - `2026-06-03 07:23:26 +08 v19 functional continuation merge rows=54 traces=540`

### H10 结论

H10 没有达成 v19 目标，但把 functional blocker 从 “KANCarrierBlocked” 推进到了更具体的 “KANCarrierPartialRetainedH3200ButH4800Washout”：

- 真实改进：M31 在 D-CHE repaired KAN carrier 上出现 2 个 grouped retained candidates through h3200。
- 仍未达标：h4800 grouped mean washout；不是 9/9 pass；S4/S5 real-transfer official gate 未执行；promotion_allowed 仍为 0。
- 下一步不能继续声称“没有 KAN source”；现在应该做 H11 independent confirmation / washout analysis，确认 h3200 source 是否可复现，以及 h4800 是 outlier、commit schedule、AdamW residual coupling 还是 dataset/seed heterogeneity 导致。

## 2026-06-03 H11 M31 Independent Confirmation

### H11 为什么继续

H10 的 M31 full rerun 给出 2 个 D-CHE grouped retained candidates through h3200，但它仍可能只是特定初始化/采样轨迹的正例。v19 计划中 S3b/S5 都要求 independent rerun 和 controls；因此 H11 不扩大机制搜索，只做 independent confirmation：

- 保留 D-CHE + `CHE-R2-low-degree-k3-triton`。
- 保留 H10 最好的两个 M31 specs。
- 保留 matched controls：`CTRL-AdamW`、`CTRL-SGD`、`CTRL-RandomSameRankBlock`。
- 新增 `--init-seed-offset`，让模型初始化和训练 batch RNG 独立偏移；默认值为 0，不影响历史结果。
- 运行 3 个 independent offsets：100000、200000、300000。

### H11 修改记录

- `experiments/run_v19_functional_continuation.py`
  - 新增 `--init-seed-offset`。
  - 每个 job 记录 `init_seed_offset`。
  - `carrier_model()` seed 和 `train_one()` seed 都加上 offset，使 rerun 不只是同一确定性轨迹重放。

### H11 4GPU Result

Aggregate summary:

| spec | rows | h800 mean | h800 pass | h1600 mean | h1600 pass | h3200 mean | h3200 pass | h4800 mean | h4800 pass | retained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M31-adamw-consensus-alt100-fu0p00005 | 27 | -0.06243820322884454 | 2 | -0.10373381994388721 | 2 | -0.1303710142771403 | 2 | -0.16738004154629177 | 2 | 0 |
| M31-adamw-consensus-alt100-fu0p0001 | 27 | -0.06181360836382265 | 3 | -0.09745090979116934 | 3 | -0.12961135970221627 | 4 | -0.16300636309164543 | 3 | 0 |

Per-rep/spec confirmation:

| run_label | spec | h800 | h1600 | h3200 | h4800 | retained_to_h3200 |
|---|---|---:|---:|---:|---:|---:|
| h11r1 | M31-adamw-consensus-alt100-fu0p00005 | -0.06798232263988918 | -0.12056319581137763 | -0.15427301989661324 | -0.19875314500596789 | 0 |
| h11r1 | M31-adamw-consensus-alt100-fu0p0001 | -0.07854105366600884 | -0.13639258676105076 | -0.1909844676653544 | -0.24075057771470812 | 0 |
| h11r2 | M31-adamw-consensus-alt100-fu0p00005 | -0.05512229601542155 | -0.0970979200469123 | -0.1239466733402676 | -0.15559087859259713 | 0 |
| h11r2 | M31-adamw-consensus-alt100-fu0p0001 | -0.06664018498526679 | -0.09262755182054308 | -0.12433444791369969 | -0.15142179860009086 | 0 |
| h11r3 | M31-adamw-consensus-alt100-fu0p00005 | -0.06420999103122288 | -0.09354034397337171 | -0.11289334959454006 | -0.14779610104031032 | 0 |
| h11r3 | M31-adamw-consensus-alt100-fu0p0001 | -0.04025958644019233 | -0.0633325907919142 | -0.07351516352759467 | -0.09684671296013726 | 0 |

H11 diagnostics:

- matrix rows: 135
- M31 rows: 54
- traces: 1350
- reproduced retained-to-h3200 rep/spec count: 0/6
- h4800 positive rep/spec count: 0/6
- M31 gate accepts/trials: 109/378
- GPU snapshot rows: 180
- max memory used MB: 693.0
- max GPU util percent: 20.0

### H11 结论

H11 否定了“H10 M31 已经形成稳健 KAN retained source”的解释：

- H10 offset0 的 h3200 retained 没有在 3 个 independent init-seed-offset rerun 中复现。
- aggregate h800/h1600/h3200/h4800 全为负。
- 因此 H10 只能保留为 fragile positive signal，不能升级 S3/S3b，更不能 S5。

下一步仍有一个具体、可修的实现假设：M31 在 AdamW step 前计算 consensus residual，但在 AdamW step 后应用 residual。这个 order mismatch 可能让 source 向量在 post-AdamW 参数点变成 stale update。H12 将尝试 post-AdamW recompute consensus residual；若 H12 smoke 仍负，应关闭 M31/M32 same-family 路线。

## 2026-06-03 H12 M32 Post-AdamW Consensus Residual Smoke

### H12 为什么继续

H11 已经否定了 H10 的独立复现，但仍留下一个具体实现假设：M31 在 commit step 中先基于 pre-AdamW 参数点计算 consensus residual，然后执行 AdamW，再把 residual 应用到 post-AdamW 参数点。这可能导致 source residual stale。H12 做最小修复：

- 新增 `M32-PostAdamWConsensusResidualFU`。
- commit step 先执行 AdamW。
- 在 post-AdamW 参数点重新计算 channel A/B consensus、corrupted-label projection、slow-state 和 gate。
- gate 接受后才应用 residual。
- 用 H11 失败的 independent offset `100000` 做 D-CHE/MNIST seed0 smoke。

### H12 修改记录

- `dgkan/fu/mechanisms.py`
  - 新增 `M32-PostAdamWConsensusResidualFU`。
  - mechanism contract: `source_space=slow_state`，`optimizer_primary=AdamW+post_step_consensus_residual`。
- `experiments/run_v17_common.py`
  - M32 在 consensus branch 中走 post-AdamW recompute path。
  - `source` 记录为 `train_stream_post_adamw_low_threshold_cross_batch_consensus_residual`。
- `experiments/run_v19_functional_continuation.py`
  - 新增 `M32-postadamw-consensus-alt50-fu0p0001`。
  - 新增 `M32-postadamw-consensus-alt100-fu0p0001`。
  - 新增 `M32-postadamw-consensus-alt100-fu0p00005`。

### H12 Smoke Result

| carrier | variant | spec | rows | h800 source | h1600 source | retained | decision |
|---|---|---|---:|---:|---:|---:|---|
| D-CHE | CHE-R2-low-degree-k3-triton | M32-postadamw-consensus-alt100-fu0p00005 | 1 | -0.12051677703857422 | -0.15167248249053955 | 0 | negative_smoke_no_full_escalation |
| D-CHE | CHE-R2-low-degree-k3-triton | M32-postadamw-consensus-alt100-fu0p0001 | 1 | -0.10039818286895752 | -0.151373028755188 | 0 | negative_smoke_no_full_escalation |
| D-CHE | CHE-R2-low-degree-k3-triton | M32-postadamw-consensus-alt50-fu0p0001 | 1 | -0.08437776565551758 | -0.12100136280059814 | 0 | negative_smoke_no_full_escalation |

Gate diagnostics:

- trace rows: 42
- diagnostic rows: 12
- gate accepts/trials: 6/12
- consensus density range/mean: 0.26520270109176636 / 0.5279654860496521 / 0.37151214232047397
- signal gain range/mean: 0.0 / 2.7194619178771973e-06 / 6.032957268568376e-07
- corrupt gain range/mean: -2.6747584342956543e-06 / 7.450580596923828e-08 / -9.390835960706075e-07

GPU evidence:

- GPU snapshot rows: 24
- max memory used MB: 693.0
- max GPU util percent: 16.0

### H12 结论

H12 没有修复 H11 blocker：

- post-AdamW recompute 后，三个 M32 specs 仍然 h800/h1600 全负。
- gate 不是完全关闭，6/12 accepted，但 accepted residual 没有形成 source。
- 因此 H10 offset0 positive 更像 fragile initialization-specific signal，而不是当前 source-state family 的稳健机制。

截至 H12，v19 目标没有达成，且继续在 M31/M32 same-family 内调 alt-period、阈值或 lr 已经没有新的因果假设。当前诚实边界是：

- `promotion_allowed=0`
- `continue_same_family_recommended=0`
- `new_source_theory_required=1`
- 需要先提出新的 source-observability toy correctness，再继续 4GPU 机制实验；否则只是换名重跑。

## 2026-06-03 H13 H10/H11 Offline Anatomy

### H13 为什么继续

H12 已经关闭了 post-AdamW stale residual 这个实现假设，但为了避免把 H10 的 h3200 正例简单归因成“偶然”，H13 做了一次不重跑训练的离线解剖：读取 H10/H11 已生成的 matrix/trace，比较 M31 自身 loss、最佳 control loss、recorded source、computed source 和 source-state gate 诊断。

H13 只生成分析 artifact，不新增训练数据；所有数字来自 H10/H11 既有 fresh rows。

### H13 结果摘要

| phase | spec | rows | h800 mean | h1600 mean | h3200 mean | h4800 mean | retained_to_h3200 | h4800_positive |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H10-offset0 | M31-adamw-consensus-alt100-fu0p00005 | 9 | 0.016972078217400446 | 0.022935476568010118 | 0.02758270502090454 | -0.007561306158701579 | 1 | 0 |
| H10-offset0 | M31-adamw-consensus-alt100-fu0p0001 | 9 | 0.016534394688076444 | 0.02917080455356174 | 0.02548849582672119 | -0.016299128532409668 | 1 | 0 |
| H10-offset0 | M31-adamw-consensus-alt50-fu0p0001 | 9 | -0.018990496794382732 | -0.027270734310150146 | -0.042707622051239014 | -0.0903786751959059 | 0 | 0 |
| H11-independent | M31-adamw-consensus-alt100-fu0p00005 | 27 | -0.06243820322884454 | -0.10373381994388721 | -0.1303710142771403 | -0.16738004154629177 | 0 | 0 |
| H11-independent | M31-adamw-consensus-alt100-fu0p0001 | 27 | -0.06181360836382265 | -0.09745090979116934 | -0.12961135970221627 | -0.16300636309164543 | 0 | 0 |

H13 的核心判定：

- H10 有 2 个 spec 在 h3200 grouped mean 为正，但 H10 自身 h4800 grouped mean 全部转负。
- H11 三个 independent init offsets 下，2 个候选 spec 在 h800/h1600/h3200/h4800 grouped mean 全部为负。
- H10 h3200 positive spec count = 2；H11 h3200 positive spec count = 0。
- H10/H11 h4800 positive spec count = 0 / 0。
- 因此 H10 partial positive 不能作为 retained source，也不能升级为 S3/S4/S5。

### H10/H11 Delta

| spec | h11-h10 h800 | h11-h10 h1600 | h11-h10 h3200 | h11-h10 h4800 |
|---|---:|---:|---:|---:|
| M31-adamw-consensus-alt100-fu0p00005 | -0.07941028144624498 | -0.12666929651189732 | -0.15795371929804483 | -0.1598187353875902 |
| M31-adamw-consensus-alt100-fu0p0001 | -0.0783480030518991 | -0.12662171434473107 | -0.15509985552893746 | -0.14670723455923576 |

这个 delta 说明 H11 不是轻微退化，而是每个 horizon 都整体向负方向移动。H13 也核对了 computed source 与 recorded source 的方向一致，未发现“source 字段计算写错导致 H11 假负”的证据。

### Gate 诊断

| phase | spec | gate accepts/trials | accept rate | density mean | signal_gain_mean | corrupt_gain_mean |
|---|---|---:|---:|---:|---:|---:|
| H10-offset0 | M31 alt100 fu5e-5 | 17/63 | 0.2698412698412698 | 0.327497147141941 | 2.0202216027038438e-07 | -3.8157616342817035e-07 |
| H10-offset0 | M31 alt100 fu1e-4 | 19/63 | 0.30158730158730157 | 0.32893012771530755 | 4.287631738753546e-07 | -8.258318144177633e-07 |
| H11 r1 | M31 alt100 fu5e-5 | 18/63 | 0.2857142857142857 | 0.3419729338751899 | 2.1429935706749794e-07 | -3.4798941914997406e-07 |
| H11 r1 | M31 alt100 fu1e-4 | 18/63 | 0.2857142857142857 | 0.34662639692662256 | 4.2979705280491285e-07 | -7.879875955127534e-07 |
| H11 r2 | M31 alt100 fu5e-5 | 18/63 | 0.2857142857142857 | 0.3325349239129869 | 2.0707929728641396e-07 | -3.686559105676318e-07 |
| H11 r2 | M31 alt100 fu1e-4 | 18/63 | 0.2857142857142857 | 0.33801063682351795 | 3.947752125058619e-07 | -7.919198463833522e-07 |

Gate accept rate 在 H10/H11 之间接近，说明 H11 失败不是因为 gate 完全关闭；而是 accepted residual 没有转化为独立初始化下相对 control 的 source。继续单纯降低 threshold 或提高 accept rate 没有新的因果证据。

### H13 修改记录（便于审计）

- 新增 official artifact：`v19_h13_h10_h11_anatomy_summary.csv`。
- 新增 official artifact：`v19_h13_h10_h11_controls_vs_m31.csv`。
- 新增 official artifact：`v19_h13_h10_h11_gate_diagnostics.csv`。
- 新增 official artifact：`v19_h13_h10_h11_phase_rollup.csv`。
- 新增 official artifact：`v19_h13_h10_h11_anatomy_decision.md`。
- 新增复现脚本：`experiments/run_v19_h13_h10_h11_anatomy.py`，用于重放 H13 离线解剖并刷新 official artifacts。
- 修复 `v19_required_artifact_manifest.csv` 写入兼容：旧 manifest 没有 `note` 字段，H13 生成脚本改为取字段并集后再追加 H13 artifact 行；不改动实验数值。
- 更新 `v19_route_decision.json`：加入 `h13_anatomy_completed=1`、`h13_actionable_same_family_repair=0`、`new_source_theory_required=1`。
- 刷新 `v19_results_bundle.zip`，bundle size = 24768261 bytes。

### H13 结论

H13 后，v19 仍未达成目标：

- `promotion_allowed=0`
- `official_success_reached=0`
- `continue_same_family_recommended=0`
- `new_source_theory_required=1`

当前不是“还没多扫几个参数”的问题。H10 正例无法独立复现，H12 的实现修复也失败，H13 进一步显示 gate 本身并不是主要 blocker。继续推进需要先提出新的 source-observability/toy-correctness 理论，再把它落到机制实现；否则继续同族 M31/M32 参数扫描不具备审计价值。

### 最终验证

- py_compile 通过：`dgkan/fu/mechanisms.py`、`experiments/run_v17_common.py`、`experiments/run_v19_functional_continuation.py`、`experiments/run_v19_h13_h10_h11_anatomy.py`。
- H13 official artifact 行数：summary 6、controls-vs-M31 82、gate diagnostics 10、phase rollup 4。
- `v19_results_bundle.zip` size = 24768261 bytes，并包含全部 H13 official artifacts 与 `v19_route_decision.json`。
- 结束时未发现持续运行的 v19 training 或 GPU monitor 进程；`ps` 仅匹配验证命令自身。
