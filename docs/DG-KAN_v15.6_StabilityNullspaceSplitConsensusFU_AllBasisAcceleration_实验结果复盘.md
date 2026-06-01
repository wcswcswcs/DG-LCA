# DG-KAN v15.6 StabilityNullspaceSplitConsensusFU AllBasisAcceleration 实验结果复盘

生成时间：2026-05-31（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 stability-nullspace diagnostic、substrate-only rows 或 MLP/generic control 写成 promotion。

## 1. 计划理解

v15.6 的目标不是新增 action family，而是把 v15.04/v15.5 的 G7 metric-only 正信号分解为 source/hazard tangent，并测试 train-stream-only hazard-nullspace 与 source-retaining constrained projection 是否能保留 source gain 同时降低 tail/LineC bad events。

## 2. 本轮代码修改

新增：

```text
experiments/run_v156_stability_nullspace_split_consensus_fu_allbasis.py
```

修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v15.6 substrate-only candidates:
  D-FOU67..71, D-RBF65..69, D-WAV57..60。
```

过程说明：

```text
1. N0 复现 v15.04/v15.5 G7 metric-only semantics。
2. N1..N6 只用当前 train stream split-loss、recovery-lag、logit/entropy、degree/update/projection proxy tangent 做 hazard-null projection。
3. LineC/CEp99/NLL/ECE/AUCtime/Brier 只用于 audit/gate/failure taxonomy，不进入方向。
4. Q1..Q5 使用固定 r_min=0.60 的 source-retaining constrained projection；不做网格搜索。
5. 所有训练/finalizer 命令均使用 --device cuda:0；cpu_offload_used=0。
6. 自查发现初版把 minibatch generator 与 method 名字绑定，
   已修正为同一 dataset/seed/family 下 N0/NQ/controls 使用 matched batch stream，
   random direction 单独使用 method-specific generator；修正后不沿用旧训练指标。
7. Line U 只做 source-hazard decomposition readback，不提交 update。
8. Line B proxy/leaveout 只作为 readback audit，不作为 gating direction。
9. Line P 使用计划 P1..P7/P0 failure taxonomy。
10. 用户再次追问后自查发现 Q 分支 telemetry 初版记录的是 full-null retention，
    不是 fixed-alpha constrained projection 实际提交 update 的 retention；已修正并重跑 official GPU matrix。
11. 同次自查发现 Line U 对 N0/G7R 用 method-specific empty basis 读数，
    会把 G7R source-hazard overlap 低估为 0；已改为统一 combined-hazard basis readback。
```

<!-- V15.6_MANUAL_ANALYSIS_START -->
## 2.1 人工复核分析 / Insight

这段是人工复核分析，不是 Python 自动生成的指标结论；下方 Line S / U / NQ / B / M / D / route 数据仍由 runner 从 artifact 写入。

我的判断是：v15.6 确认了 v15.5 的核心信号不是偶然消失的。N0-G7R-Replay 仍有很强 source gain，`source_vs_best_control_mean = 0.46147918701171875`，`control_equivalent_fraction = 0.0`，Line M 仍有 6/9 KAN-specific delta 为正；也就是说，split-consensus metric-only 方向依然不是被 generic/MLP controls 完全解释掉的普通 optimizer artifact。

但 v15.6 的 nullspace 试验也说明当前 source 与 hazard 没有被成功解耦。用户再次追问后的自查修复非常关键：Line U 改成统一 combined-hazard basis readback 后，`source_hazard_colinear_rows = 63 / 63`，`mean_hazard_overlap = 0.956108654302264`，`mean_source_retention_after_null = 0.04575427500383249`。这比早期读数更严厉，也更符合实际：当前 train-stream hazard tangent 和 G7R source direction 基本同线，简单 null 掉 hazard 会把 source 一起 null 掉。

Q 分支的 telemetry 也已修正：`source_retention_after_null` 现在表示 fixed-alpha constrained projection 实际提交 update 的 retention，而 full-null retention 保存在 `*_full_null` 字段。修复后 Q2/Q4/Q5 的实际 retention 被拉回到约 `0.60`，但它们的 full-null retention 只有约 `0.036-0.058`；这说明约束确实在“保 source”，但代价是 hazard 没有被充分移除，所以 bad_event 仍然没有过 gate。N3/N5/N6 虽然降低了 tail/LineC failure fraction，但 source 变成负数；Q2/Q4/Q5 保留了一部分正 source，却仍然 bad_event=1.0。当前 fixed `r_min=0.60` constrained projection 没有找到“保 source、去 hazard”的可行边界。

Line B 也支持这个判断：B6/B10 对 bad_event 的 AUC 很高，但 best proxy 的 leaveout min AUC 仍只有 0.25，说明 train-stream proxy 对不稳定性有读数，却不够稳健，不能升级为 gating/controller，也不能拿它反推方向。

因此 v15.6 的科学结论不是“没有信号”，而是“当前 hazard-nullspace definition 把 source 和 hazard 视作高度重叠”。继续在本版里加 N7/Q6/G9/G10、action bank、controller 或 reset route 都会偏离计划精神；下一步需要新的 theory-level stability definition，或者换 substrate/base architecture，让 source component 在参数空间里不再和 instability tangent 强绑定。
<!-- V15.6_MANUAL_ANALYSIS_END -->

## 2.2 完整计划执行对照（artifact 自动写入）

执行合同覆盖：

| contract item | status | details |
|---|---:|---|
| Line R provenance/no-action audit | 1 | audit files present |
| Line S stability surface | 1 | K=2/4/8 and batch=128/256 sensitivity |
| Line U source-hazard decomposition | 1 | G7R/top NQ/controls hazard overlap readback |
| Line N/Q nullspace + constrained FU controls | 1 | N0..N6 + Q1..Q5 + C0..C8 |
| Line B proxy audit | 1 | B1..B10 readback-only leaveout |
| Line P micro-horizon audit | 1 | top methods + controls |
| Line M generic controls | 1 | MLP/generic controls |
| Line D all-basis + monitors | 1 | v15.6 substrate candidates + no-regression |
| Required figures | 1 | figures=11 |
| No forbidden continuation | 1 | no G9/G10/action/controller/reset/audit-directed branch |

深度覆盖审计：

| audit item | status | details |
|---|---:|---|
| Line N/Q exact surface | 1 | N0..N6/Q1..Q5/C0..C8 present |
| Line B exact surface | 1 | B1..B10 present |
| Line D exact surface | 1 | D-FOU/RBF/WAV candidates present |
| Direction provenance train-stream-only | 1 | direction_rows=1764 |
| No remaining legal v15.6 continuation | 1 | if no S5, continuation requires next theory/substrate plan; no G9/G10/action/controller/reset |

required / forbidden / no-action / provenance：

```text
required_artifact_manifest_rows = 43
required_artifact_missing_rows = 0
forbidden_information_audit_rows = 12
forbidden_information_violation_sum = 0.0
no_action_search_audit_rows = 12
no_action_search_violation_sum = 0.0
direction_provenance_rows = 1764
direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction
method_surface_rows = 45
```

## 3. Line S split-consensus stability

```text
line_s_rows = 108
line_s_k_batch_sensitivity_rows = 648
line_s_gate_pass = 1
line_s_best_snr = 1.9560231276069324
```

| fallback | rows | K2 SNR | K4 SNR | K8 SNR | gate pass rows | certificate |
|---|---:|---:|---:|---:|---:|---:|
| S-FB1 | 648 | 10.519598875774431 | 1.9560231276069324 | 0.49137829373764685 | 64 | 0 |
| S-FB2 | 648 | 10.519598875774431 | 1.9560231276069324 | 0.49137829373764685 | 64 | 0 |
| S-FB3 | 648 | 10.519598875774431 | 1.9560231276069324 | 0.49137829373764685 | 64 | 0 |
| S-FB4 | 648 | 10.519598875774431 | 1.9560231276069324 | 0.49137829373764685 | 64 | 0 |
| S-FB5 | 648 | 10.519598875774431 | 1.9560231276069324 | 0.49137829373764685 | 64 | 1 |

## 4. Line U source-hazard tangent decomposition

```text
line_u_rows = 63
source_hazard_colinear_rows = 63
decomposition_support_rows = 0
mean_hazard_overlap = 0.956108654302264
mean_source_retention_after_null = 0.04575427500383249
```

| method | rows | hazard overlap | source retention | rank | condition | support rows | colinear rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0-D-CHE-AdamW | 9 | 0.9545291132397122 | 0.047797572074664965 | 3.0 | 62.859605153401695 | 0 | 9 |
| C3-G7R-RandomMatchedNorm | 9 | 0.9700355297989316 | 0.0321867550826735 | 3.0 | 90.97002410888672 | 0 | 9 |
| C5-SameHazardNullRandomDirection | 9 | 0.9700638254483541 | 0.032146703348391585 | 3.0 | 91.04695256551106 | 0 | 9 |
| N0-G7R-Replay | 9 | 0.9388772282335494 | 0.062118563801050186 | 3.0 | 62.09211858113607 | 0 | 9 |
| N3-EntropyRecoveryNull | 9 | 0.956397145986557 | 0.045629958104756146 | 3.0 | 67.40273920694987 | 0 | 9 |
| N5-CombinedHazardNull-rank4 | 9 | 0.958784368303087 | 0.04326781930608882 | 3.0 | 72.05510499742296 | 0 | 9 |
| Q4-SourceRetainingCombinedConstraint-r4 | 9 | 0.944073369105657 | 0.057132553309202194 | 3.0 | 61.26084179348416 | 0 | 9 |

## 5. Line B train-stream bad-event proxy audit

```text
best_proxy = B6-update_cosine_to_adam
best_auc_bad_event = 0.8983050847457628
best_leaveout_min_auc = 0.25
best_false_positive_controls = 0.05555555555555555
line_b_gate_pass = 0
line_b_route = R5-StabilityProxyUnobservable
```

| proxy | auc bad | auc tail | auc linec | control FPR | precision top20 |
|---|---:|---:|---:|---:|---:|
| B1-split_loss_disagreement | 0.3389830508474576 | 0.8371775793650794 | 0.5830239405965946 | 0.0 | 1.0 |
| B2-recovery_lag | 0.5 | 0.5 | 0.5 | 1.0 | 1.0 |
| B3-logit_rms_drift | 0.3163841807909605 | 0.15699404761904762 | 0.42388938676225835 | 0.4861111111111111 | 1.0 |
| B4-entropy_collapse | 0.01694915254237288 | 0.2109375 | 0.5300217641787223 | 0.3333333333333333 | 0.9166666666666666 |
| B5-margin_p10_drift | 0.3502824858757062 | 0.18092757936507936 | 0.4501344258097555 | 0.4583333333333333 | 1.0 |
| B6-update_cosine_to_adam | 0.8983050847457628 | 0.7687251984126984 | 0.3442580975547305 | 0.05555555555555555 | 1.0 |
| B7-loss_q95_over_median | 0.3615819209039548 | 0.822296626984127 | 0.5931378824734349 | 0.0 | 1.0 |
| B8-split_consensus_eigengap | 0.5 | 0.5 | 0.5 | 1.0 | 1.0 |
| B9-degree_energy_drift | 0.3446327683615819 | 0.2693452380952381 | 0.3460504416848035 | 0.4027777777777778 | 1.0 |
| B10-projection_retention_drift | 0.8926553672316384 | 0.3453621031746032 | 0.5087696837792857 | 0.3055555555555556 | 1.0 |

Line B leaveout rows = 260

## 6. Line N/Q stability-nullspace FU 结果

```text
candidate_count = 12
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = 0.46147918701171875
control_equivalent_fraction = 0.0
bad_event_fraction = 1.0
best_method = N0-G7R-Replay
line_nq_exploration_gate_pass = 0
best_hazard_overlap_mean = 0.0
best_source_retention_after_null_mean = 1.0
best_tail_fail_fraction = 1.0
best_linec_fail_fraction = 0.7777777777777778
```

| method | rows | pass | source | control equiv | bad event | tail | linec | hazard | retention | bad reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N0-G7R-Replay | 9 | 0/9 | 0.46147918701171875 | 0.0 | 1.0 | 1.0 | 0.7777777777777778 | 0.0 | 1.0 | 0.0 |
| N1-LossQuantileNull | 9 | 0/9 | -1.1610007286071777 | 1.0 | 1.0 | 1.0 | 0.7777777777777778 | 0.7813852760526869 | 0.345229131480058 | 0.0 |
| N2-LogitRMSNull | 9 | 0/9 | -0.6269314024183485 | 1.0 | 1.0 | 0.4444444444444444 | 0.5555555555555556 | 0.9554226795832316 | 0.045955720047156014 | 0.0 |
| N3-EntropyRecoveryNull | 9 | 0/9 | -0.1916492912504408 | 1.0 | 1.0 | 0.1111111111111111 | 0.4444444444444444 | 0.9137168526649475 | 0.09994600216547649 | 0.0 |
| N4-DegreeEnergyNull | 9 | 0/9 | -1.1527938048044841 | 1.0 | 1.0 | 1.0 | 0.6666666666666666 | 0.8310294681125217 | 0.28771906594435376 | 0.0 |
| N5-CombinedHazardNull-rank4 | 9 | 0/9 | -0.6376314957936605 | 1.0 | 1.0 | 0.4444444444444444 | 0.5555555555555556 | 0.958784368303087 | 0.04326781930608882 | 0.0 |
| N6-CombinedHazardNull-rank8 | 9 | 0/9 | -0.5494010183546278 | 1.0 | 1.0 | 0.2222222222222222 | 0.5555555555555556 | 0.972361660665936 | 0.02992668603029516 | 0.0 |
| Q1-SourceRetainingLossQuantileConstraint | 9 | 0/9 | -0.11434917979770237 | 0.8888888888888888 | 1.0 | 1.0 | 0.2222222222222222 | 0.4981648094124264 | 0.6068678895632426 | 0.0 |
| Q2-SourceRetainingLogitEntropyConstraint | 9 | 0/9 | 0.22676610284381443 | 0.0 | 1.0 | 0.6666666666666666 | 0.7777777777777778 | 0.40034615993499756 | 0.6000000238418579 | 0.0 |
| Q3-SourceRetainingDegreeEnergyConstraint | 9 | 0/9 | 0.019044140974680584 | 0.4444444444444444 | 1.0 | 1.0 | 0.2222222222222222 | 0.5065517773230871 | 0.6000000370873345 | 0.0 |
| Q4-SourceRetainingCombinedConstraint-r4 | 9 | 0/9 | 0.22781628370285034 | 0.0 | 1.0 | 0.6666666666666666 | 0.7777777777777778 | 0.40044379896587795 | 0.6000000105963813 | 0.0 |
| Q5-SourceRetainingCombinedConstraint-r8 | 9 | 0/9 | 0.3111630810631646 | 0.0 | 1.0 | 0.7777777777777778 | 1.0 | 0.4007369743453132 | 0.6000000172191196 | 0.0 |

Line N/Q fallback/exhaustion：

```text
fallback_rows = 72
fallback_line_counts = NQ-FB1=12, NQ-FB2=12, NQ-FB3=12, NQ-FB4=12, NQ-FB5=12, NQ-FB6=12
exhaustion_certificate_rows = 1
```
```text
main_surface_executed = 1
fallback_ladder_executed = 1
controls_executed = 1
failure_taxonomy_complete = 1
consumed_budget = 180
final_stop_allowed = 1
```

## 6. Line P / M / X / D 结果

```text
line_p_rows = 45
Line P failure classes = P3-MicroHorizonGoodRealBad=34, P4-ControlEquivalentPath=11
line_m_rows = 72
generic_metric_stability_explains = 0
kan_specific_pass_count = 6
line_x_rows = 540
Line X failure classes = X-NoPositiveSource=414, X-LocalPositiveNoTransfer=86, X-TransferSupported=40
line_d_source = v156_actual_v149_substrate_acceleration
line_d_rows = 126
line_d_best_non_dche_family = D-FOU
line_d_best_non_dche_dataset_seed_pass_count = 0 / 9
```

Line M KAN-specific delta：

| dataset | seed | best D-CHE | best MLP | D-CHE gain | MLP gain | delta | generic explains | KAN-specific pass |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | 0 | N0-G7R-Replay | M5-MLP-SameTrustScalarControl | 0.4166508913040161 | 0.18532490730285645 | 0.23132598400115967 | 0 | 1 |
| Fashion-MNIST | 1 | N0-G7R-Replay | M5-MLP-SameTrustScalarControl | 0.42288124561309814 | 0.2431725263595581 | 0.17970871925354004 | 0 | 1 |
| Fashion-MNIST | 2 | N0-G7R-Replay | M5-MLP-SameTrustScalarControl | 0.4217958450317383 | 0.43658220767974854 | -0.014786362648010254 | 1 | 0 |
| KMNIST | 0 | N0-G7R-Replay | M5-MLP-SameTrustScalarControl | 0.4872676134109497 | 0.3588491678237915 | 0.1284184455871582 | 0 | 1 |
| KMNIST | 1 | N0-G7R-Replay | M5-MLP-SameTrustScalarControl | 0.3674060106277466 | 1.0444484949111938 | -0.6770424842834473 | 1 | 0 |
| KMNIST | 2 | N0-G7R-Replay | M5-MLP-SameTrustScalarControl | 0.4051225185394287 | 0.7388170957565308 | -0.33369457721710205 | 1 | 0 |
| MNIST | 0 | N0-G7R-Replay | M5-MLP-SameTrustScalarControl | 0.557703971862793 | 0.29002076387405396 | 0.267683207988739 | 0 | 1 |
| MNIST | 1 | N0-G7R-Replay | M5-MLP-SameTrustScalarControl | 0.515714168548584 | 0.2301355004310608 | 0.2855786681175232 | 0 | 1 |
| MNIST | 2 | N0-G7R-Replay | M5-MLP-SameTrustScalarControl | 0.5587704181671143 | 0.4656698703765869 | 0.09310054779052734 | 0 | 1 |

Line D summary：

| family | rows | pass | best candidate | max mean delta vs MLP | official eligibility |
|---|---:|---:|---|---:|---:|
| D-FOU | 45 | 0/9 | D-FOU68-BandwiseConsensusMetricV2 | -0.109375 | 0 |
| D-RBF | 45 | 0/9 | D-RBF68-GaussianLocalK4NoDenseMaterializationV2 | -0.328125 | 0 |
| D-WAV | 36 | 0/9 | D-WAV59-SupportOverlapDampingV3 | -0.0234375 | 0 |

Line D family failure taxonomy：

| family | failure class | fallback executed | official FU proof | deferred reason |
|---|---|---:|---:|---|
| D-FOU | D-SubstrateBelow6of9 | 1 | 0 | family substrate gate below FU eligibility |
| D-RBF | D-SubstrateBelow6of9 | 1 | 0 | family substrate gate below FU eligibility |
| D-WAV | D-SubstrateBelow6of9 | 1 | 0 | family substrate gate below FU eligibility |

## 7. 最终 route / 覆盖复核

```text
route = R4-LineCOrTailDominated
minimum_success = S1-StabilityNullspaceObservable
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
contract_unclosed_rows = 0
deep_coverage_unclosed_rows = 0
```

| gate or route | pass | route consistent | details |
|---|---:|---:|---|
| S1-StabilityNullspaceObservable | 1 | 1 | source=0.46147918701171875;bad=1.0;hazard=0.956108654302264;retention=0.04575427500383249;B=0;D=0/9 |
| S2-StabilityDecouplingExploration | 0 | 1 | source=0.46147918701171875;bad=1.0;hazard=0.956108654302264;retention=0.04575427500383249;B=0;D=0/9 |
| S3-MeaningfulStabilizedPositive | 0 | 1 | source=0.46147918701171875;bad=1.0;hazard=0.956108654302264;retention=0.04575427500383249;B=0;D=0/9 |
| S4-RealTransferExploration | 0 | 1 | source=0.46147918701171875;bad=1.0;hazard=0.956108654302264;retention=0.04575427500383249;B=0;D=0/9 |
| S5-OfficialFunctionalSuccess | 0 | 1 | source=0.46147918701171875;bad=1.0;hazard=0.956108654302264;retention=0.04575427500383249;B=0;D=0/9 |
| R-U-SourceHazardColinear | 1 | 1 | source=0.46147918701171875;bad=1.0;hazard=0.956108654302264;retention=0.04575427500383249;B=0;D=0/9 |
| R4-LineCOrTailDominated | 1 | 1 | source=0.46147918701171875;bad=1.0;hazard=0.956108654302264;retention=0.04575427500383249;B=0;D=0/9 |
| R5-StabilityProxyUnobservable | 1 | 1 | source=0.46147918701171875;bad=1.0;hazard=0.956108654302264;retention=0.04575427500383249;B=0;D=0/9 |
| R6-AllBasisSubstrateBlocked | 1 | 1 | source=0.46147918701171875;bad=1.0;hazard=0.956108654302264;retention=0.04575427500383249;B=0;D=0/9 |

## 8. 科学结论

```text
1. v15.6 已执行 Line R/S/U/N/Q/B/C/P/D/M/X/Z，并生成 required artifacts。
2. N/Q hazard-null projection 只使用 train-stream tangent/proxy；没有用 LineC/tail/AUC/calibration 反推方向。
3. 当前 route = R4-LineCOrTailDominated，promotion_allowed = 0。
4. MLP/generic controls、substrate-only rows 与 no-regression monitor 不写成 KAN-specific promotion。
5. 若 S5 未达成，v15.6 内不允许新增 G9/G10、action bank、controller 或 reset route。
```

## 9. 覆盖反方复核

本节由 runner 从 artifact 自动写入；用于确认完整计划的 surface / figures / taxonomy / controls 没有漏项。

```text
1. required figures 覆盖：计划要求 11 张指定 figures，runner 已按计划生成并纳入 manifest/contract。
   figures_present = 11 / 11
   required_artifact_manifest_rows = 43, missing = 0
   required_figures_contract_status = 1
2. Line P taxonomy：使用计划 P1..P7 taxonomy，并保留 P0-OK 表示未触发失败类。
   Line P failure classes = P3-MicroHorizonGoodRealBad=34, P4-ControlEquivalentPath=11
3. 复核 positive-looking rows matched controls：D-CHE controls 与 MLP/generic controls 均已覆盖 9 个 dataset/seed key。
4. 修复 Q telemetry 后，Q1..Q5 的 source_retention_after_null 表示实际提交 update；full-null retention 保留在 *_full_null 字段。
   Q full-null retention failed rows = 40
   Q source-retention repaired rows = 40
5. 修复 Line U readback 后，N0/G7R 也使用 combined-hazard basis 分解，避免空 basis 低估 source-hazard overlap。
   source_hazard_colinear_rows = 63 / 63
   mean_hazard_overlap = 0.956108654302264
   mean_source_retention_after_null = 0.04575427500383249
6. 覆盖复核不改变 N/Q/B/D/M/S 训练方向，不使用 audit metric 构造方向。
```

修复后最终判断：

```text
route = R4-LineCOrTailDominated
minimum_success = S1-StabilityNullspaceObservable
promotion_allowed = 0
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = 0.46147918701171875
bad_event_fraction = 1.0
line_b_best_proxy = B6-update_cosine_to_adam
line_b_gate_pass = 0
line_d_best_non_dche_dataset_seed_pass_count = 0 / 9
v15.6 计划内可执行分支已闭合；继续需要下一版 theory-level stability definition 或 substrate/base-architecture 计划。
```
