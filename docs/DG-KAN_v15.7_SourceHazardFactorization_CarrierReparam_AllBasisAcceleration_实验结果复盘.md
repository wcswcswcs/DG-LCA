# DG-KAN v15.7 SourceHazardFactorization CarrierReparam AllBasisAcceleration 实验结果复盘

生成时间：2026-05-31（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 source-hazard factorization diagnostic、substrate-only rows 或 MLP/generic control 写成 promotion。

## 1. 计划理解

v15.7 的目标不是新增 action family，而是验证 v15.6 的 source/hazard 高重叠是否只是 metric artifact，并测试预注册 carrier reparameterization 是否能把 source-carrying tangent 与 hazard-carrying tangent 分离。

## 2. 本轮代码修改

新增：

```text
experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py
```

修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v15.7 substrate-only candidates:
  D-FOU72..76, D-RBF70..74, D-WAV61..64。
```

过程说明：

```text
1. G7R 复现 v15.04/v15.5 metric-only semantics。
2. U2 在 M0..M5 六种 metric/space 下读取 G7R/Q2/Q4/Q5/random/AdamW/MLP analog source-hazard colinearity。
3. LineC/CEp99/NLL/ECE/AUCtime/Brier 只用于 audit/gate/failure taxonomy，不进入方向。
4. K1..K5 为预注册 carrier reparameterization；不做 G9/G10/N7/Q6/action/controller/reset。
5. 所有训练/finalizer 命令均使用 --device cuda:0；cpu_offload_used=0。
6. 自查沿用 v15.5/v15.6 的 matched batch stream 约束：
   同一 dataset/seed/family 下 G7R/G7K/controls 使用 matched batch stream，
   random direction 单独使用 method-specific generator；修正后不沿用旧训练指标。
7. Line U2/B/P 只做 readback/audit，不提交 audit-directed update。
8. Line B proxy/leaveout 只作为 readback audit，不作为 gating direction。
9. Line P 使用计划 P1..P7/P0 failure taxonomy。
10. U2 的 Q2/Q4/Q5 仅作为 v15.6 source-retaining readback source；G promotion 只看 G7R/K1..K5。
11. 自查发现 U2 初版 gate 被 U4 random / U5 AdamW control support 打开，
    已修正为只有 U0/G7R 与 U1..U3 Q-source readback 才能打开 S2；control support 只保留为 audit。
```

<!-- V15.7_MANUAL_ANALYSIS_START -->
## 2.1 人工复核分析 / Insight

这段是人工复核分析，不是 Python 自动生成的指标结论；下方 Line S / U2 / K / G / B / M / D / route 数据仍由 runner 从 artifact 写入。

本轮最重要的自查修复是 U2 gate 语义。初版 finalizer 把 U4 RandomSameSourceNorm 与 U5 AdamW control 的 decoupling support 也计入 S2，导致 route 一度被 control-only support 打开为 `S2-SourceHazardDecouplingObserved`。这不符合计划精神，因为 U4/U5 是对照，不是 source-carrying candidate。修复后只有 U0/G7R 与 U1..U3 Q-source readback eligible；结果是 `eligible_decomposition_support_rows = 0`，route 回到 `R4-StabilizationKillsSource`。这个修复只改 gate/route/docs，不改训练指标。

科学上看，v15.7 把 v15.6 的结论进一步收紧了：在 M0..M5 六种 metric/space 下，真正的 source-carrying sources U0/G7R、U1/Q2、U2/Q4、U3/Q5 都没有出现合法 decoupling support；它们的平均 hazard overlap 约 0.97-0.98，source retention after null 只有约 0.02-0.034。也就是说，高重叠不是单一 AdamV/DegreeRole metric 的 artifact。

K carrier 也没有解决问题。K1/K3/K5 基本保留 G7R update，但因此也保留 hazard 与 bad events；K2 明显降低 source retention 并把 source 变负；K4 保留约 0.64 source-retention，但 source 仍为负且 LineC/tail 仍失败。当前 K1..K5 没有找到“source 保留、hazard 降低、bad event 降低”的可行点。

有趣的是，U4 random 与 U5 AdamW control 在 U2 readback 上出现 decoupling support，说明 metric-space 本身不是完全无法表达 decoupling；失败点更像是“G7R/Q source-carrying direction 仍被绑定在 hazard tangent 上”，而不是所有方向都必然 colinear。这也解释了为什么 Line M 仍有 6/9 KAN-specific delta 为正，但 G/K 的 official source_vs_best_control 是 0 且 control-equivalent=1：收益信号存在于 replay/gain 对比里，但 carrier FU 没能形成可 promotion 的非对照优势。

因此本版结论不是“完全没有信号”，而是“当前 carrier reparameterization 没有把 source 从 hazard 中剥离出来”。继续在 v15.7 内加 G9/G10、N7/Q6、action/controller/reset 会变成 audit-directed search；下一步需要新的 theory-level factorization 或 substrate/base-architecture 设计，而不是在当前 carrier surface 上继续补 token。
<!-- V15.7_MANUAL_ANALYSIS_END -->

## 2.2 完整计划执行对照（artifact 自动写入）

执行合同覆盖：

| contract item | status | details |
|---|---:|---|
| Line R provenance/no-action audit | 1 | audit files present |
| Line S stability surface | 1 | K=2/4/8 and batch=128/256 sensitivity |
| Line U2 source-hazard metric-space validation | 1 | M0..M5 over U0..U6 readback |
| Line K/G carrier factorization controls | 1 | K1..K5 + G7R/G0/G1/C0..C7 |
| Line B proxy audit | 1 | B1..B10 readback-only leaveout |
| Line P micro-horizon audit | 1 | top methods + controls |
| Line M generic controls | 1 | MLP/generic controls |
| Line D all-basis + monitors | 1 | v15.7 substrate candidates + no-regression |
| Required figures | 1 | figures=10 |
| No forbidden continuation | 1 | no G9/G10/action/controller/reset/audit-directed branch |

深度覆盖审计：

| audit item | status | details |
|---|---:|---|
| Line K/G exact surface | 1 | G7R/K1..K5/U2 readback/C0..C7 present |
| Line B exact surface | 1 | B1..B10 present |
| Line D exact surface | 1 | D-FOU/RBF/WAV candidates present |
| Direction provenance train-stream-only | 1 | direction_rows=1638 |
| No remaining legal v15.7 continuation | 1 | if no S5, continuation requires next theory/substrate plan; no G9/G10/action/controller/reset |

required / forbidden / no-action / provenance：

```text
required_artifact_manifest_rows = 22
required_artifact_missing_rows = 0
forbidden_information_audit_rows = 12
forbidden_information_violation_sum = 0.0
no_action_search_audit_rows = 12
no_action_search_violation_sum = 0.0
direction_provenance_rows = 1638
direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction
method_surface_rows = 43
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

## 4. Line U2 source-hazard metric-space colinearity

```text
line_u2_rows = 378
source_hazard_colinear_rows = 275
decomposition_support_rows = 100
eligible_decomposition_support_rows = 0
mean_hazard_overlap = 0.7798357372044058
mean_source_retention_after_null = 0.2547888235569247
```

| source | metric rows | hazard overlap | source retention | support rows | colinear rows |
|---|---:|---:|---:|---:|---:|
| U0 G7R-original-replay | 54 | 0.9800781298566748 | 0.02022690376290686 | 0 | 54 |
| U1 Q2-SourceRetainingLogitEntropyConstraint | 54 | 0.980966752326047 | 0.019371475361333945 | 0 | 54 |
| U2 Q4-SourceRetainingCombinedConstraint-r4 | 54 | 0.9807703351532971 | 0.0195556350212832 | 0 | 54 |
| U3 Q5-SourceRetainingCombinedConstraint-r8 | 54 | 0.9735685178527126 | 0.034100033058267504 | 0 | 54 |
| U4 C1-RandomSubspaceSameRank | 54 | 0.0071097214154347225 | 0.9999322653920563 | 54 | 0 |
| U5 G0-D-CHE-AdamW | 54 | 0.5414301264617178 | 0.685259434360045 | 46 | 5 |
| U6 M3-MLP-SameSplitConsensusMetric | 54 | 0.9949265773649569 | 0.005076017942580317 | 0 | 54 |

## 5. Line B train-stream bad-event proxy audit

```text
best_proxy = B1-split_loss_disagreement
best_auc_bad_event = 0.5
best_leaveout_min_auc = 0.5
best_false_positive_controls = 0.05555555555555555
line_b_gate_pass = 0
line_b_route = LineB-ProxyRobustnessFail
```

| proxy | auc bad | auc tail | auc linec | control FPR | precision top20 |
|---|---:|---:|---:|---:|---:|
| B1-split_loss_disagreement | 0.5 | 0.8659906823787339 | 0.6949830717143737 | 0.05555555555555555 | 1.0 |
| B2-recovery_lag | 0.5 | 0.5 | 0.5 | 1.0 | 1.0 |
| B3-logit_rms_drift | 0.5 | 0.17141682652781584 | 0.20806401969836874 | 0.3888888888888889 | 1.0 |
| B4-entropy_collapse | 0.5 | 0.35927651411345574 | 0.3261003385657125 | 0.3 | 1.0 |
| B5-margin_p10_drift | 0.5 | 0.20704302548643463 | 0.24684518313327178 | 0.3888888888888889 | 1.0 |
| B6-update_cosine_to_adam | 0.5 | 0.5135653603727048 | 0.18898122499230532 | 0.3888888888888889 | 1.0 |
| B7-loss_q95_over_median | 0.5 | 0.8566730611126336 | 0.7023699599876885 | 0.05555555555555555 | 1.0 |
| B8-split_consensus_eigengap | 0.5 | 0.5 | 0.5 | 1.0 | 1.0 |
| B9-degree_energy_drift | 0.5 | 0.15031515483694163 | 0.2146814404432133 | 0.3888888888888889 | 1.0 |
| B10-projection_retention_drift | 0.5 | 0.5430254864346397 | 0.4390581717451524 | 0.28888888888888886 | 1.0 |

Line B leaveout rows = 250

## 6. Line K / G carrier factorization FU 结果

Line K carrier readback：

| method | rows | hazard overlap | source retention vs G7R | source | bad event | carrier gate |
|---|---:|---:|---:|---:|---:|---:|
| G7K1-LowDegreeSignalCarrier | 9 | 0.9388772282335494 | 1.0 | 0.0 | 1.0 | 0 |
| G7K2-ReadoutBasisDecoupledCarrier | 9 | 0.8440453145239089 | 0.13638209799925485 | -0.6939109298917983 | 1.0 | 0 |
| G7K3-OrthogonalDegreeBankCarrier | 9 | 0.9388772282335494 | 1.0 | 0.0 | 1.0 | 0 |
| G7K4-OutputJacobianCarrier | 9 | 0.969813883304596 | 0.6437858906057146 | -0.2580561505423652 | 1.0 | 0 |
| G7K5-DualBankCarrier | 9 | 0.9388772282335494 | 1.0 | 0.0 | 1.0 | 0 |

```text
candidate_count = 6
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = 0.0
control_equivalent_fraction = 1.0
bad_event_fraction = 1.0
best_method = G7K1-LowDegreeSignalCarrier
line_g_exploration_gate_pass = 0
best_hazard_overlap_mean = 0.9388772282335494
best_source_retention_after_null_mean = 0.062118563801050186
best_tail_fail_fraction = 1.0
best_linec_fail_fraction = 0.7777777777777778
```

| method | rows | pass | source | control equiv | bad event | tail | linec | hazard | retention | bad reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G7K1-LowDegreeSignalCarrier | 9 | 0/9 | 0.0 | 1.0 | 1.0 | 1.0 | 0.7777777777777778 | 0.9388772282335494 | 1.0 | 0.0 |
| G7K2-ReadoutBasisDecoupledCarrier | 9 | 0/9 | -0.6939109298917983 | 1.0 | 1.0 | 0.2222222222222222 | 0.6666666666666666 | 0.8440453145239089 | 0.13638209799925485 | 0.0 |
| G7K3-OrthogonalDegreeBankCarrier | 9 | 0/9 | 0.0 | 1.0 | 1.0 | 1.0 | 0.7777777777777778 | 0.9388772282335494 | 1.0 | 0.0 |
| G7K4-OutputJacobianCarrier | 9 | 0/9 | -0.2580561505423652 | 1.0 | 1.0 | 0.5555555555555556 | 1.0 | 0.969813883304596 | 0.6437858906057146 | 0.0 |
| G7K5-DualBankCarrier | 9 | 0/9 | 0.0 | 1.0 | 1.0 | 1.0 | 0.7777777777777778 | 0.9388772282335494 | 1.0 | 0.0 |
| G7R-original-replay | 9 | 0/9 | 0.0 | 1.0 | 1.0 | 1.0 | 0.7777777777777778 | 0.9388772282335494 | 1.0 | 0.0 |

Line G fallback/exhaustion：

```text
fallback_rows = 36
fallback_line_counts = G-FB1=6, G-FB2=6, G-FB3=6, G-FB4=6, G-FB5=6, G-FB6=6
exhaustion_certificate_rows = 1
```
```text
main_surface_executed = 1
fallback_ladder_executed = 1
controls_executed = 1
failure_taxonomy_complete = 1
consumed_budget = 171
final_stop_allowed = 1
```

## 6. Line P / M / X / D 结果

```text
line_p_rows = 54
Line P failure classes = P3-MicroHorizonGoodRealBad=31, P4-ControlEquivalentPath=14, P2-HazardNotRemoved=9
line_m_rows = 63
generic_metric_stability_explains = 0
kan_specific_pass_count = 6
line_x_rows = 513
Line X failure classes = X-NoPositiveSource=513
line_d_source = v157_actual_v149_substrate_acceleration
line_d_rows = 126
line_d_best_non_dche_family = D-FOU
line_d_best_non_dche_dataset_seed_pass_count = 0 / 9
```

Line M KAN-specific delta：

| dataset | seed | best D-CHE | best MLP | D-CHE gain | MLP gain | delta | generic explains | KAN-specific pass |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | 0 | G7R-original-replay | M4-MLP-SameTrustScalar | 0.4166508913040161 | 0.18532490730285645 | 0.23132598400115967 | 0 | 1 |
| Fashion-MNIST | 1 | G7R-original-replay | M4-MLP-SameTrustScalar | 0.42288124561309814 | 0.2431725263595581 | 0.17970871925354004 | 0 | 1 |
| Fashion-MNIST | 2 | G7R-original-replay | M4-MLP-SameTrustScalar | 0.4217958450317383 | 0.43658220767974854 | -0.014786362648010254 | 1 | 0 |
| KMNIST | 0 | G7R-original-replay | M4-MLP-SameTrustScalar | 0.4872676134109497 | 0.3588491678237915 | 0.1284184455871582 | 0 | 1 |
| KMNIST | 1 | G7R-original-replay | M4-MLP-SameTrustScalar | 0.3674060106277466 | 1.0444484949111938 | -0.6770424842834473 | 1 | 0 |
| KMNIST | 2 | G7R-original-replay | M4-MLP-SameTrustScalar | 0.4051225185394287 | 0.7388170957565308 | -0.33369457721710205 | 1 | 0 |
| MNIST | 0 | G7R-original-replay | M4-MLP-SameTrustScalar | 0.557703971862793 | 0.29002076387405396 | 0.267683207988739 | 0 | 1 |
| MNIST | 1 | G7R-original-replay | M4-MLP-SameTrustScalar | 0.515714168548584 | 0.2301355004310608 | 0.2855786681175232 | 0 | 1 |
| MNIST | 2 | G7R-original-replay | M4-MLP-SameTrustScalar | 0.5587704181671143 | 0.4656698703765869 | 0.09310054779052734 | 0 | 1 |

Line D summary：

| family | rows | pass | best candidate | max mean delta vs MLP | official eligibility |
|---|---:|---:|---|---:|---:|
| D-FOU | 45 | 0/9 | D-FOU73-BandwiseConsensusMetricV2 | -0.109375 | 0 |
| D-RBF | 45 | 0/9 | D-RBF73-GaussianLocalK4TaskHealthV2 | -0.328125 | 0 |
| D-WAV | 36 | 0/9 | D-WAV63-SupportOverlapDampingV3 | -0.0234375 | 0 |

Line D family failure taxonomy：

| family | failure class | fallback executed | official FU proof | deferred reason |
|---|---|---:|---:|---|
| D-FOU | D-SubstrateBelow6of9 | 1 | 0 | family substrate gate below FU eligibility |
| D-RBF | D-SubstrateBelow6of9 | 1 | 0 | family substrate gate below FU eligibility |
| D-WAV | D-SubstrateBelow6of9 | 1 | 0 | family substrate gate below FU eligibility |

## 7. 最终 route / 覆盖复核

```text
route = R4-StabilizationKillsSource
minimum_success = S1-SourceSignalObservable
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
| S1-SourceSignalObservable | 1 | 1 | source=0.0;bad=1.0;hazard=0.7798357372044058;retention=0.2547888235569247;B=0;D=0/9 |
| S2-SourceHazardDecouplingObserved | 0 | 1 | source=0.0;bad=1.0;hazard=0.7798357372044058;retention=0.2547888235569247;B=0;D=0/9 |
| S3-StabilizedFunctionalExplorationPositive | 0 | 1 | source=0.0;bad=1.0;hazard=0.7798357372044058;retention=0.2547888235569247;B=0;D=0/9 |
| S4-RealTransferExplorationPositive | 0 | 1 | source=0.0;bad=1.0;hazard=0.7798357372044058;retention=0.2547888235569247;B=0;D=0/9 |
| S5-OfficialFunctionalSuccess | 0 | 1 | source=0.0;bad=1.0;hazard=0.7798357372044058;retention=0.2547888235569247;B=0;D=0/9 |
| R1-MetricArtifactColinearity | 0 | 1 | source=0.0;bad=1.0;hazard=0.7798357372044058;retention=0.2547888235569247;B=0;D=0/9 |
| R2-SourceHazardColinearCurrentCarrier | 1 | 1 | source=0.0;bad=1.0;hazard=0.7798357372044058;retention=0.2547888235569247;B=0;D=0/9 |
| R3-CarrierFactorizationFail | 1 | 1 | source=0.0;bad=1.0;hazard=0.7798357372044058;retention=0.2547888235569247;B=0;D=0/9 |
| R4-StabilizationKillsSource | 1 | 1 | source=0.0;bad=1.0;hazard=0.7798357372044058;retention=0.2547888235569247;B=0;D=0/9 |
| R5-HazardRetainedWithSource | 0 | 1 | source=0.0;bad=1.0;hazard=0.7798357372044058;retention=0.2547888235569247;B=0;D=0/9 |
| LineB-ProxyRobustnessFail | 1 | 1 | source=0.0;bad=1.0;hazard=0.7798357372044058;retention=0.2547888235569247;B=0;D=0/9 |
| R6-AllBasisCarrierBlocked | 1 | 1 | source=0.0;bad=1.0;hazard=0.7798357372044058;retention=0.2547888235569247;B=0;D=0/9 |

## 8. 科学结论

```text
1. v15.7 已执行 Line R/S/U2/K/G/B/C/P/D/M/X/Z，并生成 required artifacts。
2. U2/K/G carrier factorization 只使用 train-stream tangent/proxy；没有用 LineC/tail/AUC/calibration 反推方向。
3. 当前 route = R4-StabilizationKillsSource，promotion_allowed = 0。
4. MLP/generic controls、substrate-only rows 与 no-regression monitor 不写成 KAN-specific promotion。
5. 若 S5 未达成，v15.7 内不允许新增 G9/G10、action bank、controller 或 reset route。
```

## 9. 覆盖反方复核

本节由 runner 从 artifact 自动写入；用于确认完整计划的 surface / figures / taxonomy / controls 没有漏项。

```text
1. required figures 覆盖：计划要求 10 张指定 figures，runner 已按计划生成并纳入 manifest/contract。
   figures_present = 10 / 10
   required_artifact_manifest_rows = 22, missing = 0
   required_figures_contract_status = 1
2. Line P taxonomy：使用计划 P1..P7 taxonomy，并保留 P0-OK 表示未触发失败类。
   Line P failure classes = P3-MicroHorizonGoodRealBad=31, P4-ControlEquivalentPath=14, P2-HazardNotRemoved=9
3. 复核 positive-looking rows matched controls：D-CHE controls 与 MLP/generic controls 均已覆盖 9 个 dataset/seed key。
4. U2 中的 Q2/Q4/Q5 仅作为 v15.6 source-retaining readback source；G promotion 只看 G7R/K1..K5。
   Q full-null retention failed rows = 27
   Q source-retention repaired rows = 27
5. Line U2 在 M0..M5 六种 metric/space 下读取 source-hazard overlap。
   source_hazard_colinear_rows = 275 / 378
   mean_hazard_overlap = 0.7798357372044058
   mean_source_retention_after_null = 0.2547888235569247
6. 覆盖复核不改变 U2/K/G/B/D/M/S 训练方向，不使用 audit metric 构造方向。
```

修复后最终判断：

```text
route = R4-StabilizationKillsSource
minimum_success = S1-SourceSignalObservable
promotion_allowed = 0
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = 0.0
bad_event_fraction = 1.0
line_b_best_proxy = B1-split_loss_disagreement
line_b_gate_pass = 0
line_d_best_non_dche_dataset_seed_pass_count = 0 / 9
v15.7 计划内可执行分支已闭合；继续需要下一版 theory-level stability definition 或 substrate/base-architecture 计划。
```
