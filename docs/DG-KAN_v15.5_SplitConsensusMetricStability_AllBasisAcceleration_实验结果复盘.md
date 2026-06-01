# DG-KAN v15.5 SplitConsensusMetricStability AllBasisAcceleration 实验结果复盘

生成时间：2026-05-31（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 stability trust scalar diagnostic、substrate-only rows 或 MLP/generic control 写成 promotion。

## 1. 计划理解

v15.5 的目标不是新增 FU family，而是在 v15.04 的 G7 metric-only 正信号上加入 train-stream-only stability trust scalar，检验是否能保留 source gain 并降低 tail/LineC bad events。

## 2. 本轮代码修改

新增：

```text
experiments/run_v155_split_consensus_metric_stability_allbasis.py
```

修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v15.5 substrate-only candidates:
  D-FOU62..66, D-RBF60..64, D-WAV53..56。
```

过程说明：

```text
1. G7R 复现 v15.04 metric-only semantics。
2. G7S1..G7S5 只用当前 train stream split-loss、recovery-lag、logit/entropy、loss-quantile proxy 生成 trust scalar。
3. LineC/CEp99/NLL/ECE/AUCtime/Brier 只用于 audit/gate/failure taxonomy，不进入方向。
4. C6 SameTrustScalarRandomDirection 检查 trust scalar 本身是否解释收益。
5. 所有训练/finalizer 命令均使用 --device cuda:0；cpu_offload_used=0。
6. 自查发现初版把 minibatch generator 与 method 名字绑定，
   已修正为同一 dataset/seed/family 下 G7R/G7S/controls 使用 matched batch stream，
   random direction 单独使用 method-specific generator；修正后不沿用旧训练指标。
7. G7R replay 的 trust_scalar_applied 审计字段已修正为 0；G7S/C6/M4 才标记 trust applied。
8. 再次反方复核发现计划第 15 节要求 11 张指定 figure，初版 runner 只生成 6 张内部命名 figure；
   已补齐为计划指定的 11 张 required figures，并纳入 required manifest / contract。
9. 继续反方复核发现 Line P failure taxonomy 旧版使用内部名 P-B3/P-B5/P-OK，
   已改为计划第 11.4 节的 P1..P6 taxonomy，并保留 P0-OK 表示未触发失败类。
```

<!-- V15.5_MANUAL_ANALYSIS_START -->
## 2.1 人工复核分析 / Insight

这段是人工复核分析，不是 Python 自动生成的指标结论；下方 Line S / B / G / M / D / route 数据仍由 runner 从 artifact 写入。

本轮最重要的自查修复是 matched batch stream。初版 G7R/G7S 的 minibatch generator 跟 method 名字绑定，这会让 trust scalar 的效果混入不同 batch 序列的偶然性。修复后，同一 dataset/seed/family 下 G7R、G7S 和 controls 使用同一 batch stream，random direction 单独使用 method-specific generator。修复后的结论比初版更可信：best method 回到 G7R replay，说明当前这组 G7S trust scalar 没有真正改善 v15.04 的核心问题。

v15.5 不是完全没信号。Line S 仍然打开，K=4 主设置 best SNR=1.956，K=2 sensitivity best SNR=10.52；Line G 的 G7R source gain 也很强，mean source_vs_best_control=0.461。Line M 里 KAN-specific delta 有 6/9 为正，generic/MLP controls 没有把这个 gain 完全解释掉。所以“split-consensus metric-only 方向有真实可见收益”这个观察仍然成立。

但 v15.5 没有解决稳定性。G7S3 是唯一把 bad_event 从 9/9 降到 8/9 的 trust scalar，同时拿到 1/9 real-lite pass；代价是 source 从 G7R 的 0.461 降到 0.439，tail/LineC 仍分别是 8/9 和 7/9。G7S4/G7S5 更强地压低 trust scalar，source retention 只剩约 0.55 左右，tail 有下降但 LineC 仍高，整体 bad_event 仍没有过 weak gate。也就是说，train-stream trust scalar 目前只是在“削弱更新幅度”，没有找到能选择性保留 source gain 且避免 tail/LineC failure 的因果边界。

Line B 的结果也支持这个判断。B10 projection_retention_drift 对 bad_event 的总体 AUC=0.809，看起来有信号，但 leaveout min AUC 只有 0.25，说明这个 proxy 不稳健，不能作为 gating 依据。B1/B7 对 tail 的 AUC 很高，但对 bad_event/source 的整体解释不稳；B6 对 bad_event 有 0.676，但仍不足以通过 leaveout 和 control FPR 合同。因此不能用这些 proxy 反推新方向，也不能把它们升级成 controller。

我对 v15.5 的结论是：当前失败不是“没有收益”，而是“收益和不稳定性绑定”。G7 metric-only 继续带来 source gain；但是只靠 train-stream scalar damping 不能把 tail/LineC failure 解耦出来。下一步如果继续，应该避免再加 G9/G10 或 action/reset，而要提出一个更有结构的非 audit-directed 稳定性定义，或者换 substrate/base architecture，让 metric-only 的收益不要沿着同一条路径带来 tail/LineC 崩坏。
<!-- V15.5_MANUAL_ANALYSIS_END -->

## 2.2 完整计划执行对照（artifact 自动写入）

执行合同覆盖：

| contract item | status | details |
|---|---:|---|
| Line R provenance/no-action audit | 1 | audit files present |
| Line S stability surface | 1 | K=2/4/8 and batch=128/256 sensitivity |
| Line G G7R/G7S controls | 1 | G7R + G7S1..S5 + C0..C8 |
| Line B proxy audit | 1 | B1..B10 AUC and leaveout |
| Line P micro-horizon audit | 1 | top methods + controls |
| Line M generic controls | 1 | MLP/generic controls |
| Line D all-basis + monitors | 1 | v15.5 substrate candidates + no-regression |
| Required figures | 1 | figures=11 |
| No forbidden continuation | 1 | no G9/G10/action/controller/reset/audit-directed branch |

深度覆盖审计：

| audit item | status | details |
|---|---:|---|
| Line G exact surface | 1 | G7R/G7S/C0..C7 present |
| Line B exact surface | 1 | B1..B10 present |
| Line D exact surface | 1 | D-FOU/RBF/WAV candidates present |
| Direction provenance train-stream-only | 1 | direction_rows=1323 |
| No remaining legal v15.5 continuation | 1 | if no S5, continuation requires next theory/substrate plan |

required / forbidden / no-action / provenance：

```text
required_artifact_manifest_rows = 42
required_artifact_missing_rows = 0
forbidden_information_audit_rows = 10
forbidden_information_violation_sum = 0.0
no_action_search_audit_rows = 10
no_action_search_violation_sum = 0.0
direction_provenance_rows = 1323
direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction
method_surface_rows = 38
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

## 4. Line B train-stream bad-event proxy audit

```text
best_proxy = B10-projection_retention_drift
best_auc_bad_event = 0.8094262295081968
best_leaveout_min_auc = 0.25
best_false_positive_controls = 0.2638888888888889
line_b_gate_pass = 0
line_b_route = R5-StabilityProxyUnobservable
```

| proxy | auc bad | auc tail | auc linec | control FPR | precision top20 |
|---|---:|---:|---:|---:|---:|
| B1-split_loss_disagreement | 0.3155737704918033 | 0.8641975308641975 | 0.6123456790123457 | 0.0 | 0.9615384615384616 |
| B2-recovery_lag | 0.5 | 0.5 | 0.5 | 1.0 | 1.0 |
| B3-logit_rms_drift | 0.32991803278688525 | 0.15792181069958847 | 0.2771604938271605 | 0.3611111111111111 | 1.0 |
| B4-entropy_collapse | 0.07991803278688525 | 0.3531378600823045 | 0.39166666666666666 | 0.3333333333333333 | 0.8846153846153846 |
| B5-margin_p10_drift | 0.3422131147540984 | 0.17181069958847736 | 0.31666666666666665 | 0.375 | 1.0 |
| B6-update_cosine_to_adam | 0.6762295081967213 | 0.706275720164609 | 0.24012345679012345 | 0.3055555555555556 | 1.0 |
| B7-loss_q95_over_median | 0.35450819672131145 | 0.867798353909465 | 0.6299382716049383 | 0.0 | 0.9615384615384616 |
| B8-split_consensus_eigengap | 0.5 | 0.5 | 0.5 | 1.0 | 1.0 |
| B9-degree_energy_drift | 0.4036885245901639 | 0.12371399176954732 | 0.29259259259259257 | 0.3611111111111111 | 1.0 |
| B10-projection_retention_drift | 0.8094262295081968 | 0.5897633744855967 | 0.42685185185185187 | 0.2638888888888889 | 1.0 |

Line B leaveout rows = 200

## 5. Line G stability FU 结果

```text
candidate_count = 6
real_lite_pass_count = 1 / 9
source_vs_best_control_mean = 0.46147918701171875
control_equivalent_fraction = 0.0
bad_event_fraction = 1.0
best_method = G7R-D-CHE-MetricOnlyReplay
line_g_exploration_gate_pass = 0
best_tail_fail_fraction = 1.0
best_linec_fail_fraction = 0.7777777777777778
```

| method | rows | pass | source | control equiv | bad event | tail | linec | trust | source retention | bad reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G7R-D-CHE-MetricOnlyReplay | 9 | 0/9 | 0.46147918701171875 | 0.0 | 1.0 | 1.0 | 0.7777777777777778 | 1.0 | 1.0 | 0.0 |
| G7S1-D-CHE-SplitAgreementTrust | 9 | 0/9 | 0.3996993766890632 | 0.0 | 1.0 | 0.8888888888888888 | 0.8888888888888888 | 0.9163434529335708 | 0.8661265511827151 | 0.0 |
| G7S2-D-CHE-RecoveryLagTrust | 9 | 0/9 | 0.4571366177664863 | 0.0 | 1.0 | 1.0 | 0.7777777777777778 | 1.0 | 0.9905898914459121 | 0.0 |
| G7S3-D-CHE-LogitEntropyTrust | 9 | 1/9 | 0.4394976562923855 | 0.0 | 0.8888888888888888 | 0.8888888888888888 | 0.7777777777777778 | 0.9444337566732445 | 0.9523672327203457 | 0.11111111111111116 |
| G7S4-D-CHE-LossQuantileTrust | 9 | 0/9 | 0.2526608043246799 | 0.0 | 1.0 | 0.7777777777777778 | 0.7777777777777778 | 0.6206608813253198 | 0.5475020573750466 | 0.0 |
| G7S5-D-CHE-CombinedTrainStreamTrust | 9 | 0/9 | 0.24862919251124063 | 0.0 | 1.0 | 0.7777777777777778 | 0.7777777777777778 | 0.6155199559700205 | 0.5387657764616088 | 0.0 |

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
consumed_budget = 126
final_stop_allowed = 1
```

## 6. Line P / M / X / D 结果

```text
line_p_rows = 36
Line P failure classes = P5-ControlEquivalentPath=27, P3-TailProxySpike=9
line_m_rows = 63
generic_metric_stability_explains = 0
kan_specific_pass_count = 6
line_x_rows = 378
Line X failure classes = X-NoPositiveSource=216, X-LocalPositiveNoTransfer=118, X-TransferSupported=44
line_d_source = v155_actual_v149_substrate_acceleration
line_d_rows = 126
line_d_best_non_dche_family = D-FOU
line_d_best_non_dche_dataset_seed_pass_count = 0 / 9
```

Line M KAN-specific delta：

| dataset | seed | best D-CHE | best MLP | D-CHE gain | MLP gain | delta | generic explains | KAN-specific pass |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | 0 | G7R-D-CHE-MetricOnlyReplay | M4-MLP-SameTrustScalar | 0.4166508913040161 | 0.18532490730285645 | 0.23132598400115967 | 0 | 1 |
| Fashion-MNIST | 1 | G7R-D-CHE-MetricOnlyReplay | M4-MLP-SameTrustScalar | 0.42288124561309814 | 0.2431725263595581 | 0.17970871925354004 | 0 | 1 |
| Fashion-MNIST | 2 | G7R-D-CHE-MetricOnlyReplay | M4-MLP-SameTrustScalar | 0.4217958450317383 | 0.43658220767974854 | -0.014786362648010254 | 1 | 0 |
| KMNIST | 0 | G7R-D-CHE-MetricOnlyReplay | M4-MLP-SameTrustScalar | 0.4872676134109497 | 0.3588491678237915 | 0.1284184455871582 | 0 | 1 |
| KMNIST | 1 | G7R-D-CHE-MetricOnlyReplay | M4-MLP-SameTrustScalar | 0.3674060106277466 | 1.0444484949111938 | -0.6770424842834473 | 1 | 0 |
| KMNIST | 2 | G7R-D-CHE-MetricOnlyReplay | M4-MLP-SameTrustScalar | 0.4051225185394287 | 0.7388170957565308 | -0.33369457721710205 | 1 | 0 |
| MNIST | 0 | G7R-D-CHE-MetricOnlyReplay | M4-MLP-SameTrustScalar | 0.557703971862793 | 0.29002076387405396 | 0.267683207988739 | 0 | 1 |
| MNIST | 1 | G7R-D-CHE-MetricOnlyReplay | M4-MLP-SameTrustScalar | 0.515714168548584 | 0.2301355004310608 | 0.2855786681175232 | 0 | 1 |
| MNIST | 2 | G7R-D-CHE-MetricOnlyReplay | M4-MLP-SameTrustScalar | 0.5587704181671143 | 0.4656698703765869 | 0.09310054779052734 | 0 | 1 |

Line D summary：

| family | rows | pass | best candidate | max mean delta vs MLP | official eligibility |
|---|---:|---:|---|---:|---:|
| D-FOU | 45 | 0/9 | D-FOU63-BandwiseConsensusMetric | -0.109375 | 0 |
| D-RBF | 45 | 0/9 | D-RBF63-GaussianLocalK4NoDenseMaterialization | -0.328125 | 0 |
| D-WAV | 36 | 0/9 | D-WAV55-SupportOverlapDampingV2 | -0.0234375 | 0 |

Line D family failure taxonomy：

| family | failure class | fallback executed | official FU proof | deferred reason |
|---|---|---:|---:|---|
| D-FOU | D-SubstrateBelow6of9 | 1 | 0 | family substrate gate below FU eligibility |
| D-RBF | D-SubstrateBelow6of9 | 1 | 0 | family substrate gate below FU eligibility |
| D-WAV | D-SubstrateBelow6of9 | 1 | 0 | family substrate gate below FU eligibility |

## 7. 最终 route / 覆盖复核

```text
route = R4-LineCOrTailDominated
minimum_success = S1-StableSplitConsensusSubspaceObservable
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
| S1-StableSplitConsensusSubspaceObservable | 1 | 1 | source=0.46147918701171875;bad=1.0;B=0;D=0/9 |
| S2-StabilizedExplorationPositive | 0 | 1 | source=0.46147918701171875;bad=1.0;B=0;D=0/9 |
| S3-StabilizedMeaningfulPositive | 0 | 1 | source=0.46147918701171875;bad=1.0;B=0;D=0/9 |
| S4-StabilizedRealTransfer | 0 | 1 | source=0.46147918701171875;bad=1.0;B=0;D=0/9 |
| S5-OfficialFunctionalSuccess | 0 | 1 | source=0.46147918701171875;bad=1.0;B=0;D=0/9 |
| R4-LineCOrTailDominated | 1 | 1 | source=0.46147918701171875;bad=1.0;B=0;D=0/9 |
| R5-StabilityProxyUnobservable | 1 | 1 | source=0.46147918701171875;bad=1.0;B=0;D=0/9 |
| R6-AllBasisSubstrateBlocked | 1 | 1 | source=0.46147918701171875;bad=1.0;B=0;D=0/9 |

## 8. 科学结论

```text
1. v15.5 已执行 Line R/S/G/B/C/P/D/M/X/Z，并生成 required artifacts。
2. G7S trust scalar 只使用 train-stream proxy；没有用 LineC/tail/AUC/calibration 反推方向。
3. 当前 route = R4-LineCOrTailDominated，promotion_allowed = 0。
4. MLP/generic controls、substrate-only rows 与 no-regression monitor 不写成 KAN-specific promotion。
5. 若 S5 未达成，v15.5 内不允许新增 G9/G10、action bank、controller 或 reset route。
```

## 9. 用户再次追问后的反方复核与覆盖修复

本节由 runner 从 artifact 自动写入；本次复核不新增训练，只修复覆盖/日志层缺口。

```text
1. 修复 required figures 覆盖：计划第 15 节要求 11 张指定 figures，runner 已按计划生成并纳入 manifest/contract。
   figures_present = 11 / 11
   required_artifact_manifest_rows = 42, missing = 0
   required_figures_contract_status = 1
2. 修复 Line P taxonomy：旧版内部名 P-B3/P-B5/P-OK 已改为计划第 11.4 节 P1..P6 taxonomy，并保留 P0-OK 表示未触发失败类。
   Line P failure classes = P5-ControlEquivalentPath=27, P3-TailProxySpike=9
3. 复核 positive-looking rows matched controls：D-CHE controls 与 MLP/generic controls 均已覆盖 9 个 dataset/seed key。
4. 上述修复不改变 G/B/D/M/S 训练指标，不使用 audit metric 构造方向。
```

修复后最终判断：

```text
route = R4-LineCOrTailDominated
minimum_success = S1-StableSplitConsensusSubspaceObservable
promotion_allowed = 0
real_lite_pass_count = 1 / 9
source_vs_best_control_mean = 0.46147918701171875
bad_event_fraction = 1.0
line_b_best_proxy = B10-projection_retention_drift
line_b_gate_pass = 0
line_d_best_non_dche_dataset_seed_pass_count = 0 / 9
v15.5 计划内可执行分支已闭合；继续需要下一版 theory-level stability definition 或 substrate/base-architecture 计划。
```
