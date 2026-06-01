# DG-KAN v15.04 SplitConsensusSignalSubspace FunctionalUpdate AllBasisAcceleration 实验结果复盘

生成时间：2026-05-31（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 split-consensus diagnostic、substrate-only rows 或 MLP/generic control 写成 promotion。

## 1. 计划理解

v15.04/v15.4 的目标是先用当前 train stream 的多个 split 识别跨 split 一致 signal subspace，再让 ordinary gradient 在该 subspace / metric 中更新。

## 2. 本轮代码修改

新增：

```text
experiments/run_v154_split_consensus_signal_subspace_fu_allbasis.py
```

修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v15.4 substrate-only candidates:
  D-FOU57..61, D-RBF55..59, D-WAV49..52。
```

过程修正：

```text
首次 official finalizer 写入 route 前先生成 required manifest，
manifest 将 route/gate/no-go/contract 这些自举 artifact 暂计为 missing，
导致 route 一度错误写为 R0-ArtifactOrProvenanceViolation。
已修正 finalizer 自举顺序：先排除待生成自举 artifact 计算初始 route，
再写 route / gate recompute / no-go / contract / manifest，并重算最终 route。
该修正只影响 route 写入顺序，不改变任何训练指标。
修正后用 --reuse-if-present 1 重建 route / manifest / 两份日志。
所有实际训练与 finalizer 命令均使用 --device cuda:0。
再次复核 Line G surface manifest 时发现 C0/C1/C2/C8 控制面只以
G0/G1/G2 与 Line M rows 隐式存在，已补齐 C0..C8 显式 alias 映射。
该补齐只改覆盖审计表与日志，不新增训练、不改实验指标。
用户再次追问后复核 signal-subspace 实现，发现 S2/S3 低秩分支
初版只对 split gradients 做 SVD，未严格用 A_split=C_split-lambda*N_split
求正信号子空间；同时 S-FB1 只标记 K2/K4 sensitivity，未落实际 K2/K8 rows。
已修正为在 split-gradient span 内构造小型 A_split 特征分解，
并补齐 v154_line_s_k_sensitivity.csv 的 K=2/K=4/K=8 actual diagnostic。
修正后重跑 official GPU 矩阵，不沿用旧训练指标。
由于 K=2 diagnostic 出现 gate rows 而 K=4/K=8 primary 仍为 0，
最终执行 batch_size=256 的 GPU official run 作为 micro-split budget repair。
该修复不使用 validation/test/future/query 或 audit metric 生成方向。
再次复核发现 G7-MetricNoProjection 初版仍对 signal mask 做 soft attenuation，
已修正为真正 no-projection metric update；G3/G4/G5/G6/G8 统一使用
metric base update 后再做 signal projection。
该修复按计划区分 metric-only 与 projection value，不新增 G9/G10。
修复后再次用 --reuse-if-present 0 重跑 batch256 official GPU 矩阵。
最终复核发现独立 gate/route recompute 缺少 R4-LineCOrTailDominated 优先级，
已补齐 R4 recompute 条件并用 --reuse-if-present 1 重写 route / gate / docs。
用户再次追问后复核完整计划最低合同，发现 D-CHE/Rational no-regression monitor
未纳入 v15.04 required manifest 与 contract；已补齐两份 monitor artifact、
method surface manifest rows 和 contract item。该修复只补覆盖审计，不改变训练指标。
用户指出数据/分析边界后，runner 改为只自动写 artifact 数据，并保留手工分析 marker 区。
用户质疑复盘过薄后，runner 补齐完整计划执行对照、fallback ladder、gate recompute、
contract/deep coverage、Line M delta 与 Line D taxonomy 自动写入。
```

<!-- V15.04_MANUAL_ANALYSIS_START -->
## 2.1 人工复核分析 / Insight

这段是人工复核分析，不是 Python 自动生成的指标结论；下方 Line S / G / M / C / route 数据仍由 runner 从 artifact 写入。

我的判断是：v15.04 不是“idea 完全没信号”。自动数据段已经显示 split-consensus subspace 可观测，且最强的正向效果来自 G7 metric-only，而不是带 projection 的 variants。这说明当前方向里确实存在 transfer-visible 的 NLL gain；问题不是没有信号，而是这个信号没有被当前 functional definition 稳定地约束住。

更具体地说，projection variants 比 G7 metric-only 更弱这一点很重要：它提示“把 signal 子空间硬投影成更新方向”可能本身就在放大 tail / LineC 风险。这个判断不能反过来用于构造新 direction，因为计划明确禁止用 LineC/tail/AUC/calibration 做 audit-directed search；它只能作为下一版 theory-level 设计的线索。

因此当前结论应该写得更细：split-consensus 作为观测/度量是有价值信号的，generic/MLP controls 也没有把这个 gain 全部解释掉；但 v15.04 仍不能 promotion，因为收益进入了 tail/LineC 不稳定区域。下一步如果继续，核心不是再加 G9/G10 或 action/controller/reset，而是要提出一个非 audit-directed 的稳定性约束，或者换一个更底层的 substrate/base architecture，让 metric-only 的正向信号不要转化成 tail failure。
<!-- V15.04_MANUAL_ANALYSIS_END -->

## 2.2 完整计划执行对照（artifact 自动写入）

执行合同覆盖：

| contract item | status | details |
|---|---:|---|
| Line R provenance/no-action audit | 1 | audit files present |
| Line S subspace + fallbacks | 1 | S0/S1/S2/S3 + K2/K4/K8 + S-FB1..S-FB4 |
| Line G main/controls/fallback | 1 | G0..G8 + C0..C8 aliases + G-FB1..G-FB6 |
| Line P micro-horizon audit | 1 | top methods + controls |
| Line M generic controls | 1 | MLP/generic controls |
| Line D all-basis + no-regression monitors | 1 | D-FOU/RBF/WAV substrate only;monitors=1 |
| Required figures | 1 | figures=7 |
| Deep fallback/control coverage audit | 1 | S/G/P/C/D/M exact coverage rows present |
| Gate semantics audit | 1 | weak gate components and R4 condition recomputed |
| No forbidden continuation | 1 | no action/controller/reset/audit-direction branch |

深度覆盖审计：

| audit item | status | details |
|---|---:|---|
| Line S fallback exact coverage | 1 | present=S-FB1,S-FB2,S-FB3,S-FB4 |
| Line G fallback exact coverage | 1 | all G-FB1..G-FB6 per G3..G8 |
| Line G method/control surface exact coverage | 1 | missing_g=;missing_alias= |
| Line P top/control audit coverage | 1 | methods=C3-RandomSubspaceSameRank,C7-NoOpMatchedOverhead,G0-D-CHE-AdamW,G4-D-CHE-SplitConsensusRoleBlockMetric,G7-D-CHE-SplitConsensusMetricNoProjection |
| Line C failure taxonomy coverage | 1 | rows=972;failure_reason_complete=1;used_for_direction=0 |
| Line D family taxonomy coverage | 1 | families=D-FOU,D-RBF,D-WAV |
| Line M generic controls coverage | 1 | M0..M6 present |
| No-regression monitor coverage | 1 | D-CHE and D-RAT monitors present |
| Contract row status coverage | 1 | contract_rows=10 |
| No remaining legal v15.04 continuation | 1 | R4 requires audit-directed tail/LineC direction to continue; plan forbids G9/G10/action/controller/reset |

required / forbidden / no-action / provenance：

```text
required_artifact_manifest_rows = 36
required_artifact_missing_rows = 0
forbidden_information_audit_rows = 8
forbidden_information_violation_sum = 0.0
no_action_search_audit_rows = 8
no_action_search_violation_sum = 0.0
direction_provenance_rows = 2709
direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction
```

method surface manifest：

```text
method_surface_rows = 46
surface_role_counts = main_or_C0_C2_control=14, substrate_only_candidate=14, matched_control_alias=9, generic_control=7, no_regression_monitor=2
line_g_main_surface_missing_count = 0
line_g_control_alias_missing_count = 0
```

自动记录的过程修正：

```text
1. 初始实现完成后，先发现 finalizer 自举 manifest 会把尚未写出的 route/gate/contract
   计为 missing，导致训练 artifact 完整却 route=R0；修复后只改变审计写入顺序。
2. 之后复核控制面，发现 C0/C1/C2/C8 只有隐式映射，已补成显式 surface rows。
3. 再复核 signal-subspace 数学实现，发现低秩 S2/S3 需要在 split-gradient span 内
   对 A_split=C_split-lambda*N_split 做正特征子空间，而不是直接 SVD split gradients。
4. K sensitivity diagnostic 已补齐 K=2/K=4/K=8 actual rows；具体数值见 Line S 数据段。
5. 再复核 G7 发现 no-projection 版本仍有 soft attenuation；已修正为 metric-only update，
   并与 projection variants 分开统计；具体结果见 Line G 数据段。
6. 最后补齐 D-CHE/Rational no-regression monitor 与 deep coverage audit，确保停止不是覆盖漏项。
```

## 3. Line S 结果

```text
line_s_rows = 108
line_s_k_sensitivity_rows = 324
line_s_gate_pass = 1
line_s_best_snr = 1.9560231276069324
k2_best_signal_to_noise_ratio = 10.519598875774431
k4_best_signal_to_noise_ratio = 1.9560231276069324
k8_best_signal_to_noise_ratio = 0.49137829373764685
```

Line S fallback ladder：

| fallback | rows | K2 SNR | K4 SNR | K8 SNR | gate pass rows | certificate |
|---|---:|---:|---:|---:|---:|---:|
| S-FB1 | 108 | 10.519598875774431 | 1.9560231276069324 | 0.49137829373764685 | 6 | 0 |
| S-FB2 | 108 | 10.519598875774431 | 1.9560231276069324 | 0.49137829373764685 | 6 | 0 |
| S-FB3 | 108 | 10.519598875774431 | 1.9560231276069324 | 0.49137829373764685 | 6 | 0 |
| S-FB4 | 108 | 10.519598875774431 | 1.9560231276069324 | 0.49137829373764685 | 6 | 1 |

## 4. Line G 结果

```text
candidate_count = 18
real_lite_pass_count = 3 / 9
source_vs_best_control_mean = 0.4495026138093736
control_equivalent_fraction = 0.0
bad_event_fraction = 1.0
best_method = G7-D-CHE-SplitConsensusMetricNoProjection [M0-identity/S0-diagonal]
line_g_exploration_gate_pass = 0
best_tail_fail_fraction = 1.0
best_linec_fail_fraction = 0.6666666666666666
```

Method summary：

| method | rows | pass | source | control equiv | bad event | projection retention | SNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| G3-D-CHE-SplitConsensusDiagMetric [M0-identity/S0-diagonal] | 9 | 1/9 | 0.37763949235280353 | 0.0 | 1.0 | 0.8814818764944932 | 0.3403889854640307 |
| G3-D-CHE-SplitConsensusDiagMetric [M1-AdamVDiag/S0-diagonal] | 9 | 0/9 | -0.6570734050538805 | 1.0 | 1.0 | 0.7221075072766084 | 0.2580302752599955 |
| G3-D-CHE-SplitConsensusDiagMetric [M2-DegreeRoleSecondMoment/S0-diagonal] | 9 | 0/9 | -0.11331796646118164 | 0.6666666666666666 | 1.0 | 0.846166405607331 | 0.4410796511064702 |
| G4-D-CHE-SplitConsensusRoleBlockMetric [M0-identity/S1-role-block] | 9 | 2/9 | 0.4346072276433309 | 0.0 | 1.0 | 0.970505600905367 | 0.3918288803904928 |
| G4-D-CHE-SplitConsensusRoleBlockMetric [M1-AdamVDiag/S1-role-block] | 9 | 0/9 | -0.6802335182825724 | 1.0 | 1.0 | 0.3923535501813421 | 0.24735689695054283 |
| G4-D-CHE-SplitConsensusRoleBlockMetric [M2-DegreeRoleSecondMoment/S1-role-block] | 9 | 0/9 | -0.1626066631740994 | 0.8888888888888888 | 1.0 | 1.0 | 0.8063666091831122 |
| G5-D-CHE-SplitConsensusLowRank-r4 [M0-identity/S2-lowrank-r4] | 9 | 1/9 | 0.05855613284640842 | 0.0 | 1.0 | 0.6629681991755154 | 0.3974044283581352 |
| G5-D-CHE-SplitConsensusLowRank-r4 [M1-AdamVDiag/S2-lowrank-r4] | 9 | 0/9 | -0.4453839063644409 | 1.0 | 1.0 | 0.512694509445251 | 0.2424466426646121 |
| G5-D-CHE-SplitConsensusLowRank-r4 [M2-DegreeRoleSecondMoment/S2-lowrank-r4] | 9 | 0/9 | -0.6836266650093926 | 1.0 | 1.0 | 0.5222549598862863 | 0.25033762902814155 |
| G6-D-CHE-SplitConsensusLowRank-r8 [M0-identity/S3-lowrank-r8] | 9 | 1/9 | 0.04941464795006646 | 0.3333333333333333 | 1.0 | 0.6580419247056489 | 0.4011727654300487 |
| G6-D-CHE-SplitConsensusLowRank-r8 [M1-AdamVDiag/S3-lowrank-r8] | 9 | 0/9 | -0.4456138610839844 | 1.0 | 1.0 | 0.5150935317267652 | 0.24888219832924255 |
| G6-D-CHE-SplitConsensusLowRank-r8 [M2-DegreeRoleSecondMoment/S3-lowrank-r8] | 9 | 0/9 | -0.6748351653416952 | 1.0 | 1.0 | 0.5309170992954741 | 0.25572489793529285 |
| G7-D-CHE-SplitConsensusMetricNoProjection [M0-identity/S0-diagonal] | 9 | 3/9 | 0.4495026138093736 | 0.0 | 1.0 | 0.8769549880671013 | 0.44335903902170337 |
| G7-D-CHE-SplitConsensusMetricNoProjection [M1-AdamVDiag/S0-diagonal] | 9 | 0/9 | -0.6456518040763007 | 1.0 | 1.0 | 0.7137822920989131 | 0.23359587745154048 |
| G7-D-CHE-SplitConsensusMetricNoProjection [M2-DegreeRoleSecondMoment/S0-diagonal] | 9 | 0/9 | -0.022069699234432645 | 0.6666666666666666 | 1.0 | 0.8625112349145176 | 0.8467805053722675 |
| G8-D-CHE-SplitConsensusProjectionPlusAdamV [M0-identity/S0-diagonal] | 9 | 2/9 | 0.3856801721784804 | 0.0 | 1.0 | 0.8832627214128851 | 0.31061122524400314 |
| G8-D-CHE-SplitConsensusProjectionPlusAdamV [M1-AdamVDiag/S0-diagonal] | 9 | 0/9 | -0.6568552520540025 | 1.0 | 1.0 | 0.719329898497019 | 0.2542658506569324 |
| G8-D-CHE-SplitConsensusProjectionPlusAdamV [M2-DegreeRoleSecondMoment/S0-diagonal] | 9 | 0/9 | -0.10249135229322645 | 0.6666666666666666 | 1.0 | 0.8510359241555943 | 0.4437286080612954 |

Line G fallback ladder summary：

```text
fallback_rows = 36
fallback_line_counts = G-FB1=6, G-FB2=6, G-FB3=6, G-FB4=6, G-FB5=6, G-FB6=6
exhaustion_certificate_rows = 1
```

| fallback | method | projection kills | B2/B1 median | noise ratio | best metric source | control equiv | positive rows | step ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G-FB1 | G3-D-CHE-SplitConsensusDiagMetric | 0.5555555555555556 | 0.9731904698582636 | - | - | - | - | - |
| G-FB2 | G3-D-CHE-SplitConsensusDiagMetric | - | - | 2.4211097339500256 | - | - | - | - |
| G-FB3 | G3-D-CHE-SplitConsensusDiagMetric | - | - | - | 0.511093020439148 | - | - | - |
| G-FB4 | G3-D-CHE-SplitConsensusDiagMetric | - | - | - | - | 0.5555555555555556 | 12 | - |
| G-FB5 | G3-D-CHE-SplitConsensusDiagMetric | - | - | - | - | - | - | 1.0029898034320066 |
| G-FB6 | G3-D-CHE-SplitConsensusDiagMetric | - | - | - | - | - | - | - |
| G-FB1 | G4-D-CHE-SplitConsensusRoleBlockMetric | 0.6296296296296297 | 0.9525997005580509 | - | - | - | - | - |
| G-FB2 | G4-D-CHE-SplitConsensusRoleBlockMetric | - | - | 1.0934526978399195 | - | - | - | - |
| G-FB3 | G4-D-CHE-SplitConsensusRoleBlockMetric | - | - | - | 0.560836911201477 | - | - | - |
| G-FB4 | G4-D-CHE-SplitConsensusRoleBlockMetric | - | - | - | - | 0.6296296296296297 | 10 | - |
| G-FB5 | G4-D-CHE-SplitConsensusRoleBlockMetric | - | - | - | - | - | - | 1.0376812476124317 |
| G-FB6 | G4-D-CHE-SplitConsensusRoleBlockMetric | - | - | - | - | - | - | - |
| G-FB1 | G5-D-CHE-SplitConsensusLowRank-r4 | 0.6666666666666666 | 0.9876972268879992 | - | - | - | - | - |
| G-FB2 | G5-D-CHE-SplitConsensusLowRank-r4 | - | - | 3.336163751301054 | - | - | - | - |
| G-FB3 | G5-D-CHE-SplitConsensusLowRank-r4 | - | - | - | 0.12321352958679199 | - | - | - |
| G-FB4 | G5-D-CHE-SplitConsensusLowRank-r4 | - | - | - | - | 0.6666666666666666 | 9 | - |
| G-FB5 | G5-D-CHE-SplitConsensusLowRank-r4 | - | - | - | - | - | - | 1.1761203426544946 |
| G-FB6 | G5-D-CHE-SplitConsensusLowRank-r4 | - | - | - | - | - | - | - |
| G-FB1 | G6-D-CHE-SplitConsensusLowRank-r8 | 0.7037037037037037 | 0.956094366050802 | - | - | - | - | - |
| G-FB2 | G6-D-CHE-SplitConsensusLowRank-r8 | - | - | 3.2063083313279077 | - | - | - | - |
| G-FB3 | G6-D-CHE-SplitConsensusLowRank-r8 | - | - | - | 0.1517263650894165 | - | - | - |
| G-FB4 | G6-D-CHE-SplitConsensusLowRank-r8 | - | - | - | - | 0.7777777777777778 | 6 | - |
| G-FB5 | G6-D-CHE-SplitConsensusLowRank-r8 | - | - | - | - | - | - | 1.176847310166781 |
| G-FB6 | G6-D-CHE-SplitConsensusLowRank-r8 | - | - | - | - | - | - | - |
| G-FB1 | G7-D-CHE-SplitConsensusMetricNoProjection | 0.5555555555555556 | 0.976135233539466 | - | - | - | - | - |
| G-FB2 | G7-D-CHE-SplitConsensusMetricNoProjection | - | - | 1.086808154582286 | - | - | - | - |
| G-FB3 | G7-D-CHE-SplitConsensusMetricNoProjection | - | - | - | 0.5631861686706543 | - | - | - |
| G-FB4 | G7-D-CHE-SplitConsensusMetricNoProjection | - | - | - | - | 0.5555555555555556 | 12 | - |
| G-FB5 | G7-D-CHE-SplitConsensusMetricNoProjection | - | - | - | - | - | - | 1.0037120894011937 |
| G-FB6 | G7-D-CHE-SplitConsensusMetricNoProjection | - | - | - | - | - | - | - |
| G-FB1 | G8-D-CHE-SplitConsensusProjectionPlusAdamV | 0.5555555555555556 | 0.9896192946323492 | - | - | - | - | - |
| G-FB2 | G8-D-CHE-SplitConsensusProjectionPlusAdamV | - | - | 2.605426627661945 | - | - | - | - |
| G-FB3 | G8-D-CHE-SplitConsensusProjectionPlusAdamV | - | - | - | 0.509329080581665 | - | - | - |
| G-FB4 | G8-D-CHE-SplitConsensusProjectionPlusAdamV | - | - | - | - | 0.5555555555555556 | 12 | - |
| G-FB5 | G8-D-CHE-SplitConsensusProjectionPlusAdamV | - | - | - | - | - | - | 1.002581577944069 |
| G-FB6 | G8-D-CHE-SplitConsensusProjectionPlusAdamV | - | - | - | - | - | - | - |

Line G exhaustion certificate：

```text
main_surface_executed = 1
fallback_ladder_executed = 1
controls_executed = 1
failure_taxonomy_complete = 1
consumed_budget = 324
final_stop_allowed = 1
```

## 5. Line P / M / D 结果

```text
line_p_rows = 117
generic_split_consensus_explains = 0
kan_specific_pass_count = 6
line_d_best_non_dche_family = D-FOU
line_d_best_non_dche_dataset_seed_pass_count = 0 / 9
v154_dche_no_regression_monitor.csv rows = 1
v154_rational_no_regression_monitor.csv rows = 8
monitor_promotion_allowed = 0
Line P failure classes = P-OK=48, P-B5-TailDominated=37, P-B3-NoiseSubspace=24, P-B2-ProjectionKillsTransfer=5, P-B1-LocalOnly=3
Line C fail reasons = linec,source=309, source=279, tail,source=105, linec,tail,source=96, linec,tail=89, tail=49, linec=36, none=9
```

Line D summary：

| family | rows | pass | best candidate | max mean delta vs MLP | official eligibility |
|---|---:|---:|---|---:|---:|
| D-FOU | 45 | 0/9 | D-FOU58-BandwiseSNRWarmupV3 | -0.109375 | 0 |
| D-RBF | 45 | 0/9 | D-RBF58-GaussianLocalK4NoDenseV3 | -0.328125 | 0 |
| D-WAV | 36 | 0/9 | D-WAV51-SupportOverlapDampingV3 | -0.0234375 | 0 |

Line M KAN-specific delta：

| dataset | seed | best D-CHE | best MLP | D-CHE gain | MLP gain | delta KAN-specific | generic explains | KAN-specific pass |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | 0 | G7-D-CHE-SplitConsensusMetricNoProjection | M4-MLP-SplitConsensusLowRank-r4 | 0.42302143573760986 | 0.19756275415420532 | 0.22545868158340454 | 0 | 1 |
| Fashion-MNIST | 1 | G7-D-CHE-SplitConsensusMetricNoProjection | M3-MLP-SplitConsensusDiagMetric | 0.4231250286102295 | 0.2571852207183838 | 0.1659398078918457 | 0 | 1 |
| Fashion-MNIST | 2 | G7-D-CHE-SplitConsensusMetricNoProjection | M4-MLP-SplitConsensusLowRank-r4 | 0.4127216339111328 | 0.4356881380081177 | -0.022966504096984863 | 1 | 0 |
| KMNIST | 0 | G4-D-CHE-SplitConsensusRoleBlockMetric | M4-MLP-SplitConsensusLowRank-r4 | 0.46503114700317383 | 0.43818098306655884 | 0.02685016393661499 | 0 | 1 |
| KMNIST | 1 | G4-D-CHE-SplitConsensusRoleBlockMetric | M3-MLP-SplitConsensusDiagMetric | 0.36638200283050537 | 1.2126801013946533 | -0.846298098564148 | 1 | 0 |
| KMNIST | 2 | G4-D-CHE-SplitConsensusRoleBlockMetric | M4-MLP-SplitConsensusLowRank-r4 | 0.40830373764038086 | 0.7936310768127441 | -0.3853273391723633 | 1 | 0 |
| MNIST | 0 | G7-D-CHE-SplitConsensusMetricNoProjection | M4-MLP-SplitConsensusLowRank-r4 | 0.5474544763565063 | 0.26033371686935425 | 0.2871207594871521 | 0 | 1 |
| MNIST | 1 | G7-D-CHE-SplitConsensusMetricNoProjection | M4-MLP-SplitConsensusLowRank-r4 | 0.5170896053314209 | 0.33735090494155884 | 0.17973870038986206 | 0 | 1 |
| MNIST | 2 | G7-D-CHE-SplitConsensusMetricNoProjection | M4-MLP-SplitConsensusLowRank-r4 | 0.5631861686706543 | 0.47636449337005615 | 0.08682167530059814 | 0 | 1 |

Line D family failure taxonomy：

| family | failure class | fallback executed | official FU proof | deferred reason |
|---|---|---:|---:|---|
| D-FOU | D-SubstrateBelow6of9 | 1 | 0 | family substrate gate below FU eligibility |
| D-RBF | D-SubstrateBelow6of9 | 1 | 0 | family substrate gate below FU eligibility |
| D-WAV | D-SubstrateBelow6of9 | 1 | 0 | family substrate gate below FU eligibility |

## 6. 最终 route

```text
route = R4-LineCOrTailDominated
minimum_success = S1-SplitConsensusSubspaceObservable
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 7. 覆盖复核

```text
line_g_main_surface_missing_count = 0
line_g_control_alias_missing_count = 0
line_g_control_aliases = C0,C1,C2,C3,C4,C5,C6,C7,C8
C8-MLP-SameSplitConsensusControl = M3/M4 Line M rows
D-CHE/D-RAT no-regression monitors = present
gate_route_inconsistent_rows = 0
contract_unclosed_rows = 0
deep_coverage_unclosed_rows = 0
gate_semantics_failed_components = 2
```

Gate / route independent recompute：

| gate or route | pass | route consistent | details |
|---|---:|---:|---|
| S1-SplitConsensusSubspaceObservable | 1 | 1 | expected_route=R4-LineCOrTailDominated;actual=R4-LineCOrTailDominated |
| S2-SplitConsensusFUExplorationPositive | 0 | 1 | expected_route=R4-LineCOrTailDominated;actual=R4-LineCOrTailDominated |
| S3-SplitConsensusFUMeaningfulPositive | 0 | 1 | expected_route=R4-LineCOrTailDominated;actual=R4-LineCOrTailDominated |
| S4-SplitConsensusFURealLitePositive | 0 | 1 | expected_route=R4-LineCOrTailDominated;actual=R4-LineCOrTailDominated |
| S5-OfficialFunctionalSuccess | 0 | 1 | expected_route=R4-LineCOrTailDominated;actual=R4-LineCOrTailDominated |
| R1-NoSplitConsensusSignalSubspace | 0 | 1 | best_snr=1.9560231276069324 |
| R2-SubspaceExistsProjectionKillsValue | 0 | 1 | source=0.4495026138093736;projection_kills=5;local_only=3 |
| R3-GenericControlsExplainSplitConsensus | 0 | 1 | fraction=0.3333333333333333 |
| R4-LineCOrTailDominated | 1 | 1 | source=0.4495026138093736;tail_fail=1.0;linec_fail=0.6666666666666666 |
| R5-AllBasisSubstrateBlocked | 1 | 1 | best=0/9 |
| R6-SplitConsensusFUCurrentDefinitionNoGo | 1 | 1 | source=0.4495026138093736;real_lite=3 |

Gate semantics audit：

| audit item | status | details |
|---|---:|---|
| best_method_selection | 1 | best=G7-D-CHE-SplitConsensusMetricNoProjection [M0-identity/S0-diagonal];rows=9 |
| weak_gate_real_lite | 1 | real_lite=3/9 |
| weak_gate_source | 1 | source_mean=0.4495026138093736 |
| weak_gate_control_equivalence | 1 | control_equivalent_fraction=0.0 |
| weak_gate_bad_event | 0 | bad_event_fraction=1.0;tail_fail=1.0;linec_fail=0.6666666666666666;auc_median=0.8296299477236473 |
| weak_gate_overall | 0 | reported=0;components={'real_lite_ok': 1, 'source_ok': 1, 'control_ok': 1, 'bad_event_ok': 0} |
| r4_route_condition | 1 | route=R4-LineCOrTailDominated;source=0.4495026138093736;tail_fail=1.0;linec_fail=0.6666666666666666 |
| direction_provenance_train_stream_only | 1 | direction_rows=2709;no_audit_metric_direction=1 |
| generic_controls_not_explaining_best | 1 | generic_split_consensus_explains=0;kan_specific_pass_count=6 |
| substrate_gate_still_blocked | 1 | best_family=D-FOU;best_count=0/9 |

## 8. 科学结论

```text
1. v15.04 已执行 Line R/S/G/P/M/D/C/Z，并生成 required artifacts。
2. 没有把 LineC/tail/AUC/calibration 用作 direction source。
3. 当前 route = R4-LineCOrTailDominated，promotion_allowed = 0。
4. MLP/generic controls、substrate-only rows 与 no-regression monitor 不写成 KAN-specific promotion。
5. R4 表示 source positive 但 LineC/tail failure dominates；计划禁止用这些 audit metric 反推方向，
   也禁止新增 G9/G10/action/controller/reset，因此当前 v15.04 内无合法继续分支。
6. 人工复核 insight 已写入手工分析区；runner 只自动写入 artifact 数据并保留该手工区。
```
