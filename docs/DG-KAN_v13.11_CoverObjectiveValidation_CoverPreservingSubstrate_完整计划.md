# DG-KAN v13.11：Cover Objective Validation + Cover-Preserving Substrate Architecture 完整计划

> 版本：v13.11 execution plan  
> 基于：v13.10 `CoverFormingSubstrateArchitectureReset` 真实结果复盘；`A Theory of Generalization in Deep Learning`；`Deep Manifold Part 2`  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no sampler / class weight；no dataset-name branch；no label-informed initialization；functional / optimizer direction 不使用 validation/test/future/query；LineC / CEp99 / NLL / ECE 只能作为 audit/gate，不生成方向；MLP row 只能作为 generic optimizer / cover control，不写成 KAN promotion。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标仍然是：

$$
\boxed{
\text{label-free strict FC-PureKAN substrate/base}
+
\text{loss-interface-generic functional update}
>
\text{same substrate/base + ordinary backprop / controls}
}
$$

这里的 “better” 不是单一 accuracy，而是同时要求：

```text
1. 表达力不打折；
2. forward / backward / step / memory 接近 MLP；
3. 训练轨迹健康，AUC-step / AUC-time 不输；
4. 几何健康：真实 signal 进入 signal channel，noise 不泄漏到 signal channel，真实 signal 不困在 reservoir；
5. functional update 的收益必须打过 AdamW / NoOp / Random / AdamWParallel / SNR-only controls；
6. 不靠 CE-tail / dataset-specific / validation/test/future 信息设计方向。
```

当前我们已经知道：

```text
1. B320-current / FHQ 历史工程能力强，但 label-informed init 已禁止，不能 official。
2. Rational 是当前唯一稳定的 active no-BSpline substrate family。
3. PopRisk-SNR signal 可以被计算，而且 parameter-to-group signal retention 并没有完全丢失。
4. v13.10 证明：只做 A-RCF1..A-RCF6 architecture reset、group permutation、overcomplete h40 bank、K14-K18 training，仍然没有形成 stable basis cover。
5. 因此当前 blocker 不是 signal retention，而是 cover formation。
```

## 0.2 v13.10 的关键结果

v13.10 最终 route：

```text
route = R1-NoCoverSubstrate
minimum_success = S0-ArchitectureScoutExecuted
promotion_allowed = 0
official_success_reached = 0
kan_real_short_run_open_allowed = 0
```

关键事实：

```text
A-RCF1..A-RCF6 均执行；
cover_substrate_pass_count = 0；
is_new_substrate_architecture = 1；
is_k_token_only_extension = 0；
official cover_purity_median = 0.0330；
official K14-K18 summary rows = 504；
S3 family pass = 2/7；
S4 rows = 0；
top-2 A-RCF hardening S3 family pass = 1/7；
A-RCF2/A-RCF3/A-RCF6 fallback S3/S4 = 0；
Non-RAT vertical slice pass = 0；
MLP control pass = 0/3；
required artifacts missing = 0；
forbidden information violation = 0。
```

这说明 v13.10 是有效负结果：它不是 fail-fast，也不是 token 小修伪装成 architecture reset。它真正测试了一个 architecture-reset 层级，但没有打开 cover substrate。

---

# 1. 各线当前进度百分比：v13.09 后估计 vs v13.10 后估计

这些百分比不是 artifact 官方字段，而是研究完成度估计，综合 gate 通过、机制清晰度、代码可信度和距离 official success 的距离。

| 线 | v13.09 后估计 | v13.10 后估计 | 变化 | 判断 |
|---|---:|---:|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 99% | 99% | 0 | 工程闭包强，不是当前 blocker |
| Historical FHQ / B320-current 工程能力 | 85% frozen | 85% frozen | 0 | 历史强，但 label-informed init 禁用 |
| Label-free FHQ / A-DYN monitor | 8% | 6%-8% | 0 到 -2 | 已非主线，只保留低预算 monitor |
| Line C 几何审计 | 88% | 88% | 0 | 审计仍可用，但不能作为 direction source |
| Rational S1 efficient substrate | 85% | 85% | 0 | 仍是唯一稳定 substrate family |
| Rational S1C channel controllability | 60% | 60% | 0 | 可执行 basis-channel movement，但 value 未成立 |
| PopRisk-SNR implementation | 88% | 88% | 0 | 已可执行，不是本轮 blocker |
| K0 SNR signal transfer audit | 82% | 84% | +2 | signal retention/cosine 继续可见 |
| Signal-to-cover mechanism | 12%-15% | 13%-16% | +1 | A-RCF/K14-K18 有局部 source，但不能覆盖 family |
| Cover formation / specialization | 5%-10% | 4%-8% | -1 到 -2 | architecture reset 后 purity 仍只有约 0.033，说明更难 |
| Cover-forming substrate architecture | 0% | 20% implementation / 0% pass | +20 impl | A-RCF1..6 已真实执行，但 pass=0 |
| Generic MLP optimizer claim | 15%-20% | 12%-15% | -3 | MLP monitor 0/3，不能作为突破 |
| KAN basis-cover functional official | 0%-5% | 0%-5% | 0 | S3/S4/S5 未达成 |
| Non-RAT substrate / S1C | 5%-10% | 3%-8% | -2 | vertical slice 仍 0 pass |
| Classic no-BSpline portfolio 总体 | 55%-60% | 53%-58% | -2 | Rational 稳定，其它 basis 未成 substrate |
| Cover objective validity | 新增 | 0% | 0 | 下一轮必须先测，否则继续架构搜索会盲目 |
| 整体 next-gen MLP claim | 35%-42% | 33%-40% | -2 | v13.10 证明现有 architecture reset 不够，路线需要更深重置 |

重点：v13.10 不是工程没做，而是科学目标没有推进。它把当前问题从 “maybe architecture reset 可以形成 cover” 推进到 “A-RCF1..6 这种程度的 architecture reset 仍然不能形成 cover”。

---

# 2. v13.10 独立分析：不是没跑，是假设被证伪

## 2.1 v13.10 的正面价值

v13.10 做对了三件事：

```text
1. 它没有继续 K8-K13 token 小修，而是实现 A-RCF1..A-RCF6 真实 substrate architecture reset。
2. 它把 functional direction 限制在 train-stream gradient / parameter / group / basis telemetry 上，没有用 LineC / CEp99 / NLL / ECE 生成方向。
3. 它没有把局部 source/AUC positive rows 写成 promotion。
```

这使得 v13.10 的负结果是可信的。

## 2.2 v13.10 的负面事实

v13.10 表明：

```text
1. 多 band rational cover 不够；
2. SNR-cluster warmup 不够；
3. persistent cover memory 不够；
4. signal/reservoir split groups 不够；
5. readout-basis decoupled cover 不够；
6. overcomplete sparse cover bank 不够。
```

这些机制都没有让 cover substrate gate 打开。cover purity 仍然很低，family coverage 不足，S4 没有 row，real short-run 不能打开。

## 2.3 当前不能再做什么

不能继续：

```text
1. K19/K20 token extension；
2. K14-K18 scale / tau / phase 小网格；
3. A-RCF2/A-RCF3/A-RCF6 轻微参数微调；
4. response dictionary；
5. oracle DeltaZ；
6. BN/BM parameter metric；
7. LineC / CEp99 / NLL / ECE direction；
8. Non-RAT 未过 substrate-health 就进入 functional proof。
```

这些方向已经被前几轮充分约束。继续做只是扩大无效网格。

---

# 3. 文档启发：为什么现在必须先验证 cover objective

## 3.1 `A Theory of Generalization in Deep Learning` 的启发

Generalization paper 的核心是 output-space 分解：

```text
signal channel:
  coherent population signal 可以 transfer 到 test；

reservoir:
  test-invisible，噪声在这里不伤 test；

failure modes:
  signal trapped in reservoir；
  noise enters signal channel。
```

它给出的 population-risk / SNR rule 是：

$$
\bar g_B=\frac{1}{b}\sum_i g_i,
$$

$$
\Sigma_B=\frac{1}{b}\sum_i(g_i-\bar g_B)(g_i-\bar g_B)^T,
$$

$$
A_B=\bar g_B\bar g_B^T-\frac{1}{b-1}\Sigma_B.
$$

对 diagonal rule：

$$
\mu_k^2>\frac{\sigma_k^2}{b-1}.
$$

这告诉我们：PopRisk-SNR 是 **coherent signal detector**，但不是 cover generator。它能告诉我们哪里有 population-safe signal，却不能自动让 KAN basis groups 形成 stable cover。

## 3.2 `Deep Manifold Part 2` 的启发

Deep Manifold 强调：

```text
1. 网络坐标系随训练变化；
2. node covers 是局部 piecewise-smooth manifold 单元；
3. boundary condition 是迭代方向来源；
4. fixed-point regions 是训练过程中逐步构造出来；
5. plasticity 先升后降，后期几何变硬。
```

这告诉我们：KAN basis 不是静态函数库，而是训练中的 cover / coordinate chart。一个 substrate 如果没有机制让 cover 竞争、稳定、分工，PopRisk-SNR 只能在混合坐标里产生局部 source，而不会形成可泛化 cover。

## 3.3 当前最深的疑问

v13.09/v13.10 的根本问题不是：

```text
SNR signal 是否存在？
```

而是：

```text
我们目前定义的 cover_purity / cover_formation gate 是否真的是正确目标？
```

如果 cover_purity 这个目标本身不预测 future advantage，那么继续优化 cover_purity 是错的。如果 cover_purity 是对的，但 A-RCF1..6 都无法提高它，说明需要更强的 cover-preserving architecture。

因此 v13.11 必须先做：

$$
\boxed{
\text{Cover Objective Validity Test}
}
$$

再做：

$$
\boxed{
\text{Cover-Preserving Substrate Architecture}
}
$$

---

# 4. 当前本质 blocker

当前 blocker 可以写成：

$$
\boxed{
\text{coherent training signal 已经能进入 Rational group/basis telemetry，}
\text{但现有 substrate 没有把它组织成稳定、可泛化、可训练的 cover。}
}
$$

更具体：

```text
1. Signal retention 不是主 blocker。
2. Cover purity 极低，是主 blocker之一。
3. 但 cover purity 是否是正确目标，还没有被验证。
4. Dense or semi-dense readout/mixing 可能洗掉 group identity。
5. A-RCF architecture 主要改变初始化/分组/overcomplete bank，但没有强制 cover identity 在训练中保持。
6. Non-RAT 没有可用 substrate，因此 functional 仍被 Rational-only 绑定。
```

这解释了为什么会出现局部 source/AUC positive rows，但没有 S3/S4：信号在局部方向可用，但没有稳定分配给一组长期存在的 basis cover。

---

# 5. v13.11 核心假设

## H1：当前 cover objective 可能未被验证

如果 oracle cover / diagnostic cover 也不能提升 future advantage，则当前 cover_purity 不是有价值目标。此时不能再做 cover architecture，需要重新定义 cover objective。

## H2：如果 cover objective 有效，当前 substrate 缺少 cover identity preservation

如果 oracle cover 可以带来 future advantage，但 A-RCF legal cover 不能，则问题是合法训练中缺少 cover identity preservation。需要显式 architectural bottleneck：

```text
block-sparse readout；
delayed cross-group mixing；
winner-take-some group update；
persistent group prototypes；
signal/reservoir dual-bank；
cover split/merge with capacity constraints。
```

## H3：dense readout / early full mixing 洗掉 cover

如果 readout 混合太早，group specialization 会变成不可辨认的线性组合。需要先局部化 group-to-output，然后再逐步释放 mixing。

## H4：cover formation 必须是训练过程，而不是初始化

初始化或 group reset 只能提供起点。真正的 cover formation 需要在训练中持续执行：

```text
Phase 1: identity-preserving plasticity；
Phase 2: competitive signal assignment；
Phase 3: cover consolidation；
Phase 4: controlled cross-cover mixing。
```

---

# 6. v13.11 实验总览

v13.11 分成八条线。

```text
Line R:
  Code / provenance / implementation readback。

Line V:
  Cover objective validity test。
  判断 cover_purity / cover_churn / cover_specialization 是否真的预测 future advantage。

Line A:
  Cover-preserving Rational substrate architecture。
  不再只是 group reset / overcomplete bank；加入 explicit group identity preservation。

Line K:
  Signal-to-cover training on cover-preserving substrate。

Line N:
  Non-RAT substrate vertical slices。
  只做 exact/no-materialize + task-health，不进入 functional proof。

Line G:
  MLP cover analog monitor。
  只做 control，不写成 KAN promotion。

Line C:
  Manifold-channel audit。

Line Z:
  Finalizer / route / no-go boundary。
```

预算建议：

```text
Line V cover objective validity: 20%
Line A/K Rational cover-preserving substrate: 45%
Line N Non-RAT vertical slices: 15%
Line G MLP monitor: 5%
Line C audit: 5%
Line R/Z infrastructure: 10%
```

---

# 7. Line R：代码与实现审计

## 7.1 目标

确保 v13.11 不再把 token 小修包装成 architecture reset。

## 7.2 必须输出

```text
v1311_code_review_manifest.csv
v1311_architecture_diff_manifest.csv
v1311_cover_objective_provenance.csv
v1311_forbidden_information_audit.csv
v1311_substrate_architecture_readback.md
v1311_required_manifest.csv
v1311_code_review_packet.zip
```

## 7.3 Gate

必须满足：

```text
uses_label_informed_init = 0
uses_validation_for_direction = 0
uses_test_for_direction = 0
uses_future_for_direction = 0 except oracle diagnostic rows
uses_linec_for_direction = 0
uses_cep99_nll_ece_for_direction = 0
is_k_token_only_extension = 0
readout_feature_proxy_only = 0
feature_table_proxy_only = 0
```

如果不满足，route：

```text
R0-ProvenanceOrArchitectureViolation
```

---

# 8. Line V：Cover Objective Validity Test

## 8.1 目标

先判断当前 cover objective 是否值得优化。

要回答：

```text
V-Q1:
  如果给 diagnostic oracle cover assignment，future advantage 是否提高？

V-Q2:
  cover_purity / cover_churn / cover_specialization 与 source_vs_best、AUC、LineC 是否相关？

V-Q3:
  当前 low cover_purity 是否真的解释失败，还是只是一个无关指标？
```

## 8.2 Oracle cover 只做 diagnostic

允许使用 future / label / LineC 的 oracle cover assignment，但必须标记：

```text
promotion_allowed = 0
oracle_diagnostic_only = 1
uses_future_or_label_or_linec = 1
```

这些 oracle 不能作为 functional direction，只用于 upper-bound：

```text
如果 oracle cover 都无效，说明 cover objective no-go；
如果 oracle cover 有效，说明 legal cover formation 是真正 blocker。
```

## 8.3 Oracle cover methods

```text
V1-FutureGradientClusterOracle
  用 future successful update 的 gradient clusters 作为 cover assignment。

V2-LineCSignalReservoirOracle
  用 LineC signal/reservoir audit 后验划分 cover，只做 diagnostic。

V3-TaskFamilyOracle
  用 synthetic task family identity 划分 cover，只做 diagnostic。

V4-LabelClassCentroidOracle
  用 label class centroid 作为 oracle cover，只做 diagnostic。

V5-RandomCoverControl
  随机 cover assignment control。

V6-PermutedOracleControl
  打乱 oracle assignment control。
```

## 8.4 Metrics

```text
cover_objective_method
oracle_diagnostic_only
promotion_allowed
cover_purity
cover_churn
cover_specialization_entropy
cover_load_gini
source_vs_best_control
AUC_time_ratio
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
NLL_delta
ECE_delta
task_family_pass
seed_pass_count
loss_interface_pass_count
```

## 8.5 Gate

Cover objective is valid if:

```text
V1/V2/V3 至少一种 oracle cover 达到：
  synthetic task families >= 5/7 pass；
  source_vs_best_control >= 0.005；
  LineC non-harm；
  CEp99/NLL/ECE non-harm；

且 random/permuted control 不过。
```

如果 oracle cover pass：

```text
route tag = CoverObjectiveValid_LegalFormationNeeded
```

如果 oracle cover fail：

```text
route = R2-CoverObjectiveInvalid
```

此时不继续 Line A/K 的大规模训练，只执行 no-go 和 objective reset。

---

# 9. Line A：Cover-Preserving Rational Substrate Architecture

## 9.1 目标

如果 Line V 证明 cover objective 有价值，Line A 构造真正保持 group identity 的 substrate。

不再做：

```text
只改 init；
只改 group reset；
只加 overcomplete h40；
只加 K-token。
```

而是做：

```text
限制早期 cross-group mixing；
显式 group identity；
竞争式 signal assignment；
持久 group prototype；
signal/reservoir dual-bank；
delayed global mixing。
```

## 9.2 Candidate architectures

### A-CPF1：BlockSparseReadoutRationalCover

```text
每组 basis 只连接到一个或少数 output blocks；
前期禁止 full dense readout；
后期逐步解冻 cross-block mixing。
```

新增指标：

```text
cross_group_mixing_norm
within_group_readout_norm
group_to_class_entropy
readout_block_sparsity
```

### A-CPF2：DelayedMixingRationalCover

```text
Phase 1:
  no cross-group mixing；
Phase 2:
  limited cross-group mixing；
Phase 3:
  full readout optional。
```

目标：避免 dense readout 在 cover 形成前把 group identity 混掉。

### A-CPF3：WinnerTakeSomeSNRGroupCover

```text
每个 batch 根据 group SNR 只更新 top-k groups；
其余 groups 保持 reservoir/plasticity。
```

这不是 loss modification，而是 optimizer routing。

记录：

```text
updated_group_count
winner_group_entropy
winner_group_stability
false_drop_fraction
false_keep_fraction
```

### A-CPF4：PersistentPrototypeCover

```text
每个 group 维护 train-stream gradient prototype EMA；
新 batch 按 gradient-cosine 分配 group；
prototype 只用 train-stream gradient，不用 label/LineC。
```

记录：

```text
prototype_norm
prototype_cosine_to_batch_gradient
prototype_drift
assignment_churn
```

### A-CPF5：SignalReservoirDualBankCover

```text
把 groups 分成 signal-bank 和 reservoir-bank；
high-SNR coherent updates 进入 signal-bank；
low-SNR / high-variance updates 限制在 reservoir-bank 或不更新；
允许 reservoir-bank 后期迁移。
```

这对应 Generalization paper 的 signal/reservoir 观点，但 direction 只用 train-stream SNR。

### A-CPF6：SparseOvercompleteCoverBankV2

```text
overcomplete bank 不再只增加数量；
加入 sparse top-k routing、capacity cap、delayed readout mixing。
```

## 9.3 Substrate Gate

进入 Line K 前必须满足：

$$
workspace\_raw\_ratio \le 1.10,
$$

$$
workspace\_incremental\_ratio \le 1.75,
$$

$$
step\_ratio \le 1.75,
$$

$$
mean\_delta \ge -0.05,
$$

$$
worst\_delta \ge -0.10,
$$

$$
AUCtime\_ratio \le 2.0,
$$

$$
median(signal\_retention\_group)\ge0.70,
$$

$$
median(cos\_group\_vs\_param)\ge0.60,
$$

$$
median(cover\_purity)\ge0.12,
$$

$$
mean(cover\_churn)\le0.50.
$$

v13.10 的 cover purity 约 0.033，因此 v13.11 先把 substrate gate 从 0.15 降到 0.12 作为 architecture-scout gate，但 S3/S4 gate 不降。

---

# 10. Line K：Signal-to-Cover Training on Cover-Preserving Substrate

## 10.1 目标

只在 Line A substrate gate 过后运行。

## 10.2 Methods

```text
K0 AdamW baseline
K14 PopRisk parameter-SNR
K15 group-SNR lift
K16 dynamic gradient-cluster cover
K17 prototype cover consolidation
K18 signal/reservoir bank update
K19 delayed mixing schedule
K20 winner-take-some group update
```

这里 K19/K20 允许出现，因为它们不是在旧 A-RCF substrate 上小修，而是配合 A-CPF cover-preserving architecture 使用。旧 runner 内继续 K19/K20 被禁止；新 architecture 里作为预注册机制允许。

## 10.3 Gate

S2：

```text
cover substrate + >=3/7 task family pass
```

S3：

```text
>=5/7 synthetic task family pass
source_vs_best_control >= 0.005
AUC_time_ratio <= 1.0
CEp99/NLL/ECE non-harm
LineC majority pass
```

S4：

```text
S3 + LineC multisketch stable + cover formation pass
```

S5：

```text
S4 + real 3x3 short-run + controls + tail/calibration pass
```

---

# 11. Line N：Non-RAT Substrate Vertical Slices

## 11.1 目标

Non-RAT 不能直接进入 functional proof。v13.11 只做 substrate-health vertical slice。

## 11.2 Families

```text
Fourier:
  frequency-band grouped cover + delayed mixing；
Chebyshev:
  degree-band grouped cover + high-degree quarantine；
RBF/FastKAN:
  center-occupancy grouped cover；
Wavelet:
  scale/support grouped cover。
```

## 11.3 Gate

Non-RAT substrate pass if:

```text
workspace_raw_ratio <= 1.20
workspace_incremental_ratio <= 2.00
step_ratio <= 2.00
mean_delta >= -0.08
worst_delta >= -0.15
AUCtime_ratio <= 2.50
LineC_pass_rate >= 0.30
cover_purity >= 0.10
```

如果一个 Non-RAT family 只过 workspace，不过 task-health：

```text
status = WorkspaceOnly_NotFunctionalSubstrate
```

---

# 12. Line G：MLP Cover Analog Monitor

## 12.1 目标

判断 cover-preserving 机制是否只是 generic hidden-group optimizer。

MLP analog：

```text
MLP-AdamW
MLP-PopRiskSNR
MLP-HiddenGroupTopK
MLP-HiddenPrototypeCover
MLP-DelayedMixing
```

如果 MLP analog 也过：

```text
generic cover optimizer positive；
KAN-specific claim requires extra margin。
```

如果 MLP fail，KAN pass：

```text
KAN explicit basis cover likely contributes。
```

---

# 13. Line C：Manifold-Channel Audit

Line C 只做审计，不做方向源。

Metrics：

```text
CouplingR2
CouplingCorr
NoiseSignalLeak
RealSignalReservoirRatio
KernelDrift
CEp99
NLL
ECE
Brier
margin_p10
signal_mass_topk
reservoir_fraction
train-probe displacement R2
```

---

# 14. Required artifacts

```text
v1311_route_decision.json
v1311_progress_table.csv
v1311_code_review_manifest.csv
v1311_architecture_diff_manifest.csv
v1311_cover_objective_provenance.csv
v1311_cover_objective_validity.csv
v1311_oracle_cover_results.csv
v1311_random_permuted_cover_controls.csv
v1311_cover_preserving_substrate.csv
v1311_cover_identity_telemetry.csv
v1311_signal_to_cover_training.csv
v1311_signal_to_cover_summary.csv
v1311_nonrat_vertical_slice.csv
v1311_mlp_cover_analog.csv
v1311_linec_audit.csv
v1311_failure_table.csv
v1311_no_go_boundary.md
v1311_next_hypothesis_queue.md
v1311_required_manifest.csv
v1311_code_review_packet.zip
```

---

# 15. Required visualizations

```text
fig_cover_objective_oracle_vs_random.svg
fig_cover_purity_vs_future_advantage.svg
fig_cover_churn_vs_source.svg
fig_cover_specialization_heatmap.svg
fig_group_to_task_assignment_matrix.svg
fig_cross_group_mixing_norm.svg
fig_winner_group_stability.svg
fig_signal_reservoir_bank_update_fraction.svg
fig_s3_s4_family_heatmap.svg
fig_nonrat_substrate_status_matrix.svg
fig_mlp_vs_kan_cover_analog.svg
fig_failure_taxonomy.svg
```

---

# 16. Stop / continue policy

## 16.1 If Cover Objective invalid

```text
route = R2-CoverObjectiveInvalid
stop Line A/K large-scale run
next = redefine cover objective
```

No further A-CPF architecture search is allowed until objective is redefined.

## 16.2 If Cover Objective valid but Line A fails substrate

```text
route = R3-CoverObjectiveValidButNoSubstrate
next = stronger substrate architecture or reduce scope to Rational-only architecture research
```

## 16.3 If Line A substrate passes but K fails S3

```text
route = R4-CoverSubstrateNoFunctionalBreakthrough
next = task-family failure autopsy; no K-token grid unless tied to specific failure.
```

## 16.4 If S3 passes but S4 fails

```text
route = R5-SyntheticTaskPositiveGeometryUnstable
next = LineC multisketch / tail / calibration stabilization.
```

## 16.5 If S4 passes

Open real short-run.

---

# 17. Codex failure handling

Codex must not stop after first fail. It must execute:

```text
1. Line V oracle cover validity;
2. if valid, A-CPF1..A-CPF6 scout;
3. top-2 A-CPF hardening;
4. Line K training;
5. Non-RAT at least Fourier + Chebyshev vertical slices;
6. MLP analog monitor;
7. no-go boundary;
8. next hypothesis queue.
```

If Line V invalid, Codex must stop architecture search and write objective-reset recommendations instead of running meaningless A-CPF sweeps.

---

# 18. Final expected outcomes

## Best case

```text
Cover objective valid;
A-CPF substrate passes cover gate;
K methods reach S3/S4;
real short-run opens.
```

## Useful negative 1

```text
Oracle cover invalid:
  current cover_purity/churn objective is wrong.
  Redefine cover objective before architecture search.
```

## Useful negative 2

```text
Oracle cover valid but A-CPF fails:
  objective is right, architecture cannot form cover.
  Need stronger substrate or restrict claim.
```

## Useful negative 3

```text
A-CPF substrate passes but K fails:
  cover exists but update policy cannot use it.
  Functional policy, not architecture, is blocker.
```

---

# 19. 一句话总结

v13.10 告诉我们：**只做 architecture reset seed / group permutation / overcomplete h40 bank 仍然不能形成 stable basis cover。**

v13.11 的第一步不是再造一个 architecture token，而是先问：

$$
\boxed{
\text{cover objective 本身是否有效？}
}
$$

如果 cover objective 有效，再构造真正 preserving cover identity 的 substrate；如果 cover objective 无效，就不要继续优化一个错误目标。
