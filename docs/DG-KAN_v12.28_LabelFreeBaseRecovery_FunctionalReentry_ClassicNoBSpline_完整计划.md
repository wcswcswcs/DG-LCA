# DG-KAN v12.28：Label-Free Signal Geometry Recovery、Functional Re-entry 与 Classic No-BSpline 并行计划

> 版本：v12.28 execution plan  
> 基于：v12.27 `LabelFreeSignalSource FunctionalBridge ClassicNoBSpline` 实际执行日志与结果复盘  
> 生成目的：让下一位查看项目的人只看本文件，也能理解 DG-KAN 当前在做什么、已经做到哪里、为什么慢、下一步要怎么并行推进。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；不使用 label-informed initialization；functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target；CE / NLL / ECE / CEp99 只能作为审计和坏化约束，不能作为 functional direction 的来源。

---

# 0. 项目总目标与当前进展

## 0.1 总目标

DG-KAN 的目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找到一个只在某个单项指标上好看的 KAN 变体。项目的总目标是：

$$
\boxed{
\text{构建一个 label-free strict FC-PureKAN base，}
\text{并通过 loss-agnostic functional update 得到比普通反向传播更好的训练几何与模型。}
}
$$

这里的“更好”必须同时满足：

```text
1. 表达力不打折，最好强于同规模 MLP；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹健康，AUC-step / AUC-time 不输 MLP；
4. 几何更健康：train-probe coupling、signal/reservoir/noise、tail、calibration 不坏；
5. functional update 的收益必须独立于 AdamW / NoOp / RandomMatchedNorm / SNR-only / matched controls；
6. functional update 必须 loss-agnostic，不能针对 CE / label 设计。
```

最终要证明的是：

$$
\boxed{
\text{label-free PureKAN base + loss-agnostic functional update}
>
\text{same base + ordinary backprop / AdamW controls}
}
$$

而不是只证明：

$$
\text{label-informed B320-current 很强。}
$$

也不是只证明：

$$
\text{某个 functional event 在单个 seed 上提高了 accuracy。}
$$

## 0.2 当前历史锚点

过去几轮已经证明了几件重要事实。

第一，FHQ / B320-current 工程路线很强，曾经做到非常好的 step / memory / task / AUC / LineC 指标。但代码审查确认，B320-current 依赖 label-informed trainprobe initialization。因此它只能作为历史 reference 或 diagnostic artifact，不能作为当前 official base。

第二，去掉 label-informed init 后，当前 label-free A-LF / A-S 系列还没有恢复 B320-current 的 signal geometry。v12.26.1 已经执行到 Depth 7，仍没有 label-free near-anchor，也没有 label-free S4a support row；v12.27 继续测试了 MultiView / TemporalDrift / BlockLocal / CrossProjection / PseudoPartition / AffinityAnchor 等新 signal-source family，最终仍是 `R1-LabelFreeSignalSourceMissing`。

第三，functional update 不是完全没有过局部信号。v12.25 曾经打开 S4a exploration：有 precommit bridge signal、source/control AUC、tail-safe AUC、LineC-majority AUC 等正信号。但这些信号来自旧 anchor 语境，且没有达成 S5；在 label-free base 上重新验证后，v12.26.1 / v12.27 没有恢复 S4a。

第四，classic no-BSpline basis portfolio 仍是并行支线。Rational 最值得继续，但 v12.27 的 D34-D38 memory repair 没有 exploration pass；Chebyshev / Wavelet 主要是 task/geometry blocker；RBF/FastKAN / Fourier 主要是 expression blocker。B-spline 已 frozen，不再投入 active budget。

## 0.3 v12.27 当前结论

v12.27 的最终状态是：

```text
route = R1-LabelFreeSignalSourceMissing
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
functional_candidate_rows = 108
functional_aggregate_rows = 36
precommit_value_source_rows = 108
code_review_packet_entries = 70
```

解释：这不是 Codex 第一个 gate fail 后就停。v12.27 执行了 scout、hardening、repair、shadow functional、Rational memory repair、Depth 6/7 pseudo/affinity extension、finalizer 幂等修复和 stop-contract 复核。但结果仍然没有 label-free near-anchor，也没有 functional exploration pass。

所以这轮的真实科学结论是：

$$
\boxed{
\text{现有 label-free signal-source family 不能替代 label-informed trainprobe signal frame。}
}
$$

现在主 blocker 不是 FHQ kernel efficiency，也不是单个 functional lambda，而是：

$$
\boxed{
\text{label-free base 的 task signal 与 LineC geometry 无法同位。}
}
$$

---

# 1. 各条线当前完成度评估

下面百分比不是官方 metric，而是根据当前 gate、artifact、失败模式给出的阶段完成度估计。

| 线 | 完成度 | 当前状态 | 主要 blocker |
|---|---:|---|---|
| Line R：代码 / provenance / gate 审计 | 80% | v12.27 发现并修复 finalizer 聚合 bug、zip 自包含 bug、重复行 bug；required artifacts 无缺失 | 还需要更严格区分 historical reference / official candidate；每轮必须继续审查 forbidden tokens |
| Historical B320-current / FHQ 工程能力 | 85% | 历史上强，但 label-informed init，不能 official | claim 范围受限，不能再作为 official base |
| Line A：label-free base / signal source | 35% | 已执行 A-LF / A-S 多轮，task 有局部正信号，但 near-anchor pass = 0 | label-free signal frame 缺失；task / AUC / LineC 不同位 |
| Line C：Manifold-Channel 审计 | 80% | 审计可用，能识别 LineC pass、NoiseSignalLeak、ReservoirRatio、tail risk | 作为 deployable value source 仍弱，不能直接生成 direction |
| Line T：precommit value source | 35% | 旧锚点下有 S4a signal；label-free v12.27 没恢复 | T1/T2 features 不能稳定预测 label-free hard geometry support |
| Line F：functional update / S4a-to-S5 | 15% | label-free shadow functional rows 已跑，但 exploration pass = 0 | 没有 label-free near-anchor；task/control 与 LineC 不同位 |
| Line D：Classic no-BSpline portfolio | 45% | Rational 仍最有价值；D34-D38 memory repair pass = 0；其他 family 不抢主线 | Rational memory/task/LineC 不闭合；RBF/Fourier expression 不闭合 |
| 整体 next-gen MLP claim | 50% | 系统效率和历史 base 经验强，但 label-free base + functional 缺口大 | official label-free base 和 official functional success 都未成立 |

最重要的变化是：**我们已经不能再用 B320-current 的历史强表现安慰自己。** 去掉 label-informed init 后，项目回到更干净、更难的问题：怎样构造 label-free signal geometry。

---

# 2. 对 v12.27 的独立分析

## 2.1 有进展吗？

有，但不是 performance success。v12.27 的进展是把“现有 label-free signal-source family 不够”这个结论钉得更硬。

v12.27 实际执行了：

```text
1. py_compile / registry audit；
2. Line A signal-source scout；
3. Line A hardening top-4；
4. Line A repair depth；
5. Line F shadow functional bridge；
6. Line D Rational D34-D38 memory repair；
7. finalizer bug repair；
8. Depth 6 / Depth 7 pseudo-partition / affinity-anchor extension；
9. finalizer 去重与 stop-contract 复核。
```

这不是“没跑”。但它没有产生新的能力闭环：

```text
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
p4_pass = 0
promotion_allowed = 0
```

因此 v12.27 的价值是失败定位，而不是成功。

## 2.2 为什么感觉慢？

因为我们现在不允许使用 label-informed init。这个约束是正确的，但它把原来 B320-current 的优势拿掉了。之前很多 functional S4a 近似信号建立在一个强 supervised-initialized anchor 上；现在要求 label-free base 自己产生足够的 signal geometry。

这导致每个实验都要同时满足：

```text
1. label-free initialization；
2. strict FC-PureKAN；
3. MLP-like efficiency；
4. task / worst / AUC 不差；
5. LineC geometry 不撕裂；
6. functional direction 不用 label/CE/query/validation/future；
7. controls 打过；
8. train-shuffle robust。
```

这比普通模型调参难很多。慢不是因为无效，而是因为现在的 claim 变干净了。

## 2.3 发现了什么本质问题？

v12.27 发现的不是“某个候选差一点”，而是一个结构性问题：

$$
\boxed{
\text{当前无标签几何 frame 只提供了 generic variation，}
\text{没有提供任务相关的 class-discriminative signal geometry。}
}
$$

A-S 系列尝试了不同的无标签 frame：multi-view stable、temporal drift、block-local covariance、cross random projection、pseudo partition、affinity anchor、rank consensus 等。它们有些能给一点 task 或一点 LineC，但没有哪个能同时带来：

```text
1. positive / near-zero task delta；
2. worst delta 不坏；
3. AUC-time 不坏；
4. LineC pass；
5. functional event 可承载。
```

这说明现有 label-free frame 太像“输入几何 cover”，但没有变成“训练可用 signal channel”。

用 signal-channel 语言说：

```text
label-informed trainprobe 直接把 class-separating directions 放进初始 signal channel；
label-free PCA/covariance/affinity frame 主要捕捉输入方差或局部结构；
它们不一定对应分类任务中真正可泛化的 signal channel。
```

因此，简单替换 trainprobe 为 PCA、augmentation、affinity、pseudo-partition，不会自然恢复 B320-current。

## 2.4 是否在正确道路上？

大方向正确：去掉 label-informed init 是必要的；functional 必须 loss-agnostic 也是必要的；不按 dataset 调参也是必要的。

但 v12.28 不能继续沿着同一 A-S token family 扩排列组合。v12.27 已经给出足够证据：继续 pseudo / affinity / rank / viewmix / blockguard 小组合会变成低价值网格搜索。

新的正确道路应该是：

```text
1. 不再尝试“初始化时用无标签统计猜 class directions”；
2. 改成“label-free init + supervised training 中自然学习 signal frame”；
3. 在架构/参数化层面让 signal frame 可快速学习、可保持 LineC；
4. functional update 只在 label-free near-anchor 上重启 official；
5. 没有 near-anchor 时，functional 只能 shadow diagnostic。
```

换句话说，问题不再是“用哪个无标签 frame 初始化 P”，而是：

$$
\boxed{
\text{如何让 label-free PureKAN base 在正常训练中快速形成 stable signal channel。}
}
$$

---

# 3. v12.28 的核心假设

## H-A：label-free base 缺少可学习的 signal-frame formation mechanism

现有 A-LF / A-S family 主要靠初始化 frame。它们没有一个机制保证训练早期能把 task signal 快速写入 stable signal channel。

假设：

$$
\boxed{
\text{必须把 signal frame 从“初始化猜测”改成“训练早期可学习的结构”。}
}
$$

这不是 label-informed init，因为初始不读 label；训练过程仍然是正常监督学习。关键是 architecture / parameterization 允许 frame 在 early training 中快速形成，而不是靠 label statistics 直接注入。

## H-C：LineC 失败不是纯 noise，而是 signal-frame formation 的症状

如果 label-free base 的 task mean 可以为正，但 LineC pass rate 低，说明它能学一些分类边界，但训练运动没有进入稳定可泛化的 signal channel。

假设：

$$
\boxed{
\text{LineC pass rate 是 label-free signal-frame formation 是否健康的必要审计，}
\text{不能用 task accuracy 替代。}
}
$$

## H-F：functional S4a 不能在 base 未 near-anchor 时 official

v12.27 的 Line F shadow 失败说明：没有 label-free near-anchor 时，functional bridge 没有稳定承载对象。

假设：

$$
\boxed{
\text{functional update 当前应从 official promotion 退回到 shadow mechanism probing，}
\text{直到 label-free near-anchor 出现。}
}
$$

## H-D：Classic no-BSpline 支线仍应并行，但预算上限明确

Rational D34-D38 没过，但 Rational 仍是 classic family 中最有价值的；Chebyshev/Wavelet/RBF/Fourier 也不能凭旧结果完全放弃。

假设：

$$
\boxed{
\text{Classic basis 不是主线，但必须保持低预算并行推进，}
\text{尤其 Rational 的 memory / task / LineC 三重闭合。}
}
$$

---

# 4. v12.28 总体实验结构

v12.28 分为六条线并行执行。

```text
Line R：代码与 provenance 审计
Line A：label-free signal-frame formation base
Line C：LineC geometry audit 与 failure atlas
Line F：functional shadow / re-entry gate
Line D：classic no-BSpline portfolio
Line Z：finalizer / no-go / next hypothesis queue
```

关键原则：

```text
1. Line A 是主线，目标是 label-free near-anchor。
2. Line F 不再抢跑 official；没有 label-free near-anchor时只 shadow。
3. Line D 预算上限 25%，不能拖慢 Line A/F。
4. Codex 不能只写 no-go；必须按 failure type 自动执行下一层 pre-registered repair。
5. 任何使用 y_for_stats / trainprobe / trainProbeP / label init 的 candidate 直接 R0 fail。
```

---

# 5. Line R：代码与 provenance 审计

## 5.1 目标

确保 v12.28 的所有 official / exploratory candidates 都是真正 label-free、loss-agnostic、strict FC-PureKAN。

## 5.2 必须检查的代码路径

Codex 必须在复盘里写明实际文件、class/function、line range、shape 与语义：

```text
R0: candidate registry / ablation_specs / v1228_candidates
R1: SimpleFastTaskGeometryKAN init path
R2: y_for_stats / trainprobe / trainProbeP / trainprobeDirect / signalBroad / signalBlock token audit
R3: FHQ fused forward/backward/update path
R4: LineC computation path
R5: functional shadow runner
R6: controls implementation
R7: finalizer aggregation and deduplication
R8: classic family runner path
```

## 5.3 必须落盘 artifact

```text
v1228_code_provenance_manifest.csv
v1228_forbidden_token_audit.csv
v1228_core_symbol_map.json
v1228_implementation_readback.md
v1228_finalizer_idempotence_audit.csv
```

## 5.4 Hard gate

所有 official/exploration candidate 必须满足：

```text
uses_y_for_stats = 0
forbidden_token_present = 0
uses_label_for_init = 0
uses_ce_vector_for_direction = 0
uses_query_batch = 0
uses_validation_or_test_for_commit = 0
uses_dataset_name_branch = 0
```

如果任何一个为 1：

```text
route = R0-LabelOrCELegalityViolation
promotion_allowed = 0
```

---

# 6. Line A：label-free signal-frame formation base

## 6.1 目标

找到一个 label-free FHQ / B320-like base，它不靠 label-informed init，但能达到 near-anchor：

```text
mean_delta_vs_mlp >= -0.005
worst_delta_vs_mlp >= -0.015
AUC_time_ratio_vs_mlp <= 1.05
LineC_pass_rate >= 5/9 exploratory, >= 7/9 strong
no fake/proxy/cpu
```

Official anchor 需要更强：

```text
mean_delta_vs_mlp >= 0
worst_delta_vs_mlp >= -0.005
AUC_time_ratio_vs_mlp <= 1.00
LineC_all_pass = 1
step_ratio_q90 <= 1.00
memory_ratio_q90 <= 0.30
```

## 6.2 方向改变：从 frame initialization 到 frame formation

v12.27 已经充分测试了“初始化时用无标签统计构造 frame”这一大类。v12.28 不再主投同类 token组合，而是改成训练早期 frame formation。

### Family A：Learnable signal-frame warmup

候选：

```text
A-F1a-OrthoP-LearnableFrameWarmup
A-F1b-SRHTP-LearnableFrameWarmup
A-F1c-BlockOrthoP-LearnableFrameWarmup
A-F1d-RandomLowCoherenceP-LearnableFrameWarmup
```

机制：

```text
1. 初始 P 完全 label-free：orthogonal / SRHT / block-orthogonal / low-coherence random。
2. 前 warmup_steps 只训练 frame-related P / quad readout / direct readout 的一部分。
3. 之后释放 full model。
4. 不引入新 loss，不读 label 初始化；训练本身仍用普通 task loss。
```

假设：

$$
\text{label-free P 不需要一开始知道 class directions，}
\text{但必须有足够自由度和早期训练预算快速形成 signal frame。}
$$

记录：

```text
warmup_steps
frame_param_fraction_trainable
P_update_norm
P_effective_rank
P_column_coherence
role_energy_direct
role_energy_quad
LineC_pass_rate
AUC_time_ratio
```

失败后 Codex 自动尝试：

```text
如果 task 好但 LineC 低：增加 quad role warmup，降低 direct role dominance。
如果 LineC 好但 task 差：增加 direct readout warmup，不改变 P init。
如果 AUC_time 差：缩短 warmup 或改为 cosine release。
如果 rank collapse：加入 orthogonal reparameterization / QR refresh diagnostic。
```

### Family B：Self-aligning role-balanced FHQ

候选：

```text
A-F2a-RoleBalancedFHQ-LF
A-F2b-QuadDirectBalanceRamp-LF
A-F2c-BranchGainFrozenEarly-LF
A-F2d-RoleEnergyEqualizedWarmup-LF
```

机制：

```text
1. 初始化仍 label-free。
2. 通过 role-wise parameterization 控制 direct / quad / branch 的早期能量比例。
3. 不使用 label stats；不改 loss。
```

假设：

$$
\text{label-free base 的 LineC fail 可能来自 direct role 过早支配，}
\text{使模型靠 easy task direction 提高 accuracy 但没有稳定 signal/reservoir 几何。}
$$

记录：

```text
role_energy_direct
role_energy_quad
role_energy_branch
role_energy_entropy
role_energy_shift_over_epoch
CEp99
NLL
ECE
LineC_pass_rate
```

失败后 Codex 自动尝试：

```text
如果 direct role dominates and LineC fail：降低 direct gain ramp，增加 quad residual budget。
如果 quad dominates and task fail：增加 direct warmup but keep quad readout stable。
如果 ECE/CEp99 fail：启用 loss-agnostic logit scale cap，不使用 CE direction。
```

### Family C：Frame formation from optimizer-observable but label-free state

候选：

```text
A-F3a-UpdateSpectrumFrame-LF
A-F3b-MomentumCovFrame-LF
A-F3c-GradientNormOnlyFrame-LF
A-F3d-AdamSecondMomentFrame-LF
```

注意：这些不能直接读取 CE vector 或 label residual。允许使用 normal training 过程中 optimizer state 的 unlabeled aggregate statistics，例如 update covariance、moment norm、role-wise update energy。它们是 optimizer-observable，但不作为 functional direction source。

机制：

```text
1. 初始 label-free。
2. 经过短 warmup 后，用 optimizer state aggregate 形成 frame refresh。
3. refresh 只使用 aggregate covariance / norm / spectrum，不使用 per-example labels 或 CE vector。
```

风险：如果实现读取 per-example CE / label，直接 R0 fail。

记录：

```text
update_cov_rank
moment_cov_rank
frame_refresh_step
frame_refresh_norm
frame_refresh_cos_to_prev
LineC_before_after_refresh
AUC_before_after_refresh
```

失败后 Codex 自动尝试：

```text
如果 refresh 后 task 掉：降低 refresh strength。
如果 refresh 后 LineC 不变：增加 role-conditioned update spectrum。
如果 refresh 只在单 dataset 有效：route = DatasetSpecificNoPromotion，不继续按数据集调参。
```

### Family D：Capacity-without-label overcomplete frame

候选：

```text
A-F4a-OvercompleteSRHTP-h192-LF
A-F4b-OvercompleteBlockP-h192-LF
A-F4c-OvercompleteSparseP-h192-LF
A-F4d-OvercompleteThenPruneP-LF
```

机制：

```text
1. 给 label-free base 更多初始 directions，但保持 FHQ fused efficiency。
2. 使用 sparse/structured P 保持 step/memory gate。
3. 不通过 labels 指定 directions。
```

假设：

$$
\text{label-free frame 不是方向错，而是初始 cover 太窄；}
\text{增加 structured overcomplete cover 可能让训练自己选择 signal directions。}
$$

记录：

```text
hidden_dim
P_active_columns
P_sparsity
step_ratio_q90
memory_ratio_q90
LineC_pass_rate
basis_usage_entropy
rank_trajectory
```

失败后 Codex 自动尝试：

```text
如果效率 fail：减少 active columns 或使用 update-every-k。
如果 expression/task 不变：停止 overcomplete family。
如果 LineC 提升但 task 不变：与 Family B role balance 组合一次。
```

## 6.3 Line A 执行流程

### Stage A0：Scout

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val/test = 256
epochs = 3
linec_batch = 32
linec_sketch_dim = 8
families = A-F1, A-F2, A-F3, A-F4
```

通过 scout 至少需要：

```text
mean_delta_vs_mlp >= -0.030
worst_delta_vs_mlp >= -0.080
AUC_time_ratio_vs_mlp <= 1.40
LineC_pass_rate >= 1/9
```

### Stage A1：Hardening top candidates

每个 family 至少保留一个 top candidate；总数最多 8 个。

```text
train_size = 1024
val/test = 512
epochs = 8
linec_batch = 64
linec_sketch_dim = 24
```

Near-anchor gate：

```text
mean_delta_vs_mlp >= -0.005
worst_delta_vs_mlp >= -0.015
AUC_time_ratio_vs_mlp <= 1.05
LineC_pass_rate >= 5/9
```

### Stage A2：No-go boundary

如果所有 family hardening 后仍为 0 near-anchor，Codex 必须输出：

```text
v1228_label_free_base_no_go_boundary.md
```

其中要回答：

```text
1. 是 task gap 主导，还是 LineC gap 主导？
2. 是 AUC-time 主导，还是 worst delta 主导？
3. 哪个 family 至少改善一个 blocker？
4. 是否已有证据说明 FHQ label-free 不足，需要 classic basis 接管？
```

---

# 7. Line C：geometry audit 与 failure atlas

## 7.1 目标

Line C 不生成 direction，只做审计。它要解释 label-free base 为什么失败，以及 functional event 为什么不能 promotion。

## 7.2 必须记录字段

```text
candidate_id
dataset
seed
train_seed_base
linec_seed
sketch_id
CouplingR2
CouplingCorr
NoiseSignalLeak
RealSignalReservoirRatio
KernelDrift
CEp99
NLL
ECE
margin_p10
logit_drift
role_energy_direct
role_energy_quad
role_energy_branch
P_effective_rank
P_coherence
basis_usage_entropy
linec_pass
linec_fail_reason
```

## 7.3 Failure atlas

Codex 必须生成：

```text
v1228_linec_failure_atlas.csv
v1228_linec_failure_atlas.md
```

分类：

```text
TaskPositive_LineCFail
LineCPositive_TaskFail
AUCTrajectoryFail
TailCalibrationFail
RankCollapse
DirectDominance
QuadUnderuse
NoiseLeakHigh
ReservoirHigh
```

## 7.4 可视化

```text
fig_linec_pass_rate_by_family.svg
fig_task_vs_linec_scatter.svg
fig_auc_vs_linec_scatter.svg
fig_role_energy_vs_linec.svg
fig_rank_vs_linec.svg
fig_noise_reservoir_heatmap.svg
fig_ce_tail_calibration_by_family.svg
```

---

# 8. Line F：functional shadow / re-entry gate

## 8.1 目标

在没有 label-free near-anchor 前，functional 只允许 shadow diagnostic。只有 Line A 出现 near-anchor，才能打开 S4a exploration。

## 8.2 Shadow diagnostic

对每个 near-miss base 运行：

```text
F28-S1-geometryGuardThenTask
F28-S2-taskThenGeometryGuard
F28-S3-controlResidualizedDirectQuad
F28-S4-quadReservoirReleaseShadow
F28-S5-lowRankLogitSubspaceShadow
F28-S6-unlabeledCovTransportShadow
```

这些仍然必须满足：

```text
uses_label_for_direction = 0
uses_ce_vector_for_direction = 0
uses_query_batch = 0
promotion_allowed = 0 unless base_near_anchor = 1
```

## 8.3 S4a re-entry gate

如果 Line A 有 near-anchor，则 S4a exploration 需要：

```text
source_vs_noop >= +0.0078125
source_vs_best_control >= +0.0078125
LineC_seed_pass_count >= 3/5
CEp99_delta_vs_noop <= +0.50
NLL_delta_vs_noop <= +0.05
ECE_delta_vs_noop <= +0.03
train_shuffle_majority_pass >= 2/3
```

S5 official 需要：

```text
source_vs_best_control >= +0.01171875
LineC_seed_pass_count = 5/5
train_shuffle_all_or_majority_robust = 1
CEp99_delta_vs_noop <= 0
NLL_delta_vs_noop <= 0
ECE_delta_vs_noop <= 0.02
matched controls beaten
```

## 8.4 如果 functional partial signal 出现，Codex 必须自动尝试

```text
Case F1: task/control positive but LineC fail
  -> run geometryGuardThenTask, quadReservoirReleaseShadow, LineC residual projection.

Case F2: LineC positive but task/control fail
  -> run controlResidualizedDirectQuad, direct role re-entry, lower-amplitude task event.

Case F3: tail / CEp99 fail
  -> run loss-agnostic logit-scale cap, no CE-gradient tail repair.

Case F4: train-shuffle not robust
  -> run train-shuffle failure atlas, not dataset-specific branch.

Case F5: S4a on one base only
  -> replicate on second near-anchor or route as candidate-specific diagnostic.
```

---

# 9. Line D：Classic no-BSpline portfolio

## 9.1 Budget policy

Line D 继续，但预算上限为本轮总 compute 的 20%-25%。它不能抢 Line A/F 主线。

Active family：

```text
Rational
Chebyshev
Wavelet
RBF / FastKAN
Fourier
```

Frozen：

```text
B-spline
```

## 9.2 Rational：最高优先

v12.27 D34-D38 memory repair没有 exploration pass，但 Rational 仍是 classic line 最值得保留的。

下一步只做 targeted repair，不做大网格。

候选：

```text
D39-RationalGroupWorkspaceRecomputeV2
D40-RationalReadoutGradChunked
D41-RationalHiddenResLowMemV2
D42-RationalDenStateFP16PlusCheckpoint
D43-RationalPairNormLineCNoExtraMem
```

记录：

```text
step_ratio_q90
memory_ratio_q90
A4_expression_pass
mean_delta_vs_mlp
worst_delta_vs_mlp
AUC_time_ratio
LineC_pass_rate
den_p01
den_condition
RealSignalReservoirRatio
NoiseSignalLeak
```

Exploration gate：

```text
memory_ratio_q90 <= 1.25
step_ratio_q90 <= 1.25
A4_expression_pass = 1
mean_delta_vs_mlp >= -0.03
LineC_pass_rate >= 3/9
```

如果失败：

```text
memory fail -> reduce saved state / checkpoint / chunked readout grad
A4 fail -> stop low-memory variant, do not promote
LineC fail with task near -> try LineC-no-extra-mem role balance
task fail with LineC pass -> try direct readout capacity but keep memory cap
```

## 9.3 Chebyshev / Wavelet

目标：修 task trajectory，不再修 efficiency。

候选：

```text
D-CHE1-DegreeEnergyDampedK3
D-CHE2-LateEnableK4
D-CHE3-RoleEnergyCappedCheby
D-WAV1-HatWaveletScaleDiversity
D-WAV2-TriangleWaveletLocalTail
D-WAV3-WaveletRoleBalance
```

Gate：

```text
L3 efficiency pass inherited or rechecked
A4 pass required
mean_delta_vs_mlp >= -0.03
AUC_time_ratio <= 1.20
LineC_pass_rate >= 3/9
```

## 9.4 RBF / FastKAN / Fourier

目标：修 expression，不牺牲 efficiency。

候选：

```text
D-RBF1-CompactKactive4PlusResidual
D-RBF2-TrainStreamCenterWidthStats
D-RBF3-LocalBumpHybridNoDense
D-FOU1-LowFreqPlusIdentityResidual
D-FOU2-PhaseAmplitudeSharedK4
D-FOU3-LateEnableLowHighFreq
```

Gate：

```text
L3 efficiency pass must remain
A4 expression delta >= -0.02
memory_ratio <= 1.00
LineC diagnostic recorded
```

---

# 10. 执行与 stop 规则

## 10.1 Promotion fail-closed

任何 promotion 必须通过对应 gate。不能因为局部 task good、LineC majority、single-seed pass 就写 success。

## 10.2 Exploration continue-open

Codex 不能只停在第一层 no-go。v12.28 至少要求：

```text
Depth 1: scout all new label-free frame-formation families
Depth 2: hardening top candidates per family
Depth 3: failure-type-specific repair
Depth 4: functional shadow if near-miss exists
Depth 5: classic portfolio targeted repair
Depth 6: final no-go boundary + next hypothesis queue
```

如果缺少任何一层：

```text
route = R0-ExploreDepthIncomplete
```

## 10.3 Final stop 条件

只有以下情况允许 final stop：

```text
1. official success reached；
2. hard_compute_budget_exhausted = 1 且 fallback_depth >= 6；
3. mechanism-level no-go boundary 已写清楚；
4. user_stop_flag = 1。
```

只写 `next_hypothesis_queue` 不允许停止。

---

# 11. 必须落盘的 artifacts

```text
v1228_code_provenance_manifest.csv
v1228_forbidden_token_audit.csv
v1228_label_free_frame_scout.csv
v1228_label_free_frame_scout_summary.csv
v1228_label_free_frame_hardening.csv
v1228_label_free_frame_hardening_summary.csv
v1228_linec_failure_atlas.csv
v1228_linec_failure_atlas.md
v1228_functional_shadow.csv
v1228_functional_shadow_summary.csv
v1228_precommit_value_source.csv
v1228_policy_aware_p3.csv
v1228_classic_rational_repair.csv
v1228_classic_family_status.csv
v1228_required_artifact_manifest.csv
v1228_fallback_execution_manifest.csv
v1228_route_decision.json
v1228_no_go_boundary.md
v1228_next_hypothesis_queue.md
v1228_code_review_packet.zip
```

---

# 12. 必须生成的可视化

```text
fig_v1228_progress_dashboard.svg
fig_linea_task_vs_linec.svg
fig_linea_auc_vs_linec.svg
fig_linea_role_energy_heatmap.svg
fig_linea_rank_trajectory.svg
fig_functional_task_control_vs_linec.svg
fig_functional_tail_safety.svg
fig_train_shuffle_failure_atlas.svg
fig_classic_family_status.svg
fig_rational_memory_task_linec_pareto.svg
```

---

# 13. 判断标准汇总

## 13.1 Label-free near-anchor

$$
\Delta Acc_{mean} \ge -0.005
$$

$$
\Delta Acc_{worst} \ge -0.015
$$

$$
AUCtime_{ratio} \le 1.05
$$

$$
LineC_{pass\_rate} \ge 5/9
$$

## 13.2 Label-free official anchor

$$
\Delta Acc_{mean} \ge 0
$$

$$
\Delta Acc_{worst} \ge -0.005
$$

$$
AUCtime_{ratio} \le 1.00
$$

$$
LineC_{all\_pass} = 1
$$

$$
step_{ratio,q90} \le 1.00
$$

$$
memory_{ratio,q90} \le 0.30
$$

## 13.3 Functional S4a

$$
source\_vs\_control \ge 0.0078125
$$

$$
source\_vs\_noop \ge 0.0078125
$$

$$
LineC_{seed\_pass\_count} \ge 3/5
$$

$$
CEp99_{delta} \le 0.50
$$

$$
NLL_{delta} \le 0.05
$$

$$
ECE_{delta} \le 0.03
$$

## 13.4 Functional S5

$$
source\_vs\_control \ge 0.01171875
$$

$$
LineC_{seed\_pass\_count}=5/5
$$

$$
CEp99_{delta}\le0
$$

$$
NLL_{delta}\le0
$$

$$
ECE_{delta}\le0.02
$$

$$
train\_shuffle\_robust=1
$$

---

# 14. 本轮预期路线

可能结果与对应动作：

```text
Route A: label-free near-anchor appears
  -> open functional S4a shadow-to-official re-entry.

Route B: task near but LineC fail persists
  -> focus on role balance / signal-frame formation; no functional promotion.

Route C: LineC improves but task fails
  -> add task-role direct path under label-free constraints; no dataset tuning.

Route D: all label-free frame-formation families fail
  -> declare FHQ label-free signal source no-go for this version;
     shift more budget to Rational / Chebyshev as alternate base families.

Route E: Rational memory/task/LineC near-pass appears
  -> create separate classic-family base track, still label-free and no-BSpline.

Route F: functional S4a appears on label-free near-anchor
  -> run S5 confirmation with train-shuffle robustness and matched controls.
```

---

# 15. 最终建议

v12.28 的重点不是继续“找一个更复杂的无标签初始化 token”。v12.27 已经说明这条路边际价值低。下一步要把问题从：

```text
初始化时怎样猜对 signal frame？
```

改成：

```text
label-free base 怎样在训练早期形成 stable signal frame？
```

因此 v12.28 的主线是 **label-free signal-frame formation**，functional 只在 near-anchor 出现后重新 official，classic no-BSpline 作为预算受限的并行支线继续推进。

