# DG-KAN v12.27：Label-Free Signal Source、Functional Bridge 重建与 Classic No-BSpline 并行计划

> 版本：v12.27 execution plan  
> 基于：v12.26.1 LabelFreeOnly S4aToS5 FunctionalBridge 真实结果复盘  
> 目标读者：不了解本轮历史的人只看本文件，也应知道 DG-KAN 当前在做什么、已经完成什么、卡在哪里、下一步如何执行。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；official candidate 不使用 label-informed init；functional direction 不使用 label、CE vector、permuted-label CE、query batch、validation/test、future outcome 或 LineC hard target；CE / NLL / ECE / CEp99 只允许作为审计与坏化约束。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，而是建立一个新的 MLP 替代路线：

$$
\boxed{
\text{label-free strict FC-PureKAN base}
+
\text{loss-agnostic functional update}
>
\text{same base + ordinary backprop / AdamW controls}
}
$$

这里“更好”必须同时满足：

```text
1. 表达力不打折，不能用欠拟合换几何；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹健康，AUC-step / AUC-time 不输；
4. 几何更健康：CouplingR2、NoiseSignalLeak、RealSignalReservoirRatio、tail、calibration 不坏；
5. functional update 的收益必须击败 NoOp / RandomMatchedNorm / AdamWParallel / SNR-only / matched controls；
6. functional direction 必须 loss-agnostic，不能针对 CE 设计；
7. official base 必须 label-free，不能使用 trainprobe / y_for_stats / label-informed initialization。
```

最终我们想证明的不是：

$$
\text{某个 supervised initialization 的 KAN base 很强。}
$$

而是：

$$
\boxed{
\text{一个 label-free PureKAN base 已经足够强，}
\text{并且 functional update 在该 base 上带来独立几何收益。}
}
$$

## 0.2 当前已经完成的部分

此前 B320-current / FHQ 路线已经证明：simple hinge / quadratic / fused workspace 这类低成本 PureKAN primitive 可以进入 MLP-like efficiency envelope，并且在 label-informed trainprobe initialization 下表现很强。但代码审计确认，B320-current 使用训练标签统计构造 trainprobe signal frame，所以它不能再作为 official base 或 functional source。

v12.26.1 已经严格移除了 label-informed init：

```text
1. 不运行、不实例化 B320-current labelInit official path；
2. official / exploration candidate 不允许 trainprobe token；
3. official / exploration candidate 必须 y_for_stats=None / uses_y_for_stats=0；
4. functional direction 不允许 label、CE、query batch、validation/test、LineC hard target；
5. blocker 出现后必须执行 Depth 1-4 fallback，不能早停。
```

本轮新增了 A-LF0..A-LF7 label-free base registry，并新增 label-free-only functional runner，强制 `y_stats=None`，同时写入 `base_is_label_free`、`uses_label_for_init=0`、`uses_y_for_stats=0`、`forbidden_token_present` 等审计字段。

## 0.3 v12.26.1 的最终状态

v12.26.1 没有达成 S5，也没有打开 label-free S4a。最终合法状态是：

```text
route = R1-LabelFreeBaseMissing
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
line_a_near_anchor_pass_count = 0
line_a_official_pass_count = 0
line_f_exploration_gate_pass = 0
line_f_official_gate_pass = 0
s4a_single_positive_rows = 0
support_count = 0
```

但它不是原地失败。它完成了大量 label-free-only scout、hardening、Depth 5/6/7 fallback，且 provenance / code audit 闭合：

```text
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
forbidden_token_official_count = 0
uses_y_for_stats_official_count = 0
uses_label_for_init_count = 0
uses_ce_for_direction_count = 0
uses_query_batch_for_direction_count = 0
code_semantics_review_pass = 1
missing_core_code_refs = 0
```

因此，v12.26.1 的价值不是成功，而是把问题定位到：

$$
\boxed{
\text{现有 label-free A-LF token family 无法替代 B320-current 的 trainprobe-informed signal frame。}
}
$$

## 0.4 当前最重要的发现

v12.26.1 反复显示：

```text
1. A-LF0 / A-LF1 可以产生正 task delta，但 AUC-time 和 LineC 不闭合；
2. A-LF4 / A-LF7 可带来一点 LineC pass rate，但 task / worst / AUC 不够；
3. A-LF10 low-quad direct-read 可以恢复一部分 task mean delta，但 worst / AUC / LineC 仍不过；
4. A-LF24 / A-LF25 等新 signal-frame estimator 可以产生局部 task 或 LineC 迹象，但仍不能同位；
5. functional 复验中，task/control-positive 与 LineC-positive 仍分离，没有 support row；
6. 继续排列 PCA / cotangent / augmentation / block-local / low-quad / direct-read token 的边际价值已经很低。
```

当前真正 blocker 是：

$$
\boxed{
\text{缺少一个 label-free signal source，}
\text{能在初始化或早期训练前提供类似 trainprobe 的稳定 signal frame，}
\text{同时不使用 label / CE。}
}
$$

---

# 1. 对 v12.26.1 的独立分析

## 1.1 这次有没有进展？

有，但它是“清理 claim 与定位缺口”的进展，不是能力成功。

过去的 B320-current 给了我们一种强性能幻觉：base 很强，functional S4a 局部信号也能出现。但 label-informed init 去掉后，v12.26.1 证明：现有 label-free token family 无法保住相同的 signal geometry。这是非常关键的科学发现，因为它说明 B320-current 的强表现不是单纯由 hinge/quadratic primitive 的结构带来的，trainprobe-informed signal frame 至少承担了重要角色。

这意味着，项目路线从：

```text
在 B320-current 上推进 functional S4a -> S5
```

改为：

```text
先构建 label-free signal source / signal frame，
再把 functional S4a-to-S5 bridge 迁移到 label-free base。
```

这不是倒退，而是把 claim 做干净。

## 1.2 为什么感觉很慢？

因为我们主动移除了一个强但不干净的 shortcut。B320-current 在使用 label-informed trainprobe 时已经很强；去掉它后，任务和几何重新分裂：

```text
task-positive 行存在；
LineC-majority 行存在；
但 task/control + LineC + tail/calibration + robustness 同位行不存在。
```

这类进展不会马上提高最终指标，但它避免了错误论文结论：

```text
不能把 supervised label-informed init 的收益，写成 KAN architecture 或 functional update 的收益。
```

所以慢的原因不是实验没跑，而是现在的问题更本质：我们要找到 **label-free signal geometry**。

## 1.3 当前卡在哪里？

当前不是卡在：

```text
1. FHQ / hinge-quadratic kernel 完全慢；
2. functional runner 不会动；
3. CEp99 / NLL / ECE gate 太苛刻；
4. 某个数据集需要特殊调参。
```

当前卡在：

$$
\boxed{
\text{label-free base 的任务轨迹与 LineC geometry 没有同位。}
}
$$

更具体地说：

```text
A-LF0/A-LF1：task mean 正，但 AUC/LineC fail；
A-LF4/A-LF7：LineC 略有改善，但 task/worst/AUC fail；
A-LF10：task mean 正，但 worst/AUC/LineC fail；
A-LF24/A-LF25：functional 复验中 task/control 与 LineC 分离；
所有 A-LF depth 扩展：没有 label-free near-anchor，没有 S4a support row。
```

这个 failure mode 表明：现有 label-free frames 只是“几何特征”，不是“任务相关 signal frame”。它们能捕捉输入结构、局部稳定性或随机 cotangent 响应，但没有把真实分类相关的 signal channel 放到 B320-current 曾经拥有的坐标里。

## 1.4 是否在正确道路上？

是，但下一步不能继续现在的局部组合搜索。

正确的地方：

```text
1. label-informed init 已经从 official 路线移除；
2. provenance audit 闭合，没有 label/CE/query/forbidden-token promotion violation；
3. Depth 1-7 fallback 已执行，不是轻易早停；
4. 没有把 task-positive 或 LineC-majority diagnostic 写成 S4a/S5；
5. 没有按数据集调参。
```

需要改变的地方：

```text
1. 不再继续扩 A-LF token 网格；
2. 不再把 PCA / augmentation / cotangent frame 当成独立候选小修；
3. 不再用 B320-current 作为 functional diagnostic anchor；
4. 需要重新设计 label-free signal source，而不是继续排列 projection 初始化；
5. classic no-BSpline family 要并行推进，但预算受控，不能抢 functional / label-free signal source 主线。
```

---

# 2. 当前各线进展状态

## 2.1 Line A：Label-free FHQ / B320-like base

状态：未恢复。

完成度估计：

```text
B320-current historical label-informed anchor：85%，但已退出 official 主线。
Label-free FHQ near-anchor：35%-45%。
```

当前最强证据：A-LF0 / A-LF1 / A-LF10 能产生正 mean task delta，但 AUC-time、worst delta、LineC pass rate 不足；A-LF4 / A-LF7 / A-LF23 可提高局部 LineC，但 task 不稳。

关键 blocker：

```text
label-free signal source missing。
```

## 2.2 Line F：Functional bridge

状态：在 label-free base 上退回到 no support。

完成度估计：

```text
old B320-current S4a exploration：历史参考，不再 official。
label-free functional bridge：10%-20%。
```

v12.26.1 功能复验中，A-LF10 / A-LF21 / A-LF24 / A-LF25 上的 functional event 没有 exploration pass；task/control 与 LineC 仍分离。

关键 blocker：

```text
没有 label-free near-anchor 时，functional bridge 只能产生 shadow diagnostic；不能 promotion。
```

## 2.3 Line C：Manifold-Channel Geometry Diagnostics

状态：作为审计仍可用，但当前 label-free base 无法通过。

完成度估计：

```text
LineC audit：80%。
LineC as deployable value source：35%。
```

当前要注意：LineC 本身不是方向源。NoiseSignalLeak / RealSignalReservoirRatio 依赖 label-audited residual，只能做审计与坏化约束，不能被 functional direction 使用。

## 2.4 Line D：Classic no-BSpline basis portfolio

状态：继续保留，但不是主 blocker。

完成度估计：

```text
Rational：55%-65%，最值得继续；
Chebyshev：50%-55%，task trajectory blocker；
Wavelet：45%-55%，task/geometry blocker；
RBF/FastKAN：35%-40%，expression blocker；
Fourier：35%-40%，expression blocker；
B-spline：frozen，不再 active。
```

当前不应平均扫。Line D 的策略是：Rational 做 memory-specific repair；Chebyshev/Wavelet 做 task-geometry repair；RBF/Fourier 做 expression repair，且必须保持 L3 efficiency。

---

# 3. v12.27 总体目标

v12.27 的目标不是继续扩大 A-LF 网格，也不是重新引入 B320-current。v12.27 要回答一个更核心的问题：

$$
\boxed{
\text{是否存在一个不使用标签、不使用 CE 的 label-free signal source，}
\text{可以替代 trainprobe-informed signal frame，}
\text{让 FHQ base 同时恢复 task trajectory 与 LineC geometry？}
}
$$

只有当 label-free base 达到 near-anchor 后，functional bridge 才能重新进入 S4a-to-S5。

v12.27 的四个并行目标：

```text
Line R：代码与 provenance 审计，确保 label-free-only 不被破坏。
Line A：label-free signal source 重建，不再做局部 token grid。
Line C/T：loss-agnostic geometry/value source 校准，不把审计指标当方向源。
Line F：functional bridge 只在 label-free near-anchor 上复验；否则只 shadow diagnostic。
Line D：classic no-BSpline portfolio 继续，但预算受控。
```

---

# 4. 实验设计：Line R 代码 / provenance / claim 审计

## 4.1 目标

确保 v12.27 不再意外使用 B320-current / trainprobe / label-informed path。Codex 必须写清楚核心代码在哪些文件、哪些 symbol、哪些 line range，不能只给 CSV 结果。

## 4.2 必审查文件

```text
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1226_label_free_only_bridge.py
experiments/run_v1226_finalize_label_free_only.py
experiments/run_v1227_label_free_signal_source.py, new
dgkan/models/fc_purekan_primitives.py
dgkan/kernels/fused_hinge_quadratic.py
```

## 4.3 必须记录的 artifact

```text
v1227_code_review_manifest.csv
v1227_core_symbol_map.json
v1227_label_free_contract_audit.csv
v1227_forbidden_token_audit.csv
v1227_code_review_packet.zip
```

`v1227_label_free_contract_audit.csv` 字段：

```text
candidate_id
source_file
source_symbol
line_start
line_end
uses_y_for_stats
uses_trainprobe_token
uses_signalBroad_token
uses_signalBlock_token
uses_trainprobeDirect_token
uses_label_for_init
uses_label_for_direction
uses_ce_for_direction
uses_query_batch
uses_validation_or_test
uses_dataset_name_branch
functional_direction_loss_agnostic
promotion_allowed
manual_review_required
```

## 4.4 Gate

硬门：

```text
uses_y_for_stats = 0
uses_trainprobe_token = 0
uses_signalBroad_token = 0
uses_signalBlock_token = 0
uses_trainprobeDirect_token = 0
uses_label_for_init = 0
uses_label_for_direction = 0
uses_ce_for_direction = 0
uses_query_batch = 0
uses_validation_or_test = 0
uses_dataset_name_branch = 0
```

若任意 official / exploration candidate 违反：

```text
route = R0-LabelFreeContractViolation
promotion_allowed = 0
final_stop_allowed = 0
```

---

# 5. 实验设计：Line A label-free signal source 重建

## 5.1 核心假设

现有 A-LF 系列失败，是因为它们只构造了输入几何 frame，没有构造“任务相关但无标签”的 signal source。

新的假设是：

$$
\boxed{
\text{真实任务 signal 可由无标签数据流中的稳定预测结构近似，}
\text{不一定需要 label-informed trainprobe。}
}
$$

这里的“无标签 signal source”不能使用标签或 CE，但可以使用：

```text
1. 输入分布的多尺度结构；
2. augment consistency；
3. train-stream temporal stability；
4. random cotangent Jacobian response；
5. unlabeled covariance / local patch covariance；
6. pseudo-partition 但不能用 true label 或 validation feedback；
7. optimizer-state-free geometry，不读 AdamW CE update。
```

## 5.2 新 candidate family

v12.27 不再继续 A-LF23..A-LF30 token 排列，而新增四类 signal source。

### Family S1：Unlabeled Multi-View Signal Frame

构造方式：对同一 train batch 生成多种 label-free view：

```text
identity view
small translation / crop view
Gaussian noise view
stroke-thinning / pixel dropout view, dataset-agnostic implementation
low-pass / high-pass view
```

只使用输入 $x$，不使用 $y$。

计算 view-invariant 与 view-sensitive 子空间：

$$
C_{stable}=\mathbb{E}_{v,v'}[\phi_v(x)\phi_{v'}(x)^T],
$$

$$
C_{unstable}=\mathbb{E}_{v}[\phi_v(x)\phi_v(x)^T]-C_{stable}.
$$

frame 取 stable top directions + bounded unstable residual。候选：

```text
A-S1a-MultiViewStableFrame
A-S1b-MultiViewStablePlusResidual
A-S1c-MultiViewBlockLocalStable
A-S1d-MultiViewLowQuadDirect
```

### Family S2：Unlabeled Temporal Consistency Frame

不使用 label，只用同一 sample 在 early training 的 hidden/logit drift consistency。做一个 short warmup，但 direction source 不能读 CE gradient；可以读 model output drift under no-label probes。

定义：

$$
D_t(x)=f_{\theta_t}(x)-f_{\theta_0}(x).
$$

用 random-cotangent probes 估计 drift covariance：

$$
C_D=\mathbb{E}[D_t(x)D_t(x)^T].
$$

候选：

```text
A-S2a-TemporalDriftStableFrame
A-S2b-TemporalDriftResidualFrame
A-S2c-TemporalDriftLowRankDirect
```

注意：如果 warmup 使用 ordinary CE optimizer 产生 $\theta_t$，该 frame 只能 diagnostic，不能 official；official 版本必须用 label-free probes 或 frozen random probes。

### Family S3：Local Patch / Stroke Geometry Frame

针对图像输入但不按数据集调参。把输入分成固定 block，构造局部 covariance / edge-energy / stroke continuity frame。

候选：

```text
A-S3a-BlockLocalCovFrame
A-S3b-BlockLocalEdgeEnergyFrame
A-S3c-BlockLocalStableAugFrame
A-S3d-BlockLocalLowQuadDirect
```

### Family S4：Self-Predictive Reconstruction-Free Frame

不训练 autoencoder，不引入 reconstruction loss，只用随机 projections 的 self-predictive covariance：

$$
z_1=R_1x,
\quad
z_2=R_2x,
$$

$$
C_{12}=\mathbb{E}[z_1z_2^T].
$$

frame 取 cross-view predictive directions。

候选：

```text
A-S4a-CrossRandomProjectionFrame
A-S4b-CrossProjectionStableResidual
A-S4c-CrossProjectionLowQuadDirect
```

## 5.3 执行策略

Codex 必须并行跑 scout + hardening，不允许 scout fail 后停。

### Scout

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val/test = 256
epochs = 3
LineC batch = 32
sketch_dim = 8
candidate_count >= 12
```

### Hardening

选择 top-4：

```text
2 个按 task mean/worst；
1 个按 LineC pass rate；
1 个按 AUC-time / calibration。
```

hardening：

```text
train_size = 1024
val/test = 512
epochs = 8
LineC batch = 64
sketch_dim = 24
seeds = 0,1,2
datasets = MNIST,Fashion-MNIST,KMNIST
```

## 5.4 必须记录指标

```text
candidate_id
family_id
signal_source_type
uses_label_for_init
uses_ce_for_init
uses_y_for_stats
train_size
val_size
epochs
dataset
seed
val_acc
test_acc
mean_delta_vs_mlp
worst_delta_vs_mlp
max_AUC_step_ratio_vs_mlp
max_AUC_time_ratio_vs_mlp
ECE_delta
NLL_delta
CEp99_delta
LineC_CouplingR2
LineC_NoiseSignalLeak
LineC_RealSignalReservoirRatio
LineC_pass
LineC_pass_rate
LineC_all_pass
feature_rank
frame_condition
frame_entropy
frame_stability_across_views
frame_drift_stability
basis_usage_entropy
logit_norm_p95
margin_p10
```

## 5.5 Label-free near-anchor gate

Exploration near-anchor：

$$
\Delta Acc_{mean}\ge -0.005,
$$

$$
\Delta Acc_{worst}\ge -0.020,
$$

$$
AUCtime_{ratio}\le 1.10,
$$

$$
LineC\_pass\_rate\ge 5/9.
$$

Official near-anchor：

$$
\Delta Acc_{mean}\ge 0,
$$

$$
\Delta Acc_{worst}\ge -0.005,
$$

$$
AUCtime_{ratio}\le 1.00,
$$

$$
LineC\_all\_pass=1.
$$

If no candidate passes exploration near-anchor after scout/hardening + one required repair depth:

```text
route = R1-LabelFreeSignalSourceMissing
functional_official_open = 0
```

但 Codex 仍必须完成 Line T / Line D / finalizer，不能提前停止。

## 5.6 失败后 Codex 必须先尝试什么

若 task mean 好但 LineC fail：

```text
1. view-stability frame mix；
2. block-local stable augmentation frame；
3. reduce direct-read dominance；
4. add low-rank LineC residual frame, but not label/CE-derived；
5. run large LineC recheck to distinguish sketch noise vs true geometry failure。
```

若 LineC 好但 task fail：

```text
1. add low-quad direct readout cap；
2. add identity-tail residual with bounded norm；
3. reduce high-frequency frame energy；
4. check CEp99 / ECE for tail explosion；
5. do not tune by dataset name。
```

若 AUC-time fail：

```text
1. inspect early loss slope by step and time；
2. compare AUC-step vs AUC-time；
3. if both fail, primitive trajectory fail；
4. if only time fails, profile kernel timing；
5. do not fix with extra epochs alone。
```

---

# 6. 实验设计：Line F functional bridge on label-free candidate

## 6.1 进入条件

Line F official 只在 label-free near-anchor 后打开。若 Line A 没有 near-anchor，Line F 只能 shadow diagnostic，并且 `promotion_allowed=0`。

进入条件：

```text
label_free_near_anchor_pass_count >= 1
base_candidate_label_free = 1
uses_y_for_stats = 0
forbidden_token_present = 0
```

## 6.2 Functional mechanism families

不继续直接复用旧 I26/I27 小修。只测试能解释 v12.26.1 blocker 的 mechanism：

### F1：Guard-then-task sequential event

先用 label-free geometry guard 做小幅 LineC stabilization，再做 task-positive direct/gain event。

```text
F27-G1-GeometryGuardThenDirect
F27-G2-GeometryGuardThenQuadDirect
F27-G3-GeometryGuardThenLowRank
```

### F2：Task event with LineC residual preservation

把 task-positive event 投影到不破坏 LineC sketch 的 null / low-damage 子空间。

```text
F27-R1-TaskEventLineCResidualized
F27-R2-ControlResidualizedLineCPreserved
F27-R3-DirectGainLineCNullProjected
```

### F3：LineC-positive event with task readout compensation

把 LineC-positive event 加上 task readout compensation，但 compensation 必须来自 train-stream label-free observable，不能用 CE / label。

```text
F27-C1-LineCEventUnlabeledReadoutComp
F27-C2-LineCEventCovTransportComp
F27-C3-LineCEventViewStableComp
```

### F4：Policy-aware event selection

不再只测 event；测 event + planned policy：

```text
policy = no_post_calibration
policy = unlabeled_logit_scale_calibration
policy = weight_anchor
policy = geometry_guard_refresh
```

## 6.3 必须记录指标

```text
base_candidate_id
functional_candidate_id
policy_id
dataset
seed
train_shuffle_seed
linec_seed_count
source_vs_noop
source_vs_best_control
source_vs_adamwparallel
source_vs_random
source_vs_snr
LineC_seed_pass_count
LineC_all_pass
CouplingR2_delta_min
NoiseSignalLeak_delta_max
RealSignalReservoirRatio_delta_max
CEp99_delta_vs_noop
NLL_delta_vs_noop
ECE_delta_vs_noop
AUC_step_delta_vs_noop
AUC_time_delta_vs_noop
logit_drift_max
amortized_overhead
train_shuffle_robust_majority
strict_majority_pass
strict_all_pass
exploration_gate_pass
promotion_allowed
fail_reason
```

## 6.4 S4a / S5 gate

S4a exploration：

```text
source_vs_noop >= +0.015625
source_vs_best_control >= +0.0078125
LineC_seed_pass_count >= 3/5
CEp99_delta_vs_noop <= +0.05
NLL_delta_vs_noop <= +0.02
ECE_delta_vs_noop <= +0.02
train_shuffle_robust_majority >= 2/3
```

S5 official：

```text
source_vs_noop >= +0.015625
source_vs_best_control >= +0.015625
LineC_all_pass = 1
CEp99_delta_vs_noop <= +0.05
NLL_delta_vs_noop <= 0
ECE_delta_vs_noop <= 0
AUC_time_delta_vs_noop <= 0
train_shuffle_robust_all_pass >= 2/3
matched_control_scope_pass = 1
amortized_overhead <= 1.05
```

## 6.5 失败后 Codex 必须尝试什么

If task/control positive but LineC fail:

```text
1. run LineC residual projection;
2. reduce direct/gain dominance;
3. add geometry guard before task event;
4. check multi-sketch variance;
5. if still fail, mark task-only event, not functional success.
```

If LineC positive but control margin fail:

```text
1. add control-residualized readout compensation;
2. test AdamWParallel-matched control;
3. test NoOp / RandomMatchedNorm matched overhead;
4. if still fail, mark geometry-only event.
```

If CEp99 / NLL / ECE fail:

```text
1. try unlabeled logit scale calibration;
2. try weight-anchor;
3. try guard-then-task order;
4. if calibration kills task margin, mark tail/task conflict.
```

---

# 7. 实验设计：Line C/T value source calibration

## 7.1 目标

Line C/T 不能再尝试直接预测 label-audited hard release。v12.27 的目标是找到一个 label-free proxy $G_{LF}$，使它至少能预测：

```text
1. task-positive 与 LineC-positive 是否可能同位；
2. event 是否有 tail-risk；
3. event 是否会被 matched controls 解释；
4. event 是否在 train-shuffle seed 上不稳定。
```

## 7.2 Candidate features

```text
unlabeled view-stability score
frame entropy / condition
random-cotangent response spread
logit covariance spectrum
train-probe drift stability
branch/direct/quad role energy transport
multi-sketch variance estimate
LineC pre-event risk score, label-free approximation
```

## 7.3 记录字段

```text
row_id
candidate_id
base_id
functional_id
policy_id
feature_family
precommit_available
uses_label
uses_ce
uses_query
uses_validation
G_LF_score
G_task_proxy
G_linec_proxy
G_tail_proxy
G_control_proxy
observed_source_vs_noop
observed_source_vs_control
observed_LineC_seed_pass_count
observed_CEp99_delta
observed_NLL_delta
observed_ECE_delta
s4a_proxy_positive
s5_proxy_positive
```

## 7.4 Gate

Precommit value-source exploration：

$$
AUC(G_{LF},S4a\_proxy)\ge 0.70,
$$

$$
Precision@K\ge 0.30,
$$

$$
Recall@K\ge 0.30.
$$

Official value source：

$$
AUC(G_{LF},S5\_proxy)\ge 0.80,
$$

$$
Precision@K\ge 0.50,
$$

$$
Recall@K\ge 0.50.
$$

If not met:

```text
G_LF remains diagnostic; no selector promotion.
```

---

# 8. 实验设计：Line D classic no-BSpline portfolio

## 8.1 目标

Line D 是 mandatory but budget-capped branch。B-spline 已 frozen；active families：

```text
Rational
Chebyshev
Wavelet
RBF / FastKAN
Fourier
```

Line D 不抢 Line A/F 主预算，但每轮必须推进至少一个明确 blocker。

## 8.2 Budget policy

```text
Total compute budget:
  Line A/F/C/T/R = 75%
  Line D = 25%

Within Line D:
  Rational = 50%
  Chebyshev/Wavelet = 30%
  RBF/Fourier = 20%
```

## 8.3 Rational plan

Rational 当前最接近。下一步只做 memory-specific repair，不再泛扫 accuracy。

Candidates：

```text
D34-RationalActivationCheckpointNoReadoutCache
D35-RationalGroupedWorkspaceReuse
D36-RationalReadoutGradRecompute
D37-RationalDenominatorStateFP16Audit
D38-RationalLowMemNoPairLineCRepeat
```

Gate：

```text
memory_ratio_vs_mlp <= 1.20
step_ratio_vs_mlp <= 1.25
A4 expression pass
LineC pass on at least one dataset-seed without task collapse
```

If memory still around 2.0:

```text
route = Rational_MemoryBlocked_NotTaskBlocked
```

## 8.4 Chebyshev / Wavelet plan

These are TaskBlocked. Do not optimize efficiency further unless it regresses.

Try：

```text
degree-energy damping
late-enable high-degree / scale components
role-wise energy cap
LineC non-tearing trajectory repair
local-support occupancy regularity, without adding loss
```

Gate：

```text
L3 efficiency pass remains 1
A4 pass remains 1
mean_delta_vs_mlp >= -0.02
worst_delta_vs_mlp >= -0.05
LineC not worse than MLP by > tolerance
```

## 8.5 RBF / Fourier plan

These are ExpressionBlocked. Do not go dense.

Try：

```text
compact capacity increase
small low-rank residual
basis usage entropy repair
K_active small expansion
low-frequency + identity residual for Fourier
```

Gate：

```text
L3 efficiency pass remains 1
A4 expression pass becomes 1
memory_ratio <= 1.20
no high-frequency noise leak
```

## 8.6 Line D artifacts

```text
v1227_classic_family_status.csv
v1227_rational_memory_repair.csv
v1227_cheby_wavelet_task_geometry.csv
v1227_rbf_fourier_expression_repair.csv
fig_v1227_classic_basis_status_matrix.svg
fig_v1227_rational_memory_task_linec_pareto.svg
fig_v1227_expression_vs_efficiency_classic.svg
```

---

# 9. Required artifacts

v12.27 必须生成：

```text
v1227_route_decision.json
v1227_project_status_summary.md
v1227_required_artifact_manifest.csv
v1227_fallback_execution_manifest.csv
v1227_code_review_manifest.csv
v1227_label_free_contract_audit.csv
v1227_label_free_signal_source_scout.csv
v1227_label_free_signal_source_hardening.csv
v1227_label_free_signal_source_failure_table.csv
v1227_functional_bridge_shadow_or_official.csv
v1227_functional_bridge_aggregate.csv
v1227_linec_multisketch_recheck.csv
v1227_precommit_value_source.csv
v1227_classic_family_status.csv
v1227_no_go_boundary.md
v1227_next_generation_queue.csv
v1227_code_review_packet.zip
```

Missing any required artifact:

```text
route = R0-ArtifactIncomplete
promotion_allowed = 0
final_stop_allowed = 0
```

---

# 10. 必须生成的可视化

```text
fig_v1227_project_status_dashboard.svg
fig_v1227_label_free_task_linec_pareto.svg
fig_v1227_label_free_auc_worst_linec_heatmap.svg
fig_v1227_signal_source_frame_spectrum.svg
fig_v1227_frame_stability_across_views.svg
fig_v1227_linec_pass_by_family.svg
fig_v1227_task_positive_vs_linec_positive_scatter.svg
fig_v1227_functional_source_control_linec_scatter.svg
fig_v1227_tail_nll_ece_tradeoff.svg
fig_v1227_precommit_value_source_calibration.svg
fig_v1227_classic_basis_status_matrix.svg
fig_v1227_rational_memory_task_linec_pareto.svg
```

---

# 11. Route definitions

```text
S5-OfficialFunctionalSuccess:
  label-free base + functional event passes S5 official gate.

S4a-LabelFreeFunctionalExplorationOpened:
  label-free base near-anchor exists and at least one functional event passes S4a exploration.

S1-LabelFreeNearAnchorRecovered:
  label-free base passes near-anchor gate, but functional not yet open.

R1-LabelFreeSignalSourceMissing:
  label-free base still cannot satisfy task/AUC/LineC near-anchor after required signal-source attempts.

R2-FunctionalCoLocationMissing:
  label-free near-anchor exists, but functional task/control and LineC signals do not co-locate.

R3-ClassicFamilyOnlyProgress:
  main label-free/functional fail, but Line D produces a meaningful family status improvement.

R0-LabelFreeContractViolation:
  label / CE / trainprobe / forbidden path used in official or exploration candidate.

R0-ArtifactIncomplete:
  required artifacts missing.

R0-ExploreDepthIncomplete:
  mandatory fallback levels not executed.
```

---

# 12. Codex execution rules

Codex must not stop after first gate failure. It must execute the following minimum depths:

```text
Depth 1:
  Line R audit + Line A signal-source scout + Line D scheduled repair.

Depth 2:
  Line A hardening top-4 + Line C/T value-source calibration.

Depth 3:
  If no near-anchor, run one signal-source repair depth based on failure mode.
  If near-anchor exists, run Line F functional bridge.

Depth 4:
  If Line F fails, run guard-then-task and residualized-task alternatives.
  If Line A fails, run mechanism-level no-go and next signal-source queue.

Depth 5:
  Finalizer writes route, no-go boundary, next-generation queue, and code packet.
```

Codex may stop only if:

```text
1. official success reached；
2. hard budget exhausted with all mandatory depths executed；
3. R0 violation or artifact incomplete requires user intervention；
4. user_stop_flag = 1。
```

Writing `no_go_boundary.md` alone is not sufficient to stop unless all required fallback depths are completed.

---

# 13. 判断标准汇总

## 13.1 Base label-free near-anchor

Exploration：

$$
\Delta Acc_{mean}\ge -0.005,
$$

$$
\Delta Acc_{worst}\ge -0.020,
$$

$$
AUCtime_{ratio}\le 1.10,
$$

$$
LineC\_pass\_rate\ge 5/9.
$$

Official：

$$
\Delta Acc_{mean}\ge 0,
$$

$$
\Delta Acc_{worst}\ge -0.005,
$$

$$
AUCtime_{ratio}\le 1.00,
$$

$$
LineC\_all\_pass=1.
$$

## 13.2 Functional S4a

```text
source_vs_noop >= +0.015625
source_vs_best_control >= +0.0078125
LineC_seed_pass_count >= 3/5
CEp99_delta_vs_noop <= +0.05
NLL_delta_vs_noop <= +0.02
ECE_delta_vs_noop <= +0.02
train_shuffle_robust_majority >= 2/3
```

## 13.3 Functional S5

```text
source_vs_noop >= +0.015625
source_vs_best_control >= +0.015625
LineC_all_pass = 1
CEp99_delta_vs_noop <= +0.05
NLL_delta_vs_noop <= 0
ECE_delta_vs_noop <= 0
AUC_time_delta_vs_noop <= 0
train_shuffle_robust_all_pass >= 2/3
matched_control_scope_pass = 1
amortized_overhead <= 1.05
```

---

# 14. 本轮最终预期

v12.27 的最低有价值结果不是“再多跑一堆 A-LF token”。最低有价值结果是以下之一：

```text
Minimum Success A:
  找到 label-free near-anchor，恢复 S1。

Minimum Success B:
  找到 label-free S4a functional exploration row，恢复 functional bridge。

Minimum Success C:
  明确证明当前无标签 input/view/cotangent/temporal signal-source family 都不足以替代 trainprobe，并给出下一代 base 机制方向。

Minimum Success D:
  Line D Rational memory blocker 被实质降低，classic portfolio 有独立推进。
```

如果 A/B 都失败，不能写“functional update 没希望”。只能写：

$$
\boxed{
\text{在当前 FHQ label-free signal-source family 下，}
\text{任务轨迹与 LineC geometry 仍不能同位；}
\text{需要下一代 label-free signal source 或不同 primitive family。}
}
$$

---

# 15. 最终判断

v12.26.1 的关键价值是把项目带回了干净问题：去掉 label-informed init 后，现有 label-free A-LF family 无法恢复 B320-current 的 signal frame。下一步不能继续拼接已有 projection token，而要重新设计 label-free signal source。

v12.27 因此不是小修计划，而是一次主线重置：

```text
1. 保持 label-free-only 硬约束；
2. 重建 signal source，而不是继续排列 A-LF token；
3. functional bridge 只在 label-free near-anchor 上 official；
4. Line C 继续审计，但不作为方向源；
5. Classic no-BSpline family 保持并行、预算受控推进。
```

一句话：

$$
\boxed{
\text{现在的核心不是 functional event 还差一点，}
\text{而是 label-free base 缺少可替代 trainprobe 的 signal geometry。}
}
$$

