# DG-KAN 项目终极目标与当前进展总结

> 本文总结 DG-KAN 项目到 v9.2.82 为止的目标、实验进展、主要结论与当前判断。  
> 重点不是罗列每一轮实验，而是抽取项目的第一性原理、已经被证实的事实、已经被证伪的路线，以及当前真正的主 blocker。

---

## 1. 一句话总结

DG-KAN 的终极目标不是在某个小数据集上调出一个比 MLP 高一点的模型，而是构建一个可以作为 MLP 替代甚至超越 MLP 的 **Clean FullEdge PureKAN Functional Training System**。

这个系统必须同时满足：

```text
1. 结构上是 Clean / FullEdge / PureKAN；
2. 训练上是 graph-free manual forward / backward / update；
3. task loss 仍是标准 CE，不引入 teacher / distillation / auxiliary loss；
4. functional update 是 update rule，不是 loss trick；
5. 系统成本与 MLP 可比；
6. 在 task、geometry、calibration、robustness 或 sample efficiency 上有真实优势；
7. 能在 matched controls、leave-dataset-out、leave-stratum-out、paired replay 下证明 causal advantage。
```

用公式写，终极目标是：

$$
\boxed{
\text{Next-Gen Beyond-MLP}
=
\text{Clean FullEdge PureKAN}
+
\text{Graph-Free Manual Training}
+
\text{Kernel-Native Efficiency}
+
\text{Task-Safe Functional Update}
+
\text{Outcome-Grounded Controller}
+
\text{External Fair Advantage}
}
$$

现在的项目还没有达到这个终极目标。当前最准确状态是：

```text
route = R16-StableAcceptDecisionRegionUnsafe
base_candidate = LQ-t2-h256
strict PureKAN functional success = false
full functional success = false
external ready = false
```

但项目也不是原地踏步。它已经证明了多个关键子命题，并把失败边界从“有没有 KAN 表达力 / functional 是否有信号”推进到更深层的两个问题：

```text
1. outcome-grounded decision region 是否能安全选择 functional update event；
2. full-system runtime 是否能在 measured P6 path 中保持 MLP-comparable efficiency。
```

---

## 2. 项目的第一性原理

当前项目最核心的科学问题可以压缩成一句话：

$$
\boxed{
\text{是否存在一种 PureKAN edge-function 训练系统，能以 MLP 级成本获得比 MLP 更好的泛化 / 几何 / 校准 / 稳定性？}
}
$$

为了避免“刷榜式成功”，项目把成功定义得很严格。

### 2.1 PureKAN 结构原则

官方候选必须是 edge-function system。FullEdge 层写为：

$$
y_j=\sum_i \phi_{ij}(x_i).
$$

如果使用 residual / identity basis，也必须属于 edge function 内部：

$$
\phi_{ij}(x_i)=w^{(0)}_{ij}\tilde{x}_i+\rho_{ij}(\tilde{x}_i).
$$

这意味着不能把普通 Linear-SiLU-Linear MLP 结构伪装成 KAN，也不能在官方路线中混入 non-KAN trainable parameter。

### 2.2 Graph-free manual training 原则

官方 KAN path 必须使用：

```text
manual forward
manual backward
manual update
```

基础任务更新可以是 AdamW-equivalent：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\theta_{t+1}
=
\theta_t-\eta
\left(
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
+\lambda\theta_t
\right).
$$

functional update 只能作为 update rule：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\Delta\theta_{\text{functional}}.
$$

不能把 functional update 改写成：

$$
L=CE+\lambda L_{\text{functional}}.
$$

### 2.3 系统公平原则

最低系统 envelope 是：

$$
Params_{\text{KAN}}\leq1.05Params_{\text{MLP}},
$$

$$
ForwardFLOPs_{\text{KAN}}\leq1.05ForwardFLOPs_{\text{MLP}},
$$

$$
BackwardFLOPs_{\text{KAN}}\leq1.50BackwardFLOPs_{\text{MLP}},
$$

$$
StepTime_{\text{KAN}}\leq1.50StepTime_{\text{MLP}},
$$

$$
PeakMemory_{\text{KAN}}\leq1.05PeakMemory_{\text{MLP}}.
$$

强目标是：

$$
StepTime_{\text{KAN}}\leq1.20StepTime_{\text{MLP}},
$$

$$
PeakMemory_{\text{KAN}}<PeakMemory_{\text{MLP}}.
$$

### 2.4 Outcome-grounded functional update 原则

现在已经很清楚：functional update 不能只靠一个“看起来合理”的 accept score。它必须回答四个第一性问题：

```text
1. Value：这个 update 是否有正期望收益？
2. Risk：这个 update 是否有坏事件风险？
3. Support：这个判断是否有足够统计支持？
4. Cost：这个 update 是否便宜到能进入 system envelope？
```

对应形式应更接近：

$$
Accept(e)=1
\iff
LCB(V(e))>0
\land
UCB(Bad(e))\leq\tau_b
\land
UCB(Null(e))\leq\tau_n
\land
LCB(Support(e))\geq\tau_s
\land
Cost(e)\leq C_{\max}.
$$

这也是为什么当前不应继续把 StableAccept 当作最终 controller，而应该把它降级为 feature / prefilter。

---

## 3. 已经验证的重要事实

## 3.1 KAN 表达力不是完全没戏

早期 v7.x 的重要结果是：KAN-family 确实存在超过 MLP 的表达力信号。

v7.0 中，dense manual KAN 的 `D3-dense-poly2-gate` 在 3 个 primary datasets 上打开了 basic Beyond-MLP task gate：

```text
D3 val acc mean = 0.8670
D3 test acc mean = 0.8218
D3 val gap vs MLP = +0.0206
datasets KAN >= MLP = 3/3
```

并且 D3 的 ECE / NLL 也优于 MLP。这说明：

$$
\boxed{
\text{KAN basis / poly2-gate 方向存在真实表达力信号。}
}
$$

但 dense D3 不是 kernel-native official candidate，full-step efficiency 失败，因此它只能证明“表达力存在”，不能证明“系统可用”。

## 3.2 Strict KAN / macro advantage 曾经闭合过最低标准

v7.5 中，`M12` 达成过最低成功标准：

```text
StrictPass = true
GradPass = true
MacroSignificantPass = true
FullGridS2Pass = true
```

关键数据包括：

```text
macro val gap = +0.0208
CI95 low = +0.0129
Holm p = 1.40e-03
test gap = +0.0214
memory ratio = 1.0182
step ratio = 1.3688
FullGridS2 = 9/9
```

这说明：

$$
\boxed{
\text{严格 KAN 路线曾经能同时保住表达力、梯度正确性和 S2 系统效率。}
}
$$

但这个阶段仍有 teacher / assisted / external-fair 边界，不能直接等同于当前 strict Clean FullEdge PureKAN functional success。

## 3.3 v8.x 证明 functional update 不是完全没有价值

v8.3 完成了 functional re-entry minimum success。它证明了 FT7 role-wise guarded functional update 在 system-gated 设置下有真实 task/geometry 信号，但当时仍未声明 formal strong success。

v8.7 的 formal selected route 在 FMNIST / KMNIST 上完成 external fair pass。其最终状态包括：

```text
success_v87_stability = true
success_v87_adaptive = true
success_v87_external_fair = true
success_v87_no_manual_tuning = true
success_v87_formal = true
```

关键结果包括：

```text
P7 FMNIST:
  DG functional acc = 88.279998
  KB-MLP acc = 86.869997
  delta = +1.410002

P7 KMNIST:
  DG functional acc = 85.929996
  KB-MLP acc = 83.749998
  delta = +2.179998
```

这说明：

$$
\boxed{
\text{functional update 在历史 v8 路线中确实有真实 task / geometry / external fair 信号。}
}
$$

但 v8.x 的成功是 transitional / route-level success，不是当前 v9 strict FC-PureKAN functional system success。它缺少当前要求的 clean full-edge、same-run outcome、system-legal online controller、paired replay strong controls 等完整闭环。

---

## 4. v9.2 主线进展

v9.2 的核心目标是把历史经验成功点推进到：

```text
Clean FullEdge PureKAN
+ graph-free manual training
+ kernel-native efficiency
+ strict functional update controller
+ full online outcome grounding
```

### 4.1 LQ 基函数成为当前 base anchor

v9.2.6 / v9.2.7 找到并确认了 `LinearLiftQuadraticEdgeBasis`，当前主 base 是：

```text
base_candidate = LQ-t2-h256
```

v9.2.6 中 `LQ-t2-h256` 通过 P4 system gate：

```text
forward = 1.086593
backward = 0.735494
step = 1.104636
memory = 0.969501
```

并且在三任务三 seed 上达到 `8/9` near-pass。

v9.2.7 进一步确认：

```text
strict FC-PureKAN equivalence = true
P4 = true
P5 near-pass = true
P5 full-pass = false
functional = false
```

这说明：

$$
\boxed{
\text{LQ-t2-h256 是当前最可靠的 FC-PureKAN base anchor。}
}
$$

但它仍只是 base / near-pass anchor，不是 functional success。

### 4.2 v9.2.20 证明 v8 functional core 仍值得保留

v9.2.20 重新执行 v8 50/240-step strong-control replay，结果：

```text
v8_short_run_pass = 1
v8_full_replay_pass = 1
functional_core_retained = true
```

这把 v9.2.18 / v9.2.19 的“functional 暂停状态”推进为：

```text
functional core retained
```

但 strict PureKAN functional interface 仍未实现，external-ready 不能打开。

### 4.3 v9.2.22-v9.2.23：strict interface 有 actuatability，但 causal advantage 不成立

v9.2.22 中，`N2a-TinyInit-RationalFunc-BranchRatioCap` 达成：

```text
contract = 1
P4 = 1
P5 near-pass = 1
functional actuatability pass = 1
```

但 paired replay 未打过 AdamWParallel / best LR。

v9.2.23 进一步测量 actuatability proxy 到 realized logit movement 的关系：

```text
proxy-realized correlation = 0.869758
mean actual r_z = 0.077576
mean actual r_z_tail = 0.078286
failure mode = functional_event_silent
```

结论是：

$$
\boxed{
\text{接口可以动，但真实 functional event 仍太 silent / 无法形成 causality survivor。}
}
$$

### 4.4 v9.2.30-v9.2.34：pre-event observability / legal probe / observable primitive 未闭合

v9.2.30-v9.2.31 表明 pre-event features 对 grounded value 的预测不够强。

v9.2.32 发现 source-logged CP5 / horizon consistency 有 AUC 信号，但不是 legal train-stream probe。

v9.2.34 实现 OP1-OP6 observable primitive smoke，但 best OP 仍未过 grounded value observability：

```text
best OP = OP4-ObservableOrthogonalTailChannel
corr = 0.099935
AUC = 0.415519
observability pass = 0
```

结论是：

$$
\boxed{
\text{observable primitive 不能只靠静态接口特征；需要 train-stream outcome grounding。}
}
$$

### 4.5 v9.2.40-v9.2.47：base / carrier / value score 一度推进，但 heldout / support 不稳

v9.2.40 修复 base 并实现 snapshot attach / carrier：

```text
repaired base robust pass = 1
snapshot attach pass = 1
inactive equivalence pass = 1
no-event preservation pass = 1
functional carrier pass = 1
```

但 value observability fail。

v9.2.41 找到 legal value score `S7-HybridMonotoneLegal`：

```text
AUC = 0.736068
corr = 0.343287
accepted precision = 0.777778
coverage = 0.04
bad-event = 0.0
```

但 leave-dataset-out fail，leave-stratum-out 不可评估。

v9.2.42-v9.2.44 继续尝试 fresh multi-stratum / risk-first support，证明 oracle high 但 legal score / controller 仍低。

结论是：

$$
\boxed{
\text{value / support / risk 方向有信号，但当 fresh heldout 和 support 扩大后不稳定。}
}
$$

### 4.6 v9.2.48-v9.2.72：online true-delta / payload binding / exact reference controller 逐步闭合

这一阶段是 v9.2 的重要工程推进。

v9.2.48 实现 real online train-stream microprobe，best feature `gap_probe` 有 strong safe-good AUC：

```text
safe-good AUC = 0.884406
```

但 probe overhead 太高：

```text
probe overhead q90 = 5.121495
step ratio q90 = 6.121495
```

v9.2.56-v9.2.57 实现 true branch-delta tensor interface / K7d fusion，并证明 true branch-delta signal 强：

```text
AUC = 0.888401
agreement = 1.0
```

但 step ratio 仍过高。

v9.2.67 找到 reference controller `C3-T2PlusBackfill`：

```text
precision = 0.838028
coverage = 0.031305
bad-event = 0.024648
null-rate = 0.130282
precision LCB = 0.790716
bad-event UCB = 0.049995
```

v9.2.68 进一步确认 PF5 cheap prefilter：

```text
candidate rate = 0.103051
reference accept recall = 1.0
```

v9.2.70-v9.2.72 完成 full-online row binding 和 payload binding：

```text
candidate tensor payload missing = 0
candidate branch logits missing = 0
candidate true-delta logits missing = 0
functional update payload missing = 0
accepted count = 951
```

这一阶段的重要结论是：

$$
\boxed{
\text{online reference controller / PF5 / true-delta / payload binding 这些合法性链条基本被推进到了可用边界。}
}
$$

但 system cost 仍没有过：

```text
step ratio q90 > 1.50
```

### 4.7 v9.2.73-v9.2.77：system cost 被拆到 basis_norm / quantile-tail / native bucket kernel

v9.2.75 首次闭合 kernel-internal attribution：

```text
kernel_internal_attribution_pass = 1
unknown fraction = 0.012946
dominant subcomponent = basis_norm_time_ms
basis_norm_time_ms = 5.664183 ms
```

v9.2.76 进一步定位 basis_norm 内部主成本：

```text
dominant = quantile_tail_compute_time_ms
quantile_tail_compute_time_ms = 0.371753 ms
```

v9.2.77 用 kthvalue exact-tail runtime 把 quantile-tail time 大幅降低：

```text
quantile_tail_time_before = 0.578957 ms
quantile_tail_time_after = 0.194465 ms
reduction = 0.664111
audit agreement = 1.0
```

v9.2.77 还推进到 native CUDA bucket kernel，但 accept bit 在 median 边界仍有 disagreement，因此不能 official。

结论是：

$$
\boxed{
\text{system cost 不再是黑箱，主成本与可修复组件已经被逐层定位。}
}
$$

### 4.8 v9.2.78-v9.2.81：StableAccept local closure 后暴露 accepted region unsafe

v9.2.78 解决的是 native/reference accept 的局部数值一致性：

```text
root cause = median_boundary_fragility
accept disagreement = 12 -> 0
best contract = AC2Q2-stable-quantized-1e5-event-tie
native q90 reduction = 0.444326
```

但 v9.2.78 route 明确是：

```text
R18-StableAcceptLocalClosedButSystemNotOfficial
```

v9.2.79 发现 official promotion 缺 full-row stable accept fields / outcome labels / P6 full-system native trace。

v9.2.80 重跑 full train-stream materializer：

```text
event_count = 24192
candidate_count = 2876
stable_candidate_rows = 2876
outcome_labels_present = 1
missing_primary_label_count = 0
label_join_mode = direct_same_run_event_id
native_bucket_kernel_used_in_p6 = 1
new_full_system_step_ratio_measured = 1
```

但真实 decision gate 失败：

```text
precision = 0.512821
bad-event = 0.303419
null-rate = 0.117521
```

v9.2.81 修掉 score_q / rank mismatch：

```text
score_quantized_disagreement = 18 -> 0
rank_disagreement = 2 -> 0
accept_disagreement = 0
```

但 decision repair 仍失败。best repair：

```text
DR2-BadLevelTailRiskFilter
precision = 0.630037
coverage = 0.030093
bad-event = 0.183150
null-rate = 0.194139
```

P4 bad accepted autopsy 显示：

```text
heldout accepted = 468
safe = 240
bad = 142
null = 57
dominant failure mode = FMA3-risk-ucb-underestimation
other modes = candidate-set drift, null/bad conflict, horizon-tail risk
```

因此 v9.2.81 的真实结论是：

$$
\boxed{
\text{StableAccept 的数值合同已局部闭合，但 StableAccept accepted region 不是 safe-good sufficient statistic。}
}
$$

### 4.9 v9.2.82 最新结论：risk/support repair 有信号但未过，StableAccept patching 接近耗尽

v9.2.82 继续执行 outcome-grounded risk-support rebuild。

当前 terminal route 是：

```text
route = R16-StableAcceptDecisionRegionUnsafe
base_candidate = LQ-t2-h256
success_v9282_strict_purekan_functional = False
success_v9282_full_functional = False
success_v9282_external_ready = False
```

关键结果：

```text
P1:
  scoreq/rank/accept contract 仍可由 QR5 fallback 闭合
  score quantized disagreement = 0
  rank disagreement = 0
  accept disagreement = 0
  native_scoreq_emit_bitexact = 0
  fallback_row_count = 20

P2:
  candidate lifecycle 仍未闭合
  old candidate count = 2493
  new candidate count = 2876
  Jaccard = 0.654036
  candidate_id_mismatch_count = 2120
  payload_hash_mismatch_count = 468

P3:
  primary outcome labels 可靠
  missing primary label = 0
  ambiguous label = 0
  secondary outcome delta fields 仍缺 14380

P4:
  bad accepted autopsy 过 gate
  bad accepted count = 142
  attribution fraction = 1.0
  primary submode = FMA3a-risk_mean_underestimated

P5:
  best bad-risk feature = RSF6-RiskResidualScore
  bad-event AUC = 0.750961
  best safe-good AUC = 0.693120

P6:
  dataset-agnostic decision repair 失败
  best selected candidate = DR7-risk_residual_score-thr-0.715503-n1.0-s0.05
  calibration precision = 0.763043
  calibration bad-event = 0.106522
  heldout accepted count = 0
  heldout coverage = 0.0
  decision gate pass = 0
```

这轮说明 risk/support 有一点信号，但仍不足以形成 official decision frontier。

核心判断：

$$
\boxed{
\text{StableAccept patching 已接近耗尽；现在需要 outcome-grounded accept primitive，而不是继续修 StableAccept 的小边界。}
}
$$

---

## 5. 当前主 blocker

当前不是以下问题：

```text
不是 KAN 完全没表达力；
不是 functional update 完全没有历史信号；
不是 LQ base 不可用；
不是 payload binding 缺失；
不是 accept bit 数值 disagreement；
不是 full-row materializer 缺失；
不是 primary outcome labels 缺失；
不是 native kernel 完全没有进入 P6。
```

当前真正的主 blocker 是两个。

## 5.1 Decision blocker：accepted region 不安全

StableAccept 能稳定地产生 accept bit，但它选出来的 rows 不够 safe-good。

当前最好数据仍未达到 gate：

```text
precision < 0.75
bad-event > 0.05
coverage / null-rate 难以同时满足
```

v9.2.82 中 best risk feature AUC 约 `0.750961`，刚过计划门槛，但最终 decision repair 在 heldout 上 accepted count = `0`，coverage = `0.0`，说明它没有形成可部署的 frontier。

当前应承认：

$$
\boxed{
\text{StableAccept 只能作为 feature / prefilter，不能作为 final accept rule。}
}
$$

## 5.2 Runtime blocker：full-system runtime 仍碎片化

v9.2.81 发现 runtime 主因是：

```text
zero_candidate_step_count = 6652
active_step_count = 1412
empty_step_kernel_fraction = 0.824901
dominant mode = per-step-sync-empty-step-fixed-launch
```

v9.2.82 的 runtime 主线仍没有 official system pass。系统 gate 仍需：

$$
StepRatio_{q90}\leq1.50.
$$

当前需要 empty-step-free active-step scheduling 和 measured batch-major native runtime，而不是继续只报 local kernel q90。

---

## 6. 项目目前真正拥有的资产

虽然终极目标未达成，但项目已经积累了很多可以复用的资产。

### 6.1 结构资产

```text
LQ-t2-h256 base
LinearLiftQuadraticEdgeBasis
FC-PureKAN equivalence audit
FullEdge manual forward/backward/update paths
basis / conditioning / P4/P5 gate framework
```

### 6.2 Functional / controller 资产

```text
v8 functional core replay evidence
FT7 / role-wise functional mechanism history
full-row materializer
same-run primary outcome labels
true-delta / payload binding machinery
PF5 candidate prefilter
C3-T2PlusBackfill reference frontier
StableAccept local numerical contract
risk/support feature factory
```

### 6.3 Runtime 资产

```text
kernel-internal attribution
quantile-tail exact runtime
native CUDA bucket kernel
persistent workspace experiments
empty-step fragmentation attribution
batch-major runtime design constraints
```

### 6.4 审计资产

```text
no fake / no proxy audits
source gap / formula proxy negative controls
full-online row binding
candidate payload binding
artifact hashes
route/failure taxonomy
gate-blocked downstream discipline
```

这些资产应该保留。需要放弃或降级的不是整个项目，而是“StableAccept 作为 final controller”的执念。

---

## 7. 当前不应该继续做什么

现在不应该继续：

```text
1. 只修 StableAccept 的 score_q / rank / tie policy；
2. 只看 accept_disagreement = 0；
3. 继续把 StableAccept 当作 sufficient controller；
4. 继续对 AC2Q2 加局部补丁；
5. 按 dataset 调 threshold / risk filter / kernel route；
6. 用 local native q90 替代 full-system runtime；
7. 用 old artifact join outcome labels；
8. 在 P9/P10 未过前打开 paired replay / full-run；
9. 为了保住沉没成本而继续保护一个已经不安全的 accepted region。
```

这些会把系统推向复杂、脆弱、次优的补丁堆。

---

## 8. 下一阶段建议方向

下一阶段应该从：

```text
StableAccept official promotion
```

转为：

```text
Outcome-grounded minimal functional controller
```

### 8.1 新 controller 的第一性目标

新的 controller 应直接估计：

```text
value
bad risk
null risk
support confidence
cost
```

形式：

$$
Accept(e)=1
\iff
LCB(V(e))>0
\land
UCB(Bad(e))\leq\tau_b
\land
UCB(Null(e))\leq\tau_n
\land
LCB(Support(e))\geq\tau_s
\land
Cost(e)\leq C_{\max}.
$$

StableAccept 可以保留为 feature：

```text
stable_score_q
stable_rank
score_margin
```

但不能作为 final accept rule。

### 8.2 新 runtime 的第一性目标

runtime 只先做两个原则：

```text
1. empty-step skip：
   zero-candidate step 不 launch functional kernel，不 sync；

2. active-step batch-major grouping：
   有 candidates 的 step 按 bucket/family/horizon 聚合后一次执行。
```

最低目标：

```text
empty_step_kernel_fraction <= 0.05
avg_candidates_per_kernel >= 8
step_ratio_q90 <= 1.50
```

---

## 9. 当前项目阶段判断

DG-KAN 当前处在：

$$
\boxed{
\text{从 StableAccept patching 转向 outcome-grounded primitive/controller redesign 的分水岭。}
}
$$

如果下一步 risk/support rebuild 仍无法达成：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
BadEventRate_{\text{heldout}}\leq0.05,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

那么应该正式判定：

```text
StableAccept patching exhausted.
```

然后进入：

```text
new observable risk primitive
new outcome-grounded accept score
minimal functional controller
```

如果新的 outcome-grounded controller 过 decision gate，同时 empty-step-free batch-major runtime 过 system gate，才可以重新打开：

```text
leave-dataset-out
leave-stratum-out
official paired replay
short-run
full-run
robustness / strong baseline
```

---

## 10. 最终结论

截至 v9.2.82，项目没有达成终极目标，但已经完成了三个关键阶段：

```text
1. 证明 KAN / functional 路线不是完全无信号；
2. 构建出 strict PureKAN base、full-row materializer、same-run outcome、native runtime 等关键平台资产；
3. 证明 StableAccept 作为 final accept region 不够安全，必须转向 outcome-grounded risk/support primitive。
```

因此当前最准确判断是：

$$
\boxed{
\text{项目仍值得继续，但不能继续保护 StableAccept 主线；必须回到第一性原理，重构 outcome-grounded minimal functional controller。}
}
$$

这不是放弃现有探索，而是把已有探索沉淀成平台资产，然后停止对已暴露缺陷的子设计做无限小修小补。

