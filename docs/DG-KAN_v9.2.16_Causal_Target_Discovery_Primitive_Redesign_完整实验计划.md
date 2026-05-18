# DG-KAN v9.2.16 Causal Target Discovery 与 Primitive Redesign：从 Control-Dominance 到 Functional Advantage 的完整实验计划

> 本计划基于 v9.2.15 `Control-Resistant Functional Causality` 的真实 terminal route 制定。  
> v9.2.15 已经证明：**P4-qualified actuator 存在，A7c / A4-family 等 actuator 也具备强 controllability；但所有 RealFunctional 分支在聚合口径下都没能击败 matched controls，P3 control-contrastive posthoc 也没有 survivor。**  
> 因此 v9.2.16 不再继续微调 SNR gate、event threshold、step fraction，也不再继续只做 target-fit 或 actuator-fit。  
> 本轮的核心任务是更本质地回答：
>
> $$
> \boxed{
> \text{当前 functional target 是否真的有训练轨迹因果价值？如果没有，应该如何发现新的 causal target 或回到 primitive/basis 设计？}
> }
> $$
>
> 本计划继续严格遵守：no teacher、no self-teacher、no distillation、no loss modification、no label smoothing、no focal/margin/calibration loss、no sampler/class weight、no CPU offload、no fake/proxy rows、KAN path 不使用 PyTorch loss.backward graph、PureKANConv/PureKANFormer 继续 deferred。

---

## 0. 当前结果与独立判断

### 0.1 v9.2.15 的真实 terminal route

v9.2.15 最终执行到：

```text
route = R10-ReturnToTargetOrPrimitiveDesign
base_candidate = LQ-t2-h256
success_v9215_paired_replay_causality = false
success_v9215_short_run = false
success_v9215_full_functional = false
success_v9215_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9215_control_resistant_functional_causality_first_20260510T030000Z/
```

核心事实：

```text
P2 full causality matrix rows = 51840
actuators = A4b,A4d,A4e,A7c
targets = O1,O2,O3,O4,O6
solvers = SOL1,SOL2,SOL3
events = E1,E2,E6,E7
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
horizons = 1,5,20,80
branches = AdamW,RealFunctional,NoOp,Random,ShuffledTarget,AdamWParallel
p2_survivor_count = 0
p3_pass_count = 0/240
```

另一个关键细节是：

```text
row-level beat = 45/8640 RealFunctional rows
all row-level beats concentrated in A4e + KMNIST + horizon80
aggregate survivor = 0
```

所以，本轮没有功能性成功，也没有 paired replay causality。P4 short-run、P5 full re-entry、P6 robustness、P7 external-ready 全部正确保持 `not_run`。

### 0.2 是否达到目标

没有。

v9.2.15 没有证明：

```text
RealFunctional beats controls
short-run functional advantage
full 10-seed functional advantage
noise robustness advantage
external-ready
significant Beyond-MLP
```

但 v9.2.15 也不是普通失败。它排除了很多旧解释：

```text
不是没有 P4-qualified actuator；
不是 actuator 没有 controllability；
不是只看了 A7c 一个 route；
不是 event coverage 没修；
不是 solver 没尝试；
不是没有 strong controls；
不是没做 paired replay。
```

因此当前最准确的结论是：

$$
\boxed{
\text{当前 actuator/target/solver family 下，RealFunctional 没有表现出超过 matched controls 的聚合因果收益。}
}
$$

### 0.3 当前不该继续做什么

v9.2.16 不应该继续做以下事情：

```text
1. 继续调 SNR threshold；
2. 继续调 event coverage；
3. 继续调 functional step fraction；
4. 继续把 direct oracle target 当作 functional 成功；
5. 继续只最大化 target-fit R2；
6. 直接进入 full functional run；
7. 用 NoOp/Random/AdamWParallel control 失败之外的指标包装成功；
8. 提前研究 PureKANConv / PureKANFormer；
9. 改 loss、teacher、sampler、class weight 或 label smoothing。
```

现在最重要的不是再把旧方向“变安全”，也不是再把 actuator 做得更可控。v9.2.15 已经告诉我们：**可控并不等于有因果收益**。下一步必须重新定义 functional target 的来源和验证方式。

---

## 1. 当前进度在总计划中的位置

DG-KAN 总计划的终极目标是：

$$
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
\text{External Fair Advantage}
+
\text{Scalable Extension}.
$$

截至当前：

| 总计划模块 | 当前状态 |
|---|---|
| Clean FC-PureKAN | LQ-t2-h256 strict equivalence 已通过 |
| Graph-free manual training | LQ path manual forward/backward/update 成立 |
| Kernel-native efficiency | LQ base 与 A7c actuator P4 成立 |
| AdamW-only task | LQ/A7c near-pass，未 full-pass |
| Functional update | control-resistant causality 未成立 |
| External fair | 未打开 |
| PureKANConv / PureKANFormer | 继续 deferred |

所以当前已经不是 v9.2.3/v9.2.4 的系统实现阶段，也不是 v9.2.13 的 “没有 P4 actuator” 阶段。当前阶段是：

$$
\boxed{
\text{P4-qualified functional actuator 已存在，但 functional target / causal mechanism 未成立。}
}
$$

---

## 2. 当前问题的本质

### 2.1 不是系统实现问题

v9.2.14 找到的 A7c 指标是：

```text
forward q90 = 1.104638
backward q90 = 1.387102
step q90 = 1.213123
compact memory = 0.969731
target fit R2 = 1.0
rz = 1.106568
```

这意味着 P4 envelope 已经满足：

$$
T_{\text{forward}}\leq1.25T_{\text{MLP}},
$$

$$
T_{\text{backward}}\leq1.50T_{\text{MLP}},
$$

$$
T_{\text{step}}\leq1.50T_{\text{MLP}},
$$

$$
M_{\text{peak}}\leq1.05M_{\text{MLP}}.
$$

因此，当前不是 kernel-native P4 blocker。

### 2.2 不是 AdamW base 训练不了

A7c 的 AdamW-only base qualification 是：

```text
near-pass = 8/9
macro delta = -0.004167
```

这说明 base 仍然是一个可用的 near-pass FC-PureKAN learner。它不是一个为了 actuator controllability 而牺牲 trainability 的模型。

### 2.3 不是 actuator 没有可控性

A7c target-fit 是：

$$
R^2_{\text{fit}}=1.0,
$$

$$
r_z=1.106568.
$$

A4-family 也在 v9.2.14 中通过 P4 + controllability gate。v9.2.15 进一步验证了多个 P4-qualified actuators，而不是只验证 A7c。没有 survivor 的原因不是 “没能力移动 logits”，而是 “移动之后没有比 controls 更好”。

### 2.4 真正 blocker

真正 blocker 是：

$$
\boxed{
\text{functional target / solver / event 组合没有可证明的 control-resistant causal advantage。}
}
$$

更直白地说：

```text
我们现在能移动输出；
也能低成本移动；
但移动的方向没有比 NoOp / Random / ShuffledTarget / AdamWParallel 更有信息。
```

这已经进入 functional update 的核心科学问题：**什么样的函数空间移动才不是随机扰动、不是额外 AdamW step、不是局部 tail metric 偶然改善，而是有稳定因果价值的 functional update？**

---

## 3. v9.2.16 总体目标

v9.2.16 的总体目标是：

$$
\boxed{
\text{发现、验证或证伪 control-resistant causal functional target。}
}
$$

本轮不以 final accuracy 为第一目标，而是要回答五个问题。

### Q1：controls 为什么赢？

必须分清楚 best control 到底是哪一类：

```text
NoOp:
  表示 functional update 多数时候是扰动，最好不动。

RandomMatchedNorm:
  表示随机正则化/噪声注入比 functional target 更有效。

ShuffledTarget:
  表示 target 的样本对应关系不重要，当前 target 不是因果结构。

AdamWParallel:
  表示额外沿 AdamW 方向走一步更好，functional 方向没有独立价值。

InvertedEvent/Target:
  表示 event/target 的方向性甚至可能是错的。
```

只有知道 controls 为什么赢，才能判断是 target、solver、event、scale，还是 primitive 需要重设计。

### Q2：A4e + KMNIST + horizon80 的 row-level signal 是否真实？

v9.2.15 唯一 row-level beat 集中在：

```text
A4e + KMNIST + horizon80
```

但 aggregate 不成立。v9.2.16 必须对这个信号做复验。它可能是：

```text
1. 随机噪声；
2. horizon80 才显现的 delayed causal effect；
3. KMNIST hard-mode 特有信号；
4. A4 bounded-rational actuator 比 A7c 更适合 functional causality；
5. metric aggregation 抹掉了稀疏 hard-mode effect。
```

### Q3：能否从 winning controls 中发现新的 causal target？

如果 Random/AdamWParallel/NoOp 赢，不能只说 functional 失败。要分析这些 control 对 logits 的实际位移：

$$
\Delta z_{\text{control}} = z(\theta+\Delta\theta_{\text{control}})-z(\theta).
$$

如果某类 control consistently 改善 CE-tail/margin/curvature，它可能揭示了一个 target family。v9.2.16 要从 control-winning branches 中提取 prototype target，而不是继续凭直觉写 O1/O2/O3/O4/O6。

### Q4：新的 target 是否能在 paired replay 中击败 controls？

任何新 target 必须先在 paired replay 中过：

$$
RealFunctional > BestControl.
$$

不允许直接进入 short-run/full-run。

### Q5：如果 target discovery 仍失败，是否应暂停 functional update，回到 primitive / basis design？

如果所有 target discovery 都 control-equivalent，就必须承认：

```text
当前 FC-LQ/A4/A7c family 尚未提供可提取 functional advantage。
```

那时下一步不该继续调 gate，而应回到：

```text
basis factory
primitive redesign
AdamW-only full-pass
external fair base validation
```

---

## 4. v9.2.16 核心假设

### H1：v9.2.15 失败不是 actuator P4 问题，而是 target causality 问题

H1 已由 v9.2.14/v9.2.15 基本支持。P4-qualified actuators 存在，但 paired replay survivor 为 0。

H1 成立标准：

P0 复现后仍满足：

```text
P4-qualified actuator pass = 1
p2_survivor_count = 0
p3_pass_count = 0
```

### H2：A4e + KMNIST + horizon80 信号可能是唯一值得保留的 functional clue

H2 假设：虽然全局 aggregate 不成立，但 row-level beats 全部集中在一个特定 region，说明可能存在 delayed hard-mode functional effect。

H2 成立标准：

在 dedicated replication 中：

$$
BeatRate_{\text{A4e,KMNIST,h80}}
\geq0.10,
$$

且至少一个机制指标相对 best control 有稳定优势：

$$
CEp99_{\text{real}}\leq CEp99_{\text{best-control}}-0.03|CEp99_{\text{AdamW}}|,
$$

或：

$$
MarginP10_{\text{real}}\geq MarginP10_{\text{best-control}}+0.01.
$$

如果复验后 beat rate 回到接近 0，则该 signal 视为噪声，不进入 target redesign。

### H3：当前 target family 是 direct-oracle 有效但 trajectory-causal 无效

O1/O2/O3/O4/O6 在 direct logit oracle 中有 useful signal，但参数实现后的 paired replay 不赢 controls。H3 假设这些 target 是 myopic target，只优化即时 tail/logit metric，不优化未来 trajectory。

H3 成立标准：

若 horizon-aware 或 control-derived target 明显优于 O1/O2/O3/O4/O6，则 H3 成立。

### H4：winning controls 可以揭示 functional target 的缺失方向

如果某类 control consistently 改善 CEp99/margin，它的 logit displacement 可以用于发现 target prototypes。

定义 control prototype：

$$
p_c
=
\mathbb{E}_{e\in \mathcal{E}_{win}}
\left[
\frac{\Delta z_c(e)}
{\|\Delta z_c(e)\|+\epsilon}
\right].
$$

若 functional target matching $p_c$ 能击败 original targets，则 H4 成立。

### H5：如果 control-derived target 仍失败，则当前 LQ/A4/A7c family 不具备可提取 functional advantage

H5 是停止条件。  
若 P2/P3/P4 均无 survivor，则 v9.2.16 route 应写为：

```text
R8-NoExtractableFunctionalAdvantage
```

并建议回到 primitive/basis design，而不是继续 functional gate tuning。

---

## 5. Candidate 设计

### 5.1 Base candidates

```text
B0-MLP-match:
  official same-parameter MLP baseline。

QF-QuadraticFeatureMLP:
  strong diagnostic baseline。

LQ0-LQ-t2-h256:
  current FC-PureKAN base。

A7c-BasisEntropy-ValueOnly:
  v9.2.14 route best P4-qualified actuator。

A4e-BoundedRational-FusedCoeffGrad:
  v9.2.15 row-level signal region candidate。

A4b/A4d:
  secondary bounded rational P4-qualified actuators。
```

### 5.2 Existing targets

```text
O1-HardTailLogitCorrection
O2-MarginTailExpansion
O3-CalibrationTailCompression
O4-CurvatureOutputFlattening
O6-KMNISTHardModeOutputTarget
```

### 5.3 New control-derived targets

```text
CD1-RandomWinnerPrototype:
  从 RandomMatchedNorm winning branches 的 logit displacement 提取 prototype。

CD2-AdamWParallelWinnerPrototype:
  从 AdamWParallel winning branches 提取 prototype。

CD3-NoOpDeltaNullTarget:
  估计 no-op favorable events，目标是 abstain 而不是 update。

CD4-BestControlMixturePrototype:
  按事件类型混合 best-control displacement。

CD5-KMNIST-A4e-H80-Prototype:
  专门从 A4e + KMNIST + horizon80 的 row-level beats 提取 delayed hard-mode target。

CD6-ControlContrastiveTailTarget:
  直接最大化 Real vs BestControl 的 CEp99 / margin gap。

CD7-ControlContrastiveCurvatureTarget:
  直接最大化 Real vs BestControl 的 curvature/local-Lipschitz gap。
```

### 5.4 Solvers

```text
SOL1-LeastSquaresSketch:
  fit target。

SOL2-ConstrainedLeastSquares:
  fit target + holdout non-increase。

SOL3-TrustRegionQP:
  fit target + trust radius。

SOL4-HorizonAwareReplaySolver:
  optimize predicted horizon 20/80 mechanism gain。

SOL5-ControlContrastiveSolver:
  maximize predicted Real - BestControl mechanism gap。

SOL6-AbstainUnlessAdvantage:
  skip update unless predicted Real > BestControl with high confidence。

SOL7-ScaleBracketSolver:
  evaluate rho = 0.003, 0.01, 0.03, 0.10 under same target.
```

### 5.5 Event controllers

```text
E1-CalibratedCEp99Tail
E2-CalibratedMarginTail
E6-CompositeSparseEvent
E7-KMNISTHardModeEvent
E8-A4e-KMNIST-H80-DelayedEvent
E9-ControlWinnerEvent
E10-AbstentionEventController
```

Coverage target:

$$
0.03\leq Coverage\leq0.15.
$$

### 5.6 Controls

```text
C0-AdamWOnly
C1-NoOpMatchedOverhead
C2-RandomMatchedNorm
C3-ShuffledTarget
C4-ShuffledSNRMask
C5-AdamWParallelDirection
C6-InvertedTargetSign
C7-InvertedEventController
C8-ControlPrototypeShuffled
```

---

## 6. 实验阶段

## P0：v9.2.15 boundary reproduction

### 目标

复现 v9.2.15 的 terminal boundary，确认本轮不是在不稳定结果上继续。

### 必须记录

```text
source_route
p2_paired_replay_pass
p2_survivor_count
p3_pass_count
row_level_beat_count
row_level_beat_region
best_control_by_metric
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R10-ReturnToTargetOrPrimitiveDesign
p2_survivor_count = 0
p3_pass_count = 0
fake_proxy_count = 0
```

### 可视化

```text
p0_v9215_boundary_dashboard.svg
p0_real_vs_best_control_summary.svg
p0_row_level_beat_distribution.svg
```

---

## P1：Control-dominance autopsy

### 目标

拆清 controls 为什么赢，以及哪个 control 在什么 metric / dataset / horizon / actuator 上赢。

### 必须记录

```text
actuator
target
solver
event
dataset
seed
horizon
metric
best_control
real_delta
best_control_delta
real_minus_best_control
rank_of_real
logit_displacement_norm_real
logit_displacement_norm_control
cos_real_control
cos_real_adamw
cos_control_adamw
task_safety_real
task_safety_control
```

### 判断标准

P1 必须形成以下分类之一：

```text
D1-NoOpDominance:
  不动最好，functional 是扰动。

D2-RandomRegularizationDominance:
  random direction consistently better。

D3-AdamWParallelDominance:
  extra AdamW step consistently better。

D4-ShuffledTargetDominance:
  target sample alignment not causal。

D5-MixedControlDominance:
  different metrics/datasets/horizons have different best controls。

D6-A4eKMNISTDelayedSignal:
  A4e + KMNIST + horizon80 is a reproducible exception.
```

### 可视化

```text
p1_control_rank_heatmap.svg
p1_best_control_by_horizon.svg
p1_best_control_by_dataset.svg
p1_real_vs_control_logit_displacement.svg
p1_cosine_real_control_adamw.svg
```

---

## P2：A4e + KMNIST + horizon80 signal replication

### 目标

验证 v9.2.15 唯一 row-level beat region 是否真实。这个阶段是必要的，因为它决定是否继续追 A4e hard-mode delayed functional。

### 设置

```text
actuator = A4e
dataset = KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
horizons = 20,80,160
targets = O1,O2,O6,CD5
solvers = SOL2,SOL3,SOL4,SOL5
events = E7,E8,E9
branches = RealFunctional,AdamW,NoOp,Random,ShuffledTarget,AdamWParallel
```

### 必须记录

```text
seed
event_id
horizon
target
solver
branch
CEp99_delta
margin_p10_delta
curvature_delta
ECE_delta
NLL_delta
acc_delta
real_beats_best_control
control_rank
event_coverage
bad_event_rate
```

### 判断标准

Delayed hard-mode signal pass：

$$
BeatRate_{\text{real}}\geq0.10,
$$

and at least one:

$$
CEp99_{\text{real}}\leq CEp99_{\text{best-control}}-0.03|CEp99_{\text{AdamW}}|,
$$

$$
MarginP10_{\text{real}}\geq MarginP10_{\text{best-control}}+0.01.
$$

If pass, A4e delayed hard-mode target enters P4/P5.  
If fail, row-level signal is treated as noise.

### 可视化

```text
p2_a4e_kmnist_horizon_effect.svg
p2_a4e_row_level_beat_rate.svg
p2_a4e_ce_margin_by_horizon.svg
p2_a4e_controls_rank.svg
```

---

## P3：Control-derived target discovery

### 目标

从 winning controls 的 logit displacement 中发现新的 target prototypes。P3 不直接训练模型，只做 target discovery 和 paired replay qualification。

### 方法

对 P1 中 winning control branches 计算：

$$
\Delta z_c(e)=z(\theta+\Delta\theta_c)-z(\theta).
$$

归一化后按 dataset / event / horizon / metric 聚类：

$$
\hat p_{g}
=
\frac{1}{|\mathcal{E}_g|}
\sum_{e\in\mathcal{E}_g}
\frac{\Delta z_c(e)}
{\|\Delta z_c(e)\|+\epsilon}.
$$

得到 CD1-CD7 target prototypes。然后用 P4-qualified actuators 通过 solver 实现这些 prototypes，并做 paired replay。

### 必须记录

```text
prototype_id
source_control
source_metric
source_dataset
source_horizon
cluster_size
prototype_norm
intra_cluster_cos_mean
target_fit_R2
rz
paired_replay_real_delta
paired_replay_best_control_delta
real_beats_best_control
```

### 判断标准

Prototype quality：

$$
IntraClusterCosMean\geq0.30.
$$

Prototype implementability：

$$
R^2_{\text{fit}}\geq0.20,
$$

$$
r_z\geq0.05.
$$

Prototype causality：

Real beats best control in paired replay on at least one mechanism metric.

### 可视化

```text
p3_control_prototype_clusters.svg
p3_prototype_fit_vs_causality.svg
p3_prototype_source_control_matrix.svg
p3_real_vs_best_control_for_prototypes.svg
```

---

## P4：Control-contrastive solver validation

### 目标

测试 solver 是否应该从 target-fit 改成直接优化 RealFunctional vs BestControl 的机制差异。

### Control-contrastive score

定义：

$$
S_{\text{cc}}
=
w_1 G_{CEp99}
+
w_2 G_{margin}
+
w_3 G_{curvature}
+
w_4 G_{ECE}
-
w_5 BadEventPenalty
-
w_6 TaskDropPenalty.
$$

其中：

$$
G_{CEp99}
=
CEp99_{\text{best-control}}
-
CEp99_{\text{real}},
$$

$$
G_{margin}
=
MarginP10_{\text{real}}
-
MarginP10_{\text{best-control}},
$$

$$
G_{curvature}
=
Curvature_{\text{best-control}}
-
Curvature_{\text{real}}.
$$

### 必须记录

```text
solver
target
actuator
event
dataset
seed
horizon
predicted_Scc
actual_Scc
CEp99_gap_vs_control
margin_gap_vs_control
curvature_gap_vs_control
ECE_gap_vs_control
task_drop
bad_event
coverage
```

### 判断标准

Prediction pass：

$$
Corr(S_{\text{pred}},S_{\text{actual}})\geq0.30.
$$

Control-contrastive pass：

$$
S_{\text{cc}}>0,
$$

$$
BadEventRate\leq0.05,
$$

$$
Coverage\in[0.03,0.15].
$$

### 可视化

```text
p4_predicted_vs_actual_scc.svg
p4_control_contrastive_pareto.svg
p4_solver_comparison_matrix.svg
p4_abstention_precision_curve.svg
```

---

## P5：Short-run causal validation

### 目标

只有 P2/P3/P4 出现 survivor 后才打开。验证 paired replay survivor 是否能转化为连续训练收益。

### 设置

```text
steps = 50,240
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
functional_candidates = top survivors
controls = AdamW,NoOp,Random,ShuffledTarget,AdamWParallel
```

### 必须记录

```text
candidate
actuator
target
solver
event_controller
dataset
seed
steps
train_loss
holdout_loss
val_acc_proxy
CEp99
margin_p10
ECE_proxy
NLL_proxy
curvature
local_lipschitz
basis_usage_entropy
actuator_usage_entropy
event_count
event_coverage
bad_event_rate
step_ratio_q90
memory_ratio
```

### 判断标准

Task safety：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Mechanism pass：

至少一个：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

Control pass：

Functional must beat best control on at least one mechanism metric.

System pass：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p5_short_run_task_mechanism_pareto.svg
p5_short_run_controls.svg
p5_event_timeline.svg
p5_ce_margin_curvature_panel.svg
p5_step_ratio_distribution.svg
```

---

## P6：Full 10-seed functional re-entry

### 目标

只有 P5 pass 后打开。验证 functional 是否能在完整训练中带来 task-safe advantage。

### 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
base = P5 survivor
controls = AdamW,NoOp,Random,ShuffledTarget,AdamWParallel
```

### 必须记录

```text
candidate
actuator
target
solver
dataset
seed
val_acc
test_acc
delta_vs_MLP
delta_vs_AdamW_base
delta_vs_QuadraticFeatureMLP
near_pass
full_pass
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature_ratio
local_lipschitz_ratio
basis_usage_entropy
actuator_usage_entropy
event_count
event_coverage
bad_event_rate
step_ratio_q90
memory_ratio
```

### 判断标准

Task safety：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Macro improvement：

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003.
$$

KMNIST repair：

$$
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
\geq0.005.
$$

Mechanism pass：

至少一个：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
NLL_{\text{functional}}\leq NLL_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

System pass：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p6_macro_delta_vs_adamw.svg
p6_kmnist_repair_matrix.svg
p6_seedwise_win_matrix.svg
p6_task_geometry_pareto.svg
p6_ce_tail_margin_panel.svg
p6_ece_nll_panel.svg
p6_controls_comparison.svg
```

---

## P7：Noise / robustness / signal separation

### 目标

验证 functional advantage 不是 clean setting 偶然有效，而是符合 signal/noise 分离预期。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
```

### 必须记录

```text
noise_type
noise_level
dataset
seed
candidate
clean_acc
noisy_acc
acc_drop
CEp99
margin_p10
ECE
NLL
curvature
event_coverage
bad_event_rate
SNR_active_fraction
```

### 判断标准

Robustness pass：

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{AdamW}}.
$$

Noise-tail pass：

$$
CEp99_{\text{functional}}\leq CEp99_{\text{AdamW}}.
$$

Signal separation pass：

当 label noise 增大时：

$$
BadEventRate_{\text{functional}}
\leq
BadEventRate_{\text{controls}},
$$

且低 SNR / high noise events 被更多 abstain。

### 可视化

```text
p7_noise_robustness_curve.svg
p7_noise_ce_tail.svg
p7_event_coverage_vs_noise.svg
p7_signal_noise_dashboard.svg
```

---

## P8：Strong baseline / external-ready gate

### 目标

判断是否可以进入 external fair validation。

### Baselines

```text
MLP-match
same-shape MLP
hidden-bracket MLP
QuadraticFeatureMLP
LQ/A7c AdamW-only
LQ/A7c functional
```

### 必须记录

```text
baseline_id
params
flops
forward_ratio
backward_ratio
step_ratio
memory_ratio
dataset
seed
acc
ECE
NLL
CEp99
margin_p10
curvature
local_lipschitz
```

### 判断标准

External-ready if：

```text
functional task-safe = 1
functional mechanism pass = 1
functional control pass = 1
functional system pass = 1
strong baseline challenge pass = 1
```

且至少一个成立：

$$
\Delta Acc_{\text{macro,functional}}\geq0,
$$

或：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{MLP}},
$$

with no task drop.

### 可视化

```text
p8_strong_baseline_pareto.svg
p8_external_ready_scorecard.svg
p8_lq_functional_vs_quadratic_mlp.svg
```

---

## 7. Required artifacts

```text
run_manifest.json
contract_audit_v9216.csv
p0_v9215_boundary_reproduction.csv
p1_control_dominance_autopsy.csv
p2_a4e_kmnist_h80_signal_replication.csv
p3_control_derived_target_discovery.csv
p4_control_contrastive_solver_validation.csv
p5_short_run_causal_validation.csv
p6_full_functional_reentry_10seed.csv
p7_noise_robustness_signal_separation.csv
p8_strong_baseline_external_ready.csv
control_displacement_trace_v9216.csv
prototype_target_trace_v9216.csv
paired_replay_branch_trace_v9216.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9215_boundary_unstable
F3_control_dominance_unattributed
F4_a4e_kmnist_h80_signal_not_replicated
F5_control_prototype_low_coherence
F6_control_derived_target_unfit
F7_control_derived_target_control_equivalent
F8_control_contrastive_prediction_fail
F9_no_paired_replay_survivor
F10_short_run_task_drop
F11_short_run_control_equivalent
F12_full_run_no_macro_or_kmnist_gain
F13_functional_system_fail
F14_quadratic_baseline_explains_gain
F15_noise_robustness_fail
F16_fake_or_proxy_violation
F17_artifact_missing
```

---

## 8. Route decision

### Route cases

```text
R1-A4eKMNISTDelayedSignalConfirmed:
  A4e + KMNIST + horizon80 signal replicates and enters short-run.

R2-ControlDerivedTargetPass:
  target prototype from winning controls beats best matched controls.

R3-ControlContrastiveSolverPass:
  solver that optimizes Real-vs-Control score produces survivor.

R4-ShortRunFunctionalPass:
  short-run shows task-safe mechanism gain.

R5-FullFunctionalAdvantage:
  full 10-seed shows macro/KMNIST/geometry/tail gain.

R6-ControlDominanceExplainedButNoFunctionalSurvivor:
  controls explain current gains; no RealFunctional survivor.

R7-A4eRowSignalNoise:
  row-level A4e signal fails replication.

R8-NoExtractableFunctionalAdvantage:
  no control-derived or contrastive target beats controls.

R9-ReturnToPrimitiveOrBasisDesign:
  current actuator-target-solver family exhausted; return to primitive/basis design.

R10-ExternalReady:
  functional passes task/geometry/system/control/robustness/strong-baseline gates.
```

### route_decision.json 必须记录

```text
route
base_candidate
best_actuator_candidate
best_target
best_solver
best_event_controller
best_control
control_dominance_type
a4e_kmnist_h80_signal_pass
control_derived_target_pass
control_contrastive_pass
paired_replay_pass
short_run_pass
full_reentry_pass
functional_task_safe
functional_mechanism_pass
functional_control_pass
functional_system_pass
functional_kmnist_repair_pass
noise_robustness_pass
strong_baseline_pass
external_ready
primary_blocker
next_required_implementation
success_v9216_paired_replay_causality
success_v9216_short_run
success_v9216_full_functional
success_v9216_external_ready
```

---

## 9. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.15 boundary。

Step 2:
  P1 control-dominance autopsy。
  必须先知道 controls 为什么赢。

Step 3:
  P2 复验 A4e + KMNIST + horizon80 row-level signal。
  这是唯一保留的 functional clue。

Step 4:
  P3 从 winning controls 中提取 target prototypes。
  不再凭直觉扩 O1/O2/O3/O4/O6。

Step 5:
  P4 control-contrastive solver validation。
  solver 直接优化 Real-vs-BestControl，而不是只拟合 target。

Step 6:
  若 P2/P3/P4 出现 survivor，进入 P5 short-run。
  没 survivor 则停止，route 写 R8/R9。

Step 7:
  P6 full 10-seed functional re-entry。

Step 8:
  P7 noise / robustness / signal separation。

Step 9:
  P8 strong baseline / external-ready。
```

---

## 10. 停止条件

### Minimum success

```text
P0 boundary reproduced
P1 control dominance attributed
at least one paired replay survivor from P2/P3/P4
no fake/proxy/offload/loss/teacher violation
```

### Functional advantage success

```text
Minimum success
+
short-run task-safe mechanism gain
+
full 10-seed task-safe macro/KMNIST/geometry/tail gain
```

### External-ready success

```text
Functional advantage success
+
noise robustness pass
+
strong baseline challenge pass
```

### Failure stop

```text
1. v9.2.15 boundary cannot be reproduced；
2. control dominance cannot be attributed；
3. A4e + KMNIST + horizon80 signal does not replicate；
4. control-derived targets have low coherence or cannot be fit；
5. control-derived targets remain control-equivalent；
6. control-contrastive solver has no positive survivor；
7. short-run harms task；
8. full run gives no macro/KMNIST/geometry/tail gain；
9. functional breaks system gate；
10. gains are explained by QuadraticFeatureMLP；
11. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 11. 最终解释规则

### Case A：A4e delayed KMNIST signal replicates

可以声明：

```text
Functional advantage may be sparse and delayed, concentrated in KMNIST hard mode; continue with A4e-targeted short/full validation.
```

但不能声明 broad functional advantage unless P5/P6 pass.

### Case B：control-derived target passes

可以声明：

```text
Original hand-designed functional targets were weak; control-derived target discovery found a causally useful function-space movement.
```

### Case C：control-contrastive solver passes

可以声明：

```text
Target-fit was insufficient; functional update needs a solver that optimizes Real-vs-Control causal score.
```

### Case D：all target discovery remains control-equivalent

必须声明：

```text
Current FC-LQ/A4/A7c actuator-target-solver family does not provide extractable functional advantage; return to primitive/basis design.
```

### Case E：full run improves KMNIST or macro safely

可以声明：

```text
Functional update becomes a valid task-safe advantage route and can enter external fair validation.
```

---

## 12. 最终建议

v9.2.16 的一句话策略是：

$$
\boxed{
\text{不要再手写新的 functional target；先解释 controls 为什么赢，再从 winning controls 和 A4e/KMNIST 延迟信号中发现 causally useful target。}
}
$$

当前最关键的问题不是系统实现、AdamW、SNR、event coverage 或 actuator controllability，而是：

```text
1. controls 为什么赢？
2. A4e + KMNIST + horizon80 是否是真信号？
3. winning controls 的 logit displacement 是否揭示了新的 functional target？
4. solver 是否必须从 target-fit 变成 control-contrastive / horizon-aware？
5. 如果这些都失败，是否应停止 functional gate 修补，回到 primitive/basis 设计？
```
