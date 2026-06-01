# DG-KAN v12.26.1：Label-Free-Only S4a-to-S5 Functional Bridge 完整计划

> 版本：v12.26.1 execution plan  
> 基于：v12.25 S4-to-S5 precommit functional 真实结果、v12.26 原计划、代码审查中发现的 B320 label-informed trainprobe initialization 风险  
> 本版最大修正：**彻底移除 label-informed initialization。B320-current 不再作为 official base、diagnostic anchor、functional source 或 promotion reference。**  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；no label-informed init；functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target；CE / NLL / ECE / CEp99 只能作为审计与坏化约束，不能作为方向源。

---

# 0. 项目总目标、当前进展和本版重锚

## 0.1 项目总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找一个只在某个指标上好看的更新规则。项目目标是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 loss-agnostic functional update 获得比普通反向传播更好的训练几何和模型。}
}
$$

这里的“更好”必须同时满足：

```text
1. 表达力不打折，最好优于同规模 MLP；
2. forward / backward / step / memory 与 MLP 可比，不能靠慢很多换效果；
3. 收敛轨迹健康，AUC-step / AUC-time 不输；
4. 几何更好：train-probe coupling、signal/reservoir/noise、tail、calibration 不坏；
5. functional update 的收益必须独立于 AdamW / Random / SNR-only / NoOp controls；
6. functional direction 必须 loss-agnostic，不能针对 CE 设计；
7. base 初始化必须 label-free，不能依赖训练标签构造初始投影、readout 或 probe directions。
```

最终要证明的是：

$$
\boxed{
\text{Label-free PureKAN base + functional update}
>
\text{同一 Label-free PureKAN base + ordinary backprop / AdamW controls}
}
$$

不是只证明：

$$
\text{label-informed PureKAN base} > \text{MLP}.
$$

也不是只证明：

$$
\text{某个 diagnostic geometry score 上升。}
$$

## 0.2 本版为什么必须去掉 label-informed init

前几轮代码审查发现，B320-current 的 `trainProbeP / trainprobedirect / signalBroad / signalBlock` 路径在 `y_for_stats` 存在时，会使用训练标签计算 class mean directions，并把这些方向写入 projection / direct readout / signal mixing 初始化。

这不等于使用 test label，也不等于 functional update 本身用了 label；但它会改变 claim 的性质：

```text
B320-current 不是干净的 label-free architecture anchor。
它是 supervised label-informed initialization anchor。
```

本项目最终不能依赖这类初始化。否则别人会合理质疑：

```text
优势到底来自 KAN / functional update，还是来自训练标签统计初始化？
```

因此 v12.26.1 的硬规则是：

$$
\boxed{
\text{所有 official base、functional source、promotion reference、P3/P4 candidate 都必须 label-free。}
}
$$

B320-current 只允许作为历史记录中已经完成的 deprecated reference，不允许新跑，不允许参与新一轮 direction construction、controls、promotion、functional bridge 或 claim。

## 0.3 当前进展重新解释

v12.25 的原始进展是：functional update 已经不是完全没有信号。它从此前的 no-go 推进到：

```text
S4a-PrecommitBridgeExplorationOpened
line_f_exploration_gate_pass = 1
multisketch_any_majority_pass = 1
multisketch_max_pass_count = 4
p4_pass = 0
promotion_allowed = 0
```

关键 partial signals 包括：

```text
NG48：首次把 task/control 和 LineC majority 放到同一行，但 CEp99 爆。
NG70：post-train unlabeled calibration 压住 CEp99，出现单点 exploration。
NG103/NG106：weight-anchor + bootstrap_mom 出现 single-seed strict majority。
NG127/NG129：epochs=12 + calibration/anchor 让 exploration 接近 train-shuffle robust。
NG144/NG156：low-rank / cross-ref coupling 复现 S4a 近似信号。
```

但是这些信号大多建立在 B320-current 的历史 anchor 语境下。现在 label-informed init 必须去掉，所以 v12.25 的 S4a 不能直接作为 official functional re-entry。它只能提供机制线索：

```text
1. direct/gain role 可能提供 task/control margin；
2. quad/reservoir role 可能提供 LineC geometry；
3. post-calibration / weight-anchor / bootstrap_mom 可以部分修 tail / NLL / ECE；
4. 但这些必须在 label-free base 上重新验证。
```

## 0.4 v12.26.1 的核心问题

本版要回答三个更干净的问题：

$$
\boxed{
Q1: \text{能否构造一个 label-free B320-like base，保住 B320-current 的 task / efficiency / LineC 大部分优势？}
}
$$

$$
\boxed{
Q2: \text{在 label-free base 上，v12.25 的 S4a functional mechanisms 是否仍能产生 precommit-safe partial signal？}
}
$$

$$
\boxed{
Q3: \text{能否把 task/control margin 与 LineC geometry 改善在同一 train-shuffle seed、同一 candidate、同一 policy 下稳定同位？}
}
$$

如果 Q1 失败，functional 只能做 shadow diagnostic，不能 promotion。  
如果 Q1 近似成功但 Q2 失败，说明此前 S4a 信号依赖 label-informed anchor。  
如果 Q1/Q2 成功但 Q3 失败，说明 functional direction 仍然不是 S5，而是 S4a 局部机制。

---

# 1. 全局硬约束与禁用项

## 1.1 Label-free hard contract

所有 official / exploratory promotion candidate 必须满足：

```text
uses_y_for_stats = 0
trainprobe_signal_init_applied = 0
probe_dirs_from_labels = 0
trainprobeP_enabled = 0
trainprobeDirect_enabled = 0
label_used_for_initialization = 0
small_label_oracle_used = 0
shuffled_label_init_used = 0
validation_or_test_used_for_init = 0
dataset_name_used_for_init = 0
```

强制禁用 token：

```text
trainProbeP
trainprobedirect
trainprobe
smallLabelOracle
oracleLabel
labelInit
classMeanFromY
yForStats
```

如果 Codex 发现某 candidate 名称含这些 token，即使 `y_for_stats=None`，也不得进入 official route。原因是 variant token 本身容易触发隐藏路径或造成复盘语义混乱。若确实要复用旧结构，必须新建显式无标签版本，并在 candidate id 中写明：

```text
noTrainProbeToken
labelFreeP
noYStats
```

## 1.2 Functional loss-agnostic hard contract

Functional direction 不允许使用：

```text
label
CE vector
per-example CE
permuted-label CE
validation/test/future outcome
query batch reference
LineC hard release target
CEp99 / NLL / ECE gradient
dataset name branch
```

允许使用：

```text
train-stream unlabeled activations
train-stream logits, without labels
random cotangent gradient sketches
optimizer update as opaque vector, if label/CE not decomposed
precommit clone-probe response, only if no label/CE/validation/test/future/query is read
unlabeled calibration / logit covariance / confidence-free scale statistics
```

CE / NLL / ECE / CEp99 只能作为 audit / safety gate：

$$
CEp99_{source} \le CEp99_{NoOp}+\epsilon_{tail}.
$$

它们不能作为 direction source。

## 1.3 Fail-closed, explore-open hard rule

Promotion 必须 fail-closed：gate 不过绝不 promotion。  
Exploration 必须 continue-open：gate 不过不能立刻停，必须继续预注册 fallback。

`final_stop_allowed = 1` 只能来自：

```text
1. official success reached；
2. hard compute budget truly exhausted；
3. user_stop_flag = 1；
4. 所有 fallback depths 都执行完，并写出机制级 no-go boundary。
```

不能由以下条件触发 final stop：

```text
single gate fail；
no-go boundary written；
next hypothesis generator written；
only one fallback depth executed；
Line A base fail；
Line F functional fail。
```

若未执行完整 fallback depth，则 route 必须是：

```text
R0-ExploreDepthIncomplete
```

---

# 2. 总体实验结构

v12.26.1 采用七条并行线：

```text
Line R: 代码与 provenance 审查，尤其 label-free contract。
Line A: Label-free B320-like base reconstruction。
Line C: Manifold-Channel Geometry Diagnostics 与 LineC protocol hardening。
Line T: Precommit loss-agnostic value source。
Line F: Label-free functional bridge，复验 S4a mechanisms。
Line P: Policy-aware P3/P4 alignment。
Line D: Classic no-BSpline monitor，低预算不抢主线。
Line Z: Finalizer / route / no-go / next queue。
```

本版优先级：

```text
Priority 1: Remove label-informed init and establish label-free base near-anchor。
Priority 2: Re-test S4a functional mechanisms only on label-free candidates。
Priority 3: Solve task/control and LineC co-location under train-shuffle robustness。
Priority 4: Keep classic family only as low-priority monitor。
```

---

# 3. Line R：代码审查与 label-free contract

## 3.1 目标

确认本轮没有任何 hidden label-informed initialization、hidden CE direction、query reference 或 dataset branch。Codex 必须在复盘中写出核心代码位置、调用链、tensor shapes 和语义，不允许只写 summary。

## 3.2 必须审查的核心路径

Codex 必须定位并解释：

```text
R1: model construction entrypoint
R2: SimpleFastTaskGeometryKAN.__init__
R3: y_for_stats / build_y_stats_for_mode / probe_dirs path
R4: direct_readout init path
R5: quad_proj / projection frame init path
R6: signalBroad / signalBlock path
R7: label-free candidate registry
R8: LineC metric computation
R9: precommit feature builder
R10: functional event construction
R11: P3/P4 controls
R12: timing / memory profiler
R13: finalizer / route decision
```

每项必须写入：

```text
file_path
line_start
line_end
symbol
called_by
calls_into
uses_label
uses_ce_vector
uses_y_for_stats
uses_validation_or_test
uses_dataset_name_branch
uses_query_batch
uses_linec_hard_target_for_direction
artifact_fields_written
codex_summary
manual_review_required
```

## 3.3 Artifact

```text
v1226_label_free_code_audit.csv
v1226_label_free_symbol_map.json
v1226_forbidden_token_scan.csv
v1226_feature_provenance.csv
v1226_direction_provenance.csv
v1226_route_decision.json
```

## 3.4 Gate

Line R pass：

```text
forbidden_token_official_count = 0
uses_y_for_stats_official_count = 0
uses_label_for_init_count = 0
uses_ce_for_direction_count = 0
uses_query_batch_for_direction_count = 0
missing_core_code_refs = 0
```

如果失败，Codex 必须自动尝试：

```text
1. remove trainprobe tokens from candidate registry；
2. force y_for_stats=None at construction；
3. add runtime assertion in model init；
4. rerun smoke audit；
5. regenerate route。
```

如果自动修复后仍失败，route：

```text
R0-LabelInformedInitViolation
```

但不允许把失败写成 scientific no-go。

---

# 4. Line A：Label-free B320-like base reconstruction

## 4.1 目标

构造一个不使用标签初始化的 B320-like base，尽量恢复 B320-current 的效率、task、AUC 与 LineC。它是所有 official functional update 的前提。

## 4.2 核心假设

B320-current 的强表现来自两部分：

```text
H-A1: FHQ / hinge / absdiag / classbranch / identitytailquad primitive 的确有高效 task-geometry 能力；
H-A2: label-informed trainprobe 初始化提供了额外 signal frame；
H-A3: 如果用 label-free train-stream geometry 构造替代 signal frame，有机会恢复大部分优势。
```

本线要检验 H-A3。

## 4.3 候选 family

所有候选必须不含 trainprobe token。

```text
A-LF0: noTrainProbe baseline
  去掉 trainprobe token，保留 FHQ primitive。

A-LF1: Orthogonal / SRHT label-free projection bank
  使用固定随机正交 / SRHT / low-coherence projection，不读 label。

A-LF2: Unlabeled PCA / covariance frame
  用 train-stream x 或 hidden activations 的无标签 covariance 初始化 P。

A-LF3: Augmentation-stable frame
  用无标签 augment consistency 找稳定方向。

A-LF4: Residual-on-A1 low-rank frame
  不替换整个 projector，只在 A1-noYForStats 上加低秩 residual。

A-LF5: Role-energy balanced frame
  用 direct / quad / branch role energy 的无标签统计平衡初始化。

A-LF6: LineC-aware but label-free frame
  只使用 CouplingR2 / unlabeled train-probe drift，不使用 NoiseSignalLeak / Reservoir hard target。

A-LF7: EMA covariance adapt
  训练早期用无标签 covariance EMA 低频更新 projection frame。
```

禁止候选：

```text
small label oracle
shuffled label oracle
unsupervised clusters that are later matched to labels
any candidate with trainprobe token retained
```

若需要上界诊断，只能写成 historical note，不跑新实验。

## 4.4 两级预算

Scout：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 512
val/test = 256
epochs = 3
LineC batch = 32
sketch_dim = 8
```

Hardening：自动选择 Scout top-3，不能由 Codex 手动停止。

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024
val/test = 512
epochs = 8
LineC batch = 64
sketch_dim = 24
```

Final label-free confirm：只有 hardening 达到 near-anchor 才跑。

```text
seeds = 0..9
train_size = 1024 or 2048
val/test = 512
epochs = 8 or 12
```

## 4.5 记录指标

```text
candidate_id
dataset
seed
uses_y_for_stats
forbidden_token_present
step_ratio_q90
memory_ratio_q90
mean_delta_vs_mlp
worst_delta_vs_mlp
near_pass_rate
AUC_step_ratio
AUC_time_ratio
ECE_delta
NLL_delta
CEp99_delta
LineC_pass_count
LineC_pass_rate
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
signal_mass_topk
reservoir_fraction
logit_drift_p95
feature_rank
role_energy_direct
role_energy_quad
role_energy_branch
```

## 4.6 Gates

Official label-free base：

$$
uses\_y\_for\_stats=0.
$$

$$
forbidden\_token\_present=0.
$$

$$
T_{step,q90}/T_{MLP}\le 1.00.
$$

$$
M_{q90}/M_{MLP}\le 0.30.
$$

$$
mean\_delta\_{vsMLP}\ge 0.
$$

$$
worst\_delta\_{vsMLP}\ge -0.005.
$$

$$
\max(AUC_{step},AUC_{time})\le 1.00.
$$

$$
ECE_{KAN}\le ECE_{MLP}+0.02.
$$

$$
LineC\_pass\_rate=1.0.
$$

Near-anchor exploration：

$$
mean\_delta\_{vsMLP}\ge -0.005.
$$

$$
worst\_delta\_{vsMLP}\ge -0.02.
$$

$$
\max(AUC_{step},AUC_{time})\le 1.10.
$$

$$
LineC\_pass\_rate\ge 0.5.
$$

## 4.7 如果不满足条件，Codex 必须先尝试

若 task 接近但 LineC 失败：

```text
1. add residual-on-A1 low-rank frame rather than replacing full P；
2. reduce role imbalance with label-free role-energy normalization；
3. add train-probe unlabeled coupling-preserving regularized initialization, not loss term；
4. run LineC detailed decomposition to identify noise vs reservoir vs coupling failure；
5. rerun top-2 hardening。
```

若 LineC 接近但 task fail：

```text
1. add identitytailquad / absdiag scale bracket without labels；
2. adjust role gain schedule, but dataset-agnostic；
3. check AUC-step vs AUC-time to separate trajectory vs timing；
4. try lower-rank residual frame before wider model。
```

若 both fail：

```text
1. route candidate to FamilyRejectedForThisVersion；
2. switch to next candidate family；
3. do not final stop until all Line A families executed。
```

---

# 5. Line C：Manifold-Channel Geometry Diagnostics

## 5.1 目标

Line C 要判断 label-free base 和 functional event 是否真的改善训练几何，而不是只让 accuracy 或 CouplingR2 局部变好。

Line C 仍可使用 label / CE 构造 audit-only metrics，但这些字段不能进入 direction source。

## 5.2 核心指标

Train-probe coupling：

$$
\Delta U_B=f_{\theta_{t+\Delta}}(B)-f_{\theta_t}(B),
$$

$$
\Delta U_Q=f_{\theta_{t+\Delta}}(Q)-f_{\theta_t}(Q).
$$

$$
A_t=\arg\min_A \|\Delta U_Q-A\Delta U_B\|_F^2+\lambda\|A\|_F^2.
$$

$$
CouplingR^2=1-\frac{\|\Delta U_Q-A_t\Delta U_B\|_F^2}{\|\Delta U_Q\|_F^2+\epsilon}.
$$

Signal/reservoir audit：

$$
RealSignalReservoirRatio=
\frac{\|P_{res}r_{real}\|^2}{\|r_{real}\|^2+\epsilon}.
$$

$$
NoiseSignalLeak=
\frac{\|P_{sig}r_{noise}\|^2}{\|r_{noise}\|^2+\epsilon}.
$$

Kernel drift 不作为越小越好的指标：

$$
KernelDrift=
\frac{\|\hat K(t)-\hat K(0)\|_F}{\|\hat K(0)\|_F+\epsilon}.
$$

## 5.3 Multi-sketch robustness

每个 candidate 必须用多个 sketch source 复核：

```text
sketch_id = random_cotangent_0..4
batch_size = 64 and 128 if budget allows
rank = 3 and 5
window = 3,5,10
```

LineC pass 不再只看单一 sketch。

## 5.4 记录指标

```text
candidate_id
method
control_id
dataset
seed
train_shuffle_seed
window
sketch_id
batch_size
rank
CouplingR2
CouplingCorr
NoiseSignalLeak
RealSignalReservoirRatio
KernelDrift
signal_mass_topk
reservoir_fraction
top_eigen_share
CEp99
NLL
ECE
margin_p10
logit_drift_p95
linec_pass
linec_fail_reason
is_audit_only_metric
used_for_direction
```

## 5.5 Gates

Functional / base LineC pass：

$$
\Delta CouplingR^2\ge 0.02.
$$

$$
\Delta NoiseSignalLeak\le -0.01.
$$

$$
\Delta RealSignalReservoirRatio\le -0.01.
$$

Tail safety：

$$
CEp99_{source}\le CEp99_{NoOp}+0.05.
$$

$$
NLL_{source}\le NLL_{NoOp}+0.01.
$$

$$
ECE_{source}\le ECE_{NoOp}+0.02.
$$

For base candidate, absolute nontearing gate：

```text
CouplingR2 >= MLP - 0.02
NoiseSignalLeak <= MLP + 0.02
RealSignalReservoirRatio <= MLP + 0.02
```

## 5.6 如果不满足条件，Codex 必须先尝试

If CouplingR2 good but NoiseSignalLeak bad：

```text
1. classify as coupling-only movement；
2. do not promote；
3. add noise-leak veto in Line F；
4. inspect role contribution: direct/gain vs quad/reservoir。
```

If NoiseSignalLeak good but task/control fail：

```text
1. treat as geometry-only direction；
2. test sequential guard-then-task event；
3. do not lower task gate。
```

If LineC unstable across train-shuffle seeds：

```text
1. build failure atlas by train_shuffle_seed；
2. do not tune per seed；
3. construct risk-aware veto/downscale policy based on precommit state features。
```

---

# 6. Line T：Precommit loss-agnostic value source

## 6.1 目标

找到不读 label / CE / future / validation 的 precommit feature，使其能预测 functional event 是否会同时带来 task-control gain 和 LineC gain。

## 6.2 Feature tiers

T1 official deployable features：

```text
unlabeled logit covariance spectrum
unlabeled output drift spectrum
activation / branch occupancy entropy
primitive role energy balance
projector stability
random-cotangent gradient sketch spectrum
optimizer-update opaque statistics, without reading label/CE components
train-stream window stability
precommit clone micro-probe with unlabeled response only
```

T2 diagnostic features：

```text
completed response deltas
post-event LineC deltas
clone-probe after using candidate response
method identity flags
```

T3 audit-only targets：

```text
NoiseSignalLeak hard release
RealSignalReservoirRatio hard release
CEp99 / NLL / ECE deltas
accuracy deltas
```

Official scorer only uses T1.

## 6.3 Visibility tasks

```text
Task T-a: predict LineC majority pass。
Task T-b: predict task/control positive margin。
Task T-c: predict CEp99/NLL/ECE safety。
Task T-d: predict joint S4a candidate。
Task T-e: predict S5 candidate, if support exists。
```

## 6.4 Metrics

```text
AUC_joint
AUC_task
AUC_linec
AUC_tail_safe
precision_at_k_joint
recall_at_k_joint
calibration_ECE_of_scorer
leave_dataset_out_AUC
leave_seed_out_AUC
leave_train_shuffle_out_AUC
support_count
support_concentration
feature_ablation_gain
```

## 6.5 Gate

Exploration visibility：

$$
AUC_{joint}\ge 0.65.
$$

$$
precision@k_{joint}\ge 0.20.
$$

$$
recall@k_{joint}\ge 0.10.
$$

Official visibility：

$$
AUC_{joint}\ge 0.75.
$$

$$
precision@k_{joint}\ge 0.40.
$$

$$
recall@k_{joint}\ge 0.25.
$$

and no support collapse：

```text
support must cover at least 2 datasets and 2 seeds;
support must not be concentrated in one train_shuffle_seed;
leave-one-dataset AUC >= 0.65。
```

## 6.6 如果不满足条件，Codex 必须先尝试

If T1 fails but T2 passes：

```text
1. distill T2 response feature into T1 precommit proxy;
2. do not promote T2;
3. run T1B optimizer-update opaque features;
4. run sign-invariant and scale-invariant ablations。
```

If T1 AUC moderate but precision low：

```text
1. variable-k threshold scan;
2. class-agnostic calibration;
3. train-shuffle leaveout calibration;
4. hard-negative mining from false positives。
```

If support concentrated：

```text
1. run additional train-shuffle seeds;
2. report concentration;
3. no promotion;
4. build risk-aware policy in Line P。
```

---

# 7. Line F：Label-free functional bridge

## 7.1 目标

在 label-free base 或 label-free near-anchor 上复验 v12.25 的 S4a mechanisms，并尝试把它们推到 S5。

Important：B320-current 禁用。本线只能使用 Line A 的 label-free candidates。

## 7.2 Functional candidate families

F1 direct/gain task source：

```text
direct_gain_centered_denoise
low_budget_direct_only
tail_budgeted_direct_branch
```

F2 quad/reservoir geometry source：

```text
control_residual_quad_direct
quad_reservoir_guard
low_rank_quad_coupling
```

F3 composite event：

```text
direct/gain task event + quad/reservoir geometry guard
```

F4 post-event unlabeled calibration：

```text
unlabeled logit covariance calibration
unlabeled confidence-free scale correction
weight anchor without labelInit reference
bootstrap_mom from train-stream only
```

F5 control residualization：

```text
project event away from AdamWParallel / RandomMatchedNorm / SNR-only control span
```

F6 sequential policy：

```text
geometry guard event -> task event
or
small task event -> geometry correction event
```

## 7.3 Candidate construction constraints

Every functional row must record：

```text
base_candidate_id
base_is_label_free
uses_label_for_init
uses_label_for_direction
uses_ce_for_direction
uses_query_reference
uses_validation_or_test
uses_linec_hard_target_for_direction
precommit_available
control_residualized
policy_id
role_mix_direct
role_mix_quad
role_mix_branch
lambda
norm_budget
```

## 7.4 Controls

Every candidate must be compared to：

```text
NoOpMatchedOverhead
RandomMatchedNorm
AdamWParallelDirection
SNR-only
MatchedRoleEnergyRandom
ShuffledFunctionalPayload
SameNormDirectOnlyControl
SameNormQuadOnlyControl
MLPAnalogMaintenance, if applicable
```

## 7.5 Gates

S4a exploration：

$$
source\_vs\_noop\ge 0.005.
$$

$$
source\_vs\_best\_control\ge 0.003.
$$

$$
LineC\_pass\_count\ge 3/5.
$$

$$
CEp99_{source}\le CEp99_{NoOp}+0.10.
$$

S5 official：

$$
source\_vs\_noop\ge 0.005.
$$

$$
source\_vs\_best\_control\ge 0.005.
$$

$$
LineC\_pass\_count=5/5.
$$

$$
CEp99_{source}\le CEp99_{NoOp}+0.05.
$$

$$
NLL_{source}\le NLL_{NoOp}+0.01.
$$

$$
ECE_{source}\le ECE_{NoOp}+0.02.
$$

Train-shuffle robustness：

```text
at least 3/3 train-shuffle seeds pass S4a;
at least 2/3 train-shuffle seeds pass S5-near;
no single train-shuffle seed accounts for all success。
```

## 7.6 如果不满足条件，Codex 必须先尝试

If task/control positive but LineC fail：

```text
1. add quad/reservoir guard event;
2. downscale direct/gain role;
3. try sequential geometry guard -> task event;
4. run policy-aware P3 with event+planned policy;
5. do not simply rescan lambda。
```

If LineC positive but task/control fail：

```text
1. add direct/gain task residual after geometry event;
2. test control residualization;
3. compare against same-norm controls;
4. do not promote geometry-only rows。
```

If CEp99/NLL/ECE fail：

```text
1. apply loss-agnostic post-event logit covariance calibration;
2. use confidence-free scale correction, not CE calibration;
3. apply weight anchor to label-free base weights only;
4. if tail repair erases task signal, mark as tail-task conflict and go to Line P。
```

If only one train-shuffle seed fails repeatedly：

```text
1. build geometry-risk atlas;
2. do not tune for that seed;
3. learn dataset-agnostic risk veto/downscale rule from precommit features;
4. test across all datasets/seeds。
```

---

# 8. Line P：Policy-aware P3/P4 alignment

## 8.1 目标

v12.25 显示 P3/P4 解耦：某些 event P3 pass 但 P4 fail，某些 event P4 gain 但 P3 fail。本线要让 P3 测的是：

$$
\text{functional event} + \text{planned post-event policy}
$$

而不是只测 event 本身。

## 8.2 Policies

```text
P0: event only
P1: event + unlabeled logit covariance calibration
P2: event + weight anchor to label-free base
P3: event + bootstrap_mom train-stream reference
P4: geometry guard -> task event
P5: task event -> geometry correction
P6: risk-aware downscale / veto
```

## 8.3 Metrics

```text
policy_id
precommit_risk_score
post_event_task_margin
post_policy_task_margin
post_event_LineC_pass_count
post_policy_LineC_pass_count
CEp99_before_after_policy
NLL_before_after_policy
ECE_before_after_policy
control_gap_after_policy
train_shuffle_pass_count
```

## 8.4 Gates

Policy is useful if：

$$
\Delta task\_margin_{policy}\ge \Delta task\_margin_{event}-0.002.
$$

$$
LineC\_pass\_count_{policy}\ge LineC\_pass\_count_{event}.
$$

$$
CEp99_{policy}\le CEp99_{NoOp}+0.05.
$$

$$
control\_gap_{policy}\ge 0.005.
$$

## 8.5 如果不满足条件，Codex 必须先尝试

```text
1. compare event-only vs event+policy to identify which policy component destroys signal；
2. run role-specific ablation；
3. try two-event sequence with smaller individual norm budgets；
4. if all policies destroy signal, switch primitive mechanism instead of scanning alpha。
```

---

# 9. Line D：Classic no-BSpline monitor

## 9.1 目标

Keep Rational / Fourier / Chebyshev / Wavelet / RBF/FastKAN active only as low-priority portfolio. B-spline remains frozen.

## 9.2 Active family

```text
Rational
Fourier
Chebyshev
Wavelet
RBF/FastKAN
```

## 9.3 This-version priority

```text
Rational: memory blocker targeted repair and LineC coupling collapse audit.
Fourier: lowfreq expression repair only if cheap.
Chebyshev/Wavelet/RBF: monitor only unless new hypothesis is explicit.
```

## 9.4 Gates

FamilyPass requires：

```text
L3 efficiency pass
A4 expression pass
A5 task/AUC pass
LineC nontearing pass
label-free / no CE / no dataset branch
```

If not executed, status must be：

```text
NotExecuted_BudgetDeferred
```

not：

```text
Rejected
```

---

# 10. Artifacts and figures

## 10.1 Required CSV / JSON

```text
v1226_label_free_code_audit.csv
v1226_forbidden_token_scan.csv
v1226_label_free_base_manifest.csv
v1226_label_free_base_scout.csv
v1226_label_free_base_hardening.csv
v1226_label_free_base_linec.csv
v1226_linec_multisketch.csv
v1226_precommit_value_features.csv
v1226_visibility_scores.csv
v1226_functional_candidates.csv
v1226_functional_controls.csv
v1226_policy_alignment.csv
v1226_train_shuffle_robustness.csv
v1226_failure_atlas.csv
v1226_classic_monitor.csv
v1226_required_artifact_manifest.csv
v1226_fallback_manifest.csv
v1226_route_decision.json
v1226_no_go_boundary.md
v1226_next_hypothesis_queue.md
```

## 10.2 Required figures

```text
fig_label_free_base_task_vs_linec.svg
fig_label_free_base_auc_time.svg
fig_forbidden_token_scan.svg
fig_linec_multisketch_heatmap.svg
fig_task_vs_linec_colocation.svg
fig_train_shuffle_failure_atlas.svg
fig_ce_tail_safety.svg
fig_functional_control_gap.svg
fig_policy_alignment_before_after.svg
fig_value_source_roc_pr.svg
fig_classic_family_status.svg
```

---

# 11. Route definitions

```text
S1-LabelFreeBaseOfficial:
  Label-free base passes official task/efficiency/LineC gates.

S2-LabelFreeBaseNearAnchor:
  Label-free base passes exploration near-anchor gates; functional may run shadow diagnostic but not official.

S3-LabelFreeFunctionalS4aOpened:
  Label-free candidate obtains task/control + LineC majority exploration signal.

S4-LabelFreeFunctionalP4Opened:
  Policy-aware P3 opens P4 under label-free base.

S5-OfficialFunctionalSuccess:
  Label-free base + functional passes task/control, LineC all-pass, tail safety, train-shuffle robustness, and matched controls.

R1-LabelFreeBaseMissing:
  No label-free candidate reaches near-anchor after all Line A fallback depths.

R2-ValueSourceInvisible:
  Label-free base near-anchor exists, but T1 precommit value source cannot predict joint success.

R3-TaskLineCMismatch:
  task/control signal and LineC signal exist but cannot co-locate across train-shuffle seeds.

R4-TailSafetyConflict:
  task/control + LineC co-locate but CEp99/NLL/ECE fail and loss-agnostic calibration cannot repair.

R5-ControlsExplainFunctional:
  source beats NoOp but not matched controls.

R0-LabelInformedInitViolation:
  Any official candidate or direction uses label-informed initialization.

R0-ExploreDepthIncomplete:
  Required fallback depths not executed.
```

---

# 12. Mandatory fallback depths

Codex must execute at least the following depths before final no-go.

## Depth 1：Label-free base scout and hardening

```text
Run A-LF0..A-LF7 scout.
Promote top-3 to hardening.
Record failure atlas.
```

## Depth 2：Functional S4a transfer to label-free base

```text
Run F1/F2/F3/F4/F5/F6 on top-2 label-free candidates.
Use full controls.
Record train-shuffle robustness.
```

## Depth 3：Geometry-risk-aware policy

```text
Build risk atlas from failed seeds.
Test veto/downscale/role-switch/sequential policy.
```

## Depth 4：Primitive replacement or mechanism no-go

```text
If co-location still fails, test a different functional primitive family:
  role-specific gain transport;
  quad-reservoir guard;
  low-rank logit-subspace event;
  unlabeled covariance transport.
If all fail, write mechanism-level no-go, not just candidate-level no-go.
```

Final stop before Depth 4 is not allowed unless user_stop_flag=1 or hard budget is genuinely exhausted and recorded.

---

# 13. Success criteria

Minimum Success A：label-free base near-anchor

```text
At least one label-free candidate reaches near-anchor exploration gate.
```

Minimum Success B：label-free S4a transfer

```text
At least one label-free base + functional event gets source_vs_control > 0 and LineC >= 3/5.
```

Minimum Success C：policy-aware co-location

```text
Task/control margin, LineC majority, CEp99/NLL/ECE safety co-locate on at least 2/3 train-shuffle seeds.
```

Minimum Success D：S5 official functional success

```text
Label-free base + functional passes all official S5 gates.
```

Minimum Success E：mechanism-level no-go

```text
All four fallback depths executed;
no label-informed paths;
all artifacts complete;
no-go identifies whether blocker is base, value source, actuator, tail, controls, or co-location.
```

---

# 14. Final execution note for Codex

Codex must not optimize for producing a clean route quickly. It must optimize for resolving the scientific blocker.

Hard instructions:

```text
1. Do not run or instantiate B320-current labelInit path.
2. Do not keep trainprobe tokens in official candidate names.
3. Do not use y_for_stats under any official or exploration candidate.
4. Do not stop after first label-free base fail.
5. Do not stop after first functional fail.
6. If S4a appears on a label-free candidate, continue to policy-aware S5 attempts.
7. If only task improves, diagnose LineC mismatch.
8. If only LineC improves, diagnose task/control mismatch.
9. If CEp99/NLL/ECE fail, try only loss-agnostic tail calibration.
10. If all fail, write mechanism-level no-go with completed fallback evidence.
```

The key scientific question for v12.26.1 is no longer whether B320-current is strong. It is:

$$
\boxed{
\text{Can the project survive after removing label-informed initialization?}
}
$$

If yes, DG-KAN has a much cleaner path toward an external-ready next-generation MLP claim. If no, then the project must honestly separate supervised-initialized KAN anchors from label-free architecture / functional-update claims.
