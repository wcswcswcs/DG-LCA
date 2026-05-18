# DG-KAN v9.5.3 Geometry-Quality Functional Update / Signal-Channel Primitive / Anti-Forgetting Closure 完整实验计划

> 本计划基于 v9.5.2 的真实执行结果、Deep Manifold Part 2、以及 A Theory of Generalization in Deep Learning 重新制定。  
> 本轮不再继续给 APX / APY / CERT 阈值打补丁，也不再只问“某个动作是否让 acc 涨”。  
> v9.5.3 的核心目标是把“好动作”重新定义为：一个额外参数改动必须让模型的表示空间变得更容易训练、更不容易记噪声、更不容易忘旧知识，并且能在短期、中期、长期都不伤模型。

---

# 0. 一句话结论

v9.5.2 已经说明：在当前 canonical AP0 action universe 里，能同时满足短期收益、中期不坏、长期安全、bad/null 可控、coverage 足够的动作集合太稀疏；APY1-APY8 虽然工程上都能生成 payload、certificate 和 branch-horizon outcomes，但没有生成 value-positive / horizon-safe actions。

所以 v9.5.3 不应继续问：

```text
能不能再调一个 target threshold？
能不能再调 APY3？
能不能再调 CERT30？
```

v9.5.3 应该问：

```text
什么样的参数改动会让 KAN 的“几何形状”更好？
这个更好的几何形状能否带来更快收敛、更强抗噪声、更强泛化、更少遗忘？
训练当下能否检测或生成这种参数改动？
```

本轮核心命题是：

$$
\boxed{
\text{functional update 应该先改变训练几何，再通过训练几何带来 acc / loss / robustness 改善。}
}
$$

---

# 1. 为什么 v9.5.2 之后必须换思路

## 1.1 v9.5.2 的真实结果

v9.5.2 的 route 是：

```text
R1-CanonicalActionDensityInsufficientForMultiHorizonTarget
```

关键数据：

```text
target_lattice_candidate_count = 8000
official_target_candidate_count = 0
weak_target_candidate_count = 0
selected_target_action_count = 27
selected_target_coverage = 0.009388038942976356
selected_target_V_integrated_lcb = 0.24423130340766275
selected_target_h240_longrisk = 0.0
selected_target_bad_event = 0.1111111111111111
selected_target_null_event = 0.25925925925925924
```

coverage 下限是 $0.03$，full action count 是 $2876$，所以至少需要：

$$
N_{min}=\lceil 0.03\times 2876\rceil = 87.
$$

现在 best target 只有 $27$ 个 action，差至少 $60$ 个 action。

更重要的是，它不只是数量少。它的 bad / null 也不合格：

```text
bad_event = 0.1111 > 0.05
null_event = 0.2593 > 0.15
```

按 $27$ 个 action 计算，bad 约 $3$ 个，null 约 $7$ 个。若要满足 gate：

$$
BadCount \le 0.05\times 27 = 1.35,
$$

$$
NullCount \le 0.15\times 27 = 4.05.
$$

所以 best diagnostic target 同时存在三类问题：

```text
数量不够；
bad 太多；
null 太多。
```

这不是简单调权重能解决的问题。

## 1.2 APY1-APY8 不是没实现，而是生成方向错了

v9.5.2 中 APY1-APY8 工程链路已经过了：

```text
generated actions = 512
payload/certificate hash missing = 0 / 0
action apply L∞ max = 0.0
no-transform equivalence pass = 1
branch-horizon rows = 9216 / 9216
unresolved exception = 0
```

但 best primitive `APY3-LongRiskBarrierPrimitive` 的结果是：

```text
target precision = 0.046875
V_integrated LCB = -0.9770587887784598
h240 long-risk = 0.71875
```

这说明 APY 不是“跑不起来”，而是生成出来的动作在训练几何上没有真正对准“有用且安全”的方向。

## 1.3 CERT30 有 AUC，但不能选动作

v9.5.2 中 best certificate 是 `CERT30-BasisEdgeLocalityCertificate`：

```text
AUC_target = 0.691913214990138
TopK64 target precision = 0.015625
TopK64 long-risk = 0.671875
```

AUC 看起来有一点排序信号，但 controller 真正在乎的是 top accepted region。TopK64 里只有约 $1$ 个 target action，却有约 $43$ 个 long-risk action。这个 certificate 不能用于 functional update。

结论：

$$
\boxed{
\text{现在不是缺一个更好的阈值；是动作生成和动作证书没有描述正确的训练几何。}
}
$$

---

# 2. 相关工作给我们的启发

## 2.1 Deep Manifold 的启发：好动作应该让局部片区更稳，而不是只让某个 batch loss 下降

Deep Manifold 把网络看成很多层叠的小片区，每个节点或边在局部片区里工作。训练不是在一个固定坐标系里拟合曲线，而是在不断移动坐标、调整局部片区、寻找稳定点。

这给 DG-KAN 一个直接启发：

```text
一个 functional update 不应该只看 loss/acc；
它应该检查这次参数改动有没有让局部片区更容易稳定，
有没有让多个片区之间的过渡更平滑，
有没有避免某些片区变得过硬或过脆。
```

Deep Manifold 还强调：训练中局部片区会变硬，弹性下降；dropout、skip、normalization 的作用可以理解为延缓这种变硬、保持 residual 修正稳定。对 DG-KAN 来说，我们可以把 functional update 设计成一种“受控的小修正”，它不是为了猛拉 loss，而是为了避免局部 KAN edge basis 过早僵硬。

## 2.2 Generalization paper 的启发：好动作应该把更新放进“信号方向”，不要把噪声也学进去

A Theory of Generalization in Deep Learning 给了另一个非常实用的判据：训练过程中，有些输出方向会影响测试集，这些方向可以看成 signal channel；另一些方向虽然能降低训练 loss，但对测试集不可见，可以看成 noise reservoir。它还给出一个简单的 per-parameter gate：当梯度均值平方大于 minibatch 方差项时才更新，即：

$$
\mu_k^2 > \frac{\sigma_k^2}{b-1}.
$$

这对 DG-KAN 很有用。我们不能只问：

```text
这个动作能不能降低当前 batch loss？
```

而要问：

```text
这个动作的方向是不是由多个样本一致支持？
还是只是在记住某几个样本的噪声？
```

所以 v9.5.3 需要新增一组训练当下可算的指标：

```text
per-edge / per-block gradient mean；
per-edge / per-block gradient variance；
SNR = mean² / variance；
functional action 与 high-SNR AdamW direction 的夹角；
functional action 是否主要落在 low-SNR noisy subspace；
leave-one-out batch risk estimate。
```

## 2.3 两篇工作的共同启发

两篇工作合在一起，给出一个更完整的“好几何”定义：

```text
Deep Manifold 关注：局部片区、固定点、曲率、弹性、扰动吸收、长期稳定。
Generalization theory 关注：训练动作是否走在 signal channel，是否避开 noise reservoir，是否有 population-risk 意义。
```

因此，DG-KAN 的好动作不应该再定义为一个二元标签，而应该定义为一张 geometry card：

```text
这个动作是否短期有效？
这个动作是否中期稳定？
这个动作是否长期安全？
这个动作是否降低 hard tail？
这个动作是否保持旧知识？
这个动作是否有高 SNR 支持？
这个动作是否让局部 KAN basis 不塌缩？
这个动作是否让多个数据片区都受益，而不是只服务某个小片区？
这个动作是否便宜？
```

---

# 3. v9.5.3 的新定义：什么是一个好的 functional update 动作

一个动作 $a$ 不再只用 acc 或 target label 判断，而用下面这组指标判断。

## 3.1 基础可执行性

动作必须先满足：

```text
payload hash 不缺失；
action apply error L∞ 接近 0；
no-transform replay 等价；
negative controls 有 divergence；
没有 fake / proxy / old table official；
没有使用 dataset_name / validation / test / future outcome at commit。
```

对应 gate：

$$
ApplyOK(a)=1.
$$

## 3.2 打过强对照

对每个 horizon $h\in\{20,80,240\}$，定义：

$$
V_h(a)=Score_h(RealFunctional,a)-\max_{b\in Controls}Score_h(b,a),
$$

其中：

```text
Controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledPayload。
```

一个动作不能只是比 NoOp 好，它必须至少在关键 horizon 上打过 AdamWParallel / bestLR。

## 3.3 短期、中期、长期都要看

记录：

```text
V20, V80, V240；
CEp99_delta_20/80/240；
MarginP10_delta_20/80/240；
NLL_delta_20/80/240；
ECE_delta_20/80/240；
LongRisk_240；
Bad_20/80/240；
Null_20/80/240。
```

一个 official 候选动作至少需要：

$$
LCB(V_{20})>0,
$$

$$
LCB(V_{80})\ge -\epsilon_{80},
$$

$$
UCB(LongRisk_{240})\le \tau_L,
$$

$$
UCB(Bad)\le \tau_B,
$$

$$
UCB(Null)\le \tau_N.
$$

## 3.4 好几何指标

我们定义一张 `GeometryCard(a)`：

```text
signal_snr_score
noise_reservoir_score
gradient_agreement_score
local_curvature_barrier_score
basis_cover_entropy
edge_basis_utilization
jacobian_spectral_proxy
margin_tail_improvement
hard_tail_fraction_delta
memory_probe_drift
old_stratum_loss_delta
plasticity_score
rigidity_score
```

这些指标不是为了写漂亮名词，而是为了回答具体问题。

### 3.4.1 是否走在多数样本支持的方向

对 action 涉及的每个参数块 $k$，记录：

$$
SNR_k=\frac{\mu_k^2}{\sigma_k^2/(b-1)+\epsilon}.
$$

其中 $\mu_k$ 是当前 batch per-example gradient 的均值，$\sigma_k^2$ 是方差。

动作的 SNR 得分：

$$
SNR(a)=\frac{\sum_k |\Delta\theta_{a,k}|\cdot \mathbf{1}[SNR_k>1]}{\sum_k |\Delta\theta_{a,k}|+\epsilon}.
$$

判断：

```text
SNR(a) 高：动作主要落在多数样本同意的方向；
SNR(a) 低：动作可能在记噪声或过拟合少数样本。
```

### 3.4.2 是否破坏局部曲率

用几个低成本 JVP / VJP probe 估计动作前后局部敏感度：

$$
CurvProxy(a)=\frac{\|J(\theta+\Delta\theta_a)v-J(\theta)v\|}{\|\Delta\theta_a\|\|v\|+\epsilon}.
$$

官方动作不能让这个值暴涨。

### 3.4.3 是否保持 basis 覆盖分布

对 KAN lifted basis 或 edge basis，记录每个 basis 的激活 / 梯度使用量：

$$
CoverEntropy=-\sum_i p_i\log(p_i+\epsilon).
$$

动作后如果 CoverEntropy 急剧下降，说明动作让少数 basis 过度占用，可能导致过拟合、抗噪差、遗忘强。

### 3.4.4 是否保护旧知识

在 memory probe 上记录：

```text
old CEp99 delta；
old margin p10 delta；
old NLL delta；
old ECE delta；
old representation drift；
old source family failure count。
```

定义：

$$
ForgetRisk(a)=\max(0,CEp99_{old,after}-CEp99_{old,before})
+\max(0,MarginP10_{old,before}-MarginP10_{old,after}).
$$

official 动作必须满足：

$$
UCB(ForgetRisk(a))\le \tau_F.
$$

## 3.5 最终动作等级

动作分成五级：

```text
E: replay / legality 不可信；
D: 有明显 bad / long-risk / forgetting；
C: 安全但基本没用；
B: 短期有用，但中期/长期或泛化证据不足；
A: 可以进入 controller 的候选动作。
```

A 级动作必须满足：

$$
GoodGeomAction(a)=1
$$

当且仅当：

$$
ApplyOK(a)=1,
$$

$$
LCB(V_{20})>0,
$$

$$
LCB(V_{80})\ge -\epsilon_{80},
$$

$$
UCB(LongRisk_{240})\le \tau_L,
$$

$$
UCB(Bad)\le \tau_B,
$$

$$
UCB(Null)\le \tau_N,
$$

$$
SNR(a)\ge \tau_{snr},
$$

$$
UCB(ForgetRisk(a))\le \tau_F,
$$

$$
Cost(a)\le C_{max}.
$$

---

# 4. v9.5.3 总体目标

v9.5.3 的目标不是直接 full functional success，而是建立一个新的判断系统：

$$
\boxed{
\text{用 geometry card 定义、生成、筛选 functional update 动作。}
}
$$

最低有效推进目标：

```text
1. 建立 canonical GeometryCard table；
2. 证明旧 target failure 是否来自目标过稀疏、动作空间不够、还是几何指标没对准；
3. 在 canonical AP0 action 上重算 geometry-action relationship；
4. 实现 APG1-APG8 geometry-aware action generator；
5. 真实 materialize APG branch-horizon outcomes；
6. 训练一个小而可解释的 geometry certificate；
7. 如果 APG 有 survivor，则打开 minimal controller 和 selected runtime；
8. 保留 Base-Acc Sentinel，但不把它用于 controller。
```

强目标：

```text
Geometry target count >= 87；
coverage >= 0.03；
bad_event <= 0.05；
null_event <= 0.15；
h240 longrisk <= 0.05；
ForgetRisk UCB <= threshold；
TopK64 geometry certificate precision >= 0.25；
TopK64 longrisk <= 0.10；
selected runtime step_ratio_q90 <= 1.50。
```

---

# 5. 核心假设

## H1：v9.5.2 的 target failure 不是因为表还不可信，而是因为当前二元 target 太窄且没有几何解释

成立标准：

```text
canonical full table ready = 1；
quality audit pass = 1；
recomputed v9.5.2 best target count = 27；
8000 lattice still no weak pass；
GeometryCard 能解释 selected target 的 bad/null/longrisk/forgetting 分布。
```

失败标准：

```text
发现 v9.5.2 target lattice 有 label bug、aggregation bug 或 row join bug。
```

## H2：好动作有共同的几何特征

成立标准：

```text
至少 3 个 geometry metrics 对 GoodGeomAction 的 AUC >= 0.70；
TopK64 GoodGeomAction precision >= 0.25；
TopK64 longrisk <= 0.15；
leave-dataset-out AUC drop <= 0.10。
```

失败标准：

```text
所有 geometry metrics AUC <= 0.60，TopK64 precision 接近 base rate。
```

## H3：SNR / population-risk gate 可以减少噪声记忆和 long-risk

成立标准：

```text
SNR-filtered actions 的 h240 longrisk 明显低于 unfiltered；
SNR-filtered actions 的 memory forget risk 明显低于 unfiltered；
SNR-filtered actions 的 integrated V LCB 不为负。
```

失败标准：

```text
SNR gate 降低 longrisk 但也把 value 全部删掉；或完全没有区分力。
```

## H4：当前 APY family 失败，因为它没有保护 basis cover / memory / SNR，而不是因为实现错误

成立标准：

```text
APY3 generated actions 的 payload/apply/branch-horizon 仍 pass；
但 GeometryCard 显示：SNR 低、CoverEntropy 降、ForgetRisk 高或 CurvProxy 高。
```

失败标准：

```text
APY3 在 GeometryCard 上很好，但 outcome 仍极差；说明 GeometryCard 还缺关键维度。
```

## H5：新的 APG generator 可以生成更多 A 级动作

成立标准：

```text
APG generated actions >= 512；
APG branch-horizon rows complete；
best APG target precision >= 0.20；
best APG V_integrated LCB > 0；
best APG h240 longrisk <= 0.15；
APG GoodGeomAction count >= 87 或 coverage >= 0.03。
```

失败标准：

```text
APG implementation pass，但 outcome 与 GeometryCard 都不形成 survivor。
```

## H6：geometry certificate 可以替代 outcome oracle

成立标准：

```text
certificate TopK64 GoodGeomAction precision >= 0.25；
certificate TopK64 longrisk <= 0.10；
certificate ECE <= 0.10；
certificate monotone sign pass = 1；
LDO/LSO TopK 不崩。
```

失败标准：

```text
AUC 有信号但 TopK collapse，或 ECE 高，或 longrisk 高。
```

---

# 6. 实验阶段总览

v9.5.3 分成 16 个阶段。为了避免继续“一轮只发现一个 blocker”，本轮必须并行跑 P1/P2/P3/P4/P5/P14。

```text
P0  v9.5.2 boundary reproduction
P1  GeometryCard schema and metric materialization
P2  target-vs-geometry audit
P3  signal-channel / SNR gate audit
P4  cover / curvature / plasticity audit
P5  memory / anti-forgetting audit
P6  canonical GoodGeomAction label construction
P7  APY failure geometric autopsy
P8  APG1-APG8 geometry-aware generator implementation
P9  APG deterministic preflight and branch-horizon smoke
P10 APG outcome + GeometryCard evaluation
P11 geometry certificate v1
P12 minimal geometry controller
P13 selected runtime
P14 Base-Acc Sentinel continuation
P15 conditional leave-out / paired replay
P16 short/full/sample-efficiency/continual boundary
```

---

# 7. P0：v9.5.2 boundary reproduction

## 目标

确认本轮没有跳过 v9.5.2 的失败边界，也没有重新使用旧表或旧 target。

## 必须记录

```text
route_v9520
canonical_full_control_outcome_ready
target_lattice_candidate_count
official_target_candidate_count
weak_target_candidate_count
selected_target_id
selected_target_action_count
selected_target_coverage
selected_target_V_integrated_lcb
selected_target_h240_longrisk
selected_target_bad_event
selected_target_null_event
best_apy_primitive
best_apy_target_precision
best_apy_V_integrated_lcb
best_apy_h240_longrisk
best_certificate_id
best_certificate_AUC_target
best_certificate_TopK64_precision
best_certificate_TopK64_longrisk
system_legal_controller_pass
```

## 判断标准

P0 pass：

```text
v9.5.2 route reproduced；
canonical truth base ready；
old table not used for official；
APY implementation pass but APY smoke weak pass = 0。
```

P0 fail：

```text
任何 row/hash/label 与 v9.5.2 不一致且无法解释。
```

## 可视化

```text
p0_v9520_route_reproduction_dashboard.svg
p0_target_lattice_boundary.svg
p0_apy_cert_boundary.svg
```

---

# 8. P1：GeometryCard schema and metric materialization

## 目标

为每个 canonical AP0 action 和每个 generated APY/APG action 生成同一张 GeometryCard。P1 不判断 pass/fail，只负责把指标真实落盘。

## 数据输入

```text
canonical_full_control_outcome_table_v9480_or_later
v9520 target table
APY generated action payloads
APY branch-horizon outcomes
current train-stream batch / per-example gradients
memory probe set
hard-tail probe set
basis activation trace
edge-basis gradient trace
```

## 必须记录

基础字段：

```text
action_id
source_action_id
primitive_id
dataset
seed
step
family_id
stratum_id
payload_hash
state_before_hash
branch_config_hash
horizon_config_hash
```

结果字段：

```text
V20_ctrl
V80_ctrl
V240_ctrl
V_integrated
V20_lcb
V80_lcb
V240_lcb
V_integrated_lcb
bad_event_rate
null_event_rate
h240_longrisk
CEp99_delta_20/80/240
MarginP10_delta_20/80/240
NLL_delta_20/80/240
ECE_delta_20/80/240
```

几何字段：

```text
snr_action_weighted
snr_edge_mean
snr_edge_p10
snr_edge_p90
action_adamw_cosine_high_snr
action_adamw_cosine_low_snr
cover_entropy_before
cover_entropy_after
cover_entropy_delta
edge_basis_utilization_before
edge_basis_utilization_after
edge_basis_utilization_delta
curv_proxy_before
curv_proxy_after
curv_proxy_delta
jacobian_spectral_proxy_before
jacobian_spectral_proxy_after
local_contraction_proxy
plasticity_score
rigidity_score
hard_tail_fraction_before
hard_tail_fraction_after
hard_tail_fraction_delta
memory_CEp99_delta
memory_margin_p10_delta
memory_forget_risk
old_stratum_drift
```

成本字段：

```text
feature_compute_ms
certificate_compute_ms
payload_apply_ms
extra_kernel_count
extra_sync_count
memory_delta_bytes
```

## 判断标准

P1 pass：

```text
geometry_card_rows == action rows；
missing required fields = 0；
NaN/Inf count = 0；
feature cost recorded；
no fake/proxy/cpu offload；
commit-time fields separated from outcome fields。
```

P1 fail：

```text
any geometry feature uses future outcome at commit；
missing essential hash；
NaN/Inf in required metrics；
feature rows not joinable to actions。
```

## 可视化

```text
p1_geometry_card_missingness_heatmap.svg
p1_geometry_metric_correlation_matrix.svg
p1_geometry_metric_cost_bar.svg
p1_geometry_card_distribution_by_dataset.svg
p1_geometry_card_distribution_by_family.svg
```

---

# 9. P2：target-vs-geometry audit

## 目标

判断 v9.5.2 的 target failure 是因为 target 太严格、动作太少，还是因为当前 target 根本没有描述好几何。

## 实验设置

把 canonical AP0 actions 分成：

```text
T9520_best_target
T_A legacy YRobust
T_B StableHorizon
T_C StrictAllH
T_D IntegratedRobustScoreTop
GoodGeomAction preliminary
NonTargetControl
LongRiskControl
NullControl
```

比较每组 GeometryCard。

## 必须记录

```text
group_id
action_count
coverage
V_integrated_lcb
h20/h80/h240 V_lcb
h240_longrisk
bad_event
null_event
snr_action_weighted_mean
cover_entropy_delta_mean
curv_proxy_delta_mean
memory_forget_risk_mean
hard_tail_fraction_delta_mean
family_count
stratum_count
dataset_count
```

## 判断标准

P2 pass：

```text
至少一组 target 的 GeometryCard 有清晰正向模式：
  snr_action_weighted 高；
  cover entropy 不下降；
  curv proxy 不暴涨；
  forget risk 低；
  hard tail 改善；
  V_integrated LCB > 0。
```

P2 fail：

```text
所有 target 组与 non-target 组的 geometry distribution 基本重叠；
或者 target 组在几何上反而更差。
```

## 可视化

```text
p2_target_vs_geometry_radar.svg
p2_target_geometry_boxplot.svg
p2_target_density_vs_quality.svg
p2_v9520_target_bad_null_decomposition.svg
p2_target_jaccard_geometry_overlay.svg
```

---

# 10. P3：signal-channel / SNR gate audit

## 目标

用 per-example gradient 估计动作是否落在多数样本一致支持的方向，避免把噪声当成 signal。

## 实验设置

对每个 action 相关参数块计算：

$$
\mu_k=\frac{1}{b}\sum_{i=1}^{b}g_{i,k},
$$

$$
\sigma_k^2=\frac{1}{b}\sum_{i=1}^{b}(g_{i,k}-\mu_k)^2,
$$

$$
SNR_k=\frac{\mu_k^2}{\sigma_k^2/(b-1)+\epsilon}.
$$

对 action 加权：

$$
SNR(a)=\frac{\sum_k |\Delta\theta_{a,k}|SNR_k}{\sum_k |\Delta\theta_{a,k}|+\epsilon}.
$$

同时计算 off-diagonal batch agreement：

$$
\Omega_B(a)=\frac{1}{b(b-1)}\sum_{i\ne j} g_i^T M_a g_j.
$$

其中 $M_a$ 是 action payload 对应的参数掩码或低秩方向。

## 必须记录

```text
snr_action_weighted
snr_action_pass_fraction
omega_batch_action
mean_grad_alignment
variance_penalty
snr_high_payload_fraction
snr_low_payload_fraction
h20/h80/h240 V_lcb
h240_longrisk
forget_risk
```

## 判断标准

P3 pass：

```text
SNR(a) 对 GoodGeomAction AUC >= 0.70；
TopK64 by SNR target precision >= 0.20；
TopK64 h240 longrisk <= 0.20；
SNR-filtered APY/APG actions 的 longrisk 显著低于 unfiltered。
```

P3 fail：

```text
SNR 对 target / longrisk / forgetting 均无区分力。
```

## 可视化

```text
p3_snr_vs_Vintegrated.svg
p3_snr_vs_longrisk.svg
p3_snr_topk_frontier.svg
p3_signal_noise_quadrant.svg
p3_omega_batch_distribution.svg
```

---

# 11. P4：cover / curvature / plasticity audit

## 目标

判断动作是否让 KAN 的 edge basis 覆盖变窄、局部曲率变坏、表示变硬。

## 实验设置

在 fixed probe batch 上，记录 action 前后：

```text
basis activation distribution；
edge gradient distribution；
local output Jacobian probes；
curvature probes；
hard-tail samples；
```

计算：

$$
CoverEntropy=-\sum_i p_i\log(p_i+\epsilon),
$$

$$
BasisCollapse=\max_i p_i,
$$

$$
CurvDelta=CurvProxy_{after}-CurvProxy_{before}.
$$

## 必须记录

```text
cover_entropy_before/after/delta
basis_collapse_before/after/delta
edge_basis_utilization_delta
curv_proxy_delta
jacobian_spectral_proxy_delta
local_contraction_proxy
hard_tail_fraction_delta
```

## 判断标准

P4 pass：

```text
GoodGeomAction 组满足：
  cover_entropy_delta >= -small_eps；
  basis_collapse_delta <= threshold；
  curv_proxy_delta <= threshold；
  hard_tail_fraction_delta <= 0。
```

P4 fail：

```text
所有 value-positive actions 都伴随 cover collapse 或 curvature spike；
说明当前 primitive 以牺牲几何稳定换短期 value。
```

## 可视化

```text
p4_cover_entropy_before_after.svg
p4_basis_collapse_by_primitive.svg
p4_curv_delta_vs_longrisk.svg
p4_hardtail_delta_vs_V.svg
p4_geometry_stability_phase_plot.svg
```

---

# 12. P5：memory / anti-forgetting audit

## 目标

判断动作是否会伤害旧 stratum、旧 family 或前面已经稳定的样本。这个阶段必须和 P3/P4 并行运行。

## 数据设置

每个 dataset/seed 构造 memory probe：

```text
old easy samples；
old hard-tail samples；
old stable-correct samples；
old borderline-margin samples；
old family-balanced samples；
```

不能用 test labels 做 controller；memory probe 只来自 training stream 或 calibration-safe pool。

## 必须记录

```text
memory_CEp99_delta
memory_NLL_delta
memory_ECE_delta
memory_margin_p10_delta
memory_accuracy_delta_diagnostic
old_stratum_drift
old_family_fail_count
forget_risk_score
new_task_gain_score
plasticity_stability_ratio
```

定义：

$$
PSR(a)=\frac{Gain_{new}(a)}{ForgetRisk_{old}(a)+\epsilon}.
$$

## 判断标准

P5 pass：

```text
GoodGeomAction 组 forget_risk UCB <= threshold；
PSR TopK64 中 longrisk <= 0.15；
memory_margin_p10_delta 不显著为负；
old hard-tail CEp99 不显著上升。
```

P5 fail：

```text
所有 value-positive actions 都造成 old hard-tail 或 old family 明显退化。
```

## 可视化

```text
p5_forget_risk_distribution.svg
p5_plasticity_stability_ratio.svg
p5_old_vs_new_tradeoff.svg
p5_memory_margin_delta_heatmap.svg
p5_family_forgetting_matrix.svg
```

---

# 13. P6：canonical GoodGeomAction label construction

## 目标

把 P1-P5 的结果合成一个 official candidate target。它不是一个随便调的二元标签，而是由固定规则组成。

## Label 定义

初版 GoodGeomAction：

$$
GGA(a)=1
$$

当且仅当：

$$
ApplyOK(a)=1,
$$

$$
LCB(V_{20})>0,
$$

$$
LCB(V_{80})\ge -\epsilon_{80},
$$

$$
UCB(LongRisk_{240})\le \tau_L,
$$

$$
UCB(Bad)\le \tau_B,
$$

$$
UCB(Null)\le \tau_N,
$$

$$
SNR(a)\ge \tau_{snr},
$$

$$
CoverEntropyDelta(a)\ge -\tau_{cover},
$$

$$
CurvDelta(a)\le \tau_{curv},
$$

$$
ForgetRisk(a)\le \tau_F.
$$

## 必须记录

```text
GGA_count
GGA_coverage
GGA_coverage_lcb
GGA_bad_event_rate
GGA_null_event_rate
GGA_h240_longrisk
GGA_V_integrated_lcb
GGA_family_count
GGA_stratum_count
GGA_dataset_count
GGA_support_balance_pass
comparison_to_T9520
comparison_to_TA_TB_TC_TD
```

## 判断标准

P6 weak pass：

```text
GGA_count >= 64；
GGA_bad_event <= 0.05；
GGA_null_event <= 0.15；
GGA_h240_longrisk <= 0.10；
GGA_V_integrated_lcb > 0；
support_balance_pass = 1。
```

P6 official pass：

```text
GGA_count >= 87；
GGA_coverage >= 0.03；
GGA_bad_event <= 0.05；
GGA_null_event <= 0.15；
GGA_h240_longrisk <= 0.05；
GGA_V_integrated_lcb > 0；
LDO/LSO sanity not collapsed。
```

P6 fail：

```text
No GGA weak pass；
或者 GGA 过窄且 leaveout 不稳定。
```

## 可视化

```text
p6_gga_density_quality_frontier.svg
p6_gga_vs_old_targets_venn.svg
p6_gga_metric_dashboard.svg
p6_gga_support_balance.svg
p6_gga_leaveout_sanity.svg
```

---

# 14. P7：APY failure geometric autopsy

## 目标

解释 APY 为什么失败。不是简单写 APY3 failed，而是判断它失败在 SNR、cover collapse、curvature、forgetting，还是 objective mismatch。

## 必须记录

```text
primitive_id
APY_target_precision
APY_V_integrated_lcb
APY_h240_longrisk
APY_snr_action_weighted
APY_cover_entropy_delta
APY_curv_proxy_delta
APY_forget_risk
APY_hard_tail_delta
APY_damage_integrated_lcb
failure_reason_primary
failure_reason_secondary
```

## 判断标准

P7 pass：

```text
>=90% failed APY actions assigned a concrete failure reason；
每个 APY primitive 有 dominant failure mode；
failure mode 可以指导 APG design。
```

P7 fail：

```text
APY failed but GeometryCard 无法解释。
```

## 可视化

```text
p7_apy_failure_reason_stacked_bar.svg
p7_apy_geometry_damage_matrix.svg
p7_apy_target_vs_geometry.svg
p7_apy_longrisk_attribution.svg
```

---

# 15. P8：APG1-APG8 geometry-aware generator implementation

## 目标

实现新的 APG primitive family。APG 不再只生成“看起来安全”的 payload，而是生成时就检查 SNR、cover、curvature、memory。

## APG primitive family

```text
APG1-SNRProjectedEdgeUpdate
  只保留高 SNR edge/block 上的 functional delta。

APG2-CoverBalancedBasisUpdate
  约束 action 后 basis cover entropy 不下降。

APG3-CurvatureBarrierTrustRegion
  在 low-rank JVP probe 上限制 curvature spike。

APG4-MemoryPreservingResidualUpdate
  显式限制 memory probe CEp99 / margin 退化。

APG5-HardTailRepairNoForgetUpdate
  只修 hard-tail，但要求 old hard-tail 不恶化。

APG6-SignalChannelLowRankUpdate
  用 per-example gradient agreement 构造低秩 update。

APG7-EdgeLocalityWithSNRGate
  保留 KAN edge locality，但加 SNR gate。

APG8-CompositeGeometryGuardedUpdate
  APG1-APG7 的交集/加权版本，作为强保守候选。
```

## 必须记录

```text
primitive_id
generated_action_count
payload_hash_missing_count
certificate_hash_missing_count
action_apply_error_linf_max
commit_time_geometry_fields_present
uses_dataset_name
uses_outcome_at_commit
uses_future_step
feature_compute_ms
payload_apply_ms
```

## 判断标准

P8 pass：

```text
primitive_count = 8；
generated_action_count >= 512；
payload/certificate hash missing = 0；
action apply L∞ max <= 1e-7；
commit-time geometry fields present；
no outcome at commit；
no dataset name；
no fake/proxy/cpu offload。
```

P8 fail：

```text
任何 primitive 没有真实 payload；
或 action apply 不闭合；
或使用 outcome/future/test information。
```

## 可视化

```text
p8_apg_generation_counts.svg
p8_apg_apply_error.svg
p8_apg_cost_breakdown.svg
p8_apg_commit_time_contract_dashboard.svg
```

---

# 16. P9：APG deterministic preflight and branch-horizon smoke

## 目标

避免像以前一样跑完大实验才发现 replay 或 materializer 问题。P9 必须分层：1-action、3-action、16-action、full-smoke。

## Preflight ladder

```text
Stage A: one action, RealFunctional, horizon 1/2/5/20
Stage B: three actions, all branches, horizon 20
Stage C: sixteen actions, all branches, horizon 20/80/240
Stage D: all APG generated actions, all branches, horizon 20/80/240
```

## 必须记录

```text
stage_id
action_count
branch_count
horizon_count
expected_rows
actual_rows
row_completion_rate
metric_abs_diff_no_transform
label_match_no_transform
negative_control_divergence
unresolved_exception_count
rows_per_sec
wallclock_sec
```

## 判断标准

P9 pass：

```text
all stages complete；
no-transform equivalence = 1；
negative controls diverge；
branch/horizon completion = 1；
unresolved exception = 0。
```

P9 fail：

```text
any replay/materializer inconsistency；
any missing branch/horizon rows；
negative controls fail to diverge。
```

## 可视化

```text
p9_preflight_ladder_completion.svg
p9_branch_horizon_completion_heatmap.svg
p9_negative_control_divergence.svg
p9_rows_per_sec_by_stage.svg
```

---

# 17. P10：APG outcome + GeometryCard evaluation

## 目标

判断 APG 是否真的产生了好动作，并把结果和 APY / APX / AP0 对比。

## 必须记录

Per primitive：

```text
primitive_id
action_count
target_precision_T9520
GGA_precision
V20_lcb
V80_lcb
V240_lcb
V_integrated_lcb
bad_event
null_event
h240_longrisk
forget_risk
snr_action_weighted
cover_entropy_delta
curv_proxy_delta
support_balance
```

## 判断标准

P10 weak pass：

```text
best APG GGA precision >= 0.20；
V_integrated_lcb > 0；
h240_longrisk <= 0.15；
forget_risk <= threshold；
bad_event <= 0.10；
null_event <= 0.20。
```

P10 official candidate pass：

```text
accepted_count >= 87 或 coverage >= 0.03；
GGA precision >= 0.50；
V_integrated_lcb > 0；
h240_longrisk <= 0.05；
bad_event <= 0.05；
null_event <= 0.15；
support_balance_pass = 1。
```

P10 fail：

```text
APG implementation pass but no primitive improves over APY/APX/AP0 baseline。
```

## 可视化

```text
p10_apg_vs_apy_quality_frontier.svg
p10_apg_primitive_radar.svg
p10_apg_longrisk_vs_value.svg
p10_apg_forgetting_vs_gain.svg
p10_apg_geometry_outcome_phase_plot.svg
```

---

# 18. P11：geometry certificate v1

## 目标

训练或构造一个小而可解释的证书，用训练当下可见的 GeometryCard commit-time fields 预测 GGA。

## 候选 certificate

```text
GCERT1-SNRThresholdCertificate
GCERT2-CoverEntropyCertificate
GCERT3-CurvatureBarrierCertificate
GCERT4-MemoryNoForgetCertificate
GCERT5-SignalCoverCompositeCertificate
GCERT6-SmallMonotoneLinearCertificate
GCERT7-ThreeRuleCertificate
GCERT8-MinimalControllerCertificate
```

## 约束

```text
feature groups <= 5；
monotone sign required；
thresholds frozen on calibration；
no dataset name；
no outcome at commit；
no validation/test；
feature cost measured。
```

## 必须记录

```text
certificate_id
feature_group_count
monotone_sign_pass
AUC_GGA
AUC_LongRisk
AUC_ForgetRisk
TopK16_GGA_precision
TopK64_GGA_precision
TopK64_longrisk
TopK64_forgetrisk
ECE_GGA
feature_compute_ms_q90
certificate_compute_ms_q90
LDO_AUC_drop_max
LSO_AUC_drop_max
```

## 判断标准

P11 weak pass：

```text
AUC_GGA >= 0.70；
TopK64_GGA_precision >= 0.20；
TopK64_longrisk <= 0.15；
ECE_GGA <= 0.15；
feature cost q90 <= 0.50 ms。
```

P11 official pass：

```text
TopK64_GGA_precision >= 0.25；
TopK64_longrisk <= 0.10；
TopK64_forgetrisk <= threshold；
ECE_GGA <= 0.10；
monotone_sign_pass = 1；
LDO/LSO AUC drop <= 0.10。
```

P11 fail：

```text
AUC-only signal but TopK collapse；
or high TopK longrisk；
or certificate cost too high；
or LDO/LSO collapse。
```

## 可视化

```text
p11_certificate_roc_pr.svg
p11_certificate_topk_precision.svg
p11_certificate_longrisk_topk.svg
p11_certificate_calibration_curve.svg
p11_certificate_feature_ablation.svg
p11_certificate_leaveout_matrix.svg
```

---

# 19. P12：minimal geometry controller

## 目标

如果 P10/P11 有 survivor，则构造最小 controller：只接受 GCERT 通过且成本可控的动作。

## Controller form

$$
Accept(a)=1
$$

当且仅当：

$$
GCERT(a)=1,
$$

$$
Cost(a)\le C_{max},
$$

$$
Support(a)\ge S_{min}.
$$

不允许额外复杂黑盒 threshold search。

## 必须记录

```text
controller_id
certificate_id
primitive_id
accepted_count
coverage
precision_GGA
bad_event
null_event
h240_longrisk
forget_risk
V_integrated_lcb
support_balance
family_count
stratum_count
dataset_count
max_family_share
max_stratum_share
feature_cost_q90
payload_apply_q90
```

## 判断标准

P12 weak pass：

```text
accepted_count >= 64；
coverage >= 0.02；
precision_GGA >= 0.25；
h240_longrisk <= 0.15；
V_integrated_lcb > 0；
forget_risk <= threshold。
```

P12 official pass：

```text
accepted_count >= 87；
coverage >= 0.03；
precision_GGA >= 0.50；
bad_event <= 0.05；
null_event <= 0.15；
h240_longrisk <= 0.05；
V_integrated_lcb > 0；
support_balance_pass = 1。
```

P12 fail：

```text
controller accepted count too low；
或 TopK risk high；
或 support collapse。
```

## 可视化

```text
p12_controller_quality_frontier.svg
p12_controller_acceptance_by_family.svg
p12_controller_acceptance_by_dataset.svg
p12_controller_support_balance.svg
p12_controller_failure_modes.svg
```

---

# 20. P13：selected runtime

## 目标

测 selected controller 真实在线成本。不能用 no-payload smoke，也不能用 offline materializer cost。

## 必须记录

```text
runtime_candidate_id
controller_id
primitive_id
feature_compute_ms_q50/q90/q99
certificate_compute_ms_q50/q90/q99
payload_apply_ms_q50/q90/q99
base_step_ms_q50/q90/q99
total_step_ms_q50/q90/q99
step_ratio_q50/q90/q99
memory_ratio
extra_kernel_count
extra_sync_count
zero_candidate_kernel_count
active_step_launches_q90
materializer_in_timed_path
old_step_ratio_reused
```

## 判断标准

P13 pass：

```text
materializer_in_timed_path = 0；
old_step_ratio_reused = 0；
zero_candidate_kernel_count = 0；
step_ratio_q90 <= 1.50；
memory_ratio <= 1.05；
payload_apply error L∞ <= 1e-7。
```

P13 fail：

```text
controller quality pass but runtime > 1.50；
then route = RuntimePayloadOrFeatureTooSlow。
```

## 可视化

```text
p13_runtime_waterfall.svg
p13_step_ratio_distribution.svg
p13_feature_payload_cost_breakdown.svg
p13_kernel_sync_count.svg
p13_runtime_vs_accept_count.svg
```

---

# 21. P14：Base-Acc Sentinel continuation

## 目标

继续确认 LQ-t2-h256 base 没有 catastrophic fail，但不得用于 controller 或 generator。

## 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0..9
models = LQ-t2-h256, MatchedMLP, AdamWStrongLRGridMLP, QuadraticFeatureMLP
same seed schedule
same budget
no dataset-specific tuning
```

## 必须记录

```text
train_acc
val_acc
test_acc
CE
NLL
ECE
CEp99
margin_p10
steps_to_target
wallclock
params
step_time
```

## 判断标准

P14 pass：

```text
LQ catastrophic fail = 0；
base_acc_used_for_controller = 0；
all rows complete。
```

P14 fail：

```text
LQ base collapses on any dataset/seed beyond predefined threshold。
```

## 可视化

```text
p14_base_acc_by_dataset_seed.svg
p14_lq_vs_mlp_acc.svg
p14_ce_nll_ece_dashboard.svg
p14_base_health_trace.svg
```

---

# 22. P15：conditional leave-out / paired replay

## 目标

只有 P12/P13 pass 后打开。验证 controller 不是只在 pooled data 上好，也不是只对某个 dataset 好。

## Leave-dataset-out

```text
calibrate on two datasets, evaluate the third；
rotate all three heldout datasets。
```

## Leave-stratum-out

```text
calibrate on all but one family/stratum；
evaluate heldout family/stratum。
```

## Paired replay branches

```text
RealFunctionalSelected
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
CertificatePassNoPayload
```

## 必须记录

```text
heldout_type
heldout_id
accepted_count
coverage
precision_GGA
bad_event
null_event
longrisk
forget_risk
V20/V80/V240
CEp99_delta
margin_delta
NLL_delta
ECE_delta
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
shuffle_control_fail_expected
```

## 判断标准

P15 pass：

```text
LDO: at least 2/3 heldout datasets pass weak gate；
LSO: at least 70% heldout strata pass safety gate；
paired replay: RealFunctional beats AdamWParallel and bestLR in >=50% of accepted cells；
shuffled payload does not pass。
```

P15 fail：

```text
pooled pass but leaveout collapses；
RealFunctional does not beat controls；
shuffled payload passes。
```

## 可视化

```text
p15_leave_dataset_out_matrix.svg
p15_leave_stratum_out_matrix.svg
p15_paired_replay_winrate.svg
p15_shuffle_control_dashboard.svg
p15_real_vs_controls_horizon_plot.svg
```

---

# 23. P16：short/full/sample-efficiency/continual boundary

## 目标

只有 P15 pass 后打开。验证 functional update 是否进入真正训练收益。

## 设置

```text
short run: 20/80/240/1000 steps
full run: fixed budget full training
sample efficiency: train fraction 5%, 10%, 25%, 50%, 100%
noise robustness: label noise / input noise diagnostic
continual: task order MNIST -> Fashion -> KMNIST and rotations
```

## 必须记录

```text
final_test_acc
best_val_acc
time_to_target_acc
steps_to_target_acc
train_val_gap
NLL
ECE
CEp99
margin_p10
hard_stratum_acc
noise_robust_acc
retained_acc_old_task
forgetting_score
runtime_step_ratio
memory_ratio
```

## 判断标准

P16 weak pass：

```text
functional update improves at least one of:
  steps_to_target;
  CEp99;
  ECE;
  sample efficiency;
  forgetting;
without hurting test acc by > 0.005.
```

P16 strong pass：

```text
LQ+Functional beats LQ AdamW-only and AdamWStrongLRGridMLP on at least two datasets or two robustness axes；
paired replay causal pass remains valid；
runtime still <= 1.50；
no dataset-specific rule used。
```

## 可视化

```text
p16_short_run_learning_curves.svg
p16_full_run_acc_loss_ece.svg
p16_sample_efficiency_curve.svg
p16_noise_robustness_curve.svg
p16_continual_forgetting_matrix.svg
p16_runtime_quality_pareto.svg
```

---

# 24. 并行执行安排

为了避免继续“一轮一个 blocker”，v9.5.3 必须并行执行：

```text
Batch A: P0 + P1 GeometryCard materialization
Batch B: P2 target-vs-geometry + P3 SNR audit
Batch C: P4 cover/curvature + P5 memory audit
Batch D: P7 APY failure autopsy
Batch E: P8 APG implementation preflight
Batch F: P14 Base-Acc Sentinel
```

只要 P1 schema pass，就可以并行启动 P2/P3/P4/P5/P7。P8 可以先用 APY failure modes 设计 APG，但 APG official evaluation 必须等 P1/P6 label ready。

---

# 25. Route decision

v9.5.3 route 按下面顺序判定：

```text
R0-ReproductionFail
  P0 failed。

R1-GeometryCardMaterializationFail
  P1 failed。

R2-NoGoodGeometryActionInCanonicalAP0
  P6 no weak GGA target in AP0。

R3-GoodGeometryExistsButLegalInvisible
  P6 pass, P11 fail。

R4-APYFailureExplainedPrimitiveResetRequired
  P7 explains APY failure, APG not yet pass。

R5-APGImplementationFail
  P8/P9 fail。

R6-APGGeneratedGeometryValueFail
  P10 fail。

R7-CertificateFail
  P11 fail after APG survivor exists。

R8-ControllerSupportCollapse
  P12 fail。

R9-RuntimeFail
  P12 pass, P13 fail。

R10-LeaveoutOrPairedReplayFail
  P13 pass, P15 fail。

R11-ShortFullFail
  P15 pass, P16 fail。

R12-v9530LocalFunctionalSuccess
  P12/P13/P15 weak pass。

R13-v9530StrongFunctionalCandidate
  P16 strong pass。
```

---

# 26. 本轮最重要的停止条件

必须停止的情况：

```text
1. 如果 GoodGeomAction 在 AP0 full universe 中 coverage 仍 < 0.01，且 APG 也不能提高 coverage，停止在 action space insufficient。
2. 如果 SNR/cover/curvature/memory 指标都不能解释 target / longrisk / forgetting，停止手工 geometry features，转向更直接的 train-test transfer probe。
3. 如果 APG 生成的动作仍普遍 long-risk > 0.50，停止 primitive 小变体，重写生成目标。
4. 如果 certificate AUC 高但 TopK precision 低，停止 AUC-driven certificate，改成 topK/risk-calibrated certificate。
5. 如果 runtime > 1.50，停止 downstream，先修 selected runtime。
```

---

# 27. 最终判断标准

v9.5.3 不是必须 full success。但它必须回答下面四个问题：

```text
1. “好几何”是否能被实际测量？
2. 当前 AP0/APY action failure 是否可以用几何指标解释？
3. APG 是否能生成比 APY 更好的动作？
4. geometry certificate 是否能在训练当下选中动作？
```

如果答案是 yes，项目进入：

```text
v9.5.4 selected geometry controller + paired replay + runtime closure。
```

如果答案是 no，项目进入：

```text
v9.5.4 train-test transfer probe / population-risk update rule reset。
```

---

# 28. 预期结论模板

最终复盘必须用下面格式写：

```text
v9.5.3 route = ...

是否有进展：
  是 / 否；具体是哪种进展。

是否能力进展：
  是 / 否。

是否仍在正确道路上：
  是 / 否；理由。

当前 primary blocker：
  ...

最重要发现：
  1. ...
  2. ...
  3. ...

是否有 dataset-specific tuning：
  0 / 1。

Base-Acc Sentinel：
  LQ / MLP / StrongLRGridMLP / QuadraticFeatureMLP。

下一步：
  ...
```

---

# 29. 本轮核心提醒

不要再把 “好动作” 简化成 acc 或一个二元 target。

一个真正好的 functional update 动作应该同时满足：

```text
它能短期帮忙；
它不会中期变坏；
它长期不制造风险；
它不只是记噪声；
它不破坏 basis 覆盖；
它不让曲率暴涨；
它不忘旧知识；
它能被训练当下的证据识别；
它运行得足够快；
它能打过 AdamW / bestLR / NoOp / Random。
```

这才和 DG-KAN 的终极目标一致。
