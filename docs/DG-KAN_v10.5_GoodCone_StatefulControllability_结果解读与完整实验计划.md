# DG-KAN v10.4 结果解读与 v10.5 GoodCone / Stateful Controllability 完整实验计划

> 本文件基于 v10.4 `FPAU Subspace Controllability` 的真实结果制定。v10.5 不继续小修 `FPAU-A7`、`FPAU14B`、`FPO13D`、`GoodSubspace rank`、`AUV/RAUV threshold` 或 generated family 名字。  
> v10.5 的核心目标是把问题从“好动作是否落在某个线性 GoodSubspace”升级为：**好 functional update 是否位于一个由模型状态、优化器状态、未来路径类型和安全约束共同定义的 state-conditioned good cone / controllable region**。

---

# 0. v10.4 的一句话判断

v10.4 不是能力成功。它的真实价值是把 blocker 从：

```text
FPAU toy 可解，但真实 generated subspace 缺 value direction
```

推进到更具体的判断：

```text
已知 Fast/Slow 动作不能被当前 GoodSubspace sketch 稳定分离；
当前 generated 更新虽然高投影到 GoodSubspace，却仍然生成 BadPath；
因此“投到好子空间”不是充分条件。
```

用公式说：

$$
\boxed{
\Delta \in \mathcal{S}_{good}
\not\Rightarrow
\text{FuturePathGood}(\Delta)
}
$$

更准确的假设应该变成：

$$
\boxed{
\text{GoodUpdate}(\Delta)
=
\left[\Delta \in \mathcal{C}_{good}(s_t)\right]
\land
\text{RiskSafe}(\Delta)
\land
\text{MemorySafe}(\Delta)
}
$$

其中 $s_t$ 是当前训练状态，包括模型参数、AdamW 动量、方差状态、batch / memory / hard-tail 响应、old-family 状态、offdiag 风险状态等。$\mathcal{C}_{good}(s_t)$ 不是固定线性子空间，而是状态条件化的 cone / curved region。

---

# 1. v10.4 的独立复盘

## 1.1 有进展，但不是能力进展

v10.4 的 route 是：

```text
route = CaseB-GoodSubspaceAbsentOrTooDiffuse
primary_blocker = good_value_subspace_not_stable
secondary_blocker = current_action_representation_insufficient
system_legal_controller_pass = 0
generated_route_status = stopped_good_subspace_absent_or_diffuse
```

关键数据：

```text
P1 known-action rows = 15080
FastGood / SlowBurnGood / RiskyHighAUV / SafeLowValue / BadPath = 4 / 22 / 321 / 26 / 14707
matched slowburn pairs = 44
best matched AUC = 0.8920118343195266

P2 GoodSubspace rank = 16
explained variance = 0.9998584670459338
good coverage = 1.0
good-bad separation = 0.021201327577448192
generated projection = 0.988584558169047
P2 pass = 0

P3 future adjoint signal count = 0
max partial corr V240 | V1 = 0.01635610475924454
max CCI = 0.13271891503173477

P4 joint retention = 0.1986895203590393
P4 pass = 1
repair = soft_trust_region_alpha50_projection_attempted

P5 toy beats current-gradient = 7 / 7
slow / memory / offdiag toy pass = 1 / 1 / 1

P6 generated Stage4/8/16/64 opened = 6 / 0 / 0 / 0
generated discovery pass = 0

P7 TopK64 precision = 0.03125
SlowBurn recall = 0.09090909090909091
risky rate = 0.0
V240 = -0.9347178489800314
P7 discovery pass = 0
```

这组数据的关键不是“又失败了”，而是揭示了三个结构性事实。

### 事实 1：GoodSubspace 有覆盖，但没有区分力

P2 显示 GoodSubspace rank 只有 16，explained variance 接近 1，good coverage 是 1。这看起来像好消息。可是 good-bad separation 只有 0.0212，而且 generated projection 高达 0.9886 仍然生成失败。

这说明：

```text
好动作和坏动作都落在这个 rank-16 区域附近。
```

所以问题不是 generated update 没进 GoodSubspace，而是 GoodSubspace 本身不是好动作判据。

数学上，当前假设：

$$
\Delta \in \mathcal{S}_{good}
$$

太弱。需要判断的是：

$$
\Delta \in \mathcal{C}_{good}(s_t)
$$

也就是在同一个子空间里，还要判断方向、符号、幅度、状态相位、优化器状态、memory/offdiag 约束是否匹配。

### 事实 2：FPAU toy 可解，但真实系统中的 adjoint 没有 future signal

P5 toy 全过，说明“future-path objective”本身不是空想。但 P3 的 future adjoint signal count 是 0，max partial corr $V240|V1$ 只有 0.016。这说明真实 FPAU 仍没有捕捉到：

$$
D U_h(\theta_t)^\top \nabla_{\theta_{t+h}}M_h
$$

这种未来路径传播信息。它更像是在当前响应、风险响应、局部 surrogate 上转圈。

### 事实 3：memory/offdiag 约束不是无用，但硬投影会毁掉 value

P4 joint retention 只有 0.1987。这说明 memory/offdiag 约束不是完全不可行，但它会保留很少的 Fast/Slow value direction。硬约束可能把未来价值方向投掉；soft trust region 是合理方向，但还没有变成生成规则。

---

# 2. 四线当前状态

## A 线：Future Path Type

A 线仍然是最有理论价值的一条线。v10.4 进一步确认 Fast/Slow 极稀疏：FastGood 4，SlowBurnGood 22，BadPath 14707。好动作不是没有，但在当前 known-action representation 里是非常薄的集合。

A 线当前问题不是“未来路径不成立”，而是：

```text
未来路径类型还没有被压缩成训练当下可用的机制。
```

v10.5 的 A 线必须固定 path type，不再用单一 AUV / RAUV：

```text
FastGood：h1 好，h20/h80/h240 也好，risk 低。
SlowBurnGood：h1 不强甚至负，但 h20/h80/h240 变好，risk 低。
RiskyHighAUV：AUV 高，但 longrisk / memory / offdiag 高。
SafeLowValue：risk 低，但 value 不够。
BadPath：收益差或风险高。
```

核心目标是 SlowBurnGood，因为它最符合：

```text
不是当前一步降 loss，而是把后续训练带到更好轨迹。
```

## B 线：FPAU / FPO 训练当下识别

B 线当前失败。FPAU-A 系列、FPO13、FPO14 path-type rebuild 都没有产生可用 TopK。v10.4 P7 的 TopK64 precision 只有 0.03125，V240 是 -0.9347。

B 线的核心问题是：

```text
它还在用当前动作表示、当前响应或低阶 sketch 试图预测未来路径。
```

下一步必须改变目标：从 scalar score 变成 state-conditioned path-type predictor + controllability test。

## C 线：Natural harvesting

C 线已经降级为 reference / sanity line。v9.9.8 到 v10.1 已经显示 natural density insufficient；v10.4 不新增 C-line panel是合理的。C 线下一步只需要低成本 sanity audit，不应该再作为主线投入大量算力。

## D 线：Generated update

D 线仍然失败。v10.4 P6 只打开 Stage4，Stage8/16/64 全不打开；这说明 generated update 不是数量不够，而是当前 family 产生的方向本身错。

D 线下一步不能继续“新增 FPAU15A/B/C”。必须先确认：

```text
1. 当前 update basis 是否覆盖 Fast/Slow value direction？
2. 如果覆盖，为什么 generated coefficient 选不到？
3. 如果不覆盖，需要新增什么 basis？
4. 如果 memory/offdiag 投影毁掉 value，是否需要 soft trust region 而不是 hard projection？
```

---

# 3. v10.5 的核心假设

v10.5 不再问：

```text
GoodSubspace 是否存在？
FPAU score 是否更高？
generated family 是否多跑一点就出好动作？
```

v10.5 问四个更硬的问题。

## H1：好动作不是线性子空间，而是状态条件化 cone

假设：Fast/Slow 不由一个固定 GoodSubspace 决定，而由同一子空间内的方向、符号、幅度、优化器状态相位决定。

$$
\text{FastSlow}(\Delta)=1
\iff
\Delta \in \mathcal{C}_{good}(s_t)
$$

其中：

$$
\mathcal{C}_{good}(s_t)
=
\{\Delta: q_{s_t}(\Delta)>0,\ r_{s_t}(\Delta)\le\tau\}
$$

$q_{s_t}$ 是未来路径价值，$r_{s_t}$ 是风险 / memory / offdiag 约束。

## H2：当前 generated subspace 高投影但低价值，是因为缺少 orientation / cone constraint

P2 generated projection = 0.9886，但 generated 仍失败。假设原因是：

```text
generated update 进入了子空间，但方向/符号/尺度错；
或者它进入的是 good-bad 共用子空间，而非 good cone。
```

## H3：memory/offdiag 不能做硬投影，需要 Pareto / soft trust region

P4 joint retention 只有 0.1987。假设硬 projection 毁掉大量 value direction。需要探索：

$$
\max_{\Delta} V_{future}(\Delta)
-
\lambda_M M(\Delta)
-
\lambda_O O(\Delta)
$$

而不是：

$$
M(\Delta)=0,\quad O(\Delta)=0
$$

## H4：如果 GoodCone / soft constraint 仍失败，说明当前 action representation 不足

若 H1-H3 都失败，则当前 action representation 不能表达 future-path-improving update。此时必须回到更底层：模型结构、basis、optimizer-state propagation、甚至 compositional FullEdge kernel-native route。

---

# 4. v10.5 实验计划

## P0：v10.4 boundary lock

### 目标

复现 v10.4 boundary，确认后续实验不是建立在错误 artifact 上。

### 必须记录

```text
source_route
P1 Fast/Slow/Risky/SafeLow/Bad
P2 GoodSubspace rank / EV / separation / generated_projection
P3 future_adjoint_signal_count
P4 joint_retention
P5 toy_pass
P6 stage opened/pass
P7 precision/V240
fake/proxy/cpu audit
```

### Pass 标准

```text
source_route = CaseB-GoodSubspaceAbsentOrTooDiffuse
system_pass = 0
fake/proxy/cpu = 0/0/0
P0_boundary_pass = 1
```

### 如果不满足，Codex 先尝试

```text
如果 artifact 缺失：只补 P0 manifest / route parsing，不跑 science stages。
如果 counts 不一致：检查 v10.1/v10.3/v10.4 action universe join key。
如果 fake/proxy 非 0：停止全轮，先修 data provenance。
```

---

## P1：State-conditioned path contrast

### 目标

判断 Fast/Slow 是否只在特定训练状态相位下可分。不是再用全局 feature 排序，而是检查：

```text
在相同 dataset/family/template/action_norm/payload_norm 下，
Fast/Slow 与 Bad/Risky 的差异是否由 optimizer state、exact transfer、current response、memory/offdiag state 解释。
```

### 假设

$$
H1:
P(\text{FastSlow}|Z,s_t)
\text{ 在状态条件化后可分，但全局不可分。}
$$

### 必须记录

```text
action_id
path_type
dataset/family/template/step_bucket/action_norm_bucket/payload_norm_bucket
exact_transfer
current_response
optimizer_state_alignment
adamw_cosine
momentum_cosine
variance_scale
memory_score
old_family_margin_delta
hard_tail_response
offdiag_score
V1/V5/V20/V80/V240
RAUV
longrisk
memory_fail
offdiag_fail
matched_pair_id
matching_level
feature_delta
AUC_by_state_bucket
effect_size_by_state_bucket
```

### 判断标准

P1 weak pass：

```text
至少一个 state bucket 内：
  FastSlow_count >= 8
  Bad/Risky matched_count >= 32
  AUC(FastSlow vs Bad/Risky) >= 0.75
  max_dataset_share <= 0.70
  max_template_share <= 0.50
```

P1 strong pass：

```text
至少两个非重叠 state buckets 满足 weak；
并且关键 feature family 一致，例如 optimizer_state_alignment 或 exact_transfer；
FastSlow vs RiskyHighAUV 也能 AUC >= 0.65。
```

### 可视化

```text
p1_path_type_by_state_bucket_heatmap.svg
p1_matched_contrast_effect_forest.svg
p1_fastslow_vs_risky_scatter_exact_transfer_optimizer_alignment.svg
p1_state_bucket_support_table.svg
```

### 如果不满足，Codex 先尝试

```text
如果 FastSlow 太少：合并 FastGood + SlowBurnGood，不合并 Risky。
如果 matched pairs 太少：放宽 action_norm / payload_norm caliper，但不放宽 dataset/family/template matching。
如果 AUC 高但 group share 高：标记为 diagnostic，不进入 generation。
如果 exact_transfer 高但 value 低：检查是否又落入 immediate-response trap。
如果 optimizer_state_alignment 有信号：进入 P2 GoodCone；否则进入 P4 basis coverage audit。
```

---

## P2：GoodCone / Quadratic metric discovery

### 目标

把 P2 GoodSubspace 从“线性子空间”升级为“方向性 cone / quadratic metric”。解释为什么 generated projection 高但仍失败。

### 假设

$$
H2:
\text{FastSlow 与 BadPath 在同一 rank-16 子空间内，但分布在不同 cone / orientation / signed region。}
$$

### 方法

构造 GoodSubspace basis $B_{16}$ 后，不只看投影范数：

$$
z = B_{16}^\top \Delta
$$

而是拟合：

$$
ConeScore(\Delta)=u^\top z - \lambda z^\top Qz
$$

或 Fisher / LDA / quadratic form：

$$
Score(\Delta)=z^\top A z + b^\top z
$$

所有拟合只作为 discovery，不直接 official。

### 必须记录

```text
basis_id
basis_rank
explained_variance
good_coverage
good_bad_projection_separation
good_bad_cone_margin
generated_projection_norm
generated_cone_score
FastSlow precision TopK16/32/64
RiskyHighAUV rate TopK16/32/64
V240 LCB TopK16/32/64
RAUV LCB TopK16/32/64
longrisk UCB TopK16/32/64
memory/offdiag UCB
heldout split precision
```

### 判断标准

P2 weak pass：

```text
TopK32 FastSlow precision >= 0.15
TopK32 V240 LCB > 0
TopK32 longrisk UCB <= 0.10
RiskyHighAUV rate <= 0.20
```

P2 strong pass：

```text
TopK64 FastSlow precision >= 0.25
TopK64 V240 LCB > 0
TopK64 longrisk UCB <= 0.05
cone score separates generated failures from known FastSlow with AUC >= 0.75
```

### 可视化

```text
p2_goodsubspace_projection_vs_conescore.svg
p2_cone_score_by_path_type_violin.svg
p2_generated_projection_vs_cone_failure.svg
p2_quadratic_metric_eigen_spectrum.svg
```

### 如果不满足，Codex 先尝试

```text
如果 projection high 但 cone score low：generated family orientation 错，进入 P6 orientation-repaired generation。
如果 cone score 对 known actions 也低：GoodSubspace basis 不足，进入 P4 basis expansion。
如果 cone score precision 高但 risk 高：进入 P3 soft safety constraint。
如果 score 只在某 dataset/template 好：标记为 non-official diagnostic，不做 controller。
```

---

## P3：Memory/offdiag soft trust-region Pareto

### 目标

判断 memory/offdiag 是否把 value direction 硬投影掉。v10.4 P4 joint retention 只有 0.1987，所以必须做 Pareto，而不是继续硬 veto。

### 假设

$$
H3:
\text{memory/offdiag hard projection 过强；soft trust region 能保留更多 value direction。}
$$

### 方法

对每个 candidate update 计算：

$$
\Delta_\alpha
=
(1-\alpha)\Delta_{value}+\alpha\Pi_{safe}(\Delta_{value})
$$

或 penalty form：

$$
\max_\Delta
V_{future}(\Delta)
-
\lambda_M M(\Delta)
-
\lambda_O O(\Delta)
-
\lambda_N \|\Delta\|^2
$$

扫：

```text
alpha = 0.0, 0.1, 0.25, 0.5, 0.75, 1.0
lambda_M / lambda_O grid
```

### 必须记录

```text
alpha
lambda_M
lambda_O
value_retention
FastSlow coverage after projection
V240 LCB
RAUV LCB
longrisk UCB
memory_fail UCB
offdiag_fail UCB
BadPath rate
RiskyHighAUV rate
action_norm_delta
apply_linf
```

### 判断标准

P3 weak pass：

```text
exists alpha/lambda:
  value_retention >= 0.40
  V240 LCB > 0
  longrisk UCB <= 0.10
  memory/offdiag UCB <= 0.10
```

P3 strong pass：

```text
value_retention >= 0.60
V240 LCB > 0
longrisk UCB <= 0.05
memory/offdiag UCB <= 0.05
FastSlow precision TopK32 >= 0.20
```

### 可视化

```text
p3_value_vs_risk_pareto_front.svg
p3_alpha_retention_curve.svg
p3_memory_offdiag_projection_damage.svg
p3_soft_trust_region_path_type_heatmap.svg
```

### 如果不满足，Codex 先尝试

```text
如果 alpha 低时 value 好但 risk 高：调 risk penalty，不调 value score。
如果 alpha 高时 risk 好但 value 负：说明 hard projection 毁 value，进入 P4 basis expansion。
如果所有 alpha value 负：当前 value direction 本身错，回 P2/P4。
如果 apply_linf 非 0 或 payload drift：先修 action apply，不解释 science。
```

---

## P4：Basis coverage / controllability audit

### 目标

判断当前 update basis 是否包含 Fast/Slow value direction。若 basis 不覆盖，再好的 selector / adjoint 都没用。

### 假设

$$
H4:
\text{当前 generated basis 缺少 future-path value direction。}
$$

### 构造 basis

```text
B0: AdamW update direction
B1: momentum direction
B2: variance-normalized direction
B3: exact-transfer matched contrast direction
B4: optimizer-state-alignment direction
B5: old-family-preserving direction
B6: memory-safe projected direction
B7: offdiag-safe projected direction
B8-B15: low-rank Krylov / JVP response directions
B16-B31: FastSlow-minus-Bad matched contrast basis
```

### 必须记录

```text
basis_id
basis_dim
condition_number
FastSlow projection coverage
Risky/Bad projection coverage
FastSlow cone margin
basis direction memory/offdiag risk
basis direction V240 proxy
basis direction exact_transfer
basis direction optimizer_alignment
```

### 判断标准

P4 weak pass：

```text
FastSlow projection coverage >= 0.60
FastSlow-vs-Bad cone margin > 0
memory/offdiag risk of top basis directions <= 0.20
```

P4 strong pass：

```text
FastSlow projection coverage >= 0.80
FastSlow-vs-Bad cone margin >= 0.10
risk-safe basis subset dimension >= 4
```

### 可视化

```text
p4_basis_projection_by_path_type.svg
p4_basis_condition_spectrum.svg
p4_fastslow_cone_margin_by_basis_family.svg
p4_basis_risk_value_quadrant.svg
```

### 如果不满足，Codex 先尝试

```text
如果 coverage 低：增加 matched contrast basis，不增加 random basis。
如果 condition number 高：orthogonalize / QR basis。
如果 risk-safe dimension 太低：不要 hard-project，全用 soft trust region。
如果 FastSlow 和 Bad 都高 coverage：必须用 GoodCone，不是 subspace。
```

---

## P5：Mini-unroll adjoint validation

### 目标

验证是否能用短 unroll 近似未来路径 co-state，而不是退化成 current response。

### 方法

对 subset actions 运行 tiny virtual AdamW unroll：

```text
unroll steps = 1, 3, 5, 10
anchor samples = train-memory / hard-tail / old-family / current batch stratified subset
```

估计：

$$
\hat p_t^{(k)}
=
\nabla_{\theta_t}
M(U_k(\theta_t))
$$

并测试：

$$
\langle \hat p_t^{(k)},\Delta \rangle
$$

与 path type / V240 / RAUV 的关系。

### 必须记录

```text
unroll_k
anchor_set_id
cost_ms_q90
corr_with_V1
partial_corr_V240_given_V1
AUC_FastSlow_vs_Bad
AUC_SlowBurn_vs_Risky
TopK precision
V240 LCB
longrisk UCB
memory/offdiag UCB
```

### 判断标准

P5 weak pass：

```text
partial_corr(V240 | V1) >= 0.10
AUC_SlowBurn_vs_Risky >= 0.65
cost q90 <= 10 ms discovery
```

P5 strong pass：

```text
partial_corr(V240 | V1) >= 0.20
TopK32 FastSlow precision >= 0.20
V240 LCB > 0
longrisk UCB <= 0.05
```

### 可视化

```text
p5_partial_corr_by_unroll_k.svg
p5_unroll_score_vs_V240_scatter.svg
p5_anchor_set_ablation.svg
p5_cost_vs_signal_tradeoff.svg
```

### 如果不满足，Codex 先尝试

```text
如果 k=1/3 无信号但 k=10 有信号：future path 需要 longer unroll，进入 cost compression。
如果所有 k 无信号：adjoint approximation still wrong，回 P4 basis/P2 cone。
如果 signal 有但 cost 太高：做 cached JVP/VJP 或 smaller anchor set。
如果 signal 只对 FastGood 有效不对 SlowBurn：目标仍 immediate-biased，重设 SlowBurn loss。
```

---

## P6：GoodCone + soft trust + basis-expanded generated discovery

### 目标

只有 P2/P3/P4/P5 至少一条给出可用线索，才生成新动作。生成不是 APG-style primitive，而是在 basis + cone + soft trust region 下解系数。

### 解法

$$
\alpha^*
=
\arg\max_\alpha
ConeScore(B\alpha)
+
\beta UnrollScore(B\alpha)
-
\lambda_M M(B\alpha)
-
\lambda_O O(B\alpha)
-
\lambda_N\|\alpha\|^2
$$

然后：

$$
\Delta^*=B\alpha^*
$$

### 阶梯

```text
Stage4: 4 actions per family
Stage8: only if Stage4 has >=1 Fast/Slow-like and BadPath <= 0.75
Stage16: only if Stage8 weak pass
Stage32: only if Stage16 weak pass
Stage64: only if Stage32 weak pass
```

### 必须记录

```text
family_id
stage
action_count
basis_set
cone_score
unroll_score
alpha_norm
payload_norm
apply_linf
V1/V5/V20/V80/V240
RAUV
FastGood count
SlowBurnGood count
RiskyHighAUV count
SafeLowValue count
BadPath count
FastSlow precision
V240 LCB
longrisk UCB
memory/offdiag UCB
bad/null UCB
```

### 判断标准

Stage4 weak-like：

```text
FastSlow count >= 1
BadPath count <= 3
longrisk UCB <= 0.50
```

Stage8 weak：

```text
FastSlow precision >= 0.125
V240 LCB > -0.10
longrisk UCB <= 0.25
```

Stage32 discovery pass：

```text
FastSlow precision >= 0.20
V240 LCB > 0
longrisk UCB <= 0.10
memory/offdiag UCB <= 0.10
```

Stage64 strong discovery pass：

```text
FastSlow precision >= 0.25
V240 LCB > 0
RAUV LCB > 0
longrisk UCB <= 0.05
memory/offdiag UCB <= 0.05
BadPath rate <= 0.50
```

### 可视化

```text
p6_generated_stage_waterfall.svg
p6_generated_path_type_stack.svg
p6_generated_value_risk_scatter.svg
p6_alpha_coefficients_heatmap.svg
p6_basis_family_contribution.svg
```

### 如果不满足，Codex 先尝试

```text
如果 Stage4 全 BadPath：不要扩大；检查 cone_score 与 unroll_score 是否都偏高，若是 objective sign 错。
如果 value 正但 longrisk 高：提高 memory/offdiag penalty，不调 value target。
如果 risk 低但 value 负：说明只生成 veto-safe 动作，加入 value basis / delayed gain basis。
如果 apply_linf 非 0：先修 payload apply，不解释结果。
如果 all generated projection high but all BadPath：GoodCone 仍不是好判据，回 P2。
```

---

## P7：FPO path-type predictor rebuild

### 目标

在 P2/P5/P6 有可用 signal 后，重建 FPO。FPO 不再预测 scalar good score，而预测 path type。

### 输出

```text
p_fast
p_slow
p_risky
p_safelow
p_bad
```

官方只能使用 legal commit-time features；所有 future label 只用于训练 / heldout evaluation，不进入 commit feature。

### 必须记录

```text
feature_family
cost_ms_q90
TopK64 FastSlow precision
SlowBurn recall
Risky rejection rate
Bad rejection rate
V240 LCB
longrisk UCB
memory/offdiag UCB
ECE_path_type
leaveout drop
```

### 判断标准

P7 weak pass：

```text
TopK64 FastSlow precision >= 0.15
SlowBurn recall >= 0.20
Risky rate <= 0.20
V240 LCB > -0.05
cost q90 <= 1.5 ms discovery
```

P7 strong pass：

```text
TopK64 FastSlow precision >= 0.25
SlowBurn recall >= 0.30
Risky rate <= 0.10
V240 LCB > 0
longrisk UCB <= 0.05
cost q90 <= 1.5 ms
```

### 如果不满足，Codex 先尝试

```text
如果 precision 低但 risky rejection 高：FPO 是 risk veto，不是 selector；拆成 veto head。
如果 SlowBurn recall 低：增加 unroll / optimizer-state features。
如果 cost 高：移除 expensive virtual features，保留 cached basis projections。
如果 leaveout drop 高：分层诊断，不按 dataset 调参。
```

---

## P8：Controller boundary

### 前提

P6 Stage64 strong discovery 或 P7 strong pass。

### 目标

形成 source controller candidate，但不直接跑 paired replay。

### 必须记录

```text
accepted_count
FastSlow precision
V20/V80/V240 LCB
RAUV LCB
longrisk UCB
bad/null UCB
memory/offdiag UCB
LDO/LSO/LTO/LFO
cost_ms_q90
feature legality ledger
```

### Pass 标准

```text
accepted_count >= 87
FastSlow precision >= 0.25 discovery 或 GradeAB-like precision >= 0.75 official equivalent
V240 LCB > 0
RAUV LCB > 0
longrisk UCB <= 0.05
bad UCB <= 0.05
null UCB <= 0.15
memory/offdiag UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10 strong, <=0.20 weak
```

### 如果不满足，Codex 先尝试

```text
如果 accepted_count < 87：不要放宽 risk；先增加 generated families 或 natural reference only。
如果 precision 高但 LDO 高：查 support/backfill，不按 dataset 调参。
如果 V240 负：回 P6 objective。
如果 risk 高：回 P3 soft trust region。
```

---

## P9：Runtime boundary

### 前提

P8 pass。

### 目标

测 selected controller runtime，不使用 offline materializer timing。

### 必须记录

```text
step_ratio_q50/q90/q95
feature_cost_q90
controller_kernel_count
sync_count
active_step_count
empty_step_count
payload_apply_ms_q90
memory_usage_peak
```

### Pass 标准

```text
step_ratio_q90 <= 1.50
feature_cost_q90 <= 1.5 ms
no CPU offload
```

### 如果不满足，Codex 先尝试

```text
如果 feature cost 高：缓存 basis projection / remove virtual unroll features。
如果 empty-step kernels 多：active-step batching。
如果 payload apply 慢：fuse payload apply / bucket by action type。
```

---

## P10：Paired replay / short-full boundary

### 前提

P8 + P9 pass。

### 目标

只在 controller + runtime 过线后验证 causal advantage。

### 必须记录

```text
RealFunctional
AdamWParallel
BestLR
NoOp
Random
ShuffledPayload
train/test acc
CE/NLL/ECE
CEp99
margin_p10
time_to_target
memory/offdiag
longrisk
runtime step ratio
```

### Pass 标准

```text
RealFunctional beats AdamWParallel and BestLR on paired replay
ShuffledPayload fails
NoOp/Random fail
ECE/NLL/CEp99 not worse
runtime remains <= 1.50 q90
```

---

# 5. 并行执行策略

v10.5 不再串行等一个 blocker。分三类 runner。

## Engineering Gate Runner

```text
P0, apply_linf, payload hash, branch-horizon smoke, runtime smoke。
失败立即停，不跑 science。
```

## Science Discovery Runner

```text
P1-P7 可并行。
不写 official pass。
允许小规模 generated discovery。
```

## Official Gate Runner

```text
只有 P6/P7 强信号后才跑 P8-P10。
```

---

# 6. v10.5 的核心判定树

```text
Case A: GoodCone 成立，generated Stage64 过
  -> 进入 controller/runtime。

Case B: GoodCone 成立，但 generation 失败
  -> 当前 basis 不可控；做 basis expansion / architecture route。

Case C: GoodCone 不成立，但 unroll adjoint 有信号
  -> 用 unroll-FPO 做 path-type selector。

Case D: GoodCone 和 unroll adjoint 都失败
  -> 当前 representation 不足；停止 FPAU/FPO 小修，回到 architecture / optimizer-state propagation / compositional FullEdge。

Case E: memory/offdiag hard conflict 毁掉所有 value
  -> 不能硬 veto；必须 soft constraint / trust region。
```

---

# 7. 最终目标

v10.5 的最低有效推进不是 system success，而是要把 v10.4 的模糊 blocker：

```text
good_value_subspace_not_stable
```

拆成明确归因：

```text
1. good cone exists / not exists；
2. current basis covers / does not cover FastSlow value direction；
3. memory/offdiag constraints destroy / preserve value；
4. mini-unroll adjoint carries / does not carry future signal；
5. generated family orientation correct / wrong。
```

只有这些问题回答后，后续才不是继续猜动作。

