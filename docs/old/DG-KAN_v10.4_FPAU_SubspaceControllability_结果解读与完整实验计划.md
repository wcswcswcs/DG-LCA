# DG-KAN v10.3 结果解读与 v10.4 完整实验计划：FPAU Subspace Controllability / Future-Path Value Direction

> 本文基于 v10.3 `Update Rule Theory Rebuild / FPAU` 的真实执行结果制定。目标不是继续给 FPAU-A0-A7、GUP13、FPO13 或 generated family 做阈值小修，而是把失败拆成更基础的问题：当前 update 子空间是否有能力表达未来路径改善方向；当前 adjoint 是否真的在估计未来训练动力学；memory/offdiag 约束是否把所有价值方向都投掉了。
>
> 本计划中公式使用 `$...$` 或 `$$...$$`，Typora 友好。

---

# 0. 一句话判断

v10.3 的结果说明：

$$
\boxed{
\text{FPAU toy 可解，但真实系统里的 FPAU 子空间没有产生未来路径价值方向。}
}
$$

这不是“FPAU 理论完全失败”，而是一个更具体、更硬的失败：

```text
1. 在 toy mechanism 上，repair 后 FPAU 确实能打过 current-gradient baseline；
2. 但在真实 known actions 上，FPAU scoring 几乎选不到 FastGood / SlowBurnGood；
3. 在真实 generated branch-horizon 上，Stage8 仍没有 weak signal；
4. 失败主因已经从“没有理论”推进到“update 子空间缺少 value direction 或违反 memory/offdiag 约束”。
```

因此 v10.4 不应该继续问：

```text
FPAU-A7 阈值能不能调一点？
Stage8 能不能直接扩大到 Stage32？
FPO13D 能不能换个权重？
GUP13 能不能换个 family 名字？
```

v10.4 要回答的是：

$$
\boxed{
\text{真实 Fast/Slow 好路径是否存在一个可表达、可约束、可生成的 value-direction 子空间？}
}
$$

如果答案是否定的，就应该承认当前 action/update space 不够，需要更高层的 update-rule 或 architecture subspace reset，而不是继续采样和筛选。

---

# 1. v10.3 独立复盘

## 1.1 v10.3 有进展，但不是能力进展

v10.3 的 terminal route 是：

```text
route = R5-SubspaceMissingValueDirection
primary_blocker = fpau_known_action_or_toy_signal_present
secondary_blocker = generated_subspace_failed
system_legal_controller_pass = 0
generated_route_status = stopped_fpau_subspace_missing_value_direction
```

核心结果：

```text
P0 boundary lock pass = 1
P1 Fast / Slow / Risky / SafeLow / Bad = 12 / 48 / 842 / 60 / 34286
P1 FastSlow total = 60 < 64
P2 FPAU weak/discovery/controller = 0 / 0 / 0
P2 best = FPAU-A7-ensemble-costate-hard-risk-veto
P2 TopK64 FastSlow precision = 0.015625
P2 TopK87 V240 LCB = -0.41331319354703405
P3 repaired toy mechanism pass = 1
P3 beats current-gradient toys = 4 / 4
P3 slowburn 3x random toys = 4 / 4
P3 memory toy nonpositive = 1
P4 Stage4 / Stage8 / Stage32 / Stage64 opened = 6 / 6 / 0 / 0
P4 repair Stage8 opened/pass = 3 / 0
P4 generated discovery pass = 0
P5 dominant failure = F2-adjoint-current-response-collapse
P4 major failure classes = F4-subspace-missing-value-direction and F5-subspace-violates-memory-offdiag
P6/P7/P8 = not_run / not_run / not_run
no fake / proxy / cpu = 0 / 0 / 0
```

这说明 v10.3 不是 runner 没跑，也不是 toy 没过。真实情况更尖锐：

$$
\boxed{
\text{toy 上的 future-path objective 可解；真实 update 子空间里却没有把该 objective 转成好路径。}
}
$$

## 1.2 这轮最重要的正信号

P3 toy repair pass 很重要。它说明 FPAU 的数学形式不是完全荒谬：

```text
repair 后 mechanism pass = 1；
beats current-gradient toys = 4 / 4；
slowburn 3x random toys = 4 / 4；
memory toy nonpositive = 1。
```

这意味着至少在受控机制里，future-path objective 比 current-gradient objective 更接近我们要的东西。

但这个正信号非常有限。toy 成功不能说明真实模型子空间可控，因为 toy 缺少真实系统里的几个困难：

```text
1. multi-horizon dynamics；
2. AdamW optimizer-state propagation；
3. memory/offdiag constraints；
4. action payload discretization / certificate route；
5. old-family / hard-tail interference；
6. parameter subspace capacity limit；
7. branch-horizon stochasticity。
```

## 1.3 这轮最重要的负信号

P2 和 P4 的失败比 P3 的 toy pass 更关键。

P2 在 `15080` 个真实 known actions 上评估 FPAU A0-A7，最好的 FPAU-A7 TopK64 FastSlow precision 只有 `0.015625`，TopK87 V240 LCB 为负。这说明 FPAU scoring 没有解释真实好路径。

P4 在 generated branch-horizon 上失败更严重：Stage4/8 虽然打开，但 Stage8 仍没有 weak signal，Stage32/64 正确不打开。

这不是“数据集调参不够”，而是：

$$
\boxed{
\text{真实 generated update 子空间里缺少可用 value direction，或 value direction 被 memory/offdiag 约束压掉。}
}
$$

---

# 2. 四线当前状态

## 2.1 A 线：Future Path Type

A 线仍然是最有科学价值的线，但 v10.3 说明好路径非常稀疏。

```text
FastGood = 12
SlowBurnGood = 48
RiskyHighAUV = 842
SafeLowValue = 60
BadPath = 34286
FastSlow total = 60
```

P1 discovery gate 要求 FastSlow >= 64，所以 P1 没过。

这个结果有两个含义：

```text
1. FastGood / SlowBurnGood 不是空集；future-path 好动作确实存在。
2. 它们在当前 action universe 里极稀疏，不能靠自然 harvesting 或粗 rank 形成 controller。
```

A 线下一步不能再只数 Fast/Slow 数量，也不能只看 AUV。必须进入 **path mechanism contrast**：

```text
FastGood vs BadPath；
SlowBurnGood vs RiskyHighAUV；
SlowBurnGood vs SafeLowValue；
SlowBurnGood vs Exact/current-response selected bad actions。
```

核心问题是：

$$
\boxed{
\text{SlowBurnGood 到底在参数扰动、函数响应、memory/offdiag、optimizer-state 传播上有什么共同结构？}
}
$$

## 2.2 B 线：FuturePathOperator / FPAU scoring

B 线继续失败。v10.3 的 P2 说明 FPAU-A0-A7 都没有在 known actions 上选出好路径。v10.2 的 FPO13 也失败，best FPO13D TopK64 precision 只有 `0.015625`，V240 LCB 为负。

现在可以明确降级以下假设：

```text
H_B_old_1: current-response / transfer / SNR 可以作为好路径 proxy。
H_B_old_2: memory/offdiag hardgate + delayed-gain proxy 可以识别 SlowBurnGood。
H_B_old_3: FPAU scoring 用一个 ensemble-costate 就能解释 known Fast/Slow。
```

新的 B 线假设必须是：

$$
\boxed{
\text{好路径不能用一个 scalar proxy 解释，必须先找到真实 value-direction 子空间。}
}
$$

## 2.3 C 线：Natural AP0 density

C 线已经不应继续作为主线。v9.9.8 / v10.0 / v10.1 / v10.2 已经把 natural harvesting 降级：保真自然 panel 给出了 density insufficient，C3 20000 因 cost cap 没跑，但 5000 / 10000 已足够说明 natural harvesting 不应再承担主希望。

C 线在 v10.4 里只保留为：

```text
1. 参照分布；
2. sanity audit；
3. 确认 generated / solved update 是否偏离自然动作太远；
4. 不再作为主路线。
```

不要再把主要资源投入 20000/50000 natural panels。

## 2.4 D 线：Generated / Solved Update

D 线是必须继续的，但形式要改。

v10.3 证明：

```text
Stage4 / Stage8 可以打开；
但 Stage8 没有 weak signal；
repair Stage8 也没过；
Stage32 / Stage64 正确不打开。
```

这说明不是规模问题。现在不是把 Stage8 扩到 32/64，而是要问：

$$
\boxed{
\text{当前 generated subspace 是否包含 Fast/Slow 的 value direction？}
}
$$

如果不包含，再多生成也是在坏子空间里采样。

---

# 3. 当前真正卡在哪里

## 3.1 不是缺动作，而是缺 value-direction subspace

过去几轮已经排除很多低层问题：

```text
branch-horizon materializer 可跑；
payload apply 可闭合；
no-fake/no-proxy/cpu audit 通过；
natural generator 可做保真采样；
exact transfer 可准确测量；
toy FPAU 可 repair；
```

但真实系统仍然失败。

这说明关键 blocker 已经上移：

$$
\boxed{
\text{我们缺的不是更多动作样本，而是一个能表达未来价值方向的 update 子空间。}
}
$$

## 3.2 FPAU 的 adjoint 可能退化成 current-response

P5 dominant failure 是：

```text
F2-adjoint-current-response-collapse
```

这意味着 FPAU 的 co-state / adjoint 在真实实现里可能仍在看当前响应，而不是未来训练动力学。

形式上，我们想要的是：

$$
p_t=
\sum_{h\in H}
D U_h(\theta_t)^\top
\nabla_{\theta_{t+h}} M_h.
$$

但实际算到的可能更像：

$$
p_t \approx \nabla_{\theta_t} M_1,
$$

也就是又回到了 one-step response。

下一步必须测 **collapse index**：

$$
CI=\operatorname{corr}(Score_{FPAU}, V_1)-\operatorname{corr}(Score_{FPAU}, V_{240}).
$$

如果 $CI$ 很大，说明 FPAU 仍然被当前响应主导。

## 3.3 Memory/offdiag 约束可能把 value direction 投掉

P4 失败类包含：

```text
F5-subspace-violates-memory-offdiag
```

这说明另一个可能：真实 value direction 存在，但一加 memory/offdiag hard constraints 就被投掉，留下的只是不伤但无收益的方向。

这要用 constrained projection audit 测：

$$
\Delta_{value}
\rightarrow
\Pi_{\mathcal{N}_{memory/offdiag}}(\Delta_{value}).
$$

记录 value retention：

$$
Retention =
\frac{
\langle \Delta_{value}, \Pi_{\mathcal{N}}(\Delta_{value}) \rangle
}{
\|\Delta_{value}\|^2
}.
$$

如果 retention 很低，说明 memory/offdiag 安全空间和 value 空间冲突，需要重新设计约束，不是简单 hard veto。

## 3.4 Toy 机制太弱，不足以代表真实系统

P3 toy pass 是好消息，但 toy 可能没有覆盖真实失败模式。v10.4 必须升级 toy：

```text
Toy 1: delayed value with memory-neutral direction；
Toy 2: delayed value but current-response misleading；
Toy 3: value direction conflicts with memory nullspace；
Toy 4: offdiag-safe but low-value direction；
Toy 5: optimizer-state propagation matters；
Toy 6: stochastic branch-horizon noise。
```

如果 FPAU 只能过简单 toy，不会过 conflict toy，就不能说 theory 有用。

---

# 4. v10.4 总体目标

v10.4 的总目标不是 system pass，而是判断 FPAU 是否还有科学路线。

$$
\boxed{
\text{v10.4 要证明或证伪：真实 Fast/Slow 是否存在可表达、可约束、可生成的 update subspace。}
}
$$

v10.4 的四个硬问题：

```text
Q1. Fast/Slow 的真实 delta 是否形成可区分的 value-direction basis？
Q2. FPAU score 是否真的在估计 future co-state，而不是 current response？
Q3. memory/offdiag projection 是否保留 value direction？
Q4. 如果用 value-direction basis / repaired adjoint / memory-aware projection 生成 update，是否能在 4/8/16 actions 内产生至少一个 Fast/Slow-like？
```

如果 Q1-Q4 都失败，就应停止 FPAU 路线，转向 architecture/subspace reset。

---

# 5. P0：复现 v10.3 边界

## 目标

锁定 v10.3 的真实 terminal boundary，不重复争论旧路线。

## 要记录

```text
route
P1 Fast/Slow/Risky/SafeLow/Bad
P2 best FPAU known-action score
P3 toy repair pass
P4 Stage4/8/32/64 opened/pass
P5 dominant failure class
no-fake/no-proxy/no-cpu audit
```

## 判断标准

P0 pass：

```text
source route = R5-SubspaceMissingValueDirection
P3 mechanism pass = 1
P4 generated discovery pass = 0
P6/P7/P8 not_run
fake/proxy/cpu = 0/0/0
```

## 不满足时 Codex 先尝试

```text
如果 source artifact missing：定位 v10.3 out-dir / manifest hash；
如果 P3/P4 字段缺失：只读取 landed CSV/JSON，不重新构造；
如果 no-fake audit 缺失：先补 audit，不进入 P1。
```

---

# 6. P1：Path-Type Mechanism Contrast

## 目标

把 Fast/Slow/Risky/SafeLow/Bad 从“计数标签”变成机制对象。

## 假设

$$
H_{P1}:
\text{SlowBurnGood 与 RiskyHighAUV / BadPath 在参数扰动或函数响应上存在稳定差异。}
$$

## 实验设计

构造 matched contrast pairs：

```text
SlowBurnGood vs RiskyHighAUV
SlowBurnGood vs BadPath
FastGood vs BadPath
SafeLowValue vs SlowBurnGood
Exact/current-response selected bad actions vs Old/FastSlow actions
```

匹配轴：

```text
dataset
family
template
step_bucket
action_norm_bucket
payload_norm_bucket
memory/offdiag bucket
source/generator family
```

只使用训练当下或 payload-level 可见字段做 contrast；future label 只用于分组，不进入 controller。

## 要记录指标

```text
matched_pair_count per contrast
coverage of Fast/Slow actions by matched pairs
mean_delta_action_norm
mean_delta_payload_norm
mean_delta_adamw_alignment
mean_delta_current_response
mean_delta_exact_transfer
mean_delta_memory_score
mean_delta_offdiag_score
mean_delta_old_family_margin
mean_delta_hard_tail_response
mean_delta_basis_rank_proxy
mean_delta_cover_entropy_proxy
mean_delta_optimizer_state_alignment
```

## 可视化

```text
fig_p1_path_type_counts_bar.svg
fig_p1_slowburn_vs_risky_feature_effects.svg
fig_p1_matched_pair_delta_heatmap.svg
fig_p1_path_type_embedding_pca.svg
fig_p1_slowburn_current_vs_future_scatter.svg
```

## 判断标准

P1 discovery pass：

```text
matched SlowBurn pairs >= 30
at least 3 commit-time feature families show effect size >= 0.50
SlowBurn vs Risky/Bad separation AUC >= 0.70 on matched contrast
no single dataset/family/template share > 0.50
```

P1 fail：

```text
matched SlowBurn pairs < 10
or all commit-time feature AUC < 0.60
or SlowBurn concentrated in one source family/template
```

## 不满足时 Codex 先尝试

```text
如果 matched pairs 太少：放宽 matching 中 action_norm/payload_norm，但不放宽 dataset/family/template；
如果 SlowBurn 太少：合并 FastGood + SlowBurnGood 做 first pass，再单独拆 SlowBurn；
如果所有 feature 无效：检查 payload/action delta 是否缺少必要字段；
如果 feature effect 只由 dataset 解释：标为 diagnostic，不进入 controller。
```

---

# 7. P2：Known Good Value-Direction Basis Discovery

## 目标

判断 FastGood / SlowBurnGood 是否在参数扰动空间或函数响应空间形成可表达的 value-direction basis。

这一步是 v10.4 最重要的新实验。

## 假设

$$
H_{P2}:
\text{Fast/Slow actions 在 delta-space 中形成一个低维 value subspace，当前 FPAU generated subspace 没覆盖它。}
$$

## 实验对象

```text
Good set = FastGood + SlowBurnGood
Risky set = RiskyHighAUV
Bad set = BadPath
SafeLow set = SafeLowValue
GeneratedFPAU set = v10.3 Stage4/8 generated deltas
```

## 表示空间

并行建立三种表示：

```text
R_param：参数 delta 向量或 low-rank sketch；
R_func：per-example output response / JVP response sketch；
R_mem：memory / hard-tail / old-family response vector。
```

如果全量参数太贵，使用以下 sketch：

```text
last-edge block sketch；
top-k norm blocks；
random projection 512 / 2048 dims；
blockwise norm + sign + cosine sketch；
JVP response on stratified mini-buffer。
```

## 要记录指标

```text
GoodSubspace rank_k explained variance
GoodSubspace projection of Fast/Slow actions
GoodSubspace projection of Risky/Bad actions
FPAU_generated_projection_to_GoodSubspace
cosine(mean_good_delta, mean_bad_delta)
cosine(mean_good_delta, mean_generated_delta)
projection_retention_after_memory_projection
projection_retention_after_offdiag_projection
within_group_cosine_FastSlow
between_group_cosine_FastSlow_vs_Bad
```

定义：

$$
Coverage_{good}(S)=
\frac{1}{|G|}
\sum_{a\in G}
\mathbf{1}
\left(
\frac{\|P_S\Delta_a\|}{\|\Delta_a\|}\ge \tau_p
\right).
$$

$$
Separation(S)=
E_{a\in Good}\frac{\|P_S\Delta_a\|}{\|\Delta_a\|}
-
E_{a\in Bad}\frac{\|P_S\Delta_a\|}{\|\Delta_a\|}.
$$

## 可视化

```text
fig_p2_delta_pca_good_risky_bad.svg
fig_p2_projection_hist_by_path_type.svg
fig_p2_generated_vs_good_subspace_projection.svg
fig_p2_projection_retention_memory_offdiag.svg
fig_p2_subspace_rank_explained_variance.svg
```

## 判断标准

P2 pass：

```text
GoodSubspace k<=16 explains >= 60% Fast/Slow variance
Coverage_good >= 0.60 at tau_p=0.50
Separation >= 0.25
FPAU_generated_projection_to_GoodSubspace <= 0.25
```

解释：如果 pass，说明 FPAU 失败很可能是 generated subspace 没覆盖 good value direction，而不是 future path objective 完全错。

P2 fail：

```text
GoodSubspace cannot explain Fast/Slow variance
or Fast/Slow indistinguishable from Bad/Risky in all sketches
or generated subspace already covers GoodSubspace but outcomes still bad
```

## 不满足时 Codex 先尝试

```text
如果参数 delta 缺失：优先落 durable delta sketch，不重跑 branch-horizon；
如果 full delta 太大：使用 random projection + blockwise sketch；
如果 GoodSubspace variance 低：按 FastGood 和 SlowBurnGood 分开建 subspace；
如果 generated projection 高但结果坏：转 P4 memory/offdiag feasibility，说明约束或 scale 破坏；
如果 GoodSubspace 不存在：停止 FPAU subspace route，转 architecture/subspace reset。
```

---

# 8. P3：Adjoint Collapse Audit

## 目标

判断 FPAU 的 adjoint / co-state 是否真的在估计 future path，还是退化成 current response。

## 假设

$$
H_{P3}:
Score_{FPAU}
\text{ 与 } V_1 \text{ 的相关性显著高于与 } V_{240} \text{ 或 SlowBurn 标签的相关性。}
$$

如果 H 成立，说明 FPAU 仍然 current-response collapse。

## 要记录指标

对每个 FPAU A0-A7：

```text
corr(score, V1)
corr(score, V5)
corr(score, V20)
corr(score, V80)
corr(score, V240)
corr(score, RAUV)
corr(score, longrisk)
corr(score, memory_fail)
corr(score, offdiag_fail)
partial_corr(score, V240 | V1)
partial_corr(score, SlowBurn | V1)
TopK64 FastSlow precision
TopK64 RiskyHighAUV rate
current_response_collapse_index
```

定义：

$$
CCI =
\operatorname{corr}(Score,V_1)
-
\operatorname{corr}(Score,V_{240}).
$$

$$
PCI =
\operatorname{corr}(Score,V_{240}\mid V_1).
$$

## 可视化

```text
fig_p3_score_corr_by_horizon.svg
fig_p3_current_response_vs_score.svg
fig_p3_score_vs_v240_partial.svg
fig_p3_topk_path_type_stack.svg
```

## 判断标准

P3 collapse confirmed：

```text
CCI >= 0.20
partial_corr(score, V240 | V1) <= 0.05
TopK64 FastSlow precision <= 0.05
TopK64 RiskyHighAUV or BadPath >= 0.70
```

P3 future-adjoint signal exists：

```text
partial_corr(score, V240 | V1) >= 0.20
TopK64 FastSlow precision >= 0.15
RiskyHighAUV UCB <= 0.30
```

## 不满足时 Codex 先尝试

```text
如果 correlation noisy：bootstrap over dataset/family/template；
如果 V1 missing：use h1 landed rows or exact apply response；
如果 score tied/degenerate：inspect score distribution entropy；
如果 score dominated by risk hardgate：separate value score and risk veto before correlation。
```

---

# 9. P4：Memory/Offdiag Constraint Feasibility

## 目标

判断 value direction 与 memory/offdiag safety 是否天然冲突。

## 假设

$$
H_{P4}:
\text{Fast/Slow value direction 在 memory/offdiag nullspace 中仍有足够 retention。}
$$

如果不成立，说明 hard memory/offdiag projection 会把 value 投掉。

## 实验设计

从 P2 得到 value basis $S_v$。构造 memory/offdiag risk subspace $S_r$。计算：

$$
S_{safe}=S_v \cap S_r^\perp.
$$

比较：

```text
raw value basis；
memory-projected value basis；
offdiag-projected value basis；
combined memory+offdiag projected value basis。
```

## 指标

```text
value_retention_memory
value_retention_offdiag
value_retention_joint
predicted_longrisk_delta
memory_harm_delta
offdiag_harm_delta
old_family_margin_delta
hard_tail_loss_delta
projection_norm_shrinkage
```

## 判断标准

P4 pass：

```text
joint value retention >= 0.50
predicted longrisk proxy <= threshold
memory/offdiag harm <= threshold
Fast/Slow GoodSubspace coverage after projection >= 0.40
```

P4 fail：

```text
joint value retention < 0.20
or projected basis becomes SafeLowValue-like but low-value
or longrisk remains high after projection
```

## 不满足时 Codex 先尝试

```text
如果 memory projection destroys value：use soft constraint / trust-region penalty instead of hard nullspace；
如果 offdiag projection destroys value：separate old-family and hard-tail constraints；
如果 risk remains high：add h240 longrisk proxy hard veto；
如果 retention depends on action family：split subspace by family/template。
```

---

# 10. P5：FPAU Toy Upgrade

## 目标

让 toy 不再过于简单。v10.3 toy pass 不能代表真实系统，v10.4 必须加入真实失败模式。

## Toy family

```text
ToyA-current-gradient-baseline:
  current loss descent sufficient 的简单 toy。

ToyB-slowburn-delayed-gain:
  h1 不好，但 h20/h80 好。

ToyC-current-response-misleading:
  current response 高的方向未来变坏。

ToyD-memory-value-conflict:
  value direction 与 memory-safe direction 部分冲突。

ToyE-offdiag-risk-conflict:
  offdiag-safe projection 可能投掉 value。

ToyF-optimizer-state-propagation:
  同一个 delta 在不同 AdamW moment state 下未来效果不同。

ToyG-branch-noise:
  horizon outcome 有随机噪声，需要 LCB 而不是 mean。
```

## 指标

```text
beats_current_gradient per toy
slowburn_3x_random per toy
memory_nonpositive per toy
risk_nonpositive per toy
value_retention_after_projection
adjoint_collapse_index per toy
```

## 判断标准

P5 pass：

```text
FPAU repaired objective beats current-gradient on >= 5 / 7 toy families
SlowBurn toy pass = 1
MemoryConflict toy pass = 1
OffdiagConflict toy pass = 1
```

P5 fail：

```text
Only simple toy pass；
conflict toys fail；
optimizer-state toy fail。
```

## 不满足时 Codex 先尝试

```text
如果 SlowBurn toy fail：increase unroll horizon, not current-response weight；
如果 memory conflict fail：replace hard projection by constrained optimization；
如果 optimizer-state fail：include AdamW moment state in adjoint；
if branch-noise fail：use LCB objective and robust aggregation。
```

---

# 11. P6：Generated Discovery with Subspace Diagnostics

## 目标

只有 P2/P3/P4 至少给出一个可用线索，才允许 generated discovery。它不是 official route，只用于机制学习。

## Candidate families

```text
FPAU14A-GoodSubspaceProjection:
  使用 P2 good value subspace 做低维 delta。

FPAU14B-GoodSubspaceMemorySoftConstraint:
  value basis + memory/offdiag soft penalty。

FPAU14C-AdjoinResidualNotCurrent:
  从 adjoint 中减去 current-response projection。

FPAU14D-OptimizerStateAwareAdjoint:
  把 AdamW moments 纳入 virtual unroll。

FPAU14E-SlowBurnTargetedDelta:
  专门优化 h20/h80/h240 delayed gain。

FPAU14F-NegativeControlShuffledSubspace:
  shuffled basis negative control。
```

## Stage ladder

```text
Stage4: 每个 family 4 actions；
Stage8: 仅 Stage4 出现至少 1 个 weak-like 才打开；
Stage16: 仅 Stage8 出现至少 1 个 Fast/Slow-like 且 longrisk UCB <= 0.30 才打开；
Stage32/64: 仅 Stage16 weak pass 才打开。
```

## 指标

```text
FastGood precision
SlowBurnGood precision
FastSlow precision
V1/V20/V80/V240 LCB
RAUV LCB
longrisk UCB
bad/null UCB
memory/offdiag UCB
new-positive count
longrisk-created rate
value-direction projection
memory/offdiag retention
current-response collapse index
payload apply error
rows expected/actual
```

## 判断标准

Stage8 weak pass：

```text
FastSlow count >= 1
V240 LCB > -0.20
longrisk UCB <= 0.30
memory/offdiag UCB <= 0.30
```

Stage16 discovery pass：

```text
FastSlow precision >= 0.10
V240 LCB > 0
longrisk UCB <= 0.20
memory/offdiag UCB <= 0.20
negative control worse
```

Stage64 candidate pass：

```text
FastSlow precision >= 0.25
V240 LCB > 0
RAUV LCB > 0
longrisk UCB <= 0.10
bad/null UCB <= 0.15
memory/offdiag UCB <= 0.10
```

## 不满足时 Codex 先尝试

```text
如果 Stage4 all BadPath：stop family，inspect norm/scale/current-response alignment；
如果 value negative but risk low：likely SafeLowValue; add value direction, not risk threshold；
如果 value positive but longrisk high：add memory/offdiag soft constraint, not scale up；
如果 high current-response but bad future：subtract current-response projection；
如果 negative control matches real family：subspace not meaningful; return to P2/P3；
if payload/apply fail：fix lifecycle before science interpretation。
```

---

# 12. P7：Path-Type FPO Rebuild

## 目标

不再让 FPO 预测单一 score，而是预测 path type。

## 模型

只作为 discovery，不作为 official controller。

输入特征：

```text
current response
exact transfer
memory/offdiag response
hard-tail response
old-family response
FPAU score components
value-subspace projection
risk-subspace projection
adjoint-collapse metrics
optimizer-state features
cost
```

输出：

```text
FastGood probability
SlowBurnGood probability
RiskyHighAUV probability
SafeLowValue probability
BadPath probability
```

## 指标

```text
SlowBurn recall@TopK
FastSlow precision@64
RiskyHighAUV rejection rate
BadPath rejection rate
V240 LCB@TopK
longrisk UCB@TopK
cost q90
leave-family/template split
```

## 判断标准

P7 discovery pass：

```text
TopK64 FastSlow precision >= 0.15
SlowBurn recall@64 >= 0.25
RiskyHighAUV rate@64 <= 0.25
V240 LCB > 0
longrisk UCB <= 0.10
cost q90 <= 1.5 ms discovery
```

P7 fail：

```text
TopK64 FastSlow precision < 0.05
or V240 LCB <= 0
or RiskyHighAUV rate high
```

## 不满足时 Codex 先尝试

```text
如果 precision low but risk low：model is veto-only; add value-subspace features；
if slowburn recall low：increase SlowBurn class weight only in discovery, not official；
if risky false positives high：hard-gate memory/offdiag before rank；
if cost high：cache JVP / reduce probe samples / use sketch features only。
```

---

# 13. P8：Controller Boundary

## 目标

只有 P6 或 P7 产生真实 candidate，才打开 controller boundary。

## Controller candidate 来源

```text
1. Generated Stage64 candidate；
2. Path-type FPO discovery candidate；
3. GoodSubspace + memory-soft constrained candidate。
```

## Official pass 条件

```text
accepted_count >= 87
FastSlow precision >= 0.75
V240 LCB > 0
RAUV LCB > 0
longrisk UCB <= 0.05
bad UCB <= 0.05
null UCB <= 0.15
memory/offdiag UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
no dataset-specific selector
no outcome-derived commit feature
```

## 不满足时 Codex 先尝试

```text
如果 accepted_count < 87 but precision high：do not relax risk; enlarge action source or generated family；
if precision low but risk clean：need value selector；
if LDO high：diagnose support vs quality vs backfill；
if memory/offdiag high：return to P4 constraint feasibility；
if feature red/yellow：quarantine, do not promote。
```

---

# 14. P9：Runtime Boundary

## 目标

仅 controller candidate 过 P8 时打开。

## 指标

```text
feature cost q90
payload apply q90
controller step ratio q90
peak memory ratio
active-step density
kernel launch count
sync count
```

## 判断标准

```text
step_ratio_q90 <= 1.50
peak_memory_ratio <= 1.05
payload_apply_error_linf <= tolerance
no offline materializer in timed path
```

## 不满足时 Codex 先尝试

```text
if feature cost high：cache sketches, reduce probe samples, remove high-cost features；
if payload apply slow：fuse apply kernels, compact active steps；
if empty-step overhead high：active-step scheduler；
if memory high：avoid storing per-action full vectors, use sketches。
```

---

# 15. P10：Paired Replay Boundary

## 目标

仅 P8/P9 过线后打开。

## 对照

```text
RealFunctional
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
CertificatePassNoPayload
```

## 指标

```text
CE/test proxy if available
V20/V80/V240
Fast/Slow event rate
longrisk
bad/null
memory/offdiag
calibration
runtime
```

## 判断标准

```text
RealFunctional beats AdamWParallel / bestLR / NoOp / Random
ShuffledPayload fails
CertificatePassNoPayload fails
leave-dataset/stratum/template/family pass
runtime still within envelope
```

---

# 16. 最终路线判断

v10.4 后根据结果做硬分叉：

## Case A：GoodSubspace exists, FPAU generated still misses

结论：generated construction wrong。继续修 subspace-to-payload / scale / memory-soft constraint。

## Case B：GoodSubspace does not exist

结论：current action/update representation 不足。停止 FPAU，在 architecture / compositional FullEdge / broader update subspace 上重置。

## Case C：Adjoint collapse confirmed

结论：当前 FPAU 不是真 future adjoint。必须引入 optimizer-state-aware unroll 或 implicit adjoint。

## Case D：Memory/offdiag projection destroys value

结论：hard veto 不可用，改 constrained optimization / soft penalty / trust region。

## Case E：Stage16/64 finally yields Fast/Slow

结论：进入 controller/runtime boundary，但不能跳过 official gates。

---

# 17. 当前最重要的结论

v10.3 让我们看到，项目不能再停留在：

```text
找 action；
筛 action；
调 FPO；
扩 generated family。
```

现在的科学问题是：

$$
\boxed{
\text{未来路径改善方向是否在当前可生成 update 子空间中可表达？}
}
$$

如果不可表达，任何 selector 都只是事后解释，任何 generated sampler 都是在坏子空间中采样。

v10.4 必须先解决 **subspace controllability**，再谈 controller。
