# DG-KAN v10.5 结果解读与 v10.6 Local Future Response Operator 完整实验计划

> 本文件基于 v10.5 `GoodCone / Stateful Controllability` 真实实验结果制定。目标不是继续小修 GoodCone、FPAU、FPO 或 generated family，而是把问题从“被动识别已有动作”推进到“主动测量并求解当前训练状态下的未来路径响应算子”。
>
> 所有公式使用 `$...$` 或 `$$...$$`，Typora 友好。

---

## 0. 当前结论

v10.5 的路线是：

```text
route = CaseD-GoodConeAndUnrollBothFailRepresentationInsufficient
primary_blocker = goodcone_not_valid
secondary_blocker = mini_unroll_adjoint_no_future_signal
system_legal_controller_pass = 0
generated_route_status = stopped_representation_insufficient
```

核心结果是：

```text
P1 state buckets = 720，weak/strong = 0 / 0，best bucket AUC = 1.0
P2 GoodCone weak/strong = 0 / 0，TopK32 precision/V240/risk = 0.03125 / -0.20918 / 0.96760
P3 soft Pareto weak/strong = 0 / 0，best alpha/lambda = 0.0 / 0.25，retention = 1.0
P4 basis coverage weak/strong = 0 / 0，FastSlow/Bad coverage = 1.0 / 1.0，risk-safe dim = 0
P5 mini-unroll weak/strong = 0 / 0，best k = 1，partial corr V240|V1 = -0.001765
P6 generated Stage4/8/32/64 opened = 0 / 0 / 0 / 0
P7 TopK64 precision/SlowBurn recall/risky/V240 = 0.046875 / 0.13636 / 0.15625 / -0.19644
```

一句话判断：

$$
\boxed{
\text{好路径存在，但当前表示、当前 GoodCone、当前 mini-unroll、当前 FPO 都不能把它变成可求解或可部署的更新规则。}
}
$$

这不是“阈值差一点”。这是路线层面的失败。v10.5 说明：

```text
1. state-conditioned contrast 过度碎片化，bucket 内样本太少；
2. GoodCone / quadratic metric 无法分开 FastSlow 和 Bad；
3. soft Pareto 能改善一部分风险/收益，但仍不过硬门；
4. 当前 basis 同时覆盖 FastSlow 和 Bad，说明 basis coverage 不是 controllability；
5. mini-unroll 没有产生 V240 residual signal；
6. FPO path-type predictor 仍选出负 V240、高 longrisk 的动作；
7. generated discovery 没有资格打开。
```

因此 v10.6 不应该继续：

```text
调 GoodCone threshold；
调 FPO path-type threshold；
调 soft Pareto alpha/lambda；
增加 FPAU/FPO/GUP 小 family；
把 Stage4 失败的 generated route 强行扩大；
继续用固定 action representation 解释 future path。
```

v10.6 的核心转向是：

$$
\boxed{
\text{从被动分类 action，转向主动识别当前训练状态下的 Local Future Response Operator。}
}
$$

---

## 1. 为什么 v10.5 令人失望但有价值

### 1.1 P1：state-conditioned contrast 不是机制，只是碎片化诊断

P1 的 best bucket AUC 是 `1.0`，但所有 weak/strong 都是 `0`。原因很直观：大多数 top bucket 里 FastSlow 数只有 `1` 或 `2`，controls 也很少，而且 effect 经常是 `0.0`。

这说明：

```text
state bucket 可以事后把少数好动作隔开；
但它没有产生可泛化机制。
```

用数学语言说，P1 找到的是：

$$
P(Y=1\mid bucket) \text{ 在极小样本 bucket 上看起来高}
$$

但我们需要的是：

$$
P(Y=1\mid z_{state}, z_{action}) \text{ 在新动作和新训练状态上仍稳定。}
$$

现在还没有。

### 1.2 P2/P4：GoodSubspace / GoodCone 失败说明“落在子空间里”不是好动作充分条件

P2 中 GoodCone 的 basis rank 是 `16`，解释方差接近 `1.0`，good coverage 也是 `1.0`，但 TopK32 precision 只有 `0.03125`，V240 为负，longrisk 接近 `0.968`。

P4 更关键：FastSlow coverage = `1.0`，Bad coverage = `1.0`，risk-safe basis subset dim = `0`。

这说明当前 basis 不是没覆盖好动作，而是同时覆盖了好动作和坏动作。真正的问题不是 coverage，而是方向、符号、幅度、状态相位和风险边界。

所以不能再用：

$$
\Delta \in \mathcal{S}_{good}
$$

作为好动作判据。更合理的是：

$$
\Delta \in \mathcal{C}_{good}(s_t),
$$

其中 $\mathcal{C}_{good}(s_t)$ 是依赖当前训练状态 $s_t$ 的局部可控 cone 或非线性区域。

### 1.3 P3：soft Pareto 有一点信号，但不是 controller

P3 中某些组合能把 TopK32 FastSlow precision 提到 `0.4375`，V240 变成正数 `0.0753`，但 longrisk 仍有约 `0.107`，没有过 weak/strong。

这说明：

```text
value-risk tradeoff 不是完全没结构；
但 hard threshold / alpha-lambda sweep 还不能产生安全可部署区域。
```

P3 的正确用途不是继续调 alpha，而是提示下一步必须做 constrained optimization，而不是手工加权：

$$
\max_\Delta FutureValue(\Delta)
$$

subject to：

$$
LongRisk(\Delta)\le \tau_L,
$$

$$
MemoryHarm(\Delta)\le \tau_M,
$$

$$
OffdiagRisk(\Delta)\le \tau_O.
$$

### 1.4 P5：mini-unroll 没有未来 adjoint 信号

P5 中 k=3 的 AUC SlowBurn vs Risky 可以到 `0.744`，但 TopK32 FastSlow precision 只有 `0.125`，V240 为负，longrisk 很高。partial corr $V240|V1$ 多数接近 0 或负数。

这说明短 mini-unroll 当前形式没有测到真正的未来路径传播算子。它可能仍然只是在测：

```text
当前响应；
局部梯度一致性；
短期风险变化；
而不是 h20/h80/h240 的未来路径收益。
```

---

## 2. 当前项目最本质的 blocker

现在 blocker 已经不是：

```text
没有 outcome table；
没有 action apply；
没有 branch-horizon materializer；
没有 future path rows；
没有 natural generator；
没有 GoodSubspace；
没有 toy mechanism。
```

这些都已经在前序版本里不同程度解决或证伪。

真正的 blocker 是：

$$
\boxed{
\text{当前 update 表示无法区分“同一局部子空间里的好方向”和坏方向。}
}
$$

更具体地说：

```text
1. 好动作和坏动作共享高维子空间；
2. 好路径动作极稀疏；
3. 当前 FPO / FPAU / GoodCone 都没有捕捉未来动力学；
4. memory/offdiag 风险边界不是简单 hard projection；
5. generated route 没有可用的目标函数；
6. 自然动作池已经不能作为主希望。
```

这意味着下一步不能再“猜一个动作 family”。必须先主动测量当前训练状态附近的局部未来响应。

---

## 3. v10.6 总体目标

v10.6 的目标是建立 **Local Future Response Operator**，简称 LFRO。

它要回答：

$$
\boxed{
\text{在当前模型状态 }s_t\text{ 下，一个小扰动 }\Delta\text{ 会如何影响后续训练路径？}
}
$$

不再先假设 action family 好坏，而是用 controlled perturbation 做系统辨识。

对于当前参数 $\theta_t$ 和 base optimizer 后续训练算子 $U_h$，定义：

$$
R_h(\Delta)=M_h(U_h(\theta_t))-M_h(U_h(\theta_t+\Delta)),
$$

其中 $M_h$ 是 h-step 后的路径指标，例如 V、V240、longrisk、memory/offdiag、bad/null、hard-tail。

我们不直接在全参数空间搜索，而是在低维基 $B$ 中写：

$$
\Delta=B\alpha.
$$

v10.6 要做的是通过主动扰动设计估计：

$$
R_h(B\alpha)\approx c_h+g_h^\top\alpha+\alpha^\top H_h\alpha.
$$

然后解：

$$
\alpha^*=
\arg\max_\alpha
\sum_{h\in\{20,80,240\}} w_h R_h(B\alpha)
$$

subject to：

$$
LongRisk(B\alpha)\le \tau_L,
$$

$$
MemoryHarm(B\alpha)\le \tau_M,
$$

$$
OffdiagRisk(B\alpha)\le \tau_O,
$$

$$
\|\alpha\|\le \epsilon.
$$

这一步的核心不是“换一个 score”，而是从当前状态主动估计局部动力学。

---

## 4. v10.6 四线重排

### A 线：Path Type 固化与局部响应目标

A 线不再只统计 AUV / RAUV，而要固定 path type，并为 LFRO 提供目标。

路径类型如下：

```text
FastGood:
  h1 好，h20/h80/h240 也好，longrisk/memory/offdiag 低。

SlowBurnGood:
  h1 不强甚至负，但 h20/h80/h240 转好，longrisk/memory/offdiag 低。

RiskyHighAUV:
  AUV 或 V240 高，但 longrisk/memory/offdiag 高。

SafeLowValue:
  longrisk/memory/offdiag 低，但 V / V240 不足。

BadPath:
  value 低或 risk 高。
```

A 线要把这些路径类型转成 LFRO 的监督目标，不再只用二元 FastSlow。

### B 线：从 FPO 分类器改成 LFRO 系统辨识

B 线不再训练一个 FPO score。B 线做的是：选择一组低维方向 $B$，主动采样扰动 $\alpha$，真实测量短/中/长 horizon 响应，然后估计局部响应算子。

方向基 $B$ 初始包含：

```text
b1: AdamW update direction;
b2: momentum direction;
b3: current gradient direction;
b4: hard-tail gradient direction;
b5: old-family preserving gradient direction;
b6: memory buffer gradient direction;
b7: offdiag-risk relief direction;
b8: exact-transfer residual direction;
b9-b16: low-rank random/Krylov/JVP directions;
b17-b24: layer/block-local directions;
b25-b32: generated prior directions from previous FPAU/GUP families, only as basis not as action family.
```

判断不是看某个 feature AUC，而是看局部模型能否预测 holdout perturbations 的 future path。

### C 线：natural harvesting 降级为 reference

C 线不再作为主线，不再继续 20000 / 50000 panel。它只保留：

```text
1. reference distribution;
2. sanity audit;
3. 如果 LFRO 产生新动作，检查它是否远离 natural distribution；
4. 如果远离，要判断是好事还是 OOD risk。
```

### D 线：generated route 改成 solved-update discovery

D 线不再生成 APG/GUP family。D 线只接收 B 线解出来的 $\Delta=B\alpha^*$，然后做：

```text
4-action fail-fast；
8-action discovery；
16-action verification；
32-action stress；
64-action boundary。
```

如果 4-action 全是 BadPath，就立即回到 B 线，不扩大。

---

# 5. v10.6 完整实验计划

## P0. Boundary Lock 与 artifact audit

### 目标

确认 v10.5 的边界稳定，不重新争论旧路线。

### 必须记录

```text
source_route_v1050
system_legal_controller_pass_v1050
generated_route_status_v1050
P1/P2/P3/P4/P5/P7 pass flags
no_fake/proxy/cpu flags
known_action_count
Fast/Slow/Risky/SafeLow/Bad counts
```

### 判断标准

P0 pass：

```text
source route reproduced = CaseD-GoodConeAndUnrollBothFailRepresentationInsufficient
no_fake/proxy/cpu = 0 / 0 / 0
P8/P9/P10 not_run confirmed
```

### 不满足时 Codex 先尝试

```text
如果 artifact missing：先补 manifest path，不重跑 science；
如果 route mismatch：检查是否读取了旧 v10.4/v10.3 文件；
如果 fake/proxy/cpu 非 0：停止后续，修 contract audit。
```

---

## P1. Path Type 再定义与稳定性审计

### 目标

把 FastGood / SlowBurnGood / RiskyHighAUV / SafeLowValue / BadPath 固定为 LFRO 的监督目标。

### 必须记录

每个 action 记录：

```text
action_id
dataset / family / template / state_bucket
V1 / V5 / V20 / V80 / V240
RAUV / AUV
longrisk
bad / null
memory_fail / offdiag_fail if available
path_type
path_type_reason
```

### 判断标准

P1 discovery pass：

```text
FastGood + SlowBurnGood >= 32
SlowBurnGood >= 16
RiskyHighAUV >= 64
BadPath >= 256
path_type exclusivity violation = 0
```

P1 strong pass：

```text
FastGood + SlowBurnGood >= 64
SlowBurnGood >= 32
path types cover all evaluated rows
leave-family/template distribution not single-pocket
```

### 可视化

```text
fig_p1_path_type_counts.svg
fig_p1_V_by_horizon_path_type.svg
fig_p1_risk_vs_RAUV_path_type.svg
fig_p1_slowburn_examples.svg
fig_p1_dataset_family_template_distribution.svg
```

### 不满足时 Codex 先尝试

```text
如果 SlowBurnGood 太少：
  放宽 V1 条件，但不能放宽 h240 risk / memory / offdiag；
  单独从 OldOnly / Core77 / RiskCleanButLowValue 补样本；
  不要用 dataset 条件调标签。

如果 RiskyHighAUV 和 BadPath 边界混乱：
  强化 longrisk / memory / offdiag hard definition；
  不调 AUV threshold 逃避风险。

如果 path_type overlap：
  按优先级 Risky > Bad > Fast/Slow > SafeLow 重排，并记录冲突原因。
```

---

## P2. Local Future Response Operator Materializer

### 目标

主动构造小扰动，估计当前训练状态下未来路径响应，而不是被动分类已有 action。

### 设计

选择 $S$ 个训练状态，每个状态构造 $K$ 个 basis direction：

```text
S = 8 states for preflight；
K = 16 basis directions for P2a；
K = 32 basis directions for P2b if P2a passes；
perturbation design = signed Hadamard / random orthogonal alpha；
alpha count = 2K + 8 validation perturbations；
horizons = 1, 5, 20 for preflight；h80/h240 only for candidate solutions。
```

### 必须记录

```text
state_id
basis_id
basis_type
alpha_id
alpha_vector
perturb_norm
apply_linf
horizon
V_h
longrisk_h
bad_h
null_h
memory/offdiag if available
runtime cost
NaN/Inf/apply error
```

### 判断标准

P2 materializer pass：

```text
all planned perturbation rows materialized
apply_linf finite and nonzero for nonzero alpha
NaN/Inf = 0
no fake/proxy/cpu = 0
cost per perturbation recorded
```

P2 weak pass：

```text
At least 1 state has >= 1 perturbation with V20 LCB > 0 and longrisk UCB <= 0.10
```

P2 strong pass：

```text
At least 3 states have >= 1 perturbation with V20 LCB > 0 and longrisk UCB <= 0.05
```

### 可视化

```text
fig_p2_response_matrix_heatmap.svg
fig_p2_basis_type_value_risk.svg
fig_p2_alpha_norm_vs_future_value.svg
fig_p2_perturbation_materializer_cost.svg
```

### 不满足时 Codex 先尝试

```text
如果 apply_linf = 0：检查 basis direction 是否被 zeroed / mask 错；
如果 NaN/Inf：缩小 perturbation norm，检查 optimizer state copy；
如果 cost 太高：减少 states，先只跑 h1/h5/h20；
如果所有 perturbation 都 BadPath：检查 basis 是否只包含 current response directions，加入 memory-safe / hard-tail / random orthogonal basis。
```

---

## P3. LFRO 拟合与可控性测试

### 目标

估计：

$$
R_h(B\alpha)\approx c_h+g_h^\top\alpha+\alpha^\top H_h\alpha
$$

并判断是否存在可控的 future-good direction。

### 方法

```text
Fit linear model;
Fit ridge quadratic model;
Fit robust Huber model;
Fit monotone constrained value-risk model;
Hold out validation perturbations;
Compare predicted vs actual V20/V80/longrisk.
```

### 必须记录

```text
model_id
state_id
basis_count
train_R2_V20 / validation_R2_V20
train_R2_longrisk / validation_R2_longrisk
rank_of_response_operator
condition_number
best_predicted_alpha
predicted V20/V80/V240
predicted longrisk
actual validation V20/longrisk if available
```

### 判断标准

P3 weak pass：

```text
validation_R2_V20 >= 0.20
and validation_R2_longrisk >= 0.20
and predicted best alpha has actual V20 LCB > 0 on validation.
```

P3 strong pass：

```text
validation_R2_V20 >= 0.40
validation_R2_longrisk >= 0.40
rank_of_response_operator >= 3
at least one alpha satisfies value-positive and risk-safe on heldout perturbations.
```

### 可视化

```text
fig_p3_pred_vs_actual_V20.svg
fig_p3_pred_vs_actual_longrisk.svg
fig_p3_response_operator_spectrum.svg
fig_p3_feasible_region_value_risk.svg
```

### 不满足时 Codex 先尝试

```text
如果 R2 低：增加 orthogonal alpha samples；
如果 condition number 高：orthogonalize basis，remove near-duplicate directions；
如果 value predictable but risk not：增加 memory/offdiag/hard-tail basis and risk labels；
如果 risk predictable but value not：basis lacks value direction，add gradient/momentum/Krylov/function-space directions；
如果 all predicted good alpha fail validation：use nonlinear/interaction terms, but do not open controller.
```

---

## P4. Constrained Update Solve

### 目标

用 LFRO 解出 $\alpha^*$，而不是猜 generated family。

### 求解问题

$$
\max_\alpha
\quad
w_{20}R_{20}(B\alpha)+w_{80}R_{80}(B\alpha)+w_{240}R_{240}(B\alpha)
$$

subject to：

$$
\widehat{LongRisk}(B\alpha)\le \tau_L,
$$

$$
\widehat{MemoryHarm}(B\alpha)\le \tau_M,
$$

$$
\widehat{OffdiagRisk}(B\alpha)\le \tau_O,
$$

$$
\|\alpha\|_2\le \epsilon.
$$

### 候选 solver

```text
S1 linear constrained solve;
S2 quadratic trust-region solve;
S3 cross-entropy method over alpha;
S4 Pareto frontier selection;
S5 robust worst-state alpha;
S6 state-specific alpha with shared basis.
```

### 必须记录

```text
solver_id
state_id
alpha_solution
basis_coefficients
predicted V20/V80/V240
predicted longrisk/memory/offdiag
actual branch-horizon V1/V5/V20/V80/V240
actual path_type
apply_linf
cost
```

### 判断标准

P4 weak pass：

```text
Among 4 solved updates, at least 1 FastGood or SlowBurnGood;
longrisk_created_rate <= 0.25;
V240 LCB not significantly negative.
```

P4 strong pass：

```text
Among 8 solved updates, FastGood + SlowBurnGood >= 2;
V240 LCB > 0;
longrisk UCB <= 0.10;
memory/offdiag fail UCB <= 0.10 if available.
```

### 可视化

```text
fig_p4_alpha_coefficients.svg
fig_p4_solution_future_path.svg
fig_p4_pred_actual_solution.svg
fig_p4_solver_comparison.svg
```

### 不满足时 Codex 先尝试

```text
如果 solver predicts good but actual BadPath：model misspecified，return to P3 with interaction terms；
如果 all solutions low-risk low-value：risk constraints too hard or basis lacks value direction；add value basis and rerun P2/P3；
如果 all solutions high-risk：risk operator not fitted; add risk labels or hard-tail memory directions；
如果 alpha dominated by one basis：check basis scale normalization；
如果 apply_linf too large：reduce trust radius epsilon.
```

---

## P5. Discovery Staircase

### 目标

只在 P4 有弱信号时进行 generated discovery，避免盲目扩大。

### 设计

```text
Stage4: 4 solved updates；
Stage8: only if Stage4 has >= 1 Fast/Slow-like；
Stage16: only if Stage8 weak pass；
Stage32: only if Stage16 weak pass；
Stage64: only if Stage32 weak pass。
```

### 必须记录

```text
stage
n_actions
branch_rows
FastGood_count
SlowBurnGood_count
RiskyHighAUV_count
SafeLowValue_count
BadPath_count
FastSlow precision
V20/V80/V240 LCB
RAUV LCB
longrisk UCB
bad/null UCB
memory/offdiag UCB if available
```

### 判断标准

Stage8 weak：

```text
FastGood + SlowBurnGood >= 1
V240 LCB > -0.10
longrisk UCB <= 0.50
```

Stage32 weak：

```text
FastGood + SlowBurnGood >= 4
FastSlow precision >= 0.125
V240 LCB > 0
longrisk UCB <= 0.20
```

Stage64 strong：

```text
FastGood + SlowBurnGood >= 12
FastSlow precision >= 0.1875
V240 LCB > 0
longrisk UCB <= 0.10
bad/null UCB <= 0.10 / 0.15
```

### 不满足时 Codex 先尝试

```text
Stage4 all BadPath:
  stop family; inspect basis value direction; do not scale to 8.

Stage8 has SafeLowValue only:
  check delayed horizon; maybe add h240+ diagnostic, but do not call pass.

Stage8 high RiskyHighAUV:
  add hard memory/offdiag/risk gate to solver, not to output label.

Stage32 precision low but some SlowBurn:
  cluster successes; use them to add new basis to P2.
```

---

## P6. Failure Attribution

### 目标

如果 v10.6 仍失败，必须知道失败在哪个层级。

### failure classes

```text
F1: path label too sparse even after P1;
F2: basis has no value direction;
F3: basis has value direction but risk constraints kill it;
F4: LFRO cannot predict heldout perturbations;
F5: solver exploits model error;
F6: solved update apply/replay bug;
F7: generated actions all BadPath;
F8: SlowBurn exists but cannot be predicted at commit time;
F9: runtime cost too high;
F10: base architecture / representation insufficient.
```

### 判断标准

P6 pass：

```text
assigned_fraction >= 0.95
unknown_fraction <= 0.05
each failure class has evidence rows
recommended_fix is concrete
```

### 不满足时 Codex 先尝试

```text
如果 unknown high：add per-action diagnostic fields;
如果 multiple causes overlap：assign primary by earliest violated gate;
如果 failure class repeats from v10.5/v10.4：stop repeating that family.
```

---

## P7. Controller Boundary

### 目标

只有 discovery 出现强信号，才允许 controller boundary。

### 打开条件

```text
P5 Stage64 strong pass = 1
or
P5 Stage32 weak pass + P3 strong pass + P4 strong pass = 1
```

### controller candidate

```text
controller = solved-update rule;
not existing-action selector;
not GoodCone threshold;
not FPO scalar threshold.
```

### 判断标准

Controller pass：

```text
accepted_count >= 87
FastSlow precision >= 0.75 or GradeAB precision >= 0.75
V240 LCB > 0
RAUV LCB > 0
longrisk UCB <= 0.05
bad UCB <= 0.05
null UCB <= 0.15
memory/offdiag UCB <= 0.05 if available
LDO/LSO/LTO/LFO <= 0.10
no dataset-specific branch
```

### 不满足时 Codex 先尝试

```text
If accepted_count < 87 but quality high:
  do not lower quality gate; increase action source through LFRO states.

If leaveout fail:
  diagnose state/family/template shift; do not add dataset threshold.

If longrisk fail:
  adjust solver constraints and rerun discovery, not controller threshold.
```

---

## P8. Runtime Boundary

### 打开条件

```text
P7 controller pass = 1
```

### 指标

```text
step_ratio_q50/q90/q95
controller cost q90
payload apply cost q90
kernel launches
sync count
active-step fraction
memory peak
```

### 判断标准

```text
step_ratio_q90 <= 1.50
no CPU offload
no materializer in timed path
no fake/proxy rows
```

---

## P9. Paired Replay / Short-Full Boundary

### 打开条件

```text
P7 controller pass = 1
P8 runtime pass = 1
```

### 必须比较

```text
RealFunctional
AdamWParallel
bestLR
NoOp
Random
shuffled payload
```

### 判断标准

```text
RealFunctional beats AdamWParallel / bestLR / NoOp / Random
shuffled payload fails
leave-dataset-out stable
leave-stratum-out stable
short-run positive
full-run no catastrophic forgetting
```

---

## 6. 最终路线判断

v10.6 的最重要问题不是：

```text
GoodCone 能不能调好？
FPO 能不能调好？
GUP 能不能调好？
```

而是：

$$
\boxed{
\text{当前训练状态附近是否存在可测、可预测、可求解的 future-path-improving perturbation？}
}
$$

如果 v10.6 证明存在 LFRO 且能解出 FastGood / SlowBurnGood，那么项目重新进入 functional controller 路线。

如果 v10.6 证明不存在，或者当前 basis / architecture 无法控制 future path，那么下一步应停止 functional update controller 路线，转向：

```text
1. base architecture / compositional FullEdge trainability；
2. optimizer state redesign；
3. function-space natural-gradient / K-FAC-like update；
4. larger structural primitive rather than local action primitive。
```

这才是本轮真正的决策边界。
