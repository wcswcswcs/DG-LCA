# DG-KAN v10.7 Controllability Reset：深度反思与完整实验计划

> 目标：不再继续小修 FPO、GoodCone、FPAU、LFRO 或 generated family，而是判断当前系统在局部更新空间里是否真的具备 **future-path controllability**。如果当前参数扰动空间本身不能产生 risk-safe positive future response，那么继续做 selector / proxy / generator 都只是绕圈。

---

## 0. 当前边界的直接判断

v10.6 的结果很清楚：Local Future Response Operator 真实跑了，但没有给出可用机制。

本轮已经完成真实 materialization：

```text
states = 8
actions = 320
branch rows = 5760
P2_materializer_pass = 1
```

但 scientific gate 没过：

```text
P2 weak / strong = 0 / 0
P3 weak / strong = 0 / 0
P4 weak / strong = 0 / 0
P5 Stage4/8/16/32/64 opened = 0 / 0 / 0 / 0 / 0
```

最关键的负结果是：

```text
P3 best model = LFRO-L3-robust-huber-clipped
validation R2 V20 = -0.6366622673887112
validation R2 risk = -0.9202397748287701

P4 all8 FastSlow precision = 0.0
P4 all8 V240 = -0.8158580619097676
P4 all8 longrisk = 1.0
```

这说明当前 LFRO 不是“差一点”。它没有学到一个能预测未来路径收益的局部响应面。更严重的是，求解出的 all8 动作不但没有 Fast/Slow，V240 还为负，longrisk 直接到 1.0。

因此本轮后我认为旧式路线必须停止：

```text
不再继续调 FPO / GoodCone / LFRO 阈值；
不再继续扩大 generated family；
不再继续把当前 action representation 当成足够表达好更新的空间；
不再把 toy pass 当真实机制；
不再把“不能 official”作为“不能 discovery”的理由。
```

---

## 1. 新的核心判断

当前最准确的判断是：

$$
\boxed{
\text{future-path 好动作可信，但当前局部更新表示没有暴露 risk-safe positive future response。}
}
$$

这句话比“LFRO 失败”更重要。

过去几轮已经排除了这些假设：

```text
1. 好动作不只是 current loss descent；exact one-step transfer 测得很准，但选不出好动作。
2. 好动作不只是 GoodSubspace；GoodSubspace 覆盖好动作，也覆盖坏动作。
3. 好动作不只是 GoodCone；GoodCone TopK precision 很低且 high-risk。
4. 好动作不只是 mini-unroll adjoint；mini-unroll 没有 future signal。
5. 好动作不只是 local response surface；LFRO validation R2 为负，solver 产生 high-risk BadPath。
6. 好动作也不能靠 natural AP0 harvesting 获得足够密度；natural density 已经降级为 reference。
```

所以当前必须回答更底层的问题：

$$
\boxed{
\text{在当前模型状态和当前允许更新空间中，是否存在任何可控的未来好路径方向？}
}
$$

如果不存在，继续设计 selector 是无意义的。selector 只能挑已有空间里的动作；如果空间本身没有 value direction，selector 再强也没用。

---

## 2. v10.7 总体实验目标

v10.7 的总体目标不是生成一个 controller，而是做一个硬判断：

$$
\boxed{
\text{当前 DG-KAN base 是否具有 local future-path controllability？}
}
$$

更具体地说，本轮要判断四件事：

```text
Q1. 当前低维参数扰动 basis 里，是否存在 risk-safe positive future response？
Q2. 如果存在，它是在哪些 basis family、哪些状态相位、哪些 norm scale 中出现？
Q3. 如果不存在，是否是 basis 不够、风险约束过硬、path label 太稀、还是 base architecture 本身不可控？
Q4. 是否应从参数扰动转向 optimizer-state update 或 architecture / compositional FullEdge trainability？
```

本轮不追求 official controller。官方 controller 只有在 discovery 明确出现足够 FastGood / SlowBurnGood 后才打开。

---

## 3. 新假设

### H1：当前 action representation 不足

当前已有证据表明 GoodSubspace / GoodCone / LFRO 都不能分离好坏路径。因此第一个假设是：

$$
H_1:
\text{当前 action representation 不包含足够可控的 future-path value direction。}
$$

判断标准：如果扩展 basis 后仍然找不到 risk-safe positive response，则支持 $H_1$。

### H2：好更新是 state-conditioned cone，不是固定子空间

好动作可能只在某些训练状态相位、某些 norm scale、某些 optimizer-state 条件下成立：

$$
H_2:
\text{GoodUpdate}(\Delta, s_t)
\neq
\text{GoodUpdate}(\Delta)
$$

其中 $s_t$ 包括 current batch、memory buffer、hard-tail、old-family、optimizer moments、local curvature 和 edge/basis activation state。

### H3：value direction 与 memory/offdiag safety 不是硬投影关系

过去 hard memory/offdiag projection 常常保住风险但毁掉 value。因此假设：

$$
H_3:
\text{value direction 与 safety direction 需要 Pareto / trust-region 处理，而不是 hard projection。}
$$

### H4：若局部参数扰动不可控，应转向 optimizer-state 或 architecture line

如果参数扰动 basis 不存在安全正响应，则 functional update 可能不应该是直接改参数，而应改：

```text
optimizer state；
preconditioner；
edge/basis activation routing；
compositional FullEdge architecture；
training schedule 的 local boundary condition。
```

---

## 4. Runner 组织方式重置

从 v10.7 开始，必须拆成三类 runner。

### 4.1 Engineering Gate Runner

只检查：

```text
payload 是否能 apply；
branch-horizon rows 是否完整；
NaN/Inf 是否为 0；
runtime 是否能测；
CUDA / no CPU offload 是否合规。
```

Engineering runner 不写科学结论。

### 4.2 Science Discovery Runner

允许非 official 探索：

```text
basis exploration；
local response mapping；
optimizer-state update；
small generated sandbox；
path-type mechanism audit。
```

Discovery runner 的输出必须标红：

```text
not controller；
not official pass；
not runtime pass；
not paired replay。
```

### 4.3 Official Gate Runner

只有 discovery 出现硬信号后才打开：

```text
accepted_count >= 87；
precision / V / V240 / longrisk / bad / null / memory / offdiag 同时过线；
LDO / LSO / LTO / LFO 稳定；
selected runtime step_ratio_q90 <= 1.50；
paired replay beats controls。
```

---

## 5. P0：边界复现与停止旧路线

### 目标

复现 v10.6 边界，确认本轮不再小修 LFRO-L3、FPO13、GoodCone 或 FPAU-A7。

### 记录指标

```text
v10.6 route
P1 Fast / Slow / Risky / SafeLow / Bad
P2 states/actions/branch rows
P3 best LFRO model and validation R2
P4 all8 FastSlow precision / V240 / longrisk
P5 opened stages
no fake / proxy / cpu audit
```

### 判断标准

如果以下事实成立，则旧路线停止：

```text
P3 validation R2 V20 < 0；
P3 validation R2 risk < 0；
P4 all8 FastSlow precision = 0；
P4 all8 longrisk >= 0.50；
P5 no generated stage opened。
```

### Codex 不满足时先尝试

如果 P0 复现失败：

```text
先检查 source artifact path；
再检查 v10.6 manifest 是否读取错；
再检查 path-type label 口径是否改变；
不允许用新结果覆盖旧 v10.6 boundary。
```

---

## 6. P1：Path-type label 固化与连续目标替代

### 目标

固定 path type，但不把 path type 当唯一训练目标。由于 Fast/Slow 太稀疏，本轮同时建立连续 response 向量。

### 记录指标

每个 action 记录：

```text
action_id
state_id
dataset / family / template / step_bucket
V1 / V5 / V20 / V80 / V240
RAUV
longrisk
bad / null
memory_fail / offdiag_fail
path_type
norm / payload norm / AdamW alignment
```

定义：

```text
FastGood:
  V1 LCB > 0，V20/V80/V240 LCB > 0，risk safe。

SlowBurnGood:
  V1 <= 0 或弱，V20/V80/V240 LCB > 0，risk safe。

RiskyHighAUV:
  RAUV > 0，但 longrisk 或 memory/offdiag fail 高。

SafeLowValue:
  longrisk/memory/offdiag safe，但 V20/V80/V240 不够正。

BadPath:
  value negative 或 risk high。
```

连续 response 向量：

$$
Y(\Delta)=
(V_1,V_5,V_{20},V_{80},V_{240},RAUV,LongRisk,MemoryFail,OffdiagFail,Bad,Null)
$$

### 判断标准

P1 pass 不要求 Fast/Slow 数量很高。它要求标签口径稳定：

```text
path_type mutually exclusive = 1；
missing horizon = 0；
label NaN/Inf = 0；
SlowBurnGood risk UCB <= 0.05；
RiskyHighAUV longrisk or memory/offdiag clearly high。
```

### 可视化

```text
1. path type count bar；
2. V1/V5/V20/V80/V240 path curves；
3. RAUV vs longrisk scatter；
4. SlowBurnGood vs RiskyHighAUV memory/offdiag heatmap；
5. path type by state/family/template distribution。
```

### Codex 不满足时先尝试

```text
如果 SlowBurnGood 太少：
  只放宽 V1 条件，不放宽 h240 risk；
  合并相邻 horizon 的 V20/V80 条件；
  不把 RiskyHighAUV 合并进 SlowBurnGood。

如果 RiskyHighAUV 太多：
  强化 risk hard label；
  不调 RAUV threshold 来掩盖。

如果 SafeLowValue 太多：
  检查是否存在 delayed value at h240；
  必要时增加 h480 diagnostic，但不写 official。
```

---

## 7. P2：Controllability Basis Expansion

### 目标

不再只用旧 action representation。主动构造一组更广的低维 basis，判断当前系统附近是否存在任何 risk-safe positive future response。

### 更新表示

令：

$$
\Delta = B\alpha
$$

其中 $B$ 是不同 basis family 的集合。

### Basis families

本轮并行构造：

```text
B0: AdamW direction；
B1: momentum direction；
B2: memory-safe projected gradient；
B3: hard-tail correction direction；
B4: old-family preserving direction；
B5: signal-channel / SNR direction；
B6: low-rank random orthogonal directions；
B7: blockwise edge/basis directions；
B8: optimizer-state perturbation direction；
B9: curvature / trust-region scaled directions。
```

每个 state 至少采样：

```text
positive / negative sign；
small / medium norm；
with and without memory/offdiag soft penalty。
```

### 记录指标

```text
state_id
basis_family
basis_id
alpha_norm
sign
AdamW_alignment
memory_alignment
offdiag_proxy
V1/V5/V20/V80/V240
RAUV
longrisk
bad/null
memory/offdiag fail
path_type
cost
```

### 判断标准

P2 weak pass：

```text
至少 3 个 state 出现 risk-safe positive future response；
或整体 risk-safe positive rate >= 0.03；
且这些 response 不是单一 dataset / family / template。
```

P2 strong pass：

```text
risk-safe positive rate >= 0.05；
V240 LCB > 0；
longrisk UCB <= 0.05；
memory/offdiag UCB <= 0.05；
至少 2 个 basis family 有 positive response。
```

### 可视化

```text
1. basis_family × path_type heatmap；
2. alpha_norm vs V240 / longrisk curves；
3. value-risk Pareto scatter；
4. state-conditioned response heatmap；
5. sign symmetry plot。
```

### Codex 不满足时先尝试

```text
如果所有 basis 都 BadPath：
  先检查 alpha norm 是否过大；
  再降低 norm；
  再加入 negative sign；
  再检查 payload apply 是否正确；
  不扩大 generated family。

如果 value 正但 risk 高：
  不直接 hard project；
  尝试 soft trust-region penalty；
  记录 value-risk Pareto。

如果 risk safe 但 value 低：
  先检查是否 SafeLowValue / delayed gain；
  再看 h80/h240 是否晚起；
  不把它写成 Fast/Slow success。

如果只有一个 state 有 positive response：
  标成 state-pocket diagnostic；
  不进入 controller。
```

---

## 8. P3：Risk-Value Feasibility 与 Pareto Frontier

### 目标

判断 value direction 和 risk safety 是否可兼容。过去 hard projection 常常把 value 投掉，本轮不再使用单一硬 veto。

### 公式

对每个候选更新：

$$
Score(\Delta)=V_{future}(\Delta)
$$

但 safety 不加权成一个分数，而是作为 Pareto 维度：

$$
(V_{future}, LongRisk, MemoryHarm, OffdiagRisk, Cost)
$$

### 记录指标

```text
Pareto efficient count
best V under longrisk <= 0.05
best V under memory/offdiag <= 0.05
best V under all safety gates
value lost by hard projection
value retained by soft trust region
```

### 判断标准

P3 weak pass：

```text
存在至少 8 个 Pareto-efficient candidate；
其中至少 2 个满足 longrisk <= 0.05 且 V240 LCB > 0。
```

P3 strong pass：

```text
存在 >= 16 个 risk-safe positive candidates；
hard projection value loss < 0.30 或 soft trust region retention >= 0.70；
Fast/Slow precision among Pareto candidates >= 0.50。
```

### 可视化

```text
1. value-risk Pareto frontier；
2. hard projection vs soft trust retention bar；
3. memory/offdiag risk vs V240 scatter；
4. alpha norm vs Pareto membership。
```

### Codex 不满足时先尝试

```text
如果 hard projection 毁掉 value：
  用 soft trust-region；
  尝试 barrier penalty；
  不继续 hard veto。

如果 no risk-safe positive：
  回到 P2 扩展 basis；
  不拟合 response operator。

如果 Pareto 只在 high norm 出现：
  检查是否 payload instability；
  不能直接扩大 norm 写 success。
```

---

## 9. P4：Local Future Response Operator 只在 P2/P3 过线后拟合

### 目标

过去 v10.6 的 LFRO 在没有足够 positive response 的情况下拟合，R2 为负。v10.7 只有在 P2/P3 确认有可控 response 后才拟合 LFRO。

### 模型

在低维 $\alpha$ 上拟合：

$$
R_h(B\alpha) \approx c_h + g_h^\top \alpha + \alpha^\top H_h\alpha
$$

分别拟合：

```text
V20
V80
V240
RAUV
longrisk
memory_fail
offdiag_fail
```

### 记录指标

```text
train/validation split by state
R2_V20 / R2_V80 / R2_V240
risk AUC
sign match
calibration by path_type
solver predicted best vs actual branch-horizon
```

### 判断标准

P4 weak pass：

```text
validation R2_V240 >= 0.20；
risk AUC >= 0.75；
sign match >= 0.65；
solver top4 actual has at least 1 Fast/Slow-like candidate。
```

P4 strong pass：

```text
validation R2_V240 >= 0.35；
risk AUC >= 0.85；
solver top8 Fast/Slow precision >= 0.25；
longrisk UCB <= 0.10。
```

### Codex 不满足时先尝试

```text
如果 R2 < 0：
  不再求解；
  检查 basis condition number；
  检查 path labels是否极端稀疏；
  回到 P2/P3。

如果 value R2 好但 risk AUC 差：
  不允许 controller；
  增加 risk samples；
  用 separate risk model。

如果 solver exploit model error：
  加 trust region；
  限制 alpha norm；
  使用 adversarial validation set。
```

---

## 10. P5：Optimizer-State Update Discovery

### 目标

如果直接参数扰动不可控，尝试不直接改参数，而是改 optimizer state 或 preconditioner。

### 候选 update

```text
O1: momentum damp / amplify；
O2: second-moment trust scaling；
O3: memory-safe preconditioner；
O4: signal-channel SNR preconditioner；
O5: blockwise orthogonalized update；
O6: hard-tail local step rescale；
O7: old-family preserving update state correction。
```

### 记录指标

```text
optimizer_state_delta_norm
parameter_delta_induced_norm
V1/V20/V80/V240
RAUV
longrisk
memory/offdiag
path_type
runtime cost
```

### 判断标准

P5 weak pass：

```text
Stage4 中至少 1 个 risk-safe positive candidate；
longrisk UCB <= 0.10；
V240 LCB > 0。
```

P5 strong pass：

```text
Stage8 Fast/Slow precision >= 0.25；
longrisk UCB <= 0.05；
memory/offdiag UCB <= 0.05。
```

### Codex 不满足时先尝试

```text
如果 optimizer-state update 无效：
  检查是否实际参数 delta 太小；
  检查 AdamW state 是否被 reset；
  检查 state update 是否在 branch horizon 生效。

如果 value 正但风险高：
  加 trust scaling；
  不把它写成 success。

如果全 BadPath：
  停止 optimizer-state family；
  转向 architecture line。
```

---

## 11. P6：Architecture / Compositional Trainability Bridge

### 目标

如果 P2/P5 都说明当前 base 不可控，则必须判断是不是 base architecture 的表达/几何结构不适合 functional update。

### 本轮只做 bridge，不做 full architecture campaign

记录：

```text
current base future response rate
compositional candidate synthetic interaction pass
kernel-native feasibility
P4 wall-clock ratio
P5 trainability near-pass
```

### 判断标准

P6 bridge pass：

```text
当前 base no controllability；
但 compositional / architecture candidate 在 synthetic 或 limited real setting 有更高 risk-safe positive response。
```

### Codex 不满足时先尝试

```text
如果 compositional candidate wall-clock fail：
  先做 fused / materialization-free path；
  不进入 functional update。

如果 synthetic pass but real fail：
  做 synthetic-to-real gap audit；
  不把 synthetic pass 写成 functional success。
```

---

## 12. P7：Generated Discovery Staircase

### 目标

只有 P2/P3/P4/P5 至少一条有 weak signal，才允许 generated discovery。它不是 official route，只是发现机制。

### 阶梯

```text
Stage4 -> Stage8 -> Stage16 -> Stage32 -> Stage64
```

每一级都必须真实 branch-horizon。

### 判断标准

Stage4 打开条件：

```text
P2/P3/P4/P5 任一 weak pass。
```

Stage8 打开条件：

```text
Stage4 至少 1 个 Fast/Slow-like；
longrisk_created_rate <= 0.25。
```

Stage16 打开条件：

```text
Stage8 Fast/Slow precision >= 0.125；
V240 LCB > -0.10；
longrisk UCB <= 0.25。
```

Stage32 打开条件：

```text
Stage16 Fast/Slow precision >= 0.20；
V240 LCB > 0；
longrisk UCB <= 0.15。
```

Stage64 打开条件：

```text
Stage32 Fast/Slow precision >= 0.25；
V240 LCB > 0；
longrisk UCB <= 0.10。
```

### Codex 不满足时先尝试

```text
如果 Stage4 全 BadPath：
  停止该 family；
  不扩大。

如果 high AUV high risk：
  标为 RiskyHighAUV；
  加 risk-safe trust region；
  不调 AUV threshold。

如果 low-risk low-value：
  标为 SafeLowValue；
  检查 delayed gain；
  不写 success。

如果 collision / apply error：
  修 payload lifecycle；
  不进入 science conclusion。
```

---

## 13. P8：Controller Boundary

只有满足以下任一条件，才打开 controller：

```text
P4 strong pass；
P5 strong pass；
P7 Stage64 weak pass。
```

Controller gate：

```text
accepted_count >= 87；
Fast/Slow precision >= 0.75；
V240 LCB > 0；
RAUV LCB > 0；
longrisk UCB <= 0.05；
bad UCB <= 0.05；
null UCB <= 0.15；
memory/offdiag UCB <= 0.05；
LDO / LSO / LTO / LFO <= 0.10。
```

Codex 不满足时：

```text
如果 accepted_count < 87：
  不降 precision/risk gate；
  回到动作来源问题。

如果 precision 高但 leaveout fail：
  做 group/state/family support audit；
  不按 dataset 调参。

如果 risk fail：
  回到 P3 Pareto frontier；
  不调 value threshold 掩盖。
```

---

## 14. P9：Runtime 与 Paired Replay

Runtime 只在 controller pass 后打开。

Runtime gate：

```text
selected runtime step_ratio_q90 <= 1.50；
no CPU offload；
materializer not in timed path；
feature / certificate cost accounted；
```

Paired replay gate：

```text
RealFunctional beats AdamWParallel；
beats bestLR；
beats NoOp；
beats Random；
shuffled payload fails；
leave-dataset-out / leave-stratum-out / leave-template-out stable。
```

---

## 15. 本轮最终决策表

v10.7 结束时必须给出以下四类判断之一：

### Case A：当前 base 可控

条件：P2/P3/P4/P7 至少一个 strong pass。

下一步：controller/runtime/paired replay。

### Case B：当前 base 只在 optimizer-state 可控

条件：P5 strong pass，但 P2/P3/P4 fail。

下一步：optimizer-state functional update。

### Case C：当前 base 不可控，但 architecture bridge 有信号

条件：P2/P5 fail，P6 pass。

下一步：compositional / architecture trainability line。

### Case D：当前 base、当前 basis、optimizer-state、architecture bridge 都无信号

条件：P2/P5/P6 全 fail。

下一步：停止当前 DG-KAN functional update 主线，重新定义 base architecture 或训练目标。不得继续新增 APG/FPAU/GUP 名字。

---

## 16. 本计划的核心原则

本轮不是继续猜动作，而是判断可控性：

$$
\boxed{
\text{如果局部扰动空间没有 risk-safe positive future response，任何 selector 都无意义。}
}
$$

因此 v10.7 的最低有效成果不是 controller，而是一个硬判断：

```text
current base controllable / not controllable；
parameter update controllable / optimizer-state controllable / architecture needed；
future-path objective feasible / infeasible under current representation。
```

只有这个判断清楚，后续才不会继续在 proxy、rank、generator family 之间绕圈。
