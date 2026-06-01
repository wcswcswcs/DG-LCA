# DG-KAN v10.8 Decisive Controllability Exit / Known-Good Replay / Architecture Bridge 完整实验计划

> 目标：不再继续小修 FPO、FPAU、GoodCone、LFRO 或 generated family。v10.8 要做一次硬判断：**当前 DG-KAN base + 当前允许更新空间，是否真的具备可控的 future-path-improving functional update 能力？**
>
> 本计划使用 Typora 友好公式，所有公式使用 `$...$` 或 `$$...$$`。本计划不按数据集调参，不把 diagnostic 写成 official，不用 fake/proxy/CPU offload，不把 discovery sandbox 写成 controller pass。

---

# 0. 当前结论：为什么必须重置

v10.7 的结果不是“差一点”。它显示：

```text
route = CaseD-WeakParameterPocketsNotControllable
primary_blocker = weak_parameter_pockets_not_pareto_feasible
secondary_blocker = optimizer_state_and_architecture_bridge_failed
system_legal_controller_pass = 0
```

核心边界是：

```text
P1 Fast/Slow/Risky/SafeLow/Bad = 4 / 22 / 321 / 26 / 14707
P2 basis actions = 480
P2 risk-safe positive = 4
P2 risk-safe positive rate = 0.008333333333333333
P2 weak/strong = 1 / 0
P3 Pareto efficient = 4
P3 risk-safe positive Pareto = 1
P3 weak/strong = 0 / 0
P5 optimizer-state candidates = 7
P5 risk-safe positive = 0
P6 architecture bridge pass = 0
P6 blockwise limited-real rate = 0.020833333333333332
```

这意味着：

$$
\boxed{
\text{当前系统存在少量弱 parameter pockets，但这些 pocket 不能形成可控、Pareto-feasible、可部署的 functional update。}
}
$$

v10.8 不再问：

```text
FPO 阈值能不能再调？
GoodCone 能不能换 metric？
LFRO 能不能换 Huber / clip？
generated family 能不能多跑几个？
```

v10.8 要问更硬的问题：

$$
\boxed{
\text{已知好动作的真实方向，当前系统能不能复现、局部扰动、状态化利用、或结构化生成？}
}
$$

如果答案是否，就应该停止当前 DG-KAN functional-update 主线，转向 base architecture / compositional trainability / update representation 重建。

---

# 1. v10.8 总体设计原则

v10.8 把实验分成三类 runner，防止继续“所有事情都被 official gate 串行卡死”。

## 1.1 Engineering Gate Runner

只验证：payload、action apply、branch-horizon、state hash、runtime入口是否真实可跑。

这类 runner 不回答 science question，不写 system pass。

## 1.2 Science Discovery Runner

允许 non-official 探索：known-good replay、local cone、basis coverage、optimizer-state、architecture bridge。

这类 runner 必须显式标注：

```text
no controller
no runtime pass
no paired replay pass
no short/full pass
not official
```

## 1.3 Official Gate Runner

只有 discovery 明确出现强机制后才打开：controller、runtime、paired replay、short/full。

硬门保持不变：

$$
accepted\_count \ge 87
$$

$$
Precision \ge 0.75
$$

$$
V240^{LCB} > 0
$$

$$
LongRisk^{UCB} \le 0.05
$$

$$
Bad^{UCB} \le 0.05
$$

$$
Null^{UCB} \le 0.15
$$

$$
MemoryFail^{UCB} \le 0.05
$$

$$
OffdiagFail^{UCB} \le 0.05
$$

$$
StepRatio_{q90} \le 1.50
$$

---

# 2. P0：复现 v10.7 terminal boundary

## 2.1 目标

锁定 v10.7 真实边界，不重新解释旧失败。

## 2.2 必须记录

```text
v10.7 route
P1 path type counts
P2 basis actions / branch rows / risk-safe positive count
P3 Pareto efficient count / risk-safe Pareto count
P5 optimizer-state candidate outcomes
P6 architecture bridge outcomes
fake/proxy/cpu audit
```

## 2.3 通过标准

```text
boundary_reproduced = 1
no_fake = 1
no_proxy = 1
no_cpu_offload = 1
old_LFRO_FPO_GoodCone_small_fix_stopped = 1
```

## 2.4 不满足条件时 Codex 先尝试

```text
如果 artifact 缺失：先补 manifest / recap，不要重跑 full science。
如果 rows mismatch：先检查 branch-horizon join key。
如果 fake/proxy/cpu > 0：整轮作废，先修 replay/materializer。
```

---

# 3. P1：Known-Good Replay Controllability

## 3.1 目标

回答最关键的问题：**已知 FastGood / SlowBurnGood 的真实 delta，当前系统能否复现它的 future path？**

这一步不是 selector，也不是 generator。它是 controllability oracle test。

## 3.2 假设

$$
H_1:
\text{Fast/Slow 的真实方向可以在当前允许参数更新空间中被复现。}
$$

如果 $H_1$ 不成立，说明当前 action representation 连已知好动作都无法复现，继续做 FPO / generated selector 没意义。

## 3.3 实验对象

选择四组动作：

```text
FastGood-known
SlowBurnGood-known
RiskyHighAUV-known
BadPath-matched
```

每个 Fast/Slow 动作构造四类 replay：

```text
D0: original delta direct replay
D1: projection onto current B0-B9 basis
D2: residual-only replay
D3: sign/norm-scaled replay
D4: memory/offdiag soft-trust replay
```

## 3.4 要记录的指标

对每个动作和 replay 记录：

```text
action_id
path_type_original
replay_type
state_id
dataset/family/template/step_bucket
projection_r2
projection_residual_norm
cosine_to_original_delta
norm_ratio
V1/V5/V20/V80/V240
RAUV
longrisk
bad/null
memory_fail/offdiag_fail
old_family_loss_delta
hard_tail_CEp99_delta
action_apply_linf
runtime_cost_ms
```

## 3.5 判断标准

P1 weak pass：

```text
至少 8 个 known Fast/Slow 在 D0 direct replay 下仍保持 Fast/Slow-like；
longrisk UCB <= 0.05；
memory/offdiag UCB <= 0.05。
```

P1 strong pass：

```text
至少 8 个 known Fast/Slow 在 D1 projection replay 下仍保持 Fast/Slow-like；
projection_r2 >= 0.80；
V240 LCB > 0；
longrisk UCB <= 0.05。
```

## 3.6 不满足条件时 Codex 先尝试

```text
如果 D0 direct replay 也失败：
  先检查 state mismatch / optimizer state mismatch / RNG mismatch / batch sequence mismatch；
  不要继续生成新动作。

如果 D0 成功但 D1 失败：
  说明当前 B0-B9 basis 不覆盖好方向；
  先从 residual directions 做 SVD / low-rank basis expansion；
  不要调 FPO threshold。

如果 D1 成功但 D4 失败：
  说明 memory/offdiag hard/soft constraint 破坏 value；
  改做 Pareto soft trust region，不要 hard projection。

如果 FastGood 可复现但 SlowBurnGood 不可复现：
  说明 delayed-gain 方向需要 optimizer-state / multi-step basis；
  进入 P4 optimizer-state phase，不要扩大普通 parameter basis。
```

## 3.7 可视化

```text
projection_r2 vs V240 scatter
cosine_to_original_delta vs RAUV scatter
replay_type × path_type heatmap
D0/D1/D4 的 V1/V20/V80/V240 曲线
memory/offdiag risk vs projection residual 图
```

---

# 4. P2：Local Good-Cone Around Known-Good

## 4.1 目标

判断好动作是不是孤立点，还是局部有可控 cone。

如果已知好动作周围没有任何 risk-safe positive neighborhood，说明它们不是可泛化控制方向，而是历史偶然点。

## 4.2 设计

围绕 P1 中 D0 成功的 Fast/Slow 动作，构造局部扰动：

$$
\Delta(\alpha)=\Delta_{good}+B\alpha
$$

其中 $B$ 包含：

```text
original residual basis
AdamW direction
momentum direction
memory-safe direction
hard-tail correction direction
old-family preserving direction
low-rank random tangent directions
```

每个 anchor 采样：

```text
radial scales: 0.25, 0.5, 1.0, 1.5
angular perturbations: 8
sign flips: original / negative / orthogonalized
```

## 4.3 要记录的指标

```text
anchor_action_id
anchor_path_type
perturbation_scale
angle_to_anchor
basis_component_weights
V1/V5/V20/V80/V240
RAUV
longrisk
memory/offdiag
bad/null
Pareto efficient flag
connected_component_id in good region
```

## 4.4 判断标准

P2 weak pass：

```text
至少 3 个 anchors 周围存在 >= 4 个 risk-safe positive perturbations；
这些 perturbations 不是全在 scale=0 的 trivial original action。
```

P2 strong pass：

```text
存在一个 connected good cone：
  >= 12 risk-safe positive perturbations；
  V240 LCB > 0；
  LongRisk UCB <= 0.05；
  memory/offdiag UCB <= 0.05；
  angular span >= 30 degrees。
```

## 4.5 不满足条件时 Codex 先尝试

```text
如果只有 anchor 本身好，邻域都坏：
  标记为 isolated-good，不能生成；
  转向 architecture / state-specific diagnosis。

如果小 scale 好、大 scale 坏：
  增加 trust-region bound，测试 epsilon sweep。

如果 value 好但 risk 高：
  不调 value；先查 memory/offdiag/hard-tail 对应 component。

如果 risk safe 但 value 低：
  查是否 SafeLowValue / delayed gain；增加 h240+ diagnostic only，不写 official。
```

## 4.6 可视化

```text
2D cone map: angle × scale colored by path_type
Pareto frontier: V240 vs longrisk
RAUV vs memory/offdiag scatter
anchor neighborhood trajectory curves
```

---

# 5. P3：State Phase Controllability

## 5.1 目标

判断好 update 是否只在特定训练状态相位有效。

v10.5 的 state bucket AUC 局部高但样本太少，不能 controller。P3 要用 replay 方式验证 state phase，而不是只做 bucket AUC。

## 5.2 设计

对 P1/P2 中可复现的 known-good delta，在不同 state buckets 重放：

```text
same dataset/family/template but different step phase
same step phase but different family/template
different optimizer state norm bucket
different memory/offdiag pre-state
```

## 5.3 记录指标

```text
source_state_id
target_state_id
state_distance
optimizer_m_norm
optimizer_v_norm
grad_norm
memory_pre_state
offdiag_pre_state
same_delta_path_type_after_transfer
V1/V20/V80/V240
longrisk
memory/offdiag
```

## 5.4 判断标准

P3 weak pass：

```text
state-conditioned validity explains >= 50% of transfer success/failure；
state feature不包含 dataset_name，不包含 future outcome。
```

P3 strong pass：

```text
训练当下 state feature 能把可用状态和不可用状态分开：
AUC >= 0.75；
TopK state-conditioned replay precision >= 0.50；
longrisk UCB <= 0.10。
```

## 5.5 不满足条件时 Codex 先尝试

```text
如果同一 delta 跨 state 全失败：
  好动作不是可迁移控制方向，降低 generated-route 希望。

如果只在 source state 有效：
  必须做 state-local controller，不允许 global selector。

如果 state feature 靠 dataset 才有效：
  丢弃该 feature，不允许 dataset-specific controller。

如果 optimizer state 是主因素：
  进入 P4 optimizer-state intervention。
```

---

# 6. P4：Optimizer-State Controllability

## 6.1 目标

判断未来路径改善是否主要来自 optimizer state，而不是参数 delta。

v10.7 P5 的 7 个 optimizer-state candidate 全部 risk-safe positive = 0，但样本太小、family 太少。P4 只在 P1/P3 提示 optimizer-state 重要时运行，否则跳过。

## 6.2 候选干预

```text
O1: momentum damp/amplify along known-good residual
O2: second-moment trust scaling
O3: memory-safe preconditioner
O4: signal-channel SNR preconditioner
O5: blockwise orthogonalized update
O6: hard-tail local step rescale
O7: old-family preserving state correction
O8: known-good optimizer-state replay
O9: state residual SVD component injection
```

## 6.3 记录指标

```text
optimizer_update_id
state_delta_norm
m_delta_norm
v_delta_norm
alignment_with_good_delta
V1/V20/V80/V240
RAUV
longrisk
memory/offdiag
old_family_loss_delta
runtime_cost
```

## 6.4 判断标准

P4 weak pass：

```text
>= 2 optimizer-state updates produce risk-safe positive future path；
V240 LCB > 0；
longrisk UCB <= 0.10。
```

P4 strong pass：

```text
>= 8 risk-safe positive optimizer-state updates；
Fast/Slow precision >= 0.25 at Stage16；
runtime cost estimate <= 1.5 ms discovery gate。
```

## 6.5 不满足条件时 Codex 先尝试

```text
如果 all BadPath：
  停止 optimizer-state route，不新增 O10/O11。

如果 V20 好但 V240 坏：
  这是 short-term boost，不是 functional update；加入 longrisk/h240 gate。

如果 risk 低但 value 低：
  降级为 safety modifier，不作为 generator。
```

---

# 7. P5：Architecture / Representation Bridge

## 7.1 目标

判断当前 base architecture 是否缺乏 controllability。如果 current base 不可控，就不能继续在它上面找 update rule。

## 7.2 候选 bridge

不按数据集调参，只做结构性对照：

```text
A0: current LQ-t2-h256 base
A1: blockwise-limited-real bridge
A2: shallow compositional FullEdge-lite
A3: state-conditioned edge cover adapter
A4: low-rank dynamic basis expansion
A5: memory-preserving edge residual basis
```

每个 bridge 只跑小 panel：

```text
states = 4
basis actions = 64
branch horizons = h1,h20,h80,h240
```

## 7.3 记录指标

```text
bridge_id
params_ratio
step_time_ratio_proxy
risk_safe_positive_rate
FastSlow count
V240 LCB
longrisk UCB
memory/offdiag UCB
materializer pass
action apply linf
```

## 7.4 判断标准

P5 weak pass：

```text
某个 bridge 的 risk_safe_positive_rate >= 0.03；
V240 LCB > 0；
longrisk UCB <= 0.10。
```

P5 strong pass：

```text
risk_safe_positive_rate >= 0.05；
FastSlow precision >= 0.25；
params_ratio <= 1.05；
step_time_ratio_proxy <= 1.50。
```

## 7.5 不满足条件时 Codex 先尝试

```text
如果 all bridges risk_safe_positive_rate < 0.01：
  标记 current architecture family not controllable under current constraints；
  停止 functional-update mainline。

如果 compositional bridge 有信号但 runtime 差：
  转 kernel-native compositional FullEdge 工程，不继续 selector。

如果 dynamic basis 有 value 但 risk 高：
  加 memory/offdiag soft trust，不做 hard projection。
```

---

# 8. P6：Decision Matrix / Route Exit

## 8.1 目标

v10.8 必须输出路线裁决，不允许再只是“继续 v10.9 小修”。

## 8.2 路线裁决

```text
Case A: Known-good direct replay fail
  结论：artifact/state mismatch 或 old good labels 不可复现。
  下一步：修 replay/state identity，不做 generated。

Case B: Direct works, projection fails
  结论：当前 basis 不覆盖好方向。
  下一步：basis expansion / residual SVD / architecture bridge。

Case C: Projection works, neighborhood absent
  结论：好动作是 isolated pocket，不可生成。
  下一步：state-specific mechanism 或 architecture reset。

Case D: Neighborhood exists, FPO fails
  结论：机制存在但识别失败。
  下一步：训练 path-type predictor / local response operator。

Case E: Architecture bridge works
  结论：current base 不可控，但新 structure 可控。
  下一步：转 bridge base，不再在旧 base 上调 selector。

Case F: P1-P5 全失败
  结论：当前 DG-KAN functional update mainline 暂停。
  下一步：回到 architecture / trainability / compositional kernel-native research。
```

---

# 9. P7：条件性 controller / runtime / paired replay

只有以下条件满足才打开 P7：

```text
P1 strong 或 P2 strong 或 P5 strong = 1
```

否则：

```text
controller = not_run
runtime = not_run
paired replay = not_run
short/full = not_run
```

如果打开 P7，controller 仍需满足：

```text
accepted_count >= 87
FastSlow precision >= 0.50 discovery / 0.75 official
V240 LCB > 0
longrisk UCB <= 0.05
bad/null UCB <= 0.05 / 0.15
memory/offdiag UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
step_ratio_q90 <= 1.50
```

---

# 10. 必须画的图

```text
1. Known-good replay projection residual vs V240
2. Good-cone angle × scale heatmap
3. Pareto frontier: V240 vs LongRisk
4. State transfer matrix: source_state × target_state
5. Optimizer-state intervention path curves
6. Architecture bridge risk-safe-positive rate bar plot
7. Route decision Sankey: P1/P2/P3/P4/P5 -> Case A-F
```

---

# 11. 最终 stop rule

v10.8 必须有停机条件：

```text
如果 P1 direct known-good replay 不可复现：
  不再讨论 generated optimizer，先修 state/replay identity。

如果 P1 direct 可复现但 P2/P5 都没有局部 controllability：
  停止当前 update representation 主线。

如果 P5 architecture bridge 也没有任何 signal：
  停止 DG-KAN functional update mainline，转向 architecture/trainability research。

如果 P2 或 P5 出现 strong signal：
  才允许重新设计 generated discovery。
```

---

# 12. 本轮最重要的判断

v10.8 不是为了再找到一个小分数，而是为了回答：

$$
\boxed{
\text{当前系统到底是否可控？}
}
$$

如果不可控，最科学的结论不是继续试，而是停止当前 functional-update 搜索，把资源转向 architecture / compositional trainability / kernel-native base reconstruction。
