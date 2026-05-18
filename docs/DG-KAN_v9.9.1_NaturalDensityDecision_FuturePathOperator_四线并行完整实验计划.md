# DG-KAN v9.9.1 自然动作密度裁决 / FuturePathOperator / 四线并行完整实验计划

> 本计划基于 v9.9.0 真实结果制定。v9.9.0 已经把 `natural AP0 extension materializer` 从“找不到入口”推进到“工程 smoke 全过”，但还没有完成 5000 / 10000 / 20000 的自然动作密度裁决，也没有打开 controller、runtime、paired replay 或 short/full training。
>
> 本轮不继续小修 AUV、LC proxy、CoreExpansion、APG 变体，也不把 256-action smoke 当成 density closure。v9.9.1 的核心任务是把“四线并行”变成可裁决的实验：先判断自然动作源是否真的有足够好动作密度，再判断 future path 是否能被训练当下低成本预测，最后才允许 controller 或 generated sandbox。

---

## 0. 当前状态一句话

v9.9.0 的真实推进是：

```text
natural extension generator entrypoint 已落地；
1 / 16 / 256 action smoke 全部通过；
新增 natural actions = 273；
新增 branch-horizon rows = 8190；
没有旧 action / payload collision；
没有 fake / proxy / CPU offload。
```

但 v9.9.0 仍然没有系统成功：

```text
5000 / 10000 / 20000 density panel 未运行；
P3 future path decomposition 未运行；
P4 FuturePathOperator sketch 未运行；
controller 未运行；
generated sandbox 未打开；
runtime / paired replay / short-full 未打开。
```

因此当前状态是：

$$
\boxed{\text{自然动作扩流工程门已经打开，但自然动作密度和可部署机制还没有裁决。}}
$$

---

## 1. 对 v9.9.0 的独立判断

### 1.1 这轮有真实进展

v9.8.9 的主 blocker 是：

```text
natural extension generator missing
```

v9.9.0 已经把这个 blocker 推进掉：

```text
entrypoint_count = 1
generator_found = 1
generator module = experiments/natural_ap0_extension_materializer.py
```

并且工程 smoke 是真实闭合的：

```text
P1b single-action rows = 30 / 30
P1c 16-action rows = 480 / 480
P1d 256-action rows = 7680 / 7680
action_apply_linf_max = 0.0
NaN / Inf = 0 / 0
cpu_offload = 0
old action collision = 0
old payload collision = 0
```

这说明现在不再是“没有新自然动作生成入口”的阶段。

### 1.2 但这不是能力进展

v9.9.0 的 273 个 new natural actions 只能算 smoke panel。它不能替代 5000 / 10000 / 20000 density panel。

当前 smoke density 是：

```text
natural smoke action count = 273
CoreLike count = 2
CoreLike LCB / UCB = 0.002011 / 0.026314
PathGood count = 3
PathGood LCB / UCB = 0.003744 / 0.031805
```

这个结果不能写成“density insufficient”，因为样本太小；但它也不能被忽略。特别是 CoreLike 只有 2 / 273，UCB 已经低于 0.03，这对自然动作源是一个明显预警。

如果真实 CoreLike rate 是 0.03，在 273 个样本里只观察到 2 个或更少的概率大约为 1.1%。这说明 smoke panel 可能不是简单随机低估，而可能存在以下问题：

```text
1. natural extension generator 与 canonical AP0 action distribution 不一致；
2. 新动作 provenance / family / step bucket 偏离旧 good-action 区；
3. generator 能产生合法动作，但没有产生足够 CoreLike / PathGood 动作；
4. 273 action 太小，仍需要更大 panel 裁决。
```

所以 v9.9.1 不能直接跳到 controller，也不能只跑一个 5000 panel然后看总数。必须先审计 natural generator 的分布保真度，然后再跑序贯 density panel。

---

## 2. 四线总体目标

v9.9.1 保留四线，但优先级更硬。

### A 线：未来训练路径

目标是回答：

$$
\boxed{\text{好动作是否真的把后续训练带到更好路径，而不是只让当前一步 loss 下降？}}
$$

本线不再只看 AUV。必须同时看：

```text
h1 / h5 / h20 / h80 / h240 的 V 曲线；
RiskAdjustedAUV；
longrisk；
bad / null；
memory fail；
offdiag fail；
hard-tail；
不同 dataset / family / template 的稳定性。
```

### B 线：训练当下合法 FuturePathOperator

目标是回答：

$$
\boxed{\text{训练当下有没有低成本信号，可以预测动作的未来路径类型？}}
$$

本线不再调 `LC1`、`LC8` 或 `FO8` 阈值。要重新设计低成本 future-operator sketch。

### C 线：自然 AP0 动作扩流

目标是回答：

$$
\boxed{\text{自然动作源里 CoreLike / PathGood / SlowBurnGood 的真实密度到底够不够？}}
$$

这是当前最高优先级。没有 C 线结果，就无法判断 existing-action route 是否可持续。

### D 线：生成新动作

目标是回答：

$$
\boxed{\text{如果自然动作源密度不足，是否能基于 A/B 线发现的机制生成好动作？}}
$$

D 线默认停止。只有 A/B/C 满足 reopen 条件，才允许 64-action sandbox。

---

## 3. 本轮不做什么

v9.9.1 明确不做：

```text
1. 不把 256-action smoke 写成 density closure；
2. 不继续调 LC8 / LC1 / FO8 threshold；
3. 不只看 AUV / RAUV；
4. 不只看 Core77；
5. 不按 dataset 调阈值；
6. 不在 full density panel 未完成前打开 controller；
7. 不在 B/C 线没有证据前重开 generated route；
8. 不把工程 smoke pass 写成 science pass；
9. 不把 natural generator found 写成 functional success；
10. 不用 fake / proxy / CPU offload。
```

允许做：

```text
1. dataset / family / template 只用于诊断和 leaveout；
2. 使用 natural generator 的 provenance、family、step bucket、payload/action stats 做分布审计；
3. 使用真实 branch-horizon replay rows 做 density 和 future path；
4. 使用 legal commit-time fields 设计低成本 FuturePathOperator sketch；
5. 使用 staged panel fail-fast，避免一次性大跑后才发现分布错；
6. 只有满足硬 gate 后才进入 controller/runtime。
```

---

## 4. P0：复现 v9.9.0 边界

### 目标

确认 v9.9.0 的工程边界没有回退。

### 假设

v9.9.0 已经解决 `natural_extension_generator_missing`，但没有解决 full density。

### 记录指标

```text
source_route_v9900
P1a_entrypoint_search_pass
P1b_single_action_preflight_pass
P1c_16_action_preflight_pass
P1d_256_action_smoke_pass
new_natural_action_count
new_branch_horizon_rows
old_action_collision_count
old_payload_collision_count
fake/proxy/cpu counts
system_legal_controller_pass
generated_route_status
```

### 判断标准

必须满足：

```text
P1a/P1b/P1c/P1d = 1/1/1/1
old_action_collision_count = 0
old_payload_collision_count = 0
fake/proxy/cpu = 0/0/0
system_legal_controller_pass = 0
```

如果 P0 失败，本轮只允许修工程，不允许进入科学判断。

---

## 5. P1：Natural Generator 分布保真度审计

### 目标

判断新 natural extension generator 产生的动作是否像原 canonical AP0 action stream 的自然延伸，而不是另一个偏置 action source。

### 核心问题

v9.9.0 的 273 smoke actions 里 CoreLike 只有 2 个，PathGood 只有 3 个。这可能只是小样本噪声，也可能说明 new natural generator 采样分布偏离了原 canonical AP0 universe。

### 要比较的集合

```text
A. canonical AP0 panel: 2876 old actions
B. v9.9.0 natural smoke: 273 new actions
C. v9.9.1 new 1024-action pilot panel
D. v9.9.1 5000 / 10000 / 20000 panels, if opened
```

### 记录字段

```text
action_id
candidate_id
payload_hash
provenance
family
step_bucket
dataset
seed
template_id
payload_norm
action_norm
payload_linf
action_linf
AdamW alignment
action sparsity
memory_score
offdiag_score
hard_tail_score
old_family_probe
branch-horizon completion
CoreLike label
PathGood label
SlowBurnGood label
RiskyHighAUV label
```

### 统计指标

对每个字段比较 old canonical vs new natural：

```text
PSI
KL
Wasserstein distance
max group share
max group gap
coverage per family
coverage per step bucket
coverage per dataset
coverage per template
old/new collision count
```

### 判断标准

弱通过：

$$
PSI_{major}\le 0.20
$$

$$
max\_group\_share\le 0.50
$$

$$
old\_collision=0
$$

强通过：

$$
PSI_{major}\le 0.10
$$

$$
max\_group\_share\le 0.35
$$

$$
\text{no major family / step bucket missing}
$$

如果 P1 不过，不允许直接跑 5000 / 10000 / 20000 density conclusion。应该先修 natural generator 的采样分布。

### 可视化

```text
1. old vs new action family histogram；
2. old vs new step bucket histogram；
3. old vs new payload norm density；
4. old vs new AdamW alignment density；
5. PSI heatmap by feature group；
6. CoreLike / PathGood rate by family and step bucket。
```

---

## 6. P2：序贯自然动作密度裁决

### 目标

裁决自然 AP0 action stream 中是否有足够高密度的好动作。

### 核心假设

存在两种可能：

```text
H_density_enough:
  自然动作源中 CoreLike / PathGood / SlowBurnGood 密度足够，
  现有 Core77 只是 2876 动作样本太小导致数量不足。

H_density_low:
  自然动作源本身好动作密度低于可部署 coverage 下限，
  existing-action harvesting 无法支持 controller。
```

### Panel 设计

采用序贯 panel：

```text
pilot_1024
panel_5000
panel_10000
panel_20000
```

每一步都必须真实 materialize branch-horizon rows，不允许 fake labels。

### 标签定义

至少记录以下标签：

```text
CoreLike:
  类似 Core77 的严格干净动作。

PathGood:
  future path 综合好，risk-adjusted AUV 正，longrisk 低。

SlowBurnGood:
  h1 / h5 不一定好，但 h20/h80/h240 明显变好。

RiskCleanButLowImmediate:
  即时 value 低，但 longrisk / memory / offdiag 干净，未来路径可能变好。

RiskyHighAUV:
  AUV 高，但 longrisk / memory / offdiag 高。

BadPath:
  value 低或 bad/null/longrisk 高。
```

### 记录指标

每个 panel 记录：

```text
panel_action_count
branch_horizon_expected_rows
branch_horizon_actual_rows
completion_rate
rows_per_sec
wallclock
peak_gpu_mb
CoreLike_count / rate / LCB / UCB
PathGood_count / rate / LCB / UCB
SlowBurnGood_count / rate / LCB / UCB
RiskCleanButLowImmediate_count / rate / LCB / UCB
RiskyHighAUV_count / rate / LCB / UCB
BadPath_count / rate / LCB / UCB
per-dataset rates
per-family rates
per-template rates
per-step-bucket rates
```

### 判断标准

密度足够：

$$
LCB(CoreLike)\ge 0.03
$$

或：

$$
LCB(PathGood)\ge 0.03
$$

并且：

$$
LongRisk^{UCB}\le 0.05
$$

$$
Bad^{UCB}\le 0.05
$$

$$
Null^{UCB}\le 0.15
$$

密度不足：

$$
UCB(CoreLike)<0.03
$$

且：

$$
UCB(PathGood)<0.03
$$

在 $N\ge 5000$ 时如果已经满足不足条件，可以直接进入 pivot；不需要等到 20000。

密度不确定：

```text
LCB < 0.03 <= UCB
```

则继续下一个 panel。

### 可视化

```text
1. CoreLike / PathGood rate vs panel size，带 Wilson interval；
2. SlowBurnGood / RiskyHighAUV density 曲线；
3. per-dataset density 曲线；
4. per-family density heatmap；
5. panel throughput 与 peak GPU memory 曲线；
6. old canonical vs new natural density 对比图。
```

---

## 7. P3：未来路径类型分解

### 目标

不再只看 AUV，而是拆清楚不同动作的未来训练路径类型。

### 分组

至少比较：

```text
Core77
OldOnly
ExactOnly
RandomMatched
CoreExpansion10
RiskCleanButLowValue
NewNaturalCoreLike
NewNaturalPathGood
NewNaturalRiskyHighAUV
NewNaturalBadPath
```

### 路径指标

每个 action × horizon 记录：

```text
horizon = 1, 5, 20, 80, 240
V_h
RiskAdjustedV_h
CE_delta_h
NLL_delta_h
margin_delta_h
CEp99_delta_h
LongRisk_h
Bad_h
Null_h
MemoryFail_h
OffdiagFail_h
HardTailFail_h
```

综合指标：

```text
AUV
RiskAdjustedAUV
DelayedGain = V240 - V1
SlowBurnIndex = max(0, V240 - V5)
RiskPath = max LongRisk over horizons
MemoryPath = max MemoryFail over horizons
OffdiagPath = max OffdiagFail over horizons
PathStability = min(V20,V80,V240)
```

### 路径类型定义

```text
FastGood:
  V1 > 0 且 V20/V80/V240 稳定为正，risk 低。

SlowBurnGood:
  V1 <= 0 或 V5 <= 0，但 V20/V80/V240 转正，risk 低。

RiskCleanButLowImmediate:
  immediate value 不强，但 risk/memory/offdiag 很干净，future value 非负。

RiskyHighAUV:
  AUV 高，但 longrisk/memory/offdiag 高。

BadPath:
  V 不稳定或 risk/bad/null 高。
```

### 判断标准

A 线 weak pass：

```text
Core77 或 NewNaturalCoreLike 的 RiskAdjustedAUV LCB > 0；
LongRisk UCB <= 0.05；
Memory/Offdiag UCB <= 0.05。
```

A 线 strong pass：

```text
至少 87 个 actions 组成 accepted region；
RiskAdjustedAUV LCB > 0；
PathStability LCB > 0；
LongRisk/Bad/Memory/Offdiag UCB 全部过线；
LDO/LSO/LTO/LFO <= 0.10。
```

### 可视化

```text
1. group × horizon 的 V 曲线；
2. group × horizon 的 risk 曲线；
3. DelayedGain vs RiskPath scatter；
4. AUV vs LongRisk scatter；
5. Core77 / OldOnly / ExactOnly 对比雷达图；
6. NewNatural actions 的 path type 堆叠条形图。
```

---

## 8. P4：低成本 FuturePathOperator Sketch v4

### 目标

训练当下低成本预测未来路径类型，而不是预测当前函数响应大小。

### 禁止路线

```text
不再调 LC1 / LC8；
不使用 future outcome；
不使用 old table diagnostic；
不使用 dataset_name；
不把 AUC 写成 pass；
不接受 cost q90 > runtime budget 的 proxy。
```

### Candidate Sketches

#### FOS1：Tiny Virtual AdamW Sketch

在极小样本上模拟 $k=1$ 或 $k=3$ 步 AdamW 方向，但不做 full branch-horizon。

记录：

```text
virtual_step_count
small_batch_size
memory_sample_size
hard_tail_sample_size
estimated_future_value
estimated_memory_harm
estimated_longrisk_proxy
cost_q90
```

#### FOS2：JVP/VJP Path Sketch

估计 action 对 memory/hard-tail 输出的低秩响应。

记录：

```text
JVP action response norm
VJP memory response alignment
hard-tail response alignment
old-family response alignment
cost_q90
```

#### FOS3：Delayed Gain Sketch

专门预测 `SlowBurnGood`：即时响应不强，但未来路径转正。

记录：

```text
immediate_response
memory_response
hard_tail_response
AdamW alignment
estimated delayed gain
```

#### FOS4：Risk-Adjusted Hard Gate

不做加权总分，只做硬门：

```text
if memory_harm > threshold: reject
if offdiag_risk > threshold: reject
if hard_tail_risk > threshold: reject
if cost > threshold: reject
else use value rank
```

### 判断标准

B 线 weak pass：

$$
accepted\_count \ge 87
$$

$$
Precision \ge 0.65
$$

$$
RiskAdjustedAUV^{LCB} > 0
$$

$$
LongRisk^{UCB}\le 0.05
$$

$$
Cost_{q90}\le 0.20ms
$$

B 线 strong pass：

$$
Precision \ge 0.75
$$

$$
V^{LCB}>0
$$

$$
LDO,LSO,LTO,LFO\le 0.10
$$

$$
Cost_{q90}\le 0.05ms
$$

### 可视化

```text
1. proxy score vs RiskAdjustedAUV scatter；
2. proxy score vs LongRisk scatter；
3. cost-quality Pareto；
4. accepted region path-type distribution；
5. rejected vs accepted memory/offdiag distribution；
6. LDO/LSO/LTO/LFO drop dashboard。
```

---

## 9. P5：Existing-Action Controller Boundary

### 目标

只有当 C 线密度足够或 B 线 proxy 过线时，才尝试 existing-action minimal controller。

### Controller 形式

不能复杂。优先使用：

```text
rank by FuturePathOperatorSketch；
apply hard veto: memory/offdiag/longrisk/cost；
select TopK or fixed accepted count；
threshold frozen on calibration split；
evaluate heldout + leaveout。
```

### 记录指标

```text
accepted_count
coverage
precision
V_LCB
RiskAdjustedAUV_LCB
LongRisk_UCB
Bad_UCB
Null_UCB
Memory_UCB
Offdiag_UCB
LDO_drop
LSO_drop
LTO_drop
LFO_drop
cost_q90
feature compute time
payload apply time
```

### 通过标准

$$
accepted\_count\ge87
$$

$$
Precision\ge0.75
$$

$$
V^{LCB}>0
$$

$$
LongRisk^{UCB}\le0.05
$$

$$
Bad^{UCB}\le0.05
$$

$$
Null^{UCB}\le0.15
$$

$$
Memory^{UCB}\le0.05
$$

$$
Offdiag^{UCB}\le0.05
$$

$$
LDO,LSO,LTO,LFO\le0.10
$$

如果 P5 不过，不允许打开 runtime。

---

## 10. P6：Generated Route Reopen Gate

### 目标

判断是否允许重新生成动作。

### 允许重开条件

满足以下任一条件才允许 64-action generated sandbox：

```text
Condition D1:
  C 线证明 natural action density insufficient，且 A/B 给出明确 path mechanism。

Condition D2:
  B 线 FuturePathOperator sketch strong pass，可以作为生成目标。

Condition D3:
  A 线发现明确 path type，例如 SlowBurnGood，且已有合法 commit-time proxy 能预测。
```

否则：

```text
generated_sandbox_allowed = 0
```

### 64-action sandbox 记录

```text
generated_action_count
branch_horizon_rows
PathGood precision
RiskAdjustedAUV_LCB
LongRisk_UCB
Memory/Offdiag UCB
new positive created rate
longrisk created rate
cost
```

### 通过标准

$$
PathGoodPrecision\ge0.25
$$

$$
RiskAdjustedAUV^{LCB}>0
$$

$$
LongRisk^{UCB}\le0.10
$$

如果 64-action sandbox 失败，不允许扩到 512。

---

## 11. P7：Selected Runtime Boundary

### 目标

只有 controller 过线后，才测 selected runtime。

### 指标

```text
step_ratio_q50/q90/q99
memory_ratio_peak
accepted events per step
controller feature compute ms
payload apply ms
kernel launch count
sync count
empty-step overhead
```

### 通过标准

$$
step\_ratio_{q90}\le1.50
$$

$$
memory\_ratio\le1.05
$$

强目标：

$$
step\_ratio_{q90}\le1.20
$$

---

## 12. P8：Paired Replay Boundary

### 目标

只有 runtime 过线后，才进入 paired replay。

### 对照分支

```text
RealFunctional
AdamWParallel
bestLR
NoOp
RandomPayload
ShuffledFunctionalPayload
```

### 通过标准

```text
RealFunctional beats all listed controls；
shuffled payload fail；
leave-dataset-out pass；
leave-stratum-out pass；
leave-template-out pass；
leave-family-out pass。
```

---

## 13. 全局 Stop / Pivot 规则

### Stop C 线

如果 5000 panel 已满足：

$$
UCB(CoreLike)<0.03
$$

且：

$$
UCB(PathGood)<0.03
$$

则判定自然动作源密度不足，不必等待 20000。

### Pivot B 线

如果 B 线连续两轮 proxy 都满足：

```text
precision < 0.50 或 V_LCB < 0
```

则停止 cheap proxy 搜索，转向更明确的 future-operator approximation。

### Stop D 线

如果 generated 64-action sandbox 满足：

```text
PathGood precision < 0.10
或 LongRisk_UCB > 0.50
或 RiskAdjustedAUV_LCB < 0
```

则不允许扩展到 512。

---

## 14. 最终判断口径

v9.9.1 不以“有没有跑很多行”为成功。成功只分以下几类。

### Case A：自然动作密度足够

```text
C 线 full panel 证明 CoreLike / PathGood LCB >= 0.03；
然后进入 existing-action controller。
```

### Case B：自然动作密度不足

```text
C 线 full panel 证明 CoreLike / PathGood UCB < 0.03；
停止 existing-action harvesting，转向 generated update。
```

### Case C：密度足够但 legal proxy 不够

```text
说明好动作存在，但训练当下识别仍失败；
继续 B 线，不打开 controller。
```

### Case D：B 线 proxy 过线

```text
允许 minimal controller + runtime。
```

### Case E：A/B/C 都不过

```text
说明当前 AP0 natural stream、future proxy 和 generated route都不足；
需要回到更根本的 optimizer rule 设计。
```

---

## 15. 本轮一句话目标

$$
\boxed{
\text{v9.9.1 要把 v9.9.0 的工程 smoke 推进为真实密度裁决，并寻找低成本未来路径算子；如果密度或机制不过，就明确停止对应路线。}
}
$$
