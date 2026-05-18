# DG-KAN v9.9.0 自然动作扩流工程门 / FuturePath Operator / 四线并行完整实验计划

> 本计划基于 v9.8.9 的真实执行结果制定。v9.8.9 没有进入新的 science full run，而是在 P1a 阶段确认 `natural_extension_generator_missing`，随后按 fail-fast 停止 P1b/P1c/P1d/P2/P3/P4/P5/P6/P7/P8。这个结果不是 functional success，也不是未来路径路线失败；它说明下一步必须优先落地自然 AP0 extension action generator / materializer，否则 density、controller、generated sandbox 都没有实验对象。

---

## 0. 一句话目标

v9.9.0 的目标不是继续调 `LC8`、`AUV threshold`、`CoreExpansion10` 或 APG 变体，而是先把四线的实验对象补齐：

$$
\boxed{
\text{先生成真实的新自然 AP0 actions，再判断好路径动作密度是否足够，并寻找训练当下可用的 future-path proxy。}
}
$$

本轮的最低有效推进不是 system pass，而是完成下面三件事：

```text
1. C 线自然动作扩流工程门闭合：至少 1 / 16 / 256 个新自然 AP0 actions 完整通过 lifecycle + branch-horizon replay。
2. A 线 future path 类型继续拆解：确认 CoreLike / SlowBurnGood / RiskyHighAUV 等路径类型在新自然动作中是否出现。
3. B 线训练当下 proxy 重写：不再调 cheap response threshold，而是验证 low-cost FuturePathOperator sketch 是否能预测未来路径。
```

只有当 A/B/C 至少一条给出可操作证据，D 线才允许打开最多 64-action generated sandbox。

---

## 1. 当前状态的独立判断

v9.8.9 的 route 是：

```text
route = R-C1-NaturalGeneratorEntryMissing
primary_blocker = natural_extension_generator_missing
secondary_blocker = science_full_run_stopped_by_fail_fast
route_recommendation = engineering_only_materializer_task
system_legal_controller_pass = 0
generated_route_status = stopped_P1a_generator_missing
```

这轮最重要的事实不是 “P3/P4/P5 都没跑”，而是：

```text
P1a entrypoint_count = 0
natural_extension_action_generator_found = 0
P1b/P1c/P1d = not_run
P2 5000/10000/20000 density panel = not_run
P3 future path decomposition = not_run
P4 FuturePathOperator Sketch = not_run
controller/runtime/paired replay = not_run
no fake/proxy/cpu = 0/0/0
```

因此，v9.8.9 不能被解读成：

```text
future path 失败；
自然动作密度不足；
low-cost future operator 已经被充分证伪；
generated route 永久失败。
```

它只能说明：

$$
\boxed{
\text{核心实验对象缺失：没有新的自然 AP0 action generator，就无法判断 density 或 controller。}
}
$$

---

## 2. 四线当前判断

### 2.1 A 线：未来训练路径

A 线是目前最有科学价值的线。过去几轮已经反复看到：Core77、OldOnly、Core+RiskClean 这类动作不是简单的当前一步 loss descent，而是会把后续训练带到更好路径。

但 v9.8.9 没有继续跑 A 线，因为 P1a fail-fast 停止。A 线当前状态是：

```text
future path phenomenon = 有可信诊断信号
future path controller = 还没有
future path legal mechanism = 还没有
future path density in new natural actions = 不知道
```

A 线下一步不能只看 AUV。必须同时看：

```text
V1 / V5 / V20 / V80 / V240；
RAUV；
delayed gain；
longrisk path；
bad/null path；
memory/offdiag path；
hard-tail path；
LDO/LSO/LTO/LFO；
runtime cost。
```

### 2.2 B 线：训练当下合法机制

B 线的旧做法已经失败：

```text
FO1-FO8: 有些太贵，且 value 负；
LC1/LC8: 很便宜，但 precision 和 V 都崩；
memory/hard-tail cheap response: 能清风险，不能找收益。
```

所以 v9.9.0 不再做 “cheap proxy threshold sweep”。B 线要改成 FuturePathOperator sketch：用训练当下可算的小计算，近似动作对未来路径的影响，而不是近似当前函数响应大小。

核心假设是：

$$
H_B:
\text{存在一个低成本训练当下算子 } S_{FPO}(a),
\text{能预测 } RAUV(a),V_{240}(a),LongRisk(a).
$$

### 2.3 C 线：自然 AP0 动作扩流

C 线现在是最高优先级。现有 AP0 panel 只有 2876 个 labeled actions：

```text
CoreLike count = 77
CoreLike rate = 0.02677
Wilson LCB/UCB = 0.02148 / 0.03333
```

这个区间刚好跨过 $0.03$，所以不能判断：

```text
自然动作流扩大后，CoreLike/PathGood 密度是否够；
Core77 只有 77 个，是因为动作池太小，还是自然 AP0 源本身密度不足。
```

C 线必须回答：

$$
\boxed{
\hat p_{good}(N)=\frac{\#\text{good natural AP0 actions}}{N}
}
$$

随着 $N=5000,10000,20000$ 是否稳定超过 $0.03$。

### 2.4 D 线：生成新动作

D 线继续停止，除非 A/B/C 给出明确证据。

原因很简单：过去 APG/APGA/APGF/APGT 等多轮已经证明，工程能生成 payload、branch-horizon 能完整跑，但新动作经常 value-negative / high-longrisk。没有新的 future-path 目标或 density 结论之前，重开 generated route 只会增加人工设计复杂度。

D 线的重新打开条件是：

```text
条件 1：C 线证明自然 AP0 density 不足，必须生成新动作；
或条件 2：B 线找到 legal low-cost FuturePathOperator sketch，可作为生成目标；
或条件 3：A 线给出清晰 future-path mechanism，可转成 64-action sandbox objective。
```

---

## 3. 本轮不做什么

v9.9.0 明确禁止：

```text
1. 不继续调 LC8 / LC1 threshold。
2. 不继续调 AUV / RAUV 单阈值。
3. 不用 support-aware gate 直接替换 official raw gate。
4. 不在没有 natural extension generator 时讨论 5000/10000/20000 density 成功或失败。
5. 不把旧 2876 AP0 replay 当成新自然动作扩流。
6. 不把 existing AP0 preflight materializer 当成 natural extension generator。
7. 不因为 A 线 weak signal 存在就打开 controller。
8. 不因为 B 线 proxy 便宜就打开 controller。
9. 不在 B/C 不过时重开 generated sandbox。
10. 不按 dataset 写 selector 或 threshold。
11. 不使用 future outcome、old table、GradeAB、RAUV、V240 等作为 commit-time official feature。
12. 不使用 fake/proxy/cpu-offload rows。
```

---

## 4. v9.9.0 总体实验结构

v9.9.0 分成四条线并行执行，但 C 线是硬前置门。

```text
P0: 复现 v9.8.9 boundary。
P1: C 线 natural AP0 extension generator / materializer 工程门。
P2: C 线自然动作 density panels。
P3: A 线 future path 类型分解。
P4: B 线 FuturePathOperator sketch。
P5: A/B/C 综合 gate 与 existing-action controller boundary。
P6: D 线 generated sandbox reopen gate。
P7: selected runtime boundary。
P8: paired replay / short-full boundary。
```

---

## 5. P0：复现 v9.8.9 边界

### 5.1 目标

确认本轮输入状态没有漂移：

```text
source route = R-C1-NaturalGeneratorEntryMissing
P1a entrypoint = 0
PanelA density inconclusive
controller = not_run
generated sandbox = stopped
no fake/proxy/cpu = 0/0/0
```

### 5.2 记录指标

```text
source_route
source_artifact_hash
PanelA_action_count
PanelA_CoreLike_count
PanelA_CoreLike_rate
PanelA_CoreLike_LCB
PanelA_CoreLike_UCB
P1a_entrypoint_count
system_legal_controller_pass
generated_route_status
fake/proxy/cpu count
```

### 5.3 通过标准

P0 必须满足：

$$
PanelA\_action\_count=2876
$$

$$
P1a\_entrypoint\_count=0
$$

$$
fake=proxy=cpu=0.
$$

若 P0 不一致，本轮停止，先修 artifact lineage。

---

## 6. P1：C 线自然 AP0 extension generator / materializer 工程门

### 6.1 目标

落地真实新自然 AP0 action generator。它必须产生新的自然 AP0 actions，而不是复制旧 2876 个 action。

建议新增模块：

```text
experiments/natural_ap0_extension_materializer.py
```

至少暴露函数：

```python
def generate_natural_ap0_extension_actions(
    seed: int,
    data_root: str,
    target_action_count: int,
    cursor: str | None,
    device: str = "auto",
    profile: str = "full-gated",
) -> NaturalAP0ActionBatch:
    ...
```

### 6.2 生成器必须满足的定义

一个新自然 AP0 action 必须来自同一套训练流和 AP0 action lifecycle：

```text
1. 使用合法训练当下 state；
2. 使用当前 train batch / train-memory / optimizer state；
3. 不使用 future outcome；
4. 不使用 validation/test；
5. 不使用 dataset_name 分支；
6. 不复用旧 action_id；
7. 不复用旧 payload_hash；
8. 不把旧 action row 复制成新 action；
9. payload 可以由现有 AP0 rule 生成，但必须对应新的 state/action lifecycle；
10. 每个 action 必须能进入 action apply replay 和 branch-horizon replay。
```

### 6.3 输出 schema

每个新 action 必须写入：

```text
action_id
candidate_id
event_id
seed
step_idx
family_id
stratum_id
template_id
state_before_hash
optimizer_state_hash
batch_sequence_hash
payload_hash
payload_tensor_path
payload_norm
payload_linf
action_norm
action_adamw_cosine
provenance = natural_ap0_extension
source_cursor
created_at
```

严禁字段：

```text
dataset_name_as_selector
future outcome label
GradeAB
CoreLike
PathGood
RAUV
V240
longrisk label
validation/test metric
old table label
```

### 6.4 分层 smoke

P1 必须按以下顺序执行，不能跳级。

#### P1a：entrypoint audit

```text
查找 natural_ap0_extension_materializer.py
查找 generate_natural_ap0_extension_actions
查找 schema writer
查找 action apply hook
查找 branch-horizon hook
```

通过标准：

```text
entrypoint_count >= 1
all required functions importable
signature match = 1
```

若 P1a fail，立即停止；不得进入 P2/P3/P4/P5/P6。

#### P1b：single-action smoke

生成 1 个新 action，跑完整 lifecycle。

记录：

```text
unique action_id
old_action_id_collision_count
payload_hash_missing
payload_tensor_written
action_apply_linf_max
action_apply_relative_max
no_transform_sanity
branch_horizon_rows for h1/h5/h20/h80/h240
fake/proxy/cpu
```

通过标准：

$$
old\_action\_collision=0
$$

$$
action\_apply\_linf\_max \le 10^{-7}
$$

$$
branch\_horizon\_completion=1.0
$$

$$
fake=proxy=cpu=0.
$$

#### P1c：16-action smoke

生成 16 个新 actions，检测小规模多样性。

记录：

```text
action_count = 16
unique action_id count
unique payload_hash count
unique template count
unique family count
branch-horizon expected/actual
label exclusivity violation
duplicate row id
NaN/Inf
rows/sec
peak GPU memory
```

通过标准：

```text
action_count = 16
unique_action_id = 16
completion = 1.0
label_exclusivity_violation = 0
duplicate_row_id = 0
NaN/Inf = 0
```

#### P1d：256-action smoke

生成 256 个新 actions，检查吞吐、显存、分布、失败模式。

记录：

```text
action_count = 256
branch_horizon_rows = 256 * branch_count * horizon_count
rows/sec
wallclock
peak GPU memory
exception count
exception type
payload norm distribution
action norm distribution
action-adamw cosine distribution
template diversity
family diversity
old payload hash collision count
```

通过标准：

```text
completion = 1.0
unresolved_exception_count = 0
old_payload_hash_collision_count = 0
old_action_id_collision_count = 0
rows/sec recorded
peak GPU memory recorded
```

---

## 7. P2：C 线自然动作 density panels

### 7.1 目标

用真实新自然 AP0 actions 判断好动作密度是否足够。

要跑 panels：

```text
PanelA: existing 2876 reference
PanelB: 5000 total natural actions
PanelC: 10000 total natural actions
PanelD: 20000 total natural actions
```

如果成本太高，必须先跑：

```text
PanelB-smoke: 1024 total natural actions
```

但不能用 smoke 替代 final density 判定。

### 7.2 标签定义

本轮至少记录四类标签。

#### CoreLike

CoreLike 是目前最干净的旧定义，保持 continuity：

```text
GradeAB positive；
V LCB > 0；
longrisk UCB <= 0.05；
bad UCB <= 0.05；
null UCB <= 0.15；
memory/offdiag clean。
```

#### PathGood

PathGood 更贴合未来路径：

$$
RAUV^{LCB}>0
$$

$$
V_{240}^{LCB}>0
$$

$$
LongRisk^{UCB}\le 0.05
$$

$$
MemoryFail=0,\quad OffdiagFail=0.
$$

#### SlowBurnGood

SlowBurnGood 是 “当前一步不强，但未来变好” 的动作：

$$
V_1^{LCB}\le 0
$$

$$
RAUV^{LCB}>0
$$

$$
V_{80}^{LCB}>0 \text{ or } V_{240}^{LCB}>0
$$

$$
LongRisk^{UCB}\le 0.05.
$$

#### RiskyHighAUV

RiskyHighAUV 是负控：

$$
RAUV^{LCB}>0
$$

但：

$$
LongRisk^{UCB}>0.20 \text{ or } MemoryFail=1 \text{ or } OffdiagFail=1.
$$

### 7.3 记录指标

每个 panel 记录：

```text
action_count
CoreLike_count / rate / Wilson LCB / UCB
PathGood_count / rate / Wilson LCB / UCB
SlowBurnGood_count / rate / Wilson LCB / UCB
RiskyHighAUV_count / rate / Wilson LCB / UCB
per-dataset rate
per-family rate
per-template rate
per-step-bucket rate
unique template count
max template share
LDO/LSO/LTO/LFO drop
branch-horizon completion
rows/sec
wallclock
peak GPU memory
```

### 7.4 判断标准

C 线 density sufficient：

$$
LCB(p_{CoreLike})\ge 0.03
$$

或：

$$
LCB(p_{PathGood})\ge 0.03
$$

同时：

$$
max\_template\_share\le 0.25
$$

$$
LDO,LSO,LTO,LFO\le 0.10.
$$

C 线 density insufficient：

$$
UCB(p_{CoreLike})<0.03
$$

且：

$$
UCB(p_{PathGood})<0.03
$$

在至少 10000 actions 后仍成立。

若 PanelB/PanelC/PanelD 仍 inconclusive，扩展到 50000 前必须先做成本评估。

### 7.5 可视化

必须画：

```text
1. density vs action_count 曲线，带 Wilson CI；
2. CoreLike / PathGood / SlowBurnGood / RiskyHighAUV 分组柱状图；
3. per-dataset rate heatmap；
4. per-template support histogram；
5. max-template-share 随 panel size 变化；
6. rows/sec 与 peak memory 曲线。
```

---

## 8. P3：A 线未来路径机制分解

### 8.1 目标

不再只看 AUV，拆清不同动作类型的未来路径形态。

比较组：

```text
Core77
CoreExpansion10
Core77+RiskCleanButLowValueTop10
OldOnly
ExactOnly
RandomMatched
NaturalNewCoreLike
NaturalNewSlowBurnGood
NaturalNewRiskyHighAUV
```

### 8.2 记录指标

每个 action，每个 horizon $h\in\{1,5,20,80,240\}$ 记录：

```text
CE_delta_h
NLL_delta_h
margin_delta_h
V_h
V_h_LCB
longrisk_h
bad_h
null_h
memory_fail_h
offdiag_fail_h
hard_tail_CEp99_delta_h
old_family_loss_delta_h
old_stratum_margin_delta_h
cover_entropy_delta_h
basis_effective_rank_delta_h
curvature_proxy_delta_h
RAUV
RiskAdjustedAUV
delayed_gain = V240 - V1
risk_path_integral
```

### 8.3 判断标准

A 线 weak pass：

```text
Core77 或 NaturalNewCoreLike 的 RAUV_LCB > 0；
V240_LCB > 0；
longrisk_UCB <= 0.05；
memory/offdiag = 0。
```

A 线 mechanism pass：

```text
OldOnly / SlowBurnGood 显示 V1 <= 0 但 RAUV > 0 或 V240 > 0；
ExactOnly 或 RandomMatched 显示 immediate/response 高但 longrisk/memory/offdiag 失败；
未来路径类型与好坏标签有清晰分离。
```

A 线 strong pass：

```text
存在 >= 87 个 actions 的 PathGood 或 CoreLike accepted region；
RAUV_LCB > 0；
V240_LCB > 0；
longrisk/bad/null/memory/offdiag 全部过线；
LDO/LSO/LTO/LFO <= 0.10。
```

### 8.4 可视化

必须画：

```text
1. group × horizon 的 V 曲线；
2. group × horizon 的 longrisk 曲线；
3. group × horizon 的 memory/offdiag 曲线；
4. RAUV vs longrisk 散点；
5. delayed_gain vs risk_path_integral 散点；
6. Core77 / OldOnly / ExactOnly / NaturalNewCoreLike 的路径雷达图。
```

---

## 9. P4：B 线 FuturePathOperator sketch

### 9.1 目标

构造低成本、训练当下合法的 future-path 代理。它不是用 outcome label，也不是用 future rows，而是在 commit time 做小计算，预测动作是否可能改善未来路径。

### 9.2 候选 sketch

#### FPO1：tiny virtual AdamW sketch

在当前 batch + memory microbatch 上模拟 $k=1,3,5$ 个极小 AdamW-style virtual steps，但不进入真实训练状态。

记录：

```text
virtual_V1
virtual_V3
virtual_V5
virtual_delayed_gain
virtual_memory_harm
cost_ms
```

#### FPO2：JVP/VJP path sketch

用 JVP/VJP 估计 action 对 output-space Jacobian 和 memory response 的影响。

记录：

```text
JVP_current_response
JVP_memory_response
JVP_hard_tail_response
VJP_signal_alignment
operator_norm_proxy
cost_ms
```

#### FPO3：signal-reservoir sketch

用 per-example response 的均值/方差判断动作是否进入 signal channel，而不是 noise/reservoir。

记录：

$$
SNR_\Delta = \frac{\mu_\Delta^2}{\sigma_\Delta^2+\epsilon}
$$

以及：

```text
old_family_signal
hard_tail_signal
memory_signal
reservoir_leak_proxy
```

#### FPO4：risk-adjusted future sketch

组合 delayed-gain proxy 和 hard safety veto，但不做手工加权大 score。

接受形式是硬门：

$$
GainProxy(a)>0
$$

$$
MemoryHarmProxy(a)\le \tau_M
$$

$$
OffdiagProxy(a)\le \tau_O
$$

$$
Cost(a)\le C_{max}.
$$

### 9.3 记录指标

每个 sketch 记录：

```text
TopK87 precision
TopK87 V_LCB
TopK87 RAUV_LCB
TopK87 V240_LCB
TopK87 longrisk_UCB
bad/null/memory/offdiag UCB
LDO/LSO/LTO/LFO
cost q50/q90/q99
correlation with RAUV
correlation with V240
correlation with longrisk
rank overlap with Core77 / OldOnly / NaturalNewCoreLike
```

### 9.4 判断标准

B 线 weak pass：

$$
precision_{TopK87}\ge 0.60
$$

$$
V_{LCB}>0
$$

$$
LongRisk_{UCB}\le 0.10
$$

$$
cost_{q90}\le 1.0\text{ ms}.
$$

B 线 strong pass：

$$
precision_{TopK87}\ge 0.75
$$

$$
V_{LCB}>0
$$

$$
RAUV_{LCB}>0
$$

$$
LongRisk_{UCB}\le 0.05
$$

$$
Bad_{UCB}\le 0.05
$$

$$
Null_{UCB}\le 0.15
$$

$$
LDO,LSO,LTO,LFO\le 0.10
$$

$$
cost_{q90}\le 1.0\text{ ms}.
$$

如果 B 线所有 sketch 都只做到低 risk 但 V 负，则停止 B 线阈值扫，转向理论推导或 C/D 线。

---

## 10. P5：existing-action controller boundary

### 10.1 打开条件

只有满足以下任一条件才打开 P5：

```text
C 线 density sufficient；
或 A 线 strong pass；
或 B 线 strong pass。
```

### 10.2 controller 形式

controller 不能使用 dataset_name，也不能使用 future outcome。候选形式：

```text
Controller-1: natural CoreLike harvesting + safety veto
Controller-2: FPO sketch TopK87 + memory/offdiag veto
Controller-3: PathGood density-based accepted region
Controller-4: Core77 + NaturalNewSlowBurnGood hybrid
```

### 10.3 official gates

必须满足：

$$
accepted\_count \ge 87
$$

$$
precision \ge 0.75
$$

$$
V_{LCB}>0
$$

$$
LongRisk_{UCB}\le 0.05
$$

$$
Bad_{UCB}\le 0.05
$$

$$
Null_{UCB}\le 0.15
$$

$$
MemoryFail_{UCB}\le 0.05
$$

$$
OffdiagFail_{UCB}\le 0.05
$$

$$
LDO,LSO,LTO,LFO\le 0.10.
$$

若 P5 不过，不打开 runtime。

---

## 11. P6：generated route reopen gate

### 11.1 打开条件

只有以下情况允许 generated sandbox：

```text
C 线 density insufficient：自然动作源不够；
或 B 线 strong pass：有可生成目标；
或 A 线 mechanism pass：明确未来路径机制。
```

### 11.2 sandbox 限制

第一次只允许：

```text
64 generated actions
h1/h5/h20/h80/h240 branch-horizon replay
no fake/proxy/cpu
action apply replay
negative controls
```

不得直接跑 512/1024 generated actions。

### 11.3 判断标准

generated sandbox weak pass：

$$
PathGood\ precision\ge0.30
$$

$$
RAUV_{LCB}>0
$$

$$
LongRisk_{UCB}\le0.20
$$

强 pass：

$$
PathGood\ precision\ge0.60
$$

$$
RAUV_{LCB}>0
$$

$$
V240_{LCB}>0
$$

$$
LongRisk_{UCB}\le0.05.
$$

如果 generated sandbox 再次 value-negative / high-longrisk，则 D 线继续停止。

---

## 12. P7：selected runtime boundary

只有 P5 或 P6 过线后才测 runtime。

记录：

```text
controller feature cost q50/q90/q99
candidate generation cost
action apply cost
kernel launches per active step
empty-step kernels
step_ratio_q90
memory_ratio
selected action count per step
```

通过标准：

$$
step\_ratio_{q90}\le1.50
$$

$$
memory\_ratio\le1.05
$$

$$
empty\_step\_kernel\_fraction\le0.05.
$$

---

## 13. P8：paired replay / short-full boundary

只有 P7 过线后才打开。

必须比较：

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
CertificateNoPayload
```

通过标准：

```text
RealFunctional beats AdamWParallel / bestLR / NoOp / Random；
ShuffledPayload fails；
CertificateNoPayload fails；
LDO/LSO/LTO/LFO stable；
short-run 不出现 catastrophic fail。
```

---

## 14. 并行执行策略

为了避免继续“完整 runner 后才发现 P1a 缺失”，v9.9.0 必须按 fail-fast 并行执行：

```text
Day 0 / First run:
  P0 + P1a only。
  如果 P1a fail，不跑任何 science stage。

P1a pass 后：
  P1b single-action smoke。
  A 线可以继续用旧 landed rows 做 P3 mechanism analysis。
  B 线可以继续在旧 landed rows 上做 sketch prototype。

P1b pass 后：
  P1c 16-action smoke。

P1c pass 后：
  P1d 256-action smoke。

P1d pass 后：
  P2 PanelB 5000。
  同时 P3/P4 用新 actions + old actions 合并分析。
```

任何阶段 fail，都写明确 route，不继续跑后续。

---

## 15. 最终 route 决策表

| 条件 | route | 下一步 |
|---|---|---|
| P1a fail | `R-C1-NaturalGeneratorEntryMissing` | engineering-only materializer task |
| P1b/P1c/P1d fail | `R-C2-NaturalMaterializerSmokeFail` | 修 action lifecycle / apply / branch-horizon |
| C density sufficient, B fail | `R-C3-NaturalDensityEnoughButProxyAbsent` | existing-action harvesting + better proxy |
| C density insufficient, B fail | `R-C4-NaturalSourceInsufficientProxyAbsent` | 重设 generated objective |
| A mechanism pass, B fail | `R-A1-FuturePathMechanismFoundProxyAbsent` | 研究 future operator proxy |
| B strong pass | `R-B1-LegalFutureOperatorProxyFound` | existing-action controller |
| P5 controller pass | `R-S1-ControllerReadyRuntimeNext` | selected runtime |
| P7 runtime pass | `R-S2-SystemReadyPairedReplayNext` | paired replay |
| P8 pass | `R-S3-StrictFunctionalCandidate` | short/full validation |

---

## 16. 本轮成功的最低标准

v9.9.0 不要求 functional success。最低有效成功是：

```text
1. P1a/P1b/P1c/P1d 全过，真实新自然 AP0 actions 可生成可回放；
2. 至少跑完 256-action natural extension smoke；
3. 写出 density 初步 estimate；
4. P3/P4 不把 diagnostic 写成 official；
5. generated route 只有满足条件才打开。
```

强推进标准是：

```text
1. PanelB 5000 完成；
2. CoreLike 或 PathGood density LCB >= 0.03；
3. 或证明 UCB < 0.03，明确自然 AP0 源不足；
4. B 线出现 TopK87 precision >= 0.60 且 cost q90 <= 1ms 的 proxy；
5. P5 controller 至少进入 candidate boundary。
```

---

## 17. 最重要的反思

当前项目不是没有好动作，也不是 future path 没信号。真正的问题是：

$$
\boxed{
\text{我们还没有把“未来好路径”转成训练当下合法、低成本、跨分布稳定的选择或生成规则。}
}
$$

v9.9.0 的核心任务不是再调分数，而是补齐实验对象和裁决条件：

```text
C 线回答：自然动作源够不够？
A 线回答：好路径具体长什么样？
B 线回答：训练当下能不能低成本预测它？
D 线回答：如果自然动作源不够，能不能生成它？
```

只有这四个问题至少有一个闭合，项目才应该继续向 controller/runtime/paired replay 推进。
