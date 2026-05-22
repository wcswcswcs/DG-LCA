# DG-KAN v10.3 Update-Rule Theory Rebuild / Future-Path Adjoint Update 完整实验计划

> 本计划基于 v10.2 `R4-BDDiscoveryFail_UpdateRuleTheoryRebuild` 的真实结果制定。v10.3 不再小修 FPO13、GUP13、AUV/RAUV 阈值、natural density panel 或 generated family 名字。v10.3 的目标是把项目从“采样动作再验证”改成“直接求解能改善未来训练路径的参数更新”。
>
> 本文公式均使用 `$...$` 或 `$$...$$`，Typora 友好。

---

## 0. 当前判断

v10.2 的 route 是：

```text
route = R4-BDDiscoveryFail_UpdateRuleTheoryRebuild
primary_blocker = natural_density_insufficient
secondary_blocker = FPO13_and_GUP13_generated_discovery_failed
system_legal_controller_pass = 0
generated_route_status = stopped_update_rule_discovery_reset
```

v10.2 的关键结果：

```text
P1 Fast / Slow / Risky / SafeLow / Bad = 12 / 48 / 842 / 60 / 34286
P2 FPO13 weak / discovery / controller = 0 / 0 / 0
best FPO13D TopK64 precision = 0.015625
best FPO13D V240 LCB = -0.3264349469551365
P3 natural density close / retain / inconclusive = 1 / 0 / 0
C3 20000 run = 0
P4 GUP13 Stage8 / Stage32 / Stage64 opened = 8 / 0 / 0
P4 generated discovery pass = 0
P5 dominant failure = F2-FPO-measures-current-response-not-future-path
P6/P7/P8 controller/runtime/paired replay = not_run / not_run / not_run
No-fake rows checked = 37410
fake/proxy/cpu = 0 / 0 / 0
```

独立判断：

$$
\boxed{
\text{自然动作池已降级，FPO 识别失败，GUP 生成失败；下一步必须重建 update rule theory。}
}
$$

这里的重点不是再找一个更好的动作 family，而是承认当前路线已经暴露出结构性问题：

```text
1. 自然 AP0 action source 的好路径动作密度太低。
2. FPO 系列不能在训练当下识别 future-path-improving 动作。
3. GUP13 只打开 Stage8，且不能产生可扩展的 FastGood / SlowBurnGood。
4. 当前 generated route 仍像“合法 payload 生成器”，不是“未来路径优化器”。
```

---

## 1. 核心路线重置

v10.3 不再以 “sample action -> evaluate action” 为主线，而改成：

$$
\boxed{
\text{solve update -> verify update}
}
$$

以前的流程：

```text
1. 人工设计一批动作；
2. 跑 branch-horizon；
3. 事后找哪些动作好；
4. 再尝试找 commit-time signal 解释它们。
```

这个流程已经多次失败。新流程：

```text
1. 先定义未来训练路径目标；
2. 在低维 update 子空间里直接求一个 delta theta；
3. 对这个 delta 做 8-action / 32-action / 64-action discovery branch-horizon；
4. 只有它产生 FastGood / SlowBurnGood，才进入 controller / runtime / paired replay。
```

---

## 2. 新核心假设：Future-Path Adjoint Update

设当前参数为 $\theta_t$，基础训练更新为手写 AdamW-equivalent 更新 $U$。没有 functional update 时：

$$
\theta_{t+h}^{0}=U_h(\theta_t).
$$

加入一个小参数扰动 $\Delta$ 后：

$$
\theta_{t+h}^{\Delta}=U_h(\theta_t+\Delta).
$$

我们真正关心的不是一步 loss 下降，而是未来路径收益：

$$
V_H(\Delta)=
\sum_{h\in H}
\operatorname{LCB}
\left(
M_h(\theta_{t+h}^{0})-M_h(\theta_{t+h}^{\Delta})
\right),
$$

其中：

$$
H=\{20,80,240\}.
$$

同时必须满足安全硬门：

$$
LongRisk_H(\Delta)\le \tau_L,
$$

$$
MemoryHarm_H(\Delta)\le \tau_M,
$$

$$
OffdiagRisk_H(\Delta)\le \tau_O,
$$

$$
Bad_H(\Delta)\le \tau_B,
$$

$$
Null_H(\Delta)\le \tau_N,
$$

$$
Cost(\Delta)\le C_{max}.
$$

v10.3 的核心目标是近似求解：

$$
\Delta^*=
\arg\max_{\Delta\in\mathcal{S}}
V_H(\Delta)
$$

subject to safety constraints.

这里 $\mathcal{S}$ 是训练当下可构造的低维 update 子空间，不是手工 APG / GUP family 名字。

---

## 3. 为什么是 adjoint / co-state

未来路径目标的一阶近似是：

$$
V_H(\Delta)\approx \langle p_t,\Delta\rangle.
$$

其中 $p_t$ 是未来路径目标对当前参数的 adjoint：

$$
p_t=
\nabla_{\theta_t}
\left[
\sum_{h\in H}M_h(U_h(\theta_t))
\right].
$$

展开为：

$$
p_t=
\sum_{h\in H}
D U_h(\theta_t)^\top
\nabla_{\theta_{t+h}}M_h.
$$

这就是 v10.3 和 FPO/transfer 的本质差异。

FPO / exact transfer 主要在看：

$$
-g_t^\top\Delta.
$$

而 FPAU 要近似的是：

$$
\sum_{h\in H}
D U_h(\theta_t)^\top
\nabla_{\theta_{t+h}}M_h.
$$

也就是说：

```text
不是当前点的响应；
而是这个扰动经过后续训练动力学传播后，对未来路径的影响。
```

---

## 4. v10.3 总体实验结构

v10.3 分成三类 runner，避免继续“一轮只发现一个 blocker”。

### 4.1 Engineering Gate Runner

只验证工程入口，不做科学结论。

目标：

```text
1. FPAU 子空间方向能否构造；
2. 低维 alpha 求解是否数值稳定；
3. 8-action generated discovery 是否能完整 branch-horizon；
4. 无 fake / proxy / CPU offload；
5. action apply error = 0。
```

### 4.2 Science Discovery Runner

允许非 official 机制探索，但明确不能写成 success。

目标：

```text
1. 验证 FPAU score 是否能解释已有 path type；
2. 生成 8 / 32 / 64 个 FPAU actions；
3. 判断是否出现 FastGood / SlowBurnGood；
4. 不打开 controller / runtime / paired replay。
```

### 4.3 Official Gate Runner

只有 discovery 出现强机制后才允许打开。

目标：

```text
1. accepted_count >= 87；
2. precision / V / V240 / risk / memory / offdiag 同时过线；
3. LDO / LSO / LTO / LFO 稳定；
4. selected runtime step_ratio_q90 <= 1.50；
5. official paired replay beats AdamWParallel / bestLR / NoOp / Random。
```

---

## 5. P0：复现 v10.2 boundary

### 目标

确认本轮不是在错误边界上继续推进。

### 必须记录

```text
source_route_v1020
P1_Fast_count
P1_Slow_count
P1_Risky_count
P1_SafeLow_count
P1_Bad_count
FPO13_best_id
FPO13_best_precision
FPO13_best_V240_LCB
GUP13_stage8_opened_count
GUP13_stage32_opened_count
GUP13_stage64_opened_count
controller_status
runtime_status
paired_replay_status
fake/proxy/cpu
```

### 判断标准

P0 pass：

```text
source route == R4-BDDiscoveryFail_UpdateRuleTheoryRebuild
FPO13 discovery pass == 0
GUP13 discovery pass == 0
controller/runtime/paired replay all not_run
fake/proxy/cpu == 0/0/0
```

### 不满足条件时 Codex 先尝试

```text
如果 source route 不匹配：
  先检查 artifact path / run_manifest / route_decision 是否读取错。

如果 fake/proxy/cpu 非 0：
  停止本轮，不进入 P1；先修 no-fake audit。

如果 v10.2 artifacts 缺失：
  不补造；只允许重新跑 v10.2 boundary lock。
```

---

## 6. P1：Path type 固化

### 目标

把 future path target 固定为 path type，而不是单个 AUV / RAUV 分数。

### Path type 定义

#### FastGood

```text
V1_LCB > 0
V20_LCB > 0
V80_LCB > 0
V240_LCB > 0
LongRisk_UCB <= 0.05
MemoryFail_UCB <= 0.05
OffdiagFail_UCB <= 0.05
Bad_UCB <= 0.05
Null_UCB <= 0.15
```

#### SlowBurnGood

```text
V1_LCB <= 0 或 V1 不强
V20_LCB > 0
V80_LCB > 0
V240_LCB > 0
RAUV_LCB > 0
LongRisk_UCB <= 0.05
MemoryFail_UCB <= 0.05
OffdiagFail_UCB <= 0.05
Bad_UCB <= 0.05
Null_UCB <= 0.15
```

#### RiskyHighAUV

```text
RAUV_LCB > 0
但 LongRisk_UCB > 0.05 或 MemoryFail_UCB > 0.05 或 OffdiagFail_UCB > 0.05
```

#### SafeLowValue

```text
LongRisk_UCB <= 0.05
MemoryFail_UCB <= 0.05
OffdiagFail_UCB <= 0.05
但 V240_LCB <= 0 或 RAUV_LCB <= 0
```

#### BadPath

```text
V240_LCB < 0 或 LongRisk_UCB 高 或 Bad_UCB 高
```

### 必须记录

```text
action_id
source_family
source_dataset
source_template
V1_LCB
V5_LCB
V20_LCB
V80_LCB
V240_LCB
AUV_LCB
RAUV_LCB
LongRisk_UCB
Bad_UCB
Null_UCB
MemoryFail_UCB
OffdiagFail_UCB
path_type
```

### 判断标准

P1 discovery pass：

```text
FastGood_count + SlowBurnGood_count >= 64
SlowBurnGood_count >= 20
RiskyHighAUV_count >= 50
BadPath_count >= 1000
path_type_exclusivity_violation = 0
```

P1 controller pass 不允许直接通过；P1 只定义 discovery target。

### 可视化

```text
p1_path_type_count_bar.svg
p1_V1_to_V240_path_lines_by_type.svg
p1_RAUv_vs_longrisk_scatter.svg
p1_memory_offdiag_by_path_type_heatmap.svg
p1_dataset_family_template_distribution_by_path_type.svg
```

### 不满足条件时 Codex 先尝试

```text
如果 FastGood+SlowBurnGood < 64：
  不放宽 h240 longrisk；
  先检查 V1 条件是否过严；
  再尝试 SlowBurnGoodRelaxed，但必须保持 V240_LCB > 0 与 risk gate。

如果 RiskyHighAUV 太多：
  不调 RAUV；
  增加 risk-hard-gate 分析，把它作为 negative class。

如果 SafeLowValue 太多：
  检查 longer-horizon delayed gain；
  不能直接把 SafeLowValue 写成 good。

如果 path_type overlap：
  优先级按 BadPath > RiskyHighAUV > FastGood > SlowBurnGood > SafeLowValue 固化。
```

---

## 7. P2：FPAU adjoint 在已知动作上的解释力

### 目标

在生成新动作前，先验证 FPAU adjoint 是否能解释已有动作的 path type。

### 方法

对已有动作 $a$ 的扰动 $\Delta_a$，计算：

$$
S_{FPAU}(a)=\langle \hat p_t, \Delta_a\rangle.
$$

其中 $\hat p_t$ 是低成本近似 adjoint。

### Adjoint 近似候选

```text
A0: current-gradient baseline
A1: h1 virtual AdamW co-state
A2: h3 virtual AdamW co-state
A3: memory/hard-tail co-state
A4: old-family preserving co-state
A5: risk-adjusted co-state
A6: low-rank Krylov JVP/VJP co-state
A7: ensemble co-state with hard risk veto
```

### 必须记录

```text
action_id
path_type
S_FPAU_A0 ... S_FPAU_A7
rank_percentile
TopK64_path_precision
TopK87_path_precision
FastSlow_precision
RiskyHighAUV_rejection_rate
BadPath_rejection_rate
cost_ms_q50/q90/q99
```

### 判断标准

P2 weak pass：

```text
TopK64 FastSlow precision >= 0.10
RiskyHighAUV rejection rate >= 0.80
BadPath rejection rate >= 0.90
cost_q90 <= 5 ms
```

P2 discovery pass：

```text
TopK64 FastSlow precision >= 0.25
TopK87 V240_LCB > 0
LongRisk_UCB <= 0.10
cost_q90 <= 5 ms
```

P2 controller candidate pass：

```text
TopK87 FastSlow precision >= 0.50
V240_LCB > 0
LongRisk_UCB <= 0.05
MemoryFail_UCB <= 0.05
OffdiagFail_UCB <= 0.05
cost_q90 <= 1.5 ms
```

### 可视化

```text
p2_fpau_score_by_path_type_boxplot.svg
p2_fpau_topk_path_composition.svg
p2_risky_rejection_curve.svg
p2_cost_vs_precision_pareto.svg
p2_adjoints_correlation_heatmap.svg
```

### 不满足条件时 Codex 先尝试

```text
如果 A0 当前梯度与 A1/A2 差别很小：
  说明虚拟 horizon 太短；尝试 h5 low-rank co-state，但限制 samples。

如果 precision 低但 risk rejection 高：
  该 adjoint 只是 veto；加入 value co-state，不调 risk threshold。

如果 value 好但 longrisk 高：
  加 memory/offdiag hard gate；不要把 risk 放进加权分数。

如果 cost 高：
  缩小 virtual buffer；
  缓存 JVP/VJP；
  降低 Krylov rank；
  不允许进入 official controller。

如果所有 adjoint 都失败：
  停止 FPAU-generated P4；转 P3 synthetic mechanism check。
```

---

## 8. P3：机制可解性 toy / controlled check

### 目标

避免在复杂 dataset 上继续盲试。用可控任务判断 FPAU 是否能在已知 future-path 环境中求出正确 update。

这不是 dataset 调参，不参与 official controller。它是机制单元测试。

### 任务

```text
T1: 1D piecewise regression with delayed benefit
T2: 2D interaction regression
T3: old-family memory preservation toy
T4: noisy-label signal/reservoir toy
```

### 必须记录

```text
toy_id
update_method
FastGood_rate
SlowBurnGood_rate
RiskyHighAUV_rate
BadPath_rate
future_loss_auc
memory_loss_delta
curvature_proxy_delta
basis_cover_delta
cost_ms
```

### 对照

```text
AdamW-only
current-gradient delta
random delta
old GUP13 family
FPAU-A2
FPAU-A5
FPAU-A7
```

### 判断标准

P3 pass：

```text
FPAU 在至少 3/4 toy 上 future_loss_auc 优于 current-gradient delta；
SlowBurnGood_rate 高于 random 至少 3x；
RiskyHighAUV_rate 不高于 current-gradient；
Memory toy 上 memory_loss_delta <= 0。
```

### 可视化

```text
p3_toy_future_loss_curves.svg
p3_toy_path_type_distribution.svg
p3_memory_preservation_curve.svg
p3_signal_reservoir_split_plot.svg
```

### 不满足条件时 Codex 先尝试

```text
如果 toy 上 FPAU 也失败：
  说明 adjoint objective 或 subspace 错；不要进 P4。

如果只在 interaction toy 失败：
  增加 low-rank cross-feature direction；检查 base architecture 表达性。

如果 memory toy 失败：
  增加 old-family projection hard constraint，而不是加权惩罚。

如果 noisy-label toy 失败：
  增加 signal/reservoir drift-diffusion gate。
```

---

## 9. P4：FPAU-generated discovery staircase

### 目标

直接生成未来路径优化 update，而不是采样 APG/GUP family。

### Update 子空间

令：

$$
\Delta=B\alpha.
$$

候选 basis：

```text
b1: AdamW update direction
b2: AdamW momentum direction
b3: memory-safe projected gradient
b4: hard-tail correction direction
b5: old-family preserving direction
b6: signal-channel SNR direction
b7: offdiag-safe correction direction
b8-bk: low-rank Krylov / JVP directions
```

求解：

$$
\alpha^*=\arg\max_\alpha \langle \hat p_t, B\alpha\rangle
$$

subject to：

$$
\|B\alpha\|\le \epsilon,
$$

$$
MemoryHarm(B\alpha)\le \tau_M,
$$

$$
OffdiagRisk(B\alpha)\le \tau_O,
$$

$$
LongRiskProxy(B\alpha)\le \tau_L.
$$

### Staircase

```text
Stage 4: 4 actions, engineering smoke
Stage 8: 8 actions, discovery fail-fast
Stage 32: only if Stage8 has >= 1 Fast/Slow-like
Stage 64: only if Stage32 weak pass
Stage 256: not in v10.3 unless Stage64 discovery pass and cost acceptable
```

### 必须记录

```text
generated_action_id
basis_coefficients_alpha
subspace_basis_norms
predicted_FPAU_score
predicted_memory_harm
predicted_offdiag_risk
predicted_longrisk_proxy
payload_norm
action_apply_linf
branch_horizon_rows
path_type
V1/V5/V20/V80/V240
RAUV
longrisk
bad/null
memory/offdiag
new_positive_rate
longrisk_created_rate
```

### 判断标准

Stage8 weak pass：

```text
FastGood_count + SlowBurnGood_count >= 1
LongRisk_created_rate <= 0.25
V240_LCB not strongly negative
```

Stage32 weak pass：

```text
FastGood_count + SlowBurnGood_count >= 3
FastSlow_precision >= 0.10
V240_LCB > 0 or RAUV_LCB > 0
LongRisk_UCB <= 0.25
```

Stage64 discovery pass：

```text
FastGood_count + SlowBurnGood_count >= 8
FastSlow_precision >= 0.125
V240_LCB > 0
LongRisk_UCB <= 0.15
MemoryFail_UCB <= 0.15
OffdiagFail_UCB <= 0.15
```

### 可视化

```text
p4_generated_path_type_bar.svg
p4_generated_v_path_lines.svg
p4_alpha_coefficients_heatmap.svg
p4_predicted_vs_real_future_value.svg
p4_memory_offdiag_risk_scatter.svg
p4_stage_waterfall.svg
```

### 不满足条件时 Codex 先尝试

```text
如果 Stage4 action_apply_linf != 0：
  修 payload apply，不进入 Stage8。

如果 Stage8 全 BadPath：
  检查 alpha norm 是否过大；
  加 trust-region；
  检查是否与 AdamW 冲突；
  不扩大到 Stage32。

如果 Stage8 low-risk low-value：
  说明只有 safety constraint，没有 value objective；
  增强 value co-state，不放松 risk gate。

如果 Stage8 high-value high-risk：
  加 memory/offdiag hard constraint；
  不调 value score。

如果 Stage32 出现 Fast/Slow 但 LDO/family collapse：
  加 family/template diversity constraint。

如果 generated Fast/Slow 只在一个 dataset：
  不能按 dataset 调参；
  做 leave-dataset diagnostic，改子空间 basis。
```

---

## 10. P5：Failure attribution

### 目标

如果 FPAU / FPO / generated discovery 失败，必须归因，不允许只给 route 名称。

### Failure classes

```text
F1-path-label-too-sparse
F2-adjoint-current-response-collapse
F3-co-state-cost-too-high
F4-subspace-missing-value-direction
F5-subspace-violates-memory-offdiag
F6-trust-region-too-loose
F7-trust-region-too-tight
F8-generated-payload-OOD
F9-path-type-objective-wrong
F10-base-architecture-expressivity-limited
```

### 必须记录

```text
failure_class
assigned_count
assigned_fraction
evidence_fields
recommended_fix
```

### 判断标准

P5 pass：

```text
assigned_fraction >= 0.90
unknown_fraction <= 0.10
dominant_failure_class identified
```

### 不满足条件时 Codex 先尝试

```text
如果 unknown > 0.10：
  增加 per-action failure record；
  记录 alpha、norm、risk、path type、basis contributions。

如果 dominant = F4：
  增加 new basis directions，不调 selector。

如果 dominant = F5：
  加 hard constraints，不做 weighted penalty。

如果 dominant = F10：
  转入 architecture / compositional FullEdge line。
```

---

## 11. P6：Controller boundary

### 打开条件

满足任一：

```text
P2 controller candidate pass = 1
或 P4 Stage64 discovery pass = 1
```

### 目标

把 discovery 结果 frozen 成 accepted region。

### 必须记录

```text
accepted_count
coverage
precision
V_LCB
V240_LCB
RAUV_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO
LSO
LTO
LFO
cost
```

### Controller pass

```text
accepted_count >= 87
precision >= 0.75
V240_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory_UCB <= 0.05
offdiag_UCB <= 0.05
LDO <= 0.10
LSO <= 0.10
LTO <= 0.10
LFO <= 0.10
```

### 不满足条件时 Codex 先尝试

```text
如果 accepted_count < 87：
  不降低 quality gate；
  回到 P4 生成更多 family-diverse candidate。

如果 precision 够但 longrisk 高：
  加 risk hard gate；
  不调 precision threshold。

如果 LDO 高：
  查 path type 是否 dataset concentrated；
  不能用 dataset-specific rule。

如果 cost 高：
  转 P7 runtime preflight，不写 controller pass。
```

---

## 12. P7：Selected runtime boundary

### 打开条件

```text
P6 controller pass = 1
```

### 目标

证明 FPAU / FPO / generated controller 能在 runtime 中执行。

### 指标

```text
step_ratio_q50/q90/q99
feature_cost_ms_q90
controller_kernel_count
sync_count
active_step_count
empty_step_count
payload_apply_ms_q90
materializer_in_timed_path
```

### Runtime pass

```text
step_ratio_q90 <= 1.50
materializer_in_timed_path = 0
cpu_offload = 0
```

### 不满足条件时 Codex 先尝试

```text
如果 feature cost 高：
  cache co-state sketch；
  reduce virtual samples；
  batch-major active-step grouping。

如果 empty-step kernel 多：
  active-step compaction。

如果 payload apply 慢：
  fuse payload apply kernel；
  pre-pack basis deltas。
```

---

## 13. P8：Paired replay / short-full boundary

### 打开条件

```text
P6 controller pass = 1
P7 runtime pass = 1
```

### 目标

证明 functional update 不是 diagnostic，而是真正因果有效。

### Controls

```text
AdamWParallel
bestLR
NoOp
RandomPayload
ShuffledPayload
FPOScoreOnlyNoPayload
RiskVetoOnly
```

### 指标

```text
RealFunctional V20/V80/V240
beats controls rate
bad/null/longrisk
memory/offdiag
calibration
sample efficiency
forgetting
runtime
```

### Paired replay pass

```text
RealFunctional beats AdamWParallel / bestLR / NoOp / RandomPayload / ShuffledPayload
V240_LCB > 0
longrisk_UCB <= 0.05
memory/offdiag_UCB <= 0.05
runtime pass retained
```

---

## 14. 并行执行策略

v10.3 必须避免“一轮只发现一个 blocker”。

### 并行批次 A：当天先跑

```text
P0 boundary lock
P1 path type 固化
P2 FPAU known-action adjoint scoring
P3 toy mechanism check
```

### 并行批次 B：P2/P3 任一有 weak signal 后立即跑

```text
P4 Stage4/Stage8 generated discovery
P5 failure attribution skeleton
```

### 并行批次 C：条件触发

```text
P4 Stage32/Stage64
P6 controller boundary
P7 runtime
P8 paired replay
```

### 严禁

```text
P2/P3 全失败仍扩大 P4；
Stage8 全 BadPath 仍跑 Stage32；
Stage64 未过仍开 controller；
controller 未过仍跑 runtime；
runtime 未过仍跑 paired replay。
```

---

## 15. 本轮最终 route 格式

```text
R1-FPAUKnownActionMechanismPass
R2-FPAUGeneratedDiscoveryPass
R3-FPAUGeneratedDiscoveryWeakButRisky
R4-FPAUAdjointCannotExplainPathType
R5-SubspaceMissingValueDirection
R6-SafetyConstraintsKillValue
R7-BaseArchitectureExpressivityLimit
R8-UpdateRuleTheoryRebuildAgain
```

---

## 16. 本轮最重要的一句话

v10.3 不再问：

```text
哪个已有动作或 generated family 也许会好？
```

v10.3 直接问：

$$
\boxed{
\text{能否用未来路径目标的 adjoint/co-state，在训练当下求出一个小参数更新？}
}
$$

如果 FPAU 也失败，下一步就不是继续加 FPO / GUP 名字，而是回到更高层：update 子空间、base architecture、或者 future-path objective 本身。 
