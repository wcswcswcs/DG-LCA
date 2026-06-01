# DG-KAN v9.8.4 结果解读与 v9.8.5 下一步实验计划

> 版本：v9.8.5 planning draft  
> 目的：解释 v9.8.4 的真实进展、四条线各自卡在哪里，并给出下一轮不靠小修补的并行实验计划。  
> 约束：不按数据集调参；不使用 validation/test/outcome-at-commit；不使用 fake/proxy rows；不把诊断结果写成 official controller；公式使用 Typora 友好的 `$...$` 与 `$$...$$`。

---

# 1. 一句话判断

v9.8.4 有真实进展，但不是能力成功。

它最重要的推进是：新增 A 线 existing-action 组真实 branch-horizon replay 了，不再只是复用旧表；Core77 的未来路径仍然强，新增的 CoreExpansion10 和 RiskCleanButLowValue 给了更多对照。但是，训练当下合法机制仍然没有找到，自然 AP0 动作扩流 materializer 仍然没落地，虚拟路径 probe 没跑，生成路线也没有理由重开。

所以当前状态不是：

```text
好动作不存在。
```

也不是：

```text
未来训练轨迹思想失败。
```

更准确是：

$$
\boxed{
\text{好动作的未来路径已经越来越清楚，但训练当下还没有合法、通用、可部署的识别或生成机制。}
}
$$

v9.8.4 的 terminal route 是：

```text
route = R1-FuturePathMechanismFoundLegalMechanismAbsent
primary_blocker = no_legal_training_time_mechanism
secondary_blocker = natural_stream_materializer_missing
system_legal_controller_pass = 0
generated_route_status = stopped_no_legal_mechanism_or_density_evidence
```

这个 route 基本合理，但我会把它再往深处改写成：

$$
\boxed{
\text{未来路径现象存在；现在缺的是训练当下可算的未来路径代理，以及足够大的自然动作池。}
}
$$

---

# 2. 四条线的真实进展

## 2.1 A 线：未来训练路径

A 线是 v9.8.4 最大进展。

本轮新增了 `97` 个 selected actions，并真实 materialize 了 `2910` 条 future-path rows。也就是说，A 线不是只读旧表，而是补跑了新的 existing-action 组。

关键结果如下：

| group | actions | AUV LCB | V1 LCB | V20 LCB | V80 LCB | V240 LCB | RiskPath UCB | memory/offdiag UCB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Core77 | 77 | 2.8297 | 0.0882 | 1.5735 | 3.1200 | 4.7509 | 0.0383 | 0.0 |
| CoreExpansion10 | 10 | 0.6225 | -0.0404 | -0.0296 | 0.5669 | 1.6236 | 0.0 | 0.0 |
| ExactOnly | 68 | 1.5250 | 0.1370 | 0.4931 | 1.5824 | 2.9221 | 0.0 | 0.0 |
| OldOnly | 10 | 1.0287 | -0.1748 | 0.3001 | 1.1094 | 2.0678 | 0.0 | 0.0 |
| RandomMatched | 87 | 1.8285 | 0.0941 | 0.5411 | 1.8306 | 3.5923 | 0.8485 | 1.9039 |
| RiskCleanButLowValue | 87 | 2.1603 | 0.1180 | 0.5926 | 2.2688 | 4.2696 | 0.0545 | 0.0 |

A 线的结论不是简单的 “Core77 好”。更深的结论有三个。

第一，**有些动作不是 h1 最强，但后面会变好**。OldOnly 的 `V1 LCB = -0.1748`，但 `AUV LCB = 1.0287`，`V240 LCB = 2.0678`。这继续支持我们的核心观点：好 functional update 不一定是当前一步马上降 loss 的动作，而可能是把后续训练带到更好路径的小改动。

第二，**高 AUV 不等于好动作**。RandomMatched 的 `AUV LCB = 1.8285`，看起来不低，但它的 `RiskPath UCB = 0.8485`，memory/offdiag UCB 也非常高。这说明我们不能只用 AUV 或未来 value 做 score；路径必须和风险、memory、offdiag 一起看。

第三，**RiskCleanButLowValue 这个名字可能已经过时**。它在 v9.8.4 的 future path 中其实不低：`AUV LCB = 2.1603`，`V240 LCB = 4.2696`，risk 也低。这说明某些过去被视为“low value”的安全区域，在完整未来路径下可能有 delayed value。下一轮需要重审它到底是好动作族，还是因为未来路径口径导致的标签错位。

A 线 weak pass 过了，但 strong pass 没过。原因不是 materializer，而是强判据要求 Core77/OldOnly 相对 ExactOnly/RandomMatched 有更清楚的优势。现在 Core77 很强，但 ExactOnly 和 RiskCleanButLowValue 也出现不弱的未来路径，说明“好路径”的类型不止一种。下一轮不能继续只把 Core77 当唯一正例。

---

## 2.2 B 线：训练当下合法机制

B 线仍然失败。

v9.8.4 的 best mechanism 是 `B2_signal_consistency_diagnostic`，但 legality 是 `red:old_table_or_windowed_diagnostic`。也就是说，它不是训练当下可以 official 使用的字段。之前 v9.8.3 也有同样问题：真正能解释 Core77 / OldOnly 的信号大多是 red diagnostic；green 的 memory/cost 类字段能压风险，但不能选出高 value 动作。

这说明我们现在不是缺 “再加一个人工特征”。问题更深：

$$
\boxed{
\text{未来路径好坏目前主要能被事后信息解释，不能被训练当下的合法信息稳定解释。}
}
$$

这就是 primary blocker。

B 线下一步不能继续：

```text
调 B2 threshold；
调 OldRank diagnostic；
把 windowed future diagnostic 包装成 legal signal；
用 memory/offdiag 单独当收益 score。
```

B 线应该转成更硬的问题：

```text
训练当下是否能构造一个小成本 virtual path probe，模拟未来路径的一小段，并且不使用 validation/test/outcome label？
```

如果 virtual path probe 仍然不能解释 Core77 / OldOnly，那就说明 existing-action route 没有可部署机制，必须转向更根本的更新规则。

---

## 2.3 C 线：自然 AP0 动作扩流

C 线没有实质进展，这是很严重的执行 blocker。

现有 panel 仍然只有 `2876` 个 labeled AP0 actions。5000 / 10000 / 20000 panel 全部没有跑，因为 natural AP0 labeled stream extension materializer 仍然没有 landed entrypoint。

当前已知：

```text
PanelA CoreLike rate = 0.0267732962
PanelA LCB = 0.0214752401
PanelA UCB = 0.0333338855
```

这个置信区间横跨 `0.03`，所以不能判断好动作密度是否够。换句话说，我们还不知道：

```text
是动作池太小，导致 Core-like 只有 77 个；
还是当前自然 AP0 动作分布下，干净好动作真实密度就是低于 gate。
```

这个问题已经拖了很多轮。没有 C 线，A/B 线就很难进入 controller，因为如果动作密度本身不够，selector 再好也很难满足 coverage。

C 线下一步必须作为工程最高优先级之一。

---

## 2.4 D 线：生成新动作

D 线继续停止是正确的。

v9.8.4 的 generated route 是：

```text
generated_route_status = stopped_no_legal_mechanism_or_density_evidence
generated_reopen_allowed = 0
```

这不是保守，而是必要。因为：

```text
B 线没有合法机制；
C 线没有自然动作密度证据；
virtual path probe 没跑；
过去 APG/APGL/APGH/APGT 等生成路线多次 value-negative / high-longrisk。
```

在这种情况下重开 generated route，只会回到“生成一堆合法 payload，但没有好的训练几何”的老问题。

D 线下一步只允许在两个条件之一成立后重开：

```text
条件 1：B 线发现合法 virtual path proxy，可以作为生成目标；
条件 2：A 线找到明确的路径机制，并且 C 线证明 natural actions 密度不足，需要生成补足。
```

否则 generated route 继续停止。

---

# 3. 为什么你会感觉非常慢

你的感觉是对的。

从系统能力角度看，v9.8.4 仍然没有：

```text
selected controller；
selected runtime；
official paired replay；
short/full training；
external-ready result。
```

所以它没有让项目进入真正的能力闭环。

但是它不是完全没进度。它把问题从“有没有 future path”推进到了更精确的边界：

```text
1. future path 确实有强信号；
2. AUV alone 不够；
3. risk/memory/offdiag 必须作为路径安全约束；
4. 训练当下合法机制仍缺；
5. natural stream materializer 是硬工程 blocker；
6. generated route 没有合法目标，不能重开。
```

慢的根本原因不是实验少，而是我们现在追的是一个比以前更难的问题：

$$
\boxed{
\text{不是事后找好动作，而是让优化器在训练当下理解哪种参数改动会改善未来训练路径。}
}
$$

这比 “选一个 score、调一个 threshold” 难很多。

---

# 4. 本轮发现的本质问题

## 4.1 未来路径必须是向量，不是一个标量

v9.8.4 说明 AUV 不能单独用。RandomMatched AUV 不低，但风险高；RiskCleanButLowValue AUV 很强但名字显示它可能与旧标签口径冲突；CoreExpansion10 在 V1/V20 不强，但 V80/V240 转正。这说明好动作要看完整路径：

$$
P(a)=\left(V_1,V_5,V_{20},V_{80},V_{240},Risk,Memory,Offdiag,Cost\right).
$$

不能压成一个简单分数：

$$
Score(a)=AUV(a).
$$

如果一定要定义接受规则，也必须是多门限：

$$
V_{80}^{LCB}(a)>0,
$$

$$
V_{240}^{LCB}(a)>0,
$$

$$
RiskPath^{UCB}(a)\le \tau_R,
$$

$$
MemoryOffdiag^{UCB}(a)\le \tau_M.
$$

## 4.2 当前 legal mechanism 不是弱一点，而是方向不对

如果 B 线只是 precision 差一点，下一步可以做 calibration。但现在的情况是：强信号是 red，green signal 只能保安全，不能保 value。这说明训练当下可见的机制没有抓住“未来路径变好”的原因。

数学上可以写成：我们需要找到一个训练当下可算的 $Z_t(a)$，使得：

$$
P\left(Y_{future}=1\mid Z_t(a),D\right)
$$

不再强依赖具体 dataset / table / outcome diagnostic。

现在的问题是：存在好的 $Y_{future}$，但没有好的 $Z_t$。

## 4.3 自然动作密度仍是未解变量

当前 Core-like density 的 Wilson 区间跨过 0.03，因此 existing-action route 的生死还不能判定。必须回答：

$$
p_{core}(N)=\frac{\#\text{CoreLike actions in natural AP0 stream of size }N}{N}
$$

当 $N=5000,10000,20000$ 时，$p_{core}$ 是否稳定高于 0.03？

如果高于，existing-action route 还有希望；如果低于，就说明动作源本身不够，需要几何自适应生成。

## 4.4 当前生成路线没有目标

生成路线过去反复失败，不是因为 payload/apply/materializer 不行，而是因为生成目标错。v9.8.4 没有给出新合法目标，所以停止是正确的。

---

# 5. 是否还在正确道路上

高层路线仍然对：没有按数据集调参，没有把 red diagnostic 写成 official，没有在 controller 不过时打开 runtime，也没有伪造 natural stream labels。

但执行上要更硬。下一步不能继续：

```text
调 B2；
调 AUV threshold；
只看 Core77；
只看 RiskCleanButLowValue；
用 support-aware gate 绕过 raw gate；
没有 natural stream materializer 就谈密度；
没有 legal mechanism 就重开 generated route。
```

正确方向是：

```text
A 线：把未来路径从“好像存在”拆成路径类型和机制；
B 线：落地合法 virtual path probe，寻找训练当下可算的未来路径代理；
C 线：优先完成 natural AP0 labeled stream extension；
D 线：只有 B/C 给出证据后才有限重开生成路线。
```

---

# 6. 离目标还差多远

离 system-legal local functional controller 至少还差四道门：

```text
1. accepted_count >= 87；
2. precision / V / longrisk / bad / null / memory / offdiag 同时过线；
3. LDO / LSO / LTO / LFO 稳定，或者用自然扩流证明 raw LDO 是 density/backfill artifact；
4. selected runtime step_ratio_q90 <= 1.50。
```

离 strict PureKAN functional causal evidence 还要：

```text
official paired replay；
RealFunctional beats AdamWParallel / bestLR / NoOp / Random；
shuffled controls fail；
leave-dataset-out / leave-stratum-out / leave-template-out。
```

离 external-ready Beyond-MLP 还要：

```text
short/full training；
sample efficiency；
calibration；
robustness；
continual / anti-forgetting；
strong baseline 排除；
external reproducibility。
```

所以不能说快成功了。当前最准确的状态是：

$$
\boxed{
\text{未来路径信号可信，局部好动作存在；但训练当下合法机制、自然动作密度和生成目标仍未闭合。}
}
$$

---

# 7. v9.8.5 下一步实验计划

v9.8.5 的总体目标不是小修 v9.8.4 的某个分数，而是把四条线推进到能做路线裁决的程度。

总命题是：

$$
\boxed{
\text{能否找到一个训练当下合法、跨数据分布稳定、低成本的未来路径代理，或证明必须转向新的几何自适应生成器？}
}
$$

v9.8.5 分四线并行。

---

## A 线：未来路径类型与机制分解

### 目标

不是继续证明 Core77 好，而是弄清楚不同“好路径”到底有几种：

```text
1. Core77：高 value、低 risk 的干净路径；
2. CoreExpansion10：h1/h20 弱，但 h80/h240 转正的延迟收益路径；
3. RiskCleanButLowValue：名字显示旧标签低 value，但完整路径显示 value 很强的风险干净路径；
4. OldOnly：immediate 不强但后续变好的路径；
5. ExactOnly：immediate 或 exact response 可能好，但不能代表真正好路径；
6. RandomMatched：AUV 可以不低，但 risk/memory/offdiag 高的危险路径。
```

### 核心假设

$$
H_A:
\text{好 functional update 的本质是路径形状，而不是单点 loss 或单个 AUV。}
$$

### 必须记录

对每个 action、每个 horizon $h\in\{1,5,20,80,240\}$ 记录：

```text
action_id
group_id
dataset
seed
family
template_id
V_h
CE_delta_h
NLL_delta_h
margin_delta_h
CEp99_delta_h
hard_tail_delta_h
memory_loss_delta_h
old_family_margin_delta_h
offdiag_fail_h
longrisk_h
bad_h
null_h
cover_entropy_delta_h
basis_effective_rank_delta_h
```

额外构造：

$$
AUV(a)=\sum_h w_h V_h(a),
$$

$$
DelayedGain(a)=V_{240}(a)-V_1(a),
$$

$$
RiskAdjustedAUV(a)=AUV(a)-\lambda_R RiskPath(a)-\lambda_M MemoryOffdiag(a).
$$

注意：`RiskAdjustedAUV` 只能作为诊断，不能直接 official controller。

### 判断标准

A 线 strong pass 不是“Core77 好”，而是必须证明至少一种路径类型满足：

```text
action_count >= 87 或者能作为明确可扩展机制；
V80_LCB > 0；
V240_LCB > 0；
RiskPath_UCB <= 0.05；
memory/offdiag_UCB <= 0.05；
bad_UCB <= 0.05；
null_UCB <= 0.15；
LDO/LSO/LTO drop <= 0.10。
```

如果只在事后 future labels 上成立，A 线只能 mechanism pass，不能 controller pass。

### 可视化

```text
A1_group_future_path_V_curve.svg
A2_group_future_path_risk_curve.svg
A3_delayed_gain_vs_immediate_value.svg
A4_AUV_vs_longrisk_pareto.svg
A5_RiskCleanButLowValue_relabel_audit.svg
A6_path_type_heatmap.svg
A7_core_expansion_delayed_gain_curve.svg
```

---

## B 线：训练当下合法 future proxy

### 目标

寻找一个训练当下可以算、且不使用未来 outcome 的未来路径代理。

它不能使用：

```text
future V；
future risk；
old table diagnostic；
windowed future outcome label；
dataset name；
validation/test metric。
```

允许使用：

```text
当前 train batch；
当前 train-memory buffer；
当前 hard-tail subset；
per-example gradient / Jacobian-vector response；
小步 virtual train probe；
payload apply cost；
current logits / margins / CE；
action norm / AdamW alignment；
memory/offdiag pre-commit proxy。
```

### 核心假设

$$
H_B:
\text{可以用训练当下的小成本 virtual path probe 预测 action 的未来路径类型。}
$$

### 实验设计

对每个 action 执行三个层级的合法 probe：

```text
B1: current response
  只看当前 batch / memory buffer 上的 linearized response。

B2: virtual 1-step probe
  apply action 后在 train batch 上模拟 1 个 AdamW-equivalent update。

B3: virtual 3/5-step probe
  只用 train stream 或 train-memory buffer，不用 validation/test，不用 future outcome table。
```

### 必须记录

```text
action_id
probe_type
probe_steps
probe_batch_source
probe_memory_source
feature_compute_ms
payload_apply_ms
virtual_step_ms
current_CE_delta
memory_CE_delta
hard_tail_CE_delta
old_family_margin_delta
action_AdamW_cosine
per_example_response_mean
per_example_response_std
response_SNR
virtual_path_AUV_proxy
virtual_path_risk_proxy
virtual_memory_harm_proxy
virtual_offdiag_proxy
```

### 判断标准

B 线 weak pass：

```text
TopK87 precision >= 0.60
V_LCB > 0
longrisk_UCB <= 0.10
memory/offdiag_UCB <= 0.10
feature+probe cost q90 <= budget_prelim
no red field
```

B 线 strong pass：

```text
TopK87 precision >= 0.75
V_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO <= 0.10
probe cost q90 + payload apply q90 不导致 step_ratio_q90 > 1.50
```

如果 B 线失败，不能继续微调 B2/B3 阈值。需要判断：

```text
失败是因为 probe 太短；
还是因为训练当下信息本身不足；
还是因为 action pool 的好动作太稀疏。
```

### 可视化

```text
B1_proxy_vs_true_AUV_scatter.svg
B2_proxy_vs_longrisk_pareto.svg
B3_virtual_steps_quality_curve.svg
B4_probe_cost_vs_quality.svg
B5_green_vs_red_mechanism_gap.svg
B6_memory_offdiag_veto_effect.svg
```

---

## C 线：自然 AP0 动作扩流

### 目标

回答一个硬问题：自然 AP0 action stream 中，好动作密度到底够不够。

### 核心假设

$$
H_C:
\text{当前 2876 个 action 太少，扩展到更大的自然 action stream 后，Core-like density 可能稳定超过 0.03。}
$$

### 必须先落地的 materializer

C 线第一优先级不是继续算密度，而是实现：

```text
natural_AP0_labeled_stream_extension_materializer
```

它必须能对 panel target：

```text
5000
10000
20000
```

产出 labeled actions 与 branch-horizon outcomes。

### 必须记录

```text
panel_target
actual_action_count
labeled_action_count
missing_label_count
branch_horizon_rows_expected
branch_horizon_rows_actual
rows_per_sec
unresolved_exception_count
quality_audit_pass
fake_row_count
proxy_row_count
CoreLike_count
CoreLike_rate
GradeAB_count
ValuePositiveNoLongRisk_count
RiskClean_count
per_dataset_rate
per_family_rate
per_template_rate
Wilson_LCB/UCB
```

### 判断标准

C 线 preflight pass：

```text
materializer_entrypoint_found = 1
Panel5000 actual >= 5000
quality_audit_pass = 1
fake/proxy = 0
```

C 线 weak pass：

```text
Panel10000 complete
CoreLike_rate_LCB >= 0.03
per_dataset_min_rate_LCB >= 0.02
quality_audit_pass = 1
```

C 线 strong pass：

```text
Panel20000 complete
CoreLike_rate_LCB >= 0.03
ValuePositiveNoLongRisk_rate_LCB >= 0.03 或 Core+Expansion density >= 0.03
per_dataset / per_family / per_template no collapse
LDO/LSO/LTO density stable
```

如果 C 线 LCB 明确低于 0.03：

```text
existing-action route 不能继续只靠筛选；必须转向生成器或新 update rule。
```

如果 C 线 LCB 高于 0.03：

```text
existing-action controller 可以继续，但必须解决 B 线合法机制。
```

### 可视化

```text
C1_density_curve_panel_size.svg
C2_corelike_rate_ci.svg
C3_per_dataset_density.svg
C4_per_family_density_heatmap.svg
C5_materializer_throughput_curve.svg
C6_missing_label_dashboard.svg
C7_density_vs_quality_pareto.svg
```

---

## D 线：几何自适应生成路线的重开条件

### 目标

不再盲目新增 APG/APGU/APGX。只在 A/B/C 给出明确证据后，有限重开 generated route。

### 重开条件

D 线允许重开的条件是：

```text
B 线 strong pass；
或者 B 线 weak pass + C 线证明自然动作密度不足；
或者 A 线找到明确路径机制，且该机制能写成合法 virtual probe objective。
```

如果没有满足，D 线保持 stopped。

### 如果重开，生成器必须基于明确目标

不允许继续：

```text
APG 小变体；
payload norm tweak；
trust region tweak；
SNR mask tweak；
memory/offdiag 单独投影。
```

允许的生成目标应类似：

$$
\Delta^* = \arg\max_{\Delta\in\mathcal{S}}
VirtualPathGain(\Delta)
$$

subject to：

$$
MemoryHarm(\Delta)\le \tau_M,
$$

$$
OffdiagRisk(\Delta)\le \tau_O,
$$

$$
LongRiskProxy(\Delta)\le \tau_L,
$$

$$
Cost(\Delta)\le C_{max}.
$$

这里 $\mathcal{S}$ 不应只绑定 KAN 当前 basis，而应该是一个函数空间子空间，例如：

```text
last-layer output-response subspace；
low-rank function-response subspace；
memory-safe residual subspace；
hard-tail relief subspace；
AdamW-compatible small residual subspace。
```

### 必须记录

```text
generated_action_count
subspace_id
objective_id
constraint_violation_count
payload_hash_missing
certificate_hash_missing
action_apply_error
branch_horizon_rows_expected/actual
V1/V5/V20/V80/V240
AUV
longrisk
bad/null
memory/offdiag
negative_control_pass
source_to_generated_damage
runtime_cost
```

### 判断标准

D weak pass：

```text
generated_action_count >= 256
best generated group precision >= 0.30
V_AUV_LCB > 0
longrisk_UCB <= 0.20
memory/offdiag_UCB <= 0.20
new_positive_created_rate >= 0.10
```

D strong pass：

```text
accepted_count >= 87
precision >= 0.75
V_AUV_LCB > 0
V240_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO <= 0.10
negative control fails
```

### 可视化

```text
D1_generated_quality_pareto.svg
D2_generated_future_path_curves.svg
D3_generated_memory_offdiag_dashboard.svg
D4_generated_vs_existing_density.svg
D5_runtime_apply_cost.svg
D6_source_to_generated_damage_matrix.svg
```

---

# 8. Controller / Runtime / Downstream 打开规则

v9.8.5 不能直接打开 controller/runtime。只有满足下面条件后才允许。

## Minimal controller opening condition

至少满足：

```text
B strong pass 或 C strong pass + B weak pass；
accepted_count >= 87；
precision >= 0.75；
V_LCB > 0；
longrisk_UCB <= 0.05；
bad_UCB <= 0.05；
null_UCB <= 0.15；
memory/offdiag_UCB <= 0.05；
LDO/LSO/LTO <= 0.10；
no red fields；
no dataset-specific branch。
```

## Runtime opening condition

```text
controller selected = 1
feature/probe cost measured = 1
payload apply cost measured = 1
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

## Paired replay opening condition

```text
controller pass = 1
runtime pass = 1
RealFunctional beats AdamWParallel / bestLR / NoOp / Random
shuffled payload fails
leave-dataset-out / leave-stratum-out / leave-template-out stable
```

## Short/full training opening condition

```text
paired replay pass = 1
runtime pass = 1
no dataset-specific tuning
base acc sentinel healthy
strong MLP baselines present
```

---

# 9. v9.8.5 Stop / Pivot 规则

v9.8.5 要避免无限绕圈。设置硬 stop rules。

## Stop A

如果 A 线发现所有 positive future path 都只来自 red diagnostic 或不稳定 group，则停止 future-path-as-controller 路线，只保留为机制研究。

## Stop B

如果 B 线 virtual path probe 在 TopK87 上：

```text
precision < 0.40
或 V_LCB <= 0
或 longrisk_UCB > 0.20
```

并且 cost 高，则停止 legal proxy route，转向 action generation theory。

## Stop C

如果 C 线 Panel20000 完整后：

```text
CoreLike_rate_LCB < 0.03
```

则 existing-action route 不再作为主线。

## Stop D

如果 D 线重开后 best generated group 仍：

```text
V_AUV_LCB <= 0
或 longrisk_UCB >= 0.50
```

则停止 generated route，回到更新规则数学定义。

---

# 10. Dashboard 要求

v9.8.5 必须输出一个 `v9850_dashboard.md`，包括：

```text
1. Route summary；
2. 四线 pass/fail 表；
3. A 线 future path group curves；
4. B 线 legal proxy quality/cost；
5. C 线 natural stream density curve；
6. D 线 generated route stop/reopen decision；
7. Controller/runtime gate boundary；
8. No-fake / no-proxy / no-dataset-tuning audit；
9. Stop/pivot recommendation。
```

必须有这些图：

```text
fig_A1_group_future_path_V_curve.svg
fig_A2_group_future_path_risk_curve.svg
fig_A4_AUV_vs_longrisk_pareto.svg
fig_B1_proxy_vs_true_AUV_scatter.svg
fig_B4_probe_cost_vs_quality.svg
fig_C1_density_curve_panel_size.svg
fig_C2_corelike_rate_ci.svg
fig_D1_generated_quality_pareto.svg
fig_stop_pivot_matrix.svg
```

---

# 11. 最终期待

v9.8.5 的最低有效推进不是 system pass，而是必须把路线判清楚：

```text
Case 1:
  B 线找到合法 virtual path proxy，进入 minimal controller。

Case 2:
  C 线证明自然动作密度足够，但 B 线还没机制，继续 existing-action controller mechanism。

Case 3:
  C 线证明自然动作密度不足，且 B 线有生成目标，有限重开 generated route。

Case 4:
  B/C/D 全失败，停止 current AP0 existing-action + APG generated 路线，回到 optimizer-level theory。
```

我认为 v9.8.5 最关键的不是再多一个 pass，而是给出一个不可回避的路线裁决。

