# DG-KAN v12.22：Fail-Closed Explore-Open、Label-Free Signal Frame 与 Loss-Agnostic Functional Update 重建计划

> 版本：v12.22 execution plan  
> 基于：v12.21 `FailClosedContinue2 LabelFreeFunctional` 实验复盘、`v1221_code_review_packet.zip` 本地静态审查、v12.18-v12.20 去混淆/visibility/continuation 结果。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 核心约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；functional direction 不使用 CE vector、label、permuted-label CE、validation/test/future outcome；CE/ECE/CEp99/Brier 只能作为 audit 与坏化约束，不能作为 direction source。  
> 本计划的关键修正：**promotion fail closed，但 exploration 必须 continue open**。Codex 不能在首个 promotion gate fail 后停止；也不能只写 no-go/next-hypothesis 文档就算完成，必须按预注册探索队列继续执行，直到预算耗尽、成功、或明确 no-go boundary 完成。

---

# 0. 一句话结论

v12.21 比 v12.19/v12.20 有纪律进步：它完成了 continuation manifest、artifact contract、code readback、no-go boundary 和 next-hypothesis generator，没有把失败包装成 success。但它仍然不够合理，因为它把“生成下一轮假设”当成了“可以停止”的条件。

当前真实状态是：

```text
B320-current / FHQ anchor:
  仍然是强 diagnostic anchor，但存在 label-informed trainprobe init confound。

Label-free B320-like base:
  A1-noYForStats 仍是 best label-free baseline，但 task/worst/AUC/LineC 均未同时达标。

Loss-agnostic value source:
  G_LA 只有 exploratory signal，方向和强度都不足，不能作为 functional direction source。

Visibility:
  T1A 有弱 AUC，但 precision/recall 太低；T1B native optimizer-update features 没救回来。

Actuator:
  有些 actuator 可以产生 sketch/projector movement，但 safe movement = 0，release/control-resistant survivor = 0。

Functional P3/P4:
  没有打开；official functional success 仍为 0。
```

所以 v12.21 的结论不是“functional update 没希望”，而是：

$$
\boxed{
\text{当前实验设置仍过于 fail-fast；}
\text{我们还没有一个 label-free、precommit、loss-agnostic、control-resistant 的 value source。}
}
$$

v12.22 的核心是把执行方式从：

```text
gate fail -> write no-go -> stop
```

改成：

```text
promotion gate fail -> 不 promotion；
continuation gate fail -> 自动执行下一层机制诊断；
hypothesis queue 未耗尽 -> 不允许 final stop。
```

---

# 1. v12.21 独立结果分析

## 1.1 v12.21 有进展，但不是能力突破

v12.21 的最终 route 是：

```text
route = R2-LabelFreeSignalFrameMissing
minimum_success = Success E
label_free_official_pass_count = 0
g_la_official_pass = 0
T1B_visibility_pass = 0
actuator_executor_success = 0
p4_open = 0
continuation_missing_count = 0
final_stop_allowed = 1
```

这说明它完成了 fail-closed continuation，但没有任何合法 promotion 路径。这个结果本身可信；问题是 `final_stop_allowed = 1` 的语义不合理。只要没有 success，并且 next hypotheses 还没有被实际执行，就不应该允许 final stop。

v12.21 的 route 说 “continuation complete and next hypotheses generated”。这只能证明 Codex 写出了下一步假设，不等价于“已经探索完”。

## 1.2 Line A：label-free signal frame 没有补上 B320-current 的 label-init 缺口

v12.21 的 label-free frame 表共 8 个候选：A0 labelInit anchor、A1-noYForStats，以及 A30-A35。关键结果是：

| candidate | mean delta vs A0 | worst delta vs A0 | AUC-time vs MLP | LineC all pass | official |
|---|---:|---:|---:|---:|---:|
| A1-noYForStats | -0.00716 | -0.03711 | 1.1323 | 0 | 0 |
| A30 optimizer observable frame | -0.02007 | -0.05273 | 1.1880 | 0 | 0 |
| A31 augmentation tangent frame | -0.01432 | -0.05469 | 1.1590 | 0 | 0 |
| A32 persistent drift frame | -0.01910 | -0.04688 | 1.1838 | 0 | 0 |
| A33 role-balanced primitive energy frame | -0.73850 | -0.82617 | NaN due real NaN loss | 0 | 0 |
| A34 hybrid A1+optimizer | -0.02007 | -0.05273 | 1.1918 | 0 | 0 |
| A35 hybrid A1+aug drift | -0.02821 | -0.06836 | 1.2566 | 0 | 0 |

独立判断：

```text
A1-noYForStats 仍然是 best label-free baseline，但它不够。
A30/A34 说明 optimizer-observable frame 不能直接替代 supervised trainprobe。
A31/A35 说明 augmentation tangent 方向有一点信号，但没有修复 worst/AUC/LineC。
A33 是明确负例：role-balanced primitive energy 可以造成真实 NaN / 任务坍塌。
```

这不是 dataset-specific 问题。Fashion / KMNIST / MNIST 只是暴露了不同 failure slice。不能因为某个数据集接近就写 dataset 分支。

## 1.3 Line C：G_LA 只有 diagnostic，不是可用 value source

v12.21 的 deployable target 聚合是：

```text
g_la_spearman_noise = +0.2481
g_la_spearman_reservoir = -0.0839
g_la_exploratory = 1
g_la_official_pass = 0
```

这代表两件事：

1. `G_LA` 与 noise audit 有一定相关，但方向是正相关；如果我们的目标是降低 `NoiseSignalLeak`，这个方向不能直接用于 promotion。
2. reservoir 相关太弱，只有 `-0.0839`，离一个可用 value source 很远。

因此不能把 `G_LA exploratory=1` 当成功。它只是说明某些 deployable feature 和 audit target 不是完全独立，但还没有达到 functional direction source 的证据等级。

## 1.4 Line T：T1A/T1B visibility 失败，不能继续让 Codex停在“visibility fail”

v12.21 的 visibility score：

| tier | feature | AUC joint | precision@k joint | recall@k joint | soft R2 noise | soft R2 reservoir | pass |
|---|---|---:|---:|---:|---:|---:|---:|
| T1A | precommit geometry composite | 0.6367 | 0.10 | 0.10 | -2.120 | -0.299 | 0 |
| T1B | optimizer-update opaque composite | 0.4571 | 0.00 | 0.00 | NaN | NaN | 0 |

进一步 leaveout 显示 T1A 在 held-out KMNIST 上 AUC 只有 `0.0845`，而 held-out seed/window 表现也不稳。这说明 T1A 不是一个跨 split 稳定的 release predictor。T1B native logging 是进展，因为以前没有原生记录 optimizer-update features；但它没有提供可用 visibility。

更关键的是，soft target regression 的 R2 强负：

```text
Delta_NoiseSignalLeak_audit:
  leave_dataset_out_r2_min = -84.85

Delta_RealSignalReservoirRatio_audit:
  leave_dataset_out_r2_min = -50.89
```

这说明当前 T1 features 连软目标都解释不了。继续只换一个 scorer 或阈值，没有本质意义。

## 1.5 Line I：actuator 不是完全不能动，但还不是 safe executor

v12.21 actuator 总计 216 行：

```text
actuator_rows = 216
actuator_safe_movement_rows = 0
actuator_release_audit_rows = 3
actuator_control_resistant_rows = 0
```

静态审计显示：

```text
sketch_delta_fro max ≈ 0.3027
projector_angle_deg max ≈ 29.43
logit_max_abs_drift median ≈ 0.0513, max ≈ 1.229
NoiseSignalLeak_delta mean ≈ +0.0657
RealSignalReservoirRatio_delta mean ≈ -0.0446
control_gap max ≈ +0.0323
```

这说明 actuator 有能力移动 sketch/projector，但代价是 logit drift 过大，并且平均会增加 noise leak。少数 row 有 reservoir release 或正 control gap，但没有同时满足 safe movement、release audit 和 matched-control gap。

因此，当前 actuator blocker 不是“完全动不了”，而是：

$$
\boxed{
\text{movement / release / safety / controls 无法同时成立。}
}
$$

下一轮不能再单纯放大 actuator norm；应该用 response matrix + drift-constrained composition。

## 1.6 Line B：functional P3/P4 实际上没有执行新机制

v12.21 的 `v1221_functional_p3.csv` 有 6 行，但全部是：

```text
P3_open = 0
failure_reason = P3 not opened: C/T/I official gates did not pass
```

所以 Line B 没有新的 functional evidence。它只是正确关闭了 P3/P4。

问题是：正确关闭不等于研究推进。关闭后必须继续 C/T/I 的 continuation，不应该写完 no-go 就结束。

## 1.7 Line D：经典基函数被“记录为假设”，不是被真正执行

v12.21 的 classic status 表里，Rational / Chebyshev / Wavelet / RBF/FastKAN / Fourier 都是：

```text
new_hypothesis_defined = 1
executed_this_version = 0
v1221_status = RejectedForThisVersion
```

这不应该被解释为 family 失败。更准确的状态是：

```text
NotExecuted_HypothesisGenerated
```

这一点暴露了实验计划的执行问题：Codex 把“写出下一轮假设”当成 continuation 完成，而不是实际执行至少一个 candidate 或 smoke。

---

# 2. 当前真正卡在哪里

我现在把 blocker 拆成两层。

## 2.1 科学 blocker

$$
\boxed{
\text{没有一个 label-free、loss-agnostic、precommit 的 value source，}
\text{能稳定预测或驱动 signal/reservoir/noise 改善。}
}
$$

更具体：

```text
1. B320-current 很强，但依赖 label-informed trainprobe init。
2. label-free frame 能学一点 task，但 worst/AUC/LineC 过不了。
3. deployable G_LA 与 audit release 的相关方向/强度不够。
4. T1A/T1B visibility 对 joint hard release 的 precision/recall 低。
5. actuator 有 movement，但 unsafe、noise leak、control gap 不同时成立。
```

## 2.2 实验执行 blocker

$$
\boxed{
\text{Codex 的执行策略仍然是“gate fail 后生成文档并停止”，}
\text{不是“gate fail 后继续机制探索直到预算耗尽”。}
}
$$

本地审查 `run_v1221_failclosed_continue2_label_free_functional.py` 后，关键问题是：

```text
continuation_executed 的定义过弱：
  Line D 只要 classic_new_hypothesis_count > 0 就算 continuation_executed，
  即使 executed_this_version = 0。

Line B 只要 functional_p3_rows > 0 就算 continuation_executed，
  即使 P3_open = 0。

final_stop_allowed = 1 只要求 no-go boundary 和 next-hypothesis generator 写出，
  不要求下一层 hypotheses 被执行。
```

这正是你说“Codex 一直停下不肯继续探索”的根因之一。计划本身允许它停。

---

# 3. 是否在正确道路上

方向仍然是对的：

```text
1. 不再小修 B320-current；
2. 不把 CE-specific projector 写成 success；
3. 不用 CouplingR2 单指标 promotion；
4. 不按 dataset 调 controller；
5. 不在 P2/P3 fail 时强开 P4；
6. 开始审查代码实现与 feature provenance。
```

但推进策略必须改变：

```text
旧策略：
  gate fail -> no promotion -> write no-go -> stop。

新策略：
  gate fail -> no promotion -> execute fallback queue -> update hypothesis -> continue until budget exhausted。
```

这不是降低 gate，而是提高探索效率。

---

# 4. 离目标还差多远

| 模块 | 当前完成度 | 判断 |
|---|---:|---|
| B320-current diagnostic anchor | 80%-85% | 强，但 label-informed，claim scope 受限 |
| label-free B320-like base | 35%-45% | A1 接近但 worst/AUC/LineC 不过，A30-A35 未修复 |
| Line C audit metric | 75%-80% | 审计可用，但 target/source 分层仍需严守 |
| loss-agnostic deployable value source | 10%-20% | T1A/T1B 都未过，G_LA 仅 exploratory |
| actuator executor | 20%-30% | 能动，但不安全、不 release、不 control-resistant |
| official functional update | 0% | P3/P4 仍未打开 |
| classic no-BSpline portfolio | 25%-35% | 有历史进展，但 v12.21 没执行新候选 |
| execution policy / Codex autonomy | 40% | artifact 完整，但 fail-fast 停止问题仍在 |

整体项目如果以“B320-current + diagnostic anchor”为标准，大约 60% 以上；如果以“label-free PureKAN + loss-agnostic functional update official success”为标准，仍然只有 35%-45%。

---

# 5. v12.22 总目标

v12.22 不再把目标写成“跑完 continuation”。总目标是：

$$
\boxed{
\text{在不降低 promotion gate 的前提下，建立一个不会提前停止的探索执行系统，}
\text{并推进 label-free base、loss-agnostic value source、safe actuator 三个瓶颈。}
}
$$

具体目标：

```text
G1. 修正 Codex 执行协议：promotion fail closed, exploration continue open。
G2. 对 label-free B320-like base 做结构性 frame search，而不是继续 token 小修。
G3. 对 G_LA/T1 visibility 做 target reset，而不是继续单一 composite scorer。
G4. 对 actuator 做 drift-constrained response composition，而不是继续放大 norm。
G5. 只有 C/T/I 至少一个合法路径打开，才进入 official P3/P4。
G6. Classic no-BSpline line 不再用 “hypothesis generated” 冒充 executed。
```

---

# 6. v12.22 核心假设

## H-A：label-free signal frame 不能靠替换整体 projector 修复，需要 residual-on-A1

v12.21 的 A30-A35 多数是替换或强混入新 frame，结果比 A1 更差。下一步应该保持 A1-noYForStats 的主体行为，只加低秩 residual frame：

$$
P_{new}=P_{A1}+\epsilon R_{LF},
$$

其中 $R_{LF}$ 来自 label-free observables，如 covariance、augmentation tangent、persistent drift、random cotangent stability。

成功信号：

```text
mean_delta_vs_A0 >= -0.005
worst_delta_vs_A0 >= -0.020
AUC_time_ratio_vs_MLP <= 1.05
LineC_nontearing pass rate >= 6/9 exploratory
```

## H-C：hard release 不一定可由单一 G_LA 预测，需要拆成多个 role-conditioned target

v12.21 的 `G_composite` 不能 official。下一步拆成：

```text
G_cov
G_iso
G_coupling_stability
G_role_transition
G_optimizer_spectrum
G_projector_stability
```

每个分量单独做 sign sanity、leaveout、top-k precision；只允许通过 source-run leaveout 的分量进入组合。

## H-T：T1B static optimizer-update spectrum 不够，需要 temporal / transition features

v12.21 的 T1B AUC joint 只有 `0.457`。但它是首次 native logging，因此不能直接放弃。下一步增加：

```text
update half-life
role-energy transition
quad/direct/branch energy flow
update cosine decay
update spectrum entropy change
```

这些仍然不能读 label / CE，但要明确标记为：

```text
T1B-semi-deployable / optimizer-observable diagnostic
```

除非我们明确接受“base optimizer state 可作为 loss-agnostic opaque state”，否则 T1B 不可直接作为 official direction source。

## H-I：actuator 不是弱，而是 drift-constrained composition 缺失

v12.21 有 actuator movement，但 logit drift 和 noise leak 坏。下一步不增加 norm，而是做 bisection + low-rank composition：

$$
\max_a \Delta ProjectorAngle(a)
$$

subject to:

$$
logit\_max\_abs\_drift(a)\le 0.05,
$$

$$
NoiseSignalLeak\_delta(a)\le 0,
$$

$$
\|a\|\le budget.
$$

成功信号：safe movement rows 不再是 0，并且 matched controls 不解释。

## H-B：functional update 必须先通过 value-source / actuator gate，不应在 closed P3 上构造空 P4

v12.21 的 P3/P4 都是空闭合。下一步仍然保持 official P4 gate，但允许 **shadow P4 capacity diagnostic**：只用于判断 executor 能不能影响训练轨迹，不允许 promotion。

---

# 7. v12.22 执行规则：Fail-Closed Explore-Open

## 7.1 两类 gate 必须分开

### Promotion gate

用于决定是否能 claim / open official P4。必须严格：

```text
P2 visibility official pass
P3 actuator safe+release+control pass
P4 short-run task/nontearing/control pass
```

失败后：

```text
promotion_allowed = 0
```

### Exploration gate

用于决定是否继续执行下一层预注册诊断。必须宽开：

```text
如果 promotion fail，但 diagnostic 信息不足，则继续。
如果 diagnostic 信息已经足够，但 next hypothesis 未执行，则继续。
如果预算未耗尽，则继续。
```

失败后不能停，只能进入下一 fallback。

## 7.2 新 route 语义

新增 route：

```text
R0-FailFastIncomplete:
  gate fail 后没有执行 required fallback queue。

R0-NextHypothesisNotExecuted:
  写了 next hypothesis，但没有实际执行至少一个 child run / smoke / artifact。

R1-ExplorationBudgetExhausted:
  所有预注册 fallback 已执行，预算耗尽，可 final stop。

R2-LabelFreeSignalFrameMissing:
  label-free base 仍缺，但 fallback 已完整执行。

R3-LossAgnosticValueSourceMissing:
  T1/T1B/G_LA value source 仍缺，但 fallback 已完整执行。

R4-ActuatorExecutorMissing:
  value source 有信号，但 safe actuator / controls 不成立。

R5-P4ShortRunFailed:
  P4 打开但 short-run 未成功。

S1-FunctionalDiagnosticSurvivor:
  P2/P3 通过，P4 可打开。

S2-OfficialFunctionalSuccess:
  P4/P5 通过。
```

## 7.3 final stop allowed 的新定义

只有下面任一条件成立，才允许 final stop：

```text
1. official success reached；
2. exploration_budget_exhausted = 1；
3. no-go boundary + all fallback levels executed + manual_review_required = 0；
4. user_stop_flag = 1。
```

禁止：

```text
next_hypothesis_generator_written = 1 -> final_stop_allowed = 1
```

这条必须从 runner 里删除。

---

# 8. v12.22 分阶段计划

## P0：Execution Contract / No-FailFast 审计

### 目标

先修实验执行语义，确保 Codex 不能再在首个 gate fail 后停止。

### 必须实现

新增 artifact：

```text
v1222_exploration_queue.json
v1222_child_run_manifest.csv
v1222_fallback_execution_manifest.csv
v1222_budget_accounting.csv
v1222_final_stop_audit.json
```

`v1222_exploration_queue.json` 每个 job 必须包含：

```text
job_id
line
parent_failure
hypothesis
command
expected_artifacts
promotion_gate
continuation_gate
fallback_if_fail
max_runtime_minutes
max_candidates
priority
parallel_group
```

### Gate

P0 通过要求：

```text
route cannot be final_stop_allowed unless budget_exhausted or official success。
Line D generated hypothesis must not count as executed unless a new candidate artifact exists。
Line B closed P3 rows cannot count as functional continuation unless at least one diagnostic child run exists。
```

### Codex 失败后必须尝试

如果 queue 缺失：生成 queue。  
如果 child run 没执行：先执行 P0 smoke child jobs。  
如果 budget 字段缺失：route = R0-ExecutionContractIncomplete。

---

## P1：Line A Label-Free Signal Frame v2

### 目标

构造不依赖 label 的 B320-like base，使它接近 A0 labelInit 的 task trajectory，并修复 Line C nontearing。

### 候选 family

#### A36：Residual-on-A1 LowRankFrame

$$
P=P_{A1}+\epsilon UV^T,
$$

其中 $U,V$ 来自 unlabeled covariance / random cotangent stability。

#### A37：Convex MultiFrame Mixture

$$
P=\sum_m \alpha_m P_m,
\quad \alpha_m\ge0,
\quad \sum_m\alpha_m=1.
$$

不使用 label。$P_m$ 包括 PCA、SRHT、augmentation-stable、persistent drift、block-local frames。

#### A38：Persistent Drift Cotangent Bank

用训练早期 unlabeled logit/hidden drift，通过 frozen random cotangent bank 映射成 frame。

#### A39：Role-Conditioned Frame

分别为 direct / quad / branch 构造 frame，不再用一个 frame 同时服务所有 role。

#### A40：Self-Conditioned Frame with Stop-Grad

用当前 model 的 unlabeled activation covariance 构造 frame，但 stop-grad，避免变成 unstable self-training。

#### A41：Unlabeled Frame Adaptation Schedule

先用 A1 启动，epoch 1 后用 unlabeled covariance 更新小幅 residual frame。

#### A42：Small-label Oracle Upper Bound，diagnostic only

少量 label 只做 upper-bound，不允许 promotion。用于判断 label signal 的最小需求。

### 指标

记录：

```text
candidate_id
frame_family
uses_label
uses_ce_vector
uses_validation_or_test
uses_dataset_name
uses_optimizer_update_opaque
mean_delta_vs_A0
worst_delta_vs_A0
AUC_step_ratio_vs_mlp
AUC_time_ratio_vs_mlp
ECE_delta_vs_A0
CEp99_delta_vs_A0
LineC_nontearing_all_pass
CouplingR2_delta_vs_A0
NoiseSignalLeak_delta_vs_A0
RealSignalReservoirRatio_delta_vs_A0
frame_rank
frame_condition
frame_energy_by_role
P_energy_on_frame
quad_feature_std_mean
branch_scale_norm
NaN_count
```

### Promotion gate

正式 label-free base 候选必须满足：

$$
mean\_delta\_vs\_A0 \ge -0.003,
$$

$$
worst\_delta\_vs\_A0 \ge -0.010,
$$

$$
AUCtime\_ratio\_vs\_MLP \le 1.00,
$$

$$
LineC\_nontearing\_all\_pass=1.
$$

探索 gate：

$$
mean\_delta\_vs\_A0 \ge -0.010,
$$

$$
worst\_delta\_vs\_A0 \ge -0.030,
$$

$$
AUCtime\_ratio\_vs\_MLP \le 1.15,
$$

$$
LineC\_pass\_rate \ge 5/9.
$$

### 可视化

```text
fig_v1222_linea_task_linec_pareto.svg
fig_v1222_frame_condition_vs_auc.svg
fig_v1222_role_energy_by_candidate.svg
fig_v1222_label_free_gap_waterfall.svg
fig_v1222_linec_failure_by_frame_family.svg
```

### 不满足条件时 Codex 必须继续

```text
mean_delta 失败：
  先试 residual-on-A1，而不是替换 projector。

worst_delta 失败：
  加 role-conditioned frame，检查 hard-tail / CEp99。

AUC_time 失败：
  检查 AUC_step 是否也失败；若 AUC_step 过、AUC_time 不过，转 timing；若两者都不过，转 trajectory frame。

LineC 失败：
  记录 NoiseSignalLeak / ReservoirRatio 的具体 blocker，进入 Line C target reset。

NaN：
  降 frame energy、clip projector condition、记录 NaN source，不把 NaN summary 写成 0。
```

---

## P2：Line C / T Value Source v4

### 目标

找到 loss-agnostic、precommit、跨 split 稳定的 value source。注意：这一步不是构造 update，而是证明 value 可见。

### Feature tier

#### T1A：strict precommit label-free state features

允许：

```text
unlabeled logit covariance spectrum
hidden covariance spectrum
projector condition / angle stability
role energy / branch energy
activation occupancy entropy
random-cotangent gradient sketch spectrum without labels
train-probe unlabeled drift stability
```

禁止：

```text
CE vector
label residual
permuted-label CE
validation/test outcome
future outcome
method identity flags as value source
post-response actual delta
```

#### T1B：optimizer-observable diagnostic features

允许记录：

```text
opaque base optimizer update norm
role-wise update energy
update half-life
update cosine transition
quad/direct/branch energy flow
```

但默认：

```text
T1B 不直接作为 official direction source。
```

除非另行证明它不等价于 CE-specific direction。

#### T2：clone-probe diagnostic

只能用于 upper-bound，不 promotion。

#### T3：audit target

允许使用 CE/label，只评价，不生成 direction。

### 新 target 分解

分别构造：

```text
G_cov
G_iso
G_coupling_stability
G_role_transition
G_projector_stability
G_update_half_life
G_spectrum_entropy
```

再构造 sign-stable composite：

$$
G_{LA}=\sum_j w_j\operatorname{norm}(G_j),
$$

其中 $w_j$ 只能由 train-source leaveout 选择，不能由 test/heldout outcome 选择。

### 记录指标

```text
feature_tier
feature_family
feature_name
source_column
uses_label
uses_ce_vector
uses_validation_or_test
uses_future_outcome
uses_dataset_name
precommit_available
native_logged
clone_probe_only
score_direction
auc_noise
auc_reservoir
auc_joint
precision_at_k_joint
recall_at_k_joint
bottom_precision_at_k_joint
bottom_recall_at_k_joint
soft_target_r2_noise
soft_target_r2_reservoir
leave_dataset_auc_min
leave_seed_auc_min
leave_window_auc_min
sign_stability
support_count
support_concentration
visibility_pass
failure_reason
```

### Promotion gate

T1A official value source：

$$
AUC_{joint,min} \ge 0.70,
$$

$$
precision@k_{joint,min} \ge 0.25,
$$

$$
recall@k_{joint,min} \ge 0.25,
$$

$$
softR^2_{noise,min} > 0,
$$

$$
softR^2_{reservoir,min} > 0,
$$

并且：

```text
leave-dataset / leave-seed / leave-window 不可崩；
support 不能集中在单一 seed/window；
NoOp / Random false positive = 0。
```

探索 gate：

$$
AUC_{joint,min} \ge 0.62,
$$

并且 precision/recall 至少一项 $\ge 0.15$。

### 可视化

```text
fig_v1222_visibility_auc_precision_recall.svg
fig_v1222_leaveout_visibility_heatmap.svg
fig_v1222_feature_family_ablation.svg
fig_v1222_target_sign_sanity.svg
fig_v1222_soft_target_regression_residuals.svg
fig_v1222_support_distribution.svg
```

### 不满足条件时 Codex 必须继续

```text
AUC_joint < 0.5:
  先做 sign inversion audit，再做 target convention sanity。

AUC_joint 高但 precision/recall 低:
  做 calibration / top-k threshold / support concentration audit。

soft R2 强负:
  检查 join key、normalization、constant target、NaN/Inf。

T1A 失败:
  继续 T1B diagnostic，不允许 stop。

T1B 失败:
  执行 T2 upper-bound，判断是否 target 本身不可见。

T2 也失败:
  进入 target reset，不允许直接 final stop，除非 budget_exhausted。
```

---

## P3：Line I Drift-Constrained Actuator v3

### 目标

构造安全 executor。v12.21 显示 actuator 能动，但不安全、不 release、不 control-resistant。v12.22 要把 actuator 当 executor，不当 value source。

### 候选 actuator

```text
I1 direct role scale-shift
I2 quad projection local rotation
I3 hinge amplitude/threshold
I4 absdiag energy
I5 branch gain redistribution
I6 role-balanced combined
I7 response-matrix-composed actuator
I8 random matched role-energy control
I9 shuffled actuator basis control
I10 AdamWParallel role-energy control
```

### 新机制：norm bisection

对每个 role actuator 做 norm bisection：

```text
budgets = 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02
```

先找到满足：

$$
logit\_max\_abs\_drift \le 0.05
$$

的最大 budget，再评估 release / controls。

### 新机制：response composition

给定 actuator response matrix $R$，解：

$$
\min_a
\quad
\beta \widehat{NoiseLeak}(Ra)
+
\gamma \widehat{ReservoirTrap}(Ra)
+
\eta \widehat{LogitDrift}(Ra)
-
\alpha \widehat{Coupling}(Ra)
$$

subject to:

$$
\|a\|_2 \le B,
$$

$$
logit\_drift(Ra) \le 0.05.
$$

### 记录指标

```text
actuator_id
role
basis_type
norm_budget
signed_direction
uses_label
uses_ce_vector
matched_control_id
sketch_delta_fro
projector_angle_deg
logit_max_abs_drift
CouplingR2_delta
NoiseSignalLeak_delta_audit
RealSignalReservoirRatio_delta_audit
CEp99_delta
ECE_delta
safe_movement_pass
release_audit_pass
control_gap
matched_control_gap
failure_reason
```

### Gate

Safe movement exploratory：

$$
sketch\_delta\_fro \ge 0.01,
$$

$$
projector\_angle \ge 1^\circ,
$$

$$
logit\_max\_abs\_drift \le 0.05.
$$

Release gate：

$$
NoiseSignalLeak\_delta \le -0.005,
$$

$$
RealSignalReservoirRatio\_delta \le -0.005.
$$

Official actuator gate：

$$
control\_gap \ge 0.005
$$

on at least 6/9 dataset-seed rows, with no dataset-specific branch.

### 可视化

```text
fig_v1222_actuator_movement_release_pareto.svg
fig_v1222_actuator_norm_bisection.svg
fig_v1222_role_response_heatmap.svg
fig_v1222_actuator_control_gap.svg
fig_v1222_logit_drift_vs_release.svg
```

### 不满足条件时 Codex 必须继续

```text
movement weak:
  increase norm only until drift boundary; do not jump to large norm.

movement strong but unsafe:
  response composition / role-balanced projection, not norm increase.

release absent:
  return to Line C/T target, not actuator amplitude.

controls beat actuator:
  add matched role-energy random / AdamWParallel controls; orthogonalize actuator to controls.
```

---

## P4：Line B Functional Candidate Construction

### 目标

只有 C/T/I 至少有合法 opening 时，才构造 official functional candidate。否则只允许 shadow diagnostic。

### Official construction condition

必须满足：

```text
T1A official value source pass OR T1B explicitly approved as optimizer-observable source；
AND actuator official gate pass；
AND control gap checked；
AND loss-agnostic direction audit pass。
```

### Candidate family

```text
B1-G_LA geometry stabilizer
B2-T1B update-spectrum preconditioner, diagnostic unless approved
B3-role-balanced actuator maintenance
B4-noise-leak vetoed functional event
B5-reservoir-release low-frequency event
B6-response-matrix constrained functional update
```

### Update form

$$
\theta_{t+1}=\theta_t+\Delta\theta_{task}+\lambda_t\Delta\theta_{func}.
$$

其中：

```text
Delta theta_task:
  normal manual AdamW / B320 task optimizer。

Delta theta_func:
  loss-agnostic, precommit, Line C/T/I gated。
```

### P3 row gate

$$
\Delta CouplingR^2 \ge 0.02,
$$

$$
\Delta NoiseSignalLeak \le -0.01,
$$

$$
\Delta RealSignalReservoirRatio \le -0.01,
$$

$$
control\_gap \ge 0.005,
$$

$$
logit\_max\_abs\_drift \le 0.05.
$$

### P4 short-run gate

$$
AUCtime_{func}\le AUCtime_{base},
$$

$$
ECE_{func}\le ECE_{base}+0.01,
$$

$$
CEp99_{func}\le CEp99_{base}+0.05,
$$

$$
LineC_{func}\succ LineC_{base},
$$

$$
amortized\_overhead\le 1.05.
$$

And must beat:

```text
NoOpMatchedOverhead
RandomMatchedNorm
AdamWParallelDirection
SNR-only
ShuffledPayload
Matched actuator controls
```

### Shadow diagnostic when official not open

If C/T/I does not open, Codex must still run **non-promotable shadow capacity** with explicit label:

```text
shadow_p4_capacity = 1
promotion_allowed = 0
```

This answers:

```text
If target existed, can executor affect trajectory?
```

but cannot be used as functional success.

---

## P5：Line D Classic No-BSpline Portfolio

### 目标

v12.21 不能把 generated hypotheses 当 executed. v12.22 要么实际执行一个 family repair smoke，要么标记 NotExecuted，不得写 Rejected。

Active family:

```text
Rational
Chebyshev
Wavelet
RBF/FastKAN
Fourier
```

B-spline remains frozen.

### Required per-family status

```text
FamilyPass
FamilyNearPass
KernelBlocked
ExpressionBlocked
TaskBlocked
GeometryBlocked
NotExecuted_HypothesisGenerated
RejectedForThisVersion
```

### Minimum v12.22 action

Each active family must do at least one of:

```text
1. execute one new candidate smoke;
2. execute one code-level gradient/efficiency repair;
3. explicitly mark deferred with reason and budget tradeoff.
```

It is not allowed to set `RejectedForThisVersion` solely because a hypothesis document exists.

### Records

```text
family
candidate_id
new_hypothesis_id
executed_this_version
L3_efficiency_pass
A4_expression_pass
A5_task_pass
LineC_pass
blocker
next_candidate
```

---

# 9. 必须生成的统一 artifacts

## CSV / JSON

```text
v1222_route_decision.json
v1222_exploration_queue.json
v1222_child_run_manifest.csv
v1222_fallback_execution_manifest.csv
v1222_budget_accounting.csv
v1222_final_stop_audit.json
v1222_code_review_manifest.csv
v1222_feature_provenance.csv
v1222_label_free_signal_frame.csv
v1222_label_free_failure_decomposition.csv
v1222_linec_deployable_targets.csv
v1222_visibility_scores.csv
v1222_visibility_leaveout.csv
v1222_target_sign_sanity.csv
v1222_soft_target_regression.csv
v1222_t1b_optimizer_update_features.csv
v1222_actuator_response_dictionary.csv
v1222_actuator_safety.csv
v1222_actuator_controls.csv
v1222_functional_p3.csv
v1222_functional_p4_short.csv
v1222_shadow_p4_capacity.csv
v1222_classic_family_status.csv
v1222_no_go_boundary.md
v1222_next_hypothesis_generator.md
v1222_implementation_readback.md
```

## Figures

```text
fig_v1222_queue_progress.svg
fig_v1222_linea_task_linec_pareto.svg
fig_v1222_frame_condition_vs_auc.svg
fig_v1222_role_energy_by_candidate.svg
fig_v1222_visibility_auc_precision_recall.svg
fig_v1222_leaveout_visibility_heatmap.svg
fig_v1222_target_sign_sanity.svg
fig_v1222_soft_target_regression_residuals.svg
fig_v1222_actuator_movement_release_pareto.svg
fig_v1222_actuator_norm_bisection.svg
fig_v1222_actuator_control_gap.svg
fig_v1222_p3_control_comparison.svg
fig_v1222_p4_linec_trajectory.svg
fig_v1222_classic_family_status.svg
fig_v1222_no_go_decision_tree.svg
```

---

# 10. 成功标准

## Minimum Success F：不再 fail-fast

必须满足：

```text
P0 execution contract pass；
all required artifacts written；
exploration queue executed at least one fallback for every failed open line；
final_stop_allowed = 0 unless budget_exhausted / official success；
no fake/proxy/cpu；
implementation readback complete。
```

## Success G：label-free base near-recovered

```text
best label-free candidate:
  mean_delta_vs_A0 >= -0.005
  worst_delta_vs_A0 >= -0.020
  AUC_time <= 1.05
  LineC pass rate >= 6/9
```

## Success H：legal value source exploratory survivor

```text
T1A or approved T1B:
  AUC_joint_min >= 0.62
  precision/recall >= 0.15
  soft target R2 not strongly negative
  sign stable across source-run leaveout
```

## Success I：safe actuator exploratory survivor

```text
safe_movement_rows >= 6/9
release_audit_rows >= 3/9
control_gap checked and not negative on average
```

## Success J：functional P3 open

```text
C/T + I both open;
P3 has at least one candidate beating matched controls;
P4 can legally open.
```

## Official Functional Success

```text
P4 short-run pass;
P5 3-seed confirmation pass;
no CE/label/dataset-specific direction;
beats NoOp/Random/AdamWParallel/SNR-only/MLP analog controls;
no task/ECE/tail/LineC regression;
amortized overhead <= 1.05.
```

---

# 11. 最终执行优先级

1. **先修 runner 语义**：`final_stop_allowed` 不得由 no-go + next-hypothesis 文档触发。
2. **并行启动 A/C/T/I**：label-free frame、deployable target、visibility、actuator response 同时跑。
3. **P4 不强开**：official P4 仍然 gated，但 shadow capacity 必须跑。
4. **Classic family 不假拒绝**：未执行就是 NotExecuted，不是 Rejected。
5. **每轮必须有下一层实际 child run**：没有 child run，route = `R0-NextHypothesisNotExecuted`。

---

# 12. 本轮最重要的思考结论

v12.21 不是完全没进展。它确认了：

```text
A. label-free frame 仍没替代 supervised trainprobe；
B. G_LA 有弱信号但方向/强度不足；
C. T1B native logging 没有直接打开 visibility；
D. actuator 能移动但不安全、不 release、不 control-resistant；
E. functional P3/P4 仍不该 official；
F. 经典 family 本轮没有真正执行。
```

但 v12.21 的实验设置仍不合理，因为它让 Codex 在“写出 no-go 和 next hypothesis”后停止。下一步必须改成 **探索队列驱动**。失败不能 promotion，但必须继续探索。真正的目标不是让 Codex给一个漂亮 route，而是让它在 gate 失败后继续产出机制证据，直到我们明确知道：

```text
1. 哪种 label-free signal frame 可以接近 B320-current；
2. 哪种 loss-agnostic value source 能预测 release；
3. 哪种 actuator 能安全执行；
4. functional update 是否真的有独立价值。
```

这才是 v12.22 的核心。
